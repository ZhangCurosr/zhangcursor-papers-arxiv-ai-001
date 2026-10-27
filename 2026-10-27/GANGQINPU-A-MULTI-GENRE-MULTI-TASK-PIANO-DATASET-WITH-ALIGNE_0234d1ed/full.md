# GANGQINPU: A MULTI-GENRE, MULTI-TASK PIANO DATASET WITH ALIGNED AUDIO, MIDI, AND MUSICAL SCORES

Wei Zeng<sup>1,2</sup> Anvay Mathur<sup>3</sup> Ye Wang<sup>1,2</sup>

<sup>1</sup> Integrative Sciences and Engineering Programme, NUS Graduate School

<sup>2</sup> School of Computing, National University of Singapore

<sup>3</sup> Singapore American School

w.zeng@u.nus.edu, mathur47349@sas.edu.sg, dcswangy@nus.edu.sg

## ABSTRACT

Aligned audio, performance MIDI, and musical scores provide valuable supervision for music analysis and performance modeling, yet datasets that jointly align all three modalities remain scarce and limited in scale and genre diversity. We introduce Gangqinpu, a multi-genre piano dataset containing 18,786 score–performance pairs and 1,007.1 hours of source audio, with automatic notelevel alignments and derived beat/downbeat annotations. It spans diverse genres, including pop, classical, jazz, folk/traditional, rock, and blues. Our automated pipeline transcribes piano audio, aligns the resulting MIDI with musical scores, assigns genre labels, and filters out nonexpressive performances with a score-informed binary classifier. We further construct joint splits with ASAP using conservative score-similarity grouping to reduce data leakage. We evaluate Gangqinpu on performance MIDIto-score transcription, expressive performance rendering, and beat tracking. Downstream experiments demonstrate the value of Gangqinpu as complementary training data, with cross-dataset gains in beat tracking and primarily in-domain benefits for transcription and expressive performance rendering. The code and dataset are publicly available at https://github.com/wei-zeng98/ gangqinpu.

## 1. INTRODUCTION

Piano music has long been an important research domain in music information retrieval (MIR), owing to its extensive and stylistically diverse repertoire, rich polyphonic structure, and availability in both acoustic and symbolic representations. A piano piece and its performance can be represented through three complementary modalities. Audio captures the acoustic realization, performance MIDI describes note-level performance details, and a musical score encodes the underlying compositional structure.

Figure 1 illustrates these relationships. Transcription converts audio into performance MIDI [1–7] or musical scores [8–13]. It can also recover scores from performance MIDI [14–16]. Expressive performance rendering predicts timing and dynamics from scores [16–24], while beat tracking estimates beat and downbeat times from audio or performance MIDI [25–28]. Aligned datasets with rhythmic annotations thus support multiple tasks in music analysis and performance modeling.

![](images/21c850977e0616642dd0db2e2ac25ad94fd1c8aad04d35106bfc2f87becc1195.jpg)  
Figure 1. Musical representations and associated tasks. Arrows indicate task input–output relationships; colors distinguish transcription (red), rendering (blue), and beat tracking (green).

Despite the importance of data connecting these modalities, existing piano datasets typically provide only a subset of audio, performance MIDI, and musical scores. Datasets that jointly align all three modalities remain scarce. ASAP [29] is a notable example, but it provides audio for only 519 of its 1,067 MIDI performances, with a small number of performances not fully aligned to their corresponding scores. Moreover, its repertoire is restricted to Western classical piano music. The broader predominance of classical repertoire in existing datasets limits their coverage of musical styles and applications.

Beyond classical music, pop and jazz performances may involve lead sheets, informal arrangements, or improvisation rather than fully notated scores, making transcription from recordings particularly useful. Expressive performance rendering also has potential applications in contemporary music production by transforming quantized MIDI into natural-sounding performances within digital audio workstations (DAWs). However, research on expressive performance rendering has focused predominantly on classical piano [30]. These applications motivate larger aligned piano datasets spanning a broader range of genres.

To address this gap, we introduce Gangqinpu, a multigenre piano dataset with aligned audio, transcribed performance MIDI, and musical scores. We construct the dataset from online audio–score pairs using an automated pipeline for audio transcription, expressive performance filtering, genre annotation, and note-level score alignment. A scoreinformed binary classifier filters out non-expressive performances with limited timing and dynamic variation, including renderings synthesized directly from scores. The dataset spans diverse genres, including pop, classical, jazz, folk/traditional, rock, and blues, with additional samples labeled as other or unknown.

The main contributions of this work are as follows:

• We introduce Gangqinpu, a multi-genre piano dataset with 18,786 score–performance pairs, 1,007.1 hours of source audio, and automatic notelevel and beat/downbeat annotations.

• We develop an automated construction pipeline for transcription, alignment, genre annotation, and expressive performance filtering, with joint score grouping across Gangqinpu and ASAP to reduce data leakage.

• We demonstrate Gangqinpu’s value as complementary training data for beat tracking, performance MIDI-to-score transcription, and expressive performance rendering. Gains extend across datasets for beat tracking and are primarily in-domain for transcription and rendering.

## 2. RELATED WORK

## 2.1 Piano Datasets

Piano performance MIDI datasets are typically constructed through direct capture or automatic music transcription (AMT). MAESTRO [31] provides synchronized audio and MIDI captured using Yamaha Disklavier pianos, but this approach requires specialized recording equipment. AMT enables more scalable collection from existing recordings: ATEPP [32] provides transcribed performances by professional pianists, while Aria-MIDI [33] scales up to over one million MIDI files from online recordings, both subject to transcription errors.

For tasks that relate performances to musical notation, performance MIDI must be paired with corresponding scores. ASAP [29] provides beat-level alignments, and (n)ASAP [34] extends these to the note level through automatic alignment and manual inspection. ACPAS [35] combines real and synthesized audio with MIDI scores and beat-level annotations. ATEPP provides corresponding MusicXML scores for a subset of its recordings, while PianoCoRe [36] consolidates existing corpora into a large note-aligned MIDI collection.

Table 1 summarizes these datasets in terms of scale, modalities, and annotations. Existing score-aligned resources predominantly cover Western classical repertoire, where performances typically follow established scores. Extending such resources to popular music is challenging because recordings may be produced without a written score [30]. POP909 [37] supports popular-music research through MIDI arrangements with beat, chord, and key annotations, but does not provide piano recordings paired with independently notated scores. Gangqinpu addresses this gap by connecting multi-genre piano recordings with musical scores through transcribed performance MIDI and automatic note-level alignments.

## 2.2 Audio-to-MIDI Piano Transcription and Score–Performance Alignment

Recent piano transcription systems include autoregressive encoder–decoder models such as Aria-AMT [38] and interval-based models such as Transkun v2 [2]. Aria-AMT predicts note and pedal tokens, whereas Transkun v2 combines Transformer-based interval scoring with nonautoregressive semi-CRF decoding. We select Transkun v2 for its structured prediction of note and pedal intervals and its ability to preserve note offsets without pedal-based extension, allowing note offsets and sustain-pedal activity to be represented separately.

For symbolic alignment, Nakamura et al. [39] combine HMM-based matching with error detection and local realignment using merged-output HMMs. DualDTWNoteMatcher [40], implemented in Parangonar, instead separates pitch-sequence matching from pitch-wise onset alignment. Across four evaluated piano datasets, it achieved a higher combined note-matching F1 than the Nakamura baseline (99.0% versus 98.5%). These reported results and its integration with our symbolic processing and refinement pipeline motivate our choice of DualDTWNoteMatcher.

## 3. DATASET CONSTRUCTION

## 3.1 Overall Pipeline

Figure 2 summarizes the construction of Gangqinpu from online audio–score pairs. After filtering by audio duration, score length, and instrumentation, we transcribe audio into performance MIDI using Transkun v2 [2], clean the resulting MIDI, and convert MusicXML scores into score MIDI. DualDTWNoteMatcher [40] establishes note-level correspondences between the two representations, followed by alignment and MIDI refinement using RAScoP [36]. We then derive beat/downbeat annotations and metadata, retaining samples that satisfy raw alignment error thresholds and are classified as expressive performances (see Section 3.4). Accepted samples receive genre labels through Qwen-Flash [41], with LAION-CLAP [42] as a fallback. Finally, score-similarity grouping and duplicateaudio checks guide joint data splitting with ASAP to reduce leakage across partitions and datasets.

## 3.2 Data Acquisition and Preprocessing

We collect publicly accessible piano audio and associated sheet music from online sources, together with metadata such as titles and contributor information. We retain recordings lasting at least 60 seconds (Filter 1) and scores containing at least 32 measures with piano-only parts, excluding short excerpts and scores for other instrumentation (Filter 2).

Scores are converted into MusicXML and processed using the SyMuPe and Partitura toolkits [43, 44]. Repeats and alternative endings are unfolded, and a score MIDI representation is generated while retaining links to score positions and metrical information. Before alignment, per-

<table><tr><td></td><td></td><td colspan="3">Size</td><td colspan="3">Modalities</td><td>Origin</td><td colspan="2">Annotations</td></tr><tr><td>Dataset</td><td>Genre</td><td>Pieces / Scores</td><td>Perf.</td><td>Hours</td><td>Audio</td><td>MIDI</td><td>Score</td><td>Audio / MIDI Source</td><td>Align.</td><td>Beats</td></tr><tr><td>MAESTRO v3 [31]</td><td>Mostly classical</td><td>一</td><td>1,276</td><td>198.7</td><td>V</td><td>√</td><td>一</td><td>Real / Captured</td><td></td><td></td></tr><tr><td>ASAP/(n)ASAP [29,34]</td><td>Classical</td><td>235</td><td>1,067</td><td>94.5</td><td>External</td><td>√</td><td>√</td><td>Real / Captured</td><td>Beat / Note†</td><td>√</td></tr><tr><td>ATEPP v1.2 [32]</td><td>Classical</td><td>1,595</td><td>11,674</td><td>~1,000</td><td>External</td><td>√</td><td>△*</td><td>Real (web) / AMT</td><td></td><td></td></tr><tr><td>ACPAS [35]</td><td>Classical</td><td>497</td><td>2,189</td><td>179.8</td><td>Ext./Synth.</td><td>√</td><td>√</td><td>Real+Synth. / Mixed</td><td>Beatβ</td><td>√</td></tr><tr><td>PianoCoRe-A [36]</td><td>Primarily classical</td><td>1,591</td><td>157,207</td><td>12,509</td><td></td><td>√</td><td>√</td><td>None / Mixed</td><td>Note</td><td></td></tr><tr><td>POP909 [37]</td><td>Pop</td><td>909</td><td></td><td>~60</td><td></td><td>√</td><td>一</td><td>Reference / Arrangement</td><td></td><td>√</td></tr><tr><td>Gangqinpu</td><td>Multiple</td><td>15,645</td><td>18,786</td><td>1,007.1</td><td>External</td><td>√</td><td>√</td><td>Web / AMT</td><td>Note</td><td>√</td></tr></table>

✓ indicates availability, △ partial coverage, and – unavailable or unreported information. External (Ext.) audio is obtained separately; Gangqinpu provides audio URLs and a download script. Align. denotes score–performance alignment. Captured denotes instrument-recorded MIDI, AMT automatic transcription, and Web online audio of unverified recording/synthesis origin. <sup>†</sup> Statistics follow (n)ASAP [34]; 1,062 performances have note alignments. <sup>∗</sup> MusicXML covers roughly half the tracks. <sup>§</sup> 30 performances lack alignment.

Table 1. Comparison of piano datasets by scale, modalities, data origin, and score–performance annotations.

formance tracks are merged into a single note sequence, duplicate and implausibly short notes are removed, and overlapping notes of the same pitch are truncated. The resulting representations retain score structure and expressive performance information for note-level alignment and annotation.

## 3.3 Alignment-Derived Beat Annotation and Confidence

We obtain initial note correspondences using Parangonar’s DualDTWNoteMatcher [40] and refine them with RAScoP [36]. Interpolated performance notes introduced during refinement are explicitly marked to distinguish them from observed transcription events. To assess score– performance consistency, we compute insertion and deletion rates from the raw alignment, normalized by the numbers of performance and score notes, respectively. Pairs are excluded if either rate exceeds 10% (Filter 3), preventing subsequent interpolation from masking substantial mismatches.

A metrical grid is extracted from the unfolded score, preserving time-signature changes and using dottedquarter-note beats in compound meters such as 6/8, 9/8, and 12/8. Score beats are transferred to performance time through the refined note correspondences. Inspired by the ASAP annotation procedure [29], we identify notes within ±0.05 quarter notes of each score beat and use the median of their corresponding performance onset times as the beat time. Beats without direct support are linearly interpolated between neighboring onset anchors or extrapolated at the boundaries using a local tempo estimate. Downbeats and time/key-signature changes are mapped to the same performance timeline.

We assign heuristic confidence scores based on the evidence supporting each beat. A beat initially receives a confidence of 0.95 if any nearby mapped note is observed, or 0.40 if all are interpolated. When beat times are interpolated between neighboring onset anchors, confidence is 0.75 times the lower anchor confidence; extrapolated beats receive 0.25. If any score note within ±0.05 quarter notes of the beat is marked as interpolated, confidence is multiplied by 0.35. We mark beats with confidence below 0.50 as low confidence and those below 0.35 as ambiguous.

These scores reflect heuristic annotation reliability rather than calibrated probabilities and support masking uncertain regions in downstream tasks.

## 3.4 Expressive Performance Filtering

Online audio–score collections often include deadpan renderings synthesized directly from notation, which sound mechanical due to rigid timing and little or no expressive variation in dynamics. Although useful for some transcription tasks, such examples provide limited supervision for expressive performance rendering, where the objective is to learn musically meaningful deviations from the score. We therefore distinguish expressive from non-expressive performances based on perceived timing and dynamics, regardless of production origin. Human recordings and digitally edited MIDI renderings are both eligible if they exhibit sufficient expressiveness.

To automate this selection, we train a binary classifier that takes a performance MIDI and its corresponding score as input (Filter 4). We construct a labeled subset of Gangqinpu through independent listening assessments by two human experts. Each expert assigns an expressive or non-expressive label according to the audible performance. Disagreements are discussed and reassessed through repeated listening until consensus is reached. The resulting subset contains 779 expressive and 370 non-expressive examples.

The classifier is a multilayer perceptron (MLP) operating on eight handcrafted features summarized in Table 2. Features are extracted from observed note correspondences in the raw alignment, avoiding interpolated events introduced during refinement. To measure timing variation, we first group notes with the same score onset and represent each group by its median performed onset. At each onset position, we fit a local linear mapping from score time to performance time within a window of ±4 quarter notes, widening it when necessary to include at least six distinct onset positions, where available. Differences between observed and predicted onset times yield three features: timing variability, correlation, and entropy. Separate features measure onset differences within chords and variation in note velocities.

The standardized feature vector is processed by two hidden layers with 32 and 16 units and ReLU activations, followed by a binary probabilistic output. Training uses Adam, L<sub>2</sub> regularization with $\alpha = 1 0 ^ { - 3 }$ , and early stopping. The labeled subset is supplemented with 982 usable ASAP performances as positive examples and synthetically perturbed score renderings as negative examples. Each source is partitioned into training and held-out subsets at a 75:25 ratio, with synthetic variants grouped by their underlying score; class imbalance is addressed by oversampling within the training set. Samples with a predicted expressive-class probability of at least 0.5 are retained.

![](images/6efc205f8fde58467eef83e2e42dcfac651f0373e834a3868e2a1fd136505a06.jpg)  
Dotted horizontal links indicate paired representations. Filters retain samples satisfying: (1) audio duration ≥ 60 s; (2) at least 32 measures and piano-only parts; (3) insertion and deletion rates each $\leq 1 0 \%$ measured from the raw alignment before refinement; (4) predicted expressiveperformance probability ≥ 0.5.  
Figure 2. Overview of the Gangqinpu dataset construction pipeline.

## 3.5 Genre Annotation and Leakage-Aware Splitting

We assign sample-level genre labels using a two-stage procedure. Qwen-Flash [41] first classifies each sample from its textual metadata, including the title and contributor information. Samples whose genre remains undetermined are subsequently assessed using LAION-CLAP [42] with the music checkpoint, which compares audio embeddings with textual genre descriptions. Predictions that do not meet the confidence criteria remain labeled as unknown. Labels are normalized into six named genres (pop, classical, jazz, folk/traditional, rock, and blues), along with the additional categories of other and unknown.

<table><tr><td>Feature</td><td>Definition</td></tr><tr><td>Timing variability</td><td>Standard deviation of local onset residuals.</td></tr><tr><td>Timing correlation</td><td>Lag-one autocorrelation of onset residuals.</td></tr><tr><td>Timing entropy</td><td>Shannon entropy of onset residuals dis- cretized into 10 ms bins.</td></tr><tr><td>Chord asynchrony</td><td>Median onset range within groups of simul- taneously notated notes.</td></tr><tr><td>Velocity variability</td><td>Standard deviation of note velocities.</td></tr><tr><td>Velocity diversity</td><td>Number of distinct note velocities.</td></tr><tr><td>Staff contrast</td><td>Mean velocity of upper-staff notes minus that of notes on other staves.</td></tr><tr><td>Pitch contrast</td><td>Mean within-chord velocity difference be- tween the highest note and the remaining notes.</td></tr></table>

Table 2. Score-informed timing and dynamic features used for expressive performance classification.

To reduce leakage, we identify related scores before assigning samples to training, validation, and test sets. Candidate pairs are retrieved using transposition-invariant pitch-interval patterns from the upper and lower voices, then verified through monotonic alignment and polyphonic note matching. We maintain two grouping levels in the metadata. Score-similarity groups (score\_group\_id) are used to report the number of score groups in the dataset. Broader split groups (group\_id) additionally link potentially related arrangements and excerpts, including matches between Gangqinpu and ASAP, and are used only for partition assignment. Each split group is assigned entirely to one partition, retaining multiple performances of the same score together. Exact audio duplicates are also checked across partitions.

We jointly group Gangqinpu and (n)ASAP v2.1.1 [29, 34] to account for cross-dataset overlap. Groups containing the ASAP test pieces used by [15] are assigned to the test set. The remaining groups are assigned to training, validation, or test, with each group kept in a single partition. The resulting train/validation/test splits contain 17,773/499/514 Gangqinpu samples and 955/50/61 (n)ASAP samples. All downstream tasks in Section 5 use these fixed assignments, selecting samples with the required inputs and annotations. Evaluation is reported separately on the two datasets.

## 4. DATASET STATISTICS

Gangqinpu contains 18,786 paired performances totaling 1,007.1 hours of source audio, organized into 15,645 automatically identified score groups. These groups capture identical or highly similar scores, including transposed variants, and serve as an operational measure of score diversity rather than a verified count of distinct compositions. Among them, 2,445 groups contain multiple performances, providing alternative realizations of similar musical material. Table 3 summarizes the dataset scale, and Figure 3 presents its principal distributions.

Figure 3 summarizes the genre, duration, and notecount distributions. Pop and classical dominate the automatically assigned genre labels, alongside folk/traditional, jazz, rock, and blues; uncertain assignments remain unknown. Audio duration and note counts exhibit long-tailed distributions, reflecting variation in recording length and sequence size. Note counts are computed from refined performance MIDI and include interpolated notes.

<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Performances</td><td>18,786</td></tr><tr><td>Score groups</td><td>15,645</td></tr><tr><td>Groups with multiple performances</td><td>2,445</td></tr><tr><td>Total audio duration (hours)</td><td>1,007.1</td></tr><tr><td>Mean audio duration (min) Median audio duration (min)</td><td>3.22</td></tr><tr><td></td><td>3.04</td></tr><tr><td>Median notes per performance</td><td>956</td></tr></table>

Table 3. Summary statistics of Gangqinpu. Score groups are identified automatically through score-similarity analysis.

## 5. EXPERIMENTS

## 5.1 Expressive Performance Classification

We evaluate whether the proposed classifier distinguishes expressive from non-expressive performances according to expert listening judgments, irrespective of recording or production provenance, using four subsets: ASAP, real human performances; expressive and non-expressive, the two classes of our own expert-labeled subset; and hard\_neg, a synthetic negative class rendered directly from scores with small Gaussian onset/velocity noise $( \sigma _ { \mathrm { o n s e t } } \leq 3 0 \mathrm { m s } ,$ $\sigma _ { \mathrm { v e l o c i t y } } \leq 9 )$ . After alignment and feature-validity checks, the expert-labeled subset contains 779 expressive and 370 non-expressive samples; a fixed 75/25 split within each class yields 195 expressive and 93 non-expressive held-out samples, with ASAP and hard\_neg likewise split 75/25 to supplement training and evaluation.

The eight-feature MLP achieves 98.61% accuracy and 98.41% macro-F1 on the 288 expert-labeled held-out samples (Table 4). It correctly identifies 193 of 195 expressive performances and 91 of 93 non-expressive performances, with two errors in each class. These results support its use as an automatic filtering component for the collected data, although they do not establish generalization to other transcription pipelines or data sources.

Comparison with existing classifiers. We compare against two published, readily available classifiers addressing related tasks: the SyMuPe quality classifier [36, 43], a four-way MIDI quality model (score, high quality, low quality, corrupted), mapped to binary here by treating high quality as expressive and the remaining three categories as non-expressive; and NOMML, the median-metric-level measure from GigaMIDI [45], applied with its reported threshold (median level = 12). Evaluation uses the same 534 held-out samples as Table 4, plus 480 synthetic samples (120 each) from the hard\_neg and 46/60/80 ms noise conditions.

Table 4 shows that our classifier achieves the highest pooled accuracy and macro-F1 on the real-data evaluation subset, while its detection rate decreases as synthetic onset noise increases. SyMuPe recalls expressive performances well but labels 76 of 93 non-expressive samples as high quality, reflecting its design for general quality triage rather than score-synchronized expressiveness detection. NOMML assigns a metric level of 12 to every sample and therefore predicts all samples as expressive. Its pooled accuracy of 82.58% reflects the class distribution (441 expressive and 93 non-expressive), while its macro-F1 is 45.23%. These comparisons assess the classifiers’ suitability for our filtering task rather than their general performance.

<table><tr><td>Subset</td><td>n</td><td>Ours</td><td>SyMuPe</td><td>NOMML</td></tr><tr><td>ASAP (human)</td><td>246</td><td>100.0%</td><td>99.6%</td><td>100.0%</td></tr><tr><td>expressive</td><td>195</td><td>99.0%</td><td>86.7%</td><td>100.0%</td></tr><tr><td>non-expressive</td><td>93</td><td>97.8%</td><td>18.3%</td><td>0.0%</td></tr><tr><td>hard_neg (synthetic)</td><td>120</td><td>100.0%</td><td>100.0%</td><td>0.0%</td></tr><tr><td>noise 46 ms (synthetic)</td><td>120</td><td>98.3%</td><td>100.0%</td><td>0.0%</td></tr><tr><td>noise 60 ms (synthetic)</td><td>120</td><td>83.3%</td><td>100.0%</td><td>0.0%</td></tr><tr><td>noise 80 ms (synthetic)</td><td>120</td><td>30.8%</td><td>100.0%</td><td>0.0%</td></tr><tr><td>Pooled accuracy (n=534)</td><td></td><td>99.25%</td><td>80.71%</td><td>82.58%</td></tr><tr><td>Pooled macro-F1</td><td></td><td>98.70%</td><td>56.88%</td><td>45.23%</td></tr></table>

Table 4. Comparison with SyMuPe and NOMML. Persubset values are detection rates for that subset’s single ground-truth class; pooled rows combine the three realdata splits (n=534), the only ones mixing both classes.

## 5.2 Downstream Task Evaluation

We evaluate the utility of Gangqinpu on three complementary tasks: beat tracking, performance MIDI-to-score transcription, and expressive performance rendering. We use Beat This! [26] for estimating beat and downbeat positions from audio, PM2S [14] for recovering scorerelated symbolic attributes from performance MIDI, and an adapted S2A [24] model for predicting expressive timing and velocity from a musical score. These tasks assess the usefulness of the dataset’s rhythmic annotations, score–performance correspondences, and expressive performance representations, respectively.

For each baseline, we compare training on ASAP alone with training on ASAP combined with Gangqinpu and evaluate both conditions separately on the ASAP and Gangqinpu test sets. All tasks follow the fixed joint split assignments, with usable subsets determined by the required inputs and annotations. The ASAP test subsets contain 26 samples for Beat This!, 59 for PM2S hand assignment, and 61 for the remaining PM2S tasks and S2A. All tasks use 514 Gangqinpu test samples.

## 5.2.1 Beat Tracking

We evaluate Beat This! [26] on beat and downbeat tracking using F1, CMLt, and AMLt [46]. Low-confidence and temporally inconsistent annotation regions, together with regions outside the annotated span, are masked during training and evaluation. Both predicted and reference events within masked regions are excluded from metric computation. Table 5 compares the two training conditions on both test sets. Adding Gangqinpu improves all reported metrics on both datasets. On ASAP, beat and downbeat F1 increase by 4.30 and 4.31 percentage points, respectively. The gains are larger on Gangqinpu, where beat F1 improves from 60.18 to 78.16 and downbeat F1 from 39.60 to 63.66. Higher CMLt and AMLt scores indicate more reliable beat-sequence tracking. These results suggest that Gangqinpu provides useful rhythmic supervision, with benefits extending to the ASAP test set.

![](images/e82e3487cb95624b788b942ee3951dd358fe533493359a41883e4edf4e1cfb25.jpg)

![](images/93bfdb37200acd833532336bc12c1285a735c482a863dcf9f41d37c428e96a71.jpg)

![](images/8a4757f1270ecf4ef23f88f1cb01a82793822bee44d93f0a302dbf429913c70b.jpg)

![](images/e21c11101be8cd70df7db3c971c3538684393d27392111f727f577875d4dfdb3.jpg)  
Figure 3. Distributions of (a) automatically assigned genre labels, (b) audio duration, (c) refined performance MIDI note counts, including interpolated notes, and (d) performances per score group. Panels (b) and (c) use logarithmic horizontal axes; panel (d) uses a logarithmic vertical axis. Dashed lines indicate medians.

<table><tr><td>Target</td><td>Training data</td><td>F1↑</td><td>CMLt ↑ AMLt ↑</td><td></td></tr><tr><td colspan="5">ASAP test</td></tr><tr><td>Beat</td><td>ASAP</td><td>57.69</td><td>21.66</td><td>26.63</td></tr><tr><td>Downbeat</td><td>ASAP + Gangqinpu ASAP</td><td>61.98 43.23</td><td>28.57 10.90</td><td>30.64</td></tr><tr><td></td><td>ASAP + Gangqinpu</td><td>47.55</td><td>13.41</td><td>21.30 22.79</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="5">Gangqinpu test</td></tr><tr><td>Beat</td><td>ASAP</td><td>60.18</td><td>28.64</td><td>35.09</td></tr><tr><td></td><td>ASAP + Gangqinpu</td><td>78.16</td><td>58.42</td><td>63.63</td></tr><tr><td>Downbeat</td><td>ASAP</td><td>39.60</td><td>9.26</td><td>19.91</td></tr><tr><td></td><td>ASAP + Gangqinpu</td><td>63.66</td><td>30.51</td><td>51.63</td></tr></table>

Table 5. Beat This! results under two training conditions. All metrics are expressed as percentages.

## 5.2.2 Performance MIDI-to-Score Transcription

We evaluate PM2S [14] on the prediction of score-related attributes from performance MIDI, including beat and downbeat labels, hand assignment, key and time signatures, and rhythmic quantisation. We report positive-class F1 for beat and downbeat labels, macro-F1 for hand assignment and key/time signatures, and support-weighted F1 for quantised onsets and note values. Here, beat and downbeat scores measure note-level label prediction rather than the audio-based event detection evaluated in the preceding task.

Table 6 shows that adding Gangqinpu improves most evaluated attributes on the Gangqinpu test set. Beat and downbeat F1 increase by 14.02 and 20.95 percentage points, respectively, while onset and note-value weighted F1 improve by 18.87 and 19.32 points. Hand assignment also improves, whereas time-signature macro-F1 decreases. Transfer to ASAP is mixed: beat, downbeat, and note-value prediction improve, while hand assignment, key/time signatures, and onset quantisation decline. These results demonstrate benefits for transcription within the Gangqinpu domain, but do not indicate uniform improvements across attributes or datasets.

## 5.2.3 Expressive Performance Rendering

We evaluate the adapted S2A [24] model to predict expressive performance attributes from musical scores. We report mean absolute error (MAE) for MIDI velocity, interonset interval (IOI), and note duration, with timing errors expressed in milliseconds. These metrics measure agreement with the reference performance rather than perceptual expressiveness.

As shown in Table 7, adding Gangqinpu reduces all three errors on the Gangqinpu test set. Velocity MAE decreases from 17.50 to 13.05, IOI MAE from 78.51 to 54.66 ms, and duration MAE from 309.47 to 269.04 ms. On ASAP, all three errors increase modestly, indicating that the benefits do not transfer uniformly across domains. Together, these results support the utility of Gangqinpu for modeling performances within its domain, while highlighting the challenge of improving performance across different musical and data distributions.

## 6. CONCLUSION

We introduced Gangqinpu, a multi-genre piano dataset comprising 18,786 paired performances and 1,007.1 hours of source audio, with transcribed performance MIDI and musical scores connected by automatic note-level alignments. Our construction pipeline combines expressive performance filtering, confidence-aware beat annotation, genre labeling, and leakage-aware splitting to support multiple downstream tasks. Experiments demonstrate improved beat tracking on both Gangqinpu and ASAP, alongside gains in most performance MIDI-to-score transcription targets and all evaluated rendering metrics on Gangqinpu. Transfer to ASAP remains mixed for the latter two tasks, highlighting the challenges of cross-domain generalization. Future work will focus on manually auditing automatic annotations.

<table><tr><td colspan="2"></td><td colspan="2">ASAP test</td><td colspan="2">Gangqinpu test</td></tr><tr><td>Prediction target</td><td>Metric ↑</td><td>ASAP</td><td> $\mathbf { A S A P + G }$ </td><td>ASAP</td><td>ASAP + G</td></tr><tr><td>Beat label</td><td>F1</td><td>62.97</td><td>64.89</td><td>69.28</td><td>83.30</td></tr><tr><td>Downbeat label</td><td>F1</td><td>43.36</td><td>48.68</td><td>42.02</td><td>62.97</td></tr><tr><td>Hand assignment</td><td>Macro-F1</td><td>86.65</td><td>84.81</td><td>86.99</td><td>93.99</td></tr><tr><td>Key signature</td><td>Macro-F1</td><td>36.26</td><td>30.87</td><td>78.39</td><td>79.33</td></tr><tr><td>Time signature</td><td>Macro-F1</td><td>45.40</td><td>43.96</td><td>51.00</td><td>45.43</td></tr><tr><td>Quantised onset</td><td>Weighted-F1</td><td>53.59</td><td>49.71</td><td>55.67</td><td>74.54</td></tr><tr><td>Note value</td><td>Weighted-F1</td><td>40.39</td><td>41.37</td><td>34.86</td><td>54.18</td></tr></table>

Table 6. PM2S results under ASAP-only and combined training (ASAP + G, where G denotes Gangqinpu). All scores are percentages. Bold indicates the better training condition for each target within each test set.

<table><tr><td>Training data</td><td></td><td></td><td>Velocity ↓ IOI (ms) ↓ Duration (ms) ↓</td></tr><tr><td></td><td colspan="3">ASAP test</td></tr><tr><td>ASAP</td><td>16.40</td><td>66.40</td><td>999.66</td></tr><tr><td>ASAP + Gangqinpu</td><td>16.62</td><td>68.96</td><td>1,058.28</td></tr><tr><td></td><td colspan="3">Gangqinpu test</td></tr><tr><td>ASAP</td><td>17.50</td><td>78.51</td><td>309.47</td></tr><tr><td> $\mathrm { A S A P + G a n g q i n p u }$ </td><td>13.05</td><td>54.66</td><td>269.04</td></tr></table>

Table 7. Expressive performance rendering results using the adapted S2A model. All metrics are MAE. Velocity errors are in MIDI velocity units.

## 7. REFERENCES

[1] Y. Yan, F. Cwitkowitz, and Z. Duan, “Skipping the frame-level: Event-based piano transcription with neural semi-crfs,” in Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, M. Ranzato, A. Beygelzimer, Y. N. Dauphin, P. Liang, and J. W. Vaughan, Eds., 2021, pp. 20 583–20 595.

[2] Y. Yan and Z. Duan, “Scoring time intervals using non-hierarchical transformer for automatic piano transcription,” in Proceedings of the 25th International Society for Music Information Retrieval Conference, ISMIR 2024, San Francisco, California, USA and Online, November 10-14, 2024, B. Kaneshiro, G. J. Mysore, O. Nieto, C. Donahue, C. A. Huang, J. H. Lee, B. McFee, and M. C. McCallum, Eds., 2024, pp. 973– 980.

[3] Q. Kong, B. Li, X. Song, Y. Wan, and Y. Wang, “Highresolution piano transcription with pedals by regressing onset and offset times,” IEEE ACM Trans. Audio Speech Lang. Process., vol. 29, pp. 3707–3717, 2021.

[4] C. Hawthorne, I. Simon, R. Swavely, E. Manilow, and J. H. Engel, “Sequence-to-sequence piano transcription with transformers,” in Proceedings of the 22nd International Societyfor Music Information Retrieval Conference, ISMIR 2021, Online, November 7-12, 2021, J. H. Lee, A. Lerch, Z. Duan, J. Nam, P. Rao, P. van Kranenburg, and A. Srinivasamurthy, Eds., 2021, pp. 246–253.

[5] K. Toyama, T. Akama, Y. Ikemiya, Y. Takida, W. Liao, and Y. Mitsufuji, “Automatic piano transcription with hierarchical frequency-time transformer,” in Proceedings of the 24th International Society for Music Information Retrieval Conference, ISMIR 2023, Milan, Italy, November 5-9, 2023, A. Sarti, F. Antonacci, M. Sandler, P. Bestagini, S. Dixon, B. Liang, G. Richard, and J. Pauwels, Eds., 2023, pp. 215–222.

[6] C. Hawthorne, E. Elsen, J. Song, A. Roberts, I. Simon, C. Raffel, J. H. Engel, S. Oore, and D. Eck, “Onsets and frames: Dual-objective piano transcription,” in Proceedings of the 19th International Society for Music Information Retrieval Conference, ISMIR 2018, Paris, France, September 23-27, 2018, E. Gómez, X. Hu, E. Humphrey, and E. Benetos, Eds., 2018, pp. 50–57.

[7] J. Jung, D. Kim, S. Lee, S. Cho, H. Soh, I. Bukey, C. Donahue, and D. Jeong, “U-must: A unified framework for cross-modal translation of score images, symbolic music, and performance audio,” IEEE Transactions on Audio, Speech and Language Processing, vol. 34, pp. 1876–1891, 2026.

[8] W. Zeng, X. He, and Y. Wang, “End-to-end real-world polyphonic piano audio-to-score transcription with hierarchical decoding,” in Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI 2024, Jeju, South Korea, August 3-9, 2024. ijcai.org, 2024, pp. 7788–7795.

[9] R. G. C. Carvalho and P. Smaragdis, “Towards endto-end polyphonic music transcription: Transforming music audio directly to a score,” in 2017 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics, WASPAA 2017, New Paltz, NY, USA, October 15-18, 2017. IEEE, 2017, pp. 151–155.

[10] M. A. Román, A. Pertusa, and J. Calvo-Zaragoza, “A holistic approach to polyphonic music transcription with neural networks,” in Proceedings of the 20th International Society for Music Information Retrieval Conference, ISMIR 2019, Delft, The Netherlands, November 4-8, 2019, A. Flexer, G. Peeters, J. Urbano, and A. Volk, Eds., 2019, pp. 731–737.

[11] N. C. Tamer, V. Ebert, G. Yang, and N. A. Smith, “Rubato: Transcribing piano music with timestamps,” arXiv preprint arXiv:2605.24291, 2026.

[12] J. Hu, Y. Wang, S. Wu, Z. Guo, S. Liang, W. Meng, C. Yang, X. Li, F. Yu, and M. Sun, “TUTTI: toward generalizable audio-to-score transcription via fully synthesized data,” in Proceedings of the International Society for Music Information Retrieval Conference (ISMIR), 2026.

[13] M.-H. Wang and T.-R. Huang, “A learned musicaltime coordinate for piano audio-to-score transcription,” arXiv preprint arXiv:2610.29928, 2026.

[14] L. Liu, Q. Kong, V. Morfi, and E. Benetos, “Performance midi-to-score conversion by neural beat tracking,” in Proceedings of the 23rd International Societyfor Music Information Retrieval Conference, ISMIR 2022, Bengaluru, India, December 4-8, 2022, P. Rao, H. A. Murthy, A. Srinivasamurthy, R. M. Bittner, R. C. Repetto, M. Goto, X. Serra, and M. Miron, Eds., 2022, pp. 395–402.

[15] T. Beyer and A. Dai, “End-to-end piano performancemidi to score conversion with transformers,” in Proceedings of the 25th International Society for Music Information Retrieval Conference, ISMIR 2024, San Francisco, California, USA and Online, November 10-14, 2024, B. Kaneshiro, G. J. Mysore, O. Nieto, C. Donahue, C. A. Huang, J. H. Lee, B. McFee, and M. C. McCallum, Eds., 2024, pp. 319–326.

[16] W. Zeng, J. Zhao, and Y. Wang, “Bridging piano transcription and rendering via disentangled score content and style,” in The Fourteenth International Conference on Learning Representations, ICLR 2026, Rio de Janeiro, Brazil, April 23-27, 2026, C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust, Eds. proceedings.iclr.cc, 2026.

[17] A. Maezawa, K. Yamamoto, and T. Fujishima, “Rendering music performance with interpretation variations using conditional variational RNN,” in Proceedings of the 20th International Society for Music Information Retrieval Conference, ISMIR 2019, Delft, The Netherlands, November 4-8, 2019, A. Flexer, G. Peeters, J. Urbano, and A. Volk, Eds., 2019, pp. 855–861.

[18] D. Jeong, T. Kwon, Y. Kim, K. Lee, and J. Nam, “Virtuosonet: A hierarchical rnn-based system for modeling expressive piano performance,” in Proceedings of the 20th International Society for Music Information Retrieval Conference, ISMIR 2019, Delft, The Netherlands, November 4-8, 2019, A. Flexer, G. Peeters, J. Urbano, and A. Volk, Eds., 2019, pp. 908–915.

[19] S. Rhyu, S. Kim, and K. Lee, “Sketching the expression: Flexible rendering of expressive piano performance with self-supervised learning,” in Proceedings of the 23rd International Society for Music Information

Retrieval Conference, ISMIR 2022, Bengaluru, India, December 4-8, 2022, P. Rao, H. A. Murthy, A. Srinivasamurthy, R. M. Bittner, R. C. Repetto, M. Goto, X. Serra, and M. Miron, Eds., 2022, pp. 178–185.

[20] I. Borovik and V. Viro, “Scoreperformer: Expressive piano performance rendering with fine-grained control,” in Proceedings of the 24th International Society for Music Information Retrieval Conference, IS-MIR 2023, Milan, Italy, November 5-9, 2023, A. Sarti, F. Antonacci, M. Sandler, P. Bestagini, S. Dixon, B. Liang, G. Richard, and J. Pauwels, Eds., 2023, pp. 588–596.

[21] L. Renault, R. Mignot, and A. Roebel, “Expressive piano performance rendering from unpaired data,” in International Conference on Digital Audio Effects (DAFx23), 2023, pp. 355–358.

[22] J. Tang, G. Wiggins, and G. Fazekas, “Reconstructing human expressiveness in piano performances with a transformer network,” arXiv preprint arXiv:2306.06040, 2023.

[23] H. Zhang, S. Chowdhury, C. E. Cancino-Chacón, J. Liang, S. Dixon, and G. Widmer, “Dexter: Learning and controlling performance expression with diffusion models,” Applied Sciences, vol. 14, no. 15, p. 6543, 2024.

[24] J. Tang, E. Cooper, X. Wang, J. Yamagishi, and G. Fazekas, “Towards an integrated approach for expressive piano performance synthesis from music scores,” in 2025 IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2025, Hyderabad, India, April 6-11, 2025. IEEE, 2025, pp. 1–5. [Online]. Available: https://doi.org/10. 1109/ICASSP49660.2025.10890623

[25] J. Zhao, G. Xia, and Y. Wang, “Beat transformer: Demixed beat and downbeat tracking with dilated selfattention,” in Proceedings of the 23rd International Society for Music Information Retrieval Conference, ISMIR 2022, Bengaluru, India, December 4-8, 2022, P. Rao, H. A. Murthy, A. Srinivasamurthy, R. M. Bittner, R. C. Repetto, M. Goto, X. Serra, and M. Miron, Eds., 2022, pp. 169–177. [Online]. Available: https://archives.ismir.net/ismir2022/paper/ 000019.pdf

[26] F. Foscarin, J. Schlüter, and G. Widmer, “Beat this! accurate beat tracking without DBN postprocessing,” in Proceedings of the 25th International Society for Music Information Retrieval Conference, ISMIR 2024, San Francisco, California, USA and Online, November 10-14, 2024, B. Kaneshiro, G. J. Mysore, O. Nieto, C. Donahue, C. A. Huang, J. H. Lee, B. McFee, and M. C. McCallum, Eds., 2024, pp. 962–969. [Online]. Available: https://doi.org/10.5281/zenodo.14877491

[27] Z. He, H. Meng, D. D. Huang, and R. Togneri, “Joint estimation of piano dynamics and metrical structure

with a multi-task multi-scale network,” in ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026, pp. 14 607–14 611.

[28] S. Murgul and M. Heizmann, “Beat and downbeat tracking in performance midi using an endto-end transformer architecture,” arXiv preprint arXiv:2507.00466, 2025.

[29] F. Foscarin, A. McLeod, P. Rigaux, F. Jacquemard, and M. Sakai, “ASAP: a dataset of aligned scores and performances for piano transcription,” in Proceedings of the 21th International Society for Music Information Retrieval Conference, ISMIR 2020, Montreal, Canada, October 11-16, 2020, J. Cumming, J. H. Lee, B. McFee, M. Schedl, J. Devaney, C. McKay, E. Zangerle, and T. de Reuse, Eds., 2020, pp. 534–541.

[30] P. Bontempi, S. Canazza, F. Carnovalini, and A. Rodà, “Research in computational expressive music performance and popular music production: A potential field of application?” Multimodal Technologies and Interaction, vol. 7, no. 2, p. 15, 2023.

[31] C. Hawthorne, A. Stasyuk, A. Roberts, I. Simon, C.-Z. A. Huang, S. Dieleman, E. Elsen, J. Engel, and D. Eck, “Enabling factorized piano music modeling and generation with the MAE-STRO dataset,” in International Conference on Learning Representations, 2019. [Online]. Available: https://openreview.net/forum?id=r1lYRjC9F7

[32] H. Zhang, J. Tang, S. R. M. Rafee, S. Dixon, G. Fazekas, and G. A. Wiggins, “ATEPP: A dataset of automatically transcribed expressive piano performance,” in Proceedings ofthe 23rd International Society for Music Information Retrieval Conference, 2022, pp. 446–453.

[33] L. Bradshaw and S. Colton, “Aria-midi: A dataset of piano midi files for symbolic music modeling,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 99 894–99 910.

[34] S. D. Peter, C. E. Cancino-Chacón, F. Foscarin, A. P. McLeod, F. Henkel, E. Karystinaios, and G. Widmer, “Automatic note-level score-to-performance alignments in the ASAP dataset,” Transactions of the International Society for Music Information Retrieval, vol. 6, no. 1, pp. 27–42, 2023.

[35] L. Liu, V. Morfi, and E. Benetos, “ACPAS: A dataset of aligned classical piano audio and scores for audioto-score transcription,” in Extended Abstracts for the Late-Breaking Demo Session ofthe 22nd International Society for Music Information Retrieval Conference, 2021.

[36] I. Borovik, “PianoCoRe: Combined and refined piano MIDI dataset,” Transactions of the International Society for Music Information Retrieval, vol. 9, no. 1, pp. 144–163, 2026.

[37] Z. Wang, K. Chen, J. Jiang, Y. Zhang, M. Xu, S. Dai, X. Gu, and G. Xia, “POP909: A pop-song dataset for music arrangement generation,” in Proceedings of the 21st International Society for Music Information Retrieval Conference, Montréal, Canada, 2020, pp. 38– 45.

[38] L. Bradshaw, S. Colton, A. Spangher, and S. Biderman, “Musically aware automatic piano transcription using synthetic pretraining,” unpublished manuscript. [Online]. Available: https: //www.alexander-spangher.com/papers/aria\_amt.pdf

[39] E. Nakamura, K. Yoshii, and H. Katayose, “Performance error detection and post-processing for fast and accurate symbolic music alignment.” in ISMIR. Suzhou, 2017, pp. 347–353.

[40] S. D. Peter, “Online symbolic music alignment with offline reinforcement learning,” in Proceedings of the 24th International Society for Music Information Retrieval Conference, Milan, Italy, 2023, pp. 634–641.

[41] Alibaba Cloud, “Qwen-Flash: Model documentation,” https://www.alibabacloud.com/help/en/model-studio/ qwen-flash, accessed: October 12, 2026.

[42] Y. Wu, K. Chen, T. Zhang, Y. Hui, T. Berg-Kirkpatrick, and S. Dubnov, “Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023. [Online]. Available: https://arxiv.org/abs/2211.06687

[43] I. Borovik, D. Gavrilev, and V. Viro, “SyMuPe: Affective and controllable symbolic music performance,” in Proceedings of the 33rd ACM International Conference on Multimedia. Association for Computing Machinery, 2025, pp. 10 699–10 708.

[44] C. E. Cancino-Chacón, S. D. Peter, E. Karystinaios, F. Foscarin, M. Grachten, and G. Widmer, “Partitura: A python package for symbolic music processing,” in Proceedings ofthe Music Encoding Conference (MEC 2022), Halifax, Canada, 2022.

[45] K. J. M. Lee, J. Ens, S. Adkins, P. Sarmento, M. Barthet, and P. Pasquier, “The gigamidi dataset with features for expressive music performance detection,” Trans. Int. Soc. Music. Inf. Retr., vol. 8, no. 1, 2025. [Online]. Available: https://doi.org/10.5334/tismir.203

[46] M. E. Davies, N. Degara, and M. D. Plumbley, “Evaluation methods for musical audio beat tracking algorithms,” Queen Mary University ofLondon, Centrefor Digital Music, Tech. Rep. C4DM-TR-09-06, 2009.