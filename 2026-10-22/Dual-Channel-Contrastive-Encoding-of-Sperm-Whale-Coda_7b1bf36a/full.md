# Dual-Channel Contrastive Encoding of Sperm Whale Coda

João Victor Quintanilha<sup>†</sup>, Emma Virnelli<sup>†</sup>, Shuo Zhang

Tufts University, Medford, MA, USA

Joao.Quintanilha@tufts.edu, Emma.Virnelli@tufts.edu, Shuo.Zhang@tufts.edu

Abstract—Sperm whales (Physeter macrocephalus) communicate through short click sequences called codas, whose structure includes the rhythmic pattern of inter-click intervals (ICI) and the spectral composition of the clicks. In this paper, we propose the Dual-Channel Contrastive Encoder (DCCE), which explicitly encodes rhythm and spectral information with separate encoders before fusing them. We evaluate DCCE on three codaunderstanding tasks: social unit, coda type, and individual identification. We compare DCCE against strong baselines from literature under a common linear-probe protocol and analyze the role of each channel and training objective through ablations. On 1,383 codas, shared partitions and validation-only layer selection support three random splits and three recording-date-disjoint folds. DCCE with unit pairing and type supervision achieves mean random-split macro-F1 of 0.901/0.865/0.597 for unit/type/identity. Raw ICI remains strongest for type (0.987), and ICI plus temporal mel statistics is a strong identity reference (0.724). An auxiliary identity objective improves spectral-branch identity prediction but does not improve the mean fused-identity score in this configuration. Date-grouped evaluation and four year holdouts expose generalization limits. The results support task-dependent combinations of timing and spectral information, without establishing independent biological channels or general architectural superiority. Code, notebooks, and complete results are available at https://dcce-research-site.vercel.app.

Index Terms—Sperm whale codas, contrastive learning, bioacoustics, representation learning, individual identification

## 1. INTRODUCTION

Sperm whales (Physeter macrocephalus) live in stable, matrilineal family units that belong to vocal clans, each clan sharing a distinct vocal dialect [1], [2]. That dialect is built from codas: short, stereotyped sequences of clicks that whales exchange during social contact. Deciphering what codas encode is a central goal of computational bioacoustics and of the Cetacean Translation Initiative [1], [3]. The rhythm channel, the pattern of inter-click intervals (ICIs), contributes to coda type: a category defined by click count and timing [2], [4]. The spectral channel, the frequency content within each click, includes vowel- and diphthong-like patterns [5], [6]. The analogy to human voice motivates asking which features distinguish callers, without assuming that the two channels are independent.

The individual signal in a coda depends both on the coda type produced and on the set of whales one compares against. Gero et al. [2] make these two dependencies concrete: discriminant analysis of 5R1 timing classified individuals within units J, U, and F at 73%, 78%, and 83% accuracy, respectively, but at 46% when individuals across units were pooled. The 1+1+3 type did not reliably discriminate individuals within units. Recovering who produced a coda therefore depends on which coda was produced and on how large and diverse the candidate pool is.

In this paper we propose the Dual-Channel Contrastive Encoder (DCCE), which learns a representation of sperm whale codas by encoding rhythm and spectral information with separate encoders followed by a fusion layer. Our guiding question is: does encoding rhythm and spectral texture as separate input channels, rather than learning implicitly from one input representation, improve what a model can learn from sperm whale codas? The study contributes a dual-input architecture, matched supervision and channel ablations, and an evaluation of recording-context generalization.

## 2. RELATED WORK

Coda structure and identity. In the notation of Gero et al. [2], 5R1 denotes a five-click, regularly spaced type, while 1+1+3 groups five clicks into one, one, and three. Sharma et al. [4] showed that rhythm, tempo, rubato, and ornamentation yield at least 143 frequently realized combinations. Beguš et al. [5] identified vowel- and diphthong-like spectral patterns; subsequent work reports interactions between vowel quality and conventional coda type [6]. Bermant et al. [7] already investigated neural sperm whale classification, including individual identity.

Learned models for cetacean audio. WhAM [3] provides a spermwhale-specific reference, learning acoustic-token representations whose downstream classification ability is evaluated by probing. Broader bioacoustic references are AVES [8] and BirdNET [9]. Tolkova et al. [10] applied BirdNET embeddings to individual identification using 234 upcalls of 11 North Atlantic right whales at three sites.

Representation-learning lineages. Self-supervised audio models such as wav2vec 2.0 [11] and HuBERT [12] demonstrate learning speech representations from unlabeled audio. SimCLR uses the NT-Xent objective [13], while CLIP learns aligned representations across image and text inputs [14]. Finally, the speaker-verification lineage motivates mapping variable-length acoustic input to identity-relevant vectors: i-vectors and x-vectors provide statistical and neural approaches [15], [16], and metric-learning studies examine alternatives to classification losses [17]. These precedents motivate the design, but neither establish its superiority on codas nor turn our closed-set ID task into open-set speaker verification.

## 3. METHODOLOGY

We frame coda understanding as a representation-learning problem: given a coda, DCCE learns a single fixed-dimensional embedding from timing and spectral inputs, and we measure how much task-relevant structure that embedding captures with lightweight linear probes. Figure 1 summarizes the design. DCCE is trained by combining a contrastive objective on z with optional auxiliary classification losses on the branches.

## 3.1. Rhythm Encoder

The rhythm of a coda is captured by its sequence of inter-click intervals (ICIs): the elapsed times, in seconds, between consecutive clicks. A coda of k clicks yields k − 1 ICIs; since the codas in our data contain up to ten clicks, each coda has at most nine ICIs. Annotated ICIs are zero-padded on the right and standardized using training examples only. A two-layer GRU with hidden size 64 and inter-layer dropout 0.2 reads the sequence one ICI at a time; a linear projection and layer normalization map its final hidden state to $r \in \mathbb { R } ^ { 6 4 }$ . We choose a compact recurrent encoder for these short sequences and the small dataset.

![](images/8f19fb34a1c410a447fb35686d8fa0e668ab73f73dce5366f808c5af50a3a979.jpg)  
Fig. 1: DCCE: separate input encoders, fusion, and optional branch objectives. 1Positive-pair construction varies across conditions. The linear probe is trained after freezing the representation.

## 3.2. Spectral Encoder

The spectral channel is derived from a log-mel-spectrogram of the coda audio. Each recording is converted to mono and resampled to 48 kHz; native rates are 44.1, 48, 96, and 120 kHz. A short-time Fourier transform uses a 2,048-sample Hann window and a 1,152-sample hop, mapped onto 64 mel bands spanning 0–8 kHz. Log power is peak-relative, clipped to 80 dB, padded at the minimum level to a fixed $6 4 \times 2 5 6$ representation, and transformed by $x _ { \mathrm { d B } } / 4 0 + 1$ . No mel input is truncated.

Three convolutional blocks (Conv2d–BatchNorm–ReLU–MaxPool, with 16, 32, and 64 channels) use $3 \times 3$ kernels and 2 × 2 pooling to reduce the input to a $6 4 \times 8 \times 3 2$ feature map. The flattened output passes through dropout 0.3 and a two-layer MLP with layer normalization to $s \in \mathbb { R } ^ { 6 4 }$ . The full-coda spectrogram retains temporal structure, so this branch is not restricted to within-click identity cues.

## 3.3. Fusion MLP

We concatenate r and s, then apply LayerNorm, Linear(128,64), ReLU, and Linear(64,64) to obtain z. Embeddings are normalized inside the contrastive loss. Primary probes use the saved fusion output; separate branch probes are post-hoc diagnostics.

## 3.4. Training Objective

The objective combines NT-Xent [13] with optional weighted crossentropies for type on r and identity on s. The type head encourages the rhythm branch to retain coda-type information rather than only unit-level structure. The ID head encourages individual discrimination, following the use of identity-classification supervision for speaker embeddings [16]:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { N T X e n t } } + 0 . 5 a _ { T } \mathcal { L } _ { \mathrm { t y p e } } + 0 . 5 a _ { I } \mathcal { L } _ { \mathrm { I D } } ,\tag{1}
$$

where $a _ { T } , a _ { I } \in \{ 0 , 1 \}$ . Inverse-frequency class weights are computed from training counts only; ineligible identity labels are masked. For positive views $i , j ,$ the contrastive term uses

$$
\ell ( i , j ) = - \log \frac { \exp ( \sin ( z _ { i } , z _ { j } ) / \tau ) } { \sum _ { k \ne i } \exp ( \sin ( z _ { i } , z _ { k } ) / \tau ) } , \qquad \tau = 0 . 0 7 ,\tag{2}
$$

with cosine similarity and a symmetric average over positive pairs. Assigning different auxiliary tasks to the branches encourages discriminative features but does not enforce statistical independence.

## 3.5. Positive-Pair Construction

The primary construction pairs distinct training codas A, B from the same social unit: $f ( r _ { A } , s _ { A } )$ and $f ( r _ { B } , s _ { B } )$ . This uses unit supervision. Actual channel swapping instead pairs $f ( r _ { A } , s _ { B } )$ and $f ( r _ { B } , s _ { A } )$ Without unit pairing, positives are independently augmented views of the same coda. Other batch views act as negatives, including possible same-unit examples; the objective is not a full supervised-contrastive treatment of all same-unit codas.

Each view masks a random four-bin frequency band and eightframe time band, and adds Gaussian noise of SD 0.01 to standardized ICI entries corresponding to nonzero raw intervals. Padded intervals retain their standardized values. All main DCCE conditions use uniform shuffled sampling, 50 epochs of AdamW (learning rate $1 0 ^ { - 3 }$ weight decay $1 0 ^ { - 4 } ) ,$ cosine decay, and batch size 64. Temperature, auxiliary weights, sampling, and optimizer settings were fixed for this exploratory batch, not validation-tuned.

## 3.6. DCCE Variants

We denote unit pairing, type loss, and ID loss by U/T/I; three binary digits specify their presence. The main pair, DCCE 110 versus 111, differs only in ID loss. Both are task-informed models: absence of ID loss does not mean absence of supervision. We tested all eight U/T/I settings; selected contrasts are reported here, including 000, which removes task labels from the representation objective and sampler, but retains annotated ICI and supervised downstream probing.

Rhythm-only and spectrum-only controls apply both auxiliary heads to the available branch, with the same coefficients. Joint concatenation produces 128 dimensions without the fusion MLP; it is not independently trained late fusion. These architectures are evaluated with and without ID loss. Additional conditions test actual swapping and a 16-kHz mel ceiling; the latter also changes filter spacing at fixed band count.

## 3.7. Linear Probe Evaluation

All encoder weights are frozen after training. Balanced logistic regression predicts social unit, coda type, or individual ID. Features are standardized on training data, with fixed C = 1 and at most 2,000 iterations. WhAM/AVES layers are selected per task using validation macro-F1 only, with earliest-layer tie breaking; BirdNET provides one representation. There is no training-plus-validation refit. Neural checkpoints use a fixed final epoch.

## 4. EXPERIMENTS

## 4.1. Dataset and Evaluation Protocol

The dataset used in our experiment consists of 1,501 audio recordings from the Dominica Sperm Whale Project (DSWP) public HuggingFace release (orrp/DSWP) [3]. After excluding 118 noise-flagged files, 1,383 clean codas remain, spanning three social units (A, D, F) and 22 coda types. Labels (social unit, coda type, individual ID) are reconstructed from DominicaCodas.csv, using its codaNUM2018 index to match DSWP file indices. Reconstructed metadata place these codas in 2005–2010 and clan EC1. ID evaluation uses 754 codas from 11 individuals: 620 unknown-ID codas, eight ambiguous two-individual annotations, and one identified singleton are excluded from ID training/probing, while remaining eligible for other tasks. Severe class imbalance (unit F contributes 821/1,383 codas, or 59.4%) motivates macro-F1 and balanced class weights. All tasks share the same coda partition, keeping every locally trained representation separate from its test examples.

We use three random splits (seeds 0–2), approximately 64/16/20% train/validation/test, stratifying eligible identities and otherwise unit.

Table 1: Main comparison: macro-F1 mean ± sample SD over three runs. Fixed/frozen rows train only the downstream probe locally; + ICI adds annotated timing. DCCE uses unit pairing (U), type loss (T), and optional ID loss (I), with uniform sampling. AVES FT uses unit/type/ID supervision while freezing the waveform frontend. All learned representations are frozen before probing. Primary scores exclude unsupported test classes; grouped ID coverage differs from random splits.
<table><tr><td></td><td colspan="3">Random split</td><td colspan="3">Recording-date grouped</td></tr><tr><td>Model</td><td>Unit</td><td>Type</td><td>ID</td><td>Unit</td><td>Type</td><td>ID</td></tr><tr><td>Raw ICI</td><td> $0 . 5 7 0 \pm 0 . 0 2 3$ </td><td> $0 . 9 8 7 \pm 0 . 0 0 9$ </td><td> $0 . 4 3 9 \pm 0 . 0 0 7$ </td><td> $0 . 5 4 0 \pm 0 . 0 3 6$ </td><td> $0 . 9 5 6 \pm 0 . 0 3 6$ </td><td> $0 . 3 7 9 \pm 0 . 1 0 3$ </td></tr><tr><td>Temporal mel</td><td> $0 . 7 7 3 \pm 0 . 0 0 7$ </td><td> $0 . 2 0 2 \pm 0 . 0 2 5$ </td><td> $0 . 4 2 9 \pm 0 . 0 4 5$ </td><td> $0 . 5 6 9 \pm 0 . 0 9 4$ </td><td> $0 . 0 6 9 \pm 0 . 0 1 2$ </td><td> $0 . 1 3 0 \pm 0 . 0 7 4$ </td></tr><tr><td>ICI + temporal mel</td><td> $0 . 8 3 7 \pm 0 . 0 1 3$ </td><td> $0 . 8 0 4 \pm 0 . 0 1 6$ </td><td> $0 . 7 2 4 \pm 0 . 0 3 9$ </td><td> $0 . 6 0 8 \pm 0 . 1 5 0$ </td><td> $0 . 6 6 8 \pm 0 . 0 5 1$ </td><td> $0 . 2 5 5 \pm 0 . 0 6 2$ </td></tr><tr><td>WhAM (frozen)</td><td> $0 . 9 0 6 \pm 0 . 0 1 1$ </td><td> $0 . 2 5 8 \pm 0 . 0 4 6$ </td><td> $0 . 4 5 0 \pm 0 . 0 5 2$ </td><td> $0 . 6 6 7 \pm 0 . 1 5 5$ </td><td> $0 . 0 6 6 \pm 0 . 0 1 6$ </td><td> $0 . 1 0 1 \pm 0 . 0 7 5$ </td></tr><tr><td>AVES (frozen)</td><td> $0 . 9 1 0 \pm 0 . 0 1 0$ </td><td> $0 . 2 5 3 \pm 0 . 0 2 8$ </td><td> $0 . 4 9 1 \pm 0 . 0 2 9$ </td><td> $0 . 7 0 0 \pm 0 . 1 0 0$ </td><td> $0 . 0 6 1 \pm 0 . 0 2 2$ </td><td> $0 . 0 7 4 \pm 0 . 0 5 1$ </td></tr><tr><td>BirdNET (frozen)</td><td> $0 . 8 8 7 \pm 0 . 0 2 0$ </td><td> $0 . 2 1 5 \pm 0 . 0 5 3$ </td><td> $0 . 4 8 0 \pm 0 . 0 2 2$ </td><td> $0 . 6 1 2 \pm 0 . 1 3 3$ </td><td> $0 . 0 6 3 \pm 0 . 0 1 1$ </td><td> $0 . 0 8 5 \pm 0 . 0 2 2$ </td></tr><tr><td>WhAM + ICI (frozen)</td><td> $0 . 9 2 1 \pm 0 . 0 0 5$ </td><td> $0 . 7 2 2 \pm 0 . 0 2 7$ </td><td> $0 . 5 9 8 \pm 0 . 0 2 8$ </td><td> $0 . 7 6 8 \pm 0 . 0 5 7$ </td><td> $0 . 3 7 6 \pm 0 . 1 5 1$ </td><td> $0 . 1 5 3 \pm 0 . 1 3 5$ </td></tr><tr><td> $\mathrm { A V E S } + \mathrm { I C I } ( \mathrm { f r o z e n } )$ </td><td> $0 . 9 2 6 \pm 0 . 0 1 1$ </td><td> $0 . 6 6 7 \pm 0 . 0 4 4$ </td><td> $0 . 6 5 2 \pm 0 . 0 4 4$ </td><td> $0 . 7 1 2 \pm 0 . 1 1 1$ </td><td> $0 . 3 6 6 \pm 0 . 0 8 4$ </td><td> $0 . 1 2 8 \pm 0 . 1 0 3$ </td></tr><tr><td>BirdNET + ICI (frozen)</td><td> $0 . 8 8 4 \pm 0 . 0 1 8$ </td><td> $0 . 4 9 7 \pm 0 . 0 2 8$ </td><td> $0 . 5 8 0 \pm 0 . 0 3 1$ </td><td> $0 . 6 2 8 \pm 0 . 0 9 5$ </td><td> $0 . 2 8 7 \pm 0 . 0 6 1$ </td><td> $0 . 1 0 0 \pm 0 . 0 5 5$ </td></tr><tr><td>DCCE U+T</td><td> $0 . 9 0 1 \pm 0 . 0 2 0$ </td><td> $0 . 8 6 5 \pm 0 . 1 1 5$ </td><td> $0 . 5 9 7 \pm 0 . 0 5 1$ </td><td> $0 . 5 9 8 \pm 0 . 0 7 8$ </td><td> $0 . 6 9 2 \pm 0 . 0 8 8$ </td><td> $0 . 2 4 8 \pm 0 . 1 3 0$ </td></tr><tr><td>DCCE U+T+I</td><td> $0 . 9 0 7 \pm 0 . 0 1 0$ </td><td> $0 . 7 3 5 \pm 0 . 0 9 1$ </td><td> $0 . 5 8 1 \pm 0 . 0 1 2$ </td><td> $0 . 5 9 6 \pm 0 . 0 5 2$ </td><td> $0 . 6 2 8 \pm 0 . 0 7 4$ </td><td> $0 . 2 2 3 \pm 0 . 1 7 5$ </td></tr><tr><td>AVES FT</td><td> $0 . 9 2 4 \pm 0 . 0 0 7$ </td><td> $0 . 2 4 3 \pm 0 . 0 2 5$ </td><td> $0 . 4 1 4 \pm 0 . 0 7 8$ </td><td> $0 . 6 2 6 \pm 0 . 1 5 7$ </td><td> $0 . 0 7 2 \pm 0 . 0 5 0$ </td><td> $0 . 0 6 0 \pm 0 . 0 2 6$ </td></tr><tr><td> $\mathrm { A V E S } + \mathrm { I C I } \mathrm { F T }$ </td><td> $0 . 9 1 3 \pm 0 . 0 1 1$ </td><td> $0 . 4 6 6 \pm 0 . 0 5 9$ </td><td> $0 . 5 4 4 \pm 0 . 0 1 7$ </td><td> $0 . 6 4 1 \pm 0 . 1 5 8$ </td><td> $0 . 2 1 1 \pm 0 . 0 6 9$ </td><td> $0 . 0 7 3 \pm 0 . 0 3 8$ </td></tr></table>

![](images/b1b450e1c9c819e5a540764d79e5f5f352393e04b9fd3d93ee6835a5a0f8df40.jpg)  
Fig. 2: Random versus recording-date-grouped evaluation for four representative models. Points show mean macro-F1 over three runs; sample SD and the full model comparison appear in Table 1. Lines connect protocol means. Grouped identity evaluation covers different supported classes and examples, so these differences are not isolated estimates of a recording-context penalty.

Recording-date splits use the first three outer folds (indices 0–2) of a five-fold stratified-group partition (shuffle seed 0), with grouped validation partitions. Dates are session proxies. Four additional year holdouts exclude the test year completely; validation uses a dategrouped fold from the remaining years, and neural seed is fixed at 0.

Primary macro-F1 averages over true test classes observed in training, on supported examples. Grouped ID support is 186/213, 111/121, and 145/145 eligible examples, spanning three, six, and six scored identities. Thus random/grouped ID scores describe different supported populations. Full-eligible secondary scores include unseen classes. We report means and sample SD over three runs, not confidence intervals.

The design is exploratory: earlier unit-balanced experiments informed these comparisons, and year/input-matched controls followed partial main results. Earlier training-regime scores are not pooled with the matched uniform-sampling ablations.

frame bins (512 dimensions), and their concatenation (521 dimensions). Temporal binning retains coarse timing that a single time-averaged mel vector would discard. Frozen WhAM, AVES, and BirdNET are evaluated both alone and concatenated with ICI. These last controls match timing-input availability, with only probes trained locally.

Pretrained frontends retain native procedures: AVES at 16 kHz over the full segment; the released WhAM codec at 44.1 kHz with mono/loudness processing; BirdNET at 48 kHz with a three-second window. Three waveforms exceed that window.

We evaluate three literature-grounded baselines under the same splits and linear-probe protocol as DCCE: WhAM, AVES, and BirdNET. Fixed references are raw ICI, temporal mel means/std within four valid-

## 4.2. DCCE vs. Baselines

Supervised AVES adaptation updates the projection and all 12 transformer blocks, keeping the waveform CNN frozen. Unit/type/ID loss weights are 1/0.5/0.5, with uniform sampling. Ten fixed AdamW epochs use backbone/head rates $2 \times 1 0 ^ { - 5 } / 1 \bar { 0 } ^ { - 3 } ,$ , weight decay $1 0 ^ { - 4 } .$ microbatches of four accumulated to 64, and clip gradients at 1. Labels are comparable in availability, but objectives, capacity, pretraining, and budgets remain different.

## 4.3. Results

No single model wins on all three tasks (Table 1). Raw ICI attains $0 . 9 8 7 \pm 0 . 0 0 9$ random type macro-F1; ICI plus temporal mel attains $0 . 7 2 4 \pm 0 . 0 3 9 .$ ID macro-F1. Including unseen type labels changes the raw-ICI random/grouped type means from 0.987/0.956 to 0.968/0.919.

Table 2: Matched architecture controls: macro-F1 mean ± sample SD, three runs. R/G denote random/date-grouped; digits denote U/T/I. All use uniform sampling, 50 epochs, and whole-coda pairing. Enabled auxiliary heads act on the available branch; representations are frozen before probing. Joint concatenation is 128-dimensional; other outputs are 64-dimensional. Fused DCCE anchors appear in Table 1.
<table><tr><td>Model</td><td>R Type</td><td>R ID</td><td>G ID</td></tr><tr><td>Rhythm 110</td><td> $. 9 7 8 \pm . 0 2 1$ </td><td> $. 4 7 7 \pm . 0 3 7$ </td><td> $. 3 5 2 \pm . 1 3 4$ </td></tr><tr><td>Rhythm 111</td><td> $. 9 7 3 \pm . 0 1 3$ </td><td> $. 4 5 4 \pm . 0 6 1$ </td><td> $. 3 3 1 \pm . 1 4 3$ </td></tr><tr><td>Spectrum 110</td><td> $. 2 0 2 \pm . 0 2 1$ </td><td> $. 3 5 8 \pm . 1 0 3$ </td><td> $. 0 5 4 \pm . 0 4 7$ </td></tr><tr><td>Spectrum 111</td><td> $. 1 9 9 \pm . 0 3 7$ </td><td> $. 4 4 4 \pm . 0 9 2$ </td><td> $. 0 9 2 \pm . 0 1 6$ </td></tr><tr><td>Joint concat. 110 Joint concat. 111</td><td> $. 9 3 9 \pm . 0 6 2$   $. 9 2 7 \pm . 0 6 0$ </td><td> $. 6 3 6 \pm . 0 1 5$   $. 6 9 6 \pm . 0 5 5$ </td><td> $. 2 7 5 \pm . 1 1 7$   $. 2 3 9 \pm . 0 4 3$ </td></tr></table>

Strong timing references are essential when type labels themselves reflect interval structure.

DCCE 110 attains random unit/type/ID means of 0.901/0.865/0.597. Its ID mean exceeds either separately probed branch (rhythm 0.459, spectrum 0.367), but remains below the simple combined-feature reference. DCCE 111 gives 0.907/0.735/0.581: adding ID loss does not improve mean fused-ID performance in this configuration. These descriptive three-run differences do not establish a universal effect of identity supervision.

Frozen AVES plus ICI attains random ID 0.652, versus 0.491 for audio alone. Fine-tuned AVES plus ICI attains 0.544, versus 0.414 without ICI. Input availability therefore materially changes the comparison, while these fixed adaptation schedules do not establish optimal pretrained performance.

## 4.4. Ablation Study

Without auxiliary losses, same-unit pairing changes random unit/ID means from 0.693/0.624 (000) to 0.861/0.434 (100). The grouped ID ordering reverses, 0.181 versus 0.245. Pairing therefore imposes task and partition-dependent tradeoffs.

Table 2 compares channel and concatenation controls under matched label access, against the fused DCCE rows in Table 1. Neither these results nor the dimensionality-different concatenation control establish an exclusively spectral identity code or a necessary benefit from the fusion MLP.

Actual swapping raises random ID from 0.581 to 0.657 relative to whole-coda 111, while grouped ID falls from 0.223 to 0.189. A 16-kHz ceiling raises random ID to 0.631 without and 0.623 with ID loss; grouped scores are 0.250 and 0.173. These are bandwidth sensitivities, not localization of biological identity cues.

Post-hoc probes of fixed checkpoints expose a branch/fusion distinction: ID supervision raises random spectral-branch ID from 0.367 to 0.443, while fused ID falls from 0.597 to 0.581. Rhythmbranch type remains 0.983/0.982 without/with ID loss, whereas fused type is 0.865/0.735. Auxiliary branch performance need not carry through fusion.

## 4.5. Recording Context and Generalization

Explicit day/month/year parsing includes all 1,383 codas. Year-by-unit counts (A/D/F) are 10/0/358 in 2005, 27/171/65 in 2008, 30/18/47 in 2009, and 174/132/351 in 2010. Pearson $\chi ^ { 2 } = 5 3 0 . 0 4 ~ ( \mathrm { d f } { = } 6 )$ , with uncorrected Cramér’s $V = \sqrt { \chi ^ { 2 } / [ N \operatorname* { m i n } ( 3 , 2 ) ] } = 0 . 4 3 8 .$ . This is an association in the dataset, not proof of a model’s causal reliance on recording artifacts.

Figure 2 shows lower grouped means across all three tasks for the four illustrated models, with task-dependent orderings. Holding out 2005/2008/2009/2010 yields unit macro-F1 of 0.592/0.362/0.767/0.346 for DCCE 110 and 0.448/0.391/0.713/0.357 for DCCE 111. Each year uses one fit; 2005 contains two test units, other years three. Both variants score below frozen AVES in every year; these outcomes do not establish general protection against recording-context variation.

We report selected unit/type ID diagnostics. For the previously discussed 1+1+3 type, DCCE 110’s grouped-fold ID macro-F1 is 0.000/0.233/0.234 (52/53/34 codas; 3/3/4 identities); single-identity strata are unscored. Separate within-unit probes require at least two supported identities in train and test. Only unit F supports these probes on all grouped folds: mean ID is 0.384 for raw ICI, 0.267 for DCCE 110, and 0.238 for DCCE 111. These comparisons reduce one group-membership shortcut without removing all type/context dependence.

## 5. DISCUSSION

## 5.1. Interpretation and Implications

DCCE’s performance across the three tasks supports studying rhythmic and spectral information as separate inputs, but does not assign each biological attribute exclusively to one branch. Fusion can improve over individually probed branches, while simple combined features remain strong references. For small bioacoustic datasets, these findings motivate matching supervision and annotation access, retaining simple temporal controls, and evaluating across recording groups. The branch/fusion distinction makes cross-channel coupling a design question for future work rather than a fixed architectural choice. Objective interactions or compression are possible explanations, not mechanisms established by these experiments.

## 5.2. Limitations

Following our analysis, a key limitation is the dataset itself: a single Dominica population with three social units, all from the EC1 clan. Another restriction is that there are no vowel annotations for the 1,383 codas evaluated here: the available hand-annotated vowel metadata covers different coda identifiers. The spectral branch is trained through the fused contrastive objective and optional ID classification, not explicit phonological categories.

Annotated ICI is additional information unavailable to a waveformonly system. Closed-set ID tests do not establish recognition of unseen animals or communicative meaning. Sparse identity coverage and three runs limit precision; recording dates are not verified sessions.

WhAM’s published training uses DSWP+CETI [3], and this evaluation also uses DSWP. Exact checkpoint training membership is unavailable, so local holdouts do not establish external pretraining independence. The exploratory design and heterogeneous model histories further limit causal or universal comparisons.

## 6. CONCLUSION AND FUTURE WORK

We asked whether encoding rhythm and spectral texture separately before fusion improves what a model can learn from sperm whale codas. DCCE combines useful information across unit, type, and identity tasks, but gains depend on supervision and evaluation conditions. These experiments do not establish a benefit of learned fusion over joint concatenation. A further question is whether DCCE generalizes to whale populations beyond the single Dominica clan studied in this work. Beyond adjusting the dataset, future work could also treat the strength of the identity objective as a tunable parameter, testing how auxiliary weighting and fusion affect the retention of type and identity information. Such evaluations should use independent recordings and explicit annotation budgets. The paper, code, executed notebooks, complete result tables, and audio examples are available at https://dcce-research-site.vercel.app.

## REFERENCES

[1] J. Andreas, G. Beguš, M. M. Bronstein, R. Diamant, D. Delaney, S. Gero, S. Goldwasser, D. F. Gruber, S. de Haas, P. Malkin, N. Pavlov, R. Payne, G. Petri, D. Rus, P. Sharma, D. Tchernov, P. Tønnesen, A. Torralba, D. Vogt, and R. J. Wood, “Toward understanding the communication in sperm whales,” iScience, vol. 25, no. 6, p. 104393, 2022.

[2] S. Gero, H. Whitehead, and L. Rendell, “Individual, unit and vocal clan level identity cues in sperm whale codas,” Royal Society Open Science, vol. 3, no. 1, p. 150372, 2016.

[3] O. Paradise, P. Muralikrishnan, L. Chen, H. Flores García, B. Pardo, R. Diamant, D. F. Gruber, S. Gero, and S. Goldwasser, “WhAM: Towards a translative model of sperm whale vocalization,” arXiv:2512.02206, 2025. [Online]. Available: https://arxiv.org/abs/2512.02206

[4] P. Sharma, S. Gero, R. Payne, D. F. Gruber, D. Rus, A. Torralba, and J. Andreas, “Contextual and combinatorial structure in sperm whale vocalisations,” Nature Communications, vol. 15, no. 1, p. 3617, 2024.

[5] G. Beguš, R. L. Sprouse, A. Leban, M. Silva, and S. Gero, “Vowel- and Diphthong-Like Spectral Patterns in Sperm Whale Codas,” Open Mind, vol. 9, pp. 1849–1874, 2025.

[6] G. Beguš, M. D ˛abkowski, R. L. Sprouse, D. F. Gruber, and S. Gero, “The phonology of sperm whale coda vowels,” Proceedings of the Royal Society B, vol. 293, no. 2069, p. 20252994, 2026.

[7] P. C. Bermant, M. M. Bronstein, R. J. Wood, S. Gero, and D. F. Gruber, “Deep Machine Learning Techniques for the Detection and Classification of Sperm Whale Bioacoustics,” Scientific Reports, vol. 9, no. 1, p. 12588, 2019.

[8] M. Hagiwara, “AVES: Animal vocalization encoder based on selfsupervision,” arXiv:2210.14493, 2022. [Online]. Available: https: //arxiv.org/abs/2210.14493

[9] S. Kahl, C. M. Wood, M. Eibl, and H. Klinck, “BirdNET: A deep learning solution for avian diversity monitoring,” Ecological Informatics, vol. 61, p. 101236, 2021.

[10] I. Tolkova, H. Klinck, D. A. Cusano, A. Kügler, and S. E. Parks, “Detection, communication, and individual identification with deep audio embeddings: A case study with North Atlantic right whales,” PLOS Computational Biology, vol. 22, no. 6, p. e1013321, 2026.

[11] A. Baevski, H. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: A framework for self-supervised learning of speech representations,” arXiv:2006.11477, 2020. [Online]. Available: https://arxiv.org/abs/2006. 11477

[12] W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, “HuBERT: Self-supervised speech representation learning by masked prediction of hidden units,” arXiv:2106.07447, 2021. [Online]. Available: https://arxiv.org/abs/2106.07447

[13] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” in Proceedings of ICML, 2020, pp. 1597–1607.

[14] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” in Proceedings of ICML, vol. 139, 2021, pp. 8748–8763. [Online]. Available: https://proceedings.mlr.press/v139/radford21a.html

[15] N. Dehak, P. J. Kenny, R. Dehak, P. Dumouchel, and P. Ouellet, “Front end factor analysis for speaker verification,” IEEE Transactions on Audio, Speech, and Language Processing, vol. 19, no. 4, pp. 788–798, 2011.

[16] D. Snyder, D. Garcia-Romero, G. Sell, D. Povey, and S. Khudanpur, “X-vectors: Robust DNN embeddings for speaker recognition,” in Proceedings of ICASSP, 2018, pp. 5329–5333.

[17] J. S. Chung, J. Huh, S. Mun, M. Lee, H. S. Heo, S. Choe, C. Ham, S. Jung, B.-J. Lee, and I. Han, “In defence of metric learning for speaker recognition,” arXiv:2003.11982, 2020. [Online]. Available: https://arxiv.org/abs/2003.11982