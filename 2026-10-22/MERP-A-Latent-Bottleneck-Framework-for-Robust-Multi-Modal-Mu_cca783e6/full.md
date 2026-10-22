# MERP: A Latent-Bottleneck Framework for Robust Multi-Modal Multi-Label Emotion Recognition

Chuangxin Zhao, Meimei Zhang, Canran Xiao, Jiachen Luo, Jiajun He, Kaijie Chen, Huy Phan, Tomoki Toda, and Bjorn W. Schuller,¨ Fellow, IEEE

Abstract—Multimodal multi-label emotion recognition is important for affect-aware systems, but models trained with complete text, visual, and acoustic streams often degrade sharply when some streams are unavailable at deployment. We address this train–test gap by learning a compact representation that preserves emotion-specific evidence from any available subset of modalities. We propose the Multi-modal Emotion Recognition Perceiver (MERP), which maps the joint sequence into a fixedsize latent bank, thereby keeping repeated reasoning independent of raw input length; it extracts label-specific evidence before and after latent reasoning and aligns the two levels through reconstruction. MERP further uses positive and negative class prototypes, a noise-gated feature enhancer with frequency-shaped perturbations during training, and confidence-aware class-wise fusion to strengthen separability, reduce sensitivity to missing inputs, and combine complementary predictors. Under 20% testtime random modality drop, MERP achieves micro-F1 scores of 0.563 on CMU-MOSEI and 0.780 on M<sup>3</sup>ED, exceeding the strongest evaluated baselines by 0.004 and 0.005, respectively; in the severe CMU-MOSEI audio-only setting, it retains 0.473 micro-F1 versus 0.022 for LDDU. These results show that fixed latent reasoning provides a practical, testable route to robust multimodal emotion recognition without requiring missing-modality examples during training. Code is available at https://github.com/Creative-zcx/MERP.

Index Terms—Multi-modal emotion recognition, multi-label classification, missing modality, Perceiver-IO, latent bottleneck.

## I. INTRODUCTION

and facial behaviour is central to affect-aware interfaces, conversational systems, and large-scale media analysis [1]. These modalities provide complementary evidence, making their joint analysis more informative than isolated unimodal prediction [2]. Multi-modal multi-label emotion recognition (MMER) is especially valuable because an utterance can express several emotions simultaneously, but this setting must resolve both cross-modal ambiguity and correlations among co-occurring labels [3].

![](images/6be0dcd318583cf62fe61972ffd6089797c7f0aad76ddc92a70317edb366390b.jpg)  
Fig. 1. Fixed latent reasoning preserves class-specific evidence under compound modality loss. With text and visual streams unavailable, the schematic input-bound path loses the Surprise label after late fusion (a). MERP maps the same zeroed input into a fixed latent workspace and uses class-specific readouts to retain both labels (b).

MMER has advanced substantially through increasingly structured dependency and representation learning. HHMPN propagates information over a hierarchy of modality features and emotion labels [4], while TAILOR separates common and private modality factors before label-guided decoding [5]. More recent systems improve label discrimination through contrastive reconstruction in CARAT [6] and probabilistic uncertainty modeling in LDDU [7]. In parallel, recent studies have strengthened categorical multi-modal emotion recognition through emotion-shift-aware cross-modal fusion [8], relational subgraph interaction [9], and interpretable hybrid fusion [10]. These advances demonstrate the value of structured fusion, but the categorical models assign one emotion to each instance, whereas MMER must distinguish several correlated labels. Moreover, their computation remains organized around modality-specific or joint input sequences. Directional crossmodal attention, for example, captures fine-grained interactions but retains token-level costs that grow with the input streams [11]; label-specific decoding and late probabilistic fusion improve prediction without removing this structural dependence.

Incomplete input exposes a second, closely coupled limitation. Existing approaches recover missing representations through cross-modal imagination [12], align complete and incomplete views in a shared space [13], construct a stable proxy modality [14], or adapt a pretrained model to missing inputs with a small parameter overhead [15]. Recent work further suppresses missing-input noise through heterogeneous graph denoising for conversational emotion recognition [16]. These strategies have broadened the scope of robust multimodal learning, but they predominantly address single-label sentiment or emotion tasks and usually treat missingness through imputation, alignment, denoising, or missing-pattern adaptation [17]. They therefore do not resolve the central tension in deployable MMER: the model must preserve distinct evidence for multiple correlated emotions as modalities disappear, while its computational burden should not grow with repeated processing of the full input streams. Robustness, label-specific reasoning, and efficiency remain largely separate design objectives.

This work asks whether MMER can retain class-specific evidence under incomplete input within a compact representation whose deep computation is decoupled from the raw sequence length. We propose MERP (Multi-modal Emotion Recognition Perceiver), which follows the latent-query principle of Perceiver IO [18]: available multi-modal evidence is compressed into a fixed latent workspace before iterative reasoning, and emotion-specific evidence is extracted and reinforced across representation levels. This design supports a single predictor for complete and incomplete inputs while bounding the cost of the deeper computation. As illustrated in Figure 1, MERP replaces input-bound late fusion with fixedlatent, class-specific reasoning under compound text–visual loss.

The main contributions are:

• We show that robust MMER should not be framed as recovering an absent modality or assigning a single reliability score to each stream. Our depth-wise diagnostic reveals that repeated full-sequence fusion amplifies missingness and corrupts label evidence unevenly across classes and feature channels; the central design objective is therefore to aggregate available cues into a compact workspace and preserve class-specific evidence throughout deeper reasoning.

• We design MERP around a fixed latent bottleneck, crosslevel class-specific readouts, prototype-guided noisegated enhancement, and confidence-aware class-wise fusion. The same predictor handles complete and incomplete inputs while decoupling deeper reasoning from the raw sequence length.

• MERP achieves the best micro-F1 under 20% testtime modality drop on CMU-MOSEI (0.563) and M<sup>3</sup>ED (0.780), leads LDDU throughout $p _ { \mathrm { m i s s } } \in [ 0 , 0 . 5 ]$ , and avoids its compound-failure collapse, with 5.28 M parameters and 5.28 ms latency on CMU-MOSEI.

## II. RELATED WORK

## A. Multi-Modal Multi-Label Emotion Recognition

Multi-modal multi-label emotion recognition extends categorical emotion recognition by allowing several affective states to coexist in one utterance. Early work explicitly mod eled modality-to-label and label-to-label dependencies to exploit this structure [3]. Transformer-based label-set generation subsequently captured label correlations through sequential decoding [19], while HHMPN represented modalities and emotions as a heterogeneous hierarchy and propagated information between them [4]. These approaches established the importance of structured label reasoning, although sequential generation imposes a decoding order and hierarchical message passing repeatedly operates on modality-specific features. TAI-LOR later separated modality-common and modality-private factors before label-guided decoding [5], and AMP used adversarial masking and perturbation to improve the stability of multi-modal representations [20].

Recent MMER methods place greater emphasis on discriminative label evidence. CARAT combines contrastive learning with feature reconstruction and aggregation [6]; UniVA transfers valence–arousal structure learned from individual modalities [21]; LDDU represents prediction uncertainty through decoupled latent distributions [7]; and HGE models heterogeneous interactions among modalities and labels [22]. Related advances in categorical conversational emotion recognition capture emotion shifts [8] and inter- and intra-speaker dependencies [9], but produce one emotion category per instance. Collectively, these methods improve cross-modal interaction and inter-label discrimination, yet their deep computation remains tied to modality streams, and most do not preserve class-specific evidence under single- and multiple-stream loss. MERP addresses both limitations by conducting repeated reasoning in a fixed latent bank and extracting emotion-specific evidence at successive latent depths.

## B. Robust Learning with Incomplete Modalities

Robust multi-modal learning under missing inputs has developed along two complementary directions [17]. Recoverybased methods estimate the information carried by an absent stream. SMIL combines Bayesian meta-learning with feature reconstruction for severe missingness [23]; MMIN imagines missing modality representations from the observed streams [12]; and TFR-Net reconstructs missing temporal segments while learning task representations [24]. Recent main-conference approaches use duplex diffusion to reduce modality-dependent recovery bias [25] or structured knowledge extracted by large multi-modal models for training-free, out-of-distribution completion [26]. Such recovery can restore a complete input interface, but the estimated features may propagate cross-modal prediction error to the downstream classifier. It also introduces auxiliary generation and reconstruction paths whose cost and reliability depend on the missing pattern.

Representation-based methods instead encourage observed subsets to remain predictive. MissModal aligns complete and incomplete views through geometric and distributional constraints [13], whereas parameter-efficient adaptation modifies a pretrained multi-modal model with few additional parameters [15]. P-RMF constructs a stable proxy modality to reduce the discrepancy between complete and incomplete samples [14], and contrastive masked autoencoding learns modality-agnostic emotion representations [27]. SimMLM dynamically gates modality experts and regularizes predictions so that additional available modalities do not reduce task performance [28]. More recently, heterogeneous graph denoising has been used to suppress noisy propagation in incomplete multi-modal conversations [16]. These methods substantially improve missing-modality robustness, but they predominantly optimize a global representation for single-label sentiment and emotion prediction; they do not explicitly protect the distinct evidence required by multiple correlated labels. Across this family, training commonly involves simulated missing views, auxiliary completion, and missing-pattern adaptation. MERP uses one predictor for complete and incomplete inputs, avoids synthesizing an absent stream, and maintains class-specific evidence within a shared fixed-size latent space.

## C. Latent-Space Multi-Modal Fusion

Multi-modal fusion architectures differ chiefly in where cross-modal interaction occurs. MulT applies directional crossattention between unaligned modality sequences, enabling fine-grained temporal exchange at a cost that grows with pairwise token interactions [11]. Common–private factorization in MISA compresses each modality into invariant and modalityspecific components before fusion [29], while VISTANet combines intermediate and late fusion with adaptive modality weighting for interpretable categorical prediction [10]. These designs retain useful modality structure, but their fusion computation remains coupled to the number and length of the input streams.

Latent-bottleneck architectures decouple repeated processing from raw input length. Perceiver IO first cross-attends from a fixed latent array to arbitrary inputs and then performs deep self-attention only among the latent tokens [18]. The Multimodal Bottleneck Transformer similarly restricts audio–visual exchange to a small collection of fusion tokens [30]. Recent robust-fusion methods further model unimodal aleatoric uncertainty [31] and learn an information-bottleneck latent space shared by complete and incomplete inputs [32]. Nevertheless, these architectures target general perception and single-label sentiment tasks and provide no mechanism ensuring that the compact space retains separable evidence for every cooccurring emotion. MERP connects fixed-size latent computation to label-attentive readouts, cross-level consistency, and class-wise fusion, thereby making the bottleneck directly support robust MMER rather than only global multi-modal compression.

## III. PRELIMINARY STUDY

Before introducing MERP, we examine why a multi-modal fusion model trained with complete observations becomes fragile under modality loss. We conduct a controlled diagnostic study with a conventional dense-fusion Transformer, isolating failure patterns of the fusion process from the components introduced in our framework.

## A. Diagnostic Protocol

Probe model. We train a six-layer Transformer fusion model on the CMU-MOSEI training split using only complete multimodal inputs. Text, visual, and acoustic features are independently projected to a common hidden dimension and concatenated into one sequence, followed by six self-attention layers and a multi-label classification head. The probe excludes latent queries, cross-level reconstruction, class prototypes, featurenoise enhancement, and adaptive branch fusion.

All reported diagnostic metrics are computed on the CMU-MOSEI validation split with the probe parameters frozen; the training split is used only to fit the probe and estimate the clean class centroids. For each validation utterance x, we evaluate the intact input and six corrupted views,

$$
S = \{ \mathrm { d r o p \mathrm { - } T , d r o p \mathrm { - } V , d r o p \mathrm { - } A , T \mathrm { - } o n l y , V \mathrm { - } o n l y , A \mathrm { - } o n l y } \} ,\tag{1}
$$

where drop-m removes modality m, and m-only retains only modality m. Removed streams are replaced by zeros, matching the modality-drop operation in Eq. (8).

Depth-wise representation retention. Let $\mathbf { H } ^ { ( \ell ) }$ and $\widetilde { \mathbf { H } } _ { s } ^ { ( \ell ) }$ denote the sample representations of the intact and corrupted validation sets after fusion layer ℓ under corruption pattern s. Their agreement is measured using linear centered-kernel alignment (CKA) [33]:

$$
R _ { \ell , s } = \mathrm { C K A } \Big ( \mathbf { H } ^ { ( \ell ) } , \widetilde { \mathbf { H } } _ { s } ^ { ( \ell ) } \Big ) .\tag{2}
$$

A smaller $R _ { \ell , s }$ indicates a larger corruption-induced change in the representation.

We also measure the attention assigned to unavailable tokens. Let I<sub>s</sub> contain the positions of the removed modalities, and let $\widetilde { \mathbf { A } } _ { s } ^ { ( \ell ) }$ be the corrupted-view attention matrix averaged over heads and validation samples. The unavailable-token attention ratio is

$$
U _ { \ell , s } = \frac { N _ { \mathrm { q } } ^ { - 1 } \sum _ { i = 1 } ^ { N _ { \mathrm { q } } } \sum _ { j \in \mathcal { T } _ { s } } \widetilde { A } _ { s , i j } ^ { ( \ell ) } } { | \mathcal { T } _ { s } | / T _ { \Sigma } } .\tag{3}
$$

The denominator is the attention share under a uniform distribution. Thus, $U _ { \ell , s } = 1$ corresponds to uniform attention, and values above one indicate disproportionate attention to positions with no modality content.

Class-wise geometry and prediction stability. Using clean training representations from the frozen probe, we compute positive and negative centroids $\mu _ { c } ^ { + }$ and ${ \pmb \mu } _ { c } ^ { - }$ for each emotion $c ,$ with $\mu _ { c } ^ { 1 } = \mu _ { c } ^ { + }$ and $\mu _ { c } ^ { 0 } = \mu _ { c } ^ { - }$ . For a representation h with binary label $y _ { c }$ , the class-separation margin is

$$
m _ { c } ( { \bf h } , y _ { c } ) = \cos ( { \bf h } , \mu _ { c } ^ { y _ { c } } ) - \cos ( { \bf h } , \mu _ { c } ^ { 1 - y _ { c } } ) .\tag{4}
$$

![](images/fa94bb1ee388abeaac21f79aa4384c843f5699d8d126370bf112f68c595fd1d9.jpg)  
Fig. 2. Depth-wise missing-modality diagnosis on the CMU-MOSEI validation split. Each colored strip reports linear CKA across all six fusion layers for one corruption pattern, with white-edged dots marking the measured layer outputs. Color is interpolated only along depth within each categorical row. The aligned marginal plot reports the mean and range of class-wise flip rates. The horizontal divider separates single-modality drops from single-retained inputs.

The relative margin collapse under corruption s is

$$
\Delta M _ { c , s } = 1 0 0 \left( 1 - \frac { \mathbb { E } [ m _ { c } ( \widetilde { \mathbf { h } } _ { s } , y _ { c } ) ] } { \mathbb { E } [ m _ { c } ( \mathbf { h } , y _ { c } ) ] } \right) .\tag{5}
$$

Prediction instability is quantified by the clean-to-corrupted label flip rate

$$
F _ { c , s } = \frac { 1 0 0 } { N } \sum _ { b = 1 } ^ { N } \mathbb { I } [ \widehat { y } _ { b , c } \neq \widetilde { y } _ { b , c , s } ] .\tag{6}
$$

Finally, we form the class-conditioned perturbation

$$
\delta _ { b , c , s } = ( \widetilde { \mathbf { h } } _ { b , s } - \mathbf { h } _ { b } ) \odot \mathbf { w } _ { c } ,\tag{7}
$$

where ${ \bf w } _ { c }$ is the class-c classifier weight, and report the fraction of its energy contained in the largest 10% of feature channels.

## B. Findings and Design Implications

Modality loss is amplified with fusion depth. Figure 2 shows a progressive shift toward lower CKA across all six fusion layers and every corruption pattern. Averaged over the three single-modality drops, CKA decreases from 0.921 at layer 1 to 0.794 at layer 6. When only one modality remains, the decrease is larger, from 0.720 to 0.480. Meanwhile, the unavailable-token attention ratio increases from 1.049 to 1.240 for single-modality drops and from 1.171 to 1.595 for singleretained views. Repeated full-sequence fusion therefore amplifies a local input failure by continually mixing the surviving evidence with positions whose modality content has been removed.

The failure is class-structured and feature-concentrated. The layer-6 CKA values are strongly anticorrelated with the mean label flip rate across the six corruption patterns $( r = - 0 . 9 6 )$ . The aligned ranges in Fig. 2 further show that emotions subjected to the same corruption differ by 10.9–19.4 percentage points in flip rate. Across all 36 emotion–corruption pairs, margin collapse and label instability are strongly associated $( r = 0 . 9 4 )$ . The most affected 10% of classifier-relevant channels contain 46.1–87.8% of the perturbation energy, with a mean of 65.5%. A global modality reliability score therefore cannot represent either the class variation or the concentration of corruption within the feature space.

These findings impose three requirements on robust MMER. Repeated reasoning must operate on a compact representation after the available evidence has been aggregated, limiting depth-wise error amplification. Emotion-specific evidence must remain identifiable as the representation evolves. Feature enhancement and prediction fusion must respond to classand feature-dependent corruption. MERP instantiates these requirements through a fixed latent workspace, cross-level label readouts, prototype-guided feature organization, selective enhancement, and class-wise fusion.

## IV. THE MERP FRAMEWORK

## A. Problem Setup and Notation

Let $\mathcal { M } = \{ t , v , a \}$ denote the text, visual, and acoustic modalities. A mini-batch contains feature sequences $\mathbf { X } _ { m } \in$ $\mathbb { R } ^ { B \times T _ { m } \times d _ { m } }$ for $m \in \mathcal { M }$ and binary emotion labels $\textbf { Y } \in$ $\{ 0 , 1 \} ^ { B \times C }$ , where B is the batch size, $T _ { m }$ is the modalityspecific sequence length, $d _ { m }$ is the input dimension, and $C$ is the number of emotion classes. The experiments use $T _ { m } { = } 6 0$ and $C { = } 6 .$ . An availability indicator $a _ { b , m } \in \{ 0 , 1 \}$ specifies whether modality m is observed for sample b. Missing content is represented by

$$
\bar { \mathbf { X } } _ { b , m } = a _ { b , m } \mathbf { X } _ { b , m } .\tag{8}
$$

All indicators equal one during training. At test time, the indicators are set by the random-drop and fixed-combination protocols in Sections V and V-E. The task is to estimate class probabilities $\mathbf { P } \in [ 0 , 1 ] ^ { B \times C }$ from the available multi-modal evidence.

## B. Architecture Overview

Figure 3 presents the complete computation flow. After aligning and gating the available modalities, MERP maps the joint sequence to low- and high-level fixed-size latent banks. Class-specific readouts produce $\mathbf { F } _ { \mathrm { l o w } }$ and $\mathbf { F } _ { \mathrm { h i g h } }$ , while a query decoder reconstructs the low-level evidence from $\mathbf { Z } _ { \mathrm { h i g h } }$ . The high-level features are then shaped by positive and negative class prototypes, concatenated with the lowlevel and prototype-conditioned features, and enhanced by SNGA. Finally, the high-level, enhanced, and aggregate probability branches are combined by confidence-aware fusion. Figure 4 expands only the internal SNGA computation; complete layer definitions and auxiliary objectives are given in Appendices A–D. Algorithm 1 summarises the corresponding end-to-end training and inference procedure.

![](images/d6ba67600c7cfe24c620d5e4295a659ca036b7c8a0e7f2b8dff30c0b548b45d8.jpg)  
Fig. 3. Overall computation pipeline of MERP. (1) Joint multimodal tokens U are compressed by learnable latent queries ${ \bf Z } ^ { ( 0 ) }$ into $\mathbf { Z } _ { \mathrm { l o w } }$ and refined to $\mathbf { Z } _ { \mathrm { h i g h } } .$ . Class-specific readouts yield $\mathbf { F } _ { \mathrm { l o w } }$ and $\mathbf { F } _ { \mathrm { h i g h } }$ , while a class-query decoder reconstructs $\dot { { \bf F } _ { \mathrm { r e c } } }$ and enforces cross-level consistency. (2) Normalised high-level features are matched to positive and negative class prototypes to form $\mathbf { F } _ { \mathrm { p r o t o } } ;$ the prototype, low-level, and high-level features are concatenated and enhanced by SNGA to obtain ${ \dot { \mathbf { F } } } _ { \mathrm { e n } } . ~ ( 3 )$ High-level, enhanced, and aggregate probabilities are fused class-wise using learned gate preferences calibrated by $r = | p - 0 . 5 |$ |, then thresholded for multi-label prediction.

## C. Multi-Modal Preprocessing and Latent Encoding

For modality m, layer-normalised content features are concatenated with a one-dimensional Fourier position code $\mathbf { P } _ { m }$ and padded with trainable channel embeddings to a common width D. Denoting the resulting tensor by $\bar { \mathbf E } _ { m } ^ { - } \in \mathbb R ^ { B \times T _ { m } \times D } .$ the modality gate is

$$
\mathbf { R } _ { m } = \mathbf { W } _ { m } \bar { \mathbf { E } } _ { m } + \mathbf { b } _ { m } , \qquad \mathbf { U } _ { m } = \sigma ( \mathbf { R } _ { m } ) \odot \mathbf { R } _ { m } .\tag{9}
$$

The gate parameters are modality-specific. The joint input is

$$
{ \bf U } = \mathrm { C o n c a t } _ { m \in \mathcal { M } } ( { \bf U } _ { m } ) \in \mathbb { R } ^ { B \times T _ { \Sigma } \times D } , \qquad T _ { \Sigma } = \sum _ { m \in \mathcal { M } } T _ { m } .\tag{10}
$$

The exact Fourier construction and channel-alignment operation are specified in Appendix A.

MERP maintains a trainable latent bank $\mathbf { Z } ^ { ( 0 ) } \in \mathbb { R } ^ { L \times H }$ which is broadcast across the batch. A cross-attention block maps the joint input to a low-level latent representation,

$$
\mathbf { Z } _ { \mathrm { l o w } } = \mathrm { C r o s s A t t n } ( \mathbf { Z } ^ { ( 0 ) } , \mathbf { U } ) \in \mathbb { R } ^ { B \times L \times H } .\tag{11}
$$

The final latent representation is obtained with $\begin{array} { r l } { N } & { { } = } \end{array}$ $N _ { \mathrm { b l o c k s } } N _ { \mathrm { s e l f } }$ self-attention blocks,

$$
\mathbf { Z } _ { \mathrm { h i g h } } = \mathrm { S e l f A t t n } ^ { N } ( \mathbf { Z } _ { \mathrm { l o w } } ) \in \mathbb { R } ^ { B \times L \times H } .\tag{12}
$$

We use $L { = } 8 0$ , H=256, $N _ { \mathrm { b l o c k s } } { = } 1$ , and $N _ { \mathrm { s e l f } } { = } 6$ . The attention cost is $O ( L T _ { \Sigma } + N L ^ { 2 } )$ : only the first block depends on the input length, while all deeper reasoning operates on L latent tokens. Under modality drop, Eq. (8) removes the designated content stream and the same fixed-size latent computation is applied to the remaining evidence.

## D. Label-Specific Readout and Cross-Level Consistency

Let $\mathbf { r } _ { c } \in \mathbb { R } ^ { H }$ be the trainable readout vector for class c. For representation level $q \ \in \ \{ \mathrm { l o w } , \mathrm { h i g h } \}$ , label attention is computed over the $L$ latent tokens:

$$
\alpha _ { b , c , i } ^ { q } = \frac { \exp ( \mathbf { r } _ { c } ^ { \top } \mathbf { Z } _ { q , b , i } ) } { \sum _ { j = 1 } ^ { L } \exp ( \mathbf { r } _ { c } ^ { \top } \mathbf { Z } _ { q , b , j } ) } , \qquad \mathbf { F } _ { q , b , c } = \sum _ { i = 1 } ^ { L } \alpha _ { b , c , i } ^ { q } \mathbf { Z } _ { q , b , i } .\tag{13}
$$

Thus $\mathbf { F } _ { \mathrm { l o w } } , \mathbf { F } _ { \mathrm { h i g h } } \in \mathbb { R } ^ { B \times C \times H }$ . A shared class-specific classifier produces auxiliary logits

$$
z _ { b , c } ^ { q } = \mathbf { w } _ { c } ^ { \top } \mathbf { F } _ { q , b , c } , \qquad q \in \{ \mathrm { l o w , h i g h } \} .\tag{14}
$$

To preserve class-specific evidence across latent depth, MERP uses C trainable decoder queries $\mathbf { Q } \in \mathbb { R } ^ { C \times H }$ . Crossattention from these queries to $\mathbf { Z } _ { \mathrm { h i g h } }$ yields $\mathbf { F } _ { \mathrm { r e c } } \in \mathbb { R } ^ { B \times C \times H }$ and the cross-level reconstruction loss is

$$
\mathcal { L } _ { \mathrm { r e c } } = \frac { 1 } { B C H } \left\| \mathbf { F } _ { \mathrm { r e c } } - \mathbf { F } _ { \mathrm { l o w } } \right\| _ { F } ^ { 2 } .\tag{15}
$$

This objective aligns decoded high-level semantics with the class-specific evidence extracted immediately after input-tolatent aggregation. Decoder and attention-block details appear in Appendix A-B.

## E. Prototype-Guided Feature Enhancement

The high-level class features are projected and normalised,

$$
\mathbf { h } _ { b , c } = \frac { \phi ( \mathbf { F } _ { \mathrm { h i g h } , b , c } ) } { \| \phi ( \mathbf { F } _ { \mathrm { h i g h } , b , c } ) \| _ { 2 } } \in \mathbb { R } ^ { d _ { z } } .\tag{16}
$$

Each class maintains a positive prototype $\mu _ { c } ^ { + }$ and a negative prototype ${ \pmb \mu } _ { c } ^ { - }$ . Their similarity to $\mathbf { h } _ { b , c }$ defines a two-way distribution

$$
\pi _ { b , c } ^ { s } = \frac { \exp ( \mathbf { h } _ { b , c } ^ { \top } \mu _ { c } ^ { s } ) } { \sum _ { u \in \{ + , - \} } \exp ( \mathbf { h } _ { b , c } ^ { \top } \mu _ { c } ^ { u } ) } , \qquad s \in \{ + , - \} ,\tag{17}
$$

![](images/d83cf0f064833769f34061f619d675d9730ca8641c18be489ee938d65e668e34.jpg)  
Fig. 4. Architecture of the spectral noise-gated enhancement (SNGA) block. The concatenated class features $\mathbf { F } _ { \mathrm { c a t } }$ are reduced to R. The response branch filters white noise ξ in the Fourier domain to obtain a training-only $1 / f$ perturbation ϵ, whereas the gating branch computes γ and ${ \bf G } = { \bar { \bf R } } \odot { \dot { \gamma } }$ . The modulated response is regularised by dropout, added residually to R, and projected to $\mathbf { F } _ { \mathrm { e n } } ;$ at inference, $\epsilon = { \bf 0 }$

from which the prototype-conditioned feature is

$$
\mathbf { F } _ { \mathrm { p r o t o } , b , c } = \psi \left( \sum _ { s \in \{ + , - \} } \pi _ { b , c } ^ { s } \pmb { \mu } _ { c } ^ { s } \right) \in \mathbb { R } ^ { H } .\tag{18}
$$

The prototypes are updated from labelled training samples, and the projected high-level and enhanced features are regularised with supervised contrastive learning [34] and a memory queue. Appendix B gives the update rule, queue construction, and contrastive objective.

The enhancement input concatenates the prototype, lowlevel, and high-level class features:

$$
\mathbf { F } _ { \mathrm { c a t } } = [ \mathbf { F } _ { \mathrm { p r o t o } } ; \mathbf { F } _ { \mathrm { l o w } } ; \mathbf { F } _ { \mathrm { h i g h } } ] \in \mathbb { R } ^ { B \times C \times 3 H } .\tag{19}
$$

A spectral noise-gated feature block maps $\mathbf { F } _ { \mathrm { c a t } }$ to ${ \bf F } _ { \mathrm { e n } } \in \mathbf { \Sigma }$ $\mathbb { R } ^ { B \times C \times H }$ . It injects normalised $1 / f$ noise along the feature axis during training, applies a sigmoid feature gate, and projects the residual representation to width H. No noise is applied at inference. The complete block is defined in Appendix C. The enhanced class logit is

$$
z _ { b , c } ^ { \mathrm { e n } } = \mathbf { w } _ { c } ^ { \top } \mathbf { F } _ { \mathrm { e n } , b , c } .\tag{20}
$$

An aggregate head additionally maps $\operatorname { v e c } ( \mathbf { F } _ { \mathrm { e n } } )$ to ${ \bf z } ^ { \mathrm { a g g } } \in $ $\mathbb { R } ^ { B \times \breve { C } }$ , retaining dependencies among emotion classes.

## F. Class-Wise Fusion and Training Objective

The final prediction combines the aggregate, high-level, and enhanced branches. For $j \in \mathcal { I } = \{ \mathrm { a g g } , \mathrm { h i g h } , \mathrm { e n } \}$ , let $p _ { b , c } ^ { j } =$ $\sigma ( z _ { b , c } ^ { j } )$ and $r _ { b , c } ^ { j } = | p _ { b , c } ^ { j } - 0 . 5 |$ . A class-wise gate receives the three branch probabilities and returns branch preferences

$$
\begin{array} { r } { \mathbf { a } _ { b , c } = \mathrm { s o f t m a x } _ { j } \Big ( G _ { \theta } \big ( [ p _ { b , c } ^ { \mathrm { a g g } } , p _ { b , c } ^ { \mathrm { h i g h } } , p _ { b , c } ^ { \mathrm { e n } } ] \big ) \Big ) . } \end{array}\tag{21}
$$

Algorithm 1 End-to-end training and inference of MERP   
Require: Streams $\{ \mathbf { X } _ { m } \} _ { m \in \mathcal { M } } ,$ , indicators $\left\{ a _ { b , m } \right\}$ , parameters Θ, prototype   
buffers $\mu ^ { \pm }$ , memory queue Q; labels Y during training   
Ensure: Fused probabilities $\mathbf { P } ^ { \mathrm { f i n a l } }$ ; predictions Y<sup>ˆ</sup> at inference   
1: Zero unavailable content; align, gate, and concatenate the streams to form   
U using Eqs. (8)–(10)   
2: $\mathbf { Z } _ { \mathrm { l o w } }  \mathrm { C r o s s A t t n } ( \mathbf { Z } ^ { ( 0 ) } , \mathbf { U } )$   
3: $\mathbf { Z } _ { \mathrm { h i g h } }  \mathrm { S e l f A t t n } ^ { N } ( \mathbf { Z } _ { \mathrm { l o w } } )$   
4: for q ∈ {low, high} do   
5: (F<sub>q</sub>, z<sup>q</sup>) ← LabelReadout ${ \bf \nabla } ( \mathbf { Z } _ { q } )$   
6: end for   
7: $\mathbf { F } _ { \mathrm { r e c } }  \mathrm { C r o s s A t t n } ( \mathbf { Q } , \mathbf { Z } _ { \mathrm { h i g h } } )$   
8: Project $\mathbf { F } _ { \mathrm { h i g h } } ;$ construct F<sub>proto</sub> from $\mu ^ { \pm }$ by soft mixing in training and   
hard selection at inference   
9: $\mathbf { F } _ { \mathrm { e n } } \gets \mathrm { S N G A } ( [ \mathbf { F } _ { \mathrm { p r o t o } } ; \mathbf { F } _ { \mathrm { l o w } } ; \mathbf { F } _ { \mathrm { h i g h } } ] )$ , with noise disabled at inference   
10: Obtain ${ \bf z } ^ { \mathrm { e n } }$ and ${ \bf \ddot { z } } ^ { \mathrm { a } } { \bf \dot { \mathrm { g } } } { \bf \dot { \mathrm { ; } } }$ ; fuse the three probability branches by Eqs. (21)–   
(22)   
11: if training then   
12: Compute $\mathcal { L } _ { \mathrm { r e c } }$ and the total objective L in Eq. (23); update Θ with   
AdamW   
13: Update $\mu ^ { \pm }$ with labelled high-level features and enqueue the high  
level and enhanced views in Q   
14: else   
15: $\bar { Y } _ { b , c } \gets \mathbb { I } [ p _ { b , c } ^ { \mathrm { f i n a l } } \geq \tau ]$   
16: end if   
17: return $\mathbf { P } ^ { \mathrm { f i n a l } }$ and, at inference, Y<sup>ˆ</sup>

Confidence-calibrated fusion weights and the final probability are

$$
\omega _ { b , c } ^ { j } = \mathrm { s o f t m a x } _ { j } ( r _ { b , c } ^ { j } a _ { b , c } ^ { j } ) , \qquad p _ { b , c } ^ { \mathrm { f i n a l } } = \sum _ { j \in \mathcal { I } } \omega _ { b , c } ^ { j } p _ { b , c } ^ { j } .\tag{22}
$$

The gate is trained through binary cross-entropy on $\mathbf { P } ^ { \mathrm { f i n a l } }$ Predictions are $\hat { Y } _ { b , c } = \mathbb { I } [ p _ { b , c } ^ { \mathrm { f i n a l } } \geq \tau ]$ , with $\tau { = } 0 . 2 5$

The training objective is

$$
\begin{array} { r l } & { \mathcal { L } = \mathcal { L } _ { \mathrm { f i n a l } } + \mathcal { L } _ { \mathrm { s u p } } + \lambda _ { \mathrm { r e c } } \mathcal { L } _ { \mathrm { r e c } } + \lambda _ { \mathrm { c l } } \mathcal { L } _ { \mathrm { c l } } } \\ & { ~ + \lambda _ { \mathrm { i n f o } } \mathcal { L } _ { \mathrm { i n f o } } + \lambda _ { \mathrm { s h u f } } \mathcal { L } _ { \mathrm { s h u f } } + \lambda _ { \mathrm { r e g } } \mathcal { L } _ { \mathrm { r e g } } , } \end{array}\tag{23}
$$

where ${ \mathcal { L } } _ { \mathrm { f i n a l } }$ supervises the fused probabilities, $\mathcal { L } _ { \mathrm { s u p } }$ collects logit-level supervision for the low, high, enhanced, and aggregate heads, $\mathcal { L } _ { \mathrm { c l } }$ is the queue-based supervised contrastive loss, ${ \mathcal { L } } _ { \mathrm { i n f c } }$ aligns $\mathbf { F } _ { \mathrm { e n } }$ with $\mathbf { F } _ { \mathrm { l o w } } , \mathcal { L } _ { \mathrm { s h u f } }$ regularises classwise aggregation, and $\mathcal { L } _ { \mathrm { r e g } }$ penalises the enhancement-block weights. Appendix D provides the complete formulas and coefficients.

## V. EXPERIMENTS

## A. Experimental Setup

Datasets. We evaluate on CMU-MOSEI [35] (16,326 / 1,871 / 4,659 train/valid/test utterances; English; six-class multi-label) and M<sup>3</sup>ED [36] (Chinese conversational corpus; 685 / 126 / 179 train/valid/test dialogues with utterance-level annotations mapped to the same six classes). Pre-extracted features follow common practice [6]: GloVe (300d) for text, FACET (35d) for visual, and COVAREP (74d) for audio on CMU-MOSEI; RoBERTa (768d), DenseFace (1024d), and Wav2Vec (342d) for M<sup>3</sup>ED.

Test-time modality-drop protocol. We evaluate every method under a unified test-time modality-drop protocol. With probability $p _ { \mathrm { m i s s } }$ , one uniformly sampled modality of each selected test example is replaced by zeros; the other two modalities remain intact. The corresponding sequence mask is also zeroed for baseline implementations that consume modality masks. MERP receives the zeroed feature tensor defined in Eq. (8). The decisions are drawn once with a fixed RNG (seed 1) and reused across all methods, so every method sees the same corrupted examples and the same per-example modality choice. Training uses complete inputs, $p _ { \mathrm { m i s s } } { = } 0 .$ . Unless stated otherwise, we report $p _ { \mathrm { m i s s } } { = } 0 . 2$ (900 of 4,659 CMU-MOSEI test utterances and 43 of 179 ${ \bf M } ^ { 3 } { \mathrm { E D } }$ test dialogues are corrupted).

Implementation details. MERP uses the Perceiver-IO fusion backbone described in Section IV-C, with $L { = } 8 0$ latents, $H { = } 2 5 6 , N _ { \mathrm { b l o c k s } } { = } 1 , N _ { \mathrm { s e l f } } { = } 6 .$ , one cross-attention head, and eight self-attention heads. We optimise with AdamW for 35 epochs at a learning rate of $3 . 7 \times 1 0 ^ { - 5 }$ with cosine decay (warmup ratio 0.15), batch size 64, dropout 0.18, and binary classification threshold $\tau { = } 0 . 2 5$ . Appendix D reports all objective coefficients. All baselines are retrained from their official code on identical features, splits, and preprocessing. The 2025 state-of-the-art LDDU [7] does not release pre-trained weights; we therefore retrain it under our hardware budget (batch size 16 and queue size 512 on CMU-MOSEI; batch size 8 and queue size 256 on M<sup>3</sup>ED). GPU memory is measured during single-utterance inference on one NVIDIA A100; per-sample latency is averaged over 200 forward passes after a 20-step warm-up.

## B. Main Results under Modality Drop

Table I reports all metrics under $p _ { \mathrm { m i s s } } { = } 0 . 2$ on CMU-MOSEI and M<sup>3</sup>ED. All baselines are retrained from their official releases and evaluated with the same per-sample missing schedule. Parameter counts marked by <sup>∗</sup> for the Transformerfusion baselines (HHMPN, TAILOR, CARAT, MISA, MulT, and AMP) are taken from prior measurements using their official implementations [6]; GPU memory for all baselines is measured by us under the same inference setup as MERP and LDDU.

Observations: Three observations emerge. First, several Transformer-fusion baselines degrade sharply on CMU-MOSEI: HHMPN reaches 0.479 micro-F1 and TAILOR reaches 0.446, with TAILOR’s recall falling to 0.374. Second, MERP attains the best micro-F1 on both datasets (0.563 on CMU-MOSEI and 0.780 on $ { \mathbf { M } } ^ { 3 }  { \mathrm { E D } } )$ , surpassing the 2024 contrastive SOTA CARAT (0.510 / 0.744) and the 2025 probabilistic SOTA LDDU (0.559 / 0.758) while using 2 048 MiB of inference memory. The asymmetric $F _ { \beta }$ scores further show that this advantage is not tied to one precision–recall tradeoff: MERP ranks first on both $F _ { 0 . 5 }$ and $F _ { 2 }$ on CMU-MOSEI, and first on $F _ { 2 }$ and second on $F _ { 0 . 5 }$ on M<sup>3</sup>ED. Third, among the methods approaching MERP’s accuracy, the Transformerfusion baselines use approximately $3 \times \mathrm { ~ t o ~ } 7 \times$ more memory.

Clean-to-missing degradation: Figure 5(a) summarises the absolute micro-F1 decrease from complete input to the $p _ { \mathrm { m i s s } } { = } 0 . 2$ setting. To give the baselines their strongest clean reference, the complete-input values for CARAT [6] and LDDU [7] are taken from their published full-budget reports; their missing-input values come from our reproduction under matched preprocessing and the common missing schedule.

MERP uses the single-seed checkpoint reported in Table I. CARAT and LDDU lose 4.4–7.1 and 2.8–4.9 F1 points, respectively, whereas MERP loses 2.2 points on CMU-MOSEI (0.585 to 0.563) and 0.2 points on ${ \bf M } ^ { 3 } { \bf E } { \bf D }$ . MERP therefore becomes the strongest model in the deployment-relevant missing regime, showing that complete-input performance alone does not predict robustness.

## C. Inference Cost

Table II reports per-sample inference latency and parameter counts for MERP and LDDU—the only two methods with directly comparable architectural scale—on a single A100 GPU. MERP is 27% faster than LDDU on CMU-MOSEI and 28% faster on M<sup>3</sup>ED; on the smaller-feature CMU-MOSEI benchmark MERP also uses 11% fewer parameters. On ${ \bf M } ^ { 3 } { \bf E } { \bf D }$ , MERP carries more parameters because the higherdimensional pre-trained features enlarge the input projections, while the fusion backbone remains unchanged. Both methods use 2,048 MiB of inference memory, whereas the Transformerfusion baselines in Table I use approximately 3–7× more. This gap is dominated by activation cost: the latent bottleneck reduces the self-attention footprint from $O ( T _ { \mathrm { t o t a l } } ^ { 2 } )$ to $O ( L T _ { \mathrm { t o t a l } } + L ^ { 2 } )$ with $L \ll T _ { \mathrm { t o t a l } }$

## D. Robustness across Missing Rates on CMU-MOSEI

We sweep $p _ { \mathrm { m i s s } } \in \{ 0 . 0 , 0 . 1 , \ldots , 0 . 5 \}$ on CMU-MOSEI, the larger of the two test sets (4,659 utterances), and compare MERP with LDDU under identical per-sample missing schedules. We restrict the sweep to CMU-MOSEI because $ { \mathbf { M } } ^ { 3 }  { \mathrm { E D } ^ { \prime } \mathrm { s } }$ test split contains only 179 dialogues, which is too small for a stable rate-wise comparison across six operating points. Figure 5(b) reports the sweep, and panel (c) summarises the fixed-combination experiment of Section V-E. The LDDU sweep uses one reproduced checkpoint throughout for internal consistency; its $p _ { \mathrm { m i s s } } { = } 0$ score of 0.571 therefore differs from the published full-budget value used in panel (a).

MERP outperforms LDDU at every evaluated missing rate. The micro-F1 gap is 0.014 at $p _ { \mathrm { m i s s } } { = } 0$ , narrows to 0.002 at 0.1, and then increases to 0.013 at 0.5. MERP loses 2.4 absolute F1 points across the sweep, compared with LDDU’s 2.3-point drop.

## E. Fixed-Combo Modality Drop

The random per-sample drop of Section V-D mixes complete and single-missing examples. To probe more severe modality loss, we additionally evaluate every CMU-MOSEI test example with a fixed subset of modalities zeroed. We restrict this analysis to LDDU because no other published baseline reports fixed-combination results, while CARAT’s clean-to-missing degradation is already shown in Figure 5(a). Figure 5(c) reports the six non-trivial combinations together with the intact reference.

LDDU collapses under text+visual drop on CMU-MOSEI: LDDU’s micro-F1 falls to 0.022 when both the text and visual streams are zeroed, while MERP retains 0.473— a 0.451 absolute gap. LDDU independently encodes each modality and merges the resulting distribution parameters at a late fusion stage; the audio-only input therefore differs substantially from its complete-input training condition. MERP applies the same input-to-latent mapping to the zeroed multimodal tensor and performs all deeper reasoning on the fixed latent bank. With only audio content available, MERP retains

TABLE I  
PERFORMANCE UNDER 20% TEST-TIME RANDOM MODALITY DROP $( p _ { \mathrm { { M I S S } } } { = } 0 . 2 , \mathrm { { S E E D } { = } 1 ) }$ ON CMU-MOSEI AND M<sup>3</sup>ED. ACC IS EXAMPLE-BASED JACCARD ACCURACY; P AND R ARE MICRO-AVERAGED OVER ALL SAMPLE–LABEL DECISIONS, AND $F _ { \beta } = ( 1 + \beta ^ { 2 } ) P R / ( \beta ^ { 2 } P + R )$ (MICRO-F1 USES β=1). WITHIN EACH DATASET, THE BEST RESULT IN EACH COLUMN IS IN BOLD AND THE SECOND-BEST IS UNDERLINED; HIGHER IS BETTER EXCEPT FOR PARAMS AND GPU MEM. MERP ROWS ARE HIGHLIGHTED IN VERY LIGHT PURPLE.
<table><tr><td>Dataset</td><td>Method</td><td>Acc</td><td>P</td><td>R</td><td> $\mathbf { F _ { 0 . 5 } }$ </td><td> $\mathbf { F _ { 2 } }$ </td><td>micro-F1</td><td>Params</td><td>GPU Mem.</td></tr><tr><td rowspan="9">CMU- MOSEI</td><td>HHMPN [4]</td><td>0.380</td><td>0.554</td><td>0.422</td><td>0.521</td><td>0.443</td><td>0.479</td><td> $3 2 . 5 \mathrm { M ^ { * } }$ </td><td>&gt;15 360 MiB</td></tr><tr><td>TAILOR [5]</td><td>0.333</td><td>0.553</td><td>0.374</td><td>0.505</td><td>0.400</td><td>0.446</td><td> $1 4 . 6 \mathrm { M ^ { * } }$ </td><td>6518MiB</td></tr><tr><td>CARAT [6]</td><td>0.390</td><td>0.520</td><td>0.500</td><td>0.516</td><td>0.504</td><td>0.510</td><td>6.8M*</td><td>11 236 MiB</td></tr><tr><td>MISA [29]</td><td>0.445</td><td>0.588</td><td>0.475</td><td>0.561</td><td>0.494</td><td>0.525</td><td>1.6M*</td><td>376MiB</td></tr><tr><td>MulT [11]</td><td>0.470</td><td>0.610</td><td>0.511</td><td>0.587</td><td>0.528</td><td>0.556</td><td>4.0M*</td><td>1336MiB</td></tr><tr><td>AMP [20]</td><td>0.475</td><td>0.607</td><td>0.509</td><td>0.584</td><td>0.526</td><td>0.554</td><td>11.0M*</td><td>11 722 MiB</td></tr><tr><td>LDDU [7]</td><td>0.470</td><td>0.624</td><td>0.507</td><td>0.596</td><td>0.527</td><td>0.559</td><td>5.97M</td><td>2 048 MiB</td></tr><tr><td>MERP (ours)</td><td>0.469</td><td>0.626</td><td>0.512</td><td>0.599</td><td>0.531</td><td>0.563</td><td>5.28 M</td><td>2 048 MiB</td></tr><tr><td>HHMPN [4]</td><td>0.586</td><td>0.743</td><td>0.721</td><td>0.738</td><td>0.725</td><td>0.732</td><td>33.0M*</td><td>&gt;15 360 MiB</td></tr><tr><td rowspan="8">MED</td><td>MISA [29]</td><td>0.623</td><td>0.727</td><td>0.760</td><td>0.733</td><td>0.753</td><td>0.743</td><td>1.7M*</td><td>1 882MiB</td></tr><tr><td>TAILOR [5]</td><td>0.652</td><td>0.769</td><td>0.767</td><td>0.769</td><td>0.767</td><td>0.768</td><td>14.7M*</td><td>6 604 MiB</td></tr><tr><td>MulT [11]</td><td>0.651</td><td>0.777</td><td>0.762</td><td>0.774</td><td>0.765</td><td>0.769</td><td>4.1M*</td><td>1 408 MiB</td></tr><tr><td>AMP [20]</td><td>0.652</td><td>0.790</td><td>0.761</td><td>0.784</td><td>0.767</td><td>0.775</td><td>11.5M*</td><td>11 664 MiB</td></tr><tr><td>CARAT [6]</td><td>0.489</td><td>0.780</td><td>0.712</td><td>0.765</td><td>0.725</td><td>0.744</td><td> ${ 6 . 9 } \mathbf { M } ^ { * }$ </td><td>11 236 MiB</td></tr><tr><td>LDDU [7]</td><td>0.614</td><td>0.887</td><td>0.662</td><td>0.831</td><td>0.697</td><td>0.758</td><td>6.41 M</td><td>2 048 MiB</td></tr><tr><td>MERP (ours)</td><td>0.659</td><td>0.796</td><td>0.765</td><td>0.790</td><td>0.771</td><td>0.780</td><td>10.92M</td><td>2 048 MiB</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/c566f792288e83ab0c020a01b4c645cd6bcf46df54ae2a6b2e91233d03b430cb.jpg)

![](images/1c639da91c15e3cf8869051ac218a2c2593d08a35400a002565d59600540adc5.jpg)

![](images/2fabed120d04e64d6d07ff74437a3b4b8ba485613455ca8329d045f8e4aafdae.jpg)  
Fig. 5. MERP remains robust across complementary missing-modality evaluations. (a) Absolute micro-F1 decrease from complete input to 20% random modality drop on CMU-MOSEI and M<sup>3</sup>ED; lower is better. (b) Micro-F1 across increasing random-drop rates, with endpoint labels giving the absolute change from $p _ { \mathrm { m i s s } } { = } 0$ to 0.5. (c) Exact micro-F1 under fixed missing-modality combinations applied to every CMU-MOSEI test sample. The shaded regions highlight MERP in (a) and LDDU’s text–visual collapse in (c).

TABLE II  
INFERENCE COST ON A SINGLE A100 GPU. LATENCY IS AVERAGED OVER200 SINGLE-SAMPLE FORWARD PASSES AFTER A 20-STEP WARM-UP. THEBEST RESULT IN EACH COLUMN IS IN BOLD; LOWER IS BETTER, AND THEMERP ROW IS HIGHLIGHTED IN VERY LIGHT PURPLE.
<table><tr><td rowspan="2">Method</td><td colspan="2">CMU-MOSEI</td><td colspan="2">M3ED</td></tr><tr><td>Params</td><td>Latency</td><td>Params</td><td>Latency</td></tr><tr><td>LDDU [7]</td><td>5.97M</td><td>7.26 ms</td><td>6.41 M</td><td>7.29 ms</td></tr><tr><td>MERP (ours)</td><td>5.28M</td><td>5.28 ms</td><td>10.92M</td><td>5.21 ms</td></tr></table>

80.9% of its complete-input F1, whereas LDDU retains less than 4%.

Graceful degradation across all combos: Beyond the catastrophic case, MERP outperforms LDDU on six of the seven combos (and is within 0.005 on the seventh). The single-modality drop costs MERP 2.6–8.6 absolute points; the double-modality drop costs 5.1–11.2 points but never causes a collapse, in contrast to LDDU.

Qualitative case study: Figure 6 presents six representative CMU-MOSEI examples spanning complete input, singlemodality removal, and single-retained views. With complete input (Case 1), MERP recovers all three emotions, whereas both baselines omit Fear; with the visual stream removed (Case 2), it retains all four labels, while LDDU and HHMPN lose two and three, respectively. In the audio-missing example (Case 3), MERP and LDDU recover both Happy and Surprise, whereas HHMPN misses Surprise. Under the more challenging visual-only and audio-only settings (Cases 4–5), MERP

Case 1 — All modalities available  
![](images/fa0f80cfd377bef625e17e97461a0523bb44279ae0e55b65734cf5bef0ab5670.jpg)

![](images/0cf9df0f8f0099592c8ccd2f5949c174d9dc0180039fb701a3a1a80864023f63.jpg)

![](images/6725a5e92aa86ba381740bdf57bba98edf1fa2e3f0795e84b498706c63731c7e.jpg)  
Case 3— Audio modality removed

![](images/9b35a9d71bd37b7795b3d9b036b92f250787597aa6a59685b836402eebfba994.jpg)

![](images/c8d36bea942980ec64686ebfaac054338aa905dcad96a008d62f05ef6fcf2153.jpg)

![](images/7114437a2326e91c4562fed2308b19fc6b7aae90b4a90c0683de3caf0bb995af.jpg)  
Fig. 6. MERP more consistently preserves multi-label predictions across diverse modality-loss patterns. Six CMU-MOSEI examples cover complete input, visual or audio removal, and visual-only, audio-only, or text-only views. We compare ground truth (GT) with LDDU, HHMPN, and MERP; gray hatched boxes denote missed labels, while red-hatched labels denote spurious predictions. MERP most closely matches the ground-truth label sets across the six cases, including the severe audio-only condition.

![](images/2eed33b20bbf31a786bec4ff379a9545488f6bcee66fe37b0a479fa64c108141.jpg)

(b) $N _ { \mathsf { s e l f } }$  
![](images/d16c6da1a5bea320c148a6fc0616242bc66216aba5d832c6d9fe03b3a79a16ee.jpg)

(c) N<sub>blocks</sub>  
![](images/43984260180ab44108afe29930107704da0d10a4318fe64c945e40ef5c8edc04.jpg)

![](images/84363f344a4a9d2f8c83e05c3c21ddf00b0616495d56490ae2aa9014d9b04496.jpg)  
Fig. 7. MERP is stable across structural settings, while reconstruction and enhancement objectives have the largest ablation effects. (a)–(c) Structural ablations with two-seed micro-F1 means and sample-standard-deviation whiskers; purple denotes the default. (d) Loss ablations, with micro-F1 recomputed from micro-P/R. Bar labels are scores and the right column gives changes from the dashed full-model reference; plum marks the largest drop. All ablations use CMU-MOSEI with $p _ { \mathrm { m i s s } } { = } 0$

## F. Ablation Study

exactly matches the ground-truth sets, whereas the baselines omit valid labels and HHMPN additionally introduces Surprise in the audio-only case. In the text-only example (Case 6), MERP retains Sad and Surprise but misses Fear, indicating graceful rather than perfect recovery under extreme information loss. These cases suggest that MERP preserves classspecific decisions from surviving evidence instead of simply suppressing predictions, complementing the aggregate trends in Table I and Figure 5(c).

We assess MERP’s architecture and objectives on CMU-MOSEI with $p _ { \mathrm { m i s s } } { = } 0$ . Figure 7(a)–(c) shows that varying L, $N _ { \mathrm { s e l f } }$ , or $N _ { \mathrm { b l o c k s } }$ changes the two-seed mean by at most 0.31% relative to the default (0.5850), with means ranging from 0.5832 to 0.5858 (a span of 0.0026). In panel (d), removing $\mathcal { L } _ { \mathrm { r e c } }$ causes the largest single-loss drop, from 0.585 to 0.550 (−6.0%); removing ${ \mathcal { L } } _ { \mathrm { { i n f o } } } ,$ ${ \mathcal { L } } _ { \mathrm { e n } }$ , or $\mathcal { L } _ { \mathrm { a g g } }$ gives 0.563 (about −3.8%), while removing $\mathcal { L } _ { \mathrm { c l } }$ gives 0.584 (−0.2%). Removing ${ \mathcal { L } } _ { \mathrm { { i n f o } } }$ and ${ \mathcal { L } } _ { \mathrm { e n } }$ jointly produces the largest drop, to 0.414 (−29.2%). Loss symbols follow Eqs. (23) and (38).

## VI. CONCLUSION

This work addresses robust multi-modal multi-label emotion recognition when parts of the multi-modal evidence are unavailable. MERP compresses the cues that remain into a fixed latent workspace and preserves emotion-specific evidence through cross-level readout and reconstruction, prototypeguided noise-gated enhancement, and confidence-aware classwise fusion. Experiments across two benchmarks and complementary modality-drop settings show that this design achieves leading recognition performance while degrading gracefully under increasingly severe and compound modality loss. These results suggest a broader lesson for robust multi-modal learning: protecting task-relevant evidence after compact aggregation can be more effective than reconstructing missing streams or assigning a single reliability estimate to each modality. Building on this principle, future work can investigate adaptive latent workspaces that retain bounded deep reasoning while supporting evolving label spaces and longer multi-modal contexts.

## REFERENCES

[1] S. Poria, E. Cambria, R. Bajpai, and A. Hussain, “A review of affective computing: From unimodal analysis to multimodal fusion,” Information Fusion, vol. 37, pp. 98–125, 2017.

[2] P. K. Atrey, M. A. Hossain, A. E. Saddik, and M. S. Kankanhalli, “Multimodal fusion for multimedia analysis: A survey,” Multimedia Systems, vol. 16, no. 6, pp. 345–379, 2010.

[3] D. Zhang, X. Ju, J. Li, S. Li, Q. Zhu, and G. Zhou, “Multi-modal multi-label emotion detection with modality and label dependence,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics, 2020, pp. 3584–3593.

[4] D. Zhang, X. Ju, W. Zhang, J. Li, S. Li, Q. Zhu, and G. Zhou, “Multimodal multi-label emotion recognition with heterogeneous hierarchical message passing,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 16, 2021, pp. 14 338–14 346.

[5] Y. Zhang, M. Chen, J. Shen, and C. Wang, “TAILOR: Versatile multimodal learning for multi-label emotion recognition,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, no. 8, 2022, pp. 9100–9108.

[6] C. Peng, K. Chen, L. Shou, and G. Chen, “CARAT: Contrastive feature reconstruction and aggregation for multi-modal multi-label emotion recognition,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 13, 2024, pp. 14 581–14 589.

[7] J. Huang, J. Zhong, Q. Lei, J. Gao, Y. Yang, S. Wang, P. Li, and K. Wei, “Latent distribution decouple for uncertain-aware multimodal multi-label emotion recognition,” in Findings of the Association for Computational Linguistics: ACL 2025. Association for Computational Linguistics, 2025, pp. 24 123–24 138.

[8] J. Li, X. Wang, Y. Liu, and Z. Zeng, “CFN-ESA: A cross-modal fusion network with emotion-shift awareness for dialogue emotion recognition,” IEEE Transactions on Affective Computing, vol. 15, no. 4, pp. 1919– 1933, 2024.

[9] Y. Wang, W. Zhang, K. Liu, W. Wu, F. Hu, H. Yu, and G. Wang, “Dynamic emotion-dependent network with relational subgraph interaction for multimodal emotion recognition,” IEEE Transactions on Affective Computing, vol. 16, no. 2, pp. 712–725, 2025.

[10] P. Kumar, S. Malik, B. Raman, and X. Li, “VISTANet: VIsual Spoken Textual Additive Net for interpretable multimodal emotion recognition,” IEEE Transactions on Affective Computing, vol. 16, no. 4, pp. 2881– 2893, 2025.

[11] Y.-H. H. Tsai, S. Bai, P. P. Liang, J. Z. Kolter, L.-P. Morency, and R. Salakhutdinov, “Multimodal transformer for unaligned multimodal language sequences,” in Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics, 2019, pp. 6558–6569.

[12] J. Zhao, R. Li, and Q. Jin, “Missing modality imagination network for emotion recognition with uncertain missing modalities,” in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). Association for Computational Linguistics, 2021, pp. 2608–2618.

[13] R. Lin and H. Hu, “MissModal: Increasing robustness to missing modality in multimodal sentiment analysis,” Transactions of the Association for Computational Linguistics, vol. 11, pp. 1686–1702, 2023.

[14] A. Zhu, M. Hu, X. Wang, J. Yang, Y. Tang, and N. An, “Proxydriven robust multimodal sentiment analysis with incomplete data,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 2025, pp. 22 123–22 138.

[15] M. K. Reza, A. Prater-Bennette, and M. S. Asif, “Robust multimodal learning with missing modalities via parameter-efficient adaptation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 2, pp. 742–754, 2025.

[16] K. Zeng, F. Chen, J. Shao, and H. T. Shen, “Heterogeneous graph denoising network for incomplete multimodal emotion recognition in conversation,” IEEE Transactions on Affective Computing, pp. 1–14, 2026, early access.

[17] R. Wu, H. Wang, H.-T. Chen, and G. Carneiro, “Deep multimodal learning with missing modality: A survey,” Transactions on Machine Learning Research, 2026. [Online]. Available: https://openreview.net/ forum?id=tc7RFcx4hT

[18] A. Jaegle, S. Borgeaud, J.-B. Alayrac, C. Doersch, C. Ionescu, D. Ding, S. Koppula, D. Zoran, A. Brock, E. Shelhamer, O. J. Henaff, M. M.´ Botvinick, A. Zisserman, O. Vinyals, and J. Carreira, “Perceiver IO: A general architecture for structured inputs and outputs,” in International Conference on Learning Representations, 2022. [Online]. Available: https://openreview.net/forum?id=fILj7WpI-g

[19] X. Ju, D. Zhang, J. Li, and G. Zhou, “Transformer-based label set generation for multi-modal multi-label emotion detection,” in Proceedings of the 28th ACM International Conference on Multimedia. Association for Computing Machinery, 2020, pp. 512–520.

[20] S. Ge, Z. Jiang, Z. Cheng, C. Wang, Y. Yin, and Q. Gu, “Learning robust multi-modal representation for multi-label emotion recognition via adversarial masking and perturbation,” in Proceedings of the ACM Web Conference 2023. Association for Computing Machinery, 2023, pp. 1510–1518.

[21] W. Zheng, J. Yu, and R. Xia, “A unimodal valence-arousal driven contrastive learning framework for multimodal multi-label emotion recognition,” in Proceedings of the 32nd ACM International Conference on Multimedia. Association for Computing Machinery, 2024, pp. 622– 631.

[22] D. Hu, X. Jiang, Z. Sun, F. Shen, and X. Xu, “Heterogeneous graph embedding for multimodal multi-label emotion recognition,” in Proceedings of the 2025 International Conference on Multimedia Retrieval. Association for Computing Machinery, 2025, pp. 460–468.

[23] M. Ma, J. Ren, L. Zhao, S. Tulyakov, C. Wu, and X. Peng, “SMIL: Multimodal learning with severely missing modality,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 3, 2021, pp. 2302–2310.

[24] Z. Yuan, W. Li, H. Xu, and W. Yu, “Transformer-based feature reconstruction network for robust multimodal sentiment analysis,” in Proceedings of the 29th ACM International Conference on Multimedia. Association for Computing Machinery, 2021, pp. 4400–4407.

[25] R. Dai, C. Li, Y. Yan, L. Mo, K. Qin, and T. He, “Unbiased missingmodality multimodal learning,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 24 507–24 517.

[26] G. Ke, S. He, X. Wang, B. Wang, G. Chao, Y. Zhang, Y. Xie, and H. Su, “Knowledge bridger: Towards training-free missing modality completion,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 25 864–25 873.

[27] G. Chochlakis, T. Iqbal, W. H. Kang, and Z. Huang, “Modalityagnostic multimodal emotion recognition using a contrastive masked autoencoder,” in Proceedings of Interspeech 2025, 2025, pp. 3005–3009.

[28] S. Li, C. Chen, and J. Han, “SimMLM: A simple framework for multimodal learning with missing modality,” in Proceedings ofthe IEEE/CVF International Conference on Computer Vision, 2025, pp. 24 068–24 077.

[29] D. Hazarika, R. Zimmermann, and S. Poria, “MISA: Modality-invariant and -specific representations for multimodal sentiment analysis,” in Proceedings of the 28th ACM International Conference on Multimedia. Association for Computing Machinery, 2020, pp. 1122–1131.

[30] A. Nagrani, S. Yang, A. Arnab, A. Jansen, C. Schmid, and C. Sun, “Attention bottlenecks for multimodal fusion,” in Advances

in Neural Information Processing Systems, vol. 34, 2021, pp. 14 200– 14 213. [Online]. Available: https://proceedings.neurips.cc/paper/2021/ hash/76ba9f564ebbc35b1014ac498fafadd0-Abstract.html

[31] Z. Gao, X. Jiang, X. Xu, F. Shen, Y. Li, and H. T. Shen, “Embracing unimodal aleatoric uncertainty for robust multimodal fusion,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 26 866–26 875.

[32] R. Lin, Q. He, S. Mai, Y. Zeng, A. Xiong, L. Huang, Y.-P. Tan, and H. Hu, “CyIN: Cyclic informative latent space for bridging complete and incomplete multimodal learning,” in Advances in Neural Information Processing Systems, vol. 38, 2025.

[33] S. Kornblith, M. Norouzi, H. Lee, and G. Hinton, “Similarity of neural network representations revisited,” in Proceedings of the 36th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 97. PMLR, 2019, pp. 3519–3529. [Online]. Available: https://proceedings.mlr.press/v97/kornblith19a.html

[34] P. Khosla, P. Teterwak, C. Wang, A. Sarna, Y. Tian, P. Isola, A. Maschinot, C. Liu, and D. Krishnan, “Supervised contrastive learning,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 18 661–18 673.

[35] A. B. Zadeh, P. P. Liang, S. Poria, E. Cambria, and L.-P. Morency, “Multimodal language analysis in the wild: CMU-MOSEI dataset and interpretable dynamic fusion graph,” in Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 2018, pp. 2236–2246.

[36] J. Zhao, T. Zhang, J. Hu, Y. Liu, Q. Jin, X. Wang, and H. Li, “M3ED: Multi-modal multi-scene multi-label emotional dialogue database,” in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 2022, pp. 5699–5710.

[37] J. R. Pierce, An Introduction to Information Theory: Symbols, Signals and Noise. New York, NY, USA: Dover Publications, 1980.

## APPENDIX A PREPROCESSING AND ATTENTION DETAILS

## A. Fourier Positions and Channel Alignment

For a sequence of length $T _ { m } .$ , the scalar position $x _ { i }$ is linearly spaced on [−1, 1]. With K=32 frequency bands $f _ { k }$ linearly spaced between 1 and $T _ { m } / 2$ , the position vector is

$$
\mathbf { p } _ { i } = \left[ x _ { i } , \left\{ \sin ( \pi f _ { k } x _ { i } ) \right\} _ { k = 1 } ^ { K } , \left\{ \cos ( \pi f _ { k } x _ { i } ) \right\} _ { k = 1 } ^ { K } \right] \in \mathbb { R } ^ { 2 K + 1 } .\tag{24}
$$

The content-position representation is

$$
{ \bf E } _ { m } = [ { \bf L } { \bf N } ( \bar { \bf X } _ { m } ) ; { \bf P } _ { m } ] \in \mathbb { R } ^ { B \times T _ { m } \times ( d _ { m } + 2 K + 1 ) } .\tag{25}
$$

Because the input dimensions differ across modalities, trainable padding embeddings extend each $\mathbf { E } _ { m }$ to the common width $D = \mathrm { m a x } _ { m \in \mathcal { M } } ( d _ { m } + 2 K + 1 )$ . The modality-specific linear gate in Eq. (9) is applied after this channel alignment. Missingness zeros the content term in Eq. (25); the deterministic position code is retained.

## B. Attention Blocks

For one attention head with query width $d _ { h }$ , scaled dotproduct attention is

$$
\mathrm { A t t n } ( \mathbf { Q } , \mathbf { K } , \mathbf { V } ) = \mathrm { s o f t m a x } \left( \frac { ( \mathbf { Q } \mathbf { W } _ { Q } ) ( \mathbf { K } \mathbf { W } _ { K } ) ^ { \top } } { \sqrt { d _ { h } } } \right) ( \mathbf { V } \mathbf { W } _ { V } ) .\tag{26}
$$

Multi-head outputs are concatenated and linearly projected. Each cross-attention and self-attention block uses pre-layer normalisation, a residual attention update, and a residual feedforward update:

$$
\begin{array} { c } { { { \bf H } ^ { \prime } = { \bf H } + \mathrm { M H A } ( \mathrm { L N } ( { \bf H } ) , \mathrm { L N } ( { \bf S } ) , \mathrm { L N } ( { \bf S } ) ) , } } \\ { { \mathrm { B l o c k } ( { \bf H } , { \bf S } ) = { \bf H } ^ { \prime } + \mathrm { M L P } ( \mathrm { L N } ( { \bf H } ^ { \prime } ) ) . \qquad } } \end{array}\tag{27}
$$

For encoder cross-attention, $\mathbf { H } = \mathbf { Z } ^ { ( 0 ) }$ and $\mathrm { ~ \bf ~ S ~ } = \mathrm { ~ \bf ~ U ~ }$ . Selfattention sets $\mathbf { S } = \mathbf { H }$ . The encoder uses one cross-attention head and eight self-attention heads. The decoder applies Eq. (27) with C broadcast trainable queries as H and $\mathbf { Z } _ { \mathrm { h i g h } }$ as S, followed by a linear projection to width H.

## APPENDIX B

PROTOTYPE AND CONTRASTIVE LEARNING DETAILS

## A. Dynamic Class Prototypes

The model maintains 2C prototype buffers $\{ \mu _ { c } ^ { + } , \mu _ { c } ^ { - } \} _ { c = 1 } ^ { C }$ in the $d _ { z }$ -dimensional projection space. For a training pair $( b , c )$ the selected sign is $s _ { b , c } = +$ when $Y _ { b , c } = 1$ and $s _ { b , c } = -$ when $Y _ { b , c } = 0$ . Its retention coefficient is

$$
m _ { b , c } = m _ { 0 } \frac { 1 + c o s ( \mathbf { h } _ { b , c } , \pmb { \mu } _ { c } ^ { s _ { b , c } } ) } { 2 } , \qquad m _ { 0 } = 0 . 9 9 ,\tag{28}
$$

and the selected prototype is updated sequentially within the mini-batch:

$$
\pmb { \mu } _ { c } ^ { s _ { b , c } } \gets m _ { b , c } \pmb { \mu } _ { c } ^ { s _ { b , c } } + ( 1 - m _ { b , c } ) \mathbf { h } _ { b , c } .\tag{29}
$$

The opposite-sign prototype is unchanged for that pair. Equation (17) uses both class prototypes during training. At inference, the sign with the larger prototype score is selected before the inverse projection $\psi .$

## B. Queue-Based Supervised Contrastive Loss

Two class-conditioned views are used for contrastive learning: the projected high-level feature $\phi ( { \bf F } _ { \mathrm { h i g h } } )$ and the projected enhanced feature $\phi ( \mathbf { F } _ { \mathrm { e n } } )$ . After $\ell _ { 2 }$ normalisation, each view receives the pseudo-label

$$
\ell _ { b , c } = \left\{ { c , \atop C + c , } \right. \ Y _ { b , c } = 1 ,\tag{30}
$$

A first-in–first-out queue of 768 feature–label pairs supplies additional comparisons. Let A denote current anchors together with queued features, $\mathcal { P } ( i ) = \{ p \neq i : \ell _ { p } = \ell _ { i } \}$ , and $\tau _ { \mathrm { c l } } =$ 0.07. The supervised contrastive loss is

$$
\mathcal { L } _ { \mathrm { c l } } = - \frac { 1 } { \vert A \vert } \sum _ { i \in A } \frac { 1 } { \vert \mathcal { P } ( i ) \vert } \sum _ { p \in \mathcal { P } ( i ) } \log \frac { \exp ( \mathbf { h } _ { i } ^ { \top } \mathbf { h } _ { p } / \tau _ { \mathrm { c l } } ) } { \sum _ { a \in A \backslash \{ i \} } \exp ( \mathbf { h } _ { i } ^ { \top } \mathbf { h } _ { a } / \tau _ { \mathrm { c l } } ) } .\tag{31}
$$

Anchors without a positive partner contribute zero to the batch mean.

## APPENDIX C SPECTRAL NOISE-GATED FEATURE BLOCK

Let $D _ { e } = 3 H / 2$ . The concatenated feature in Eq. (19) is first reduced:

$$
\mathbf { R } = \mathrm { D r o p o u t } ( \mathrm { R e L U } ( \mathrm { L N } ( \mathbf { W } _ { r } \mathbf { F } _ { \mathrm { c a t } } + \mathbf { b } _ { r } ) ) ) \in \mathbb { R } ^ { B \times C \times D _ { e } } .\tag{32}
$$

The feature response and gate are

$$
{ \mathbf A } = \operatorname { t a n h } ( { \mathbf W _ { a } \mathbf R + \mathbf b _ { a } } ) + \epsilon ,\tag{33}
$$

$$
\boldsymbol { \mathbf { G } } = \boldsymbol { \mathbf { R } } \odot \sigma ( \boldsymbol { \mathbf { W } } _ { g 2 } \operatorname { R e L U } ( \boldsymbol { \mathbf { W } } _ { g 1 } \boldsymbol { \mathbf { R } } + \boldsymbol { \mathbf { b } } _ { g 1 } ) + \boldsymbol { \mathbf { b } } _ { g 2 } ) .\tag{34}
$$

The output is

$$
\mathbf { F } _ { \mathrm { e n } } = \mathrm { L N } ( \mathbf { W } _ { o } \mathrm { L N } ( \mathrm { D r o p o u t } ( \mathbf { A } \odot \mathbf { G } ) + \mathbf { R } ) + \mathbf { b } _ { o } ) .\tag{35}
$$

During training, ϵ is pink feature noise generated independently for each sample and class [37]. For white noise $\xi ,$ let $\widehat { \pmb { \xi } } = \mathrm { R F F T } ( \pmb { \xi } )$ along the last feature axis and define $q _ { 0 } = 1$ $q _ { k } = k ^ { - 1 / 2 }$ for $k \geq 1$ . The injected noise is

$$
\epsilon = \rho \frac { \mathrm { I R F F T } ( \mathbf { q } \odot \widehat { \pmb { \xi } } ) } { \mathrm { S t d } [ \mathrm { I R F F T } ( \mathbf { q } \odot \widehat { \pmb { \xi } } ) ] } , \qquad \rho = 0 . 1 .\tag{36}
$$

This filter produces a $1 / f$ power spectrum over feature channels. At inference, $\epsilon = 0$ . The block regulariser is the sum of Frobenius norms of $\mathbf { W } _ { a } , \mathbf { W } _ { g 1 } , \mathbf { W } _ { g 2 }$ , and $\mathbf { W } _ { o }$

## APPENDIX D LOSSES AND FUSION DETAILS

## A. Classification Losses

For logits z and targets Y, define the mean binary crossentropy

$$
\begin{array} { l } { { \displaystyle { \mathcal B } ( { \bf z } , { \bf Y } ) = - \frac { 1 } { B C } \sum _ { b , c } \left[ Y _ { b , c } \log \sigma ( z _ { b , c } ) \right. } \ ~ } \\ { { \displaystyle \left. + ( 1 - Y _ { b , c } ) \log ( 1 - \sigma ( z _ { b , c } ) ) \right] } . } \end{array}\tag{37}
$$

The supervised branch loss is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s u p } } = \lambda _ { \mathrm { l o w } } \mathcal { B } ( \mathbf { z } ^ { \mathrm { l o w } } , \mathbf { Y } ) + \lambda _ { \mathrm { h i g h } } \mathcal { B } ( \mathbf { z } ^ { \mathrm { h i g h } } , \mathbf { Y } ) } \\ { + \lambda _ { \mathrm { e n } } \mathcal { B } ( \mathbf { z } ^ { \mathrm { e n } } , \mathbf { Y } ) + \lambda _ { \mathrm { a g g } } \mathcal { B } ( \mathbf { z } ^ { \mathrm { a g g } } , \mathbf { Y } ) . } \end{array}\tag{38}
$$

The fused-output loss is binary cross-entropy on Eq. (22):

$$
\mathcal { L } _ { \mathrm { f i n a l } } = - \frac { 1 } { B C } \sum _ { b , c } \left[ Y _ { b , c } \log p _ { b , c } ^ { \mathrm { f i n a l } } \right.\tag{39}
$$

The gate $G _ { \theta }$ is $\mathrm { ~ a ~ 3 ~ }  \mathrm { ~ 3 2 ~ }  \mathrm { ~ 3 ~ }$ multilayer perceptron with a ReLU hidden activation; its gradients are supplied by ${ \mathcal { L } } _ { \mathrm { f i n a l } }$

## B. Feature-Consistency and Shuffle Losses

The feature-consistency loss aligns the low-level and enhanced representations. After flattening the class and channel dimensions and applying $\ell _ { 2 }$ normalisation, let $\begin{array} { r l } { \mathbf { u } _ { b } } & { { } = } \end{array}$ norm $\left( \mathrm { v e c } ( \mathbf { F } _ { \mathrm { e n } , b } ) \right)$ and $\mathbf { v } _ { b } = \operatorname { n o r m } ( \operatorname { v e c } ( \mathbf { F } _ { \mathrm { l o w } , b } ) )$ . With $\tau _ { \mathrm { i n f o } } =$ $0 . 0 7$

$$
\mathcal { L } _ { \mathrm { i n f o } } = - \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \log \frac { \exp ( \mathbf { u } _ { b } ^ { \top } \mathbf { v } _ { b } / \tau _ { \mathrm { i n f o } } ) } { \sum _ { k = 1 } ^ { B } \exp ( \mathbf { u } _ { b } ^ { \top } \mathbf { v } _ { k } / \tau _ { \mathrm { i n f o } } ) } .\tag{40}
$$

For shuffle regularisation, an independent permutation $\pi _ { c }$ is sampled across the batch for every class. Define $\tilde { \mathbf { F } } _ { b , c } =$ ${ \mathbf { F } } _ { \mathrm { e n } , \pi _ { c } ( b ) , c }$ and $\tilde { Y } _ { b , c } = Y _ { \pi _ { c } ( b ) , c } .$ . Applying the aggregate classifier $h _ { \mathrm { a g g } }$ to the shuffled class features gives

$$
\mathcal { L } _ { \mathrm { s h u f } } = \mathcal { B } \Big ( h _ { \mathrm { a g g } } \big ( \mathrm { v e c } ( \tilde { \mathbf { F } } ) \big ) , \tilde { \mathbf { Y } } \Big ) .\tag{41}
$$

## C. Objective Coefficients

The coefficients used in Eq. (23) are

$$
\begin{array} { r l r } & { } & { \lambda _ { \mathrm { l o w } } = 0 . 0 1 , \quad \lambda _ { \mathrm { h i g h } } = \lambda _ { \mathrm { e n } } = 0 . 1 , \quad \lambda _ { \mathrm { a g g } } = 1 , } \\ & { } & { \lambda _ { \mathrm { r e c } } = \lambda _ { \mathrm { c l } } = \lambda _ { \mathrm { i n f o } } = 1 , \quad \lambda _ { \mathrm { s h u f } } = 0 . 1 , \quad \lambda _ { \mathrm { r e g } } = 1 0 ^ { - 4 } . } \end{array}\tag{42}
$$

The coefficient of ${ \mathcal { L } } _ { \mathrm { f i n a l } }$ is one.