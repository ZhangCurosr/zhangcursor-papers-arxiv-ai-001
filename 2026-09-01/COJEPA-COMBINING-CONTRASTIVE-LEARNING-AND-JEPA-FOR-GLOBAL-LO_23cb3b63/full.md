# COJEPA: COMBINING CONTRASTIVE LEARNING AND JEPA FOR GLOBAL-LOCAL MUSIC REPRESENTATIONS

Gabriel Meseguer-Brocal Yuexuan Kong Romain Hennequin

Deezer Research, Paris, France

research@deezer.com

## ABSTRACT

Joint-Embedding Predictive Architecture (JEPA) has shown strong performance in learning rich representations through self-supervised prediction in latent space. However, it typically relies on teacher–student architecture with an EMA to stabilise training, and can tend to yield uninformative representations. Contrastive learning is stable to train and produces strong global representations, but remains limited on local tasks by the global nature of its objective. In this work, we combine both into CoJEPA: a single shared backbone jointly trained with a JEPA objective on masked sequence tokens and a contrastive objective on the class token. The contrastive gradient provides stability, removing the need for an EMA teacher entirely, while JEPA enriches the sequence tokens via local predictions that contrastive learning alone cannot provide. Crucially, no extra parameters are added to the backbone: the same model is guided towards richer representations purely through the design of its training signal. CoJEPA takes the best of both worlds, outperforming or matching both individual methods across global and local MIR tasks, with a particularly strong advantage on tonal and harmonic understanding, and without any task-specific architectural changes. CoJEPA shows that combining objectives with complementary inductive biases can substitute for scale, encouraging future work to invest in smarter training objectives over ever-larger models.

## 1 Introduction

Self-supervised learning (SSL) has emerged as an impactful paradigm in music information retrieval (MIR) in recent years. It enables the learning of rich audio representations without requiring heavy manual annotations, thereby circumventing common issues seen in supervised learning, such as label ambiguity, annotation cost, and lack of generalization. Moreover, SSL aligns with how human learns internal musical representations, through exposure, interaction and prediction [1–3], as musical training is not necessary for requiring musical knowledge [4].

In this paradigm, a backbone is trained via a pretext task where the supervisory signal is derived directly from the data itself, rather than from human annotations. The backbone aims to produce useful representations that can be adapted to a wide range of downstream tasks with minimal effort, usually via probing, i.e., training a lightweight classifier on top of the frozen backbone representations. What makes SSL particularly attractive is not only the little requirement of annotations and the reusability of the learned representations, but also the fact that they have managed to surpass supervised approaches on a broad set of downstream tasks [5, 6]. Yet a central challenge remains: designing better training objectives that guide models toward more rich and diverse features.

As one of the latest family in SSL, Joint-Embedding Predictive Architecture (JEPA) was recently proposed [7, 8]. The core idea of of JEPA is to predict masked regions directly in a learned latent space. In practice, JEPA relies on a teacher–student framework in which a teacher encoder produces latent targets while a student network learns to predict them. The teacher is updated through an exponential moving average (EMA) mechanism to stabilize training and avoid collapse [8]. However, while powerful in learning representations, JEPA is still hard and unstable to train, with extensive hyperparameter search, and prone to collapse [8, 9]. At the same time, as an easy-to-train and stable SSL paradigm, contrastive learning performs worse on local tasks since the losses are typically applied to capture global variance and invariance [10–12]. Moreover, recent analyses of transformerbased contrastive learning models have revealed emergent musical properties in sequence tokens [13]. Surprisingly, even when contrastive objectives are applied only to a global class token, sequence tokens still encode local musical information useful for frame-level downstream tasks.

Building on these ideas, we propose CoJEPA an efficient and novel method for combining the strengths of JEPA and contrastive learning. To a traditional JEPA, we apply a contrastive loss on the class token of a transformer, as a regularisation to constrain the latent space, removing the need for EMA and thereby stabilizing training. As a result, CoJEPA improves the quality of global and local representations compared to standard JEPA and contrastive pre-training and it outperforms either method alone on downstream tasks, and remains competitive against a much larger model, MusicFM [5].

## 2 Literature review

We review prior work on contrastive learning and JEPA: self-supervised methods that propose different ways to derive the training signal from the data itself.

Joint embedding architectures (JEA) learn representations by mapping different views of the same input to nearby locations in a shared embedding space. These views are typically obtained through data augmentations or different contextual transformations of the same audio excerpt. Different approaches use different losses to control the invariance and variance needed in the latent space [14–16]. In music representation learning, joint embedding methods-and more specifically, contrastive learning—have proven particularly effective for learning global semantic representations that transfer well across downstream tasks [10, 13, 17–19]. However, since the losses are applied to a global representation, these approaches often underperform on tasks requiring fine-grained temporal information, such as beat tracking or chord estimation [13].

![](images/4db0fc44303d081dfb5302a215ff84d3209185de5641e2bbbb3de1178630082b.jpg)  
Figure 1. Overview of the proposed method. Two full views $\mathbf { x } _ { a }$ and $\mathbf { x } _ { b }$ are encoded by a shared backbone; their CLS tokens are contrasted via $\mathcal { L } _ { \mathrm { c o n t r a s t i v e } }$ . The same backbone acts as a teacher (full $\mathbf { x } _ { a } ,$ stop-gradient) and as a student (masked $\mathbf { x } _ { a } )$ , a predictor $g _ { \phi }$ reconstructs the missing tokens from context representations and learnable positional placeholders, with gradients from $\mathcal { L } _ { \mathrm { J E P A } }$ flowing back through the predictor and student path. The backbone is thus trained simultaneously as a teacher via contrastive and as a student via JEPA, removing the need for an EMA teacher.

Masking modeling methods learn by predicting masked portions of the input [20, 21]. By forcing the model to predict masked regions from contextual information, these methods encourage representations to capture rich information. In audio and music, masking-based approaches have shown strong performance, with models such as MusicFM [5] and MERT [6] establishing state-ofthe-art results across a wide range of MIR tasks. However, reconstructing signals directly in the input domain may encourage the model to focus on low-level details that are not always aligned with perceptually meaningful musical semantic features.

To address the limitations of masked modelling, Joint-Embedding Predictive Architectures (JEPA) was recently proposed [7, 8]. Rather than reconstructing missing content in the raw input space, JEPA predicts masked regions directly in a learned latent space. This shifts the objective away from low-level reconstruction and toward semantic prediction. It has shown strong performance in audio and music domain [22, 23]. Although effective, JEPA requires EMA, therefore remains difficult to optimize in practice due to training instability and sensitivity to hyperparameters [24, 25].

Equivariant SSL learns representations that transform equivariantly under known input transformations [26–31]. Applying a transformation to the input is expected to induce an equivariant transformation in the latent representation. Specific to this family, the pretext task is explicitly aligned with the downstream task of interest. While effective for specialized tasks, equivariant approaches are tightly coupled to the modelled transformation and therefore less transferable across diverse downstream tasks.

While these works have traditionally been studied in isolation, recent work has begun exploring hybrid approaches that leverage multiple SSL principles simultaneously. For instance, in MIR, [12] combines joint embedding learning with equivariance constraints, demonstrating that incorporating geometric structure can enhance representation quality. Similarly, other works have explored combinations such as contrastive learning with masked prediction [32, 33], suggesting that different SSL objectives capture complementary aspects of the data structure.

Furthermore, to address the limitations of JEPA, recent work combines predictive latent-space learning with regularized joint embedding objectives [34], by combine I-JEPA [8] with VICReg [15], introducing explicit variance and covariance regularization terms to stabilize the latent space and prevent collapse. However, EMA is still necessary to stabilize the training.

## 3 Methodology

In this section, we describe the two SSL paradigms at the core of our framework, contrastive learning and JEPA, and present how we combine them into a unified training framework. We work with an updated ViT-1D transformer [13] (Section 4) as the backbone encoder, which processes audio inputs as sequences of tokens, with a class token prepended at the beginning of the token sequence. This choice is central to our approach, as the self-attention mechanism and the class token architecture of transformers play a key role in enabling the combination of both paradigms.

## 3.1 JEPA

JEPA [7, 8] extends masked modeling by operating in the representation space: tokens are masked after encoding, and a predictor is trained to reconstruct the missing latent representations from the visible ones. Given a sequence of $T$ input audio tokens $\mathbf { x } = \{ x _ { 1 } , \dots , x _ { T } \}$ where $x _ { i } \in \mathbb { R } ^ { d _ { \mathrm { e } } }$ (see Section 4.1):

Teacher. A teacher encoder $f _ { \theta _ { t } }$ processes the full sequence x, producing a sequence of token representations $\mathbf { z } ^ { t } = [ z _ { 1 } ^ { t } , \dots , z _ { T } ^ { t } ] = f _ { \theta _ { t } } ( \mathbf { x } )$ , where each $z _ { i } ^ { t } \in \mathbb { R } ^ { d _ { \mathrm { e } } }$ captures the contextualised semantics of the whole sequence. The tokens at the masked positions serve as target tokens.

Student. A student encoder $f _ { \theta _ { s } }$ processes only the visible (context) tokens $\mathbf { x } ^ { c } = \{ x _ { i } : m _ { i } = 1 \}$ , defined by a binary mask m $\in \{ 0 , 1 \} ^ { T }$ , producing $\textstyle \sum _ { i } m _ { i }$ context tokens $\mathbf { z } ^ { c } = \{ z _ { i } ^ { c } : m _ { i } = 1 \}$ with $z _ { i } \in \mathbb { R } ^ { d _ { \mathrm { e } } }$

Predictor. A lightweight predictor $g _ { \phi }$ takes z and prior information about the missing tokens as input; commonly, the positional embeddings p<sub>i</sub> of the masked tokens. It then estimates the representations:

$$
\hat { \mathbf { z } } = g _ { \phi } ( \mathbf { z } ^ { c } , \{ p _ { i } : m _ { i } = 0 \} )
$$

$g _ { \phi }$ is itself a transformer where masked tokens attend to context positions via cross-attention, encouraging z to encode useful information for predicting the missing content.

Loss. The training objective combines L2 distance and cosine similarity between predicted and target tokens, computed only at the masked positions:

$$
\mathcal { L } _ { \mathrm { J E P A } } = \frac { 1 } { \sum _ { i } m _ { i } } \sum _ { i : m _ { i } = 0 } \left( \Vert \hat { \boldsymbol { z } } _ { i } - \boldsymbol { z } _ { i } ^ { t } \Vert _ { 2 } ^ { 2 } + 1 - \frac { \hat { \boldsymbol { z } } _ { i } ^ { t } \cdot \boldsymbol { z } _ { i } ^ { t } } { \Vert \hat { \boldsymbol { z } } _ { i } ^ { t } \Vert \Vert \boldsymbol { z } _ { i } ^ { t } \Vert } \right)
$$

The L2 term penalises magnitude differences and prevents scale ambiguity. Cosine similarity enforces directional alignment between predicted and target. Their combination is motivated by the decomposition of SSL objectives into alignment and uniformity components [11]: L2 alone is prone to representation collapse through magnitude shrinkage, whereas cosine similarity induces higherorder gradient dynamics that provide stronger collapse resistance in non-contrastive settings [35].

By predicting in latent space rather than in the raw signal domain, JEPA encourages the model toward semantically meaningful representations, abstracting away lowlevel reconstruction details.

A key challenge in JEPA is that the target $\mathbf { z } ^ { t }$ is not fixed but evolve throughout training, since the teacher encoder $f _ { \theta _ { t } }$ is updated through an EMA of the student weights:

$$
\theta _ { t }  \tau \theta _ { t } + ( 1 - \tau ) \theta _ { s }
$$

where $\tau \in [ 0 , 1 )$ is a momentum coefficient progressively increased toward 1 [8,36]. This moving target creates a coadaptation dynamic: the student is pushed toward semantically rich representations in order to be useful to predict the targets, while the targets are pressured to become easier to predict, leading to the potential representational collapse [34]. Although EMA slows the process by ensuring targets evolve smoothly, it does not fully prevent collapse and does not necessarily ensure useful features $[ 9 , 1 5 , 3 4 ] .$ while also introducing more hyperparameters related to τ .

## 3.2 Contrastive learning

Contrastive learning learns representations by mapping different views of the same input to nearby locations in a shared embedding space. We obtain the two views by sampling from the same track two random temporally proximate segments [10, 13, 17–19]. Intuitively, two excerpts from the same song should share sufficient musical semantics to yield similar representations, while segments from different tracks should be distinguishable. Given a batch of B audio samples, each sample yields two views $\mathbf { x }$ and $\mathbf { x } ^ { \prime } .$ . A dedicated class token is prepended at the beginning of the token sequence, i.e., position 0, initialized with trainable parameters and the average of the sequence tokens, used to aggregate information from the full sequence via self-attention. Two views are passed through the same encoder and class tokens are obtained as: $z _ { 0 } = f _ { \theta } ( \mathbf { x } ) _ { \scriptscriptstyle \left[ \mathrm { C L S } \right] } , z _ { 0 } ^ { \prime } = f _ { \theta } ( \mathbf { x } ^ { \prime } ) _ { \scriptscriptstyle \left[ \mathrm { C L S } \right] }$

Loss. We use the NT-Xent loss [14]. For a positive pair $( z _ { 0 , q } , z _ { 0 , q } ^ { \prime } )$ obtained from the same audio sample q (we add $\mathsf { q }$ for a clearer demonstration of contrastive loss), the perpair loss is defined as:

$$
\ell ( q ) = - \log \frac { \exp \left( { \sin ( z _ { 0 , q } , z _ { 0 , q } ^ { \prime } ) / t } \right) } { \sum _ { k \ne q } ^ { } \exp ( { \sin ( z _ { 0 , q } , z _ { 0 , k } ) / t } ) } ,
$$

the denominator includes representations from all other samples in the batch, which act as negative examples. The total loss $\mathcal { L } _ { \mathrm { N T - X e n t } }$ is computed for all positive pairs.

## 3.3 Combining JEPA and contrastive learning

Recent work has shown that transformers trained with a contrastive objective on the class token already develop semantically meaningful representations in their sequence tokens, even without any direct supervision on them [13]. This suggests that contrastive learning naturally sets up a representation space where JEPA can operate effectively and benefit from. We leverage this observation and combine the two paradigms into CoJEPA: contrastive learning provides a well-structured, stable global embedding space, while JEPA explicitly pushes individual sequence tokens to encode rich local information. Figure 1 provides an overview of the proposed method and the flow of gradients through each objective.

Shared backbone. Our key design contribution is to use a single shared backbone $f _ { \theta }$ (i.e. with $\theta = \theta _ { t } = \theta _ { s } )$ for the teacher and student roles, without any EMA mechanism. Depending on the input it receives, $f _ { \theta }$ operates in two modes (annotations slightly differ from Section 3.1):

• Teacher mode: $f _ { \theta }$ processes thefull sequence x and produces a complete representation $\mathbf { z } = f _ { \boldsymbol { \theta } } ( \mathbf { x } )$ , including a class token $z _ { \mathrm { 0 } }$ and sequence tokens $\mathbf { z } _ { 1 : N }$

• Student mode: $f _ { \theta }$ processes only the visible context tokens $\mathbf { x } ^ { c }$ , producing context representations $\mathbf { z } ^ { c }$

Forward pass. Each training step processes two views $( { \bf x } , { \bf x } ^ { \prime } )$ sampled from the same track. Both views are passed through $f _ { \theta }$ in teacher mode, producing:

$$
\begin{array} { r } { \mathbf { z } = f _ { \theta } ( \mathbf { x } ) , \qquad \mathbf { z } ^ { \prime } = f _ { \theta } ( \mathbf { x } ^ { \prime } ) } \end{array}
$$

The class tokens $z _ { 0 }$ and $z _ { 0 } ^ { \prime }$ are used to compute the NT-Xent contrastive loss $\mathcal { L } _ { \mathrm { N T - X e n t } } .$ . In parallel, the sequence tokens at the masked positions $\{ z _ { i } : m _ { i } = 0 \}$ serve as JEPA prediction targets. These are detached from the computational graph $( \mathbf { s g } ( \{ z _ { i } : m _ { i } = 0 \} ) )$ ), so that the JEPA loss does not propagate through teacher mode. The masked views $\mathbf { x } ^ { c }$ are then processed in student mode, and the predictor recovers the target representations:

$$
\hat { \mathbf { z } } = g _ { \phi } ( f _ { \theta } ( \mathbf { x } ^ { c } ) , \{ p _ { i } : m _ { i } = 0 \} )
$$

Gradient flow. The backbone $f _ { \theta }$ receives gradients from two sources:

$\mathcal { L } _ { \mathrm { N T - X e n t } } .$ flows through the teacher forward pass, shaping the global structure of the embedding space.

$\mathcal { L } _ { \mathrm { J E P A } }$ : flows through the predictor and the student forward pass, pushing sequence tokens to encode locally predictive information.

The total training objective is:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { N T - X e n t } } + \mathcal { L } _ { \mathrm { J E P A } }
$$

Eliminating EMA. Unlike standard JEPA, CoJEPA does not require EMA mechanism. The contrastive loss directly regularizes the backbone, preventing representational collapse by enforcing a well-structured global embedding space. The teacher is thus not a slowly drifting copy of the student but the same model, updated jointly and stably through $\mathcal { L } _ { \mathrm { N T - X e n t } }$

## 4 Model

## 4.1 Inputs

Our model takes as input the raw waveform sampled at 16 kHz and computes a normalized mel-frequency spectrogram with 128 frequency bins and a frame rate of 31.5 Hz. This representation is widely used in MIR for contrastive learning and audio tagging task [10, 18]. The spectrogram is patchified such that each time frame is projected into the transformer embedding dimension, yielding a sequence of tokens ${ \pmb x } = [ { \pmb x } _ { 1 } , \ldots , { \pmb x } _ { T } ]$ with $\pmb { x } _ { i } \in \mathbb { R } ^ { d _ { \mathrm { e } } } , d _ { \mathrm { e } } = 1 9 2$ , and $T = 1 2 6$ for a 4-second segment.

## 4.2 Backbone

The backbone is updated based on a Vision Transformer with 1-D patches (ViT-1D) [13], using an embedding dimension of 192, a depth of 12 blocks, and 3 attention heads per block, Tiny configuration from the original ViT paper [37]. We incorporate three architectural upgrades from modern large language models [38]: (a) rotary positiona embeddings (RoPE) in place of standard learned positiona embeddings, (b) RMSNorm instead of LayerNorm, and (c) feed-forward layers with gated activations. These upgrades result in a backbone with 7,1M trainable parameters.

Class token. The class token (CLS token) is a special token prepended to the input sequence, responsible for aggregating global sequence information via self-attention. It is defined as a learnable vector of dimension $d _ { \mathrm { e } } = 1 9 2 .$ to which we add the mean of the projected input tokens, x<sub>spe</sub>. And it is where the joint-embedding loss is applied.

<table><tr><td>Task</td><td>Dataset</td><td>Size</td><td>nb. classes</td></tr><tr><td colspan="4">Global tasks</td></tr><tr><td>Tagging</td><td>MTAT</td><td>25k</td><td>50</td></tr><tr><td>Key</td><td>FMAKv2</td><td>5.5k</td><td>24</td></tr><tr><td>Pitch</td><td>TinySOL</td><td>3k</td><td>84</td></tr><tr><td>Instruments</td><td>TinySOL</td><td>3k</td><td>14</td></tr><tr><td>BPM</td><td> $\mathrm { G i a n t S t e p s _ { t e m p o } }$ </td><td>664</td><td>72</td></tr><tr><td colspan="4">Local tasks</td></tr><tr><td>Chords</td><td> $\mathrm { R W C } _ { \mathrm { p o p } } + \mathrm { S W D }$ </td><td>124</td><td>25</td></tr><tr><td>Beat</td><td> $\mathrm { B a l l r o o m + G T Z A N _ { r h y t h m } }$ </td><td>1.7k</td><td></td></tr></table>

Table 1. Statistics of downstream datasets. We omit the nb. classes for beat tracking (More details in 5.2).

Masking technique. We apply contiguous temporal masking: a single block of consecutive time frames is masked, with all frequency bins zeroed out. Since our patchification maps one time frame to one token, this is equivalent to masking a contiguous block of tokens in the input sequence. The mask length per sequence is sampled uniformly from 50% to 75% of the sequence length T. The starting position is drawn uniformly at random from the valid range. Each sample in the batch receives an independently sampled mask length, introducing variability in the amount of context available to the student encoder.

## 4.3 Predictor

The predictor has the same architecture as the backbone but smaller: an embedding dimension of 192, a depth of 6 blocks, resulting in 3.5M parameters. Ablation of these design choices is left for future work; which is relevant given that intermediate layers of the backbone are the ones capturing the richest representations, suggesting the predictor’s depth plays a role in shaping them Section 5.

Masked tokens. At each masked position, a learnable token of dimension 192 is injected as a placeholder, to which the corresponding rotary embedding is added to encode the position of the missing token within the sequence. This allows the predictor to attend to masked positions while distinguishing them from the visible context tokens.

## 5 Experiments

## 5.1 Pretraining

We train three models: JEPA, contrastive, and CoJEPA. For the JEPA baseline we follow the standard EMA-based training procedure [8].

Dataset. We use an in-house dataset consisting of about 2M full tracks to pre-train our models. The dataset is curated to be as musically balanced and diverse as possible, in order to improve generalization to unseen data.

Hyperparameters. All models are trained for 999 epochs with 512 steps per epoch and a batch size of 128. We note that the batch size of 128 is small relative to common contrastive learning practice, which typically relies on large batches to provide sufficient negative pairs [10]. We use the AdamW optimizer with $\beta = ( 0 . 9 , 0 . 9 8 )$ and a cosine learning rate schedule with 10 warm-up epochs and a minimum learning rate of $1 . 5 \times 1 0 ^ { - 6 }$ . Weight decay follows a cosine schedule from 0.04 to 0.01. For the JEPA EMA baseline, the teacher momentum is scheduled from 0.996 to 0.9999 over the full training. The masking ratio is fixed at 75%. All models are trained on a single GPU.

<table><tr><td rowspan="2"></td><td rowspan="2"></td><td colspan="5">Layer 4 - early</td><td colspan="5">Layer 8 - mid</td><td colspan="5">Layer 12 - last</td><td rowspan="2">Baseline</td></tr><tr><td>JEPA</td><td colspan="2">Contrastive</td><td colspan="2">CoJEPA</td><td>JEPA</td><td>Contrastive</td><td colspan="2"></td><td>CoJEPA</td><td>JEPA</td><td>Contrastive</td><td></td><td>CoJEPA</td><td>MusicFM</td></tr><tr><td>Task</td><td>Metric</td><td>seq</td><td>CLS</td><td>seq</td><td>CLS</td><td>seq</td><td>seq</td><td>CLS</td><td>seq</td><td>CLS</td><td>seq</td><td>seq</td><td>CLS</td><td>seq</td><td>CLS</td><td>seq</td><td>seq</td></tr><tr><td rowspan="2">Tagging</td><td> $\mathbf { M A P _ { m a c r o } }$ </td><td>.404</td><td>.265</td><td>.266</td><td>.410</td><td>.429</td><td>.393</td><td>.266</td><td>.266</td><td>.436</td><td>.436</td><td>.376</td><td>.393</td><td>.395</td><td>.395</td><td>.421</td><td>.454</td></tr><tr><td>ROC  ${ \bf \cdot A U C } _ { \mathrm { m a c r o } }$ </td><td>.888</td><td>.804</td><td>.803</td><td>.891</td><td>.896</td><td>.884</td><td>.804</td><td>.802</td><td>.903</td><td>.902</td><td>.875</td><td>.882</td><td>.881</td><td>.884</td><td>.897</td><td>.906</td></tr><tr><td>Key</td><td>w-acc</td><td>.245</td><td>.232</td><td>.243</td><td>.633</td><td>.609</td><td>.167</td><td>.234</td><td>.236</td><td>.640</td><td>.518</td><td>.147</td><td>.539</td><td>.550</td><td>.503</td><td>.396</td><td>.558</td></tr><tr><td>Pitch</td><td>acc</td><td>.959</td><td>.962</td><td>.966</td><td>.983</td><td>.979</td><td>.877</td><td>.962</td><td>.962</td><td>.973</td><td>.973</td><td>.699</td><td>.925</td><td>.962</td><td>.873</td><td>.921</td><td>.969</td></tr><tr><td>Instruments</td><td>acc</td><td>.938</td><td>.692</td><td>.688</td><td>.935</td><td>.945</td><td>.911</td><td>.695</td><td>.688</td><td>.949</td><td>.952</td><td>.873</td><td>.942</td><td>.952</td><td>.942</td><td>.925</td><td>.897</td></tr><tr><td rowspan="2">BPM</td><td>acc@1</td><td>.375</td><td>.444</td><td>.403</td><td>.389</td><td>.604</td><td>.806</td><td>.389</td><td>.396</td><td>.757</td><td>.826</td><td>.833</td><td>.854</td><td>.847</td><td>.840</td><td>.826</td><td>.771</td></tr><tr><td>acc@2</td><td>.389</td><td>.444</td><td>.403</td><td>.389</td><td>.632</td><td>.847</td><td>.389</td><td>.396</td><td>.799</td><td>.861</td><td>.875</td><td>.917</td><td>.895</td><td>.910</td><td>.875</td><td>.819</td></tr></table>

Table 2. Global tasks at three transformer depths. JEPA is evaluated only on mean-pooled sequence tokens (seq). Contrastive and CoJEPA are evaluated on both the CLS token and mean-pooled sequence tokens. MusicFM is reported as a single baseline without layer ablation. Underlined: best overall. Bold: best among our three models. Red : best JEPA. Yellow : best Contrastive. Blue : best CoJEPA.

## 5.2 Downstream tasks

We group tasks into global (sequence-level prediction: tagging, key, pitch, BPM) and local (frame-level prediction: beat tracking, chord estimation). Table 1 summarizes all evaluation datasets. Global tasks require a sequence-level prediction capturing the overall musical content, while local tasks require frame-level predictions at a resolution adapted to the downstream task.

Global tasks. For music tagging, we train and evaluate on MagnaTagATune using the split of [39], and report macro ROC-AUC and mAP. For instrument recognition and pitch estimation, we use TinySOL [40] with a random 8:1:1 split, evaluating top-1 accuracy over 14 instrument classes and 82 pitch classes respectively. For key estimation, we train on FMAKv2 [28] and test on GiantSteps [41], using the MIREX weighted accuracy metric [42]. For BPM estimation, we use Giantsteps-tempodataset [41] with a split of 8:1:1 for training, validation and evaluation. We report ACC<sub>1</sub>, accuracy within a 4% tolerance, and $\mathrm { A C C _ { 2 } }$ accuracy with the same tolerance additionally allowing octave errors of 2, 3, 1/2, and 1/3.

Local tasks. For beat tracking, we train on Ballroom Dataset [43] and evaluate on GTZAN Rhythm [44]. We apply temporal smoothing, reporting the $F _ { 1 }$ -score with a ±70 ms tolerance window. For chord estimation, we use a combined corpus of RWC-POP [45] and Schubert Winterreise Dataset [46] with an 8:1:1 split, and evaluate framelevel accuracy over 24 major/minor chord classes plus a “no chord” class.

<table><tr><td colspan="2"></td><td>Chords</td><td colspan="2">Beat</td></tr><tr><td>Layer</td><td>Model</td><td>overlap</td><td>F-measure</td><td>P-score</td></tr><tr><td rowspan="3">4 - early</td><td>JEPA</td><td>.211</td><td>.778</td><td>.745</td></tr><tr><td>Cont.</td><td>.258</td><td>.549</td><td>.551</td></tr><tr><td>CoJEPA</td><td>.399</td><td>.778</td><td>.747</td></tr><tr><td rowspan="3">8 - mid</td><td>JEPA</td><td>.097</td><td>.803</td><td>.774</td></tr><tr><td>Cont.</td><td>.256</td><td>.551</td><td>.551</td></tr><tr><td>CoJEPA</td><td>.275</td><td>.793</td><td>.764</td></tr><tr><td rowspan="3">12 - last</td><td>JEPA</td><td>.070</td><td>.806</td><td>.780</td></tr><tr><td>Cont.</td><td>.241</td><td>.737</td><td>.707</td></tr><tr><td>CoJEPA</td><td>.160</td><td>.789</td><td>.757</td></tr><tr><td>Baseline</td><td>MusicFM</td><td>.193</td><td>.866</td><td>.851</td></tr></table>

Table 3. Local tasks. The probe is applied independently to each sequence token (no pooling).

## 5.3 Evaluation protocols

We follow the standard SSL linear probing protocol: backbone weights are frozen and a single linear layer (192 → nb\_classes + softmax) is trained per task for 50 epochs, 128 steps per epoch, batch size 128, learning rate $1 0 ^ { - 3 }$ and fixed random seed. We probe at three transformer depths: layer 4 (early), layer 8 (mid), and layer 12 (last), since different layers capture qualitatively different information [13,47], allowing us to better attribute performance differences to each training objective.

Global tasks. We probe a single fixed-length sequence representation using one of two pooling strategies:

• CLS token: for Contrastive and CoJEPA, explicitly optimized to aggregate global information.

• Mean-pooled sequence tokens: The only option for JEPA (no CLS token). We computed for all three models as it provides a complementary view of how well the local tokens capture global musical content.

Local tasks. The linear probe is applied independently to each token in the sequence. This evaluates the quality of token-level representations and is relevant for assessing the benefit of the JEPA objective on local features.

Baseline. We compare against MusicFM [5], a recent SSL foundation model based on a Conformer encoder (330M parameters, $d _ { e } = 1 0 2 4 )$ . We extract features from its last transformer layer only and train our linear probes on 10-second segments (consistent with pretraining) or on the longest available segment when the audio is shorter. We acknowledge that intermediate layers may yield stronger results [47]. MusicFM thus serves as an indicative reference point against a recent large-scale model.

## 6 Results

Comparison of JEPA, Contrastive, and CoJEPA. Tables 2 and 3 report linear probing results across all tasks at three layers. CoJEPA achieves the best or tied-best result on the majority of tasks, confirming that combining both objectives takes the best of both worlds: the local representational richness of JEPA and the globally structured embedding space of contrastive learning. The only exceptions are rhythmic tasks: BPM, where Contrastive (CLS) marginally leads, and beat tracking, where JEPA leads; yet CoJEPA remains competitive in both cases. Tasks requiring different musical understanding: tagging, key, and chords; show the largest benefit, with substantial margins over either individual method.

JEPA representations peak early. For most tasks, JEPA representations are strongest at layer 4 and degrade with depth: pitch drops from .959 to .699, key from .245 to .147, and chords from .211 to .070. This suggests that the JEPA pretext task, as a local temporal prediction objective, produces its most transferable features early in the network but does not provide sufficient guidance for deeper layers to develop music rich semantic representations. This is consistent with findings showing that JEPA alone requires large-capacity models to avoid representation collapse in later layers [8]. The notable exception is BPM and beat tracking, where JEPA improves with depth (BPM acc@1: .375 → .833, beat F-measure: .778 → .806). This suggests that rhythmic regularity, which is temporally localised and periodically structured, is naturally captured by the JEPA when representations mature, unlike tonal and semantic features that require explicit auxiliary objectives to steer training beyond JEPA’s inherent temporal bias.

Contrastive representations crystallise at the last layer. Contrastive models produce weak early-layer representations for most tasks (e.g., key w-acc at layer 4: .243, instruments at layer 4: .692) but improve dramatically at the final layer (key: .550, instruments: .952). This indicates that contrastive learning encodes discriminative information primarily in the last-layer global embedding, leaving intermediate representations relatively uninformative. On local tasks, this weakness is most apparent: beat F-measure reaches only .737 even at the last layer, well below JEPA (.806) and CoJEPA (.793), reflecting the lack of direct supervision on individual sequence tokens.

CoJEPA peaks at intermediate depth. CoJEPA achieves its best performance at layer 8 for most tasks: tagging MAP (.436), key (.640), instruments (.952), beat (.793); a qualitatively distinct behaviour from JEPA (peaks early) and Contrastive (peaks late), suggesting the two objectives interact to produce well-structured representations at intermediate depth. BPM follows the rhythmic trend of JEPA, improving steadily with depth (CLS at L12: .840). Particularly striking is the early emergence of tonal representations. Pitch, and chords peak at layer 4 (.983 and .399) and key at layer-8 (.64). This is unexpected given that harmonic information has been shown to be difficult to capture in general music representation learning models [6, 48] without a specifically designed training objective [28]. These features are not preserved at depth, likely discarded as contrastive increasingly dominates.

CLS vs. Sequence Tokens. For Contrastive models, the gap between CLS and mean-pooled sequence tokens is consistently small, suggesting the contrastive signal passively enriches sequence tokens through self-attention. For CoJEPA, the gap is larger and task-dependent: the CLS token dominates on key (.640 vs. .518 at layer 8), While sequence tokens are better on BPM and tagging, particularly at early and intermediate layers where the JEPA objective is most influential. This layer-dependent divergence suggests that the two token types carry complementary information in CoJEPA: the CLS encodes a globally summary, enforced by the contrastive objective; the sequence tokens retain local structure, encouraged by the JEPA objective.

Comparison with MusicFM. Despite its much larger scale, MusicFM leads only on tagging and beat tracking, and matching results for on pitch. As a purely masked prediction model, MusicFM also mirrors the behaviour of JEPA on rhythmic versus tonal tasks (strong on beat, weaker on chords) suggesting that the masked prediction objective, without a global discriminative component, faces the same structural limitations regardless of scale. Given the large gap in model capacity (7.1M, $d _ { e } = 1 9 2$ vs 330M, $d _ { e } = 1 0 2 4 )$ , these results further support the effectiveness of principled objective design over sheer scaling.

## 7 Conclusion

We presented CoJEPA, a self-supervised framework that unifies JEPA and contrastive learning. The two objectives are complementary: JEPA drives sequence tokens to encode rich local structure; while contrastive learning shapes a globally discriminative embedding space, its gradient preventing collapse without the need for an EMA teacher. Crucially, both objectives operate on the same backbone: rather than adding parameters, CoJEPA guides the model towards richer representations purely through the design of its training signal. Despite its small scale (7.1M), CoJEPA outperforms or matches both individual methods across global and local tasks, with a particularly strong advantage on harmonic understanding. Interestingly, the richest representations emerge at intermediate depth rather than the final layer, a behaviour that neither JEPA nor contrastive learning exhibits alone. While CoJEPA is expected to benefit from scaling model and batch sizes (as contrastive learning improves with more negative pairs [10]) its strong performance at small scale shows the benefits of complementary training objectives. We encourage future work to invest more in smarter designs, notably the combination of different self-supervised learning objectives.

## 8 References

[1] J. S. Bamberger, The mind behind the musical ear: How children develop musical intelligence. Harvard University Press, 1991.

[2] D. J. Levitin, “What does it mean to be musical?” Neuron, vol. 73, no. 4, pp. 633–637, 2012.

[3] H. Honing, The illiterate listener: On music cognition, musicality and methodology. Amsterdam University Press, 2011.

[4] C. L. Krumhansl, Cognitive Foundations of Musical Pitch. Oxford University Press, Nov. 2001. [Online]. Available: https://doi.org/10.1093/acprof: oso/9780195148367.001.0001

[5] M. Won, Y.-N. Hung, and D. Le, “A foundation model for music informatics,” in Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2024.

[6] L. Yizhi, R. Yuan, G. Zhang, Y. Ma, X. Chen, H. Yin, C. Xiao, C. Lin, A. Ragni, E. Benetos et al., “Mert: Acoustic music understanding model with large-scale self-supervised training,” in International Conference on Learning Representations, 2024.

[7] Y. LeCun, “A path towards autonomous machine intelligence,” Open Review, vol. 62, no. 1, pp. 1–62, 2022.

[8] M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, and N. Ballas, “Self-supervised learning from images with a jointembedding predictive architecture,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 15 619–15 629.

[9] R. Balestriero and Y. LeCun, “Lejepa: Provable and scalable self-supervised learning without the heuristics,” arXiv preprint arXiv:2511.08544, 2025.

[10] M. C. McCallum, F. Korzeniowski, S. Oramas, F. Gouyon, and A. F. Ehmann, “Supervised and unsupervised learning of audio representations for music understanding,” in Proc. International Society for Music Information Retrieval Conference (ISMIR), 2022.

[11] T. Wang and P. Isola, “Understanding contrastive representation learning through alignment and uniformity on the hypersphere,” in Proceedings of the 37th International Conference on Machine Learning (ICML). PMLR, 2020.

[12] Y. Kong, V. Lostanlen, R. Hennequin, M. Lagrange, and G. Meseguer-Brocal, “Multi-class-token transformer for multitask self-supervised music information retrieval,” in 2025 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA). IEEE, 2025, pp. 1–5.

[13] Y. Kong, G. Meseguer-Brocal, V. Lostanlen, M. Lagrange, and R. Hennequin, “Emergent musical properties of a transformer under contrastive self-supervised learning,” in Proceedings of the International Society on Music Information Retrieval (ISMIR) Conference, 2025.

[14] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” in Proceedings of the 37th International Conference on Machine Learning (ICML). PMLR, 2020, pp. 1597–1607.

[15] A. Bardes, J. Ponce, and Y. LeCun, “Vicreg: Variance-invariance-covariance regularization for selfsupervised learning,” in International Conference on Learning Representations, 2022.

[16] J. Zbontar, L. Jing, I. Misra, Y. LeCun, and S. Deny, “Barlow Twins: Self-Supervised Learning via Redundancy Reduction,” Jun. 2021, arXiv:2103.03230 [cs]. [Online]. Available: http://arxiv.org/abs/2103. 03230

[17] J. Spijkervet and J. A. Burgoyne, “Contrastive learning of musical representations,” arXiv preprint arXiv:2103.09410, 2021.

[18] G. Meseguer-Brocal, D. Desblancs, and R. Hennequin, “An experimental comparison of multi-view self-supervised methods for music tagging,” in ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024, pp. 1141–1145.

[19] A. Ferraro, X. Favory, K. Drossos, Y. Kim, and D. Bogdanov, “Enriched music representations with multiple cross-modal contrastive learning,” IEEE Signal Processing Letters, vol. 28, pp. 733–737, 2021.

[20] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick, “Masked autoencoders are scalable vision learners,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 16 000–16 009.

[21] W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, “Hubert: Selfsupervised speech representation learning by masked prediction of hidden units,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 3451–3460, 2021.

[22] A. Riou, S. Lattner, G. Hadjeres, M. Anslow, and G. Peeters, “Stem-JEPA: A Joint-Embedding Predictive Architecture for Musical Stem Compatibility Estimation,” Aug. 2024, arXiv:2408.02514 [cs]. [Online]. Available: http://arxiv.org/abs/2408.02514

[23] A. Quelennec, P. Chouteau, G. Peeters, and S. Essid, “Matpac++: Enhanced masked latent prediction for self-supervised audio representation learning,” arXiv preprint arXiv:2508.12709, 2025.

[24] X. Chen and K. He, “Exploring simple siamese representation learning,” CoRR, vol. abs/2011.10566, 2020. [Online]. Available: https://arxiv.org/abs/2011. 10566

[25] Y. Tian, X. Chen, and S. Ganguli, “Understanding self-supervised learning dynamics without contrastive pairs,” CoRR, vol. abs/2102.06810, 2021. [Online]. Available: https://arxiv.org/abs/2102.06810

[26] E. Quinton, “Equivariant self-supervision for musical tempo estimation,” in Proceedings ofthe International Society on Music Information Retrieval (ISMIR) Conference, 2022.

[27] A. Riou, S. Lattner, G. Hadjeres, and G. Peeters, “Pesto: Pitch estimation with self-supervised transposition-equivariant objective,” in Ismir 2023 Hybrid Conference, 2023.

[28] Y. Kong, V. Lostanlen, G. Meseguer-Brocal, S. Wong, M. Lagrange, and R. Hennequin, “Stone: Selfsupervised tonality estimator,” in International Society for Music Information Retrieval Conference (ISMIR), 2024.

[29] Y. Kong, G. Meseguer-Brocal, V. Lostanlen, M. Lagrange, and R. Hennequin, “S-key: Self-supervised learning of major and minor keys from audio,” in ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2025, pp. 1–5.

[30] B. Gfeller, C. Frank, D. Roblek, M. Sharifi, M. Tagliasacchi, and M. Velimirovic, “SPICE: Self-´ supervised pitch estimation,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 28, pp. 1118–1128, 2020.

[31] F. Cwitkowitz and Z. Duan, “Toward fully selfsupervised multi-pitch estimation,” arXiv preprint arXiv:2402.15569, 2024.

[32] Y. Gong, A. Rouditchenko, A. H. Liu, D. Harwath, L. Karlinsky, H. Kuehne, and J. Glass, “Contrastive audio-visual masked autoencoder,” in International Conference on Learning Representations, 2023.

[33] Y.-A. Chung, Y. Zhang, W. Han, C.-C. Chiu, J. Qin, R. Pang, and Y. Wu, “W2v-bert: Combining contrastive learning and masked language modeling for self-supervised speech pre-training,” arXiv preprint arXiv:2108.06209, 2021.

[34] S. Mo and S. Tong, “Connecting joint-embedding predictive architecture with contrastive self-supervised learning,” in Advances in Neural Information Processing Systems, 2024.

[35] H. Bao, Y. Nagano, and K. Nozawa, “Feature normalization prevents collapse of non-contrastive learning dynamics,” arXiv preprint arXiv:2309.16109, 2023.

[36] J.-B. Grill, F. Strub, F. Altché, C. Tallec, P. Richemond, E. Buchatskaya, C. Doersch, B. Avila Pires, Z. Guo, M. Gheshlaghi Azar et al., “Bootstrap your own latent: A new approach to self-supervised learning,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 21 271–21 284.

[37] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.

[38] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan et al., “The llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.

[39] J. Lee, J. Park, K. L. Kim, and J. Nam, “Sample-level deep convolutional neural networks for music autotagging using raw waveforms,” in Proc. International Conference on Sound and Music Computing (SMC), 2017.

[40] C. E. Cella, D. Ghisi, V. Lostanlen, F. Lévy, J. Fineberg, and Y. Maresz, “Orchideasol: A dataset of extended instrumental techniques for computer-aided orchestration,” in Proc. International Computer Music Conference (ICMC), 2020.

[41] P. Knees, Ángel Faraldo, P. Herrera, R. Vogl, S. Böck, F. Hörschläger, and M. L. Goff, “Two datasets for tempo estimation and key detection in electronic dance music annotated from user corrections,” in Proc. International Society for Music Information Retrieval Conference (ISMIR), 2015.

[42] C. Raffel, B. McFee, E. J. Humphrey, J. Salamon, O. Nieto, D. Liang, D. P. Ellis, and C. C. Raffel, “mir\_eval: A transparent implementation of common mir metrics.” in Proc. International Society for Music Information Retrieval Conference (ISMIR), vol. 10, 2014, p. 2014.

[43] F. Gouyon and S. Dixon, “A review of rhythm description systems,” in Proc. International Societyfor Music Information Retrieval Conference (ISMIR), 2004.

[44] U. Marchand, Q. Fresnel, and G. Peeters, “Gtzanrhythm: Extending the gtzan test-set with beat, downbeat and swing annotations,” Proc. International Society for Music Information Retrieval Latebreaking/Demo (ISMIR-LBD), 2015.

[45] M. Goto and H. Hashiguchi, “RWC music database: Popular, classical, and jazz music databases,” Proc. International Society for Music Information Retrieval Conference (ISMIR), 2002.

[46] C. Weiß, F. Zalkow, V. Arifi-Müller, M. Müller, H. V. Koops, A. Volk, and H. G. Grohganz, “Schubert winterreise dataset: A multimodal scenario for music analysis,” Journal on Computing and Cultural Heritage (JOCCH), vol. 14, no. 2, pp. 1–18, 2021.

[47] A. Pasad, J.-C. Chou, and K. Livescu, “Layer-wise analysis of a self-supervised speech representation model,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021, pp. 3179–3183.

[48] Y. Ding and A. Lerch, “Parameter-efficient transfer learning for music foundation models,” arXiv preprint arXiv:2411.19371, 2024.