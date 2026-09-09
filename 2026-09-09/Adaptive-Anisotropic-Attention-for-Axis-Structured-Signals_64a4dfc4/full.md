# Adaptive Anisotropic Attention for Axis-Structured Signals

Mahir Jain Mannas AI mahir@mannas.ai

Parshva Runwal Mannas AI parshva@mannas.ai

Aditya Ray Mishra Mannas AI aditya@mannas.ai

Arvasu Kulkarni Mannas AI arvasu@mannas.ai

Siddharth Panwar Mannas AI siddharth@mannas.ai

Sandeep Singh Mannas AI sandeep@mannas.ai

## Abstract

Dense self-attention treats all token pairs as equally plausible before learning—an interaction-isotropic prior that can be mismatched to structured signals. For structured, low signal-to-noise ratio (SNR) signals such as EEG, dependencies are organized along the electrode and time axes, and this uniform prior exposes each token to many irrelevant interactions. We introduce Adaptive Anisotropic Attention (AAA), which splits attention into two paths: a temporal path, where each token attends to the tokens of its own electrode across time, and a spatial path, where it attends to the tokens of the other electrodes at the same time step. A small gate predicts, for every token, a convex combination and the token’s update is the weighted sum of the two path outputs. On six EEG downstream tasks, the resulting model, AXON (AXis-factorized Operator Network), improves mean balanced accuracy over a dense baseline under both linear probing and full fine-tuning. We show that both paths (temporal and spatial) are necessary and that the weighted sum beats a hard choice of one path; most of the benefit comes from the gate learning a different temporal/spatial balance at each layer of the network. Controlled audio spectrogram experiments show that axis factorization transfers beyond EEG. These results suggest that aligning attention with the natural axes of structured signals provides a useful inductive bias.

## 1 Introduction

Transformers [Vaswani et al., 2017] process tokens: small input segments represented as vectors. In EEG and audio, tokens form two-dimensional grids. An EEG token contains one second of signal from one electrode; an audio token covers one frequency band over one spectrogram frame. Dense self-attention connects all tokens directly, allowing long-range interactions without distinguishing the electrode–time or frequency–time axes.

For structured spatiotemporal signals such as EEG, which lie on electrodes × time, this interactionisotropic prior may be mismatched. In masked autoencoding (MAE) [He et al., 2022], the model learns to reconstruct masked patches, not to distinguish downstream classes. Our hypothesis is that dense global attention encourages reconstruction shortcuts: for example, estimating a masked patch from a broad average across electrodes and time. This dilutes localized or axis-specific signals that are critical for downstream tasks such as motor imagery classification.

We introduce Adaptive Anisotropic Attention (AAA), which replaces dense encoder attention with two parallel paths. The temporal path attends within each electrode across time; the spatial path attends across electrodes at the same time step. A small gate combines their outputs through a convex combination. This soft mixture can vary across tokens and layers. We call the resulting model AXON (AXis-factorized Operator Network) and evaluate it on six EEG tasks under linear probing (LP) and full fine-tuning (FT), against dense baselines including a parameter-matched model.

We analyse learned axis mixtures and task-dependent temporal context, and test the design on audio spectrograms as a second axis-structured modality. Axis factorization helps both modalities.

Related work. Factorized attention. Axial attention [Ho et al., 2019] and divided space-time attention [Bertasius et al., 2021, Arnab et al., 2021] were built for images and video and attend along one axis at a time. The axis order is set by hand and is the same for every token and every layer, and these designs were not built for, or tested on, low-SNR signals such as EEG. AXON keeps the two axes but runs them in parallel and lets a gate set the balance per token and per layer. EEGfoundation models. EEGPT, LaBraM, CBraMod, BIOT, CSBrain and REVE [Wang et al., 2024, Jiang et al., 2024, Wang et al., 2025, Yang et al., 2023, Zhou et al., 2025, El Ouahidi et al., 2025] pretrain large encoders on unlabelled EEG so that one model transfers to many tasks. They differ in tokenisation and pretraining objective; CBraMod, attends along time and along channels in parallel and combines the two with a fixed split of attention heads. None of them tests whether the dense all-to-all path should be removed, or lets the time/channel balance be learned per token and per layer. We evaluate all six under one protocol (Tables 3 and 4).

## 2 Method: Adaptive Anisotropic Attention

EEG tasks require different temporal and spatial contexts (Appendix A). The encoder is a stack of 22 identical blocks, which we call layers; each layer has its own attention paths and its own gate weights, so “per layer” below means a separate value in each of the 22 blocks.

## 2.1 Input mannas.ai

We process non-overlapping 10-second windows during pretraining and task-specific windows of 4–30 seconds downstream (Table 8). Each window contains $C$ available electrodes and $T$ patches per electrode. A token $\boldsymbol { i } = \left( c _ { i } , t _ { i } \right)$ is a channel–time patch on the grid $\Omega = \mathcal { C } \times \mathcal { T }$ , where $\bar { \mathcal { C } }$ and $\tau$ index electrodes and temporal patches, respectively. The full grid contains $C T$ tokens (231 for the pretraining grid of 21 electrodes and 11 patches). Each electrode has a known 3D head coordinate $\bar { p } _ { c } \in \mathbb { R } ^ { 3 }$ from the standard 10–20 montage [Jasper, 1958], and $x _ { i } \in \mathbb { R } ^ { d }$ denotes the token embedding after patch projection and positional encoding.

Dense attention uses shared $Q , K , V$ projections across all tokens:

$$
\mathrm { A t t n } ( x ) _ { i } = \sum _ { j \in \Omega } a _ { i j } V x _ { j } , \qquad a _ { i j } \propto \exp \left( \frac { q _ { i } ^ { \top } k _ { j } } { \sqrt { d _ { h } } } \right) ,
$$

giving $( C T ) ^ { 2 }$ token pairs per layer. Dense-L widens this baseline to match AXON’s parameter count(by increasing dimensions) (Appendix C.2); And Divided-ST applies temporal then spatial attention within each block [Bertasius et al., 2021] (Appendix F.2).

## 2.2 Factorized attention

The core idea is to replace the dense attention operator with a weighted mixture of two axis-restricted operators, each attending within one axis of the token grid $\Omega = \bar { \mathcal { C } } \times \mathcal { T }$ : the temporal path over time within a channel, the spatial path over channels within a time step.

Let $T ( x ) _ { i }$ <sub>i</sub> denote the temporal attention output and $S ( x ) .$ <sub>i</sub> the spatial attention output for token i (defined below). A single AXON block computes:

$$
y _ { i } \ = \ \lambda T ( x ) _ { i } \ + \ ( 1 - \lambda ) S ( x ) _ { i } , \qquad \lambda \in ( 0 , 1 ) ,
$$

where λ controls the axis mixture. In the simplest variant (AXON-Fixed), $\lambda = \sigma ( \ell )$ is the sigmoid of a single learnable scalar ℓ per layer, shared across all tokens.

The temporal and spatial paths use separate QKV and output projections $( Q ^ { ( T ) } , K ^ { ( T ) } , V ^ { ( T ) } , O ^ { ( T ) } )$ and $( Q ^ { ( \bar { S } ) } , K ^ { ( S ) } , \bar { V ^ { ( S ) } } , \bar { O ^ { ( S ) } } )$ , allowing each axis to specialise its feature space. The full cost analysis is in Appendix C.2. Although no token sees all others in one layer, on a full channel-time grid any two tokens are connected after two layers: one temporal step and one spatial step (Appendix E.1).

![](images/c6034adee23a7ffae5251273031a318d26df38f9dff089fab3465220f1909bf2.jpg)  
Figure 1: (a) Channel–time grid with query token i (red), temporal neighbours T(i) (blue), and spatial neighbours $s ( i )$ (orange); dense attention uses the full grid. (b) An AXON block mixes both paths using weights from an MLP applied to $\operatorname { s g } ( x _ { i } )$ . The encoder stacks 22 blocks without a dense global branch.

Temporal path. The temporal neighbourhood of token i is all tokens on the same channel:

$$
\mathcal { T } ( i ) = \{ j \in \Omega : c _ { j } = c _ { i } \} , \quad | \mathcal { T } ( i ) | = T .
$$

Each channel group of $T$ tokens is processed as an independent sequence; channels do not interact through the temporal path. This gives each token access to the full temporal context of its electrode: short transients, rhythm-band oscillations, and longer event windows are all contained in $\tau ( i )$

Spatial path. The spatial neighbourhood of token i is all tokens at the same time step:

$$
S ( i ) = \{ j \in \Omega : t _ { j } = t _ { i } \} , \quad | S ( i ) | = C .
$$

$T ( x ) _ { i }$ and $S ( x ) _ { i }$ are standard multi-head attention restricted to these neighbourhoods.

The spatial path is not given any explicit information about electrode coordinates; electrode positions enter only through the positional encoding. (Appendix F).

## 2.3 Token-conditioned anisotropy gate

A fixed mixing weight shared by all tokens imposes one mixture on all of them. The token-conditioned gate lets each token choose its own.

AXON-TokenGated replaces the shared layer weight with token-conditioned mixing:

$$
( \alpha _ { i } , \beta _ { i } ) = \mathrm { s o f t m a x } ( g ( \mathrm { s g } ( x _ { i } ) ) / \tau ) .
$$

The two-layer MLP $g : \mathbb { R } ^ { d }  \mathbb { R } ^ { 2 }$ has hidden width $d / 4$ , GELU activation, and a zero-initialised output layer; sg refers to stop gradient. We anneal τ from 2.0 to 1.0 over the first 1500 steps; higher values keep the weights closer to equal (Appendix C).

The block output is the soft-gated mixture of the temporal and spatial paths:

$$
y _ { i } = \alpha _ { i } T ( x ) _ { i } + \beta _ { i } S ( x ) _ { i } , \qquad \alpha _ { i } + \beta _ { i } = 1 ,
$$

where $T$ and $S$ are the temporal and spatial paths defined in Section 2.2. Each layer computes a gate for every token on every forward pass. Because $x _ { i }$ includes positional encoding, weights can vary with content, channel–time position, layer, and input sample. AXON-TokenGated uses one full-length temporal window and no global path; final AXON replaces $T$ with the two-window path (Section 2.5). Gate interventions are in Section 3.2 and Appendix H.

## 2.4 Global path

A natural extension adds a third path: dense attention over all tokens, with its own projections. The gate then predicts three weights that sum to one:

$$
y _ { i } = \alpha _ { i } T ( x ) _ { i } + \beta _ { i } S ( x ) _ { i } + \gamma _ { i } G ( x ) _ { i } , \qquad \alpha _ { i } + \beta _ { i } + \gamma _ { i } = 1 ,
$$

Table 1: Main EEG results — Linear Probe (LP, frozen encoder) and Full Finetune (FT). Balanced accuracy, mean ± std over 3 downstream seeds; subject-disjoint splits shared across all models.
<table><tr><td>Model</td><td>motor</td><td>workload</td><td>hmc</td><td>siena</td><td>adftd</td><td>bcic</td><td>Mean LP</td></tr><tr><td>Dense</td><td> $0 . 3 5 3 { \pm } . 0 0 2$ </td><td> $0 . 6 1 2 { \scriptstyle \pm . 0 2 1 }$ </td><td> $0 . 6 6 0 { \pm } . 0 0 3$ </td><td> $\mathbf { 0 . 8 7 6 { \scriptstyle \pm . 0 0 7 } }$ </td><td> $0 . 5 1 6 { \pm } . 0 3 4$ </td><td> $0 . 2 8 2 { \pm } . 0 0 8$ </td><td>0.550</td></tr><tr><td>Dense-L</td><td> $0 . 3 8 4 { \pm } . 0 0 3$ </td><td> $0 . 6 4 8 { \scriptstyle \pm . 0 1 1 }$ </td><td> $0 . 6 5 3 { \pm } . 0 0 2$ </td><td> $0 . 8 2 8 { \scriptstyle \pm . 0 0 0 }$ </td><td> $0 . 5 3 1 { \pm } . 0 0 4$ </td><td> $0 . 2 8 8 { \pm } . 0 0 1$ </td><td>0.555</td></tr><tr><td>Divided-ST</td><td> $0 . 3 9 3 { \scriptstyle \pm . 0 0 1 }$ </td><td> $0 . 6 2 1 { \pm } . 0 0 8$ </td><td> $0 . 6 4 9 { \pm } . 0 0 2$ </td><td> $0 . 8 7 2 { \scriptstyle \pm . 0 0 3 }$ </td><td> $0 . 5 2 3 { \pm } . 0 2 8$ </td><td> $\mathbf { 0 . 2 9 9 } 2 2 . 0 0 2$ </td><td>0.560</td></tr><tr><td>AXON-Fixed</td><td> $\mathrm { 0 . 3 7 0 { \pm } . 0 0 3 }$ </td><td> $\mathbf { 0 . 6 7 4 { \scriptstyle \pm . 0 0 7 } }$ </td><td> $0 . 6 6 6 { \pm } . 0 0 0$ </td><td> $0 . 8 4 1 { \scriptstyle \pm . 0 0 1 }$ </td><td> $0 . 5 0 2 { \scriptstyle \pm . 0 0 8 }$ </td><td> $0 . 2 9 6 { \pm } . 0 0 3$ </td><td>0.558</td></tr><tr><td>AXON-TokenGated</td><td> $0 . 4 4 7 { \scriptstyle \pm . 0 0 1 }$ </td><td> $0 . 6 6 2 { \scriptstyle \pm . 0 1 2 }$ </td><td> $\mathbf { 0 . 6 7 3 { \scriptstyle \pm . 0 0 1 } }$ </td><td> $0 . 8 5 5 { \pm } . 0 0 0$ </td><td> $0 . 5 3 1 { \pm } . 0 1 4$ </td><td> $0 . 2 9 8 { \pm } . 0 0 4$ </td><td>0.578</td></tr><tr><td>AXON</td><td> $\mathbf { 0 . 4 5 5 { \pm } . 0 0 5 }$ </td><td> $\mathbf { 0 . 6 7 3 { \scriptstyle \pm . 0 0 9 } }$ </td><td> $0 . 6 5 0 { \pm } . 0 0 2$ </td><td> $0 . 8 6 7 { \scriptstyle \pm . 0 0 1 }$ </td><td> $\mathbf { 0 . 5 3 8 } { \pm . 0 0 6 }$ </td><td> $0 . 2 9 2 { \scriptstyle \pm . 0 0 7 }$ </td><td>0.579</td></tr><tr><td colspan="8">Full Finetune</td></tr><tr><td>Model</td><td>motor</td><td>workload</td><td>hmc</td><td>siena</td><td>adftd</td><td>bcic</td><td>Mean FT</td></tr><tr><td>Dense</td><td> $0 . 5 6 1 { \pm } . 0 1 5$ </td><td> $0 . 6 4 8 { \pm } . 0 0 8$ </td><td> $\mathbf { 0 . 7 3 6 { \pm } . 0 0 3 }$ </td><td> $0 . 8 5 3 { \scriptstyle \pm . 0 0 1 }$ </td><td> $0 . 5 0 4 { \scriptstyle \pm . 0 4 6 }$ </td><td> $0 . 3 3 5 { \pm } . 0 3 8$ </td><td>0.606</td></tr><tr><td>Dense-L</td><td> $0 . 6 2 3 { \scriptstyle \pm . 0 0 1 }$ </td><td> $0 . 6 1 3 { \pm } . 0 1 2$ </td><td> $0 . 7 2 8 { \pm } . 0 0 5$ </td><td> $0 . 8 2 6 { \scriptstyle \pm . 0 0 9 }$ </td><td> $0 . 5 3 6 { \pm } . 0 1 1$ </td><td> $\mathbf { 0 . 4 6 9 } \pm . \mathbf { 0 5 1 }$ </td><td>0.633</td></tr><tr><td>Divided-ST</td><td> $0 . 6 0 0 { \scriptstyle \pm . 0 0 6 }$ </td><td> $0 . 6 6 1 { \scriptstyle \pm . 0 5 7 }$ </td><td> $0 . 7 2 0 { \scriptstyle \pm . 0 0 3 }$ </td><td> $0 . 8 5 3 { \scriptstyle \pm . 0 1 1 }$ </td><td> $0 . 5 5 5 { \pm } . 0 2 1$ </td><td> $0 . 3 7 9 { \scriptstyle \pm . 0 1 7 }$ </td><td>0.628</td></tr><tr><td>AXON-Fixed</td><td> $0 . 6 1 9 { \pm } . 0 0 9$ </td><td> $\mathbf { 0 . 7 3 5 { \pm } . 0 0 8 }$ </td><td> $0 . 7 3 1 { \pm } . 0 0 6$ </td><td> $0 . 8 5 3 { \pm } . 0 1 3$ </td><td> $0 . 5 0 1 { \scriptstyle \pm . 0 2 9 }$ </td><td> $0 . 4 0 3 { \pm } . 0 2 0$ </td><td>0.640</td></tr><tr><td>AXON-TokenGated</td><td> $\mathbf { 0 . 6 3 0 { \pm } . 0 1 2 }$ </td><td> $0 . 6 7 0 { \scriptstyle \pm . 0 3 3 }$ </td><td> $0 . 7 2 9 { \pm } . 0 0 3$ </td><td> $\mathbf { 0 . 8 6 7 \pm . 0 1 3 }$ </td><td> $0 . 5 6 3 { \scriptstyle \pm . 0 0 5 }$ </td><td> $0 . 3 9 9 { \scriptstyle \pm . 0 4 6 }$ </td><td>0.643</td></tr><tr><td>AXON</td><td> $0 . 6 2 5 { \pm } . 0 0 5$ </td><td> $0 . 6 8 5 { \pm } . 0 0 6$ </td><td> $0 . 7 3 2 { \pm } . 0 0 9$ </td><td> $0 . 8 5 9 { \scriptstyle \pm . 0 0 7 }$ </td><td> $\mathbf { 0 . 6 0 4 \pm . 0 2 2 }$ </td><td> $0 . 4 2 8 { \pm } . 0 3 3$ </td><td>0.656</td></tr></table>

where $\begin{array} { r } { G ( x ) _ { i } = \sum _ { j \in \Omega } g _ { i j } V _ { G } x _ { j } } \end{array}$ . This is AXON-withGlobal. The motivation was that clinical tasks may benefit from direct all-to-all context in a single layer, which the two axis paths only reach after two layers. AXON does not use this path; Section 3.2 reports its effect.

## 2.5 Two temporal windows

EEG carries information at different time scales: a motor-imagery response or a seizure onset develops over a few seconds, while a sleep stage or a cognitive state lasts the whole window [Pfurtscheller and Lopes da Silva, 1999]. A temporal path that always sees the whole window can in principle use both, but it has to learn to separate a short event from the slow background on its own. We make the separation explicit by giving the temporal path two windows: a short one that sees only five consecutive patches, and a long one that sees the whole visible window. Each layer learns how much to use each:

$$
T _ { \mathrm { m s } } ( x ) _ { i } = \lambda ^ { s } T _ { \mathrm { s h o r t } } ( x ) _ { i } + ( 1 - \lambda ^ { s } ) T _ { \mathrm { l o n g } } ( x ) _ { i } , \qquad \lambda ^ { s } \in [ 0 , 1 ] ,
$$

with one $\lambda ^ { s }$ per layer, initialised to favour the long window so that early reconstruction is easy. This choice is separate from the axis gate: the axis gate sets the temporal/spatial split for each token, and $\lambda ^ { s }$ sets how far in time the temporal path looks. AXON, the final model, is AXON-TokenGated with this two-window temporal path and no global path.

## 3 Experiments

## 3.1 Setup

All encoders are pretrained with masked autoencoding [He et al., 2022] on pooled TUH-EEG [Obeid and Picone, 2016], I-CARE [Amorim et al., 2023], and internal EEG data for 50 epochs with batch size 4096. Recordings are mapped to 21 canonical 10–20 electrode positions and divided into 10- second windows with $T = 1 1$ patches per channel. We mask 55% of tokens; the encoder processes only visible tokens, and a small decoder reconstructs masked patches. Missing channels are excluded from tokenization and reconstruction loss; attention masks cover only present tokens.

We evaluate on six public datasets covering motor imagery (motor, bcic), cognitive workload (workload), sleep staging (hmc), seizure detection (siena), and dementia diagnosis (adftd). We report balanced accuracy (BAC) on subject-disjoint splits using linear probing (LP; shallow MLP head) and full fine-tuning (FT). For each model, Table 1 reports the checkpoint selected by the highest mean LP on held-out validation subjects, rather than the final epoch. Dataset summaries appear in Table 8; implementation, training-budget analysis, and evaluation details are in Appendices C and D.

## 3.2 Results

AXON achieves the highest Mean LP (0.579) and Mean FT (0.656) in Table 1. It improves LP and FT over Dense by 2.9 and 5.0 balanced-accuracy points, respectively. The gains remain over Dense-L (2.4 and 2.3 points) and Divided-ST (1.9 and 2.8 points). The largest task-level gains over Dense are on motor LP (+10.2 points) and adftd FT (+10.0 points).

Table 2: Controlled audio spectrogram results — full AudioSet-2M pretraining. All variants use identical optimizer, schedule, and compute budget; only the attention operator differs.
<table><tr><td>Model</td><td>AudioSet FT mAP</td><td>AudioSet LP mAP</td><td>ESC-50 LP Acc</td><td>SC LP Acc</td></tr><tr><td>Dense</td><td> $1 1 . 5 9 { \pm } 0 . 2 5 $ </td><td>4.08±0.01</td><td> $4 5 . 2 5 { \pm } 0 . 9 4 $ </td><td>19.53±0.07</td></tr><tr><td>Fixed gate (AXON-Fixed)</td><td> $1 3 . 7 5 { \pm } 0 . 1 6$ </td><td> $4 . 0 1 { \pm } 0 . 0 2$ </td><td> $4 4 . 8 3 { \pm } 0 . 3 1 $ </td><td> $\mathbf { 2 2 . 0 1 } \pm \mathbf { 0 . 0 6 }$ </td></tr><tr><td>Token gate (AXON-TokenGated)</td><td> $1 3 . 8 2 { \pm } 0 . 2 5 $ </td><td> $4 . 8 4 { \pm } 0 . 0 3$ </td><td> $4 5 . 7 5 { \pm } 0 . 7 1 $ </td><td> $2 1 . 9 6 { \pm } 0 . 1 5$ </td></tr><tr><td>Token + Global (AXON-withGlobal)</td><td> $\mathbf { 1 4 . 2 9 { \scriptstyle \pm 0 . 0 5 } }$ </td><td>5.08±0.04</td><td> $\mathbf { 4 7 . 1 7 \pm 0 . 9 2 }$ </td><td> $2 1 . 8 7 { \pm } 0 . 1 6$ </td></tr></table>

Three further controls show that the gain is not simply from attending to fewer tokens, and not simply from position information: restricting the spatial path to each electrode’s nearest neighbours drops Mean LP to 0.510; the trained dense model still attends to pairs that share neither electrode nor time step (Table 16); and adding 2D rotary position embeddings [Su et al., 2024] to the dense model gains only +0.008 (Appendix F). The ranking is stable at every pretraining budget we tested (Appendix D.2). A variant that biases the temporal path toward nearby time steps (AXON-Decay) matches AXON on Mean LP but lowers Mean FT (Appendix F.1). Appendix H shows what the gate learns and tests whether the model depends on it. AXON-withGlobal, the variant with a third dense path (Section 2.4), underperforms both AXON and AXON-Fixed on Mean LP and FT (Table 11) despite reaching comparable reconstruction loss (Figure 2). Its first-layer global weight is $\gamma = 0 . 7 2 4$ , leaving about a quarter for the axis paths. We hypothesise that global averaging provides a reconstruction shortcut that weakens the learning of axis-specific features (Appendix G).

Comparison with released EEG foundation models. Under the same six-task protocol, AXON has the highest mean scores among six released EEG foundation models: 0.579 vs. 0.527 Mean LP and 0.656 vs. 0.614 Mean FT against the next-best, REVE (Appendix B).

## 4 Audio Spectrogram Experiments

We apply the same axis-factorization principle to time–frequency spectrograms, comparing four attention variants under the AudioMAE pretraining recipe [Huang et al., 2022]. A spectrogram is a grid too: one axis is time, the other is frequency, and a token is one frequency band over one time frame. The temporal path attends across time within a frequency band; the spatial path becomes a frequency path and attends across frequency bands within a time frame. The full-scale experiment uses AudioSet-2M corpus [Gemmeke et al., 2017] (∼2M clips); smaller-scale results and the comparison with published AudioMAE are in Appendix J. Downstream evaluation uses AudioSet, ESC-50 [Piczak, 2015] and SpeechCommands (SC) [Warden, 2018]; AudioSet is multi-label, so we report mean average precision (mAP), and the other two report accuracy.

All factorized variants improve over Dense on AudioSet FT and SpeechCommands LP (Table 2). Unlike EEG, Token+Global achieves the highest scores on three of the four metrics. On AudioSet FT and SpeechCommands LP this advantage holds at all three pretraining scales we tested, 18K, 200K and 2M clips (Tables 18 and 19). The audio experiment therefore extends the factorization result while showing that the global branch is not uniformly detrimental across the tested settings.

## 5 Conclusion

AXON combines temporal and spatial attention through learned soft mixing. Across six subjectdisjoint EEG tasks, it improves mean balanced accuracy over dense and parameter-matched dense baselines under Linear Probing and full fine-tuning. Gate interventions show larger performance drops from removing either axis or using hard routing than from replacing token gates with layer means. Axis factorization extends to audio, a second axis-structured modality. These results support a simple design rule for structured signals: align attention with the signal’s axes and let the model learn how much to use each.

## References

Diego Alvarez-Estevez and Roselyne Rijsman. Haaglanden Medisch Centrum sleep staging database. PhysioNet, 2022. Version 1.1.

Edilberto Amorim, Wei-Long Zheng, Mohammad Ghassemi, Mahsa Aghaeeaval, Pradyot Kandhare, Vishal Karukonda, Jong Woo Lee, Susan T. Herman, Adithya Sivaraju, Nicolas Gaspard, Jeannette Hofmeijer, Michel J. A. M. van Putten, Reza Sameni, Matthew A. Reyna, Gari D. Clifford, and M. Brandon Westover. The International Cardiac Arrest Research Consortium electroencephalography database. Critical Care Medicine, 2023. doi: 10.1097/CCM.0000000000006074.

Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Luciˇ c, and Cordelia Schmid.´ ViViT: A video vision transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In Proceedings of the International Conference on Machine Learning (ICML), 2021.

Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15750–15758, 2021.

Paolo Detti. Siena scalp EEG database. PhysioNet, 2020. Version 1.0.0.

Yassine El Ouahidi, Jonathan Lys, Philipp Tholke, Nicolas Farrugia, Bastien Pasdeloup, Vincent¨ Gripon, Karim Jerbi, and Giulia Lioi. REVE: A foundation model for EEG – adapting to any setup with large-scale pretraining on 25,000 subjects. In Advances in Neural Information Processing Systems 38, 2025. arXiv:2510.21585.

Jort F. Gemmeke, Daniel P. W. Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R. Channing Moore, Manoj Plakal, and Marvin Ritter. Audio set: An ontology and human-labeled dataset for audio events. In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 776–780, 2017. doi: 10.1109/ICASSP.2017.7952261.

Ary L. Goldberger, Lu´ıs A. N. Amaral, Leon Glass, Jeffrey M. Hausdorff, Plamen Ch. Ivanov, Roger G. Mark, Joseph E. Mietus, George B. Moody, Chung-Kang Peng, and H. Eugene Stanley. PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation, 101(23):e215–e220, 2000. doi: 10.1161/01.CIR.101.23.e215.

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. Masked ´ autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16000–16009, 2022.

Byeongho Heo, Song Park, Dongyoon Han, and Sangdoo Yun. Rotary position embedding for vision transformer. In European Conference on Computer Vision (ECCV), 2024. arXiv:2403.13298.

Jonathan Ho, Nal Kalchbrenner, Dirk Weissenborn, and Tim Salimans. Axial attention in multidimensional transformers. arXiv preprint arXiv:1912.12180, 2019.

Po-Yao Huang, Hu Xu, Juncheng Li, Alexei Baevski, Michael Auli, Wojciech Galuba, Florian Metze, and Christoph Feichtenhofer. Masked autoencoders that listen. In Advances in Neural Information Processing Systems 35, 2022.

Herbert H. Jasper. The ten-twenty electrode system of the International Federation. Electroencephalography and Clinical Neurophysiology, 10:371–375, 1958.

Weibang Jiang, Liming Zhao, and Bao-liang Lu. Large brain model for learning generic representations with tremendous EEG data in BCI. In The Twelfth International Conference on Learning Representations, 2024.

Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In Proceedings of the 36th International Conference on Machine Learning (ICML), volume 97 of Proceedings of Machine Learning Research, pages 3519–3529, 2019.

Wei Lun Lim, Olga Sourina, and Lipo Wang. STEW: Simultaneous task EEG workload dataset. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 26(11):2106–2114, 2018. doi: 10.1109/TNSRE.2018.2872924.

Andreas Miltiadous, Katerina D. Tzimourta, Theodora Afrantou, Panagiotis Ioannidis, Nikolaos Grigoriadis, Dimitrios G. Tsalikakis, Pantelis Angelidis, Markos G. Tsipouras, Euripidis Glavas, Nikolaos Giannakeas, and Alexandros T. Tzallas. A dataset of scalp EEG recordings of Alzheimer’s Disease, Frontotemporal Dementia and healthy subjects from routine EEG. Data, 8(6):95, 2023. doi: 10.3390/data8060095.

Iyad Obeid and Joseph Picone. The Temple University Hospital EEG data corpus. Frontiers in Neuroscience, 10:196, 2016. doi: 10.3389/fnins.2016.00196.

Gert Pfurtscheller and F. H. Lopes da Silva. Event-related EEG/MEG synchronization and desynchronization: basic principles. Clinical Neurophysiology, 110(11):1842–1857, 1999. doi: 10.1016/S1388-2457(99)00141-8.

Karol J. Piczak. ESC: Dataset for environmental sound classification. In Proceedings ofthe 23rd ACM International Conference on Multimedia, pages 1015–1018, 2015. doi: 10.1145/2733373.2806390.

Olivier Roy and Martin Vetterli. The effective rank: A measure of effective dimensionality. In 2007 15th European Signal Processing Conference (EUSIPCO), pages 606–610, 2007.

Gerwin Schalk, Dennis J. McFarland, Thilo Hinterberger, Niels Birbaumer, and Jonathan R. Wolpaw. BCI2000: A general-purpose brain-computer interface (BCI) system. IEEE Transactions on Biomedical Engineering, 51(6):1034–1043, 2004. doi: 10.1109/TBME.2004.827072.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. doi: 10.1016/j.neucom.2023.127063. arXiv:2104.09864 (2021).

Michael Tangermann, Klaus-Robert Muller, Ad Aertsen, Niels Birbaumer, Christoph Braun, Clemens¨ Brunner, Robert Leeb, Carsten Mehring, Kai J. Miller, Gernot R. Muller-Putz, Guido Nolte, Gert¨ Pfurtscheller, Hubert Preissl, Gerwin Schalk, Alois Schlogl, Carmen Vidaurre, Stephan Waldert,¨ and Benjamin Blankertz. Review of the BCI Competition IV. Frontiers in Neuroscience, 6:55, 2012. doi: 10.3389/fnins.2012.00055.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems 30, 2017.

Guangyu Wang, Wenchao Liu, Yuhong He, Cong Xu, Lin Ma, and Haifeng Li. EEGPT: Pretrained transformer for universal and reliable representation of EEG signals. In Advances in Neural Information Processing Systems 37, pages 39249–39280, 2024.

Jiquan Wang, Sha Zhao, Zhiling Luo, Yangxuan Zhou, Haiteng Jiang, Shijian Li, Tao Li, and Gang Pan. CBraMod: A criss-cross brain foundation model for EEG decoding. In The Thirteenth International Conference on Learning Representations, 2025.

Pete Warden. Speech commands: A dataset for limited-vocabulary speech recognition. arXiv preprint arXiv:1804.03209, 2018.

Chaoqi Yang, M. Brandon Westover, and Jimeng Sun. BIOT: Biosignal transformer for crossdata learning in the wild. In Advances in Neural Information Processing Systems 36, pages 78240–78260, 2023.

Yuchen Zhou, Jiamin Wu, Zichen Ren, Zhouheng Yao, Weiheng Lu, Kunyu Peng, Qihao Zheng, Chunfeng Song, Wanli Ouyang, and Chao Gou. CSBrain: A cross-scale spatiotemporal brain foundation model for EEG decoding. In Advances in Neural Information Processing Systems 38, 2025. arXiv:2506.23075.

## Appendix

## A Background & mannas.ai Context

Electroencephalography (EEG) measures electrical potential differences across a sparse array of scalp electrodes at millisecond resolution. Each recording is a matrix of C channels by T time samples. The signal is low-amplitude $( \sim 1 0 { - } 1 0 0 \mu \mathrm { v } )$ , contaminated by muscle artifacts, eye movements, and ambient noise, and varies across subjects and sessions. Despite this noise, EEG encodes clinically and cognitively meaningful structure: motor imagery induces localized mu-rhythm desynchronization over sensorimotor cortex; sleep staging depends on broadband slow-wave and spindle patterns; epileptic events manifest as sharp, spatially propagating discharges.

Two properties make EEG a natural proving ground for anisotropic attention. First, the token mannas.ai is explicitly two-dimensional: a patch token at (c, t) has a known spatial identity (electrode c with head coordinate) and a temporal identity (time window t). Second, different tasks depend on structurally different contexts: motor imagery requires preserving localized, lateralized, temporally precise structure; clinical classification benefits from broader spatial and temporal integration. A single isotropic attention operator may be a poor shared prior across tasks with such different channel-time structure.

## B External EEG Foundation Models

## B.1 External EEG baselines — Linear Probe

We compare AXON against six published EEG foundation models under a standardised evaluation protocol: the same six tasks, the same disjoint subject splits, the same balanced-accuracy metric, and the same three downstream seeds as our internal ablations. All external models are evaluated from their officially released pretrained checkpoints. Models are evaluated with both linear probing (LP, frozen encoder) and fine-tuning (FT, all weights updated).

Table 3: External EEG foundation model comparison — Linear Probe (LP, frozen encoder). Format: balanced accuracy. Bold = column best. Our dense baseline is included as a within-setup reference. All models evaluated under the same 6-task protocol with identical subject splits.
<table><tr><td>Model</td><td>motor</td><td>workload</td><td>hmc</td><td>siena</td><td>adftd</td><td>bcic</td><td>Mean</td></tr><tr><td>EEGPT [Wang et al., 2024]</td><td>0.381±.015</td><td>0.574±.020</td><td>0.665±.006</td><td>0.804±.019</td><td>0.393±.034</td><td>0.281±.007</td><td>0.516</td></tr><tr><td>LaBraM [Jiang et al., 2024]</td><td>0.268±.012</td><td>0.500±.000</td><td>0.381±.019</td><td>0.500±.000</td><td>0.309±.019</td><td>0.285±.022</td><td>0.374</td></tr><tr><td>CBraMod [Wang et al., 2025]</td><td>0.259±.013</td><td>0.500±.000</td><td>0.510±.001</td><td>0.619±.006</td><td>0.358±.002</td><td>0.270±.010</td><td>0.419</td></tr><tr><td>BIOT [Yang et al., 2023]</td><td>0.284±.008</td><td>0.577±.081</td><td>0.644±.003</td><td>0.610±.025</td><td>0.492±.033</td><td>0.261±.010</td><td>0.478</td></tr><tr><td>CSBrain [Zhou et al., 2025]</td><td>0.272±.005</td><td>0.500±.000</td><td>0.568±.002</td><td>0.500±.000</td><td>0.364±.014</td><td>0.270±.009</td><td>0.412</td></tr><tr><td>REVE [El Ouahidi et al., 2025]</td><td>0.315±.003</td><td>0.709±.035</td><td>0.653±.006</td><td>0.688±.033</td><td>0.532±.052</td><td>0.267±.012</td><td>0.527</td></tr><tr><td>Dense</td><td>0.353±.002</td><td>0.612±.021</td><td> $0 . 6 6 0 { \scriptstyle \pm . 0 0 3 }$ </td><td>0.876±.007</td><td> $0 . 5 1 6 { \pm } . 0 3 4$ </td><td>0.282±.008</td><td>0.550</td></tr><tr><td>AXON</td><td>0.455±.005</td><td>0.673±.009</td><td>0.650±.002</td><td>0.867±.001</td><td> $\mathbf { 0 . 5 3 8 } { \pm . 0 0 6 }$ </td><td>0.292±.007</td><td>0.579</td></tr></table>

AXON achieves the highest Mean LP and Mean FT among all eight models, ahead of the next-best external model (REVE) by +5.2 points LP (+9.9% relative) and +4.2 points FT (+6.8% relative). The full fine-tuning comparison is in Appendix Table 4. Note that our own dense baseline already outperforms all six external models on mean LP. So part of AXON’s gap to the external models comes from our training setup rather than from the attention design, and AXON’s gain over that dense baseline (Table 1) is measured on top of it. The architectural claim rests on that controlled comparison, not on this table.

Task-level analysis reveals that motor imagery exhibits the largest gap between AXON and external models (+19% LP over the best competitor). This fits the motivation for preserving axis structure: motor imagery depends on a left/right difference in mu and beta rhythms over the sensorimotor electrodes. This comparison does not, however, show which features account for the gain. REVE retains advantages on workload and siena FT, suggesting its architecture provides broader temporal integration suited for sustained cognitive states.

Table 4: External EEG foundation model comparison — Full Finetune (FT). Bold = column best. All models evaluated under the same 6-task protocol with identical subject splits.
<table><tr><td>Model</td><td>motor</td><td>workload</td><td>hmc</td><td>siena</td><td>adftd</td><td>bcic</td><td>Mean</td></tr><tr><td>EEGPT [Wang et al., 2024]</td><td>0.513±.007</td><td> $0 . 6 6 8 { \scriptstyle \pm . 0 1 5 }$ </td><td> $0 . 7 1 2 { \scriptstyle \pm . 0 0 5 }$ </td><td> $0 . 7 9 5 { \pm } . 0 3 1$ </td><td> $0 . 4 0 4 { \pm } . 0 2 2$ </td><td>0.280±.041</td><td>0.562</td></tr><tr><td>LaBraM [Jiang et al., 2024]</td><td>0.249±.001</td><td> $0 . 5 0 1 { \scriptstyle \pm . 0 0 2 }$ </td><td> $0 . 6 4 5 { \pm } . 0 1 1$ </td><td>0.813±.018</td><td> $0 . 2 9 0 { \pm } . 0 6 0$ </td><td>0.259±.009</td><td>0.460</td></tr><tr><td>CBraMod [Wang et al., 2025]</td><td>0.426±.015</td><td> $0 . 5 5 4 { \pm } . 0 4 4$ </td><td>0.709±.011</td><td>0.847±.036</td><td> $0 . 3 1 9 { \scriptstyle \pm . 0 2 6 }$ </td><td>0.303±.031</td><td>0.526</td></tr><tr><td>BIOT [Yang et al., 2023]</td><td>0.372±.013</td><td>0.570±.092</td><td>0.710±.005</td><td>0.747±.026</td><td>0.449±.038</td><td>0.314±.040</td><td>0.527</td></tr><tr><td>CSBrain [Zhou et al., 2025]</td><td>0.570±.023</td><td>0.605±.028</td><td>0.705±.010</td><td>0.782±.036</td><td>0.432±.021</td><td>0.350±.015</td><td>0.574</td></tr><tr><td>REVE [El Ouahidi et al., 2025]</td><td>0.612±.004</td><td>0.703±.011</td><td>0.724±.002</td><td>0.863±.033</td><td>0.460±.050</td><td>0.322±.014</td><td>0.614</td></tr><tr><td>Dense</td><td>0.561±.015</td><td>0.648±.008</td><td>0.736±.003</td><td>0.853±.001</td><td>0.504±.046</td><td>0.335±.038</td><td>0.606</td></tr><tr><td>AXON</td><td>0.625±.005</td><td>0.685±.006</td><td>0.732±.009</td><td>0.859±.007</td><td>0.604±.022</td><td>0.428±.033</td><td>0.656</td></tr></table>

## B.2 External EEG baselines — Full Finetune

## C Extended Implementation Details & Pretraining

Pretraining corpus. The pretraining corpus pools four data sources (Table 5). Two are publicly available: the Temple University Hospital EEG Corpus (TUH-EEG) [Obeid and Picone, 2016] and the I-CARE dataset [Amorim et al., 2023]. Two are internal clinical EEG collections acquired under institutional ethics approval and de-identified before use; these are not publicly released but are described below to enable reproducibility assessment.

Table 5: Pretraining corpus composition. All sources are pooled into a single unlabeled pretraining set; no downstream task labels are used during pretraining. Internal sources are marked †.
<table><tr><td>Source</td><td>Subjects</td><td>Hours</td><td>Clinical context</td><td>Availability</td></tr><tr><td>TUH-EEG [Obeid and Picone, 2016]</td><td>~15,000</td><td>~25,000</td><td>Mixed clinical referrals</td><td>Public</td></tr><tr><td>I-CARE [Amorim et al., 2023]</td><td>~600</td><td>~33,000</td><td>Post-cardiac-arrest ICU</td><td>Public</td></tr><tr><td>Internal-A †</td><td>4,539</td><td>2,546</td><td>Routine clinical neurophysiology</td><td>Not released</td></tr><tr><td>Internal-B †</td><td>1,050</td><td>435</td><td>Multi-centre research EEG</td><td>Not released</td></tr></table>

Internal-A comprises routine clinical EEG recordings (resting-state, hyperventilation, and photic stimulation protocols) collected across hospital neurophysiology departments. Internal-B comprises multi-centre research EEG recordings acquired under a national research programme. Both internal datasets were recorded with standard 10-20 montage systems at sampling rates of 250–512 Hz and deidentified (all patient identifiers, dates, and institution codes removed) before inclusion. All internal data collection was conducted under institutional ethics board approval with informed consent or waiver of consent for retrospective de-identified use. Subject identities are verified disjoint across all four pretraining sources and all six downstream evaluation datasets.

Recordings span diverse acquisition settings with variable electrode configurations: systems range from compact 16-channel ambulatory devices to full 256-channel research amplifiers, and not every recording contains all standard 10-20 electrodes. We retain only recordings whose channel header resolves to a subset of the international 10-20 montage, then extract the available 10-20 electrode positions per recording. Because channel count varies across sources, we adopt C =21 as the representative value throughout this paper; this is the mode channel count across the retained pretraining corpus. Preprocessing: (1) resample to $f _ { s } = 2 0 0 \mathrm { H z } ;$ (2) notch filter at 50 and 60 Hz; (3) bandpass [0.5, 99.5] Hz; (4) per-channel z-score normalisation; (5) clip at ±15σ; (6) segment into non-overlapping 10-second windows.

We train encoders using masked autoencoding (MAE). EEG signal is divided into patches of 200 samples (1 s at $f _ { s } = 2 0 0 \mathrm { H z } )$ with a 20-sample overlap and 180-sample (0.9 s) stride between consecutive patch start positions, yielding $T = \bar { 1 } 1$ patches per channel per 10-second window. Each channel-time patch becomes a single token via a learnable linear projection. Tokens receive a split positional encoding: spatial coordinates $p _ { c } = ( x , y , z )$ (3D head positions in millimetres, standard 10-20 montage) are projected with a learned linear layer to produce $\mathrm { P E } _ { S } ;$ the temporal patch index receives a fixed sinusoidal encoding $\mathrm { P E } _ { T }$ . The two components are summed: $\mathrm { P E } ( i ) = \mathrm { P E } _ { S } ( c _ { i } ) +$ $\mathrm { P E } _ { T } ( t _ { i } )$ . The encoder processes only the visible tokens (55% masking ratio, spatiotemporal block masking with spatial radius 3.0 and temporal radius 3.0); masked token positions receive no encoder gradient. A lightweight 4-layer dense Transformer decoder takes the encoded visible tokens plus learned mask-slot embeddings and reconstructs all patches. Training minimises L1 reconstruction loss over masked patches plus an auxiliary pooled-attention reconstruction loss $( \lambda { = } 0 . 5 )$ . The auxiliary head applies cross-attention pooling over the concatenated outputs of all encoder MHA layers: a single learned query token attends over the layer-wise output tokens to produce a compact global representation. This pooled token is then repeated to match the number of masked positions, enriched with positional encodings, and passed through a 2-layer FFN to reconstruct the masked patches under a separate L1 loss. The total pretraining loss is $\mathcal { L } = \mathcal { L } _ { \mathrm { p r i m a r y } } + \lambda \cdot \mathcal { L } _ { \mathrm { a u x } }$

## C.1 Pretraining hyperparameters

Table 6: Pretraining hyperparameters (AXON). The dense baseline uses the same schedule and corpus; it differs only in the encoder attention operator.  
Hyperparameter Value   
Batch size 4096   
Epochs 50   
Peak LR $2 . 4 \times 1 0 ^ { - 4 }$   
LR schedule CosineAnnealingLR ${ \mathrm { : } } ( T _ { \operatorname* { m a x } } = 2 0 ) { \mathrm { : } }$   
Optimiser fused AdamW $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5 , \lambda = 0 . 0 5 )$   
Gradient clip (ℓ<sub>2</sub> norm) 1.0   
Gate τ warmup 2.0→1.0 over 1500 steps   
Mask ratio 0.55 (spatiotemporal block masking)   
Spatial block radius 3.0 patch indices (i.e., 3 electrode positions in the canonical 10-20 ordering)   
Temporal block radius 3.0 patch indices (i.e., 3 consecutive time patches, ≈2.7 s)   
Dropout mask ratio 0.3   
Coordinate noise σ 0.25   
Auxiliary loss weight 0.5   
Model dim d 512   
Encoder layers / heads 22 / 8   
Decoder layers 4 (dense, full attention)   
Precision bfloat16 (autocast)

Pretraining was conducted on a single compute node equipped with 8 NVIDIA H200 GPUs (143,771 MiB memory each, ≈1.1 TiB total GPU memory) and ≈2.2 TiB of host RAM.

## C.2 Parameter count and compute

AXON is not parameter-matched to the dense baseline. Replacing one dense attention operator with two separate full-width temporal and spatial operators adds approximately 26M parameters (22.5%).

Independent projections. The temporal and spatial paths use separate QKV and output projection matrices $( Q ^ { ( \bar { T } ) } , \dot { K } ^ { ( T ) } , V ^ { ( T ) } , O ^ { ( T ) } )$ and $( Q ^ { ( S ) } , K ^ { ( S ) } , V ^ { ( S ) } , O ^ { ( S ) } )$ . This is deliberate: the features needed to select which temporal patch to attend to (e.g., spectral power at a rhythm band) are different from those needed to select which electrode to attend to (e.g., lateralised activation). Shared projections would force both axes through the same feature bottleneck, limiting specialisation. The cost is a 2× parameter increase in the attention projections per layer relative to a single dense attention, but the number of pairwise attention interactions is reduced by a factor of ≈ $( { \check { C } } { \check { T } } ) / ( C + T )$ (from $( C T ) ^ { 2 } : 0 C T ^ { 2 } + \dot { T } C ^ { 2 } )$ .

To control for this, we trained a parameter-matched dense baseline with hidden dimension 568 (142.5M total parameters, within 0.5% of AXON’s 141.9M). This model uses identical pretraining (same data, optimizer, epochs, mask ratio) and differs only in hidden dimension.

Three observations address the parameter concern. First, the parameter-matched dense baseline (dim=568, 142.5M) uses nearly identical capacity to AXON (141.9M) with the same dense attention topology. It achieves Mean LP 0.555 and Mean FT 0.633: higher than the original dense baseline (0.550 / 0.606), demonstrating that the extra parameters provide some benefit, but still falling short of AXON by 2.4 points on Mean LP and 2.3 points on Mean FT (+4.3% and +3.6% relative). The AXON advantage persists after capacity matching. Second, AXON replaces global quadratic mixing with two axis-factorized operators, reducing pairwise attention interactions by ${ \sim } 7 \times$ per layer despite the added projections. Because the sequence length $( N = 2 3 1 )$ is small relative to $d = 5 1 2 ,$ the $O ( N d ^ { 2 } )$ projection costs dominate total FLOPs; the computational advantage of axis factorization therefore lies in the structured prior, not in raw speed. Third, the AXON-withGlobal variant has substantially more parameters (∼167M, adding a global QKV on top of temporal and spatial branches) and still underperforms AXON by 2.4 points on Mean LP and 3.7 points on Mean FT. If the gains were explained by parameter count, AXON-withGlobal should win. The parameter-matched dense and AXON-withGlobal comparisons together isolate the inductive bias, not a capacity effect.

Table 7: Parameter count, attention complexity, and downstream performance. Dense-L controls for the capacity difference by matching $\mathrm { { A X O N } ^ { \bullet } \mathrm { { s } } }$ total parameter count.
<table><tr><td>Model</td><td>Total params</td><td>Attn ops/layer‡</td><td>Mean LP</td><td>Mean FT</td></tr><tr><td>Dense</td><td>115.9M</td><td> $O ( ( C T ) ^ { 2 } ) = 5 3 , 3 6 1$ </td><td>0.550</td><td>0.606</td></tr><tr><td>Dense-L</td><td>142.5M</td><td> $O ( ( C T ) ^ { 2 } ) = 5 3 , 3 6 1$ </td><td>0.555</td><td>0.633</td></tr><tr><td>AXON-withGlobal</td><td>~167M</td><td> $> ( C T ) ^ { 2 }$ </td><td>0.555</td><td>0.619</td></tr><tr><td>AXON</td><td>141.9M</td><td> $O ( C T ^ { 2 } + \dot { T } C ^ { 2 } ) = 7 { , } 3 9 2$ </td><td>0.579</td><td>0.656</td></tr></table>

<sup>‡</sup>Computed for $C = 2 1$ channels, $T = 1 1$ time patches (mode of pretraining corpus): Dense $( C T ) ^ { 2 } = 5 3 { , } 3 6 1$ AXON $C T ^ { 2 } + T C ^ { 2 } = 7 { , } 3 9 2$ (≈7× less).

Empirical gate statistics. The gate diagnostic (22 layers × 6 datasets) shows that the trained gate is neither trivially uniform nor collapsed. Mean axis weights: $\alpha _ { \mathrm { m e a n } } = 0 . 4 4 1$ (temporal), $\beta _ { \mathrm { { m e a n } } } = \mathrm { { \bar { 0 } } } . 5 5 9$ (spatial). Mean gate entropy ratio: 0.932 (scale 0–log 2, where 1.0 is fully uniform). No layer falls below the collapse threshold of 0.40. Gate intervention experiments (Table 14) show that the dominant learned structure is a soft depth-dependent anisotropy schedule (layer-mean gates drop only −2.1% vs learned), while exact per-token gate assignment is a secondary effect (shuffled gates drop only −1.3%).

Stop-gradient. The stop-gradient on the gate input means the encoder receives no gradient from the routing decision; it is trained only by the reconstruction loss. We included it as a precaution so that the gate could not reshape the encoder’s representations during training. The ablation shows it makes no measurable difference: removing it, so that the gate reads the live representation instead of a detached copy [Chen and He, 2021], changes Mean LP by only −0.002 (0.577 vs. 0.579). It is a safe default, not a source of gain; our reported results do not hinge on it.

Temperature annealing. The gate softmax is divided by a temperature τ annealed from $\tau _ { \mathrm { s t a r t } } = 2 . 0$ to $\tau _ { \mathrm { e n d } } = 1 . 0$ over the first 1500 training steps:

$$
\tau _ { t } = \tau _ { \mathrm { s t a r t } } + ( \tau _ { \mathrm { e n d } } - \tau _ { \mathrm { s t a r t } } ) \cdot \mathrm { m i n } \Bigg ( 1 , \frac { t } { 1 5 0 0 } \Bigg ) .
$$

During warmup the gate is soft, allowing both axes to receive gradient from the MAE reconstruction loss. Both branches therefore develop useful representations before the gate sharpens. Gate entropy is monitored throughout warmup; a drop below 0.40 before warmup ends would indicate premature routing commitment and would be corrected by increasing $\tau _ { \mathrm { s t a r t } }$ or extending the warmup window.

## D Downstream Tasks & Evaluation Protocol

We evaluate with two protocols: linear probing (LP), where the encoder is frozen and only the classification head is trained, and full finetuning (FT), where all encoder weights are updated. The classification head is: AdaptiveAvgPool1d → Linear(512, 128) → ELU → Dropout(0.3) → Linear(128, K), where $K$ is the number of classes. The downstream optimiser is AdamW (weight decay 0.01) with cosine-annealing LR (peak $2 \times 1 0 ^ { - 4 }$ , min $2 \times 1 0 ^ { - 5 } )$ preceded by a 5-epoch linear warmup from $2 \times 1 0 ^ { - 6 }$ . 30 training epochs. The metric is balanced accuracy throughout. Subject splits are disjoint across train, validation, and test for every dataset; the same splits are shared by all models.

Table 8: Downstream evaluation datasets. All splits are subject-disjoint.
<table><tr><td>Dataset</td><td>Task</td><td>Classes</td><td>Window</td><td>Train split</td><td>Val / Test split</td></tr><tr><td>motor_mv_img [Schalk et al., 2004, Goldberger et al., 2000]</td><td>Motor imagery (L/R/both/foot)</td><td></td><td>4s</td><td>Subj. 0–69</td><td>Subj. 70–88 / 89–109</td></tr><tr><td>bcic_2a [Tangermann et al., 2012]</td><td>Motor imagery BCI (4 limbs)</td><td></td><td>4s</td><td>Subj. 1–5</td><td>Subj. 6–7 / 8–9</td></tr><tr><td>workload [Lim et al., 2018]</td><td>Cognitive workload (low/high)</td><td>4２5２３</td><td>4s</td><td>≈72% subj.</td><td>≈14% / 14% subj.</td></tr><tr><td>hmc [Alvarez-Estevez and Rijsman, 2022]</td><td>Sleep staging (5 stages)</td><td></td><td>30s</td><td>≈67.6% subj.</td><td>≈16.2% / 16.2% subj.</td></tr><tr><td>siena_scalp [Detti, 2020]</td><td>Seizure detection (ictal/interictal)</td><td></td><td>10s</td><td>≈70% subj.</td><td>≈15% / 15% subj.</td></tr><tr><td>adftd [Miltiadous et al., 2023]</td><td>Dementia (AD/FTD/Healthy)</td><td></td><td>10s</td><td>≈70% subj.</td><td>≈15% / 15% subj.</td></tr></table>

## D.1 Downstream tasks

We evaluate across six datasets covering BCI, cognitive, and clinical tasks. All evaluations use disjoint subject splits and balanced accuracy. Table 9 summarises the key discriminative challenge of each task and explains why isotropic attention is an unfavourable inductive bias.

Table 9: Downstream evaluation tasks.
<table><tr><td>Dataset</td><td>Family</td><td>Classes</td><td>Key discriminative structure</td></tr><tr><td>motor_mv_img</td><td>BCI</td><td>4</td><td>Lateralized mu/beta ERD at movement onset</td></tr><tr><td>bcic_2a</td><td>BCI</td><td>4</td><td>Fine lateralization, 4 limb classes</td></tr><tr><td>workload</td><td>Cognitive</td><td>2</td><td>Sustained frontal-parietal synchrony</td></tr><tr><td>hmc</td><td>Clinical</td><td>5</td><td>Broadband spectral stage transitions</td></tr><tr><td>siena_scalp</td><td>Clinical</td><td>2</td><td>Spatially propagating ictal discharge</td></tr><tr><td>adftd</td><td>Clinical</td><td>3</td><td>Diffuse cortical slowing, theta excess</td></tr></table>

Motor imagery tasks are most sensitive to interaction-isotropic mixing: mu/beta ERD lateralization [Pfurtscheller and Lopes da Silva, 1999] (left vs. right hand) is the primary discriminative signal, and averaging across all tokens via dense global attention can suppress this asymmetry.

## D.2 Training budget

Table 1 reports the best-validation-Mean-LP checkpoint (held-out subjects, never test), not the final epoch. To show the budget does not drive the result, we linear-probed AXON and Dense at epochs 5–50 (Mean LP over the six tasks, single seed; Table 10). Validation-best was epoch 10 (AXON) and epoch 9 (Dense); at the final epoch AXON still leads 0.568 vs. 0.543. The ranking is stable at every budget: AXON leads by +0.025–0.028 Mean LP and Dense never catches up, so the gain is not faster learning that more compute would erase. We do not claim it holds for unlimited training, only that it is stable across every budget we tested.

Table 10: Mean LP over the six tasks at matched pretraining budgets (single seed). AXON leads at every epoch.
<table><tr><td>Epoch</td><td>5</td><td>10</td><td>20</td><td>30</td><td>40</td><td>50</td></tr><tr><td>AXON</td><td>0.584</td><td>0.579</td><td>0.580</td><td>0.574</td><td>0.570</td><td>0.568</td></tr><tr><td>Dense</td><td>0.556</td><td>0.551</td><td>0.552</td><td>0.549</td><td>0.545</td><td>0.543</td></tr></table>

## E Why Two Layers Connect Every Pair of Tokens

## E.1 Axis graph diameter

AXON removes the dense all-to-all path. In one layer a token attends only to tokens on its own electrode (the temporal path) and to tokens at its own time step (the spatial path). A concern is that this cuts the model off from the rest of the recording: a token on electrode c at time t never sees electrode c<sup>′</sup> at time t<sup>′</sup> directly. The proposition below shows that this is not so. On a full channel-time grid, any two tokens are connected after two layers, so the model keeps its global reach. What changes is which pairs interact directly within one layer, and how many. This is why we describe the design as changing which pairs interact directly, not the model’s reach, and it is the fact behind the statement in Section 2.2 that any two tokens can still meet after two layers.

Proposition 1 (Axis graph has diameter at most two). On the full channel-time grid $\Omega = \mathcal { C } \times \mathcal { T }$ the graph with edges between tokens that share either the same channel or the same time index has diameter at most two. After two stacked axis-attention layers, any token can receive informationfrom any other token. The number ofpossible one-hop attention interactions is

$$
| E _ { \mathrm { a x i s } } | \leq C T ^ { 2 } + T C ^ { 2 } = C T ( C + T ) ,
$$

versus $| E _ { \mathrm { d e n s e } } | = ( C T ) ^ { 2 }$ for the dense graph.

Proof. Take any two tokens $\boldsymbol { u } = ( c , t )$ and $v = ( c ^ { \prime } , t ^ { \prime } ) . \operatorname { I f } c = c ^ { \prime } \operatorname { o r } t = t ^ { \prime } .$ , then u and v are connected by one axis edge. Otherwise, u is connected to $( c ^ { \prime } , t )$ by a spatial edge (same time step), and $( c ^ { \prime } , t )$ is connected to $v = ( c ^ { \prime } , t ^ { \prime } )$ by a temporal edge (same channel). Every pair is thus connected by a path of length at most two. The edge-count bound follows from C temporal groups of size $T$ and $T$ spatial groups of size $C .$ □

## F Ablation Summary

## F.1 Interpreted ablation summary

Table 11: EEG ablation summary. Values are unweighted mean balanced accuracy across six downstream tasks. Diagnostic variants are included to show task-specific tradeoffs; bold indicates the best mean in each column among evaluated variants.
<table><tr><td>Variant</td><td>Mean LP</td><td>Mean FT</td><td>Purpose of comparison</td></tr><tr><td>Dense</td><td>0.550</td><td>0.606</td><td>Dense attention baseline with split positional encod- ing.</td></tr><tr><td>Dense-L</td><td>0.555</td><td>0.633</td><td>Parameter-matched dense control.</td></tr><tr><td>Divided-ST</td><td>0.560</td><td>0.628</td><td>Sequential divided space-time operator [Bertasius et al., 2021] in the identical setup (Appendix F.2).</td></tr><tr><td>Dense + joint PE</td><td>0.540</td><td>0.600</td><td>Tests whether joint positional encoding improves over split PE.</td></tr><tr><td>Dense + 2D-RoPE</td><td>0.554</td><td></td><td>Axial 2D rotary position embedding on the dense graph; single seed vs. a single-seed dense reference</td></tr><tr><td>AXON-Fixed</td><td>0.558</td><td>0.640</td><td>(0.546). Tests axis factorization without token-dependent rout-</td></tr><tr><td>Fixed gate, schedule init</td><td>0.555</td><td>0.636</td><td>ing. Tests whether initializing a fixed gate to AXON&#x27;s learned layer schedule is sufficient.</td></tr><tr><td>AXON-TokenGated</td><td>0.578</td><td>0.643</td><td>Tests token-gated temporal/spatial factorization with- out multiscale temporal windows.</td></tr><tr><td>AXON</td><td>0.579</td><td>0.656</td><td>Final no-global model with token axis gate and two- scale temporal branch.</td></tr><tr><td>AXON without stop-gradient</td><td>0.577</td><td></td><td>Gate reads the live representation instead of a detached copy; single seed.</td></tr><tr><td>AXON-withGlobal</td><td>0.555</td><td>0.619</td><td>Tests whether adding a dense global branch improves the factorized encoder.</td></tr><tr><td>KNN spatial constraint</td><td>0.510</td><td>0.609</td><td>Tests whether local electrode-neighbour masking helps</td></tr><tr><td>Head-split multiscale</td><td>0.553</td><td>0.628</td><td>the spatial path. Tests an alternative multiscale temporal implementa-</td></tr><tr><td>Temporal decay</td><td>0.556</td><td>0.642</td><td>tion. Diagnostic locality bias; helps some event-like tasks</td></tr><tr><td>AXON-Decay continuation</td><td>0.579</td><td>0.633</td><td>but reduces mean performance. Diagnostic continuation run; maintains LP but reduces  $\mathrm { F T } .$ </td></tr></table>

Reading the ablations. The ablation summary supports three conclusions. First, axis factorization improves over dense attention even after controlling for parameter count (Dense-L). Second, token gating provides the largest additional LP gain over fixed factorization $( + 0 . 0 2 0 )$ , while the two-scale temporal path contributes a smaller task-selective refinement. Third, the negative variants show that adding a dense global branch, hard spatial masks, or extra temporal-scale machinery does not improve mean transfer. AXON is the only variant that achieves both the best mean LP and the best mean

Table 12: Architectural variants summary. All models use the same pretraining corpus and schedule; they differ only in attention structure.
<table><tr><td>Variant</td><td>Temporal windows</td><td>Global path</td><td>Key change</td><td>Mean LP</td></tr><tr><td>Dense</td><td>W=-1</td><td>Full</td><td>Reference</td><td>0.550</td></tr><tr><td>Divided-ST</td><td>W=-1</td><td>No</td><td>Sequential T→S blocks</td><td>0.560</td></tr><tr><td>AXON-Fixed</td><td>W=-1</td><td>No</td><td>Fixed factorized axes</td><td>0.558</td></tr><tr><td>AXON-TokenGated</td><td>W=-1</td><td>No</td><td>Per-token axis gating</td><td>0.578</td></tr><tr><td>AXON-withGlobal</td><td>W=[5,−1]</td><td>Yes</td><td>+Global dense path</td><td>0.555</td></tr><tr><td>AXON</td><td>[5,−1]</td><td>No</td><td>+Two-scale gate</td><td>0.579</td></tr><tr><td>AXON-Decay</td><td>[5,-1] + decay</td><td>No</td><td>+Temporal locality</td><td>0.579</td></tr><tr><td>Head-split multiscale</td><td>[5,—1] static</td><td>No</td><td>Static scale split</td><td>0.553</td></tr><tr><td>3-scale gate</td><td>[2,5,-1]</td><td>No</td><td>3-scale gate</td><td>0.558</td></tr></table>

FT; variants that improve selected tasks (e.g., AXON-Decay) do not improve the mean FT objective and are therefore treated as diagnostic, task-selective extensions. The token gate’s LP advantage over AXON-Fixed arises from training-time gradient diversity rather than inference-time routing; see Appendix H.4 for the schedule-initialized fixed gate experiment that isolates this effect.

Three windows. We also tried three windows (two, five, and all patches) with an entropy penalty that pushes the scale mix to use all three. Mean LP fell to 0.558. The reconstruction loss barely distinguishes the three windows, so the penalty dominates and the mix stays close to uniform; two windows with a long-window start was the best we found.

Temporal locality (AXON-Decay). AXON-Decay adds a learnable temporal-distance bias to the full-window temporal path, so that nearby time steps get more weight. It helps tasks driven by short events, motor imagery (LP 0.455 → 0.487) and seizure detection (siena LP 0.867 → 0.894); it matches AXON on Mean LP (0.579), lowers Mean FT (0.656 → 0.633), and hurts workload, a sustained-state task. So the best temporal context length depends on the task, and we treat AXON-Decay as a diagnostic, not a replacement for AXON.

2D RoPE control. We ran axial 2D RoPE [Su et al., 2024, Heo et al., 2024]: each head (dim 64) is split so attention depends only on the relative (∆t, ∆c) offset while the graph stays fully dense. Result (same pipeline as Table 1; single seed, vs. a single-seed dense reference): Dense 0.546 → Dense+2D-RoPE 0.554, a +0.008 Mean LP gain concentrated almost entirely on adftd (our highest-variance task, ±0.034 across seeds). Axis factorization gives +0.029, ≈3.6× the RoPE delta. Thus 2D RoPE does not substitute for factorization (though it does not fail either). This is expected: RoPE changes how position enters the scores but leaves the graph dense; every off-axis pair stays available, whereas AXON removes those edges by construction. After pretraining the dense encoder still places 30–63% of its attention on off-axis pairs (token pairs sharing neither the same electrode nor the same time step; Table 16, Figure 8), so it does not suppress those interactions on its own, and a relative-position code gives it no mechanism to. The two are orthogonal, not substitutes.

## F.2 Divided space-time baseline

The divided space-time baseline is pretrained in our exact EEG setup: same pooled corpus, same 10-20 montage and tokenisation, batch size 4096, peak LR 2.4 × 10<sup>−4</sup>, fused AdamW, bfloat16, the identical pretraining budget, the identical MAE objective (L1 on masked patches plus the pooledattention auxiliary loss), and the same split positional encoding. Only the attention operator differs. We implement the divided space-time block of Bertasius et al. [2021] faithfully: attention along time, then attention along channels, each with its own residual connection and with the temporal output projection, in place of AXON’s parallel token-gated axis mixture. Parameter count and per-layer compute are comparable to AXON’s ∼141.9M. The two models converge to matched reconstruction loss, so neither is under-trained relative to the other. Metric is balanced accuracy on subject-disjoint splits identical across models; per-task results are in Table 1, where ± is the standard deviation across the 3 downstream seeds.

Divided-ST recovers part of the gap to AXON, so restricting attention to the two axes helps on its own. The rest of the gap is what the parallel, gated composition adds. In Divided-ST the axis order is fixed and identical for every token and every layer; in AXON the gate sets the temporal/spatial balance per token and per layer, and Appendix H shows that this learned per-layer balance is where most of the gate’s benefit lies.

## G Analysis of the Global-Path Variant

Section 2.4 states our hypothesis: under masked pretraining the dense global path is a shortcut, averaging over visible tokens to guess a masked patch, so the encoder builds fewer axis-specific features. Here we give the evidence behind that reading.

Same reconstruction loss, different features. AXON and AXON-withGlobal reach the same reconstruction loss (Figure 2), yet AXON-withGlobal transfers worse (Mean LP 0.555 vs. 0.579, Mean FT 0.619 vs. 0.656). So the difference is in the features the two encoders build, not in how well they reconstruct.

![](images/5ca516ff036c4747b2d79e33e191d5e5c527d9fd7c12e4e1ff81f215c3742744.jpg)  
Figure 2: MAE reconstruction loss during pretraining. AXON (no global path) and AXONwithGlobal converge to comparable reconstruction loss (≈0.38–0.39), yet AXON-withGlobal underperforms on mean LP (0.555 vs. 0.579) and mean FT (0.619 vs. 0.656). The global path does not improve reconstruction; it changes how the encoder reconstructs, routing mass through dense averaging (γ =0.724 at layer 1) rather than through axis-structured features.

Effective rank. Effective rank [Roy and Vetterli, 2007] counts how many dimensions a set of features actually uses; a higher value means richer, less collapsed features. We compute it on the final mean-pooled embeddings of up to 512 validation samples per dataset (Table 13). Removing the global path raises the effective rank on five of the six datasets. The largest jump is on motor imagery, from 19.97 to 29.05, and motor imagery is also the task where AXON gains most over Dense (Table 1). This fits the shortcut reading: motor imagery depends on fine, local spatio-temporal structure that averaging over all tokens washes out.

CKA similarity. Linear CKA [Kornblith et al., 2019] measures how similar two sets of features are (1.0 identical, 0.0 unrelated). The features of AXON and AXON-withGlobal are least similar on motor imagery (0.84), sleep staging (0.86) and workload (0.87), and almost identical on seizure detection and dementia (above 0.96). The global path changes the features most on the tasks with the most temporal or spatial structure, and least where the two models also perform alike. Per-layer curves for all six datasets are in Figure 3.

Table 13: Effective rank and representation similarity (Linear CKA) of final mean-pooled embeddings between the factorized noGlobal model and the withGlobal model.
<table><tr><td rowspan="2">Dataset</td><td colspan="2">Effective Rank ↑</td><td rowspan="2">CKA Similarity ↓ withGlobal vs noGlobal</td></tr><tr><td>withGlobal</td><td>noGlobal</td></tr><tr><td>adftd</td><td>17.76</td><td>21.29</td><td>0.9748</td></tr><tr><td>bcic_2a</td><td>66.08</td><td>77.66</td><td>0.9363</td></tr><tr><td>hmc</td><td>22.83</td><td>28.79</td><td>0.8662</td></tr><tr><td>motor</td><td>19.97</td><td>29.05</td><td>0.8412</td></tr><tr><td>siena</td><td>28.93</td><td>27.19</td><td>0.9634</td></tr><tr><td>workload</td><td>51.58</td><td>65.48</td><td>0.8698</td></tr></table>

![](images/40591cd6996247abdabc0a2de6f670daedcdede444031fd543eae5519ccfbbe3.jpg)  
(a) Layer-wise effective rank, all six datasets.

![](images/68fb95669fd56797bbdde88645e708088f27261748f1d91eafd176cf0cbc0f22.jpg)  
(b) Layer-wise Linear CKA, all six datasets.  
Figure 3: Layer-wise effective rank and CKA divergence between factorized and dense-global representations.

## H Gate Mechanism Analysis

## H.1 Understanding the gate

Figure 4 shows the average gate weight per layer over the six downstream datasets. The gate is not a uniform mixer. The first layer leans on the temporal path $( \alpha = 0 . 7 4 5 )$ , layers 4–7 lean on the spatial path $( \beta = 0 . 6 8 6 – 0 . 7 0 7 )$ , and late layers return toward the temporal path $( \alpha = 0 . 6 5 1$ at layer 19). So the gate learns a different temporal/spatial balance at each depth. The rest of this appendix asks whether the model actually depends on these values.

![](images/4c6a9af1df91b01d0fd3c0684e6f635d0cedf2ac850cc24b13b3574484a771d4.jpg)  
Figure 4: AXON axis gate across the 22 encoder layers: gate weights averaged over all six downstream datasets (mean $\pm \sigma )$ , on validation batches with the frozen encoder. Temporal-heavy in the first layer, spatial-heavy in the middle layers, and back toward temporal in the late layers.

## H.2 Gate intervention diagnostics

We override the axis gate while keeping the pretrained encoder frozen, then run linear probe evaluation (full dataset splits). Each intervention replaces the learned per-token gate $[ \alpha _ { i } , \beta _ { i } ]$ with a modified version that removes a specific component of the routing, isolating what the gate actually contributes. We group the seven interventions by the question they answer.

Is axis factorization itself necessary?

• temporal only: Force $[ \alpha _ { i } , \beta _ { i } ] = [ 1 , 0 ]$ for all tokens. Only the temporal path contributes.

• spatial only: Force $[ \alpha _ { i } , \beta _ { i } ] \stackrel { \cdot } { = } [ 0 , \stackrel { \cdot } { 1 } ]$ . Only the spatial path contributes.

Does soft mixing matter, or can the model commit to one axis?

• hard argmax: Take the learned gate, find the dominant axis, and set it to 1.0 with the other at 0.0. E $g _ { \cdot } , [ 0 . \bar { 6 } , 0 . 4 ]  [ 1 . 0 , 0 . 0 ]$

• uniform: Force $[ \alpha _ { i } , \beta _ { i } ] = [ 0 . 5 , 0 . 5 ]$ for all tokens. No routing at all equal weight to both paths.

Is the gate’s value per-token content routing, or a depth/position schedule?

• layer mean: Calibrate over 50 forward passes to compute the average gate per layer, averaging across all tokens and batches. At inference, every token in layer ℓ receives the layer-ℓ mean gate, regardless of content. Tests whether the depth schedule alone is sufficient.

• position mean: Calibrate the average gate per (layer, channel, time-patch) position. At inference, a token at position (c, t) in layer ℓ receives the calibrated mean for that position, regardless of signal content. Tests whether position-aware routing adds value beyond the depth schedule.

• shuffled: Run the gate MLP normally to produce $[ \alpha _ { i } , \beta _ { i } ]$ for each token, then randomly permute the gate values within each layer. The marginal distribution of gate values per layer is exactly preserved, but the token↔gate correspondence is destroyed. Tests whether it matters which token gets which gate value.

Table 14 reports summary statistics; Figure 5 shows per-dataset results.

Table 14: Gate intervention diagnostics under frozen-encoder LP. Mean is over all 6 downstream datasets. Scores are balanced accuracy on the test subjects at the validation-selected epoch, averaged over 3 downstream seeds for five datasets; siena uses a single seed and the class-balanced training loader.
<table><tr><td>Gate mode</td><td>Mean BAC</td><td>∆ vs learned</td></tr><tr><td>Learned (reference)</td><td>0.571</td><td></td></tr><tr><td>Uniform  $( 1 / 2 , 1 / 2 )$ </td><td>0.539</td><td>-5.6%</td></tr><tr><td>Layer mean</td><td>0.559</td><td>-2.1%</td></tr><tr><td>Position mean</td><td>0.567</td><td>-0.8%</td></tr><tr><td>Shuffled</td><td>0.564</td><td>-1.3%</td></tr><tr><td>Hard argmax</td><td>0.501</td><td>-12.2%</td></tr><tr><td>Temporal only</td><td>0.471</td><td>-17.5%</td></tr><tr><td>Spatial only</td><td>0.446</td><td>-22.0%</td></tr></table>

Protocol note. The learned-gate reference is 0.571 in this intervention setting, compared with the main AXON LP score of 0.579 in Table 1. All intervention results are measured relative to this within-table learned reference.

## H.3 Token diversity and content dependence

We compute two scalar metrics to characterise within-layer gate variation. $D _ { \mathrm { t o k e n } }$ measures how different individual token gates are from the layer mean; $D _ { \mathrm { { c o n t e n t } } }$ subtracts out fixed channel/time position effects, isolating variation that depends on the signal content of the current input. $D _ { \mathrm { { c o n t e n t } } }$ peaks sharply at layers 9 and 12 (Figure 6), indicating that signal-driven routing is concentrated at mid-depth rather than distributed uniformly across the encoder. However, the gate intervention results (Table 14) show that this token-level variation is not the dominant source of downstream gain: shuffled gates drop only −1.3% vs. learned.

![](images/fb3d53cdb9a4568e41e84da98a51425e572dc9823c95f432901fd2985abb9ebc.jpg)  
Figure 5: Per-dataset gate intervention heatmap (frozen-encoder LP, 6 datasets; same protocol as Table 14). Cell colour is the change from the learned gate. The learned gate outperforms uniform and hard-argmax routing on most datasets, but layer-mean, position-mean, and shuffled gates stay close to learned, indicating that exact token-gate alignment is not the dominant effect.

D\_content: position-conditioned routing variation across layers  
![](images/6b2cf794d4b123cf632f26a37240da84d7c35db4d14762d23917cedcb3c8e9bd.jpg)  
Figure 6: $D _ { \mathrm { t o k e n } }$ and $D _ { \mathrm { { c o n t e n t } } }$ across 22 encoder layers. Peaks at layers 9 and 12 show that some layers exhibit genuine token-level gate diversity. However, the intervention results above show that this variation is not the dominant source of downstream gain: shuffled gates (which preserve the gate distribution but destroy token-gate alignment) drop only −1.3% vs learned.

## H.4 Analysis of the AXON-TokenGated vs AXON-Fixed gap

Giving every token in a layer that layer’s mean gate costs only −2.1% (Section H.2). So at test time, one mixing weight per layer is almost enough. But AXON-Fixed learns exactly one weight per layer, and it reaches only 0.558 Mean LP, against 0.578 for AXON-TokenGated. Why?

One guess is initialisation: AXON-Fixed starts from a neutral mixture and may never find the right weight for each layer. We tested this. We trained AXON-Fixed again, this time starting each layer’s weight at the value the AXON gate had learned for that layer (Section H.1). Nothing else changed. Table 15 shows the result: 0.555 Mean LP, the same as before (0.558), still far below 0.578. The right starting point does not help. So initialisation is not the reason.

What is left is how the two models train. With a per-token gate, tokens in the same layer get different mixtures during training, so the temporal and spatial paths are trained on more varied signals. At the end of training the exact per-token values no longer matter much (shuffling them costs only −1.3%), but the paths they trained are better. Dropout works the same way: it does nothing at test time, but it changes the weights that training ends with.

Table 15: AXON-Fixed re-trained with each layer’s weight initialised to AXON’s learned per-layer balance. For comparison: AXON-Fixed with neutral initialisation reaches 0.558 Mean LP, AXON-TokenGated 0.578.
<table><tr><td>Dataset</td><td>LP BAC</td><td>FT BAC</td></tr><tr><td>motor</td><td>0.354</td><td>0.598</td></tr><tr><td>workload</td><td>0.585</td><td>0.714</td></tr><tr><td>hmc</td><td>0.673</td><td>0.732</td></tr><tr><td>siena</td><td>0.878</td><td>0.873</td></tr><tr><td>adftd</td><td>0.557</td><td>0.588</td></tr><tr><td>bcic</td><td>0.283</td><td>0.311</td></tr><tr><td>Mean</td><td>0.555</td><td>0.636</td></tr></table>

## H.5 Temporal scale gate routing

Figure 7 shows the per-layer routing between the short-window $( W _ { s } = 5 )$ and full-window $( W _ { l } = - 1 )$ temporal branches across all 22 encoder layers. The scale gate strongly favours the long-window branch $( \approx 7 9 \%$ routing mass), consistent with the initialisation bias toward $W _ { l }$ . A small subset of layers routes appreciable mass to the short window, suggesting that local transient structure is selectively useful at those depths.

![](images/7abd72025a8dea72d897ea604b3cb9c6d019547f0d572561688900b9c3195687.jpg)  
Figure 7: Scale gate routing: short window $( W = 5 )$ vs. full window $( W = \infty )$ across 22 layers. The model usually favours broad temporal context, but uses short temporal windows in a few selected layers. The two-scale temporal design contributes +0.001 Mean $\bar { \mathrm { L P } }$ and +0.013 Mean FT over the single-scale variant; the scale gate is a secondary improvement.

## I Interpreting the Attention

## I.1 Does the trained dense model use the off-axis pairs that AXON removes?

For a query token in the modal $C = 2 1 , T = 1 1$ grid, 200 of the 231 tokens share neither its channel nor its time step. We call these off-axis pairs. An untrained dense model spreads its attention uniformly, so about $87 \%$ of its attention starts on off-axis pairs. The question is how much of that the dense model learns to remove during pretraining. We measured it on the motor imagery grid $( C = 6 4$ $T = 4 )$ , because motor imagery is the task with AXON’s largest gain and the 64-channel layout makes the pair types easy to separate; there the uniform off-axis share is 73.8%. After pretraining, attention within the same time step rises to 62.4% and the off-axis share falls to 30.9% on average (Table 16). But it never goes away: the first layer still places 63% of its attention off-axis, almost the untrained value, and the last layer drifts back to 46% (Figure 8). The dense model learns the axis structure only partly and unevenly across depth. AXON assigns zero off-axis attention within a layer by construction.

Table 16: Relation-type attention mass (%) in the dense baseline on the default motor mv img downstream grid $( C = 6 4 , T = 4 )$ , averaged across all 22 layers. After MAE pretraining, spatial mass rises and off-axis mass drops, but residual off-axis routing remains. AXON assigns zero one-layer off-axis mass by construction.
<table><tr><td>Condition</td><td>Self</td><td>Temporal</td><td>Spatial</td><td>Off-axis</td></tr><tr><td>Uniform reference</td><td>0.39</td><td>1.17</td><td>24.6</td><td>73.8</td></tr><tr><td>Dense (trained)</td><td>4.65</td><td>2.06</td><td>62.4</td><td>30.9</td></tr><tr><td>AXON (by construction)</td><td></td><td>100 (temporal path)</td><td>100 (spatial path)</td><td>0</td></tr></table>

Attention mass by EEG-grid relation type — dense baseline (motor\_mv\_img) (dashed lines = uniform reference per relation)  
![](images/eba7371fa99db301c19efea1945504e7982ac4d1eb309830cce71b6e622abbd8.jpg)

![](images/3995ee7085a27aeef4941572e599768b088d658ab6f68b526c6cd5d8e50b6c3a.jpg)  
Figure 8: Relation-type attention mass per layer in the dense baseline $( C = 6 4 , T = 4 )$ Top: Random initialisation matches the uniform complete-graph prior (dashed lines). Bottom: After MAE pretraining, dense attention discovers spatial dominance in mid-depth layers but fails to fully suppress off-axis interactions. Early and late layers retain up to 63% off-axis mass, leaving off-axis interactions that AXON removes by construction.

## I.2 Does AXON rely on the electrodes that physiology predicts?

The task is four-class motor imagery (motor mv img, Table 8). In each trial the subject imagines moving the left hand, the right hand, both fists, or the feet. The body is controlled from the opposite side of the brain: the left hand from the right hemisphere, the right hand from the left. Both fists use both sides. The feet are controlled from the midline. So for each class we know which electrodes a good classifier should be using.

We take the trained AXON motor-imagery classifier and its held-out test subjects. We cover the seven electrodes over the left motor cortex, so the model gets no signal from them, and measure how much each class’s recall changes. Recall is the fraction of a class’s trials that the model labels correctly. Then we do the same for the seven electrodes over the right motor cortex. We chose the electrodes and the measure before looking at any result.

If the classifier uses the correct electrodes, covering one side should mainly hurt the opposite hand, and should not hurt both fists or feet in a one-sided way. If it used all electrodes alike, covering either side would hurt all four classes alike. Table 17 shows the first pattern. Covering the left side drops right-hand recall by 0.101 and does not hurt the left hand (+0.058). Covering the right side drops left-hand recall by 0.127 and does not hurt the right hand (−0.004). Both fists and feet show no one-sided change. So the drop is not just “fewer electrodes, worse accuracy”: each hand’s decision depends on the electrodes over the opposite hemisphere, exactly where physiology says it should. We use this test rather than an attention map because it changes the input and watches the decision; an attention map only shows where the weights point.

Table 17: Change in recall after covering the left or right motor strip (motor imagery, held-out subjects); negative means worse.
<table><tr><td>Imagined movement</td><td>Cover LEFT strip</td><td>Cover RIGHT strip</td></tr><tr><td>Left hand</td><td>+0.058</td><td>-0.127</td></tr><tr><td>Right hand</td><td>-0.101</td><td>-0.004</td></tr><tr><td>Both fists</td><td>-0.027</td><td>-0.001</td></tr><tr><td>Feet</td><td>+0.013</td><td>+0.052</td></tr></table>

## I.3 Does the accuracy depend on particular electrodes?

Deleting randomly chosen electrodes at test time, AXON stays ahead of dense (balanced accuracy) at every level, from 0.377 vs. 0.321 intact to 0.274 vs. 0.263 with 94% of electrodes removed. Here the classification head is trained on mean-pooled frozen features rather than through our full evaluation harness, so absolutes differ from Table 1 and only the model-to-model comparison is meaningful. This matters clinically, where reduced montages and failed electrodes are routine.

## J Cross-Modal Generalization (AudioMAE)

Gap to published AudioMAE. The absolute mAP values are not directly comparable to those reported by Huang et al. [2022]. For example, our best full AudioSet-2M FT result is 14.29 mAP, whereas published AudioMAE-style results report much higher absolute mAP under a substantially different training recipe (e.g., 37.0 on AudioSet-20K). This gap reflects five controlled differences: (1) spectrogram resolution (128 vs. 8 frequency bins, a 16× reduction that brings the frequency axis to a size comparable to the EEG electrode axis); (2) pretraining compute (4 epochs vs. 32 epochs, ∼9× fewer sample-views); (3) model capacity (d = 512 vs. d = 768, ∼50% fewer parameters); (4) decoder depth (4 vs. 16 layers); and (5) FT augmentation (no Mixup, SpecAugment, or DropPath). Crucially, the factorized and dense variants share all five of these constraints, so the within-setup comparison is valid.

## J.1 Small-scale preliminary results (18K AudioSet clips)

Before scaling to 200K clips, we verified factorized attention at 1% of AudioSet (∼18K clips). The same four-variant design (Dense, Factorized fixed gate, Factorized token gate, Token+Global) was trained for 33 epochs with identical optimizer and schedule.

Table 18: Audio results at 1% scale (18K AudioSet clips)
<table><tr><td>Model</td><td>AudioSet FT mAP</td><td>AudioSet LP mAP</td><td>ESC-50 LP Acc</td><td>SC LP Acc</td></tr><tr><td>Dense</td><td> $6 . 7 6 { \pm } 0 . 0 7$ </td><td>1.18±0.01</td><td>20.50±0.43</td><td> $1 1 . 2 3 { \pm } 0 . 7 6 $ </td></tr><tr><td>Factorized (fixed gate)</td><td>10.36±0.12</td><td>1.21±0.01</td><td>21.33±2.75</td><td> $1 1 . 7 0 { \scriptstyle \pm 0 . 4 6 }$ </td></tr><tr><td>Factorized (token gate)</td><td> $\mathbf { 1 0 . 4 8 { \pm } 0 . 3 3 }$ </td><td>1.32±0.01</td><td>20.67±0.63</td><td> $1 2 . 2 6 { \pm } 0 . 0 3$ </td></tr><tr><td> $\mathrm { T o k e n + G l o b a l }$ </td><td>10.09±0.14</td><td>1.40±0.01</td><td>21.50±0.20</td><td> $\mathbf { 1 2 . 6 4 } \pm 0 . 1 0$ </td></tr></table>

All factorized variants improved AudioSet FT mAP by +49–55% relative over the dense baseline even at this small scale, demonstrating that the factorized advantage is present from the smallest dataset scale tested.

## J.2 200K-scale results

Table 19: Audio results at 200K scale (∼10% of AudioSet-2M)
<table><tr><td>Model</td><td></td><td>AudioSet FT mAP AudioSet LP mAP</td><td>ESC-50 LP Acc</td><td> ${ \bf S C L P A c c }$ </td></tr><tr><td>Dense</td><td> $1 1 . 0 4 { \pm } 0 . 2 5 $ </td><td> $3 . 6 1 { \pm } 0 . 0 0$ </td><td> $4 0 . 3 3 { \pm } 0 . 5 1 $ </td><td> $1 7 . 7 1 { \pm } 0 . 1 2 $ </td></tr><tr><td>Factorized (fixed gate)</td><td> $1 3 . 5 2 { \pm } 0 . 1 5 $ </td><td> $3 . 7 6 { \pm } 0 . 0 3$ </td><td> $4 3 . 4 2 { \pm } 0 . 9 4 $ </td><td> $1 9 . 8 8 { \pm } 0 . 0 8 $ </td></tr><tr><td>Factorized (token gate)</td><td> $1 3 . 7 5 { \pm } 0 . 2 0 $ </td><td> $4 . 0 9 { \pm } 0 . 0 4$ </td><td> $4 4 . 4 2 { \pm } 0 . 4 2 $ </td><td> ${ \bf 2 1 . 0 8 } { \pm } 0 . 1 0 $ </td></tr><tr><td> $\mathrm { T o k e n + G l o b a l }$ </td><td> $\mathbf { 1 3 . 9 9 2 0 . 1 5 }$ </td><td> $\mathbf { 4 . 4 0 { \pm } } 0 . 0 4$ </td><td> ${ \bf 4 4 . 7 5 { \pm } 1 . 2 7 }$ </td><td> $2 0 . 0 0 { \pm } 0 . 1 8$ </td></tr></table>

At 200K, the factorized advantage persists across all four metrics. Token+Global wins 3 of 4 metrics but underperforms the token-gate model on SpeechCommands LP (20.00 vs. 21.08). The SpeechCommands ordering varies across scale: Token+Global is ahead at 18K (Table 18: 12.64 vs. 12.26), behind at 200K, and approximately tied with the token-gate model at 2M (Table 2: 21.87 vs. 21.96). We therefore avoid drawing a stable architectural conclusion from this single metric and focus on the consistent factorized-vs-dense improvement.

## K Cross-Modal Gate Mechanism Analysis

To test whether the gate intervention findings from EEG (§H.2) are EEG-specific or reflect a general property of axis-factorized attention, we run the same seven gate overrides on the audio AXON-TokenGated model (200K AudioSet pretraining). The encoder is frozen; only the linear probe head is trained. We evaluate on ESC-50 and SpeechCommands v2.

Table 20: Audio gate intervention results. The same seven overrides from the EEG analysis are applied to the frozen audio AXON-TokenGated encoder. $\Delta$ is the absolute change in accuracy relative to the learned baseline.
<table><tr><td>Intervention</td><td>ESC-50 Acc</td><td>SC Acc</td><td> $\mathrm { E S C } { - } 5 0 \Delta$ </td><td> $\mathtt { S C } \Delta$ </td></tr><tr><td>Learned (baseline)</td><td>44.50</td><td>20.93</td><td></td><td></td></tr><tr><td>Shuffled</td><td>44.50</td><td>20.98</td><td>+0.00</td><td>+0.05</td></tr><tr><td>Layer-mean</td><td>44.25</td><td>20.71</td><td>-0.25</td><td>-0.22</td></tr><tr><td>Uniform  $( 1 / 2 , 1 / 2 )$ </td><td>44.00</td><td>21.01</td><td>-0.50</td><td>+0.08</td></tr><tr><td>Position-mean</td><td>43.50</td><td>20.59</td><td>-1.00</td><td>-0.34</td></tr><tr><td>Hard argmax</td><td>39.00</td><td>20.39</td><td>-5.50</td><td>-0.54</td></tr><tr><td>Frequency-only  $( \alpha { = } 0 )$ </td><td>29.75</td><td>18.03</td><td>-14.75</td><td>-2.90</td></tr><tr><td>Temporal-only (β = 0)</td><td>23.75</td><td>13.64</td><td>-20.75</td><td>-7.29</td></tr></table>

Cross-modal comparison. Table 21 compares the mean relative degradation of each intervention across EEG (6 tasks, balanced accuracy) and audio (2 tasks, accuracy). Both columns report the mean relative change $( \Delta / \mathrm { b a s e l i n e } ) \times \dot { 1 } 0 0$ . The interventions show a similar broad pattern across mannas.ais: removing either axis or replacing soft mixing with hard selection is more damaging than averaging or shuffling the gates. The exact ordering and magnitudes differ: spatial-only is the worst override on EEG but temporal-only is the worst on audio, and uniform mixing costs 5.6% on EEG but almost nothing on audio.

Audio layer-mean gate profile. The calibrated per-layer gate means reveal an interpretable depth schedule. Early layers favour the frequency axis $( \alpha _ { 1 } = 0 . 4 2 2$ , frequency-heavy), mid layers are approximately balanced $( \alpha _ { 3 - 5 } \approx 0 . 4 8 5 )$ , and late layers shift toward the temporal axis $( \alpha _ { 1 0 } = 0 . 5 6 4$ $\alpha _ { 1 1 } = 0 . 5 5 6 )$ . This is the opposite direction from EEG, where early layers are temporal-heavy $( \alpha _ { 0 } = 0 . 7 4 5 )$ and mid layers are spatial-heavy. The reversal is consistent with mannas.ai structure: early audio layers capture spectral features (pitch, harmonics) that require cross-frequency integration, while late layers capture temporal dynamics (onsets, rhythm) that require cross-time integration. In EEG, the early temporal bias captures fast transient features (spikes, ERD onset) before spatial mixing integrates across electrodes.

![](images/ba9beffa87f18a9c9c515b89bb730c989dfeb7fd59f9d5f8e71cf58015d0f944.jpg)  
Figure 9: Audio gate intervention heatmap (∆% vs. learned baseline). The pattern mirrors EEG: single-axis routing is catastrophic, while shuffled and layer-mean gates are indistinguishable from the learned gate.

Table 21: Gate intervention mean relative degradation (%): EEG vs. audio. Both columns report the mean relative change from the learned baseline. The broad pattern is shared across mannas.ais; the exact ordering and magnitudes differ.
<table><tr><td>Intervention</td><td>EEG Mean ∆%</td><td>Audio Mean ∆%</td></tr><tr><td>Shuffled</td><td>-1.3</td><td>+0.1</td></tr><tr><td>Layer-mean</td><td>-2.1</td><td>-0.8</td></tr><tr><td>Position-mean</td><td>-0.8</td><td>-1.9</td></tr><tr><td>Uniform</td><td>-5.6</td><td>-0.4</td></tr><tr><td>Hard argmax</td><td>-12.2</td><td>-7.5</td></tr><tr><td>Spatial/Freq-only</td><td>-22.0</td><td>-23.5</td></tr><tr><td>Temporal-only</td><td>-17.5</td><td>-40.7</td></tr></table>

Uniform robustness gap. The most notable cross-modal difference is the uniform intervention: −5.6% in EEG but only −0.4% in audio. This indicates that the audio schedule is flatter, the per-layer gate values range from α = 0.422 to 0.564 (range 0.14), compared to α = 0.294 to 0.745 (range 0.45) in EEG. Audio representations benefit nearly equally from both axes at all depths, while EEG requires stronger layer-varying axis preferences. This is consistent with audio spectrograms containing genuine cross-axis harmonic structure at all levels, whereas EEG temporal and spatial dynamics are more separable.

Summary. The gate interventions show the same broad pattern in both mannas.ais: both axes and soft mixing are necessary, while the dominant useful gate structure is the learned per-layer temporal/spatial balance. Token-level content routing is secondary rather than dominant in both EEG (−1.3% shuffled) and audio (+0.1% shuffled). The token gate serves as a training mechanism that discovers an appropriate layer-wise axis schedule, and the optimal schedule direction differs between mannas.ais (temporal-first in EEG, frequency-first in audio).

## K.1 Limitations

Several limitations should be noted. First, the theoretical support for removing the global path rests on empirical ablations (Table 11) rather than a formal information-theoretic proof for nonlinear masked autoencoders; such a treatment remains open. Second, the audio experiment uses a reduced spectrogram resolution (8 frequency bins vs. AudioMAE’s 128), limiting the absolute performance achievable; while the within-setup comparison is valid, the factorized advantage under full spectral resolution has not been verified. Third, the pretraining corpus pools clinical EEG recordings from a limited number of sources; performance on substantially different populations or recording protocols has not been evaluated.

## K.2 Future Work

Priority directions include: (1) repeating the AudioMAE experiment at full spectrogram resolution (128 frequency bins) to determine whether the factorized advantage persists when spectral detail is not bottlenecked; (2) investigating whether a task-conditioned gate temperature could resolve the temporal locality tradeoff across clinical and BCI tasks simultaneously; and (3) evaluating AXON on additional downstream mannas.ais such as sleep staging with polysomnography and intracranial EEG.