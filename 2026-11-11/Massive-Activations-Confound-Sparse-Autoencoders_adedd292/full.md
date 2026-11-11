# Massive Activations Confound Sparse Autoencoders

Nathan Delisle University of Chicago ndelisle@uchicago.edu

Chenhao Tan University of Chicago chenhao@uchicago.edu

## Abstract

Sparse autoencoders (SAEs) are commonly evaluated by variance explained and ablation damage. We show how massive activations undermine both evaluations, and propose a correction. Massive activations cause up to 99.9% of activation variance to lie along the first principal component (PC1). Because the PC1 component has no unique sparse decomposition, each SAE run rebuilds it from a different set of features. These features dominate importance rankings: an average of 47 of the 50 most important features are unmatched in a separate run, compared with about 27 of a random 50. A dictionary can also report 99.96% of variance explained while leaving 33% of the variance outside PC1 unexplained. We bypass PC1 by training the SAE on activations with PC1 removed, then adding PC1 to the reconstruction. In affected models, the top-50 feature sharing rate rises from 4–39.5% to 73–99.5%. At the same TopK sparsity, the resulting SAE degrades the model up to 44% less and reduces content reconstruction error by up to 33.9%.

## 1 Introduction

Sparse autoencoders (SAEs) are trained to reconstruct a language model’s internal activations using a sparse set of learned directions, with the aim that each direction corresponds to a meaningful feature used by the model (Cunningham et al., 2023; Bricken et al., 2023). SAEs are primarily evaluated by variance explained, which measures how well a dictionary reconstructs the activations, and ablation damage, which measures how much the model depends on a single feature (Gao et al., 2025; Templeton et al., 2024). On these metrics, SAEs appear to succeed: in GPT-2, SAEs explain 99.6% of activation variance, and removing just one learned feature can raise perplexity by 17% (Figure 1). However, SAE features are known to be unstable across training seeds: Paulo & Belrose (2025) found that about 70% of features differ across training seeds on the same model and data.

Naively, features that are important to the model should recur more reliably across training seeds, because their directions should be more strongly determined by the model and its training data. We find the opposite: the most important features are the least reproducible (Figure 1, left). In five highly anisotropic models, the 50 most important features have mean pairwise sharing rates of only 5–11% across training seeds, compared with dictionary-wide sharing rates of 47–63%.

We trace this inversion to massive activations. In middle layers, 70–99.9% of activation variance lies along the first principal component (PC1) of the residual stream. This is a known effect of massive activations, outlier dimensions written by early feed-forward layers to implement attention sinks (Sun et al., 2024; Xiao et al., 2024; Queipo-de-Llano et al., 2026). While PC1 itself is well-defined, under the TopK objective, there are infinitely many ways to exactly reconstruct an isolated rank-one signal (Appendix C). SAEs trained with different seeds therefore “tile” PC1 with different sets of features. The tiles’ components outside PC1 can cancel, so ablating one tile can cause damage by breaking that cancellation. The tiling can be directly observed by grouping features by their decoder’s alignment with PC1. Cross-seed sharing is at the dictionary baseline for unaligned features, is near zero for moderately PC1-aligned features (tiles), and reaches one for the rare “canonical features” highly aligned with PC1 (Section 4).

![](images/2c0e8044018156a6e136b89785f6ba06dd40bac26ece001e7f6bae4a2e83030c.jpg)

![](images/6236adffaabe52c5d1fa7bea1abccc6fcd09d105fa64cede5df5e29b58ca08dc.jpg)  
Figure 1: Massive activations confound both standard SAE evaluations. Left: The 50 most important features (red) are seed artifacts: 47 of 50 fail to recur across training runs (GPT-2; dashed line indicates sharing threshold). Right: Variance explained, reported vs. excluding PC1. Models are ordered by PC1 dominance. The reported score understates content error by up to 890×.

Massive activations also confound variance explained, which is reported equivalently as the fraction of variance unexplained (FVU). Because reconstruction error is variance-weighted, the reported score is dominated by how well the SAE reconstructs PC1. When reconstruction error along PC1 is negligible, the error on the variance outside PC1, or “content variance,” is understated by the dominance factor: $\mathrm { F V U } _ { \mathrm { c o n t e n t } } / \mathrm { F V U } _ { \mathrm { t o t a l } } \approx 1 / ( 1 - w _ { 1 } )$ , where $w _ { 1 }$ is PC1’s share of activation variance. We measure this approximation and find that it holds on all measured models to within 0.2% (Section 4.3). The reported score can therefore hide substantial content error. Our Qwen-2.5-7B SAEs explain 99.96% of total variance while leaving 33.0% of the content variance unexplained (Figure 1, right).

We test how the sparsity objective contributes to this instability. Because TopK constrains only which features fire, different decompositions of PC1 achieve the same reconstruction loss within the sparsity budget. Adding a magnitude penalty $\beta \| \mathbf { z } \| _ { 1 }$ raises top-50 sharing from 6.4% to 94.9% with less than 1% change in sparsity or reconstruction error (Section 5). In our experiments, TopK, BatchTopK (Bussmann et al., 2024), and JumpReLU with scheduled thresholds (Rajamanoharan et al., 2024b) show the inversion between reproducibility and importance, whereas ReL $\mathbf { \nabla } _ { \mathbf { \boldsymbol { U } } + \mathbf { \boldsymbol { \ell } } _ { 1 } }$ does not. Released dictionaries also show the confound: FVU outside PC1 is approximately 2.1× total FVU in Gemma Scope 2, although normalization substantially reduces this effect in other dictionaries (Section 7.1).

To remove the confound without altering the SAE’s sparsity objective, we train on activations with the PC1 projection removed and add the projection back to the SAE’s output at inference. With PC1 absent from the SAE’s input, top-50 feature sharing rises to 73–99.5%, and importance predicts reproducibility: the ten most important features share at 1.4–2.0 times the dictionary-wide rate in every model. A post-hoc variant that excludes PC1-aligned features from the ranking reaches the same result without retraining. Where the direction dominates, the resulting SAE is also better outright, with up to 44% less degradation of the model it is inserted into and 33.9% less content reconstruction error at the same TopK sparsity (Section 6; Appendix E.4).

Finally, we establish the scope of the confound. In Pythia-1.4B, the inversion disappears at the final layer, where PC1 accounts for much less activation variance (Section 7.2). Excluding the first position removes PC1 dominance where massive activations are mainly present there, as in Llama-3.1-8B (97% to 3%), while Qwen-2.5-7B remains at 99.7% (Appendix J.4). Furthermore, while separating PC1 changes which features the metrics select, the dictionary-wide sharing rate remains between 44% and 68%.

## 2 Background

Feature reproducibility. Because SAE training is non-convex, different random initializations can result in different learned features even when the training data and hyperparameters are fixed. Paulo & Belrose (2025) compare independently initialized SAEs trained on identical data and find that only 30% of features are shared between runs on Llama-3-8B. They also find that at similar sparsity, ReL $\mathbf { \nabla } _ { \mathbf { \boldsymbol { U } } + \mathbf { \boldsymbol { \ell } } _ { 1 } }$ SAEs share more features across runs than TopK SAEs do. Gerasimov et al. (2026) find that while individual features often fail to recur across runs, the subspaces they span largely do. They find that stable features carry most of the reconstruction- and prediction-relevant signal. Sun et al. (2025) identify pairs of dense latents with opposing decoder directions aligned with PC1 in Gemma Scope. We identify massive activations as a particularly impactful source of this instability: independently trained SAEs reconstruct the same dominant PC1 component using different features, which can dominate the importance ranking.

SAE evaluation. Alongside reconstruction quality, SAEBench evaluates SAE performance using concept detection, feature disentanglement, and automated interpretability, and finds that improvements in the sparsity–fidelity trade-off do not reliably improve these measures (Karvonen et al., 2025). Saraswatula & Klindt (2025) find that PCA whitening improves sparse probing for both ReLU and TopK SAEs, despite reducing reconstruction fidelity. Chanin (2026) identifies evaluation noise and failures to track ground-truth feature recovery in several benchmark metrics. Engels et al. (2025b) show that much of SAE reconstruction error is linearly predictable from the input activations. Timkey & van Schijndel (2021) show that a few rogue dimensions dominate cosine similarity between contextual embeddings. Our results demonstrate that PC1 plays an analogous role in SAE fidelity.

Massive activations and attention sinks. Massive activations are unusually large values concentrated in a few activation dimensions and token positions. Sun et al. (2024) show that these values can be mostly input-independent and attract attention to the associated tokens, connecting them to attention sinks (Xiao et al., 2024). Yu et al. (2024) identify individual weights in early MLPs that generate massive activations and whose removal can severely degrade language-model performance. These outliers can cause PC1 to account for most of the activation variance. For uncentered activations, Queipo-de-Llano et al. (2026) show how large token norms can make a single direction account for most of the total squared activation magnitude. In vision transformers, Darcet et al. (2024) find that high-norm background patches are reused for internal computation, and introduce register tokens so that the model can store intermediate results without using background patch tokens. Jiang et al. (2025) find the neurons that produce these high-norm patches and move their output into register tokens added at inference, without retraining the model. These studies explain the function and origin of massive activations; we investigate how massive activations affect SAE learning and evaluation.

## 3 Experimental Setup

SAEs. We train SAEs on residual stream activations $\textbf { x } \in \mathbb { R } ^ { d }$ from GPT-2, Pythia-70Mdeduped/160M/1.4B/6.9B, Qwen-2.5-7B, Llama-3.1-8B/70B, and Gemma-2-2B/9B. Unless stated otherwise, activations are taken at the output of the middle transformer block over 5M tokens of WikiText-103 (Merity et al., 2017), with one document per sequence and padding excluded. We train TopK SAEs (Gao et al., 2025) with $m = 4 d$ dictionary features and sparsity $k = 6 4 .$ , using Adam for 20,000 steps with a learning rate of $3 \times 1 0 ^ { - 4 }$ and 4096 activations per batch, and renormalizing the decoder directions to unit norm after each step. Initialization is varied across seeds. The appendix lists mini-batch order, seed counts and further information on the training settings.

Metrics. Reconstruction fidelity is measured by the fraction of variance unexplained, $\mathrm { F V U = }$ $\mathbb { E } \| \mathbf { x } - \hat { \mathbf { x } } \| ^ { 2 } / \mathbb { E } \| \mathbf { x } - \bar { \mathbf { x } } \| ^ { 2 }$ , where x¯ is the training mean. Let $\mathbf { v } _ { 1 }$ be the unit PC1 direction of the centered training activations, $w _ { 1 }$ its share of variance in the evaluated population, and $P _ { \perp } = I - \mathbf { v } _ { 1 } \mathbf { v } _ { 1 } ^ { \top }$ . Content FVU measures reconstruction outside PC1: $\mathrm { F V U } _ { \mathrm { c o n t e n t } } = \hat { \mathbb { E } } \Vert \hat { P } _ { \perp } ( \mathbf { x } - \hat { \mathbf { x } } ) \Vert ^ { 2 } / \mathbb { E } \Vert P _ { \perp } ( \mathbf { x } - \bar { \mathbf { x } } ) \Vert ^ { 2 }$ . Feature importance, $I _ { j }$ , is the increase in cross-entropy when feature j is set to zero in the SAE reconstruction (relative to the same model with the non-ablated SAE). Splice cost is the increase in cross-entropy from replacing the original activation with its SAE reconstruction.

Reproducibility and PC1 alignment. We measure cross-seed reproducibility using decoder similarity (Paulo & Belrose, 2025). Feature j is shared with another seeded SAE if max<sub>l</sub> cos $( \mathbf { d } _ { j } , \mathbf { d } _ { l } ^ { \prime } ) \geq 0 . 7$ where the maximum is over all decoder directions in that SAE. The sharing rate is the fraction of selected features satisfying this criterion. Dictionary-wide sharing uses all alive features. A feature’s PC1 alignment is $a _ { j } = \cos ^ { 2 } ( \mathbf { d } _ { j } , \mathbf { v } _ { 1 } )$

## 4 The Confound

## 4.1 Reproducibility of Important Features

For each model, we rank SAE features by zero-ablation importance and compare the mean pairwise sharing rate of the top 50 with that of the full dictionary (Figure 2, left). In every model where PC1 explains at least 93% of activation variance $( w _ { 1 } \ge 0 . 9 3 )$ ), the top 50 most important features are shared at a lower rate than the full dictionary. In five of these models (GPT-2, Qwen-2.5-7B, and Pythia-160M/1.4B/6.9B), an average of 43.5–48 of the top 50 features fail to recur in another run, where choosing 50 features at random from the full dictionary would leave an average of 19–27 without a match. Across all models we evaluate, the top-50 sharing rate falls strongly with PC1 dominance (Pearson $r = - 0 . 9 3 .$ , Spearman $\rho = - 0 . 7 3$ against $w _ { 1 } )$ .

This failure mainly affects the very top of the importance ranking. Across all features, importance and reproducibility are weakly correlated: the rank correlation between importance and reproducibility is between +0.11 and +0.19 in every model. Although the overall correlation is weak, the top of the ranking is commonly used to select features. For example, ablation-based importance is used to select features when tracing feature circuits (Cunningham et al., 2023), and identifying spurious features in classifiers (Marks et al., 2025).

It is reasonable to ask whether these results depend on measuring sharing among the top 50 features specifically. Figure 2 (right) varies the number of top features N from 10 to the full dictionary. For the five PC1-dominated models, the mean stays below the dictionary-wide baseline for $1 0 \leq N < 1 5 0 .$ . In the Gemma models, which do not show the inversion, the ranking already selects ordinary features, so sharing declines as more features are included. Both curves reach 1 at the right edge by construction, where the top N is the whole dictionary. Per-model values are given in Table 8.

In a separate replication using GPT-2 and 10 seeds, we find that the confound persists under both cross-entropy and KL rankings. One-to-one matching changes the baseline top-50 sharing rates by less than 0.3 percentage points. Jointly resampling seeds and evaluation articles also does not affect the result. Our controls are described in Appendix D.3.

![](images/545818c58f350cc11a1da4d5a55d66450d59ee66a2150e33c2d57e38357bc061.jpg)

![](images/eb849d663d851a7c6156a89c1b97a672bd768d4c46b6561d83915a953388d54f.jpg)  
Figure 2: Importance selects less reproducible features in seven of ten models. Left: mean sharing across seed comparisons for the top 50 features by importance (red) and the whole dictionary (gray), ordered by PC1 dominance. The Gemma models below the dotted line show the opposite pattern. Right: mean and range of top-N sharing relative to dictionary-wide sharing: GPT-2, Qwen-2.5-7B, and Pythia-160M/1.4B/6.9B (red); Gemma-2-2B/9B (blue). The dashed line denotes equal sharing.

Gemma-2-2B and Gemma-2-9B, the two models with the lowest PC1 dominance $( w _ { 1 } = 0 . 7 8 , 0 . 7 0 )$ have top-50 sharing rates of 87% and 92%. This result is consistent with the confound depending on PC1. Llama-3.1-70B and Llama-3.1-8B have top-50 sharing rates of 38% and 56%, below their dictionary-wide rates but above the fully inverted models. In these models, a smaller fraction of ablation importance falls on PC1-aligned features. This is because Llama’s outlier signal is concentrated at the BOS position (Sun et al., 2024), so its PC1-tiling features fire almost exclusively there (62 features, 3% shared across seeds; Appendix J.4). Qwen shows a stronger inversion, with top-50 sharing of 6%.

## 4.2 Reproducibility and PC1 Alignment

We group each model’s features by their decoder’s alignment with PC1 and measure the sharing rate within each group (Figure 3, left). The resulting features fall into three categories. Features with alignment $a _ { j } = \cos ^ { 2 } ( \mathbf { d } _ { j } , \mathbf { v } _ { 1 } )$ below $1 0 ^ { - 2 }$ share at the normal, dictionary-wide rate. Features at intermediate alignment share at $0 . 0 { - } 2 . 6 \%$ : these are the features that jointly reconstruct PC1, which we call tiles. In every model, the tiles occupy one contiguous range of alignment, beginning at 0.01 in GPT-2, Qwen-2.5-7B, and Llama-3.1-70B and at 0.03 in the other seven models. Each model has 38–106 tiles, about 1% of all alive features (Appendix B).

The third category is the canonical feature: a feature that is highly-aligned with PC1 (alignment at least 0.6). Four of the ten dictionaries contain exactly one, Gemma-2-2B contains two, and the remaining five contain none. Where present, canonical features recur in every run.

Tiles make up about 1% of alive features, but are responsible for 2–66 times their share of total ablation importance, with the two lowest values in the Gemma family (Figure 3, middle). In the most affected models, these irreproducible features occupy the top of the importance ranking, explaining the inversion in Section 4.1.

We test whether these large ablation effects arise from removing PC1 by separately ablating the PC1 and orthogonal components of each top-50 feature’s reconstruction contribution. Removing only the orthogonal component preserves the SAE’s PC1 reconstruction but reproduces 88–100% of the mean whole-feature cross-entropy increase across GPT-2, Pythia-160M/1.4B, and Qwen-2.5-7B (Appendix C.2), consistent with ablation disrupting cancellation between features outside PC1.

## 4.3 Fidelity and the Dominance Factor

Because FVU is a variance-weighted average over directions, it can be split along PC1. Writing FVU<sub>PC1</sub> for the same ratio computed on the PC1 component alone, $\mathrm { F V } \mathrm { \bar { U } _ { t o t a l } } = \mathrm { \bar { \it w } _ { 1 } F V U _ { P C 1 } } \mathrm { \bar { + } }$ $( 1 - w _ { 1 } ) \mathrm { F V U _ { c o n t e n t } }$ . The SAE reconstructs PC1 nearly exactly: error along PC1 accounts for less than 0.3% of total reconstruction error. Neglecting this contribution, the reported score understates content error by approximately the dominance factor: $\mathrm { F V U } _ { \mathrm { c o n t e n t } } / \mathrm { F V U } _ { \mathrm { t o t a l } } \approx 1 / ( 1 - w _ { 1 } )$ .

![](images/312d4778412afb2829a355c8916d187856e2d6c2d3aa4ec0055a7f0fb13521df.jpg)

![](images/b9a483c6e054a16a45c79ad5b5b410cef237b1aa12126afaadf65e3146f82155.jpg)

![](images/edc546305384ee22a91c66b389797506f35a8246ab34f9fc12d4fe92b8e5669c.jpg)  
Figure 3: Left: cross-seed similarity (best-match decoder cosine) against PC1 alignment for every sampled feature in Pythia-1.4B. Middle: the tiles’ share of total ablation importance divided by their share of the dictionary, for all ten models. Right: the measured ratio $\mathrm { F V U _ { c o n t e n t } / F V U _ { t o t a l } }$ against the prediction $1 / ( 1 - w _ { 1 } )$ for standard SAEs across nine models, using the same held-out evaluation as Figure 1.

This approximation has a relative error below 0.2% (Figure 3, right; full values in Appendix E.1). The reported score therefore conceals substantial reconstruction error outside PC1. Our Qwen-2.5-7B SAEs explain 99.96% of activation variance while leaving 33.0% of the content variance unexplained, an understatement of approximately 890×. Figure 1 (right) shows this comparison across all models.

Prior work has argued that some model structure resists sparse linear decomposition (Engels et al., 2025a; Bhalla et al., 2026). In our experiments, while SAEs reconstruct content better than PCA baselines, they still leave substantial error. At 64 active coefficients, our standard SAEs outperform both rank-64 PCA and the adaptive 64-component PCA baseline, but still leave 12–33% of the content variance unexplained (Appendix E.2). While bypass improves this reconstruction, reducing content error by 33.9% in Qwen-2.5-7B and by 0.8–4.9% in the other tested models (Section 6), a substantial fraction of the content variance remains unexplained.

## 5 Cause of the Confound

The PC1 component can be reconstructed exactly by one aligned feature or several partially aligned features whose components outside PC1 cancel. For an isolated PC1 component, either representation achieves zero reconstruction loss within the sparsity budget; TopK adds no penalty on activation magnitudes.

We test this explanation by adding an activation-magnitude penalty $\beta \| \mathbf { z } \|$ <sub>1</sub> to the loss. Because our decoder directions have unit norm, a partially aligned feature needs a larger activation to contribute the same amount to PC1. For an exact reconstruction of PC1, partially aligned features require a larger sum of activation magnitudes, so the $\ell _ { 1 }$ penalty prefers PC1-aligned features (Appendix C). We test whether this change makes important features more reproducible, then compare the pattern across SAE objectives.

## 5.1 Objective and Canonical Initialization

We add a penalty $\beta \| \mathbf { z } \| _ { 1 }$ to the TopK objective and measure the reproducibility of important features while tracking sparsity and reconstruction loss. On GPT-2 at block 6, we train five seeds at four levels of magnitude penalty $\beta \in \{ 0 , 0 . 0 3 , 0 . 1 , 0 . 3 \}$ , keeping the remaining training settings fixed. At $\beta = 0 . 3$ , mean pairwise sharing among the top 50 features rises from 6.4% to 94.9%. Sparsity and reconstruction stay roughly constant: mean $L _ { 0 }$ goes from 64.00 to 63.47, and content FVU goes from 0.1394 to 0.1387 (Figure 4, left).

![](images/7630fe468fc30b44d99946ef99e0dfc424339a01c536f24e8cc08cb00142fc9f.jpg)

Aligned features become more reproducible  
![](images/b00df018f1c80565db9de24f31b93ea15161f765982be9d5a7573c524772e0b6.jpg)  
Figure 4: Left: top-50 sharing on GPT-2 against the magnitude penalty $\beta .$ . Sharing rises from 0.06 to 0.95, while $L _ { 0 }$ and content FVU stay within 1%; both controls normalized to $\beta = 0$ . Right: cross-seed similarity against PC1 alignment for every feature at $\beta = 0 ( \mathrm { { g r a y } ) }$ and $\beta = 0 . 3$ (red), one point per feature. The dashed line indicates the sharing threshold. The dotted curve is the similarity two features obtain from their PC1 components alone.

<table><tr><td>Model</td><td>W1</td><td>TopK</td><td>BatchTopK</td><td>JumpReLU*</td><td>Gated</td><td>ReLU+l1</td></tr><tr><td>Qwen-2.5-7B</td><td>.999</td><td>6.0</td><td>33.0</td><td>32.0</td><td>47.0†</td><td>97.0</td></tr><tr><td>GPT-2</td><td>.970</td><td>5.5</td><td>17.5</td><td>11.5</td><td>11.0</td><td>100.0</td></tr><tr><td>Llama-3.1-8B</td><td>.969</td><td>56.0</td><td>62.0</td><td>61.0</td><td>99.0</td><td>97.0</td></tr><tr><td>Pythia-1.4B</td><td>.964</td><td>10.0</td><td>14.0</td><td>12.0</td><td>86.0</td><td>98.5</td></tr><tr><td>Pythia-6.9B</td><td>.957</td><td>10.0</td><td>18.0</td><td>26.0</td><td>15.0</td><td>97.0</td></tr><tr><td>Pythia-160M</td><td>.934</td><td>9.0</td><td>9.5</td><td>14.5</td><td>87.0</td><td>96.5</td></tr><tr><td>Gemma-2-2B</td><td>.781</td><td>80.5</td><td>72.5</td><td>76.0</td><td>79.0</td><td>97.0</td></tr><tr><td>Gemma-2-9B</td><td>.700</td><td>90.0</td><td>90.0</td><td>75.0</td><td>63.0</td><td>95.0</td></tr></table>

Table 1: Top-50 sharing (%), averaged over comparisons between a reference seed and the other training seeds. Red, bold entries fall below the corresponding dictionary-wide sharing rate; blue entries exceed or match it. <sup>∗</sup>Scheduled thresholds. <sup>†</sup>One Qwen Gated seed diverged. <sup>‡</sup>Sensitive to sparsity calibration (Appendix D.2).

Features with PC1 alignment $0 . 0 1 \leq a _ { j } < 0 . 3$ still make up 73% of the top 50 at $\beta = 0 . 3$ , compared to 93.6% of the top 50 without the penalty. Their number increases from 65 to $^ { 1 , 3 0 7 }$ per seed, but their sharing rate also rises from 3% to 72%. Full results are in Appendix D.1.

The magnitude penalty also increases cross-seed similarity in the decoder components outside PC1. For a fixed pair of seeds, we match the top-50 features by decoder cosine and then project PC1 out of both matched directions. Mean cosine similarity between the remaining components rises from 0.165 to 0.831 on GPT-2 as $\beta$ increases from 0 to 0.3, and from 0.114 to 0.910 on Pythia-1.4B as $\beta$ increases from 0 to 0.1 (Appendix D.1).

We also test whether initializing the SAE with a single PC1-aligned feature prevents PC1 tiling under the original TopK objective. We initialize one feature’s decoder direction along PC1 and do not hold it fixed during training. In a GPT-2 experiment with 10 seeds, the top-50 sharing rate rises from 5.5% to 68%, and the median tile count drops from 65 to one. Canonical initialization worsens reconstruction loss, however: median content FVU rises from 0.149 to 0.165. Doubling dictionary width with standard initialization leaves the median tile count at 64 (Appendix D.3).

## 5.2 Reproducibility Across SAE Objectives

Table 1 shows the sharing rates of the top-50 most important features across different SAE objectives. For TopK, BatchTopK, and JumpReLU with scheduled thresholds, in models where PC1 explains at least 93% of variance, the top 50 features are shared 7–53 percentage points below the dictionary-wide sharing rate (Table 1). ReL $\mathbf { \nabla } . \mathbf { U } \mathbf { + } \boldsymbol { \ell } _ { 1 }$ , which directly penalizes the reconstruction coefficients, has top-50 sharing of 95–100% in every model, consistently above its dictionary-wide rate. Neither Gemma model shows the inversion under any objective.

Gated’s sparsity penalty acts on gate activations, giving no gradient to the magnitude-only parameters (Rajamanoharan et al., 2024a). We test whether this contributes to seed dependence by adding a penalty on the reconstruction coefficients. On GPT-2, adding a penalty $\beta = 0 . 3$ raises top-50 sharing from 11% to 98%. Mean held-out content FVU decreases from 0.157 to 0.146, while mean $L _ { 0 }$ increases from 56.8 to 59.9. At $\beta = 0 . 1$ , we apply the added penalty’s gradient only to the magnitude parameters or only through the shared encoder. Top-50 sharing is 89% and 10%, respectively (Appendix D.2).

## 6 Repair

## 6.1 Bypass

We compute a unit PC1 direction $\mathbf { v } _ { 1 }$ from centered training activations. For each centered activation $\mathbf { x } ,$ we subtract $( \mathbf { v } _ { 1 } ^ { \top } \mathbf { x } ) \mathbf { v } _ { 1 }$ and train the SAE to reconstruct the remaining activation. Before passing the reconstructed activation to the language model, we add the removed PC1 component and training mean back. This preserves the signal that the underlying language model uses for attention sinks, while allowing the SAE to reconstruct the remaining structure.

In all models that show the inversion, top-50 sharing increases, from 4–39.5% to 73–99.5% (Figure 5, left). The two Gemma models already have highly reproducible top-ranked features and show smaller changes: 87% to 93.5% on Gemma-2-2B and 92% to 94.5% on Gemma-2-9B.

Because removing PC1 alters the SAE’s training inputs, improved reproducibility could come at the cost of less accurate reconstruction of the remaining activation. We test this by comparing held-out content FVU between standard and PC1-bypass SAEs. Both SAEs use a TopK budget of 64 active features per token. Bypass also carries a separate PC1 coefficient, computed directly from the input.

Content reconstruction improves under bypass. Held-out content FVU decreases by 0.8–33.9% in every model (Figure 5, right). The gain likely comes from freed SAE capacity: on massive-activation tokens, a standard SAE spends about 63 of its 64 active coefficients on features with $a _ { j } \geq 0 . 0 1$ in GPT-2, Pythia-1.4B, and Qwen-2.5-7B. Bypass also improves reconstruction on ordinary tokens, which account for approximately 84% of GPT-2’s reduction in content error. Giving bypass 63 sparse coefficients plus the PC1 coefficient (the same total budget as the standard SAE) retains 68–80% of the content reconstruction gain on GPT-2, Pythia-1.4B, and Gemma-2-2B, and 99.4% on Qwen-2.5-7B (Appendix E.3).

Resample ablation. Zeroing a feature could itself cause the instability we observe. We rank features by resample ablation, which replaces a feature’s activation with draws from its own activation distribution, under marginal resampling (at every token) and support-preserving resampling (nonzero draws only where the feature fires). On GPT-2, marginal resampling results in a top-50 sharing rate of 4.0% before bypass and 77.5% after. Support-preserving resampling results in top-50 rates of 83.0% and 85.0% respectively. In Pythia-1.4B, the support-preserving rates are 27.5% and 95.5%. Thus, instability persists under nonzero replacement, although the extent to which this is true depends on the specific replacement used (Appendix B.4).

## 6.2 Post-hoc Feature Selection

PC1 bypass is a substantial intervention: it requires retraining the SAE on activations with PC1 removed, which changes the features it learns. For an existing SAE, a more conservative alternative is to exclude PC1-aligned features from the importance ranking post-hoc.

We exclude features with squared PC1 alignment $a _ { j } \geq 0 . 0 1$ from the ranking pool, targeting the PC1-tiling features described in Section 4.2. This removes a small population of features (only 0.72–3.92% of alive features) from consideration, while the 50 most important remaining features are shared at 77.5–96% across seeds, comparable to the rates obtained with trained bypass (Figure 5, left). Because the dictionary is unchanged, its reconstruction remains identical.

On GPT-2, the selected top 50 achieve 77.5% cross-seed sharing and have a mean ablation effect of ∆CE = 0.0130. Their individual effects are smaller than those of the original, confounded top 50 (0.0966), but similar to those obtained with trained bypass (0.0141).

## 6.3 Interpretability and Sparse Probing

Automated interpretability. We evaluate the 100 most important features and 100 randomly selected alive features per condition, using one middle-layer SAE seed on each of eight models. An instruction-tuned model explains each feature from activating contexts; a second model scores whether that explanation predicts activation on held-out contexts. We exchange the explainer and evaluator roles between Qwen2.5-32B-Instruct and Mistral-Small-24B-Instruct-2501. For the topranked features, mean AUROC against random inactive contexts increases after bypass on GPT-2, all three evaluated Pythia models, and Qwen-2.5-7B in both directions. On Pythia-1.4B, it rises from 0.551 to 0.732 with Qwen explaining and Mistral scoring, and from 0.669 to 0.743 with Mistral explaining and Qwen scoring. Scores fall on Llama-3.1-8B and Gemma-2-2B and rise slightly on Gemma-2-9B (Appendix G).

Sparse probing. We compare standard and bypass features by fitting logistic probes on SST-2, AG News, Amazon Polarity, and Emotion. With the 20 standard or bypass features of highest trainingsplit mutual information, and mean pooling over up to 64 tokens, the bypass–standard difference in accuracy averaged over the four tasks is less than one percentage point in magnitude for every model. Single model–task differences run from −2.44 to +3.13 points (Appendix H).

![](images/7240e44e75498db5a1a4689b27ea709c6603ccdb7f76b151b9666e7261926311.jpg)

![](images/2180d2c92ef3c78eccf68a0643fb70ef44ec19b91e9ae4859fab360f5a3581bd.jpg)  
Figure 5: Left: top-50 sharing for standard TopK SAEs, for the same dictionaries with PC1-aligned features excluded from the ranking (post-hoc), and for SAEs trained with PC1 bypass. Right: heldout content FVU for the standard SAE and for bypass, with the relative reduction. Because post-hoc selection uses the standard dictionary, its fidelity is the standard value.

## 6.4 Behavioral Interventions

We test whether the corrected feature importance ranking gives better features for steering, using subject–verb agreement as the task. We greedily select five of each ranking’s top 50 features and patch their activations to flip a sentence’s subject–verb agreement, with a control that patches from a sentence where only a non-subject noun changed (setup in Appendix F). Success requires switching the model’s preference between is and are after the subject edit, while leaving it unchanged after the control edit. Averaged over three seeds, post-hoc selection raises success from 0% to 9.0%, 31.9%, and 4.2% on GPT-2, Pythia-1.4B, and Qwen-2.5-7B; bypass gives success rates of 7.1%, 37.5%, and 2.0%. Gemma-2-2B’s success rate is constant at 64–65% regardless of condition. Features matched across seeds give the same rates to within 0.5 points.

## 7 Prevalence

## 7.1 Released SAE Dictionaries

We compare total and content FVU for publicly released SAE dictionaries (Lieberum et al., 2024; He et al., 2024; McDougall et al., 2025; Deng et al., 2026) to assess how accurately they reconstruct activations outside PC1 (Figure 6, left). The median content-to-total FVU ratios are 2.1 for Gemma Scope 2, 1.49 for Pythia, and 98.8 for unfiltered Qwen Scope. Total and content FVU are much closer in Gemma Scope, Llama Scope, and OpenAI’s GPT-2 dictionaries, with median ratios of 1.02–1.05.

Preprocessing helps explain these differences. OpenAI’s GPT-2 SAEs center and normalize each token’s activation before encoding (Gao et al., 2025). For the same GPT-2 dictionary, the ratio is 10.9 in the original activation space and 1.03 in the normalized space. Deng et al. (2026) remove extreme-norm activations when training Qwen Scope, but do not specify the cutoff that they use. We test three cutoffs: 5, 10, and 20 times the median activation norm. Each removes about 0.8% of activations at the early and middle layers tested. Without these extreme-norm activations, the content-to-total FVU ratios fall from 92–220 to about 1.03.

We train SAEs on Qwen-2.5-7B with regularized PCA whitening and compare them with PC1 bypass at 64 total coefficients. Whitening lowers content error by 39.4% relative to the standard SAE; bypass lowers content error by 37.6%. Total error increases to 2.4 times the standard value with whitening but decreases under bypass (Appendix K.2).

![](images/9346d4aa293173bd41f76ed497d8e794ca09b4e59564a53a3686896b9940caf1.jpg)

![](images/7dce90f1b6e10c679f70784ab5612e3fa75a0c901a9af0ba26ed055e4634f91a.jpg)  
Figure 6: Left: content-to-total FVU ratios for released SAEs on held-out text. Each point denotes a single checkpoint. Qwen Scope is shown with and without approximate norm filtering. Hollow markers indicate total FVU above one. Normalization and token-selection conventions are described in Appendix K.1. Right: total and content FVU for BatchTopK SAEs trained on DINOv2-base block-8 patch activations, averaged over three seeds after 10,000 training steps.

## 7.2 Variation Across Layers

In a separate experiment, PC1 explains 93–94% of activation variance in Pythia-1.4B at layers 3 and 12. Bypass raises top-50 sharing from 14% to 68% at block 3 and from 8% to 92% at block 12. At final block 23, where PC1 explains only 11% of variance, top-50 sharing is already 96% without bypass and does not improve under bypass.

## 7.3 Vision Transformers

In DINOv2-base, a self-supervised vision transformer (Oquab et al., 2023), PC1 explains 96.2% of the variance in patch-token activations. BatchTopK SAEs achieve a total FVU of 0.013, while content FVU is 0.343—approximately 26 times larger (Figure 6, right). PC1 bypass modestly improves content FVU to 0.333, compared with 0.344 under random-direction bypass. Pretrained and test-time registers, following (Darcet et al., 2024) and (Jiang et al., 2025), reduce PC1’s share of patch-activation variance to approximately 4% and nearly eliminate the gap between total and content FVU. Full results are reported in Appendix L.

Crosscoders. We train crosscoders on four pairs of activation sets from Pythia-1.4B. With PC1 bypassed in both inputs, 83–93% of the top 50 features are shared across seeds, up from 3–4%. Normalizing each input or bypassing a random direction does not give comparable gains (Appendix M).

## 8 Conclusion

Massive activations confound both feature importance and reconstruction loss in sparse autoencoder evaluation. Accurately reconstructing a dominant PC1 component can hide substantial error in reconstructing the remaining activation, and selecting features with large ablation effects often yields seed-dependent, PC1-reconstructing features.

PC1 bypass improves content reconstruction and the reproducibility of top-ranked features in affected models. For existing dictionaries, post-hoc selection is a simple way to obtain both reproducible and important features without retraining or altering the existing dictionary. Our results show how accounting for dominant activation structure can improve evaluation and feature selection. More broadly, our findings motivate testing whether other interpretability methods are similarly confounded by dominant structure.

## Acknowledgments and Disclosure of Funding

This work was supported in part by compute credits from Amazon Web Services through the AWS Activate program.

The authors declare no competing interests.

## References

U. Bhalla, T. Fel, C. Rager, S. Feucht, et al. Do sparse autoencoders capture concept manifolds? arXiv:2604.28119, 2026.

T. Bricken, A. Templeton, J. Batson, B. Chen, et al. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread, 2023. https: //transformer-circuits.pub/2023/monosemantic-features/.

B. Bussmann, P. Leask, and N. Nanda. BatchTopK sparse autoencoders. arXiv:2412.06410, 2024.

D. Chanin. Are sparse autoencoder benchmarks reliable? arXiv:2605.18229, 2026.

H. Cunningham, A. Ewart, L. Riggs, R. Huben, and L. Sharkey. Sparse autoencoders find highly interpretable features in language models. arXiv:2309.08600, 2023.

T. Darcet, M. Oquab, J. Mairal, and P. Bojanowski. Vision transformers need registers. In ICLR, 2024.

B. Deng, X. Wang, Y. Wang, Y. Wan, et al. Qwen-Scope: Turning sparse features into development tools for large language models. arXiv:2605.11887, 2026.

J. Engels, E. J. Michaud, I. Liao, W. Gurnee, and M. Tegmark. Not all language model features are one-dimensionally linear. In ICLR, 2025a.

J. Engels, L. Smith, and M. Tegmark. Decomposing the dark matter of sparse autoencoders. Transactions on Machine Learning Research, 2025b.

J. Gao, D. He, X. Tan, T. Qin, L. Wang, and T.-Y. Liu. Representation degeneration problem in training natural language generation models. In ICLR, 2019.

L. Gao, T. Dupré la Tour, H. Tillman, G. Goh, R. Troll, A. Radford, I. Sutskever, J. Leike, and J. Wu. Scaling and evaluating sparse autoencoders. In ICLR, 2025.

G. Gerasimov, T. Rusalev, N. Balagansky, D. Laptev, V. Kurochkin, and D. Gavrilov. Unstable features, reproducible subspaces: Understanding seed dependence in sparse autoencoders. arXiv:2606.12138, 2026.

A. Gokaslan and V. Cohen. OpenWebText corpus, 2019. https://skylion007.github.io/ OpenWebTextCorpus/.

Z. He, W. Shu, X. Ge, L. Chen, et al. Llama Scope: Extracting millions of features from Llama-3.1-8B with sparse autoencoders. arXiv:2410.20526, 2024.

N. Jiang, A. Dravid, A. A. Efros, and Y. Gandelsman. Vision transformers don’t need trained registers. In NeurIPS, 2025.

A. Karvonen, C. Rager, J. Lin, C. Tigges, et al. SAEBench: A comprehensive benchmark for sparse autoencoders in language model interpretability. In ICML, 2025.

T. Lieberum, S. Rajamanoharan, A. Conmy, L. Smith, N. Sonnerat, V. Varma, J. Kramár, A. Dragan, R. Shah, and N. Nanda. Gemma Scope: Open sparse autoencoders everywhere all at once on Gemma 2. arXiv:2408.05147, 2024.

S. Marks, C. Rager, E. J. Michaud, Y. Belinkov, D. Bau, and A. Mueller. Sparse feature circuits: Discovering and editing interpretable causal graphs in language models. In ICLR, 2025.

C. McDougall, A. Conmy, J. Kramár, T. Lieberum, S. Rajamanoharan, and N. Nanda. Gemma Scope 2—Technical paper. Technical report, Google, 2025. https://deepmind.google/ models/gemma/gemma-scope/.

S. Merity, C. Xiong, J. Bradbury, and R. Socher. Pointer sentinel mixture models. In ICLR, 2017.

M. Oquab, T. Darcet, T. Moutakanni, H. Vo, et al. DINOv2: Learning robust visual features without supervision. arXiv:2304.07193, 2023.

G. Paulo and N. Belrose. Sparse autoencoders trained on the same data learn different features. arXiv:2501.16615, 2025.

E. Queipo-de-Llano, Á. Arroyo, F. Barbero, X. Dong, M. Bronstein, Y. LeCun, and R. Shwartz-Ziv. Attention sinks and compression valleys in LLMs are two sides of the same coin. In ICLR, 2026.

S. Rajamanoharan, A. Conmy, L. Smith, T. Lieberum, V. Varma, J. Kramár, R. Shah, and N. Nanda. Improving dictionary learning with gated sparse autoencoders. arXiv:2404.16014, 2024a.

S. Rajamanoharan, T. Lieberum, N. Sonnerat, A. Conmy, V. Varma, J. Kramár, and N. Nanda. Jumping ahead: Improving reconstruction fidelity with JumpReLU sparse autoencoders. arXiv:2407.14435, 2024b.

A. Saraswatula and D. Klindt. Data whitening improves sparse autoencoder learning. arXiv:2511.13981, 2025.

M. Sun, X. Chen, J. Z. Kolter, and Z. Liu. Massive activations in large language models. In COLM, 2024.

S. Sun, A. Canziani, Y. LeCun, and J. Zhu. The spike, the sparse and the sink: Anatomy of massive activations and attention sinks. arXiv:2603.05498, 2026.

X. Sun, A. Stolfo, J. Engels, B. Wu, S. Rajamanoharan, M. Sachan, and M. Tegmark. Dense SAE latents are features, not bugs. In NeurIPS, 2025.

M. Sundararajan, A. Taly, and Q. Yan. Axiomatic attribution for deep networks. In ICML, pp. 3319– 3328, 2017.

A. Templeton, T. Conerly, J. Marcus, J. Lindsey, et al. Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet. Transformer Circuits Thread, 2024. https:// transformer-circuits.pub/2024/scaling-monosemanticity/.

W. Timkey and M. van Schijndel. All bark and no bite: Rogue dimensions in transformer language models obscure representational quality. In EMNLP, pp. 4527–4546, 2021.

G. Xiao, Y. Tian, B. Chen, S. Han, and M. Lewis. Efficient streaming language models with attention sinks. In ICLR, 2024.

M. Yu, D. Wang, Q. Shan, C. J. Reed, and A. Wan. The super weight in large language models. arXiv:2411.07191, 2024.

## A Experimental Methods

## A.1 Models, activations, and training data

Table 2 describes where activations are taken from. In all models, transformer layers are numbered starting from 0. Unless output normalization is explicitly stated, we take activations after the layer’s attention and MLP, before the next layer’s normalization. Pythia-70M uses the deduplicated checkpoint, whereas the larger Pythia models use the standard checkpoints.

Table 2: Models, layers used to collect activations, and default SAE widths.
<table><tr><td>Model</td><td></td><td>Layers Layer used</td><td></td><td> $d \ m = 4 d$ </td></tr><tr><td>GPT-2</td><td>12</td><td>6</td><td>768</td><td>3,072</td></tr><tr><td>Pythia-70M</td><td>6</td><td>3</td><td>512</td><td>2,048</td></tr><tr><td>Pythia-160M</td><td>12</td><td>6</td><td>768</td><td>3,072</td></tr><tr><td>Pythia-1.4B</td><td>24</td><td></td><td>12 2,048</td><td>8,192</td></tr><tr><td>Pythia-6.9B</td><td>32</td><td></td><td>164,096</td><td>16,384</td></tr><tr><td>Llama-3.1-8B</td><td>32</td><td></td><td>164,096</td><td>16,384</td></tr><tr><td>Llama-3.1-70B</td><td>80</td><td></td><td>408,192</td><td>32,768</td></tr><tr><td>Qwen-2.5-7B</td><td>28</td><td></td><td>14 3,584</td><td>14,336</td></tr><tr><td>Gemma-2-2B</td><td>26</td><td></td><td>13 2,304</td><td>9,216</td></tr><tr><td>Gemma-2-9B</td><td>42</td><td></td><td>21 3,584</td><td>14,336</td></tr></table>

By default, we train five standard and five bypass SAEs per model, using seeds $\{ 4 2 , 1 3 7 , 0 , 1 , 2 \}$ Some experiments use fewer seeds. Seed counts are given alongside the experimental results.

We train SAEs on five million token activations from the wikitext-103-v1 training split of WikiText-103 (Merity et al., 2017). Each nonempty text entry is tokenized separately, truncated to 128 tokens, and right-padded within batches. Default special tokens are retained; padding activations are excluded. Language models run in evaluation mode in float16, except Gemma, which uses bfloat16. SAEs are trained in float32.

Before SAE training, we subtract the mean $\pmb { \mu }$ of these five million activations. We estimate PC1, $\mathbf { v } _ { 1 } ,$ by a float64 covariance eigendecomposition of the first 50,000 activations centered by this mean. We use $P = \mathbf { v } _ { 1 } \mathbf { v } _ { 1 } ^ { \top }$ to project onto PC1 and $Q = I - P$ to remove it.

## A.2 SAE training and bypass

We use TopK SAEs (Gao et al., 2025) with $m = 4 d$ features and $k = 6 4$ unless stated otherwise. Each column of the decoder matrix D is a feature direction:

$$
\begin{array} { r } { \mathbf { z } ( \mathbf { x } ) = \mathrm { T o p K } _ { k } ( \mathrm { R e L U } [ W _ { \mathrm { e n c } } ( \mathbf { x } - \pmb { \mu } - \mathbf { b } _ { \mathrm { d e c } } ) + \mathbf { b } _ { \mathrm { e n c } } ] ) , } \\ { \widehat { \mathbf { x } } = \pmb { \mu } + \mathbf { b } _ { \mathrm { d e c } } + D \mathbf { z } ( \mathbf { x } ) . \qquad } \end{array}\tag{1}
$$

(2)

TopK keeps at most $k$ positive entries. We sample Gaussian decoder directions and normalize each to unit length. The encoder is initialized as $D ^ { \top }$ , but its weights and the decoder weights are updated separately. Biases start at zero.

Table 3: Default SAE training settings.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Training activations</td><td>5,000,000</td></tr><tr><td>Dictionary width</td><td>4d</td></tr><tr><td>TopK budget</td><td>64</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Adam moments</td><td>(0.9,0.999)</td></tr><tr><td>Adam epsilon</td><td>10−8</td></tr><tr><td>Steps</td><td>20,000</td></tr><tr><td>Batch size</td><td>4,096</td></tr><tr><td>Sampling</td><td>With replacement</td></tr><tr><td>Loss</td><td>Mean squared reconstruction error</td></tr><tr><td></td><td>Decoder normalization Unit-length columns after every update</td></tr></table>

The same seed sets the initial weights and minibatch order, unless we fix the order across runs. The bypass SAE takes $Q ( \mathbf { x } - \pmb { \mu } )$ as input. Writing its coefficients as $\mathbf { z } _ { B }$ , we reconstruct

$$
\widehat { \mathbf { x } } _ { B } = \pmb { \mu } + \mathbf { b } _ { \mathrm { d e c } } + D \mathbf { z } _ { B } + P ( \mathbf { x } - \pmb { \mu } ) .\tag{3}
$$

We do not project the decoder output off PC1.

## A.3 Evaluation data

For feature importance, resampling, PC1 prediction, and automated interpretability, we select Wiki-Text articles outside the SAE training sample using seed 20261005. These articles are divided into non-overlapping sets (Table 4). The token counts are targets based on GPT-2 tokenization; other models use the same articles but may produce different counts. Paragraphs are truncated to 128 tokens. Cross-entropy is averaged over valid next-token predictions, excluding padding and positions without a target.

Table 4: WikiText evaluation sets and target token counts.
<table><tr><td>Article set</td><td>Target tokens Use</td><td></td></tr><tr><td>Calibration</td><td>65,536</td><td>Resampling and fitting controls</td></tr><tr><td>Ranking</td><td></td><td>32,768 Ranking features and component ablations</td></tr><tr><td>Validation</td><td>32,768</td><td>Validation</td></tr><tr><td>PC1 prediction test</td><td>65,536</td><td>Testing predictions of model quantities</td></tr><tr><td>Explanation</td><td></td><td>500,000 Writing feature descriptions</td></tr><tr><td>Detection</td><td></td><td>500,000 Testing feature descriptions</td></tr></table>

## A.4 Feature matching and uncertainty

To match feature j from seed s against seed t, we find its largest decoder cosine:

$$
c _ { j } ^ { s  t } = \operatorname* { m a x } _ { \ell } \frac { ( { \bf d } _ { j } ^ { s } ) ^ { \top } { \bf d } _ { \ell } ^ { t } } { \Vert { \bf d } _ { j } ^ { s } \Vert _ { 2 } \Vert { \bf d } _ { \ell } ^ { t } \Vert _ { 2 } } .\tag{4}
$$

A feature is shared if $c _ { j } ^ { s  t } \geq 0 . 7$ . We do not take the absolute value of cosine. Several source features may match the same target, and we search all target decoder directions unless stated otherwise. Dictionary-wide sharing averages over alive source features; top-N sharing uses the N highest-ranked alive features.

For the default SAEs, the original code counted feature firings over 106,496 training activations but divided by 100,000. We keep the resulting feature selection: the $1 0 ^ { - 3 }$ cutoff corresponds to a firing rate above approximately $9 . { \overset { - } { 3 } } 9 \times 1 0 ^ { - 4 }$ . We correct mean $L _ { 0 }$ values from this counter by dividing by 1.06496. Experiments using direct nonzero counts are identified separately.

Unless stated otherwise, sharing is averaged over comparisons from seed 42 to seeds 137, 0, 1, and 2. We also report the average over all 20 comparisons between the five seeds, counting both directions. For confidence intervals, we resample the 75 ranking articles with replacement 1,000 times. Each sample is used to rerank features in both SAEs.

## A.5 Compute resources

The original experiments ran on Modal H100 GPUs with 128 GiB of host memory. Llama-3.1-70B used two H200 GPUs and 256 GiB. Later experiments ran on AWS A10G, L4, L40S and RTX PRO 6000 GPUs, including two RTX PRO 6000s for the 70B evaluation.

Recorded evaluations of linear baselines and model loss with SAE reconstructions took about 2–28 minutes per model. Hardware configurations and available run records are included in the supplement.

## B Feature Reproducibility and Importance

## B.1 Zero ablation of all alive features

We remove one feature at a time from the SAE reconstruction and rank all alive features by the resulting increase in cross-entropy.

Table 5: Sharing (%). S/B denotes standard/bypass. Post-hoc ranks only standard-SAE features with $a _ { j } < . 0 1$
<table><tr><td>Model</td><td>Dictionary S/B</td><td>Top 50 S/B</td><td>All-pairs top 50 S/B</td><td>Post-hoc</td></tr><tr><td>GPT-2</td><td>46.6/44.3</td><td>5.5/73.0</td><td>5.4/72.7</td><td>77.5</td></tr><tr><td>Pythia-70M</td><td>47.0/45.5</td><td>47.5/81.0</td><td>43.2/77.9</td><td>81.5</td></tr><tr><td>Pythia-160M</td><td>49.0/46.5</td><td>13.0/83.0</td><td>13.3/86.1</td><td>87.0</td></tr><tr><td>Pythia-1.4B</td><td>62.9/58.7</td><td>6.0/92.0</td><td>6.8/88.4</td><td>90.0</td></tr><tr><td>Pythia-6.9B</td><td>62.4/61.1</td><td>4.0/95.5</td><td>4.4/96.1</td><td>96.0</td></tr><tr><td>Llama-3.1-8B</td><td>66.2/68.0</td><td>39.5/94.5</td><td>45.7/93.8</td><td>90.5</td></tr><tr><td>Qwen-2.5-7B</td><td>47.8/61.4</td><td>6.0/99.5</td><td>6.0/96.8</td><td>95.0</td></tr><tr><td>Gemma-2-2B</td><td>58.1/62.1</td><td>87.0/93.5</td><td>84.1/93.6</td><td>96.0</td></tr><tr><td>Gemma-2-9B</td><td>58.1/60.6</td><td>92.0/94.5</td><td>88.3/94.7</td><td>92.0</td></tr></table>

Table 6: 95% confidence intervals from resampling articles: top-50 sharing (%) and the bypass–standard difference (percentage points).
<table><tr><td>Model</td><td>Standard</td><td>Bypass</td><td>Difference</td></tr><tr><td>GPT-2</td><td>[5.5, 5.5]</td><td>[69.5, 78.5]</td><td>[64.0, 73.0]</td></tr><tr><td>Pythia-70M Pythia-160M</td><td>[40.5, 52.5] [9.0, 13.0]</td><td>[74.0, 83.0] [80.5, 86.5]</td><td>[25.0, 39.5] [68.0, 75.5]</td></tr><tr><td>Pythia-1.4B</td><td>[4.0, 8.0]</td><td>[88.5, 93.5]</td><td>[82.0, 88.0]</td></tr><tr><td>Pythia-6.9B</td><td>[4.0, 6.0]</td><td>[92.0, 99.0]</td><td>[87.0, 95.0]</td></tr><tr><td>Llama-3.1-8B</td><td>[36.5, 46.5]</td><td>[91.0, 96.5]</td><td>[47.0, 58.5]</td></tr><tr><td>Qwen-2.5-7B</td><td></td><td>[98.5, 99.5]</td><td></td></tr><tr><td></td><td>[6.0, 6.0]</td><td></td><td>[92.5, 93.5]</td></tr><tr><td>Gemma-2-2B</td><td>[81.0, 90.5]</td><td>[88.5, 95.5]</td><td>[1.5, 11.0]</td></tr><tr><td>Gemma-2-9B</td><td>[90.0, 96.5]</td><td>[92.0, 94.5]</td><td>[-3.0, 4.0]</td></tr></table>

For Llama-3.1-70B, we use seeds 42 and 137. Standard dictionary-wide sharing is 56%; top-50 sharing rises from 38% to 84% with bypass.

## B.2 Post-hoc selection and top-N sharing

Post-hoc selection removes features with $a _ { j } \geq . 0 1$ from the ranking pool, leaving the SAE itself unchanged.

Table 7: Mean individual zero-ablation effect among the selected top 50, in nats per prediction. Removed is the percentage of alive standard features excluded by $a _ { j } \geq . 0 1$
<table><tr><td rowspan=1 colspan=1>Model       Removed (%)StandardPost-hocBypass</td></tr><tr><td rowspan=1 colspan=1>GPT-2               2.26  0.0966  0.01300.0141</td></tr><tr><td rowspan=1 colspan=1>Pythia-70M          3.92  0.0372  0.02480.0256</td></tr><tr><td rowspan=1 colspan=1>Pythia-160M         2.36  0.0690  0.01830.0182</td></tr><tr><td rowspan=1 colspan=1>Pythia-1.4B           1.04  0.0536  0.00640.0059</td></tr><tr><td rowspan=1 colspan=1>Pythia-6.9B          0.74  0.0401  0.00500.0044</td></tr><tr><td rowspan=1 colspan=1>Llama-3.1-8B         0.74  0.0165  0.00700.0064</td></tr><tr><td rowspan=1 colspan=1>Qwen-2.5-7B         2.08  0.1101  0.01810.0100</td></tr><tr><td rowspan=1 colspan=1>Gemma-2-2B         1.57  0.0159  0.00870.0110</td></tr><tr><td rowspan=1 colspan=1>Gemma-2-9B         0.72  0.0180  0.01520.0177</td></tr></table>

Table 8: Standard-SAE sharing (%) among the top-N features.
<table><tr><td>Model Dictionary Top 10</td></tr><tr><td>Top 50 Top 150 GPT-2 46.6 10.0 5.5 45.8</td></tr><tr><td>60.1 Pythia-70M 47.0 30.0 47.5 58.3 62.3</td></tr><tr><td>Pythia-160M 49.0 22.5 13.0 48.5 63.8</td></tr><tr><td>Pythia-1.4B 62.9 20.0 6.0 52.8 69.4</td></tr><tr><td>Pythia-6.9B 62.4 20.0 4.0 55.5 71.6</td></tr><tr><td>Llama-3.1-8B 66.2 67.5 39.5 63.2 73.1</td></tr></table>

<table><tr><td>Model</td><td>Dictionary</td><td>Top 10</td><td>Top 50</td><td>Top 150</td><td>Top 300</td></tr><tr><td>Qwen-2.5-7B</td><td>47.8</td><td>10.0</td><td>6.0</td><td>39.0</td><td>62.0</td></tr><tr><td>Gemma-2-2B</td><td>58.1</td><td>100.0</td><td>87.0</td><td>83.3</td><td>81.0</td></tr><tr><td>Gemma-2-9B</td><td>58.1</td><td>97.5</td><td>92.0</td><td>90.2</td><td>84.4</td></tr></table>

For component ablations and subspace comparisons, we group features whose PC1 alignment exceeds five times the median among alive features: $a _ { j } > 5$ media ${ \mathrm { 1 } } \ell \in A \ a \ell ;$ , where A is the alive set. For uniformly random unit directions, $a _ { j } \sim \mathrm { B e t a } ( 1 / 2 , ( d - 1 ) / 2 )$ ; this cutoff selects approximately 13% of directions at these dimensions.

## B.3 Other measures of feature importance

Mean ablation replaces a feature’s activation with its average on the calibration data. Gradient– activation importance is the sum of $| z _ { j } \partial \ell / \partial z _ { j } |$ over valid tokens. Integrated gradients integrate along the straight path from all-zero $\mathrm { \check { S } A E }$ coefficients to their observed values, then sum absolute attributions over tokens. For bypass, the PC1 coefficient stays fixed along the path.

Table 9: Top-50 sharing (%), standard/bypass.
<table><tr><td>Model</td><td>Zero</td><td>Mean</td><td>Gradient× activation</td><td>Integrated gradients</td></tr><tr><td>GPT-2</td><td>5.5/73.0</td><td>4.0/71.5</td><td>12.5/83.5</td><td>7.0/82.0</td></tr><tr><td>Pythia-70M</td><td>47.5/81.0</td><td>46.0/84.5</td><td>53.5/64.5</td><td>44.5/71.0</td></tr><tr><td>Pythia-160M</td><td>13.0/83.0</td><td>10.0/84.5</td><td>17.0/74.0</td><td>15.0/78.0</td></tr><tr><td>Pythia-1.4B</td><td>6.0/92.0</td><td>2.0/92.0</td><td>6.0/89.5</td><td>10.0/90.5</td></tr><tr><td>Pythia-6.9B</td><td>4.0/95.5</td><td>2.0/93.5</td><td>7.5/93.5</td><td>9.0/94.5</td></tr><tr><td>Llama-3.1-8B</td><td>39.5/94.5</td><td>39.5/93.0</td><td>17.0/93.0</td><td>11.0/96.5</td></tr><tr><td>Qwen-2.5-7B</td><td>6.0/99.5</td><td>6.0/99.5</td><td>0.0/94.0</td><td>0.0*/96.0</td></tr><tr><td>Gemma-2-2B</td><td>87.0/93.5</td><td>86.5/90.5</td><td>45.5/63.0</td><td>34.0/86.5</td></tr><tr><td>Gemma-2-9B</td><td>92.0/94.5</td><td>95.0/94.5</td><td>16.0/46.5</td><td>31.0*/88.0</td></tr></table>

We approximate the integral with midpoint gradients, starting from 16 intervals and doubling to at most 128. A run is accepted once the attributions sum to within 1% of the loss change, and a further doubling keeps at least 48 of the top 50 features. Of 92 total runs, 13 do not meet both conditions; the two affected table entries are marked by asterisks.

We check the gradients themselves on GPT-2. Finite-difference estimates in float64 match autograd to a relative error of $1 0 ^ { - 6 }$ . Furthermore, recomputing the gradient–activation rankings in float32 leaves the top 50 unchanged. An inconsistency should be noted: while the convergence check was applied to the signed sum | P IG|, the reported score is $\sum \left| \mathrm { I G } \right|$ . Rerunning the check on GPT-2 with the reported score keeps 50 top features for the standard SAE and 49 for bypass.

## B.4 Resample ablation

Table 10: Resample ablation. Each feature uses 16 draws, with replacement values taken from calibration articles separate from the ranking articles.

<table><tr><td>Method</td><td>Replacement rule</td></tr><tr><td>Marginal</td><td>Replace every activation with a value sampled from that feature&#x27;s calibration activations, including zeros.</td></tr><tr><td></td><td>Support-preserving Leave zeros unchanged; replace positive activations with sampled positive values for the same feature.</td></tr><tr><td>Position-stratified1</td><td>Leave zeros unchanged; sample positive values from the same range of token positions.</td></tr></table>

Token positions are grouped as 0, 1, 2–3, 4–7, 8–15, 16–31, 32–63, and 64 onward. If a group has fewer than 32 positive calibration values, we sample from all positive calibration values for that feature.

We evaluate all alive features in GPT-2, Pythia-70M, and Pythia-160M. For each other model and SAE type, we evaluate features appearing in any of the top-256 lists for zero ablation, gradient–activation, or integrated gradients, together with the feature cluster defined above and 256 random alive features.

Table 11: Top-50 sharing (%), standard/bypass. Counts give evaluated/alive features. Baseline is sharing within the evaluated set; only the first three models evaluate all alive features.
<table><tr><td>Model</td><td> $N _ { S }$ </td><td> $N _ { B }$ </td><td>Marginal</td><td>Support</td><td>Position</td><td>Baseline S/B</td></tr><tr><td>GPT-2</td><td>2962/2962</td><td>3070/3070</td><td>4.0/77.5</td><td>83.0/85.0</td><td>87.0/89.0</td><td>46.6/44.3</td></tr><tr><td>Pythia-70M</td><td>2016/2016</td><td>2035/2035</td><td>32.0/80.0</td><td>51.5/83.0</td><td>71.0/78.5</td><td>47.0/45.5</td></tr><tr><td>Pythia-160M</td><td>3046/3046</td><td>3066/3066</td><td>6.0/76.5</td><td>50.5/90.0</td><td>75.0/88.5</td><td>49.0/46.5</td></tr><tr><td>Pythia-1.4B</td><td>1207/6552</td><td>880/7223</td><td>0.0/93.5</td><td>27.5/95.5</td><td>42.5/97.5</td><td>64.9/73.3</td></tr><tr><td>Pythia-6.9B</td><td>697/9107</td><td>1138/10209</td><td>2.0/95.5</td><td>51.5/94.0</td><td>74.0/94.0</td><td>69.4/73.0</td></tr><tr><td>Llama-3.1-8B</td><td>1418/9795</td><td>1001/11996</td><td>8.0/96.5</td><td>96.5/97.5</td><td>97.5/99.5</td><td>67.0/75.2</td></tr><tr><td>Qwen-2.5-7B</td><td>644/5098</td><td>951/7638</td><td>0.0/99.5</td><td>52.0/98.0</td><td>52.0/99.0</td><td>55.3/74.4</td></tr><tr><td>Gemma-2-2B</td><td>1077/7501</td><td>1749/8047</td><td>73.0/93.5</td><td>90.0/95.5</td><td>92.5/98.0</td><td>60.4/64.2</td></tr><tr><td>Gemma-2-9B</td><td>872/9603</td><td>1082/10264</td><td>92.5/94.5</td><td>92.5/99.0</td><td>92.5/100.0</td><td>65.7/64.0</td></tr></table>

## C Rank-One Reconstruction and Component Ablations

## C.1 Non-unique sparse reconstruction

Take $\mathbf { x } = a \mathbf { v }$ , where $a \in \mathbb { R } , \| \mathbf { v } \| _ { 2 } = 1$ , and $0 < \mathbb { E } [ a ^ { 2 } ] < \infty$ . Decoder columns have unit norm.   
Codes are nonnegative, with at most k nonzeros. Biases are zero; encoder weights are unrestricted.

Proposition C.1 (Rank-one reconstruction). Assume $d \geq 3 , m \geq 4 ,$ , and $2 \leq k \leq$ m. Infinitely many TopK SAEs have zero reconstruction error on av. For any $\tau \in ( 0 , 1 )$ , two ofthem have active decoder sets $\mathcal { D } , \mathcal { D } ^ { \prime }$ satisfying

$$
\mathbf { v } \in \mathrm { s p a n } ( \mathcal { D } ) \cap \mathrm { s p a n } ( \mathcal { D } ^ { \prime } ) , \qquad \operatorname* { m a x } _ { \mathbf { d } \in \mathcal { D } , \mathbf { d } ^ { \prime } \in \mathcal { D } ^ { \prime } } | \mathbf { d } ^ { \top } \mathbf { d } ^ { \prime } | < \tau .
$$

Proof. Extend v to an orthonormal triple $( { \bf v } , { \bf u } , { \bf u } ^ { \prime } )$ . Let $\theta \in ( 0 , \pi / 2 )$ and $a _ { \pm } = \operatorname* { m a x } ( \pm a , 0 )$ . The decoder and encoder are

$$
\begin{array} { l } { { { \bf d } _ { \pm } = \cos \theta { \bf v } \pm \sin \theta { \bf u } , } } \\ { { { \cal D } = ( { \bf d } _ { + } , { \bf d } _ { - } , - { \bf d } _ { + } , - { \bf d } _ { - } ) , } } \\ { { { \cal W } = \frac { ( 1 , 1 , - 1 , - 1 ) ^ { \top } { \bf v } ^ { \top } } { 2 \cos \theta } . } } \end{array}
$$

Pad to width m using zero encoder rows and unit decoder columns. The code and reconstruction are

$$
\begin{array} { r l } & { \mathbf { z } = \mathrm { T o p K } ( \mathrm { R e L U } ( W \mathbf { x } ) ) = \frac { ( a _ { + } , a _ { + } , a _ { - } , a _ { - } , 0 , \dots , 0 ) ^ { \top } } { 2 \cos \theta } , } \\ & { D \mathbf { z } = ( a _ { + } - a _ { - } ) \mathbf { v } = \mathbf { x } . } \end{array}
$$

Use u<sup>′</sup> in place of u for $D ^ { \prime }$ . Its columns satisfy

$$
| \mathbf { d } ^ { \top } \mathbf { d } ^ { \prime } | = \cos ^ { 2 } \theta , \qquad \mathbf { v } = \frac { \mathbf { d } _ { + } + \mathbf { d } _ { - } } { 2 \cos \theta } = \frac { \mathbf { d } _ { + } ^ { \prime } + \mathbf { d } _ { - } ^ { \prime } } { 2 \cos \theta } .
$$

Here $\mathbf { d } , \mathbf { d ^ { \prime } }$ range over the four constructed columns of $D , D ^ { \prime }$ . Choose $\cos ^ { 2 } \theta < \tau$ . Changing θ gives distinct exact solutions. □

Two columns suffice for a single sign of a. The aligned decoder $( \mathbf { v } , - \mathbf { v } )$ works for both signs, with code $( a _ { + } , a _ { - } )$

Proposition C.2 (Magnitude cost). Let $\begin{array} { r } { \mathbf { x } = \sum _ { j } z _ { j } \mathbf { d } _ { j } } \end{array}$ , with $z _ { j } \geq 0$ and $\| \mathbf d _ { j } \| _ { 2 } = 1$ . Then

$$
\| \mathbf { x } \| _ { 2 } \leq \sum _ { j } z _ { j } .
$$

Equality for $\mathbf { x } \neq 0$ holds $i f f \mathbf { d } _ { j } = \mathbf { x } / \lVert \mathbf { x } \rVert _ { 2 }$ whenever $z _ { j } > 0$ . Above, the total coefficient is $| a | /$ cos θ.

Proof. For $\mathbf { x } \neq 0 ,$ , put $\mathbf { e } = \mathbf { x } / \lVert \mathbf { x } \rVert _ { 2 }$ <sub>2</sub>. Then

$$
\| \mathbf { x } \| _ { 2 } = \sum _ { j } z _ { j } ( \mathbf { e } ^ { \top } \mathbf { d } _ { j } ) \leq \sum _ { j } z _ { j } ,
$$

with equality iff $\mathbf { e } ^ { \top } \mathbf { d } _ { j } = 1$ for all $z _ { j } > 0$ . The bound at $\mathbf { x } = 0$ follows from $z _ { j } \geq 0$ . In our example, add the two coefficients $| a | / ( 2 \cos \tilde { \theta } )$ □

Squared reconstruction errors after removing features:

<table><tr><td>Features removed</td><td>Total error</td><td>Error perpendicular to v</td></tr><tr><td>None</td><td>0</td><td>0</td></tr><tr><td>Either one</td><td> $ { a ^ { 2 } / ( 4 \cos ^ { 2 } \theta ) }$ </td><td>a² tan²θ/4</td></tr><tr><td>Both</td><td></td><td>0</td></tr></table>

The error in the middle row diverges as $\theta \to \pi / 2$ for any fixed $a \neq 0$

## C.2 Individual and joint component ablations

For contribution $\mathbf { q } _ { j } = z _ { j } \mathbf { d } _ { j }$ , we separately remove $\mathbf { q } _ { j } , P \mathbf { q } _ { j } , \mathrm { o r } Q \mathbf { q } _ { j }$ from the SAE-spliced activation. We use seed 42’s top 50 by zero ablation and the same ranking articles for all three interventions (Table 12).

Table 12: Sums of individual top-50 ablation effects, in nats per prediction. The final column is a ratio of sums, not a causal decomposition of loss.
<table><tr><td>Model</td><td>Whole</td><td>Parallel</td><td>Orthogonal</td><td>Orthogonal/whole (%)</td></tr><tr><td>GPT-2</td><td>4.834</td><td>0.055</td><td>4.713</td><td>97.5</td></tr><tr><td>Pythia-70M</td><td>1.862</td><td>0.028</td><td>1.961</td><td>105.3</td></tr><tr><td>Pythia-160M</td><td>3.376</td><td>-0.027</td><td>3.389</td><td>100.4</td></tr><tr><td>Pythia-1.4B</td><td>2.676</td><td>0.564</td><td>2.347</td><td>87.7</td></tr><tr><td>Pythia-6.9B</td><td>1.992</td><td>0.686</td><td>1.545</td><td>77.6</td></tr><tr><td>Llama-3.1-8B</td><td>0.825</td><td>0.138</td><td>0.589</td><td>71.3</td></tr><tr><td>Qwen-2.5-7B</td><td>5.504</td><td>0.276</td><td>5.072</td><td>92.2</td></tr><tr><td>Gemma-2-2B</td><td>0.793</td><td>-0.021</td><td>0.837</td><td>105.6</td></tr><tr><td>Gemma-2-9B</td><td>0.898</td><td>-0.055</td><td>0.959</td><td>106.7</td></tr></table>

Removing only the orthogonal contributions retains most or all of the whole-feature effect, although PC1 reconstruction remains unchanged. Cross-entropy is nonlinear: parallel and orthogonal effects need not add, and either can be negative.

Table 13: Joint ablation of the geometric cluster, source seed 42. Effects are nats per prediction. <sup>∗</sup>Qwen is evaluated in bfloat16.
<table><tr><td>Model</td><td>Features</td><td>Whole</td><td>Parallel</td><td>Orthogonal</td></tr><tr><td>GPT-2</td><td>455</td><td>2.385</td><td>0.825</td><td>1.318</td></tr><tr><td>Pythia-70M</td><td>318</td><td>1.071</td><td>0.128</td><td>1.240</td></tr><tr><td>Pythia-160M</td><td>486</td><td>1.202</td><td>0.224</td><td>1.044</td></tr><tr><td>Pythia-1.4B</td><td>713</td><td>1.041</td><td>0.572</td><td>0.437</td></tr><tr><td>Pythia-6.9B</td><td>114</td><td>0.588</td><td>0.421</td><td>0.120</td></tr><tr><td>Llama-3.1-8B</td><td>935</td><td>2.626</td><td>2.148</td><td>0.558</td></tr><tr><td>Qwen  $2 . 5 \ – 7 \mathbf { B } ^ { * }$ </td><td>141</td><td>7.719</td><td>6.114</td><td>0.662</td></tr><tr><td>Gemma-2-2B</td><td>585</td><td>0.977</td><td>0.185</td><td>0.780</td></tr><tr><td>Gemma-2-9B</td><td>352</td><td>1.193</td><td>0.348</td><td>0.831</td></tr></table>

Qwen’s joint top-50 ablation overflowed in float16, so we repeated it in bfloat16. Over 33,244 predictions, removing the whole contributions increases loss by 5.652 nats per prediction; removing only the parallel or orthogonal parts gives 4.433 and 1.982.

## C.3 Subspace agreement and cancellation

Table 14 compares the decoder spans of each model’s seed-42 cluster and the other four seeds. We take an orthonormal basis for each span and compute the cosines of their principal angles by SVD. We average the squared cosines over angles, then over the four seed comparisons. Independent, uniformly random subspaces of dimensions p and $q$ in $\mathbb { R } ^ { d }$ have expected score max $( p , q ) / d .$

Table 14: Geometric-cluster subspace agreement. The null uses the ranks of both subspaces in each comparison.
<table><tr><td>Model</td><td>Source rank  $\operatorname { M e a n } \cos ^ { 2 } \theta$ </td><td>Isotropic null</td></tr><tr><td>GPT-2</td><td>455 0.683</td><td>0.592</td></tr><tr><td>Pythia-70M</td><td>318</td><td>0.761 0.686</td></tr><tr><td>Pythia-160M</td><td>486</td><td>0.703 0.635</td></tr><tr><td>Pythia-1.4B</td><td>713</td><td>0.591 0.350</td></tr><tr><td>Pythia-6.9B</td><td>114</td><td>0.274 0.029</td></tr><tr><td>Llama-3.1-8B</td><td>935</td><td>0.601 0.311</td></tr><tr><td>Qwen-2.5-7B</td><td>141</td><td>0.358 0.041</td></tr><tr><td>Gemma-2-2B</td><td>585</td><td>0.548 0.272</td></tr><tr><td>Gemma-2-9B</td><td>352 0.457</td><td>0.117</td></tr></table>

Cancellation is measured directly in activation space by

$$
C _ { J } = \frac { \sum _ { t } \sum _ { j \in J } \Vert Q z _ { t j } \mathbf { d } _ { j } \Vert _ { 2 } ^ { 2 } } { \sum _ { t } \Vert \sum _ { j \in J } Q z _ { t j } \mathbf { d } _ { j } \Vert _ { 2 } ^ { 2 } } .
$$

For standard SAEs, $C _ { J }$ ranges from 1.93 in Gemma-2-9B to 456.6 in Qwen; random feature sets of the same size give $0 . 8 9 \mathrm { - } 0 . { \bar { 9 9 } }$

We tried 40 random restarts across eight models, training only the cluster for 5,000 steps. None came within 1% of the held-out reconstruction error of the original cluster given the same number of training steps.

## D SAE Objectives and Initialization

## D.1 Magnitude penalties

The GPT-2 sweep follows Section 5.1, with minibatch order fixed across the five seeds. We use separate WikiText validation articles for calibration and selection, and the test split for evaluation. We include only features that fire on calibration data. We average sharing over all 20 seed comparisons.

Table 15: GPT-2 magnitude penalties, five seeds. Sharing is in percent; content FVU is mean ± seed SD.
<table><tr><td>SAE</td><td> $\beta$ </td><td> $S _ { 5 0 }$  sel.</td><td> $S _ { 5 0 }$  test</td><td>Dict.</td><td> $L _ { 0 }$ </td><td>Total FVU</td><td>Content FVU</td></tr><tr><td>Standard</td><td>0.00</td><td>6.4</td><td>6.4</td><td>45.8</td><td>64.00</td><td>0.004838</td><td> $0 . 1 3 9 4 1 \pm 0 . 0 0 0 3 4$ </td></tr><tr><td>Standard</td><td>0.03</td><td>17.9</td><td>14.8</td><td>47.9</td><td>63.97</td><td>0.004824</td><td> $0 . 1 3 8 9 4 \pm 0 . 0 0 0 2 8$ </td></tr><tr><td>Standard</td><td>0.10</td><td>49.9</td><td>52.1</td><td>51.2</td><td>63.80</td><td>0.004820</td><td> $0 . 1 3 8 6 5 \pm 0 . 0 0 0 1 4$ </td></tr><tr><td>Standard</td><td>0.30</td><td>95.2</td><td>94.9</td><td>57.3</td><td>63.47</td><td>0.004830</td><td> $0 . 1 3 8 6 9 \pm 0 . 0 0 0 2 2$ </td></tr><tr><td>Bypass</td><td>0.00</td><td>69.1</td><td>65.8</td><td>44.3</td><td>63.84</td><td>0.004739</td><td> $0 . 1 3 6 6 6 \pm 0 . 0 0 0 2 1$ </td></tr><tr><td>Bypass</td><td>0.03</td><td>82.9</td><td>82.2</td><td>47.1</td><td>63.77</td><td>0.004734</td><td> $0 . 1 3 6 4 4 \pm 0 . 0 0 0 1 1$ </td></tr><tr><td>Bypass</td><td>0.10</td><td>91.4</td><td>91.9</td><td>51.1</td><td>63.62</td><td>0.004730</td><td> $0 . 1 3 6 3 3 \pm 0 . 0 0 0 1 3$ </td></tr><tr><td>Bypass</td><td>0.30</td><td>94.3</td><td>94.8</td><td>57.8</td><td>63.40</td><td>0.004740</td><td> $0 . 1 3 6 6 2 \pm 0 . 0 0 0 1 4$ </td></tr></table>

Table 16: PC1 alignment in the standard GPT-2 penalty sweep. Intermediate: $. 0 1 \leq a _ { j } <$ .3; high: $a _ { j } \geq$ .3; low: $a _ { j } < . 0 1$ Sharing and top- $. { \overset { \sim } { 5 0 } }$ fractions are percentages.
<table><tr><td> $\beta$ </td><td></td><td></td><td>Int. count Int. shared Int. in top 50</td><td></td><td>High count High shared Low shared</td><td></td></tr><tr><td>0.00</td><td>65.2</td><td>2.7</td><td>93.6</td><td>1.8</td><td>95.0</td><td>46.7</td></tr><tr><td>0.03</td><td>96.6</td><td>25.9</td><td>93.2</td><td>2.4</td><td>90.0</td><td>48.6</td></tr><tr><td> $\beta$ </td><td>Int. count</td><td>Int. shared</td><td>Int. in top 50</td><td>High count</td><td>High shared</td><td>Low shared</td></tr><tr><td>0.10</td><td>461.4</td><td>63.3</td><td>80.0</td><td>5.0</td><td>54.4</td><td>49.0</td></tr><tr><td>0.30</td><td>1306.6</td><td>71.7</td><td>72.8</td><td>9.8</td><td>70.9</td><td>46.3</td></tr></table>

The Pythia-1.4B sweep compares seeds 42 and 137 over a wider range of penalties (Table 17). Reconstruction and exact nonzero counts are reevaluated on the first 100,000 training-store rows.

Table 17: Pythia-1.4B magnitude penalties. Sharing is in percent.
<table><tr><td> $\beta$ </td><td> $S _ { 5 0 }$ </td><td>Dict.</td><td> $L _ { 0 } \colon 4 2$ </td><td> $\boldsymbol { L } _ { 0 } \colon$  137</td><td>Total FVU</td><td>Content FVU</td></tr><tr><td> $_ 0$ </td><td>10.0</td><td>62.5</td><td>63.96</td><td>63.99</td><td>0.009167</td><td>0.22767</td></tr><tr><td>1e-06</td><td>10.0</td><td>62.5</td><td>63.97</td><td>64.00</td><td>0.009163</td><td>0.22756</td></tr><tr><td>1e-05</td><td>8.0</td><td>62.6</td><td>63.98</td><td>63.99</td><td>0.009166</td><td>0.22764</td></tr><tr><td>0.0001</td><td>8.0</td><td>63.2</td><td>63.98</td><td>63.98</td><td>0.009163</td><td>0.22756</td></tr><tr><td>0.001</td><td>10.0</td><td>62.1</td><td>63.98</td><td>63.98</td><td>0.009162</td><td>0.22753</td></tr><tr><td>0.003</td><td>12.0</td><td>63.7</td><td>63.98</td><td>63.98</td><td>0.009160</td><td>0.22748</td></tr><tr><td>0.01</td><td>16.0</td><td>63.5</td><td>63.95</td><td>63.98</td><td>0.009150</td><td>0.22721</td></tr><tr><td>0.03</td><td>46.0</td><td>63.8</td><td>63.81</td><td>63.67</td><td>0.009145</td><td>0.22700</td></tr><tr><td>0.1</td><td>94.0</td><td>67.5</td><td>62.91</td><td>62.86</td><td>0.009137</td><td>0.22656</td></tr><tr><td>0.3</td><td>94.0</td><td>71.8</td><td>62.64</td><td>62.64</td><td>0.009191</td><td>0.22764</td></tr><tr><td>1</td><td>98.0</td><td>79.4</td><td>62.29</td><td>62.29</td><td>0.009448</td><td>0.23382</td></tr><tr><td>3</td><td>100.0</td><td>87.6</td><td>50.21</td><td>49.25</td><td>0.011084</td><td>0.27288</td></tr><tr><td>10</td><td>98.0</td><td>92.8</td><td>8.43</td><td>8.41</td><td>0.022775</td><td>0.53503</td></tr></table>

To measure agreement outside PC1, we match each source feature to its best full-decoder match once, then project PC1 out of both directions and renormalize. Pairs in which either remainder is below 0.01 of the original norm are discarded. Results are reported in Table 18.

Table 18: Top-50 matches before and after projecting out PC1. Cosines are means; sharing uses cosine $\geq 0 . 7$ and is in percent.
<table><tr><td>Model</td><td></td><td>β Full cosine</td><td>Perp. cosine</td><td></td><td>Full shared Perp. shared</td><td>Excluded</td></tr><tr><td>GPT-2</td><td>0</td><td>0.178</td><td>0.165</td><td>6.0</td><td>4.0</td><td>0</td></tr><tr><td>GPT-2</td><td>0.03</td><td>0.316</td><td>0.235</td><td>18.0</td><td>16.0</td><td>0</td></tr><tr><td>GPT-2</td><td>0.1</td><td>0.662</td><td>0.456</td><td>50.0</td><td>50.0</td><td>0</td></tr><tr><td>GPT-2</td><td>0.3</td><td>0.901</td><td>0.831</td><td>96.0</td><td>88.0</td><td>0</td></tr><tr><td>Pythia-1.4B</td><td>0</td><td>0.268</td><td>0.114</td><td>10.0</td><td>10.0</td><td>0</td></tr><tr><td>Pythia-1.4B</td><td>0.001</td><td>0.281</td><td>0.115</td><td>10.0</td><td>10.0</td><td>0</td></tr><tr><td>Pythia-1.4B</td><td>0.01</td><td>0.295</td><td>0.219</td><td>16.0</td><td>16.0</td><td>0</td></tr><tr><td>Pythia-1.4B</td><td>0.03</td><td>0.613</td><td>0.446</td><td>46.0</td><td>42.0</td><td>0</td></tr><tr><td>Pythia-1.4B</td><td>0.1</td><td>0.928</td><td>0.910</td><td>94.0</td><td>92.0</td><td>0</td></tr><tr><td>Pythia-1.4B</td><td>0.3</td><td>0.916</td><td>0.911</td><td>94.0</td><td>94.0</td><td>0</td></tr><tr><td>Pythia-1.4B</td><td>1</td><td>0.965</td><td>0.964</td><td>98.0</td><td>98.0</td><td>0</td></tr></table>

## D.2 Comparison across objectives

The objective comparison uses the same activation stores, width, and training budget as the default TopK runs. It ranks the 1,500 alive features with largest mean squared activation on 64 validation sequences and compares seed 42 with the other seeds.

BatchTopK (Bussmann et al., 2024) keeps the largest 64B nonnegative entries in each batch of B tokens and, at inference, uses a threshold tracked during training (decay 0.999). ReL $\mathbf { \nabla } _ { \mathbf { \boldsymbol { U } } + \mathbf { \boldsymbol { \ell } } _ { 1 } }$ penalizes the coefficients. Gated (Rajamanoharan et al., 2024a) penalizes positive gate preactivations and adds a gate reconstruction loss with a detached decoder. JumpReLU (Rajamanoharan et al., 2024b) initializes thresholds from the data and recalibrates them periodically; we call this variant scheduledthreshold JumpReLU. ReLU and Gated penalties are tuned toward $\dot { L } _ { 0 } = 6 4$ . Tables 19 and 20 give feature sharing, sparsity, and penalty settings. S and B denote standard and bypass SAEs.

Table 19: Feature sharing (%) and mean number of active features $( L _ { 0 } )$ across SAE objectives.
<table><tr><td>Model</td><td>Objective</td><td>Seeds</td><td> $S _ { 5 0 }$  S</td><td> $S _ { 5 0 }$  B</td><td>Dict. S</td><td>Dict. B</td><td> $L _ { 0 } S$ </td><td> $L _ { \mathrm { 0 } } \mathrm { B }$ </td></tr><tr><td>GPT-2</td><td>TopK</td><td>5/5</td><td>5.5</td><td>79.5</td><td>46.6</td><td>44.3</td><td>64.0</td><td>63.8</td></tr><tr><td>GPT-2</td><td>BatchTopK</td><td>5/5</td><td>17.5</td><td>64.0</td><td>42.6</td><td>45.1</td><td>63.0</td><td>63.8</td></tr><tr><td>GPT-2</td><td>JumpReLU (sched.)</td><td>5/5</td><td>11.5</td><td>66.5</td><td>42.8</td><td>46.4</td><td>65.1</td><td>66.0</td></tr><tr><td>GPT-2</td><td>ReLU+l1</td><td>5/5</td><td>100.0</td><td>100.0</td><td>78.0</td><td>82.4</td><td>49.2</td><td>57.5</td></tr><tr><td>GPT-2</td><td>Gated</td><td>3/3</td><td>11.0</td><td>92.0</td><td>41.1</td><td>51.5</td><td>56.7</td><td>59.9</td></tr><tr><td>Pythia-160M</td><td>TopK</td><td>5/5</td><td>9.0</td><td>83.5</td><td>49.0</td><td>46.5</td><td>64.0</td><td>64.0</td></tr><tr><td>Pythia-160M</td><td>BatchTopK</td><td>5/5</td><td>9.5</td><td>77.0</td><td>44.5</td><td>46.0</td><td>63.2</td><td>63.8</td></tr><tr><td>Pythia-160M</td><td>JumpReLU (sched.)</td><td>5/5</td><td>14.5</td><td>88.5</td><td>44.7</td><td>47.2</td><td>65.6</td><td>66.3</td></tr><tr><td>Pythia-160M</td><td>ReLU+l1</td><td>5/5</td><td>96.5</td><td>94.5</td><td>78.1</td><td>84.4</td><td>52.9</td><td>50.5</td></tr><tr><td>Pythia-160M</td><td>Gated</td><td>5/5</td><td>87.0</td><td>91.0</td><td>46.4</td><td>50.6</td><td>56.3</td><td>56.4</td></tr><tr><td>Pythia-1.4B</td><td>TopK</td><td>5/5</td><td>10.0</td><td>85.5</td><td>62.9</td><td>58.7</td><td>64.0</td><td>64.0</td></tr><tr><td>Pythia-1.4B</td><td>BatchTopK</td><td>5/5</td><td>14.0</td><td>87.0</td><td>62.1</td><td>54.1</td><td>63.4</td><td>64.1</td></tr><tr><td>Pythia-1.4B</td><td>JumpReLU (sched.)</td><td>5/5</td><td>12.0</td><td>89.0</td><td>62.3</td><td>53.6</td><td>65.8</td><td>66.8</td></tr><tr><td>Pythia-1.4B</td><td>ReLU+l1</td><td>5/5</td><td>98.5</td><td>97.0</td><td>83.9</td><td>87.4</td><td>52.3</td><td>46.8</td></tr><tr><td>Pythia-1.4B</td><td>Gated</td><td>5/5</td><td>86.0</td><td>94.5</td><td>58.5</td><td>61.7</td><td>62.1</td><td>55.2</td></tr><tr><td>Pythia-6.9B</td><td>TopK</td><td>3/3</td><td>10.0</td><td>90.0</td><td>62.2</td><td>61.0</td><td>64.0</td><td>63.9</td></tr><tr><td>Pythia-6.9B</td><td>BatchTopK</td><td>3/3</td><td>18.0</td><td>88.0</td><td>67.0</td><td>60.5</td><td>63.6</td><td>64.5</td></tr><tr><td>Pythia-6.9B</td><td>JumpReLU (sched.)</td><td>3/3</td><td>26.0</td><td>91.0</td><td>64.5</td><td>60.6</td><td>66.2</td><td>68.3</td></tr><tr><td>Pythia-6.9B</td><td>ReLU+l1</td><td>3/3</td><td>97.0</td><td>96.0</td><td>84.5</td><td>84.3</td><td>54.6</td><td>53.4</td></tr><tr><td>Pythia-6.9B</td><td>Gated</td><td>3/3</td><td>15.0</td><td>92.0</td><td>60.9</td><td>69.1</td><td>70.1</td><td>46.1</td></tr><tr><td>Llama-3.1-8B</td><td>TopK</td><td>3/3</td><td>56.0</td><td>86.0</td><td>65.9</td><td>67.8</td><td>63.7</td><td>63.4</td></tr><tr><td>Llama-3.1-8B</td><td>BatchTopK</td><td>3/3</td><td>62.0</td><td>91.0</td><td>70.5</td><td>69.3</td><td>64.6</td><td>65.2</td></tr><tr><td>Llama-3.1-8B</td><td>JumpReLU (sched.)</td><td>3/3</td><td>61.0</td><td>91.0</td><td>67.8</td><td>69.6</td><td>66.2</td><td>67.1</td></tr><tr><td>Llama-3.1-8B</td><td>ReLU+l1</td><td>3/3</td><td>97.0</td><td>99.0</td><td>80.0</td><td>79.8</td><td>62.8</td><td>65.0</td></tr><tr><td>Llama-3.1-8B</td><td>Gated</td><td>3/3</td><td>99.0</td><td>99.0</td><td>68.7</td><td>64.1</td><td>57.9</td><td>60.2</td></tr><tr><td>Qwen-2.5-7B</td><td>TopK</td><td>3/3</td><td>6.0</td><td>98.0</td><td>46.4</td><td>61.3</td><td>64.0</td><td>64.0</td></tr><tr><td>Qwen-2.5-7B</td><td>BatchTopK</td><td>3/3</td><td>33.0</td><td>95.0</td><td>56.3</td><td>63.7</td><td>63.3</td><td>63.5</td></tr><tr><td>Qwen-2.5-7B</td><td>JumpReLU (sched.)</td><td>3/3</td><td>32.0</td><td>100.0</td><td>56.3</td><td>62.9</td><td>64.7</td><td>66.0</td></tr><tr><td>Qwen-2.5-7B</td><td>ReLU+l1</td><td>3/3</td><td>97.0</td><td>99.0</td><td>69.2</td><td>80.0</td><td>73.6</td><td>51.9</td></tr><tr><td>Qwen-2.5-7B</td><td>Gated</td><td>3/3</td><td>47.0</td><td>97.0</td><td>27.8</td><td>64.4</td><td>53.9</td><td>54.5</td></tr><tr><td>Gemma-2-2B</td><td>TopK</td><td>5/5</td><td>80.5</td><td>88.0</td><td>58.0</td><td>62.1</td><td>63.9</td><td>63.8</td></tr><tr><td>Gemma-2-2B</td><td>BatchTopK</td><td>5/5</td><td>72.5</td><td>87.0</td><td>60.3</td><td>62.0</td><td>63.9</td><td>64.4</td></tr><tr><td>Gemma-2-2B</td><td>JumpReLU (sched.)</td><td>3/3</td><td>76.0</td><td>89.0</td><td>61.2</td><td>62.1</td><td>65.6</td><td>67.6</td></tr><tr><td>Gemma-2-2B</td><td>ReLU+l1</td><td>3/3</td><td>97.0</td><td>99.0</td><td>71.3</td><td>71.7</td><td>74.0</td><td>72.8</td></tr><tr><td>Gemma-2-2B</td><td>Gated</td><td>3/3</td><td>79.0</td><td>96.0</td><td>57.2</td><td>61.3</td><td>47.9</td><td>51.9</td></tr><tr><td>Gemma-2-9B</td><td>TopK</td><td>3/3</td><td>90.0</td><td>87.0</td><td>58.3</td><td>60.5</td><td>64.0</td><td>63.9</td></tr><tr><td>Gemma-2-9B</td><td>BatchTopK</td><td>3/3</td><td>90.0</td><td>87.0</td><td>61.5</td><td>62.6</td><td>64.3</td><td>64.6</td></tr><tr><td>Gemma-2-9B</td><td>JumpReLU (sched.)</td><td>3/3</td><td>75.0</td><td>87.0</td><td>61.0</td><td>63.1</td><td>66.4</td><td>68.5</td></tr><tr><td>Gemma-2-9B</td><td>ReLU+l1</td><td>3/3</td><td>95.0</td><td>94.0</td><td>72.4</td><td>71.6</td><td>69.2</td><td>68.4</td></tr><tr><td>Gemma-2-9B</td><td>Gated</td><td>3/3</td><td>63.0</td><td>80.0</td><td>56.3</td><td>55.2</td><td>49.8</td><td>51.2</td></tr></table>

Table 20: ReLU and Gated calibration settings. c is the sparsity penalty coefficient. S/B: standard/bypass; $L _ { 0 }$ ranges are across seeds.
<table><tr><td>Model</td><td>Objective</td><td> $c \operatorname { S }$ </td><td>cB</td><td> $L _ { 0 }$  range S</td><td> $L _ { 0 }$  range B</td></tr><tr><td>GPT-2</td><td> $\mathbf { \mathrm { R e L U } } { + } \ell _ { 1 }$ </td><td>5.062</td><td>4.171</td><td>48.8–49.4</td><td>57.0–57.8</td></tr><tr><td>GPT-2</td><td>Gated</td><td>5.624</td><td>5.273</td><td>54.5–61.0</td><td>59.7–60.0</td></tr><tr><td>Pythia-160M</td><td> $\mathbf { \mathrm { R e L U } } { + } \ell _ { 1 }$ </td><td>1.155</td><td>1.138</td><td>52.4–53.7</td><td>50.2–51.2</td></tr><tr><td>Pythia-160M</td><td>Gated</td><td>1.221</td><td>1.241</td><td>56.0–56.5</td><td>56.0–56.8</td></tr><tr><td>Pythia-1.4B</td><td> $\mathbf { \mathrm { R e L U } } { + } \ell _ { 1 }$ </td><td>3.186</td><td>2.939</td><td>52.0–52.7</td><td>46.7–46.9</td></tr><tr><td>Pythia-1.4B</td><td> $\mathrm { G a t e d }$ </td><td>3.326</td><td>3.681</td><td>60.8–65.8</td><td>54.5–55.9</td></tr><tr><td>Pythia-6.9B</td><td> $\mathbf { \mathrm { R e L U } } { + } \ell _ { 1 }$ </td><td>6.427</td><td>5.639</td><td>54.2–54.9</td><td>53.2–53.5</td></tr><tr><td>Pythia-6.9B</td><td>Gated</td><td>7</td><td>9.105</td><td>69.2–71.3</td><td>45.846.3</td></tr><tr><td>Llama-3.1-8B</td><td> $\mathbf { \mathrm { R e L U } } { + } \ell _ { 1 }$ </td><td>0.7296</td><td>0.7313</td><td>62.6–63.0</td><td>64.9–65.1</td></tr><tr><td>Llama-3.1-8B</td><td>Gated</td><td>1</td><td>1</td><td>57.9–58.0</td><td>60.0–60.4</td></tr><tr><td>Qwen-2.5-7B</td><td> $\mathbf { \mathrm { R e L U } } { + } \ell _ { 1 }$ </td><td>4.417</td><td>4.426</td><td>73.1-74.1</td><td>49.8–55.0</td></tr><tr><td>Qwen-2.5-7B</td><td>Gated</td><td>5.644</td><td>5.709</td><td>47.7-57.6</td><td>54.4–54.7</td></tr><tr><td>Gemma-2-2B</td><td> $\mathbf { \mathrm { R e L U } } { + } \ell _ { 1 }$ </td><td>9.802</td><td>9.871</td><td>73.7-74.1</td><td>72.4–73.1</td></tr><tr><td>Gemma-2-2B</td><td>Gated</td><td>16.48</td><td>15</td><td>47.3-48.6</td><td>51.7–52.1</td></tr><tr><td>Gemma-2-9B</td><td> $\mathbf { \mathrm { R e L U } } { + } \ell _ { 1 }$ </td><td>11.93</td><td>12.06</td><td>69.2–69.4</td><td>68.4–68.5</td></tr><tr><td>Gemma-2-9B</td><td>Gated</td><td>22.22</td><td>20.69</td><td>48.0–52.9</td><td>51.0–51.4</td></tr></table>

Gated controls. The gate penalty gives no gradient to the bias and scale parameters used only to set feature magnitudes. We add $\beta \widetilde { \left| \mathbf { z } \right| }$ <sub>1</sub> to the loss and vary which parameters this penalty updates (Table 21).

Table 21: GPT-2 Gated SAEs with penalties on feature magnitudes. Sharing is in percent.
<table><tr><td>β</td><td>Parameters updated</td><td> $S _ { 5 0 }$ </td><td>Dict.</td><td> $L _ { 0 }$ </td><td>Content FVU</td></tr><tr><td>0</td><td>Original gate penalty</td><td>11.0</td><td>41.1</td><td>56.76</td><td>0.15699</td></tr><tr><td>0.03</td><td>All coefficient paths</td><td>57.0</td><td>43.1</td><td>56.03</td><td>0.15188</td></tr><tr><td>0.10</td><td>All coefficient paths</td><td>92.0</td><td>50.7</td><td>55.08</td><td>0.14712</td></tr><tr><td>0.30</td><td>All coefficient paths</td><td>98.0</td><td>54.7</td><td>59.88</td><td>0.14577</td></tr><tr><td>1.00</td><td>All coefficient paths</td><td>93.0</td><td>63.7</td><td>65.15</td><td>0.14573</td></tr><tr><td>0.10</td><td>Magnitude bias/scale</td><td>89.0</td><td>49.7</td><td>54.92</td><td>0.14704</td></tr><tr><td>0.10</td><td>Shared encoder only</td><td>10.0</td><td>47.8</td><td>54.97</td><td>0.14777</td></tr></table>

We raise the gate penalty coefficient from 7 to $^ { 8 , }$ and observe the sharing rate rising from 15% to 77% on Pythia-6.9B (Table 22). The Qwen grid includes a failed seed-0 run with content FVU approximately 2.4.

Table 22: Pythia-6.9B Gated calibration. Sharing is in percent; content FVU is averaged across checkpoints.
<table><tr><td>Gate coefficient</td><td> $S _ { 5 0 }$ </td><td>Dict.</td><td> $L _ { 0 } \colon 4 2$ </td><td> $\boldsymbol { L } _ { 0 } \boldsymbol { : }$  137</td><td> $L _ { 0 } \colon 0$ </td><td>Content FVU</td></tr><tr><td>7.0</td><td>15.0</td><td>60.9</td><td>71.39</td><td>69.97</td><td>69.28</td><td>0.31909</td></tr><tr><td>8.0</td><td>77.0</td><td>56.1</td><td>54.96</td><td>58.66</td><td>59.48</td><td>0.32546</td></tr></table>

Learned JumpReLU thresholds. We also train JumpReLU with learned thresholds under the same training budget (Table 23).

Table 23: JumpReLU threshold controls, mean $\pm \ : \mathrm { S D }$ across three seeds. CE is the SAE-spliced loss.
<table><tr><td>Model</td><td>Threshold/arm</td><td>Seeds</td><td> $L _ { 0 }$ </td><td>Content FVU</td><td>Splice CE</td></tr><tr><td>GPT-2</td><td>learned</td><td>3</td><td> $5 6 . 9 9 \pm 1 . 4 1$ </td><td> $0 . 1 6 3 3 7 \pm 0 . 0 0 0 8 6$ </td><td>4.12589</td></tr><tr><td>GPT-2</td><td>learned bypass</td><td>3</td><td> $6 8 . 3 5 \pm 0 . 5 6$ </td><td> $0 . 1 1 9 7 1 \pm 0 . 0 0 0 4 9$ </td><td>4.08565</td></tr><tr><td>GPT-2</td><td>scheduled</td><td>3</td><td> $6 5 . 2 4 \pm 0 . 1 3$ </td><td> $0 . 1 4 0 0 5 \pm 0 . 0 0 0 5 4$ </td><td>4.09868</td></tr><tr><td>Pythia-1.4B</td><td>learned</td><td>3</td><td> $5 9 . 7 1 \pm 1 . 4 7$ </td><td> $0 . 2 3 6 2 1 \pm 0 . 0 0 0 4 5$ </td><td>3.20498</td></tr></table>

## D.3 Initialization controls

We train GPT-2 SAEs from ten initializations, using the same minibatches in the same order. We use the settings in Table 3 and clip gradient norms at 1. We rank features on 64 sequences and evaluate them on 256 test sequences. In the pinned condition, the decoder direction aligned with PC1 stays fixed throughout training.

Table 24: GPT-2 initialization and training controls. Entries are seed medians.
<table><tr><td>Training change</td><td> $L _ { 0 }$ </td><td>Alive</td><td>Tiles</td><td>High alignment</td><td>Total FVU</td><td>Content FVU</td></tr><tr><td>Standard</td><td>64.00</td><td>2864.0</td><td>65.0</td><td>1.0</td><td>0.00676</td><td>0.14898</td></tr><tr><td>Canonical init.</td><td>63.99</td><td>2106.0</td><td>1.0</td><td>2.0</td><td>0.00748</td><td>0.16501</td></tr><tr><td>Canonical pinned</td><td>63.98</td><td>1879.0</td><td>0.0</td><td>2.0</td><td>0.00788</td><td>0.17362</td></tr><tr><td>Half width</td><td>64.00</td><td>1529.5</td><td>65.5</td><td>1.0</td><td>0.00799</td><td>0.17624</td></tr><tr><td>Double width</td><td>64.00</td><td>4851.5</td><td>64.0</td><td>1.0</td><td>0.00593</td><td>0.13083</td></tr><tr><td>k = 32</td><td>32.00</td><td>2977.0</td><td>35.0</td><td>1.0</td><td>0.00832</td><td>0.18334</td></tr><tr><td>k = 128</td><td>128.00</td><td>2639.0</td><td>70.5</td><td>0.0</td><td>0.00514</td><td>0.11343</td></tr><tr><td>Bypass</td><td>63.86</td><td>3072.0</td><td>0.0</td><td>0.0</td><td>0.00655</td><td>0.14443</td></tr><tr><td> $\beta = 0 . 3$ </td><td>0.05</td><td>53.0</td><td>0.0</td><td>4.5</td><td>0.04643</td><td>1.00362</td></tr></table>

Table 25 uses CE and KL rankings relative to the unmodified model, all evaluation positions, and all 90 ordered pairs of seeds. We calculate the intervals as follows:

<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Confidence level</td><td>95%</td></tr><tr><td>Bootstrap draws</td><td>2,000</td></tr><tr><td>Resampled in each draw</td><td>Seeds and 30 selection articles</td></tr><tr><td>Articles within each draw</td><td>Same sample for every seed</td></tr><tr><td>Kept fixed</td><td>Decoders, alive features, and matching threshold</td></tr></table>

Table 25: GPT-2 sharing (%) under CE and KL rankings, with joint 95% bootstrap intervals.
<table><tr><td>Training</td><td>N</td><td>Dict.</td><td>CE sharing</td><td>CE interval</td><td>KL sharing</td><td>KL interval</td></tr><tr><td>Standard</td><td>10</td><td>40.0</td><td>18.8</td><td>13.2–21.7</td><td>12.0</td><td>9.1-17.0</td></tr><tr><td>Standard</td><td>50</td><td>40.0</td><td>5.5</td><td>5.0–8.1</td><td>4.9</td><td>4.2-5.6</td></tr><tr><td>Standard</td><td>100</td><td>40.0</td><td>26.4</td><td>24.7-28.4</td><td>30.8</td><td>29.8–32.4</td></tr><tr><td>Bypass</td><td>10</td><td>37.3</td><td>81.0</td><td>72.1–85.6</td><td>92.0</td><td>87.0–95.5</td></tr><tr><td>Bypass</td><td>50</td><td>37.3</td><td>65.4</td><td>58.8–69.1</td><td>77.0</td><td>74.1–81.8</td></tr><tr><td>Bypass</td><td>100</td><td>37.3</td><td>60.4</td><td>52.6–61.8</td><td>78.0</td><td>75.2–79.8</td></tr><tr><td>Canonical init.</td><td>10</td><td>36.2</td><td>79.7</td><td>73.2–85.8</td><td>83.9</td><td>76.8–89.8</td></tr><tr><td>Canonical init.</td><td>50</td><td>36.2</td><td>68.0</td><td>62.9–72.8</td><td>74.4</td><td>71.3-79.4</td></tr><tr><td>Canonical init.</td><td>100</td><td>36.2</td><td>65.0</td><td>56.5-64.2</td><td>72.7</td><td>68.9-74.6</td></tr><tr><td>Train after first</td><td>50</td><td>38.1</td><td>40.4</td><td>36.3-43.9</td><td>35.0</td><td>31.6–38.8</td></tr><tr><td>β = 0.3</td><td>50</td><td>1.4</td><td>0.2</td><td>0.0-1.4</td><td>9.0</td><td>8.4–9.6</td></tr></table>

Excluding the first position from the ranking loss raises standard-SAE top-50 sharing without retraining (Table 26).

Table 26: Top-50 sharing (%) under position exclusion and one-to-one matching; each pair of rows uses the same trained dictionaries.
<table><tr><td>Training</td><td>Ranking tokens</td><td>CE</td><td>CE 1:1</td><td>KL KL 1:1</td></tr><tr><td>Standard</td><td>All</td><td>5.5</td><td>5.5</td><td>4.9 4.9</td></tr><tr><td>Standard</td><td>After first</td><td>70.0</td><td>69.6 80.0</td><td>79.3</td></tr><tr><td>Bypass</td><td>All</td><td>65.4</td><td>65.3 77.0</td><td>76.8</td></tr><tr><td>Bypass</td><td>After first</td><td>65.7</td><td>65.5 77.1</td><td>76.8</td></tr><tr><td>Canonical init. All</td><td></td><td>68.0</td><td>68.0 74.4</td><td>74.3</td></tr><tr><td>Canonical init.</td><td>After first</td><td>69.3</td><td>69.2 77.9</td><td>77.8</td></tr></table>

## D.4 Objective comparison with matched seeds

For each model in Table 27, we train standard and bypass SAEs with TopK, BatchTopK, and ReL $\mathbf { \nabla } J + \ell _ { 1 }$ There are five seeds for each setting. We use the same rule to select 1,500 features, rank them on 64 validation sequences, and measure sharing for every pair of seeds in both directions (20 comparisons).

Table 27: Sharing (%) and mean number of active features (L<sub>0</sub>) for the repeated objective comparison, standard/bypass.
<table><tr><td>Model</td><td>Objective</td><td>Top 50</td><td>Dictionary</td><td> $L _ { 0 }$ </td></tr><tr><td>GPT-2</td><td>TopK</td><td>5.8/79.1</td><td>46.5/44.0</td><td>64.00/63.82</td></tr><tr><td>GPT-2</td><td>BatchTopK</td><td>15.9/70.0</td><td>43.2/45.0</td><td>63.04/63.77</td></tr><tr><td>GPT-2</td><td>ReLU+l1</td><td>99.1/99.2</td><td>75.3/82.1</td><td>60.02/59.52</td></tr><tr><td>Pythia-160M</td><td>TopK</td><td>9.3/85.2</td><td>49.3/46.0</td><td>63.99/64.00</td></tr><tr><td>Pythia-160M</td><td>BatchTopK</td><td>14.6/82.9</td><td>45.2/46.3</td><td>63.23/63.76</td></tr><tr><td>Pythia-160M</td><td>ReLU+l1</td><td>96.8/96.8</td><td>77.0/82.7</td><td>58.76/59.00</td></tr><tr><td>Pythia-1.4B</td><td>TopK</td><td>9.2/85.3</td><td>62.8/58.8</td><td>63.99/64.00</td></tr><tr><td>Pythia-1.4B</td><td>BatchTopK</td><td>15.7/90.0</td><td>62.3/53.6</td><td>63.35/64.18</td></tr><tr><td>Pythia-1.4B</td><td>ReLU+l1</td><td>97.8/98.5</td><td></td><td>83.5/86.7 58.40/59.67</td></tr><tr><td>Pythia-6.9B</td><td>TopK</td><td>8.8/89.6</td><td>62.4/61.1</td><td>63.99/63.89</td></tr><tr><td>Pythia-6.9B</td><td>BatchTopK</td><td>17.9/90.5</td><td>66.2/60.9</td><td>63.71/64.59</td></tr><tr><td>Pythia-6.9B</td><td>ReLU+l1</td><td>95.9/96.2</td><td>83.5/84.2</td><td>60.06/60.32</td></tr><tr><td>Llama-3.1-8B</td><td>TopK</td><td>60.4/91.3</td><td>66.8/67.9</td><td>63.71/63.37</td></tr><tr><td>Llama-3.1-8B</td><td>BatchTopK</td><td>58.6/92.2</td><td>70.3/69.6</td><td>64.49/65.25</td></tr><tr><td>Llama-3.1-8B</td><td>ReLU+l1</td><td>98.9/97.6</td><td>80.5/79.9</td><td>62.70/61.14</td></tr><tr><td>Qwen-2.5-7B</td><td>TopK</td><td>6.4/95.0</td><td>49.9/61.1</td><td>63.99/63.97</td></tr><tr><td>Qwen-2.5-7B</td><td>BatchTopK</td><td>32.3/92.9</td><td>54.7/62.5</td><td>63.41/63.77</td></tr><tr><td>Qwen-2.5-7B</td><td>ReLU+l1</td><td>96.7/97.9</td><td>71.0/72.7</td><td>59.15/58.36</td></tr><tr><td>Gemma-2-2B</td><td>TopK</td><td>82.4/88.5</td><td>58.3/62.3</td><td>63.91/63.77</td></tr><tr><td>Gemma-2-2B</td><td>BatchTopK</td><td>78.7/85.8</td><td>60.4/61.8</td><td>63.94/64.38</td></tr><tr><td>Gemma-2-2B</td><td>ReLU+l1</td><td>98.1/98.2</td><td>73.8/74.3</td><td>59.58/59.20</td></tr><tr><td>Gemma-2-9B</td><td>TopK</td><td>83.7/92.2</td><td>58.1/60.7</td><td>63.97/63.92</td></tr><tr><td>Gemma-2-9B</td><td>BatchTopK</td><td>84.0/85.9</td><td>61.6/62.7</td><td>64.20/64.59</td></tr><tr><td>Gemma-2-9B</td><td>ReLU+l1</td><td>94.6/95.2</td><td>74.3/74.8</td><td>59.73/59.36</td></tr></table>

## E Reconstruction Fidelity

## E.1 Total and content error

For reconstruction error $\mathbf { e } = \mathbf { x } - { \widehat { \mathbf { x } } } ,$ , let $P = \mathbf { v } _ { 1 } \mathbf { v } _ { 1 } ^ { \top }$ and $Q = I - P .$ . On the evaluation set,

$$
\mathrm { F V U } _ { \mathrm { t o t a l } } = \frac { \sum \| \mathbf { e } \| _ { 2 } ^ { 2 } } { \sum \| \mathbf { x } - \mu \| _ { 2 } ^ { 2 } } , \qquad \mathrm { F V U } _ { \| } = \frac { \sum \| P \mathbf { e } \| _ { 2 } ^ { 2 } } { \sum \| P ( \mathbf { x } - \mu ) \| _ { 2 } ^ { 2 } } ,\tag{5}
$$

$$
\mathrm { F V U _ { \perp } } = \frac { \sum \| Q \mathbf { e } \| _ { 2 } ^ { 2 } } { \sum \| Q ( \mathbf { x } - \pmb { \mu } ) \| _ { 2 } ^ { 2 } } , \qquad w _ { 1 } = \frac { \sum \| P ( \mathbf { x } - \pmb { \mu } ) \| _ { 2 } ^ { 2 } } { \sum \| \mathbf { x } - \pmb { \mu } \| _ { 2 } ^ { 2 } } .\tag{6}
$$

Since $P \mathbf { e }$ and $Q \mathbf { e }$ are orthogonal,

$$
\mathrm { F V U } _ { \mathrm { t o t a l } } = w _ { 1 } \mathrm { F V U } _ { \parallel } + ( 1 - w _ { 1 } ) \mathrm { F V U } _ { \perp } .\tag{7}
$$

For $\mathrm { F V U } _ { \parallel } \approx 0 $ , this gives $\mathrm { F V U _ { \perp } / F V U _ { t o t a l } } \approx 1 / ( 1 - w _ { 1 } )$

Table 28 gives total and content FVU on 200,000 WikiText-103 test tokens. Content FVU is $\mathrm { F V U _ { \perp } }$ S and B mean standard and bypass. For the standard SAE, $R = \mathrm { F V U _ { \perp } / F V U _ { \mathrm { t o t a l } } }$ . Gain is the percentage decrease in content FVU.

Table 28: Total and content FVU on WikiText-103 test tokens.
<table><tr><td>Model</td><td>Seeds</td><td>w1 (%)</td><td>Total S</td><td>Content S</td><td>R</td><td>Total B</td><td>Content B</td><td>Gain</td></tr><tr><td>GPT-2</td><td>5</td><td>96.850</td><td>0.004428</td><td>0.14056</td><td>31.7</td><td>0.004330</td><td>0.13746</td><td>2.2</td></tr><tr><td>Pythia-70M</td><td>5</td><td>81.918</td><td>0.020368</td><td>0.11251</td><td>5.5</td><td>0.020168</td><td>0.11154</td><td>0.9</td></tr><tr><td>Pythia-160M</td><td>5</td><td>93.096</td><td>0.010274</td><td>0.14874</td><td>14.5</td><td>0.010123</td><td>0.14662</td><td>1.4</td></tr><tr><td>Pythia-1.4B</td><td>5</td><td>96.206</td><td>0.008595</td><td>0.22650</td><td>26.4</td><td>0.008459</td><td>0.22294</td><td>1.6</td></tr><tr><td>Pythia-6.9B</td><td>3</td><td>95.501</td><td>0.012763</td><td>0.28363</td><td>22.2</td><td>0.012520</td><td>0.27831</td><td>1.9</td></tr><tr><td>Llama-3.1-8B</td><td>3</td><td>96.738</td><td>0.008330</td><td>0.25510</td><td>30.6</td><td>0.007914</td><td>0.24259</td><td>4.9</td></tr><tr><td>Qwen-2.5-7B</td><td>3</td><td>99.888</td><td>0.000370</td><td>0.32978</td><td>891.5</td><td>0.000244</td><td>0.21791</td><td>33.9</td></tr><tr><td>Gemma-2-2B</td><td>5</td><td>77.575</td><td>0.046726</td><td>0.20827</td><td>4.5</td><td>0.046164</td><td>0.20586</td><td>1.2</td></tr><tr><td>Gemma-2-9B</td><td>3</td><td>69.286</td><td>0.066676</td><td>0.21699</td><td>3.3</td><td>0.066122</td><td>0.21528</td><td>0.8</td></tr></table>

## E.2 PCA baselines

We compare SAEs with fixed and adaptive PCA, each using 64 coefficients outside PC1. Fixed PCA uses the next 64 principal components. Adaptive PCA selects the 64 coefficients with the largest squared values for each token. We fit PCA on one million randomly sampled training activations and evaluate all methods on the final 100,000 training activations, using SAE seed 42 (Table 29).

Table 29: Content FVU for PCA reconstructions and TopK SAEs.
<table><tr><td>Model</td><td>Fixed PCA</td><td>Adaptive PCA</td><td>Standard SAE</td><td>Bypass SAE</td></tr><tr><td>GPT-2</td><td>0.54452</td><td>0.36167</td><td>0.14137</td><td>0.13786</td></tr><tr><td>Pythia-70M</td><td>0.33588</td><td>0.21147</td><td>0.11594</td><td>0.11471</td></tr><tr><td>Pythia-160M</td><td>0.43916</td><td>0.30388</td><td>0.15142</td><td>0.14952</td></tr><tr><td>Pythia-1.4B</td><td>0.66039</td><td>0.52780</td><td>0.22899</td><td>0.22582</td></tr><tr><td>Pythia-6.9B</td><td>0.74459</td><td>0.63860</td><td>0.28156</td><td>0.27637</td></tr><tr><td>Llama-3.1-8B</td><td>0.69284</td><td>0.58520</td><td>0.24665</td><td>0.23346</td></tr><tr><td>Qwen-2.5-7B</td><td>0.59454</td><td>0.47230</td><td>0.31584</td><td>0.21153</td></tr><tr><td>Gemma-2-2B</td><td>0.52252</td><td>0.42584</td><td>0.20689</td><td>0.20397</td></tr><tr><td>Gemma-2-9B</td><td>0.52175</td><td>0.43506</td><td>0.21028</td><td>0.20839</td></tr></table>

## E.3 Matched coefficient budgets

The extra PC1 coefficient gives bypass 65 coefficients in total. We repeat training with $k = 6 3$ to bring this down to 64. Table 30 gives content FVU and the fraction of the improvement retained, $G _ { 6 3 } \mathrm { : }$

$$
G _ { 6 3 } = \frac { \mathrm { F V U } _ { \perp , \mathrm { s t d } } - \mathrm { F V U } _ { \perp , 6 3 + 1 } } { \mathrm { F V U } _ { \perp , \mathrm { s t d } } - \mathrm { F V U } _ { \perp , 6 4 + 1 } } .
$$

The first three models use the test set above. For Qwen, we use three seeds and 64 validation sequences. The Total column counts PC1; $L _ { 0 }$ does not.

Table 30: Reconstruction with 64 or 65 total coefficients.
<table><tr><td>Model</td><td>Method</td><td>Seeds</td><td>Total</td><td> $L _ { 0 }$ </td><td>Content FVU</td><td> $G _ { 6 3 }$  (%)</td></tr><tr><td>GPT-2</td><td>Standard</td><td>5</td><td></td><td>64 64.00</td><td> $0 . 1 4 0 5 6 \pm 0 . 0 0 0 2 8$ </td><td></td></tr><tr><td>GPT-2</td><td>Bypass 63+1</td><td>5</td><td></td><td>64 62.82</td><td> $0 . 1 3 8 2 2 \pm 0 . 0 0 0 1 1$ </td><td>75.3</td></tr><tr><td>GPT-2</td><td>Bypass 64+1</td><td>5</td><td>65</td><td>63.82</td><td> $0 . 1 3 7 4 6 \pm 0 . 0 0 0 1 2$ </td><td>一</td></tr><tr><td>Pythia-1.4B</td><td>Standard</td><td>3</td><td>64</td><td>63.98</td><td> $0 . 2 2 6 4 2 \pm 0 . 0 0 0 1 5$ </td><td>一</td></tr><tr><td>Pythia-1.4B</td><td>Bypass 63+1</td><td>3</td><td></td><td>64 63.00</td><td> $0 . 2 2 3 6 4 \pm 0 . 0 0 0 1 0$ </td><td>80.1</td></tr><tr><td>Pythia-1.4B</td><td>Bypass 64+1</td><td>3</td><td></td><td>65 64.00</td><td> $0 . 2 2 2 9 5 \pm 0 . 0 0 0 1 2$ </td><td>一</td></tr><tr><td>Gemma-2-2B</td><td>Standard</td><td>3</td><td>64</td><td>63.92</td><td> $0 . 2 0 8 4 9 \pm 0 . 0 0 0 3 0$ </td><td>一</td></tr><tr><td>Gemma-2-2B</td><td>Bypass 63+1</td><td>3</td><td></td><td>64 62.81</td><td> $0 . 2 0 6 7 8 \pm 0 . 0 0 0 1 3$ </td><td>67.5</td></tr><tr><td>Gemma-2-2B</td><td>Bypass 64+1</td><td>3</td><td>65</td><td>63.77</td><td> $0 . 2 0 5 9 7 \pm 0 . 0 0 0 0 5$ </td><td>一</td></tr><tr><td>Qwen-2.5-7B</td><td>Standard</td><td>3</td><td>64</td><td>63.98</td><td> $0 . 3 2 3 2 9 \pm 0 . 0 0 8 3 9$ </td><td>一</td></tr><tr><td>Qwen-2.5-7B</td><td>Bypass 63+1</td><td>3</td><td>64</td><td>62.95</td><td> $0 . 2 0 1 8 4 \pm 0 . 0 0 0 1 3$ </td><td>99.4</td></tr><tr><td>Qwen-2.5-7B</td><td>Bypass 64+1</td><td>3</td><td>65</td><td>63.95</td><td> $0 . 2 0 1 0 9 \pm 0 . 0 0 0 5 6$ </td><td></td></tr></table>

Table 31 gives counts of PC1-aligned features $( a _ { j } \geq 0 . 0 1 )$ and mean squared reconstruction error outside PC1 for large-norm and ordinary tokens. The norm cutoffs are GPT-2: 752.45; Pythia-1.4B: 525.76; Qwen: 2847.50. We use 200,000 tokens per model and average over five, five, and three seeds, respectively. Gain share gives the percentage of the total error reduction due to each group, averaged over seeds.

Table 31: Active PC1-aligned features and reconstruction error by token norm.
<table><tr><td>Model</td><td>Tokens</td><td>Tokens (%) Active</td><td> $a _ { j }$  ≥.01</td><td>Error S</td><td>Error B</td><td>Gain share (%)</td></tr><tr><td>GPT-2</td><td>Large norm</td><td>1.40</td><td>63.11</td><td>247.22</td><td>105.95</td><td>15.9</td></tr><tr><td>GPT-2</td><td>Ordinary</td><td>98.60</td><td>2.85</td><td>575.84</td><td>565.08</td><td>84.1</td></tr><tr><td>Pythia-1.4B</td><td>Large norm</td><td>2.87</td><td>62.94</td><td>158.64</td><td>61.08</td><td>47.0</td></tr><tr><td>Pythia-1.4B</td><td>Ordinary</td><td>97.13</td><td>1.26</td><td>385.30</td><td>382.05</td><td>53.0</td></tr><tr><td>Qwen-2.5-7B</td><td>Large norm</td><td>1.88</td><td>63.17</td><td>21003.88</td><td>2506.10</td><td>66.8</td></tr><tr><td> $\mathbf { Q } \mathrm { w e n } { - 2 . 5 { - 7 } \mathbf { B } }$ </td><td>Ordinary</td><td>98.12</td><td>26.91</td><td>1166.54</td><td>988.66</td><td>33.2</td></tr></table>

## E.4 Splice cost

Splice cost is the increase in cross-entropy when model activations are replaced by their SAE reconstructions, including restored PC1 for bypass. Table 32 uses seed 42 and 256 validation sequences per model. Model CE is cross-entropy before replacement; reduction is the percentage decrease in splice cost from standard to bypass.

Table 32: Cross-entropy and its increase under SAE reconstruction, in nats per prediction.
<table><tr><td>Model</td><td>Predictions</td><td>Model CE</td><td>Standard cost</td><td>Bypass cost</td><td>Reduction (%)</td></tr><tr><td>GPT-2</td><td>17,558</td><td>3.95775</td><td>0.08054</td><td>0.07386</td><td>8.3</td></tr><tr><td>Pythia-70M</td><td>17,571</td><td>4.53997</td><td>0.17583</td><td>0.17470</td><td>0.6</td></tr><tr><td>Pythia-160M</td><td>17,571</td><td>3.95300</td><td>0.15933</td><td>0.14691</td><td>7.8</td></tr><tr><td>Pythia-1.4B</td><td>17,571</td><td>3.19029</td><td>0.10813</td><td>0.09693</td><td>10.4</td></tr><tr><td>Pythia-6.9B</td><td>17,571</td><td>2.96687</td><td>0.23805</td><td>0.21291</td><td>10.6</td></tr><tr><td>Llama-3.1-8B</td><td>17,931</td><td>2.72509</td><td>0.08931</td><td>0.08647</td><td>3.2</td></tr><tr><td>Qwen-2.5-7B</td><td>18,059</td><td>2.50963</td><td>0.11548</td><td>0.06485</td><td>43.8</td></tr><tr><td>Gemma-2-2B</td><td>18,097</td><td>3.10721</td><td>0.08857</td><td>0.08438</td><td>4.7</td></tr><tr><td>Gemma-2-9B</td><td>18,097</td><td>2.88440</td><td>0.08041</td><td>0.07812</td><td>2.8</td></tr></table>

## F Behavioral Interventions

## F.1 Subject–verb agreement prompts

We construct 32,000 prompts from ten sentence templates, 40 subject words, and 20 distractor words, each in singular and plural form. Changing the subject gives the donor prompt; changing the distractor gives the control. The donor requires a different verb, while the control keeps the original answer. We keep examples of equal token length with corresponding words at the same positions. Calibration, feature selection, validation, test, and confirmation use separate templates and words.

For each SAE, we select from four sets of 50 features: the standard top 50, the top 50 after excluding $a _ { j } \geq 0 . 0 1$ (post-hoc), the bypass top 50, and a random set of active features. We rank features by zero-ablation damage on WikiText, then choose edits on the selection examples. At each step, we add the feature that most increases the logit difference between the donor verb and the original verb. For the random comparison, we repeat with three sets and average.

## F.2 Editing activations

Let h<sub>r</sub> be the original prompt’s activation and $\mathbf { h } _ { d }$ the donor’s. For selected features J, we apply

$$
\mathbf { h } _ { r } ^ { \prime } = \mathbf { h } _ { r } + \lambda \sum _ { j \in J } [ z _ { j } ( \mathbf { h } _ { d } ) - z _ { j } ( \mathbf { h } _ { r } ) ] \mathbf { d } _ { j } .\tag{8}
$$

For the control, use $\mathbf { h } _ { f }$ in place of $\mathbf { h } _ { d } .$ These edits leave the reconstruction error and bypass PC1 coefficient in place.

We edit the final token for Pythia-160M and all non-padding tokens for the other models.

Table 33 uses five features and λ = 1. Success means that copying from the donor changes the preferred verb, while copying from the control does not. We exclude examples that already favor the donor verb. Control is the percentage of all test prompts whose preferred verb is unchanged by the control edit, averaged over seeds.

Table 33: Agreement-edit success (%) with five features and $\lambda = 1$
<table><tr><td>Model</td><td>Set</td><td>Eligible/total</td><td>42</td><td>137</td><td></td><td>0 Mean</td><td>Control</td></tr><tr><td>GPT-2</td><td>Standard</td><td>512/512</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>100.00</td></tr><tr><td>GPT-2</td><td>Post-hoc</td><td>512/512</td><td>6.25</td><td>17.58</td><td>3.32</td><td>9.05</td><td>100.00</td></tr><tr><td>GPT-2</td><td>Bypass</td><td>512/512</td><td>4.10</td><td>15.23</td><td>1.95</td><td>7.10</td><td>100.00</td></tr><tr><td>GPT-2</td><td>Random</td><td>512/512</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>100.00</td></tr><tr><td>Pythia-160M</td><td>Standard</td><td>264/264</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>100.00</td></tr><tr><td>Pythia-160M</td><td>Post-hoc</td><td>264/264</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>99.75</td></tr><tr><td>Pythia-160M</td><td>Bypass</td><td>264/264</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>99.87</td></tr><tr><td>Pythia-160M</td><td>Random</td><td>264/264</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>100.00</td></tr><tr><td>Pythia-1.4B</td><td>Standard</td><td>264/264</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>100.00</td></tr><tr><td>Pythia-1.4B</td><td>Post-hoc</td><td>264/264</td><td>34.47</td><td>32.20</td><td>29.17</td><td>31.94</td><td>100.00</td></tr><tr><td>Pythia-1.4B</td><td>Bypass</td><td>264/264</td><td>39.39</td><td>36.74</td><td>36.36</td><td>37.50</td><td>100.00</td></tr><tr><td>Pythia-1.4B</td><td>Random</td><td>264/264</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>100.00</td></tr><tr><td>Qwen-2.5-7B</td><td>Standard</td><td>506/512</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>100.00</td></tr><tr><td>Model</td><td>Set</td><td>Eligible/total</td><td>42</td><td>137</td><td>0</td><td>Mean</td><td>Control</td></tr><tr><td>Qwen-2.5-7B</td><td>Post-hoc</td><td>506/512</td><td>3.56</td><td>3.75</td><td>5.34</td><td>4.22</td><td>98.31</td></tr><tr><td>Qwen-2.5-7B</td><td>Bypass</td><td>506/512</td><td>2.37</td><td>2.37</td><td>1.38</td><td>2.04</td><td>98.24</td></tr><tr><td>Qwen-2.5-7B</td><td>Random</td><td>506/512</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>100.00</td></tr><tr><td>Gemma-2-2B</td><td>Standard</td><td>512/512</td><td>64.45</td><td>64.06</td><td>65.82</td><td>64.78</td><td>100.00</td></tr><tr><td>Gemma-2-2B</td><td>Post-hoc</td><td>512/512</td><td>64.84</td><td>63.87</td><td>65.82</td><td>64.84</td><td>100.00</td></tr><tr><td>Gemma-2-2B</td><td>Bypass</td><td>512/512</td><td>62.70</td><td>65.43</td><td>64.26</td><td>64.13</td><td>100.00</td></tr><tr><td>Gemma-2-2B</td><td>Random</td><td>512/512</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>100.00</td></tr></table>

## F.3 Number of features, edit strength, and transfer

Table 34 gives success rates for b edited features and strength λ. We vary b with λ = 1 and vary λ with b = 5. Here λ = 1 copies the donor’s coefficients; $\lambda = 2$ doubles the change.

Table 34: Agreement-edit success (%) by number of features and edit strength.
<table><tr><td>Model</td><td>Set</td><td>b = 1</td><td>b = 3</td><td>b = 5</td><td>λ = .25</td><td>λ = .5</td><td>λ = 2</td></tr><tr><td>GPT-2</td><td>Standard</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>GPT-2</td><td>Post-hoc</td><td>8.66</td><td>8.98</td><td>9.05</td><td>0.00</td><td>0.00</td><td>49.41</td></tr><tr><td>GPT-2</td><td>Bypass</td><td>7.03</td><td>7.10</td><td>7.10</td><td>0.00</td><td>0.00</td><td>48.70</td></tr><tr><td>GPT-2</td><td>Random</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Pythia-160M</td><td>Standard</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Pythia-160M</td><td>Post-hoc</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Pythia-160M</td><td>Bypass</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Pythia-160M</td><td>Random</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Pythia-1.4B</td><td>Standard</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Pythia-1.4B</td><td>Post-hoc</td><td>31.82</td><td>31.94</td><td>31.94</td><td>0.00</td><td>0.00</td><td>72.85</td></tr><tr><td>Pythia-1.4B</td><td>Bypass</td><td>37.50</td><td>37.63</td><td>37.50</td><td>0.00</td><td>0.63</td><td>72.35</td></tr><tr><td>Pythia-1.4B</td><td>Random</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Qwen-2.5-7B</td><td>Standard</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.53</td></tr><tr><td>Qwen-2.5-7B</td><td>Post-hoc</td><td>4.22</td><td>4.22</td><td>4.22</td><td>0.00</td><td>0.13</td><td>47.36</td></tr><tr><td>Qwen-2.5-7B</td><td>Bypass</td><td>1.84</td><td>1.98</td><td>2.04</td><td>0.00</td><td>0.07</td><td>29.58</td></tr><tr><td>Qwen-2.5-7B</td><td>Random</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Gemma-2-2B</td><td>Standard</td><td>64.45</td><td>64.71</td><td>64.78</td><td>0.00</td><td>0.26</td><td>78.78</td></tr><tr><td>Gemma-2-2B</td><td>Post-hoc</td><td>64.45</td><td>64.84</td><td>64.84</td><td>0.00</td><td>0.33</td><td>78.65</td></tr><tr><td>Gemma-2-2B</td><td>Bypass</td><td>64.13</td><td>64.00</td><td>64.13</td><td>0.00</td><td>0.39</td><td>78.39</td></tr><tr><td>Gemma-2-2B</td><td>Random</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr></table>

Table 35 gives success after matching the selected features to another SAE by signed decoder cosine. The edit strength stays the same. Top 50 restricts matching to the other SAE’s top 50. Full dict. searches all features; Full + filter then excludes matches outside the top 50. A dash means all matches were excluded. Results average six ordered seed pairs. Original gives success before transfer; Cosine and Repeats give the mean cosine and number of duplicate matches when searching the top 50.

Table 35: Agreement-edit success (%) after transferring five-feature edits between SAEs.
<table><tr><td>Model</td><td>Set</td><td>Original</td><td>Top 50</td><td>Full dict.</td><td>Full + filter</td><td>Cosine</td><td>Repeats</td></tr><tr><td>GPT-2</td><td>Standard</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.126</td><td>0.17</td></tr><tr><td>GPT-2</td><td>Post-hoc</td><td>9.05</td><td>8.59</td><td>8.85</td><td>8.72</td><td>0.827</td><td>0.50</td></tr><tr><td>GPT-2</td><td>Bypass</td><td>7.10</td><td>7.23</td><td>7.75</td><td>7.23</td><td>0.888</td><td>0.17</td></tr><tr><td>GPT-2</td><td>Random</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.127</td><td>0.22</td></tr><tr><td>Pythia-160M</td><td>Standard</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.297</td><td>0.00</td></tr><tr><td>Pythia-160M</td><td>Post-hoc</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.965</td><td>0.33</td></tr><tr><td>Pythia-160M</td><td>Bypass</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.860</td><td>0.33</td></tr><tr><td>Pythia-160M</td><td>Random</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.105</td><td>0.22</td></tr><tr><td>Pythia-1.4B</td><td>Standard</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.269</td><td>0.67</td></tr><tr><td>Pythia-1.4B</td><td>Post-hoc</td><td>31.94</td><td>31.94</td><td>31.94</td><td>31.94</td><td>0.961</td><td>0.33</td></tr><tr><td>Pythia-1.4B</td><td>Bypass</td><td>37.50</td><td>37.25</td><td>37.18</td><td>37.50</td><td>0.943</td><td>0.00</td></tr><tr><td>Pythia-1.4B</td><td>Random</td><td>0.00</td><td>0.00</td><td>0.00</td><td></td><td>0.085</td><td>0.11</td></tr><tr><td>Qwen-2.5-7B</td><td>Standard</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.446</td><td>0.33</td></tr><tr><td>Qwen-2.5-7B</td><td>Post-hoc</td><td>4.22</td><td>4.22</td><td>4.22</td><td>4.22</td><td>0.967</td><td>0.00</td></tr><tr><td>Model</td><td>Set</td><td>Original Top 50</td><td></td><td></td><td>Full dict. Full + filter</td><td>Cosine</td><td>Repeats</td></tr><tr><td>Qwen-2.5-7B</td><td>Bypass</td><td>2.04</td><td>1.94</td><td>1.91</td><td>1.94</td><td>0.807</td><td>0.33</td></tr><tr><td>Qwen-2.5-7B</td><td>Random</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.088</td><td>0.28</td></tr><tr><td>Gemma-2-2B</td><td>Standard</td><td>64.78</td><td>64.45</td><td>64.45</td><td>64.45</td><td>0.955</td><td>0.17</td></tr><tr><td>Gemma-2-2B</td><td>Post-hoc</td><td>64.84</td><td>64.84</td><td>64.78</td><td>64.84</td><td>0.856</td><td>0.00</td></tr><tr><td>Gemma-2-2B</td><td>Bypass</td><td>64.13</td><td>63.90</td><td>64.00</td><td>64.03</td><td>0.955</td><td>0.33</td></tr><tr><td>Gemma-2-2B</td><td>Random</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.096</td><td>0.33</td></tr></table>

## F.4 Agreement and indirect-object identification

We also test subject–verb agreement (SVA) and indirect-object identification (IOI) with edits applied to SAE reconstructions. The calibration, selection, validation, and test sets have separate templates and words, with at most 512, 2,048, 1,024, and 4,096 examples.

Let $y _ { r }$ and $y _ { d }$ be the original and donor answers, and $\ell _ { r }$ and $\ell _ { d }$ their unedited logit vectors. We keep examples satisfying

$$
\ell _ { r } ( y _ { r } ) > \ell _ { r } ( y _ { d } ) , \qquad \ell _ { d } ( y _ { d } ) > \ell _ { d } ( y _ { r } ) .
$$

We require at least 60% calibration accuracy and 200 eligible examples, which excludes Pythia-70M from IOI.

We use the same three top-50 sets and three random sets as above. On the first 256 eligible selection examples, we add whichever feature gives the highest success rate. Ties are broken by the mean difference between donor-answer and original-answer logits. Edits apply at every non-padding position:

$$
\widehat { \mathbf { h } } _ { r } ^ { \prime } = \widehat { \mathbf { h } } _ { r } + \lambda \sum _ { j \in J } [ z _ { j } ( \mathbf { h } _ { d } ) - z _ { j } ( \mathbf { h } _ { r } ) ] \mathbf { d } _ { j } .\tag{9}
$$

Here $\widehat { \mathbf { h } } _ { r }$ is the SAE reconstruction, including the original prompt’s PC1 component for bypass. Success requires the donor edit to favor the donor answer and the control edit to favor the original answer. Table 36 reports the mean success rate across seeds with five features and λ = 1. We also test one, three, five, and ten features and $\lambda \in \{ 0 . 2 5 , 0 . 5 , 1 , 2 \}$

Table 36: Agreement and indirect-object edit success (%).
<table><tr><td>Task Model</td><td>Seeds</td><td>Standard</td><td>Post-hoc</td><td>Bypass</td></tr><tr><td>SVA GPT-2</td><td>5</td><td>0.00</td><td>34.27</td><td>26.40</td></tr><tr><td>SVA Pythia-70M</td><td>5</td><td>74.85</td><td>74.60</td><td>76.31</td></tr><tr><td>SVA Pythia-160M</td><td>5</td><td>0.02</td><td>15.55</td><td>17.60</td></tr><tr><td>SVA Pythia-1.4B</td><td>5</td><td>0.00</td><td>67.04</td><td>67.14</td></tr><tr><td>SVA Pythia-6.9B</td><td>1</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>SVA Llama-3.1-8B</td><td>1</td><td>31.85</td><td>29.82</td><td>31.97</td></tr><tr><td>SVA Qwen-2.5-7B</td><td>1</td><td>0.00</td><td>17.41</td><td>9.63</td></tr><tr><td>SVA Gemma-2-2B</td><td>5</td><td>77.14</td><td>77.09</td><td>75.10</td></tr><tr><td>SVA</td><td>Gemma-2-9B 1</td><td>56.26</td><td>56.26</td><td>63.95</td></tr><tr><td>IOI GPT-2</td><td></td><td>5</td><td>0.44 12.26</td><td>9.90</td></tr><tr><td>IOI</td><td>Pythia-160M</td><td>5</td><td>0.74</td><td>1.23 1.09</td></tr><tr><td>IOI</td><td>Pythia-1.4B</td><td>5</td><td>0.21</td><td>0.42 0.60</td></tr><tr><td>IOI</td><td>Pythia-6.9B</td><td>1</td><td>1.23</td><td>3.29 2.99</td></tr><tr><td>IOI</td><td>Llama-3.1-8B</td><td>1</td><td>0.27</td><td>1.15 1.27</td></tr><tr><td>IOI</td><td>Qwen-2.5-7B</td><td>1</td><td>0.02</td><td>0.93 1.50</td></tr><tr><td>IOI</td><td>Gemma-2-2B</td><td>5</td><td>2.73</td><td>3.20 2.16</td></tr><tr><td>IOI</td><td>Gemma-2-9B</td><td>1</td><td>2.42</td><td>2.42 2.49</td></tr></table>

Some runs retain slightly different test examples. For Table 37, we use seed 42 and keep only examples shared by the compared runs. Its 95% intervals come from 2,000 bootstrap samples of these examples, sampled within each sentence-template group.

Table 37: Differences in edit success on the same test examples, in percentage points [95% interval].
<table><tr><td>Task Model</td><td>Examples</td><td>Post-hoc — standard</td><td>Bypass — standard</td></tr><tr><td>SVA GPT-2</td><td>4096</td><td>35.1 [33.6, 36.5]</td><td>20.0 [18.8, 21.2]</td></tr><tr><td>SVA Pythia-70M</td><td>3346</td><td>0.9 [0.1, 1.6]</td><td>5.1 [4.0, 6.2]</td></tr><tr><td>SVA Pythia-160M</td><td>3454</td><td>7.9 [7.0, 8.8]</td><td>20.4 [19.2, 21.8]</td></tr><tr><td>SVA Pythia-1.4B</td><td>3456</td><td>65.2 [63.6, 66.8]</td><td>66.8 [65.2, 68.3]</td></tr><tr><td>SVA Pythia-6.9B</td><td>3456</td><td>0.0 [0.0, 0.0]</td><td>0.0 [0.0, 0.0]</td></tr><tr><td>SVA Llama-3.1-8B</td><td>4094</td><td>-2.0 [-2.5, -1.6]</td><td>0.1 [-0.8, 1.0]</td></tr><tr><td>SVA Qwen-2.5-7B</td><td>4060</td><td>17.4 [16.4, 18.4]</td><td>9.6 [8.8, 10.5]</td></tr><tr><td>SVA Gemma-2-2B</td><td>4093</td><td>0.0 [0.0, 0.1]</td><td>-0.6 [-1.5, 0.2]</td></tr><tr><td>SVA Gemma-2-9B</td><td>4092</td><td>0.0 [0.0, 0.0]</td><td>7.7 [6.7, 8.8]</td></tr><tr><td>IOI GPT-2</td><td>3246</td><td>10.3 [9.2, 11.3]</td><td>11.2 [10.1, 12.3]</td></tr><tr><td>IOI</td><td>Pythia-160M 3561</td><td>0.1 [-0.3, 0.5]</td><td>-0.2 [-0.7, 0.2]</td></tr><tr><td>IOI Pythia-1.4B</td><td>4094</td><td>-0.2 [-0.4, 0.0]</td><td>-0.2 [-0.5, 0.0]</td></tr><tr><td>IOI Pythia-6.9B</td><td>4068</td><td>2.1 [1.5, 2.6]</td><td>1.7 [1.2, 2.3]</td></tr><tr><td>IOI Llama-3.1-8B</td><td>4093</td><td>0.9 [0.5, 1.2]</td><td>1.0 [0.7, 1.4]</td></tr><tr><td>IOI Qwen-2.5-7B</td><td>4077</td><td>0.9 [0.6, 1.2]</td><td>1.5 [1.1, 1.8]</td></tr><tr><td>IOI</td><td>Gemma-2-2B 4095</td><td>0.0 [-0.1, 0.2]</td><td>-0.5 [-1.1, 0.1]</td></tr><tr><td>IOI</td><td>Gemma-2-9B 4096</td><td>0.0 [0.0, 0.0]</td><td>0.1 [-0.4, 0.6]</td></tr></table>

Controls include no edit, copying a prompt’s own feature values, replacing the full activation, replacing only PC1, and replacing only the component outside PC1. In IOI examples, the original and donor PC1 coefficients can be identical.

## G Automated Interpretability

Feature and context selection. We evaluate the seed-42 SAEs at the layers in Table 2. For each SAE, we take the 100 features ranked highest by zero-ablation damage and a random sample of 100 active features. The explanation and evaluation articles are separate, with about 500,000 GPT-2 tokens in each set. Examples are selected as follows:

<table><tr><td>Use</td><td>Examples per feature</td></tr><tr><td>Explanation</td><td>Up to ten top-activating examples and five from each quartile of nonzero activations.</td></tr><tr><td>Evaluation</td><td>Up to 50 activating examples across those quartiles and 50 random inactive examples.</td></tr><tr><td>Harder negatives</td><td>Up to 50 inactive examples where a feature with a similar decoder fires.</td></tr></table>

Quartiles are fitted on the explanation articles. An example contains the target token and up to 32 preceding tokens.

Generating and testing explanations. Qwen2.5-32B-Instruct and Mistral-Small-24B-Instruct-2501 each write descriptions and test the other’s descriptions. The prompt asks for one sentence; generation is greedy with a 120-token limit.

We then test how well the other model predicts feature firing from just the description and a text excerpt. For each prediction, we take the log odds of Yes over No, combining the first-token forms of each answer. Tables 38 and 39 give mean AUROC. Top denotes the top 100 features; Control denotes the random 100. Each pair is standard/bypass.

Table 38: Feature-detection AUROC with Qwen explaining and Mistral evaluating.
<table><tr><td>Model</td><td>Top: random</td><td>Control: random</td><td>Top: hard</td><td>Control: hard</td></tr><tr><td>GPT-2</td><td>0.470/0.727</td><td>0.753/0.757</td><td>0.411/0.633</td><td>0.677/0.679</td></tr><tr><td>Pythia-160M</td><td>0.566/0.737</td><td>0.723/0.737</td><td>0.539/0.618</td><td>0.641/0.657</td></tr><tr><td>Pythia-1.4B</td><td>0.551/0.732</td><td>0.773/0.755</td><td>0.475/0.621</td><td>0.668/0.646</td></tr><tr><td>Pythia-6.9B</td><td>0.568/0.757</td><td>0.812/0.799</td><td>0.527/0.636</td><td>0.700/0.694</td></tr><tr><td>Qwen-2.5-7B</td><td>0.521/0.722</td><td>0.789/0.818</td><td>0.509/0.629</td><td>0.670/0.705</td></tr><tr><td>Llama-3.1-8B</td><td>0.835/0.742</td><td>0.819/0.805</td><td>0.780/0.599</td><td>0.675/0.665</td></tr><tr><td>Gemma-2-2B</td><td>0.800/0.760</td><td>0.832/0.806</td><td>0.679/0.623</td><td>0.708/0.695</td></tr><tr><td>Gemma-2-9B</td><td>0.761/0.768</td><td>0.833/0.801</td><td>0.620/0.602</td><td>0.675/0.680</td></tr></table>

Table 39: Feature-detection AUROC with Mistral explaining and Qwen evaluating.
<table><tr><td>Model</td><td>Top: random</td><td>Control: random</td><td>Top: hard</td><td>Control: hard</td></tr><tr><td>GPT-2</td><td>0.665/0.715</td><td>0.714/0.720</td><td>0.629/0.619</td><td>0.672/0.648</td></tr><tr><td>Pythia-160M</td><td>0.626/0.708</td><td>0.704/0.704</td><td>0.589/0.615</td><td>0.638/0.661</td></tr><tr><td>Pythia-1.4B</td><td>0.669/0.743</td><td>0.734/0.755</td><td>0.554/0.631</td><td>0.665/0.663</td></tr><tr><td>Pythia-6.9B</td><td>0.645/0.737</td><td>0.791/0.766</td><td>0.594/0.608</td><td>0.713/0.684</td></tr><tr><td>Qwen-2.5-7B</td><td>0.586/0.699</td><td>0.764/0.779</td><td>0.625/0.621</td><td>0.673/0.700</td></tr><tr><td>Llama-3.1-8B</td><td>0.831/0.733</td><td>0.783/0.772</td><td>0.779/0.581</td><td>0.691/0.689</td></tr><tr><td>Gemma-2-2B</td><td>0.789/0.751</td><td>0.786/0.757</td><td>0.684/0.629</td><td>0.705/0.705</td></tr><tr><td>Gemma-2-9B</td><td>0.728/0.757</td><td>0.777/0.746</td><td>0.626/0.629</td><td>0.675/0.666</td></tr></table>

## G.1 Uncertainty

Table 40 compares the top features using random negatives. We compute 95% intervals from 2,000 bootstrap samples of features, sampling separately from features in the top set only, the random set only, and both sets. A feature belonging to both is included in both averages whenever sampled. Standard and bypass features are sampled independently.

Table 40: Bypass minus standard AUROC for the top features, with 95% intervals.
<table><tr><td>Model</td><td>Qwen explains</td><td>Mistral explains</td></tr><tr><td>GPT-2</td><td>0.257 [0.196, 0.313]</td><td>0.049 [0.008, 0.092]</td></tr><tr><td>Pythia-160M</td><td>0.171 [0.119, 0.225]</td><td>0.082 [0.037, 0.123]</td></tr><tr><td>Pythia-1.4B</td><td>0.181 [0.123, 0.235]</td><td>0.074 [0.030, 0.117]</td></tr><tr><td>Pythia-6.9B</td><td>0.188 [0.139, 0.237]</td><td>0.093 [0.049, 0.138]</td></tr><tr><td>Llama-3.1-8B</td><td>-0.093 [-0.139, -0.048]</td><td>-0.097 [-0.147, -0.047]</td></tr><tr><td>Qwen-2.5-7B</td><td>0.200 [0.136, 0.267]</td><td>0.113 [0.064, 0.158]</td></tr><tr><td>Gemma-2-2B Gemma-2-9B</td><td>-0.040 [-0.079, -0.001] 0.007 [-0.030, 0.045]</td><td>-0.038 [-0.079, 0.004] 0.030 [-0.015, 0.074]</td></tr></table>

## H Sparse Probing

Data and fitting. We train classifiers on SAE features for SST-2, AG News, Amazon Polarity, and Emotion, using five standard and five bypass SAEs for each model in Table 41 (seeds 42, 137, 0, 1, and 2). For each task, we randomly sample up to 2,000 training examples and 500 test examples per class. We remove duplicate texts, including test texts that appear in training. Test examples come from each dataset’s official held-out split; for SST-2, we use the labelled validation split.

We use logistic regression with $k \in \{ 1 , 2 0 , 1 0 0 \}$ nonconstant features. Features are chosen by mutual information between firing and class label. We choose $C \in \{ 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 1 0 ^ { - 1 } , 1 , 1 0 , 1 0 0 \}$ by five-fold cross-validation, repeating feature selection and standardization on each training fold. The selected classifier is refitted on the training set before testing.

Inputs are mean feature activations over at most 64 non-padding tokens, or the final token’s activations in inputs of at most 128 tokens.

Table 41 gives accuracy with 20 features, averaged over five seeds. Each pair is standard/bypass. Table 42 gives their mean difference across tasks (∆), with 95% intervals from resampling test examples. Table 43 compares the feature counts and choice of tokens.

Table 41: Test accuracy (%) using 20 SAE features averaged across tokens.
<table><tr><td>Model</td><td>SST-2</td><td>AG News</td><td>Amazon</td><td>Emotion</td><td>∆</td></tr><tr><td>GPT-2</td><td>80.78/79.22</td><td>80.84/80.61</td><td>79.58/80.24</td><td>59.35/59.22</td><td>-0.31</td></tr><tr><td>Pythia-160M</td><td>78.12/78.35</td><td>80.92/79.91</td><td>79.76/80.62</td><td>55.84/57.76</td><td>+0.50</td></tr><tr><td>Pythia-1.4B</td><td>85.69/85.41</td><td>80.63/83.76</td><td>86.60/86.96</td><td>62.10/62.67</td><td>+0.95</td></tr><tr><td>Pythia-6.9B</td><td>87.82/87.64</td><td>80.00/78.44</td><td>86.98/88.00</td><td>61.24/63.24</td><td>+0.32</td></tr><tr><td>Qwen-2.5-7B</td><td>87.71/88.19</td><td>83.17/81.49</td><td>85.06/86.82</td><td>59.76/60.70</td><td>+0.38</td></tr><tr><td>Llama-3.1-8B</td><td>88.83/90.05</td><td>77.42/76.97</td><td>88.80/89.80</td><td>62.70/60.27</td><td>-0.17</td></tr><tr><td>Gemma-2-2B</td><td>90.50/90.60</td><td>80.99/80.56</td><td>89.56/89.86</td><td>61.03/62.10</td><td>+0.26</td></tr><tr><td>Gemma-2-9B</td><td>91.19/90.02</td><td>79.15/80.17</td><td>89.64/89.76</td><td>59.24/60.33</td><td>+0.26</td></tr></table>

Table 42: Bypass minus standard accuracy with 20 features, in percentage points.
<table><tr><td rowspan=1 colspan=2>Model       Difference95% interval</td></tr><tr><td rowspan=1 colspan=2>GPT-2            -0.31 [-0.78, 0.17]Pythia-160M       0.50 [0.03, 1.00]</td></tr><tr><td rowspan=1 colspan=1>Pythia-1.4B        0.95</td><td rowspan=1 colspan=1>[0.54, 1.36]</td></tr><tr><td rowspan=1 colspan=1>Pythia-6.9B        0.32[</td><td rowspan=1 colspan=1>-0.10, 0.75]</td></tr><tr><td rowspan=1 colspan=1>Llama-3.1-8B     -0.17 [</td><td rowspan=1 colspan=1>-0.61, 0.28]</td></tr><tr><td rowspan=1 colspan=1>Qwen-2.5-7B      0.38[</td><td rowspan=1 colspan=1>-0.12, 0.89]</td></tr><tr><td rowspan=1 colspan=1>Gemma-2-2B      0.26[</td><td rowspan=1 colspan=1>-0.15, 0.65]</td></tr><tr><td rowspan=1 colspan=1>Gemma-2-9B       0.26 [</td><td rowspan=1 colspan=1>-0.18, 0.67]</td></tr></table>

Table 43: Mean test accuracy (%) by number of features and choice of tokens.
<table><tr><td>Model</td><td>Mean:  $k = 1$ </td><td>Mean:  $k = 2 0$ </td><td>Mean:  $k = 1 0 0$ </td><td>Last:  $k = 2 0$ </td></tr><tr><td>GPT-2</td><td>59.10/56.55</td><td>75.14/74.82</td><td>78.99/79.13</td><td>63.49/63.75</td></tr><tr><td>Pythia-160M</td><td>55.61/55.52</td><td>73.66/74.16</td><td>77.82/78.19</td><td>55.92/61.78</td></tr><tr><td>Pythia-1.4B</td><td>56.43/58.16</td><td>78.75/79.70</td><td>82.63/82.98</td><td>65.11/64.84</td></tr><tr><td>Pythia-6.9B</td><td>59.67/59.13</td><td>79.01/79.33</td><td>82.96/83.37</td><td>66.38/66.94</td></tr><tr><td>Llama-3.1-8B</td><td>58.20/58.82</td><td>79.44/79.27</td><td>83.28/82.89</td><td>71.18/70.66</td></tr><tr><td>Qwen-2.5-7B</td><td>58.19/59.96</td><td>78.92/79.30</td><td>81.99/82.89</td><td>67.72/69.90</td></tr><tr><td>Gemma-2-2B</td><td>59.22/58.50</td><td>80.52/80.78</td><td>83.90/84.27</td><td>71.50/71.01</td></tr><tr><td>Gemma-2-9B</td><td>62.73/60.79</td><td>79.81/80.07</td><td>83.91/83.88</td><td>69.11/68.82</td></tr></table>

Table 44 compares probes using the full residual activation, the activation with PC1 removed, PC1 alone, and 20 randomly chosen active SAE features. Random-feature results average five selections per feature count. Bypass + PC1 adds the PC1 coefficient to 20 selected bypass features. All use mean activations over the first 64 tokens; accuracy is averaged over tasks and SAE seeds where applicable.

Table 44: Mean test accuracy (%) for probe baselines.
<table><tr><td>Model</td><td>Dense</td><td>Dense ⊥</td><td>PC1</td><td>Random S/B</td><td> $\mathrm { B y p a s s + P C l }$ </td></tr><tr><td>GPT-2</td><td>80.02</td><td>80.04</td><td>38.19</td><td>47.78/48.60</td><td>74.77</td></tr><tr><td>Pythia-160M</td><td>79.19</td><td>79.24</td><td>38.57</td><td>48.03/48.25</td><td>74.16</td></tr><tr><td>Pythia-1.4B</td><td>83.89</td><td>83.87</td><td>38.71</td><td>46.79/47.76</td><td>79.83</td></tr><tr><td>Pythia-6.9B</td><td>85.98</td><td>85.97</td><td>39.66</td><td>45.70/45.85</td><td>79.49</td></tr><tr><td>Llama-3.1-8B</td><td>86.41</td><td>86.40</td><td>38.38</td><td>45.54/46.20</td><td>79.37</td></tr><tr><td>Qwen-2.5-7B</td><td>86.20</td><td>86.27</td><td>38.72</td><td>46.65/46.72</td><td>79.22</td></tr><tr><td>Gemma-2-2B</td><td>86.04</td><td>86.04</td><td>39.66</td><td>48.25/47.00</td><td>80.86</td></tr><tr><td>Gemma-2-9B</td><td>86.40</td><td>86.57</td><td>39.75</td><td>47.07/46.24</td><td>80.27</td></tr></table>

## I What PC1 Encodes

## I.1 Correlations with norm and token frequency

For Pythia-1.4B at layer 12, we use 65,535 activations from 87 held-out articles. Table 45 reports the absolute Pearson correlation between the centered PC1 coefficient and activation norm, and the Spearman correlation between its absolute value and token frequency. Frequencies are counted in the five-million-token training sample. The 95% intervals come from 200 bootstrap samples of articles. The absolute cosine between PC1 and the training mean is 0.949.

Table 45: Correlations of PC1 with activation norm and token frequency in Pythia-1.4B.
<table><tr><td>Tokens included</td><td>Count</td><td>|r|: norm</td><td>95% interval</td><td> $\rho \mathrm { : }$  frequency</td><td>95% interval</td></tr><tr><td>All</td><td>65,535</td><td>0.999808</td><td>[0.999786, 0.999826]</td><td>0.543</td><td>[0.532, 0.553]</td></tr><tr><td>First position</td><td>862</td><td>0.999994</td><td>[0.999993, 0.999994]</td><td>-0.621</td><td>[-0.684, -0.568]</td></tr><tr><td>Other positions</td><td>64,673</td><td>0.999616</td><td>[0.999568, 0.999656]</td><td>0.552</td><td>[0.540, 0.562]</td></tr></table>

## I.2 Predicting model quantities

We fit linear ridge regressions on calibration articles and test on separate articles (Table 46). The targets are next-token entropy and surprisal, the current token’s log training frequency, log token position, and attention to the first token in the next layer, averaged across heads. The first position is excluded from the attention prediction.

For each regression, we standardize the inputs and choose the ridge penalty with five-fold crossvalidation. We try 15 log-spaced penalties from $1 0 ^ { - 3 }$ to $1 0 ^ { 4 }$ for a scalar input, or 19 from $1 0 ^ { - 3 }$ to $1 0 ^ { 6 }$ for a full vector. Scalar results average five split seeds, and vector results average two.

Table 46 gives test $R ^ { 2 }$ for PC1, norm, and full activations with or without PC1 (Raw and $Q \mathbf { x } )$ . Each Random column is the 95th percentile over 32 random unit directions; Random ⊥ restricts these to directions orthogonal to PC1. We also fit regressions using PC2 and log position.

Table 46: Test $R ^ { 2 }$ for predicting model quantities.
<table><tr><td>Model</td><td>Target</td><td>PC1</td><td>Norm</td><td>Random</td><td>Random ⊥</td><td> $Q \mathbf { x }$ </td><td>Raw</td></tr><tr><td>GPT-2</td><td>Entropy</td><td>0.057</td><td>0.052</td><td>0.062</td><td>0.013</td><td>0.597</td><td>0.649</td></tr><tr><td>GPT-2</td><td>Surprisal</td><td>0.019</td><td>0.017</td><td>0.023</td><td>0.005</td><td>0.241</td><td>0.258</td></tr><tr><td>GPT-2</td><td>Frequency</td><td>0.000</td><td>0.000</td><td>0.012</td><td>0.062</td><td>0.938</td><td>0.938</td></tr><tr><td>GPT-2</td><td>Position</td><td>0.170</td><td>0.169</td><td>0.207</td><td>0.078</td><td>0.826</td><td>0.988</td></tr><tr><td>GPT-2</td><td>Sink attention</td><td>0.000</td><td>0.058</td><td>0.040</td><td>0.039</td><td>0.896</td><td>0.896</td></tr><tr><td>Pythia-160M</td><td>Entropy</td><td>0.040</td><td>0.029</td><td>0.064</td><td>0.046</td><td>0.687</td><td>0.697</td></tr><tr><td>Pythia-160M</td><td>Surprisal</td><td>0.024</td><td>0.020</td><td>0.028</td><td>0.010</td><td>0.198</td><td>0.204</td></tr><tr><td>Pythia-160M</td><td>Frequency</td><td>0.002</td><td>0.001</td><td>0.048</td><td>0.108</td><td>0.915</td><td>0.916</td></tr><tr><td>Pythia-160M</td><td>Position</td><td>0.185</td><td>0.172</td><td>0.185</td><td>0.056</td><td>0.882</td><td>0.915</td></tr><tr><td>Pythia-160M</td><td>Sink attention</td><td>-0.004</td><td>-0.004</td><td>0.036</td><td>0.060</td><td>0.735</td><td>0.736</td></tr><tr><td>Pythia-1.4B</td><td>Entropy</td><td>0.048</td><td>0.044</td><td>0.062</td><td>0.035</td><td>0.736</td><td>0.736</td></tr><tr><td>Pythia-1.4B</td><td>Surprisal</td><td>0.046</td><td>0.043</td><td>0.053</td><td>0.015</td><td>0.323</td><td>0.324</td></tr><tr><td>Pythia-1.4B</td><td>Frequency</td><td>0.002</td><td>0.001</td><td>0.048</td><td>0.088</td><td>0.926</td><td>0.926</td></tr><tr><td>Pythia-1.4B</td><td>Position</td><td>0.205</td><td>0.201</td><td>0.207</td><td>0.050</td><td>0.925</td><td>0.926</td></tr><tr><td>Pythia-1.4B</td><td>Sink attention</td><td>0.011</td><td>0.010</td><td>0.031</td><td>0.047</td><td>0.850</td><td>0.850</td></tr><tr><td>Pythia-6.9B</td><td>Entropy</td><td>0.058</td><td>0.055</td><td>0.063</td><td>0.016</td><td>0.761</td><td>0.762</td></tr><tr><td>Pythia-6.9B</td><td>Surprisal</td><td>0.054</td><td>0.052</td><td>0.056</td><td>0.007</td><td>0.341</td><td>0.342</td></tr><tr><td>Pythia-6.9B</td><td>Frequency</td><td>0.002</td><td>0.001</td><td>0.013</td><td>0.025</td><td>0.923</td><td>0.923</td></tr><tr><td>Pythia-6.9B</td><td>Position</td><td>0.208</td><td>0.206</td><td>0.209</td><td>0.022</td><td>0.931</td><td>0.936</td></tr><tr><td>Pythia-6.9B</td><td>Sink attention</td><td>0.016</td><td>0.016</td><td>0.032</td><td>0.023</td><td>0.744</td><td>0.743</td></tr><tr><td>Llama-3.1-8B</td><td>Entropy</td><td>0.024</td><td>0.024</td><td>0.035</td><td>0.034</td><td>0.780</td><td>0.624</td></tr><tr><td>Llama-3.1-8B</td><td>Surprisal</td><td>0.319</td><td>0.318</td><td>0.323</td><td>0.008</td><td>0.192</td><td>0.551</td></tr><tr><td>Llama-3.1-8B</td><td>Frequency</td><td>0.014</td><td>0.013</td><td>0.033</td><td>0.023</td><td>0.919</td><td>0.933</td></tr><tr><td>Llama-3.1-8B</td><td>Position</td><td>0.167</td><td>0.166</td><td>0.185</td><td>0.036</td><td>0.739</td><td>0.949</td></tr><tr><td>Llama-3.1-8B</td><td>Sink attention</td><td>0.009</td><td>0.166</td><td>0.068</td><td>0.068</td><td>0.942</td><td>0.943</td></tr><tr><td>Qwen-2.5-7B</td><td>Entropy</td><td>0.043</td><td>0.042</td><td>0.045</td><td>0.021</td><td>0.677</td><td>0.673</td></tr><tr><td>Qwen-2.5-7B</td><td>Surprisal</td><td>0.061</td><td>0.060</td><td>0.062</td><td>0.013</td><td>0.308</td><td>0.308</td></tr><tr><td>Qwen-2.5-7B</td><td>Frequency</td><td>0.000</td><td>0.000</td><td>0.001</td><td>0.048</td><td>0.916</td><td>0.915</td></tr><tr><td>Model</td><td>Target</td><td>PC1</td><td>Norm</td><td>Random</td><td>Random ⊥</td><td>Qx</td><td>Raw</td></tr><tr><td>Qwen-2.5-7B</td><td>Position</td><td>0.189</td><td>0.190</td><td>0.193</td><td>0.062</td><td>0.948</td><td>0.959</td></tr><tr><td>Qwen-2.5-7B</td><td>Sink attention</td><td>-0.000</td><td>-0.000</td><td>0.001</td><td>0.009</td><td>0.214</td><td>0.193</td></tr><tr><td>Gemma-2-2B</td><td>Entropy</td><td>0.021</td><td>0.020</td><td>0.040</td><td>0.026</td><td>0.681</td><td>0.697</td></tr><tr><td>Gemma-2-2B</td><td>Surprisal</td><td>0.153</td><td>0.153</td><td>0.158</td><td>0.016</td><td>0.235</td><td>0.378</td></tr><tr><td>Gemma-2-2B</td><td>Frequency</td><td>0.005</td><td>0.002</td><td>0.051</td><td>0.081</td><td>0.940</td><td>0.945</td></tr><tr><td>Gemma-2-2B</td><td>Position</td><td>0.191</td><td>0.196</td><td>0.226</td><td>0.071</td><td>0.789</td><td>0.964</td></tr><tr><td>Gemma-2-2B</td><td>Sink attention</td><td>0.011</td><td>0.009</td><td>0.062</td><td>0.062</td><td>0.898</td><td>0.898</td></tr><tr><td>Gemma-2-9B</td><td>Entropy</td><td>0.044</td><td>0.042</td><td>0.045</td><td>0.012</td><td>0.663</td><td>0.698</td></tr><tr><td>Gemma-2-9B</td><td>Surprisal</td><td>0.185</td><td>0.181</td><td>0.145</td><td>0.006</td><td>0.200</td><td>0.370</td></tr><tr><td>Gemma-2-9B</td><td>Frequency</td><td>0.003</td><td>0.000</td><td>0.046</td><td>0.050</td><td>0.938</td><td>0.942</td></tr><tr><td>Gemma-2-9B</td><td>Position</td><td>0.258</td><td>0.256</td><td>0.214</td><td>0.041</td><td>0.675</td><td>0.967</td></tr><tr><td>Gemma-2-9B</td><td>Sink attention</td><td>0.126</td><td>0.123</td><td>0.161</td><td>0.131</td><td>0.913</td><td>0.913</td></tr></table>

## I.3 Removing and restoring PC1

We replace the PC1 coefficient with its calibration mean, h $\mapsto \mathbf { h } - P ( \mathbf { h } - \pmb { \mu } )$ , or with its mean at the same token position, and compare these with removing PC2 or a random direction (Table 47); random results average five directions.

Table 47: Cross-entropy after residual-stream interventions, in nats per prediction.
<table><tr><td>Model</td><td>Unedited</td><td>Delete PC1</td><td>Position mean</td><td>Delete PC2</td><td>Random</td></tr><tr><td>GPT-2</td><td>3.929</td><td>4.728</td><td>3.933</td><td>4.219</td><td>3.938</td></tr><tr><td>Pythia-160M</td><td>3.975</td><td>4.198</td><td>4.054</td><td>3.997</td><td>3.986</td></tr><tr><td>Pythia-1.4B</td><td>3.149</td><td>3.694</td><td>3.292</td><td>3.174</td><td>3.162</td></tr><tr><td>Pythia-6.9B</td><td>2.968</td><td>3.379</td><td>3.271</td><td>3.046</td><td>2.969</td></tr><tr><td>Llama-3.1-8B</td><td>2.600</td><td>4.610</td><td>2.600</td><td>2.747</td><td>2.599</td></tr><tr><td>Qwen-2.5-7B</td><td>2.578</td><td>8.443</td><td>4.437</td><td>2.575</td><td>2.585</td></tr><tr><td>Gemma-2-2B</td><td>3.004</td><td>3.188</td><td>3.007</td><td>2.981</td><td>3.005</td></tr><tr><td>Gemma-2-9B</td><td>2.759</td><td>3.089</td><td>2.761</td><td>2.665</td><td>2.759</td></tr></table>

Table 48 gives loss after removing PC1 from an earlier MLP output (m − Pm). We choose the two earlier MLPs with the largest mean absolute output along the SAE layer’s PC1. Random removes a random direction orthogonal to PC1. Restore later reinstates the original PC1 coefficient at the SAE layer. We also test edits confined to first or later positions and restoration immediately after removal.

Table 48: Cross-entropy after removing PC1 from MLP outputs, in nats per prediction.
<table><tr><td>Model</td><td></td><td>Layer Unedited</td><td>Remove PC1</td><td>Random ⊥</td><td>Restore later</td></tr><tr><td>GPT-2</td><td>2</td><td>3.929</td><td>3.913</td><td>3.930</td><td>3.927</td></tr><tr><td>GPT-2</td><td>0</td><td>3.929</td><td>3.926</td><td>3.929</td><td>3.927</td></tr><tr><td>Pythia-160M</td><td>3</td><td>3.975</td><td>4.746</td><td>3.976</td><td>4.520</td></tr><tr><td>Pythia-160M</td><td>0</td><td>3.975</td><td>4.105</td><td>3.977</td><td>4.101</td></tr><tr><td>Pythia-1.4B</td><td>3</td><td>3.149</td><td>3.806</td><td>3.149</td><td>3.808</td></tr><tr><td>Pythia-1.4B</td><td>4</td><td>3.149</td><td>3.176</td><td>3.149</td><td>3.146</td></tr><tr><td>Pythia-6.9B</td><td>4</td><td>2.969</td><td>3.209</td><td>2.968</td><td>3.028</td></tr><tr><td>Pythia-6.9B</td><td>5</td><td>2.969</td><td>3.121</td><td>2.970</td><td>2.980</td></tr><tr><td>Llama-3.1-8B</td><td>1</td><td>2.600</td><td>10.910</td><td>2.600</td><td>10.819</td></tr><tr><td>Llama-3.1-8B</td><td>10</td><td>2.600</td><td>2.600</td><td>2.600</td><td>2.600</td></tr><tr><td>Qwen-2.5-7B</td><td>3</td><td>2.579</td><td>10.111</td><td>2.580</td><td>10.381</td></tr><tr><td>Qwen-2.5-7B</td><td>4</td><td>2.579</td><td>2.703</td><td>2.580</td><td>2.581</td></tr><tr><td>Gemma-2-2B</td><td>7</td><td>3.004</td><td>2.999</td><td>3.004</td><td>3.004</td></tr><tr><td>Gemma-2-2B</td><td>11</td><td>3.004</td><td>2.994</td><td>3.004</td><td>3.002</td></tr><tr><td>Gemma-2-9B</td><td>0</td><td>2.758</td><td>2.765</td><td>2.757</td><td>2.784</td></tr><tr><td>Gemma-2-9B</td><td>16</td><td>2.758</td><td>2.746</td><td>2.759</td><td>2.761</td></tr></table>

## J Layers, Positions, and Principal-Component Controls

## J.1 Variance across layers

Table 49 reports variance explained by PC1 and PC2 at each Pythia-160M layer, including the output before and after final normalization. We use 65,705 activations from 88 articles, computing model outputs in float32 and centered covariance in float64. Untrained PC1 is the range across three randomly initialized models, with seeds 0, 1, and 2, evaluated on the same inputs.

Table 49: Variance explained (%) across Pythia-160M layers.
<table><tr><td>Activations</td><td>Trained PC1</td><td>Trained PC2</td><td>Untrained PC1</td></tr><tr><td>Embedding</td><td>5.02</td><td>4.42</td><td>4.49-4.88</td></tr><tr><td>Layer 0</td><td>9.82</td><td>4.24</td><td>4.35-4.64</td></tr><tr><td>Layer 1</td><td>21.02</td><td>4.66</td><td>4.23-4.89</td></tr><tr><td>Layer 2</td><td>59.39</td><td>2.53</td><td>3.90-5.04</td></tr><tr><td>Layer 3</td><td>92.70</td><td>0.81</td><td>3.84-4.93</td></tr><tr><td>Layer 4</td><td>93.28</td><td>0.74</td><td>3.81-4.55</td></tr><tr><td>Layer 5</td><td>92.80</td><td>0.79</td><td>3.73-4.39</td></tr><tr><td>Layer 6</td><td>92.40</td><td>0.69</td><td>3.82-4.42</td></tr><tr><td>Layer 7</td><td>90.75</td><td>0.90</td><td>4.06–4.41</td></tr><tr><td>Layer 8</td><td>87.33</td><td>1.15</td><td>4.11-4.52</td></tr><tr><td>Layer 9</td><td>78.05</td><td>2.03</td><td>4.06-4.57</td></tr><tr><td>Layer 10</td><td>45.39</td><td>7.97</td><td>4.01-4.53</td></tr><tr><td>Layer 11</td><td>64.01</td><td>1.90</td><td>4.05-4.41</td></tr><tr><td>Final normalization</td><td>5.97</td><td>2.19</td><td>3.88-4.07</td></tr></table>

## J.2 SAEs at additional layers

Table 50 gives standard/bypass sharing and sparsity at two additional layers per model. We use five seeds and the default training settings, with 1,500 features ranked on 64 validation sequences. Sharing averages all 20 ordered seed pairs.

Table 50: Feature sharing (%) and sparsity at additional layers.
<table><tr><td>Model</td><td>Layer</td><td>Top 50</td><td>Dictionary</td><td> $L _ { 0 }$ </td></tr><tr><td>GPT-2</td><td>0</td><td>53.7/50.7</td><td>6.2/6.0</td><td>63.94/64.00</td></tr><tr><td>GPT-2</td><td>11</td><td>94.0/95.1</td><td>58.4/58.5</td><td>64.00/64.00</td></tr><tr><td>Pythia-160M</td><td>1</td><td>48.6/50.0</td><td>7.4/7.3</td><td>64.00/64.00</td></tr><tr><td>Pythia-160M</td><td>9</td><td>18.4/93.1</td><td>54.7/53.4</td><td>63.98/64.00</td></tr><tr><td>Pythia-1.4B</td><td>2</td><td>55.8/54.9</td><td>15.2/14.6</td><td>63.98/64.00</td></tr><tr><td>Pythia-1.4B</td><td>23</td><td>87.0/90.8</td><td>59.3/59.2</td><td>63.90/63.98</td></tr><tr><td>Pythia-6.9B</td><td>2</td><td>58.4/57.9</td><td>20.2/17.5</td><td>63.84/64.00</td></tr><tr><td>Pythia-6.9B</td><td>31</td><td>91.8/92.6</td><td>60.1/58.8</td><td>63.65/63.77</td></tr><tr><td>Llama-3.1-8B</td><td>7</td><td>67.8/96.4</td><td>64.5/65.1</td><td>63.68/63.27</td></tr><tr><td>Llama-3.1-8B</td><td>31</td><td>92.1/93.4</td><td>64.5/65.2</td><td>63.38/63.81</td></tr><tr><td>Qwen-2.5-7B</td><td>2</td><td>78.0/79.1</td><td>36.1/34.1</td><td>63.85/64.00</td></tr><tr><td>Qwen-2.5-7B</td><td>27</td><td>90.4/95.3</td><td>68.3/68.5</td><td>63.76/64.00</td></tr><tr><td>Gemma-2-2B</td><td>2</td><td>58.3/57.4</td><td>16.0/15.8</td><td>64.00/63.99</td></tr><tr><td>Gemma-2-2B</td><td>22</td><td>92.3/94.1</td><td>67.5/70.4</td><td>64.00/63.60</td></tr><tr><td>Gemma-2-9B</td><td>9</td><td>94.0/95.0</td><td>70.3/73.8</td><td>63.96/63.39</td></tr><tr><td>Gemma-2-9B</td><td>33</td><td>83.4/94.9</td><td>70.7/74.3</td><td>64.00/63.76</td></tr></table>

Table 51 gives top-50 sharing $( S _ { 5 0 } ) ,$ dictionary sharing, and loss with SAE reconstructions (CE) for a separate Pythia-1.4B experiment. We use layers 3, 12, and 23, before final normalization. Here the data are wikitext-103-raw-v1 entries joined with newlines and split into 128-token sequences. We use seeds 42 and 137 and the default 8,192-feature training settings (Table 3). Model outputs are float32. PCA is fitted on a random sample of 50,000 training activations; feature rankings use 64 validation sequences.

Table 51: Pythia-1.4B feature sharing (%) and cross-entropy across layers.
<table><tr><td>Layer w1 (%)</td><td></td><td> $S _ { 5 0 } \ : S$ </td><td> $S _ { 5 0 }$  B</td><td>Dict. S</td><td>Dict. B</td><td>CE S</td><td>CE B</td></tr><tr><td>3</td><td>93.82</td><td>14.0</td><td>68.0</td><td>30.0</td><td>23.0</td><td>3.25249</td><td>3.24169</td></tr><tr><td>12</td><td>92.95</td><td>8.0</td><td>92.0</td><td>63.2</td><td>62.4</td><td>3.22981</td><td>3.22407</td></tr><tr><td>23</td><td>11.09</td><td>96.0</td><td>92.0</td><td>64.6</td><td>63.7</td><td>3.67123</td><td>3.66968</td></tr></table>

## J.3 Removing additional principal components

Table 52 gives top-50 sharing after removing leading PCs from Pythia-160M layer-6 activations. We use the settings in Table 3 with 4,096 features, one million training tokens, float32 model outputs, and seeds 42 and 137. The first 50,000 activations are used for $\mathrm { P } \bar { \mathrm { C } } \mathbf { A } ;$ 64 validation sequences are used for sharing.

Table 52: Pythia-160M top-50 sharing after removing leading principal components.
<table><tr><td>PCs removed 0</td><td>1 2</td><td>2</td><td></td><td>5 10 20 50</td><td></td></tr><tr><td>Sharing (%)</td><td>4</td><td>86 90</td><td></td><td>84 80</td><td>74 84</td></tr></table>

## J.4 Excluding first positions and special tokens

Table 53 gives PC1 variance after refitting on different sets of tokens, from up to 128 validation sequences. After first omits array position zero; No special also omits tokenizer special tokens. Counts exclude padding and are taken before the other exclusions. No added BOS is a separate model run with automatic special-token addition disabled.

Table 53: PC1 variance share (%) with different token exclusions and tokenization settings.
<table><tr><td>Model</td><td>Context</td><td>Tokenization</td><td>Tokens</td><td>All</td><td></td><td>After first No special</td></tr><tr><td>GPT-2</td><td>128</td><td>Default</td><td>9811</td><td>96.66</td><td>3.89</td><td>3.89</td></tr><tr><td>GPT-2</td><td>512</td><td>Default</td><td>12831</td><td>95.56</td><td>3.94</td><td>3.94</td></tr><tr><td>Pythia-160M</td><td>128</td><td>Default</td><td>9821</td><td>92.41</td><td>88.16</td><td>88.16</td></tr><tr><td>Pythia-160M</td><td>512</td><td>Default</td><td>13103</td><td>91.06</td><td>87.06</td><td>87.06</td></tr><tr><td>Pythia-1.4B</td><td>128</td><td>Default</td><td>9821</td><td>95.65</td><td>91.60</td><td>91.60</td></tr><tr><td>Pythia-1.4B</td><td>512</td><td>Default</td><td>13103</td><td>95.14</td><td>92.01</td><td>92.01</td></tr><tr><td>Llama-3.1-8B</td><td>128</td><td>Default</td><td>10015</td><td>96.59</td><td>2.85</td><td>2.85</td></tr><tr><td>Llama-3.1-8B</td><td>128</td><td>No added BOS</td><td>9936</td><td>98.24</td><td>91.25</td><td>91.25</td></tr><tr><td>Llama-3.1-8B</td><td>512</td><td>Default</td><td>13597</td><td>95.45</td><td>2.64</td><td>2.64</td></tr><tr><td>Llama-3.1-8B</td><td>512</td><td>No added BOS</td><td>13469</td><td>97.65</td><td>88.52</td><td>88.52</td></tr><tr><td>Qwen-2.5-7B</td><td>128</td><td>Default</td><td>10093</td><td>99.87</td><td>99.68</td><td>99.68</td></tr><tr><td>Qwen-2.5-7B</td><td>512</td><td>Default</td><td>14160</td><td>99.83</td><td>99.56</td><td>99.56</td></tr><tr><td>Gemma-2-2B</td><td>128</td><td>Default</td><td>10111</td><td>76.52</td><td>66.73</td><td>14.87</td></tr><tr><td>Gemma-2-2B</td><td>128</td><td>No added BOS</td><td>10033</td><td>8.79</td><td>8.69</td><td>8.72</td></tr><tr><td>Gemma-2-2B</td><td>512</td><td>Default</td><td>13810</td><td>70.79</td><td>70.79</td><td>11.84</td></tr><tr><td>Gemma-2-2B</td><td>512</td><td>No added BOS</td><td>13682</td><td>8.45</td><td>8.45</td><td>8.48</td></tr></table>

## K Released SAEs and Preprocessing

## K.1 Released SAEs

We evaluate released SAEs on WikiText-103, using 50,000 activations for calibration and up to 100,000 further activations for evaluation, with 128-token contexts.

Gemma Scope, Gemma Scope 2, and Llama Scope (Lieberum et al., 2024; McDougall et al., 2025; He et al., 2024) exclude start, end, and padding tokens. We use their released weights directly. For OpenAI GPT-2 (Gao et al., 2025), we subtract the coordinate mean, normalize the vector, and subtract decoder bias before encoding; FVU uses the normalized inputs.

Table 54 gives total and content FVU for 44 checkpoints. Layer and Tag are the release labels. $L _ { 0 }$ is measured on our data; 1k = 1024 and $R = \mathrm { F V U \bar { \Omega } _ { \perp } / F V U _ { \mathrm { t o t a l } } } .$

Table 54: Total and content FVU for released SAEs.
<table><tr><td>Model</td><td>Layer</td><td>Width</td><td> $\mathrm { T a g }$ </td><td> $L _ { 0 }$ </td><td>Total FVU</td><td>Content FVU</td><td>R</td></tr><tr><td colspan="8">Gemma Scope</td></tr><tr><td>Gemma-2-2B</td><td>6</td><td>16k 36</td><td></td><td>38.3</td><td>0.22550</td><td>0.23807</td><td>1.056</td></tr><tr><td>Gemma-2-2B</td><td>6</td><td></td><td>16k 70</td><td>76.1</td><td>0.18626</td><td>0.19670</td><td>1.056</td></tr><tr><td>Gemma-2-2B</td><td>6</td><td></td><td>64k 107</td><td>118.4</td><td>0.12121</td><td>0.12802</td><td>1.056</td></tr><tr><td>Gemma-2-2B</td><td>6</td><td></td><td>64k 56</td><td>60.9</td><td>0.15026</td><td>0.15866</td><td>1.056</td></tr><tr><td>Gemma-2-2B</td><td>12</td><td></td><td>16k 41</td><td>42.0</td><td>0.24474</td><td>0.30489</td><td>1.246</td></tr><tr><td>Gemma-2-2B</td><td>12</td><td></td><td>16k 82</td><td>81.5</td><td>0.20612</td><td>0.25682</td><td>1.246</td></tr><tr><td>Gemma-2-2B</td><td>12</td><td></td><td>64k 38</td><td>38.0</td><td>0.21258</td><td>0.26483</td><td>1.246</td></tr><tr><td>Gemma-2-2B</td><td>12</td><td></td><td>64k 72</td><td>73.6</td><td>0.17669</td><td>0.22015</td><td>1.246</td></tr><tr><td>Gemma-2-2B</td><td>22</td><td></td><td>16k 38</td><td>38.4</td><td>0.28973</td><td>0.30137</td><td>1.040</td></tr><tr><td>Gemma-2-2B</td><td>22</td><td></td><td>16k 72</td><td>72.4</td><td>0.24844</td><td>0.25852</td><td>1.041</td></tr><tr><td>Gemma-2-2B</td><td>22</td><td>64k</td><td>116</td><td>121.2</td><td>0.17346</td><td>0.18051</td><td>1.041</td></tr><tr><td>Gemma-2-2B</td><td>22</td><td>64k 62</td><td></td><td>63.6</td><td>0.20709</td><td>0.21548</td><td>1.041</td></tr><tr><td>Gemma-2-9B</td><td>9</td><td>16k 100</td><td></td><td>113.9</td><td>0.22771</td><td>0.23644</td><td>1.038</td></tr><tr><td>Gemma-2-9B</td><td>9</td><td>16k 51</td><td></td><td>57.7</td><td>0.26426</td><td>0.27425</td><td>1.038</td></tr><tr><td>Gemma-2-9B</td><td>9</td><td>128k</td><td>42</td><td>49.4</td><td>0.17345</td><td>0.18003</td><td>1.038</td></tr><tr><td>Gemma-2-9B</td><td>9</td><td>128k 75</td><td></td><td>88.8</td><td>0.14682</td><td>0.15246</td><td>1.038</td></tr><tr><td>Gemma-2-9B</td><td>21</td><td>16k</td><td>129</td><td>127.2</td><td>0.18127</td><td>0.28192</td><td>1.555</td></tr><tr><td>Gemma-2-9B</td><td>21</td><td>16k 36</td><td></td><td>35.1</td><td>0.24853</td><td>0.38642</td><td>1.555</td></tr><tr><td>Gemma-2-9B</td><td>21</td><td>128k</td><td>109</td><td>110.0</td><td>0.14596</td><td>0.22699</td><td>1.555</td></tr><tr><td>Gemma-2-9B</td><td>21</td><td>128k 58</td><td></td><td>57.8</td><td>0.17516</td><td>0.27237</td><td>1.555</td></tr><tr><td>Gemma-2-9B</td><td>35</td><td>16k</td><td>120</td><td>121.9</td><td>0.25059</td><td>0.26311</td><td>1.050</td></tr><tr><td>Gemma-2-9B</td><td>35</td><td>16k 61</td><td></td><td>60.5</td><td>0.29364</td><td>0.30825</td><td>1.050</td></tr><tr><td>Gemma-2-9B</td><td>35</td><td>128k 51</td><td></td><td>51.1</td><td>0.22114</td><td>0.23212</td><td>1.050</td></tr><tr><td>Gemma-2-9B</td><td>35</td><td>128k 94</td><td></td><td>95.2</td><td>0.18734</td><td>0.19668</td><td>1.050</td></tr><tr><td colspan="8">Gemma Scope 2</td></tr><tr><td>Gemma-3-1B</td><td>7</td><td>16k</td><td>medium</td><td>58.2</td><td>0.06124</td><td>0.12955</td><td>2.116</td></tr><tr><td>Gemma-3-1B</td><td>7</td><td>16k</td><td>small</td><td>20.2</td><td>0.07998</td><td>0.16794</td><td>2.100</td></tr><tr><td>Gemma-3-1B</td><td>7</td><td>64k</td><td>medium</td><td>57.3</td><td>0.05039</td><td>0.10590</td><td>2.102</td></tr><tr><td>Gemma-3-1B</td><td>7</td><td>64k</td><td>small</td><td>20.1</td><td>0.06831</td><td>0.14159</td><td>2.073</td></tr><tr><td>Gemma-3-1B</td><td>13</td><td>16k</td><td>medium</td><td>61.2</td><td>0.12413</td><td>0.27350</td><td>2.203</td></tr><tr><td>Gemma-3-1B</td><td>13</td><td>16k</td><td>small</td><td>21.5</td><td>0.16630</td><td>0.36211</td><td>2.178</td></tr><tr><td>Gemma-3-1B</td><td>13</td><td>64k</td><td>medium</td><td>60.4</td><td>0.11313</td><td>0.24833</td><td>2.195</td></tr><tr><td>Gemma-3-1B</td><td>13</td><td>64k</td><td>small</td><td>21.2</td><td>0.15443</td><td>0.33454</td><td>2.166</td></tr><tr><td>Gemma-3-1B</td><td>22</td><td>16k</td><td>medium</td><td>61.9</td><td>0.20049</td><td>0.27266</td><td>1.360</td></tr><tr><td>Gemma-3-1B</td><td>22</td><td>16k</td><td>small</td><td>23.6</td><td>0.25709</td><td>0.34737</td><td>1.351</td></tr><tr><td>Gemma-3-1B</td><td>22</td><td>64k</td><td>medium</td><td>61.6</td><td>0.17096</td><td>0.23173</td><td>1.355</td></tr><tr><td>Gemma-3-1B</td><td>22</td><td>64k</td><td>small</td><td>23.5</td><td>0.22078</td><td>0.29734</td><td>1.347</td></tr><tr><td colspan="8">Llama Scope</td></tr><tr><td>Llama-3.1-8B</td><td>6</td><td>32k</td><td></td><td>31.0</td><td>0.40442</td><td>0.41059</td><td>1.015</td></tr><tr><td>Llama-3.1-8B</td><td>6</td><td>128k</td><td></td><td>35.3</td><td>0.33362</td><td>0.33949</td><td>1.018</td></tr><tr><td>Llama-3.1-8B</td><td>16</td><td>32k</td><td></td><td>32.0</td><td>0.37068</td><td>0.37834</td><td>1.021</td></tr><tr><td>Llama-3.1-8B</td><td>16</td><td>128k</td><td></td><td>35.1</td><td>0.31843</td><td>0.32480</td><td>1.020</td></tr><tr><td>Llama-3.1-8B</td><td>27</td><td>32k</td><td></td><td>29.4</td><td>0.41986</td><td>0.44304</td><td>1.055</td></tr><tr><td>Llama-3.1-8B</td><td>27</td><td>128k</td><td></td><td>32.2</td><td>0.35982</td><td>0.37968</td><td>1.055</td></tr><tr><td colspan="8">OpenAI GPT-2</td></tr><tr><td>GPT-2</td><td>8</td><td>32k</td><td></td><td>32.0</td><td>0.17666</td><td>0.18238</td><td>1.032</td></tr><tr><td>GPT-2</td><td>8</td><td>128k</td><td></td><td>32.0</td><td>0.14626</td><td>0.15093</td><td>1.032</td></tr></table>

Qwen Scope. Table 55 gives reconstruction results for six Qwen Scope SAEs on Qwen3-8B-Base (Deng et al., 2026). Qwen Scope filters large-norm activations, but the exact threshold is not specified. We therefore try 5, 10, and 20 times the median raw norm in the calibration data, filtering both calibration and test activations. All SAEs have 65,536 features; k is the release’s sparsity label, and Removed is the percentage of evaluation activations excluded.

Table 55: Qwen Scope reconstruction after excluding high-norm activations.
<table><tr><td>Layer</td><td>k</td><td>Cutoff</td><td>Removed (%)</td><td> $w _ { 1 }$  (%)</td><td>Total FVU</td><td>Content FVU</td><td>R</td></tr><tr><td>7</td><td>50</td><td>All</td><td>0.00</td><td>99.71</td><td>1.02107</td><td>178.42778</td><td>174.746</td></tr><tr><td>7</td><td>50</td><td>5×</td><td>0.80</td><td>2.91</td><td>0.24005</td><td>0.24709</td><td>1.029</td></tr><tr><td>7</td><td>50</td><td>10×</td><td>0.80</td><td>2.91</td><td>0.24005</td><td>0.24709</td><td>1.029</td></tr><tr><td>Layer</td><td></td><td>k Cutoff</td><td>Removed (%)</td><td>w1 (%)</td><td>Total FVU</td><td>Content FVU</td><td>R</td></tr><tr><td>7</td><td>50</td><td>20×</td><td>0.80</td><td>2.91</td><td>0.24005</td><td>0.24709</td><td>1.029</td></tr><tr><td></td><td>7 100</td><td>All</td><td>0.00</td><td>99.71</td><td>1.10307</td><td>242.81489</td><td>220.125</td></tr><tr><td></td><td>7 100</td><td>5×</td><td>0.80</td><td>2.91</td><td>0.20789</td><td>0.21402</td><td>1.029</td></tr><tr><td></td><td>7 100</td><td>10×</td><td>0.80</td><td>2.91</td><td>0.20789</td><td>0.21402</td><td>1.029</td></tr><tr><td>7</td><td>100</td><td>20×</td><td>0.80</td><td>2.91</td><td>0.20789</td><td>0.21402</td><td>1.029</td></tr><tr><td>18</td><td>50</td><td>All</td><td>0.00</td><td>99.10</td><td>0.22524</td><td>20.76911</td><td>92.209</td></tr><tr><td>18</td><td>50</td><td>5×</td><td>0.82</td><td>2.54</td><td>0.32881</td><td>0.33690</td><td>1.025</td></tr><tr><td>18</td><td>50</td><td>10×</td><td>0.82</td><td>2.54</td><td>0.32881</td><td>0.33690</td><td>1.025</td></tr><tr><td>18</td><td>50</td><td>20×</td><td>0.82</td><td>2.54</td><td>0.32881</td><td>0.33690</td><td>1.025</td></tr><tr><td>18</td><td>100</td><td>All</td><td>0.00</td><td>99.10</td><td>0.14942</td><td>15.73571</td><td>105.313</td></tr><tr><td>18</td><td>100</td><td>5×</td><td>0.82</td><td>2.54</td><td>0.27656</td><td>0.28349</td><td>1.025</td></tr><tr><td>18</td><td>100</td><td>10×</td><td>0.82</td><td>2.54</td><td>0.27656</td><td>0.28349</td><td>1.025</td></tr><tr><td>18</td><td>100</td><td>20×</td><td>0.82</td><td>2.54</td><td>0.27656</td><td>0.28349</td><td>1.025</td></tr><tr><td>30</td><td>50</td><td>All</td><td>0.00</td><td>74.97</td><td>0.07401</td><td>0.29520</td><td>3.988</td></tr><tr><td>30</td><td>50</td><td>5×</td><td>0.82</td><td>7.91</td><td>0.28395</td><td>0.30821</td><td>1.085</td></tr><tr><td>30</td><td>50</td><td>10×</td><td>0.82</td><td>7.99</td><td>0.28308</td><td>0.30754</td><td>1.086</td></tr><tr><td>30</td><td>50</td><td>20×</td><td>0.00</td><td>74.94</td><td>0.07414</td><td>0.29527</td><td>3.982</td></tr><tr><td>30</td><td>100</td><td>All</td><td>0.00</td><td>74.97</td><td>0.06269</td><td>0.25042</td><td>3.994</td></tr><tr><td>30</td><td>100</td><td>5×</td><td>0.82</td><td>7.91</td><td>0.24174</td><td>0.26243</td><td>1.086</td></tr><tr><td>30</td><td>100</td><td>10×</td><td>0.82</td><td>7.99</td><td>0.24100</td><td>0.26186</td><td>1.087</td></tr><tr><td>30</td><td>100</td><td>20×</td><td>0.00</td><td>74.94</td><td>0.06280</td><td>0.25048</td><td>3.988</td></tr></table>

Pythia and context length. Table 56 evaluates the released Pythia-160M layer-6 SAE, which has 32,768 features and k = 32 and subtracts decoder bias before encoding. We use separate 128- or 512-token chunks without added BOS, or 2,049-token contexts formed by joining text with end-ofsequence (EOS) separators. After first excludes the first position from evaluation; Exclude EOS omits the separators.

Table 56: Released Pythia-160M SAE reconstruction at different context lengths.
<table><tr><td>Context</td><td>Tokens included</td><td>Count</td><td>w1 (%)</td><td>Total FVU</td><td>Content FVU</td><td>R</td></tr><tr><td>128</td><td>After first</td><td>99,211</td><td>80.04</td><td>0.04049</td><td>0.20275</td><td>5.007</td></tr><tr><td>128</td><td>All</td><td>100,000</td><td>87.61</td><td>0.02407</td><td>0.19416</td><td>8.067</td></tr><tr><td>512</td><td>After first</td><td>99,795</td><td>54.99</td><td>0.08597</td><td>0.19084</td><td>2.220</td></tr><tr><td>512</td><td>All</td><td>100,000</td><td>66.63</td><td>0.06303</td><td>0.18873</td><td>2.994</td></tr><tr><td>2049</td><td>All packed</td><td>100,000</td><td>33.03</td><td>0.11928</td><td>0.17796</td><td>1.492</td></tr><tr><td>2049</td><td>Exclude EOS</td><td>99,978</td><td>31.59</td><td>0.12201</td><td>0.17819</td><td>1.460</td></tr></table>

## K.2 Normalization, filtering, and whitening

We compare preprocessing methods with the default training settings and seeds 42, 137, and 0. We fit preprocessing parameters on the first 200,000 training activations and evaluate on 64 validation sequences. Before measuring reconstruction error or model loss, we undo the preprocessing.

Unit-norm preprocessing divides each centered activation by its norm and saves the norm to rescale the reconstruction. Coordinate normalization also subtracts the mean across coordinates, saving both that mean and the norm. Norm filtering removes training activations whose centered norm exceeds ten times the median training norm.

Whitening uses the training covariance $V \mathrm { d i a g } ( \lambda _ { i } ) V ^ { \top }$

$$
\begin{array} { r } { \mathbf { y } = ( \mathbf { x } - \pmb { \mu } ) V \operatorname { d i a g } ( ( \lambda _ { i } + \epsilon ) ^ { - 1 / 2 } ) , \qquad \epsilon = 1 0 ^ { - 4 } \lambda _ { 1 } . } \end{array}
$$

We invert this transform after reconstruction.

Tables 57 and 58 give reconstruction results on all tokens and on tokens below the norm cutoff, respectively, averaged over three seeds. Coefficients counts sparse features plus the extra values needed to undo preprocessing. CE is loss with SAE reconstructions.

Table 57: Reconstruction after preprocessing, evaluated on all tokens.
<table><tr><td>Model</td><td>Method</td><td>Coefficients</td><td> $L _ { 0 }$ </td><td>Total FVU</td><td>Content FVU</td><td>CE</td></tr><tr><td>GPT-2</td><td>Standard</td><td>64+0</td><td>64.00</td><td>0.004348</td><td>0.13650</td><td>4.11441</td></tr><tr><td>GPT-2</td><td>Bypass 63+1</td><td>63+1</td><td>62.77</td><td>0.004276</td><td>0.13427</td><td>4.11617</td></tr><tr><td>GPT-2</td><td>Bypass 64+1</td><td>64+1</td><td>63.77</td><td>0.004243</td><td>0.13323</td><td>4.11435</td></tr><tr><td>GPT-2</td><td>Unit norm</td><td>64+1</td><td>63.98</td><td>0.004398</td><td>0.13799</td><td>4.12948</td></tr><tr><td>GPT-2</td><td>Coord. norm</td><td>64+2</td><td>64.00</td><td>0.004405</td><td>0.13822</td><td>4.12936</td></tr><tr><td>GPT-2</td><td>Coord. norm+bypass</td><td>63+3</td><td>62.51</td><td>0.004665</td><td>0.14650</td><td>4.16077</td></tr><tr><td>GPT-2</td><td>Norm-filter train</td><td>64+0</td><td>64.00</td><td>3.533275</td><td>76.87259</td><td>5.31978</td></tr><tr><td>GPT-2</td><td>Whiten</td><td> $6 4 + 0$ </td><td>64.00</td><td>0.004921</td><td>0.13433</td><td>4.13626</td></tr><tr><td>Pythia-1.4B</td><td>Standard</td><td>64+0</td><td>63.98</td><td>0.009074</td><td>0.22139</td><td>3.19588</td></tr><tr><td>Pythia-1.4B</td><td>Bypass 63+1</td><td>63+1</td><td>63.00</td><td>0.008965</td><td>0.21876</td><td>3.18367</td></tr><tr><td>Pythia-1.4B</td><td>Bypass 64+1</td><td>64+1</td><td>64.00</td><td>0.008936</td><td>0.21805</td><td>3.18461</td></tr><tr><td>Pythia-1.4B</td><td>Unit norm</td><td>64+1</td><td>64.00</td><td>0.009084</td><td>0.22153</td><td>3.19334</td></tr><tr><td>Pythia-1.4B</td><td>Coord. norm</td><td>64+2</td><td>64.00</td><td>0.009091</td><td>0.22169</td><td>3.20145</td></tr><tr><td>Pythia-1.4B</td><td>Coord. norm+bypass</td><td>63+3</td><td>61.82</td><td>0.009115</td><td>0.22240</td><td>3.21050</td></tr><tr><td>Pythia-1.4B</td><td>Norm-filter train</td><td>64+0</td><td>64.00</td><td>0.079522</td><td>1.85201</td><td>3.50372</td></tr><tr><td>Pythia-1.4B</td><td>Whiten</td><td>64+0</td><td>64.00</td><td>0.009515</td><td>0.21826</td><td>3.20243</td></tr><tr><td>Qwen-2.5-7B</td><td>Standard</td><td>64+0</td><td>63.98</td><td>0.000398</td><td>0.32329</td><td>2.50462</td></tr><tr><td>Qwen-2.5-7B</td><td>Bypass 63+1</td><td>63+1</td><td>62.95</td><td>0.000248</td><td>0.20184</td><td>2.46515</td></tr><tr><td>Qwen-2.5-7B</td><td>Bypass 64+1</td><td>64+1</td><td>63.95</td><td>0.000247</td><td>0.20109</td><td>2.46161</td></tr><tr><td>Qwen-2.5-7B</td><td>Coord. norm</td><td>64+2</td><td>63.93</td><td>0.000586</td><td>0.47437</td><td>2.65635</td></tr><tr><td>Qwen-2.5-7B</td><td>Norm-filter train</td><td>64+0</td><td>63.97</td><td>0.315440</td><td>253.28876</td><td>2.86624</td></tr><tr><td>Qwen-2.5-7B</td><td>Whiten</td><td> $6 4 + 0$ </td><td>63.99</td><td>0.000962</td><td>0.19604</td><td>2.46488</td></tr><tr><td>Gemma-2-2B</td><td>Standard</td><td>64+0</td><td>63.90</td><td>0.043965</td><td>0.19754</td><td>3.04176</td></tr><tr><td>Gemma-2-2B</td><td>Bypass 63+1</td><td> $^ { 6 3 + 1 }$ </td><td>62.82</td><td>0.043429</td><td>0.19522</td><td>3.03572</td></tr><tr><td>Gemma-2-2B</td><td>Bypass 64+1</td><td> $^ { 6 4 + 1 }$ </td><td>63.77</td><td>0.043272</td><td>0.19451</td><td>3.03512</td></tr><tr><td>Gemma-2-2B</td><td>Coord. norm</td><td> $^ { 6 4 + 2 }$ </td><td>63.82</td><td>0.043401</td><td>0.19503</td><td>3.03799</td></tr><tr><td>Gemma-2-2B</td><td>Norm-filter train</td><td> $6 4 + 0$ </td><td>63.90</td><td>6.437689</td><td>24.27273</td><td>3.76865</td></tr><tr><td>Gemma-2-2B</td><td>Whiten</td><td>64+0</td><td>64.00</td><td>0.049051</td><td>0.19986</td><td>3.04060</td></tr></table>

Table 58: Reconstruction evaluated on tokens below the norm cutoff.
<table><tr><td>Model</td><td>Method</td><td>Tokens</td><td>Total FVU</td><td>Content FVU</td></tr><tr><td>GPT-2</td><td>Standard</td><td>4545</td><td>0.094406</td><td>0.13591</td></tr><tr><td>GPT-2</td><td>Bypass 63+1</td><td>4545</td><td>0.093174</td><td>0.13416</td></tr><tr><td>GPT-2</td><td>Norm-filter train</td><td>4545</td><td>0.092957</td><td>0.13384</td></tr><tr><td>GPT-2</td><td>Whiten</td><td>4545</td><td>0.106670</td><td>0.13429</td></tr><tr><td>Pythia-1.4B</td><td>Standard</td><td>4461</td><td>0.128739</td><td>0.22867</td></tr><tr><td>Pythia-1.4B</td><td>Bypass 63+1</td><td>4461</td><td>0.127931</td><td>0.22725</td></tr><tr><td>Pythia-1.4B</td><td>Norm-filter train</td><td>4461</td><td>0.128220</td><td>0.22774</td></tr><tr><td>Pythia-1.4B</td><td>Whiten</td><td>4461</td><td>0.132677</td><td>0.22671</td></tr><tr><td>Qwen-2.5-7B</td><td>Standard</td><td>4593</td><td>0.013540</td><td>0.27829</td></tr><tr><td>Qwen-2.5-7B</td><td>Bypass 63+1</td><td>4593</td><td>0.011463</td><td>0.23562</td></tr><tr><td>Qwen-2.5-7B</td><td>Norm-filter train</td><td>4593</td><td>0.016704</td><td>0.34329</td></tr><tr><td>Qwen-2.5-7B</td><td>Whiten</td><td>4593</td><td>0.011431</td><td>0.22822</td></tr><tr><td>Gemma-2-2B</td><td>Standard</td><td>4624</td><td>0.173933</td><td>0.19784</td></tr><tr><td>Gemma-2-2B</td><td>Bypass 63+1</td><td>4624</td><td>0.171828</td><td>0.19553</td></tr><tr><td>Gemma-2-2B</td><td>Norm-filter train</td><td>4624</td><td>0.171564</td><td>0.19516</td></tr><tr><td>Gemma-2-2B</td><td>Whiten</td><td>4624</td><td>0.194046</td><td>0.20018</td></tr></table>

## L Vision Transformers

## L.1 Models and SAE training

We use DINOv2-base and DINOv2-with-registers-base (Oquab et al., 2023; Darcet et al., 2024), with 512 training and 128 test images drawn from CIFAR-100’s training split. We take the 16 × 16 patch activations after layer 8, before final normalization, omitting CLS and register tokens.

We train BatchTopK SAEs with 6,144 features and mean $L _ { 0 } = 3 2 .$ , using the settings in Table 3 for 10,000 steps. Here we normalize decoder vectors only at initialization. The BatchTopK cutoff is

recalculated for each batch, including test batches, and tied values are retained. We use float32 with seeds 0, 1, and 2. Random bypass uses seed 4242. Both bypasses have one extra coefficient.

## L.2 Registers

Following Jiang et al. (2025), we redirect two or 50 MLP neurons to an added register token. We run 32 training images through the model, keep patches with norms above ten times the median, and rank the neurons in layers 0–8 by their activity on these patches. The register starts at zero. At each layer we find the largest activation in absolute value among the selected neurons over the original token positions, and copy it, including its sign, to the selected register coordinates. The pretrained comparison uses DINOv2 with four registers.

## L.3 Reconstruction and feature matching

Table 59 reports reconstruction for each model and register setting. FVU is the mean $\pm$ standard deviation across SAE seeds. PC1 variance $w _ { 1 }$ and the maximum-to-median norm ratio are measured on training patches. R is mean content FVU divided by mean total FVU.

Table 59: PC1 variance, activation norms, and reconstruction in vision models.
<table><tr><td>Condition</td><td></td><td>w1 (%) Max/median</td><td>Total FVU</td><td>Content FVU</td><td>R</td></tr><tr><td>Standard</td><td>96.19</td><td>57.24</td><td> $0 . 0 1 3 1 0 \pm 0 . 0 0 0 0 9$ </td><td> $0 . 3 4 3 4 9 \pm 0 . 0 0 2 3 6$ </td><td>26.21</td></tr><tr><td>Random bypass</td><td>96.19</td><td>57.24</td><td> $0 . 0 1 3 1 2 \pm 0 . 0 0 0 0 9$ </td><td> $0 . 3 4 3 9 2 \pm 0 . 0 0 2 2 4$ </td><td>26.21</td></tr><tr><td>PC1 bypass</td><td>96.19</td><td>57.24</td><td> $0 . 0 1 2 6 8 \pm 0 . 0 0 0 0 4$ </td><td> $0 . 3 3 2 7 7 \pm 0 . 0 0 0 9 6$ </td><td>26.24</td></tr><tr><td>Test-time, 50 neurons</td><td>3.96</td><td>2.32</td><td> $0 . 4 1 4 7 3 \pm 0 . 0 0 1 7 2$ </td><td> $0 . 4 2 8 7 9 \pm 0 . 0 0 1 6 3$ </td><td>1.03</td></tr><tr><td>Test-time, 2 neurons</td><td>75.66</td><td>18.57</td><td> $0 . 0 8 4 8 0 \pm 0 . 0 0 0 2 9$ </td><td> $0 . 3 4 9 3 1 \pm 0 . 0 0 1 2 5$ </td><td>4.12</td></tr><tr><td>Pretrained registers</td><td>4.00</td><td>1.71</td><td> $0 . 4 2 7 5 1 \pm 0 . 0 0 0 3 5$ </td><td> $0 . 4 4 1 3 5 \pm 0 . 0 0 0 1 5$ </td><td>1.03</td></tr></table>

For Table 60, we normalize decoder directions and match features across all six directed comparisons among three seeds. The table reports the mean of the per-comparison median best-match cosine and sharing rate over all active features.

Table 60: Feature matching across vision SAEs.
<table><tr><td>Condition</td><td> $L _ { 0 }$  Alive</td><td>Median best cosine</td><td>Shared &gt; 0.7 (%)</td></tr><tr><td>Standard</td><td>32.00 6,144</td><td>0.3954</td><td>28.84</td></tr><tr><td>Random bypass</td><td>32.00 6,144</td><td>0.3924</td><td>28.50</td></tr><tr><td>PC1 bypass</td><td>32.00 6,144</td><td>0.4186</td><td>16.62</td></tr><tr><td>Test-time, 50 neurons</td><td>32.00 6,144</td><td>0.4655</td><td>13.84</td></tr><tr><td>Test-time, 2 neurons</td><td>32.00 6,144</td><td>0.4531</td><td>25.28</td></tr><tr><td>Pretrained registers</td><td>32.00 6,144</td><td>0.4793</td><td>15.91</td></tr></table>

## M Crosscoders

## M.1 Training and evaluation

Crosscoder settings follow Table 3, with the following changes and additions:

<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Features</td><td>8,192</td></tr><tr><td>Training tokens</td><td>Approximately three million</td></tr><tr><td>Initialization seeds</td><td>0, 42, 137</td></tr><tr><td>Data-order seed</td><td>101</td></tr><tr><td>Checkpoint selection</td><td>Lowest validation loss, checked every 500 steps</td></tr><tr><td>Mean, scale, and PCA sample</td><td>50,000 training activations</td></tr><tr><td>Features ranked and matched</td><td>Features active on the test set</td></tr><tr><td>Matching</td><td>Signed decoder cosine  $> 0 . 7 ,$  in original activation units</td></tr><tr><td>Sharing</td><td>Mean over all six directed seed pairs</td></tr><tr><td>FVU</td><td>Mean over three seeds, in original activation units</td></tr></table>

We take at most 40 entries per article, truncating each to 128 tokens. Validation uses the ranking articles, and test uses the held-out articles in Table 4; neither enters training. Activations are stored in float16 and trained on in float32.

We subtract the input mean and divide by the root mean squared norm. Outputs use either a shared scale or separate scales (Table 61), with unit-length decoder vectors in the scaled coordinates. Bypass removes the chosen components from the decoder and bias after each training step.

For normalized target $\widetilde { \mathbf { y } }$ and reconstruction $\widehat { \widetilde { \mathbf { y } } } ,$ the importance of feature $j$ is

$$
I _ { j } ^ { \mathrm { r e c } } = \mathbb { E } \left[ \left\| \widehat { \widetilde { \mathbf { y } } } - z _ { j } \mathbf { d } _ { j } - \widetilde { \mathbf { y } } \right\| _ { 2 } ^ { 2 } - \left\| \widehat { \widetilde { \mathbf { y } } } - \widetilde { \mathbf { y } } \right\| _ { 2 } ^ { 2 } \right] .
$$

## M.2 Activation pairs and reconstruction

One encoder takes the two activation vectors concatenated, and one decoder reconstructs both. The four Pythia-1.4B pairs use the same tokens in both inputs:

1. Two copies of layer-12 activations.

2. The same copies, with the second copy’s centered PC1 coefficient doubled.

3. Activations from layers 12 and 16.

4. Layer-12 activations at training step 71,000 and at the final checkpoint.

For doubling, we fit the mean and PC1 on 200,000 training activations.

Table 61 gives the methods. Table 62 gives sharing of the top 50 features ranked by $I _ { j } ^ { \mathrm { r e c } }$ . Table 63 gives dictionary sharing and content FVU, excluding each input’s PC1.

Table 61: Crosscoder methods.
<table><tr><td></td><td>Method Definition</td></tr><tr><td>X0</td><td>Reconstruct both inputs with one output scale.</td></tr><tr><td>X1</td><td>Reconstruct both inputs with a separate output scale for each.</td></tr><tr><td>X2</td><td>Bypass PC1 in each input, saving two coefficients.</td></tr><tr><td>X3</td><td>Bypass PC1 of the concatenated inputs, saving one coefficient.</td></tr><tr><td>X4</td><td>Bypass one fixed random direction in each input, saving two coefficients.</td></tr></table>

Table 62: Cross-seed sharing (%) of the top 50 crosscoder features.
<table><tr><td>Activation pair</td><td>X0 X1</td><td>X2</td><td>X3</td><td>X4</td></tr><tr><td>Identical</td><td>2.7 2.0</td><td>91.3</td><td>92.3</td><td>3.3</td></tr><tr><td>PC1 doubled</td><td>3.7 2.0</td><td>83.0</td><td>81.7</td><td>1.3</td></tr><tr><td>Layers 12 and 16</td><td>2.7 3.3</td><td>93.0</td><td>92.7</td><td>3.3</td></tr><tr><td>Step 71,000 and final</td><td>2.7 2.0</td><td>93.0</td><td>92.7</td><td>1.3</td></tr></table>

Table 63: Dictionary sharing (%) and content FVU for standard and PC1-bypass crosscoders.
<table><tr><td>Activation pair</td><td>Dict. X0</td><td>Dict. X2</td><td>Content FVU X0</td><td>Content FVU X2</td></tr><tr><td>Identical</td><td>66.8</td><td>67.2</td><td>0.3661</td><td>0.2753</td></tr><tr><td>PC1 doubled</td><td>60.4</td><td>66.6</td><td>0.4453</td><td>0.3053</td></tr><tr><td>Layers 12 and 16</td><td>72.7</td><td>71.8</td><td>0.3495</td><td>0.2741</td></tr><tr><td>Step 71,000 and final</td><td>69.7</td><td>66.4</td><td>0.3561</td><td>0.2828</td></tr></table>

Features exclusive to one input. Let A and B denote the first and second inputs in each pair. For decoder vectors in the original activation units, define

$$
r _ { j } = \frac { \Vert \mathbf { d } _ { j , B } \Vert _ { 2 } } { \Vert \mathbf { d } _ { j , A } \Vert _ { 2 } + \Vert \mathbf { d } _ { j , B } \Vert _ { 2 } } .
$$

A-only features have $r _ { j } < 0 . 1 $ ; B-only features have $r _ { j } > 0 . 9$

The two ReLU penalties, applied in the scaled coordinates, are

$$
\underbrace { \lambda \mathbb { E } \left[ \sum _ { j } z _ { j } \| \mathbf { d } _ { j } \| _ { 2 } \right] } _ { \mathrm { j o i n t } } \underbrace { \lambda \mathbb { E } \left[ \sum _ { j } z _ { j } \left( \| \mathbf { d } _ { j , A } \| _ { 2 } + \| \mathbf { d } _ { j , B } \| _ { 2 } \right) \right] } _ { \mathrm { b l o c k - s u m } } .
$$

Table 64 gives sparsity and percentages of A-only and B-only features for seed 42 and $\lambda \ \in$ {0.001, 0.003, 0.01}. The percentages are for block-sum. Joint-penalty and TopK runs produce no exclusive features. Both ReLU penalties also produce none on the identical-input and doubled-PC1 pairs.

Table 64: ReLU crosscoder sparsity and features exclusive to one input.
<table><tr><td>Activation pair</td><td>λ</td><td> $L _ { 0 }$  block-sum</td><td> $L _ { 0 }$ </td><td>joint A-only (%) B-only (%)</td><td></td></tr><tr><td>Layers 12 and 16</td><td>0.001</td><td>2707.5</td><td>3044.8</td><td>7.27</td><td>20.75</td></tr><tr><td></td><td>0.003</td><td>619.5</td><td>1003.9</td><td>0.06</td><td>7.11</td></tr><tr><td></td><td>0.010</td><td>62.9</td><td>104.1</td><td>0.02</td><td>1.07</td></tr><tr><td>Step 71,000 and final</td><td>0.001</td><td>2433.1</td><td>2766.4</td><td>13.43</td><td>0.64</td></tr><tr><td></td><td>0.003</td><td>756.3</td><td>1193.1</td><td>0.15</td><td>0.08</td></tr><tr><td></td><td>0.010</td><td>72.6</td><td>125.7</td><td>0.10</td><td>0.02</td></tr></table>

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction describe the measured confounds, the bypass intervention, and the scope of the experiments.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Section 6.3 and Appendix C discuss evaluation limitations and the restricted scope of the rank-one result.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: Appendix C states the assumptions and provides proofs for the isolated rankone setting.

## Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: Section 3 describes the models and data. The appendices give the SAE training and evaluation procedures.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

## Answer: [Yes]

Justification: Code and reproduction instructions are released on GitHub, and SAE checkpoints and preprocessing assets on Hugging Face. The experiments use publicly available datasets.

## Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: Appendix A gives the default settings and data splits; we describe any changes per experiment.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We give bootstrap intervals and results across seeds; the appendices explain how we calculate the intervals and compare seeds.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [No]

Justification: Appendix A.5 lists the hardware and some runtimes.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The research follows the NeurIPS Code of Ethics and involves no humansubject experiments.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No]

Justification: The paper does not explicitly discuss potential positive and negative societal impacts.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: This work does not introduce any such high-risk models or data.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [No]

Justification: The paper does not yet explicitly list licenses and terms for all datasets, models, and code used.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: The code and SAE checkpoints are accompanied by training configurations, usage instructions, licenses, and documentation of known limitations.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: The research involves no crowdsourcing or human participants.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: The research involves no human participants.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [Yes]

Justification: Section 6.3 and Appendix G describe the LLM explainers, evaluators, and scoring procedure.

Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.