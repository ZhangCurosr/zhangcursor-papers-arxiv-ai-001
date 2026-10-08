# Eficient Patch Based Anomaly Detection Fused with Difusion Driven Generative Modeling for Semiconductor Wafer Bin Map Open Set Anomaly Detection

Limon Bin Hossain<sup>1,\*</sup> and Md Sadib Rahman Ananta<sup>1</sup>

<sup>1</sup>Department of Industrial and Production Engineering, Bangladesh University of Engineering and Technology (BUET), Dhaka, 1000, Bangladesh Corresponding author: mdlimonbinhossain@gmail.com

## Abstract

Spatial defect signatures on wafer bin maps (WBMs) trace yield loss to specific process faults, yet supervised classifiers recognize only the defect types seen during training, and one-class detectors built on a single mechanism tend to capture either local structural deviations or global distributional violations, but rarely both. This work proposes a hybrid one-class framework that couples a patch-based student–teacher detector (EficientAD) with a denoising difusion probabilistic model (DDPM) used for partial-difusion reconstruction, and fuses their percentile-calibrated scores through a fixed convex combination. Trained on only 700 normal wafers from the WM-38K mixed-type dataset and evaluated on 18,658 heldout wafers, the fused detector reached an AUROC of 0.9985 and reduced misclassifications from 852 (DDPM) and 1,412 (EficientAD) to 618, with all pairwise diferences significant at $p < 0 . 0 0 1$ . Beyond aggregate accuracy, the analysis shows that the gain arises from weakly overlapping errors between the two modules, yet fixed-weight fusion recovers only 40–70% of the correction available to an oracle selector. Under the benchmark’s inverted class balance, average precision and F1 saturate, while the Matthews correlation coeficient and negative predictive value expose unreliable normal predictions. Pixel-level maps further show that strong image-level separability does not imply spatial localization, and the difusion module succeeds as a local density prior rather than through global geometric reasoning. These findings motivate sample-adaptive fusion and imbalance-aware evaluation of hybrid wafer anomaly detectors.

Keywords: wafer bin map; anomaly detection; difusion model; knowledge distillation; semiconductor manufacturing; score fusion

## 1 Introduction

In semiconductor fabrication, each die on a wafer is tested, and results are recorded on a wafer bin map (WBM). The spatial patterns from these tests provide diagnostic insights, revealing issues like scratches and defects that can be traced to specific equipment or process steps<sup>1</sup>. Unclassified signatures can lead to yield loss and production delays, underscoring the importance of not only detecting them but also quickly resolving them. Traditionally, defect recognition has been approached as a supervised multi-label classification problem, where convolutional networks excel under closed-set conditions <sup>2</sup>. However, these classifiers can only identify trained defect types and may fail on significantly diferent patterns. An alternative approach is to frame wafer defect detection as a one-class anomaly detection problem. This method primarily trains models on defect-free wafers <sup>3</sup>, thereby flagging any significant deviation as anomalous. Two main families of one-class methods exist: one that compares a test wafer to a reference feature representation and another that models a learned data distribution. The structural approach uses a pretrained teacher network and a student network trained on defect-free images to identify anomalies by comparing patch-level features<sup>4</sup>. This has been further developed for patch-level feature distribution modeling <sup>5</sup>, one-class bottleneck reconstruction<sup>6</sup>, and rapid inference in the EficientAD architecture <sup>7</sup>. While structural detectors efectively identify small anomalies, their patch-level focus may hinder the detection of patterns that emerge only across the entire image.

The second family of one-class methods addresses the limitation of patch-based modeling by analyzing the image as a whole. Generative models learn the distribution of normal images and use reconstruction quality to identify deviations. Denoising difusion probabilistic models (DDPMs) are one such approach<sup>8</sup>, enhanced by learned variance schedules<sup>9</sup> and deterministic implicit sampling<sup>10</sup>. They adapt to reconstruction tasks by partially difusing a test image and measuring the discrepancy between the input and its denoised version<sup>11</sup>. This approach captures the overall structure of an image, allowing it to detect distributional anomalies that patch-level methods may miss. However, global reconstruction methods are less sensitive to small, localized texture defects, and difusion models using standard Gaussian noise can lose fine-grained local information during the difusion process <sup>11</sup>.

The complementary limitations of structural and generative detectors motivate their combination in a hybrid framework by calibrating anomaly scores and applying fixed weights. However, ensemble methods do not always enhance performance through simple score averaging, especially if weaker detector errors correlate with stronger ones or contribute uninformative variation, which can dilute the reliable detector’s contribution <sup>12</sup>. An efective hybrid model requires a strategy that accounts for both the reliability and the correlation of its detectors. This study investigates the impact of detector combinations on anomaly detection performance, treating the fusion mechanism as a core aspect of the detection framework to systematically evaluate each detector’s contributions and interactions.

These considerations raise three research questions that this study is designed to answer.

RQ1: Does fixed-weight score fusion of a patch-based structural anomaly detector and a difusion-based generative anomaly detector improve wafer defect detection performance compared to the individual detectors, and when does fusion lead to performance degradation<sup>12</sup>?

To answer it, we train both detectors, fuse their calibrated scores with a fixed convex combination, and compare all three detectors under bootstrap-validated and McNemartested conditions.

RQ2: How does the fused score relate geometrically to each detector’s calibrated score across evaluation metrics, and is this relationship monotonic?

To answer it, we examine the fusion mechanism using score distributions, receiver operating characteristic (ROC) curves, precision-recall curves, and calibration reliability diagrams for the structural, generative, and fused detectors.

RQ3: What structural properties and difusion-based scoring explain the resulting detector ranking, and what does this imply for the fusion and evaluation of hybrid anomaly detectors?

To answer it, we inspect per-sample anomaly localization maps, the DDPM’s generative sampling behavior, and its detection performance.

## 2 Materials and Methods

## 2.1 Dataset and Preprocessing

WM-38K (MixedWM38) comprises 38,015 wafer bin maps with a resolution of $5 2 \times 5 2 ^ { 2 }$ Each map is an integer array over {0, 1, 2} (blank, normal-die, broken-die), paired with an 8-bit label vector over eight defect primitives (center, donut, edge-loc, edge-ring, loc, nearfull, scratch, random) that decodes to one of 38 classes. The mixing degree $m \in \{ 0 , 1 , 2 , 3 , 4 \}$ (the sum of the label vector) has 1,000, 7,015, 13,000, 13,000, and 4,000 wafers at $m = 0$ through $m = 4$ . Binary labels were set as $y = \nVdash [ m > 0 ]$ , giving 1,000 normal wafers (2.6%) and 37,015 defective wafers (97.4%), the inverse of the typical industrial base rate.

Each map was one-hot encoded into three binary channels (blank, normal-die, brokendie), zero-padded by six pixels on each side to $3 \times 6 4 \times 6 4$ , and rescaled to [−1, 1] by $x \mapsto 2 x - 1$

The 1,000 normal wafers were split $7 0 \% / 1 5 \% / 1 5 \%$ into training (700), validation (150), and test (150); only the 700 training normals were used to fit any parameter. The 37,015 defective wafers, never used for training, were split $5 0 \% / 5 0 \%$ between validation (18,507) and test (18,508). The validation partition (18,657 wafers) was used for percentile calibration and threshold selection (Section 2.4); the test partition (18,658 wafers) was reserved for the final metrics in Section 3.

## 2.2 Structural Module: Patch-Based Student-Teacher Detection

The structural branch follows EficientAD, comparing a frozen pretrained teacher with a lightweight trainable student.<sup>4,7</sup> The teacher is an ImageNet-pretrained ResNet-18<sup>13</sup> truncated after its third residual stage (256 channels, stride 16) and projected to 384 channels by a frozen 1 × 1 convolution; inputs are mapped to [0, 1] and standardized with ImageNet statistics before the teacher, which outputs a $3 8 4 \times 4 \times 4$ map for a $6 4 \times 6 4$ input.

The student is a multi-scale patch description network (PDN) operating on [−1, 1] inputs, with three parallel branches (128 channels each) using kernel sizes $k \in \{ 4 , 6 , 8 \} , 2 \times 2$ average pooling with stride 2, and LeakyReLU (slope 0.1). The branches are cropped to a common size, concatenated, and fused by a $1 \times 1$ convolution to a $3 8 4 \times 2 5 \times 2 5$ output. A convolutional encoder-decoder with skip connections (three $3 \times 3$ stages with 64, 128, and 256 channels) is trained jointly to regress the teacher features but does not contribute to the anomaly score. Discrepancies are computed on the overlapping top-left $4 \times 4$ window of the teacher and student grids. With cropped features $T ( x )$ and S(x) (C = 384 channels),

$$
d ( \boldsymbol { x } ) _ { i j } = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } \bigl ( T ( \boldsymbol { x } ) _ { c i j } - S ( \boldsymbol { x } ) _ { c i j } \bigr ) ^ { 2 } .\tag{1}
$$

Training uses a hard-feature loss: only the $k = \operatorname* { m a x } ( 1 , \lfloor 0 . 2 h w \rfloor ) = 3$ positions with the largest discrepancy contribute to the student loss $\mathcal { L } _ { \mathrm { s t } }$ . The autoencoder loss $\mathcal { L } _ { \mathrm { a e } }$ is the mean squared error between the teacher features and the autoencoder’s bilinearly resized output. An out-of-distribution term regularizes the student on channel-permuted inputs (an independent permutation π per sample),

$$
\mathcal { L } _ { \mathrm { o o d } } = \Big ( \overline { { S ( \pi ( x ) ) ^ { 2 } } } + 1 0 ^ { - 8 } \Big ) ^ { - 1 } ,\tag{2}
$$

which pushes the student’s feature energy on permuted inputs upward. The total objective is

$$
\mathcal { L } _ { \mathrm { E A D } } = \mathcal { L } _ { \mathrm { s t } } + \mathcal { L } _ { \mathrm { a e } } + 0 . 1 \mathcal { L } _ { \mathrm { o o d } } .\tag{3}
$$

## 2.3 Generative Module: Difusion-Based Reconstruction Detection

The generative branch trains a DDPM on normal wafers only and scores anomalies by reconstruction error under partial difusion.<sup>8,11</sup> Forward difusion uses a cosine schedule over $T = 1 0 0 0$ steps (ofset $s = 0 . 0 0 8 , \beta _ { t }$ clipped to $[ 1 0 ^ { - 4 } , 0 . 9 9 9 9 ] ) ^ { 9 }$

$$
x _ { t } = \sqrt { \bar { \alpha } _ { t } } x _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon , \qquad \epsilon \sim { \mathcal N } ( 0 , I ) , \qquad \bar { \alpha } _ { t } = \prod _ { s = 1 } ^ { t } ( 1 - \beta _ { s } ) .\tag{4}
$$

The reverse process is a U-Net $\epsilon _ { \theta } ( x _ { t } , t )$ with a 128-dimensional sinusoidal timestep embedding projected through an MLP to 320 dimensions. Residual blocks (group normalization, SiLU, 3×3 convolutions) operate at three resolutions $( 6 4 \times 6 4 , 3 2 \times 3 2 , 1 6 \times 1 6 )$ with channel widths [48, 96, 192] and two blocks per level; self-attention <sup>14</sup> with four heads is applied at $1 6 \times 1 6$ . The bottleneck at $1 6 \times 1 6$ with 192 channels uses three residual blocks, rather than the two used elsewhere, to add capacity without changing the receptive field. The decoder mirrors the encoder, using skip-connection concatenation and transposed convolution upsampling.

Training minimizes the simplified noise-prediction objective <sup>8</sup>

$$
\mathcal { L } _ { \mathrm { D D P M } } = \mathbb { E } _ { x _ { 0 } , t , \epsilon } \big \| \epsilon - \epsilon _ { \theta } ( x _ { t } , t ) \big \| _ { 2 } ^ { 2 } ,\tag{5}
$$

with t drawn uniformly on $\{ 0 , \ldots , T - 1 \}$ , AdamW (learning rate $2 \times 1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 5 } )$ , cosine annealing over 30 epochs, gradient clipping at 1.0, batch size 32 (630 updates total), and an exponential moving average of the weights (decay 0.9999) used for all subsequent sampling and scoring. Sampling and scoring both use deterministic DDIM updates $( \eta =$ 0).<sup>10</sup> At each update, $\hat { x } _ { 0 } = ( x _ { t } - \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon _ { \theta } ( x _ { t } , t ) ) / \sqrt { \bar { \alpha } _ { t } }$ is clipped to $[ - 1 , 1 ]$ , and $\begin{array} { r l } { x _ { t ^ { \prime } } } & { { } = } \end{array}$ $\sqrt { \bar { \alpha } _ { t ^ { \prime } } } \hat { x } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t ^ { \prime } } } \epsilon _ { \theta } ( x _ { t } , t )$ . Unconditional generation uses 50 steps from $t = 9 9 9$ to $t = 0$ For scoring, a test image is forward-difused once to $t ^ { * } = 5 0 0$ (where $\bar { \alpha } _ { 5 0 0 } = 0 . 4 9 2 1 )$ and denoised back with 50 DDIM steps to give the reconstruction ${ \hat { x } } .$ . The anomaly map is

$$
A _ { \mathrm { D D P M } } ( x ) _ { i j } = \frac { 1 } { 3 } \sum _ { c = 1 } ^ { 3 } \bigl ( x _ { 0 , c i j } - \hat { x } _ { c i j } \bigr ) ^ { 2 } ,\tag{6}
$$

computed over all three channels and all $6 4 \times 6 4$ positions, with the raw score $r _ { \mathrm { D D P M } } ( x )$ its spatial mean. Partial difusion to $t ^ { * } = 5 0 0$ preserves enough input structure that a normal wafer should reconstruct with low error while a distributional violation should not, even though both share low-level statistics such as die density.

## 2.4 Score Calibration and Fusion

Raw scores are calibrated independently per module because they have unrelated units and dynamic ranges. The lower anchor $q _ { \mathrm { l o } }$ is the 1st percentile of validation normal scores; the upper anchor $q _ { \mathrm { h i } }$ is the 99th percentile of validation defective scores (falling back to the pooled distribution if too few defective validation samples are available). The calibrated score is

$$
s ( x ) = \mathrm { c l i p } \left( \frac { r ( x ) - q _ { \mathrm { l o } } } { q _ { \mathrm { h i } } - q _ { \mathrm { l o } } + 1 0 ^ { - 8 } } , 0 , 1 \right) .\tag{7}
$$

This anchors 0 to a typical normal wafer and 1 to a clearly defective one for each module separately; calibrated scores are monotone transforms of the raw scores and are not probabilities. Per-module thresholds $\tau$ were set on the validation set by maximizing Youden’s J statistic $( \mathrm { T P R - F P R } )$ ; a wafer is labeled defective when $s ( x ) \geq \tau$

The fused score is

$$
s _ { \mathrm { f u s e d } } ( x ) = \alpha s _ { \mathrm { E A D } } ( x ) + ( 1 - \alpha ) s _ { \mathrm { D D P M } } ( x ) ,\tag{8}
$$

with $\alpha = 0 . 5$ throughout this study. The fused threshold was recalibrated on the fused validation scores by the same criterion. Spatial maps are fused analogously after per-sample min-max normalization, as α $\tilde { A } _ { \mathrm { E A D } } ( x ) + ( 1 - \alpha ) \tilde { A } _ { \mathrm { D D P M } } ( x )$

## 2.5 Evaluation Protocol

Threshold-free metrics (AUROC, average precision) were computed from calibrated scores, while threshold-dependent metrics (F1, Matthews correlation coeficient, balanced accuracy) were derived from predictions at the validation-selected threshold. AUROC and AP are reported together due to their divergence under class imbalance, with precision-recall often being more informative on skewed data <sup>15</sup>. MCC is noted for its lower susceptibility to imbalance-driven inflation compared to F1 or accuracy<sup>16</sup>. Uncertainty was quantified using a nonparametric bootstrap (1,000 resamples with replacement, with degenerate single-class resamples discarded, and 95% confidence intervals from the 2.5th and 97.5th percentiles), keeping thresholds fixed<sup>17</sup>. Pairwise comparisons included a paired bootstrap on the AU-ROC diference (shared resample indices across detectors, with a two-sided empirical p-value computed as twice the smaller tail fraction, capped at 1), the DeLong test on paired testset scores<sup>18</sup>, and the continuity-corrected McNemar test on paired thresholded predictions <sup>19</sup>, where $\chi ^ { 2 } = ( | b - c | - 1 ) ^ { 2 } / ( b + c )$ with b and c as counts correct under only one detector. Fusionweight sensitivity was assessed by sweeping α from 0 to 1 in 0.1 increments, re-estimating the fused threshold on the validation set at each step, with $\alpha = 0 . 5$ fixed beforehand for this analysis. Finally, 500 test wafers were randomly drawn, their student-network feature maps were global-average-pooled to 384 dimensions, and the resulting vectors were projected to two dimensions with t-SNE (perplexity 40, 1,000 iterations, automatic learning rate) <sup>20</sup>.

## 3 Results and Discussion

## 3.1 Dataset Characteristics

The WM-38K dataset contains 1 normal pattern, 8 single-defect patterns, and 29 mixeddefect patterns. The images are sized at 52×52 pixels<sup>21</sup>. Thirty-five classes have exactly 1,000 wafers, with exceptions for Near-Full (C07, about 149), Random (C09, about 866), and Center+Edge-Loc+Scratch (C24, 2,000). Two and three-primitive mixtures account for 13,000 wafers, making up 81.0% of defective wafers; wafers with at least two defect primitives are included. The pixel-wise defect probability map shows three spatial structures: a central hotspot near 0.40 from the Center and Donut patterns, a thin peripheral ring above 0.50 from the Edge-Ring and Edge-Loc patterns, and a faint arc in the lower half from the Scratch. The remaining die field sits near 0.25, reflecting random background failures across all classes. The normal class (C01) is not defect-free, containing sparse, random broken dies. The oneclass boundary needs to separate structured patterns from a nonzero random background rather than just detect failing dies.

## 3.2 Model Configuration and Training Dynamics

Each epoch contains 21 minibatches of 32 normal wafers. EficientAD receives 4,200 parameter updates over 200 epochs, while DDPM receives 630 updates over 30 epochs. EficientAD converges early (Figure 1), with its total loss dropping from 0.40984 to 0.01025 by epoch 50 (a 97.5% reduction) and reaching 0.00763 at epoch 200. The final 150 epochs contribute only 25.6% of the additional reduction. The student and autoencoder terms decrease by 98.0% (0.18333 to 0.00361) and 97.7% (0.04799 to 0.00111), respectively. The objective sums the student loss, the autoencoder loss, and the out-of-distribution (OOD) penalty, weighted at 0.1, with the residual 0.00291 assigned to the weighted OOD term, accounting for 38.1% of the final objective. This indicates that the regularizer suppresses the student’s of-manifold response throughout training. The DDPM loss drops from 0.62178 to 0.05161 (91.7%) but continues to decline at epoch 30, with an additional 4.5% decrease over the last five epochs.

![](images/68a3fc4f6ec1ac317db0e269e4610d2219f0b54028eb471a3e683878a2c3de73.jpg)  
(a) EficientAD total

![](images/8a207cc2c81724d01971b21c40b84990743ab7b55577bd74fe01f94e4b6cc8c6.jpg)  
(b) EficientAD student

![](images/bceb1bfdf9b705247c9d9ee84c9448cd5a23466d1e1f0efad4ba6942527ab677.jpg)  
(c) EficientAD autoencoder

![](images/7b8aa4f607e427bc647a36fc529fba89b87ccb20b7111519265b796cd49dd525.jpg)  
(d) DDPM noise prediction  
Figure 1: Training loss curves (raw and smoothed). Dotted lines mark final values.

Table 1: Test-set performance with 95% bootstrap confidence intervals. Best values are in bold. Params count trainable weights only.
<table><tr><td>Model</td><td>AUROC</td><td>AP</td><td>F1</td><td>MCC</td><td>Bal. Acc.</td><td>Params (M)</td></tr><tr><td>EfficientAD</td><td>0.9799 [0.9728, 0.9866]</td><td>0.9998 [0.9998, 0.9999]</td><td>0.9604 [0.9582, 0.9624]</td><td>0.2781 [0.2541, 0.3033]</td><td>0.9288 [0.9072, 0.9473]</td><td>5.21</td></tr><tr><td>DDPM</td><td>0.9931 [0.9907, 0.9952]</td><td>0.9999 [0.9999, 1.0000]</td><td>0.9764 [0.9747, 0.9779]</td><td>0.3620 [0.3310, 0.3920]</td><td>0.9538 [0.9353, 0.9692]</td><td>7.83</td></tr><tr><td>Fusion (α = 0.5)</td><td>0.9985 [0.9978, 0.9991]</td><td>1.0000 [1.0000, 1.0000]</td><td>0.9830 [0.9817, 0.9843]</td><td>0.4321 [0.3996, 0.4645]</td><td>0.9800 [0.9721, 0.9843]</td><td>13.04</td></tr></table>

It receives 6.7 times fewer updates than EficientAD.

## 3.3 Detection Performance

Table 1 presents test-set performance, with Hybrid fusion leading in all metrics. Its AU-ROC of 0.9985 surpasses DDPM (0.9931) and EficientAD (0.9799), with non-overlapping bootstrap intervals for AUROC, F1, and balanced accuracy between fusion and EficientAD. Notably, the generative module outperforms the structural module across metrics, contrary to expectations for patch-level detectors on geometric defects. Average precision is saturated at 0.9998 for all detectors, and with a defective prevalence of 99.2%, even random ranking yields an AP close to 0.992, limiting real diferences and discriminative information <sup>15</sup>. Additionally, MCC remains low (0.278 to 0.432) despite F1 exceeding 0.96 for all models, as MCC penalizes errors on the minority normal class, which F1 does not address <sup>16</sup>.

Figure 2 explains the ranking. For EficientAD, normal scores concentrate between 0 and 0.30, but the defective distribution has a long left tail that reaches about 0.10 and overlaps the normal mode at the threshold $\tau = 0 . 2 7 3$ . The DDPM places its defective mode near 0.62 and, more importantly, its defective density vanishes below about 0.30 $\left( \tau = 0 . 4 2 3 \right)$ The DDPM’s advantage therefore comes from compressing the lower tail of the defective distribution, not from pulling normal scores further down.

Figure 3 compares the three detectors. The ROC curves separate most clearly in the highspecificity regime. At a false-positive rate near zero, fusion detects about 97% of defective wafers, while DDPM detects about 94% and EficientAD about 73%. This is crucial in production as every false alarm prompts manual review. The precision-recall curves reflect a similar trend. EficientAD’s precision drops near a recall of 0.73, DDPM’s near 0.93, while fusion maintains a precision of 1.0 until a recall of about 0.97. In the reliability diagram, all curves lie above the diagonal, with scores being percentile-anchored ranks rather than probabilities, thus underestimating defect frequency. Post-hoc recalibration, such as isotonic regression, would be necessary for risk estimates. The lowest EficientAD bin (mean score around 0.06) includes 36% defective wafers, indicating left-tail contamination. In contrast, the DDPM and fusion bins with scores below 0.16 contain no defective wafers.

![](images/0c051bc54622fb078410210c1aaa3be8d6ce6cdde5fe8d9e766e1d42503de04f.jpg)  
(a) EficientAD (τ = 0.273)

![](images/014e05f34f754197550435f1107f31a55e6e7f92ba19f6e238d320ad85750152.jpg)  
(b) DDPM (τ = 0.423)  
Figure 2: Normalized test-set anomaly score distributions. Dashed lines mark validation-selected thresholds.

## 3.4 Operating Point, Error Complementarity, and Significance

Table 2 presents the confusion counts at the selected validation thresholds. These counts correspond to the reported F1, MCC, and balanced accuracy under fixed class totals, and they reproduce all McNemar discordance counts in Table 3. Fusion reduces the total error count from 1,412 (EficientAD) and 852 (DDPM) to 618, and false alarms from 10 and 7 to one wafer. The table also clarifies the low MCC; of the 766 wafers labeled normal by fusion, only 149 are genuinely normal, resulting in a negative predictive value (NPV) of 19.5%. This reflects the test composition rather than a model issue. Even at a 96.7% true-positive rate, the 617 missed defects outnumber the 150 normal wafers by a factor of four. Thus, a "defective" verdict is nearly certain (PPV of 0.9999), while a "normal" verdict is unreliable given this prevalence.

Table 3 shows significant pairwise diferences at $p < 0 . 0 0 1$ across three tests: a paired bootstrap on the AUROC diference, the DeLong test for correlated ROC curves <sup>18</sup>, and the continuity-corrected McNemar test on thresholded predictions <sup>19</sup>. The McNemar counts reveal why fusion works. EficientAD and the DDPM disagree on 1,720 wafers but fail together on only 272. The DDPM correctly classifies 1,140 of the 1,412 EficientAD errors (80.7%), while EficientAD correctly classifies 580 of the 852 DDPM errors (68.1%). This low joint-error rate indicates diversity, under which outlier ensembles improve on their best member<sup>12</sup>. Fusion repairs 895 EficientAD errors with only 101 new ones and repairs 482 DDPM errors with 248 new ones. An oracle would make only 272 errors. Fixed-weight fusion (618 errors) captures 69.6% of available headroom relative to EficientAD and 40.3% relative to the DDPM. The remaining 346 recoverable errors define the target for a sample-adaptive gating function.

![](images/dbcbd2615d4b9d65502aa7adbd5efb4f4fa07944be1b995c26bf569c0561f9cb.jpg)  
(a) ROC

![](images/06ddf76987ae6c30ceae699c2c70482fce4f4627e1d7fc197e66407c9639650a.jpg)  
(b) Precision-recall

![](images/d73b8a0f3353d1fb0721128ee874dd3d23c0be62aa35286171a485428939231e.jpg)  
(c) Reliability diagram  
Figure 3: Three-way comparison of EficientAD, DDPM, and hybrid fusion on the test set.

Table 2: Confusion counts and derived rates at the validation-selected thresholds (test set; 18,508 defective, 150 normal).
<table><tr><td>Model</td><td>TP</td><td>FN</td><td>TN</td><td>FP</td><td>Errors</td><td>TPR</td><td>TNR</td><td>PPV</td><td>NPV</td></tr><tr><td>EfficientAD</td><td>17,106</td><td>1,402</td><td>140</td><td>10</td><td>1,412</td><td>0.924</td><td>0.933</td><td>0.9994</td><td>0.091</td></tr><tr><td>DDPM</td><td>17,663</td><td>845</td><td>143</td><td>7</td><td>852</td><td>0.954</td><td>0.953</td><td>0.9996</td><td>0.145</td></tr><tr><td>Fusion</td><td>17,891</td><td>617</td><td>149</td><td>1</td><td>618</td><td>0.967</td><td>0.993</td><td>0.9999</td><td>0.195</td></tr></table>

## 3.5 Sensitivity to the Fusion Weight

Figure 4 shows the fusion weight α for EficientAD from 0 to 1. AUROC is concave in α, peaking at $\alpha = 0 . 5 ~ ( 0 . 9 9 8 5 4 )$ , with $\alpha = 0 . 4$ close behind (0.99847). Weights from 0.1 to 0.7 outperform both single modules. The curve is asymmetric: moving from $\alpha = 0 . 5$ to pure EficientAD reduces AUROC by 0.0186, whereas transitioning to pure DDPM reduces it by 0.0054, reflecting the stronger performance of DDPM. Even a 10% EficientAD weight increases AUROC from 0.9931 to 0.9956. F1 fluctuates between 0.9788 and 0.9873 for α from 0.1 to 0.6, peaking at $\alpha = 0 . 4$ , with fluctuations driven by threshold noise due to limited validation normals. AP remains at or above 0.99983 for all α. The default $\alpha = 0 . 5$ was established before this sweep, making the results in Table 1 not tuned on test data; the sweep itself uses the test set for sensitivity analysis.

Table 3: Pairwise significance tests on the test set. For McNemar, $\mathrm { \Delta ^ { 6 6 } A { - } o n l y ^ { \prime } }$ and “B-only” count wafers misclassified by one model alone; “Both” counts shared errors. EAD: EficientAD. \*\*\* denotes $p < 0 . 0 0 1$
<table><tr><td></td><td colspan="2">Paired bootstrap</td><td colspan="2">DeLong</td><td colspan="4">McNemar</td></tr><tr><td>Comparison (A vs. B) ∆AUROC</td><td></td><td>95% CI</td><td>z</td><td>p</td><td></td><td></td><td>A-only B-only Both</td><td> $\chi ^ { 2 }$ </td></tr><tr><td>Fusion vs. EAD</td><td>+0.0184</td><td> $\mathrm { \ t + 0 . 0 1 2 3 , + 0 . 0 2 5 6 ] }$ </td><td>+5.335</td><td> $9 . 6 \times 1 0 ^ { - 8 }$ </td><td>101</td><td>895</td><td>517</td><td>631.37***</td></tr><tr><td>Fusion vs. DDPM</td><td>+0.0054</td><td> $\left[ + 0 . 0 0 3 4 , + 0 . 0 0 7 8 \right]$ </td><td>+5.124</td><td> $3 . 0 \times 1 0 ^ { - 7 }$ </td><td>248</td><td>482</td><td>370</td><td>74.37***</td></tr><tr><td>EAD vs. DDPM</td><td>-0.0130</td><td>[-0.0209, -0.0056]</td><td>-3.423</td><td> $6 . 2 \times 1 0 ^ { - 4 }$ </td><td>1,140</td><td>580</td><td>272</td><td>181.68***</td></tr></table>

![](images/d46216d246a3dc785ebe540e96cdf04506145a56417293dfe3e73a4a32725184.jpg)  
Figure 4: Test-set AUROC, AP, and F1 as a function of the fusion weight α (α = 0: DDPM only; α = 1: EficientAD only).

## 3.6 Spatial Anomaly Maps and Generative Fidelity

Figure 5 decomposes one test wafer from each outcome category. The EficientAD map shows a diamond-shaped hotspot slightly below and to the right of the wafer center, which is an architectural artifact. With a 64 × 64 input, the truncated ResNet-18 teacher generates a 4 × 4 feature grid (stride 16), while the student produces a $2 5 \times 2 5$ grid. The discrepancy is calculated over the overlapping 4 × 4 window, which does not align with the teacher cells. This 4 × 4 map is bilinearly upsampled by a factor of 16. It cannot resolve 52-die defect geometry but separates the classes based on the global mass of broken dies, lacking pixellevel localization. The DDPM map is unstructured and spreads across the blank padding region due to a single random noise draw during the forward difusion to $t ^ { * } = 5 0 0$ , followed by averaging over three channels. This results in a stable image-level score, but the per-pixel map lacks localization. The fused map combines the EficientAD hotspot with the DDPM texture, improving image-level decisions without enhancing localization. The error cases highlight the hardest defect type. Both the true positive (score 0.428) and false negative (score 0.307) are Scratch wafers, with the missed one being a thin arc along the lower-left edge. The false positive is a normal wafer with a score of 0.347. Thus, the fused threshold lies between 0.307 and 0.347. Thin, low-density scratches contribute few broken dies compared to random background, causing detectors that score global defect mass to rate them similarly to normal wafers.

![](images/f2438a6ead2da2982a457b54cf17a8652827cdb0f6658987fc32c37e1f71de47.jpg)  
Figure 5: Spatial anomaly map decomposition for one true-positive, true-negative, false-positive, and false-negative test wafer (top to bottom). Columns: original wafer map, EficientAD map, DDPM reconstruction-error map, and fused map. Scores are fused image-level scores.

Figure 6 presents unconditional samples from a trained DDPM using 50-step deterministic DDIM sampling, achieving 11.14 iterations per second on the T4 (approximately 4.5 s per 50-step pass). The samples fill the 64 × 64 canvas with dense die texture but lack a circular wafer boundary, each displaying a single blob-like cluster of broken dies at random positions. The model captures local die statistics and cluster priors but fails on global wafer geometry, given only 630 updates across 700 training images. Nevertheless, it achieves an AUROC of 0.9931 as a detector. This performance is attributed to the scoring timestep; at $t ^ { * } = 5 0 0$ the noisy input retains 70% of its signal amplitude, making the reverse process rely on local denoising rather than global synthesis. Thus, the generative module functions as a learned local density prior. Its detection ability is separate from generative fidelity, and the global "logical anomaly" reasoning typically associated with difusion models does not drive its performance on this benchmark.

![](images/250696777e3f555827df450d245993c70eef21d7bae17db899b8ce60a22ae7f3.jpg)  
Figure 6: Unconditional DDIM samples (50 steps) from the DDPM trained on 700 normal wafers (broken-die channel shown).

## 4 Conclusion

This study developed a hybrid one-class framework for wafer bin map anomaly detection that pairs a patch-based student–teacher detector (EficientAD) with a difusion-based reconstruction detector (DDPM) and fuses their calibrated scores through a fixed convex weight. The framework was trained exclusively on 700 normal wafers from WM-38K, evaluated on 18,658 held-out wafers, and examined not only for their detection accuracy but also for the behavior of the fusion mechanism itself. Fixed equal-weight fusion outperformed both constituent detectors on every metric. It raised AUROC to 0.9985, compared with 0.9931 for the DDPM and 0.9799 for EficientAD, reduced the total error count to 618, and limited false alarms to a single normal wafer. All pairwise diferences were significant at $p < 0 . 0 0 1$ under the paired bootstrap, DeLong, and McNemar tests. Fusion did not degrade performance for any interior weight $0 . 1 \le \alpha \le 0 . 7 ;$ degradation appeared only as the weight approached pure EficientAD. The gain stems from error complementarity, since the two modules disagreed on 1,720 wafers but failed together on only 272. Fixed weighting, however, captured only 69.6% of the oracle headroom relative to EficientAD and 40.3% relative to the DDPM, leaving 346 recoverable errors unexploited. The two modules produce qualitatively diferent score geometries. EficientAD assigns defective wafers a long left tail that overlaps the normal mode, whereas the DDPM compresses the lower tail of the defective distribution so that its density vanishes below a calibrated score of about 0.30. The fused score inherits this clean lower tail while adding structural evidence. The relationship between the fusion weight and AUROC is not monotonic; it is concave and asymmetric, peaking at $\alpha = 0 . 5$ and penalizing reliance on the weaker module far more than reliance on the stronger one. All three calibrated scores lie above the diagonal of the reliability diagram, confirming that they are percentile-anchored ranks rather than probabilities. The inverted class balance of WM-38K saturates the average precision $\left( \ge \ 0 . 9 9 9 8 \right)$ and F1 (> 0.96), whereas MCC (0.278–0.432) and the negative predictive value of the fused detector (19.5%) reveal that “normal” verdicts remain unreliable. Both modules efectively score global broken-die mass: the EficientAD map is an architectural artifact of a 4 × 4 teacher grid, and the DDPM map is spatially unstructured. The DDPM detects well despite generating samples that lack wafer geometry, because partial difusion to t<sup>∗</sup> = 500 conditions reconstruction strongly on the input and reduces it to local denoising. Consequently, thin, low-density scratches are the dominant residual failure mode.

This study has several limitations. WM-38K inverts the industrial base rate: 97.4% of all wafers and 99.2% of the test partition are defective, leaving only 150 normal wafers in each validation and test partition. All experiments use a single dataset with a native resolution of 52 × 52 and lack cross-dataset or cross-fab validation. The comparison includes only the constituent modules and their fusion; established one-class detectors were not evaluated under the same protocol. The structural module is limited in its ability to adapt to small inputs. Future work could include sample-adaptive fusion. A gating function using signals such as inter-module score disagreement or local defect density could target the 346 errors that remain unrecovered. To remain consistent with the one-class setting, the gate should rely on unsupervised cues or a separate held-out partition. Evaluation under realistic prevalence is important. Benchmarks like WM-811K with larger normal pools would stabilize threshold selection and allow reporting of operationally relevant quantities, including the false-alarm rate at a fixed recall. Conformal calibration could provide guaranteed control over the falsealarm rate on normal wafers.

## Abbreviations

AD, anomaly detection; AE, autoencoder; AP, average precision; AUROC, area under the receiver operating characteristic curve; CI, confidence interval; DDIM, denoising difusion implicit model; DDPM, denoising difusion probabilistic model; EMA, exponential moving average; MCC, Matthews correlation coeficient; OOD, out-of-distribution; PDN, patch description network; ROC, receiver operating characteristic; t-SNE, t-distributed stochastic neighbor embedding; WBM, wafer bin map.

## Author Contributions

Limon Bin Hossain: conceptualization, methodology, software, formal analysis, writing – original draft. Md. Sadib Rahman Ananta: writing – review and editing, validation, visualization.

## Funding Sources

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Notes

The authors declare no competing financial interest.

## Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of this work, the author(s) used QuilBot Premium /Grammarly Premium/ChatGPT/Claude Premium to improve the quality of the writing and check for any grammatical errors. After using this tool/service, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the publication.

## Acknowledgments

The authors thank the maintainers of the WM-38K (MixedWM38) dataset for making it publicly available for anomaly detection research.

## References

[1] Nakazawa, T.; Kulkarni, D. V. Wafer Map Defect Pattern Classification and Image Retrieval Using Convolutional Neural Network. IEEE Trans. Semicond. Manuf. 2018, 31, 309–314.

[2] Wang, J.; Xu, C.; Yang, Z.; Zhang, J.; Li, X. Deformable Convolutional Networks for Eficient Mixed-Type Wafer Defect Pattern Recognition. IEEE Trans. Semicond. Manuf. 2020, 33, 587–596.

[3] Batzner, K.; Heckler, L.; König, R. Eficientad: Accurate visual anomaly detection at millisecond-level latencies. 2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). 2024; pp 127–137.

[4] Bergmann, P.; Fauser, M.; Sattlegger, D.; Steger, C. Uninformed Students: Student-Teacher Anomaly Detection with Discriminative Latent Embeddings. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2020; pp 4183–4192.

[5] Defard, T.; Setkov, A.; Loesch, A.; Audigier, R. PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization. International Conference on Pattern Recognition Workshops (ICPR-W). 2021; pp 475–489.

[6] Deng, H.; Li, X. Anomaly Detection via Reverse Distillation from One-Class Embedding. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2022; pp 9737–9746.

[7] Batzner, K.; Heckler, L.; König, R. EficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies. Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). 2024; pp 128–138.

[8] Ho, J.; Jain, A.; Abbeel, P. Denoising Difusion Probabilistic Models. Advances in Neural Information Processing Systems (NeurIPS). 2020; pp 6840–6851.

[9] Nichol, A. Q.; Dhariwal, P. Improved Denoising Difusion Probabilistic Models. Proceedings of the 38th International Conference on Machine Learning (ICML). 2021; pp 8162–8171.

[10] Song, J.; Meng, C.; Ermon, S. Denoising Difusion Implicit Models. International Conference on Learning Representations (ICLR). 2021.

[11] Wyatt, J.; Leach, A.; Schmon, S. M.; Willcocks, C. G. AnoDDPM: Anomaly Detection with Denoising Difusion Probabilistic Models Using Simplex Noise. IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). 2022; pp 650–656.

[12] Aggarwal, C. C.; Sathe, S. Theoretical Foundations and Algorithms for Outlier Ensembles. ACM SIGKDD Explor. Newsl. 2015, 17, 24–47.

[13] He, K.; Zhang, X.; Ren, S.; Sun, J. Deep Residual Learning for Image Recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2016; pp 770–778.

[14] Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; Polosukhin, I. Attention Is All You Need. Advances in Neural Information Processing Systems (NeurIPS). 2017; pp 5998–6008.

[15] Saito, T.; Rehmsmeier, M. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE 2015, 10, e0118432.

[16] Chicco, D.; Jurman, G. The Advantages of the Matthews Correlation Coeficient (MCC) over F1 Score and Accuracy in Binary Classification Evaluation. BMC Genomics 2020, 21, 6.

[17] Efron, B.; Tibshirani, R. J. An Introduction to the Bootstrap; Chapman & Hall/CRC: New York, 1994.

[18] DeLong, E. R.; DeLong, D. M.; Clarke-Pearson, D. L. Comparing the Areas under Two or More Correlated Receiver Operating Characteristic Curves: A Nonparametric Approach. Biometrics 1988, 44, 837–845.

[19] McNemar, Q. Note on the Sampling Error of the Diference between Correlated Proportions or Percentages. Psychometrika 1947, 12, 153–157.

[20] van der Maaten, L.; Hinton, G. Visualizing Data Using t-SNE. J. Mach. Learn. Res. 2008, 9, 2579–2605.

[21] Wang, J.; Xu, C.; Yang, Z.; Zhang, J.; Li, X. Deformable Convolutional Networks for Eficient Mixed-Type Wafer Defect Pattern Recognition. IEEE Transactions on Semiconductor Manufacturing 2020, 33, 587–596.