# DECODER DESIGN MATTERS FOR ECG DELINEATION

Joseph Scharpf<sup>∗1</sup>, William Han<sup>∗1</sup>, Chaojing Duan<sup>2</sup>, Michael A. Rosenberg<sup>3</sup>, Emerson Liu<sup>2</sup>, Ding Zhao<sup>1</sup>

<sup>1</sup>Carnegie Mellon University, <sup>2</sup>Allegheny Health Network, <sup>3</sup>University of Colorado

## ABSTRACT

Electrocardiogram (ECG) delineation identifies the boundaries of P waves, QRS complexes, and T waves, providing structural annotations that can guide AI models in learning to interpret ECGs. However, training accurate delineation models requires manual annotations that are scarce and timeconsuming to obtain. Recent work addresses this limitation through semi-supervised learning (SSL), but the design of the architecture, particularly the decoder, has received less attention. To this end, we propose R-U-Net, an ECG delineation model that pairs a ResNet-18 encoder with a U-Net decoder. On SemiSegECG, R-U-Net outperforms the strongest evaluated ResNet-18 + fully convolutional network (FCN) head baseline in each of the 16 in-domain settings by 3.3–13.0 mIoU and achieves 82.6 mIoU in the cross-domain setting, an improvement of 8.1 mIoU. Controlled ablations show that decoder design contributes more to performance gains than the evaluated SSL methods, motivating further exploration of architectures for ECG delineation. All code is open-source at github.com/ELM-Research/ECG-Delineation.

Index Terms— Electrocardiograms, ECG Delineation, Deep Learning, Semi-Supervised Learning

## 1. INTRODUCTION

Applying artificial intelligence (AI) to interpret ECGs is a step towards scalability and automation. While most recent works largely focus on tasks such as classification [1] and ECG-conditioned language generation [2], ECG delineation has been a growing area of interest [3].

ECG delineation is the task of partitioning the ECG in time into four categories: (1) background, (2) P wave, (3) QRS complex, and (4) T wave. Early automated ECG delineation methods relied on wavelet transforms and handdesigned rules [4]. More recent approaches formulate delineation as dense sample-level labeling using architectures such as CNN-LSTM models [5] and 1D U-Net models [6]. Waveform annotations provide explicit ECG structure and granular supervision for downstream tasks such as ECG-conditioned language generation [7]. However, training accurate delineation models requires manual boundary annotations that are scarce and time-consuming to obtain.

![](images/21ffc84130da21138ab7c27f5a010ee773af5d70c080663c260e1bb6a58141d8.jpg)  
Fig. 1. A high-level architectural overview of R-U-Net.

To address this limitation, SemiSegECG [3] benchmarks semi-supervised learning (SSL) approaches for ECG delineation across six datasets. Its comparison primarily focuses on SSL strategies, pairing ResNet [8] and ViT [9] encoders with a lightweight fully convolutional network (FCN) head. Although U-Net architectures have previously been applied to ECG delineation, the contribution of decoder design within this benchmark remains unexplored.

In this study, we investigate decoder design for ECG delineation through R-U-Net, which pairs a ResNet-18 encoder with a U-Net decoder [10] in place of the FCN head. R-U-Net outperforms the evaluated baselines in all 16 in-domain settings and the cross-domain setting of SemiSegECG [3]. To isolate the contribution of decoder design, we conduct two ablation studies: (1) comparing decoder variants (Table 2) and (2) comparing SSL approaches (Table 3). In both studies, we find that the U-Net decoder contributes most to the performance gains. Altogether, these experiments highlight decoder design as a key factor in ECG delineation performance and motivate closer attention to architectural details.

![](images/2e36692694f4b580ff0d07e935478fd21ad0312cd01fe2f835836b2762428374.jpg)  
Fig. 2. Example of successful ECG delineation by R-U-Net compared with the ground truth, showing a 2-second segment from the middle of a 10-second ECG.

## 2. METHOD

## 2.1. Problem Formulation

We follow SemiSegECG and formulate ECG delineation as sample-wise classification. For a single-lead segment $\boldsymbol { x } \in \mathbb { R } ^ { \hat { T } }$ , the target $y \in \{ 0 , 1 , 2 , 3 \} ^ { T }$ assigns each sample to background, P wave, QRS complex, or T wave, respectively. Given labeled segments $\mathcal { D } _ { \mathrm { l } }$ and unlabeled segments ${ \mathcal { D } } _ { \mathrm { u } } ,$ we learn a model $f _ { \theta }$ whose output includes a class-wise softmax:

$$
f _ { \theta } ( x ) \in [ 0 , 1 ] ^ { 4 \times T } , \qquad { \hat { y } } _ { t } = \arg \operatorname* { m a x } _ { c \in \{ 0 , 1 , 2 , 3 \} } [ f _ { \theta } ( x ) ] _ { c , t } .\tag{1}
$$

The first and last samples of each contiguous run of a nonbackground class define its predicted onset and offset.

## 2.2. R-U-Net Architecture Details

R-U-Net combines a one-dimensional ResNet-18 encoder [8] with a U-Net-style decoder [10], both initialized from scratch. Figure 1 provides an overview of the architecture.

The encoder contains four residual stages, each comprising two basic residual blocks. Given an augmented input ${ \widetilde { x } } ,$ we retain the output of every stage:

$$
H _ { \mathrm { e n c } } = \left( H _ { \mathrm { e n c } } ^ { ( 1 ) } , \cdot \cdot \cdot , H _ { \mathrm { e n c } } ^ { ( 4 ) } \right) = E _ { \theta _ { \mathrm { e n c } } } ( \widetilde { x } ) ,
$$

where $H _ { \mathrm { e n c } } ^ { ( s ) } \in \mathbb { R } ^ { d _ { s } \times T _ { s } }$ . The stage widths are (64, 128, 256, 512), with corresponding temporal lengths (625, 313, 157, 79) for $T = 2 5 0 0$

Starting from the deepest encoder representation, the decoder progressively upsamples the features and concatenates them with the corresponding encoder outputs through skip connections. Each fusion stage applies two kernel-size-3 convolutions, each followed by batch normalization and ReLU. The three stages produce 256, 128, and 64 channels, respectively.

The final decoder features are passed to a segmentation head. Dropout with probability 0.1 is followed by a pointwise convolution that produces $C = 4$ class logits. The head then linearly interpolates these logits to the original signal length and applies a class-wise softmax to obtain the predicted class probabilities:

$$
\begin{array} { r } { f _ { \theta } (  { \widetilde { x } } ) = \mathrm { s o f t m a x } _ { \mathrm { c l a s s } } \left( \mathcal { U } _ { T } \left( D _ { \theta _ { \mathrm { d e c } } } ( H _ { \mathrm { e n c } } ) \right) \right) , } \end{array}
$$

where $D _ { \theta _ { \mathrm { d e c } } }$ comprises the decoder, dropout, and pointwise classifier, and $\boldsymbol { \mathcal { U } } _ { T }$ denotes linear interpolation to length T.

## 2.3. Boundary-Aware Mean Teacher

We use Mean Teacher [11] with a student $f _ { \theta }$ and a teacher $f _ { \bar { \theta } } .$ The teacher is initialized from the student and updated after each optimization step as ${ \bar { \theta } }  0 . 9 9 { \bar { \theta } } + 0 . 0 1 \theta .$ . It operates in evaluation mode without gradient updates. For each unlabeled segment, the teacher receives a weak view $x ^ { \mathrm { w } }$ obtained by random temporal resizing and padding or cropping. The student receives a strong view $x ^ { \mathrm { s } }$ with additional stochastic amplitude perturbations and powerline, white, or sinusoidal noise. The two views remain temporally aligned. Labeled segments and their masks undergo the same weak transformation.

For a labeled minibatch $B _ { \mathrm { l } } .$ , we use sample-wise crossentropy:

$$
\mathcal { L } _ { \mathrm { s u p } } = - \frac { 1 } { | \mathcal { B } _ { 1 } | T } \sum _ { ( \boldsymbol { x } , \boldsymbol { y } ) \in \mathcal { B } _ { 1 } } \sum _ { t = 1 } ^ { T } \log [ f _ { \theta } ( \boldsymbol { x } ) ] _ { \boldsymbol { y } _ { t } , t } ,\tag{2}
$$

where $( x , y )$ denotes an augmented segment and its aligned mask. For each unlabeled segment, write $q = f _ { \bar { \theta } } ( x ^ { \mathrm { w } } )$ and $p = f _ { \theta } ( x ^ { \mathrm { s } } )$ . Both consistency terms use the soft-target crossentropy $\begin{array} { r } { \ell _ { t } = - \sum _ { c = 0 } ^ { 3 } q _ { c , t } \log p _ { c , t } } \end{array}$

To emphasize potential waveform boundaries, we measure changes between adjacent teacher probability vectors:

$$
\delta _ { t } = \frac { 1 } { 2 } \sum _ { c = 0 } ^ { 3 } | q _ { c , t } - q _ { c , t - 1 } | , \qquad t = 2 , \ldots , T ,\tag{3}
$$

with $\delta _ { 1 } \ : = \ : 0$ . We spread these changes over a ±4-sample neighborhood and normalize over the full segment:

$$
e _ { t } = \operatorname* { m a x } _ { 1 \leq u \leq T \atop | u - t | \leq 4 } \delta _ { u } , \qquad b _ { t } = \frac { e _ { t } } { \operatorname* { m a x } ( 1 0 ^ { - 6 } , \operatorname* { m a x } _ { u } e _ { u } ) } .\tag{4}
$$

Let $a _ { t } = \operatorname* { m a x } _ { c } q _ { c , t }$ denote teacher confidence. Set $g = 1$ if its mean over the segment is at least 0.50, and $g = 0$ otherwise. The region and boundary weights are

$$
\begin{array} { l } { w _ { t } ^ { \mathrm { r } } = g ( 1 - b _ { t } ) \mathbf { 1 } [ a _ { t } \geq 0 . 8 0 ] , } \\ { w _ { t } ^ { \mathrm { b } } = g b _ { t } ( 1 + a _ { t } ) / 2 . } \end{array}\tag{5}
$$

The region term favors confident positions away from probability changes, including background. The boundary term

emphasizes these changes without the hard sample-wise confidence threshold.

For an unlabeled minibatch $\boldsymbol { B _ { \mathrm { u } } }$ , we normalize the two terms independently:

$$
\mathcal { L } _ { k } = \frac { \sum _ { x \in \mathcal { B } _ { \mathrm { u } } } \sum _ { t = 1 } ^ { T } w _ { t } ^ { k } \ell _ { t } } { \operatorname* { m a x } \Bigl ( 1 , \sum _ { x \in \mathcal { B } _ { \mathrm { u } } } \sum _ { t = 1 } ^ { T } w _ { t } ^ { k } \Bigr ) } , \quad k \in \{ \mathrm { r } , \mathrm { b } \} ,\tag{6}
$$

where the dependence of $w _ { t } ^ { k }$ and $\ell _ { t }$ on x is implicit. The student minimizes

$$
\mathcal { L } = \frac { 1 } { 2 } \left( \mathcal { L } _ { \mathrm { s u p } } + \mathcal { L } _ { \mathrm { r } } + \mathcal { L } _ { \mathrm { b } } \right) .\tag{7}
$$

## 3. EXPERIMENTAL SETTINGS

## 3.1. Datasets

We follow the public in-domain and merged cross-domain protocols of SemiSegECG [3], using its supplied training, validation, and test splits. LUDB [12], QTDB [13], ISP [14], and Zhejiang [15] provide delineation labels. Each lead is treated as an independent input. Under the in-domain protocol (Table 1), labeled and unlabeled data come from the same dataset. Random subsets comprising 1/16, 1/8, 1/4, or 1/2 of the training set serve as labeled data, while the entire training set serves as unlabeled data. Under the merged cross-domain protocol (Figure 3), the four labeled datasets are combined while preserving their original splits, and PTB-XL [16] serves as an external unlabeled dataset. We evaluate on the merged in-domain test set to assess performance when labeled and unlabeled training data come from different sources. Following SemiSegECG preprocessing, waveforms are resampled to 250 Hz, producing $T = 2 5 0 0$ samples, and processed with high-pass and low-pass filters at 0.67 and 40 Hz. Z-score normalization is applied to all model inputs.

## 3.2. Training and Evaluation

We train for 100 epochs using AdamW with a learning rate of $1 0 ^ { - 3 }$ and weight decay of 0.05. The learning rate increases linearly during the first 10 epochs and subsequently follows a cosine schedule toward $1 0 ^ { - 4 }$ . Each step uses 16 labeled and 16 unlabeled examples. We select the student checkpoint with the highest validation mean intersection-over-union (mIoU) and evaluate it on the test set. mIoU includes all four classes, including background. All mIoU results with standard deviations are from our experiments and are reported as the mean ± standard deviation across three random seeds.

## 4. RESULTS

## 4.1. In-Domain Evaluation

Table 1 presents the 16 in-domain evaluations on SemiSegECG. We compare R-U-Net trained using Boundary-aware MT with six baseline training methods using the ResNet-18 + FCN architecture, as reported in SemiSegECG [3]. R-U-Net achieves the highest mIoU across all four datasets and four labeled-data proportions, outperforming the strongest baseline in each setting by 3.3 to 13.0 mIoU. At the lowest labeleddata proportion (1/16), the improvements are 13.0, 8.9, 12.3, and 4.5 mIoU on LUDB, QTDB, ISP, and Zhejiang, respectively. These results demonstrate consistent improvements across datasets and levels of labeled-data availability, including settings with limited annotations.

Table 1. In-domain test mIoU (%) under varying labeleddata ratios. Baseline results are from SemiSegECG [3] and use a ResNet-18 + FCN. R-U-Net results are reported as mean±standard deviation over three seeds.
<table><tr><td rowspan="2">Method</td><td colspan="4">ResNet-18</td></tr><tr><td>1/16</td><td>1/8</td><td>1/4</td><td>1/2</td></tr><tr><td></td><td></td><td>LUDB</td><td></td><td></td></tr><tr><td>Scratch</td><td>67.3</td><td>71.3</td><td>72.9</td><td>74.1</td></tr><tr><td>MT</td><td>70.8</td><td>72.3</td><td>73.6</td><td>74.3</td></tr><tr><td>FixMatch</td><td>70.9</td><td>72.2</td><td>72.9</td><td>74.1</td></tr><tr><td>CPS</td><td>68.6</td><td>71.6</td><td>73.1</td><td>74.3</td></tr><tr><td>ReCo</td><td>71.5</td><td>72.5</td><td>73.1</td><td>73.9</td></tr><tr><td>ST++</td><td>69.2</td><td>71.6</td><td>73.7</td><td>74.5</td></tr><tr><td>R-U-Net</td><td>84.5±0.1</td><td>85.5±0.1</td><td>85.5±0.3</td><td>85.9±0.2</td></tr><tr><td></td><td></td><td>QTDB</td><td></td><td></td></tr><tr><td>Scratch</td><td>47.5</td><td>56.2</td><td>60.5</td><td>64.9</td></tr><tr><td>MT</td><td>47.8</td><td>44.8</td><td>63.0</td><td>66.7</td></tr><tr><td>FixMatch</td><td>46.7</td><td>53.3</td><td>58.2</td><td>66.3</td></tr><tr><td>CPS</td><td>53.2</td><td>57.1</td><td>64.7</td><td>68.0</td></tr><tr><td>ReCo</td><td>53.4</td><td>53.4</td><td>58.7</td><td>64.5</td></tr><tr><td>ST++</td><td>52.9</td><td>57.8</td><td>62.5</td><td>68.0</td></tr><tr><td>R-U-Net</td><td>62.3±3.0</td><td>67.7±0.5</td><td>75.1±0.1</td><td>76.8±0.6</td></tr><tr><td></td><td></td><td>ISP</td><td></td><td></td></tr><tr><td>Scratch</td><td>62.3</td><td>64.6</td><td>68.1</td><td>69.3</td></tr><tr><td>MT</td><td>64.2</td><td>65.9</td><td>68.0</td><td>69.1</td></tr><tr><td>FixMatch</td><td>63.1</td><td>64.4</td><td>67.9</td><td>68.9</td></tr><tr><td>CPS</td><td>64.4</td><td>66.1</td><td>68.5</td><td>69.4</td></tr><tr><td>ReCo</td><td>62.3</td><td>64.7</td><td>67.4</td><td>68.1</td></tr><tr><td>ST++</td><td>63.7</td><td>65.7</td><td>68.5</td><td>69.5</td></tr><tr><td>R-U-Net</td><td>76.7±0.2</td><td>77.9±0.2</td><td>79.3±0.2</td><td>79.9±0.1</td></tr><tr><td></td><td></td><td>Zhejiang</td><td></td><td></td></tr><tr><td>Scratch</td><td>76.7</td><td>78.9</td><td>80.7</td><td>82.3</td></tr><tr><td>MT</td><td>79.2</td><td>80.1</td><td>81.5</td><td>82.9</td></tr><tr><td>FixMatch</td><td>79.0</td><td>80.4</td><td>81.2</td><td>82.7</td></tr><tr><td>CPS</td><td>77.6</td><td>79.5</td><td>81.3</td><td>82.7</td></tr><tr><td>ReCo</td><td>77.6</td><td>78.9</td><td>79.7</td><td>80.6</td></tr><tr><td>ST++</td><td>78.8</td><td>79.8</td><td>81.9</td><td>82.9</td></tr><tr><td>R-U-Net</td><td>83.7±0.1</td><td>84.8±0.1</td><td>85.4±0.0</td><td>86.2±0.1</td></tr></table>

![](images/cece0f198b4dd2d1a9862a967f7dd951a4bdd92ad9175e8daee4dc545e1be8f0.jpg)  
Fig. 3. Cross-domain test mIoU on the merged in-domain test set. Baselines use an FCN decoder [3].

Table 2. Ablation study of decoder variants on the LUDB dataset [12] using a labeled-data proportion of 1/16.
<table><tr><td>Decoder</td><td>mIoU</td></tr><tr><td>FCN</td><td>67.3</td></tr><tr><td>Wide FCN</td><td>67.2±0.3</td></tr><tr><td>U-Net (w/o skip connections)</td><td>80.0±0.1</td></tr><tr><td>U-Net</td><td>81.8±0.2</td></tr></table>

## 4.2. Cross-Domain Evaluation

Figure 3 compares R-U-Net trained using Boundary-aware MT with six baseline training methods using the ResNet-18 + FCN architecture in the cross-domain setting. R-U-Net achieves the highest mIoU of 82.6, exceeding the strongest baseline, Scratch (74.5 mIoU), by 8.1 mIoU. These results show that R-U-Net maintains its performance advantage when labeled and unlabeled training data come from different sources, supporting its effectiveness beyond the in-domain training setting.

## 4.3. Comparing Decoder Variants

We evaluate two intermediate decoder variants: (1) Wide FCN, which adds an additional convolutional layer and increases the channel widths to match the parameter count of our U-Net decoder (∼1.03M), and (2) U-Net (w/o skip connections), which is the same U-Net decoder used in R-U-Net without skip connections. To isolate the effect of the decoder, we train Wide FCN, U-Net w/o skip connections, and U-Net with the scratch training method. Wide FCN achieves comparable performance to the original FCN (67.2 versus 67.3 mIoU). The U-Net decoder without skip connections achieves 80.0 mIoU, while adding skip connections improves mIoU by 1.8 points to 81.8. These results suggest that the gains primarily arise from the U-Net decoder, with skip connections providing a minor increase.

Table 3. Ablation study on applying different semisupervised training methods to R-U-Net. We conduct experiments on LUDB 1/16.
<table><tr><td>Architecture</td><td>Method</td><td>mIoU</td></tr><tr><td rowspan="5">ResNet-18 + FCN</td><td>Scratch</td><td>67.3</td></tr><tr><td>MT</td><td>70.8</td></tr><tr><td>FixMatch</td><td>70.9</td></tr><tr><td>CPS</td><td>68.6</td></tr><tr><td>ReCo ST++</td><td>71.5</td></tr><tr><td rowspan="5">R-U-Net</td><td></td><td>69.2</td></tr><tr><td>Scratch</td><td>81.8±0.2</td></tr><tr><td>MT</td><td>84.0±0.1</td></tr><tr><td>FixMatch</td><td>83.5±0.2</td></tr><tr><td>Boundary-aware MT (Ours)</td><td>84.5±0.1</td></tr></table>

## 4.4. Comparing SSL Approaches

Table 3 compares different training methods for R-U-Net with the ResNet-18 + FCN baselines on LUDB 1/16. Under scratch training, replacing the FCN decoder with the U-Net decoder increases mIoU from 67.3 to 81.8, a gain of 14.5 mIoU. Scratch R-U-Net also exceeds the strongest FCN baseline, ReCo (71.5 mIoU), by 10.3 mIoU. SSL provides smaller additional gains: FixMatch, standard MT, and boundary-aware MT achieve 83.5, 84.0, and 84.5 mIoU, respectively. Boundary-aware MT improves over scratch training by 2.7 mIoU and standard MT by only 0.5 mIoU. These results indicate that decoder design accounts for most of the improvement in this setting, with SSL providing modest additional gains.

## 5. CONCLUSION

In this paper, we introduce R-U-Net, which pairs a ResNet-18 encoder with a U-Net decoder for ECG delineation. Across the SemiSegECG benchmark, R-U-Net outperforms the evaluated ResNet-18 + FCN baselines in all 16 in-domain settings and the cross-domain setting, demonstrating consistent improvements across datasets and levels of labeled-data availability. Controlled ablations indicate that decoder design accounts for most of these gains. Increasing the FCN decoder’s parameter count provides no improvement, whereas the U-Net decoder substantially improves performance even without skip connections. Under scratch training, R-U-Net also exceeds the strongest evaluated FCN baseline by 10.3 mIoU points. Together, these findings highlight the importance of establishing strong architectural baselines when assessing SSL for ECG delineation, particularly under limited supervision. Future work could examine whether these decoder-level findings extend to other encoders and datasets, and develop SSL methods that further improve delineation when annotations are scarce.

## 6. COMPLIANCE WITH ETHICAL STANDARDS

This study retrospectively analyzed publicly available, deidentified ECG data from LUDB [12], QTDB [13], ISP [14], Zhejiang [15], and PTB-XL [16]. No additional ethical approval was required for this secondary analysis.

## 7. ACKNOWLEDGMENTS

This work was conducted in collaboration with the Mario Lemieux Center for Heart Rhythm Care at Allegheny General Hospital.

## 8. REFERENCES

[1] Pranav Rajpurkar, Awni Y. Hannun, Masoumeh Haghpanahi, Codie Bourn, and Andrew Y. Ng, “Cardiologistlevel arrhythmia detection with convolutional neural networks,” 2017.

[2] Yubao Zhao, Jiaju Kang, Tian Zhang, Puyu Han, and Tong Chen, “Ecg-chat: A large ecg-language model for cardiac disease diagnosis,” 2025.

[3] Minje Park, Jeonghwa Lim, Taehyung Yu, and Sunghoon Joo, “Semisegecg: A multi-dataset benchmark for semi-supervised semantic segmentation in ecg delineation,” in Proceedings of the 34th ACM International Conference on Information and Knowledge Management, New York, NY, USA, 2025, CIKM ’25, p. 5099–5104, Association for Computing Machinery.

[4] Juan Pablo Mart´ınez, Rute Almeida, Salvador Olmos, Ana Paula Rocha, and Pablo Laguna, “A wavelet-based ECG delineator: Evaluation on standard databases,” IEEE Transactions on Biomedical Engineering, vol. 51, no. 4, pp. 570–581, 2004.

[5] Abdolrahman Peimankar and Sadasivan Puthusserypady, “DENS-ECG: A deep learning approach for ECG signal delineation,” Expert Systems with Applications, vol. 165, 2021, Art. no. 113911.

[6] Guillermo Jimenez-Perez, Alejandro Alcaine, and Oscar Camara, “Delineation of the electrocardiogram with a mixed-quality-annotations dataset using convolutional neural networks,” Scientific Reports, vol. 11, 2021, Art. no. 863.

[7] Jungwoo Oh, Hyunseung Chung, Junhee Lee, Min-Gyu Kim, Hangyul Yoon, Ki Seong Lee, Youngchae Lee, Muhan Yeo, and Edward Choi, “Ecg-reasoningbenchmark: A benchmark for evaluating clinical reasoning capabilities in ecg interpretation,” 2026.

[8] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, “Deep residual learning for image recognition,” 2015.

[9] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” 2021.

[10] Olaf Ronneberger, Philipp Fischer, and Thomas Brox, “U-net: Convolutional networks for biomedical image segmentation,” 2015.

[11] Antti Tarvainen and Harri Valpola, “Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results,” in Advances in Neural Information Processing Systems, 2017, vol. 30.

[12] Alena I. Kalyakulina, Igor I. Yusipov, Victor A. Moskalenko, Alexander V. Nikolskiy, Konstantin A. Kosonogov, Grigory V. Osipov, Nikolai Yu. Zolotykh, and Mikhail V. Ivanchenko, “Ludb: a new open-access validation tool for electrocardiogram delineation algorithms,” 2020.

[13] Pablo Laguna, R.G. Mark, A. Goldberg, and G.B. Moody, “Database for evaluation of algorithms for measurement of qt and other waveform intervals in the ecg,” Computers in Cardiology, vol. 1997, pp. 673 – 676, 10 1997.

[14] Aram Avetisyan, Nikolas Khachaturov, Ariana Asatryan, Shahane Tigranyan, and Yury Markin, “Isp ecg delineation dataset,” June 2024.

[15] Jianwei Zheng, Guohua Fu, Kyle Anderson, Huimin Chu, and Cyril Rakovski, “A 12-lead ECG database to identify origins of idiopathic ventricular arrhythmia containing 334 patients,” Scientific Data, vol. 7, no. 1, pp. 98, 2020.

[16] Patrick Wagner, Nils Strodthoff, Ralf-Dieter Bousseljot, Dieter Kreiseler, Fatima I. Lunze, Wojciech Samek, and Tobias Schaeffter, “PTB-XL, a large publicly available electrocardiography dataset,” Scientific Data, vol. 7, no. 1, pp. 154, May 2020, Number: 1 Publisher: Nature Publishing Group.