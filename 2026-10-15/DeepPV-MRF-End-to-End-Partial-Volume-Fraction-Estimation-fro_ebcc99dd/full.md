# DeepPV-MRF: End-to-End Partial-Volume Fraction Estimation from MR Fingerprinting Time Series

Tianyi Ding<sup>1</sup>, Yang Gao<sup>1,2</sup>, Zhuang Xiong<sup>3</sup>, Feng Liu<sup>1</sup>, Martijn A. Cloos<sup>4,5</sup>,Jin Jin<sup>6</sup>, Hongfu Sun<sup>1,7</sup>

<sup>3</sup> Image X Institute, Sydney School of Health Sciences, Faculty of Medicine and Health, University of Sydney, NSW 2006, Australia

<sup>4</sup> Donders Institute for Brain Cognition and Behaviour, Radboud University, Nijmegen, Netherlands <sup>5</sup> Australian Institute for Bioengineering and Nanotechnology, University of Queensland, St Lucia, QLD, Australia

<sup>6</sup> Siemens Healthineers Pty Ltd, Brisbane, Queensland, Australia <sup>7</sup> School of Engineering, University of Newcastle, Callaghan, NSW 2308, Australia

Corresponding author: Hongfu Sun School of Engineering, University of Newcastle, Callaghan, NSW 2308, Australia Email: hongfu.sun@newcastle.edu.au

## Abstract

Background: Accurate partial volume (PV) estimation, i.e., subvoxel tissue fractions, from Magnetic Resonance Fingerprinting (MRF) remains challenging, as conventional methods such as dictionary matching or segmentation tools like FSL FAST rely on multi-stage iterative pipelines, where inaccuracies in early steps (e.g., parametric map estimation) can propagate and compromise the final tissue fraction maps.

Purpose: This study introduces DeepPV-MRF, a deep learning framework designed to directly map raw MRF signals to voxel-wise tissue fraction maps. The goal is to provide a more accurate and eficient alternative to existing iterative and dictionary-based PV estimation methods.

Methods: DeepPV-MRF employs a cascaded spatial-temporal neural network trained on subject-derived simulations generated through Bloch modeling with radial k-space undersampling. The integrates voxel-wise temporal encoding with spatial U-Net refinement to estimate CSF, GM, and WM fractions without intermediate T1/T2 maps. Performance was evaluated against PV dictionary matching (PV-DM), and PV pseudoinverse (PV-Pinv), using both simulated and in vivo data.

Results: The proposed method substantially outperformed competing methods across all acquisition settings. In single-shot simulations, the CSF/WM/GM MAEs were 0.0418/0.0744/0.0757, compared with 0.1281/0.2443/0.2958 for PV-DM and 0.2021/0.2793/0.2630 for PV-Pinv; the corresponding SSIM values were 0.960/0.916/0.922. The in vivo experiments qualitatively showed sharper anatomical boundaries and smoother tissue transitions, while the cross-shot analysis supported improved consistency across volunteers.

Conclusions: In conclusion, DeepPV-MRF ofers a robust, accurate, and eficient framework for direct partial volume estimation in MRF. By eliminating reliance on intermediate reconstructions and improving performance particularly in low-shot scenarios, the method addresses critical weaknesses of existing approaches. These results support the practical feasibility of DeepPV-MRF and motivate further validation before clinical deployment.

Keywords: Magnetic Resonance Fingerprinting, Partial Volume Estimation, Deep Learning, Tissue Quantification

## 1 Introduction

Quantitative MRI (qMRI) is a non-invasive imaging framework that enables numerical characterization of tissue microstructure<sup>1</sup>. Broadly, qMRI applications fall into two major categories: parametric mapping and tissue composition estimation. The former focuses on extracting intrinsic tissue properties such as longitudinal relaxation time (T<sub>1</sub>) and transverse relaxation time (T ), which serve as biophysical biomarkers for disease diagnosis. The latter aims to estimate metrics such as volume or thickness of tissue such as gray matter (GM), white matter (WM), and cerebrospinal fluid (CSF) in the brain.

While parametric maps provide valuable insights into tissue biophysics, they do not directly account for tissue mixing within a voxel <sup>2</sup>. Conventionally, tissue composition estimation assumes that each voxel contains a single homogeneous tissue type <sup>3</sup>. However, this assumption fails in regions where voxels span across anatomical boundaries. This limitation becomes particularly important in clinical brain imaging, where partial volume (PV) efects are prevalent due to the limited spatial resolution of MRF<sup>4,5,6</sup>. PV efects can substantially impair the accuracy of tissue quantification<sup>2</sup>, lesion size estimations <sup>7</sup>, and brain atrophy assessment <sup>8</sup>.

To characterize sub-voxel tissue heterogeneity, voxel-wise tissue fraction estimation—i.e., quantifying the relative proportions of GM, WM, and CSF in each voxel—is essential. Traditional approaches to PV estimation typically rely on regularized statistical models, such as Gaussian mixture models (GMMs) combined with expectation-maximization (EM) <sup>9,10,11,12</sup>, often applied to multi-contrast structural MRI. These methods assume that each tissue class exhibits a welldefined intensity distribution across modalities and that voxel intensities result from a linear mixture of pure tissue responses. While efective in structural imaging, these assumptions do not hold in quantitative techniques like Magnetic Resonance Fingerprinting (MRF), where voxel signals represent non-linear temporal evolutions rather than static intensities.

MRF<sup>13</sup> signal representation captures multi-parametric information in a single scan, making it an attractive candidate for direct tissue composition estimation. One early attempt to incorporate PV modeling into MRF is the partial volume dictionary matching (PV-DM) method <sup>3,13</sup>, which estimates tissue fractions through dictionary matching. While promising, this method inherits several limitations from the underlying dictionary-based paradigm. First, dictionary-based MRF reconstruction becomes computationally demanding as the size and resolution of the dictionary increase, especially for applications involving partial volume modeling where dense sampling of tissue parameter space may be necessary <sup>13,14,15,16</sup>. Second, PV-DM relies on a strong “pure-tissue” assumption, requiring manually defined reference clusters with fixed relaxation times for GM, WM, and CSF. This introduces subjectivity and operational complexity, making it dificult to deploy in automated or clinical pipelines. These challenges motivate the need for more flexible, data-driven approaches to sub-voxel tissue fraction estimation in the MRF context.

Recent studies have explored deep learning as an alternative to dictionary-based MRF reconstruction, demonstrating improved eficiency and scalability <sup>17,18,19,20,21,22,23,24</sup>. These methods typically train convolutional or transformer-based models to map MRF time-series signals to voxel-wise tissue parameters such as $T _ { 1 }$ and $T _ { 2 }$ . For example, DRONE<sup>17</sup> first introduced a fully connected neural network to directly regress quantitative maps from signal evolutions. SCQ<sup>18</sup> extended this approach using convolutional architectures. More recent transformer-based models<sup>21</sup> have shown further improvements in spatial consistency and generalization. However, most existing models are designed with a single-compartment model and do not explicitly estimate tissue fractions in partial volume voxels. While these models have demonstrated strong performance in parametric map reconstruction, they are not designed to resolve sub-voxel tissue mixing. As a result, partial volume efects remain unaddressed in most existing MRF-based deep learning pipelines, limiting their utility in tasks that require accurate tissue composition analysis.

This limitation motivates the need for models capable of explicitly estimating sub-voxel tissue fractions from MRF signals.

To this end, we present DeepPV-MRF, an end-to-end deep learning framework for directly estimating voxel-wise tissue fraction maps from aliased MRF signal evolutions. The goal is to estimate the relative fractions of CSF, GM, and WM within each voxel, without relying on intermediate $T _ { 1 }$ or $T _ { 2 }$ maps. Extending the methodology proposed in MRF-Mixer<sup>24</sup>, we construct a fully simulated training dataset in which each voxel is assigned reference tissue fractions derived from high-resolution anatomical segmentations and corresponding pure-tissue relaxometry estimated from in vivo acquisitions. Unlike conventional partial volume estimation techniques such as partial volume pseudoinverse (PV-Pinv) and PV-DM<sup>3</sup>, which depend on predefined pure tissue fingerprints and manually designed subdictionaries, our method learns the mapping from MRF signals to tissue fractions directly from data, removing the need for dictionary construction, signal matching, or parameter discretization.

## 2 Related Work

## 2.1 Magnetic Resonance Fingerprinting Parameter Mapping

MRF represents an approach to MRI data acquisition and analysis that enables simultaneous quantification of multiple tissue parameters through a single abbreviated scan <sup>13</sup>. During the dictionary matching (DM) process, the signal evolution observed in each voxel i, termed the tissue fingerprint, is represented as $s _ { i } \in \mathbb { C } ^ { 1 \times t }$ , comprising t time points. This fingerprint is systematically compared against a precomputed dictionary $D _ { \mathrm { M R F } } \in \mathbb { C } ^ { t \times n }$ , where each column $d _ { j } \in \mathbb { C } ^ { t }$ represents a signal evolution simulated from a unique combination of tissue parameters $\{ T _ { 1 } , T _ { 2 } \}$ . The best-matching dictionary entry $d _ { i , \mathrm { M R F \ m a t c h } }$ is identified as the one that maximizes the normalized dot product magnitude with the observed signal <sup>3</sup>. Mathematically, this matching procedure can be expressed as:

$$
d _ { i , \mathrm { M R F ~ m a t c h } } = \arg \operatorname* { m a x } _ { j = 1 , \dots , n } \left| \langle s _ { i } , d _ { j } \rangle \right| .\tag{1}
$$

The tissue parameters corresponding to the selected dictionary entry are then assigned to voxel i. To accelerate the DM process, many approaches have been proposed <sup>14,15,25,26</sup>.Recently, deep learning methodologies have been proposed to bypass the DM process altogether <sup>17,18,19,21,23,24</sup>. However, these deep learning approaches typically do not address the partial volume estimation.

## 2.2 Partial Volume Estimation

Conventional approaches to addressing the PV efect typically model the voxel intensity as a linear combination of signals from constituent tissue types, weighted by their fractional volumes within the voxel<sup>1,27</sup>. Mathematically, this relationship is described by:

$$
X _ { i } = \sum _ { k = 1 } ^ { K } f _ { i , k } x _ { i , k } , \quad \mathrm { s u b j e c t } \mathrm { t o } \quad \sum _ { k = 1 } ^ { K } f _ { i , k } = 1 ,\tag{2}
$$

where $X _ { i }$ denotes the observed intensity at voxel $i , x _ { i , k }$ represents the pure intensity of tissue type k at voxel i, $f _ { i , k }$ denotes the fractional volume of tissue type k within voxel $i ,$ and K indicates the total number of tissue types present.

Given a suficient number of independent data channels, fractional tissue volumes can be directly estimated by solving this linear system via a least squares approach.

## 2.3 Integrating Partial Volume Estimation with MRF

Within the MRF framework, PV estimation has been approached by modeling each voxel’s signal as a linear combination of pure tissue fingerprints. Let $s _ { i }$ denote the MRF signal at voxel i, and let $D _ { s u b } \in \mathbb { C } ^ { t \times m }$ represent a subdictionary of signal evolutions for m tissue types. Then, the signal can be modeled as:

$$
s _ { i } = D _ { s u b } \mathbf { w } _ { i } ,\tag{3}
$$

where ${ \bf w } _ { i }$ is the vector of tissue volume fractions. Based on this formulation, two representative approaches have been developed. The first, partial volume pseudoinverse $\left( \mathrm { P V  – P i n v } \right) ^ { 3 }$ , uses a pseudoinverse projection to estimate $\mathbf { w } _ { i }$ . The second, $\mathrm { P V - D M ^ { 3 } }$ , constructs a dictionary of signal evolutions corresponding to all possible tissue fraction combinations and applies conventional dictionary matching to recover $\mathbf { w } _ { i }$

While both methods provide a mathematically grounded approach to PV estimation in MRF, they depend heavily on predefined pure-tissue relaxation parameters and manual clustering for ROI selection. These assumptions complicate robust application across large datasets containing many subjects scanned on a variety of systems.

## 3 Methodology

As illustrated in Figure 1, DeepPV-MRF consists of two main modules: (1) a voxel-wise complexvalued multilayer perceptron (cMLP) that compresses the temporal fingerprint at each voxel into a low-dimensional latent representation, and (2) a spatial refinement module based on a U-Net architecture that predicts tissue fraction maps. To train the network, we generated a synthetic dataset using forward Bloch simulations based on subject-derived PV maps and estimated pure-tissue relaxation parameters. The dataset incorporates radial-arm k-space undersampling and complex-valued signal formation to closely mimic real-world MRF acquisitions. We validated the proposed method using both simulated and in vivo test data.

The remainder of this section is organized as follows: Section 3.1 details the network structure, Section 3.2 describes the synthetic data generation pipeline, Section 3.3 describes the training data, Section 3.4 outlines the synthetic testing data and in vivo data acquisition process, Section 3.5 covers the network training process, Section 3.6 presents the evaluation metrics, and Section 3.7 describes the baseline comparison methods.

## 3.1 Network Architecture

The overall structure of DeepPV-MRF follows a two-stage design derived from the MRF-Mixer framework<sup>24,28</sup>. The original MRF-Mixer was developed to jointly extract spatial and temporal features from aliased MRF data. In this study, we adapted and extended that architecture to explicitly target voxel-wise partial volume estimation. Figure 1 provides an overview of the network, including the complex-valued temporal encoder and spatial tissue-fraction decoder.

The network consisted of two sequential modules: a voxel-wise temporal encoder and a spatial refinement module. The encoder was implemented as a complex-valued multilayer perceptron (cMLP) that independently processes each voxel’s 200-point MRF fingerprint obtained after temporal compression via singular value decomposition (SVD). It comprised four complex-valued Conv2D layers with kernel size set to $1 \times 1$ , efectively functioning as point-wise MLPs that reduce the temporal dimension to 16 latent channels. The real and imaginary parts of the resulting features were then concatenated to form a 32-channel per-voxel embedding, which was passed to the spatial decoder.

![](images/9f86033aabf0781cd889ad083ae163c91da66e2a6818687309a377533e3d293d.jpg)  
Figure 1: (a) Overview of the DeepPV-MRF framework, including real and imaginary input channels and tissue-fraction (TF) prediction. (b) U-Net encoder structure with channel depths and block functions (left), and decoder structure for TF prediction (right).

The refinement module adopted a modified U-Net to incorporate spatial context and regress tissue fraction maps. The encoder comprised four downsampling blocks with increasing channel widths [32, 64, 128, 256], each consisting of a 3 × 3 convolution, batch normalization, ReLU activation, and 2 × 2 max pooling. The decoder was constructed symmetrically using transposed convolutions for upsampling and skip connections for spatial resolution recovery. A SoftMax activation was applied to the tissue fraction decoder output to enforce probabilistic constraints across CSF, GM, and WM fractions.

For completeness, we also evaluated an auxiliary-head variant that added B0 and PD prediction branches during training. These branches were introduced as an additional supervisory signal to regularize the shared encoder, but, as shown in the ablation study, they did not improve the primary PV estimation task under the current setting. Therefore, all main simulation and in vivo results use the no-auxiliary-head model. Overall, DeepPV-MRF enabled direct inference of PV maps from aliased MRF signals, without relying on intermediate parameter maps or dictionary matching.

## 3.2 Pre-processing Synthetic Dataset

To supervise voxel-level PV estimation, we constructed a synthetic dataset with reference PV maps and corresponding MRF signal evolutions (Figure 2 (a)). Although specialized phantoms can be designed with known tissue mixing<sup>29</sup>, acquiring accurate voxel-wise PV maps and puretissue relaxation parameters from in vivo human subjects remains extremely challenging in practice. Therefore, we adopted a forward simulation strategy that generates synthetic MRF data from tissue compositions and relaxation values derived from real subject anatomy. This provides a practical supervisory signal for network training while maintaining precise control over the signal formation process.

![](images/bdbdd6bdc29595edd5526f20dbff65a6843f36ca79388bcef45415ab9e6f2d79.jpg)  
Figure 2: (a) Synthetic data generation pipeline, (b) inversion recovery balanced steady-state free precession sequence, (c) radial subsampling trajectory, (d) random cropping, and (e) examples of the first four real and imaginary channels of the training input data.

Under this framework, the MRF signal $s _ { i } \in \mathbb { C } ^ { t }$ at voxel i was generated using the following linear mixture model:

$$
s _ { i } = \sum _ { l = 1 } ^ { m } w _ { i , l } d _ { l } = D _ { \mathrm { s u b } } \mathbf { w } _ { i } .\tag{4}
$$

which is mathematically equivalent to the partial volume dictionary matching (PV-DM) formulation<sup>3</sup>. Here, $w _ { i , l }$ represents the tissue fraction of type l (CSF, GM, or WM), and $d _ { l }$ denotes the pure-tissue fingerprint simulated by the Bloch equation. In this study, both $\mathbf { w } _ { i }$ and $D _ { \mathrm { s u b } }$ were treated as known during data generation, providing precise supervision for the learning model.

We began by extracting the tissue fraction maps from high-resolution ME-MP2RAGE datasets<sup>30</sup>. Specifically, FMRIB’s Automated Segmentation Tool $( \mathrm { F A S T } ) ^ { 3 1 }$ was applied to subject-specific $T _ { 1 }$ maps to segment each voxel into fractional volumes of CSF $( w _ { C S F } )$ , WM (w<sub>WM</sub>), and GM $\left( w _ { G M } \right)$ . These maps were treated as reference tissue composition maps $\mathbf { w } _ { i }$ at voxel i for supervision during synthetic data generation.

To estimate realistic pure-tissue T1 and T2 values, we identified high-confidence regions for each tissue type using a threshold $p \geq 0 . 9 9$ on the FSL FAST output:

$$
M _ { l } ( i ) = \left\{ { \begin{array} { l l } { 1 , } & { p _ { l } ( i ) \geq 0 . 9 9 } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } } \end{array} } . \right.\tag{5}
$$

These binary masks $M _ { l } ( i )$ were then used to compute average relaxation values.

Because quantitative T2 maps were not available in the source data, approximate T2 maps were heuristically generated from the available $\mathrm { T 2 ^ { * } }$ maps for MRF signal simulation. The $\mathrm { T 2 ^ { * } }$ maps were first scaled using a subject-level random factor sampled between 1.2 and 1.7, followed by empirical corrections to keep the resulting values within plausible simulation ranges. These corrections included enforcing $\mathrm { T 2 } < \mathrm { T 1 }$ , compressing unusually large T2 values $( \mathrm { i . e . , } > 1 0 0 0 ~ \mathrm { m s } )$ applying an additional global rescaling. The resulting ranges were selected to be broadly consistent with representative 3T brain relaxation values reported in prior studies <sup>32,33,34</sup>. This procedure was not intended as a tissue-specific physical conversion from $\mathrm { T 2 ^ { * } }$ to T2, but rather as a heuristic step to introduce subject-level variability and generate approximate relaxation parameters for synthetic training-data generation.

To introduce natural variability and prevent overfitting, smooth spatial perturbations were added to $\bar { T } 1 _ { l }$ and $\bar { T ^ { 2 } } _ { l }$ before Bloch simulation. These perturbations mimic intra-tissue heterogeneity and improve the realism of the synthetic signals.

Each perturbed pure-tissue relaxation pair $( \bar { T } 1 _ { l } , \bar { T } 2 _ { l } )$ was passed through Bloch simulation to generate the corresponding fingerprint:

$$
d _ { l } = P D \cdot \mathrm { B l o c h } ( \bar { T } 1 _ { l } , \bar { T } 2 _ { l } , B 0 , F A s , T R s ) .\tag{6}
$$

where PD is the voxel-wise proton density scaling factor at voxel i, flip angles $( F A s )$ and repetition times $( T R s )$ are the time-varying flip angles and repetition times, and B0 is the local of-resonance frequency. The same B0 value was applied to all tissue components within each voxel as a voxel-level approximation of the dominant local field condition. This assumption was adopted to keep the multi-component signal model identifiable and computationally tractable, while acknowledging that component-specific microscopic frequency diferences may exist in practice.

To emulate realistic acquisition conditions, we simulated undersampled MRF using radial k-space trajectories<sup>24,35</sup>, as shown in Figure 2(c). The image time series $S _ { \mathrm { i m a g e } }$ was transformed into k-space using non-uniform fast Fourier transform (NUFFT):

$$
S _ { \mathrm { k s p a c e \_ r a d i a l } } = \mathrm { N U F F T } ( S _ { \mathrm { i m a g e } } , u _ { \mathrm { t r j } } ) .\tag{7}
$$

and the aliased image series was reconstructed via:

$$
\begin{array} { r } { S _ { \mathrm { i m a g e \_ a l i a s e d } } = \mathrm { N U F F T ^ { - 1 } } ( S _ { \mathrm { k s p a c e \_ r a d i a l } } , u _ { \mathrm { t r j } } ) . } \end{array}\tag{8}
$$

The aliased MRF signal S<sub>image\_aliased</sub> served as the network input.

## 3.3 Synthetic Training Dataset

We applied the above procedure to ten subjects from the ME-MP2RAGE dataset <sup>30</sup> for training. All imaging procedures were approved by the institutional ethics committee, and written informed consent was obtained.

To reduce memory usage and accelerate training, we applied temporal compression using $\mathrm { S V D ^ { 1 4 } }$ , reducing the original 1000 time points to 200 dominant components. This choice was guided by prior MRF studies showing that, for bSSFP-based dictionaries, the first 200 singular components retain nearly all dominant signal energy <sup>14</sup>. Accordingly, we used 200 components as a robust and computationally eficient setting for evaluating the core feasibility of deep learning-based partial volume estimation from undersampled MRF signals, rather than treating SVD dimensionality itself as a primary optimization target in this study.

The SVD basis was computed once from the precomputed MRF dictionary rather than from the training or testing image data. The same fixed temporal basis was then applied to the training, synthetic testing, and in vivo datasets. Therefore, the temporal compression step was independent of the image-level training/testing split and did not introduce information leakage from the test subjects into model training.

The final training dataset contained 1200 paired samples of size 192 × 192, each consisting of compressed aliased MRF signals as input and the corresponding CSF, WM, and GM tissue fraction maps as labels. To enhance spatial diversity and improve generalization, we applied random cropping during training, as illustrated in Figure 2(d). Specifically, each original sample was randomly cropped five times, resulting in a total of 30,000 training samples of size $6 4 \times 6 4 \times 2 0 0$

## 3.4 Testing Dataset

To evaluate model performance under controlled conditions, we first constructed a synthetic test set using three subjects from the ME-MP2RAGE dataset <sup>30</sup>, which were not used during training. For each subject, we selected slices primarily located in the central brain region to ensure anatomical diversity. A total of 120 synthetic test samples were generated, each of size $1 9 2 \times 1 9 2$ . These samples were used exclusively during inference to provide quantitative comparisons of reconstruction accuracy across diferent models.

In addition to the synthetic test data, in vivo datasets were acquired from three healthy volunteers (sub 01, sub 02, sub 03). The sub 01 dataset was acquired on a 3 T Siemens Prisma scanner, whereas the sub 02 and sub 03 datasets were acquired after the scanner system was upgraded to a 3 T Siemens MAGNETOM Cima.X Fit platform. Each volunteer was scanned using single-shot, three-shot, and six-shot MRF acquisitions, resulting in nine in vivo datasets in total. Imaging was performed across 32 slices using a 1000-point IR-bSSFP sequence with varying radial undersampling strategies, an in-plane resolution of 1 mm, and a slice thickness of 3 mm. Written informed consent was obtained from all volunteers prior to participation. The trained models were directly applied to these datasets without retraining or fine-tuning, and the results were used to evaluate cross-shot consistency and multi-volunteer in vivo robustness across the expanded scanner/protocol setting.

## 3.5 Network Training

The DeepPV-MRF network was trained using the Adam optimizer on an NVIDIA GeForce RTX 5090 GPU for 100 epochs. All model parameters were initialized from a normal distribution with a mean of 0 and a standard deviation of 0.01, and the full training process required approximately one hour under the current implementation settings. After training, computational profiling of the final model showed that it contained 7.14 million trainable parameters and required approximately 201.81 GFLOPs for one 192 × 192 slice. On the same GPU, the inference time was 3.06 ms per slice, with peak GPU memory usage of 0.243 GB allocated and 0.303 GB reserved.

The network was trained by minimizing a tissue fraction estimation loss. This loss contains two terms: an $L _ { 1 }$ loss for voxel-wise tissue fraction accuracy and a spatial gradient consistency loss for preserving tissue boundaries:

$$
{ \mathcal { L } } _ { \mathrm { { P V } } } = { \mathcal { L } } _ { \mathrm { { L 1 } } } ( { \hat { W } } , W ) + \lambda _ { \mathrm { { s p a t i a l } } } \cdot { \mathcal { L } } _ { \mathrm { { s p a t i a l } } } ( { \hat { W } } , W ) ,\tag{9}
$$

where $\hat { W }$ and W denote the predicted and reference tissue fraction maps, respectively. The $L _ { 1 }$ loss promotes voxel-wise accuracy and is defined as:

$$
\mathcal { L } _ { \mathrm { L 1 } } ( \hat { W } , W ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left| \hat { W } _ { i } - W _ { i } \right| .\tag{10}
$$

To enforce spatial smoothness, we used a first-order gradient consistency loss:

$$
\begin{array} { r } { X _ { x , i , j } = \left| \nabla _ { x } \hat { W } _ { i , j } - \nabla _ { x } W _ { i , j } \right| , } \end{array}\tag{11}
$$

$$
\boldsymbol { X _ { y , i , j } } = \left| \nabla _ { y } \hat { W } _ { i , j } - \nabla _ { y } W _ { i , j } \right| ,\tag{12}
$$

$$
\mathcal { L } _ { \mathrm { s p a t i a l } } ( \hat { W } , W ) = \frac { 1 } { N } \sum _ { i , j } \left( X _ { x , i , j } + X _ { y , i , j } \right) ,\tag{13}
$$

where $\nabla _ { x }$ and $\nabla _ { y }$ denote finite diference approximations of spatial gradients in the horizontal and vertical directions, respectively.

The hyperparameter $\lambda _ { \mathrm { s p a t i a l } }$ controls the trade-of between zeroth-order voxel-wise fidelity and first-order spatial gradient consistency. In the main experiments, we used $\lambda _ { \mathrm { s p a t i a l } } = 0 . 0 5$ as a default setting because it provided stable training and satisfactory reconstruction quality during model development. As shown later in the ablation study, performance remained relatively stable across a range of values.

For the auxiliary-head ablation only, the objective was extended as:

$$
\mathcal { L } _ { \mathrm { T o t a l } } ^ { \mathrm { a u x } } = \mathcal { L } _ { \mathrm { P V } } + \mathcal { L } _ { B 0 } + \mathcal { L } _ { P D } ,\tag{14}
$$

where $\mathcal { L } _ { B 0 } = \mathrm { M S E } ( \hat { B 0 } , B 0 )$ and $\mathcal { L } _ { P D } = \mathrm { M S E } ( \hat { P D } , P D )$ represent voxel-wise mean squared errors between predicted and reference B0 and PD maps, respectively.

## 3.6 Evaluation Metrics

To evaluate estimation performance, we employed voxel-wise metrics to reflect numerical accuracy. These quantitative voxel-wise metrics were computed for the synthetic test data, where reference tissue fraction maps were available. For the in vivo data, no voxel-wise reference PV maps were available; therefore, metrics requiring reference tissue fractions were not computed in that setting. Instead, an expert manually delineated the brain contour in ImageJ <sup>36</sup> to generate a brain mask, which was used to restrict the analysis and visualization to the brain region.

## 3.6.1 Voxel-wise Accuracy Metrics

We used the following voxel-wise metrics to quantify the accuracy of the estimated tissue fraction maps on the synthetic test set: root mean square error (RMSE), tissue-specific mean absolute error (MAE), peak signal-to-noise ratio (PSNR), structural similarity index (SSIM), and weighted Pearson correlation coeficient $( r _ { \mathrm { w e i g h t e d } } )$ . These metrics collectively assess both numerical fidelity (RMSE, MAE, PSNR) and structural similarity (SSIM). Because these quantities require voxelwise reference tissue fractions, they were not computed for the in vivo data.

For a given tissue $t \in \{ \mathrm { C S F } , \mathrm { W M } , \mathrm { G M } \}$ , let $w _ { t } ( i )$ denote the reference fraction of tissue t at voxel i, and let $y _ { \mathrm { p r e d } , t , i }$ and $y _ { \mathrm { t r u e } , t , i }$ denote the predicted and reference values, respectively. The tissue-specific partial-volume mask was defined as

$$
\Omega _ { P V , t } = \{ i \mid 0 < w _ { t } ( i ) < 1 , \ i \in \Omega _ { \mathrm { b r a i n } } \} ,\tag{15}
$$

where $\Omega _ { \mathrm { b r a i n } }$ denotes the brain mask, and $N _ { P V , t }$ is the number of voxels in $\Omega _ { P V , t }$

• Tissue-specific Mean Absolute Error (MAE): The tissue-specific MAE was evaluated on the corresponding partial-volume region to assess accuracy in transition areas:

$$
\mathrm { M A E } _ { t } = \frac { 1 } { N _ { P V , t } } \sum _ { i \in \Omega _ { P V , t } } \left| y _ { \mathrm { p r e d } , t , i } - y _ { \mathrm { t r u e } , t , i } \right| .\tag{16}
$$

In addition to the PV-restricted tissue-specific MAE, we also report whole-brain metrics computed over all voxels within the brain mask to facilitate comparison with related work in Supplementary Table 2.

• Weighted Pearson Correlation $( r _ { \mathbf { w e i g h t e d } } )$ : We computed a tissue-specific weighted Pearson correlation on $\Omega _ { P V , t }$ , with weights defined by

$$
\omega _ { t , i } = w _ { t } ( i ) ( 1 - w _ { t } ( i ) ) .\tag{17}
$$

The weighted correlation was then computed as

$$
\tilde { y } _ { t , i } = y _ { \mathrm { t r u e } , t , i } - \bar { y } _ { \mathrm { t r u e } , t } ,\tag{18}
$$

$$
\hat { y } _ { t , i } = y _ { \mathrm { p r e d } , t , i } - \bar { y } _ { \mathrm { p r e d } , t } ,\tag{19}
$$

$$
\begin{array} { r } { r _ { \mathrm { w e i g h t e d } , t } = \frac { - \frac { \sum \substack { \omega \leq - s - \nu , \iota } } { \nu } } { \sqrt { \sum _ { i \in \Omega _ { P V , t } } \omega _ { t , i } \tilde { y } _ { t , i } ^ { 2 } } \sqrt { \sum _ { i \in \Omega _ { P V , t } } \omega _ { t , i } \hat { y } _ { t , i } ^ { 2 } } } . } \end{array}\tag{20}
$$

In addition to quantitative evaluations, we generated voxel-wise error maps for qualitative analysis, enabling visual inspection of regional estimation errors and anatomical plausibility.

## 3.7 Comparison with Other Methods

To comprehensively assess the performance of the proposed method, we conducted comparative analyses against several established approaches, including $\mathrm { P V - P i n v ^ { 3 } }$ , and $\mathrm { P V - D M ^ { 3 } }$ . Because these competing methods operate on diferent signal representations and require diferent prior information, we explicitly distinguish below between matched and unmatched comparison conditions.

In the simulation experiments, DeepPV-MRF, PV-Pinv, and PV-DM were evaluated using the same synthetic MRF inputs generated from the same underlying tissue composition and relaxation parameter framework. For PV-Pinv and PV-DM, the required dictionaries $D _ { s u b }$ and $D _ { P V }$ were constructed from the same reference tissue parameters used in the simulation setup. Therefore, the simulation comparison among these MRF-based methods was performed under closely matched input conditions and parameter assumptions.

For the in vivo experiment, some diferences in preconditions were unavoidable because the methods operate in diferent input domains. Following the original PV-DM study<sup>3</sup>, we first generated T1 and T2 maps through dictionary matching and then applied k-means clustering with $k = 1 1$ to select representative T1 and T2 values for constructing $D _ { s u b }$ and $D _ { P V }$ for PV-Pinv and PV-DM. The value k = 11 was adopted from the original PV-DM implementation rather than tuned in this study. The resulting cluster centers were used as representative T1/T2 parameter candidates spanning the observed tissue distribution, not as eleven final tissue classes. Candidate centers corresponding to CSF, GM, and WM were then used to construct the three-component PV dictionary for tissue fraction estimation. In contrast, DeepPV-MRF directly used aliased MRF time-series signals as input.

Accordingly, the comparison with PV-Pinv and PV-DM can be regarded as an MRF-based methodological comparison under closely aligned simulation settings. We have made these distinctions explicit to clarify both the fairness and the limitations of the comparative evaluation.

![](images/8e0a24ae013e9b647e295cb279137c9ffc192c360a8e1e7ff1cde1c0f1c05697.jpg)  
Figure 3: Qualitative comparison of tissue fraction maps for cerebrospinal fluid (CSF), white matter (WM), and gray matter (GM) under single-shot, three-shot, and six-shot acquisitions in the synthetic test set. For each tissue type and acquisition setting, four methods are compared: Reference Label, partial volume dictionary matching (PV-DM), partial volume pseudoinverse (PV-Pinv), and the proposed DeepPV-MRF. Corresponding absolute diference maps are shown in the second row for each acquisition setting to visualize voxel-wise reconstruction errors. All maps are displayed using a common intensity window of [0, 1] for tissue fractions and [0, 0.2] for diference maps.

## 4 Results

## 4.1 Simulation Results

Figure 3 shows a qualitative comparison of partial volume estimation results from PV-DM, PV-Pinv, and DeepPV-MRF across single-shot, three-shot, and six-shot acquisitions. For each method, CSF, WM, and GM tissue fraction maps are shown along with their corresponding diference maps against the reference labels. For all shot settings, DeepPV-MRF consistently showed the best image contrasts compared with the reference labels, especially around tissue boundaries and regions with partial volume efects. The improvement is most noticeable in the single-shot case, where PV-DM and PV-Pinv showed clear structural errors, as was confirmed in Table 1 which summarized the quantitative comparisons of diferent methods. Particularly, DeepPV-MRF outperformed both PV-DM and PV-Pinv on all metrics. For example, under the single-shot setting, DeepPV-MRF achieved lower MAE values $( 4 1 . 7 6 \times 1 0 ^ { - 3 }$ for CSF, $7 4 . 3 9 \times 1 0 ^ { - 3 }$ for WM, and $7 5 . 6 9 \times 1 0 ^ { - 3 }$ for GM) and higher SSIM scores (0.960, 0.916, and 0.922 respectively). Figure 4 shows scatter plots of predicted PV values of DeepPV-MRF against the reference labels. As the number of shots increases from 1 to 6, the data points become more concentrated around the diagonal line, indicating better agreement with the reference labels. Correlation values and trend line slopes also increased accordingly. For the 6-shot case, DeepPV-MRF achieved MAE values of $2 2 . 9 3 \times 1 0 ^ { - 3 }$ (CSF), $4 3 . 1 6 \times 1 0 ^ { - 3 }$ (WM), and $4 3 . 9 6 \times 1 0 ^ { - 3 }$ (GM), with PSNR values reaching 35.12 dB (CSF), 29.35 dB (WM), and 28.22 dB (GM). These results confirm that DeepPV-MRF provides more accurate and stable partial volume estimation under diferent acquisition settings, especially in low-shot scenarios.

To further assess statistical robustness, paired non-parametric statistical tests were performed across the synthetic test slices. The Holm-adjusted Friedman tests and post-hoc paired Wilcoxon signed-rank tests are reported in Supplementary Table 1. These results support the quantitative

Table 1: Performance comparison of diferent tissue fraction estimation methods under diferent shot numbers using simulated magnetic resonance fingerprinting (MRF) signals. Quantitative results are reported as mean ± standard deviation across all synthetic test slices pooled from the 3 unseen test subjects.
<table><tr><td></td><td>Tissue</td><td>Method</td><td> $\mathbf { M A E } \left( \times 1 0 ^ { 3 } \right) \left( \downarrow \right)$ </td><td>PSNR (↑)</td><td>SSIM (↑)</td><td>rweighted (↑)</td></tr><tr><td rowspan="5">Siun-hot GM</td><td rowspan="4">CSF</td><td>DeepPV-MRF</td><td> ${ \bf 4 1 . 7 6 \pm 5 . 3 9 }$ </td><td> $\mathbf { 3 0 . 8 1 \ : \pm { \ : 1 . 3 7 } }$ </td><td> $\mathbf { 0 . 9 6 0 \ : \pm { \ : 0 . 0 1 4 } }$ </td><td> $\mathbf { 0 . 9 5 7 \pm 0 . 0 1 3 }$ </td></tr><tr><td>PV-DM</td><td> $1 2 8 . 0 7 \pm 2 8 . 6 0$ </td><td> $1 9 . 9 3 \pm 1 . 7 2$ </td><td> $0 . 6 4 3 \pm 0 . 0 8 1$ </td><td> $0 . 7 1 \pm 0 . 1 0 3$ </td></tr><tr><td>PV-Pinv</td><td> $2 0 2 . 1 0 \pm 3 5 . 9 6$ </td><td> $1 5 . 2 2 \pm 1 . 2 7$ </td><td> $0 . 3 9 6 \pm 0 . 0 7 6$ </td><td> $0 . 3 3 8 \pm 0 . 3 1 4$ </td></tr><tr><td>DeepPV-MRF</td><td> $\mathbf { 7 4 . 3 9 \ : \pm { \ : 1 0 . 6 5 } }$ </td><td> ${ \bf 2 4 . 8 0 \pm 1 . 3 8 }$ </td><td> $\mathbf { 0 . 9 1 6 \ : \pm { \ : 0 . 0 1 8 } }$ </td><td> $\mathbf { 0 . 8 4 6 \ : \pm { \ : 0 . 0 3 4 } }$ </td></tr><tr><td rowspan="4">WM</td><td>PV-DM</td><td> $2 4 4 . 2 5 \pm 3 5 . 0 3$ </td><td> $1 2 . 5 7 \pm 1 . 2 2$ </td><td> $0 . 5 2 7 \pm 0 . 0 3 9$ </td><td> $0 . 4 1 4 \pm 0 . 0 9 6$ </td></tr><tr><td>PV-Pinv</td><td> $2 7 9 . 2 7 \pm 2 6 . 8 9$ </td><td> $1 1 . 7 6 \pm 0 . 7 9$ </td><td> $0 . 4 4 8 \pm 0 . 0 4 7$ </td><td> $0 . 3 4 6 \pm 0 . 1 0 5$ </td></tr><tr><td>DeepPV-MRF</td><td> ${ \bf 7 5 . 6 9 \pm 9 . 4 1 }$ </td><td> $\mathbf { 2 3 . 8 4 \ : \pm 1 . 3 3 }$ </td><td> $\mathbf { 0 . 9 2 2 \pm 0 . 0 1 4 }$ </td><td> $\mathbf { 0 . 8 7 9 \ : \pm { \ : 0 . 0 2 6 } }$ </td></tr><tr><td>PV-DM</td><td> $2 9 5 . 8 2 \pm 5 4 . 1 0$ </td><td> $1 1 . 5 1 \pm 1 . 4 6$ </td><td> $0 . 5 2 5 \pm 0 . 0 7 3$ </td><td> $0 . 3 3 7 \pm 0 . 1 1 6$ </td></tr><tr><td rowspan="6">Thee-ot GM</td><td rowspan="4">CSF</td><td>PV-Pinv</td><td> $2 6 3 . 0 2 \pm 2 8 . 9 7$ </td><td> $1 2 . 0 3 \pm 0 . 6 7$ </td><td> $0 . 4 7 \pm 0 . 0 6 6$ </td><td> $0 . 3 2 5 \pm 0 . 1 3 8$ </td></tr><tr><td>DeepPV-MRF</td><td> ${ \bf 2 5 . 2 6 \pm 6 . 2 6 }$ </td><td> ${ \bf 3 4 . 4 8 \pm 2 . 0 4 }$ </td><td> $\mathbf { 0 . 9 7 8 \pm 0 . 0 1 1 }$ </td><td> $\mathbf { 0 . 9 8 1 \pm 0 . 0 0 7 }$ </td></tr><tr><td>PV-DM</td><td> $1 1 3 . 9 1 \pm 2 8 . 9 1$ </td><td> $2 1 . 1 7 \pm 1 . 8 9$ </td><td> $0 . 6 9 4 \pm 0 . 0 9 2$ </td><td> $0 . 7 7 2 \pm 0 . 0 9 6$ </td></tr><tr><td>PV-Pinv</td><td> $1 7 3 . 5 4 \pm 1 8 . 6 2$ </td><td> $1 6 . 2 7 \pm 0 . 8 7$ </td><td> $0 . 4 6 2 \pm 0 . 0 4 7$ </td><td> $0 . 5 7 4 \pm 0 . 1 3 3$ </td></tr><tr><td rowspan="4">WM PV-DM</td><td>DeepPV-MRF</td><td> ${ \bf 5 3 . 4 2 \pm 8 . 7 4 }$ </td><td> ${ \bf 2 7 . 6 9 \pm 1 . 3 8 }$ </td><td> $\mathbf { 0 . 9 4 7 \ : \pm { \ : 0 . 0 1 1 } }$ </td><td> $\mathbf { 0 . 9 2 4 \ : \pm { \ : 0 . 0 1 6 } }$ </td></tr><tr><td>PV-Pinv</td><td> $2 0 4 . 9 7 \pm 3 7 . 8 8$   $2 6 6 . 4 9 \pm 3 7 . 6 0$ </td><td> $1 4 . 0 2 \pm 1 . 5 9$ </td><td> $0 . 5 8 6 \pm 0 . 0 4 5$ </td><td> $0 . 5 5 7 \pm 0 . 1 2 7$ </td></tr><tr><td>DeepPV-MRF</td><td></td><td> $1 2 . 2 8 \pm 1 . 2 5$ </td><td> $0 . 4 7 2 \pm 0 . 0 5 8$ </td><td> $0 . 4 0 9 \pm 0 . 1 3 7$ </td></tr><tr><td>PV-DM</td><td> ${ \bf 5 2 . 6 3 \pm 9 . 6 1 }$   $2 5 6 . 9 1 \pm 6 3 . 7 3$ </td><td> $\mathbf { 2 6 . 6 4 \ : \pm { \ : 1 . 4 6 } }$ </td><td> $\mathbf { 0 . 9 4 6 \ : \pm { \ : 0 . 0 1 3 } }$ </td><td> $\mathbf { 0 . 9 3 5 \ : \pm 0 . 0 1 9 }$ </td></tr><tr><td rowspan="4"></td><td>PV-Pinv</td><td> $2 7 2 . 0 8 \pm 4 0 . 9 4$ </td><td> $1 2 . 8 6 \pm 1 . 9 9$   $1 2 . 1 0 \pm 1 . 1 3$ </td><td> $0 . 5 9 2 \pm 0 . 0 8 9$   $0 . 4 5 1 \pm 0 . 0 9 0$ </td><td> $0 . 4 3 1 \pm 0 . 1 4 9$   $0 . 2 9 7 \pm 0 . 1 8 0$ </td></tr><tr><td rowspan="3">CSF</td><td></td><td></td><td></td><td></td><td> $\mathbf { 0 . 9 8 4 \ : \pm { \ : 0 . 0 0 7 } }$ </td></tr><tr><td>DeepPV-MRF PV-DM</td><td> ${ \bf 2 2 . 9 3 \pm 4 . 9 3 }$   $1 1 0 . 9 1 \pm 2 9 . 5 1$ </td><td> $\mathbf { 3 5 . 1 2 \ : \pm { \ : 1 . 8 0 } }$   $2 1 . 4 1 \pm 1 . 9 5$ </td><td> $\mathbf { 0 . 9 8 0 \ : \pm { \ : 0 . 0 1 0 } }$ </td><td></td></tr><tr><td>PV-Pinv</td><td> $1 7 1 . 6 4 \pm 1 9 . 5 5$ </td><td> $1 6 . 3 4 \pm 0 . 9 2$ </td><td> $0 . 7 0 9 \pm 0 . 0 9 4$   $0 . 4 6 8 \pm 0 . 0 5 0$ </td><td> $0 . 7 8 2 \pm 0 . 0 9 6$ </td></tr><tr><td rowspan="4">Six--hot</td><td></td><td></td><td></td><td></td><td></td><td> $0 . 5 8 4 \pm 0 . 1 3 6$ </td></tr><tr><td rowspan="3">WM</td><td> $\mathrm { D e e p P V - M R F }$ </td><td> ${ \bf 4 3 . 1 6 \pm 6 . 2 5 }$ </td><td> ${ \bf 2 9 . 3 5 \pm 1 . 3 9 }$ </td><td> $\mathbf { 0 . 9 5 4 \ : \pm { \ : 0 . 0 1 1 } }$ </td><td> $\mathbf { 0 . 9 4 6 \ : \pm { \ : 0 . 0 1 2 } }$ </td></tr><tr><td>PV-DM</td><td> $1 9 4 . 3 9 \pm 3 6 . 4 1$ </td><td> $1 4 . 4 4 \pm 1 . 5 9$ </td><td> $0 . 6 0 6 \pm 0 . 0 4 2$ </td><td> $0 . 6 0 1 \pm 0 . 1 2 0$ </td></tr><tr><td>PV-Pinv</td><td> $2 6 3 . 5 8 \pm 3 9 . 6 8$ </td><td> $1 2 . 4 1 \pm 1 . 3 6$ </td><td> $0 . 4 8 2 \pm 0 . 0 6 1$ </td><td> $0 . 4 3 0 \pm 0 . 1 4 5$ </td></tr><tr><td rowspan="3">GM</td><td>DeepPV-MRF</td><td> ${ \bf 4 3 . 9 6 \pm 6 . 9 7 }$ </td><td> ${ \bf 2 8 . 2 2 \pm 1 . 3 6 }$ </td><td> $\mathbf { 0 . 9 5 8 \ : \pm { \ : 0 . 0 1 1 } }$ </td><td></td><td> $\mathbf { 0 . 9 5 4 \ : \pm 0 . 0 1 3 }$ </td></tr><tr><td>PV-DM</td><td> $2 4 6 . 4 9 \pm 6 3 . 8 1$ </td><td> $1 3 . 2 5 \pm 2 . 0 2$ </td><td> $0 . 6 1 3 \pm 0 . 0 8 8$ </td><td></td><td> $0 . 4 5 6 \pm 0 . 1 5 0$ </td></tr><tr><td>PV-Pinv</td><td> $2 7 0 . 9 8 \pm 4 4 . 0 4$ </td><td> $1 2 . 1 8 \pm 1 . 2 3$ </td><td> $0 . 4 5 5 \pm 0 . 0 9 6$ </td><td></td><td> $0 . 3 0 8 \pm 0 . 1 9 2$ </td></tr></table>

Arrows (↑/↓) indicate whether higher or lower values are better for each metric. Bold indicates the best performance per tissue and metric.  
trends in Table 1, showing consistent improvements of DeepPV-MRF over PV-DM and PV-Pinv across shot settings.

## 4.2 in vivo results

Figure 5 shows the in vivo tissue fraction maps for PV-DM, PV-Pinv, and DeepPV-MRF under single-shot, three-shot, and six-shot acquisitions from sub 01. CSF, WM, and GM maps are displayed in separate columns. Qualitatively, PV-DM produced relatively stable CSF maps across shot settings, but the WM and GM maps exhibited visible noise and aliasing, particularly in the single-shot and three-shot acquisitions. PV-Pinv generated smoother CSF maps, yet the WM and GM maps appeared spatially blurred and showed reduced structural detail across all acquisition settings. In contrast, DeepPV-MRF produced tissue fraction maps with more consistent intensity distributions and clearer anatomical structures for CSF, WM, and GM under all three shot settings.

Figure 6 shows the corresponding DeepPV-MRF results from the two additional volunteers, sub 02 and sub 03, to further evaluate the model performance under in vivo acquisition conditions. As shown in Figure 6, the proposed method produced visually consistent CSF, WM, and GM fraction maps across both subjects and diferent acquisition settings, including single-shot, three-shot, and six-shot scans. The main anatomical structures, including the ventricles, cortical GM, and deep WM regions, were well preserved in all three tissue fraction maps. Compared with the single-shot results, the three-shot and six-shot acquisitions showed improved spatial smoothness and clearer tissue boundaries, while maintaining a similar overall tissue distribution pattern.

![](images/5295f0d725f487c83a2221810a4ea19be529d51e4d505c3f918e4c0e5f588d92.jpg)

![](images/7af3f4e5d9792dbe1cb88c600123d5a81d4348dcc5717ddd53aadc6ed5bb2fa7.jpg)  
WM

![](images/bd9f96c204d3b7c60c91b702069150f6474ffe836bf6557239cbb4ec134d5d98.jpg)

![](images/5fd1d3f2b5889d50a0fada4f39c28b1dd5d83bcaed75c3cc5e5f9b9cfdd9b433.jpg)

![](images/f7f2287f35b0cfd3192f70c6f55d60137ee48ba66163fa8cb0bb9e8a726cb4f3.jpg)

![](images/a6467b354150095782998b9f3be8137469944b9446cea7b8987d022c0501551f.jpg)

![](images/40c44c790744e7fe63166120f29878a6c89aa5a234b6cac1547fcac8501b8cbb.jpg)

![](images/b4d74350dfbfeba8ec2307469acaf15d01abba3bb3d1483f915ff7c5c65b9ad9.jpg)

![](images/0813818c4a1c7706180c12000b8b6081f73f9e9602933dff0423653dde6a329f.jpg)

![](images/e531ac3c0eb65bb75db073a41949034d9f4d8eac0b00540ebf24cca9d40af798.jpg)  
Figure 4: Scatter plots comparing DeepPV-MRF predicted tissue fraction values against reference FSL-FAST labels for cerebrospinal fluid (CSF), white matter (WM), and gray matter (GM) across single-shot (top row), three-shot (middle row), and six-shot (bottom row) acquisition settings. Each plot displays voxel-wise predictions (y-axis) versus corresponding reference values (x-axis). Quantitative metrics include Pearson correlation coeficient, MAE, coeficient of determination (R<sup>2</sup>), and the slope of the fitted regression line. The red dashed line indicates the line of perfect agreement.

![](images/286b82253d35f5d8a0e862e17e3693ec981f7f1aa0bf15e40a8efff0d7b7951b.jpg)  
Figure 5: Comparison of predicted tissue fraction maps for cerebrospinal fluid (CSF), white matter (WM), and gray matter (GM) across single-shot, three-shot, and six-shot in vivo MRF acquisitions for sub 01. For each tissue type, results from three methods—partial volume dictionary matching (PV-DM), partial volume pseudoinverse (PV-Pinv), and the proposed DeepPV-MRF—are shown from left to right. Each row corresponds to a diferent acquisition setting, with intensity values mapped from 0 to 1.

![](images/97912a4a2db51510f53bdf9a4f5ba958ee04e774edfc70639177c7d359f27830.jpg)  
Figure 6: Representative in vivo partial-volume maps from two volunteers (sub 02 and sub 03) reconstructed using the proposed DeepPV-MRF model under 1-shot, 3-shot, and 6-shot acquisitions. For each volunteer, the same anatomical slice is shown across shot settings to qualitatively assess consistency across acquisition schemes. Separate maps are shown for CSF, WM, and GM tissue fractions. All maps are displayed on the same scale from 0 to 1.

Table 2: Cross-shot consistency of DeepPV-MRF in the three in vivo volunteers, using the six-shot prediction as an internal high-shot reference. Lower mean absolute error (MAE) indicates greater agreement between the lower-shot and six-shot predictions. The six-shot prediction is used only as an internal consistency reference and should not be interpreted as in vivo ground truth.
<table><tr><td>Tissue</td><td>Subject</td><td>1-shot vs. 6-shot</td><td>3-shot vs. 6-shot</td><td>Improvement (%)</td></tr><tr><td rowspan="3">CSF</td><td>sub01</td><td>0.164</td><td>0.138</td><td>15.9</td></tr><tr><td>sub02</td><td>0.131</td><td>0.126</td><td>3.5</td></tr><tr><td>sub03</td><td>0.130</td><td>0.109</td><td>16.5</td></tr><tr><td rowspan="3">WM</td><td>sub01</td><td>0.245</td><td>0.179</td><td>27.1</td></tr><tr><td>sub02</td><td>0.227</td><td>0.194</td><td>14.8</td></tr><tr><td>sub03</td><td>0.191</td><td>0.181</td><td>5.1</td></tr><tr><td rowspan="3">GM</td><td>sub01</td><td>0.317</td><td>0.226</td><td>28.7</td></tr><tr><td>sub02</td><td>0.286</td><td>0.247</td><td>13.6</td></tr><tr><td>sub03</td><td>0.256</td><td>0.225</td><td>12.1</td></tr></table>

Improvement denotes the relative MAE reduction from 1-shot vs. 6-shot to 3-shot vs. 6-shot.

To complement this qualitative assessment, Table 2 summarizes the cross-shot consistency of DeepPV-MRF across the three in vivo volunteers by using each subject’s six-shot prediction as an internal high-shot reference. Across all nine subject-tissue pairs, the three-shot predictions showed lower MAE than the corresponding single-shot predictions, with an average MAE reduction of 15.3%, supporting improved agreement with the six-shot reference as the acquisition increased from single-shot to three-shot. These results suggest that DeepPV-MRF can provide stable tissue fraction estimation across diferent subjects, anatomical instances, and the scanner/protocol settings included in this in vivo evaluation.

To further assess robustness under degraded acquisition conditions, we performed an additional in vivo noise-sensitivity analysis for the 1-shot setting across the three volunteers by perturbing the input MRF signals with complex Gaussian noise over a range of SNR levels. As shown in Figure 7, the MAE between noisy-input predictions and the corresponding clean-input predictions decreased monotonically as SNR increased for CSF, WM, and GM in all three volunteers, indicating that the proposed framework remained stable under progressively noisier inputs.

![](images/52548c3de7a9c0256dbc3e30f3577f914f73d2bb57850ff037c9295ed07ae49b.jpg)

![](images/ac8510c29285524b0dca7722d264228dce67665979f21ba809e483475397fad4.jpg)

![](images/c058523eafe676bb28e3ac83a9dbc8619a803d4b954bc380eef364b8e41d5924.jpg)

![](images/b12a5f24832d31ec22c9f9efe019d1897c5a4ab1115cbc0bc23e31a33f070eda.jpg)

![](images/ad3ce77c9ffe7cc424a9f59dc50488d271a334a2f2b9916694486421fe139026.jpg)  
Figure 7: In vivo noise-sensitivity analysis for the single-shot setting across the three volunteers. The input MRF signals were perturbed with complex Gaussian noise over signal-to-noise ratio (SNR) levels from 10 dB to 40 dB. Rows correspond to sub01, sub02, and sub03, and columns show cerebrospinal fluid (CSF), white matter (WM), and gray matter (GM). The plotted values represent the mean absolute error (MAE) between predictions from noisy inputs and the corresponding clean-input predictions. Solid lines indicate the mean across slices, and shaded regions denote the 95% confidence intervals.

## 4.3 Ablation Study

To quantify the contribution of the main architectural and loss design choices, we conducted ablation experiments on the one-shot external evaluation set, including removal of the spatial refinement module, addition of auxiliary B0/PD heads to the final model, removal of the spatial gradient loss, variation of the latent dimension, and a sweep of the spatial regularization weight λ<sub>spatial</sub>. The results are summarized in Table 3.

Among the core ablations, removing the spatial refinement module led to the largest performance drop, confirming that spatial context is essential for accurate PV estimation. The temporal encoder alone increased the macro tissue-wise MAE from 63.95 to 89.54 and reduced average PSNR/SSIM from 26.48/0.933 to 23.95/0.890, indicating that the main performance gain arises from the joint temporal-spatial design rather than from voxel-wise temporal encoding alone.

Table 3: Ablation study on the external 1-shot simulation test set. Mean absolute error (MAE) denotes tissue-wise $\mathrm { M A E } _ { \mathrm { t i s s u e } }$ and is reported in $\times 1 0 ^ { 3 }$ . Macro-averaged Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM) are unweighted averages across cerebrospinal fluid (CSF), white matter (WM), and gray matter (GM). The best value for each metric within each ablation family is shown in bold, with lower MAE and higher PSNR/SSIM indicating better performance. The full model denotes the architecture used for the main results; the B0/PD-head row adds auxiliary heads to this model.
<table><tr><td>Family</td><td>Setting</td><td>CSF MAE</td><td>WM MAE</td><td>GM MAE</td><td>Avg. MAE</td><td>Avg. PSNR</td><td>Avg. SSIM</td></tr><tr><td colspan="2">Core ablations</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Full model</td><td>41.76</td><td>74.39</td><td>75.69</td><td>63.95</td><td>26.48</td><td>0.933</td></tr><tr><td></td><td>With B0/PD heads</td><td>43.42</td><td>77.70</td><td>78.51</td><td>66.55</td><td>26.14</td><td>0.922</td></tr><tr><td></td><td>Temporal encoder only</td><td>52.12</td><td>110.25</td><td>106.26</td><td>89.54</td><td>23.95</td><td>0.890</td></tr><tr><td></td><td>No spatial gradient loss</td><td>44.45</td><td>77.69</td><td>79.55</td><td>67.23</td><td>26.05</td><td>0.925</td></tr><tr><td colspan="2">Latent dimension sweep</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Latent dim = 8</td><td>41.10</td><td>73.62</td><td>75.11</td><td>63.28</td><td>26.61</td><td>0.933</td></tr><tr><td></td><td>Latent dim = 16</td><td>41.76</td><td>74.39</td><td>75.69</td><td>63.95</td><td>26.48</td><td>0.933</td></tr><tr><td></td><td>Latent dim = 32</td><td>38.32</td><td>84.34</td><td>80.72</td><td>67.79</td><td>25.92</td><td>0.919</td></tr><tr><td></td><td>Latent dim = 64</td><td>42.13</td><td>88.46</td><td>85.54</td><td>72.05</td><td>25.53</td><td>0.911</td></tr><tr><td></td><td>Latent dim = 128</td><td>43.41</td><td>87.44</td><td>85.54</td><td>72.13</td><td>25.63</td><td>0.911</td></tr><tr><td colspan="2">Spatial regularization sweep</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td> $\lambda _ { \mathrm { s p a t i a l } } = 0 . 0 1$ </td><td>40.89</td><td>76.35</td><td>76.94</td><td>64.73</td><td>26.45</td><td>0.932</td></tr><tr><td></td><td> $\lambda _ { \mathrm { s p a t i a l } } = 0 . 0 5$ </td><td>41.76</td><td>74.39</td><td>75.69</td><td>63.95</td><td>26.48</td><td>0.933</td></tr><tr><td></td><td> $\lambda _ { \mathrm { s p a t i a l } } = 0 . 1 0$ </td><td>39.22</td><td>78.75</td><td>77.23</td><td>65.07</td><td>26.20</td><td>0.925</td></tr><tr><td></td><td> $\lambda _ { \mathrm { s p a t i a l } } = 0 . 5 0$ </td><td>41.29</td><td>76.25</td><td>77.13</td><td>64.89</td><td>26.38</td><td>0.929</td></tr></table>

The final full model does not include auxiliary B0/PD heads. Adding these heads slightly degraded PV estimation performance in the current setting, increasing the macro tissue-wise MAE from 63.95 to 66.55 and reducing average PSNR/SSIM from 26.48/0.933 to 26.14/0.922. This suggests that, although these auxiliary branches were originally introduced to regularize the shared encoder, their supervision was not optimally aligned with the primary PV estimation objective and may have introduced mild optimization competition<sup>37</sup>. Removing the spatial gradient loss also caused a modest degradation relative to the full model, increasing the macro tissue-wise MAE to 67.23. In the latent-dimension sweep, latent dimensions of 8 and 16 produced comparable performance, with the latent dimension of 8 yielding the lowest macro tissue-wise MAE on this one-shot external set, whereas larger latent dimensions of 32, 64, and 128 degraded the WM and GM metrics. For the spatial regularization sweep, performance remained relatively stable across a range of nonzero $\lambda _ { \mathrm { s p a t i a l } }$ values, and the default setting of $\lambda _ { \mathrm { s p a t i a l } } = 0 . 0 5$ provided the best macro-averaged performance among the tested nonzero weights.

## 5 Discussion

This study presents DeepPV-MRF, a deep learning framework for direct partial volume estimation from undersampled MRF signals. Unlike conventional methods such as PV-DM, PV-Pinv, and FSL FAST, the proposed approach avoids strong reliance on hand-crafted priors and multi-step intermediate processing. By directly mapping aliased MRF time-series signals to tissue fraction maps, DeepPV-MRF simplifies the workflow and reduces the risk of error propagation from intermediate reconstructions or model assumptions.

The performance gains of DeepPV-MRF appear to arise from two main factors. First, the simulation-based training pipeline combines subject-derived tissue property maps and volumetric fraction maps with realistic radial undersampling, allowing the network to learn tissue-specific signal behavior under practical aliasing conditions. Second, the network architecture combines voxel-wise temporal encoding with spatial refinement, which is particularly important in regions with complex tissue mixing. The ablation study supports this interpretation: removing the spatial refinement module caused the largest performance drop, whereas adding auxiliary B0/PD heads and removing the spatial gradient loss had smaller efects on the final PV estimation accuracy.

![](images/9f41a3fd530b6db6e988ec9cd2f75864d55fb1f77dc4e9f09379adf65af17e31.jpg)  
Figure 8: in vivo comparison of T1 and T2 maps from MRF-Mixer (left) and composite maps derived from DeepPV-MRF (middle), with their absolute diference shown on the right. The composite maps combine estimated tissue fractions with pure-tissue relaxation values. The diference maps confirm tha the largest discrepancies are concentrated at tissue boundaries, consistent with partial volume efects.

The simulation results demonstrated clear advantages of DeepPV-MRF over PV-DM and PV-Pinv across all shot settings, with particularly strong performance under the more challenging single-shot condition. The in vivo results further supported these findings, as DeepPV-MRF produced tissue fraction maps with smoother transitions and better anatomical continuity, especially in GM and WM regions.

The multi-volunteer in vivo cross-shot consistency analysis provided additional support for this interpretation. Because voxel-wise in vivo tissue-fraction ground truth was unavailable, the six-shot prediction was used only as an internal high-shot reference rather than as an accuracy standard. Across all nine subject-tissue pairs, the three-shot estimates showed lower MAE relative to the six-shot reference than the single-shot estimates, with an average MAE reduction of 15.3%. This trend indicates that increasing the number of shots improved agreement with the high-shot reference across subjects and tissues, while preserving the visually observed anatomical structures in the estimated fraction maps.

The comparison between composite multi-component T1/T2 maps and single-component MRF-Mixer estimates further highlights the value of explicit partial volume modeling. As shown in Figure 8, the largest discrepancies were observed near tissue interfaces, where partial volume efects are expected to be strongest. This suggests that mixed-tissue MRF voxels are not always well represented by single-component relaxation estimates, and that explicit PV estimation may support future frameworks for joint estimation of tissue fractions and intrinsic tissue parameters.

From a practical perspective, the inference benchmark reported in the Network Training section indicates that DeepPV-MRF can generate tissue-fraction maps with low per-slice latency and modest GPU memory usage. In addition, the three-volunteer in vivo noise perturbation analysis showed monotonically decreasing prediction deviation as SNR increased, providing preliminary evidence of robustness to moderate noise variation.

Several limitations should be acknowledged. Although the in vivo evaluation included three healthy volunteers acquired from two diferent scanner platforms (one from 3T Prisma and two from 3T Cima.x Fit), the validation does not yet establish full generalizability across sites, vendors, acquisition protocols, or pathological populations. In addition, the supervisory tissue fraction maps were derived from FSL FAST on high-resolution anatomical images, so they should be regarded as surrogate references rather than absolute ground truth, and some FAST-related bias may therefore be inherited by the model. The synthetic signal model also assumes linear combinations of pure-tissue fingerprints <sup>3</sup>, which may not fully capture efects such as magnetization transfer, exchange, or other multi-compartment nonlinearities in real tissues. Future work should therefore include broader validation across scanners, protocols, B1 conditions, motion levels, and pathological tissues, together with more independent reference standards such as phantoms or expert-curated annotations.

## 6 Conclusion

In summary, we proposed a deep learning-based method for partial volume estimation that directly maps MRF signals to tissue fraction maps in an end-to-end approach. By integrating subject-specific simulation data with radial arm undersampling and employing a spatial-temporal cascaded network architecture, the method achieves improved accuracy and robustness compared to traditional approaches such as PV-DM and PV-Pinv. These advantages simplify the processing pipeline and improve its suitability for practical applications. Future work will focus on validating the method across more diverse anatomical regions and clinical conditions, exploring self-supervised or unsupervised training strategies, and extending the approach to additional acquisition sequences.

## References

<sup>1</sup> J. Tohka, Partial volume efect modeling for segmentation and tissue classification of brain magnetic resonance images: A review, World journal of radiology 6, 855 (2014).

<sup>2</sup> E. Plante and L. Turkstra, Sources of error in the quantitative analysis of MRI scans, Magnetic resonance imaging 9, 589–595 (1991).

A. Deshmane, D. F. McGivney, D. Ma, Y. Jiang, C. Badve, V. Gulani, N. Seiberlich, and M. A. Griswold, Partial volume mapping using magnetic resonance fingerprinting, NMR in Biomedicine 32, e4082 (2019).

4 B. J. Bedell and P. A. Narayana, Volumetric analysis of white matter, gray matter, and CSF using fractional volume analysis, Magnetic resonance in medicine 39, 961–969 (1998).

<sup>5</sup> M. E. Shenton, C. C. Dickey, M. Frumin, and R. W. McCarley, A review of MRI findings in schizophrenia, Schizophrenia research 49, 1–52 (2001).

<sup>6</sup> M. W. Weiner et al., The Alzheimer’s Disease Neuroimaging Initiative: a review of papers published since its inception, Alzheimer’s & Dementia 9, e111–e194 (2013).

<sup>7</sup> R. Kikinis, C. R. Guttmann, D. Metcalf, W. M. Wells III, G. J. Ettinger, H. L. Weiner, and F. A. Jolesz, Quantitative follow-up of patients with multiple sclerosis using MRI: technical aspects, Journal of Magnetic Resonance Imaging: An Oficial Journal of the International Society for Magnetic Resonance in Medicine 9, 519–530 (1999).

<sup>8</sup> K. C. Gao, G. Nair, I. C. Cortese, A. Koretsky, and D. S. Reich, Sub-millimeter imaging of brain-free water for rapid volume assessment in atrophic brains, Neuroimage 100, 370–378 (2014).

<sup>9</sup> W. Wells, W. Grimson, R. Kikinis, and F. Jolesz, Adaptive segmentation of MRI data, IEEE Transactions on Medical Imaging 15, 429–442 (1996).

<sup>10</sup> J. Tohka, A. Zijdenbos, and A. Evans, Fast and robust parameter estimation for statistical partial volume models in brain MRI, Neuroimage 23, 84–97 (2004).

<sup>11</sup> M. B. Cuadra, L. Cammoun, T. Butz, O. Cuisenaire, and J.-P. Thiran, Comparison and validation of tissue modelization and statistical classification methods in T1-weighted MR brain images, IEEE transactions on medical imaging 24, 1548–1565 (2005).

<sup>12</sup> M. A. Chappell, A. R. Groves, B. J. MacIntosh, M. J. Donahue, P. Jezzard, and M. W. Woolrich, Partial volume correction of multiple inversion time arterial spin labeling MRI data, Magnetic resonance in medicine 65, 1173–1183 (2011).

<sup>13</sup> D. Ma, V. Gulani, N. Seiberlich, K. Liu, J. L. Sunshine, J. L. Duerk, and M. A. Griswold, Magnetic resonance fingerprinting, Nature 495, 187–192 (2013).

<sup>14</sup> D. F. McGivney, E. Pierre, D. Ma, Y. Jiang, H. Saybasili, V. Gulani, and M. A. Griswold, SVD compression for magnetic resonance fingerprinting in the time domain, IEEE transactions on medical imaging 33, 2311–2322 (2014).

15 M. Yang, D. Ma, Y. Jiang, J. Hamilton, N. Seiberlich, M. A. Griswold, and D. McGivney, Low rank approximation methods for MR fingerprinting with large scale dictionaries, Magnetic resonance in medicine 79, 2392–2400 (2018).

16 M. E. Poorman, M. N. Martin, D. Ma, D. F. McGivney, V. Gulani, M. A. Griswold, and K. E. Keenan, Magnetic resonance fingerprinting Part 1: Potential uses, current challenges, and recommendations, Journal of Magnetic Resonance Imaging 51, 675–692 (2020).

<sup>17</sup> O. Cohen, B. Zhu, and M. S. Rosen, MR fingerprinting deep reconstruction network (DRONE), Magnetic resonance in medicine 80, 885–894 (2018).

<sup>18</sup> Z. Fang, Y. Chen, M. Liu, L. Xiang, Q. Zhang, Q. Wang, W. Lin, and D. Shen, Deep learning for fast and spatially constrained tissue quantification from highly accelerated data in magnetic resonance fingerprinting, IEEE transactions on medical imaging 38, 2364–2374 (2019).

19 M. Golbabaee, D. Chen, P. A. Gómez, M. I. Menzel, and M. E. Davies, Geometry of deep learning for magnetic resonance fingerprinting, in ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 7825–7829, IEEE, 2019.

<sup>20</sup> P. Cao, D. Cui, V. Vardhanabhuti, and E. S. Hui, Development of fast deep learning quantification for magnetic resonance fingerprinting in vivo, Magnetic Resonance Imaging 70, 81–90 (2020).

<sup>21</sup> P. Li and Y. Hu, Deep magnetic resonance fingerprinting based on local and global vision transformer, Medical Image Analysis 95, 103198 (2024).

<sup>22</sup> P. Li and Y. Hu, Enhancing MRF Reconstruction: A Model-Based Deep Learning Approach Leveraging Learned Sparsity and Physics Priors, IEEE Transactions on Computational Imaging (2024).

<sup>23</sup> P. Li and Y. Hu, Deep graph embedding based on Laplacian eigenmaps for MR fingerprinting reconstruction, Medical Image Analysis , 103481 (2025).

<sup>24</sup> T. Ding, Y. Gao, Z. Xiong, F. Liu, M. A. Cloos, and H. Sun, MRF-Mixer: A Simulation-Based Deep Learning Framework for Accelerated and Accurate Magnetic Resonance Fingerprinting Reconstruction, Information 16, 218 (2025).

<sup>25</sup> D. Wang, J. Ostenson, and D. S. Smith, snapMRF: GPU-accelerated magnetic resonance fingerprinting dictionary generation and matching using extended phase graphs, Magnetic resonance imaging 66, 248–256 (2020).

<sup>26</sup> S. F. Cauley, K. Setsompop, D. Ma, Y. Jiang, H. Ye, E. Adalsteinsson, M. A. Griswold, and L. L. Wald, Fast group matching for MR fingerprinting reconstruction, Magnetic resonance in medicine 74, 523–528 (2015).

<sup>27</sup> H. S. Choi, D. R. Haynor, and Y. Kim, Partial volume tissue classification of multichannel magnetic resonance images-a mixel model, IEEE Transactions on Medical Imaging 10, 395–407 (1991).

<sup>28</sup> Y. Gao, T. Ding, M. Cloos, and H. Sun, MRF-Mixer: A Self-Supervised Deep Learning MRF Framework, Proc. Intl. Soc. Mag. Reson. Med. 31, 954 (2023), 2023, Available: https://archive.ismrm.org/2023/0954.html.

29 S. Tang, C. Fernandez-Granda, S. Lannuzel, B. Bernstein, R. Lattanzi, M. Cloos, F. Knoll, and J. Assländer, Multicompartment magnetic resonance fingerprinting, Inverse problems 34, 094005 (2018).

<sup>30</sup> H. Sun, J. O. Cleary, R. Glarin, S. C. Kolbe, R. J. Ordidge, B. A. Mofat, and G. B. Pike, Extracting more for less: multi-echo MP2RAGE for simultaneous T1-weighted imaging, T1 mapping, mapping, SWI, and QSM from a single acquisition, Magnetic resonance in medicine 83, 1178–1191 (2020).

<sup>31</sup> Y. Zhang, M. Brady, and S. Smith, Segmentation of brain MR images through a hidden Markov random field model and the expectation-maximization algorithm, IEEE transactions on medical imaging 20, 45–57 (2001).

32 A. M. Peters, M. J. Brookes, F. G. Hoogenraad, P. A. Gowland, S. T. Francis, P. G. Morris, and R. Bowtell, T2\* measurements in human brain at 1.5, 3 and 7 T, Magnetic resonance imaging 25, 748–753 (2007).

<sup>33</sup> J. P. Wansapura, S. K. Holland, R. S. Dunn, and W. S. Ball Jr, NMR relaxation times in the human brain at 3.0 tesla, Journal of Magnetic Resonance Imaging: An Oficial Journal of the International Society for Magnetic Resonance in Medicine 9, 531–538 (1999).

<sup>34</sup> G. J. Stanisz, E. E. Odrobina, J. Pun, M. Escaravage, S. J. Graham, M. J. Bronskill, and R. M. Henkelman, T1, T2 relaxation and magnetization transfer in tissue at 3T, Magnetic Resonance in Medicine: An Oficial Journal of the International Society for Magnetic Resonance in Medicine 54, 507–512 (2005).

<sup>35</sup> M. A. Cloos, J. Assländer, B. Abbas, J. Fishbaugh, J. S. Babb, G. Gerig, and R. Lattanzi, Rapid radial T1 and T2 mapping of the hip articular cartilage with magnetic resonance fingerprinting, Journal of Magnetic Resonance Imaging 50, 810–815 (2019).

<sup>36</sup> C. A. Schneider, W. S. Rasband, and K. W. Eliceiri, NIH Image to ImageJ: 25 years of image analysis, Nature methods 9, 671–675 (2012).

<sup>37</sup> S. Ruder, An overview of multi-task learning in deep neural networks, arXiv preprint arXiv:1706.05098 (2017).

## Supplementary Material

Supplementary Table 1: Paired statistical analysis using the same simulation slices as in main manuscript Table 1. For each shot setting, tissue, and metric, we report the Holm-adjusted Friedman p-value across the three methods and the median paired diference (∆) of DeepPV-MRF relative to each baseline. For MAE, negative values indicate lower error; for PSNR, SSIM, and $r _ { \mathrm { w e i g h t e d } } ,$ positive values indicate improved performance. Across all post-hoc paired Wilcoxon signed-rank tests, the Holm-adjusted p-value was $6 . 8 3 \times 1 0 ^ { - 2 0 }$
<table><tr><td>Shot setting</td><td>Tissue</td><td>Metric</td><td>Friedman pHolm</td><td>∆ vs PV-DM</td><td>∆ vs PV-Pinv</td></tr><tr><td rowspan="10">Single-shot</td><td rowspan="4">CSF</td><td> $\mathrm { M A E } \left( \times 1 0 ^ { 3 } \right)$ </td><td> $2 . 7 8 \times 1 0 ^ { - 5 2 }$ </td><td>-0.094</td><td>-0.153</td></tr><tr><td>PSNR</td><td> $2 . 7 8 \times 1 0 ^ { - 5 2 }$ </td><td>11.434</td><td>16.549</td></tr><tr><td>SSIM</td><td> $1 . 3 7 \times 1 0 ^ { - 5 2 }$ </td><td>0.332</td><td>0.554</td></tr><tr><td>rweighted</td><td> $1 . 3 7 \times 1 0 ^ { - 5 2 }$ </td><td>0.259</td><td>0.477</td></tr><tr><td rowspan="4">WM</td><td> $\mathrm { M A E } \left( \times 1 0 ^ { 3 } \right)$ </td><td> $6 . 0 1 \times 1 0 ^ { - 5 1 }$ </td><td>-0.165</td><td>-0.209</td></tr><tr><td>PSNR</td><td> $1 . 9 3 \times 1 0 ^ { - 4 8 }$ </td><td>12.611</td><td>13.350</td></tr><tr><td>SSIM</td><td> $1 . 3 7 \times 1 0 ^ { - 5 2 }$ </td><td>0.379</td><td>0.467</td></tr><tr><td>rweighted</td><td> $3 . 8 5 \times 1 0 ^ { - 4 8 }$ </td><td>0.424</td><td>0.509</td></tr><tr><td rowspan="4">GM</td><td> $\mathrm { M A E } \left( \times 1 0 ^ { 3 } \right)$ </td><td> $2 . 9 1 \times 1 0 ^ { - 4 2 }$ </td><td>-0.233</td><td>-0.181</td></tr><tr><td>PSNR</td><td> $4 . 0 3 \times 1 0 ^ { - 4 2 }$ </td><td>12.749</td><td>12.262</td></tr><tr><td>SSIM</td><td> $4 . 3 3 \times 1 0 ^ { - 4 4 }$ </td><td>0.414</td><td>0.430</td></tr><tr><td>rweighted</td><td> $4 . 0 3 \times 1 0 ^ { - 4 2 }$ </td><td>0.582</td><td>0.528</td></tr><tr><td rowspan="9">Three-shot</td><td rowspan="4">CSF</td><td> $\mathrm { M A E } \left( \times 1 0 ^ { 3 } \right)$ </td><td> $1 . 4 3 \times 1 0 ^ { - 5 1 }$ </td><td>-0.090</td><td>-0.150</td></tr><tr><td>PSNR</td><td> $6 . 0 1 \times 1 0 ^ { - 5 2 }$ </td><td>13.099</td><td>18.162</td></tr><tr><td>SSIM</td><td> $1 . 3 7 \times 1 0 ^ { - 5 2 }$ </td><td>0.302</td><td>0.509</td></tr><tr><td>rweighted</td><td> $1 . 4 3 \times 1 0 ^ { - 5 1 }$ </td><td>0.218</td><td>0.402</td></tr><tr><td rowspan="4">WM</td><td> $\mathrm { M A E } \left( \times 1 0 ^ { 3 } \right)$ </td><td> $2 . 7 8 \times 1 0 ^ { - 5 2 }$ </td><td>-0.157</td><td>-0.231</td></tr><tr><td>PSNR</td><td> $2 . 7 5 \times 1 0 ^ { - 5 1 }$ </td><td>14.640</td><td>16.480</td></tr><tr><td>SSIM</td><td> $1 . 3 7 \times 1 0 ^ { - 5 2 }$ </td><td>0.363</td><td>0.486</td></tr><tr><td>rweighted</td><td> $2 . 7 8 \times 1 0 ^ { - 5 2 }$ </td><td>0.391</td><td>0.544</td></tr><tr><td rowspan="4">GM</td><td> $\mathrm { M A E } \left( \times 1 0 ^ { 3 } \right)$ </td><td> $2 . 6 2 \times 1 0 ^ { - 4 2 }$ </td><td>-0.221</td><td>-0.235</td></tr><tr><td>PSNR</td><td> $3 . 6 5 \times 1 0 ^ { - 4 3 }$ </td><td>14.765</td><td>15.502</td></tr><tr><td>SSIM</td><td> $1 . 3 7 \times 1 0 ^ { - 5 2 }$ </td><td>0.371</td><td>0.514</td></tr><tr><td>rweighted</td><td> $1 . 9 2 \times 1 0 ^ { - 4 9 }$ </td><td>0.557</td><td>0.691</td></tr><tr><td rowspan="8">Six-shot</td><td rowspan="4">CSF</td><td> $\mathrm { M A E } \left( \times 1 0 ^ { 3 } \right)$ </td><td> $1 . 4 3 \times 1 0 ^ { - 5 1 }$ </td><td>-0.082</td><td>-0.149</td></tr><tr><td>PSNR</td><td> $1 . 4 3 \times 1 0 ^ { - 5 1 }$ </td><td>13.155</td><td>18.550</td></tr><tr><td>SSIM</td><td> $1 . 3 7 \times 1 0 ^ { - 5 2 }$ </td><td>0.282</td><td>0.508</td></tr><tr><td>rweighted</td><td> $1 . 4 3 \times 1 0 ^ { - 5 1 }$ </td><td>0.197</td><td>0.389</td></tr><tr><td rowspan="3">WM</td><td> $\mathrm { M A E } \left( \times 1 0 ^ { 3 } \right)$ </td><td> $6 . 0 1 \times 1 0 ^ { - 5 2 }$ </td><td>-0.152</td><td>-0.238</td></tr><tr><td>PSNR</td><td> $2 . 7 5 \times 1 0 ^ { - 5 1 }$ </td><td>15.773</td><td>17.881</td></tr><tr><td>SSIM</td><td> $1 . 3 7 \times 1 0 ^ { - 5 2 }$ </td><td>0.347</td><td>0.479</td></tr><tr><td rowspan="3"></td><td>Γweighted</td><td> $2 . 7 8 \times 1 0 ^ { - 5 2 }$ </td><td>0.363</td><td>0.545</td></tr><tr><td> $\mathrm { M A E } \left( \times 1 0 ^ { 3 } \right)$ </td><td> $2 . 6 9 \times 1 0 ^ { - 4 3 }$ </td><td>-0.206</td><td>-0.241</td></tr><tr><td>PSNR</td><td> $1 . 1 3 \times 1 0 ^ { - 4 3 }$ </td><td>15.351</td><td>16.838</td></tr><tr><td rowspan="3">GM</td><td>SSIM</td><td> $1 . 3 7 \times 1 0 ^ { - 5 2 }$ </td><td>0.357</td><td>0.524</td></tr><tr><td>rweighted</td><td> $4 . 1 0 \times 1 0 ^ { - 4 9 }$ </td><td>0.536</td><td>0.701</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

<table><tr><td>Shot setting</td><td>Tissue</td><td>MAE  $( \times 1 0 ^ { 3 } )$  (↓)</td><td>RMSE  $( \times 1 0 ^ { 3 } )$  (4)</td></tr><tr><td rowspan="3">Single-shot</td><td>CSF</td><td> $1 9 . 2 5 \pm 3 . 6 2$ </td><td> $4 3 . 3 5 \pm 7 . 5 0$ </td></tr><tr><td>WM</td><td> $3 8 . 9 8 \pm 7 . 5 3$ </td><td> $8 3 . 0 0 \pm 1 3 . 4 1$ </td></tr><tr><td>GM</td><td> $5 5 . 3 4 \pm 1 0 . 1 2$ </td><td> $9 5 . 2 4 \pm 1 5 . 3 4$ </td></tr><tr><td rowspan="3">Three-shot</td><td>CSF</td><td> $1 2 . 5 6 \pm 3 . 2 6$ </td><td> $2 8 . 9 0 \pm 7 . 5 7$ </td></tr><tr><td>WM</td><td> $2 9 . 2 3 \pm 5 . 2 4$ </td><td> $5 9 . 4 9 \pm 9 . 1 5$ </td></tr><tr><td>GM</td><td> $3 9 . 8 8 \pm 8 . 0 6$ </td><td> $6 9 . 2 6 \pm 1 2 . 5 1$ </td></tr><tr><td rowspan="3">Six-shot</td><td>CSF</td><td> $1 1 . 5 3 \pm 2 . 5 9$ </td><td> $2 6 . 6 6 \pm 6 . 3 0$ </td></tr><tr><td>WM</td><td> $2 4 . 4 9 \pm 4 . 4 7$ </td><td> $4 9 . 1 9 \pm 8 . 2 5$ </td></tr><tr><td>GM</td><td> $3 3 . 7 1 \pm 6 . 1 1$ </td><td> $5 7 . 5 7 \pm 9 . 5 5$ </td></tr></table>

Supplementary Table 2: Whole-brain error metrics for the DeepPV-MRF model under diferent shot settings. Metrics were computed over all voxels within the brain mask across the same 123 external synthetic test slices used for the main simulation evaluation. Values are reported as mean ± standard deviation.