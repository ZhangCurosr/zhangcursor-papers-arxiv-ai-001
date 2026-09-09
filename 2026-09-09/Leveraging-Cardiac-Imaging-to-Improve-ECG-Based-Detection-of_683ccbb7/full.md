# Leveraging Cardiac Imaging to Improve ECG-Based Detection of Chagas Disease in Resource-Constrained Settings

Laura Alvarez-Florez<sup>1,2,3\*</sup> Daniel Uyterlinde<sup>3\*</sup>, Samuel Ruipérez-Campillo<sup>5</sup>, Lukas P. A. Arts<sup>3,4</sup>, Folkert W. Asselbergs<sup>3,4</sup>, and Fleur V. Y. Tjong<sup>3,4</sup>

<sup>1</sup> Department of Biomedical Engineering and Physics, Amsterdam University Medical Center, The Netherlands

<sup>2</sup> Quantitative Healthcare Analysis Group, University of Amsterdam, The Netherlands

3 Department of Clinical and Experimental Cardiology, Amsterdam University Medical Center, The Netherlands

<sup>4</sup> Amsterdam Cardiovascular Sciences, Amsterdam University Medical Center, The Netherlands

5 Department of Computer Science, ETH Zurich, Switzerland ∗ These authors contributed equally to this work. Corresponding author: l.alvarezflorez@amsterdamumc.nl

Abstract. Chagas disease is a major cause of cardiomyopathy in Latin America. Cardiac magnetic resonance (CMR) imaging can characterize its structural abnormalities, but scanners and expert readers remain scarce in endemic regions. Electrocardiography (ECG) is inexpensive and widely available, yet structural disease must be inferred indirectly from electrical signals. We propose to transfer CMR-derived structural knowledge to ECG through contrastive pre-training. Using 63,193 paired ECG–CMR examinations from the UK Biobank, we align an ECG encoder with a clinically grounded CMR embedding space using an asymmetric InfoNCE objective. Despite seeing no Chagas cases during pre-training, the resulting representation improves ECG-based Chagas detection. Across CODE-15% and SaMi-Trop, a frozen linear probe achieves an AUROC of 0.851 and sensitivity at the top 5% of predicted risk (Top5%-TPR) of 0.427 in five-fold cross-validation, compared with 0.827 and 0.377 for an unaligned ECG-FM baseline. On the PhysioNet/CinC 2025 Challenge test set, our model obtains the highest AU-ROC on SaMi-Trop-3 and the best ELSA-Brasil challenge score among the three top-performing methods, indicating that imaging-supervised ECG representations can generalize to populations and settings beyond the pre-training distribution.

Keywords: Chagas disease · Electrocardiogram · Cardiac Magnetic Resonance Imaging · Contrastive Multimodal Learning · Resource-Constrained Diagnostics

![](images/cbf9ded26dbf5d9fc4d67d4b0620fcea4d4e5cbec96fdcbb8df374a37f7db669.jpg)  
Fig. 1. Overview of the proposed method for ECG prediction of Chagas disease.

## 1 Introduction

Cardiac magnetic resonance (CMR) imaging is widely regarded as the reference standard for assessing cardiac structure and function and for phenotyping cardiovascular disease [14]. Yet the cost, infrastructure, and specialist expertise it demands place it out of reach for routine use in much of the world: Latin America, South-East Asia, and Africa have roughly 4-, 15-, and up to 30-fold fewer MRI scanners per capita respectively compared to European countries [2, 9]. The 12-lead electrocardiogram (ECG), by contrast, is inexpensive, portable, and available across nearly all healthcare settings. As a result, much of the world’s cardiovascular disease burden, particularly in low- and middle-income countries (LMICs), is assessed with a modality that records the heart’s electrical activity but reflects its underlying structure and function only indirectly.

This disparity in cardiac assessment is consequential for Chagas cardiomyopathy. Caused by Trypanosoma cruzi infection, Chagas disease is a neglected tropical disease and a leading cause of non-ischemic cardiomyopathy in Latin America [6]. Its manifestation is fundamentally structural, characterized by myocardial fibrosis, ventricular remodeling, and progressive ventricular dysfunction [6]. These abnormalities can be directly visualized with CMR, yet, access to the imaging modalities capable of directly characterizing these structural abnormalities remains limited in many afected and endemic regions.

One promising approach to narrowing this gap is multimodal representation learning, where paired ECG and CMR examinations are used to teach ECG models about cardiac structure and function without requiring imaging at deployment. Several studies have demonstrated that contrastive alignment between ECG and CMR representations yields ECG encoders that outperform ECG-only self-supervised approaches on downstream cardiovascular prediction tasks [4, 19, 1, 5, 17]. By learning from paired ECG and imaging, these approaches encourage ECG representations to capture aspects of cardiac structure and function that are otherwise only directly observable through imaging. However, their evaluation has largely been restricted to populations and disease distributions similar to those used during model development, predominantly in European and North American cohorts. It therefore remains unknown whether the structural representations transfer under domain shift, and in particular, whether they retain diagnostic value for diseases whose patients and imaging are entirely absent from the pre-training data.

In this work, we propose ECG-CMR contrastive pre-training as a mechanism for transferring structural cardiac knowledge from imaging populations to ECG-based diagnosis in imaging-scarce settings. We pre-train an ECG encoder by contrastive alignment against paired ECG-CMR examinations from the UK Biobank [18], and subsequently fine-tune the resulting representation on Chagas disease detection. Because CMR is required only during pre-training, deployment relies exclusively on ECG, preserving applicability in resource-constrained settings. Importantly, Chagas disease is absent from the paired ECG-CMR cohorts used in the training data, providing a stringent test of whether structural cardiac information learned from imaging can be transferred through ECG representations to a distinct disease never observed during learning.

We make three contributions. First, we propose an imaging-supervised ECG pre-training framework that distills structural cardiac information from paired ECG-CMR data into deployable ECG representations. Second, we demonstrate that this structural information transfers to Chagas disease detection despite the complete absence of Chagas patients and Chagas-related imaging during pre-training. Third, we validate the approach on the PhysioNet/Computing in Cardiology 2025 Chagas Disease Detection Challenge benchmark [15], achieving competitive performance while outperforming strong ECG-only baselines. Together, our findings suggest that structural cardiac knowledge acquired from imaging cohorts can be distilled into ECG representations and transferred to other diseases, populations, and healthcare settings beyond those observed during pre-training, providing a pathway toward more equitable and accessible cardiovascular diagnoses.

## 2 Related Work

## 2.1 Deep Learning for ECG and Chagas Cardiomyopathy

Self-supervised learning and large-scale foundation models have substantially advanced ECG representation learning by learning generalizable features that can be transferred across a wide range of downstream cardiovascular tasks [13,

11]. Driven in part by initiatives such as the PhysioNet Challenge, deep learning methods have been extended to the prediction of Chagas cardiomyopathy from ECG. These eforts, focused on convolutional neural networks and Transformers [10, 7], are trained on disease-specific cohorts or fine-tuned from models pre-trained on large ECG datasets (e.g., PTB-XL [21] or CODE-15% [16]), thus remaining strictly unimodal. The models must therefore infer the complex structural patters associated with Chagas disease purely from electrical signals, without structural supervision during training.

## 2.2 Multimodal ECG-CMR Representation Learning

To bridge the gap between electrical signals and cardiac mechanics, recent works have proposed multimodal contrastive learning to align ECG signals with CMR images. Methods such as MMCL [19], PTACL [17], and ECCL [4] utilize largescale paired datasets (predominantly the UK Biobank [18]) to project ECGs into a shared latent space with CMR embeddings. At inference, the ECG encoder can be deployed unimodally, retaining structural representations acquired during pre-training. While these models achieve state-of-the-art performance from ECG on structural phenotype regression, their evaluations are strictly confined to the populations and pathology distributions of their pre-training cohorts and do not assess the transferability of these representations beyond the training data.

## 3 Methodology

Our framework operates in two phases (Fig 1). In the pre-training phase, we use paired ECG and CMR examinations from the UK Biobank [18] to align an ECG encoder with a clinically grounded structural target space via contrastive learning. In the deployment phase, the structurally-aware ECG encoder is deployed unimodally and adapted for Chagas disease classification in an unseen, imaging-scarce cohort.

## 3.1 Unimodal ECG and CMR Encoders

Rather than learning cardiac features from scratch, we initialize our framework with high-capacity unimodal foundation models to isolate the efect of crossmodal transfer. For the ECG modality, we use ECG-FM [12], a hybrid CNN-Transformer architecture pre-trained via masked autoencoding and contrastive learning on 1.4 million 12-lead ECGs. For CMR, we use CineMA [8], a multiview convolution-transformer masked autoencoder pre-trained on cine CMR sequences that maps a single timeframe of a CMR volume of short-axis and (2-, 3-, and 4-chamber) long-axis views into a consistent spatial representation. During cross-modal alignment, the CMR encoder remains strictly frozen, both for computational eficiency and to provide a stable target.

## 3.2 Clinically Grounded Target Space

For ECG-CMR alignment, the target representation must encode not only intersubject anatomical variation but also phase-dependent cardiac function. The raw CineMA embedding space, however, is near-degenerate: end-diastolic and end-systolic representations are nearly indistinguishable (average end-diastole to end-systole cosine similarity $> 0 . 9 9 \AA$ ), so static anatomy dominates the variance and obscures subtle functional dynamics.

To construct a well-conditioned target that balances anatomy and function, we extract the end-diastolic $\mathbf { \Pi } ( \mathbf { z } _ { \mathrm { E D } } )$ and end-systolic $\mathbf { \Pi } ( \mathbf { z } _ { \mathrm { E S } } )$ representations from the frozen CineMA encoder. Rather than aligning to these raw vectors, we forward their concatenation through a supervised multi-layer perceptron (MLP):

$$
\begin{array} { r } { \mathbf { z } _ { \mathrm { p h e n o } } ^ { \left( i \right) } = h _ { \psi } \left( \left[ \mathbf { z } _ { \mathrm { E D } } ^ { \left( i \right) } ; \mathbf { z } _ { \mathrm { E S } } ^ { \left( i \right) } \right] \right) \in \mathbb { R } ^ { 2 5 6 } . } \end{array}\tag{1}
$$

The MLP $h _ { \psi }$ is supervised to regress a set of ground-truth structural and functional phenotypes: left ventricle ejection fraction (LVEF), left ventricle enddiastolic volume (LVEDV), LV mass, right ventricle end-diastolic volume (RVEDV) left ventricle global longitudinal strain (LVGLS), Atrial Fibrillation (AFib), myocardial infarction (MI). Once trained, $h _ { \psi }$ is frozen. This creates a 256- dimensional, clinically grounded bottleneck $\left( \mathbf { z } _ { \mathrm { p h e n o } } \right)$ that concentrates representational capacity on the dimensions of CMR space most relevant to cardiovascular pathology, stripping away redundant anatomical noise.

## 3.3 Asymmetric Contrastive Pre-training

To transfer this structural knowledge to the ECG, we employ an asymmetric contrastive alignment objective. Given a batch of $B$ patient pairs, let $\begin{array} { r } { \mathbf { z } _ { \mathrm { e c g } } ^ { ( i ) } = } \end{array}$ $g _ { \phi } ( f _ { \theta } ( \mathbf { x } _ { \mathrm { e c g } } ^ { ( i ) } ) )$ denote the $L _ { 2 } .$ -normalized ECG projection extracted from the optimal intermediate layer of the ECG-FM encoder (layer 9, found by linear probing every intermediate layer against several cardiac phenotypes), and let $\mathbf { z } _ { \mathrm { p h e n o } } ^ { ( i ) }$ denote the corresponding $L _ { 2 } .$ -normalized CMR target.

We optimize an asymmetric InfoNCE loss, where gradients flow exclusively through the ECG encoder $f _ { \theta }$ and its projection head $g _ { \phi }$

$$
\mathcal { L } _ { \mathrm { C L } } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \exp ( \mathbf { z } _ { \mathrm { e c g } } ^ { ( i ) } \cdot \mathbf { z } _ { \mathrm { p h e n o } } ^ { ( i ) } / \tau ) } { \sum _ { k = 1 } ^ { B } \exp ( \mathbf { z } _ { \mathrm { e c g } } ^ { ( i ) } \cdot \mathbf { z } _ { \mathrm { p h e n o } } ^ { ( k ) } / \tau ) } ,\tag{2}
$$

where $\tau = 0 . 1$ is the temperature parameter. By holding the target space strictly fixed, the asymmetric InfoNCE objective forces the ECG embeddings to organize themselves according to the clinical geometry of the CMR space, preventing the contrastive update from distorting the underlying structural manifold.

## 3.4 Downstream Unimodal Adaptation

Following pre-training, the CMR branch is discarded. The aligned ECG encoder $f _ { \theta }$ is deployed unimodally on the Chagas disease dataset (CODE-15% and SaMi-Trop). We evaluate it as a frozen feature extractor with a linear classification head trained on Chagas labels. Because neither Chagas patients nor Chagasspecific structural remodeling were present in the UK Biobank pre-training data, performance on this task directly measures the zero-shot generalizability of the learned structural inductive biases to a neglected cardiovascular domain.

## 4 Experiments

## 4.1 Datasets

We pretrain our proposed method on the UK Biobank [18]. The UK Biobank provides 63,193 matched 12-lead ECG and multi-view cine CMR volumes. We use 47,683 for alignment training, and 15,510 for validation and held-out evaluation, stratified by LVEF, LV mass, sex and AFib diagnosis. UK Biobank ECGs are 10-second 12-lead recordings sampled at 500 Hz; each is split into two 5-second segments compatible with ECG-FM input. During training, either of these segments is stochastically sampled as data augmentation. CMR sequences provide short-axis and long-axis (2-, 3-, 4-chamber) cine views.

We evaluate on the training data released for the PhysioNet 2025 [15]. The combined datasets of CODE-15% [16] and SaMi-Trop [3] yields approximately 345,000 ECGs with 8,192 Chagas-positive cases (∼2.4% prevalence). For the challenge-protocol comparison we additionally include PTB-XL [21] as additional Chagas-negative training data (21,801 ECGs), matching the evaluation pool of van Santvliet et al. [20].

## 4.2 Experimental Setup

For the ablation study, we first compare our proposed model against ECG-FM (no alignment), which uses the frozen ECG-FM encoder without crossmodal alignment, establishing the performance of ECG-only self-supervised pretraining. We further compare our method with Jidling et al. [10], who trained an ensemble of fifteen 1D-ResNet models end-to-end using the full CODE dataset together with SaMi-Trop. Finally, we compare against van Santvliet et al. [20], the winning approach in the PhysioNet 2025 Chagas Challenge, which fine-tuned a ViT-based ECG foundation model on CODE-15%, SaMi-Trop, and PTB-XL.

## 4.3 Implementation Details

ECG-FM is fine-tuned with AdamW $( \eta _ { \mathrm { e n c } } = 1 0 ^ { - 5 } , \eta _ { \mathrm { h e a d } } = 1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 3 } )$ , batch size 256 on a single NVIDIA A100, with cosine annealing and early stopping on held-out linear probing scores. For downstream Chagas prediction, layer 9 of the aligned encoder serves as a frozen feature extractor with a linear head trained in five-fold cross-validation. Layer 9 was selected independently of Chagas labels based on linear-probe performance across clinically relevant cardiac phenotypes. Class imbalance is addressed via positive-class reweighting and checkpoint selection uses AUPRC. The challenge submission ensembles this linear probe with a full end-to-end fine-tuning run initialized from the contrastive checkpoint $( \eta _ { \mathrm { e n c } } = 1 0 ^ { - 5 } , \eta _ { \mathrm { h e a d } } = 1 0 ^ { - 4 } )$

## 5 Results

In this section, we evaluate the value of performing cross modal alignment as a pretraining step, and compare our proposed method with existing state-of-theart methods for prediction of Chagas disease.

Value of cross-modal alignment. To evaluate the benefit of learning from imaging during the pretraining phase, we compared our proposed method with the baseline ECG-FM method before cross modal pretraining. The results reported on Table 1 show an increase of 5 points in Top5%-TPR, showing that CMR-derived structural information, such as ventricular geometry or myocardial mass, transferred through contrastive alignment, provides genuine diagnostic value for Chagas cardiomyopathy detection, even though none of this structural information was observed in the context of Chagas disease during pre-training.

Comparison with existing methods for Chagas prediction. Table 1 reports five-fold cross-validation on CODE-15% and SaMi-Trop, enabling direct comparison to the Jidling et al. baseline. ECG-CMR contrastive pre-training outperforms the unaligned ECG-FM baseline by 5.0 percentage points in Top5%- TPR and surpasses the AUROC of 0.80 reported by Jidling et al., whose model was trained specifically on Chagas data. Our encoder, by contrast, has never encountered a Chagas patient or Chagas-related CMR.

Table 1. Five-fold cross-validation on $\mathrm { C O D E - 1 5 \% + \Delta S a M i - T r o p }$ (without PTB-XL). All models use a frozen encoder and linear probe. The Jidling et al. model was trained end-to-end on Chagas data; ours was not. Dash line indicates values not reported.
<table><tr><td>Method</td><td>Top5%-TPR AUROC</td></tr><tr><td>Jidling et al. [10]</td><td>0.800</td></tr><tr><td>ECG-FM (no alignment)</td><td> $0 . 3 7 7 \pm 0 . 0 1 4$  0.827</td></tr><tr><td>Ours (CMR-aligned, linear probe)  $\mathbf { 0 . 4 2 7 \pm 0 . 0 2 2 }$ </td><td>0.851</td></tr></table>

Table 2. Five-fold cross-validation on CODE-15% + SaMi-Trop + PTB-XL (challenge protocol). Both our method and the Van Santvliet et al. frozen ablation use a frozen encoder and linear probe. Dash line indicates values not reported.
<table><tr><td>Method</td><td>Top5%-TPR</td><td>AUROC</td></tr><tr><td>Van Santvliet et al. [20] (linear probe)</td><td> $0 . 3 8 1 \pm 0 . 0 0 3$ </td><td></td></tr><tr><td>Ours (CMR-aligned, linear probe)</td><td> $\mathbf { 0 . 4 3 1 \pm 0 . 0 0 5 0 . 8 5 2 \pm 0 . 0 0 7 }$ </td><td></td></tr><tr><td></td><td></td><td></td></tr></table>

Table 2 reports results under the full challenge protocol. Our frozen linear probe outperforms the challenge winner’s frozen-backbone ablation by 5.0 percentage points in Top5%-TPR; the remaining gap to their end-to-end fine-tuned result reflects the frozen-probe constraint rather than a diference in representation quality. Table 3 reports our oficial submission on the hidden test sets. Our overall Challenge score (0.269) places just below the top three teams (0.280– 0.323). Two results stand out. First, on the SaMi-Trop-3 dataset we attain the highest AUROC among compared methods (0.773 vs. 0.767 for the winner). Second, on ELSA-Brasil, the hardest cohort owing to demographic and acquisition diferences from the training pool, our submission achieves the best Challenge score (0.132).

Table 3. Oficial PhysioNet/CinC 2025 Chagas Challenge test-set results (challenge score / AUROC per cohort). Overall is the mean challenge score across the three test cohorts, matching the oficial leaderboard aggregation. Ranked teams had access to the REDS-II validation set for threshold calibration; we did not.
<table><tr><td>Method</td><td>REDS-II</td><td></td><td>SaMi-Trop-3</td><td></td><td>ELSA-Brasil</td><td>Overall</td></tr><tr><td>1st, Van Santvliet et al. [20]</td><td>0.468</td><td>0.777 0.376</td><td>0.767</td><td>0.125</td><td>0.566</td><td>0.323</td></tr><tr><td>2nd, Nicolson et al.</td><td>0.357</td><td>0.735</td><td>0.375</td><td>0.739</td><td>0.118 0.567</td><td>0.283</td></tr><tr><td>3rd, Hong et al.</td><td>0.382 0.704</td><td>0.329</td><td>0.749</td><td>0.129</td><td>0.626</td><td>0.280</td></tr><tr><td>Ours</td><td>0.350</td><td>0.739</td><td>0.326</td><td>0.773</td><td>0.132 0.556</td><td>0.269</td></tr></table>

## 6 Discussion

We presented an imaging-supervised ECG pre-training framework for Chagas disease detection. By aligning ECG representations with a clinically grounded CMR target space on the UK Biobank, we distill structural cardiac knowledge into a deployable ECG encoder without needing access to CMR during inference.

Our results show that contrastive ECG-CMR pre-training provides diagnostic benefit beyond the diseases and populations present in the imaging cohort. By comparing against an unaligned ECG baseline, we show that structural cardiac knowledge distilled from paired CMR transfers to Chagas cardiomyopathy, despite Chagas-specific imaging manifestations not being explicitly represented during pre-training. The results suggest that the model captures general structural and functional cardiac characteristics that remain informative for Chagas disease across populations. These results could be extrapolated in future work to other cardiac conditions where imaging is unavailable, suggesting a broader pathway for improving ECG-based diagnosis of neglected diseases in resourceconstrained settings.

We have also positioned our method within the current landscape of Chagas disease prediction. The CMR-aligned encoder outperforms both ECG-only selfsupervised baselines and models trained end-to-end directly on Chagas-labeled data, establishing a new reference point for what is achievable without any disease-specific supervision. On the oficial challenge benchmark, while overall performance was just below top-ranked, results on the ELSA-Brasil cohort outperform those of the top three challenge winners. As the dataset representing the greatest demographic and acquisition shift from the training distribution, this improvement suggests that structural features learned from imaging transfer most reliably to populations and settings furthest from the pre-training data. Future work could build on these findings by incorporating additional cardiac imaging modalities during pre-training to further enrich the structural and functional information transferred to the ECG encoder. Moreover, the present analysis does not identify which specific structural phenotypes contribute to Chagas prediction. Investigating the relationship between the learned ECG representations and Chagas-relevant cardiac manifestations would provide further insight into the mechanisms underlying this cross-disease transfer.

From a translational perspective, the proposed framework separates the resource intensive multimodal pre-training stage from downstream application. While developing the representation requires access to large paired ECG–CMR datasets and substantial computational resources, these requirements are limited to pre-training. Once trained, the CMR branch is discarded and the resulting encoder operates using only a standard 12-lead ECG. This creates the possibility of learning multimodal cardiac representations in data-rich settings and subsequently transferring them to populations and healthcare settings where advanced cardiac imaging is less accessible. Future work should evaluate this transfer prospectively across Latin American populations and assess the computational requirements for deployment in resource-constrained settings.

## 7 Impact in RCS

By distilling structural cardiac knowledge into an ECG encoder, this framework enables diagnosis in settings where advanced imaging infrastructure is not available.

## References

1. Alvarez-Florez, L., Bujalance-Gomez, A., Raijmakers, F., Ruiperez-Campillo, S., Kolk, M.Z.H., Wiers, J., Vogt, J., Bekkers, E.J., Išgum, I., Tjong, F.V.Y.: Dualphase cross-modal contrastive learning for cmr-guided ecg representations for cardiovascular disease assessment (2026)

2. Brady, A.P., Paulo, G., Brkljacic, B., Loewe, C., Szucsich, M., Hierath, M., of Radiology, E.S.: Current status of radiologist stafing, education and training in the 27 eu member states. Insights into Imaging 16(1), 59 (2025)

3. Cardoso, C.S., Sabino, E.C., Oliveira, C.D.L., de Oliveira, L.C., Ferreira, A.M., Cunha-Neto, E., Bierrenbach, A.L., Ferreira, J.E., Haikal, D.S., Reingold, A.L., et al.: Longitudinal study of patients with chronic chagas cardiomyopathy in brazil (sami-trop project): a cohort profile. BMJ open 6(5), e011181 (2016)

4. Ding, Z., Hu, Y., Li, Z., Zhang, H., Wu, F., Xiang, Y., Li, T., Liu, Z., Chu, X., Huang, Z.: Cross-modality cardiac insight transfer: A contrastive learning approach to enrich ecg with cmr features. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 109–119. Springer (2024)

5. Ding, Z., Li, Z., Hu, Y., Xu, Y., Zhao, C., Mao, Y., Li, H., Li, Z., Li, Q., Wang, J., Chen, Y., Chen, M., Wang, L., Chu, X., Pan, W., Liu, Z., Wu, F., Zhang, H., Chen, T., Huang, Z.: Generating cardiac magnetic resonance images from electrocardiograms — a multicenter study. NEJM AI 3(4) (2026)

6. Echavarría, N.G., Echeverría, L.E., Stewart, M., Gallego, C., Saldarriaga, C.: Chagas disease: Chronic chagas cardiomyopathy. Current Problems in Cardiology 46(3), 100507 (2021)

7. Erlacher, L., Agostini, A., Ruiperez-Campillo, S., Ozkan, E., Sutter, T.M., Vogt, J.E.: SwissBeatsNet: A multilead masked autoencoder for chagas disease detection. In: Computing in Cardiology (2025)

8. Fu, Y., Bai, W., Yi, W., Manisty, C., Bhuva, A.N., Treibel, T.A., Moon, J.C., Clarkson, M.J., Davies, R.H., Hu, Y.: A versatile foundation model for cine cardiac magnetic resonance image analysis tasks. arXiv preprint (2025)

9. Hilabi, B.S., Alghamdi, S.A., Almanaa, M., Alghamdi Sr, S.A.: Impact of magnetic resonance imaging on healthcare in low-and middle-income countries. Cureus 15(4) (2023)

10. Jidling, C., Gedon, D., Schön, T.B., Oliveira, C.D.L., Cardoso, C.S., Ferreira, A.M., Giatti, L., Barreto, S.M., Sabino, E.C., Ribeiro, A.L., et al.: Screening for chagas disease from the electrocardiogram using a deep neural network. PLoS neglected tropical diseases 17(7), e0011118 (2023)

11. Li, J., Aguirre, A.D., Junior, V.M., Jin, J., Liu, C., Zhong, L., Sun, C., Cliford, G., Brandon Westover, M., Hong, S.: An electrocardiogram foundation model built on over 10 million recordings. Nejm ai 2(7), AIoa2401033 (2025)

12. McKeen, K., Masood, S., Toma, A., Rubin, B., Wang, B.: ECG-FM: an open electrocardiogram foundation model. JAMIA Open 8(5) (2025)

13. Mehari, T., Strodthof, N.: Self-supervised representation learning from 12-lead ecg data. Computers in Biology and Medicine 141, 105114 (12 2021)

14. Pennell, D.J.: Cardiovascular magnetic resonance. Circulation 121(5), 692–705 (2010)

15. Reyna, M.A., Koscova, Z., Pavlus, J., Weigle, J., Saghafi, S., Gomes, P., Elola, A., Hassannia, M.S., Campbell, K., Bahrami Rad, A., Ribeiro, A.H., Ribeiro, A.L., Sameni, R., Cliford, G.D.: Detection of Chagas Disease from the ECG: The George B. Moody PhysioNet Challenge 2025. In: Computing in Cardiology 2025. vol. 52, pp. 1–4 (2025)

16. Ribeiro, A.H., Paixão, G.M., Lima, E.M., Ribeiro, M.H., Pinto Filho, M.M., Gomes, P.R., Oliveira, D.M., Meira Jr, W., Schon, T.B., Ribeiro, A.L.P.: Code-15%: A large scale annotated dataset of 12-lead ecgs. Zenodo, Jun 9, 10–5281 (2021)

17. Selivanov, A., Müller, P., Turgut, Ö., Stolt-Ansó, N., Rueckert, D.: Global and Local Contrastive Learning for Joint Representations from Cardiac MRI and ECG. In: Medical Image Computing and Computer Assisted Intervention, MICCAI 2025 - 28th International Conference, 2025, Proceedings. pp. 217–227 (2026)

18. Sudlow, C., Gallacher, J., Allen, N., Beral, V., Burton, P., Danesh, J., Downey, P., Elliott, P., Green, J., Landray, M., Liu, B., Matthews, P., Ong, G., Pell, J., Silman, A., Young, A., Sprosen, T., Peakman, T., Collins, R.: Uk biobank: An open access resource for identifying the causes of a wide range of complex diseases of middle and old age. PLOS Medicine 12(3), 1–10 (2015)

19. Özgün Turgut, Müller, P., Hager, P., Shit, S., Starck, S., Menten, M.J., Martens, E., Rueckert, D.: Unlocking the diagnostic potential of electrocardiograms through information transfer from cardiac magnetic resonance imaging. Medical Image Analysis 101, 103451 (2025)

20. Van Santvliet, L., Nguyen, P.X., Vandenberk, B., De Vos, M.: Detecting chagas disease using a vision transformer-based ecg foundation model. Proceedings of CinC 2025 52 (2025)

21. Wagner, P., Strodthof, N., Bousseljot, R.D., Samek, W., Schaefter, T.: PTB-XL, a large publicly available electrocardiography dataset. PhysioNet (2022)