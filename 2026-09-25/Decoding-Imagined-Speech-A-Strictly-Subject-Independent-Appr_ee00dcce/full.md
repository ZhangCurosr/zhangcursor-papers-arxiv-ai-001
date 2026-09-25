# Decoding Imagined Speech: A Strictly Subject-Independent Approach Using EEG

1<sup>st</sup> Frederik Møllskov Trier

2<sup>nd</sup> Xiaopeng Mao

3<sup>rd</sup> Sadasivan Puthusserypady

Department of Health Technology

Department of Health Technology

Department of Health Technology

Technical University of Denmark

Technical University of Denmark

Technical University of Denmark

Ørsteds Plads, 2800

Ørsteds Plads, 2800

Ørsteds Plads, 2800

Kgs. Lyngby, Denmark

Kgs. Lyngby, Denmark

s211923@student.dtu.dk

xiama@dtu.dk

Kgs. Lyngby, Denmark

sapu@dtu.dk

Abstract—Imagined speech decoding from electroencephalography (EEG) has gained increasing attention as a potential communication pathway for individuals with severe motor impairments, yet reported performance often relies on evaluation protocols that do not clearly reflect cross-subject generalization. This study presents a transparent baseline investigation of a multi-class imagined speech EEG dataset under a strictly subjectindependent evaluation framework. Two preprocessing and feature extraction pipelines were compared: a time-domain statistical feature approach and a frequency-domain spectral bandpower approach, evaluated using subject-wise cross-validation and trial-level majority voting with a random forest classifier. The spectral pipeline achieved a significantly higher mean trialwise accuracy than the statistical pipeline (49.03 ± 4.18% vs. 37.97 ± 3.79%) for coarse-level classification across subjects. Forward feature selection further indicated that a limited subset of frequency bands captured most of the discriminative information. Overall, this work provides a strong basis for future brain-computer interface studies targeting improved crosssubject generalization in EEG-based imagined speech decoding.

Index Terms—Imagined speech decoding, EEG, brain–computer interface, subject-independent evaluation, spectral features, machine learning.

## I. INTRODUCTION

Communication is central to human life, and the inability to express one-self can be extremely challenging. Patients suffering from locked-in syndrome have functioning cognition, but very limited motor control, resulting in severely restricted communication [1]. Imagined speech decoding has been proposed as a potential future solution, with the idea revolving around translation of brain activity into text or symbols as a means of communication. However, this technology is far from being implemented in real life situations due to the complexity of brain signals and the inherent low signal-to-noise ratio in electroencephalography (EEG) [2]. Decoding brain activity is a complex task that relies on the quality of the input features to a given machine learning (ML) model [3].

The publicly available imagined speech EEG dataset introduced by Kumar et al. [4], commonly referred to as Kumar’s EEG dataset, has become a widely used benchmark. The dataset contains three ”coarse-grained” classes (letters, digits, and objects), each further divided into ten ”fine-grained”

subclasses. In the original study, simple statistical time-domain features extracted from minimally preprocessed EEG were classified using a random forest (RF) model, achieving an accuracy of 85.20% and 67.03% at the coarse and finegrained classification levels, respectively. Subsequent studies have applied more advanced models, including convolutional neural networks [5], [6] and transformer-based architectures [7], both of which outperform the original approach, with the latter achieving the highest reported fine-grained accuracy of approximately 97%. However, recent work on EEG-totext decoding has shown that reported performance can be substantially inflated by leaky evaluation schemes, leading to unreliable assessments of model capability [8]. Despite its widespread adoption, the Kumar dataset is often evaluated under validation protocols that are not described in sufficient detail, which limits comparability across studies. In particular, it is unclear how the existing works have conducted the data split, i.e., whether the data are split at the window or trial level and in a subject-dependent or subject-independent manner. As a result, the generalizability of the existing works remains uncertain despite their great performances. For practical brain–computer interface (BCI) applications, generalization across subjects is particularly important and appropriately assessed using strictly subject-independent evaluation schemes.

Accordingly, this work aims to establish a transparent and reproducible framework for imagined speech decoding on Kumar’s EEG dataset under a strictly subject-independent evaluation protocol. Two distinct preprocessing and featureextraction pipelines are compared with respect to their downstream classification performance using an RF classifier. The first pipeline is based on the statistical time-domain feature framework proposed by [4], while the second pipeline focuses on frequency-domain spectral features. Both pipelines are evaluated using subject-wise cross-validation (CV) and trial-level majority voting. To the best of our knowledge, this study provides the first evaluation of the Kumar dataset under these constraints, offering a realistic reference point for future methodological advances with respect to cross-subject generalization.

![](images/010266859521b7b15837d9fb9300871d8ceff20ca387a12eab070688e7ed9f67.jpg)  
Fig. 1. Conceptual overview of the pipeline comparison approach. The diagram goes from left to right. The dataset is preprocessed and feature-extracted i two different manner, but cross-validation (CV) is identical. Finally, their performances are compared statistically.

## II. METHODOLOGY

This section outlines the key methods employed in the study and includes an explanation of the comparison framework, details of Kumar’s EEG dataset, and finally a description of each pipeline.

## A. Proposed Method

This study compares two different preprocessing and feature extraction pipelines, denoted as the Kumar Pipeline and the Spectral Pipeline. A general overview of the pipeline comparison is illustrated in Fig. 1 to clarify the methodological approach.

## B. Dataset

The data, as described in [4], were collected using Emotiv EPOC+ with 14 channels (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, and AF4) placed according to the international 10-20 system, with DRL and CMS, positioned above the ears, serving as reference electrodes. The EEG was captured at 2048 Hz and later down-sampled to 128 Hz. Only the down-sampled data is publicly available. The experimental setup was as follows: Each participant looked at a computer screen where an object was presented, followed by 10 s of imagination with closed eyes. A 20-second rest period was provided between imagination trials to allow the participant to return to a resting state. EEG data from 23 subjects aged 15-40 were collected while imagining the following targets: characters (A,C,F,H,J,M,P,S,T,Y), digits (0-9) and images from everyday life (apple, car, dog, ring, phone, rose, scooter, tiger, wallet, and watch). Fig 2 illustrates representative examples from each class. A total of 690 (23 × 30) EEG recordings were obtained, each lasting 10 seconds. Note that this work mainly focuses on coarse-level classification, discriminating between characters, digits, and images.

## C. Common Preprocessing and Segmentation

This section describes the preprocessing and segmentation steps shared across both pipelines. Inspection of the publicly available dataset revealed several inconsistencies relevant to data handling, which are briefly summarized here. Each trial contains 12 s of EEG data without explicit markers indicating trial onset; therefore, the central 10 s segment was retained for analysis. Additionally, folders of the digit condition contain indices ranging from 0–24, with indices 2 and 18 missing, whereas the character and image conditions span from 0–22. As the dataset documentation consistently reports 23 subjects, these discrepancies were attributed to indexing errors. Accordingly, all folders were manually re-indexed prior to data loading, without removing or modifying any trials.

Following pipeline-specific filtering, which will be described in the subsequent sections, each 10 s trial was segmented into 1 s windows with 75% overlap, with each window containing data from all 14 channels. A window length of 1 s provides an appropriate trade-off between temporal resolution and frequency resolution for subsequent spectral feature extraction. Prior to classification, subject-wise z-score normalization was applied in accordance with the subjectindependent CV scheme and is defined as:

$$
z = { \frac { x - \mu } { \sigma } } ,\tag{1}
$$

where x denotes the feature value, z the normalized feature, and $\mu$ and σ are the subject-wise mean and standard deviation of the given feature, respectively.

![](images/e4e041b4d9c19165ac4cf849bd64d31e612351d8a432e98dbd74ff163ccce459.jpg)  
Fig. 2. Examples of stimulus within the coarse-grained classes—characters, digits, and images—presented from left to right. These images are adapted from [4].

![](images/79598c755c8872899911802483228e3d537fa2cb19baa3342986df3e697e5167.jpg)  
Fig. 3. Schematic overview of the subject-wise cross-validation and trial-level evaluation procedure. In each fold, models are trained on a subset of subjects and tested on held-out subjects. Window-level predictions are aggregated to the trial level using majority voting (MV), and final performance is obtained by averaging trial-wise accuracies across folds.

## D. Kumar Pipeline

This pipeline closely follows the feature extraction approach proposed by Kumar et al. [4]. The down-sampled EEG data (128 Hz) were first smoothed using a 5-point moving average filter. Trials were then segmented as described previously; the use of 1 s overlapping windows differs from the original pipeline and was introduced to ensure comparability with the spectral pipeline.

Four statistical features—sd, rms, sum, and signal energy—were computed per channel for each window, resulting in a 56-dimensional feature vector (14 channels × 4 features) and a feature matrix of size 27600 × 56.

Inspection of the extracted features revealed that the sd exhibited a positively skewed distribution; therefore, a logarithmic transformation was applied prior to normalization. Finally, all features were normalized as described in Section II-C.

## E. Spectral Pipeline

The raw EEG data were first filtered with a 50 Hz finite impulse response notch filter with zero phase to remove powerline interference, while preserving phase characteristics. A zero-phase 4th-order infinite impulse response Butterworth bandpass filter (0.5–60 Hz) was applied to reduce slow drift while preserving the temporal structure of the EEG signal, which is required for accurate fast-Fourier transform-based spectral feature estimation. Following filtering, trials were segmented as described previously.

Spectral features were extracted by estimating the power spectral density (PSD) of each window using a periodogram approach. Absolute band power was computed by numerically integrating the PSD within each frequency band using the trapezoidal rule:

$$
P _ { b } = \sum _ { f \in b } S _ { x x } ( f ) \Delta f ,\tag{2}
$$

where $P _ { b }$ denotes the power within frequency band $b ,$ $S _ { x x } ( f )$ is the PSD estimate at frequency $f ,$ and $\Delta f$ is the frequency resolution. Frequency bands were defined as halfopen intervals to avoid overlap: delta [0.5, 4) Hz, theta [4, 8) Hz, alpha [8, 14) Hz, beta [14, 30) Hz, and gamma [30, 64] Hz. [9], [10]. The upper limit of 64 Hz for the gamma band is determined by the Nyquist theorem, given the sampling frequency of 128 Hz [11]. As band-power features typically exhibit positively skewed distributions, a logarithmic transformation was applied to reduce skewness, followed by z-score normalization. Each window was represented by a 70- dimensional feature vector (14 channels × 5 features), resulting in a feature matrix of size 27600 × 70 after aggregation across all trials.

## F. Evaluation and Statistical Analysis

Classification was performed using an RF classifier, consistent with the original work of Kumar et al. [4]. The RF employed bootstrap aggregation, majority voting across decision trees, and the Gini index as the impurity criterion. The number of decision trees was fixed at 200, while all remaining hyperparameters were set to their default values in scikit-learn.

Model performance was evaluated using a 5-fold CV scheme with subject-wise splits to ensure subject independence (see Fig. 3). Stratification and grouping by subject were implemented using Python scikit-learns StratifiedGroupKFold, which ensures constant class distribution for the training and test folds. Within each test fold, window-level predictions were aggregated to the trial level using majority voting. Accuracy for each test fold was defined as the trial-level accuracy across all test subjects. Final model performance was obtained by averaging trialwise accuracies across the five folds. For model comparison, trial-wise accuracies were first averaged within each subject to avoid subject-wise correlation between predictions. The resulting mean subject-wise accuracies were compared between pipelines using a paired Wilcoxon signed-rank test, which is recommended for classifier comparisons [12]. Let $d _ { i } = a _ { i } ^ { ( 1 ) } - a _ { i } ^ { ( 2 ) }$ denote the difference in mean subject-wise trial accuracies between the two pipelines for subject i, i.e. $a _ { i } ^ { ( 1 ) }$ and $a _ { i } ^ { ( 2 ) }$ . The Wilcoxon signed-rank test statistic is defined as:

$$
T = \sum _ { i = 1 } ^ { N } \mathrm { s g n } ( d _ { i } ) R _ { i } ,\tag{3}
$$

where $R _ { i }$ is the rank of $| d _ { i } |$ among all non-zero absolute differences $\{ | d _ { 1 } | , \dotsc , | d _ { N } | \}$ , N is the number of subjects, and sgn(·) denotes the sign function [13].

## G. Forward Feature Selection

In addition to the pipeline performance comparison, a forward feature selection procedure was applied to the spectral pipeline to assess the contribution of individual frequency bands to classification performance [14]. Starting from singleband feature sets, bands were incrementally added based on improvements in trial-wise classification accuracy, using the same subject-wise CV and evaluation protocol as described above.

## III. RESULTS

While the Kumar’s EEG dataset also supports fine-level classification, the present study focuses on coarse-level decoding. The mean trial-wise accuracies for coarse-level prediction are reported in Table I. The Spectral Pipeline achieved a higher accuracy than the Kumar Pipeline $( 4 9 . 0 3 \pm 4 . 1 8 \%$ vs. $3 7 . 9 7 \pm 3 . 7 9 \ \% )$ . The difference in pipeline performance was statistically significant according to the paired Wilcoxon signed-rank test $( \mathtt { p } = 0 . 0 1 8 )$ , with a large effect size (rankbiserial correlation = 0.57). Table II displays the class-wise accuracies for each pipeline. Both pipelines achieved the highest accuracy for the character class, while performance for digits and images was lower.

TABLE I  
FINAL EVALUATION OF PIPELINE PERFORMANCE
<table><tr><td>Pipeline</td><td> $\overline { { \mathrm { A c c u r a c y ~ } ( \% ) } }$  一</td></tr><tr><td>Spectral Pipeline</td><td> $\mathbf { 4 9 . 0 3 \pm 4 . 1 8 }$ </td></tr><tr><td>Kumar Pipeline</td><td> $3 7 . 9 7 \pm 3 . 7 9$ </td></tr></table>

TABLE II  
CLASS-WISE CLASSIFICATION ACCURACY FOR EACH PIPELINE.
<table><tr><td>Class</td><td>Kumar Pipeline</td><td> $\overline { { { \mathrm { S p e c t r a l ~ P i p e l i n e } } } }$ </td></tr><tr><td>Character</td><td> $\mathbf { 4 8 . 9 0 \pm 9 . 1 9 }$ </td><td> $\mathbf { \overline { { 5 6 . 3 0 \pm 8 . 9 4 } } }$ </td></tr><tr><td>Digit</td><td> $3 6 . 9 0 \pm 8 . 7 0$ </td><td> $4 6 . 0 0 \pm 1 0 . 2 4$ </td></tr><tr><td>Image</td><td> $2 9 . 4 0 \pm 1 4 . 1 6$ </td><td> $4 8 . 5 0 \pm 1 3 . 9 4$ </td></tr></table>

A forward selection procedure was applied to determine which individual frequency band, or combination of bands, yielded the highest classification accuracy. As depicted by Table III, the alpha band achieved the highest single-band mean accuracy. Adding the delta band resulted in a substantial performance increase, while the highest mean accuracy was obtained using four bands $( \alpha + \delta + \gamma + \theta )$ , excluding beta. All band combinations were statistically compared against the fullband model using the Wilcoxon test. A Bonferroni correction was applied to account for multiple comparisons, and no statistically significant differences were observed.

TABLE III  
STEPWISE FORWARD SELECTION OF FREQUENCY BANDS. BOLD VALUES INDICATE MEAN ACCURACIES EXCEEDING THE FULL-BAND MODEL.
<table><tr><td>Step</td><td>Band set</td><td>Accuracy (%)</td></tr><tr><td>1</td><td>α</td><td> $\overline { { 4 6 . 6 7 \pm 6 . 9 3 } }$ </td></tr><tr><td>2</td><td> $\alpha + \delta$ </td><td> ${ \bf 5 1 . 1 7 \pm 5 . 6 6 }$ </td></tr><tr><td>3</td><td> $\alpha + \delta + \gamma$ </td><td> ${ \bf 5 1 . 0 3 \pm 6 . 5 8 }$ </td></tr><tr><td>4</td><td> $\alpha + \delta + \gamma + \theta$ </td><td> ${ \pm } { \bf 2 . 8 3 } \pm { \bf 4 . 6 4 }$ </td></tr><tr><td>5</td><td> $\alpha + \delta + \gamma + \theta + \beta$ </td><td> $4 9 . 0 3 \pm 4 . 1 8$ </td></tr></table>

## IV. DISCUSSION

The Spectral Pipeline outperformed the Kumar Pipeline, indicating that frequency-domain features generalize better across subjects than time-domain statistical features in this setting. Forward feature selection showed that this improvement could not be explained by the higher feature dimensionality, as a comparable performance was achieved using only a subset of frequency bands (see Table III). Both pipelines achieved higher accuracy for characters than for digits and images, indicating class-dependent differences in signal discriminability (see Table II). Absolute classification accuracies were lower than those reported by Kumar et al. [4]. This was expected given the use of a strictly subject-independent evaluation protocol, which provides a more realistic estimate of generalization performance for practical BCI implementation. Without careful and realistic evaluation, models that seemingly perform well may still fail in real-world scenarios. The forward selection analysis further indicated that not all frequency bands contribute equally to class separation. Alpha-band power showed the highest discriminative value, potentially reflecting the closedeyes imagination task. The second-highest mean trial-wise accuracy in Table III was found using only the alpha and delta band. This was not significantly different from the full-band model, suggesting that much of the relevant information for coarse-level imagined speech classification can be captured by a limited set of frequency bands.

The minimal artifact removal in the pipelines is a limitation of this work, as the removal of artifacts improves data quality and enables a better representation of relevant brain activity as a foundation for ML [15]. Channel selection was not performed in this work and may prove to be of interest, as prior imagined speech studies have shown that selecting a subset of task-relevant EEG channels can enhance discriminative performance [16]. Cortical regions may contribute differently depending on the given task, and within this experimental framework, visual imagery might engage the occipital electrodes to a higher degree. Since the Spectral Pipeline only used absolute band power as spectral features, future work may benefit from including additional spectral features, as in [17], which may provide further discriminative information for imagined speech decoding.

## V. CONCLUSION

This study established a transparent subject-independent investigation of imagined speech decoding using the widely adopted Kumar EEG dataset. In this work, a time-domain statistical feature pipeline was compared with a frequencydomain band-power feature pipeline under a strict subjectwise CV. This provided a realistic assessment of crosssubject generalization, which is essential for practical BCI applications. The spectral pipeline consistently outperformed the statistical approach, demonstrating that frequency-domain representations offer more robust and transferable information for coarse-level imagined speech classification.

Forward feature selection further revealed that a limited subset of frequency bands, particularly alpha and delta captured most of the discriminative structure in the data, indicating that improved generalization does not require high-dimensional feature spaces. The observed class-dependent performance differences also highlight the inherent variability in the discriminability of imagined speech categories.

Overall, this work offers a reproducible and clearly defined baseline for future research on subject-independent imagined speech decoding. The findings underscore the importance of feature representation when evaluation protocols are aligned with real-world constraints and provide a foundation for developing more advanced, generalizable EEG-based BCI systems.

## REFERENCES

[1] S. Laureys et al., “The locked-in syndrome: what is it like to be conscious but paralyzed and voiceless?,” Progress in Brain Research, vol. 150, pp. 495–611, Jan. 2005, doi: 10.1016/s0079-6123(05)50034-7.

[2] D. Lopez-Bernal, D. Balderas, P. Ponce, and A. Molina, “A State-ofthe-Art Review of EEG-Based Imagined Speech Decoding,” Frontiers in Human Neuroscience, vol. 16, p. 867281, Apr. 2022, doi: 10.3389/fnhum.2022.867281.

[3] M. Saeidi et al., “Neural Decoding of EEG Signals with Machine Learning: A Systematic Review,” Brain Sciences, vol. 11, no. 11, p. 1525, Nov. 2021, doi: 10.3390/brainsci11111525.

[4] P. Kumar, R. Saini, P. P. Roy, P. K. Sahu, and D. P. Dogra, “Envisioned speech recognition using EEG sensors,” Personal and Ubiquitous Computing, vol. 22, no. 1, pp. 185–199, Sep. 2017, doi: 10.1007/s00779- 017-1083-4.

[5] P. Tirupattur, Y. Rawat, C. Spampinato, and M. Shah, “ThoughtViz: Visualizing human thoughts using generative adversarial network,” in Proc. 2018 ACM Multimedia Conference (ACM MM), Oct. 2018, doi: 10.1145/3240508.3240641.

[6] A. Tripathi, “Analysis of EEG frequency bands for envisioned speech recognition,” arXiv:2203.15250 [eess.SP], 2022. Available: https://arxiv.org/abs/2203.15250.

[7] I. Gallo and S. Corchs, “Thinking is like processing a sequence of spatial and temporal words,” in Proc. 2024 Int. Joint Conf. Neural Networks (IJCNN), Yokohama, Japan, 2024, pp. 1–8, doi: 10.1109/IJCNN60899.2024.10650922.

[8] H. Jo, Y. Yang, J. Han, Y. Duan, H. Xiong, and W. H. Lee, “Are EEGto-text models working?” arXiv preprint, arXiv:2405.06459, 2024.

[9] M. Abo-Zahhad, S. M. Ahmed, and S. N. Abbas, “A New EEG Acquisition Protocol for Biometric Identification Using Eye Blinking Signals,” International Journal of Intelligent Systems and Applications, vol. 7, no. 6, pp. 48–54, May 2015, doi: 10.5815/ijisa.2015.06.05.

[10] J. Fernandez, B. Innocenti, and B. Lopez, “EEG classification for´ neurological disorders using frequency band deciles,” Scientific Reports, vol. 15, no. 1, p. 45142, Dec. 2025, doi: 10.1038/s41598-025-33535-0.

[11] S. Puthusserypady, Applied Signal Processing. Now Published, 2021, ISBN: 978-1-68083-979-1.

[12] J. Demsar, “Statistical Comparisons of Classifiers overˇ Multiple Data Sets,” Journal of Machine Learning Research, vol. 7, no. 1, pp. 1–30, Dec. 2006, [Online]. Available: http://citeseerx.ist.psu.edu/viewdoc/summary? doi=10.1.1.141.3142

[13] J.D. Gibbons and S. Chakraborti, Nonparametric statistical inference: revised and expanded. CRC press, 2014.

[14] P. B. Brockhoff, J. K. Møller, E. W. Andersen, P. Bacher, and L. E. Christiansen, “Introduction to Statistics at DTU,” DTU Compute, Kgs. Lyngby, Denmark, 2018.

[15] S. Alzahrani, H. Banjar, and R. Mirza, “Systematic Review of EEG-Based Imagined Speech Classification Methods,” Sensors, vol. 24, no. 24, p. 8168, Dec. 2024, doi: 10.3390/s24248168.

[16] J. T. Panachakel and A. G. Ramakrishnan, “A novel deep learning architecture for decoding imagined speech using selected EEG channels,” arXiv:2003.09374, 2020.

[17] S. Chengaiyan, A. S. Retnapandian, and K. Anandan, “Identification of vowels in consonant–vowel–consonant words from speech imagery based EEG signals,” Cognitive Neurodynamics, vol. 14, no. 1, pp. 1–19, Oct. 2019, doi: 10.1007/s11571-019-09558-5.