# Cycle-Aware Autoencoder with Cross-Signal Consistency for Railway Door Anomaly Detection

Ammar Bouketta<sup>∗‡</sup>, Smail Niar<sup>†</sup>, Hamza Ouarnoughi<sup>†§</sup>, and Eva Mutuzo Brindle<sup>‡</sup>

<sup>∗</sup>Universite Polytechnique Hauts-de-France, LAMIH CNRS UMR 8201, Valenciennes, France´

<sup>†</sup>Universite Polytechnique Hauts-de-France, LAMIH CNRS UMR 8201, INSA Hauts-de-France, Valenciennes, France´ <sup>‡</sup>Alstom, Crespin, France

<sup>§</sup>Computer Science Department, College of Computing and Informatics, University of Sharjah, Sharjah, UAE Emails: <sup>∗†</sup>{firstname.lastname}@uphf.fr; <sup>‡</sup> eva.mutuzo-brindle@alstomgroup.com

Abstract—Passenger access doors are safety-critical subsystems in railway vehicles, yet detecting abnormal door behavior in real operation is challenging because faults are rare, diverse, and often unlabeled. This paper addresses railway door condition monitoring as a cycle-level unsupervised anomaly detection problem, where each complete opening–dwell–closing cycle is treated as a single monitoring unit. We propose the Temporal Cycle-Aware Attention Autoencoder with Cross-Signal Consistency (TCAA-CS), trained exclusively on nominal cycles. It combines a dualstream encoder that processes continuous physical measurements (position, current, voltage) and binary logical states (door-closed, door-locked) through separate 1D-CNN branches, an LSTM encoder with temporal attention pooling, and a triple hybrid anomaly score fusing reconstruction error, latent-space deviation, and phase-aware cross-signal consistency. The consistency term helps identify cases where individual signals appear plausible but their inter-signal relationships become physically or logically inconsistent. On real industrial data from a passenger train in commercial service, TCAA-CS achieves 93.8% recall, 97.3% precision, and a 0.5% false-alarm rate, outperforming representative unsupervised baselines. System-level evaluation on an NVIDIA Jetson AGX Xavier supports the feasibility of real-time onboard deployment.

Index Terms—Railway door, unsupervised anomaly detection, cycle-level monitoring, temporal attention autoencoder, dualstream encoding, cross-signal consistency, embedded deployment

## I. INTRODUCTION

Passenger access doors are safety-critical subsystems in railway vehicles. Abnormal door behavior can compromise passenger safety, reduce service availability, and increase maintenance costs. Reliable monitoring is therefore essential to support preventive maintenance and dependable railway operation. In this work, monitoring is treated as an onboard condition-monitoring function for maintenance support, rather than a certified safety-control function. Safety-critical door control remains handled by certified train systems. Monitoring passenger doors under real operating conditions is challenging. Onboard data predominantly reflect nominal behavior, while faults are rare, diverse, and generally unlabeled. Door operation is also inherently structured: each passenger access corresponds to a complete door operation cycle, composed of opening, dwell, and closing phases governed by coupled mechanical, electrical, and logical components. Moreover, the monitored signals are heterogeneous, combining continuous physical measurements, such as motor position, current, and supply voltage, with binary logical states, such as door-closed and door locked. Treating these signals uniformly may obscure signal-specific patterns and weaken the learned representation of normal behavior.

Abnormal behavior may not appear as an isolated signal deviation, but as an inconsistency across signals during a full door cycle. For example, a door may report a locked state while the position signal still indicates motion. Such inconsistencies are difficult to capture with methods that evaluate signals independently, rely only on reconstruction error, or operate on short windows instead of complete door cycles.

To address these challenges, we propose TCAA-CS, a cyclelevel unsupervised anomaly detection framework tailored to railway door monitoring. The method combines signal-typeaware encoding, attention-based temporal representation learning, and hybrid anomaly scoring to detect both signal-level deviations and cross-signal inconsistencies.

Contributions: The main contributions of this work are:

• We formulate railway door monitoring as a cycle-level unsupervised anomaly detection problem, where each opening–dwell–closing operation is treated as one monitoring unit.

• We propose TCAA-CS, a domain-tailored attention autoencoder that combines dual-stream physical/logical encoding with temporal attention pooling.

• We introduce a hybrid anomaly score combining reconstruction error, latent-space deviation, and phase-aware cross-signal consistency to detect both signal-level deviations and inter-signal inconsistencies.

• We validate TCAA-CS on real industrial railway door data, compare it with representative unsupervised baselines, and assess embedded deployment feasibility in terms of latency, memory, and energy consumption.

## II. RELATED WORK

## A. Fault Detection in Railway Door Systems

Railway passenger doors account for a substantial share of railway vehicle malfunctions [1], making their monitoring an important reliability and maintenance issue. Early studies mainly relied on model-based methods, such as parameter estimation from motor current signals [2] and Bond Graph modeling for fault detection and isolation [3]. These approaches are interpretable, but require accurate system models and can be sensitive to modeling assumptions.

Recent work has explored data-driven diagnosis. Ham et al. [4] compare handcrafted features with convolutional neural networks (CNNs) applied to motor current signals, while Sun et al. [5] use acoustic signals with empirical mode decomposition and support vector machines (SVMs). In the unsupervised setting, Shimizu et al. [6] combine deep autoencoders with a one-class SVM trained on nominal data, with follow-up work addressing transfer learning and generative adversarial network (GAN)-based domain adaptation across door types [1].

Despite this progress, three limitations remain. Existing methods often rely on a single signal modality, analyze samples or short windows rather than complete opening–dwell– closing cycles, and generally use a single detection criterion without explicitly assessing consistency between physical measurements and logical states.

## B. Anomaly Detection in Multivariate Time Series

Unsupervised anomaly detection in multivariate time series is commonly addressed using reconstruction-based autoencoders trained on nominal data [7]. More advanced methods introduce additional criteria, such as adversarial reconstruction in UnSupervised Anomaly Detection (USAD) [8], latent density estimation in the Deep Autoencoding Gaussian Mixture Model (DAGMM) [9], and association discrepancy in the Anomaly Transformer [10]. Although effective on general benchmarks, these methods usually process fixed-length windows, treat input variables as homogeneous channels, and do not explicitly evaluate phase-wise consistency between continuous physical measurements and binary logical states.

## C. Positioning of This Work

TCAA-CS addresses these gaps by aligning anomaly detection with railway door operation. It performs cycle-level detection, processes physical and logical signals through separate encoding streams, and combines reconstruction error, latentspace deviation, and phase-aware cross-signal consistency. The framework is evaluated on real industrial door data and assessed for onboard deployment feasibility in terms of latency, memory footprint, and energy consumption.

## III. DATASET OVERVIEW

The dataset was collected from onboard monitoring systems of a Regio 2N passenger train operating in regular commercial service. The train, manufactured by Alstom, consists of 5 cars equipped with 16 automatic passenger access doors. In this study, we use data recorded over 30 operating days.

The dataset covers multiple doors, cars, and operating days, and therefore includes natural variability induced by real commercial service.

The data predominantly reflect normal behavior, while fault events are rare and not systematically annotated. A small number of abnormal door operation cycles corresponding to real operational issues were identified through expert analysis, but they do not constitute exhaustive fault labeling. The dataset is industrial and proprietary. To improve transparency and reproducibility, we provide a precise description of the data organization, and cycle segmentation procedure.

## A. Cycle-Based Data Representation

Each door-day recording is segmented into door operation cycles. A cycle corresponds to one complete opening–dwell– closing operation, from the onset of door opening to the end of the subsequent closing phase. This cycle represents the natural operational unit of passenger door systems and constitutes the atomic sample used for anomaly detection. Each extracted cycle captures the mechanical, electrical, and logical behavior of the door during a single passenger access event. The five synchronized signals are acquired at a fixed sampling rate of 50 Hz and processed at the cycle level. Figure 1 illustrates the transformation from raw door-day recordings to cycle-level samples used as model input.

![](images/a277652a81fbd8e174a9f0aa3736eb9776e49e84fc18a26308a8e26888df99b4.jpg)  
Fig. 1. Cycle-level data representation. Left: segmentation of a day-long door recording into individual door operation cycles. Right: example of a temporally normalized cycle $( T \ = \ \bar { 2 5 6 } )$ showing continuous signals and logical states aligned with opening and closing phases.

a) Temporal resampling: Door operation cycles have variable durations due to real operating conditions. To enable batch training and consistent cycle-level comparison, each cycle is temporally resampled to a fixed length of $T = 2 5 6$ samples, while preserving the temporal ordering of the opening, dwell, and closing phases.

b) Amplitude scaling: Continuous signals, namely door position, motor current, and supply voltage, are independently normalized to a common numerical range to ensure balanced learning across features. Binary logical state signals, namely door-closed and door-locked, are kept in their original representation {0, 1}, where 0 indicates the door is open or unlocked, respectively.

After segmentation and normalization, each door operation cycle is represented as a multivariate sequence $X \in \overset { \overline { { \mathbb { R } } } ^ { 2 5 6 \times 5 } } { \mathbb { R } ^ { \left. 2 5 6 \times 5 \right. } }$ capturing the synchronized temporal evolution of three continuous physical variables and two binary logical states. Each cycle is also associated with metadata, including operating day, door identifier, and original duration. These metadata are used only for traceability and result analysis, and are not provided to the model during training.

## B. Dataset Statistics

Table I summarizes the main characteristics of the dataset after cycle segmentation and temporal normalization.

TABLE I  
MAIN CHARACTERISTICS OF THE RAILWAY DOOR OPERATION DATASET.
<table><tr><td rowspan=1 colspan=1>Days</td><td rowspan=1 colspan=1>Cars</td><td rowspan=1 colspan=1>Doors</td><td rowspan=1 colspan=1>Cycles</td><td rowspan=1 colspan=1>Features (F)</td><td rowspan=1 colspan=1>Cyclelength (T)</td><td rowspan=1 colspan=1>Freq.(Hz)</td></tr><tr><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>4960</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>256</td><td rowspan=1 colspan=1>50</td></tr></table>

## IV. METHODOLOGY

This section presents the proposed anomaly detection framework for railway passenger door monitoring. Following the cycle-based representation introduced in Section III, detection is performed at the cycle level: each complete door operation constitutes one monitoring instance and produces one anomaly score and one decision.

## A. Problem Setting

Each door operation cycle is represented as a fixed-length multivariate time series

$$
\boldsymbol { X } \in \mathbb { R } ^ { T \times F } ,\tag{1}
$$

where $T = 2 5 6$ is the normalized cycle length and $F = 5$ is the number of synchronized signals, including three continuous physical signals and two binary logical states.

Fault labels are generally unavailable in real operation, and abnormal events are rare. The model is therefore trained exclusively on nominal cycles. Given an unseen test cycle $X _ { j } ,$ the objective is to compute a scalar anomaly score $S ( X _ { j } )$ that measures its deviation from nominal behavior. This score is compared with a threshold τ calibrated on nominal validation data, and the cycle is declared anomalous when $S ( X _ { j } ) > \tau$

## B. Model Overview

Figure 2 provides a high-level overview of the proposed TCAA-CS framework. During training, each door operation cycle is processed by a dual-stream signal encoding stage, which separates continuous physical measurements from binary logical states. The resulting representation is then passed to an attention-based autoencoder, which compresses the cycle into a compact latent representation and reconstructs the original sequence. The model is trained on nominal cycles only by minimizing reconstruction error.

At inference time, TCAA-CS assigns one anomaly score to each complete door operation cycle. This score combines three complementary criteria: signal fidelity, measured by reconstruction error; latent-space deviation, which quantifies how far the cycle representation deviates from nominal latent behavior; and cross-signal consistency, which evaluates whether phase-wise inter-signal relationships remain coherent with nominal operation. These criteria are fused into a single triple hybrid anomaly score and compared against a calibrated threshold.

![](images/87c3c56ddb88b710d6f7f5572a58d06b802dafcb47e0336a94d82dd4b986abe6.jpg)  
Fig. 2. High-level overview of the TCAA-CS framework. Nominal cycles are used to train the reconstruction model, while inference combines signal fidelity, latent-space deviation, and cross-signal consistency into one anomaly decision per cycle.

Figure 3 presents the detailed TCAA-CS architecture. The upper part shows dual-stream encoding, temporal modeling with LSTM and attention pooling, and reconstruction through an LSTM decoder. The lower part shows the triple hybrid scoring mechanism derived from the input cycle X, the reconstructed cycle ${ \hat { X } } ,$ and the latent representation z.

## C. Dual-Stream Signal Encoding

The five input signals have different statistical properties. Motor position, motor current, and supply voltage are continuous measurements that describe mechanical and electrical dynamics, whereas door-closed and door-locked are binary logical states. Processing all signals through the same initial representation treats them as homogeneous channels and may obscure signal-type-specific patterns.

To preserve this distinction, TCAA-CS separates the input cycle into physical and logical streams, denoted by $X ^ { \mathrm { p h y s } } \in$ $\bar { \mathbb { R } ^ { T \times 3 } }$ and $\bar { X } ^ { \mathrm { l o g } } ~ \in ~ \mathbb { R } ^ { T \times 2 }$ . Each stream is processed by a dedicated temporal convolutional branch:

$$
H ^ { \mathrm { p h y s } } = \mathrm { C N N } _ { \mathrm { p h y s } } ( X ^ { \mathrm { p h y s } } ) , \qquad H ^ { \mathrm { l o g } } = \mathrm { C N N } _ { \mathrm { l o g } } ( X ^ { \mathrm { l o g } } ) ,
$$

where both outputs are in $\mathbb { R } ^ { T \times d _ { c } }$ and $d _ { c } = 3 2$ . The physical branch learns local position, current, and voltage patterns, while the logical branch learns state-transition patterns. The two encoded streams are concatenated as

$$
H ^ { \mathrm { i n } } = [ H ^ { \mathrm { p h y s } } \parallel H ^ { \mathrm { l o g } } ] \in \mathbb { R } ^ { T \times 2 d _ { c } } .\tag{2}
$$

![](images/7bf4998c65ecadca4bbff2a3b832c11ea36b5eee3c00ee3c9b69936bb6b829e1.jpg)  
Fig. 3. Detailed architecture of TCAA-CS for cycle-level railway door anomaly detection. The upper part shows dual-stream signal encoding, attention-based reconstruction, and latent representation learning. The lower part shows the triple hybrid anomaly scoring mechanism.

This produces a unified $T \times 6 4$ feature sequence while preserving the distinction between physical and logical behavior before joint temporal modeling.

## D. LSTM Encoder with Temporal Attention Pooling

The fused feature sequence $H ^ { \mathrm { i n } }$ is processed by a two-layer LSTM encoder with hidden dimension $d = 1 2 8 ,$ producing hidden states $( h _ { 1 } , \ldots , h _ { T } )$ , where each $h _ { t } \in \mathbb { R } ^ { d }$ represents the temporal context at timestep t.

A standard LSTM autoencoder often represents the full sequence using only the final hidden state h<sub>T</sub>. This can be limiting for railway door cycles because informative events may occur at different phases of the operation, such as motion onset, phase transitions, closing impact, or lock engagement. TCAA-CS instead constructs the cycle-level representation using additive temporal attention pooling. A relevance score is computed for each hidden state:

$$
e _ { t } = v ^ { \top } \operatorname { t a n h } ( W h _ { t } + b ) ,\tag{3}
$$

where W, v, and b are learnable parameters. The scores are normalized with a softmax function:

$$
\alpha _ { t } = \frac { \exp ( e _ { t } ) } { \sum _ { k = 1 } ^ { T } \exp ( e _ { k } ) } .\tag{4}
$$

The latent representation of the full cycle is then obtained as

$$
z = \sum _ { t = 1 } ^ { T } \alpha _ { t } h _ { t } \in \mathbb { R } ^ { d } .\tag{5}
$$

The attention weights $\alpha _ { t }$ allow the model to emphasize informative parts of the opening–dwell–closing operation while reducing the influence of less informative segments.

## E. Decoder and Training Objective

The latent representation z is passed to a symmetric twolayer LSTM decoder, which reconstructs the complete multivariate sequence:

$$
\hat { X } = \mathrm { D e c o d e r } ( z ) \in \mathbb { R } ^ { T \times F } .
$$

The model is trained using nominal cycles only by minimizing the mean squared reconstruction error:

$$
\mathcal { L } _ { \mathrm { t r a i n } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { M S E } ( X _ { i } , \hat { X } _ { i } ) ,\tag{6}
$$

where N is the number of nominal training cycles, $X _ { i }$ denotes the i-th nominal training cycle, and ${ \hat { X } } _ { i }$ is its reconstruction. The MSE is computed over all timesteps and signal dimensions. After training, deviations from learned nominal behavior are quantified through the triple hybrid anomaly score.

## F. Triple Hybrid Anomaly Scoring

At inference time, TCAA-CS evaluates each test cycle $X _ { j }$ using three complementary anomaly indicators. Reconstruction error captures signal-level deviations, latent-space

deviation captures global cycle-level abnormality, and crosssignal consistency captures violations of expected relationships between heterogeneous signals.

1) Reconstruction Error: The first component measures how accurately the model reconstructs the observed test cycle:

$$
\mathcal { L } _ { \mathrm { r e c } } ( X _ { j } ) = \mathrm { M S E } ( X _ { j } , \hat { X } _ { j } ) .\tag{7}
$$

This score captures deviations that alter the expected shape or amplitude of one or more signals, such as abnormal current peaks or distorted motion profiles.

2) Latent-Space Deviation (Cycle-Level Abnormality): The second component measures whether the latent representation of a test cycle is consistent with nominal cycle behavior. The nominal latent center is computed from the training cycles:

$$
\mu = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } z _ { i } .\tag{8}
$$

Here, $z _ { i }$ is the latent representation of the i-th nominal training cycle $X _ { i }$ . For a test cycle $X _ { j }$ with latent representation $z _ { j }$ , the latent-space deviation is defined as

$$
\mathcal { D } _ { \mathrm { l a t } } ( X _ { j } ) = \| z _ { j } - \mu \| _ { 2 } ^ { 2 } .\tag{9}
$$

This term captures cycles whose global temporal structure differs from nominal behavior, including unusual timing, abnormal phase durations, or atypical overall dynamics.

3) Phase-Aware Cross-Signal Consistency: The third component evaluates whether relationships between door signals remain consistent with nominal operation. During normal opening and closing, physical measurements and logical states evolve in coordinated ways. A cycle may therefore be abnormal even when individual signals appear locally plausible, if their inter-signal relationships become physically or logically inconsistent.

Each cycle is divided into opening and closing phases based on the position trajectory after temporal normalization. For each phase $k \in \{ \mathrm { o p e n } , \mathrm { c l o s e } \}$ , we compute a pairwise Pearson correlation matrix $R _ { k } ( X _ { j } ) ~ \in ~ \mathbb { R } ^ { F \times \bar { F } }$ , where each entry measures how two signals co-vary during that phase. For each phase, a nominal reference matrix is estimated from the nominal training cycles as

$$
\bar { R } _ { k } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } R _ { k } ( X _ { i } ) .\tag{10}
$$

The cross-signal consistency score is then defined as

$$
\mathcal { C } _ { \mathrm { c r o s s } } ( X _ { j } ) = \sum _ { k \in \{ \mathrm { o p e n } , \mathrm { c l o s e } \} } \left. R _ { k } ( X _ { j } ) - \bar { R } _ { k } \right. _ { F } ^ { 2 } .\tag{11}
$$

A high value indicates that the relationships between signals during opening or closing deviate from nominal behavior. This term is particularly useful for detecting physical or logical inconsistencies, such as the door-locked signal being active while the position signal still indicates motion.

4) Score Normalization and Fusion: The three anomaly components have different numerical scales. Before fusion, each component is standardized using statistics computed on nominal validation data:

$$
\tilde { Q } ( X _ { j } ) = \frac { Q ( X _ { j } ) - \mu _ { Q } } { \sigma _ { Q } + \epsilon } ,\tag{12}
$$

where $Q$ denotes any score component, $\mu _ { Q }$ and $\sigma _ { Q }$ are computed on the nominal validation set, and ϵ is a small constant for numerical stability.

The final anomaly score is defined as

$$
\ S _ { \mathrm { T C A A - C S } } ( X _ { j } ) = \lambda _ { 1 } \tilde { \mathcal { L } } _ { \mathrm { r e c } } ( X _ { j } ) + \lambda _ { 2 } \tilde { \mathcal { D } } _ { \mathrm { l a t } } ( X _ { j } ) + \lambda _ { 3 } \tilde { \mathcal { C } } _ { \mathrm { c r o s s } } ( X _ { j } ) ,\tag{13}
$$

where $\lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { 3 } \geq 0$ and $\lambda _ { 1 } + \lambda _ { 2 } + \lambda _ { 3 } = 1$ . The weights control the relative contribution of each anomaly component and are selected through the sensitivity analysis reported in Section V-D.

## G. Decision Rule

The decision threshold τ is calibrated exclusively on nominal validation data as a high percentile of the nominal validation score distribution:

$$
\tau = \mathrm { P e r c e n t i l e } _ { p } \left( \{ S _ { \mathrm { T C A A - C S } } ( X _ { m } ) \} _ { X _ { m } \in \mathcal { D } _ { \mathrm { v a l } } } \right) ,
$$

where $X _ { m }$ denotes a nominal validation cycle and $p$ controls the tolerated false-alarm level. A test cycle $X _ { j }$ is classified as

$$
\hat { y } _ { j } = \left\{ { \begin{array} { l l } { 1 , } & { \mathrm { i f ~ } S _ { \mathrm { T C A A - C S } } ( X _ { j } ) > \tau , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} } \right.\tag{14}
$$

where $\hat { y } _ { j } = 1$ denotes an anomalous cycle and $\hat { y } _ { j } = 0$ denotes a nominal cycle. This formulation produces one decision per complete door operation cycle and requires only nominal data for training and calibration.

## V. EXPERIMENTAL EVALUATION

This section evaluates the proposed TCAA-CS framework on real-world railway door operation data. All experiments follow a fixed and reproducible protocol. We first describe the evaluation setup, then report detection results against representative unsupervised baselines. We then present ablation studies to justify the main design choices, followed by systemlevel deployment experiments and qualitative analysis of real abnormal cycles.

## A. Experimental Protocol

1) Dataset Split: The dataset contains 4 960 door operation cycles collected over 30 operating days from 16 doors across 5 cars. To evaluate generalization across operating conditions, a day-wise split is adopted: all cycles from a given day are assigned to the same subset, and no day appears in more than one split. Table II reports the resulting split and the final testset composition after test-time anomaly injection.

Expert-identified abnormal cycles are excluded from the training and validation subsets and retained only for test evaluation. Consequently, the training and validation sets contain nominal cycles only. The validation set is used for

TABLE II  
DAY-WISE TRAIN/VALIDATION/TEST SPLIT AND FINAL TEST-SET COMPOSITION USED FOR EVALUATION.
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Days</td><td rowspan=1 colspan=1>Cycles</td><td rowspan=1 colspan=1>Nominal</td><td rowspan=1 colspan=1>Anomalous</td></tr><tr><td rowspan=1 colspan=1>Training</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>3179</td><td rowspan=1 colspan=1>3179</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>Validation</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>620</td><td rowspan=1 colspan=1>620</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>1161</td><td rowspan=1 colspan=1>969</td><td rowspan=1 colspan=1>192</td></tr></table>

early stopping, score normalization, and threshold calibration.   
Anomalous cycles are used exclusively in the test set.

2) Test-Time Anomaly Construction: Real abnormal door cycles are present but remain limited compared with nominal operation. Over the 30 operating days, 12 abnormal cycles were identified and confirmed through expert analysis. Since these events alone are insufficient for class-level quantitative evaluation, additional anomalies are introduced through controlled injection at test time only.

The injection mechanisms reproduce fault patterns derived from documented maintenance records and expert observations of real operational failures. Each anomaly class corresponds to a known failure mode encountered in railway door systems. The injection parameters and resulting signal profiles were validated by domain engineers to ensure physical and operational plausibility. Table III summarizes the injected anomaly classes.

TABLE III  
ABNORMAL CYCLE CLASSES USED FOR TEST-TIME EVALUATION.
<table><tr><td rowspan=1 colspan=1>Class</td><td rowspan=1 colspan=1>Type</td><td rowspan=1 colspan=1>Description</td></tr><tr><td rowspan=1 colspan=1>A</td><td rowspan=1 colspan=1>Energy /Load</td><td rowspan=1 colspan=1>Increased or biased motor current and/or voltageduring opening or closing, simulating friction.</td></tr><tr><td rowspan=1 colspan=1>B</td><td rowspan=1 colspan=1>Temporal/Transient</td><td rowspan=1 colspan=1>Local stretching or compression of motion phases,short spikes, dropouts, or sensor noise.</td></tr><tr><td rowspan=1 colspan=1>C</td><td rowspan=1 colspan=1>Logical</td><td rowspan=1 colspan=1>Inconsistent signals, e.g., door locked active whilethe door is moving.</td></tr></table>

From the nominal portion of the test set, 180 cycles are selected using a fixed random seed, with 60 cycles assigned to each injected anomaly class. These cycles are replaced by perturbed versions, each containing exactly one injected anomaly. Together with the 12 real expert-confirmed abnormal cycles, the final test set contains 192 anomalous cycles and 969 nominal cycles. Injected anomalies are never used during training, validation, early stopping, score normalization, or threshold calibration.

3) Evaluation Metrics: Performance is assessed using recall, precision, F1-score, and false-alarm rate (FAR). Recall is the primary metric because missed anomalies correspond to abnormal door operations that remain undetected. FAR is computed on nominal test cycles only and reflects the proportion of nominal cycles incorrectly flagged as anomalous, which directly affects maintenance workload.

4) Compared Methods: The proposed method is compared against unsupervised baselines representing complementary modeling paradigms:

• LSTM-AE: recurrent autoencoder using the final hidden state as cycle representation, with reconstruction error as anomaly score [7].

• GRU-AE: recurrent autoencoder using GRU units instead of LSTM units [11].

• TCN-AE: temporal convolutional autoencoder [12].

• OC-SVM: one-class SVM trained on handcrafted cyclelevel features [13].

• Anomaly Transformer: transformer-based anomaly detection model using association discrepancy [10]. For fair comparison, it is trained and evaluated on the same fixedlength cycle representation, and timestep-level anomaly scores are aggregated to produce one score per cycle.

All methods are trained on nominal data only, use comparable training budgets with early stopping, and produce one anomaly score per cycle. Thresholds are calibrated using the same percentile-based rule on nominal validation data.

## B. TCAA-CS Configuration

a) Architecture: The dual-stream encoder uses two 1D-CNN branches, each with two convolutional layers, kernel size 5, output dimension $d _ { c } \ = \ 3 2 .$ , ReLU activation, and batch normalization. The LSTM encoder and decoder each have two layers with hidden dimension d = 128. Dropout of 0.15 is applied between recurrent layers.

b) Training: The model is trained by minimizing reconstruction error using AdamW, with batch size 64 and a maximum of 120 epochs. Early stopping is applied based on nominal validation loss.

c) Scoring weights: The triple hybrid score weights are fixed to $\lambda _ { 1 } ~ = ~ 0 . 6 , ~ \lambda _ { 2 } ~ = ~ 0 . 1$ , and $\lambda _ { 3 } ~ = ~ 0 . 3$ based on the sensitivity analysis reported in Section V-D. The decision threshold is set at the $p \ : = \ : 9 5  – 1 \mathrm { h }$ percentile of the nominal validation score distribution. This percentile was selected to favor high recall while keeping the false-alarm rate low; it can be adjusted depending on the maintenance operator’s tolerance to false alarms.

## C. Main Detection Results

Table IV summarizes detection performance on the test set, which contains 192 anomalous cycles, including 12 real expert-confirmed anomalies and 180 injected anomalies, together with 969 nominal cycles.

TABLE IV  
CYCLE-LEVEL ANOMALY DETECTION RESULTS. FAR IS COMPUTED ON NOMINAL TEST CYCLES ONLY.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>TP</td><td rowspan=1 colspan=1>FN</td><td rowspan=1 colspan=1>FP</td><td rowspan=1 colspan=1>TN</td><td rowspan=1 colspan=1>Recall</td><td rowspan=1 colspan=1>Prec.</td><td rowspan=1 colspan=1>F1</td><td rowspan=1 colspan=1>FAR</td></tr><tr><td rowspan=1 colspan=1>LSTM-AE [7]</td><td rowspan=1 colspan=1>145</td><td rowspan=1 colspan=1>47</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>947</td><td rowspan=1 colspan=1>0.755</td><td rowspan=1 colspan=1>0.868</td><td rowspan=1 colspan=1>0.808</td><td rowspan=1 colspan=1>0.023</td></tr><tr><td rowspan=1 colspan=1>GRU-AE [11]</td><td rowspan=1 colspan=1>163</td><td rowspan=1 colspan=1>29</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>949</td><td rowspan=1 colspan=1>0.849</td><td rowspan=1 colspan=1>0.891</td><td rowspan=1 colspan=1>0.870</td><td rowspan=1 colspan=1>0.021</td></tr><tr><td rowspan=1 colspan=1>TCN-AE [12]</td><td rowspan=1 colspan=1>136</td><td rowspan=1 colspan=1>56</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>953</td><td rowspan=1 colspan=1>0.708</td><td rowspan=1 colspan=1>0.895</td><td rowspan=1 colspan=1>0.791</td><td rowspan=1 colspan=1>0.017</td></tr><tr><td rowspan=1 colspan=1>OC-SVM [13]</td><td rowspan=1 colspan=1>120</td><td rowspan=1 colspan=1>72</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>939</td><td rowspan=1 colspan=1>0.625</td><td rowspan=1 colspan=1>0.800</td><td rowspan=1 colspan=1>0.702</td><td rowspan=1 colspan=1>0.031</td></tr><tr><td rowspan=1 colspan=1>Transformer [10]</td><td rowspan=1 colspan=1>158</td><td rowspan=1 colspan=1>34</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>951</td><td rowspan=1 colspan=1>0.823</td><td rowspan=1 colspan=1>0.898</td><td rowspan=1 colspan=1>0.859</td><td rowspan=1 colspan=1>0.019</td></tr><tr><td rowspan=1 colspan=1>TCAA-CS</td><td rowspan=1 colspan=1>180</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>964</td><td rowspan=1 colspan=1>0.938</td><td rowspan=1 colspan=1>0.973</td><td rowspan=1 colspan=1>0.955</td><td rowspan=1 colspan=1>0.005</td></tr></table>

The baseline methods achieve competitive performance, confirming that standard unsupervised sequence models capture a substantial part of abnormal door-cycle behavior. GRU-AE provides the strongest baseline, with 84.9% recall and an F1-score of 0.870, followed by the Anomaly Transformer with 82.3% recall and an F1-score of 0.859. TCN-AE shows high precision but lower recall, suggesting a more conservative detection behavior, while OC-SVM remains less competitive because it relies on handcrafted cycle-level features rather than learned temporal representations.

TCAA-CS achieves the best overall performance, with 93.8% recall, 97.3% precision, an F1-score of 0.955, and a false-alarm rate of 0.5%. Compared with the strongest baseline, GRU-AE, TCAA-CS improves recall by 8.9 percentage points and F1-score by 8.5 percentage points under the same threshold calibration protocol. This improvement suggests that dual-stream signal encoding and triple hybrid scoring capture anomaly patterns that are not fully addressed by reconstruction-based recurrent baselines.

Out of 192 anomalous cycles, TCAA-CS correctly detects 180 and misses 12. Notably, all 12 expert-confirmed real operational anomalies are detected, whereas none of the baseline methods detects all of them. The 12 missed detections correspond to low-magnitude controlled perturbations with amplitudes close to nominal variability. The method produces 5 false alarms on 969 nominal cycles, mainly arising from rare but valid operational events such as passenger-triggered reopening commands or atypical but non-faulty cycle timing.

## D. Ablation Study

To justify the main design choices, we perform a systematic ablation study. Starting from the full TCAA-CS model, we individually remove key components and measure the impact on detection performance. When a scoring component is removed, the remaining score weights are renormalized to sum to one, and the decision threshold is recalibrated on nominal validation scores using the same percentile rule.

1) Component Ablation: Table V reports the effect of removing each key component from the full model.

TABLE V  
COMPONENT ABLATION. EACH ROW REMOVES ONE COMPONENT FROM THE FULL TCAA-CS MODEL; “W/O” DENOTES “WITHOUT”.
<table><tr><td rowspan=1 colspan=1>Configuration</td><td rowspan=1 colspan=1>Recall</td><td rowspan=1 colspan=1>Prec.</td><td rowspan=1 colspan=1>F1</td><td rowspan=1 colspan=1>FAR</td></tr><tr><td rowspan=1 colspan=1>Full TCAA-CS</td><td rowspan=1 colspan=1>0.938</td><td rowspan=1 colspan=1>0.973</td><td rowspan=1 colspan=1>0.955</td><td rowspan=1 colspan=1>0.005</td></tr><tr><td rowspan=1 colspan=1>w/o dual-stream</td><td rowspan=1 colspan=1>0.906</td><td rowspan=1 colspan=1>0.946</td><td rowspan=1 colspan=1>0.925</td><td rowspan=1 colspan=1>0.012</td></tr><tr><td rowspan=1 colspan=1>w/o attention $\overline { { ( h _ { T } \ \mathrm { o n l y } ) } }$ </td><td rowspan=1 colspan=1>0.880</td><td rowspan=1 colspan=1>0.939</td><td rowspan=1 colspan=1>0.909</td><td rowspan=1 colspan=1>0.014</td></tr><tr><td rowspan=1 colspan=1>w/o Ccross</td><td rowspan=1 colspan=1>0.891</td><td rowspan=1 colspan=1>0.950</td><td rowspan=1 colspan=1>0.920</td><td rowspan=1 colspan=1>0.011</td></tr><tr><td rowspan=1 colspan=1>w/o $\overline { { \mathcal { D } _ { \mathrm { l a t } } } }$ </td><td rowspan=1 colspan=1>0.917</td><td rowspan=1 colspan=1>0.941</td><td rowspan=1 colspan=1>0.929</td><td rowspan=1 colspan=1>0.014</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathcal { L } _ { \mathrm { { r e c } } } \mathrm { { o n l y } } } }$ </td><td rowspan=1 colspan=1>0.781</td><td rowspan=1 colspan=1>0.926</td><td rowspan=1 colspan=1>0.847</td><td rowspan=1 colspan=1>0.015</td></tr></table>

Removing any single component degrades performance, confirming that each contributes to overall detection. The largest drop occurs when temporal attention is removed, with recall decreasing from 93.8% to 88.0%. This indicates that the attention-based cycle representation is important for capturing informative phases of the door operation. Removing the cross-signal consistency term reduces recall to 89.1%, with the loss concentrated on logical anomalies where individual signals appear plausible but their relationships are inconsistent. Removing dual-stream encoding reduces recall to 90.6%, confirming that separate processing of physical and logical signals improves the learned representation. Using reconstruction error alone reduces recall to 78.1%, demonstrating the necessity of combining complementary anomaly indicators.

2) Score Weight Sensitivity: We evaluate the sensitivity of the triple hybrid score to the choice of weights using a grid with increments of 0.1 under the constraint $\lambda _ { 1 } + \lambda _ { 2 } + \lambda _ { 3 } = 1$ The selected configuration $\lambda _ { 1 } = 0 . 6 , \lambda _ { 2 } = 0 . 1$ , and $\lambda _ { 3 } = 0 . 3$ achieved the best F1-score, with recall 0.938, precision 0.973, F1-score 0.955, and FAR 0.005. Neighboring configurations remained competitive, such as (0.5, 0.2, 0.3) with F1-score 0.952 and (0.7, 0.1, 0.2) with F1-score 0.950, indicating limited sensitivity to small weight variations. In contrast, singlecomponent settings were less effective, with reconstruction only (1, 0, 0) reaching F1-score 0.847, latent deviation only (0, 1, 0) reaching 0.474, and cross-signal consistency only (0, 0, 1) reaching 0.456. These results support the use of a reconstruction-dominant but complementary hybrid score.

## E. System-Level Evaluation

Beyond detection accuracy, onboard deployment requires low inference latency, limited memory usage, and feasible energy consumption. We therefore evaluate the computational cost of the final TCAA-CS model during cycle-by-cycle inference. Each measurement includes the complete inference pipeline for one completed door cycle: forward pass, anomaly score computation, and decision rule. Latency is measured after a warm-up phase and averaged over repeated inference runs to reduce initialization effects.

Table VI reports the results on three execution environments: an Intel i7-8850H CPU, an NVIDIA Quadro P2000 desktop GPU, and an NVIDIA Jetson AGX Xavier embedded platform. In the table, mean lat. denotes the average inference latency per cycle, p95 lat. denotes the 95th percentile latency, max lat. denotes the maximum observed latency, and GPU mem. denotes peak GPU memory usage during inference.

TABLE VI  
SYSTEM-LEVEL PERFORMANCE OF TCAA-CS UNDER CYCLE-BY-CYCLE INFERENCE.
<table><tr><td rowspan=1 colspan=1>Platform</td><td rowspan=1 colspan=1>Meanlat. (ms)</td><td rowspan=1 colspan=1>p95lat.(ms)</td><td rowspan=1 colspan=1>Maxlat.(ms)</td><td rowspan=1 colspan=1>GPUmem.(MB)</td><td rowspan=1 colspan=1>Energy</td></tr><tr><td rowspan=1 colspan=1>Intel i7-8850H CPU</td><td rowspan=1 colspan=1>8.52</td><td rowspan=1 colspan=1>10.41</td><td rowspan=1 colspan=1>14.8</td><td rowspan=1 colspan=1>I</td><td rowspan=1 colspan=1>一</td></tr><tr><td rowspan=1 colspan=1>Quadro P2000 GPU</td><td rowspan=1 colspan=1>5.21</td><td rowspan=1 colspan=1>6.08</td><td rowspan=1 colspan=1>8.23</td><td rowspan=1 colspan=1>32.7</td><td rowspan=1 colspan=1>一</td></tr><tr><td rowspan=1 colspan=1>Jetson AGX Xavier</td><td rowspan=1 colspan=1>4.98</td><td rowspan=1 colspan=1>6.52</td><td rowspan=1 colspan=1>8.92</td><td rowspan=1 colspan=1>32.7</td><td rowspan=1 colspan=1>8-12 W</td></tr></table>

On the embedded target, Jetson AGX Xavier, TCAA-CS achieves a mean inference latency of 4.98 ms per cycle, with a p95 latency of 6.52 ms and maximum latency of 8.92 ms. Since a complete door operation typically lasts several seconds, this latency is negligible compared with the physical duration of the monitored process. The small gap between mean and p95 latency also indicates stable execution under cycle-by-cycle inference. Across GPU-based platforms, peak memory usage remains below 34 MB, which is consistent with the compact model size of approximately 560k trainable parameters and leaves margin for a shared onboard monitoring unit to supervise multiple doors or run additional onboard modules. Energy consumption on the Jetson platform is approximately 8–12 W under inference load. These results support the feasibility of real-time onboard deployment for cycle-level railway door monitoring.

## F. Qualitative Analysis on Real Anomalies

Figure 4 presents one nominal cycle and three anomalous cycles illustrating distinct fault mechanisms. Each column shows a complete door cycle with observed signals in blue and TCAA-CS reconstruction in red.

![](images/88cd4570c7808a9cf17a269376ba18332acc84d5fb3361380d0b0d946962ad3f.jpg)

![](images/04fbf39d5f5bcfca6800e0d8189e15b55f8841fb1bb22fd944cc6aa5bb0d23d8.jpg)

![](images/5b077b29242b6c969bbde3fa2ac27ef09c50b9e844b835b4ce81751c78b10200.jpg)  
Fig. 4. Representative door cycles: one nominal cycle and three anomalous cycles illustrating load-related, kinematic, and logical inconsistency fault mechanisms. Blue: observed signals. Red: TCAA-CS reconstruction. Red dashed boxes: anomaly regions.

a) Nominal cycle: The reconstruction closely matches the observed behavior across all signals, indicating that TCAA-CS captures normal cycle dynamics.

b) Load-related anomaly: The opening phase is abnormally slow, with stalled motion visible in the position signal. The reconstruction predicts a normal trajectory, producing a large mismatch also reflected in current and voltage signals.

c) Kinematic anomaly: The door reverses before reaching the fully open position. The reconstruction follows the expected trajectory, highlighting the missing motion.

d) Logical inconsistency: The door-closed and doorlocked signals activate while the position signal indicates ongoing motion. Each signal individually appears plausible, but their relationship is physically inconsistent. The reconstruction preserves nominal logical timing, exposing the inconsistency. This fault is primarily captured by $\mathcal { C } _ { \mathrm { c r o s s } }$ , illustrating the value of the cross-signal consistency term.

Overall, the qualitative analysis shows that the three scoring components complement each other, supporting a unified framework rather than separate handcrafted checks for each failure mode.

## VI. CONCLUSION

This paper presented TCAA-CS, a cycle-level unsupervised anomaly detection framework for railway passenger door monitoring. By combining dual-stream signal encoding, temporal attention pooling, and a triple hybrid anomaly score, the model detects both signal-level deviations and inter-signal inconsistencies across complete door operation cycles. Experiments on real industrial data show 93.8% recall, 97.3% precision, and a 0.5% false-alarm rate, with all expert-confirmed anomalies detected. System-level evaluation on an NVIDIA Jetson AGX Xavier confirms real-time onboard deployment feasibility with millisecond-level latency and limited memory usage. Future work will focus on validation across larger multi-fleet datasets, additional door types, and adaptive consistency models that account for varying operating conditions while preserving low computational cost.

## ACKNOWLEDGMENT

The authors would like to thank Alstom for providing access to the operational railway door data used in this study. The authors also thank Eddy Doba and Nordine Saim from Alstom for their guidance and support throughout this work.

## REFERENCES

[1] M. Shimizu, Y. Zhao, and N. P. Avdelidis, “A fault detection approach based on one-sided domain adaptation and generative adversarial networks for railway door systems,” Sensors, vol. 23, no. 24, p. 9688, 2023.

[2] H. Dassanayake, C. Roberts, C. Goodman, and A. Tobias, “Use of parameter estimation for the detection and diagnosis of faults on electric train door systems,” Proceedings ofthe Institution ofMechanical Engineers, Part O: Journal of Risk and Reliability, vol. 223, no. 4, pp. 271–278, 2009.

[3] L. Cauffriez, S. Grondel, P. Loslever, and C. Aubrun, “Bond graph modeling for fault detection and isolation of a train door mechatronic system,” Control Engineering Practice, vol. 49, pp. 212–224, 2016.

[4] S. Ham, S.-Y. Han, S. Kim, H. J. Park, K.-J. Park, and J.-H. Choi, “A comparative study of fault diagnosis for train door system: traditional versus deep learning approaches,” Sensors, vol. 19, no. 23, p. 5160, 2019.

[5] Y. Sun, Y. Cao, and L. Ma, “A fault diagnosis method for train plug doors via sound signals,” IEEE Intelligent Transportation Systems Magazine, vol. 13, no. 3, pp. 107–117, 2020.

[6] M. Shimizu, S. Perinpanayagam, and B. Namoano, “A real-time fault detection framework based on unsupervised deep learning for prognostics and health management of railway assets,” IEEE Access, vol. 10, pp. 96 442–96 458, 2022.

[7] P. Malhotra, A. Ramakrishnan, G. Anand, L. Vig, P. Agarwal, and G. Shroff, “Lstm-based encoder-decoder for multi-sensor anomaly detection,” arXiv preprint arXiv:1607.00148, 2016.

[8] J. Audibert, P. Michiardi, F. Guyard, S. Marti, and M. A. Zuluaga, “Usad: Unsupervised anomaly detection on multivariate time series,” in Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, 2020, pp. 3395–3404.

[9] B. Zong, Q. Song, M. R. Min, W. Cheng, C. Lumezanu, D. Cho, and H. Chen, “Deep autoencoding gaussian mixture model for unsupervised anomaly detection,” in International conference on learning representations, 2018.

[10] J. Xu, H. Wu, J. Wang, and M. Long, “Anomaly transformer: Time series anomaly detection with association discrepancy,” International Conference on Learning Representations, 2022.

[11] X. Gong, S. Liao, F. Hu, X. Hu, and C. Liu, “Autoencoder-based anomaly detection for time series data in complex systems,” in 2022 IEEE Asia Pacific Conference on Circuits and Systems (APCCAS). IEEE, 2022, pp. 428–433.

[12] M. Thill, W. Konen, H. Wang, and T. Back, “Temporal convolutional¨ autoencoder for unsupervised anomaly detection in time series,” Applied Soft Computing, vol. 112, p. 107751, 2021.

[13] B. Scholkopf, J. C. Platt, J. Shawe-Taylor, A. J. Smola, and R. C.¨ Williamson, “Estimating the support of a high-dimensional distribution,” Neural computation, vol. 13, no. 7, pp. 1443–1471, 2001.