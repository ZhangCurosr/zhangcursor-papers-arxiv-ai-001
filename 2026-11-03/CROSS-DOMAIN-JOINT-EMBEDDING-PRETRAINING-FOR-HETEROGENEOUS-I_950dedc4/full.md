# CROSS-DOMAIN JOINT-EMBEDDING PRETRAINING FOR HETEROGENEOUS INDUSTRIAL TIME SERIES

Jonathan Adam Rico<sup>⋆†</sup> Nagarajan Raghavan<sup>⋆</sup> Senthilnath Jayavelu<sup>†</sup>

<sup>⋆</sup>Engineering Product Development (EPD), SUTD, Singapore <sup>†</sup>Institute of Advanced Intelligence and Computing (IAIC), A\*STAR, Singapore

## ABSTRACT

Industrial predictive maintenance and manufacturing tasks frequently contain abundant sensor measurements, but with limited labeled data. We investigate cross-domain self-supervised pretraining using joint-embedding predictive architecture for heterogeneous industrial time series (JEPA-HITS), which learns one encoder from various domains such as battery, semiconductor chemical mechanical planarization (CMP), computer numerical control (CNC) machining, and bearing while preserving dataset-specific temporal scales. We evaluate representation reuse on regression and classification problems under reducedlabel downstream training, and leave-one-domain-out (LODO) pretraining. Compared with the same architecture initialized randomly, LODO pretraining reduces RMSE from 10.72 to 7.95 on battery remaining useful life (RUL), 4.34 to 3.89 on CMP, and 13.53 to 11.72 on tool wear regression, while improving bearing fault diagnosis macro-F1 from 0.547 to 0.737. Overall results suggest that unlabeled time series measurements from heterogeneous industrial processes can provide reusable representations for downstream learning with limited labeled data.

Index Terms— Self-supervised, Representation learning, Predictive maintenance, Multi-task, Multidomain

## 1. INTRODUCTION

Industrial sensor signals are abundant, but measurements with degradation and fault labels are scarce because they are expensive. Self-supervised learning (SSL) [1], [2] have shown that pretrained time series representations are efective even with limited labeled data. However, existing cross-domain models demonstrate transfer learning between closely related datasets. For example, [3] demonstrated transfer from FordA to FordB and [4] from PU bearings to

CWRU bearings, leaving transfer across physically distinct industrial domains comparatively underexplored.

Many existing approaches address this heterogeneity through fixed input-context or modality-specific preprocessing assumptions. MOMENT [2], for example, operates on a fixed sequence length of 512 while FISHER [5] relies on short-time Fourier transform (STFT) representation and targets signals with suficiently high sampling rates. FISHER [5] and UniFault [6] successfully demonstrate transfer across heterogeneous industrial time series data but their evaluation is primarily on rotating machineries and fault diagnosis.

Moreover, several large pretrained models [7], [8], [9] are limited to forecasting tasks. In contrast, many industrial monitoring problems require sequence-level regression tasks, such as RUL regression, average material removal rate (MRR) prediction, and tool wear prediction. The Joint-embedding predictive architecture (JEPA) [10] provides an appealing alternative by predicting missing information in the latent space rather than the signal space. Recent TS-JEPA [3] extends latent predictive learning to time series, while TP-JEPA [11] applies JEPA to industrial prognostics. These studies motivate JEPA for temporal and industrial representation learning, but do not address joint pretraining across heterogeneous industrial processes within a shared encoder.

In this paper, we investigate whether a single JEPA encoder can learn transferable representations across physically distinct industrial domains, sensor characteristics, sequence scales, and downstream objectives. Table 1 details the heterogeneity of the four industrial time series datasets. Our contributions are summarized as follows:

• We investigate joint self-supervised representation learning across physically distinct industrial domains whose sampling characteristics, sequence scales, channel dimensionalities, and downstream objectives difer substantially.

Table 1. Heterogeneity of industrial time series data in terms of task, sensor signals, total number of samples, median sequence length per sample, and sampling rate.
<table><tr><td>Domain</td><td>Dataset</td><td>Task</td><td>Signals</td><td>Samples</td><td>Median len</td><td>Sampling</td></tr><tr><td>Battery</td><td>NASA [12]</td><td>RUL reg.</td><td>voltage, current temperature</td><td>499</td><td>320</td><td>~0.1 Hz</td></tr><tr><td>Semiconductor</td><td>CMP [13]</td><td>MRR reg.</td><td>pressure, flowrate rotation rate, usage</td><td>2321</td><td>344</td><td>~1 Hz</td></tr><tr><td>Machining</td><td>CNC [14]</td><td>tool wear reg.</td><td>vibration (tri-axis), sound</td><td>225</td><td> $1 . 0 3 \times 1 0 ^ { 6 }$ </td><td>25.6 kHz</td></tr><tr><td>Rotating machinery</td><td>PU [15]</td><td>fault cls.</td><td>vibration</td><td>2239</td><td>256001</td><td>64 kHz</td></tr></table>

• We develop a heterogeneous JEPA formulation in which each domain retains its dataset-specific temporal representation while sharing a channelindependent latent encoder through multi-domain loss aggregation.

• We demonstrate transferable representation learning through label-scarcity evaluation, and leaveone-domain-out pretraining, in which the target industrial domain is completely unseen during SSL.

## 2. METHODOLOGY

## 2.1. Preliminaries

We consider self-supervised representation learning over a collection of D, heterogeneous industrial time series datasets, $\mathcal { D } = \{ { \mathcal { D } } _ { 1 } , \ldots , { \mathcal { D } } _ { D } \}$ , where the d-th dataset is

$$
\mathcal { D } _ { d } = \left\{ \left( \mathbf { X } _ { i } ^ { ( d ) } , y _ { i } ^ { ( d ) } \right) \right\} _ { i = 1 } ^ { N _ { d } }\tag{1}
$$

where, $\mathbf { X } _ { i } ^ { ( d ) } \in \mathbb { R } ^ { T _ { i } ^ { ( d ) } \times C _ { i } ^ { ( d ) } } , T _ { i } ^ { ( d ) }$ and $C _ { i } ^ { ( d ) }$ denote the sequence length and number of sensor channels, respectively, while $y _ { i } ^ { ( d ) }$ denotes a downstream target when labels are available.

The objective is to learn a single encoder that captures transferable temporal structure across these heterogeneous domains without using downstream labels during pretraining. To avoid requiring a fixed number or ordering of sensors, pretraining is performed in a channelindependent manner. Each channel of a multivariate sample is treated as an individual univariate sequence,

$$
\begin{array} { r } { \mathbf { x } _ { i , c } ^ { ( d ) } = \mathbf { X } _ { i , : , c } ^ { ( d ) } \in \mathbb { R } ^ { T _ { i } ^ { ( d ) } } , \qquad c = 1 , \dots , C _ { i } ^ { ( d ) } . } \end{array}\tag{2}
$$

The same encoder parameters θ are shared across all channels and domains.

## 2.2. Self-Supervised Architecture

We build upon the TS-JEPA [3] framework and extend it to heterogeneous industrial time series through channelindependent raw-signal pretraining, dataset-dependent temporal patching, joint multi-domain loss aggregation, latent anti-collapse (AC) and spectral regularization.

The JEPA-HITS architecture follows a JEPA [10] framework comprising an online encoder $f _ { \theta } ,$ an exponential moving average (EMA) target encoder $f _ { \bar { \theta } } ,$ and a predictor $g _ { \phi }$ . Given visible patch indices V and masked indices ${ \mathcal { M } } ,$ the online encoder extracts contextual representations while the target encoder provides stop-gradient latent targets $\mathbf { z } ^ { t }$ . Unlike waveform-reconstruction objectives, the predictor estimates the representations zˆ of M directly in latent space. The primary objective uses Huber loss $H _ { \delta }$ between layer-normalized LN predicted and target representations,

$$
\mathcal { L } _ { \mathrm { p r e d } } = \frac { 1 } { \left| \mathcal { M } \right| } \sum _ { m \in \mathcal { M } } H _ { \delta } \left( \mathrm { L N } ( \hat { \mathbf { z } } _ { m } ) - \mathrm { L N } ( \mathbf { z } _ { m } ^ { t } ) \right) .\tag{3}
$$

To prevent representation collapse, we use VICReginspired variance and covariance regularization [16] as anti-collapse loss term $\mathcal { L } _ { A C }$ . A spectral consistency term $\mathcal { L } _ { \mathrm { s p e c } } = \left\| \bar { \mathcal { S } } ( \hat { \mathbf { Z } } ) - \mathcal { S } ( \mathbf { Z } ^ { t } ) \right\| _ { 1 }$ additionally encourages agreement between the frequency structure of predicted and target latent trajectories, where the magnitude is logscaled $\begin{array} { r } { \mathcal { S } ( \mathbf { Z } ) = \log ( 1 + | \operatorname { r F F T } ( \mathbf { Z } ) | ) } \end{array}$ . The domain-specific objective is

$$
{ \mathcal { L } } _ { d } = { \mathcal { L } } _ { \mathrm { p r e d } , d } + { \mathcal { L } } _ { \mathrm { A C } , d } + \lambda _ { s p e c } { \mathcal { L } } _ { \mathrm { s p e c } , d }\tag{4}
$$

To jointly optimize heterogeneous datasets without forcing a common sequence length or channel dimensionality, we use domain-specific training buckets. Each source retains its dataset-specific temporal window, patch size, and mini-batch size, while channels are independently processed by the shared encoder. Domain losses are then aggregated, for all the domains $D = 4$ in our case, before a single optimizer update.

$$
\mathcal { L } _ { m u l t i } = \frac { 1 } { D } \sum _ { d = 1 } ^ { D } \mathcal { L } _ { d }\tag{5}
$$

This framework, motivated by [17], permits joint pretraining of datasets that difer in sequence characteristics, and prediction task without padding or resampling to force all domains to a common sequence length, channel dimensionality, or sampling rate.

For the downstream configuration, we use a linear head or multi-layer perceptron (MLP), frozen or lowrank adaptation (LoRA) [18] of rank four depending on validation results, yielding frozen-linear for NASA, LoRA-MLP for CNC, and frozen-MLP for both CMP and PU. Figure 1 shows the overall architecture of the JEPA-HITS framework.

![](images/50702d7809bd08d51de4dc0365e7b054328bd3070b2d1e5d27bab9bfaddf3bd2.jpg)  
Fig. 1. Overview of the JEPA-HITS architecture for heterogeneous industrial time series.

## 3. EXPERIMENTS

## 3.1. Experimental Setup

Datasets. We evaluate our proposed framework on four diferent industrial domains such as battery, semiconductor manufacturing, computer numerical control (CNC) machining, and rotating machinery represented by the following datasets: National Aeronautics and Space Administration (NASA) Prognostics Center of Excellence (PCoE) battery [12], Chemical Mechanical Planarization (CMP) PHM 2016 data challenge [13] for material removal rate (MRR) prediction, CNC tool wear PHM-Asia Pacific (PHM-AP) 2025 data challenge [14], and Paderborn University (PU) bearing [15]. To avoid data leakage, we perform transparent pretraining from scratch and train-test partition before preprocessing. For CMP and CNC datasets, we use the oficial train-test splits of the data challenge [13][14] with specific chamber IDs {4,5,6} for CMP, while for NASA and PU datasets we use the train-test splits in [19] and [20]. All selfsupervised pretraining and downstream model selection use training partitions only.

Implementation Details. Experiments were conducted on an NVIDIA RTX 3070 8-GB GPU computer. Experiments were run on three diferent seeds and the average values are reported in the results. We compare our model against fully-supervised models MiniRocket [21] and PatchTST [22], pretrained MOMENT-small [2] and FISHER-small [5], and baseline TS-JEPA [3], which was pretrained on each dataset depending on the target domain. Pretrained models are adapted using rank-4 LoRA and an MLP head. FISHER is not evaluated for NASA and CMP to avoid artificially resampling their low-frequency < 100 Hz signals to satisfy its STFT requirements. JEPA-HITS uses 2,000 joint multi-domain optimizer updates, learning rate $3 \times 1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 4 }$ , and initial EMA momentum 0.99. For NASA/CMP/CNC/PU, respectively, mask ratios are 0.85/0.70/0.50/0.50, window lengths are 256/1024/4096/4096, and $\lambda _ { \mathrm { s p e c } }$ values are 0.01/1.0/0.5/0.1. The encoder has three Transformer layers, four heads, and 192-dimensional embeddings; the predictor also uses 192 dimensions. Patch size is 32 except for PU, where it is 1024.

Performance Evaluation. For leave-one-domain-out (LODO) evaluation, the entire target domain is additionally excluded from self-supervised pretraining. The evaluation metrics are root mean squared error for regression and macro-F1, which is the average of F1 scores for all classes, for classification. The label eficiency plot uses coeficient of determination $R ^ { 2 }$ to visualize comparison with the baseline.

![](images/3a6e74bd9359271d0fc07295837a0134e4c1acf61c0b421cd102f4f44ac36473.jpg)  
Fig. 2. Label eficiency of JEPA-HITS on NASA, CMP, CNC, and PU compared against the same architecture with randomly initialized encoder.

Table 2. Performance comparison of JEPA-HITS against diferent models across datasets.
<table><tr><td>Model</td><td>NASA  $\mathrm { ( R M S E ) } \downarrow$ </td><td>CMP  $\left( \mathrm { R M S E } \right) \downarrow$ </td><td>CNC  $\left( \mathrm { R M S E } \right) \downarrow$ </td><td>PU  $( \mathrm { m a c r o - F 1 } ) \uparrow$ </td><td> $\#$  Params (M) ↓</td></tr><tr><td>MiniRocket [21]</td><td> $2 9 . 4 4 \pm 1 . 0 1$ </td><td> $4 . 0 2 \pm 0 . 0 5$ </td><td> $2 8 . 9 7 \pm 3 . 8 2$ </td><td> $0 . 4 9 6 2 \pm 0 . 0 1$ </td><td></td></tr><tr><td>PatchTST [22]</td><td> $3 4 . 8 4 \pm 4 . 3 7$ </td><td> $8 . 3 9 \pm 0 . 3 6$ </td><td> $1 2 . 8 7 \pm 2 . 5 1$ </td><td> $0 . 1 8 5 1 \pm 0 . 0 0$ </td><td>0.135</td></tr><tr><td>MOMENT-small [2]</td><td> $4 0 . 5 3 \pm 5 . 6 1$ </td><td> $5 . 3 4 \pm 0 . 0 3$ </td><td> $1 2 . 9 5 \pm 1 . 1 0$ </td><td> $0 . 5 8 2 4 \pm 0 . 0 4$ </td><td>37.9</td></tr><tr><td>FISHER-small [5]</td><td></td><td></td><td> $1 4 . 5 4 \pm 2 . 5 2$ </td><td> $0 . 4 8 6 0 \pm 0 . 0 4$ </td><td>21.4</td></tr><tr><td>TS-JEPA (adapted) [3]</td><td> $1 9 . 4 0 \pm 1 . 9 6$ </td><td> $4 . 0 9 \pm 0 . 0 7$ </td><td> $3 1 . 0 2 \pm 1 9 . 8 4$ </td><td> $0 . 6 1 4 7 \pm 0 . 0 3$ </td><td>0.363</td></tr><tr><td>Random-init JEPA-HITS</td><td> $1 0 . 7 2 \pm 1 . 0 1$ </td><td> $4 . 3 4 \pm 0 . 2 6$ </td><td> $1 3 . 5 3 \pm 0 . 6 1$ </td><td> $0 . 5 4 6 6 \pm 0 . 0 1$ </td><td>1.55</td></tr><tr><td> $\mathrm { J E P A - H I T S } _ { L O D O }$ </td><td> $7 . 9 5 \pm 0 . 0 0$ </td><td> $3 . 8 9 \pm 0 . 1 0$ </td><td> $1 1 . 7 2 \pm 0 . 7 4$ </td><td> $0 . 7 3 6 7 \pm 0 . 0 1$ </td><td>1.55</td></tr><tr><td>JEPA-HITS</td><td> $\mathbf { 6 . 6 5 \ : \pm { \ : 0 . 0 0 } }$ </td><td> $\mathbf { 3 . 8 7 \ : \pm { \ : 0 . 0 4 } }$ </td><td> ${ \bf 1 0 . 4 7 \pm 0 . 5 4 }$ </td><td> $\mathbf { 0 . 7 7 9 3 \ : \pm { \ : 0 . 0 4 } }$ </td><td>1.55</td></tr></table>

## 3.2. Results and Discussion

Table 2 shows that across all four downstream datasets, the pretrained encoder improves upon the same 1.55Mparameter architecture initialized randomly. The largest diferences occur on NASA and Paderborn, while CMP shows a smaller but consistent improvement. This comparison establishes that the downstream advantage cannot be explained solely by increasing the capacity of the TS-JEPA architecture. In LODO pretraining, the representation learned exclusively from the other three domains reaches 7.95, 3.89, and 11.72 RMSE on NASA, CMP, and CNC and 0.737 macro-F1 on Paderborn after downstream adaptation. This indicates that at least part of the pretraining benefit arises from features shared across industrial systems rather than unlabeled exposure to the target data itself. We additionally evaluated learning-progress-based adaptive domain weighting, however, it did not provide consistent improvements across domains.

## 3.3. Label Eficiency

Figure 2 shows that the efect of pretraining is not restricted to the fully supervised setting. The largest gains occur for NASA, CMP, and PU showing a persistent advantage with 63.70%, 33.79% average $R ^ { 2 }$ and 14.85% average macro-F1 improvements relative to baseline across the evaluated fractions. CNC is less consistent, with 1.40% average relative $R ^ { 2 }$ improvement, where pretraining does not dominate the random encoder at every intermediate fraction, indicating that the usefulness of the learned prior depends on the downstream process and supervision regime.

## 3.4. Ablation Study

Table 3 shows that the addition of both anti-collapse (AC) and spectral (spec) loss regularizers to JEPA-HITS yielded 38.31%, 4.69%, and 1.13% reductions in RMSE for NASA, CMP, and CNC respectively, while 2.75% improvement in macro-F1 on PU. The AC loss and spectral loss provide domain-dependent efects when applied individually, whereas their combination provides the strongest overall cross-domain configuration.

Table 3. Ablation study of the loss components.
<table><tr><td colspan="2">Model</td><td colspan="4">Dataset</td></tr><tr><td>AC</td><td>spec</td><td>NASA↓</td><td>CMP↓</td><td>CNC↓</td><td>PU↑</td></tr><tr><td rowspan="3"></td><td rowspan="3">√</td><td>10.78</td><td>4.05</td><td>10.59</td><td>0.7578</td></tr><tr><td>18.07</td><td>4.01</td><td>10.08</td><td>0.6819</td></tr><tr><td>7.35</td><td>3.92</td><td>10.55</td><td>0.6077</td></tr><tr><td>√</td><td> $\checkmark$ </td><td>6.65</td><td>3.86</td><td>10.47</td><td>0.7793</td></tr></table>

## 4. CONCLUSION

We studied whether unlabeled measurements from physically distinct industrial processes can improve downstream learning when labels are scarce. A shared JEPA representation pretrained across battery, semiconductor, machining, and bearing data consistently improves a same capacity random baseline under full supervision and remains useful when each target domain is removed from pretraining. Although the magnitude of the gain varies by domain and label fraction, the results provide evidence that heterogeneous industrial sensor collections contain transferable representation structure. This suggests a practical route toward reusable industrial pretrained encoders without requiring every downstream asset to contribute unlabeled pretraining data. Our work is not evaluated on extreme cases with signal outliers, missing data, and distribution shifts. Future work will extend JEPA-HITS to additional industrial domains and downstream tasks, including forecasting, leveraging its shared encoder design to scale across heterogeneous time series data and tasks.

## 5. REFERENCES

[1] G. Zerveas, S. Jayaraman, D. Patel, A. Bhamidipaty, and C. Eickhof, “A transformer-based framework for multivariate time series representation learning,” in In Proceedings of the 27th ACM SIGKDD conference on knowledge discovery and data mining, 2021, pp. 2114–2124.

[2] M. Goswami, K. Szafer, A. Choudhry, Y. Cai, S. Li, and A. Dubrawski, “Moment: A family of open timeseries foundation models,” in Proceedings of the 41st International Conference on Machine Learning, 2024, pp. 16115–16152.

[3] S. Ennadir, S. Golkar, and L. Sarra, “Joint embedding go temporal,” in NeurIPS Workshop on Time Series in the Age of Large Models, 2025.

[4] A. Zhou and A. Farimani, “Faultformer: Pretraining transformers for adaptable bearing fault classification,” in IEEE Access, 2024, vol. 12, pp. 70719– 70728.

[5] P. Fan et al., “Fisher: A foundation model for multimodal industrial signal comprehensive representation,” IEEE Transactions on Industrial Informatics, 2026.

[6] E. Eldele, Z. Chen, M. Wu, M. Ragab, and X. Li, UniFault: A Fault Diagnosis Foundation Model from Bearing Data, AI for Time Series, CRC Press, 2026.

[7] A. Das, W. Kong, R. Sen, and Y. Zhou, “A decoder-only foundation model for time-series forecasting,” International Conference on Machine Learning, 2024.

[8] K. Rasul et al., “Lag-llama: Towards foundation models for probabilistic time series forecasting,” in Conference on Neural Information Processing Systems, 2023.

[9] A.F. Ansari et al., “Chronos: Learning the language of time series,” Transactions on Machine Learning Research, 2024.

[10] M. Assran et al., “Self-supervised learning from images with a joint-embedding predictive architecture,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2023, pp. 15619–15629.

[11] B. Zhang, “Uncertainty-aware deep neural network for multi-scale temporal modeling in industrial equipment prognostics,” Advanced Manufacturing, 2026.

[12] B. Saha and K. Goebel, “Battery data set,” NASA Ames Prognostics Data Repository, 2007.

[13] Society of Prognostic and Health Management, “Phm data challenge,” 2016.

[14] Prognostics and Health Management Society, “Data challenge,” PHM Asia Pacific Conference, 2025.

[15] C. Lessmeier, J. Kimotho, D. Zimmer, and W. Sextro, “Condition monitoring of bearing damage in

electromechanical drive systems by using motor current signals of electric motors: A benchmark data set for data-driven classification,” in PHM Society European Conference, 2016, vol. 3.

[16] A. Bardes, J. Ponce, and Y. LeCun, “VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning,” in International Conference on Learning Representations, 2022.

[17] R. Schäfer et al., “Overcoming data scarcity in biomedical imaging with a foundational multi-task model,” Nat Comput Sci, vol. 4, pp. 495–509, 2024.

[18] E. Hu et al., “Lora: Low-rank adaptation of large language models,” in International Conference on Learning Representations, 2022.

[19] D. Yu and H. Wei, “Signal-quality-aware adaptive prediction for lithium-ion battery remaining useful life under noisy and resource-constrained conditions,” in Measurement, 2026.

[20] V. Coors and H. Simon, “Development of a fault diagnosis system for future (hybrid) electric aircraft with real-time capabilities,” Aerospace Systems, pp. 1–9, 2026.

[21] A. Dempster, D. Schmidt, and G. Webb, “Minirocket: A very fast (almost) deterministic transform for time series classification,” in In Proceedings of the 27th ACM SIGKDD conference on knowledge discovery and data mining, 2021, pp. 248–257.

[22] Y. Nie, N. Nguyen, P. Sinthong, and J. Kalagnanam, “A time series is worth 64 words: Long-term forecasting with transformers,” in International Conference on Learning Representations, 2023.