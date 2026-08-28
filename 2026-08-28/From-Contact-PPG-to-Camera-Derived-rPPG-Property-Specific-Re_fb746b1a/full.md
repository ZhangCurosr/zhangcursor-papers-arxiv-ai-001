# From Contact PPG to Camera-Derived rPPG: Property-Specific Recoverability, Dynamical Correspondence, and Fitzpatrick-Associated Heterogeneity

Timothy Oladunni<sup>1,∗</sup> Farouk Ganiyu Adewumi<sup>1</sup>

<sup>1</sup>Department of Computer Science, Morgan State University, Baltimore, Maryland, USA <sup>∗</sup>Corresponding author: timothy.oladunni@morgan.edu

## 1 Highlights

• A fixed camera-rPPG observation pathway reproduced the published MMPD CHROM correlation regime without post-hoc tuning, with modestly higher absolute error.

• Recoverability difered across endpoint, temporal, spectral, and nonlinear signal properties.

• No detectable recording-specific PPG-rPPG maximal-Lyapunov correspondence was found (rλ=0.0231; permutation p=0.5584).

• Observation context reduced subject-held-out PPG–rPPG HR MAE by up to 13.32%, with motion and lighting providing the predictive gain.

• Fitzpatrick-associated endpoint heterogeneity persisted after SNR adjustment, while the tested dynamical discrepancy showed a diferent heterogeneity pattern.

## 2 Abstract

Camera-derived remote photoplethysmography (rPPG) is commonly validated through endpoint accuracy, but endpoint performance alone does not establish whether other physiological properties of the source contact photoplethysmography (PPG) remain preserved recording by recording. We evaluated property-specific $\mathrm { P P G - t o - r P P G }$ recoverability on 655 recordings from the Multi-Domain Mobile Video Physiology Dataset (MMPD) using the established CHROM method as a fixed, reproducible camera-rPPG observation pathway. The pathway reproduced the published MMPD CHROM correlation regime (heart-rate MAE 15.26 bpm; Pearson $r = 0 . 0 8 0 1 )$ . Matched-versus-shufled validation revealed modest recording-specific autocorrelation correspondence, whereas spectral and recurrence-rate measures showed little matched discrimination; determinism was saturated and therefore non-discriminative under the present parameterization. Maximal Lyapunov exponents showed essentially no recording-specific PPG–rPPG correspondence $( r _ { \lambda } = 0 . 0 2 3 1$ ; condition-preserving permutation $p = 0 . 5 5 8 4 )$ ), despite population-level overlap. Endpoint discrepancy also exhibited Fitzpatrick-associated heterogeneity after adjustment for lighting and motion, including a Fitzpatrick VI versus III contrast of +9.32 bpm (95% CI 3.78–14.86), while the tested dynamical discrepancy showed no corresponding gradient. Adding aggregate RGB signal-to-noise ratio (SNR) did not

materially account for the endpoint contrast, and a simple Beer–Lambert attenuation model did not reproduce the measured SNR pattern. Finally, a subject-held-out condition-aware experiment showed that observation context provided incremental predictive information. Across linear, ridge, and random-forest learners, adding motion and lighting consistently reduced subject-held-out MAE relative to the corresponding rPPG-HR-only calibration baseline, with reductions of up to 13.32%; adding Fitzpatrick group or aggregate SNR provided no further improvement. These findings support the conclusion that, under the evaluated observation pathway, recoverability is not a single global property of camera-derived rPPG: physiological properties can difer in recording-specific preservation and in their dependence on observation conditions, and population-level plausibility does not establish preservation of individual recordings.

Keywords: remote photoplethysmography; contact PPG; signal recoverability; Lyapunov exponent; Fitzpatrick classification; signal-to-noise ratio

## 3 Introduction

Remote photoplethysmography (rPPG) estimates cardiovascular pulsatility from subtle camerarecorded variations in skin reflectance and enables contactless physiological sensing for telemonitoring, home monitoring, human–computer interaction, and related applications [2]. Contact photoplethysmography (PPG), by contrast, measures cardiac-synchronous changes in peripheral blood volume through direct optical interaction with tissue [1]. Camera-derived rPPG is therefore not a direct substitute for contact PPG: tissue optics, illumination, camera response, spatial sampling, compression, facial-region selection, and signal reconstruction transform the physiological observation before an endpoint is estimated [3, 2, 4].

Most rPPG validation consequently emphasizes endpoint performance, particularly heart-rate (HR) mean absolute error (MAE), root-mean-square error (RMSE), percentage error, and correlation [11, 14, 5]. These metrics are necessary but do not establish signal equivalence. A sensing transformation may preserve one physiological or signal property while degrading another. Moreover, similarity at the population level does not imply preservation of recording identity. For a property operator $\phi _ { j }$ , similarity between the marginal distributions of $\phi _ { j } ( X ^ { ( p ) } )$ and $\phi _ { j } ( X ^ { ( r ) } )$ does not establish that $\phi _ { j } ( x _ { i } ^ { ( p ) } )$ corresponds to $\phi _ { j } ( x _ { i } ^ { ( r ) } )$ for the same recording i. Thus, endpoint accuracy, signal equivalence, and property-specific recoverability are distinct questions.

This distinction motivates a property-specific view of PPG-to-rPPG validation. Under a camera observation pathway, some properties may remain recoverable while others may not. Conventional temporal, spectral, and recurrence structure provide complementary tests of waveform preservation, while nonlinear dynamics provide a more stringent test of whether recording-specific organization survives the observation transformation. State-space reconstruction provides a framework for characterizing dynamical structure from scalar observations [7]; within the reconstructed trajectory, the maximal Lyapunov exponent $\left( \lambda _ { \operatorname* { m a x } } \right)$ characterizes local exponential divergence and can be estimated from short time series using practical methods such as Rosenstein’s algorithm [8]. Prior rPPG work has examined dynamical reconstruction beyond endpoint HR [9], but estimating a dynamical quantity separately in PPG and rPPG does not by itself demonstrate preservation. Recording-specific correspondence must exceed what remains when the PPG–rPPG pairing is deliberately broken.

Recoverability may also depend on the observation conditions under which the camera signal is formed. Skin optical properties, illumination, motion, camera processing, and algorithmic assumptions can afect diferent components of rPPG [4, 5, 3]. The Multi-Domain Mobile Video Physiology Dataset (MMPD) was designed to benchmark rPPG across variation in Fitzpatrick classification, lighting, and motion/activity [6], making it suitable for asking whether observation-associated discrepancies are uniform across physiological properties. Fitzpatrick classification is treated here as a measured grouping variable rather than a direct measure of melanin or an identified causal optical mechanism [13]. Likewise, aggregate signal-to-noise ratio (SNR) is evaluated as one possible quality descriptor rather than assumed to explain all downstream discrepancy [15].

The unresolved problem is therefore not merely whether camera-derived rPPG produces plausible HR estimates, but which properties of synchronized contact PPG retain recording-specific correspondence after camera observation and reconstruction, whether loss varies across measured observation conditions, and whether any of that context is actionable on unseen subjects. Existing rPPG studies provide endpoint benchmarks, signal-quality analyses, nonlinear characterization, and systematic evaluation frameworks [2, 3, 5, 9, 11, 14]; however, these approaches do not by themselves establish matched preservation of multiple properties from the same synchronized contact-PPG recordings. We address this gap using a fixed CHROM-based camera-rPPG observation pathway, matched-versus-shufled correspondence controls, subject-aware heterogeneous analyses, and subjectheld-out prediction. The objective is to characterize property-specific and condition-dependent recoverability under the evaluated pathway, not to optimize a new rPPG estimator or claim general interchangeability between contact PPG and camera-derived rPPG.

Relation to Prior Representation Frameworks. The present study follows a progression from representation-level information to observation-level recoverability. Complementary Feature Domains (CFD) motivates the premise that physiological information can depend on the representation in which a signal is analyzed: distinct feature domains may retain diferent task-relevant information rather than serving as interchangeable descriptions of the same observation [16]. Cardiac Stability Theory (CST) extends this representation-level perspective to cardiovascular signals by considering temporal and dynamical organization in contact PPG as physiologically informative structure beyond a single conventional endpoint [17]. Both perspectives therefore motivate asking not only what information is represented in an observed physiological signal, but what happens to that information when the observation mechanism itself changes.

The present work addresses that observation-level question. Let $x _ { i } ^ { ( p ) }$ denote contact PPG from recording $i ,$ and represent the corresponding camera-derived signal as

$$
\begin{array} { r } { \boldsymbol { x } _ { i } ^ { ( r ) } = \mathcal { H } _ { \boldsymbol { \theta } _ { i } } \Big ( \boldsymbol { x } _ { i } ^ { ( p ) } \Big ) , } \end{array}
$$

where $\mathcal { H } _ { \theta _ { i } }$ denotes the camera observation and reconstruction pathway under recording-specific conditions $\theta _ { i }$ . For a physiological property operator $\phi _ { j }$ , the relevant question is whether the property extracted before observation, $\phi _ { j } ( x _ { i } ^ { ( p ) } )$ , remains associated with the corresponding property after observation, $\phi _ { j } ( x _ { i } ^ { ( r ) } )$ . Importantly, similarity of the population-level distributions,

$$
P \Big [ \phi _ { j } ( { \cal X } ^ { ( p ) } ) \Big ] \approx P \Big [ \phi _ { j } ( { \cal X } ^ { ( r ) } ) \Big ] ,
$$

does not establish recording-specific preservation $\phi _ { j } ( x _ { i } ^ { ( p ) } )  \phi _ { j } ( x _ { i } ^ { ( r ) } )$ . This motivates the propertyspecific recoverability framework evaluated here: endpoint, conventional structural, and nonlinear dynamical properties are assessed separately across the contact-PPG–to–rPPG observation boundary.

The relationship to CFD and CST is therefore conceptual rather than assumptive. The present analysis does not require either framework to hold and does not constitute a direct test of either theory.

Instead, it examines the observation-level question that follows naturally from their representationcentered perspective: when a physiologically informative signal representation is subjected to a diferent sensing and reconstruction process, which of its properties remain recoverable for the same recording?

Prior work investigated topological signal representations as a framework for characterizing and mitigating skin-tone-associated efects in optical physiological measurements [18]. The present study addresses a distinct question: whether specific physiological and dynamical properties of a contact-PPG source remain recoverable after transformation through a camera-rPPG observation pathway. Figure 1 summarizes the study’s property-specific view of PPG-to-rPPG recoverability and distinguishes the source properties, camera observation pathway, and corresponding recovered properties.

![](images/7178339055abbe31da2f23f8a012c6a1cf3e940fba60e64c2f741830d55e168b.jpg)  
Figure 1: Conceptual Framework for Property-Specific PPG-to-rPPG Recoverability. Contact PPG is transformed through a heterogeneous camera observation pathway, and recoverability is evaluated separately for physiological properties. Population-level plausibility does not by itself establish recording-specific correspondence.

## 3.1 Research questions and empirical mapping

To make the study logic explicit, Table 1 maps each research question to its empirical test and the result that answers it. The table is an organizational synthesis of the analyses reported below and introduces no new analysis.

Table 1 | Research Questions, Empirical Tests, and Principal Findings  
Each row maps a research question to the mathematical or statistical analysis used to evaluate it and the principal empirical conclusion. Detailed efect estimates and uncertainty measures are reported in the corresponding Results tables.
<table><tr><td>RQ</td><td>Research question</td><td>Empirical test</td><td>Empirical answer</td></tr><tr><td>RQ1</td><td>How accurately is heart Endpoint rate recovered from con- tact PPG after transfor- summarized by mation to camera-derived RMSE, MAPE, signed dicating substantial endpoint error and rPPG under the fixed ob- error, and Pearson correla- weak recording-level association (Ta- servation pathway? Which</td><td> $D _ { H R } \ = \ | H R ^ { ( p ) } \ - \ H R ^ { ( r ) } |$  tion. conventional Property-specific matched- Selective recording-specific struc-</td><td>discrepancy Limited HR recovery under the evaluated observation pathway. MAE, MAE=15.26 bpm and  $r = 0 . 0 8 0 1$  , in- ble 2).</td></tr><tr><td></td><td>temporal, spectral, and versus-shuffled recurrence</td><td>assessed for discriminative parameterization (Table 3). variability. Is maximal Lyapunov ex- Study-specific  $\hat { \lambda } _ { \mathrm { m a x } }$ </td><td>struc- tural correspondence. ACF showed properties tural correspondence for a modest matched advantage, whereas retain recording-specific ACF similarity, cardiac- PSD and recurrence-rate differences correspondence between band PSD distance, and provided little matched-versus-shuffled matched PPG and rPPG? recurrence-rate difference separation. Determinism was non- (Algorithm 1); determinism discriminative under the implemented</td></tr><tr><td></td><td>ponent  $\left( \lambda _ { \operatorname* { m a x } } \right)$  a recording-specific dynami- discrepancy (Algorithm 2), population-level dynamical plausibility, cal property across matched followed by recording- matched PPG-rPPG recordings?</td><td>preserved as timationandpaired dynamical correspondence. Despite specific correspondence sentially uncorrelated and a 10,000-permutation and the observed association was not condition-preserving subject-permutation null preserving subject-permutation null test (Algorithm 1). Which measured observa- Subject-aware</td><td>es- No detectable recording-specific  $\mathrm { P P G - r P P G }$  estimates were es-  $( r _ { \lambda } ~ = ~ 0 . 0 2 3 1 )$  distinguishable from the condition- (Table 4). mixed- Observation-associated discrep- and ancy was property-specific.</td></tr><tr><td>RQ4</td><td>tion factors are associated effects models for with endpoint and dynam- ical discrepancies, and are lighting, motion/activity, motion/activityand Fitzpatrick the heterogeneity patterns interaction block tests, group, including evidence of a property-specific?</td><td> $D _ { H R }$   $D _ { \lambda }$  , with Fitzpatrick group, and aggregateRGB- Fitzpatrick×motion SNR sensitivity analysis whereas (Algorithm 3).</td><td> $D _ { H R }$  variedsubstantiallywith interaction,  $D _ { \lambda }$  showed a different hetero- geneity pattern with little Fitzpatrick- or lighting-associated variation. Ag- gregate RGB SNR did not materially account for the endpoint heterogeneity (Tables 6–11).</td></tr><tr><td>RQ5</td><td>sured observation context information subjects?</td><td>∆MAE relative to the each learner&#x27;s model (Algorithm 4).</td><td>To what extent can mea- Subject-disjoint nested- Partially. Motion and lighting reduced prediction subject-held-out MAE beyond rPPG- reduce the PPG-rPPG end- with linear, ridge, and HR-only calibration across linear, ridge, point discrepancy, as mea- random-forest learners; and random-forest learners. At M2, sured by MAE, on unseen out-of-fold MAE and MAE was reduced by 13.32%, 13.32%, subject-level bootstrap and 12.77%, respectively, relative to  $M _ { 0 }$  baseline. Adding Fitz- 1 rPPG-HR-only calibration patrick group or aggregate RGB SNR provided no further improvement, and residual error remained substantial (Fig- ure 8).</td></tr></table>

## 3.2 Contributions

This study makes four principal contributions.

1. Property-specific formulation of physiological recoverability. We formulate PPG-torPPG validation as a property-specific problem rather than a single modality-level fidelity judgment, separating endpoint, conventional structural, and nonlinear dynamical properties.

2. Recording-specific correspondence as a criterion distinct from marginal plausibility. For structural and dynamical properties, matched $\mathrm { P P G - r P P G }$ comparisons are evaluated against correspondence-breaking shufled controls so that population-level similarity is not interpreted as preservation of individual recordings.

3. Subject-aware characterization of property-specific heterogeneity. We quantify endpoint and dynamical discrepancies across Fitzpatrick group, lighting, and motion/activity using repeated-measures models, interaction tests, and an aggregate RGB-SNR sensitivity analysis to determine whether heterogeneous observation conditions afect diferent properties in the same way.

4. Subject-held-out test of whether observation context is actionable. Using nested information sets and simple linear, ridge, and random-forest learners, we test whether measured observation context reduces the PPG–rPPG HR gap on unseen subjects beyond an rPPG-HRonly calibration baseline.

Together, these contributions provide a unified test of what survives the PPG-to-rPPG observation pathway, whether preservation is recording-specific, where discrepancies vary, and whether measured observation context can partially narrow the endpoint gap.

## 4 Methods

## 4.1 Study design and analysis sequence

This was a retrospective, recording-matched analysis of synchronized contact PPG and facial video from the Multi-Domain Mobile Video Physiology Dataset (MMPD). The analysis followed a fixed hierarchy rather than an algorithm-selection exercise. First, we established a fixed camera-rPPG observation pathway and verified its benchmark behavior. Second, we quantified endpoint heart-rate discrepancy. Third, we tested conventional temporal and spectral structure against shufled-pair negative controls. Fourth, we applied the same study-specific maximal-Lyapunov estimator to PPG and rPPG and tested recording-specific correspondence. Fifth, we quantified heterogeneity of endpoint and dynamical discrepancies across Fitzpatrick group, lighting, and motion using subject-aware mixed-efects models. Finally, we evaluated measured RGB SNR against a simple optical-attenuation model, examined SNR-adjusted endpoint heterogeneity, and tested whether measured observation context reduced held-out endpoint error. No Fitzpatrick-specific tuning, alternative rPPG-method search, or post-hoc selection of SNR definitions was performed.

The property-specific quantities, recording-correspondence criterion, and heterogeneous-observation model used throughout the analysis are formalized in Section 4.2.

The unit of analysis was the recording, while repeated recordings from the same participant were handled through subject-level random intercepts in inferential models. The expected MMPD design comprised 33 subjects with 20 recordings each (660 expected recordings). Five source recordings were absent from the available local dataset, yielding 655 available recordings for the fixed-pathway benchmark. All exclusions and estimator failures in downstream dynamical analysis followed the stated validity rules and were not determined by observed cross-modal agreement.

## 4.2 Mathematical formulation of property-specific recoverability

The analysis is organized around a latent physiological process observed through two sensing pathways. Let $s _ { i } ( t )$ denote the underlying cardiovascular process for recording i. The synchronized contact PPG and camera pathway are represented as

$$
x _ { i } ^ { ( p ) } = \mathcal { P } [ s _ { i } ] , \qquad v _ { i } = \mathcal { V } _ { \boldsymbol { \theta } _ { i } } [ s _ { i } ] , \qquad x _ { i } ^ { ( r ) } = \mathcal { O } _ { \boldsymbol { \phi } } ( v _ { i } ) ,\tag{1}
$$

where $x _ { i } ^ { ( p ) }$ is the reference contact-PPG waveform, $v _ { i }$ is the facial video, $\nu _ { \theta _ { i } }$ denotes the camera observation process under measured condition vector $\theta _ { i } .$ , and $\mathcal { O } _ { \phi }$ is a fixed camera-rPPG observation pathway that produces $x _ { i } ^ { ( r ) }$ . In this experiment, $\mathcal { O } _ { \phi } = \mathcal { O } _ { \mathrm { C H R O M } }$ using the established CHROM method [10]. In the present dataset, $\theta _ { i }$ includes Fitzpatrick group, lighting, and motion/activity descriptors. Equation 1 is an organizational observation model, not an identified causal optica model and not an assumption that an exact inverse from rPPG to PPG exists.

Let $\phi _ { j }$ denote a physiological or signal property indexed by j. The property values extracted from the two observations are

$$
z _ { i j } ^ { ( p ) } = \phi _ { j } \left( x _ { i } ^ { ( p ) } \right) , \qquad z _ { i j } ^ { ( r ) } = \phi _ { j } \left( x _ { i } ^ { ( r ) } \right) , \qquad j \in \mathcal { I } ,\tag{2}
$$

with the evaluated property family

$$
\mathcal { I } = \left\{ H R , A C F , P S D , R Q A , \lambda _ { \operatorname* { m a x } } \right\} .\tag{3}
$$

The notation is deliberately property-specific: the study does not assume that successful recovery of one member of $\mathcal { I }$ implies recovery of another.

For properties naturally compared by a distance or absolute diference, recording-level recoverability loss is written generically as

$$
D _ { i j } = d _ { j } \Big ( z _ { i j } ^ { ( p ) } , z _ { i j } ^ { ( r ) } \Big ) .\tag{4}
$$

The two principal scalar discrepancies are therefore

$$
D _ { i , H R } = \left| H R _ { \mathrm { r P P G } , i } - H R _ { \mathrm { P P G } , i } \right| , \qquad D _ { i , \lambda } = \left| \lambda _ { \mathrm { m a x } , i } ^ { ( r ) } - \lambda _ { \mathrm { m a x } , i } ^ { ( p ) } \right| .\tag{5}
$$

For ACF, PSD, and recurrence analyses, $d _ { j }$ is replaced by the corresponding similarity or distance statistic defined below.

A second requirement is recording specificity. Similar marginal distributions alone do not establish that the property belonging to recording i survives the observation pathway. Let $\pi ( i )$ denote a correspondence-breaking permutation that maps a source PPG recording to a nonmatched rPPG recording while preserving the specified condition structure. For a similarity statistic $q _ { j }$ , evidence of recording-specific preservation is supported when

$$
\begin{array} { r } { \mathbb { E } \left[ q _ { j } \left( z _ { i j } ^ { ( p ) } , z _ { i j } ^ { ( r ) } \right) \right] > \mathbb { E } \left[ q _ { j } \left( z _ { i j } ^ { ( p ) } , z _ { \pi ( i ) j } ^ { ( r ) } \right) \right] , } \end{array}\tag{6}
$$

with the direction reversed for a distance statistic. For $\lambda _ { \mathrm { m a x } }$ , the analogous test is whether the matched recording correlation exceeds its condition-preserving subject-permutation null. Equation 6 formalizes why the manuscript distinguishes population-level similarity from recording-specific recoverability.

Finally, heterogeneity is modeled as variation in the property-specific discrepancy conditional on measured observation factors. For property j, the subject-aware model is

$$
D _ { i j } = \beta _ { 0 j } + \beta _ { F , j } ^ { \top } F _ { i } + \beta _ { L , j } ^ { \top } L _ { i } + \beta _ { M , j } ^ { \top } M _ { i } + u _ { s ( i ) , j } + \varepsilon _ { i j } ,\tag{7}
$$

where $F _ { i } , L _ { i }$ , and $M _ { i }$ encode Fitzpatrick group, lighting, and motion/activity; $u _ { s ( i ) , j }$ is a subjectspecific random intercept; and $\varepsilon _ { i j }$ is the recording-level residual. The fitted models in this study instantiate Equation 7 separately for $D _ { H R }$ and $D _ { \lambda }$ . Thus, Fitzpatrick group, lighting, and motion/activity enter as parallel explanatory factors rather than as definitions of recoverability.

This formulation separates three questions that are otherwise easily conflated: whether a property can be estimated from $\mathrm { r P P G }$ , whether its recording identity is preserved relative to the synchronized PPG, and whether its loss changes across observation conditions. Algorithms 1–3 operationalize the property-specific characterization framework, while Algorithm 4 separately tests whether measured observation context can translate the identified heterogeneity into improved held-out endpoint prediction.

## 4.3 Dataset, recording structure, and metadata

Each available MMPD record contained facial RGB video and synchronized ground-truth PPG. Video consisted of 1,800 frames at 30 frames/s, corresponding to approximately 60 s per recording. The analysis retained the dataset's heterogeneous acquisition labels, including Fitzpatrick group (III-VI), lighting condition, and motion/activity condition. These variables were treated as observed acquisition descriptors rather than manipulated causal exposures.

Raw inspection identified inconsistent Fitzpatrick metadata for Subject 24: one recording was labeled Fitzpatrick IV and the remaining recordings were labeled Fitzpatrick III. To avoid assigning a post-hoc skin-type label, Subject 24 was excluded only from Fitzpatrick-dependent analyses and retained for analyses not requiring skin-type classification. After removal of one duplicated matched key, the heterogeneous-analysis table contained 645 unique matched records; the Fitzpatrick analysis population contained 625 recordings from 32 subjects.

## 4.4 Fixed camera-to-rPPG observation operator

Camera-derived rPPG was generated with a fixed observation operator, $\mathcal { O } _ { \phi }$ , instantiated in this experiment by the established chrominance-based CHROM method [10]. CHROM is not a methodological contribution of this study and was not optimized against the downstream recoverability results. It was selected as a reproducible, unsupervised camera-to-rPPG operator and held fixed across structural, dynamical, heterogeneous, and condition-aware analyses.

For reproducibility, facial ROIs were fixed within each recording, resized to $7 2 \times 7 2$ pixels, and reduced to frame-wise RGB spatial means. The standard CHROM construction was applied in 1.6-s windows with 50% overlap, channel normalization, chrominance projection, bandpass filtering, adaptive scaling, and overlap-add reconstruction [10]. Benchmark HR estimation used the same detrending, 0.6–3.3-Hz bandpass, and frequency-domain peak-estimation pathway for all recordings.

Benchmark results are reported in Section 6.1. Thereafter, the scientific analyses treat $x ^ { ( r ) } = \mathcal { O } _ { \phi } ( v )$ as the camera-derived observation; conclusions concern recoverability under this fixed operator rather than CHROM development itself.

## 4.5 Endpoint heart-rate recoverability

Endpoint and structural recoverability were evaluated according to Algorithm 1.

Reference heart rate, HR<sub>PPG</sub>, was derived from the synchronized contact PPG, whereas cameraderived heart rate, $H R _ { \mathrm { r P P G } }$ , was estimated from the fixed rPPG waveform using the benchmark evaluation procedure. For each recording, signed endpoint error was $H R _ { \mathrm { r P P G } } - H R _ { \mathrm { P P G } }$ and absolute endpoint discrepancy was $D _ { H R } = \lvert H R _ { \mathrm { r P P G } } - H R _ { \mathrm { P P G } } \rvert$ . In the reproducibility code, these quantities are stored as HR CHROM and HR GT, respectively. The primary heterogeneous-observation outcome was DHR because it measures the magnitude of endpoint discrepancy without cancellation of positive and negative diferences. No recording was removed because its HR discrepancy was large.

Overall endpoint performance was summarized using MAE, RMSE, MAPE, signed error, and Pearson correlation. These metrics were used as endpoint-level evidence only; they were not interpreted as measures of waveform or nonlinear-structure preservation.

## 4.6 Conventional temporal and spectral structure

To test whether weak endpoint agreement implied complete loss of lower-level structure, we compared matched PPG-rPPG recordings using temporal and spectral measures. Autocorrelation-function (ACF) similarity quantified similarity of temporal repetition structure. Cardiac-band power-spectraldensity (PSD) distance quantified diferences in normalized spectral organization over the prespecified physiological band. Recurrence-rate discrepancy was evaluated as an additional state-space summary.

Each structural metric was subjected to a matched-versus-shufled negative control. The matched analysis used the synchronized PPG and rPPG from the same recording. The shufled analysis deliberately broke recording correspondence while preserving the collection of signals. A structural measure was considered evidence of recording specificity only when matched pairs were more similar than shufled pairs in the expected direction. This control was essential because physiological recordings can share generic cardiac-band structure even when they do not preserve recording identity.

An initially evaluated determinism statistic was identically 1.0 across the analyzed signals. Because a constant statistic cannot discriminate matched from mismatched recordings, it was classified as a metric failure and excluded from evidentiary claims rather than interpreted as perfect dynamical preservation.

## 4.7 Study-specific maximal-Lyapunov estimator

The estimator, validity rules, and recording-level matching procedure are specified in Algorithm 2.

Maximal Lyapunov exponent (λmax) was used as the nonlinear property for the PPG-to-rPPG correspondence test. The same estimator and validity criteria were applied to reference PPG and fixed camera-derived rPPG; estimator settings were not adapted according to the observed cross-modal agreement.

Signals entered a common analysis pathway and were resampled to 100 Hz so that delay, Theiler exclusion, and divergence time were expressed on the same time base for PPG and rPPG. A 20-s analysis segment was used. The delay tau was estimated from average mutual information with a maximum search of 50 samples. State-space reconstruction used embedding dimension m=5. Temporally adjacent points were excluded from nearest-neighbor selection using a 100-ms Theiler window. Lambda max was then estimated from the approximately linear region of the mean log-divergence curve using a Rosenstein-style procedure.

A recording could fail because of the stated signal-quality gate, failure to estimate tau, insuficient embedding trajectory, failure of the divergence fit, or a non-finite/non-positive exponent. Failure status was recorded rather than replaced or imputed. The primary cross-observation population was the exact intersection of recordings with valid PPG and valid rPPG λmax estimates. Consequently, the correlation analysis did not use modality-specific imputation or selective deletion based on agreement.

## 4.8 Recording-specific Lyapunov correspondence and permutation test

Recording-specific dynamical recoverability was quantified by Pearson correlation between λmax estimated from the reference PPG and λmax estimated from the matched rPPG. The primary efect was rλ, accompanied by a 95% confidence interval and conventional two-sided significance test.

Because both signals can share condition-dependent population structure without preserving recording identity, a subject-level condition-preserving permutation test was used as the principal negative control. Across 10,000 permutations, subject correspondence between PPG and rPPG was broken while acquisition/recording condition structure was preserved. The empirical two-sided probability was calculated with the finite-sample correction p perm = (1 + number of $\mathrm { | r \mathrm { - p e r m } | \mathrm { > } } \mathrm { = } \mathrm { | r \mathrm { - o b s } | ) / ( B }$ + 1), where $_ \mathrm { B = 1 0 , 0 0 0 }$ . The prespecified interpretation required the matched rλ to exceed what could be obtained after breaking subject correspondence; similarity of marginal λmax distributions alone was not considered evidence of preservation.

## 4.9 Construction of heterogeneous-recoverability outcomes

For each deduplicated matched recording, two absolute discrepancies were defined: $D _ { H R } \ =$ $| H R _ { \mathrm { r P P G } } - H R _ { \mathrm { P P G } } |$ and $\mathrm { D } \lambda = | \lambda \mathrm { m a x } , \mathrm { P P G } - \lambda \mathrm { m a x } , \mathrm { r P P G } |$ . DHR measures endpoint loss, whereas Dλ measures numerical disagreement in the selected nonlinear property. Dλ was not treated as evidence of preserved dynamics unless recording-specific correspondence was independently demonstrated by the rλ analysis.

Descriptive summaries were computed by Fitzpatrick group before adjustment. Subject 24 was excluded from all models containing Fitzpatrick because of inconsistent metadata, leaving 625 recordings from 32 subjects.

## 4.10 Mixed-efects models for Fitzpatrick, lighting, and motion

Two primary mixed-efects models were fitted as specified in Algorithm 3. Model 1 used DHR as the dependent variable; Model 2 used Dλ. Both models included Fitzpatrick group, lighting condition, and motion/activity condition as fixed efects and a random intercept for subject to account for repeated recordings. Fitzpatrick III, incandescent lighting, and rotation were reference categories.

Models were fitted by maximum likelihood using statsmodels MixedLM.

Fixed-efect coeficients represent adjusted mean diferences in the outcome relative to the reference category while holding the other included acquisition variables constant. Ninety-five percent confidence intervals and p-values were taken from the fitted mixed-efects model output. Because the dataset contains repeated observations but relatively few independent subjects in some Fitzpatrick strata, inferential interpretation emphasizes efect estimates and confidence intervals rather than treating the number of recordings as the number of independent participants.

The primary models were additive. To test compounded heterogeneity directly rather than infer it from stratified cell means, two secondary interaction models were fitted for each outcome: Fitzpatrick×motion and Fitzpatrick×lighting, retaining the same subject random intercept and the four Fitzpatrick categories. Joint Wald tests evaluated the full interaction coeficient blocks (12 degrees of freedom for motion and 9 for lighting). Interaction interpretation was based on the block test rather than isolated cell coeficients. Subject 24 remained excluded because of inconsistent Fitzpatrick metadata. These observational interactions are not interpreted as causal optical mechanisms.

Fitzpatrick III-IV-V-VI categories were not collapsed into a binary light/dark variable.

## 4.11 Subject-held-out condition-aware endpoint-discrepancy experiment

A follow-up experiment tested the extent to which observation variables identified in the heterogeneous analysis could reduce the endpoint discrepancy on unseen subjects, with out-of-fold MAE as the primary outcome. The complete computational sequence is summarized in Algorithm 4. The target was contact-PPG reference HR, and the base predictor was camera-derived rPPG HR. Here, $M _ { 0 } – M _ { 4 }$ denote nested predictor information sets rather than diferent learning algorithms; each information set was evaluated independently using linear regression, ridge regression, and random forest regression. Five nested specifications were evaluated: $M _ { 0 }$ used rPPG HR alone; $M _ { 1 }$ added motion/activity; $M _ { 2 }$ added lighting; $M _ { 3 }$ additionally added Fitzpatrick group; and $M _ { 4 }$ additionally added aggregate RGB SNR. Subject 24 remained excluded, leaving 625 recordings from 32 subjects.

Generalization was evaluated with five-fold GroupKFold cross-validation at the subject level so that all recordings from a participant appeared entirely in either training or test data within a fold. Three deliberately simple learner families were compared: ordinary linear regression, ridge regression $( \alpha = 1 )$ , and random forest regression. Categorical variables were one-hot encoded within the training pipeline; numeric variables were standardized for the linear and ridge models. No deep neural network or end-to-end video model was used because the purpose was to test whether measured observation context reduced subject-held-out endpoint discrepancy, as measured primarily by MAE, rather than to maximize benchmark performance.

The primary outcome was out-of-fold MAE against the synchronized contact-PPG HR. RMSE, median absolute error, Pearson correlation, and $R ^ { 2 }$ were secondary diagnostic summaries and were not used to define the primary RQ5 comparison. Incremental improvement was evaluated relative to $M _ { 0 }$ within each learner family. To respect subject dependence, the MAE diference between $M _ { 0 }$ and each expanded model was assessed with a 5,000-resample subject-level bootstrap; a confidence interval entirely below zero was interpreted as evidence that the expanded model reduced MAE. This experiment was analyzed as a follow-up test of actionability, not as part of the original fixed-pathway benchmark.

## 4.12 Measured RGB SNR

Recording-level red-, green-, and blue-channel SNR values were computed from the extracted color traces and retained for all 655 available recordings. The three channel SNRs were first summarized separately by Fitzpatrick group to determine whether aggregate measured signal quality showed the large monotonic decline that would be expected under a simple attenuation-only explanation.

For the final sensitivity model, one aggregate SNR variable was defined before fitting: SNRRGB $= ( \mathrm { S N R . R \mathrm { ~ + ~ } S N R . G \mathrm { ~ + ~ } S N R . B } ) / 3$ . No alternative definitions such as minimum channel SNR, maximum channel SNR, individual-channel model selection, or post-hoc weighted combinations were searched.

## 4.13 Simple optical-attenuation adequacy check

A Beer-Lambert-inspired attenuation model was evaluated only as a deliberately simplified mechanistic adequacy check. The model used SNR dB = baseline SNR - 20 × OD melanin to generate a monotonic attenuation pattern across Fitzpatrick groups. Its purpose was not to estimate melanin concentration or to provide a complete tissue-optics model; rather, it tested whether a simple aggregate attenuation explanation could reproduce the measured RGB SNR pattern.

Observed and predicted group-level SNR values were compared and summarized by RMSE. Because the predicted decline was much larger than the measured Fitzpatrick variation, the model was rejected as an adequate explanation. No coeficients were subsequently tuned to force agreement with the empirical data, and the model was not used to correct HR or rPPG waveforms.

## 4.14 SNR-adjusted sensitivity model

The mixed-efects heterogeneity models and SNR-adjusted sensitivity analysis are summarized in Algorithm 3.

Model 1b tested whether measured aggregate RGB SNR statistically accounted for the Fitzpatrickassociated endpoint coeficient. It retained the exact Model 1 structure and added only SNRRGB: $\mathrm { D H R } \ \tilde { \mathrm { ~ \sim ~ } } \ \mathrm { F i t z p a t r i c k } + \mathrm { L i g h t i n g ~ \mathrm { + ~ M o t i o n ~ + ~ S N R R G B } + \Delta ( 1 | s u b j e c t ) }$ . The analysis population, reference categories, maximum-likelihood fitting procedure, and subject random intercept were unchanged.

The principal quantities were the SNRRGB coeficient and the change in the Fitzpatrick VI-versus-III coeficient relative to Model 1. Coeficient attenuation was calculated descriptively as 100 × (βVI,Model1 - βVI,Model1b)/βVI,Model1. This quantity was explicitly not interpreted as causal mediation because the analysis was observational and did not satisfy a formal mediation design.

## 4.15 Software, reproducibility, and analysis freeze

Video processing, signal analysis, nonlinear estimation, statistical modeling, and figure generation were implemented in Python. For the maximal-Lyapunov analysis, the authoritative implementation was the corrected prospective/frozen pipeline: all 33 subjects were retained for dynamical analysis, recording indices were 0–19, and every expected subject-recording combination was prospectively assigned a success or failure status. The rPPG analysis applied the same frozen estimator and parameters used for contact PPG; only waveform input/output handling difered. The reproducibility record includes the fixed rPPG observation implementation, frozen rPPG waveforms, prospective PPG and rPPG maximal-Lyapunov outputs, failure/accounting tables, matched structural analyses, mixed-efects models, RGB SNR analysis, optical-attenuation adequacy check, Model 1b sensitivity analysis, Fitzpatrick-by-condition interaction tests, the subject-held-out condition-aware gap-reduction experiment, and scripts that regenerate the principal figures from frozen result files.

The primary evidence sequence was frozen through Model 1b. Two follow-up analyses were then added as explicitly labeled extensions rather than used to redefine the frozen primary results: (i) joint Fitzpatrick-by-motion and Fitzpatrick-by-lighting interaction tests, and (ii) a subject-held-out condition-aware endpoint learner using the already measured motion, lighting, Fitzpatrick, and aggregate-SNR variables. We did not proceed to Fitzpatrick-specific correction, alternative rPPG algorithms, channel-specific model searching, additional nonlinear metrics selected after observing λmax results, or open-ended covariate searching to reduce the Fitzpatrick coeficient.

## 5 Algorithmic specification

The following algorithms express the reported analyses at the level of their mathematical operations. They define the quantities a reader must reproduce while leaving software-specific details to the corresponding Methods subsections. The fixed CHROM observation operator is documented in Section 4.4; no algorithm below changes that observation pathway or introduces post-hoc optimization.

## Algorithm 1 | Property-specific PPG–rPPG recoverability assessment

Mathematical procedure for evaluating endpoint discrepancy and recording-specific structural and dynamical correspondence between synchronized contact PPG and camera-derived rPPG.

Input: synchronized pairs $\mathcal { X } = \{ ( x _ { i } ^ { P } , x _ { i } ^ { R } ) \} _ { i = 1 } ^ { N }$ , where P denotes contact PPG and R denotes camera-derived $\operatorname { r P P G } ;$ property operators $\Phi = \{ \phi _ { j } \} _ { j = 1 } ^ { J }$

1. For each property $\phi _ { j }$ , map every synchronized recording pair into the corresponding property space,

$$
z _ { i j } ^ { P } = \phi _ { j } ( x _ { i } ^ { P } ) , \qquad z _ { i j } ^ { R } = \phi _ { j } ( x _ { i } ^ { R } ) .
$$

The operator $\phi _ { j }$ may return a scalar endpoint, a structural descriptor, a recurrence statistic, or a nonlinear dynamical quantity.

2. For scalar endpoint HR, define the recording-level loss

$$
D _ { H R , i } = \left| H R _ { i } ^ { R } - H R _ { i } ^ { P } \right| .
$$

For each conventional structural property $j \in \{ A C F , P S D , R E C \}$ , compute the implemented matched-pair comparison functional

$$
S _ { i j } ^ { \mathrm { m a t c h } } = S _ { j } ( z _ { i j } ^ { P } , z _ { i j } ^ { R } ) ,
$$

where $S _ { j }$ denotes ACF similarity, cardiac-band PSD distance, or recurrence-rate discrepancy as defined in the corresponding Methods subsection.

3. Construct the correspondence-breaking structural control by replacing the synchronized rPPG identity i with a shufled identity π(i),

$$
\begin{array} { r } { S _ { i j } ^ { \mathrm { s h u f } } = S _ { j } \Big ( z _ { i j } ^ { P } , z _ { \pi ( i ) j } ^ { R } \Big ) . } \end{array}
$$

For similarity statistics, recording specificity requires $S _ { j } ^ { \mathrm { m a t c h } }$ to exceed its shufled counterpart in the implemented comparison; for distance/discrepancy statistics, the expected direction is reversed. The shufled control breaks recording identity while retaining the analyzed signal collection.

4. Evaluate separately whether any candidate statistic is intrinsically non-discriminative under the implemented parameterization. For a statistic $q _ { j }$ , if

$$
\mathrm { V a r } ( q _ { j } ) \approx 0
$$

or its values are efectively concentrated at an admissible boundary, do not interpret that statistic as evidence for or against recording-specific correspondence. In the reported analysis this rule applies to the saturated determinism statistic; recurrence-rate discrepancy remains part of the matched-versus-shufled structural comparison in Step 3.

5. For the maximal-Lyapunov property, obtain $\lambda _ { i } ^ { P }$ and $\lambda _ { i } ^ { R }$ from Algorithm 2 and restrict the paired analysis to the common valid index set

$$
\begin{array} { r } { \mathcal { T } _ { \lambda } = \{ i : \lambda _ { i } ^ { P } \mathrm { ~ a n d ~ } \lambda _ { i } ^ { R } \mathrm { ~ a r e ~ v a l i d } \} . } \end{array}
$$

Compute observed recording-specific correspondence

$$
r _ { \lambda } ^ { \mathrm { o b s } } = \mathrm { c o r r } \bigl ( \{ \lambda _ { i } ^ { P } \} _ { i \in \mathbb { Z } _ { \lambda } } , \{ \lambda _ { i } ^ { R } \} _ { i \in \mathbb { Z } _ { \lambda } } \bigr ) .
$$

6. Generate the condition-preserving subject-permutation null

$$
\begin{array} { r } { r _ { \lambda } ^ { ( b ) } = \mathrm { c o r r } \left( \{ \lambda _ { i } ^ { P } \} _ { i \in \mathcal { T } _ { \lambda } } , \{ \lambda _ { \pi _ { b } ( i ) } ^ { R } \} _ { i \in \mathcal { T } _ { \lambda } } \right) , \qquad b = 1 , \ldots , B , } \end{array}
$$

and evaluate $r _ { \lambda } ^ { \mathrm { o b s } }$ relative to $\{ r _ { \lambda } ^ { ( b ) } \} _ { b = 1 } ^ { B }$

7. Report recoverability property by property as the joint evidence from endpoint loss, matchedversus-shufled structural comparison, recurrence-statistic informativeness, and recording-specific dynamical correspondence. Marginal similarity is treated as population-level plausibility, not by itself as synchronized-recording preservation.

Output: a property-specific recoverability profile

$$
\mathcal { R } = \{ \mathcal { R } _ { H R } , \mathcal { R } _ { A C F } , \mathcal { R } _ { P S D } , \mathcal { R } _ { R E C } , \mathcal { R } _ { \lambda } \} .
$$

## Algorithm 2 | Study-specific maximal-Lyapunov estimation and paired discrepancy

Study-specific procedure for estimating $\hat { \lambda } _ { \mathrm { m a x } }$ from each modality under identical validity rules and forming the matched PPG–rPPG dynamical discrepancy.

Input: one processed waveform $x ( t )$ from either contact PPG or camera-derived rPPG, with the preprocessing, analysis interval, SQI gate, and numerical constants specified in the Methods.

1. Let $\tilde { { \boldsymbol { x } } } ( t ) = \mathcal { P } [ { \boldsymbol { x } } ( t ) ]$ denote the implemented preprocessing operator, including resampling and filtering. Apply the prespecified spectral-quality gate

$$
Q ( \tilde { x } ) \geq q _ { 0 } ,
$$

where $Q ( \cdot )$ and threshold $q _ { 0 }$ are the study-specific quality criterion defined by the implemented estimator and described in Methods. Waveforms failing this gate do not enter the Lyapunov estimate.

2. Estimate the delay from the average-mutual-information sequence,

$$
\tau = \mathcal { T } ( A M I _ { \tilde { x } } ( 1 ) , \dots , A M I _ { \tilde { x } } ( L ) ) ,
$$

using the implemented first-local-minimum rule and its prespecified fallback.

3. Reconstruct the delay-coordinate trajectory in $\mathbb { R } ^ { m }$

$$
\mathbf { X } _ { t } = \left[ \widetilde { x } _ { t } , \widetilde { x } _ { t - \tau } , \ldots , \widetilde { x } _ { t - ( m - 1 ) \tau } \right] ^ { T } ,
$$

with the embedding dimension m specified in the Methods.

4. Let $T _ { W }$ denote the temporal-exclusion duration and let

$$
W _ { s } = \lceil f _ { s } T _ { W } \rceil
$$

be the corresponding exclusion width in samples at analysis sampling frequency $f _ { s }$ . For each admissible state $\mathbf { X } _ { t } ,$ , define its temporally admissible nearest neighbor

$$
j ( t ) = \arg \operatorname* { m i n } _ { s : | s - t | > W _ { s } } \| \mathbf { X } _ { t } - \mathbf { X } _ { s } \| _ { 2 } .
$$

In the reported implementation, $T _ { W } = 1 0 0$ ms after resampling to the common analysis time base. 5. For forward step k, compute the ensemble mean log-divergence

$$
\bar { d } ( k ) = \frac { 1 } { \left| \mathscr { A } _ { k } \right| } \sum _ { t \in \mathscr { A } _ { k } } \log \left\| \mathbf { X } _ { t + k } - \mathbf { X } _ { j ( t ) + k } \right\| _ { 2 } ,
$$

where $\mathcal { A } _ { k }$ contains neighbor pairs for which both trajectories remain defined at step k.

6. Express forward divergence time as

$$
t _ { k } = \frac { k } { f _ { s } } ,
$$

and estimate the study-specific maximal-divergence slope by the implemented linear fit

$$
\hat { \lambda } _ { \operatorname* { m a x } } = \mathrm { s l o p e } \{ ( t _ { k } , \bar { d } ( k ) ) \} _ { k \in \mathcal { K } } ,
$$

where K is the fitting region used in the reported analysis. Apply the same operational validity rule used to generate the reported results; failed estimates remain missing rather than being imputed.

7. Apply Steps 1–6 independently to $x _ { i } ^ { P }$ and $x _ { i } ^ { R }$ . For recording $i ,$ form the paired dynamical discrepancy only when both estimates are valid,

$$
\begin{array} { r } { D _ { \lambda , i } = \left| \hat { \lambda } _ { \operatorname* { m a x } , i } ^ { P } - \hat { \lambda } _ { \operatorname* { m a x } , i } ^ { R } \right| . } \end{array}
$$

Recording-specific correspondence across the common valid set is then evaluated by Algorithm 1.   
Output: $\hat { \lambda } _ { \mathrm { m a x } }$ with validity status for each modality and the paired quantity $D _ { \lambda }$ where defined.

## Algorithm 3 | Mathematical model of recoverability heterogeneity

Subject-aware mixed-efects framework for quantifying how endpoint and dynamical discrepancies vary across measured observation conditions and for assessing RGB-SNR coeficient sensitivity.

Input: recording-level discrepancies $D _ { H R , i }$ and, where valid, $D _ { \lambda , i } ;$ Fitzpatrick group $F _ { i } ;$ lighting $L _ { i } ;$ motion $M _ { i } ;$ subject $s ( i )$ ; channel-wise SNR indices.

1. Define the two discrepancy variables

$$
D _ { H R , i } = \left| H R _ { i } ^ { R } - H R _ { i } ^ { P } \right| , \qquad D _ { \lambda , i } = \left| \lambda _ { i } ^ { R } - \lambda _ { i } ^ { P } \right| .
$$

Use the Fitzpatrick-consistent population for models containing $F _ { i }$ and the common valid Lyapunov set for $D _ { \lambda }$

2. Represent repeated measurements by a model-specific subject random intercept $b _ { s ( i ) }$ , with

$$
b _ { s } \sim { \mathcal { N } } ( 0 , \sigma _ { b } ^ { 2 } ) .
$$

For endpoint loss, fit

$$
D _ { H R , i } = \beta _ { 0 } + \beta _ { F } ^ { T } F _ { i } + \beta _ { L } ^ { T } L _ { i } + \beta _ { M } ^ { T } M _ { i } + b _ { s ( i ) } + \varepsilon _ { i } ,
$$

and for dynamical loss fit the analogous model

$$
D _ { \lambda , i } = \gamma _ { 0 } + \gamma _ { F } ^ { T } F _ { i } + \gamma _ { L } ^ { T } L _ { i } + \gamma _ { M } ^ { T } M _ { i } + b _ { s ( i ) } + \eta _ { i } .
$$

The random-intercept variance is estimated separately for each fitted model.

3. For the prespecified secondary interaction analyses, augment one condition family at a time. For $D _ { i } \in \{ D _ { H R , i } , D _ { \lambda , i } \}$ , the Fitzpatrick-by-motion model is

$$
D _ { i } = \alpha _ { 0 } + \alpha _ { F } ^ { T } F _ { i } + \alpha _ { L } ^ { T } L _ { i } + \alpha _ { M } ^ { T } M _ { i } + \alpha _ { F M } ^ { T } ( F _ { i } \otimes M _ { i } ) + b _ { s ( i ) } + e _ { i } ,
$$

whereas the Fitzpatrick-by-lighting model replaces the interaction term by

$$
\begin{array} { r } { \alpha _ { F L } ^ { T } ( F _ { i } \otimes L _ { i } ) . } \end{array}
$$

Test the full interaction coeficient block jointly rather than interpreting selected cells post hoc.

4. Define the descriptive aggregate channel-quality index

$$
S N R _ { R G B , i } = \frac { S N R _ { R , i } + S N R _ { G , i } + S N R _ { B , i } } { 3 } .
$$

On exactly the endpoint-model population, fit the SNR-adjusted sensitivity model

$$
D _ { H R , i } = \theta _ { 0 } + \theta _ { F } ^ { T } F _ { i } + \theta _ { L } ^ { T } L _ { i } + \theta _ { M } ^ { T } M _ { i } + \theta _ { S } S N R _ { R G B , i } + b _ { s ( i ) } + \xi _ { i } .
$$

5. Let $\beta _ { V I , 1 }$ denote the Fitzpatrick VI-versus-III coeficient before SNR adjustment and $\beta _ { V I , 1 b }$ the corresponding coeficient after SNR adjustment. Quantify descriptive attenuation as

$$
A _ { V I } = 1 0 0 \left( \frac { \beta _ { V I , 1 } - \beta _ { V I , 1 b } } { \beta _ { V I , 1 } } \right) \% .
$$

Interpret $A _ { V I }$ as coeficient sensitivity to aggregate SNR adjustment, not as a causal mediation quantity.

Output: fixed-efect contrasts for endpoint and dynamical loss, joint interaction tests, and the SNR-adjusted coeficient-sensitivity measure $A _ { V I }$

## Algorithm 4 | Mathematical formulation of condition-aware endpoint-discrepancy reduction

Subject-disjoint prediction procedure for testing whether incrementally added observation context reduces held-out PPG–rPPG endpoint discrepancy, with MAE as the primary performance measure.

Input: contact-PPG target $Y _ { i } = H R _ { i } ^ { P }$ , camera-derived HR $H _ { i } = H R _ { i } ^ { R }$ , observation variables $( M _ { i } , L _ { i } , F _ { i } , S _ { i } )$ , and subject identifier $s ( i )$ , with $S _ { i } = S N R _ { R G B , i }$

1. Define the nested information sets

$$
\begin{array} { r l } & { \mathcal { X } _ { 0 } = \{ H \} , \qquad \mathcal { X } _ { 1 } = \{ H , M \} , \qquad \mathcal { X } _ { 2 } = \{ H , M , L \} , } \\ & { \qquad \mathcal { X } _ { 3 } = \{ H , M , L , F \} , \qquad \mathcal { X } _ { 4 } = \{ H , M , L , F , S \} . } \end{array}
$$

Thus each successive model asks whether an additional observation variable contains predictive information about contact-PPG HR beyond the preceding set.

2. Partition the subject set into five disjoint folds $\mathcal { G } _ { 1 } , \ldots , \mathcal { G } _ { 5 }$ . For fold g, define

$$
\mathcal { D } _ { t e s t } ^ { ( g ) } = \{ i : s ( i ) \in \mathcal { G } _ { g } \} , \qquad \mathcal { D } _ { t r a i n } ^ { ( g ) } = \{ i : s ( i ) \notin \mathcal { G } _ { g } \} ,
$$

so that

$$
\{ s ( i ) : i \in \mathcal { D } _ { t r a i n } ^ { ( g ) } \} \cap \{ s ( i ) : i \in \mathcal { D } _ { t e s t } ^ { ( g ) } \} = \emptyset .
$$

3. For learner family ℓ and information set $\mathcal { X } _ { k }$ , estimate a fold-specific prediction function using training subjects only,

$$
\hat { f } _ { k , \ell } ^ { \left( g \right) } = \mathcal { A } _ { \ell } \left( \mathcal { D } _ { t r a i n } ^ { \left( g \right) } , \mathcal { X } _ { k } , Y \right) ,
$$

where $\mathbf { \mathcal { A } } _ { \ell }$ denotes the implemented linear, ridge, or random-forest learner with the preprocessing specified in Methods.

4. Generate exactly one held-out prediction for each recording,

$$
\hat { Y } _ { i , k , \ell } = \hat { f } _ { k , \ell } ^ { ( g ) } ( X _ { i , k } ) , \qquad i \in \mathcal { D } _ { t e s t } ^ { ( g ) } ,
$$

and concatenate predictions over $g = 1 , \ldots , 5 .$

5. Evaluate each $( k , \ell )$ using the pooled out-of-fold errors

$$
e _ { i , k , \ell } = Y _ { i } - \hat { Y } _ { i , k , \ell } ,
$$

Let $\mathcal { D } _ { \mathrm { e v a l } }$ denote the complete out-of-fold evaluation population and $N _ { \mathrm { e v a l } } = | \mathcal { D } _ { \mathrm { e v a l } } |$ . Then

$$
M A E _ { k , \ell } = \frac { 1 } { N _ { \mathrm { e v a l } } } \sum _ { i \in \mathcal { D } _ { \mathrm { e v a l } } } \vert e _ { i , k , \ell } \vert , \qquad R M S E _ { k , \ell } = \sqrt { \frac { 1 } { N _ { \mathrm { e v a l } } } \sum _ { i \in \mathcal { D } _ { \mathrm { e v a l } } } e _ { i , k , \ell } ^ { 2 } } ,
$$

along with the reported median absolute error, Pearson correlation, and $R ^ { 2 }$

6. Within each learner family, treat $\mathcal { X } _ { 0 }$ as the learned camera-HR calibration baseline and define the incremental endpoint benefit

$$
\Delta M A E _ { k , \ell } = M A E _ { k , \ell } - M A E _ { 0 , \ell } , \qquad k = 1 , \dots , 4 .
$$

Hence $\Delta M A E _ { k , \ell } < 0$ denotes lower subject-held-out error after adding observation context.

7. Let $B ^ { ( b ) }$ be a bootstrap resample of subjects. Using the fixed out-of-fold predictions, recompute

$$
\Delta M A E _ { k , \ell } ^ { ( b ) } = M A E _ { k , \ell } ^ { ( b ) } - M A E _ { 0 , \ell } ^ { ( b ) } , \qquad b = 1 , \ldots , B .
$$

Form the reported confidence interval from the empirical distribution of $\{ \Delta M A E _ { k , \ell } ^ { ( b ) } \} _ { b = 1 } ^ { B }$

8. Declare evidence of incremental held-out MAE reduction only when the complete confidence interval for $\Delta M A E _ { k , \ell }$ lies below zero. This criterion concerns endpoint prediction only and is not interpreted as waveform reconstruction or PPG–rPPG equivalence.

Output: subject-held-out performance for $\mathcal { X } _ { 0 } - \mathcal { X } _ { 4 }$ and uncertainty-calibrated incremental endpoint benefit $\Delta M A E _ { k , \ell }$

## 6 Results

## 6.1 Endpoint recovery under the fixed observation pathway

The MMPD design comprises 33 participants with 20 recording indices (0–19), yielding 660 expected subject-recording combinations. Five files were unavailable, leaving 655 recordings for the fixedpathway benchmark. The observation pipeline processed all 655 available recordings without technical failure. The dynamics pipeline retained all 33 subjects and assigned an explicit success or failure status to every expected combination; Subject 24 was excluded only from Fitzpatrick stratified/model-based analyses because its skin-tone metadata were internally inconsistent, not from the dynamics computation. As summarized in Table 2, HR recovery remained weak: MAE was 15.26 beats/min, RMSE 20.81 beats/min, MAPE 17.97%, and Pearson $\mathrm { r { = } 0 . 0 8 0 1 }$ . Mean signed error was -6.80 beats/min. Reference HR averaged 83.06 beats/min (SD 15.88), compared with 76.27 beats/min (SD 12.96) for camera-derived rPPG. As an implementation check, these results were compared with the published CHROM benchmark on MMPD reported through the rPPG-Toolbox framework: MAE $1 3 . 6 6 \pm 0 . 5 0$ bpm, RMSE $1 8 . 7 6 \pm 2 3 . 8 2$ bpm, MAPE $1 6 . 0 0 \pm 0 . 5 7 \%$ , and Pearson $r = 0 . 0 8 \pm 0 . 0 4 \ [ 6 , 1 1 ]$ . The present Pearson correlation $( r = 0 . 0 8 0 1 )$ therefore closely reproduces the published correlation, while the present MAE (15.26 bpm) is modestly higher than the published 13.66-bpm benchmark. This agreement in correlation and broadly comparable absolute-error regime supported use of the same camera-to-rPPG implementation for the downstream recoverability analyses.

## Table 2 | Fixed camera-rPPG observation benchmark on the 655 available MMPD recordings.

Heart rate from the fixed camera-rPPG observation pathway is compared with reference HR derived from synchronized contact PPG. MAE, RMSE, and MAPE quantify absolute endpoint error; Pearson r quantifies recording-level linear correspondence; mean signed error indicates systematic underestimation when negative. The benchmark is used as an implementation and endpoint-recoverability check rather than as an optimization result.
<table><tr><td>Metric</td><td>Fixed camera-rPPG observation benchmark</td></tr><tr><td>Available recordings</td><td>655</td></tr><tr><td>MAE</td><td>15.26 bpm</td></tr><tr><td>RMSE</td><td>20.81 bpm</td></tr><tr><td>MAPE</td><td>17.97%</td></tr><tr><td>Pearson r</td><td>0.0801</td></tr><tr><td>Mean SNR</td><td>-6.56 dB</td></tr><tr><td>Mean signed error</td><td>-6.80 bpm</td></tr><tr><td>GT HR mean (SD)</td><td>83.06 (15.88) bpm</td></tr><tr><td>rPPG HR mean (SD)</td><td>76.27 (12.96) bpm</td></tr></table>

## 6.2 Conventional structure shows selective recording-specific correspondence

Table 3 summarizes the matched-versus-shufled validation used to determine whether apparent structural similarity was recording-specific. ACF similarity was 0.2863 for matched pairs and 0.2440 after shufling, a diference of +0.0422. Cardiac-band PSD distance was 0.0515 for matched pairs and 0.0524 for shufled pairs (diference -0.0009), while recurrence-rate discrepancy was 0.0246 and 0.0263, respectively (diference -0.0017). Thus, temporal autocorrelation retained modest recording specificity, whereas the spectral and recurrence-rate measures provided little separation from the shufled control. Determinism was identically 1.0 and was excluded because it was non-discriminative.

Table 3 | Matched-versus-shufled structural recoverability analysis.  
Matched PPG–rPPG property correspondence is compared with a shufled-pair null distribution to distinguish recording-specific preservation from cross-recording similarity. Larger matched-versusshufled separation indicates stronger evidence of recording-specific structural correspondence.
<table><tr><td>Metric</td><td>Matched</td><td>Shuffled</td><td>Matched-shuffled interpretation</td></tr><tr><td>ACF similarity</td><td>0.2863</td><td>0.2440</td><td>+0.0422; modest recording specificity</td></tr><tr><td>PSD distance</td><td>0.0515</td><td>0.0524</td><td>-0.0009; marginal</td></tr><tr><td>Recurrence-rate difference</td><td>0.0246</td><td>0.0263</td><td>-0.0017; marginal</td></tr><tr><td>Determinism</td><td>1.000</td><td>1.000</td><td>Non-discriminative; excluded</td></tr></table>

## 6.3 Recording-specific maximal-Lyapunov correspondence is absent

The study-specific λmax estimator was applied identically to PPG and camera-derived rPPG. The primary analysis used the exact intersection of recordings with valid estimates in both modalities. As reported in Table 4, across 645 matched recordings, recording-specific correspondence was essentially absent $\mathrm { ( r \lambda ) { = } 0 . 0 2 3 1 }$ , 95% CI -0.0542 to 0.1001; $\mathrm { p { = } 0 . 5 5 7 9 ) }$ . A 10,000-iteration conditionpreserving subject permutation test produced $\mathrm { p { = } 0 . 5 5 8 4 }$ , providing no evidence that the observed correspondence exceeded that expected after breaking subject identity.

## Table 4 | Recording-specific maximal-Lyapunov correspondence test.

Pearson correlation quantifies matched recording-level correspondence between valid contact-PPG and camera-rPPG maximal-Lyapunov estimates. The confidence interval and permutation test assess whether the observed correspondence difers detectably from zero and from correspondence expected under disrupted pairing.
<table><tr><td>N matched</td><td>r_lambda</td><td>95% CI</td><td>Pearson p</td><td>Permutation p</td></tr><tr><td>645</td><td>0.0231</td><td>[-0.0542, 0.1001]</td><td>0.5579</td><td>0.5584</td></tr></table>

Marginal similarity did not alter this conclusion. Mean λmax was approximately 2.063 for PPG and 2.256 for $\mathrm { r P P G }$ but the corresponding SDs were 0.9795 and 0.2809. Thus, rPPG retained only about 29% of the PPG between-recording variability in this descriptor. The combination of near-zero matched correlation and strong variance compression indicates absence of recording-specific dynamical correspondence despite overlap in the marginal value ranges. The matched and marginal maximal-Lyapunov results are visualized in Supplementary Figure S4.

## 6.4 Observation-associated discrepancy is property-specific

## 6.4.1 Endpoint discrepancy across Fitzpatrick skin type

After removal of one duplicate match key, 645 unique matched records remained. Subject 24 was excluded only from Fitzpatrick analyses because its raw Fitzpatrick-group metadata were inconsistent, leaving 625 recordings from 32 subjects. Table 5 shows that mean absolute HR discrepancy (DHR) increased from 11.43 beats/min for Fitzpatrick III to 20.64 beats/min for Fitzpatrick VI. By contrast, mean absolute λmax discrepancy (Dλ) remained approximately flat across Fitzpatrick groups.

## 6.4.2 Subject-aware adjusted models confirm a Fitzpatrick endpoint gradient

In Model 1, DHR was modeled as a function of Fitzpatrick group, lighting, and motion/activity, with a subject-specific random intercept. The adjusted Model 1 contrasts are reported in Table 6. Relative to Fitzpatrick III, the adjusted diference was +6.37 beats/min for IV (95% CI -0.12 to 12.87), +6.72 for V (95% CI 1.19 to 12.25), and +9.32 for VI (95% CI 3.78 to 14.86). The increasing endpoint discrepancy therefore persisted after adjustment for measured lighting and motion conditions. A coeficient-only display of the Fitzpatrick contrasts is provided in Supplementary Figure S1.

Table 5 | Unadjusted Fitzpatrick-stratified endpoint and dynamical discrepancies for the 625-record, 32-subject analysis set.  
$D _ { H R }$ is the absolute contact-PPG versus camera-rPPG heart-rate discrepancy in beats/min; $D _ { \lambda }$ is the absolute discrepancy in maximal Lyapunov exponent. Subject $\it { 2 4 }$ is excluded from this stratified analysis because of inconsistent Fitzpatrick metadata. These summaries are descriptive and do not adjust for repeated measures, lighting, or motion/activity.
<table><tr><td>Fitzpatrick</td><td>Subjects</td><td>N</td><td>Mean  $D _ { H R }$ </td><td>Median  $D _ { H R }$ </td><td>Mean  $D _ { \lambda }$ </td><td>Median  $D _ { \lambda }$ </td></tr><tr><td>III</td><td>16</td><td>315</td><td>11.43</td><td>8.79</td><td>0.8187</td><td>0.6238</td></tr><tr><td>IV</td><td>4</td><td>75</td><td>18.01</td><td>15.82</td><td>0.8082</td><td>0.7274</td></tr><tr><td>V</td><td>6</td><td>119</td><td>18.29</td><td>16.70</td><td>0.8016</td><td>0.6058</td></tr><tr><td>VI</td><td>6</td><td>116</td><td>20.64</td><td>18.46</td><td>0.8246</td><td>0.6801</td></tr></table>

Table 6 | Subject-aware Model 1 Fitzpatrick contrasts for endpoint discrepancy. Estimates are adjusted diferences in absolute PPG–rPPG HR discrepancy $( D _ { H R } )$ , in beats/min, relative to Fitzpatrick III after adjustment for lighting and motion/activity, with a subject-specific random intercept. Positive estimates indicate larger endpoint discrepancy than the Fitzpatrick III reference group.
<table><tr><td>Contrast</td><td>Estimate (bpm)</td><td>95% CI</td><td>p</td></tr><tr><td></td><td>+6.37</td><td>[-0.12, 12.87]</td><td>0.051</td></tr><tr><td>IV vs III</td><td></td><td></td><td></td></tr><tr><td>V vs III</td><td>+6.72</td><td>[1.19, 12.25]</td><td>0.017</td></tr><tr><td>VI vs III</td><td>+9.32</td><td>[3.78, 14.86]</td><td>0.001</td></tr></table>

Lighting efects were smaller than the largest Fitzpatrick and motion efects. Relative to incandescent lighting, LED-low was associated with +3.06 beats/min (95% CI 0.47 to 5.66) and natural lighting with +2.93 beats/min (95% CI 0.31 to 5.55). Relative to rotation, walking was associated with +9.69 beats/min (95% CI 6.77 to 12.60), and stationary-after-exercise with +9.05 beats/min (95% CI 6.13 to 11.98).

![](images/d5899152badb2c188eb73562adc1ecad43e588315472f89a88e587bd424b8285.jpg)  
Figure 2: Heterogeneous recoverability of endpoint and dynamical quantities across Fitzpatrick groups. (A) Distribution of absolute PPG–rPPG heart-rate discrepancy, D<sub>HR</sub>, across Fitzpatrick groups. (B) Distribution of absolute maximal-Lyapunov discrepancy, $D _ { \lambda }$ . (C) Subject-aware adjusted Fitzpatrick contrasts for $D _ { H R }$ relative to Fitzpatrick III after adjustment for lighting and motion/activity. (D) Corresponding adjusted contrasts for $D _ { \lambda }$ . The endpoint discrepancy increases from Fitzpatrick III toward higher-numbered Fitzpatrick groups, whereas the tested dynamical discrepancy does not exhibit the same gradient. Error bars in C–D denote 95% confidence intervals; contrasts are relative to Fitzpatrick III. Panels A–B are unadjusted distributions, whereas C–D are estimates from subject-aware mixed-efects models adjusted for lighting and motion/activity. Fitzpatrick analyses include 625 recordings from 32 subjects after exclusion of the subject with inconsistent Fitzpatrick-group metadata. The contrast between the endpoint and dynamical panels is interpreted as property-specific heterogeneity, not as evidence that nonlinear dynamics are preserved.

## 6.4.3 The dynamical discrepancy does not show the same Fitzpatrick gradient

Model 2 applied the same mixed-efects structure to Dλ. The adjusted Model 2 contrasts are reported in Table 7. Relative to Fitzpatrick III, coeficients were -0.006 for IV (95% CI -0.224 to 0.212), -0.017 for V (95% CI -0.202 to 0.167), and +0.003 for VI (95% CI -0.183 to 0.188); all p-values exceeded 0.80. The absence of a corresponding Fitzpatrick gradient is visualized in Figure 2B,D. This result should not be interpreted as preservation of nonlinear dynamics, because recording-specific λmax correspondence was already absent overall. Rather, the magnitude of the PPG-rPPG λmax discrepancy did not exhibit an additional Fitzpatrick-associated gradient.

Table 7 | Subject-aware Model 2 Fitzpatrick contrasts for dynamical discrepancy.  
Estimates are adjusted diferences in absolute PPG–rPPG maximal-Lyapunov discrepancy $( D _ { \lambda } )$ relative to Fitzpatrick III after adjustment for lighting and motion/activity, with a subject-specific random intercept. Confidence intervals and p-values quantify uncertainty in the adjusted contrasts.
<table><tr><td>Contrast</td><td>D_lambda estimate</td><td>95% CI</td><td>p</td></tr><tr><td></td><td>-0.006</td><td>[-0.224, 0.212]</td><td>0.955</td></tr><tr><td>IV vs III</td><td></td><td></td><td></td></tr><tr><td>V vs III VI vs III</td><td>-0.017 +0.003</td><td>[-0.202, 0.167] [-0.183, 0.188]</td><td>0.856 0.980</td></tr></table>

Figure 3 compares endpoint and dynamical discrepancy across lighting (panels A–B) and motion/activity (panels C–D), showing that the two properties do not vary in parallel across acquisition conditions.

## 6.4.4 Fitzpatrick-by-condition interactions are specific to endpoint motion heterogeneity

The follow-up interaction experiment tested whether the efect of motion/activity or lighting on discrepancy depended on Fitzpatrick group. Cell counts were inspected before inference; all Fitzpatrick-by-motion and Fitzpatrick-by-lighting combinations were represented, although the number of independent participants remained limited in Fitzpatrick IV–VI.

For $D _ { H R }$ , the joint Fitzpatrick×motion interaction block was significant (Wald $\chi ^ { 2 } = 2 8 . 7 8 9 , \mathrm { d f = 1 2 }$ $p = 0 . 0 0 4 2 3 4 )$ , indicating that motion-related changes in endpoint discrepancy were not constant across Fitzpatrick groups. In contrast, the Fitzpatrick×lighting block was not significant (Wald $\chi ^ { 2 } = 9 . 9 7 0 , \mathrm { d f } { = } 9 , p = 0 . 3 5 2 9 0 0 )$ ; therefore, the data do not support a compounded Fitzpatrick-bylighting efect on endpoint discrepancy.

The same interaction structure was not detected for $D _ { \lambda }$ . The Fitzpatrick×motion block yielded Wald $\chi ^ { 2 } = 9 . 4 6 8 ~ ( \mathrm { d f = 1 2 } , p = 0 . 6 6 2 5 2 1 )$ , and the Fitzpatrick×lighting block yielded Wald $\chi ^ { 2 } = 8 . 5 3 4$ $( \mathrm { d f } { = } 9 , p = 0 . 4 8 1 3 7 0 )$ . Thus, the significant endpoint interaction did not generalize to the tested dynamical discrepancy, providing a second level of endpoint–dynamical dissociation beyond the additive contrasts. The four joint interaction tests are summarized in Table 8.

![](images/458f5cc13f9bbdc353e8e8e5496ac4fec52bc6a1f004317fb6d16fc57133047b.jpg)

![](images/c364bf3e98f8e3a4741b99e71c866e7c02f9f1d8490c38b7ba70120902c2b488.jpg)

C. Endpoint discrepancy by motion/activity  
![](images/62cafa17a2da43cea7e10e49cd4aeadc626ec2e2fb1b4ff2318c16ed1ef44199.jpg)

D. Dynamical discrepancy by motion/activity  
![](images/bd0d554424f307855143a86b3e2573d290114a900428197174d8532cd7acd948.jpg)  
Figure 3: Acquisition-condition dependence of endpoint and dynamical recoverability. Mean $D _ { H R }$ and mean $D _ { \lambda }$ are shown across lighting conditions (A–B) and motion/activity conditions (C–D). The two discrepancy measures do not vary in parallel across acquisition conditions, motivating property-specific rather than single-metric characterization of PPG-to-rPPG recoverability. Bars summarize observed condition-specific means from the analysis. D<sub>HR</sub> is expressed in beats/min, whereas $D _ { \lambda }$ is dimensionless under the estimator used; their numerical magnitudes are therefore not directly comparable across axes. The panels are intended to compare patterns across conditions, are descriptive, and should not be interpreted as causal efects.

Table 8 | Joint Wald tests of prespecified Fitzpatrick-by-observation-condition interaction blocks.  
A significant block test indicates heterogeneity of the condition efect across Fitzpatrick groups; nonsignificant blocks are not interpreted from individual cell means alone.
<table><tr><td>Outcome and interaction block</td><td>df</td><td>Wald  $\chi ^ { 2 }$ </td></tr><tr><td> $D _ { H R } \colon \mathrm { F i t z p a t r i c k \times M o t i o n }$ </td><td>12</td><td>28.789</td></tr><tr><td> $D _ { H R } \colon \mathrm { F i t z p a t r i c k } \times \mathrm { L i g h t i n g }$ </td><td>9</td><td>9.970</td></tr><tr><td> $D _ { \lambda } \colon$   $\mathrm { F i t z p a t r i c k \times M o t i o n }$ </td><td>12</td><td></td></tr><tr><td> $D _ { \lambda } \colon \mathrm { F i t z p a t r i c k } \times \mathrm { L i g h t i n g }$ </td><td>9</td><td>8.534</td></tr></table>

The resulting adjusted profiles are shown in Figure 4. The nonparallel motion profiles visualize the significant interaction, with the largest separation occurring in the post-exercise stationary condition.

![](images/154722a3c51c5aad9fa4c040e4109d44d473f2c7dc8fb8dd441c0cad89f88f2d.jpg)  
Figure 4: Adjusted Fitzpatrick-by-motion heterogeneity in endpoint recoverability. Population-level adjusted mean absolute $P P G - r P P G$ HR discrepancy $( D _ { H R } )$ from the subject-aware Fitzpatrick×motion model is shown across motion/activity conditions. The nonparallel profiles visualize the significant joint interaction block (Wald $\chi ^ { 2 } ( 1 2 ) = 2 8 . 7 8 9 , p = 0 . 0 0 4 2 3 4 )$ . The largest separation is observed in the post-exercise stationary condition, where adjusted discrepancy rises sharply for Fitzpatrick IV–VI relative to III. The figure visualizes population-level model-based adjusted means from the repeated-measures model; inferential interpretation is based on the joint 12-df interaction block test rather than on isolated plotted cell diferences. A significant block indicates that the motion/activity pattern difers across Fitzpatrick groups, but does not imply that every motion contrast difers for every group.

The significant interaction is therefore not a generic claim that every adverse condition amplifies a Fitzpatrick-associated gap. The corresponding Fitzpatrick×lighting block was nonsignificant despite visible variation among adjusted cell means (Supplementary Figure S6). This contrast is important because it separates an interaction supported by the repeated-measures model from visually suggestive but statistically unsupported subgroup patterns.

## 6.4.5 Aggregate RGB SNR does not materially account for the endpoint gradient

Measured channel SNR was low across the dataset and varied only modestly by Fitzpatrick group. As summarized in Table 9, group-mean values for the red, green, and blue channels remained near -15 to -16 dB, with approximately 0.67 dB of aggregate variation across Fitzpatrick groups. The channel-specific distributions underlying these summaries are shown in Supplementary Figure S2. This small SNR diference contrasted with the 9.21 beats/min increase in mean DHR from Fitzpatrick III to VI. Across the 625-record Fitzpatrick analysis set, recording-level aggregate RGB SNR was essentially uncorrelated with absolute HR discrepancy (Pearson r=0.014, p=0.718; Figure 5).

Table 9 | Measured recording-level RGB SNR by Fitzpatrick group (dB).  
Recording-level red-, green-, and blue-channel SNR values and their aggregate mean are summarized by Fitzpatrick group for the 625-record analysis set. These values are descriptive signal-quality measures and are not interpreted as direct estimates of melanin concentration or tissue optical absorption.
<table><tr><td>Fitzpatrick</td><td>N</td><td>SNR_R</td><td>SNR_G</td><td>SNR_B</td></tr><tr><td></td><td>338</td><td>-15.55</td><td>-15.51</td><td>-15.22</td></tr><tr><td>III IV</td><td>77</td><td>-15.96</td><td>-15.85</td><td>-15.64</td></tr><tr><td>V</td><td>120</td><td>-16.52</td><td>-16.02</td><td>-15.76</td></tr><tr><td>VI</td><td>120</td><td>-15.97</td><td>-16.02</td><td>-16.26</td></tr></table>

## 6.4.6 A simple Beer–Lambert-inspired attenuation adequacy check does not reproduce measured SNR

We next evaluated a deliberately simplified Beer-Lambert-inspired attenuation model, motivated by the wavelength-dependent absorption and scattering of biological tissue and melanin [4], SNRdB = baseline SNR - 20 × ODmelanin, as an adequacy check rather than as a complete optical model. Table 10 reports the observed-versus-predicted comparison. The predicted SNR decreased sharply from Fitzpatrick III to VI, whereas the measured pattern was nearly flat; the resulting RMSE was 6.47 dB. The tested attenuation model therefore did not reproduce the observed SNR pattern and was not used for correction or causal inference. Figure 6A shows the predicted-versus-observed mismatch, while Figure 6B shows the group-level residual pattern.

Measured RGB SNR versus endpoint discrepancy  
![](images/4a4ed75b475d3bc9a7360716f1015300231848af121f5476d8c67f59c1f1f0a7.jpg)  
Figure 5: Measured aggregate RGB SNR versus endpoint discrepancy. Recording-level aggregate RGB SNR is plotted against absolute heart-rate discrepancy $D _ { H R }$ for the 625-record Fitzpatrick analysis set. The linear association is negligible (Pearson $r ~ = ~ 0 . 0 1 4$ $p = 0 . 7 1 8 )$ indicating that the measured aggregate SNR metric does not materially account for recording-level variation in endpoint discrepancy. Points are individual recordings and are identified by Fitzpatrick group for visualization. Aggregate RGB SNR is the mean of the recording-level red-, green-, and blue-channel SNR measures used in the sensitivity analysis. The near-zero correlation indicates that this measured aggregate SNR metric has little linear association with $D _ { H R }$ in these data; it is not interpreted as a causal test of optical mechanisms or as evidence that optical factors are unimportant.

Table 10 | Observed versus predicted SNR for the simplified Beer–Lambert-inspired adequacy check.  
Observed group-level aggregate RGB SNR is compared with the monotonic attenuation pattern generated by the simplified Beer–Lambert-inspired relation. Residuals quantify observed-minuspredicted SNR and are used only to assess adequacy of the specified pattern, not to estimate melanin or establish a causal optical mechanism.
<table><tr><td>Fitzpatrick</td><td>Observed SNR</td><td>Predicted SNR</td><td>Prediction error</td></tr><tr><td></td><td>-15.43</td><td>-10.48</td><td>-4.94</td></tr><tr><td>III IV</td><td>-15.82</td><td>-14.48</td><td>-1.33</td></tr><tr><td>V</td><td>-16.10</td><td>-19.48</td><td>+3.39</td></tr><tr><td>VI</td><td>-16.08</td><td>-27.48</td><td>+11.40</td></tr></table>

This negative result does not identify the source of the endpoint disparity. Optical acquisition, camera response and compression, ROI selection, color-channel interactions, rPPG extraction, and interactions among these factors remain plausible contributors.

![](images/ead4d772738b550ec360868d50bdd3f29b43852827deff4ad00f372d834f4ad7.jpg)

![](images/3711e776c3fd43f7eb05a25d05c7879a14dc2d84347a339240599251b864e58a.jpg)  
Figure 6: Predicted versus observed SNR and model residuals across Fitzpatrick groups. (A) Observed aggregate RGB SNR is compared with the monotonic pattern predicted by the deliberately simplified Beer–Lambert-inspired relation $\mathrm { S N R } _ { d B } \ =$ baseline $\mathrm { S N R } - 2 0 { \cal O } D _ { m e l a n i n }$ . The model RMSE is 6.47 dB. (B) Group-level prediction residuals show increasing positive error toward the darker-skin groups, with the largest discrepancy for Fitzpatrick VI. Residuals are observed minus predicted SNR, so increasingly positive residuals indicate that the simple model predicts substantially more attenuation than was measured. This adequacy check shows that the specified one-parameter monotonic attenuation pattern does not reproduce the group-level SNR observations. The model is not a physiological melanin estimator, does not represent the full camera/illumination/tissue imaging chain, and is not used to correct rPPG signals or establish causality.

## 6.4.7 Measured-SNR adjustment leaves the Fitzpatrick coeficient unchanged

The SNR-adjusted sensitivity model (Model 1b) added a single aggregate covariate, $\mathrm { S N R } _ { \mathrm { R G B } } =$ $( \mathrm { S N R } _ { R } + \mathrm { S N R } _ { G } + \mathrm { S N R } _ { B } ) / 3$ , to Model 1. The final coeficients are reported in Table 11. SNR<sub>RGB</sub> was not associated with $D _ { H R }$ (0.0769 beats/min/dB, 95% CI -0.2283 to 0.3821; p=0.621). More importantly, the Fitzpatrick VI-versus-III coeficient changed only from +9.3218 to +9.3678 beats/min, corresponding to -0.49% descriptive coeficient attenuation. Adjustment for measured aggregate RGB SNR therefore produced essentially no attenuation of the Fitzpatrick-associated endpoint discrepancy. The near-invariance of the Fitzpatrick coeficients before and after SNR adjustment is also shown in Supplementary Figure S3.

## Table 11 | SNR-adjusted sensitivity model.

Model 1b adds recording-level aggregate RGB SNR to the subject-aware endpoint-discrepancy model while retaining Fitzpatrick group, lighting, motion/activity, and the subject random intercept. The table evaluates whether SNR adjustment materially changes the Fitzpatrick-associated coeficients.
<table><tr><td rowspan="2">Contrast</td><td rowspan="2">Model 1</td><td colspan="2">Model 1b +</td><td rowspan="2">Model 1b p</td></tr><tr><td>SNR_RGB</td><td>Model 1b 95% CI</td></tr><tr><td>IV vs III</td><td>6.37</td><td>6.40</td><td>[-0.08, 12.89]</td><td>0.053</td></tr><tr><td>V vs III</td><td>6.72</td><td>6.77</td><td>[1.25, 12.30]</td><td>0.016</td></tr><tr><td>VI vs III</td><td>9.32</td><td>9.37</td><td>[3.83, 14.90]</td><td>0.001</td></tr><tr><td>SNR_RGB</td><td></td><td>0.077 bpm/dB</td><td>[-0.228, 0.382]</td><td>0.621</td></tr></table>

The condition-level endpoint–dynamical dissociation is summarized graphically in Figure 7; notably, walking combines high mean endpoint discrepancy with the lowest mean dynamical discrepancy among the displayed motion/activity conditions.

## 6.5 Observation context partially reduces held-out endpoint discrepancy

The subject-held-out experiment tested the extent to which measured observation state reduced the PPG–rPPG endpoint discrepancy, with MAE as the primary performance measure. On the 625-record condition-aware analysis subset, raw camera-derived rPPG HR had MAE 15.24 bpm, compared with 15.26 bpm in the full 655-record fixed-pathway benchmark. A linear learner using rPPG HR alone $( M _ { 0 } )$ reduced MAE to 12.54 bpm. Thus, $M _ { 0 }$ is a learned calibration baseline; comparisons with $M _ { 1 } { - } M _ { 4 }$ quantify the incremental MAE reduction attributable to observation context beyond that calibration rather than comparing contextual models directly with the raw 15.24-bpm error. Adding motion (M<sub>1</sub>) reduced MAE to 11.15 bpm, and adding lighting (M<sub>2</sub>) reduced MAE further to 10.87 bpm.

The same incremental pattern was observed across all three learner families. $M _ { 2 }$ produced the lowest MAE for linear regression (10.87 bpm), ridge regression (10.87 bpm), and random forest (11.23 bpm), corresponding to reductions of 13.32%, 13.32%, and 12.77% relative to their respective $M _ { 0 }$ baselines. The subject-bootstrap 95% CIs for the $M _ { 2 } - M _ { 0 }$ MAE diference were entirely below zero for linear regression (−2.36 to −0.98 bpm), ridge regression (−2.35 to −0.99 bpm), and random forest (−2.38 to −0.93 bpm). Adding Fitzpatrick group $( M _ { 3 } )$ or aggregate RGB SNR $( M _ { 4 } )$ did not further reduce the observed MAE in any learner family. Thus, the predictive contribution of motion and lighting was consistent across linear, regularized-linear, and nonlinear learners rather than being specific to a single model class.

Table 12 summarizes the subject-held-out MAE across the five nested information sets and all three learner families.

Adding Fitzpatrick group did not further improve held-out performance: linear-model MAE increased from 10.87 bpm in $M _ { 2 }$ to 11.16 bpm in $M _ { 3 }$ , and the bootstrap CI for improvement relative to $M _ { 0 }$ crossed zero at its upper bound. Adding aggregate RGB SNR increased MAE further to 11.23 bpm in $M _ { 4 } ,$ again with a bootstrap interval crossing zero. The same ordering appeared in ridge and random-forest models. These results distinguish factors associated with heterogeneity from factors that improve prediction: motion and lighting provided useful observation context, whereas Fitzpatrick category and aggregate RGB SNR did not add generalizable predictive benefit after motion and lighting were already included.

![](images/9ec03201a3e5002273a84740f34aed7e572cf548dc516d5087fabab79c2388c4.jpg)  
Figure 7: Motion/activity reveals endpoint–dynamical dissociation. Each point represents the condition-specific mean endpoint discrepancy D<sub>HR</sub> and mean dynamical discrepancy $D _ { \lambda }$ Conditions with similar or larger endpoint discrepancy do not necessarily exhibit correspondingly larger $D _ { \lambda } ,$ for example, walking has high mean endpoint discrepancy but the lowest mean dynamical discrepancy among the displayed conditions. Because $D _ { H R }$ and $D _ { \lambda }$ have diferent units and scales, the figure evaluates condition ordering and co-variation rather than equality of numerical magnitude. The plot is descriptive and illustrates that acquisition conditions can afect endpoint and dynamical recoverability diferently.

Table 12: Subject-held-out endpoint prediction across learner families. Out-of-fold MAE (bpm) is reported for each nested information set and learner family. Percentages in parentheses denote MAE reduction relative to the corresponding $M _ { 0 }$ baseline within the same learner family. $M _ { 0 }$ rPPG HR only; $M _ { 1 }$ : +motion; $M _ { 2 } .$ : +lighting; $M _ { 3 } \colon$ +Fitzpatrick group; $M _ { 4 } .$ +aggregate RGB SNR.
<table><tr><td>Information set</td><td>Linear</td><td>Ridge</td><td>Random forest</td></tr><tr><td> $M _ { 0 }$ </td><td>12.54 (0.00%)</td><td>12.54 (0.00%)</td><td>12.88 (0.00%)</td></tr><tr><td> $M _ { 1 }$ </td><td>11.15 (11.14%)</td><td>11.15 (11.13%)</td><td>11.48 (10.87%)</td></tr><tr><td> $M _ { 2 }$ </td><td>10.87 (13.32%)</td><td>10.87 (13.32%)</td><td>11.23 (12.77%)</td></tr><tr><td> $M _ { 3 }$ </td><td>11.16 (11.02%)</td><td>11.15 (11.15%)</td><td>11.65 (9.53%)</td></tr><tr><td> $M _ { 4 }$ </td><td>11.23 (10.45%)</td><td>11.22 (10.58%)</td><td>11.89 (7.69%)</td></tr></table>

Does observation context reduce the PPG-rPPG HR gap?  
![](images/5f8719e8423568de711988db8504ec09cdacb26430d5c8fefa52875017157cbd.jpg)  
Figure 8: Subject-held-out reduction of the PPG–rPPG HR gap using measured observation context. Out-of-fold MAE is shown for linear, ridge, and random-forest learners as predictors are added incrementally: rPPG HR only $( M _ { 0 } )$ , +motion $( M _ { 1 } )$ , +lighting (M<sub>2</sub>), +Fitzpatrick group $( M _ { 3 } )$ , and +aggregate RGB SNR $( M _ { 4 } )$ . Motion and lighting consistently reduce MAE, with the best result at $M _ { 2 } ;$ adding Fitzpatrick group or aggregate SNR does not improve the best model. The analysis uses the 625-record, 32-subject condition-aware subset; its raw camera-derived rPPG HR MAE is 15.24 bpm, compared with 15.26 bpm in the full 655-record benchmark. Five-fold GroupKFold validation is subject-held-out, so recordings from a participant do not appear in both training and test folds. For the linear learner, MAE decreases from 12.54 bpm at M<sub>0</sub> to 10.87 bpm at M<sub>2</sub> (13.32% reduction; subject-bootstrap 95% CI for the MAE diference, -2.36 to -0.98 bpm). The remaining error is substantial and does not establish equivalence between rPPG and contact PPG.

The corresponding predicted-versus-reference plot (Supplementary Figure S5) shows substantial regression toward the central HR range despite the MAE improvement. Accordingly, this experiment supports only partial narrowing of the endpoint gap. It does not show that camera-derived rPPG can be used interchangeably with contact PPG or that observation context reconstructs the source waveform.

## 7 Discussion

## 7.1 Property-specific preservation across an observation pathway

The central signal-processing finding is that PPG-to-rPPG conversion does not have a single level of recoverability. Under the same fixed CHROM-based observation pathway, endpoint HR, conventional temporal and spectral structure, and maximal-Lyapunov dynamics exhibited diferent degrees of recording-specific correspondence. Temporal autocorrelation retained a modest matched pair advantage, whereas spectral and recurrence-rate measures showed little separation from shufled controls, and recording-specific $\lambda _ { \mathrm { m a x } }$ correspondence was not detectable. Recoverability should therefore be assigned to a defined physiological property under a defined observation pathway rather than globally to the rPPG modality. Equivalently, the relevant object is the recoverability of a defined property $\phi _ { j }$ under an observation transformation $\mathcal { H } _ { \theta }$ , rather than recoverability of the derived modality as a whole.

These findings refine our earlier work on skin-tone-robust topological signal processing [18]. Robustness of a topological descriptor across population or observation groups does not by itself establish that the corresponding property is preserved recording by recording across heterogeneous sensing pathways.

The nonlinear analysis provides the clearest counterexample to inference from population-level plausibility. PPG and rPPG $\lambda _ { \mathrm { m a x } }$ estimates occupied overlapping ranges, yet matched-recording correspondence was essentially indistinguishable from the recording-identity null and rPPG variability was compressed. Thus, a descriptor can appear physiologically plausible at the population level without preserving which recording had which value. This result is consistent with state-space and Lyapunov methods, for which the estimated dynamical quantity is tied to the reconstructed trajectory rather than to marginal population overlap alone [7, 8]. In this manuscript, failure of matched correspondence is interpreted as evidence against preservation of the tested property; it is not a formal information-theoretic estimate of information loss.

The conventional structural controls reinforce the same distinction. ACF similarity retained modest recording specificity, whereas PSD and recurrence-rate diferences provided little matched-versusshufled separation and determinism was non-discriminative under the implemented parameterization. The correspondence-breaking control is therefore important for distinguishing recording-specific preservation from structure that can also arise in nonmatched signals. This use of matched identity is an analysis design choice of the present study rather than an established rPPG benchmark convention.

## 7.2 Observation-associated heterogeneity is also property-specific

Observation conditions did not afect all recovered properties in parallel. Endpoint discrepancy varied substantially with motion/activity and Fitzpatrick group after simultaneous subject-aware adjustment, whereas the tested dynamical discrepancy showed a diferent pattern. The Fitzpatrickby-motion block provided secondary evidence that endpoint heterogeneity may depend jointly on subject-associated and acquisition conditions, but the smaller Fitzpatrick strata contain few independent participants; this interaction therefore motivates replication rather than a definitive characterization of subgroup-specific motion efects. The corresponding Fitzpatrick-by-lighting interaction was not supported, and neither interaction was supported for $D _ { \lambda }$

The Fitzpatrick-associated endpoint contrast persisted after adjustment for the measured acquisition covariates, and adding aggregate RGB SNR did not materially account for that heterogeneity. This does not establish that optical signal quality is irrelevant. Rather, the simple pathway

$$
\mathrm { F i t z p a t r i c k ~ g r o u p }  \mathrm { a g g r e g a t e ~ R G B ~ S N R }  D _ { H R }
$$

was not supported by the specified analyses. Aggregate RGB SNR may not capture wavelengthdependent tissue optics, camera response, spatial ROI efects, channel mixing, compression, or algorithm-by-acquisition interactions that contribute to rPPG formation [4, 12, 3].

The optical-attenuation adequacy check narrows the mechanism further without identifying it. The simple Beer–Lambert-inspired model predicted a substantially stronger Fitzpatrick-dependent SNR decline than was measured. The appropriate conclusion is therefore that this simplified monotonic attenuation model is inadequate for reproducing the observed SNR pattern, not that Beer–Lambert physics or tissue optics are unimportant. The end-to-end camera pathway contains coupled optical, camera, behavioral, spatial-sampling, and algorithmic transformations that were not separately manipulated here.

The pattern for $D _ { \lambda }$ further illustrates why heterogeneity should not be generalized from one property to another. Fitzpatrick- and lighting-associated diferences were not detected for this discrepancy, while several motion/activity contrasts difered from the reference condition. Smaller $D _ { \lambda }$ values under particular conditions cannot be interpreted as improved dynamical preservation because recording-specific $\lambda _ { \mathrm { m a x } }$ correspondence was absent overall. The relevant result is the dissociation: factors associated with endpoint discrepancy do not map monotonically onto the tested dynamica discrepancy. A condition associated with degradation of one recovered property should therefore not automatically be assumed to degrade another property in the same way.

## 7.3 Diagnostic association does not imply predictive actionability

The condition-aware experiment asks a diferent question from the mixed-efects analysis: whether measured observation context can improve prediction of contact-PPG HR for unseen subjects. The rPPG-HR-only model $( M _ { 0 } )$ serves as a learned calibration baseline, so subsequent improvements quantify the incremental value of context beyond systematic scale and ofset correction. Motion and lighting reduced held-out error beyond this baseline, whereas adding Fitzpatrick group or aggregate RGB SNR did not improve the best model. Ridge and random forest reproduced the same ordering and did not outperform the linear specification.

This separation between inferential association and predictive utility is important. A variable can identify where discrepancy difers without providing incremental predictive information once other observation context is available. Conversely, motion and lighting were useful for partial endpoint-gap reduction even though the remaining error and compressed prediction range were substantial. The experiment therefore supports condition-aware gap reduction, not reconstruction of the source PPG waveform, physiological correction, or interchangeability of camera-derived rPPG with contact PPG.

## 7.4 Scope and relation to prior representation frameworks

The weak CHROM endpoint benchmark does not establish an intrinsic limit of camera-based physiological sensing. CHROM was used as an established, unsupervised observation pathway whose benchmark behavior on MMPD could be reproduced without selecting an rPPG method according to downstream results. The conclusions are therefore operator-specific: they characterize which tested properties remain accessible under this CHROM-based transformation, not what is fundamentally unrecoverable by every possible camera-rPPG method.

The present findings also provide an observation-level extension of the representation question introduced in the prior CFD and CST work discussed in Section 3. CFD motivates the broader premise that physiological information available for inference can depend on representation, while CST emphasizes temporal and dynamical organization in contact PPG. The present study asks the next observation-level question: when the sensing pathway changes from contact PPG to camera-derived rPPG, which tested properties retain recording-specific correspondence? This relationship is conceptual rather than assumptive; the current framework and empirical conclusions stand independently of CFD or CST [16, 17].

Taken together, the results show that endpoint discrepancy, conventional structural correspondence, maximal-Lyapunov correspondence, measured RGB SNR, and acquisition-condition efects do not collapse to a single recoverability axis. Camera-derived physiological validation should therefore specify the property claimed to be recovered, test recording-specific correspondence for that property when appropriate, and evaluate its stability across relevant observation conditions.

## 8 Limitations

Several limitations define the scope of these findings. First, all analyses used MMPD. Although the dataset contains hundreds of recordings spanning multiple acquisition conditions, the number of independent participants is much smaller, particularly within some Fitzpatrick strata. Subject-aware mixed-efects models, grouped cross-validation, and subject-level bootstrap resampling address repeated measurements but do not substitute for independent external validation. The Fitzpatrickby-motion finding should therefore be treated as replication-worthy evidence rather than a definitive characterization of subgroup-specific motion efects.

Second, the study intentionally used one fixed, established CHROM-based camera-rPPG observation pathway. The observed endpoint, structural, and dynamical correspondence therefore characterize this operator on MMPD and do not establish a fundamental limit of camera-based physiological sensing. Alternative rPPG extraction methods, acquisition systems, or learned observation models may preserve diferent properties and should be evaluated using the same recording-specific framework.

Third, the tested properties–HR, autocorrelation, spectral structure, recurrence measures, and maximal Lyapunov exponent–are representative rather than exhaustive. The observed behavior of these properties does not determine the behavior of every physiological property that could be extracted from PPG or rPPG. Moreover, each property is observed through a finite-sample estimator. In particular, $\lambda _ { \mathrm { m a x } }$ depends on preprocessing, state-space reconstruction, validity criteria, and estimator behavior [7, 8]. The null $\lambda _ { \mathrm { m a x } }$ correspondence result therefore applies to the implemented dynamical representation and should not be generalized to all nonlinear cardiovascular structure.

Fourth, the observation-condition analyses are associational rather than causal. Fitzpatrick category is not a direct measure of melanin concentration or tissue optical properties [13], and aggregate RGB SNR is not a complete description of camera-domain information quality [15]. Likewise, the Beer–Lambert-inspired analysis was a deliberately simplified adequacy check rather than a full physical model of camera-rPPG formation. The persistence of Fitzpatrick-associated endpoint heterogeneity after RGB-SNR adjustment therefore neither identifies a causal mechanism nor establishes that optical signal quality is irrelevant. Direct optical measurements, controlled acquisition, and prospective designs that separate subject-associated from acquisition factors are needed for mechanistic attribution.

Finally, the condition-aware experiment was designed primarily to test whether measured observation context reduces subject-held-out MAE in the PPG–rPPG endpoint discrepancy, rather than to maximize explained variance. Motion and lighting produced consistent MAE reductions across all three learner families, with $M _ { 2 }$ reducing MAE by 13.32%, 13.32%, and 12.77% relative to $M _ { 0 }$ for linear regression, ridge regression, and random forest, respectively. However, the maximum out-of-fold $R ^ { 2 }$ was 0.189, indicating that the measured observation descriptors account for only a limited portion of recording-level variation. Thus, the RQ5 findings support partial reduction of endpoint error as measured by MAE, rather than complete explanation or correction of the PPG–rPPG discrepancy. The experiment also predicts contact-PPG HR rather than reconstructing the contact-PPG waveform and therefore does not establish physiological waveform recovery or interchangeability of camera-derived rPPG with contact PPG. Future validation should extend the framework across independent datasets, multiple rPPG operators and sensing systems, direct optical measurements, and additional physiological property operators.

## 9 Conclusion

This study supports a property-specific framework for evaluating physiological measurements across heterogeneous observation pathways. Under the evaluated contact-PPG-to-camera-rPPG transformation, population-level plausibility did not guarantee recording-specific preservation, and observation-associated degradation was neither uniform across properties nor completely captured by simple signal-quality descriptors. These findings characterize the evaluated observation pathway rather than a fundamental limit of camera sensing.

More broadly, cross-modal physiological validation should identify the property claimed to be recovered and establish its recording-level preservation and stability under relevant observation conditions rather than infer signal equivalence from endpoint performance alone.

## 10 Data availability

The source MMPD dataset is available from the dataset authors under its stated access conditions. Derived matched-analysis tables and exclusion/accounting records will be made available with the accepted article subject to the original dataset terms.

## 11 Code availability

The analysis code used in this study will be made publicly available in a public repository upon publication of the article.

## Supplementary figures

The supplementary analyses provide additional validation and sensitivity checks for the main empirical findings. Supplementary Figure S1 provides the coeficient-focused Fitzpatrick endpoint gradient; Supplementary Figure S2 shows RGB-channel SNR distributions by Fitzpatrick group; Supplementary Figure S3 shows the SNR-adjusted coeficient sensitivity analysis; Supplementary Figure S4 shows the PPG–rPPG maximal-Lyapunov correspondence analysis; Supplementary Figure S5 shows out-of-fold predictions from the best condition-aware learner; and Supplementary Figure S6 shows the nonsignificant Fitzpatrick-by-lighting adjusted profiles.

Supplementary Figure S1: Subject-aware Fitzpatrick-associated endpoint gradient  
![](images/87c701913e7a00a4ad13ceaf6e69a96b385d9ab44d90fc029d658c3fa86a0088.jpg)  
Figure S1: Subject-aware Fitzpatrick-associated endpoint gradient. Mixed-efects coeficients for Fitzpatrick IV, V, and VI relative to Fitzpatrick III after adjustment for lighting and motion/activity with a subject random intercept (N = 625 recordings, 32 subjects). Coeficients are expressed in beats/min and use Fitzpatrick III as the reference group; error bars denote 95% confidence intervals. This coeficient-focused display complements the combined descriptive-and-adjusted presentation in Figure 2.

Supplementary Figure S2: RGB-channel SNR distributions by Fitzpatrick group  
![](images/dd59a9005524c08632b6aed46cacc02d5b189f206702934491392fc12ca88274.jpg)  
Figure S2: RGB-channel SNR distributions by Fitzpatrick group. Recording-level red-, green-, and blue-channel SNR distributions are shown for the 625-record Fitzpatrick analysis set. The extensive overlap and comparatively small shifts in central tendency provide the distributional context for Table 9; these measurements are descriptive signal-quality summaries and are not direct measures of melanin concentration or tissue optical absorption.

Supplementary Figure S3: Sensitivity of Fitzpatrick coefficients to aggregate RGB-SNR adjustment  
![](images/067d64a48ddf1161b9268db68d633a4be34f7187eb9355baedb61ca2fc3b231d.jpg)  
Figure S3: Sensitivity of Fitzpatrick coeficients to aggregate RGB-SNR adjustment. Model 1b adds measured aggregate RGB SNR to Model 1 while retaining Fitzpatrick group, lighting, motion/activity, and the subject random intercept. The displayed Fitzpatrick contrasts change by less than 1% relative to Model 1, indicating negligible descriptive attenuation after SNR adjustment. This is a sensitivity analysis and is not interpreted as causal mediation.

Supplementary Figure S4: Frozen PPG-rPPG maximal-Lyapunov analysis  
![](images/a1b1e29300c67b86471085e13df8a16bb1edb579073e76ae60af3cc0ccf92641.jpg)

![](images/4278422bee3829574a510c152fcf37e6bbdc1e128d5992b8dc738ada761b4df1.jpg)

![](images/67b7d07ab16e67f1bcab5a7e6b0283fd5ee3db065b57808b94e147bf28434275.jpg)  
Figure S4: Frozen PPG–rPPG maximal-Lyapunov analysis. The primary matched analysis used $6 \% 5$ recordings with valid PPG and $r P P G \lambda _ { \operatorname* { m a x } }$ estimates. No detectable matched recordinglevel correspondence was observed $( r _ { \lambda } = 0 . 0 2 3 1$ , 95% ${ C I \ - 0 . 0 5 4 2 }$ to 0.1001; permutation $p =$ 0.5584), despite overlap in marginal values. The rPPG estimates also show marked variance compression relative to contact $P P G ,$ supporting the distinction between population-level similarity and preservation of recording-specific dynamical information.

Supplementary Figure S5: Predicted versus reference HR for the best condition-aware learner  
![](images/0acfc0832e3644c21af6480092ddba7f38d248d0589f60c0b6db3314a163c7be.jpg)  
Figure S5: Predicted versus reference HR for the best condition-aware learner. Outof-fold predictions from the linear $M _ { 2 }$ model (rPPG HR + motion + lighting) are plotted against synchronized contact-PPG reference HR. The identity line represents perfect recovery of contact-PPG HR. Although $M _ { 2 }$ reduced MAE relative to the rPPG-only learner, predictions remain compressed toward central HR ranges and visibly depart from the identity line at the extremes, illustrating why the result is interpreted as partial gap reduction rather than PPG–rPPG equivalence. Predictions are strictly out-of-fold from subject-held-out validation.

Supplementary Figure S6: Adjusted Fitzpatrick-by-lighting endpoint profiles  
![](images/214bbaba68a251aeac00e6aa39127757c15bcf2b6ba7057a14d339d06c36a1b6.jpg)  
Figure S6: Adjusted Fitzpatrick-by-lighting endpoint profiles. Model-based adjusted mean $D _ { H R }$ is shown across lighting conditions for Fitzpatrick III–VI. Despite visible diferences among cell means, the joint Fitzpatrick×lighting block was not significant (Wald $\chi ^ { 2 } ( 9 ) = 9 . 9 7 0 , p = 0 . 3 5 2 9 )$ therefore, individual plotted diferences are not interpreted as evidence of a compounded lighting-by-Fitzpatrick efect. The figure is retained to distinguish visually apparent subgroup variation from interaction heterogeneity supported by the repeated-measures model.

## 12 Declarations

## 12.1 Ethics statement

This study is a secondary analysis of the existing de-identified Multi-Domain Mobile Video Physiology Dataset (MMPD) [6]. No new participants were recruited and no new data were collected for the present study.

## References

[1] J. Allen, “Photoplethysmography and its application in clinical physiological measurement,” Physiological Measurement, vol. 28, no. 3, pp. R1–R39, 2007. DOI: https://doi.org/10.1088/ 0967-3334/28/3/R01.

[2] D. J. McDuf, J. R. Estepp, A. M. Piasecki, and E. B. Blackford, “A survey of remote optical photoplethysmographic imaging methods,” in 2015 37th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), pp. 6398–6404, 2015. DOI: https://doi.org/10.1109/EMBC.2015.7319857.

[3] W. Wang, A. C. den Brinker, S. Stuijk, and G. de Haan, “Algorithmic principles of remote PPG,” IEEE Transactions on Biomedical Engineering, vol. 64, no. 7, pp. 1479–1491, 2017. DOI: https://doi.org/10.1109/TBME.2016.2609282.

[4] S. L. Jacques, “Optical properties of biological tissues: A review,” Physics in Medicine and Biology, vol. 58, no. 11, pp. R37–R61, 2013. DOI: https://doi.org/10.1088/0031-9155/58/ 11/R37.

[5] A. Dasari, S. K. A. Prakash, L. A. Jeni, and C. S. Tucker, “Evaluation of biases in remote photoplethysmography methods,” npj Digital Medicine, vol. 4, p. 91, 2021. DOI: https: //doi.org/10.1038/s41746-021-00462-z.

[6] J. Tang, K. Chen, Y. Wang, Y. Shi, S. Patel, D. McDuf, and X. Liu, “MMPD: Multi-Domain Mobile Video Physiology Dataset,” in 2023 45th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), pp. 1–5, 2023. DOI: https: //doi.org/10.1109/EMBC40787.2023.10340857.

[7] F. Takens, “Detecting strange attractors in turbulence,” in Dynamical Systems and Turbulence, Warwick 1980, Springer, pp. 366–381, 1981. DOI: https://doi.org/10.1007/BFb0091924.

[8] M. T. Rosenstein, J. J. Collins, and C. J. De Luca, “A practical method for calculating largest Lyapunov exponents from small data sets,” Physica D: Nonlinear Phenomena, vol. 65, nos. 1–2, pp. 117–134, 1993. DOI: https://doi.org/10.1016/0167-2789(93)90009-P.

[9] L. He, K. S. Alam, J. Ma, R. Povinelli, and S. I. Ahamed, “Dynamics reconstruction of remote photoplethysmography,” in Pervasive Computing Technologies for Healthcare: 15th EAI International Conference, PervasiveHealth 2021, Virtual Event, December 6–8, 2021, Proceedings, Lecture Notes of the Institute for Computer Sciences, Social Informatics and Telecommunications Engineering, vol. 431, Springer, pp. 96–110, 2022. DOI: https://doi. org/10.1007/978-3-030-99194-4\_8.

[10] G. de Haan and V. Jeanne, “Robust pulse rate from chrominance-based rPPG,” IEEE Transactions on Biomedical Engineering, vol. 60, no. 10, pp. 2878–2886, 2013. DOI: https: //doi.org/10.1109/TBME.2013.2266196.

[11] X. Liu, G. Narayanswamy, A. Paruchuri, X. Zhang, J. Tang, Y. Zhang, R. Sengupta, S. Patel, Y. Wang, and D. McDuf, “rPPG-Toolbox: Deep remote PPG toolbox,” Advances in Neural Information Processing Systems, vol. 36, Datasets and Benchmarks Track, pp. 68485–68510, 2023. DOI: https://doi.org/10.52202/075280-2995.

[12] K. Setchfield, A. Gorman, A. H. R. W. Simpson, M. G. Somekh, and A. J. Wright, “Efect of skin color on optical properties and the implications for medical optical technologies: A review,” Journal of Biomedical Optics, vol. 29, no. 1, p. 010901, 2024. DOI: https://doi.org/ 10.1117/1.JBO.29.1.010901.

[13] V. R. Weir, K. Dempsey, J. W. Gichoya, V. Rotemberg, and A.-K. I. Wong, “A survey of skin tone assessment in prospective research,” npj Digital Medicine, vol. 7, p. 191, 2024. DOI: https://doi.org/10.1038/s41746-024-01176-8.

[14] H. Xiao, T. Liu, Y. Sun, Y. Li, S. Zhao, and A. Avolio, “Remote photoplethysmography for heart rate measurement: A review,” Biomedical Signal Processing and Control, vol. 88, Art. no. 105608, 2024. DOI: https://doi.org/10.1016/j.bspc.2023.105608.

[15] M. Elgendi, I. Martinelli, and C. Menon, “Optimal signal quality index for remote photoplethysmogram sensing,” npj Biosensing, vol. 1, Art. no. 5, 2024. DOI: https://doi.org/10.1038/ s44328-024-00002-1.

[16] T. Oladunni and A. Wong, “Rethinking Multimodality: Optimizing Multimodal Deep Learning for Biomedical Signal Classification,” IEEE Access, vol. 13, pp. 156436–156464, 2025. DOI: https://doi.org/10.1109/ACCESS.2025.3605315.

[17] T. Oladunni and F. G. Adewumi, “Cardiac Stability Theory: An Axiomatically Grounded Framework for Continuous Cardiac Health Monitoring via Smartphone Photoplethysmography,” arXiv:2604.23876, 2026. DOI: https://doi.org/10.48550/arXiv.2604.23876.

[18] T. Oladunni and F. G. Adewumi, “Skin-Tone-Robust Topological Signal Processing: A Framework for Bias-Reducing Optical Measurement Systems,” medRxiv, 2026, doi: 10.64898/2026.08.01.26359472.