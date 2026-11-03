# Kolmogorov-Arnold Networks vs. Multi-Layer Perceptrons: A Rigorous Statistical Benchmark on Clinical Tabular Data

Rafał Wodzinski´ <sup>a,b,∗</sup>, Paweł Szczerbiak<sup>a,c,∗</sup>

<sup>a</sup>Sano Centrefor Computational Medicine, Kraków, Poland aSano Centre for Computational Medicine, Kraków, Poland

<sup>b</sup>Department of Computer Science, Faculty of Computer Science and Mathematics, Cracow University of Technology, Kraków, Poland <sup>c</sup>Faculty of Computer Science, AGH University of Krakow, Kraków, Poland

## Abstract

Kolmogorov-Arnold Networks (KANs) have recently been proposed as a promising alternative to Multi-Layer Perceptrons (MLPs). However, empirical evaluations on structured medical datasets remain scarce, and commonly reported metrics like AU ROC can yield overly optimistic performance estimates on imbalanced clinical cohorts. In this study, we present a standardized benchmark comparing eleven architectures—nine KAN variants (FastKAN, WavKAN, ReLUKAN, ChebyKAN, TaylorKAN, GramKAN, JacobiKAN, LegendreKAN, and TabKAN), a tuned StandardMLP, and a Random Forest baseline—across seven heterogeneous medical tabular datasets spanning oncology, cardiology, nephrology, neurology, endocrinology, and obstetrics. Our experimental protocol employs a 3 × 5 Repeated Stratified K-Fold cross-validation scheme with nested Bayesian hyperparameter optimization and evaluates discrimination via Area Under the Precision-Recall Curve (AUPRC), calibration through the Brier Score, and computational eficiency. Formal equivalence testing is conducted using the Bayesian Correlated t-Test with a Region of Practical Equivalence (ROPE) at ±0 01. The results indicate that on datasets with severe class imbalance, dedicated KAN architectures (particularly TabKAN and ReLUKAN) can improve minority-class detection and achieve higher AUPRC than standard MLPs. Nevertheless, across pooled datasets, KAN variants demonstrate comparable overall predictive discrimination rather than a systematic advantage, while requiring 1 20× to 1 48× longer inference latency per sample and up to 12 5× more trainable parameters. Consequently, StandardMLP remains a strong baseline on the eficiency Pareto frontier, while KAN variants provide a viable alternative when precision on rare positive cases is the primary clinical objective. The code for reproducing the results is available at https://github.com/rafalwodzinski/kan\_mlp\_comparison.

Keywords: Kolmogorov-Arnold Networks, Multi-Layer Perceptrons, Medical Tabular Data, Bayesian ROPE

## 1. Introduction

Structured tabular data represents the primary modality in clinical medicine. Electronic health records, laboratory test panels, patient registries, and continuous physiological streams support diagnostic decisions, treatment selection, and patient triage across health systems. Machine learning models deployed in clinical decision support must satisfy three practical requirements simultaneously: high discriminative performance, reliable probability calibration, and low inference latency suitable for interactive workflows [1]. These requirements often conflict. For example, uncalibrated risk predictions can cause misleading probability estimates, potentially compromising patient safety in critical care environments [2]. Furthermore, real-world clinical datasets almost invariably exhibit severe class imbalance, where positive disease cases represent only a small minority of the cohort. In such settings, standard aggregate metrics such as the Area Under the Receiver Operating Characteristic Curve (AUROC) can provide an overly optimistic evaluation. Because AUROC incorporates the large pool of true negatives into the false positive rate, it can mask high false discovery rates. Consequently, the Area Under the Precision-Recall Curve (AUPRC) has emerged as an essential evaluation metric, as it concentrates directly on the retrieval and precision of the minority class [3].

Empirical studies consistently indicate that tree-based ensembles—primarily XGBoost [4] and Random Forests [5]—remain the dominant choice for tabular machine learning benchmarks [6, 7]. Several factors explain this phenomenon: tabular features typically possess independent semantic meanings, missing values are common, and target decision surfaces are often sharp and non-smooth. Traditional deep neural networks, by contrast, rely on inductive biases such as rotational invariance and spatial correlation, which can impair performance on heterogeneous tabular inputs. Nonetheless, Multi-Layer Perceptrons (MLPs) [8] remain the standard neural baseline for tabular data due to their simple design, fast inference, and direct compatibility with standard GPU and accelerated hardware kernels.

Recently, Liu et al. [9] introduced Kolmogorov-Arnold Networks (KANs) as an alternative neural framework based on the Kolmogorov-Arnold representation theorem [10, 11]. Instead of placing fixed activation functions at nodes, KANs position learnable univariate activation functions on the network edges. In the original implementation, these functions were parameterized via B-splines. While KANs demonstrated promising accuracy on symbolic regression and mathematical toy problems, they exhibited a substantial computational drawback: the original authors reported roughly a tenfold increase in training time relative to MLPs of comparable width [9], driven by the sequential evaluation of spline bases. To mitigate this computational cost, several alternative KAN variants have been proposed: FastKAN approximates splines via Gaussian Radial Basis Functions [12]; Wav-KAN adopts wavelet transforms [13]; ReLU-KAN reconstructs spline-like activation curves using native linear and ReLU operations [14]; ChebyKAN applies Chebyshev polynomial expansions [15]; TaylorKAN combines Taylor series expansions with dimensionality reduction [16]; and related studies explore orthogonal functional bases such as Jacobi [17, 18, 19], Legendre [20], and Gram [21] polynomials. More recently, TabKAN [22] was designed specifically for tabular domains by combining KAN layers with feature gating mechanisms.

Despite active methodological interest, existing evaluations of KAN architectures feature notable limitations. Most published studies evaluate models primarily on synthetic equations, vision benchmarks such as MNIST, or small nonmedical datasets. Few works examine KANs on heterogeneous medical tabular data with class imbalance, and existing reports rarely evaluate threshold-free metrics like AUPRC or quantify predictive calibration. Furthermore, published comparisons frequently report raw point estimates from single cross-validation runs without formal statistical equivalence testing. As a result, whether KANs ofer a meaningful advantage over tuned MLPs on clinical tabular tasks remains an open empirical question.

To address this gap, we present a systematic empirical benchmark comparing eleven model architectures (nine KAN variants, a StandardMLP baseline, and a Random Forest ensemble) across seven medical tabular datasets spanning diverse clinical domains. Our experimental design applies a 3×5 Repeated Stratified K-Fold cross-validation protocol with nested Bayesian hyperparameter optimization (via Optuna [23]) and strict fold-isolated preprocessing to prevent information leakage. We assess models across discrimination (AUPRC, AUROC, Matthews Correlation Coeficient, F1-Score), calibration (Brier Score), and computational cost (per-sample inference latency, training time, parameter count). In addition, we apply the Bayesian Correlated t-Test with Region of Practical Equivalence (ROPE) [24, 25] to test whether observed performance diferences represent practical equivalence or meaningful improvements.

The results of our benchmark provide two primary insights:

1. On datasets characterized by severe class imbalance (such as cervical cancer screening and cardiotocography), specific KAN variants—notably TabKAN and

ReLUKAN—achieve competitive or superior AUPRC and minority recall relative to the StandardMLP baseline.

2. Across the broader pooled benchmark, the discriminative performance and calibration of KAN variants and the StandardMLP are largely equivalent within a ±0 01 ROPE interval. However, all evaluated KAN variants incur a noticeable computational penalty: per-sample inference latency is 1 20× to 1 48× higher, and parameter counts are 2 8× to 12 5× larger than those of the StandardMLP.

These findings suggest that while standard MLPs remain the more computationally eficient choice for general clinical applications on the Pareto frontier, tailored KAN architectures warrant consideration when maximizing detection of rare clinical endpoints is of primary importance.

## 2. Related Work

## 2.1. Kolmogorov-Arnold Networks and Their Variants

Kolmogorov-Arnold Networks were introduced as a mathematically principled alternative to the Multi-Layer Perceptron, grounded in the Kolmogorov-Arnold Representation Theorem. In place of the fixed nodal activation functions and learned scalar weights of the standard MLP, KANs place learnable univariate functions—parametrised as B-splines— on the edges of the network graph, yielding a representation of the form $\begin{array} { r } { f ( { \bf x } ) = \sum _ { q } \phi _ { q } \left( \sum _ { p } \phi _ { q , p } ( x _ { p } ) \right) } \end{array}$ The original work demonstrated reduced parameter counts at comparable approximation error on symbolic regression and physics simulation benchmarks, and proposed interpretable Pareto frontier analyses suggesting favourable scaling relative to MLPs. However, the original authors themselves explicitly conceded that KANs are “currently 10× slower to train than MLPs for a given number of parameters” [9], attributing this bottleneck to the sequential nature of B-spline computations, which cannot exploit GPU tensor parallelism as eficiently as dense matrix multiplications. No evaluation on real-world structured tabular data was conducted.

The computational bottleneck of the original architecture immediately motivated a series of alternative formulations. FastKAN [12] demonstrated that third-order B-splines can be efectively approximated using Gaussian Radial Basis Functions, yielding a 3 33× forward-pass speedup over efficient\_kan (though the combined forward-backward speedup shrank to 1 25×), while noting in a footnote that MNIST is not an optimal setting for evaluating KAN capabilities. Wav-KAN [13] proposed wavelet transforms as edge activations for noise robustness, though its parameter complexity across an L-layer network is $O ( 3 N ^ { 2 } L )$ (accounting for scale, translation, and weight parameters) compared to O(N<sup>2</sup>L) for a standard MLP of equivalent width N. ReLU-KAN [14] reconstructed bell-shaped basis functions using matrix operations and squared ReLU units, achieving up to a 20× training speedup in four-layer networks over Bsplines on synthetic benchmark functions. ChebyKAN [15] replaced B-splines with Chebyshev polynomial bases, achieving 98% accuracy on MNIST with a compact parameterization, though without reporting comparative inference latency against MLPs. TaylorKAN [16] applied Taylor series expansions as edge activations, demonstrating competitive results on image sharpness evaluation when combined with principal component analysis.

Beyond these models, several orthogonal polynomial formulations have been introduced that replace the spline grid with a polynomial degree hyperparameter. JacobiKAN [18, 19] incorporates orthogonal Jacobi polynomial bases, a concept generalized by fKAN [17] with trainable fractional-order parameters for curvature adaptation. LegendreKAN [20] uses classical Legendre orthogonal polynomial recurrences to construct activation curves with theoretically supported convergence properties. Similarly, GramKAN [21] utilizes Gram polynomials—natively designed for discrete data—which are approximated here via continuous recurrence for tabular features. In an independent comparative analysis, Yu et al. [26] examined KANs versus MLPs across computer vision, natural language processing, and standard machine learning tasks, finding that KANs primarily excelled at symbolic mathematical representation while standard MLPs maintained superior accuracy and eficiency on broader classification tasks. To address the specific requirements of structured data, Eslamian et al. [22] proposed TabKAN, which integrates KAN layers with input feature gating to suppress uninformative tabular variables.

Collectively, the existing literature on KAN architectures presents four notable gaps: (i) evaluations rely predominantly on synthetic functions or image classification rather than structured tabular data; (ii) comparisons are rarely performed against properly tuned neural and tree baselines across multidataset cohorts; (iii) inference latency and parameter counts are seldom profiled simultaneously; and (iv) evaluations typically overlook class imbalance and clinical calibration metrics. Consequently, the practical performance of KANs on real-world medical tabular benchmarks warrants systematic empirical evaluation.

## 2.2. Deep Learning on Tabular Data

Whether deep neural networks can reliably outperform tree-based ensembles on tabular data remains a central question in applied machine learning. Gradient-boosted decision trees, particularly XGBoost [4], established a dominant baseline across industrial and scientific tabular benchmarks due to their native handling of sparse features, missing values, and uninformative variables. Shwartz-Ziv and Armon [6] conducted an extensive benchmark across 11 tabular datasets with standardized Bayesian hyperparameter optimization, showing that tree ensembles consistently outperformed deep architectures (such as TabNet and NODE). They also noted that deep models frequently underperformed on datasets beyond those highlighted in their original publications. Grinsztajn et al. [7] further examined this dynamic across 45 OpenML benchmarks, observing that deep models struggle with irregular decision boundaries, uninformative features, and non-rotationally invariant feature spaces. These challenges apply directly to KANs, whose univariate edge functions were originally conceived for smooth mathematical compositions. Within this landscape, the StandardMLP with Batch Normalization and Dropout [8] serves as the canonical neural baseline, while the Random Forest [5] provides an established non-neural reference.

## 2.3. Statistical Frameworks for Multi-Dataset Algorithm Comparison

Comparing machine learning models across multiple datasets requires statistical methodologies that account for cross-dataset variability and cross-validation dependencies. In a foundational methodological study, Demšar [27] highlighted that standard parametric tests (such as paired t-tests and ANOVA) often violate distributional assumptions when applied to cross-validated classifier scores, leading to inflated Type I errors. As a robust non-parametric alternative, Demšar recommended the Wilcoxon signed-rank test for pairwise comparisons and the Friedman test with post-hoc rank corrections for multi-model comparisons. However, frequentist tests have an important epistemological limitation: failing to reject the null hypothesis $( p \ge 0 . 0 5 )$ indicates only an absence of evidence against equality, not afirmative evidence for practical equivalence.

To overcome this limitation, Benavoli et al. [24] proposed a Bayesian testing framework incorporating the Bayesian Correlated t-Test and a Region of Practical Equivalence (ROPE). By defining a clinically negligible diference interval $[ - \delta , + \delta ]$ and integrating the posterior probability density within this region, the Bayesian ROPE approach explicitly quantifies the probability of practical equivalence (P(ROPE)). To account for the correlation between folds in cross-validation resulting from overlapping training splits, Corani and Benavoli [25] derived an analytical variance adjustment based on the test fraction $\rho = 1 / k$ . This Bayesian framework provides the statistical foundation for our evaluation.

## 2.4. Predictive Uncertainty, Calibration, and Imbalance in Clinical AI

Deploying machine learning models in healthcare requires properties beyond raw overall accuracy. First, models must deliver calibrated probability estimates: predicted probabilities must correspond to observed event rates [2]. Poorly calibrated models that produce overconfident risk estimates can result in inappropriate clinical prioritization and patient harm. Moreover, probabilistic calibration should be routinely evaluated using proper scoring rules such as the Brier Score. Rajpurkar et al. [1] emphasized that model robustness, computational eficiency, and calibration are essential for safe clinical deployment.

Second, medical tabular datasets typically present pronounced class imbalance, with positive conditions (e.g., biopsy-confirmed malignancies, fetal distress) accounting for only a small fraction of cases. Standard receiver operating characteristic curves (AUROC) can provide an overly optimistic assessment in this setting because the large number of true negatives keeps the false positive rate low. By contrast, the Precision-Recall curve and its area (AUPRC) focus exclusively on the true positive class, making AUPRC particularly informative for evaluating detection quality in rare clinical conditions.

Table 1: Summary of Benchmark Medical Datasets
<table><tr><td>Dataset</td><td>N</td><td>d</td><td>Minority (%)</td><td>Task</td><td>Clinical Domain</td></tr><tr><td>Cervical Cancer [37]</td><td>858</td><td>32</td><td>6.4%</td><td>Binary</td><td>Gynaecology / Oncology</td></tr><tr><td>Cardiotocography [38]</td><td>2,126</td><td>21</td><td>8.3%</td><td>Multiclass</td><td>Obstetrics</td></tr><tr><td>Parkinson&#x27;s [36]</td><td>195</td><td>22</td><td>24.6%</td><td>Binary</td><td>Neurology</td></tr><tr><td>Pima Diabetes [33]</td><td>768</td><td>8</td><td>34.9%</td><td>Binary</td><td>Endocrinology</td></tr><tr><td>Breast Cancer [32]</td><td>569</td><td>30</td><td>37.3%</td><td>Binary</td><td>Oncology</td></tr><tr><td>Chronic Kidney [35]</td><td>400</td><td>24</td><td>37.5%</td><td>Binary</td><td>Nephrology</td></tr><tr><td>Heart Disease [34]</td><td>303</td><td>13</td><td>45.9%</td><td>Multiclass</td><td>Cardiology</td></tr></table>

Note: N – number of samples; d – number of features.

Finally, clinical deployment imposes strict latency and resource constraints. While techniques like Monte Carlo Dropout [28] and Deep Ensembles [29] can enhance uncertainty estimation, they increase computational latency linearly with the number of passes or models [30]. Because point-of-care decision support systems may operate on local hospital hardware or mobile workstations, computational cost (inference latency and memory footprint) represents an essential operational constraint.

Our study combines these complementary perspectives: evaluating KAN architectures and a tuned StandardMLP baseline on real-world medical tabular datasets using threshold-free AUPRC discrimination, Brier Score calibration, Bayesian ROPE equivalence testing, and comprehensive computational profiling.

## 3. Materials and Methods

## 3.1. Datasets and Preprocessing

Seven publicly available medical tabular datasets, collectively denoted $\mathcal { D } = \{ ( \mathbf { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N } ;$ were retrieved from the UCI Machine Learning Repository [31] and are summarized in Table 1. The benchmark spans seven distinct clinical domains: Breast Cancer Wisconsin [32] (N = 569, d = 30, binary); Pima Indians Diabetes [33] (N = 768, d = 8, binary); Cleveland Heart Disease [34] (N = 303, d = 13, multiclass); Chronic Kidney Disease [35] (N = 400, d = 24, binary); Parkinson’s Disease [36] (N = 195, d = 22, binary); Cervical Cancer [37] (N = 858, d = 32, binary); and Cardiotocography [38] (N = 2 126, d = 21, multiclass).

These datasets reflect realistic challenges encountered in clinical data science, including varying sample sizes $( N \ \in$ [195 2 126]), difering feature dimensions $( d ~ \in ~ [ 8 , 3 2 ] )$ missing laboratory readings, and pronounced class imbalance. In particular, the Cervical Cancer dataset exhibits an acute class imbalance with only 6 4% biopsy-positive cases (55 of 858), and the Cardiotocography dataset contains an 8<sub>.</sub>3% pathological minority class, presenting demanding test conditions for minority-class detection.

All preprocessing steps were implemented inside a strict pipeline to prevent data leakage. Estimators were fitted exclusively on the training split of each cross-validation fold and applied to the validation split via stateless transformations. For datasets with missing entries, continuous attributes were imputed using the column-wise median, except for Chronic Kidney Disease and Cervical Cancer, where K-Nearest Neighbors $\left( K = 5 \right)$ imputation was applied to handle structured missingness. Categorical features were imputed using the mode and one-hot encoded. Continuous variables were standardized to zero mean and unit variance.

## 3.2. Model Architectures

We evaluated eleven model architectures across the benchmark suite:

Neural Baseline. The StandardMLP baseline comprises a multi-layer feedforward network with Batch Normalization, Dropout regularisation $( p ~ = ~ 0 . 1 )$ , and Gaussian Error Linear Unit (GELU) activations [8]. The architecture follows a pyramidal hidden-layer structure [h<sub>,</sub> ⌊h/2⌋].

Tree-Based Baseline. The Random Forest classifier [5] provides an established non-neural reference for tabular data, implemented via Scikit-learn [39].

KAN Architectures. Nine Kolmogorov-Arnold Network variants were evaluated:

1. FastKAN [12]: Approximates third-order B-splines through Gaussian Radial Basis Functions to eliminate knot-search operations.

2. Wav-KAN [13]: Employs continuous/discrete wavelet transforms as edge activations for multiscale representation.

3. ReLU-KAN [14]: Replaces splines with combinations of matrix additions, dot products, and rectified linear units.

4. ChebyKAN [15]: Uses truncated Chebyshev polynomial bases of degree k.

5. TaylorKAN [16]: Utilizes low-order Taylor polynomial series expansions on network edges.

6. JacobiKAN [18]: Jacobi orthogonal polynomial bases of degree k with fixed shape parameters $\alpha = \beta = 1$ (inputs squashed to [−1 1] via tanh).

7. LegendreKAN [20]: Legendre orthogonal polynomial bases of degree k over [−1 1] (inputs squashed via tanh).

8. GramKAN [21]: Gram polynomial bases of degree k. Following common deep-learning implementations, we evaluate the basis through a continuous three-term recurrence on tanh-normalized inputs, which approximates the discrete Gram formulation for tabular features.

9. TabKAN [22]: Integrates KAN layers with a dedicated soft feature-gating mechanism that dynamically weights informative tabular features.

## 3.3. Training Protocol and Hyperparameter Optimisation

Evaluation was conducted using a 3×5 Repeated Stratified K-Fold Cross-Validation design, yielding 15 independent validation evaluations per model-dataset pair (105 evaluations per architecture across all seven datasets). Stratified partitioning ensured preserved class ratios across each fold.

To ensure fair comparison, all models underwent automated hyperparameter optimization (HPO) using Optuna [23] with the Tree-structured Parzen Estimator (TPE) algorithm. A budget of 15 trials was allocated per fold, with unpromising configurations pruned using Optuna’s MedianPruner. HPO was conducted inside an inner stratified 80/20 train-validation split of the training fold data. The inner validation loss guided parameter selection, ensuring no information from the outer test fold leaked into the tuning process.

The search space for neural models included the learning rate (LogUniform[ $1 0 ^ { - 4 } , 1 0 ^ { - 2 } ] )$ weight decay (LogUniform[ $1 0 ^ { - 6 } , 1 0 ^ { - 3 } ] )$ and hidden dimension $h \in \{ 3 2 , 6 4 , 1 2 8 \}$ . Architecture-specific parameters included grid\_size $\in \quad \{ 3 , 5 , 1 0 \}$ for spline models, polynomial degree $\in \enspace \{ 3 , 4 , 6 \}$ for polynomial KANs, and num\_grids $\in \{ 4 , 8 , 1 2 \}$ for FastKAN and TabKAN. For the Random Forest, estimators were sampled from [50 300], maximum depth from [3 15], and minimum split samples from [2 10].

Neural models were trained using AdamW with early stopping (patience of 10 epochs, minimum validation loss improvement $\delta ~ = ~ 1 0 ^ { - 4 } )$ . Binary tasks were optimized using binary cross-entropy with logits loss, and multiclass tasks using categorical cross-entropy.

## 3.4. Evaluation Metrics

Models were evaluated along two primary dimensions: predictive performance and computational eficiency.

## 3.4.1. Discriminative Performance and AUPRC

Given the prominent class imbalance across the benchmark datasets, the Area Under the Precision-Recall Curve (AUPRC) serves as our primary discriminative metric. In medical contexts with low disease prevalence, AUROC can yield misleadingly high scores because the large quantity of true negatives inflates the denominator of the false positive rate $( F P / ( F P + T N ) )$ [3]. In contrast, AUPRC focuses exclusively on the positive minority class, capturing the trade-of between positive predictive value (precision) and sensitivity (recall):

$$
{ \mathrm { A U P R C } } = \sum _ { n } ( R _ { n } - R _ { n - 1 } ) P _ { n } ,
$$

where $P _ { n }$ and $R _ { n }$ represent the precision and recall values at the n-th classification threshold. For multiclass tasks, macroaveraged One-vs-Rest (OvR) binarization was applied across all classes.

Supplementary discrimination metrics include the Matthews Correlation Coeficient (MCC) [40], AUROC, macro-averaged F1-Score, and Balanced Accuracy.

## 3.4.2. Calibration

Probabilistic calibration was assessed using the Brier Score [2], measuring the mean squared diference between predicted probabilities $\hat { p } _ { i }$ and observed binary outcomes $y _ { i } \in$ {0 1}:

$$
\mathrm { B S } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } { ( \hat { p } _ { i } - y _ { i } ) ^ { 2 } } .
$$

For multiclass datasets, the multi-class formulation based on one-vs-all decomposition was applied [41]. Lower Brier scores indicate better calibration.

## 3.4.3. Computational Eficiency

Three computational metrics were recorded: (i) per-sample inference latency $( t _ { \mathrm { i n f } }$ , in milliseconds), calculated by dividing the wall-clock evaluation time of the validation fold by the sample count; (ii) trainable parameter count (|θ|); and (iii) total training time $( T _ { \mathrm { t r a i n } } .$ , in seconds per fold).

## 3.5. Statistical Analysis Framework

To diferentiate genuine architectural diferences from sampling variance across datasets, we employed both frequentist and Bayesian statistical tests [27, 24].

## 3.5.1. Frequentist Testing

Dataset-level performance scores (averaged across the 15 validation folds per dataset) were compared pairwise against the StandardMLP baseline using the two-tailed nonparametric Wilcoxon signed-rank test [27]. To control the family-wise error rate across multiple architecture comparisons, the Holm-Bonferroni step-down correction was applied at $\alpha = 0 . 0 5$ . Omnibus multi-model comparisons across the benchmark were evaluated using the Friedman test on datasetlevel model ranks (1 = best to $k = 1 1 = \mathrm { w o r s t } )$ , followed by the Nemenyi post-hoc test. Cross-dataset average ranks and pairwise significance thresholds were visualized via Critical Diference (CD) diagrams, where models whose average rank diference is smaller than $\mathrm { C D } = q _ { \alpha } \sqrt { k ( k + 1 ) / ( 6 N ) }$ are connected by a horizontal clique bar [27].

## 3.5.2. Bayesian Correlated t-Test with ROPE

Because non-significant frequentist p-values cannot formally establish equivalence, we implemented the Bayesian Correlated t-Test with a Region of Practical Equivalence (ROPE) [25, 24]. For models A and B, paired fold diferences $\Delta _ { j } = s _ { i } ^ { ( A ) } - s _ { i } ^ { ( B ) }$ across all $n = 1 5$ folds were evaluated. To account for the correlation between folds sharing training data in cross-validation, the posterior variance was adjusted using:

$$
\hat { \sigma } _ { \mathrm { c o r r } } ^ { 2 } = \left( \frac { 1 } { n } + \frac { \rho } { 1 - \rho } \right) \hat { \sigma } _ { \Delta } ^ { 2 } , \quad \rho = \frac { 1 } { k } = 0 . 2 ,
$$

where $\rho$ represents the test-to-training ratio for $k = 5$ folds [25].

A clinically motivated ROPE interval of ±0 01 absolute units (1%) was established [24]. Integrating the posterior Student’s t-distribution over this interval yields three mutually exclusive probabilities:

$$
\begin{array} { r l } & { P ( A \succ B ) = P ( \Delta > 0 . 0 1 ) , } \\ & { P ( B \succ A ) = P ( \Delta < - 0 . 0 1 ) , } \\ & { P ( \mathrm { R O P E } ) = P ( | \Delta | \leq 0 . 0 1 ) . } \end{array}
$$

When $P ( { \mathrm { R O P E } } ) \geq 0 . 9 0$ , practical equivalence between architectures is considered supported by the evidence.

## 3.6. Hardware and Execution Environment

To ensure reproducible computational latency profiling, all benchmarks were executed locally in a dedicated environment on an Apple MacBook Pro equipped with an Apple M5 Pro system-on-chip with 16-core CPU, high-performance GPU, and unified memory architecture. Neural network operations were accelerated using Apple’s Metal Performance Shaders (MPS) backend within PyTorch 2.0+ [42]. Consistent seeds (random state 42 for data partitioning and initialization) were enforced across all trials. Inference latency was recorded using high-resolution monotonic clocks over warm forward passes to prevent cold-start artifacts. Numerical processing and data manipulation relied on NumPy [43], pandas [44], and SciPy [45], while all figures were produced using Matplotlib [46] and Seaborn [47].

## 4. Results

The following section presents the empirical findings from the 3×5 Repeated Stratified K-Fold benchmark across all seven medical datasets and eleven model architectures $( N =$ 105 evaluations per architecture). Results are reported as mean ± standard deviation aggregated across all 15 independent folds per dataset. Unless stated otherwise, crossdataset comparative summaries refer to grand means computed across the full benchmark.

## 4.1. Predictive Performance Under Class Imbalance

## 4.1.1. Discrimination Metrics

Across all seven medical datasets $( N \mathrm { ~  ~ { ~ = ~ } ~ } 1 0 5 \mathrm { ~ \ f o l d s ) . }$ TabKAN achieved the highest grand mean AUPRC (see Table E.5) at $0 . 9 0 6 1 \pm 0 . 1 2 6 2$ with an average rank of 3 43, followed by the baseline StandardMLP (0 9028±0 1278, average rank 4 43), FastKAN $( 0 . 9 0 1 3 \pm 0 . 1 3 2 3 )$ , ReLUKAN $( 0 . 8 9 9 9 ~ \pm ~ 0 . 1 3 5 8 )$ , TaylorKAN $( 0 . 8 9 9 2 \pm 0 . 1 3 1 3 )$ , and WavKAN (0 8980 ± 0 1384). Polynomial KANs recorded grand mean AUPRC values between 0 8894 (LegendreKAN) and 0 8962 (GramKAN), while the RandomForest baseline attained 0 8923±0 1455. Note that the high variability across datasets results in a large standard deviation for AUPRC but it concerns also the other metrics analyzed (Table E.5).

The slight advantage of dedicated KAN architectures was most evident on datasets characterized by pronounced minority-class prevalence:

• Cervical Cancer (6 4% positive prevalence): TabKAN attained an AUPRC of 0<sub>.</sub>7346 ± 0<sub>.</sub>0891, outperforming StandardMLP $( 0 . 7 1 5 0 \pm 0 . 0 8 8 0 )$ and the RandomForest baseline $( 0 . 6 6 0 9 \pm 0 . 0 8 4 9 )$ . Furthermore, TabKAN achieved a precision of 79 4% on biopsy-positive cases, compared to 66 5% for StandardMLP.

• Cardiotocography (8 3% minority class): WavKAN and TabKAN attained mean AUPRC scores of 0 9902 ± 0 0091 and $0 . 9 8 8 4 \pm 0 . 0 0 7 3$ , respectively, exceeding StandardMLP $( 0 . 9 7 4 1 \pm 0 . 0 1 9 4 )$ and achieving superior minority recall (96 9% and 95 6% vs. 92 7%).

Table 2: Bayesian Correlated t-Test Results (AUPRC). ROPE interval = ±0 01.
<table><tr><td>Dataset</td><td>Best KAN</td><td>P(MLP &gt;)</td><td>P(KAN &gt;)</td><td>P(ROPE)</td><td>Δ</td></tr><tr><td>Breast Cancer</td><td>TaylorKAN</td><td>&lt;0.001</td><td>&lt;0.001</td><td>1.000</td><td>+0.0014</td></tr><tr><td>Cardiotocography</td><td>WavKAN</td><td>&lt;0.001</td><td>0.885</td><td>0.115</td><td>-0.0161</td></tr><tr><td>Chronic Kidney Disease</td><td>LegendreKAN</td><td>&lt;0.001</td><td>&lt;0.001</td><td>1.000</td><td>-0.0004</td></tr><tr><td>Heart Disease</td><td>GramKAN</td><td>0.081</td><td>0.116</td><td>0.803</td><td>-0.0008</td></tr><tr><td>Parkinson&#x27;s Disease</td><td>WavKAN</td><td>0.130</td><td>0.005</td><td>0.865</td><td>+0.0043</td></tr><tr><td>Pima Diabetes</td><td>TabKAN</td><td>0.289</td><td>0.343</td><td>0.368</td><td>-0.0016</td></tr><tr><td>Cervical Cancer</td><td>TabKAN</td><td>0.223</td><td>0.598</td><td>0.179</td><td>-0.0195</td></tr><tr><td>OVERALL (Pooled)</td><td>Per-Dataset Best</td><td>0.169</td><td>0.364</td><td>0.467</td><td>-0.0047</td></tr></table>

Note: The representative KAN variant for each dataset was identified based on the highest mean Area Under the Precision-Recall Curve (AUPRC) across all 15 cross-validation folds. Acronyms and notation: ROPE = Region of Practical Equivalence (margin [−0 01 +0 01]); $P ( { \mathrm { M L P } } > ) = P ( \Delta > 0 . 0 1 )$ denotes posterior probability that StandardMLP outperforms KAN by more than the practical equivalence margin; P(KAN $> ) = P ( \Delta < - 0 . 0 1 )$ denotes posterior probability that KAN outperforms StandardMLP; P(ROPE) = P(|∆| ≤ 0 01) denotes posterior probability of practical equivalence; ∆ =<sup>¯</sup> Posterior Mean Diference (StandardMLP − KAN), where negative values indicate higher AUPRC for KAN.

• Pima Diabetes (34 9% positive prevalence): TabKAN reached an AUPRC of $0 . 7 2 7 9 \pm 0 . 0 6 3 2$ , performing slightly ahead of StandardMLP (0 7263 ± 0 0652) and ReLUKAN (0 7240 ± 0 0617).

Precision-Recall trajectories for two representative cohorts with high (Fig. 2a) and moderate (Fig. 2b) class imbalance are shown in Fig. 2. In both settings, TabKAN and related KAN variants demonstrate higher precision across intermediate-tohigh recall ranges compared to standard MLPs and tree ensembles.

The Critical Diference (CD) Diagram for AUPRC is shown in Fig. 3. While TabKAN obtained the lowest average rank across the seven datasets (3 43), the critical diference threshold $( C D = 5 . 7 1 )$ connects the evaluated models within a shared clique under the conservative Nemenyi posthoc test $( \alpha = 0 . 0 5 )$ . This indicates that, when averaged across highly diverse clinical domains, rank diferences alone do not achieve family-wise statistical separation.

## 4.1.2. Frequentist and Bayesian Statistical Analysis

Pairwise Wilcoxon signed-rank tests comparing Standard-MLP against the best-performing KAN variants across the seven datasets yielded unadjusted p-values between 0 109 and 0 813 for AUPRC. After Holm-Bonferroni correction, no architecture achieved formal statistical significance at $\alpha ~ = ~ 0 . 0 5$ As noted in Section 3.5, non-parametric tests have limited power when aggregating across a small number of dataset-level observations $( n = 7 )$ , where the minimum achievable two-sided p-value cannot clear conservative stepdown thresholds.

The Bayesian Correlated t-Test with ROPE [25, 24] provides a more nuanced probabilistic characterization of these diferences, summarized in Table 2. On datasets with balanced or moderately balanced classes, the posterior probability of practical equivalence reaches or approaches unity: $P ( \mathrm { R O P E } ) ~ = ~ 1 . 0 0 0$ on Breast Cancer and Chronic Kidney

Distribution of Metric AUPRC in Cross Validation  
![](images/cc448bdc1c68f9774456c293db8365492e981122ec62fd81cc24a2820d18d287.jpg)  
Figure 1: Distribution of per-fold Area Under the Precision-Recall Curve (AUPRC) scores for all eleven evaluated architectures across the seven medical datasets, shown as box-and-whisker plots (median, interquartile range, and 1.5-IQR whiskers). The results illustrate that while overall performance distributions overlap across balanced datasets, dedicated architectures such as TabKAN and FastKAN demonstrate competitive or superior retention of discriminative capacity on imbalanced cohorts (e.g., Cervical Cancer and Cardiotocography).

Disease, 0 803 on Heart Disease, and 0 865 on Parkinson’s Disease. In these contexts, KAN variants and Standard-MLP exhibit practically indistinguishable discriminative performance.

Conversely, on datasets with severe class imbalance, the Bayesian posterior shifts toward KAN superiority. On Cardiotocography, WavKAN records $P ( \mathrm { K A N } > ) \ = \ 0 . 8 8 5$ with $P ( \mathrm { R O P E } ) \ = \ 0 . 1 1 5$ and a posterior mean diference of $\bar { \Delta } = - 0 . 0 1 6 1$ , providing strong Bayesian evidence of higher AUPRC than StandardMLP. On Cervical Cancer, TabKAN achieves $P ( \mathrm { K A N ~ } > ) ~ = ~ 0 . 5 9 8$ versus $P ( \mathrm { M L P } ~ \succ ) ~ = ~ 0 . 2 2 3$ (P(ROPE) = 0 179, $\bar { \Delta } ~ = ~ - 0 . 0 1 9 5 )$ , indicating a substantial probability of clinically meaningful improvement on rare positive cases. In the pooled cross-dataset analysis $( N \mathbf { \Psi } =$ 105), the probability of KAN superiority exceeds that of the MLP (P(KAN ≻) = 0 364 vs. $P ( \mathrm { M L P } \ \mathrm { > } ) \ = \ 0 . 1 6 9 )$ , with $P ( \mathrm { R O P E } ) = 0 . 4 6 7 $

These findings suggest that while KANs and MLPs display comparable predictive capacity in balanced settings, specialized KAN architectures (particularly TabKAN and WavKAN) can ofer meaningful discriminative benefits when detecting rare pathological conditions.

## 4.2. Model Calibration

Safe clinical integration requires not only accurate classification, but also reliable probability estimates [2]. Model calibration was evaluated through the Brier Score, where lower scores indicate superior probabilistic fidelity. Distributional profiles across cross-validation folds are presented in Fig. 4, and grand summary scores are provided in Table 3.

Table 3: Grand Mean Brier Score across all datasets and folds (Grand Mean± Std, lower is better).
<table><tr><td rowspan=1 colspan=1>Model           Grand Mean Brier Score</td></tr><tr><td rowspan=1 colspan=1>TabKAN               $0 . 0 5 8 0 \pm 0 . 0 5 4 7$ StandardMLP          $0 . 0 5 8 5 \pm 0 . 0 5 4 2$ </td></tr><tr><td rowspan=1 colspan=1>WavKAN              $0 . 0 5 9 1 \pm 0 . 0 5 6 9$ </td></tr><tr><td rowspan=1 colspan=1>FastKAN               $0 . 0 6 0 0 \pm 0 . 0 5 5 9$ </td></tr><tr><td rowspan=1 colspan=1>ReLUKAN             $0 . 0 6 0 6 \pm 0 . 0 5 5 8$ </td></tr><tr><td rowspan=1 colspan=1>TaylorKAN            $0 . 0 6 0 6 \pm 0 . 0 5 6 0$ </td></tr><tr><td rowspan=1 colspan=1>LegendreKAN         $0 . 0 6 1 1 \pm 0 . 0 6 2 8$ </td></tr><tr><td rowspan=1 colspan=1>GramKAN             $0 . 0 6 1 5 \pm 0 . 0 5 8 9$ </td></tr><tr><td rowspan=1 colspan=1>JacobiKAN            $0 . 0 6 2 4 \pm 0 . 0 5 8 3$ </td></tr><tr><td rowspan=1 colspan=1>ChebyKAN            $0 . 0 6 3 4 \pm 0 . 0 6 2 8$ </td></tr><tr><td rowspan=1 colspan=1>RandomForest         $0 . 0 6 6 5 \pm 0 . 0 5 2 3$ </td></tr></table>

Note: BS = Brier Score (quadratic calibration loss measuring mean squared diference between predicted class probabilities and binary ground-truth labels; ranges from 0 to 1, where lower scores indicate better probabilistic calibration). Values represent grand mean ± standard deviation aggregated across all seven clinical datasets (N = 105 evaluations per architecture).

![](images/5d1679f6629845577082385fa174f7a82f9bdb11f21db9123cd947a335992ed9.jpg)  
(a) Cervical Cancer (6 4% pos.)

![](images/e8f5f9a3e155fe3d35f1fb521952cd1eaffaae3a0ccbad2514d666d8cca80e53.jpg)  
(b) Pima Diabetes (34 9% pos.)  
Figure 2: Precision-Recall curves for representative clinical cohorts with high and moderate class imbalance, comparing neural architectures and Random Forest baseline across cross-validation folds.

TabKAN achieved the lowest grand mean Brier Score $( 0 . 0 5 8 0 ~ \pm ~ 0 . 0 5 4 7 )$ , closely matched by StandardMLP $( 0 . 0 5 8 5 \pm 0 . 0 5 4 2 )$ and WavKAN $( 0 . 0 5 9 1 \pm 0 . 0 5 6 9 )$ . Splineand wavelet-based KANs maintained competitive calibration alongside the neural baseline, whereas polynomial variants exhibited marginally higher Brier scores (0 0611–0 0634). RandomForest recorded the highest overall calibration error $( 0 . 0 6 6 5 \pm 0 . 0 5 2 3 )$ , reflecting the known tendency of uncalibrated decision trees to produce stepped, miscalibrated posterior estimates.

Bayesian ROPE analysis on Brier scores confirms practical equivalence between leading KAN variants and the MLP baseline across most datasets: posterior probabilities P(ROPE) exceeded 0 73 on five of the seven datasets (reaching 0 999 on Cervical Cancer and Pima Diabetes). Overall, these data indicate that modern KAN implementations can achieve calibration quality comparable to tuned MLPs without requiring secondary post-hoc calibration procedures.

Table 4: Computational Cost Summary (Grand Mean across all datasets and folds).
<table><tr><td>Model</td><td> $\bar { t } _ { \mathrm { i n f } } \left( \mathbf { m } \mathbf { s } \right)$ </td><td>Ratio vs. MLP</td><td>|θ| (Grand Mean)</td><td> $\bar { T } _ { \mathbf { t r a i n } } \left( \mathbf { s } \right)$ </td></tr><tr><td>StandardMLP</td><td>0.0611</td><td>1.00×</td><td>8,273</td><td>1.22</td></tr><tr><td>ReLUKAN</td><td>0.0735</td><td>1.20×</td><td>73,906</td><td>2.04</td></tr><tr><td>FastKAN</td><td>0.0745</td><td>1.22×</td><td>39,143</td><td>1.66</td></tr><tr><td>TabKAN</td><td>0.0752</td><td>1.23×</td><td>56,990</td><td>2.04</td></tr><tr><td>TaylorKAN</td><td>0.0816</td><td>1.34×</td><td>23,885</td><td>1.86</td></tr><tr><td>WavKAN</td><td>0.0826</td><td>1.35×</td><td>103,347</td><td>2.23</td></tr><tr><td>ChebyKAN</td><td>0.0881</td><td>1.44×</td><td>23,361</td><td>1.64</td></tr><tr><td>GramKAN</td><td>0.0882</td><td>1.44×</td><td>23,678</td><td>1.86</td></tr><tr><td>LegendreKAN</td><td>0.0885</td><td>1.45×</td><td>23,622</td><td>1.73</td></tr><tr><td>JacobiKAN</td><td>0.0906</td><td>1.48×</td><td>22,892</td><td>1.61</td></tr><tr><td>RandomForest</td><td>0.1178</td><td>1.93×</td><td>一</td><td>0.07</td></tr></table>

Note: Computational metrics: $\bar { t } _ { \mathrm { i n f } } =  { \mathrm { M e a n } }$ per-sample inference latency in milliseconds (ms), measured with batch size 1 on Apple Silicon MPS hardware; Ratio vs. MLP = Relative inference latency factor normalized to the StandardMLP baseline (1 00×); |θ| = Mean count of trainable network parameters; $\bar { T } _ { \mathrm { t r a i n } }$ = Mean wall-clock training duration per cross-validation fold in seconds (s). Values denote grand means across all 7 benchmark datasets $( N = 1 0 5$ cross-validation evaluations per architecture).

## 4.3. Computational Cost and Eficiency Trade-ofs

While predictive discrimination and calibration are closely matched or slightly in favor of specific KAN variants on imbalanced tasks, computational profiling reveals systematic diferences in inference latency, parameter footprint, and training duration.

## 4.3.1. Inference Latency

Per-sample inference latencies measured on dedicated Apple Silicon hardware (MPS backend) are reported in Table 4. StandardMLP achieved the lowest median latency at $\bar { t } _ { \mathrm { i n f } } = 0 . 0 6 1 1 \pm 0 . 0 1 6 5$ ms per sample. In comparison, KAN architectures introduced measurable latency overheads: Re-LUKAN was 1 20× slower $( 0 . 0 7 3 5 \pm 0 . 0 2 5 6 \mathrm { m s } )$ , FastKAN was 1 22× slower $( 0 . 0 7 4 5 \pm 0 . 0 2 4 4 \mathrm { m s } )$ , and TabKAN was 1 23× slower $( 0 . 0 7 5 2 \pm 0 . 0 2 6 0 \mathrm { m s } )$ . Polynomial and wavelet variants imposed larger overheads, reaching 1 34× for TaylorKAN (0 0816 ms), 1 35× for WavKAN (0 0826 ms), and up to 1 48× for JacobiKAN $( 0 . 0 9 0 6 \pm 0 . 0 2 9 8 \mathrm { m s } )$ The tree-based RandomForest recorded an inference latency of $0 . 1 1 7 8 \pm 0 . 0 6 5 8$ ms (1 93× the MLP baseline).

## 4.3.2. Parameter Complexity

The StandardMLP proved notably more parametereficient than all evaluated KAN models. With an average trainable parameter count of $8 , 2 7 3 \pm 5 , 2 1 0 .$ , the MLP required between 2 8× and 12 5× fewer parameters than the KAN family. Polynomial KAN variants ranged from 22 892 (JacobiKAN) to 23 885 (TaylorKAN), while spline and wavelet models carried larger parameter footprints: 39 143 (FastKAN), 56 990 (TabKAN), 73 906 (ReLUKAN), and 103 347 (WavKAN). This expansion reflects the structural mechanism of KANs, which substitute scalar synaptic weights with parameter vectors for edge-based activation bases.

![](images/2e293ea5a4f60041e25031f73f34a4f345004dfa33c02d386d57d2ec01432d7a.jpg)  
Figure 3: Critical Diference (CD) Diagram for AUPRC constructed from average ranks across all seven benchmark datasets under the Nemenyi post-hoc test $( \alpha = 0 . 0 5 )$ . The horizontal axis represents the average rank of each model, where a lower rank indicates higher AUPRC. To interpret the diagram: the solid red line at the top visualizes the required critical diference distance $( C D = 5 . 7 1 )$ , and the thick gray line connects models whose pairwise rank diferences are smaller than this threshold (i.e., they are not statistically significantly diferent). While TabKAN achieves the lowest average rank (3 43), the critical diference connects all evaluated models in a single clique. This indicates that rank diferences across the heterogeneous benchmark do not cross the familywise significance threshold, rendering all architectures statistically equivalent under this frequentist test.

Distribution of Metric BRIER\_SCORE in Cross Validation  
![](images/131f8ed8f345dcd104bcbc6efa75e77585ee60b41bbe37bd7aa16adb16af988b.jpg)  
Figure 4: Distribution of Brier Scores across the repeated cross-validation folds. Lower scores indicate better probabilistic calibration, a critical requirement for safe clinical risk stratification.

## 4.3.3. Training Duration

Per-fold training times were consistent with the architectural complexity patterns. The StandardMLP completed training in an average of $1 . 2 2 \pm 1 . 1 2 \mathrm { s }$ per fold, whereas KAN variants required between 1 61 s (JacobiKAN) and 2 23 s (WavKAN), with TabKAN averaging $2 . 0 4 \pm 1 . 5 1 \mathrm { s } .$ While non-neural RandomForest trained significantly faster $( 0 . 0 7 \pm$ 0 05 s), the training overhead of KANs remained manageable within standard automated machine learning pipelines.

![](images/2dcd54d26a27592965c69b16714a70d3fdbcfd6d4855f220b64b29435c4463e2.jpg)  
Mean Inference Time per Sample (ms) [Log Scale]  
Figure 5: Pareto Frontier of grand mean AUPRC versus mean per-sample inference latency (log scale) across all eleven evaluated architectures. The empirical Pareto frontier is formed by StandardMLP (lowest latency: 0 0611 ms) and TabKAN (highest AUPRC: 0 9061). Other KAN variants and baseline models remain within the dominated region, highlighting the operational trade-of between baseline execution speed and KAN minorityclass retrieval performance.

## 4.3.4. Pareto Frontier Analysis

The operational trade-of between discriminative quality (AUPRC) and execution latency is visualised in Fig. 5. Unlike single-metric evaluations that might suggest unconditional dominance, the empirical Pareto frontier is defined jointly by two models:

1. StandardMLP: Establishes the lower bound on inference latency (0 0611 ms) while maintaining high discriminative utility (0 9028 AUPRC).

2. TabKAN: Achieves the highest overall discriminative performance (0 9061 AUPRC) at a moderate latency cost (0 0752 ms, a 1 23× overhead).

All other architectures occupy strictly interior (dominated) positions within the trade-of space. In practical terms, this result suggests that selection between an MLP and a tailored KAN depends on clinical deployment constraints: when latency is critical (such as bedside monitoring or edge diagnostics), StandardMLP remains optimal; when maximizing sensitivity and precision on imbalanced disease classes takes priority, TabKAN provides a viable, higher-performing alternative.

## 5. Conclusion

This study provided a standardized benchmark comparing Kolmogorov-Arnold Networks (KANs), Multi-Layer Perceptrons (MLPs), and tree-based baselines across seven medical tabular datasets under repeated stratified cross-validation with nested Bayesian hyperparameter optimization. Motivated by the prevalence of class imbalance in clinical decision-making, the evaluation centered on the Area Under the Precision-Recall Curve (AUPRC), probabilistic calibration via the Brier Score, and computational eficiency, supplemented by formal Bayesian ROPE testing.

Our findings lead to three main conclusions regarding the practical utility of KANs in medical tabular domains:

First, on datasets with severe class imbalance, specialized KAN architectures demonstrate tangible discriminative advantages. In particular, TabKAN attained the highest grand mean AUPRC across the benchmark (0 9061 ± 0 1262, rank 3 43), outperforming StandardMLP on cohorts such as Cervical Cancer (0 7346 vs. 0 7150, with precision reaching 79 4% compared to 66 5%) and Cardiotocography (0 9884 vs. 0 9741). On these imbalanced tasks, Bayesian equivalence tests indicated elevated posterior probabilities of KAN superiority $( P ( \mathrm { K A N \ > } ) ~ \in ~ [ 0 . 5 9 8 , 0 . 8 8 5 ] )$ , suggesting that edge-based activation functions and soft gating can better capture subtle non-linear boundaries of rare positive classes.

Second, across balanced or moderately balanced clinical datasets, KAN variants and StandardMLP exhibit practical equivalence. Posterior probabilities of falling within the ±0 01 ROPE interval reached unity on Breast Cancer and Chronic Kidney Disease, and exceeded 0 80 on Heart Disease and Parkinson’s Disease. Furthermore, calibration analysis showed that leading KAN variants maintain low Brier scores (0 0580–0 0606) that are on par with tuned MLPs (0 0585), indicating that their probability estimates are well calibrated without post-hoc recalibration.

Third, computational overhead remains a tangible tradeof for KAN architectures. On tested hardware, KAN variants incurred inference latencies that were 1 20× to 1 48× higher than the StandardMLP (0 0611 ms per sample), accompanied by a 2 8× to 12 5× increase in trainable parameter count (8 273 parameters for the MLP vs. 22 892–103 347 for KANs). As a result, the empirical Pareto frontier for AUPRC versus inference latency is co-defined by Standard-MLP (minimum latency) and TabKAN (maximum AUPRC).

In clinical practice, these findings suggest that the choice of architecture should be guided by task requirements. For latency-sensitive applications such as emergency triage, bedside telemetry, or deployment on resource-constrained medical edge devices, StandardMLP remains a compelling and computationally eficient choice. Conversely, for screening tasks and clinical diagnostics where the primary objective is maximizing precision and recall on rare pathological events, architectures like TabKAN ofer a viable and efective alternative.

Future research could explore several directions. First, given the rapid development of novel KAN architectures, expanding this comparative analysis to encompass newly emerging variants could help identify architectures specifically suited for clinical tabular data. Second, leveraging the symbolic interpretability of learned edge functions presents an opportunity for clinicians to inspect non-linear biomarker interactions directly. Finally, extending this benchmarking framework to large-scale longitudinal cohorts, such as MIMIC-IV [48] and PhysioNet [49] databases, will help assess the generalizability of these observations across diverse clinical workflows.

Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work, the authors used Gemini and Claude (Sonnet and Opus models) within the Antigravity IDE framework to assist in developing and debugging the experimental codebase (Python scripts for model evaluation). After using these tools, the authors thoroughly reviewed and edited the output as needed, verified all empirical results against the codebase, and take full responsibility for the content and integrity of the published code.

## Acknowledgment

The publication was created within the project of the Minister of Science and Higher Education "Support for the activity of Centers of Excellence established in Poland under Horizon 2020" on the basis of the contract number MEiN/2023/DIR/3796. This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 857533.

## References

[1] P. Rajpurkar, E. Chen, O. Banerjee, E. J. Topol, Ai in health and medicine, Nature Medicine 28 (1) (2022) 31–38. doi:10.1038/ s41591-021-01614-0. URL http://dx.doi.org/10.1038/s41591-021-01614-0

[2] B. and Van Calster, D. J. McLernon, M. van Smeden, L. Wynants, E. W. Steyerberg, Calibration: the achilles heel of predictive analytics, BMC Medicine 17 (1) (Dec. 2019). doi:10.1186/s12916-019-1466-7. URL http://dx.doi.org/10.1186/s12916-019-1466-7

[3] T. Saito, M. Rehmsmeier, The precision-recall plot is more informative than the roc plot when evaluating binary classifiers on imbalanced datasets, PLOS ONE 10 (3) (2015) e0118432. doi:10.1371/ journal.pone.0118432. URL http://dx.doi.org/10.1371/journal.pone.0118432

[4] T. Chen, C. Guestrin, Xgboost: A scalable tree boosting system, in: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’16, ACM, 2016, p. 785–794. doi:10.1145/2939672.2939785. URL http://dx.doi.org/10.1145/2939672.2939785

[5] L. Breiman, Random forests, Machine Learning 45 (1) (2001) 5–32. doi:10.1023/a:1010933404324. URL http://dx.doi.org/10.1023/A:1010933404324

[6] R. Shwartz-Ziv, A. Armon, Tabular data: Deep learning is not all you need, Information Fusion 81 (2022) 84–90. doi:10.1016/j. inffus.2021.11.011. URL http://dx.doi.org/10.1016/j.inffus.2021.11.011

[7] L. Grinsztajn, E. Oyallon, G. Varoquaux, Why do tree-based models still outperform deep learning on typical tabular data?, in: Advances in Neural Information Processing Systems 35, NeurIPS 2022, Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2022, p. 507–520. doi:10.52202/068431-0037. URL http://dx.doi.org/10.52202/068431-0037

[8] K. Hornik, M. Stinchcombe, H. White, Multilayer feedforward networks are universal approximators, Neural Networks 2 (5) (1989) 359–366. doi:10.1016/0893-6080(89)90020-8. URL http://dx.doi.org/10.1016/0893-6080(89)90020-8

[9] Z. Liu, Y. Wang, S. Vaidya, F. Ruehle, J. Halverson, M. Soljaciˇ c, T. Y.´ Hou, M. Tegmark, Kan: Kolmogorov-arnold networks (2025). arXiv: 2404.19756. URL https://arxiv.org/abs/2404.19756

[10] A. N. Kolmogorov, On the representation of continuous functions of many variables by superposition of continuous functions of one variable and addition (1963). doi:10.1090/trans2/028/04. URL http://dx.doi.org/10.1090/trans2/028/04

[11] Springer Berlin Heidelberg, 2009, p. 5–8. doi:10.1007/ 978-3-642-01742-1\_2, [link]. URL http://dx.doi.org/10.1007/978-3-642-01742-1\_2

[12] Z. Chao, X. Liu, Z. Wu, X. Li, Rbf-kan: Radial basis function–kolmogorov–arnold network, IEEE Internet of Things Journal 13 (10) (2026) 21018–21027. doi:10.1109/jiot.2026.3665400. URL http://dx.doi.org/10.1109/jiot.2026.3665400

[13] Z. Bozorgasl, H. Chen, Wav-kan: Wavelet kolmogorov-arnold networks (2024). doi:10.2139/ssrn.4835325. URL http://dx.doi.org/10.2139/ssrn.4835325

[14] Q. Qiu, T. Zhu, H. Gong, L. Chen, H. Ning, Relu-kan: New kolmogorov-arnold networks that only need matrix addition, dot multiplication, and relu, in: 2025 IEEE Smart World Congress (SWC), IEEE, 2025, p. 1686–1694. doi:10.1109/swc65939.2025.00262. URL http://dx.doi.org/10.1109/swc65939.2025.00262

[15] S. SS, K. AR, G. R, A. KP, Chebyshev polynomial-based kolmogorovarnold networks: An eficient architecture for nonlinear function approximation (2024). arXiv:2405.07200. URL https://arxiv.org/abs/2405.07200

[16] S. Yu, Z. Chen, Z. Yang, J. Gu, B. Feng, Q. Sun, Exploring kolmogorov-arnold networks for realistic image sharpness assessment, in: ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), IEEE, 2025, p. 1–5. doi:10.1109/icassp49660.2025.10890447. URL http://dx.doi.org/10.1109/icassp49660.2025. 10890447

[17] A. Afzal Aghaei, fkan: Fractional kolmogorov–arnold networks with trainable jacobi basis functions, Neurocomputing 623 (2025) 129414. doi:10.1016/j.neucom.2025.129414. URL http://dx.doi.org/10.1016/j.neucom.2025.129414

[18] SpaceLearner, JacobiKAN: Kolmogorov-arnold networks using jacobi polynomials, https://github.com/SpaceLearner/JacobiKAN, software repository (2024).

[19] X. Xiong, K. Lu, Z. Zhang, Z. Zeng, S. Zhou, Z. Deng, R. Hu, Jpikan: A physics-informed kan network based on jacobi orthogonal polynomials for solving fluid dynamics, Communications in Nonlinear Science and Numerical Simulation 152 (2026) 109414. doi: 10.1016/j.cnsns.2025.109414.

[20] B. Hu, L. Jin, Z. Li, Solving the fully nonlinear monge-ampÈre equation using the legendre-kolmogorov-arnold network method (2025). doi:10.2139/ssrn.5210735. URL http://dx.doi.org/10.2139/ssrn.5210735

[21] Khochawongwat, GRAMKAN: Kolmogorov-arnold networks based on gram polynomials, https://github.com/Khochawongwat/ GRAMKAN, software repository (2024).

[22] A. Eslamian, A. Afzal Aghaei, Q. Cheng, Tabkan: Advancing tabular data analysis using kolmogorov-arnold network, Machine Learning for Computational Science and Engineering 1 (2) (Nov. 2025). doi:10. 1007/s44379-025-00042-y. URL http://dx.doi.org/10.1007/s44379-025-00042-y

[23] T. Akiba, S. Sano, T. Yanase, T. Ohta, M. Koyama, Optuna: A nextgeneration hyperparameter optimization framework, in: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’19, ACM, 2019, p. 2623–2631. doi:10.1145/3292500.3330701. URL http://dx.doi.org/10.1145/3292500.3330701

[24] A. Benavoli, G. Corani, J. Demšar, M. Zafalon, Time for a change: a tutorial for comparing multiple classifiers through bayesian analysis, Journal of Machine Learning Research 18 (77) (2017) 1–36. URL http://jmlr.org/papers/v18/16-305.html

[25] G. Corani, A. Benavoli, A bayesian approach for comparing crossvalidated algorithms on multiple data sets, Machine Learning 100 (2-3) (2015) 285–304. doi:10.1007/s10994-015-5486-z. URL http://dx.doi.org/10.1007/s10994-015-5486-z

[26] R. Yu, W. Yu, X. Wang, Kan or mlp: A fairer comparison (2024). arXiv:2407.16674.

URL https://archive.ics.uci.edu/dataset/336

URL http://dx.doi.org/10.1038/s41597-022-01899-x

URL http://dx.doi.org/10.1038/s41592-019-0686-2

URL https://arxiv.org/abs/2407.16674

[27] J. Demšar, Statistical comparisons of classifiers over multiple data sets, Journal of Machine Learning Research 7 (1) (2006) 1–30. URL http://jmlr.org/papers/v7/demsar06a.html

[28] Y. Gal, Z. Ghahramani, Dropout as a bayesian approximation: Representing model uncertainty in deep learning, in: M. F. Balcan, K. Q. Weinberger (Eds.), Proceedings of The 33rd International Conference on Machine Learning, Vol. 48 of Proceedings of Machine Learning Research, PMLR, New York, New York, USA, 2016, pp. 1050–1059. URL https://proceedings.mlr.press/v48/gal16.html

[29] B. Lakshminarayanan, A. Pritzel, C. Blundell, Simple and scalable predictive uncertainty estimation using deep ensembles, in: I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, R. Garnett (Eds.), Advances in Neural Information Processing Systems, Vol. 30, Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper\_files/paper/ 2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf

[30] M. Abdar, F. Pourpanah, S. Hussain, D. Rezazadegan, L. Liu, M. Ghavamzadeh, P. Fieguth, X. Cao, A. Khosravi, U. R. Acharya, V. Makarenkov, S. Nahavandi, A review of uncertainty quantification in deep learning: Techniques, applications and challenges, Information Fusion 76 (2021) 243–297. doi:10.1016/j.inffus.2021.05.008. URL http://dx.doi.org/10.1016/j.inffus.2021.05.008

[31] M. Kelly, R. Longjohn, K. Nottingham, The UCI machine learning repository. URL https://archive.ics.uci.edu

[32] P. Ghosh, Breast cancer wisconsin (diagnostic) prediction, International Journal of Science and Research (IJSR) 11 (5) (2022) 178–185. doi:10.21275/sr22501213650. URL http://dx.doi.org/10.21275/sr22501213650

[33] J. W. Smith, J. Everhart, W. Dickson, W. Knowler, R. Johannes, Using the ADAP learning algorithm to forecast the onset of diabetes mellitus, 1988, pp. 261–265.

[34] R. Detrano, A. Janosi, W. Steinbrunn, M. Pfisterer, J.-J. Schmid, S. Sandhu, K. H. Guppy, S. Lee, V. Froelicher, International application of a new probability algorithm for the diagnosis of coronary artery disease, The American Journal of Cardiology 64 (5) (1989) 304–310. doi:10.1016/0002-9149(89)90524-9. URL http://dx.doi.org/10.1016/0002-9149(89)90524-9

[35] P. S. L. Rubini, Chronic kidney disease (2015). doi:10.24432/ C5G020.

[36] M. Little, P. McSharry, S. Roberts, D. Costello, I. Moroz, Exploiting nonlinear recurrence and fractal scaling properties for voice disorder detection, Nature Precedings (2007). doi:10.1038/npre.2007. 326.1.

URL http://dx.doi.org/10.1038/npre.2007.326.1

[37] K. Fernandes, J. S. Cardoso, J. Fernandes, Transfer Learning with Partial Observability Applied to Cervical Cancer Screening, Springer International Publishing, 2017, p. 243–250. doi:10.1007/ 978-3-319-58838-4\_27.

[38] D. Ayres-de Campos, J. Bernardes, A. Garrido, J. Marques-de Sá’, L. Pereira-Leite, Sisporto 2.0: A program for automated analysis of cardiotocograms, The Journal of Maternal-Fetal Medicine 9 (5) (2000) 311–318. doi:10.1002/1520-6661(200009/10)9: 5<311::aid-mfm12>3.0.co;2-9. URL http://dx.doi.org/10.1002/1520-6661(200009/10)9: 5<311::aid-mfm12>3.0.co;2-9

[39] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, Édouard Duchesnay, Scikit-learn: Machine learning in python, Journal of Machine Learning Research 12 (85) (2011) 2825–2830. URL http://jmlr.org/papers/v12/pedregosa11a.html

[40] D. Chicco, G. Jurman, The advantages of the matthews correlation coeficient (mcc) over f1 score and accuracy in binary classification evaluation, BMC Genomics 21 (1) (Jan. 2020). doi:10.1186/ s12864-019-6413-7.

URL http://dx.doi.org/10.1186/s12864-019-6413-7

[41] C. A. T. Ferro, T. E. Fricker, A bias-corrected decomposition of the

brier score, Quarterly Journal of the Royal Meteorological Society 138 (668) (2012) 1954–1960. doi:10.1002/qj.1924. URL http://dx.doi.org/10.1002/qj.1924

[42] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, S. Chintala, Pytorch: An imperative style, highperformance deep learning library, in: H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, R. Garnett (Eds.), Advances in Neural Information Processing Systems, Vol. 32, Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper\_files/paper/ 2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf

[43] C. R. Harris, K. J. Millman, S. J. van der Walt, R. Gommers, P. Virtanen, D. Cournapeau, E. Wieser, J. Taylor, S. Berg, N. J. Smith, R. Kern, M. Picus, S. Hoyer, M. H. van Kerkwijk, M. Brett, A. Haldane, J. F. del Río, M. Wiebe, P. Peterson, P. Gérard-Marchant, K. Sheppard, T. Reddy, W. Weckesser, H. Abbasi, C. Gohlke, T. E. Oliphant, Array programming with numpy, Nature 585 (7825) (2020) 357–362. doi:10.1038/s41586-020-2649-2. URL http://dx.doi.org/10.1038/s41586-020-2649-2

[44] W. McKinney, Data structures for statistical computing in python, in: Proceedings of the 9th Python in Science Conference, SciPy, SciPy, 2010, p. 56–61. doi:10.25080/majora-92bf1922-00a. URL http://dx.doi.org/10.25080/Majora-92bf1922-00a

[45] P. Virtanen, R. Gommers, T. E. Oliphant, M. Haberland, T. Reddy, D. Cournapeau, E. Burovski, P. Peterson, W. Weckesser, J. Bright, S. J. van der Walt, M. Brett, J. Wilson, K. J. Millman, N. Mayorov, A. R. J. Nelson, E. Jones, R. Kern, E. Larson, C. J. Carey, <sup>˙</sup>I. Polat, Y. Feng, E. W. Moore, J. VanderPlas, D. Laxalde, J. Perktold, R. Cimrman, I. Henriksen, E. A. Quintero, C. R. Harris, A. M. Archibald, A. H. Ribeiro, F. Pedregosa, P. van Mulbregt, A. Vijaykumar, A. P. Bardelli, A. Rothberg, A. Hilboll, A. Kloeckner, A. Scopatz, A. Lee, A. Rokem, C. N. Woods, C. Fulton, C. Masson, C. Häggström, C. Fitzgerald, D. A. Nicholson, D. R. Hagen, D. V. Pasechnik, E. Olivetti, E. Martin, E. Wieser, F. Silva, F. Lenders, F. Wilhelm, G. Young, G. A. Price, G.-L. Ingold, G. E. Allen, G. R. Lee, H. Audren, I. Probst, J. P. Dietrich, J. Silterra, J. T. Webber, J. Slavic,ˇ J. Nothman, J. Buchner, J. Kulick, J. L. Schönberger, J. V. de Miranda Cardoso, J. Reimer, J. Harrington, J. L. C. Rodríguez, J. Nunez-Iglesias, J. Kuczynski, K. Tritz, M. Thoma, M. Newville, M. Kümmerer, M. Bolingbroke, M. Tartre, M. Pak, N. J. Smith, N. Nowaczyk, N. Shebanov, O. Pavlyk, P. A. Brodtkorb, P. Lee, R. T. McGibbon, R. Feldbauer, S. Lewis, S. Tygier, S. Sievert, S. Vigna, S. Peterson, S. More, T. Pudlik, T. Oshima, T. J. Pingel, T. P. Robitaille, T. Spura, T. R. Jones, T. Cera, T. Leslie, T. Zito, T. Krauss, U. Upadhyay, Y. O. Halchenko, Y. Vázquez-Baeza, Scipy 1.0: fundamental algorithms for scientific computing in python, Nature Methods 17 (3) (2020) 261–272. doi:10.1038/s41592-019-0686-2.

[46] J. D. Hunter, Matplotlib: A 2d graphics environment, Computing in Science and Engineering 9 (3) (2007) 90–95. doi:10.1109/mcse. 2007.55. URLhttp://dx.doi.org/10.1109/MCSE.2007.55 URL http://dx.doi.org/10.1109/MCSE.2007.55

[47] M. Waskom, seaborn: statistical data visualization, Journal of Open Source Software 6 (60) (2021) 3021. doi:10.21105/joss.03021. URL http://dx.doi.org/10.21105/joss.03021

[48] A. E. W. Johnson, L. Bulgarelli, L. Shen, A. Gayles, A. Shammout, S. Horng, T. J. Pollard, S. Hao, B. Moody, B. Gow, L.-w. H. Lehman, L. A. Celi, R. G. Mark, Mimic-iv, a freely accessible electronic health record dataset, Scientific Data 10 (1) (Jan. 2023). doi:10.1038/ s41597-022-01899-x.

[49] A. L. Goldberger, L. A. N. Amaral, L. Glass, J. M. Hausdorf, P. C. Ivanov, R. G. Mark, J. E. Mietus, G. B. Moody, C.-K. Peng, H. E. Stanley, Physiobank, physiotoolkit, and physionet: Components of a new research resource for complex physiologic signals, Circulation 101 (23) (2000). doi:10.1161/01.cir.101.23.e215. URL http://dx.doi.org/10.1161/01.CIR.101.23.e215

## Appendix A. Exhaustive Performance Benchmarks

Here, we provide a consolidated overview of predictive performance across all seven medical tabular datasets. Table E.5 presents the Grand Summary across six key evaluation metrics, aggregated over N = 105 cross-validation folds per architecture (15 folds across seven datasets). In addition to the primary AUPRC metric, the summary includes AUROC, Matthews Correlation Coeficient (MCC), macroaveraged F1-Score, Balanced Accuracy, and the Brier Score.

Figure E.6 illustrates the distributional profiles across all eleven evaluated architectures for key supplementary metrics: AUROC, Matthews Correlation Coeficient (MCC), macroaveraged F1-Score, and per-sample inference latency.

## Appendix B. Computational Profiling

Detailed profiling of computational overhead across all architectures is summarized in Table E.6. Figure E.6 (bottom panel) presents the per-sample inference latency distributions on a logarithmic scale.

The empirical latency disparity is attributable to computational primitives: StandardMLP computes layer outputs using heavily optimized General Matrix Multiply (GEMM) kernels with scalar weights, which fully leverage the hardware architecture of modern processors. In contrast, KAN architectures evaluate univariate basis functions along network edges, requiring additional memory bandwidth and kernel dispatches. While specialized variants like ReLUKAN and TabKAN mitigate this gap compared to higher-degree polynomial KANs, the structural latency overhead remains an important consideration for real-time clinical deployment.

## Appendix C. Sample Eficiency and Data Regime Ablation

To examine how architectural choices influence resilience to data scarcity, we evaluated model performance in terms of AUPRC as the training sample size was progressively ablated from 100% down to 10% of available training data. Figure E.7 displays the degradation trajectories across six representative datasets spanning high, moderate, and low class imbalance: Cervical Cancer, Cardiotocography, Parkinson’s Disease, Pima Diabetes, Breast Cancer, and Heart Disease.

Across all evaluated models, performance decreases steadily as the training data fraction is reduced. All tested models exhibit comparable degradation slopes, suggesting that the non-linear basis expansions of KANs neither accelerate sample eficiency in severely restricted regimes nor introduce excessive sample ineficiency relative to a wellregularized MLP.

## Appendix D. Training Dynamics and Convergence Stability

Convergence behavior across training epochs was recorded to assess optimization stability. Figure E.8 presents training and validation loss curves comparing the baseline Standard-MLP with representative KAN variants across four clinical domains.

Both model families converge reliably within the allocated epoch budget under AdamW optimization. Due to their flexible univariate activations, KAN variants often exhibit slightly steeper initial training loss reductions. However, validation loss curves plateau at similar epochs, highlighting the importance of the early-stopping mechanism implemented in our experimental protocol to prevent overparameterized KAN variants from overfitting on smaller clinical cohorts.

## Appendix E. Diagnostic Confusion Matrices

To inspect class-specific prediction distributions, Fig. E.9 compares confusion matrices between StandardMLP and the top-performing KAN variant on four benchmark datasets.

On cohorts with high class imbalance, such as Cervical Cancer and Cardiotocography, TabKAN and WavKAN demonstrate improved retrieval of minority positive instances. In Cervical Cancer, TabKAN achieves higher precision on positive biopsy outcomes while reducing false alarms. Similarly, in Cardiotocography, WavKAN achieves higher pathological class recall (96 9% vs. 92 7%). In balanced settings, both architectures exhibit nearly identical error distributions, indicating consistent classification behavior across diverse pathologies.

Table E.5: Summary of Predictive Performance (Grand Mean ± Std across all datasets and folds)
<table><tr><td>Architecture</td><td>AUPRC</td><td>AUROC</td><td>MCC</td><td>F1-Score</td><td>Balanced Accuracy</td><td>Brier Score</td></tr><tr><td>TabKAN</td><td> $0 . 9 0 6 1 \pm 0 . 1 2 6 2$ </td><td> $0 . 9 4 8 1 \pm 0 . 0 6 5 1$ </td><td> $0 . 7 8 9 0 \pm 0 . 1 9 5 9$ </td><td> $0 . 8 5 9 7 \pm 0 . 1 5 6 5$ </td><td> $0 . 8 8 5 8 \pm 0 . 1 0 6 2$ </td><td> $0 . 0 5 8 0 \pm 0 . 0 5 4 7$ </td></tr><tr><td>StandardMLP</td><td> $0 . 9 0 2 8 \pm 0 . 1 2 7 8$ </td><td> $0 . 9 5 4 4 \pm 0 . 0 6 1 1$ </td><td> $0 . 7 8 6 9 \pm 0 . 1 9 9 5$ </td><td> $0 . 8 5 4 3 \pm 0 . 1 5 4 9$ </td><td> $0 . 8 8 7 3 \pm 0 . 1 0 3 8$ </td><td> $0 . 0 5 8 5 \pm 0 . 0 5 4 2$ </td></tr><tr><td>FastKAN</td><td> $0 . 9 0 1 3 \pm 0 . 1 3 2 3$ </td><td> $0 . 9 4 8 0 \pm 0 . 0 6 3 0$ </td><td> $0 . 7 8 1 9 \pm 0 . 1 9 5 0$ </td><td> $0 . 8 6 2 4 \pm 0 . 1 3 9 3$ </td><td> $0 . 8 8 4 4 \pm 0 . 1 0 0 8$ </td><td> $0 . 0 6 0 0 \pm 0 . 0 5 5 9$ </td></tr><tr><td>ReLUKAN</td><td> $0 . 8 9 9 9 \pm 0 . 1 3 5 8$ </td><td> $0 . 9 4 6 4 \pm 0 . 0 6 4 2$ </td><td> $0 . 7 7 6 2 \pm 0 . 2 1 2 6$ </td><td> $0 . 8 5 8 1 \pm 0 . 1 4 7 0$ </td><td> $0 . 8 8 3 3 \pm 0 . 1 0 9 3$ </td><td> $0 . 0 6 0 6 \pm 0 . 0 5 5 8$ </td></tr><tr><td>TaylorKAN</td><td> $0 . 8 9 9 2 \pm 0 . 1 3 1 3$ </td><td> $0 . 9 4 7 0 \pm 0 . 0 6 3 4$ </td><td> $0 . 7 8 6 9 \pm 0 . 2 0 0 5$ </td><td> $0 . 8 6 3 7 \pm 0 . 1 3 4 0$ </td><td> $0 . 8 9 1 7 \pm 0 . 1 0 0 0$ </td><td> $0 . 0 6 0 6 \pm 0 . 0 5 6 0$ </td></tr><tr><td>WavKAN</td><td> $0 . 8 9 8 0 \pm 0 . 1 3 8 4$ </td><td> $0 . 9 4 7 1 \pm 0 . 0 6 4 6$ </td><td> $0 . 7 8 5 8 \pm 0 . 1 9 8 5$ </td><td> $0 . 8 5 9 7 \pm 0 . 1 5 2 0$ </td><td> $0 . 8 8 6 7 \pm 0 . 1 0 2 6$ </td><td> $0 . 0 5 9 1 \pm 0 . 0 5 6 9$ </td></tr><tr><td>GramKAN</td><td> $0 . 8 9 6 2 \pm 0 . 1 3 9 2$ </td><td> $0 . 9 4 5 1 \pm 0 . 0 6 7 0$ </td><td> $0 . 7 8 6 6 \pm 0 . 2 1 2 9$ </td><td> $0 . 8 6 8 1 \pm 0 . 1 3 0 9$ </td><td> $0 . 8 9 4 5 \pm 0 . 1 0 6 5$ </td><td> $0 . 0 6 1 5 \pm 0 . 0 5 8 9$ </td></tr><tr><td>JacobiKAN</td><td> $0 . 8 9 5 2 \pm 0 . 1 3 8 4$ </td><td> $0 . 9 4 3 4 \pm 0 . 0 6 7 9$ </td><td> $0 . 7 8 8 8 \pm 0 . 1 9 9 3$ </td><td> $0 . 8 6 8 8 \pm 0 . 1 3 2 3$ </td><td> $0 . 8 9 4 0 \pm 0 . 1 0 0 0$ </td><td> $0 . 0 6 2 4 \pm 0 . 0 5 8 3$ </td></tr><tr><td>RandomForest</td><td> $0 . 8 9 2 3 \pm 0 . 1 4 5 5$ </td><td> $0 . 9 4 8 4 \pm 0 . 0 6 1 0$ </td><td> $0 . 7 6 2 1 \pm 0 . 1 9 8 1$ </td><td> $0 . 8 4 5 8 \pm 0 . 1 5 7 8$ </td><td> $0 . 8 7 0 3 \pm 0 . 1 0 5 1$ </td><td> $0 . 0 6 6 5 \pm 0 . 0 5 2 3$ </td></tr><tr><td>ChebyKAN</td><td> $0 . 8 9 1 1 \pm 0 . 1 4 6 5$ </td><td> $0 . 9 3 9 4 \pm 0 . 0 7 8 9$ </td><td> $0 . 7 6 5 8 \pm 0 . 2 4 2 1$ </td><td> $0 . 8 4 4 2 \pm 0 . 1 7 7 9$ </td><td> $0 . 8 8 0 4 \pm 0 . 1 2 7 7$ </td><td> $0 . 0 6 3 4 \pm 0 . 0 6 2 8$ </td></tr><tr><td>LegendreKAN</td><td> $0 . 8 8 9 4 \pm 0 . 1 4 6 6$ </td><td> $0 . 9 4 2 0 \pm 0 . 0 7 3 0$ </td><td> $0 . 7 8 3 8 \pm 0 . 2 2 8 5$ </td><td> $0 . 8 5 5 9 \pm 0 . 1 7 1 0$ </td><td> $0 . 8 9 1 9 \pm 0 . 1 1 8 0$ </td><td> $0 . 0 6 1 1 \pm 0 . 0 6 2 8$ </td></tr></table>

Note: Acronyms and evaluation metrics: AUPRC = Area Under the Precision-Recall Curve; $\mathbf { A U R O C } = \mathbf { A r e a }$ Under the Receiver Operating Characteristic Curve; MCC = Matthews Correlation Coeficient; F1-Score = Macro-averaged harmonic mean of precision and recall; Balanced Accuracy = Macro-averaged arithmetic mean of sensitivity and specificity; Brier Score = Quadratic scoring rule measuring mean squared diference between predicted probabilities and binary outcomes (lower values indicate superior probability calibration). Values denote grand mean ± standard deviation aggregated across N = 105 cross-validation evaluations (15 independent folds across 7 benchmark datasets). Bold typography highlights the optimal mean score per metric column.

Table E.6: Computational Complexity and Latency Profiling (Grand Mean ± Std across all datasets and folds)
<table><tr><td>Architecture</td><td>Trainable Params</td><td>Inference (ms/sample)</td><td>Latency Ratio vs. MLP</td><td>Train Time (s/fold)</td></tr><tr><td>StandardMLP</td><td> $8 , 2 7 3 \pm 5 , 2 1 0$ </td><td> $0 . 0 6 1 1 \pm 0 . 0 1 6 5$ </td><td>1.00×</td><td> $1 . 2 2 4 0 \pm 1 . 1 2 2 6$ </td></tr><tr><td>ReLUKAN</td><td> $7 3 , 9 0 6 \pm 3 5 , 9 6 3$ </td><td> $0 . 0 7 3 5 \pm 0 . 0 2 5 6$ </td><td>1.20×</td><td> $2 . 0 4 3 4 \pm 1 . 6 1 0 9$ </td></tr><tr><td>FastKAN</td><td> $3 9 , 1 4 3 \pm 1 7 , 1 1 9$ </td><td> $0 . 0 7 4 5 \pm 0 . 0 2 4 4$ </td><td>1.22×</td><td> $1 . 6 5 6 6 \pm 0 . 9 3 5 8$ </td></tr><tr><td>TabKAN</td><td> $5 6 , 9 9 0 \pm 2 6 , 9 5 6$ </td><td> $0 . 0 7 5 2 \pm 0 . 0 2 6 0$ </td><td>1.23×</td><td> $2 . 0 3 6 7 \pm 1 . 5 1 3 1$ </td></tr><tr><td>TaylorKAN</td><td> $2 3 , 8 8 5 \pm 8 , 1 0 1$ </td><td> $0 . 0 8 1 6 \pm 0 . 0 2 5 4$ </td><td>1.34×</td><td> $1 . 8 6 2 1 \pm 1 . 5 2 1 5$ </td></tr><tr><td>WavKAN</td><td> $1 0 3 , 3 4 7 \pm 5 2 , 5 0 0$ </td><td> $0 . 0 8 2 6 \pm 0 . 0 2 7 3$ </td><td>1.35×</td><td> $2 . 2 3 0 9 \pm 2 . 3 6 9 8$ </td></tr><tr><td>ChebyKAN</td><td> $2 3 , 3 6 1 \pm 7 , 5 5 0$ </td><td> $0 . 0 8 8 1 \pm 0 . 0 2 9 7$ </td><td>1.44×</td><td> $1 . 6 4 1 6 \pm 1 . 3 8 4 3$ </td></tr><tr><td>GramKAN</td><td> $2 3 , 6 7 8 \pm 7 , 9 0 9$ </td><td> $0 . 0 8 8 2 \pm 0 . 0 2 8 5$ </td><td>1.44×</td><td> $1 . 8 6 1 8 \pm 1 . 5 9 2 1$ </td></tr><tr><td>LegendreKAN</td><td> $2 3 , 6 2 2 \pm 7 , 8 7 6$ </td><td> $0 . 0 8 8 5 \pm 0 . 0 2 9 5$ </td><td>1.45×</td><td> $1 . 7 3 0 2 \pm 1 . 3 7 6 1$ </td></tr><tr><td>JacobiKAN</td><td> $2 2 , 8 9 2 \pm 8 , 4 5 4$ </td><td> $0 . 0 9 0 6 \pm 0 . 0 2 9 8$ </td><td>1.48×</td><td> $1 . 6 0 6 0 \pm 1 . 2 5 7 3$ </td></tr><tr><td>RandomForest</td><td></td><td> $0 . 1 1 7 8 \pm 0 . 0 6 5 8$ </td><td>1.93×</td><td> $0 . 0 7 4 6 \pm 0 . 0 5 2 1$ </td></tr></table>

Note: Architectural complexity and runtime metrics: Trainable Params = Mean total count of learnable network parameters (|θ|); Inference = Mean per-sample inference latency in milliseconds (ms), measured with batch size 1 on Apple Silicon MPS hardware; Latency Ratio = Relative latency factor normalized to the StandardMLP baseline (1 00×); Train Time = Mean wall-clock training duration per cross-validation fold in seconds (s). Values denote grand means ± standard deviation across all N = 105 cross-validation folds.

Distribution of Metric AUROC in Cross Validatio  
![](images/dc610410eac7dc3d5e4c3278e9b96b16a3c679dacb15ef0ecc4e459c8a6f789c.jpg)

Distribution of Metric MCC in Cross Validation  
![](images/e53b6ea9b35547f4cab7fcea0f0238a6fe8223928a383caf49938312fb698a01.jpg)

Distribution of Metric F1\_SCORE in Cross Validation  
![](images/e1ddb8c18e2478862f4821fdad62822104551a45d0cd2b004f1855e75e042ce2.jpg)

Distribution of Metric INFERENCE\_TIME\_PER\_SAMPLE\_MS in Cross Validation  
![](images/ff2f9c953e6251d06c3e0de4627c4eb01b5de9e3b00151b589d59fea6f864026.jpg)  
Figure E.6: Gallery of cross-validation metric distributions for all evaluated architectures across seven medical datasets (from top to bottom): AUROC, MCC, macro-averaged F1-Score, and per-sample inference latency measured on Apple Silicon MPS hardware (log scale).

![](images/eb3ef92fc94babab7f70f4278760a219b5ceb7cf7c54d3ee9ad81445130aa454.jpg)

![](images/8fc3ffa7369838efbedf2896a696deb28a3c00c55710bbdde2a1b45a845506a6.jpg)

Data Scarcity Degradation: parkinsons (AUPRC)  
![](images/1f961916d8b10f88a7dc9238b986924bbab4be55f53c0fa68ec815eef8309cc2.jpg)

Data Scarcity Degradation: pima\_diabetes (AUPRC)  
![](images/5be5770a39e0e03e129b7d65a6550810b8bf295816082fd828a5acc28c01d8cc.jpg)

Data Scarcity Degradation: breast\_cancer (AUPRC)  
![](images/ad643ce37d7a0d82dc011fa6dca0cf0462b85e7bb99662d0af4e896602dafbd1.jpg)

Data Scarcity Degradation: heart\_disease (AUPRC)  
![](images/51d6d804efba1391ffe0b743c63191337ddcf38b6831a2a8331558ba1bf3b987.jpg)  
Figure E.7: Sample eficiency ablation curves illustrating AUPRC degradation as training data is reduced from 100% to 10% across six clinical cohorts (left to right, top to bottom): Cervical Cancer, Cardiotocography, Parkinson’s Disease, Pima Diabetes, Breast Cancer, and Heart Disease. Shaded bands denote ±1 standard deviation across folds.

Learning Curve with standard deviation Dataset: cervical\_cancer | Model: StandardMLP  
![](images/20a04e2ec79be50d3eeff579433499d723aed162da177489e8c4da6155e5f73e.jpg)

![](images/4faaccabd71c9c0796817605202abd95c6a8ad3980e03908db4eb40c430cd2f3.jpg)

Learning Curve with standard deviation Dataset: cardiotocography | Model: StandardMLP  
![](images/9804c9e371d9fbd20c8de7bfc40ef81a1d4392e8270c2b48d2914ba378a148a8.jpg)

Learning Curve with standard deviation Dataset: cardiotocography | Model: TabKAN  
![](images/a884cfef086b9870fc783c3491741dcdb28f539575c3be49c7c60730f0534258.jpg)

Learning Curve with standard deviation Dataset: heart\_disease | Model: StandardMLP  
![](images/13550d5a9c8aab05e9252ef6328c75a01f3c21ad77cba6b674c5b6b961a2d806.jpg)

Learning Curve with standard deviation Dataset: heart\_disease | Model: TaylorKAN  
![](images/18d4d1ebef67110e066d461457b4245ef8a63667144cef6abc592ab0132d7815.jpg)

Learning Curve with standard deviation Dataset: breast\_cancer | Model: StandardMLP  
![](images/1fef345ce0af38cb1faf86b01312a49fd971932c63d55c920471748c05f743d9.jpg)

Learning Curve with standard deviation Dataset: breast\_cancer | Model: TabKAN  
![](images/bace46e485af7b1b24da58d93f700f1f0cc887b4e7af0016fa770ae4a81af9fc.jpg)  
Figure E.8: Training and validation loss trajectories across four distinct clinical datasets, comparing baseline StandardMLP against representative KAN variants (separate columns). Rows from top to bottom: Cervical Cancer (6 4% positive prevalence), Cardiotocography (8 3% minority class), Heart Disease, and Breast Cancer.

![](images/bc46dc2cbb01163d949db9ce8c695144556e1566b1d3a3ffe1ffe5407de2d207.jpg)

![](images/5a2ef151e49724a749d59d9108f7b3e2c04ea53df4c466b923cb3ed45cd8d9d7.jpg)  
Sum of Confusion Matrices Dataset: cardiotocography | Model: WavKAN

Sum of Confusion Matrices Dataset: cardiotocography | Model: StandardMLP  
![](images/7241da87ca214e3b44f154c4e12a4632eaf09bbf565f1541f959eac84ab72ed9.jpg)

![](images/cfc9000c0a8a1469d450d31f712c0509cef1ca1ac9f632e56b9c0b62cef44774.jpg)

Sum of Confusion Matrices Dataset: pima\_diabetes | Model: StandardMLP  
![](images/389966715f5462423fa775b70ebb428ea8885457b263d6e103a6d2ebbab66d52.jpg)

Sum of Confusion Matrices Dataset: pima\_diabetes | Model: TabKAN  
![](images/53b33a5b1fe06a33b79484fc520d312d6de9cf1a57bb6b777f908f3289f84fab.jpg)

Sum of Confusion Matrices Dataset: breast\_cancer | Model: StandardMLP  
![](images/1c6b30d7baf4c4dc9e226239c46478e2add8aa47659809a11ad1dada3e0eb83b.jpg)

![](images/8d2f910bf341922eb411259f900543dc04e16bca1b8ffd7195fe3d89b45658d5.jpg)  
Figure E.9: Confusion Matrix Gallery comparing StandardMLP against representative KAN variants (separate columns). Rows from top to bottom: Cervical Cancer (6 4% positive prevalence), Cardiotocography (8 3% minority class), Pima Diabetes, and Breast Cancer.