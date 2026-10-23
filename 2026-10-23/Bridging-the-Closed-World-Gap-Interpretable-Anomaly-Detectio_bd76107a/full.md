# Bridging the Closed-World Gap: Interpretable Anomaly Detection via a Hybrid TSK-FCM Ensemble on the BETH Dataset

Lohith Srikanth Pentapalli , Josette Riep , Justin Ouwerkerk, and Kelly Cohen

University of Cincinnati

pentapls@mail.uc.edu, riepjr@ucmail.uc.edu, ouwerkjn@ucmail.uc.edu, cohenky@ucmail.uc.edu

Abstract. A fundamental paradox plagues the deployment of Machine Learning in cybersecurity: supervised models generalize poorly to novel, Out-of-Distribution (OoD) attacks, whereas unsupervised models often sufer from poor separability, leading to unmanageable false-alarm rates. This challenge is exemplified by the BETH dataset, a kernel-level benchmark in which the training data consists solely of benign and “suspicious” insider behaviors, whereas the test set consists almost entirely of novel botnet activation attacks. To bridge this detection gap, we propose a Hybrid Fuzzy Ensemble that synergizes the precision of supervised learning with the adaptability of unsupervised clustering. Our architecture integrates a gradient-optimized Takagi-Sugeno-Kang (TSK) fuzzy classifier to act as an expert on known insider threats, with a parallel Fuzzy C-Means (FCM) module to detect geometric anomalies. A critical finding of our research is that traditional fuzzy membership scores degrade in high-dimensional feature spaces due to the curse of dimensionality. We overcome this by introducing a robust Min Distance metric that exploits the raw geometric isolation of botnet activation attacks. This approach yields a state-of-the-art AUROC of 0.9951 on the BETH test set, outperforming the original Isolation Forest baseline by 17.1%. By combining near-perfect OoD detection with the intrinsic interpretability of fuzzy logic, our framework ofers a viable path toward robust, transparent, and autonomous network defense.

Keywords: Anomaly Detection · Intrusion Detection Systems (IDS) · TSK Fuzzy Systems · Out-of-Distribution Data · Explainable AI · Hybrid Ensemble

## 1 Introduction

The evolution of cyber threats has necessitated the use of increasingly sophisticated systems to prevent and mitigate the organizational impact of cyberattacks. While Machine Learning (ML) based Intrusion Detection Systems (IDS) have shown promise, they face a fundamental dichotomy: supervised models excel at detecting known threats but fail catastrophically against zero-day exploits (closed-world assumption), while unsupervised models can detect anomalies but often sufer from prohibitive false alarm rates due to a lack of semantic context [1].

This challenge is clearly demonstrated by the BETH dataset [2], a modern cybersecurity benchmark collected from real-world honeypots. Unlike legacy datasets derived from synthetic trafic (e.g., NSL-KDD), BETH captures raw kernel-level system calls and exhibits a strict distributional shift: the training set contains only benign trafic and “suspicious” insider anomalies, while the test set introduces entirely novel “evil” botnet activation [2]. This structure mimics the hardest reality of cybersecurity: an IDS must defend against attack vectors it has never seen before.

Standard supervised classifiers, including Deep Neural Networks, struggle with the BETH dataset because the “evil” class is Out-of-Distribution (OoD) relative to the training manifold [2]. Conversely, purely unsupervised methods often fail to distinguish between rare benign administrative tasks and genuine malicious behavior. To bridge this gap, we propose a Hybrid Fuzzy Ensemble that synergizes the strengths of both paradigms.

Our architecture integrates a gradient-optimized Takagi-Sugeno-Kang (TSK) fuzzy system with a Fuzzy C-Means (FCM) anomaly detector. The TSK module acts as a supervised expert for known insider threats, providing interpretable rule-based detection. Simultaneously, the FCM module constructs a geometric manifold of normality, flagging novel botnet activation attacks based on their raw Euclidean distance from benign clusters.

Our proposed hybrid architecture yields an AUROC of 0.9951, surpassing the state-of-the-art Isolation Forest [1] baseline by 17.1% while maintaining high operational interpretability.

## 2 Background

Contemporary Intrusion Detection Systems (IDS) exhibit a fundamental weakness when exposed to Out-of-Distribution (OoD) data, particularly zero-day attacks that deviate from the training distribution. Supervised learning models, including Random Forests and Deep Neural Networks (DNNs), operate under a closed-world assumption and rely on rigid decision boundaries derived from labeled data. While efective for known attack signatures, these models degrade sharply when confronted with novel threats, often producing confident misclassifications due to poor generalization beyond the training manifold [3].

This limitation is especially pronounced in kernel-level telemetry environments such as those represented by the BETH dataset. In such settings, anomalous behavior is more accurately characterised as a distributional shift driven by unseen adversarial techniques rather than isolated statistical outliers [2]. Streambased unsupervised learning approaches have therefore been shown to be more suitable, as they adapt to concept drift and can identify emergent process behaviors, such as unexpected system call sequences, that supervised baselines fail to detect [4]. Among these approaches, clustering-based methods have demonstrated particular promise. Fuzzy C-Means (FCM) clustering outperforms hard partitioning techniques by allowing soft membership assignments, enabling the identification of suspicious instances that reside in sparse or ambiguous regions of the feature space. This capacity to explicitly model uncertainty makes FCM well suited for zero-day attack detection, where class labels and attack semantics are unknown [5].

As IDS models increase in complexity, explainability has emerged as a critical operational requirement. Black-box deep learning models typically provide probability scores without semantic justification, which limits analyst trust and slows response in Security Operations Centers (SOCs) [6]. In contrast, Fuzzy Logic systems ofer intrinsic interpretability through rule-based reasoning expressed in linguistic terms, closely approximating human expert judgment. This transparency allows analysts to trace alerts directly to the conditions and features that triggered them, improving trust and operational usability.

Recent advances have significantly reduced the performance gap between fuzzy systems and deep learning. Gradient-optimized Takagi–Sugeno–Kang (TSK) fuzzy frameworks leverage modern optimization techniques to fine-tune membership functions, achieving near-perfect classification performance in security tasks while retaining full rule-level explainability [7]. These findings support the use of TSK models as white-box detection engines capable of delivering both high accuracy and transparent decision logic in high-stakes cybersecurity environments. Additionally, TSK fuzzy systems have been proven to be universal approximators [8].

The literature further indicates that hybrid architectures provide an efective solution to the trade-of between detection performance and false-alarm rate. Purely supervised models tend to miss novel attacks, whereas unsupervised detectors often overproduce false positives. Hybrid systems that combine multiple learning paradigms have been shown to significantly reduce false alarm rates by leveraging complementary strengths across models [9]. In fuzzy-based systems, initializing supervised or rule-based learners with clustering-derived structure has been demonstrated to improve convergence stability and prevent suboptimal local minima during optimization [10]. Large-scale ensemble frameworks further validate this approach, achieving extremely low false alarm rates by combining weak learners with probabilistic calibration mechanisms [11].

Overall, the literature supports a hybrid paradigm in which unsupervised learning discovers latent structure and emergent attack patterns in data, while supervised or rule-based components refine decision boundaries and provide interpretable explanations. In this context, combining FCM with a gradientoptimized TSK fuzzy system ofers a principled approach to minimizing both missed detections and false alarms while maintaining operational transparency.

## 3 Methodology

We propose a three-stage hybrid fuzzy ensemble architecture that combines supervised detection of known suspicious behaviors with unsupervised identification of out-of-distribution (OoD) malicious events. The system consists of a gradient-optimized Takagi-Sugeno-Kang (TSK) fuzzy classifier for supervised detection and a Fuzzy C-Means (FCM) clustering module for unsupervised OoD detection. This dual-module approach addresses the fundamental challenge in botnet activation detection: identifying novel attack patterns that were not present during training.

## 3.1 Data Preprocessing

The BETH dataset [2] consists of kernel log traces captured from 23 honeypot networks during normal operation and botnet infection periods. Each system call event is represented by 14 raw features, including process identifiers, user contexts, and system call parameters.

Following the BETH benchmark methodology, we extract 9 discriminative features for our detection system:

Process Features: processId, parentProcessId

– User Context: userId, mountNamespace

– System Call Context: eventId, argsNum, returnValue

– Categorical Features: processName, hostName

To handle the heterogeneous nature of system call features, we apply domainspecific transformations:

1. Binary Encoding: Following security best practices, process and user identifiers are binarized based on privilege levels:

$$
\mathrm { p r o c e s s I d ^ { \prime } } = \mathbf { 1 } [ \mathrm { p r o c e s s I d } \in \{ 0 , 1 , 2 \} ]\tag{1}
$$

$$
\mathrm { \ u s e r I d ^ { \prime } } = \mathbf { 1 } [ \mathrm { u s e r I d } < 1 0 0 0 ]\tag{2}
$$

where system-critical processes (PIDs 0-2) and system users $\left( \mathrm { U I D } < 1 0 0 0 \right)$ are flagged.

2. Categorical Encoding: Process and host names are encoded using label encoding to preserve ordinal relationships discovered during training.

3. Return Value Mapping: System call return values are mapped to three semantic categories:

$$
\mathrm { r e t u r n V a l u e } ^ { \prime } = \left\{ \begin{array} { l l } { - 1 } & { \mathrm { i f ~ e r r o r ~ ( \it ~ < ~ 0 ) } } \\ { 0 } & { \mathrm { i f ~ s u c c e s s ~ ( \it ~ = ~ 0 ) } } \\ { 1 } & { \mathrm { i f ~ d a t a ~ r e t u r n e d ~ ( \it ~ > ~ 0 ) } } \end{array} \right.\tag{3}
$$

4. Standardization: All features are normalized using z-score standardization using the training data and applied consistently to validation and test data.

## 3.2 System Architecture

Stage 1: Supervised TSK Fuzzy Classifier – The first detection module is a gradient-optimized first-order Takagi-Sugeno-Kang fuzzy inference system trained on labeled suspicious behaviors. Building upon our prior work on explainable phishing detection [7], we adapt the TSK architecture to the botnet detection domain.

Each input feature $x _ { i }$ is fuzzified using Gaussian membership functions:

$$
\mu _ { A _ { i j } } ( x _ { i } ) = \exp \left( - \frac { ( x _ { i } - c _ { i j } ) ^ { 2 } } { 2 \sigma _ { i j } ^ { 2 } } \right)\tag{4}
$$

where $c _ { i j }$ and $\sigma _ { i j }$ are the center and width parameters of the $j { \mathrm { - t h } }$ membership function for feature i. These parameters are initialized using k-means clustering on training data and subsequently optimized via gradient descent.

For each rule $k ,$ the firing strength $\alpha _ { k }$ is computed using the product t-norm:

$$
\alpha _ { k } = \prod _ { i = 1 } ^ { n } \mu _ { A _ { i k } } ( x _ { i } )\tag{5}
$$

where $A _ { i k }$ denotes the antecedent membership function selected for feature i in rule k.

Each rule has a linear consequent function:

$$
f _ { k } ( { \mathbf { x } } ) = b _ { k } + \sum _ { i = 1 } ^ { n } w _ { k i } x _ { i }\tag{6}
$$

The final TSK output is the normalized weighted average:

$$
y _ { T S K } = \frac { \sum _ { k = 1 } ^ { R } \alpha _ { k } f _ { k } ( \mathbf { x } ) } { \sum _ { k = 1 } ^ { R } \alpha _ { k } + \epsilon }\tag{7}
$$

where $\epsilon = 1 0 ^ { - 6 }$ prevents division by zero. This output is passed through a sigmoid activation function $\sigma ( \cdot )$ to produce a probability score $P _ { s u s } \in [ 0 , 1 ]$

We formulate the entire TSK system as a diferentiable computation graph. We jointly optimize all antecedent parameters $\{ c , \sigma \}$ and consequent parameters $\{ w , b \}$ using the Adam optimizer to minimize the Binary Cross Entropy (BCE) loss:

$$
\mathcal { L } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left[ y _ { i } \log ( P _ { s u s } ^ { ( i ) } ) + ( 1 - y _ { i } ) \log ( 1 - P _ { s u s } ^ { ( i ) } ) \right] + \lambda \| \mathbf { w } \| _ { 2 } ^ { 2 }\tag{8}
$$

The inclusion of $L _ { 2 }$ regularization on the consequent weights prevents overfitting and ensures smooth decision boundaries. The optimization is performed with weighted loss to handle class imbalance, setting the positive class weight proportional to the class ratio in the training data. We used the Adam optimizer with a learning rate of 0.01 trained over 10 epochs. The Adam optimizer was chosen due to its adaptive learning rates and stable convergence properties [12].

Stage 2: Unsupervised FCM OoD – The second module addresses the fundamental limitation of supervised approaches: detecting novel attack patterns absent from training data. We employ Fuzzy C-Means clustering trained exclusively on benign system call patterns to model the “manifold of normality.”

Critically, the FCM module is trained only on samples labeled as non-suspicious $( \mathrm { s u s } = 0 )$ in the training set. This aligns with the true out-of-distribution detection paradigm where malicious samples represent genuinely novel patterns not seen during training.

The FCM algorithm partitions the benign training data $\{ \mathbf { x } _ { 1 } , \dotsc , \mathbf { x } _ { N } \}$ into c fuzzy clusters by minimizing:

$$
J _ { m } = \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { c } u _ { i j } ^ { m } \| \mathbf x _ { i } - \mathbf v _ { j } \| ^ { 2 }\tag{9}
$$

subject to the constraint $\textstyle \sum _ { j = 1 } ^ { c } u _ { i j } = 1$ for all $i ,$ where:

$- \textbf { v } _ { j } \in \mathbb { R } ^ { n }$ is the centroid of cluster $j$

$- \ u _ { i j } \in [ 0 , 1 ]$ is the membership degree of sample i to cluster $j$

$- \ m = 2$ is the fuzzification parameter controlling cluster overlap

In standard fuzzy anomaly detection, the "outlierness" of a sample is often defined as $1 - \operatorname* { m a x } ( u _ { i j } )$ . However, our empirical analysis reveals that this metric underperforms in high-dimensional spaces due to the "curse of dimensionality." As the volume of the feature space expands, OoD samples often become equidistant to multiple cluster centers, resulting in low but uniform membership distributions that are indistinguishable from benign boundary points.

To overcome the limitations of membership scores, we propose a geometric anomaly metric based on raw Euclidean distance. We define the OoD score $S _ { O o D }$ as the distance to the nearest benign prototype:

$$
S _ { O o D } ( \mathbf { x } ) = \operatorname* { m i n } _ { j \in \{ 1 \ldots c \} } \| \mathbf { x } - \mathbf { v } _ { j } \| _ { 2 }\tag{10}
$$

This metric leverages the geometric concentration of measure. Benign samples, by definition, cluster tightly around the learned centroids $\mathbf { v } _ { j }$ . Malicious botnet attacks, which generate anomalous system call sequences (e.g., rapid process forking or unauthorized file access), manifest as geometric outliers far removed from any benign cluster. By measuring the raw Euclidean distance, we bypass the normalization efects of the softmax/membership function and directly quantify the geometric isolation of the attack.

Stage 3: Hybrid Fuzzy Aggregation – The final decision logic combines the supervised probability $P _ { s u s }$ and the unsupervised distance score $S _ { O o D }$ into a unified threat score.

Since $P _ { s u s } \in [ 0 , 1 ]$ and $S _ { O o D } \in [ 0 , \infty )$ , they cannot be compared directly. We employ a data-driven calibration step to map the distance score to a probability space. We fit a Sigmoid function to the distances observed in the validation set:

$$
P _ { O o D } ( \mathbf { x } ) = \frac { 1 } { 1 + \exp ( - \alpha ( S _ { O o D } ( \mathbf { x } ) - \beta ) ) }\tag{11}
$$

where $\beta$ is the median distance of benign validation samples, and α is a scaling factor derived from the interquartile range.

The final ensemble prediction is derived using a fuzzy T-Conorm (Maximum) operator:

$$
Y _ { F i n a l } ( \mathbf { x } ) = \operatorname* { m a x } ( P _ { s u s } ( \mathbf { x } ) , P _ { O o D } ( \mathbf { x } ) )\tag{12}
$$

This "Max-Dominance" strategy creates a logical OR gate between the two experts:

– If the sample resembles a known insider threat, $P _ { s u s }$ will be high $( > 0 . 5 )$ triggering a detection.

– If the sample is a novel botnet attack, it will be geometrically distant, causing $P _ { O o D }$ to be high $( > 0 . 5 )$ , triggering a detection.

This hybrid architecture ensures robust coverage of the threat landscape, allowing the system to detect specific known threats with high precision while maintaining the generalization capability required for zero-day defense.

## 4 Results and Discussion

In this section, we present a comprehensive evaluation of the proposed Hybrid TSK-FCM Ensemble. We analyze the performance of the constituent modules, compare the ensemble against state-of-the-art baselines, and provide a theoretical justification for the eficacy of the proposed geometric OoD detection metrics.

## 4.1 Validation of Supervised TSK Component

The first phase of our experimental pipeline focused on validating the supervised TSK component. Since the BETH training and validation sets contain only benign trafic and “Suspicious” (insider threat) anomalies, this phase was critical to ensure that the fuzzy rule-base correctly learned the signatures of known administrative misuse before attempting to detect novel botnet attacks.

We evaluated two configurations of the TSK model to analyze the impact of rule-base complexity on detection precision:

1. Compact Model: utilizing only the top 10 generated fuzzy rules.

2. Full Model: utilizing the complete set of generated rules.

The performance comparison on the validation set is presented in Table 1.

Table 1: Performance of Gradient-Optimized TSK Model on Validation Data (Target: Suspicious). While both models achieve perfect recall, the full rule-base is required to minimize false positives.
<table><tr><td>Configuration</td><td></td><td>Recall Precision</td><td>FPR AUROC</td></tr><tr><td>Top 10 Rules</td><td>1.0000</td><td>0.1149</td><td>3.20% 0.9839</td></tr><tr><td>All Rules (Selected) 1.0000</td><td></td><td>0.4634</td><td>0.48% 1.0000</td></tr></table>

Precision-Recall Analysis: As demonstrated in Table 1, both configurations achieved a perfect Recall of 1.0, successfully identifying every single instance of suspicious insider activity (786 samples). This confirms that the TSK gradient optimization process successfully converged and captured the fundamental signatures of the “Suspicious” class.

However, a significant divergence was observed in precision. The compact “10 Rules” model sufered from a high False Positive Rate (FPR) of 3.2%, flagging over 6,000 benign samples as anomalous. This suggests that while a small rule set is suficient for detection, it lacks the granularity required for discrimination. By utilizing the full rule set, the model successfully refined its decision boundaries, reducing the FPR to 0.48% (910 samples). Consequently, the “All Rules” model was frozen and selected as the expert supervised component for the ensemble.

## 4.2 Ensemble Performance & Baseline Comparison

We subsequently evaluated the full Hybrid Ensemble on the BETH test set. This dataset introduces the “Evil” class (setting up a botnet), representing Outof-Distribution (OoD) threats that were completely absent during the training phase.

Comparison with State-of-the-Art: To quantify the contribution of our hybrid approach, we compared the ensemble against three baselines: the supervised TSK-only model, the unsupervised FCM-only model, and the Isolation Forest baseline established in the original BETH study [2].

Figure 1 presents the Receiver Operating Characteristic (ROC) curves for all evaluated models.

The proposed ensemble achieved a Test AUROC of 0.9951, hugging the topleft corner of the plot. This represents a massive performance leap compared to the baselines:

vs. Isolation Forest (0.850): An improvement of +17.1%. This highlights the limitations of standard anomaly detection algorithms when applied to the high-dimensional, noisy feature space of the BETH dataset.

vs. TSK Only (0.932): An improvement of +6.8%. The supervised TSK model performs admirably (beating the Isolation Forest AUROC by 9.6%) but plateaus because it relies solely on finding “Suspicious” signatures. It misses novel botnet invocation behaviors that do not resemble insider threats. The ensemble closes this gap by activating the unsupervised OoD module.

Detailed Classification Analysis: Table 2 details the confusion matrix for the ensemble on the test set.

Attack Detection (Recall): The most critical metric for a NIDS (Network Intrusion Detection System) is Recall (Sensitivity), as missing an active botnet attack can lead to total network compromise. The ensemble achieved a Recall of 0.9930, correctly identifying 157,329 out of 158,432 malicious samples. This confirms that the geometric OoD detector is highly efective at flagging novel threats.

![](images/dae39d6836c51975ae3c651a97a3ad7665265a8a13991e270c42b22e709b7afd.jpg)  
Fig. 1: ROC Curves Comparison. The proposed Ensemble (Green) achieves an AUROC of 0.9951, significantly outperforming the supervised TSK baseline (Blue, 0.9318) and the original BETH Isolation Forest baseline. The curve demonstrates that the ensemble maintains a high True Positive Rate even at very low False Positive thresholds.

Table 2: Confusion Matrix of the Hybrid Ensemble on the Test Set (Target: Evil). The model successfully identifies over 99.3% of unknown botnet activation attacks.
<table><tr><td rowspan="2">Actual</td><td colspan="3">Predicted |Benign Malicious</td></tr><tr><td>Benign Malicious</td><td>25,862 1,103</td><td>4,673 157,329</td></tr></table>

False Alarm Analysis: The model generated 4,673 false positives, resulting in an FPR of approximately 15.3%. While higher than the validation FPR, this is an expected consequence of distribution shift in the test set. Crucially, the high AUROC (0.995) indicates that these false positives are not randomly distributed but are likely clustered near the decision boundary. This implies that the FPR can be drastically reduced by a slight adjustment of the decision threshold without significantly impacting recall, a property we analyze in the following section.

## 4.3 Mechanism Analysis & Ablation Study

To provide a theoretical justification for the ensemble’s performance, we conducted an ablation study to analyze the impact of the specific fuzzy metrics used for OoD detection.

The Failure of Membership in High Dimensions: A key finding of our research is the failure of traditional fuzzy membership scores in high-dimensional anomaly detection. As shown in Table 3, metrics based on partition coeficients (Max Membership and Fuzzy Entropy) plateaued at an AUROC of ≈ 0.87, failing to match the performance of the proposed distance metric.

Table 3: Impact of Ensemble OoD Metric on Detection Performance (Averaged across $c \in [ 6 , 1 8 ]$ , where c is the number of clusters). The geometric Min Distance metric significantly outperforms probabilistic membership metrics.
<table><tr><td>Metric</td><td>Mean AUROC</td><td>Peak AUROC</td><td>Std. Dev.</td></tr><tr><td>Max Membership</td><td>0.8798</td><td>0.9003</td><td>0.0106</td></tr><tr><td>Fuzzy Entropy</td><td>0.8732</td><td>0.8736</td><td>0.0003</td></tr><tr><td>Min Distance</td><td>0.9950</td><td>0.9956</td><td>0.0007</td></tr></table>

We hypothesize that this is due to the “Curse of Dimensionality.” In highdimensional spaces, the volume of the feature space increases exponentially, causing data points to become sparse. As a result, an OoD outlier often ends up being equidistant from multiple cluster centers, leading to low but uniform membership degrees (e.g., 0.1 membership in all 10 clusters). A metric like Max Membership fails to distinguish this state from a benign boundary point.

In contrast, the Min Distance metric measures the absolute Euclidean distance. Even if an outlier is equidistant to multiple clusters, its absolute distance to them is massive. This geometric property allows the distance metric to maintain separability where probabilistic membership fails.

Visual Confirmation of Geometric Isolation: This hypothesis is visually confirmed in Figure 2.

![](images/3c73201f6f974db255a7a32dc3d9ced93e412b1eb4935d4f2371f76ba2f2090a.jpg)  
(a) Score Density: Clear bimodal separation.

![](images/58856acb91dd65e45df3e529b6950cfcdf6a4f698ef71c60c29ed42da381c2a7.jpg)  
(b) Raw Distance (Log): Attacks are outliers.  
Fig. 2: Geometric Mechanism. (a) The score distribution shows a clear “empty valley” between benign and malicious modes. (b) The box plot reveals that attacks are massive geometric outliers, with distances $( > 1 0 ^ { 2 } )$ orders of magnitude larger than benign samples.

Figure 2b plots the raw distance to the nearest cluster center on a logarithmic scale. The separation is stark: benign samples consistently reside within a distance of 10<sup>1</sup>, while malicious samples cluster above 10<sup>2</sup>. This order-of-magnitude diference creates the wide “safety margin” observed in the score density plot (Fig. 2a), allowing for robust thresholding.

## 4.4 Operational Implications

The results of this study have significant implications for the deployment of NIDS in real-world Security Operations Centers (SOCs).

Threshold Sensitivity: The stability of the Min Distance metric across cluster counts $( c \in [ 6 , 1 8 ] )$ , as shown in our sensitivity analysis, implies that the model is not hyper-sensitive to initial configuration. This reduces the burden of hyperparameter tuning for security analysts.

Alert Fatigue Mitigation: While the test FPR of 15.3% is non-negligible, the bimodal nature of the anomaly scores (Fig. 2a) suggests that this can be managed via “Alert Grading.” A graded approach leverages the high certainty of the geometric distance metric to prioritize analyst attention on the most likely threats, efectively mitigating alert fatigue.

## 5 Conclusion and Future Work

In this work, we addressed the critical challenge of detecting Out-of-Distribution (OoD) botnet activation attacks in kernel-level system call data. Using the BETH dataset, we demonstrated that traditional supervised models, while efective against known insider threats, fail to generalize to novel attack vectors absent from the training distribution. Similarly, we showed that standard fuzzy membership metrics degrade in high-dimensional feature spaces, limiting the eficacy of unsupervised anomaly detection.

To bridge this gap, we proposed a Hybrid Fuzzy Ensemble that synergizes a gradient-optimized TSK expert for known threats with a geometric OoD detector for novel anomalies. Our key findings are threefold:

1. Geometric Superiority: We empirically validated that the proposed Min Distance metric (AUROC 0.9951) significantly outperforms probabilistic membership scores (AUROC 0.8798), proving that botnet attacks manifest as massive geometric outliers rather than merely uncertain boundary points.

2. State-of-the-Art Performance: The ensemble achieved a Test AUROC of 0.9951, surpassing the original Isolation Forest baseline by 17.1% and detecting 99.3% of zero-day botnet attacks.

3. Operational Robustness: Sensitivity analysis confirmed that the unsupervised component is stable across hyperparameter configurations $( c \in [ 6 , 1 8 ] )$ , reducing the tuning burden for deployment.

Future Work: While our approach demonstrates robust detection capabilities, several avenues for future research remain. First, we plan to evaluate the system’s resilience against adversarial examples, specifically investigating whether gradient-based attacks can perturb malicious samples to minimize their geometric distoance to benign clusters. Second, to address concept drift in dynamic IoT environments, we aim to extend the FCM module with online learning capabilities, allowing the benign manifold (cluster centroids) to adapt incrementally to legitimate behavioral shifts without full retraining. Finally, we intend to develop a linguistic explanation generator for the OoD module, translating raw distance vectors into human-readable alerts to further enhance SOC analyst trust.

## References

1. Zoppi, T., Ceccarelli, A., Capecchi, T., Bondavalli, A.: Unsupervised Anomaly Detectors to Detect Intrusions in the Current Threat Landscape. ACM/IMS Trans. Data Sci. 2(2), 7:1–7:26 (2021). https://doi.org/10.1145/3441140

2. Highnam, K., Arulkumaran, K., Hanif, Z., Jennings, N.R.: Beth dataset: Real cybersecurity data for anomaly detection research. ICML Workshop on Uncertainty and Robustness in Deep Learning (2021)

3. Rakine, I., Oukaira, A., El Guemmat, K., Atouf, I., Ouahabi, S., Talea, M., Bouragba, T.: Comprehensive Review of Intrusion Detection Techniques: ML and DL in Diferent Networks. IEEE Access 13, 104,345–104,367 (2025). https: //doi.org/10.1109/ACCESS.2025.3579990

4. Eremin, E.: Unsupervised anomaly detection on cybersecurity data streams: A case with BETH dataset (2025). https://doi.org/10.48550/arXiv.2503.04178

5. Sahoo, S.K., Pattanaik, P., Mohanty, M.N.: Modified Fuzzy C-Means Clustering for Anomaly Detection in Bio-medical Data. International Journal of Intelligent Systems and Applications in Engineering 12(17s), 519–526 (2024)

6. Iantorno, M., Beladda, K.: Fuzzy Logic for Cybersecurity: Intrusion Detection and Privacy Preservation with Synthetic Data. In: 17th International Conference on Agents and Artificial Intelligence, pp. 376–382 (2025)

7. Pentapalli, L.S., Salisbury, J., Riep, J., Cohen, K.: A Gradient-Optimized TSK Fuzzy Framework for Explainable Phishing Detection (2025). https://doi.org/ 10.48550/arXiv.2504.18636

8. Ying, H.: General Takagi-Sugeno fuzzy systems are universal approximators. In: 1998 IEEE International Conference on Fuzzy Systems Proceedings. IEEE World Congress on Computational Intelligence (Cat. No.98CH36228), vol. 1, pp. 819–823 vol.1 (1998). https://doi.org/10.1109/FUZZY.1998.687596

9. Hariharan, S., Annie Jerusha, Y., Suganeshwari, G., Syed Ibrahim, S.P., Tupakula, U., Varadharajan, V.: A Hybrid Deep Learning Model for Network Intrusion Detection System Using Seq2Seq and ConvLSTM-Subnets. IEEE Access 13, 30,705– 30,721 (2025). https://doi.org/10.1109/ACCESS.2025.3541399

10. de Campos Souza, P.V., Guimarães, A.J., Rezende, T.S., Silva Araujo, V.J., Araujo, V.S.: Detection of Anomalies in Large-Scale Cyberattacks Using Fuzzy Neural Networks. AI 1(1), 92–116 (2020). https://doi.org/10.3390/ai1010005

11. Abou Elasaad, M.M., Sayed, S.G., El-Dakroury, M.M.: AegisGuard: A Multi-Stage Hybrid Intrusion Detection System with Optimized Feature Selection for Industrial IoT Security. Sensors (Basel, Switzerland) 25(22), 6958 (2025). https://doi.org/ 10.3390/s25226958

12. Kingma, D.P., Ba, J.: Adam: A Method for Stochastic Optimization (2017). https: //doi.org/10.48550/arXiv.1412.6980