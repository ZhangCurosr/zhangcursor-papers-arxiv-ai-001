# Coverage-Aware Virtual IMU Augmentation for Low-Resource Human Activity Recognition

Jiayuan Gao, Yingwei Zhang, Ziyao Tang, Yuejia Ma, Yuanzhe Chen, Shuchao Song, and Boshi Tang

This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

Abstract—IMU-based human activity recognition (HAR) enables continuous, privacy-friendly monitoring of daily activities using wearable sensors. However, building reliable HAR models that generalize across diverse users and real-world conditions requires large amounts of labeled IMU data, which are expensive and difficult to collect. Existing approaches mainly rely on augmentation or synthesis to expand available data, but indiscriminately adding virtual samples may provide little new coverage and introduce unreliable supervision. To overcome these challenges, we propose a novel coverage-aware virtual IMU augmentation framework that decides where to supplement real data, how to generate and select virtual candidates, and how strongly to weight them during training. Specifically, we select diversity and scarcity anchors in a learned sensor embedding space, convert anchor dynamics into prompts, and generate virtual IMU candidates for each anchor. We then rank candidates by a selection cost combining anchor proximity and label consistency, and incorporate the selected candidates into HAR training with reliability-based weights. Experiments on public HAR benchmarks show that our method consistently improves recognition performance over competitive baselines, and ablation studies confirm the effectiveness of the proposed framework design.

Index Terms—Human activity recognition, wearable sensing, synthetic IMU data, data augmentation, large language models.

## I. INTRODUCTION

IMU-based human activity recognition (HAR) recognizes daily activities from signals collected by wearable inertial measurement units [1]. It supports applications such as daily activity monitoring, fitness assistance, and home-based rehabilitation [2]. Compared with camera-based systems, wearable IMUs reduce visual privacy concerns and support continuous monitoring in everyday environments [3]. However, reliable HAR models require training data that capture important variations across users, activity execution styles, sensor placements, and sensing environments. A finite labeled dataset only provides an incomplete view of the activity distribution, and collecting data across these sources of variation is costly and often constrained by time, supervision, and deployment conditions.

![](images/52e816da89b0cd4efc2e18a2bef5ed9a9d03e5326a25aa0e604745a092e7aa27.jpg)  
Fig. 1. Motivation for coverage-aware virtual IMU augmentation. Limited real IMU collections provide reliable but incomplete observations of activity patterns, whereas virtual IMU data can expand training coverage but may introduce shifted or unreliable samples.

As illustrated in Fig. 1, virtual IMU data can expand training coverage beyond what a limited real IMU collection can provide. Since such data can be generated at scale and conditioned on target activities, they provide a practical way to supplement limited real training data [4], [5]. Recent studies on diffusion-based dataset distillation also suggest that synthetic samples should be evaluated not only by realism but also by their downstream utility, representativeness, diversity, and artifact mitigation [6], [7]. This issue is particularly important for IMU-based HAR. Even a seemingly reasonable generated sequence may provide little additional coverage if it falls in a region already well represented by real training data, or it may deviate from activity-specific patterns. Treating such samples the same as real data can degrade model performance. Therefore, we treat virtual IMU samples as candidate training data: their utility depends on the coverage they provide and their consistency with the target activity, while their influence during training is determined by their reliability.

To address these challenges, we propose a coverage-aware virtual IMU augmentation framework for HAR with limited real training data. The framework addresses three practical questions: where to supplement the real training distribution, how to generate virtual candidates, and how to control their influence during training.

First, we select real training windows as anchors in a learned sensor embedding space. Diversity anchors represent distinct within-class motion patterns, while scarcity anchors are selected using a local-density criterion to target sparsely covered regions within the observed training distribution. Second, each anchor is converted into an anchor-conditioned prompt using anchor-level dynamics attributes such as tempo, intensity, and periodicity, and multiple virtual IMU candidates are generated for each anchor. Third, the generated candidates are assessed using their embedding-space proximity to the corresponding anchor and label consistency estimated by the classifier head of a seed encoder trained only on real training data. Based on this assessment, selected candidates are incorporated into HAR training with reliability-based sample weights, so that less reliable virtual samples contribute less to model optimization.

The main contributions are summarized as follows:

1) We propose a coverage-aware virtual IMU augmentation framework for HAR with limited real training data, where anchors selected from the real training data guide virtual IMU generation, and the resulting candidates are assessed to determine which are retained and how strongly they contribute to HAR training.

2) We introduce an anchor-conditioned virtual IMU generation strategy. It selects real training windows as diversity and scarcity anchors in a learned sensor embedding space and constructs generation prompts from anchor-level dynamics attributes, thereby guiding virtual candidate generation toward diverse within-class patterns and sparsely covered regions of the observed training distribution.

3) We propose a reliability-based candidate selection and weighting scheme. It computes selection costs from embedding-space proximity to the corresponding anchor and label consistency, retains the candidates with the lowest selection costs, and incorporates them into HAR training with reliability-based sample weights.

The remainder of this paper is organized as follows. Section 2 reviews related work on virtual and synthetic data for activity recognition, label-efficient learning and domain generalization, and noisy-label learning and robust training. Section 3 introduces the problem setup and presents the proposed framework for anchor-conditioned virtual IMU generation, candidate selection, and reliability-weighted integration. Section 4 describes the experimental setup and reports the evaluation results. Section 5 discusses limitations and future work. Section 6 concludes the paper.

## II. RELATED WORK

## A. Virtual and Synthetic Data for Activity Recognition

Virtual and synthetic data are widely used to mitigate labeled data scarcity in human activity recognition. Existing approaches can be broadly categorized into four directions: signal-level augmentation, generative time-series modeling, text-driven virtual IMU generation, and model-based IMU simulation. Signal-level methods apply transformations to temporal structure, amplitude, or orientation, while recent work explores automated augmentation policy search and physically plausible augmentation strategies [8]–[10]. Generative models synthesize inertial sequences for low-resource settings, with recent work exploring diffusion-based time-series generation [9], [11]. Text-driven methods generate virtual IMU signals from textual descriptions through cross-modal motion synthesis pipelines [4], [5], [12]. Meanwhile, model-based simulation approaches use human-body and sensor models to generate synthetic inertial data with explicit control over motion and sensor configurations [10], [13], [14].

Overall, these approaches have advanced synthetic inertial data generation in terms of scale, diversity, realism, and physical plausibility across augmentation, generative, and simulation paradigms [5], [9]–[11], [13], [14]. Recent studies on diffusion-based dataset distillation further suggest that synthetic data should be evaluated not only by realism but also by downstream utility, diversity, and artifact mitigation [6], [7]. However, the above augmentation and generation methods are primarily designed to optimize transformation policies, generative fidelity, or physical consistency, without explicitly considering which regions of the real training distribution are well represented or underrepresented. Consequently, synthetic samples may satisfy their method-specific objectives while providing little additional coverage of underrepresented regions in the empirical class-conditional distribution. This motivates a coverage-aware view of virtual IMU generation that considers both the empirical training distribution and the reliability of generated samples during downstream learning.

## B. Label-Efficient Learning and Domain Generalization

Label-efficient learning in sensor-based human activity recognition aims to reduce annotation costs by exploiting unlabeled or sparsely labeled sensor streams. Semi-supervised approaches improve HAR models by jointly using labeled and unlabeled samples, including through interpolation-based training [15]. Self-training methods further exploit unlabeled sequences via pseudo-labeling, iteratively refining models with high-confidence predictions [16]. In addition, unsupervised cross-user domain adaptation transfers knowledge from labeled source subjects to unlabeled target subjects, addressing user variation without target-domain labels [17].

Beyond annotation efficiency, recent domain-general HAR studies focus on robustness under distribution shift. Invariant representation learning has been studied under cross-subject, cross-dataset, and cross-position settings [18], while recent benchmarks evaluate the generalization of self-supervised HAR models to unseen target distributions [19]. Together, these studies highlight the importance of learning sensor representations that generalize across users, datasets, and sensor positions.

However, the above label-efficient and domain-general HAR methods primarily improve learning from the available observations. They improve data utilization and representation transfer, but do not explicitly expand the empirical coverage of underrepresented activity patterns. Therefore, they do not address how generated sensor data can complement limited real-data coverage without introducing unreliable training signals.

## C. Noisy-Label Learning and Robust Training

Noisy-label learning and robust training aim to reduce the effect of corrupted labels by making optimization less sensitive to unreliable supervision. Representative approaches include robust loss functions such as the generalized cross-entropy loss, which reduces sensitivity to incorrect labels [20]. Another line of work focuses on sample selection and reweighting, where small-loss or high-confidence instances are prioritized during training. Co-teaching improves robustness by training two networks that select small-loss samples for each other, thereby reducing error accumulation from noisy labels [21]. Subsequent methods extend this line through instance-specific sample selection, label correction, and confidence-aware regularization or tracking [22]–[26].

These studies show that treating all training samples as equally reliable can degrade learning, and that confidenceaware or selection-based strategies can mitigate noise-induced error propagation. However, they are mainly designed for fixed training sets with potentially corrupted labels, rather than for settings where additional training samples are produced by a generative process. For virtual IMU data, unreliability may arise from generation artifacts, distribution mismatch, or inconsistency with the intended activity. Accordingly, generated sequences should not be treated as uniformly reliable training data, and their contribution to learning should reflect their estimated reliability.

## III. METHODOLOGY

The proposed framework is summarized in Fig. 2. Building robust sensor-based HAR models is challenging because labeled IMU data are costly to obtain and activity signals vary across users, execution styles, and sensing conditions [27]. A finite real collection therefore provides an incomplete empirical view of the activity distribution. Virtual IMU generation can broaden this empirical coverage, but generated samples should not be assumed to be as reliable as real labeled samples. We address this issue by grounding virtual IMU generation in real training anchors and integrating generated samples according to their estimated reliability. Specifically, we select real training windows as anchors using a learned sensor embedding space, convert their dynamics attributes into prompts, and generate a pool of virtual IMU candidates. The candidates are assessed by their embedding-space proximity to the corresponding anchors and label consistency, and the selected candidates are integrated into HAR training with reliability-based weights.

We first formalize the window-level HAR setting and the leave-one-subject-out (LOSO) evaluation protocol in Sec. III-A. We then introduce anchor-conditioned virtual IMU generation in Sec. III-B, virtual candidate assessment and selection in Sec. III-C, and reliability-weighted integration in Sec. III-D. Algorithm 1 summarizes the complete procedure.

## A. Problem Setup

We formulate IMU-based HAR as window-level multi-class classification: a model receives a multichannel IMU time window and predicts its activity label. Let $\mathcal { U } = \{ 1 , \ldots , S \}$

denote the set of subjects. For each subject $m \in \mathcal { U } .$ , let $\mathcal { D } _ { m }$ denote the set of labeled real IMU windows:

$$
\mathcal { D } _ { m } = \{ ( x _ { m , i } , y _ { m , i } ) \} _ { i = 1 } ^ { N _ { m } } , \qquad x _ { m , i } \in \mathbb { R } ^ { T \times d } , \quad y _ { m , i } \in \mathcal { V } .\tag{1}
$$

Here $x _ { m , i }$ is the i-th IMU window from subject m, $y _ { m , \astrosun }$ <sub>i</sub> is its activity label, T is the window length, d is the number of IMU channels, and $\mathcal { V }$ is the activity label set. The subject index is used only to define cross-subject splits and is not part of the model input.

We evaluate under the LOSO protocol. For a held-out subject u, the training and test sets are

$$
\mathcal { D } _ { \mathrm { t r } } ^ { ( u ) } = \bigcup _ { m \in \mathcal { U } \backslash \{ u \} } \mathcal { D } _ { m } , \qquad \mathcal { D } _ { \mathrm { t e } } ^ { ( u ) } = \mathcal { D } _ { u } .\tag{2}
$$

All fold-specific components are constructed without access to $\mathcal { D } _ { \mathrm { t e } } ^ { ( u ) }$ . These include seed encoder training, anchor selection, dynamics-attribute statistics, candidate assessment, weight assignment, and normalization statistics. The held-out set $\mathcal { D } _ { \mathrm { t e } } ^ { ( u ) }$ is used only for final evaluation on real IMU windows.

For notational simplicity, we re-index the real training windows in $\mathcal { D } _ { \mathrm { t r } } ^ { ( u ) }$ as $( x _ { i } , y _ { i } )$ when the subject index is not needed.

## B. Anchor-Conditioned Virtual IMU Generation

A finite IMU training set provides only a partial empirical view of the within-class variation of each activity. To address this issue, we select real training windows as generation anchors based on their positions within each class in a learned embedding space. Specifically, we train a seed network on the real IMU windows in the current LOSO training split and use the resulting embedding space to select two complementary types of anchors. Diversity anchors represent distinct withinclass motion patterns, whereas scarcity anchors target sparsely covered regions while excluding extreme outliers. Each anchor provides a real sensor reference and the dynamics attributes used to construct its generation prompt.

For each real training window $( x _ { i } , y _ { i } ) \in \mathcal { D } _ { \mathrm { t r } } ^ { ( u ) }$ , we compute the seed embedding

$$
z _ { i } = E _ { \psi } ( x _ { i } ) .\tag{3}
$$

Here $E _ { \psi }$ and $H _ { \psi }$ denote the encoder and classifier head, respectively, of the seed network trained on $\mathcal { D } _ { \mathrm { t r } } ^ { ( u ) }$ , and $z _ { i }$ is the embedding of window $x _ { i } .$ The classifier head $H _ { \psi }$ is later used to assess the label consistency of generated candidates.

For each class $y \in \mathcal { V }$ , let $\mathcal { T } _ { y } ^ { ( u ) }$ denote the indices of its real training windows in $\mathcal { D } _ { \mathrm { t r } } ^ { ( u ) }$ . The full anchor set for split u is

$$
\mathcal { A } ^ { ( u ) } = \bigcup _ { y \in \mathcal { V } } \left( \mathcal { A } _ { y } ^ { \mathrm { d i v } } \cup \mathcal { A } _ { y } ^ { \mathrm { s c a r } } \right) .\tag{4}
$$

We represent each anchor by the index of its corresponding real training window; accordingly, $x _ { a } , y _ { a }$ , and $z _ { a }$ denote the anchor window, label, and embedding, respectively. For each class, diversity-anchor selection is initialized with the smallestindex training sample of that class, with $a _ { y , 1 } ^ { \mathrm { d i v } } = \operatorname* { m i n } \mathcal { T } _ { y } ^ { ( u ) }$ and $\mathcal { A } _ { y , 1 } ^ { \mathrm { d i v } } = \{ a _ { y , 1 } ^ { \mathrm { d i v } } \}$ . The remaining anchors are then selected using class-wise greedy farthest-point sampling:

![](images/61536100dc9b615ba52f5426debe53c2efed3f7ccd461cffc1bf73d6f49ce522.jpg)  
Fig. 2. Framework overview. (1) Anchor-guided virtual IMU generation: select real training windows as diversity and scarcity anchors in a learned sensor embedding space, construct anchor-conditioned prompts from activity labels and anchor-level tempo, intensity, and periodicity attributes, and synthesize multiple virtual IMU candidates. (2) Virtual candidate assessment: assess candidates using their embedding-space proximity to the corresponding anchors and label consistency estimated by the seed classifier head, combine the two criteria into a selection cost, and retain the candidates with the lowest costs for each anchor. (3) Reliability-weighted HAR training: assign reliability-based sample weights using candidate selection costs and within-anchor ranks, and train the HAR model on real IMU windows and selected virtual candidates.

$$
a _ { y , t } ^ { \mathrm { { d i v } } } = \arg \operatorname* { m a x } _ { \substack { i \in \mathcal { X } _ { y } ^ { ( u ) } \backslash \mathcal { A } _ { y , t - 1 } ^ { \mathrm { { d i v } } } } } \operatorname* { m i n } _ { a \in \mathcal { A } _ { y , t - 1 } ^ { \mathrm { { d i v } } } } \| z _ { i } - z _ { a } \| _ { 2 } \qquad t = 2 , \ldots , K _ { \mathrm { { d i v } } } .\tag{5}
$$

Here, $\mathcal { A } _ { y , t - 1 } ^ { \mathrm { d i v } }$ contains the diversity anchors selected before iteration t. The procedure yields $K _ { \mathrm { d i v } }$ diversity anchors for each class.

Scarcity anchors are selected by a class-conditional localdensity criterion:

$$
\delta _ { i } ^ { \mathrm { s c a r } } = d _ { k } \left( z _ { i } ; \mathcal { T } _ { y } ^ { ( u ) } \right) , \qquad Q _ { y } ^ { \mathrm { l o w } } \leq \delta _ { i } ^ { \mathrm { s c a r } } \leq Q _ { y } ^ { \mathrm { h i g h } } .\tag{6}
$$

Here, $d _ { k } \Big ( z _ { i } ; \mathcal { T } _ { y } ^ { ( u ) } \Big )$ is the Euclidean distance from $z _ { i }$ to its $k \mathrm { - }$ th nearest same-class neighbor, excluding itself, and $Q _ { y } ^ { \mathrm { l o w } }$ and $Q _ { y } ^ { \mathrm { h i g h } }$ are class-wise empirical quantile thresholds computed within the current training fold. Samples satisfying this interval are treated as scarcity candidates. For each class, we retain up to $K _ { \mathrm { s c a r } }$ candidates with the largest sparsity scores to form $\mathcal { A } _ { y } ^ { \mathrm { s c a r } }$ . The lower threshold focuses selection on sparsely covered regions, while the upper threshold excludes extreme outliers.

Real IMU anchors provide low-level sensor dynamics, whereas text-conditioned motion generators operate on natural-language motion descriptions. To bridge this gap, we express the anchor dynamics as textual motion cues and combine them with the activity label to construct an anchorconditioned prompt. For each anchor $^ { a , }$ we extract a compact dynamics descriptor from the anchor window:

$$
\begin{array} { r l } & { { \bf b } _ { a } = \Phi ( x _ { a } ) = \big ( b _ { a } ^ { \mathrm { t e m p o } } , b _ { a } ^ { \mathrm { i n t e n s i t y } } , b _ { a } ^ { \mathrm { p e r i o d i c i t y } } \big ) , } \\ & { { \bf \Phi } _ { \tau _ { a } } = \mathrm { P r o m p t } ( y _ { a } , { \bf b } _ { a } ) . } \end{array}\tag{7}
$$

The anchor dynamics are described by three attributes:

• Tempo describes the movement rate of the anchor, indicating whether the target motion is slow, moderate, or fast.

• Intensity reflects the inertial energy in the accelerometer and gyroscope channels and represents the magnitude of movement.

• Periodicity characterizes repeated temporal structure, distinguishing stable rhythmic patterns from weakly periodic or irregular motion.

Here, $\Phi ( \cdot )$ extracts the dynamics descriptor $ { \mathbf { b } } _ { a }$ from the anchor window, and Prompt maps $( y _ { a } , \mathbf { b } _ { a } )$ to an anchor-conditioned motion prompt $\tau _ { a } .$ . In implementation, each prompt contains one activity field and three anchor-derived dynamics fields: tempo, intensity, and periodicity. For example, an anchor labeled “walking forward” with medium tempo, low intensity, and strong periodicity is represented by a prompt describing walking at a moderate tempo with low movement intensity and a regular rhythm. Finally, a fixed synthesis interface converts the prompt into virtual IMU candidates:

$$
\{ \tilde { x } _ { a , j } \} _ { j = 1 } ^ { M } = G ( \tau _ { a } ) , \qquad a \in \mathcal { A } ^ { ( u ) } .\tag{8}
$$

Here, G denotes the fixed synthesis interface, $\tilde { x } _ { a , j }$ is the $j -$ th virtual IMU candidate generated for anchor a, and M is the candidate budget per anchor. We instantiate G with the synthesis pipeline adopted by IMUGPT [4], [5], using a pretrained T2M-GPT model [28] followed by IMUSim [29]. T2M-GPT maps $\tau _ { a }$ to a 3D body-motion sequence, which is converted to BVH through inverse kinematics. IMUSim then simulates ideal accelerometer and gyroscope signals at the skeleton joint nearest to the target sensor location. The synthesis backend is kept fixed throughout all experiments.

## C. Virtual Candidate Assessment and Selection

Virtual IMU candidates are generated to supplement motion patterns that are sparsely represented in the real training data. However, apparent realism alone does not ensure that a candidate is suitable for HAR training. In IMU-based HAR, generated signals that fail to preserve activity-consistent inertial patterns may introduce label noise and degrade downstream training. We therefore assess and select generated candidates before adding them to the training set. The assessment uses the corresponding anchor embeddings and the seed network trained within the current LOSO training split. We select candidates that remain close to their corresponding real anchors in the embedding space and receive high predicted probabilities for the intended activity labels.

For each generated candidate $\tilde { x } _ { a , j }$ , we compute its embedding and the predicted probability of the intended activity label using the seed network:

$$
\tilde { z } _ { a , j } = E _ { \psi } ( \tilde { x } _ { a , j } ) , \qquad p _ { a , j } = [ H _ { \psi } ( \tilde { z } _ { a , j } ) ] _ { y _ { a } } .\tag{9}
$$

Here, $\tilde { z } _ { a , j }$ is the candidate embedding, and $p _ { a , j }$ is the probability assigned to the intended activity label $y _ { a }$

We quantify anchor proximity using the cosine distance between the candidate and anchor embeddings:

$$
D _ { a , j } ^ { \mathrm { e m b } } = 1 - \frac { \mathbf { z } _ { a } ^ { \top } \tilde { \mathbf { z } } _ { a , j } } { \| \mathbf { z } _ { a } \| _ { 2 } \| \tilde { \mathbf { z } } _ { a , j } \| _ { 2 } } .\tag{10}
$$

A smaller $D _ { a , j } ^ { \mathrm { e m b } }$ indicates greater similarity to the corresponding anchor, while a larger $p _ { a , j }$ indicates stronger support for the intended activity label.

The candidate selection cost is defined as

$$
c _ { a , j } = \lambda _ { d } D _ { a , j } ^ { \mathrm { e m b } } + \lambda _ { p } ( 1 - p _ { a , j } ) , \qquad \lambda _ { d } , \lambda _ { p } \geq 0 , \quad \lambda _ { d } + \lambda _ { p } = 1 .\tag{11}
$$

Lower values of $c _ { a , j }$ indicate candidates that remain closer to the anchor representation and receive stronger support for the intended activity label.

For each anchor, we retain the candidates with the lowest selection costs. Let rank $_ a ( j )$ denote the rank of candidate $j$ when the costs $c _ { a , l }$ are sorted in ascending order. The selected candidate indices are:

$$
\mathcal { T } _ { a } = \{ j \in \{ 1 , \dots , M \} : \mathrm { r a n k } _ { a } ( j ) \leq K _ { \mathrm { s e l } } \}\tag{12}
$$

Here, $K _ { \mathrm { s e l } }$ is the maximum number of candidates retained for each anchor, and $\mathcal { I } _ { a }$ is the corresponding set of selected candidate indices. The selected candidates are subsequently integrated into HAR training using reliability-based weights.

## D. Reliability-Weighted Integration

The selected virtual candidates are not treated as equally reliable. We assign weights at two levels: an anchor-level risk factor determined by the best candidate cost for each anchor, and a candidate-level rank factor determined by the within-anchor cost ordering. For each anchor, we summarize candidate reliability using its lowest selection cost:

$$
c _ { a } ^ { \star } = \operatorname* { m i n } _ { j = 1 , \ldots , M } c _ { a , j } .\tag{13}
$$

A lower $c _ { a } ^ { \star }$ indicates that the synthesis process produces at least one candidate with strong anchor proximity and label consistency.

We divide anchors into low, medium, and high risk groups using the lower and upper empirical tertiles of $\{ c _ { a } ^ { \star } : a \in \mathcal { A } ^ { ( u ) } \}$ within the current training fold. Let $\tau _ { \mathrm { l o w } } ^ { ( u ) }$ and $\tau _ { \mathrm { h i g h } } ^ { ( \bar { u } ) }$ denote these fold-specific thresholds. The risk category of anchor a is

$$
R _ { a } = \left\{ \begin{array} { l l } { \mathrm { L } , } & { c _ { a } ^ { \star } \leq \tau _ { \mathrm { l o w } } ^ { ( u ) } , } \\ { \mathrm { M } , } & { \tau _ { \mathrm { l o w } } ^ { ( u ) } < c _ { a } ^ { \star } \leq \tau _ { \mathrm { h i g h } } ^ { ( u ) } , } \\ { \mathrm { H } , } & { c _ { a } ^ { \star } > \tau _ { \mathrm { h i g h } } ^ { ( u ) } . } \end{array} \right.\tag{14}
$$

We map the anchor-level risk categories to discrete reliability factors:

$$
w _ { \mathrm { r i s k } } ( R _ { a } ) = \left\{ \begin{array} { l l } { w _ { \mathrm { L } } , \quad R _ { a } = \mathrm { L } , } \\ { w _ { \mathrm { M } } , \quad R _ { a } = \mathrm { M } , \quad } & { w _ { \mathrm { L } } \geq w _ { \mathrm { M } } \geq w _ { \mathrm { H } } \geq 0 . } \\ { w _ { \mathrm { H } } , \quad R _ { a } = \mathrm { H } , } \end{array} \right.\tag{15}
$$

For each selected candidate $j \in \mathcal { I } _ { a }$ , the final training weight is

$$
\omega _ { a , j } = \frac { \rho } { | \mathcal { T } _ { a } | } w _ { \mathrm { r i s k } } ( R _ { a } ) \gamma _ { \mathrm { r a n k } _ { a } ( j ) } , \qquad j \in \mathcal { I } _ { a } .\tag{16}
$$

Here, $\rho$ controls the overall contribution of virtual samples, and $\gamma _ { \mathrm { r a n k } _ { a } ( j ) }$ determines the relative contribution of candidates within the same anchor, with

$$
1 = \gamma _ { 1 } \geq \gamma _ { 2 } \geq \cdot \cdot \cdot \geq \gamma _ { K _ { \mathrm { s e l } } } \geq 0 .\tag{17}
$$

The factor $1 / | { \mathcal { I } } _ { a } |$ prevents the aggregate contribution of an anchor from growing linearly with the number of selected candidates. The risk factor is shared by candidates generated from the same anchor, while the rank factor assigns greater influence to lower-cost candidates within that anchor.

After weight assignment, the weighted virtual set for split u is defined as

$$
\mathcal V ^ { ( u ) } = \{ ( \tilde { x } _ { a , j } , y _ { a } , \omega _ { a , j } ) : a \in \mathcal A ^ { ( u ) } , j \in \mathcal T _ { a } \} .\tag{18}
$$

To assign unit weight to real samples, we train the HAR model $f _ { \theta }$ using the weighted empirical risk minimization objective.

$$
\begin{array} { r l } & { \mathcal { L } ( \theta ) = \displaystyle \sum _ { ( x _ { i } , y _ { i } ) \in \mathcal { D } _ { \mathrm { t r } } ^ { ( u ) } } \ell \big ( f _ { \theta } ( x _ { i } ) , y _ { i } \big ) } \\ & { \quad \quad \quad + \displaystyle \sum _ { a \in \mathcal { A } ^ { ( u ) } } \sum _ { j \in \mathcal { I } _ { a } } \omega _ { a , j } \ell \big ( f _ { \theta } ( \tilde { x } _ { a , j } ) , y _ { a } \big ) . } \end{array}\tag{19}
$$

Here, ℓ denotes the supervised classification loss. All virtual samples, weights, and selection costs are computed strictly from the training split, without access to the held-out subject during generation, selection, weighting, or optimization.

Algorithm 1: Coverage-Aware Virtual IMU Augmen  
tation for HAR.   
Input: Training set $\mathcal { D } _ { \mathrm { t r } } ^ { ( u ) } .$ , fixed synthesis interface $G ,$ and the   
anchor-selection, candidate-selection, and integration   
hyperparameters   
Output: Trained HAR model $f _ { \theta }$   
Train the seed network $( E _ { \psi } , H _ { \psi } )$ on $\mathcal { D } _ { \mathrm { t r } } ^ { ( u ) }$ and compute   
$z _ { i } = E _ { \psi } ( x _ { i } )$ for all real training windows   
For each class, select $K _ { \mathrm { d i v } }$ diversity anchors by greedy   
farthest-point sampling and up to $K _ { \mathrm { s c a r } }$ scarcity anchors by the   
class-conditional sparsity criterion   
foreach anchor $a \in \mathcal { A } ^ { ( u ) }$ do   
$\mathbf { b } _ { a } \gets \Phi ( x _ { a } )$   
$\tau _ { a } \gets \mathrm { P r o m p t } ( y _ { a } , \mathbf { b } _ { a } )$   
$\{ \tilde { x } _ { a , j } \} _ { j = 1 } ^ { M } \stackrel {  } {  } \tilde { G } ( \tau _ { a } )$   
for $j \doteq 1 , \ldots , M$ do   
Compute $\tilde { z } _ { a , j } , p _ { a , j } , D _ { a , j } ^ { \mathrm { e m b } }$ , and $c _ { a , j }$ using Eqs. (9)–(11)   
Rank candidates in ascending order of $c _ { a , j }$ and let rank<sub>a</sub>(j)   
denote the within-anchor rank   
$\mathcal { I } _ { a }  \{ j : \mathrm { r a n k } _ { a } ( j ) \leq K _ { \mathrm { s e l } } \}$   
$c _ { a } ^ { \star } \gets \operatorname* { m i n } _ { j = 1 , \dots , M } c _ { a , j }$   
Compute $\tau _ { \mathrm { l o w } } ^ { ( u ) }$ and $\tau _ { \mathrm { h i g h } } ^ { ( u ) }$ as the lower and upper empirical tertiles of   
$\{ c _ { a } ^ { \star } : a \in \mathcal { A } ^ { ( u ) } \}$   
$\mathcal { V } ^ { ( u ) }  \emptyset$   
foreach anchor a $\mathfrak { a } \in \mathcal { A } ^ { ( u ) }$ do   
Assign $R _ { a }$ according to Eq. (14)   
foreach $j \in \mathcal { I } _ { a }$ do   
Compute $\omega _ { a , j }$ using Eq. (16)   
Add $( \tilde { x } _ { a , j } , y _ { a } , \omega _ { a , j } )$ to $\mathcal { \nu } ^ { ( u ) }$   
Train $f _ { \theta }$ on $\mathcal { D } _ { \mathrm { t r } } ^ { ( u ) }$ and $\mathcal { \nu } ^ { ( u ) }$ using Eq. (19)   
return $f _ { \theta }$

## IV. EXPERIMENTS

## A. Experimental Setup

1) Dataset: We evaluate the proposed method on three IMU-based HAR benchmarks: USC-HAD [30], PAMAP2 [31], and RealWorld [32]. We further evaluate the method in an exploratory medical IMU classification setting using PADS [33]–[35]. USC-HAD contains daily activities recorded using a single IMU worn at the front-right hip. PAMAP2 provides recordings of diverse physical activities from multiple wearable sensors; in our experiments, we use the chest IMU recordings. RealWorld captures wearable activity data in less constrained daily environments, exhibiting stronger real-world sensing variability; we use the forearm IMU recordings in our experiments. PADS contains wristworn IMU recordings from clinical movement assessments and is treated as an exploratory medical extension rather than a standard HAR benchmark. Table I summarizes the dataset scope and evaluation protocols.

TABLE I  
EXPERIMENTAL DATASETS, SENSOR CONFIGURATIONS, AND EVALUATION PROTOCOLS.
<table><tr><td colspan="5">HAR benchmarks</td></tr><tr><td>Dataset</td><td>Subjects</td><td>Activities</td><td>Sensor placement</td><td>Protocol</td></tr><tr><td>USC-HAD</td><td>14</td><td>7</td><td>front-right hip</td><td>LOSO</td></tr><tr><td>PAMAP2</td><td>8</td><td>5</td><td>chest</td><td>LOSO</td></tr><tr><td>RealWorld</td><td>15</td><td>5</td><td>forearm</td><td>LOSO</td></tr><tr><td colspan="5">Exploratory medical extension</td></tr><tr><td>Dataset</td><td>Folds</td><td>Task</td><td>Sensor placement</td><td>Protocol</td></tr><tr><td>PADS</td><td>5</td><td>PD-vs-HC classification</td><td>wrist</td><td>subject-wise split</td></tr></table>

For USC-HAD, PAMAP2, and RealWorld, we evaluate dynamic activity subsets under the LOSO protocol to assess cross-subject generalization. We focus on dynamic activities because their temporal dynamics and signal-amplitude variations make them more suitable than near-static postures for evaluating virtual IMU generation. In each LOSO fold, all windows from one subject form the test set, while the remaining subjects provide the data used for training and any fold-specific model selection. Virtual IMU data are used only for training. The held-out subject is never used for normalization, virtual candidate construction, or model fitting within each LOSO fold. LOSO thus evaluates generalization under subject-level distribution shift. For these HAR benchmarks, we segment IMU streams using a sliding window of 2s with a step size of 1s. All real IMU streams are downsampled to 20 Hz to match the virtual data sampling rate, and each channel is normalized using statistics computed from the training subjects in the current fold. The same preprocessing pipeline is used across all compared methods. For PADS, we use the five-fold subject-wise split reported in Table I and report the results separately as an exploratory medical extension.

2) Evaluation Metrics: For USC-HAD, PAMAP2, and RealWorld, we evaluate performance on the held-out subject in each LOSO fold using Macro-F1 as the primary metric and overall accuracy (Acc) as a secondary metric. Macro-F1 is used as the primary metric because it weights all activity classes equally and reduces the dominance of majority classes, whereas accuracy summarizes overall recognition performance. We first compute metrics on each LOSO fold, average them across folds for each seed, and then report the mean ± standard deviation across seeds. All methods use identical dataset splits and evaluation procedures within each benchmark to ensure fair comparison. For PADS, we report Macro-F1 and balanced accuracy separately as an exploratory medical extension.

3) Implementation Details: We evaluate three downstream HAR models: DeepConvLSTM [36], DeepConvLSTM-Attention [37], and MLP-HAR [38]. DeepConvLSTM combines convolutional feature extraction with recurrent temporal modeling, while DeepConvLSTM-Attention further incorporates temporal attention pooling. MLP-HAR serves as a lightweight feed-forward alternative. Within each benchmark and downstream model, all compared training configurations use the same splits, preprocessing, and model configuration to ensure fair comparison.

TABLE II  
MAIN SETTINGS FOR VIRTUAL IMU CONSTRUCTION AND HAR TRAINING.
<table><tr><td>Aspect</td><td>Setting</td></tr><tr><td>Data preprocessing Anchor selection</td><td>2-s windows; 1-s stride; 20 Hz class-wise diversity and scarcity anchors from real</td></tr><tr><td>Prompt conditioning</td><td>training windows activity label; tempo, intensity, periodicity</td></tr><tr><td>Virtual generation</td><td>T2M-GPT followed by IMUSim</td></tr><tr><td>Candidate pool</td><td>20 candidates per anchor</td></tr><tr><td>Candidate selection</td><td>lowest-cost  $\overline { { K _ { \mathrm { s e l } } } }$  candidates based on embedding distance and label consistency</td></tr><tr><td>Training integration</td><td>anchor-level risk and within-anchor candidate rank</td></tr><tr><td>Evaluation</td><td>LOSO; each held-out subject is used only for evaluation</td></tr></table>

Reliability-based weighting is applied only during training. Real samples are assigned unit weight, while selected virtual IMU samples are weighted according to the anchor-level risk category and their within-anchor ranks. For classical classifiers, weights are incorporated through sample-weighted optimization; for neural models, they are applied as persample loss weights. All neural models are implemented in PyTorch 1.9.1 and trained with CUDA 11.1 and cuDNN 8.0.5.

Table II summarizes the main settings for data construction and HAR model training. The main HAR results are averaged over three random seeds (45, 46, and 47) using the same LOSO splits. Some ablation studies use different random-seed sets from the main experiments; their absolute values should therefore be compared within the corresponding table or figure. Within each table or figure, all compared configurations use the same random seeds and evaluation protocol.

4) Training Configurations: For the three HAR benchmarks, we evaluate the following six training configurations under the LOSO protocol. Within each benchmark and downstream model, all configurations use identical data splits, preprocessing, model settings, and evaluation procedures; they differ only in how additional training data are constructed and integrated.

1) Real-only. The model is trained only on the real IMU training data, without any virtual data or augmentation. This serves as the primary supervised baseline.

2) Traditional Augmentation [9]. The model is trained on real IMU data with standard sensor-level augmentation, including rotation, Gaussian noise, and additive sensor bias.

3) TimeGAN [39]. The training set is augmented with synthetic IMU sequences generated using the official opensource implementation of TimeGAN, trained separately on the real data in each LOSO training fold.

4) Diffusion-TS [40]. For the Diffusion-TS baseline, we use the official Diffusion-TS implementation [40] with a class-wise unconditional adaptation, training a separate generator for each activity class within each LOSO fold using only the corresponding real training windows.

5) IMUGPT [5]. The model is trained on real IMU data augmented with IMUGPT-style sequences generated from activity-level prompts.

6) Ours. The full method trains on real data and anchorconditioned virtual IMU candidates, with their selection costs determining candidate retention and training weights.

For a fair comparison, all configurations using additional data are limited to 150 sequences per activity class before window segmentation.

## B. Main Results

Table III summarizes the main HAR results on USC-HAD, PAMAP2, and RealWorld using DeepConvLSTM, DeepConvLSTM-Attention, and MLP-HAR. Values are reported as percentages in the form mean ± standard deviation. F1 denotes Macro-F1, and ∆F1 denotes the absolute Macro-F1 improvement over the Real-only baseline. The reported results use 20% of the labeled real data on USC-HAD and PAMAP2 and 10% on RealWorld.

For every combination of dataset and model, Ours achieves the highest mean Macro-F1 and accuracy. Compared with Real-only, it improves Macro-F1 by 2.15 to 11.24 percentage points on USC-HAD, 5.28 to 14.40 points on PAMAP2, and 3.77 to 4.92 points on RealWorld. It also outperforms the strongest non-Ours baseline by 0.18 to 5.39 points on USC-HAD, 1.16 to 5.18 points on PAMAP2, and 0.76 to 2.50 points on RealWorld. These results show that the method remains effective across convolutional-recurrent, attention-based, and feed-forward HAR models, although the gains vary across datasets and downstream models.

On PAMAP2, the gains over the strongest non-Ours baseline are 5.18, 2.78, and 1.16 points with DeepConvL STM, DeepConvLSTM-Attention, and MLP-HAR, respectively. With DeepConvLSTM, Ours improves Macro-F1 from 62.95% to 77.35% and exceeds the strongest competing baseline by 5.18 points. On USC-HAD, Ours shows a large improvement over the strongest non-Ours baseline with Deep-ConvLSTM and MLP-HAR. However, it is only 0.18 points higher than Traditional Augmentation with DeepConvLSTM-Attention. On RealWorld, the gains over the strongest non-Ours baselines are smaller but remain positive across all three downstream models.

Traditional Augmentation is the strongest non-Ours Macro-F1 baseline in seven of the nine dataset–model pairs. Diffusion-TS improves over Real-only in all nine settings, and TimeGAN does so in eight, whereas IMUGPT improves Macro-F1 in only three of the nine settings. Under the common sequence-level data budget, none of the evaluated syntheticgeneration baselines consistently outperforms Traditional Augmentation.

## C. Ablation Study

a) Coverage-aware vs. random anchor selection.: Table IV compares the full method with a random class-balanced anchor variant, while keeping the remaining pipeline unchanged, to isolate the effect of coverage-aware anchor selection. The random-anchor variant still outperforms Real-only on all three datasets, showing that virtual candidates can benefit HAR training even when their generation anchors are selected randomly. However, the full method consistently outperforms random anchors, with additional Macro-F1 gains of 0.76, 3.41, and 0.70 percentage points on USC-HAD, PAMAP2, and RealWorld, respectively. The larger gain on PAMAP2 suggests that coverage-aware anchor selection is more beneficial on this dataset. Since both variants use the same virtual-data budget, the improvement over random selection shows that the choice of generation anchors matters beyond the number of generated samples.

TABLE III  
MAIN RESULTS ACROSS THREE DOWNSTREAM HAR MODELS.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="4">DeepConvLSTM [36]</td><td colspan="4">DeepConvLSTM-Attention [37]</td><td colspan="4">MLP-HAR [38]</td></tr><tr><td>Acc. ↑</td><td></td><td>F1 ↑</td><td>∆F1</td><td>Acc. ↑</td><td></td><td> $\mathbf { F 1 } \uparrow \qquad \Delta \mathbf { F 1 }$ </td><td></td><td>Acc. ↑</td><td></td><td>F1 ↑</td><td>∆F1</td></tr><tr><td rowspan="6">USC-HAD</td><td>Real-only</td><td> $7 5 . 4 4 \pm 3 . 0 5 ~ 7 4 . 1 8 \pm 1 . 9 9$ </td><td></td><td></td><td></td><td> $7 9 . 2 1 \pm 1 . 1 7 7 8 . 2 8 \pm 1 . 6 3 - $ </td><td></td><td></td><td></td><td></td><td> $5 8 . 2 0 \pm 0 . 6 3 5 4 . 1 4 \pm 0 . 9 7$ </td><td></td><td></td></tr><tr><td>Trad. Aug. [9]</td><td></td><td>80.23 ± 0.44 78.94 ± 0.87 +4.76</td><td></td><td></td><td> $8 1 . 0 8 \pm 0 . 5 9 ~ 8 0 . 2 5 \pm 0 . 4 0 + 1 . 9 7$ </td><td></td><td></td><td></td><td> $6 2 . 2 2 \pm 1 . 2 7$ </td><td></td><td>58.26 ± 1.22 +4.12</td><td></td></tr><tr><td>TimeGAN [39]</td><td></td><td> $7 7 . 4 7 \pm 1 . 1 5 7 6 . 2 0 \pm 1 . 3 8$ </td><td></td><td>+2.02</td><td> $7 9 . 3 1 \pm 1 . 4 3 7 8 . 0 5 \pm 1 . 0 7 - 0 . 2 3$ </td><td></td><td></td><td></td><td> $6 0 . 0 5 \pm 1 . 3 0 ~ 5 5 . 4 1 \pm 1 . 4 8 ~ + 1 . 2 7$ </td><td></td><td></td><td></td></tr><tr><td>Diffusion-TS [40]</td><td></td><td> $8 0 . 1 3 \pm 2 . 4 7 \ 7 8 . 7 7 \pm 2 . 2 9$ </td><td></td><td>+4.59</td><td>80.33 ± 1.05 79.22 ± 0.87 +0.94</td><td></td><td></td><td></td><td> $6 3 . 1 8 \pm 1 . 3 3$ </td><td></td><td> $5 9 . 9 9 \pm 1 . 5 1$ </td><td>+5.84</td></tr><tr><td>IMUGPT [5]</td><td></td><td> $7 5 . 4 1 \pm 1 . 3 3 7 3 . 0 2 \pm 0 . 9 8 - 1 . 1 6$ </td><td></td><td></td><td> $7 5 . 0 6 \pm 0 . 3 6 7 3 . 2 2 \pm 0 . 2 6 - 5 . 0 6$ </td><td></td><td></td><td></td><td> $6 1 . 4 3 \pm 0 . 5 6$ </td><td></td><td> $5 8 . 5 9 \pm 0 . 6 4 + 4 . 4 5$ </td><td></td></tr><tr><td>Ours</td><td> $8 2 . 9 3 \pm \mathbf { 0 . 6 6 8 0 . 9 9 } \pm \mathbf { 0 . 2 8 }$ </td><td></td><td></td><td>+6.80</td><td> ${ \bf 8 1 . 9 7 } \pm { \bf 0 . 9 0 ~ 8 0 . 4 3 } \pm { \bf 1 . 1 2 } + 2 . { \bf 1 5 }$ </td><td></td><td></td><td></td><td> ${ \bf 6 7 . 0 4 \pm 1 . 8 1 }$ </td><td></td><td> ${ \bf 6 5 . 3 8 \pm 1 . 3 4 + 1 1 . 2 4 }$ </td><td></td></tr><tr><td rowspan="6">PAMAP2</td><td>Real-only</td><td>70.97 ± 0.11 62.95 ± 1.07</td><td></td><td></td><td></td><td> $7 5 . 3 6 \pm 2 . 5 5 6 8 . 1 2 \pm 1 . 8 8$ </td><td></td><td></td><td></td><td> $6 8 . 5 8 \pm 7 . 5 6$ </td><td>53.80 ± 6.53</td><td></td><td></td></tr><tr><td>Trad. Aug.</td><td></td><td> $7 9 . 5 8 \pm 1 . 8 5 7 2 . 1 7 \pm 3 . 4 1$ </td><td></td><td>+9.22</td><td> $8 0 . 5 0 \pm 1 . 1 8 7 4 . 4 7 \pm 2 . 4 4 + 6 . 3 5$ </td><td></td><td></td><td></td><td> $7 0 . 0 6 \pm 7 . 7 6$ </td><td></td><td> $5 7 . 2 0 \pm 6 . 7 4 + 3 . 4 0$ </td><td></td></tr><tr><td>TimeGAN</td><td>77.64 ± 1.83 68.25 ± 1.73 +5.30</td><td></td><td></td><td></td><td> $8 0 . 2 7 \pm 0 . 8 6 7 2 . 9 3 \pm 4 . 0 7 + 4 . 8 1$ </td><td></td><td></td><td></td><td> $6 7 . 7 1 \pm 6 . 9 9$ </td><td></td><td>54.50 ± 5.93 +0.70</td><td></td></tr><tr><td>Diffusion-TS</td><td> $8 0 . 7 8 \pm 0 . 4 9 6 8 . 8 9 \pm 1 . 6 4 + 5 . 9 4$ </td><td></td><td></td><td></td><td> $8 0 . 5 9 \pm 0 . 9 8 6 8 . 8 1 \pm 3 . 2 8 + 0 . 6 9$ </td><td></td><td></td><td></td><td> $6 8 . 5 0 \pm 7 . 5 0$ </td><td></td><td> $5 4 . 8 5 \pm 6 . 9 6 + 1 . 0 4$ </td><td></td></tr><tr><td>IMUGPT</td><td> $7 3 . 8 9 \pm 0 . 5 0 6 5 . 8 3 \pm 3 . 0 6 + 2 . 8 8$ </td><td></td><td></td><td></td><td> $7 3 . 9 8 \pm 1 . 1 6 6 3 . 6 8 \pm 2 . 2 5 - 4 . 4 4$ </td><td></td><td></td><td></td><td> $7 1 . 1 8 \pm 7 . 3 7$ </td><td></td><td> $5 7 . 9 2 \pm 6 . 6 5 + 4 . 1 2$ </td><td></td></tr><tr><td>Ours</td><td> $8 0 . 9 8 \pm 0 . 3 3 7 7 . 3 5 \pm 0 . 5 1 + 1 4 . 4 0$ </td><td></td><td></td><td></td><td></td><td></td><td> $8 1 . 9 2 \pm 0 . 7 2 7 7 . 2 5 \pm 1 . 7 2 + 9 . 1 3$ </td><td></td><td> $7 2 . 5 1 \pm 7 . 7 8 5 9 . 0 8 \pm 8 . 7 2 + 5 . 2 8$ </td><td></td><td></td><td></td></tr><tr><td rowspan="6">RealWorld</td><td>Real-only</td><td> $6 5 . 7 5 \pm 1 . 4 3 6 6 . 7 6 \pm 0 . 9 5$ </td><td></td><td></td><td></td><td> $6 8 . 7 9 \pm 1 . 3 2$ </td><td>68.84 ± 1.63</td><td></td><td></td><td> $6 2 . 0 7 \pm 1 . 3 4 6 0 . 1 4 \pm 3 . 1 2$ </td><td></td><td></td><td></td></tr><tr><td>Trad. Aug.</td><td></td><td>68.33 ± 2.12 69.18 ± 2.03 +2.42</td><td></td><td></td><td>71.27 ± 1.27 71.85 ± 0.51 +3.01</td><td></td><td></td><td></td><td></td><td> $6 2 . 8 9 \pm 1 . 2 5 6 3 . 1 0 \pm 1 . 4 8 + 2 . 9 6$ </td><td></td><td></td><td></td></tr><tr><td>TimeGAN</td><td></td><td> $6 8 . 5 8 \pm 1 . 4 2 6 8 . 8 8 \pm 1 . 0 7$ </td><td></td><td>+2.12</td><td> $6 9 . 9 9 \pm 1 . 2 3 7 0 . 8 2 \pm 1 . 1 4 + 1 . 9 8$ </td><td></td><td></td><td></td><td></td><td> $6 2 . 2 1 \pm 1 . 2 4 6 1 . 3 8 \pm 1 . 7 9 + 1 . 2 4$ </td><td></td><td></td><td></td></tr><tr><td>Diffusion-TS</td><td></td><td>68.68 ± 0.96 68.34 ± 0.81 +1.58</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>70.91 ± 0.67 70.96 ± 0.89 +2.12 62.43 ± 0.79 62.40 ± 1.14 +2.26</td><td></td><td></td><td></td></tr><tr><td>IMUGPT</td><td> $6 6 . 2 9 \pm 1 . 2 3 6 5 . 5 0 \pm 1 . 3 1 - 1 . 2 6$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td> $6 9 . 0 5 \pm 1 . 3 5 6 8 . 2 6 \pm 1 . 4 0 - 0 . 5 8$ </td><td></td><td></td><td></td><td> $6 0 . 4 4 \pm 0 . 8 6 5 9 . 4 5 \pm 0 . 7 9 - 0 . 6 9$ </td><td></td></tr><tr><td>Ours</td><td></td><td> $7 1 . 6 5 \pm 1 . 7 2 7 1 . 6 8 \pm 1 . 4 5 + 4 . 9 2$ </td><td></td><td></td><td></td><td></td><td></td><td> $7 2 . 4 7 \pm 1 . 1 8 7 2 . 6 1 \pm 1 . 7 7 + 3 . 7 7$ </td><td></td><td> $6 4 . 3 3 \pm 1 . 2 8 6 4 . 0 0 \pm 1 . 0 6 + 3 . 8 6$ </td><td></td><td></td><td></td></tr></table>

TABLE IV  
COVERAGE-AWARE VERSUS RANDOM ANCHOR SELECTION.

![](images/22345d5a8fcb3a008959331e70c5f7681d0fce1a09d1a30fce11a0ad359ed150.jpg)

<table><tr><td>Dataset</td><td>Real-only</td><td>Random</td><td>Full Ours</td><td>∆F1</td></tr><tr><td>USC-HAD</td><td> $7 4 . 3 2 \pm 1 . 9 3$ </td><td> $8 0 . 1 8 \pm 0 . 8 1$ </td><td> ${ \bf 8 0 . 9 4 \pm 0 . 7 4 }$ </td><td>+0.76</td></tr><tr><td>PAMAP2</td><td> $6 2 . 9 5 \pm 1 . 0 7$ </td><td> $7 3 . 9 4 \pm 0 . 5 0$ </td><td> $7 7 . 3 5 \pm \mathbf { 0 . 5 1 } + 3 . 4 1$ </td><td></td></tr><tr><td> $\mathrm { R e a l W o r l d }$ </td><td> $6 6 . 9 7 \pm 1 . 1 2$ </td><td> $7 0 . 9 8 \pm 1 . 7 4$ </td><td> ${ \bf 7 1 . 6 8 \pm 1 . 4 5 }$ </td><td>+0.70</td></tr><tr><td>Ours over random class-balanced anchors.</td><td>∆F1 is the absolute Macro-F1 improvement of Full</td><td></td><td></td><td></td></tr></table>

Fig. 3. Effect of anchor-source composition. Bars show the Macro-F1 gain over the corresponding Real-only baseline. Combining diversity and scarcity anchors performs best on USC-HAD and PAMAP2, whereas scarcity-only is slightly higher on RealWorld.

b) Anchor-source composition.: Fig. 3 compares the effects of diversity-only, scarcity-only, and combined anchorsource configurations across the three datasets. On USC-HAD, the diversity-only and scarcity-only variants improve

Macro-F1 over Real-only by 2.45 and 0.24 percentage points, respectively, while the combined variant increases the gain to 5.30 points. Diversity anchors are the stronger single source in this setting. In particular, the 5.30-point gain of the combined variant is larger than the sum of the gains from the two single-source variants (2.69 points). This result suggests that, although scarcity anchors contribute little when used alone in this setting, they may provide additional local coverage or training stability when combined with diversity anchors. On PAMAP2, the diversity-only and scarcity-only variants improve over Real-only by 9.49 and 10.51 points, respectively, while their combination achieves the largest gain of 14.40 points. On RealWorld, the scarcity-only variant achieves the largest gain of 5.29 points and exceeds the combined variant by 0.37 points. This result shows that adding another anchor source does not always lead to better performance.

Overall, the contribution of each anchor source is dataset dependent. Diversity is the stronger single source on USC-HAD, both sources are effective on PAMAP2, and scarcity is the strongest source on RealWorld. Given that RealWorld was collected under less controlled daily-life conditions, the higher F1 of scarcity-only suggests that targeting sparsely represented regions is particularly useful on this dataset. These results indicate that both anchor sources are useful in the coverageaware framework, but their contributions differ across datasets.

c) Cost-based vs. order-based candidate selection.: To test whether candidate-level scoring improves the usefulness of virtual samples for HAR training, we compare three selection strategies on the same candidate pool generated for each anchor. All three strategies operate on the same pool of up to M generated candidates. Cost-based selection ranks candidates by the selection cost defined from anchor proximity and target-label consistency and uses the $K _ { \mathrm { s e l } }$ candidates with the lowest costs. Generation-order does not use the cost and selects the first $K _ { \mathrm { s e l } }$ valid candidates in their original generation order. Random-order shuffles the valid candidates using a fixed random seed and uses the first $K _ { \mathrm { s e l } }$ candidates. Within each dataset, the three strategies use identical anchors, generated candidate pools, candidate budgets, candidate postprocessing, reliability weights, and downstream HAR settings; only the candidate ordering and resulting selection differ.

![](images/1507b661c3911f7a462b86f59c723f3076e26026365102e1be50607a0971d474.jpg)  
Fig. 4. Effect of candidate ordering across the three datasets.

On USC-HAD, Cost-based achieves a Macro-F1 of 80.94%, exceeding Generation-order and Random-order by 2.02 and 0.84 percentage points. Both order-based variants still outperform Real-only, showing that virtual candidates can benefit HAR training even without cost-based ordering. On USC-HAD, ordering candidates by selection cost provides a further gain. On PAMAP2, Cost-based achieves the highest mean Macro-F1 of 77.35%, exceeding the order-based baseline by 2.58 percentage points. On RealWorld, Generation-order achieves 71.92%, while Random-order achieves 71.78%. Both perform slightly better than Cost-based at 71.68%. Cost-based also has the lowest standard deviation on all three datasets: 0.68 on USC-HAD, 0.51 on PAMAP2, and 1.45 on RealWorld. Overall, Cost-based selection improves mean Macro-F1 on USC-HAD and PAMAP2 and shows the lowest observed variability on all three datasets. These results show that anchor proximity and target-label consistency can provide useful information for candidate selection, but their effectiveness varies across datasets.

d) Rank-aware vs. equal-rank weighting.: Table V evaluates whether candidate rank remains informative after candidate selection. Full Ours and Equal-rank use the same selected candidates and differ only in their within-anchor rank weighting. Full Ours assigns decreasing rank multipliers, whereas Equal-rank assigns the same multiplier to all selected candidates from the same anchor. This comparison tests whether selected candidates should receive equal or rankdependent training weights.

TABLE V  
COMPARISON OF RANK-AWARE AND EQUAL-RANK WEIGHTING. VALUES ARE MACRO-F1 (%); ∆ IS COMPUTED AS EQUAL-RANK MINUS FULL OURS.
<table><tr><td>Dataset</td><td>Full Ours</td><td>Equal-rank</td><td> $\Delta$ </td></tr><tr><td>USC-HAD</td><td> ${ \bf 8 0 . 9 4 \pm 0 . 6 8 }$ </td><td> $7 8 . 7 1 \pm 1 . 7 8$ </td><td>-2.23</td></tr><tr><td>PAMAP2</td><td> $7 7 . 3 5 \pm \mathbf { 0 . 5 1 }$ </td><td> $7 3 . 5 9 \pm 0 . 1 8$ </td><td>-3.76</td></tr><tr><td>RealWorld</td><td> ${ \bf 7 1 . 6 8 \pm 1 . 4 5 }$ </td><td> $7 0 . 1 3 \pm 1 . 9 5 - 1 . 5 5$ </td><td></td></tr></table>

Compared with the Full Ours reference, Equal-rank weighting is lower by 2.23, 3.76, and 1.55 percentage points in mean Macro-F1 on USC-HAD, PAMAP2, and RealWorld. The lower results under Equal-rank weighting indicate that the retained candidates should not receive identical weights. Higher-ranked candidates should receive larger weights than lower-ranked candidates.

## D. Data Efficiency Analysis

a) Label efficiency.: Fig. 5 compares Macro-F1 on USC-HAD, PAMAP2, and RealWorld at labeled real-data ratios of 5%, 10%, 20%, and 50%. For each method, we vary only the labeled real-data ratio in each LOSO training fold while keeping its augmentation or virtual-data pipeline and downstream training settings unchanged. At the 5% ratio, Ours improves over Real-only by 32.28 points on USC-HAD, 31.24 points on PAMAP2, and 24.32 points on RealWorld. The gains decrease to 15.03, 18.95, and 4.92 points at 10%, and to 6.81, 14.40, and 1.57 points at 20%. At 50%, the differences between Ours and Real-only are −0.06, −0.39, and −0.38 points, leaving their mean performance nearly identical. Other methods show a similar trend. At 5%, Diffusion-TS improves over Real-only by 23.87 points on USC-HAD, 25.92 points on PAMAP2, and 20.25 points on RealWorld. At 50%, the corresponding differences are −0.63, −5.40, and 0.02 points. The largest gains from augmentation and virtual data occur at low labeled real-data ratios. Ours achieves the highest mean Macro-F1 on all three datasets at 5%, 10%, and 20%, but its advantage over Real-only narrows as more labeled real data are used and is almost absent at 50%. Other methods show the same trend, and some settings exhibit slight negative transfer at higher labeled ratios. Virtual IMU data are most useful when labeled real data are limited; their benefit becomes small or negative as the real training set grows.

b) Virtual-data scaling.: Fig. 6 compares Macro-F1 with different numbers of selected virtual candidates per anchor. We fix the labeled real-data subset and vary only the maximum number of selected candidates per anchor, $K _ { \mathrm { s e l } } ~ \in$ {0, 1, 3, 5, 10}, where $K _ { \mathrm { s e l } } ~ = ~ 0$ corresponds to Real-only. USC-HAD and PAMAP2 use 20% labeled real data, whereas RealWorld uses 10%. All settings reuse the same candidate pool and ranking, and candidates shared across settings use the same training weights. With only the top-ranked candidate retained for each anchor, Macro-F1 improves over Real-only by 3.00 points on USC-HAD, 2.49 points on PAMAP2, and 2.24 points on RealWorld. At $K _ { \mathrm { s e l } } = 3 ,$ , the gains increase to 4.84, 6.20, and 4.29 points. Further increases in $K _ { \mathrm { s e l } }$ bring only small changes in performance. On USC-HAD, Macro-F1 increases from $7 9 . 3 6 \% \pm \ : 0 . 8 2 \%$ at $K _ { \mathrm { s e l } } = 3$ to 79.76% ± 1.87% at $K _ { \mathrm { s e l } } = 1 0$ . On PAMAP2, Macro-F1 reaches 72.65% $\pm \ 4 . 4 2 \%$ at $K _ { \mathrm { s e l } } = 5$ and then decreases to $7 1 . 9 8 \% \pm 2 . 1 7 \%$ at $K _ { \mathrm { s e l } } = 1 0$ . On RealWorld, Macro-F1 reaches 71.52% ± 1.26% at $K _ { \mathrm { s e l } } = 5$ and increases by only 0.03 points when $K _ { \mathrm { s e l } }$ is increased to 10.

![](images/42a46da647016a56e385cd574763d4b376194774dbfaaaca278bdadd1688190f.jpg)  
Fig. 5. Macro-F1 under different labeled real-data ratios on (a) USC-HAD, (b) PAMAP2, and (c) RealWorld.

![](images/6f94b3a4fc599521e1894f4b5aa02eb5f5e99cb112f32ad759653d5d544b3426.jpg)  
Fig. 6. Macro-F1 with different numbers of selected virtual candidates per anchor on USC-HAD, PAMAP2, and RealWorld. Error bars indicate the standard deviation across random seeds.

Most of the improvement comes from the first few highranked candidates. At $K _ { \mathrm { s e l } } = 3$ , USC-HAD and RealWorld already obtain about 92% and 90% of the total improvement observed at $K _ { \mathrm { s e l } } ~ = ~ 1 0$ , while PAMAP2 performs slightly better than at $K _ { \mathrm { s e l } } = 1 0$ . Thus, more virtual candidates do not necessarily lead to better performance. The results show that most of the gains come from a small number of top-ranked candidates, while adding lower-ranked candidates provides little further benefit and can even slightly degrade performance on some datasets.

## E. Feature Visualization

Fig. 7 visualizes the learned representations of real training samples and selected virtual IMU samples on USC-HAD, PAMAP2, and RealWorld. For each dataset, we use one fixed

LOSO outer fold and jointly project the learned representations of its real training and virtual samples into a two-dimensional t-SNE space; samples from the held-out subject are excluded. Colors denote activity classes, circles and triangles denote real and virtual samples, respectively, and triangle size indicates the effective training weight.

The three datasets show different relationships between real and virtual samples. On USC-HAD, real and virtual samples show noticeable separation, and virtual representations from different activities overlap substantially. On PAMAP2, the virtual samples show clearer class structure, although some remain separated from their corresponding real clusters. On RealWorld, real and virtual samples from the same activity often occupy nearby regions, with weaker separation by data source. The selected virtual samples show activity-related structure in the learned representation space. Their correspondence with real samples varies across datasets, consistent with the main results in the full LOSO evaluation. The t-SNE plots show how the selected virtual samples relate to real training samples, while the full LOSO evaluation measures their effect on HAR performance.

## F. Exploratory Extension to Medical Wearable IMU

Beyond the standard HAR benchmarks, we conduct an exploratory evaluation on PADS [33], [34], a wearable IMU dataset for Parkinson’s disease (PD) versus healthy-control (HC) classification. The evaluation uses both-wrist recordings from the StretchHold and HoldWeight tasks, with 20% labeled real data. We use a subject-level five-fold split and an RBF-SVM classifier; hyperparameters are selected by inner crossvalidation using only the training subjects in each outer fold.

Table VI reports the PADS results. Compared with Realonly, Ours improves Balanced Accuracy from 63.64% to 80.52% and Macro-F1 from 65.02% to 78.94%, corresponding to gains of 16.88 and 13.92 percentage points, respectively. Diffusion-TS reaches $7 7 . 2 7 \% \pm \ : 4 . 3 1 \%$ Balanced Accuracy and $7 7 . 2 2 \% \pm 4 . 5 3 \%$ Macro-F1, improving over Real-only by 13.63 and 12.19 percentage points, respectively. It outperforms Traditional Augmentation and TimeGAN on both metrics but remains below IMUGPT and Ours. Ours obtains the highest mean results, exceeding IMUGPT by 0.32 points in Balanced Accuracy and 0.66 points in Macro-F1. These results indicate that the proposed virtual IMU pipeline can extend to a lowlabel medical wearable classification task.

![](images/6f5e1e63af9d6464f6e05d660becba653c2e68ffef45cdb191d9127d6ae4ee56.jpg)  
Fig. 7. t-SNE visualization of low-label real training data and final selected virtual IMU samples on (a) USC-HAD, (b) PAMAP2, and (c) RealWorld. Colors denote activity classes, circles and triangles indicate real and virtual samples, respectively, and triangle size represents the effective training weight of each virtual sample.

TABLE VI  
PD-VS-HC CLASSIFICATION RESULTS ON PADS.
<table><tr><td>Method</td><td>Bal. Acc. ↑</td><td>Macro-F1 ↑</td><td>∆Macro-F1</td></tr><tr><td>Real-only Trad. Aug. TimeGAN Diffusion-TS</td><td> $6 3 . 6 4 \pm 8 . 9 0$   $7 3 . 8 2 \pm 1 0 . 4 7$   $7 6 . 8 2 \pm 9 . 8 4$ </td><td> $6 5 . 0 2 \pm 1 1 . 3 0$   $7 4 . 9 4 \pm 1 0 . 1 0$   $7 3 . 2 9 \pm 1 0 . 3 6$ </td><td>+9.92 +8.27 +12.19</td></tr><tr><td colspan="4">IMUGPT  $8 0 . 2 0 \pm 8 . 7 6$   $7 8 . 2 8 \pm 7 . 7 4$  +13.26 Ours  ${ \bf 8 0 . 5 2 \pm 9 . 0 8 }$   ${ \bf 7 8 . 9 4 \pm 8 . 6 4 }$  +13.92 Results use 20% labeled real data. Values are percentages reported as mean ± standard deviation across five subject- level outer folds. ∆F1 is computed from unrounded Macro-</td></tr></table>

## V. LIMITATIONS AND FUTURE WORK

## A. Limitations

a) Dependence on the upstream motion generator.: Our current implementation uses a fixed text-to-motion model to produce motions for virtual IMU synthesis. The scope and quality of the resulting candidate pool depend on the motion patterns represented by this generator. Motion patterns that are weakly represented in its training data may be generated less reliably.

b) Simplified sensor simulation.: Our current pipeline uses IMUSim to generate ideal accelerometer and gyroscope signals at the skeleton joint nearest to each target sensor location. It does not explicitly model changes in sensor placement and orientation during wear or device-specific noise and bias. These factors can change the measured IMU signals even when the underlying body motion is the same.

c) Limited real-world and clinical evaluation.: All experiments in this study are conducted offline using existing datasets. We have not evaluated the method in long-term deployments or under changes in real-world data collection conditions. The PADS experiment provides an exploratory evaluation of PD-versus-HC classification rather than a clinical validation.

## B. Future Work

a) Sensor-aware virtual IMU generation.: Future work could extend virtual IMU generation beyond a fixed motion-to-IMU conversion. The generation process could model sensor placement, orientation, and device-specific noise and bias. The goal is to make virtual IMU signals reflect changes in sensor configuration, rather than simply generating more candidates.

b) Self-evolving agent for virtual IMU generation.: A self-evolving agent could manage virtual IMU generation, evaluation, and refinement. The agent could record effective generation strategies and common failure patterns across activities. It could then update prompts, sampling rules, and candidate-selection criteria based on this experience. The agent could reuse and refine these strategies across activities and datasets.

c) Extending virtual IMU generation to clinical events.: Future work could apply virtual IMU generation to transient clinical events such as freezing of gait. For freezing of gait, generated sequences should cover the transition into and out of each episode, rather than only the event segment. The generation process could also include hard negatives such as slow walking, hesitation, and brief stops. This would add training examples near event boundaries instead of only increasing the number of positive events. Models trained with these data could be evaluated on event localization, duration estimation, and false-alarm rate.

## VI. CONCLUSION

In this paper, we proposed a coverage-aware virtual IMU augmentation framework for HAR under limited real training data. The framework selects diversity and scarcity anchors in a learned sensor embedding space, converts their dynamics attributes into prompts for virtual IMU generation, and assesses generated candidates using anchor proximity and label consistency. The selected candidates are then integrated into downstream HAR training with weights determined by their estimated reliability. Experiments on three public HAR benchmarks show consistent improvements over competitive baselines under limited labeled-data settings, while ablation studies examine where virtual candidates are generated, which candidates are retained, and how strongly they influence training. In the future, we will further study how to improve the reliability of virtual IMU generation under more diverse realworld sensing conditions.

## REFERENCES

[1] H. Xu, P. Zhou, R. Tan, and M. Li, “Practically adopting human activity recognition,” in Proceedings of the 29th Annual International Conference on Mobile Computing and Networking, 2023, pp. 1–15.

[2] C. Wang, Y. Feng, L. Zhong, S. Zhu, C. Zhang, S. Zheng, C. Liang, Y. Wang, C. He, C. Yu et al., “Ubiphysio: Support daily functioning, fitness, and rehabilitation with action understanding and feedback in natural language,” Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, vol. 8, no. 1, pp. 1–27, 2024.

[3] Q. Xie, H. Guo, W. Wang, Y. Huang, L. Jiang, J. Wu, S. Zhong, L. Wang, and K. Wu, “Harmony: A privacy-preserving and sensor-agnostic telemonitoring system,” in Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, 2025, pp. 9945–9953.

[4] Z. Leng, H. Kwon, and T. Plotz, “Generating virtual on-body ac-¨ celerometer data from virtual textual descriptions for human activity recognition,” in Proceedings of the 2023 ACM International Symposium on Wearable Computers, 2023, pp. 39–43.

[5] Z. Leng, A. Bhattacharjee, H. Rajasekhar, L. Zhang, E. Bruda, H. Kwon, and T. Plotz, “Imugpt 2.0: Language-based cross modality transfer for¨ sensor-based human activity recognition,” Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, vol. 8, no. 3, pp. 1–32, 2024.

[6] M. Chen, J. Du, B. Huang, Y. Wang, X. Zhang, and W. Wang, “Influence-guided diffusion for dataset distillation,” in The Thirteenth International Conference on Learning Representations, 2025.

[7] J. A. Chan-Santiago, P. Tirupattur, G. K. Nayak, G. Liu, and M. Shah, “MGD<sup>3</sup>: Mode-guided dataset distillation using diffusion models,” arXiv preprint arXiv:2505.18963, 2025.

[8] Y. Zhou, H. Zhao, Y. Huang, T. Roddiger, M. Kurnaz, T. Riedel, and¨ M. Beigl, “Autoaughar: automated data augmentation for sensor-based human activity recognition,” Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, vol. 8, no. 2, pp. 1–27, 2024.

[9] Z. Leng, A. Iyer, and T. Plotz, “Scaling human activity recognition: A¨ comparative evaluation of synthetic data generation and augmentation techniques,” arXiv preprint arXiv:2506.07612, 2025.

[10] N. Oishi, P. Birch, D. Roggen, and P. Lago, “Physically plausible data augmentations for wearable imu-based human activity recognition using physics simulation,” IEEE Sensors Journal, 2026.

[11] H. Oppel and M. Munz, “A diffusion model for inertial based time series generation on scarce data availability to improve human activit recognition,” Scientific Reports, vol. 15, no. 1, p. 16841, 2025.

[12] L. O. Haeusler, L. Uhlenberg, and O. Amft, “Text2imu: Advancing human activity recognition by text-driven imu data synthesis,” in 2025 IEEE 21st International Conference on Body Sensor Networks (BSN). IEEE, 2025, pp. 1–4.

[13] L. Uhlenberg, L. O. Haeusler, and O. Amft, “Synhar: Augmenting human activity recognition with synthetic inertial sensor data generated from human surface models,” IEEE Access, 2024.

[14] N. Oishi, P. Birch, D. Roggen, and P. Lago, “Wimusim: simulating realistic variabilities in wearable imus for human activity recognition,” Frontiers in Computer Science, vol. 7, p. 1514933, 2025.

[15] H. Duan, S. Wang, V. Ojha, S. Wang, Y. Huang, Y. Long, R. Ranjan, and Y. Zheng, “Wearable-based behaviour interpolation for semi-supervised human activity recognition,” Information Sciences, vol. 665, p. 120393, 2024.

[16] C. I. Tang, I. Perez-Pozuelo, D. Spathis, S. Brage, N. Wareham, and C. Mascolo, “Selfhar: Improving human activity recognition through self-training with unlabeled data,” Proceedings of the ACM on interactive, mobile, wearable and ubiquitous technologies, vol. 5, no. 1, pp. 1–30, 2021.

[17] R. Hu, L. Chen, S. Miao, and X. Tang, “Swl-adapt: An unsupervised domain adaptation model with sample weight learning for cross-user wearable human activity recognition,” in Proceedings of the AAAI Conference on artificial intelligence, vol. 37, no. 5, 2023, pp. 6012– 6020.

[18] D. Xiong, S. Wang, L. Zhang, W. Huang, and C. Han, “Generalizable sensor-based activity recognition via categorical concept invariant learning,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 1, 2025, pp. 923–931.

[19] Y. Cai, R. Feng, A. Yu, B. Guo, and Z. Hong, “Benchhar: Benchmarking self-supervised learning for generalizable sensor-based activity recognition,” arXiv preprint arXiv:2605.08296, 2026.

[20] Z. Zhang and M. Sabuncu, “Generalized cross entropy loss for training deep neural networks with noisy labels,” Advances in neural information processing systems, vol. 31, 2018.

[21] B. Han, Q. Yao, X. Yu, G. Niu, M. Xu, W. Hu, I. Tsang, and M. Sugiyama, “Co-teaching: Robust training of deep neural networks with extremely noisy labels,” Advances in neural information processing systems, vol. 31, 2018.

[22] Y. Li, H. Han, S. Shan, and X. Chen, “Disc: Learning from noisy labels via dynamic instance-specific selection and correction,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 24 070–24 079.

[23] Z. Zhang, W. Chen, C. Fang, Z. Li, L. Chen, L. Lin, and G. Li, “Rankmatch: Fostering confidence and consistency in learning with noisy labels,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 1644–1654.

[24] M. Sheng, Z. Sun, T. Chen, S. Pang, Y. Wang, and Y. Yao, “Foster adaptivity and balance in learning with noisy labels,” in European Conference on Computer Vision. Springer, 2024, pp. 217–235.

[25] W. Pan, W. Wei, F. Zhu, and Y. Deng, “Enhanced sample selection with confidence tracking: Identifying correctly labeled yet hard-to-learn samples in noisy data,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 19, 2025, pp. 19 795–19 803.

[26] S. Yuan, L. Feng, B. Han, and T. Liu, “Enhancing sample selection by cutting mislabeled easy examples,” arXiv e-prints, pp. arXiv–2502, 2025.

[27] K. Chen, D. Zhang, L. Yao, B. Guo, Z. Yu, and Y. Liu, “Deep learning for sensor-based human activity recognition: Overview, challenges, and opportunities,” ACM Computing Surveys (CSUR), vol. 54, no. 4, pp. 1–40, 2021.

[28] J. Zhang, Y. Zhang, X. Cun, Y. Zhang, H. Zhao, H. Lu, X. Shen, and Y. Shan, “T2M-GPT: Generating human motion from textual descriptions with discrete representations,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 14 730–14 740.

[29] A. D. Young, M. J. Ling, and D. K. Arvind, “IMUSim: A simulation environment for inertial sensing algorithm design and evaluation,” in Proceedings of the 10th ACM/IEEE International Conference on Information Processing in Sensor Networks. IEEE, 2011, pp. 199–210.

[30] M. Zhang and A. A. Sawchuk, “Usc-had: A daily activity dataset for ubiquitous activity recognition using wearable sensors,” in Proceedings of the 2012 ACM conference on ubiquitous computing, 2012, pp. 1036– 1043.

[31] A. Reiss and D. Stricker, “Introducing a new benchmarked dataset for activity monitoring,” in 2012 16th international symposium on wearable computers. IEEE, 2012, pp. 108–109.

[32] T. Sztyler and H. Stuckenschmidt, “On-body localization of wearable devices: An investigation of position-aware activity recognition,” in 2016 IEEE international conference on pervasive computing and communications (PerCom). IEEE, 2016, pp. 1–9.

[33] J. Varghese, A. Brenner, M. Fujarski, C. M. van Alen, L. Plagwitz, and T. Warnecke, “Machine learning in the parkinson’s disease smartwatch (pads) dataset,” npj Parkinson’s Disease, vol. 10, no. 1, p. 9, 2024.

[34] J. Varghese, A. Brenner, L. Plagwitz, C. van Alen, M. Fujarski, and T. Warnecke, “PADS - Parkinsons Disease Smartwatch dataset,” PhysioNet, Mar. 2024, version 1.0.0. [Online]. Available: https: //doi.org/10.13026/m0w9-zx22

[35] A. Brenner, M. Fujarski, T. Warnecke, and J. Varghese, “Reducing a complex two-sided smartwatch examination for parkinson’s disease to an efficient one-sided examination preserving machine learning accuracy,” arXiv preprint arXiv:2205.05361, 2022.

[36] F. J. Ordo´nez and D. Roggen, “Deep convolutional and LSTM recurrent˜ neural networks for multimodal wearable activity recognition,” Sensors, vol. 16, no. 1, p. 115, 2016.

[37] V. S. Murahari and T. Plotz, “On attention models for human activity¨ recognition,” in Proceedings of the 2018 ACM International Symposium on Wearable Computers, 2018, pp. 100–103.

[38] Y. Zhou, T. King, H. Zhao, Y. Huang, T. Riedel, and M. Beigl, “MLP-HAR: Boosting performance and efficiency of HAR models on edge devices with purely fully connected layers,” in Proceedings of the 2024 ACM International Symposium on Wearable Computers, 2024, pp. 133– 139.

[39] J. Yoon, D. Jarrett, and M. van der Schaar, “Time-series generative adversarial networks,” in Advances in Neural Information Processing Systems, vol. 32, 2019.

[40] X. Yuan and Y. Qiao, “Diffusion-TS: Interpretable diffusion for general time series generation,” in The Twelfth International Conference on Learning Representations, 2024.