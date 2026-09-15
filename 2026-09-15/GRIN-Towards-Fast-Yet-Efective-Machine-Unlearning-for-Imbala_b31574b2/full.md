# GRIN+: Towards Fast Yet Efective Machine Unlearning for Imbalanced Medical Data

Minghui Huang<sup>a</sup>, Junxiao Wang<sup>a,∗</sup>

<sup>a</sup>Guangzhou University, China

A R T I C L E I N F O

Keywords: Machine Unlearning Medical Artificial Intelligence

## A BS T RA C T

As deep learning models become fundamental to modern healthcare, the “Right to be Forgotten” mandated by privacy regulations like GDPR and HIPAA necessitates efective machine unlearning (MU) to remove sensitive patient data from trained models. However, existing MU techniques often struggle with a fundamental “privacy-eficiency-utility” (PEU) trilemma, particularly in medical scenarios where data is frequently characterized by severe class imbalance and long-tailed distributions. In such cases, standard unlearning methods can fail to protect key clinical knowledge or mistakenly delete features essential for diagnosing rare conditions due to the gradient dominance of majority classes. To address these challenges, we propose GRIN+, a novel machine unlearning framework designed for fast and precise data erasure in imbalanced medical scenarios. GRIN+ decouples unlearningspecific knowledge from generalized representations at the parameter level by analyzing the gradient contributions of both “forget” and “retain” sets. It introduces a class-adaptive influence scoring mechanism to rectify gradient dominance and employs a direction-constrained update strategy to prevent the unintended erosion of vital clinical knowledge. Comprehensive benchmarking across multiple medical datasets, including skin cancer (ISIC), brain tumor (MRI), and breast ultrasound (BUSI), demonstrates that GRIN+ achieves an optimal balance of the PEU trilemma. Experimental results show that GRIN+ maintains high diagnostic accuracy and robust privacy while significantly enhancing runtime eficiency compared to existing baselines. We open-source the GRIN+ code and benchmarks to support further research.

## 1. INTRODUCTION

Background. As medical AI rapidly enters clinical use, deep learning models have become a cornerstone of modern healthcare [15]. They facilitate critical tasks such as medical image analysis (including MRI [22], ultrasound [8], and skin screening [14]), early disease detection [36], and clinical decision-making [1]. Since these models are trained on sensitive patient data, including images and clinical metadata, they are subject to strict international privacy laws. For instance, the GDPR [27] and HIPAA [3] grant patients the “Right to be Forgotten”, allowing them to request the total removal of their data. This creates a significant hurdle for deep learning: removing data from a database does not erase its impact on a trained model.

Studies [10] have reported that models often retain traces of individual training samples within their architecture. In a medical context, this residual data can be targeted by membership inference attacks [9], risking the exposure of private patient information and leading to serious legal and clinical consequences.

Machine unlearning [6] has been proposed as a solution to these challenges, aiming to precisely and eficiently remove the influence of specific training data from pretrained models. Existing techniques generally fall into two categories: exact unlearning [19] and approximate unlearning [21]. Exact unlearning focuses on providing certified deletion or provable guarantees. Within this category, retraining the model from scratch on the retained data is considered the gold standard [34]. Nevertheless, such retrainingbased methods incur excessive computational costs. This makes them increasingly impractical for clinical settings that demand low latency and frequent model updates [24].

Alternatively, approximate machine unlearning ofers a practical way to improve eficiency. These methods avoid the burden of full retraining by fine-tuning model parameters, utilizing techniques like influence functions [23] or Fisher Information Matrices (FIM) [28] to estimate parameter importance. While they have made impressive contributions in standard computer vision tasks, current validation remains largely confined to vanilla datasets like CIFAR [33, 18, 37, 35]. This suggests that their eficiency gains typically rely on the implicit assumption that the underlying training data is class-balanced.

Motivation. The assumption of balanced data, however, is often violated in medical imaging tasks. Clinical data in the real world usually follows a long-tailed distribution characterized by severe class imbalance (as shown in the lower left of Figure 1). In these cases, minority classes (like rare diseases) have very few samples but carry much greater clinical importance [32]. During training, gradients are dominated by majority-class samples, which weakens the model’s ability to recognize rare but critical pathological patterns. Benchmarks such as CXR-LT [26] have demonstrated that standard models tend to prioritize head-class performance while neglecting the tail classes, a problem that intensifies during the unlearning process [39]. Eficiencyoriented methods like influence functions and FIM typically rely on the implicit assumption of class balance [23, 28]. This causes their parameter importance estimates to skew toward majority classes. On imbalanced medical data, these methods fail to protect key clinical knowledge and may mistakenly delete features essential for diagnosing rare conditions. Consequently, existing unlearning solutions cannot fully meet clinical requirements, while maintaining eficiency and reliability in long-tailed medical scenarios, as illustrated in the lower right of Figure 1.

![](images/ac240f7b9b2707f88e3358a27214d5d4ff0a16c12aa945db8872b3c4b19d595c.jpg)  
Figure 1: Limitations of approximate machine unlearning in medical scenarios. Top: Existing methods rely on uniform parameter updates and iterative optimization, leading to two critical issues: (1) gradient dominance from majority classes in imbalanced data suppresses minority representations, causing degraded recognition of rare diseases (Misdiagnosis); (2) high computational overhead results in significant latency, hindering timely privacy compliance (Violation). Bottom: Left: Class distribution comparison between CIFAR-10 and the highly imbalanced ISIC dataset. Right: Unlearning performance comparison. RTE (x-axis, higher is better) vs. R-Acc (y-axis, higher is better); bubble size directly reflects |MIA − 0.5| (the smaller and lighter color the bubble, the better the unlearning efectiveness and the lower the privacy risk). Top-right with small and light-colored bubbles indicates fast and efective unlearning.

As our first contribution, we identify a fundamental “privacy-eficiency-utility” (PEU) trilemma inherent in machine unlearning for imbalanced medical data. Specifically, an ideal unlearning framework must fulfill three core requirements: (1) Privacy: the unlearned model must not leak membership information from the “forget set”, thereby complying with stringent data protection laws; (2) Eficiency: the unlearning process must be suficiently rapid to support clinical environments that demand low latency and frequent model updates; and (3) Utility: the resulting model must retain high diagnostic performance, even when navigating the complexities of imbalanced medical data.

Our second contribution: Motivated by the aforementioned trilemma, we propose GRIN+, a novel machine unlearning framework designed for fast and efective data erasure in imbalanced medical scenarios. GRIN+ explicitly decouples unlearning-specific knowledge from generalized representation capabilities at the parameter level. By jointly analyzing the gradient contributions of both the “forget” and “retain” sets, the framework quantifies parameter-wise influence to pinpoint a sparse subset closely associated with the target data, thereby mitigating parameter misattribution. Furthermore, GRIN+ introduces a class-adaptive influence scoring mechanism to rectify the gradient dominance of majority classes inherent in long-tailed distributions. Integrated with a direction-constrained update strategy, this mechanism prevents the unintended erosion of vital clinical knowledge during parameter adjustment. Ultimately, by executing directional updates based on gradient disparities, GRIN+ precisely eliminates target information while preserving stable diagnostic performance.

As our third contribution, we comprehensively benchmark GRIN+ across multiple medical datasets, and the results demonstrate that GRIN+ achieves an optimal balance between unlearning eficiency, model utility, and privacy. Specifically, the resulting models maintain high accuracy on both the retain and test sets. Notably, the Membership

Inference Attack (MIA) score reaches 49.90%, nearly equivalent to a random guess, indicating robust privacy preservation. Furthermore, GRIN+ significantly enhances runtime eficiency compared to baselines. Finally, comprehensive ablation studies confirm the indispensable contribution of each individual component to the overall performance.

This paper can be summarized as three-fold:

• Problem. We first reveal a “privacy-eficiency-utility” (PEU) trilemma in machine unlearning for imbalanced medical data, requiring a PEU instance to simultaneously achieve privacy (preventing data leakage from the forget set), eficiency (ensuring rapid updates for clinical use), and utility (maintaining high diagnostic accuracy despite data imbalances).

• Method. To address the PEU trilemma, we propose GRIN+, an unlearning framework that achieves fast, precise data erasure in imbalanced medical scenarios by decoupling unlearning-specific knowledge from general representations at the parameter level, utilizing a sparse subset selection based on gradient analysis of “forget” and “retain” sets, and employing a classadaptive influence scoring mechanism with directionconstrained updates to prevent majority-class dominance and preserve vital clinical performance.

• Results. We conduct comprehensive benchmarks and demonstrate that GRIN+ achieves an optimal balance of the PEU trilemma by maintaining high diagnostic accuracy, ensuring robust privacy, and significantly improving runtime eficiency.

Open-source. We release the GRIN+ code<sup>1</sup> to the community for further research.

## 2. Related Work

## 2.1. Machine Unlearning

Machine unlearning [6] aims to precisely and eficiently remove the influence of specific training data from pretrained models. Existing methods are mainly divided into exact unlearning [19] and approximate unlearning [21].

While exact unlearning provides strict data deletion guarantees, early methods like SISA [6], which use data sharding and submodel retraining, face significant hurdles. They disrupt standard training pipelines and impose high storage and computational costs, making them dificult to scale for medical AI. Consequently, approximate unlearning has become the primary focus for improving eficiency. These techniques use tailored loss functions to update parameters so that the model mimics a retrained state while protecting against membership inference [9] and reconstruction attacks [4]. Representative approaches include CF-K [17] (freezing shallow layers) and SCRUB [25] (using distillation and adversarial learning). Although more efficient than retraining, these methods still rely on global fine-tuning, which can be costly. Moreover, managing the delicate balance between forgetting and retaining knowledge remains a challenge, often leading to a trade-of where either unlearning is incomplete or the model’s accuracy on the remaining data is compromised.

## 2.2. Unlearning in Medical Context

Closest to our research are strategies that involves parameter importance estimation followed by selective updates, another mainstream paradigm in approximate unlearning. These methods typically leverage influence functions (approximating the inverse Hessian) [23, 37] or the Fisher Information Matrix (FIM) [28, 18, 33] to quantify the impact of individual training samples on model predictions. Nevertheless, these techniques face significant hurdles. First, calculating inverse Hessian or Hessian-vector products is still too expensive for large-scale deep models. Second, influence functions are unstable in deep, non-convex landscapes, especially when dealing with a nearly singular Hessian [5]. In medical contexts where data is long-tailed, rare diseases have few samples and low curvature, making the importance estimates highly unreliable. Finally, because FIM is calculated as an expectation over the entire data distribution, it is easily biased by class imbalance. Since head classes dominate the gradients, the importance of majority-class features is overestimated. Consequently, the model may mistakenly delete information vital for recognizing rare pathologies.

Research specifically tailored to medical scenarios remains limited and somewhat fragmented. While some approaches utilize low-rank adaptation or distillation for selective erasure [12], or adjust decision boundaries through bilevel optimization [29], these methods often rely on auxiliary constraints or teacher models, making them vulnerable to the unstable boundaries inherent in imbalanced data. Other studies remain confined to niche tasks, such as reconstruction [38], or specific frameworks like federated [13] and multimodal unlearning [20], which restricts their general applicability. In summary, machine unlearning in medical imaging continues to face two formidable challenges: biased parameter importance estimation under long-tailed distributions, and the persistent dificulty in reconciling eficiency with model utility.

## 3. Preliminaries

We formalize the machine unlearning problem in the medical domain as follows:

Given a training set $D _ { \mathrm { t r a i n } } ~ = ~ \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ and a pretrained model  (called the original model) with parameters $\theta ^ { 0 }$ , the core goal of machine unlearning (MU) is to eliminate the influence of a specific subset $\begin{array} { r l } { \boldsymbol D _ { f } } & { { } = } \end{array}$ $\{ ( x _ { j } ^ { ( f ) } , y _ { j } ^ { ( f ) } ) \} _ { j = 1 } ^ { M } \subseteq D _ { \operatorname { t r a i n } }$ . This subset, defined as the forget set, corresponds to a single patient’s record or a small batch of patient data, with � ≪ � holding true. The remaining part of the training set, $D _ { r } = D _ { \mathrm { t r a i n } } \setminus D _ { f } .$ , is named the retain set. In a medical context, $D _ { r }$ statistically reflects the knowledge distribution the model needs to retain.

![](images/306dec42bdaac3b4b277788fcd73ee50c88734cc4548b54e7b43eff152e54fcc.jpg)  
Figure 2: Overview of the proposed GRIN+ framework. After a patient requests the removal of relevant private data, the framework first partitions the data into retain and forget sets. It then estimates parameter-wise influence using gradient-based scores to identify key parameters. Finally, constrained perturbations are applied to these parameters to update the model, enabling selective knowledge removal while preserving performance on the retained data.

Summary of main notations.
<table><tr><td>Symbol</td><td>Description</td></tr><tr><td> $\mathcal { D } _ { \mathrm { t r a i n } }$ </td><td>Full training dataset</td></tr><tr><td> $\scriptstyle { \mathcal { D } } _ { r }$   $D _ { f }$ </td><td>Retain (remaining) dataset Forget dataset</td></tr><tr><td> $( x , y )$   $\mathcal { M } _ { U }$ </td><td>Input sample and corresponding label Unlearned model</td></tr><tr><td> $\pmb { \theta } ^ { 0 }$ </td><td>Original trained model parameters</td></tr><tr><td> $\mathcal { L } ( \cdot )$   $\nabla _ { \boldsymbol { \theta } } \mathcal { L }$ </td><td>Loss function Gradient of loss w.r.t. parameters</td></tr><tr><td> $G _ { f }$ </td><td>Gradient computed on forget set</td></tr><tr><td> $\mathbf { G } _ { \scriptscriptstyle r }$ </td><td>Gradient computed on retain set</td></tr><tr><td></td><td></td></tr><tr><td> $S _ { i }$ </td><td>Gradient influence score for parameter i</td></tr><tr><td> $p$ </td><td>Selection ratio</td></tr><tr><td> $| \Theta |$ </td><td>Total number of parameters</td></tr><tr><td> $\delta _ { i }$ </td><td>Parameter update perturbation</td></tr><tr><td></td><td></td></tr><tr><td> $\| \cdot \|$ </td><td>Norm operator (default:  $\ell _ { 2 }$  norm)</td></tr><tr><td> $\odot$ </td><td>Element-wise multiplication</td></tr></table>

These two sets are disjoint and complementary components of the original training set. Applying MU yields an unlearned model $\mathcal { M } _ { U }$ . Its objective is to achieve performance comparable to a model retrained from scratch on $D _ { r }$ . We use ${ \bf \bar { \theta } } ^ { 0 }$ and $\theta _ { r }$ to represent the weights of the original and retrained models, respectively. For evaluation, we introduce two held-out sets: the validation set $\mathcal { D } _ { V }$ and the test set $D _ { T }$ Both sets follow the same distribution as $D _ { \mathrm { t r a i n } }$ . We regard the retrained model’s accuracy as the optimal standard. $\mathbf { A }$ key assumption here is that the MU method can access $\theta ^ { 0 }$

$D _ { r } ,$ and $\mathcal { D } _ { V }$ . The main symbols used throughout this paper are summarized in Table 1.

## 4. Methodology

The core of our method $\mathrm { G R I N } { + ^ { 2 } }$ is that, given a pretrained model, we quantify the relative importance of each parameter for the forgetting operation by analyzing the gradient diferences of the loss function on the forget set and a representative retain set. Based on this, we apply only small, constrained perturbations in a specific direction to a filtered subset of critical parameters, thereby efectively ‘erasing the memory of target samples while preserving the stability of the model’s overall performance. The overview of the pipeline is shown in Figure 2.

## 4.1. Gradient Ratio-Based Adaptive Influence Estimation

We first compute the loss function gradients of the model with respect to $D _ { f }$ and $D _ { r }$ at the current parameters:

$$
\begin{array} { r } { \pmb { G } _ { f } = \nabla _ { \theta } \pmb { \mathcal { L } } ( \boldsymbol { D } _ { f } ; \theta ) , \quad \pmb { G } _ { r } = \nabla _ { \theta } \pmb { \mathcal { L } } ( \boldsymbol { D } _ { r } ; \theta ) , } \end{array}\tag{1}
$$

where $\mathcal { L }$ is the model’s loss function $( \mathrm { e . g . }$ , cross-entropy loss), and ${ \cal { G } } _ { f } , { \cal { G } } _ { r }$ are gradient vectors with the same dimensionality as the parameters.

Based on the intuition that “ideal forget parameters should have large gradients on $D _ { f }$ (sensitive and easily modified) and small gradients on $D _ { r }$ (harmless to modify)”, we define a base influence score for each parameter $\theta _ { i } .$ . To address the prevalent class imbalance problem in medical data, we introduce a class balancing factor for adaptive adjustment:

$$
\begin{array} { r } { s _ { i } ^ { \mathrm { b a s e } } = \frac { | \boldsymbol { G } _ { f , i } | } { | \boldsymbol { G } _ { r , i } | + \varepsilon } , \quad S _ { i } = s _ { i } ^ { \mathrm { b a s e } } \cdot w _ { c } . } \end{array}\tag{2}
$$

Here, $| G _ { f , i } |$ and $| \pmb { G } _ { r , i } |$ represent the magnitudes of the �-th parameter dimension components of gradients $G _ { f }$ and ${ \bf G } _ { r }$ , respectively. � is a small positive number for numerical stability, set to the �-th percentile of the sequence of absolute values of all components of $G _ { f }$ (empirically $k = 5 ) . w _ { c }$ is the weight of class � to which the forget samples belong, defined as:

$$
w _ { c } = \frac { N } { C \cdot n _ { c } } ,\tag{3}
$$

where � is the total number of samples in the forget set $D _ { f } ,$ $C$ is the total number of classes, and $n _ { c }$ is the number of samples of class �. This weight assigns higher weights to minority class samples and lower weights to majority class samples, thereby preventing the forgetting operation from excessively damaging the model’s recognition capability for rare diseases or uncommon cases (minority classes) due to gradient dominance by majority class samples. A higher $S _ { i }$ value indicates that parameter $\theta _ { i }$ better conforms to the “high forget influence, low retain relevance” criterion, and its update’s impact on class balance has been corrected, making it the preferred target for intervention.

## 4.2. Selection of Critical Parameter Subset

All parameters are sorted in descending order according to their adaptive influence scores $S _ { i } . \mathrm { A }$ selection ratio $p \in$ (0, 1] is set $( \mathrm { e } . \mathrm { g } . , p = 0 . 1 \mathrm { o r } 0 . 2 )$ , and the top $p { \cdot } | \Theta |$ parameters are selected to constitute the critical parameter subset. To formally represent this selection, we define a binary mask vector $\pmb { M } \in \{ 0 , 1 \} ^ { | \Theta | }$

$$
M _ { i } = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ } } i { \mathrm { ~ b e l o n g s ~ t o ~ t h e ~ t o p ~ } } p \% } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }\tag{4}
$$

Only parameters $\theta _ { i }$ for which $M _ { i } ~ = ~ 1$ will be updated in subsequent steps.

## 4.3. Regularization-Constrained and Directed Parameter Perturbation

To achieve efective forgetting of $D _ { f }$ while strictly minimizing potential damage to the generalization capability on $\textstyle D _ { r } ,$ we apply perturbations along the gradient diference direction, but introduce a regularization term to constrain the update direction.

First, compute the gradient direction diference for each parameter:

$$
\Delta G _ { i } = G _ { f } ^ { ( i ) } - G _ { r } ^ { ( i ) } .\tag{5}
$$

Next, generate the perturbation vector. To ensure the update does not severely deviate from the optimization direction of retain data, we compute a direction constraint factor

$r _ { i }$ for each parameter:

$$
r _ { i } = \operatorname* { m a x } \left( 0 , 1 - \beta \cdot \left| \cos ( \Delta G _ { i } , G _ { r } ^ { ( i ) } ) \right| \right) ,\tag{6}
$$

where cos(⋅) calculates the sign consistency between two scalars (here, gradient components), with a negative value when their directions are opposite. | cos(⋅)| measures the intensity of opposite directionality. � is a sensitivity hyperparameter (empirically set to 0.5). When $\Delta G _ { i }$ strongly opposes $G _ { r } ^ { ( i ) }$ (i.e., in a direction that would significantly increase loss on $D _ { r } ) , r _ { i }$ decreases, thereby suppressing the perturbation magnitude on that parameter. This efectively prevents “destroying” the model’s knowledge on important retain features in pursuit of forgetting.

Finally, the applied perturbation is:

$$
\delta _ { i } = \alpha \cdot \Delta G _ { i } \cdot M _ { i } \cdot r _ { i } , \quad \theta _ { i } ^ { \mathrm { n e w } } = \theta _ { i } ^ { \mathrm { o l d } } + \delta _ { i } ,\tag{7}
$$

where � is a hyperparameter controlling the overall perturbation scale. After incorporating the direction constraint factor $r _ { i } ,$ the tuning of � becomes more robust, and its optimal value is determined through cross-validation on validation set, aiming to optimize the trade-of between unlearning efectiveness (e.g., MIA score approaching 0.5) and knowledge preservation (e.g., accuracy maintenance on $\textstyle { \mathcal { D } } _ { r }$ and independent test sets). The above process can be summarized as Algorithm 1.

## 4.4. Theoretical Motivation

To demonstrate the rationality of our chosen parameter update direction (based on gradient diference $\Delta G =$ $G _ { f } \mathrm { ~ - ~ } G _ { r } )$ , we formalize the forgetting objective from an optimization theory perspective. We aim to find a parameter update that satisfies two objectives simultaneously: 1) Promote forgetting: Significantly increase the model’s loss on the forget data $D _ { f }$ . This degrades the model’s distinguishability on $\mathcal { D } _ { f } . 2 )$ Preserve knowledge: Minimize loss changes on the retain set $\textstyle { \mathcal { D } } _ { r }$ . This maintains the model’s original generalization capability.

This can be formulated as the following constrained optimization problem:

$$
\operatorname* { m i n } _ { \delta } \| \delta \| ^ { 2 } , \quad \mathrm { s . t . } \left\{ \begin{array} { l l } { \mathcal { L } ( \mathcal { D } _ { f } ; \theta + \delta ) - \mathcal { L } ( \mathcal { D } _ { f } ; \theta ) \ge \Delta _ { f } , } \\ { \mathcal { L } ( \mathcal { D } _ { r } ; \theta + \delta ) - \mathcal { L } ( \mathcal { D } _ { r } ; \theta ) \le \Delta _ { r } , } \end{array} \right.\tag{8}
$$

where $\Delta _ { f } ~ > ~ 0$ is the lower bound of desired forgetting intensity. $\Delta _ { r } \geq 0$ is the upper bound of tolerable retention performance loss. $\| \cdot \|$ denotes the $\ell _ { 2 }$ norm of a vector. Minimizing $\| \delta \| ^ { 2 }$ ensures the parameter update is as small as possible, preventing drastic model changes.

Assuming the update magnitude is suficiently small, we can approximate the loss function change using a first-order Taylor expansion:

$$
\begin{array} { r l } & { \mathcal { L } ( D _ { f } ; \theta + \delta ) \approx \mathcal { L } ( D _ { f } ; \theta ) + G _ { f } ^ { T } \delta , } \\ & { \mathcal { L } ( D _ { r } ; \theta + \delta ) \approx \mathcal { L } ( D _ { r } ; \theta ) + G _ { r } ^ { T } \delta , } \end{array}\tag{9}
$$

where $G _ { f } = \nabla \mathcal { L } ( D _ { f } ; \theta )$ and $G _ { r } = \nabla \mathcal { L } ( D _ { r } ; \theta )$ . Substituting into the constraint conditions, we obtain:

$$
\begin{array} { r } { G _ { f } ^ { T } \delta \geq \Delta _ { f } , \quad G _ { r } ^ { T } \delta \leq \Delta _ { r } . } \end{array}\tag{10}
$$

Algorithm 1 GRIN+   
Input: Pre-trained model parameters �; medical data to be   
forgotten $D _ { f } ;$ retain set $D _ { r } ;$ selection ratio $p ;$ perturba  
tion coeficient �; regularization sensitivity �; percentile   
� (default = 5).   
Output: Post-forgetting model parameters $\theta _ { \mathrm { n e w } } .$   
1: 1. Compute gradients and class weights:   
2: $G _ { f } \gets \nabla _ { \theta } \mathcal { L } ( D _ { f } ; \theta )$   
3: $\dot { G _ { r } } \gets \nabla _ { \theta } \mathcal { L } ( D _ { r } ; \theta )$   
4: Count class distribution of $D _ { f } ,$ , compute weight $w _ { c }$   
for each class �.   
5: 2. Compute adaptive influence scores:   
6: � ← Percentile(|� |, �)   
7: for � = 1 to $| \theta |$ do   
8: $s _ { i } ^ { \mathrm { b a s e } } \gets | G _ { f , i } | / ( | G _ { r , i } | + \varepsilon )$   
9: $S _ { i }  s _ { i } ^ { \mathrm { b a s e } } \cdot w _ { c }$ // where � is the class of   
forget samples   
10: end for   
11: 3. Select critical parameters:   
12: Sort parameters by $S _ { i }$ in descending order.   
13: Create mask �, set $M _ { i } = 1$ for top $p \times | \theta |$ parameters,   
others 0.   
14: 4. Compute and apply regularized perturbation:   
15: $\Delta G \gets G _ { f } - G _ { r }$   
16: for � = 1 to |�| do   
17: Compute direction constraint factor:   
$r _ { i } \gets \operatorname* { m a x } \left( 0 , 1 - \beta \cdot \left| \mathrm { s i g n } ( \Delta G _ { i } , G _ { r , i } ) \right| \right) \quad \mathrm { ~ / / ~ }$ sign is   
the sign function   
18: $\delta _ { i } \gets \alpha \cdot \Delta G _ { i } \cdot M _ { i } \cdot r _ { i }$   
19: $\theta _ { i } ^ { \mathrm { n e w } }  \theta _ { i } + \delta _ { i }$   
20: end for   
21: 5. Return $\theta _ { \mathrm { n e w } }$

We want to maximize $G _ { f } ^ { T } \delta$ to promote forgetting. At the same time, we aim to minimize $G _ { r } ^ { T } \delta$ to preserve knowledge. A natural compromise is to maximize the following linear combination used as a surrogate objective:

$$
\boldsymbol { J } ( \delta ) = \boldsymbol { G } _ { f } ^ { T } \delta - \boldsymbol { G } _ { r } ^ { T } \delta = ( \boldsymbol { G } _ { f } - \boldsymbol { G } _ { r } ) ^ { T } \delta .\tag{11}
$$

Considering the update magnitude constraint, we construct the following unconstrained optimization problem (introducing Lagrange multiplier $\lambda > 0 )$

$$
\operatorname* { m i n } _ { \delta } \| \delta \| ^ { 2 } - \lambda ( G _ { f } - G _ { r } ) ^ { T } \delta .\tag{12}
$$

Taking the derivative of this objective function with respect to � and setting it to zero:

$$
2 \delta - \lambda ( G _ { f } - G _ { r } ) = 0 \Longrightarrow \delta = \frac { \lambda } { 2 } ( G _ { f } - G _ { r } ) .\tag{13}
$$

This suggests that under the premise of a suficiently small update and first-order approximation, the optimal update direction follows the gradient diference $G _ { f } - G _ { r }$ . In our algorithm, the hyperparameter � plays a role similar to $\lambda / 2$ controlling the update step size. Note that in practice, we only apply updates to a sparse subset of critical parameters (defined by mask �). Therefore, the complete update formula is $\delta = \alpha { \cdot } ( G _ { f } { - } G _ { r } ) \odot M$ , where ⊙ denotes element-wise multiplication. This sparsity allows us to modify parameters most efective for forgetting more precisely. It further reduces unnecessary perturbations, aligning with our sparsity assumption.

## 5. Experiments

To comprehensively evaluate the efectiveness of GRIN+, we conducted extensive experiments on multiple medical datasets. The following sections aim to answer the following three research questions (RQs):

RQ1: (Overall Performance). Compared to representative unlearning baselines, can our method achieve higher global accuracy and better privacy performance while maintaining comparable or faster speed?

RQ2: (Computational Eficiency). What is the actual computational eficiency of GRIN+ compared to existing unlearning baselines?

RQ3: (Component Efectiveness). How much does each key component contribute to the final result? Do they provide consistent gains in ablation studies?

## 5.1. Experimental Setup

Datasets. We evaluate our method on three publicly available medical imaging datasets, covering diverse imaging modalities and clinical tasks to comprehensively assess unlearning performance. (1) Skin Cancer Dataset (ISIC) [11]. This dataset contains 2,357 dermoscopic images across nine categories of skin lesions, including melanoma and basal cell carcinoma. (2) Brain Tumor Magnetic Resonance Images Dataset (MRI) [31]. Consisting of 7,023 brain MRI scans classified into glioma, meningioma, pituitary tumor, and no tumor. (3) Breast Ultrasound Images Dataset (BUSI) [2]. This dataset includes 780 ultrasound images labeled as normal, benign, or malignant.

Evaluation Metrics. We employ seven principal metrics across three categories corresponding to PEU to holistically assess the performance of unlearning algorithms. (1) Utility. Retain Accuracy (R-Acc) captures the accuracy on the retain set $D _ { r } ;$ higher values reflect better retention of model utility. ForgetAccuracy (F-Acc) measures the accuracy on the forget set $D _ { f }$ . Test Accuracy (T-Acc) measures the classification accuracy on an independent test set $D _ { T }$ that is disjoint from both the training and forget sets; higher values indicate better generalization and retention of knowledge. Retention Deviation (RetDev) quantifies the cumulative deviation of the unlearned model’s accuracy from that of a model retrained from scratch on only the retain data; a lower RetDev score indicates that the unlearned model’s utility more closely matches the ideal retrained model, with zero representing a perfect match. (2) Privacy. Indiscernibility (Indisc) is a score based on membership inference attacks. Closer to 1 means better privacy. It means it is hard to tell if a sample was used in training. Membership Inference Attack (MIA) assesses unlearning quality by training logistic regression

Performance comparison of diferent methods on the MRI, ISIC and BUSI dataset in terms of utility, privacy, and eficiency. Arrows indicate whether higher (↑) or lower (↓) values are better. MIA measures the membership inference attack success rate, where values closer to 0.5 indicate stronger privacy protection. Error values denote the standard error of the mean (SEM).

$$
\mathsf { F } _ { - } \mathsf { A c c }
$$

$$
\mathsf { T } \mathrm { - } \mathsf { A c c } \uparrow
$$

$$
8 3 . 5 7 _ { \pm 0 . 0 0 }
$$

$$
8 0 . 4 5 _ { \pm 0 . 0 0 }
$$

$$
7 6 . 1 3 _ { \pm 0 . 0 0 }
$$

$$
0 _ { \pm 0 . 0 0 }
$$

$$
9 3 . 2 9 _ { \pm 2 . 5 6 }
$$

$$
5 2 . 3 5 _ { \pm 0 . 0 0 }
$$

$$
9 9 . 8 0 _ { \pm 0 . 0 0 }
$$

$$
9 5 . 6 1 _ { \pm 2 . 5 8 }
$$

$$
1 . 0 0 0 0 _ { \pm 0 . 0 0 }
$$

$$
5 0 . 6 2 _ { \pm 1 0 . 4 4 }
$$

$$
8 1 . 1 9 _ { \pm 1 1 . 8 3 }
$$

$$
9 . 2 8 5 5 _ { \pm 6 . 1 9 }
$$

$$
8 0 . 3 5 _ { \pm 1 1 . 1 2 }
$$

$$
9 7 . 4 3 _ { \pm 0 . 7 3 }
$$

$$
8 1 . 2 1 _ { \pm 1 3 . 0 9 }
$$

$$
6 . 7 9 _ { \pm 2 3 . 7 3 }
$$

$$
4 9 . 7 3 _ { \pm 1 . 3 4 }
$$

$$
8 0 . 4 5 _ { \pm 1 2 . 8 0 }
$$

$$
9 8 . 2 0 _ { \pm 0 . 9 8 }
$$

$$
7 7 . 8 5 _ { \pm 1 2 . 4 2 }
$$

$$
5 . 0 8 _ { \pm 2 9 . 3 0 }
$$

$$
7 8 . 8 3 _ { \pm 1 4 . 0 5 }
$$

$$
5 2 . 9 6 _ { \pm 0 . 6 1 }
$$

$$
7 6 . 5 4 _ { \pm 1 3 . 5 2 }
$$

$$
4 . 3 9 9 7 _ { \pm 0 . 9 3 }
$$

$$
9 7 . 1 0 _ { \pm 1 . 4 5 }
$$

$$
7 4 . 6 6 _ { \pm 1 3 . 1 7 }
$$

$$
1 2 . 4 6 _ { \pm 3 6 . 1 8 }
$$

$$
9 7 . 0 6 _ { \pm 0 . 9 3 }
$$

$$
5 2 . 5 7 _ { \pm 0 . 5 7 }
$$

$$
7 . 8 2 1 4 _ { \pm 3 . 9 9 }
$$

$$
8 8 . 8 7 _ { \pm 2 . 6 4 }
$$

$$
8 5 . 0 6 _ { \pm 2 . 4 4 }
$$

$$
8 4 . 5 0 _ { \pm 2 . 3 7 }
$$

$$
2 3 . 0 7 _ { \pm 9 . 0 1 }
$$

$$
9 6 . 4 1 _ { \pm 1 . 2 0 }
$$

$$
4 9 . 6 3 _ { \pm 0 . 7 4 }
$$

$$
5 6 . 9 7 _ { \pm 9 . 6 8 }
$$

$$
5 4 . 8 6 _ { \pm 9 . 8 0 }
$$

$$
5 2 . 8 3 _ { \pm 8 . 9 6 }
$$

$$
9 4 . 2 4 _ { \pm 3 4 . 5 1 }
$$

$$
9 6 . 9 8 _ { \pm 0 . 8 2 }
$$

$$
5 1 . 9 2 _ { \pm 1 . 0 2 }
$$

$$
7 . 9 9 3 9 _ { \pm 4 . 3 0 }
$$

$$
9 6 . 5 4 _ { \pm 1 . 0 3 }
$$

$$
9 4 . 7 3 _ { \pm 1 . 4 6 }
$$

$$
9 2 . 6 8 _ { \pm 1 . 4 0 }
$$

$$
5 5 . 0 1 _ { \pm 4 . 8 8 }
$$

$$
9 7 . 8 4 _ { \pm 0 . 2 6 }
$$

$$
5 3 . 1 0 _ { \pm 0 . 4 8 }
$$

$$
1 5 . 1 3 8 6 _ { \pm 3 . 0 0 }
$$

$$
8 6 . 8 6 _ { \pm 4 . 5 4 }
$$

$$
8 4 . 7 3 _ { \pm 4 . 3 9 }
$$

$$
8 0 . 5 8 _ { \pm 4 . 8 9 }
$$

$$
1 5 . 1 0 _ { \pm 9 . 0 0 }
$$

$$
9 8 . 9 0 _ { \pm 0 . 3 5 }
$$

$$
5 1 . 6 7 _ { \pm 1 . 0 0 }
$$

$$
4 9 . 6 6 _ { \pm 1 2 . 5 3 }
$$

$$
4 9 . 0 5 _ { \pm 1 2 . 7 1 }
$$

$$
5 0 . 4 2 _ { \pm 1 1 . 7 5 }
$$

$$
1 1 3 . 3 8 _ { \pm 4 2 . 2 7 }
$$

$$
9 6 . 6 5 _ { \pm 0 . 9 8 }
$$

$$
5 1 . 8 6 _ { \pm 1 . 3 1 }
$$

$$
1 5 . 5 0 8 2 _ { \pm 0 . 9 1 }
$$

$$
8 4 . 1 3 _ { \pm 5 . 0 4 }
$$

$$
8 1 . 6 9 _ { \pm 4 . 5 1 }
$$

$$
8 0 . 2 4 _ { \pm 4 . 9 2 }
$$

$$
7 . 6 1 _ { \pm 9 . 5 9 }
$$

$$
9 7 . 7 5 _ { \pm 0 . 8 8 }
$$

$$
4 9 . 9 0 _ { \pm 1 . 1 8 }
$$

$$
1 7 . 2 4 1 9 _ { \pm 0 . 5 1 }
$$

$$
5 9 . 6 9 _ { \pm 0 . 0 0 }
$$

$$
5 6 . 1 3 _ { \pm 0 . 0 0 }
$$

$$
3 3 . 9 0 _ { \pm 0 . 0 0 }
$$

$$
0 _ { \pm 0 . 0 0 }
$$

$$
9 8 . 9 7 _ { \pm 0 . 0 0 }
$$

$$
5 2 . 0 8 _ { \pm 0 . 0 0 }
$$

$$
1 _ { \pm 0 . 0 0 }
$$

$$
5 5 . 3 5 _ { \pm 1 . 6 6 }
$$

$$
5 6 . 4 4 _ { \pm 2 . 6 0 }
$$

$$
3 8 . 9 8 _ { \pm 0 . 8 5 }
$$

$$
2 2 . 8 1 _ { \pm 2 . 5 1 }
$$

$$
9 6 . 0 0 _ { \pm 0 . 5 2 }
$$

$$
5 3 . 0 8 _ { \pm 0 . 6 3 }
$$

$$
1 7 . 3 0 8 9 _ { \pm 7 . 8 8 }
$$

$$
5 5 . 3 3 _ { \pm 2 . 3 4 }
$$

$$
5 7 . 1 7 _ { \pm 3 . 4 5 }
$$

$$
3 8 . 8 1 _ { \pm 1 . 4 5 }
$$

$$
2 3 . 6 4 _ { \pm 2 . 1 3 }
$$

$$
9 3 . 6 4 _ { \pm 1 . 9 4 }
$$

$$
5 5 . 0 0 _ { \pm 0 . 6 6 }
$$

$$
5 . 5 0 3 0 _ { \pm 2 . 5 1 }
$$

$$
5 0 . 4 2 _ { \pm 6 . 0 9 }
$$

$$
5 0 . 5 8 _ { \pm 7 . 8 0 }
$$

$$
3 6 . 1 0 _ { \pm 4 . 5 8 }
$$

$$
3 1 . 9 1 _ { \pm 2 7 . 6 3 }
$$

$$
9 7 . 7 4 _ { \pm 0 . 9 4 }
$$

$$
5 5 . 9 2 _ { \pm 1 . 0 7 }
$$

$$
5 . 1 1 9 9 _ { \pm 1 . 9 4 }
$$

$$
4 6 . 7 9 _ { \pm 4 . 6 5 }
$$

$$
4 5 . 7 6 _ { \pm 5 . 6 6 }
$$

$$
2 8 . 8 1 _ { \pm 4 . 1 3 }
$$

$$
5 5 . 1 0 _ { \pm 2 6 . 0 0 }
$$

$$
9 1 . 0 8 _ { \pm 1 . 3 6 }
$$

$$
5 1 . 5 8 _ { \pm 3 . 3 1 }
$$

$$
6 . 3 6 7 3 _ { \pm 1 . 0 8 }
$$

$$
5 3 . 4 9 _ { \pm 2 . 3 9 }
$$

$$
5 6 . 7 5 _ { \pm 2 . 8 1 }
$$

$$
3 8 . 3 1 _ { \pm 1 . 7 5 }
$$

$$
2 4 . 5 0 _ { \pm 6 . 4 3 }
$$

$$
9 4 . 2 6 _ { \pm 0 . 7 7 }
$$

$$
5 6 . 5 8 _ { \pm 0 . 4 8 }
$$

$$
3 4 . 4 9 _ { \pm 7 . 3 8 }
$$

$$
2 0 . 2 7 0 4 _ { \pm 9 . 1 7 }
$$

$$
3 3 . 6 1 _ { \pm 7 . 8 9 }
$$

$$
2 2 . 7 1 _ { \pm 3 . 8 1 }
$$

$$
4 9 . 7 4 _ { \pm 4 2 . 9 8 }
$$

$$
5 2 . 5 0 _ { \pm 3 . 1 8 }
$$

$$
5 8 . 0 5 _ { \pm 0 . 0 8 }
$$

$$
4 . 0 6 6 7 _ { \pm 0 . 5 2 }
$$

$$
5 4 . 1 4 _ { \pm 0 . 3 6 }
$$

$$
4 0 . 3 4 _ { \pm 0 . 3 4 }
$$

$$
2 5 . 2 9 _ { \pm 0 . 4 1 }
$$

$$
9 1 . 9 0 _ { \pm 1 . 2 7 }
$$

$$
5 6 . 7 5 _ { \pm 0 . 0 8 }
$$

$$
5 1 . 9 0 _ { \pm 5 . 3 1 }
$$

$$
2 1 . 7 1 5 4 _ { \pm 2 . 1 5 }
$$

$$
4 8 . 5 6 _ { \pm 4 . 5 1 }
$$

$$
3 3 . 0 5 _ { \pm 2 . 5 7 }
$$

$$
2 9 . 0 4 _ { \pm 1 4 . 2 9 }
$$

$$
8 5 . 6 4 _ { \pm 1 . 5 7 }
$$

$$
5 3 . 8 5 _ { \pm 0 . 9 7 }
$$

$$
4 3 . 8 4 _ { \pm 8 . 3 0 }
$$

$$
5 1 . 1 0 _ { \pm 1 0 . 8 8 }
$$

$$
3 2 . 8 8 _ { \pm 4 . 8 5 }
$$

$$
3 . 1 0 2 7 _ { \pm 1 . 0 9 }
$$

$$
3 8 . 5 2 _ { \pm 3 7 . 7 5 }
$$

$$
9 1 . 2 8 _ { \pm 1 . 6 1 }
$$

$$
5 6 . 1 7 _ { \pm 0 . 7 5 }
$$

$$
1 4 . 3 1 2 4 _ { \pm 3 . 9 7 }
$$

$$
5 6 . 8 3 _ { \pm 4 . 8 9 }
$$

$$
5 3 . 9 3 _ { \pm 4 . 5 3 }
$$

$$
3 7 . 2 9 _ { \pm 1 . 9 4 }
$$

$$
1 8 . 7 1 _ { \pm 1 . 6 0 }
$$

$$
9 7 . 4 4 _ { \pm 0 . 7 7 }
$$

$$
5 2 . 9 2 _ { \pm 1 . 2 7 }
$$

$$
1 5 . 0 2 0 9 _ { \pm 1 . 0 8 }
$$

$$
8 6 . 1 1 _ { \pm 0 . 0 0 }
$$

$$
8 3 . 0 1 _ { \pm 0 . 0 0 }
$$

$$
8 2 . 8 6 _ { \pm 0 . 0 0 }
$$

$$
0 _ { \pm 0 . 0 0 }
$$

$$
8 7 . 2 7 _ { \pm 0 . 0 0 }
$$

$$
7 7 . 0 6 _ { \pm 7 . 4 5 }
$$

$$
5 5 . 4 5 _ { \pm 0 . 0 0 }
$$

$$
7 4 . 8 1 _ { \pm 7 . 3 8 }
$$

$$
1 _ { \pm 0 . 0 0 }
$$

$$
7 4 . 9 2 _ { \pm 6 . 5 1 }
$$

$$
2 9 . 9 7 _ { \pm 2 4 . 6 4 }
$$

$$
9 2 . 5 5 _ { \pm 2 . 0 2 }
$$

$$
7 5 . 8 3 _ { \pm 5 . 4 6 }
$$

$$
7 0 . 3 7 _ { \pm 6 . 9 7 }
$$

$$
4 7 . 2 7 _ { \pm 1 . 3 3 }
$$

$$
7 . 5 2 8 4 _ { \pm 4 . 7 8 }
$$

$$
7 5 . 1 7 _ { \pm 4 . 1 4 }
$$

$$
3 6 . 4 5 _ { \pm 1 9 . 0 7 }
$$

$$
8 8 . 7 3 _ { \pm 1 . 9 6 }
$$

$$
4 8 . 8 2 _ { \pm 2 . 9 9 }
$$

$$
4 . 0 9 1 2 _ { \pm 2 . 9 5 }
$$

$$
7 1 . 5 2 _ { \pm 1 1 . 5 2 }
$$

$$
7 0 . 0 0 _ { \pm 9 . 3 0 }
$$

$$
6 8 . 3 2 _ { \pm 1 1 . 6 0 }
$$

$$
5 0 . 1 6 _ { \pm 3 8 . 1 6 }
$$

$$
8 7 . 4 5 _ { \pm 2 . 2 2 }
$$

$$
5 . 1 1 3 7 _ { \pm 1 . 3 8 }
$$

$$
4 4 . 7 3 _ { \pm 0 . 3 4 }
$$

$$
6 5 . 1 9 _ { \pm 5 . 8 9 }
$$

$$
7 2 . 1 9 _ { \pm 3 . 6 5 }
$$

$$
5 1 . 8 3 _ { \pm 1 6 . 9 1 }
$$

$$
9 0 . 7 3 _ { \pm 1 . 6 9 }
$$

$$
4 9 . 2 7 _ { \pm 1 . 1 8 }
$$

$$
3 . 6 5 6 4 _ { \pm 1 . 4 5 }
$$

$$
7 5 . 3 2 _ { \pm 4 . 0 4 }
$$

$$
7 4 . 4 4 _ { \pm 4 . 0 2 }
$$

$$
7 6 . 1 3 _ { \pm 3 . 4 1 }
$$

$$
3 0 . 9 8 _ { \pm 1 3 . 0 0 }
$$

$$
8 7 . 4 5 _ { \pm 2 . 1 4 }
$$

$$
5 2 . 5 5 _ { \pm 1 . 9 4 }
$$

$$
6 . 4 2 5 2 _ { \pm 4 . 4 8 }
$$

$$
4 4 . 5 8 _ { \pm 5 . 2 4 }
$$

$$
4 2 . 7 8 _ { \pm 3 . 3 3 }
$$

$$
4 7 . 1 1 _ { \pm 6 . 3 1 }
$$

$$
1 3 9 . 8 4 _ { \pm 1 7 . 6 1 }
$$

$$
9 3 . 8 2 _ { \pm 2 . 3 6 }
$$

$$
5 1 . 0 0 _ { \pm 0 . 9 0 }
$$

$$
3 . 0 2 8 0 _ { \pm 1 . 4 9 }
$$

$$
8 3 . 2 7 _ { \pm 0 . 2 3 }
$$

$$
8 1 . 7 1 _ { \pm 0 . 2 6 }
$$

$$
8 2 . 0 4 _ { \pm 0 . 3 7 }
$$

$$
5 . 8 5 _ { \pm 0 . 3 6 }
$$

$$
8 9 . 8 2 _ { \pm 0 . 7 3 }
$$

$$
4 5 . 0 0 _ { \pm 0 . 4 5 }
$$

$$
7 . 8 2 7 4 _ { \pm 0 . 8 1 }
$$

$$
6 5 . 5 5 _ { \pm 7 . 8 4 }
$$

$$
6 4 . 6 3 _ { \pm 6 . 3 2 }
$$

$$
6 6 . 1 0 _ { \pm 7 . 6 6 }
$$

$$
6 6 . 2 5 _ { \pm 2 5 . 3 9 }
$$

$$
\mathsf { F o r g e t - M I }
$$

$$
4 6 . 8 6 _ { \pm 4 2 . 2 3 }
$$

$$
4 9 . 1 8 _ { \pm 1 . 4 4 }
$$

$$
1 . 3 2 0 1 _ { \pm 0 . 2 8 }
$$

$$
4 8 . 3 5 _ { \pm 1 1 . 4 5 }
$$

$$
4 7 . 7 8 _ { \pm 9 . 7 5 }
$$

$$
4 6 . 5 4 _ { \pm 1 2 . 0 1 }
$$

$$
1 3 0 . 1 2 _ { \pm 3 9 . 3 6 }
$$

$$
9 0 . 0 0 _ { \pm 1 . 6 0 }
$$

$$
5 0 . 0 0 _ { \pm 1 . 3 9 }
$$

$$
1 0 . 9 7 3 8 _ { \pm 0 . 6 0 }
$$

$$
5 8 . 3 4 _ { \pm 2 . 0 7 }
$$

$$
5 1 . 4 8 _ { \pm 2 . 4 1 }
$$

$$
6 2 . 9 8 _ { \pm 1 . 4 0 }
$$

$$
9 4 . 2 3 _ { \pm 6 . 9 8 }
$$

$$
9 1 . 8 2 _ { \pm 2 . 1 1 }
$$

$$
5 0 . 4 5 _ { \pm 0 . 3 8 }
$$

$$
7 . 0 9 8 5 _ { \pm 0 . 3 5 }
$$

via cross-validation on losses from equal-sized $D _ { f }$ and $\mathcal { D } _ { V }$ where accuracy near 1.0 indicates perfect distinguishability and 0.5 means random guessing. (3) Eficiency. Runtime eficiency (RTE) measures the speed improvement of an unlearning method compared to retraining, and is defined as the ratio of the time required for retraining to the time required for unlearning. The RTE for retraining is 1. An RTE greater than 1 indicates that it is faster than retraining, while an RTE less than 1 indicates that it is slower.

the model on the retained data; (b) Classical Unlearning Methods. This group includes fundamental and widelyused strategies mentioned in the review [30]: Fine-Tuning (FT), Random Relabeling (RL), Saliency Unlearning (SaLUN)[16]; (c) State-of-the-Art General Algorithms. We select the three top-performing algorithms identified in a recent comprehensive benchmark study [7] as representatives of the cutting-edge in general machine unlearning: Masked-Small-Gradients (MSG), Convolution-Transpose (CT), Knowledge-Distillation-Entropy (KDE); $( d _ { 1 } )$ Existing Medical Unlearning Methods. This category includes method proposed within the medical imaging domain: Bilevel-Optimization (BiO) [29]; $( d _ { 2 } )$ Furthermore, to acknowledge advancements in specialized scenarios, we also consider methods from federated or multi-modal medical

Baselines & Model. We conduct comparisons against several representative unlearning baselines. The baselines are categorized into three groups: classical approaches, top-performing recent general-purpose algorithms, and existing methods specifically designed for or applied in medical imaging. Including: (a) Retrain, which retrains contexts (FCU [13] and Forget-MI [20]), and simply modify them to fit the experimental setup described in this paper. All baseline comparisons are included in the main text results (Section 5.2). We evaluate these methods on ResNet18.

Data splits. To simulate clinical privacy removal scenarios, we perform data splitting with a fixed random seed of $s = 1 2 3$ . For each dataset, if a standard train/test split already exists, we use the original training set as the development set. Otherwise, we split the full data into a development set and a test set at an 80%/20% ratio. The development set is further divided into a training set $D _ { \mathrm { t r a i n } }$ and a validation set $\mathcal { D } _ { V }$ with an 80%/20% split. From the training set, we randomly select 10% as the forget set $D _ { f }$ (corresponding to private patient data). The remaining 90% forms the retain set $\textstyle { \mathcal { D } } _ { r }$ . This forgetting ratio follows the common setting for medical unlearning tasks. Most training is performed using cross-entropy loss, SGD optimizer (momentum = 0.9), a learning rate of $1 { \cdot } 1 0 ^ { - 3 }$ , and $L _ { 2 }$ regularization with coeficient $1 { \cdot } 1 0 ^ { - 4 }$ . All experiments are run on 1 NVIDIA GeForce RTX 4090 GPU.

## 5.2. Main Results

To address RQ1, we comprehensively compare GRIN+ with representative unlearning baselines on three medical datasets: MRI, ISIC, and BUSI (Table 2). These baselines include classical methods, state-of-the-art general-purpose algorithms, and medical-specific methods. The key findings and analysis are summarized below from three perspectives: Utility, Privacy and Eficiency.

Utility Perspective. Retrain achieves the best utility but has very high computational cost, making it impractical for clinical use. Classical Fine-Tuning (FT) achieves 95.61% R-Acc on MRI yet lacks targeted forgetting mechanism, while RL and SaLUN impair model generalization and reduce T-Acc due to unconstrained parameter updates. State-ofthe-art general-purpose algorithms rely on class-balanced assumptions, leading to severe utility collapse on long-tailed medical data; for example, KDE obtains only 44.58% R-Acc on BUSI due to majority-class gradient dominance. Medical-specific methods also underperform: Forget-MI causes excessive utility degradation, and FCU is ill-suited for centralized medical image classification. In contrast, GRIN+ maintains stable R-Acc and T-Acc with low RetDev across all datasets, reaching 84.13% R-Acc and 80.24% T-Acc on MRI to efectively preserve clinical utility.

Privacy Perspective. Most baselines perform poorly in privacy. Classical methods fail to eliminate parameter residual traces, resulting in high MIA scores and weak privacy. For example, on ISIC, RL has a high MIA of 55.00%, indicating strong distinguishability. State-of-the-art generalpurpose algorithms lack directional forgetting constraints, leaving exploitable model signatures for membership inference attacks. Among them, CT has a high MIA value of 56.58 on ISIC. Most medical-specific methods obtain low Indisc values and cannot meet clinical privacy rules. On BUSI, FCU has a low Indisc of only 46.86%, showing very weak privacy protection. GRIN+ achieves MIA scores close to random guess on all datasets (49.90% on MRI, 52.92% on ISIC, 50.45% on BUSI) with high Indisc, fully erasing target data memory traces and complying with medical AI privacy regulations.

Table 3  
Quantitative results for the ablation study.
<table><tr><td>GR</td><td>CW</td><td>DCF</td><td>R-Acc ↑</td><td>T-Acc ↑</td><td>MIA</td><td>RTE ↑</td></tr><tr><td>√</td><td></td><td></td><td>65.21</td><td>68.42</td><td>49.49</td><td>23.6583</td></tr><tr><td>√</td><td>√</td><td></td><td>81.94</td><td>76.28</td><td>48.67</td><td>18.7780</td></tr><tr><td>√</td><td>√</td><td>√</td><td>84.13</td><td>80.24</td><td>49.90</td><td>17.2419</td></tr></table>

Eficiency Perspective. To answer RQ2, we compared the computational eficiency of diferent methods, measured by RTE (higher values indicate faster speed). Classical methods rely on full-model tuning or complex computations, yielding low RTE and excessive latency (the RTE of RL is 4.10 on MRI). State-of-the-art general-purpose algorithms involve redundant full-parameter iteration, providing only moderate eficiency. Some medical-specific methods (Forget-MI, BiO) achieve relatively high RTE but still depend on full-parameter updates. GRIN+ only updates sparse critical parameters, avoiding full-model optimization; it reaches the highest RTE of 17.24 on MRI, outperforming all baselines and satisfying clinical low-latency demands.

PEU Trilemma Balance Analysis. All baselines fail to balance the “privacy-eficiency-utility” (PEU) trilemma for imbalanced medical data. Classical methods are comparable to the GRIN+ method in terms of utility and privacy, but because they still rely on global fine-tuning, leading to extremely high computational cost and low eficiency. Stateof-the-art general-purpose algorithms are incompatible with long-tailed data, as class imbalance biases parameter estimation and degrades all three metrics. Medical-specific methods sufer from poor task adaptability, with incomplete forgetting, utility loss or low eficiency. The core issue is that existing methods cannot address majority-class gradient dominance, high eficiency and knowledge retention simultaneously. GRIN+ resolves these via class-adaptive scoring, direction-constrained perturbation and sparse selection, achieving optimal PEU balance for clinical medical AI.

Ablation Study. To answer RQ3, we conduct an ablation study on three key modules of GRIN+: Gradient Ratio (GR), Class-balanced Weight (CW), and Direction Constraint Factor (DCF). The results are shown in Table 3 (analysis using the MRI dataset as an example). GR only. The model selects parameters based solely on the gradient ratio. The R-Acc is only 65.21%, and the T-Acc is 68.42%. The performance on retained knowledge is significantly degraded. However, due to its simple update process, it achieves the highest RTE (23.66). $\mathbf { G R } + \mathbf { C W } .$ After introducing the class-balanced weight (CW), the R-Acc increases significantly to 81.94% (+16.73pp). This shows that CW efectively corrects the parameter selection bias caused by the dominance of majority class gradients. Meanwhile, MIA drops slightly to 48.67%, and the RTE is 18.78. GR + CW + DCF (GRIN+). After adding the direction constraint factor (DCF), the R-Acc further improves to 84.13%. The T-Acc reaches 80.24%, MIA rises back to 49.90% (closest to random guess), and the RTE is 17.24. DCF protects the model’s generalization ability by suppressing updates that conflict with the direction of retained gradients.

In summary, the three modules work together. GR locates forgetting-related parameters, CW alleviates class imbalance, and DCF stabilizes the update direction. Together, they achieve the optimal balance among “privacy-eficiencyutility” (PEU) trilemma.

## 6. Conclusion

We present GRIN+, a gradient-guided selective forgetting framework for medical image classification. By comparing gradient responses between forgotten and retained sets, GRIN+ identifies and perturbs only a small subset of key parameters. The framework incorporates class weights to handle data imbalance and directional constraints to preserve model generalization, ensuring eficient and controllable unlearning. As a lightweight solution, GRIN+ meets strict privacy requirements while ensuring the clinical utility of trustworthy medical AI.

## References

[1] Adlung, L., Cohen, Y., Mor, U., Elinav, E., 2021. Machine learning in clinical decision making. Med 2, 642–665.

[2] Al-Dhabyani, W., Gomaa, M., Khaled, H., Fahmy, A., 2020. Dataset of breast ultrasound images. Data in brief 28, 104863. [dataset].

[3] Atchinson, B.K., Fox, D.M., 1997. From the field: the politics of the health insurance portability and accountability act. Health afairs 16, 146–150.

[4] Balle, B., Cherubin, G., Hayes, J., 2022. Reconstructing training data with informed adversaries, in: 2022 IEEE Symposium on Security and Privacy (SP), IEEE. pp. 1138–1156.

[5] Basu, S., Pope, P., Feizi, S., 2021. Influence functions in deep learning are fragile, in: International Conference on Learning Representations (ICLR).

[6] Bourtoule, L., Chandrasekaran, V., Choquette-Choo, C.A., Jia, H., Travers, A., Zhang, B., Lie, D., Papernot, N., 2021. Machine unlearning, in: 2021 IEEE symposium on security and privacy (SP), IEEE. pp. 141–159.

[7] Cadet, X.F., Borovykh, A., Malekzadeh, M., Ahmadi-Abhari, S., Haddadi, H., 2025. Deep unlearn: Benchmarking machine unlearning for image classification, in: 2025 IEEE 10th European Symposium on Security and Privacy (EuroS&P), IEEE. pp. 939–962.

[8] Cao, Z., Duan, L., Yang, G., Yue, T., Chen, Q., 2019. An experimental study on breast lesion detection and classification from ultrasound images using deep learning architectures. BMC medical imaging 19, 51.

[9] Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., Tramer, F., 2022. Membership inference attacks from first principles, in: 2022 IEEE symposium on security and privacy (SP), IEEE. pp. 1897–1914.

[10] Carlini, N., Liu, C., Erlingsson, Ú., Kos, J., Song, D., 2019. The secret sharer: Evaluating and testing unintended memorization in neural networks, in: 28th USENIX security symposium (USENIX security 19), pp. 267–284.

[11] Codella, N., Rotemberg, V., Tschandl, P., Celebi, M.E., Dusza, S., Gutman, D., Helba, B., Kalloo, A., Liopyris, K., Marchetti, M., et al., 2019. Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (isic). arXiv preprint arXiv:1902.03368 [dataset].

[12] Datta, N., Alam, M.G.R., 2025. Erase to retain: Low rank adaptation guided selective unlearning in medical segmentation networks. arXiv preprint arXiv:2511.16574 .

[13] Deng, Z., Luo, L., Chen, H., 2024. Enable the right to be forgotten with federated client unlearning in medical imaging, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 240–250.

[14] Dildar, M., Akram, S., Irfan, M., Khan, H.U., Ramzan, M., Mahmood, A.R., Alsaiari, S.A., Saeed, A.H.M., Alraddadi, M.O., Mahnashi, M.H., 2021. Skin cancer detection: a review using deep learning techniques. International journal of environmental research and public health 18, 5479.

[15] Esteva, A., Robicquet, A., Ramsundar, B., Kuleshov, V., DePristo, M., Chou, K., Cui, C., Corrado, G., Thrun, S., Dean, J., 2019. A guide to deep learning in healthcare. Nature medicine 25, 24–29.

[16] Fan, C., Liu, J., Zhang, Y., Wong, E., Wei, D., Liu, S., 2023. Salun: Empowering machine unlearning via gradient-based weight saliency in both image classification and generation. arXiv preprint arXiv:2310.12508 .

[17] Goel, S., Prabhu, A., Sanyal, A., Lim, S.N., Torr, P., Kumaraguru, P., 2022. Towards adversarial evaluations for inexact machine unlearning. arXiv preprint arXiv:2201.06640 .

[18] Golatkar, A., Achille, A., Soatto, S., 2020. Eternal sunshine of the spotless net: Selective forgetting in deep networks, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9304–9312.

[19] Guo, C., Goldstein, T., Hannun, A., van der Maaten, L., 2023. Certified data removal from machine learning models. URL: https: //arxiv.org/abs/1911.03030, arXiv:1911.03030.

[20] Hardan, S., Taratynova, D., Essofi, A., Nandakumar, K., Yaqub, M., 2025. Forget-mi: Machine unlearning for forgetting multimodal information in healthcare settings, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 204–213.

[21] Izzo, Z., Smart, M.A., Chaudhuri, K., Zou, J., 2021. Approximate data deletion from machine learning models. URL: https://arxiv. org/abs/2002.10077, arXiv:2002.10077.

[22] Jyothi, P., Singh, A.R., 2023. Deep learning models and traditional automated techniques for brain tumor segmentation in mri: a review. Artificial intelligence review 56, 2923–2969.

[23] Koh, P.W., Liang, P., 2017. Understanding black-box predictions via influence functions, in: International conference on machine learning, PMLR. pp. 1885–1894.

[24] Kumar, A., Masud, M., Alsharif, M.H., Gaur, N., Nanthaamornphong, A., 2025. Integrating 6g technology in smart hospitals: challenges and opportunities for enhanced healthcare services. Frontiers in medicine 12, 1534551.

[25] Kurmanji, M., Triantafillou, P., Hayes, J., Triantafillou, E., 2023. Towards unbounded machine unlearning. Advances in neural information processing systems 36, 1957–1987.

[26] Lin, M., Holste, G., Wang, S., Zhou, Y., Wei, Y., Banerjee, I., Chen, P., Dai, T., Du, Y., Dvornek, N.C., et al., 2025. Cxr-lt 2024: A miccai challenge on long-tailed, multi-label, and zero-shot disease classification from chest x-ray. Medical Image Analysis , 103739.

[27] Mantelero, A., 2013. The eu proposal for a general data protection regulation and the roots of the ‘right to be forgotten’. Computer Law & Security Review 29, 229–235.

[28] Martens, J., 2020. New insights and perspectives on the natural gradient method. Journal of Machine Learning Research 21, 1–76.

[29] Nahass, G.R., Wang, Z., Rashidisabet, H., Kim, W.H., Hubschman, S., Peterson, J.C., Purnell, C.A., Setabutr, P., Tran, A.Q., Yi, D., et al., 2025. Targeted unlearning using perturbed sign gradient methods with applications on medical images. arXiv preprint arXiv:2505.21872 .

[30] Nasirigerdeh, R., Razmi, N., Schnabel, J.A., Rueckert, D., Kaissis, G., 2024. Machine unlearning for medical imaging. arXiv preprint arXiv:2407.07539 .

[31] Nickparvar, M., 2021. Brain tumor mri dataset. URL: https://www. kaggle.com/dsv/2645886, doi:10.34740/KAGGLE/DSV/2645886. [dataset].

[32] Pan, L., Zhang, Y., Yang, Q., Li, T., Chen, Z., 2025. Long-tailed medical diagnosis with relation-aware representation learning and iterative classifier calibration. Computers in Biology and Medicine 188, 109772.

[33] Shi, J., Gourgoulias, K., Buford, J.F., Moran, S.J., Ghalyan, N., 2024. Deepclean: Machine unlearning on the cheap by resetting privacy sensitive weights using the fisher diagonal, in: European Conference on Computer Vision, Springer. pp. 1–16.

[34] Thudi, A., Jia, H., Shumailov, I., Papernot, N., 2022. On the necessity of auditable algorithmic definitions for machine unlearning, in: 31st USENIX security symposium (USENIX Security 22), pp. 4007– 4022.

[35] Wang, J., Guo, S., Xie, X., Qi, H., 2022. Federated unlearning via class-discriminative pruning, in: Proceedings of the ACM web conference 2022, pp. 622–632.

[36] Wang, W., Lee, J., Harrou, F., Sun, Y., 2020. Early detection of parkinson’s disease using deep learning and machine learning. IEEE access 8, 147635–147646.

[37] Warnecke, A., Pirch, L., Wressnegger, C., Rieck, K., 2023. Machine unlearning of features and labels, in: Proceedings 2023 Network and Distributed System Security Symposium, Internet Society.

[38] Xue, Y., Liu, J., McDonagh, S., Tsaftaris, S.A., 2024. Erase to enhance: Data-eficient machine unlearning in mri reconstruction, in: Proceedings of The 7nd International Conference on Medical Imaging with Deep Learning, PMLR. pp. 1785–1800.

[39] Yu, L., Zhao, Z., Wang, Y., Wang, P., Cao, X., Wang, B., Wang, Y., 2026. Falw: A forgetting-aware loss reweighting for long-tailed unlearning. arXiv preprint arXiv:2601.18650 .