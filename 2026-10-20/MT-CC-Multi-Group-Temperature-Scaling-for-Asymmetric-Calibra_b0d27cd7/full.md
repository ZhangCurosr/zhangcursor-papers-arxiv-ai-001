# MT-CC: Multi-Group Temperature Scaling for Asymmetric Calibration Behavior in Class-Incremental Learning

Sungjun YunYonghee ChoiDong-Jun Han

Yonsei University

{sungjun.yun, 2025321337, djh}@yonsei.ac.kr

## Abstract

As deep neural networks are required to continuously adapt to evolving data, class-incremental learning (CIL) has become an active research area. However, most existing works primarily focus on improving accuracy while overlooking confidence calibration, which measures the trustworthiness of the model's predictions. Although temperature scaling (TS) is effective for static calibration, it has rarely been explored in continual settings where data distributions evolve over time. In this paper, we introduce Multi-Group Temperature Scaling for Continual Calibration (MT-CC), a novel post-hoc calibration framework that addresses the asymmetric calibration behavior in CIL, where the rate of confidence changes does not align with the rate of accuracy degradation across tasks, leading to different levels of miscalibration. Our key contribution is to design multiple temperature parameters through a lightweight grouping network that separates samples by confidence statistics and task-aware signals in CIL. Furthermore, a Groupwise Difference-between-Confidence-and-Accuracy (GDCA) regularization is incorporated to promote intra-group consistency. Experiments demonstrate that MT-CC significantly improves calibration performance while maintaining accuracy, offering a principled framework for reliable continual learning.

## 1 Introduction

In modern deep learning applications, it has become increasingly important for a model to provide reliable indications of whether its predictions are likely to be correct. In particular, in a well-calibrated model, the predicted confidence should align with the empirical likelihood that the prediction is correct. This allows users to trust the model's outputs when it exhibits high confidence, while rejecting predictions made with low confidence. However, in practice, neural networks are often miscalibrated, producing confidence scores that do not align with their actual accuracy, frequently being overconfident in wrong predictions.

In this context, confidence calibration has attracted increasing attention, aiming to align the model's predicted confidence with the true likelihood of correctness. Various methods have been proposed including temperature scaling (TS) [12], which performs a post-hoc adjustment of logits to improve calibration without altering the model's predictions. In addition to such post-hoc approaches [7, 12, 16, 22, 37, 45–47, 50], train-time calibration methods have also been explored [8, 14, 23, 27, 28, 36, 41, 49], where the model is explicitly trained to produce well-calibrated confidence scores. Compared to train-time schemes, post-hoc methods such as TS offer a clear advantage, as they can be directly applied to any pre-trained model without sacrificing accuracy.

![](images/efcf63a888b767dc8d01d5073a812f68639bd42efad117424b6aa32910b8334c.jpg)

![](images/d1b959970bf7e11c78ad8b8dcdf60cfd430182a19e778828ee0f55f8386faba1.jpg)  
Figure 1: The reliability diagram visualizes calibration by comparing empirical accuracy and predicted confidence within each confidence bin, while a lower ECE indicates better calibration. (a) In class-incremental learning, accuracy and confidence evolve at different rates across tasks, producing asymmetric calibration behaviors: the gap between confidence and accuracy varies across tasks due to catastrophic forgetting. (b) As a result, a single global temperature cannot simultaneously calibrate these heterogeneous groups. (c) Our MT-CC routes each sample to a group-specific temperature to capture asymmetric calibration behaviors in class-incremental learning.

Fundamental challenges. However, most prior works on calibration have focused on static scenarios, where the target task and data distribution remain fixed. In many practical settings, however, models are required to learn continuously from a stream of tasks where new classes are introduced over time without revisiting all previous data. In such dynamic environments, maintaining well-calibrated confidence becomes particularly challenging, as the distribution shifts induced by new tasks often lead to varying levels of miscalibration across tasks.

Concretely, as illustrated in Figure 1a, models in class-incremental learning (CIL) [9, 35, 43] exhibit asymmetric calibration behaviors, where accuracy and confidence evolve at different rates across tasks under continual updates (see Figure 2). As a result, a single temperature value T fails to calibrate all classes effectively, since different task groups exhibit distinct confidence characteristics and require differentiated scaling.

Although T-CIL [16] recently proposed a TS strategy tailored to CIL, it still relies on a single temperature value and thus cannot effectively address the asymmetric calibration behaviors of CIL, as illustrated in Figure 1b. Joy et al. [17], Tomani et al. [42] propose sample-dependent temperature scaling, but they are developed under static, fixed-distribution settings and do not explicitly capture the CIL-specific calibration heterogeneity induced by continual task shifts. To the best of our knowledge, these aspects have been largely overlooked in existing calibration research and remain unresolved. Our work seeks to bridge that gap by addressing the following question:

How can we ensure reliable model calibration in continual learning, where asymmetric miscalibration behaviors across tasks pose persistent challenges?

![](images/e8554ab4cd603b0666832145cba0d5a7a5d502632bdb3bd08f5a1b8fa1f3bbea.jpg)  
Figure 2: Confidence-accuracy gap of selected tasks over incremental stages of CIFAR-100. The results reveal asymmetric calibration behaviors in continual learning, where the discrepancy between confidence and accuracy varies across tasks due to catastrophic forgetting. Results for all tasks are provided in Appendix D.

Contributions. In this paper, we propose multi-group temperature scaling for continual calibration (MT-CC), a novel calibration strategy that introduces multiple temperature parameters determined by a grouping network, effectively tackling the asymmetric calibration behavior in continual learning. To this end, we train the grouping network using CIL-specific confidence statistics, including the probability mass assigned to old and new classes and the logit margin, which together capture heterogeneous calibration needs. This enables the grouping network to assign an appropriate temperature to each group, thereby achieving calibration both within each group and across the entire dataset, as shown in Figure 1c. In addition, we propose a Groupwise Difference-between-Confidence-and-Accuracy (GDCA) loss to balance confidence and accuracy within each group, thereby further improving calibration.

Our main contributions are summarized as follows:

• We analyze asymmetric calibration behaviors in class-incremental learning, where different tasks exhibit varying degrees of miscalibration, and exploit these behaviors to design a CIL-aware calibration framework.

• We propose multi-group temperature scaling for continual calibration (MT-CC), a novel calibration framework that introduces multiple temperature parameters determined by a grouping network, enabling input-dependent calibration. The grouping network is conditioned on old/new confidence statistics together with a logit margin signal, allowing it to capture asymmetric calibration behaviors and assign an appropriate temperature to each group.

• We conduct extensive experiments using various benchmark datasets, demonstrating the superiority of MT-CC in class-incremental settings.

## 2 Related Work

Confidence calibration. Compared to train-time calibration methods [8, 14, 23, 27, 28, 36, 41, 49] that incorporate calibration objectives during the training process, post-hoc methods [7, 12, 16, 22, 37, 45–47, 50] calibrate the confidence by adjusting the output logits of an already trained model. Post-hoc methods offer a clear advantage in that they preserve accuracy and can be directly applied to pre-trained models. A representative approach is temperature scaling (TS), which rescales logits with a single scalar temperature [12]. Subsequent extensions, including ETS, IRM [50], PTS [42], AdaTS [17], and grouping calibration [45], relax the single-temperature assumption of TS. However, they are primarily designed for static settings and thus struggle to handle continual task shifts in CIL. As shown in Section 5, directly applying these methods to CIL yields limited performance because they overlook the asymmetric calibration behavior induced by continual updates and catastrophic forgetting. MT-CC instead learns a grouping function from CIL-aware signals, including old/new confidence and logit margin: while the former directly captures the old/new class structure unique to CIL, the latter becomes CIL-relevant because catastrophic forgetting alters its relationship with miscalibration relative to static settings.

Class-incremental learning. The central challenge in CIL is catastrophic forgetting, where learning new classes degrades performance on previously learned ones [9, 31, 35, 43]. To address this, prior work has proposed regularization, parameter-isolation, and replay-based methods [1, 2, 4– 6, 18, 20, 26, 29, 30, 38–40, 44, 48, 51], all aiming to preserve past knowledge and maintain accuracy across learned classes. However, most CIL studies focus primarily on classification accuracy, while largely overlooking confidence calibration.

Confidence calibration in continual learning. Li et al. [25] recently studied confidence calibration in continual learning. T-CIL [16], a recent state-of-the-art calibration method tailored to CIL, estimates a temperature after each task using augmented replay data from validation samples. However, its singletemperature formulation struggles to capture the asymmetric calibration behaviors that emerge across tasks under continual updates. MT-CC addresses this limitation by assigning multiple temperatures through a grouping network for task-aware calibration.

## 3 Background and Problem Setup

## 3.1 Background: Confidence Calibration and Temperature Scaling

Confidence calibration. In a K-way classification problem with label space $\mathcal { V } = \{ 1 , \ldots , K \}$ , a model parameterized by θ outputs a probability vector $s = \mathrm { s o f t m a x } ( f _ { \theta } ( x ) \hat { ) } \in [ 0 , 1 ] ^ { K }$ for an input x, where $f _ { \theta } ( x ) \in \mathbb { R } ^ { K }$ denotes the logit vector. The predicted label is $\hat { y } = \arg \operatorname* { m a x } _ { y \in \mathcal { Y } } s [ y ]$ , where $s [ y ]$ denotes the predicted probability for class $y \in \mathcal { V }$ , and its associated confidence is $\hat { s } = \operatorname* { m a x } _ { y \in \mathcal { Y } } s [ y ]$ A model is perfectly calibrated if $\mathbb { P } ( y = \hat { y } \mid \hat { s } = p ) = p$ for all $p \in [ 0 , 1 ]$ , meaning that predictions with confidence $p$ are correct with probability $p .$

Temperature scaling. Temperature scaling (TS) [12] is a post-hoc method that achieves calibration by rescaling the model's output logit z using a single scalar temperature $T > 0$ Given a sample x and its corresponding logit $z = \bar { \mathcal { f } } _ { \boldsymbol { \theta } } ( x ) \bar { \in } \mathbb { R } ^ { K }$ , the calibrated probability vector for sample x is computed as $\begin{array} { r } { s ( z , T ) = \operatorname { s o f t m a x } \left( \frac { z } { T } \right) } \end{array}$ . The temperature $T$ is learned on a validation set $\mathcal { D } ^ { \mathrm { v a l } } \bar { = } \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { \bar { N } _ { \mathrm { v a l } } }$ by minimizing the negative log-likelihood (NLL) loss.

## 3.2 Problem Setup: Calibration in Class-Incremental Learning

Class-incremental learning setup. We consider the standard class-incremental learning (CIL) setting, consisting of a sequence of $B$ tasks $\{ \mathcal { D } ^ { 1 } , \mathcal { D } ^ { 2 } , \ldots , \mathcal { D } ^ { B } \}$ , where each task $\mathcal { D } ^ { b } = \{ ( x _ { i } ^ { b } , y _ { i } ^ { \top } ) \} _ { i = 1 } ^ { n _ { b } }$ contains $n _ { b }$ training samples. Here, $x _ { i } ^ { b }$ denotes an input sample and $y _ { i } ^ { b } \in \mathcal { V } _ { b }$ represents its corresponding label. Each task has a distinct label space, i.e $, \mathcal { V } _ { b } \cap \mathcal { V } _ { b ^ { \prime } } = \emptyset$ for $b \neq b ^ { \prime }$ . After completing task $b ,$ the model is evaluated on all classes encountered so far, denoted as $\mathcal { V } _ { 1 : b } = \mathcal { V } _ { 1 } \cup \bar { \mathcal { V } _ { 2 } } \cup \cdots \cup \mathcal { V } _ { b }$ . Following Hwang et al. [16], we maintain a small memory buffer $\mathcal { M } _ { b - 1 }$ storing representative samples from previous tasks to mitigate catastrophic forgetting. During training on task $b ,$ the model learns jointly from the current task data $\mathcal { D } ^ { b }$ and the memory buffer $\mathcal { M } _ { b - 1 }$ . After training, the buffer is updated to $\mathcal { M } _ { b }$ by retaining a subset of samples from $\mathcal { D } ^ { \boldsymbol { b } } \cup \mathcal { M } _ { \boldsymbol { b } - 1 }$

Goal and problem formulation. Unlike prior works that primarily focus on improving accuracy, we aim to address confidence calibration in CIL explicitly. To this end, we integrate TS into the CIL framework by optimizing the temperature after each task b and applying it to calibrate model confidence. Specifically, each task b consists of two steps:

• Training step. The model θ learns on $\mathcal { D } ^ { b } \cup \mathcal { M } _ { b - 1 }$ to acquire new knowledge while preserving performance on previous tasks.

• Calibration step. After training is completed, the model is recalibrated using validation data $\mathcal { D } _ { b } ^ { \mathrm { v a l } }$ adjusting its confidence mapping to reflect the new data distribution.

Our goal is to build a well-calibrated model that remains reliable as tasks evolve, thereby reducing miscalibration arising from distributional and label-space shifts.

## 3.3 Aren't the Existing Solutions Good Enough?

In CIL, we identify a key challenge that prior works in static settings have overlooked: the confidenceaccuracy gap generally widens under continual updates, with different rates and trajectories across tasks, resulting in different levels of miscalibration (Figure 2). Consequently, conventional TS methods that employ a single global temperature are unable to simultaneously achieve good calibration across heterogeneous confidence distributions. As illustrated in Figure 1b, the single-temperature approach in T-CIL [16], the current state-of-the-art calibration method for CIL, fails to maintain consistent calibration, highlighting its limitation in handling asymmetric miscalibration behaviors. Moreover, while several adaptive per-sample temperature scaling methods have been proposed [17, 42], they typically assume a static data distribution and do not explicitly account for continual task evolution. We later show that these prior works exhibit significant miscalibration in CIL settings. These observations underscore the need for a more flexible, group-aware calibration strategy tailored to CIL.

## 4 Proposed MT-CC Algorithm

We propose MT-CC (Multi-Group Temperature Scaling for Continual Calibration), a new calibration framework designed for class-incremental learning.

Overview of approach. At its core, MT-CC introduces a lightweight grouping module (Sections 4.1 and 4.2) that learns to partition samples into multiple groups based on confidence- and margin-based indicators, reflecting heterogeneous miscalibration behaviors in CIL. Each group is then equipped with its own temperature parameter, allowing calibration to adapt locally rather than relying on a single global scaling factor. In addition, a Groupwise Difference-between-Confidence-and-Accuracy (GDCA) loss (Section 4.3) is employed to align predicted confidence with correctness within each group, further ensuring intra-group reliability and enhancing overall calibration.

![](images/d66cc775113b12a27baef620b827224a5a3bb67024d2aa5c7dc39c2e6bac0d9e.jpg)  
(a) Confidence-accuracy gap across logitmargin quantile bins on CIFAR-100.

![](images/708e9626718b96176e429947a6ada4797446fd8d2ffce098e0e5d106ff24f7b0.jpg)  
(b) Task-wise accuracy and confidence on CIFAR-100 after learning all 10 tasks.  
Figure 3: Calibration behavior in CIL. (a) The final-stage CIL model is compared with a static model jointly trained on all classes using five equal-frequency logit-margin bins from low to high margin. The confidenceaccuracy gap increases with margin only under CIL. (b) Task-wise accuracy and confidence are measured after learning all 10 tasks, showing larger gaps for earlier tasks than for newly introduced tasks.

## 4.1 Grouping Network Design

We define a set of group-specific temperature parameters $\{ T _ { 1 } , T _ { 2 } , \dots , T _ { M } \}$ , where M denotes the total number of groups. We then introduce a grouping network $g _ { \phi } ,$ ,whose goal is to map each sample to the most appropriate temperature according to its confidence-related characteristics.

Grouping network mechanism (Inference). The grouping network $g _ { \phi }$ is applied after the classifier, and takes the model's logits along with derived confidence statistics as inputs. For each sample x with classifier output $z = \ l _ { f _ { \theta } ( x ) } \in \mathbb { R } ^ { K }$ , we construct a calibration feature vector v(x) as:

$$
v ( x ) = [ \delta , p _ { \mathrm { o l d } } , p _ { \mathrm { n e w } } , \ p ] ,\tag{1}
$$

where $v ( x )$ concatenates informative signals capturing different aspects of model confidence and uncertainty. Specifically, $v ( x )$ consists of the softmax probability vector $p = \operatorname { s o f t m a x } ( z )$ , the logit margin $\delta ,$ the old-task confidence $p _ { \mathrm { o l d } }$ , and the new-task confidence $p _ { \mathrm { n e w } }$ . Then, the concatenated vector v(x) passes through the grouping network $g _ { \phi }$ as follows: gφ(x) = softmax $( W _ { \phi } v ( x ) + b _ { \phi } )$ where $\dot { W _ { \phi } } \in \mathbb { R } ^ { M \times d }$ and $b _ { \phi } \in \mathbb { R } ^ { M }$ are learnable parameters of the grouping network, M is the total number of calibration groups, and $v ( x ) \in \mathbb { R } ^ { d }$ denotes the calibration feature vector in Eq. 1. The output of the grouping function, $\mathrm { i . e . , } g _ { \phi } ( x )$ , produces a probability distribution over the M calibration groups. Consequently, the sample x is assigned to the group with the highest probability (e.g., group m), and the final calibrated output is given by $\begin{array} { r } { p _ { \mathrm { c a l } } ( x ) = \mathrm { s o f t m a x } \Big ( \frac { f _ { \theta } ( x ) } { T _ { m } } \Big ) } \end{array}$ . Here, $T _ { m }$ denotes the temperature associated with the group to which sample x is assigned by $g _ { \phi } ( x )$

Design of grouping network inputs. One of our key contributions is the CIL-aware input design in Eq. 1. The input features are designed to capture complementary aspects of prediction confidence under continual learning: the logit margin tracks overconfidence caused by forgetting under CIL, the old/new-task confidences explicitly summarize the old-versus-new class structure that is otherwise implicit in the full softmax vector, while the softmax vector preserves class-wise detail. Together, these signals allow the grouping network to distinguish samples with different calibration needs.

• Logit margin (δ): We define the logit margin as the difference between the two largest logits, $\delta ( x ) = z _ { ( 1 ) } - z _ { ( 2 ) }$ . To analyze calibration behavior across logit margins, we compare the final-stage CIL model with a static model jointly trained on all CIFAR-100 classes. Using the uncalibrated outputs of each model on the same test set, we divide samples into five equal-frequency logit-margin bins, $b _ { 1 } , \ldots , b _ { 5 }$ , ordered from low to high margin. We report the confidence-accuracy gap in each bin, averaged over five seeds. As shown in Figure 3a, unlike in the static setting, overconfidence under CIL increases systematically with the logit margin. Consistently, the NLL-optimal temperature increases with δ under CIL, while remaining nearly constant in the static model (Appendix B.10) indicating that high-margin predictions require stronger calibration correction under CIL. We exploit this relationship to predict calibration needs, using δ as a CIL-aware input signal.

• Old/new-task confidence $( p _ { \mathbf { 0 } \mathbf { l d } } , p _ { \mathbf { n e w } } ) \colon$ For a given sample x, we define the old-task confidence as the total probability mass assigned to previously learned classes, $\begin{array} { r } { p _ { \mathrm { o l d } } ( x ) = \sum _ { y \in \mathcal { C } _ { \mathrm { o l d } } } p ( y \mid x ) } \end{array}$ , where $p ( y \mid x )$ denotes the softmax probability for class y. If the current task is $b ,$ the old-class set is given by $\mathcal { C } _ { \mathrm { o l d } } = \mathcal { N } _ { 1 : b - 1 }$ . Similarly, we define the new-task confidence as $\begin{array} { r } { p _ { \mathrm { n e w } } ( x ) = \sum _ { y \in \mathcal { C } _ { \mathrm { n e w } } } p ( y \mid x ) } \end{array}$ , where $\mathcal { C } _ { \mathrm { n e w } } = \mathcal { D } _ { b }$ denotes the set of classes introduced in the current task. As shown in Figure 3b, after all 10 tasks, old tasks generally exhibit larger confidence-accuracy gaps than new tasks, revealing distinct calibration behavior. We therefore use their respective probability masses as CIL-aware signals to capture sample-specific calibration needs. These quantities explicitly summarize the old-versus-new class structure contained in the full softmax vector, directly exposing this CIL-relevant structure to the grouping network. Appendix B.12 shows consistent improvements over using only $[ \delta , p ]$ across routing-training data sizes, whereas random class-mass summaries yield no comparable benefit.

• Softmax probability vector (p): We use the full softmax probability vector, $p = \operatorname { s o f t m a x } ( f _ { \theta } ( x ) )$ to capture fine-grained class-wise confidence information. Even when samples have similar logit margins, their probability distributions across classes can differ, providing additional information for identifying different calibration needs. We further analyze this complementarity in Appendix B.11.

## 4.2 Training Grouping Network

At the beginning of each task, the validation set is divided into two disjoint subsets of equal size: $ { S _ { \mathrm { g r p } } }$ is used to train the grouping network $g _ { \phi }$ , while $ { S _ { \mathrm { c a l } } }$ is the set reserved for groupwise temperature scaling in the subsequent stage. This split prevents overfitting and ensures that temperature parameters are tuned on unseen calibration samples.

Grouping network training. During optimization, $g _ { \phi }$ and the group temperatures $\{ T _ { 1 } , T _ { 2 } , \dots , T _ { M } \}$ are trained jointly on $ { s } _ { \mathrm { g r p } }$ under the following objective:

$$
\mathcal { L } _ { \mathrm { M T C C } } = - \frac { 1 } { | { \cal S } _ { \mathrm { g r p } } | } \sum _ { ( x , y ) \in \cal S _ { \mathrm { g r p } } } \log \sum _ { m = 1 } ^ { M } g _ { \phi } ( x ) _ { m } \cdot s ( f _ { \theta } ( x ) , T _ { m } ) [ y ] .\tag{2}
$$

This objective is designed as a differentiable relaxation of hard groupwise temperature scaling. Ideally, each sample should be assigned to a single calibration group, but directly optimizing such discrete assignments is non-differentiable. We therefore use the soft routing weights $g _ { \phi } ( x ) _ { m }$ during this stage so that the grouping network and temperatures can be optimized jointly.

Groupwise temperature scaling. After training, the grouping network $g _ { \phi }$ partitions $ { S _ { \mathrm { c a l } } }$ into M disjoint groups according to the most probable assignment. Each sample x is thereby assigned to a single group and temperature scaling is applied independently within each group. Following the standard temperature scaling, the optimal temperature $\hat { T _ { m } ^ { * } }$ for group m is obtained by minimizing the NLL loss over its corresponding subset. Each group's initial temperature $T _ { m }$ is initialized with the value learned during the joint MT-CC optimization, which provides a well-informed starting point for fine-tuning on its local calibration subset. The resulting groupwise temperatures $\{ T _ { m } ^ { * } \}$ are then stored and reused as initialization in the next task's MT-CC stage.

## 4.3 Groupwise DCA Loss

We introduce the Groupwise Difference-between-Confidence-and-Accuracy (GDCA) loss as an auxiliary objective. The standard DCA loss penalizes the gap between the average confidence and empirical accuracy of a model [27]. We extend this idea to the group level by enforcing such alignment within each calibration group. Let $G _ { m }$ denote the subset of samples assigned to the m-th calibration group. The GDCA loss is defined as

$$
\mathcal { L } _ { \mathrm { G D C A } } = \sum _ { m = 1 } ^ { M } w _ { m } \left| \operatorname { A c c } ( G _ { m } ) - \operatorname { C o n f } ( G _ { m } ) \right| , \qquad w _ { m } = \frac { \sqrt { | G _ { m } | } } { \sum _ { j = 1 } ^ { M } \sqrt { | G _ { j } | } } ,\tag{3}
$$

where $\operatorname { A c c } ( G _ { m } )$ and $\mathrm { C o n f } ( G _ { m } )$ denote the empirical accuracy and average calibrated confidence within group $G _ { m } ,$ respectively. Here, $\operatorname { C o n f } ( G _ { m } )$ is computed using the calibrated mixture confidence $\begin{array} { r } { a ( x , \hat { y } ) \triangleq \sum _ { k = 1 } ^ { M } g _ { \phi } ( x ) _ { k } s ( z , T _ { k } ) [ \hat { y } ] } \end{array}$ for the predicted class $\hat { y } .$ GDCA gradients are carried by the confidence term, updating the temperature parameters. Through these shared temperatures, GDCA stabilizes the coupled routing-temperature optimization and promotes groupwise calibration consistency (see Appendix C for a detailed mechanism-level discussion). The overall training objective of MT-CC is written as $\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { M T C C } } + \lambda \mathcal { L } _ { \mathrm { G D C A } }$ , where λ controls the strength of GDCA.

Table 1: Comparison with various baseline methods. Since all evaluated methods are post-hoc calibration techniques, they preserve the original classification accuracy of the base models (65.24% for CIFAR-10, 56.47% for CIFAR-100, and 29.42% for Tiny-ImageNet). Each performance value is averaged over all sequential tasks in CIL. While maintaining identical classification performance, our method robustly outperforms the SOTA T-CIL in every calibration metric across diverse setups.
<table><tr><td rowspan="2">Dataset Method</td><td colspan="3">CIFAR-10</td><td colspan="3">CIFAR-100</td><td colspan="3">Tiny-ImageNet</td></tr><tr><td>ECE (↓)</td><td>AECE (↓)</td><td>SCE (↓)</td><td>ECE (↓)</td><td>AECE (↓)</td><td>SCE (↓)</td><td>ECE (↓)</td><td>AECE (↓)</td><td>SCE (↓)</td></tr><tr><td>Vanilla</td><td>28.51±0.85</td><td>28.49±0.86</td><td>8.83±0.31</td><td>27.59±0.11</td><td>27.57±0.11</td><td>1.35±0.02</td><td> $4 5 . 7 2 { \scriptstyle \pm 0 . 1 6 }$ </td><td>45.72±0.16</td><td>1.28±0.01</td></tr><tr><td>TS [12]</td><td>24.27±0.84</td><td>24.24±0.85</td><td> $8 . 1 6 { \pm } 0 . 3 0$ </td><td>16.13±0.19</td><td> $1 6 . 0 9 2 0 . 2 3$ </td><td> $0 . 9 9 2 0 . 0 1$ </td><td> $2 7 . 2 8 { \scriptstyle \pm 0 . 5 8 }$ </td><td> $2 7 . 3 0 { \scriptstyle \pm 0 . 6 0 }$ </td><td>0.93±0.01</td></tr><tr><td>ETS [50]</td><td>22.75±0.83</td><td>22.65±0.81</td><td> $5 . 9 2 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $1 6 . 7 0 { \scriptstyle \pm 0 . 3 9 }$ </td><td> $1 7 . 6 2 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $0 . 5 6 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $2 8 . 9 3 { \scriptstyle \pm 0 . 5 9 }$ </td><td> $2 8 . 9 0 { \scriptstyle \pm 0 . 6 3 }$ </td><td>0.52±0.01</td></tr><tr><td>IRM [50]</td><td>22.16±0.69</td><td> $2 2 . 0 1 { \scriptstyle \pm 0 . 6 5 }$ </td><td> $5 . 8 5 { \pm } 0 . 1 6 $ </td><td> $1 7 . 6 4 \pm 0 . 3 7$ </td><td> $1 7 . 5 1 { \pm } 0 . 3 9$ </td><td> $0 . 5 6 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $2 8 . 3 5 { \scriptstyle \pm 0 . 8 4 }$ </td><td> $2 8 . 3 2 { \scriptstyle \pm 0 . 8 0 }$ </td><td>0.52±0.01</td></tr><tr><td>PTS [42]</td><td>25.24±2.01</td><td> $2 5 . 1 0 { \scriptstyle \pm 1 . 9 9 }$ </td><td> $6 . 0 8 { \scriptstyle \pm 0 . 4 2 }$ </td><td>18.48±0.53</td><td> $1 8 . 3 7 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $0 . 5 7 { \scriptstyle \pm 0 . 0 1 }$ </td><td>36.39±1.08</td><td> $3 6 . 3 2 { \scriptstyle \pm 1 . 0 8 }$ </td><td>0.56±0.01</td></tr><tr><td> $\mathrm { \bf A d a } \mathrm { \bar { T S } } \mathrm { \bar { [ 1 7 ] } }$ </td><td> $2 2 . 4 5 { \scriptstyle \pm 0 . 9 6 }$ </td><td> $2 2 . 3 3 { \scriptstyle \pm 0 . 9 7 }$ </td><td> $5 . 9 3 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $1 7 . 5 4 { \pm } 0 . 5 1 $ </td><td> $1 7 . 4 8 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $0 . 5 6 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $2 8 . 7 7 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $2 8 . 7 4 { \scriptstyle \pm 0 . 7 0 }$ </td><td> $0 . 5 3 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td>MT-CC w/o aug</td><td>21.25±1.42 21.22±1.39 5.52±0.23</td><td></td><td></td><td></td><td></td><td></td><td>15.96±1.05 15.88±1.09 0.54±0.01 26.00±0.92 25.97±0.90 0.51±0.01</td><td></td><td></td></tr><tr><td> $\mathrm { T } { \mathrm { C I L } } \left[ 1 6 \right]$ </td><td>15.34±1.34</td><td> $1 5 . 3 2 { \scriptstyle \pm 1 . 3 2 }$ </td><td> $7 . 1 4 \pm 0 . 3 8$ </td><td> $6 . 0 3 { \scriptstyle \pm 0 . 4 2 }$ </td><td>6.04±0.35</td><td>0.94±0.03</td><td> $1 8 . 9 6 { \scriptstyle \pm 0 . 3 5 }$ </td><td>18.99±0.40</td><td>0.82±0.01</td></tr><tr><td> $\mathrm { T } \mathrm { - } \mathrm { C I L } + \mathrm { E T S }$ </td><td> $1 3 . 9 4 { \scriptstyle \pm 1 . 3 5 }$ </td><td> $1 4 . 0 4 { \scriptstyle \pm 1 . 3 0 }$ </td><td> $5 . 3 8 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $5 . 6 3 { \pm } 0 . 5 2 $ </td><td> $5 . 8 4 \pm 0 . 4 6$ </td><td> $0 . 5 3 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $2 2 . 1 4 { \pm } 0 . 2 2$ </td><td> $2 2 . 2 0 { \scriptstyle \pm 0 . 2 2 }$ </td><td>0.51±0.01</td></tr><tr><td> $\mathrm { T } \mathrm { - } \mathrm { C I L } + \mathrm { I R M }$ </td><td> $1 4 . 6 9 { \scriptstyle \pm 1 . 1 6 }$ </td><td> $1 4 . 9 4 { \scriptstyle \pm 1 . 0 1 }$ </td><td> $5 . 3 7 { \scriptstyle \pm 0 . 3 7 }$ </td><td> $6 . 7 8 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $6 . 8 1 \pm 0 . 6 1$ </td><td> $\mathbf { 0 . 5 1 \pm 0 . 0 1 }$ </td><td> $1 7 . 8 7 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $1 8 . 0 1 { \scriptstyle \pm 0 . 3 6 }$ </td><td>0.49±0.01</td></tr><tr><td> $\mathrm { T } \mathrm { - } \mathrm { C I L } + \mathrm { P T S }$ </td><td> $1 5 . 0 0 { \scriptstyle \pm 1 . 6 7 }$ </td><td> $1 4 . 9 3 { \scriptstyle \pm 1 . 5 5 }$ </td><td> $5 . 3 2 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $9 . 8 7 { \scriptstyle \pm 0 . 7 7 }$ </td><td> $1 0 . 1 4 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $0 . 5 5 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $2 1 . 8 1 { \scriptstyle \pm 0 . 4 7 }$ </td><td> $2 2 . 0 4 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $0 . 5 1 { \scriptstyle \pm 0 . 0 1 }$ </td></tr><tr><td> $\mathrm { T } \mathrm { - } \mathrm { C I L } + \mathrm { A d a T S }$ </td><td> $1 4 . 0 3 { \pm } 1 . 5 9 $ </td><td> $1 4 . 0 0 { \scriptstyle \pm 1 . 5 5 }$ </td><td> $5 . 4 1 { \scriptstyle \pm 0 . 3 7 }$ </td><td> $7 . 6 7 { \pm } 0 . 5 2 $ </td><td> $7 . 6 6 \pm 0 . 4 9$ </td><td> $0 . 5 3 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $2 0 . 2 5 { \scriptstyle \pm 0 . 5 8 }$ </td><td> $2 0 . 2 3 { \scriptstyle \pm 0 . 5 9 }$ </td><td> $0 . 5 8 { \scriptstyle \pm 0 . 0 1 }$ </td></tr><tr><td>MT-CC</td><td>10.18±1.48 10.16±1.54 4.64±0.25</td><td></td><td></td><td> $\mathbf { 4 . 4 9 \pm 0 . 5 3 }$ </td><td></td><td></td><td>4.44±0.49 0.51±0.01|15.82±0.29 15.80±0.32 0.52±0.01</td><td></td><td></td></tr></table>

![](images/a9ed2c051ccad58d22d45bd294375862b2efa51ef036bddff0383b0e466b4e1d.jpg)

![](images/7818edf77aafc665bf61af33e1085a4fe5f039945b97a3cbbabdbd67149e977d.jpg)  
(a) Naive grouping  
(b) MT-CC grouping  
Figure 4: Portion of samples from each class allocated to each group. Results are shown after training on the 10th task of CIFAR-100. Compared to (a) naive confidence-based grouping, which yields imbalanced groups, (b) MT-CC achieves a more effective group allocation, leading to lower ECE.

## 5 Experimental Results

We evaluate the proposed MT-CC framework by measuring the calibration performance under the CIL setting. All results are reported over five seeds for CIFAR-10, CIFAR-100, Tiny-ImageNet, and ViT-based CIL experiments, and over three seeds for ImageNet.

Datasets and CIL setup. We conduct experiments on the widely used benchmarks for CIL: CIFAR-10 [21], CIFAR-100 [21], Tiny-ImageNet [24] and ImageNet [10]. Following the training protocol of T-CIL [16], for CIFAR-10 and CIFAR-100, we use a ResNet-32 backbone [13] with 5 tasks (2 classes per task) and 10 tasks (10 classes per task), retaining memory sizes of 200 and 2,000 samples, respectively. For Tiny-ImageNet, we adopt ResNet-18 and construct 10 tasks of 20 classes each, using 2,000 memory samples. For ImageNet, we employ ResNet-18 and partition the 1,000 classes into 10 tasks, forming a challenging setup with memory sizes of 5,000, 10,000, and 20,000 samples. We also evaluate ViT-B/16 [11] with ER on CIFAR-100 and ImageNet-R [15], using 10 tasks, a 2,000-sample replay buffer, and 500 validation samples.

Calibration metrics. We evaluate calibration using three standard metrics: Expected Calibration Error (ECE) [33], Adaptive Expected Calibration Error (AECE) [32], and Static Calibration Error (SCE) [34]. ECE measures the gap between predicted confidence and empirical accuracy using fixed confidence bins, while AECE uses adaptive binning based on the confidence distribution of the test data. SCE extends calibration evaluation to the multi-class setting by considering class-wise probabilities. In our experiments, we use 10 bins for all metrics.

Baselines. We compare MT-CC with representative post-hoc calibration methods. (i) Vanilla denotes the continual learning model without calibration. (ii) TS [12] learns a single global temperature. (iii) ETS [50] extends TS through an ensemble of probability transformations. (iv) IRM [50] performs non-parametric calibration via multi-class isotonic regression. (v) PTS [42] learns prediction-specific temperatures using a lightweight neural network. (vi) AdaTS [17] estimates instance-wise temperatures using a class-conditional VAE and a lightweight neural network. (vii) T-CIL [16] optimizes TS using adversarially perturbed exemplars for class-incremental learning. (viii) For stronger baselines, we apply ETS, IRM, PTS, and AdaTS to T-CIL [16], denoted as T-CIL+[method]. To ensure a fair comparison in CIL, we report calibration results under two validation protocols: the raw new-task validation set and the T-CIL-augmented validation set. Within each protocol, all baselines use the same number of validation samples for temperature tuning. This allows us to compare standard post-hoc methods and CIL-specific methods under consistent validation settings. Importantly, all methods preserve the same accuracy since they operate in a post-hoc manner.

Table 2: ImageNet results with varying replay buffer sizes. Since all evaluated methods are post-hoc, they share the same accuracies of the base models: for ER, 32.86% (5K), 41.23% (10K), 49.52% (20K); for iCaRL, 44.08% (5K), 49.29% (10K), 54.24% (20K). Reflecting that SCE values vary with the number of labels, per mille (%o) is used.
<table><tr><td colspan="2">Replay buffer</td><td colspan="3">5K</td><td colspan="3">10K</td><td colspan="3">20K</td></tr><tr><td>CL</td><td>ČA</td><td>ECE(↓)</td><td>AECE(↓)</td><td>SCE(↓)</td><td>ECE(↓)</td><td>AECE(↓)</td><td>SCE(↓)</td><td>ECE(↓)</td><td>AECE(↓)</td><td>SCE(↓)</td></tr><tr><td>ER</td><td>Vanilla</td><td>29.95±0.30</td><td>29.95±0.30</td><td>2.18±0.02</td><td>|22.46±0.19</td><td>22.46±0.19</td><td>1.80±0.02</td><td>16.39±0.23 16.38±0.23</td><td></td><td>1.45±0.02</td></tr><tr><td></td><td>TS [12]</td><td>29.66±0.33</td><td>29.63±0.33</td><td>2.16±0.02</td><td>22.67±0.17</td><td>22.65±0.18</td><td>1.79±0.02</td><td>16.83±0.18 16.80±0.20</td><td></td><td>1.44±0.01</td></tr><tr><td></td><td>ETS [50]</td><td>30.68±0.25</td><td>30.65±0.26</td><td>1.09±0.01</td><td>23.53±0.16</td><td>23.53±0.16</td><td>1.09±0.01</td><td>17.62±0.22 17.58±0.23</td><td></td><td>0.91±0.00</td></tr><tr><td></td><td>PTS [42]</td><td>30.32±0.44</td><td>30.32±0.44</td><td>1.09±0.01</td><td>22.79±0.36</td><td>22.80±0.36</td><td>1.89±0.01</td><td>16.52±0.27 16.51±0.28</td><td></td><td>0.91±0.01</td></tr><tr><td></td><td>T-CIL [16]</td><td>13.45±0.15</td><td>13.43±0.13</td><td>1.79±0.01</td><td>7.74±0.23</td><td>7.72±0.24</td><td>1.49±0.01</td><td>10.57±0.22 10.52±0.24</td><td></td><td>1.27±0.00</td></tr><tr><td></td><td>MT-CC</td><td>12.59±0.56</td><td>12.62±0.56 1.05±0.01</td><td></td><td>7.25±0.56</td><td>7.26±0.56 0.91±0.01</td><td></td><td>8.44±0.69 8.44±0.69</td><td></td><td>0.82±0.02</td></tr><tr><td>iCaRL Vanilla</td><td></td><td>22.85±0.18</td><td>22.85±0.19</td><td>1.93±0.02</td><td>17.91±0.2317.90±0.23</td><td></td><td>1.68±0.02</td><td>12.94±0.83 12.93±0.82</td><td></td><td>1.44±0.05</td></tr><tr><td></td><td>TS [12]</td><td>22.24±0.31</td><td>22.22±0.29</td><td>1.93±0.02</td><td>18.08±0.38 18.07±0.37</td><td></td><td>1.68±0.02</td><td>16.18±2.88 16.16±2.88</td><td></td><td>1.47±0.10</td></tr><tr><td></td><td>ETS [50]</td><td>22.79±0.12</td><td>22.78±0.12</td><td>1.16±0.01</td><td>18.67±0.43 18.64±0.44</td><td></td><td>1.03±0.01</td><td>15.10±0.11 15.08±0.11</td><td></td><td>0.90±0.01</td></tr><tr><td></td><td>PTS [42]</td><td>21.82±0.05</td><td>21.83±0.04</td><td>0.96±0.00</td><td>17.63±0.16 17.63±0.17</td><td></td><td>0.83±0.01</td><td>14.07±0.09 14.09±0.10</td><td></td><td>0.99±0.01</td></tr><tr><td></td><td>T-CIL [16]</td><td>5.99±0.30</td><td>5.98±0.31</td><td>1.80±0.02</td><td>6.84±0.01</td><td>6.80±0.04</td><td>1.57±0.01</td><td>9.69±1.48 9.64±1.47</td><td></td><td>1.36±0.03</td></tr><tr><td></td><td>MT-CC</td><td>4.87±0.26</td><td>4.88±0.25</td><td>1.05±0.01</td><td>5.16±0.14 5.12±0.12</td><td></td><td>0.95±0.01</td><td>6.88±0.79 6.80±0.79</td><td></td><td>0.85±0.01</td></tr></table>

Table 3: ViT-based CIL post-hoc calibration results over five seeds. Base accuracy is preserved at 90.99% on CIFAR-100 and 79.23% on ImageNet-R. Metrics are in $1 0 ^ { - 2 }$ scale.
<table><tr><td rowspan="2">Method</td><td colspan="3">CIFAR-100</td><td colspan="3">ImageNet-R</td></tr><tr><td>ECE</td><td>AECE</td><td>SCE</td><td>ECE</td><td>AECE</td><td>SCE</td></tr><tr><td>Vanilla</td><td>6.23±0.19</td><td>0.41±0.02</td><td>0.34±0.02</td><td>10.68±0.31</td><td>0.38±0.01</td><td>0.37±0.03</td></tr><tr><td>TS [12]</td><td>6.44±0.20</td><td>0.30±0.02</td><td>0.34±0.03</td><td>11.80±0.29</td><td>0.32±0.01</td><td>0.38±0.03</td></tr><tr><td>ETS [50]</td><td>9.82±0.37</td><td>0.29±0.02</td><td>0.34±0.03</td><td>17.30±0.93</td><td>0.29±0.02</td><td>0.39±0.03</td></tr><tr><td>IRM [50]</td><td>8.39±0.34</td><td>0.31±0.01</td><td>0.36±0.03</td><td>15.51±0.99</td><td>0.28±0.02</td><td>0.41±0.03</td></tr><tr><td>PTS [42]</td><td>9.66±0.44</td><td>0.29±0.02</td><td>0.34±0.03</td><td>18.43±1.03</td><td>0.27±0.02</td><td>0.38±0.03</td></tr><tr><td>AdaTS [17]</td><td>6.00±0.68</td><td>0.28±0.02</td><td>0.34±0.02</td><td>8.93±0.66</td><td>0.28±0.01</td><td>0.37±0.02</td></tr><tr><td>MT-CC w/o aug</td><td>5.10±0.49</td><td>0.27±0.01</td><td>0.33±0.02</td><td>7.99±0.63</td><td>0.28±0.02</td><td>0.36±0.03</td></tr><tr><td>T-CIL [16]</td><td>5.23±0.54</td><td>0.30±0.02</td><td>0.34±0.03</td><td>8.04±0.71</td><td>0.29±0.02</td><td>0.36±0.03</td></tr><tr><td>T-CIL + ETS</td><td>5.49±0.53</td><td>0.29±0.01</td><td>0.35±0.03</td><td>8.13±0.72</td><td>0.28±0.01</td><td>0.37±0.02</td></tr><tr><td>T-CIL + IRM</td><td>5.51±0.57</td><td>0.29±0.02</td><td>0.34±0.03</td><td>7.95±0.64</td><td>0.28±0.01</td><td>0.37±0.03</td></tr><tr><td>T-CIL + PTS</td><td>6.03±0.66</td><td>0.29±0.02</td><td>0.34±0.03</td><td>10.00±0.84</td><td>0.29±0.01</td><td>0.38±0.02</td></tr><tr><td> $\mathrm { T } \mathrm { - } \mathrm { C I L } + \mathrm { A d a T S }$ </td><td>6.04±0.61</td><td>0.27±0.02</td><td>0.33±0.02</td><td>9.03±0.85</td><td>0.28±0.01</td><td>0.37±0.02</td></tr><tr><td>MT-CC</td><td>4.57±0.28</td><td>0.26±0.02</td><td>0.33±0.03</td><td>7.55±0.29</td><td>0.26±0.01 0.30±0.03</td><td></td></tr></table>

Implementation details. For all datasets, the ResNet-based main experiments use a single fully connected layer for the grouping network $g _ { \phi } ( x )$ with a learning rate of 0.01. The GDCA regularization weight is set to 1.0 for ResNet and 0.1 for ViT experiments. The initial temperature values are set to [1.5, 1.5], and the validation set is generated using the T-CIL framework and divided into $ { s } _ { \mathrm { g r p } }$ and $ { S _ { \mathrm { c a l } } }$ with equal sizes. The module is optimized using the Adam optimizer [19] with a batch size of 16 for 50 epochs. The number of groups is set to M = 2 in our experiments, as the two-group configuration is sufficient to provide reliable calibration performance, as shown in Appendix B.1. The main experiments are conducted based on ER [6]. To further verify the applicability of MT-CC we additionally compare the performance of different approaches using representative CIL methods. ViT-based experiments and their implementation details are provided in Appendix A.

## 5.1 Main Experimental Results

Comparison with baselines. Table 1 compares MT-CC with representative accuracy-preserving posthoc calibration methods. MT-CC achieves the lowest ECE and AECE across CIFAR-10, CIFAR-100, and Tiny-ImageNet, while remaining competitive in SCE. We further report MT-CC w/o aug, which denotes MT-CC trained on the raw new-task validation set without T-CIL's augmented validation samples. It remains competitive among raw-validation baselines, suggesting that the multi-group temperature design drives the calibration gains.

Analysis of group partitioning behavior. Figure 4 compares the group allocation results of naive grouping and MT-CC. Naive grouping, which only takes confidence values as input, leads to a highly imbalanced class-group distribution in the final task. MT-CC maintains a balanced distribution across groups and effectively distinguishes samples with different calibration behaviors. The grouping network in MT-CC leads to a higher temperature in Group 2, which contains not only the majority of new-task samples (classes 91 – 100) but also many of the old-task samples (classes 1 – 90). As a result, this adaptive grouping and temperature assignment lead to improved calibration performance

Table 4: Ablation on grouping and input signals of MT-CC.
<table><tr><td rowspan="2">Exp</td><td colspan="3">Ablated Components</td><td colspan="2">CIFAR-10</td><td colspan="2">CIFAR-100</td><td colspan="2">Tiny-ImageNet</td></tr><tr><td>Logit Margin</td><td>Old/New Confidence</td><td>GDCA</td><td>ECE (↓)</td><td>AECE (↓)</td><td>ECE (↓)</td><td>AECE (↓)</td><td>ECE (↓)</td><td>AECE (↓)</td></tr><tr><td>A (All)</td><td>√</td><td></td><td></td><td>10.18±1.48</td><td>10.16±1.54</td><td>4.49±0.53</td><td>4.44±0.49</td><td>15.82±0.29</td><td>15.80±0.32</td></tr><tr><td>B</td><td>×</td><td></td><td>√ √</td><td>10.12±1.52</td><td>10.12±1.53</td><td>4.95±0.37</td><td>4.93±0.34</td><td>15.90±0.36</td><td>15.90±0.41</td></tr><tr><td>C</td><td>√</td><td>√ √</td><td>×</td><td> $1 1 . 8 1 { \pm } 1 . 4 1 $ </td><td>11.81±1.40</td><td>4.78±0.42</td><td>4.79±0.40</td><td>15.93±0.37</td><td>15.94±0.42</td></tr><tr><td>D</td><td>√</td><td></td><td>√</td><td>10.91±1.26</td><td>10.91±1.26</td><td>4.63±0.30</td><td>4.65±0.26</td><td>19.48±0.40</td><td>19.47±0.44</td></tr><tr><td>ε</td><td>X</td><td>× ×</td><td>X</td><td>12.45±1.23</td><td>12.42±1.20</td><td>5.54±0.42</td><td>5.54±0.41</td><td>19.69±0.83</td><td>19.68±0.86</td></tr></table>

![](images/e96b564392c8245872d49fca0b87ed549f51b2c816c9b32802dee1ba41393ca4.jpg)  
Figure 5: Effect of GDCA weight λ on overall calibration. Across a broad range of λ values, both MT-CC variants (with and without GDCA) consistently outperform the baseline.

Table 5: Versatility with other CIL methods.
<table><tr><td>Method</td><td>ECE</td><td>AECE</td><td>SCE</td></tr><tr><td>EEIL + TS</td><td>14.29±0.63</td><td>14.29±0.64</td><td>1.05±0.02</td></tr><tr><td>EEIL + T-CIL</td><td>4.28±0.35</td><td>4.34±0.28</td><td>1.01±0.03</td></tr><tr><td>EEIL + MT-CC</td><td>3.67±0.20</td><td>3.71±0.18</td><td>0.58±0.01</td></tr><tr><td>BiC + TS</td><td>34.19±1.54</td><td>34.13±1.55</td><td>1.55±0.05</td></tr><tr><td>BiC + T-CIL</td><td>27.52±0.85</td><td>27.44±0.83</td><td>1.41±0.04</td></tr><tr><td>BiC + MT-CC</td><td>26.73±0.99</td><td>26.65±0.97</td><td>0.84±0.03</td></tr><tr><td>iCaRL + TS</td><td>17.89±0.41</td><td>17.83±0.42</td><td>1.24±0.03</td></tr><tr><td>iCaRL + T-CIL</td><td>8.64±1.12</td><td>8.62±1.14</td><td>1.13±0.04</td></tr><tr><td>iCaRL + MT-CC</td><td>6.00±1.02</td><td>6.10±0.99</td><td>0.68±0.02</td></tr><tr><td>IL2M + TS</td><td>10.15±0.43</td><td>10.10±0.41</td><td>0.88±0.02</td></tr><tr><td>IL2M + T-CIL</td><td>8.54±1.37</td><td>8.62±1.33</td><td>0.90±0.02</td></tr><tr><td>IL2M + MT-CC</td><td>7.72±0.89</td><td>7.75±0.86</td><td>0.47±0.02</td></tr><tr><td>DER + TS</td><td>17.68±0.18</td><td>17.63±0.20</td><td>0.90±0.01</td></tr><tr><td>DER + T-CIL</td><td>6.18±0.31</td><td>6.12±0.30</td><td>0.79±0.02</td></tr><tr><td> $\mathrm { D E R } + \mathbf { M T } \mathbf { - C C }$ </td><td>4.10±0.39</td><td>4.06±0.41</td><td>0.69±0.01</td></tr></table>

Table 6: Comparison of task-wise grouping, per-task TS, and the proposed MT-CC.
<table><tr><td>Method</td><td>T1</td><td>T2</td><td>T3</td><td>T4</td><td>T5</td><td>T6</td><td>T7</td><td>T8</td><td>T9</td><td>T10</td><td>Overall</td></tr><tr><td>GT Old/New TS</td><td>2.18±0.63</td><td>11.50±1.77</td><td>5.43±2.05</td><td>4.81±1.11</td><td>3.47±1.28</td><td>3.76±0.46</td><td>4.35±1.97</td><td>6.00±2.34</td><td>7.88±2.22</td><td>10.37±2.02</td><td>5.97±0.32</td></tr><tr><td>GT Per-task TS</td><td>2.18±0.63</td><td>10.79±1.60</td><td>5.42±2.03</td><td>4.80±1.11</td><td>3.47±1.27</td><td>3.77±0.46</td><td>4.34±1.96</td><td>6.00±2.33</td><td>7.88±2.22</td><td>10.42±1.93</td><td>5.91±0.31</td></tr><tr><td>MT-CC</td><td>2.18±0.63</td><td>6.99±2.03</td><td>3.22±0.94</td><td>2.32±0.50</td><td>3.16±0.65</td><td>2.49±0.36</td><td>4.75±2.39</td><td>4.72±3.27</td><td>7.11±2.75</td><td>7.97±2.13</td><td>4.49±0.53</td></tr></table>

## 5.2 Further Studies

Scalability and backbone generality (ImageNet and ViT). We further evaluate MT-CC on ImageNet and ViT-based CIL settings. Table 2 shows that MT-CC remains effective on ImageNet with 1,000 classes across different memory budgets and replay-based CIL methods. Table 3 further demonstrates that MT-CC generalizes beyond CNN backbones, achieving strong calibration performance with ViT on CIFAR-100 and ImageNet-R while preserving the same base accuracy. Overall, these results indicate that MT-CC is robust to larger output spaces and different backbone architectures.

Ablation on grouping and input signals. Table 4 analyzes the contribution of MT-CC components. Removing either the logit margin or old/new confidence feature generally increases ECE and AECE, indicating that both contribute useful calibration signals. Eliminating GDCA further degrades performance, confirming its role in improving groupwise calibration.

Effect of GDCA regularization. Figure 5 illustrates the effect of the GDCA weight, evaluated on the CIFAR-100 dataset. GDCA improves ECE across a wide range of λ values, confirming its effectiveness. Notably, both MT-CC variants, with and without GDCA, consistently outperform the T-CIL baseline, further demonstrating the robustness of our method.

Compatibility with other CIL methods. To further evaluate the compatibility of MT-CC, we conduct experiments by integrating it with several milestone replay-based CIL methods, including EEIL [4], iCaRL [38], BiC [43], IL2M [3], and DER [44]. CIFAR-100 is adopted for experiments. As shown in Table 5, MT-CC consistently achieves the best calibration performance across all frameworks, demonstrating its plug-and-play applicability to diverse CIL methods.

Comparison with ground-truth grouping baselines. Table 6 compares MT-CC with two oracle ground-truth grouping baselines on CIFAR-100. The first baseline, GT Old/New TS, splits samples into two groups, old-task classes $\scriptstyle { \mathcal { D } } _ { 1 : b - 1 }$ and the new-task classes Yb, and applies temperature scaling to each group. The second baseline, GT Per-task TS, uses a more fine-grained oracle partition by assigning samples to their corresponding task groups and learning a separate temperature for each task. Since each per-task group contains fewer calibration samples, we use a smaller learning rate for GT Per-task TS to ensure stable and fair optimization. Despite this oracle advantage, MT-CC achieves lower ECE than both grouping baselines in most tasks. These results show that MT-CC learns a more effective calibration boundary than rule-based task grouping.

## 6 Conclusion

In this work, we proposed MT-CC, a multi-group temperature scaling strategy for handling asymmetric calibration behavior in CIL. MT-CC incorporates a strategically designed grouping network that leverages old/new-task confidence and logit margin as inputs. The GDCA loss further stabilizes this network and promotes intra-group consistency without sacrificing accuracy. Extensive experiments demonstrate that MT-CC achieves state-of-the-art calibration performance in CIL, providing a promising direction for building reliable models in continual learning settings.

Limitations and future work. A deeper theoretical understanding of MT-CC remains to be developed. Extending MT-CC to different continual learning domains (e.g., large language models) is also an important direction for future work.

## Acknowledgments

This work was supported in part by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2026-25470983), the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (RS-2024- 00457882, AI Research Hub Project, IITP-2026-RS-2026-25544188, Leading Generative AI Human Resources Development), and the Technology Development Program of MSS (RS-2025-25464351). Dong-Jun Han is the corresponding author.

## References

[1] Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny, Marcus Rohrbach, and Tinne Tuytelaars. Memory aware synapses: Learning what (not) to forget. In Proceedings of the European conference on computer vision (ECCV), pages 139–154, 2018.

[2] Rahaf Aljundi, Eugene Belilovsky, Tinne Tuytelaars, Laurent Charlin, Massimo Caccia, Min Lin, and Lucas Page-Caccia. Online continual learning with maximal interfered retrieval. Advances in neural information processing systems, 32, 2019.

[3] Eden Belouadah and Adrian Popescu. Il2m: Class incremental learning with dual memory. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2019.

[4] Francisco M Castro, Manuel J Marín-Jiménez, Nicolás Guil, Cordelia Schmid, and Karteek Alahari. End-to-end incremental learning. In Proceedings of the European conference on computer vision (ECCV), pages 233–248, 2018.

[5] Arslan Chaudhry, Marc'Aurelio Ranzato, Marcus Rohrbach, and Mohamed Elhoseiny. Efficient lifelong learning with a-gem. arXiv preprint arXiv:1812.00420, 2018.

[6] Arslan Chaudhry, Marcus Rohrbach, Mohamed Elhoseiny, Thalaiyasingam Ajanthan, Puneet K Dokania, Philip HS Torr, and Marc'Aurelio Ranzato. On tiny episodic memories in continual learning. arXiv preprint arXiv:1902.10486, 2019.

[7] Wonjeong Choi, Jungwuk Park, Dong-Jun Han, Younghyun Park, and Jaekyun Moon. Consistency-guided temperature scaling using style and content information for out-of-domain calibration. Proceedings of the AAAI Conference on Artificial Intelligence, 38(10):11588–11596, 2024. doi: 10.1609/aaai.v38i10.29041.

[8] Yun-Wei Chu, Dong-Jun Han, Seyyedali Hosseinalipour, and Christopher Brinton. Unlocking the potential of model calibration in federated learning. In International Conference on Learning Representations, 2025.

[9] Matthias De Lange, Rahaf Aljundi, Marc Masana, Sarah Parisot, Xu Jia, Aleš Leonardis, Gregory Slabaugh, and Tinne Tuytelaars. A continual learning survey: Defying forgetting in

classification tasks. IEEE transactions on pattern analysis and machine intelligence, 44(7): 3366–3385, 2021.

[10] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

[11] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale, 2020.

[12] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In International conference on machine learning, pages 1321–1330. PMLR, 2017.

[13] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

[14] Ramya Hebbalaguppe, Jatin Prakash, Neelabh Madan, and Chetan Arora. A stitch in time saves nine: A train-time regularizing loss for improved neural network calibration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16081–16090, 2022.

[15] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8340–8349, 2021.

[16] Seong-Hyeon Hwang, Minsu Kim, and Steven Euijong Whang. T-cil: Temperature scaling using adversarial perturbation for calibration in class-incremental learning. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15339–15348, 2025.

[17] Tom Joy, Francesco Pinto, Ser-Nam Lim, Philip HS Torr, and Puneet K Dokania. Sampledependent adaptive temperature scaling for improved calibration. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 14919–14926, 2023.

[18] Do-Yeon Kim, Dong-Jun Han, Jun Seo, and Jaekyun Moon. Warping the space: Weight space rotation for class-incremental few-shot learning. In The Eleventh International Conference on Learning Representations, 2023.

[19] Diederik P Kingma. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

[20] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13):3521–3526, 2017.

[21] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

[22] Meelis Kull, Miquel Perello Nieto, Markus Kängsepp, Telmo Silva Filho, Hao Song, and Peter Flach. Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with dirichlet calibration. Advances in neural information processing systems, 32, 2019.

[23] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in neural information processing systems, 30, 2017.

[24] Yann Le and Xuan Yang. Tiny imagenet visual recognition challenge. CS 231N, 7(7):3, 2015.

[25] Lanpei Li, Elia Piccoli, Andrea Cossu, Davide Bacciu, and Vincenzo Lomonaco. Calibration of continual learning models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4160–4169, 2024.

[26] Zhizhong Li and Derek Hoiem. Learning without forgetting. IEEE transactions on pattern analysis and machine intelligence, 40(12):2935–2947, 2017.

[27] Gongbo Liang, Yu Zhang, Xiaoqin Wang, and Nathan Jacobs. Improved trainable calibration method for neural networks on medical imaging classification. arXiv preprint arXiv:2009.04057, 2020.

[28] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision, pages 2980–2988, 2017.

[29] David Lopez-Paz and Marc'Aurelio Ranzato. Gradient episodic memory for continual learning. Advances in neural information processing systems, 30, 2017

[30] Arun Mallya, Dillon Davis, and Svetlana Lazebnik. Piggyback: Adapting a single network to multiple tasks by learning to mask weights. In Proceedings of the European conference on computer vision (ECCV), pages 67–82, 2018.

[31] Marc Masana, Xialei Liu, Bartłomiej Twardowski, Mikel Menta, Andrew D Bagdanov, and Joost Van De Weijer. Class-incremental learning: survey and performance evaluation on image classification. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(5):5513– 5533, 2022.

[32] Jishnu Mukhoti, Viveka Kulharia, Amartya Sanyal, Stuart Golodetz, Philip Torr, and Puneet Dokania. Calibrating deep neural networks using focal loss. Advances in neural information processing systems, 33:15288–15299, 2020.

[33] Mahdi Pakdaman Naeini, Gregory Cooper, and Milos Hauskrecht. Obtaining well calibrated probabilities using bayesian binning. In Proceedings of the AAAI conference on artificial intelligence, volume 29, 2015.

[34] Jeremy Nixon, Mike Dusenberry, Ghassen Jerfel, Timothy Nguyen, Jeremiah Liu, Linchuan Zhang, and Dustin Tran. Measuring calibration in deep learning. arXiv preprint arXiv:1904.01685, 2020.

[35] German I Parisi, Ronald Kemker, Jose L Part, Christopher Kanan, and Stefan Wermter. Continual lifelong learning with neural networks: A review. Neural networks, 113:54–71, 2019.

[36] Gabriel Pereyra, George Tucker, Jan Chorowski, Łukasz Kaiser, and Geoffrey Hinton. Regularizing neural networks by penalizing confident output distributions. arXiv preprint arXiv:1701.06548, 2017.

[37] John Platt et al. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. Advances in large margin classifiers, 10(3):61–74, 1999.

[38] Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H Lampert. icarl: Incremental classifier and representation learning. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pages 2001–2010, 2017.

[39] Andrei A Rusu, Neil C Rabinowitz, Guillaume Desjardins, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, and Raia Hadsell. Progressive neural networks. arXiv preprint arXiv:1606.04671, 2016.

[40] Joan Serra, Didac Suris, Marius Miron, and Alexandros Karatzoglou. Overcoming catastrophic forgetting with hard attention to the task. In International conference on machine learning, pages 4548–4557. PMLR, 2018.

[41] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2818–2826, 2016.

[42] Christian Tomani, Daniel Cremers, and Florian Buettner. Parameterized temperature scaling for boosting the expressive power in post-hoc uncertainty calibration. In European conference on computer vision, pages 555–569. Springer, 2022.

[43] Yue Wu, Yinpeng Chen, Lijuan Wang, Yuancheng Ye, Zicheng Liu, Yandong Guo, and Yun Fu. Large scale incremental learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 374–382, 2019.

[44] Shipeng Yan, Jiangwei Xie, and Xuming He. Der: Dynamically expandable representation for class incremental learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3014–3023, 2021.

[45] Jia-Qi Yang, De-Chuan Zhan, and Le Gan. Beyond probability partitions: Calibrating neural networks with semantic aware grouping. Advances in Neural Information Processing Systems, 36:58448–58460, 2023.

[46] Bianca Zadrozny and Charles Elkan. Obtaining calibrated probability estimates from decision trees and naive bayesian classifiers. In Icml, volume 1, 2001.

[47] Bianca Zadrozny and Charles Elkan. Transforming classifier scores into accurate multiclass probability estimates. In Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining, pages 694–699, 2002.

[48] Friedemann Zenke, Ben Poole, and Surya Ganguli. Continual learning through synaptic intelligence. In International conference on machine learning, pages 3987–3995. PMLR, 2017.

[49] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412, 2017.

[50] Jize Zhang, Bhavya Kailkhura, and T Yong-Jin Han. Mix-n-match: Ensemble and compositional methods for uncertainty calibration in deep learning. In International conference on machine learning, pages 11117–11128. PMLR, 2020.

[51] Bowen Zhao, Xi Xiao, Guojun Gan, Bin Zhang, and Shu-Tao Xia. Maintaining discrimination and fairness in class incremental learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13208–13217, 2020.

## Appendix

A Training Hyperparameters and Implementation Details 15   
A.1 Representative Replay-Based CIL Methods 16   
A.2 Details of Calibration Metrics 16   
B Additional Ablation Studies 17   
B.1 Ablation on the Number of Groups 17   
B.2 Ablation on the Replay Memory Size 17   
B.3 Comparison of ECE, AECE, SCE Across Sequential Tasks on CIFAR-100 . 18   
B.4 Results on CIFAR-100: 60-5 Split 18   
B.5 Robustness to ECE Binning and Brier Score 19   
B.6 Applicability to Static Classification 19   
B.7 Effect of Validation Set Size 19   
B.8 Inference Time Analysis 20   
B.9 Comparison with Feature-Matched and Grouped Calibration Methods 20   
B.10 CIL Changes the Calibration Meaning of Logit Margin 20   
B.11 Complementarity of Margin and Confidence 21   
B.12 Benefit of Explicit Old/New Class Structure 22   
B.13 Continual Update and Stability of the Grouping Network 23   
C How GDCA Stabilizes Routing-Temperature Optimization 24   
D Catastrophic Forgetting and Asymmetric Calibration Behaviors 25   
E Training and Inference Procedure of MT-CC 26   
F Reliability Diagram 27   
F.1 Group-wise Reliability Diagrams at the Final Task . 27   
F.2 Reliability Diagrams across Sequential Tasks 27

Table 7: Hyperparameters for Group Calibration Experiments
<table><tr><td rowspan="2">Hyperparameter</td><td colspan="4">ResNet-based</td><td colspan="2">ViT-based</td></tr><tr><td>CIFAR-10</td><td>CIFAR-100</td><td>Tiny-ImageNet</td><td>ImageNet</td><td>CIFAR-100</td><td>ImageNet-R</td></tr><tr><td colspan="7">Dataset Configuration</td></tr><tr><td>Dataset</td><td>cifar10</td><td>cifar100</td><td>tiny-imagenet</td><td>imagenet</td><td>cifar100</td><td>ImageNet-R</td></tr><tr><td>Number of Classes</td><td>10</td><td>100</td><td>200</td><td>1000</td><td>100</td><td>200</td></tr><tr><td>New Classes per Task</td><td>2</td><td>10</td><td>20</td><td>100</td><td>10</td><td>20</td></tr><tr><td>Total Tasks</td><td>5</td><td>10</td><td>10</td><td>10</td><td>10</td><td>10</td></tr><tr><td>Batch Size</td><td>128</td><td>128</td><td>128</td><td>256</td><td>128</td><td>128</td></tr><tr><td>Validation Size</td><td>500</td><td>500</td><td>100</td><td>5000</td><td>500</td><td>500</td></tr><tr><td>Replay Buffer Size</td><td>0.2K</td><td>2K</td><td>2K</td><td>5K/10K/20K</td><td>2K</td><td>2K</td></tr><tr><td colspan="7">Model Architecture</td></tr><tr><td>Model</td><td>ResNet-32</td><td>ResNet-32</td><td>ResNet-18</td><td>ResNet-18</td><td>ViT-B/16</td><td>ViT-B/16</td></tr><tr><td>Image Size</td><td>32</td><td>32</td><td>64</td><td>224</td><td>224</td><td>224</td></tr><tr><td colspan="7">Temperature Scaling</td></tr><tr><td>Calibration Epochs</td><td>100</td><td>100</td><td>100</td><td>100</td><td>50</td><td>50</td></tr><tr><td>Calibration Batch Size</td><td>128</td><td>128 0.1</td><td>128</td><td>128</td><td>128</td><td>128</td></tr><tr><td>Temp Learning Rate</td><td>0.1</td><td></td><td>0.1</td><td>0.1</td><td>0.03</td><td>0.03</td></tr><tr><td colspan="7">MT-CC</td></tr><tr><td>Number of Groups</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Group Data Ratio</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td></tr><tr><td colspan="7">Grouping Network</td></tr><tr><td>Optimizer</td><td>Adam</td><td>Adam</td><td>Adam</td><td>Adam</td><td>Adam</td><td>Adam</td></tr><tr><td>Learning Rate</td><td>0.01</td><td>0.01 50</td><td>0.01</td><td>0.01</td><td>0.001</td><td>0.001</td></tr><tr><td>Epochs</td><td>50</td><td></td><td>50</td><td>50</td><td>50</td><td>50</td></tr><tr><td colspan="7">Temperature Initialization</td></tr><tr><td>Init Values</td><td>[1.5, 1.5]</td><td>[1.5, 1.5]</td><td>[1.5, 1.5]</td><td>[1.5, 1.5]</td><td>[1.0, 1.0]</td><td>[1.0, 1.0]</td></tr><tr><td colspan="7">Groupwise DCA</td></tr><tr><td>GDCA Weight</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>0.1</td><td>0.1</td></tr></table>

## A Training Hyperparameters and Implementation Details

Implementation details. All calibration hyperparameters used for MT-CC and the corresponding baselines are summarized in Table 7. Calibration is performed post-hoc and independently of backbone training. For the underlying CIL methods, we follow their original training protocols unless otherwise specified. For ER, CIFAR-10 and CIFAR-100 are trained for 200 epochs with an initial learning rate of 0.1 and step decay at epochs 100 and 150, using momentum 0.9 and an $\ell _ { 2 }$ weight decay of $2 \times 1 0 ^ { - 4 }$ . Tiny-ImageNet follows the same optimizer and learning-rate schedule with a batch size of 256. Following the original DER protocol, DER additionally uses a 50-epoch decoupled stage with an initial learning rate of 0.05 and decay at epochs 15 and 30, together with a distillation temperature of 5.0. Its pretraining stage is run for 200 epochs with an initial learning rate of 0.1 and decay at epochs 60, 120, and 160. For the remaining CIL baselines, we follow the hyperparameters reported in their original publications. Before being provided to the grouping network, the logit margin is min-max normalized to ensure a comparable input scale with the other calibration features. To ensure positive temperature values and stable optimization, each group-specific temperature is parameterized in log-space. For each group $m \in \{ 1 , \ldots , M \}$ , we introduce an unconstrained parameter $\tau _ { m } \in \mathbb { R }$ and define $T _ { m } = \exp ( \tau _ { m } )$ . Accordingly, $\begin{array} { r } { { \frac { \partial { \mathcal { L } } } { \partial \tau _ { m } } } = { \frac { \partial { \mathcal { L } } } { \partial T _ { m } } } T _ { m } } \end{array}$ . At the first task, $\tau _ { m }$ is initialized from the predefined initial temperature, while at subsequent tasks it is initialized from the optimized temperature obtained at the previous task.

Grouping network input dimensionality. The grouping network takes the full softmax probability vector p as part of its input. To keep its input dimensionality fixed throughout continual learning, we represent p in the full class space and zero-pad the entries corresponding to unseen classes. Therefore, although the number of encountered classes increases over tasks, the dimensionality of the groupingnetwork input remains unchanged. Since the grouping network consists of only a lightweight fully connected layer, this fixed-dimensional representation introduces negligible additional overhead.

## A.1 Representative Replay-Based CIL Methods

We provide brief descriptions of the replay-based CIL methods evaluated in our experiments.

EEIL performs end-to-end incremental learning by jointly optimizing the classification loss and a knowledge distillation objective over replayed exemplars, mitigating feature drift across tasks.

iCaRL combines exemplar replay with knowledge distillation and a nearest-mean-of-exemplar classifier, preserving old knowledge and stable decision boundaries across tasks.

BiC introduces a lightweight bias-correction layer to compensate for logit shifts from old/new class imbalance, trained on a held-out validation set to reduce bias accumulation.

IL2M addresses score decay in old classes by leveraging stored classifier norms and historical prediction statistics to rescale logits post-hoc, reducing imbalance-induced degradation.

DER adopts a dynamically expandable architecture that freezes previously learned feature extractors and allocates new branches for incoming tasks, preserving prior knowledge by structural isolation rather than distillation alone. All methods follow their original protocols, with MT-CC applied as a post-hoc calibration module on top of each trained backbone.

## A.2 Details of Calibration Metrics

This section provides detailed definitions of the calibration metrics used in the main paper. We also introduce the multiclass Brier score, which evaluates probabilistic predictions without binning, and report the corresponding results later in Appendix B.5.

Expected Calibration Error (ECE) partitions predictions into $Q$ confidence bins and computes

$$
\mathrm { E C E } = \sum _ { i = 1 } ^ { Q } \frac { \left| { B } _ { i } \right| } { N } \left| { A } _ { i } - { C } _ { i } \right| ,
$$

where $| B _ { i } |$ is the number of samples in bin $i , N$ is the total number of samples, and $A _ { i }$ and $C _ { i }$ denote the average accuracy and confidence in bin i, respectively

Adaptive Expected Calibration Error (AECE) employs adaptive binning based on the cumulative confidence distribution within each test domain:

$$
\mathrm { A E C E } = \sum _ { i = 1 } ^ { Q } \frac { | \tilde { B } _ { i } | } { N } \left| A _ { i } - C _ { i } \right| ,
$$

where $| \tilde { B _ { i } } |$ denotes the number of samples in adaptive bin ¿.

Static Calibration Error (SCE) extends calibration evaluation to the multiclass setting:

$$
\mathrm { S C E } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \sum _ { b = 1 } ^ { Q } \frac { n _ { b k } } { N } \left| \operatorname { a c c } ( b , k ) - \operatorname { c o n f } ( b , k ) \right| ,
$$

where K is the number of classes, $n _ { b k }$ is the number of predictions in bin b for class $k ,$ and acc(b, k) and con $\mathrm { { f } } ( b , k )$ are the accuracy and confidence of that bin.

Brier Score measures the mean squared difference between the predicted probability distribution and the one-hot encoded ground-truth label:

$$
{ \mathrm { B r i e r } } = { \frac { 1 } { N } } \sum _ { n = 1 } ^ { N } \sum _ { k = 1 } ^ { K } \left( p _ { n , k } - \mathbf { 1 } [ y _ { n } = k ] \right) ^ { 2 } ,
$$

where $p _ { n , k }$ is the predicted probability of class k for sample $n ,$ and $\lfloor [ y _ { n } = k ]$ is one if the ground-truth label $y _ { n }$ is class k and zero otherwise. A lower Brier score indicates better probabilistic predictions.

## B Additional Ablation Studies

## B.1 Ablation on the Number of Groups

Table 8: Ablation on the number of groups M on CIFAR-100 over five seeds. $M = 1$ corresponds to single-temperature scaling (T-CIL).
<table><tr><td>M</td><td>ECE↓</td><td>AECE↓</td><td>Brier↓</td></tr><tr><td>1 (T-CIL)</td><td>6.03±0.42</td><td> $6 . 0 4 { \pm } 0 . 3 5 $ </td><td>0.570±0.005</td></tr><tr><td>2</td><td>4.49±0.53</td><td> $\mathbf { 4 . 4 4 { \overset { . } { \bot } } 0 . 4 9 }$ </td><td>0.556±0.006</td></tr><tr><td>3</td><td>4.51±0.47</td><td>4.48±0.45</td><td>0.558±0.007</td></tr><tr><td></td><td>4.50±0.41</td><td>4.50±0.42</td><td>0.559±0.007</td></tr><tr><td>45</td><td>4.79±0.40</td><td> $4 . 7 7 { \pm } 0 . 4 1$ </td><td> $0 . 5 6 1 { \scriptstyle \pm 0 . 0 0 7 }$ </td></tr></table>

We evaluate the sensitivity of MT-CC to the number of groups M by varying M from 1 to 5, where $M = 1$ corresponds to single-temperature scaling (T-CIL). As shown in Table 8, nearly all calibration improvement occurs from $M = 1$ to $M = 2$ (ECE: 6.03 → 4.49); $M = 3$ and $M = 4$ provide no meaningful additional gains across metrics, while $M = 5$ degrades performance. Thus, our choice of $M = 2$ is based on the observed saturation of performance rather than evaluating only two groups.

One possible reason is that increasing M introduces more group-specific temperatures to optimize, making the joint routing-temperature optimization more complex. At the same time, the calibration data are partitioned among more groups, leaving fewer samples to reliably estimate each temperature, particularly for sparsely populated groups. Consistent with this explanation, when $M > 2 .$ the learned routing effectively collapses to two dominant groups, leaving the remaining groups sparsely populated and their temperatures poorly estimated.

## B.2 Ablation on the Replay Memory Size

![](images/b64fd6563514bbd660a5a7b476e6ae14505ad2be441c7974491257ac80dba5da.jpg)

![](images/177bf5d443cffb174e3eb1d0eefccccedada256ff07b52a73919f6036ccb404d.jpg)  
(a) Replay buffer 3K ECE  
(b) Replay buffer 4K ECE  
Figure 6: Ablation study on replay buffer size on CIFAR-100. We report ECE for each task when using a 3K or 4K replay memory.

We vary the replay buffer size while keeping the same CIFAR-100 experimental setup as in the main paper. Across all buffer configurations, our method maintains consistently strong calibration performance. Moreover, MT-CC outperforms T-CIL on nearly all tasks, demonstrating its robustness to memory size changes.

## B.3 Comparison of ECE, AECE, SCE Across Sequential Tasks on CIFAR-100

![](images/9a913d1523897eed8f7972268f524ab8dd2cb053dcff5e57d6388eb324de2c4f.jpg)  
(a) ECE

![](images/e7861703f9d3cb16b61a048b4883ab2e6f9900fab4daa36f093d45dcb14eafb9.jpg)  
(b) AECE

![](images/1895ed9b938f0bbba199921212520d1f08116ff67c0612058db4336c90988e17.jpg)  
(c) SCE  
Figure 7: Comparison of calibration metrics across sequential tasks using CIFAR-100.  
Figure 7 presents a comparison of ECE, SCE, and AECE across sequential tasks on CIFAR-100. MT-CC consistently achieves lower calibration errors than T-CIL across all three metrics throughout the continual learning process.

## B.4 Results on CIFAR-100: 60-5 Split

![](images/0569002bcbc19c433a8deeb6c1e86162064b17ea3492dc36dbf2fbc7323b89a3.jpg)

<table><tr><td>Method</td><td>Acc. (↑)</td><td>ECE (↓)</td><td>AECE (↓)</td></tr><tr><td>Vanilla</td><td>54.84±0.41</td><td>32.45±0.35</td><td>32.45±0.36</td></tr><tr><td>TS</td><td>54.84±0.41</td><td>21.15±0.84</td><td>21.12±0.84</td></tr><tr><td>ETS</td><td>54.84±0.41</td><td>22.32±0.46</td><td>22.31±0.45</td></tr><tr><td>IRM</td><td>54.84±0.41</td><td>23.29±0.54</td><td>23.28±0.53</td></tr><tr><td>T-CIL</td><td>54.84±0.41</td><td>5.48±0.32</td><td>5.48±0.33</td></tr><tr><td>T-CIL + ETS T-CIL + IRM</td><td>54.84±0.41</td><td>5.89±0.34</td><td>5.87±0.38</td></tr><tr><td></td><td>54.84±0.41</td><td>6.98±0.69</td><td>7.00±0.66</td></tr><tr><td>MT-CC</td><td>54.84±0.41</td><td>4.79±0.31</td><td>4.80±0.30</td></tr></table>

Figure 8: Per-task accuracy and confidence on the final step of the CIFAR-100 60-5 split, illustrating the discrepancy between confidence and accuracy across tasks.  
Table 9: Calibration performance on CIFAR-100 under the 60-5 split. All values are averaged over all nine steps.

We further evaluate our method on the CIFAR-100 60-5 split, where the model is first trained on 60 base classes and then updated over eight incremental steps of five new classes each. This setting enlarges the initial task and induces a more imbalanced class distribution than the standard 10-task protocol, allowing us to verify that the proposed calibration framework remains effective under a different and more challenging CIL configuration. Except for the change in the number of classes per step, experimental settings follow the standard CIFAR-100 protocol.

Figure 8 compares per-task accuracy and confidence after the final incremental step. The results show pronounced discrepancies across tasks, where confidence remains consistently high while accuracy varies substantially, revealing stronger asymmetric miscalibration than in the standard 10-task configuration. This imbalance increases calibration difficulty because the model assigns similar confidence levels to tasks that differ widely in accuracy.

Despite this more challenging regime, MT-CC achieves the lowest ECE and AECE among all baselines in Table 9. MT-CC surpasses T-CIL as well as T-CIL combined with ETS or IRM. This shows that learned grouping and multi-temperature calibration generalize beyond the standard protocol and remain effective under enlarged and imbalanced task structures.

## B.5 Robustness to ECE Binning and Brier Score

Table 10: Calibration performance on CIFAR-10, CIFAR-100, and Tiny-ImageNet under different numbers of ECE bins and the multiclass Brier score. Lower values indicate better calibration. Results are averaged over five seeds.
<table><tr><td></td><td colspan="3">CIFAR-10</td><td colspan="3">CIFAR-100</td><td colspan="3">Tiny-ImageNet</td></tr><tr><td>Method</td><td>ECE@15↓</td><td>ECE@20↓</td><td>Brier ↓</td><td>ECE@15↓</td><td>ECE@20↓</td><td>Brier ↓</td><td>ECE@15↓</td><td>ECE@20↓</td><td>Brier ↓</td></tr><tr><td>T-CIL</td><td>15.36±1.39</td><td>15.37±1.40</td><td>0.498±0.019</td><td>6.19±0.39</td><td>6.27±0.38</td><td>0.570±0.005</td><td>19.04±0.42</td><td>19.13±0.41</td><td>0.883±0.004</td></tr><tr><td>T-CIL+ETS</td><td>14.01±1.31</td><td>14.05±1.30</td><td>0.497±0.019</td><td>5.80±0.45</td><td>5.87±0.44</td><td>0.571±0.004</td><td>22.21±0.21</td><td>22.25±0.21</td><td>0.901±0.003</td></tr><tr><td>T-CIL+IRM</td><td>14.90±0.95</td><td>14.83±1.11</td><td>0.506±0.019</td><td>6.88±0.66</td><td>6.97±0.67</td><td>0.571±0.006</td><td>18.08±0.42</td><td>18.05±0.51</td><td>0.884±0.005</td></tr><tr><td>T-CIL+AdaTS</td><td> $1 4 . 3 1 { \pm } 1 . 4 4$ </td><td> $1 4 . 3 4 { \pm } 1 . 4 4$ </td><td>0.501±0.023</td><td>7.76±0.50</td><td>7.86±0.51</td><td>0.586±0.004</td><td>21.75±1.25</td><td>21.80±1.28</td><td>0.905±0.012</td></tr><tr><td>MT-CC</td><td>10.22±1.51</td><td>10.27±1.48</td><td>0.447±0.015</td><td>4.62±0.51</td><td>4.72±0.54</td><td>0.556±0.006</td><td>15.87±0.37</td><td>15.93±0.36</td><td>0.839±0.003</td></tr></table>

To ensure that the observed calibration gains are not artifacts of specific metric configurations, we investigate the binning sensitivity of our evaluation. Specifically, we re-evaluate all methods on CIFAR-100 using 15 and 20 bins for ECE. Furthermore, to provide a strictly binning-free assessment, we incorporate the multiclass Brier score. We use the same trained models and calibrated probabilities across all evaluations, and report the mean and standard deviation over five seeds.

As shown in Table 10, MT-CC consistently achieves the lowest ECE across all three bin configurations, and the relative ranking of the methods remains unchanged as the number of bins increases. Crucially MT-CC also obtains the lowest multiclass Brier score, confirming its overall robustness.

## B.6 Applicability to Static Classification

Table 11: Comparison of single-temperature scaling and MT-CC in static and class-incremental learning settings on CIFAR-100 over five seeds.
<table><tr><td>Setting</td><td></td><td>Single-T ECE↓ MT-CC ECE ↓</td></tr><tr><td>Static joint training</td><td> $0 . 8 7 \pm 0 . 1 4$ </td><td> $0 . 8 7 \pm 0 . 1 4$ </td></tr><tr><td>CIL (task average)</td><td> $6 . 0 3 \pm 0 . 4 2$ </td><td> ${ \bf 4 . 4 9 \pm 0 . 5 3 }$ </td></tr></table>

Although groupwise temperature scaling is applicable to static classification, its benefit in MT-CC arises from modeling the asymmetric calibration behavior induced by continual learning. To verify that the improvement does not simply result from using multiple temperatures, we apply the same calibration pipeline to a jointly trained static CIFAR-100 model.

As shown in Table 11, MT-CC provides no additional improvement over single-temperature scaling in the static setting. The learned router effectively collapses to a single group. Specifically, the fraction of samples assigned to the smaller group is 0.000 ± 0.000, and the active-group temperature differs from the single-temperature solution by less than 0.003 in every seed. MT-CC therefore reduces to standard single-temperature scaling in this setting.

In contrast, under CIL, MT-CC reduces the task-averaged ECE from 6.03 to 4.49. This comparison confirms that MT-CC effectively targets the unique calibration shifts inherent to CIL, rather than merely benefiting from increased parameter capacity.

## B.7 Effect of Validation Set Size

Table 12: Calibration performance under varying validation set sizes (lower is better).
<table><tr><td>Validation Size</td><td>0</td><td>500</td><td>1000</td><td>1500</td><td>2000</td></tr><tr><td>T-CIL</td><td>16.13±0.19</td><td>6.19±0.66</td><td>6.38±0.47</td><td>6.24±0.39</td><td> $6 . 0 3 { \scriptstyle \pm 0 . 4 2 }$ </td></tr><tr><td>MT-CC</td><td>15.96±1.05</td><td>5.92±0.95</td><td>6.04±0.78</td><td>5.89±0.77</td><td>4.49±0.53</td></tr></table>

Table 12 reports the effect of validation set size. MT-CC consistently outperforms T-CIL, a state-of-theart method tailored to temperature scaling in class-incremental learning. Even under limited validation data regimes, MT-CC consistently outperforms T-CIL, demonstrating robustness to calibration data scarcity. Notably, even when the validation set size is 0, MT-CC achieves an ECE of 15.96 compared to 16.13 for T-CIL, relying solely on samples from the current task for grouping network training and temperature scaling.

## B.8 Inference Time Analysis

Table 13: The average per-sample inference time across all tasks on ImageNet over 3 seeds. Relative time is reported with TS as the baseline (1×). MT-CC requires negligible additional inference overhead compared to standard TS.

<table><tr><td>Method</td><td>Per-sample inference time (µs)</td><td>Relative</td></tr><tr><td>TS</td><td>1250.41</td><td>1.0000×</td></tr><tr><td>PTS</td><td>1250.65</td><td>1.0002×</td></tr><tr><td>AdaTS</td><td>1252.63</td><td>1.0018×</td></tr><tr><td>MT-CC</td><td>1250.80</td><td>1.0003×</td></tr></table>

Table 13 reports the average inference time across all tasks on ImageNet in microsecond scale, measured on a single NVIDIA RTX A6000 GPU. AdaTS incurs slightly higher inference cost than others due to the use of a class-conditional VAE. In contrast, MT-CC introduces negligible additional overhead compared to standard temperature scaling (TS). Since the grouping module is a lightweight network, the additional latency introduced by MT-CC is negligible in practice.

## B.9 Comparison with Feature-Matched and Grouped Calibration Methods

MT-CC outperforms both feature-matched instance-wise temperature scaling and unsupervised group-wise temperature scaling, showing that its improvement cannot be attributed solely to the CIL-aware inputs or the use of multiple temperatures. We evaluate progressively stronger baselines that isolate the contributions of the input features, heterogeneous calibration groups, and learned calibration-aware routing.

Table 14: Comparison with feature-matched and grouped calibration methods on CIFAR-100 over 5 seeds. The CIL-aware setting uses the same feature vector as MT-CC.
<table><tr><td>Method</td><td>Calibration / Routing</td><td>Feature</td><td>ECE↓</td><td>AECE↓</td><td>Brier ↓</td></tr><tr><td>PTS</td><td>Instance-wise TS</td><td>Top-10 logits</td><td> $9 . 8 7 \pm 0 . 7 7$ </td><td> $1 0 . 1 4 \pm 0 . 6 7$ </td><td> $0 . 5 9 6 \pm 0 . 0 0 8$ </td></tr><tr><td>PTS</td><td>Instance-wise TS</td><td>CIL-aware features</td><td> $6 . 8 8 \pm 0 . 9 1$ </td><td> $6 . 8 7 \pm 0 . 9 1$ </td><td> $0 . 5 7 5 \pm 0 . 0 0 8$ </td></tr><tr><td>AdaTS</td><td>Instance-wise TS</td><td>Penultimate feature</td><td> $7 . 6 7 \pm 0 . 5 2$ </td><td> $7 . 6 6 \pm 0 . 4 9$ </td><td> $0 . 5 8 6 \pm 0 . 0 0 4$ </td></tr><tr><td>AdaTS</td><td>Instance-wise TS</td><td>CIL-aware features</td><td> $6 . 5 5 \pm 0 . 7 4$ </td><td> $6 . 5 2 \pm 0 . 7 5$ </td><td> $0 . 5 7 4 \pm 0 . 0 0 4$ </td></tr><tr><td>GC+TS [45]</td><td>Semantic-aware learned grouping + group-wise TS</td><td>Penultimate feature</td><td> $7 . 2 3 \pm 0 . 2 5$ </td><td> $7 . 2 4 \pm 0 . 2 6$ </td><td> $0 . 5 7 6 \pm 0 . 0 0 5$ </td></tr><tr><td>K-means Group-TS</td><td>K-means  $( K = 2 ) \cdot$  + group-wise TS</td><td>CIL-aware features</td><td> $5 . 4 1 \pm 0 . 5 1$ </td><td> $5 . 3 7 \pm 0 . 5 1$ </td><td> $0 . 5 6 1 \pm 0 . 0 0 5$ </td></tr><tr><td>MT-CC</td><td>Learned routing + group-wise TS</td><td>CIL-aware features</td><td> ${ \bf 4 . 4 9 \pm 0 . 5 3 }$ </td><td>一  ${ \bf 4 . 4 4 \pm 0 . 4 9 }$ </td><td> $\mathbf { 0 . 5 5 6 \pm 0 . 0 0 6 }$ </td></tr></table>

Contribution of CIL-aware features. CIL-aware features provide more informative calibration signals under continual distribution shifts. We provide PTS and AdaTS with the same input vector used by MT-CC, $v ( x ) = [ \delta , p _ { \mathrm { o l d } } , p _ { \mathrm { n e w } } , p ]$ . Both methods improve over their original feature settings, confirming the effectiveness of the proposed input design.

Contribution of asymmetric calibration groups. Explicitly modeling asymmetric calibration groups improves upon instance-wise temperature prediction. K-means Group-TS clusters the same CILaware feature vectors into two groups, optimizes a separate temperature for each cluster, and assigns each test sample the temperature of its nearest centroid. It outperforms the feature-matched PTS and AdaTS variants, supporting the use of group-specific calibration.

Contribution of learned calibration-aware routing. Learning groups according to the calibration objective is more effective than clustering the feature space independently of calibration. MT-CC outperforms K-means Group-TS even though both use the same input features and two group-specific temperatures.

Comparison with existing grouping calibration. We further compare MT-CC with GC+TS [45], a prior semantic-aware grouping calibration method developed for static classification. GC+TS learns a grouping function from deep semantic features and applies a separate temperature to each learned group. The comparison shows that CIL-aware grouping is more effective for capturing calibration heterogeneity under continual task shifts.

## B.10 CIL Changes the Calibration Meaning of Logit Margin

The logit margin is strongly associated with the required calibration correction under CIL, whereas this relationship is nearly absent in static classification. For a jointly trained static CIFAR-100 model and the CIL model at the final task, we divide samples into 10 equal-count margin bins and independently fit the NLL-optimal temperature $T ^ { * }$ in each bin.

Table 15: NLL-optimal temperature $T ^ { * }$ across logit-margin bins on CIFAR-100 averaged over 5 seeds.
<table><tr><td>Setting</td><td>b0 (low)</td><td>b1</td><td>b2</td><td> $\mathbf { b } 3$ </td><td>b4</td><td>b5</td><td>b6</td><td>b7</td><td>b8</td><td>b9 (high)</td></tr><tr><td>Static</td><td>1.98</td><td>1.99</td><td>2.00</td><td>1.95</td><td>2.00</td><td>1.94</td><td>1.96</td><td>1.92</td><td>2.04</td><td>2.10</td></tr><tr><td>CIL</td><td>1.58</td><td>1.68</td><td>1.77</td><td>1.84</td><td>2.00</td><td>2.13</td><td>2.36</td><td>2.59</td><td>2.99</td><td>3.82</td></tr></table>

In the static model, $T ^ { * }$ remains nearly constant across margin bins even though accuracy increases with the margin. Under CIL, however, catastrophic forgetting degrades accuracy without a corresponding drop in prediction margin, so high-margin predictions become increasingly overconfident. Thus, the logit margin becomes a CIL-aware signal of the calibration correction required by different samples.

## B.11 Complementarity of Margin and Confidence

Confidence provides additional calibration information even after the logit margin is controlled. We divide samples into two equal-count margin regions and further split each region into low- and high-confidence halves. We independently fit $T ^ { * }$ to each subset and define $\Delta T ^ { * }$ as the difference between the optimal temperatures of the high- and low-confidence subsets.

Table 16: Optimal temperatures after controlling for the logit margin.
<table><tr><td>Setting</td><td>Margin region</td><td>Low-conf.  $T ^ { * }$ </td><td>High-conf.  $T ^ { * }$ </td><td> $\Delta \mathbf { T } ^ { * }$ </td></tr><tr><td rowspan="2">Static</td><td>Low</td><td>1.98</td><td>1.99</td><td>0.01</td></tr><tr><td>High</td><td>1.94</td><td>2.03</td><td>0.09</td></tr><tr><td rowspan="2">CIL</td><td>Low</td><td>1.60</td><td>1.95</td><td>0.34</td></tr><tr><td>High</td><td>2.26</td><td>3.32</td><td>1.05</td></tr></table>

Confidence produces only small temperature differences in the static model but substantially larger differences under CIL. This result shows that margin and confidence capture complementary aspects of CIL-induced calibration heterogeneity.

Table 17: Benefit of explicitly providing the old/new class structure on CIFAR-100 across different grouping-set sizes. The optimization budget is matched across data sizes and $S _ { \mathrm { c a l } }$ is fixed. Results report ECE (%) and Brier score over five seeds.
<table><tr><td rowspan="2"> $| S _ { \mathrm { g r p } } |$ </td><td colspan="2"> $[ \delta , p _ { \mathrm { o l d } } , p _ { \mathrm { n e w } } , p ]$ </td><td colspan="2"> $[ \delta , p ]$ </td><td colspan="2"> $[ \delta , p _ { \mathrm { r a n d 1 } } , p _ { \mathrm { r a n d 2 } } , p ]$ </td><td colspan="2"> $[ \delta , p _ { \mathrm { o l d } } , p _ { \mathrm { n e w } } ]$ </td></tr><tr><td>ECE</td><td>Brier</td><td>ECE</td><td>Brier</td><td>ECE</td><td>Brier</td><td>ECE</td><td>Brier</td></tr><tr><td>50</td><td>4.77±0.43</td><td>0.565±0.009</td><td> $4 . 9 2 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $0 . 5 7 3 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $5 . 0 0 { \scriptstyle \pm 0 . 4 3 }$ </td><td>0.573±0.006</td><td>5.98±0.48</td><td>0.572±0.004</td></tr><tr><td>100</td><td> $\mathbf { 4 . 3 6 { \scriptstyle \pm 0 . 4 6 } }$ </td><td>0.560±0.007</td><td> $4 . 6 9 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $0 . 5 7 1 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $4 . 8 1 { \pm } 0 . 4 5$ </td><td>0.571±0.006</td><td>5.78±0.30</td><td>0.571±0.005</td></tr><tr><td>250</td><td>4.49±0.45</td><td>0.558±0.006</td><td> $4 . 7 0 { \scriptstyle \pm 0 . 2 7 }$ </td><td> $0 . 5 7 2 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $4 . 7 3 { \scriptstyle \pm 0 . 3 4 }$ </td><td>0.572±0.006</td><td>5.74±0.33</td><td>0.570±0.005</td></tr><tr><td>500</td><td>4.54±0.56</td><td>0.557±0.006</td><td> $4 . 7 2 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $0 . 5 7 1 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $4 . 7 1 { \scriptstyle \pm 0 . 3 7 }$ </td><td>0.571±0.006</td><td>5.83±0.30</td><td>0.571±0.005</td></tr><tr><td>1000</td><td> $\mathbf { 4 . 4 9 \pm 0 . 5 3 }$ </td><td> $\mathbf { 0 . 5 5 6 { \scriptstyle \pm 0 . 0 0 6 } }$ </td><td> $4 . 6 3 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $0 . 5 7 1 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td>4.59±0.40</td><td>0.571±0.007</td><td>5.82±0.31</td><td>0.571±0.005</td></tr></table>

## B.12 Benefit of Explicit Old/New Class Structure

Although $p _ { \mathrm { o l d } }$ and $p _ { \mathrm { n e w } }$ are linear functions of the full softmax vector $p$ and therefore do not increase the expressiveness of the linear grouping network, explicitly providing them exposes the old-versusnew class structure directly to the router. We investigate whether explicitly exposing this structure benefits calibration despite the same information already being contained in $p .$

Input variants. To isolate the effect of explicitly providing the old/new class structure, we compare the following input configurations:

$[ \delta , p _ { \mathrm { o l d } } , p _ { \mathrm { n e w } } , p ]$ : The full MT-CC input, which combines the logit margin, explicit old/new probability masses, and the full softmax vector.

$[ \delta , p ]$ : Removes the explicit old/new probability masses while retaining the logit margin and full softmax vector.

$[ \delta , p _ { \mathrm { r a n d 1 } } , p _ { \mathrm { r a n d 2 } } , p ]$ : Replaces the old/new probability masses with probability masses computed from a size-matched random partition of the classes. Since these random class-mass features are also linear functions of $p ,$ this variant controls for the effect of simply adding redundant aggregate features.

$[ \delta , p _ { \mathrm { o l d } } , p _ { \mathrm { n e w } } ]$ : Retains the logit margin and explicit old/new probability masses while removing the full softmax vector.

Experimental setup. To verify that the benefit of the explicit old/new structure is not specific to a particular grouping-data regime, we vary the number of distinct samples used to train the grouping network, $| \bar { S } _ { \mathrm { g r p } } | \in \{ 5 0 , \bar { 1 } 0 0 , 2 5 0 , 5 0 0 , 1 0 0 0 \}$ , while keeping the temperature-fitting subset $S _ { \mathrm { c a l } }$ fixed. The same grouping samples are used across input variants at each data size. We match the optimization budget across grouping-set sizes so that each setting receives approximately the same number of gradient updates, isolating the effect of the number of distinct grouping samples. All other hyperparameters are kept identical.

Benefit of explicit old/new structure. As shown in Table 17, explicitly providing the old/new class structure consistently improves both ECE and Brier score over using only $[ \delta , p ]$ across grouping-data budgets. This indicates that the grouping network learns more effective calibration routing when the CIL-specific old/new structure is directly exposed.

CIL-specificity of the improvement. Replacing the true old/new probability masses with sizematched random class-mass summaries does not reproduce the same improvement and performs similarly to $[ \delta , p ]$ . Thus, the gain cannot be explained merely by adding redundant linear aggregate features. Instead, it depends on explicitly exposing the old-versus-new structure aligned with continual learning.

Complementarity with the full softmax vector. Using only $[ \delta , p _ { \mathrm { o l d } } , p _ { \mathrm { n e w } } ]$ yields substantially higher ECE, showing that the old/new summaries do not replace the fine-grained class-wise information in $p .$ Instead, they provide complementary CIL-aware structure.

Sample efficiency. The full MT-CC input achieves an ECE of 4.36% with only 100 grouping samples comparable to 4.49% with 1,000 samples. This indicates that effective routing can be learned from a relatively small number of distinct grouping samples when the optimization budget is controlled.

## B.13 Continual Update and Stability of the Grouping Network

We continually update the grouping network so that it can adapt to the evolving classifier across tasks. Continually updating the grouping network improves calibration while preserving the routing structure learned across tasks. We validate this design choice by comparing alternative update strategies and measuring group-assignment consistency across adjacent tasks.

Table 18: Average calibration performance of different grouping-network update strategies on CIFAR-100 over five seeds.
<table><tr><td>Grouping-network strategy</td><td>ECE↓</td><td>AECE↓</td><td> $\mathrm { S C E } \left( \times 1 0 ^ { - 3 } \right) \downarrow$ </td><td>Brier ↓</td></tr><tr><td>MT-CC</td><td> ${ \bf 4 . 4 9 \pm 0 . 5 3 }$ </td><td> ${ \bf 4 . 4 4 \pm 0 . 4 9 }$ </td><td>—  ${ \bf 5 . 1 2 \pm 0 . 1 1 }$ </td><td> $\mathbf { 0 . 5 5 6 \pm 0 . 0 0 6 }$ </td></tr><tr><td>Lag-1</td><td> $4 . 8 2 \pm 0 . 3 6$ </td><td> $4 . 8 2 \pm 0 . 3 6$ </td><td> $5 . 3 0 \pm 0 . 1 2$ </td><td> $0 . 5 5 8 \pm 0 . 0 0 6$ </td></tr><tr><td>Fresh each task</td><td> $4 . 9 8 \pm 0 . 3 5$ </td><td> $4 . 9 6 \pm 0 . 3 4$ </td><td> $5 . 3 8 \pm 0 . 1 4$ </td><td> $0 . 5 6 1 \pm 0 . 0 0 7$ </td></tr></table>

Effectiveness of continual router updates. Updating the previous router is more effective than either freezing it or discarding its previously learned routing structure. We compare the following three grouping-network update strategies on CIFAR-100:

• MT-CC : The router learned at task t — 1 is carried forward and updated using the calibration data available at task t.

• Lag-1 : The router learned at task t — 1 is used without further updates to calibrate the model at task t, while only the group temperatures are re-fitted.

• Fresh each task : The router is reinitialized and trained independently at every task.

As shown in Table 18, continually updating the router outperforms both freezing and reinitializing it across all calibration metrics. This validates the MT-CC design of retaining previously learned routing information while adapting it to the evolving classifier.

Stability of continual router updates. Continually updating the router causes little routing drift. When the classifier outputs are held fixed, both groups retain more than 90% of their assignments across adjacent tasks. We measure group retention over all adjacent task transitions and five seeds. Groups are ordered by their learned temperatures, with G1 denoting the lower-temperature group and G2 the higher-temperature group. For each group, retention is defined as the fraction of samples assigned to that group by the task t — 1 router that remain in the same group under the task t router

We evaluate retention under two conditions. In the actual transition, both the classifier and router change from task t — 1 to task t. In the router-only condition, the task t classifier outputs are held fixed, and the task t — 1 and task t routers are evaluated using the same outputs.

Table 19: Group retention across adjacent tasks on CIFAR-100, averaged over five seeds. G1 and G2 denote the lower- and higher-temperature groups, respectively.
<table><tr><td>Condition</td><td></td><td>G1 retention (%) ↑ G2 retention (%) ↑</td></tr><tr><td>Classifier and router change</td><td> $7 1 . 8 \pm 7 . 5$ </td><td> $4 2 . 0 \pm 1 2 . 4$ </td></tr><tr><td>Router change only</td><td> $9 4 . 3 \pm 6 . 3$ </td><td> $9 0 . 3 \pm 9 . 6 $ </td></tr></table>

As shown in Table 19, the task t — 1 and task t routers produce highly consistent assignments when evaluated on the same classifier outputs. The lower G2 retention in the actual transition is expected because the introduction of new classes substantially changes the composition of the high-temperature group.

## C How GDCA Stabilizes Routing-Temperature Optimization

We investigate how GDCA stabilizes the coupled routing-temperature optimization. By complementing the sample-level NLL with groupwise confidence-accuracy alignment, GDCA helps the joint optimizer distinguish routing solutions with similar likelihood, guides the estimation of group-specific temperatures, and mitigates routing collapse into a single dominant group. Together, these effects explain the calibration improvement reported in the main paper. We first show that NLL alone is insufficient for calibration-relevant routing, then explain the optimization mechanism of GDCA, and finally analyze routing degeneracy with and without GDCA.

![](images/82f5f661a23b6441cc894e1ddd97af0169723e4c50a1ae1f4ce80965f93abfd8.jpg)  
Figure 9: Profile-likelihood analysis of perturbed routing solutions at the final task on CIFAR-100 over five seeds. The group temperatures are re-optimized from scratch for each routing perturbation. For each seed, profiled ∆NLL is measured relative to the minimum profiled NLL across all perturbations, while relative ∆ECE is measured with respect to the learned routing. The shaded region indicates routing solutions whose profiled NLL lies within 0.1% of the seed-wise minimum.

NLL alone is insufficient for calibration-relevant routing. Optimizing the routing with NLL alone may not distinguish routing solutions with different calibration quality. Even after re-optimizing the temperatures for each routing configuration, substantially different routing solutions can achieve nearly identical NLL values. To examine this limitation, we perturb the learned routing and reoptimize the temperatures from scratch at each perturbation point.

As shown in Figure 9, some routing solutions within the near-optimal NLL band produce a relative increase of more than 15% in test ECE compared with the learned routing. Thus, NLL can be nearly indifferent among routing solutions with materially different calibration quality, making it insufficient as the sole objective for learning calibration-relevant routing.

Optimization mechanism. Each group $G _ { m }$ is constructed using a hard routing assignment, and thus gradients do not pass through the group membership or the empirical accuracy $\operatorname { A c c } ( G _ { m } )$ . Instead, they are propagated through $\operatorname { C o n f } ( G _ { m } )$ , which is computed from the calibrated mixture confidence using the soft routing weights. Consequently, GDCA directly adjusts the group temperatures toward groupwise confidence-accuracy alignment. Once the temperatures become differentiated, the mixture confidence also becomes sensitive to the routing weights, allowing GDCA to update the router indirectly through the mixture confidence while jointly optimizing the temperatures.

Table 20: Routing-group sizes with and without GDCA over five seeds. The minority-group fraction denotes the proportion of test samples assigned to the less populated of the two groups.
<table><tr><td>Routing behavior</td><td>w/o GDCA</td><td>with GDCA</td></tr><tr><td>Frequency of minority-group fraction below 20%</td><td>20.0%</td><td>4.4%</td></tr><tr><td>Mean minority-group fraction</td><td>29.7%</td><td>37.5%</td></tr></table>

Routing degeneracy with and without GDCA. During training, the router may assign most samples to one dominant group, causing the other group to become sparsely populated. As shown in Table 20 GDCA reduces the frequency with which the minority-group fraction falls below 20% from 20.0% to 4.4%, while increasing its mean fraction from 29.7% to 37.5%. These results indicate that GDCA mitigates routing degeneracy and maintains sufficiently populated groups for more reliable groupspecific temperature scaling.

## D Catastrophic Forgetting and Asymmetric Calibration Behaviors

Table 21: Average first-task performance degradation over 5 runs under the CIL setup.
<table><tr><td rowspan="2">Dataset</td><td colspan="2">After First Task</td><td colspan="2">After Middle Task</td><td colspan="2">After Final Task</td></tr><tr><td>Acc (%)</td><td>Conf (%)</td><td>Acc (%)</td><td>Conf (%)</td><td>Acc (%)</td><td>Conf (%)</td></tr><tr><td>CIFAR-10</td><td>98.79</td><td>96.06</td><td>72.69</td><td>92.09</td><td>20.37</td><td>91.38</td></tr><tr><td>CIFAR-100</td><td>85.62</td><td>92.60</td><td>42.38</td><td>79.14</td><td>30.84</td><td>75.08</td></tr><tr><td>Tiny-ImageNet</td><td>70.60</td><td>88.47</td><td>9.10</td><td>68.69</td><td>4.70</td><td>69.82</td></tr></table>

![](images/2d9f90d646c5225a65d8bf060bea4df2886f61a0f65d8be880fad535c263b9f5.jpg)  
(a) Per-task accuracy across incremental stages. Accuracy degradation follows different trajectories across tasks, indicating asymmetric forgetting dynamics.

![](images/18af155933c8680488eb8fda2c40a259a2b1bd750cf4380af588d943e790329c.jpg)  
(b) Per-task confidence-accuracy gaps. The gaps generally widen at different rates across tasks, revealing asymmetric calibration behaviors.

Figure 10: Catastrophic forgetting and asymmetric calibration behaviors on CIFAR-100. Each curve tracks a task from its introduction through subsequent incremental stages.

Table 21 shows that catastrophic forgetting is accompanied by a growing mismatch between confidence and accuracy. For the first task of CIFAR-10, accuracy decreases from 98.79% to 20.37%, whereas confidence decreases only from 96.06% to 91.38%. Consequently, the confidence-accuracy gap reaches 71.01 percentage points after the final task. CIFAR-100 and Tiny-ImageNet exhibit the same pattern: accuracy declines substantially more than confidence. These results show how forgetting can contribute to increasing overconfidence when confidence does not decrease in proportion to the loss of predictive accuracy.

Figure 10a further shows that the trajectories of accuracy degradation differ across tasks. Figure 10b complements this analysis by directly tracking the resulting confidence-accuracy gaps for all tasks on CIFAR-100. The gaps generally increase after subsequent tasks are introduced, but the increase is neither uniform nor strictly monotonic: different tasks exhibit distinct rates of increase and trajectories. Thus, the mismatch between confidence and accuracy evolves differently across tasks under continual updates, revealing asymmetric calibration behaviors.

Taken together, these results connect catastrophic forgetting with asymmetric calibration: predictive accuracy deteriorates without a proportional decrease in confidence, and the resulting gap evolves differently across tasks. This motivates the use of multiple temperatures in MT-CC to accommodate heterogeneous calibration needs.

## E Training and Inference Procedure of MT-CC

Algorithm 1 Training Procedure of MT-CC   
Input: Task sequence $\overline { { \{ \mathcal { D } ^ { 1 } , \ldots , \mathcal { D } ^ { B } \} } }$ ; calibration sets $\{ { \mathcal { C } } ^ { 1 } , \ldots , { \mathcal { C } } ^ { B } \} ;$ number of groups M   
Output: Grouping network $g _ { \phi }$ and group temperatures $\{ T _ { m } ^ { * } \} _ { m = } ^ { M }$   
1: Initialize grouping network $g _ { \phi }$ and group temperatures $\{ T _ { m } ^ { 0 } \} _ { m = 1 } ^ { \bar { M } }$   
2: for $b = 1$ to B do   
3: Split ${ \mathcal { C } } ^ { b }$ into two disjoint subsets $\mathcal { C } ^ { b } \to ( S _ { \mathrm { g r p } } ^ { b } , S _ { \mathrm { c a l } } ^ { b } )$   
4: Initialize $\{ T _ { m } ^ { b } \} _ { m = 1 } ^ { M }  \{ T _ { m } ^ { b - 1 } \} _ { m = 1 } ^ { M }$   
5: For each x $\in S _ { \mathrm { g r p } } ^ { b } ,$ compute $v ( x ) { \overset { } { = } } { \overset { \mid } { [ } } \delta ,$ Pold, Pnew, P]   
6: Update $g _ { \phi }$ and $\{ T _ { m } ^ { b } \} _ { m = 1 } ^ { M }$ by minimizing $\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { M T - C C } } + \lambda \mathcal { L } _ { \mathrm { G D C A } }$   
7: Assign each x ∈ $S _ { \mathrm { c a l } } ^ { b }$ to m\*(x) = arg maxm $g _ { \phi } ( x ) _ { r }$ n   
8: For each group m, refine $T _ { m } ^ { b }$ on its assigned samples in $S _ { \mathrm { c a l } } ^ { b }$ by minimizing NLL   
9: end for   
10: return $g _ { \phi }$ and $\{ T _ { m } ^ { B } \} _ { m = 1 } ^ { M }$

Training. Algorithm 1 summarizes MT-CC training. For each task, calibration data are split into grouping and calibration subsets. We jointly optimize the grouping network and group temperatures using ${ \mathcal { L } } _ { \mathrm { t o t a l } } .$ then freeze $g _ { \phi }$ and refine each group temperature with NLL. The resulting temperatures initialize the next task.

Algorithm 2 Inference Procedure of MT-CC   
Input: Grouping network $g _ { \phi } ,$ optimized group temperatures $\{ T _ { m } ^ { * } \} _ { m = 1 } ^ { M }$ , test sample x and global temperature $T _ { \mathrm { g l o b } } ^ { * }$   
Output: Calibrated probability vector $p _ { \mathrm { c a l } } ( x )$   
1: if b = 1 then   
2: pcal(x) ← softmax $\left( \frac { f _ { \theta } ( x ) } { T _ { \mathrm { g l o b } } ^ { * } } \right)$   
3: else   
4: $g ^ { * } ( x ) \gets \arg \operatorname* { m a x } _ { m } g _ { \phi } ( x ) _ { m } $   
5: pcal(x) ← softmax $\left( \frac { f _ { \theta } ( x ) } { T _ { g ^ { * } ( x ) } ^ { * } } \right)$   
6: end if

Inference. Algorithm 2 summarizes the inference procedure of MT-CC. For the first task, the logits $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ are calibrated using a single optimized global temperature $T _ { \mathrm { g l o b } } ^ { \ast }$ through standard temperature scaling. For subsequent tasks, the grouping network $g _ { \phi }$ assigns each test sample to the group with the highest predicted probability. The calibrated confidence is then obtained by applying the corresponding optimized group-specific temperature to the logits before the softmax operation.

## F Reliability Diagram

![](images/3c1e197ddaf738ff7d133534d2ffdce011cdc85d83d5d3ca95c25591da3f98a9.jpg)  
(a) Single T (group 1)

![](images/25a7f910ebb63c801d4514c81c0377d7d023412ad67704130ad3cd86d0ec4cc4.jpg)  
(b) Single T (group 2)

![](images/aaa8981637d40be6baa6e84f4f6cd2c47e1273fe6a198c02525fff062377a7c0.jpg)  
(c) Single T (overall)

![](images/76a4200efbe8edc3fdeed7784a4464c6dc5a311767ae99ef25f8b16cf0566803.jpg)  
(d) MT-CC (group 1)

![](images/6cfe23518101e854859e2a3d1aa66e1a93e832639a9523a765efe1bac4ce93db.jpg)  
(e) MT-CC (group 2)

![](images/8a9a672696677771555611e21acd2376eb6ab1e3383ba3109e4a6e38dd82ed67.jpg)  
(f) MT-CC (overall)  
Figure 11: (a), (b), (c): Reliability diagrams of T-CIL [16] using a single temperature T. (d), (e), (f): Reliability diagrams of the proposed MT-CC with optimized grouping. Results are shown after training on the 10th task of CIFAR-100, comparing group 1, group 2, and all classes as determined by the proposed grouping function.

## F.1 Group-wise Reliability Diagrams at the Final Task

Figure 11 compares the reliability diagrams of T-CIL and MT-CC after training on the 10th (final) task of CIFAR-100, separately for group 1, group 2, and the overall distribution determined by the proposed grouping network. With a single global temperature, T-CIL exhibits clearly distinct miscalibration patterns between the two groups: one group remains overconfident while the other becomes underconfident, so a single T cannot satisfy both simultaneously, leaving residual gaps in the overall reliability diagram. In contrast, MT-CC assigns a group-specific temperature to each subset of samples, substantially reducing the gap between confidence and accuracy within both groups and yielding a tighter alignment with the diagonal in the overall reliability diagram. This visually confirms our central claim that addressing the asymmetric calibration behaviors of CIL with multiple temperatures leads to more effective calibration than the single-temperature strategy.

## F.2 Reliability Diagrams across Sequential Tasks

Figure 12 presents the reliability diagrams for CIFAR-100 from Task 2 to Task 10, comparing T-CIL and MT-CC across Group 1, Group 2, and the overall distribution. The diagrams illustrate how closely the model's predicted confidence aligns with the empirical accuracy, where curves closer to the diagonal indicate better calibration. Across most tasks, MT-CC exhibits consistently stronger overall calibration than T-CIL, confirming its advantage.

![](images/baa363048f0d2f64f30cdd61f22af296201129d65134f6a8a4dba63d01de6520.jpg)  
Figure 12: Reliability diagrams for CIFAR-100 comparing T-CIL and MT-CC across Tasks 2–10. Overall calibration performance shows consistently strong reliability across most tasks.