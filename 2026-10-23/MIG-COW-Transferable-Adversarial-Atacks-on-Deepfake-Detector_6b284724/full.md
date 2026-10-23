# MIG-COW: Transferable Adversarial Atacks on Deepfake Detectors via Gradient Decomposition

Wonjune Seo Ulsan National Institute of Science and Technology Ulsan, Republic of Korea wonjuneseo@unist.ac.kr

Yeseong Jung Ulsan National Institute of Science and Technology Ulsan, Republic of Korea y2ahsong@unist.ac.kr

Joonhyuk Baek<sup>∗</sup> Ulsan National Institute of Science and Technology Ulsan, Republic of Korea jjhk7330@unist.ac.kr

Saerom Park<sup>†</sup> Ulsan National Institute of Science and Technology Ulsan, Republic of Korea srompark@unist.ac.kr

## Abstract

Despite recent advances, deepfake detectors remain vulnerable to adversarial examples, particularly in diverse, real-world settings. We propose MIG-COW, a novel adversarial attack framework that generates highly generalizable and visually imperceptible adversarial examples. By combining momentum-integrated gradients with a consensus-orthogonal decomposition, MIG-COW captures both shared and model-specific vulnerabilities across heterogeneous CNN and ViT detectors. On the AADD-2025 Challenge benchmarks, MIG-COW achieves a 99.96% white-box attack success rate (ASR) with high perceptual similarity (SSIM), significantly outperforming existing baselines. However, its limited 7.16% ASR against oficial black-box targets—despite achieving the best overall score—highlights the ongoing challenge of transferability. We also demonstrate that incorporating low-performing but diverse models in the ensemble can degrade attack efectiveness, underscoring the need for careful surrogate model selection in real-world adversarial settings.

## CCS Concepts

• Computing methodologies → Computer vision; • Security and privacy → Social aspects of security and privacy; • Ap plied computing → Computer forensics.

## Keywords

Generative Artificial Intelligence, Deepfake Detection, Adversarial Attack

## ACM Reference Format:

Wonjune Seo, Joonhyuk Baek, Yeseong Jung, and Saerom Park. 2025. MIG-COW: Transferable Adversarial Attacks on Deepfake Detectors via Gradient Decomposition. In Proceedings ofthe 33rd ACM International Conference on Multimedia (MM ’25), October 27–31, 2025, Dublin, Ireland. ACM, New York, NY, USA, 7 pages. https://doi.org/10.1145/3746027.3761986

## 1 Introduction

Deepfake detection models have been widely integrated into modern security frameworks, often demonstrating impressive accuracy and robustness under conventional operating conditions. Nevertheless, a critical question remains: how resilient are these systems in the presence of sophisticated adversarial threats? If a deepfake image—once confidently identified as “face”—can be subtly manipulated to evade detection and be misclassified as “real,” it reveals a fundamental vulnerability in current detection systems. This highlights a significant challenge that current detection technologies have not yet fully addressed.

Most state-of-the-art deepfake detectors can be broadly categorized into two architectural paradigms: convolutional neural network (CNN)-based models, which capture local texture and artifact patterns, and vision transformer (ViT)-based models, which use self-attention to model global context. Despite their success, both remain susceptible to adversarial attacks. In particular, CNNs possess strong local structural inductive biases, whereas ViTs are intentionally designed to reduce such biases, relying more on data-driven learning of relationships. Consequently, adversarial perturbations often transfer poorly between these two distinct architectures [2, 9].

This observation motivates a more fundamental question: Can we generate a single adversarial example capable of simultaneously deceiving multiple detection architectures? Addressing this question requires uncovering the underlying vulnerabilities that are common across diverse models while minimizing reliance on architecture-specific components such as direct gradient with respect to input. To address it, we propose a novel ensemble attack framework, MIG-COW (Momentum-Integrated Gradient with Consensus-Orthogonal Weighting), designed to enhance the generalizability of adversarial examples in deepfake detection without compromising white-box attack performance. By systematically identifying and disentangling both shared and distinct decision patterns, we can more efectively evaluate the true resilience of both existing and unknown detectors under realistic adversarial conditions.

In this work, we make the following key contributions:

• We propose MIG-COW, a novel ensemble adversarial attack framework that generates highly generalizable and visually imperceptible adversarial examples by efectively exploiting both shared and model-specific vulnerabilities across CNN and ViT-based deepfake detectors.

• We conduct extensive experiments across diverse white-box and black-box detectors, comparing against existing adversarial attack methods as baselines. Our proposed method achieves the highest overall score—balancing both attack success and visual imperceptibility—across two white-box and two black-box models.

• We provide novel insights into transfer-based attacks, revealing that incorporating low-performing source models can degrade overall attack efectiveness despite architectural diversity, underscoring the critical importance of functional equivalence for robust adversarial transfer.

## 2 Related work

Recent research on deepfake detection can be broadly categorized into CNN-based and ViT-based approaches. CNN-based detectors focus on local textures and manipulation artifacts, with models like Xception [23] and EficientNet-B7 [24] performing well on datasets such as FaceForensics++ and DFDC. In contrast, ViT-based detectors capture global dependencies using self-attention. The ICT[5] enforces identity consistency across video frames, while UIA-ViT[28] detects intra-frame forgery signals in an unsupervised manner.

Despite the progress made by both architectural paradigms, significant challenges remain. Their inherent structural diferences that exhibit diferent inductive biases lead to varying vulnerabilities to adversarial attacks and limited generalizability to unseen manipulations or domain shifts.

Gradient-based adversarial attacks are widely used to test model robustness. FGSM [8] and PGD [20] serve as baseline white-box attacks. To enhance transferability, MI-FGSM [6] leverages momemtumbased updates that stabilizes gradient updates and achieves higher success rates against unseen models.

Another direction for enhancing transferability involves using attribution-based gradients, which ofer more semantically meaningful perturbation directions. MIG [19] introduced integrated gradients with momentum to produce more architecture-invariant perturbations, efectively targeting both CNNs and ViTs. More recently, MuMoDIG [22] refined the integration path and demonstrated that using a log-based loss instead of a raw output loss further enhances the efectiveness of the attack.

Nevertheless, few studies have explicitly investigated the performance of these general-purpose gradient-based attacks on deepfake detection models. Unlike standard classifiers, deepfake detectors often depend on subtle cues such as fine-grained textures and identity consistency, making them qualitatively diferent. This gap highlights the need for tailored attack strategies that account for the heterogeneous nature of deepfake detection systems and can expose shared vulnerabilities without sacrificing attack efectiveness on individual models across diverse architectures.

## 3 Methodology

## 3.1 Overview

We propose an attack framework called Momentum Integrated Gradient with Consensus-Orthogonal Weighting (MIG-COW)

to generate adversarial examples with enhanced generalizability across a wide range of deepfake detection models, including both CNNs and ViT. An overview ofthe proposed MIG-COW framework is illustrated in Figure 1.

Our method leverages two key properties to promote the generalization of adversarial perturbations from source models (known detection models) to target models (unseen detection models).

• Implementation invariance [19, 22, 26]: We adopt Integrated Gradients (IG), which are invariant to the model’s implementation details and depend only on the input-output mapping. Since adversarial transferability is more likely when source and target models share similar functional behavior, leveraging IG improves attack generalization across architectures.

• Gradient Ensemble [19]: As highlighted by [19], ensembling gradients from multiple models or inputs is known to enhance the robustness and generalizability of adversarial attacks. Specifically, combining IG or logit gradients yields better results than directly aggregating raw perturbations.

A critical challenge in gradient ensembling—especially in achieving cross-model generalizability—is how to aggregate gradients in a way that preserves attack efectiveness on source models. To address this, we propose a novel aggregation strategy that combines consensus gradient directions with complementary architecturespecific information. The resulting MIG-COW framework consists of two main components: (1) momentum-integrated gradients and (2) gradient aggregation via orthogonal decomposition.

## 3.2 Proposed Method

3.2.1 Integrated Gradient (IG). In our proposed method, we compute IG for each source model with respect to the target label (i.e., “real” class in deepfake detection) to identify which input features contribute most to the prediction. IG computes the path-integrated gradient from a black baseline image to the input, producing stable and interpretable attributions across interpolation steps. This approach is beneficial for both CNNs and ViTs due to its robustness to model-specific variations. Figure 2 shows the IG attribution maps of individual models, MIG, and our proposed method (MIG-COW). By comparing the attribute maps, we can see how the proposed method improves transferability. In particular, our method efectively captures the core areas common to diferent models and the unique contributions of each architecture.

For a given input $x \in X ,$ baseline � ∈ X (i.e., a black image), and source model $f : X \to [ 0 , 1 ]$ , the IGs for the �-th input dimension are defined as:

$$
\mathbf { g } _ { i } = I G _ { i } ( \boldsymbol { f } , \boldsymbol { x } , \boldsymbol { b } ) = ( x _ { i } - b _ { i } ) \times \int _ { \xi = 0 } ^ { 1 } \frac { \partial \boldsymbol { f } \big ( \boldsymbol { b } + \xi ( \boldsymbol { x } - \boldsymbol { b } ) \big ) } { \partial x _ { i } } d \xi .\tag{1}
$$

In practice, this path integral is approximated by a Riemann sum with � discrete steps:

$$
I G _ { i } ( f , x , b ) \approx ( x _ { i } - b _ { i } ) \times \sum _ { k = 1 } ^ { s } \frac { \partial f \big ( b + \frac { k } { s } ( x - b ) \big ) } { \partial x _ { i } } \times \frac { 1 } { s } .\tag{2}
$$

We also employ momentum-integrated gradients (MIG) [19] to stabilize the optimization process and help escape undesirable local optima by updating the accumulated gradient with a momentum term at each iteration. To enhance generalizability and gradient scaling, inspired by Ren et al. [22], we define the source model $f ( x )$ in Eq. 1 and Eq. 2 as the cross-entropy loss computed directly on the raw logits for the target class, rather than using the raw probability. This implicitly leverages the log-probability, enhancing gradient scaling and resulting in more robust gradients without requiring extra softmax normalization.

![](images/92bbaa36ccda72930d086e77c93a9be8320f21ce63aee76294a1d21feb61b853.jpg)

Figure 1: Overview of the proposed MIG-COW framework. The attack integrates momentum-based integrated gradients and consensus-orthogonal weighting to generate transferable adversarial examples across diverse deepfake detectors.  
![](images/b733acfda30789fe3e51606f63e52d080795a405ef471976ba75f175a4594a35.jpg)  
Figure 2: Comparison of IG visualizations across diferent model: ResNet50, DenseNet121, ViT-P, MIG, and our proposed method.

Furthermore, we decompose the gradient space into two complementary components: a consensus direction that averages L2- normalized attribution maps to capture shared vulnerabilities, and an orthogonal direction derived from the least-aligned eigenvector of the gradient Gram matrix to remove redundant correlations. The final attack direction is then constructed as a weighted combination of these two components.

3.2.2 Consensus Orthogonal Weighting. Given � source models, let $\{ \mathbf { g } _ { i } \} _ { i = 1 } ^ { N }$ denote the IGs obtained from each model, where each g<sub>�</sub> ∈ $\mathbb { R } ^ { C \times \dot { H } \times W }$ has the same shape as the input image � $\in \mathcal { X } = \mathbb { R } ^ { C \times \bar { H } \times W }$ The consensus gradient $\mathbf { g } _ { \mathrm { c o n } }$ is then obtained by ensembling the individual IGs:

$$
\mathbf { g } _ { \mathrm { c o n } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { g } _ { i } .\tag{3}
$$

However, relying solely on this direct averaging approach may overemphasize redundant or shared information while obscuring unique discriminatory information that might be present in individual model gradients.

Compute model-specific directions. To analyze the relationships among the gradients, we construct a Gram matrix �. We first flatten each gradient g<sub>�</sub> into a vector $\widetilde { \bf g } _ { i } = \mathrm { v e c } ( { \bf g } _ { i } ) \in \mathbb { R } ^ { D }$ (where $D =$ $C \times H \times W )$ , then stack these vectors to obtain a matrix $G ,$ and compute the Gram matrix $K \colon$

$$
\begin{array} { r } { \boldsymbol { G } = [ \tilde { \mathbf { g } } ^ { ( 1 ) } , \dots , \tilde { \mathbf { g } } ^ { ( N ) } ] \in \mathbb { R } ^ { D \times N } , \quad \boldsymbol { K } = \boldsymbol { G } ^ { \top } \boldsymbol { G } \in \mathbb { R } ^ { N \times N } , } \end{array}\tag{4}
$$

where each entry $K _ { i j }$ corresponds to the inner product $\langle \tilde { \bf g } _ { i } , \tilde { \bf g } ^ { ( j ) } \rangle$ capturing the degree of correlation between the gradients from models � and �. This matrix � helps to identify the correlation structure among the ensemble gradients.

To extract the least aligned direction relative to the consensus g , we perform eigen decomposition of the Gram matrix and select the eigenvector $v _ { \mathrm { m i n } }$ corresponding to the smallest eigenvalue $\lambda _ { \operatorname* { m i n } }$ , such that $K { \boldsymbol { \upsilon } } _ { \mathrm { m i n } } ~ = ~ \lambda _ { \mathrm { m i n } } { \boldsymbol { \upsilon } } _ { \mathrm { m i n } }$ . This eigenvector captures the direction with minimum redundancy across model gradients—highlighting model-specific or orthogonal components that are underrepresented in the consensus. Using $v _ { \mathrm { m i n } }$ , we construct the aggregated gradient $\mathbf { g } _ { \mathrm { a g g } }$ through its smallest eigenvector as a weighted sum of IGs:

$$
{ \bf g } _ { \mathrm { a g g } } = \sum _ { i = 1 } ^ { N } v _ { \mathrm { m i n } } ^ { ( i ) } { \bf g } _ { i } ,\tag{5}
$$

where $v _ { \mathrm { m i n } } ^ { ( i ) }$ is the �-th entry of the eigenvector $v _ { m i n } .$

However, since the consensus vector $\mathbf { g } _ { \mathrm { c o n } }$ and the aggregated vector $\mathbf { g } _ { \mathrm { a g g } }$ may still share redundant information, we remove the component of $\mathbf { g } _ { \mathrm { a g g } }$ that lies in the direction of $\mathbf { g } _ { \mathrm { c o n } }$ to retain only the orthogonal and non-redundant parts as follows:

$$
\mathbf { g } _ { \mathrm { o r t h } } = \mathbf { g } _ { \mathrm { a g g } } - \frac { \left. \mathbf { g } _ { \mathrm { a g g } } , \mathbf { g } _ { \mathrm { c o n } } \right. } { \| \mathbf { g } _ { \mathrm { c o n } } \| ^ { 2 } + \delta } \cdot \mathbf { g } _ { \mathrm { c o n } } .\tag{6}
$$

As a result, $\mathbf { g } _ { \mathrm { o r t h } }$ captures the model-specific directions and enhances adversarial transferability.

Final Attack Direction. Finally, we construct the combined direction for the attack by weighting the consensus vector and the orthogonal vector as follows:

$$
\mathbf { g } _ { \mathrm { c b } } = \beta \mathbf { g } _ { \mathrm { c o n } } + \left( 1 - \beta \right) \mathbf { g } _ { \mathrm { o r t h } } ,\tag{7}
$$

where $0 < \beta < 1$ . We also normalize the gradient (7).

3.2.3 Perform adversarial update with momentum. At each iteration, the adversarial example is updated with a momentum term:

$$
\mathbf { g } _ { ( t ) } = \mu \mathbf { g } _ { ( t - 1 ) } + { \frac { \mathbf { g } _ { \mathrm { c b } } } { \left\| \mathbf { g } _ { \mathrm { c b } } \right\| } } , \quad \mathbf { x } _ { ( t ) } = \mathrm { C l i p } _ { \epsilon } { \big ( } x _ { ( t - 1 ) } + \alpha \cdot \mathrm { s i g n } ( \mathbf { g } _ { ( t ) } ) { \big ) } ,\tag{8}
$$

where $\mu$ is the momentum decay factor and � is the perturbation budget.

This design allows our attack to benefit from both the robust consensus direction and model-specific diverse cues, thus boosting generalizability across diferent architectures. Algorithm 1 details the overall procedure of MIG-COW.

Algorithm 1: MIG-COW: Momentum Integrated Gradients   
with Consensus-Orthogonal Weighting   
Input: White-box source models $\{ f _ { i } \} _ { i = 1 } ^ { N }$ , original clean   
image � and target label �.   
Parameter: Perturbation budget $\epsilon ,$ iteration number $T ,$   
momentum factor $\mu ,$ weight factor $\beta ,$ baseline   
image �. Small constant $\delta$ for numerical   
stability;   
Output: Adversarial image $\pmb { x } _ { \mathrm { a d v } } .$   
1 Initialize accumulated gradient $\mathbf { g } ( 0 )  \mathbf { 0 } ;$   
2 Set step size $\alpha \gets \epsilon / T ,$ and $\mathbf { \boldsymbol { x } } _ { ( 0 ) } \gets \mathbf { \boldsymbol { x } } ;$   
3 for � = 1 to � do   
// Integrated Gradients for each model   
4 for � = 1 to � do   
5 $\mathbf { g } _ { i }  I G ( f _ { i } , \pmb { x } _ { ( t - 1 ) } , \pmb { b } ) ;$   
6 $\begin{array} { r } { \mathbf { g } _ { \mathrm { c o n } }  \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { g } _ { i } ; } \end{array}$   
// Flattened gradients and Gram matrix   
7 $\mathbf { G } \gets [ \mathrm { v e c } ( \mathbf { g } _ { 1 } ) , \mathrm { ~ . ~ . ~ . , v e c } ( \mathbf { g } _ { N } ) ] \in \mathbb { R } ^ { D \times N } ;$   
8 $K \gets G ^ { T } G ;$   
// Compute the least-redundant direction   
9 Compute eigenvector $v _ { \mathrm { m i n } }$ of � with smallest   
eigenvalue;   
10 $\mathbf { g } _ { \mathrm { a g g } }  \sum _ { i = 1 } ^ { N } \upsilon _ { \mathrm { m i n } } ^ { ( i ) } \mathbf { g } _ { i }$   
11 g<sub>orth</sub> $\begin{array} { r } {  \mathbf { g } _ { \mathrm { a g g } } - \frac { \langle \mathbf { g } _ { \mathrm { a g g } } , \mathbf { g } _ { \mathrm { c o n } } \rangle } { \| \mathbf { g } _ { \mathrm { c o n } } \| ^ { 2 } + \delta } } \end{array}$ · g<sub>con</sub>;   
// Final attack direction   
12 $\mathbf { g } _ { \mathrm { c b } }  \beta \cdot \mathbf { g } _ { \mathrm { c o n } } + ( 1 - \beta ) \cdot \mathbf { g } _ { \mathrm { o r t h } }$   
13 $\begin{array} { r } { \mathbf { g } _ { ( t ) }  \mu \cdot \mathbf { g } _ { ( t - 1 ) } + \frac { \mathbf { g } _ { \mathrm { c b } } } { \| \mathbf { g } _ { \mathrm { c b } } \| _ { 1 } } ; } \end{array}$   
// Update adversarial image   
14 $\begin{array} { r } { \pmb { x } _ { ( t ) } = \mathrm { C l i p } _ { \epsilon } \big ( \pmb { x } _ { ( t - 1 ) } + \alpha \cdot \mathrm { s i g n } \big ( \pmb { \mathrm { g } } _ { ( t ) } \big ) \big ) } \end{array}$   
15 $\boldsymbol { x } _ { \mathrm { a d v } } \gets \boldsymbol { x } _ { ( T ) } ;$   
16 return $x _ { a d y }$

## 4 Experiments

## 4.1 Datasets and Evaluation

We evaluate the efectiveness of MIG-COW using the dataset provided by the AADD-2025 organizers. The dataset is divided into HQ and LQ subsets. Specifically, the HQ subset consists of 693 fake images generated by various models, including Adobe Firefly [13], DeepAI [4], Flux 1.1 Pro [17], HotPotAI [11], Nvidia SanaPAG [27], Stable Difusion 3.5 [1], StyleGAN 2 [16], StyleGAN 3 [14], and Tencent Hunyuan [18]. The LQ subset includes 710 fake images created using DeepAI, Flux 1.1 [17], Freepik [25], HotPotAI, Nvidia SanaPAG Stable Difusion Attend and Excite [3], StyleGAN [15], StyleGAN 3, and Tencent Hunyuan. LQ images are produced by resizing the originals and applying variable Quality Factor compression to simulate the degradation typically observed in social media environments.

All 1,403 fake images were verified in advance to be correctly classified as fake by the two white-box detection models provided by the organizers. We then generate adversarial examples for these images using MIG-COW and evaluate the attack success rate (ASR). Specifically, we calculate the white-box ASR by counting the number of images that successfully deceive the white-box models and dividing this by the total number of images. Similarly, we compute the black-box ASR by evaluating how many adversarial examples successfully fool unseen black-box detectors. Additionally, to assess the visual quality of the adversarial examples, we measure their similarity to the original images using SSIM.

Furthermore, we follow the oficial evaluation criteria provided by the AADD-2025 challenge. The score is computed based on a combination of visual similarity and attack efectiveness. The score is calculated as:

$$
\sum _ { C _ { f } \in C } \sum _ { k = 1 } ^ { n _ { \mathrm { t e s t } } } S S I M ( I _ { k } , I _ { k } ^ { A D V } ) \cdot \mathbb { I } \left[ C _ { f } ( I _ { k } ^ { A D V } ) = \mathrm { L A B E L } _ { r e a l } \right]\tag{9}
$$

where C denotes the set of evaluation classifiers, and $n _ { \mathrm { t e s t } }$ is the number of test images. $I _ { k }$ and $I _ { k } ^ { A D V }$ represent the original and adversarial images, respectively. $\mathrm { L A B E L } _ { r e a l }$ refers to the groundtruth label for the real class. The indicator function I[·] returns 1 if the classifier’s prediction matches the real label, and 0 otherwise.

## 4.2 Experiment Settings

We evaluate the efectiveness of adversarial examples generated by various methods by measuring both their white-box and blackbox ASRs. As discussed in Section $^ { 2 , }$ recent research in deepfake detection can be broadly categorized into CNN-based and ViTbased approaches. To reflect this architectural diversity and analyze diferences in attack robustness, we adopt two CNN-based detectors—ResNet50 [10] and DenseNet121 [12]—provided by the competition organizers as white-box source models, and additionally include a publicly available pretrained ViT-based model [7, 21] (referred to as ViT-P) as a supplementary white-box source. Since the ViT-P was not fine-tuned on the AADD-2025 deepfake dataset, it correctly classified only approximately 70% of the 1,403 clean fake images as fake, reflecting a domain mismatch that degrades its baseline performance, unlike other white-box models.

<table><tr><td rowspan="2">Setting</td><td rowspan="2">Method</td><td colspan="4">White-box ASR (%)</td><td colspan="3">Black-box ASR (%)</td><td rowspan="2">SSIM↑</td><td rowspan="2">Score</td></tr><tr><td>ResNet50</td><td>DenseNet121</td><td>ViT-P</td><td>Avg</td><td>ViT-B/16</td><td>DenseNet121-DCT</td><td>Avg</td></tr><tr><td rowspan="4">w/o ViT-P</td><td>PGD</td><td>56.45</td><td>60.37</td><td></td><td>58.31</td><td>0.50</td><td>7.98</td><td>4.02</td><td>0.9304</td><td>1655</td></tr><tr><td>MI-FGSM</td><td>88.03</td><td>93.94</td><td></td><td>91.00</td><td>3.42</td><td>9.10</td><td>6.26</td><td>0.9096</td><td>2485</td></tr><tr><td>MIG</td><td>96.94</td><td>90.02</td><td>-</td><td>93.48</td><td>3.64</td><td>9.90</td><td>6.77</td><td>0.9169</td><td>2577</td></tr><tr><td>MIG-COW</td><td>100</td><td>99.93</td><td></td><td>99.96</td><td>3.50</td><td>10.83</td><td>7.16</td><td>0.9143</td><td>2747</td></tr><tr><td rowspan="4">w/ ViT-P</td><td>PGD</td><td>57.02</td><td>63.64</td><td>25.74</td><td>48.80</td><td>0.64</td><td>9.10</td><td>4.87</td><td>0.9280</td><td>1716</td></tr><tr><td>MI-FGSM</td><td>86.74</td><td>92.59</td><td>51.53</td><td>76.95</td><td>3.06</td><td>12.12</td><td>7.59</td><td>0.9166</td><td>2480</td></tr><tr><td>MIG</td><td>94.80</td><td>85.79</td><td>90.96</td><td>90.52</td><td>2.00</td><td>8.70</td><td>5.35</td><td>0.9182</td><td>2464</td></tr><tr><td>MIG-COW</td><td>99.93</td><td>99.93</td><td>93.87</td><td>97.91</td><td>1.50</td><td>9.70</td><td>5.60</td><td>0.9161</td><td>2709</td></tr></table>

Table 1: Comparison of attack methods on white-box and black-box detectors, including average ASR, SSIM, and overall score, where the best results are highlighted in bold, and the second-best results are underlined. The Score is computed based on th four models excluding ViT-P (i.e., ResNet50, DenseNet121, ViT-B/16, and DenseNet121-DCT).

For black-box evaluation, we use two oficial challenge models based on the $\mathrm { V i T - B } / 1 6 ^ { 1 }$ and DenseNet121-DCT architectures. Additional details on the black-box setting are provided in Section 4.3.2.

To provide a benchmark for the proposed MIG-COW, we compare its performance against widely used gradient-based attack methods, including PGD, MI-FGSM, and the base MIG framework. For these baseline methods, we adopt a simple ensemble strategy (averaging) as in [22] to enhance their transferability and ensure a fair comparison with MIG-COW’s multi-model design. For consistency across all experiments, we set the perturbation budget $\epsilon = 0 . 0 2$ , the number of iterations $T \ : = \ : 2 5$ , and the momentum factor $\mu = 1$

## 4.3 Experimental Results

Table 1 summarizes results for both white-box and black-box attacks across two evaluation settings—based on whether the ViT-P is included among the white-box detectors.<sup>2</sup> For comparison, overall scores reported in Table 1 are computed using only the two oficial white-box models and two black-box models provided by the AADD-2025 challenge.

4.3.1 Atack on White-box Detectors. In the white-box setting, we evaluate the efectiveness ofadversarial examples when the attacker has full knowledge of the deepfake detector’s architecture and parameters.

In the first setting, which considers only CNN-based detectors (ResNet50 and DenseNet121), gradient-based attacks such as PGD and MI-FGSM achieve average white-box ASRs of 58.31% and 91.00%, respectively. In contrast, MIG attains a white-box ASR of 93.48%, while MIG-COW achieves 99.96%, demonstrating its superior efectiveness. As expected, MIG-COW achieves the highest ASR by leveraging both consensus and model-specific complementary features through its COW mechanism.

In the second setting, the ViT-P (a transformer-based architecture) is included as an additional white-box detector. However, in this setting, PGD and MI-FGSM exhibit limited generalization to the transformer architecture, achieving notably low ASRs of 25.74% and 51.53% on ViT-P, respectively. Meanwhile, MIG and MIG-COW maintain high performance, with ASRs of 90.96% and 93.87% on ViT-P. Notably, MIG-COW still achieves 99.93% ASR on both CNNbased models, despite the inclusion of ViT-P. One potential reason for the baseline MIG’s lower ASR on ViT-P compared to CNNs, despite its overall strength, is that ViT-P, due to its relatively poorer baseline performance on the AADD-2025 dataset (approximately 70% accuracy compared to near-perfect accuracy for other models), may exhibit diferent integrated gradients from the perspective of output equivalence (c.f., implementation invariance). Our results suggest that MIG-COW efectively enhances the robustness of integrated gradients, even when dealing with a classifier that has a relatively poorer baseline performance.

4.3.2 Atack on Black-box Detectors. Our black-box evaluation strictly follows the AADD-2025 Challenge threat model, targeting oficial ViT-B/16 and DenseNet121-DCT models. The adversary has no access to model architecture, parameters, training data, or gradients, and is limited to a single adversarial query per input—receiving only a binary prediction result at test time, without confidence scores. Given these constraints, we employ a transfer-based strategy: adversarial examples are crafted using white-box models and evaluated on the black-box targets to measure transferability.

In this setting, MIG-COW achieves the highest average black-box ASR among all methods, while also delivering significantly better white-box ASR and SSIM. As a result, MIG-COW attains the best overall score by efectively balancing transferability, robustness, and imperceptibility. However, similar to the white-box scenario, including ViT-P as a source model generally degrades black-box ASRs for IG-based methods. In this case, MI-FGSM achieves the highest black-box ASR, but its performance remains poor in the white-box setting. Especially, for both MIG and MIG-COW, black-box ASR was reduced when ViT-P is included, suggesting that the baseline accuracy of source models plays a critical role in transfer-based attacks. Although ViT-P shares the same architectural structure (ViT-B/16) as one of the black-box targets, the results indicate that functional similarity—rather than structural alignment alone—is more important for achieving efective transferability.

![](images/e345e3d81322766f91e8ba95f3c84f950245af5bb1584648420e59851be20dc8.jpg)  
(a) Tuning �

![](images/11beb15091637006002414f85d1eab6c7cc065d337d96838df3e50f0d2411ca1.jpg)  
(b) Tuning �

![](images/594d162a05e8d272d2da47e5f238c7260342842f458ccd7d184edc6729ccce78.jpg)  
(c) Tuning consensus weight

Figure 3: Hyperparameter tuning results across perturbation budget �, steps �, and consensus weight $\mathbf { g _ { c o n } }$ for ASR(%) and SSIM, which are plotted as solid red and dashed blue lines, respectively, in each graph.
<table><tr><td colspan="2">Setting</td><td colspan="3">Detector</td><td rowspan="2">SSIM</td><td rowspan="2">Score</td></tr><tr><td>COW</td><td>CE loss</td><td>ResNet50</td><td>DenseNet121</td><td>ViT-P</td></tr><tr><td rowspan="2">√</td><td></td><td>92.37</td><td>93.08</td><td>94.51</td><td>0.9172</td><td>3,602</td></tr><tr><td>√</td><td>100</td><td>99.93</td><td>89.02</td><td>0.9168</td><td>3,716</td></tr><tr><td>√</td><td>√</td><td>99.93</td><td>99.93</td><td>93.87</td><td>0.9161</td><td>3,772</td></tr></table>

Table 2: White-box ablation study. Each detector score indicates performance under diferent CE loss and COW config urations. The Score is calculated based on the three models presented in the table.

## 4.4 Component and Hyperparameter Analysis

4.4.1 Ablation Study. We conduct an ablation study to evaluate the contribution of key components in the proposed MIG-COW framework: (i) replacing the logit-based cross-entropy loss (CE loss) with the original probability-based formulation used in MIG, and (ii) the inclusion of the orthogonal component from the consensusorthogonal weighting (COW) strategy. These experiments aim to assess how each component impacts the generalizability and robustness of adversarial perturbations.

As shown in Table 2, removing either the CE loss or the COW module consistently leads to evident performance drops in at least one detector. Specifically, excluding CE loss significantly reduces attack success on CNN-based models (ResNet50 and DenseNet121), while removing the COW module causes a moderate drop on ViT-P. This indicates that the CE loss provides more stable gradient signals for CNNs, while the COW module is critical for capturing modelspecific features relevant to ViT-based detectors. MIG-COW—using both components—achieves consistently high performance across all models, with the highest overall score of 3,772.

These results demonstrate that the CE loss and the COW module play complementary roles in balancing generalization and specificity. In particular, the orthogonal component of COW is vital for capturing model-specific decision boundaries not solely reflected in the consensus gradient, enabling simultaneous deception of detectors with diverse underlying architectures.

4.4.2 Hyperparameters. We conduct a sensitivity analysis to evaluate the efect of key hyperparameters on attack performance, focusing on: (1) the perturbation budget �, (2) the number of attack steps �, and (3) the consensus weight in our proposed gradient weighting method. For �, we vary its value from 0.01 to 0.03 (step size 0.002), keeping $T = 2 5$ fixed. As depicted in Figure 3a, increasing � generally improves ASR but concurrently degrades SSIM, highlighting the inherent trade-of between attack strength and visual imperceptibility. We also vary � from 10 to 30 (step size 1), with $\epsilon = 0 . 0 2$ fixed to explore its impact. Figure 3b shows that increasing � yields gradual improvements in both ASR and SSIM, with diminishing returns observed beyond $T = 2 5 .$ . To optimize the gradient weighting, we vary the consensus weight � from 0.1 to 0.9 (step size 0.1), with the orthogonal weight set to 1−�, while keeping $\epsilon = 0 . 0 2$ and $T = 2 5$ fixed. Figure 3c indicates that a consensus weight in the range of 0.7 to 0.8 maximizes ASR while preserving high perceptual similarity, suggesting an optimal balance between shared and model-specific gradient directions.

## 5 Conclusion

In this paper, we propose MIG-COW, a novel adversarial attack framework designed to enhance ensembling across diverse deepfake detection architectures. By leveraging momentum-based optimization alongside a consensus-orthogonal gradient decomposition, MIG-COW efectively captures both shared and model-specific vul nerabilities in heterogeneous ensembles of CNNs and ViT. Our experiments demonstrate MIG-COW’s high attack success rates in white-box settings while preserving visual similarity with benign input. However, its reduced efectiveness against the oficial black-box ViT highlights a key challenge in real-world adversarial settings: bridging the gap between attacker-side surrogate models and unseen, deployment-time detectors with difering architectures, training objectives, and data distributions. We also show that incorporating a low-performing model into the white-box ensemble—despite increasing architectural diversity—can degrade overall attack performance in both white-box and black-box settings. These findings underscore the importance of carefully selecting surrogate models and motivate future research into adaptive, robust attack strategies. Addressing these challenges is essential for advancing adversarial machine learning and for more rigorous evaluations of the robustness of deployed deepfake detection systems.

## Acknowledgments

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS-2020-II201336, Artificial Intelligence Graduate School Program (UNIST), 20%) and ITRC (Information Technology Research Center) grant funded by the Korea government(Ministry of Science and ICT) (IITP-2025-RS-2024-00436936, 50%). This work was also supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2025-00515481, 30%).

## References

[1] Stability AI. 2024. sd3.5: Inference-only tiny reference implementation of Stable Difusion 3.5 and SD3. https://github.com/Stability-AI/sd3.5.

[2] Srinadh Bhojanapalli, Ayan Chakrabarti, Daniel Glasner, Daliang Li, Thomas Unterthiner, and Andreas Veit. 2021. Understanding robustness of transformers for image classification. In Proceedings of the IEEE/CVF international conference on computer vision. IEEE, New York, NY, 10231–10241.

[3] Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. 2023. Attend-and-excite: Attention-based semantic guidance for text-to-image difusion models. ACM transactions on Graphics (TOG) 42, 4 (2023), 1–10.

[4] DeepAI. 2022. Text-to-Image API. https://deepai.org/machine-learning-model/ text2img.

[5] Xiaoyi Dong, Jianmin Bao, Dongdong Chen, Ting Zhang, Weiming Zhang, Neng hai Yu, Dong Chen, Fang Wen, and Baining Guo. 2022. Protecting Celebrities From DeepFake With Identity Consistency Transformer. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, New York, NY, 9468–9478.

[6] Yinpeng Dong, Fangzhou Liao, Tianyu Pang, Hang Su, Jun Zhu, Xiaolin Hu, and Jianguo Li. 2018. Boosting adversarial attacks with momentum. In Proceedings of the IEEE conference on computer vision and pattern recognition. IEEE, New York, NY, 9185–9193.

[7] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xi aohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 2021. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In International Conference on Learning Representations. https://openreview.net/forum?id= YicbFdNTTy

[8] Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. 2014. Explaining and harnessing adversarial examples.

[9] Amira Guesmi, Bassem Ouni, and Muhammad Shafique. 2025. TESSER: Transfer-Enhancing Adversarial Attacks from Vision Transformers via Spectral and Semantic Regularization.

[10] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition. 770–778.

[11] Hotpot.ai. 2021. Hotpot.ai: AI Image Generator & Creative Tools. https://hotpot. ai/.

[12] Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. 2017. Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition. 4700–4708.

[13] Adobe Inc. 2023. Adobe Firefly. https://firefly.adobe.com/.

[14] Tero Karras, Miika Aittala, Samuli Laine, Erik Härkönen, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. 2021. Alias-free generative adversarial networks. Advances in neural information processing systems 34 (2021), 852–863.

[15] Tero Karras, Samuli Laine, and Timo Aila. 2019. A style-based generator architecture for generative adversarial networks. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 4401–4410.

[16] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. 2020. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 8110–8119.

[17] Black Forest Labs. 2024. flux: Oficial inference repo for FLUX.1 models. https: //github.com/black-forest-labs/flux.

[18] Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, et al. 2024. Hunyuandit: A powerful multi-resolution difusion transformer with fine-grained chinese understanding. arXiv preprint arXiv:2405.08748 (2024).

[19] Wenshuo Ma, Yidong Li, Xiaofeng Jia, and Wei Xu. 2023. Transferable adversarial attack for both vision transformers and convolutional networks via momentum integrated gradients. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. IEEE, New York, NY, 4630–4639.

[20] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. 2017. Towards deep learning models resistant to adversarial attacks.

[21] prithivMLmods. [n. d.]. Deep-Fake-Detector-v2-Model.

[22] Yuchen Ren, Zhengyu Zhao, Chenhao Lin, Bo Yang, Lu Zhou, Zhe Liu, and Chao Shen. 2025. Improving integrated gradient-based transferable adversarial examples by refining the integration path. Proceedings ofthe AAAI Conference on Artificial Intelligence 39, 7 (2025), 6731–6739.

[23] Andreas Rossler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, and Matthias Niessner. 2019. FaceForensics++: Learning to Detect Manipulated Facial Images. In Proceedings ofthe IEEE/CVFInternational Conference on Computer Vision (ICCV). IEEE, New York, NY, 1–11.

[24] Selim Seferbekov. 2020. DFDC 1st Place Solution. https://github.com/selimsef/ dfdc\_deepfake\_challenge.

[25] Freepik Company S.L. 2024. Freepik. https://www.freepik.com

[26] Mukund Sundararajan, Ankur Taly, and Qiqi Yan. 2017. Axiomatic attribution for deep networks. In International conference on machine learning. PMLR, Sydney, 3319–3328.

[27] Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, et al. 2024. Sana: Eficient highresolution image synthesis with linear difusion transformers. arXiv preprint arXiv:2410.10629 (2024).

[28] Wanyi Zhuang, Qi Chu, Zhentao Tan, Qiankun Liu, Haojie Yuan, Changtao Miao, Zixiang Luo, and Nenghai Yu. 2022. UIA-ViT: Unsupervised inconsistency-aware method based on vision transformer for face forgery detection. In European conference on computer vision. Springer, Berlin, Heidelberg, 391–407.