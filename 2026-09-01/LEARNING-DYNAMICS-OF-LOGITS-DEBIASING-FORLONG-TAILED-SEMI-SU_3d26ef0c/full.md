# LEARNING DYNAMICS OF LOGITS DEBIASING FORLONG-TAILED SEMI-SUPERVISED LEARNING

Yue Cheng<sup>1,2,∗</sup> Jiajun Zhang<sup>1,∗</sup> Xiaohui Gao<sup>3</sup> Weiwei Xing<sup>1,B</sup> Zhanxing Zhu<sup>4,B</sup>

<sup>1</sup>Beijing Jiaotong University <sup>2</sup>AntGroup

<sup>3</sup>Northwestern Polytechnical University <sup>4</sup>University of Southampton

{yuecheng,jiajunzhang}@bjtu.edu.cn gaitxh@foxmail.com wwxing@bjtu.edu.cn z.zhu@soton.ac.uk

## ABSTRACT

Long-tailed distributions are prevalent in real-world semi-supervised learning (SSL), where pseudo-labels tend to favor majority classes, leading to degraded generalization. While many long-tailed semi-supervised learning (LTSSL) methods have been proposed, the mechanisms by which they implicitly debias logits remain poorly understood. In this work, we revisit LTSSL through the lens of learning dynamics and provide a theoretical characterization of logits debiasing. Specifically, we derive a step-wise decomposition of the logits updates, showing that predictions are dominated by class-imbalance bias that reliably reflects label priors. To expose this effect, we use the logits of a task-irrelevant baseline image as an indicator of accumulated bias and prove that they converge to the class prior. This provides a unified view where LTSSL remedies such as logit adjustment, reweighting, and resampling correspond to reshaping gradient dynamics. Based on this insight, we propose DyTrim, a principle-based dynamic pruning framework that reallocates gradient budget through class-aware pruning on labeled data and confidence-based soft pruning on unlabeled data. We provide theoretical guarantees that DyTrim reduces class bias and improves generalization. Extensive experiments on standard LTSSL benchmarks show consistent gains across architectures and methods. Code available at: https: //jiajun0425.github.io/DyTrim

## 1 INTRODUCTION

Semi-supervised learning (SSL), exemplified by FixMatch (Sohn et al., 2020) and ReMix-Match (Berthelot et al., 2019), has been proven to demonstrate significant generalization advantages over supervised learning, particularly in deep neural networks (Li et al., 2025). However, many existing SSL variants, e.g. FlexMatch (Zhang et al., 2021), FreeMatch (Wang et al., 2023b) implicitly assume that both labeled and unlabeled data are drawn from a balanced class distribution, i.e., class imbalance. In practice, real-world datasets commonly exhibit a long-tailed label distribution, leading to biased pseudo-label toward majority classes. This discrepancy poses significant challenges to the effectiveness of SSL algorithms on real-world datasets.

Recent studies on long-tailed semi-supervised learning (LTSSL) have emerged to mitigate the bias introduced by class imbalance in both labeled and unlabeled data. These methods range from distri bution alignment (Wei et al., 2021; Kim et al., 2020), data rebalancing (Fan et al., 2022; Lee et al., 2021), logit adjustment variants (Wei & Gan, 2023; Zhou et al., 2024), to foundation model-based methods (e.g., LADaS; Zheng et al., 2025). In particular, the approach employs a baseline image introduced by Lee & Kim, 2024 as a simple yet effective tool for quantifying classifier bias, which has garnered significant attention in the community (Xing et al., 2025; Yi et al., 2025). Despite these advancements, the underlying mechanisms of how class bias emerges and why existing approaches can mitigate it remain largely unexplored and poorly understood. That also prevents us from exploring a principle-based method to improve performance.

In this paper, we analyze the underlying mechanisms of class debiasing through the lens of learning dynamics in long-tailed semi-supervised learning (LTSSL), investigating how inputs, the classifier, and pseudo-labels interact and recursively shape one another during training. Specifically, we derive a stepwise decomposition of logit updates in SSL, showing that class imbalance dominates the predictions and prevents the model from leveraging inter-sample similarity, thereby impairing generalization. We further point out that in the learning dynamics of LTSSL, the logits of the baseline image serve as an indicator of the accumulated influence of the network’s bias. Building on this framework, we offer a unified view of existing debiasing methods, including logit adjustment (LA) (Menon et al., 2021), reweighting (Wang et al., 2017), and resampling (JAPKOWICZ, 2000), which can all be understood through the lens of learning dynamics.

As a side product of this analysis, we propose a pruning-based debiasing framework for long-tailed remedies, named DyTrim. For labeled data, we compute class-wise pruning ratios to rebalance samples. For unlabeled data, we apply a label-agnostic criterion that prunes low-confidence, inconsistent samples. Beyond empirical improvements, we provide theoretical guarantees demonstrating how our method alleviates class bias and improves generalization. Extensive experiments demonstrate that our method consistently improves LTSSL performance across standard benchmarks and various backbone architectures.

## 2 PRELIMINARIES

Notions. We consider a labeled dataset $\mathcal { X } = \{ ( x ^ { n } , y ^ { n } ) \} _ { n = 1 } ^ { N }$ with N samples and an unlabeled dataset $\mathcal { U } = \{ u ^ { m } \} _ { m = 1 } ^ { M }$ with M samples, where $x ^ { n } \in \mathbb { R } ^ { d }$ is the n-th labeled sample with label $y ^ { n } \in [ C ] = \{ 1 , \dotsc , C \}$ , and $u ^ { m } \in \bar { \mathbb { R } } ^ { d }$ is the m-th unlabeled sample. Let $N _ { c }$ and $M _ { c }$ denote the number of labeled and unlabeled samples in class c, such that $\textstyle \sum _ { c = 1 } ^ { C } N _ { c } = N$ and $\textstyle \sum _ { c = 1 } ^ { C } M _ { c } = M$ If classes are sorted by size, we have $N _ { 1 } \geq N _ { 2 } \geq \dots \geq N _ { C } \overline { { { \textnormal { o } } } }$ , and define the imbalance ratios as $\gamma _ { l } = N _ { 1 } / N _ { c } \ge 1$ and $\gamma _ { u } = \mathrm { m a x } \{ M _ { i } \} \big / \mathrm { m i n } \{ M _ { i } \} \ge 1$ , respectively. We denote the classifier by $f _ { \theta }$ $\mathbb { R } ^ { d } \mapsto 1 , \dots , C$ with parameters θ, and its logits by $g _ { \theta } ( \bar { x } ) \in \mathbb { R } ^ { \bar { C } }$ , where $f _ { \theta } ( x ) = \arg \operatorname* { m a x } _ { c } g _ { \theta } ( x ) _ { c }$ and $( \cdot ) _ { c }$ <sub>c</sub> denotes the c-th component. For each iteration of training, we sample minibatches $\mathcal { M } \mathcal { X } =$ $\{ ( x _ { b } ^ { n } , y _ { b } ^ { n } ) : b \in ( 1 , \ldots , B ) \} \subset \mathcal { X }$ and $\mathcal { M U } = \{ ( u _ { b } ^ { m } ) : b \in ( 1 , \bar { \ } . . . , \mu B ) \bar { \} } \subset \mathcal { U }$ from the training set, where B denotes the minibatch size and µ denotes the relative size of MU to MX . For brevity, when clear from context we drop the superscript on $u _ { b } ^ { m } \left( x _ { b } ^ { m } \right)$ and simply write $u _ { b } \left( x _ { b } \right)$

Base SSL algorithms. We use FixMatch (Sohn et al., 2020) as the base SSL algorithm, following other LTSSL studies. Specifically, FixMatch first predicts the class probability of a weakly augmented unlabeled data point $\alpha ( u _ { b } )$ as $q _ { b } = \pi _ { \theta } ( y | \alpha ( u _ { b } ) )$ ) and then generates hard pseudo-label $\hat { q } _ { b } = \mathrm { a r g m a x } _ { c } ( q _ { b , c } )$ , where $\pi _ { \theta } ( y | \cdot ) = \operatorname { S o f t m a x } ( g _ { \theta } ( \cdot ) )$ ). For consistency regularization, FixMatch uses a hard pseudo-label $\hat { q } _ { b }$ only when max<sub>c</sub> $\mathbf { \Phi } ( q _ { b , c } ) \geq \tau$ , where τ denotes a predefined confidence threshold, to improve the quality of the pseudo-labels used for training. We express the training losses of FixMatch L as:

$$
\mathcal { L } ( \boldsymbol { x } _ { b } , \boldsymbol { u } _ { b } , \boldsymbol { \hat { q } } , \tau ; \boldsymbol { \theta } ) = \mathcal { L } _ { s u p } ( \alpha ( \boldsymbol { x } _ { b } ) ; \boldsymbol { \theta } ) + \mathcal { L } _ { c o n } ( \boldsymbol { A } ( \boldsymbol { u } _ { b } ) , \boldsymbol { \hat { q } } _ { b } , \tau ; \boldsymbol { \theta } ) ,\tag{1}
$$

where $x _ { b } \ \left( u _ { b } \right)$ denotes the b-th labeled (unlabeled) samples in a minibatch $\mathcal { M X } \left( \mathcal { M } \mathcal { U } \right) . \ A ( u _ { b } )$ denotes the strongly augmented of $u _ { b }$ . The losses and other SSL algorithms, i.e. FlexMatch (Zhang et al., 2021) and FreeMatch (Wang et al., 2023b), are detailed in Appendix B.1 to B.3.

Learning dynamics and its per-step decomposition. Inspired by Ren & Sutherland (2025), we study how a single gradient update changes the model’s confidence on an observation $x _ { o }$ . With $\pi _ { \boldsymbol { \theta } } ( y \mid x )$ denoting the predicted class probability distribution, the learning dynamics become,

$$
\Delta \theta \triangleq \theta ^ { t + 1 } - \theta ^ { t } = - \eta \cdot \nabla { \mathcal { L } } ( f _ { \theta } ( x _ { b } ) , y _ { b } ) ; \quad \Delta \log \pi ^ { t } ( y | x _ { o } ) \triangleq \log \pi _ { \theta ^ { t + 1 } } ( y | x _ { o } ) - \log \pi _ { \theta ^ { t } } ( y | x _ { o } ) .\tag{2}
$$

where the update of $\theta$ during step $t  t + 1$ is given by one gradient update on the sample pair $\left( x _ { b } , y _ { b } \right)$ with learning rate $\eta . \mathcal { L }$ is the loss function, we use the cross-entropy loss H in our setting.

Proposition 1 (Per-step decomposition of learning dynamics; Ren & Sutherland 2025). Let $\pi =$ Softmax(z) with $\mathbf { z } = g _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ . Then the one-step learning dynamics decompose as

$$
\Delta \log \pi _ { \theta } ^ { t } ( y \mid x _ { o } ) = - \eta T ^ { t } ( x _ { o } ) K ^ { t } ( x _ { o } , x _ { b } ) \mathcal { G } ^ { t } ( x _ { b } , y _ { b } ) + \mathcal { O } \left( \eta ^ { 2 } \| \nabla _ { \theta } \mathbf { z } ( x _ { b } ) \| _ { o p } ^ { 2 } \right) ,\tag{3}
$$

where $\mathcal { T } ^ { t } ( x _ { o } ) ~ = ~ \nabla _ { z } \log \pi _ { \theta ^ { t } } ( x _ { o } ) ~ = ~ I - \mathbf { 1 } \pi _ { \theta ^ { t } } ^ { \top } ( x _ { o } )$ only depends on the model’s current predicted probability, $\mathcal { K } ^ { t } ( \boldsymbol { x } _ { o } , \boldsymbol { x } _ { b } ) = ( \nabla _ { \theta } z ( \boldsymbol { x } _ { o } ) | _ { \theta ^ { t } } ) ( \nabla _ { \theta } z ( \boldsymbol { x } _ { b } ) | _ { \theta ^ { t } } ) ^ { \top }$ is the empirical neural tangent kernel

(eNTK, Jacot et al. 2018) of the model, the product of the model’s gradients with respect to $x _ { o }$ and x<sub>b</sub>. $\mathcal { G } ^ { t } ( x _ { b } , y _ { b } ) = \nabla _ { \mathbf { z } } \mathcal { L } ( x _ { b } , y _ { b } ) | _ { \mathbf { z } ^ { t } }$ is the loss gradient. $\operatorname { \bar { \| } } \cdot \operatorname { \| } _ { o p } ^ { 2 }$ denotes the spectral norm, which bounds the second-order remainder term.

![](images/e4bd68dd560e8d3ba86b998730d983ffad5668967f2656a027fe74a250f98ecd.jpg)

![](images/2c63c7dec8831d1bd49637f6bfcbacf91739f2dd1e1cec42df8565294cf1f666.jpg)

![](images/5ad929cee2851d61ea29a480b9fdfbf6c8ec527561a92cc387966750860d07e8.jpg)

![](images/e7316c6456f804a2716d923e8593217f9770a9d6ce2cb7c67d9798ec2416f210.jpg)  
Figure 1: Accumulated influence in the MNIST experiment using a labeled sample $x _ { b } = 0$ and an unlabeled sample $u _ { b } = 0$ for training, with $x _ { o } = 4$ for testing. (a) and (b) shows results from the Balanced experiment (MNIST), (c) and (d) from the Imbalanced experiment (MNIST-LT). (a) and (c) show the influence with accurate pseudo-labels, (b) and (d) with inaccurate pseudo-labels. In (a) and (b), the cumulative influence of pseudo-label authenticity is evident, with the false pseudo-label affecting predictions for similar samples $( e . g .$ , probability of 9, 7 and 4). In (c) and (d), the class imbalance masks the influence of false pseudo-label authenticity due to class bias.

This decomposition characterizes how each update at $( x _ { b } , y _ { b } )$ influences predictions at $x _ { o } ,$ , forming the basis for our SSL analysis under class imbalance.

## 3 LEARNING DYNAMICS OF LONG-TAILED SEMI-SUPERVISED DEBIASING

## 3.1 LEARNING DYNAMICS OF SEMI-SUPERVISED LEARNING

In this section, we characterize the learning dynamics of the semi-supervised version of gradient descent (GD) for the FixMatch algorithm Eq. (1),

$$
\begin{array} { c } { \Delta \theta \triangleq \theta ^ { t + 1 } - \theta ^ { t } = - \eta \cdot \left( \nabla \mathcal { L } _ { s u p } \big ( f _ { \theta } ( \alpha ( x _ { b } ) ) , y _ { b } \big ) + \nabla \mathcal { L } _ { c o n } \big ( f _ { \theta } ( \alpha ( u _ { b } ) ) , f _ { \theta } ( \mathcal { A } ( u _ { b } ) ) \big ) \right) ; } \\ { \Delta f ( x _ { o } ) \triangleq f _ { \theta ^ { t + 1 } } ( x _ { o } ) - f _ { \theta ^ { t } } ( x _ { o } ) . } \end{array}\tag{4}
$$

where $x _ { o }$ denotes the observation data point, the update of θ during step $t  t + 1$ is given by one gradient update on the labeled sample pair $( x _ { b } , y _ { b } )$ and unlabeled sample (u ) with learning rate $\eta .$ Previous work (Ren & Sutherland, 2025) showed how a single gradient update influences model predictions in supervised learning. We now examine whether such characterization extends to the semi-supervised setting. Since FixMatch (Sohn et al., 2020) update naturally consists of a supervised part $\mathcal { L } _ { s u p }$ and a consistency part $\mathcal { L } _ { c o n } .$ , the gradient update can be decomposed accordingly. For an unlabeled sample $u _ { b }$ with target $\hat { q } _ { b } ^ { t } = \arg \operatorname* { m a x } _ { c } q _ { b , c } ^ { t }$ , where $q _ { b } ^ { t } = \pi _ { \theta ^ { t } } ( \cdot ~ | ~ \alpha ( u _ { b } ) )$ ). The per-step learning dynamics of semi-supervised learning become

$$
\Delta \log \pi ^ { t } ( y | x _ { o } ) \triangleq \Delta \log \pi _ { \theta } ^ { t , \mathrm { s u p } } ( y \mid x _ { o } ; x _ { b } ) + \Delta \log \pi _ { \theta } ^ { t , \mathrm { c o n } } ( y \mid x _ { o } ; u _ { b } )\tag{5}
$$

where $\Delta \pi _ { \theta } ^ { t , \mathrm { s u p } }$ denotes the influence caused by $x _ { b }$ and $\Delta \pi _ { \theta } ^ { t , \mathrm { c o n } }$ denotes the influence caused by $u _ { b } .$ respectively. Inspired by Definition 1, we now state the decomposition of the per-step influence in semi-supervised learning below:

Proposition 2. For an labeled (unlabeled) sample $x _ { b } \left( u _ { b } \right)$ with target y<sub>b</sub> $( \hat { q } _ { b } ^ { t } )$ . The one-step learning dynamics of SSL decompose as

$$
\begin{array} { r l } & { \Delta \log \pi _ { \theta } ^ { t , \mathrm { s u p } } ( y \mid x _ { o } ; x _ { b } ) = - \eta T ^ { t } ( x _ { o } ) K ^ { t } ( x _ { o } , \alpha ( x _ { b } ) ) \mathcal { G } _ { \mathrm { s u p } } ^ { t } ( \alpha ( x _ { b } ) , y _ { b } ) + \mathcal { O } \left( \eta ^ { 2 } \| \nabla _ { \theta } \mathbf { z } ( \alpha ( x _ { b } ) ) \| _ { \rho p } ^ { 2 } \right) } \\ & { \Delta \log \pi _ { \theta } ^ { t , \mathrm { c o n } } ( y \mid x _ { o } ; u _ { b } ) = - \eta T ^ { t } ( x _ { o } ) K ^ { t } ( x _ { o } , A ( u _ { b } ) ) \mathcal { G } _ { \mathrm { c o n } } ^ { t } ( A ( u _ { b } ) , \hat { q } _ { b } ^ { t } ) + \mathcal { O } \left( \eta ^ { 2 } \| \nabla _ { \theta } \mathbf { z } ( A ( u _ { b } ) ) \| _ { \rho p } ^ { 2 } \right) } \end{array}\tag{6}
$$

where $\mathcal { K } ^ { t } ( x _ { o } , \alpha ( x _ { b } ) )$ and $\mathcal { K } ^ { t } ( x _ { o } , \mathcal { A } ( u _ { b } ) )$ are eNTK evaluations of the logit network $\begin{array} { r l } { \mathbf { z } ( \cdot ) } & { { } = } \end{array}$ $g _ { \boldsymbol { \theta } } ( \cdot )$ , with different inputs. $\dot { \mathcal { G } } _ { \mathrm { s u p } } ^ { t } ( \alpha ( x _ { b } ) , y _ { b } ) \ = \ \nabla _ { \mathbf { z } } \mathcal { L } _ { \mathrm { s u p } } ( \alpha ( x _ { b } ) , y _ { b } ) | _ { \mathbf { z } ^ { t } }$ and $\mathcal { G } _ { \mathrm { c o n } } ^ { t } ( \hat { q } _ { b } , \mathcal { A } ( u _ { b } ) ) \ =$ $\nabla _ { \mathbf { z } } \mathcal { L } _ { \mathrm { c o n } } ( \hat { q } _ { b } , \mathcal { A } ( u _ { b } ) ) | _ { \mathbf { z } ^ { t } }$ , respectively.

As shown in Proposition 2, each update of θ in FixMatch decomposes into a supervised part driven by $\left( x _ { b } , y _ { b } \right)$ and a consistency part driven by $( u _ { b } , \hat { q } _ { b } ^ { t } )$ . While this decomposition captures the perstep influence on $\pi _ { \boldsymbol { \theta } } ( \boldsymbol { y } \mid \boldsymbol { x } _ { o } )$ , in practice training consists of many such steps, and the accumulated effect is governed by the iterative interaction between labeled and unlabeled updates. The detailed technical proofs are deferred to Appendix C.1.

Accumulated influence and a demonstration on MNIST. To demonstrate this, we train a WRN-28-2 on MNIST and visualize the accumulated influence in Figure 1. In Figure 1(a), when $\hat { q } _ { b }$ is correct, the consistency term reinforces the supervised signal, gradually pulling the prediction of $x _ { o }$ toward the correct class, $i . e . , q _ { b , 4 _ { \uparrow } }$ and ${ q _ { b , 9 _ { \perp } } } ,$ consistent with the constructive dynamics implied by Eq. (6). In contrast, when $\hat { q } _ { b }$ is incorrect (Figure 1(b)), the consistency update exerts the opposite effect, $i . e . , q _ { b , 4 \downarrow } , q _ { b , 7 _ { \uparrow } }$ and ${ q _ { b , 9 } } _ { \downarrow }$ , systematically reducing the correct probability of $x _ { o }$ . This illustrates how pseudo-label errors, even if small at each step, can accumulate across iterations into a negative loop. The Figure 1(c) and (d) show that under class imbalance, such accumulated influence can drive the classifier to consistently predict the majority class (here $q _ { b , 0 } > q _ { b , 4 } )$ , regardless of the true label. This confirms the implication of our dynamics analysis: in SSL, the imbalance influence of labeled data is passed to the pseudo-labels through the classifier, so imbalance bias can be amplified rather than averaged out, leading to catastrophic bias.

## 3.2 LEARNING DYNAMICS ANALYSIS OF ACCUMULATED BIAS UNDER CLASS IMBALANCE

The aforementioned phenomenon, together with the learning dynamics of the semi-supervised framework, illustrates how class imbalance accumulates into systematic bias. While per-update dynamics capture the influence of individual samples on predictions, they fall short of reflecting the global effect of imbalance. This motivates the search for an indicator that bridges class-imbalance bias with the underlying learning dynamics. Replacing the inputs $x _ { o }$ with a task irrelevant baseline image $\mathcal { T } ,$ we can regard the Eq. (6) as such an attributing indicator (Sundararajan et al., 2017). To justify this choice, we analyze its theoretical properties in both linear and deep settings, and then incorporate it into the per-step influence decomposition.

Baseline image and its invariance property. For simplicity, we first consider a two-layer MLP with no bias in the first layer and a bias vector $\pmb { b } \in \mathbb { R } ^ { C }$ in the output layer $h ( x ) = h ^ { ( 2 ) } \circ h ^ { ( 1 ) } ( x )$ where $h ^ { ( 1 ) } ( x ) = \sigma ( W _ { 1 } x )$ and $h ^ { ( 2 ) } = { W _ { 2 } } x + b .$ . This setting allows us to isolate and examine the predicted class probability $\pi _ { \boldsymbol { \theta } } ( \mathcal { T } )$ of a baseline image. For a baseline image $\mathcal { T } \in \mathbb { R } ^ { d }$ , we have

$$
h ( \mathcal { T } ) = W _ { 2 } h ^ { ( 1 ) } ( \mathcal { T } ) + b .\tag{7}
$$

In modern neural networks, the explicit bias term $^ { b }$ is often absorbed into the normalization layer, $e . g .$ , BatchNorm, LayerNorm, with other layers typically set without bias. Without loss of generality, we take BatchNorm as an example for analysis. Since the BatchNorm transformation can be equivalently viewed as an affine linear layer with learnable parameters, we may replace $h ^ { ( 2 ) }$ with a BatchNorm(·) layer, $i . e .$

$$
h ( \mathbf { \mathcal { T } } ) = \mathtt { B a t c h N o r m } \bigl ( h ^ { ( 1 ) } ( \mathbf { \mathcal { T } } ) \bigr ) = \frac { h ^ { ( 1 ) } ( \mathbf { \mathcal { T } } ) - \mathbb { E } [ h ^ { ( 1 ) } ( \mathbf { \mathcal { T } } ) ] } { \sqrt { \operatorname { V a r } [ h ^ { ( 1 ) } ( \mathbf { \mathcal { T } } ) ] + \epsilon } } \cdot W _ { 2 } + b .\tag{8}
$$

where ϵ is a small positive constant that ensures numerical stability. The baseline image is typically a solid color image, which inherently lacks task-related patterns, see Appendix D.1 for more discussions. This representation shows that, for baseline images, the dependence of $h ( \mathcal T )$ on the input is effectively controlled only through the affine parameters $( W _ { 2 } , \pmb { b } )$ of the normalization layer. We now state the main results regarding the prediction $\pi _ { \boldsymbol { \theta } } ( \mathcal { T } )$ for such baseline images:

Proposition 3 (Invariance of baseline image under affine normalization). Let $\mathcal { T } = \boldsymbol { k } \cdot \mathbf { 1 } _ { d }$ be a solid color image, where $k \in \{ 0 , 1 , \ldots , 2 5 5 \}$ and $\mathbf { 1 } _ { d } \in \mathbb { R } ^ { d }$ is an all-one vector. Suppose the output ofthe first hidden transformation is normalized by a normalization layer $( \mathrm { e . g . }$ , BatchNorm, InstanceNorm, or GroupNorm) with affine parameters $( \dot { W } _ { 2 } , b )$ . Then the logits $h ( \mathcal T )$ are independent of k and reduce to

$$
h ( \mathcal { T } ) = b , \quad \pi _ { \theta } ( \mathcal { T } ) = S o f t m a x ( b ) .\tag{9}
$$

One can immediately notice that $\pi _ { \boldsymbol { \theta } } ( \mathcal { T } )$ in Eq. (9) does not contain any term related to the pixel value k of I. This observation implies that the representation $\pi _ { \boldsymbol { \theta } } ( \mathcal { T } )$ of a baseline image is entirely determined by the BatchNorm bias term b, and is invariant to the actual pixel value k. The detailed technical proofs are deferred to Appendix C.2.

Building upon this invariance, we now establish a direct connection between the baseline image and the underlying class distribution. Specifically, for the classifier formulation in Eq. (8) and Eq. (9), we show that the logits of the baseline image encode the class-imbalance ratio present in the training data, thus providing a direct bridge between $\pi _ { \boldsymbol { \theta } } ( \mathcal { T } )$ and the class prior induced by the long-tailed distribution in training. We empirically validate this connection on CIFAR10-LT by analyzing the distribution of baseline logits: as shown in Figure 2, the baseline logits closely align with the empirical class prior. When we remove the bias term in our ablation model, this alignment vanishes, indicating that the baseline logits lose their responsiveness to the class prior.

![](images/0403d380f8d8f589edadfdaea3d73a9a13e251263dc7d7eeed61a4eaa619269f.jpg)  
(a) Labeled dataset

![](images/3fc01694cc26e2dc686486e3b5256946eb4d1100bb59f63ae414d59814b7a74f.jpg)  
(b) Unlabeled dataset

![](images/9b0f8713159e91748829a160c3cec26544b3c7f5bd69724f434ee9a14ba93906.jpg)  
(c) Full dataset  
Figure 2: Class distributions and measured biaseddegree under $\gamma _ { l } = 1 0 0$ and $\gamma _ { u } = 1 0 0$ . The bar plots show the class distributions for (a) labeled, (b) unlabeled, and (c) full datasets.

Theorem 1 (Bias as the conditional distribution prior). Assume the model $h ( x )$ as characterized in Eq. (8) is trained using cross-entropy loss:

$$
\mathcal { L } = \mathbb { E } _ { ( x , y ) } \big [ - y ^ { \top } \log S o f t m a x ( h ( x ) ) \big ] .\tag{10}
$$

At a population risk minimizer $( W _ { 2 } ^ { \star } , b ^ { \star } )$ we have

$$
\begin{array} { r } { \hat { p } ^ { \star } ( x ) = P ( y \mid x ) , \qquad \hat { p } ^ { \star } ( \mathcal { T } ) = S o f t m a x \big ( { b } ^ { \star } \big ) = P \big ( y \big | \frac { h ^ { ( 1 ) } ( \underline { { \mathcal { T } } } ) - \underline { { \mathbb { E } } } [ h ^ { ( 1 ) } ( \underline { { \mathcal { T } } } ) ] } { \sqrt { \mathrm { V a r } [ h ^ { ( 1 ) } ( \underline { { \mathcal { T } } } ) ] + \epsilon } } = \bf { 0 } \big ) . } \end{array}\tag{11}
$$

For the baseline image I in Proposition 3, the baseline prediction thus coincides with the conditional class distribution at the normalized-zero feature state, capturing the class prior induced by the longtailed training distribution. See the detailed to Appendix C.3.

Thus, $\pi _ { \boldsymbol { \theta } } ( \mathcal { T } )$ serves as a natural proxy for the accumulated bias of the model, bridging the class imbalance in the training set to the learning dynamics of the classifier.

Per-step influence decomposition of the baseline image. Let $\pi _ { \boldsymbol { \theta } } ( y | \cdot )$ denote the estimate of the underlying class prior. Then we can track the change in the model’s confidence by observing log $\pi _ { \boldsymbol { \theta } } ( y | \mathcal { T } )$ . Then the learning dynamics on the baseline image become,

$$
\Delta \log \pi ^ { t } ( y | \mathcal { T } ) \triangleq \log \pi _ { \theta ^ { t + 1 } } ( y | \mathcal { T } ) - \log \pi _ { \theta ^ { t } } ( y | \mathcal { T } ) .\tag{12}
$$

Proposition 4. Let $\pi = \mathtt { S o f t m a x ( z ) }$ and $\mathbf { z } = g _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ . The one-step dynamics on the baseline image decompose as

$$
\Delta \log \pi _ { \boldsymbol { \theta } } ^ { t } ( \boldsymbol { y } \mid \mathcal { Z } ; \boldsymbol { x } ) = - \eta \mathcal { T } ^ { t } ( \mathcal { Z } ) K ^ { t } ( \mathcal { T } , \boldsymbol { x } ) \mathcal { G } ^ { t } ( \boldsymbol { x } , \boldsymbol { y } ) + \mathcal { O } \left( \eta ^ { 2 } \Vert \nabla _ { \boldsymbol { \theta } } \mathbf { z } ( \boldsymbol { x } ) \Vert _ { o p } ^ { 2 } \right)\tag{13}
$$

where $\mathcal { T } ^ { t } ( \mathcal { Z } ) = \nabla _ { \mathbf { z } } \log \pi ^ { t } ( \mathcal { T } ) = I - \mathbf { 1 } \pi _ { \theta ^ { t } } ^ { T } ( \mathcal { Z } ) , \mathcal { K } ^ { t } ( \mathcal { Z } , x ) = \left( \nabla _ { \theta } \mathbf { z } ( \mathcal { T } ) \big | _ { \theta ^ { t } } \right) \left( \nabla _ { \theta } \mathbf { z } ( x ) \big | _ { \theta ^ { t } } \right) ^ { T }$ is the eNTK of the logit network $\mathbf { z } ,$ x can be $\alpha ( x _ { b } ) o r \mathcal { A } ( u _ { b } )$ , y can be $y _ { b }$ and α(u<sub>b</sub>). See Appendix C.4 for more details.

Compared with Proposition 2, the main difference is that the $\mathcal { T } ^ { t } ( \mathcal { T } )$ and $\displaystyle { \mathcal { K } } ^ { t } ( { \mathcal { T } } , x )$ term. Since the baseline image I lies far from the data manifold, the coupling kernel $\mathcal { K } ^ { t } ( \mathcal { T } , x )$ is typically small.

![](images/5e35556a553be60b4cba232c602728e0e1e6c1fa644735e5fa729e28f383dc93.jpg)

![](images/a651ebeaa0081b1cb3c78a03f78e23834d4fc0d8a3b243a8d1d78f0c1c1872c4.jpg)

![](images/5207c7834539d8de3a3e606e0786b7db320849bde921f67fa1897b91331114e8.jpg)

![](images/8127f122ea629535d5451c1e5fa70076655d8fe98921266b33007f9331591363.jpg)  
Figure 3: The change of logits’s probability distribution $\pi _ { \boldsymbol { \theta } } ( \mathcal { T } )$ for the baseline image on CIFAR-10- LT. The left three panels depict the dynamics of reference logits under FixMatch: at epoch 1, epochs 1-10, and full epochs. The rightmost panel illustrates the dynamics after removing all bias terms.

Thus, the learning dynamics in Eq. (13) are mainly governed by the output-sensitivity term $\mathcal T ^ { t } ( \mathcal T )$ and the gradient signal $\mathcal { G } ^ { t }$ , with the latter providing both the energy and direction for the model’s adaptation. Under this formulation, the baseline image I serves as an indicator that isolates the model’s global bias state. Tracking $\pi _ { \theta } ^ { t } ( \mathcal { T } )$ over training therefore provides a direct and interpretable measurement of how class-level bias accumulates during semi-supervised learning. Therefore, as the number of labeled and unlabeled samples from the majority class increases, the output of $\pi _ { \theta } ^ { t } ( \mathcal { T } )$ will be progressively squeezed into a biased long-tailed distribution. Even with $\mathcal { G } ^ { t }$ guiding the adaptation direction, this process can still be steered by the biased state encoded in $\pi _ { \theta } ^ { t } ( \mathcal { T } )$ , further amplifying the long-tailed shift, as illustrated in Figure 3.

## 4 DYNAMICS ANALYSIS OF LOGITS DEBIASING IN SEMI-SUPERVISEDLEARNING

Analyzing the dynamics of logits debiasing methods in long-tailed semi-supervised learning is challenging because different algorithms such as Logits Adjustment, Reweighting, and Resampling employ distinct formulations. In this section, we propose a unified framework based on the per-step influence decomposition (Proposition 4). This framework enables us to analyze how these methods modify the update gradient flow, thereby influencing the model’s bias evolution during training. We also introduce a pruning-based method, DyTrim, as a byproduct of our analysis. It can be integrated in a plug-and-play manner with other logits debiasing methods.

## 4.1 PER-STEP DECOMPOSITION OF LOGITS ADJUSTMENT

The typical logits debias method used during long-tail semi-supervised learning is logits adjustment (LA) (Menon et al., 2021), which introduces a class-dependent shift in the logits, expressed as:

$$
\tilde { \pi } _ { \boldsymbol { \theta } } ( y \vert x ) = \mathrm { S o f t m a x } ( \tilde { \mathbf { z } } ( x ) ) , \qquad \tilde { \mathbf { z } } ( x ) = g _ { \boldsymbol { \theta } } ( x ) - \lambda \boldsymbol { \phi } ,\tag{14}
$$

where $\lambda \geq 0$ controls the adjustment strength and $\phi \in \mathbb { R } ^ { C }$ is estimates of the class priors. Thanks to the z˜ implemented in CDMAD (Lee & Kim, 2024), the resulting logits adjustment is almost identical to such simple subtraction, $i . e . , \tilde { z } ( x ) = g _ { \theta } ( x ) - \log \pi$ , where $\pi = \pi _ { \boldsymbol { \theta } } ( \mathcal { T } )$ . Thus, the change of the model’s prediction on the baseline image I can be represented as,

$$
\Delta \log \tilde { \pi } _ { \boldsymbol { \theta } } ^ { t } ( y \mid \mathcal { Z } ; x _ { b } ) = - \eta \mathcal { T } ^ { t } ( \mathcal { T } ) K ^ { t } ( \mathcal { Z } , x _ { b } ) \tilde { \mathcal { G } } _ { L A } ^ { t } ( x , y ) \ + \ Q \big ( \eta ^ { 2 } \| \nabla _ { \boldsymbol { \theta } } \tilde { \mathbf { z } } ( x _ { b } ) \| _ { \mathrm { o p } } ^ { 2 } \big ) .\tag{15}
$$

where $\tilde { \mathcal { G } } _ { L A } ( x , y ) = \pi _ { \theta } ^ { t } ( \alpha ( u _ { b } ) | \boldsymbol { A } ( u _ { b } ) ) - \pi$ represents the influence of the adjusted logits. Compared with Proposition 4, the main difference is that the gradient term has been modified by class prior π, which allows us to answer how does learning with debiasing affect the gradients for unlabeled samples? When adjusting the model’s logits by class prior, the gradient flow will ensure that the model compensates for the class imbalance during training. See more discussions in Appendix C.5. We also conducted experiments on CIFAR10-LT to demonstrate the effectiveness of this debiasing, as illustrated in Figure 3, which illustrates that the bias measured in the baseline image after applying LA to the CDMAD method is alleviated.

## 4.2 PER-STEP DECOMPOSITION OF REWEIGHTING

Reweighting is another prevalent debiasing technique in long-tail semi-supervised learning (Lai et al., 2022), which introduces class-dependent weights in the loss function, expressed as:

$$
\mathcal { L } _ { s u p } ^ { r w } = \sum _ { k = 1 } ^ { C } w _ { k } ^ { l } \mathcal { L } _ { s u p } ( \alpha ( x _ { b } ^ { k } ) ; \theta ) ; \quad \mathcal { L } _ { c o n } ^ { r w } = \sum _ { k = 1 } ^ { C } w _ { k } ^ { u } \mathcal { L } _ { c o n } ( A ( u _ { b } ^ { k } ) , \hat { q } _ { b } , \tau ; \theta ) ;\tag{16}
$$

where $w _ { k } ^ { l } \left( w _ { k } ^ { u } \right)$ is the weight of the k-th class in labeled (unlabeled) samples. For simplicity, we assume the class weight distributions are consistent between labeled and unlabeled data, $i . e . , w _ { k } ^ { l }$ and $w _ { k } ^ { u }$ follow the same proportional relationship and remain fixed during training. Under this reweighting scheme, the gradient signals for both supervised and consistency terms are scaled by their respective class weights. Hence, we can decompose the learning dynamics for reweighting similarly to Eq. (15),

$$
\Delta \log \pi _ { \theta } ^ { t , r w } ( y \mid \mathcal { Z } ; x ) = - \eta \mathcal { T } ^ { t } ( \mathcal { Z } ) \tilde { K } _ { r w } ^ { t } ( \mathcal { Z } , x ; w ^ { c } ) \tilde { \mathcal { G } } _ { r w } ^ { t } ( x , y ; w ^ { c } ) + \mathcal { O } \left( \eta ^ { 2 } | \nabla \theta \mathbf { z } ( x ) | \mathrm { o p } ^ { 2 } \right) ,\tag{17}
$$

where $\tilde { \mathcal { K } } _ { r w } ^ { t } ( \mathbb { Z } , x ; w ^ { c } ) = w ^ { c } \mathcal { K } _ { r w } ^ { t } ( \mathbb { Z } , x )$ and $\tilde { \mathcal { G } } _ { r w } ^ { t } ( x , y ; w ^ { c } ) = w ^ { c } \mathcal { G } ^ { t } ( x , y )$ . Thus, reweighting acts by scaling both the similarity kernel and the gradient term with the class weight $w ^ { c }$ . Intuitively, this modulates the strength of interaction between samples and the magnitude of their gradients in a class-dependent manner: samples from classes with larger $w ^ { c }$ exert a stronger influence on the update of $\theta ,$ while those from classes with smaller $w ^ { c }$ contribute less. When $w ^ { c }$ is designed as a function of class frequency $( e . g .$ , inverse frequency), this mechanism increases the effective contribution of under-represented classes and attenuates that of head classes. See more discussions in Appendix C.5.

## 4.3 DYTRIM: A BASELINE IMAGE GUIDED DATA PRUNING FRAMEWORK FOR LTSSL

Under the per-step influence framework of Proposition 4, logits adjustment and reweighting reshape the gradient flow by modifying the update direction or magnitude, while resampling acts directly on the data distribution by changing the frequency with which different classes enter training. Yet all these methods leave the sample set itself intact at each step and ignore the heterogeneous per-step utility of individual samples, allowing redundant head-class examples to continue dominating the learning dynamics. This motivates debiasing at the data-selection level, where dynamically controlling which samples participate in each update provides a more direct mechanism for mitigating accumulated bias in LTSSL, as illustrated in Figure 4.

Per-step decomposition of dynamic pruning. Differs from logits adjustment, reweighting, or resampling, dynamic pruning directly alters the set of samples that participate in each gradient update, instead of modifying the loss or sampling distribution. We define step-dependent scoring functions $\mathcal { H } _ { t } ^ { l } ( \cdot )$ for labeled samples $\mathcal { X }$ and $\mathcal { H } _ { t } ^ { u } ( \cdot )$ for unlabeled samples $u ,$ which dynamically quantify sample utility at training step t. For the dynamic pruning process, samples are discarded by the step-dependent pruning probabilities $\mathcal { P } _ { t } ^ { l }$ and $\mathcal { P } _ { t } ^ { u }$

$$
\mathcal { P } _ { t } ^ { l } ( x ; \mathcal { H } _ { t } ^ { l } ) = \mathbb { 1 } ( \mathcal { H } _ { t } ^ { l } ( x ) , \bar { H } _ { t } ^ { l } ) ; \quad \mathrm { a n d } \quad \mathcal { P } _ { t } ^ { u } ( u ; \mathcal { H } _ { t } ^ { u } ) = \mathbb { 1 } ( \mathcal { H } _ { t } ^ { u } ( u ) , \bar { H } _ { t } ^ { u } ) ,\tag{18}
$$

where $\bar { \pmb { H } } _ { t } ^ { l }$ and $\bar { \pmb { H } } _ { t } ^ { u }$ are adaptive thresholds, $\mathbb { 1 } ( \cdot , \cdot )$ is the indicator function. Under this dynamic pruning mechanism, the one-step decomposition of dynamic pruning decomposes as

$$
\begin{array} { r l } & { \Delta \log \pi _ { \theta } ^ { t , \mathrm { p r u n e } } ( y \mid \mathcal { Z } ; x ) = - \eta \mathcal { T } ^ { t } ( \mathcal { Z } ) \mathcal { K } ^ { t } ( \mathcal { Z } , x ) \tilde { \mathcal { G } } _ { d y t r } ^ { t } ( x , y ) + \mathcal { O } \left( \eta ^ { 2 } | \nabla \theta \mathbf { z } ( x ) | \mathrm { o p } ^ { 2 } \right) } \\ & { \qquad \tilde { \mathcal { G } } _ { d y t r } ^ { t } ( x , y ) = \mathcal { P } _ { t } ( x ) \mathcal { G } ^ { t } ( x , y ) } \end{array}\tag{19}
$$

where

$$
\mathcal { P } _ { t } ( x ) = \left\{ \begin{array} { r l } { \mathcal { P } _ { t } ^ { l } ( x ; \mathcal { H } _ { t } ^ { l } ) } & { \quad x \in \mathcal { X } , } \\ { \mathcal { P } _ { t } ^ { u } ( u ; \mathcal { H } _ { t } ^ { u } ) } & { \quad x \in \mathcal { U } , } \end{array} \right.\tag{20}
$$

This decomposition shows that dynamic pruning reshapes the update dynamics by gating sample participation through $\mathcal { P } _ { t } ^ { l }$ and $\mathcal { P } _ { t } ^ { u }$ , effectively zeroing out the kernel–gradient interactions $\dot { \mathcal { K } } ^ { t } ( \dot { \mathcal { T } } , x ) \mathcal { G } ^ { \dot { t } } ( x , y )$ of low-utility samples. In contrast to logits adjustment and reweighting, which only alter gradient signals, or resampling, which changes the sampling measure, pruning directly removes redundant head-class examples and underlearned unlabeled ones from the optimization path, thereby reallocating the model’s effective update budget toward samples that meaningfully influence bias correction. Although the kernel $\mathcal { K } ^ { \bar { t } } ( \mathcal { T } , x )$ itself remains unchanged, its operational contribution becomes $\mathbb { E } _ { \boldsymbol { x } \sim p } [ \mathcal { P } _ { t } ( \boldsymbol { x } ) \mathbf { \bar { \mathcal { K } } } ^ { t } ( \boldsymbol { \mathcal { T } } , \boldsymbol { x } ) ]$ ], selectively amplifying informative interactions while suppressing those that drive long-tailed drift. This sample-level intervention yields a more direct and fine-grained control of the learning dynamics than existing debiasing strategies.

Building on this perspective, we now instantiate how dynamic pruning is implemented in practice. We introduce DyTrim, a baseline-guided dynamic pruning framework designed to accommodate the distributional mismatch that real-world LTSSL typically exhibits between labeled and unlabeled data. Since such mismatch renders a single participation rule inadequate, DyTrim employs two complementary pruning mechanisms, one tailored to the long-tailed labeled set and the other to the noisy and imbalance-unknown unlabeled set. See more details about Appendix C.6.

Dynamic pruning for labeled data. Since the labeled data follow a long-tailed class distribution, we design a class-aware pruning policy $\mathcal { P } _ { t } ^ { l }$ guided by $\pi _ { \boldsymbol { \theta } } ( \mathcal { T } )$ . Critically, the classifier’s pseudolabels are primarily influenced by the labeled samples, which introduce bias toward majority classes. Since Proposition 3 shows that the baseline image has invariance to solid-color intensity, from first principles, we leverage the logits from a black image I to calibrate pruning probabilities. Given the labeled dataset $\chi$ in the t-th epoch, a class-aware pruning probability is assigned to each sample based on its score, which is formulated as:

$$
\mathcal { P } _ { t } ^ { l } ( x _ { b } ^ { n } ) = \left\{ \begin{array} { l l } { 1 \quad } & { \mathcal { H } _ { t } ^ { l } ( x _ { b } ^ { n } ) \in H _ { \prec r _ { c } , t } ^ { l } , } \\ { 0 \quad } & { \mathcal { H } _ { t } ^ { l } ( x _ { b } ^ { n } ) \notin H _ { \prec r _ { c } , t } ^ { l } , } \end{array} \right.\tag{21}
$$

where ${ \boldsymbol { H } _ { \prec { \boldsymbol { r } } _ { c } , t } ^ { l } }$ denotes the $r _ { c } \times N _ { c }$ smallest scoring values of the class c and $r _ { c } = \pi _ { \theta } ( \mathcal { T } ) _ { c }$ <sub>c</sub> is the classaware pruning probability. The labeled scoring function $\mathcal { H } _ { t } ^ { l } ( x _ { b } ^ { n } )$ is defined using the supervised loss $\mathcal { L } _ { s u p } ( \bar { x } _ { b } ^ { n } , y _ { b } ^ { n } )$ to quantify sample utility. See more details about Appendix E.1.

Dynamic pruning for unlabeled data. While the distribution of the label of the unlabeled data and its imbalance ratio $\gamma _ { u }$ are unknown. To address the uncertainty and bias of pseudo-labels, we design a label-insensitive soft pruning policy $\mathcal { P } _ { t } ^ { u }$ inspired by (Qin et al., 2024), which introduces randomness and gradient scaling into the pruning process. Specifically, for an unlabeled dataset U at the t-th epoch, a pruning probability is assigned to each sample based on its score, which is formulated as:

$$
\mathcal { P } _ { t } ^ { u } ( u _ { b } ^ { m } ) = \left\{ \begin{array} { r l } { r } & { \mathcal { H } _ { t } ^ { u } ( u _ { b } ^ { m } ) < \bar { \mathcal { H } } _ { t } ^ { m } \mathrm { ~ a n d ~ } p ^ { * } ( u _ { b } ^ { m } ) \ge \tau , } \\ { 0 } & { \mathcal { H } _ { t } ^ { u } ( u _ { b } ^ { m } ) \ge \bar { \mathcal { H } } _ { t } ^ { u } \mathrm { ~ o r ~ } p ^ { * } ( u _ { b } ^ { m } ) < \tau , } \end{array} \right.\tag{22}
$$

where $\bar { \mathcal { H } } _ { t } ^ { u }$ is the adaptive threshold and r is a randomized pruning rate, τ is the confidence threshold τ and $p ^ { * } \mathsf { \tilde { ( } } u _ { b } ^ { m } ) = \operatorname* { m a x } ( \mathsf { s o f t r a x } ( g _ { \theta } ^ { * } ( \alpha ( u _ { b } ^ { m } ) ) ) )$ ) denote the debiased pseudo-label confidence. See more details about Appendix E.2.

## 5 EXPERIMENT

In this section, we conducted comprehensive experiments to verify the effectiveness of the proposed DyTrim on CIFAR10-LT, CIFAR100-LT (Cui et al., 2019), STL10-LT (Kim et al., 2020), and ImageNet-127 (Deng et al., 2009; Huh et al., 2016) datasets. Due to limited space, we defer the detailed experimental settings and additional experiments to the Appendix G.

## 5.1 RESULTS ON CIFAR10/100-LT, STL10-LT AND IMAGENET-LT

Under the consistent condition where $\gamma _ { u }$ is known and matched to $\gamma _ { l } ,$ , the results in Table 1 show that CISSL algorithms consistently outperform their vanilla SSL counterparts by mitigating class imbalance while effectively exploiting unlabeled data. Among them, the proposed DyTrim achieves the best performance across all imbalance ratios. Compared with the state-of-the-art CDMAD, DyTrim improves bACC by 1.2% and GM by 1.4% on average, without incurring additional computational overhead. Furthermore, when integrated into FlexMatch and FreeMatch, DyTrim yields substantial improvements, boosting bACC/GM by 2–3% on average. Table 2 evaluates the methods on CIFAR-100-LT, which involves more classes and a stronger imbalance. The results demonstrate that DyTrim consistently outperforms all competing approaches under this more challenging setting.

Table 1: Comparison of bACC/GM on CIFAR-10-LT under different imbalance ratio $\gamma = \gamma _ { l } = \gamma _ { u } .$ where $\gamma _ { u }$ is assumed to be known. “\*” indicates our own implementation.
<table><tr><td rowspan="2">Base SSL Algorithm</td><td rowspan="2">Debiasing Strategy</td><td colspan="2"> $\gamma = 5 0$ </td><td colspan="2"> $\gamma = \mathbf { 1 0 0 }$ </td><td colspan="2"> $\gamma = \mathbf { 1 5 0 }$ </td></tr><tr><td> $\mathbf { b A C C }$ </td><td>GM</td><td>bACC</td><td>GM</td><td> $\mathsf { b A C C }$ </td><td>GM</td></tr><tr><td colspan="2">Vanilla</td><td> $6 5 . 2 \pm 0 . 0 5$ </td><td> $6 1 . 1 \pm 0 . 0 9$ </td><td> $5 8 . 8 \pm 0 . 1 3 $ </td><td> $5 8 . 2 \pm 0 . 1 1 $ </td><td> $5 5 . 6 \pm 0 . 4 3$ </td><td> $4 4 . 0 \pm 0 . 9 8$ </td></tr><tr><td colspan="2" rowspan="2">Re-sampling LDAM-DRW</td><td> $6 4 . 3 \pm 0 . 4 8 $ </td><td> $6 0 . 6 \pm 0 . 6 7$ </td><td> $5 5 . 8 \pm 0 . 4 7$ </td><td> $4 5 . 1 \pm 0 . 3 0$ </td><td> $5 2 . 2 \pm 0 . 0 5$ </td><td> $3 8 . 2 \pm 1 . 4 9$ </td></tr><tr><td> $6 8 . 9 \pm 0 . 0 7 $ </td><td> $6 7 . 0 { \pm } 0 . 0 8 $ </td><td> $6 2 . 8 \pm 0 . 1 7$ </td><td> $5 8 . 9 \pm 0 . 6 0 $ </td><td> $5 7 . 9 \pm 0 . 2 0 $ </td><td> $5 0 . 4 \pm 0 . 3 0$ </td></tr><tr><td colspan="2">cRT</td><td> $6 7 . 8 \pm 0 . 1 3$ </td><td> $6 6 . 3 \pm 0 . 1 5$ </td><td> $6 3 . 2 \pm 0 . 4 5$ </td><td> $5 9 . 9 { \pm } 0 . 4 0 $ </td><td> $5 9 . 3 \pm 0 . 1 0 $ </td><td> $5 4 . 6 \pm 0 . 7 2$ </td></tr><tr><td rowspan="10">FixMatch</td><td>FixMatch</td><td> $7 9 . 2 \pm 0 . 3 3$ </td><td> $7 7 . 8 \pm 0 . 3 6$ </td><td> $7 1 . 5 { \pm } 0 . 7 2 $ </td><td> $6 6 . 8 \pm 1 . 5 1$ </td><td> $6 8 . 4 \pm 0 . 1 5$ </td><td> $5 9 . 9 \pm 0 . 4 3 $ </td></tr><tr><td> $\mathrm { D A R P + c R T }$ </td><td> $8 5 . 8 \pm 0 . 4 3$ </td><td> $8 5 . 6 \pm 0 . 5 6 $ </td><td> $8 2 . 4 \pm 0 . 2 6$ </td><td> $8 1 . 8 \pm 0 . 1 7$ </td><td> $7 9 . 6 \pm 0 . 4 2$ </td><td> $7 8 . 9 \pm 0 . 3 5$ </td></tr><tr><td>CReST+LA</td><td> $8 5 . 6 \pm 0 . 3 6$ </td><td> $8 1 . 9 \pm 0 . 4 5$ </td><td> $8 1 . 2 \pm 0 . 7 0 $ </td><td> $7 4 . 5 \pm 0 . 9 9$ </td><td> $7 1 . 9 \pm 2 . 2 4$ </td><td> $6 4 . 4 \pm 1 . 7 5$ </td></tr><tr><td>ABC</td><td> $8 5 . 6 \pm 0 . 2 6$ </td><td> $8 5 . 2 \pm 0 . 2 9$ </td><td> $8 1 . 1 \pm 1 . 1 4$ </td><td> $8 0 . 3 \pm 1 . 2 9$ </td><td> $7 7 . 3 \pm 1 . 2 5$ </td><td> $7 5 . 6 \pm 1 . 6 5$ </td></tr><tr><td>CoSSL</td><td> $8 6 . 8 \pm 0 . 3 0$ </td><td> $8 6 . 6 \pm 0 . 2 5$ </td><td> $8 3 . 2 \pm 0 . 4 9$ </td><td> $8 2 . 7 \pm 0 . 6 0$ </td><td> $8 0 . 3 \pm 0 . 5 5$ </td><td> $7 9 . 6 \pm 0 . 5 7$ </td></tr><tr><td> $\mathrm { S A W + L A }$ </td><td> $8 6 . 2 \pm 0 . 1 5$ </td><td> $8 3 . 9 \pm 0 . 3 5$ </td><td> $8 0 . 7 \pm 0 . 1 5$ </td><td> $7 7 . 5 \pm 0 . 2 1$ </td><td> $7 3 . 7 \pm 0 . 0 6$ </td><td> $7 1 . 2 \pm 0 . 1 7$ </td></tr><tr><td>Adsh</td><td> $8 3 . 4 \pm 0 . 0 6$ </td><td> $8 2 . 9 \pm 0 . 1 3$ </td><td> $7 6 . 5 \pm 0 . 3 5$ </td><td> $7 4 . 8 \pm 0 . 3 4$ </td><td> $7 1 . 5 { \pm } 0 . 3 0 $ </td><td> $6 8 . 8 \pm 0 . 3 5$ </td></tr><tr><td>DebiasPL</td><td> $8 5 . 6 \pm 0 . 2 0$ </td><td> $8 5 . 2 \pm 0 . 2 3 $ </td><td> $8 0 . 6 \pm 0 . 5 0 $ </td><td> $7 9 . 9 \pm 0 . 5 7$ </td><td> $7 6 . 6 \pm 0 . 1 2$ </td><td> $7 5 . 8 \pm 0 . 7 1$ </td></tr><tr><td>UDAL</td><td> $8 6 . 5 \pm 0 . 2 9$ </td><td> $8 6 . 2 \pm 0 . 2 6 $ </td><td> $8 1 . 4 \pm 0 . 3 9$ </td><td> $8 0 . 6 \pm 0 . 3 8$ </td><td> $7 7 . 9 \pm 0 . 3 3$ </td><td> $7 5 . 8 \pm 0 . 7 1$ </td></tr><tr><td>L2AC CDMAD</td><td> $8 6 . 6 \pm 0 . 3 1$ </td><td> $8 6 . 7 \pm 0 . 3 0$ </td><td> $8 2 . 1 \pm 0 . 5 7$ </td><td> $8 1 . 5 { \pm } 0 . 6 4$ </td><td> $7 7 . 6 \pm 0 . 5 3$ </td><td> $7 5 . 8 \pm 0 . 7 1$ </td></tr><tr><td></td><td>DyTrim</td><td> $8 7 . 3 { \pm } 0 . 1 2 $   ${ \bf 8 8 . 0 \pm 0 . 3 1 }$  </td><td> $8 7 . 0 { \pm } 0 . 1 5 $   ${ \bf 8 7 . 8 \pm 0 . 3 2 }$ </td><td> $8 3 . 6 \pm 0 . 4 6$   ${ \bf 8 4 . 8 \pm 0 . 4 8 }$  </td><td> $8 3 . 1 \pm 0 . 5 7$   ${ \bf 8 4 . 4 \pm 0 . 5 1 }$ </td><td> $8 0 . 8 \pm 0 . 8 6$   ${ \bf 8 2 . 0 \pm 0 . 0 9 }$ </td><td> $7 9 . 9 \pm 1 . 0 7$   ${ \bf 8 1 . 3 \pm 0 . 0 3 }$ </td></tr><tr><td rowspan="3">FlexMatch</td><td>FlexMatch*</td><td> $7 2 . 6 \pm 0 . 7 2$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CDMAD*</td><td> $7 4 . 4 \pm 0 . 8 2$ </td><td> $7 0 . 2 \pm 0 . 8 8$   $7 3 . 0 { \pm } 1 . 1 2 $ </td><td> $6 7 . 7 \pm 0 . 7 3$   $6 8 . 4 \pm 0 . 4 6 $ </td><td> $6 3 . 6 \pm 1 . 2 7$ </td><td> $6 2 . 6 \pm 0 . 6 3$   $6 7 . 0 { \pm } 0 . 5 2 $ </td><td> $5 6 . 1 \pm 1 . 1 3$ </td></tr><tr><td>DyTrim</td><td> $7 7 . 2 \pm 0 . 4 2$ </td><td> $7 6 . 2 \pm 0 . 4 4$ </td><td> $\mathbf { 7 0 . 7 \pm 0 . 4 9 }$  </td><td> $6 6 . 8 \pm 0 . 5 3$   ${ \bf 6 7 . 8 \pm 0 . 7 0 }$ </td><td> ${ \bf 6 8 . 6 \pm 0 . 2 2 }$ </td><td> $6 3 . 2 \pm 0 . 4 4$   ${ \bf 6 6 . 3 \pm 0 . 0 7 }$ </td></tr><tr><td rowspan="3">FreeMatch</td><td>FreeMatch*</td><td> $7 1 . 9 \pm 0 . 2 4$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathrm { C D M A D ^ { * } }$ </td><td> $7 4 . 7 \pm 0 . 6 4$ </td><td> $6 9 . 4 \pm 0 . 6 1$   $7 3 . 6 \pm 1 . 2 3$ </td><td> $6 5 . 7 \pm 0 . 1 8$   $6 9 . 9 { \pm } 0 . 6 5$ </td><td> $6 0 . 9 \pm 0 . 6 9$   $6 8 . 2 \pm 0 . 7 4 $ </td><td> $6 2 . 5 { \pm } 0 . 1 2 $   $6 6 . 2 \pm 0 . 2 7$ </td><td> $5 7 . 3 \pm 0 . 5 3 $   $6 3 . 2 \pm 0 . 4 4$ </td></tr><tr><td> $\mathbf { D } \mathbf { y } \mathbf { T r i m }$ </td><td> ${ \bf 7 6 . 9 \pm 0 . 4 5 }$ </td><td> ${ \bf 7 5 . 9 \pm 0 . 5 2 }$ </td><td> $7 2 . 3 \pm 0 . 1 2$  </td><td> ${ \bf 7 1 . 4 \pm 0 . 5 7 }$ </td><td> ${ \bf 6 9 . 4 \pm 0 . 3 5 }$ </td><td> ${ \bf 6 7 . 5 \pm 0 . 6 3 }$ </td></tr></table>

As shown in Table 3, DyTrim consistently outperforms prior techniques such as CDMAD on the large-scale ImageNet-LT benchmark (Liu et al., 2019), demonstrating its complementary benefits rather than merely overlapping with existing rebalancing approaches. See more details about large-resolution $( 2 2 4 ~ \times ~ 2 2 4 )$ in $\mathsf { A p - }$ pendix H.4. Under the inconsistent condition where $\gamma _ { u }$ was unknown and mismatched to $\gamma _ { l } ,$ , the results in Table 4 show that DyTrim remains the most effective method overall. When the labeled and unlabeled data distributions deviate, DyTrim consistently outperforms CDMAD on both CIFAR-10- LT and STL-10-LT. See more details in Appendix H.2 for the dynamics of baseline image logits during training.

Table 2: Comparison of bACC on CIFAR-100-LT under different imbalance ratio, where $\gamma _ { u }$ is assumed to be known. “\*” indicates our own implementation.
<table><tr><td rowspan=1 colspan=2>Base SSLAlogrithm</td><td rowspan=1 colspan=2>DebiasingStrategy</td><td rowspan=1 colspan=1> $\gamma = 2 0$      $\gamma = 5 0$     $\gamma = 1 0 0$ </td></tr><tr><td rowspan=10 colspan=2>FixMatch</td><td rowspan=5 colspan=2>FixMatchDARPDARP+cRTCReST</td><td rowspan=1 colspan=1>49.6±0.78   $4 2 . 1 \pm 0 . 3 3$    $3 7 . 6 \pm 0 . 4 8$ </td></tr><tr><td rowspan=1 colspan=1>50.8 ±0.77   $4 3 . 1 \pm 0 . 5 4$    $3 8 . 3 \pm 0 . 4 7$ </td></tr><tr><td rowspan=1 colspan=1>51.4 ±0.68   $4 4 . 9 \pm 0 . 5 4$    $4 0 . 4 \pm 0 . 7 8$ </td></tr><tr><td rowspan=1 colspan=1> $5 1 . 8 \pm 0 . 1 2$    $4 4 . 9 \pm 0 . 5 0 $   $4 0 . 1 \pm 0 . 6 5$ </td></tr><tr><td rowspan=2 colspan=2>ABC</td><td rowspan=1 colspan=1> $5 2 . 9 \pm 0 . 0 7$    $4 7 . 3 \pm 0 . 1 7$    $4 2 . 7 \pm 0 . 7 0$ </td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $5 3 . 3 \pm 0 . 7 9$   46.7±0.26  $4 1 . 2 \pm 0 . 0 6$ </td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>CoSSL</td><td rowspan=1 colspan=1>53.9±0.78  47.6±0.26   $4 3 . 0 { \pm } 0 . 6 1$ </td></tr><tr><td rowspan=1 colspan=2>UDAL</td><td rowspan=1 colspan=1>54.1 ±0.23  48.0±0.56   $4 3 . 7 \pm 0 . 4 1$ </td></tr><tr><td rowspan=1 colspan=2>CPE</td><td rowspan=1 colspan=1>52.4±0.17   $4 5 . 6 \pm 0 . 6 8$   $3 9 . 9 \pm 0 . 4 0 $ </td></tr><tr><td rowspan=1 colspan=2>CDMAD</td><td rowspan=1 colspan=1>54.3 ±0.44   $4 8 . 8 \pm 0 . 7 5$   $4 4 . 1 \pm 0 . 2 9$ </td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>DyTrim</td><td rowspan=1 colspan=1>55.5 ±0.53  ${ \bf 5 0 . 8 \pm 0 . 8 0 }$   ${ \pm } 4 . 8 \pm 0 . 2 7$ </td></tr><tr><td rowspan=2 colspan=2>FlexMatch</td><td rowspan=2 colspan=2>FlexMatch*CDMAD*</td><td rowspan=1 colspan=1> $3 6 . 5 \pm 0 . 5 1 $    $2 9 . 6 \pm 0 . 3 5$    $2 5 . 8 \pm 0 . 7 9$ </td></tr><tr><td rowspan=1 colspan=1> $3 9 . 2 \pm 0 . 4 7$    $3 1 . 9 { \pm } 0 . 4 6 $   $2 7 . 0 { \pm } 0 . 6 6 $ </td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>DyTrim</td><td rowspan=1 colspan=1> ${ \bf 4 0 . 9 \pm 0 . 0 9 }$   $3 3 . 5 \pm 0 . 2 1 $    ${ \bf 2 9 . 8 \pm 0 . 6 7 }$ </td></tr><tr><td rowspan=1 colspan=2>FreeMatch</td><td rowspan=1 colspan=2>FreeMatch*CDMAD*</td><td rowspan=1 colspan=1> $3 5 . 9 \pm 0 . 6 9$  31.3 ±0.65   $2 4 . 5 \pm 0 . 6 6$  $3 6 . 9 \pm 0 . 9 6$  32.8±0.93   $2 8 . 0 { \pm } 0 . 6 8 $ </td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>DyTrim</td><td rowspan=1 colspan=1> ${ \bf 3 9 . 0 \pm 0 . 6 1 }$    $3 3 . 4 \pm 0 . 7 0$   ${ \bf 2 9 . 8 \pm 0 . 0 9 }$ </td></tr></table>

## 5.2 RESULTS ON VIT BACKBONES

Table 3: Comparison of bACC on ImageNet-LT.

In addition, Table 5 highlights the performance of various algorithms under both consistent and inconsistent imbalance settings with ViT backbones. On CIFAR-10-LT, DyTrim yields the best results, improving bACC 0.6% over CDMAD and nearly 4% over FixMatch when $\gamma _ { l } = \gamma _ { u } = 1 0 0$ . Under the inconsistent condition, DyTrim maintains a clear margin, surpassing CDMAD almost 2%. On CIFAR-100-LT, although the absolute accuracies are lower due to the increased diffi-

<table><tr><td>Algorithm</td><td>ImageNet-LT</td></tr><tr><td> $\overline { { { \mathrm { F i x M a t c h } } ^ { * } } }$ </td><td>20.0</td></tr><tr><td> $\mathrm { w / + C D M A D ^ { * } }$ </td><td>35.4</td></tr><tr><td> $\mathrm { w } / { + } \mathrm { D y T r i m }$ </td><td>37.2</td></tr></table>

Table 4: Comparison of bACC/GM on CIFAR-10-LT and STL-10-LT under different imbalance ratio $\gamma _ { l } \neq \gamma _ { u } ,$ where $\gamma _ { u }$ is assumed to be unknown. “\*” indicates our own implementation.
<table><tr><td rowspan="2">Base SSL Algorithm</td><td rowspan="2">Debiasing Strategy</td><td colspan="4"> $\mathrm { C I F A R - 1 0 - L T } \left( \gamma _ { l } = 1 0 0 , \gamma _ { u } = \mathrm { U n k n o w n } \right)$ </td><td colspan="4"> $\mathrm { S T L - 1 0 \mathrm { - } L T } \left( \gamma _ { u } = \mathrm { U n k n o w n } \right)$ </td></tr><tr><td> $\overline { { \gamma _ { u } = 5 0 } }$  bACC</td><td>GM</td><td> $\overline { { \gamma _ { u } = 1 5 0 } }$  bACC</td><td>GM</td><td> $\overline { { \gamma _ { l } = 1 0 } }$  bACC</td><td>GM</td><td> $\overline { { \gamma _ { l } = 2 0 } }$  bACC</td><td>GM</td></tr><tr><td rowspan="9">FixMatch</td><td rowspan="9">FixMatch DARP DARP+LA</td><td> $7 3 . 9 \pm 0 . 2 5$ </td><td> $7 0 . 5 \pm 0 . 5 2$ </td><td> $6 9 . 6 \pm 0 . 6 0$ </td><td> $6 2 . 6 \pm 1 . 1 1$ </td><td> $7 2 . 9 \pm 0 . 0 9$ </td><td>69.6±0.01</td><td> $6 3 . 4 \pm 0 . 2 1 $ </td><td> $5 2 . 6 \pm 0 . 0 9$ </td></tr><tr><td> $7 7 . 3 \pm 0 . 1 7$ </td><td> $7 5 . 5 \pm 0 . 2 1$ </td><td> $7 2 . 9 \pm 0 . 2 4$ </td><td> $6 9 . 5 \pm 0 . 1 8 $ </td><td></td><td> $7 6 . 5 \pm 0 . 4 0$ </td><td></td><td></td></tr><tr><td> $8 2 . 3 \pm 0 . 3 2 $ </td><td></td><td></td><td></td><td> $7 7 . 8 \pm 0 . 3 3$ </td><td></td><td> $6 9 . 9 \pm 1 . 7 7$ </td><td> $6 5 . 4 \pm 3 . 0 7$ </td></tr><tr><td> $8 2 . 7 \pm 0 . 2 1$ </td><td> $8 1 . 5 \pm 0 . 2 9$   $8 2 . 3 \pm 0 . 2 5$ </td><td> $7 8 . 9 \pm 0 . 2 3 $   $8 0 . 7 \pm 0 . 4 4$ </td><td> $7 7 . 7 \pm 0 . 0 6$ </td><td> $7 8 . 6 \pm 0 . 3 0$ </td><td> $7 7 . 4 \pm 0 . 4 0$ </td><td> $7 1 . 9 \pm 0 . 4 9$ </td><td> $6 8 . 7 \pm 0 . 5 1 $ </td></tr><tr><td> $\mathrm { D A R P + c R T }$ </td><td> $8 2 . 7 \pm 0 . 6 4 $   $8 2 . 0 \pm 0 . 7 6$ </td><td></td><td> $8 0 . 2 \pm 0 . 6 1$ </td><td> $7 9 . 3 \pm 0 . 2 3$ </td><td> $7 8 . 7 \pm 0 . 2 1 $ </td><td> $7 4 . 1 \pm 0 . 6 1$ </td><td> $7 3 . 1 \pm 1 . 2 1$ </td></tr><tr><td>ABC</td><td> $7 9 . 1 \pm 0 . 3 2$ </td><td> $7 8 . 4 \pm 0 . 8 7$   $7 4 . 5 \pm 0 . 9 7$ </td><td> $7 7 . 2 \pm 1 . 0 7$ </td><td> $7 9 . 1 \pm 0 . 4 6$ </td><td> $7 8 . 1 \pm 0 . 5 7$ </td><td> $7 3 . 8 \pm 0 . 1 5$ </td><td> $7 2 . 1 \pm 0 . 1 5$ </td></tr><tr><td>SAW</td><td> $7 9 . 8 \pm 0 . 2 5$ </td><td>79.1 ±0.81</td><td> $7 2 . 5 \pm 1 . 3 7$ </td><td> $7 8 . 3 \pm 0 . 2 5$ </td><td> $7 7 . 0 { \pm } 0 . 1 9$ </td><td> $7 1 . 9 \pm 0 . 8 1$ </td><td> $6 9 . 0 \pm 0 . 8 1 $ </td></tr><tr><td>SAW+LA</td><td> $8 2 . 9 \pm 0 . 3 8 $   $8 1 . 6 \pm 0 . 3 8$ </td><td> $8 2 . 6 \pm 0 . 3 8$   $8 1 . 3 \pm 0 . 3 2 $ </td><td> $7 8 . 6 \pm 0 . 9 1$ </td><td> $7 9 . 4 \pm 0 . 2 6$ </td><td> $7 8 . 4 \pm 0 . 1 7$ </td><td> $7 3 . 9 \pm 0 . 9 1 $ </td><td> $7 1 . 8 \pm 0 . 9 9$ </td></tr><tr><td>SAW+cRT CPE</td><td> $8 6 . 2 \pm 0 . 2 6 $   $8 5 . 9 \pm 0 . 3 3 $ </td><td>77.6±0.40 82.4 ±0.49</td><td> $7 7 . 1 \pm 0 . 4 1$   $8 2 . 1 \pm 0 . 5 3 $ </td><td> $7 8 . 9 \pm 0 . 2 2$   $7 9 . 0 \pm 0 . 0 5$ </td><td> $7 7 . 8 \pm 0 . 1 4$   $7 8 . 7 \pm 0 . 5 4$ </td><td> $7 2 . 3 \pm 0 . 8 6$   $7 7 . 0 \pm 0 . 7 3$ </td><td> $6 9 . 5 \pm 0 . 8 3 $   $7 6 . 1 \pm 0 . 6 8$ </td></tr><tr><td rowspan="3"></td><td>CDMAD</td><td> $8 5 . 7 \pm 0 . 3 6 $ </td><td> $8 5 . 3 \pm 0 . 3 8$ </td><td> $8 2 . 3 \pm 0 . 2 3 $ </td><td> $8 1 . 8 \pm 0 . 2 9$ </td><td> $7 9 . 9 \pm 0 . 2 3 $ </td><td> $7 8 . 9 \pm 0 . 3 8 $ </td><td> $7 5 . 2 \pm 0 . 4 0$ </td><td> $7 3 . 5 \pm 0 . 3 1$ </td></tr><tr><td>DyTrim</td><td> ${ \bf 8 6 . 4 \pm 0 . 4 3 }$ </td><td> ${ \bf 8 6 . 0 \pm 0 . 4 3 }$ </td><td> ${ \bf 8 3 . 8 \pm 0 . 3 4 }$ </td><td> ${ \bf 8 3 . 4 \pm 0 . 3 3 }$ </td><td> $\mathbf { 8 0 . 7 \pm 0 . 6 4 }$ </td><td> $\mathbf { 7 9 . 8 \pm 0 . 7 0 }$ </td><td> $7 7 . 9 \pm 1 . 0 4$ </td><td> ${ \bf 7 6 . 7 \pm 1 . 2 6 }$ </td></tr><tr><td>FlexMatch*</td><td> $6 7 . 7 \pm 0 . 6 7$ </td><td> $6 2 . 8 \pm 0 . 6 5$ </td><td> $6 3 . 0 \pm 0 . 7 7$ </td><td> $5 6 . 3 \pm 1 . 7 0$ </td><td> $6 2 . 1 \pm 0 . 2 9$ </td><td> $6 0 . 8 \pm 0 . 4 3$ </td><td> $5 6 . 9 \pm 0 . 9 0$ </td><td> $5 1 . 4 \pm 0 . 8 1$ </td></tr><tr><td rowspan="3">FlexMatch</td><td>CDMAD*</td><td> $6 9 . 2 \pm 0 . 2 2$ </td><td> $6 7 . 0 \pm 0 . 1 1 $ </td><td> $6 7 . 0 \pm 1 . 6 9$ </td><td> $6 3 . 4 \pm 0 . 9 1$ </td><td> $6 5 . 5 \pm 1 . 0 5$ </td><td> $6 3 . 7 \pm 1 . 0 2 $ </td><td> $6 2 . 4 \pm 1 . 0 5$ </td><td> $6 0 . 5 \pm 0 . 9 9$ </td></tr><tr><td>DyTrim</td><td> $7 2 . 5 \pm 0 . 3 9$ </td><td> ${ \bf 7 0 . 7 \pm 0 . 4 5 }$ </td><td> ${ \bf 7 0 . 3 \pm 1 . 0 1 }$ </td><td> ${ \bf 6 7 . 4 \pm 0 . 2 1 }$ </td><td> ${ \bf 6 8 . 0 \pm 0 . 9 4 }$ </td><td> ${ \bf 6 6 . 4 \pm 0 . 8 5 }$ </td><td> ${ \bf 6 3 . 9 \pm 0 . 1 6 }$ </td><td> ${ \bf 6 1 . 7 \pm 0 . 2 8 }$ </td></tr><tr><td> $_ \mathrm { F r e e M a t c h ^ { * } }$ </td><td> $6 9 . 3 \pm 0 . 9 9$ </td><td> $6 5 . 4 \pm 1 . 4 5$ </td><td> $6 3 . 5 \pm 0 . 7 6$ </td><td> $5 5 . 7 \pm 0 . 7 7$ </td><td> $6 3 . 9 \pm 0 . 7 7$ </td><td> $6 2 . 0 { \pm } 0 . 9 0 $ </td><td> $5 9 . 0 \pm 1 . 4 3 $ </td><td> $5 7 . 6 \pm 0 . 6 7$ </td></tr><tr><td rowspan="3">FreeMatch</td><td> $\mathrm { C D M A D ^ { * } }$ </td><td> $7 1 . 0 { \pm } 0 . 9 8 $ </td><td> $6 9 . 0 \pm 1 . 0 5$ </td><td> $6 7 . 1 \pm 0 . 9 6$ </td><td> $6 4 . 3 \pm 0 . 9 9$ </td><td> $6 6 . 1 \pm 0 . 3 2$ </td><td> $6 3 . 8 \pm 0 . 9 7$ </td><td> $6 1 . 5 \pm 0 . 4 7$ </td><td> $5 9 . 5 \pm 0 . 6 3 $ </td></tr><tr><td>DyTrim</td><td> $7 2 . 3 \pm 0 . 6 9$ </td><td> ${ \bf 7 1 . 1 \pm 1 . 2 3 }$ </td><td> ${ \bf 6 9 . 9 \pm 0 . 1 5 }$ </td><td> ${ \bf 6 7 . 4 \pm 0 . 3 7 }$ </td><td> ${ \bf 6 8 . 0 \pm 0 . 6 4 }$ </td><td> ${ \bf 6 6 . 5 \pm 1 . 2 0 }$ </td><td> ${ \bf 6 4 . 6 \pm 0 . 7 7 }$ </td><td> ${ \bf 6 2 . 7 \pm 1 . 1 6 }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

culty, DyTrim still matches or slightly improves upon CDMAD, while consistently outperforming FixMatch. Additional experimental results are provided in Appendix H.

Table 5: Comparison of bACC/GM on CIFAR-10-LT and CIFAR-100-LT with TinyViT under different imbalance ratio, where $\gamma _ { u }$ is assumed to be known. “\*” indicates our own implementation.
<table><tr><td rowspan="2">Base SSL Algorithm</td><td rowspan="2">Debiasing Strategy</td><td colspan="4">CIFAR-10-LT  $( \gamma _ { l } = 1 0 0 )$ </td><td colspan="2">CIFAR-100-LT  $( \gamma _ { l } = 1 0 0 )$ </td></tr><tr><td> $\overline { { \gamma _ { u } = 1 0 0 } }$ </td><td>GM</td><td> $\overline { { \gamma _ { u } = 1 5 0 } }$ </td><td></td><td> $\overline { { \gamma _ { u } = 1 0 0 } }$ </td><td></td></tr><tr><td rowspan="3">FixMatch</td><td> $\operatorname { F i x M a t c h ^ { * } }$ </td><td> $\mathsf { b A C C }$ </td><td></td><td>bACC</td><td>GM</td><td>bACC</td><td>GM</td></tr><tr><td></td><td> $4 5 . 5 \pm 0 . 1 4$ </td><td> $3 0 . 0 \pm 0 . 4 1$ </td><td> $4 5 . 3 \pm 0 . 1 2$ </td><td> $2 8 . 9 \pm 0 . 9 6$ </td><td> $2 3 . 2 \pm 0 . 1 3$ </td><td> $5 . 7 \pm 0 . 3 3$ </td></tr><tr><td> $\mathrm { C D M A D ^ { * } }$  DyTrim</td><td> $4 8 . 7 \pm 0 . 4 9$   ${ \bf 4 9 . 3 \pm 0 . 4 7 }$ </td><td> $\mathbf { 4 0 . 5 \pm 0 . 2 6 }$   $4 0 . 3 \pm 0 . 3 6$ </td><td> $4 5 . 4 \pm 0 . 1 3$   $\pm 7 . 3 \pm 0 . 1 2$ </td><td> ${ \bf 3 9 . 9 \pm 0 . 1 0 }$   $3 9 . 7 \pm 0 . 5 7$ </td><td> $2 4 . 0 { \pm } 0 . 1 5$   $2 4 . 1 \pm 0 . 2 2$ </td><td> ${ \bf 9 . 0 \pm 0 . 7 7 }$   $8 . 9 \pm 0 . 1 5$ </td></tr></table>

## 5.3 SCALABILITY EVALUATION OF DYTRIM

DyTrim exhibited robust extensibility as a universal plug-in component, consistently boosting performance across diverse SSL frameworks (CDMAD/CCL), datasets (CIFAR/STL10-LT), and imbalance ratios $( \gamma = 1 \sim 1 5 0 )$ , as shown in Table 6. Notably, it achieved up to +1.4% (CDMAD on CIFAR10-LT) and +2.7% (STL10-LT, $\gamma _ { l } = 2 0 )$ gains without architecture-specific tuning, validating its versatility in semi-supervised long-tailed scenarios.

Table 6: Comparison of bACC with two SOTA algorithms with and without DyTrim on CIFAR-10, CIFAR-100, and STL-10. <sup>↓</sup> and <sup>↑</sup> indicate improvements or degradations over the baseline.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Imbalance ratio</td><td colspan="3">FixMatch+</td><td colspan="3">FixMatch+</td></tr><tr><td>CDMAD</td><td>CDMAD+DyTrim</td><td>Gain</td><td>CCL</td><td>CCL+DyTrim</td><td>Gain</td></tr><tr><td rowspan="3">CIFAR10-LT</td><td> $\gamma _ { l } = \gamma _ { u } = 1 0 0$ </td><td> $8 3 . 6 \pm 0 . 4 6$ </td><td>84.8±0.48</td><td>↑1.2</td><td> $8 6 . 2 \pm 0 . 3 5$ </td><td> ${ \bf 8 6 . 7 \pm 0 . 3 9 }$ </td><td>↑0.5</td></tr><tr><td> $\gamma _ { l } = \gamma _ { u } = 1 5 0$ </td><td> $8 0 . 8 \pm 0 . 8 6$ </td><td> ${ \bf 8 2 . 0 \pm 0 . 0 9 }$ </td><td>↑1.2</td><td> $8 4 . 0 { \pm } 0 . 2 1 $ </td><td> $\mathbf { 8 4 . 0 \pm 0 . 2 6 }$ </td><td>↑0.0</td></tr><tr><td> $\gamma _ { l } = 1 0 0 , \gamma _ { u } = 1$ </td><td> $8 7 . 5 \pm 0 . 4 6$ </td><td> ${ \bf 8 8 . 9 \pm 0 . 8 8 }$ </td><td>↑1.4</td><td> $9 3 . 9 { \pm } 0 . 1 2 $ </td><td> ${ \bf 9 4 . 1 \pm 0 . 1 7 }$ </td><td>↑0.2</td></tr><tr><td>CIFAR100-LT</td><td> $\gamma _ { l } = \gamma _ { u } = 2 0$ </td><td> $5 4 . 3 \pm 0 . 4 4$ </td><td>55.5±0.53</td><td>↑1.2</td><td>57.5±0.16</td><td>58.1 ±0.49</td><td>↑0.6</td></tr><tr><td rowspan="2">STL10-LT</td><td> $\gamma _ { l } = 1 0$ </td><td> $7 9 . 9 \pm 0 . 2 3$ </td><td> $\mathbf { 8 0 . 7 \pm 0 . 6 4 }$ </td><td>↑1.2</td><td> $8 4 . 8 \pm 0 . 1 5$ </td><td>85.1 ±0.33</td><td>↑0.3</td></tr><tr><td> $\gamma _ { l } = 2 0$ </td><td> $7 5 . 2 \pm 0 . 4 0$ </td><td> $7 7 . 9 \pm 1 . 0 4$ </td><td>↑2.7</td><td> $8 3 . 1 \pm 0 . 1 8$ </td><td> $\mathbf { 8 3 . 3 \pm 0 . 4 0 }$ </td><td>↑0.2</td></tr></table>

## 6 CONCLUSION

In this work, we provide a theoretical characterization of class bias in LTSSL through an in-depth analysis of the learning dynamics. We derive a step-wise decomposition of logit updates, demonstrating how class imbalance dominates predictions and how debiasing methods, such as logit adjust ment, reweighting, and resampling. Our theoretical insights bridge the gap between existing methods and their effect on gradient dynamics, highlighting the critical role of sample-level interventions. Based on this foundation, we introduce DyTrim, a dynamic pruning framework that mitigates class imbalance by reallocating gradient budgets. Empirical results across multiple benchmarks and SSL methods demonstrate that DyTrim consistently improves performance.

## ACKNOWLEDGEMENT

This work was supported by the Beijing Natural Science Foundation under Grant (No.L231005), and by the National Key Research and Development Program of China under Grant (No.2024YFB3312200). Yue Cheng would like to thank Bochen Lyu, Qianying Tang, and Xiang Wei for the useful discussion.

## REFERENCES

Jean-Michel Attendu and Jean-Philippe Corbeil. Nlu on data diets: Dynamic data subset selection for nlp classification tasks. arXiv preprint arXiv:2306.03208, 2023. 15

David Berthelot, Nicholas Carlini, Ekin D Cubuk, Alex Kurakin, Kihyuk Sohn, Han Zhang, and Colin Raffel. Remixmatch: Semi-supervised learning with distribution alignment and augmentation anchoring. arXiv preprint arXiv:1911.09785, 2019. 1

Leon Bottou. Stochastic gradient descent tricks. In´ Neural networks: tricks of the trade: second edition, pp. 421–436. Springer, 2012. 23

Jiarui Cai, Yizhou Wang, and Jenq-Neng Hwang. Ace: Ally complementary experts for solving long-tailed recognition in one-shot. In Proceedings ofthe IEEE/CVF international conference on computer vision, pp. 112–121, 2021. 15

Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Arechiga, and Tengyu Ma. Learning imbalanced datasets with label-distribution-aware margin loss. Advances in neural information processing systems, 32, 2019. 24

Dingshuo Chen, Zhixun Li, Yuyan Ni, Guibin Zhang, Ding Wang, Qiang Liu, Shu Wu, Jeffrey Xu Yu, and Liang Wang. Beyond efficiency: Molecular data pruning for enhanced generalization. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 15

Hao Chen, Ran Tao, Yue Fan, Yidong Wang, Jindong Wang, Bernt Schiele, Xing Xie, Bhiksha Raj, and Marios Savvides. Softmatch: Addressing the quantity-quality tradeoff in semi-supervised learning. In The Eleventh International Conference on Learning Representations, 2023. 15

Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, and Serge Belongie. Class-balanced loss based on effective number of samples. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pp. 9268–9277, 2019. 8

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009. 8

Guodong Du, Jia Zhang, Ning Zhang, Hanrui Wu, Peiliang Wu, and Shaozi Li. Semi-supervised imbalanced multi-label classification with label propagation. Pattern Recognition, 150:110358, 2024. 15

Yue Fan, Dengxin Dai, Anna Kukleva, and Bernt Schiele. Cossl: Co-learning of representation and classifier for imbalanced semi-supervised learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14574–14584, June 2022. 1, 24

Qianhan Feng, Lujing Xie, Shijie Fang, and Tong Lin. Bacon: Boosting imbalanced semi-supervised learning via balanced feature-level contrastive learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 11970–11978, 2024. 15

Lan-Zhe Guo and Yu-Feng Li. Class-imbalanced semi-supervised learning with adaptive thresholding. In International conference on machine learning, pp. 8082–8094. PMLR, 2022. 15, 24

Xiaoyu Guo, Xiang Wei, Shunli Zhang, Wei Lu, and Weiwei Xing. Dcrp: Class-aware feature diffusion constraint and reliable pseudo-labeling for imbalanced semi-supervised learning. IEEE Transactions on Multimedia, 26:7146–7159, 2024. 24

Yangyang Guo and Mohan Kankanhalli. Scan: Bootstrapping contrastive pre-training for data effi ciency. arXiv preprint arXiv:2411.09126, 2024. 15

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016. 23

Chen Huang, Yining Li, Chen Change Loy, and Xiaoou Tang. Learning deep representation for imbalanced classification. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pp. 5375–5384, 2016. 24

Minyoung Huh, Pulkit Agrawal, and Alexei A Efros. What makes imagenet good for transfer learning? arXiv preprint arXiv:1608.08614, 2016. 8

Arthur Jacot, Franck Gabriel, and Clement Hongler. Neural tangent kernel: Convergence and gen-´ eralization in neural networks. Advances in neural information processing systems, 31, 2018. 3

N JAPKOWICZ. The class imbalance problem: Significance and strategies. In Proc. 2000 International Conference on Artificial Intelligence, volume 1, pp. 111–117, 2000. 2, 24

Bingyi Kang, Saining Xie, Marcus Rohrbach, Zhicheng Yan, Albert Gordo, Jiashi Feng, and Yannis Kalantidis. Decoupling representation and classifier for long-tailed recognition. In International Conference on Learning Representations, 2020. 24

Kenji Kawaguchi and Haihao Lu. Ordered sgd: A new stochastic optimization framework for empirical risk minimization. In International Conference on Artificial Intelligence and Statistics, pp. 669–679. PMLR, 2020. 15

Krishnateja Killamsetty, Sivasubramanian Durga, Ganesh Ramakrishnan, Abir De, and Rishabh Iyer. Grad-match: Gradient matching based data subset selection for efficient deep model training. In International Conference on Machine Learning, pp. 5464–5474. PMLR, 2021. 15

Jaehyung Kim, Youngbum Hur, Sejun Park, Eunho Yang, Sung Ju Hwang, and Jinwoo Shin. Distribution aligning refinery of pseudo-label for imbalanced semi-supervised learning. Advances in neural information processing systems, 33:14567–14579, 2020. 1, 8, 24

Miroslav Kubat. Addressing the curse of imbalanced training sets: one-sided selection. In Proceedings of the 14th international conference on machine learning, pp. 179–186. Morgan Kaufmann, 1997. 24

Zhengfeng Lai, Chao Wang, Henrry Gunawan, Sen-Ching S Cheung, and Chen-Nee Chuah. Smoothed adaptive weighting for imbalanced semi-supervised learning: Improve reliability against unknown distribution data. In Proceedings of the 39th International Conference on Ma chine Learning, volume 162, pp. 11828–11843, 2022. 7, 24

Justin Lazarow, Kihyuk Sohn, Chen-Yu Lee, Chun-Liang Li, Zizhao Zhang, and Tomas Pfister. Unifying distribution alignment as a loss for imbalanced semi-supervised learning. In 2023 IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), pp. 5633–5642. IEEE Computer Society, 2023. 24

Doyup Lee, Sungwoong Kim, Ildoo Kim, Yeongjae Cheon, Minsu Cho, and Wook-Shin Han. Contrastive regularization for semi-supervised learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pp. 3911–3920, June 2022. 15

Hyuck Lee and Heeyoung Kim. Cdmad: Class-distribution-mismatch-aware debiasing for classimbalanced semi-supervised learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 23891–23900, 2024. 1, 6, 20, 24

Hyuck Lee, Seungjae Shin, and Heeyoung Kim. Abc: Auxiliary balanced classifier for classimbalanced semi-supervised learning. In Advances in Neural Information Processing Systems, volume 34, pp. 7082–7094, 2021. 1, 15, 24

Jingyang Li, Jiachun Pan, Vincent Y. F. Tan, Kim chuan Toh, and Pan Zhou. Towards understanding why fixmatch generalizes better than supervised learning. In The Thirteenth International Conference on Learning Representations, 2025. 1

Ziwei Liu, Zhongqi Miao, Xiaohang Zhan, Jiayun Wang, Boqing Gong, and Stella X Yu. Largescale long-tailed recognition in an open world. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2537–2546, 2019. 9, 25

Chengcheng Ma, Ismail Elezi, Jiankang Deng, Weiming Dong, and Changsheng Xu. Three heads are better than one: Complementary experts for long-tailed semi-supervised learning. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 38, pp. 14229–14237, 2024. 15

Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Himanshu Jain, Andreas Veit, and Sanjiv Kumar. Long-tail learning via logit adjustment. In Proceedings of the International Conference on Learning Representations, 2021. 2, 6, 15

Youngtaek Oh, Dong-Jin Kim, and In So Kweon. Daso: Distribution-aware semantics-oriented pseudo-label for imbalanced semi-supervised learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9786–9796, 2022. 24

Ziheng Qin, Kai Wang, Zangwei Zheng, Jianyang Gu, Xiangyu Peng, Daquan Zhou, Lei Shang, Baigui Sun, Xuansong Xie, Yang You, et al. Infobatch: Lossless training speed up by unbiased dynamic data pruning. In The Twelfth International Conference on Learning Representations, 2024. 8, 15, 23, 26

Ravi S Raju, Kyle Daruwalla, and Mikko Lipasti. Accelerating deep learning with dynamic data pruning, 2021. 15

Yi Ren and Danica J. Sutherland. Learning dynamics of LLM finetuning. 2025. URL https: //openreview.net/forum?id=tPNHOoZFl9. 2, 3, 19

Yi Ren, Shangmin Guo, and Danica J. Sutherland. Better supervisory signals by observing learning paths. In International Conference on Learning Representations, 2022. 19

Shiori Sagawa, Pang Wei Koh, Tatsunori B Hashimoto, and Percy Liang. Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generaliza tion. arXiv preprint arXiv:1911.08731, 2019. 15

Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver. Prioritized experience replay. arXiv preprint arXiv:1511.05952, 2015. 15

Kihyuk Sohn, David Berthelot, Chun-Liang Li, Zizhao Zhang, Nicholas Carlini, Ekin D. Cubuk, Alex Kurakin, Han Zhang, and Colin Raffel. Fixmatch: Simplifying semi-supervised learning with consistency and confidence. arXiv preprint arXiv:2001.07685, 2020. 1, 2, 3, 15, 24

Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In Proceedings of the 34th International Conference on Machine Learning, volume 70, pp. 3319– 3328, 06–11 Aug 2017. 4

Truong Thao Nguyen, Balazs Gerofi, Edgar Josafat Martinez-Noriega, Franc¸ois Trahay, and Mohamed Wahib. Kakurenbo: adaptively hiding samples in deep neural network training. Advances in Neural Information Processing Systems, 36:37900–37922, 2023. 15

Renzhen Wang, Xixi Jia, Quanziang Wang, Yichen Wu, and Deyu Meng. Imbalanced semisupervised learning with bias adaptive classifier. In The Eleventh International Conference on Learning Representations, 2023a. 24

Xudong Wang, Zhirong Wu, Long Lian, and Stella X Yu. Debiased learning from naturally imbalanced pseudo-labels. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14627–14637. IEEE, 2022. 15, 24

Yidong Wang, Hao Chen, Qiang Heng, Wenxin Hou, Yue Fan, Zhen Wu, Jindong Wang, Marios Savvides, Takahiro Shinozaki, Bhiksha Raj, Bernt Schiele, and Xing Xie. Freematch: Self-adaptive thresholding for semi-supervised learning. arXiv preprint arXiv:2205.07246, 2023b. 1, 2, 15, 16

Yu-Xiong Wang, Deva Ramanan, and Martial Hebert. Learning to model the tail. In Advances in Neural Information Processing Systems, volume 30, 2017. 2

Chen Wei, Kihyuk Sohn, Clayton Mellina, Alan Yuille, and Fan Yang. Crest: A classrebalancing self-training framework for imbalanced semi-supervised learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10857–10866, 2021. 1, 15

Tong Wei and Kai Gan. Towards realistic long-tailed semi-supervised learning: Consistency is all you need. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3469–3478, 2023. 1, 15, 24

Kan Wu, Jinnian Zhang, Houwen Peng, Mengchen Liu, Bin Xiao, Jianlong Fu, and Lu Yuan. Tinyvit: Fast pretraining distillation for small vision transformers. In European conference on computer vision, pp. 68–85. Springer, 2022. 23

Liuyu Xiang, Guiguang Ding, and Jungong Han. Learning from multiple experts: Self-paced knowledge distillation for long-tailed classification. In 16th European Conference on Computer Vision, ECCV 2020, pp. 247–263. Springer Nature, 2020. 24

Weiwei Xing, Yue Cheng, Hongzhu Yi, Xiaohui Gao, Xiang Wei, Xiaoyu Guo, Yumin Zhang, and Xinyu Pang. Lcgc: Learning from consistency gradient conflicting for class-imbalanced semi-supervised debiasing. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 21697–21706, 2025. 1, 15, 20, 24

Hongzhu Yi, Yue Cheng, Weiwei Xing, and Xiang Wei. Abm: Adaptive bias mitigation for classimbalanced semi-supervised learning. Neurocomputing, 658:131617, 2025. 1

Zhuoran Yu, Yin Li, and Yong Jae Lee. Inpl: Pseudo-labeling the inliers first for imbalanced semisupervised learning. arXiv preprint arXiv:2303.07269, 2023. 15

Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. arXiv preprint arXiv:1605.07146, 2016. 23

Bowen Zhang, Yidong Wang, Wenxin Hou, HAO WU, Jindong Wang, Manabu Okumura, and Takahiro Shinozaki. Flexmatch: Boosting semi-supervised learning with curriculum pseudo labeling. In Advances in Neural Information Processing Systems, volume 34, pp. 18408–18419, 2021. 1, 2, 15, 16

Guibin Zhang, Haonan Dong, Yuchen Zhang, Zhixun Li, Dingshuo Chen, Kai Wang, Tianlong Chen, Yuxuan Liang, Dawei Cheng, and Kun Wang. Gder: Safeguarding efficiency, balancing, and robustness via prototypical graph pruning. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 15

Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412, 2017. 32

Na Zheng, Xuemeng Song, Xue Dong, Aashish Nikhil Ghosh, Liqiang Nie, and Roger Zimmermann. Language-assisted debiasing and smoothing for foundation model-based semi-supervised learning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recogni tion (CVPR), pp. 25708–25717, June 2025. 1

Zi-Hao Zhou, Siyuan Fang, Zi-Jing Zhou, Tong Wei, Yuanyu Wan, and Min-Ling Zhang. Continuous contrastive learning for long-tailed semi-supervised recognition. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 1, 15

## APPENDIX

## A RELATED WORK

## A.1 MORE ABOUT MECHANISMS OF LONG-TAILED DEBIASING

This paper considers learning dynamics to study the debiasing mechanisms of SSL algorithms. We briefly introduce differences between the settings considered here and those in previous works. For debiasing on long-tailed learning, Menon et al. (2021) considered a unified framework for debiasing from the perspective of logits adjustment, which requires statistical label frequency. CCL (Zhou et al., 2024) considered debiasing from an information-theoretical lens. LCGC (Xing et al., 2025) used gradient flow to analyze the debiasing process. However, these methods only elucidate the model’s behavior from an ad hoc perspective. We aim to develop a more comprehensive framework that enables a principle-based lens of the bias generation mechanisms inherent in long-tailed semisupervised learning.

## A.2 MORE ABOUT SEMI-SUPERVISED LEARNING

Modern SSL methods typically integrate diverse strategies for exploiting unlabeled data, such as entropy minimization (Zhou et al., 2024), consistency regularization (Sohn et al., 2020), and contrastive learning (Zhou et al., 2024; Lee et al., 2022). Among them, most SSL approaches rely on selecting reliable pseudo-labels during training. FixMatch (Sohn et al., 2020) adopts a fixed confidence threshold of 0.95, whereas FlexMatch (Zhang et al., 2021) adapts thresholds per class based on learning difficulty and training progress. FreeMatch (Wang et al., 2023b) integrates global and local adjustments with a class-fairness regularizer to promote prediction diversity, while Soft-Match (Chen et al., 2023) employs a soft thresholding scheme that reweights samples to balance quantity and quality. In contrast, our method bypasses threshold tuning altogether and directly enforces class-balanced pseudo-labeling through dynamic pruning.

## A.3 MORE ABOUT LONG-TAILED SEMI-SUPERVISED DEBIASING

Existing debiasing methods for LTSSL dominantly rely on consistent distribution assumptions (Guo & Li, 2022; Lee et al., 2021) and logit adjustment strategies (Wei & Gan, 2023). Notable approaches include CReST (Wei et al., 2021), which focuses on minority classes through selective self-training, and CoSSL (Cai et al., 2021), which balances representations using tail-class feature augmentation. Recent advances, like BaCon (Feng et al., 2024), utilize contrastive learning for bal anced features, while SMCLP (Du et al., 2024) exploits collaborative label-instance correlations, and CPE (Ma et al., 2024) employs multiple expert classifiers. Innovative methods such as InPL (Yu et al., 2023) and DebiasMatch (Wang et al., 2022) move beyond traditional pseudo-labeling; InPL uses energy scores to detect reliable inliers, whereas DebiasMatch applies adaptive debiasing with a marginal loss to reduce long-tailed pseudo-label bias. Despite these advances, LTSSL techniques often demand intricate mechanisms or additional modules (Lee et al., 2021), posing challenges in minimizing bias while maintaining simplicity.

## A.4 MORE ABOUT DYNAMIC DATASET PRUNING

To reduce training cost on datasets, dynamic dataset pruning methods (Chen et al., 2024; Killamsetty et al., 2021; Sagawa et al., 2019; Schaul et al., 2015; Zhang et al., 2024) aim to reduce the number of training iterations while maintaining performance. Existing methods employ a variety of criteria to guide pruning, among which loss-based (Attendu & Corbeil, 2023; Kawaguchi & Lu, 2020; Thao Nguyen et al., 2023) method is the most popular. UCB (Raju et al., 2021) applies the crossentropy loss with exponential moving average (EMA) smoothing to mitigate noise. Infobatch (Qin et al., 2024) randomly prunes low-loss samples and amplifies the gradients of retained ones to preserve the expected gradient. SCAN (Guo & Kankanhalli, 2024) categorizes samples as redundant or ill-matched based on their loss and gradually increases the pruning ratio using cosine annealing. While thsese methods effectively accelerate training and can yield nearly unbiased results, none have explored their potential to mitigate class imbalance in SSL by pruning.

## B MORE BASE SSL ALGORITHMS

## B.1 MORE ABOUT TRAINING LOSSES OF FIXMATCH

Training losses of FixMatch on a minibatch for the labeled set MX and a minibatch for the unlabeled set MU can be expressed as follows:

$$
\mathcal { L } _ { s u p } ( x _ { b } ; \theta ) = \frac { 1 } { B } \sum _ { x _ { b } \in \mathcal { M X } } \mathbf { H } \left( \pi _ { \theta } ( y | \alpha ( x _ { b } ) ) , p _ { b } \right)\tag{23}
$$

with

$$
\mathcal { L } _ { c o n } ( u _ { b } , \hat { q } , \tau ; \theta ) = \frac { 1 } { \mu B } \sum _ { b = 1 } ^ { B } \mathbb { 1 } ( \operatorname* { m a x } ( \hat { q } _ { b } ) \geq \tau ) \mathbf { H } ( P _ { \theta } ( y | A ( u _ { b } ) , \hat { q } _ { b } ) ,\tag{24}
$$

where $\hat { q }$ denote the concatenations of $\hat { q } _ { b } . \mathcal { L } _ { s u p }$ denotes the supervised loss for weakly augmented labeled data points $u _ { b } . \mathcal { L } _ { c o n }$ denotes the consistency regularization loss with the confidence threshold τ .

## B.2 MORE ABOUT FLEXMATCH

To overcome the limitation of FixMatch using a fixed threshold $\tau$ across all classes, Flex-Match (Zhang et al., 2021) introduces the Curriculum Pseudo Labeling (CPL) strategy. The key idea is to dynamically adjust the confidence threshold according to the learning status of each class. Specifically, FlexMatch first predicts the class probability for a weakly augmented unlabeled sample $u _ { b }$ as $q _ { b } = \pi _ { \theta } ( y | \alpha ( u _ { b } ) )$ , and then estimates the learning effect of each class c by $\sigma _ { t } ( c ) , i . e .$ , the number of samples predicted as class c that exceed the fixed threshold τ. After normalization, a ratio coefficient $\beta _ { t } ( c )$ is obtained, which defines the class-adaptive threshold:

$$
T _ { t } ( c ) = \beta _ { t } ( c ) \cdot \tau .\tag{25}
$$

In this way, hard-to-learn classes receive a lower threshold to include more samples in training, while easy-to-learn classes gradually increase their thresholds to ensure pseudo-label quality. The unsupervised loss is defined as:

$$
\mathcal { L } _ { c o n } ( u _ { b } , \hat { q } , T _ { t } ; \theta ) = \frac { 1 } { \mu B } \sum _ { b = 1 } ^ { \mu B } \mathbb { 1 } \left( \operatorname* { m a x } ( q _ { b } ) > T _ { t } ( \arg \operatorname* { m a x } ( q _ { b } ) ) \right) \mathbf { H } ( \hat { q } _ { b } , \pi _ { \theta } ( y | A ( u _ { b } ) ) ) ,\tag{26}
$$

where $\hat { q } _ { b } = \mathrm { a r g m a x } _ { c } q _ { b , c }$ denotes the hard pseudo-label, and $\boldsymbol { \mathcal { A } } ( \cdot )$ is the strong augmentation function. The overall training objective is

$$
\begin{array} { r } { \mathcal { L } _ { t } = \mathcal { L } _ { s u p } + \lambda \mathcal { L } _ { c o n } . } \end{array}\tag{27}
$$

where λ is weighting hyperparameter.

## B.3 MORE ABOUT FREEMATCH

Unlike FixMatch and FlexMatch, which rely on fixed or indirectly adjusted thresholds, FreeMatch (Wang et al., 2023b) proposes Self-Adaptive Thresholding (SAT) that dynamically determines thresholds based on the model’s prediction confidence. Specifically, FreeMatch first estimates a global threshold $\tau _ { t }$ using an exponential moving average (EMA) of model confidence:

$$
\tau _ { t } = \rho \tau _ { t - 1 } + ( 1 - \rho ) \frac { 1 } { \mu B } \sum _ { b = 1 } ^ { \mu B } \operatorname* { m a x } ( q _ { b } ) ,\tag{28}
$$

and further refines it with class-specific local statistics $\tilde { p } _ { t } ( c )$

$$
\tau _ { t } ( c ) = \frac { \tilde { p } _ { t } ( c ) } { \operatorname* { m a x } _ { c ^ { \prime } } \tilde { p } _ { t } ( c ^ { \prime } ) } \cdot \tau _ { t } .\tag{29}
$$

At the early stage of training, thresholds are low to encourage more unlabeled data utilization and faster convergence. As the model becomes more confident, thresholds increase to suppress incorrect pseudo-labels and reduce confirmation bias. The unsupervised loss at iteration t is thus:

$$
\mathcal { L } _ { c o n } ( u _ { b } , \hat { q } , \tau _ { t } ; \theta ) = \frac { 1 } { \mu B } \sum _ { b = 1 } ^ { \mu B } \mathbb { 1 } ( \operatorname* { m a x } ( q _ { b } ) > \tau _ { t } ( \arg \operatorname* { m a x } ( q _ { b } ) ) ) \ \mathbf { H } ( \hat { q } _ { b } , \pi _ { \theta } ( y | \mathcal { A } ( u _ { b } ) ) ) .\tag{30}
$$

In addition, FreeMatch introduces Self-Adaptive Fairness (SAF) regularization $\mathcal { L } _ { f }$ , which dynamically calibrates the prediction distribution to encourage diverse predictions and prevent class collapse during early training. Concretely, let $h _ { t } \in \mathbb { R } ^ { C }$ denotes the normalized class histogram of model predictions at iteration $t ,$ and let $h ^ { \ast } \in \mathbb { R } ^ { C }$ denotes the target distribution (e.g., a uniform distribution). The SAF regularization is defined as

$$
\begin{array} { r } { \mathcal { L } _ { f } = D _ { \mathrm { K L } } ( h _ { t } \parallel h ^ { * } ) , } \end{array}\tag{31}
$$

where $D _ { \mathrm { K L } } ( \cdot | | \cdot )$ is the Kullback–Leibler divergence. The final training objective is:

$$
\mathcal { L } = \mathcal { L } _ { s u p } + w _ { u } \mathcal { L } _ { c o n } + w _ { f } \mathcal { L } _ { f } ,\tag{32}
$$

where $w _ { u }$ and $w _ { f }$ are weighting hyperparameters.

## C PROOF FOR SECTION 3 AND SECTION 4.

## C.1 PROOF OF PROPOSITION 2

Proposition 1. For an labeled (unlabeled) sample $x _ { b } \ ( u _ { b } )$ with target y<sub>b</sub> $( \hat { q } _ { b } ^ { t } = \arg \operatorname* { m a x } _ { c } q _ { b , c } ^ { t } ) ,$ where $q _ { b } ^ { t } = \pi _ { \theta ^ { t } } ( y | \alpha ( u _ { b } ) )$ . The one-step learning dynamics ofSSL decompose as

$$
\begin{array} { r l } & { \Delta \log \pi _ { \theta } ^ { t , \mathrm { s u p } } ( y \mid x _ { o } ; x _ { b } ) = - \eta T ^ { t } ( x _ { o } ) K ^ { t } ( x _ { o } , \alpha ( x _ { b } ) ) \mathcal { G } _ { \mathrm { s u p } } ^ { t } ( \alpha ( x _ { b } ) , y _ { b } ) + \mathcal { O } \left( \eta ^ { 2 } \| \nabla _ { \theta } \mathbf { z } ( \alpha ( x _ { b } ) ) \| _ { \rho p } ^ { 2 } \right) } \\ & { \Delta \log \pi _ { \theta } ^ { t , \mathrm { c o n } } ( y \mid x _ { o } ; u _ { b } ) = - \eta T ^ { t } ( x _ { o } ) K ^ { t } ( x _ { o } , A ( u _ { b } ) ) \mathcal { G } _ { \mathrm { c o n } } ^ { t } ( A ( u _ { b } ) , \hat { q } _ { b } ^ { t } ) + \mathcal { O } \left( \eta ^ { 2 } \| \nabla _ { \theta } \mathbf { z } ( A ( u _ { b } ) ) \| _ { \rho p } ^ { 2 } \right) } \end{array}\tag{6}
$$

where $\mathcal { K } ^ { t } ( x _ { o } , \alpha ( x _ { b } ) )$ and $\mathcal { K } ^ { t } ( x _ { o } , \mathcal { A } ( u _ { b } ) )$ are eNTK evaluations of the logit network $\begin{array} { r l } { \mathbf { z } ( \cdot ) } & { { } = } \end{array}$ $g _ { \theta } ( \cdot ) _ { ; }$ , with different inputs. $\mathcal { G } _ { \mathrm { s u p } } ^ { t } ( \alpha ( x _ { b } ) , y _ { b } ) \ = \ \nabla _ { \mathbf { z } } \mathcal { L } _ { \mathrm { s u p } } ( \alpha ( x _ { b } ) , y _ { b } ) | _ { \mathbf { z } ^ { t } }$ and $\mathcal { G } _ { \mathrm { c o n } } ^ { t } ( \hat { q } _ { b } , \mathcal { A } ( u _ { b } ) ) \ =$ $\nabla _ { \mathbf { z } } \mathcal { L } _ { \mathrm { c o n } } ( \hat { q } _ { b } , \mathcal { A } ( u _ { b } ) ) | _ { \mathbf { z } ^ { t } }$ , respectively.

Proof. We aim to derive the one-step learning dynamics of SSL for both supervised and contrastive terms. Suppose that we want to observe the model’s prediction on an “observing example” $\mathbf { \nabla } ^ { \prime } x _ { o } .$ Starting from Eq. (5), we first approximate log $\pi ^ { t + 1 } ( \dot { y } | x _ { o } )$ using first Taylor expansion (with a slight abuse of notation, we write $\pi ^ { t }$ for $\pi _ { \theta } ^ { t } ) \ d t$

$$
\begin{array} { r } { \log \pi ^ { t + 1 } ( y | x _ { o } ) = \log \pi ^ { t } ( y | x _ { o } ) + < \nabla \log \pi ^ { t } ( y | x _ { o } ) , \theta ^ { t + 1 } - \theta ^ { t } > + \mathcal O ( \| \theta ^ { t + 1 } - \theta ^ { t } \| ^ { 2 } ) . } \end{array}
$$

Then, assuming the model updates its parameters using SGD calculated by a “labeled updating exampl $\mathrm { e } ^ { \prime \prime } \left( x _ { b } , y _ { b } \right)$ and an “unlabeled updating exampl $\mathrm { e } ^ { 3 \cdot } \bar { ( } A ( u _ { b } ) , \hat { q } _ { b } ^ { t } )$

Thus, for for supervised learning dynamics, we have, we have

$$
\begin{array} { r l } & { \Delta \log \pi ^ { t + 1 , s u p } ( y \mid x _ { o } ; x _ { b } ) = \log \pi ^ { t + 1 , s u p } ( y \mid x _ { o } ; x _ { b } ) - \log \pi ^ { t , s u p } ( y \mid x _ { o } ; x _ { b } ) } \\ & { \qquad = \nabla _ { \theta } \log \pi ^ { t } ( y \mid x _ { o } ) \big | _ { \theta ^ { t } } ( \theta ^ { t + 1 } - \theta ^ { t } ) + \mathcal { O } ( \| \theta ^ { t + 1 } - \theta ^ { t } \| ^ { 2 } ) } \end{array}
$$

Assuming this step is driven solely by supervised loss, we plug in the definition of SGD and repeatedly use the chain rule:

$$
\begin{array} { r l } & { \nabla _ { \theta } \log \pi _ { \theta } ^ { t } ( y \mid x _ { o } ) \big \vert _ { \theta ^ { t } } ( \theta ^ { t + 1 } - \theta ^ { t } ) = \nabla _ { \theta } \log \pi _ { \theta } ^ { t } ( x _ { o } ) \big \vert _ { \theta ^ { t } } \left( - \eta \nabla _ { \theta } \mathcal { L } _ { \mathrm { s u p } } ( \alpha ( x _ { b } ) ) \big \vert _ { \theta ^ { t } } \right) ^ { \top } } \\ & { \qquad = \left( \nabla _ { z } \log \pi _ { \theta } ^ { t } ( x _ { o } ) \big \vert _ { z ^ { t } } \nabla _ { \theta } z ^ { t } ( x _ { o } ) \big \vert _ { \theta ^ { t } } \right) \left( - \eta \nabla _ { \theta } \mathcal { L } _ { s u p } ( \alpha ( x _ { b } ) ) \big \vert _ { \theta ^ { t } } \right) } \\ & { = \nabla _ { z } \log \pi _ { \theta } ^ { t } ( x _ { o } ) \big \vert _ { z ^ { t } } \nabla \theta z ^ { t } ( x _ { o } ) \big \vert _ { \theta ^ { t } } \Big ( - \eta \big ( \nabla _ { z } \mathcal { L } _ { \mathrm { s u p } } ( \alpha ( x _ { b } ) ) \big ) \big \vert _ { z ^ { t } } \nabla _ { \theta } z ^ { t } ( \alpha ( x _ { b } ) ) \big ) \Big ) ^ { \top } } \\ & { = - \eta \nabla _ { \mathbf { z } } \log \pi _ { \theta } ^ { t } ( x _ { o } ) \big \vert _ { \mathbf { z } ^ { t } } \Big [ \nabla _ { \theta } \mathbf { z } ^ { t } ( x _ { o } ) \big \vert _ { \theta ^ { t } } \left( \nabla _ { \theta } \mathbf { z } ^ { t } ( \alpha ( x _ { b } ) ) \big \vert _ { \theta ^ { t } } \right) ^ { \top } \Big ] ( \nabla _ { \mathbf { z } } \mathcal { L } _ { \mathrm { s u p } } ( \alpha ( x _ { b } ) ) \big \vert _ { \mathbf { z } ^ { t } } ) ^ { \top } } \\ &  \qquad = - \eta \mathcal  \end{array}
$$

Similarly, for consistency learning dynamics, the only difference is that the update sample is changed from $\alpha ( \boldsymbol { x } _ { b } )$ to $\boldsymbol { \mathcal { A } } ( \boldsymbol { u } _ { b } )$ , and the loss is changed from $\mathcal { L } _ { s u p } \mathrm { ~ t o ~ } \mathcal { L } _ { c o n } ( \mathcal { A } ( u _ { b } ) , \hat { q } _ { b } ^ { t } )$ . Note that $\hat { q } _ { b } ^ { t } = \arg \operatorname* { m a x } _ { c } q _ { b , c } ^ { t }$ is treated as a constant in this small step (stop-grad), so the gradient can still be directly calculated w.r.t. z. Thus,

$$
\boldsymbol { \theta } ^ { t + 1 } = \boldsymbol { \theta } ^ { t } - \eta \nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { c o n } ( \boldsymbol { A } ( u _ { b } ) , \hat { q } _ { b } ^ { t } ) \big | _ { \boldsymbol { \theta } ^ { t } } .
$$

Parallel to the above derivation, we obtain

$$
\begin{array} { r l } & { \Delta \log \pi _ { \theta } ^ { t , c o n } ( y \mid x _ { o } ; u _ { b } ) = - \eta T ^ { t } ( x _ { o } ) \underbrace { \nabla _ { \theta } z ^ { t } ( x _ { o } ) \big \vert _ { \theta ^ { t } } \big ( \nabla _ { \theta } z ^ { t } ( A ( u _ { b } ) ) \big \vert _ { \theta ^ { t } } \big ) ^ { \top } } _ { K ^ { t } ( x _ { o } , A ( u _ { b } ) ) } \underbrace { \nabla _ { z } \mathcal { L } _ { \mathrm { c o n } } ( \widehat { q } _ { b } ^ { t } , A ( u _ { b } ) ) \big \vert _ { z ^ { t } } } _ { \mathcal { G } _ { \mathrm { c o n } } ^ { t } ( A ( u _ { b } ) , \widehat { q } _ { b } ^ { t } ) } } \\ & { \phantom { \Delta p a c e { 2 p c } } + \mathcal { O } \left( \eta ^ { 2 } \vert \nabla _ { \theta } \mathbf { z } ( \mathcal { A } ( u _ { b } ) ) \vert _ { \mathrm { l o p } } ^ { 2 } \right) . } \end{array}
$$

## C.2 PROOF OF PROPOSITION 3

Proposition 2. (Invariance of baseline image under affine normalization) Let $\mathcal { T } = \boldsymbol { k } \cdot \mathbf { 1 } _ { d }$ be a baseline image, where $k \in \{ 0 , 1 , \ldots , 2 5 5 \}$ and $\mathbf { 1 } _ { d } ~ \in ~ \mathbb { R } ^ { d }$ is an all-one vector. Suppose the output of the first hidden transformation is normalized by a normalization layer (e.g., BatchNorm, LayerNorm, InstanceNorm, or GroupNorm) with affine parameters $( W _ { 2 } , \pmb { b } )$ . Then the logits $h ( \mathcal T )$ are independent of k and reduce to

$$
h ( \mathbb { Z } ) = b , \quad \pi _ { \theta } ( \mathbb { Z } ) = \operatorname { S o f t m a x } ( b ) .\tag{9}
$$

Proof. Consider a neural network with two layers: the first layer is a linear transformation, and the second layer is a normalization layer followed by an affine transformation. For an input $\mathcal { T } \in \mathbb { R } ^ { d }$ assume the model has the following structure:

$$
h ^ { ( 1 ) } ( \mathcal { T } ) = \sigma ( W _ { 1 } \mathcal { Z } ) ; \quad h ( \mathcal { T } ) = \mathtt { B a t c h N o r m } ( h ^ { ( 1 ) } ( \mathcal { T } ) ) = \frac { h ^ { ( 1 ) } ( \mathcal { T } ) - \mathbb { E } [ h ^ { ( 1 ) } ( \mathcal { T } ) ] } { \sqrt { \mathrm { V a r } [ h ^ { ( 1 ) } ( \mathcal { T } ) ] + \epsilon } } \cdot W _ { 2 } + b ,
$$

Let the baseline image $\mathcal { T } = \boldsymbol { k } \cdot \mathbf { 1 } _ { d } ,$ where ${ \bf 1 } _ { d }$ is a vector of ones, and $k$ is a scalar. Our goal is to show that the output $h ( \mathcal T )$ for the baseline image is independent of k and depends only on the bias term b. For the baseline image $\mathcal { T } = \boldsymbol { k } \cdot \mathbf { 1 } _ { d }$ , the output of this neural network is:

$$
h ^ { ( 1 ) } ( \mathbb { Z } ) = \sigma ( W _ { 1 } \cdot ( k \cdot \mathbf { 1 } _ { d } ) ) = \sigma ( k \cdot W _ { 1 } \mathbf { 1 } _ { d } ) = \sigma ( k \cdot w ) .
$$

where $\pmb { w } = \pmb { W } _ { 1 } \pmb { 1 } _ { d } \in \mathbb { R } ^ { m }$ , which is a constant vector. We see that the output of the first layer depends on k and the constant vector w, and it is passed through the activation function σ. Now, consider the effect of the BatchNorm layer. For the baseline image $\mathcal { T } = \boldsymbol { k } \cdot \mathbf { 1 } _ { d }$ , since $h ^ { ( 1 ) } ( { \mathcal { T } } ) = \sigma ( k \cdot w )$ is a constant vector, the mean $\mathbb { E } [ h ^ { ( 1 ) } ( \underline { { \tau } } ) ]$ and variance $\mathrm { V a r } [ h ^ { ( 1 ) } ( \mathcal { I } ) ]$ are constants that depend only on w.From first principles, we can set $k = 0$ □

Note that if the input I is random Gaussian noise or a batch mean, the situation would be different.

• Gaussian Noise. Let $\mathcal { T } _ { n } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) \in \mathbb { R } ^ { d }$ be a random Gaussian noise vector. After normalization:

$$
h ( \mathbb { Z } _ { n } ) = \frac { h ^ { ( 1 ) } ( \mathbb { Z } _ { n } ) - \mathbb { E } ( h ^ { ( 1 ) } ( \mathbb { Z } _ { n } ) ) } { \sqrt { V a r [ h ^ { ( 1 ) } ( \mathbb { Z } _ { n } ) ] + \epsilon } } \cdot W _ { 2 } + b
$$

Since the input pixel values are random, the mean and variance of the first-layer output depend on the noise distribution characteristics. These statistics fluctuate with the randomness of the input, in contrast to the baseline image, where the normalized output is solely determined by the bias term b.

• Batch Mean. Let $\begin{array} { r } { \mathcal { T } _ { \mu } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } x _ { i } \in \mathbb { R } ^ { d } } \end{array}$ be the batch mean vector. After normalization, the affine transformation:

$$
h ( \mathbb { Z } _ { \mu } ) = \frac { h ^ { ( 1 ) } ( \mathbb { Z } _ { \mu } ) - \mathbb { E } ( h ^ { ( 1 ) } ( \mathbb { Z } _ { \mu } ) ) } { \sqrt { V a r [ h ^ { ( 1 ) } ( \mathbb { Z } _ { \mu } ) ] + \epsilon } } \cdot W _ { 2 } + b
$$

Unlike Gaussian noise images, the mean input of data within a batch does not contain complete randomness; the mean and variance are relatively stable but still do not solely depend on the b.

## C.3 PROOF OF THEOREM 1

Theorem 1. (Bias as the conditional distribution prior) Assume the model $h ( x )$ as characterized in Eq. (8) is trained using cross-entropy loss:

$$
\mathcal { L } = \mathbb { E } _ { ( x , y ) } \big [ - y ^ { \top } \log S o f t m a x ( h ( x ) ) \big ] .\tag{10}
$$

At a population risk minimizer $( W _ { 2 } ^ { \star } , b ^ { \star } )$ we have

$$
\begin{array} { r } { \hat { p } ^ { \star } ( x ) = P ( y \mid x ) , \qquad \hat { p } ^ { \star } ( \mathcal { T } ) = S o f t m a x \big ( \pmb { b } ^ { \star } \big ) = P \big ( y \big | \frac { h ^ { ( 1 ) } ( \mathcal { T } ) - \mathbb { E } [ h ^ { ( 1 ) } ( \mathcal { T } ) ] } { \sqrt { \mathrm { V a r } [ h ^ { ( 1 ) } ( \mathcal { T } ) ] + \epsilon } } = \mathbf { 0 } \big ) . } \end{array}\tag{11}
$$

For the baseline image I in Proposition 3, the baseline prediction thus coincides with the conditional class distribution at the normalized-zerofeature state, capturing the class prior induced by the longtailed training distribution.

Proof. Consider the two-layer network $\begin{array} { r } { f _ { \theta } ( x ) = \frac { h ^ { ( 1 ) } ( x ) - \mathbb { E } [ h ^ { ( 1 ) } ( x ) ] } { \sqrt { \mathrm { V a r } [ h ^ { ( 1 ) } ( x ) ] + \epsilon } } \cdot \gamma + \beta } \end{array}$ , where $h ^ { ( 1 ) } ( x ) = W _ { 1 } x$ The cross-entropy loss is given by:

$$
\mathcal { L } = \mathbb { E } _ { ( x , y ) } \left[ - y ^ { \top } \log { \mathrm { S o f t m a x } ( h ( x ) ) } \right] .
$$

Minimizing the population risk results in $\hat { p } ^ { \star } ( x ) = \operatorname { S o f t m a x } ( h ( x ) ) = P ( y \mid x )$

For the baseline image I, we analyze the model’s output:

$$
\hat { p } ^ { \star } ( \mathcal { T } ) = \operatorname { S o f t m a x } ( b ^ { \star } ) .
$$

Since $\begin{array} { r } { \frac { h ^ { ( 1 ) } ( \mathcal { T } ) - \mathbb { E } [ h ^ { ( 1 ) } ( \mathcal { T } ) ] } { \sqrt { \operatorname { V a r } [ h ^ { ( 1 ) } ( \mathcal { T } ) ] } } \to 0 } \end{array}$ for a baseline image with no input signal, the model’s output is determined solely by $b ^ { \star }$

Thus, we have:

$$
P \left( y \mid \frac { h ^ { ( 1 ) } ( \mathcal { T } ) - \mathbb { E } [ h ^ { ( 1 ) } ( \mathcal { T } ) ] } { \sqrt { \operatorname { V a r } [ h ^ { ( 1 ) } ( \mathcal { T } ) ] + \epsilon } } = 0 \right) = \operatorname { S o f t m a x } ( \pmb { b } ^ { \star } ) .
$$

Finally, we conclude that the baseline prediction corresponds to the conditional class distribution at the normalized-zero feature state, capturing the class prior induced by the long-tailed distribution.

## C.4 PROOF OF PROPOSITION 4

Proposition 3. Let π = Softma $\mathsf { \Omega } _ { \mathsf { L } } ( z )$ and $z = g _ { \boldsymbol { \theta } } ( x )$ . The one-step dynamics decompose as

$$
\Delta \log \pi ^ { t } ( y \mid \mathcal { Z } ) = - \eta \mathcal { T } ^ { t } ( \mathbb { Z } ) K ^ { t } ( \mathbb { Z } , x ) \mathcal { G } ^ { t } ( x , y ) + \mathcal { O } ( \eta ^ { 2 } \Vert \nabla _ { \theta } z ( x ) \Vert _ { \mathrm { o p } } ^ { 2 } ) ,\tag{13}
$$

where $\mathcal { T } ^ { t } ( \mathcal { T } ) = \nabla _ { z } \log _ { \pi ^ { t } } ( \mathcal { T } ) = I - \mathbf { 1 } \pi _ { \theta ^ { t } } ^ { T } ( \mathcal { T } ) , K ^ { t } ( \mathcal { T } , x ) = ( \nabla _ { \theta } z ( \mathcal { T } ) | _ { \theta ^ { t } } ) ( \nabla _ { \theta } z ( x ) | _ { \theta ^ { t } } ) ^ { T }$ is the empirical neural tangent kernel of the logit network z, and $\mathcal { G } ^ { t } ( x , y ) = \nabla _ { z } \mathcal { L } ( x , y ) \mid _ { z ^ { t } }$

Proof. Inspired by the analysis of the learning dynamic of (Ren et al., 2022; Ren & Sutherland, 2025). In this work, we want to observe the classifier’s prediction on the baseline image I. Starting from Eq (12), we first approximate log $\pi ^ { t + 1 } ( y \mid \mathcal { T } )$ using first-order Talyor expansion, with slightly abused symbols, we use $\pi ^ { t }$ to represent $\pi _ { \theta ^ { t + 1 } } ,$ x to represent labeled sample $x _ { b } ^ { n }$ and u to represent unlabeled sample $u _ { b } ^ { m }$

$$
\begin{array} { r } { \log \pi ^ { t + 1 } ( y | \mathcal { Z } ) = \log \pi ^ { t } ( y | \mathcal { Z } ) + < \nabla \log \pi ^ { t } ( y | \mathcal { Z } ) , \theta ^ { t + 1 } - \theta ^ { t } > + \mathcal { O } ( \| \theta ^ { t + 1 } - \theta ^ { t } \| ^ { 2 } ) } \end{array}
$$

Then, assuming the model updates its parameters using SGD calculated by an “updating labeled exampl $\underline { { \circ } } ^ { , , } \left( x , y \right)$ or an “updating unlabeled examp $\mathrm { e } ^ { \prime \prime } \bar { u , }$ we can rearrange the terms in the above equation to get the following expression:

$$
\Delta \log \pi ^ { t } ( y | \mathcal { T } ) = \log \pi ^ { t + 1 } ( y | \mathcal { T } ) - \log \pi ^ { t + 1 } ( y | \mathcal { T } ) = \nabla _ { \theta } \log \pi ^ { t } ( y | \mathcal { T } ) | _ { \theta ^ { t } } ( \theta ^ { t + 1 } - \theta ^ { t } ) + \mathcal { O } ( \| \theta ^ { t + 1 } - \theta ^ { t } \| ^ { 2 } ) ,
$$

To evaluate the leading term, we first take a labeled sample as an example plug in the definition of SGD, and repeatedly use the chain rule:

$$
\begin{array} { r l } & { \nabla _ { \theta } \log \pi ^ { t } ( y | \mathcal { Z } ) | _ { \theta ^ { t } } ( \theta ^ { t + 1 } - \theta ^ { t } ) = ( \nabla _ { z } \log \pi ^ { t } ( y | \mathcal { Z } ) | _ { z ^ { t } } ) ( - \eta \nabla _ { \theta } \mathcal { L } ( x ) | _ { \theta ^ { t } } ) ^ { T } } \\ & { \qquad = ( \nabla _ { z } \log \pi ^ { t } ( y | \mathcal { Z } ) | _ { z ^ { t } } ) ( - \eta \nabla _ { \theta } \mathcal { L } ( x ) | _ { z ^ { t } } - \nabla _ { \theta } z ^ { t } ( x ) | _ { \theta ^ { t } } ) ^ { T } } \\ & { \qquad = - \eta \nabla _ { z } \log \pi ^ { t } ( \mathcal { Z } ) | _ { z _ { t } } [ \nabla _ { \theta } z ( \mathcal { Z } ) | _ { \theta ^ { t } } ( \nabla _ { \theta } z ( x ) | _ { \theta ^ { t } } ) ^ { T } ] ( \nabla _ { z } \mathcal { L } ( x ) | _ { z ^ { t } } ) ^ { T } } \\ & { \qquad = - \eta \mathcal { T } ^ { t } ( \mathcal { Z } ) K ^ { t } ( \mathcal { Z } , x ) \mathcal { G } ^ { t } ( x , y ) } \end{array}\tag{33}
$$

## C.5 MORE ABOUT ANALYZING THE DYNAMICS OF THE LOGITS DEBIASING ALGORITHM

## C.5.1 PER-STEP DECOMPOSITION OF RESAMPLING

Resampling is another widely used strategy for mitigating class imbalance in long-tail semisupervised learning. Instead of modifying the loss, resampling adjusts the data distribution by altering the frequency with which each class is drawn. Let $\mathbb { P } _ { \mathrm { r s } } ( x \in \mathfrak { c } ) = r ^ { { \mathfrak { c } } }$ denote the (possibly normalized) sampling ratio for class c, which determines the probability of selecting samples from that class during training. Then the per-step update of the log-posterior under resampling becomes

$$
\Delta \log \pi _ { \theta } ^ { t , \mathrm { r s } } ( y \mid \mathcal { Z } ; x ) = - \eta \mathcal { T } ^ { t } ( \mathcal { Z } ) \tilde { K } _ { r s } ^ { t } ( \mathcal { Z } , x ; r ^ { c } ) \tilde { \mathcal { G } } _ { r s } ^ { t } ( x , y ; r ^ { c } ) + \mathcal { O } \big ( \eta ^ { 2 } \| \nabla _ { \theta } \mathbf { z } ( x ) \| _ { \mathrm { \mathrm { o p } } } ^ { 2 } \big ) ,\tag{34}
$$

where $\tilde { \mathcal { K } } _ { r s } ^ { t } ( \mathcal { T } , x ; r ^ { c } ) ~ = ~ \mathbb { E } _ { x \sim r ^ { c } } [ \mathcal { K } ^ { t } ( \mathcal { T } , x ) ] , ~ \tilde { \mathcal { G } } _ { r s } ^ { t } ( x , y ; r ^ { c } ) ~ = ~ \mathbb { E } _ { x \sim r ^ { c } } [ \mathcal { G } ^ { t } ( x , y ) ]$ . This decomposition highlights that resampling influences learning solely through changing the expectation measure. The modified kernel $\tilde { \mathcal { K } } _ { r s } ^ { t }$ reshapes how training samples transfer influence to the test input, while the modified residual term $\tilde { \mathcal { G } } _ { r s } ^ { t }$ reweights the magnitude of each update. Increasing the sampling ratio of tail classes therefore amplifies their effective contribution at every step, accelerating their representation and decision boundary updates to match those of head classes, i.e. offering a direct dynamical explanation for the effectiveness of resampling in long-tail regimes.

## C.5.2 PER-STEP DECOMPOSITION OF CDMAD

In this section, we use the loss function of a specific method in logits adjustment, CDMAD (Lee & Kim, 2024), as a case study and integrate it into the learning dynamics framework we propose. The consistency loss of CDMAD as:

$$
\mathcal { L } _ { c o n } ( u _ { b } , \hat { q } , \tau ; \theta ) = \frac { 1 } { \mu B } \sum _ { b = 1 } ^ { B } \mathbb { 1 } ( \operatorname* { m a x } ( \hat { q } _ { b } ) \geq \tau ) \mathbf { H } ( P _ { \theta } ( y | A ( u _ { b } ) , q _ { b } ^ { * } ) ,\tag{35}
$$

where H is cross-entropy loss, $q _ { b } ^ { * } = \arg \operatorname* { m a x } ( \pi _ { \theta } ( y | \alpha ( u _ { b } ) ) - \pi _ { \theta } ( y | \mathcal { T } ) )$ . Our framework reveals that CDMAD operates through two complementary dynamical mechanisms:

$$
\begin{array} { c } { { \Delta \log \pi _ { \theta } ^ { t } ( y \mid \mathcal { T } ) = - \eta \mathcal { T } ^ { t } ( \mathbb { Z } ) ( K ^ { t } ( \mathbb { Z } , \alpha ( x _ { b } ) ) \mathcal { G } _ { \mathrm { s u p } } ^ { t } ( \alpha ( x _ { b } ) , y _ { b } ) + } } \\ { { \mathcal { K } ^ { t } ( \mathbb { Z } , \mathcal { A } ( u _ { b } ) ) \mathcal { G } _ { \mathrm { c o n } } ^ { t } ( \mathcal { A } ( u _ { b } ) , \alpha ( x _ { b } ) ) ) + \mathcal { O } ^ { 2 } } } \end{array}\tag{36}
$$

According to the analysis of Xing et al. (2025), $\mathcal { G } ^ { t }$ using the baseline image enhances the balance of the base SSL model implicitly utilizing the integrated gradient flow $\nabla _ { \theta } { \mathcal { L } } _ { \mathrm { C o n } } \ =$ $\textstyle \sum _ { b } \left( \sum _ { i = 1 } ^ { d } \sum _ { \lambda = 1 } ^ { d } \right.$ IntegratedGrads $\begin{array} { r } { \mathbf { \sigma } _ { ; } ( u _ { b } ) \Big ) \nabla g _ { b } + \sum _ { b } q _ { A , b } \frac { \partial q _ { A , b } } { \partial \theta } } \end{array}$ . We now place $\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { C o n } }$ directly into $\mathcal { G } _ { \mathrm { c o n } } ^ { t }$ to capture the influence of the consistency loss on the model’s update dynamics. The updated $\mathcal { G } _ { \mathrm { c o n } } ^ { t }$ is:

$$
\mathcal { G } _ { \mathrm { c o n } } ^ { t } ( \mathcal { A } ( u _ { b } ) , \alpha ( x _ { b } ) ) = \sum _ { b } \left( \sum _ { i = 1 } ^ { d } \mathrm { I n t e g r a t e d G r a d s } _ { i } ( u _ { b } ) \right) \nabla g _ { b } + \sum _ { b } q _ { A , b } \frac { \partial q _ { A , b } } { \partial \theta } .\tag{37}
$$

The term $\mathcal { G } _ { \mathrm { c o n } } ^ { t } ( \mathcal { A } ( u _ { b } ) , \alpha ( u _ { b } ) )$ now explicitly includes the consistency loss gradient $\nabla _ { \boldsymbol { \theta } } { \mathcal { L } } _ { \mathrm { C o n } } ,$ which involves the Integrated Gradients over the perturbations $u _ { b }$ as well as the change in model output probabilities.

Table 7: Comparison of bACC/GM on CIFAR-10-LT under different baseline images.
<table><tr><td>FixMatch+DyTrim</td><td colspan="2">CIFAR-10-LT</td></tr><tr><td>Type of baseline</td><td> $\gamma _ { l } = \gamma _ { u } = 1 0 0$ </td><td> $\gamma _ { l } = 1 0 0 , \gamma _ { u } = 1 5 0$ </td></tr><tr><td>Noise</td><td>77.7 / 76.8</td><td>76.7 / 75.8</td></tr><tr><td>Batch Means Red</td><td>78.0 / 76.1</td><td>76.7 / 74.2</td></tr><tr><td>Green</td><td>83.5 / 83.2</td><td>82.2 / 81.7</td></tr><tr><td>Blue</td><td>83.7 / 83.3</td><td>81.5 / 81.0</td></tr><tr><td>Gray</td><td>84.5 / 84.2</td><td>83.1 / 82.6</td></tr><tr><td>White</td><td>84.1 / 83.7 84.2 / 83.8</td><td>82.3 / 81.9 82.4 / 82.0</td></tr><tr><td>Black</td><td></td><td></td></tr><tr><td></td><td>84.8 / 84.4</td><td>83.8 / 83.4</td></tr></table>

## C.6 EFFECT OF THE BASELINE IMAGE FOR GUIDING DATA PRUNING

The training objective can be interpreted as the minimization of the empirical risk L. Assuming that all labeled samples $x _ { b } ^ { n }$ from X and unlabeled samples $u _ { b } ^ { m }$ from $\mathcal { U }$ are drawn from continuous distributions $\rho ^ { l } ( x _ { b } ^ { n } )$ and $\rho ^ { u } ( u _ { b } ^ { m } )$ , respectively, the training objective can be formulated as:

$$
\underset { \theta \in \Theta } { \arg \operatorname* { m i n } } \ \underset { x _ { b } ^ { n } \in \mathcal { X } , u _ { b } ^ { m } \in \mathcal { U } } { \mathbb { E } } [ \mathcal { L } ( x _ { b } ^ { n } , u _ { b } ^ { m } ; \theta ) ] = \int _ { x _ { b } ^ { n } } \mathcal { L } _ { s u p } ( x _ { b } ^ { n } , \theta ) \rho ^ { l } ( x _ { b } ^ { n } ) d x _ { b } ^ { n } + \int _ { u _ { b } ^ { m } } \mathcal { L } _ { c o n } ( u _ { b } ^ { m } , \theta ) \rho ^ { l } ( u _ { b } ^ { m } ) d u _ { b } ^ { m } .\tag{38}
$$

After applying a data pruning policy, we sample $x _ { b } ^ { n }$ and $u _ { b } ^ { m }$ to obtain the labeled pruned subset $S _ { t } ^ { l }$ and the unlabeled pruned subset $S _ { t } ^ { u }$ , according to the labeled pruning probabilities $\mathcal P _ { t } ^ { l } ( x _ { b } ^ { n } )$ ) and unlabeled pruning probabilities $\mathscr { P } _ { t } ^ { u } ( \dot { u } _ { b } ^ { m } )$ , respectively. For the labeled samples, we directly optimize over the pruned subset $S _ { t } ^ { l }$ without reweighting the loss terms. Notably, the class-aware pruning probability $r _ { c } = \pi _ { \theta } ( \mathcal { T } )$ <sub>c</sub> inherently adjusts $S _ { t } ^ { l }$ toward an asymptotically balanced class distribution. By retaining more samples from minority classes (lower $r _ { c } )$ and pruning more samples from majority classes (higher $r _ { c } )$ , the pruned subset $S _ { t } ^ { l }$ naturally mitigates class imbalance. As a result, even without explicit rescaling, the empirical risk over $S _ { t } ^ { l }$ approximates:

$$
\underset { \theta \in \Theta } { \arg \operatorname* { m i n } } \ \underset { x _ { b } ^ { n } \in S _ { t } ^ { l } } { \mathbb { E } } [ \mathcal { L } _ { s u p } ( x _ { b } ^ { n } , \theta ) ] \propto \frac { 1 - \mathcal { P } _ { t } ^ { l } ( x _ { b } ^ { n } ) } { c _ { t } ^ { l } } \int _ { z } \mathcal { L } _ { s u p } ( x _ { b } ^ { n } , \theta ) \rho _ { l } ( x _ { b } ^ { n } ) d x _ { b } ^ { n } ,\tag{39}
$$

where $c _ { t } ^ { l } = \mathbb { E } _ { x _ { b } ^ { n } \sim \rho _ { l } } [ 1 - \mathcal { P } _ { t } ^ { l } ( x _ { b } ^ { n } ) ]$ . The term $\frac { 1 - \mathcal { P } _ { t } ^ { l } ( z ) } { c _ { t } ^ { l } }$ acts as an implicit reweighting due to the classaware pruning policy. For unlabeled samples, pruning with uniform probability r and rescaling losses by $\begin{array} { r } { \gamma _ { t } ( u ) = \frac { 1 } { 1 - \mathcal { P } _ { t } ^ { u } ( u ) } } \end{array}$ yields

$$
\underset { \theta \in \Theta } { \arg \operatorname* { m i n } } \ \underset { u _ { b } ^ { m } \in S _ { t } ^ { u } } { \mathbb { E } } \left[ \gamma _ { t } ( u _ { b } ^ { m } ) \mathcal { L } _ { c o n } ( u _ { b } ^ { m } , \theta ) \right] \propto \frac { 1 } { c _ { t } ^ { u } } \int _ { z } \mathcal { L } _ { c o n } ( u _ { b } ^ { m } , \theta ) \rho ^ { l } ( u _ { b } ^ { m } ) d u _ { b } ^ { m } ,\tag{40}
$$

where $c _ { t } ^ { u } = \mathbb { E } _ { u _ { h } ^ { m } \sim \rho _ { u } } [ 1 - \mathcal { P } _ { t } ^ { u } ( u _ { b } ^ { m } ) ]$ . Crucially, even with uniform pruning rates, the interplay of consistency regularization and confidence thresholding ensures $S _ { t } ^ { u }$ to be implicitly balanced, thus training on $S _ { t } ^ { u }$ with rescaled factor $\gamma _ { t } ( u _ { b } ^ { m } )$ could achieve a better result as training on the $\mathcal { U }$

## D MORE ABOUT THE BASELINE IMAGE

## D.1 MORE DETAIL ABOUT THE SELECTION OF BASELINE IMAGE

Sensitivity of different baseline images I. We further examined the sensitivity of DyTrim to the choice of baseline image by conducting ablation studies on CIFAR-10-LT with different types of inputs, including noise, dataset means, and solid colors. Table 7 shows that solid-color images consistently outperform noise or mean-based baselines. Among them, white and black images deliver the strongest results.

![](images/85bd1ef9734f5cc0cc2dfe94e3b5e5975d40ec46b5ded3cf72d5f5266df872b7.jpg)  
Figure 4: Illustration of the proposed DyTrim framework. DyTrim mainly consists of two operations, named labeled pruning and unlabeled pruning. ${ H _ { \prec r _ { c } , t } ^ { l } }$ and $\bar { \pmb { H } } _ { t } ^ { u }$ denote the adaptive thresholds of scores of labeled samples and unlabeled samples, with slight abuse of symbols. $\bar { \boldsymbol { S } } _ { \prec \tau } ^ { u }$ denote the low confidence unlabeled sample which $p ^ { * } ( u _ { b } ^ { m } ) \geq \tau$ . Labeled pruning provides a class-aware pruning policy for each sample from class c. Unlabeled pruning provides a random pruning policy from the original unlabeled $\mathcal { U }$ and uses a gradient rescaling strategy $( \times 1 / ( 1 - r )$ for which samples from $s _ { 1 } ^ { u }$ is selected to pruning) to keep the approximately same gradient expectation.

## D.2 DETAIL OF THE BIAS TERM AND RUNNING STATISTICS

Effects of bias term. When the bias term $\beta$ of the BN layer is frozen and equal to $0 , h (  { \mathcal { T } } )$ becomes $\gamma \ast ( \langle \mathbf { w } , k \rangle - \mathbb { E } [ \langle \mathbf { w } , k \rangle ] ) / \sqrt { \mathrm { V a r } [ \langle \mathbf { w } , k \rangle ] }$ which is the same as the $\operatorname { E q . } ( 7 )$ except for a bias term. Ignoring the running statistics strategy, the form of $h ( \mathcal T )$ only depends on the $\beta .$ As a result, $h ( \mathcal T )$ becomes $h ( \mathcal { T } )  0$ during training and $h ( \mathbb { Z } ) \to - \gamma * \mathbb { E } _ { m o m } [ \langle \mathbf { w } , x _ { b } \rangle ] / \sqrt { \mathrm { V a r } _ { m o m } [ \langle \mathbf { w } , x _ { b } \rangle ] }$ during testing. This shows that the $g _ { \theta } ^ { * }$ operation has no effect in the training phase and only eliminates the impact of the unbalanced running means in the testing phase. This will affect the ability to benefit h from $g _ { \theta } ^ { * }$ , as shown in Table. 8.

Effects of running statistics. When we do not keep running estimates, batch statistics are instead used during evaluation time as well. The form of $h ( \mathcal T )$ becomes $h ( \mathcal { T } ) \ \to \ \beta$ both training and testing. We can rewrite $g _ { \theta } ^ { \ast } ( x _ { t } ) = \gamma \ast ( \langle \mathbf { w } , x _ { t } \rangle - \mathbb { E } [ \langle \mathbf { w } , x _ { t } \rangle ] ) / \sqrt { \mathrm { V a r } [ \langle \mathbf { w } , x _ { t } \rangle ] }$ ]. On the other hand, as $h ( \mathcal { T } )  0$ , the benefit of $g _ { \theta } ^ { * }$ is also vanishes, also shown in Table. 8.

We then extend our results to a non-linear neural network, thus we have the following corollary:

Table 8: Comparison of bACC/GM on CIFAR-10-LT.
<table><tr><td>Metric</td><td>With original  $g _ { \theta } ^ { * }$ </td><td> $g _ { \theta } ^ { * }$  without  $\beta$ </td><td> $g _ { \theta } ^ { * }$  without  $\mathbf { x } _ { m o m }$ </td><td> $g _ { \theta } ^ { * }$  without  $\beta \ \& \ \mathbf { x } _ { m o m }$ </td></tr><tr><td>bACC</td><td> $8 3 . 6 \pm 0 . 4 6$ </td><td> $8 0 . 9 2 \pm 0 . 0 2 \downarrow 2 . 6 8$ </td><td> $7 1 . 6 3 \pm 0 . 3 5 \downarrow 1 1 . 9 7$ </td><td> $6 4 . 0 1 \pm 0 . 1 4 \downarrow 1 9 . 5 9$ </td></tr><tr><td>GM</td><td> $8 3 . 1 \pm 0 . 5 7$ </td><td> $8 0 . 3 7 \pm 0 . 2 3 \downarrow 2 . 7 3$ </td><td> $6 7 . 8 5 \pm 0 . 5 1 \downarrow 1 5 . 2 5$ </td><td> $5 4 . 4 8 \pm 0 . 3 6 { \downarrow } 2 8 . 6 2$ </td></tr></table>

## E MORE DETAILS ABOUT DYTRIM

## E.1 MORE ABOUT LABELED PRUNING

Specifically, we exploit the pruning policy to prune samples based on their scores. Then, for the pruned labeled samples, their scores remain unmodified as previously. For the remaining samples,

their scores are updated by the losses in the current epoch. To ensure dynamic adaptation:

$$
\mathcal { H } _ { c , t + 1 } ^ { l } ( x _ { b } ^ { n } ) = \left\{ \begin{array} { l l } { \mathcal { H } _ { c , t } ^ { l } ( x _ { b } ^ { n } ) \quad } & { x _ { b } ^ { n } \in \mathcal { X } n S ^ { l } , } \\ { \mathcal { L } _ { s u p } ( x _ { b } ^ { n } ) \quad } & { x _ { b } ^ { n } \in \mathcal { S } ^ { l } . } \end{array} \right.\tag{41}
$$

where $S ^ { l }$ denotes the pruned subset formed for labeled datasets.

## E.2 MORE ABOUT UNLABELED PRUNING

For a remaining sample with score $\mathcal { H } _ { t } ^ { u } ( u _ { b } ^ { m } ) < \bar { \mathcal { H } } _ { t } ^ { m }$ , whose corresponding pruning probability is $r ,$ its gradient is scaled to $1 / ( 1 - { \dot { r } } )$ times of the original, otherwise the gradient remains unchanged. The score $\mathscr { H } _ { t + 1 } ^ { u } ( \dot { u } _ { b } ^ { m } )$ is derived from the consistency regularization loss values $\mathcal { L } _ { c o n } ( \bar { \alpha ( } u _ { b } ^ { m } ) , \mathcal { A ( } u _ { b } ^ { m } ) )$ for unlabeled data points. To enhance pseudo-label reliability, we further apply a confidence threshold $\tau .$ , where only samples with $p ^ { * } ( \bar { u _ { b } ^ { m } } ) > \tau$ contribute to $\mathcal { L } _ { c o n }$ , where $\begin{array} { r } { \mathcal { L } _ { c o n } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \mathbb { I } ( p ^ { * } ( u _ { b } ^ { m } ) > \tau ) \mathbf { H } ( P _ { \theta } ( y | \mathbfcal { A } ( u _ { b } ^ { m } ) , \hat { q } _ { b } ) } \end{array}$ . Thus, we formulate the update of $\mathcal { H } _ { t + 1 } ^ { u } ( u _ { b } ^ { m } )$ as:

$$
\mathcal { H } _ { t + 1 } ^ { u } ( u _ { b } ^ { m } ) = \left\{ \begin{array} { c c } { \mathcal { H } _ { t } ^ { u } ( u _ { b } ^ { m } ) } & { \quad u _ { b } ^ { m } \in \mathcal { U } _ { n } S ^ { u } , } \\ { \mathcal { L } _ { c o n } ( u _ { b } ^ { m } ) } & { \quad u _ { b } ^ { m } \in S ^ { u } . } \end{array} \right.\tag{42}
$$

where $S ^ { u }$ denotes the pruned subset formed for labeled datasets. Initialization: at $t = 0 ,$ , scores $\mathcal { H } _ { t } ^ { u }$ and $\mathcal { H } _ { t } ^ { l }$ are all set to {1}, as no prior loss is available.

## F PSEUDO CODE OF THE PROPOSED ALGORITHM

The pseudo-code that describes the DyTrim is presented in Algorithm 1 and Algorithm 2.

Algorithm 1 DyTrim for Labeled Data Selection   
Input: Labeled set of N samples $\mathcal { X } = \{ ( x ^ { n } , y ^ { n } ) \} _ { n = 1 } ^ { N }$ , score set of the samples $\mathcal { V } ^ { l } .$ , number of   
classes $n _ { c } ,$ biased degree b   
Output: Labeled pruned set $S ^ { l } \left( { \mathcal { S } } ^ { l } \subseteq { \mathcal { X } } , | { \mathcal { S } } ^ { l } | < = | { \mathcal { X } } | \right)$   
1: $S ^ { l } \gets \emptyset$ ▷ Initialize the labeled pruned set   
2: for $c = 0$ to $n _ { c } - 1$ do   
3: ${ \mathcal { T } } _ { c } \gets \{ i \ | \ y _ { i } = c \}$   
4: $\mathcal { V } _ { c } ^ { l }  \bar { \{ \mathcal { V } _ { i } ^ { l } \mid i \in \bar { \mathcal { L } } _ { c } \} }$ ▷ Select scores of class c samples   
5: $k _ { c } \gets \dot { \lfloor ( 1 - b _ { c } ) \cdot | \mathcal { X } _ { c } | } \rfloor$ ▷ Compute target pruned set size of class c based on biased degree   
6: $\mathcal { T } _ { c } ^ { \mathrm { t o p } } $ TopK $( \mathcal { T } _ { c } , \mathcal { V } _ { c } ^ { l } , k _ { c } )$ ▷ Select indices of top-k scored samples   
7: $S ^ { l } \gets S ^ { l } \cup \mathcal { T } _ { c } ^ { \mathrm { t o p } }$   
8: end for   
9: return $S ^ { l }$

## G EXPERIMENTAL SETTINGS

## G.1 MODELS

Unless otherwise specified, we adopt Wide ResNet (WRN) (Zagoruyko & Komodakis, 2016) as the default backbone following common practice in semi-supervised learning. Additionally, we also evaluate Tiny Vision Transformers (TinyViT) (Wu et al., 2022) on CIFAR-10-LT and CIFAR-100- LT. For ImageNet-127, we employ ResNet-50 (He et al., 2016) as the backbone to ensure scalability on large-scale datasets.

## G.2 IMPLEMENTATION DETAILS

All experiments are trained for 500 epochs with 500 steps per epoch, resulting in a total of 250,000 iterations. We use Stochastic Gradient Descent (SGD) (Bottou, 2012) with a fixed learning rate of $\eta = 0 . 0 0 1 5$ and a batch size of 32. The pruning ratio of the unlabeled dataset is set to 0.7, and the parameter δ is aligned with InfoBatch (Qin et al., 2024), fixed at 0.875. For CIFAR-10-LT, the largest labeled class contains 1,500 samples, while the largest unlabeled class contains 3,000 samples. For CIFAR-100-LT, the largest labeled and unlabeled classes contain 150 and 300 samples, respectively. For STL-10-LT, the largest labeled class contains 450 samples. To assess classification performance, we adopt balanced accuracy (bACC) (Huang et al., 2016) and geometric mean (GM) (Kubat, 1997) for CIFAR-10-LT and STL-10-LT. For CIFAR-100-LT and ImageNet-127, evaluation is conducted solely using bACC. Each experiment is repeated three times on RTX 4090 GPUs to ensure repro ducibility, and we report both the mean and the standard error.

Algorithm 2 DyTrim for Unlabeled Data Selection   
Input: Unlabeled set of M samples $\mathcal { U } = \{ ( \boldsymbol { u } ^ { m } ) \} _ { m = 1 } ^ { M }$ , score set of the samples $\mathcal { V } ^ { u }$ , pruning ratio r,   
weight of samples w   
Output: Unlabeled pruned set $S ^ { l } \left( S ^ { l } \subseteq \mathcal { U } , \left| S ^ { u } \right| < = \left| \mathcal { U } \right| \right)$   
1: $S ^ { u } \gets \emptyset$ ▷ Initialize the unlabeled pruned set   
2: $\mathcal { T } _ { 0 }  \{ i \mid \mathcal { V } _ { i } ^ { u } = 0 \}$ ▷ Select low confidence samples   
3: ${ \mathcal { T } } _ { \neq 0 } \gets \dot { \{ i \ | \ \} } _ { i } ^ { u } \neq \bar { 0 } \}$ ▷ Select high confidence samples   
4: $\hat { S ^ { u } }  \hat { S ^ { u } } \hat { \cup } \hat { \mathbb { Z } _ { 0 } }$   
5: $\mu  \mathbf { M e a n } ( \{ \mathcal { \bar { V } } _ { i } ^ { u } \mid i \in \mathbb { Z } _ { \neq 0 } \} )$   
6: ${ \mathcal { T } } _ { \mathrm { w e l l } }  \{ i \in { \mathcal { I } } _ { \neq 0 } \mid { \mathcal { V } } _ { i } ^ { u } < \mu \}$ ▷ Select well-learned samples   
7: ${ \mathcal { T } } _ { \mathrm { p o o r } }  { \dot { \mathcal { T } } } _ { \neq 0 } \setminus { \dot { \mathcal { T } } } _ { \mathrm { w e l l } }$ ▷ Select poorly-learned samples   
8: $\dot { S ^ { u } }  S ^ { u } \cup \mathcal { T } _ { \mathrm { p o o r } }$   
9: $\mathcal { T } _ { \mathrm { s e l e c t } } $ Randomly select $\lfloor ( 1 - r ) \cdot | \mathbb { Z } _ { \mathrm { w e l l } } | \rfloor$ samples from $\mathcal { T } _ { \mathrm { w e l l } }$   
10: $S ^ { u }  S ^ { u } \cup \mathcal { T } _ { \mathrm { s e l e c t } }$   
11: $w _ { i } \gets 1 , \forall i \in \{ 1 , \dots , M \}$ ▷ Reset weights   
12: $w _ { i }  \frac { 1 } { 1 - r } , \forall i \in \mathcal { T } _ { \mathrm { s e l e c t } }$ ▷ Rescaling   
13: return $S ^ { u }$

## H ADDITIONAL EXPERIMENTAL RESULTS

## H.1 BASELINES

The classification performance of the DyTrim was compared with those of the following algorithms: 1. vanilla algorithm - Deep CNN trained with cross-entropy loss, 2. CIL algorithms - Resampling (JAPKOWICZ, 2000), LDAM-DRW (Cao et al., 2019), and cRT (Kang et al., 2020), 3. SSL algorithms - FixMatch (Sohn et al., 2020), and 4. CISSL algorithms - DARP, DARP+LA, DARP+cRT (Kim et al., 2020), CReST, CReST+LA (Wei & Gan, 2023), ABC (Lee et al., 2021), CoSSL (Fan et al., 2022), DASO (Oh et al., 2022), SAW, SAW+LA and SAW+cRT (Lai et al., 2022) combined with FixMatch. Adsh(Guo & Li, 2022), DebiasPL (Wang et al., 2022), UDAL(Lazarow et al., 2023) and L2AC (Wang et al., 2023a) combined with FixMatch. We report the performance of the baseline algorithms reported in Tables of Lai et al. (2022) and Fan et al. (Fan et al., 2022) when it is reproducible; the performance measured using the uploaded code was reported otherwise.

## H.2 ADDITIONAL RESULTS ON CIFAR-10-LT

Following prior works (Xing et al., 2025; Lee & Kim, 2024; Guo et al., 2024), we evaluate under a more challenging scenario where the unlabeled set is imbalanced in the reverse direction of the labeled set (Table 9). Across all settings, DyTrim delivers consistent gains by applying balanced pruning on the labeled data. Notably, when combined with FixMatch, DyTrim surpasses CDMAD by more than 1% in both bACC and GM. Similar benefits are observed for FlexMatch and FreeMatch: DyTrim improves FlexMatch by approximately 1.1–1.3% and FreeMatch by around 0.9–1.5%.

We also compared the classification performance of CDMAD with ACR (Xiang et al., 2020) and BaCon, two recent CISSL algorithms. From Table. 10, we can observe that CDMAD outperforms both ACR and BaCon.

To further validate the balanced classification effect of DyTrim, we visualized the dynamics of baseline image logits during training as shown in Figure. 5 (a), (b) and (c). The results clearly showed that DyTrim significantly reduced classifier bias induced by class imbalance.

Table 9: Comparison of bACC/GM on CIFAR-10-LT(γ<sub>l</sub> = 100, γ<sub>u</sub> = 100(reversed)).
<table><tr><td rowspan="2">Algorithm</td><td colspan="4"> $\mathrm { C I F A R - 1 0 - L T } , \gamma _ { l } = 1 0 0 , \gamma _ { u } = 1 0 0 ( \mathrm { r e v e r s e d } )$ </td></tr><tr><td>ABC</td><td>SAW  $\mathrm { S A W + L A }$   $\mathbf { S } \mathbf { A } \mathbf { W } { + } \mathbf { c } \mathbf { R } \mathbf { T }$ </td><td>CDMAD</td><td>DyTrim</td></tr><tr><td>FixMatch+</td><td> $\overline { { 6 9 . 5 / 6 6 . 8 } }$ </td><td> $\overline { { 7 2 . 3 / 6 8 . 7 } }$   $\overline { { 7 4 . 1 / 7 2 . 0 } }$ </td><td> $\overline { { 7 5 . 5 / 7 3 . 9 } }$   $7 7 . 1 / 7 5 . 4$ </td><td>78.2 / 76.7</td></tr><tr><td>FlexMatch+</td><td> $- / -$ </td><td> $- / -$   $- / -$   $- / -$ </td><td> $6 7 . 2 \dot { / 6 5 } . 1$ </td><td>68.3 / 66.4</td></tr><tr><td>FreeMatch+</td><td> $- / -$ </td><td> $- / -$  -1-  $- / -$ </td><td> $6 8 . 5 / 6 6 . 4 $ </td><td>69.4 / 67.9</td></tr></table>

Table 10: Comparison of bACC/GM on CIFAR-10-LT
<table><tr><td>Algorithm/CIFAR-10-LT</td><td> $\gamma _ { l } = \gamma _ { u } = 1 0 0$ </td><td> $\gamma _ { l } = \gamma _ { u } = 1$ </td></tr><tr><td>FixMatch+ACR</td><td> $8 1 . 8 / 8 1 . 4 $ </td><td> $8 5 . 6 / 8 5 . 3$ </td></tr><tr><td>FixMatch+BaCon</td><td> $8 4 . 4 / 8 4 . 0 $ </td><td> $8 2 . 0 / 8 1 . 5 $ </td></tr><tr><td>FixMatch+CDMAD</td><td> $8 3 . 6 / 8 3 . 1 $ </td><td> $8 7 . 5 / 8 7 . 1 $ </td></tr><tr><td>FixMatch+DyTrim</td><td> $\mathbf { 8 4 . 8 / 8 4 . 4 }$ </td><td> $\mathbf { 8 7 . 9 } / \mathbf { 8 7 . 5 }$ </td></tr></table>

![](images/162f9730ad871115ce11f5ca486602cea330d06c5dccca8c554b2d9cd3f38793.jpg)  
(a) FixMatch

![](images/ad4d644b4e749ac061d51bd0c53ae556a6cfac0ab22a5108e0f92e44b3ef9f8c.jpg)  
(b) CDMAD

![](images/22f2a54c15750b4e927934ad8f535065bae853e4fee97427dae320c7b59f51f0.jpg)  
(c) DyTrim

![](images/bda144f3628a3e06158b3c100e2a93f0bd6fb01127aaab2550211e5d28cb70ab.jpg)  
(d) Metric  
Figure 5: (a), (b) and (c) present the change of $\pi _ { \boldsymbol { \theta } } ( \mathcal { T } )$ for the baseline image on CIFAR-10-LT with $\gamma _ { l } = \gamma _ { u } = 1 0 0$ across different methods. (d) present the bACC and GM on those methods.

## H.3 RESULTS ON SMALL-IMAGENET-127

ImageNet-127 is a naturally long-tailed dataset, widely used to evaluate class-imbalanced semi-supervised learning (CISSL) algorithms at scale. Following standard protocol, we downsample images to resolutions of $3 2 \times 3 2$ and $6 4 \times 6 4$ using the box interpolation method from the Pillow library, and randomly select 10% of the training samples as labeled data. Under such limited supervision and class imbalance, learning discriminative representations and a balanced classifier is particularly challenging. As reported in Table. 11, DyTrim achieves the highest balanced accuracy (bACC) at both resolutions, outperforming the strongest baseline CD-MAD by 3.0% at $3 2 \times 3 2$ and 1.2% at $6 4 \times 6 4$ . These improvements demonstrate the robustness of our method, especially under low-resolution and low-resource conditions. The performance gain at lower resolutions suggests that DyTrim effectively handles the compounded difficulty of reduced visual fidelity and severe label scarcity. This makes it a promising solution for real-world applications where high-resolution data and abundant labels are often unavailable.

Table 11: Comparison of bACC on Small-ImageNet-127.
<table><tr><td rowspan="2">Algorithm</td><td colspan="2">Small-ImageNet-127</td></tr><tr><td> $3 2 \times 3 2$ </td><td> $6 4 \times 6 4$ </td></tr><tr><td>FixMatch</td><td>29.7</td><td>42.3</td></tr><tr><td>w/+DARP</td><td>30.5</td><td>42.5</td></tr><tr><td>w/+DARP+cRT</td><td>39.7</td><td>51.0</td></tr><tr><td>w/+CReST</td><td>32.5</td><td>44.7</td></tr><tr><td>w/+CReST+LA</td><td>40.9</td><td>55.9</td></tr><tr><td>w/+ABC</td><td>46.9</td><td>56.1</td></tr><tr><td>w/+CoSSL</td><td>43.7</td><td>53.8</td></tr><tr><td>w/+CPE</td><td>47.8</td><td>58.2</td></tr><tr><td>w/+CDMAD</td><td>48.4</td><td>59.3</td></tr><tr><td>w/+DyTrim</td><td>50.6</td><td>60.0</td></tr></table>

## H.4 MORE RESULTS ON IMAGENET-LT

ImageNet-LT (Liu et al., 2019) is a long-tailed variant of ImageNet, constructed to exhibit a heavy class-imbalance that better reflects real-world data distributions. To assess the scalability of our method on large-resolution inputs (224 × 224), we conducted experiments on ImageNet-LT. Due to hardware constraints, we set the batch size to 2.

As shown in Table 3, CDMAD yields a substantial improvement over the FixMatch baseline, increasing bACC from 20.0% to 35.4%, which highlights the effectiveness of incorporating classdistribution modeling under long-tailed imbalance. Building upon the same baseline, our method further pushes performance to 37.2%, achieving the best result among all compared approaches. Notably, the improvement over CDMAD remains consistent despite their strong performance, suggesting that our approach introduces complementary benefits rather than merely overlapping with prior re-balancing techniques.

## H.5 RESULTS ON DYNAMIC DATA PRUNING EXPERIMENT

Recently, Infobatch (Qin et al., 2024) provides a no-bias dynamic data pruning method. In this section, we compare it with DyTrim in the framework of CISSL. The experiment is conducted on the CIFAR-10-LT dataset, comparing the settings of $\gamma _ { l } = \gamma _ { u }$ and $\gamma _ { l } \neq \gamma _ { u } .$ . Specifically, we directly apply the pruning policy of InfoBatch to labeled samples and unlabeled samples without distinction, and the results are shown in the Table. 12 and Table. 13. It can be seen that compared with the proposed method, the pruning policy directly combined with InfoBatch is not consistently effective in all settings. In particular, when $\gamma _ { l } \neq \gamma _ { u } ,$ it will cause a decrease in accuracy, which is caused by the mismatch in the distribution of labeled samples and unlabeled samples.

Table 12: Comparison of bACC/GM on CIFAR-10-LT.
<table><tr><td rowspan="2">Algorithm</td><td colspan="3">CIFAR-10-LT  $( \gamma = \gamma _ { l } = \gamma _ { u } , \gamma _ { u }$  is assumed to be known)</td></tr><tr><td> $\gamma _ { l } = 5 0 , \gamma _ { u } = 5 0$ </td><td> $\gamma _ { l } = 1 0 0 , \gamma _ { u } = 1 0 0$ </td><td> $\gamma _ { l } = 1 5 0 , \gamma _ { u } = 1 5 0$ </td></tr><tr><td>FixMatch</td><td> $7 9 . 2 \pm 0 . 3 3 / 7 7 . 8 \pm 0 . 3 6$ </td><td> $7 1 . 5 \pm 0 . 7 2 / 6 6 . 8 \pm 1 . 5 1$ </td><td> $6 8 . 4 \pm 0 . 1 5 / 5 9 . 9 \pm 0 . 4 3$ </td></tr><tr><td>w/+CDMAD</td><td> $8 7 . 3 \pm 0 . 1 2 / 8 7 . 0 \pm 0 . 1 5$ </td><td> $8 3 . 6 \pm 0 . 4 6 / 8 3 . 1 \pm 0 . 5 7$ </td><td> $8 0 . 8 \pm 0 . 8 6 / 7 9 . 9 \pm 1 . 0 7$ </td></tr><tr><td>w/+InfoBatch*</td><td> $8 7 . 2 \pm 0 . 1 8 / 8 6 . 9 \pm 0 . 1 9$ </td><td> $8 4 . 1 \pm 0 . 6 1 / 8 3 . 7 \pm 0 . 6 9$ </td><td> $8 1 . 6 \pm 0 . 4 5 / 8 0 . 9 \pm 0 . 5 9$ </td></tr><tr><td>w/+DyTrim</td><td> ${ \bf 8 8 . 0 \pm 0 . 3 1 / 8 7 . 8 \pm 0 . 3 2 }$ </td><td> $8 4 . 8 \pm 0 . 4 8 / 8 4 . 4 \pm 0 . 5 1$ </td><td> ${ \bf 8 2 . 0 \pm 0 . 0 9 / 8 1 . 3 \pm 0 . 0 3 }$ </td></tr></table>

Table 13: Comparison of bACC/GM on CIFAR-10-LT $( \gamma _ { l } \neq \gamma _ { u } , \gamma _ { u }$ is assumed to be unknown).
<table><tr><td rowspan="2">Algorithm</td><td colspan="3"> $\mathrm { C I F A R - 1 0 - L T } \left( \gamma _ { l } = 1 0 0 , \gamma _ { u } = \mathrm { U n k n o w n } \right)$ </td></tr><tr><td> $\gamma _ { u } = 1$ </td><td> $\gamma _ { u } = 5 0$ </td><td> $\gamma _ { u } = 1 5 0$ </td></tr><tr><td>FixMatch</td><td> $6 8 . 9 \pm 1 . 9 5 / 4 2 . 8 \pm 8 . 1 1$ </td><td> $7 3 . 9 \pm 0 . 2 5 / 7 0 . 5 \pm 0 . 5 2$ </td><td> $6 9 . 6 \pm 0 . 6 0 / 6 2 . 6 \pm 1 . 1 1$ </td></tr><tr><td>w/+CDMAD</td><td> $8 7 . 5 \pm 0 . 4 6 / 8 7 . 1 \pm 0 . 5 0$ </td><td> $8 5 . 7 \pm 0 . 3 6 / 8 5 . 3 \pm 0 . 3 8$ </td><td> $8 2 . 3 \pm 0 . 2 3 / 8 1 . 8 \pm 0 . 2 9$ </td></tr><tr><td>w/+InfoBatch*</td><td> $8 6 . 4 \pm 0 . 6 3 / 8 5 . 9 \pm 0 . 7 3$ </td><td> $8 5 . 5 \pm 0 . 3 3 / 8 5 . 1 \pm 0 . 3 7$ </td><td> $8 3 . 3 \pm 0 . 0 8 / 8 2 . 8 \pm 0 . 1 1$ </td></tr><tr><td>w/+DyTrim</td><td> $\mathbf { 8 8 . 9 \pm 0 . 8 8 / 8 8 . 6 \pm } 1 . 0 3$ </td><td> ${ \bf 8 6 . 4 \pm 0 . 4 3 / 8 6 . 0 \pm 0 . 4 3 }$ </td><td> $8 3 . 8 \pm 0 . 3 4 / 8 3 . 4 \pm 0 . 3 3$ </td></tr></table>

## H.6 ABLATION STUDY

Effectiveness of each component. We conducted ablation studies on CIFAR-10-LT to assess the contribution of each component in DyTrim, varying the hyperparameter $\gamma = \gamma _ { l } = \gamma _ { u }$ across 50, 100, and 150. As shown in Table. 14, the best performance was achieved when both labeled and unlabeled pruning were combined with rescaling. Removing rescaling led to a bACC drop of 0.8–2.1 points across γ values. Excluding either pruning component also reduced performance (e.g., -0.5 and $- 0 . 3 \mathrm { ~ a t ~ } \gamma \mathrm { ~ = ~ } 5 0$ without unlabeled or labeled pruning, respectively). Removing both pruning strategies resulted in the most significant degradation. These results highlighted the complementary benefits of pruning and rescaling.

## H.7 QUALITATIVE ANALYSES

Since the baseline image could implicitly reflect the bias of the classifier, we argued that by customizing dynamic data pruning methods for labeled and unlabeled data, DyTrim significantly reduced classifier bias while improving performance. To verify this claim, in Figure. 6 (a) and (b), we analyzed the class probabilities predicted on the baseline image using FixMatch+DyTrim, trained on CIFAR-10-LT under various settings. We observed that classifiers trained with DyTrim consistently produced more balanced predictions than CDMAD across all settings, with improved accuracy on tail classes. We defined r as the probability of pruning an unlabeled sample $u _ { b } ^ { m }$ when $\mathcal { H } _ { t } ^ { u } ( u _ { b } ^ { m } ) < \bar { \mathcal { H } } _ { t } ^ { m }$ and max $( P _ { \theta } ( y | \alpha ( u _ { b } ^ { m } ) ) ) \ge \tau$ . In Figure. 7, we evaluated different pruning ratios for unlabeled samples on CIFAR-10-LT. Results showed that setting $r \geq 0 . 1$ yields higher performance across both architectures, indicating that DyTrim was relatively robust with respect to the hyperparameter r, with the best performance achieved when $r = 0 . 3$

Table 14: Ablation study for the proposed algorithm on CIFAR-10-LT.
<table><tr><td rowspan="2">Labeled Pruning</td><td rowspan="2">Unlabeled Pruning</td><td rowspan="2">Rescaling</td><td colspan="2"> $\gamma _ { l } = \gamma _ { u } = 5 0$ </td><td colspan="2"> $\gamma _ { l } = \gamma _ { u }$  = 100</td><td colspan="2"> $\gamma _ { l } = \gamma _ { u }$  = 150</td></tr><tr><td>bACC</td><td>GM</td><td>bACC</td><td>GM</td><td>bACC</td><td>GM</td></tr><tr><td></td><td></td><td></td><td>87.3</td><td>87.0</td><td>83.6</td><td>83.1</td><td>80.8</td><td>79.9</td></tr><tr><td>√</td><td></td><td></td><td>87.5</td><td>87.2</td><td>84.4</td><td>84.0</td><td>81.3</td><td>80.6</td></tr><tr><td></td><td>√</td><td>√</td><td>87.7</td><td>87.4</td><td>84.0</td><td>83.6</td><td>81.4</td><td>80.6</td></tr><tr><td>√</td><td>√</td><td></td><td>87.2</td><td>86.9</td><td>83.6</td><td>83.1</td><td>79.9</td><td>79.0</td></tr><tr><td>√</td><td>√</td><td></td><td>88.0</td><td>87.8</td><td>84.8</td><td>84.4</td><td>82.0</td><td>81.3</td></tr></table>

![](images/64822c55a930c5453780d1c8d8e99c9470ca21db2ba87e3811188274575d1764.jpg)  
(a) FixMatch+CDMAD

![](images/4b356ac206891c5a4ad2deca9478eb787cafa6179a194dc7823dd4aa907a6882.jpg)  
(b) FixMatch+DyTrim

![](images/825fa9003dd6fd004f4b5e3897ccc706d4f36c687d26bb98c0d6b001759d1633.jpg)  
(c) FixMatch+CDMAD

![](images/b3de01666c76c7613774a1e72ca7f2ce136c77be91e3ca419c8b4a8cc7e408f1.jpg)  
(d) FixMatch+ DyTrim  
Figure 6: (a) and (b) present the $\pi _ { \boldsymbol { \theta } } ( \mathcal { T } )$ using the CDMAD and DyTrim. (c) and (d) present the confusion matrices of the class predictions on test samples on CIFAR-10-LT $( \gamma _ { l } = \gamma _ { u } = 1 0 0 )$

![](images/97c1b2cdd95cdce2ab3b8095115603b95f351d787c2ba4312d9dc85b08324c15.jpg)  
Figure 7: Evaluation curves of hyper-parameter r on CIFAR-10-LT under bACC and GM.

## H.8 COMPARISON OF CLASS DISTRIBUTIONS BEFORE AND AFTER PRUNING

Figure 8 compares the class distributions before and after applying DyTrim on the labeled, unlabeled and full training sets. Across all three subsets, pruning consistently reduces the proportion of head classes while preserving or slightly increasing the relative proportion of tail classes. This produces a noticeably flatter long-tailed distribution. Unlike traditional pruning methods, which typically remove samples that contribute least to training progress, the behavior of DyTrim is different because the pruning decision is guided by baseline logits and the reliability of pseudo-labels. This tends to eliminate redundant head-class samples and low-quality unlabeled samples while rarely discarding the already scarce tail-class data. Consequently, the resulting effective training subset becomes more balanced without sacrificing essential information from tail classes.

![](images/1b23fb5e7e834422f0cfcc1bddf816b65bc23b3fa9e6e79faf1884001185996a.jpg)  
(a) Labeled dataset

![](images/f56cba4da591bf4237f779c7100fe4a3709ca3ab7379d823ad53540c2eabedcf.jpg)  
(b) Unlabeled dataset

![](images/ad3f02c61135c5e19ad8566d1b2694567ca180a7c6ff3e2173931d6c2a79d74f.jpg)  
(c) Full dataset  
Figure 8: Comparison of class distribution before and after pruning across three datasets: (a) Labeled dataset, (b) Unlabeled dataset, (c) Full dataset.

## H.9 ANALYSIS OF SAMPLE SELECTION FREQUENCY

![](images/442a652c34239bd6550c9243801a14cecd528bac45b627c59b5926432614e347.jpg)  
Figure 9: Illustration of per-class maximum, average, and minimum sample selection frequencies during training.

![](images/7bbaec065e8a6ee0031994c679d0cf9c308e5a36c4530cde6f6643d86494f0d0.jpg)  
Figure 10: Comparison of class-probability distributions with and without scaling.

Figure 9 reports the maximum, average and minimum sample selection frequencies for each class. Three observations emerge clearly. First, the maximum frequency remains close to 1 for all classes, which indicates that each class contains at least a subset of highly informative samples that are almost always preserved during pruning. Second, the average frequency increases from head to tail classes, showing that

Table 15: Comparison of bACC and GM on CIFAR-10-LT on fixed and dynamic scaling factors.
<table><tr><td rowspan="2">Algorithm</td><td colspan="2">CIFAR-10-LT</td></tr><tr><td> $\gamma _ { l } = 1 0 0 , \gamma _ { u } = 1 0 0$ </td><td> $\gamma _ { l } = 1 0 0 , \gamma _ { u } = 1 / 1 0 0$ </td></tr><tr><td>Fixed Scaling</td><td>84.8 / 84.4</td><td>78.2 / 76.7</td></tr><tr><td>Dynamic Scaling</td><td>84.9 / 84.4</td><td>78.9 / 78.1</td></tr></table>

DyTrim removes a larger fraction of redundant samples from majority classes while retaining more samples in minority classes. This behavior matches the intended effect of mitigating class domi nance through selective pruning. Third, the minimum frequency stays within a narrow and rela tively high range across all classes, suggesting that even the least frequently selected samples are not entirely discarded. This prevents the severe under-sampling of tail classes that often occurs in traditional pruning strategies.

## H.10 EFFECT OF SCALING STRATEGIES ON CLASS-BIAS

Figure 10 compares the class probability distributions obtained with and without the proposed scaling strategy. Although the two curves differ for several head and mid-frequency classes, the overall decay pattern remains consistent, and the probabilities of head classes do not increase when scaling is applied. This shows that the scaling mechanism does not intensify the influence of high confidence samples and preserves the long-tailed structure shaped by DyTrim.

Additionally, to provide each class with an adaptive scaling factor that assigns smaller scaling to head classes and larger scaling to tail classes, we further compare fixed and dynamic scaling in Table 15. Dynamic scaling leads to higher bACC and GM under both matched and mismatched imbalance conditions, indicating that adapting the scaling factor to the current pruning state yields a more reliable correction for changes in the effective batch size. The dynamic scaling factor is computed as $1 - \pi _ { \boldsymbol { \theta } } ( \boldsymbol { \mathcal { T } } ) _ { \hat { q } _ { b } } + 1 / ( 1 - \boldsymbol { r } )$ , which stabilizes the loss magnitude during training and prevents undesirable shifts toward majority class predictions.

## H.11 DYNAMICS OF SAMPLE SCORE ACROSS HEAD AND TAIL CLASSES

![](images/7b042857d9d5ab290503fdbe314cc12359b82aaae7334feabb5f95bc4729f931.jpg)  
Figure 11: Scores of a representative head class sample and a representative tail class sample over the first 50,000 training steps, recorded every 500 steps.

Figure 11 shows the dynamics of scores for one head class sample and one tail class sample over the first 50,000 training steps. The two trajectories exhibit a clear contrast. The tail class sample maintains consistently higher and more volatile scores throughout training, reflecting its larger contribution to reducing class bias and its higher utility for updating the classifier. In comparison, the head class sample quickly drops to very low scores and remains close to zero for most of training. This indicates that the head sample becomes saturated early and provides little additional information, which aligns with the design of DyTrim that aims to remove redundant head class samples.

## H.12 PRUNING DYNAMICS ACROSS LABELED AND UNLABELED DATASETS

![](images/22ca17bddf8e2da56c636ea25b23f29af4bffe3a82eea92cda4a188f2909de0d.jpg)  
(a) Head classes (labeled)

![](images/079ef2edcaf90a15991642d3ed460ff3475f03aa2839fe23abbb89b4872d3804.jpg)  
(b) Tail classes (labeled)

![](images/83ff673eeded2857cfec8668b690908a50f3315ee3d7f96e5e5fcb4582d70b24.jpg)  
(c) Head classes (unlabeled)

![](images/6f7676cd53bc98bd42f986e5725948ccda95a6421c44ffcc3949d43ce4d156d3.jpg)  
(d) Tail classes (unlabeled)  
Figure 12: Number of pruned samples for each class across training process on CIFAR-10-LT. (a) and (b) show the evolution for head and tail classes in the labeled set, and (c) and (d) show the corresponding results for the unlabeled set. Each curve indicates how many samples of a given class have been removed up to each pruning step, recorded every 100 iterations.

![](images/3427eb23f64bbaefd7e239d685ccd340904e93128883e76723e648e5b43d74bb.jpg)  
Figure 13: Comparison of the change of logits’s probability distribution $\pi _ { \boldsymbol { \theta } } ( \mathcal { T } )$ for the baseline image on CIFAR-10-LT with $\gamma _ { l } = \gamma _ { u } = 1 0 0$ across different CISSL methods.

Figure 12 reports the number of pruned samples per class over the course of training. The results from both the labeled and unlabeled subsets exhibit a consistent pattern. Head classes experience a rapid increase in pruned samples at the beginning of training and maintain high pruning counts throughout the process, which reflects the large amount of redundant information contained in these majority classes. In contrast, tail classes show much slower growth curves with considerably lower pruning volumes, indicating that DyTrim preserves most of the scarce minority samples and avoids aggravating the long-tailed imbalance. The same trend appears in the unlabeled subset, where head classes accumulate substantially more pruned samples due to the prevalence of high confidence but less informative pseudo-labeled instances. These observations confirm that DyTrim adaptively modulates pruning according to class frequency and sample utility, removing redundant head-class samples while retaining informative tail-class data.

## I VISUALIZATION

## I.1 DETAILS OF THE CHANGE OF LOGITS’S PROBABILITY DISTRIBUTION

In this section, we conduct some visualization experiments to demonstrate the advantages of the DyTrim in debiasing and improving classifier performance. We first analyze the change of logits’s probability distribution Softmax $( g _ { \boldsymbol { \theta } } ( \mathcal { T } ) )$ for the baseline image on CIFAR-10-LT with $\gamma _ { l } = \gamma _ { u } =$ 100 for fixmatch, CDMAD, and the DyTrim as shown in Figure. 13. It can be seen intuitively that in the first epoch, the classifier has bias due to the imbalance of categories in the data. This situation increases significantly with the number of network training times, as shown in the second column of the figure. However, we can see that DyTrim can effectively slow down the increase of this bias. Furthermore, after the model is fully trained for 500 epochs, it can be seen that after the 100th epoch, CDMAD starts to use the baseline image for post-hoc debiasing, which significantly reduces the representation of the model. However, by dynamically pruning the data set, DyTrim obtains a more distinct debias effect as shown in Figure. 14.

## I.2 DETAILS OF THE CHANGE OF LOGITS’S PROBABILITY DISTRIBUTION

Figure. 15 and Figure. 16 compare the confusion matrices of the class predictions on the test set of CIFAR-10 using (a) FixMatch, (b) FixMatch+Infobatch, (c) FixMatch+CDMAD, and (d) Fix-

Predicted Label  
![](images/bb1a96ea9b91fd280a66965acc9ff7cabaaea173319ff82ac1273a15b92d3c7c.jpg)  
(a) FixMatch

![](images/56a8155f9ac2050b053c9dbfa15aea97d9aa847e68232ec028470c3cb0b22072.jpg)  
(b) FixMatch+InfoBatch

![](images/f9ccfe955b0f88a02db350b2d5ab49d84c33e41bf256abadf8726e574b9d19e0.jpg)  
(c) FixMatch+CDMAD

![](images/1a7c1a4c4b61a92653ef0df59fb95180f4a0bc978f7a2e096ca8d2d1d758d2c3.jpg)  
(d) FixMatch+ DyTrim

Figure 14: Class probabilities predicted on a baseline image using (a) FixMatch, (b) Fix-Match+InfoBatch, (c) FixMatch+CDMAD, (d) FixMatch+DyTrim.  
![](images/a69bfd83636add044c7dfcfd4d804bae0224b1038a31ab0faa4b56b0e0b1f5ff.jpg)  
(a) FixMatch

![](images/29b97e79006f3b6d3cbc1df8b428122087dc5ddd1ec6fdac986c6a2b2575f1f2.jpg)  
(b) FixMatch+InfoBatch

![](images/6268be1564529d6dd89d7ab291ea38dcf179e12b8c514fd3c0a5f100482fbe99.jpg)  
(c) FixMatch+CDMAD

![](images/be860bcf571c38deaf195465fc7ca5030f0cbd28eed7bdf66b77794b28eada18.jpg)  
(d) FixMatch+ DyTrim

Figure 15: Confusion matrices of the class predictions on the test set of CIFAR-10 using (a) Fix-Match, (b) FixMatch+InfoBatch, (c) FixMatch+CDMAD, and (d) FixMatch+DyTrim trained on CIFAR-10-LT under $\gamma _ { l } = 1 0 0$ and $\gamma _ { u } = 1 0 0$  
![](images/3225a9117cd85f404bc7634fa824ce67a4f4b8058e7d64a08cd8c834e8e492a9.jpg)  
(a) FixMatch

![](images/20f7220fc8bd36236eefa95a9bac860c9d0e02cf90aadab5c3c557cb5df3a7eb.jpg)  
(b) FixMatch+InfoBatch

![](images/ad2398cf14ab427dc3c2d51f478077f089423d4c3cfc975d49ddbd8fa58b3803.jpg)  
(c) FixMatch+CDMAD

![](images/bc0370ea63a6c2d43e70bce7fc547769317f9d659ad6f8dca5f122548bf7b6cf.jpg)  
(d) FixMatch+ DyTrim  
Figure 16: Confusion matrices of the class predictions on the test set of CIFAR-10 using (a) Fix-Match, (b) FixMatch+InfoBatch, (c) FixMatch+CDMAD, and (d) FixMatch+DyTrim trained on CIFAR-10-LT under $\gamma _ { l } = 1 0 0$ and $\gamma _ { u } = 1$

Match+DyTrim trained on CIFAR-10-LT under $\gamma _ { l } = 1 0 0 , \gamma _ { u } = 1 , 1 0 0$ . FixMatch+DyTrim made more balanced predictions across classes. Furthermore, we also conducted experiments under a balanced setting $( \gamma = \gamma _ { 1 } = \gamma _ { u } = 1 )$ , as shown in Figure. 17. The results show that even under a balanced data distribution, DyTrimcan still achieve better results on the pruned dataset than methods such as CDMAD trained on the full dataset.

Similar to confusion matrices, we also compare t-distributed stochastic neighbor embedding (t-SNE) of representations obtained for the test set of CIFAR-10 using FixMatch, FixMatch+CDMAD, FixMatch+InfoBatch, and FixMatch+DyTrim trained on CIFAR-10 with $\gamma _ { l } ~ = ~ 1 0 0$ and $\gamma _ { u } =$ 1, 100(unknown $\gamma _ { u } )$ , where different colors indicate different classes in CIFAR-10 Figure. 18, Fig ure. 19. We can observe that the representations obtained using FixMatch+DyTrim are separated into classes with clearer boundaries compared the those from FixMatch and CDMAD. This is prob ably because CDMAD appropriately refined the biased pseudo-labels and used them for training, whereas FixMatch failed to learn the representations properly because they used the biased pseudolabels for training. These results demonstrate that the quality of representations can be improved by using well-refined pseudo-labels for training.

![](images/945a46656e1eb70a23eda4018dc96d77a4e102976ff62c1430c9f0510ce88703.jpg)  
(a) FixMatch

![](images/f2c5b975c21a19c589657622dbf4429cf01b364f333beb152a0fb9b08bd4708b.jpg)  
Predicted Label  
(b) FixMatch+InfoBatch

![](images/3c59355eafc15e3d31d3609da17dcf96c66af54d95331b40921956ff86475862.jpg)  
Predicted Label  
(c) FixMatch+CDMAD

![](images/ea789bf598a71ccd3fefa68547668e68885de4e3e74e7ab5fcd805cc1f7bc647.jpg)  
(d) FixMatch+ DyTrim

Figure 17: Confusion matrices of the class predictions on the test set of CIFAR-10 using (a) Fix-Match, (b) FixMatch+InfoBatch, (c) FixMatch+CDMAD, and (d) FixMatch+DyTrim trained on CIFAR-10-LT under $\gamma _ { l } = 1$ and $\gamma _ { u } = 1$  
![](images/cc9ad3d62f6db7403ac51ad96733add358c1e32eed8ec72bce5c397e138968ef.jpg)  
(a) FixMatch

![](images/f3e3296fc9747defd9d56451329488fc22d5c7513a1f9c8aa384e2562a0f13fe.jpg)  
(b) FixMatch+InfoBatch

![](images/4b02a72b1491d0b383680e7f5fd7d3d21d0eacb16af6e489341491d05cb58c46.jpg)  
(c) FixMatch+CDMAD

![](images/2070d3ecc71f7f150987cc2f33f2aa10f78952a86ca9965aa2a5b28ca473bb86.jpg)

Figure 18: t-SNE of representations obtained for the test set of CIFAR-10 using (a) FixMatch, (b) FixMatch+InfoBatch, (c) FixMatch+CDMAD, and (d) FixMatch+DyTrim trained on CIFAR-10-LT under $\gamma _ { l } = 1 0 0$ and $\gamma _ { u } = 1 0 0$  
![](images/848685d86e3b4447e18cad8609582bf9885564bb8dcedfec44054d91772a8b90.jpg)

![](images/abc594a3259e298432672a0b80bd35fc039782d8ee23bd649fe085af624cc2d2.jpg)  
(b) FixMatch+InfoBatch

![](images/c919a58648e2e7ccbca24d486ebd425c8ae5bea9222cc457df28ac064b92fb2e.jpg)  
(c) FixMatch+CDMAD

![](images/1d9beb89eb376c433c43aec0aed6db704441962c295815fff8a7c76c7aa81928.jpg)  
(d) FixMatch+ DyTrim  
Figure 19: t-SNE of representations obtained for the test set of CIFAR-10 using (a) FixMatch, (b) FixMatch+InfoBatch, (c) FixMatch+CDMAD, and (d) FixMatch+DyTrim trained on CIFAR-10-LT under $\gamma _ { l } = 1 0 0$ and $\gamma _ { u } = 1$

## J LIMITATION

A key limitation of our method is its reliance on a task-irrelevant baseline image as a bias indicator. If this baseline image is used as a training sample, it may no longer reflect the accumulated bias, reducing the effectiveness of our debiasing mechanism. Additionally, our framework does not account for architectures with auxiliary classification heads or semi-supervised methods based on mixup-style (Zhang et al., 2017) interpolations, limiting DyTrim’s applicability to these models. Extending our approach to these settings is an interesting avenue for future work.

## K USE OF LLMS

Large language models (LLMs) were used solely to assist with minor language polishing during manuscript preparation. All scientific components of this work, including the design of experiments, data processing, analysis, and interpretation, were carried out entirely by the authors using established computational methods and human expertise, without reliance on automated reasoning or model-generated content.