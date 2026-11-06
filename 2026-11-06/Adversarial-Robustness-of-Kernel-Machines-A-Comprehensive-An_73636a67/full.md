# Adversarial Robustness of Kernel Machines A Comprehensive Analysis

Rahul K. Yadav† Tunir Ghosh† Sudhendu Ahir†

Parthe Pandit†\* Arjun Bhagoji†\*

†Center for Machine Intelligence and Data Science, IIT Bombay

## Abstract

Kernel machines are an important class of models in machine learning. However, the question of how robust modern kernel machines are to adversarial examples has not been studied comprehensively. In this paper, we provide a thorough empirical investigation of the adversarial robustness of kernelbased models for both binary and multiclass classification tasks, via white-box and black-box evasion attacks. We evaluate the native robustness of kernel machines, random feature classifiers, and the relatively newer Recursive Feature Machines. In addition, we provide an algorithm for approximate adversarial training of kernel machines. Our experiments on classification tasks on image and tabular datasets strongly indicate that kernel machines are natively more robust than neural networks, i.e.. their performance degrades more gracefully against adversarial perturbations.

## Contents

1 Introduction 2   
1.1 Main findings 4   
1.2 Prior work on robustness of KMs 4   
2 Preliminaries 5   
2.1 Adversarial examples 6   
2.2 Kernel-based models 6   
3 Main Experimental Results 7   
3.1 Setup Summary . 7   
3.2 KMs are more robust than neural nets 8   
3.3 RFMs are nearly as robust as base KMs 9   
3.4 Robustness of RFCs increases with number of random features 9   
3.5 Robustness of KMs is comparable with adversarially trained neural networks 9   
4 Adversarial training of kernel classifiers 10   
4.1 Convex approximation for binary classification 10   
4.2 Nonconvex approximation for multiclass 11   
4.3 Evaluation of adversarially trained KMs 12   
5 Discussion 12   
5.1 Summary of Ablations 12   
5.2 Are KMs more robust point-wise?. 13   
5.3 Limitations 13   
5.4 Future Work 13   
A Further Technical Details and Setup 17   
A.1 Gradient-free (black-box) attacks using SPSA 17   
A.2 Attacks on Tabular datasets using CaFA 18   
A.3 Kernels preliminaries . 18   
A.4 Adversarial training of NNs 19   
A.5 Adversarial training of Kernel Machines 19   
B Detailed Experimental Setup 19   
B.1 Kernel Machines training details 19   
B.2 Neural Networks training details 20   
B.3 Attacks 20   
B.4 Adversarial training of neural networks 20   
B.5 Adversarial training of KLR 21   
B.6 ν vs. robustness of Matérn Kernels 22   
B.7 Computational efficiency details . 22   
C Detailed Experimental Results 23   
C.1 Adversarial Robustness under $\ell _ { 2 }$ and $\ell _ { \infty }$ Perturbations 23   
C.2 Linf summary for KMNIST/QMNIST vs FCNN and LeNet 23   
C.3 Comparison of KMs with FCNN and LeNet 29   
C.4 Laplacian Results for FMNIST, KMNIST, QMNIST 30   
C.5 Gaussian Results for FMNIST, KMNIST, QMNIST 56   
C.6 Kernels on SVHN and CIFAR-10 83   
C.7 WideResNet vs LeNet 88   
C.8 KMs and adversarially trained KMs vs adversarially trained nets 89   
D Further Ablations 93   
D.1 Results for gradient-free (black-box) attacks . 93   
D.2 Robustness on Tabular datasets 102   
D.3 Results for Matérn Kernel ν versus robust accuracy 103   
D.4 Robust Accuracy variation with iteration in RFMs 104   
D.5 Robust Accuracy variation with length scale (σ) 105   
D.6 Robust Accuracy variation with ridge (λ) 107   
D.7 Number of features in RFC versus robust accuracy 108   
D.7.1 FMNIST Results for Gaussian RFC 108   
D.7.2 KMNIST Results for Gaussian RFC 110   
D.7.3 QMNIST Results for Gaussian RFC 112   
D.7.4 FMNIST Results for Laplace RFC 114   
D.7.5 KMNIST Results for Laplace RFC 116   
D.7.6 QMNIST Results for Laplace RFC 118

## 1 Introduction

In domains such as vision and language, machine learning (ML) models based on neural networks (NNs) are often purported to have so-called superhuman capabilities. However, this claim is brought into question by the prevalence of test-time attacks known as adversarial examples [14, 8], even in limited-information settings [27, 6], and the lack of effective defenses against them [2, 10]. Adversarial examples (§2.1) are minimally-perturbed inputs that induce misclassification in otherwise well-performing classification models. A vast array of attacks and defenses [10] have been proposed over the years, with a bulk of them focusing on NNs. Apart from robust training, in both its empirical |28| and certified variants |25], few defenses have been shown to be effective. Robust training which solves a min-max problem, however, is significantly more computationally expensive than standard training, and can lead to unfavourable trade-offs [51].

In that light, we revisit the question of whether there exists a class of models that is inherently more robust than neural networks. Kernel Machines (KMs) (§2.2) are a popular class of ML models which are seeing a renewed interest, following results showing that in certain regimes, NNs are approximated by KMs [19]. In this paper, we empirically compare neural networks and kernel machines, seeking an answer to:

Is there a hierarchy of adversarial robustness (possibly dataset dependent) between function classes?

Table 1: Overview of datasets, models, and attacks compared in this paper.
<table><tr><td>Choices</td><td>Image data</td><td>Tabular data</td></tr><tr><td>Datasets</td><td>FashionMNIST, KMNIST, QMNIST, SVHN, CIFAR-10</td><td>Adult,Phishing,BankMarketing</td></tr><tr><td>Models</td><td>CNN, FCNN, FCNN-Adv, KRR, KLR, KSVM, RFC, RFM, Adv-KLR</td><td>FCNN, KRR, RFM</td></tr><tr><td>Attack</td><td> $\ell _ { \infty } , \ell _ { 2 } { \mathrm { - b a l l s } }$  with white-box (autoattack), black-box (spsa)</td><td> $\ell _ { \infty } { - } \mathrm { b a l l }$  with cafa</td></tr></table>

In other words, are some classes of models fundamentally harder to attack for a test-time adversary?

Three popular versions of kernel machines exist with different qualities: (i) the standard KMs trained using kernel ridge regression (KRR), Kernel Support Vector Machines (KSVM) or Kernel Logistic Regression (KLR), (ii) Random feature classifiers (RFCs) which approximate KMs in a scalable manner [33], and (iii) Recursive Feature Machines (RFMs) which enhance performance via feature learning [32]. Apart from performance in terms of accuracy, other important properties of KMs such as adversarial robustness have received little attention [40]

![](images/45c0edd73f4c7934021bec1605861178414633a4dac0f95b2c703229f49b3171.jpg)  
Figure 1: Test for robustness as a pointwise property. We plot the empirical CDF of the ratio (KLR/other models) of the smallest $\ell _ { \infty }$ perturbations $\widehat { \varepsilon }$ needed to successfully attack 388 random samples which were classified correctly by all the models from the test set of FashionMNIST. Note the x-axis is in log-scale. See §5.2 for more details.

For most samples, KRR is harder to attack than the FCNN. For the median datum, KLR is more robust than Adv-FCNN and Adv-LeNet. The adversarially trained FCNN, LeNet and KLR were trained with an $\ell _ { \infty }$ (matched) adversary with a budget of $\textstyle \varepsilon = { \frac { 8 } { 2 5 5 } }$ . Adversarially trained KM remains most robust. In the case of FMNIST, KRR and RFM overlap. We have used Gaussian KLR, Laplacian KRR, and Laplacian RFM.

We present a thorough empirical comparison of the robustness of different kernel-based classifiers (See Table 1A in the Appendix for abbreviations.)¹: KMs, RFCs, and RFMs against fully connected neural networks (FCNNs) and convolutional neural networks (CNNs), namely LeNet-5 [24] and WideResNet-28- 10 (WRN-28-10) [50], as well as their robustly trained variants across 8 datasets and several different attacks (Table 1). We provide a fine-grained analysis at 3 levels: (i) robust accuracy, which is the aggregate behaviour across the test dataset across all classes, (ii) one-v-one robust accuracy, which breaks down the robust accuracy across $\binom { C } { 2 }$ binary classification tasks, and (iii) pointwise robustness, which estimates the distance of an individual datum from the decision boundary.

While the adversarial training of KMs is a challenging non-convex problem, we derive an approximate formulation for the case of Lipschitz loss functions (§4). For binary classification this is a convex formulation, however for multiclass problems this yields a minimization problem of a convex function over rank-1 matrices, which is a non-convex domain. We provide an iterative algorithm for training these models along with an empirical analysis.

## 1.1 Main findings

Kernel machines are more robust than neural networks (§3.2): The robustness of KMs degrades more gracefully than NNs as the attack budget ε increases. KMs are consistently more robust than NNs trained in the standard fashion, i.e. without adversarial training. In some cases, the difference is up to 40 percentage points in multiclass classification tasks and 70 percentage points in binary classification tasks, with KMs providing near equivalent performance on clean data (Fig. 2).

Recursive Feature Machines retain the robustness of the initial KMs (§3.3): RFMs are a variant of KMs equipped with feature learning. Our experiments show that RFMs retain nearly all of the robustness of their native KMs, but their robustness can degrade with additional iterations of feature learning.

Random Feature classifiers can be robust at a sufficiently high number of features (§3.4): RFCs are a scalable approximation to KMs which can match their performance in terms of zeroth order metrics such as generalisation error and accuracy. We observe that increasing the number of random features consistently makes these models more robust, except in the case of Laplacian RFCs trained using square loss. This behaviour is stable for Gaussian RFCs whereas it exhibits a large variability for Laplacian RFCs, in part due to the heavy tailed nature of the weight matrices appearing in the Random Fourier map for the Laplacian kernel.

Kernel Machines can be nearly as robust as adversarially trained NNs (§3.5): On certain datasets, KMs, even without adversarial training, have a robustness profile similar to adversarially trained NNs. While this does not hold across all datasets and training budgets, it is a surprising finding even with these caveats. Their robustness can be further improved using our approximate adversarial training method, providing high robust accuracy even under powerful attacks.

Higher robustness of KMs is likely a point-wise property (§5.2): It is instructive to understand, on a per-sample basis, how far a particular data point is from the classification boundary. For a median datum, we find that KLR is more robust than adversarially trained FCNN on FashionMNIST. We use bisection coupled with AutoAttack [11] to find the boundary distance empirically.

In Appendix D, we carry out extensive ablations. We first (i) check for the presence of gradient masking; and (ii) evaluate our models on tabular datasets. We also ablate over kernel classifier parameters such as the (a) smoothness parameter ν in the Matérn kernel, (b) RFM iterations, (c) length scale (σ), (d) ridge penalty (λ), and (e) number of features used in RFC. Overall, this paper motivates further rigorous analyses of the robustness of KMs. It is interesting that interpolation or near-interpolation on training data, which is the behaviour of KRR, may not be at odds with adversarial robustness with respect to test data. In future work, a satisfactory theoretical result would provide a pointwise gap between the margins of KMs and NNs, which could perhaps be due to a flatter loss landscape for the adversary to climb out of.

## 1.2 Prior work on robustness of KMs

The problem of the robustness of kernel-based classifiers has received some limited attention. However, a comprehensive comparative analysis of the adversarial robustness of this class of models across the 3 variants- KMs, RFC, and RFM, listed above has not been carried out before.

Robustness to contamination: KMs with outlier-resilient loss functions have been studied. These include the Cauchy loss [43], Huber loss [41], l1 loss [39], among others [13]. For example, Li et al. [26] show robustness to label noise for KMs with the Neural Tangent Kernel [19] trained using early stopping.

![](images/9eca154d05f1d30c5f5b6c2040215eccae4c98a636805831b27ee41304f9507b.jpg)  
Figure 2: Summary of results for the multiclass FashionMNIST dataset. We compare robust accuracy of 13 models (3 adversarially trained, 10 benign) on FashionMNIST (10 classes). Columns with yellow borders correspond to adversarially trained models with training adversary's attack level highlighted by the red row. Black boxes mark the best robust accuracy across each row. Note the best models for high ε are kernel machines with logistic loss.

Test-time perturbations against kernel machines: Xu et al. [47] showed that KSVMs are robust to point-wise uncertainty at test time, by using an upper bound to the hinge loss with a regularized variant. However, they do not provide any empirical evidence of robustness, and only operate with perturbations in feature space. Hassani and Javanmard [17] carried out a theoretical analysis of the robustness of random feature maps in the overparametrized setting, and show that overparametrization hurts robust generalization in a proportional asymptotic regime where $d , p , n  \infty$ . However, this analysis is more attuned to understanding the RF approximation of KSVMs. Wu et al. [44] provide a method to adversarially train KMs in feature space via doubly stochastic gradients over a convex upper bound of the adversarial loss on the RFC approximation. In the work closest to ours, Wang [40] analyses the empirical robustness of kernel SVMs trained with the linear and Gaussian kernels, and finds they are not robust. Han and Rubinstein [16] study gradient-descent evasion attacks on Gaussian SVMs and find that carefully reduced kernel smoothness substantially increases robustness. We find that KLR, KRR and KSVM with the Laplacian kernel, and the Gaussian kernel, with proper tuning, are robust.

## 2 Preliminaries

We focus on the setting of multiclass classification with C classes. For $\Omega \subset \mathbb { R } ^ { d } .$ let $f : \Omega \to \mathbb { R } ^ { C }$ be a model associated with a classifier $x \mapsto \arg \operatorname* { m a x } _ { i \in [ C ] } f _ { i } ( x )$ for $x \in \Omega$ . Note $f ( x )$ are the logits for a test sample x, and $f _ { i } ( x ) = e _ { i } ^ { \top } f ( x )$ , the $i ^ { \mathrm { t h } }$ logit, for $i \in [ C ]$ . The loss function is represented as $\mathsf { L } : \mathbb { R } ^ { C } \times [ C ] \mapsto \mathbb { R }$

## 2.1 Adversarial examples

At a point x, we say δ is an adversarial perturbation within a lp-budget of $\varepsilon { \mathrm { ~ i f } } , \| \delta \| _ { p } \leq \varepsilon$ and

$$
\begin{array} { r } { \arg \operatorname* { m a x } _ { i \in [ C ] } f _ { i } ( x ) \neq \arg \operatorname* { m a x } _ { i \in [ C ] } f _ { i } ( x + \delta ) . } \end{array}
$$

We also call $x { + \delta }$ an adversarial example. A popular class of methods for finding adversarial perturbations $[ 8 ,$ 14, 28] is to solve the following non-convex optimisation

$$
\operatorname* { m a x } _ { \delta \in \mathbb { R } ^ { d } } \ell ( \delta ) : = \mathsf { L } ( f ( x + \delta ) , \arg \operatorname* { m a x } _ { i \in C } f _ { i } ( x ) )\tag{1a}
$$

$$
\mathrm { s u b j e c t ~ t o } \ \| \delta \| _ { p } \leq \varepsilon , \quad x + \delta \in \Omega\tag{1b}
$$

for some loss function L, using a local search method.

The typical choice for L is the cross-entropy loss. Note that even though $u \mapsto \mathsf { L } ( u , v )$ may be convex for all $v ,$ the $\delta \mapsto \mathsf { L } ( f ( x + \delta ) , v )$ is not convex in general, unless of course $f$ is linear in its input $x .$ In this paper we focus on models that are nonlinear in the input such as NNs and KMs. Note KMs, Eq. (2), are linear in the parameters $\alpha ,$ not in the input x.

Tools for finding adversarial perturbations (also called attacks) are classified based on whether they provide the adversary input-output access $f ,$ called black-box [6, 27] attacks, or access to the implementation of $f ,$ called whitebox attacks. Most white-box attacks, typically solve Eq. (1) approximately via an iterative first order method starting from $\delta = 0$ . In this paper we use a popular tool for white-box attacks: AutoPGD [11] which adaptively selects the step size for the non-convex optimisation in Eq. (1). Athalye et al.[2] recommend also evaluating with black-box attacks to ensure there is no inadvertent gradient masking. We use SPSA [36, 38] (details in §A.1). On tabular data, we use the Cost-aware Feasible Attack (CaFA) [5] which ensures input validity.

## 2.2 Kernel-based models

For a compact domain $\Omega \subset \mathbb { R } ^ { d }$ , let $K : \Omega \times \Omega $ R be a kernel, $i . e . ,$ a bivariate positive definite map. For a collection of vectors $\boldsymbol { X } = \left\{ \boldsymbol { x } _ { i } \right\} _ { i = 1 } ^ { n }$ we denote the kernel matrix with $K ( X , X ) = ( K ( x _ { i } , x _ { j } ) ) _ { 1 \leq i , j \leq n } .$

Kernel machines (KM): A kernel machine is a model with parameter α $\in \mathbb { R } ^ { n \times C }$ , and takes the form

$$
f ( x ) = \sum _ { i = 1 } ^ { n } \alpha _ { i } K ( x , x _ { i } ) .\tag{2}
$$

where $\alpha _ { i } \in \mathbb { R } ^ { C }$ . This class of models are of interest in machine learning because the representer theorem [34] tells us that any solution to a non-parametric statistical estimation problem,

$$
\operatorname* { m i n } _ { f \in \mathbb { H } _ { K } } J \left( \{ x _ { i } , f ( x _ { i } ) , y _ { i } \} _ { i = 1 } ^ { n } \right) + \lambda \cdot R ( \left\| f \right\| _ { \mathbb { H } _ { K } } ^ { 2 } ) ,\tag{3}
$$

with $\lambda > 0$ , must have the form in $\operatorname { E q . } \ ( 2 )$ . Here the search space $\mathbb { H } _ { K }$ is the reproducing kernel hilbert space (RKHS) corresponding to K. Note here that J is an arbitrary loss function, and R is any increasing regularization function. We refer the reader to [18, 35, 37] for a review of kernel machines. We note that the case of $C > 1$ , calls for the use of so-called vector valued RKHSs corresponding to vector valued kernels $K _ { v v } ,$ such that $K _ { v v } ( x _ { i } , x _ { j } )$ is a $C \times C$ matrix and $K _ { v v } ( X , X ) \in \mathbb { R } ^ { n C \times n C }$ is a symmetric positive definite matrix with $C \times C$ blocks. However, in what follows, we have considered $K _ { v v } ( x _ { i } , x _ { j } ) = K ( x _ { i } , x _ { j } ) I _ { C }$ . We train KMs using the square loss (Kernel Ridge Regression), hinge loss (Kernel Support Vector Machine) and logistic loss (Kernel Logistic Regression) (§A.3).

Derivatives: We define $\begin{array} { r } { G _ { 1 } ( x , z ) : = \frac { \partial } { \partial x } K ( x , z ) \in \mathbb { R } ^ { d } } \end{array}$ Similarly $\begin{array} { r } { G _ { 2 } ( x , z ) : = \frac { \partial } { \partial z } K ( x , z ) } \end{array}$ , and $G _ { 1 2 } ( x , z ) : =$ $\begin{array} { r } { \frac { \partial ^ { 2 } } { \partial x \partial z } K ( x , z ) \in \mathbb { R } ^ { d \times d } } \end{array}$ . Observe that $G _ { 1 } ( x , z ) = G _ { 2 } ( z , x )$ , and $G _ { 1 2 } ( x , z ) = G _ { 1 2 } ( z , x ) ^ { \top }$ . We use these for adversarial training of kernel machines later in $\ S 4$

Recursive feature machines (RFM): A new family of kernel machines was proposed by [32] where the kernel function was parameterized by a positive semidefinite matrix $M \in \mathbb { R } ^ { \bar { d } \times d }$ as $K _ { M } ( x , z ) =$ $K ( \sqrt { M } x , \sqrt { M } z )$ , where $\sqrt { M }$ is the unique symmetric square root matrix of a positive semidefinite matrix M, and K is some base kernel. The RFM is the model Eq. (2) with K replaced by $K _ { M }$ , and the trainable parameter M in RFMs is updated recursively using the average gradient outer product (AGOP) of the model, using the training data $\{ x _ { i } \} _ { i = 1 } ^ { n }$

$$
\begin{array} { r } { M _ { t + 1 }  \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } \nabla _ { x } f _ { M _ { t } } ( x _ { i } ) ( \nabla _ { x } f _ { M _ { t } } ( x _ { i } ) ) ^ { \top } } \end{array}\tag{4}
$$

where $f _ { M }$ denotes the model in Eq. (2) with kernel $K = K _ { M }$ , and $\alpha \in \mathbb { R } ^ { n \times C }$ obtained by solving Eq. (14) with $K = K _ { M }$ , and $\nabla _ { x } f _ { M } \in \mathbb { R } ^ { d \times C }$ is the Jacobian. RFMs provide the model a property formally studied as feature learning, which is natively absent in KMs. RFMs are state of the art for certain tasks, such as prediction on tabular datasets [3].

Random feature classifiers (RFCs): Kernel machines for a large class of kernels can be learned via the so-called random features approximation [33, 20, 49]. This approximation makes inference independent of the size of the dataset n, unlike Eq. (2), which may become prohibitive for tasks with large datasets. If K has a special form

$$
K ( x , z ) = \mathbb { E } _ { w \sim \mathbb { P } _ { w } } \varsigma \left( \langle w , x \rangle _ { \mathbb { R } ^ { d } } \right) \cdot \varsigma \left( \langle w , z \rangle _ { \mathbb { R } ^ { d } } \right) ,
$$

then Eq. (2) can be approximated as $f _ { \theta } ( x ) = \langle \theta , \phi ( x ) \rangle _ { \mathbb { R } ^ { m } }$ , where $\begin{array} { r } { \phi ( x ) = \frac { 1 } { \sqrt { m } } \varsigma ( W x ) } \end{array}$ , the vector $\theta =$ $\scriptstyle \sum _ { i = 1 } ^ { n } \alpha _ { i } \phi ( x _ { i } )$ , and the rows of $W \in \mathbb { R } ^ { m \times d }$ are m i.i.d. samples from $\mathbb { P } _ { w } .$ where $\left( \varsigma , \mathbb { P } _ { w } \right)$ depend on K. Several classes of kernel functions such as the Gaussian and Laplacian kernel, can be approximated in this manner.

## 3 Main Experimental Results

Our goal is to test whether kernel-based classifiers, in any of the 3 variants described above, are more robust than neural networks, whether fully connected or convolutional. A summary of the settings considered is in Table 1. Our key findings are i) KMs are more robust than both FCNNs and CNNs (LeNet) on average (§3.2) and point-wise (§5.2); ii) RFMs are nearly as robust as KMs (§3.3); and iii) RFCs become robust with an increase in the number of features (§3.4). These results are captured succinctly in Figs. 1A, 2, 4A and 5A for the case of $\ell _ { \infty }$ and $\ell _ { 2 }$ attacks on the FashionMNIST, KMNIST and QMNIST dataset.

## 3.1 Setup Summary

We briefly summarise our experimental setup here. All experiments were run on a cluster with 8 NVIDIA RTX 6000 GPUs. All further details on hyperparameter searches and attack budgets used are in §B.

Datasets: Our experiments for image data are with FashionMNIST [46], KMNIST [9], QMNIST [48], SVHN [31] and CIFAR-10 [22]. For tabular data, we use the Adult [4], Bank Marketing [30] and Phishing [29] binary classification datasets.

Choice of best model to attack: We define the best model to be the one with the highest benign accuracy on the validation set, across all tunable hyperparameters.

Measurements and comparisons: We measure the robust accuracy of a model given f via

$$
\mathrm { r a } _ { \varepsilon } ( f ) : = 1 - \mathbb { P } \left\{ \exists \delta : \| \delta \| _ { p } \leq \varepsilon , \arg \operatorname* { m a x } _ { i \in [ C ] } f _ { i } ( x + \delta ) \neq y \right\} .
$$

using a plug-in estimator for P and commonly used search criteria for finding δ. We report robust accuracies for both settings - multiclass classification and binary classification. For a C class classification problem we have $\binom { C } { 2 }$ binary classification problems, which allow us to provide a fine-grained analysis.

Kernel machines: We consider kernel machines trained using kernels from the Matérn family (§A.3), of which both the Laplacian and Gaussian kernels are a part. KMs considered in this section were trained using kernel logistic regression (KLR), kernel ridge regression (KRR) and kernel SVM (KSVM), with $M = I _ { d }$ KLR uses the cross-entropy loss, KRR uses the square loss and KSVM uses the hinge loss. For KRR and KSVM, we tuned the length scale σ and the regularisation strength λ by grid search. For KLR, we searched over length scale σ and learning rate. Detailed results for the Laplacian $( \gamma = 1 )$ and Gaussian $( \gamma = 2 )$ kernels are in Appendix C.

RFCs and RFMs: The RFCs were trained using the square loss as well as the cross-entropy loss, each with ridge regularisation. For RFMs, the matrix M is set adaptively rather than fixed to ${ { I } _ { d } } \ ( \ S 2 . 2 )$ , and the models were trained with the square loss. The RFMs were tuned for up to 5 iterations, with search over length scale σ and regularisation strength λ being the same as in KRR.

Neural networks. We train both fully connected neural networks (FCNNs) and Convolutional neural networks (CNNs). For the latter, the modified LeNet5 (using ReLU activation and max-pooling layers) [24] and WideResNet-28-10 [50] architectures were used. Runtime comparisons are in Table 3A.

Attacks: For a rigorous evaluation of robustness, and so as not to fall into common pitfalls [7], we use two different white-box attacks, as well as a black box attack. In all cases, we use the cross-entropy loss. In the white-box case, we consider both i) a fixed iteration projected gradient descent (PGD) attack [28] and ii) AutoAttack, using apgd-ce [11, 12] (the standard in benchmarks such as RobustBench [10]), and report the lower of the two robust accuracy numbers.

## 3.2 KMs are more robust than neural nets

For multi-class classification problems, we see that the robustness of both Laplacian and Gaussian KRR degrades more gracefully than that of both FCNNs and LeNets (Figs. 1A, 2, 4A and 5A), particularly in the case of FMNIST. A comprehensive evaluation for all the 1v1 sub-problems for FashionMNIST. KMNIST and QMNIST is presented in Figs. 11A and 36A, which clearly establishing that, as the attack budget increases for both $\ell _ { \infty }$ and $\ell _ { 2 }$ attacks, kernel machines are more robust in a majority of cases. These demonstrate that while the Laplace kernel seems to be better choice for robustness, the Gaussian kernel also offers robustness benefits. Perhaps surprisingly, fine-grained class-wise analysis reveals that the robustness of Laplacian KRR is even higher than of the LeNet as the perturbation budget increases (Figs. 12A and 57A). In terms of the benign accuracy $( \varepsilon = 0 )$ , we find that the LeNet is marginally better on these datasets, but KRR can outperform the FCNN. For example, in the case of 5v9 binary classification problem on FMNIST, KRR has 0.8 percentage points better benign accuracy. We further compare LeNet with the substantially larger WRN-28-10 (§C.7). Despite its higher clean accuracy, WRN-28-10 loses robust accuracy faster than LeNet as the perturbation budget increases (Table 5A). Taken together, these results suggest that the robustness advantage of KRR over LeNet is not due to the model size used for comparison.

KSVMs: Kernel SVMs (both Laplacian and Gaussian) also outperform FCNNs and LeNets in terms of robust accuracies for both multiclass (Figs. 1A, 2, 4A and 5A) and binary classification problems (Figs. 15A and 16A). At the same time they can be better than their corresponding KRR. For example, in Fig. 2, Laplacian SVM performs better than the Laplacian KRR for FMNIST, while for QMNIST (Fig. 5A) Gaussian SVM is better than the Gaussian KRR. KSVMs also remain competitive in terms of benign accuracy. For instance, in FMNIST, the Laplacian SVM is 0.6 percentage points better than FCNN on the same task.

KLR: Like KRR and KSVM, the robust accuracy of KLR degrades quite gracefully as compared to FCNNs and LeNets. They are also quite robust in comparison to KRRs and KSVMs as seen in Figs. 2, 4A and 5A. For example, on KMNIST at $\varepsilon = 4 0 / 2 5 5$ , Gaussian KLR retains 73.4% accuracy, whereas adversarially trained LeNet, adversarially trained FCNNs, Gaussian KRR, and Gaussian SVM retain 28.0%, 20.1%, 12.5%, and 16.2% respectively. Laplacian KLRs also show the similar behaviour, but are less robust in comparison to Gaussian KLR. In terms of benign accuracy, however, KLR lags behind almost all the other models. For example, in the case of FashionMNIST, adversarially trained Gaussian KLR has the lowest benign accuracy followed by Gaussian KLR, Gaussian KRR, adversarially trained FCNN and Laplacian KLR. Similar observations can be seen in the case of KMNIST and QMNIST. The 1v1 results show both Laplacian and Gaussian KLR being more robust than all the other benign trained models at larger perturbation radii. In addition, for the 0v8 problem on FashionMNIST, Gaussian KLR is having 70.3 percentage points more accuracy than FCNN at the $\begin{array} { r } { \varepsilon = \frac { 2 0 } { 2 5 5 } } \end{array}$ . The 1v1 benign accuracies are competitive with FCNN and LeNet in most cases as seen in Figs. 9A, 10A, 35A and 56A. Overall, we find that the Gaussian KLR is more robust than the Laplacian KLR. Similar observations hold across the other datasets.

## 3.3 RFMs are nearly as robust as base KMs

RFMs allow for feature learning in kernel machines. As observed in previous work [32], we find this can lead to improved accuracy on the base benign task. However, the improved feature learning does not always translate to improved robustness (Fig. 20A for Laplacian and Fig. 42A for Gaussian). In particular, we find that KRR (with M = I in Eq. (4)) is often more robust when the RFM has improved benign accuracy, i.e. some feature learning has taken place. In the case of multi-class classification (Figs. 1A, 2, 4A and 5A), there is a clear separation, with KRR being more robust for the KMNIST and QMNIST datasets across budgets and attack types. For FMNIST, the RFM has competitive robust accuracy.

RFMs vs NNs: Fine-grained class-wise analysis (Figs. 13A and 37A) shows that RFMs are almost always more robust compared to the FCNNs. Additionally, unlike in the case of KRR, RFMs can have higher benign accuracy than the FCNNs, and are competitive with the LeNet. Thus, RFMs are a good choice of model, at least for simpler datasets, offering attractive best-of-both worlds performance.

## 3.4 Robustness of RFCs increases with number of random features

We observe that random feature classifiers (RFCs) suffer in terms of benign and robust accuracies compared to the base KMs when the number of features is inadequate. To compare RFCs with other models, we use 5000 and 20000 features for binary classification problems using square loss (Fig. 45A), while in the multiclass case we carry out an extensive ablation over the number of features (§D.7) to study its effect on robust accuracy. In multiclass case, RFCs are trained using both cross-entropy and square loss.

For Gaussian RFCs, trained with either the cross-entropy or the square loss, robustness is low when the number of features is small, but increases steadily as the number of features grows, in both the 1v1 and multiclass settings. The same trend holds under both SPSA and PGD, and the robust accuracy under SPSA is consistently higher hence, we can say for Gaussian RFCs, no gradient masking is happening. With enough features, Gaussian RFCs become more robust than FCNNs and LeNet in many cases.

However, we find that Laplacian RFCs trained with either loss exhibit significant gradient masking. SPSA attacks reduce their robust accuracy substantially compared to white-box attacks (§D.1). Robust accuracy under first-order gradient-based attacks therefore overestimates their true robustness and should not be taken at face value. This includes the high robust accuracy of the cross-entropy-trained Laplacian RFC in Fig. 89A, which holds up under white-box attacks even at very large budgets. For the cross-entropy loss, the robustness still increases with the number of features (when attacked against SPSA and PGD). For the square loss the robust accuracy of Laplacian RFCs varies erratically with the number of features, and more features do not yield more robustness, unlike for Gaussian RFCs. These observations warrant further theoretical investigation.

Overall, these results call into question the practical validity of the results from Hassani and Javanmard [17] which indicates that overparametrisation hurts robustness for RFCs. A plausible explanation is that their feature isotropy assumption yields pessimistic results. Thus, the implications of overparametrisation for robustness must be revisited.

## 3.5 Robustness of KMs is comparable with adversarially trained neural networks

The dominant approach to training robust NNs has been adversarial training [28] and its variants [51, 42]. While adversarial training is significantly more computationally expensive than benign training, it is instructive to compare whether NNs, particularly FCNNs, match the intrinsic robustness of KMs post adversarial training. Experimental details for adversarial training are in §A.4. In the multiclass setting for $\ell _ { \infty }$ in Figs. 2, 4A and 5A, we see that for the FMNIST, KMNIST and QMNIST datasets that Gaussian KLR outperforms adversarially trained neural networks. In the case of $\ell _ { 2 }$ attacks, as in Fig. 1A, we can see that Laplacian KRR performs at par with adversarially trained FCNN and LeNet (both of which are trained at $\ell _ { 2 }$ perturbation radius of 1). Fine-grained class-wise analysis for the Laplacian kernel compared to $\ell _ { 2 }$ training at $\varepsilon = 1$ and $\ell _ { \infty }$ training at $\textstyle \varepsilon = { \frac { 6 } { 2 5 5 } } \ ( \mathrm { F i g . ~ 7 0 A } )$ reveal that depending on the threat model, and the specific class pair, KRR can be more robust than adversarially trained neural networks

## 4 Adversarial training of kernel classifiers

A particularly important problem which remains open is that of adversarially training kernel machines. Prior approaches [40, 44] considered the problem in the feature space, and hence do not perform adversarial training in the input space. Both these methods use the Xu et al. [47] upper bound to create an Euclidean boundary in feature space which is often over-conservative. In what follows, we provide a method for adversarially training KMs by approximation, via an upper bound on the robust loss.

Consider adversarial training over the RKHS, for $\varepsilon > 0$

$$
\operatorname* { m i n } _ { f \in \mathbb { H } _ { K } } \sum _ { i = 1 } ^ { n } \operatorname* { m a x } _ { \delta _ { i } \in \mathbb { R } ^ { d } } \ell _ { i } ( f ( x _ { i } + \delta _ { i } ) ) + \frac { \lambda } { 2 } \left\| f \right\| _ { \mathbb { H } _ { K } } ^ { 2 }\tag{5a}
$$

$$
\mathrm { s u b j e c t ~ t o } \ \| \delta _ { i } \| _ { p } \leq \varepsilon , \forall i \in [ n ]\tag{5b}
$$

$p \geq 1$ . We look at binary classification and multiclass classification separately below.

## 4.1 Convex approximation for binary classification

The objective function in Eq. (5) no longer has the form of Eq. (3) whereby there is no guarantee of a finitely represented solution like Eq. (2). However one could consider an approximation by minimizing a linearization for small ε. By Hölders inequality, for a twice differentiable loss function and predictor function, this yields

$$
\displaystyle \operatorname* { m i n } _ { f \in \mathbb { H } } \sum _ { i = 1 } ^ { n } \ell _ { i } ( f ( \boldsymbol { x } _ { i } ) ) + \varepsilon \left\| \ell _ { i } ^ { \prime } ( f ( \boldsymbol { x } _ { i } ) ) ^ { \top } \nabla f ( \boldsymbol { x } _ { i } ) \right\| _ { q } + \frac { \lambda } { 2 } \left\| f \right\| _ { \mathbb { H } } ^ { 2 }\tag{6}
$$

where $\textstyle { \frac { 1 } { q } } + { \frac { 1 } { p } } = 1$ , and $\ell _ { i } ^ { \prime } ( f ( x _ { i } ) ) \in \mathbb { R } ^ { C }$ and $\nabla f ( x _ { i } ) \in \mathbb { R } ^ { C \times d }$ . The following proposition provides a way to represent the solution to problem (6). The proof is a direct consequence of [52, Thm. 2] with $s = 1$

Proposition 1. Consider Eq. (5), where $K \in C ^ { 2 }$ and the loss function $\ell \in C ^ { 2 }$ . Then, any solution f to Eq. (6), satisfies

$$
f ( x ) = \sum _ { i = 1 } ^ { n } \alpha _ { i } K ( x , x _ { i } ) + \beta _ { i } ^ { \top } G _ { 2 } ( x , x _ { i } )\tag{7}
$$

for some coefficients $\alpha _ { i } \in \mathbb { R }$ and $\beta _ { i } \in \mathbb { R } ^ { d }$ . Furthermore, for such an $f ,$ we have that $\left\| f \right\| _ { \mathbb { H } _ { K } } ^ { 2 }$ equals

$$
\sum _ { i , j } \alpha _ { i } \alpha _ { j } K ( x _ { i } , x _ { j } ) + 2 \alpha _ { i } \beta _ { j } ^ { \top } G _ { 2 } ( x _ { i } , x _ { j } ) + \beta _ { i } ^ { \top } G _ { 1 2 } ( x _ { i } , x _ { j } ) \beta _ { j } .
$$

The assumption $K \in C ^ { 2 }$ implies that $f \in \mathcal H$ is twice differentiable. Along with l' bounded, this allows the 2nd-order remainder term in the Taylor series expansion to be bounded and can vanish for small ε.

A key point to note however, is that solving (6) with parameterization in Eq. (7) may lead to a non-convex optimization problem, since the second term in the objective function is non-convex in $f .$ With an additional assumption: $\| \ell _ { i } ^ { \prime } ( f ( x _ { i } ) ) \| \leq L$ , which holds for Lipschitz losses, one may obtain the convex approximation below which can be solved efficiently.

$$
\operatorname* { m i n } _ { f \in \mathbb { H } _ { K } } \sum _ { i = 1 } ^ { n } \ell _ { i } ( f ( x _ { i } ) ) + L \varepsilon \left\| \nabla f ( x _ { i } ) \right\| _ { q } + \frac { \lambda } { 2 } \left\| f \right\| _ { \mathbb { H } _ { K } } ^ { 2 }\tag{8}
$$

We take a closer look at the case of $p = \infty$ below.

$\ell _ { \infty }$ perturbations: Consider a bounded $\ell _ { \infty }$ adversary, and the logistic loss $\ell ( \hat { y } , y ) = \log ( 1 + e ^ { - y \hat { y } } )$ for $y \in \{ - 1 , 1 \}$ , for which $| \ell _ { i } ^ { \prime } ( f ( x _ { i } ) ) | \leq 1$ 2

$$
\operatorname* { m i n } _ { f \in \mathbb { H } } \sum _ { i = 1 } ^ { n } \ell _ { i } ( f ( x _ { i } ) ) + \varepsilon \| \nabla f ( x _ { i } ) \| _ { 1 } + \frac { \lambda } { 2 } \| f \| _ { \mathbb { H } } ^ { 2 } .\tag{9}
$$

The dual formulation of this is derived as follows. Define $E f = ( f ( x _ { i } ) ) _ { i = 1 } ^ { n }$ and $D f = ( \nabla f ( x _ { i } ) ) _ { i = 1 } ^ { n }$ . Using $\mathcal { C } = \left\{ \delta \in \mathbb { R } ^ { n \times d } : | \delta _ { i j } | \leq \varepsilon \right\}$ , and $\varepsilon \| D f \| _ { 1 } = \operatorname* { m a x } _ { \delta \in { \mathcal { C } } } \langle D f , \delta \rangle$ , we obtain the saddle point problem

$$
\operatorname* { m i n } _ { f \in \mathbb { H } } \operatorname* { m a x } _ { \delta \in \mathcal { C } } \mathcal { L } ( f , \delta ) = \sum _ { i } \ell _ { i } ( f ( x _ { i } ) ) + \frac { \lambda } { 2 } \| f \| _ { \mathbb { H } } ^ { 2 } + \langle D f , \delta \rangle .
$$

Starting from zero, all primal iterates have the form $\begin{array} { r } { f _ { t } ( x ) = \sum _ { i } \alpha _ { t , i } K ( x , x _ { i } ) + \beta _ { t , i } G _ { 2 } ( x , x _ { i } ) } \end{array}$ . Consequently, we update coefficients as follows. Sample a uniform subset $B _ { t } \subseteq \{ 1 , \dots , n \}$ of size b. Let $P _ { B _ { t } }$ retain the sampled rows and set the others to zero.

$$
\begin{array} { r } { \alpha _ { t + 1 } = \alpha _ { t } - \eta _ { t } \frac { n } { b } P _ { B _ { t } } \big ( s ( E f _ { t } ) - y \big ) , } \end{array}\tag{10a}
$$

$$
\beta _ { t + 1 } = \beta _ { t } - \eta _ { t } \frac { n } { b } P _ { B _ { t } } \delta _ { t } ,\tag{10b}
$$

$$
\delta _ { t + 1 } [ B _ { t } ] = \operatorname * { c l i p } _ { [ - \varepsilon , \varepsilon ] } \big ( \delta _ { t } [ B _ { t } ] + \eta _ { t } \frac { n } { b } \nabla f _ { t } ( X [ B _ { t } ] ) \big ) .\tag{10c}
$$

$s ( t ) = ( 1 + e ^ { - t } ) ^ { - 1 }$ . §A.5 provides details on how this in fact tracks a SGD in the RKHS. We have a matrix-free implementation of the above updates which evaluates $f _ { t } ( x _ { i } )$ and $\nabla f _ { t } ( x _ { i } )$ only on the minibatch $x _ { i } \in B _ { t }$ which can be implemented in $O ( n ( d + 1 ) )$ memory.

We use a diminishing step size $\eta _ { t } = \eta _ { 0 } ( t { + } 1 ) ^ { - p }$ and optionally return the weighted average $\begin{array} { r } { \bar { f } _ { T } = \frac { \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } f _ { t } } { \sum _ { r = 1 } ^ { T - 1 } n _ { t } } } \end{array}$ ∠t=0 Convergence is only guaranteed for $\lambda > 0$ For $\lambda = 0$ along with early-stopping may be implemented, but needs a separate analysis. We also note that $\varepsilon = 0$ recovers the kernel logistic regression which has the form of a usual kernel machine since $\beta = 0$

## 4.2 Nonconvex approximation for multiclass

The same procedure along with [52, Thm. 2] applied to (6) for vector-valued RKHSs, where $K _ { C } ( x , z ) =$ $K _ { C } ( z , x ) ^ { \top } \in \mathbb { R } ^ { C \times C }$ , which consists of multioutput functions $f : \mathbb { R } ^ { d }  \mathbb { R } ^ { C }$ from an RKHS $\mathbb { H } _ { K _ { C } } ^ { C }$ would have representation of the form, $\begin{array} { r } { \widehat { f } ( x ) = \sum _ { i = 1 } ^ { n } K ( x , x _ { i } ) \alpha _ { i } + G _ { 2 } ( x , x _ { i } ) ^ { \top } \beta _ { i } \in \mathbb { R } ^ { C } } \end{array}$ where $\alpha _ { i } \in \mathbb { R } ^ { C } , G _ { 2 } ( x , z ) \in$ $\mathbb { R } ^ { d \times C \times \tilde { C } }$ is the Jacobian, and $\beta _ { i } \in \mathbb { R } ^ { d \times C }$ . Note the resulting function has $n c ( d + 1 )$ unknown coefficients that need to be trained.

However for $K _ { C } ( x , z ) = K ( x , z ) I _ { C } , { \mathrm { i . e } }$ , isotropic and separable kernels, Eq. (6) yields simpler solutions with $2 n c + n d$ coefficients. Suppose l is a twice differentiable loss function, and $K$ is twice differentiable, consider the problem of learning $f : \mathbb { R } ^ { d }  \mathbb { R } ^ { C }$ via Eq. (6) where $\nabla f : \mathbb { R } ^ { d }  \mathbb { R } ^ { C \times d }$ is the Jacobian of $f ,$ and $\ell ^ { \prime } : \mathbb { R } ^ { C } \times [ C ] \to \mathbb { R } ^ { \breve { C } }$ is the gradient of the loss function. Finally, define $M = { \left| \begin{array} { l l } { K } & { G } \\ { G ^ { \top } } & { H } \end{array} \right| } \in \mathbb { R } ^ { n \times n \times ( d + 1 ) \times ( d + 1 ) }$ where $\ b { K } \in \mathbb { R } ^ { n \times n }$ is the kernel matrix, $G \in \mathbb { R } ^ { n \times n \times d }$ is the gradient tensor $\textstyle { \frac { \partial } { \partial z } } K ( x , z ) \big | _ { z = x _ { i } }$ , and $\pmb { H } \in \mathbb { R } ^ { n \times n \times d \times d }$ is the second-derivative tensor $\frac { \partial ^ { 2 } } { \partial x \partial z } K ( x , z ) | _ { z = x _ { i } }$ . We also use einsum – Einstein notation for expressing summation over tensors. We now state our result.

Proposition 2. There exists an optimal solution f to problem (6) of the form

$$
f ( \boldsymbol { x } ) = \sum _ { i = 1 } ^ { n } \mu _ { i } K ( \boldsymbol { x } , \boldsymbol { x } _ { i } ) + \gamma _ { i } \theta _ { i } ^ { \top } G _ { 2 } ( \boldsymbol { x } , \boldsymbol { x } _ { i } ) \in \mathbb { R } ^ { C } ,\tag{11}
$$

where $\mu _ { i } , \gamma _ { i } \in \mathbb { R } ^ { C }$ and $\theta _ { i } \in \mathbb { R } ^ { d }$ . Furthermore, if M is positive definite, then every optimal solution must have the above form. For such a solution, $\| f \| _ { \mathbb { H } } ^ { 2 }$ equals $\begin{array} { r l } { \sum _ { i , j } \gamma _ { i } ^ { \top } \gamma _ { j } \cdot \theta _ { i } ^ { \top } G _ { 1 2 } ( x _ { i } , x _ { j } ) \theta _ { j } + \mu _ { i } ^ { \top } \mu _ { j } K ( x _ { i } , x _ { j } ) + } & { { } } \end{array}$ $\mu _ { i } ^ { \top } \gamma _ { j } \theta _ { j } ^ { \top } ( G _ { 2 } ( x _ { i } , x _ { j } ) + G _ { 1 } ( x _ { i } , x _ { j } ) )$

Observe that the solution above only has 2nc + nd coefficients, far fewer than the general case $n c ( d + 1 )$ Since the objective function in problem (6) is not convex in $f _ { i }$ we consider a convex upper bound under the assumption of Lipschitz losses whereby $\| \ell ^ { \prime } \| \leq L$

$\ell _ { \infty }$ adversary: For the crossentropy loss we have $\ell ^ { \prime } ( \hat { y } , y ) = \mathsf { s o f t m a x } ( \hat { y } ) - \mathsf { o n e h o t } ( y )$ , which satisfies $\| \ell ^ { \prime } ( \hat { y } , y ) \| _ { 1 } \leq 2$ , whereby we can approximate $\left\| \ell _ { i } ^ { \prime } ( f ( x _ { i } ) ) ^ { \top } \nabla f ( x _ { i } ) \right\| _ { 1 } \leq 2 \left\| \nabla f ( x _ { i } ) ^ { \top } \right\| .$ where $\| A \| _ { 1 }$ is the induced 1 → 1-norm. Thus $\left\| \nabla f ( x _ { i } ) ^ { \top } \right\| _ { 1 }$ is the maximum absolute row sum of $\nabla f ( x _ { i } )$ . We propose solving

$$
\operatorname* { m i n } _ { f \in \mathbb { H } } \sum _ { i = 1 } ^ { n } \ell _ { i } ( f ( x _ { i } ) ) + 2 \varepsilon \| \nabla f ( x _ { i } ) ^ { \top } \| _ { 1 } + \frac { \lambda } { 2 } \| f \| _ { \mathbb { H } } ^ { 2 } ,\tag{12}
$$

subject to constraint Eq. (11).

Observe that although the above problem has a convex objective function, the constraint set is non-convex. Our iterative algorithm, derived below, to solve the above problem is based on delayed projections similar to [1]. We propose Algorithm 1 to solve this problem. It involves a projection step every few epochs which requires solving a non-convex problem solved approximately using Algorithm 2. More details on this methodology are provided in §A.5

## 4.3 Evaluation of adversarially trained KMs

We use the stochastic primal-dual procedure described above to train the adversarially regularized kernel classifier, which we denote by MARK. Starting from zero primal and dual variables, at each iteration we sample a minibatch of training examples and update the predictor coefficients according to Eqs. (10). The dual variables are updated using the input gradients of the current classifier and are subsequently projected element-wise onto the admissible $\ell _ { \infty }$ perturbation set $[ - \varepsilon , \varepsilon ]$ . We use the matrix-free implementation described above, which requires evaluating the predictor and its input gradients only on the current minibatch. The results in this section apply only to twice differentiable losses and kernels like Gaussian KLR. Figs. 1, 2 and 4A to 7A show that adversarial training of KM boosts their robustness as expected. We also have a direct comparison between KLR and the adversarially trained KLR in Figs. 67A to 69A.

## 5 Discussion

Through our experiments we established that KMs are more robust than both FCNN and CNNs over most vision datasets we consider.

## 5.1 Summary of Ablations

We also carried out a limited number of experiments on the SVHN and CIFAR-10 datasets (§C.6). We find that KMs are consistently more robust than FCNN and CNN over CIFAR-10. We show that in 4 out of 5 randomly chosen 1v1 classifications from CIFAR-10, KMs out-perform FCNNs by upto 4% (Figs. 63A to 66A). LeNets however, show comparable performance on CIFAR-10 for $\ell _ { 2 }$ and improved robustness on $\ell _ { \infty }$ attacks. We also find that the Gaussian RFM outperforms all the models on SVHN (Fig. 62A), as it improves accuracy of the already robust base kernel by 20 percentage points (Table 4A)

To measure robustness accurately, we check for gradient masking, as sometimes robustness measured under white-box attacks can be overestimated when a model produces uninformative or misleading gradients, causing gradient-based attacks to fail. To validate our PGD results, we perform black-box SPSA attacks (§D.1), finding no evidence for gradient masking overall, except in the case of Laplacian RFCs, where SPSA yields lower robust accuracy than white-box attacks. We attribute this to the heavy-tailed spectral distribution of the Laplace kernel. Laplace RFCs sample frequencies from a heavy-tailed (Cauchy) distribution. Gaussian RFCs also sample some large frequencies, but PGD can recover from uninformative gradients over multiple steps. For Laplace RFCs this recovery is much weaker, and we expect the same for RFCs drawn from other heavy-tailed distributions. The gradient estimation done by SPSA avoids this issue. We also investigate the effect of variation in kernel parameters like the ν in Matérn Kernels (§D.3), length scale (σ) (§D.5), ridge regularisation (λ) (§D.6). For RFMs we additionally study the effect of feature learning by varying number of iterations (§D.4). We also show variation in robustness with number of features in RFC models (§D.7)

For changes in kernel parameters we see that increase in smoothness (ν) reduces robustness (Fig. 77A) while a lower length scale increases robustness (Figs. 79A and 80A). The ridge parameter (λ) is not particularly consequential for robustness except at much larger values, where it aids robustness at the cost of benign accuracy (Fig. 81A), though the exact values are dataset dependent. In the case of RFMs more feature learning at a higher number of iterations adversely affects robustness as demonstrated for the Laplacian RFM (Fig. 78A).

## 5.2 Are KMs more robust point-wise?

Ideally, we would like a certificate that some types of KMs are provably more robust than NNs. However, this certification is known to be hard for NNs [21, 25], and made challenging for KMs by the non-convex nature of the boundary they induce. We take a direct, but approximate, approach. Let us define the minimal perturbation needed for a successful attack,

$$
{ \widehat { \varepsilon } } ( f , ( x , y ) ) = \operatorname* { m i n } _ { \delta \in \mathbb { R } ^ { d } } \left\{ \| \delta \| _ { p } \ : \ \arg \operatorname* { m a x } _ { i \in C } f _ { i } ( x + \delta ) \neq y \right\}
$$

The quantity above is calculated approximately by using a bisection method on top of Eq. (1). For a fixed f and g, and $( X , Y ) \sim \mathbb { P }$ , the ratio $\frac { \hat { \varepsilon } ( f , ( X , Y ) ) } { \hat { \varepsilon } ( g , ( X , Y ) ) }$ is a random variable. We plot the CDF of this random variable in Figs. 1, 6A and 7A for different pairs $( f , g )$ , where f is the KLR for 3 datasets. We see that adversarially trained KMs are most pointwise robust of all the models we consider.

## 5.3 Limitations

We identify 5 key limitations of our exploration in this paper. First, since the main attacks considered have been tuned to effectively attack NNs, they may over-estimate the robustness of KMs. However we believe this is unlikely, as KMs should have smooth loss surfaces for the adversary, and first-order attacks would work well. Second, we only consider $\ell _ { p }$ adversaries for vision data, and it would be instructive to consider others like StAdv [45] and ReColor [23]. Third, we only consider less complex datasets where KMs are competitive with NNs. This is, however, a limitation that is out of the scope of this paper, and we believe there the tasks we consider are those where lightweight KMs are a good alternative. Additionally, we do not provide any theoretical guarantees that KMs are more robust than NNs. Finally, we do not fully characterise the phenomena of gradient masking in Laplacian ${ \mathrm { R F C s } } ,$ and the impact of training loss on robustness with respect to the number of features in RFCs.

## 5.4 Future Work

In addition to directly addressing the limitations discussed above, we believe there are several fruitful directions for future work. From a theoretical perspective, obtaining a version of the Representer Theorem for exact robust training and finding KM-specific bounds on robustness are particularly important.

On tabular data, it would be interesting to compare KMs against tree-based methods such as gradientboosted trees, which outperform neural networks [15]. Finally, analysing the adversary's loss landscape, and the confidence of KMs on adversarial examples, and out-of-distribution examples in general, can provide further evidence of their enhanced robustness.

## Acknowledgements

All authors acknowledge funding from the SBI Foundation Hub at IIT Bombay. PP acknowledges funding for compute from DST INSPIRE faculty fellowship, Schmidt Sciences, Amazon, and the Nilekani Research Fund. AB acknowledges funding from the ANRF ARG MATRICS program.

## AI Use Statement

In this work, we used generative AI tools for coding an implementation of Algorithm 1 and Algorithm 2. A very preliminary draft of §4 was generated using a generative AI tool, which was then completely rewritten. Apart from this, we have not used generative AI tools for writing, and the rest of the required disclosure tasks are not applicable to this work. We did not use generative AI tools for any tasks with recommended disclosure. We have reviewed all AI-assisted work. LLM-generated code was verified and tested for correctness by two authors. We take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.

## References

[1] A. Abedsoltan, S. Ma, P. Pandit, and M. Belkin. Fast training of large kernel models with delayed projections. arXiv preprint arXiv:2411.16658, 2024

[2] A. Athalye, N. Carlini, and D. Wagner. Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. In International conference on machine learning, pages 274–283. PMLR, 2018.

[3] D. Beaglehole, D. Holzmüller, A. Radhakrishnan, and M. Belkin. xrfm: Accurate, scalable, and interpretable feature learning models for tabular data. arXiv preprint arXiv:2508.10053, 2025.

[4] B. Becker and R. Kohavi. Adult. UCI Machine Learning Repository, 1996. DOI: https://doi.org/10.24432/C5XW20.

[5] M. Ben-Tov, D. Deutch, N. Frost, and M. Sharif. CaFA: Cost-aware, feasible attacks with database constraints against neural tabular classifiers. In 2024 IEEE Symposium on Security and Privacy (SP), pages 1345–1364. IEEE, 2024.

[6] A. N. Bhagoji, W. He, B. Li, and D. Song. Practical black-box attacks on deep neural networks using efficient query mechanisms. In Proceedings of the European conference on computer vision (ECCV), pages 154–169, 2018.

[7] N. Carlini, A. Athalye, N. Papernot, W. Brendel, J. Rauber, D. Tsipras, I. Goodfellow, A. Madry, and A. Kurakin. On evaluating adversarial robustness. arXiv preprint arXiv:1902.06705, 2019.

[8] N. Carlini and D. Wagner. Towards evaluating the robustness of neural networks. In IEEE Symposium on Security and Privacy, pages 39–57, 2017.

[9] T. Clanuwat, M. Bober-Irizar, A. Kitamoto, A. Lamb, K. Yamamoto, and D. Ha. Deep learning for classical japanese literature, 2018.

[10] F. Croce, M. Andriushchenko, V. Sehwag, E. Debenedetti, N. Flammarion, M. Chiang, P. Mittal, and M. Hein. Robustbench: a standardized adversarial robustness benchmark. Advances in Neural Information Processing Systems, 2020.

[11] F. Croce and M. Hein. Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. In ICML, 2020

[12] F. Croce and M. Hein. Mind the box: l1-apgd for sparse adversarial attacks on image classifiers. In ICML, 2021.

[13] Y. Feng, X. Huang, L. Shi, Y. Yang, and J. A. Suykens. Learning with the maximum correntropy criterion induced losses for regression. Journal of Machine Learning Research, 16(30):993–1034, 2015.

[14] I. J. Goodfellow, J. Shlens, and C. Szegedy. Explaining and harnessing adversarial examples. International Conference on Learning Representations, 2015.

[15] L. Grinsztajn, E. Oyallon, and G. Varoquaux. Why do tree-based models still outperform deep learning on typical tabular data? Advances in neural information processing systems, 35:507–520, 2022.

[16] Y. Han and B. I. P. Rubinstein. Adequacy of the gradient-descent method for classifier evasion attacks. arXiv preprint arXiv:1704.01704, 2017.

[17] H. Hassani and A. Javanmard. The curse of overparametrization in adversarial training: Precise analysis of robust generalization for random features regression. The Annals of Statistics, 52(2):441– 465, 2024.

[18] T. Hofmann, B. Schölkopf, and A. J. Smola. Kernel methods in machine learning. The Annals of Statistics, 36(3):1171–1220, 2008.

[19] A. Jacot, F. Gabriel, and C. Hongler. Neural tangent kernel: Convergence and generalization in neural networks. Advances in neural information processing systems, 31, 2018.

[20] P. Kar and H. Karnick. Random feature maps for dot product kernels. In Artificial intelligence and statistics, pages 583–591. PMLR, 2012.

[21] G. Katz, C. Barrett, D. L. Dill, K. Julian, and M. J. Kochenderfer. Reluplex: An efficient smt solver for verifying deep neural networks. In International conference on computer aided verification, pages 97–117. Springer, 2017.

[22] A. Krizhevsky and G. Hinton. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.

[23] C. Laidlaw and S. Feizi. Functional adversarial attacks, 2019.

[24] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

[25] L. Li, T. Xie, and B. Li. Sok: Certified robustness for deep neural networks. In IEEE Symposium on Security and Privacy, pages 1289–1310, 2023.

[26] M. Li, M. Soltanolkotabi, and S. Oymak. Gradient descent with early stopping is provably robust to label noise for overparameterized neural networks. In International conference on artificial intelligence and statistics, pages 4313–4324. PMLR, 2020.

[27] Y. Liu, X. Chen, C. Liu, and D. Song. Delving into transferable adversarial examples and black-box attacks. International Conference on Learning Representations, 2017.

[28] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu. Towards deep learning models resistant to adversarial attacks. International Conference on Learning Representations, 2018.

[29] R. Mohammad and L. McCluskey. Phishing Websites. UCI Machine Learning Repository, 2012. DOI: https://doi.org/10.24432/C51W2X.

[30] S. Moro, P. Rita, and P. Cortez. Bank Marketing. UCI Machine Learning Repository, 2014. DOI: https://doi.org/10.24432/C5K306.

[31] Y. Netzer, T. Wang, A. Coates, A. Bissacco, B. Wu, and A. Y. Ng. Reading digits in natural images with unsupervised feature learning. In NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011, 2011.

[32] A. Radhakrishnan, D. Beaglehole, P. Pandit, and M. Belkin. Mechanism for feature learning in neural networks and backpropagation-free machine learning models. Science, 383(6690):1461–1467, 2024.

[33] A. Rahimi and B. Recht. Random features for large-scale kernel machines. Advances in neural information processing systems, 20, 2007.

[34] B. Schölkopf, R. Herbrich, and A. J. Smola. A generalized representer theorem. In International conference on computational learning theory, pages 416–426. Springer, 2001.

[35] B. Schölkopf and A. J. Smola. Learning with kernels: support vector machines, regularization, optimization, and beyond. MIT press, 2002.

[36] J. C. Spall. Multivariate stochastic approximation using a simultaneous perturbation gradient approximation. IEEE Transactions on Automatic Control, 37(3):332–341, 1992.

[37] I. Steinwart and A. Christmann. Support Vector Machines. Springer Publishing Company, Incorporated, 1st edition, 2008.

[38] J. Uesato, B. O'Donoghue, P. Kohli, and A. van den Oord. Adversarial risk and the dangers of evaluating against weak attacks. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 5025–5034. PMLR, 2018.

[39] K. Wang, J. Zhang, Y. Chen, and P. Zhong. Least absolute deviation support vector regression. Mathematical Problems in Engineering, 2014(1):169575, 2014.

[40] T. T. Wang. Adversarial examples in simpler settings. M.eng. thesis, Massachusetts Institute of Technology, 2021.

[41] Y. Wang, B. Wang, C. Peng, X. Li, and H. Yin. Huber regression analysis with a semi-supervised method. Mathematics, 10(20):3734, 2022.

[42] Y. Wang, D. Zou, J. Yi, J. Bailey, X. Ma, and Q. Gu. Improving adversarial robustness requires revisiting misclassified examples. In International conference on learning representations, 2019.

[43] H. Wen, A. Betken, and W. Koolen. On the robustness of kernel ridge regression using the cauchy loss function. arXiv preprint arXiv:2503.20120, 2025.

[44] H. Wu, Z. Hu, and B. Gu. Fast and scalable adversarial training of kernel svm via doubly stochastic gradients. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pages 10329– 10337, 2021.

[45] C. Xiao, J.-Y. Zhu, B. Li, W. He, M. Liu, and D. Song. Spatially transformed adversarial examples. arXiv preprint arXiv:1801.02612, 2018.

[46] H. Xiao, K. Rasul, and R. Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017.

[47] H. Xu, C. Caramanis, and S. Mannor. Robustness and regularization of support vector machines. Journal of machine learning research, 10(7), 2009.

[48] C. Yadav and L. Bottou. Cold case: The lost mnist digits. In Advances in Neural Information Processing Systems 32. Curran Associates, Inc., 2019.

[49] F. X. X. Yu, A. T. Suresh, K. M. Choromanski, D. N. Holtmann-Rice, and S. Kumar. Orthogonal random features. Advances in neural information processing systems, 29, 2016.

[50] S. Zagoruyko and N. Komodakis. Wide residual networks. In Proceedings of the British Machine Vision Conference (BMVC), pages 87.1–87.12. BMVA Press, 2016.

[51] H. Zhang, Y. Yu, J. Jiao, E. Xing, L. El Ghaoui, and M. Jordan. Theoretically principled trade-off between robustness and accuracy. In International conference on machine learning, pages 7472–7482. PMLR, 2019.

[52] D.-X. Zhou. Derivative reproducing properties for kernel methods in learning theory. Journal of computational and Applied Mathematics, 220(1-2):456–463, 2008.

## Appendix

Table 1A: Abbreviations (sorted in alphabetical order)
<table><tr><td>advKLR</td><td>adversarially trained KLR</td></tr><tr><td>CAFA</td><td>Cost-aware Feasible Attack [5]</td></tr><tr><td>CNN</td><td>convolutional neural network</td></tr><tr><td>FCN</td><td>fully connected neural network (a.k.a. MLP)</td></tr><tr><td>KM</td><td>kernel machine</td></tr><tr><td>KRR</td><td>kernel ridge regression</td></tr><tr><td>KSVM</td><td>kernel support vector machines</td></tr><tr><td>KLR MLP</td><td>kernel logistic regression</td></tr><tr><td>NN</td><td>multi-layer perceptron (a.k.a. FCN)</td></tr><tr><td>PGD</td><td>neural network</td></tr><tr><td>RFM</td><td>projected gradient descent</td></tr><tr><td>RFC</td><td>recursive feature machine</td></tr><tr><td>SPSA</td><td>random feature classifier</td></tr><tr><td>WRN</td><td>Simultaneous Perturbation Stochastic Approximation [36] Wide Residual Network</td></tr></table>

## A Further Technical Details and Setup

## A.1 Gradient-free (black-box) attacks using SPSA

Let $\mathsf { L } ( x , y )$ denote the loss at input $x \in \mathbb { R } ^ { d }$ with label y. At a given iterate, SPSA samples a random perturbation vector $\Delta \in \{ - 1 , + 1 \} ^ { d }$ . Using a two-sided finite-difference approximation, the SPSA gradient estimator for the i-th coordinate is

$$
\hat { g } _ { i } ( x ) = \frac { \mathsf { L } ( x + \alpha \Delta , y ) - \mathsf { L } ( x - \alpha \Delta , y ) } { 2 \alpha \Delta _ { i } } , \qquad i = 1 , \hdots , d ,
$$

where $\alpha > 0$ is a small perturbation magnitude. Since $\Delta _ { i } \in \{ - 1 , + 1 \}$ , this can equivalently be written as

$$
\hat { g } _ { i } ( x ) = \frac { \mathsf { L } ( x + \alpha \Delta , y ) - \mathsf { L } ( x - \alpha \Delta , y ) } { 2 \alpha } \Delta _ { i } .
$$

In vector form, the estimator is

$$
\hat { g } ( x ) = \frac { \mathsf { L } ( x + \alpha \Delta , y ) - \mathsf { L } ( x - \alpha \Delta , y ) } { 2 \alpha } \Delta .
$$

In practice, we use a vectorised, variance-reduced estimate of the gradient over m independent perturbation vectors $\Delta ^ { ( \mathrm { i } ) } , \ldots , \Delta ^ { ( m ) }$ •

$$
\hat { g } ( x ) = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \frac { \mathsf { L } ( x + \alpha \Delta ^ { ( j ) } , y ) - \mathsf { L } ( x - \alpha \Delta ^ { ( j ) } , y ) } { 2 \alpha } \Delta ^ { ( j ) } .
$$

Table 2A: Overview of figures showing results regarding $\ell _ { 2 }$ and $\ell _ { \infty }$ attacks on Laplacian and Gaussian kernel machines compared with neural networks and different variants of kernel machines for FashionMNIST, KMNIST, and QMNIST. Each link is clickable and takes you to the relevant figure.
<table><tr><td rowspan="2"></td><td colspan="7">Laplacian Kernel</td><td colspan="7">Gaussian Kernel</td></tr><tr><td>KLR</td><td>KRR</td><td>RFM</td><td>KSVM RFC</td><td></td><td>FCNN</td><td>LeNet</td><td>KLR</td><td>KRR</td><td>RFM</td><td>KSVM</td><td>RFC</td><td>FCNN</td><td>LeNet</td></tr><tr><td>KLR</td><td></td><td>17A</td><td>18A</td><td>19A</td><td>25A, 26A 9A</td><td></td><td>10A</td><td></td><td>39A</td><td>40A</td><td>41A</td><td>48A,49A</td><td>35A</td><td>56A</td></tr><tr><td>KRR</td><td></td><td></td><td>20A</td><td>21A</td><td>27A, 28A 11A</td><td></td><td>12A</td><td></td><td></td><td>42A</td><td>43A</td><td>50A, 51A 36A</td><td></td><td>57A</td></tr><tr><td>RFM</td><td></td><td></td><td></td><td>22A</td><td>29A, 30A 13A</td><td></td><td>14A</td><td></td><td></td><td></td><td>44A</td><td>52A, 53A 37A</td><td></td><td>58A</td></tr><tr><td>KSVM</td><td></td><td></td><td></td><td></td><td>31A, 32A 15A</td><td></td><td>16A</td><td></td><td></td><td></td><td></td><td>54A, 55A 38A</td><td></td><td>59A</td></tr><tr><td>RFC</td><td></td><td></td><td></td><td></td><td></td><td></td><td>23A, 24A 33A, 34A</td><td></td><td></td><td></td><td></td><td></td><td>46A, 47A 60A, 61A</td><td></td></tr></table>

## A.2 Attacks on Tabular datasets using CaFA

The objective in the CaFA attacks is as follows:

$$
\begin{array} { r } { \mathrm { F i n d ~ } \delta \in \mathbb { R } ^ { d } } \\ { \mathrm { s . t . ~ } f ( x + \delta ) \neq y } \\ { x + \delta \vdash T } \\ { \mathrm { C o s t } ( \delta ) \leq \varepsilon } \end{array}
$$

Note here the additional constraint denoted by $x + \delta \models T$ , which indicates that the generated adversarial example must satisfy feature space constraints. Unlike in the case of images, where simple boundary clipping is sufficient, tabular data is more complex as it has a combination of continuous, integer, and categorical features. In addition, the range of each feature can be different. CaFA proceeds in the following 3 steps:

1. Mining constraints: There are two main types of constraints to be considered. The first are structural constraints, which are available via feature engineering. The second type are relation constraints, which in CaFA are set to be denial constraints. These prevent pairs of samples from having a set of predicates that hold simultaneously. These are difficult to specify and are mined from the data.

2. Perturbing samples: The samples are first perturbed using a method known as TabPGD, which works by first creating temporary adversarial examples. These are then modified feature by feature depending on the type of the feature.

3. Constraint projection: The generated adversarial examples are checked for compliance with the set of mined constraints T using a SAT solver. If the constraints are not satisfied, the samples are then projected by relaxing constraints on certain features. The final obtained adversarial example should now satisfy both structural and relational constraints.

In our experiments we sweep over ε in {0.01, 0.03, 0.075}.

## A.3 Kernels preliminaries

Matérn Kernels are a class of kernels parameterized by a smoothness parameter $\nu > 0$ and length scale $\sigma > 0$ . The general Matérn kernel is defined as

$$
K _ { \nu } ( x , z ) = \frac { 2 ^ { 1 - \nu } } { \Gamma ( \nu ) } \left( \frac { \sqrt { 2 \nu } \left\| x - z \right\| } { \sigma } \right) ^ { \nu } J _ { \nu } \left( \frac { \sqrt { 2 \nu } \left\| x - z \right\| } { \sigma } \right) ,\tag{13}
$$

where $\Gamma ( \cdot )$ is the Gamma function and $J _ { \nu }$ denotes the modified Bessel function of the second kind. Here $\sigma$ is the length scale, while ν defines the smoothness of functions associated with the kernel, with larger ν corresponding to smoother functions. The Laplacian kernel $( \nu = 1 / 2 )$ , denoted $K ^ { \mathsf { l a p } }$ , and the Gaussian kernel $( \nu \to \infty )$ , denoted $K ^ { \tt g a u s s }$ , are also members of the Matérn family. In addition, Matérn Kernels with $\nu = ( 2 k + 1 ) / 2$ for $k \in \mathbb N$ have known closed form expressions. For example, $\nu = 3 / 2$ and $\nu = 5 / 2$ are given below:

$$
\begin{array} { r l } & { K _ { M } ^ { \mathsf { l a p } } ( x , z ) = \exp \left( - \frac { \| x - z \| _ { M } ^ { \gamma } } { \sigma } \right) , K _ { M } ^ { \mathsf { g } } ( x , z ) = \exp \left( - \frac { \| x - z \| _ { M } ^ { 2 } } { 2 \sigma ^ { 2 } } \right) } \\ & { K _ { 3 / 2 } ( x , z ) = \left( 1 + \frac { \sqrt { 3 } \| x - z \| } { \sigma } \right) \exp \left( - \frac { \sqrt { 3 } \| x - z \| } { \sigma } \right) , } \\ & { K _ { 5 / 2 } ( x , z ) = \left( 1 + \frac { \sqrt { 5 } \| x - z \| } { \sigma } + \frac { 5 \| x - z \| ^ { 2 } } { 3 \sigma ^ { 2 } } \right) \exp \left( - \frac { \sqrt { 5 } \| x - z \| } { \sigma } \right) . } \end{array}
$$

Kernel Ridge Regression (KRR), Kernel SVM (KSVM) and Kernel Logistic Regression (KLR): While the Representer Theorem allows for arbitrary loss functions, in practice most KMs are trained for the square loss $\begin{array} { r } { J \left( \{ x _ { i } , f ( x _ { i } ) , y _ { i } \} _ { i = 1 } ^ { n } \right) = \sum _ { i = 1 } ^ { n } \| y _ { i } - f ( x _ { i } ) \| ^ { 2 } } \end{array}$ , due to the simplicity in implementation. KRR is the solution to the linear equation

$$
Y = ( K ( X , X ) + \lambda I _ { n } ) \alpha\tag{14}
$$

where $Y \in \mathbb { R } ^ { n \times C }$ is the matrix of targets in the training data. We also trained KSVM to minimize hinge loss, or soft-margin $\begin{array} { r } { \mathrm { S V M } , J \left( \left\{ x _ { i } , f ( x _ { i } ) , y _ { i } \right\} _ { i = 1 } ^ { n } \right) = \sum _ { i = 1 } ^ { n } } \end{array}$ max $( 0 , 1 - y _ { i } f ( x _ { i } ) )$ , using one-vs-one for multiclass classification problems.

## A.4 Adversarial training of NNs

To train neural networks adversarially, we use the standard min-max optimisation approach from [28]:

$$
f _ { \mathrm { r o b u s t } } = \underset { f \in \mathcal { F } } { \arg \operatorname* { m i n } } \mathbb { E } _ { ( x , y ) \sim \mathbb { P } _ { \mathrm { t r a i n } } } \left[ \operatorname* { s u p } _ { \delta : \| \delta \| _ { q } \leq \varepsilon } \mathsf { L } \left( x + \delta , y \right) \right] .
$$

In practice, an alternating minimisation approach is used to obtain $f _ { \mathrm { r o b u s t } }$ . First, δ is set to 0, and the standard empirical risk is minimised. The function f thus obtained is fixed, and the inner supremum computed with respect to it, typically using Projected Gradient Descent. The empirical risk with respect to the resulting δ per sample is then computed, and a new $f$ is obtained. This procedure continues until the training loss plateaus or a sufficiently high robust accuracy is obtained.

## A.5 Adversarial training of Kernel Machines

Minibatch stochastic updates for binary classification: The derivatives of the loss function are

$$
\nabla _ { f } \mathcal { L } = E ^ { * } \big ( s ( E f ) - y \big ) + \lambda f + D ^ { * } \delta , \qquad \nabla _ { \delta } \mathcal { L } = D f .
$$

We use the following procedure for training. At iteration t, sample a uniform subset $B _ { t } \subseteq \{ 1 , \dots , n \}$ of size b. Let $P _ { B _ { t } }$ retain the sampled rows and set the others to zero. Since $\mathbb { E } [ ( n / b ) P _ { B _ { t } } ] = I _ { n }$ , unbiased stochastic primal and dual directions give

$$
\begin{array} { r l } & { f _ { t + 1 } = f _ { t } - \eta _ { t } \displaystyle \frac { n } { b } \left[ E ^ { * } P _ { B _ { t } } \big ( s ( E f _ { t } ) - y \big ) + D ^ { * } P _ { B _ { t } } \delta _ { t } \right] , } \\ & { } \\ & { \delta _ { t + 1 } = \Pi _ { \mathcal { C } } \left( \delta _ { t } + \eta _ { t } \displaystyle \frac { n } { b } P _ { B _ { t } } D f _ { t } \right) . } \end{array}
$$

Thus only the sampled rows of δ change. Clipping acts component-wise. Both updates use the old iterates.

Approximate RKHS projection: Let $\mathcal { F }$ be the subspace of functions $f$ of the form $\textstyle \sum _ { i = 1 } ^ { n } \alpha _ { i } K ( x , x _ { i } ) +$ $\beta _ { i } ^ { \top } G _ { 2 } ( x , x _ { i } )$ where $\alpha _ { i } \in \mathbb { R } ^ { C } , \beta _ { i } \in \mathbb { R } ^ { d \times C }$ , and let $\mathcal { F } _ { 1 }$ be the cone of functions $f$ of the form (11).

Proposition 3. Suppose $g \in { \mathcal { F } }$ with coefficients $( \alpha , \beta ) \in \mathbb { R } ^ { n \times c } \times \mathbb { R } ^ { n \times c d }$ , then solving

$$
\operatorname* { m i n } _ { f \in \mathscr { F } _ { 1 } } \| f - g \| _ { \mathbb { H } }
$$

is equivalent to solving

$$
\operatorname* { m i n } _ { \mu , \theta , \gamma } \sum _ { i , j = 1 } ^ { n } \frac { 1 } { 2 } \left. \left( { \mu _ { i } - \alpha _ { i } \atop \theta _ { i } \gamma _ { i } ^ { \top } - \beta _ { i } } \right) , M _ { i j } \left( { \mu _ { j } \atop \theta _ { j } \gamma _ { j } ^ { \top } - \beta _ { j } } \right) \right. ,\tag{15}
$$

where $( \mu , \theta , \gamma ) \in \mathbb { R } ^ { n \times c } \times \mathbb { R } ^ { n \times d } \times \mathbb { R } ^ { n \times c } .$

We provide Algorithm 2 to solve problem (15) approximately

## B Detailed Experimental Setup

## B.1 Kernel Machines training details

Base KM: For KRR and RFM, we searched for $\lambda \in \{ 1 , 0 . 1 , 0 . 0 1 , 0 . 0 0 1 , 0 . 0 0 0 1 \}$ and $\sigma \in \{ 0 . 0 5 , 0 . 1 2 5$ 0.25, 0.5, 1, 2, 4, 6}. For KSVM, the search over σ is the same as for KRR, while λ was searched over the set {10000, 1000, 100, 10, 1, 0.1, 0.01}. For KLR, we searched for length scale over the set {0.5, 1.0, 2.0, 4.0}, while the learning rate is searched over {0.1, 0.5, 1.0, 2.0, 5.0}.

RFCs: For binary classification problems, we use 5000 and 20000 random features. The 1v1 results for RFCs are accuracies averaged over the seeds in {100, 150, 200, 250, 300, 350, 400, 450, 500, 550}. For multiclass classification, we searched over the number of features in {10k, 20k, 30k, 40k, 50k, 60k}, using the same seeds as in the 1v1 problem, to study the dependence between the number of features and robust accuracy in RFCs. We searched the length scale over σ ∈ {0.1, 0.3, 0.5, 0.75, 1, 2, 3, 5, 7, 10} and the ridge parameter λ such that $- \log _ { 1 0 } \lambda \in \{ 0 , 1 , 2 , 3 \}$

Algorithm 1 MARK: Multiclass Adversarially Robust Kernel machines   
Require: Data $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ , kernel functions $\left( K , G _ { 1 } , G _ { 2 } , G _ { 1 2 } \right)$ , budget ε, Epochs E, learning rate η,   
projection interval T   
1: Initialize $\mu , \gamma = 0 _ { n \times c } , \theta = 0 _ { { \underline { { n } } } \times d }$ (if not provided)   
2: Initialize $\alpha _ { j } = \mu _ { j } , \beta _ { j } = \theta _ { j } \gamma _ { j } ^ { \top }$   
3: r₀:=einsum $( \cdot \circ \mathrm { i n } , \mathrm { n c } \to \mathsf { b c } ^ { \flat } )$   
4: $r _ { 2 } : = \mathtt { e i n s u m ( \cdots b n d , n d c  b c " ) }$   
5: $r _ { 1 } : = \mathrm { e i n s u m ( ^ { \cdot * } b n d , n c  b d c ^ { \cdot \cdot } ) }$   
6: r12:= einsum("bndD, nDc → bdc")   
7: for $e = 1 , \ldots , E$ do   
8: for minibatch $b \subset [ n ] ~ \mathrm { w / o }$ replacement do   
9: Fetch minibatch inputs $B  X [ b ]$   
10: $\mathbf { \boldsymbol { F } } \gets r _ { 0 } ( K ( B , X ) , \alpha ) + r _ { 2 } ( G _ { 2 } ( \bar { B , X } ) , \beta )$   
11: R ← softmax $( F ) -$ one\_hot $\left( Y [ B ] \right)$   
12: $J \gets r _ { 1 } ( G _ { 1 } ( B , X ) , \alpha ) + r _ { 1 2 } ( \dot { H } ( \dot { B } , X ) , \beta )$   
13: $k _ { i } \gets \arg \operatorname* { m a x } _ { 1 \leq k \leq C } \sum _ { a = 1 } ^ { d } | { \pmb J } _ { i k a } | \quad \forall i \in \cal B$   
14: $Q \gets 2 \varepsilon \cdot \left( \mathrm { s i g n } ( J _ { i k _ { i } , : } ^ { \top } ) e _ { k _ { i } } ^ { \top } \right) $ rank-1 dual maximizer   
15: $\alpha [ b ]  \alpha [ b ] - \eta \frac { n } { | b | } R$   
16: $\beta [ b ]  \beta [ b ] - \eta \frac { \dot { n } } { | b | } Q$   
17: end for   
18: if $e = 0$ mod T then   
19: $( \mu , \theta , \gamma )  \mathrm { P R O J E C T } ( \alpha , \beta )$ solve (15)   
20: reset $\alpha _ { j } = \mu _ { j }$ , and $\beta _ { j } = \dot { \theta _ { j } } \gamma _ { j } ^ { \top }$ for $j \in [ n ]$   
21: end if   
22: end for   
23: return $f = \sum _ { j = 1 } ^ { n } K ( \cdot , x _ { j } ) \mu _ { j } + \theta _ { j } ^ { \top } G _ { 2 } ( \cdot , x _ { j } ) \gamma _ { j }$

## B.2 Neural Networks training details

The FCNNs were trained using stochastic gradient descent (SGD) with momentum to minimise the cross-entropy loss. We perform an exhaustive search over the network's widths in {64, 128, 256}, depths in {2, 3, 4, 5}, learning rates in {0.01, 0.03, 0.1}, and weight decay rates in {0.01, 0.001, 0.0001}, with momentum fixed at 0.9. We did the above search for both robust training (using Projected Gradient Descent) and benign training. Convolutional neural networks (CNNs) with the modified LeNet5 architecture (using ReLU activation and max-pooling layers) [24], and WideResNet-28-10 [50] were trained with standard training hyperparameters. We additionally evaluated the sensitivity of the FCNN baselines to random initialization across multiple training seeds. We observed only minor variations in clean and adversarial accuracy, with the overall robustness trends remaining consistent across runs. Therefore, we present the results for FCNNs trained using seed 42.

## B.3 Attacks

For the $\ell _ { \infty }$ constrained attacks, we use perturbation budgets $\varepsilon = { \frac { \varepsilon _ { 0 } } { 2 5 5 } }$ with $\varepsilon _ { 0 } \in \{ 0 , . 1 , 1 , 2 , 4 , 6 , 8 , 1 2 .$ 20} for 1v1 comparison, while for multiclass we used these $\ell _ { \infty }$ perturbation budgets, $\varepsilon = \frac { \varepsilon _ { 0 } } { 2 5 5 }$ with ${ \varepsilon _ { 0 } } \in \{ 0 , . 1 , 1 , 2 , 4 , 6 , 8 , 1 2 , 2 0 , 3 2 , 4 0 , 5 5 \}$ for multiclass classification. For $\ell _ { 2 }$ constrained attacks, we use $\varepsilon \in \{ 0 , . 5 , 1 , 2 , 4 , 6 , 8 \}$ for 1v1, and $\varepsilon \in \{ 0 , . 5 , 1 , 2 , 4 , 6 \}$ . The fixed iteration attack uses 100 iterations, with step size $\varepsilon / 1 0$ . For details on other attacks used, see black-box attacks (§D.1) and attacks on tabular data (§D.2).

## B.4 Adversarial training of neural networks

We have trained our models for multiclass with $\ell _ { \infty }$ perturbation for FashionMNIST with $\frac { 8 } { 2 5 5 }$ , for KMNIST it is $\frac { 1 6 } { 2 5 5 }$ , for QMNIST it is $\frac { 2 4 } { 2 5 5 }$ . For $\ell _ { 2 }$ training, we have used perturbation budget $\{ \varepsilon = 1 \}$ . For binary classification we have used $\ell _ { \infty }$ perturbation budget $\begin{array} { r } { \left\{ \varepsilon = \frac { 1 2 } { 2 5 5 } \right\} } \end{array}$ for FashionMNIST. $\begin{array} { r } { \left\{ \varepsilon = \frac { 1 6 } { 2 5 5 } \right\} } \end{array}$ for KMNIST, $\begin{array} { r } { \left\{ \varepsilon = \frac { 2 4 } { 2 5 5 } \right\} } \end{array}$ budget for QMNIST for 200 epochs while doing robust training. Depth is searched over {2, 4} for pairwise classification, width is searched over {128, 256}, learning rate is in {0.03, 0.1}, and weight decay in {0.0001, 0.001}, momentum was fixed as 0.9, dropout was fixed at 0. While for multiclass adversarial training, we searched over depth $\in \{ 3 , 5 \}$ , width over {128, 256}, and searched for learning rate in {0.03, 0.1}, and weight decay in {0.0001, 0.001}, with momentum kept at 0.9, and with 0 dropout. For adversarial LeNets we searched over the learning rate in 0.03, 0.1, and weight decay in {0.0001, 0.001}, with momentum kept at 0.9, and dropout being 0. LeNets were trained for 50 epochs, while WRNs were trained for 20 epochs.

Algorithm 2 Approximate rank-1 projection in RKHS   
Require: Coefficients $( \alpha , \beta ) , K , G , H ,$ refinement steps L   
1: $( \theta _ { i } , \gamma _ { i } )  \mathbf { r a n k 1 - S V D } ( \beta _ { i } ) \quad \forall i \in [ n ]$   
2: for $l = 1 , \ldots , L { \bf d o }$   
3: $E _ { i } \gets \theta _ { i } \boldsymbol { \gamma } _ { i } ^ { \top } - \beta _ { i } \quad \forall i \in [ n ]$   
4: $Z \gets \mathsf { s o l v e } ( K , r _ { 2 } ( G _ { 2 } ( \dot { X } , \dot { X } ) , E ) )$ EigenPro   
5: $R  r _ { 1 2 } ( H , E ) - r _ { 1 } ( G _ { 1 } ( X , X ) , Z )$   
6: $P _ { i }  R _ { i } \gamma _ { i } , \quad Q _ { i }  R _ { i } ^ { \top } \theta _ { i } \quad \forall i \in [ n ]$   
7: $\eta \gets \mathtt { b a c k t r a c k } ( \theta , \gamma ; P , Q )$ Armijo rule   
8: $\theta _ { i }  \theta _ { i } - \eta P _ { i } , \quad \gamma _ { i }  \gamma _ { i } - \eta Q _ { i } \quad \forall i \in [ n ]$   
9: end for   
10: $E _ { i } \gets \theta _ { i } \boldsymbol { \gamma } _ { i } ^ { \top } - \beta _ { i } \quad \forall i \in [ n ]$   
11: $\mu  \alpha - \operatorname { s o l v e } ( K , r _ { 2 } ( G _ { 2 } ( X , X ) , E ) )$ EigenPro   
12: return $( \mu , \theta , \gamma )$

## B.5 Adversarial training of KLR

Adversarial training of kernel machines. For binary adversarial KLR, we used Gaussian kernels with length scales explored over {1, 2, 4} across class pairs and swept the RKHS regularization coefficient over $\lambda \in \{ 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } \}$ , selecting λ by clean validation accuracy. Training used 500 stochastic primal–dual updates, a minibatch size of 128, an initial learning rate of 0.1, a learning-rate decay exponent of 0.75, weighted iterate averaging, and random seed 0. The gradient-penalty coefficient ρ was set to 12/255, 20/255, and 24/255 for Fashion-MNIST, KMNIST, and QMNIST, respectively.

For multiclass adversarial KLR, we compared Gaussian length scales {1,2} and selected 2 based on improved clean accuracy and adversarial robustness. Training used 10 epochs, a minibatch size of 128, an initial learning rate of 0.01, a learning-rate decay exponent of 0.75, RKHS regularization $\lambda = 1 0 ^ { - 4 }$ and random seed 42. The radius parameter € was set to 8/255, 16/255, and 24/255 for Fashion-MNIST, KMNIST, and QMNIST, respectively, with a penalty multiplier of 2. Projection intervals of one and two epochs yielded comparable performance; the reported checkpoints used one epoch for Fashion-MNIST and two epochs for KMNIST and QMNIST. Projection used at most 100 refinement steps and a tolerance of $1 0 ^ { - 4 }$ The EigenPro solver used relative and absolute tolerances of $1 0 ^ { - 3 }$ and $1 0 ^ { - 8 }$ , respectively, at most 100 solver epochs, a subsample size of 1024, rank 100, a minibatch size of 128, and matrix-free storage. Query and center tile sizes were 64 and 1024, respectively.

Table 3A: Computational complexity of training on a single NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation GPU. (†time in seconds(s) to train a single model on Fashion MNIST. Note that KRR and RFM have fewer hyperparameters to tune, hence the total time is smaller.)
<table><tr><td>model</td><td>hyperparameters</td><td>time  $( \mathrm { 1 v 1 } ) ^ { \dagger }$ </td><td>time (multi-class)†</td></tr><tr><td>FCNN</td><td>width, depth, lr, weight decay</td><td>≈ 40s</td><td>≈ 300s</td></tr><tr><td>FCNN-adv</td><td> $\cdots + \varepsilon$ </td><td>≈ 300s</td><td>≈ 900s</td></tr><tr><td>LeNet</td><td>N.A.</td><td>≈40s</td><td>≈ 240s</td></tr><tr><td>adv-LeNet</td><td>N.A.</td><td>≈ 120s</td><td>≈ 720s</td></tr><tr><td>KRR</td><td> $\lambda , \sigma$ </td><td>≈ 5s</td><td>≈ 30s</td></tr><tr><td>RFM</td><td> $\cdots + \# \mathrm { i t e r a t i o n s }$ </td><td>≈ 25s</td><td>≈ 100s</td></tr><tr><td>RFC</td><td> $\cdots + \# \mathrm { f e a t u r e s }$ </td><td>≈ 0.5s</td><td>≈1s</td></tr><tr><td>Kernel SVM</td><td></td><td>≈ 2s</td><td>≈ 45s</td></tr><tr><td>KLR</td><td>learning rate, σ</td><td>≈ 5s</td><td>≈ 30s</td></tr><tr><td>adv-KLR</td><td>learning rate, σ, λ</td><td>≈ 900s</td><td>≈ 86,000s</td></tr></table>

## B.6 ν vs. robustness of Matérn Kernels

In our experiments, we fine-tuned over ridge $\lambda \in \{ 1 0 ^ { - 6 } , 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 1 0 ^ { - 1 } , 1 0 ^ { 0 } \}$ and length scale $\sigma \in \{ 0 . 0 5 , 0 . 1 2 5 , 0 . 2 5 , 0 . 5 , 1 , 2 , 4 , 6 , 8 , 1 0 \}$ , for each value of $\nu \in \{ 0 . 5 , 1 . 5 , 2 . 5 , \infty \}$ . We chose models corresponding to each ν such that their λ and σ are the same with benign accuracy being close to the best benign accuracy found in the grid search, so as to isolate the effect of ν on robustness. Square loss is used for the training.

## B.7 Computational efficiency details

Model training runtimes: Kernel Machines take around 5 seconds for each dataset to train for one 1v1 problem, while these take around 30 seconds for multiclass classification problem. RFMs take around 25 seconds to train for one 1v1 classification problem, while for multiclass classification they take around 2 minutes of time. Training for RFC is instant no matter the type of problem. Kernel SVM takes around 45 seconds to train using scikit-learn. Benign training of FCNNs for one 1v1 classification problems without hyperparameter training took approximately 40 seconds for FashionMNIST, 25 seconds for KMNIST and 50 seconds for QMNIST. While for the multiclass classification over FashionMNIST. KMNIST and QMNIST, FCNN took around 5 minutes to train for 200 epochs. The robust training took around 15 to 30 minutes for all the datasets, no matter the training budget. LeNets benign training for one 1v1 classification problem takes around 40 seconds, while the robust training for 1v1 takes around 2 minutes for 50 epochs. Training of adv-LeNet for multiclass takes around 12 minutes, and for 1v1 it is around 2 minutes. WRN benign training took around 40 minutes to get trained for 20 epochs. Training for robust adversarial KLR for multiclass classification took around 24 hours for 10 epochs, for projection at every step. Robust binary KLR took around 15 minutes for training at 500 iterations.

Attack runtimes: For multiclass FCNNs and CNNs the attack takes around 15 - 30 minutes, for RFC the attack takes around 5 - 10 minutes, for Kernel machines and RFMs the attack takes around 15 - 20 minutes. For one 1v1 attack for any model the time is less than 2 minutes. These timings are irrespective of the norm being used, and these are the time to attack a particular model on all the perturbation budgets.

## C Detailed Experimental Results

Table 2A provides references to all comparisons for the Laplacian and Gaussian KMs.

## C.1 Adversarial Robustness under $\ell _ { 2 }$ and $\ell _ { \infty }$ Perturbations

Here we provide result for the multiclass problem on FMNIST, KMNIST, QMNIST, and 1v1 for the FMNIST dataset. In the $\ell _ { 2 }$ attack, the error bars for different seeded attacks have been provided, albeit they are too small to be relevant. Similar small error bars for attacks were observed for $\ell _ { \infty }$ attacks, but due to them being quite small only values corresponding to seed 42 have been plotted.

Figure 1A: Comparison of the robust accuracy against $\ell _ { 2 }$ perturbations (↑ is better) for 10 models across 3 datasets. Results were obtained using autoattack, using apgd-ce.  
![](images/1594bbf1052058e70f1c86775676058c4b0c3ec869ef3096c23d5bfbe308025c.jpg)  
C.2 Linf summary for KMNIST/QMNIST vs FCNN and LeNet $\ell _ { \infty }$ attack multiclass results are in Figs. 4A and 5A.

Figure 2A: Gap in robust accuracy (ra) for $\ell _ { 2 }$ perturbation on 45 1v1 binary classification tasks from FashionMNIST comparing 2 neural networks (columns) and 3 kernel machines (rows) with the Laplacian kernel. Perturbation size increases as we go down each heatmap, and values correspond to ra(row model) - ra(col model). The teal colour indicates that the kernel machine is more robust, whereas the orange color indicates that the neural network is more robust. Each of the 6 heatmaps compares 45 pairs of models, each finetuned for best accuracy on clean data. For each heatmap, class pairs for binary classification are sorted according to values in the bottom row. PGD attack is used to attack the models. ∆ robust accuracy (row - col) for $\ell _ { 2 }$ adversarial perturbations on FashionMNIST  
![](images/bef7c1252882df7436b0a9885696ad9e2a2d0beb1b6f5b69b8600aa47a25922f.jpg)

LeNet  
Figure 3A: Gap in robust accuracy (ra) for $\ell _ { 2 }$ perturbation on 45 1v1 binary classification tasks from FashionMNIST comparing 2 neural networks (columns) and 2 RFCs with different number of features (rows) with the Laplacian kernel. Perturbation size increases as we $_ \mathrm { g o }$ down each heatmap, and values correspond to ra(row model) - ra(col model). The teal colour indicates that the RFC is more robust, whereas the orange color indicates that the neural network is more robust. Each of the 4 heatmaps compares 45 pairs of models. For each heatmap, class pairs for binary classification are sorted according to values in the bottom row. (Top) RFC with 5000 random features (Bottom) RFC with 20000 random features. PGD attack is used to attack the models.  
∆ robust accuracy (row - column) for $\ell _ { 2 }$ adversarial perturbations on FashionMNIST  
![](images/5be17be7f2acbf81620c66577c20910d5ebc9d0087621bf2773b0d720199f8b7.jpg)

![](images/9b296c59475a712bc34b27acc247a6ae70b01814dbc23cb896cf40001fa2e1cd.jpg)

![](images/21340f764b2e2d2fec70f4d5d1608a89c2a96325d00d6eda44d9f03625a15df0.jpg)

![](images/6fe645c5cdfa2e8493262d19c7312a8c80c9e7bd31c23b1fa82f7b5ab62a1db6.jpg)

![](images/415e8c1bd4ced895d106ceded1ed6373ad0a1d4553bb9d808c925a998f30eece.jpg)  
Figure 4A: Summary of results for the multiclass KMNIST dataset. We compare robust accuracy of 13 models (3 adversarially trained, 10 benign) on KMNIST (10 classes) attacked via PGD using $\ell _ { \infty }$ perturbations. Columns with yellow borders correspond to adversarially trained models with training adversary's attack level highlighted by the red row. Black boxes mark the best robust accuracy across each row. Note the best models for high ε are kernel machines with logistic loss.

![](images/6eac4d0939372fbcfa6e06d1c02a39b085a98ccc96c0ace11ada10a53cb1c431.jpg)  
Figure 5A: Summary of results for the multiclass QMNIST dataset. We compare robust accuracy of 13 models (3 adversarially trained, 10 benign) on QMNIST (10 classes), attacked via PGD using $\ell _ { \infty }$ perturbations.. Columns with yellow borders correspond to adversarially trained models with training adversary's attack level highlighted by the red row. Black boxes mark the best robust accuracy across each row. Note the best models for high ε are kernel machines with logistic loss.

![](images/471958655f3d26c405ef70abb3b559be42f400ba302a1eb354c89b314e56b673.jpg)  
Figure 6A: Test for robustness as a pointwise property. We plot the empirical CDF of the ratio (KLR/other models) of the smallest $\ell _ { \infty }$ perturbations $\widehat { \varepsilon }$ needed to successfully attack 423 random samples from the test set of KMNIST. Note the x-axis is in log-scale. See §5.2 for more details. KRR is more robust than both FCNN and LeNet. For the median datum, KLR is nearly 2x more robust than Adv-FCNN and Adv-LeNet. The adversarially trained FCNN, LeNet and KLR were trained with an $\ell _ { \infty }$ (matched) adversary with a budget of $\begin{array} { r } { \varepsilon = \frac { \bar { 1 6 } } { 2 5 5 } } \end{array}$ . Adversarially trained KM remains most robust.

![](images/f2b89a88ae402af01f17fb540e67c9735aa93512ca1992b47aef83c2662e4698.jpg)  
Figure 7A: Test for robustness as a pointwise property. We plot the empirical CDF of the ratio (KLR/other models) of the smallest $\ell _ { \infty }$ perturbations ê needed to successfully attack 479 random samples from the test set of QMNIST. Note the x-axis is in log-scale. See §5.2 for more details. For most samples, KRR is harder to attack than the FCNN. For the median datum, KLR is more robust than Adv-FCNN and Adv-LeNet. The adversarially trained FCNN, LeNet and KLR were trained with an $\ell _ { \infty }$ (matched) adversary with a budget of $\textstyle \varepsilon = { \frac { 2 4 } { 2 5 5 } }$ . Adversarially trained KM remains most robust.

## C.3 Comparison of KMs with FCNN and LeNet

Figure 8A: Gap in robust accuracy (ra) for $\ell _ { \infty }$ perturbation on 45 1v1 binary classification tasks from FashionMNIST comparing 2 neural networks (columns) and 3 kernel machines (rows) with the Laplacian kernel. Perturbation size increases as we go down each heatmap, and values correspond to ra(row model) - ra(col model). The teal colour indicates that the kernel machine is more robust, whereas the orange color indicates that the neural network is more robust. Each of the 6 heatmaps compares 45 pairs of models. For each heatmap, class pairs for binary classification are sorted according to values in the bottom row. (Top) Kernel machines are more robust than fully connected neural networks and convolutional networks. (Middle) Recursive Feature Machines retain much of the robustness of the base kernel machine $( M = I _ { d } )$ . (Bottom) Laplacian SVM are more robust to both FCNN and LeNet. PGD attack is used to attack the models.

∆ robust accuracy (row - col) for $\ell _ { \infty }$ adversarial perturbations on FashionMNIST

![](images/ed9d912564f53273b429da7667f56aec716e603a1ca587340f347e375841468b.jpg)

## C.4 Laplacian Results for FMNIST, KMNIST, QMNIST

Here we provide results comparing laplacian kernels in any type of model over the datasets FMNIST, KMNIST, QMNIST

Figure 9A: Laplacian KLR versus FCNN: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Laplacian KLR - FCNN)  
![](images/f2b3e084d362ebdffeb23f40b32df96b33521b938c05f4c20daf8af79fa36a98.jpg)

![](images/6e0da2ebcda606f53e7256f813330ea67d22b0329a8ae39569cf74fccc1c2c2b.jpg)  
Figure 10A: Laplacian KLR versus LeNet: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Laplacian KLR - LeNet)

![](images/85b5dea6654c2e9a5553a98e89b45d8f7cdb25bc41ad6ec7e0425ab40a175b1c.jpg)

Figure 11A: Laplacian KRR versus FCNN: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Laplacian KRR - FCNN)  
![](images/ed5f698e98ce5a09105ce38ff39612bf4650637666e68ec1f1ee17d586c1da2f.jpg)

![](images/95dc5bd96aa2a09c17e34d9490711b60fe7c54b5c88dc11c9f952f82238191ca.jpg)

![](images/c45f82e788e0de30d1e1ad884ce2088f9805ff4bff4d3cf7304c93c8aab8b451.jpg)  
Figure 12A: Laplacian KRR versus LeNet: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Laplacian KRR - LeNet)

![](images/4e284b086a9f02d8d9a66a18f9588bf86ec06a8b0af9d343114ecff3f4ac7c9a.jpg)  
Figure 13A: Laplacian RFM versus FCNN: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Laplacian RFM - FCNN)

Figure 14A: Laplacian RFM versus LeNet: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Laplacian RFM - LeNet)  
![](images/790031e31a79efd31db7c5f01a67c4627fdf249ef4f0bfa853bace168bb073d4.jpg)

Figure 15A: Laplacian SVM versus FCNN: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Laplacian SVM - FCNN)  
![](images/686848c585a383044dcecbdb203710f89cedc7a958ee6aaef12fc11b8fed5700.jpg)

![](images/ad1bc2283dfba9a851d2d34d269e2c08f09d646d2810319fd0097a9020ba7ce6.jpg)

Figure 16A: Laplacian SVM versus LeNet: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Laplacian SVM - LeNet)  
![](images/47f1a8d0893f9a3ab6559a28555777cecb7d07ca30c67dac16f53a134e0eb882.jpg)

![](images/00e71529ccc0a35088d43494bff821462a5a9fcc8b9da2d14a9158c6963d27a2.jpg)

Figure 17A: Laplacian KLR vs Laplacian KRR: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Laplacian KLR - Laplacian KRR)  
![](images/c980a1ff36162eca8403a4bc8d88a0370c95834ffa5bcd4a09bd97fa258c9618.jpg)

![](images/747736c021c500660dea0fe9fe25d3663d4e633f0e35c52f9e33ddf269ac6df3.jpg)  
Figure 18A: Laplacian KLR vs Laplacian RFM: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Laplacian KLR - Laplacian RFM)

![](images/d651c85af661a6b5cdf03eff8c2e8c0a91fbfbd31fa29cad3a51817b26aafdb9.jpg)  
Figure 19A: Laplacian KLR vs Laplacian $\mathbf { S V M } \colon \ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Laplacian KLR - Laplacian SVM)

Figure 20A: Laplacian KRR vs Laplacian RFM: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Laplacian KRR - Laplacian RFM)  
![](images/57d44538945254ad8fb3b880d0802519f7ca60c89b28b189ab1b45ef575af8d3.jpg)

![](images/9fd6284ec78ec946000092eebe099d49d96399a757d6406a1f4b0f8f3ca4a41f.jpg)

![](images/3f84adadc84ddf4a2d7ee28c6cd483f6bf2065d586121246bfe3179de27306e8.jpg)

![](images/4e7642c739b7f38b471b856e7b86505e23872b32507cb275feb7a895c22accc0.jpg)

Figure 21A: Laplacian KRR vs Laplacian SVM: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Laplacian KRR - Laplacian SVM)  
![](images/738204c043f802424a4836c098b7d1bc4eed121ef94fde88d44b3ded7ce7c5bf.jpg)

Figure 22A: Laplacian RFM vs Laplacian SVM: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Laplacian RFM - Laplacian SVM)  
![](images/06ba4568c708cd7ac6cccb6f570ae4d8dd4f72428e7a9bd5eec0d127eb8782d7.jpg)

![](images/3a5d9746797fb78933b732b250b53a07ea0db5fee74898f922590ed5b8eed82c.jpg)

![](images/8723cd645e64879991e71cd213168e19221339f40920ddc19bc2d5d3b1100c7f.jpg)  
Figure 23A: Laplacian RFC (5k features) versus FCNN: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. Laplace RFC is trained using square loss. PGD attack is used to attack the models.  
∆ robust accuracy (Laplacian RFC (5k features) - FCNN)

![](images/783e3c20f6ad45d5654d6a202faf3f60c7369a6d8014d7c867b44724f7d44dbf.jpg)

Figure 24A: Laplacian RFC (20k features) versus FCNN: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these. (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. Laplace RFC is trained using square loss. PGD attack is used to attack the models.  
![](images/38b21c1dd96ebbd527c80979e223131480bd6ab9016107434eda3df5f9e27a3a.jpg)

Figure 25A: Laplacian KLR vs Laplacian RFC (5k features): $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Laplacian KLR - Laplacian RFC (5k features))  
![](images/bc773110d40b25f4d38e523f631a57d0a2e66ddc2db723229c24bec93e229a08.jpg)

Figure 26A: Laplacian KLR vs Laplacian RFC (20k features): $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Laplacian KLR - Laplacian RFC (20k features))  
![](images/6e6ef20f976663393e4f4e7d9c3ed580919f81c17b2a3e9f7473024f82d0eea6.jpg)

![](images/6bc041bcfa65dde556c827c3b6832f540193af1b4ab39a051240ceabede21bb9.jpg)

Figure 27A: Laplacian KRR vs Laplacian RFC (5k features): $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Laplacian KRR - Laplacian RFC (5k features))  
![](images/103cdaa6f136cca3177f2e45d0fe2f8c494e4a525c5dca46435a75d383f70cfb.jpg)

![](images/ee01101178023579dd9fd8378851f09bb97121d7f716f036fe4f968a7872ddf0.jpg)

Figure 28A: Laplacian KRR vs Laplacian RFC (20k features): $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Laplacian KRR - Laplacian RFC (20k features))  
![](images/be7e4c853a38cc3bfd07cb665a93248355754d37d7351dd0f7c9e2c4ea37dbfd.jpg)

![](images/2f47340d6dccf1a0ab93def6b69f8f987f112602ef9aeeaccc4c86636c715d3e.jpg)

![](images/a41ab00e2f7d278c46b8c7766eddb8eb73fddca5292b9d6b24323d46f4d736fe.jpg)

![](images/db6346aa62b310efe7a0c0d202f8f178232fae6b9621eefc6bd7fa5f53faa6c0.jpg)

Figure 29A: Laplacian RFM vs Laplacian RFC (5k features): $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Laplacian RFM - Laplacian RFC (5k features))  
![](images/02627ed986ae7fe2c7c220243ab3218f72f5be4075bd86e4ec10817f15f4b157.jpg)

![](images/d81858d5a1a94aa9b64de7fd98dcdd93be97a67516277231f652b3a23d2c719a.jpg)

![](images/0024285cd087d669e14f6376c90891317afe4c28bcedc27e561c831cee48e448.jpg)

Figure 30A: Laplacian RFM vs Laplacian RFC (20k features): $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Laplacian RFM - Laplacian RFC (20k features))  
![](images/5294ae4e2526093bcff7f6743423cd1fec82b6a2035c525da74d96572b309854.jpg)

![](images/cd7f3bc2fb91a2121134b89ff8aa34e5b9559946af410bf6dbe8413b74186543.jpg)

![](images/579c4f2e43c2b156708d35893440884c6c18822694bdca74a6770d4b239c5592.jpg)

![](images/e943d0f518cec39dd66ecee1282622ce253d7c7ee816009b78da049927fe7582.jpg)

Figure 31A: Laplacian SVM vs Laplacian RFC (5k features): $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Laplacian SVM - Laplacian RFC (5k features))  
![](images/65724fa07c31ae6d924869034306eba019ebb8e1c1fcd632c6226575b2e24c89.jpg)

![](images/34ad17b842884eae1272273015f01751947b769a54f5f0b8adbd23e71df4d37b.jpg)

Figure 32A: Laplacian SVM vs Laplacian RFC (20k features): $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Laplacian SVM - Laplacian RFC (20k features))  
![](images/3ad2b53f45fcacd55c71975e8baa8f358fb6ad087f6d2fc1e850553937b55079.jpg)

![](images/eb1febf6c42c332d8f933b550d54a8ec378884839498a6947213aa04d6dd605e.jpg)

![](images/9494fc3a26950468c977c2ac758e8d06e6d9b47a03c5e1ed60ddd2691879f637.jpg)

![](images/2159312161faf4175555b4d0ae8d2e4a004860fc0189a47dd88bb0ce3f805ffa.jpg)

![](images/9cafda2e00486934b246d9ffdaa2bbc03beda580e5d99fc0c78b5bf6632fd32a.jpg)

Figure 33A: Laplacian RFC (5k features) versus LeNet: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Laplacian RFC (5k features) - LeNet)  
![](images/8de3ba4a41cf4eab28aa2a67df164eeb2c4ae24f902e36269ec6e8b656d0e3c3.jpg)

Figure 34A: Laplacian RFC (20k features) versus LeNet: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Laplacian RFC (20k features) - LeNet)  
![](images/25d29d07635f34311045f6720b19807afbc9100cb4eed879a580ea221db9d1ca.jpg)

## C.5 Gaussian Results for FMNIST, KMNIST, QMNIST

Here we provide results comparing gaussian kernels in any type of model over the datasets FMNIST, KMNIST, QMNIST

Figure 35A: Gaussian KLR versus FCNN: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. Δ robust accuracy (Gaussian KLR - FCNN)  
![](images/9c0314792a6f8028f351052b9ccba69fa7c3cfbc00a92764c2ca975a3a2c2a04.jpg)

![](images/3369b99fd1ee1817bc6e240c2f13b7691104f46a42c549598577df28a0cb4f1a.jpg)  
Figure 36A: Gaussian KRR versus FCNN: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. Δ robust accuracy (Gaussian KRR - FCNN)

![](images/3862a34c7d97bccb646829589c39871b02e530b81dc744529e851d9313070a05.jpg)

Figure 37A: Gaussian RFM versus FCNN: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. Δ robust accuracy (Gaussian RFM - FCNN)  
![](images/63185cba0a18cad901649ec523f3cf4bf47f02e74db108edb1b1761db3f93566.jpg)

![](images/db4bb39fad58f8ad65babb2244b9f0af58e16bed6baa649bde7aa095b59041d1.jpg)

![](images/a33457046d0e85cbde7833fecf32ab73a399b77a5d478f724d14bcfe6ebdd1fe.jpg)

![](images/dc26ade34d985629cebe59e9624c5c9b7a97f3925c8f76f3bdd376caf0791770.jpg)

![](images/2b31754180e52515f79a37c8569f5ae919eaca131cfe2259698351e3029af5c1.jpg)

![](images/136083a8cee3987a7dc30d26941f7fda44093b0ad55f2b78d2cd1f34624d0314.jpg)

![](images/b5435b6f4017b39b613248420ec0a33a8583dedd69796b484e1abfe833f9e86f.jpg)

Figure 38A: Gaussian SVM versus FCNN: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. Δ robust accuracy (Gaussian SVM - FCNN)  
![](images/8adb1e0b0319173f467c78c16230c4dca41dfdb1c8045c75e368ef16438f3176.jpg)

![](images/2ffd495a49f310c6e15b4d3ba9b2cd877256fa16be1e35d391faa382a1c5d582.jpg)

Figure 39A: Gaussian KLR versus Gaussian KRR: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Gaussian KLR - Gaussian KRR)  
![](images/48c2670bda8b836b8f70639c2059f4f945bdac67b4b7509ab797ac1157d94cc0.jpg)

Figure 40A: Gaussian KLR versus Gaussian RFM: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Gaussian KLR - Gaussian RFM)  
![](images/14257c75de4c60c0018df244b5389bb8c176f073cb36a64594d3eb122f901c51.jpg)

Figure 41A: Gaussian KLR versus Gaussian SVM: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Gaussian KLR - Gaussian SVM)  
![](images/bbbe185aa034be468b787c4270d5f08195f03f3a15765bace94dea47b7e2dbd5.jpg)

![](images/46cf49772e76bbf6c3454feec47869d9d4ed70342a7f9c8203219e610d0b92c2.jpg)

![](images/fdadbcdea1a5410eeab0ca73d67ee61cea6368a2408b2af266f79469f3d4ea11.jpg)

![](images/38684d183e076f9763bdf94512f19fa583c1301971ebebc044e87901aaff4a62.jpg)

![](images/b51618226006602b4766c0bf2e024e571950206ede31a300076d9f8b4cc0e600.jpg)

![](images/1d7b8df9733cae09a596d6cb5ae0eb4cad5488804fb3f86d2976410875a4542a.jpg)

![](images/ad542b1700f4fea1dc6be328202198d8fc564e50bea117bea70561df56ece5a1.jpg)

Figure 42A: Gaussian KRR versus Gaussian RFM: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Gaussian KRR - Gaussian RFM)  
![](images/dd18db653ca2b607305ab0084c66e971b86dadef9b30aa327d4fc704d4928652.jpg)

![](images/4b685e56ec894d2dcfa15fb9f9be156d5ecfeb402f4df4774b0d990b02d25648.jpg)

![](images/29d66c5c844e2b15e855544decc34d71fc1257fa00785748157fd05207eb3727.jpg)

![](images/d2f93df2993d55e81945b0e74d1686e1e8e53364630c5c50042accc07be4cca9.jpg)

![](images/efad1a9745ffd8ced867054512049bd1102b35a0685acfa8ef0833c2ae8f44fe.jpg)

![](images/b5466a001902a642332339e72007c24a785ee029f3af3b83fc833ce07d629672.jpg)

Figure 43A: Gaussian KRR versus Gaussian SVM: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Gaussian KRR - Gaussian SVM)  
![](images/977dc0dfe4c9409400cd4d3a26c94f2534bfb31d21d452d611e6edec9016ed5f.jpg)

![](images/24a197ea2d387ff42d264675e27fefa93b021a0fe1928f17ff8cb7b15e85a8d0.jpg)

Figure 44A: Gaussian RFM versus Gaussian SVM: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Gaussian RFM - Gaussian SVM)  
![](images/18f2d8ea72b90daf4f4b22a0f76d1db75be2734108c581c777ef5cdc6c9a2e96.jpg)

![](images/ae8800c12048d57a0f0fec0daec23a9a44c8899c069f6f7968727630fc3e4180.jpg)

![](images/f552cbfbb468fafbebb52032982ff1b3d3e8e633189d84520b83f23ed3d25a0d.jpg)

![](images/b68c795ed677157f8192dfe82fa2b837abb955b77677543ce3aeeb709870daf8.jpg)

![](images/0372ed42419d48c1b2a4b7856ec08a78f2c63f0a8db72d85c075e8c19d974c53.jpg)

![](images/f24c8285fe0c0834bede90d208940b9a001a32b173fbd36e6300cb78d393d1b2.jpg)

![](images/5ce37f68192ce37533a9ec102f7cce94e6c32f788f0bf849a0ad20014aabc315.jpg)

![](images/5798c66c162e9c7ce08ac6d4401a639d442978f3cf8d32cd16421ba7286f59dc.jpg)  
Figure 45A: Gap in robust accuracy (ra) for $\ell _ { \infty }$ perturbation on 45 1v1 binary classification tasks from FashionMNIST comparing 2 neural networks (columns) and 2 RFCs with different number of features (rows) with the Laplacian kernel. Perturbation size increases as we go down each heatmap, and values correspond to ra(row model) - ra(col model). The teal colour indicates that the RFC is more robust, whereas the orange color indicates that the neural network is more robust. Each of the 4 heatmaps compares 45 pairs of models. For each heatmap, class pairs for binary classification are sorted according to values in the bottom row. (Top) RFC with 5000 random features are a bit less robust compared to fully connected neural networks and convolutional neural networks. (Bottom) RFC with 20000 random features are competing in robustness with fully connected neural networks and convolutional neural networks. Although, these results shouldn't be taken at face value as we see in §D.7 that Laplace RFCs exhibit gradient masking. PGD attack is used to attack the models.  
∆ robust accuracy (row - column) for l∞adversarial perturbations on FashionMNiST

Figure 46A: Gaussian RFC (5k features) versus FCNN: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Gaussian RFC (5k features) - FCNN)  
![](images/75f4651aaf26755d0396f7df9d357f7272746f4ab4216f7689518ad2995d4d54.jpg)

Figure 47A: Gaussian RFC (20k features) versus FCNN: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Gaussian RFC (20k features) - FCNN)  
![](images/ee00fc32bc9cf87e37bf078ccece47fd958c592bd7e45bf00b8dd12ae3aeaa3d.jpg)

Figure 48A: Gaussian KLR versus Gaussian RFC (5k features): $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Gaussian KLR - Gaussian RFC (5k features))  
![](images/1bbad4f93a95195393b7ceeda73a91b19901a6b0c558fc0d2a7e34a1497c2b56.jpg)

Figure 49A: Gaussian KLR versus Gaussian RFC (20k features): $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Gaussian KLR - Gaussian RFC (20k features))  
![](images/bb43b1d2b70a2c4e20bcb6b75a6b2f7c44e782882b1b11b56a3f6b43d5da83f6.jpg)

![](images/8e50df4e742ea05929c3c4fc4a70939484f93371c8282c865cb8e091afc0b05c.jpg)  
Figure 50A: Gaussian KRR versus Gaussian RFC (5k features): $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Gaussian KRR - Gaussian RFC (5k features))

![](images/763f403d00b13adc14466f2ca2caea6d2a2840adde45e11f0b1a991754200678.jpg)

Figure 51A: Gaussian KRR versus Gaussian RFC (20k features): $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Gaussian KRR - Gaussian RFC (20k features))  
![](images/06f9fd76c257f421678e6e5c607e3f23af0d56b60c127a8f7110e40cdbf0bdca.jpg)

![](images/a7a7c0b230401746e4e5fcbae4289d038cc414576521d6a2fc0a39b8363cf6ed.jpg)

![](images/025e82879e187a08e235167915776501af19bba623eeb8781601142e69beaa72.jpg)

![](images/da50e0a2c1b66539a82b9b4d61ea134149347b2651423c4015f813d8b012de10.jpg)

![](images/30bb5cae68ea08bf9f0e867415f472626bff6657e6bbc0b4ed624d0243487b77.jpg)

![](images/7e7e348fd667f5295aae9a6b64f9a43aff3a04194ea61f43357c12932c3511f5.jpg)

Figure 52A: Gaussian RFM versus Gaussian RFC (5k features): $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Gaussian RFM - Gaussian RFC (5k features))  
![](images/88b60cf09be35499b74834432de6fdafe1986c24352072d022159bec18f8c89a.jpg)

![](images/3776042aaea3d915990671bcb1a3e1f8d49f5832d77d35ff27b00de97b2f757e.jpg)

![](images/16867ef4161cf1b3b60ce6e2b0726fddc0641063d04d8211f7095822742ffe78.jpg)

![](images/8e3eca632ef59c3b15f3b50b35de2f02074ad51a3ece58bc84d93ea0dfcc5999.jpg)

![](images/e413b7d4165dab0c8aca24675968c2a4edaa92bb5dc92768eac644471bb93daf.jpg)

![](images/947e7ab88cd6fe49d9f44e82e8013ee5bd6d6699cab7db0ac44f86bee63fc9e9.jpg)

Figure 53A: Gaussian RFM versus Gaussian RFC (20k features): $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Gaussian RFM - Gaussian RFC (20k features))  
![](images/18fc8bfd465e73bf75d7f932f1a828c562d24e3f05d5f3ce2136e81e8676ab3b.jpg)

![](images/d083de1b98ddad6b4ec4a17f62b278565aab6505797fb1426023b5494d2d2eb6.jpg)

![](images/55a1821700b399ed9dbb147951618f66322d949b3c040ae8711e12c4f50bd335.jpg)

![](images/4ea2c8c51d922014c12ab7e05dafde3b460cc9af5f6a9d7a1b386b9ea884c9b9.jpg)

![](images/dfb7f5a1b160682529556061f56dbfd9f76eea3f79170aafc8dae46552fc2a07.jpg)

![](images/5f136105ddef53572c9e50a399b8b7cbeacabec2d1f473c9b6ddd207ec177953.jpg)

![](images/15fb7b559136371d0cdad75197a49fed84480cb20088d4ce4b4b7f6c0414a7fc.jpg)

Figure 54A: Gaussian SVM versus Gaussian RFC (5k features): $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Gaussian SVM - Gaussian RFC (5k features))  
![](images/84ba26059ecc8d695c1175e1ab89ba9f80fb37584521d76049143dbc4bd89a3d.jpg)

![](images/cff17141ff573d3355ef1c1314d830b16bf22fd238688078a9a03873a2c8f645.jpg)

![](images/4b70915aa38f724fbd8c7bbf719cc2d2184da925f0dea7bb6fb8be4c8503186a.jpg)

![](images/579064d1b10007507cbf706e0645edcd549c3f043e45ed66c3b151b80ddd97ee.jpg)

Figure 55A: Gaussian SVM versus Gaussian RFC (20k features): $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Gaussian SVM - Gaussian RFC (20k features))  
![](images/3f0bb363df5f2729571edf3e3e7827e6d9f4a78e72d2161a308f87fb4ca2a4a2.jpg)

![](images/42d8782f1e0a5805da64d60b8c78d38a58ac6f0556758347edfa66736c058465.jpg)

Figure 56A: Gaussian KLR versus LeNet: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Gaussian KLR - LeNet)  
![](images/70e9ce02fd8015ec2551b901c8e8485a3847240f81ed112c464540f661c6fb2f.jpg)

![](images/9420caf373b8a32ca3b02f1924335e8a019c8710f6fffbf4931c3fe1e6f735d2.jpg)

![](images/601e6c86a05802762ef2ca53f451965eff38a1a658747a99e78dbae643702590.jpg)

![](images/0d743a8e355fda32c5714a9937fd51cae18506b8aa4838130604086e8500ae0b.jpg)

![](images/4ca5e5eabd0965d28e69e7f0778726e64b09cd3dca85b597fcd1074cc79d21b7.jpg)

![](images/1cc92deee344493fbc35aa3c1e2c911c54ddb3dc737842bb7885f8dcb4cac1c0.jpg)

![](images/f24bec16b4792be99f17a50540a16706787b7cc9dcbf4c5c21832f4a9183291f.jpg)

![](images/2053fa15bd20845d05a9d2520f2ae91e6e6199745f2d7ad2e487281db1698acb.jpg)

Figure 57A: Gaussian KRR versus LeNet: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Gaussian KRR - LeNet)  
![](images/0b26ae6a600654a983dba425d43f6ac70da45821263ad88a74cc286be8633014.jpg)

![](images/faa80940dbb194ad2a8e657a7951bb2b3b4f725ad0dc140779063de193cb8452.jpg)

![](images/14acf70c724c4802e4dc3fe38442da296d34ddb366b6dc97a8853942506c0055.jpg)

![](images/10eb1d48d2e76818b0d759a2731aea91758f1b24f8c90d8f1efc56b8a7c52d14.jpg)

![](images/d1951b5f4edbaf08e6c76e3758a355ba0191a63a57c585766a77bb47d8f33cbf.jpg)

![](images/297aa86824f8eeedf03c8d18709ede7068809a59214aeca8ef0e57afc48da6ba.jpg)

Figure 58A: Gaussian RFM versus LeNet: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Gaussian RFM - LeNet)  
![](images/429f6543c8fdc0167d9a911662601c93bc7156b76c89ca5c2c08ff7f0c61df45.jpg)

![](images/ac2c295618324e46849340bf870b3bce754a4867c3c544d673ff3e0c3e4418b8.jpg)

Figure 59A: Gaussian SVM versus LeNet: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Gaussian SVM - LeNet)  
![](images/61a36bd799d5f22fde41403d5bc68aac41b6b686c1ed57cf7b5b0762ca40719c.jpg)

![](images/b92d558376358be351ab939854bf3b48cc2b5d09c7a9812953f16e34e6a68591.jpg)

Figure 60A: Gaussian RFC (5k features) versus LeNet: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models ∆ robust accuracy (Gaussian RFC (5k features) - LeNet)  
![](images/ce7390a92a3776bf82cd982a065c3e8c0823626c1aafd1759990436152534fc8.jpg)

Figure 61A: Gaussian RFC (20k features) versus LeNet: $\ell _ { 2 }$ and $\ell _ { \infty }$ perturbation for FashionMNIST KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD attack is used to attack the models. ∆ robust accuracy (Gaussian RFC (20k features) - LeNet)  
![](images/5ff86bb9777623f8a6bb450178e56a2437566f3b48228b2b5b4ca65496b1f9b9.jpg)

## C.6 Kernels on SVHN and CIFAR-10

We trained Gaussian KRR, Gaussian RFM, Laplacian KRR, Laplacian RFM, FCNN, and LeNet on 10 class SVHN and different binary problems of CIFAR-10 (namely Aeroplane vs frog (0v6), Aeroplane vs Horse (0v7), Automobile vs Cat (1v3), Deer vs Truck (4v9) and Dog vs Ship (5v8)). Hyperparameter tuning of models used same grid search as for the other three datasets namely FMNIST, KMNIST, QMNIST. As in Fig. 62A, Laplacian KRR and FCNN match in robustness with Gaussian RFM beating them all. The reason behind Gaussian RFM being the best model can be inferred from the fact that its base kernel despite being poor in benign accuracy is quite robust (Table 4A). Hence, even though further iterations in RFM generally lose the robustness (as we will see in §D.4), the loss in robustness is not much to make it brittle. Here, the base KM for Gaussian RFM is having different hyperparameters than the Gaussian KRR in Fig. 62A. Results for CIFAR-10 are in Fig. 63A, Fig. 64A, Fig. 65A, Fig. 66A.

Figure 62A: $\ell _ { \infty }$ and $\ell _ { 2 }$ attacks on SVHN. (Left) corresponds to $\ell _ { \infty }$ and (Right) corresponds to $\ell _ { 2 }$ autoattacks.  
![](images/0bc2b67347a6f670b70ee008d5e9a09cd904c7e1c2ee2f625b431b3e7dc5a157.jpg)

Table 4A: Robust accuracies for the Gaussian kernel and Gaussian RFM with bandwidth 0.05 on SVHN $\overline { { \epsilon \left( L _ { \infty } \right) } }$ Gaussian Kernel (%) Gaussian RFM (%)
<table><tr><td>C(L∞)</td><td>(0)</td><td>(0) 82.46</td></tr><tr><td>0</td><td>62.56 62.11</td><td>81.48</td></tr><tr><td>0.1/255 1/255</td><td>58.22</td><td>71.10</td></tr><tr><td>2/255</td><td>53.86</td><td>56.59</td></tr><tr><td>4/255</td><td>46.01</td><td>36.08</td></tr><tr><td>6/255</td><td>39.48</td><td>26.30</td></tr><tr><td>8/255</td><td>34.44</td><td></td></tr><tr><td></td><td>27.60</td><td>21.29</td></tr><tr><td>12/255</td><td>21.27</td><td>16.26</td></tr><tr><td>20/255</td><td>17.25</td><td>9.77</td></tr><tr><td>32/255 40/255</td><td>15.81</td><td>3.91</td></tr><tr><td></td><td></td><td>2.25</td></tr><tr><td>55/255</td><td>13.58</td><td>1.40</td></tr></table>

Figure 63A: Gap in robust accuracy (ra) for $\ell _ { \infty }$ perturbation on 5 1v1 binary classification tasks from CIFAR-10. (Top) row has Laplace Kernel, and (Bottom) row has Laplace RFM. (First) Column is FCNN, and (Second) column is LeNet. PGD is used to attack the models. ∆ robust accuracy (row - col) for $\ell _ { \infty }$ adversarial perturbations on CIFAR-10  
![](images/336edace6ca37d1f06baffc86865455f37b32e08ee23297c1f3d15bc25c94cf1.jpg)

Figure 64A: Gap in robust accuracy (ra) for $\ell _ { 2 }$ perturbation on 5 1v1 binary classification tasks from CIFAR-10. (Top) row has Laplace Kernel, and (Bottom) row has Laplace RFM. (First) column is FCNN, and (Second) column is LeNet. PGD is used to attack the models. ∆ robust accuracy (row - col) for $\ell _ { 2 }$ adversarial perturbations on CIFAR-10  
![](images/cef4c2b20a2751ccf3f60b4af63048b80aa8cc612e2c6aecdf8f0da2147cded8.jpg)

Figure 65A: Gap in robust accuracy (ra) for $\ell _ { \infty }$ perturbation on 5 1v1 binary classification tasks from CIFAR-10. (Top) row has Gaussian Kernel, and (Bottom) row has Gaussian RFM. (First) column is FCNN, and (Second) column is LeNet. PGD is used to attack the models. ∆ robust accuracy (row - col) for $\ell _ { \infty }$ adversarial perturbations on CIFAR-10  
![](images/8c5a57ae7d902ebdfc04ad2e28e728379ef18fbc5c7072ac57c0bfbf7548655a.jpg)

Figure 66A: Gap in robust accuracy (ra) for $\ell _ { 2 }$ perturbation on 5 1v1 binary classification tasks from CIFAR-10. (Top) row has Gaussian Kernel, and (Bottom) row has Gaussian RFM. (First) column is FCNN, and (Second) column is LeNet. PGD is used to attack the models. ∆ robust accuracy (row - col) for $\ell _ { 2 }$ adversarial perturbations on CIFAR-10  
![](images/49779555d7869d233b897013edbeaf5ef93bbb4159aced1708a83bbc48d6d9a7.jpg)

## C.7 WideResNet vs LeNet

We compare the PGD robust accuracy of WideResNet-28-10 (WRN) and LeNet-5 on multiclass FMNIST, KMNIST, and QMNIST datasets. Table 5A reports the accuracy (%) for $\ell _ { \infty }$ perturbation budgets €. Across all datasets, WRN attains higher clean accuracy than LeNet-5 (93.57% vs. 90.54% on FMNIST, 98.61% vs. 95.49% on KMNIST, and 99.46% vs. 98.91% on QMNIST). However, this advantage does not persist as ε grows. Hence, our comparison of KMs against LeNets is transferable to larger CNN models like WideResNets as well.

Table 5A: PGD robust accuracy (%) of WRN-28-10 and LeNet-5 on FMNIST, KMNIST, and QMNIST Bold marks the better model per dataset.
<table><tr><td rowspan="2">€</td><td colspan="2">FMNIST</td><td colspan="2">KMNIST</td><td colspan="2">QMNIST</td></tr><tr><td>WRN</td><td>LeNet</td><td>WRN</td><td>LeNet</td><td>WRN</td><td>LeNet</td></tr><tr><td>0</td><td>93.57</td><td>90.54</td><td>98.61</td><td>95.49</td><td>99.46</td><td>98.91</td></tr><tr><td>0.1/255</td><td>92.64</td><td>90.03</td><td>98.58</td><td>95.42</td><td>99.46</td><td>98.90</td></tr><tr><td>1/255</td><td>77.60</td><td>84.04</td><td>98.45</td><td>94.92</td><td>99.37</td><td>98.75</td></tr><tr><td>2/255</td><td>55.64</td><td>74.82</td><td>98.19</td><td>94.20</td><td>99.27</td><td>98.54</td></tr><tr><td>4/255</td><td>13.54</td><td>55.67</td><td>97.60</td><td>92.40</td><td>99.03</td><td>98.00</td></tr><tr><td>6/255</td><td>0.80</td><td>39.68</td><td>96.61</td><td>90.35</td><td>98.70</td><td>97.39</td></tr><tr><td>8/255</td><td>0.02</td><td>26.88</td><td>94.99</td><td>87.34</td><td>98.31</td><td>96.42</td></tr><tr><td>12/255</td><td>0.00</td><td>8.95</td><td>84.18</td><td>80.23</td><td>96.72</td><td>93.43</td></tr><tr><td>20/255</td><td>0.00</td><td>0.18</td><td>30.65</td><td>57.57</td><td>75.85</td><td>79.66</td></tr><tr><td>32/255</td><td>0.00</td><td>0.00</td><td>1.01</td><td>17.58</td><td>1.68</td><td>35.52</td></tr><tr><td>40/255</td><td>0.00</td><td>0.00</td><td>0.05</td><td>4.06</td><td>0.07</td><td>11.61</td></tr><tr><td>55/255</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.07</td><td>0.00</td><td>0.32</td></tr></table>

## C.8 KMs and adversarially trained KMs vs adversarially trained nets

Figs. 67A to 70A contains the fine-grained 1v1 results. The multiclass results are in Figs. 1A, 2, 4A and 5A.

Adversarial KLR - Benign KLR for l∞ attacks on KMNIST  
Figure 67A: Adv-KLR versus KLR for FashionMNIST and KMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (bottom) row corresponds to the KMNIST dataset. PGD is used to attack the models.  
![](images/255b9f8817631f6446837113a116f9513914a6b8d8b884e7eb1233a817640c48.jpg)

![](images/52a168e466d336445a71a3d2809035b9f55b6702c46aa655c6d8fd195cff37a0.jpg)

Figure 68A: Adv-KLR versus FCNN for FashionMNIST and KMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (bottom) row corresponds to the KMNIST dataset. PGD is used to attack the models. Δ robust accuracy (Adv-KLR - FCNN)  
![](images/2054b31c25a57bdce4d98b7836fde4dfc3050fc012796ef5d9f18e33dc79a5f7.jpg)

Figure 69A: Adv-KLR versus Adv-FCNN trained at norm $\begin{array} { r } { \ell _ { \infty } = \frac { 1 2 } { 2 5 5 } \colon } \end{array}$ for FashionMNIST and KMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Bottom) row corresponds to the KMNIST dataset. PGD is used to attack the models. Δ robust accuracy (Adv-KLR - Adv-FCNN)  
![](images/98f12038667711f2886df2f49316af2376bd2198187a11d3288e6d5ca656231c.jpg)

![](images/33987ea9e2b43ea91de7c45b31da2c4ae66f1d229619b058c6128d91b1ba8536.jpg)

![](images/dc34da7d35c87eec9f00b210bde80d81c13adade0f6236dfbefc41e3409fc18e.jpg)

Figure 70A: Laplacian KRR versus $\ell _ { 2 } = 1 . 0$ norm and $\ell _ { \infty } = \frac { 6 } { 2 5 5 }$ adversarially trained FCNNs: $\ell _ { 2 }$ and $\ell _ { \infty }$ norm adversarially trained FCNNs were attacked at their respective norm adversarial perturbations for FashionMNIST, KMNIST and QMNIST. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. PGD is used to attack the models. ∆ robust accuracy (Laplacian kernel - adv-FCNN)  
![](images/183d59cde09926e9322c7c2b562f0746cc7b08200122665fe580dcba3856ccd5.jpg)

## D Further Ablations

## D.1 Results for gradient-free (black-box) attacks

Carlini et al. [7] strongly recommend the use of gradient-free attacks to ensure that robustness is not over-estimated due to gradient masking. We use SPSA as recommended by Uesato et. al. [38]. The number of samples used to approximate the gradient is 16. For KLR the number of samples used is 32. For both $\ell _ { \infty }$ and $\ell _ { 2 }$ attacks, the step size in the finite difference step is chosen as $\begin{array} { r } { \alpha = \frac { \varepsilon } { 2 } } \end{array}$ , with lower bound and upper bound being 0.0001, and 0.001 respectively. In the case of $\ell _ { 2 }$ attacks and $\ell _ { \infty }$ , we set PGD step size to be $\frac { \varepsilon } { 1 0 }$ . Our results in Tables 6A to 11A and Fig. 71A, Fig. 72A, Fig. 73A, Fig. 74A, Fig. 75A, Fig. 76A demonstrate that SPSA is a weaker attack than PGD, indicating no gradient masking, except for the case of Laplace RFC.

We attack the Laplacian KRR, Laplacian RFM, Gaussian KRR, Gaussian RFM, Laplacian KLR, Gaussian KLR, Laplacian SVM, Gaussian SVM, Adversarially trained Gaussian KLR, FCNN and LeNet on FashionMNIST, KMNIST, and QMNIST using SPSA and compare its robust accuracy to that under PGD attacks. We can see the SPSA attack is weaker than the standard PGD, hence confirming there is no gradient masking for the aforementioned models. For the multiclass case, Table 6A summarizes this result for the $\ell _ { \infty }$ attack for Laplacian KRR. Table 7A has the SPSA vs. PGD results for $\ell _ { \infty }$ attack for Gaussian KLR. Table 8A contains SPSA results for Laplacian KLR. Table 9A contains the SPSA vs. PGD results for Adversarially trained Gaussian KLR. Table 10A contains the SPSA vs. PGD results for Laplacian SVC. Table 11A contains the SPSA vs. PGD results for Gaussian SVC. Tables 12A and 13A contains preliminary results for Gaussian RFC and Laplacian RFC respectively. From Table 13A, we can see that Laplace RFCs exhibit gradient masking. Comprehensive results corresponding to RFCs are given in §D.7.

Table 6A: Comparison of robust accuracies of Laplacian KRR in the multi-class setting for SPSA, PGD. and the difference of the two across datasets for $\ell _ { \infty }$ perturbations. The difference is computed as SPSA minus PGD.
<table><tr><td rowspan="2">ε</td><td colspan="3">FMNIST</td><td colspan="3">KMNIST</td><td colspan="3">QMNIST</td></tr><tr><td>ra(SPSA)</td><td>ra(PGD)</td><td>diff</td><td>ra(SPSA)</td><td>ra(PGD)</td><td>diff</td><td>ra(SPSA)</td><td>ra(PGD)</td><td>diff</td></tr><tr><td>0</td><td>90.26</td><td>90.26</td><td>0.00</td><td>92.62</td><td>92.62</td><td>0.00</td><td>98.23</td><td>98.23</td><td>0.00</td></tr><tr><td>1</td><td>89.28</td><td>88.57</td><td>0.71</td><td>92.23</td><td>91.95</td><td>0.28</td><td>98.12</td><td>98.05</td><td>0.07</td></tr><tr><td> $\overline { { 2 5 5 } }$   $\overline { { 2 5 5 } }$ </td><td>88.16</td><td>86.14</td><td>2.02</td><td>91.92</td><td>91.33</td><td>0.59</td><td>98.02</td><td>97.91</td><td>0.11</td></tr><tr><td> $\overline { { 2 5 5 } }$ </td><td>85.25</td><td>81.19</td><td>4.06</td><td>91.3</td><td>89.95</td><td>1.35</td><td>97.86</td><td>97.51</td><td>0.35</td></tr><tr><td> $\overline { { 2 5 5 } }$ </td><td>82.34</td><td>74.02</td><td>8.32</td><td>90.47</td><td>88.29</td><td>2.18</td><td>97.62</td><td>97.06</td><td>0.56</td></tr><tr><td></td><td>78.63</td><td>66.33</td><td>12.30</td><td>89.73</td><td>86.67</td><td>3.06</td><td>97.35</td><td>96.52</td><td>0.83</td></tr><tr><td></td><td>69.83</td><td>51.74</td><td>18.09</td><td>87.93</td><td>82.49</td><td>5.44</td><td>96.74</td><td>95.11</td><td>1.63</td></tr><tr><td></td><td>51.83</td><td>30.35</td><td>21.48</td><td>83.35</td><td>69.53</td><td>13.82</td><td>95.02</td><td>89.59</td><td>5.43</td></tr><tr><td>徑號號號</td><td>31.68</td><td>11.73</td><td>19.95</td><td>72.79</td><td>35.47</td><td>37.32</td><td>89.17</td><td>61.80</td><td>27.37</td></tr><tr><td>255</td><td>22.36</td><td>3.20</td><td>19.16</td><td>61.42</td><td>13.52</td><td>47.90</td><td>79.32</td><td>30.93</td><td>48.40</td></tr></table>

Table 7A: Comparison of robust accuracies of Gaussian KLR in the multi-class setting for SPSA, PGD, and the difference between the two across datasets for $\ell _ { \infty }$ perturbations. The difference is computed as SPSA minus PGD.
<table><tr><td rowspan="2"></td><td colspan="3">FMNIST</td><td colspan="3">QMNIST</td></tr><tr><td>ra(SPSA)</td><td>ra(PGD)</td><td>diff</td><td>ra(SPSA)</td><td>ra(PGD)</td><td>diff</td></tr><tr><td>0</td><td>86.95</td><td>86.95</td><td>0.00</td><td>97.47</td><td>97.47</td><td>0.00</td></tr><tr><td>1</td><td>86.10</td><td>85.56</td><td>0.54</td><td>97.37</td><td>97.33</td><td>0.05</td></tr><tr><td>15號5</td><td>78.51</td><td>72.30</td><td>6.21</td><td>96.32</td><td>95.66</td><td>0.66</td></tr><tr><td></td><td>43.25</td><td>35.07</td><td>8.18</td><td>82.31</td><td>65.17</td><td>17.13</td></tr></table>

Table 8A: Comparison of robust accuracies of Laplacian KLR in the multi-class setting under SPSA and PGD attacks across FMNIST and KMNIST for $\ell _ { \infty }$ perturbations. The difference is computed as SPSA minus PGD. All values are reported in percentages.
<table><tr><td></td><td colspan="3">FMNIST</td><td colspan="3">KMNIST</td></tr><tr><td>ε</td><td>ra(SPSA)</td><td> $\mathrm { r a } ( \mathrm { P G D } )$ </td><td>diff</td><td>ra(SPSA)</td><td>ra(PGD)</td><td>diff</td></tr><tr><td>0</td><td>88.18</td><td>88.18</td><td>0.00</td><td>92.26</td><td>92.26</td><td>0.00</td></tr><tr><td>1</td><td>87.10</td><td>86.64</td><td>0.46</td><td>91.82</td><td>91.56</td><td>0.26</td></tr><tr><td></td><td>85.86</td><td>84.99</td><td>0.87</td><td>91.39</td><td>90.92</td><td>0.47</td></tr><tr><td></td><td>83.56</td><td>81.44</td><td>2.12</td><td>90.29</td><td>89.31</td><td>0.98</td></tr><tr><td></td><td>77.93</td><td>73.03</td><td>4.90</td><td>88.00</td><td>85.08</td><td>2.92</td></tr><tr><td></td><td>71.72</td><td>63.73</td><td>7.99</td><td>85.05</td><td>79.24</td><td>5.81</td></tr><tr><td></td><td>59.20</td><td>48.19</td><td>11.01</td><td>76.55</td><td>62.61</td><td>13.94</td></tr><tr><td></td><td>45.04</td><td>33.83</td><td>11.21</td><td>58.34</td><td>29.30</td><td>29.04</td></tr><tr><td>2P4387號0號30</td><td>38.01</td><td>30.61</td><td>7.40</td><td>42.70</td><td>12.49</td><td>30.21</td></tr><tr><td>255</td><td>30.38</td><td>26.00</td><td>4.38</td><td>20.02</td><td>1.31</td><td>18.71</td></tr></table>

Table 9A: Comparison of robust accuracies of adversarially trained Gaussian KLR in the multi-class setting under SPSA and PGD attacks for $\ell _ { \infty }$ perturbations on KMNIST. The difference is computed as $\mathrm { r a ( S P S A ) } - \mathrm { r a ( P G D ) }$
<table><tr><td rowspan="2"></td><td colspan="2">KMNIST</td></tr><tr><td>ra(SPSA) ra(PGD)</td><td>diff</td></tr><tr><td>0</td><td>88.40 88.40</td><td>0.00</td></tr><tr><td>1</td><td>88.35 88.24</td><td>0.11</td></tr><tr><td>25</td><td>87.26 86.27</td><td>0.99</td></tr><tr><td></td><td>85.24 82.40</td><td>2.84</td></tr><tr><td></td><td>82.78 77.63</td><td>5.15</td></tr><tr><td>號靈號5</td><td>77.17 66.61</td><td>10.56</td></tr><tr><td></td><td></td><td></td></tr></table>

Table 10A: Comparison of robust accuracies of Laplacian SVM in the multi-class setting for SPSA, PGD, and the difference between the two across datasets for $\ell _ { \infty }$ perturbations.
<table><tr><td rowspan="2">ω</td><td colspan="3">FMNIST</td><td colspan="3">KMNIST</td></tr><tr><td>ra(SPSA)</td><td>ra(PGD)</td><td>diff</td><td>ra(SPSA)</td><td>ra(PGD)</td><td>diff</td></tr><tr><td>0</td><td>90.01</td><td>90.01</td><td>0.00</td><td>90.54</td><td>90.54</td><td>0.00</td></tr><tr><td>0.1 255</td><td>89.90</td><td>89.87</td><td>0.03</td><td>90.53</td><td>90.47</td><td>0.06</td></tr><tr><td>1 255</td><td>89.17</td><td>88.39</td><td>0.78</td><td>90.27</td><td>89.87</td><td>0.40</td></tr><tr><td>2 255</td><td>88.38</td><td>86.63</td><td>1.75</td><td>90.03</td><td>89.02</td><td>1.01</td></tr><tr><td>4 255</td><td>86.76</td><td>82.18</td><td>4.58</td><td>89.43</td><td>87.34</td><td>2.09</td></tr><tr><td>6 255</td><td>84.86</td><td>76.01</td><td>8.85</td><td>88.85</td><td>85.41</td><td>3.44</td></tr><tr><td>8 255</td><td>82.87</td><td>69.11</td><td>13.76</td><td>88.32</td><td>82.89</td><td>5.43</td></tr><tr><td>12 255</td><td>78.20</td><td>55.21</td><td>22.99</td><td>87.02</td><td>76.84</td><td>10.18</td></tr><tr><td>20 255</td><td>66.61</td><td>34.76</td><td>31.85</td><td>83.66</td><td>60.43</td><td>23.23</td></tr><tr><td>32 255</td><td>49.28</td><td>12.67</td><td>36.61</td><td>76.26</td><td>26.91</td><td>49.35</td></tr><tr><td>40 255</td><td>40.51</td><td>4.50</td><td>36.01</td><td>69.24</td><td>11.37</td><td>57.87</td></tr><tr><td>55 255</td><td>26.59</td><td>0.27</td><td>26.32</td><td>51.21</td><td>0.91</td><td>50.30</td></tr></table>

Table 11A: Comparison of robust accuracies of Gaussian SVM in the multi-class setting under SPSA and PGD attack, along with their differences across FMNIST and KMNIST datasets for $\ell _ { \infty }$ perturbations.
<table><tr><td></td><td colspan="3">FMNIST</td><td colspan="3">KMNIST</td></tr><tr><td>ε</td><td>ra(SPSA)</td><td>ra(PGD)</td><td>diff</td><td>ra(SPSA)</td><td>ra(PGD)</td><td>diff</td></tr><tr><td>0</td><td>89.62</td><td>89.62</td><td>0.00</td><td>93.14</td><td>93.14</td><td>0.00</td></tr><tr><td>0.1</td><td>89.49</td><td>89.29</td><td>0.20</td><td>93.09</td><td>93.00</td><td>0.09</td></tr><tr><td>25</td><td>88.42</td><td>86.28</td><td>2.14</td><td>92.76</td><td>92.11</td><td>0.65</td></tr><tr><td></td><td>86.91</td><td>82.41</td><td>4.50</td><td>92.40</td><td>91.02</td><td>1.38</td></tr><tr><td></td><td>84.02</td><td>73.17</td><td>10.85</td><td>91.72</td><td>88.74</td><td>2.98</td></tr><tr><td></td><td>80.98</td><td>61.94</td><td>19.04</td><td>90.88</td><td>86.09</td><td>4.79</td></tr><tr><td></td><td>77.25</td><td>50.67</td><td>26.58</td><td>90.10</td><td>82.99</td><td>7.11</td></tr><tr><td></td><td>68.34</td><td>32.45</td><td>35.89</td><td>88.26</td><td>74.43</td><td>13.83</td></tr><tr><td></td><td>50.23</td><td>8.29</td><td>41.94</td><td>83.44</td><td>49.51</td><td>33.93</td></tr><tr><td></td><td>28.78</td><td>0.19</td><td>28.59</td><td>71.08</td><td>13.68</td><td>57.40</td></tr><tr><td>152彈4767徑號號狮號</td><td>18.90</td><td>0.01</td><td>18.89</td><td>58.86</td><td>3.89</td><td>54.97</td></tr><tr><td>255</td><td>7.60</td><td>0.00</td><td>7.60</td><td>35.80</td><td>0.13</td><td>35.67</td></tr></table>

Table 12A: Comparison of PGD and SPSA robust accuracy (%) for Gaussian Random Feature Classifiers (RFC) trained with cross-entropy loss on Fashion-MNIST. The model uses 50,000 random features and random seed 150.
<table><tr><td> $\overline { { \varepsilon \left( L _ { \infty } \right) } }$ </td><td>PGD (%)</td><td>SPSA (%)</td></tr><tr><td>0</td><td>89.66</td><td>89.66</td></tr><tr><td>0.1/255</td><td>89.28</td><td>89.34</td></tr><tr><td>1/255</td><td>84.63</td><td>85.69</td></tr><tr><td>2/255</td><td>78.54</td><td>81.16</td></tr><tr><td>4/255</td><td>63.72</td><td>69.58</td></tr><tr><td>6/255</td><td>51.09</td><td>58.07</td></tr><tr><td>8/255</td><td>42.91</td><td>49.19</td></tr><tr><td>12/255</td><td>28.26</td><td>37.12</td></tr><tr><td>20/255</td><td>6.85</td><td>15.96</td></tr><tr><td>32/255</td><td>0.07</td><td>1.41</td></tr><tr><td>40/255</td><td>0.00</td><td>0.17</td></tr><tr><td>55/255</td><td>0.00</td><td>0.07</td></tr></table>

Table 13A: Comparison of PGD and SPSA robust accuracy (%) for Laplacian Random Feature Classifiers (RFC) trained with cross-entropy loss on KMNIST. The model uses 50,000 random features and random seed 150.
<table><tr><td> $\overline { { \varepsilon \left( L _ { \infty } \right) } }$ </td><td>PGD (%)</td><td>SPSA (%)</td></tr><tr><td>0</td><td>89.43</td><td>89.43</td></tr><tr><td>0.1/255</td><td>88.46</td><td>87.57</td></tr><tr><td>1/255</td><td>87.22</td><td>77.11</td></tr><tr><td>2/255</td><td>86.66</td><td>69.48</td></tr><tr><td>4/255</td><td>85.84</td><td>56.25</td></tr><tr><td>6/255</td><td>85.20</td><td>42.82</td></tr><tr><td>8/255</td><td>84.62</td><td>30.12</td></tr><tr><td>12/255</td><td>83.59</td><td>15.28</td></tr><tr><td>20/255</td><td>81.08</td><td>8.55</td></tr><tr><td>32/255</td><td>75.82</td><td>8.08</td></tr><tr><td>40/255</td><td>71.15</td><td>8.07</td></tr><tr><td>55/255</td><td>59.37</td><td>8.04</td></tr></table>

![](images/82da7cd48d0af57e12cd9a8a7b82b18569e6bda95b3abb274923863ddfbce541.jpg)

Figure 71A: SPSA attacked Laplacian KRR versus PGD attacked Laplacian KRR. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries.  
Δ robust accuracy (Laplacian KRR (SPSA - PGD))  
![](images/744b4c15620312169ca05a9dc63e2bec1ef91d70b21a8842b311530b31c1dfbc.jpg)

Figure 72A: SPSA attacked Laplacian RFM versus PGD attacked Laplacian RFM. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries.  
Δ robust accuracy (Laplacian RFM (SPSA - PGD))  
![](images/45ad9e2496f5374e8ac2e930ce9e08cfe82c2441fcb668cc42e949fb11e3bdc7.jpg)

Figure 73A: SPSA attacked Gaussian KRR versus PGD attacked Gaussian KRR. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries.  
∆ robust accuracy (Gaussian KRR (SPSA - PGD))  
![](images/8ce4bf8bc2caef80ae8e7ceb31bda6a5c27d457db0845586287d58eb177951a0.jpg)

Figure 74A: SPSA attacked Gaussian RFM versus PGD attacked Gaussian RFM. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries.  
Δ robust accuracy (Gaussian RFM (SPSA - PGD))  
![](images/7baa4dd5ca63f0c6bcc31eb729548aa388f011abcbf2c49e82ad662ed62611a5.jpg)

Figure 75A: SPSA attacked FCNN versus PGD attacked FCNN. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. Δ robust accuracy (FCNN (SPSA - PGD))  
![](images/0a2e0d4a5899768e997fd8a7898ee7e8bc2aa7729aa07c5f9ec73e080280bd66.jpg)

Figure 76A: SPSA attacked LeNet versus PGD attacked LeNet. (Top) row corresponds to the results for FashionMNIST dataset, (Middle) row corresponds to the KMNIST dataset, and (Bottom) row corresponds to the QMNIST dataset. Of these, (First) column corresponds to the robust accuracies for $\ell _ { \infty }$ adversaries while (Second) column portrays the robust accuracies for $\ell _ { 2 }$ adversaries. ∆ robust accuracy (LeNet (SPSA - PGD))  
![](images/0e814064e04ab4ce39aee331f575c7b4b2eae6dc32a3a940964b4810f37eeb01.jpg)

## D.2 Robustness on Tabular datasets

Recent work [32, 3] has shown that RFMs and their variants are state-of-the-art on tabular data benchmarks. Given this, it is instructive to determine whether kernel-based classifiers are also more robust than neural networks on tabular data. However, since tabular data has dimension-wise semantic constraints, we use the Cost-aware Feasible Attack (CaFA) method from [5] to ensure that all generated adversarial examples remain feasible. Further details about the attack are in §A.2

Results: When running CaFA, it marks points as infeasible if they do not satisfy the automatically determined integrity constraints. All accuracies in Table 14A are only reported on the portion of the dataset found to be feasible. We find that on the Phishing and Bank datasets, KMs are indeed more robust. However, the FCNN is comfortably more robust on the Adult dataset. This is the only dataset where we find KMs to be less robust, and we plan to investigate the reason for this in future work.

Table 14A: CaFA attacks on FCNN, KRR and RFM on the Phishing, Bank Marketing and Adult datasets. Robust accuracy numbers are reported only on compliant samples after each stage. The accuracy values are rounded to the nearest tenth.
<table><tr><td></td><td></td><td colspan="3">Phishing</td><td colspan="3">Bank Marketing</td><td colspan="3">Adult</td></tr><tr><td>ε</td><td>stage</td><td>FCNN</td><td>KRR</td><td>RFM</td><td>FCNN</td><td>KRR</td><td>RFM</td><td>FCNN</td><td>KRR</td><td>RFM</td></tr><tr><td rowspan="3">0.01</td><td>Benign</td><td>94.8</td><td>92.6</td><td>95.2</td><td>87.8</td><td>89.7</td><td>89.7</td><td>84.3</td><td>72.4</td><td>80.8</td></tr><tr><td>Post CaFA</td><td>53.0</td><td>52.4</td><td>70.5</td><td>90.3</td><td>89.5</td><td>78.6</td><td>73.0</td><td>61.2</td><td>43.1</td></tr><tr><td>Post projection</td><td>42.0</td><td>31.1</td><td>63.5</td><td>84.3</td><td>85.7</td><td>75.0</td><td>64.2</td><td>51.2</td><td>38.0</td></tr><tr><td rowspan="3">0.03</td><td>Benign</td><td>94.8</td><td>92.6</td><td>95.2</td><td>87.8</td><td>89.7</td><td>89.7</td><td>84.3</td><td>72.4</td><td>80.8</td></tr><tr><td>Post CaFA</td><td>46.7</td><td>44.3</td><td>60.6</td><td>89.5</td><td>90.5</td><td>79.9</td><td>69.6</td><td>55.6</td><td>39.2</td></tr><tr><td>Post projection</td><td>38.1</td><td>36.8</td><td>56.4</td><td>80.8</td><td>85.6</td><td>66.2</td><td>61.3</td><td>45.4</td><td>33.7</td></tr><tr><td rowspan="3">0.075</td><td>Benign</td><td>94.8</td><td>92.6</td><td>95.2</td><td>87.8</td><td>89.7</td><td>89.7</td><td>84.3</td><td>72.4</td><td>80.8</td></tr><tr><td>Post CaFA</td><td>42.8</td><td>42.3</td><td>49.6</td><td>78.5</td><td>91.9</td><td>76.4</td><td>68.5</td><td>49.1</td><td>36.6</td></tr><tr><td>Post projection</td><td>31.5</td><td>32.2</td><td>41.8</td><td>64.6</td><td>85.9</td><td>70.8</td><td>60.0</td><td>40.0</td><td>30.4</td></tr></table>

## D.3 Results for Matérn Kernel ν versus robust accuracy

We observe that the robust accuracy of Matérn Kernel decreases with increase in the smoothness parameter(ν) value in SVHN, KMNIST, QMNIST. In FashionMNIST robust accuracies stay quite close to each other, although not following a clear pattern. Further details about the experiment are in §B.6, with results in Fig. 77A. PGD is used to attack the Kernel.

![](images/2639b6a1ff1bd46b5fe30235433888b9355a6c67b956dbad4a7f800351e48754.jpg)  
Figure 77A: Robust Accuracy variation with Matérn Kernel smoothness parameter ν for SVHN, KMNIST, QMNIST, FMNIST.

## D.4 Robust Accuracy variation with iteration in RFMs

RFMs are KM variant that learn features iteratively. This compels us to ask what happens to the robust accuracy as RFM learns features. The results in Fig. 78A show that RFMs' robust accuracy decreases with an increase in the feature learning (or with number of iterations), specifically for Laplacian RFMs. We attacked the models using PGD.

![](images/fec25e5cb8fac7b27d8e8327dee441aeef7026ac0281eb0b7e6c694cfd09804c.jpg)  
Figure 78A: Robust Accuracy variation with iterations in Laplacian RFM for SVHN, KMNIST, QMNIST, FMNIST.

## D.5 Robust Accuracy variation with length scale (σ)

Fig. 79A shows that, for the Laplacian kernel, adversarial robustness generally favours smaller length scales σ. Even though smaller σ can yield lower benign accuracy, it often achieves higher robust accuracy as the attack strength increases, indicating a trade-off between benign performance and robustness. The corresponding Gaussian-kernel results are shown in Fig. 80A, where the dependence on σ is less monotonic and more dataset dependent. Ridge (λ) was kept the same across different length scales (σ). Results are given for the KRR. Models are attacked using PGD.

![](images/f0f457c713e1c0a0decc992e3ae068172e51564c4fdd6fbe88658c4a91d5a145.jpg)  
Figure 79A: Robust Accuracy variation with length scale (σ) in Laplacian Kernel for SVHN, KMNIST, QMNIST, FMNIST across different attack ε.

![](images/5ba6625e79c847798bb030430b65cafe4b6841366971e86601808121ea0baa1f.jpg)  
Figure 80A: Robust Accuracy variation with length scale (σ) in Gaussian Kernel for SVHN, KMNIST, QMNIST, FMNIST across different attack ε.

## D.6 Robust Accuracy variation with ridge (λ)

We observe in Fig. 81A that the effect of ridge regularization on KRR is dataset dependent. While smallto-moderate ridge values largely preserve performance, stronger regularization can improve robustness under larger perturbations for FMNIST, KMNIST and QMNIST, but does not provide a consistent benefit for SVHN. This indicates a trade-off between benign accuracy and adversarial robustness. No single ridge value comes out to be optimal across all datasets and attack strengths. Length Scale (σ) was kept the same across different ridge (λ) values. PGD is used to attack KRRi

![](images/d8f4f48e78bf6f3ab5f017d787d5662be4cc847d065f63f1dac1211eac7abe35.jpg)  
Figure 81A: Robust Accuracy variation with ridge (λ) in Laplacian KRR for SVHN, KMNIST, QMNIST, FMNIST across different attack ε.

<table><tr><td colspan="3">Gaussian RFC on FMNIST (Square): SPSA vs PGD across seeds</td></tr><tr><td>SPSA --FCNN</td><td></td><td>LeNet</td></tr><tr><td>▲ PGD</td><td>Laplacian KRR</td><td>…x…… Gaussian KLR</td></tr></table>

## D.7 Number of features in RFC versus robust accuracy

As is evident from Fig. 82A to Fig. 93A, both the benign accuracy as well as robust accuracy improves with an increase in the number of features, except when we train Laplace RFC with square loss. In Laplace RFCs, trained with square loss, benign accuracy improves with an increase in the number of features, but that is not the case with robust accuracy. We observe gradient masking in the case of Laplace RFC, no matter the training loss, as is seen in Figs. 88A to 93A. 5 different seeds are used for RFCs.

Ablation results for number of features vs. robust accuracy for Gaussian RFC on FMNIST, KMNIST QMNIST for square and cross-entropy loss is given in Fig. 82A to Fig. 87A. Results for Laplacian Kernel trained using square and cross entropy training loss range from Fig. 88A to Fig. 93A.

## D.7.1 FMNIST Results for Gaussian RFC

![](images/3e9bba7945a6559bb948f49c564bf8f6cd6626ff90abe034bb05a40e77eb19e3.jpg)  
Number of random features (×1000)  
Figure 82A: Robust Accuracy variation with Number of Features in Gaussian RFC (trained using Square Loss) for FMNIST across different seeds per attack. SPSA and PGD denote attacks on the RFC, each line shows the robust accuracy of a model at a given perturbation radius (ε)

Gaussian RFC on FMNIST (Cross-Entropy): SPSA vs PGD across seeds  
![](images/0c1d6e97fd470fdc711c2e4031d76a9c20bf1a6e1f1f10024568b9d44120ef20.jpg)

![](images/a0967513f5f2e6f3762d84398f6287acf452a65fdf07c2a7fa371eba9fd8ee2d.jpg)  
Number of random features (×1000)  
Figure 83A: Robust Accuracy variation with Number of Features in Gaussian RFC (trained using Cross Entropy) for FMNIST across different seeds per attack. SPSA and PGD denote attacks on the RFC, each line shows the robust accuracy of a model at a given perturbation radius (ε).

![](images/d4c5dcd080ee2605ba5cf6dfd27d344aa4880f03e079e1358beebbd04023ce40.jpg)

![](images/0afd2eaaf38684672fa275f508caf3d552b3e182abef4ec79922eaf9533cdfb3.jpg)  
Number of random features (×1000)  
Figure 84A: Robust Accuracy variation with Number of Features in Gaussian RFC (trained using Square Loss) for KMNIST across different seeds per attack. SPSA and PGD denote attacks on the RFC, each line shows the robust accuracy of a model at a given perturbation radius (ε).

Gaussian RFC on KMNIST (Cross-Entropy): SPSA vs PGD across seeds  
![](images/fc022debd102773b831f4c6aca14de83237ddd684bc4486704cb4502f523c32b.jpg)

![](images/7e4e52e9c3592dec37dea145eb13dab53132795b726b3cbf5751c964b2690c6f.jpg)  
Number of random features (×1000)  
Figure 85A: Robust Accuracy variation with Number of Features in Gaussian RFC (trained using Cross Entropy) for KMNIST across different seeds per attack. SPSA and PGD denote attacks on the RFC, each line shows the robust accuracy of a model at a given perturbation radius (ε).

## D.7.3 QMNIST Results for Gaussian RFC

Gaussian RFC on QMNIST (Square): SPSA vs PGD across seeds  
![](images/8be6e9186b257ca4c9c29dc52082fb6fe10fff0123c4cc0f7c9f07306e405ea4.jpg)

![](images/39bd6bb1986fb023eec3d3c08623a12ac75505d890d1d6dd94180121a5c35dcf.jpg)  
Number of random features (×1000)  
Figure 86A: Robust Accuracy variation with Number of Features in Gaussian RFC (trained using Square Loss) for QMNIST across different seeds per attack. SPSA and PGD denote attacks on the RFC, each line shows the robust accuracy of a model at a given perturbation radius (ε).

Gaussian RFC on QMNIST (Cross-Entropy): SPSA vs PGD across seeds  
![](images/908dbb27113a5d5fd06b2a06f318f0da8d1e81ed46a4886d4b7fa2c46dfedaba.jpg)

![](images/65a23d247d5652492789d16d5f59d74c0efa2d7bc7e6cc1a32ca794262087b4c.jpg)  
Number of random features (×1000)  
Figure 87A: Robust Accuracy variation with Number of Features in Gaussian RFC (trained using Cross Entropy Loss) for QMNIST across different seeds per attack. SPSA and PGD denote attacks on the RFC, each line shows the robust accuracy of a model at a given perturbation radius (ε).

## D.7.4 FMNIST Results for Laplace RFC

Laplace RFC on FMNIST (Square): SPSA vs PGD across seeds  
![](images/0f84bf58f2c3d49e5566661d5271d2529b21c5f6aa8cdf64850a307bd10d842a.jpg)

![](images/5ffa5b9a57b562d96fa8652dddb099c411e7207ef8de76ff6b39cf8a43aad6fb.jpg)  
Number of random features (×1000)  
Figure 88A: Robust Accuracy variation with Number of Features in Laplace RFC (trained using Square Loss) for FMNIST across different seeds per attack. SPSA and PGD denote attacks on the RFC, each line shows the robust accuracy of a model at a given perturbation radius (ε).

Laplace RFC on FMNIST (Cross-Entropy): SPSA vs PGD across seeds
<table><tr><td>◇</td><td>SPSA --- FCNN</td><td></td><td></td><td>LeNet</td></tr><tr><td>▲</td><td>PGD</td><td>—Laplacian KRR</td><td></td><td>……Gaussian KLR</td></tr></table>

![](images/7b5d2d36fce791f9429cb628f37a2931e20283e904900407169a0812f045918a.jpg)  
Number of random features (×1000)  
Figure 89A: Robust Accuracy variation with Number of Features in Laplacian RFC (trained using Cross Entropy) for FMNIST across different seeds per attack. SPSA and PGD denote attacks on the RFC, each line shows the robust accuracy of a model at a given perturbation radius (ε).

Laplace RFC on KMNIST (Square): SPSA vs PGD across seeds  
![](images/5211e38bdcbbc8c0781aacdf40e63444f2229fe058956fd342e33dcad310ed22.jpg)

![](images/4556fe8ce9cb9bdd4cc96508c4e71091d8811c0ad52af48a8e79103b3a5bcd63.jpg)  
Number of random features (×1000)  
Figure 90A: Robust Accuracy variation with Number of Features in Laplace RFC (trained using Square Loss) for KMNIST across different seeds per attack. SPSA and PGD denote attacks on the RFC, each line shows the robust accuracy of a model at a given perturbation radius (ε).

Laplace RFC on KMNIST (Cross-Entropy): SPSA vs PGD across seeds  
![](images/6124a1d13d09ce40f560883d483da2a0dd2e99e2979a2dc72a2dd45e359f933d.jpg)

![](images/08e6c1060f06a516156a4e32645856a99313718d762c6319147894b48e1eb835.jpg)  
Number of random features (×1000)  
Figure 91A: Robust Accuracy variation with Number of Features in Laplacian RFC (trained using Cross Entropy) for KMNIST across different seeds per attack. SPSA and PGD denote attacks on the RFC, each line shows the robust accuracy of a model at a given perturbation radius (ε).

## D.7.6 QMNIST Results for Laplace RFC

Laplace RFC on QMNIST (Square): SPSA vs PGD across seeds  
![](images/e60ce082dba826eb465c290c52901dfce02625811062573c44fbd2a75916123c.jpg)

![](images/c648a1a7f34c2103251aa0834b79a92acc6cabec490b16ce6e9d4ec939fa4952.jpg)  
Number of random features (×1000)  
Figure 92A: Robust Accuracy variation with Number of Features in Laplace RFC (trained using Square Loss) for QMNIST across different seeds per attack. SPSA and PGD denote attacks on the RFC, each line shows the robust accuracy of a model at a given perturbation radius (ε).

Laplace RFC on QMNIST (Cross-Entropy): SPSA vs PGD across seeds
<table><tr><td>◇</td><td>SPSA --●-- FCNN</td><td></td><td></td><td>LeNet</td></tr><tr><td>▲</td><td>PGD</td><td>—— Laplacian KRR</td><td></td><td>……Gaussian KLR</td></tr></table>

![](images/a01ac2dc081e887f8877f48cbedab586bef66c14714a0eb06e2e45b492a215e5.jpg)  
Number of random features (×1000)  
Figure 93A: Robust Accuracy variation with Number of Features in Laplace RFC (trained using Cross Entropy Loss) for QMNIST across different seeds per attack. SPSA and PGD denote attacks on the RFC, each line shows the robust accuracy of a model at a given perturbation radius (ε).