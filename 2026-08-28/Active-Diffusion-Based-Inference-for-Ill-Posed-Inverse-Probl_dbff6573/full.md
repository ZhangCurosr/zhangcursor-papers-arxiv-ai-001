# Active Diffusion-Based Inference for Ill-Posed Inverse Problems under Incomplete Priors

Jitao Xu<sup>1</sup> , Nobuo Sato<sup>2</sup> , Yaohang Li<sup>1</sup>

<sup>1</sup>Department of Computer Science, Old Dominion University, Norfolk, Virginia 23529, USA <sup>2</sup>Jefferson Lab, Newport News, Virginia 23606, USA jxu004@odu.edu, nsato@jlab.org, yaohang@cs.odu.edu

## Abstract

Many scientific and engineering applications require estimating unknown parameters from experimentally observable data – an inverse problem that is inherently challenging due to nonlinearity, noise, and ill-posedness. In this paper, we propose an active diffusion-based inverse problem solver. A DM is trained to learn the mapping between the parameter space and the observable space. By iteratively detecting and correcting model misspecification through posterior uncertainty, the method discovers and learns the correct region of parameter space, even when initial training bounds exclude the true parameters. This provides a principled, Bayesian justification for adaptive domain augmentation and ensures robust inference for inverse problems under incomplete prior knowledge. We demonstrate the effectiveness of our inverse solver for a toy inverse problem with infinite solutions, and for the parameterization of the quantum correlation functions to event observables in a Quantum Chromodynamics analysis of nucleon structure.

## 1 Introduction

Recovering unknown model parameters from measured observables is a fundamental inverse problem that arises across a wide range of scientific and engineering disciplines, including medical imaging [Kruse et al., 2025], remote sensing [Rodgers, 2000; Yuan et al., 2020], non-destructive inspection [Spaeth and Li, 2025], and nuclear physics [Almaeen et al., 2025]. In these applications, the forward model mapping from parameters to observables is well-posed: given a set of parameters, the corresponding observables can be uniquely determined, although carrying out the forward computation can be computationally costly, requiring solving differential equations or applying linear algebraic operators. In contrast, the inverse problem, mapping from observable space back to parameter space, is often ambiguous, unstable, and sensitive to noise. As a result, inverse problems are typically ill-posed, ill-conditioned, with significantly greater computational challenges [Tarantola, 2005].

Recent advances in machine learning have introduced generative AI models as powerful alternatives to solve inverse problems. Unlike deterministic regression-based approaches, these generative models, such as Invertible Neural Networks (INN) [Ardizzone et al., 2019], Variational Autoencoder Inverse Mapper (VAIM) [Almaeen et al., 2021], Normalizing Flows (NF) [Dinh et al., 2014; Rezende and Mohamed, 2015], and Diffusion Models (DM) [Sohl-Dickstein et al., 2015; Song and Ermon, 2019; Ho et al., 2020], learn probability distributions over parameters and observables, enabling inference through sampling. By explicitly modeling uncertainty and generating posterior solution distribution from sampling a latent space, generative AI models provide a natural mechanism for addressing the ill-posedness issue that is intrinsic to inverse problems [Song et al., 2020; Kawar et al., 2022; Chung et al., 2022].

However, the generative AI-based inverse solvers implicitly assume that the admissible parameter space is known in advance. In practice, although sometimes the approximate guesses are available, precise bounds on the true parameters are often unknown, particularly in scientific discovery problems. When the true parameters lie outside the training domain, learned inverse solvers have to extrapolate unpredictably and produce misleading confidence. Particularly, the generative models can fit complex distributions but do not inherently signal when they are operating beyond their training support.

In this paper, we propose an active diffusion-based inverse solver without precise prior knowledge of the parameter domain. Here, DMs are trained to learn the mapping between parameter space and observable space and can be used to do inference by sampling from the conditional distribution of parameters when observables are given. Rather than relying on the fixed parameter bounds, active learning method [Settles, 2009] is adopted to iteratively detect model misspecification through posterior distribution analysis and actively augments the training domain when necessary. By generating new forward simulations in regions indicated by the model’s inference uncertainty, our method progressively discovers and learns the relevant region of parameter space with respect to the true parameters, even when they are outside the initial bounds. A principled Bayesian justification is provided for adaptive domain augmentation. We demonstrate the effectiveness of the proposed method on a toy inverse problem with infinitely many solutions, as well as on a realistic application in nuclear physics involving the parameterization of quantum correlation functions and their mapping to eventlevel observables in a Quantum Chromodynamics analysis of nucleon structure.

## 2 Related Work

Inverse problems are fundamentally ill-posed with nonunique solutions and unstable, which substantially increases their computational complexity. Classical numerical approaches thus rely on incorporating additional assumptions or regularization, such as sparsity [Foucart and Rauhut, 2013] or low dimensionality [Yu et al., 2017], to constrain the solution space. Beyond these generic assumptions, many inverse problem solvers are highly application specific, which depend on detailed domain knowledge. Comprehensive surveys of classical approaches to ill-posed inverse problems across various scientific disciplines can be found in the literature [Colton et al., 2000].

The emergence of data-driven methods and deep learning has significantly expanded the set of available methods for solving inverse problems [Arridge et $a l .$ , 2019]. Deep neural networks have been used to learn the mappings between observables and parameters directly from data, often outperforming traditional numerical methods in not only accuracy, but also computational efficiency. Prior work has explored incorporating deep neural networks as learned regularizers [Li et al., 2020] and extracting prior knowledge [Adler and Oktem, 2017<sup>¨</sup> ].

More recently, generative modeling has provided a probabilistic framework for inverse problems by explicitly learning distributions over parameters and observables. Mixture density networks (MDNs) [Bishop, 1994] represent one of the earliest generative modeling approaches for probabilistic inversion by modeling the conditional posterior as a weighted mixture of parametric distributions, enabling the representation of multimodal solutions but often relying on restrictive distributional assumptions. Invertible neural networks (INN) [Ardizzone et al., 2019] enable exact likelihood evaluation and uncertainty quantification through bijective mappings, but their architectural constraints can limit expressivity. Variational autoencoder–based approaches (VAIM) [Almaeen et $a l .$ , 2021] introduce latent variables to capture multimodal posterior distributions, though approximation errors in the variational objective may affect inference quality. Normalizing flow (NF) models [Dinh et al., 2014; Rezende and Mohamed, 2015] offer expressive density estimation with tractable likelihoods, but similarly require strict invertibility and Jacobian constraints. Diffusion models (DM) [Sohl-Dickstein et al., 2015; Song and Ermon, 2019; Ho et al., 2020] have emerged as a highly expressive alternative, capable of modeling complex high-dimensional distributions through iterative denoising without requiring explicit invertibility.

Several recent studies have applied DMs to inverse problems by conditioning the generative process on observed data, demonstrating strong performance in image reconstruction and scientific inference tasks. [Alghamdi et al., 2025] However, most existing approaches assume that the admissible parameter domain is known in advance and fixed during training. When this assumption is violated, DMs, as expressive generative models, extrapolate beyond their training support without reliably signaling model misspecification. Addressing this limitation remains an open challenge.

Building on the generative approaches, this work introduces an active diffusion-based framework for solving inverse problems, which adaptively expands the parameter domain during training. By detecting inconsistencies through posterior analysis and/or ensemble disagreement, the method iteratively augments the training support. This distinguishes our approach from prior diffusion-based inverse solvers, providing an effective mechanism for addressing incomplete prior knowledge in inverse problems.

## 3 Methods

We consider an inverse problem where a set of unknown physical parameters $\theta \in \mathbb { R } ^ { d }$ to be inferred from experimentally measured observables $y \in \mathbb { R } ^ { m }$ with a forward model

$$
\mathcal { F } : \theta \mapsto y .
$$

For a given observable $y ^ { \star }$ , our goal is to estimate $\theta ^ { \star }$ satisfying

$$
y ^ { \star } = { \mathcal { F } } ( \theta ^ { \star } ) .
$$

While the forward model $\mathcal { F }$ may be known, the inverse ${ \mathcal { F } } ^ { - 1 }$ is often analytically intractable and is probably not unique. Since the admissible parameter domain Θ is unknown a priori, training a generative surrogate over a fixed range may exclude the true parameters. To overcome this limitation, we propose an active diffusion-based framework that adaptively expands and refines the support of the training distribution until it captures the region of parameter space consistent with the observed data. We begin by introducing posterior variance as an indicator of domain expansion, and then present a more practically reliable indicator based on ensemble inference.

## 3.1 Diffusion Model for Forward and Inverse Mapping

We train a conditional DM $p _ { \phi } ( \theta _ { t } \mid y , t )$ to approximate the data distribution $p ( \theta )$ and the conditional distribution $p ( \theta \mid$ $y )$ , where t denotes the diffusion timestep and ϕ the model parameters. During training, parameters are drawn from an initial guessed domain $\Theta _ { 0 }$ , and synthetic observables are generated using the forward model:

$$
y = { \mathcal { F } } ( \theta ) .
$$

The DM is trained to represent the joint distribution

$$
p ( \theta , y ) = p ( \theta ) p ( y \mid \theta ) ,
$$

with $y$ serving as conditioning information during the reverse diffusion process. Given a fixed observable $y ^ { \star }$ , the reversetime sampling process approximates the posterior over parameters:

$$
\theta \sim p _ { \phi } ( \theta \mid y ^ { \star } ) .
$$

We denote by $\hat { \theta }$ the posterior mean and by $\Sigma _ { \theta }$ its covariance.

## 3.2 Parameter-Space Uncertainty Estimation as a Diagnostic

Because the initial domain $\Theta _ { 0 }$ may not contain the true parameters, the DM may be forced to extrapolate outside its training support. In such situations, the inferred posterior variance becomes inflated. Formally, if $\theta ^ { \star } \notin \Theta _ { 0 }$ , then after training,

$$
{ \mathrm { T r } } ( \Sigma _ { \theta } ) \to { \mathrm { l a r g e } } .
$$

Hence, the posterior variance acts as an estimator of the model misspecification error:

$$
\mathcal { E } _ { \mathrm { m i s s } } \approx \mathrm { T r } ( \Sigma _ { \theta } ) .
$$

## 3.3 Active Learning Loop

To correct for insufficient training support, we introduce an active-learning procedure that expands the parameter domain adaptively. The algorithm is described as follows.

Step 1: Initial guess. Select an initial parameter domain $\Theta _ { 0 }$ based on a physical guess and generate a dataset on the initial training region

$$
\mathcal { D } _ { 0 } = \{ ( \theta _ { i } , \mathcal { F } ( \theta _ { i } ) ) : \theta _ { i } \in \Theta _ { 0 } \} .
$$

Train the DM on $\mathcal { D } _ { 0 }$

Step 2: Inference on target data. Given the control observable $y ^ { \star }$ , perform reverse diffusion to obtain inferred parameter samples

$$
\tilde { \theta } \sim p _ { \phi } ( \tilde { \theta } \mid y ^ { \star } ) ,
$$

and compute $\Sigma _ { \tilde { \theta } }$ .

Step 3: Uncertainty-triggered domain update. If the uncertainty is small, the inferred parameters are deemed reliable. If the uncertainty is large, i.e.,

$$
\operatorname { T r } ( \Sigma _ { \tilde { \theta } } ) > \tau ,
$$

for some threshold $\tau ,$ we treat the inferred samples $\tilde { \theta }$ as informative proposal points indicating missing training support. τ encodes the acceptable level of posterior uncertainty for a specific inverse problem and should be chosen with respect to observation noise and desired accuracy. Specifically, we define an expanded domain

$$
\Theta _ { k + 1 } = \operatorname { E x p a n d } ( \Theta _ { k } , \tilde { \theta } ) ,
$$

where the expansion operator enlarges $\Theta _ { k }$ to include ${ \tilde { \theta } } .$ New synthetic samples are then generated:

$$
\mathcal { D } _ { k + 1 } = \{ ( \theta _ { j } , \mathcal { F } ( \theta _ { j } ) ) : \theta _ { j } \in \Theta _ { k + 1 } \} .
$$

The DM is fine-tuned on the augmented dataset $\mathcal { D } _ { k + 1 }$

Step 4: Iteration. Steps 2–3 are repeated unti

$$
\operatorname { T r } ( \Sigma _ { \tilde { \theta } } ) \leq \tau ,
$$

indicating that the posterior has contracted to a wellsupported region containing the true parameter $\theta ^ { \star }$ . Figure 1 illustrates the overall active diffusion-based inverse problem solver workflow. The training phase (Step 1) samples parameters from the current domain, generates observables through the forward model, and trains the DM. The inference phase (Steps 2–4) performs inference on the target observables, evaluates uncertainty, and iteratively expands the parameter domain until the uncertainty falls below the threshold.

![](images/e4c083bbf6cc28e65f5b35438b70071f7fc6b1ea506caf2853e603be65195a11.jpg)  
Figure 1: Overview of the active diffusion-based inference framework. Left (Training Phase): parameters are sampled from the prior domain, passed through the forward model to generate training data, and used to train the DM. Right (Inference Phase): given a target observable $y ^ { * }$ , the trained model performs inference; if the estimated uncertainty exceeds the threshold $\tau ,$ the domain is adaptively expanded and the model is retrained until convergence.

## 3.4 Theoretical Justification

## (a) Uncertainty awareness in approximating Bayesian inversion using diffusion models.

The reverse diffusion process converges (under standard assumptions) to samples from the true posterior $p ( \theta \mid y )$ . If the training support does not include the true parameter region, then the learned posterior necessarily exhibits inflated uncertainty:

$$
\Sigma _ { \theta } \uparrow \quad \mathrm { w h e n } \quad \theta ^ { \star } \notin \Theta _ { k } .
$$

Thus, the posterior covariance provides a principled diagnostic for missing support.

## (b) Active domain refinement as an optimal-design problem.

Let $\Theta _ { k }$ be the current domain. If $\theta ^ { \star } \notin \Theta _ { k }$ , the Maximum A Posteriori (MAP) estimate

$$
\hat { \theta } = \arg \operatorname* { m a x } _ { \theta } p _ { \phi } ( \theta \mid y ^ { \star } )
$$

identifies the nearest region of parameter space where additional training samples will maximally reduce epistemic uncertainty. Expanding $\Theta _ { k }$ to include $\hat { \theta }$ minimizes the discrepancy

$$
\mathrm { K L } \big ( p ( \boldsymbol { \theta } \mid \boldsymbol { y } ^ { \star } ) \big | \big | p _ { \phi } ( \boldsymbol { \theta } \mid \boldsymbol { y } ^ { \star } ) \big ) ,
$$

where KL(.) denotes the Kullback-Leibler divergence.

Consequently, the sequence of expanded domains

$$
\Theta _ { 0 } \subset \Theta _ { 1 } \subset \Theta _ { 2 } \subset \cdots
$$

eventually satisfies

$$
\theta ^ { \star } \in \Theta _ { n } { \mathrm { ~ f o r ~ s o m e ~ f i n i t e ~ } } n ,
$$

after which the diffusion posterior contracts around the true parameter region.

## 3.5 Practical Limitations of Posterior-Covariance Diagnostics

In practice, the posterior variance generated by the diffusionbased inverse model is not always a reliable indicator of model misspecification [Nalisnick et al., 2018]. Occasionally, when the true parameters lie outside the training domain, the DM may extrapolate overconfidently and produce posterior distributions with narrow variance. This is well-known in expressive generative models, which can assign high confidence to regions that are unsupported by training data [Zhang et al., 2021]. Therefore, uncertainty estimated from a single DM may fail to signal the inferred parameters being inconsistent with the training domain, resulting in premature convergence. This practical limitation motivates the ensemble-based inference strategy described next, where disagreement across independently trained models is used as a more robust signal of missing training support.

## 3.6 Ensemble-based Inference

To address this issue, an ensemble-based strategy can be used, where multiple DMs are trained independently, each with different random initializations. The key observation is that, when these DMs are properly trained, if the true parameters lie within the training region, these models converge to similar conditional distributions, due to the fact that the learned inverse mappings are well constrained by data. In contrast, when the true parameters are outside the training domain, the inverse problem becomes underconstrained and the learned extrapolations depend sensitively on initialization and training noise. As a result, the ensemble DMs yield systematic disagreement, even though individual ones may predict with low posterior variance.

Let $\{ p _ { \phi _ { k } } ( \theta | y ) \} _ { k = 1 } ^ { K }$ denote an ensemble of independently trained DMs. The ensemble disagreement can be quantified by the variance of posterior means,

$$
\begin{array} { r } { \operatorname { V a r } _ { k } ( \mathbb { E } _ { p _ { \phi _ { k } } } [ \theta | y ] ) . } \end{array}
$$

A small variance indicates that parameter inference is well supported by the training data, whereas a large variance suggests extrapolation beyond the training domain. A more informative measure of ensemble disagreement compares the predicted posterior distributions directly, for example, by computing pairwise distributional similarities or divergences.

Incorporating ensemble disagreement into the active learning loop provides a more robust criterion for domain refinement than simply posterior variance. When the ensemble predictions diverge, new parameter samples are generated in the regions suggested by the ensemble outputs, forward model simulations are carried out, and the DMs are retrained. The active learning process continues until ensemble agreement is reached.

## 4 Results

## 4.1 A Toy Problem

Let $\theta = ( x , y ) \in \mathbb { R } ^ { 2 }$ denote unknown parameters and $z \in$ R the observable, related through the forward model

$$
F : \theta \mapsto z = x ^ { 2 } + y ^ { 2 } .\tag{1}
$$

Given an observable $z ^ { * }$ , the inverse problem admits infinitely many solutions lying on a circle of radius $\sqrt { z ^ { * } }$ centered at the origin. This toy inverse problem is ill-posed, making it wellsuited for evaluating whether the DM can correctly capture the full posterior distribution.

The DM consists of a 4-layer MLP with hidden dimension 256, conditioned on the observable z through a separate embedding network. We use $T = 1 0 0$ diffusion timesteps with a linear noise schedule ranging from $\beta _ { 1 } = 1 0 ^ { - 4 } \mathrm { t o } \beta _ { T } = 0 . 0 2 .$ The initial training domain is $\Theta _ { 0 } = \{ ( x , y ) : x ^ { 2 } + y ^ { 2 } \leq 4 \}$ corresponding to radii $r \in [ 0 , 2 ]$ We generate $5 0 { , } 0 0 0$ training samples by uniformly sampling parameters from $\Theta _ { 0 }$ and computing the corresponding observables via (1), with Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , 0 . 0 1 )$ added to simulate measurement uncertainty. The model is trained for 20,000 epochs using Adam optimizer [Kingma and Ba, 2017] with learning rate $1 0 ^ { - 4 }$ and batch size 256.

After training, we perform inference for both in-domain $( z ^ { * } = 4 ,$ , corresponding to $r = 2 )$ and out-of-domain $( z ^ { * } =$ 25, corresponding to $r \ = \ 5 )$ observables. For the out-ofdomain case, we apply the active learning loop described in Section 3.3, training until convergence is reached. To assess robustness, we repeat the entire procedure by training 4 DMs with independent random initializations.

Figure 2 summarizes the results. In case (a), the observable $z ^ { * } = 4$ lies within the training range. All four independently trained models consistently recover the posterior distribution, producing samples uniformly distributed along the target circle of radius 2. The resulting marginal distributions closely match the theoretical densities, and the pairwise Wasserstein distances between model predictions are small (mean $\bar { W } = 0 . 1 2 )$ , indicating strong agreement across the ensemble. In contrast, case (b) corresponds to an out-of-domain observable, $z ^ { * } = 2 5$ . Here, the independently trained models yield inconsistent posterior predictions scattered across parameter space. This behavior is reflected by substantially larger pairwise Wasserstein distances (mean $\dot { \bar { W } } = 3 . 0 0 )$ , signaling ensemble disagreement and model extrapolation beyond the training support. Case (c) shows the inference results for $z ^ { * } = \bar { 2 } 5$ after training with active domain expansion. Following the active learning loop, all four models recover the correct posterior, with samples concentrated along the target circle of radius 5. The pairwise Wasserstein distances decrease remarkably (mean $\bar { W } = 0 . 1 9 )$ , returning to levels comparable with the in-domain case. These results demonstrate that ensemble disagreement, quantified by pairwise Wasserstein distance, provides a reliable indicator of out-of-domain extrapolation. Then, the active learning loop systematically expands the training domain toward regions supported by the observations, enabling the DM to recover the correct posterior distribution. Here in Figure 2, we use $K = 4$ independently trained models in the experiments to balance visualization clarity and trigger robustness. In practice, larger ensembles can improve reliability at higher computational cost.

![](images/797883e9894600b55b84f611a82d54d247acc75f90f56e06e096f1eada0623ae.jpg)  
Figure 2: Ensemble results (4 independent runs) for the toy inverse problem. Top: posterior samples with true solution (solid circle) and marginal distributions. Bottom: pairwise Wasserstein distance matrices between runs (lower values indicate better agreement). (a) In-domain: consistent predictions with low pairwise Wasserstein distances $( \bar { W } = 0 . 1 2 )$ . (b) Out-of-domain: inconsistent predictions with high pairwise Wasserstein distances $( \bar { W } = 3 . \bar { 0 } 0 )$ . (c) After active learning: predictions converge with low pairwise Wasserstein distances $( \bar { W } \overset { ^ { \mathrm { ~ \textstyle ~ } } } { = } 0 . 1 9 )$ .

## 4.2 Parameterization of Particle Momentum Distribution

We use a simulated inclusive Deep Inelastic Scattering (DIS) setup, in which the Particle Momentum Distribution (PMD) is parameterized as a quark correlation function (QCF), to demonstrate the effectiveness of the active diffusion-based inverse solver in a practical application. In inclusive DIS experiments, high-energy leptons scatter off nuclear targets, producing complex final-state particle showers. By measuring the kinematics of the scattered lepton, one can extract information about the internal quark and gluon structure of the target nucleus.

The PMD encodes the probability density of observing particles with specific momenta following the lepton–nucleus interaction. As such, it provides an indirect representation of the underlying QCFs that characterize the momentum and spatial correlations of quarks and gluons. Inferring QCFs from measured PMDs constitutes a challenging inverse problem that is commonly addressed using Bayesian inference methods with physically motivated priors [Bishop and Nasrabadi, 2006].

The dimensionality of the PMD is directly determined by that of the underlying three-dimensional QCFs, which in principle requires at least three final-state particle measurements to fully constrain the distribution. Traditional histogrambased reconstruction methods are widely used to estimate PMDs, but these approaches become increasingly inadequate in higher dimensions, where they tend to obscure important multi-particle correlations. To illustrate the key ideas of our framework in a controlled setting, we therefore perform a simplified proxy calculation that captures essential features of realistic QCD-based models used in global analyses [Cocuzza et al., 2022].

The PMDs for a simplified version of DIS on protons and neutrons are given by

$$
\begin{array} { r } { \pmb { \sigma } _ { 1 } ( x ; p ) = 4 u ( x ; p ) + d ( x ; p ) , } \\ { \pmb { \sigma } _ { 2 } ( x ; p ) = u ( x ; p ) + 4 d ( x ; p ) . } \end{array}\tag{2}
$$

where $\pmb { \sigma } _ { 1 }$ and $\pmb { \sigma } _ { 2 }$ are cross sections (un-normalized probability distributions) and $u ( x )$ and $d ( x )$ are the universal 1D QCFs called up- and down-quark PDFs, respectively, which are weighted by their charge squared in the PMDs. QCFs for the proxy problem with two channels are defined by:

$$
\begin{array} { c } { { u ( x ; p ) = N _ { u } x ^ { a _ { u } } ( 1 - x ) ^ { b _ { u } } } } \\ { { d ( x ; p ) = N _ { d } x ^ { a _ { d } } ( 1 - x ) ^ { b _ { d } } } } \end{array}\tag{3}
$$

where $x \in \mathsf { \Gamma } ( 0 , 1 )$ and $\{ N _ { u } , a _ { u } , b _ { u } , N _ { d } , a _ { d } , b _ { d } \}$ is the unknown parameter vector to be determined. We collect events $\{ \sigma _ { p } ^ { o } , \sigma _ { n } ^ { o } \}$ generated by model (3) and filtered through crosssections defined in (2) for specific ranges of the shape parameters $\{ N _ { u } , a _ { u } , b _ { u } , N _ { d } , a _ { d } , b _ { d } \}$ . From these QCFs, we create PMDs and then sample the PMDs to generate the physics events as the observables.

## 4.3 QCD Parameter Inference Results

We implement the active diffusion-based inverse solver using a PointNet-Transformer architecture [Qi et al., 2017;

![](images/3e66aaf4de36ab200fe39f5bc61a086dd972215191919af62b9d6e7bd319cc63.jpg)  
Figure 3: In-domain inference with true parameters in the training domain. The model accurately recovers all parameters with low uncertainty. The shaded orange region denotes the prior support.

Vaswani et al., 2017] as the backbone network, which processes event-level observables through permutation-invariant encoding followed by self-attention layers. The model is trained using an Adam optimizer with learning rate $1 0 ^ { - 4 }$ and batch size 64, where each parameter sample is conditioned on 10,000 events generated through the forward model described by Equations (2) and (3). The diffusion process uses $T = 1 0 0$ timesteps with a linear noise schedule from $\beta _ { 1 } ~ = ~ 1 0 ^ { - 4 }$ to $\beta _ { T } = 0 . 0 2$ . The initial parameter bounds are $N _ { u } , N _ { d } \in [ 0 , 1 ]$ $a _ { u } \in [ - 1 , 0 ]$ , and $a _ { d } , b _ { u } , b _ { d } \in [ 0 , 1 ]$

In-domain Inference. When ground truth parameters are sampled from the interior of the prior distribution, the trained DM predicts all six QCD parameters with high fidelity (Figure 3). The posteriors concentrate tightly around true values with small standard deviations ranging from 0.03 to 0.06, confirming reliable inference in the training support.

Out-of-domain Inference without Active Learning. We construct a challenging out-of-domain test case in which all true parameters lie outside the training bounds: $N _ { u } = 2 . 0$ $a _ { u } = 0 . 6 , b _ { u } = - 0 . 5 , N _ { d } = 1 . 5 , a _ { d } = - 0 . 4 ,$ , and $b _ { d } = 1 . 8$ An ensemble of four independently initialized DMs produces the failed inference results shown in Figure 4. For each model in the ensemble, the inferred posterior distributions exhibit substantially larger standard deviations than those observed in in-domain inference, indicating increased uncertainty and weakened posterior concentration.

![](images/e1cea8b82c49f0cc917be4467a265a0ddb25ead9e2a347995064fce2c51c13d3.jpg)  
Figure 4: Out-of-domain inference without active learning. Four independently trained models (shown in different colors) produce inconsistent predictions with high ensemble disagreement $( \sigma _ { e n s } )$ indicating model misspecification.

More importantly, the four independently trained DMs yield inconsistent posterior estimates. The ensemble disagreement $\sigma _ { \mathrm { e n s } } .$ , defined as the standard deviation of posterior means across models, is significantly elevated for these parameters, ranging from 0.04 to 0.18. The lack of consensus reveals that the predictions are not supported by the training data distribution, serving as a clear and practical indicator of out-of-domain extrapolation. Overall, these results demonstrate that ensemble disagreement provides a practical and reliable signal for detecting model misspecification.

Out-of-domain Inference with Active Learning. We apply the active diffusion-based inference to the same out-ofdomain test case by iteratively expanding the parameter domain through active learning. At each iteration, the DM generates candidate parameter samples via reverse diffusion conditioned on the observed events. These event samples identify regions of parameter space associated with high posterior uncertainty or ensemble disagreement. Additional training data are then generated in these regions using the forward model by Equations (2) and (3), and the DM is refined on the augmented training set. This process is repeated until convergence is reached.

Figure 5 summarizes the inference results after convergence of the active learning loop. The posterior means are estimated as $\hat { N } _ { u } = 1 . 9 2 , \hat { a _ { u } } = 0 . 6 5 , \hat { b _ { u } } = - 0 . 5 3 , \hat { N _ { d } } = 1 . 4 1$ $\hat { a _ { d } } = - 0 . 4 2$ , and $\hat { b _ { d } } = 1 . 8 0$ . The inferred posterior distributions accurately recover the ground-truth parameters, with all ground-truth values lying within three standard deviations of the corresponding posterior means. Moreover, posterior uncertainties contract to levels comparable to in-domain inference $( \hat { \sigma } \approx 0 . 0 3 – 0 . 0 5 )$ , despite all inferred posteriors lying entirely outside the original training support. These results demonstrate that the active diffusion framework can reliably detect and correct out-of-domain extrapolation, autonomously discovering relevant parameter regions and enabling accurate inverse inference without prior knowledge of the valid parameter domains.

![](images/3c65e9bd2f22d60655be8cf48cf74b99aa24dc3ecf208a71e317166d6998b34a.jpg)  
Figure 5: Out-of-domain inference with active learning. After adaptive domain expansion, the model successfully recovers all parameters, where all ground-truth values are within three standard deviations of the corresponding posterior means.

## 5 Discussion

## 5.1 Relation to Shooting Method

Our method is conceptually close to the classical shooting method in classical numerical analysis [Stoer et al., 1980]. In shooting methods for boundary-value problems, the initial conditions, typically treated as unknown parameters, are iteratively adjusted so that the forward solution of a differential equation can gradually satisfy the boundary condition, a prescribed terminal constraint. Each iteration includes executing the forward model, evaluating the mismatch at the boundary, and updating the parameter estimate accordingly. It is important to note that the shooting method does not require prior knowledge of the domain of the parameter values; instead, it progressively refines them based on the residual error as feedback from the forward model.

<table><tr><td>Method</td><td>In-domain</td><td>OOD fixed</td><td>OOD +AL</td></tr><tr><td>MDN</td><td>0.067</td><td>2.497</td><td>0.082</td></tr><tr><td>INN</td><td>0.003</td><td>1.679</td><td>0.025</td></tr><tr><td>NF</td><td>0.045</td><td>107.243</td><td>0.079</td></tr><tr><td>DM</td><td>0.015</td><td>1.279</td><td>0.032</td></tr></table>

Table 1: Comparison of various generative models on the toy circle inverse problem.

In the inverse problem, the unknown parameters play a similar role as the unknown initial conditions in the shooting methods. The DM serves as a surrogate inverse mapper, proposing candidate parameter solutions that are consistent with the observables. When the true parameters lie outside the initial training domain, the diffusion posterior exhibits large uncertainty, indicating a mismatch with respect to the observables. The active learning loop can therefore be interpreted as a probabilistic generalization of the shooting methods. Instead of updating the estimate of a single parameter, the support of the parameter distribution is updated by selectively expanding the training domain in new regions suggested by the DM. The forward model is then carried out on these candidate parameters, providing new parameterobservable mapping information that further improves the inverse surrogate.

Analogue to the shooting methods, the evaluations of the forward models are often computationally costly and need to be chosen smartly to maximize information gain. A key distinction between this method and the shooting methods is that shooting methods use deterministic residuals to guide parameter updates, whereas the DM provides a full posterior distribution over parameters conditioned on the observables. From this point of view, the active learning DM may be considered as a shooting method in distribution space. The iterative expansion of the parameter domain replaces the iterative correction of initial conditions, and the convergence is reached when the posterior distribution contracts to a stable region consistent with the observables.

## 5.2 Ensemble-based Analysis

The ensemble disagreement is similar to residual sensitivity in shooting methods. Different initial guesses leading to divergent terminal behavior indicate that either the boundary value problem is ill-conditioned or no valid solutions in the current search region. Similarly, disagreement among trained DMs ensemble suggests that the parameter domain lacks sufficient coverage. Using the measure of ensemble disagreement to trigger active learning allows the algorithm to identify the regions of parameter space where additional forwardmodel evaluations are most likely informative toward the true parameters.

## 5.3 Comparison with Other Generative Models

In addition to DM, other generative AI models can also be incorporated into the active learning inverse inference framework under incomplete priors. For completeness, we compare against other learned inverse solvers, including MDN, INN, and RealNVP-style NF, on the same toy problem specified in Section 4.1. Table 1 reports the mean radial error (MRE), where all methods are trained $\mathrm { o n } ~ z ~ \in ~ [ 0 , 4 ]$ and tested in-domain at $z ^ { * } = 4$ and out-of-domain (OOD) at $z ^ { * } = 2 5$ One can find that all fixed-domain methods perform well within the training domain but degrade substantially under OOD conditions, as reflected by large MRE values. When augmented with the proposed active learning framework $( ^ { 6 6 } { + } \mathrm { { A L } } ^ { 9 } )$ , all methods recover near in-domain accuracy. This result demonstrates that the proposed framework is model-agnostic and can be effectively integrated with a broad class of generative models for inverse problems, rather than being specific to DMs.

## 5.4 Computational Cost and Practical Considerations

The main computational cost of the active framework comes from additional forward-model evaluations in expanded parameter regions and from retraining or fine-tuning the generative model. Particularly, in the QCD proxy experiment, each training step requires approximately 2.6 seconds for forward simulation with 64 parameter samples and 10,000 events per sample, while the DM training update takes approximately 0.03 seconds per step. More generally, the overall computational efficiency is inherently application-dependent. It is primarily governed by the cost of the forward simulation, the distance from the true parameters to the initially covered training domain, the dimensionality of the parameter space, and the geometric complexity of the parameter landscape.

## 6 Conclusion

In this work, we introduce an active diffusion-based framework for solving inverse problems under incomplete prior knowledge of the parameter domain. By learning the probabilistic mapping between parameters and observables, the DM achieves inverse inference as conditional generation, addressing nonlinearity and ill-posedness in inverse problems. Without relying on fixed parameter bounds, the active learning loop enables adaptively discovering and refining the relevant regions of parameter space, ensuring reliable inference even when the true parameters are not included in the initial training domain. The active diffusion-based inference is validated using a toy inverse problem with infinitely many solutions and a QCF parameterization problem in Quantum Chromodynamics analysis of nucleon structure. The posterior distributions are correctly recovered and the out-of-domain failures are effectively corrected through active retraining. It is important to note that when a sufficiently large valid parameter domain is known in advance and sampling this domain is computationally affordable, fixed-domain training is sufficient. Our setting addresses the complementary case in which the valid parameter domain is unknown, effectively unbounded, or too high-dimensional to cover uniformly.

Several promising research directions emerge from this work. In the future, we will focus on scaling the active diffusion-based model to higher dimensional parameter spaces and more sophisticated forward models. In particular, we will investigate surrogate approximated forward modeling and physics-informed constraints to achieve better efficiency and stability. Moreover, we will apply our methods to real experimental data where data uncertainties, systematic uncertainties, and model uncertainties coexist. This will be a critical step toward building active generative inference as a practical tool with uncertainty quantification for scientific discovery in large-scale inverse problems.

## Acknowledgments

This work is partially supported by the U.S. Department of Energy, Office of Science, Office of Nuclear Physics, Office of Advanced Scientific Computing Research through the Scientific Discovery through Advanced Computing (SciDAC) program, under contracts DE-AC02-06CH11357, DE-AC05- 06OR23177, and DE-SC0023472, for the award Femtoscale Imaging ofNuclei using Exascale Platforms, the EXCLAIM collaboration under the DOE grants DE-SC0016286 and DE-SC0024644, and by the Center for Nuclear Femtography (CNF), administrated by the Southeastern Universities Research Association under an appropriation from the Commonwealth of Virginia under contract No. C2024-FEMT-011-02.

## References

[Adler and Oktem, 2017<sup>¨</sup> ] Jonas Adler and Ozan Oktem.<sup>¨</sup> Solving ill-posed inverse problems using iterative deep neural networks. Inverse Problems, 33(12):124007, 2017.

[Alghamdi et al., 2025] Tareq Alghamdi, Jitao Xu, Nesar Ramachandra, Nobuo Sato, and Yaohang Li. Towards an event-level analysis in hadronic physics using generative ai-based surrogates. In Proceedings of The IEEE International Conference on Tools with Artificial Intelligence (ICTAI2025), 2025.

[Almaeen et al., 2021] Manal Almaeen, Yasir Alanazi, Nobuo Sato, W. Melnitchouk, Michelle P. Kuchera, and Yaohang Li. Variational autoencoder inverse mapper: An end-to-end deep learning framework for inverse problems. In Proceedings of2021 International Joint Conference on Neural Networks (IJCNN), pages 1–8, 2021.

[Almaeen et al., 2025] Manal Almaeen, Tareq Alghamdi, Brandon Kriesten, Douglas Adams, Yaohang Li, Huey-Wen Lin, and Simonetta Liuti. Vaim-cff: a variational autoencoder inverse mapper solution to compton form factor extraction from deeply virtual exclusive reactions. Eur. Phys. J. C, 85:449, 2025.

[Ardizzone et al., 2019] Lynton Ardizzone, Jakob Kruse, Carsten Rother, and Ullrich Kothe. Analyzing inverse ¨ problems with invertible neural networks. In Proceedings of 7th International Conference on Learning Representations, 2019.

[Arridge et al., 2019] Simon Arridge, Peter Maass, Ozan Oktem, and Carola-Bibiane Sch <sup>¨</sup> onlieb. Solving inverse¨ problems using data-driven models. Acta Numerica, 28:1–174, 2019.

[Bishop and Nasrabadi, 2006] Christopher M Bishop and Nasser M Nasrabadi. Pattern recognition and machine learning, volume 4. Springer, 2006.

[Bishop, 1994] Christopher M. Bishop. Mixture density networks. Working paper, Aston Univ., 1994.

[Chung et al., 2022] Hyungjin Chung, Jeongsol Kim, Michael T Mccann, Marc L Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. arXiv preprint arXiv:2209.14687, 2022.

[Cocuzza et al., 2022] Christopher Cocuzza, Wally Melnitchouk, Andreas Metz, Nobuo Sato, and (Jefferson Lab Angular Momentum (JAM) Collaboration). Polarized antimatter in the proton from a global qcd analysis. Physical Review D, 106(3):L031502, 2022.

[Colton et al., 2000] David Colton, Heinz W. Engl, Alfred K. Louis, Joyce R. McLaughlin, and William Rundell. Surveys on Solution Methods for Inverse Problems. Springer, 2000.

[Dinh et al., 2014] Laurent Dinh, David Krueger, and Yoshua Bengio. Nice: Non-linear independent components estimation. arXiv preprint arXiv:1410.8516, 2014.

[Foucart and Rauhut, 2013] Simon Foucart and Holger Rauhut. A Mathematical Introduction to Compressive Sensing. Birkhauser, 2013.¨

[Ho et al., 2020] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[Kawar et al., 2022] Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising diffusion restoration models. Advances in neural information processing systems, 35:23593–23606, 2022.

[Kingma and Ba, 2017] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2017.

[Kruse et al., 2025] Carola Kruse, Masha Sosonkina, Md Fayaz Bin Hossen, and Yaohang Li. Material parameter estimation for a viscoelastic stenosis model using a variational autoencoder inverse mapper. Journal of Inverse and Ill-posed Problems, 33(5):617–632, 2025.

[Li et al., 2020] Housen Li, Johannes Schwab, Stephan Antholzer, and Markus Haltmeier. NETT: Solving inverse problems with deep neural networks. Inverse Problems, 36(6):065005, 2020.

[Nalisnick et al., 2018] Eric Nalisnick, Akihiro Matsukawa, Yee Whye Teh, Dilan Gorur, and Balaji Lakshminarayanan. Do deep generative models know what they don’t know? arXiv preprint arXiv:1810.09136, 2018.

[Qi et al., 2017] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 652–660, 2017.

[Rezende and Mohamed, 2015] Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows.

In International conference on machine learning, pages 1530–1538. PMLR, 2015.

[Rodgers, 2000] Clive D Rodgers. Inverse methods for atmospheric sounding: theory and practice, volume 2. World scientific, 2000.

[Settles, 2009] Burr Settles. Active learning literature survey. 2009.

[Sohl-Dickstein et al., 2015] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. pmlr, 2015.

[Song and Ermon, 2019] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019.

[Song et al., 2020] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

[Spaeth and Li, 2025] Peter W. Spaeth and Yaohang Li. Estimate composite bond line properties in composite structures using variational autoencoder inverse mapper. In Giovanni Ferrarini, Peter Spaeth, and Fernando Lopez, ed-´ itors, Thermosense: Thermal Infrared Applications XLVII, volume 13470, page 134700Q. International Society for Optics and Photonics, SPIE, 2025.

[Stoer et al., 1980] Josef Stoer, Roland Bulirsch, R Bartels, Walter Gautschi, and Christoph Witzgall. Introduction to numerical analysis, volume 1993. Springer, 1980.

[Tarantola, 2005] Albert Tarantola. Inverse problem theory and methods for model parameter estimation. SIAM, 2005.

[Vaswani et al., 2017] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017.

[Yu et al., 2017] Wenjian Yu, Yu Gu, Jian Li, Shenghua Liu, and Yaohang Li. Single-pass PCA of large highdimensional data. In Proceedings of 26th International Joint Conference on Artificial Intelligence, (IJCAI), 2017.

[Yuan et al., 2020] Qiangqiang Yuan, Huanfeng Shen, Tongwen Li, Zhiwei Li, Shuwen Li, Yun Jiang, Hongzhang Xu, Weiwei Tan, Qianqian Yang, Jiwen Wang, et al. Deep learning in environmental remote sensing: Achievements and challenges. Remote sensing of Environment, 241:111716, 2020.

[Zhang et al., 2021] Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning (still) requires rethinking generalization. Communications ofthe ACM, 64(3):107–115, 2021.