# Forward-Value Discretization: Faster Stochastic Difusion Sampling Without High-Order Approximation

Yuchen Jiao<sup>∗†</sup> Na Li<sup>∗†</sup> Changxiao Cai<sup>‡</sup> Yuting Wei <sup>§</sup> Gen Li<sup>†</sup>

November 11, 2026

## Abstract

Stochastic difusion samplers often require substantially more neural function evaluations (NFEs) than deterministic samplers to achieve comparable sample quality. Although higher-order discretization accelerates deterministic sampling, its benefits for stochastic sampling remains unclear. We establish matching upper and worst-case lower bounds showing that stochastic samplers with higher-order approximations remain first-order convergent, even with exact score functions. Thus, increasing the approximation order does not generally improve the convergence order. Motivated by this limitation, we introduce forward-value discretization, a training-free framework that evaluates the data/noise prediction model at a lookahead estimate constructed from previous predictions. For deterministic sampling, the resulting Forward DPMSolver (F-DPMSolver), improves sample quality at low NFE budgets through its DDIM-based variant, despite remaining first-order convergent. For stochastic sampling, we develop Forward DDPM (F-DDPM), which evaluates the model at a cleaner lookahead point adjusted for subsequent noise injection. This estimate reuses both the previous model prediction and the previously injected noise. Both variants admit first-order convergence guarantees with one model evaluation per iteration. Across CIFAR-10, ImageNet, FFHQ, and LSUN, F-DDPM consistently achieves lower FID than the evaluated stochastic baselines at matched NFE budgets. With 1.5 times as many NFEs as deterministic baselines, it achieves competitive sample quality, narrowing the practical eficiency gap between stochastic and deterministic sampling. Code is available at https://github.com/Na-Li66/F-DDPM.

Keywords: difusion model, training-free acceleration, first-order sampler, forward-value discretization, stochastic sampler

## Contents

1 Introduction 2   
1.1 Our contributions . . 4   
1.2 Other related work . 4   
2 Preliminaries 5   
2.1 Difusion processes and parameterization . . 5   
2.2 Difusion ODE and deterministic samplers 6   
2.3 Difusion SDE and stochastic samplers 7   
3 Main results 7   
3.1 Forward-value discretization 8   
3.2 F-DDPM: first-order forward-value stochastic sampler 10   
3.3 Convergence analysis . 12   
4 Experiments 14   
4.1 Evaluation of F-DDIM 15   
4.2 Evaluation of F-DDPM 16   
5 Discussion 18   
A Additional sampler details and experimental results 23   
A.1 Baseline deterministic samplers 23   
A.2 Additional results for F-DDIM 24   
A.3 Higher-order lookahead predictors 29   
A.4 Larger NFE budgets and runtime comparisons 30   
A.5 Additional samples from F-DDPM 31   
B Proof of deterministic samplers convergence lower bound (Theorem 1) 34   
C Proof of stochastic samplers convergence upper bounds (Theorem 2) 42   
C.1 Convergence analysis of SDE-DPMSolver++1 43   
C.2 Convergence analysis of SDE-DPMSolver++2 50   
D Proof of stochastic samplers convergence lower bounds (Theorem 3) 55   
D.1 Convergence analysis of SDE-DPMSolver++1 57   
D.2 Convergence analysis of SDE-DPMSolver++2 58   
E Proof of convergence order upper bound for F-DDIM (Theorem 4) 62   
E.1 Proof of Theorem 6 . 65   
E.2 Proof of Theorem 7 69   
F Proof of convergence order upper bound for F-DDPM (Theorem 5) 72

## 1 Introduction

Difusion probabilistic models (DPMs) have rapidly emerged as a leading paradigm in modern generative modeling, achieving state-of-the-art performance across a wide range applications in generative AI (Ho et al., 2020; Song et al., 2020b; Song and Ermon, 2019; Song et al., 2020a; Dhariwal and Nichol, 2021). Building on ideas from non-equilibrium thermodynamics (Sohl-Dickstein et al., 2015), a DPM first defines a forward process that gradually injects noise into the clean data drawn from the true data distribution, and then generates new samples by reversing this process, turning pure noise into high-fidelity outputs. The reversal is driven by a pre-trained neural network that approximate the score functions of the noisy data distributions. DPMs now have shown remarkable success across various domains, including image synthesis (Rombach et al., 2022; Ramesh et al., 2022; Saharia et al., 2022), audio generation (Kong et al., 2021), language models (Li et al., 2022; Sahoo et al., 2024; Li et al., 2026), and molecular design (Hoogeboom et al., 2022), underscoring their versatility and impact. See Yang et al. (2023); Croitoru et al. (2023); La et al. (2025) for overviews of recent development.

Once a DPM is trained, sampling amounts to discretizing a diferential equation that describes the reverse process: either the difusion SDE (Ho et al., 2020; Nichol and Dhariwal, 2021; Bao et al., 2022) or the difusion ODE (Song et al., 2020a; Lu et al., 2022a). In practice, a DPM is typically trained as a noise prediction model: given a noisy sample and a time step, it predicts the noise added to the original clean data at that time in the forward process. Since subtracting this noise prediction from the noisy sample yields an estimate of the clean data, the same network equivalently acts as a clean-data predictor. As a result, sample quality relies on two factors: how the difusion SDE/ODE is discretized, and how the pre-trained DPM is queried along the way.

NFE=5  
10  
15  
20  
![](images/cd45c0c96076b6af8490b01b3be58c80929420cd2723243ff7aaa1f42367f859.jpg)  
(a) A cat in a suitcase.  
25

NFE=5  
10  
15  
20  
25  
![](images/2183173a2452e4d63330546640a942672c5dbabde046b349adc6c8d68fda5e8b.jpg)  
(b) A rodeo scene.  
Figure 1: Stable Difusion samples at diferent NFEs for the cat and rodeo prompts described in the text. In these examples, SDE-DPMSolver++1 and SDE-DPMSolver++2 depict the cat and human figures more clearly than DDIM and DPMSolver-3.

Difusion samplers can be broadly classified into two categories, depending on whether noise is injected beyond the initial step. Stochastic samplers (e.g., DDPM (Ho et al., 2020)) add Gaussian noise with during sampling iterations, while deterministic samplers (e.g., DDIM (Song et al., 2020a)) add none beyond initialization. Stochastic samplers tend to produce more diverse samples and give a more natural framework for classifier guidance, which often improves the quality of conditional samples. For example, Figure 1 compares Stable Difusion samples generated with the prompts (a) “A hand pets a cat in a suitcase” and (b) “A rodeo area with men in cowboy hats, rodeo clowns and an audience in the stands”. For the first prompt, the stochastic samplers SDE-DPMSolver++1 and SDE-DPMSolver++2 (see (17a) and (17b) for their update rules) produce clearly recognizable cats, whereas the deterministic samplers DDIM and DPMSolver-3 (see (32) for the latter’s update rule) primarily depict the suitcase. For the second prompt, the stochastic samplers produce cowboy hats and audience in stands. The corresponding details are less clearly resolved in the deterministic samples. Nevertheless, these benefits come at a price. Empirically, stochastic samplers converge far more slowly than their deterministic counterparts. For instance, On CIFAR-10 with the EDM model, DDPM needs roughly 40 iterations to achieve lower than 3 in Fréchet inception distance (FID) (Heusel et al., 2017), a metric used to assess the quality of images, whereas DDIM achieves comparable quality in fewer than 15 iterations. Since each iteration costs one neural function evaluation of the pre-trained DPM and sampling speed is often the primary bottleneck in deployment, this gap has driven growing interest in deterministic samplers.

Higher-order discretization is a widely used training-free approach to accelerating deterministic difusion sampling (Lu et al., 2022a,b; Zhao et al., 2023). For stochastic samplers, however, increasing the order of a deterministic approximation does not automatically improve the strong convergence order to the underlying difusion SDE. Strong convergence measures the discrepancy between the numerical and continuous trajectories driven by the same Brownian motion, whereas distributional convergence measures the discrep ancy between their laws or between the generated and target distributions. In particular, we show that SDE-DPM-Solver++2 remains first-order convergent despite its higher-order approximation.

This gap motivates the central question of this work:

Can we design a fast stochastic sampler that narrows the performance gap with deterministic samplers?

## 1.1 Our contributions

We begin by analyzing the convergence rates of existing deterministic and stochastic samplers, including DDIM, DPMSolver(Lu et al., 2022a), and SDE-DPMSolver++(Lu et al., 2022a). For deterministic samplers, prior works have established that a higher-order approximation achieves a corresponding higher-order convergence rate. For instance, DPMSolver-p converges at a rate of $\widetilde { \cal O } ( M ^ { - p } )$ , where M denotes the number of iterations. This work reveals that this correspondence between approximation order and convergence order does not hold for the stochastic SDE-DPM-Solver++ schemes. In particular, SDE-DPM-Solver++2 remains first-order convergent in the strong sense, as established by matching upper and worst-case lower bounds. The reason is that the injected noise makes the generated samples vary too much at each step, for a higher-order expansion to help. This ofers one explanation for the practical ineficiency of stochastic samplers.

Next, we explore a simple but under-explored idea: a forward-value discretization of the difusion ODE. Given the current iterate and time, the idealized forward-value update would evaluate the DPM at the quantities corresponding the next time rather than the current one. We observe that such an update scheme can perform efectively even in the extremely low-NFE regime, which motivates a practical sampler that approximates it. The resulting sampler, named Forward DPMSolver (F-DPMSolver), first produces a rough estimate of the next iterate using a vanilla update (e.g., DDIM) with cached model predictions, and then combines the current iterate with a DPM evaluation at this estimate. Critically, it maintains one model evaluation per iteration by reusing the previous evaluation, so its number of model evaluations (NFE) is the same as that of the vanilla sampler. Empirical results show that F-DPMSolver improves FID substantially over higher-order samplers including DPMSolver-2, DPMSolver-3 (Lu et al., 2022b), and UniPC-3 (Zhao et al., 2023), even though its convergence rate still scales as $\widetilde { \Theta } ( 1 / M )$ . These results convey a surprising insight: a first-order sampler can also achieve high practical sampling eficiency.

Inspired by this insight, we extend the forward-value idea to stochastic samplers and propose a fast firstorder stochastic sampler, termed Forward DDPM (F-DDPM). Whereas the deterministic sampler evaluates the DPM at a lookahead estimate for the next time step, the stochastic sampler uses an estimate at a carefully chosen cleaner intermediate time to account for subsequent noise injection. To construct this estimate, we form a tailored noise estimate that combines the available noise prediction with the injected noise from the previous step. We also establish a first-order strong error upper bound for F-DDPM. Empirically, F-DDPM consistently improves FID over the evaluated stochastic baselines at the same NFEs and substantially narrows the performance gap with deterministic samplers. With 1.5 times as many NFEs, it achieves sample quality comparable to or better than the evaluated deterministic baselines in many tested settings.

## 1.2 Other related work

Training-free/based acceleration. Existing acceleration strategies for difusion sampling can be broadly divided into training-free and training-based approaches. Training-free methods reuse a fixed pre-trained DPM and speed up sampling solely by modifying the sampling update rule, making them broadly applicable to of-the-shelf DPMs. The methods discussed in the introduction belong to this category. In contrast, training-based strategies introduce an additional stage of training (e.g., distillation (Luhman and Luhman, 2021; Salimans and Ho, 2022; Meng et al., 2023; Zhang et al., 2026) and consistency models (Li et al., 2024b; Song et al., 2023)). They aim to shorten sampling trajectories by adapting pre-trained models into related architectures, at the cost of additional training computation. Since our focus is training-free acceleration, we next summarize related work from both practical and theoretical viewpoints.

Training-free acceleration: practice. Most practical training-free accelerations are driven by higherorder numerical discretizations of the reverse-time SDE/ODE. For ODE-based samplers, a prominent line of work exploits the structure of the difusion ODE to design higher-order solvers, including DPM-Solver++ (Lu et al., 2022a,b), exponential-integrator-based methods (Zhang and Chen, 2022), and predictor-correctorbased schemes such as UniPC (Zhao et al., 2023). For SDE-based samplers, acceleration is comparatively less explored due to the intrinsic dificulty of SDE discretization, but notable progress includes stochastic Improved Euler’s methods (Jolicoeur-Martineau et al., 2021), stochastic Adams methods (Xue et al., 2024), and stochastic Runge-Kutta methods (Wu et al., 2024).

Training-free acceleration: theory. Convergence theory has been established for a wide range of diffusion samplers (Chen et al., 2022; Lee et al., 2023; Chen et al., 2023a; Li et al., 2023; Chen et al., 2023b; Huang et al., 2024a; Benton et al., 2023; Li and Yan, 2024; Li et al., 2025a; Cai et al., 2026; Jiao et al., 2025; Zhang et al., 2025), mostly focusing on standard DDPM and DDIM. For provable acceleration, most existing results likewise concentrate on higher-order discretization and on controlling the resulting discretization error for the reverse-time dynamics (Li and Cai, 2024; Li and Jiao, 2024; Jiao and Li, 2024; Huang et al., 2024a,b; Li et al., 2024a; Yu and $\mathrm { Y u . }$ , 2025; Li et al., 2025b; Gupta et al., 2024; Wu et al., 2024). Beyond designing improved solvers, another line of work accelerates generation via parallel sampling that implements multiple denoising steps in parallel (Shih et al., 2023; Chen et al., 2024; Gupta et al., 2024). In addition to continuous difusion models, sampling acceleration has also been studied for discrete difusion models (Li and Cai, 2025; Chen et al., 2025; Dmitriev et al., 2026a; Zhao and Cai, 2026; Cai and Li, 2026; Dmitriev et al., 2026b; Chen and Wang, 2026).

Notation. For integer $n \geq 1$ , we denote $[ n ] : = \{ 1 , 2 , . . . , n \}$ . For vector x, we denote by $\| \pmb { x } \| _ { 2 }$ or $\lVert \boldsymbol { x } \rVert$ its $\ell _ { 2 }$ norm. For matrix $A _ { ; }$ , we denote by $\| A \| \ \mathrm { o r } \ \| A \| _ { \mathsf { o p } }$ its operator norm. In the rest of the paper, we use $\lambda ( t )$ or $\lambda _ { t }$ interchangeably to denote the mapping from the parameter t to λ. For random vector $\mathbf { \Delta } \mathbf { x } _ { t }$ defined in (1), we let $q _ { t }$ denote its probability density function. For function $f ( M )$ , we write $f ( M ) = O ( M ^ { - k } )$ if there exists some constant $C > 0$ such that $| f ( M ) | \leq C M ^ { - k }$ for all $M \geq 1$ . Similarly, we denote $f ( \dot { M } ) = \operatorname { \bar { \Omega } } ( M ^ { - k } )$ if $| f ( M ) | \ge c M ^ { - k }$ holds for some constant $c > 0$ , and $f ( M ) = \Theta ( M ^ { - k } )$ if both $f ( M ) = O ( M ^ { - k } )$ and $f ( M ) = \Omega ( M ^ { - k } )$ hold. Finally, we say $f ( M ) = o ( M ^ { - k } )$ if lim $\mathsf { \Omega } ^ { \mathsf { l } } M \to \infty  M ^ { k } f ( M ) = 0$

## 2 Preliminaries

In this section, we review the basics of DPM sampling.

## 2.1 Difusion processes and parameterization

Forward and reverse processes. We consider a forward process $( \pmb { x } _ { t } ) _ { 0 \leq t \leq T }$ in $\mathbb { R } ^ { d }$ whose marginal distri bution at time $t \in [ 0 , T ]$ satisfies

$$
{ \pmb x } _ { t } \overset { \mathrm { d } } { = } \alpha _ { t } { \pmb x } _ { 0 } + \sigma _ { t } { \pmb z } ,\tag{1}
$$

where $\mathbf { \boldsymbol { x } } _ { 0 } \sim \mathbf { \boldsymbol { q } } _ { 0 }$ is independent of the Gaussian noise $\boldsymbol { z } \sim \mathcal { N } ( \mathbf { 0 } , I _ { d } )$ . We denote the distribution of ${ \mathbf { } } _ { \mathbf { \lambda } } \mathbf { x } _ { t } ,$ or its density, by $q _ { t } .$ . The noise schedules $\alpha _ { t }$ and $\sigma _ { t }$ are positive for $t \in ( 0 , T )$ and are chosen so that the signal-to-noise (SNR) ratio $\alpha _ { t } / \sigma _ { t }$ decreases strictly with t and the terminal distribution is approximately Gaussian: $q _ { T } \approx \mathcal { N } ( 0 , \tilde { \sigma } ^ { 2 } I _ { d } )$

These forward marginals can be obtained from the SDE

$$
\mathrm { d } \pmb { x } _ { t } = f ( t ) \pmb { x } _ { t } \mathrm { d } t + g ( t ) \mathrm { d } \pmb { \omega } _ { t } , \quad t \in [ 0 , T ] ; \qquad \pmb { x } _ { 0 } \sim q _ { 0 } ,\tag{2}
$$

where $( \omega _ { t } ) _ { t \geq 0 }$ denotes a standard d-dimensional Brownian motion. The coeficients are determined by the noise schedules as follows:

$$
f ( t ) = \frac { \mathrm { d } \log \alpha _ { t } } { \mathrm { d } t } \quad \mathrm { a n d } \quad g ^ { 2 } ( t ) = \frac { \mathrm { d } \sigma _ { t } ^ { 2 } } { \mathrm { d } t } - 2 \sigma _ { t } ^ { 2 } \frac { \mathrm { d } \log \alpha _ { t } } { \mathrm { d } t } .
$$

For further details on this formulation, we refer to Kingma et al. (2021).

The time-reversal result of Anderson (1982) gives the reverse SDE associated with (2), which is solved from $T$ to 0:

$$
\mathrm { d } x _ { t } = \left[ f ( t ) x _ { t } - g ^ { 2 } ( t ) s _ { t } ^ { \star } ( x _ { t } ) \right] \mathrm { d } t + g ( t ) \mathrm { d } \widetilde { \omega } _ { t } , \quad t \in [ 0 , T ] ; \qquad x _ { T } \sim q _ { T } ,\tag{3}
$$

where $( \widetilde { \omega } _ { t } ) _ { t \in [ 0 , T ] }$ denotes standard Brownian motion running backward from $T$ to 0, and $\pmb { s } _ { t } ^ { \star } ( \cdot ) : = \nabla \log q _ { t } ( \cdot )$ is the score of the time-t marginal. Sampling can alternatively be based on a deterministic process governed by the probability flow ODE:

$$
\dot { { \pmb x } } _ { t } = f ( t ) { \pmb x } _ { t } - \frac { 1 } { 2 } g ^ { 2 } ( t ) { \pmb s } _ { t } ^ { \star } ( { \pmb x } _ { t } ) , \quad t \in [ 0 , T ] ; \qquad { \pmb x } _ { T } \sim q _ { T } .\tag{4}
$$

At every time t, both reverse processes have marginal distribution $q _ { t } .$ , matching that of the forward process; see Song et al. (2020b). With the exact score $s _ { t } ^ { \star } ( \cdot )$ and initialization from $q _ { T }$ , both reverse processes recover $q _ { 0 }$ . In practice, sampling is approximate because the score is estimated and $q _ { T }$ is replaced by a Gaussian approximation.

Noise and data prediction. A noise prediction model $\varepsilon _ { \boldsymbol { \theta } } ( \boldsymbol { x } , t )$ , parameterized by $\theta ,$ approximates the scaled score function $- \sigma _ { t } \mathbf { s } _ { t } ^ { \star } ( \mathbf { \boldsymbol { x } } )$ . The ideal noise predictor satisfies

$$
\varepsilon _ { \theta ^ { \star } } ( \pmb { x } , t ) = - \sigma _ { t } \pmb { s } _ { t } ^ { \star } ( \pmb { x } ) = \mathbb { E } \big [ \pmb { z } \bigm | \alpha _ { t } \pmb { x } _ { 0 } + \sigma _ { t } \pmb { z } = \pmb { x } \big ] .\tag{5}
$$

For any noise predictor, the corresponding data predictor is defined by

$$
\mu _ { \boldsymbol { \theta } } ( \mathbf { \boldsymbol { x } } , t ) : = \frac { \mathbf { \boldsymbol { x } } - \sigma _ { t } \pmb { \varepsilon } _ { \boldsymbol { \theta } } ( \mathbf { \boldsymbol { x } } , t ) } { \alpha _ { t } } .\tag{6}
$$

In particular, the ideal data predictor satisfies

$$
\pmb { \mu } _ { \theta ^ { \star } } ( \pmb { x } , t ) = \mathbb { E } \big [ \pmb { x } _ { 0 } \mid \alpha _ { t } \pmb { x } _ { 0 } + \sigma _ { t } \pmb { z } = \pmb { x } \big ] .\tag{7}
$$

These conditional expectations motivate the names: $\varepsilon _ { \theta }$ and $\pmb { \mu } _ { \theta }$ predict the noise z and clean data $\scriptstyle { \pmb { x } } _ { 0 } .$ respectively, from the noisy input.

Time reparameterization and sampling grid. We use the time parameterization

$$
\lambda ( t ) = \lambda _ { t } : = \log ( \alpha _ { t } / \sigma _ { t } ) ,\tag{8}
$$

which is strictly decreasing on $( 0 , T )$ . Its inverse is denoted by t(λ). For notational convenience, we use the abbreviations

$$
\begin{array} { r } { \varepsilon _ { t } : = \varepsilon _ { \theta } ( { \boldsymbol x } _ { t } , t ) \qquad \mathrm { a n d } \qquad \mu _ { t } : = \mu _ { \theta } ( { \boldsymbol x } _ { t } , t ) } \end{array}
$$

whenever it is clear from context.

For numerical sampling, we discretize a truncated interval $[ t _ { M } , t _ { 0 } ] \subset ( 0 , T )$ using $M + 1$ time points:

$$
T > t _ { 0 } > t _ { 1 } > \cdots > t _ { M - 1 } > t _ { M } > 0 .\tag{9}
$$

The corresponding step sizes in the log-SNR variable are

$$
\delta _ { i } : = \lambda _ { t _ { i } } - \lambda _ { t _ { i - 1 } } > 0 , \qquad i = 1 , \ldots , M .\tag{10}
$$

We initialize $\mathbf { \boldsymbol { x } } _ { t _ { 0 } } \sim \mathcal { N } ( 0 , \sigma _ { t _ { 0 } } ^ { 2 } I _ { d } )$ , which is a Gaussian approximation to $q _ { t _ { 0 } }$ . Both deterministic and stochastic samplers then generate $\pmb { x } _ { t _ { 1 } } , \ldots , \pmb { x } _ { t _ { M } }$ sequentially to approximate the corresponding difusion ODE or SDE introduced below.

## 2.2 Difusion ODE and deterministic samplers

Difusion ODE. Replacing the score function ${ \mathbf { } } s _ { t } ^ { \star } ( x )$ with its approximation $- \varepsilon _ { \boldsymbol { \theta } } ( \mathbf { x } , t ) / \sigma _ { t }$ in the probability flow ODE (4) yields the difusion ODE:

$$
\dot { \pmb { x } } _ { t } = f ( t ) \pmb { x } _ { t } + \frac { g ^ { 2 } ( t ) } { 2 \sigma _ { t } } \pmb { \varepsilon } _ { \theta } ( \pmb { x } _ { t } , t ) , \qquad t \in [ t _ { M } , t _ { 0 } ] .\tag{11}
$$

Under time reparameterization (8), its solution $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ at any time $t < s$ can be expressed as

$$
\pmb { x } _ { t } = \frac { \alpha _ { t } } { \alpha _ { s } } \pmb { x } _ { s } - \alpha _ { t } \int _ { \lambda ( s ) } ^ { \lambda ( t ) } \mathrm { e } ^ { - \lambda } \pmb { \varepsilon } _ { \theta } \big ( \pmb { x } _ { t ( \lambda ) } , t ( \lambda ) \big ) \mathrm { d } \lambda\tag{12a}
$$

$$
= \frac { \sigma _ { t } } { \sigma _ { s } } \pmb { x } _ { s } + \sigma _ { t } \int _ { \lambda ( s ) } ^ { \lambda ( t ) } \mathrm { e } ^ { \lambda } \pmb { \mu } _ { \theta } \big ( \pmb { x } _ { t ( \lambda ) } , t ( \lambda ) \big ) \mathrm { d } \lambda .\tag{12b}
$$

Difusion ODE solvers. Starting from $\mathbf { \boldsymbol { x } } _ { t _ { 0 } }$ , an ODE solver produces the states $\pmb { x } _ { t _ { 1 } } , \ldots , \pmb { x } _ { t _ { M } }$ in sequence, each approximating the continuous solution at the corresponding grid point. The deterministic DDIM sampler (Song et al., 2020a) admits the following first-order update:

$$
\pmb { x } _ { t _ { i } } = \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } \pmb { x } _ { t _ { i - 1 } } - \alpha _ { t _ { i } } \left( \frac { \sigma _ { t _ { i - 1 } } } { \alpha _ { t _ { i - 1 } } } - \frac { \sigma _ { t _ { i } } } { \alpha _ { t _ { i } } } \right) \pmb { \varepsilon } _ { t _ { i - 1 } } = \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } \pmb { x } _ { t _ { i - 1 } } - \sigma _ { t _ { i } } ( \mathbf { e } ^ { \delta _ { i } } - 1 ) \pmb { \varepsilon } _ { t _ { i - 1 } } .\tag{13}
$$

Equivalently, using the data prediction parameterization (6), the DDIM update becomes

$$
\pmb { x } _ { t _ { i } } = \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \pmb { x } _ { t _ { i - 1 } } + \sigma _ { t _ { i } } \bigg ( \frac { \alpha _ { t _ { i } } } { \sigma _ { t _ { i } } } - \frac { \alpha _ { t _ { i - 1 } } } { \sigma _ { t _ { i - 1 } } } \bigg ) \pmb { \mu } _ { t _ { i - 1 } } .\tag{14}
$$

Some popular higher-order multistep solvers approximate the noise predictor along the trajectory by polynomial interpolation of previous model evaluations. For example, DPMSolver-p uses $\pmb { \varepsilon } _ { t _ { i - 1 } } , \ldots , \pmb { \varepsilon } _ { t _ { i - p } }$ to approximate the integral in (12a) (Lu et al., 2022a). We provide the general update and the commonly used cases $p = 2 , 3$ in Appendix A.1.

## 2.3 Difusion SDE and stochastic samplers

Difusion SDE. Given a noise predictor $\varepsilon _ { \boldsymbol { \theta } } ( \boldsymbol { x } , t )$ , we can instantiate the SDE (3) by substituting the unknown score function $s _ { t } ^ { \star } ( { \pmb x } )$ with $- \varepsilon _ { \boldsymbol { \theta } } ( \mathbf { x } , t ) / \sigma _ { t }$ , which yields the difusion SDE:

$$
\mathrm { d } \pmb { x } _ { t } = \left[ f ( t ) \pmb { x } _ { t } + \frac { g ^ { 2 } ( t ) } { \sigma _ { t } } \varepsilon _ { \theta } ( \pmb { x } _ { t } , t ) \right] \mathrm { d } t + g ( t ) \mathrm { d } \widetilde { \pmb { \omega } } _ { t } , \qquad t \in [ t _ { M } , t _ { 0 } ] .\tag{15}
$$

Under the time reparameterization (8), the solution $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ to the difusion SDE (15) at any time $t < s$ admits the data prediction representation

$$
\pmb { x } _ { t } = \frac { \sigma _ { t } } { \sigma _ { s } } \mathrm { e } ^ { - ( \lambda _ { t } - \lambda _ { s } ) } \pmb { x } _ { s } + 2 \alpha _ { t } \int _ { \lambda _ { s } } ^ { \lambda _ { t } } \mathrm { e } ^ { - 2 ( \lambda _ { t } - \lambda ) } \pmb { \mu } _ { \theta } ( \pmb { x } _ { t ( \lambda ) } , t ( \lambda ) ) \mathrm { d } \lambda + \sqrt { 2 } \sigma _ { t } \int _ { \lambda _ { s } } ^ { \lambda _ { t } } \mathrm { e } ^ { - ( \lambda _ { t } - \lambda ) } \mathrm { d } \overline { { \omega } } _ { \lambda } ,\tag{16}
$$

Here, $\overline { { \omega } } _ { \lambda }$ is a standard d-dimensional Brownian motion in the increasing variable $\lambda ,$ obtained by timechanging the reverse-time noise. For a detailed derivation, see Lu et al. (2022b).

Difusion SDE solvers. We consider two stochastic samplers based on (16) (Lu et al., 2022b). In both updates, $z _ { 1 } , \dots , z _ { M }$ are independent standard Gaussian vectors, independent of the initial state.

• SDE-DPM-Solver++1 (DDPM). The SDE-DPM-Solver++1 can be interpreted as a first-order solver of (16), which admits the following update rule:

$$
\begin{array} { r } { \pmb { x } _ { t _ { i } } = \mathrm { e } ^ { - \delta _ { i } } \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \pmb { x } _ { t _ { i - 1 } } + \alpha _ { t _ { i } } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } ) \pmb { \mu } _ { \theta } ( \pmb { x } _ { t _ { i - 1 } } , t _ { i - 1 } ) + \sigma _ { t _ { i } } \sqrt { 1 - \mathrm { e } ^ { - 2 \delta _ { i } } } \mathcal { z } _ { i } . } \end{array}\tag{17a}
$$

• SDE-DPM-Solver++2. SDE-DPM-Solver++2 augments the first-order update with a correction based on a diference estimate of the variation of the data predictor:

$$
x _ { t _ { i } } = \mathrm { e } ^ { - \delta _ { i } } \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } x _ { t _ { i - 1 } } + \alpha _ { t _ { i } } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } ) \mu _ { \theta } ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) + \frac { 1 } { 2 } \delta _ { i } \alpha _ { t _ { i } } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } ) D _ { t _ { i } } + \sigma _ { t _ { i } } \sqrt { 1 - \mathrm { e } ^ { - 2 \delta _ { i } } } z _ { i } ,\tag{17b}
$$

where

$$
D _ { t _ { 1 } } = \mathbf { 0 } , \qquad D _ { t _ { i } } : = { \frac { \mu _ { \theta } ( { \boldsymbol { x } } _ { t _ { i - 1 } } , t _ { i - 1 } ) - \mu _ { \theta } ( { \boldsymbol { x } } _ { t _ { i - 2 } } , t _ { i - 2 } ) } { \lambda _ { t _ { i - 1 } } - \lambda _ { t _ { i - 2 } } } } , \quad i = 2 , \ldots , M .
$$

## 3 Main results

In this section, we introduce a forward-value stochastic sampler and present the convergence guarantees of both the existing and our proposed stochastic samplers.

## 3.1 Forward-value discretization

This subsection reviews the forward-value discretization framework and its deterministic implementations from our preliminary conference version (Jiao et al., 2026). This framework provides the basis for the stochastic extension in Section 3.2.

![](images/55365189a0f5ec0e46e736e9f70d860aacd84eebb0d0cef4faf9ac5f4557e3ee.jpg)  
(a) CIFAR-10

![](images/c547a6282bae5ad9a913c02c122b35425ddc21e8c8d7cc036cce17899d9f843a.jpg)  
(b) ImageNet-64  
Figure 2: FID comparison of DDIM, DPMSolver-2, and a forward-value discretization on CIFAR-10 (EDMuncond) and ImageNet-64 (EDM2-S), versus the number of outer iterations M. The forward-value discretization uses additional model evaluations to estimate endpoints. FID is computed using 50K generated samples.

Ideal forward-value discretization. The DDIM update (14) approximates the data predictor along each integration interval by its value at the current iterate. With few sampling steps, the model evaluation placement can substantially afect sample quality. We therefore consider evaluating the data predictor at the next state-time pair, leading to the ideal forward-value discretization:

$$
\pmb { x } _ { t _ { i } } ^ { \mathrm { { f o r } } } = \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \pmb { x } _ { t _ { i - 1 } } ^ { \mathrm { { f o r } } } + \left( \alpha _ { t _ { i } } - \frac { \sigma _ { t _ { i } } \alpha _ { t _ { i - 1 } } } { \sigma _ { t _ { i - 1 } } } \right) \pmb { \mu } _ { \theta } ( \pmb { x } _ { t _ { i } } ^ { \mathrm { { f o r } } } , t _ { i } ) .\tag{18}
$$

Here, “forward-value” refers to evaluation at the next endpoint along the sampling direction, even though it is a discretization of the reverse process. Unlike DDIM, this update is implicit because the unknown next state $\pmb { x } _ { t _ { i } } ^ { \mathsf { f o r } }$ also appears in the model input. To obtain an explicit sampler, we replace this input with an estimate of the next state. Under suitable regularity and stability conditions, both DDIM and (18) converge to the same difusion ODE as the maximum step size vanishes. Our focus is their behavior with few sampling steps.

Figure 2 explores forward-value discretization with few sampling steps. For this experiment, we use five DDIM substeps within each interval $\left( t _ { i - 1 } , t _ { i } \right)$ to obtain an endpoint estimate at $t _ { i } ,$ , and then use the data prediction at that estimate in the forward-value update. This reference implementation approximates endpoint evaluation without solving the implicit equation (18). It achieves lower FID than DDIM and DPMSolver-2 in the tested settings on CIFAR-10 and ImageNet-64. Since the endpoint estimates require additional model evaluations, these results motivate the exploration of forward-value evaluation but do not establish an eficiency gain at matched NFE budgets. This raises the practical question: can we approximate forward-value evaluation while retaining one model evaluation per iteration?

F-DPMSolver: first-order forward-value deterministic sampler. We next describe a practical F-DPMSolver, which replaces the implicit endpoint in (18) with a lookahead estimate. At iteration $i \ =$ $1 , \ldots , M$ , we first construct a lookahead estimate $\widehat { \pmb { x } } _ { t _ { i } }$ of the next state using only information available from earlier iterations, without a new model evaluation. We then evaluate the data prediction model at $( \widehat { \pmb { x } } _ { t _ { i } } , t _ { i } )$ and update the current iterate via

Algorithm 1 F-DDIM   
Input: Data prediction mode $\mu _ { \boldsymbol { \theta } } ( \boldsymbol { x } , t )$ , time grid $t _ { 0 } > \cdots > t _ { M }$ , noise schedule $\left( \alpha _ { t } , \sigma _ { t } \right)$ , and initialization   
$\mathbf { \boldsymbol { x } } _ { t _ { 0 } } .$   
1: Initialize $\widetilde { \varepsilon } _ { t _ { 0 } } \gets x _ { t _ { 0 } }$   
2: for $i = 1 , \dots , M$ do   
3: Compute a lookahead estimate $\widehat { \pmb { x } } _ { t _ { i } } \gets \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } \pmb { x } _ { t _ { i - 1 } } - \left( \frac { \alpha _ { t _ { i } } \sigma _ { t _ { i - 1 } } } { \alpha _ { t _ { i - 1 } } } - \sigma _ { t _ { i } } \right) \widetilde { \pmb { \varepsilon } } _ { t _ { i - 1 } } .$   
4: Update $\pmb { x } _ { t _ { i } } \gets \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \pmb { x } _ { t _ { i - 1 } } + \left( \alpha _ { t _ { i } } - \frac { \sigma _ { t _ { i } } \alpha _ { t _ { i - 1 } } } { \sigma _ { t _ { i - 1 } } } \right) \pmb { \mu } _ { \theta } ( \widehat { \pmb { x } } _ { t _ { i } } , t _ { i } )$   
5: Set and cache $\widetilde { \varepsilon } _ { t _ { i } } \gets \big ( \widehat { \pmb { x } } _ { t _ { i } } - \alpha _ { t _ { i } } \pmb { \mu } _ { \theta } ( \widehat { \pmb { x } } _ { t _ { i } } , t _ { i } ) \big ) / \sigma _ { t _ { i } }$   
6: end for   
Output: Generated sample $\mathbf { \mathcal { x } } _ { t _ { M } }$

$$
\pmb { x } _ { t _ { i } } = \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \pmb { x } _ { t _ { i - 1 } } + \left( \alpha _ { t _ { i } } - \frac { \sigma _ { t _ { i } } \alpha _ { t _ { i - 1 } } } { \sigma _ { t _ { i - 1 } } } \right) \pmb { \mu } _ { \theta } ( \widehat { \pmb { x } } _ { t _ { i } } , t _ { i } ) .\tag{19}
$$

This update has the same form as DDIM (14), except that the current-state evaluation $\mu _ { \theta } ( \pmb { x } _ { t _ { i - 1 } } , t _ { i - 1 } )$ is replaced with the lookahead evaluation $\mu _ { \theta } ( \widehat { \mathbf { x } } _ { t _ { i } } , t _ { i } )$

The data predictor output gives the noise prediction

$$
\widetilde { \varepsilon } _ { t _ { i } } : = \frac { \widehat { \mathbf { x } } _ { t _ { i } } - \alpha _ { t _ { i } } \pmb { \mu } _ { \theta } ( \widehat { \mathbf { x } } _ { t _ { i } } , t _ { i } ) } { \sigma _ { t _ { i } } } = \varepsilon _ { \theta } ( \widehat { \mathbf { x } } _ { t _ { i } } , t _ { i } ) .\tag{20}
$$

The noise prediction $\widetilde { \varepsilon } _ { t _ { i } }$ is cached for subsequent lookahead steps. We initialize the cache by setting $\widetilde { \varepsilon } _ { t _ { 0 } } = \mathbf { { x } } _ { t _ { 0 } }$ without querying the model. Thus, each iteration requires exactly one new model evaluation, giving a tota NFE of M. Algorithm 3 in Appendix A.3 summarizes the framework.

F-DDIM. A simple choice of lookahead predictor is the DDIM update formula (13) with the cached noise estimate in place of a fresh model evaluation:

$$
\widehat { \pmb { x } } _ { t _ { i } } = \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } \pmb { x } _ { t _ { i - 1 } } - \left( \frac { \alpha _ { t _ { i } } \sigma _ { t _ { i - 1 } } } { \alpha _ { t _ { i - 1 } } } - \sigma _ { t _ { i } } \right) \widetilde { \pmb { \varepsilon } } _ { t _ { i - 1 } } .\tag{21}
$$

For $i \geq 2 ,$ , the cached estimate is $\widetilde { \pmb { \varepsilon } } _ { t _ { i - 1 } } = \pmb { \varepsilon } _ { \pmb { \theta } } ( \widehat { \pmb { x } } _ { t _ { i - 1 } } , t _ { i - 1 } )$ . Thus, the predictor starts from the updated state $\mathbf { \Delta } \mathbf { x } _ { t _ { i - 1 } }$ but reuses the model output computed at the previous lookahead point $\widehat { \pmb { x } } _ { t _ { i - 1 } }$ . We call this instantiation F-DDIM and give its complete procedure in Algorithm 1.

Higher-order lookahead predictors. The framework also accommodates predictors that use multiple cached noise estimates to construct the lookahead prediction. We denote the variant based on a p-th order DPMSolver predictor by F-DPMSolver-p. At iteration i, we use a predictor of order $k _ { i } = \mathrm { m i n } \{ p , i \}$ with the cached estimates $\widetilde { \varepsilon } _ { t _ { i - 1 } } , \ldots , \widetilde { \varepsilon } _ { t _ { i - k _ { i } } }$ . The procedure starts with the F-DDIM predictor and increases the predictor order as more cached outputs become available, without additional model evaluations. For example, the F-DPMSolver-2 predictor uses cached estimates $\widetilde { \varepsilon } _ { t _ { i - 1 } }$ and $\widetilde { \varepsilon } _ { t _ { i - 2 } }$ to construct $\widehat { \pmb { x } } _ { t _ { i } }$ for $i \geq 2$ . Here, p specifies the lookahead predictor; it does not imply p-th order convergence of the resulting sampler.

The deterministic framework shifts model evaluation to a lookahead point while retaining one model evaluation per iteration. Its stochastic extension must additionally account for the noise injected during each update, which afects both the appropriate evaluation time and the construction of the cached noise estimate. We address these two issues in Section 3.2.

Algorithm 2 F-DDPM   
Input: Data prediction model $\mu _ { \boldsymbol { \theta } } ( \boldsymbol { x } , t )$ , time grid $t _ { 0 } > \cdots > t _ { M }$ , noise schedule $\left( \alpha _ { t } , \sigma _ { t } \right)$ , initialization $\mathbf { \nabla } _ { \mathbf { x } _ { t _ { 0 } } . }$   
1: Compute $\lambda _ { t _ { i } }  \log ( \alpha _ { t _ { i } } / \sigma _ { t _ { i } } )$ for $i = 0 , \ldots , M$   
2: Generate Gaussian noise $z _ { 1 } , \dots , z _ { M } \overset { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( 0 , I _ { d } )$   
3: Initialize $\widetilde { \varepsilon } _ { t _ { 0 } } \gets x _ { t _ { 0 } }$   
4: for i = 1, . . . , M do   
5: Set $\delta _ { i } \gets \lambda _ { t _ { i } } - \lambda _ { t _ { i - 1 } }$ and compute $t _ { i } ^ { \prime }$ such that $\lambda _ { t _ { i } ^ { \prime } } = \lambda _ { t _ { i } } + \delta _ { i }$   
6: Compute a lookahead estimate $\widehat { \pmb { x } } _ { t _ { i } ^ { \prime } } \gets \frac { \alpha _ { t _ { i } ^ { \prime } } } { \alpha _ { t _ { i - 1 } } } \pmb { x } _ { t _ { i - 1 } } - \sigma _ { t _ { i } ^ { \prime } } \big ( \mathrm { e } ^ { 2 \delta _ { i } } - 1 \big ) \widetilde { \pmb { \varepsilon } } _ { t _ { i - 1 } }$   
7: Update $x _ { t _ { i } } \gets \mathrm { e } ^ { - \delta _ { i } } \frac { \sigma _ { t _ { i } } } { \sigma _ { t \gets . } } x _ { t _ { i - 1 } } + \alpha _ { t _ { i } } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } ) \mu _ { \theta } ( \widehat { x } _ { t _ { i } ^ { \prime } } , t _ { i } ^ { \prime } ) + \sigma _ { t _ { i } } \sqrt { 1 - \mathrm { e } ^ { - 2 \delta _ { i } } } z _ { i }$   
8: Set and cache $\widetilde { \varepsilon } _ { t _ { i } } \gets \mathrm { e } ^ { - \delta _ { i } } \varepsilon _ { \theta } ( \widehat { x } _ { t _ { i } ^ { \prime } } , t _ { i } ^ { \prime } ) + \sqrt { 1 - \mathrm { e } ^ { - 2 \delta _ { i } } } z _ { i } ,$ with $\varepsilon _ { \theta } ( \widehat { x } _ { t _ { i } ^ { \prime } } , t _ { i } ^ { \prime } ) = \left( \widehat { x } _ { t _ { i } ^ { \prime } } - \alpha _ { t _ { i } ^ { \prime } } \mu _ { \theta } ( \widehat { x } _ { t _ { i } ^ { \prime } } , t _ { i } ^ { \prime } ) \right) / \sigma _ { t _ { i } ^ { \prime } }$   
9: end for   
Output: Generated sample $\mathbf { \mathcal { x } } _ { t _ { M } }$

## 3.2 F-DDPM: first-order forward-value stochastic sampler

Building on the forward-value discretization framework, we introduce F-DDPM, a fast stochastic sampler that serves as the forward-value counterpart of DDPM under the data-prediction parametrization in (17a). The complete procedure is summarized in Algorithm 2.

At a high level, F-DDPM has the same form as DDPM, and the only diference is where the data prediction model $\mu _ { \theta }$ is evaluated. DDPM evaluates it at the current state-time pair $( \mathbf { \boldsymbol { x } } _ { t _ { i - 1 } } , t _ { i - 1 } )$ , whereas F-DDPM evaluates it at a lookahead state-time pair $( \widehat { \pmb { x } } _ { t _ { i } ^ { \prime } } , t _ { i } ^ { \prime } )$ built from information available through step i − 1 only. Concretely, for iteration $i = 1 , \ldots , M$ , we update the iterate ${ \mathbf { \mathcal { x } } } _ { t _ { i - 1 } }$ via

$$
{ \pmb x } _ { t _ { i } } = \mathrm { e } ^ { - \delta _ { i } } \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } { \pmb x } _ { t _ { i - 1 } } + \alpha _ { t _ { i } } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } ) { \pmb \mu } _ { \theta } \left( \widehat { { \pmb x } } _ { t _ { i } ^ { \prime } } , t _ { i } ^ { \prime } \right) + \sigma _ { t _ { i } } \sqrt { 1 - \mathrm { e } ^ { - 2 \delta _ { i } } } { \pmb z } _ { i } ,\tag{22}
$$

where $z _ { i } \overset { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( 0 , I _ { d } )$ for $i = 1 , \dots , M$ are independent Gaussian noise and $\delta _ { i } : = \lambda _ { t _ { i } } - \lambda _ { t _ { i - 1 } }$ is defined in (10).

Below, let us specify the evaluation time $t _ { i } ^ { \prime }$ and the lookahead estimate $\widehat { \mathbf { x } } _ { t _ { i } ^ { \prime } }$

Evaluation time $t _ { i } ^ { \prime } .$ . Recall that for the deterministic sampler F-DPMSolver in (19), the forward-value evaluation is taken at the end time $t _ { i }$ of the current step. In contrast, F-DDPM evaluates the data predictor at a cleaner time $t _ { i } ^ { \prime } < t _ { i } ,$ chosen to account for the noise injected during the stochastic update.

For each $i \geq 1$ , let $t _ { i } ^ { \prime }$ be the time whose log-SNR $\lambda _ { t _ { i } ^ { \prime } }$ exceeds that of the end time $t _ { i }$ by $\delta _ { i } > 0 !$

$$
\lambda _ { t _ { i } ^ { \prime } } = \lambda _ { t _ { i } } + \delta _ { i } = \lambda _ { t _ { i - 1 } } + 2 \delta _ { i } .\tag{23}
$$

Since $\lambda _ { t }$ is decreasing in $t ,$ the evaluation time $t _ { i } ^ { \prime } < t _ { i }$ lies past the current end time $t _ { i }$ along the sampling direction. When the time grid is uniform in log-SNR, we have $t _ { i } ^ { \prime } = t _ { i + 1 }$ for $i = 1 , \ldots , M - 1$

To motivate this choice, denote by $\boldsymbol { \mathbf { \mathit { x } } } _ { t } ^ { \star }$ the solution to the SDE (2) for the forward process. Conditioned on the state $\boldsymbol { x } _ { t _ { i - 1 } } ^ { \star }$ at time $t _ { i - 1 }$ and the clean data $\pmb { x } _ { 0 } ^ { \star }$ , the posterior distribution of the state $\boldsymbol { x } _ { t _ { i } } ^ { \star }$ at time $t _ { i }$ is Gaussian with covariance $\sigma _ { t _ { i } } ^ { 2 } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } ) I _ { d }$ and mean

$$
\mathrm { e } ^ { - \delta _ { i } } \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \pmb { x } _ { t _ { i - 1 } } ^ { \star } + \alpha _ { t _ { i } } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } ) \pmb { x } _ { 0 } ^ { \star } = \alpha _ { t _ { i } } \pmb { x } _ { 0 } ^ { \star } + \sigma _ { t _ { i } } \mathrm { e } ^ { - \delta _ { i } } \varepsilon\tag{24}
$$

where $\varepsilon = ( \pmb { x } _ { t _ { i - 1 } } ^ { \star } - \alpha _ { t _ { i - 1 } } \pmb { x } _ { 0 } ^ { \star } ) / \sigma _ { t _ { i - 1 } } \sim \mathcal { N } ( 0 , { \cal I } _ { d } )$ is independent of $\pmb { x } _ { 0 } ^ { \star }$ . The i-th update of DDPM is thus approximate sampling from this posterior distribution with $\mathbf { \boldsymbol { x } } _ { 0 } ^ { \star }$ replaced by $\mu _ { \theta } ( \pmb { x } _ { t _ { i - 1 } } , t _ { i - 1 } )$ . Now, notice that the right-hand side of (24) can be interpreted as noisy data with log-SNR

$$
\log \Big ( \frac { \alpha _ { t _ { i } } } { \sigma _ { t _ { i } } \mathrm { e } ^ { - \delta _ { i } } } \Big ) = \lambda _ { t _ { i } } + \delta _ { i } ,
$$

which is higher than $\lambda _ { t _ { i } }$ at time $t _ { i }$ . Matching this log-SNR with $\lambda _ { t _ { i } ^ { \prime } } = \log ( \alpha _ { t _ { i } ^ { \prime } } / \sigma _ { t _ { i } ^ { \prime } } )$ yields the choice of $t _ { i } ^ { \prime }$ in (23). The injected Gaussian noise with variance $\sigma _ { t _ { i } } ^ { 2 } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } )$ then restores the total noise variance $\sigma _ { t _ { i } } ^ { 2 }$ at time $t _ { i }$

This choice also yields a useful algebraic interpretation. By the definition of $t _ { i } ^ { \prime } ,$ , one can rewrite the update rule of DDPM in (17a) as

$$
\begin{array} { r } { \mathbf { { x } } _ { t _ { i } } = \underbrace { \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i } ^ { \prime } } } } _ { \mathrm { { o n e ~ D D I M ~ s t e p } } \ ( 1 4 ) \ \mathrm { f r o m } \ t _ { i - 1 } } \underbrace { \left[ \frac { \sigma _ { t _ { i } ^ { \prime } } } { \sigma _ { t _ { i - 1 } } } \mathbf { { x } } _ { t _ { i - 1 } } + \sigma _ { t _ { i } ^ { \prime } } \left( \frac { \alpha _ { t _ { i } ^ { \prime } } } { \sigma _ { t _ { i } ^ { \prime } } } - \frac { \alpha _ { t _ { i - 1 } } } { \sigma _ { t _ { i - 1 } } } \right) \mu _ { \theta } ( \pmb { x } _ { t _ { i - 1 } } , t _ { i - 1 } ) \right] } _ { \mathrm { { o n e ~ D D I M ~ s t e p } \ ( 1 4 ) \ \mathrm { f r o m } \ t _ { i - 1 } \ \mathrm { t o } \ t _ { i } ^ { \prime } } } + \sigma _ { t _ { i } } \sqrt { 1 - \mathrm { e } ^ { - 2 \delta _ { i } } } \ z _ { i } } \end{array}\tag{25}
$$

The quantity in the square bracket corresponds to a one-step DDIM update from $t _ { i - 1 }$ to $t _ { i } ^ { \prime } .$ . Therefore, the stochastic update of DDPM can be interpreted as (1) taking a DDIM step to a cleaner $t _ { i } ^ { \prime } , \ ( 2 )$ rescaling the result to the signal level $\alpha _ { t _ { i } }$ , and (3) injecting Gaussian noise to obtain the total noise level $\sigma _ { t _ { i } } ^ { 2 }$ at time $t _ { i }$

The ideal forward-value rule applied to this DDIM step from $t _ { i - 1 }$ to $t _ { i } ^ { \prime }$ would replace its start-time evaluation $\mu _ { \theta } ( \pmb { x } _ { t _ { i - 1 } } , t _ { i - 1 } )$ with the end-time evaluation $\mu _ { \theta } ( \pmb { x } _ { t _ { i } ^ { \prime } } , t _ { i } ^ { \prime } )$ . Here, $\mathbf { x } _ { t _ { i } ^ { \prime } }$ denotes the idealized endpoint of this deterministic substep, rather than an already generated stochastic iterate. As $\mathbf { \Delta } \mathbf { x } _ { t _ { i } ^ { \prime } }$ is not available at step $i - 1$ , we approximate it with a lookahead estimate $\widehat { \pmb { x } } _ { t _ { i } ^ { \prime } }$ constructed using only information available up to step $i - 1$ , leading to the modified term in the square bracket:

$$
\frac { \sigma _ { t _ { i } ^ { \prime } } } { \sigma _ { t _ { i - 1 } } } \pmb { x } _ { t _ { i - 1 } } + \sigma _ { t _ { i } ^ { \prime } } \left( \frac { \alpha _ { t _ { i } ^ { \prime } } } { \sigma _ { t _ { i } ^ { \prime } } } - \frac { \alpha _ { t _ { i - 1 } } } { \sigma _ { t _ { i - 1 } } } \right) \pmb { \mu } _ { \theta } \left( \widehat { \mathbf { x } } _ { t _ { i } ^ { \prime } } , t _ { i } ^ { \prime } \right) .
$$

Plugging this into (25) yields the update rule of F-DDPM in (22). It remains to specify the lookahead estimate $\widehat { \pmb { x } } _ { t _ { i } ^ { \prime } }$

Lookahead estimate $\widehat { \mathbf { x } } _ { t _ { \widehat { \mathbf { \Pi } } } ^ { \prime } }$ . Given the evaluation time $t _ { i } ^ { \prime } ,$ we construct the lookahead estimate $\widehat { \pmb { x } } _ { t _ { i } ^ { \prime } }$ by taking a single DDIM step (13) from $t _ { i - 1 }$ to $t _ { i } ^ { \prime } .$ , written in the noise-prediction form:

$$
\widehat { \boldsymbol { x } } _ { t _ { i } ^ { \prime } } = \frac { \alpha _ { t _ { i } ^ { \prime } } } { \alpha _ { t _ { i - 1 } } } \boldsymbol { x } _ { t _ { i - 1 } } - \alpha _ { t _ { i } ^ { \prime } } \biggl ( \frac { \sigma _ { t _ { i - 1 } } } { \alpha _ { t _ { i - 1 } } } - \frac { \sigma _ { t _ { i } ^ { \prime } } } { \alpha _ { t _ { i } ^ { \prime } } } \biggr ) \boldsymbol { \widetilde { \varepsilon } } _ { t _ { i - 1 } } = \frac { \alpha _ { t _ { i } ^ { \prime } } } { \alpha _ { t _ { i - 1 } } } \boldsymbol { x } _ { t _ { i - 1 } } - \sigma _ { t _ { i } ^ { \prime } } \bigl ( \mathrm { e } ^ { 2 \delta _ { i } } - 1 \bigr ) \boldsymbol { \widetilde { \varepsilon } } _ { t _ { i - 1 } } .\tag{26}
$$

Here, $\widetilde { \varepsilon } _ { t _ { i } }$ is an estimate of the noise prediction $\varepsilon _ { \boldsymbol { \theta } } ( x _ { t _ { i - 1 } } , t _ { i - 1 } )$ using model evaluations from the previous step $i - 1$ for $1 \geq 2$ . This avoids an additional model evaluation for $\varepsilon _ { \boldsymbol { \theta } } ( x _ { t _ { i - 1 } } , t _ { i - 1 } )$ at the current step.

For i > $i > 1$ , writing (25) at step i − 1 and replacing the square bracket by the quantity it approximates, namely $\pmb { x } _ { t _ { i - 1 } ^ { \prime } }$ , we obtain

$$
\begin{array} { r l } & { x _ { t _ { i - 1 } } \approx \frac { \alpha _ { t _ { i - 1 } } } { \alpha _ { t _ { i - 1 } ^ { \prime } } } x _ { t _ { i - 1 } ^ { \prime } } + \sigma _ { t _ { i - 1 } } \sqrt { 1 - \mathrm { e } ^ { - 2 \delta _ { i - 1 } } } z _ { i - 1 } } \\ & { \qquad = \frac { \alpha _ { t _ { i - 1 } } } { \alpha _ { t _ { i - 1 } ^ { \prime } } } \Big ( \alpha _ { t _ { i - 1 } ^ { \prime } } \mu _ { \theta } ( x _ { t _ { i - 1 } ^ { \prime } } , t _ { i - 1 } ^ { \prime } ) + \sigma _ { t _ { i - 1 } ^ { \prime } } \varepsilon _ { \theta } ( x _ { t _ { i - 1 } ^ { \prime } } , t _ { i - 1 } ^ { \prime } ) \Big ) + \sigma _ { t _ { i - 1 } } \sqrt { 1 - \mathrm { e } ^ { - 2 \delta _ { i - 1 } } } z _ { i - 1 } } \\ & { \qquad = \alpha _ { t _ { i - 1 } } \mu _ { \theta } ( x _ { t _ { i - 1 } ^ { \prime } } , t _ { i - 1 } ^ { \prime } ) + \frac { \sigma _ { t _ { i - 1 } ^ { \prime } } } { \alpha _ { t _ { i - 1 } ^ { \prime } } } \alpha _ { t _ { i - 1 } } \varepsilon _ { \theta } \big ( x _ { t _ { i - 1 } ^ { \prime } } , t _ { i - 1 } ^ { \prime } \big ) + \sigma _ { t _ { i - 1 } } \sqrt { 1 - \mathrm { e } ^ { - 2 \delta _ { i - 1 } } } z _ { i - 1 } . } \end{array}
$$

where the second line applies the data/noise prediction identity (6) at $t _ { i - 1 } ^ { \prime }$ . Meanwhile, by the same identity at $t _ { i - 1 }$ , we have

$$
\pmb { x } _ { t _ { i - 1 } } = \alpha _ { t _ { i - 1 } } \pmb { \mu } _ { \boldsymbol { \theta } } ( \pmb { x } _ { t _ { i - 1 } } , t _ { i - 1 } ) + \sigma _ { t _ { i - 1 } } \pmb { \varepsilon } _ { \boldsymbol { \theta } } ( \pmb { x } _ { t _ { i - 1 } } , t _ { i - 1 } ) .
$$

If the data predictions change little between the endpoint and intermediate time, i.e., $\mu _ { \theta } ( \pmb { x } _ { t _ { i - 1 } ^ { \prime } } , t _ { i - 1 } ^ { \prime } )$ ≈ $\mu _ { \theta } ( \pmb { x } _ { t _ { i - 1 } } , t _ { i - 1 } )$ , comparing the two expressions above suggests

$$
\begin{array} { r } { \varepsilon _ { \theta } ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) \approx \mathrm { e } ^ { - \delta _ { i - 1 } } \varepsilon _ { \theta } ( x _ { t _ { i - 1 } ^ { \prime } } , t _ { i - 1 } ^ { \prime } ) + \sqrt { 1 - \mathrm { e } ^ { - 2 \delta _ { i - 1 } } } z _ { i - 1 } . } \end{array}
$$

Replacing the idealized endpoint $\pmb { x } _ { t _ { i - 1 } ^ { \prime } }$ with the available lookahead estimate $\widehat { \pmb { x } } _ { t _ { i - 1 } ^ { \prime } }$ therefore motivates the practical estimator

$$
\begin{array} { r } { \widetilde { \varepsilon } _ { t _ { i - 1 } } : = \mathrm { e } ^ { - \delta _ { i - 1 } } \varepsilon _ { \theta } ( \widehat { x } _ { t _ { i - 1 } ^ { \prime } } , t _ { i - 1 } ^ { \prime } ) + \sqrt { 1 - \mathrm { e } ^ { - 2 \delta _ { i - 1 } } } z _ { i - 1 } . } \end{array}\tag{27}
$$

For $i = 1$ , we simply set $\widetilde { \varepsilon } _ { t _ { 0 } } = x _ { t _ { 0 } }$ as in F-DPMSolver.

To conclude, each iteration of F-DDPM requires only one new model evaluation, at $( \widehat { \pmb { x } } _ { t _ { i } ^ { \prime } } , t _ { i } ^ { \prime } )$ —the resulting noise prediction is used both to compute the data prediction in (22) and to form the noise estimate for the next iteration. Thus, F-DDPM uses M model evaluations in total.

## 3.3 Convergence analysis

We analyze the trajectory discretization errors of the forward-value samplers and the baseline difusion ODE and SDE solvers. The numerical sampler and its continuous reference process use the same initial state. In the stochastic setting, they are additionally coupled through the same Brownian motion. These convergence guarantees characterize approximation to the continuous dynamics and complement the few-step samplequality comparisons in Section 4.

Convergence criteria. We consider time grids $\{ t _ { i } \} _ { i = 0 } ^ { M }$ with fixed endpoints $t _ { 0 }$ and $t _ { M } ,$ , corresponding to a fixed finite log-SNR interval. Convergence rates are measured as $M \to \infty$ . For deterministic samplers, we fix the initial state $\mathbf { \boldsymbol { x } } _ { t _ { 0 } }$ and measure the Euclidean error at the final time.

Definition 1. A deterministic difusion sampler has convergence order at least k if

$$
\left\| \pmb { x } _ { t _ { M } } - \pmb { x } _ { t _ { M } } ^ { \star } \right\| _ { 2 } = O ( 1 / M ^ { k } ) ,
$$

where $\boldsymbol { x } _ { t } ^ { \star }$ is the exact solution to the difusion ODE (12a) when initialized at $\boldsymbol { x } _ { t _ { 0 } } ^ { \star } = \boldsymbol { x } _ { t _ { 0 } }$ . In particular, for $0 \leq t \leq t _ { 0 }$ , it satisfies

$$
\pmb { x } _ { t } ^ { \star } = \frac { \alpha _ { t } } { \alpha _ { t _ { 0 } } } \pmb { x } _ { t _ { 0 } } ^ { \star } - \alpha _ { t } \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t } } \mathrm { e } ^ { - \lambda } \pmb { \varepsilon } _ { \theta } \big ( \pmb { x } _ { t ( \lambda ) } ^ { \star } , t ( \lambda ) \big ) \mathrm { d } \lambda , \qquad 0 \leq t \leq t _ { 0 } .\tag{28}
$$

For stochastic samplers, we measure the expected Euclidean error under a common initialization and Brownian motion.

Definition 2. A stochastic difusion sampler has convergence order at least k if

$$
\mathbb { E } \Big [ \big \| \pmb { x } _ { t _ { M } } - \pmb { x } _ { t _ { M } } ^ { \star } \big \| _ { 2 } \Big ] = O ( 1 / M ^ { k } ) ,
$$

where $\boldsymbol { \mathbf { \mathit { x } } } _ { t } ^ { \star }$ is the exact solution to the difusion SDE (16), initialized at $\boldsymbol { x } _ { t _ { 0 } } ^ { \star } = \boldsymbol { x } _ { t _ { 0 } }$ and driven by the same Brownian $\overline { { \omega } } _ { \lambda }$ as the numerical sampler. In particular, for $0 \leq t < s \leq t _ { 0 }$ , it satisfies

$$
\pmb { x } _ { t } ^ { \star } = \frac { \sigma _ { t } } { \sigma _ { s } } \mathrm { e } ^ { - ( \lambda _ { t } - \lambda _ { s } ) } \pmb { x } _ { s } ^ { \star } + 2 \alpha _ { t } \int _ { \lambda _ { s } } ^ { \lambda _ { t } } \mathrm { e } ^ { - 2 ( \lambda _ { t } - \lambda ) } \pmb { \mu } _ { \theta } ( \pmb { x } _ { t ( \lambda ) } ^ { \star } , \lambda ) \mathrm { d } \lambda + \sqrt { 2 } \sigma _ { t } \int _ { \lambda _ { s } } ^ { \lambda _ { t } } \mathrm { e } ^ { - ( \lambda _ { t } - \lambda ) } \mathrm { d } \overline { { \omega } } _ { \lambda } .\tag{29}
$$

The Gaussian vectors $z _ { i }$ in the stochastic updates are coupled to this Brownian motion through

$$
\sigma _ { t _ { i } } \sqrt { 1 - \mathrm { e } ^ { - 2 \delta _ { i } } } z _ { i } = \sqrt { 2 } \sigma _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - ( \lambda _ { t _ { i } } - \lambda ) } \mathrm { d } \overline { { \omega } } _ { \lambda } .
$$

Regularity assumptions. We impose conditions on the sampling grids, noise schedules, and prediction model to control the discretization errors. The model assumptions are stated in terms of the data predictor $\pmb { \mu } _ { \theta }$ in (6), which has one-to-one correspondence with the noise predictor $\varepsilon _ { \boldsymbol { \theta } } .$

Assumption 1. We assume that the time grid satisfies max $\cdot _ { i \in [ M ] } \delta _ { i } = { \cal O } ( 1 / M )$ and $\begin{array} { r } { \operatorname* { m a x } _ { i \in [ M ] } \delta _ { i } / \operatorname* { m i n } _ { i \in [ M ] } \delta _ { i } = } \end{array}$ $O ( 1 )$ . The noise schedules $\alpha _ { t ( \lambda ) }$ and $\sigma _ { t ( \lambda ) }$ are positive and twice continuously diferentiable on $[ \lambda _ { t _ { 0 } } ,$ , max<sub>1≤m≤M</sub> $\lambda _ { t _ { m } ^ { \prime } } ]$ with $\alpha _ { t ( \lambda ) }$ nondecreasing and $\sigma _ { t ( \lambda ) }$ nonincreasing as functions of $\lambda .$ In addition, for every $\pmb { x } \in \mathbb { R } ^ { d }$ and $\lambda \in [ \lambda _ { t _ { 0 } } ,$ , max $\lambda _ { t _ { m } ^ { \prime } } ]$ , we assume that the data predictor $\mu _ { \boldsymbol { \theta } } ( \pmb { x } , t )$ satisfies

(A1) $\| \mu _ { \theta } ( \pmb { x } , t ) \| \leq L _ { \mu } \mathrm { , }$

(A2) $\| \nabla _ { x } \mu _ { \theta } ( x , t ) \| \leq L _ { x } / \alpha _ { t } ;$

(A3) $\| \nabla _ { x } ^ { 2 } \mu _ { \theta } ( x , t ) \| \leq H _ { x } ;$

(A4) $\begin{array} { r } { \left\| \frac { \partial } { \partial \lambda } \pmb { \mu } _ { \theta } \big ( \pmb { x } , t ( \lambda ) \big ) \right\| \leq L _ { t } ; } \end{array}$

(A5) $\begin{array} { r } { \left| \left| \frac { \partial ^ { 2 } } { \partial \lambda ^ { 2 } } \pmb { \mu } _ { \theta } \left( \pmb { x } , t ( \lambda ) \right) \right| \right| \leq H _ { t } ; } \end{array}$

(A6) $\begin{array} { r } { \left\| \frac { \partial } { \partial \lambda } \nabla _ { \pmb { x } } \pmb { \mu } _ { \theta } \big ( \pmb { x } , t ( \lambda ) \big ) \right\| \leq H . } \end{array}$

Assumption 1 imposes a maximum log-SNR step size of $O ( M ^ { - 1 } )$ , a bounded ratio between the largest and smallest steps, monotonicity and smoothness of the noise schedules, and boundedness and regularity of the data predictor. The maximum-step condition and spatial Lipschitz regularity have been assumed in difusion solver analyses (Lu et al., 2022a; Zhao et al., 2023). The bounded step ratio simplifies the treatment of nonuniform grids, and is mainly used for the analysis of SDE-DPMSolver++2. The derivative conditions support the Taylor expansions and remainder estimates used in our proofs. Related total-derivative assumptions along ODE trajectories appear in DPM-Solver and UniPC (Lu et al., 2022a; Zhao et al., 2023), while stochastic solver analyses also impose spatial and temporal coeficient regularity (Gonzalez et al., 2023).

## 3.3.1 Convergence characterization of deterministic samplers

Previous analyses establish error upper bounds for deterministic difusion solvers; see, for example, (Lu et al., 2022a, Theorem 3.2) and (Zheng et al., 2023, Theorem 3.3). We complement these guarantees with matching worst-case lower bounds for DDIM and DPMSolver-2. Notably, these lower bounds cannot be avoided even in the most ideal setting when the true score functions are accessible (i.e., there is no score matching error).

Theorem 1. Under Assumption 1, the following convergence guarantees hold:

• DDIM (13) converges with order 1 (in terms of Definition 1);

• The DPMSolver-2 $( c f . ~ ( 3 1 ) )$ converges with order 2.

In addition, for any time grid $\{ t _ { i } \} _ { i = 0 } ^ { M }$ satisfying $\delta _ { i } = \lambda _ { t _ { i } } - \lambda _ { t _ { i - 1 } } = O ( M ^ { - 1 } )$ , the following bounds hold, each for some target distribution:

• Deterministic DDIM has convergence order at most 1;

• DPMSolver-2 (cf. (31)) has convergence order at most 2.

Theorem 1 shows that the error upper bounds for DDIM and DPMSolver-2 are sharp. For these deterministic methods, the higher-order approximation improves the convergence order from 1 to 2. The proof is deferred to Appendix B.

## 3.3.2 Convergence characterization of stochastic samplers

We next analyze SDE-DPMSolver++1 and SDE-DPMSolver++2. Theorem 2 gives first-order error upper bounds for both methods under the expected Euclidean error in Definition 2.

Theorem 2. Under Assumption 1, both SDE-DPMSolver++1 and $S D E \ – D P M S o l v e r \substack { + + 2 }$ satisfy the firstorder error bound in Definition 2.

The proof is deferred to Appendix C. We establish matching worst-case lower bounds next. These bounds hold even with exact score functions, showing that the first-order rates cannot generally be improved for these methods.

Theorem 3. Consider any time grid $\{ t _ { i } \} _ { i = 0 } ^ { M }$ satisfying

$$
\delta _ { i } = \lambda _ { t _ { i } } - \lambda _ { t _ { i - 1 } } = O ( M ^ { - 1 } ) , \qquad a n d \qquad \frac { \operatorname * { m a x } _ { i } \delta _ { i } } { \operatorname * { m i n } _ { i } \delta _ { i } } = O ( 1 ) .
$$

For each of SDE-DPMSolver++1 and $S D E \mathrm { - } D P M S o l v e r \mathrm { + + 2 , }$ there exists a target distribution for which its convergence order is at most 1.

The proof of this result is postponed to Appendix D. Together, Theorems 2 and 3 establish sharp first-order convergence rates for both methods. In particular, the higher-order approximation used by SDE-DPMSolver++2 does not improve its strong convergence order over SDE-DPMSolver++1. Thus, increasing the order of a deterministic approximation does not automatically improve the strong convergence order of the resulting stochastic sampler.

## 3.3.3 Convergence of forward-value samplers

We now analyze the forward-value samplers introduced in Sections 3.1 and 3.2. We first establish a first-order error upper bound for F-DDIM and show that this rate is sharp in the worst case.

Theorem 4. Under Assumption 1, F-DDIM satisfies the first-order error bound in Definition 1. Moreover, for any time grids $\{ \lambda _ { t _ { i } } \} _ { i = 0 } ^ { M }$ satisfying $\delta _ { i } = O ( M ^ { - 1 } )$ , there exists a target distribution for which the convergence order of F-DDIM is at most 1.

The proof is postponed to Appendix E. Higher-order discretization is a common approach to improving difusion sampling with few steps. Theorem 4, together with the experiments in Section 4.1, shows that good few-step sample quality can also be achieved by a first-order sampler: F-DDIM remains first-order convergent while attaining competitive or better FID than higher-order baselines at small NFE budgets. This highlights model evaluation placement as a design principle that complements improvements in convergence order. The distinction is particularly relevant to stochastic sampling, where the higher-order approximation in SDE-DPMSolver++2 does not improve its first-order strong convergence rate. These observations explain the forward-value design of F-DDPM in Section 3.2, which applies this principle while accounting for the injected noise.

For completeness, we also establish a first-order error upper bound for F-DDPM.

Theorem 5. Under Assumption 1, F-DDPM satisfies the first-order error bound in Definition 2.

The detailed proof of this result is postponed to Appendix F. This guarantee complements the empirical results in Section 4 tells that: F-DDPM improves sample quality at small NFE budgets while retaining a first-order strong convergence guarantee.

## 4 Experiments

In this section, we evaluate the forward-value samplers from two perspectives. First, we examine whether the first-order deterministic sampler F-DDIM achieves competitive sample quality at small NFEs. Second, we compare F-DDPM with stochastic baselines at matched NFEs and with deterministic samplers using 1.5 times their NFEs.

Models and datasets. We evaluate the samplers on CIFAR-10, ImageNet, LSUN, and FFHQ datasets. For CIFAR-10 (Krizhevsky et al., 2009), we use unconditional and class-conditional EDM models (Karras et al., 2022). For ImageNet (Deng et al., 2009), we use EDM2-S and EDM2-L at resolution 64 × 64, and EDM2-XS and EDM2-XXL at resolution 512 × 512. For LSUN bedrooms (Yu et al., 2015) and FFHQ (Karras et al., 2019), we use pre-trained latent difusion models (Rombach et al., 2022; Blattmann et al., 2022) with noise predictors.

Evaluation settings. We measure sample quality using FID computed from 50K generated samples, with lower scores indicating better quality. In the FID tables, bold indicates the lowest FID in each row. Computational budgets are reported as the number of neural function evaluations (NFE). For LSUN and FFHQ, we compute the reference statistics ourselves; the resulting FID scores may therefore difer from those reported in other studies. The reference statistics and the code used to generate them are available at https://github.com/Na-Li66/F-DPMSolver.

For EDM and EDM2, we directly construct the sampling noise levels using their native ρ-parameterized schedule. For the latent difusion models considered here, the noise predictor is trained at discrete noise levels specified by a reference schedule $\{ ( \alpha _ { j } ^ { \mathsf { r e f } } , \sigma _ { j } ^ { \mathsf { r e f } } ) \} _ { j = 0 } ^ { M ^ { \mathsf { r e f } } }$ . To perform sampling with M iterations, we select M + 1 levels from this schedule according to

$$
\begin{array} { r l } & { r _ { i } = \mathsf { r o u n d } \left( \displaystyle \frac { i M ^ { \mathsf { r e f } } } { M } \right) , } \\ & { \alpha _ { t _ { i } } = \alpha _ { r _ { i } } ^ { \mathsf { r e f } } , \qquad \sigma _ { t _ { i } } = \sigma _ { r _ { i } } ^ { \mathsf { r e f } } , \qquad i = 0 , \hdots , M . } \end{array}
$$

## 4.1 Evaluation of F-DDIM

We first evaluate whether F-DDIM, as a first-order sampler, can achieve good few-step sample quality.

Experimental setup. We compare F-DDIM with deterministic DDIM, a second-order DPM Solver (DPMSolver-2, see (31)), a third-order DPM Solver (DPMSolver-3, see (32)), and a third-order predictor-corrector-based sampler (UniPC-3, see (33b) and (33a)). All methods are evaluated at NFE budgets in {4, 5, 6, 8, 10}. The baseline update formulas are given in Appendix A.1.

Quantitative results (FID). Tables 1 and 2, together with Tables 7 and 8 in Appendix A, report the FID results. F-DDIM substantially improves upon DDIM across the tested settings. On ImageNet-64 and ImageNet-512, it achieves the lowest FID among the compared methods at all tested NFEs and model sizes reported in Tables 2 and 7. On unconditional CIFAR-10, F-DDIM achieves the lowest FID at NFE budgets 4, 5, 6, and 8. On conditional CIFAR-10, it achieves the lowest FID at budgets 4, 5, and 6, while UniPC-3 performs better at budgets 8 and 10. These results demonstrate that F-DDIM, as a first-order sampler, can achieve competitive few-step sample quality relative to higher-order deterministic baselines.

Table 1: FID on unconditional and conditional CIFAR-10 with EDM.
<table><tr><td rowspan="2">Model</td><td rowspan="2">NFE</td><td colspan="3">Higher-order Algorithms</td><td colspan="2">First-order Algorithms</td></tr><tr><td>DPMSolver-2</td><td>DPMSolver-3</td><td>UniPC-3</td><td>DDIM</td><td>F-DDIM (ours)</td></tr><tr><td rowspan="5">EDM -uncond</td><td>4</td><td>43.55</td><td>33.77</td><td>106.07</td><td>66.72</td><td>25.01</td></tr><tr><td>5</td><td>27.86</td><td>18.20</td><td>50.35</td><td>49.62</td><td>16.04</td></tr><tr><td>6</td><td>17.70</td><td>10.68</td><td>50.09</td><td>35.58</td><td>9.47</td></tr><tr><td>8</td><td>9.74</td><td>5.60</td><td>6.54</td><td>22.30</td><td>4.91</td></tr><tr><td>10</td><td>6.52</td><td>3.88</td><td>2.73</td><td>15.67</td><td>3.46</td></tr><tr><td rowspan="5">EDM -cond</td><td>4</td><td>31.60</td><td>23.99</td><td>73.52</td><td>47.27</td><td>18.44</td></tr><tr><td>5</td><td>20.63</td><td>13.68</td><td>36.29</td><td>35.57</td><td>12.29</td></tr><tr><td>6</td><td>14.04</td><td>8.66</td><td>32.85</td><td>26.78</td><td>7.57</td></tr><tr><td>8</td><td>8.34</td><td>5.00</td><td>4.02</td><td>17.57</td><td>4.32</td></tr><tr><td>10</td><td>5.89</td><td>3.61</td><td>2.44</td><td>13.00</td><td>3.18</td></tr></table>

Table 2: FID on ImageNet-512 with EDM2-XS/XXL.
<table><tr><td rowspan="2">Model</td><td rowspan="2">NFE</td><td colspan="3">Higher-order Algorithms</td><td colspan="2">First-order Algorithms</td></tr><tr><td>DPMSolver-2</td><td>DPMSolver-3</td><td>UniPC-3</td><td>DDIM</td><td>F-DDIM (ours)</td></tr><tr><td rowspan="5">EDM2 -XS</td><td>4</td><td>66.23</td><td>59.35</td><td>200.75</td><td>87.81</td><td>50.96</td></tr><tr><td>5</td><td>38.02</td><td>28.06</td><td>87.59</td><td>61.62</td><td>22.69</td></tr><tr><td>6</td><td>28.24</td><td>16.11</td><td>23.93</td><td>53.89</td><td>14.33</td></tr><tr><td>8</td><td>12.19</td><td>7.01</td><td>7.48</td><td>31.17</td><td>6.18</td></tr><tr><td>10</td><td>7.45</td><td>4.93</td><td>4.72</td><td>20.97</td><td>4.57</td></tr><tr><td rowspan="5">EDM2 -XXL</td><td>4</td><td>68.18</td><td>60.11</td><td>57.79</td><td>91.48</td><td>52.32</td></tr><tr><td>5</td><td>40.61</td><td>28.41</td><td>29.09</td><td>65.26</td><td>23.56</td></tr><tr><td>6</td><td>28.46</td><td>16.57</td><td>14.42</td><td>53.94</td><td>14.04</td></tr><tr><td>8</td><td>12.06</td><td>6.03</td><td>5.10</td><td>32.40</td><td>5.04</td></tr><tr><td>10</td><td>6.59</td><td>3.60</td><td>3.15</td><td>21.48</td><td>3.06</td></tr></table>

Additional results, including generated samples, runtime comparisons, larger NFE budgets, and higherorder lookahead predictors, are provided in Appendix A.

## 4.2 Evaluation of F-DDPM

Experimental setup. We compare F-DDPM with SDE-DPMSolver++1 (17a) and SDE-DPMSolver++2 (17b) at NFEs in {6, 9, 12, 15}. We also compare these stochastic samplers with DDIM, DPMSolver-3, and F-DDIM at NFEs in {4, 6, 8, 10}, respectively. Thus, each stochastic sampler uses 1.5 times the NFE of the deterministic methods in the corresponding table row. The noise schedules follow the subsampling procedure described above.

For the latent difusion models considered here, model evaluations are restricted to supported reference noise levels. At iteration i, we select the noise level whose signal-to-noise ratio is closest to that of the lookahead time $t _ { i } ^ { \prime } { \mathrm { : } }$

$$
j _ { i } ^ { \star } \in \underset { j \in \mathcal { I } _ { i } ^ { { \sf r e f } } } { \arg \operatorname* { m i n } } \left| \frac { \alpha _ { t _ { i } ^ { \prime } } } { \sigma _ { t _ { i } ^ { \prime } } } - \frac { \alpha _ { j } ^ { { \sf r e f } } } { \sigma _ { j } ^ { { \sf r e f } } } \right| ,
$$

where

$$
\mathcal { T } _ { i } ^ { \mathsf { r e f } } = \Big \{ 0 \leq j \leq M ^ { \mathsf { r e f } } : \frac { \alpha _ { j } ^ { \mathsf { r e f } } } { \sigma _ { j } ^ { \mathsf { r e f } } } \geq \frac { \alpha _ { t _ { i } } } { \sigma _ { t _ { i } } } \Big \} .
$$

The selected reference noise level is used for both the model evaluation and the corresponding F-DDPM update coeficients.

Comparison with stochastic baselines. Tables 3–6 compare the stochastic samplers across various NFEs. F-DDPM achieves the lowest FID among the three stochastic methods in every tested setting. For example, on unconditional CIFAR-10 with 9 model evaluations, F-DDPM achieves an FID of 8.63, compared with 55.52 for SDE-DPMSolver++1 and 17.89 for SDE-DPMSolver++2. These improvements are consistent across datasets, model sizes, conditional and unconditional settings, and pixel-space and latent-space generation.

Comparison with deterministic baselines. The same tables compare the stochastic samplers using 1.5B model evaluations with deterministic samplers using B evaluations, where $B \in \{ 4 , 6 , 8 , 1 0 \}$ . Under this budget allocation, F-DDPM achieves lower FID than all three deterministic baselines on unconditional CIFAR-10 and ImageNet-512. It also outperforms all three on conditional CIFAR-10 at $B = 4 , 6 , 8$ and on LSUN at $B = 6 , 8 , 1 0$ . Deterministic samplers retain an advantage on ImageNet-64 and in several FFHQ settings. Overall, these comparisons show that F-DDPM narrows the empirical sampling-eficiency gap with deterministic methods, while still using a larger NFE budget.

Table 3: FID on unconditional and conditional CIFAR-10.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Deter- ministic NFE</td><td colspan="3">Stochastic Algorithms (1.5×NFE)</td><td colspan="3">Deterministic Algorithms</td></tr><tr><td>SDE DPMSolver++1</td><td>SDE DPMSolver++2</td><td>F-DDPM (ours)</td><td>DDIM</td><td>DPMSolver-3</td><td>F-DDIM (ours)</td></tr><tr><td rowspan="3">CIFAR10 -uncond</td><td>4 6</td><td>77.58</td><td>37.24</td><td>21.85</td><td>67.03</td><td>33.77</td><td>25.02</td></tr><tr><td></td><td>55.52</td><td>17.89</td><td>8.63</td><td>35.89</td><td>10.69</td><td>9.47</td></tr><tr><td>8 10</td><td>41.79</td><td>10.55</td><td>4.62</td><td>22.46</td><td>5.60</td><td>4.92</td></tr><tr><td rowspan="4">CIFAR10 -cond</td><td>4</td><td>32.64 56.21</td><td>7.07 27.83</td><td>3.34 15.11</td><td>15.85 47.27</td><td>3.88 23.99</td><td>3.46 18.44</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>6</td><td>42.03</td><td>15.18</td><td>6.85</td><td>26.78</td><td>8.66</td><td>7.57</td></tr><tr><td>8 10</td><td>33.22 27.30</td><td>9.78 6.94</td><td>4.13 3.23</td><td>17.57 13.00</td><td>5.00 3.61</td><td>4.32 3.18</td></tr></table>

Table 4: FID on ImageNet-64 with EDM2-S/L.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Deter- ministic NFE</td><td colspan="3">Stochastic Algorithms (1.5×NFE)</td><td colspan="3">Deterministic Algorithms</td></tr><tr><td>SDE DPMSolver++1</td><td>SDE DPMSolver++2</td><td>F-DDPM (ours)</td><td>DDIM</td><td>DPMSolver-3</td><td>F-DDIM (ours)</td></tr><tr><td rowspan="3">ImageNet 64-S</td><td>4 6</td><td>49.04</td><td>25.95 13.65</td><td>24.01</td><td>39.37</td><td>21.49</td><td>20.55</td></tr><tr><td>8</td><td>34.50 25.77</td><td>8.68</td><td>8.63 4.38</td><td>20.65 12.56</td><td>6.96 3.74</td><td>6.67 3.50</td></tr><tr><td>10</td><td>19.94</td><td>6.16</td><td>3.06</td><td>8.76</td><td>2.70</td><td>2.56</td></tr><tr><td rowspan="4">ImageNet 64-L</td><td>4</td><td>53.86</td><td>27.89</td><td>25.63</td><td>43.86</td><td>23.66</td><td>22.35</td></tr><tr><td>6</td><td>38.77</td><td>14.58</td><td>8.90</td><td>23.22</td><td>7.63</td><td>7.20</td></tr><tr><td>8</td><td>29.19</td><td>9.21</td><td>4.30</td><td>14.20</td><td>3.92</td><td>3.64</td></tr><tr><td>10</td><td>22.76</td><td>6.47</td><td>3.06</td><td>9.85</td><td>2.70</td><td>2.51</td></tr></table>

Table 5: FID on ImageNet-512 with EDM2-XS/XXL.
<table><tr><td rowspan=2 colspan=2>Model</td><td rowspan=2 colspan=1>Deter-ministicNFE</td><td rowspan=1 colspan=4>Stochastic Algorithms (1.5×NFE)</td><td rowspan=1 colspan=3>Deterministic Algorithms</td></tr><tr><td rowspan=1 colspan=1>SDEDPMSolver++1</td><td rowspan=1 colspan=2>SDEDPMSolver++2</td><td rowspan=1 colspan=1>F-DDPM(ours)</td><td rowspan=1 colspan=1>DDIM</td><td rowspan=1 colspan=1>DPMSolver-3</td><td rowspan=1 colspan=1>F-DDIM(ours)</td></tr><tr><td rowspan=4 colspan=2>ImageNet512-XS</td><td rowspan=4 colspan=1>46810</td><td rowspan=3 colspan=1>102.6965.3844.63</td><td rowspan=1 colspan=2>45.22</td><td rowspan=1 colspan=1>37.86</td><td rowspan=1 colspan=1>87.78</td><td rowspan=1 colspan=1>59.34</td><td rowspan=1 colspan=1>50.96</td></tr><tr><td rowspan=2 colspan=2>15.607.84</td><td rowspan=1 colspan=1>10.86</td><td rowspan=2 colspan=1>53.8831.16</td><td rowspan=2 colspan=1>16.117.01</td><td rowspan=2 colspan=1>14.336.18</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>5.75</td></tr><tr><td rowspan=1 colspan=1>31.97</td><td rowspan=1 colspan=2>5.55</td><td rowspan=1 colspan=1>4.47</td><td rowspan=1 colspan=1>20.97</td><td rowspan=1 colspan=1>4.93</td><td rowspan=1 colspan=1>4.57</td></tr><tr><td rowspan=5 colspan=2>ImageNet512-XXL</td><td rowspan=2 colspan=1>46</td><td rowspan=1 colspan=1>100.40</td><td rowspan=1 colspan=2>40.10</td><td rowspan=1 colspan=1>36.71</td><td rowspan=1 colspan=1>91.45</td><td rowspan=1 colspan=1>60.11</td><td rowspan=1 colspan=1>52.32</td></tr><tr><td rowspan=1 colspan=1>62.68</td><td rowspan=1 colspan=2>12.95</td><td rowspan=1 colspan=1>9.87</td><td rowspan=1 colspan=1>53.94</td><td rowspan=1 colspan=1>16.57</td><td rowspan=1 colspan=1>14.04</td></tr><tr><td rowspan=2 colspan=1></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>42.31</td><td rowspan=1 colspan=2>5.76</td><td rowspan=1 colspan=1>4.18</td><td rowspan=1 colspan=1>32.40</td><td rowspan=1 colspan=1>6.03</td><td rowspan=1 colspan=1>5.04</td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>30.33</td><td rowspan=1 colspan=2>3.82</td><td rowspan=1 colspan=1>2.92</td><td rowspan=1 colspan=1>21.48</td><td rowspan=1 colspan=1>3.60</td><td rowspan=1 colspan=1>3.06</td></tr></table>

Table 6: FID on LSUN bedrooms and FFHQ with latent difusion models.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Deter- ministic NFE</td><td colspan="3">Stochastic Algorithms (1.5×NFE)</td><td colspan="3">Deterministic Algorithms</td></tr><tr><td>SDE DPMSolver++1</td><td>SDE DPMSolver++2</td><td>F-DDPM (ours)</td><td>DDIM</td><td>DPMSolver-3</td><td>F-DDIM (ours)</td></tr><tr><td rowspan="4">LSUN</td><td>4</td><td>89.39</td><td>15.40</td><td>15.09</td><td>70.55</td><td>10.13</td><td>12.17</td></tr><tr><td>6</td><td>41.82</td><td>12.32</td><td>5.17</td><td>26.15</td><td>9.30</td><td>5.56</td></tr><tr><td>8</td><td>23.88</td><td>13.23</td><td>4.50</td><td>13.70</td><td>9.24</td><td>4.92</td></tr><tr><td>10</td><td>15.54</td><td>13.86</td><td>4.47</td><td>9.00</td><td>8.38</td><td>4.65</td></tr><tr><td rowspan="4">FFHQ</td><td>4</td><td>114.77</td><td>23.55</td><td>22.57</td><td>80.19</td><td>15.65</td><td>15.78</td></tr><tr><td>6</td><td>76.33</td><td>27.74</td><td>9.04</td><td>43.68</td><td>11.55</td><td>7.89</td></tr><tr><td>8</td><td>53.57</td><td>30.47</td><td>7.33</td><td>27.00</td><td>12.09</td><td>6.80</td></tr><tr><td>10</td><td>39.48</td><td>31.68</td><td>7.31</td><td>18.77</td><td>11.63</td><td>6.46</td></tr></table>

Qualitative comparisons. Figures 3 and 4 show representative samples on ImageNet-64 and FFHQ. Additional examples on CIFAR-10, ImageNet-512, and LSUN are provided in Figures 13, 14, 15, and 16 in the appendix. Within each comparison, the stochastic samplers use the same initial noise and Brownian motion. At small NFEs, F-DDPM often produces clearer structures and finer visual details than the stochastic baselines, consistent with the FID comparisons. For example, the ImageNet-64 samples show more distinct object boundaries and background details.

NFE=6  
9  
12  
![](images/eed9480e3bbc1445eb3370708a1d26dd3db5ad437f4539c994126d8288ec1f9f.jpg)  
15  
9  
NFE=6  
(a) ImageNet64, EDM2-S  
12

9  
NFE=6  
12  
15  
![](images/7c2a130fb3d56f4bf03f05f123035d592b1d493939d8234aa6cddcfecc5931b2.jpg)  
(b) ImageNet64, EDM2-L  
Figure 3: Qualitative comparison of F-DDPM, SDE-DPMSolver++1, and SDE-DPMSolver++2 on ImageNet-64 using EDM2-S and EDM2-L. All methods use the NFE indicated above each column.

15  
9  
NFE=6  
12  
15  
![](images/7c1dc095abc6772c6ffaf141f1b74b8264f0a2129a06c8df3a16d76b5d507e3d.jpg)  
Figure 4: Qualitative comparison of F-DDPM, SDE-DPMSolver++1, and SDE-DPMSolver++2 on FFHQ using a pre-trained latent difusion model. All methods use the NFE indicated above each column.

## 5 Discussion

We studied forward-value discretization as a training-free approach for accelerating difusion sampling. Our analysis establishes first-order strong error bounds and matching worst-case lower bounds for SDE-DPMSolver++1 and SDE-DPMSolver++2. Thus, the higher-order approximation in SDE-DPMSolver++2 does not improve its strong convergence order. Together with the first-order characterization of F-DDIM and its competitive few-step sample quality, these results highlight model evaluation placement as a design principle that complements increasing the convergence order.

Building on the deterministic forward-value framework, we developed F-DDPM by accounting for noise injection in the choice of evaluation time and the construction of the lookahead estimate. The method requires one model evaluation per iteration and admits a first-order strong convergence guarantee. Across the tested settings, F-DDPM achieves lower FID than the stochastic baselines at the same NFE budgets. With 1.5 times the NFE of the deterministic baselines, it also achieves competitive sample quality in many settings, although deterministic methods retain an advantage in others. These results demonstrate the practical potential of forward-value evaluation for reducing the eficiency gap between stochastic and deterministic sampling.

Several questions remain open. First, the convergence guarantees do not explain why changing evaluation placement improves few-step sample quality. Second, the behavior of the method under classifier guidance and classifier-free guidance deserves further study, where guidance strength can amplify errors and alter the efective dynamics. Finally, applications to inverse problems, such as deblurring and super-resolution, require further investigation of how forward-value updates interact with measurement consistency constraints.

## Acknowledgements

G. Li is supported in part by the Chinese University of Hong Kong Direct Grant for Research and the Hong Kong Research Grants Council ECS 24305724 and GRF 14307525. C. Cai is supported by the NSF CAREER Award CCF-2541600 and the NSF grant DMS-2515333. Y. Wei is supported in part by the NSF grants CCF-2106778, CCF-2418156 and CAREER award DMS-2143215, also by the NSF under Cooperative Agreement No. 2433450.

## References

Anderson, B. D. (1982). Reverse-time difusion equation models. Stochastic Processes and their Applications, 12(3):313–326.

Bao, F., Li, C., Zhu, J., and Zhang, B. (2022). Analytic-dpm: an analytic estimate of the optimal reverse variance in difusion probabilistic models. arXiv preprint arXiv:2201.06503.

Benton, J., De Bortoli, V., Doucet, A., and Deligiannidis, G. (2023). Nearly d-linear convergence bounds for difusion models via stochastic localization. arXiv preprint arXiv:2308.03686.

Blattmann, A., Rombach, R., Oktay, K., Müller, J., and Ommer, B. (2022). Retrieval-augmented difusion models. Advances in Neural Information Processing Systems, 35:15309–15324.

Cai, C., Jiao, Y., and Li, G. (2026). Difusion models adapt to low-dimensional structure under flexible coeficient choices. arXiv preprint arXiv:2606.23627.

Cai, C. and Li, G. (2026). Confidence-based decoding is provably eficient for difusion language models. arXiv preprint arXiv:2603.22248.

Chen, H., Lee, H., and Lu, J. (2023a). Improved analysis of score-based generative modeling: User-friendly bounds under minimal smoothness assumptions. In International Conference on Machine Learning, pages 4735–4763. PMLR.

Chen, H., Ren, Y., Ying, L., and Rotskof, G. M. (2024). Accelerating difusion models with parallel sampling: Inference at sub-linear time complexity. arXiv preprint arXiv:2405.15986.

Chen, S., Chewi, S., Lee, H., Li, Y., Lu, J., and Salim, A. (2023b). The probability flow ode is provably fast. Advances in Neural Information Processing Systems, 36:68552–68575.

Chen, S., Chewi, S., Li, J., Li, Y., Salim, A., and Zhang, A. R. (2022). Sampling is as easy as learning the score: theory for difusion models with minimal data assumptions. arXiv preprint arXiv:2209.11215.

Chen, S., Cong, K., and Li, J. (2025). Optimal inference schedules for masked difusion models. arXiv preprint arXiv:2511.04647.

Chen, S. and Wang, L. (2026). Parallelism, critical windows, and separations among difusion language models. arXiv preprint arXiv:2609.20539.

Croitoru, F.-A., Hondru, V., Ionescu, R. T., and Shah, M. (2023). Difusion models in vision: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. (2009). Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee.

Dhariwal, P. and Nichol, A. (2021). Difusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794.

Dmitriev, D., Huang, Z., and Wei, Y. (2026a). Eficient sampling with discrete difusion models: Sharp and adaptive guarantees. arXiv preprint arXiv:2602.15008.

Dmitriev, D., Huang, Z., and Wei, Y. (2026b). Provably adaptive sampling with uniform and remasking discrete difusion models. arXiv preprint arXiv:2608.23554.

Gonzalez, M., Fernandez, N., Tran, T., Gherbi, E., Hajri, H., and Masmoudi, N. (2023). Seeds: Exponential sde solvers for fast high-quality sampling from difusion models.

Gupta, S., Cai, L., and Chen, S. (2024). Faster difusion-based sampling with randomized midpoints: Sequential and parallel. arXiv preprint arXiv:2406.00924.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., and Hochreiter, S. (2017). Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30.

Ho, J., Jain, A., and Abbeel, P. (2020). Denoising difusion probabilistic models. Advances in neural information processing systems, 33:6840–6851.

Hoogeboom, E., Satorras, V. G., Vignac, C., and Welling, M. (2022). Equivariant difusion for molecule generation in 3d. In International conference on machine learning, pages 8867–8887. PMLR.

Huang, D. Z., Huang, J., and Lin, Z. (2024a). Convergence analysis of probability flow ODE for score-based generative models. arXiv preprint arXiv:2404.09730.

Huang, X., Zou, D., Dong, H., Zhang, Y., Ma, Y.-A., and Zhang, T. (2024b). Reverse transition kernel: A flexible framework to accelerate difusion inference. arXiv preprint arXiv:2405.16387.

Jiao, Y. and Li, G. (2024). Instance-dependent convergence theory for difusion models. arXiv preprint arXiv:2410.13738.

Jiao, Y., Li, N., Cai, C., and Li, G. (2026). Are first-order difusion samplers really slower? a fast forwardvalue approach. In International Conference on Machine Learning, volume 2026, pages 1–9.

Jiao, Y., Zhou, Y., and Li, G. (2025). Optimal convergence analysis of ddpm for general distributions. arXiv preprint arXiv:2510.27562.

Jolicoeur-Martineau, A., Li, K., Piché-Taillefer, R., Kachman, T., and Mitliagkas, I. (2021). Gotta go fast when generating data with score-based models. arXiv preprint arXiv:2105.14080.

Karras, T., Aittala, M., Aila, T., and Laine, S. (2022). Elucidating the design space of difusion-based generative models. In Proc. NeurIPS.

Karras, T., Laine, S., and Aila, T. (2019). A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410.

Kingma, D., Salimans, T., Poole, B., and Ho, J. (2021). Variational difusion models. Advances in neural information processing systems, 34:21696–21707.

Kong, Z., Ping, W., Huang, J., Zhao, K., and Catanzaro, B. (2021). DifWave: A versatile difusion model for audio synthesis. In International Conference on Learning Representations.

Krizhevsky, A., Hinton, G., et al. (2009). Learning multiple layers of features from tiny images.

Lai, C.-H., Song, Y., Kim, D., Mitsufuji, Y., and Ermon, S. (2025). The principles of difusion models. arXiv preprint arXiv:2510.21890.

Lee, H., Lu, J., and Tan, Y. (2023). Convergence of score-based generative modeling for general data distributions. In International Conference on Algorithmic Learning Theory, pages 946–985. PMLR.

Li, G. and Cai, C. (2024). Provable acceleration for difusion models under minimal assumptions. arXiv preprint arXiv:2410.23285.

Li, G. and Cai, C. (2025). Breaking AR’s sampling bottleneck: Provable acceleration via difusion language models. Advances in Neural Information Processing Systems, 38:11700–11725.

Li, G., Cai, C., and Wei, Y. (2025a). Dimension-free convergence of difusion models for approximate gaussian mixtures. arXiv preprint arXiv:2504.05300.

Li, G., Huang, Y., Efimov, T., Wei, Y., Chi, Y., and Chen, Y. (2024a). Accelerating convergence of scorebased difusion models, provably. arXiv preprint arXiv:2403.03852.

Li, G., Huang, Z., and Wei, Y. (2024b). Towards a mathematical theory for consistency training in difusion models. arXiv preprint arXiv:2402.07802.

Li, G. and Jiao, Y. (2024). Improved convergence rate for difusion probabilistic models. In The Thirteenth International Conference on Learning Representations.

Li, G., Wei, Y., Chen, Y., and Chi, Y. (2023). Towards faster non-asymptotic convergence for difusion-based generative models. arXiv preprint arXiv:2306.09251.

Li, G. and Yan, Y. (2024). O(d/T) convergence theory for difusion probabilistic models under minimal assumptions. arXiv preprint arXiv:2409.18959.

Li, G., Zhou, Y., Wei, Y., and Chen, Y. (2025b). Faster difusion models via higher-order approximation. arXiv preprint arXiv:2506.24042.

Li, N., Jiao, Y., Cai, C., and Li, G. (2026). Convergeflow: Language flow with provable convergence to token embeddings. arXiv preprint arXiv:2608.23551.

Li, X., Thickstun, J., Gulrajani, I., Liang, P. S., and Hashimoto, T. B. (2022). Difusion-lm improves controllable text generation. Advances in Neural Information Processing Systems, 35:4328–4343.

Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., and Zhu, J. (2022a). Dpm-solver: A fast ode solver for difusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 35:5775–5787.

Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., and Zhu, J. (2022b). Dpm-solver++: Fast solver for guided sampling of difusion probabilistic models. arXiv preprint arXiv:2211.01095.

Luhman, E. and Luhman, T. (2021). Knowledge distillation in iterative generative models for improved sampling speed. arXiv preprint arXiv:2101.02388.

Meng, C., Rombach, R., Gao, R., Kingma, D., Ermon, S., Ho, J., and Salimans, T. (2023). On distillation of guided difusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14297–14306.

Nichol, A. Q. and Dhariwal, P. (2021). Improved denoising difusion probabilistic models. In International Conference on Machine Learning, pages 8162–8171.

Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., and Chen, M. (2022). Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. (2022). High-resolution image synthesis with latent difusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695.

Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E. L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al. (2022). Photorealistic text-to-image difusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494.

Sahoo, S., Arriola, M., Schif, Y., Gokaslan, A., Marroquin, E., Chiu, J., Rush, A., and Kuleshov, V. (2024). Simple and efective masked difusion language models. Advances in Neural Information Processing Systems, 37:130136–130184.

Salimans, T. and Ho, J. (2022). Progressive distillation for fast sampling of difusion models. arXiv preprint arXiv:2202.00512.

Shih, A., Belkhale, S., Ermon, S., Sadigh, D., and Anari, N. (2023). Parallel sampling of difusion models. Advances in Neural Information Processing Systems, 36:4263–4276.

Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., and Ganguli, S. (2015). Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. pmlr.

Song, J., Meng, C., and Ermon, S. (2020a). Denoising difusion implicit models. arXiv preprint arXiv:2010.02502.

Song, Y., Dhariwal, P., Chen, M., and Sutskever, I. (2023). Consistency models. arXiv preprint arXiv:2303.01469.

Song, Y. and Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. (2020b). Score-based generative modeling through stochastic diferential equations. arXiv preprint arXiv:2011.13456.

Wu, Y., Chen, Y., and Wei, Y. (2024). Stochastic runge-kutta methods: Provable acceleration of difusion models. arXiv preprint arXiv:2410.04760.

Xue, S., Yi, M., Luo, W., Zhang, S., Sun, J., Li, Z., and Ma, Z.-M. (2024). Sa-solver: Stochastic adams solver for fast sampling of difusion models. Advances in Neural Information Processing Systems, 36.

Yang, L., Zhang, Z., Song, Y., Hong, S., Xu, R., Zhao, Y., Zhang, W., Cui, B., and Yang, M.-H. (2023). Difusion models: A comprehensive survey of methods and applications. ACM Computing Surveys, 56(4):1– 39.

Yu, F., Sef, A., Zhang, Y., Song, S., Funkhouser, T., and Xiao, J. (2015). Lsun: Construction of a large-scale image dataset using deep learning with humans in the loop. arXiv preprint arXiv:1506.03365.

Yu, Y. and Yu, L. (2025). Advancing wasserstein convergence analysis of score-based models: Insights from discretization and second-order acceleration. arXiv preprint arXiv:2502.04849.

Zhang, J., Liu, Z., Yan, L., Li, G., and Gu, Y. (2025). Improving difusion-based inverse algorithms under few-step constraint via learnable linear extrapolation. arXiv preprint arXiv:2503.10103.

Zhang, J., Xia, M., Li, G., and Gu, Y. (2026). Distilling drifting transformers with representation autoencoders. arXiv preprint arXiv:2606.15553.

Zhang, Q. and Chen, Y. (2022). Fast sampling of difusion models with exponential integrator. arXiv preprint arXiv:2204.13902.

Zhao, W., Bai, L., Rao, Y., Zhou, J., and Lu, J. (2023). Unipc: A unified predictor-corrector framework for fast sampling of difusion models. Advances in Neural Information Processing Systems, 36:49842–49869.

Zhao, Y. and Cai, C. (2026). Adaptation to intrinsic dependence in difusion language models. arXiv preprint arXiv:2602.20126.

Zheng, K., Lu, C., Chen, J., and Zhu, J. (2023). Dpm-solver-v3: Improved difusion ode solver with empirical model statistics. Advances in Neural Information Processing Systems, 36:55502–55542.

## A Additional sampler details and experimental results

## A.1 Baseline deterministic samplers

We describe the deterministic baseline samplers used in Section 4.1.

Multistep DPMSolver updates. For the noise-prediction-based multistep solvers, we write $\pmb { \varepsilon } _ { t _ { j } } = \pmb { \varepsilon } _ { \theta } ( \pmb { x } _ { t _ { j } } , t _ { j } )$ To construct a p-th order update, we approximate the noise predictor along the trajectory by a polynomia of degree at most $p - 1$ , interpolating the previous p model evaluations. Integrating this polynomial in (12a) gives (Lu et al., 2022a)

$$
\begin{array} { r l } & { { \pmb x } _ { t _ { i } } = \displaystyle \mathrm { O D E S o l v e r p } ( { \pmb x } _ { t _ { i - 1 } } , { \pmb \varepsilon } _ { t _ { i - 1 } } , \cdots , { \pmb \varepsilon } _ { t _ { i - p } } ) } \\ & { \quad \quad : = \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } { \pmb x } _ { t _ { i - 1 } } - \alpha _ { t _ { i } } \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } \displaystyle \sum _ { k = 0 } ^ { p - 1 } \frac { \varphi _ { k } \left( \delta _ { i } \right) } { k ! } { \pmb y } _ { k + 1 } , \quad i \geq p , } \end{array}\tag{30}
$$

where scalar functions $\varphi _ { k }$ are defined by

$$
\begin{array} { l } { \displaystyle \varphi _ { 0 } ( x ) = 1 - \mathrm { e } ^ { - x } , } \\ { \displaystyle \varphi _ { k } ( x ) = \int _ { 0 } ^ { x } \mathrm { e } ^ { - \lambda } \lambda ^ { k } \mathrm { d } \lambda = k \varphi _ { k - 1 } ( x ) - x ^ { k } \mathrm { e } ^ { - x } , \quad k = 1 , \cdots , p - 1 , } \end{array}
$$

The vectors $\mathbf { \pmb { y } } _ { k + 1 }$ define the interpolating polynomial

$$
P _ { i } ( \lambda ) = \sum _ { k = 0 } ^ { p - 1 } \frac { ( \lambda - \lambda _ { t _ { i - 1 } } ) ^ { k } } { k ! } \pmb { y } _ { k + 1 } ,
$$

which satisfies $P _ { i } ( \lambda _ { t _ { i - j } } ) = \pmb { \varepsilon } _ { t _ { i - j } } \ \mathrm { f o r } \ j = 1 , \ldots , p .$ . Equivalently, the matrix $\pmb { Y } = [ \pmb { y } _ { 1 } , \dots , \pmb { y } _ { p } ] ^ { \top } \in \mathbb { R } ^ { p \times d }$ solves

$$
\pmb { A } \pmb { Y } = \pmb { B } , \quad \pmb { A } = \left[ \frac { ( \lambda _ { t _ { i - j } } - \lambda _ { t _ { i - 1 } } ) ^ { k - 1 } } { ( k - 1 ) ! } \right] _ { 1 \leq j , k \leq p } \in \mathbb { R } ^ { p \times p } , \quad \mathrm { a n d } \quad \pmb { B } = [ \pmb { \varepsilon } _ { t _ { i - j } } ^ { \top } ] _ { 1 \leq j \leq p } \in \mathbb { R } ^ { p \times d } .
$$

The cases $p = 2$ and $p = 3$ give the following DPMSolver-2 and DPMSolver-3 updates.

• DPMSolver-2. The first state $\pmb { x } _ { t _ { 1 } }$ is initialized using one DDIM step (13). For $i = 2 , \dots , M$ , the update is

$$
\begin{array} { r l } & { { \pmb x } _ { t _ { i } } = \mathrm { O D E S o l v e r } 2 ( { \pmb x } _ { t _ { i - 1 } } , { \pmb \varepsilon } _ { t _ { i - 1 } } , { \pmb \varepsilon } _ { t _ { i - 2 } } ) } \\ & { \quad \quad : = \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } { \pmb x } _ { t _ { i - 1 } } + \alpha _ { t _ { i } } \Big [ \big ( \mathrm { e } ^ { - \lambda _ { t _ { i } } } - \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } \big ) { \pmb \varepsilon } _ { t _ { i - 1 } } + \phi _ { 1 , i } { \pmb D } _ { t _ { i } } \Big ] , } \end{array}\tag{31a}
$$

where

$$
D _ { t _ { i } } : = \frac { \varepsilon _ { t _ { i - 1 } } - \varepsilon _ { t _ { i - 2 } } } { \delta _ { i - 1 } } , \qquad \phi _ { 1 , i } : = ( \delta _ { i } + 1 ) \mathrm { e } ^ { - \lambda _ { t _ { i } } } - \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } , \qquad i \ge 2 .\tag{31b}
$$

• DPMSolver-3. The first state $\pmb { x } _ { t _ { 1 } }$ is initialized using one DDIM step (13), and the second state $\mathbf { \boldsymbol { x } } _ { t _ { 2 } }$ is computed using DPMSolver-2 (31) with $\varepsilon _ { t _ { 0 } }$ and $\varepsilon _ { t _ { 1 } }$ . Once three previous model evaluations are available, the update for i = 3, . . . , M is

$$
\begin{array} { r l } & { x _ { t _ { i } } = \boldsymbol { \mathrm { O D E S o l v e r 3 } } ( x _ { t _ { i - 1 } } , \varepsilon _ { t _ { i - 1 } } , \varepsilon _ { t _ { i - 2 } } , \varepsilon _ { t _ { i - 3 } } ) } \\ & { \quad \quad : = \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } x _ { t _ { i - 1 } } + \alpha _ { t _ { i } } \Bigg [ \big ( \mathrm { e } ^ { - \lambda _ { t _ { i } } } - \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } \big ) \varepsilon _ { t _ { i - 1 } } } \\ & { \qquad \quad \quad + \phi _ { 1 , i } \frac { \big ( \lambda _ { t _ { i - 1 } } - \lambda _ { t _ { i - 3 } } \big ) D _ { 1 , t _ { i } } - \big ( \lambda _ { t _ { i - 1 } } - \lambda _ { t _ { i - 2 } } \big ) D _ { 2 , t _ { i } } } { \lambda _ { t _ { i - 2 } } - \lambda _ { t _ { i - 3 } } } + \phi _ { 2 , i } \frac { D _ { 1 , t _ { i } } - D _ { 2 , t _ { i } } } { \lambda _ { t _ { i - 2 } } - \lambda _ { t _ { i - 3 } } } \Bigg ] , } \end{array}\tag{32a}
$$

where $\phi _ { 1 , i }$ is defined in (31b), and

$$
D _ { 1 , t _ { i } } : = \frac { \varepsilon _ { t _ { i - 1 } } - \varepsilon _ { t _ { i - 2 } } } { \lambda _ { t _ { i - 1 } } - \lambda _ { t _ { i - 2 } } } , \qquad D _ { 2 , t _ { i } } : = \frac { \varepsilon _ { t _ { i - 1 } } - \varepsilon _ { t _ { i - 3 } } } { \lambda _ { t _ { i - 1 } } - \lambda _ { t _ { i - 3 } } } , \qquad \phi _ { 2 , i } : = \delta _ { i } ^ { 2 } \mathrm { e } ^ { - \lambda _ { t _ { i } } } + 2 \phi _ { 1 , i } , \qquad i \ge 3 .\tag{32b}
$$

UniPC-3. UniPC-3 combines a predictor with a corrector (Zhao et al., 2023). Let ${ \mathbf { \mathcal { x } } } _ { t _ { i } }$ denote the predicted state and $\pmb { x } _ { t _ { i } } ^ { \mathsf { c o r } }$ the corrected state. The model is evaluated at the predicted state, $\pmb { \varepsilon } _ { t _ { i } } = \pmb { \varepsilon } _ { \theta } ( \pmb { x } _ { t _ { i } } , t _ { i } )$ . The corrected initialization is $\pmb { x } _ { t _ { 0 } } ^ { \mathsf { c o r } } = \pmb { x } _ { t _ { 0 } }$ . In the first step, $\pmb { x } _ { t _ { 1 } }$ is computed using one DDIM step (13) from $\mathbf { \nabla } _ { \mathbf { x } _ { t _ { 0 } } } .$ and $\pmb { x } _ { t _ { 1 } } ^ { \mathsf { c o r } }$ is computed using DPMSolver-2 (31) with $\varepsilon _ { t _ { 0 } }$ and $\varepsilon _ { t _ { 1 } }$ . For $i \geq 2$ , the corrected state $\pmb { x } _ { t _ { i - 1 } } ^ { \mathsf { c o r } }$ is used to compute the next predicted state $\mathbf { \mathcal { x } } _ { t _ { i } }$ using DPMSolver-2 (31) with $\varepsilon _ { t _ { i - 1 } }$ and $\varepsilon _ { t _ { i - 2 } } .$

$$
\pmb { x } _ { t _ { i } } = { \tt O D E S o l v e r 2 } ( \pmb { x } _ { t _ { i - 1 } } ^ { \mathrm { c o r } } , \pmb { \varepsilon } _ { t _ { i - 1 } } , \pmb { \varepsilon } _ { t _ { i - 2 } } ) ,\tag{33a}
$$

where ODESolver2 is given in (31). We then evaluate $\pmb { \varepsilon } _ { t _ { i } } = \pmb { \varepsilon } _ { \theta } ( \pmb { x } _ { t _ { i } } , t _ { i } )$ and use this output to correct the current interval:

$$
\pmb { x } _ { t _ { i } } ^ { \mathrm { c o r } } = \mathrm { O D E S o l v e r 3 } ( \pmb { x } _ { t _ { i - 1 } } ^ { \mathrm { c o r } } , \pmb { \varepsilon } _ { t _ { i } } , \pmb { \varepsilon } _ { t _ { i - 1 } } , \pmb { \varepsilon } _ { t _ { i - 2 } } ) ,\tag{33b}
$$

where ODESolver3 is defined in (32). At the final step, we return the predicted state $\mathbf { \boldsymbol { x } } _ { t _ { M } }$ without evaluating the model at $t _ { M }$ or applying the corrector. Including the initial model evaluation at $t _ { 0 } .$ , the total NFE is M.

## A.2 Additional results for F-DDIM

Quantitative results (FID). Tables 7 and 8 supplement the results in Section 4.1. On ImageNet-64, F-DDIM achieves the lowest FID among the compared methods across all tested NFEs and model sizes. On FFHQ, it achieves the lowest FID at all tested NFEs except NFE= 4, where DPMSolver-3 performs slightly better. On LSUN bedrooms, F-DDIM performs best at NFE = 5, 6, while DPMSolver-3 performs best at NFE = 4 and DPMSolver-2 at NFE = 8, 10. Across these settings, F-DDIM consistently improves upon DDIM and remains competitive with higher-order methods.

Qualitative comparisons. Figures 5–10 show additional samples on ImageNet-64, ImageNet-512, FFHQ, LSUN bedrooms, and CIFAR-10. The comparisons cover conditional and unconditional generation using both pixel-space and latent-space models. At the same NFE, F-DDIM often produces clearer structures and finer details than DDIM. In the examples shown, the higher-order solvers (DPMSolver- $- 2 / 3$ and UniPC-3) more often exhibit localized artifacts, particularly with few sampling steps.

Table 7: FID on ImageNet-64 with EDM2-S/L.
<table><tr><td rowspan="2">Model</td><td rowspan="2">NFE</td><td colspan="3">Higher-order Algorithms</td><td colspan="2">First-order Algorithms</td></tr><tr><td>DPMSolver-2</td><td>DPMSolver-3</td><td>UniPC-3</td><td>DDIM</td><td>F-DDIM (ours)</td></tr><tr><td rowspan="5">EDM2 -S</td><td>4</td><td>27.00</td><td>21.50</td><td>54.41</td><td>39.38</td><td>20.55</td></tr><tr><td>5</td><td>16.14</td><td>11.16</td><td>28.88</td><td>28.11</td><td>10.70</td></tr><tr><td>6</td><td>10.66</td><td>6.96</td><td>16.22</td><td>20.65</td><td>6.67</td></tr><tr><td>8</td><td>5.83</td><td>3.74</td><td>6.13</td><td>12.56</td><td>3.50</td></tr><tr><td>10</td><td>3.98</td><td>2.71</td><td>3.05</td><td>8.77</td><td>2.56</td></tr><tr><td rowspan="5">EDM2 -L</td><td>4</td><td>29.91</td><td>23.66</td><td>49.99</td><td>43.87</td><td>22.36</td></tr><tr><td>5</td><td>18.16</td><td>12.57</td><td>26.91</td><td>31.41</td><td>11.98</td></tr><tr><td>6</td><td>11.90</td><td>7.63</td><td>15.25</td><td>23.22</td><td>7.21</td></tr><tr><td>8</td><td>6.40</td><td>3.92</td><td>5.77</td><td>14.20</td><td>3.64</td></tr><tr><td>10</td><td>4.26</td><td>2.71</td><td>2.71</td><td>9.85</td><td>2.51</td></tr></table>

Table 8: FID on FFHQ and LSUN bedrooms with latent difusion models.
<table><tr><td rowspan=2 colspan=1>Dataset</td><td rowspan=2 colspan=1>NFE</td><td rowspan=1 colspan=3>Higher-order Algorithms</td><td rowspan=1 colspan=2>First-order Algorithms</td></tr><tr><td rowspan=1 colspan=1>DPMSolver-2</td><td rowspan=1 colspan=1>DPMSolver-3</td><td rowspan=1 colspan=1>UniPC-3</td><td rowspan=1 colspan=1>DDIM</td><td rowspan=1 colspan=1>F-DDIM (ours)</td></tr><tr><td rowspan=5 colspan=1>FFHQ</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>31.26</td><td rowspan=1 colspan=1>15.67</td><td rowspan=1 colspan=1>77.43</td><td rowspan=1 colspan=1>80.30</td><td rowspan=1 colspan=1>15.79</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>16.33</td><td rowspan=1 colspan=1>11.28</td><td rowspan=1 colspan=1>21.93</td><td rowspan=1 colspan=1>58.42</td><td rowspan=1 colspan=1>9.75</td></tr><tr><td rowspan=3 colspan=1>6810</td><td rowspan=3 colspan=1>10.477.196.58</td><td rowspan=1 colspan=1>11.53</td><td rowspan=1 colspan=1>9.96</td><td rowspan=1 colspan=1>43.75</td><td rowspan=1 colspan=1>7.89</td></tr><tr><td rowspan=1 colspan=1>12.08</td><td rowspan=1 colspan=1>7.37</td><td rowspan=1 colspan=1>27.05</td><td rowspan=1 colspan=1>6.80</td></tr><tr><td rowspan=1 colspan=1>11.62</td><td rowspan=1 colspan=1>6.82</td><td rowspan=1 colspan=1>18.81</td><td rowspan=1 colspan=1>6.46</td></tr><tr><td rowspan=5 colspan=1>LSUNBedroom</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>18.96</td><td rowspan=1 colspan=1>10.12</td><td rowspan=1 colspan=1>121.75</td><td rowspan=1 colspan=1>70.47</td><td rowspan=1 colspan=1>12.18</td></tr><tr><td rowspan=4 colspan=1>56810</td><td rowspan=3 colspan=1>8.666.024.78</td><td rowspan=1 colspan=1>8.52</td><td rowspan=1 colspan=1>24.74</td><td rowspan=1 colspan=1>41.01</td><td rowspan=1 colspan=1>6.98</td></tr><tr><td rowspan=1 colspan=1>9.33</td><td rowspan=1 colspan=1>9.41</td><td rowspan=1 colspan=1>26.12</td><td rowspan=1 colspan=1>5.57</td></tr><tr><td rowspan=1 colspan=1>9.26</td><td rowspan=1 colspan=1>5.72</td><td rowspan=1 colspan=1>13.68</td><td rowspan=1 colspan=1>4.93</td></tr><tr><td rowspan=1 colspan=1>4.49</td><td rowspan=1 colspan=1>8.41</td><td rowspan=1 colspan=1>4.83</td><td rowspan=1 colspan=1>9.00</td><td rowspan=1 colspan=1>4.66</td></tr></table>

DDIM DPM Solver-2Solver-3UniPC-3 DPM F-DDIM (ours)

5  
8  
NFE=4  
5  
6  
10  
![](images/bd8256f62d54441c5e96ce4743bd9fa9f7aaddfd2e79605774d21b3204e180f6.jpg)  
NFE=4  
(a) ImageNet64, EDM2-S

6

8

10  
![](images/5685ae94c756cf2f50445dca6a2cda30a78fa726b748a5d7fde9eb98858adf80.jpg)  
(b) ImageNet64, EDM2-L  
Figure 5: Qualitative comparisons between F-DDIM (ours) and DDIM, DPMSolver-2, DPMSolver-3, and UniPC-3. Images are sampled from pre-trained EDM2 with S and L size on ImageNet64 dataset.

5  
NFE=4  
6  
8  
10  
![](images/87810db1fa8df26afdb5b3c5c4098807debfd4e32579cc7c446843c7f6554684.jpg)  
(a) ImageNet512, EDM2-XS

NFE=4  
5  
6  
8  
10  
![](images/a85da488fffc314c9cd9fe93a3e667e9dbcd23bf3a7ffa7697a4f626dfee8059.jpg)  
(b) ImageNet512, EDM2-XXL  
Figure 6: Qualitative comparisons between F-DDIM (ours) and DDIM, DPMSolver-2, DPMSolver-3, and UniPC-3. Images are sampled from pre-trained EDM2 with XS and XXL size on ImageNet512 dataset.

![](images/b646bec8fd1c09fc1101202082907a8b3bc7996f7999547d2fb7d296b2cd3fc4.jpg)  
DDIM, NFE=5

![](images/ec52d6f7117bd3828288a38ca640658cdbb517a01910ededd6ba2760f57a5d5d.jpg)  
DPMSolver-2, NFE=5

![](images/9a1e973883ca11920a8f733c1d5d9c9d604b48bf13a0602e62bf73b9c2793f91.jpg)  
DPMSolver-3, NFE=5

![](images/787b25866c61809eb3ec28ea4008855a663c611d9149c73d086c67fad263e548.jpg)  
UniPC-3, NFE=5

![](images/49bf21f13cb83ad16229fdfa0b5893bbbe8aebbe234c40f6a69034d40ca8423f.jpg)  
F-DDIM (ours), NFE=5

Figure 7: Qualitative comparisons between F-DDIM (ours) and DDIM, DPMSolver-2, DPMSolver-3, and UniPC-3. Images are sampled from pre-trained unconditional EDM on CIFAR10 dataset.  
![](images/53ee0d1d3a572cccfaafa245ad1605b2f20dbb7c84d9619ae50570bdd842997d.jpg)  
F-DDIM (ours), NFE=5  
Figure 8: Qualitative comparisons between F-DDIM (ours) and DDIM, DPMSolver-2, DPMSolver-3, and UniPC-3. Images are sampled from pre-trained conditional EDM on CIFAR10 dataset.

6  
10  
5  
10  
5  
6  
8  
![](images/783b414a0af8cbaab135c81b885084615755cafab1975bf14a92b85a53dad72c.jpg)  
Figure 9: Qualitative comparisons between F-DDIM (ours) and DDIM, DPMSolver-2, DPMSolver-3, and UniPC-3. Images are sampled from pre-trained latent difusion model on FFHQ dataset.

5  
6  
8  
10  
5  
8  
![](images/cf778a8d4e8b5a0887302299a206780f722e67315fe9197ce54e24e6faf3c933.jpg)  
Figure 10: Qualitative comparisons between F-DDIM (ours) and DDIM, DPMSolver-2, DPMSolver-3, and UniPC-3. Images are sampled from pre-trained latent difusion model on LSUN bedroom dataset.

## A.3 Higher-order lookahead predictors

Algorithm 3 summarizes the general F-DPMSolver framework using a data prediction model. We evaluate F-DPMSolver-2, which constructs the lookahead state using the DPMSolver-2 update with cached noise predictions:

$$
\widehat { \pmb { x } } _ { t _ { i } } = \mathrm { O D E S o l v e r } 2 \big ( \pmb { x } _ { t _ { i - 1 } } , \widetilde { \pmb { \varepsilon } } _ { t _ { i - 1 } } , \widetilde { \pmb { \varepsilon } } _ { t _ { i - 2 } } \big ) , \qquad i \ge 2 .
$$

The first lookahead state is constructed using (21), with $\widetilde { \varepsilon } _ { t _ { 0 } } = \mathbf { x } _ { t _ { 0 } } .$ The higher-order predictor changes only the lookahead construction; each iteration still requires one new model evaluation.

```tcl
Algorithm 3 F-DPMSolver
Input: Data prediction model $\mu _ { \boldsymbol { \theta } } ( \boldsymbol { x } , t )$ , time grid $t _ { 0 } > \cdots > t _ { M }$ , noise schedule $\left( \alpha _ { t } , \sigma _ { t } \right)$ , initialization $\mathbf { \boldsymbol { x } } _ { t _ { 0 } } ,$
and a lookahead predictor using cached noise estimates.
1: Initialize $\widetilde { \varepsilon } _ { t _ { 0 } } \gets x _ { t _ { 0 } }$
2: for $i = 1 , \dots , M$ do
3: Construct $\widehat { \pmb { x } } _ { t _ { i } }$ from ${ \bf { x } } _ { t _ { i - 1 } }$ and the cached estimates $\{ \widetilde { \varepsilon } _ { t _ { j } } \} _ { j = 0 } ^ { i - 1 }$ , without a new model evaluation.
4: Update $\pmb { x } _ { t _ { i } } \gets \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \pmb { x } _ { t _ { i - 1 } } + \left( \alpha _ { t _ { i } } - \frac { \sigma _ { t _ { i } } \alpha _ { t _ { i - 1 } } } { \sigma _ { t _ { i - 1 } } } \right) \pmb { \mu } _ { \theta } ( \widehat { \pmb { x } } _ { t _ { i } } , t _ { i } )$
5: Set and cache $\widetilde { \varepsilon } _ { t _ { i } } \gets ( \widehat { \pmb x } _ { t _ { i } } - \alpha _ { t _ { i } } \pmb \mu _ { \theta } ( \widehat { \pmb x } _ { t _ { i } } , t _ { i } ) ) / \sigma _ { t _ { i } }$
6: end for
Output: Generated sample ${ \pmb x } _ { t _ { M } }$
```

Quantitative results (FID). Tables 9 and 10 compare F-DDIM and F-DPMSolver-2 at the same NFE. F-DPMSolver-2 achieves lower FID in all reported CIFAR-10 settings and most ImageNet-512 settings. The exception is EDM2-XS at NFE = 10, where F-DDIM achieves an FID of 4.57, compared with 4.90 for F-DPMSolver-2. These results show that a higher-order lookahead predictor can further improve sample quality in most cases. The predictor order does not determine the convergence order of the forward-value sampler. Under the lookahead accuracy condition (170), Theorem 7 in Appendix E guarantees that the sampler F-DPMSolver-2 has the first-order convergence rate of the ideal forward-value sampler.

Table 9: FID comparison of F-DDIM and F-DPMSolver-2 on CIFAR-10.
<table><tr><td rowspan=2 colspan=1>NFE</td><td rowspan=1 colspan=2>EDM-uncond</td><td rowspan=1 colspan=2>EDM-cond</td></tr><tr><td rowspan=1 colspan=1>F-DDIM</td><td rowspan=1 colspan=1>F-DPMSolver-2</td><td rowspan=1 colspan=1>F-DDIM</td><td rowspan=1 colspan=1>F-DPMSolver-2</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>25.01</td><td rowspan=1 colspan=1>22.14</td><td rowspan=1 colspan=1>18.44</td><td rowspan=1 colspan=1>17.21</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>16.04</td><td rowspan=1 colspan=1>14.97</td><td rowspan=1 colspan=1>12.29</td><td rowspan=4 colspan=1>11.376.703.572.47</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>9.47</td><td rowspan=1 colspan=1>8.44</td><td rowspan=1 colspan=1>7.57</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>4.91</td><td rowspan=1 colspan=1>4.13</td><td rowspan=1 colspan=1>4.32</td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>3.46</td><td rowspan=1 colspan=1>2.79</td><td rowspan=1 colspan=1>3.18</td></tr></table>

Table 10: FID comparison of F-DDIM and F-DPMSolver-2 on ImageNet-512.
<table><tr><td rowspan=2 colspan=1>NFE</td><td rowspan=1 colspan=2>EDM2-XS</td><td rowspan=1 colspan=2>EDM2-XXL</td></tr><tr><td rowspan=1 colspan=1>F-DDIM</td><td rowspan=1 colspan=1>F-DPMSolver-2</td><td rowspan=1 colspan=1>F-DDIM</td><td rowspan=1 colspan=1>F-DPMSolver-2</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>50.96</td><td rowspan=1 colspan=1>47.63</td><td rowspan=1 colspan=1>52.32</td><td rowspan=5 colspan=1>47.0618.9710.884.242.89</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>22.69</td><td rowspan=1 colspan=1>20.30</td><td rowspan=1 colspan=1>23.56</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>14.33</td><td rowspan=1 colspan=1>11.93</td><td rowspan=1 colspan=1>14.04</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>6.18</td><td rowspan=1 colspan=1>6.09</td><td rowspan=1 colspan=1>5.04</td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>4.57</td><td rowspan=1 colspan=1>4.90</td><td rowspan=1 colspan=1>3.06</td></tr></table>

Qualitative comparisons. Figures 11 and 12 compare the samples generated by F-DDIM and F-DPMSolver-2. The higher-order lookahead predictor can improve visual details at small NFEs, consistent with the FID comparisons.

![](images/bbd3240b370c1c8dd5a88fc0a6c09665ea872e35fe3d0f73a1468f3f44d65721.jpg)  
F-DDIM, NFE=5

![](images/18cca362be92858f52d25ccff3d747f06e4df79599be42ce60666c38a94c4723.jpg)  
F-DDIM, NFE=5

![](images/29f02d8265a6b76d9f8eebce7a61412988a90f08e8d9c2db0414bf08ad5cb3f3.jpg)  
F-DPMSolver2, NFE=5  
(a) CIFAR10, EDM-uncond

![](images/124c8634a66b0d91952d22d04cac1594c44e0e0eecdef44bcdfb5167bbff39b2.jpg)  
F-DPMSolver2, NFE=5  
(b) CIFAR10, EDM-cond

Figure 11: Qualitative comparison of F-DDIM and F-DPMSolver-2 on CIFAR-10 using unconditional and conditional EDM models.  
NFE=4  
5  
6  
![](images/462fc7b88d8de4e1d2a8dfc9fef3f7d64b31c5f33357095d2966acaf7ce70ea7.jpg)  
8  
10

NFE=4  
5  
![](images/4fa3682112b8d37cfaf23ff5ae379f6f94ce072c2c4652414557268e89e07ace.jpg)

8  
6  
10  
![](images/083de4ff18d99f45f595747166311513bba1d50fe3159deac9cda3c59092f1cb.jpg)  
(a) ImageNet512, EDM2-XS  
(b) ImageNet512, EDM2-XXL  
Figure 12: Qualitative comparison of F-DDIM and F-DPMSolver-2 on ImageNet-512 using EDM2-XS and EDM2-XXL.

## A.4 Larger NFE budgets and runtime comparisons

We examine F-DDIM at larger NFE budgets on ImageNet-512 using EDM2-XXL and compare runtime on ImageNet-64 using EDM2-S.

Larger NFE results. Table 11 reports FID at NFEs 10, 12, 15, and 20. F-DDIM achieves the lowest reported FID at NFE=10, 12, and 15, and matches UniPC-3 at NFE=20. These results show that F-DDIM remains comparable to higher-order baselines over the tested range of larger NFEs.

Table 11: FID at larger NFE budgets on ImageNet-512 with EDM2-XXL.
<table><tr><td>NFE</td><td>DPM Solver-2</td><td>DPM Solver-3</td><td> $\overline { { \mathrm { U n i P C } } }$  -3</td><td>DDIM</td><td>F-DDIM (ours)</td></tr><tr><td>10</td><td>6.58</td><td>3.60</td><td>3.15</td><td>21.48</td><td>3.06</td></tr><tr><td>12</td><td>4.44</td><td>2.80</td><td>2.53</td><td>15.02</td><td>2.48</td></tr><tr><td>15</td><td>3.19</td><td>2.39</td><td>2.24</td><td>9.76</td><td>2.23</td></tr><tr><td>20</td><td>2.52</td><td>2.16</td><td>2.11</td><td>5.98</td><td>2.11</td></tr></table>

Runtime comparison. Table 12 compares runtime on ImageNet-64 using EDM2-S. Let $T _ { \mathcal { A } } ( M )$ denote the runtime of sampler A with M model evaluations. We report the normalized runtime

$$
R _ { \cal A } ( M ) = 1 0 \frac { T _ { \cal A } ( M ) } { T _ { \mathrm { D D I M } } ( 1 0 ) } .
$$

A table entry of the form $M \times c$ denotes $R _ { \cal A } ( M ) = M c$ . Thus, c measures the average runtime per model evaluation relative to that of DDIM with NFE = 10. F-DDIM and F-DPMSolver-2 have runtimes similar to DDIM and DPMSolver-2 at the same NFE. This supports using NFE as a practical proxy for runtime when comparing these samplers under the tested setting.

Table 12: Normalized runtime on ImageNet-64 with EDM2-S. DDIM with NFE = 10 is assigned a value of 10.
<table><tr><td>NFE</td><td>4</td><td>5</td><td>6</td><td>8</td><td>10</td></tr><tr><td>DDIM DPMSolver-2 DPMSolver-3 UniPC-3</td><td>4 × 1.18  $4 \times 1 . 1 7$   $4 \times 1 . 5 9$   $4 \times 1 . 1 8$ </td><td> $5 \times 1 . 1 2$   $5 \times 1 . 1 0$   $5 \times 1 . 4 7$   $5 \times 1 . 1 6$   $6 \times 1 . 3 8$ </td><td> $6 \times 1 . 3 2$   $6 \times 1 . 0 7$   $6 \times 1 . 1 6$ </td><td> $8 \times 1 . 0 4$   $8 \times 1 . 0 2$   $8 \times 1 . 0 9$ </td><td> $\overline { { 1 0 \times 1 . 0 0 } }$   $1 0 \times 1 . 0 0$   $1 0 \times 1 . 0 5$ </td></tr></table>

## A.5 Additional samples from F-DDPM

This subsection supplements the qualitative comparisons in Section 4.2. Figures 13 and 14 show unconditional and class-conditional CIFAR-10 samples, while Figures 15 and 16 show results on ImageNet-512 and LSUN bedrooms, respectively. Within each comparison, F-DDPM and the stochastic baselines use the same NFE, initial noise, and Brownian motion. The examples illustrate that F-DDPM often produces clearer structures and finer details at small NFEs, complementing the quantitative results in Tables 3–6.

![](images/9e937f7ce8298780510ae0836e8ccd99b170c3f5f2cc2236dc8e28ad14d0f4f8.jpg)  
SDE-DPMSolver++1, NFE=6

![](images/fa7b276febd527740c845cbe213216e89341ebfb3f7768234426ecfb3dd86fef.jpg)  
SDE-DPMSolver++2, NFE=6

![](images/122bde69d6bf076bcb00ae4927f87bbead37710855332e46b0187097d069bc11.jpg)  
F-DDPM (ours), NFE=6  
Figure 13: Qualitative comparisons between F-DDPM (ours) and SDE-DPMSolver++1, and SDE-DPMSolver++2. Images are sampled from pre-trained unconditional EDM on CIFAR10 dataset.

![](images/01bb45ae53905dd7498a8cf479ac5fbbff09b99db374e175ac5f53cd64d848a2.jpg)  
SDE-DPMSolver++1, NFE=6

![](images/df6aea605db453b084418ab125b9f77c2eed760f791510baf96b1a7ee5eee54c.jpg)  
SDE-DPMSolver++2, NFE=6

![](images/fd02d70aed75f4fd0afd7ebd530c8375870f80df4d3c7448aab6c0b380a47a07.jpg)  
F-DDPM (ours), NFE=6  
Figure 14: Qualitative comparisons between F-DDPM (ours) and SDE-DPMSolver++1, and SDE-DPMSolver++2. Images are sampled from pre-trained conditional EDM on CIFAR10 dataset.

9  
NFE=6  
12  
![](images/01c2a02ef1c94b92c2692d548e906a905c3dfc81b46665309828fb9e2080f8f8.jpg)  
15

NFE=6  
9  
12  
15  
![](images/fa6e25a8b4c7973e9a1e3c86df4eb5ab1dab39e3ff08554e8fbc8434bfd694d6.jpg)  
(a) ImageNet512, EDM2-XS  
(b) ImageNet512, EDM2-XXL  
Figure 15: Qualitative comparisons between F-DDPM (ours) and SDE-DPMSolver++1, and SDE-DPMSolver++2. Images are sampled from pre-trained EDM2 with XS and XXL size on ImageNet512 dataset.

9  
NFE=6  
12  
15  
NFE=6  
9  
12  
15  
![](images/dc3438a7241d76c40fdfdec0db92ea8fcc386e1337b942a8bb9ab679b8a1ee13.jpg)  
Figure 16: Qualitative comparisons between F-DDPM (ours) and SDE-DPMSolver++1, and SDE-DPMSolver++2. Images are sampled from pre-trained latent difusion model on LSUN bedroom dataset.

## B Proof of deterministic samplers convergence lower bound (Theorem 1)

Let us consider the case where the target distribution is an isotropic Gaussian distribution $q _ { 0 } = \mathcal { N } ( 0 , \gamma ^ { 2 } I _ { d } )$ for some constant $\gamma \geq 0$

Preliminaries. Before proceeding to the main proof, we first introduce some preliminary results that will be frequently used in the following analysis.

By the choice of the forward process (1), the distribution $q _ { t }$ of $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ at any time $t \geq 0$ satisfies

$$
\begin{array} { r } { \pmb { x } _ { t } \sim q _ { t } = \mathcal { N } \big ( 0 , ( \alpha _ { t } ^ { 2 } \gamma ^ { 2 } + \sigma _ { t } ^ { 2 } ) \pmb { I } _ { d } \big ) , } \end{array}
$$

and its score function $s _ { t } ^ { \star } ( \cdot )$ can be computed in closed form as

$$
\pmb { s } _ { t } ^ { \star } ( \pmb { x } ) = \nabla _ { \pmb { x } } \log q _ { t } ( \pmb { x } ) = \nabla _ { \pmb { x } } \left( - \frac { \| \pmb { x } \| ^ { 2 } } { 2 ( \alpha _ { t } ^ { 2 } \gamma ^ { 2 } + \sigma _ { t } ^ { 2 } ) } \right) = - \frac { \pmb { x } } { \alpha _ { t } ^ { 2 } \gamma ^ { 2 } + \sigma _ { t } ^ { 2 } } .
$$

Combining this with the relationship between the noise predictor and score function $\boldsymbol { \varepsilon } _ { \boldsymbol { \theta } } ( \mathbf { x } _ { t } , t ) = - \sigma _ { t } \boldsymbol { s } _ { t } ^ { \star } ( \mathbf { x } _ { t } )$ 2 we can derive the noise predictor at noise level $t ( \lambda )$ for any $\lambda \in \mathbb { R }$ as

$$
\varepsilon _ { \theta } \left( \boldsymbol { x } _ { t ( \lambda ) } , t ( \lambda ) \right) = - \sigma _ { t ( \lambda ) } \boldsymbol { s } _ { t ( \lambda ) } ^ { \star } ( \boldsymbol { x } _ { t ( \lambda ) } ) = \frac { \sigma _ { t ( \lambda ) } \boldsymbol { x } _ { t ( \lambda ) } } { \alpha _ { t ( \lambda ) } ^ { 2 } \gamma ^ { 2 } + \sigma _ { t ( \lambda ) } ^ { 2 } } = \frac { \mathrm { e } ^ { - \lambda } \boldsymbol { x } _ { t ( \lambda ) } } { \alpha _ { t ( \lambda ) } ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } ) } ,\tag{34}
$$

where the last line holds due to the fact that $\sigma _ { t ( \lambda ) } = \alpha _ { t ( \lambda ) } \mathrm { e } ^ { - \lambda }$

We claim that in the isotropic Gaussian case, the exact solution $\boldsymbol { \mathbf { \mathit { x } } } _ { t } ^ { \star }$ of the difusion ODE (12a) remains colinear with the initial point $\boldsymbol { x } _ { t _ { 0 } } ^ { \star }$ . Specifically, for any $\lambda \in [ \lambda _ { t _ { 0 } } , \lambda _ { t _ { M } } ]$ , we have

$$
\pmb { x } _ { t ( \lambda ) } ^ { \star } = \kappa _ { t ( \lambda ) } ^ { \star } \pmb { x } _ { t _ { 0 } } ^ { \star } \quad \mathrm { w i t h } \quad \kappa _ { t ( \lambda ) } ^ { \star } = \frac { \alpha _ { t ( \lambda ) } } { \alpha _ { t _ { 0 } } } \sqrt { \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda ( t _ { 0 } ) } } } .\tag{35}
$$

To verify this, let us substitute the expression (35) into the right-hand-side of the difusion ODE (12a). For arbitrary $t , s \geq 0$ , straightforward calculation yields

$$
\begin{array} { r l } & { \frac { \langle \alpha _ { \parallel } } { \omega _ { \perp } } x _ { e } ^ { * } - \alpha _ { \parallel } \int _ { \lambda ( t ) } ^ { \lambda ( t ) } e ^ { - \lambda _ { e } } \theta _ { e } ( x _ { i ( t ) , i } ^ { e } , \ell ( \lambda ) ) \mathrm { d } \lambda \rangle } \\ & { \quad \stackrel { \mathrm { ( i ) } } { = } \frac { \langle \alpha _ { \parallel } } { \omega _ { e } } x _ { e } ^ { * } - \int _ { \lambda ( t ) } ^ { \lambda ( t ) } \frac { \mathcal { O } _ { e } } { \omega _ { \perp } ( \lambda ) ! } \frac { e ^ { - 2 \lambda _ { e } } x _ { i ( \lambda ) } ^ { * } \mathrm { d } \lambda } { \gamma ^ { 2 } + \exp ^ { - 2 \lambda _ { e } } \lambda _ { i ( \lambda ) } ^ { * } \mathrm { d } \lambda } } \\ & { \quad \stackrel { \mathrm { ( i i ) } } { = } x _ { e } ^ { * } \Bigg \{ \frac { \langle \alpha _ { t } } { \omega _ { \perp } } - \int _ { \lambda ( t ) } ^ { \lambda ( t ) } \frac { \mathcal { O } _ { e } } { \omega _ { \perp } ( \lambda ) ! } \frac { e ^ { - 2 \lambda _ { e } } x _ { i ( \lambda ) } ^ { * } } { ( \gamma ^ { 2 } + \exp ^ { - 2 \lambda _ { e } } \lambda ) \kappa _ { e } } \mathrm { d } \lambda \Bigg \} } \\ & { \quad \stackrel { \mathrm { ( i i ) } } { = } x _ { e } ^ { * } \Bigg \{ \frac { \langle \alpha _ { t } } { \omega _ { e } } - \frac { \alpha _ { t } } { \kappa _ { x } \sqrt { \gamma ^ { 2 } + \exp ^ { - 2 \lambda _ { e } } \lambda } } \int _ { \lambda ( t ) } ^ { \lambda ( t ) } \frac { \mathrm { e } ^ { - 2 \lambda _ { e } } } { \sqrt { \gamma ^ { 2 } + ( \exp ^ { - 2 \lambda _ { e } } \lambda ) } } \mathrm { d } \lambda \Bigg \} } \\ &  \quad = \frac  \langle \alpha _ { t } x _ { e }  \end{array}\tag{36}
$$

where (a) applies (34); (b) and (c) uses the expression of $\boldsymbol { x } _ { t ( \lambda ) } ^ { \star }$ from (35); and the penultimate line holds because $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } \lambda } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } = - \frac { \mathrm { e } ^ { - 2 \lambda } } { \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } } } \end{array}$ . Setting $s = t _ { 0 }$ establishes the claim.

Moreover, the calculation in (36) implies the following identity regarding κ<sub>t</sub> for any i ∈ [M]: $\kappa _ { t _ { i } }$ $i \in [ M ]$

$$
\kappa _ { t _ { i } } ^ { \star } = \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } \biggl ( 1 - \frac { 1 } { \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \frac { \mathrm { e } ^ { - 2 \lambda } } { \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } } \mathrm { d } \lambda \biggr ) \kappa _ { t _ { i - 1 } } ^ { \star } ,\tag{37}
$$

which will be frequently used in the following proof.

With the above preparation in hand, we shall establish the lower bound for deterministic DDIM (13) and second-order ODE solver (31) separately.

Lower bound for deterministic DDIM. In this paragraph, let $\mathbf { \mathcal { x } } _ { t _ { i } }$ denote the iterates produced by the deterministic DDIM sampler.

As shown in (34), the noise predictor $\varepsilon _ { \boldsymbol { \theta } } ( { \boldsymbol { \mathbf { \mathit { x } } } } , t )$ is colinear with x. Together with the DDIM update rule (13), this implies that the i-th iterate ${ \mathbf { \mathcal { x } } } _ { t _ { i } }$ is aligned with the initial point $\mathbf { \mathcal { x } } _ { t _ { 0 } }$ for all $i \in [ M ]$ . Therefore, we can write

$$
\pmb { x } _ { t _ { i } } = \kappa _ { t _ { i } } \pmb { x } _ { t _ { 0 } }
$$

where $\kappa _ { t _ { i } } \in \mathbb { R }$ denotes the scalar coeficient.

Substituting $\pmb { x } _ { t _ { i } } = \kappa _ { t _ { i } } \pmb { x } _ { t _ { 0 } }$ and the expression (34) for $\varepsilon _ { \theta }$ into the DDIM update (13) yields the following recurrence for $\kappa _ { t _ { i } }$

$$
\kappa _ { t _ { 0 } } = 1 , \quad \mathrm { a n d } \quad \kappa _ { t _ { i } } = \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } \biggl ( 1 - \frac { \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - \lambda } \mathrm { d } \lambda \biggr ) \kappa _ { t _ { i - 1 } } , \quad i \geq 1 .\tag{38}
$$

Comparing (38) with the corresponding identity for $\kappa _ { t _ { i } } ^ { \star }$ in (37), and using $\kappa _ { t _ { 0 } } ^ { \star } = \kappa _ { t _ { 0 } } = 1$ , we obtain that for any $1 \leq i \leq M$ ，

$$
\begin{array} { r l r } {  { \kappa _ { t _ { i } } ^ { * } - \kappa _ { t _ { i } } = a _ { \mathrm { i } } \bigl ( \kappa _ { t _ { i - 1 } } ^ { * } - \kappa _ { t _ { i - 1 } } \bigr ) + \kappa _ { t _ { i - 1 } } ^ { * } \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - \lambda } ( \frac { \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } } { \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } } } - \frac { \mathrm { e } ^ { - \lambda } } { \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } } ) \mathrm { d } \lambda } } \\ & { } & { = a _ { \mathrm { i } } \bigl ( \kappa _ { t _ { i - 1 } } ^ { * } - \kappa _ { t _ { i - 1 } } \bigr ) + \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { 0 } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - \lambda } ( \frac { \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } } { \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } } } - \frac { \mathrm { e } ^ { - \lambda } } { \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } } ) \mathrm { d } \lambda , \qquad ( 3 9 ) } \end{array}
$$

where the coeficient $a _ { i }$ is defined as

$$
a _ { i } : = \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } \biggl ( 1 - \frac { \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - \lambda } \mathrm { d } \lambda \biggr ) , \quad 1 \le i \le M ,\tag{40}
$$

and the second line follows from the expression of $\kappa _ { t _ { i - 1 } } ^ { \star }$ from (35).

Notice that for any $\gamma > 0$

$$
\frac { \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } } { \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } } } - \frac { \mathrm { e } ^ { - \lambda } } { \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } } \geq 0 , \quad \forall \lambda \in [ \lambda _ { t _ { i - 1 } } , \lambda _ { t _ { i } } ] .
$$

Plugging this into (39) leads to $\kappa _ { t _ { i } } ^ { \star } \geq \kappa _ { t _ { i } }$ for all $0 \leq i \leq M$ . Thus, the diference between the difusion ODE and the deterministic DDIM trajectories at step M satisfies

$$
\left\| \pmb { x } _ { t _ { M } } - \pmb { x } _ { t _ { M } } ^ { \star } \right\| = \big ( \kappa _ { t _ { M } } ^ { \star } - \kappa _ { t _ { M } } \big ) \| \pmb { x } _ { t _ { 0 } } \| .
$$

Therefore, to establish that the deterministic DDIM has convergence order at most one, it is suficient to show that $\kappa _ { t _ { M } } ^ { \star } - \kappa _ { t _ { M } } = \Omega ( 1 / M )$

To this end, note that the integral on the right-hand-side of (39) can be lower bounded by

$$
\begin{array} { r l r } {  { \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - \lambda } ( \frac { \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } } { \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } } } - \frac { \mathrm { e } ^ { - \lambda } } { \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } } ) \mathrm { d } \lambda } } \\ & { } & { = \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - \lambda ^ { \prime } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda ^ { \prime } } - \frac { \mathrm { d } } { \mathrm { d } \lambda } \frac { \mathrm { e } ^ { - \lambda } } { \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } } \mathrm { d } \lambda \mathrm { d } \lambda ^ { \prime } } \end{array}
$$

$$
\begin{array} { r l } & { = \displaystyle \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \left( \mathrm { e } ^ { - \lambda } - \mathrm { e } ^ { - \lambda _ { t _ { i } } } \right) \frac { \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } } { \left( \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } + 1 \right) ^ { \frac { 3 } { 2 } } } \mathrm { d } \lambda } \\ & { \stackrel { \mathrm { ( a ) } } { \geq } \frac { \gamma ^ { 2 } \delta _ { i } ^ { 2 } \mathrm { e } ^ { - \lambda _ { t _ { i } } } \mathrm { e } ^ { 2 \lambda _ { t _ { i - 1 } } } } { 2 \left( \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { i } } } + 1 \right) ^ { \frac { 3 } { 2 } } } } \\ & { \stackrel { \mathrm { ( b ) } } { \geq } \frac { \gamma ^ { 2 } \delta _ { i } ^ { 2 } \mathrm { e } ^ { \lambda _ { t _ { i } } } } { 4 ( \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { i } } } + 1 ) ^ { \frac { 3 } { 2 } } } , } \end{array}\tag{41}
$$

where (a) holds because $\lambda _ { t _ { i } }$ is increasing in i and thus

$$
\begin{array} { r l } & { \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \left( \mathrm { e } ^ { - \lambda } - \mathrm { e } ^ { - \lambda _ { t _ { i } } } \right) \frac { \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } } { \left( \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } + 1 \right) ^ { \frac { 3 } { 2 } } } \mathrm { d } \lambda \geq \frac { \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { i - 1 } } } } { \left( \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { i } } } + 1 \right) ^ { \frac { 3 } { 2 } } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \left( \mathrm { e } ^ { - \lambda } - \mathrm { e } ^ { - \lambda _ { t _ { i } } } \right) \mathrm { d } \lambda } \\ & { \qquad = \frac { \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { i - 1 } } } \mathrm { e } ^ { - \lambda _ { t _ { i } } } } { \left( \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { i } } } + 1 \right) ^ { \frac { 3 } { 2 } } } \left( \mathrm { e } ^ { \delta _ { i } } - 1 - \delta _ { i } \right) \geq \frac { \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { i - 1 } } } \mathrm { e } ^ { - \lambda _ { t _ { i } } } \delta _ { i } ^ { 2 } } { 2 \left( \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { i } } } + 1 \right) ^ { \frac { 3 } { 2 } } } , } \end{array}
$$

and (b) is true because $\mathrm { e } ^ { 2 \lambda _ { t _ { i - 1 } } - \lambda _ { t _ { i } } } = \mathrm { e } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - 2 \delta _ { i } } \geq \mathrm { e } ^ { \lambda _ { t _ { i } } } / 2$ provided that $\delta _ { i } = { O } ( 1 / M ) \le \log ( 2 ) / 2$ . Substituting (41) into (39), we obtain

$$
\kappa _ { t _ { i } } ^ { \star } - \kappa _ { t _ { i } } \geq a _ { i } \bigl ( \kappa _ { t _ { i - 1 } } ^ { \star } - \kappa _ { t _ { i - 1 } } \bigr ) + \frac { 1 } { \alpha _ { t _ { 0 } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { 0 } } } } } \frac { \gamma ^ { 2 } \delta _ { i } ^ { 2 } \alpha _ { t _ { i } } \mathrm { e } ^ { \lambda _ { t _ { i } } } } { 4 ( \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { i } } } + 1 ) ^ { \frac { 3 } { 2 } } } , \quad \forall 1 \leq i \leq M .\tag{42}
$$

Applying (42) recursively allows us to bound the final error as

$$
\kappa _ { t _ { M } } ^ { \star } - \kappa _ { t _ { M } } \geq \frac { 1 } { \alpha _ { t _ { 0 } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { 0 } } } } } \sum _ { i = 1 } ^ { M } \prod _ { j = i + 1 } ^ { M } a _ { j } \frac { \gamma ^ { 2 } \delta _ { i } ^ { 2 } \alpha _ { t _ { i } } \mathrm { e } ^ { \lambda _ { t _ { i } } } } { 4 ( \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { i } } } + 1 ) ^ { \frac { 3 } { 2 } } } .\tag{43}
$$

To control the right-hand-side of (43), let us introduce a subset of steps:

$$
S : = \{ 1 \leq i \leq M : c _ { 1 } \leq \mathrm { e } ^ { \lambda _ { t _ { i - 1 } } } \leq \mathrm { e } ^ { \lambda _ { t _ { i } } } \leq c _ { 2 } \} ,\tag{44}
$$

where $c _ { 1 } , \ : c _ { 2 }$ are two positive constants satisfying $\mathrm { e } ^ { \lambda _ { t _ { 0 } } } < c _ { 1 } < c _ { 2 } < \mathrm { e } ^ { \lambda _ { t _ { M } } }$ . For suficiently large $M$

$$
\sum _ { i \in \cal S } \delta _ { i } \geq \log c _ { 2 } - \log c _ { 1 } - 2 \operatorname* { m a x } _ { i } \delta _ { i } \geq \frac { 1 } { 2 } \big ( \log c _ { 2 } - \log c _ { 1 } \big ) .
$$

Note that $\mathrm { e } ^ { - \lambda _ { t _ { i } } } \le \mathrm { e } ^ { - \lambda }$ and $\gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } \geq ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } ) \mathrm { e } ^ { - 2 \delta _ { i } }$ for all $\lambda \in [ \lambda _ { t _ { i - 1 } } , \lambda _ { t _ { i } } ]$ . This together with the definition of $a _ { i }$ in (40) imply the following lower bound

$$
\begin{array} { r l r } {  { a _ { i } \geq \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } ( 1 - \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \frac { \mathrm { e } ^ { - \lambda _ { t _ { i } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } } \mathrm { e } ^ { - \lambda } \mathrm { d } \lambda ) } } \\ & { } & { \geq \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } ( 1 - \mathrm { e } ^ { 2 \delta _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \frac { \mathrm { e } ^ { - 2 \lambda } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } \mathrm { d } \lambda ) \geq \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } ( 1 - \frac { 5 } { 4 } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \frac { \mathrm { e } ^ { - 2 \lambda } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } \mathrm { d } \lambda ) } \end{array}
$$

provided that $2 \delta _ { i } = O ( 1 / M ) \leq \log ( 5 / 4 )$ . This further allows us to lower bound the cumulative product $\textstyle \prod _ { j = i + 1 } ^ { M } a _ { j }$ for any $i \in S$

$$
\begin{array} { r } { \displaystyle \prod _ { j = i + 1 } ^ { M } a _ { j } \geq \frac { \alpha _ { t _ { M } } } { \alpha _ { t _ { i } } } \displaystyle \prod _ { j = i + 1 } ^ { M } \left( 1 - \frac { 5 } { 4 } \displaystyle \int _ { \lambda _ { t _ { j - 1 } } } ^ { \lambda _ { t _ { j } } } \frac { \mathrm { e } ^ { - 2 \lambda } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } \mathrm { d } \lambda \right) } \\ { \geq \displaystyle \frac { \alpha _ { t _ { M } } } { \alpha _ { t _ { i } } } \left( 1 - \frac { 5 } { 4 } \displaystyle \sum _ { j = i + 1 } ^ { M } \displaystyle \int _ { \lambda _ { t _ { j - 1 } } } ^ { \lambda _ { t _ { j } } } \frac { \mathrm { e } ^ { - 2 \lambda } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } \mathrm { d } \lambda \right) } \end{array}
$$

$$
\geq \frac { \alpha _ { t _ { M } } } { \alpha _ { t _ { i } } } \left( 1 - \frac { 5 } { 4 } \sum _ { \mathrm { e } ^ { \lambda _ { t _ { j } } } \geq c _ { 1 } } \int _ { \lambda _ { t _ { j - 1 } } } ^ { \lambda _ { t _ { j } } } \frac { \mathrm { e } ^ { - 2 \lambda } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } \mathrm { d } \lambda \right) \overset { \mathrm { ( a ) } } { \geq } \frac { \alpha _ { t _ { M } } } { \alpha _ { t _ { i } } } \left( 1 - \frac { 5 } { 8 } \log ( 1 + \gamma ^ { - 2 } c _ { 1 } ^ { - 2 } ) \right) \geq \frac { \alpha _ { t _ { M } } } { 8 \alpha _ { t _ { i } } } ,\tag{45}
$$

provided that log $\smash { \zeta ( 1 + \gamma ^ { - 2 } c _ { 1 } ^ { - 2 } ) \leq 7 / 5 }$ , where (a) holds because

$$
\sum _ { \mathrm { e } ^ { \lambda _ { j - 1 } } \geq c _ { 1 } } \int _ { \lambda _ { j - 1 } } ^ { \lambda _ { t _ { j } } } \frac { \mathrm { e } ^ { - 2 \lambda } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } \mathrm { d } \lambda \leq \int _ { \log c _ { 1 } } ^ { \infty } \frac { \mathrm { e } ^ { - 2 \lambda } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } \mathrm { d } \lambda = \frac { 1 } { 2 } \log ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } ) \bigg | _ { \infty } ^ { \log c _ { 1 } } = \frac { 1 } { 2 } \log ( 1 + \gamma ^ { - 2 } c _ { 1 } ^ { - 2 } ) .
$$

To finish up, substituting (45) into (43) yields

$$
\begin{array} { r l } & { \kappa _ { t _ { M } } ^ { \star } - \kappa _ { t _ { M } } \geq \frac { 1 } { \alpha _ { t _ { 0 } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { 0 } } } } } \displaystyle \sum _ { i \in \mathcal { S } } \displaystyle \prod _ { j = i + 1 } ^ { M } a _ { j } \frac { \gamma ^ { 2 } \delta _ { i } ^ { 2 } \alpha _ { t _ { i } } \mathrm { e } ^ { \lambda _ { t _ { i } } } } { 4 ( \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { i } } } + 1 ) ^ { \frac { 3 } { 2 } } } } \\ & { \quad \quad \stackrel { \mathrm { ( a ) } } { \geq } \frac { \alpha _ { t _ { M } } } { \alpha _ { t _ { 0 } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { 0 } } } } } \frac { \gamma ^ { 2 } c _ { 1 } } { 3 2 ( \gamma ^ { 2 } c _ { 2 } ^ { 2 } + 1 ) ^ { \frac { 3 } { 2 } } } \displaystyle \sum _ { i \in \mathcal { S } } \delta _ { i } ^ { 2 } } \\ & { \quad \quad \stackrel { \mathrm { ( b ) } } { \geq } \frac { \alpha _ { t _ { M } } } { \alpha _ { t _ { 0 } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { 0 } } } } } \frac { \gamma ^ { 2 } c _ { 1 } } { 3 2 ( \gamma ^ { 2 } c _ { 2 } ^ { 2 } + 1 ) ^ { \frac { 3 } { 2 } } } \frac { ( \log c _ { 2 } - \log c _ { 1 } ) ^ { 2 } } { 4 M } = \Omega ( M ^ { - 1 } ) , } \end{array}\tag{46}
$$

where (a) holds because $\mathrm { e } ^ { \lambda _ { t _ { i } } } / ( \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { i } } } + 1 ) ^ { 3 / 2 } \geq c _ { 1 } / ( \gamma ^ { 2 } c _ { 2 } ^ { 2 } + 1 ) ^ { 3 / 2 }$ , and (b) is true due to the Cauchy-Schwarz inequality $\begin{array} { r } { ( \sum _ { i \in \mathcal { S } } \delta _ { i } ) ^ { 2 } \leq | \mathcal { S } | \sum _ { i \in \mathcal { S } } \delta _ { i } ^ { 2 } } \end{array}$ and the fact that $\begin{array} { r } { \sum _ { i \in \mathcal { S } } \delta _ { i } \ge \frac 1 2 \big ( \log c _ { 2 } - \log c _ { 1 } \big ) } \end{array}$ . This completes the proof for the lower bound of deterministic DDIM.

Lower bound for second-order ODE solver. In this paragraph, we use ${ \mathbf { \mathcal { x } } } _ { t _ { i } }$ to represent the iterates generated by the second-order ODE solver (31). Similar to the analysis for the deterministic DDIM, we know that $\varepsilon _ { \boldsymbol { \theta } } ( \boldsymbol { x } , t )$ is colinear with x. In particular, we can express the i-th iterate as

$$
\begin{array} { r } { \pmb { x } _ { t _ { i } } = \kappa _ { t _ { i } } \pmb { x } _ { t _ { 0 } } , } \end{array}
$$

We initialize the method with one DDIM step, so $\kappa _ { t _ { 0 } } = 1$ and $\kappa _ { t _ { 1 } }$ is given by (38) with $i = 1$ . In particular, (39) and smoothness give $0 \le \kappa _ { t _ { 1 } } ^ { \star } - \kappa _ { t _ { 1 } } = O ( \delta _ { 1 } ^ { 2 } )$ . For subsequent steps, the coeficient $\kappa _ { t _ { i } }$ satisfies that for any $i \geq 2$

$$
\begin{array} { r l } & { \kappa _ { t _ { i } } = \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } \kappa _ { t _ { i - 1 } } - \alpha _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - \lambda } \mathrm { d } \lambda \frac { \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } \kappa _ { t _ { i - 1 } } } { \alpha _ { t _ { i - 1 } } \left( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } \right) } } \\ & { ~ - \alpha _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } ( \lambda - \lambda _ { t _ { i - 1 } } ) \mathrm { e } ^ { - \lambda } \mathrm { d } \lambda \frac { \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } \kappa _ { t _ { i - 1 } } } { \lambda _ { t _ { i - 1 } } \left( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } \right) } - \frac { \mathrm { e } ^ { - \lambda _ { t _ { i - 2 } } } \kappa _ { t _ { i - 2 } } } { \alpha _ { t _ { i - 2 } } \left( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 2 } } } \right) } . } \end{array}\tag{47}
$$

Combining this with the recurrence for $\kappa _ { t _ { i } } ^ { \star }$ given in (37), we can decompose the diference $\kappa _ { t _ { i } } ^ { \star } - \kappa _ { t _ { i } }$ as follows for $i \geq 2 \colon$

$$
\kappa _ { t _ { i } } ^ { \star } - \kappa _ { t _ { i } } = a _ { i } \big ( \kappa _ { t _ { i - 1 } } ^ { \star } - \kappa _ { t _ { i - 1 } } \big ) + b _ { i } + c _ { i } ,\tag{48}
$$

where $a _ { i }$ is define in (40), and $b _ { i }$ and $c _ { i }$ are given by

$$
b _ { i } : = - \alpha _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } ( \lambda - \lambda _ { t _ { i - 1 } } ) \mathrm { e } ^ { - \lambda } \mathrm { d } \lambda \cdot \frac { 1 } { \lambda _ { t _ { i - 1 } } - \lambda _ { t _ { i - 2 } } } \Bigg ( \frac { \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } ( \kappa _ { t _ { i - 1 } } ^ { \star } - \kappa _ { t _ { i - 1 } } ) } { \alpha _ { t _ { i - 1 } } ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } ) } - \frac { \mathrm { e } ^ { - \lambda _ { t _ { i - 2 } } } \left( \kappa _ { t _ { i - 2 } } ^ { \star } - \kappa _ { t _ { i - 2 } } \right) } { \alpha _ { t _ { i - 2 } } ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 2 } } } ) } \Bigg ) ,\tag{49}
$$

$$
\begin{array} { r l } & { c _ { i } : = - \alpha _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \left( \frac { \mathrm { e } ^ { - 2 \lambda } } { \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } } \frac { \kappa _ { t _ { i - 1 } } ^ { \star } } { \alpha _ { t _ { i - 1 } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } } } - \frac { \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } - \lambda } \kappa _ { t _ { i - 1 } } ^ { \star } } { \alpha _ { t _ { i - 1 } } ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } ) } \right. } \\ & { \qquad \left. - ( \lambda - \lambda _ { t _ { i - 1 } } ) \mathrm { e } ^ { - \lambda } \frac { 1 } { \lambda _ { t _ { i - 1 } } - \lambda _ { t _ { i - 2 } } } \Big ( \frac { \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } \kappa _ { t _ { i - 1 } } ^ { \star } } { \alpha _ { t _ { i - 1 } } ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } ) } - \frac { \mathrm { e } ^ { - \lambda _ { t _ { i - 2 } } } \kappa _ { t _ { i - 2 } } ^ { \star } } { \alpha _ { t _ { i - 2 } } ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 2 } } } ) } \Big ) \right) \mathrm { d } \lambda . } \end{array}\tag{50}
$$

Applying recursion to (48) yields

$$
\begin{array} { r } { \kappa _ { t _ { M } } ^ { \star } - \kappa _ { t _ { M } } = \displaystyle \prod _ { j = 2 } ^ { M } a _ { j } \big ( \kappa _ { t _ { 1 } } ^ { \star } - \kappa _ { t _ { 1 } } \big ) } \\ { + \displaystyle \sum _ { i = 2 } ^ { M } \displaystyle \prod _ { j = i + 1 } ^ { M } a _ { j } \big ( b _ { i } + c _ { i } \big ) . } \end{array}\tag{51}
$$

In what follows, we shall analyze $c _ { i } , b _ { i }$ , and $a _ { i }$ separately.

• Controlling $c _ { i } .$ For simplicity of notation, define the auxiliary function

$$
f ( \lambda ) : = \frac { \mathrm { e } ^ { - \lambda } } { \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } } .
$$

Using the expression of $\kappa _ { t _ { j } }$ in (35) , the following factors appearing in the definition of $c _ { i }$ can be rewritten in terms of $f ( \cdot )$ for any $i \geq 2$

$$
\begin{array} { r l } & { \frac { \mathrm { e } ^ { - \lambda _ { t _ { j } } } \kappa _ { t _ { j } } ^ { \star } } { \alpha _ { t _ { j } } ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { j } } } ) } = \frac { 1 } { \alpha _ { t _ { 0 } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { 0 } } } } } \frac { \mathrm { e } ^ { - \lambda _ { t _ { j } } } } { \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { j } } } } } = \frac { f ( \lambda _ { t _ { j } } ) } { \alpha _ { t _ { 0 } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { 0 } } } } } , \quad j = i , i - 1 , i - 2 , } \\ & { \frac { \mathrm { e } ^ { - 2 \lambda } } { \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } } \frac { \kappa _ { t _ { i - 1 } } ^ { \star } } { \alpha _ { t _ { i - 1 } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } } } = \frac { \mathrm { e } ^ { - \lambda _ { f } } ( \lambda ) } { \alpha _ { t _ { 0 } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { 0 } } } } } . } \end{array}\tag{52}
$$

Consequently, the term $c _ { i }$ can be expressed as:

$$
c _ { i } = - \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { 0 } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { 0 } } } } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - \lambda } \bigg ( f ( \lambda ) - f ( \lambda _ { t _ { i - 1 } } ) - ( \lambda - \lambda _ { t _ { i - 1 } } ) \frac { f ( \lambda _ { t _ { i - 1 } } ) - f ( \lambda _ { t _ { i - 2 } } ) } { \lambda _ { t _ { i - 1 } } - \lambda _ { t _ { i - 2 } } } \bigg ) \mathrm { d } \lambda .\tag{53}
$$

The derivative of $f ( \lambda )$ can be computed as

$$
f ^ { \prime } ( \lambda ) = - \frac { \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } } { ( \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } + 1 ) ^ { \frac { 3 } { 2 } } } , \qquad f ^ { \prime \prime } ( \lambda ) = - \frac { \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } ( 2 - \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } ) } { ( \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } + 1 ) ^ { \frac { 5 } { 2 } } } .\tag{54}
$$

By a second-order Taylor’s expansion and the fact that $\delta _ { i - 1 } = O ( M ^ { - 1 } )$ and $\delta _ { i } = O ( M ^ { - 1 } )$ , we obtain

$$
\begin{array} { r l } & { f ( \lambda ) - f ( \lambda _ { t _ { i - 1 } } ) - ( \lambda - \lambda _ { t _ { i - 1 } } ) \frac { f ( \lambda _ { t _ { i - 1 } } ) - f ( \lambda _ { t _ { i - 2 } } ) } { \lambda _ { t _ { i - 1 } } - \lambda _ { t _ { i - 2 } } } } \\ & { \overset { \mathrm { ( a ) } } { = } ( \lambda - \lambda _ { t _ { i - 1 } } ) \bigg ( f ^ { \prime } ( \lambda _ { t _ { i - 1 } } ) - \frac { f ( \lambda _ { t _ { i - 1 } } ) - f ( \lambda _ { t _ { i - 2 } } ) } { \lambda _ { t _ { i - 1 } } - \lambda _ { t _ { i - 2 } } } \bigg ) + \frac { 1 } { 2 } f ^ { \prime \prime } ( \lambda _ { t _ { i - 1 } } ) ( \lambda - \lambda _ { t _ { i - 1 } } ) ^ { 2 } + O ( M ^ { - 3 } ) } \\ & { \overset { \mathrm { ( b ) } } { = } \frac { 1 } { 2 } f ^ { \prime \prime } ( \lambda _ { t _ { i - 1 } } ) ( \lambda _ { t _ { i - 1 } } - \lambda _ { t _ { i - 2 } } ) ( \lambda - \lambda _ { t _ { i - 1 } } ) + \frac { 1 } { 2 } f ^ { \prime \prime } ( \lambda _ { t _ { i - 1 } } ) ( \lambda - \lambda _ { t _ { i - 1 } } ) ^ { 2 } + O ( M ^ { - 3 } ) } \\ & { = \frac { 1 } { 2 } f ^ { \prime \prime } ( \lambda _ { t _ { i - 1 } } ) ( \lambda - \lambda _ { t _ { i - 2 } } ) ( \lambda - \lambda _ { t _ { i - 1 } } ) + O ( M ^ { - 3 } ) . } \end{array}
$$

where (a) holds because $\begin{array} { r } { f ( \lambda ) - f ( \lambda _ { t _ { i - 1 } } ) = f ^ { \prime } ( \lambda _ { t _ { i - 1 } } ) ( \lambda - \lambda _ { t _ { i - 1 } } ) + \frac { 1 } { 2 } f ^ { \prime \prime } ( \lambda _ { t _ { i - 1 } } ) ( \lambda - \lambda _ { t _ { i - 1 } } ) ^ { 2 } + O ( M ^ { - 3 } ) } \end{array}$ , and (b) is true because $f ( \lambda _ { t _ { i - 1 } } ) - f ( \lambda _ { t _ { i - 2 } } ) = f ^ { \prime } ( \lambda _ { t _ { i - 1 } } ) ( \lambda _ { t _ { i - 1 } } - \lambda _ { t _ { i - 2 } } ) - { \frac { \sim } { 2 } } f ^ { \prime \prime } ( \lambda _ { t _ { i - 1 } } ) ( \lambda _ { t _ { i - 1 } } - \lambda _ { t _ { i - 2 } } ) ^ { 2 } + O ( M ^ { - 3 } )$ Substituting this into (53) results in the following bound for c<sub>i</sub>:

$$
\begin{array} { c } { { c _ { i } \overset { \mathrm { ( a ) } } { = } - \displaystyle \frac { \alpha _ { t _ { i } } f ^ { \prime \prime } ( \lambda _ { t _ { i - 1 } } ) } { 2 \alpha _ { t _ { 0 } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { 0 } } } } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - \lambda } ( \lambda - \lambda _ { t _ { i - 2 } } ) ( \lambda - \lambda _ { t _ { i - 1 } } ) \mathrm { d } \lambda + O ( M ^ { - 4 } ) } } \\ { { \overset { \mathrm { ( b ) } } { = } - \displaystyle \frac { \alpha _ { t _ { i } } \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } f ^ { \prime \prime } ( \lambda _ { t _ { i - 1 } } ) } { 2 \alpha _ { t _ { 0 } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { 0 } } } } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } ( \lambda - \lambda _ { t _ { i - 2 } } ) ( \lambda - \lambda _ { t _ { i - 1 } } ) \mathrm { d } \lambda + O ( M ^ { - 4 } ) } } \end{array}
$$

$$
\begin{array} { r l } & { = - \frac { \alpha _ { t _ { i } } \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } f ^ { \prime \prime } ( \lambda _ { t _ { i - 1 } } ) } { 2 \alpha _ { t _ { 0 } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { 0 } } } } } \frac { 1 } { 6 } \delta _ { i } ^ { 2 } \bigg ( 2 \delta _ { i } + 3 \delta _ { i - 1 } \bigg ) + O ( M ^ { - 4 } ) } \\ & { = - \frac { \alpha _ { t _ { i } } \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } f ^ { \prime \prime } ( \lambda _ { t _ { i - 1 } } ) } { 1 2 \alpha _ { t _ { 0 } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { 0 } } } } } \delta _ { i } ^ { 2 } \bigg ( 2 \delta _ { i } + 3 \delta _ { i - 1 } \bigg ) + O ( M ^ { - 4 } ) , } \end{array}\tag{55}
$$

where (a) holds because

$$
\left| \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { 0 } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { 0 } } } } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - \lambda } O ( M ^ { - 3 } ) \mathrm { d } \lambda \right| \leq \frac { \alpha _ { t _ { i } } \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } } { \alpha _ { t _ { 0 } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { 0 } } } } } \delta _ { i } \cdot O ( M ^ { - 3 } ) = O ( M ^ { - 4 } ) ,
$$

and (b) is true because

$$
\begin{array}{c} \Biggl | \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } ( \mathrm { e } ^ { - \lambda } - \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } ) ( \lambda - \lambda _ { t _ { i - 2 } } ) ( \lambda - \lambda _ { t _ { i - 1 } } ) \mathrm { d } \lambda \Biggr | \leq ( \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } - \mathrm { e } ^ { - \lambda _ { t _ { i } } } ) \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } ( \lambda - \lambda _ { t _ { i - 2 } } ) ( \lambda - \lambda _ { t _ { i - 1 } } ) \mathrm { d } \lambda  \\ { \leq \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } \delta _ { i } \frac { 1 } { 6 } \delta _ { i } ^ { 2 } \biggl ( 2 \delta _ { i } + 3 \delta _ { i - 1 } \biggr ) = O ( M ^ { - 4 } ) . } \end{array}
$$

• Controlling $b _ { i }$ . For simplicity of notation, we introduce a sequence

$$
g _ { t _ { i } } : = \frac { \mathrm { e } ^ { - \lambda _ { t _ { i } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } } \frac { \kappa _ { t _ { i } } ^ { \star } - \kappa _ { t _ { i } } } { \alpha _ { t _ { i } } } .
$$

By definition, $b _ { i }$ can be expressed in terms of $g _ { t _ { i } }$ and $g _ { t _ { i - 2 } }$ as:

$$
b _ { i } = - \alpha _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } ( \lambda - \lambda _ { t _ { i - 1 } } ) \mathrm { e } ^ { - \lambda } \mathrm { d } \lambda \frac { g _ { t _ { i - 1 } } - g _ { t _ { i - 2 } } } { \lambda _ { t _ { i - 1 } } - \lambda _ { t _ { i - 2 } } } .\tag{56}
$$

Therefore it sufices to analyze the term $\frac { g _ { t _ { i - 1 } } - g _ { t _ { i - 2 } } } { \lambda _ { t _ { i - 1 } } - \lambda _ { t _ { i - 2 } } }$ . Leveraging that $\begin{array} { r } { \kappa _ { t _ { i } } ^ { \star } - \kappa _ { t _ { i } } = \alpha _ { t _ { i } } \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } } { \mathrm { e } ^ { - \lambda _ { t _ { i } } } } g _ { t _ { i } } } \end{array}$ recurrence for $\kappa _ { t _ { i } } ^ { \star } - \kappa _ { t _ { i } }$ in (48) also establishes a recurrence for $g _ { t _ { i } }$ as:

$$
\begin{array} { r l } & { g _ { t _ { i } } = \frac { \alpha _ { t _ { i - 1 } } \mathrm { e } ^ { - \lambda _ { t _ { i } } } ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } ) } { \alpha _ { t _ { i } } ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } ) \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } } a _ { i } g _ { t _ { i - 1 } } } \\ & { \phantom { g _ { t _ { i } } = } - \frac { \mathrm { e } ^ { - \lambda _ { t _ { i } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } ( \lambda - \lambda _ { t _ { i - 1 } } ) \mathrm { e } ^ { - \lambda } \mathrm { d } \lambda \frac { g _ { t _ { i - 1 } } - g _ { t _ { i - 2 } } } { \lambda _ { t _ { i - 1 } } - \lambda _ { t _ { i - 2 } } } } \\ & { \phantom { g _ { t _ { i } } = } + \frac { \mathrm { e } ^ { - \lambda _ { t _ { i } } } } { \alpha _ { t _ { i } } ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } ) } c _ { i } . } \end{array}\tag{57}
$$

We further have

$$
\begin{array} { r l } & { \frac { g _ { t _ { i } } - g _ { t _ { i - 1 } } } { \lambda _ { t _ { i } } - \lambda _ { t _ { i - 1 } } } = \left( \frac { \alpha _ { t _ { i - 1 } } \mathrm { e } ^ { - \lambda _ { t _ { i } } } \left( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } \right) } { \alpha _ { t _ { i } } \left( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } \right) \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } } a _ { i } - 1 \right) \frac { g _ { t _ { i - 1 } } } { \delta _ { i } } } \\ & { \phantom { \frac { g _ { t _ { i } } - g _ { t _ { i - 1 } } } { \lambda _ { t _ { i } } } = } - \frac { \mathrm { e } ^ { - \lambda _ { t _ { i } } } } { \delta _ { i } \left( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } \right) } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } ( \lambda - \lambda _ { t _ { i - 1 } } ) \mathrm { e } ^ { - \lambda } \mathrm { d } \lambda \frac { g _ { t _ { i - 1 } } - g _ { t _ { i - 2 } } } { \lambda _ { t _ { i - 1 } } - \lambda _ { t _ { i - 2 } } } } \\ & { \phantom { \frac { g _ { t _ { i } } } { \lambda _ { t _ { i } } } = } + \frac { \mathrm { e } ^ { - \lambda _ { t _ { i } } } } { \alpha _ { t _ { i } } \left( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } \right) } \frac { c _ { i } } { \delta _ { i } } . } \end{array}\tag{58}
$$

Let us control three quantities on the right-hand-side of (58) separately. Regarding the third term, identity (55) gives

$$
\frac { \mathrm { e } ^ { - \lambda _ { t _ { i } } } } { \alpha _ { t _ { i } } ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } ) } \frac { \lvert c _ { i } \rvert } { \delta _ { i } } \leq \frac { \frac { \mathrm { e } ^ { - \lambda _ { t _ { i } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } } \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } \lvert f ^ { \prime \prime } ( \lambda _ { t _ { i - 1 } } ) \rvert } { 1 2 \alpha _ { t _ { 0 } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { 0 } } } } } \delta _ { i } \bigg ( 2 \delta _ { i } + 3 \delta _ { i - 1 } \bigg ) + O ( M ^ { - 3 } ) = O ( M ^ { - 2 } ) .\tag{59}
$$

Regarding the second term, provided that $\delta _ { i } = O ( M ^ { - 1 } ) \leq \log 2$ , the coeficient before $\frac { g _ { t _ { i - 1 } } - g _ { t _ { i - 2 } } } { \lambda _ { t _ { i - 1 } } - \lambda _ { t _ { i - 2 } } }$ is bounded by:

$$
\frac { \mathrm { e } ^ { - \lambda _ { t _ { i } } } } { \delta _ { i } ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } ) } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } { ( \lambda - \lambda _ { t _ { i - 1 } } ) \mathrm { e } ^ { - \lambda } \mathrm { d } \lambda } \leq \frac { \mathrm { e } ^ { - \lambda _ { t _ { i } } - \lambda _ { t _ { i - 1 } } } } { 2 ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } ) } \delta _ { i } \leq \frac { 1 } { 2 } \mathrm { e } ^ { \delta _ { i } } \delta _ { i } \leq \delta _ { i } .\tag{60}
$$

Moreover, by the definition of $a _ { i }$ in (40), the first term is simplified as

$$
\begin{array} { r l } & { \frac { 1 } { \delta _ { i } } ( \frac { \alpha _ { t _ { i - 1 } } \mathrm { e } ^ { - \lambda _ { t _ { i } } } ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } ) } { \alpha _ { t _ { i } } ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } ) \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } } a _ { i } - 1 ) } \\ & { \quad \quad = \frac { 1 } { \delta _ { i } } ( \frac { \mathrm { e } ^ { - \lambda _ { t _ { i } } } ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } ) } { ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } ) \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } } - \frac { \mathrm { e } ^ { - \lambda _ { t _ { i } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - \lambda } \mathrm { d } \lambda - 1 ) } \\ &  \quad \quad = \frac { 1 } { \delta _ { i } } ( \frac { \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } ( 1 - \mathrm { e } ^ { - \lambda _ { t _ { i } } } ) ( \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } - \lambda _ { t _ { i } } } - \gamma ^ { 2 } ) } { ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } ) \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } } - \frac { \mathrm { e } ^ { - \lambda _ { t _ { i } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm  \end{array}\tag{61}
$$

where the second line holds because

$$
\begin{array} { r l } & { \frac { \mathrm { e } ^ { - \lambda \iota _ { i } } ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { i - 1 } } ) } { ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { i } } ) \mathrm { e } ^ { - \lambda \iota _ { i - 1 } } } - 1 = \frac { \mathrm { e } ^ { - \lambda \iota _ { i } } ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { i - 1 } } ) - ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { i } } ) \mathrm { e } ^ { - \lambda _ { i - 1 } } } { ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { i } } ) \mathrm { e } ^ { - \lambda \iota _ { i - 1 } } } } \\ & { \phantom { m m m m m m m m } = \frac { \gamma ^ { 2 } ( \mathrm { e } ^ { - \lambda \iota _ { i } } - \mathrm { e } ^ { - \lambda \iota _ { i - 1 } } ) + \mathrm { e } ^ { - \lambda \iota _ { i } } \mathrm { e } ^ { - \lambda \iota _ { i - 1 } } ( \mathrm { e } ^ { - \lambda \iota _ { i - 1 } } - \mathrm { e } ^ { - \lambda \iota _ { i } } ) } { ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { i } } ) \mathrm { e } ^ { - \lambda \iota _ { i - 1 } } } } \\ & { \phantom { m m m m m m m m } = \frac { ( \mathrm { e } ^ { - \lambda \iota _ { i - 1 } } - \mathrm { e } ^ { - \lambda \iota _ { i } } ) ( \mathrm { e } ^ { - \lambda \iota _ { i } } \mathrm { e } ^ { - \lambda \iota _ { i - 1 } } - \gamma ^ { 2 } ) } { ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { i } } ) \mathrm { e } ^ { - \lambda \iota _ { i - 1 } } } } \\ &  \phantom { m m m m m m m } = \frac  \mathrm { e } ^ { - \lambda \iota _ { i - 1 } } ( 1 - \mathrm { e } ^ { - \lambda _ { i } } ) ( \mathrm { e } ^ { - \lambda \iota _ { i } } \mathrm  e \end{array}\tag{62}
$$

and the last identity holds since

$$
\begin{array} { r l } & { \frac { \mathrm { e } ^ { - \lambda \tau _ { i - 1 } } \left( 1 - \mathrm { e } ^ { - \lambda _ { i } } \right) \left( \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } - \lambda _ { t _ { i } } } - \gamma ^ { 2 } \right) } { \left( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } \right) \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } } - \frac { \mathrm { e } ^ { - \lambda _ { t _ { i } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - \lambda } \mathrm { d } \lambda } \\ & { \quad = \frac { \left( 1 - \mathrm { e } ^ { - \delta _ { i } } \right) \left( \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } - \lambda _ { t _ { i } } } - \gamma ^ { 2 } \right) } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } } - \frac { \mathrm { e } ^ { - \lambda _ { t _ { i } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } } ( \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } - \mathrm { e } ^ { - \lambda _ { t _ { i } } } ) } \\ &  \quad = \frac { - \gamma ^ { 2 } \left( 1 - \mathrm { e } ^ { - \delta _ { i } } \right) } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } } + \frac { \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } - \lambda _ { t _ { i } } } - \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } - \mathrm { e } ^ { - \lambda _ { t _ { i } } } ( \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } - \mathrm { e } ^ { - \lambda _ { t _ { i } } } ) }  \gamma ^ { 2 }  \end{array}
$$

Equation (61) can be further simplified by leveraging the fact that $1 - \mathrm { e } ^ { - \delta _ { i } } \leq \delta _ { i }$

$$
\left| \frac { 1 } { \delta _ { i } } \left( \frac { \alpha _ { t _ { i - 1 } } \mathrm { e } ^ { - \lambda _ { t _ { i } } } ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } ) } { \alpha _ { t _ { i } } ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } ) \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } } a _ { i } - 1 \right) \right| \leq \frac { \gamma ^ { 2 } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } } \leq 1 .\tag{63}
$$

Finally, plugging the above bounds (60), (59), and (63) into (58), we find that the absolute value of $\frac { g _ { t _ { i } } - g _ { t _ { i - 1 } } } { \lambda _ { t _ { i } } - \lambda _ { t _ { i - 1 } } }$ is bounded by

$$
\left| \frac { g _ { t _ { i } } - g _ { t _ { i - 1 } } } { \lambda _ { t _ { i } } - \lambda _ { t _ { i - 1 } } } \right| \le \delta _ { i } \left| \frac { g _ { t _ { i - 1 } } - g _ { t _ { i - 2 } } } { \lambda _ { t _ { i - 1 } } - \lambda _ { t _ { i - 2 } } } \right| + | g _ { t _ { i - 1 } } | + O ( M ^ { - 2 } ) .
$$

According to the fact that the convergence order of DPMSolver-2 is 2 (Lu et al., 2022a, Theorem 3.2),(Zheng et al., 2023, Theorem 3.3), we have $g _ { t _ { i - 1 } } ~ = ~ O ( M ^ { - 2 } )$ . The DDIM initialization gives $g _ { t _ { 0 } } = 0$ and $g _ { t _ { 1 } } = O ( \delta _ { 1 } ^ { 2 } )$ , so $( g _ { t _ { 1 } } - g _ { t _ { 0 } } ) / \delta _ { 1 } = O ( \delta _ { 1 } )$ . Applying the recursion gives:

$$
\left| \frac { g _ { t _ { i } } - g _ { t _ { i - 1 } } } { \lambda _ { t _ { i } } - \lambda _ { t _ { i - 1 } } } \right| = O ( M ^ { - 2 } ) , \quad 2 \leq i \leq M , \qquad \mathrm { a n d } \quad \frac { g _ { t _ { 1 } } - g _ { t _ { 0 } } } { \delta _ { 1 } } = O ( M ^ { - 1 } ) .\tag{64}
$$

Substituting (64) into (56), we bound the term $| b _ { i } |$ for $i \geq 3$ by

$$
\begin{array} { r l } & { \left| b _ { i } \right| \leq \alpha _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } ( \lambda - \lambda _ { t _ { i - 1 } } ) \mathrm { e } ^ { - \lambda } \mathrm { d } \lambda \left| \frac { g _ { t _ { i - 1 } } - g _ { t _ { i - 2 } } } { \lambda _ { t _ { i - 1 } } - \lambda _ { t _ { i - 2 } } } \right| } \\ & { \quad \leq \frac { 1 } { 2 } \alpha _ { t _ { i } } \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } \delta _ { i } ^ { 2 } \left| \frac { g _ { t _ { i - 1 } } - g _ { t _ { i - 2 } } } { \lambda _ { t _ { i - 1 } } - \lambda _ { t _ { i - 2 } } } \right| = O \left( M ^ { - 4 } \right) , \qquad i \geq 3 . } \end{array}\tag{65}
$$

For $i = 2$ , the initial bound (64) gives

$$
| b _ { 2 } | = O ( \delta _ { 2 } ^ { 2 } \delta _ { 1 } ) = O ( M ^ { - 3 } ) .
$$

The total contribution of the $b _ { i }$ terms satisfies

$$
\sum _ { i = 2 } ^ { M } \left| \left( \prod _ { j = i + 1 } ^ { M } a _ { j } \right) b _ { i } \right| \leq C \big ( O ( M ^ { - 3 } ) + M \cdot O ( M ^ { - 4 } ) \big ) = O ( M ^ { - 3 } ) .
$$

• Controlling $a _ { i }$ . For simplicity of notation, let us denote

$$
\tilde { \delta } _ { i } : = \frac { \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - \lambda } \mathrm { d } \lambda .
$$

By the definition of $a _ { i }$ in (40), we have

$$
\prod _ { j = i + 1 } ^ { M } a _ { j } = \frac { \alpha _ { t _ { M } } } { \alpha _ { t _ { i } } } \prod _ { j = i + 1 } ^ { M } \left( 1 - \tilde { \delta } _ { j } \right) \stackrel { \left( \mathrm { a } \right) } { = } \frac { \alpha _ { t _ { M } } } { \alpha _ { t _ { i } } } \exp \Bigg ( - \sum _ { j = i + 1 } ^ { M } \tilde { \delta } _ { j } + O \big ( \sum _ { j = i + 1 } ^ { M } \tilde { \delta } _ { j } ^ { 2 } \big ) \Bigg ) ,\tag{66}
$$

where (a) holds because

$$
\log ( 1 - \tilde { \delta } _ { j } ) = - \tilde { \delta } _ { j } + O ( \tilde { \delta } _ { j } ^ { 2 } ) ,\tag{67}
$$

We can compute the sum of $\tilde { \delta } _ { j }$ and $\tilde { \delta } _ { j } ^ { 2 }$ as

$$
\begin{array} { l } { { \displaystyle \sum _ { j = i + 1 } ^ { M } \widetilde \delta _ { j } = \sum _ { j = i + 1 } ^ { M } \frac { \mathrm { e } ^ { - \lambda _ { t _ { j - 1 } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { j - 1 } } } } \displaystyle \int _ { \lambda _ { t _ { j - 1 } } } ^ { \lambda _ { t _ { j } } } \mathrm { e } ^ { - \lambda } \mathrm { d } \lambda \stackrel { \mathrm { ( a ) } } = \sum _ { j = i + 1 } ^ { M } \displaystyle \int _ { \lambda _ { t _ { j - 1 } } } ^ { \lambda _ { t _ { j } } } \frac { \mathrm { e } ^ { - 2 \lambda } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } \mathrm { d } \lambda + O ( M ^ { - 1 } ) } } \\ { { \displaystyle \quad \quad = \int _ { \lambda _ { t _ { i } } } ^ { \lambda _ { t _ { M } } } \frac { \mathrm { e } ^ { - 2 \lambda } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } \mathrm { d } \lambda + O ( M ^ { - 1 } ) } } \\ { { \displaystyle \quad \quad \stackrel { \mathrm { ( b ) } } = \frac 1 2 \log \left( \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { M } } } } \right) + O ( M ^ { - 1 } ) , } } \end{array}\tag{68}
$$

where (a) holds because

$$
\begin{array}{c} \int _ { \lambda _ { i j - 1 } } ^ { \lambda _ { i j } } \left| \frac { \mathrm { e } ^ { - \lambda _ { i j - 1 } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { i j - 1 } } } - \frac { \mathrm { e } ^ { - \lambda } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } \right| \mathrm { e } ^ { - \lambda } \mathrm { d } \lambda \le \mathrm { e } ^ { - \lambda _ { i j - 1 } } \operatorname* { m a x } _ { \lambda \in [ \lambda _ { i j - 1 } , \lambda _ { i j } ] } \left| \frac { \mathrm { d } } { \mathrm { d } \lambda } \frac { \mathrm { e } ^ { - \lambda } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } \right| \int _ { \lambda _ { i j - 1 } } ^ { \lambda _ { i j } } ( \lambda - \lambda _ { t _ { j - 1 } } ) \mathrm { d } \lambda  \\ { = \frac { \delta _ { j } ^ { 2 } } { 2 } \mathrm { e } ^ { - \lambda _ { i j - 1 } } \operatorname* { m a x } _ { \lambda \in [ \lambda _ { i j - 1 } , \lambda _ { i j } ] } \frac { \mathrm { e } ^ { - \lambda } | \gamma ^ { 2 } - \mathrm { e } ^ { - 2 \lambda } | } { ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } ) ^ { 2 } } \le \frac { \mathrm { e } ^ { \delta _ { j } } } { 2 } \delta _ { j } ^ { 2 } = O ( M ^ { - 2 } ) , } \end{array}
$$

and (b) arises from

$$
\int _ { \lambda _ { t _ { i } } } ^ { \lambda _ { t _ { M } } } \frac { \mathrm { e } ^ { - 2 \lambda } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } \mathrm { d } \lambda = \frac { 1 } { 2 } \log ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } ) \bigg | _ { \lambda _ { t _ { M } } } ^ { \lambda _ { t _ { i } } } = \frac { 1 } { 2 } \log \bigg ( \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { M } } } } \bigg ) .
$$

In addition, the sum of $\tilde { \delta } _ { j } ^ { 2 }$ is computed as

$$
\sum _ { j = i + 1 } ^ { M } \tilde { \delta } _ { j } ^ { 2 } \leq \sum _ { i = 1 } ^ { M } \frac { \mathrm { e } ^ { - 2 \lambda t _ { i - 1 } } } { ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda t _ { i - 1 } } ) ^ { 2 } } \bigg ( \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - \lambda } \mathrm { d } \lambda \bigg ) ^ { 2 } \leq \sum _ { i = 1 } ^ { M } \frac { \delta _ { i } ^ { 2 } \mathrm { e } ^ { - 4 \lambda t _ { i - 1 } } } { ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda t _ { i - 1 } } ) ^ { 2 } } \leq \sum _ { i = 1 } ^ { M } \delta _ { i } ^ { 2 } = O ( M ^ { - 1 } ) .\tag{69}
$$

Substituting (68) and (69) into (66), we have

$$
\prod _ { j = i + 1 } ^ { M } a _ { j } = \frac { \alpha _ { t _ { M } } } { \alpha _ { t _ { i } } } \sqrt { \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { M } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } } } \bigg ( 1 + O ( M ^ { - 1 } ) \bigg ) .\tag{70}
$$

Since $a _ { j } > 0$ and $\kappa _ { t _ { 1 } } ^ { \star } - \kappa _ { t _ { 1 } } \geq 0 .$ , the initialization term in (51) is nonnegative and can be omitted when deriving a lower bound. Finally, plugging the above bounds (70), (65), and (55) into (51), the discretization error at the last step is bounded below by

$$
\begin{array} { l } { { \kappa _ { t _ { M } } ^ { * } - \kappa _ { t _ { M } } \geq - \frac { \alpha _ { t _ { M } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { M } } } } } { 1 2 \alpha _ { t _ { 0 } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { 0 } } } } } \displaystyle \sum _ { i = 2 } ^ { M } \frac { \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } f ^ { \prime \prime } ( \lambda _ { t _ { i - 1 } } ) } { \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } } } \delta _ { i } ^ { 2 } \big ( 2 \delta _ { i } + 3 \delta _ { i - 1 } \big ) + { \cal O } ( M ^ { - 3 } ) } } \\ { { = \frac { \alpha _ { t _ { M } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { M } } } } } { 1 2 \alpha _ { t _ { 0 } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { 0 } } } } } \displaystyle \sum _ { i = 2 } ^ { M } \frac { \mathrm { e } ^ { - \lambda _ { t _ { i - 1 } } } } { \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } } } \frac { \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { i - 1 } } } \big ( 2 - \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { i - 1 } } } \big ) } { \big ( \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { i - 1 } } } + 1 \big ) ^ { \frac { 5 } { 2 } } } \delta _ { i } ^ { 2 } \big ( 2 \delta _ { i } + 3 \delta _ { i - 1 } \big ) + { \cal O } ( M ^ { - 3 } ) } } \\   = \frac { \alpha _ { t _ { M } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { M } } } } }  \end{array}
$$

where the second line inserts the second-order derivative of $f$ from (54). Taking $\gamma = \mathrm { e } ^ { - \lambda _ { t _ { M } } }$ , we have $2 - \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { i - 1 } } } \geq 0$ for all $0 \leq i - 1 \leq M$ . Therefore, we arrive at the desired result:

$$
\begin{array} { r l } & { \kappa _ { t _ { M } } ^ { \star } - \kappa _ { t _ { M } } \geq \displaystyle \frac { \alpha _ { t _ { M } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { M } } } } } { 1 2 \alpha _ { t _ { 0 } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { 0 } } } } } \displaystyle \sum _ { i \in \mathcal { S } } \frac { \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { i - 1 } } } \left( 2 - \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { i - 1 } } } \right) } { ( \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { i - 1 } } } + 1 ) ^ { 3 } } \delta _ { i } ^ { 2 } \bigg ( 2 \delta _ { i } + 3 \delta _ { i - 1 } \bigg ) + O ( M ^ { - 3 } ) } \\ & { \qquad \geq \displaystyle \frac { 2 \alpha _ { t _ { M } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { M } } } } \gamma ^ { 2 } c _ { 1 } ^ { 2 } ( 2 - \gamma ^ { 2 } c _ { 2 } ^ { 2 } ) } { 1 2 \alpha _ { t _ { 0 } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { 0 } } } } ( \gamma ^ { 2 } c _ { 2 } ^ { 2 } + 1 ) ^ { 3 } } \displaystyle \sum _ { i \in \mathcal { S } } \delta _ { i } ^ { 3 } + O ( M ^ { - 3 } ) } \\ &  \qquad \geq \displaystyle \frac { 2 \alpha _ { t _ { M } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { M } } } } \gamma ^ { 2 } c _ { 1 } ^ { 2 } ( 2 - \gamma ^ { 2 } c _ { 2 } ^ { 2 } ) } { 1 2 \alpha _ { t _ { 0 } } \sqrt { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { 0 } } } } ( \gamma ^ { 2 } c _ { 2 } ^ { 2 } + 1 ) ^ { 3 } } \displaystyle \frac  \end{array}
$$

where $s$ is defined in (44), and the last inequality uses Holder inequality that

$$
{ \frac { 1 } { 2 } } ( \log c _ { 2 } - \log c _ { 1 } ) \leq \sum _ { i \in { \mathcal { S } } } \delta _ { i } \leq \left( \sum _ { i \in { \mathcal { S } } } \delta _ { i } ^ { 3 } \right) ^ { \frac { 1 } { 3 } } \left( \sum _ { i \in { \mathcal { S } } } 1 \right) ^ { \frac { 2 } { 3 } } \leq \left( \sum _ { i \in { \mathcal { S } } } \delta _ { i } ^ { 3 } \right) ^ { \frac { 1 } { 3 } } M ^ { \frac { 2 } { 3 } } .
$$

This finishes the proof.

## C Proof of stochastic samplers convergence upper bounds (Theorem 2)

We begin by analyzing the discretization scheme in (17a).

## C.1 Convergence analysis of SDE-DPMSolver++1

We first state a preliminary moment bound, whose proof is deferred to the end of this section:

$$
\mathbb { E } [ \| \pmb { x } _ { t _ { i } } ^ { \star } \| _ { 2 } ^ { 2 } ] = O ( 1 ) , \qquad 0 \le i \le M .\tag{71}
$$

Equipped with this estimate, we proceed to establish (17a) in four steps.

Step 1: decompose the error $\pmb { x } _ { t _ { i } } - \pmb { x } _ { t _ { i } } ^ { \star }$ . Subtracting (29) from (17a) gives

$$
\begin{array} { r l } & { x _ { t _ { i } } - x _ { t _ { i } } ^ { \star } = \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \mathrm { e } ^ { - \delta _ { i } } \bigl ( x _ { t _ { i - 1 } } - x _ { t _ { i - 1 } } ^ { \star } \bigr ) + 2 \alpha _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i } } - \lambda ) } \bigl ( \mu ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) - \mu ( x _ { t ( \lambda ) } ^ { \star } , t ( \lambda ) ) \bigr ) \mathrm { d } \lambda } \\ & { \qquad = \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \mathrm { e } ^ { - \delta _ { i } } \bigl ( x _ { t _ { i - 1 } } - x _ { t _ { i - 1 } } ^ { \star } \bigr ) + \alpha _ { t _ { i } } \bigl ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } \bigr ) \bigl ( \mu ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) - \mu ( x _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } ) \bigr ) } \\ & { \qquad + 2 \alpha _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i } } - \lambda ) } \bigl ( \mu ( x _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } ) - \mu ( x _ { t ( \lambda ) } ^ { \star } , t ( \lambda ) ) \bigr ) \mathrm { d } \lambda . } \end{array}\tag{72}
$$

To treat the last integral, we expand the integrand $\mu ( \pmb { x } _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } ) - \pmb { \mu } ( \pmb { x } _ { t ( \lambda ) } ^ { \star } , t ( \lambda ) )$ via Taylor’s theorem:

$$
\begin{array} { r l } & { \mu ( { x } _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } ) - \mu ( { x } _ { t ( \lambda ) } ^ { \star } , t ( \lambda ) ) } \\ & { = \frac { \partial } { \partial \lambda } \mu ( { x } _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } ) ( \lambda _ { t _ { i - 1 } } - \lambda ) + \nabla _ { x } \mu ( { x } _ { t _ { i - 1 } } ^ { \star } , \lambda _ { t _ { i - 1 } } ) ^ { \top } ( { x } _ { t _ { i - 1 } } ^ { \star } - { x } _ { t ( \lambda ) } ^ { \star } ) + O ( \delta _ { i } ^ { 2 } ) + O ( \| { x } _ { t _ { i - 1 } } ^ { \star } - { x } _ { t ( \lambda ) } ^ { \star } \| _ { 2 } ^ { 2 } ) . } \end{array}\tag{73}
$$

From the SDE dynamics (29), we have

$$
\pmb { x } _ { t ( \lambda ) } ^ { \star } - \pmb { x } _ { t _ { i - 1 } } ^ { \star } = \biggl ( \frac { \sigma _ { t ( \lambda ) } } { \sigma _ { t _ { i - 1 } } } \mathrm { e } ^ { - ( \lambda - \lambda _ { t _ { i - 1 } } ) } - 1 \biggr ) \pmb { x } _ { t _ { i - 1 } } ^ { \star } + 2 \alpha _ { t ( \lambda ) } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda } \mathrm { e } ^ { - 2 ( \lambda - \lambda ^ { \prime } ) } \mu ( \pmb { x } _ { t ( \lambda ^ { \prime } ) } ^ { \star } , t ( \lambda ^ { \prime } ) ) \mathrm { d } \lambda ^ { \prime } + \sqrt { 2 } \sigma _ { t ( \lambda ) } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda } \mathrm { e } ^ { - ( \lambda - \lambda ^ { \prime } ) } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ^ { \prime } ) } .
$$

Substituting this into (73) yields

$$
\begin{array} { r l } & { \mu ( x _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } ) - \mu ( x _ { t ( \lambda ) } ^ { \star } , t ( \lambda ) ) } \\ & { = \frac { \partial } { \partial \lambda } \mu ( x _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } ) ( \lambda _ { t _ { i - 1 } } - \lambda ) - \Big ( \frac { \sigma _ { t ( \lambda ) } } { \sigma _ { t _ { i - 1 } } } \mathrm { e } ^ { - ( \lambda - \lambda _ { t _ { i - 1 } } ) } - 1 \Big ) \nabla _ { \alpha } \mu ( x _ { t _ { i - 1 } } ^ { \star } , \lambda _ { t _ { i - 1 } } ) ^ { \top } x _ { t _ { i - 1 } } ^ { \star } } \\ &  \phantom { = } - 2 \alpha _ { t ( \lambda ) } \displaystyle \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda } \mathrm { e } ^ { - 2 ( \lambda - \lambda ^ { \prime } ) \nabla \mu ( x _ { t _ { i - 1 } } ^ { \star } , \lambda _ { t _ { i - 1 } } ) ^ { \top } \mu ( x _ { t ( \lambda ^ { \prime } ) } ^ { \star } , \lambda ^ { \prime } ) \mathrm { d } \lambda ^ { \prime } - \sqrt { 2 } \sigma _ { t ( \lambda ) } \nabla \mu ( x _ { t _ { i - 1 } } ^ { \star } , \lambda _ { t _ { i - 1 } } ) ^ { \top } \displaystyle \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda } \mathrm { e } ^ { - ( \lambda - \lambda ^ { \prime } ) } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ^ { \prime } ) } } \\ & { \phantom { = } + O ( \delta _ { i } ^ { 2 } ) + O ( \| x _ { t _ { i - 1 } } ^ { \star } - x _ { t ( \lambda ) } ^ { \star } \| _ { 2 } ^ { 2 } ) . } \end{array}
$$

We now estimate the integral appearing in the last term of (72). The contributions from the terms in (74) are bounded as follows. First,

$$
\begin{array} { r } { \Big \| \displaystyle \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i } } - \lambda ) } \frac { \partial } { \partial \lambda } \mu ( x _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } ) ( \lambda _ { t _ { i - 1 } } - \lambda ) \mathrm { d } \lambda \Big \| _ { 2 } \le L _ { t } \displaystyle \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i } } - \lambda ) } ( \lambda - \lambda _ { t _ { i - 1 } } ) \mathrm { d } \lambda } \\ { = L _ { t } \Big ( \frac { \delta _ { i } } { 2 } - \frac { 1 } { 4 } + \frac { 1 } { 4 } \mathrm { e } ^ { - 2 \delta _ { i } } \Big ) \stackrel { ( \mathrm { a } ) } { \le } L _ { t } \delta _ { i } ^ { 2 } , } \end{array}
$$

where (a) holds for suficiently small $\delta _ { i } \leq \delta .$ . Second,

$$
\begin{array} { r l } { \Big | \displaystyle \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i } } - \lambda ) } \Big ( \frac { \sigma _ { t ( \lambda ) } } { \sigma _ { t _ { i - 1 } } } \mathrm { e } ^ { - ( \lambda - \lambda _ { t _ { i - 1 } } ) } - 1 \Big ) \mathrm { d } \lambda \Big | \stackrel { \mathrm { ( b ) } } = \displaystyle \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i } } - \lambda ) } \Big ( 1 - \frac { \sigma _ { t ( \lambda ) } } { \sigma _ { t _ { i - 1 } } } \mathrm { e } ^ { - ( \lambda - \lambda _ { t _ { i - 1 } } ) } \Big ) \mathrm { d } \lambda } & { } \\ { \stackrel { \mathrm { ( c ) } } \le 2 \delta _ { i } \displaystyle \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i } } - \lambda ) } \mathrm { d } \lambda } & { } \\ { = \delta _ { i } \big ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } \big ) \le 2 \delta _ { i } ^ { 2 } , } \end{array}
$$

where the equality (b) uses $\lambda \leq \lambda _ { t } ,$ and $\sigma _ { t ( \lambda ) } / \sigma _ { t _ { i - 1 } } \leq 1$ , and the inequality (c) follows from

$$
\begin{array} { r l } & { 1 - \frac { \sigma _ { t ( \lambda ) } } { \sigma _ { t _ { i - 1 } } } \mathrm { e } ^ { - ( \lambda - \lambda _ { t _ { i - 1 } } ) } \le 1 - \frac { \alpha _ { t _ { i - 1 } } \sigma _ { t ( \lambda ) } } { \sigma _ { t _ { i - 1 } } \alpha _ { t ( \lambda ) } } \mathrm { e } ^ { - ( \lambda - \lambda _ { t _ { i - 1 } } ) } } \\ & { \qquad = 1 - \mathrm { e } ^ { \lambda _ { t _ { i - 1 } } - \lambda - ( \lambda - \lambda _ { t _ { i - 1 } } ) } = 1 - \mathrm { e } ^ { 2 ( \lambda _ { t _ { i - 1 } } - \lambda ) } \le 2 ( \lambda - \lambda _ { t _ { i - 1 } } ) \le 2 \delta _ { i } . } \end{array}\tag{75}
$$

Third, the integral involving the drift term is bounded by

$$
\begin{array} { r l } & { 2 \Big \| \displaystyle \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i } } - \lambda ) } \alpha _ { t ( \lambda ) } \displaystyle \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda } \mathrm { e } ^ { - 2 ( \lambda - \lambda ^ { \prime } ) } \nabla \mu ( x _ { t _ { i - 1 } } ^ { \star } , \lambda _ { t _ { i - 1 } } ) ^ { \top } \mu ( x _ { t ( \lambda ^ { \prime } ) } ^ { \star } , \lambda ^ { \prime } ) \mathrm { d } \lambda ^ { \prime } \mathrm { d } \lambda \Big \| _ { 2 } } \\ & { \quad \leq 2 \alpha _ { t _ { i } } \displaystyle \frac { L _ { x } } { \alpha _ { t _ { i - 1 } } } L _ { \mu } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i } } - \lambda ) } \displaystyle \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda } \mathrm { e } ^ { - 2 ( \lambda - \lambda ^ { \prime } ) } \mathrm { d } \lambda ^ { \prime } \mathrm { d } \lambda } \\ & { \quad = \alpha _ { t _ { i } } \displaystyle \frac { L _ { x } } { \alpha _ { t _ { i - 1 } } } L _ { \mu } \big ( \frac { 1 } { 2 } - \frac { 1 } { 2 } \mathrm { e } ^ { - 2 \delta _ { i } } - \delta _ { i } \mathrm { e } ^ { - 2 \delta _ { i } } \big ) \leq 2 \alpha _ { t _ { i } } \displaystyle \frac { L _ { x } } { \alpha _ { t _ { i - 1 } } } L _ { \mu } \delta _ { i } ^ { 2 } . } \end{array}
$$

Finally, the quadratic remainder term satisfies

$$
\delta _ { i } ^ { 2 } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i } } - \lambda ) } \mathrm { d } \lambda = \frac { \delta _ { i } ^ { 2 } } { 2 } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } ) \leq \delta _ { i } ^ { 3 } .
$$

Collecting these estimates, the integral in (72) reduces to

$$
\begin{array} { r l } & { \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i } } - \lambda ) } \big ( \mu ( x _ { t _ { i - 1 } } ^ { \star } , \lambda _ { t _ { i - 1 } } ) - \mu ( x _ { t ( \lambda ) } ^ { \star } , \lambda ) \big ) \mathrm { d } \lambda } \\ & { \quad = O ( \delta _ { i } ^ { 2 } ) + \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i } } - \lambda ) } O ( \| x _ { t _ { i - 1 } } ^ { \star } - x _ { t ( \lambda ) } ^ { \star } \| _ { 2 } ^ { 2 } ) \mathrm { d } \lambda - \nabla _ { x } \mu ( x _ { t _ { i - 1 } } ^ { \star } , \lambda _ { t _ { i - 1 } } ) ^ { \top } x _ { t _ { i - 1 } } ^ { \star } O ( \delta _ { i } ^ { 2 } ) } \\ & { \quad \quad - \sqrt { 2 } \nabla \mu ( x _ { t _ { i - 1 } } ^ { \star } , \lambda _ { t _ { i - 1 } } ) ^ { \top } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i } } - \lambda ) } \sigma _ { t ( \lambda ) } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda } \mathrm { e } ^ { - ( \lambda - \lambda ^ { \prime } ) } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ^ { \prime } ) } \mathrm { d } \lambda . } \end{array}\tag{76}
$$

Here, the notation $O ( \delta _ { i } ^ { 2 } )$ denotes a constant or a vector whose $\ell _ { 2 } { \mathrm { - n o r m } }$ is bounded by $c \delta _ { i } ^ { 2 }$ , where c is a constant independent of the number of iterations M. Then we have simplified the last term in the righthand-side of (72).

For the second term on the right-hand-side of (72), a direct Taylor expansion yields

$$
\begin{array} { r l } & { \alpha _ { t _ { i } } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } ) \big ( \mu ( x _ { t _ { i - 1 } } , \lambda _ { t _ { i - 1 } } ) - \mu ( x _ { t _ { i - 1 } } ^ { \star } , \lambda _ { t _ { i - 1 } } ) \big ) } \\ & { \quad = \alpha _ { t _ { i } } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } ) \nabla _ { x } \mu ( x _ { t _ { i - 1 } } ^ { \star } , \lambda _ { t _ { i - 1 } } ) ^ { \top } ( x _ { t _ { i - 1 } } - x _ { t _ { i - 1 } } ^ { \star } ) + O ( \delta _ { i } \| x _ { t _ { i - 1 } } - x _ { t _ { i - 1 } } ^ { \star } \| _ { 2 } ^ { 2 } ) , } \end{array}\tag{77}
$$

where the remainder estimate follows from the bound

$$
\alpha _ { t _ { i } } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } ) \leq 2 \alpha _ { t _ { i } } \delta _ { i } = O ( \delta _ { i } ) .
$$

Substituting (76) and (77) into (72) yields the final error decomposition:

$$
\pmb { x } _ { t _ { i } } - \pmb { x } _ { t _ { i } } ^ { \star } = \left( \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \mathrm { e } ^ { - \delta _ { i } } \pmb { I } + \alpha _ { t _ { i } } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } ) \nabla _ { \pmb { x } } \mu ( \pmb { x } _ { t _ { i - 1 } } ^ { \star } , \lambda _ { t _ { i - 1 } } ) ^ { \top } \right) \left( \pmb { x } _ { t _ { i - 1 } } - \pmb { x } _ { t _ { i - 1 } } ^ { \star } \right) + \mathcal { E } _ { t _ { i } , 1 } + \mathcal { E } _ { t _ { i } , 2 } ,\tag{78}
$$

where the residual terms $\mathcal { E } _ { t _ { i } , 1 }$ and $\mathcal { E } _ { t _ { i } , 2 }$ are defined as

$$
\mathcal { E } _ { t , 1 } : = O ( \delta _ { i } ^ { 2 } ) + \int _ { \lambda _ { t - 1 } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i } } - \lambda ) } O ( \| x _ { t _ { i - 1 } } ^ { \star } - x _ { t ( \lambda ) } ^ { \star } \| _ { 2 } ^ { 2 } ) \mathrm { d } \lambda - \nabla _ { x } \mu ( x _ { t _ { i - 1 } } ^ { \star } , \lambda _ { t _ { i - 1 } } ) ^ { \top } x _ { t _ { i - 1 } } ^ { \star } O ( \delta _ { i } ^ { 2 } ) + O ( \delta _ { i } \| x _ { t _ { i - 1 } } - x _ { t _ { i - 1 } } ^ { \star } \| _ { 2 } ^ { 2 } ) .\tag{79a}
$$

$$
\mathcal { E } _ { t _ { i } , 2 } : = - 2 \sqrt { 2 } \alpha _ { t _ { i } } \nabla \mu ( { \pmb x } _ { t _ { i - 1 } } ^ { \star } , \lambda _ { t _ { i - 1 } } ) ^ { \top } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i } } - \lambda ) } \sigma _ { t ( \lambda ) } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda } \mathrm { e } ^ { - ( \lambda - \lambda ^ { \prime } ) } \mathrm { d } \overline { \omega } _ { t ( \lambda ^ { \prime } ) } \mathrm { d } \lambda .\tag{79b}
$$

Here $\mathcal { E } _ { t _ { i } , 1 }$ collects the approximation errors, while $\mathcal { E } _ { t _ { i } , 2 }$ captures the stochastic integral contribution.

Step 2: recursion for the expectation $\mathbb { E } [ \| \pmb { x } _ { t _ { i } } - \pmb { x } _ { t _ { i } } ^ { \star } \| ]$ . For notational convenience, we introduce the matrices $\pmb { A } _ { t _ { i } }$ , which depend on $\boldsymbol { x } _ { t _ { i - 1 } } ^ { \star }$ , as

$$
\pmb { A } _ { t _ { i } } : = \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \mathrm { e } ^ { - \delta _ { i } } \pmb { I } _ { d } + \alpha _ { t _ { i } } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } ) \nabla _ { \pmb { x } } \mu ( \pmb { x } _ { t _ { i - 1 } } ^ { \star } , \lambda _ { t _ { i - 1 } } ) ^ { \top }\tag{80}
$$

This definition allows us to rewrite (78) compactly. By recursion, we have

$$
\pmb { x } _ { t _ { M } } - \pmb { x } _ { t _ { M } } ^ { \star } = \pmb { A } _ { t _ { M } } ( \pmb { x } _ { t _ { M - 1 } } ^ { \star } ) \big ( \pmb { x } _ { t _ { M - 1 } } - \pmb { x } _ { t _ { M - 1 } } ^ { \star } \big ) + \pmb { \xi } _ { t _ { M } , 1 } + \pmb { \xi } _ { t _ { M } , 2 } = \sum _ { m = 1 } ^ { M } \prod _ { j = m + 1 } ^ { M } \pmb { A } _ { t _ { j } } \bigg ( \pmb { \xi } _ { t _ { m } , 1 } + \pmb { \xi } _ { t _ { m } , 2 } \bigg ) ,\tag{81}
$$

where we adopt the convention that the empty product equals the identity matrix. Taking expectations and applying the triangle inequality gives the recursive bound

$$
\mathbb { E } [ \| \mathbf { x } _ { t _ { M } } - \mathbf { x } _ { t _ { M } } ^ { \star } \| _ { 2 } ] \le \sum _ { m = 1 } ^ { M } \mathbb { E } \bigg [ \prod _ { j = m + 1 } ^ { M } \| A _ { t _ { j } } \| _ { \infty } \| { \mathcal { E } } _ { t _ { m } , 1 } \| _ { 2 } \bigg ] + \mathbb { E } \bigg [ \Big \| \sum _ { m = 1 } ^ { M } \prod _ { j = m + 1 } ^ { M } A _ { t _ { j } } { \mathcal { E } } _ { t _ { m } , 2 } \Big \| _ { 2 } \bigg ]\tag{82}
$$

To control the product of operator norms, we first estimate a single factor. For each $i ,$ we have

$$
\| A _ { t _ { i } } ( x _ { t _ { i - 1 } } ^ { \star } ) \| _ { \infty } \leq \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \mathrm { e } ^ { - \delta _ { i } } + \alpha _ { t _ { i } } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } ) \frac { L _ { x } } { \alpha _ { t _ { i - 1 } } } \stackrel { \mathrm { ( a ) } } { \leq } 1 + L _ { x } \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \mathrm { e } ^ { - \delta _ { i } } ( \mathrm { e } ^ { 2 \delta _ { i } } - 1 ) \stackrel { \mathrm { ( b ) } } { \leq } 1 + 4 L _ { x } \delta _ { i } ,\tag{83}
$$

where the second inequality (a) follows from $\sigma _ { t _ { i } } \leq \sigma _ { t _ { i - 1 } }$ and the last inequality (b) uses $\mathrm { e } ^ { - \delta _ { i } } ( \mathrm { e } ^ { 2 \delta _ { i } } - 1 ) \leq 4 \delta _ { i }$ for $\delta _ { i } \leq 1 / 2$ . Consequently, the cumulative product satisfies

$$
\begin{array} { r l r } {  { \Big \| \prod _ { j = m + 1 } ^ { M } \pmb { A } _ { t _ { j } } \Big \| _ { \mathsf { o p } } \le \prod _ { j = m + 1 } ^ { M } \| \pmb { A } _ { t _ { j } } \| _ { \mathsf { o p } } \le \prod _ { j = m + 1 } ^ { M } ( 1 + 4 L _ { x } \delta _ { j } ) } } \\ & { } & { \le \exp \big ( 4 L _ { x } \sum _ { j = m + 1 } ^ { M } \delta _ { j } \big ) = O ( 1 ) , \qquad } \end{array}\tag{84}
$$

where the last equality uses the fact that $\begin{array} { r } { \sum _ { j = m + 1 } ^ { M } \delta _ { j } \leq \sum _ { j = 1 } ^ { M } \delta _ { j } = O ( 1 ) } \end{array}$ . Substituting this bound into (82) yields

$$
\mathbb { E } [ \| \pmb { x } _ { t _ { M } } - \pmb { x } _ { t _ { M } } ^ { \star } \| _ { 2 } ] \lesssim \sum _ { m = 1 } ^ { M } \mathbb { E } [ \| \mathcal { E } _ { t _ { m } , 1 } \| _ { 2 } ] + \mathbb { E } \bigg [ \Big \| \sum _ { m = 1 } ^ { M } \prod _ { j = m + 1 } ^ { M } \pmb { A } _ { t _ { j } } \mathcal { E } _ { t _ { m } , 2 } \Big \| _ { 2 } \bigg ] .\tag{85}
$$

In the following two steps, we will bound these two contributions separately.

Step 3: Analysis of $\mathcal { E } _ { t _ { m } , 2 }$ . From its definition in (79b), the stochastic term $\mathcal { E } _ { t _ { m } , 2 }$ can be rewritten as

$$
\begin{array} { r l } & { \mathcal { E } _ { t _ { m } , 2 } = - 2 \sqrt { 2 } \alpha _ { t _ { m } } \mathrm { e } ^ { - 2 \lambda _ { t _ { m } } } \nabla \mu ( x _ { t _ { m - 1 } } ^ { \star } , \lambda _ { t _ { m - 1 } } ) ^ { \top } \displaystyle \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda _ { t _ { m } } } \int _ { \lambda ^ { \prime } } ^ { \lambda _ { t _ { m } } } \mathrm { e } ^ { \lambda _ { \sigma _ { t ( \lambda ) } } \mathrm { e } ^ { \lambda ^ { \prime } } \mathrm { d } \lambda \mathrm { d } \overline { { \omega } } _ { t ( \lambda ^ { \prime } ) } } } \\ & { \quad \quad \quad = - 2 \sqrt { 2 } \alpha _ { t _ { m } } \mathrm { e } ^ { - 2 \lambda _ { t _ { m } } } \nabla \mu ( x _ { t _ { m - 1 } } ^ { \star } , \lambda _ { t _ { m - 1 } } ) ^ { \top } \displaystyle \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda _ { t _ { m } } } \mathrm { e } ^ { \lambda ^ { \prime } } \left[ \displaystyle \int _ { \lambda ^ { \prime } } ^ { \lambda _ { t _ { m } } } \alpha _ { t ( \lambda ) } \mathrm { d } \lambda \right] \mathrm { d } \overline { { \omega } } _ { t ( \lambda ^ { \prime } ) } } \\ & { \quad \quad = : \nabla \mu ( x _ { t _ { m - 1 } } ^ { \star } , \lambda _ { t _ { m - 1 } } ) ^ { \top } \overline { { z } } _ { t _ { m } } , } \end{array}\tag{86}
$$

where the noise vector $\overline { { z } } _ { t _ { m } }$ depends on the Brownian motion increment over the interval $\left[ \lambda _ { t _ { m - 1 } } , \lambda _ { t _ { m } } \right]$ , and is defined as

$$
\overline { { z } } _ { t _ { m } } : = - 2 \sqrt { 2 } \alpha _ { t _ { m } } \mathrm { e } ^ { - 2 \lambda _ { t _ { m } } } \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda _ { t _ { m } } } \mathrm { e } ^ { \lambda ^ { \prime } } \left[ \int _ { \lambda ^ { \prime } } ^ { \lambda _ { t _ { m } } } \alpha _ { t ( \lambda ) } \mathrm { d } \lambda \right] \mathrm { d } \overline { { \omega } } _ { t ( \lambda ^ { \prime } ) } .\tag{87}
$$

Define the cumulative propagation matrices by

$$
P _ { 0 } = I _ { d } , \qquad P _ { i } = A _ { t _ { i } } P _ { i - 1 } = \prod _ { j = 1 } ^ { i } A _ { t _ { j } } , \quad i \geq 1 .\tag{88}
$$

Note that

$$
\mathrm { e } ^ { - 2 \delta _ { i } } \le \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \mathrm { e } ^ { - \delta _ { i } } \le 1 , \qquad \| A _ { t _ { i } } - I _ { d } \| _ { \mathsf { o p } } \le ( 2 + 4 L _ { x } ) \delta _ { i } .
$$

Here the first inequality uses $\alpha _ { t _ { i } } \geq \alpha _ { t _ { i - 1 } } .$ , and the second follows from the same Jacobian bound as in (83). Writing $C _ { A } = 2 + 4 L _ { x }$ , for suficiently small δ we have

$$
\| A _ { t _ { i } } ^ { - 1 } \| _ { \mathsf { o p } } \leq ( 1 - C _ { A } \delta _ { i } ) ^ { - 1 } \leq \exp ( 2 C _ { A } \delta _ { i } ) .
$$

Together with (84) and $\textstyle \sum _ { i } \delta _ { i } = O ( 1 )$ , this gives a deterministic constant C, independent of $M .$ such that

$$
\operatorname* { m a x } _ { 0 \leq i \leq M } \bigl ( \| P _ { i } \| _ { \mathsf { o p } } + \| P _ { i } ^ { - 1 } \| _ { \mathsf { o p } } \bigr ) \leq C .
$$

Then the factor in the right-hand-side of (85) becomes

$$
\sum _ { m = 1 } ^ { M } \prod _ { j = m + 1 } ^ { M } A _ { t _ { j } } \mathcal { E } _ { t _ { m } , 2 } = \sum _ { m = 1 } ^ { M } A _ { t _ { M } } \cdot \cdot \cdot A _ { t _ { m + 1 } } \mathcal { E } _ { t _ { m } , 2 } = P _ { M } \sum _ { m = 1 } ^ { M } P _ { m } ^ { - 1 } \mathcal { E } _ { t _ { m } , 2 } ,\tag{89}
$$

Substituting (86) into (89) yields

$$
\sum _ { m = 1 } ^ { M } \prod _ { j = m + 1 } ^ { M } A _ { t _ { j } } \mathcal { E } _ { t _ { m } , 2 } = P _ { M } \sum _ { m = 1 } ^ { M } P _ { m } ^ { - 1 } \nabla \pmb { \mu } ( \pmb { x } _ { t _ { m - 1 } } ^ { \star } , \lambda _ { t _ { m - 1 } } ) ^ { \top } \pmb { \overline { { z } } } _ { t _ { m } } .\tag{90}
$$

To analyze this sum, define the filtration generated by the initial state and the Brownian increments up to the current log-SNR:

$$
\mathcal { F } _ { t _ { i } } = \sigma \Big ( \pmb { x } _ { t _ { 0 } } ^ { \star } , \overline { { \omega } } _ { t ( \lambda ) } - \overline { { \omega } } _ { t _ { 0 } } : \lambda _ { t _ { 0 } } \leq \lambda \leq \lambda _ { t _ { i } } \Big ) .\tag{91}
$$

The vector $\overline { { z } } _ { t _ { m } }$ is independent of $\mathcal { F } _ { t _ { m - 1 } }$ and has zero mean. Its conditional second moment satisfies

$$
\begin{array} { r l } & { \mathbb { E } [ \| \overline { { z } } _ { t _ { m } } \| _ { 2 } ^ { 2 } \mid \mathcal { F } _ { t _ { m - 1 } } ] \leq 8 \alpha _ { t _ { m } } d \mathrm { e } ^ { - 4 \lambda _ { t _ { m } } } \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda _ { t _ { m } } } \mathrm { e } ^ { 2 \lambda ^ { \prime } } \left[ \int _ { \lambda ^ { \prime } } ^ { \lambda _ { t _ { m } } } \alpha _ { t ( \lambda ) } \mathrm { d } \lambda \right] ^ { 2 } \mathrm { d } \lambda ^ { \prime } } \\ & { \qquad \leq 8 \alpha _ { t _ { m } } d \mathrm { e } ^ { - 4 \lambda _ { t _ { m } } } \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda _ { t _ { m } } } \mathrm { e } ^ { 2 \lambda ^ { \prime } } \big ( \lambda _ { t _ { m } } - \lambda ^ { \prime } \big ) \int _ { \lambda ^ { \prime } } ^ { \lambda _ { t _ { m } } } \alpha _ { t ( \lambda ) } ^ { 2 } \mathrm { d } \lambda \mathrm { d } \lambda ^ { \prime } } \\ & { \qquad \leq 8 \alpha _ { t _ { m } } d \mathrm { e } ^ { - 4 \lambda _ { t _ { m } } } \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda _ { t _ { m } } } \mathrm { e } ^ { 2 \lambda ^ { \prime } } \delta _ { m } ^ { 2 } \alpha _ { t _ { m } } ^ { 2 } \mathrm { d } \lambda ^ { \prime } } \\ & { \qquad \leq 8 d \delta _ { m } ^ { 2 } \alpha _ { t _ { m } } ^ { 3 } \mathrm { e } ^ { - 4 \lambda _ { t _ { m } } } \left( \mathrm { e } ^ { 2 \lambda _ { t _ { m } } } - \mathrm { e } ^ { 2 \lambda _ { t _ { m } - 1 } } \right) \leq 1 6 d \delta _ { m } ^ { 3 } \alpha _ { t _ { m } } ^ { 3 } \mathrm { e } ^ { - 2 \lambda _ { t _ { m } } } = O ( \delta _ { m } ^ { 3 } ) , } \end{array}\tag{92}
$$

where the second inequality follows from Jensen’s inequality, and the third uses the monotonicity $\alpha _ { t ( \lambda ) } \leq \alpha _ { t _ { m } } .$ Since $P _ { m } ^ { - 1 }$ and $\nabla \mu ( \pmb { x } _ { t _ { m - 1 } } ^ { \star } , \lambda _ { t _ { m - 1 } } )$ are $\mathcal { F } _ { t _ { m } } .$ -measurable, and (92), the expected norm squares of the vector $\begin{array} { r } { \sum _ { m = 1 } ^ { M } P _ { m } ^ { - 1 } \nabla \pmb { \mu } ( \pmb { x } _ { t _ { m - 1 } } ^ { \star } , \lambda _ { t _ { m - 1 } } ) ^ { \top } \mp \pmb { \mu } _ { m } } \end{array}$ can be computed by:

$$
\begin{array} { r l } {  { \mathbb { E } \Big [ \Big \| \sum _ { m = 1 } ^ { M } P _ { m } ^ { - 1 } \nabla \mu ( \pmb { x } _ { t _ { m - 1 } } ^ { \star } , \lambda _ { t _ { m - 1 } } ) ^ { \top } \overline { { \pmb { z } } } _ { t _ { m } } \Big \| _ { 2 } ^ { 2 } \Big ] } \qquad } & { } \\ & { = \sum _ { m = 1 } ^ { M } \mathbb { E } \big [ \| P _ { m } ^ { - 1 } \nabla \mu ( \pmb { x } _ { t _ { m - 1 } } ^ { \star } , \lambda _ { t _ { m - 1 } } ) ^ { \top } \overline { { \pmb { z } } } _ { t _ { m } } \| _ { 2 } ^ { 2 } \big ] } \end{array}
$$

$$
\begin{array} { r l } & { \quad + 2 \displaystyle \sum _ { n > m } \mathbb { E } \big [ \langle P _ { n } ^ { - 1 } \nabla \mu ( x _ { t _ { n - 1 } } ^ { \star } , \lambda _ { t _ { n - 1 } } ) ^ { \top } \Xi _ { t _ { n } } , P _ { m } ^ { - 1 } \nabla \mu ( x _ { t _ { m - 1 } } ^ { \star } , \lambda _ { t _ { m - 1 } } ) ^ { \top } \Xi _ { t _ { m } } \rangle \big ] } \\ & { \quad \le C \displaystyle \sum _ { m = 1 } ^ { M } \delta _ { m } ^ { 3 } = O ( \delta ^ { 2 } ) , } \end{array}\tag{93}
$$

where the last inequality uses (92) and

$$
\begin{array} { r l } & { \mathbb { E } \bigl [ \| P _ { m } ^ { - 1 } \nabla \mu ( x _ { t _ { m - 1 } } ^ { \star } , \lambda _ { t _ { m - 1 } } ) ^ { \top } \overline { { z } } _ { t _ { m } } \| _ { 2 } ^ { 2 } \bigr ] = \mathbb { E } \bigl [ \mathbb { E } \bigl [ \| P _ { m } ^ { - 1 } \nabla \mu ( x _ { t _ { m - 1 } } ^ { \star } , \lambda _ { t _ { m - 1 } } ) ^ { \top } \overline { { z } } _ { t _ { m } } \| _ { 2 } ^ { 2 } \big | \mathcal { F } _ { t _ { m - 1 } } \bigr ] \bigr ] } \\ & { \qquad \leq C \mathbb { E } \bigl [ \mathbb { E } [ \| \overline { { z } } _ { t _ { m } } \| _ { 2 } ^ { 2 } \big | \mathcal { F } _ { t _ { m - 1 } } ] \bigr ] = O ( \delta _ { m } ^ { 3 } ) , } \end{array}
$$

and the fact that the cross terms vanish due to the independence of $\overline { { z } } _ { t _ { n } }$ and $\mathcal { F } _ { t _ { n - 1 } }$ . Finally, combining (89) and (90) with Jensen’s inequality yields the required bound for the second term in (85):

$$
\mathbb { E } \bigg \| \sum _ { m = 1 } ^ { M } \prod _ { j = m + 1 } ^ { M } A _ { t _ { j } } \mathscr { E } _ { t _ { m } , 2 } \bigg \| _ { 2 } \lesssim \Big ( \mathbb { E } \bigg [ \bigg \| \sum _ { m = 1 } ^ { M } P _ { m } ^ { - 1 } \nabla \mu ( x _ { t _ { m - 1 } } ^ { \star } , \lambda _ { t _ { m - 1 } } ) ^ { \top } \overline { { z } } _ { t _ { m } } \bigg \| _ { 2 } ^ { 2 } \bigg ] \Big ) ^ { 1 / 2 } = O ( \delta ) .\tag{94}
$$

For later use, we also estimate $\mathbb { E } [ \| \pmb { x } _ { t _ { m - 1 } } ^ { \star } - \pmb { x } _ { t ( \lambda ) } ^ { \star } \| _ { 2 } ^ { 2 } ]$ . From (29), we have

$$
\begin{array} { r l } & { x _ { t ( \lambda ) } ^ { \star } - x _ { t _ { m - 1 } } ^ { \star } = \Bigl ( \frac { \sigma _ { t ( \lambda ) } } { \sigma _ { t _ { m - 1 } } } \mathrm { e } ^ { - ( \lambda - \lambda _ { t _ { m - 1 } } ) } - 1 \Bigr ) x _ { t _ { m - 1 } } ^ { \star } } \\ & { \qquad + 2 \alpha _ { t ( \lambda ) } \displaystyle \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda } \mathrm { e } ^ { - 2 ( \lambda - \lambda ^ { \prime } ) } \mu ( x _ { t ( \lambda ^ { \prime } ) } ^ { \star } , \lambda ^ { \prime } ) \mathrm { d } \lambda ^ { \prime } + \sqrt { 2 } \sigma _ { t ( \lambda ) } \displaystyle \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda } \mathrm { e } ^ { - ( \lambda - \lambda ^ { \prime } ) } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ^ { \prime } ) } . } \end{array}\tag{95}
$$

Using the coeficient bound in (75) and Assumption A1, we obtain

$$
\begin{array} { r l r } & { } & { \mathbb { E } [ \| { \pmb x } _ { t ( \lambda ) } ^ { \star } - { \pmb x } _ { t _ { m - 1 } } ^ { \star } \| _ { 2 } ^ { 2 } ] \lesssim \delta _ { m } ^ { 2 } \mathbb { E } [ \| { \pmb x } _ { t _ { m - 1 } } ^ { \star } \| _ { 2 } ^ { 2 } ] + \alpha _ { t ( \lambda ) } ^ { 2 } L _ { \mu } ^ { 2 } \Big ( \displaystyle \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda } \mathrm { e } ^ { - 2 ( \lambda - \lambda ^ { \prime } ) } \mathrm { d } \lambda ^ { \prime } \Big ) ^ { 2 } } \\ & { } & { + \mathbb { E } \Big [ \Big \| \sqrt { 2 } \sigma _ { t ( \lambda ) } \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda } \mathrm { e } ^ { - ( \lambda - \lambda ^ { \prime } ) } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ^ { \prime } ) } \Big \| _ { 2 } ^ { 2 } \Big ] . \qquad } \end{array}\tag{96}
$$

The three terms are bounded as follows:

$$
\begin{array} { r } { \displaystyle \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda } \mathrm { e } ^ { - 2 ( \lambda - \lambda ^ { \prime } ) } \mathrm { d } \lambda ^ { \prime } = \frac 1 2 \big ( 1 - \mathrm { e } ^ { - 2 ( \lambda - \lambda _ { t _ { m - 1 } } ) } \big ) \le \lambda - \lambda _ { t _ { m - 1 } } \le \delta _ { m } , } \\ { \displaystyle \mathbb E \Big [ \Big \| \sqrt { 2 } \sigma _ { t ( \lambda ) } \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda } \mathrm { e } ^ { - ( \lambda - \lambda ^ { \prime } ) } \mathrm { d } \overline { \omega } _ { t ( \lambda ^ { \prime } ) } \Big \| _ { 2 } ^ { 2 } \Big ] = 2 d \sigma _ { t ( \lambda ) } ^ { 2 } \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda } \mathrm { e } ^ { - 2 ( \lambda - \lambda ^ { \prime } ) } \mathrm { d } \lambda ^ { \prime } \le 2 d \sigma _ { t ( \lambda ) } ^ { 2 } \delta _ { m } . } \end{array}
$$

Together with the moment bound $\mathbb { E } [ \| \pmb { x } _ { t _ { m - 1 } } ^ { \star } \| _ { 2 } ^ { 2 } ] = O ( 1 )$ from (71), we conclude

$$
\mathbb { E } [ \| \pmb { x } _ { t ( \lambda ) } ^ { \star } - \pmb { x } _ { t _ { m - 1 } } ^ { \star } \| _ { 2 } ^ { 2 } ] \lesssim \delta _ { m } ^ { 2 } + \alpha _ { t ( \lambda ) } ^ { 2 } L _ { \mu } ^ { 2 } \delta _ { m } + 2 d \sigma _ { t ( \lambda ) } ^ { 2 } \delta _ { m } = O ( \delta _ { m } ) .\tag{97}
$$

Step 4: Analysis of $\mathcal { E } _ { t _ { m } , 1 } .$ . We now estimate the first term on the right-hand-side of (85). From the definition of $\| \mathcal { E } _ { t _ { m } , 1 } \| _ { 2 }$ in (79a), we have

$$
\begin{array} { r l } & { \mathbb { E } [ \| \mathcal { E } _ { t _ { m } , 1 } \| _ { 2 } ] \lesssim \delta _ { m } ^ { 2 } + \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda _ { t _ { m } } } \mathrm { e } ^ { - 2 ( \lambda _ { t _ { m } } - \lambda ) } \mathbb { E } [ \| x _ { t _ { m - 1 } } ^ { \star } - x _ { t ( \lambda ) } ^ { \star } \| _ { 2 } ^ { 2 } ] \mathrm { d } \lambda } \\ & { \quad \quad + \delta _ { m } ^ { 2 } \mathbb { E } [ \| \nabla _ { x } \mu ( x _ { t _ { m - 1 } } ^ { \star } , \lambda _ { t _ { m - 1 } } ) ^ { \top } x _ { t _ { m - 1 } } ^ { \star } \| _ { 2 } ] + \delta _ { m } \mathbb { E } [ \| x _ { t _ { m - 1 } } - x _ { t _ { m - 1 } } ^ { \star } \| _ { 2 } ^ { 2 } ] . } \end{array}\tag{98}
$$

We bound each term on the right-hand side of (98) as follows.

First, from (97), we have $\mathbb { E } [ \| \pmb { x } _ { t _ { m - 1 } } ^ { \star } - \pmb { x } _ { t ( \lambda ) } ^ { \star } \| _ { 2 } ^ { 2 } ] = O ( \delta _ { m } )$ . Consequently,

$$
\int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda _ { t _ { m } } } \mathrm { e } ^ { - 2 ( \lambda _ { t _ { m } } - \lambda ) } \mathbb { E } [ \| \pmb { x } _ { t _ { m - 1 } } ^ { \star } - \pmb { x } _ { t ( \lambda ) } ^ { \star } \| _ { 2 } ^ { 2 } ] \mathrm { d } \lambda = O ( \delta _ { m } ) \frac { 1 } { 2 } ( 1 - \mathrm { e } ^ { - 2 \delta _ { m } } ) = O ( \delta _ { m } ^ { 2 } ) .\tag{99}
$$

Second, by Assumption A2 and the moment bound (71),

$$
\delta _ { m } ^ { 2 } \mathbb { E } [ \| \nabla _ { \pmb { x } } \pmb { \mu } ( \pmb { x } _ { t _ { m - 1 } } ^ { \star } , \lambda _ { t _ { m - 1 } } ) ^ { \top } \pmb { x } _ { t _ { m - 1 } } ^ { \star } \| _ { 2 } ] \leq \frac { L _ { x } \delta _ { m } ^ { 2 } } { \alpha _ { t _ { m - 1 } } } \sqrt { \mathbb { E } [ \| \pmb { x } _ { t _ { m - 1 } } ^ { \star } \| _ { 2 } ^ { 2 } ] } = O ( \delta _ { m } ^ { 2 } ) .\tag{100}
$$

Third, the remaining term $\mathbb { E } [ \| \pmb { x } _ { t _ { m - 1 } } - \pmb { x } _ { t _ { m - 1 } } ^ { \star } \| _ { 2 } ^ { 2 } ]$ is precisely the quantity we aim to control. To proceed, we first derive a crude bound for it. From (72) and the Lipschitz property of data predictor with respect to x, we have

$$
\begin{array} { r l } & { \| \pmb { x } _ { t _ { m } } - \pmb { x } _ { t _ { m } } ^ { \star } \| _ { 2 } \leq \frac { \sigma _ { t _ { m } } } { \sigma _ { t _ { m - 1 } } } \mathrm { e } ^ { - \delta _ { m } } \left\| \pmb { x } _ { t _ { m - 1 } } - \pmb { x } _ { t _ { m - 1 } } ^ { \star } \right\| _ { 2 } + \frac { \alpha _ { t _ { m } } } { \alpha _ { t _ { m - 1 } } } L _ { x } \big ( 1 - \mathrm { e } ^ { - 2 \delta _ { m } } \big ) \big \| \pmb { x } _ { t _ { m - 1 } } - \pmb { x } _ { t _ { m - 1 } } ^ { \star } \big \| _ { 2 } } \\ & { \qquad + \ 2 \alpha _ { t _ { m } } \bigg \| \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda _ { t _ { m } } } \mathrm { e } ^ { - 2 ( \lambda _ { t _ { m } } - \lambda ) } \big ( \mu ( \pmb { x } _ { t _ { m - 1 } } ^ { \star } , \lambda _ { t _ { m - 1 } } ) - \mu ( \pmb { x } _ { t ( \lambda ) } ^ { \star } , \lambda ) \big ) \mathrm { d } \lambda \bigg \| _ { 2 } . } \end{array}\tag{101}
$$

For notational convenience, denote the integral term by

$$
B _ { t _ { m } } : = \alpha _ { t _ { m } } \bigg \| \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda _ { t _ { m } } } \mathrm { e } ^ { - 2 ( \lambda _ { t _ { m } } - \lambda ) } \big ( \mu ( \pmb { x } _ { t _ { m - 1 } } ^ { \star } , \lambda _ { t _ { m - 1 } } ) - \mu ( \pmb { x } _ { t ( \lambda ) } ^ { \star } , \lambda ) \big ) \mathrm { d } \lambda \bigg \| _ { 2 } .\tag{102}
$$

Since $\sigma _ { t _ { m } } \leq \sigma _ { t _ { m - 1 } }$ , we have

$$
\begin{array} { r l } {  { \frac { \sigma _ { t _ { m } } } { \sigma _ { t _ { m - 1 } } } \mathrm { e } ^ { - \delta _ { m } } \bigl \| { \pmb x } _ { t _ { m - 1 } } - { \pmb x } _ { t _ { m - 1 } } ^ { \star } \bigr \| _ { 2 } + \frac { \alpha _ { t _ { m } } } { \alpha _ { t _ { m - 1 } } } L _ { x } ( 1 - \mathrm { e } ^ { - 2 \delta _ { m } } ) \bigl \| { \pmb x } _ { t _ { m - 1 } } - { \pmb x } _ { t _ { m - 1 } } ^ { \star } \bigr \| _ { 2 } } } \\ & { \le \Big ( \frac { \sigma _ { t _ { m } } } { \sigma _ { t _ { m - 1 } } } \mathrm { e } ^ { - \delta _ { m } } + \frac { 2 \alpha _ { t _ { m } } L _ { x } \delta _ { m } } { \alpha _ { t _ { m - 1 } } } \Big ) \bigl \| { \pmb x } _ { t _ { m - 1 } } - { \pmb x } _ { t _ { m - 1 } } ^ { \star } \bigr \| _ { 2 } } \\ & { \le \bigl ( 1 + \frac { 2 \alpha _ { t _ { m } } L _ { x } \delta _ { m } } { \alpha _ { t _ { m - 1 } } } \bigr ) \bigl \| { \pmb x } _ { t _ { m - 1 } } - { \pmb x } _ { t _ { m - 1 } } ^ { \star } \bigr \| _ { 2 } . } \end{array}\tag{103}
$$

Next, we estimate the second moment of $B _ { t _ { m } }$ . Splitting via the triangle inequality gives

$$
\begin{array} { r l } & { \mathbb { E } \big [ B _ { t _ { m - 1 } } ^ { 2 } \big ] \leq 2 \alpha _ { t _ { m } , s } ^ { 2 } \mathbb { E } \Bigg [ \Bigg \lVert \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda _ { t _ { m } } } \mathrm { e } ^ { - 2 ( \lambda _ { t _ { m } } - \lambda ) } \big ( \mu ( x _ { t _ { m - 1 } } ^ { k } , \lambda _ { t _ { m - 1 } } ) - \mu ( x _ { t ( \lambda ) } ^ { k } , \lambda _ { t _ { m - 1 } } ) \big ) \mathrm { d } \lambda \Bigg \rVert _ { 2 } ^ { 2 } \Bigg ] } \\ & { \qquad + 2 \alpha _ { t _ { m } } ^ { 2 } \mathbb { E } \Bigg [ \Bigg \lVert \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda _ { t _ { m } } } \mathrm { e } ^ { - 2 ( \lambda _ { t _ { m } } - \lambda ) } \big ( \mu ( x _ { t ( \lambda ) } ^ { k } , \lambda _ { t _ { m - 1 } } ) - \mu ( x _ { t ( \lambda ) } ^ { k } , \lambda ) \big ) \mathrm { d } \lambda \Bigg \rVert _ { 2 } ^ { 2 } \Bigg ] } \\ & { \stackrel { ( a ) } { \leq } 2 \delta _ { m } \alpha _ { t _ { m } } ^ { 2 } \mathbb { E } \Bigg [ \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda _ { t _ { m } } } \mathrm { e } ^ { - 1 ( \lambda _ { t _ { m } } - \lambda ) } \big \lVert \mu ( x _ { t _ { m - 1 } } ^ { k } , \lambda _ { t _ { m - 1 } } ) - \mu ( x _ { t ( \lambda ) } ^ { k } , \lambda _ { t _ { m - 1 } } ) \big \rVert _ { 2 } ^ { 2 } \mathrm { d } \lambda \Bigg ] } \\ &  \qquad + 2 \delta _ { m } \alpha _ { t _ { m } } ^ { 2 } \mathbb { E } \Bigg [ \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda _ { t _ { m } } } \mathrm { e } ^ { - 4 ( \lambda _ { t _ { m } } - \lambda ) } \big \lVert \end{array}
$$

$$
\stackrel { \mathrm { ( c ) } } { \lesssim } \delta _ { m } ^ { 2 } \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda _ { t _ { m } } } \mathrm { e } ^ { - 4 ( \lambda _ { t _ { m } } - \lambda ) } \mathrm { d } \lambda + \delta _ { m } ^ { 3 } \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda _ { t _ { m } } } \mathrm { e } ^ { - 4 ( \lambda _ { t _ { m } } - \lambda ) } \mathrm { d } \lambda = O ( \delta _ { m } ^ { 3 } ) ,\tag{104}
$$

where (a) uses Jensen’s inequality (with the interval length $\delta _ { m } )$ , (b) follows from Assumptions A2 and A4, and (c) uses (97). Combining (103) and (104) with (101), we obtain

$$
\begin{array} { r l r } {  { \mathbb { E } [ \| x _ { t _ { m } } - x _ { t _ { m } } ^ { * } \| _ { 2 } ^ { 2 } ] \leq \big ( 1 + \frac { 2 \alpha \epsilon _ { m } L _ { \alpha } \delta _ { m } } { \alpha _ { t _ { m - 1 } } } \big ) ^ { 2 } \mathbb { E } [ \| x _ { t _ { m - 1 } } - x _ { t _ { m - 1 } } ^ { * } \| _ { 2 } ^ { 2 } ] + 4 \big ( 1 + \frac { 2 \alpha _ { t _ { m } } L _ { \alpha } \delta _ { m } } { \alpha _ { t _ { m - 1 } } } \big ) \mathbb { E } \big [ B _ { t _ { m } } \big \| x _ { t _ { m - 1 } } - x _ { t _ { m - 1 } } ^ { * } \big \| _ { 2 } \big ] + 4 \mathbb { E } [ B _ { t _ { m } } ^ { 2 } ] } } \\ & { \stackrel { \mathrm { ( a ) } } { \leq } \big ( 1 + \frac { 2 \alpha _ { t _ { m } } L _ { \alpha } \delta _ { m } } { \alpha _ { t _ { m - 1 } } } \big ) ^ { 2 } \mathbb { E } [ \| x _ { t _ { m - 1 } } - x _ { t _ { m - 1 } } ^ { * } \| _ { 2 } ^ { 2 } ] } \\ & { } & { + 2 \big ( 1 + \frac { 2 \alpha _ { t _ { m } } L _ { \alpha } \delta _ { m } } { \alpha _ { t _ { m - 1 } } } \big ) O ( \delta _ { m } ^ { 1 . 5 } ) \sqrt { \mathbb { E } \big [ \| x _ { t _ { m - 1 } } - x _ { t _ { m - 1 } } ^ { * } \| _ { 2 } ^ { 2 } \big ] } + O ( \delta _ { m } ^ { 3 } ) , \qquad ( 1 0 5 ) } \end{array}
$$

where (a) uses (104) and Cauchy-Schwarz

$$
\begin{array} { r } { \mathbb { E } \big [ B _ { t _ { m - 1 } } \big \lVert \mathbf { x } _ { t _ { m - 1 } } - \mathbf { x } _ { t _ { m - 1 } } ^ { \star } \big \rVert _ { 2 } \big ] \leq \sqrt { \mathbb { E } [ B _ { t _ { m } } ^ { 2 } ] \mathbb { E } \big [ \big \lVert \mathbf { x } _ { t _ { m - 1 } } - \mathbf { x } _ { t _ { m - 1 } } ^ { \star } \big \rVert _ { 2 } ^ { 2 } \big ] } = O ( \delta _ { m } ^ { 1 . 5 } ) \sqrt { \mathbb { E } \big [ \big \lVert \mathbf { x } _ { t _ { m - 1 } } - \mathbf { x } _ { t _ { m - 1 } } ^ { \star } \big \rVert _ { 2 } ^ { 2 } \big ] } . } \end{array}
$$

For suficiently small $\delta _ { m } .$ , we have $\frac { 2 \alpha _ { t _ { m } } L _ { x } \delta _ { m } } { \alpha _ { t _ { m - 1 } } } \leq 1$ and

$$
\delta _ { m } ^ { 1 . 5 } \sqrt { \mathbb { E } \big [ \big \| \pmb { x } _ { t _ { m - 1 } } - \pmb { x } _ { t _ { m - 1 } } ^ { \star } \big \| _ { 2 } ^ { 2 } \big ] } \leq \frac { 1 } { 2 } \delta _ { m } ^ { 2 } + \frac { 1 } { 2 } \delta _ { m } \mathbb { E } \big [ \big \| \pmb { x } _ { t _ { m - 1 } } - \pmb { x } _ { t _ { m - 1 } } ^ { \star } \big \| _ { 2 } ^ { 2 } \big ] .
$$

Therefore we have

$$
\begin{array} { r l } { \mathbb { E } [ \| x _ { t _ { m } } - x _ { t _ { m } - 1 } ^ { * } \| _ { 2 } ^ { 2 } ] \leq \big ( 1 + \frac { 5 \alpha _ { t _ { m } } L _ { x } \delta _ { m } } { \alpha _ { t _ { m } - 1 } } + O ( \delta _ { m } ) \big ) \mathbb { E } \big [ \big \| x _ { t _ { m - 1 } } - x _ { t _ { m - 1 } } ^ { * } \big \| _ { 2 } ^ { 2 } \big ] + O ( \delta _ { m } ^ { 2 } ) } & { } \\ { \lesssim \displaystyle \sum _ { i = 1 } ^ { m } \displaystyle \prod _ { j = i + 1 } ^ { m } \big ( 1 + \frac { 5 \alpha _ { t _ { j } } L _ { x } \delta _ { j } } { \alpha _ { t _ { j - 1 } } } + O ( \delta _ { j } ) \big ) \delta _ { i } ^ { 2 } } & { } \\ { \lesssim \displaystyle \sum _ { i = 1 } ^ { m } \exp \Big ( \displaystyle \sum _ { j = i + 1 } ^ { m } \big ( \frac { 5 \alpha _ { t _ { j } } L _ { x } \delta _ { j } } { \alpha _ { t _ { j - 1 } } } + O ( \delta _ { j } ) \big ) \Big ) \delta _ { i } ^ { 2 } } & { } \\ { \overset { ( \mathrm { a } ) } { \lesssim } \displaystyle \sum _ { i = 1 } ^ { m } \delta _ { i } ^ { 2 } = O ( \delta ) , } & { } \end{array}\tag{106}
$$

where (a) uses $\begin{array} { r } { \sum _ { j = i + 1 } ^ { m } \delta _ { j } \le \sum _ { j = 1 } ^ { M } \delta _ { j } = O ( 1 ) } \end{array}$ . Substituting (99), (100), and (106) into (98) yields the desired bound:

$$
\mathbb { E } [ \| \mathcal { E } _ { t _ { m } , 1 } \| _ { 2 } ] = O ( \delta ^ { 2 } ) .\tag{107}
$$

Hence the first term on the right-hand-side of (85) satisfies

$$
\sum _ { m = 1 } ^ { M } \mathbb { E } [ \| \mathcal { E } _ { t _ { m } , 1 } \| _ { 2 } ] \lesssim M \delta ^ { 2 } = O ( \delta ) .\tag{108}
$$

Finally, combining (94) and (108) into (85) yields

$$
\mathbb { E } [ \| \pmb { x } _ { t _ { M } } - \pmb { x } _ { t _ { M } } ^ { \star } \| _ { 2 } ] = O ( \delta ) .\tag{109}
$$

This completes the proof.

Proof of (71). From the explicit solution of the SDE in (16), we have

$$
\pmb { x } _ { t _ { i } } ^ { \star } = \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { 0 } } } \mathrm { e } ^ { - ( \lambda _ { t _ { i } } - \lambda _ { t _ { 0 } } ) } \pmb { x } _ { t _ { 0 } } + 2 \alpha _ { t _ { i } } \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i } } - \lambda ) } \pmb { \mu } ( \pmb { x } _ { t ( \lambda ) } ^ { \star } , \lambda ) \mathrm { d } \lambda + \sqrt { 2 } \sigma _ { t _ { i } } \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - ( \lambda _ { t _ { i } } - \lambda ) } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ) } .\tag{110}
$$

Under Assumption A1, the data predictor term is uniformly bounded by $L _ { \mu } .$ . Applying the triangle inequality yields

$$
\| x _ { t _ { i } } ^ { \star } \| _ { 2 } \leq \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { 0 } } } \mathrm { e } ^ { - ( \lambda _ { t _ { i } } - \lambda _ { t _ { 0 } } ) } \| x _ { t _ { 0 } } \| _ { 2 } + 2 \alpha _ { t _ { i } } L _ { \mu } \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i } } - \lambda ) } \mathrm { d } \lambda + \sqrt { 2 } \sigma _ { t _ { i } } \Big \| \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - ( \lambda _ { t _ { i } } - \lambda ) } \mathrm { d } \overline { \omega } _ { t ( \lambda ) } \Big \| _ { 2 } .\tag{111}
$$

Taking expectations and using the elementary inequality $( a + b + c ) ^ { 2 } \leq 3 a ^ { 2 } + 3 b ^ { 2 } + 3 c ^ { 2 }$ , we obtain

$$
\mathbb { E } [ \| x _ { t _ { i } } ^ { \star } \| _ { 2 } ^ { 2 } ] \le 3 \frac { \sigma _ { t _ { i } } ^ { 2 } } { \sigma _ { t _ { 0 } } ^ { 2 } } \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i } } - \lambda _ { t _ { 0 } } ) } \mathbb { E } [ \| x _ { t _ { 0 } } \| _ { 2 } ^ { 2 } ] + 1 2 \alpha _ { t _ { i } } ^ { 2 } L _ { \mu } ^ { 2 } + 6 \sigma _ { t _ { i } } ^ { 2 } \mathbb { E } \Big [ \Big \| \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - ( \lambda _ { t _ { i } } - \lambda ) } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ) } \Big \| _ { 2 } ^ { 2 } \Big ] = O ( 1 ) ,\tag{112}
$$

where the last equation uses $\mathbb { E } [ \| \pmb { x } _ { t _ { 0 } } \| _ { 2 } ^ { 2 } ] = d \sigma _ { t _ { 0 } } ^ { 2 } = O ( 1 )$ , and

$$
\mathbb { E } \Big [ \Big \lVert \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - ( \lambda _ { t _ { i } } - \lambda ) } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ) } \Big \rVert _ { 2 } ^ { 2 } \Big ] = d \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i } } - \lambda ) } \mathrm { d } \lambda \leq \frac { d } { 2 } .\tag{113}
$$

This completes the proof of the preliminary bound.

## C.2 Convergence analysis of SDE-DPMSolver++2

For notational convenience, we denote by $D _ { t _ { i } }$ the diference term $D _ { 1 }$ that appears when updating ${ \bf x } _ { t _ { i - 1 } }$ to $\mathbf { x } _ { t _ { i } }$ . Similar to the analysis in Section C.1, we first establish the following preliminary bound, whose proof is deferred to the end of this section:

$$
\begin{array} { r } { \mathbb { E } [ \| \pmb { x } _ { t _ { i } } \| _ { 2 } ^ { 2 } ] = O ( 1 ) , \qquad 0 \le i \le M . } \end{array}\tag{114}
$$

Step 1: some crude bounds. Before proceeding with the main proof, we derive some crude estimates for $D _ { t _ { i } }$ and ${ \pmb x } _ { t _ { i - 1 } } - { \pmb x } _ { t _ { i - 2 } }$ . We begin with $D _ { t _ { i } }$ . By its definition in (17b),

$$
\begin{array} { r l } & { \| D _ { t _ { i } } \| _ { 2 } = \displaystyle \frac { 1 } { \delta _ { i - 1 } } \| \mu _ { \theta } ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) - \mu _ { \theta } ( x _ { t _ { i - 2 } } , t _ { i - 2 } ) \| _ { 2 } } \\ & { \quad \quad \le \displaystyle \frac { 1 } { \delta _ { i - 1 } } \| \mu _ { \theta } ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) - \mu _ { \theta } ( x _ { t _ { i - 2 } } , t _ { i - 1 } ) \| _ { 2 } + \displaystyle \frac { 1 } { \delta _ { i - 1 } } \| \mu _ { \theta } ( x _ { t _ { i - 2 } } , t _ { i - 1 } ) - \mu _ { \theta } ( x _ { t _ { i - 2 } } , t _ { i - 2 } ) \| _ { 2 } } \\ & { \quad \quad \le \displaystyle \frac { L _ { x } } { \delta _ { i - 1 } \alpha _ { t _ { i - 1 } } } \| x _ { t _ { i - 1 } } - x _ { t _ { i - 2 } } \| _ { 2 } + L _ { t } , } \end{array}\tag{115}
$$

where the last inequality follows from Assumptions 1 (A2) and (A4). Next, from the update rule (17b), we have

$$
\begin{array} { r } { x _ { t _ { i - 1 } } - x _ { t _ { i - 2 } } = \bigg ( \mathrm { e } ^ { - \delta _ { i - 1 } } \frac { \sigma _ { t _ { i - 1 } } } { \sigma _ { t _ { i - 2 } } } - 1 \bigg ) x _ { t _ { i - 2 } } + \alpha _ { t _ { i - 1 } } \big ( 1 - \mathrm { e } ^ { - 2 \delta _ { i - 1 } } \big ) \mu _ { \theta } \big ( x _ { t _ { i - 2 } } , t _ { i - 2 } \big ) } \\ { + \left. \frac { \alpha _ { t _ { i - 1 } } } { 2 } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i - 1 } } ) \delta _ { i - 1 } D _ { t _ { i - 1 } } + \sigma _ { t _ { i - 1 } } \sqrt { 1 - \mathrm { e } ^ { - 2 \delta _ { i - 1 } } } z _ { i - 1 } , \right. } \end{array}\tag{116}
$$

Using bound in (75), we obtain

$$
\begin{array} { r l r } {  { \| { \boldsymbol { x } } _ { t _ { i - 1 } } - { \boldsymbol { x } } _ { t _ { i - 2 } } \| _ { 2 } \leq 2 \delta _ { i - 1 } \| { \boldsymbol { x } } _ { t _ { i - 2 } } \| _ { 2 } + 2 \alpha _ { t _ { i - 1 } } \delta _ { i - 1 } \| \mu _ { \boldsymbol { \theta } } ( { \boldsymbol { x } } _ { t _ { i - 2 } } , t _ { i - 2 } ) \| _ { 2 } + \alpha _ { t _ { i - 1 } } \delta _ { i - 1 } ^ { 2 } \| { \boldsymbol { D } } _ { t _ { i - 1 } } \| _ { 2 } + \sigma _ { t _ { i - 1 } } \sqrt { 2 \delta _ { i - 1 } } \| { \boldsymbol { z } } _ { i - 1 } \| _ { 2 } } } \\ & { } & { \leq 2 \delta _ { i - 1 } \| { \boldsymbol { x } } _ { t _ { i - 2 } } \| _ { 2 } + 2 \alpha _ { t _ { i - 1 } } \delta _ { i - 1 } L _ { \mu } + \alpha _ { t _ { i - 1 } } \delta _ { i - 1 } ^ { 2 } \| { \boldsymbol { D } } _ { t _ { i - 1 } } \| _ { 2 } + \sigma _ { t _ { i - 1 } } \sqrt { 2 \delta _ { i - 1 } } \| { \boldsymbol { z } } _ { i - 1 } \| _ { 2 } , \qquad ( 1 1 7 ) } \end{array}
$$

where the last inequality uses Assumption 1 (A1). Substituting (117) into (115) yields the following recursion for $\| D _ { t _ { i } } \| _ { 2 } \colon$

$$
\begin{array} { r l r } {  { \| { \cal D } _ { t _ { i } } \| _ { 2 } \leq \frac { 2 L _ { x } } { \alpha _ { t _ { i - 1 } } } \| { \boldsymbol x } _ { t _ { i - 2 } } \| _ { 2 } + 2 L _ { x } L _ { \mu } + L _ { x } \delta _ { i - 1 } \| { \boldsymbol D } _ { t _ { i - 1 } } \| _ { 2 } + \frac { \sqrt { 2 } L _ { x } \sigma _ { t _ { i - 1 } } } { \sqrt { \delta _ { i - 1 } } \alpha _ { t _ { i - 1 } } } \| { \boldsymbol z } _ { i - 1 } \| _ { 2 } + L _ { t _ { i - 1 } } \| _ { 2 } } } \\ & { } & { \leq \displaystyle \sum _ { m = 1 } ^ { i - 1 } ( L _ { x } \delta ) ^ { i - m - 1 } \bigg ( \frac { 2 L _ { x } } { \alpha _ { t _ { m } } } \| { \boldsymbol x } _ { t _ { m - 1 } } \| _ { 2 } + 2 L _ { x } L _ { \mu } + L _ { t } + \frac { \sqrt { 2 } L _ { x } \sigma _ { t _ { m } } } { \sqrt { \delta _ { m } } \alpha _ { t _ { m } } } \| { \boldsymbol z } _ { m } \| _ { 2 } \bigg ) . } \end{array}\tag{118}
$$

By taking expectation, we get

$$
\begin{array} { r l } & { \mathbb { E } [ \| D _ { t _ { i } } \| _ { 2 } ^ { 2 } ] \lesssim 3 \mathbb { E } \Big [ \Big ( \displaystyle \sum _ { m = 1 } ^ { i - 1 } ( L _ { x } \delta ) ^ { i - m - 1 } \| x _ { t _ { m - 1 } } \| _ { 2 } \Big ) ^ { 2 } \Big ] + 3 \Big ( \displaystyle \sum _ { m = 1 } ^ { i - 1 } ( L _ { x } \delta ) ^ { i - m - 1 } ( 2 L _ { x } L _ { \mu } + L _ { t } ) \Big ) ^ { 2 } } \\ & { \qquad + 3 \mathbb { E } \Big [ \Big ( \displaystyle \sum _ { m = 1 } ^ { i - 1 } ( L _ { x } \delta ) ^ { i - m - 1 } \displaystyle \frac { \sqrt { 2 } L _ { x } \sigma _ { t _ { m } } } { \sqrt { \delta _ { m } } \alpha _ { t _ { m } } } \| z _ { m } \| _ { 2 } \Big ) ^ { 2 } \Big ] . } \end{array}\tag{119}
$$

We now estimate each of the three terms on the right-hand side of (119).

For the first term, by Cauchy-Schwarz and the moment bound $\mathbb { E } [ \| \pmb { x } _ { t _ { m - 1 } } \| _ { 2 } ^ { 2 } ] = O ( 1 )$ , we have

$$
\begin{array} { r l } {  { \mathbb { E } \Big [ \Big ( \sum _ { m = 1 } ^ { i - 1 } ( L _ { x } \delta ) ^ { i - m - 1 } \| { \pmb x } _ { t _ { m - 1 } } \| _ { 2 } \Big ) ^ { 2 } \Big ] \leq \mathbb { E } \Big [ \sum _ { m = 1 } ^ { i - 1 } ( L _ { x } \delta ) ^ { i - m - 1 } \sum _ { m = 1 } ^ { i - 1 } ( L _ { x } \delta ) ^ { i - m - 1 } \| { \pmb x } _ { t _ { m - 1 } } \| _ { 2 } ^ { 2 } \Big ] } } \\ & { \leq \frac { 1 } { 1 - L _ { x } \delta } \sum _ { m = 1 } ^ { i - 1 } ( L _ { x } \delta ) ^ { i - m - 1 } \mathbb { E } [ \| { \pmb x } _ { t _ { m - 1 } } \| _ { 2 } ^ { 2 } ] } \end{array}
$$

$$
\lesssim \Big ( \frac { 1 } { 1 - L _ { x } \delta } \Big ) ^ { 2 } = O ( 1 ) ,
$$

for suficiently small δ such that $L _ { x } \delta \leq 1 / 2$ . For the second term, we have

$$
\Big ( \sum _ { m = 1 } ^ { i - 1 } ( L _ { x } \delta ) ^ { i - m - 1 } ( 2 L _ { x } L _ { \mu } + L _ { t } ) \Big ) ^ { 2 } \leq \frac { ( 2 L _ { x } L _ { \mu } + L _ { t } ) ^ { 2 } } { ( 1 - L _ { x } \delta ) ^ { 2 } } = O ( 1 ) .
$$

For the third term, since $\boldsymbol { z } _ { m } \sim \mathcal { N } ( \mathbf { 0 } , I _ { d } )$ are independent standard Gaussian vectors, we have $\mathbb { E } [ \| z _ { m } \| _ { 2 } ^ { 2 } ] = d .$ Applying Cauchy-Schwarz yields

$$
\begin{array} { r l r } {  { \mathbb { E } \Big [ \Big ( \displaystyle \sum _ { m = 1 } ^ { i - 1 } ( L _ { x } \delta ) ^ { i - m - 1 } \frac { \sqrt { 2 } L _ { x } \sigma _ { t _ { m } } } { \sqrt { \delta _ { m } } \alpha _ { t _ { m } } } \| z _ { m } \| _ { 2 } \Big ) ^ { 2 } \Big ] \leq \mathbb { E } \Big [ \displaystyle \sum _ { m = 1 } ^ { i - 1 } ( L _ { x } \delta ) ^ { i - m - 1 } \Big ( \displaystyle \sum _ { m = 1 } ^ { i - 1 } ( L _ { x } \delta ) ^ { i - m - 1 } \frac { 2 L _ { x } ^ { 2 } \sigma _ { t _ { m } } ^ { 2 } } { \delta _ { m } \alpha _ { t _ { m } } ^ { 2 } } \| z _ { m } \| _ { 2 } ^ { 2 } \Big ) \Big ] } } \\ & { } & { \leq \frac { 1 } { 1 - L _ { x } \delta } \displaystyle \sum _ { m = 1 } ^ { i - 1 } ( L _ { x } \delta ) ^ { i - m - 1 } \frac { 2 d L _ { x } ^ { 2 } \sigma _ { t _ { m } } ^ { 2 } } { \delta _ { m } \alpha _ { t _ { m } } ^ { 2 } } = O \Big ( \frac { 1 } { \delta } \Big ) , } \end{array}
$$

where we have used the assumption that max $\delta _ { i } / \operatorname* { m i n } \delta _ { i } = O ( 1 )$ . Combining the above three estimates into (119) yields

$$
\mathbb { E } [ \| D _ { t _ { i } } \| _ { 2 } ^ { 2 } ] = O \Big ( \frac { 1 } { \delta } \Big ) .\tag{120}
$$

Finally, substituting (120) into (117), we obtain

$$
\begin{array} { r l r } {  { \mathbb { E } [ \| { \boldsymbol x } _ { t _ { i - 1 } } - { \boldsymbol x } _ { t _ { i - 2 } } \| _ { 2 } ^ { 2 } ] \le 1 6 \delta _ { i - 1 } ^ { 2 } \mathbb { E } [ \| { \boldsymbol x } _ { t _ { i - 2 } } \| _ { 2 } ^ { 2 } ] + 1 6 \alpha _ { t _ { i - 1 } } ^ { 2 } \delta _ { i - 1 } ^ { 2 } L _ { \mu } ^ { 2 } + 4 \alpha _ { t _ { i - 1 } } ^ { 2 } \delta _ { i - 1 } ^ { 4 } \mathbb { E } [ \| D _ { t _ { i - 1 } } \| _ { 2 } ^ { 2 } ] + 8 \sigma _ { t _ { i - 1 } } ^ { 2 } \delta _ { i - 1 } \mathbb { E } [ \| { \boldsymbol z } _ { i - 1 } \| _ { 2 } ^ { 2 } ] } } \\ & { } & { \lesssim \delta _ { i - 1 } ^ { 2 } + \delta _ { i - 1 } ^ { 3 } + \delta _ { i - 1 } = O ( \delta _ { i - 1 } ) . } \end{array}
$$

Step 2: decomposition of errors. We are now ready to analyze the convergence rate of (17b). Repeating the arguments in (72)-(78), we obtain

$$
\pmb { x } _ { t _ { i } } - \pmb { x } _ { t _ { i } } ^ { \star } = \pmb { A } _ { t _ { i } } \big ( \pmb { x } _ { t _ { i - 1 } } - \pmb { x } _ { t _ { i - 1 } } ^ { \star } \big ) + \frac { \alpha _ { t _ { i } } } { 2 } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } ) \delta _ { i } \pmb { D } _ { t _ { i } } + \pmb { \xi } _ { t _ { i } , 1 } + \pmb { \xi } _ { t _ { i } , 2 } ,\tag{122}
$$

where the matrix $\boldsymbol { A } _ { t _ { i } }$ is defined in (80).

For $i \geq 2 ,$ , we apply the Taylor expansion at $( \pmb { x } _ { t _ { i - 2 } } , t _ { i - 1 } )$ . From the definition of $D _ { t _ { i } } ,$

$$
\begin{array} { r l } {  { D _ { t _ { i } } = \frac { 1 } { \delta _ { i - 1 } } ( \mu _ { \theta } ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) - \mu _ { \theta } ( x _ { t _ { i - 2 } } , t _ { i - 2 } ) ) } } \\ & { = \frac { 1 } { \delta _ { i - 1 } } ( \mu _ { \theta } ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) - \mu _ { \theta } ( x _ { t _ { i - 2 } } , t _ { i - 1 } ) + \mu _ { \theta } ( x _ { t _ { i - 2 } } , t _ { i - 1 } ) - \mu _ { \theta } ( x _ { t _ { i - 2 } } , t _ { i - 2 } ) ) } \\ & { = \frac { 1 } { \delta _ { i - 1 } } \nabla \mu _ { \theta } ( x _ { t _ { i - 2 } } , t _ { i - 1 } ) ^ { \top } ( x _ { t _ { i - 1 } } - x _ { t _ { i - 2 } } ) + \frac { 1 } { \delta _ { i - 1 } } O \big ( \| x _ { t _ { i - 1 } } - x _ { t _ { i - 2 } } \| _ { 2 } ^ { 2 } \big ) + \frac { \partial } { \partial \lambda } \mu _ { \theta } ( x _ { t _ { i - 2 } } , t _ { i - 2 } ) + O ( \delta _ { i - 1 } ) . } \end{array}
$$

Substituting the expression for ${ \pmb x } _ { t _ { i - 1 } } - { \pmb x } _ { t _ { i - 2 } }$ from (116) yields for $i \geq 2$

$$
\begin{array} { r l r } {  { D _ { t _ { i } } = \frac { 1 } { \delta _ { i - 1 } } \bigg ( \mathrm { e } ^ { - \delta _ { i - 1 } } \frac { \sigma _ { t _ { i - 1 } } } { \sigma _ { t _ { i - 2 } } } - 1 \bigg ) \nabla \mu _ { \theta } ( x _ { t _ { i - 2 } } , t _ { i - 1 } ) ^ { \top } x _ { t _ { i - 2 } } + \frac { \alpha _ { t _ { i - 1 } } } { \delta _ { i - 1 } } \big ( 1 - \mathrm { e } ^ { - 2 \delta _ { i - 1 } } \big ) \nabla \mu _ { \theta } ( x _ { t _ { i - 2 } } , t _ { i - 1 } ) ^ { \top } \mu _ { \theta } ( x _ { t _ { i - 2 } } , t _ { i - 2 } ) } } \\ & { } & { + \frac { \alpha _ { t _ { i - 1 } } } { 2 } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i - 1 } } ) \nabla \mu _ { \theta } ( x _ { t _ { i - 2 } } , t _ { i - 1 } ) ^ { \top } D _ { t _ { i - 1 } } + \frac { 1 } { \delta _ { i - 1 } } O \big ( \| x _ { t _ { i - 1 } } - x _ { t _ { i - 2 } } \| _ { 2 } ^ { 2 } \big ) + \frac { \partial } { \partial \lambda } \mu _ { \theta } ( x _ { t _ { i - 2 } } , t _ { i - 2 } ) } \\ & { } & { + O ( \delta _ { i - 1 } ) + \frac { \sigma _ { t _ { i - 1 } } } { \delta _ { i - 1 } } \sqrt { 1 - \mathrm { e } ^ { - 2 \delta _ { i - 1 } } } \nabla \mu _ { \theta } ( x _ { t _ { i - 2 } } , t _ { i - 1 } ) ^ { \top } z _ { i - 1 } } \\ & { } & { = : \mathcal { E } _ { t _ { i } , 3 } + \mathcal { E } _ { t _ { i } , 4 } , \qquad ( 1 2 3 ) } \end{array}
$$

where the residual terms $\mathcal { E } _ { t _ { i } , 3 }$ and $\mathcal { E } _ { t _ { i } , 4 }$ are defined as

$$
\mathcal { E } _ { t _ { i } , 3 } : = \frac { 1 } { \delta _ { i - 1 } } \Bigg ( \mathrm { e } ^ { - \delta _ { i } } \cdot \frac { \sigma _ { t _ { i - 1 } } } { \sigma _ { t _ { i - 2 } } } - 1 \Bigg ) \nabla \mu _ { \theta } ( x _ { t _ { i - 2 } } , t _ { i - 1 } ) ^ { \top } x _ { t _ { i - 2 } } + \frac { \alpha _ { t _ { i - 1 } } } { \delta _ { i - 1 } } \big ( 1 - \mathrm { e } ^ { - 2 \delta _ { i - 1 } } \big ) \nabla \mu _ { \theta } ( x _ { t _ { i - 2 } } , t _ { i - 1 } ) ^ { \top } \mu _ { \theta } ( x _ { t _ { i - 2 } } , t _ { i - 2 } )
$$

$$
\begin{array} { l } { { + \displaystyle \frac { \alpha _ { t _ { i - 1 } } } { 2 } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i - 1 } } ) \nabla \mu _ { \theta } ( x _ { t _ { i - 2 } } , t _ { i - 1 } ) ^ { \top } D _ { t _ { i - 1 } } + \displaystyle \frac { 1 } { \delta _ { i - 1 } } O \big ( \| x _ { t _ { i - 1 } } - x _ { t _ { i - 2 } } \| _ { 2 } ^ { 2 } \big ) + \displaystyle \frac { \partial } { \partial \lambda } \mu _ { \theta } ( x _ { t _ { i - 2 } } , t _ { i - 2 } ) } } \\  { + O ( \delta _ { i - 1 } ) , \qquad ( } \end{array}\tag{124a}
$$

$$
\mathcal { E } _ { t _ { i } , 4 } : = \frac { \sigma _ { t _ { i - 1 } } } { \delta _ { i - 1 } } \sqrt { 1 - \mathrm { e } ^ { - 2 \delta _ { i - 1 } } } \nabla \mu _ { \theta } ( \pmb { x } _ { t _ { i - 2 } } , t _ { i - 1 } ) ^ { \top } z _ { i - 1 } ,\tag{124b}
$$

and $\mathcal { E } _ { t _ { 1 } , 3 } = \mathcal { E } _ { t _ { 1 } , 4 } = \mathbf { 0 }$ . Denote $c _ { 1 } = 0$ and $\begin{array} { r } { c _ { i } = \frac { \alpha _ { t _ { i } } } { 2 } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } ) \delta _ { i } } \end{array}$ for $i \geq 2$ . Substituting (123) into (122) gives

$$
\pmb { x } _ { t _ { i } } - \pmb { x } _ { t _ { i } } ^ { \star } = \pmb { A } _ { t _ { i } } ( \pmb { x } _ { t _ { i - 1 } } ^ { \star } ) \big ( \pmb { x } _ { t _ { i - 1 } } - \pmb { x } _ { t _ { i - 1 } } ^ { \star } \big ) + \pmb { \xi } _ { t _ { i } , 1 } + \pmb { \xi } _ { t _ { i } , 2 } + c _ { i } \pmb { \xi } _ { t _ { i } , 3 } + c _ { i } \pmb { \xi } _ { t _ { i } , 4 }
$$

where we have used the bound

$$
\frac { \alpha _ { t _ { i } } } { 2 } \big ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } \big ) \delta _ { i } \leq \alpha _ { t _ { i } } \delta _ { i } ^ { 2 } = O ( \delta _ { i } ^ { 2 } ) .
$$

By recursion, we get the final decomposition

$$
\pmb { x } _ { t _ { M } } - \pmb { x } _ { t _ { M } } ^ { \star } = \sum _ { m = 1 } ^ { M } \prod _ { j = m + 1 } ^ { M } \pmb { A } _ { t _ { j } } \bigg ( \pmb { \xi } _ { t _ { m } , 1 } + \pmb { \xi } _ { t _ { m } , 2 } + c _ { m } \pmb { \xi } _ { t _ { m } , 3 } + c _ { m } \pmb { \xi } _ { t _ { m } , 4 } \bigg ) .\tag{125}
$$

Taking expectations and applying the uniform product bound (84), we obtain

$$
\begin{array} { r l } & { \mathbb { E } [ \| \pmb { x } _ { t _ { M } } - \pmb { x } _ { t _ { M } } ^ { \star } \| _ { 2 } ] \lesssim \displaystyle \sum _ { m = 1 } ^ { M } \mathbb { E } [ \| \pmb { \mathcal { E } } _ { t _ { m } , 1 } \| _ { 2 } ] + \mathbb { E } \bigg [ \bigg \| \displaystyle \sum _ { m = 1 } ^ { M } \prod _ { j = m + 1 } ^ { M } A _ { t _ { j } } \mathcal { E } _ { t _ { m } , 2 } \bigg \| _ { 2 } \bigg ] } \\ & { \qquad + \displaystyle \sum _ { m = 1 } ^ { M } \delta _ { m } ^ { 2 } \mathbb { E } [ \| \pmb { \mathcal { E } } _ { t _ { m } , 3 } \| _ { 2 } ] + \mathbb { E } \bigg [ \bigg \| \displaystyle \sum _ { m = 2 } ^ { M } c _ { m } \displaystyle \prod _ { j = m + 1 } ^ { M } A _ { t _ { j } } \mathcal { E } _ { t _ { m } , 4 } \bigg \| _ { 2 } \bigg ] . } \end{array}\tag{126}
$$

Step 3: control each term separately. We now bound the four terms on the right-hand side of (126) separately.

• Bounding the first term. We aim to establish

$$
\sum _ { m = 1 } ^ { M } \mathbb { E } [ \| \mathcal { E } _ { t _ { m } , 1 } \| _ { 2 } ] = O ( \delta ) ,\tag{127}
$$

Recalling the analysis in Step 4 of Section C.1, it sufices to prove the second-moment bound

$$
\mathbb { E } [ \| \pmb { x } _ { t _ { m } } - \pmb { x } _ { t _ { m } } ^ { \star } \| _ { 2 } ^ { 2 } ] = O ( \delta ) .
$$

To this end, we repeat the arguments leading to (101), which now include the additional drift-diference term:

$$
\begin{array} { r l } & { \| { \boldsymbol x } _ { t _ { m } } - { \boldsymbol x } _ { t _ { m } } ^ { \star } \| _ { 2 } \leq \frac { \sigma _ { t _ { m } } } { \sigma _ { t _ { m - 1 } } } \mathrm { e } ^ { - \delta _ { m } } \left\| { \boldsymbol x } _ { t _ { m - 1 } } - { \boldsymbol x } _ { t _ { m - 1 } } ^ { \star } \right\| _ { 2 } + \frac { \alpha _ { t _ { m } } } { \alpha _ { t _ { m - 1 } } } L _ { x } \big ( 1 - \mathrm { e } ^ { - 2 \delta _ { m } } \big ) \left\| { \boldsymbol x } _ { t _ { m - 1 } } - { \boldsymbol x } _ { t _ { m - 1 } } ^ { \star } \right\| _ { 2 } } \\ & { \qquad + \left. 2 { \boldsymbol B } _ { t _ { m } } + \frac { 1 } { 2 } \delta _ { m } \alpha _ { t _ { m } } \big ( 1 - \mathrm { e } ^ { - 2 \delta _ { m } } \big ) \right\| { \boldsymbol D } _ { t _ { m } } \big \| _ { 2 } , } \end{array}\tag{128}
$$

where $B _ { t _ { m } }$ is defined in (102). Repeating the derivation of (105), we obtain

$$
\begin{array} { r l } & { \mathbb { E } [ \| x _ { t _ { m } } - x _ { t _ { m } } ^ { \star } \| _ { 2 } ^ { 2 } ] \leq \big ( 1 + \frac { 2 \alpha _ { t _ { m } } L _ { \alpha } \delta _ { m } } { \alpha _ { t _ { m } - 1 } } \big ) ^ { 2 } \mathbb { E } \big [ \| x _ { t _ { m - 1 } } - x _ { t _ { m - 1 } } ^ { \star } \| _ { 2 } ^ { 2 } \big ] } \\ & { \qquad + 2 \big ( 1 + \frac { 2 \alpha _ { t _ { m } } L _ { \alpha } \delta _ { m } } { \alpha _ { t _ { m - 1 } } } \big ) \mathbb { E } \big [ ( 2 B _ { t _ { m } } + O ( \delta _ { m } ^ { 2 } ) \| D _ { t _ { m } } \| _ { 2 } ) \big \| x _ { t _ { m - 1 } } - x _ { t _ { m - 1 } } ^ { \star } \big \| _ { 2 } \big ] + \mathbb { E } \big [ ( 2 B _ { t _ { m } } + O ( \delta _ { m } ^ { 2 } ) \| D _ { t _ { m } } \| _ { 2 } ) ^ { 2 } \big ] } \\ & { \overset { ( a ) } { \leq } \big ( 1 + \frac { 2 \alpha _ { t _ { m } } L _ { \alpha } \delta _ { m } } { \alpha _ { t _ { m - 1 } } } \big ) ^ { 2 } \mathbb { E } \big [ \| x _ { t _ { m - 1 } } - x _ { t _ { m - 1 } } ^ { \star } \| _ { 2 } ^ { 2 } \big ] } \end{array}
$$

$$
\begin{array} { r l r } {  { + 2 \big ( 1 + \frac { 2 \alpha _ { t _ { m } } L _ { x } \delta _ { m } } { \alpha _ { t _ { m - 1 } } } \big ) O \big ( \delta _ { m } ^ { 1 . 5 } \big ) \sqrt { \mathbb { E } \big [ \big \| \pmb { x } _ { t _ { m - 1 } } - \pmb { x } _ { t _ { m - 1 } } ^ { \star } \big \| _ { 2 } ^ { 2 } \big ] } + O \big ( \delta _ { m } ^ { 3 } \big ) } } \\ & { \leq \big ( 1 + \frac { 5 \alpha _ { t _ { m } } L _ { x } \delta _ { m } } { \alpha _ { t _ { m - 1 } } } + O ( \delta _ { m } ) \big ) \mathbb { E } \big [ \big \| \pmb { x } _ { t _ { m - 1 } } - \pmb { x } _ { t _ { m - 1 } } ^ { \star } \big \| _ { 2 } ^ { 2 } \big ] + O ( \delta _ { m } ^ { 2 } ) , } & \end{array}\tag{129}
$$

where (a) holds since

$$
\begin{array} { r } { \mathbb { E } [ ( 2 B _ { t _ { m } } + O ( \delta _ { m } ^ { 2 } ) \| D _ { t _ { m } } \| _ { 2 } ) ^ { 2 } ] \lesssim \mathbb { E } [ B _ { t _ { m } } ^ { 2 } ] + \delta _ { m } ^ { 4 } \mathbb { E } [ \| D _ { t _ { m } } \| _ { 2 } ^ { 2 } ] = O ( \delta _ { m } ^ { 3 } ) . } \end{array}
$$

By recursion, we have

$$
\mathbb { E } [ \| \pmb { x } _ { t _ { m } } - \pmb { x } _ { t _ { m } } ^ { \star } \| _ { 2 } ^ { 2 } ] \lesssim \sum _ { i = 1 } ^ { m } \exp \Big ( \sum _ { j = i + 1 } ^ { m } \big ( \frac { 5 \alpha _ { t _ { j } } L _ { x } \delta _ { j } } { \alpha _ { t _ { j - 1 } } } + O ( \delta _ { j } ) \big ) \Big ) \delta _ { i } ^ { 2 } = O ( \delta ) .\tag{130}
$$

This establishes (127).

• Bounding the second term. Recall the definition of $\mathcal { E } _ { t _ { m } , 2 }$ (cf. (79b)), which is independent of the update rule for $\mathbf { x } _ { t _ { i } }$ .Therefore, the analysis in Appendix C.1 remains valid, and we have

$$
\mathbb { E } \bigg [ \Big \| \sum _ { m = 1 } ^ { M } \prod _ { j = m + 1 } ^ { M } A _ { t _ { j } } \mathscr { E } _ { t _ { m } , 2 } \Big \| _ { 2 } \bigg ] = O ( \delta ) .\tag{131}
$$

• Bounding the third term. For the definition of $\mathcal { E } _ { t _ { i } , 3 }$ (cf. (124a)), we have

$$
\begin{array} { r l } & { \mathbb { E } [ \| \mathcal { E } _ { t _ { m - 3 } } \| _ { 2 } ] \le \displaystyle \frac { 1 } { \delta _ { m - 1 } } \Big | \mathrm { e } ^ { - \delta _ { m - 1 } } \frac { \sigma _ { t _ { m - 1 } } } { \sigma _ { t _ { m - 2 } } } - 1 \Big | \frac { L _ { x } } { \alpha _ { t _ { m - 1 } } } \mathbb { E } [ \| x _ { t _ { m - 2 } } \| _ { 2 } ] + \frac { \alpha _ { t _ { m - 1 } } } { \delta _ { m - 1 } } \big ( 1 - \mathrm { e } ^ { - 2 \delta _ { m - 1 } } \big ) \frac { L _ { x } L _ { \mu } } { \alpha _ { t _ { m - 1 } } } } \\ & { \qquad + \frac { \alpha _ { t _ { m - 1 } } } { 2 } \big ( 1 - \mathrm { e } ^ { - 2 \delta _ { m - 1 } } \big ) \frac { L _ { x } } { \alpha _ { t _ { m - 1 } } } \mathbb { E } [ \| D _ { t _ { m - 1 } } \| _ { 2 } ] + \frac { 1 } { \delta _ { m - 1 } } O \big ( \mathbb { E } [ \| x _ { t _ { m - 1 } } - x _ { t _ { m - 2 } } \| _ { 2 } ^ { 2 } ] \big ) + L _ { t } + O ( \delta _ { m - 1 } ) } \\ & { \overset { ( a ) } { \lesssim } \mathbb { E } [ \| x _ { t _ { m - 2 } } \| _ { 2 } ] + O ( 1 ) + O ( \delta _ { m - 1 } ) \mathbb { E } [ \| D _ { t _ { m - 1 } } \| _ { 2 } ] + \frac { 1 } { \delta _ { m - 1 } } \mathbb { E } \big [ \| x _ { t _ { m - 1 } } - x _ { t _ { m - 2 } } \| _ { 2 } ^ { 2 } \big ] \overset { ( b ) } { = } O ( 1 ) , } \end{array}
$$

where (a) uses the elementary bound (75), and (b) applies the crude estimates (114), (120), and (121) from Step 1. Consequently, the third term in (126) is bounded by

$$
\sum _ { m = 1 } ^ { M } \delta _ { m } ^ { 2 } \mathbb { E } [ \| \mathcal { E } _ { t _ { m } , 3 } \| _ { 2 } ] \lesssim \sum _ { m = 1 } ^ { M } \delta _ { m } ^ { 2 } = O ( \delta ) .\tag{132}
$$

• Bounding the fourth term. We use the cumulative matrices $P _ { i }$ and the filtration introduced in Section C.1. Since the Jacobian in (124b) is evaluated at $\mathbf { { x } } _ { t _ { m - 2 } } ,$ it is $\mathcal { F } _ { t _ { m - 2 } }$ -measurable. Independence of $z _ { m - 1 }$ and $\mathcal { F } _ { t _ { m - 2 } }$ gives

$$
\mathbb { E } [ \mathcal { E } _ { t _ { m } , 4 } \mid \mathcal { F } _ { t _ { m - 2 } } ] = \mathbf { 0 } , \qquad \mathbb { E } [ \| \mathcal { E } _ { t _ { m } , 4 } \| _ { 2 } ^ { 2 } \mid \mathcal { F } _ { t _ { m - 2 } } ] \leq \frac { C } { \delta _ { m - 1 } } , \quad m \geq 2 .\tag{133}
$$

The matrix $P _ { m } ^ { - 1 }$ still depends on the preceding noise increment. We therefore write

$$
\begin{array} { r l } & { P _ { m } ^ { - 1 } c _ { m } \mathcal { E } _ { t _ { m } , 4 } = P _ { m - 1 } ^ { - 1 } c _ { m } \mathcal { E } _ { t _ { m } , 4 } + R _ { m } , } \\ & { \qquad R _ { m } = c _ { m } ( P _ { m } ^ { - 1 } - P _ { m - 1 } ^ { - 1 } ) \mathcal { E } _ { t _ { m } , 4 } . } \end{array}
$$

Using the identity $\begin{array} { r } { \prod _ { i = m + 1 } ^ { M } \pmb { A } _ { t _ { i } } = \pmb { P } _ { M } \pmb { P } _ { m } ^ { - 1 } } \end{array}$ and the bound for $P _ { M }$ yields

$$
\mathbb { E } \Big [ \Big \lVert \sum _ { m = 2 } ^ { M } c _ { m } \prod _ { i = m + 1 } ^ { M } \pmb { A } _ { t _ { i } } \pmb { \mathcal { E } } _ { t _ { m } , 4 } \Big \rVert _ { 2 } \Big ] = \mathbb { E } \Big \lVert \mathbf { \mathit { P } } _ { M } \sum _ { m = 2 } ^ { M } \mathbf { \mathit { P } } _ { m } ^ { - 1 } c _ { m } \mathbf { \mathcal { E } } _ { t _ { m } , 4 } \Big \rVert _ { 2 }
$$

$$
\leq C \mathbb { E } \Big \| \sum _ { m = 2 } ^ { M } P _ { m - 1 } ^ { - 1 } c _ { m } \mathcal { E } _ { t _ { m } , 4 } \Big \| _ { 2 } + C \mathbb { E } \big [ \sum _ { m = 2 } ^ { M } \| \pmb { R _ { m } } \| _ { 2 } \big ] .\tag{134}
$$

By the bounds for $P _ { i } ^ { - 1 }$ and $A _ { t _ { i } } ^ { - 1 }$ established previously, we have

$$
\| P _ { m } ^ { - 1 } - P _ { m - 1 } ^ { - 1 } \| _ { \mathsf { o p } } = \| P _ { m - 1 } ^ { - 1 } ( A _ { t _ { m } } ^ { - 1 } - I _ { d } ) \| _ { \mathsf { o p } } \leq C \delta _ { m } .
$$

Since $c _ { m } = { \cal O } ( \delta _ { m } ^ { 2 } )$ and the statistical property in (133), the remainder satisfies

$$
\mathbb { E } \big [ \sum _ { m = 2 } ^ { M } \| \pmb { R _ { m } } \| _ { 2 } \big ] \leq \sum _ { m = 2 } ^ { M } \mathbb { E } \| \pmb { R _ { m } } \| _ { 2 } \leq C \sum _ { m = 2 } ^ { M } \frac { \delta _ { m } ^ { 3 } } { \sqrt { \delta _ { m - 1 } } } \leq C \delta ^ { 3 / 2 } \sum _ { m = 2 } ^ { M } \delta _ { m } = O ( \delta ^ { 3 / 2 } ) .\tag{135}
$$

Moreover, $P _ { m - 1 } ^ { - 1 }$ is $\mathcal { F } _ { t _ { m - 2 } }$ -measurable. Hence the first term in the right-hand-side of (134) satisfies

$$
\mathbb { E } \Big \| \sum _ { m = 2 } ^ { M } \pmb { P } _ { m - 1 } ^ { - 1 } c _ { m } \pmb { \mathscr { E } } _ { t _ { m } , 4 } \Big \| _ { 2 } ^ { 2 } = \sum _ { m = 2 } ^ { M } \mathbb { E } \| \pmb { P } _ { m - 1 } ^ { - 1 } c _ { m } \pmb { \mathscr { E } } _ { t _ { m } , 4 } \| _ { 2 } ^ { 2 }\tag{136}
$$

$$
\le C \sum _ { m = 2 } ^ { M } \frac { \delta _ { m } ^ { 4 } } { \delta _ { m - 1 } } \le C \sum _ { m = 2 } ^ { M } \delta _ { m } ^ { 3 } = O ( \delta ^ { 2 } ) ,\tag{137}
$$

where the first equality follows from $\mathbb { E } [ \mathcal { E } _ { t _ { m } , 4 } ~ | ~ \mathcal { F } _ { t _ { m - 2 } } ] = \mathbf { 0 }$ and the $\mathcal { F } _ { t _ { m - 2 } }$ -measurability of $P _ { m - 1 } ^ { - 1 }$ Collecting (134), (135), and (136) yields

$$
\begin{array} { r l } {  { \mathbb { E } \Big \| \sum _ { m = 2 } ^ { M } c _ { m } \prod _ { i = m + 1 } ^ { M } A _ { t _ { i } } \mathcal { E } _ { t _ { m } , 4 } \Big \| _ { 2 } } } \\ & { \leq C \Big ( \mathbb { E } \Big \| \sum _ { m = 2 } ^ { M } \mathbf { P } _ { m - 1 } ^ { - 1 } c _ { m } \mathcal { E } _ { t _ { m } , 4 } \Big \| _ { 2 } ^ { 2 } \Big ) ^ { 1 / 2 } + C \sum _ { m = 2 } ^ { M } \mathbb { E } \| R _ { m } \| _ { 2 } } \\ & { = O ( \delta ) + O ( \delta ^ { 3 / 2 } ) = O ( \delta ) . } \end{array}\tag{138}
$$

Finally, collecting (127), (131), (132), and (138) into (126) yields the desired result:

$$
\mathbb { E } [ \| \pmb { x } _ { t _ { M } } - \pmb { x } _ { t _ { M } } ^ { \star } \| _ { 2 } ] = O ( \delta ) .
$$

This completes the proof.

Proof of (114). Recall that we have already established $\mathbb { E } [ \| \pmb { x } _ { t _ { m } } ^ { \star } \| _ { 2 } ^ { 2 } ] = O ( 1 )$ in (71). Therefore, it sufices to prove

$$
\mathbb { E } [ \| \pmb { x } _ { t _ { m } } - \pmb { x } _ { t _ { m } } ^ { \star } \| _ { 2 } ^ { 2 } ] = O ( 1 ) .
$$

From the decomposition in (129), we have

$$
\begin{array} { r l } & { \mathbb { E } [ \| { \boldsymbol x } _ { t _ { m } } - { \boldsymbol x } _ { t _ { m } } ^ { \star } \| _ { 2 } ^ { 2 } ] \le \big ( 1 + \frac { 2 \alpha _ { t _ { m } } L _ { x } \delta _ { m } } { \alpha _ { t _ { m - 1 } } } \big ) ^ { 2 } \mathbb { E } [ \| { \boldsymbol x } _ { t _ { m - 1 } } - { \boldsymbol x } _ { t _ { m - 1 } } ^ { \star } \| _ { 2 } ^ { 2 } ] } \\ & { \qquad + 2 \big ( 1 + \frac { 2 \alpha _ { t _ { m } } L _ { x } \delta _ { m } } { \alpha _ { t _ { m - 1 } } } \big ) \mathbb { E } \big [ ( 2 B _ { t _ { m } } + O ( \delta _ { m } ^ { 2 } ) \| D _ { t _ { m } } \| _ { 2 } ) \big \| { \boldsymbol x } _ { t _ { m - 1 } } - { \boldsymbol x } _ { t _ { m - 1 } } ^ { \star } \big \| _ { 2 } \big ] + \mathbb { E } [ ( 2 B _ { t _ { m } } + O ( \delta _ { m } ^ { 2 } ) \| D _ { t _ { m } } \| _ { 2 } ) ^ { 2 } ] . } \end{array}\tag{139}
$$

By Assumption A1, the drift term is uniformly bounded by $L _ { \mu }$ , i.e., $\| \pmb { \mu } _ { \boldsymbol { \theta } } ( \pmb { x } _ { t _ { m - 1 } } , t _ { m - 1 } ) \| _ { 2 } \leq L _ { \mu }$ . Consequently,

$$
\| D _ { t _ { m } } \| _ { 2 } \leq \frac { 1 } { \delta _ { m - 1 } } \big ( \| \mu _ { \theta } ( \pmb { x } _ { t _ { m - 1 } } , t _ { m - 1 } ) \| _ { 2 } + \| \mu _ { \theta } ( \pmb { x } _ { t _ { m - 2 } } , t _ { m - 2 } ) \| _ { 2 } \big ) \leq \frac { 2 L _ { \mu } } { \delta _ { m - 1 } } .
$$

Using the assumption $\delta _ { m } / \delta _ { m - 1 } = { \cal O } ( 1 )$ and the bound $\mathbb { E } [ B _ { t _ { m } } ^ { 2 } ] = O ( \delta _ { m } ^ { 3 } )$ in (104), (which is independent of the update rule for $\boldsymbol { x } _ { t _ { m } } )$ , we obtain

$$
\mathbb { E } [ ( 2 B _ { t _ { m } } + O ( \delta _ { m } ^ { 2 } ) \| D _ { t _ { m } } \| _ { 2 } ) ^ { 2 } ] \lesssim \mathbb { E } [ B _ { t _ { m } } ^ { 2 } ] + \delta _ { m } ^ { 4 } \mathbb { E } [ \| D _ { t _ { m } } \| _ { 2 } ^ { 2 } ] \lesssim \mathbb { E } [ B _ { t _ { m } } ^ { 2 } ] + \frac { \delta _ { m } ^ { 4 } } { \delta _ { m - 1 } ^ { 2 } } = O ( \delta _ { m } ^ { 2 } ) .
$$

Substituting this estimate into (139) yields

$$
\begin{array} { r l } { \displaystyle \mathbb { E } [ \| \pmb { x } _ { t _ { m } } - \pmb { x } _ { t _ { m } } ^ { \star } \| _ { 2 } ^ { 2 } ] \leq \big ( 1 + \frac { 5 \alpha _ { t _ { m } } L _ { x } \delta _ { m } } { \alpha _ { t _ { m - 1 } } } + O ( \delta _ { m } ) \big ) ^ { 2 } \mathbb { E } \big [ \big \| \pmb { x } _ { t _ { m - 1 } } - \pmb { x } _ { t _ { m - 1 } } ^ { \star } \big \| _ { 2 } ^ { 2 } \big ] } & { } \\ { + O ( \delta _ { m } ^ { 3 } ) \mathbb { E } [ \| \pmb { D } _ { t _ { m } } \| _ { 2 } ^ { 2 } ] + O \Big ( \frac { 1 } { \delta _ { m } } \Big ) \mathbb { E } [ B _ { t _ { m } } ^ { 2 } ] } & { } \\ { \lesssim \displaystyle \sum _ { i = 1 } ^ { m } \exp \Big ( \displaystyle \sum _ { j = i + 1 } ^ { m } \big ( \frac { 5 \alpha _ { t _ { j } } L _ { x } \delta _ { j } } { \alpha _ { t _ { j - 1 } } } + O ( \delta _ { j } ) \big ) \Big ) \delta _ { i } = O ( 1 ) . } \end{array}\tag{140}
$$

This completes the proof of (114).

## D Proof of stochastic samplers convergence lower bounds (Theorem 3)

Assume that $\mathbf { \nabla } \mathbf { x } _ { 0 } \sim \mathcal { N } ( \mathbf { 0 } , \gamma ^ { 2 } I _ { d } )$ follows the Gaussian distribution with isotropic covariance matrix, where $\gamma > 0$ is fixed. In this case, the score function reduces to

$$
\pmb { s } _ { t } ^ { \star } ( \pmb { x } ) = \nabla \log q _ { t } ( \pmb { x } ) = - \frac { \pmb { x } } { \alpha _ { t } ^ { 2 } \gamma ^ { 2 } + \sigma _ { t } ^ { 2 } } = - \frac { 1 } { \alpha _ { t } ^ { 2 } } \frac { \pmb { x } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t } } } .
$$

Then the data predictor is given by

$$
\begin{array} { r } { \mu _ { \theta } ( \boldsymbol { x } , t ) = \displaystyle \frac { 1 } { \alpha _ { t } } \big ( \boldsymbol { x } - \boldsymbol { \sigma } _ { t } \boldsymbol { \varepsilon } _ { \theta } ( \boldsymbol { x } , t ) \big ) = \frac { 1 } { \alpha _ { t } } \big ( \boldsymbol { x } + \boldsymbol { \sigma } _ { t } ^ { 2 } \boldsymbol { s } _ { t } ^ { \star } ( \boldsymbol { x } ) \big ) } \\ { = \displaystyle \frac { 1 } { \alpha _ { t } } \big ( 1 - \frac { \mathrm { e } ^ { - 2 \lambda _ { t } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t } } } \big ) \boldsymbol { x } = \frac { \gamma ^ { 2 } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t } } } \frac { \boldsymbol { x } } { \alpha _ { t } } . } \end{array}\tag{141}
$$

Substituting into the SDE trajectory (29) yields

$$
\pmb { x } _ { t } ^ { \star } = \frac { \sigma _ { t } } { \sigma _ { t _ { 0 } } } \mathrm { e } ^ { - ( \lambda _ { t } - \lambda _ { t _ { 0 } } ) } \pmb { x } _ { t _ { 0 } } ^ { \star } + 2 \alpha _ { t } \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t } } \frac { \gamma ^ { 2 } \mathrm { e } ^ { - 2 ( \lambda _ { t } - \lambda ) } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } \frac { \pmb { x } _ { t ( \lambda ) } ^ { \star } } { \alpha _ { t ( \lambda ) } } \mathrm { d } \lambda + \sqrt { 2 } \sigma _ { t } \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t } } \mathrm { e } ^ { - ( \lambda _ { t } - \lambda ) } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ) } .\tag{142}
$$

Define

$$
\pmb { y } _ { t } ^ { \star } : = \frac { \mathrm { e } ^ { \lambda _ { t } } } { \sigma _ { t } } \pmb { x } _ { t } ^ { \star } ,\tag{143}
$$

which therefore conforms to

$$
\begin{array} { r l } & { y _ { t } ^ { \star } = \frac { \mathrm { e } ^ { \lambda _ { t } } } { \sigma _ { t _ { 0 } } } \mathrm { e } ^ { - ( \lambda _ { t } - \lambda _ { t _ { 0 } } ) } x _ { t _ { 0 } } ^ { \star } + 2 \mathrm { e } ^ { 2 \lambda _ { t } } \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t } } \frac { \gamma ^ { 2 } \mathrm { e } ^ { - 2 ( \lambda _ { t } - \lambda ) } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } \frac { x _ { t ( \lambda ) } ^ { \star } } { \alpha _ { t ( \lambda ) } } \mathrm { d } \lambda + \sqrt { 2 } \mathrm { e } ^ { \lambda _ { t } } \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t } } \mathrm { e } ^ { - ( \lambda _ { t } - \lambda ) } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ) } } \\ & { \quad = y _ { t _ { 0 } } ^ { \star } + 2 \mathrm { e } ^ { 2 \lambda _ { t } } \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t } } \frac { \gamma ^ { 2 } \mathrm { e } ^ { - 2 \lambda _ { t } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } y _ { t ( \lambda ) } ^ { \star } \mathrm { d } \lambda + \sqrt { 2 } \mathrm { e } ^ { \lambda _ { t } } \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t } } \mathrm { e } ^ { - ( \lambda _ { t } - \lambda ) } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ) } . } \end{array}\tag{144}
$$

We conclude that the solution to (144) is given by

$$
y _ { t } ^ { \star } = \frac { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t } } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { 0 } } } } y _ { t _ { 0 } } ^ { \star } + \sqrt { 2 } ( 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t } } ) \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t } } \frac { \mathrm { e } ^ { \lambda } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ) } .\tag{145}
$$

To check this, we insert it into the right-hand-side of (144) and obtain

$$
\begin{array} { l } { { y _ { t _ { 0 } } ^ { * } + 2 \mathrm { e } ^ { 2 \lambda _ { t } } \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t } } \frac { \gamma ^ { 2 } \mathrm { e } ^ { - 2 \lambda _ { t } } } { \gamma ^ { 2 } + \mathrm { e } ^ { 2 \lambda } } y _ { t ( \lambda ) } ^ { * } \mathrm { d } \lambda + \sqrt { \mathrm { 2 } } \mathrm { e } ^ { \lambda _ { t } } \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t } } \mathrm { e } ^ { - ( \lambda _ { t } - \lambda ) } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ) } } } \\ { { = y _ { t _ { 0 } } ^ { * } + 2 \mathrm { e } ^ { 2 \lambda _ { t } } \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t } } \frac { \gamma ^ { 2 } \mathrm { e } ^ { - 2 \lambda _ { t } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } \frac { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t } } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { 0 } } } } \mathrm { d } \lambda y _ { t _ { 0 } } ^ { * } + 2 \sqrt { \mathrm { 2 } } \mathrm { e } ^ { 2 \lambda _ { t } } \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t } } \frac { \gamma ^ { 2 } \mathrm { e } ^ { - 2 \lambda _ { t } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } ( 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } ) \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda } \frac { \mathrm { e } ^ { \lambda ^ { \prime } } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda ^ { \prime } } } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ^ { \prime } ) } \mathrm { d } \lambda } } \\   + \sqrt { 2 } \mathrm { e } ^ { \lambda _ { t } } \int _  \lambda  \end{array}\tag{146}
$$

where (a) holds since

$$
\begin{array} { l } { { 1 + 2 \mathrm { e } ^ { 2 \lambda _ { t } } \displaystyle \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t } } \frac { \gamma ^ { 2 } \mathrm { e } ^ { - 2 \lambda _ { t } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } \displaystyle \frac { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { 0 } } } } \mathrm { d } \lambda = 1 + 2 \mathrm { e } ^ { 2 \lambda _ { t } } \displaystyle \frac { \gamma ^ { 2 } \mathrm { e } ^ { - 2 \lambda _ { t } } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { 0 } } } } \displaystyle \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t } } \mathrm { e } ^ { 2 \lambda } \mathrm { d } \lambda } } \\ { { { } ~ = 1 + \displaystyle \frac { \gamma ^ { 2 } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { 0 } } } } \left( \mathrm { e } ^ { 2 \lambda _ { t } } - \mathrm { e } ^ { 2 \lambda _ { t _ { 0 } } } \right) } } \\ { { { } ~ = \displaystyle \frac { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t } } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { 0 } } } } . } } \end{array}
$$

For the noise term, we have

$$
\begin{array} { l } { { 2 \mathrm { e } ^ { 2 \lambda _ { t } } \displaystyle \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t } } \frac { \gamma ^ { 2 } \mathrm { e } ^ { - 2 \lambda _ { t } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } ( 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } ) \displaystyle \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda } \frac { \mathrm { e } ^ { \lambda ^ { \prime } } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda ^ { \prime } } } \mathrm { d } \overline { { \omega } } _ { t } ( \lambda ^ { \prime } ) \mathrm { d } \lambda } }  \\ { { = 2 \displaystyle \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t } } \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } \displaystyle \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda } \frac { \mathrm { e } ^ { \lambda ^ { \prime } } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda ^ { \prime } } } \mathrm { d } \overline { { \omega } } _ { t } ( \lambda ^ { \prime } ) \mathrm { d } \lambda } } \\ { { = 2 \displaystyle \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t } } \gamma ^ { 2 } \frac { \mathrm { e } ^ { \lambda ^ { \prime } } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda ^ { \prime } } } \displaystyle \int _ { \lambda ^ { \prime } } ^ { \lambda _ { t } } \mathrm { e } ^ { 2 \lambda } \mathrm { d } \lambda \mathrm { d } \overline { { \omega } } _ { t } ( \lambda ^ { \prime } ) } } \\   = \displaystyle \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t } } \gamma ^ { 2 } \frac { \mathrm { e } ^ { \lambda ^ { \prime } } ( \mathrm { e } ^ { 2 \lambda _ { t } } - \mathrm { e } ^ { 2 \lambda ^ { \prime } } ) } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda ^ { \prime } } } \mathrm { d } \overline { { \omega } } _ \end{array}
$$

and therefore

$$
\begin{array} { l } { { \displaystyle 2 \sqrt { 2 } \mathrm { e } ^ { 2 \lambda _ { t } } \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t } } \frac { \gamma ^ { 2 } \mathrm { e } ^ { - 2 \lambda _ { t } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } ( 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } ) \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda } \frac { \mathrm { e } ^ { \lambda ^ { \prime } } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda ^ { \prime } } } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ^ { \prime } ) } \mathrm { d } \lambda + \sqrt { 2 } \mathrm { e } ^ { \lambda _ { t } } \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t } } \mathrm { e } ^ { - ( \lambda _ { t } - \lambda ) } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ) } \mathrm { d } \lambda } }  \\ { { = \sqrt { 2 } \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t } } \bigg ( \gamma ^ { 2 } \frac { \mathrm { e } ^ { \lambda _ { t } } \mathrm { e } ^ { 2 \lambda _ { t } } - \mathrm { e } ^ { 2 \lambda } ) } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } } + \mathrm { e } ^ { \lambda } \bigg ) \mathrm { d } \overline { { \omega } } _ { t ( \lambda ) } = \sqrt { 2 } \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t } } \bigg ( \mathrm { e } ^ { \lambda } \frac { \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t } } + 1 } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } } \bigg ) \mathrm { d } \overline { { \omega } } _ { t ( \lambda ) } } } \\   = \sqrt { 2 } \big ( 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t } } \big ) \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t } } \frac  \mathrm { e } ^ { \lambda } \end{array}
$$

Inserting into (146) yields

$$
\begin{array} { c } { { \displaystyle { y _ { t _ { 0 } } ^ { \star } + 2 \mathrm { e } ^ { 2 \lambda _ { t } } \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t } } \frac { \gamma ^ { 2 } \mathrm { e } ^ { - 2 \lambda _ { t } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda } } { y _ { t ( \lambda ) } ^ { \star } } \mathrm { d } \lambda + \sqrt { 2 } \mathrm { e } ^ { \lambda _ { t } } \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t } } \mathrm { e } ^ { - ( \lambda _ { t } - \lambda ) } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ) } } } } \\ { { = \displaystyle { \frac { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t } } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { 0 } } } } { y _ { t _ { 0 } } ^ { \star } } + \sqrt { 2 } ( 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t } } ) \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t } } \frac { \mathrm { e } ^ { \lambda } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ) } = y _ { t } ^ { \star } . } } } \end{array}
$$

Hence we have verified that (145) is the solution to (144). Recalling the definition of $\mathbf { \Delta } _ { \mathbf { \mathcal { Y } } _ { t } ^ { \star } }$ in (143), the solution to (142) is given by

$$
\begin{array} { l } { { \displaystyle x _ { t } ^ { \star } = \sigma _ { t } \mathrm { e } ^ { - \lambda _ { t } } y _ { t } ^ { \star } = \sigma _ { t } \mathrm { e } ^ { \lambda _ { t } - 2 \lambda _ { t } } \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t } } } y _ { t _ { 0 } } ^ { \star } + \sqrt { 2 } \sigma _ { t } \mathrm { e } ^ { \lambda _ { t } } \big ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t } } \big ) \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t } } \frac { \mathrm { e } ^ { \lambda } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } } \mathrm { d } \overline { { \omega } } _ { \lambda } } }  \\ { { \displaystyle \quad = \frac { \sigma _ { t } } { \sigma _ { t _ { 0 } } } \mathrm { e } ^ { \lambda _ { t } - \lambda _ { t _ { 0 } } } \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { 0 } } } } x _ { t _ { 0 } } ^ { \star } + \sqrt { 2 } \alpha _ { t } \big ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t } } \big ) \int _ { \lambda _ { t _ { 0 } } } ^ { \lambda _ { t } } \frac { \mathrm { e } ^ { \lambda } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } } \mathrm { d } \overline { { \omega } } _ { \lambda } } } \\   \displaystyle \quad = \frac { \alpha _ { t } } { \alpha _ { t _ { 0 } } } \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { 0 } } } } x _ { t _ { 0 } } ^ { \star } + \sqrt { 2 } \alpha _ { t } \big ( \gamma ^ { 2 } + \mathrm \end{array}\tag{147}
$$

It also holds that for any $\lambda _ { t _ { i } }$

$$
{ \pmb x } _ { t _ { i } } ^ { \star } = \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } } { \pmb x } _ { t _ { i - 1 } } ^ { \star } + \sqrt { 2 } \alpha _ { t _ { i } } ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } ) \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \frac { \mathrm { e } ^ { \lambda } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } } \mathrm { d } \overline { \omega } _ { t ( \lambda ) } .\tag{148}
$$

Below we shall analyze the discretization error of (17a) and (17b) separately.

## D.1 Convergence analysis of SDE-DPMSolver++1

Substituting the expression of data predictor in (141) into the update rule (17a) yields

$$
\begin{array} { r l } & { x _ { t _ { i } } = \mathrm { e } ^ { - \delta _ { i } } \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } x _ { t _ { i - 1 } } + \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } \frac { \gamma ^ { 2 } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } ) } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } } x _ { t _ { i - 1 } } + \sigma _ { t _ { i } } \sqrt { 1 - \mathrm { e } ^ { - 2 \delta _ { i } } } z _ { i } } \\ & { \quad \quad = \mathrm { e } ^ { - 2 \delta _ { i } } \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } x _ { t _ { i - 1 } } + \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } \frac { \gamma ^ { 2 } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } ) } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } } x _ { t _ { i - 1 } } + \sigma _ { t _ { i } } \sqrt { 1 - \mathrm { e } ^ { - 2 \delta _ { i } } } z _ { i } } \\ & { \quad \quad = \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } } x _ { t _ { i - 1 } } + \sqrt { 2 } \sigma _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - ( \lambda _ { t _ { i } } - \lambda ) } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ) } . } \end{array}\tag{149}
$$

Subtracting the $\boldsymbol { x } _ { t _ { i } } ^ { \star }$ in (148) from (149) yields

$$
\begin{array} { l } { { \displaystyle x _ { t _ { i } } - x _ { t _ { i } } ^ { * } = \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } } ( x _ { t _ { i - 1 } } - x _ { t _ { i } } ^ { * } ) + \sqrt { 2 } \sigma _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - ( \lambda _ { t _ { i } } - \lambda ) } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ) } - \sqrt { 2 } \alpha _ { t _ { i } } ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } ) \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \frac { \mathrm { e } ^ { \lambda } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ) } } }  \\   = \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } } ( x _ { t _ { i - 1 } } - x _ { t _ { i } } ^ { * } ) + \sqrt { 2 } \alpha _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { \lambda } \biggl ( \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } - \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } } \biggr ) \mathrm { d } \overline { { \omega } } _  t ( \lambda ) \end{array}
$$

By recursion, we obtain

$$
\begin{array} { r } { x _ { t _ { M } } - x _ { t _ { M } } ^ { \star } = - \sqrt { 2 } \gamma ^ { 2 } \displaystyle \sum _ { m = 1 } ^ { M } \prod _ { i = m + 1 } ^ { M } \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } } \alpha _ { t _ { m } } \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda _ { t _ { m } } } \mathrm { e } ^ { \lambda } \frac { 1 - \mathrm { e } ^ { - 2 ( \lambda _ { t _ { m } } - \lambda ) } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } } \mathrm { d } \overline { \omega } _ { t ( \lambda ) } } \\ { = - \sqrt { 2 } \gamma ^ { 2 } \alpha _ { t _ { M } } \big ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { M } } } \big ) \displaystyle \sum _ { m = 1 } ^ { M } \frac { 1 } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { m } } } } \displaystyle \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda _ { t _ { m } } } \mathrm { e } ^ { \lambda } \frac { 1 - \mathrm { e } ^ { - 2 ( \lambda _ { t _ { m } } - \lambda ) } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } } \mathrm { d } \overline { \omega } _ { t ( \lambda ) } } \end{array}
$$

Note that $\overline { { \omega } } _ { t ( \lambda ) }$ is Brownian motion, $\pmb { x } _ { t _ { M } } - \pmb { x } _ { t _ { M } } ^ { \star }$ is a centered Gaussian random vector with second moment

$$
\mathbb { E } [ \| x _ { t _ { M } } - x _ { t _ { M } } ^ { \star } \| _ { 2 } ^ { 2 } ] = 2 d \gamma ^ { 4 } \alpha _ { t _ { M } } ^ { 2 } ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { M } } } ) ^ { 2 } \sum _ { m = 1 } ^ { M } \frac { 1 } { ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { m } } } ) ^ { 2 } } \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda _ { t _ { m } } } \mathrm { e } ^ { 2 \lambda } \frac { ( 1 - \mathrm { e } ^ { - 2 ( \lambda _ { t _ { m } } - \lambda ) } ) ^ { 2 } } { ( 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } ) ^ { 2 } } \mathrm { d } \lambda .
$$

Note that $( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { M } } } ) / ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { m } } } ) \ge ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { M } } } ) / ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { 0 } } } ) = \Omega ( 1 )$ and

$$
1 - \mathrm { e } ^ { - 2 ( \lambda _ { t _ { m } } - \lambda ) } \geq \lambda _ { t _ { m } } - \lambda , \qquad \mathrm { f o r } \quad \delta _ { m } \leq 1 / 2 .
$$

We introduce $\lambda _ { t _ { m } } ^ { \prime } = ( \lambda _ { t _ { m - 1 } } + \lambda _ { t _ { m } } ) / 2$ and obtain

$$
\begin{array} { r l } { \mathbb { E } [ \| x _ { i \alpha } - x _ { i \alpha } ^ { * } \| _ { \infty } ^ { 2 } ] \stackrel { \le } \sum _ { n = - 1 } ^ { M } \int _ { \Lambda _ { n - 1 } } ^ { x _ { i \alpha } ^ { * } } \int _ { 0 } ^ { 2 \pi / \alpha _ { n } } \frac { \lambda _ { n } ( x _ { i } - \lambda ) ^ { 2 } } { ( 1 + \gamma ^ { 2 } \varepsilon ^ { 2 } \nu ^ { 2 } ) ^ { n } } \mathrm { d } \lambda } & { } \\ { \ } & { \ge \sum _ { n = - 1 } ^ { M } \int _ { \Lambda _ { n - 1 } } ^ { x _ { i \alpha } ^ { * } } \gamma _ { n } ^ { 2 \varepsilon \alpha _ { n } } \frac { \lambda _ { n } ( x _ { i } - \lambda ) ^ { 2 } } { ( 1 + \gamma ^ { 2 } \varepsilon ^ { 2 } \nu ^ { 2 } ) ^ { n } } \mathrm { d } \lambda } \\ & { \ge \frac { 1 } { 4 } \frac { M ^ { 2 } } { \varepsilon _ { n - 1 } } \int _ { \lambda _ { n - 1 } } ^ { x _ { i \alpha } ^ { * } } \frac { \gamma ^ { 2 } e ^ { 2 \alpha _ { n } } } { \left( 1 + \gamma ^ { 2 } \varepsilon ^ { 2 } \nu ^ { 2 } \right) ^ { n } } \mathrm { d } \lambda } \\ & { = \frac { 1 } { 8 } \frac { M ^ { 2 } } { \varepsilon _ { n - 1 } } \delta _ { n } ^ { 2 } \Big ( \frac { 1 } { 1 + \gamma ^ { 2 } \varepsilon ^ { 2 } \nu ^ { 2 } } - \frac { 1 } { 1 + \gamma ^ { 2 } \varepsilon ^ { 2 } \nu ^ { 2 } } \Big ) \stackrel { \le } \frac { 1 } { 8 } \frac { M ^ { 2 } } { \varepsilon _ { n - 1 } } \Big ( \frac { 1 } { 1 + \gamma ^ { 2 } \varepsilon ^ { 2 } n ^ { 2 } } \Big ) \varepsilon ^ { 2 } \mathrm { d } \lambda } \\ &  \ge \frac { 1 } { 8 } \frac { M ^ { 2 } } { \varepsilon _ { n - 1 } ^ { 2 } } \varepsilon _ { n } ^  \end{array}
$$

Consequently, $\pmb { x } _ { t _ { M } } - \pmb { x } _ { t _ { M } } ^ { \star }$ is a centered Gaussian random vector with second moment $\Omega ( M ^ { - 2 } )$ . Then we have $\mathbb { E } [ \| \pmb { x } _ { t _ { M } } - \pmb { x } _ { t _ { M } } ^ { \star } \| _ { 2 } ] = \overset {  } \Omega ( M ^ { - 1 } )$ and complete the proof.

## D.2 Convergence analysis of SDE-DPMSolver++2

Similar to the analysis in (150), we have

$$
x _ { t _ { i } } - x _ { t _ { i } } ^ { \star } = \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } } ( x _ { t _ { i - 1 } } - x _ { t _ { i - 1 } } ^ { \star } ) + \frac { 1 } { 2 } \delta _ { i } \alpha _ { t _ { i } } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } ) D _ { t _ { i } } - \sqrt { 2 } \alpha _ { t _ { i } } \gamma ^ { 2 } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { \lambda } \frac { 1 - \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i } } - \lambda ) } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } } \mathrm { d } \overline { \omega } _ { t ( \lambda ) } .\tag{151}
$$

We define $D _ { t _ { 1 } } ^ { \star } = { \bf 0 }$ and for $i \geq 2$

$$
\begin{array} { r l r } {  { D _ { t _ { i } } ^ { \star } = \frac { 1 } { \delta _ { i - 1 } } \big ( \mu _ { \theta } ( x _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } ) - \mu _ { \theta } ( x _ { t _ { i - 2 } } ^ { \star } , t _ { i - 2 } ) \big ) } } \\ & { } & { = \frac { 1 } { \delta _ { i - 1 } } \bigg ( \frac { \gamma ^ { 2 } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { i - 1 } } } \frac { x _ { t _ { i - 1 } } ^ { \star } } { \alpha t _ { i - 1 } } - \frac { \gamma ^ { 2 } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { i - 2 } } } \frac { x _ { t _ { i - 2 } } ^ { \star } } { \alpha t _ { i - 2 } } \bigg ) } \\ & { } & { \overset { ( \mathrm { s } ) } { = } \frac { 1 } { \delta _ { i - 1 } } \Bigg [ \frac { \gamma ^ { 2 } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { i - 1 } } } \bigg ( \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { i - 1 } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { i - 2 } } \alpha t _ { i - 2 } } + \sqrt { 2 } ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { i - 1 } } ) \int _ { \lambda _ { i - 2 } } ^ { \lambda _ { i - 1 } } \frac { \mathrm { e } ^ { \lambda } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ) } \bigg ) - \frac { \gamma ^ { 2 } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { i - 2 } } } \frac { x _ { t _ { i - 2 } } ^ { \star } } { \alpha t _ { i - 2 } } \Bigg ] } \\ & { } &  = \frac { \sqrt { 2 } \gamma ^ { 2 } }  \delta _  i -  \end{array}
$$

where (a) applies (148). Substituting into (151) further yields

$$
\begin{array} { l } { { { \displaystyle x _ { t _ { i } } - x _ { t _ { i } } ^ { \star } = \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } } ( x _ { t _ { i - 1 } } - x _ { t _ { i - 1 } } ^ { \star } ) + \frac 1 2 \delta _ { i } \alpha _ { t _ { i } } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } ) D _ { t _ { i } } ^ { \star } } \ ~ } } \\ { { \displaystyle ~ - \sqrt { 2 } \alpha _ { t _ { i } } \gamma ^ { 2 } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { \lambda } \frac { 1 - \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i } } - \lambda ) } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ) } + \frac 1 2 \delta _ { i } \alpha _ { t _ { i } } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } ) ( D _ { t _ { i } } - D _ { t _ { i } } ^ { \star } ) } } \end{array}
$$

$$
\begin{array} { l } { { \displaystyle = \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } } ( { \pmb x } _ { t _ { i - 1 } } - { \pmb x } _ { t _ { i - 1 } } ^ { \star } ) + \frac { \sqrt { 2 } } { 2 } \delta _ { i } \alpha _ { t _ { i } } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } ) \frac { \gamma ^ { 2 } } { \delta _ { i - 1 } } \int _ { \lambda _ { t _ { i - 2 } } } ^ { \lambda _ { t _ { i - 1 } } } \frac { \mathrm { e } ^ { \lambda } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ) } } }  \\ { { \displaystyle \qquad - \sqrt { 2 } \alpha _ { t _ { i } } \gamma ^ { 2 } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { \lambda } \frac { 1 - \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i } } - \lambda ) } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ) } + \frac { 1 } { 2 } \delta _ { i } \alpha _ { t _ { i } } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } ) ( { \pmb D } _ { t _ { i } } - { \pmb D } _ { t _ { i } } ^ { \star } ) . } } \end{array}
$$

By recursion, we obtain

$$
\begin{array} { c } { { \displaystyle \alpha _ { t _ { 1 \wedge t } } - \alpha _ { t _ { 1 \wedge t } } ^ { \star } = \displaystyle \sum _ { m = 2 } ^ { M } \displaystyle \alpha _ { t _ { m } } \displaystyle \sum _ { j } ^ { t _ { 2 } } + \mathrm { e } ^ { - 2 \lambda _ { t _ { 1 \wedge t } } } \displaystyle \sum _ { i = 1 } ^ { m } \delta _ { m } \alpha _ { t _ { m } } \big ( 1 - \mathrm { e } ^ { - 2 \delta _ { m } } \big ) \displaystyle \sum _ { \delta _ { m - 1 } } ^ { \gamma ^ { 2 } } \displaystyle \int _ { \lambda _ { t _ { m } - 3 } } ^ { \lambda _ { t _ { m - 1 } } } \frac { \mathrm { e } ^ { \lambda } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { 1 } } } \displaystyle  { \overline { { \omega } } } _ { t ( \lambda ) } } }  \\ { { \displaystyle \phantom { \alpha _ { m } \alpha _ { m } \alpha _ { m } \alpha _ { m } \gamma ^ { 2 } + \gamma \frac { \lambda _ { t _ { 1 } } } { m - 1 } \alpha _ { t _ { m } } \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { 1 \wedge t } } } \displaystyle \sum _ { \delta _ { t _ { m } - 1 } } ^ { \gamma \lambda _ { t _ { m } } } \gamma ^ { 2 } \displaystyle \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda _ { t _ { m } } } \alpha _ { t _ { m } } ^ { 1 - 1 } - \mathrm { e } ^ { - 2 ( \lambda _ { t _ { m } - 3 } ) } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ) } } } } \\   \displaystyle \phantom  \alpha _ { m } \alpha _ { m } \alpha _ { m } \gamma ^ { 2 } + \gamma \frac { \lambda _ { t _ { 1 } } } { m - 1 } \alpha _ { t _ { m } } \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { 1 \wedge t } } } \delta _ { m } \alpha _ { t _ { m } } \big ( 1 - \mathrm { e } ^ { - 2 \delta _ { m } } \big ) \big  \end{array}
$$

Note that the sum of the first two terms is a centered Gaussian random vector. Denote its covariance matrix by $V ^ { 2 } I _ { d }$ . Then we have

$$
\mathbb { E } [ \Vert \mathbf { x } _ { t _ { M } } - \mathbf { x } _ { t _ { M } } ^ { \star } \Vert _ { 2 } ] \ge \frac { V \sqrt { d } } { \sqrt { 2 } } - \sum _ { m = 1 } ^ { M } \frac { \alpha _ { t _ { M } } } { \alpha _ { t _ { m } } } \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { M } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { m } } } } \frac { 1 } { 2 } \delta _ { m } \alpha _ { t _ { m } } \big ( 1 - \mathrm { e } ^ { - 2 \delta _ { m } } \big ) \mathbb { E } [ \Vert D _ { t _ { m } } - D _ { t _ { m } } ^ { \star } \Vert _ { 2 } ]\tag{154}
$$

The remaining proof is divided into two steps. We first establish that $V ^ { 2 } = \Omega ( M ^ { - 2 } )$ , namely

$$
\begin{array}{c} \begin{array} { c c } { { V ^ { 2 } = 2 \alpha _ { t _ { M } \gamma } ^ { 2 } { } ^ { \smash { \begin{array} { l } { { M - 1 } } \\ { { m _ { 0 } } } \end{array} } } } } & { { \displaystyle \sum _ { m = 1 } ^ { M - 1 } \left( \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { M } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { m } } } } \right) ^ { 2 } \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda _ { t _ { m } } } \frac { \mathrm { e } ^ { 2 \lambda } } { ( 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } ) ^ { 2 } } \left( \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { m } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { m + 1 } } } } \frac { \delta _ { m + 1 } ( 1 - \mathrm { e } ^ { - 2 \delta _ { m + 1 } } ) } { 2 \delta _ { m } } - 1 + \mathrm { e } ^ { - 2 ( \lambda _ { t _ { m } } - \lambda ) } \right) ^ { 2 } \mathrm { d } \lambda } }  \\ { { } } & { { \displaystyle \mathrm { ~ \ ~ \ } + \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda _ { t _ { M } } } \mathrm { e } ^ { 2 \lambda } \left( \frac { 1 - \mathrm { e } ^ { - 2 ( \lambda _ { t _ { M } } - \lambda ) } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } } \right) ^ { 2 } \mathrm { d } \lambda = \Omega ( \delta ^ { 2 } ) } }  & { { \displaystyle ( 1 5 5 ) } } \end{array}   & { { \begin{array} { l } { { \mathrm { ~ } } } \\ { { \mathrm { ~ } } } \end{array} } } \end{array}
$$

and then show that

$$
\sum _ { m = 1 } ^ { M } \frac { \alpha _ { t _ { M } } } { \alpha _ { t _ { m } } } \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { M } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { m } } } } \frac { 1 } { 2 } \delta _ { m } \alpha _ { t _ { m } } \big ( 1 - \mathrm { e } ^ { - 2 \delta _ { m } } \big ) \mathbb { E } [ \| D _ { t _ { m } } - D _ { t _ { m } } ^ { \star } \| _ { 2 } ] = O ( \delta ^ { 1 . 5 } ) .\tag{156}
$$

Collecting them into (154) directly leads to the desired result.

Proof of (155). We begin with the following lower bound:

$$
\begin{array} { r l } & { \bigg ( \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { m } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { m + 1 } } } } \frac { \delta _ { m + 1 } ( 1 - \mathrm { e } ^ { - 2 \delta _ { m + 1 } } ) } { 2 \delta _ { m } } - 1 + \mathrm { e } ^ { - 2 ( \lambda _ { t _ { m } } - \lambda ) } \bigg ) ^ { 2 } } \\ & { \quad \geq \bigg ( \frac { \delta _ { m + 1 } ( 1 - \mathrm { e } ^ { - 2 \delta _ { m + 1 } } ) } { 2 \delta _ { m } } - 1 + \mathrm { e } ^ { - 2 ( \lambda _ { t _ { m } } - \lambda ) } \bigg ) ^ { 2 } } \\ & { \qquad + 2 \Big ( \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { m } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { m + 1 } } } } - 1 \Big ) \frac { \delta _ { m + 1 } \big ( 1 - \mathrm { e } ^ { - 2 \delta _ { m + 1 } } \big ) } { 2 \delta _ { m } } \bigg ( \frac { \delta _ { m + 1 } \big ( 1 - \mathrm { e } ^ { - 2 \delta _ { m + 1 } } \big ) } { 2 \delta _ { m } } - 1 + \mathrm { e } ^ { - 2 ( \lambda _ { t _ { m } } - \lambda ) } \bigg ) } \end{array}\tag{157}
$$

For the second term in the right-hand-side of (157), we have

$$
\begin{array} { r l } & { \Big ( \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t m } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { m + 1 } } } } - 1 \Big ) \frac { \delta _ { m + 1 } \big ( 1 - \mathrm { e } ^ { - 2 \delta _ { m + 1 } } \big ) } { 2 \delta _ { m } } = \frac { \mathrm { e } ^ { - 2 \lambda _ { t _ { m + 1 } } } \big ( \mathrm { e } ^ { 2 \delta _ { m + 1 } } - 1 \big ) } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { m + 1 } } } } \frac { \delta _ { m + 1 } \big ( 1 - \mathrm { e } ^ { - 2 \delta _ { m + 1 } } \big ) } { 2 \delta _ { m } } } \\ & { \qquad \le \frac { 2 \mathrm { e } ^ { - 2 \lambda _ { t _ { m + 1 } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { m + 1 } } } } \frac { \delta _ { m + 1 } ^ { 3 } } { \delta _ { m } } + O ( \delta ^ { 3 } ) , } \\ & { \frac { \delta _ { m + 1 } \big ( 1 - \mathrm { e } ^ { - 2 \delta _ { m + 1 } } \big ) } { 2 \delta _ { m } } - 1 + \mathrm { e } ^ { - 2 ( \lambda _ { t _ { m } } - \lambda ) } = \frac { \delta _ { m + 1 } ^ { 2 } } { \delta _ { m } } - 2 ( \lambda _ { t _ { m } } - \lambda ) + O ( \delta ^ { 2 } ) } \\ & { \qquad \ge - 2 \big ( \lambda _ { t _ { m } } - \lambda \big ) + O ( \delta ^ { 2 } ) . } \end{array}
$$

which leads to the following expression for the second term:

$$
\begin{array} { r l } & { 2 \Bigl ( \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { m } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { m + 1 } } } } - 1 \Bigr ) \frac { \delta _ { m + 1 } \bigl ( 1 - \mathrm { e } ^ { - 2 \delta _ { m + 1 } } \bigr ) } { 2 \delta _ { m } } \biggl ( \frac { \delta _ { m + 1 } \bigl ( 1 - \mathrm { e } ^ { - 2 \delta _ { m + 1 } } \bigr ) } { 2 \delta _ { m } } - 1 + \mathrm { e } ^ { - 2 ( \lambda _ { t _ { m } } - \lambda ) } \biggr ) } \\ & { \quad \geq - \frac { 4 \mathrm { e } ^ { - 2 \lambda _ { t _ { m + 1 } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { m + 1 } } } } \frac { \delta _ { m + 1 } ^ { 3 } } { \delta _ { m } } \biggl ( 2 ( \lambda _ { t _ { m } } - \lambda ) \biggr ) + O ( \delta ^ { 4 } ) } \end{array}
$$

Moreover, substituting it into the integral in (155) yields

$$
\begin{array} { r l r } {  { \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda _ { t _ { m } } } \frac { 2 \mathrm { e } ^ { 2 \lambda } } { ( 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } ) ^ { 2 } } \Big ( \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { m } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { m + 1 } } } } - 1 \Big ) \frac { \delta _ { m + 1 } ( 1 - \mathrm { e } ^ { - 2 \delta _ { m + 1 } } ) } { 2 \delta _ { m } } \Big ( \frac { \delta _ { m + 1 } ( 1 - \mathrm { e } ^ { - 2 \delta _ { m + 1 } } ) } { 2 \delta _ { m } } - 1 + \mathrm { e } ^ { - 2 ( \lambda _ { t _ { m } } - \lambda ) } \Big ) \mathrm { d } \lambda } } \\ & { \geq - \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda _ { t _ { m } } } \frac { 8 \mathrm { e } ^ { 2 \lambda } } { ( 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } ) ^ { 2 } } \frac { \mathrm { e } ^ { - 2 \lambda _ { t _ { m + 1 } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { m + 1 } } } } \frac { \delta _ { m + 1 } ^ { 3 } } { \delta _ { m } } ( \lambda _ { t _ { m } } - \lambda ) \mathrm { d } \lambda + O ( \delta ^ { 5 } ) } \\ &  \geq - \frac { 4 \mathrm { e } ^ { 2 \lambda _ { t _ { m } } } } { ( 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { m - 1 } } } ) ^ { 2 } } \frac { \mathrm { e } ^ { - 2 \lambda _ { t _ { m + 1 } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { m + 1 } } } } \delta _ { m + 1 } ^ { 3 } \delta _ { m } + O ( \delta ^ { 5 } ) = O ( \delta ^ { 4 } )  \end{array}\tag{8}
$$

For the first term in the right-hand-side of (157), we have

$$
\begin{array} { r l } & { \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda _ { t _ { m } } } \frac { \mathrm { e } ^ { 2 \lambda } } { ( 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } ) ^ { 2 } } \bigg ( \frac { \delta _ { m + 1 } ( 1 - \mathrm { e } ^ { - 2 \delta _ { m + 1 } } ) } { 2 \delta _ { m } } - 1 + \mathrm { e } ^ { - 2 ( \lambda _ { t _ { m } } - \lambda ) } \bigg ) ^ { 2 } \mathrm { d } \lambda } \\ & { \quad \geq \displaystyle \int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda _ { t _ { m } } } \frac { \mathrm { e } ^ { 2 \lambda _ { t _ { m - 1 } } } } { ( 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { m } } } ) ^ { 2 } } \bigg ( \frac { \delta _ { m + 1 } ( 1 - \mathrm { e } ^ { - 2 \delta _ { m + 1 } } ) } { 2 \delta _ { m } } - 1 + \mathrm { e } ^ { - 2 ( \lambda _ { t _ { m } } - \lambda ) } \bigg ) ^ { 2 } \mathrm { d } \lambda . } \end{array}\tag{159}
$$

By direct calculus, we obtain

$$
\begin{array} { r l } & { \int _ { \lambda _ { \mathrm { r } _ { m - 1 } } } ^ { \lambda _ { \mathrm { f } _ { m } } } \left( \frac { \delta _ { m + 1 } \left( 1 - \mathrm { e } ^ { - 2 \delta _ { m + 1 } } \right) } { 2 \delta _ { m } } - 1 + \mathrm { e } ^ { - 2 ( \lambda _ { \mathrm { f } _ { m } } - \lambda ) } \right) ^ { 2 } \mathrm { d } \lambda } \\ & { \quad = \int _ { \lambda _ { \mathrm { t } _ { m - 1 } } } ^ { \lambda _ { \mathrm { t } _ { m } } } \mathrm { e } ^ { - 4 ( \lambda _ { \mathrm { f } _ { m } } - \lambda ) } \mathrm { d } \lambda + 2 \left( \frac { \delta _ { m + 1 } \left( 1 - \mathrm { e } ^ { - 2 \delta _ { m + 1 } } \right) } { 2 \delta _ { m } } - 1 \right) \int _ { \lambda _ { \mathrm { t } _ { m - 1 } } } ^ { \lambda _ { \mathrm { f } _ { m } } } \mathrm { e } ^ { - 2 ( \lambda _ { \mathrm { f } _ { m } } - \lambda ) } \mathrm { d } \lambda + \left( \frac { \delta _ { m + 1 } \left( 1 - \mathrm { e } ^ { - 2 \delta _ { m + 1 } } \right) } { 2 \delta _ { m } } - 1 \right) ^ { 2 } \delta _ { m } } \\ & { \quad = \frac { 1 } { 4 } ( 1 - \mathrm { e } ^ { - 4 \delta _ { m } } ) + \left( \frac { \delta _ { m + 1 } \left( 1 - \mathrm { e } ^ { - 2 \delta _ { m + 1 } } \right) } { 2 \delta _ { m } } - 1 \right) ( 1 - \mathrm { e } ^ { - 2 \delta _ { m } } ) + \left( \frac { \delta _ { m + 1 } \left( 1 - \mathrm { e } ^ { - 2 \delta _ { m + 1 } } \right) } { 2 \delta _ { m } } - 1 \right) ^ { 2 } \delta _ { m } . } \end{array}
$$

Taylor expansion gives that

$$
\begin{array} { l } { { \displaystyle 1 - \mathrm { e } ^ { - 4 \delta _ { m } } = 4 \delta _ { m } - 8 \delta _ { m } ^ { 2 } + \frac { 3 2 } { 3 } \delta _ { m } ^ { 3 } + { \cal O } ( \delta _ { m } ^ { 4 } ) , } } \\ { { \displaystyle 1 - \mathrm { e } ^ { - 2 \delta _ { m } } = 2 \delta _ { m } - 2 \delta _ { m } ^ { 2 } + \frac { 4 } { 3 } \delta _ { m } ^ { 3 } + { \cal O } ( \delta _ { m } ^ { 4 } ) . } } \end{array}
$$

Substituting into the right-hand-side of (160) yields

$$
\int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda _ { t _ { m } } } { \biggl ( } \frac { \delta _ { m + 1 } ( 1 - \mathrm { e } ^ { - 2 \delta _ { m + 1 } } ) } { 2 \delta _ { m } } - 1 + \mathrm { e } ^ { - 2 ( \lambda _ { t _ { m } } - \lambda ) } { \biggr ) } ^ { 2 } \mathrm { d } \lambda
$$

$$
\begin{array} { l } { { = \delta _ { m } - 2 \delta _ { m } ^ { 2 } + \frac 8 3 \delta _ { m } ^ { 3 } + \left( \frac { \delta _ { m + 1 } } { \delta _ { m } } ( \delta _ { m + 1 } - \delta _ { m + 1 } ^ { 2 } + \frac 2 3 \delta _ { m + 1 } ^ { 3 } + O ( \delta _ { m + 1 } ^ { 4 } ) ) - 1 \right) \left( 2 \delta _ { m } - 2 \delta _ { m } ^ { 2 } + \frac 4 3 \delta _ { m } ^ { 3 } + O ( \delta _ { m } ^ { 4 } ) \right) } } \\ { { \mathrm { ~ } + \left( \frac { \delta _ { m + 1 } } { \delta _ { m } } ( \delta _ { m + 1 } - \delta _ { m + 1 } ^ { 2 } + \frac 2 3 \delta _ { m + 1 } ^ { 3 } + O ( \delta _ { m + 1 } ^ { 4 } ) ) - 1 \right) ^ { 2 } \delta _ { m } + O ( \delta ^ { 4 } ) } } \\ { { \mathrm { ~ } = \frac 4 3 \delta _ { m } ^ { 3 } - 2 \delta _ { m } \delta _ { m + 1 } ^ { 2 } + \frac { \delta _ { m + 1 } ^ { 4 } } { \delta _ { m } } + O ( \delta ^ { 4 } ) = \frac 1 3 \delta _ { m } ^ { 3 } + \frac { \left( \delta _ { m + 1 } ^ { 2 } - \delta _ { m } ^ { 2 } \right) ^ { 2 } } { \delta _ { m } } + O ( \delta ^ { 4 } ) \geq \frac 1 3 \delta _ { m } ^ { 3 } + O ( \delta ^ { 4 } ) . } } \end{array}
$$

Combining with (159) yields

$$
\int _ { \lambda _ { t _ { m - 1 } } } ^ { \lambda _ { t _ { m } } } \frac { \mathrm { e } ^ { 2 \lambda } } { ( 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } ) ^ { 2 } } \bigg ( \frac { \delta _ { m + 1 } \big ( 1 - \mathrm { e } ^ { - 2 \delta _ { m + 1 } } \big ) } { 2 \delta _ { m } } - 1 + \mathrm { e } ^ { - 2 ( \lambda _ { t _ { m } } - \lambda ) } \bigg ) ^ { 2 } \mathrm { d } \lambda \geq \frac { \mathrm { e } ^ { 2 \lambda _ { t _ { m - 1 } } } } { \big ( 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { m } } } \big ) ^ { 2 } } \bigg ( \frac { 1 } { 3 } \delta _ { m } ^ { 3 } + O ( \delta ^ { 4 } ) \bigg ) .\tag{161}
$$

Substituting into (155) yields

$$
\begin{array} { r l r } {  { V ^ { 2 } \geq 2 \alpha _ { t _ { M } } ^ { 2 } \gamma ^ { 4 } \sum _ { m = 1 } ^ { M - 1 } ( \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { M } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { m } } } } ) ^ { 2 } \frac { \mathrm { e } ^ { 2 \lambda _ { t _ { m - 1 } } } } { ( 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { m } } } ) ^ { 2 } } \Big ( \frac { 1 } { 3 } \delta _ { m } ^ { 3 } + O ( \delta ^ { 4 } ) \Big ) } } \\ & { } & { \geq 2 \alpha _ { t _ { M } } ^ { 2 } \gamma ^ { 4 } \bigg ( \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { M } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { 1 } } } } \bigg ) ^ { 2 } \frac { \mathrm { e } ^ { - 2 \lambda _ { t _ { M } } - 2 \delta _ { M } } } { ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { M } } } ) ^ { 2 } } \sum _ { m = 1 } ^ { M - 1 } \Big ( \frac { 1 } { 3 } \delta _ { m } ^ { 3 } + O ( \delta ^ { 4 } ) \Big ) \gtrsim \delta ^ { 2 } . } \end{array}
$$

where the last inequality holds for suficiently large M, since

$$
\sum _ { m = 1 } ^ { M - 1 } \delta _ { m } ^ { 3 } \geq \frac { ( \lambda _ { t _ { M } } - \lambda _ { t _ { 0 } } - \delta _ { M } ) ^ { 3 } } { ( M - 1 ) ^ { 2 } } = \Omega ( M ^ { - 2 } ) = \Omega ( \delta ^ { 2 } ) .
$$

This completes the proof of (155).

Proof of (156). By the expression for data predictor in (141) and the definition of $D _ { t _ { i } } ^ { \star }$ in (152), we have

$$
D _ { t _ { i } } - D _ { t _ { i } } ^ { \star } = \frac { 1 } { \delta _ { i - 1 } } \Bigg ( \frac { \gamma ^ { 2 } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } } \frac { x _ { t _ { i - 1 } } - x _ { t _ { i - 1 } } ^ { \star } } { \alpha _ { t _ { i - 1 } } } - \frac { \gamma ^ { 2 } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 2 } } } } \frac { x _ { t _ { i - 2 } } - x _ { t _ { i - 2 } } ^ { \star } } { \alpha _ { t _ { i - 2 } } } \Bigg ) .\tag{162}
$$

While by (151), the above diference is calculated as

$$
\begin{array} { r l } & { \frac { \gamma ^ { 2 } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } } \frac { \mathbf { x } _ { t _ { i - 1 } } - \mathbf { x } _ { t _ { i - 1 } } ^ { \star } } { \alpha _ { t _ { i - 1 } } } - \frac { \gamma ^ { 2 } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 2 } } } } \frac { \mathbf { x } _ { t _ { i - 2 } } - \mathbf { x } _ { t _ { i - 2 } } ^ { \star } } { \alpha _ { t _ { i - 2 } } } } \\ & { \quad = \frac { \gamma ^ { 2 } \delta _ { i - 1 } \left( 1 - \mathrm { e } ^ { - 2 \delta _ { i - 1 } } \right) } { 2 \left( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } \right) } D _ { t _ { i - 1 } } - \frac { \sqrt { 2 } \gamma ^ { 4 } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } } \displaystyle \int _ { \lambda _ { t _ { i - 2 } } } ^ { \lambda _ { t _ { i - 1 } } } \mathrm { e } ^ { \lambda } \frac { 1 - \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i - 1 } } - \lambda ) } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ) } . } \end{array}
$$

Substituting it into (162) further yields

$$
\begin{array} { r l } & { D _ { t _ { i } } - D _ { t _ { i } } ^ { \star } = \frac { \gamma ^ { 2 } \left( 1 - \mathrm { e } ^ { - 2 \delta _ { i - 1 } } \right) } { 2 \left( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \delta _ { t _ { i - 1 } } } \right) } D _ { t _ { i - 1 } } - \frac { \sqrt { 2 } \gamma ^ { 4 } } { \delta _ { i - 1 } \left( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } \right) } \int _ { \lambda _ { t _ { i - 2 } } } ^ { \lambda _ { t _ { i - 1 } } } \mathrm { e } ^ { \lambda } \frac { 1 - \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i - 1 } } - \lambda ) } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ) } } \\ & { \qquad = \frac { \gamma ^ { 2 } \left( 1 - \mathrm { e } ^ { - 2 \delta _ { i - 1 } } \right) } { 2 \left( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } \right) } ( D _ { t _ { i - 1 } } ^ { \star } + D _ { t _ { i - 1 } } - D _ { t _ { i - 1 } } ^ { \star } ) - \frac { \sqrt { 2 } \gamma ^ { 4 } } { \delta _ { i - 1 } \left( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } \right) } \int _ { \lambda _ { t _ { i - 2 } } } ^ { \lambda _ { t _ { i - 1 } } } \mathrm { e } ^ { \lambda } \frac { 1 - \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i - 1 } } - \lambda ) } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ) } . } \end{array}\tag{163}
$$

For the last term, we have

$$
\begin{array} { r l r } & { } & { \mathbb { E } \bigg [ \bigg \| \int _ { \lambda _ { t _ { i - 2 } } } ^ { \lambda _ { t _ { i - 1 } } } \mathrm { e } ^ { \lambda } \frac { 1 - \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i - 1 } } - \lambda ) } } { 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } } \mathrm { d } \overline { { \omega } } _ { t ( \lambda ) } \bigg \| _ { 2 } ^ { 2 } \bigg ] = \int _ { \lambda _ { t _ { i - 2 } } } ^ { \lambda _ { t _ { i - 1 } } } \mathrm { e } ^ { 2 \lambda } \frac { \big ( 1 - \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i - 1 } } - \lambda ) } \big ) ^ { 2 } } { ( 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } ) ^ { 2 } } \mathrm { d } \lambda } \\ & { } & { \leq \int _ { \lambda _ { t _ { i - 2 } } } ^ { \lambda _ { t _ { i - 1 } } } \mathrm { e } ^ { 2 \lambda _ { t _ { i - 1 } } } \frac { 4 ( \lambda _ { t _ { i - 1 } } - \lambda ) ^ { 2 } } { ( 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { i - 2 } } } ) ^ { 2 } } \mathrm { d } \lambda } \end{array}
$$

$$
\begin{array} { r l } & { = \frac { 4 \delta _ { i - 1 } ^ { 3 } \mathrm { e } ^ { 2 \lambda _ { t _ { i - 1 } } } } { 3 ( 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda _ { t _ { i - 2 } } } ) ^ { 2 } } = \frac { 4 \delta _ { i - 1 } ^ { 3 } \mathrm { e } ^ { 2 \lambda _ { t _ { i - 1 } } - 4 \lambda _ { t _ { i - 2 } } } } { 3 ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 2 } } } ) ^ { 2 } } } \\ & { \lesssim \frac { \delta _ { i - 1 } ^ { 3 } \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 2 } } } } { ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 2 } } } ) ^ { 2 } } . } \end{array}
$$

Note that the factor before $D _ { t _ { i - 1 } } - D _ { t _ { i } } ^ { \star }$ in (163) satisfies

$$
\frac { \gamma ^ { 2 } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i - 1 } } ) } { 2 ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } ) } \leq \frac { ( 1 - \mathrm { e } ^ { - 2 \delta _ { i - 1 } } ) } { 2 } \leq \delta _ { i - 1 } .
$$

Therefore we have

$$
\mathbb { E } [ \| D _ { t _ { i } } - D _ { t _ { i } } ^ { \star } \| _ { 2 } ] \le \delta _ { i - 1 } \mathbb { E } [ \| D _ { t _ { i - 1 } } - D _ { t _ { i - 1 } } ^ { \star } \| _ { 2 } ] + \delta _ { i - 1 } \mathbb { E } [ \| D _ { t _ { i - 1 } } ^ { \star } \| ] + O \Big ( \frac { \gamma ^ { 4 } \delta _ { i - 1 } ^ { 0 . 5 } \mathrm { e } ^ { - \lambda _ { t _ { i - 2 } } } } { ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 1 } } } ) ^ { 2 } } \Big )\tag{164}
$$

By (153), we have

$$
\begin{array} { r l r } & { } & { { \left( { \mathbb { E } } [ \| D _ { t _ { i - 1 } } ^ { \star } \| ] \right) } ^ { 2 } \leq { \mathbb { E } } [ \| D _ { t _ { i - 1 } } ^ { \star } \| ^ { 2 } ] = \frac { 2 \gamma ^ { 4 } } { \delta _ { i - 2 } ^ { 2 } } \int _ { \lambda _ { t _ { i - 3 } } } ^ { \lambda _ { t _ { i - 2 } } } \frac { \mathrm { e } ^ { 2 \lambda } } { ( 1 + \gamma ^ { 2 } \mathrm { e } ^ { 2 \lambda } ) ^ { 2 } } \mathrm { d } \lambda } \\ & { } & { \leq \frac { 2 \gamma ^ { 4 } \delta _ { i - 2 } \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 3 } } } } { \delta _ { i - 2 } ^ { 2 } ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 2 } } } ) ^ { 2 } } = \frac { 2 \gamma ^ { 4 } \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 3 } } } } { \delta _ { i - 2 } ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { i - 2 } } } ) ^ { 2 } } . } \end{array}\tag{165}
$$

Substituting into (164) yields

$$
\mathbb { E } [ \| D _ { t _ { i } } - D _ { t _ { i } } ^ { \star } \| _ { 2 } ] \lesssim O ( \sqrt { \delta } ) .
$$

Substituting into the left-hand-side of (156) yields

$$
\begin{array} { r l } & { \displaystyle \sum _ { m = 1 } ^ { M } \frac { \alpha _ { t _ { M } } } { \alpha _ { t _ { m } } } \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { M } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { m } } } } \frac { 1 } { 2 } \delta _ { m } \alpha _ { t _ { m } } \bigl ( 1 - \mathrm { e } ^ { - 2 \delta _ { m } } \bigr ) \mathbb { E } \bigl [ \| D _ { t _ { m } } - D _ { t _ { m } } ^ { \star } \| _ { 2 } \bigr ] } \\ & { \quad \le \displaystyle \sum _ { m = 1 } ^ { M } \frac { \alpha _ { t _ { M } } } { \alpha _ { t _ { m } } } \frac { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { M } } } } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { m } } } } \frac { 1 } { 2 } \delta _ { m } \alpha _ { t _ { m } } \bigl ( 1 - \mathrm { e } ^ { - 2 \delta _ { m } } \bigr ) O ( \sqrt { \delta } ) } \\ & { \quad \le O ( \sqrt { \delta } ) \displaystyle \sum _ { m = 1 } ^ { M } \frac { \alpha _ { t _ { M } } \bigl ( \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { M } } } \bigr ) } { \gamma ^ { 2 } + \mathrm { e } ^ { - 2 \lambda _ { t _ { m } } } } \delta _ { m } ^ { 2 } = O ( \delta ^ { 1 . 5 } ) . } \end{array}\tag{166}
$$

This completes the proof.

## E Proof of convergence order upper bound for F-DDIM (Theorem 4)

We write $\delta = \operatorname* { m a x } _ { i \in [ M ] } \delta _ { i } = O ( M ^ { - 1 } )$ . To establish Theorem 4, we first consider the convergence order of the following (idealized) first-order forward-value discretization of the difusion ODE $\mathrm { ( 1 2 a ) }$ . For each iteration $i = 1 , \ldots , M$ , given the current iterate $\pmb { x } _ { t _ { i - 1 } } ^ { \mathsf { f o r } }$ at time $t _ { i - 1 }$ , we compute the next iterate $\cdot { \boldsymbol { x } } _ { t _ { i } } ^ { \mathsf { f o r } }$ by

$$
\pmb { x } _ { t _ { i } } ^ { \mathrm { { f o r } } } = \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \pmb { x } _ { t _ { i - 1 } } ^ { \mathrm { { f o r } } } + \Big ( \alpha _ { t _ { i } } - \frac { \sigma _ { t _ { i } } \alpha _ { t _ { i - 1 } } } { \sigma _ { t _ { i - 1 } } } \Big ) \pmb { \mu } _ { \theta } ( \pmb { x } _ { t _ { i } } ^ { \mathrm { { f o r } } } , t _ { i } ) .\tag{167}
$$

For comparison, the deterministic DDIM update (13) can be written in the data-prediction form as

$$
\pmb { x } _ { t _ { i } } ^ { \mathrm { b c k } } = \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \pmb { x } _ { t _ { i - 1 } } ^ { \mathrm { b c k } } + \bigg ( \alpha _ { t _ { i } } - \frac { \sigma _ { t _ { i } } \alpha _ { t _ { i - 1 } } } { \sigma _ { t _ { i - 1 } } } \bigg ) \pmb { \mu } _ { \theta } ( \pmb { x } _ { t _ { i - 1 } } ^ { \mathrm { b c k } } , t _ { i - 1 } ) .\tag{168}
$$

The only diference between (167) and (168) is whether the data prediction model $\pmb { \mu } _ { \boldsymbol { \theta } } ( \pmb { x } , t )$ is evaluated at the next state/time $( \pmb { x } _ { t _ { i } } ^ { \mathsf { f o r } } , t _ { i } )$ , or the current state/time $( \pmb { x } _ { t _ { i - 1 } } ^ { \mathrm { b c k } } , t _ { i - 1 } )$ . As a note, while the update rule of the naive forward-value discretization (167) is not directly implementable, it serves as a clean lens for understanding how evaluation placement afects discretization error.

Interestingly, it turns out that the first-order discretization errors introduced by these two strategies have exactly opposite signs, as formalized in the following theorem. The proof is postponed to Section E.1.

Theorem 6. Under Assumption 1, the idealized forward-value discretization scheme (167) has convergence order 1. Moreover, let $\pmb { x } _ { t _ { M } } ^ { \mathsf { f o r } }$ and $\boldsymbol { x } _ { t _ { M } } ^ { \mathrm { b c k } }$ denote the outputs of the forward-value discretization (167) and DDIM (168), respectively, and let $\boldsymbol { x } _ { t _ { M } } ^ { \star }$ be the exact ODE solution defined in (28), with the common initialization $\pmb { x } _ { t _ { 0 } } ^ { \mathrm { b c k } } = \pmb { x } _ { t _ { 0 } } ^ { \mathrm { f o r } } = \pmb { x } _ { t _ { 0 } } ^ { \star }$ . Then we have

$$
\begin{array} { r } { \left\| \boldsymbol { x } _ { t _ { M } } ^ { \mathrm { b c k } } - \boldsymbol { x } _ { t _ { M } } ^ { \star } + \boldsymbol { x } _ { t _ { M } } ^ { \mathrm { f o r } } - \boldsymbol { x } _ { t _ { M } } ^ { \star } \right\| _ { 2 } = O ( 1 / M ^ { 2 } ) . } \end{array}\tag{169}
$$

Theorem 6 implies that the leading $O ( 1 / M )$ errors of the forward-value and backward-value discretizations cancel to second order:

$$
\pmb { x } _ { t _ { M } } ^ { \mathrm { b c k } } - \pmb { x } _ { t _ { M } } ^ { \star } \approx - ( \pmb { x } _ { t _ { M } } ^ { \mathrm { f o r } } - \pmb { x } _ { t _ { M } } ^ { \star } ) .
$$

Conceptually, this suggests that if one can (even approximately) access forward-value information, then combining it with the standard backward-value update can greatly reduce the most harmful part of the discretization error.

Next, we provide the convergence guarantee of the proposed sampler in Theorem 7 below; with proof postponed to Section E.2.

Theorem 7. Suppose that Assumption 1 holds. In addition, assume that the one-step lookahead estimate $\widehat { \mathbf { x } } _ { t _ { i } }$ satisfies

$$
\| \widehat { \boldsymbol { x } } _ { t _ { i } } - \boldsymbol { x } _ { t _ { i } } ^ { \star } ( \boldsymbol { x } _ { t _ { i - 1 } } , t _ { i - 1 } ) \| = o ( 1 / M ) , \quad i \geq 2 , \qquad a n d \quad \| \widehat { \boldsymbol { x } } _ { t _ { 1 } } - \boldsymbol { x } _ { t _ { 1 } } ^ { \star } ( \boldsymbol { x } _ { t _ { 0 } } , t _ { 0 } ) \| = O ( 1 / M ) ,\tag{170}
$$

where ${ \pmb x } _ { t } ^ { \star } ( { \pmb x } _ { t _ { i - 1 } } , t _ { i - 1 } )$ denotes the solution to the difusion ODE at time $t \leq t _ { i - 1 }$ when initialized at time $t _ { i - 1 }$ with state $\mathbf { { x } } _ { t _ { i - 1 } } , \mathbf { { \ } } i . \mathbf { { e } } .$

$$
\boldsymbol { x } _ { t } ^ { \star } ( \boldsymbol { x } _ { t _ { i - 1 } } , t _ { i - 1 } ) = \frac { \alpha _ { t } } { \alpha _ { t _ { i - 1 } } } \boldsymbol { x } _ { t _ { i - 1 } } - \alpha _ { t } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t } } \mathrm { e } ^ { - \lambda } \varepsilon _ { \theta } \big ( \boldsymbol { x } _ { t ( \lambda ) } ^ { \star } ( \boldsymbol { x } _ { t _ { i - 1 } } , t _ { i - 1 } ) , t ( \lambda ) \big ) \mathrm { d } \lambda , \quad \forall t \leq t _ { i - 1 } .\tag{171}
$$

Then the proposed sampler (19) satisfies

$$
\begin{array} { r } { \left\| \pmb { x } _ { t _ { M } } - \pmb { x } _ { t _ { M } } ^ { \mathrm { f o r } } \right\| = o ( 1 / M ) , } \end{array}
$$

where $\pmb { x } _ { t } ^ { \mathsf { f o r } }$ represents the idealized forward-value iterate defined in (167) with the same initialization as $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ . In particular, the convergence order of the sampler (19) is 1. For F-DDIM in Algorithm 1, (170) holds uniformly $f o r i \geq 2 ,$ ; the initialization $\widetilde { \varepsilon } _ { t _ { 0 } } = \mathbf { { x } } _ { t _ { 0 } }$ gives only an $O ( 1 / M )$ lookahead error at i = 1. Nevertheless,

$$
\operatorname* { m a x } _ { i \in [ M ] } \left\| { \pmb x } _ { t _ { i } } - { \pmb x } _ { t _ { i } } ^ { \mathrm { f o r } } \right\| = O ( 1 / M ^ { 2 } ) ,
$$

so F-DDIM also has convergence order 1.

This theorem guarantees that lookahead errors satisfying (170) are suficiently small for the implemented method to inherit the first-order rate of the idealized forward-value scheme. Theorem 6 together with Theorem 7 directly yields Theorem 4. For the remainder of this section, we focus on the proofs of Theorems 6 and 7.

For ease of presentation, we use $\nabla ^ { k } \mu ( \pmb { x } , t )$ to denote the k-th order gradient of $\pmb { \mu } ( \pmb { x } , t )$ with respect to x. Before diving into the proof details, we note that the difusion ODE in (12a) can be equivalently written in terms of the data prediction model $\mu _ { \theta }$ . Indeed, (12a) comes from the ODE (4). Using the approximation $\pmb { s } _ { t } ^ { \star } ( \pmb { x } , t ) \approx \bigl ( \alpha _ { t } \pmb { \mu } _ { \theta } ( \pmb { x } , t ) - \pmb { x } \bigr ) / \sigma _ { t } ^ { 2 }$ and substituting it into (4) yields the following difusion ODE associated with a data prediction model:

$$
\dot { { \pmb x } } _ { t } = \left( f ( t ) + \frac { g ^ { 2 } ( t ) } { 2 \sigma _ { t } ^ { 2 } } \right) { \pmb x } _ { t } - \frac { \alpha _ { t } g ^ { 2 } ( t ) } { 2 \sigma _ { t } ^ { 2 } } { \pmb \mu } _ { \theta } ( { \pmb x } _ { t } , t ) , \quad { \pmb x } _ { T } \sim \mathcal { N } ( 0 , \tilde { \sigma } ^ { 2 } { \pmb I } _ { d } ) .\tag{172}
$$

Solving this ODE from $T$ to 0 again produces a sample $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ . Under the same change of variables $\lambda ( t )$ , the solution admits the integral form: for $t < s ,$

$$
\pmb { x } _ { t } = \frac { \sigma _ { t } } { \sigma _ { s } } \pmb { x } _ { s } + \sigma _ { t } \int _ { \lambda ( s ) } ^ { \lambda ( t ) } \mathrm { e } ^ { \lambda } \pmb { \mu } _ { \theta } \big ( \pmb { x } _ { t ( \lambda ) } , t ( \lambda ) \big ) \mathrm { d } \lambda .\tag{173}
$$

Finally, we introduce the total derivative of a function $\mu _ { \theta } ( { \pmb x } _ { t ( \lambda ) } , t ( \lambda ) )$ with respect to log-SNR λ along a trajectory $\pmb { x } _ { t ( \lambda ) }$ as

$$
\frac { \mathrm { d } } { \mathrm { d } \lambda } \pmb { \mu } _ { \theta } \big ( \pmb { x } _ { t ( \lambda ) } , t ( \lambda ) \big ) = \nabla _ { \pmb { x } } \pmb { \mu } _ { \theta } \big ( \pmb { x } _ { t ( \lambda ) } , t ( \lambda ) \big ) \frac { \mathrm { d } \pmb { x } _ { t ( \lambda ) } } { \mathrm { d } \lambda } + \frac { \partial \pmb { \mu } _ { \theta } } { \partial t } \big ( \pmb { x } _ { t ( \lambda ) } , t ( \lambda ) \big ) \frac { \mathrm { d } t ( \lambda ) } { \mathrm { d } \lambda } .\tag{174}
$$

On a fixed finite log-SNR interval $I = [ \lambda _ { t _ { 0 } } , \lambda _ { t _ { M } } ] ;$ , by the integral form (173) and the Assumption 1 (A1), there exists $R < \infty$ such that

$$
\operatorname* { m a x } _ { \lambda \in I } \| \pmb { x } _ { t ( \lambda ) } \| \le R .
$$

Let

$$
\alpha _ { \mathrm { m i n } } = \operatorname* { m i n } _ { \lambda \in I } \alpha _ { \lambda } , \qquad \mathrm { a n d } \quad \alpha _ { \mathrm { m a x } } = \operatorname* { m a x } _ { \lambda \in I } \alpha _ { \lambda }
$$

The derivative of $\mu _ { \theta } ( \pmb { x } _ { t ( \lambda ) } , t ( \lambda ) )$ with respect to $\lambda$ is bounded by

$$
\begin{array} { r l } { \displaystyle \left\| \frac { \mathrm { d } } { \mathrm { d } \lambda } \mu _ { \theta } \big ( x _ { t ( \lambda ) } , t ( \lambda ) \big ) \right\| \leq \left\| \nabla _ { x } \mu _ { \theta } \big ( x _ { t ( \lambda ) } , t ( \lambda ) \big ) \right\| \displaystyle \left\| \frac { \mathrm { d } x _ { t ( \lambda ) } } { \mathrm { d } \lambda } \right\| + \displaystyle \left\| \frac { \partial \mu _ { \theta } } { \partial t } \big ( x _ { t ( \lambda ) } , t ( \lambda ) \big ) \frac { \mathrm { d } t ( \lambda ) } { \mathrm { d } \lambda } \right\| } & { } \\ { \displaystyle \leq \frac { L _ { x } } { \alpha _ { t ( \lambda ) } } \Big \| \frac { \mathrm { d } \log \sigma _ { \lambda } } { \mathrm { d } \lambda } \pmb { x } _ { t ( \lambda ) } + \alpha _ { \lambda } \mu _ { \theta } \big ( \pmb { x } _ { t ( \lambda ) } , t ( \lambda ) \big ) \Big \| + L _ { t } } & { } \\ { \displaystyle \leq \frac { L _ { x } } { \alpha _ { \operatorname* { m i n } } } \underbrace { \bigg ( R \operatorname* { m a x } _ { \lambda \in I } \Big | \frac { \mathrm { d } \log \sigma _ { \lambda } } { \mathrm { d } \lambda } \Big | + \alpha _ { \operatorname* { m a x } } L _ { \mu } \bigg ) } _ { = : V } + L _ { t } = : \widetilde L _ { t } , } \end{array}\tag{175}
$$

where boundedness follows from , Assumption 1 (A1), and the fixed finite interval. We have also used

$$
\begin{array} { r l r } {  { \frac { \mathrm { d } \pmb { x } _ { t ( \lambda ) } } { \mathrm { d } \lambda } = \sigma _ { \lambda } \displaystyle \frac { \mathrm { d } } { \mathrm { d } \lambda } \frac { \pmb { x } _ { t ( \lambda ) } } { \sigma _ { \lambda } } + \frac { \pmb { x } _ { t ( \lambda ) } } { \sigma _ { \lambda } } \displaystyle \frac { \mathrm { d } \sigma _ { \lambda } } { \mathrm { d } \lambda } } } \\ & { } & { = \alpha _ { \lambda } \pmb { \mu } _ { \theta } \big ( \pmb { x } _ { t ( \lambda ) } , t ( \lambda ) \big ) + \frac { \mathrm { d } \log \sigma _ { \lambda } } { \mathrm { d } \lambda } \pmb { x } _ { t ( \lambda ) } . } \end{array}\tag{176}
$$

Similarly, the second derivative is bounded by

$$
\begin{array} { r l } { \Big \| \frac { \mathrm { d } ^ { 2 } } { \mathrm { d } \lambda ^ { 2 } } \mu _ { \theta } \big ( \mathbf { x } _ { t ( \lambda ) } , t ( \lambda ) \big ) \Big \| \leq \Big \| \nabla _ { x } ^ { 2 } \mu _ { \theta } \big ( \mathbf { x } _ { t ( \lambda ) } , t ( \lambda ) \big ) \Big \| \Big \| \frac { \mathrm { d } \mathbf { x } _ { t ( \lambda ) } } { \mathrm { d } \lambda } \Big \| ^ { 2 } + 2 \Big \| \frac { \partial } { \partial \lambda } \nabla _ { x } \mu _ { \theta } \big ( \mathbf { x } , t ( \lambda ) \big ) \big | _ { x = \mathbf { x } _ { t ( \lambda ) } } \frac { \mathrm { d } \mathbf { x } _ { t ( \lambda ) } } { \mathrm { d } \lambda } \Big \| } & { } \\ & { \qquad + \Big \| \nabla _ { x } \mu _ { \theta } \big ( \mathbf { x } _ { t ( \lambda ) } , t ( \lambda ) \big ) \Big \| \Big \| \frac { \mathrm { d } ^ { 2 } \mathbf { x } _ { t ( \lambda ) } } { \mathrm { d } \lambda ^ { 2 } } \Big \| + \Big \| \frac { \partial ^ { 2 } \mu _ { \theta } } { \partial \lambda ^ { 2 } } \big ( \mathbf { x } , t ( \lambda ) \big ) \big | _ { x = \mathbf { x } _ { t ( \lambda ) } } \Big \| } \\ & { \leq H _ { x } \Big \| \frac { \mathrm { d } \mathbf { x } _ { t ( \lambda ) } } { \mathrm { d } \lambda } \Big \| ^ { 2 } + 2 H \Big \| \frac { \mathrm { d } \mathbf { x } _ { t ( \lambda ) } } { \mathrm { d } \lambda } \Big \| + \frac { L _ { x } } { \alpha _ { t ( \lambda ) } } \Big \| \frac { \mathrm { d } ^ { 2 } x _ { t ( \lambda ) } } { \mathrm { d } \lambda ^ { 2 } } \Big \| + H _ { t } } \\ & { \leq H _ { x } V ^ { 2 } + 2 H V } \\ &  \qquad + \frac { L _ { x } } { \alpha _ { \operatorname* { m i n } } } \Big ( R \frac { \mathrm { m a x } }  \ \end{array}
$$

where we have used (176) and the fact that

$$
\begin{array} { r l } & { \frac { \mathrm { d } ^ { 2 } x _ { t ( \lambda ) } } { \mathrm { d } \lambda ^ { 2 } } = \frac { \mathrm { d } } { \mathrm { d } \lambda } \Big ( \frac { \mathrm { d } \log \sigma _ { \lambda } } { \mathrm { d } \lambda } x _ { t ( \lambda ) } + \alpha _ { \lambda } \mu _ { \theta } \big ( x _ { t ( \lambda ) } , t ( \lambda ) \big ) \Big ) } \\ & { \quad \quad \quad = \frac { \mathrm { d } ^ { 2 } \log \sigma _ { \lambda } } { \mathrm { d } \lambda ^ { 2 } } x _ { t ( \lambda ) } + \frac { \mathrm { d } \log \sigma _ { \lambda } } { \mathrm { d } \lambda } \frac { \mathrm { d } x _ { t ( \lambda ) } } { \mathrm { d } \lambda } + \frac { \mathrm { d } \alpha _ { \lambda } } { \mathrm { d } \lambda } \mu _ { \theta } \big ( x _ { t ( \lambda ) } , t ( \lambda ) \big ) + \alpha _ { \lambda } \frac { \mathrm { d } } { \mathrm { d } \lambda } \mu _ { \theta } \big ( x _ { t ( \lambda ) } , t ( \lambda ) \big ) . } \end{array}
$$

## E.1 Proof of Theorem 6

Proof for the first-order forward-value discretization scheme. By the difusion ODE with the data prediction model (see (173)), for each $i \in [ M ]$ , we can express $\boldsymbol { \mathbf { \mathit { x } } } _ { t _ { i } } ^ { \star }$ in terms of $\boldsymbol { x } _ { t _ { i - } } ^ { \star }$ 1 as

$$
\pmb { x } _ { t _ { i } } ^ { \star } = \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \pmb { x } _ { t _ { i - 1 } } ^ { \star } + \sigma _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { \lambda } \pmb { \mu } _ { \theta } \big ( \pmb { x } _ { t ( \lambda ) } ^ { \star } , t ( \lambda ) \big ) \mathrm { d } \lambda .\tag{177}
$$

Combining this identity with the forward-value discretization (167), we can decompose the distance between the iterates $\pmb { x } _ { t _ { i } } ^ { \mathsf { f o r } }$ and $\boldsymbol { x } _ { t _ { i } } ^ { \star }$ as

$$
\begin{array} { r l } & { x _ { t _ { i } } ^ { \mathrm { f o r } } - x _ { t _ { i } } ^ { \star } = \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \big ( x _ { t _ { i - 1 } } ^ { \mathrm { f o r } } - x _ { t _ { i - 1 } } ^ { \star } \big ) + \sigma _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { \lambda } \bigg ( \mu _ { \theta } \big ( x _ { t _ { i } } ^ { \mathrm { f o r } } , t _ { i } \big ) - \mu _ { \theta } \big ( x _ { t ( \lambda ) } ^ { \star } , t ( \lambda ) \big ) \bigg ) \mathrm { d } \lambda } \\ & { \quad \quad \quad = \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \big ( x _ { t _ { i - 1 } } ^ { \mathrm { f o r } } - x _ { t _ { i - 1 } } ^ { \star } \big ) + \sigma _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { \lambda } \mathrm { d } \lambda \cdot \Big ( \mu _ { \theta } \big ( x _ { t _ { i } } ^ { \mathrm { f o r } } , t _ { i } \big ) - \mu _ { \theta } \big ( x _ { t _ { i } } ^ { \star } , t _ { i } \big ) \Big ) } \\ & { \quad \quad \quad \quad + \sigma _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { \lambda } \Big ( \mu _ { \theta } \big ( x _ { t _ { i } } ^ { \star } , t _ { i } \big ) - \mu _ { \theta } \big ( x _ { t ( \lambda ) } ^ { \star } , t ( \lambda ) \big ) \Big ) \mathrm { d } \lambda . } \end{array}\tag{178}
$$

Let us control the last two terms on the right-hand-side of (178) separately. For the second term, we can leverage the Lipschitz property of $\mu _ { \boldsymbol { \theta } } ( \boldsymbol { x } , t )$ with respect to x (see Assumption 1 (A2)) to derive

$$
\begin{array} { r l } & { \left\| \sigma _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { \lambda } \mathrm { d } \lambda \cdot \Big ( \mu _ { \theta } \big ( { x } _ { t _ { i } } ^ { \mathrm { f o r } } , t _ { i } \big ) - \mu _ { \theta } \big ( { x } _ { t _ { i } } ^ { \star } , t _ { i } \big ) \Big ) \right\| \leq \sigma _ { t _ { i } } \mathrm { e } ^ { \lambda _ { t _ { i } } } \delta _ { i } \frac { L _ { x } } { \alpha _ { t _ { i } } } \| { x } _ { t _ { i } } ^ { \mathrm { f o r } } - { x } _ { t _ { i } } ^ { \star } \| } \\ & { \qquad = \delta _ { i } L _ { x } \| { x } _ { t _ { i } } ^ { \mathrm { f o r } } - { x } _ { t _ { i } } ^ { \star } \| , } \end{array}\tag{179}
$$

where we used $\lambda _ { t _ { i - 1 } } \leq \lambda _ { t _ { i } }$ and $\lambda _ { t } = \log ( \alpha _ { t } / \sigma _ { t } )$ . Similarly, we can control the third term by the total derivative bound $\dot { L } _ { t }$ along the exact ODE trajectory:

$$
\left\| \sigma _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { \lambda } \Big ( \mu _ { \theta } \big ( x _ { t _ { i } } ^ { \star } , t _ { i } \big ) - \mu _ { \theta } \big ( x _ { t ( \lambda ) } ^ { \star } , t ( \lambda ) \big ) \Big ) \mathrm { d } \lambda \right\| \leq \sigma _ { t _ { i } } \mathrm { e } ^ { \lambda _ { t _ { i } } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \widetilde { L } _ { t } ( \lambda _ { t _ { i } } - \lambda ) \mathrm { d } \lambda = \frac { 1 } { 2 } \sigma _ { t _ { i } } \mathrm { e } ^ { \lambda _ { t _ { i } } } \widetilde { L } _ { t } \delta _ { i } ^ { 2 }
$$

$$
= { \frac { 1 } { 2 } } \alpha _ { t _ { i } } { \widetilde L } _ { t } \delta _ { i } ^ { 2 } = { \cal O } ( M ^ { - 2 } ) ,\tag{180}
$$

where the last step arises from the condition that $\delta _ { i } = O ( M ^ { - 1 } )$ . Substituting (179) and (180) into (178), we find that for each $i \in [ M ]$

$$
\begin{array} { r } { \big \| \boldsymbol { x } _ { t _ { i } } ^ { \mathrm { f o r } } - \boldsymbol { x } _ { t _ { i } } ^ { \star } \big \| \leq \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \big \| \boldsymbol { x } _ { t _ { i - 1 } } ^ { \mathrm { f o r } } - \boldsymbol { x } _ { t _ { i - 1 } } ^ { \star } \big \| + L _ { x } \delta _ { i } \big \| \boldsymbol { x } _ { t _ { i } } ^ { \mathrm { f o r } } - \boldsymbol { x } _ { t _ { i } } ^ { \star } \big \| + O ( M ^ { - 2 } ) , } \end{array}\tag{181}
$$

or equivalently,

$$
\Big \| \boldsymbol { x } _ { t _ { i } } ^ { \mathrm { f o r } } - \boldsymbol { x } _ { t _ { i } } ^ { \star } \Big \| \leq \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } ( 1 - L _ { x } \delta _ { i } ) } \big \| \boldsymbol { x } _ { t _ { i - 1 } } ^ { \mathrm { f o r } } - \boldsymbol { x } _ { t _ { i - 1 } } ^ { \star } \big \| + O ( M ^ { - 2 } ) ,\tag{182}
$$

where the last step arises from the fact that $1 - L _ { x } \delta _ { i } = 1 - O ( 1 / M ) \asymp 1$

Applying (182) recursively allows us to bound the final error as

$$
\bigl \| \boldsymbol { x } _ { t _ { M } } ^ { \mathrm { f o r } } - \boldsymbol { x } _ { t _ { M } } ^ { \star } \bigr \| \leq O ( M ^ { - 2 } ) \sum _ { i = 1 } ^ { M } \prod _ { j = i + 1 } ^ { M } \frac { \sigma _ { t _ { j } } } { \sigma _ { t _ { j - 1 } } ( 1 - L _ { x } \delta _ { j } ) } = O ( M ^ { - 1 } ) .\tag{183}
$$

Here, the first step holds as ${ \pmb x } _ { t _ { 0 } } ^ { \mathsf { f o r } } = { \pmb x } _ { t _ { 0 } } ^ { \star }$ and the last step follows from

$$
\begin{array} { r l r } {  { \sum _ { i = 1 } ^ { M } \prod _ { j = i + 1 } ^ { M } \frac { \sigma _ { t _ { j } } } { \sigma _ { t _ { j - 1 } } ( 1 - L _ { x } \delta _ { j } ) } \le \sum _ { i = 1 } ^ { M } \frac { \sigma _ { t _ { M } } } { \sigma _ { t _ { i } } } \exp ( 2 L _ { x } \sum _ { j = i + 1 } ^ { M } \delta _ { j } ) } } \\ & { } & { \leq M \exp \bigl ( 2 L _ { x } ( \lambda _ { t _ { M } } - \lambda _ { t _ { 0 } } ) \bigr ) = O ( M ) . } \end{array}
$$

We used $1 / ( 1 - x ) \le \exp ( 2 x )$ for $0 \leq x \leq 1 / 2 , L _ { x } \delta \leq 1 / 2$ for suficiently large M, and $\sigma _ { t _ { M } } \leq \sigma _ { t _ { i } }$ . The same argument gives ma $\mathsf { r } _ { i \in [ M ] } \| \pmb { x } _ { t _ { i } } ^ { \mathrm { { f o r } } } - \pmb { x } _ { t _ { i } } ^ { \star } \| = O ( M ^ { - 1 } )$ . This justifies that the convergence order of the first-order, forward-value discretization scheme is equal to 1.

Proof of Claim (169). Let us fix an arbitrary $i \in [ M ]$ . Combining the expression of $\boldsymbol { x } _ { t _ { i } } ^ { \star }$ from (177) with the update rules for the deterministic DDIM $\pmb { x } _ { t _ { i } } ^ { \mathrm { b c k } } ~ ( \mathrm { s e e } ~ ( \bar { 1 } 3 ) )$ and the first-order forward-value discretization $\pmb { x } _ { t _ { i } } ^ { \mathsf { f o r } }$ (see (167)), we can express the target diference as

$$
\begin{array} { r l } {  { \boldsymbol { \sigma } _ { t , \epsilon - 1 } ^ { \mathrm { S t } , \epsilon } - \boldsymbol { \alpha } _ { t , \epsilon - 1 } ^ { \mathrm { S t } } + \boldsymbol { \alpha } _ { t , \epsilon - 1 } ^ { \mathrm { S t } } - \boldsymbol { \alpha } _ { t , \epsilon } ^ { \mathrm { S t } } } } \\ & { = \frac { \sigma _ { t , \epsilon } ^ { \mathrm { S t } } } { \sigma _ { t , \epsilon - 1 } } ( \boldsymbol { \alpha } _ { t , \epsilon - 1 } ^ { \mathrm { S t } , \epsilon } + \boldsymbol { \alpha } _ { t , \epsilon - 1 } ^ { \mathrm { S t } , \epsilon } ) + \sigma _ { \mathrm { t s } } ( \int _ { \mathbb { X } _ { t - 1 } } ^ { \mathbb { S t } } e ^ { \lambda } \Bigl ( \mu \boldsymbol { \nu } ( \boldsymbol { \alpha } _ { t - 1 } ^ { \mathrm { S t } } , t _ { i - 1 } ) + \mu \boldsymbol { \mu } ( \boldsymbol { \alpha } _ { t , \epsilon } ^ { \mathrm { S t } } , t _ { i } ) - 2 \mu \boldsymbol { \mu } ( \boldsymbol { \alpha } _ { t , \epsilon } ^ { \mathrm { S t } } , t ( \lambda ) ) \Bigr ) \mathrm { d } \lambda ) } \\ &  = \frac { \sigma _ { t , \epsilon } } { \sigma _ { t , \epsilon - 1 } } ( \boldsymbol { \alpha } _ { t , \epsilon - 1 } ^ { \mathrm { S t } , \epsilon } + \boldsymbol { \alpha } _ { t , \epsilon - 2 } ^ { \mathrm { S t } } - 2 \boldsymbol { \alpha } _ { t , \epsilon - 1 } ^ { \mathrm { S t } } ) + \sigma _ { t , \epsilon } \underbrace  ( \int _ { \mathbb { X } _ { t - 1 } } ^ { \mathbb { S t } } e ^ { \lambda } \Bigl ( \mu ( \boldsymbol { \alpha } _ { t - 1 } ^ { \mathrm { S t } } , t _ { i - 1 } ) + \mu \boldsymbol { \mu } ( \boldsymbol { \alpha } _ { t , \epsilon } ^ { \mathrm { S t } } , t _ { i } ) - 2 \mu \boldsymbol { \mu } ( \boldsymbol { \alpha } _ { t , \epsilon } ^ { \mathrm { S t } } ) , t ( \lambda ) \Bigr ) \mathrm  \end{array}
$$

In what follows, we shall analyze $x _ { 1 } , x _ { 2 }$ , and $x \mathrm { { 3 } }$ separately.

• Controlling $x _ { 1 }$ . For simplicity of notation, let us denote $\mu ( \lambda ) : = \mu _ { \theta } \bigl ( x _ { t ( \lambda ) } ^ { \star } , t ( \lambda ) \bigr )$ . We begin with decomposing $x _ { 1 }$ as

$$
\begin{array} { r l } & { \chi _ { 1 } = \mathrm { e } ^ { \lambda _ { t _ { i } } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \Big ( \mu _ { \theta } \big ( x _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } \big ) + \mu _ { \theta } \big ( x _ { t _ { i } } ^ { \star } , t _ { i } \big ) - 2 \mu _ { \theta } \big ( x _ { t ( \lambda ) } ^ { \star } , t ( \lambda ) \big ) \Big ) \mathrm { d } \lambda } \\ & { \quad \quad \quad + \displaystyle \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \big ( \mathrm { e } ^ { \lambda } - \mathrm { e } ^ { \lambda _ { t _ { i } } } \big ) \Big ( \mu _ { \theta } \big ( x _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } \big ) + \mu _ { \theta } \big ( x _ { t _ { i } } ^ { \star } , t _ { i } \big ) - 2 \mu _ { \theta } \big ( x _ { t ( \lambda ) } ^ { \star } , t ( \lambda ) \big ) \Big ) \mathrm { d } \lambda , } \end{array}\tag{185}
$$

and control these two terms individually.

For the first term, we first rewrite it as

$$
\begin{array} { r l } {  { \int _ { \lambda _ { \ell - 1 } } ^ { \lambda _ { \ell } } \Big ( \mu _ { \theta } \big ( x _ { t - 1 } ^ { * } , t _ { i - 1 } \big ) + \mu _ { \theta } \big ( x _ { t _ { i } } ^ { * } , t _ { i } \big ) - 2 \mu _ { \theta } \big ( x _ { t ( \lambda ) } ^ { * } , t ( \lambda ) \big ) \Big ) \mathrm { d } \lambda } } \\ & { = \int _ { \lambda _ { \ell - 1 } } ^ { \lambda _ { \ell } } \Big ( \mu \big ( \lambda _ { t - 1 } \big ) + \mu ( \lambda _ { t } ) - 2 \mu ( \lambda ) \Big ) \mathrm { d } \lambda } \\ & { = 2 \int _ { \lambda _ { t - 1 } } ^ { \lambda _ { \ell } } \frac { \Big ( \mu \big ( \lambda _ { t } \big ) - \mu \big ( \lambda _ { t - 1 } \big ) } { \lambda _ { t _ { i } } - \lambda _ { t _ { i - 1 } } } \big ( \lambda - \lambda _ { t _ { i - 1 } } \big ) + \mu ( \lambda _ { t - \cdot } ) - \mu ( \lambda ) \Big ) \mathrm { d } \lambda } \\ & { \quad + \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \Big ( \mu ( \lambda _ { t } ) - \mu ( \lambda _ { t _ { i - 1 } } ) - \frac { 2 \big ( \mu ( \lambda _ { t _ { i } } ) - \mu ( \lambda _ { t _ { i - 1 } } ) \big ) } { \lambda _ { t _ { i } } - \lambda _ { t _ { i - 1 } } } \big ( \lambda - \lambda _ { t _ { i - 1 } } \big ) \Big ) \mathrm { d } \lambda } \\ &  = 2 \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \frac { \Big ( \mu ( \lambda _ { t } ) - \mu ( \lambda _ { t - 1 } ) } { \lambda _ { t _ { i } } - \lambda _ { t _ { i - 1 } } } ( \lambda - \lambda _ { t _ { i - 1 } } ) + \mu ( \lambda _ { t _ { i - 1 } }  \end{array}\tag{186}
$$

where the last identity holds because

$$
\begin{array} { r l } {  { \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \bigg ( \mu ( \lambda _ { t _ { i } } ) - \mu ( \lambda _ { t _ { i - 1 } } ) - \frac { 2 \big ( \mu ( \lambda _ { t _ { i } } ) - \mu ( \lambda _ { t _ { i - 1 } } ) \big ) } { \lambda _ { t _ { i } } - \lambda _ { t _ { i - 1 } } } ( \lambda - \lambda _ { t _ { i - 1 } } ) \bigg ) \mathrm { d } \lambda } } \\ & { = \big ( \mu ( \lambda _ { t _ { i } } ) - \mu ( \lambda _ { t _ { i - 1 } } ) \big ) \delta _ { i } - \frac { 2 \big ( \mu ( \lambda _ { t _ { i } } ) - \mu ( \lambda _ { t _ { i - 1 } } ) \big ) } { \lambda _ { t _ { i } } - \lambda _ { t _ { i - 1 } } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } ( \lambda - \lambda _ { t _ { i - 1 } } ) \mathrm { d } \lambda } \end{array}
$$

$$
= \big ( \pmb { \mu } ( \lambda _ { t _ { i } } ) - \pmb { \mu } ( \lambda _ { t _ { i - 1 } } ) \big ) \delta _ { i } - \frac { 2 \big ( \pmb { \mu } ( \lambda _ { t _ { i } } ) - \pmb { \mu } ( \lambda _ { t _ { i - 1 } } ) \big ) } { \delta _ { i } } \cdot \frac { 1 } { 2 } \delta _ { i } ^ { 2 } = 0 .
$$

Given the expression in (186), combining Taylor’s theorem with the total second-derivative bound $\widetilde { H } _ { t }$ , we can derive

$$
\begin{array} { r l } & {  { \| \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } ( \frac { \mu ( \lambda _ { t _ { i } } ) - \mu ( \lambda _ { t _ { i - 1 } } ) } { \lambda _ { t _ { i } } - \lambda _ { t _ { i - 1 } } } ( \lambda - \lambda _ { t _ { i - 1 } } ) + \mu ( \lambda _ { t _ { i - 1 } } ) - \mu ( \lambda ) ) \mathrm { d } \lambda \| } } \\ & { \quad \leq \| \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } ( \frac { \mu ( \lambda _ { t _ { i } } ) - \mu ( \lambda _ { t _ { i - 1 } } ) } { \lambda _ { t _ { i } } - \lambda _ { t _ { i - 1 } } } ( \lambda - \lambda _ { t _ { i - 1 } } ) - \mu ^ { \prime } ( \lambda _ { t _ { i - 1 } } ) ( \lambda - \lambda _ { t _ { i - 1 } } ) ) \mathrm { d } \lambda \| } \\ & { \quad \quad + \widetilde { H } _ { t } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } ( \lambda - \lambda _ { t _ { i - 1 } } ) ^ { 2 } \mathrm { d } \lambda } \\ & { \quad \leq \| \frac { \mu ( \lambda _ { t _ { i } } ) - \mu ( \lambda _ { t _ { i - 1 } } ) } { \lambda _ { t _ { i } } - \lambda _ { t _ { i - 1 } } } - \mu ^ { \prime } ( \lambda _ { t _ { i - 1 } } ) \| \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } ( \lambda - \lambda _ { t _ { i - 1 } } ) \mathrm { d } \lambda + \frac { \widetilde { H } _ { t } } { 3 } \delta _ { i } ^ { 3 } . } \end{array}\tag{187}
$$

Applying the same total second-derivative bound again, we can further bound the first term on the right-hand side of (187) as

$$
\left\| \frac { \mu ( \lambda _ { t _ { i } } ) - \mu ( \lambda _ { t _ { i - 1 } } ) } { \lambda _ { t _ { i } } - \lambda _ { t _ { i - 1 } } } - \mu ^ { \prime } ( \lambda _ { t _ { i - 1 } } ) \right\| \leq \left\| \mu ^ { \prime } ( \lambda _ { t _ { i - 1 } } ) \frac { \lambda _ { t _ { i } } - \lambda _ { t _ { i - 1 } } } { \lambda _ { t _ { i } } - \lambda _ { t _ { i - 1 } } } - \mu ^ { \prime } ( \lambda _ { t _ { i - 1 } } ) \right\| + \widetilde H _ { t } \delta _ { i } = \widetilde H _ { t } \delta _ { i } .
$$

Substituted into (187), this yields

$$
\begin{array} { r l } & { \bigg \| \displaystyle \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \bigg ( \frac { \mu ( \lambda _ { t _ { i } } ) - \mu ( \lambda _ { t _ { i - 1 } } ) } { \lambda _ { t _ { i } } - \lambda _ { t _ { i - 1 } } } ( \lambda - \lambda _ { t _ { i - 1 } } ) + \mu ( \lambda _ { t _ { i - 1 } } ) - \mu ( \lambda ) \bigg ) \mathrm { d } \lambda \bigg \| } \\ & { \quad \quad \leq \displaystyle \frac { 1 } { 2 } \widetilde H _ { t } \delta _ { i } ^ { 3 } + \frac { 1 } { 3 } \widetilde H _ { t } \delta _ { i } ^ { 3 } = O ( M ^ { - 3 } ) . } \end{array}\tag{188}
$$

Finally, plugging (188) into (186), the first term on the right-hand side of (185) can be bounded as

$$
\int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \Big ( \mu _ { \theta } \big ( \mathbf { x } _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } \big ) + \mu _ { \theta } \big ( \mathbf { x } _ { t _ { i } } ^ { \star } , t _ { i } \big ) - 2 \mu _ { \theta } \big ( \mathbf { x } _ { t ( \lambda ) } ^ { \star } , t ( \lambda ) \big ) \Big ) \mathrm { d } \lambda = O ( M ^ { - 3 } ) .\tag{189}
$$

– Turning to the second term, one can derive

$$
\begin{array} { r l } {  { \bigg \| \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \big ( \mathrm { e } ^ { \lambda } - \mathrm { e } ^ { \lambda _ { t _ { i } } } \big ) \Big ( \mu _ { \theta } \big ( x _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } \big ) + \mu _ { \theta } \big ( x _ { t _ { i } } ^ { \star } , t _ { i } \big ) - 2 \mu _ { \theta } \big ( x _ { t ( \lambda ) } ^ { \star } , t ( \lambda ) \big ) \Big ) \mathrm { d } \lambda \bigg \| } } \\ & { \le \Big | \mathrm { e } ^ { \lambda _ { t _ { i - 1 } } } - \mathrm { e } ^ { \lambda _ { t _ { i } } } \Big | \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \Big \| \mu _ { \theta } \big ( x _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } \big ) + \mu _ { \theta } \big ( x _ { t _ { i } } ^ { \star } , t _ { i } \big ) - 2 \mu _ { \theta } \big ( x _ { t ( \lambda ) } ^ { \star } , t ( \lambda ) \big ) \Big \| \mathrm { d } \lambda } \\ & { \le \mathrm { e } ^ { \lambda _ { t _ { i } } } \delta _ { i } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \widetilde { L } _ { t } \big ( \lambda - \lambda _ { t _ { i - 1 } } + \lambda _ { t _ { i } } - \lambda \big ) \mathrm { d } \lambda } \\ & { = \mathrm { e } ^ { \lambda _ { t _ { i } } } \widetilde { L } _ { t } \delta _ { i } ^ { 3 } = O ( M ^ { - 3 } ) . } \end{array}\tag{190}
$$

where the second inequality uses the total-derivative bound $\widetilde { L } _ { t }$ along the exact trajectory.

– Putting (189) and (190) together, we conclude that

$$
\| \chi _ { 1 } \| = O ( M ^ { - 3 } ) .\tag{191}
$$

• Controlling $x _ { 2 }$ . From Taylor’s theorem and Assumption 1 (A3), we can bound

$$
\left\| \mu _ { \theta } \big ( x _ { t _ { i - 1 } } ^ { \mathrm { b c k } } , t _ { i - 1 } \big ) + \mu _ { \theta } \big ( x _ { t _ { i - 1 } } ^ { \mathrm { f o r } } , t _ { i - 1 } \big ) - 2 \mu _ { \theta } \big ( x _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } \big ) - \nabla \mu _ { \theta } \big ( x _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } \big ) \big ( x _ { t _ { i - 1 } } ^ { \mathrm { b c k } } + x _ { t _ { i - 1 } } ^ { \mathrm { f o r } } - 2 x _ { t _ { i - 1 } } ^ { \star } \big ) \right\|
$$

$$
\begin{array} { r l } & { \leq H _ { x } \Big ( \big \| { \pmb x } _ { t _ { i - 1 } } ^ { \mathrm { b c k } } - { \pmb x } _ { t _ { i - 1 } } ^ { \star } \big \| ^ { 2 } + \big \| { \pmb x } _ { t _ { i - 1 } } ^ { \mathrm { f o r } } - { \pmb x } _ { t _ { i - 1 } } ^ { \star } \big \| ^ { 2 } \Big ) = O ( M ^ { - 2 } ) , } \end{array}
$$

where the last step follows from the first-order bounds for DDIM and the forward-value iterates. This allows us to bound $x _ { 2 }$ as

$$
\begin{array} { r l } & { \| \boldsymbol { \chi } _ { 2 } \| \leq \displaystyle \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { \lambda } \Big ( \big \| \nabla \mu _ { \theta } \big ( x _ { t _ { i - 1 } } ^ { * } , t _ { i - 1 } \big ) \big ( x _ { t _ { i - 1 } } ^ { \mathrm { b c k } } + x _ { t _ { i - 1 } } ^ { \mathrm { f o r } } - 2 x _ { t _ { i - 1 } } ^ { * } \big ) \big \| + O \big ( M ^ { - 2 } \big ) \Big ) \mathrm { d } \lambda } \\ & { \quad \quad \leq \big ( \mathrm { e } ^ { \lambda _ { t _ { i } } } - \mathrm { e } ^ { \lambda _ { t _ { i - 1 } } } \big ) \left( \frac { L _ { x } } { \alpha _ { t _ { i - 1 } } } \big \| x _ { t _ { i - 1 } } ^ { \mathrm { b c k } } + x _ { t _ { i - 1 } } ^ { \mathrm { f o r } } - 2 x _ { t _ { i - 1 } } ^ { \star } \big \| + O ( M ^ { - 2 } ) \right) } \\ & { \quad \quad \leq \mathrm { e } ^ { \lambda _ { t _ { i } } } \delta _ { i } \frac { L _ { x } } { \alpha _ { t _ { i - 1 } } } \big \| x _ { t _ { i - 1 } } ^ { \mathrm { b c k } } + x _ { t _ { i - 1 } } ^ { \mathrm { f o r } } - 2 x _ { t _ { i - 1 } } ^ { \star } \big \| + \mathrm { e } ^ { \lambda _ { t _ { i } } } \delta _ { i } O ( M ^ { - 2 } ) } \\ & { \quad \quad = \frac { L _ { x } \mathrm { e } ^ { \delta _ { i } } \delta _ { i } } { \sigma _ { t _ { i - 1 } } } \big \| x _ { t _ { i - 1 } } ^ { \mathrm { b c k } } + x _ { t _ { i - 1 } } ^ { \mathrm { f o r } } - 2 x _ { t _ { i - 1 } } ^ { \star } \big \| + O ( M ^ { - 3 } ) , } \end{array}\tag{192}
$$

where the second line applies Assumption 1 (A2) and the last step holds as $\lambda _ { t } = \log ( \alpha _ { t } / \sigma _ { t } )$ and $\delta _ { i } = O ( M ^ { - 1 } )$

• Controlling χ<sub>3</sub>. By Taylor’s theorem and Assumption 1 (A3), we can bound

$$
\begin{array} { r l } & { \| \mu _ { \theta } \big ( x _ { t _ { i } } ^ { \mathsf { f o r } } , t _ { i } \big ) - \mu _ { \theta } \big ( x _ { t _ { i } } ^ { \star } , t _ { i } \big ) - \mu _ { \theta } \big ( x _ { t _ { i - 1 } } ^ { \mathsf { f o r } } , t _ { i - 1 } \big ) + \mu _ { \theta } \big ( x _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } \big ) \| } \\ & { \qquad \leq \big \| \nabla \mu _ { \theta } \big ( x _ { t _ { i } } ^ { \star } , t _ { i } \big ) ( x _ { t _ { i } } ^ { \mathsf { f o r } } - x _ { t _ { i } } ^ { \star } ) - \nabla \mu _ { \theta } \big ( x _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } \big ) ( x _ { t _ { i - 1 } } ^ { \mathsf { f o r } } - x _ { t _ { i - 1 } } ^ { \star } ) \big \| } \\ & { \qquad + H _ { x } \big \| x _ { t _ { i } } ^ { \mathsf { f o r } } - x _ { t _ { i } } ^ { \star } \big \| ^ { 2 } + H _ { x } \big \| x _ { t _ { i - 1 } } ^ { \mathsf { f o r } } - x _ { t _ { i - 1 } } ^ { \star } \big \| ^ { 2 } } \\ & { \qquad \leq \big \| \big ( \nabla \mu _ { \theta } \big ( x _ { t _ { i } } ^ { \star } , t _ { i } \big ) - \nabla \mu _ { \theta } ( x _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } ) \big ) ( x _ { t _ { i } } ^ { \mathsf { f o r } } - x _ { t _ { i } } ^ { \star } ) \big \| } \\ &  \qquad + \big \| \nabla \mu _ { \theta } \big ( x _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } \big ) \big \| \big \| x _ { t _ { i } } ^ { \mathsf { f o r } } - x _ { t _ { i } } ^ { \star } - x _ { t _ { i - 1 } } ^ \end{array}\tag{193}
$$

where the last step applies (183).

To control the first term on the right-hand-side of (193), Assumption 1 (A3) and (A6), together with the bound V in (175), give

$$
\begin{array} { r } { \left\| \nabla \mu _ { \theta } \big ( x _ { t _ { i } } ^ { \star } , t _ { i } \big ) - \nabla \mu _ { \theta } \big ( x _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } \big ) \right\| \leq H _ { x } \| x _ { t _ { i } } ^ { \star } - x _ { t _ { i - 1 } } ^ { \star } \| + H \delta _ { i } \leq ( H _ { x } V + H ) \delta _ { i } = O ( M ^ { - 1 } ) . } \end{array}\tag{194}
$$

Therefore the first term is bounded by

$$
\begin{array} { r l } & { \| \big ( \nabla \mu _ { \theta } \big ( x _ { t _ { i } } ^ { \star } , t _ { i } \big ) - \nabla \mu _ { \theta } \big ( x _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } \big ) \big ) ( x _ { t _ { i } } ^ { \mathrm { f o r } } - x _ { t _ { i } } ^ { \star } ) \| } \\ & { \quad \leq \| \nabla \mu _ { \theta } \big ( x _ { t _ { i } } ^ { \star } , t _ { i } \big ) - \nabla \mu _ { \theta } \big ( x _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } \big ) \| \| x _ { t _ { i } } ^ { \mathrm { f o r } } - x _ { t _ { i } } ^ { \star } \| = O ( M ^ { - 2 } ) , } \end{array}\tag{195}
$$

where the last equation applies (183) again.

In addition, putting collectively what we have shown in (178), (179), and (180), one knows that

$$
\begin{array} { r } { \| { \boldsymbol x } _ { t _ { i } } ^ { \mathrm { f o r } } - { \boldsymbol x } _ { t _ { i } } ^ { \star } - { \boldsymbol x } _ { t _ { i - 1 } } ^ { \mathrm { f o r } } + { \boldsymbol x } _ { t _ { i - 1 } } ^ { \star } \| \leq \bigg | \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } - 1 \bigg | \| { \boldsymbol x } _ { t _ { i - 1 } } ^ { \mathrm { f o r } } - { \boldsymbol x } _ { t _ { i - 1 } } ^ { \star } \| + L _ { x } \delta _ { i } \| { \boldsymbol x } _ { t _ { i } } ^ { \mathrm { f o r } } - { \boldsymbol x } _ { t _ { i } } ^ { \star } \| + O ( M ^ { - 2 } ) . } \end{array}
$$

Since $\sigma _ { t _ { i } } \le \sigma _ { t _ { i - 1 } } ,$ , we have

$$
\left| \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } - 1 \right| = 1 - \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } = 1 - \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } \mathrm { e } ^ { - ( \lambda _ { t _ { i } } - \lambda _ { t _ { i - 1 } } ) } \le 1 - \mathrm { e } ^ { - \delta _ { i } } \le \delta _ { i } ,
$$

which together with (182) leads to

$$
\begin{array} { r } { \big \| x _ { t _ { i } } ^ { \mathrm { f o r } } - x _ { t _ { i } } ^ { \star } - x _ { t _ { i - 1 } } ^ { \mathrm { f o r } } + x _ { t _ { i - 1 } } ^ { \star } \big \| \leq \delta _ { i } \big \| x _ { t _ { i - 1 } } ^ { \mathrm { f o r } } - x _ { t _ { i - 1 } } ^ { \star } \big \| + L _ { x } \delta _ { i } \big \| x _ { t _ { i } } ^ { \mathrm { f o r } } - x _ { t _ { i } } ^ { \star } \big \| + O ( M ^ { - 2 } ) = O ( M ^ { - 2 } ) . } \end{array}\tag{196}
$$

Therefore the second term in the right-hand-side of (193) is bounded by

$$
\big \| \nabla \mu _ { \theta } \big ( x _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } \big ) \big \| \big \| x _ { t _ { i } } ^ { \mathrm { f o r } } - x _ { t _ { i } } ^ { \star } - x _ { t _ { i - 1 } } ^ { \mathrm { f o r } } + x _ { t _ { i - 1 } } ^ { \star } \big \| \leq \frac { L _ { x } } { \alpha _ { t _ { i - 1 } } } O ( M ^ { - 2 } ) \leq \frac { L _ { x } } { \alpha _ { \operatorname* { m i n } } } O ( M ^ { - 2 } ) = O ( M ^ { - 2 } ) ,\tag{197}
$$

where we used $\alpha _ { t _ { i - 1 } } \geq \alpha _ { \operatorname* { m i n } } > 0$

Finally, plugging the above bound into the definition of $x \mathrm { { 3 } }$ , we conclude that

$$
\begin{array} { r l } & { \| \chi _ { 3 } \| \leq \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { \lambda } \mathrm { d } \lambda \cdot \| \mu _ { \theta } \big ( x _ { t _ { i } } ^ { \mathrm { f o r } } , t _ { i } \big ) - \mu _ { \theta } \big ( x _ { t _ { i } } ^ { \star } , t _ { i } \big ) - \mu _ { \theta } \big ( x _ { t _ { i - 1 } } ^ { \mathrm { f o r } } , t _ { i - 1 } \big ) + \mu _ { \theta } \big ( x _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } \big ) \| } \\ & { \qquad \leq \mathrm { e } ^ { \lambda _ { t _ { i } } } \delta _ { i } \big \| \mu _ { \theta } \big ( x _ { t _ { i } } ^ { \mathrm { f o r } } , t _ { i } \big ) - \mu _ { \theta } \big ( x _ { t _ { i } } ^ { \star } , t _ { i } \big ) - \mu _ { \theta } \big ( x _ { t _ { i - 1 } } ^ { \mathrm { f o r } } , t _ { i - 1 } \big ) + \mu _ { \theta } \big ( x _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } \big ) \big \| = O \big ( M ^ { - 3 } \big ) . } \end{array}\tag{198}
$$

• Putting bounds for $x _ { 1 } , x _ { 2 } ,$ and χ<sub>3</sub> together. Finally, combining the bounds in (191), (192), and (198), and then plugging it into (184), we find that for any $i \geq 1$

$$
\bigl \| \boldsymbol { x } _ { t _ { i } } ^ { \mathrm { b e k } } - \boldsymbol { x } _ { t _ { i } } ^ { \star } + \boldsymbol { x } _ { t _ { i } } ^ { \mathrm { f o r } } - \boldsymbol { x } _ { t _ { i } } ^ { \star } \bigr \| \leq \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } ( 1 + L _ { x } \mathrm { e } ^ { \delta _ { i } } \delta _ { i } ) \bigl \| \boldsymbol { x } _ { t _ { i - 1 } } ^ { \mathrm { b e k } } + \boldsymbol { x } _ { t _ { i - 1 } } ^ { \mathrm { f o r } } - 2 { \boldsymbol { x } } _ { t _ { i - 1 } } ^ { \star } \bigr \| + \sigma _ { t _ { i } } O \bigl ( M ^ { - 3 } \bigr ) .
$$

Applying this inequality recursively from $i = 1$ to $i = M$ , and noting that $\pmb { x } _ { t _ { 0 } } ^ { \mathrm { b c k } } + \pmb { x } _ { t _ { 0 } } ^ { \mathrm { f o r } } - 2 \pmb { x } _ { t _ { 0 } } ^ { \star } = 0$ , we obtain

$$
\begin{array} { r l } { \| x _ { t _ { M } } ^ { \mathrm { b c k } } - x _ { t _ { M } } ^ { * } + x _ { t _ { M } } ^ { \mathrm { f o r } } - x _ { t _ { M } } ^ { * } \| \le \displaystyle \sum _ { i = 1 } ^ { M } \displaystyle \prod _ { j = i + 1 } ^ { M } \displaystyle \frac { \sigma _ { t _ { j } } } { \sigma _ { t _ { j - 1 } } } ( 1 + L _ { x } \mathrm { e } ^ { \delta _ { j } } \delta _ { j } ) \cdot \sigma _ { t _ { i } } O ( M ^ { - 3 } ) } & { } \\ { \le \displaystyle \sum _ { i = 1 } ^ { M } \displaystyle \prod _ { j = i + 1 } ^ { M } \displaystyle \frac { \sigma _ { t _ { j } } } { \sigma _ { t _ { j - 1 } } } \exp ( L _ { x } \mathrm { e } ^ { \delta _ { j } } \delta _ { j } ) \cdot \sigma _ { t _ { i } } O ( M ^ { - 3 } ) } & { } \\ { = \displaystyle \sum _ { i = 1 } ^ { M } \sigma _ { t _ { M } } \exp ( L _ { x } \mathrm { e } ^ { \delta } ( \lambda _ { t _ { M } } - \lambda _ { t _ { i } } ) ) \cdot O ( M ^ { - 3 } ) } & { } \\ { \le M \sigma _ { t _ { M } } \exp ( L _ { x } \mathrm { e } ^ { \delta } ( \lambda _ { t _ { M } } - \lambda _ { t _ { 0 } } ) ) \cdot O ( M ^ { - 3 } ) = O ( M ^ { - 2 } ) , } \end{array}
$$

where the second step is true because $1 + x \leq \mathrm { e } ^ { x }$ for any $x \in \mathbb { R }$ , and the last line holds as $\lambda _ { t _ { M } } - \lambda _ { t _ { i } } \leq$ $\lambda _ { t _ { M } } - \lambda _ { t _ { 0 } }$ for all $0 \leq i \leq M$

## E.2 Proof of Theorem 7

Comparing the update rules of our sampler (19) and that of the idealized forward-value discretization (167), one can express the diference between their iterates as

$$
\boldsymbol { x } _ { t _ { i } } - \boldsymbol { x } _ { t _ { i } } ^ { \mathrm { f o r } } = \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \big ( \boldsymbol { x } _ { t _ { i - 1 } } - \boldsymbol { x } _ { t _ { i - 1 } } ^ { \mathrm { f o r } } \big ) + \sigma _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { \lambda } \mathrm { d } \lambda \boldsymbol { \cdot } \big ( \mu _ { \theta } ( \widehat { \boldsymbol { x } } _ { t _ { i } } , t _ { i } ) - \mu _ { \theta } ( \boldsymbol { x } _ { t _ { i } } ^ { \mathrm { f o r } } , t _ { i } ) \big ) ,
$$

for any $i \geq 1$ . Leveraging the Lipschitz property of $\mu _ { \theta } ( x _ { t _ { i } } , t _ { i } )$ , we can bound

$$
\begin{array} { r l } { \Big \| x _ { t _ { i } } - x _ { t _ { i } } ^ { \mathrm { f o r } } \Big \| \leq \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \big \| x _ { t _ { i - 1 } } - x _ { t _ { i - 1 } } ^ { \mathrm { f o r } } \big \| + \sigma _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { \lambda } \mathrm { d } \lambda \cdot \big \| \mu _ { \theta } ( x _ { t _ { i } } ^ { \mathrm { f o r } } , t _ { i } ) - \mu _ { \theta } ( \widehat { x } _ { t _ { i } } , t _ { i } ) \big \| } & { } \\ { \leq \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \big \| x _ { t _ { i - 1 } } - x _ { t _ { i - 1 } } ^ { \mathrm { f o r } } \big \| + \sigma _ { t _ { i } } \mathrm { e } ^ { \lambda _ { t _ { i } } } \delta _ { i } \frac { L _ { x } } { \sigma _ { t _ { i } } } \big \| x _ { t _ { i } } ^ { \mathrm { f o r } } - \widehat { x } _ { t _ { i } } \big \| } & { } \\ { \leq \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \big \| x _ { t _ { i - 1 } } - x _ { t _ { i - 1 } } ^ { \mathrm { f o r } } \big \| + \delta _ { i } L _ { x } \big \| x _ { t _ { i } } ^ { \mathrm { f o r } } - x _ { t _ { i } } ^ { \star } ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) \big \| } & { } \\ { + \delta _ { i } L _ { x } \big \| x _ { t _ { i } } ^ { \star } ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) - \widehat { x } _ { t _ { i } } \big \| , } & { } \end{array}\tag{199}
$$

where the last equation uses $\sigma _ { t _ { i } } \mathrm { e } ^ { \lambda _ { t _ { i } } } / \alpha _ { t _ { i } } = \mathrm { e } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - \lambda _ { t _ { i } } } = 1$

We claim that the second term on the right-hand-side of (199) satisfies the following bound:

$$
\big \| x _ { t _ { i } } ^ { \mathrm { f o r } } - x _ { t _ { i } } ^ { \star } ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) \big \| \leq \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } ( 1 - \delta _ { i } L _ { x } ) } \big \| x _ { t _ { i - 1 } } ^ { \mathrm { f o r } } - x _ { t _ { i - 1 } } \big \| + \frac { \sigma _ { t _ { i } } \mathrm { e } ^ { \lambda _ { t _ { i } } } \widetilde L _ { t } \delta _ { i } ^ { 2 } } { 2 ( 1 - \delta _ { i } L _ { x } ) } .\tag{200}
$$

The proof is deferred to the end of this section.

Armed with (200), for $i \geq 2$ , we can plug it into (199) to obtain the following key relationship regarding the diference between our sampler $\mathbf { x } _ { t _ { i } }$ and the idealized forward-value discretization $\pmb { x } _ { t _ { i } } ^ { \mathsf { f o r } }$

$$
\begin{array} { r l } & { \left\| x _ { t _ { i } } - x _ { t _ { i } } ^ { \mathrm { f o r } } \right\| \leq \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \big \| x _ { t _ { i - 1 } } - x _ { t _ { i - 1 } } ^ { \mathrm { f o r } } \big \| + \delta _ { i } L _ { x } \bigg ( \frac { \sigma _ { t _ { i } } \big \| x _ { t _ { i - 1 } } ^ { \mathrm { f o r } } - x _ { t _ { i - 1 } } \big \| } { \sigma _ { t _ { i - 1 } } \big ( 1 - \delta _ { i } L _ { x } \big ) } + \frac { \sigma _ { t _ { i } } \mathrm { e } ^ { \lambda _ { t _ { i } } } \widetilde L _ { t } \delta _ { i } ^ { 2 } } { 2 \big ( 1 - \delta _ { i } L _ { x } \big ) } \bigg ) } \\ & { \qquad + \delta _ { i } L _ { x } \big \| x _ { t _ { i } } ^ { * } ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) - \widehat { x } _ { t _ { i } } \big \| } \\ & { \qquad = \frac { \sigma _ { t _ { i } } \big \| x _ { t _ { i - 1 } } ^ { \mathrm { f o r } } - x _ { t _ { i - 1 } } \big \| } { \sigma _ { t _ { i - 1 } } \big ( 1 - \delta _ { i } L _ { x } \big ) } + \frac { \sigma _ { t _ { i } } \mathrm { e } ^ { \lambda _ { t _ { i } } } L _ { x } \widetilde L _ { t } \delta _ { i } ^ { 3 } } { 2 \big ( 1 - \delta _ { i } L _ { x } \big ) } + \delta _ { i } L _ { x } \big \| x _ { t _ { i } } ^ { * } ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) - \widehat { x } _ { t _ { i } } \big \| } \\ &  \qquad = \frac { \sigma _ { t _ { i } } \big \| x _ { t _ { i - 1 } } ^ { \mathrm { f o r } } - x _ { t _ { i - 1 } } \big \| }  \sigma _  t _  i  \end{array}\tag{201}
$$

where the last equation uses (170), namely $\left\| \pmb { x } _ { t _ { i } } ^ { \star } ( \pmb { x } _ { t _ { i - 1 } } , t _ { i - 1 } ) - \widehat { \pmb { x } } _ { t _ { i } } \right\| = o ( M ^ { - 1 } )$ and $\delta _ { i } = O ( M ^ { - 1 } )$ . For the first step, the common initialization $\pmb { x } _ { t _ { 0 } } = \pmb { x } _ { t _ { 0 } } ^ { \mathsf { f o r } }$ and (199)–(200) give

$$
\begin{array} { r l } & { \left\| { \pmb x } _ { t _ { 1 } } - { \pmb x } _ { t _ { 1 } } ^ { \mathrm { f o r } } \right\| \leq \frac { \sigma _ { t _ { 1 } } \mathrm { e } ^ { \lambda _ { t _ { 1 } } } L _ { x } \widetilde L _ { t } \delta _ { 1 } ^ { 3 } } { 2 \left( 1 - \delta _ { 1 } L _ { x } \right) } + \delta _ { 1 } L _ { x } \left\| { \pmb x } _ { t _ { 1 } } ^ { \star } ( { \pmb x } _ { t _ { 0 } } , t _ { 0 } ) - \widehat { { \pmb x } } _ { t _ { 1 } } \right\| } \\ & { \qquad = O ( M ^ { - 3 } ) + O ( M ^ { - 2 } ) = O ( M ^ { - 2 } ) , } \end{array}
$$

where we used $\delta _ { 1 } = O ( M ^ { - 1 } )$ and the first-step bound in (170). Applying relationship (201) recursively from $i = 2 .$ , and using the first-step bound $\| \pmb { x } _ { t _ { 1 } } - \pmb { x } _ { t _ { 1 } } ^ { \mathsf { f o r } } \| = O ( M ^ { - 2 } )$ established above, we can bound the final error at $t _ { M }$ as

$$
\left\| x _ { t _ { M } } - x _ { t _ { M } } ^ { \mathrm { f o r } } \right\| \leq \prod _ { j = 2 } ^ { M } \frac { \sigma _ { t _ { j } } } { \sigma _ { t _ { j - 1 } } ( 1 - \delta _ { j } L _ { x } ) } \Big \| x _ { t _ { 1 } } - x _ { t _ { 1 } } ^ { \mathrm { f o r } } \Big \| + \sum _ { i = 2 } ^ { M } \prod _ { j = i + 1 } ^ { M } \frac { \sigma _ { t _ { j } } } { \sigma _ { t _ { j - 1 } } ( 1 - \delta _ { j } L _ { x } ) } \cdot o \big ( M ^ { - 2 } \big ) .\tag{202}
$$

To finish up, note that

$$
\begin{array} { r l } & { \displaystyle \prod _ { j = i + 1 } ^ { M } \frac { \sigma _ { t _ { j } } } { \sigma _ { t _ { j - 1 } } \left( 1 - \delta _ { j } L _ { x } \right) } \overset { \mathrm { ( a ) } } { \le } \displaystyle \prod _ { j = i + 1 } ^ { M } \frac { \sigma _ { t _ { j } } } { \sigma _ { t _ { j - 1 } } } \exp \bigl ( 2 \delta _ { j } L _ { x } \bigr ) = \frac { \sigma _ { t _ { M } } } { \sigma _ { t _ { i } } } \exp \biggl ( 2 L _ { x } \sum _ { j = i + 1 } ^ { M } \delta _ { j } \biggr ) } \\ & { \qquad \quad = \frac { \sigma _ { t _ { M } } } { \sigma _ { t _ { i } } } \exp \bigl ( 2 L _ { x } ( \lambda _ { t _ { M } } - \lambda _ { t _ { i } } ) \bigr ) } \\ & { \qquad \quad \overset { \mathrm { ( b ) } } { \le } \exp \bigl ( 2 L _ { x } ( \lambda _ { t _ { M } } - \lambda _ { t _ { 0 } } ) \bigr ) , } \end{array}
$$

where (a) holds because $1 / ( 1 - x ) \le \exp ( 2 x )$ for $0 \leq x \leq 1 / 2$ and $\delta _ { j } L _ { x } = O ( M ^ { - 1 } ) \leq 1 / 2 ; \mathrm { ( b ) }$ is true because of the assumption $\sigma _ { t _ { i } } \geq \sigma _ { t _ { M } }$ and $\lambda _ { t _ { M } } - \lambda _ { t _ { i } } \leq \lambda _ { t _ { M } } - \lambda _ { t _ { 0 } }$ for all $0 \leq i \leq M$ . Using this bound reaches the advertised result:

$$
\begin{array} { r l } & { \left\| { { x } _ { t _ { M } } } - { { x } _ { t _ { M } } ^ { \mathsf { f o r } } } \right\| \leq \exp \left( 2 L _ { x } ( \lambda _ { t _ { M } } - \lambda _ { t _ { 0 } } ) \right) \left( O ( M ^ { - 2 } ) + ( M - 1 ) o ( M ^ { - 2 } ) \right) } \\ & { \qquad = o ( M ^ { - 1 } ) . } \end{array}\tag{203}
$$

It remains to prove the claim (200). Towards this, note that by the difusion ODE (173), we can rewrite $\pmb { x } _ { t _ { i } } ^ { \star } ( \pmb { x } _ { t _ { i - 1 } } , t _ { i - 1 } )$ defined in (171) in terms of the data predictor $\pmb { \mu } _ { \theta }$ as

$$
\pmb { x } _ { t _ { i } } ^ { \star } ( \pmb { x } _ { t _ { i - 1 } } , t _ { i - 1 } ) = \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \pmb { x } _ { t _ { i - 1 } } + \sigma _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { \lambda } \pmb { \mu } _ { \theta } \big ( \pmb { x } _ { t ( \lambda ) } ^ { \star } ( \pmb { x } _ { t _ { i - 1 } } , t _ { i - 1 } ) , t ( \lambda ) \big ) \mathrm { d } \lambda .
$$

Combined with the definition of $\pmb { x } _ { t _ { i } } ^ { \mathsf { f o r } }$ (cf. (167)), the above representation allows us to decompose their diference as

$$
x _ { t _ { i } } ^ { \mathrm { f o r } } - x _ { t _ { i } } ^ { \star } ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) = \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \big ( x _ { t _ { i - 1 } } ^ { \mathrm { f o r } } - x _ { t _ { i - 1 } } \big ) + \sigma _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { \lambda } \Big ( \mu \circ \big ( x _ { t _ { i } } ^ { \mathrm { f o r } } , t _ { i } \big ) - \mu \circ \big ( x _ { t ( \lambda ) } ^ { \star } ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) , t ( \lambda ) \big ) \Big ) \mathrm { d } \lambda
$$

$$
\begin{array} { r l } { \displaystyle } & { = \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \big ( x _ { t _ { i - 1 } } ^ { \mathrm { f o r } } - x _ { t _ { i - 1 } } \big ) + \sigma _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { \lambda } \mathrm { d } \lambda \cdot \Big ( \mu _ { \theta } \big ( x _ { t _ { i } } ^ { \mathrm { f o r } } , t _ { i } \big ) - \mu _ { \theta } \big ( x _ { t _ { i } } ^ { \star } ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) , t _ { i } \big ) \Big ) } \\ { \displaystyle } & { + \sigma _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { \lambda } \Big ( \mu _ { \theta } \big ( x _ { t _ { i } } ^ { \star } ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) , t _ { i } \big ) - \mu _ { \theta } \big ( x _ { t ( \lambda ) } ^ { \star } ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) , t ( \lambda ) \big ) \Big ) \mathrm { d } \lambda . \qquad ( \lambda _ { t _ { i } } ^ { \star } ( x _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } ) , t ( \lambda ) \big ) } \end{array}\tag{204}
$$

Let us control the second and third quantities on the right-hand-side of (204) separately. Regarding the second term, we can invoke a similar argument as in (179) to bound it as

$$
\left\| \sigma _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { \lambda } \mathrm { d } \lambda \cdot \left( \mu _ { \theta } \big ( x _ { t _ { i } } ^ { \mathrm { f o r } } , t _ { i } \big ) - \mu _ { \theta } \big ( x _ { t _ { i } } ^ { \star } ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) , t _ { i } \big ) \right) \right\| \leq \sigma _ { t _ { i } } \mathrm { e } ^ { \lambda _ { t _ { i } } } \delta _ { i } \frac { L _ { x } } { \alpha _ { t _ { i } } } \| { x } _ { t _ { i } } ^ { \mathrm { f o r } } - x _ { t _ { i } } ^ { \star } ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) \|
$$

$$
= \delta _ { i } L _ { x } \big \| \pmb { x } _ { t _ { i } } ^ { \mathrm { f o r } } - \pmb { x } _ { t _ { i } } ^ { \star } ( \pmb { x } _ { t _ { i - 1 } } , t _ { i - 1 } ) \big \| ,\tag{205}
$$

where the first inequality uses $\lambda _ { t _ { i - 1 } } \leq \lambda _ { t _ { i } }$ and the Lipschitz condition of $\mu _ { \theta }$ from Assumption 1 (A2), and the last equation uses $\lambda _ { t _ { i } } = \log ( \alpha _ { t _ { i } } / \sigma _ { t _ { i } } )$ . As for the third quantity, repeating a similar argument for (180), one can bound it as

$$
\sigma _ { t _ { i } } \left\| \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { \lambda } \Big ( \mu _ { \theta } ( x _ { t _ { i } } ^ { \star } ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) , t _ { i } ) - \mu _ { \theta } ( x _ { t ( \lambda ) } ^ { \star } ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) , t ( \lambda ) ) \Big ) \mathrm { d } \lambda \right\| \leq \frac { 1 } { 2 } \sigma _ { t _ { i } } \mathrm { e } ^ { \lambda _ { t _ { i } } } \widetilde L _ { t } \delta _ { i } ^ { 2 } .
$$

Plugging these two bounds into (204) and rearranging the inequality, we finish the proof of the claim (200).

Proof of F-DDIM. Finally, we verify the lookahead accuracy of F-DDIM satisfies (170). Let

$$
E _ { i } = \Vert \widehat { \pmb x } _ { t _ { i } } - \pmb x _ { t _ { i } } ^ { \star } ( \pmb x _ { t _ { i - 1 } } , t _ { i - 1 } ) \Vert .
$$

We first record uniform boundedness of the iterates. Since $\| \mu _ { \theta } \| \leq L _ { \mu }$ and $\sigma _ { t _ { i } } / \sigma _ { t _ { i - 1 } } \leq 1$ , the corrected update gives

$$
\| \pmb { x } _ { t _ { i } } \| \leq \| \pmb { x } _ { t _ { 0 } } \| + \alpha _ { \operatorname* { m a x } } L _ { \mu } \sum _ { j = 1 } ^ { i } \delta _ { j } = O ( 1 ) .
$$

For $i \geq 2$ , the update rule of $\widehat { \pmb { x } } _ { t _ { i } }$ gives

$$
\| \widehat { \pmb { x } } _ { t _ { i } } \| \leq C \| \pmb { x } _ { t _ { i - 1 } } \| + C \delta \| \widehat { \pmb { x } } _ { t _ { i - 1 } } \| = O ( 1 ) .
$$

The local exact trajectories are also bounded by the integral form (173). Thus there exists a ball $\| { \pmb x } \| \le R$ that contains all these states and trajectories.

We next establish the derivative bounds of the noise predictor $\varepsilon _ { \theta } .$ . From $\pmb { \varepsilon } _ { \theta } = \pmb { x } / \sigma _ { t } - \mathrm { e } ^ { \lambda _ { t } } \pmb { \mu } _ { \theta }$ , Assumption 1 (A1), (A2), and (A4) give the following inequalities for all x in a fixed ball $\| { \pmb x } \| \le R$ and $\lambda \in [ \lambda _ { t _ { i - 1 } } , \lambda _ { t _ { i } } ] \mathrm { . }$

$$
\begin{array} { l l } { \nabla _ { x } \varepsilon _ { \theta } = \frac { I - \alpha _ { t } \nabla _ { x } \mu _ { \theta } } { \sigma _ { t } } , } & { \| \nabla _ { x } \varepsilon _ { \theta } \| \leq \frac { 1 + L _ { x } } { \sigma _ { t } } \leq \frac { 1 + L _ { x } } { \sigma _ { t _ { \lambda } } } = : K _ { x , i } , } \\ { \frac { \partial } { \partial \lambda } \varepsilon _ { \theta } = - \frac { 1 } { \sigma _ { t } ^ { 2 } } \frac { \partial } { \partial \lambda } \sigma _ { t ( \lambda ) } \mathbf { x } - \mathrm { e } ^ { \lambda } \Big ( \mu _ { \theta } + \frac { \partial } { \partial \lambda } \mu _ { \theta } \Big ) , } & { \displaystyle \left\| \frac { \partial } { \partial \lambda } \varepsilon _ { \theta } \right\| \leq \frac { R \operatorname* { m a x } _ { \lambda } \left. \frac { \partial } { \partial \lambda } \log \sigma _ { t ( \lambda ) } \right. } { \sigma _ { t _ { \lambda } } } + \mathrm { e } ^ { \lambda _ { t _ { i } } } ( L _ { \mu } + L _ { t } ) = : K _ { t , i } . } \end{array}
$$

Consider F-DDIM with its cached noise predictions. We have

$$
\begin{array} { r l } & { \widehat { x } _ { t _ { i } } - x _ { t _ { i } } ^ { \star } ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) = \alpha _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - \lambda } \varepsilon _ { \theta } ( x _ { t ( \lambda ) } ^ { \star } ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) , t ( \lambda ) ) \mathrm { d } \lambda - \left( \frac { \alpha _ { t _ { i } } \sigma _ { t _ { i - 1 } } } { \alpha _ { t _ { i - 1 } } } - \sigma _ { t _ { i } } \right) \widetilde { \varepsilon } _ { t _ { i - 1 } } } \\ & { \qquad = \alpha _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - \lambda } \left( \varepsilon _ { \theta } ( x _ { t ( \lambda ) } ^ { \star } ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) , t ( \lambda ) ) - \widetilde { \varepsilon } _ { t _ { i - 1 } } \right) \mathrm { d } \lambda . } \end{array}
$$

For $i \geq 2$ , taking norms and using the definition $\widetilde { \pmb { \varepsilon } } _ { t _ { i - 1 } } = \pmb { \varepsilon } _ { \pmb { \theta } } ( \widehat { \pmb { x } } _ { t _ { i - 1 } } , t _ { i - 1 } )$ gives

$$
E _ { i } \leq \alpha _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - \lambda } \big \| \varepsilon _ { \theta } ( \pmb { x } _ { t ( \lambda ) } ^ { \star } ( \pmb { x } _ { t _ { i - 1 } } , t _ { i - 1 } ) , t ( \lambda ) ) - \widetilde { \varepsilon } _ { t _ { i - 1 } } \big \| \mathrm { d } \lambda
$$

$$
\leq \alpha _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - \lambda } \Big ( K _ { x , i } \big \| \pmb { x } _ { t ( \lambda ) } ^ { \star } ( \pmb { x } _ { t _ { i - 1 } } , t _ { i - 1 } ) - \widehat { \pmb { x } } _ { t _ { i - 1 } } \big \| + K _ { t , i } ( \lambda - \lambda _ { t _ { i - 1 } } ) \Big ) \mathrm { d } \lambda
$$

$$
\leq \sigma _ { t _ { i } } \mathrm { e } ^ { \delta _ { i } } K _ { x , i } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \big \| \pmb { x } _ { t ( \lambda ) } ^ { \star } ( \pmb { x } _ { t _ { i - 1 } } , t _ { i - 1 } ) - \widehat { \pmb { x } } _ { t _ { i - 1 } } \big \| \mathrm { d } \lambda + K _ { t , i } \sigma _ { t _ { i } } \mathrm { e } ^ { \delta _ { i } } \delta _ { i } ^ { 2 } , \quad i \geq 2 .\tag{206}
$$

It remains to prove that

$$
\begin{array} { r } { \left\| \pmb { x } _ { t ( \lambda ) } ^ { \star } ( \pmb { x } _ { t _ { i - 1 } } , t _ { i - 1 } ) - \widehat { \pmb { x } } _ { t _ { i - 1 } } \right\| = O ( \delta _ { i } ) , \qquad \lambda _ { t _ { i - 1 } } \leq \lambda \leq \lambda _ { t _ { i } } , \quad i \geq 2 . } \end{array}\tag{207}
$$

Towards this, we first note that

$$
\begin{array} { r } { \left\| \boldsymbol { x } _ { t ( \lambda ) } ^ { \star } ( \boldsymbol { x } _ { t _ { i - 1 } } , t _ { i - 1 } ) - \widehat { \boldsymbol { x } } _ { t _ { i - 1 } } \right\| \leq \left\| \boldsymbol { x } _ { t ( \lambda ) } ^ { \star } ( \boldsymbol { x } _ { t _ { i - 1 } } , t _ { i - 1 } ) - \boldsymbol { x } _ { t _ { i - 1 } } \right\| + \left\| \boldsymbol { x } _ { t _ { i - 1 } } - \widehat { \boldsymbol { x } } _ { t _ { i - 1 } } \right\| . } \end{array}\tag{208}
$$

The ODE (173) and Assumption 1 (A1) bound the first term by $C \delta _ { i } .$ , uniformly on the fixed finite λ interval. For the second term, inserting the update rules of $\mathbf { \mathcal { x } } _ { t _ { i } }$ and $\widehat { \mathbf { x } } _ { t _ { i } }$ for $i \geq 2$ yields

$$
\begin{array} { r l } & { x _ { t _ { i } } - \widehat x _ { t _ { i } } = \Big ( \frac { \sigma _ { t _ { i } } \alpha _ { t _ { i - 1 } } } { \sigma _ { t _ { i - 1 } } } - \alpha _ { t _ { i } } \Big ) \big ( \mu _ { \theta } ( \widehat x _ { t _ { i - 1 } } , t _ { i - 1 } ) - \mu _ { \theta } ( \widehat x _ { t _ { i } } , t _ { i } ) \big ) + \Big ( \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } - \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } \Big ) \big ( x _ { t _ { i - 1 } } - \widehat x _ { t _ { i - 1 } } \big ) } \\ & { \qquad = \alpha _ { t _ { i } } \big ( \mathrm { e } ^ { - \delta _ { i } } - 1 \big ) \big ( \mu _ { \theta } ( \widehat x _ { t _ { i - 1 } } , t _ { i - 1 } ) - \mu _ { \theta } ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) + \mu _ { \theta } ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) - \mu _ { \theta } ( \widehat x _ { t _ { i } } , t _ { i } ) \big ) } \\ & { \qquad + \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } \Big ( \mathrm { e } ^ { - \delta _ { i } } - 1 \Big ) \big ( x _ { t _ { i - 1 } } - \widehat x _ { t _ { i - 1 } } \big ) . } \end{array}\tag{209}
$$

Using Assumption 1 (A2) and (A4) yields

$$
\begin{array} { r l } & { \| { \pmb x } _ { t _ { i } } - \widehat { { \pmb x } } _ { t _ { i } } \| \leq \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } L _ { x } \delta _ { i } \big ( \| { \pmb x } _ { t _ { i - 1 } } - \widehat { { \pmb x } } _ { t _ { i - 1 } } \| + \| { \pmb x } _ { t _ { i - 1 } } - \widehat { { \pmb x } } _ { t _ { i } } \| \big ) } \\ & { \qquad + \alpha _ { t _ { i } } L _ { t } \delta _ { i } ^ { 2 } + \frac { \alpha _ { t _ { i } } } { \alpha _ { t _ { i - 1 } } } \delta _ { i } \| { \pmb x } _ { t _ { i - 1 } } - \widehat { { \pmb x } } _ { t _ { i - 1 } } \| } \\ & { \qquad \leq C \delta _ { i } \| { \pmb x } _ { t _ { i - 1 } } - \widehat { { \pmb x } } _ { t _ { i - 1 } } \| + C \delta _ { i } ^ { 2 } , \qquad i \geq 2 . } \end{array}\tag{210}
$$

Here the update rule of $\widehat { \pmb { x } } _ { t _ { i } }$ gives $\lVert \pmb { x } _ { t _ { i - 1 } } - \widehat { \pmb { x } } _ { t _ { i } } \rVert = O ( \delta _ { i } )$ . The initialization gives

$$
\| x _ { t _ { 1 } } - \widehat { x } _ { t _ { 1 } } \| \leq \frac { \alpha _ { t _ { 1 } } } { \alpha _ { t _ { 0 } } } | \sigma _ { t _ { 0 } } - 1 | ( 1 - \mathrm { e } ^ { - \delta _ { 1 } } ) \| x _ { t _ { 0 } } \| + \alpha _ { t _ { 1 } } ( 1 - \mathrm { e } ^ { - \delta _ { 1 } } ) \| \mu _ { \theta } ( \widehat { x } _ { t _ { 1 } } , t _ { 1 } ) \| = O ( \delta _ { 1 } ) .
$$

Induction for suficiently small δ therefore gives

$$
\operatorname* { m a x } _ { 2 \leq i \leq M } \| { \pmb x } _ { t _ { i } } - \widehat { { \pmb x } } _ { t _ { i } } \| = O ( \delta ^ { 2 } ) ,
$$

which completes the proof.

## F Proof of convergence order upper bound for F-DDPM (Theorem 5)

We begin with deriving the following preliminary bounds for $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t _ { i } } }$ and $\widehat { \mathbf { x } } _ { t _ { i } ^ { \prime } }$ :

$$
\mathbb { E } [ \| \pmb { x } _ { t _ { i } } \| _ { 2 } ^ { 2 } ] = O ( 1 ) , \qquad \mathbb { E } [ \| \widehat { \pmb { x } } _ { t _ { i } ^ { \prime } } \| _ { 2 } ^ { 2 } ] = O ( 1 ) .\tag{211}
$$

The proof is postponed to the end of this section.

Note that the diference between the update rules in (17a) and F-DDPM sampler (22) is that the data predictor $\mu ( x _ { t _ { i - 1 } } , t _ { i - 1 } )$ is replaced by $\mu ( \widehat { \pmb { x } } _ { t _ { i } ^ { \prime } } , t _ { i } ^ { \prime } )$ . Similar to the analysis in (72), we obtain

$$
x _ { t _ { i } } - x _ { t _ { i } } ^ { \star } = \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \mathrm { e } ^ { - \delta _ { i } } \bigl ( x _ { t _ { i - 1 } } - x _ { t _ { i - 1 } } ^ { \star } \bigr ) + \alpha _ { t _ { i } } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } ) \bigl ( \mu ( \widehat { x } _ { t _ { i } ^ { \prime } } , t _ { i } ^ { \prime } ) - \mu ( x _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } ) \bigr )
$$

$$
\begin{array} { r l } & { \quad + 2 \alpha _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i } } - \lambda ) } \big ( \mu ( x _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } ) - \mu ( x _ { t ( \lambda ) } ^ { \star } , t ( \lambda ) ) \big ) \mathrm { d } \lambda } \\ & { = \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } \mathrm { e } ^ { - \delta _ { i } } \big ( x _ { t _ { i - 1 } } - x _ { t _ { i - 1 } } ^ { \star } \big ) + \alpha _ { t _ { i } } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } ) \big ( \mu ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) - \mu ( x _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } ) \big ) } \\ & { \quad + \alpha _ { t _ { i } } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } ) \big ( \mu ( \widehat { x } _ { t _ { i } ^ { \prime } } , t _ { i } ^ { \prime } ) - \mu ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) \big ) } \\ & { \quad + 2 \alpha _ { t _ { i } } \int _ { \lambda _ { t _ { i - 1 } } } ^ { \lambda _ { t _ { i } } } \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i } } - \lambda ) } \big ( \mu ( x _ { t _ { i - 1 } } ^ { \star } , t _ { i - 1 } ) - \mu ( x _ { t ( \lambda ) } ^ { \star } , t ( \lambda ) ) \big ) \mathrm { d } \lambda } \end{array}\tag{212}
$$

Repeating arguments in Step 1 in Appendix C.1 yields

$$
x _ { t _ { i } } - x _ { t _ { i } } ^ { \star } = A _ { t _ { i } } \big ( x _ { t _ { i - 1 } } - x _ { t _ { i - 1 } } ^ { \star } \big ) + \mathcal { E } _ { t _ { i } , 1 } + \mathcal { E } _ { t _ { i } , 2 } + \alpha _ { t _ { i } } \big ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } \big ) \big ( \mu ( \widehat { x } _ { t _ { i } ^ { \prime } } , t _ { i } ^ { \prime } ) - \mu ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) \big ) ,\tag{213}
$$

where the residual terms $\mathcal { E } _ { t _ { i } , 1 }$ and $\mathcal { E } _ { t _ { i } , 2 }$ are defined in (79). By recursion, we obtain

$$
\begin{array} { r l r } {  { \mathbb { E } [ \| \boldsymbol { x } _ { t _ { M } } - \boldsymbol { x } _ { t _ { M } } ^ { \star } \| _ { 2 } ] \le \sum _ { m = 1 } ^ { M } \mathbb { E } \bigg [ \prod _ { j = m + 1 } ^ { M } \| A _ { t _ { j } } \| _ { \mathrm { o p } } \| \mathcal { E } _ { t _ { m } , 1 } \| _ { 2 } \bigg ] + \mathbb { E } \bigg [ \Big \| \sum _ { m = 1 } ^ { M } \prod _ { j = m + 1 } ^ { M } A _ { t _ { j } } \mathcal { E } _ { t _ { m } , 2 } \Big \| _ { 2 } \bigg ] } } \\ & { } & { + \mathbb { E } \bigg [ \Big \| \sum _ { m = 1 } ^ { M } \prod _ { j = m + 1 } ^ { M } A _ { t _ { j } } \alpha _ { t _ { m } } \big ( 1 - \mathrm { e } ^ { - 2 \delta _ { m } } \big ) \big ( \mu ( \widehat { x } _ { t _ { m } ^ { \prime } } , t _ { m } ^ { \prime } ) - \mu ( x _ { t _ { m - 1 } } , t _ { m - 1 } ) \big ) \Big \| _ { 2 } \bigg ] } \end{array}\tag{214}
$$

Below we shall bound these three terms separately.

• Control the second term. Note that the definition of $\mathcal { E } _ { t _ { m } , 2 }$ is independent of the update rule, and consequently the bound derived in Step 3 in Appendix C.1 still holds, i.e.,

$$
\mathbb { E } \bigg [ \Big \| \sum _ { m = 1 } ^ { M } \prod _ { j = m + 1 } ^ { M } A _ { t _ { j } } \mathscr { E } _ { t _ { m } , 2 } \Big \| _ { 2 } \bigg ] = O ( \delta ) .\tag{215}
$$

• Control the last term. Next, we turn to the bound of the last term in the right-hand-side of (214). By (84) and Assumption A2 and A4, we have

$$
\begin{array} { r l } {  { \mathbb { E } \bigg [ \Big \| \sum _ { m = 1 } ^ { M } \prod _ { j = m + 1 } ^ { M } A _ { t _ { j } } \alpha _ { t _ { m } } \big ( 1 - \mathrm { e } ^ { - 2 \delta _ { m } } \big ) \big ( \mu ( \widehat { \pmb x } _ { t _ { m } ^ { \prime } } , t _ { m } ^ { \prime } ) - \mu ( \pmb x _ { t _ { m - 1 } } , t _ { m - 1 } ) \big ) \Big \| _ { 2 } \bigg ] } \quad } & { } \\ & { \lesssim \sum _ { m = 1 } ^ { M } \delta _ { m } \mathbb { E } [ \| \widehat { \pmb x } _ { t _ { m } ^ { \prime } } - \pmb x _ { t _ { m - 1 } } \| _ { 2 } ] + \sum _ { m = 1 } ^ { M } \delta _ { m } ^ { 2 } , } \end{array}\tag{216}
$$

where we have used

$$
\begin{array} { r l } & { \| \mu ( \widehat { x } _ { t _ { m } ^ { \prime } } , t _ { m } ^ { \prime } ) - \mu ( x _ { t _ { m - 1 } } , t _ { m - 1 } ) \| _ { 2 } } \\ & { \quad \leq \| \mu ( \widehat { x } _ { t _ { m } ^ { \prime } } , t _ { m - 1 } ) - \mu ( x _ { t _ { m - 1 } } , t _ { m - 1 } ) \| _ { 2 } + \| \mu ( \widehat { x } _ { t _ { m } ^ { \prime } } , t _ { m } ^ { \prime } ) - \mu ( \widehat { x } _ { t _ { m } ^ { \prime } } , t _ { m - 1 } ) \| _ { 2 } } \\ & { \quad \leq \frac { L _ { x } } { \alpha _ { t _ { m - 1 } } } \| \widehat { x } _ { t _ { m } ^ { \prime } } - x _ { t _ { m - 1 } } \| _ { 2 } + L _ { t } ( \lambda _ { t _ { m } } + \delta _ { m } - \lambda _ { t _ { m - 1 } } ) } \\ & { \quad \leq \frac { L _ { x } } { \alpha _ { t _ { m - 1 } } } \| \widehat { x } _ { t _ { m } ^ { \prime } } - x _ { t _ { m - 1 } } \| _ { 2 } + 2 L _ { t } \delta _ { m } . } \end{array}\tag{217}
$$

According to the (26) and (27), we have

$$
\begin{array} { r l } & { \widehat { \mathbf { x } } _ { t _ { m } ^ { \prime } } = \frac { \alpha _ { t _ { m } ^ { \prime } } } { \alpha _ { t _ { m - 1 } } } x _ { t _ { m - 1 } } - \sigma _ { t _ { m } ^ { \prime } } \bigl ( \mathrm { e } ^ { 2 \delta _ { m } } - 1 \bigr ) \widetilde { \mathbf { e } } _ { t _ { m - 1 } } } \\ & { \qquad = \frac { \alpha _ { t _ { m } ^ { \prime } } } { \alpha _ { t _ { m - 1 } } } x _ { t _ { m - 1 } } - \sigma _ { t _ { m } ^ { \prime } } \bigl ( \mathrm { e } ^ { 2 \delta _ { m } } - 1 \bigr ) \biggl ( \mathrm { e } ^ { - \delta _ { m - 1 } } \varepsilon _ { \theta } \bigl ( \widehat { \mathbf { x } } _ { t _ { m - 1 } ^ { \prime } } , t _ { m - 1 } ^ { \prime } \bigr ) + \sqrt { 1 - \mathrm { e } ^ { - 2 \delta _ { m - 1 } } } z _ { m - 1 } \biggr ) } \end{array}
$$

$$
\begin{array} { r l } & { = \frac { \alpha _ { t _ { m } ^ { \prime } } } { \alpha _ { t _ { m - 1 } } } x _ { t _ { m - 1 } } - \frac { \sigma _ { t _ { m } ^ { \prime } } } { \sigma _ { t _ { m - 1 } ^ { \prime } } } \mathrm { e } ^ { - \delta _ { m - 1 } } \Bigl ( \mathrm { e } ^ { 2 \delta _ { m } } - 1 \Bigr ) \Biggl ( \widehat { x } _ { t _ { m - 1 } ^ { \prime } } - \alpha _ { t _ { m - 1 } ^ { \prime } } \mu ( \widehat { x } _ { t _ { m - 1 } ^ { \prime } } , t _ { m - 1 } ^ { \prime } ) \Biggr ) } \\ & { \quad - \sigma _ { t _ { m } ^ { \prime } } \bigl ( \mathrm { e } ^ { 2 \delta _ { m } } - 1 \bigr ) \sqrt { 1 - \mathrm { e } ^ { - 2 \delta _ { m - 1 } } } z _ { m - 1 } . } \end{array}\tag{218}
$$

Consequently, for suficiently small δ such that $\mathrm { e } ^ { 2 \delta } \leq 1 . 5$ , we have

$$
\begin{array} { r l } { \| \widehat { \pmb { x } } _ { t _ { m } ^ { \prime } } - \pmb { x } _ { t _ { m - 1 } } \| _ { 2 } \leq \bigg ( \frac { \alpha _ { t _ { m } ^ { \prime } } } { \alpha _ { t _ { m - 1 } } } - 1 \bigg ) \| \pmb { x } _ { t _ { m - 1 } } \| _ { 2 } + 3 \delta _ { m } \| \widehat { \pmb { x } } _ { t _ { m - 1 } ^ { \prime } } \| _ { 2 } + 3 \alpha _ { t _ { m } ^ { \prime } } \delta _ { m } \| \mu ( \widehat { \pmb { x } } _ { t _ { m - 1 } ^ { \prime } } , t _ { m - 1 } ^ { \prime } ) \| _ { 2 } } & { } \\ { + \sigma _ { t _ { m } ^ { \prime } } \big ( \mathrm { e } ^ { 2 \delta _ { m } } - 1 \big ) \sqrt { 1 - \mathrm { e } ^ { - 2 \delta _ { m - 1 } } } \| z _ { m - 1 } \| _ { 2 } } & { } \\ { \overset { \mathrm { ( a ) } } { \leq } 3 \delta _ { m } \| \pmb { x } _ { t _ { m - 1 } } \| _ { 2 } + 3 \delta _ { m } \| \widehat { \pmb { x } } _ { t _ { m - 1 } ^ { \prime } } \| _ { 2 } + 3 \alpha _ { t _ { m } ^ { \prime } } \delta _ { m } L _ { \mu } + 3 \sigma _ { t _ { m } ^ { \prime } } \delta _ { m } \sqrt { 3 \delta _ { m - 1 } } \| z _ { m - 1 } \| _ { 2 } . } \end{array}
$$

where (a) holds since Assumption A1 and the fact that

$$
\frac { \alpha _ { t _ { m } ^ { \prime } } } { \alpha _ { t _ { m - 1 } } } - 1 \leq \frac { \alpha _ { t _ { m } ^ { \prime } } \sigma _ { t _ { m - 1 } } } { \alpha _ { t _ { m - 1 } } \sigma _ { t _ { m } ^ { \prime } } } - 1 = \mathrm { e } ^ { 2 \delta _ { m } } - 1 \leq 3 \delta _ { m } ,
$$

Taking square and expectation further gives

$$
\begin{array} { r l } & { \mathbb { E } [ \| \widehat { \mathbf { x } } _ { t _ { m } ^ { \prime } } - { \boldsymbol { \mathbf { x } } } _ { t _ { m - 1 } } \| _ { 2 } ^ { 2 } ] \leq 3 6 \delta _ { m } ^ { 2 } \mathbb { E } [ \| { \boldsymbol { \mathbf { x } } } _ { t _ { m - 1 } } \| _ { 2 } ^ { 2 } ] + 3 6 \delta _ { m } ^ { 2 } \mathbb { E } [ \| \widehat { \mathbf { x } } _ { t _ { m - 1 } ^ { \prime } } \| _ { 2 } ^ { 2 } ] + 3 6 \alpha _ { t _ { m } ^ { \prime } } ^ { 2 } \delta _ { m } ^ { 2 } L _ { \mu } ^ { 2 } + 1 0 8 \sigma _ { t _ { m } ^ { \prime } } ^ { 2 } \delta _ { m } ^ { 2 } \delta _ { m - 1 } d } \\ & { \qquad \overset { \mathrm { ( a ) } } { = } O ( \delta _ { m } ^ { 2 } ) , \qquad ( } \end{array}\tag{219}
$$

where (a) arises from (211). While for $m = 1$ , we have

$$
\widehat { \pmb { x } } _ { t _ { 1 } ^ { \prime } } - \pmb { x } _ { t _ { 0 } } = \frac { \alpha _ { t _ { 1 } ^ { \prime } } } { \alpha _ { t _ { 0 } } } \pmb { x } _ { t _ { 0 } } - \sigma _ { t _ { 1 } ^ { \prime } } \big ( \mathrm { e } ^ { 2 \delta _ { 1 } } - 1 \big ) \widetilde { \pmb { \varepsilon } } _ { t _ { 0 } } - \pmb { x } _ { t _ { 0 } } = \Big ( \frac { \alpha _ { t _ { 1 } ^ { \prime } } } { \alpha _ { t _ { 0 } } } - \sigma _ { t _ { 1 } ^ { \prime } } \big ( \mathrm { e } ^ { 2 \delta _ { 1 } } - 1 \big ) - 1 \Big ) \pmb { x } _ { t _ { 0 } } .\tag{220}
$$

Therefore we have $\mathbb { E } [ \| \widehat { \pmb x } _ { t _ { 1 } ^ { \prime } } - { \pmb x } _ { t _ { 0 } } \| ] = O ( 1 )$ . Substituting into (216) yields

$$
\begin{array} { r l } {  { \mathbb { E } \bigg [ \Big \| \sum _ { m = 1 } ^ { M } \prod _ { j = m + 1 } ^ { M } A _ { t _ { j } } \alpha _ { t _ { m } } \big ( 1 - \mathrm { e } ^ { - 2 \delta _ { m } } \big ) \big ( \mu ( \widehat { x } _ { t _ { m } ^ { \prime } } , t _ { m } ^ { \prime } ) - \mu ( x _ { t _ { m - 1 } } , t _ { m - 1 } ) \big ) \Big \| _ { 2 } \bigg ] } \quad } & { } \\ & { \lesssim \sum _ { m = 1 } ^ { M } \delta _ { m } \sqrt { \mathbb { E } [ \| \widehat { x } _ { t _ { m } ^ { \prime } } - x _ { t _ { m - 1 } } \| _ { 2 } ^ { 2 } ] } + \sum _ { m = 1 } ^ { M } \delta _ { m } ^ { 2 } } \\ & { \lesssim \sum _ { m = 2 } ^ { M } \delta _ { m } ^ { 2 } + \delta _ { 1 } = O ( \delta ) . } \end{array}\tag{221}
$$

• Control the first term. The analysis on $\mathcal { E } _ { t _ { m } , 1 }$ established in Step 4 in Appendix C.1 relies on the conclusion

$$
\mathbb { E } [ \| \pmb { x } _ { t _ { m } } - \pmb { x } _ { t _ { m } } ^ { \star } \| _ { 2 } ^ { 2 } ] = O ( \delta ) .\tag{222}
$$

Armed with (222), the previous analysis implies that

$$
\sum _ { m = 1 } ^ { M } \mathbb { E } [ \| \mathcal { E } _ { t _ { m } , 1 } \| _ { 2 } ] = O ( \delta ) .\tag{223}
$$

The remaining proof focuses on establishing (222) for the update rule in (22). By analysis in (101), we have

$$
\begin{array} { r l } & { \| x _ { t _ { m } } - x _ { t _ { m } } ^ { \star } \| _ { 2 } \leq \frac { \sigma _ { t _ { m } } } { \sigma _ { t _ { m - 1 } } } \mathrm { e } ^ { - \delta _ { m } } \Big \| x _ { t _ { m - 1 } } - x _ { t _ { m - 1 } } ^ { \star } \Big \| _ { 2 } + \alpha _ { t _ { m } } \big ( 1 - \mathrm { e } ^ { - 2 \delta _ { m } } \big ) \Big \| \mu ( \widehat { x } _ { t _ { m } ^ { \prime } } , t _ { m } ^ { \prime } ) - \mu ( x _ { t _ { m - 1 } } ^ { \star } , t _ { m - 1 } ) \Big \| _ { 2 } + 2 B _ { t _ { m } } } \\ & { \qquad \stackrel { \mathrm { ( a ) } } { \leq } \frac { \sigma _ { t _ { m } } } { \sigma _ { t _ { m - 1 } } } \mathrm { e } ^ { - \delta _ { m } } \Big \| x _ { t _ { m - 1 } } - x _ { t _ { m - 1 } } ^ { \star } \Big \| _ { 2 } + \frac { \alpha _ { t _ { m } } } { \alpha _ { t _ { m - 1 } } } L _ { x } \big ( 1 - \mathrm { e } ^ { - 2 \delta _ { m } } \big ) \Big \| \widehat { x } _ { t _ { m } ^ { \prime } } - x _ { t _ { m - 1 } } ^ { \star } \Big \| _ { 2 } } \end{array}
$$

$$
\begin{array} { r l } & { \quad + 2 \alpha _ { t _ { m } } L _ { t } ( 1 - \mathrm { e } ^ { - 2 \delta _ { m } } ) \delta _ { m } + 2 B _ { t _ { m } } } \\ & { \leq \frac { \sigma _ { t _ { m } } } { \sigma _ { t _ { m - 1 } } } \mathrm { e } ^ { - \delta _ { m } } \left\| \boldsymbol { x } _ { t _ { m - 1 } } - \boldsymbol { x } _ { t _ { m - 1 } } ^ { \star } \right\| _ { 2 } + \frac { \alpha _ { t _ { m } } } { \alpha _ { t _ { m - 1 } } } L _ { x } ( 1 - \mathrm { e } ^ { - 2 \delta _ { m } } ) \left\| \boldsymbol { x } _ { t _ { m - 1 } } - \boldsymbol { x } _ { t _ { m - 1 } } ^ { \star } \right\| _ { 2 } } \\ & { \quad + \frac { \alpha _ { t _ { m } } } { \alpha _ { t _ { m - 1 } } } L _ { x } ( 1 - \mathrm { e } ^ { - 2 \delta _ { m } } ) \left\| \hat { \boldsymbol { x } } _ { t _ { m } ^ { \prime } } - \boldsymbol { x } _ { t _ { m - 1 } } \right\| _ { 2 } + 2 \alpha _ { t _ { m } } L _ { t } ( 1 - \mathrm { e } ^ { - 2 \delta _ { m } } ) \delta _ { m } + 2 B _ { t _ { m } } , } \end{array}\tag{224}
$$

where $B _ { t _ { m } }$ is defined in (102) and is independent of the update rule, and (a) uses (217). Note that by (219),

$$
\begin{array} { r l } & { \mathbb { E } \biggl [ \biggl ( \frac { \alpha _ { t _ { m } } } { \alpha _ { t _ { m - 1 } } } L _ { x } ( 1 - \mathrm { e } ^ { - 2 \delta _ { m } } ) \bigl \| \widehat { { \mathbf x } } _ { t _ { m } ^ { \prime } } - { \mathbf x } _ { t _ { m - 1 } } \bigr \| _ { 2 } + 2 \alpha _ { t _ { m } } L _ { t } ( 1 - \mathrm { e } ^ { - 2 \delta _ { m } } ) \delta _ { m } + 2 B _ { t _ { m } } \biggr ) ^ { 2 } \biggr ] } \\ & { \quad \lesssim \delta _ { m } ^ { 2 } \mathbb { E } \bigl [ \bigl \| \widehat { { \mathbf x } } _ { t _ { m } ^ { \prime } } - { \mathbf x } _ { t _ { m - 1 } } \bigr \| _ { 2 } ^ { 2 } \bigr ] + \delta _ { m } ^ { 4 } + \mathbb { E } [ B _ { t _ { m } } ^ { 2 } ] \stackrel { ( \mathrm { a } ) } { = } O ( \delta _ { m } ^ { 3 } ) , } \end{array}
$$

where (a) comes from (219) and the bound for the second moment of $B _ { t _ { m } }$ in (104). Repeating the analysis in (105) - (106) yields (222).

Collecting (215), (221), and (223) into (214) finally yields

$$
\mathbb { E } [ \| \pmb { x } _ { t _ { M } } - \pmb { x } _ { t _ { M } } ^ { \star } \| _ { 2 } ] \le O ( \delta ) .\tag{225a}
$$

This completes the proof.

Proof of (211). According to the update rule (22), we have

$$
\begin{array} { l } { \displaystyle x _ { t _ { i } } = \mathrm { e } ^ { - \delta _ { i } } \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } x _ { t _ { i - 1 } } + \alpha _ { t _ { i } } ( 1 - \mathrm { e } ^ { - 2 \delta _ { i } } ) \mu _ { \theta } ( \widehat x _ { t _ { i } ^ { \prime } } , l _ { i } ^ { \prime } ) + \sigma _ { t _ { i } } \sqrt { 1 - \mathrm { e } ^ { - 2 \delta _ { i } } } z _ { i } } \\ { \displaystyle \quad = \mathrm { e } ^ { - ( \lambda _ { t _ { i } } - \lambda _ { t _ { 0 } } ) } \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { 0 } } } x _ { t _ { 0 } } + \displaystyle \sum _ { m = 1 } ^ { i } \prod _ { j = m + 1 } ^ { i } \left( \mathrm { e } ^ { - \delta _ { j } } \frac { \sigma _ { t _ { j } } } { \sigma _ { t _ { j - 1 } } } \right) \left( \alpha _ { t _ { m } } ( 1 - \mathrm { e } ^ { - 2 \delta _ { m } } ) \mu _ { \theta } ( \widehat x _ { t _ { m } ^ { \prime } } , l _ { m } ^ { \prime } ) + \sigma _ { t _ { m } } \sqrt { 1 - \mathrm { e } ^ { - 2 \delta _ { m } } } z _ { m } \right) } \\  \displaystyle \quad = \mathrm { e } ^ { - ( \lambda _ { t _ { i } } - \lambda _ { t _ { 0 } } ) } \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { 0 } } } x _ { t _ { 0 } } + \sigma _ { t _ { i } } \sum _ { m = 1 } ^ { i } \mathrm { e } ^ { - ( \lambda _ { t _ { i } } - \lambda _ { t _ { m } } ) } \frac { \alpha _ { t _ { m } } } { \sigma _ { t _ { m } } } ( 1 - \mathrm { e } ^ { - 2 \delta _ { m } } ) \mu _ { \theta } ( \widehat x _ { t _ { m } ^ { \prime } } , l _ { m } ^ { \prime } ) + \sigma _  t _  i \end{array}\tag{226}
$$

Note that $z _ { m }$ are independent. Taking square and expectation yields

$$
\begin{array} { r l r } {  { \mathbb { E } [ \| \boldsymbol { x } _ { t _ { i } } \| _ { 2 } ^ { 2 } ] \leq 3 \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i } } - \lambda _ { t _ { 0 } } ) } \frac { \sigma _ { t _ { i } } ^ { 2 } } { \sigma _ { t _ { 0 } } ^ { 2 } } \mathbb { E } [ \| \boldsymbol { x } _ { t _ { 0 } } \| _ { 2 } ^ { 2 } ] + 3 \sigma _ { t _ { i } } ^ { 2 } L _ { \mu } ^ { 2 } \bigg ( \sum _ { m = 1 } ^ { i } \mathrm { e } ^ { - ( \lambda _ { t _ { i } } - \lambda _ { t _ { m } } ) } \frac { \alpha _ { t _ { m } } } { \sigma _ { t _ { m } } } ( 1 - \mathrm { e } ^ { - 2 \delta _ { m } } ) \bigg ) ^ { 2 } } } \\ & { } & { + 3 \sigma _ { t _ { i } } ^ { 2 } \displaystyle \sum _ { m = 1 } ^ { i } \mathrm { e } ^ { - 2 ( \lambda _ { t _ { i } } - \lambda _ { t _ { m } } ) } ( 1 - \mathrm { e } ^ { - 2 \delta _ { m } } ) \mathbb { E } [ \| \boldsymbol { z } _ { m } \| _ { 2 } ^ { 2 } ] } \\ & { } & { \lesssim 1 + \big ( \sum _ { m = 1 } ^ { i } \delta _ { m } \big ) ^ { 2 } + \sum _ { m = 1 } ^ { i } \delta _ { m } = O ( 1 ) . } \end{array}\tag{227}
$$

As for the expectation of $\| \widehat { \pmb { x } } _ { t _ { m } ^ { \prime } } \| _ { 2 } ^ { 2 } ,$ (218) implies that

$$
\begin{array} { r l } & { \| \widehat { x } _ { t _ { m } ^ { \prime } } \| _ { 2 } \leq \frac { \alpha _ { t _ { m - 1 } ^ { \prime } } } { \alpha _ { t _ { m - 1 } } } \| x _ { t _ { m - 1 } } \| _ { 2 } + \frac { \sigma _ { t _ { m } ^ { \prime } } ^ { \prime } } { \sigma _ { t _ { m - 1 } } ^ { \prime } } \mathrm { e } ^ { - \delta _ { m - 1 } } ( \mathrm { e } ^ { 2 \delta _ { m } } - 1 ) \| \widehat { x } _ { t _ { m - 1 } ^ { \prime } } \| _ { 2 } + \mathrm { e } ^ { - 2 \delta _ { m } + \delta _ { m - 1 } - \delta _ { m - 1 } } ( \mathrm { e } ^ { 2 \delta _ { m } } - 1 ) \alpha _ { t _ { m } ^ { \prime } } \| \mu ( \widehat { x } _ { t _ { m - 1 } ^ { \prime } } , t _ { m - 1 } ^ { \prime } ) \| _ { 2 } } \\ & { \qquad + \sigma _ { t _ { m } ^ { \prime } } ^ { \prime } ( \mathrm { e } ^ { 2 \delta _ { m } } - 1 ) \sqrt { 1 - \mathrm { e } ^ { - 2 \delta _ { m - 1 } } } \| z _ { m - 1 } \| _ { 2 } } \\ & { \leq \frac { \alpha _ { t _ { m } ^ { \prime } } } { \alpha _ { t _ { m - 1 } } } \| x _ { t _ { m - 1 } } \| _ { 2 } + 4 \delta _ { m } \| \widehat { x } _ { t _ { m - 1 } ^ { \prime } } \| _ { 2 } + 2 \alpha _ { t _ { m } ^ { \prime } } L _ { \mu } \delta _ { m } + 2 \sqrt { 2 } \sigma _ { t _ { m } ^ { \prime } } \mathrm { e } ^ { 2 \delta _ { m } } \widehat { \delta } _ { m } \sqrt { \delta _ { m - 1 } } \| z _ { m - 1 } \| _ { 2 } . } \end{array}
$$

By taking square and expectation, we have

$$
\mathbb { E } [ \Vert \widehat { \pmb { x } } _ { t _ { m } ^ { \prime } } \Vert _ { 2 } ^ { 2 } ] \le \frac { 4 \alpha _ { t _ { m } ^ { \prime } } ^ { 2 } } { \alpha _ { t _ { m - 1 } } ^ { 2 } } \mathbb { E } [ \Vert \pmb { x } _ { t _ { m - 1 } } \Vert _ { 2 } ^ { 2 } ] + 6 4 \delta _ { m } ^ { 2 } \mathbb { E } [ \Vert \widehat { \pmb { x } } _ { t _ { m - 1 } ^ { \prime } } \Vert _ { 2 } ^ { 2 } ] + 1 6 \alpha _ { t _ { m } ^ { \prime } } ^ { 2 } L _ { \mu } ^ { 2 } \delta _ { m } ^ { 2 } + 3 2 \sigma _ { t _ { m } ^ { \prime } } ^ { 2 } \mathrm { e } ^ { 4 \delta _ { m } } \delta _ { m } ^ { 2 } \delta _ { m - 1 } \mathbb { E } [ \Vert z _ { m - 1 } \Vert _ { 2 } ^ { 2 } ] .
$$

For $6 4 \delta _ { m } ^ { 2 } \leq 1 / 2$ , note that $\mathbb { E } [ \left. z _ { m - 1 } \right. _ { 2 } ^ { 2 } ] = d$ and $\mathbb { E } [ \| \pmb { x } _ { t _ { m - 1 } } \| _ { 2 } ^ { 2 } ] = O ( 1 )$ , we have

$$
\mathbb { E } [ \| \widehat { \pmb x } _ { t _ { m } ^ { \prime } } \| _ { 2 } ^ { 2 } ] \lesssim \operatorname* { m a x } _ { i \leq m - 1 } \mathbb { E } [ \| \pmb x _ { t _ { i } } \| _ { 2 } ^ { 2 } ] + O ( \delta ^ { 2 } ) = O ( 1 ) .
$$

where the last inequality uses (227).