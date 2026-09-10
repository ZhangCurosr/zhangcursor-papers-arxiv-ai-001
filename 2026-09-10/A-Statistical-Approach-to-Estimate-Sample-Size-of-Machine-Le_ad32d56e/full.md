# A Statistical Approach to Estimate Sample Size of Machine Learning Models

Dat Phan-Trong1, Sunil Gupta1, and Svetha Venkatesh1

1Deakin Applied Artificial Intelligence Initiative, Deakin University, Australia

## ABSTRACT

Objective: Sample size determination for machine learning (ML) prediction models is challenging because conventional power analysis typically requires the predictor-outcome relationship and effect structure to be specified a priori. Nonlinear ML models learn complex prediction surfaces that do not admit straightforward analytical power calculations. We propose a framework that approximates nonlinear ML models with localized linear representations and estimates sample size requirements by evaluating statistical power across these local regions.

Materials and Methods: A trained model fbase is approximated by a Rectified Linear Unit (ReLU) neural network, yielding a continuous piecewise linear representation that partitions the feature space into locally linear regions. Within each region, local effect sizes (R2, f²) are estimated and region-specific statistical power is calculated using the noncentral F-distribution for continuous outcomes and large-sample normal approximations for logistic regression. To aggregate these local quantities, we introduce a volume-weighted coverage power metric that accounts for both regional statistical power and the volume of the corresponding feature-space regions. The minimum global sample size is then determined by finding the smallest sample size that achieves a prespecified coverage threshold γ, subject to the target power β and effect-size filtering threshold $\tau _ { R ^ { 2 } }$

Results: In synthetic experiments with known structural boundaries, required sample size increased with surface complexity and decreased as weak-effect regions were excluded. Empirical evaluation on three UCl datasets demonstrated stable convergence of volume-weighted coverage to target levels and revealed differences in sample size requirements across model architectures.

Discussion: The framework decomposes nonlinear prediction surfaces into locally analyzable regions, enabling classical power calculations in settings where global parametric assumptions are inappropriate. Sample size estimates reflect regional effect heterogeneity and partition structure.

Conclusion: This approach extends traditional power analysis to nonlinear ML models and provides a structured method for estimating sample size requirements in predictive modeling studies.

Keywords: sample size estimation; machine learning; statistical power

## Introduction

Sample size determination is a fundamental component of clinical study design because it directly influences statistical power, model reliability, and resource allocation. Classical sample size calculations are typically derived from parametric statistical models, such as linear regression, where effect sizes and power can be expressed analytically through quantities such as the coefficient of determination (R2) and the global F-test ¹. These methods have been widely adopted because they provide a transparent relationship between study objectives, expected effect sizes, and required cohort size. However, although traditional parametric models can incorporate interactions and selected nonlinear effects, these relationships generally need to be specified a priori through a predefined functional form. This can limit their flexibility in biomedical applications where outcomes may depend on complex, high-order, and spatially varying interactions and nonlinear relationships among predictors.

The increasing adoption of machine learning (ML) methods in healthcare reflects their ability to capture complex relationships that may be difficult to specify a priori using traditional statistical models. Statistical models typically require predefined functional forms for main effects and interaction relationships, whereas modern ML models, including support vector machines, random forests, and neural networks, can learn complex nonlinear patterns without explicitly specifying these relationships. While this flexibility has contributed to improved predictive performance across a range of clinical tasks, it also complicates sample size determination because the learned relationships are not readily characterized within conventional analytical frameworks. Consequently, rigorous and practical methodological guidance for determining sample size in ML-based clinical studies remains limited2,3.

Several studies have attempted to address this problem using data-driven or performance-based criteria. Rajput et al.4 proposed empirical guidelines to evaluate sample adequacy by jointly analyzing feature effect sizes and classification accuracy across incremental subsamples, demonstrating that predictive performance often reaches a plateau beyond which additional data provide diminishing returns. Ghasemzadeh et al.5 addressed sample size determination by developing a simulationbased power analysis framework for machine learning models, demonstrating that robust validation schemes like nested k-fold cross-validation are essential to prevent optimistic performance bias from underestimating sample size requirements. In addition, Riley et al.6 emphasized that sample size calculations for prediction models should focus on controlling overfitting and ensuring precise risk estimation rather than relying exclusively on traditional hypothesis-testing frameworks. These criteria are important for developing reliable prediction models, but they do not directly address whether a given sample size provides adequate statistical power across the heterogeneous regions of a nonlinear prediction function. In particular, global criteria such as anticipated model complexity, predictor-to-sample ratios, or overall prediction precision do not account for the possibility that different regions of the feature space may exhibit substantially different effect sizes and therefore require different amounts of information to achieve adequate power. Our approach complements these existing criteria by evaluating statistical power locally across the learned prediction surface and aggregating it through a volume-weighted coverage measure, thereby providing a complementary perspective on sample size requirements for nonlinear ML models.

Despite these advances, most machine learning studies in clinical research continue to be limited by insufficient sample sizes and inadequate validation procedures2. A key reason is the absence of a framework that connects the flexibility of modern machine learning models with the well-established principles of statistical power analysis. Classical approaches generally require a known parametric model from which effect sizes and degrees of freedom can be derived, whereas machine learning models typically represent complex nonlinear mappings that do not admit a simple analytical form.

To address this gap, we propose a framework that approximates an arbitrary machine learning prediction surface using a neural network with Rectified Linear Unit (ReLU) activations. Because ReLU networks yield a continuous piecewise linear representation, the learned prediction surface can be decomposed into a collection of local linear regions. Within each region, conventional statistical quantities, including local coefficients of determination and F-test-based power, can be estimated using established methods. We then introduce a volumeweighted coverage power measure that aggregates these local power estimates according to the volume of their corresponding regions, providing a global measure of statistical power across the prediction surface. The required sample size is determined as the minimum sample size that achieves a prespecified global coverage-power threshold. By combining local linear power analysis with this global coverage criterion, the proposed framework provides a practical approach for determining sample size requirements for nonlinear machine learning models.

The main contributions of this work are: (1) a ReLUbased local power estimation framework, (2) a volumeweighted aggregation strategy for global sample size determination, and (3) empirical evaluation on synthetic and real-world datasets.

## Background: The Global F-Test and the Linearity Assumption

Classical sample size calculations for continuous outcomes are often based on multiple linear regression models that describe the relationship between a set of predictors $\mathbf { x } = ( x _ { 1 } , \ldots , x _ { p } )$ and an outcome variable $y \colon$

$$
y = \beta _ { 0 } + \beta _ { 1 } x _ { 1 } + \beta _ { 2 } x _ { 2 } + \cdots + \beta _ { p } x _ { p } + \epsilon ,\tag{1}
$$

where $\epsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ represents measurement noise. Within this framework, the overall predictive contribution of the covariates is commonly assessed through the global null hypothesis

$$
H _ { 0 } : \beta _ { 1 } = \beta _ { 2 } = \cdot \cdot \cdot = \beta _ { p } = 0 .\tag{2}
$$

The statistical power of the corresponding global $F _ { - }$ test depends on the strength of the relationship between the predictors and the outcome. This relationship is often summarized through Cohen's effect size $f ^ { 2 } { } _ { ; }$ which can be expressed in terms of the coefficient of determination $R ^ { 2 }$

$$
f ^ { 2 } = \frac { R ^ { 2 } } { 1 - R ^ { 2 } } .\tag{3}
$$

Given a desired significance level α and target power $( 1 - \beta )$ , the required sample size can be obtained using the non-central F distribution. This approach forms the basis of many widely used sample size calculations for regression-based studies.

A key assumption underlying this formulation is that the relationship between predictors and outcome can be adequately represented by a single global linear model.

In many biomedical applications, however, risk surfaces may exhibit complex nonlinear and locally varying relationships that are difficult to specify a priori using conventional statistical models. Under such conditions, a global linear approximation may not fully capture the structure of the underlying relationship. As a result, the estimated effect size and corresponding sample size requirements may not accurately reflect the complexity of the prediction problem.

The framework proposed in this study addresses this limitation by replacing the single global approximation with a collection of local linear approximations. Specifically, a Rectified Linear Unit (ReLU) network is used to approximate the learned prediction surface and partition the feature space into continuous piecewise linear regions. Within each region, conventional effect size estimation and F-test based power calculations can be performed using established statistical methodology. These local estimates are subsequently combined to derive a global sample size recommendation that accounts for heterogeneity in the underlying prediction surface. Rather than replacing classical power analysis, our approach seeks to extend its applicability to settings in which the underlying prediction surface is highly complex and nonlinear such as those modelled by machine learning models.

## Materials and Methods

## Study Overview

We developed a machine learning framework for estimating sample size requirements in settings where the underlying predictor-outcome relationship has a complex and potentially unknown functional form. The proposed approach is motivated by the observation that many machine learning models can represent complex prediction surfaces that are not readily amenable to conventional power analysis. Although flexible models such as Random Forests, Support Vector Machines, and neural networks can capture nonlinear interactions, their learned representations generally do not provide the explicit parametric structure required by classical sample size calculation methods.

The central idea of the proposed framework is to approximate the prediction surface of an arbitrary machine learning model using a Rectified Linear Unit (ReLU) neural network. Because ReLU networks represent continuous piecewise linear functions, the approximated prediction surface can be partitioned into a collection of local linear regions. Within each region, conventional statistical quantities, including local effect sizes and $F _ { - }$ test based power calculations, can be estimated using established regression methodology. These local estimates are then aggregated to obtain a global sample size recommendation that reflects the heterogeneous structure

of the prediction surface.

The proposed framework is evaluated in both synthetic and real-world settings. For synthetic experiments, the underlying data-generating functions were known, allowing the estimated sample size $( N _ { e s t } )$ to be compared against a reference sample size $( N _ { t r u e } )$ derived from the ground-truth function $f _ { \mathrm { t r u e } } : \mathcal { D }  \mathbb { R }$ from which pilot data D are sampled. Additional experiments on benchmark datasets were conducted to assess the framework's behavior across different machine learning architectures and data domains. The overall workflow is summarized in Algorithm 1 and schematically illustrated in Figure 1.

## Data Simulation

To evaluate the proposed framework under controlled conditions, we generated synthetic datasets designed to emulate nonlinear clinical risk surfaces. Each dataset was constructed within a bounded d-dimensional feature domain $\mathcal { D } \subset \mathbb { R } ^ { d }$

The underlying data-generating mechanism was defined by a ground-truth function $f _ { \mathrm { t r u e } } ~ : ~ \mathcal { D } ~ \to ~ \mathbb { R }$ constructed using continuous piecewise linear (CPWL) functions. Functional complexity was controlled by varying the number of knots (k) per dimension, thereby introducing structured changes in local gradients to simulate nonlinear threshold effects commonly observed in clinical phenomena.

Synthetic pilot dataset observations $\mathbf { \mathcal { D } } = \{ ( \mathbf { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ were generated by sampling feature vectors $\mathbf { x } _ { i } \sim$ $\mathcal { U } ( \mathcal { D } )$ from the domain and generating responses $y _ { i } =$ $f _ { \mathrm { t r u e } } ( \mathbf { x } _ { i } ) + \epsilon _ { i } ,$ where $\epsilon _ { i } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ( \mathbf { x } _ { i } ) )$ represents heteroscedastic noise. This design tests the proposed framework's capability to distinguish systematic structural signal from stochastic noise.

## ML-Assisted Sample Size Framework

We developed a multi-stage procedure to estimate sample size requirements for nonlinear prediction models. The framework consists of four primary components: (1) base model specification, (2) ReLU-based proxy approximation, (3) local linear power estimation, and (4) global sample size estimation.

## Base Model Specification and Reference Surface Construction

In practice, the first step of the framework assumes a user has collected pilot data $\mathbf { \mathcal { D } } ~ = ~ \{ ( \mathbf { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ (or trained a preliminary model) and fitted a prediction model $f _ { \mathrm { b a s e } } .$ Alternatively, when pilot data are simulated or provided in benchmark evaluations, the dataset $\begin{array} { r c l } { \mathcal { D } } & { = } & { \{ ( \mathbf { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n } } \end{array}$ is generated from an underlying ground-truth target function $f _ { \mathrm { t r u e } } ~ \colon ~ { \mathcal { D } } ~ \to ~ { \mathbb { R } } ~ \mathrm { a s } ~ y _ { i } ~ =$

$f _ { \mathrm { t r u e } } ( \mathbf { x } _ { i } ) + \epsilon _ { i }$ , where

$$
\begin{array} { r } { \mathcal { D } = \{ ( \mathbf { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n } , } \end{array}\tag{4}
$$

where $\mathbf { x } _ { i } \in \mathcal { D } \subset \mathbb { R } ^ { d }$ and $y _ { i } \in \mathbb { R }$

A nonlinear machine learning model $f _ { \mathrm { b a s e } } ,$ parameterized by estimate ${ \widehat { \theta } } ,$ is trained by minimizing empirical risk:

$$
\hat { \theta } = \arg \operatorname* { m i n } _ { \theta } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathcal { L } \big ( f _ { \mathrm { b a s e } } ( \mathbf { x } _ { i } ; \theta ) , y _ { i } \big ) ,
$$

where $\mathcal { L }$ denotes a task-specific loss function $( \mathrm { e . g . }$ , mean squared error for regression).

The fitted model defines a reference prediction surface

$$
f _ { \mathrm { b a s e } } ( \mathbf { x } ) = \hat { y } .\tag{5}
$$

Because this learned surface does not possess an explicit global parametric representation amenable to the forms allowed by traditional power calculators, such tools (traditional analytical power calculators) are not directly applicable. We approximate $f _ { \mathrm { b a s e } }$ via a continuous piecewise-linear function.

## ReLU Proxy Approximation

A secondary feedforward neural network with Rectified Linear Unit (ReLU) activations, denoted $f _ { \mathrm { R e L U } }$ , is trained on the synthetic proxy dataset to approximate the reference prediction surface:

$$
f _ { \mathrm { R e L U } } ( \mathbf { x } ) \approx f _ { \mathrm { b a s e } } ( \mathbf { x } ) .
$$

Because ReLU activations are continuous piecewise linear functions, $f _ { \mathrm { R e L U } }$ partitions the feature domain into a finite set of convex polyhedral linear regions. To ensure approximation fidelity and avoid geometric shattering, candidate proxy architectures are trained across multiple random initializations, and the network minimizing approximation error (measured by mean squared error) is selected.

## Local Linearization and Regional Power Allocation

As illustrated in Figure 1A, the selected ReLU network induces a polyhedral partition of the feature domain:

$$
{ \mathcal { K } } = \{ { \mathcal { R } } _ { 1 } , \ldots , { \mathcal { R } } _ { L } \} .\tag{6}
$$

Within each polyhedral region $\mathcal { R } _ { l } .$ the proxy prediction surface is strictly linear (Figure 1B):

$$
\hat { y } _ { l } = \beta _ { 0 , l } + \beta _ { 1 , l } x _ { 1 } + \cdot \cdot \cdot + \beta _ { p , l } x _ { p } .\tag{7}
$$

We inherit the polyhedral identification algorithm proposed by Gaines et al.7 to efficiently extract the hyperplane boundaries and regional coefficient vectors $\beta _ { l } =$ $( \beta _ { 1 , l } , \ldots , \beta _ { p , l } ) ^ { \top }$

For each region $\begin{array} { r } { \mathcal { R } _ { l } , } \end{array}$ we compute the local coefficient of determination $R _ { l } ^ { 2 }$ and local Cohen's effect size $f _ { l } ^ { 2 }$

$$
f _ { l } ^ { 2 } = \frac { R _ { l } ^ { 2 } } { 1 - R _ { l } ^ { 2 } } .\tag{8}
$$

To filter out negligible local signals, an active region set $\kappa _ { a } \subseteq \kappa$ is retained step-by-step based on an effectsize threshold τR2:

$$
\mathcal { K } _ { a } = \{ \mathcal { R } _ { l } \in \mathcal { K } : R _ { l } ^ { 2 } \geq \tau _ { R ^ { 2 } } \} .\tag{9}
$$

Regions with $R _ { l } ^ { 2 } < \tau _ { R ^ { 2 } }$ contribute minimal explanatory power and are excluded from the global coverage requirement to avoid mathematically inflating sample size demands for non-informative feature regions.

For a total candidate global sample size $N _ { ; }$ the expected local sample allocation assigned to region $\mathcal { R } _ { l }$ is:

$$
n _ { l } = N p _ { l } ,\tag{10}
$$

where $p _ { l } = \mathbb { P } ( \mathbf { x } \in \mathcal { R } _ { l } )$ denotes the empirical probability mass of region $\mathcal { R } _ { l }$ . Within region $\mathcal { R } _ { l }$ , local statistical power $\pi _ { l } ( n _ { l } )$ for detecting the local linear regression signal at significance level α is derived using the noncentral F-distribution:

$$
\pi _ { l } ( n _ { l } ) = 1 - F _ { p , n _ { l } - p - 1 , \lambda _ { l } } \left( F _ { p , n _ { l } - p - 1 , 1 - \alpha } \right) ,\tag{11}
$$

where $\begin{array} { r } { \lambda _ { l } \ = \ n _ { l } f _ { l } ^ { 2 } \ = \ n _ { l } \frac { R _ { l } ^ { 2 } } { 1 - R _ { l } ^ { 2 } } } \end{array}$ is the local noncentrality parameter.

## Volume-Weighted Coverage Power

To aggregate regional statistical power into a single global metric, we define volume-weighted coverage power:

$$
\operatorname { P o w e r } _ { \mathrm { v o l } } ( N ; \beta ) = \frac { \sum _ { l \in { \cal K } _ { a } } V _ { l } \mathbb { I } \big ( \pi _ { l } ( N p _ { l } ) \geq 1 - \beta \big ) } { \sum _ { l \in { \cal K } _ { a } } V _ { l } } ,\tag{12}
$$

where:

$\textstyle { \mathcal { K } } _ { a }$ is the active region set retained after filtering $\left( R _ { l } ^ { 2 } \ge \tau _ { R ^ { 2 } } \right)$ 2

$V _ { l }$ is the geometric volume of region $\mathcal { R } _ { l }$ computed via convex hull integration,

$p _ { l }$ is the empirical probability mass of region $\mathcal { R } _ { l }$

$\pi _ { l } ( n _ { l } )$ is the local statistical power in region $\mathcal { R } _ { l }$

$1 - \beta$ is the targeted local power requirement (e.g., 0.80),

• I(·) is the indicator function.

![](images/3990c16b4a3cf56454bd5e653f0074f2af2e93a9143fa08772628cf2fcea9f56.jpg)

![](images/31b1c717fd73dd42984f7b3db8852f6a600a29ee13d80308a45be3132773090b.jpg)

$$
f _ { l } ^ { 2 } = \frac { R _ { l } ^ { 2 } } { 1 - R _ { l } ^ { 2 } }
$$

Figure 1. Overview of the proposed localized sample size calculation framework in a two-dimensional feature space $( X _ { 1 } , X _ { 2 } )$ . (A) Global Polyhedral Partitioning into Linear Regions: An arbitrary nonlinear prediction surface is approximated by a continuous piecewise linear (CPWL) ReLU neural network proxy $f ( X _ { 1 } , X _ { 2 } )$ , partitioning the feature domain into convex polyhedral regions $K = \{ \mathcal { R } _ { 1 } , . . . , \mathcal { R } _ { L } \}$ with a highlighted target region $\mathcal { R } _ { k }$ . (B) Localized 2D Power Analysis in Region $\mathcal { R } _ { l } \mathbf { : }$ Within each polyhedral compartment, the proxy behaves as a local linear regression plane $\hat { y } _ { l } = \beta _ { 0 , l } + \beta _ { 1 , l } x _ { 1 } + \cdot \cdot \cdot + \beta _ { p , l } x _ { p }$ (indicated by parallel contour lines), populated by local pilot samples $\boldsymbol { x } ^ { ( i ) } \in \mathcal { R } _ { k }$ . Bottom Box: The four-stage mathematical formulation: (1) estimating local Cohen's effect size $f _ { l } ^ { 2 } = R _ { l } ^ { 2 } / ( 1 - R _ { l } ^ { 2 } )$ , (2) filtering out uninformative regions where $R _ { l } ^ { 2 } < \tau _ { R ^ { 2 } }$ to form the active set $\mathinner { \kappa _ { a } , ( 3 ) }$ evaluating regional statistical power $\pi _ { l } ( n _ { l } )$ under expected sample allocation $n _ { l } = N p _ { l }$ via the noncentral $F \cdot$ distribution, and (4) computing global volume-weighted coverage power Powe $\mathsf { r } _ { \mathsf { v o l } } ( N ; \beta )$ weighted by geometric volume $V _ { l }$

This metric measures the proportion of the active feature space volume for which the allocated sample size $n _ { l } = N p _ { l }$ achieves local power $\pi _ { l } ( n _ { l } ) \geq 1 - \beta .$ In the special case where the prediction surface is globally linear $( L = 1 )$ , this formulation collapses exactly to classical global F-test power analysis.

Local power estimates are aggregated using geometric volume $V _ { l }$ as a weighting factor to emphasize coverage of the prediction surface. Under volume weighting, a region contributes according to the physical feature space volume it occupies, regardless of its sample frequency in pilot data. This formulation is desirable in clinical modeling where low-density feature regions may represent critical high-risk subpopulations or key decision boundaries. Alternative weighting schemes (e.g., probability mass weighting or utility-weighted coverage) can also be substituted depending on clinical priorities.

## Global Sample Size Optimization

The optimal global sample size $\hat { N }$ is defined as the minimum candidate sample size required to meet or exceed a target global coverage threshold $\gamma \in ( 0 , 1 ) ( \mathrm { e . g . , } \gamma = 0 . 8 0$

or 0.90):

$$
\hat { N } = \operatorname* { m i n } \left\{ N \in \mathbb { N } : \mathrm { P o w e r } _ { \mathrm { v o l } } ( N ; \beta ) \geq \gamma \right\} .\tag{13}
$$

For each region $\mathcal { R } _ { l } ,$ local power $\pi _ { l } ( n _ { l } )$ is a monotonically increasing function of local sample size $n _ { l }$ . Because $n _ { l } = N p _ { l }$ scales linearly with $N _ { ; }$ each indicator $\mathbb { I } ( \pi _ { l } ( N p _ { l } ) \ge 1 - \beta )$ is non-decreasing in N. Consequently, global volume-weighted coverage Power $\operatorname { v o l } ( N ; \beta )$ is a non-decreasing monotonic function of $N .$ guaranteeing that the optimization problem in Eqn. 13 has a unique, well-defined minimum.

## Experimental Evaluation

## Synthetic Data Generation

To evaluate the proposed framework under controlled conditions with a known data-generating mechanism, we constructed multi-dimensional continuous piecewise linear (CPWL) functions with explicitly defined structural boundaries. This design enabled direct comparison between estimated and true local effect structures.

The generative domain was defined as a bounded compact set $\mathcal { D } \subset \mathbb { R } ^ { d }$ . The domain was partitioned into a structured hyperrectangular grid. Along each coordinate axis $j \in \{ 1 , \ldots , d \}$ , a sequence of knots $\{ \overline { { k } } _ { j , m } \}$ was defined to form an initial uniform mesh. To avoid artificial symmetry, interior knots were perturbed using bounded random displacements:

Algorithm 1 ML-Assisted Sample Size Calculation Framework   
Require: Pilot dataset $\begin{array} { r } { \mathcal { D } = \{ ( \mathbf { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n } , } \end{array}$ target local statistical power $1 - \beta ,$ target global coverage threshold $\gamma ,$   
effect size threshold $\tau _ { R ^ { 2 } }$ , base model fbase.   
Ensure: Globally optimized required sample size $\hat { N }$   
1: 1. Base Model Surface Specification   
2: Fit base model $f _ { \mathrm { b a s e } }$ on pilot data $\mathcal { D } .$   
3: 2. Proxy Approximation and Decomposition   
4: Sample dense domain grid to generate proxy dataset $\mathcal { D } _ { \mathrm { p r o x y } } = \{ ( { \bf x } _ { i } , f _ { \mathrm { b a s e } } ( { \bf x } _ { i } ) ) \} _ { i = 1 } ^ { n _ { \mathrm { p r o x y } } } .$   
5: Train ReLU network $f _ { \mathrm { R e L U } }$ to approximate $f _ { \mathrm { b a s e } } .$   
6: Decompose $f _ { \mathrm { R e L U } }$ into polyhedral linear regions $K = \{ \mathcal { R } _ { 1 } , \ldots , \mathcal { R } _ { L } \}$   
7: 3. Regional Profiling and Active Filtering   
8: Initialize active region set: $\kappa _ { a } \gets \emptyset .$   
9: for each region $\mathcal { R } _ { l } \in \mathcal { K }$ do   
10: Compute regional coefficient of determination $R _ { l } ^ { 2 }$ on $\mathcal { D } _ { \mathrm { p r o x y } }$   
11: if $R _ { l } ^ { 2 } \ge \tau _ { R ^ { 2 } }$ then   
12: Compute geometric volume $V _ { l }$ of polyhedral region $\mathcal { R } _ { l } .$   
13: Estimate empirical probability mass $p _ { l } = \mathbb { P } ( \mathbf { x } \in \mathcal { R } _ { l } )$ from pilot data $\mathcal { D } .$   
14: Update active set: $\bar { \boldsymbol { K } } _ { a } \gets \boldsymbol { \mathcal { K } } _ { a } \bar { \cup } \left\{ \mathcal { R } _ { l } \right\}$   
15: end if   
16: end for   
17: 4. Global Sample Size Optimization   
18: Define volume-weighted coverage power:   
Power $\operatorname { v o l } ( N ; \beta ) = \frac { \sum _ { l \in { \mathcal { K } } _ { a } } V _ { l } \mathbb { I } \big ( \pi _ { l } ( N p _ { l } ) \geq 1 - \beta \big ) } { \sum _ { l \in { \mathcal { K } } _ { a } } V _ { l } } .$   
19: Solve via monotonic binary search:   
$\hat { N } = \operatorname* { m i n } \left. N \in \mathbb { N } : \mathrm { P o w e r } _ { \mathrm { v o l } } ( N ; \beta ) \geq \gamma \right. .$   
20: return $\hat { N }$

$$
k _ { j , m } = \overline { { k } } _ { j , m } + \delta _ { j , m } ,\tag{14}
$$

where $\delta _ { j , m }$ was sampled from a uniform distribution with bounded support. This perturbation altered local cell geometry while preserving the global domain boundaries.

Scalar values were assigned to each point in the discretized multidimensional feature space and subsequently smoothed using a multidimensional Gaussian filter to reduce extreme local variability. Continuous piecewise linear interpolation over the perturbed grid yielded the target function

$$
f _ { \mathrm { t r u e } } : \mathcal { D }  \mathbb { R } .
$$

Synthetic observations were sampled uniformly from the domain,

$$
\mathbf { x } \sim \mathcal { U } ( \mathcal { D } ) ,
$$

and defining responses as

$$
y = f _ { \mathrm { t r u e } } ( \mathbf { x } ) + \epsilon ,\tag{15}
$$

where $\epsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ( \mathbf { x } ) )$ represents heteroscedastic noise. This design tests the proposed framework's capability to distinguish systematic structural signal from stochastic noise.

## Real-World Empirical Datasets

To assess generalizability in empirical settings, we evaluated the framework using three datasets from the UCI Machine Learning Repository representing heterogeneous biomedical and physical modeling tasks.

• Abalone Dataset $( d = 8 , n = 4 , 1 7 7 )$ : Predicts the number of shell rings (an age proxy) from physical measurements. The dataset exhibits nonlinear growth patterns and substantial feature collinearity.

• Concrete Compressive Strength Dataset $( d =$ $8 , \ n \ = \ 1 , 0 3 0 ) \colon$ Predicts compressive strength (MPa) from mixture composition and curing age. The relationship between predictors and outcome reflects nonlinear material dynamics.

• Liver Disorder Dataset $( d = 5 , n = 3 4 5 )$ : Predicts clinical risk based on liver enzyme concentrations and alcohol consumption. The dataset is characterized by moderate dimensionality and relatively high measurement variability.

These datasets differ in dimensionality, signal-tonoise characteristics, and structural smoothness, allowing evaluation across diverse nonlinear prediction regimes.

## Baseline ML models and Configuration

To evaluate robustness across model classes, we considered three commonly used nonlinear regression ML models:

• Support Vector Regression (SVR): Implemented with a radial basis function (RBF) kernel to model smooth nonlinear decision surfaces.

• Random Forest (RF) Regressor: An ensemble of 100 decision trees, enabling flexible partitioning of the feature space.

• Multi-Layer Perceptron (MLP): A fully connected feedforward neural network with ReLU activations, trained using standard backpropagation to approximate nonlinear prediction functions.

Hyperparameters were selected using standard validation procedures to ensure stable predictive performance prior to application of the sample size estimation framework.

## Results

We present the results below, evaluated across (1) controlled synthetic environments with known structural boundaries and (2) empirical benchmark datasets.

## Synthetic Evaluation with Known Structural Boundaries

Synthetic experiments were conducted under explicitly defined continuous piecewise linear (CPWL) datagenerating mechanisms. Feature dimensionality was varied $( d \in \{ 2 , 3 , 4 , 6 \} )$ , and structural complexity was controlled through knot density $( K \in \{ 2 , 5 , 7 , 1 3 \} )$ . Pilot sample sizes were set to $n = 5 0 0$ for lower-dimensional settings and $n = 5 0 0 0$ for $d = 6$ to ensure stable base model convergence prior to proxy approximation.

## ReLU Approximation Fidelity

The ReLU proxy network $f _ { \mathrm { R e L U } }$ was trained to approximate the fitted base model surface $f _ { \mathrm { b a s e } } .$ Approximation quality and stability were assessed by measuring the mean squared error (MSE) relative to $f _ { \mathrm { b a s e } }$ across 10 independent random initializations of $f _ { \mathrm { R e L U } } .$ as well as verifying low variance $( \sigma < 0 . 0 5 )$ in the number of induced polyhedral regions. Across configurations, the proxy consistently achieved stable convergence with low approximation error relative to the base surface, indicating that the induced polyhedral decomposition reflected the geometric structure of the underlying model rather than optimization artifacts.

Because the proxy serves strictly as a geometric representation of the fitted surface (rather than an independent predictive model), this step isolates the topological structure of the learned mapping while preserving its local gradients. The stability observed across random initializations confirms that downstream power estimates are not driven by proxy training instability.

## Effect Size Filtering and Regional Structure

Table 1 summarizes the number of active regions $( | K _ { a } | )$ and retained hyper-volume under varying $R ^ { 2 }$ thresholds $( \tau _ { R ^ { 2 } } ~ \in ~ 0 . 0 2 , 0 . 1 3 , 0 . 2 6 )$ , corresponding to small, medium, and large effect sizes based on Cohen's $f ^ { 2 } \in$ $\{ 0 . 0 2 , 0 . 1 5 , 0 . 3 5 \}$ , respectively. Note that regional counts $( \mathrm { e . g . , 7 5 . 4 \pm 9 . 0 ) }$ are reported as mean values with standard deviations averaged across 10 independent simulation replicates.

Across all configurations in Table 1, increasing τR² systematically reduces both the number of retained active regions and the total retained volume. This behavior is expected because regions with small $R _ { l } ^ { 2 }$ values contribute limited explanatory power and are therefore excluded from global coverage calculations.

For example, in Table 1 $( d = 2 , K = 1 3 )$ , the SVM model exhibits a reduction from $7 5 . 4 \pm 9 . 0$ active regions at $\tau _ { R ^ { 2 } } = 0 . 0 2$ to $2 3 . 9 \pm 2 . 6$ at $\tau _ { R ^ { 2 } } = 0 . 2 6$ , accompanied by a corresponding reduction in retained volume from 94.0% to 56.7%. This reflects selective exclusion of low-signal regions rather than instability in the decomposition procedure.

Importantly, the monotonic reduction in both $| { \cal { K } } _ { a } |$ and retained volume across thresholds in Table 1 confirms that the filtering mechanism behaves consistently across dimensionalities and model classes.

## Global Sample Size Optimization

Global sample size requirements $\hat { N }$ were computed to achieve volume-weighted coverage $\gamma ~ \geq ~ 0 . 9 0$ at local power $1 - \beta = 0 . 8 0$ . Detailed optimization results and empirical power convergence metrics are presented in Table 2. Three consistent patterns emerge from Table 2:

• Dependence on Effect Size Threshold: Across all configurations in Table 2, Î decreases as $\tau _ { R ^ { 2 } }$ increases. This relationship follows directly from

Table 1. Aggregated Topological Characteristics Across the Synthetic Benchmarking Matrix. The table reports the mean and standard deviation of the number of active linear regions $( | K _ { a } | )$ alongside the percentage of total domain hyper-volume retained across three sequential clinical effect size thresholds $( \tau _ { R ^ { 2 } } \in \{ 0 . 0 2 , 0 . 1 3 , 0 . 2 6 \} )$ 1
<table><tr><td colspan="3"></td><td colspan="3">Active Regions  $\overline { { ( | \mathcal { K } _ { a } | ) } }$ </td><td colspan="3">Mean Volume Retained (%)</td></tr><tr><td>Dim (d)</td><td>Knots (K)</td><td>Model</td><td> $\overline { { \tau _ { R ^ { 2 } } = 0 . 0 2 } }$ </td><td> $\overline { { \tau _ { R ^ { 2 } } = 0 . 1 3 } }$ </td><td> $\overline { { \tau _ { R ^ { 2 } } = 0 . 2 6 } }$ </td><td> $\overline { { \tau _ { R ^ { 2 } } = 0 . 0 2 } }$ </td><td> $\overline { { \tau _ { R ^ { 2 } } = 0 . 1 3 } }$ </td><td> $\overline { { \tau _ { R ^ { 2 } } } } = 0 . 2 6$ </td></tr><tr><td>2</td><td>2</td><td>RF</td><td> $\overline { { 9 . 1 \pm 1 . 8 } }$ </td><td> $5 . 8 \pm 0 . 7$ </td><td> $4 . 6 \pm 1 . 2$ </td><td> $\overline { { 9 9 . 7 \pm 0 . 3 } }$ </td><td> $\overline { { 9 7 . 0 \pm 2 . 6 } }$ </td><td> $9 2 . 8 \pm 7 . 2$ </td></tr><tr><td>2</td><td>2</td><td>SVM</td><td> $8 . 7 \pm 1 . 5$ </td><td> $5 . 9 \pm 1 . 3$ </td><td> $4 . 7 \pm 0 . 9$ </td><td> $9 9 . 0 \pm 1 . 0 $ </td><td> $9 5 . 9 \pm 3 . 4$ </td><td> $9 1 . 7 \pm 6 . 9$ </td></tr><tr><td></td><td></td><td>MLP</td><td> $9 . 7 \pm 1 . 6$ </td><td> $6 . 9 \pm 1 . 4$ </td><td> $5 . 5 \pm 1 . 1$ </td><td> $9 9 . 4 \pm 0 . 6 $ </td><td> $9 6 . 7 \pm 2 . 1$ </td><td> $9 3 . 5 \pm 3 . 1 $ </td></tr><tr><td></td><td></td><td>RF</td><td> $\overline { { 2 1 . 6 \pm 4 . 6 } }$ </td><td> $\overline { { 1 6 . 0 \pm 2 . 6 } }$ </td><td> $\overline { { 1 3 . 0 \pm 2 . 1 } }$ </td><td> $\overline { { 9 6 . 2 \pm 5 . 2 } }$ </td><td> $\overline { { 8 9 . 4 \pm 7 . 2 } }$ </td><td> $\overline { { 8 4 . 7 \pm 8 . 4 } }$ </td></tr><tr><td>2</td><td>5</td><td>SVM</td><td> $1 9 . 7 \pm 3 . 3$ </td><td> $1 5 . 0 \pm 3 . 6$ </td><td> $1 2 . 2 \pm 2 . 6$ </td><td> $9 9 . 3 \pm 0 . 8$ </td><td> $9 3 . 1 \pm 5 . 9$ </td><td> $8 8 . 1 \pm 7 . 1$ </td></tr><tr><td></td><td></td><td>MLP</td><td> $2 0 . 3 \pm 4 . 1$ </td><td> $1 4 . 8 \pm 2 . 3$ </td><td> $1 1 . 4 \pm 1 . 9$ </td><td> $9 4 . 0 \pm 5 . 1$ </td><td> $8 8 . 3 \pm 6 . 9$ </td><td> $8 0 . 3 \pm 7 . 3$ </td></tr><tr><td></td><td></td><td>RF</td><td> $\overline { { 2 8 . 2 \pm 5 . 6 } }$ </td><td> $\overline { { 2 2 . 3 \pm 3 . 7 } }$ </td><td> $\overline { { 1 8 . 8 \pm 3 . 6 } }$ </td><td> $\overline { { 9 8 . 1 \pm 1 . 4 } }$ </td><td> $\overline { { 9 2 . 4 \pm 4 . 0 } }$ </td><td> $\overline { { 8 2 . 4 \pm 6 . 8 } }$ </td></tr><tr><td>2</td><td>7</td><td>SVM</td><td> $2 8 . 3 \pm 5 . 3$ </td><td> $2 0 . 6 \pm 2 . 9$ </td><td> $1 6 . 6 \pm 2 . 1$ </td><td> $9 7 . 7 \pm 1 . 5$ </td><td> $9 0 . 6 \pm 3 . 4$ </td><td> $8 0 . 5 \pm 5 . 2$ </td></tr><tr><td></td><td></td><td>MLP</td><td> $2 9 . 5 \pm 4 . 7$ </td><td> $2 0 . 8 \pm 2 . 2$ </td><td> $1 6 . 0 \pm 1 . 9$ </td><td> $9 8 . 9 \pm 0 . 5$ </td><td> $9 3 . 3 \pm 2 . 8$ </td><td> $8 2 . 7 \pm 5 . 6$ </td></tr><tr><td>2</td><td></td><td>RF</td><td> $\overline { { 9 9 . 0 \pm 1 1 . 3 } }$ </td><td> $\overline { { 7 4 . 7 \pm 8 . 9 } }$ </td><td> $\overline { { 5 5 . 8 \pm 6 . 2 } }$ </td><td> $\overline { { 9 7 . 3 \pm 0 . 9 } }$ </td><td> $\overline { { 9 1 . 3 \pm 2 . 8 } }$ </td><td> $\overline { { 8 3 . 4 \pm 4 . 2 } }$ </td></tr><tr><td></td><td>13</td><td>SVM</td><td> $7 5 . 4 \pm 9 . 0$ </td><td> $3 5 . 5 \pm 4 . 0$ </td><td> $2 3 . 9 \pm 2 . 6$ </td><td> $9 4 . 0 \pm 2 . 7$ </td><td> $7 1 . 9 \pm 6 . 4$ </td><td> $5 6 . 7 \pm 7 . 4$ </td></tr><tr><td></td><td></td><td>MLP</td><td> $1 0 0 . 3 \pm 1 1 . 1$ </td><td> $6 9 . 7 \pm 5 . 0$ </td><td> $5 0 . 4 \pm 5 . 1$ </td><td> $9 7 . 3 \pm 1 . 0$ </td><td> $9 1 . 5 \pm 2 . 0$ </td><td> $8 2 . 7 \pm 6 . 1$ </td></tr><tr><td></td><td></td><td>RF</td><td> $\overline { { 1 7 3 . 1 \pm 2 0 . 5 } }$ </td><td> $\overline { { 1 1 6 . 4 \pm 1 5 . 6 } }$ </td><td> $\overline { { 7 6 . 3 \pm 1 1 . 1 } }$ </td><td> $\overline { { 9 4 . 8 \pm 1 . 4 } }$ </td><td> $\overline { { 7 9 . 1 \pm 3 . 4 } }$ </td><td> $\overline { { 6 2 . 5 \pm 3 . 9 } }$ </td></tr><tr><td>3</td><td>6</td><td>SVM</td><td> $1 6 9 . 9 \pm 1 9 . 7$ </td><td> $1 1 8 . 2 \pm { 1 1 . 3 }$ </td><td> $7 8 . 9 \pm 1 0 . 9$ </td><td> $9 5 . 7 \pm 1 . 1$ </td><td> $8 2 . 8 \pm 4 . 2$ </td><td> $6 5 . 7 \pm 5 . 4$ </td></tr><tr><td></td><td></td><td>MLP</td><td> $1 6 5 . 0 \pm 2 3 . 5$ </td><td> $1 2 5 . 4 \pm 1 5 . 6$ </td><td> $8 5 . 1 \pm 9 . 1$ </td><td> $9 5 . 5 \pm 1 . 2$ </td><td> $8 3 . 1 \pm 4 . 1$ </td><td> $6 6 . 0 \pm 6 . 5$ </td></tr><tr><td></td><td></td><td>RF</td><td> $\overline { { 8 9 . 9 \pm 2 7 . 9 } }$ </td><td> $\overline { { 7 0 . 7 \pm 2 1 . 8 } }$ </td><td> $\overline { { 4 2 . 8 \pm 1 2 . 3 } }$ </td><td> $9 8 . 1 \pm 0 . 8 $ </td><td> $\overline { { 8 9 . 7 \pm 2 . 9 } }$ </td><td> $6 1 . 9 \pm 9 . 0$ </td></tr><tr><td>4</td><td>4</td><td>SVM</td><td> $8 9 . 8 \pm 1 7 . 7$ </td><td> $7 5 . 1 \pm 1 4 . 0$ </td><td> $5 5 . 7 \pm 1 0 . 4$ </td><td> $9 8 . 1 \pm 0 . 9$ </td><td> $9 2 . 6 \pm 2 . 9$ </td><td> $8 0 . 1 \pm 4 . 5$ </td></tr><tr><td></td><td></td><td>MLP</td><td> $9 4 . 7 \pm 1 4 . 0$ </td><td> $8 0 . 0 \pm 9 . 8$ </td><td> $5 6 . 3 \pm 7 . 3$ </td><td> $9 8 . 3 \pm 0 . 3 $ </td><td> $9 4 . 4 \pm 2 . 3$ </td><td> $7 7 . 8 \pm 9 . 6$ </td></tr></table>

Table 2. Optimized Global Sample Size Requirements () and Empirical Power Convergence Metrics $( \mu \pm \sigma )$ accross synthetic linear piecewise benchmarking configurations.

<table><tr><td colspan="3"></td><td colspan="3">Optimized Global Sample Size Requirement Û (Power:  $\mu \pm \sigma )$ </td></tr><tr><td>Dim (d)</td><td>Knots (K)</td><td>Model</td><td> $\tau _ { R ^ { 2 } } = 0 . 0 2 \ ( \mathrm { S m a l l \ E f f e c t } )$ </td><td> $\tau _ { R ^ { 2 } } = 0 . 1 3 ~ ( \mathrm { M e d i u m ~ E f f e c t } )$ </td><td> $\tau _ { R ^ { 2 } } = 0 . 2 6 ~ ( \mathrm { L a r g e ~ E f f e c t } )$ </td></tr><tr><td>2</td><td>2</td><td>RF</td><td>252 (0.90 ± 0.08)</td><td> $\overline { { 2 0 7 \ ( 0 . 9 0 \pm 0 . 0 8 ) } }$ </td><td>156  $\overline { { ( 0 . 9 2 \pm 0 . 0 7 ) } }$ </td></tr><tr><td>2</td><td>2</td><td>SVM</td><td> $3 4 3 \ ( 0 . 9 0 \pm 0 . 0 8 )$ </td><td> $1 8 6 \ ( 0 . 9 0 \pm 0 . 0 8 )$ </td><td> $1 3 6 \ ( 0 . 9 1 \pm 0 . 0 7 ) $ </td></tr><tr><td rowspan="3">2</td><td></td><td>MLP</td><td> $2 7 7 \ ( 0 . 9 0 \pm 0 . 0 3 )$ </td><td> $2 2 2 \ ( 0 . 9 1 \pm 0 . 0 5 )$ </td><td> $1 3 6 \ ( 0 . 9 2 \pm 0 . 0 7 )$ </td></tr><tr><td></td><td>RF</td><td> $\overline { { 1 1 1 6 \ : \left( 0 . 9 0 \pm 0 . 0 7 \right) } }$ </td><td> $5 8 5 \ : ( 0 . 9 1 \pm 0 . 0 5 )$ </td><td> $3 8 5 \ : ( 0 . 9 0 \pm 0 . 0 5 )$ </td></tr><tr><td>5</td><td>SVM</td><td> $1 5 3 7 \ : ( 0 . 9 0 \pm 0 . 0 5 )$ </td><td> $6 9 1 \ ( 0 . 9 0 \pm 0 . 0 4 )$ </td><td>330 (0.90 ± 0.05)</td></tr><tr><td rowspan="3">2</td><td></td><td>MLP</td><td> $1 1 9 6 \ ( 0 . 9 0 \pm 0 . 0 4 )$ </td><td> $5 6 5 \ ( 0 . 9 0 \pm 0 . 0 5 )$ </td><td> $3 6 5 \ : ( 0 . 9 1 \pm 0 . 0 7 )$ </td></tr><tr><td></td><td>RF</td><td>1322  $\overline { { ( 0 . 9 0 \pm 0 . 0 5 ) } }$ </td><td> $\overline { { 7 2 1 \ ( 0 . 9 0 \pm 0 . 0 5 ) } }$ </td><td>480  $\overline { { ( 0 . 9 0 \pm 0 . 0 8 ) } }$ </td></tr><tr><td>7</td><td>SVM</td><td>1953  $( 0 . 9 0 \pm 0 . 0 5 ) $ </td><td> $9 0 6 \ ( 0 . 9 0 \pm 0 . 0 5 )$ </td><td>520  $( 0 . 9 1 \pm 0 . 0 6 )$ </td></tr><tr><td rowspan="3">2</td><td></td><td>MLP</td><td>1732  $( 0 . 9 0 \pm 0 . 0 4 ) $ </td><td> $1 1 1 6 \ ( 0 . 9 1 \pm 0 . 0 3 )$ </td><td>470 (0.90 ± 0.04)</td></tr><tr><td></td><td>RF</td><td>7881  $\overline { { ( 0 . 9 0 \pm 0 . 0 3 ) } }$ </td><td> $\overline { { 3 8 1 5 \ ( 0 . 9 0 \pm 0 . 0 2 ) } }$ </td><td>2075 (0.90 ± 0.02)</td></tr><tr><td>13</td><td>SVM</td><td>22598  $( 0 . 9 0 \pm 0 . 0 4 ) $ </td><td> $2 8 5 5 \ : ( 0 . 9 0 \pm 0 . 0 4 )$ </td><td>1335 (0.90 ± 0.06)</td></tr><tr><td rowspan="3">3</td><td></td><td>MLP</td><td>8376  $( 0 . 9 0 \pm 0 . 0 2 ) $ </td><td> $3 0 5 5 \ : ( 0 . 9 0 \pm 0 . 0 2 )$ </td><td>1825 (0.90 ± 0.03)</td></tr><tr><td></td><td>RF</td><td>32781  $\overline { { ( 0 . 9 0 \pm 0 . 0 2 ) } }$ </td><td> $\overline { { 1 0 9 9 0 \ ( 0 . 9 0 \pm 0 . 0 2 ) } }$ </td><td>5620 (0.90 ± 0.03)</td></tr><tr><td>6</td><td>SVM</td><td>29806  $( 0 . 9 0 \pm 0 . 0 2 ) $ </td><td> $9 9 1 0 \ ( 0 . 9 0 \pm 0 . 0 2 )$ </td><td>5320  $( 0 . 9 0 \pm 0 . 0 3 ) $ </td></tr><tr><td rowspan="3">4</td><td></td><td>MLP</td><td>23166  $( 0 . 9 0 \pm 0 . 0 3 ) $ </td><td> $1 1 1 7 5 \ ( 0 . 9 0 \pm 0 . 0 3 )$ </td><td>5685  $( 0 . 9 0 \pm 0 . 0 3 ) $ </td></tr><tr><td></td><td>RF</td><td>12249  $\overline { { ( 0 . 9 0 \pm 0 . 0 4 ) } }$ </td><td> $\overline { { 6 4 2 2 \ ( 0 . 9 0 \pm 0 . 0 5 ) } }$ </td><td>3941  $\overline { { ( 0 . 9 0 \pm 0 . 0 5 ) } }$ </td></tr><tr><td>4</td><td>SVM</td><td>9513  $( 0 . 9 0 \pm 0 . 0 4 ) $ </td><td>5711  $( 0 . 9 0 \pm 0 . 0 3 ) $ </td><td>3496  $( 0 . 9 0 \pm 0 . 0 3 ) $ </td></tr><tr><td rowspan="2"></td><td></td><td>MLP</td><td>8622  $( 0 . 9 0 \pm 0 . 0 3 ) $ </td><td>6272  $( 0 . 9 0 \pm 0 . 0 3 ) $ </td><td>3786  $( 0 . 9 0 \pm 0 . 0 3 ) $ </td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

the local noncentrality parameter,  
that even weak-signal regions satisfy the local power constraint. When these regions are excluded (e.g., $\tau _ { R ^ { 2 } } = 0 . 2 6 )$ , required sample sizes decrease accordingly.

$$
\lambda _ { l } = n _ { l } \frac { R _ { l } ^ { 2 } } { 1 - R _ { l } ^ { 2 } } = n _ { l } f _ { l } ^ { 2 } .\tag{16}
$$

Regions with small $R _ { l } ^ { 2 }$ require disproportionately large local allocation $n _ { l } ~ = ~ N p _ { l }$ to achieve local power $\pi _ { l } ( n _ { l } ) ~ \ge ~ 1 - \beta .$ When such regions are retained $( \mathrm { e . g . } , \ \tau _ { R ^ { 2 } } \ = \ 0 . 0 2 )$ , the global optimization must allocate sufficient total sample size N so

For instance, in the $d = 2 , K = 2 \mathrm { ~ R F }$ configuration in Table $2 , { \hat { N } }$ decreases from 252 to 156 when moving from $\tau _ { R ^ { 2 } } = 0 . 0 2$ to $\tau _ { R ^ { 2 } } = 0 . 2 6$ This magnitude of reduction is consistent with the nonlinear dependence of $\lambda _ { l }$ on $R _ { l } ^ { 2 }$

Extremely large requirements observed in highly fragmented settings (e.g., $\hat { N } ~ = ~ 2 2 , 5 9 8$ for $d \digamma =$ 2, K = 13, SVM, $\tau _ { R ^ { 2 } } = 0 . 0 2$ in Table 2) arise when many low-effect regions are retained. In such cases, the optimization must ensure adequate power simultaneously across a large number of small-probability regions, which mathematically inflates the global allocation. These values reflect the strict coverage constraint $( \gamma \ge 0 . 9 0 )$ rather than numerical instability.

## • Structural Complexity and Sample Size

For fixed τR2, increasing knot density K increases Û (Table 2). For example, under $\tau _ { R ^ { 2 } } = 0 . 2 6$ for RF at d = 2, N increases from 156 (K = 2) to 2,075 $( K = 1 3 )$

This pattern reflects growth in the number of active regions and corresponding fragmentation of probability mass pl. As K increases, the expected allocation $n _ { l } = N p _ { l }$ becomes smaller for each region unless total N increases proportionally. Thus, higher structural complexity requires larger total sample sizes to maintain local power constraints across active compartments.

## • Architectural Differences

Differences across RF, SVM, and MLP models in Table 2 are attributable to how each architecture partitions the feature space. Tree-based models tend to induce localized hyperrectangular partitions, while kernel-based or neural architectures produce smoother polyhedral decompositions.

These architectural differences affect: (a) the number of active regions $| { \cal { K } } _ { a } | .$ (b) the distribution of regional probability mass pl, and (c) the retained volume under filtering. Importantly, observed differences in $\hat { N }$ across architectures reflect structural characteristics of the learned surfaces rather than differences in optimization tolerance. Across all configurations in Table 2, empirical coverage at $\hat { N }$ consistently converges to target coverage $( \approx 0 . 9 0 )$ with small variance.

## Power Curve Dynamics

Figure 2 illustrates representative volume-weighted coverage trajectories as a function of total sample size N. In all cases, coverage increases monotonically with $N ,$ as expected from the monotonicity of the noncentral $F _ { - }$ distribution with respect to $\lambda _ { l }$

Increasing τR² shifts the coverage curve leftward, reflecting exclusion of low-effect regions. The smooth trajectories and narrow variance bands across proxy initializations confirm that global coverage estimates are numerically stable with respect to proxy approximation variability.

## Real-World Empirical Evaluation on UCI Datasets

We evaluated the framework on three benchmark datasets from the UCI Machine Learning Repository: Abalone $( d = 8 )$ , Concrete Compressive Strength $( d =$ 8), and BUPA Liver Disorders $( d \ : = \ : 5 )$ Unlike the synthetic experiments, real-world data introduce two methodological considerations.

First, no analytical ground truth function $f _ { \mathrm { t r u e } }$ is available. Accordingly, the framework operates as a post hoc structural diagnostic applied to fitted prediction models rather than as a validation against a known generating mechanism.

Second, uniform grid sampling is inappropriate in empirical datasets due to non-uniform feature distributions and sparsity in high-dimensional space. To approximate the empirical feature manifold while avoiding extrapolation into empty regions, we estimated a Gaussian Mixture Model (GMM) over the observed feature distribution and sampled candidate points from this fitted density. These samples were used to approximate regional probability masses and compute volume-weighted coverage.

For empirical experiments, the global coverage threshold was fixed at $\gamma = 0 . 8 0$ , and local power was set to $\beta = 0 . 8 0$ . Optimized sample size estimates $\hat { N }$ and corresponding empirical coverage values are reported in Table 3.

## Dataset-Specific Observations

Across datasets, three distinct patterns were observed.

## • Concrete Compressive Strength

For the Concrete Strength dataset, Î remains invariant across all effect size thresholds for each model (e.g., RF: 11,611; SVM: 12,412; MLP: 6,206). This indicates that, under the fitted models, few regions exhibit sufficiently small $R _ { l } ^ { 2 }$ values to be excluded by increasing τR2. In other words, regional explanatory strength appears relatively homogeneous across the empirical feature manifold.

Differences across architectures are attributable to variations in how each model partitions feature space and distributes regional probability mass. For example, the MLP yields smaller N than RF and SVM in this dataset, reflecting differences in induced regional fragmentation and corresponding local allocations $n _ { l } = N p _ { l }$ These differences arise from structural characteristics of the learned surfaces rather than from changes in the global coverage constraint.

## • BUPA Liver Disorders

In the Liver Disorder dataset, $\hat { N }$ decreases with increasing $\tau _ { R ^ { 2 } }$ for RF and SVM models (e.g., SVM:

MLP  
SVM  
A. Small Effect Threshold $( \tau _ { R ^ { 2 } } = 0 . 0 2 )$  
![](images/6fce1ad840162adf5aea243beb91c5cd99e9325b12d7e74123ab72a3bb5b130f.jpg)

![](images/2b042f4c62464a45618769c8995cb8523dd523c5d30fb19055aefe9aea41135d.jpg)

![](images/8ca5d9bf6973ba3e812751455c15c23e62d1c974cf562a3f84c44ecd4ed2e3c3.jpg)

B. Medium Effect Threshold $( \tau _ { R ^ { 2 } } = 0 . 1 3 )$  
![](images/ce8f5b52c62bd37cc119ed5ef97a9ac44e76766f21a0a9c95bf777640e943c49.jpg)

![](images/baf91db7c47e460dd09af28791e9764b4518e79b9501db194d43ce610802d5cc.jpg)

![](images/9151e829492b177a867976b45388a637b9f2854c11b114fb88808b8554e7d06a.jpg)

C. Large Effect Threshold $( \tau _ { R ^ { 2 } } = 0 . 2 6 )$  
![](images/0f5004ca1650ae5da045330efbca65c9d54477ecbdc254bc280102df85541912.jpg)

![](images/67c1ab7bda0678003caf2c9e42f5e5aae40332420212a7d89b516f39a29dfdfe.jpg)

![](images/c4329d966cf5204828e7e66f9d163421f97ba01b04b19cb11e591003db01e09d.jpg)  
Figure 2. Volume-weighted coverage power trajectories as a function of total cohort sample size (N) for synthetic benchmarking experiments $( d = 2 , K = 5 )$ . Columns represent the three baseline machine learning architectures: Random Forest (RF, left), Support Vector Machine (SVM, center), and Multi-Layer Perceptron (MLP, right). Rows correspond to increasing regional effect-size filtering thresholds: (A) $\tau _ { R ^ { 2 } } = 0 . 0 2$ (small effect size), (B) $\tau _ { R ^ { 2 } } = 0 . 1 3$ (medium effect size), and (C) $\tau _ { R ^ { 2 } } = 0 . 2 6$ (large effect size). In each panel, solid dark-blue curves depict the estimated coverage obtained via the localized ReLU proxy framework (with shaded 95% empirical variance bands across 10 random proxy initializations), solid dark-red curves denote the ground-truth piecewise linear (CPWL) reference, and dashed horizontal lines mark the target 90% global coverage threshold $( \gamma = 0 . 9 0 )$ $\mathsf { A s } \tau _ { R ^ { 2 } }$ increases, low-signal regions are excluded from the active partition $\kappa _ { a } .$ shifting coverage curves leftward and reducing the required global sample size $\hat { N }$ while maintaining rigorous coverage fidelity.

42,642 to 20,620). This behavior mirrors the synthetic experiments: exclusion of low-effect regions reduces the need to simultaneously power multiple weak-signal compartments.

The magnitude of $\hat { N }$ relative to the original dataset size reflects the strict requirement that 80% of the empirical feature manifold achieve local power $\ge 0 . 8 0$ . In fragmented surfaces with heterogeneous regional signal strength, satisfying this constraint may require substantial global allocation. These values therefore quantify structural heterogeneity under the fitted model rather than implying infeasibility of the modeling task.

The MLP exhibits minimal variation across thresholds (30,830 to 30,630), suggesting that few regions fall below even the strictest $\tau _ { R ^ { 2 } }$ threshold. This indicates comparatively stable regional effect sizes under that architecture.

## • Abalone

In the Abalone dataset, architectural differences are more pronounced. The RF model produces constant Ñ = 8, 008 across thresholds, indicating limited sensitivity to effect-size filtering under the induced partition.

By contrast, the SVM model exhibits a reduction from 49,649 to 38,638 as τR2 increases, suggesting the presence of multiple low-effect regions that are excluded under stricter thresholds. The MLP produces consistently large  values (approximately 50,000), reflecting substantial fragmentation and/or diffuse regional probability mass under that architecture.

These differences underscore that required sample size is a joint function of (i) the fitted surface geometry, (ii) regional probability mass distribution, and (iii) the imposed global coverage constraint.

## Stability of Empirical Coverage Estimates

Across all datasets and architectures, empirical coverage at Ñ converges to approximately 0.80 with relatively small variance (Table 3). This consistency indicates that the optimization routine reliably identifies the crossing point where volume-weighted coverage meets the specified threshold.

Because empirical integration relies on GMM-sampled feature densities, minor variability reflects stochastic sampling rather than instability in the optimization procedure. The observed variance bands remain narrow relative to the coverage threshold, supporting numerical stability of the estimation process.

## Limitations and Future Work

• Dependence on Feature Density Estimation: In empirical settings without a known datagenerating mechanism, volume-weighted integration relies on sampling from an estimated feature density. In this study, a Gaussian Mixture Model (GMM) was used to approximate the empirical feature distribution. Although this approach reduces extrapolation into unsupported regions of the feature space, results may be sensitive to density misspecification. Rare but structurally important subregions may be underrepresented if not adequately captured by the mixture model. Future work will investigate alternative density estimators, including nonparametric and adaptive sampling strategies, to improve robustness of regional probability mass estimation.

• Scalability in High Dimensions: The proxybased decomposition relies on the piecewise linear structure of ReLU networks. As dimensionality increases, the number of induced linear regions may grow rapidly with network depth and width. This expansion increases both memory and computational demands during regional enumeration and volume integration. Practical deployment in very high-dimensional settings may therefore require architectural constraints, dimensionality reduction, or region-clustering approximations to maintain tractable optimization.

• Extension to Classification and Alternative Outcome Models: The primary formulation presented in the main text anchors local power estimation on continuous outcomes using classical multiple regression assumptions and noncentral F-test mechanics. However, many biomedical and clinical applications involve binary classification outcomes (e.g., disease diagnosis, 30-day readmission, or treatment response)

To accommodate binary classification tasks, the localized power framework extends directly to multiple logistic regression models by substituting largesample normal approximations for regional log-odds coefficients. Within each induced polyhedral region Rl, the proxy network defines a locally linear log-odds surface, enabling local sample size requirements to be calculated via regional variance inflation factors (VIFl) and regional event probabilities (Pi). Monotonic volume-weighted coverage optimization is then applied analogously to derive the global sample size N. A complete mathematical derivation and empirical validation matrix for logistic regression classification tasks are detailed in the Supplementary Material (Appendix S1, Table 4). Extending localized power mapping to timeto-event endpoints (Cox survival models) and correlated longitudinal structures represents an active direction for future research.

## Conclusion

We present a framework for estimating sample size requirements for nonlinear machine learning prediction models by leveraging the continuous piecewise linear structure of ReLU networks. By decomposing complex prediction surfaces into locally linear regions, the approach enables region-specific power calculations that are subsequently aggregated through a volume-weighted coverage criterion.

Across synthetic and empirical evaluations, the framework produces internally consistent sample size estimates that reflect structural characteristics of the fitted prediction surface, including effect size heterogeneity and regional fragmentation. The method also provides insight into how model architecture influences regional partitioning and corresponding sample size requirements.

Table 3. Optimized Global Sample Size Requirements $( \hat { N } )$ and Empirical Power Convergence Metrics $( \mu \pm \sigma )$ across UCI Repository Manifolds
<table><tr><td colspan="5">Optimized Global Sample Size Requirement Ñ (Power:</td></tr><tr><td>Dataset</td><td>Dim (d)</td><td>Model</td><td> $\overline { { \tau _ { R ^ { 2 } } = 0 . 0 2 } }$  (Small Effect)</td><td> $\overline { { \tau _ { R ^ { 2 } } = 0 . 1 3 } }$  (Medium Effect)</td><td> $\mu \pm \sigma )$   $\overline { { \tau _ { R ^ { 2 } } = 0 . 2 6 } }$  (Large Effect)</td></tr><tr><td rowspan="4">Liver Disorder</td><td></td><td>RF</td><td> $\overline { { 4 3 8 4 3 \ ( 0 . 8 0 \pm 0 . 0 5 ) } }$ </td><td>39639  $\overline { { ( 0 . 8 0 \pm 0 . 0 5 ) } }$ </td><td>35235  $\overline { { ( 0 . 8 0 \pm 0 . 0 5 ) } }$ </td></tr><tr><td>5</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2"></td><td>SVM</td><td>42642  $( 0 . 8 0 \pm 0 . 0 3 ) $ </td><td>29629  $( 0 . 8 0 \pm 0 . 0 3 ) $ </td><td>20620  $( 0 . 8 0 \pm 0 . 0 3 ) $ </td></tr><tr><td>MLP</td><td>30830  $( 0 . 8 0 \pm 0 . 0 3 ) $ </td><td>30830  $( 0 . 8 0 \pm 0 . 0 3 ) $ </td><td>30630  $( 0 . 8 0 \pm 0 . 0 3 ) $ </td></tr><tr><td rowspan="4">Concrete Strength</td><td rowspan="4">8</td><td>RF</td><td>11611  $\overline { { ( 0 . 8 0 \pm 0 . 0 3 ) } }$ </td><td>11611  $\overline { { ( 0 . 8 0 \pm 0 . 0 3 ) } }$ </td><td>11611  $\overline { { ( 0 . 8 0 \pm 0 . 0 3 ) } }$ </td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>SVM</td><td>12412  $( 0 . 8 0 \pm 0 . 0 5 ) $ </td><td>12412  $( 0 . 8 0 \pm 0 . 0 5 ) $ </td><td>12412  $( 0 . 8 1 \pm 0 . 0 5 ) $ </td></tr><tr><td>MLP</td><td>6206  $( 0 . 8 0 \pm 0 . 0 2 ) $ </td><td>6206  $( 0 . 8 0 \pm 0 . 0 2 ) $ </td><td>6206  $( 0 . 8 0 \pm 0 . 0 2 ) $ </td></tr><tr><td rowspan="3">Abalone</td><td rowspan="3">8</td><td>RF</td><td>8008  $\overline { { ( 0 . 8 0 \pm 0 . 0 9 ) } }$ </td><td>8008  $\overline { { ( 0 . 8 0 \pm 0 . 0 9 ) } }$ </td><td>8008  $\overline { { ( 0 . 8 1 \pm 0 . 0 9 ) } }$ </td></tr><tr><td>SVM</td><td>49649  $( 0 . 8 0 \pm 0 . 1 1 ) $ </td><td>38638  $( 0 . 8 1 \pm 0 . 1 0 ) $ </td><td>38638  $( 0 . 8 1 \pm 0 . 1 0 ) $ </td></tr><tr><td>MLP</td><td>50050  $( 0 . 8 0 \pm 0 . 0 8 ) $ </td><td>49649  $( 0 . 8 0 \pm 0 . 0 8 ) $ </td><td>49649  $( 0 . 8 0 \pm 0 . 0 8 ) $ </td></tr></table>

5. Ghasemzadeh H, Hillman RE, Mehta DD. Toward generalizable machine learning models in speech, language, and hearing sciences: Estimating sample size and reducing overfitting. Journal of Speech, Language, and Hearing Research. 2024;67(3):753-81.

Rather than replacing classical power analysis, this approach extends its applicability to nonlinear predictive settings where global parametric assumptions are not appropriate. With further refinement of density estimation and scalability components, the framework may serve as a complementary tool for study planning in settings involving complex machine learning models.

## Funding

This research was partially supported by the Wellcome Trust [grant number: 303030, $/ \mathrm { Z } / 2 3 / \mathrm { Z } ]$ , the National Health and Medical Research Council (NHMRC) Centre of Research Excellence in Depression Treatment Precision (grant number: 2024796) and the NHMRC Synergy Grant (grant number: 2026505).

## References

1. Cohen J. Statistical power analysis for the behavioral sciences. routledge; 2013.

2. Balki I, Amirabadi A, Levman J, Martel AL, Emersic Z, Meden B, et al. Sample-size determination methodologies for machine learning in medical imaging research: a systematic review. Canadian Association of Radiologists Journal. 2019;70(4):344-53.

3. Figueroa RL, Zeng-Treitler Q, Kandula S, Ngo LH. Predicting sample size required for classification performance. BMC medical informatics and decision making. 2012;12(1):8.

4. Rajput D, Wang WJ, Chen CC. Evaluation of a decided sample size in machine learning applications. BMC bioinformatics. 2023;24(1):48.

6. Riley RD, Ensor J, Snell KI, Harrell FE, Martin GP, Reitsma JB, et al. Calculating the sample size required for developing a clinical prediction model. Bmj. 2020;368.

7. Gaines BB, Bi J. Characterizing the Discrete Geometry of ReLU Networks. In: The Fourteenth International Conference on Learning Representations; 2026.

8. Hsieh FY, Bloch DA, Larsen MD. A simple method of sample size calculation for linear and logistic regression. Statistics in medicine. 1998;17(14):1623-34.

## Supplementary Material

## Extension to Multiple Logistic Regression Tasks

To extend the proposed framework beyond continuous outcomes, we outline an adaptation for multiple logistic regression models with binary responses $Y \in \{ 0 , 1 \}$ . In many clinical applications, interest centers on estimating the effect of a target covariate $X _ { 1 }$ while adjusting for $p - 1$ additional predictors. Classical sample size calculations for multivariable logistic regression are typically derived using large-sample normal approximations and may require iterative procedures or simplifying assumptions8.

Leveraging the continuous piecewise linear (CPWL) proxy network, the feature space is partitioned into polyhedral regions K. Within each region $l ,$ the proxy induces a locally linear representation of the log-odds surface. This localized structure enables region-specific application of large-sample approximations for logistic regression power calculations.

## Localized Sample Size Estimation

Within a given active region $l \in \kappa$ , consider estimation of the coefficient associated with $X _ { 1 }$ . Following the normal approximation described by Hsieh et al.8, the baseline sample size required to detect a log-odds coefficient $\beta _ { 1 , l } ^ { * }$ with two-sided significance level α and target power $1 - \beta$ is

$$
n _ { 1 , l } = \frac { ( Z _ { 1 - \alpha / 2 } + Z _ { 1 - \beta } ) ^ { 2 } } { P _ { l } ( 1 - P _ { l } ) \beta _ { 1 , l } ^ { * 2 } } ,\tag{17}
$$

where $Z _ { 1 - \alpha / 2 }$ and $Z _ { 1 - \beta }$ denote standard normal quantiles, $\beta _ { 1 , l } ^ { * }$ is the localized log-odds slope within region $l ,$ and $P _ { l }$ is the regional event probability.

To account for correlation between $X _ { 1 }$ and the remaining covariates $( X _ { 2 } , \ldots , X _ { p } )$ , the variance of the estimated coefficient is inflated by a regional variance inflation factor (VIF)8:

$$
\mathrm { V I F } _ { l } = \frac { 1 } { 1 - \rho _ { 1 . 2 3 . . . p , l } ^ { 2 } } ,\tag{18}
$$

where $\rho _ { 1 . 2 3 . . . p , l } ^ { 2 }$ is the squared multiple correlation coefficient obtained by regressing $X _ { 1 }$ on $( X _ { 2 } , \ldots , X _ { p } )$ within region l.

Operationally, $\rho _ { 1 . 2 3 . . . p , l } ^ { 2 }$ is estimated via localized ordinary least squares (OLS). Let SSRl denote the sum of squared residuals from regressing $X _ { 1 }$ on the remaining covariates within region $l ,$ and let SSTi denote the corresponding total sum of squares. Then,

$$
\rho _ { 1 . 2 3 . . . p , l } ^ { 2 } = 1 - \frac { \mathrm { S S R } _ { l } } { \mathrm { S S T } _ { l } } .\tag{19}
$$

The multivariable regional sample size requirement becomes

$$
n _ { p , l } = n _ { 1 , l } \cdot \mathrm { V I F } _ { l } = \frac { ( Z _ { 1 - \alpha / 2 } + Z _ { 1 - \beta } ) ^ { 2 } } { ( 1 - \rho _ { 1 . 2 3 . . . p , l } ^ { 2 } ) P _ { l } ( 1 - P _ { l } ) \beta _ { 1 , l } ^ { * 2 } } .\tag{20}
$$

These expressions rely on standard large-sample approximations for logistic regression and are therefore most appropriate when regional sample allocations are not extremely small.

## Local and Global Power Aggregation

Let N denote a candidate global sample size. Under the empirical feature distribution, each region l has probability mass $p _ { l } = \mathbb { P } ( \mathbf { x } \in \mathcal { R } _ { l } )$ . The expected allocation within region l is $N _ { l } = N p _ { l }$

Using the normal approximation, the localized statistical power for detecting $\beta _ { 1 , l } ^ { * }$ can be expressed as

$$
\pi _ { l } ( N ) = \Phi \left( \sqrt { \frac { N p _ { l } } { \mathrm { V I F } _ { l } } P _ { l } ( 1 - P _ { l } ) \beta _ { 1 , l } ^ { * 2 } } - Z _ { 1 - \alpha / 2 } \right) ,\tag{21}
$$

where $\Phi ( \cdot )$ denotes the standard normal cumulative distribution function.

A region l is considered adequately powered if $\pi _ { l } ( N ) \geq 1 - \beta$ . Global coverage is then defined analogously to the regression case:

$$
\mathrm { P o w e r } _ { \mathrm { v o l } } ( N ; \beta ) = \frac { \sum _ { l \in { \mathcal { K } } } w _ { l } \mathbb { I } ( \pi _ { l } ( N ) \geq 1 - \beta ) } { \sum _ { l \in { \mathcal { K } } } w _ { l } } ,\tag{22}
$$

where $w _ { l }$ denotes the geometric volume of region l and I(·) is an indicator function.

The required global sample size is defined as

$$
\hat { N } = \operatorname* { m i n } \left. N \in \mathbb { N } \mid \mathrm { P o w e r } _ { \mathrm { v o l } } ( N ; \beta ) \geq \gamma \right. .\tag{23}
$$

This extension preserves the localized decomposition and volume-weighted aggregation structure of the regression framework while adapting the power calculation to large-sample logistic regression theory.

Table 4. Optimized Global Sample Size Requirements $( \hat { N } )$ and Empirical Power Convergence Metrics $( \mu \pm \sigma )$ for Logistic Regression across Synthetic Linear Piecewise Benchmarkings.
<table><tr><td rowspan="2">Dim (d) Knots (K)</td><td rowspan="2"></td><td rowspan="2">Model</td><td colspan="3">Optimized Global Sample Size Requirement  $\hat { N }$  [Power:  $\mu \pm \sigma ]$ </td></tr><tr><td> $\overline { { \tau _ { R ^ { 2 } } = 0 . 0 2 \ ( \mathrm { S m a l l \ E f f e c t } ) } }$ </td><td> $\tau _ { R ^ { 2 } } = 0 . 1 3 ~ ( \mathrm { M e d i u m ~ E f f e c t } )$ </td><td> $\overline { { \tau _ { R ^ { 2 } } = 0 . 2 6 } }$  (Large Effect)</td></tr><tr><td rowspan="3">2</td><td rowspan="3">5</td><td>RF</td><td> $\overline { { 1 6 1 5 \ ( 0 . 8 3 \pm 0 . 0 4 ) } }$ </td><td> $\overline { { 1 6 1 5 \ : \left( 0 . 8 5 \pm 0 . 0 7 \right) } }$ </td><td> $\overline { { 1 5 5 5 \ ( 0 . 8 0 \pm 0 . 1 1 ) } }$ </td></tr><tr><td>SVM</td><td>1245  $( 0 . 8 1 \pm 0 . 0 8 ) $ </td><td> $1 2 4 5 \ ( 0 . 8 1 \pm 0 . 0 8 )$ </td><td> $1 2 4 5 \ ( 0 . 8 1 \pm 0 . 0 8 )$ </td></tr><tr><td>MLP</td><td> $3 2 4 6 \ ( 0 . 8 0 \pm 0 . 1 0 )$ </td><td> $1 4 3 5 \ ( 0 . 8 3 \pm 0 . 0 8 )$ </td><td> $1 1 7 5 \ : ( 0 . 8 1 \pm 0 . 1 7 )$ </td></tr><tr><td rowspan="3">2</td><td rowspan="3">7</td><td>RF</td><td> $\overline { { 2 0 5 6 \ ( 0 . 8 0 \pm 0 . 0 8 ) } }$ </td><td> $\overline { { 1 8 2 0 \ ( 0 . 8 1 \pm 0 . 0 5 ) } }$ </td><td> $\overline { { 1 5 9 5 \ ( 0 . 8 1 \pm 0 . 0 9 ) } }$ </td></tr><tr><td>SVM</td><td>4532  $( 0 . 8 0 \pm 0 . 1 4 ) $ </td><td> $4 5 3 2 \ ( 0 . 8 0 \pm 0 . 1 4 )$ </td><td> $1 9 2 0 \ ( 0 . 8 0 \pm 0 . 0 8 )$ </td></tr><tr><td>MLP</td><td>1725  $( 0 . 8 3 \pm 0 . 0 9 ) $ </td><td>1545  $( 0 . 8 1 \pm 0 . 1 1 )$ </td><td> $1 1 2 5 \ : ( 0 . 8 1 \pm 0 . 0 9 )$ </td></tr><tr><td rowspan="3">2</td><td rowspan="3">13</td><td>RF</td><td>555  $\overline { { ( 0 . 8 1 \pm 0 . 1 7 ) } }$ </td><td> $\overline { { 4 2 0 \ ( 0 . 8 0 \pm 0 . 0 5 ) } }$ </td><td>420  $\overline { { ( 0 . 8 0 \pm 0 . 0 5 ) } }$ </td></tr><tr><td>SVM</td><td>845  $( 0 . 8 1 \pm 0 . 1 1 )$ </td><td>845  $( 0 . 8 1 \pm 0 . 1 1 )$ </td><td>695  $( 0 . 8 1 \pm 0 . 1 2 ) $ </td></tr><tr><td>MLP</td><td>690  $( 0 . 8 0 \pm 0 . 0 7 ) $ </td><td>690  $( 0 . 8 1 \pm 0 . 0 6 ) $ </td><td>655  $( 0 . 8 0 \pm 0 . 0 8 ) $ </td></tr></table>