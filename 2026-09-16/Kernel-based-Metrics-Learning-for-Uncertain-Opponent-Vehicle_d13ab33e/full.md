# Kernel-based Metrics Learning for Uncertain Opponent Vehicle Trajectory Prediction in Autonomous Racing

Hojin Lee, Youngim Nam, Sanghun Lee, and Cheolhyeon Kwon

Abstract—Autonomous racing confronts significant challenges in safely overtaking Opponent Vehicles (OVs) that exhibit uncertain trajectories, stemming from unknown driving policies. To address these challenges, this study proposes heterogeneous kernel metrics for Deep Kernel Learning (DKL), designed to robustly capture the diverse driving policies of OVs, and carry out precise trajectory predictions along with the associated uncertainties. A key virtue of the proposed kernel metrics lies in their ability to align similar driving policies and disjoin dissimilar ones in an unsupervised manner, given the observed interactions between the Ego Vehicle (EV) and OVs. The efficacy of the proposed method is substantiated through experimental studies on a 1/10th scale racecar platform, demonstrating improved prediction accuracy and thereby safely overtaking against OVs. Furthermore, our method is computationally efficient for onboard computing units, affirming its viability in fast-paced racing environments. The video and source code can be found at https://github.com/ HMCL-UNIST/OpponentPredictionWithKMDKL.git.

Index Terms—Planning under Uncertainty, Integrated Planning and Learning, Machine Learning for Robot Control

## I. INTRODUCTION

UTONOMOUS racing has emerged as a significant subfield of autonomous driving, attracting considerable interest and fostering competitions such as Roborace, Indy Autonomous Challenge, and F1TENTH [1]. One of the key challenges in autonomous racing lies in safely running against OVs and executing overtaking maneuvers. Various strategies have been developed to address these challenges [2], [3], among which a commonly adopted approach involves predicting the future trajectory of OV and then planning the overtaking maneuver accordingly [1].

Although previous studies have made strides in trajectory prediction for autonomous racing, they still face challenges in addressing diverse driving policies of OVs [4]. On the one hand, physics-based prediction methods are grounded in firstprinciples dynamics models. They are capable of anticipating momentary behavior but struggle to capture the long-term interactions between EV and OVs [4]. On the other hand, planning-based prediction methods infer the OVs’ actions as solutions to optimization problems that account for the strategic objectives of the race and tactical interactions with the EV [2]. However, these methods often demand intensive computation, which is critical for fast-paced racing environments. Moreover, the prediction performance heavily relies on the fidelity of the optimization problem, which necessitates encoding the OV’s real objective [4]. Lastly, recent advancements in learning-based methods, which empirically examine driving data patterns, have shown promising results in trajectory prediction [5], [6]. Nonetheless, data scarcity often hinders the implementation of learning-based methods within the autonomous racing regime. Moreover, when the encountered driving policy of the OV deviates from the training data, the resulting predictions fall short in planning safe maneuvers of the EV against OV [1].

![](images/edaeb3b8e4ba41f1075890bc8f0cc7cf43b6ccda5f925ffbf5ce5b824042ebc7.jpg)  
Fig. 1: Overtaking scenarios where the EV maximizes its progress while taking into account the predicted trajectory distribution of the OV, which is based on the driving policy inferred from the observed EV-OV interaction. (a) OV exhibits non-blocking behavior, allowing the EV to navigate without interference; (b) OV actively tries to block the EV’s progress.

This paper presents a learning-based method that can address the diverse driving policies of OVs, thereby accurately predicting the OV’s trajectory and further assessing the uncertainty of prediction results (See Fig. 1). First, the onestep OV’s state prediction model is trained from the observed interaction between the EV and OV through DKL. In detail, a multi-scale Convolutional Neural Network (CNN) is employed to learn the interactions between the EV and OV, transforming the interaction patterns into latent representations of the OV’s driving policy in an unsupervised manner. The represented OV’s driving policy is then processed through a Gaussian Process (GP) model to predict the OV’s state at the next time step. Building upon this process, we propose novel heterogeneous kernel-based metrics whereby DKL can more adeptly ascertain driving policy from complex interaction patterns. The proposed metrics are particularly devised to assess the similarities across different spaces, i.e., the latent space representing driving policies and the output space depicting the OV’s state prediction. Then, these metrics are optimized to render the latent space of DKL such that similar driving policies are clustered closely, while dissimilar ones are simultaneously spread apart. Such a strategic alignment improves the GP model’s ability to capture the correlation between the driving policy and the state prediction. As a result, the DKL model not only enhances prediction performance but also addresses calibrated uncertainty concerning both trained and untrained driving policies.

Next, the one-step OV state prediction model is recursively executed to generate multi-step trajectory samples. These samples are utilized to construct the probabilistic OV trajectory distribution, predicting the state at each time step by its mean and variance. Subsequently, the predicted trajectory distribution information is integrated into the EV’s Model Predictive Control (MPC) framework as dynamic obstacle constraints. This enables the EV to execute safe and agile overtaking maneuvers while considering the OV’s uncertain predicted state distributions. To the best of the authors’ knowledge, the proposed method for the first time attempts to apply kernelbased metrics learning to the realm of autonomous racing. Our method marks a notable advancement in OV trajectory prediction and surpasses the existing work in racing against OVs with diverse driving policies. The contributions of this paper are highlighted as follows:

• A novel kernel-based metrics learning framework for DKL is proposed, aimed at representing the diverse driving policies of OVs that are aligned with respect to their similarity.

• A probabilistic OV trajectory prediction model is developed, improving prediction accuracy and uncertainty calibration based on the inferred OV’s driving policy.

• An MPC-based planning for EV maneuver is seamlessly integrated with the trajectory prediction results, empowering the EV to safely overtake the OV.

• Extensive real autonomous racing experiments with a 1/10th scale racecar demonstrate the effectiveness of the proposed algorithm, both quantitatively and qualitatively, compared to the existing algorithms.

## II. RELATED WORK

## A. Physics and planning-based trajectory predictions

Physics-based trajectory prediction methods exploit the kinematics and dynamics of OVs, tailored by statistical filtering techniques [4]. However, these methods often face challenges in long-term forecasting due to their limited understanding of the interactions between the EV and OVs, implying a lack of consideration for the underlying driving policies of OVs. To account for the interactions, planningbased methods presume that the OVs logically decide their actions through optimization [2]. These methods consider OVs as rational entities seeking to maximize their internal objectives. However, the accuracy of predictions hinges on the coherence between the presumed objective and the true objective of the OV. Furthermore, planning-based methods often demand heavy computations [7], presenting a formidable burden in fast-paced racing circumstances.

## B. Learning-based trajectory prediction

Recent trajectory prediction research has primarily shifted towards learning-based methods that account for the complex interactions involving human social behavior [5] and multiple vehicles in urban settings [6]. Nevertheless, their applicability degenerates when situated in autonomous racing, mostly due to deficient datasets in both quantity (limited number of racing scenes) and quality (lack of semantic cues) [8]. Moreover, the diversity in driving policies across different OVs further complicates learning the prediction model, requiring pertinent representations of driving policies. Such a complication is further exacerbated by imbalanced datasets (e.g., overfitting on the dominant driving policies in the training data), making it difficult to generalize the prediction model. In response, methods such as meta-learning and representation learning have been explored to address the generalization of the prediction model against diverse driving policies [9]. These methods employ distance metrics to capture complex data patterns and establish robust representations for similar/dissimilar patterns, improving prediction performance while reducing the risk of overfitting. While the choice of distance metrics, such as Euclidean, cosine, and kernel-based metrics, greatly impacts the representation model, determining the optimal metrics for measuring similarities remains an unresolved task to date [10]. Lastly, the learning-based prediction model can be susceptible to out-of-training data distribution, meaning it is prone to yield unreliable prediction results when the encountered OV’s driving policy deviates from the training data [11].

## C. Trajectory prediction with uncertainty

Extensive investigation has been carried out to address uncertainty in trajectory prediction [6], among which GP has gained substantial attention due to its quantitative measure of uncertainty. Taking these advantages, GP has proven its effectiveness in cut-in behavior prediction [12], and multi-step trajectory prediction of OV in autonomous racing [3]. Typical GP describes the unknown function using stationary kernels, assuming that the properties of the function are constant across all input values. However, this often results in poor uncertainty calibration for non-stationary stochastic processes that exhibit intricate data patterns in practice [13]. To handle the nonstationary aspect within GP, different techniques have been explored, such as input-dependent parametric kernels, Deep GP, and DKL [13]. In particular, DKL leverages the nonlinear mapping capability of neural networks to transform complex non-stationary input space into the latent feature space that can be effectively processed with stationary kernels in GP. However, training these complex kernels remains a significant challenge, as highlighted by recent studies [14]. Addressing this issue, this paper introduces a kernel-based metrics learning framework for DKL. This framework presents heterogeneous kernels to fine-tune the alignment of latent features transformed by the neural network, thereby adequately embedding the non-stationary properties into DKL.

## III. PROBLEM FORMULATION

## A. Vehicle dynamics

The racing vehicles are modeled using a dynamic bicycle model [15], where the state is represented by the vector $X = [ p _ { x } , p _ { y } , \psi _ { z } , v _ { l o n } , v _ { l a t } , w _ { z } ] ^ { T }$ . This vector consists of the vehicle’s pose in Cartesian frame, $[ p _ { x } , p _ { y } ] ^ { \scriptscriptstyle T }$ , the heading angle, $\psi _ { z }$ , the longitudinal and lateral velocities, $[ v _ { l o n } , v _ { l a t } ] ^ { T }$ , and the yaw rate, $w _ { z }$ . The input vector $\boldsymbol { u } = [ F _ { x x } , \delta ] ^ { \scriptscriptstyle T }$ comprises the rear longitudinal tire force $F _ { x x }$ and the front steering wheel angle δ. To discretize the vehicle’s nonlinear dynamics, we apply the $4 ^ { t h }$ order Runge-Kutta method with a sampling time step size $T _ { s }$ . The resulting discrete-time vehicle dynamics is compactly described as $X _ { k + 1 } ~ = ~ f ( X _ { k } , u _ { k } )$ , where the subscription k indicates the time index. To examine the racing vehicle’s motion, we interpret the vehicle’s pose as Frenet-Serret formulas with respect to the racing track’s centerline [15]. Let $\mathcal { C } ~ : ~ \mathbb { R } ^ { 4 } ~ \mapsto ~ \mathbb { R } ^ { 4 }$ denotes the invertible operator which projects the vehicle’s pose in Cartesian frame to the curvilinear frame. Accordingly, the pose in curvilinear frame is denoted as $c = { \mathcal C } ( p )$ where $p = [ p _ { x } , p _ { y } , \psi _ { z } , v _ { l o n } ] ^ { \scriptscriptstyle T }$ and $c = [ s , e _ { l a t } , e _ { \psi } , v _ { l o n } ] ^ { \scriptscriptstyle T }$ . Here, s is the distance progress along the track’s center line, $e _ { l a t }$ is the lateral offset from the center line, and $e _ { \psi }$ is the angular deviation from the center line’s tangent. The track’s signed curvature at any distance $s \in [ 0 , \tau ]$ is represented as $\eta ( s )$ , where τ is the track length. To distinguish the state variables of EV and OV, we use superscripts in notations like $X ^ { e v } , c ^ { e v } , X ^ { o v }$ , and $c ^ { o v }$

## B. EV trajectory planning problem for racing

The EV trajectory planning problem is formulated as Model Predictive Contouring Control (MPCC), which allows the EV to plan its trajectory over N steps ahead at time step k, defined as $\hat { \mathbf { X } } _ { k : k + N } ^ { e v } \doteq \{ \hat { X } _ { k } ^ { e v } , \ldots , \hat { X } _ { k + N } ^ { e v } \}$ , with respect to the projected distance progress $\bar { s } _ { t }$ on the centerline [15]. Further, the constraints of MPCC related to dynamic obstacles are augmented to incorporate the predicted OV trajectory. Given the predicted trajectory of the OV over N steps ahead at time step k, i.e., $\{ \hat { X } _ { k } ^ { o \bar { v } } , . . . , \hat { X } _ { k + N } ^ { o v } \}$ , the MPCC problem for the EV is formulated as follows:

$$
\operatorname* { m i n } _ { \mathbf { u } , \mathbf { v } } \sum _ { t = 0 } ^ { N - 1 } | | e _ { c } ( \hat { X } _ { t } ^ { e v } , \bar { s } _ { t } ) | | _ { q _ { c } } ^ { 2 } + | | u _ { t } | | _ { R _ { u } } ^ { 2 }
$$

$$
+ \left| \left| u _ { t } - u _ { t - 1 } \right| \right| _ { R _ { d } } ^ { 2 } - q _ { s } \bar { v } _ { N }\tag{1a}
$$

$$
s . t . \ \hat { X } _ { 0 } ^ { e v } = X _ { k } ^ { e v } , \ \bar { s } _ { 0 } = s _ { k } ^ { e v } , \ u _ { - 1 } = u _ { k - 1 } ^ { e v } ,\tag{1b}
$$

$$
\hat { X } _ { t + 1 } ^ { e v } = f ( \hat { X } _ { t } ^ { e v } , u _ { t } ) , ~ t = 0 , . . . . , N - 1\tag{1c}
$$

$$
\bar { s } _ { t + 1 } = \bar { s } _ { t } + T _ { s } \bar { v } _ { t } , \qquad t = 0 , . . . , N - 1\tag{1d}
$$

$$
\hat { X } _ { t } ^ { e v } \in X _ { \mathrm { t r a c k } } , \ u _ { t } \in { \bf U } , \qquad t = 0 , . . . , N\tag{1e}
$$

$$
h ( \hat { X } _ { t } ^ { e v } , \hat { X } _ { k + t } ^ { o v } ) \leq 0 ,
$$

$$
t = 0 , . . . , N\tag{1f}
$$

where $\mathbf { u } = \{ u _ { 0 } , . . . , u _ { N - 1 } \} , \mathbf { v } = \{ \bar { v } _ { 0 } , . . . , \bar { v } _ { N - 1 } \} . \bar { v } _ { t }$ represents the progression rate, and $e _ { c }$ measures the approximated projection errors between $\hat { X } _ { t } ^ { e v }$ and the reference centerline at $\bar { s } _ { t }$ [15]. The objective cost (1a) conceives of maximizing the race progress while penalizing the centerline deviation according to weighting parameters, $q _ { c } , \ q _ { s } \ > 0$ . To ensure a smooth control profile, cost minimizes the magnitude and rate of the input with the weights $R _ { u }$ and $R _ { d } \succ 0 .$ , respectively. The dynamics of the EV is governed by Eqs. (1c) and (1d). In (1e), $X _ { t r a c k } = \{ X | - W _ { t r a c k } / 2 \leq e _ { l a t } \leq W _ { t r a c k } / 2 \}$ and U represent the constraints imposed by the track boundary and the control input limits, where $W _ { t r a c k }$ is the track width. The constraint in Eq. (1f) ensures collision avoidance, given that the predicted OV state forms a probability distribution, such as Gaussian distribution $\hat { X } _ { k + t } ^ { o v } \sim \mathcal N ( \mu ( \hat { X } _ { k + t } ^ { o v } ) , \sigma ^ { 2 } ( \hat { X } _ { k + t } ^ { o v } ) )$ . In constructing Eq. (1f), the EV’s safety margin is drawn as a collection of four circles, while an ellipse represents the $\mathrm { o v } { } _ { \mathrm { s } }$ margin. To accommodate the uncertainty encoded in the predicted state distribution $\hat { X } _ { k + t } ^ { o v } ,$ the size of the OV’s ellipse is dynamically adjusted in proportion to its variance, $\sigma ^ { 2 } ( \hat { X } _ { k + t } ^ { o v } )$ . Detailed derivation for the ellipse size can be found in Section IV of [3], which is omitted here for brevity. Notably, the ellipsoidal safety margin is widely adopted due to its effectiveness in capturing Gaussian-distributed uncertainties in the predicted OV state and its computational efficiency in solving nonlinear MPC problems [16].

## C. GP and DKL for OV trajectory prediction

To predict the OV trajectory which is implicated in the patterns of both EV and OV states, we employ a GP Regression (GPR) model to be trained using the EV-OV interaction dataset. GPR is a non-parametric Bayesian regression that embeds a GP, characterized by a mean function $m ( \cdot )$ and a covariance function $\kappa ( \cdot , \cdot )$ , commonly referred to as a kernel. Typically assuming a zero-mean GP prior for simplicity, the posterior distribution of GPR is given by $\hat { f } ( x ) \sim \mathcal { N } ( \bar { m } ( x ) , \bar { \sigma } ( x ) )$ where $\bar { m } ( x )$ and $\bar { \sigma } ( x )$ represent the posterior mean and covariance functions calculated from the training data, respectively.

In this paper, we adopt the Mate´rn kernel, $\kappa ( \boldsymbol { x } , \boldsymbol { x } ^ { \prime } )$ , which has proven to be effective in the autonomous driving domain [3], [12]. Mate´rn kernel, however, is a stationary kernel that may be insufficient to capture the complex patterns in the trajectories of different OVs with diverse driving policies. To address this, DKL applies a non-linear transformation to the input space and utilizes the same stationary kernels for the transformed space, namely feature space. Accordingly, the kernel in DKL can be reformulated as:

$$
\bar { \kappa } ( x , x ^ { \prime } ) = \kappa ( \Phi ( x ) , \Phi ( x ^ { \prime } ) )
$$

where $\Phi : \mathbb { R } ^ { n } \to \mathbb { R } ^ { n ^ { \prime } }$ denotes the transformation function designed as a neural network, mapping inputs to feature space. However, learning the optimal design of Φ in an unsupervised manner is challenging due to complex EV-OV interactions that yield diverse OV driving policies.

## IV. ALGORITHM DEVELOPMENT

## A. Algorithm overview

In this section, we introduce the main algorithm to predict the trajectory distributions of OVs. First, the one-step OV state prediction model is developed using DKL, which is trained on the EV-OV interaction dataset. Then, we introduce a DKL training framework where novel kernel metrics are employed to address the diversity of OV driving policies. Based on the trained DKL, a sampling-based method is applied to the onestep prediction model to yield a multi-step predicted trajectory distribution. The overall training and deployment phases of the proposed OV trajectory prediction are outlined in Fig. 2.

![](images/3c81d1fb2d90673fd32802bb41f9276abef3230a74c783668bd8f2b6910615e2.jpg)  
Fig. 2: The architecture of the proposed algorithm: (a) the training process of the one-step state prediction model, which utilizes the proposed kernel-based metrics learning for $\mathrm { D K L ; }$ and (b) an autonomous racing algorithm for EV, which plans its maneuver based on the predicted trajectory distribution of OV.

## B. One-step OV state prediction using DKL

To model the one-step state prediction of the OV, we employ DKL, whose input is the past trajectories of both EV and OV, signifying their interaction patterns over a high-dimensional space. The input for the DKL at time step k is formulated as a sequence $\begin{array} { r } { \mathbf { z } _ { k - M : k } ^ { - } = [ \mathbf { z } _ { k - M } ^ { T } , \dots , \mathbf { z } _ { k } ^ { T } ] ^ { T } \in \mathbb { R } ^ { 1 0 \times M } } \end{array}$ , where M denotes the sequence length that records the past interactions, including the current states of EV and OV. Specifically, the vector $\mathbf { z } _ { k }$ includes the state information of both vehicles along with track geometry, defined as:

$$
\mathbf { z } _ { k } = [ ( s _ { k } ^ { o v } - s _ { k } ^ { e v } ) , e _ { l a t , k } ^ { o v } , e _ { \psi , k } ^ { o v } , v _ { l o n , k } ^ { o v } , e _ { l a t , k } ^ { e v } , e _ { \psi , k } ^ { e v } , v _ { l o n , k } ^ { e v } , \tilde { \eta } _ { k } ^ { o v } ] ^ { T }
$$

where $\tilde { \eta } _ { k } ^ { o v } = [ \eta ( s _ { k } ^ { o v } ) , \eta ( s _ { k } ^ { o v } + \zeta ) , \eta ( s _ { k } ^ { o v } + 2 \zeta ) ] ^ { r } \in \mathbb { R } ^ { 3 }$ represents the vector of track curvatures at look-ahead distances $s _ { k } ^ { o v } + i \zeta , ~ i ~ = ~ 0 , 1 , 2$ with some offset $\zeta ~ > ~ 0 .$ . Notably, the vector $\mathbf { z } _ { k }$ is crafted to capture the interaction between the EV and OV, specifically contextualized within the racing track geometry. This facilitates a more accurate description of interaction patterns relative to the track layout, rather than relying solely on the absolute global pose.

The DKL transforms the input into the feature space, representing the OV’s latent driving policy as follows:

$$
\hat { \phi } _ { k } ^ { o v } : = \Phi ( \mathbf { z } _ { k - M : k } )\tag{2}
$$

where $\Phi : \mathbb { R } ^ { 1 0 \times M }  \mathbb { R } ^ { L }$ is a nonlinear mapping function that encodes the interaction history between EV and OV over the past M steps. By this definition, Φ is designed to map different EV-OV interaction patterns to OV driving policies. In this paper, we implement Φ as a neural network model trained on racing data against various OVs. Consequently, the resulting $\hat { \phi } _ { k } ^ { o v }$ exhibits “diverse driving policies” that arise from the different interaction patterns of the OVs. In pursuit of extracting an informative policy representation from the interaction data, we adopt a multi-scale encoder architecture as detailed in [10]. This architecture employs multi-filters with varying kernel sizes to capture both global and local interaction patterns, utilizing 1D causal convolution layers to preserve only the causal relationships in the data sequence. Subsequent to the multi-filtering, the final layers average the outputs of different kernel sizes, resulting in a comprehensive representation of the interaction patterns.

Subsequently, the driving policy $\hat { \phi } _ { k } ^ { o v }$ inputs into GPR to predict the uncertain state propagation of the OV at the next time step, expressed by the following output vector:

$$
\begin{array} { c } { y _ { k } = [ s _ { k + 1 } ^ { o v } - s _ { k } ^ { o v } , e _ { l a t , k + 1 } ^ { o v } - e _ { l a t , k } ^ { o v } , } \\ { e _ { \psi , k + 1 } ^ { o v } - e _ { \psi , k } ^ { o v } , v _ { l o n , k + 1 } ^ { o v } - v _ { l o n , k } ^ { o v } ] ^ { T } } \end{array}
$$

This one-step state difference $y _ { k } \in \mathbb { R } ^ { 4 }$ is modeled as unknown stochastic processes g as follows:

$$
y _ { k } = [ y _ { k } ^ { ( 1 ) } , y _ { k } ^ { ( 2 ) } , y _ { k } ^ { ( 3 ) } , y _ { k } ^ { ( 4 ) } ] ^ { T } = g ( \Phi ( \mathbf { z } _ { k - M : k } ) )
$$

Without loss of generality, we assume the individual output elements $y ^ { ( 1 ) } , y ^ { ( 2 ) } , y ^ { ( 3 ) }$ , and $y ^ { ( 4 ) }$ are independent of each other. Then, the GPR produces a probability distribution for each output element $y ^ { ( n ) }$ as follows:

$$
\begin{array} { r l } & { \boldsymbol { y } ^ { ( n ) } = \boldsymbol { g } ^ { ( n ) } ( \Phi ( \mathbf { z } _ { k - M : k } ) ) } \\ & { \qquad \sim \mathcal { N } ( \mu ^ { ( n ) } ( \Phi ( \mathbf { z } _ { k - M : k } ) ) , ( \boldsymbol { \sigma } ^ { ( n ) } ( \Phi ( \mathbf { z } _ { k - M : k } ) ) ) ^ { 2 } ) } \end{array}
$$

where $\mu ^ { ( n ) }$ and $( \sigma ^ { ( n ) } ) ^ { 2 } , n = 1 , \ldots , 4 ,$ , represent the posterior means and variances, respectively. It is noted that the obtained variances of the GPR capture the epistemic uncertainty, primarily emerging from the diversity of OVs’ driving policies, as well as aleatoric uncertainty stemming from measurement noise.

## C. Kernel-based metrics learning for DKL

In this section, a novel training framework for DKL is introduced. The DKL is typically optimized in regard to the negative log marginal likelihood loss, ${ \mathcal { L } } _ { m l l }$ . However, solely optimizing ${ \mathcal { L } } _ { m l l }$ is prone to overfitting and often yields suboptimal performance during testing [14]. To address this challenge, we incorporate additional regularization to better align the latent driving policies, $\hat { \phi } _ { k } ^ { o v }$ , computed by the neural network transformation, Φ. Central to our approach is the strategic clustering of similar latent driving policies while distancing dissimilar ones, based on the similarities in the output space, i.e., OV’s state differences.

Fig. 3 illustrates the impact of alignment in latent driving policies on the performance of GPR given the same stochastic unknown processes. In the left subfigure, the input and output arrangement is disorganized, leading to increased susceptibility to overfitting and difficulties in training a GPR. Such a poor alignment eventually degrades the prediction model. In contrast, the right subfigure demonstrates a strategically aligned arrangement of latent driving policies, mirroring similarities observed in the output space. This, in turn, ensures that the driving policy $\phi _ { k } ^ { o v }$ is robustly represented, enabling the trained

OV state propagation model, $^ { g , }$ to accurately predict the output $y _ { k }$ from $\phi _ { k } ^ { o v }$ , even when $\phi _ { k } ^ { o v }$ is influenced by variations or uncertainties in the input $\mathbf { z } _ { k - M : k } . \ \mathrm { A s }$ a result, this alignment reduces the risk of overfitting and improves robustness by minimizing sensitivity to input variations, thereby enhancing overall prediction performance.

![](images/bf18ca439e96d487b5d3d15f2b0e7a78ca479e58f8cbc1a5fbfdb1950ca58917.jpg)

![](images/4a16e7fe9a022cced3d44b425567445665c91b1a1981f34083f5fb8de5b1f38e.jpg)  
Fig. 3: Impact of latent space alignment on DKL prediction model: (Left) Without latent space alignment, (Right) With latent space alignment achieved by clustering similar features and dispersing dissimilar ones.

To align the latent driving policies, we introduce a distancematching loss term denoted as $\mathcal { L } _ { d i s t }$ to minimize the dissimilarity between the distances in the pair of latent features and the distances in the pair of outputs. This loss for a mini-batch $\mathcal { D } = \{ \hat { \phi } _ { i } ^ { o v } , y _ { i } \} _ { i \in [ 1 : D ] }$ can be written as:

$$
\mathcal { L } _ { d i s t } = \frac { 1 } { D } \sum _ { i \neq j } ^ { D } | | \kappa _ { \Phi } ( \hat { \phi } _ { i } ^ { o v } , \hat { \phi } _ { j } ^ { o v } ) - \kappa _ { y } ( y _ { i } , y _ { j } ) | | _ { 2 }\tag{3}
$$

where D is the size of the mini-batch. $\kappa \Phi$ and $\kappa _ { y }$ are Mate´rn kernels, the same type of kernel used in GPR. The prediction model regularized by Eq. (3) enforces the data to be aligned in the relevant distance.

On the other hand, the extent of alignment is strongly subject to the hyperparameters of $\kappa _ { \Phi }$ and $\kappa _ { y } ,$ i.e., lengthscales $l _ { \Phi }$ and $l _ { y } ,$ respectively. These hyperparameters directly influence the representation of data, thereby affecting the alignment of latent features. For instance, a smaller lengthscale in the kernel increases its sensitivity to small differences between data, causing them to be treated as significantly dissimilar. Conversely, a larger lengthscale reduces this sensitivity, allowing for a broader margin to identify similarities among distanced data. To find the best lengthscales for enhanced clustering

![](images/03970233edb53d76e36f8f12d56b5c0b435c46a46cdf49166b5a6f05f1a48a62.jpg)  
Fig. 4: Alignment in latent space with varying kernel lengthscales: (Left) $l _ { y } ^ { \mathrm { ~ ~ } } < l _ { \Phi } , ( \bar { \mathrm { R i g h t } } ) l _ { y } > l _ { \Phi }$

of similar data and dispersion of dissimilar ones in latent space, we adjust the kernel sensitivity: decreasing it in the latent space, κ , to widen the distribution of dissimilar data, while increasing it in the output space, $\kappa _ { y } ,$ for precise capture of data differences. This dual regularization is facilitated by introducing a sensitivity loss as follows:

$$
\mathcal { L } _ { s e n s e } = \ln ( \frac { l _ { y } } { l _ { \Phi } } ) + \operatorname* { m a x } ( 0 , w _ { 1 } ( \sigma _ { s e n s e } ( \hat { \phi } ^ { o v } ) - \alpha ) )\tag{4}
$$

where $w _ { 1 }$ and α denote the predefined weight and the threshold, respectively. The second term in Eq. (4) prevents excessive divergence in the latent space, where the standard deviation of the latent features of the batch data, i.e., $\{ \hat { \phi } _ { i } ^ { o v } | i \in 1 , \ldots , D \}$ is denoted as $\sigma _ { s e n s e } ( \hat { \phi } ^ { o v } )$ . Fig. 4 graphically visualizes the effectiveness of the proposed regularization by Eq. (4). Here, the left-hand side illustrates the preferred scenario in which the different latent driving policies are more widely dispersed across various data.

Incorporating all the aforementioned loss terms, DKL is trained to optimize both the neural network and the $\mathrm { G P ^ { \circ } s }$ hyperparameters by minimizing the following composite loss:

$$
\mathcal { L } _ { D K L } = \mathcal { L } _ { m l l } + w _ { 2 } \mathcal { L } _ { d i s t } + w _ { 3 } \mathcal { L } _ { s e n s e }\tag{5}
$$

where $w _ { 2 }$ and $w _ { 3 }$ are scaling constants. In a nutshell, our distinct novelty lies in the inclusion of both $\mathcal { L } _ { d i s t }$ and $\mathcal { L } _ { s e n s e }$ in DKL. While ${ \mathcal { L } } _ { m l l }$ addresses the nominal DKL model for prediction, $\mathcal { L } _ { d i s t }$ is essential for appropriately aligning the latent driving policies and $\mathcal { L } _ { s e n s e }$ is for fine-tuning of kernel sensitivity. Without $\mathcal { L } _ { s e n s e } .$ , kernel-based metrics for both output and latent space may exhibit reduced sensitivities, undermining the clustering efficacy intended by $\mathcal { L } _ { d i s t }$ . It is to be noted that $\mathcal { L } _ { d i s t }$ and $\mathcal { L } _ { s e n s e }$ are mutually complementing each other to achieve the latent feature alignment. Therefore, their interdependency obviates the need for an ablation study of each loss term individually.

## D. Sampling-based multi-step OV trajectory prediction

The one-step OV state prediction model, trained by the proposed DKL framework, is used to predict the OV trajectory over the N step horizon. The basic idea of this multi-step prediction is consecutively executing one-step predictions over future time steps, considering not only the past interaction history but also future interaction. This requires the predicted EV trajectory, $\hat { \bf X } _ { k : k + N - 1 } ^ { e v }$ , which can be obtained from the open-loop solution to (1) for the EV motion planning at $k - 1$ time step.

Analytically computing the solution to the multi-step prediction model is intractable in general [3]. To address this, we utilize a sampling-based approach with $Q$ samples to estimate the multi-step state propagation of the OV. Subsequently, we calculate the statistics of these samples to derive the nominal trajectory prediction along with its variances. Specifically, for the $i ^ { \dot { t } h }$ sample, starting at time step k, we sample OV states at time step $k + 1$ as $c _ { k + 1 } ^ { o v , i }$ using one-step prediction. Given the predicted states of the EV and OV, we construct $\mathbf { z } _ { k + 1 } ^ { i }$ along with the track information. This vector is used to update the input to the one-step prediction model at the next time step, $\mathbf { z } _ { k - M + 1 : k + 1 } ^ { i }$ . We then continue to roll out the consecutive steps for multi-step prediction, repeating over N steps, resulting in the set of multi-step samples, $\{ c _ { k + t } ^ { o v , i } | \ i \ = \ 1 , \ldots , Q , t \ = \ 1 , \ldots , N \}$ . Correspondingly, the predicted trajectory distribution at each future time step, $k + t ,$ is defined by its mean, $c _ { k + t } ^ { o v }$ , and covariance, $\Sigma _ { k + t } ^ { o v }$ , calculated as follows:

$$
\begin{array} { l } { { c _ { k + t } ^ { o v } = \displaystyle \frac { 1 } { Q } \Sigma _ { i = 1 } ^ { Q } c _ { k + t } ^ { o v , i } } } \\ { { \Sigma _ { k + t } ^ { o v } = \displaystyle \frac { 1 } { Q - 1 } \Sigma _ { i = 1 } ^ { Q } ( c _ { k + t } ^ { o v , i } - c _ { k + t } ^ { o v } ) ( c _ { k + t } ^ { o v , i } - c _ { k + t } ^ { o v } ) ^ { T } } } \end{array}\tag{6}
$$

Lastly, the prediction outcomes are utilized to define collision avoidance constraints, as formulated in Eq. (1f), ensuring safe and agile racing maneuvers of EV. The detailed procedure is described in Algorithm 1.

Algorithm 1 Multi-step trajectory prediction with DKL   
Set: Prediction horizon length N and the number of samples   
Q.   
Input : Current EV and OV states, $c _ { k } ^ { e v }$ and $c _ { k } ^ { o v } ;$ ; EV’s   
predicted states, $\hat { \bf X } _ { k : k + N - 1 } ^ { e v } ;$ interaction history between EV   
and OV, ${ \mathbf { z } } _ { k - M : k }$   
1: $c _ { k } ^ { o v , i } = c _ { k } ^ { o v } , \mathbf { z } _ { k - M : k } ^ { i } = \mathbf { z } _ { k - M : k } , \forall i = 1 , \dots , Q$   
2: for $i = 1 , \ldots , Q$ do   
3: for $t = 0 , \ldots , N - 1$ do   
4: Compute $\hat { \phi } _ { k + t } ^ { o v , i }$ in Eq. (2) given $\mathbf { z } _ { k - M + t : k + t } ^ { i }$   
5: Sample $y _ { k + t } ^ { i } \sim \mathcal { N } ( \mu ( \hat { \phi } _ { k + t } ^ { o v , i } ) , \sigma ^ { 2 } ( \hat { \phi } _ { k + t } ^ { o v , i } ) )$ with GP   
6: $c _ { k + t + 1 } ^ { o v , i } = c _ { k + t } ^ { o v , i } + y _ { k + t } ^ { i }$   
7: if $\dot { t } < N - 1$ then   
8: Update $\mathbf { z } _ { k - M + t + 1 : k + t + 1 } ^ { i }$ given $\hat { X } _ { k + t + 1 } ^ { e v } , c _ { k + t + 1 } ^ { o v }$   
9: end for   
10: end for   
11: for $t = 1 , \ldots , N$ do   
12: Compute $c _ { k + t } ^ { o v }$ and $\Sigma _ { k + t } ^ { o v }$ by Eq. (6)   
13: end for   
14: Output : $c _ { k + t } ^ { o v } , \Sigma _ { k + t } ^ { o v } \forall t = 1 , . . , N$

## V. HARDWARE EXPERIMENTS

## A. Experiment setup

The EV and OV are built on the 1/10th scale racecar shown in Fig. 5. To assure reliable real-time execution, the EV utilizes a Jetson AGX Orin, while the OV operates efficiently with a Jetson Orin NX. In terms of software configuration, both vehicles operate with ROS Noetic. They utilize the Cartographer [17] to perform local pose estimation within a shared map common to both the EV and OV. The state estimation involves a customized estimator based on GTSAM [18]. Instead of detecting and tracking the states of other vehicles using sensor measurements, the EV and OV exchange their own state estimates via ROS topics. Both the EV and OV solve optimal control problems using the FORCESPRO Nonlinear Interior-Point solver [19] through code generation. Additionally, DKL is implemented within the PyTorch framework, incorporating variational GPR from GPyTorch [20]. Control operations are executed at 20Hz, while multi-step trajectory predictions are performed at 10Hz.

## B. Diverse driving policies of opponent vehicles

The baseline driving policy for both the EV and OV is established by the MPCC formulation in (1) with a planning horizon of $N = 1 2$ and the sample time of $T _ { s } ~ = ~ 0 . 1 s$ . To facilitate overtaking maneuvers and ensure close interaction between vehicles, longitudinal speed constraints of 1.9m/s for the EV and 1.6m/s for the OV are imposed, respectively.

![](images/059d75a94939e0e1dfeb846ba28bb87610354306467148cd63cef4371c6d228a.jpg)  
Fig. 5: 1/10th scale racing experiments setup. L-shape racing track and the vehicles’ specifications.

OV driving policies are designed to interact with the EV in two different manners: through direct blocking maneuvers (i.e., aggressive driving policy) or by focusing solely on minimal lap time without blocking (i.e., passive driving policy). The implementation of the OV’s aggressive driving policy, as outlined in [3], actively obstructs the EV by minimizing the lateral distance between the OV and EV. This is achieved by including the blocking weights in the cost function (1a). Conversely, the passive driving policy is implemented by omitting the collision avoidance constraints in Eq. (1f) and not incorporating the blocking weights, thus disregarding any interactive maneuvers with the EV.

To train the prediction model, two datasets, $\mathcal { D } _ { a g g r }$ and $\mathcal { D } _ { p a s s } ,$ , are collected to capture the interaction with OVs under these two different driving policies: aggressive and passive, respectively. For our one-step OV state prediction model, we set the dimension of the latent driving policy to $L = 1 1$ and the observed interaction sequence length to $M = 1 0$ . The weights in Eqs. (4) and (5) are tuned as $w _ { 1 } = 2 , w _ { 2 } = 1$ and $w _ { 3 } ~ = ~ 0 . 0 5$ . The training, validation, and test datasets are constructed using samples that are randomly pooled from $\mathcal { D } _ { a g g r }$ and $\mathcal { D } _ { p a s s }$ without any annotations. Finally, for the multi-step prediction model, we set the number of samples in generating the predicted trajectory distribution to $Q = 2 5$

## C. Racing results and discussion

Extensive experiments are carried out to assess the prediction performance of the proposed method and further validate its effectiveness in head-to-head racing, particularly in terms of overtaking maneuvers. For comparative analysis, we evaluate our multi-step trajectory prediction method versus five distinct baseline trajectory predictors. The first baseline, referred to as NMPC (a.k.a Ground Truth, GT), utilizes a planningbased nonlinear MPC predictor. This method employs the open-loop solution to the OV’s MPCC problem, serving as the ground truth prediction from the $\mathrm { o v } { } _ { \mathrm { s } }$ perspective. The second baseline, denoted as Constant Angular Velocity (CAV), is a physics-based prediction method that propagates the $\mathrm { o v } { } _ { \mathrm { s } }$ current state under constant linear and angular velocity assumptions. The third baseline, DNN, adapted from [6], is a deep learning-based method originally developed for urban environments. The input is modified to include EV and OV state histories and track information, i.e., ${ \mathbf { z } } _ { k - M : k }$ . Additionally, the output provides the mean and variances of the multi-step

![](images/cdc864d4dc9f0bbeffd3e827d41f696f8377cc00ac399476a8bfc17f4438ed94.jpg)  
Fig. 6: Snapshots of EV racing maneuvers against OV with passive (a,c,e,g,i) and aggressive (b,d,f,h,j) driving policies. Black solid lines represent the track boundaries. A comprehensive video demonstration of the hardware experiments is accessible at https://github.com/HMCL-UNIST/ OpponentPredictionWithKMDKL.git.

TABLE I: Racing result statistics against OVs with diverse driving policies.
<table><tr><td></td><td>NMPC(GT)</td><td>CAV</td><td>DNN [6]</td><td>GPR [3]</td><td>DKL</td><td>KM-DKL(proposed)</td></tr><tr><td>Longitudinal MSE (mean ± std) Lateral MSE (mean ± std)</td><td> $\mathbf { 0 . 1 7 8 \pm 0 . 1 1 }$   $0 . 2 1 7 \pm 0 . 1 4$ </td><td> $\overline { { 0 . 7 4 6 \pm 0 . 3 0 7 } }$   $0 . 5 8 4 \pm 0 . 3 5$ </td><td> $\overline { { 0 . 8 3 9 \pm 0 . 1 2 } }$   $0 . 3 7 7 \pm 0 . 1 5$ </td><td> $0 . 2 9 0 \pm 0 . 1 5$   $0 . 2 1 9 \pm 0 . 1 4$ </td><td> $\overline { { 0 . 2 5 6 \pm 0 . 1 9 3 } }$   $0 . 2 6 1 \pm 0 . 1 7$ </td><td> $\overline { { 0 . 2 4 4 \pm 0 . 0 9 } }$   ${ \bf 0 . 1 3 9 \pm 0 . 0 8 }$ </td></tr><tr><td>Overtaking rate per race</td><td>0.85</td><td>0.55</td><td>0.65</td><td>0.35</td><td>0.65</td><td>0.95</td></tr><tr><td>Collision rate per race (minor)</td><td>0.30</td><td>0.45</td><td>0.85</td><td>0.40</td><td>0.50</td><td>0.30</td></tr><tr><td>Collision rate per race (major)</td><td>0.10</td><td>0.30</td><td>0.25</td><td>0.55</td><td>0.30</td><td>0.00</td></tr><tr><td>Average computation time (ms)</td><td>12</td><td>6</td><td>39</td><td>84</td><td>97</td><td>98</td></tr></table>

OV, i.e., $0 \leq s ^ { o v } - s ^ { e v } \leq 2 .$

predicted state propagation, which are matched with the output of Algorithm 1 for a fair comparison with other baselines. The fourth baseline, denoted as GPR, adapted from [3], is a learning-based method that employs GPR with a stationary Mate´rn kernel to predict the multi-step trajectory together with its associated uncertainties. This method utilizes the current driving scene information only, i.e., z . Consequently, it faces limitations in capturing the interaction patterns between the EV and OV that are influenced by OV’s driving policy. The fifth baseline denoted as DKL, represents an ablation version of our proposed method. It utilizes the same architecture as the one-step prediction model but omits the proposed kernelbased metrics learning. This is more or less nominal DKL that solely aims at maximizing the log marginal likelihood of observations, i.e., $\mathcal { L } _ { m l l } .$ . Lastly, our proposed method, incorporating a novel kernel-based metrics learning as in Eq. (5), is referred to as KM-DKL. Unlike the other methods that provide prediction results as trajectory distributions, CAV and NMPC (GT) predictors provide deterministic trajectories, lacking a probabilistic measure of prediction confidence. For the sake of fair comparison concerning uncertainty quantification ability, we set a fixed circular boundary to their prediction results.

The experiment involves twenty races for each baseline method. The OV is placed in various starting positions, which remain the same for all predictors. Each race concludes once the EV completes three laps or if a major collision occurs that drastically changes the course of both cars, making it impossible to continue the race. For each race, we set the OV’s driving policy to either passive or aggressive, resulting in ten races for each policy. The prediction performance is then evaluated by the mean square errors of longitudinal and lateral poses between predicted and actual trajectories at the last prediction step when the EV is in the proximity to the

Notably, given the result statistics detailed in Table. I and the prediction snapshots in Fig. 6, our method surpasses all the baselines in lateral prediction accuracy, with NMPC(GT) achieving the lowest longitudinal mean error. Even though NMPC(GT) anticipates the ground truth open-loop solution from OV’s perspective, it fails to capture the multi-step ahead closed-loop interactions between EV and OV in real racing. GPR’s prediction surpasses the physics-based CAV method, yet struggles to adapt to different OV driving policies, leading to a higher risk of collisions due to misapprehension of the driving policy, e.g., blocking as a non-blocking policy and vice versa. Similarly, DKL and DNN falls short of capturing the interactions between EV and OV, resulting in a higher collision rate compared to KM-DKL. In contrast, our KM-DKL, with its prediction ability against different OV driving policies, facilitates agile and safe overtaking maneuvers without major collisions. Furthermore, computational complexity is evaluated using the average computation time in milliseconds. Although our method requires more computation time compared to other baselines, it meets the 10Hz real-time requirement, making it feasible for high-speed autonomous racing.

## D. Ablation study for unseen driving policy

To further investigate the virtue of our method, we examine the uncertainty calibration performance of the proposed KM-DKL model compared to the nominal DKL, the ablated version of our model. This involves evaluating the prediction model against out-of-training-distribution scenarios, specifically when the encountered OV’s driving pattern is not included in the training dataset, i.e., unseen driving policies. To create the unseen driving policy, we introduce a cooperative driving policy designed to yield the pathway to the EV. We realize this policy by modifying the aggressive driving policy’s cost with the reversed sign of the blocking weight. This encourages the OV to move away from the EV’s anticipated path during the race.

We carry out the experiment in a similar fashion to the aforementioned racing setup but with the unseen OV driving policy. The prediction models, DKL and KM-DKL, are evaluated in terms of the Negative Log-Likelihood (NLL) concerning the predicted OV pose at the last prediction step. NLL quantifies the likelihood of the predicted state given the observed data, with lower values indicating more accurate uncertainty calibration to unseen data [21]. Fig. 7 indicates that KM-DKL achieves lower NLL values than the DKL predictor. This demonstrates our method’s improved uncertainty calibration, highlighting the advantages of incorporating kernel metrics learning into DKL.

![](images/57a744fa78d60fd832d42c760a3cf83c446ba5ccf6c52ff40da3c53956fb8aef.jpg)  
Fig. 7: Histograms of the negative log-likelihood for prediction results in races against OVs exhibiting unseen driving policies.

## VI. CONCLUSIONS

This study has developed a learning-based OV trajectory prediction model and downstream EV planning framework for autonomous racing against OVs with diverse driving policies. The key idea is to introduce novel heterogeneous kernel metrics and embed them into the learning pipeline of DKL. The prediction model is then learned in an unsupervised manner based on the EV-OV interaction dataset, adeptly differentiating the diverse driving policies. Compared to existing work, the proposed method enhances prediction accuracy and improves uncertainty calibration, which is instrumental for the safe and agile trajectory planning of EV. This has been validated on a 1/10th scale racecar platform, where the experimental results have shown better-calibrated prediction errors, lower collision rates, and higher overtaking success rates. Future work will focus on: i) online learning and adaptation of the algorithm to streaming data from EV-OV interactions; and ii) scaling the method to full-scale vehicles, addressing perception errors and multi-vehicle interactions in real-world racing.

## REFERENCES

[1] Johannes Betz, Hongrui Zheng, Alexander Liniger, Ugo Rosolia, Phillip Karle, Madhur Behl, Venkat Krovi, and Rahul Mangharam. Autonomous vehicles on the edge: A survey on autonomous vehicle racing. IEEE Open Journal of Intelligent Transportation Systems, 3:458–488, 2022.

[2] Mingyu Wang, Zijian Wang, John Talbot, J Christian Gerdes, and Mac Schwager. Game-theoretic planning for self-driving cars in multivehicle competitive scenarios. IEEE Transactions on Robotics, 37(4):1313– 1325, 2021.

[3] Edward L Zhu, Finn Lukas Busch, Jake Johnson, and Francesco Borrelli. A gaussian process model for opponent prediction in autonomous racing. In 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 8186–8191. IEEE, 2023.

[4] Mahir Gulzar, Yar Muhammad, and Naveed Muhammad. A survey on motion prediction of pedestrians and vehicles for autonomous driving. IEEE Access, 9:137957–137969, 2021.

[5] Alexandre Alahi, Kratarth Goel, Vignesh Ramanathan, Alexandre Robicquet, Li Fei-Fei, and Silvio Savarese. Social lstm: Human trajectory prediction in crowded spaces. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 961–971, 2016.

[6] Balakrishnan Varadarajan, Ahmed Hefny, Avikalp Srivastava, Khaled S Refaat, Nigamaa Nayakanti, Andre Cornman, Kan Chen, Bertrand Douillard, Chi Pang Lam, Dragomir Anguelov, et al. Multipath++: Efficient information fusion and trajectory aggregation for behavior prediction. In 2022 International Conference on Robotics and Automation (ICRA), pages 7814–7821. IEEE, 2022.

[7] Yuxiao Chen, Ugo Rosolia, Wyatt Ubellacker, Noel Csomay-Shanklin, and Aaron D Ames. Interactive multi-modal motion planning with branch model predictive control. IEEE Robotics and Automation Letters, 7(2):5365–5372, 2022.

[8] Lan Feng, Mohammadhossein Bahari, Kaouther Messaoud Ben Amor, Eloi Zablocki, Matthieu Cord, and Alexandre Alahi. Unitraj: A unified<sup>´</sup> framework for scalable vehicle trajectory prediction. arXiv preprint arXiv:2403.15098, 2024.

[9] Junwei Liang, Lu Jiang, and Alexander Hauptmann. Simaug: Learning robust representations from simulation for trajectory prediction. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XIII 16, pages 275–292. Springer, 2020.

[10] Xiaochen Zheng, Xingyu Chen, Manuel Schurch, Amina Mollaysa,¨ Ahmed Allam, and Michael Krauthammer. Simts: Rethinking contrastive representation learning for time series forecasting. arXiv preprint arXiv:2303.18205, 2023.

[11] Thomas Gilles, Stefano Sabatini, Dzmitry Tsishkou, Bogdan Stanciulescu, and Fabien Moutarde. Uncertainty estimation for cross-dataset performance in trajectory prediction. In ICRA 2022 Fresh Perspectives on the Future of Autonomous Driving Workshop, 2022.

[12] Youngmin Yoon, Changhee Kim, Jongmin Lee, and Kyongsu Yi. Interaction-aware probabilistic trajectory prediction of cut-in vehicles using gaussian process for proactive control of autonomous vehicles. IEEE Access, 9:63440–63455, 2021.

[13] Andrew Gordon Wilson, Zhiting Hu, Ruslan Salakhutdinov, and Eric P Xing. Deep kernel learning. In Artificial intelligence and statistics, pages 370–378. PMLR, 2016.

[14] Sebastian W Ober, Carl E Rasmussen, and Mark van der Wilk. The promises and pitfalls of deep kernel learning. In Uncertainty in Artificial Intelligence, pages 1206–1216. PMLR, 2021.

[15] Alexander Liniger, Alexander Domahidi, and Manfred Morari. Optimization-based autonomous racing of 1: 43 scale rc cars. Optimal Control Applications and Methods, 36(5):628–647, 2015.

[16] Rudolf Reiter, Armin Nurkanovic, Jonathan Frey, and Moritz Diehl.´ Frenet-cartesian model representations for automotive obstacle avoidance within nonlinear mpc. European Journal of Control, 74:100847, 2023.

[17] Wolfgang Hess, Damon Kohler, Holger Rapp, and Daniel Andor. Realtime loop closure in 2d lidar slam. In 2016 IEEE international conference on robotics and automation (ICRA), pages 1271–1278. IEEE, 2016.

[18] Frank Dellaert and GTSAM Contributors. borglab/gtsam, May 2022.

[19] A. Zanelli, A. Domahidi, J. Jerez, and M. Morari. Forces nlp: an efficient implementation of interior-point... methods for multistage nonlinear nonconvex programs. International Journal of Control, pages 1–17, 2017.

[20] Jacob Gardner, Geoff Pleiss, Kilian Q Weinberger, David Bindel, and Andrew G Wilson. Gpytorch: Blackbox matrix-matrix gaussian process inference with gpu acceleration. Advances in neural information processing systems, 31, 2018.

[21] Alonso Marco, Elias Morley, and Claire J Tomlin. Out of distribution detection via domain-informed gaussian process state space models. In 2023 62nd IEEE Conference on Decision and Control (CDC), pages 5487–5493. IEEE, 2023.