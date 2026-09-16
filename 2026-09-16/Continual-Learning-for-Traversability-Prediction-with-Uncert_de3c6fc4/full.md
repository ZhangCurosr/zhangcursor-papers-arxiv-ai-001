# Continual Learning for Traversability Prediction with Uncertainty-Aware Adaptation

Hojin Lee, Yunho Lee, Daniel A Duecker, and Cheolhyeon Kwon

Abstract—Traversability prediction is a critical component of autonomous navigation in unstructured environments, where complex and uncertain robot-terrain interactions pose significant challenges such as traction loss and dynamic instability. Despite recent progress in learning-based traversability prediction, these methods often fail to adapt to novel terrains. Even when adaptation is achieved, retaining experience from previously trained environments remains a challenge, a problem known as catastrophic forgetting. To address this challenge, we propose a continual learning framework for traversability prediction that incrementally adapts to new terrains using a generative experience recall model. A key virtue of the proposed framework is two folds: i) retain prior experience without storing past data; and ii) incorporate the uncertainty of the generated samples from the recall model, enabling uncertainty-aware adaptation. Real-world experiments with a skid-steering robot validate the effectiveness of the proposed framework, demonstrating its ability to adapt across a series of diverse environments while mitigating catastrophic forgetting.

Index Terms—Continual Learning, Planning under Uncertainty, Field Robots, Machine Learning for Robot Control

## I. INTRODUCTION

UTONOMOUS navigation in unstructured environments poses significant challenges due to varying terrain traversability, making traversability prediction a central research focus [1]. Traditionally, traversability prediction has been approached by crafting rules based on terrain geometry [2] and semantic context [3]. However, these methods often fail to capture complex robot-terrain interactions. Recently, learning-based methods have offered promising solutions by leveraging driving data without relying on handcrafted traversability rules [4].

While the learning-based traversability prediction methods perform well in conditions similar to the training environment, they often generalize poorly when deployed in novel terrains [5]. To address this limitation, adaptation techniques have been proposed to update traversability models when encountering novel terrain. [6]. However, such adaptation often leads to catastrophic forgetting, where newly acquired experience overwrites prior experience, resulting in degraded performance in previously trained environments.

![](images/cfd7247ef61c9cabc3ee37ef8f6f08982e7ccf73a2db25cd835d8b50ea900601.jpg)  
Fig. 1: Continual adaptation across different types of terrains without access to past data. The robot incrementally updates its model while retaining experience for reliable traversability prediction.

To facilitate adaptation without catastrophic forgetting, strategies such as regularization-based continual learning and dynamic architectural methods have been explored [7]. Among these methods, experience replay-based continual learning has been widely adopted in traversability learning, as it enables models to update by reusing past data without requiring complex optimization objectives or architectural changes [8], [9]. These methods incrementally expand memory or selectively store data for experience replay. However, they are often hindered by high memory costs and/or the potential loss of valuable information during data selection.

To enable adaptation without losing prior experience and without relying on memory-demanding replay mechanisms, we propose a novel continual learning framework for traversability prediction that features uncertainty-aware adaptation. This framework integrates sensory environmental information and robot state information as input modalities, effectively learning to predict both traversability and its associated uncertainty. A key attribute of the proposed framework is a generative experience recall model, which retrieves prior experience to guide continual updates of both traversability prediction model and itself. In this way, prior experience can be retained without needing memory for storing past data. The generative recall model is further empowered by an uncertainty-aware adaptation mechanism, whereby the uncertainties of recalled samples are evaluated so that trustworthy samples are used to update the recall model. Building on this foundation, the trained traversability prediction model is integrated into a model predictive control (MPC) framework to navigate a robot across a series of diverse environments.

These design choices enable a continual learning framework for traversability prediction that exploits a generative experience recall model for effective (i.e., retention of prior experience), efficient (i.e., no need to store past data), and reliable (i.e., uncertainty-aware) update. Real-world experiments with a skid-steering robot confirm the effectiveness of the proposed framework, showing its adaptability across diverse environments while retaining performance in previously trained environments. The main contributions are as follows:

TABLE I: Comparison of Learning-based Traversability Prediction Methods with Adaptation Capability.
<table><tr><td></td><td>Input modality</td><td>Adaptation technique</td><td>Prior-experience memory</td><td>Uncertainty modeling</td></tr><tr><td>[6]</td><td>Vision</td><td>Adaptation with recent experience</td><td>N/A</td><td>Epistemic</td></tr><tr><td>[9]</td><td>Vision</td><td>Experience replay</td><td>Fixed-sized memory</td><td>Aleatoric</td></tr><tr><td>[8]</td><td>Vision</td><td>Experience replay</td><td>Incremental dynamic memory</td><td></td></tr><tr><td>Proposed</td><td>Vision, geometry, dynamics</td><td>Generative experience replay</td><td>N/A</td><td>Aleatoric, epistemic</td></tr></table>

• We propose an uncertainty-aware traversability prediction method that associates terrain features with the robot dynamics to capture complex robot-terrain interactions.

• We introduce a continual learning framework leveraging generative experience recall and an uncertainty-aware adaptation mechanism to retain prior experience while adapting to novel environments.

• Real-world experiments using a skid-steering robot validate the proposed framework, demonstrating persistent navigation performance while learning uncertain traversability across diverse environments.

## II. RELATED WORK

## A. Learning-based Traversability Prediction

Learning-based methods have emerged as a powerful alternative to traditional rule-based methods in traversability prediction. They rely on real or simulated data to model complex robot-terrain interactions with minimal supervision [2]. A critical design choice in prediction models is the input data modality, which directly impacts the model’s ability to comprehend underlying dynamics. Geometry features are commonly used [2], but they are insufficient to capture semantic terrain characteristics. To address this, the Learning Applied to Ground Vehicles (LAGR) program exerted pioneering effort on employing visual features for traversability prediction [3].

Learning terrain geometry and visual features may not be sufficient, since traversability is substantially influenced by the robot’s state and dynamics. Some methods attempt to exploit proprioceptive sensor data to capture this influence, such as measuring traversability in terms of vibration level [10]. However, they still fall short in fully representing traversability, making it difficult to assess the admissible states for navigating in a given terrain. To comprehend the interdependencies between the robot’s state and terrain features in traversability prediction, [11] encompasses the robot’s speed, inertial sensor data, and terrain visual information to establish a forward kinematics model. This model can predict the robot’s dynamic response on the terrain, interpreting its traversability. A major limitation of this approach, however, is the lack of uncertainty modeling, which degrades the robustness and reliability of traversability predictions. Furthermore, the aforementioned learning-based methods often fail to generalize when deployed in environments that differ from the training conditions. In response to this shortcoming, we propose a continual learningbased framework that combines terrain features and robot state information for traversability prediction, while also taking into account uncertainty.

## B. Adaptation for Traversability Prediction

Learning-based traversability prediction faces challenges in novel environments that differ from the trained environment. Adaptation methods address these challenges by updating the model with newly collected data, focusing on rapid adaptation to new environments [6]. However, they overlook retaining experience from previously trained environments during updates. This leads to degraded navigation performance when the robot is deployed in previous environments.

Experience replay mechanisms have been proposed to address this issue by storing past data in a memory buffer and replaying it alongside newly acquired data [7]. [8] employs an incremental replay buffer that clusters visual features, but its memory footprint grows with environmental diversity, limiting scalability in long-term deployment. In contrast, [9] uses a fixed-size replay buffer, carrying the risk of discarding valuable information when adding new data. These methods present a fundamental trade-off between scalability and representational capacity, making it difficult to retain prior experience across diverse environments. Moreover, neither [8] nor [9] models epistemic uncertainty, which is essential for assessing prediction confidence in unfamiliar environments. They also primarily rely on terrain appearance while neglecting the influence of robot dynamics on traversability, limited in capturing the complexity of robot–terrain interactions.

To address the shortcomings in experience replay mechanisms, generative model-based continual learning offers an alternative by synthesizing past data from learned representations, thereby eliminating the need to store past data and supporting scalable adaptation to diverse environments [7]. While offering notable advantages, such methods often fail to account for the uncertainty associated with generated samples, which undermines the prediction model’s updates and potentially compromises performance. To this end, we propose a framework that accounts for the uncertainty of the generated samples and facilitates reliable adaptation to new data and retention of prior experience.

To provide a comparative overview, Table I summarizes key aspects of the representative traversability prediction methods with adaptation, including the proposed framework.

## III. PRELIMINARIES

## A. Robot Dynamics with Traction Parameters

Let us consider a discrete-time dynamics for a ground robot described by:

$$
\boldsymbol { x } _ { k + 1 } = f ( \boldsymbol { x } _ { k } , u _ { k } , \xi _ { k } ) ,
$$

![](images/5c7323c106316290c994e74dbfff2650ddb81faba76190647be6db3497e856de.jpg)  
Fig. 2: Overview of the proposed framework. (a) Deployment phase: uncertainty-aware traversability prediction and navigation. (b) Continual learning phase: uncertainty-aware model adaptation to new environments via generative experience recall.

where $x _ { k }$ represents the robot’s state, $u _ { k }$ is the control input, and $\xi _ { k }$ denotes the traction parameter at time step $k ,$ accounting for variability in robot-terrain interactions. $f$ is modeled by the following unicycle-based kinematic as it effectively captures the dynamics of a wide range of ground robots, including skid-steering and legged platforms:

$$
\begin{array} { r } { \left[ \boldsymbol { p } _ { k + 1 } ^ { x } \right] = \left[ \boldsymbol { p } _ { k } ^ { x } \right] } \\ { \left[ \boldsymbol { p } _ { k + 1 } ^ { y } \right] = \left[ \boldsymbol { p } _ { k } ^ { y } \right] + \Delta t \cdot \left[ \boldsymbol { \xi } _ { k } ^ { x } \cdot \boldsymbol { v } _ { k } ^ { c m d } \cdot \cos ( \theta _ { k } ) \right] , } \\ { \left[ \boldsymbol { \theta } _ { k + 1 } \right] } \end{array}\tag{1}
$$

where the state vector $\boldsymbol { x } _ { k } = [ p _ { k } ^ { x } , p _ { k } ^ { y } , \theta _ { k } ] ^ { \top }$ represents robot $\mathrm { \mathit { \Omega } } _ { \mathrm { { s } } }$ $X - Y$ position in global frame and its yaw angle. The control input $\bar { u _ { k } } = [ v _ { k } ^ { c m d } , \bar { \omega } _ { k } ^ { c m d } ] ^ { \top }$ consists of the linear and angular velocity commands, and $\Delta t > 0$ denotes the sampling time. The traction parameters $\xi _ { k } = [ \xi _ { k } ^ { x } , \xi _ { k } ^ { z } ] ^ { \top } \in \mathbb { R } ^ { 2 }$ account for the linear and angular discrepancies between the commanded and actual velocities due to uncertain robot-terrain interactions. Without loss of generality, we consider these discrepancies as a measure of traversability [12].

## B. Planning with Uncertain Traversability Prediction

In unstructured environments with varying traversability, the navigation problem involves determining the optimal control input sequence $u _ { k : k + T - 1 } ^ { \ast }$ that minimizes the robot’s navigation cost over a receding horizon of length T [12]. Given the state $x _ { k } .$ , and a navigation cost function C, the receding horizon stochastic optimal control problem can be formulated as:

$$
\begin{array} { r l } & { \underset { u _ { 0 : T - 1 } } { \operatorname* { m i n } } ~ \mathbb { E } [ \mathcal { C } ( x _ { 0 : T } , u _ { 0 : T - 1 } ) ] } \\ & { \mathrm { s . t . } \quad x _ { 0 } = x _ { k } , } \\ & { \quad \quad \quad x _ { t + 1 } = f ( x _ { t } , u _ { t } , \hat { \xi } _ { t } ) , \forall t \in \{ 0 , \cdot \cdot \cdot , T - 1 \} } \end{array}\tag{2}
$$

where $\hat { \xi } _ { t } ~ \sim ~ \mathcal { P } ( \xi | x _ { t } )$ represents the distribution of traction at time step t, which can be predicted by the traversability prediction model. To solve (2), we utilize a sampling-based MPC method, leveraging its parallelizability on GPUs for realtime computation [13].

## C. Continual Learning Problem Formulation

To reliably codify $\hat { \xi }$ across different environments, we tackle the problem of domain-incremental continual learning for traversability prediction [7]. We define the traversability prediction model $\mathcal { T } : \mathbb { R } ^ { q }  \mathbb { R } ^ { 2 } \times \mathbb { R } _ { > 0 } ^ { 2 }$ as a mapping from input features, comprising the robot’s state and terrain features, to the mean and variance of the Gaussian distributions over traction parameters $\hat { \xi } \sim \mathcal { N } ( \mu ( \cdot ) , \sigma ^ { 2 } ( \cdot ) )$ , where $( \mu ( \cdot ) , \sigma ^ { 2 } ( \cdot ) ) =$ $\tau ( \cdot )$ . The goal is to train a sequence of models $\mathcal { T } ^ { 1 } , \cdots , \mathcal { T } ^ { i - 1 }$ such that $\mathcal { T } ^ { i }$ adapts to new environments encountered during the $i ^ { \mathrm { { t h } } }$ deployment, while retaining experience from previous $( 1 ^ { s t } , \cdot \cdot \cdot , i - 1 ^ { t h } )$ deployments. In this setting, the input distribution (environmental terrain characteristics) may shift over deployments, whereas the underlying task of learning (traversability prediction model) remains consistent.

At each deployment phase i, the robot collects data $\mathcal { D } ^ { i }$ in the corresponding environment, forming a sequential dataset $\mathcal { D } ^ { 1 : i } = \{ \bar { \mathcal { D } } ^ { 1 } , . . . , \bar { \mathcal { D } } ^ { i } \}$ . Ideally, the model $\mathcal { T } ^ { i }$ would be trained on the full dataset $\mathcal { D } ^ { 1 : i }$ , but storing and accessing all the past data is impractical on resource-constrained platforms. Instead, our continual learning framework aims to update $\mathcal { T } ^ { i - 1 }$ using only $\mathcal { D } ^ { i }$ , with the goal of producing $\mathcal { T } ^ { i }$ that preserves performance without direct access to $\mathcal { D } ^ { 1 : i - 1 }$

## IV. ALGORITHM DEVELOPMENT

This section presents the proposed continual learning framework for traversability prediction, empowered by uncertainty-

aware adaptation, as depicted in Fig. 2.

## A. Self-Supervised Traversability Labeling

To train the traversability prediction model, we collect samples of traction parameters ξ while navigating over different terrain environments, including both traversable and non-traversable regions. Inspired by the nonlinear movinghorizon estimator in [4], we utilize odometry to estimate the robot’s state x and use this information to estimate the traction parameters ξ. Specifically, we employ an Extended Kalman Filter (EKF) where the traction parameters are modeled as a random walk process: $\xi _ { k + 1 } = \xi _ { k } + \nu .$ . Here, ν is zeromean Gaussian process noise that accounts for gradual variations over time [14]. Using (1) along with state $x _ { k + 1 }$ , the EKF iteratively refines the traction parameter estimates $\xi _ { k }$ establishing datasets that are used to train the traversability prediction model.

## B. Terrain Feature Processing

We use RGB images and 3D point clouds to build a multi-modal terrain representation based on [15]. The point clouds are processed into an elevation grid map, capturing the geometric properties of the terrain. Simultaneously, visual features are extracted from images using a pre-trained DINOv2 backbone [16], producing pixel-wise feature embeddings. However, projecting these high-dimensional visual features (e.g., 384 dimensions for the DINOv2-small model) into a grid map incurs substantial computational and memory overhead, making it impractical for real-time use on mobile robots.

![](images/4d7770116d4927390a96ff809f0e4835ae6e7d59aa31eb1db6e73e3d60c160ef.jpg)  
Fig. 3: Grid mapping of the terrain feature $\phi ,$ based on geometric and visual information of the corresponding terrain patches beneath the robot’s four wheels.

To address this, we employ an autoencoder network to compress the high-dimensional visual feature embeddings from DINOv2 into a low-dimensional latent feature. The network is trained in an unsupervised manner on the original dense DINOv2 features from a large-scale off-road dataset by optimizing reconstruction loss. Once trained, the encoder part is deployed for real-time processing, receiving dense features from the DINOv2 backbone and generating compact, lowdimensional visual features to be integrated into the multimodal grid map. As a result, terrain features $\phi \in \mathbb { R } ^ { p }$ , implicating both geometric and visual information, are assigned to individual grids in the map, as illustrated in Figure 3. These terrain features are stored in a local grid map centered on the robot’s pose, capturing terrain characteristics over a fixed window for traversability model training and prediction.

## C. Robot-terrain Interaction Dataset

Based on the labeled traction parameter data, driving data, and terrain feature data, we construct a robot-terrain interaction dataset:

$$
\mathcal { D } = \{ ( \phi _ { j } , v _ { j } , \xi _ { j } ) \mid j = 1 , \ldots , B \} ,
$$

where each sample consists of the terrain feature vector $\phi _ { j }$ , the robot’s velocity state $v _ { j }$ , and the labeled traction parameter $\xi _ { j }$ at time step j. B denotes the total number of samples in the dataset. The terrain feature vector $\phi _ { j } ~ =$ $\left[ \boldsymbol { \phi } _ { j , 1 } ^ { \top } , \boldsymbol { \phi } _ { j , 2 } ^ { \top } , \boldsymbol { \phi } _ { j , 3 } ^ { \top } , \boldsymbol { \phi } _ { j , 4 } ^ { \top } \right] ^ { \top }$ consists of multiple terrain features extracted at different contact points. For instance, in a fourwheeled robot, each $\phi _ { j , i }$ corresponds to the terrain features at the $i ^ { t h }$ wheel position $( p _ { j , i } ^ { x } , p _ { j , i } ^ { y } )$ at time step j. The robot’s velocity state $v _ { j } = \left[ v _ { j } ^ { x } , w _ { j } ^ { z } \right] ^ { 1 }$ includes the linear and angular velocities estimated by odometry. To construct the dataset, we retrieve the stored local terrain grid maps for each recorded robot pose. For each map, we project the trajectory history within a local window and extract terrain features at the wheel contact points along the trajectory. These features are then paired with the corresponding robot velocity and traction estimates to form the dataset. Samples with missing terrain features at any wheel contact point due to being outside the sensor field of view are discarded.

## D. Uncertainty-aware Traversability Prediction

Given the dataset D, the model T is trained to predict the traction parameter as $\hat { \xi } _ { k } = \mathcal { T } ( \phi _ { k } , v _ { k } )$ , where $\phi _ { k }$ represents the terrain features, and $v _ { k }$ denotes the robot velocity state at time step k. To account for uncertainty, we model T as a probabilistic ensemble model capable of capturing both aleatoric and epistemic uncertainties [17]. Each ensemble member receives the concatenated terrain feature $\phi _ { k }$ and velocity state $v _ { k }$ , as input and is trained independently. All ensemble members share the same architecture, comprising a multi-layer perceptron with fully connected layers of sizes [16, 32, 32, 16], each followed by a LeakyReLU non-linear activation function. The $m ^ { t h }$ ensemble model $\mathcal { T } _ { m }$ outputs the traction parameter as a Gaussian distribution:

$$
\hat { \xi } _ { k , m } \sim { \cal N } ( \mu _ { k , m } , \sigma _ { k , m } ^ { 2 } ) , ~ \mathrm { w h e r e } ~ [ \mu _ { k , m } , \sigma _ { k , m } ^ { 2 } ] = { \cal T } _ { m } ( \phi _ { k } , v _ { k } ) .
$$

Based on predictions from M ensemble models, we compute the final prediction as:

$$
\hat { \xi } _ { k } \sim { \cal N } ( \mu _ { k } , \sigma _ { k } ^ { 2 } ) , ~ \mathrm { w h e r e } ~ [ \mu _ { k } , \sigma _ { k } ^ { 2 } ] = \mathcal { T } ( \phi _ { k } , v _ { k } ) ,\tag{3}
$$

with the mean and variance computed as:

$$
\mu _ { k } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \mu _ { k , m } , \quad \sigma _ { \mathrm { k } } ^ { 2 } = \sigma _ { \mathrm { k , a l e } } ^ { 2 } + \sigma _ { \mathrm { k , e p i } } ^ { 2 } ,
$$

where the variances from aleatoric $\sigma _ { \mathrm { k , a l e } } ^ { 2 }$ and epistemic $\sigma _ { \mathrm { k , e p i } } ^ { 2 }$ uncertainties are respectively given by:

$$
\sigma _ { \mathrm { k , a l e } } ^ { 2 } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \sigma _ { k , m } ^ { 2 } , \quad \sigma _ { \mathrm { k , e p i } } ^ { 2 } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } ( \mu _ { k , m } - \mu _ { k } ) ^ { 2 } .
$$

## E. Uncertainty-Aware Adaptation with Experience Recalling

To adapt the traversability prediction model across diverse environments, we leverage a generative model–based continual learning framework. The generative model recalls past input–output samples for experience replay without storing raw data. Before delving into the details of our continual learning framework, the initial step is established as follows.

1) Initial training of the traversability prediction model: we begin with the initial model, $\tau ^ { 1 }$ , trained on the first environment dataset $\mathcal { D } ^ { 1 }$ , which serves as the foundation for adaptation. This initial model $\tau ^ { 1 }$ is trained by minimizing the Negative Log-Likelihood (NLL) loss, denoted as $\mathcal { L } _ { \mathrm { N L L } }$ , which captures the likelihood of the estimated traction parameters given the corresponding terrain features and the robot’s dynamics states, i.e., velocity.

2) Initial training ofthe generative experience recall model: we design a generative recall model $\mathcal { R } ^ { 1 }$ , which randomly synthesizes the prediction model’s input and output data, denoted as $\mathcal { D } ^ { \mathrm { r e c a l l } }$ . As with the prediction model, the input data consists of two components: terrain features $\phi ^ { \mathrm { r e c a l l } }$ and robot dynamic states, i.e., $v ^ { \mathrm { r e c a l l } }$ . Leveraging the fact that velocities are physical quantities with bounded ranges, we design a conditional recall process that first samples velocity and then generates corresponding terrain features conditioned on it. To accomplish this, we adopt a Conditional Variational Autoencoder (CVAE) [18], which consists of an encoder and a decoder, denoted as $q _ { e n c } ( z | \phi , v )$ and $p _ { d e c } ( \phi | v , z )$ , respectively. Upon training of the CVAE, the decoder is used to recall features $\phi ^ { \mathrm { r e c a l l } }$ conditioned on a sampled velocity state $v ^ { \mathrm { r e c a l l } }$ and latent variable z. During recall, both $v ^ { \mathrm { r e c a l l } }$ and z are randomly sampled, where $v ^ { \mathrm { r e c a l l } }$ respects the velocity limits defined by the robot’s hardware specifications, and z is drawn from a Gaussian prior distribution. Next, to synthesize the output data, the trained traversability prediction model $\tau$ processes the recalled inputs as follows:

$$
\xi ^ { \mathrm { r e c a l l } } \sim \mathcal { N } ( \mu ^ { \mathrm { r e c a l l } } , ( \sigma ^ { \mathrm { r e c a l l } } ) ^ { 2 } ) ,
$$

where $[ \mu ^ { \mathrm { r e c a l l } } , ( \sigma ^ { \mathrm { r e c a l l } } ) ^ { 2 } ] = \mathcal { T } ^ { 1 } ( \phi ^ { \mathrm { r e c a l l } } , v ^ { \mathrm { r e c a l l } } )$ . This prediction includes $\sigma ^ { \mathrm { r e c a l l } }$ as a measure of uncertainty, reflecting the model’s confidence in its generated samples. Specifically, a higher $\sigma ^ { \mathrm { { r e c a l l } } }$ indicates that the recalled sample likely originates from an environment that was not well represented in $\mathcal { D } ^ { 1 }$ for training $\tau ^ { 1 }$

3) Uncertainty-aware traversability prediction model update: we update the traversability prediction model from $\mathcal T ^ { i - 1 }$ to $\mathcal { T } ^ { i }$ for $i ~ \geq ~ 2$ , using only the collected dataset $\mathcal { D } ^ { i }$ and the generative recall model $\mathcal { R } ^ { i - 1 }$ . First, we generate a batch of recalled input-output samples using $\mathcal { R } ^ { i - 1 }$ , overwriting the dataset $\mathcal { D } ^ { \mathrm { r e c a l l } }$ . These recalled samples are then combined with the newly collected dataset $\mathcal { D } ^ { i }$ to update the model $\mathcal { T } ^ { i }$ by minimizing the following loss function:

$$
\begin{array} { r } { \mathcal { L } _ { \mathcal { T } ^ { i } } = \mathcal { L } _ { \mathrm { N L L } } ( \mathcal { D } ^ { i } ) + \lambda \mathcal { L } _ { \mathrm { a d a p t } } ( \mathcal { D } ^ { \mathrm { r e c a l l } } ) , } \end{array}\tag{4}
$$

where λ is a scaling constant. The first term, $\mathcal { L } _ { \mathrm { N L L } } ( \mathcal { D } ^ { i } )$ captures the experience obtained from newly acquired data. The second term, $\mathcal { L } _ { \mathrm { a d a p t } } ( \mathcal { D } ^ { \mathrm { r e c a l l } } )$ , aims to retain prior experience through recalled samples that imitate past data. This is achieved by aligning the distributions of the predicted and recalled output distributions using the Jensen–Shannon (JS) divergence as a distance measure, defined as follows:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { a d a p t } } ( \mathcal { D } ^ { \mathrm { r e c a l l } } ) : = \displaystyle \sum _ { \mathcal { D } ^ { \mathrm { r e c a l l } } } \frac { 1 } { 2 } K L \left( \mathcal { T } ^ { i } ( \phi ^ { \mathrm { r e c a l l } } , v ^ { \mathrm { r e c a l l } } ) | \xi ^ { \mathrm { r e c a l l } } \right) } \\ & { \qquad + \displaystyle \frac { 1 } { 2 } K L \left( \xi ^ { \mathrm { r e c a l l } } | \mathcal { T } ^ { i } ( \phi ^ { \mathrm { r e c a l l } } , v ^ { \mathrm { r e c a l l } } ) \right) , } \end{array}
$$

where $K L$ denotes the Kullback-Leibler (KL) divergence. This distributional alignment alleviates catastrophic forgetting of recalled samples and mitigates overfitting to unreliable samples with higher $\sigma ^ { r e c a l l }$ , thereby reducing the risk of erroneous updates.

4) Uncertainty-aware generative experience recall model update: similar to the prediction model update, we update the generative experience recall model from $\mathcal { R } ^ { i - 1 }$ to $\mathcal { R } ^ { i }$ using the newly collected dataset $\mathcal { D } ^ { i }$ and recalled dataset $\mathcal { D } ^ { \mathrm { r e c a l l } }$ , without explicit access to past data, i.e., $\mathcal { D } ^ { l - 1 } , \forall l \leq i .$ . Specifically, we recall data by randomly sampling input features, denoted as $\big ( \phi ^ { \mathrm { r e c a l l } } , v ^ { \mathrm { r e c a l l } } \big )$ , using $\mathcal { R } ^ { i - 1 }$ . The quality of the recalled samples is assessed using their uncertainty $( \overbar { \boldsymbol { \sigma } } ^ { \mathrm { r e c a l l } } ) ^ { 2 }$ , predicted by the preceding prediction model $\mathcal T ^ { i - 1 }$ . These uncertainty estimates act as proxies for distributional similarity to real data, enabling us to identify and filter recalled samples that deviate from the real data distribution. To this end, we apply a variance threshold τ to exclude uncertain recalls, yielding the refined recall dataset:

$$
\bar { \mathcal { D } } ^ { \mathrm { r e c a l l } } : = \left\{ ( \phi ^ { \mathrm { r e c a l l } } , v ^ { \mathrm { r e c a l l } } ) \big | \sigma ^ { \mathrm { r e c a l l } } < \tau \right\} .
$$

We then define the augmented dataset as the union of the newly acquired data $\mathcal { D } ^ { i }$ and the filtered recalled samples:

$$
{ \mathcal { D } } ^ { \mathrm { a u g } } : = { \mathcal { D } } ^ { i } \cup { \bar { \mathcal { D } } } ^ { \mathrm { r e c a l l } } .\tag{5}
$$

The generative experience recall model $\mathcal { R } ^ { i }$ is trained using $\mathcal { D } ^ { \mathrm { a u g } }$ , ensuring that only recalled samples with sufficiently low uncertainty contribute to retaining prior experience while recently acquired data facilitates adaptation to the new environment.

A comprehensive overview of the proposed continual learning framework <sup>1</sup> with uncertainty-aware adaptation is presented in Algorithm 1.

Algorithm 1 Continual Learning Framework with   
Uncertainty-aware Adaptation   
Initialize: Set iteration index $i = 1$   
Input: Initial dataset $\mathcal { D } ^ { 1 }$   
1: Initial train $\tau ^ { 1 }$ and $\mathcal { R } ^ { 1 }$ (Sections IV.E.1 and IV.E.2).   
2: while New dataset $\mathcal { D } ^ { i + 1 }$ is acquired do   
3: Generate recalled dataset D<sup>recall</sup> using $\mathcal { R } ^ { i }$ and ${ \mathcal { T } } ^ { i } .$   
4: Update $\mathcal { T } ^ { i }  \mathcal { T } ^ { i + 1 }$ with (4) (Section IV.E.3).   
5: Update $\mathcal { R } ^ { i }  \mathcal { R } ^ { i + 1 }$ using $\mathcal { D } ^ { \mathrm { a u g } }$ in (5)   
(Section IV.E.4).   
6: $i = i + 1 .$   
7: end while

## F. Navigation with Traversability Prediction

Based on the updated traversability prediction model, the optimal off-road navigation can be formulated as an MPC problem. The MPC minimizes a cost function associated with predicted future states as outlined in Section III-B. Specifically, the objective cost in (2) is defined as:

$$
\mathcal { C } ( x _ { t } , \mu _ { t } , \sigma _ { t } ^ { 2 } ) = w _ { 1 } \mathrm { G o a l } ( p _ { t } ^ { x } , p _ { t } ^ { y } ) + w _ { 2 } \sigma _ { t } ^ { 2 } + w _ { 3 } | 1 - \mu _ { t } | ,
$$

where Goa ${ \mid } ( p _ { t } ^ { x } , p _ { t } ^ { y } )$ encourages progress toward the goal over a finite horizon, and $\sigma _ { t } ^ { 2 }$ penalizes regions of high predictive uncertainty stemming from limited environment knowledge and/or irreducible measurement noise. The slip cost term $\left| 1 - \mu _ { t } \right|$ encourages the predicted traction to approach 1, representing ideal alignment between the robot’s actual motion and the commanded input under the nominal dynamics. Notably, $\mu _ { t }$ is not constrained to lie within [0, 1], allowing the model to capture a wider range of robot-terrain interactions. The weights $w _ { 1 } , w _ { 2 } , w _ { 3 } , \in \mathbb { R } ^ { + }$ are tunable scaling factors.

To solve for (2), which is subject to stochastic dynamics $f ,$ we adopt a mean-propagation strategy for cost approximation, following the method in [17]. Specifically, the robot dynamics are propagated using the mean estimates of uncertain traction parameters. The propagated states are then used to evaluate the cost function and optimize the control inputs. Under this strategy, the navigation problem can be expressed as the following MPC problem:

$$
\begin{array} { r l } {  { \operatorname* { m i n } _ { u _ { 0 } \cdot T - 1 } \sum _ { t = 0 } ^ { T } \mathcal { C } ( x _ { t } , \mu _ { t } , \sigma _ { t } ^ { 2 } ) } } \\ { \mathrm { s . t . } \qquad x _ { 0 } = x _ { k } , } \\ & { \qquad x _ { t + 1 } = f ( x _ { t } , u _ { t } , \mu _ { t } ) , \ \forall t \in \{ 0 , \cdots , T - 1 \} } \\ & { \qquad [ \mu _ { t } , \sigma _ { t } ^ { 2 } ] = \mathcal { T } ( \phi _ { t } , v _ { t } ) , \forall t \in \{ 0 , \cdots , T - 1 \} . } \end{array}
$$

Upon optimization, only the first control $u _ { 0 }$ is applied to the robot, while the remaining $u _ { 1 : T - 1 }$ are temporarily stored in a buffer and used as fallback actions if the subsequent MPC computation fails to meet the desired control frequency.

## V. FIELD EXPERIMENT

## A. Platform and Dataset Description

The experiments are conducted on a Clearpath Jackal robot, equipped with an NVIDIA Jetson AGX Orin for onboard computation, an Ouster LiDAR, and a ZED X stereo camera to sense its surrounding environment. A Lidar-inertial odometry [19] provides high-frequency pose and velocity estimates. The ZED X camera captures both RGB images and depth point clouds, which are used to extract the terrain features. The extracted features are encoded into a multi-modal grid map with a resolution of 0.25 meters [15], and represented as latent vectors of dimension $p = 3 .$ , updated at a rate of 20 Hz. An ensemble of $M = 5$ models performs traversability prediction. The entire system is built within the ROS 2 environment. The control horizon is set to $T = 3 0$ with a time step of $\Delta t = 0 . 1$ seconds, corresponding to a 3-second prediction window. A batch of 1024 rollouts is used per optimization cycle to support a control frequency of 10 Hz.

![](images/1b70074693da0995266a1f186919db7ef1df135237bea7772f4ffd377cf25615.jpg)  
Fig. 4: PCA projection of terrain features ϕ from five distinct environments.

The evaluation of continual learning necessitates diverse environments, which enables assessment of how well prior knowledge is retained when adapting to novel environments. In light of this requirement, we consider five environments with distinct terrain characteristics: an asphalt trail in a public park (Dataset $\mathcal { D } ^ { 1 } )$ , a gravel terrain near a river shore (Dataset $\mathcal { D } ^ { 2 } )$ , a sand area (Dataset $\mathcal { D } ^ { 3 } )$ , a bike path (Dataset $\mathcal { D } ^ { 4 } )$ , and a forest with dense vegetation (Dataset $\mathcal { D } ^ { 5 } )$ . At each site, we collected approximately 10 minutes of data by manually driving through both traversable and non-traversable areas. We apply Principal Component Analysis (PCA) to the extracted terrain features $\phi ,$ projecting them into a two-dimensional space to characterize distributional differences among the five environments, as illustrated in Figure 4. The projected features reveal distinct domain shifts that are pertinent to a domain-incremental continual learning problem, as discussed in Section III-C.

## B. Evaluation Metrics

The traversability prediction model is trained sequentially on datasets $\mathcal { D } ^ { 1 }$ through $\mathcal { D } ^ { 5 }$ , each corresponding to data collected from Environments 1 through 5. Continual learning performance is evaluated in terms of adaptation and memory retention. Both are computed from the prediction performance of the traction model on test datasets using the Negative Log-Likelihood (NLL). Let $n _ { k , j } \in \mathbb { R } , \ k \leq j$ denote the NLL on the test set of $\mathcal { D } ^ { k }$ after training on datasets $\mathcal { D } ^ { 1 : j }$ . Adaptation performance at the environment $j$ is given by ${ n } _ { j , j }$ , which measures how well the updated model $\mathcal { T } ^ { j }$ fits $\mathcal { D } ^ { j }$ . A lower $n _ { j , j }$ indicates better adaptation performance. Memory retention performance is evaluated using the Forgetting Measure (FM) [7], defined as $f _ { k , j } ~ = ~ \mathrm { m a x } ( 0 , ~ n _ { k , j } ~ - ~ \mathrm { m i n } _ { i < j } ~ n _ { k , i } ) $ which captures the maximum performance degradation on $\mathcal { D } ^ { k }$ after training on $\mathcal { D } ^ { j }$ . A lower FM indicates better experience retention and improved continual learning performance.

## C. Baseline Methods

For comparative analysis, we evaluate the proposed continual learning framework against five baseline methods:

• Complete-Memory (CM): This upper-bound baseline trains $\mathcal { T } ^ { i }$ using the full dataset D<sup>j</sup>, $1 \leq j \leq i ,$ enabling batch retraining with complete access to past data.

• Incremental-Memory (IMOST) [8]: This method updates $\mathcal { T } ^ { i - 1 }$ using an incremental dynamic memory buffer that stores new samples according to an informationexpansion criterion.

• Fast Adaptation (WVN) [6]: This baseline updates $\mathcal T ^ { i - 1 }$ using only the recently acquired data $\mathcal { D } ^ { i }$ , without considering experience retention.

• LwF-fashion (LwF) [20]: This method uses the model $\mathcal T ^ { i - 1 }$ and the recently acquired data $\mathcal { D } ^ { i }$ to retrain $\mathcal { T } ^ { i }$ without accessing past data. Experience retention is achieved via distillation by minimizing the KL divergence between the outputs of $\scriptstyle { \dot { T } } ^ { i - 1 }$ and $\mathcal { T } ^ { i }$ on $\mathcal { D } ^ { i }$

• Naive Generative Rehearsal (NGR): A variant of our method that updates $\mathcal T ^ { i - 1 }$ with recalled samples $\mathcal { R } ^ { i - 1 }$ and the recently acquired data $\mathcal { D } ^ { i }$ without the uncertaintyaware adaptation in Section IV-E4, serving as an ablation of the threshold $\tau .$

To ensure a fair comparison, all baseline methods are interfaced to take terrain and robot states as inputs and to output probabilistic traction, as in the proposed framework. For each method, we identify the hyperparameter setting on the Pareto front with equal weighting of adaptation and memory retention.

## D. Field Experiment Results and Discussion

1) Evaluation of experience recalling: we begin by assessing the validity of recalled samples generated by the generative experience recall model $\mathcal { R } ^ { i }$ during adaptation. The distributions of the recalled terrain features are visualized on top of the real terrain features using PCA, as shown in Figure 5. While the proposed method produces samples that closely align with the real data distribution, the ablation method (NGR) generates dispersed samples that poorly align with real data. These results highlight that the uncertaintyaware adaptation introduced in Section IV-E4 helps the recall model produce samples that better reflect collected experience, supporting more accurate model updates during continual learning.

![](images/db959e1f2e3f7fc41c9b82dd72200cc4560bd2f35c54d84c2acf6cbf9fc659d8.jpg)  
Fig. 5: PCA projection of recalled samples alongside real terrain features collected from Environments 1–5.

2) Evaluation of adaptation performance: as part of the quantitative evaluation, Table II and Table III report the mean and standard deviation of NLL and FM over ten training sessions for each continual learning scenario. The CM method consistently achieves the lowest forgetting, as it keeps access to the full dataset throughout training. Among those without direct access to past data, our proposed method yields the lowest FM, indicating improved retention of prior experience while effectively adapting to new environments. The IMOST method achieves retention performance comparable to, and occasionally exceeding, our method, but this comes at the cost of a continually growing memory requirement. This highlights the effectiveness of our uncertainty-aware adaptation for continual learning without storing past data, retaining prior experience with only the parameters of the prediction and recall models.

TABLE II: Adaptation statistics during continual learning (mean ± std).
<table><tr><td colspan="2"> ${ \mathrm { N L L } } , n _ { j , j }$ </td><td colspan="4">Training from  $\mathcal { D } ^ { 1 } \mathrm { ~ t o ~ } \mathcal { D } ^ { j }$ </td></tr><tr><td colspan="2"></td><td>一  $\mathcal { D } ^ { 1 } \to \mathcal { D } ^ { 2 }$ </td><td> $\mathcal { D } ^ { 1 } \to \mathcal { D } ^ { 2 } \to \mathcal { D } ^ { 3 }$ </td><td> $\mathcal { D } ^ { 1 } \to \cdots \to \mathcal { D } ^ { 4 }$ </td><td> $\mathcal { D } ^ { 1 } \to \cdots \to \mathcal { D } ^ { 5 }$ </td></tr><tr><td colspan="2">Test j</td><td> $\mathcal { D } ^ { 2 }$ </td><td> $\mathcal { D } ^ { 3 }$ </td><td> $\mathcal { D } ^ { 4 }$ </td><td> $\mathcal { D } ^ { 5 }$ </td></tr><tr><td rowspan="6">Method</td><td>CM</td><td> $0 . 1 7 \pm 0 . 1 5$ </td><td> ${ \bf - 0 . 6 1 \pm 0 . 0 9 }$ </td><td> ${ \bf 0 . 8 9 \pm 0 . 1 3 }$ </td><td> ${ \bf - 0 . 6 6 \pm 0 . 0 6 }$ </td></tr><tr><td>IMOST</td><td> $0 . 5 0 \pm 0 . 2 6$ </td><td> $- 0 . 4 6 \pm 0 . 0 8$ </td><td> $1 . 2 7 \pm 0 . 0 7$ </td><td> $- 0 . 3 5 \pm 0 . 0 6$ </td></tr><tr><td>WVN</td><td> ${ \bf 0 . 1 3 \pm 0 . 0 8 }$ </td><td> $- 0 . 5 7 \pm 0 . 0 7$ </td><td> $1 . 0 6 \pm 0 . 0 1$ </td><td> $- 0 . 4 4 \pm 0 . 0 7$ </td></tr><tr><td>LwF</td><td> $0 . 1 9 \pm 0 . 1 2$ </td><td> $- 0 . 5 5 \pm 0 . 0 7$ </td><td> $1 . 0 3 \pm 0 . 1 1$ </td><td> $- 0 . 5 7 \pm 0 . 0 6$ </td></tr><tr><td>NGR</td><td> $0 . 8 2 \pm 0 . 3 4$ </td><td> $- 0 . 4 9 \pm 0 . 0 6$ </td><td> $1 . 2 0 \pm 0 . 2 1$ </td><td> $- 0 . 4 3 \pm 0 . 0 2$ </td></tr><tr><td>Proposed</td><td> $0 . 7 1 \pm 0 . 2 9$ </td><td> $- 0 . 2 5 \pm 0 . 1 1$ </td><td> $0 . 9 7 \pm 0 . 3 0$ </td><td> $- 0 . 4 7 \pm 0 . 0 5$ </td></tr></table>

TABLE III: FM statistics during continual learning (mean $\pm \ \mathrm { s t d } ) .$
<table><tr><td rowspan=1 colspan=6>FM, $f _ { k , j }$     II              Training from $\mathcal { D } ^ { 1 }$ to Dj</td></tr><tr><td rowspan=1 colspan=1>Test k</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1> $\mathcal { D } ^ { 1 } \to \mathcal { D } ^ { 2 }$ </td><td rowspan=1 colspan=1> $\mathcal { D } ^ { 1 } \to _ { \mathcal { D } ^ { 3 } } \mathcal { D } ^ { 2 } \to$ </td><td rowspan=1 colspan=1> $\mathcal { D } ^ { 1 } \to \dots \to$ </td><td rowspan=1 colspan=1> $\mathcal { D } ^ { 1 } \xrightarrow [ \mathcal { D } ^ { 5 } ] { } \cdots \xrightarrow [ ] { }$ </td></tr><tr><td rowspan=1 colspan=1> $\mathcal { D } ^ { 1 }$ </td><td rowspan=1 colspan=1>CMIMOSTWVNLwFNGRProposed</td><td rowspan=1 colspan=1> $0 . 6 0 \pm 0 . 2 9$  $0 . 7 3 \pm 0 . 1 0$  $1 . 5 3 \pm 0 . 0 1$  $1 . 6 0 \pm 0 . 1 2$  $0 . 8 0 \pm 0 . 0 5$  ${ \bf 0 . 6 7 \pm 0 . 0 5 }$ </td><td rowspan=1 colspan=1> $0 . 5 7 \pm 0 . I 2$  $1 . 1 2 \pm 0 . 1 2$  $2 . 0 8 \pm 0 . 0 3$  $2 . 2 4 \pm 0 . 2 3$  $1 . 3 8 \pm 0 . 0 8$  ${ \bf 1 . 1 1 \pm 0 . 0 7 }$ </td><td rowspan=1 colspan=1> $0 . 7 0 \pm 0 . 2 0$  $1 . 3 3 \pm 0 . 1 1$  $1 . 9 5 \pm 0 . 1 1$  $2 . 0 6 \pm 0 . 1 4$  $1 . 4 2 \pm 0 . 0 6$  ${ \bf 1 . 2 4 \pm 0 . 0 8 }$ </td><td rowspan=1 colspan=1>0.76 ± 0.171.55 ± 0.242.56 ± 0.032.68 ± 0.161.82 ± 0.211.45 ± 0.13</td></tr><tr><td rowspan=2 colspan=1> $\mathcal { D } ^ { 2 }$ </td><td rowspan=2 colspan=1>CMIMOSTWVNLwFNGRProposed</td><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1> $0 . I 2 \pm 0 . I 2$  ${ \bf 0 . 3 8 \pm 0 . 2 5 }$ </td><td rowspan=1 colspan=1> $0 . 2 0 \pm 0 . { \cal I } 4$  ${ \bf 0 . 6 7 \pm 0 . 1 9 }$ </td><td rowspan=1 colspan=1>0.22 ± 0.091.03 ± 0.51</td></tr><tr><td rowspan=1 colspan=1> $2 . 1 7 \pm 0 . 3 2$  $2 . 3 3 \pm 0 . 3 4$  $0 . 5 5 \pm 0 . 1 5$  $0 . 3 8 \pm 0 . 2 7$ </td><td rowspan=1 colspan=1> $2 . 2 8 \pm 0 . 3 9$  $2 . 2 6 \pm 0 . 3 0$  $0 . 6 8 \stackrel { \_ } { \pm } 0 . 1 4$ 0.68 ± 0.18</td><td rowspan=1 colspan=1>2.29 ± 0.192.72 ± 0.181.22 ± 0.37 ${ \bf 0 . 9 8 \pm 0 . 2 3 }$ </td></tr><tr><td rowspan=4 colspan=1> $\mathcal { D } ^ { 3 }$ </td><td rowspan=4 colspan=1>CMIMOSTWVNLwFNGRProposed</td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1> $0 . 3 5 \pm 0 . I 7$ </td><td rowspan=2 colspan=1>0.49 ± 0.201.51 ± 0.22</td></tr><tr><td rowspan=1 colspan=1>1.24 ± 0.23</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1> $3 . 0 7 \pm 0 . 3 3$  $2 . 8 5 \pm 0 . 1 1$ </td><td rowspan=2 colspan=1>3.18 ± 0.253.50 ± 0.45 $2 . 1 3 \pm 0 . 1 9$  $\mathbf { 1 . 4 9 \ : \pm { \ : 0 . 3 1 } }$ </td></tr><tr><td rowspan=1 colspan=1> $1 . 3 9 \pm 0 . 1 4$  ${ \bf 0 . 9 3 \pm 0 . 2 1 }$ </td></tr><tr><td rowspan=2 colspan=1> $\mathcal { D } ^ { 4 }$ </td><td rowspan=2 colspan=1>CMIMOSTWVNLwFNGRProposed</td><td rowspan=1 colspan=1></td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1> $0 . I 7 \pm 0 . I 2$  $0 . 3 6 \pm 0 . 1 5$  $1 . 6 0 \pm 0 . 2 1$  $1 . 5 2 \pm 0 . 1 7$  $0 . 2 9 \pm 0 . 1 1$  ${ \bf 0 . 2 5 \pm 0 . 1 6 }$ </td></tr><tr><td rowspan=1 colspan=1></td></tr></table>

Apart from the quantitative findings, we qualitatively evaluate the traversability prediction model’s ability to retain prior experience through navigation trials. Specifically, we assess the performance of model $\tau ^ { 5 }$ in Environments 1 after sequential training on Environments 1 through $5 \ ( \mathrm { i . e . , } \ D ^ { 1 } \ $ $\cdots \to { \mathcal { D } } ^ { 5 } )$ . To probe the impact of uncertainty-aware navigation under continual learning, the grass region in the park (treated as an OoD area) was excluded from data collection, while only the asphalt region (in-distribution) was traversed during training. Figure 6 illustrates representative trajectories from three trials per method, and Table IV summarizes the navigation results. Here, Goal reached denotes the number of successful goal completions out of three trials, and OoD avoidance indicates the number of trials in which the robot successfully avoided entering the grass region. The WVN method fails to reach the goal in one trial, reflecting degraded predictions due to forgetting. Other baselines, except CM and the proposed method, also fail to retain prior experience and incorrectly estimate uncertainty in the OoD region, resulting in poor avoidance performance. In contrast, the proposed method demonstrates stronger experience retention, reliably reaching the goal while avoiding OoD areas. Despite being trained without access to past data, it performs comparably to the CM method. These results demonstrate that our method effectively retains prior experience, enabling reliable navigation under continual learning scenarios.

![](images/8b5480769376a0ada14a57e7576f2ea951f15335ee8be112d6286e2cecb42490.jpg)  
Fig. 6: Navigation results in Environment 1 using the traversability prediction model T<sup>5</sup>, with the paths of each method overlaid on the field images.

TABLE IV: Navigation results across 3 trials per method. Values indicate the number of successful trials.
<table><tr><td></td><td rowspan="2">Method</td><td rowspan="2">CM IMOST</td><td rowspan="2"></td><td rowspan="2">WVN</td><td rowspan="2">LwF NGR</td><td rowspan="2"></td><td rowspan="2">Proposed</td></tr><tr><td>Metric</td></tr><tr><td>Goal reached</td><td>3</td><td>3</td><td>2</td><td>3</td><td>3</td><td></td></tr><tr><td>OoD avoidance</td><td>3</td><td>0</td><td>0</td><td>0</td><td>1</td><td>3</td></tr></table>

## VI. CONCLUSIONS

This paper presents a novel continual learning framework for traversability prediction in unstructured environments, addressing the challenges of adaptation and catastrophic forgetting without requiring explicit storage of past data. The core contribution is an uncertainty-aware adaptation strategy that selectively updates both the traversability prediction model and the generative experience recall model based on uncertainty estimates of recalled experiences. Real-world experiments with a skid-steering robot validate the effectiveness of the proposed method, demonstrating improved experience retention and reliable navigation performance. Future work will investigate more generalizable learning strategies that leverage prior knowledge, such as physics-informed foundation models, for traversability prediction.

## REFERENCES

[1] Christos Sevastopoulos and Stasinos Konstantopoulos. A survey of traversability estimation for mobile robots. IEEE Access, 10:96331– 96347, 2022.

[2] Juhana Ahtiainen, Todor Stoyanov, and Jari Saarinen. Normal distributions transform traversability maps: lidar-only approach for traversability mapping in outdoor environments. Journal ofField Robotics, 34(3):600– 621, 2017.

[3] Dongshin Kim, Jie Sun, Sang Min Oh, James M Rehg, and Aaron F Bobick. Traversability classification using unsupervised on-line visual learning for outdoor robot navigation. In Proceedings 2006 IEEE International Conference on Robotics and Automation, 2006. ICRA 2006., pages 518–525. IEEE, 2006.

[4] Mateus V Gasparino, Arun N Sivakumar, Yixiao Liu, Andres EB Velasquez, Vitor AH Higuti, John Rogers, Huy Tran, and Girish Chowdhary. Wayfast: Navigation with predictive traversability in the field. IEEE Robotics and Automation Letters, 2022.

[5] Xiaoyi Cai, James Queeney, Tong Xu, Aniket Datar, Chenhui Pan, Max Miller, Ashton Flather, Philip R Osteen, Nicholas Roy, Xuesu Xiao, et al. Pietra: Physics-informed evidential learning for traversing out-ofdistribution terrain. IEEE Robotics and Automation Letters, 2025.

[6] Jonas Frey, Matias Mattamala, Nived Chebrolu, Cesar Cadena, Maurice Fallon, and Marco Hutter. Fast Traversability Estimation for Wild Visual Navigation. In Proceedings of Robotics: Science and Systems, Daegu, Republic of Korea, July 2023.

[7] Liyuan Wang, Xingxing Zhang, Hang Su, and Jun Zhu. A comprehensive survey of continual learning: Theory, method and application. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[8] Kehui Ma, Zhen Sun, Chaoran Xiong, Qiumin Zhu, Kewei Wang, and Ling Pei. Imost: Incremental memory mechanism with online self-supervision for continual traversability learning. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 8788–8794. IEEE, 2025.

[9] Hyung-Suk Yoon, Ji-Hoon Hwang, Chan Kim, E In Son, Se-Wook Yoo, and Seung-Woo Seo. Adaptive robot traversability estimation based on self-supervised online continual learning in unstructured environments. IEEE Robotics and Automation Letters, 2024.

[10] Lorenz Wellhausen, Alexey Dosovitskiy, Rene Ranftl, Krzysztof Walas,´ Cesar Cadena, and Marco Hutter. Where should i walk? predicting terrain properties from images via self-supervised learning. IEEE Robotics and Automation Letters, 4(2):1509–1516, 2019.

[11] Anuj Pokhrel, Mohammad Nazeri, Aniket Datar, and Xuesu Xiao. Cahsor: Competence-aware high-speed off-road ground navigation in se(3). IEEE Robotics and Automation Letters, 2024.

[12] Xiaoyi Cai, Siddharth Ancha, Lakshay Sharma, Philip R Osteen, Bernadette Bucher, Stephen Phillips, Jiuguang Wang, Michael Everett, Nicholas Roy, and Jonathan P How. Evora: Deep evidential traversability learning for risk-aware off-road autonomy. IEEE Transactions on Robotics, 2024.

[13] Hojin Lee, Junsung Kwon, and Cheolhyeon Kwon. Learning-based uncertainty-aware navigation in 3d off-road terrains. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 10061–10068. IEEE, 2023.

[14] Michael Bloesch, Marco Hutter, Mark A Hoepflinger, Stefan Leutenegger, Christian Gehring, C David Remy, and Roland Siegwart. State estimation for legged robots: Consistent fusion of leg kinematics and imu. 2013.

[15] Gian Erni, Jonas Frey, Takahiro Miki, Matias Mattamala, and Marco Hutter. MEM: Multi-Modal Elevation Mapping for Robotics and Learning. In 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 11011–11018, 2023.

[16] Maxime Oquab, Timothee Darcet, Th ´ eo Moutakanni, Huy Vo, Marc´ Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

[17] Hojin Lee, Taekyung Kim, Jungwi Mun, and Wonsuk Lee. Learning terrain-aware kinodynamic model for autonomous off-road rally driving with model predictive path integral control. IEEE Robotics and Automation Letters, 8(11):7663–7670, 2023.

[18] Kihyuk Sohn, Honglak Lee, and Xinchen Yan. Learning structured output representation using deep conditional generative models. Advances in neural information processing systems, 28, 2015.

[19] Kenny Chen, Ryan Nemiroff, and Brett T Lopez. Direct lidar-inertial odometry: Lightweight lio with continuous-time motion correction. In 2023 IEEE international conference on robotics and automation (ICRA), pages 3983–3989. IEEE, 2023.

[20] Zhizhong Li and Derek Hoiem. Learning without forgetting. IEEE transactions on pattern analysis and machine intelligence, 40(12):2935– 2947, 2017.