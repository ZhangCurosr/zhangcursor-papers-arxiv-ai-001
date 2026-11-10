# Accelerated Reinforcement Learning for Soft Robotics: Hours Instead of Years

Roshan S. Kaundinya<sup>1†</sup>, Hehui Zheng<sup>2†</sup>, Robert K. Katzschmann<sup>2\*</sup>, George Haller<sup>1</sup>

<sup>1</sup>Institute for Mechanical Systems, ETH Z¨urich, Leonhardstrasse 21, 8092, Z¨urich, Switzerland.

<sup>2</sup>Soft Robotics Lab, ETH Z¨urich, Tannenstrasse 3, 8092, Z¨urich, Switzerland.

\*Corresponding author(s). E-mail(s): rkk@ethz.ch; georgehaller@ethz.ch; <sup>†</sup>These authors contributed equally to this work.

## Abstract

Reinforcement learning (RL) requires large amounts of interaction data or a reliable simulator, both hard to obtain for nonlinear soft robotic hardware. Recent work shows that a time-scale separation inherent to soft robot dynamics justifies their reduction to low-dimensional invariant manifolds, known as adiabatic spectral submanifolds (aSSMs). We demonstrate that an aSSM-reduced model, a simple polynomial ordinary diferential equation learned purely from pre-collected data, can serve as an RL training environment. On a pneumatically actuated soft arm, we train a trajectory-tracking policy in 3 hours on a six-dimensional aSSM-reduced model and deploy it without fine-tuning, whereas reproducing this policy training extrapolates to 3 years of hardware interaction. Our policy tracks dynamic targets with 40% lower error than otherwise identical policies trained on a 4.8-million-parameter recurrent neural network (RNN) or a 1.7- million-parameter transformer, which take 13 and 100 hours to identify, against 5 minutes for our mathematically justified 6,924-parameter model.

Keywords: spectral submanifolds, reinforcement learning, soft robotics, nonlinear dynamics, hardware, world models

Learning control policies for robots to excel at human- or animal-oriented skills, such as walking, swimming, flying, and playing physical or cognitive games, is an ambitious open problem of current interest in the machine learning and robotics communities (see Kober et al. [24] and Ibarz et al. [19] for a review). Since biological morphologies and their associated mechanics heavily inspire robot designs, the development of general-purpose controllers that emulate the trial-and-error learning processes of biological organisms to master complex skills has received broad interest.

Reinforcement learning (RL) formalizes control by trial and error (see Sutton and Barto [38]). An RL agent learns the control policy that maps an observation of the robot and the current target to a distribution over control actions. Control actions are sampled from the policy and applied to the robot. The resulting state is observed, and the policy parameters are updated to maximize a reward that quantifies the proximity to the target associated with the skill.

Methods that learn from rewards evaluated on interaction data, the model-free family, require significantly more data than a physical robot can usually provide, since every environment step costs wall-clock time and mechanical life of the hardware. Developments in rigid robotics alleviated this by moving the exploration of the hardware into high-fidelity, parallelizable simulators (see Makoviychuk et al. [27]), an approach that produced the locomotion (see Miki et al. [30]) and manipulation (see Kroemer et al. [25]) results now state of the art on a variety of rigid robotic platforms such as ANYmal and the Franka arm.

The same pipeline is generally not available for soft robots. For certain types of soft robotic hands (see Toshimitsu et al. [42]) and continuum arms (see Satheeshbabu et al. [36]), rigid-body physics engines such as MuJoCo, or elastic rod-based simulators (see Naughton et al. [33]) that approximate their kinematics already aid in training model-free RL policies. But they sufer from the simulation-to-reality gap for dexterous manipulation tasks. On the other hand, current soft robotic simulators that are able to capture the nonlinear dynamics (see Faure et al. [13], Mekkattu et al. [29], and Parthasarathy et al. [35]) are computationally costly to be integrated in the policy training loop. These practical limitations motivate the development of learning RL policies directly on hardware soft robot data.

Attempts to control soft robots through an RL policy trained solely on hardware data are not new. The earliest work by Braganza et al. [7] learned a model-free policy to control a soft robotic manipulator. The method, however, does not appear scalable to complex soft-robot geometries. Indeed, even with advances in RL algorithms and computational power, no successful model-free policies for complex soft robots have emerged to date. The reasons are not surprising, as model-free policies require extensive hardware exploration to converge, which, for delicate soft robots, is nearly impossible because their softness, coupled with repeated use, leads to memory efects and plasticity that tend to cause hardware failure.

At the crossroads of RL implementation dificulties: dealing with delicate hardware and no access to a reliable, fast simulator. Large black-box neural network-based models trained on dynamic hardware data to model the dynamics of soft robots ofer a solution. The underlying hope is that the neural network’s large parameter count provides a reasonable estimate of the soft robot’s high fidelity, and the more appealing aspect is that neural-network-based control policies are largely compatible with training on such models, ensuring fast, parallelizable training that explores the model’s state space. These are termed model-based RL methods and are further subclassified as online when policy training happens directly by interacting with hardware, or ofline when policy training happens using the model as a training environment.

On soft robotic manipulators, Thuruthel et al. [40] learns a nonlinear autoregressive model of the dynamics using a one-step memory window. They then employ this model inside a classical model predictive controller to generate target trajectories, on which an RL agent learns a control policy while interacting with the hardware. Recent work by Zheng et al. [44] and Nan et al. [32] uses a larger neural network with one-step memory information and trains simultaneously with a control policy on hardware data. These online model-based methods have seen success, but, as their authors point out, it is unclear whether the errors can be attributed to the policy or to model learning (see Janner et al. [20]). All in all, these approaches, though data-eficient, still converge to RL policies that are task-specific and need additional interaction with the hardware to be generally applicable.

To mitigate these issues, Nazeer et al. [34] and Zhang et al. [43] focus on training a long short-term memory (LSTM) and a multilayer perceptron (MLP), respectively, both with larger memory windows, and then train control policies on these models. However, they report accuracy gaps when applied to the hardware and hence use online model-free RL methods to fine-tune their predictions. For purely ofline modelbased methods for soft robots, Berdica et al. [3] and Centurelli et al. [12] use LSTMs to learn the dynamics of soft arms and train policies on them. Notably, the latter tracks dynamic targets using an LSTM-based RL policy, achieving accuracy on planar trajectories. Their precollected data for training the LSTM is modest. Still, they ofer no systematic guidelines for choosing the network architecture, hyperparameters, and memory windows, which are essential for LSTMs, which are prone to long-term prediction inaccuracies and poor generalization to unseen test data (see Kaundinya et al. [23]).

![](images/3c882024f31c177370ee76db86d76627d8000fd9f846935e91605fb13e619a4c.jpg)  
Fig. 1: [Overview of learning an aSSM RL policy on a soft robot]. (a) Snapshot of the SoPrA arm. (b) Geometry of the adiabatic spectral submanifold (aSSM) in the state- and control action- space. (c) General overview of training an RL agent on an aSSM-reduced model.

As an alternative to data-driven black-box modeling, the mathematical theory of spectral submanifolds (SSMs), which are low-dimensional attracting invariant manifolds (see Haller [15] for a review), provides a rigorous geometric understanding of the dominant dynamics of nonlinear dynamical systems. Specifically, restricting the dynamics to slow SSMs allows for justified model reduction of nonlinear continuum mechanical systems described by ordinary diferential equations (ODEs) (see Haller and Ponsioen [17] and Haller et al. [18]) or partial diferential equations (PDEs) (see Buza and Haller [8]). The resulting SSM-reduced model is a low-dimensional ODE, which can be extracted purely from data as a simple polynomial model (see Cenedese et al. [9] and Ax˚as et al. [2]), and has demonstrated predictive accuracy in a range of mechanical experiments subject to previously unseen external forcing (see Cenedese et al. [10], Cenedese et al. [11], Bettini et al. [4], Bettini et al. [5], and Morsy et al. [31]).

Dominant dynamics of nonlinear soft robots captured by slow SSMs have already been locally extracted near equilibria and used for controls in Alora et al. [1] for a hardware Diamond robot. Only recently, in Kaundinya et al. [21], a timescale separation phenomenon was identified as common to all soft robotics control operations, enabling the global extraction of the dominant dynamics onto adiabatic spectral submanifolds (aSSMs) via extensions to the local theory of SSMs (see Haller and Kaundinya [16]). They show, across a variety of soft robotic simulators, that aSSMs extracted from data can be used in a model predictive control (MPC) scheme successfully for accurate trajectory tracking.

This methodology has very recently been demonstrated on hardware, by further reducing aSSMs onto slow manifolds to control artificial muscles (see Bettini et al. [6]) and to precisely control helicoid soft robots along 3D target tracks (see Kaundinya et al. [22]). Existing aSSM-based controllers pair the reduced model with classical numerical control techniques, which linearize the dynamics at each solve step. As a consequence, the nonlinear structure of the aSSM model is only partially exploited. Control methods, such as an RL agent optimizing policies directly on nonlinear model rollouts, without repeated linearization, have therefore remained unexplored for aSSMreduced models. We will fill this gap with the present work.

Contribution: We develop a method to train RL control policies ofline on a datadriven aSSM-reduced model (see Fig. 1c), without the RL training interacting with the hardware robot. Specifically:

• [aSSM for a new hardware platform] We extract and learn a six-dimensional aSSM-reduced model for a pneumatically actuated soft robot, a soft continuum proprioceptive arm (SoPrA) (see Fig. 1a and Section 1.1), purely from hardware data. Previously, the data-driven aSSM methodology has been tested on tendondriven trunk robots (see Kaundinya et al. [22]). This demonstrates the generality of our method, which is agnostic to the physics of the actuating mechanics.

• [Pareto dominant aSSM-RL policies] Combining traditional RL policy optimization (see Schulman et al. [37]) with our aSSM-reduced model (see Section 1.3) leads to Pareto dominant aSSM-RL control policies across model complexity, model learning time, RL policy training time, and closedloop accuracy (see Section 1.4). We use baselines that are state-of-the-art autoregressive neural network-based models from Li et al. [26], which they term robotic world models (RWMs).

• [Faster control] Our aSSM-RL policy operates at a controller frequency of 100 [Hz], which provides a 100x speed-up over neural-network-based MPC controllers in Zheng et al. [44] and a 2x speed-up over aSSM-based MPC controllers in Kaundinya et al. [22].

• [Improved closed-loop accuracy compared to prior work] Our aSSM-RL policy reduces closed-loop tracking error by 40% relative to the bestperforming autoregressive model-based baseline, across all trajectory tracking tasks (Section 1.5), and by 20% relative to our own aSSM-MPC baseline. Furthermore, aSSM-RL reduces tracking error by 55% relative to the LSTMbased policy of Centurelli et al. [12] and by 76% relative to the online model-based policies of Nan et al. [32].

## 1 Results

We present a detailed case study on the performance of model-based RL policies for controlling a soft continuum arm. Specifically, our experimental platform is the SoPrA arm (see Toshimitsu et al. [41] for details), a pneumatically actuated soft continuum arm. The SoPrA arm is mounted vertically and hangs against gravity (see Fig. 1a). The cross-sectional radius of the arm at the free end is 2 [cm] and 3 [cm] at the base. It has a longitudinal length of 30 [cm]. The arm is assembled by putting together two modules. Each module is cast in silicone material and comprises three expandable chambers. The chambers are connected to a pressure-driven actuation system that sets the pressure in each chamber to control the extent of expansion. We refer to the Supplementary Information for further hardware details.

An external motion capture system monitors the arm’s movements. The camera system records the positions of the free end, which we refer to as the tip, and the midsection, which we refer to as the mid, where the modules meet. Our objective is to learn a model of the soft arm’s dynamics using data collected by the camera system and the control actions applied to the robot to generate it. We then use this learned model to explore the soft arm’s control action space and learn an RL control policy for target tracking tasks.

On the modeling side, in Section 1.1, we provide a detailed discussion of learning a low-dimensional 6D aSSM-reduced model for the arm. We also learn various, routinely used neural-network-based dynamics models for the arm’s end-efector: multi-layer perceptrons (MLPs), recurrent neural networks (RNNs), and Transformers. Specifically, the RNN and Transformer baselines are the current state-of-the-art in recent work by Li et al. [26], which they denote as robotic world models (RWMs). We systematically compare the predictive power of all these models, including our aSSMreduced model, for previously unseen random control actions in Section 1.2. The exact training details for all the neural network models used for comparisons are provided in the Supplementary Information.

We then use our learned aSSM model in Section 1.3 to train a model-based RL policy for closed-loop control of the soft arm and follow this up in Section 1.4 by outlining the Pareto dominance of using an aSSM-based RL policy compared to other neural-network-based policies. We finally deploy our aSSM-based RL policy on the hardware arm in Section 1.5 to perform closed-loop tracking of 2D targets.

## 1.1 Data-driven modeling of the soft robotic arm via aSSMs

As a continuum, the SoPrA soft arm is an infinite-dimensional dynamical system. Under soft-material constitutive laws and elastic continuum assumptions, the system’s continuum equations of motion can be discretized to construct high-fidelity, highdimensional dynamical systems. However, there are no robust high-fidelity simulators that faithfully model the hardware. Thus, we must rely only on data from a partially observable state space to infer a dynamical model for the SoPrA arm.

Under normal operating condition, the SoPrA arm exhibits an underlying timescale separation: its intrinsic vibration rates die out significantly faster than the rate of change of control actions generated by the actuation system. Evidence of this timescale separation has recently been documented in soft robotic hardware: the artificial muscle joint system of Bettini et al. [6] and the soft helicoid trunk robot of Kaundinya et al. [22].

Relevant to our setting, the theoretical work by Haller and Kaundinya [16] extended to the control action setting by Kaundinya et al. [21] provides proofs for the existence of a family of low-dimensional invariant manifolds called adiabatic spectral submanifolds (aSSMs), in the phase space of dynamical systems satisfying the timescale separation assumption outlined above. Specifically, for target-tracking control, normally attracting aSSMs, , arise as the nonlinear continuations of the collection of dominant slow eigenspaces of the linearization of the robotic system about the frozen-time $( \mathrm { i . e . , } \epsilon = 0 )$ limit of a target trajectory $\mathbf { \boldsymbol { \Gamma } } ( \epsilon t )$ (see Fig. 1b). An arbitrary initial condition in the phase space of the robot will converge quickly onto the aSSM and then later converge to the target trajectory. Hence, restricting the dynamics to $\mathcal { A } _ { \epsilon }$ provides an accurate reduced model of the soft robot that can be used for controls.

Kaundinya et al. [21] and Kaundinya et al. [22] provide rigorous, data-driven methods for extracting low-dimensional, normally attracting aSSMs from partially observable state space coordinates of soft robots. We will closely follow their setup but use notation consistent with standard conventions in the Reinforcement Learning (RL) literature (see Sutton and Barto [38]). We define the partially observable state space of the arm as

$$
s \in \mathcal { S } \subseteq \mathbb { R } ^ { n _ { s } } ,\tag{1}
$$

and the finite-dimensional control action,

$$
\pmb { a } \in \mathcal { C } \subseteq \mathbb { R } ^ { n _ { a } } ,\tag{2}
$$

where $s$ and denote the physically admissible state and control action subsets, respectively. In our current hardware setup for the SoPrA arm, we can only observe the tip and mid positions. Hence we have $n _ { s } = 6$ , and the observable state vector takes the form

$$
\begin{array} { r } { s = \{ x , y , z \} = \{ x _ { \mathrm { t i p } } , x _ { \mathrm { m i d } } , y _ { \mathrm { t i p } } , y _ { \mathrm { m i d } } , z _ { \mathrm { t i p } } , z _ { \mathrm { m i d } } \} \in \mathbb { R } ^ { 6 } . } \end{array}\tag{3}
$$

The pressure-driven actuation system of SoPrA has 6 air chambers, but due to current hardware limitations, only 5 are active and available to the user, which sets $n _ { a } = 5 .$

We investigate the presence of normally attracting aSSMs in the observable space s of the $\mathrm { S o P r A }$ arm by commanding random step control action sweeps to it. These sweeps sufice to probe the leading-order aSSM geometry and its internal dynamics. As shown in Fig. 2a, based on data collected from a 2.2-hour step sweep of 400 randomly sampled unique static control actions held for 20 seconds each, the SoPrA arm undergoes underdamped decays that eventually converge to unique static configurations in the observable space. Both of these characteristics support the assumption that normally attracting aSSMs exist in the phase space of the SoPrA arm (see Section 3).

Using a spectrogram analysis of each decaying signal, we infer from the collected data, that the minimal dimension of the normally attracting fibers of the aSSM is $d \ : = \ : 6$ . This can also be deduced from the geometry of the SoPrA arm, which is symmetric in the $x - y$ plane neat the rest state. This implies that the modes of vibration along these axes are 1-1 resonant, which calls for the inclusion of at least 2 modes in the aSSM construction. The arm also admits longitudinal vibrations in the z direction, which constitute another mode of vibration, thus explaining the 3 dominant frequencies observed in the spectrogram analysis. All these minimal sets of modes are underdamped for the SoPrA arm, which is qualitatively diferent from the overdamped dynamics of tendon-driven trunk helicoid robots (see Kaundinya et al. [22]).

Our next step is to use the collected step control data and the inferred dimension $d \ : = \ : 6$ of the aSSM fibers to learn the aSSM-reduced model for the SoPrA arm. That consists of three learnable functions: the critical manifold parametrization, which maps static control actions to static states; the aSSM parametrization; and the aSSM-reduced dynamics. The critical manifold parametrization is learned from the static states and control actions obtained by postprocessing the step control data in the observable space s. The aSSM parametrization and dynamics cannot be learned directly in $s ,$ as velocity information is required to model the dynamics, and an ambient space is needed to guarantee the embedding of the d-dimensional fibers of the aSSM geometry. To address these requirements, we employ delay embedding. Specifically, we construct a delay-embedded observable space by stacking the current state with its two time-delayed copies, in which a 6D fiber of the aSSM geometry can be embedded (see Takens [39]). We then use the collected step control data to learn all the aSSMrelated functions in this delay-embedded observable space, following the procedure outlined in Section 3.1.

The resulting aSSM-reduced model, which we will term the aSSM model, is of the form

$$
\begin{array} { r l } & { \mathbf { r } = \mathbf { V } ^ { \mathrm { T } } ( \pmb { a } ( t ) ) \pmb { \mu } \big ( \pmb { s } - \pmb { \bar { s } } ( \pmb { a } ( t ) ) \big ) , } \\ & { \dot { \mathbf { r } } = \mathbf { R } \big ( \mathbf { r } , \bar { \pmb { s } } ( \pmb { a } ( t ) ) \big ) , } \\ & { \pmb { s } = \mathbf { W } \big ( \mathbf { r } , \bar { \pmb { s } } ( \pmb { a } ( t ) ) \big ) + \bar { \pmb { s } } ( \pmb { a } ( t ) ) , } \end{array}\tag{4}
$$

where $\textbf { r } \in \ \mathbb { R } ^ { 6 }$ is reduced coordinates on the aSSM; $\mathbf { V } ( \cdot )$ is the chart map of the aSSM; $\bar { s } ( \cdot )$ is the critical manifold parametrziation; µ is a function performing delay embedding; $\mathbf { R } ( \cdot )$ is the aSSM reduced dynamics; and $\mathbf { W } ( \cdot )$ is the aSSM parametrization. We have found that quadratic polynomial approximations for $\mathbf { W } ( \cdot )$ and R( ), and cubic polynomial approximations for $\bar { s } ( \cdot )$ sufice to model the dynamics of the SoPrA arm accurately.

We plot the aSSM model’s predictions against the SoPrA arm’s true response on previously unseen step control actions in Fig. 2b and on a previously unseen periodic control action sequence in Fig. 2c. We observe that aSSM predictions qualitatively capture the decay profiles of all observable states. The normalized mean trajectory error achieved by the aSSM model on these test data was 9% for step control data and 3% on the periodic control action data. We note that we do not have exact matches for any of the cases, and we attribute this to a hardware inaccuracy in the pressure actuation system. To illustrate this inaccuracy, in Extended Data Fig. 1a, we plot the commanded pressure action variable $\mathbf { a } _ { 2 }$ and compare it with the actual measured pressure in the chamber. We observe that, at all times, the pressure valves keep fluctuating and exhibit lags that lead to minor fluctuations in the arm’s response (see Extended Data Fig. 1b). This hardware issue is common in pressure control systems that are operated in a more dynamic regime. A mismatch between the commanded and measured pressure will result in accuracy issues for modeling and control in the subsequent sections. We aim to report results despite these limitations and test closed-loop control performance.

## 1.2 Predictive power of models on unseen control actions across the full action space

As a comparison, we also learn the $\mathrm { d y }$ namics of the $\mathrm { S o P r A }$ arm by fitting neuralnetwork-based models to trajectory data. This is done by approximating the state evolution as an autoregressive neural network (b-m), defined as a map f,

$$
\begin{array} { r } { \left( s _ { k + 1 } , \ldots , s _ { k + m } \right) = \mathbf { f } \left( s _ { k } , s _ { k - 1 } , \ldots , s _ { k - b + 1 } , \ a _ { k } , a _ { k + 1 } , \ldots , a _ { k + m - 1 } \right) , } \end{array}\tag{5}
$$

where b represents the memory window, and m the forecasting window. The training data used to fit the neural networks is generated by issuing random-walk-like control actions that uniformly explore the state space. Importantly, to ensure a fair comparison, we keep the total amount of time-series training data collected for the neural network consistent with that collected for the aSSM model in Section 1.1. The exact training details for all neural network-based models on this dataset are provided in the Supplementary Information. We learn four neural-network-based models: two multilayer perceptrons (MLPs), one with $( b \cdot m ) \equiv ( 3 \cdot 1 )$ and the other with $( b / m ) \equiv ( 3 / - 8 )$ , a recurrent neural network (RNN), and a Transformer both with (b-m)  (32-8) and trained using state-of-the-art autoregressive training methods from Li et al. [26].

![](images/847e590789d0e87a645e75db328dcaeea301a5775533ba13b6ef6ec4750a264c.jpg)

![](images/c978a6b1dbec8918e2a984c2671baa7ed2feb4e0501e6d7ca955e9514dd4d5d0.jpg)

![](images/ff25070d7c2dad8dd24e48b7d04c1294873ebda5a99054d741022c0c77c6a7d8.jpg)

![](images/eaa575f52bf3cb29c2a7db2c6295e25bba8fa433b77719237b49b7bc8a7f72af.jpg)  
(c)

![](images/2c8796f3d39067cb05866bad2a4245395f0caace613213bf02216b80ca11df64.jpg)  
(b)  
Fig. 2: [Modeling SoPrA via an aSSM-reduced model]. (a) Response of the SoPrA arm plotted in its state-space subjected to random step control actions. (b) Time series plots of the aSSM model predictions (green) and the SoPrA arm’s true response (black) for unseen step control actions in the workspace of the SoPrA arm. (c) State-space plots of the aSSM model’s predictions (green) and the SoPrA arm’s true response (black) for a periodic control action.

We test the predictive power of all these models, including our aSSM model, on 50 previously unseen continuous random control action sequences, uniformly sampled from the entire control action space. We pass these control action sequences to the SoPrA arm to generate the true responses, and we plot this with the model predictions in the $x _ { \mathrm { t i p } } - y _ { \mathrm { t i p } }$ plane. We present these results in Fig. 3 for four randomly chosen control action sequence runs. We observe that the aSSM model across these sets has the lowest mean error, and its predictions lie close to the true response of the arm.

The RNN and Transformer perform decently, qualitatively tracking the target, but remain inaccurate. In particular, the RNN tracks the true response reasonably well in the $x _ { \mathrm { t i p } } ~ > ~ 0$ region but loses accuracy in other regions. In contrast, the Transformer uniformly exhibits larger errors throughout the workspace. On the other hand, both vanilla MLPs with simpler autoregressive network structures perform extremely poorly, with the error even diverging on some runs. Increasing the memory window from $b = 1$ to b = 8 occasionally reduces the error but also sufers from the same instabilities for other random runs. None of the autoregressive models provides a consistent and accurate prediction, which is not surprising, as the SoPrA arm does not exhibit significant memory efects due to material hysteresis or actuation delays in the collected data. We provide a detailed comparison and reasoning for these model behaviors in Section 1.4.

![](images/31abf60be592cce67b10393c441746e4b3f8271624e68555ff6973abdabb7f2f.jpg)  
Fig. 3: [Prediction results comparing the aSSM-reduced model against neural-network-based dynamics models]. Plots in the $x _ { \mathrm { t i p } } - y _ { \mathrm { t i p } }$ plane for model predictions on four randomly sampled continuous control action sequences compared against the true response of the SoPrA arm (black). The mean error [mm] of the model prediction is reported in the top-left corner of each plot.

## 1.3 aSSM-based RL policy for the soft arm

We use the learned aSSM model from Section 1.1 to learn a reinforcement learning (RL) policy for tracking 2D random target tracks $\mathbf { \Gamma } \Gamma ( t ) = \{ \mathbf { \Gamma } \Gamma _ { k } \} _ { k = 1 } ^ { T }$ in the $x _ { \mathrm { t i p } } - y _ { \mathrm { t i p } }$ plane of the SoPrA arm. The RL policy training proceeds by learning a conditional probability distribution over the space of actions, $\pi _ { \boldsymbol { \theta } } ( \boldsymbol { a } , \cdot )$ , as a neural network.

A control action $\mathbf { \delta } _ { \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } } \mathbf { \Xi } _ { \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathrm { ~ \textit ~ { ~ a ~ k ~ } ~ } }$ at time step k is sampled from $\pi _ { \theta }$ given an observation vector that depends on the current state $\mathbf { s } _ { k }$ , the previous state $\mathbf { s } _ { k - 1 }$ , the previously sampled control action $\mathbf { \delta } _ { a _ { k - 1 } }$ , and the next target state $\Gamma _ { k + 1 }$ . This sampled control action is passed to the aSSM model, which predicts the next state $\mathbf { s } _ { k + 1 }$ . A reward $\rho _ { k } = \rho ( \mathbf { s } _ { k + 1 } , \mathbf { a } _ { k } , \mathbf { r } _ { k + 1 } )$ is evaluated based on these predictions, the control action that generated it, and the target state. A cumulative reward sum, called the episode return, is computed after rolling out the model and policy across the target track horizon.

Following standard ofline model-based RL techniques, we launch and compute episode returns in parallel across a large set of initial conditions sampled from the training data used to fit these models. This procedure is referred to as imagination in the RL community. The weights θ of $\pi _ { \theta }$ are updated to maximize the mean episode return over an imagination cycle, and each update is defined as a training iteration of the RL policy. In Section 3.2, we provide details on the problem formulation and the RL training algorithm, and in the Supplementary Information, we provide the reward function, the hyperparameters, and the number of initial conditions launched in one imagination cycle.

We perform the same RL training procedure using only the RNN and Transformer models discussed in Section 1.2, since these models provided stable predictions. In Extended Data Fig. 2, we plot the mean imagination episode return for increasing training iterations to validate if the policy updates indeed converge by maximizing the reward. We plot these for three policies: aSSM-RL, a policy trained on an aSSM model; RNN-RL, a policy trained on an RNN model; and Transformer-RL, a policy trained on a Transformer model. We observe that aSSM-RL smoothly and rapidly reaches a good mean imagination episode return value within 200 iterations, and steadily improves as the training iterations increase. Both the RNN-RL and Transformer-RL only reach a decent mean imagination episode return value after 800 training iterations. Compared to aSSM-RL, training neural network model-based policies often stagnates, and improvements come as short performance gains. They also show erratic behavior as the hyperparameter is swept (shaded region in Extended Data Fig. 2), signaling the black-box nature of these models.

We perform a total of 5, 000 training iterations, which ensures convergence of all policies. We note that the successful performance of final converged policies on the model does not imply the same on the hardware SoPrA arm. We follow up on this point with a discussion in Section 1.4 that systematically compares various features of these models for hardware deployment.

## 1.4 Pareto dominance in model complexity, learning time, and accuracy

We compare here the data-driven models and the RL policies trained on them in terms of model complexity, computation time, and ability to accurately capture hardware dynamics and performance accuracy in closed-loop on hardware.

Our first comparison is a Pareto plot in Fig. 4a that focuses only on the performance of the dynamical models. We use the 50 randomized control action experiments from Section 1.2 to plot the mean prediction error of the models across these runs along the vertical axis in Fig. 4a. The horizontal axis in this figure shows the time taken to learn the various models, with the training dataset size held constant. The number of model parameters is denoted by bubbles, with each bubble’s radius indicating the size of the model. Across all metrics, the aSSM model is Pareto-dominant and defines the Pareto front. As a simple polynomial model (see Section 3.1 and eqs. (17)-(18)), the aSSM model requires only 5 minutes to learn, which is orders of magnitude lower than the learning time of the other models. The RNN model lies closest to the Pareto front, where it performs reasonably well in terms of predictive accuracy. However, it is ultimately too complex, requiring significantly longer learning times on the order of hours. The Transformer model is much further from the Pareto front. Despite having fewer parameters than the RNN model, its architecture requires a massive 100 hours of learning time. The instability of MLPs keeps them out of this plot. They are easier to learn because of their moderate sizes, but their erratic accuracy prevents their use for controls. Hence, we omit these models for learning control policies.

Pareto plot of model learning time, accuracy and complexity  
![](images/8908c920c872339f588aa9fbd49b7cdf9ce9daedeee8d23277c83db98d044fe2.jpg)  
(a)

Pareto plot for RL policy training time, accuracy and complexity  
![](images/03e3c992eca9b2abf677b7863bca26d3ef4ffe41871bf4f606ea220839ab3c02.jpg)  
(b)  
Fig. 4: [Pareto plot comparisons for learned models and policies]. (a) Mean prediction error of the SoPrA arm’s tip position for the random unseen control action experiments in Section 1.2, plotted against model learning time, with bubble size denoting the number of model parameters. The aSSM model is Pareto dominant across these metrics. (b) Mean prediction error of the SoPrA arm’s tip position across the closed-loop runs in Section 1.5, plotted against policy training time, with bubble size denoting the number of model parameters. The aSSM-RL policy is Pareto dominant across these metrics.

For the trained model-based RL policies, we ofer similar comparisons of the figure, in Fig. 4b, but now with the vertical axis denoting mean closed-loop prediction accuracy on hardware across the tasks presented in Section 1.5. The horizontal axis of the figure denotes the time taken to train the RL policies. The aSSM-RL is still a Pareto-dominant model. In this case, the RNN-RL is close to the Pareto front but ultimately loses out on closed-loop accuracy, and the RNN model architecture is approximately 500 times more complex than the aSSM model, which adds to the computational burden of training the RL policy only. We note here that Transformer, though considered state-of-the-art, appears to perform the worst across all metrics. Overall, we attribute these trends in neural network models to the limited dataset they can access during training, a practical constraint imposed by hardware.

## 1.5 Closed-loop target tracking results

We finally close the results section, with closed-loop experiments on the hardware SoPrA arm using our trained aSSM-RL policy. We set up two target shapes in the $x _ { \mathrm { t i p } } \mathrm { ~ - ~ } y _ { \mathrm { t i p } }$ plane, the lemniscate resembling a Figure-8 pattern, and a square. The lemniscate has a maximum span of 100 [mm] in length and 50 [mm] in width, and the target track moves at a speed of 17.5 $[ \mathrm { m m } / \mathrm { s } ]$ . The square’s side length is 100 [mm]. Both of these targets span the maximal regions of the SoPrA arm’s workspace (see Fig. 2a). For the square target, we additionally consider three cases varying in speed. The slowest has a speed of $2 7 . 5 [ \mathrm { m m } / \mathrm { s } ]$ , while the next two are 1.6 and 3.2 times faster.

In Fig. 5a, we provide closed-loop results for these target tracks with comparisons against RNN-RL, Transformer-RL, and also aSSM-MPC, a model predictive controller (MPC) similar to the one used in Kaundinya et al. [22], minimizing a quadratic cost to track the target. We quantitatively evaluate the closed-loop prediction error to the target and plot these as bar plots in Fig. 5b. We note that our error penalizes spatial track-following. If we also account for strict temporal accuracy, all closed-loop policy prediction errors increase by a factor of 2 [mm] on the lemniscate and the slow square due to hardware latency (see the Supplementary Information). The aSSM-RL method achieves the best performance across various target tracks, with aSSM-MPC following it closely as expected. Both neural-network-based models trace out a track, but as the error increases, they become unstable, causing them to oscillate a bit and leading to larger errors.

We recall that the validity of using the aSSM model for controls relies on the time-scale separation assumption. We can quantitatively verify the assumption on a target track by evaluating a non-dimensional slowness measure $r _ { s } ,$ defined in eq. (14). The lemniscate and the slow-speed square have $r _ { s } ~ = ~ 0 . 1$ . The faster squares have an $r _ { s } ~ = ~ 0 . 3$ and $r _ { s } ~ = ~ 0 . 8$ . We expect the theory to hold when $r _ { s } ~ \ll ~ 1$ , and in practical problems, we have observed that even for $r _ { s } ~ < ~ 1$ we obtain good closedloop performance (see Kaundinya et al. [21] and Kaundinya et al. [22]). This is indeed the case across the target tracks considered here, with performance degrading slightly as the target speed increases. The aSSM-MPC’s limitation is apparent in the fastest target tracks, since it operates at 50 Hz due to the computational burden of solving the problem in real time. The aSSM-RL policy, on the other hand, operates at 100 Hz, which avoids the latency issue of MPC methods.

![](images/7370aebaa25ded8426169de7ee4d7bad78d3fb559a6d78882e42c7f36da842bc.jpg)

Average target tracking error comparisons  
![](images/2ceecd6b5969f5b36ac9a74b0055b2fa5af45502d7b848b0ab81d9420a3d7055.jpg)  
(b)  
Fig. 5: [Closed-loop experiments using learned model-based RL policies]. (a) Closed-loop results for four target tracks for aSSM-MPC (blue), aSSM-RL (green), RNN-RL (yellow), and Transformer-RL (red). Solid lines show the mean over 10 repeated runs, and shaded regions show the deviation. (b) Average tracking error compared across all target tracks. The deviation bar indicates the performance variation across 10 repeated runs of the closed-loop experiments.

We also provide snapshots of the SoPrA arm during closed-loop operation for all target tracks and across all policies in Fig. 6. We note that the aSSM-RL achieves simpler nonlinear shape profiles for the arm, ensuring easy tracking. In contrast, the RNN-RL and transformer-RL assume extreme bending shapes, which prevent continuous, consistent tracking of the target.

aSSM-MPC  
aSSM-RL  
RNN-RL  
Transformer-RL  
![](images/cc4e3eeb8bb82bef434416810e451812524729689b34957ebe0b2e7baf5e0b16.jpg)  
Fig. 6: [Closed-loop control of SoPrA]. Snapshots of the SoPrA arm during the various closed-loop experiments for all the target tracks. Each snapshot comes from one run, randomly selected from the 10 repeated runs shown in Fig. 5a. The traced path by the tip of the arm is plotted as a thick line for each prediction in its respective color.

## 1.6 Closed-loop results with potential hardware upgrades in the pressure-driven actuation system

In Section 1.1, in particular in Extended Data Fig. 1a, we highlighted the hardware limitation that the controller-commanded pressure does not exactly match the measured pressure in the chambers. We expected this mismatch to cause accuracy issues during closed-loop experiments on hardware. Indeed, we observe that no policy prediction is spared, in Fig. 5a, they appear as minor wiggles along the entire closed-loop prediction.

Upgrading to more expensive pressure valves is expected to improve accuracy by reducing the mismatch between commanded and measured pressure caused by the dynamic nature of the tasks. It will also allow the practitioner to test their methods on tasks involving faster target tracks. To aid the practitioner in deciding on the upgrade, from the mismatch data in Extended Data Fig. 1a, we evaluated numerically that a moving-average procedure with a window of 0.3 seconds (the current time lag in the pressure valve between commanded and measured) removes most of the wiggles, thus mimicking closed-loop predictions when the mismatch is absent. We applied this filtering procedure to our policy prediction from Section 1.5 and report these results in Extended Data Fig. 3a, and also provide revised mean error bar plots in Extended Data Fig. 3b.

The results show promise that upgrading to better valves guarantees improved performance across all target tracks. Only large oscillations intrinsic to the SoPrA arm are visible, which the aSSM model accurately models and limits compared to the RNN and Transformer models. The error bars also point to an interesting trend. Our heuristic numerical filtering procedure reduces errors in the RNN-RL and Transformer-RL predictions to a greater extent than in the aSSM model, indicating that, despite these pressure mismatch issues, our aSSM-RL method explores the best possible policy to reduce the existing hardware mismatch error.

## 2 Discussion

We have demonstrated that hardware soft robots can be accurately controlled for dynamic target tracking via Reinforcement Learning (RL) control policies trained solely on a faithful, reduced dynamics model of the soft robot. This advance is made possible by the fact that the dominant dynamics of soft robots lie on a very lowdimensional adiabatic spectral submanifold (aSSM). Recent methods allow for the rapid extraction of reduced models based on aSSM theory from hardware data alone.

For a pneumatically actuated SoPrA soft robotic arm, we have learned a 6D aSSM model from only partial observational data recorded by cameras (see Section 1.1). The learned 6D aSSM model delivers strong predictive capability and generalizes to timevarying control actions not present in the training dataset (see Section 1.2). When compared with state-of-the-art autoregressive neural-network-based models, our aSSM model is Pareto dominant across prediction accuracy and model learning time, ofering the fastest model to learn and the simplest with parameters in the thousands compared to the others in the millions (see Section 1.4). This simplicity also allows the aSSM model to be used in classical model predictive control schemes for real-time control, which are otherwise intractable to use with large models (see aSSM-MPC results in Section 1.5).

RL control policies trained on our aSSM model have shown stable performance across a wide range of RL hyperparameters and converge rapidly in a couple of hours, unlike policies trained on autoregressive neural-network-based models such as the RNN and the Transformer (see Section 1.3). These RL policies were trained without interacting with the hardware soft robot and only explore the control action space using models learned on two hours of hardware data. By machine-learning standards, this is a small dataset, so one would expect the resulting RL policies to transfer poorly to the real world.

Our closed-loop results in Section 1.5 support this expectation: the RNN-RL and Transform-RL policies produce large errors when tracking targets in real time. In contrast, the aSSM-RL policy provides consistent, accurate closed-loop performance. This is because, the aSSM provides a faithful reduced-order representation of the robot’s dynamics and therefore requires only data capturing the dominant behaviors that are ubiquitous and thus do not demand large volumes of data, unlike black-box approaches.

Our discovery of reliable, data-eficient aSSM-RL policies for soft robots opens exciting avenues for applying these techniques to challenging, application-oriented problems in interactive environments, such as surgical assistance or underwater exploration. Learning aSSM-RL policies for soft robotic hands performing dynamic dexterous manipulation is already within reach and currently under exploration. A longer-term goal is to learn aSSM models from video alone, without physical markers. Policies designed using such models will allow us to control complex soft robot morphologies and operate them in unstructured environments.

## 3 Methods

## 3.1 Adiabatic Spectral Submanifolds: Theory and Data-Driven Methodology

## Theory

The governing equations of soft robots can be written abstractly as a first-order dynamical system

$$
\begin{array} { r } { \dot { \mathbf { x } } = \mathbf { F } ( \mathbf { x } , \pmb { a } ( t ) ) , \qquad \mathbf { x } \in \mathcal { X } \subseteq \mathbb { R } ^ { n } , \quad \pmb { a } \in \mathcal { C } \subseteq \mathbb { R } ^ { n _ { a } } , } \end{array}\tag{6}
$$

with n typically very large due to the continuum nature of the problem. Here x is the full phase space of the robot, a is the control action drawn from the admissible set ${ \mathcal { C } } ,$ and F is a smooth nonlinear function that captures the geometric and material deformations of the arm.

For a static control input $\bar { \textbf { a } } \in \mathcal { C }$ , the robot deforms into a static configuration. Assuming that the collection of all these static configurations are attracting across ${ \mathcal { C } } ,$ they form a normally attracting critical manifold (see Fenichel [14]),

$$
\begin{array} { r } { \mathcal { L } _ { 0 } = \big \{ ( \mathbf { x } , \bar { \pmb { a } } ) \in \mathcal { X } \times \mathcal { C } \ : \ \mathbf { x } = \bar { \mathbf { x } } ( \bar { \pmb { a } } ) \big \} . } \end{array}\tag{7}
$$

If we further assume the linearization about each static configuration has a strict spectral splitting after d slowest eigenvalues, uniformly over $\mathcal { C } _ { : }$

$$
\operatorname { R e } ( \lambda _ { n } ( { \bar { a } } ) ) \leq \cdots \leq \operatorname { R e } ( \lambda _ { d + 1 } ( { \bar { a } } ) ) < \operatorname { R e } ( \lambda _ { d } ( { \bar { a } } ) ) \leq \cdots \leq \operatorname { R e } ( \lambda _ { 1 } ( { \bar { a } } ) ) < 0 ,\tag{8}
$$

we can define the slow spectral subbundle $\mathbf { E } ( \bar { \mathbf { a } } )$ associated with the eigenspaces of the d slowest eigenvalues. Under further non-resonance conditions, this subbundle of the linearized problem survives as a slow spectral submanifold bundle

$$
\mathcal { A } _ { 0 } = \mathcal { W } \big ( \mathbf { E } ( \bar { \mathbf { a } } ) \big ) ,\tag{9}
$$

which is also normally attracting and anchored to the critical manifold $\mathcal { L } _ { 0 }$ (see Haller and Kaundinya [16]). The slow spectral submanifold bundle $\mathcal { A } _ { \mathrm { 0 } }$ captures the dominant dynamics of system eq. (6) with static control actions.

Haller and Kaundinya [16] and Kaundinya et al. [21] show that both $\mathcal { L } _ { 0 }$ and $\mathcal { A } _ { \mathrm { 0 } }$ survive into nearby difeomorphic invariant manifolds, $\mathcal { L } _ { \epsilon }$ the slow manifold and $\mathcal { A } _ { \epsilon }$ the adiabatic spectral sumanifold (aSSM) when the control action is of the form

$$
\pmb { a } ( t ) = \pmb { a } ^ { s } ( \epsilon t ) + \pmb { a } ^ { d } ( t ) ,\tag{10}
$$

with $\| \pmb { a } ^ { d } ( t ) \| < \Delta$ , and $0 < \epsilon \ll 1$ . General control actions in control operations admit this splitting, in which ${ \pmb a } ^ { s } ( t )$ is relatively slowly varying but can be large compared to a small, uniformly bounded deviation control action $\pmb { a } ^ { d } ( t )$ . We refer to Haller and Kaundinya [16] for details of the aSSM-reduced model derived from invariance equations and to Kaundinya et al. [21] for details in the control setting.

On hardware, the full phase space x is only partially observable: cameras record only a sparse set of positional data $\pmb { \mathscr { s } } \in \mathcal { S } \subseteq \mathbb { R } ^ { n _ { s } }$ with $n _ { s } \ll n _ { x }$ . In that case, the critical manifold definition carries over to the observable state space as $\bar { s } ( \bar { a } )$ . The state s has no velocity information and hence does not guarantee embedding the dynamics of the d-dimensional fibers of the frozen aSSM geometry $\mathcal { A } _ { \mathrm { 0 } }$ . To successfully embed the aSSM and its dynamics in an ambient space, by Takens’ theorem (see Takens [39]), we need to delay embed the positional observable $s ,$ at least $\begin{array} { r } { p \geq \left\lfloor \frac { 2 d + 1 } { n _ { y } } \right\rfloor } \end{array}$ times.

In this delay-embedded observable space, assuming the control action satisfies eq. (10), we denote $\mathbf { V } ( \pmb { a } ( t ) ) \in \mathbb { R } ^ { ( p + 1 ) n _ { s } \times \hat { d } }$ whose columns span the time-varying slow spectral subspace $\mathbf { E } ( \pmb { a } ( t ) )$ along the trajectory $\pmb { \mu } ( \pmb { \bar { s } } ( \pmb { a } ( t ) ) )$ , and $\mathbf { r } \in \mathbb { R } ^ { d }$ the reduced coordinate along this subbundle, the leading-order aSSM-reduced model takes the form

$$
\begin{array} { r l } & { \mathbf { r } = \mathbf { V } ^ { \mathrm { T } } ( \pmb { a } ( t ) ) \pmb { \mu } \big ( \pmb { s } - \pmb { \bar { s } } ( \pmb { a } ( t ) ) \big ) , } \\ & { \dot { \mathbf { r } } = \mathbf { R } \big ( \mathbf { r } , \bar { \pmb { s } } ( \pmb { a } ( t ) ) \big ) , } \\ & { \pmb { s } = \mathbf { W } \big ( \mathbf { r } , \bar { \pmb { s } } ( \pmb { a } ( t ) ) \big ) + \bar { \pmb { s } } ( \pmb { a } ( t ) ) , } \end{array}\tag{11}
$$

where $\mathbf { W } ( \cdot )$ is the aSSM parametrization and $\mathbf { R } ( \cdot )$ is the aSSM reduced dynamics. Remark 1. [Discrete-time aSSM-reduced model for reinforcement learning (RL)] Most model-based RL algorithms require a discrete dynamics model. We obtain the discrete reduced model by integrating the continuous leading-order dynamics (11) over a time step $\Delta t$ under a zero-order hold, $\mathrm { i . e . }$ , the action is held fixed at $\mathbf { \pmb { a } } _ { k }$ over $[ t _ { k } , t _ { k } + \Delta t )$ . This results in a discrete map of the aSSM-reduced model,

$$
\begin{array} { r l } & { \mathbf { r } _ { k } = \mathbf { V } ^ { \mathrm { T } } ( \pmb { a } _ { k } ) \pmb { \mu } \big ( \pmb { s } _ { k } - \pmb { \bar { s } } ( \pmb { a } _ { k } ) \big ) , } \\ & { \mathbf { r } _ { k + 1 } = \mathbf { R } _ { d } \big ( \mathbf { R } ; \mathbf { r } _ { k } , \pmb { \bar { s } } ( \pmb { a } _ { k } ) , \Delta t \big ) , } \\ & { \pmb { s } _ { k + 1 } = \mathbf { W } \big ( \mathbf { r } _ { k + 1 } , \pmb { \bar { s } } ( \pmb { a } _ { k } ) \big ) + \bar { \pmb { s } } ( \pmb { a } _ { k } ) , } \end{array}\tag{12}
$$

where ${ \mathbf { R } } _ { d } ( \cdot )$ is the discrete reduced dynamics map approximated via numerical methods from the continuous aSSM reduced dynamics $\mathbf { R } ( \cdot )$

Remark 2. [Higher-order correction to the aSSM-reduced model] Kaundinya et al. [22] provides higher-order corrections to the aSSM-reduced dynamics, which depend on the control action rate. Based on the assumptions of the control action splitting in eq. (10), the higher-order corrected aSSM reduced model is of the form

$$
\dot { \mathbf { r } } = \mathbf { R } \big ( \mathbf { r } , \bar { s } ( { \pmb a } ( t ) ) \big ) + \mathbf { B } ( { \pmb a } ( t ) ) \mathbf { D _ { a } } \bar { s } ( { \pmb a } ( t ) ) \dot { \pmb a } + O \big ( ( \epsilon + \Delta ) ^ { 2 } , \epsilon | \mathbf { r } | , \Delta | \mathbf { r } | \big ) ,\tag{13}
$$

with $\mathbf { B } ( \pmb { a } ( t ) )$ a calibration matrix learned from the leading-order aSSM-reduced model. We ignore this correction here, since for our pneumatically actuated arm, the inconsistencies between the commanded and measured pressures in the actuation system make it impossible to learn an accurate calibration matrix.

Remark 3. [Time-scale separation for target tracking] Our aSSM-reduced models are valid only under the time-scale separation assumption stated in Section 1.1. We quantify this with the non-dimensional slowness measure $r _ { s }$ for practical application,

$$
r _ { s } = \frac { \displaystyle \frac { 1 } { t _ { f } - t _ { i } } \int _ { t _ { i } } ^ { t _ { f } } \left\| \dot { \mathbf { r } } ( \tau ) \right\| d \tau } { \displaystyle \frac { 1 } { t _ { d } } \int _ { 0 } ^ { t _ { d } } \left\| \dot { s } _ { \mathrm { d e c a y } } ( \tau ) \right\| d \tau } ,\tag{14}
$$

where $\scriptstyle { \pmb { s } } _ { \mathrm { d e c a y } }$ is a decaying trajectory of the robot, $t _ { d }$ is the duration of that decay, and the overbar denotes an average across the various decays obtained by passing step

control actions spanning $\mathcal { C } .$ The numerator is the mean speed along the target $\mathbf { { \cal T } } ( t )$ over its horizon $[ t _ { i } , t _ { f } ]$ . The theory is guaranteed for $r _ { s } \ll 1$ , but our method performs well in practice up to $r _ { s } \approx 1$ (see Kaundinya et al. [21] and Kaundinya et al. [22]).

## Data-driven methodology

We collect the robot’s response to step control actions in the observable space. The step control actions are generated by performing jumps, each held for $t _ { \mathrm { d e c a y } }$ seconds, between the m most distant control actions for every element of the static action set, which we sample by Latin hypercube sampling (see McKay et al. [28]). We set the total number of jumps, $m = 4$ , for the step data collection for the SoPrA arm. Passing these control actions gives static pairs $\{ \boldsymbol { s } _ { \mathrm { s t a t i c } } ^ { i } , \boldsymbol { a } _ { \mathrm { s t a t i c } } ^ { i } \} _ { i = 1 } ^ { N }$ and their accompanying decay trajectories $\{ \pmb { s } _ { \mathrm { d e c a y } } ^ { i } , \pmb { a } _ { \mathrm { d e c a y } } ^ { i } \} _ { i = 1 } ^ { m N }$

• Step 1. [Learning the critical manifold]. We learn $\bar { s } ( a )$ as a nonlinear static map by regressing the static configurations onto their control actions,

$$
\bar { s } ^ { \star } = \arg \operatorname* { m i n } _ { \bar { s } } \sum _ { i = 1 } ^ { N } \big \| s _ { \mathrm { s t a t i c } } ^ { i } - \bar { s } ( \mathbf { { a } } _ { \mathrm { s t a t i c } } ^ { i } ) \big \| ^ { 2 } .\tag{15}
$$

We seek the critical manifold as a polynomial expansion of order $k _ { a }$

$$
\bar { s } ( { \pmb a } ) = \sum _ { 0 \le | { \pmb \alpha } | \le k _ { a } } { \bf c } _ { \alpha } { \pmb a } ^ { \alpha } ,\tag{16}
$$

where $_ { \pmb { \alpha } }$ is a multi-index, and $\mathbf { c } _ { \alpha }$ are the polynomial coeficients. The expansion is linear in $\mathbf { c } _ { \alpha }$ , so the regression reduces to a linear least-squares problem with a closed-form solution. We can also use radial-basis and neural-network approximations for fitting $\bar { s } ( a )$ (see Kaundinya et al. [22] for details).

• Step 2. [Learning the aSSM parametrization and reduced dynamics]. We center each decay about its static configuration to obtain $\{ { s _ { \mathrm { c e n t e r e d } } ^ { i } } , \bar { { \pmb a } } _ { \mathrm { d e c a y } } ^ { i } \} _ { i = 1 } ^ { m N }$ . We later delay embed this data p times through the map $\pmb { \mu } .$ The delays are chosen such that the condition $( p + 1 ) n _ { s } \geq 2 d + 1$ is satisfied. We collect the m decays that share a static configuration, and perform an economy singular value decomposition to obtain chart maps $\{ \mathbf { V } ^ { i } \} _ { i = 1 } ^ { N }$ , which we interpolate over $\pmb { a } _ { \mathrm { s t a t i c } }$ or, when the local charts nearly coincide, use a single global chart. The data in the reduced coordinates is then obtained by $\pmb { \mathcal { R } } ^ { i } = ( \mathbf { V } ^ { i } ) ^ { \mathrm { T } } \big [ \pmb { \mu } ( s _ { \mathrm { c e n t e r e d } } ^ { i _ { 1 } } ) , \dots , \pmb { \mu } ( s _ { \mathrm { c e n t e r e d } } ^ { i _ { m } } ) \big ]$ over the m decays sharing the $i ^ { \mathrm { { t h } } }$ static configuration, and finite diferencing this results in the velocity data $\dot { \mathcal { R } } ^ { i }$

We seek the leading-order aSSM parametrization $\mathbf { W } ( \cdot )$ and reduced dynamics $\mathbf { R } ( \cdot )$ as a polynomial expansion of the form,

$$
\mathbf { W } ( \mathbf { r } , s ) = \sum _ { \substack { 1 \leq | \alpha | \leq k _ { r } ^ { W } 0 \leq | \beta | \leq k _ { a } ^ { W } } } \mathbf { W } _ { \alpha \beta } \mathbf { r } ^ { \alpha } s ^ { \beta } ,\tag{17}
$$

$$
\mathbf { R } ( \mathbf { r } , s ) = \sum _ { 1 \leq | \alpha | \leq k _ { r } ^ { R } 0 \leq | \beta | \leq k _ { a } ^ { R } } \mathbf { R } _ { \alpha \beta } \mathbf { r } ^ { \alpha } s ^ { \beta } ,\tag{18}
$$

where $_ { \pmb { \alpha } }$ and $\beta$ are multi-indices over r and s respectively, $k _ { r } ^ { W } , k _ { a } ^ { W }$ and $k _ { r } ^ { R } , k _ { a } ^ { R }$ denote the corresponding polynomial orders, and ${ \bf W } _ { \alpha \beta } , \ { \bf R } _ { \alpha \beta }$ are the unknown coeficients. We fit these by minimizing the reconstruction and dynamics regression losses on the centered decay data,

$$
\mathbf { W } ^ { \star } = \operatorname * { a r g m i n } _ { \mathbf { W } } \sum _ { i } \big \| { S } _ { \mathrm { c e n t e r e d } } ^ { i } - \mathbf { W } \big ( \mathbf { \mathcal { R } } ^ { i } , { s } _ { \mathrm { s t a t i c } } ^ { i } \big ) \big ) \big \| ^ { 2 } ,\tag{19}
$$

$$
\mathbf { R } ^ { \star } = \underset { \mathbf { R } } { \arg \operatorname* { m i n } } ~ \sum _ { i } \big \| \dot { \pmb { \mathcal { R } } } ^ { i } - \mathbf { R } \big ( \mathcal { R } ^ { i } , s _ { \mathrm { s t a t i c } } ^ { i } \big ) \big ) \big \| ^ { 2 } ,
$$

where $\begin{array} { r } { \pmb { S } _ { \mathrm { c e n t e r e d } } ^ { i } = \left[ \pmb { s } _ { \mathrm { c e n t e r e d } } ^ { i _ { 1 } } , \ldots , \pmb { s } _ { \mathrm { c e n t e r e d } } ^ { i _ { m } } \right] } \end{array}$ . All parameters concerning the aSSMreduced model are chosen to minimize the normalized mean trajectory error on unseen test decay data.

## 3.2 Ofline model-based reinforcement learning

We learn a policy entirely from the learned dynamics model. The dynamics model is identified ofline from the dataset and is held fixed thereafter. No interaction with the hardware takes place during policy optimization. The policy $\pi _ { \theta }$ acts on the observation

$$
\mathbf o _ { k } = \left\{ s _ { k } , s _ { k - 1 } , \pmb { a } _ { k - 1 } , \pmb { \Gamma } _ { k + 1 } \right\} ,\tag{20}
$$

where $\scriptstyle { \pmb { s } } _ { k }$ is the state obtained from passing control action $\mathbf { \delta } _ { a _ { k - 1 } }$ to the dynamics model and $\Gamma _ { k + 1 }$ the target state to be reached. The control action needed to reach the target state $\Gamma _ { k + 1 }$ is sampled as,

$$
\mathbf { \sigma } _ { \mathbf { a } _ { k } } \sim \pi _ { \theta } ( \cdot \mid \mathbf { o } _ { k } ) .\tag{21}
$$

We note, that for obtaining $\scriptstyle o _ { k }$ requires $\scriptstyle { \pmb { s } } _ { k }$ that needs to evaluated using the aSSM model in eq. (12) or the autoregressive model $\mathrm { e q } .$ . (5). The aSSM model requires the control action at $k - 1$ , whereas the autoregressive model requires past control actions based on the model’s memory window. The aSSM model needs past states only to set an initial condition, but the dynamical rollouts are history-independent. The autoregressive model also needs a history of states to perform the rollout.

Algorithm 1 RL policy optimization by imagination on the learned dynamics model   
Require: Dynamics model either an aSSM model eq. (12) or a neural-network-based   
model eq. (5) identified ofline from and held fixed   
Require: Dataset , target distribution , horizon $T ,$ number of parallel imagined   
rollouts N   
1: Initialize policy $\pi _ { \theta }$   
2: for learning iterations = 1, 2, . . . do   
3: Imagination: sample N initial conditions and target tracks, then roll out in   
parallel   
4: for $i = 1 , \ldots , N$ in parallel do   
5: Sample warm-start history and initial actions from , giving the initial   
condition of rollout i   
6: Sample a target track $\{ \mathbf { T } _ { k } ^ { ( i ) } \} _ { k = 1 } ^ { T }$ from $\tau$   
7: end for   
8: for $k = 0 , 1 , \ldots , T - 1$ do   
9: for $i = 1 , \ldots , N$ in parallel do   
10: Assemble $\mathbf { o } _ { k } ^ { ( i ) }$ by Eq. (20)   
11: Sample $\mathbf { \Delta } \mathbf { a } _ { k } ^ { ( i ) } \sim \pi _ { \theta } ( \cdot | \mathbf { o } _ { k } ^ { ( i ) } )$   
12: Advance one step to $\pmb { s } _ { k + 1 } ^ { ( i ) }$ using the dynamics model   
13: Evaluate reward $\rho _ { k } ^ { ( i ) }$ by Eq. (22)   
14: end for   
15: end for   
16: Update $\pi _ { \theta }$ on the batch of N imagined rollouts using PPO   
17: end for

The target horizon length is T steps. Thus, the model must perform T rollouts to produce a prediction based on sampled control-action sequences. At each step of the rollout, we evaluate the reward, defined by

$$
\rho _ { k } = \rho { \left( { { \mathscr { s } } _ { k + 1 } } , { \pmb { a } } _ { k } , { \pmb { \Gamma } } _ { k + 1 } \right) } .\tag{22}
$$

We provide Algorithm 1 for training a policy that maximizes the above reward along the target track across diferent initial conditions.

## Acknowledgements

This work has been partially supported by grant 200021 214908 from the Swiss National Science Foundation (SNF).

## Competing Interests

All authors declare no financial or non-financial competing interests.

## References

[1] J. I. Alora, M. Cenedese, G. Haller, and M. Pavone. Discovering dominant dynamics for nonlinear continuum robot control. npj Robotics, 3(1):5, 2025. doi: 10.1038/s44182-025-00021-8. URL https://doi.org/10.1038/s44182-025-00021-8.

[2] J. Ax˚as, M. Cenedese, and G. Haller. Fast data-driven model reduction for nonlinear dynamical systems. Nonlinear Dyn., 111(9):7941–7957, 2023.

[3] U. Berdica, M. Jackson, N. E. Veronese, J. Foerster, and P. Maiolino. Reinforcement learning controllers for soft robots using learned environments. In Proc. IEEE Int. Conf. Soft Robot. (RoboSoft), pages 933–939, 2024. doi: 10.1109/RoboSoft60065.2024.10522003.

[4] L. Bettini, B. Kasz´as, B. Zybach, J. Dual, and G. Haller. Data-driven nonlinear model reduction to spectral submanifolds via oblique projection. Chaos, 35(4): 043135, Apr. 2025. doi: 10.1063/5.0243849. URL https://doi.org/10.1063/5.02 43849.

[5] L. Bettini, A. Kazemipour, R. K. Katzschmann, and G. Haller. General oblique projections for model reduction via spectral submanifolds. Chaos, 36(5):053122, 2026.

[6] L. Bettini, A. Kazemipour, R. K. Katzschmann, and G. Haller. Nonlinear spectral modeling and control of soft-robotic muscles from data, 2026. URL https://arxi v.org/abs/2601.03247. arXiv:2601.03247.

[7] D. Braganza, D. M. Dawson, I. D. Walker, and N. Nath. A neural network controller for continuum robots. IEEE Trans. Robot., 23(6):1270–1277, 2007. doi: 10.1109/TRO.2007.906248.

[8] G. Buza and G. Haller. Spectral submanifold reduction for PDEs describing nonlinear continuum vibrations, 2026. URL https://arxiv.org/abs/2607.10675. arXiv:2607.10675.

[9] M. Cenedese, J. Ax˚as, and G. Haller. SSMLearn. https://github.com/haller-gro up/SSMLearn, 2021. GitHub repository.

[10] M. Cenedese, J. Ax˚as, B. B¨auerlein, K. Avila, and G. Haller. Data-driven modeling and prediction of non-linearizable dynamics via spectral submanifolds. Nat. Commun., 13(1):872, 2022.

[11] M. Cenedese, J. Ax˚as, H. Yang, M. Eriten, and G. Haller. Data-driven nonlinear model reduction to spectral submanifolds in mechanical systems. Philos. Trans. R. Soc. A, 380:20210194, 2022.

[12] A. Centurelli, L. Arleo, A. Rizzo, S. Tolu, C. Laschi, and E. Falotico. Closed-loop dynamic control of a soft manipulator using deep reinforcement learning. IEEE Robot. Autom. Lett., 7(2):4741–4748, 2022. doi: 10.1109/LRA.2022.3146903.

[13] F. Faure, C. Duriez, H. Delingette, J. Allard, B. Gilles, S. Marchesseau, H. Talbot, H. Courtecuisse, G. Bousquet, I. Peterlik, et al. Sofa: A multi-model framework for interactive physical simulation. Soft tissue biomechanical modeling for computer assisted surgery, pages 283–321, 2012.

[14] N. Fenichel. Geometric singular perturbation theory for ordinary diferential equations. J. Difer. Equ., 31(1):53–98, 1979.

[15] G. Haller. Modeling Nonlinear Dynamics from Equations and Data — with Applications to Solids, Fluids, and Controls. SIAM, Philadelphia, PA, 2025.

[16] G. Haller and R. S. Kaundinya. Nonlinear model reduction to temporally aperiodic spectral submanifolds. Chaos, 34(4):043152, Apr. 2024. ISSN 1054-1500. doi: 10.1063/5.0187080. URL https://doi.org/10.1063/5.0187080.

[17] G. Haller and S. Ponsioen. Nonlinear normal modes and spectral submanifolds: existence, uniqueness and use in model reduction. Nonlinear Dyn., 86(3):1493– 1534, 2016.

[18] G. Haller, B. Kasz´as, A. Liu, and J. Ax˚as. Nonlinear model reduction to fractional and mixed-mode spectral submanifolds. Chaos, 33(6):063138, 2023.

[19] J. Ibarz, J. Tan, C. Finn, M. Kalakrishnan, P. Pastor, and S. Levine. How to train your robot with deep reinforcement learning: lessons we have learned. Int. J. Robot. Res., 40(4-5):698–721, 2021. doi: 10.1177/0278364920987859. URL https://doi.org/10.1177/0278364920987859.

[20] M. Janner, J. Fu, M. Zhang, and S. Levine. When to trust your model: modelbased policy optimization. In Adv. Neural Inf. Process. Syst. (NeurIPS), Red Hook, NY, 2019. Curran Associates, Inc.

[21] R. S. Kaundinya, J. I. Alora, J. G. Matt, L. A. Pabon, M. Pavone, and G. Haller. Data-driven soft robot control via adiabatic spectral submanifolds. Int. J. Robot. Res., 2026. doi: 10.1177/02783649261461630.

[22] R. S. Kaundinya, M. Leone, R. Sabella, H. Buurmeijer, L. Pabon, J. I. Alora, M. Pavone, and G. Haller. Data-Driven Precision Control of Soft Helicoid Robots. Preprint, 2026.

[23] R. S. Kaundinya, I. Thiel, B. Kasz´as, S. Jain, and G. Haller. Predicting generalized steady states in aperiodically forced mechanical systems. J. Sound Vib., 643:120026, 2026. ISSN 0022-460X. doi: https://doi.org/10.1016/j.jsv.2026 .120026. URL https://www.sciencedirect.com/science/article/pii/S0022460X260 03871.

[24] J. Kober, J. A. Bagnell, and J. Peters. Reinforcement learning in robotics: A survey. Int. J. Robot. Res., 32(11):1238–1274, 2013. doi: 10.1177/027836491349 5721. URL https://doi.org/10.1177/0278364913495721.

[25] O. Kroemer, S. Niekum, and G. Konidaris. A review of robot learning for manipulation: Challenges, representations, and algorithms. J. Mach. Learn. Res., 22(30):1–82, 2021. URL http://jmlr.org/papers/v22/19-804.html.

[26] C. Li, A. Krause, and M. Hutter. Robotic world model: A neural network simulator for robust policy optimization in robotics. In Adv. Neural Inf. Process. Syst. (NeurIPS), 2025. URL https://arxiv.org/abs/2501.10100.

[27] V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa, and G. State. Isaac gym: High performance GPU based physics simulation for robot learning. In Adv. Neural Inf. Process. Syst. (NeurIPS), Datasets and Benchmarks Track, 2021. URL https://openreview.net/forum?id=fgFBtYgJQX .

[28] M. D. McKay, R. J. Beckman, and W. J. Conover. A comparison of three methods for selecting values of input variables in the analysis of output from a computer code. Technometrics, 21(2):239–245, 1979. ISSN 00401706.

[29] M. Mekkattu, M. Y. Michelis, and R. K. Katzschmann. Sors: A modular, highfidelity simulator for soft robots. In Proc. IEEE Int. Conf. Soft Robot. (RoboSoft), pages 223–230, 2026. doi: 10.1109/RoboSoft67810.2026.11522869.

[30] T. Miki, J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, and M. Hutter. Learning robust perceptive locomotion for quadrupedal robots in the wild. Sci. Robot., 7 (62):eabk2822, 2022. doi: 10.1126/scirobotics.abk2822. URL https://www.scienc e.org/doi/abs/10.1126/scirobotics.abk2822.

[31] A. A. Morsy, Z. Xu, P. Tiso, and G. Haller. Data-driven reduction of the finiteelement model of a Tribomechadynamics benchmark problem. Nonlinear Dyn., 113(18):24539–24555, 2025. doi: 10.1007/s11071-025-11378-8.

[32] F. Nan, H. Ma, Q. Guan, J. Hughes, M. Muehlebach, and M. Hutter. Eficient model-based reinforcement learning for robot control via online optimization. Int. J. Robot. Res., 0(0):02783649261457560, 2026. doi: 10.1177/02783649261457560. URL https://doi.org/10.1177/02783649261457560.

[33] N. Naughton, J. Sun, A. Tekinalp, T. Parthasarathy, G. Chowdhary, and M. Gazzola. Elastica: A compliant mechanics environment for soft robotic control. IEEE Robot. Autom. Lett., 6(2):3389–3396, 2021. doi: 10.1109/LRA.2021.30636 98.

[34] M. S. Nazeer, C. Laschi, and E. Falotico. Rl-based adaptive controller for high precision reaching in a soft robot arm. IEEE Trans. Robot., 40:2498–2512, 2024. doi: 10.1109/TRO.2024.3381558.

[35] T. Parthasarathy, S. H. Kim, S. Cui, and M. Gazzola. Elastica++: A highperformance, multiphysics framework for large interacting assemblies of cosserat rods, 2026. URL https://arxiv.org/abs/2605.13766. arXiv:2605.13766.

[36] S. Satheeshbabu, N. K. Uppalapati, T. Fu, and G. Krishnan. Continuous control of a soft continuum arm using deep reinforcement learning. In Proc. IEEE Int. Conf. Soft Robot. (RoboSoft), pages 497–503, 2020. doi: 10.1109/RoboSoft48309. 2020.9116003.

[37] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms, 2017. URL https://arxiv.org/abs/1707.06347. arXiv:1707.06347.

[38] R. S. Sutton and A. G. Barto. Reinforcement learning: An introduction. MIT Press, Cambridge, MA, 2 edition, 2018.

[39] F. Takens. Detecting strange attractors in turbulence. In D. Rand and L. Young, editors, Dynamical Systems and Turbulence, Warwick 1980, pages 366–381. Springer, Berlin, Heidelberg, 1981.

[40] T. G. Thuruthel, E. Falotico, F. Renda, and C. Laschi. Model-based reinforcement learning for closed-loop dynamic control of soft robotic manipulators. IEEE Trans. Robot., 35:124–134, 2019. URL https://api.semanticscholar.org/Corpus ID:59618928.

[41] Y. Toshimitsu, K. W. Wong, T. Buchner, and R. Katzschmann. Sopra: Fabrication & dynamical modeling of a scalable soft continuum robotic arm with integrated proprioceptive sensing. In Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), pages 653–660, 2021. doi: 10.1109/IROS51168.2021.9636539.

[42] Y. Toshimitsu, B. Forrai, B. G. Cangan, U. Steger, M. Knecht, S. Weirich, and R. K. Katzschmann. Getting the ball rolling: Learning a dexterous policy for a biomimetic tendon-driven hand with rolling contact joints. In Proc. IEEE-RAS Int. Conf. Humanoid Robots (Humanoids), pages 1–7, 2023. doi: 10.1109/Huma noids57100.2023.10375231.

[43] X. Zhang, C. Li, H. Mo, Y. Jiang, W. Cao, X. Xu, W. Jiang, Z. Bing, Y. Yang, X. Li, Y. Yang, H. Lu, L.-l. Zeng, A. Knoll, D. Hu, L. Wen, and W. Pan. Reinforcement learning in linear embedding space unlocks generalizable control across soft robot configurations. Nat. Commun., 17(1):7327, 2026. doi: 10.1038/s41467-026-72491-9. URL https://doi.org/10.1038/s41467-026-72491-9.

[44] H. Zheng, B. Sukhija, C. Li, K. Iten, A. Krause, and R. K. Katzschmann. Learning soft robotic dynamics with active exploration, 2025. URL https://arxiv.org/ab s/2510.27428. arXiv:2510.27428.

![](images/abfc1dc922a4d11295c9d1747973406f118ef109e2caf321016b4f9942b61374.jpg)  
(a)

![](images/d11599129f2ef538866d0b3735d44d131310c425745d42db096cb39b0e28936a.jpg)  
(b)  
Extended Data Fig. 1 — Mismatch between commanded and measured pressure in SoPrA. a, Time series of the control action $\mathbf { { a } } _ { 2 }$ for the periodic case, with the pressure commanded by the user (black) and the pressure measured in the chamber (orange). b, The mismatch in a makes the measured response of the SoPrA arm (black) fluctuate about the periodic response that the aSSM model predicts (green).

![](images/2c2f5b7616c6b8effffeb273088a589537b6272f54688bd40dcd3d1ba3245eea.jpg)  
Extended Data Fig. 2 — [Convergence of model-based RL policies trained ofline for closed-loop trajectory tracking]. Plots for the evolution of the mean imagination episode return for aSSM-RL (green), RNN-RL (yellow), and Transformer-RL (red) as the policies are trained. The solid lines for each policy depict the average mean imagination episode return across hyperparameter sweeps, and the shaded region shows the deviation.

![](images/1101b4b3cd5a25d85167af6ec4eae11b0dc0fdb9ddc72a802383a4ed856e4cf1.jpg)  
(a)

Average target error comparison in the absence of commanded-to-measured pressure mismatch  
![](images/791965baa9db4a4f18f4c801f8ea569cc766eb7d910603530ef45063344d7ed3.jpg)  
(b)  
Extended Data Fig. 3 — [Closed-loop results with potential hardware upgrades]. (a) Closed-loop results for four target tracks for aSSM-MPC (blue), aSSM-RL (green), RNN-RL (yellow), and Transformer-RL (red), with numerical filtering applied to the results to demonstrate the improvement expected when we use upgraded pressure valves. (b) Revised average tracking error compared across all the target tracks when commanded-to-measure pressure mismatch is accounted for numerically. The percentage number on top of each bar shows the extent of error removed by the numerical filter.