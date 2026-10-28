# Autonomous fixed-wing interception using reinforcement learning

Daan van Hamersveld, Ewoud Smeur, Iurii Kriachko

Abstract—Medium altitude, long endurance unmanned aerial systems such as Shahed-type strike drones are an unanswered threat in modern conflicts. Conventional air defense systems are too expensive to provide a feasible large-scale solution. Quadcopter interceptors are a cheap, battle-proven alternative, but their extremely limited flight time restricts them to low altitude targets at short range, whereas fixed-wing platforms have the endurance to reach higher altitude targets from longer distances. However, such interceptors are still operated manually. Recent developments in learning-based controllers show potential for automation. In this work, a fixed-wing interceptor is presented that is controlled by a neural network that maps filtered monocular vision observations directly to actuator commands. The policy was trained in a custom Python environment using reinforcement learning and proximal policy optimization. The policy was then deployed on a fixed-wing platform and tested in real interception scenarios. As a baseline, two classical guidance controllers were also developed and tested in simulation. The policy is able to intercept targets both in simulation and in real flight. In simulation, the policy outperforms the classical controllers under most conditions. The interceptor was ultimately successfully validated in real flight by intercepting maneuvering fixed-wing targets. This novel system shows the potential of artificial intelligence for autonomous control of fixed-wing interceptor drones.

Index Terms—interception, CUAS, reinforcement learning, artificial intelligence, terminal guidance, computer vision, fixedwing

## I. INTRODUCTION

Medium Altitude Long Endurance (MALE) Unmanned Aerial Systems (UAS) pose a credible threat in modern warfare [1][2]. In both the Ukraine-Russia conflict as well as the recent Iran-Israel/US war, the systems used can be classified into two broad groups. Firstly, intelligence, surveillance and reconnaissance drones, which are generally fixed-wing UAS with varying imaging capabilities. They typically fly at 2- 3 km altitude at speeds up to 150 km/h. Secondly, Shahedtype strike drones, which are delta-wing kamikaze drones that typically fly at 1-2 km altitude at speeds reaching above 250 km/h. They carry a large explosive payload and have high endurance, allowing for long-range (up to 2000 km) targeted ground strikes.

To answer these threats, low cost interceptor drones are emerging as an alternative to expensive conventional air defenses such as fighter jets and missiles. Interceptors carry an explosive payload which is detonated when close enough to neutralize the target. Hence, such systems are singleuse, meaning a minimal onboard sensor suite is preferred for affordability. A ground-based radar system can typically provide a mid-course guidance target lock with a few hundred meters error. However, terminal guidance requires meterlevel accuracy and hence is currently mainly vision-based, as monocular cameras are inexpensive and easy to source. Traditional guidance laws such as Proportional Navigation (PN) and Pure Pursuit (PP) [3][4][5] need a constant Line Of Sight (LOS) and struggle against maneuvering targets. Consequently, current interceptors are mostly manually controlled during terminal guidance.

![](images/7feea3fb8aecaf358b18cd5893a8602c80829e9d0ded5a7a4a76f1a924405102.jpg)  
Fig. 1. The interceptor during a high bank angle.

Vision-based relative state estimation can be done in many ways for a variety of applications, including drone racing [6], target tracking [7], aerial refueling [8] and also interception [9][10][11]. For interception purposes, the boundingbox [12][13][11] is an effective and easy to implement proxy to a relative state. Turning visual observation into effective guidance for interception has been done using classical control approaches such as PP [9] and PN [14].

Recently, Reinforcement Learning (RL) has been successfully applied to UAS control tasks such as stabilization [15], attitude control [16], acrobatics [17], drone racing [18][6] and also interception [7][19][20], with controller authority varying from high level planning to direct actuator command. While promising, for interception the real challenge lies in transferring these results from simulation to a real, uncontrolled environment. Recent work suggests that end-to-end learning-based controllers can achieve high sim-to-real transfer robustness by implementing domain randomization [21].

In MALE interception scenarios, fixed-wing drones can be superior to quadrotors due to their better endurance, allowing them to intercept targets from longer ranges and at higher altitudes. Studies on RL controllers for fixed-wing platforms are sparse. However, obstacle avoidance [22], path following [23] evasion [24] and interception with perfect state knowledge [20] have been carried out in simulation. In real environments, studies have demonstrated successful attitude holding [25] and landing [26].

Fixed-wing, vision-based interception using a learningbased controller in a real environment is lacking in the literature. Most interception studies are done in simulation and using a quadrotor model, whilst realistic perception and simto-real transfer are usually not addressed.

In this work, a learning-and monocular vision-based fixedwing interceptor is presented and demonstrated to successfully intercept MALE targets in a real environment. Shown in figure 1, the interceptor is built entirely with off-the-shelf components. An RL controller is trained in simulation which replaces the guidance and control of a conventional autopilot. A neural network directly maps filtered visual and flight controller information to actuator commands. The performance of the RL controller is validated against baseline PN and PP controllers in simulation. Finally, the RL controller is tested on real flight interceptions of fixed-wing, MALE UAS.

The main contributions of this work are therefore:

• Development of an end-to-end neural network enabling autonomous fixed-wing interception of maneuvering targets with limited onboard sensing.

• Development of a monocular vision- and learning-based, autonomous, fixed-wing interceptor able to intercept MALE targets in a real environment.

• Development of a RL framework inspired by [19], incorporating fixed-wing dynamics and a real computer-vision algorithm.

The final controller is tested in real flight interceptions and assessed in terms of performance and constraints.

## II. PRELIMINARIES

## A. Problem formulation

The interceptor and the target are flying in a North-East-Down (NED) 3-dimensional space. It is assumed that to neutralize the target, an explosive device must be detonated within a maximum radius $r _ { n }$ around the target’s position p<sub>T</sub>. Hence to intercept the target, the interceptor’s position p<sub>I</sub> must satisfy:

$$
\left( \mathbf { p } _ { I } - \mathbf { p } _ { T } \right) ^ { \top } \left( \mathbf { p } _ { I } - \mathbf { p } _ { T } \right) \leq r _ { n } ^ { 2 } .\tag{1}
$$

Monocular vision suffers from Field Of View (FOV) limitations, occlusions and pixel noise. This sensor requires the problem to be framed as a partially observable Markov Decision Process (POMDP) [19][27], incorporating uncertainty as a quantitative observation.

## B. Dynamics model

To train the RL policy in simulation, a six degree of freedom, rigid-body dynamics model for a fixed-wing aircraft was built. The model is based on literature [28][29] and is described in appendix A.

## III. METHODOLOGY

## A. System identification

An X-UAV Talon<sup>1</sup> body was used for both the interceptor and the target. The interceptor was fitted with a Pixhawk 6C running Ardupilot. The inertial properties of the aircraft were estimated by modeling the fuselage as a cuboid and the wing as a flat plate with uniform density. Cross-coupling terms $I _ { x y }$ and $I _ { y z }$ were assumed to be zero due to the symmetry of the aircraft. $I _ { x z }$ was assumed to be negligible, even though the asymmetry in the horizontal axis would cause this to be non-zero.

A magnetic angle encoder with a wind-vane was mounted on the side of the fuselage to measure angle of attack (AOA). Three system identification flights were flown to estimate the aerodynamic coefficients of the aircraft using ordinary least squares regression [30]. The regression method is described in appendix B.

## B. Perception

A front-facing IMX462LQR monocular camera was installed in the nose of the interceptor. The videostream was input to a proprietary target detection algorithm running on an onboard Raspberry Pi 5 fitted with a Hailo neural network accelerator.

1) Bounding box: The algorithm outputs a Bounding Box (BB): a rectangular box encapsulating the target drone. The BB outputs the following measurement vector at time step k:

$$
\mathbf { z } _ { k } ^ { \top } = \left[ x _ { c } \quad y _ { c } \quad w \quad h \quad c \right] .\tag{2}
$$

Here, $x _ { c } , y _ { c }$ are the BB center coordinates and $w , h$ are its width and height, all in pixels. c is a confidence score ∈ [0, 1] and represents how confident the algorithm is that the bounding box actually encapsulates a drone.

2) Pinhole model: A simulation was created that maps a 3D point in the camera frame to the 2D image plane using the pinhole model adopted from [31]. If the dimensions of the object of interest are known, the distance along the optical axis of the camera can be formulated as:

$$
Z = \frac { f _ { x } W } { w _ { p x } } ,\tag{3}
$$

where W is the width of the object in meters. $w _ { p x }$ is the width of the object in the image in pixels. Equation 3 assumes that W is constant when viewed from all angles, which holds only for spherical objects. For a fixed-wing target, the perceived W will depend on the orientation from which it is viewed. To accommodate for this, the simulation computes the effective width as $W _ { \mathrm { e f f } } = W \cdot \operatorname* { m a x } \bigl ( | \cos \psi | , 0 . 6 \bigr )$ , where ψ is the angle between the target’s velocity vector and the LOS. $f _ { x }$ is the focal length of the camera:

$$
f _ { x } = f _ { y } = { \frac { W _ { \mathrm { i m g } } } { 2 \tan ( \mathrm { H F O V / 2 } ) } } = { \frac { 1 2 8 0 \ \mathrm { p x } } { 2 \tan ( 6 0 ^ { \circ } / 2 ) } } \approx 1 1 0 8 \ \mathrm { p x }\tag{4}
$$

Once Z is known, X and Y, the horizontal and vertical distance from the center of the object to the optical axis, can be formulated as follows:

$$
X = \frac { ( u _ { c } - c _ { x } ) } { f _ { x } } Z , Y = \frac { ( v _ { c } - c _ { y } ) } { f _ { y } } Z .\tag{5}
$$

Here, $u _ { c } , v _ { c }$ are the pixel center coordinates of the object. $c _ { x } , c _ { y }$ are the image center coordinates in pixels. To simulate $u _ { c } , v _ { c } .$ , the LOS vector $\textbf { r } \in \ \mathbb { R } ^ { 3 }$ must be decomposed into $X , Y , Z$ relative to the optical axis of the camera.

3) Camera noise and detection probability: A monocular camera is inherently noisy and computer vision algorithms are prone to non-detections even with a drone in the FOV. To reflect this in simulation, several error sources were introduced.

First, zero mean pixel noise was added to the BB center and its dimensions.

$$
\sigma _ { c } = 3 . 0 + 0 . 0 5 d ~ \mathrm { p x } , \qquad \sigma _ { s } = 2 . 0 + 0 . 0 2 d ~ \mathrm { p x } ,\tag{6}
$$

with d the range in meters, standard deviation $\sigma _ { c }$ applied to $x _ { c }$ and $y _ { c } ,$ , and standard deviation $\sigma _ { s }$ to w and h.

Second, computer vision algorithms are prone to anchor the BB consistently wrong for a stretch of frames. For instance, the BB can briefly only encapsulate the fuselage of the target, missing the dimensions of the wing. Hence, a BB bias $\mathbf { b } _ { k }$ was introduced:

$$
\mathbf { b } _ { k } = \rho \mathbf { b } _ { k - 1 } + { \sqrt { 1 - \rho ^ { 2 } } } \sigma _ { b } \epsilon _ { k } , \qquad \epsilon _ { k } \sim { \mathcal { N } } ( \mathbf { 0 } , \mathbf { I } _ { 4 } ) ,\tag{7}
$$

with $\mathbf { b } _ { k } = { [ \begin{array} { l } { b _ { x } , } \end{array} ] } \ b _ { y } , \ b _ { w } , \ b _ { h } { ] } ^ { \top }$ and $\rho = e ^ { - \Delta t _ { \mathrm { c t r l } } / \tau _ { b } }$ , where $\tau _ { b } = 1$ s is the time over which the bias persists and $\sigma _ { b }$ its standard deviation. $\epsilon _ { k }$ is a vector of four independent zero mean normal samples with standard deviation of one, redrawn at each control step. The bias enters the measurement as:

$$
\begin{array} { r l } { \tilde { x } _ { c } = x _ { c } + b _ { x } w , } & { { } \qquad \tilde { y } _ { c } = y _ { c } + b _ { y } h , } \\ { { } } & { { } } \\ { \tilde { w } = \left( 1 + b _ { w } \right) w , } & { { } \qquad \tilde { h } = \left( 1 + b _ { h } \right) h , } \end{array}\tag{8}
$$

so the center is displaced by a fraction of the apparent box size and the dimensions of the box are scaled.

Third, the per-frame detection is probabilistic and degrades with range:

$$
P _ { \mathrm { d e t e c t } } = p _ { \mathrm { m i n } } + \left( 1 - p _ { \mathrm { m i n } } \right) e ^ { - d / d _ { s } } ,\tag{9}
$$

with $d _ { s } = 1 5 0$ m and $p _ { \mathrm { m i n } } = 0 . 0 5$ . Detections can only occur if the target is within the FOV.

Non-detections tend to be clustered. Hence, the detector is simulated as a two-state Markov chain, moving between detecting and blind with per step transition probabilities:

$$
P ( \mathrm { b l i n d } _ { k + 1 } \mid \mathrm { d e t e c t } _ { k } ) = q , P ( \mathrm { d e t e c t } _ { k + 1 } \mid \mathrm { b l i n d } _ { k } ) = r ,\tag{10}
$$

$$
r = \frac { \Delta t _ { \mathrm { c t r l } } } { T _ { \mathrm { b l i n d } } } , \qquad q = r \frac { 1 - P _ { \mathrm { d e t e c t } } } { P _ { \mathrm { d e t e c t } } } .\tag{11}
$$

While blind, the detector recovers with probability r at each control step of duration $\Delta t _ { \mathrm { c t r l } } .$ , giving outages with a mean duration $T _ { \mathrm { b l i n d } }$ . Outages therefore keep the same mean duration at all ranges while becoming more frequent as $P _ { \mathrm { d e t e c t } }$ falls.

Finally, a computer vision algorithm does not instantly produce detections, but instead lags the camera image. This detection latency was also simulated such that the current detection lags the current camera frame. The delayed detection is interpreted with the current attitude of the interceptor, so any rotation appears as a bearing error.

BB bias, detection dropout and latency were varied through a curriculum as described in section III-G, with specific values described in appendix G.

## C. Kalman filtering

The bounding box is noisy and does not provide a continuous estimate of the relative target state. Hence, a Constant Acceleration Kalman Filter (CAKF) was applied to provide robust relative target state estimation. The same filter, with identical parameters, is used in simulation, training, and real flight. The filter is described in appendix C.

## D. RL formulation

In this study, the RL controller is a single neural network that maps the observation vector to an actuation vector $u _ { k } ~ \in ~ \mathbb { R } ^ { 4 }$ that outputs normalized throttle $u _ { t } ~ \in ~ [ 0 , 1 ]$ and flight control surface $u _ { a } , u _ { e } , u _ { r } ~ \in ~ [ - 1 , 1 ]$ commands. The controller maps a monocular vision-based, filtered relative target state estimate, the filter’s confidence in that estimate, and the interceptor’s own state directly to actuator-level outputs. Hence, the low-level FC autopilot is bypassed completely.

1) Observable state: The per-frame observation vector $\mathbf { O } \in$ $\mathbb { R } ^ { 3 8 }$ is built from the KF state estimate and the interceptor’s own state:

$$
\mathbf O ^ { \top } = \left[ \begin{array} { c c c c c c c } { \hat { \mathbf r } _ { b } } & { d } & { f } & { \mathbf r _ { b } } & { \hat { \mathbf r } _ { b } } & { \mathbf q } & { \omega _ { b } } & { \delta } \\ { t _ { \mathrm { l o s t } } } & { \log ( \pmb \sigma _ { p } ^ { 2 } ) } & { \log ( \pmb \sigma _ { v } ^ { 2 } ) } & { \delta _ { k - 1 } } & { V _ { c } } & { V } & & \end{array} \right]
$$

where:

(12)

$\hat { \mathbf { r } } _ { b } \in \mathbb { R } ^ { 3 } , d \in \mathbb { R }$ are the unit line-of-sight vector and range estimates in the body frame,

$f \in \mathbb { R }$ is a detection flag (1 if detected this step, else 0),

${ \bf r } _ { b } , \dot { \bf r } _ { b } \in \mathbb { R } ^ { 3 }$ are the relative target position and velocity estimates in the body frame,

${ \mathbf v } _ { b } \in \mathbb R ^ { 3 }$ is the interceptor’s velocity in the body frame,

$ { \mathbf { q } } \in \mathbb { R } ^ { 4 }$ is the interceptor’s attitude quaternion,

$\boldsymbol { \omega } _ { b } \in \mathbb { R } ^ { 3 }$ are the body angular rates,

$\delta \in \mathbb { R } ^ { 4 }$ are the physical control-surface positions $[ \delta _ { e } , \delta _ { a } , \delta _ { r } , \delta _ { t } ]$ after passing through a first-order lag model as described in appendix D,

$t _ { \mathrm { l o s t } } \in$ R is the time since the last successful detection,

• log $\sigma _ { p } ^ { 2 } ,$ log $\pmb { \sigma } _ { v } ^ { 2 } \in \mathbb { R } ^ { 3 }$ are the log position and velocity variances from the KF covariance,

$\pmb { \delta _ { k - 1 } } \in \mathbb { R } ^ { 4 }$ is the previous commanded action in physical units before actuator lag,

$V _ { c } \in \mathbb { R }$ is the closing speed (positive = closing),

$V \in \mathbb { R }$ is the interceptor’s airspeed [m/s].

To provide temporal information, $n _ { \mathrm { s t a c k } } = 3$ frames spaced 0.5 s apart (a 1 s window) are flattened into the network input $\mathbf { O } \times n _ { \mathrm { s t a c k } } \in \mathbb { R } ^ { 1 1 4 }$ . While this introduces aliasing for content with frequency higher than the 1 Hz Nyquist limit, this is acceptable as the newest frame is sampled at the control rate (50 Hz) and all rate quantities are observed directly instead of being inferred by differencing frames.

Although $\hat { \mathbf { r } } _ { b }$ and $d$ are technically redundant with $\mathbf { r } _ { b } ,$ they are deliberately both included in the observation vector. From $\hat { \mathbf { r } } _ { b }$ and d, $\mathbf { r } _ { b }$ can be constructed. However, this is an operation that the neural network would need to learn. $\hat { \mathbf { r } } _ { b }$ is range invariant, making it easy for the policy to interpret. $\mathbf { r } _ { b }$ preserves the true geometry, which is relevant when combining other observable states such as relative target velocity. d is also included separately to inform about closing dynamics without relying on bearing.

The observation uses the KF estimate rather than the raw bounding box. Training directly on $\mathbf { z } _ { k }$ and its finite difference approximation of velocity was attempted, but the policy did not converge to successful interception behavior.

2) Reward function: The reward function $r _ { k }$ is designed to drive the policy towards three behavioral goals: interception, perception and stable flight, and can be formulated as:

$$
{ \begin{array} { r l } { r _ { k } = f _ { \mathrm { n o r m } } { \Bigg ( } c _ { d } r _ { \mathrm { d i s t } } + \mathbf { 1 } _ { \mathrm { i n t } } c _ { \mathrm { i n t } } \left( t _ { \mathrm { m a x } } - t _ { k } \right) } & { { \mathrm { ( i n t e r c e p t i o n ) } } } \\ { + c _ { c } r _ { \mathrm { c a m } } - c _ { \Phi } p _ { \Phi } } & { { \mathrm { ( p e r c e p t i o n ) } } } \\ { - c _ { \Omega } p _ { \Omega } - c _ { u } p _ { u } - c _ { \alpha } p _ { \alpha } { \Bigg ) } } & { { \mathrm { ( s t a b l e ~ f i i g h t ) } } } \end{array} }\tag{13}
$$

where $\mathbf { 1 } _ { \mathrm { i n t } }$ activates on the intercept step and $\begin{array} { r l } { f _ { \mathrm { n o r m } } } & { { } = } \end{array}$ $R _ { \mathrm { m a x } } / t _ { \mathrm { m a x } }$ normalizes the return across episode lengths and control rates. Here, $R _ { \mathrm { m a x } } = 1 0 0 0$ is the maximum achievable return for a single episode and $t _ { \mathrm { m a x } }$ is the time limit upon which timeout triggers. The shaping terms are:

$$
r _ { \mathrm { d i s t } } = \mathrm { t a n h } \left( \frac { d _ { k - 1 } - d _ { k } } { v _ { c l , \mathrm { s a f e } } \Delta t _ { \mathrm { c t r l } } } \right) , \qquad d _ { k } = \| \mathbf { p } _ { T } - \mathbf { p } _ { I } \| ,\tag{14}
$$

$$
\begin{array} { r } { p _ { \Omega } = \displaystyle { \sum _ { i = 1 } ^ { 3 } \Big ( \operatorname* { m i n } \big ( \frac { | \Omega _ { i } | } { \Omega _ { r _ { \operatorname* { m a x } } } } , 1 \big ) \Big ) ^ { 2 } } , \qquad p _ { u } = \frac { 1 } { 1 6 } \displaystyle { \sum _ { i = 1 } ^ { 4 } ( \Delta u _ { i } ) ^ { 2 } } , } \end{array}\tag{15}
$$

$$
r _ { \mathrm { c a m } } = c \operatorname* { m a x } \Bigl ( 0 , 1 - 4 \bigl [ ( c _ { x } - \textstyle \frac { 1 } { 2 } ) ^ { 2 } + ( c _ { y } - \textstyle \frac { 1 } { 2 } ) ^ { 2 } \bigr ] \Bigr ) ,\tag{16}
$$

$$
p _ { \Phi } = \frac { \operatorname* { m i n } ( \tau _ { \mathrm { l o s t } } , \tau _ { \mathrm { m a x } } ) } { \tau _ { \mathrm { m a x } } } ,\tag{17}
$$

$$
p _ { \alpha } = \frac { e ^ { K \bar { \alpha } } - 1 } { e ^ { K } - 1 } ,\tag{18}
$$

$$
\begin{array} { r } { \bar { \alpha } = \operatorname* { m i n } \Bigl ( \operatorname* { m a x } \big ( \frac { | \alpha | - \alpha _ { \mathrm { w a r n } } } { \alpha _ { \mathrm { m a x } } - \alpha _ { \mathrm { w a r n } } } , 0 \big ) , 1 \Bigr ) , \quad K = - 3 . } \end{array}\tag{19}
$$

The distance-progress term $r _ { \mathrm { d i s t } }$ rewards a timestep reduction in true relative distance $d _ { k } \ = \ \| \mathbf { p } _ { T } \ - \mathbf { p } _ { I } \|$ , normalized by the distance the interceptor would cover in one control step at a nominal safe closing speed $v _ { c l , \mathrm { s a f e } } .$ . The tanh saturates this signal, so closing faster than $v _ { c l , \mathrm { s a f e } }$ yields no additional reward, which discourages aggressive high-speed dives and instead promotes steady, controlled closure. The interception bonus $c _ { \mathrm { i n t } } ( t _ { \mathrm { m a x } } - t _ { k } )$ is granted only on the step the target is intercepted $( \mathbf { 1 } _ { \mathrm { i n t } } \ = \ 1 )$ . Its magnitude decreases with the interception timestep $t _ { k }$ , rewarding earlier interceptions, while the $\left( { { t } _ { \operatorname* { m a x } } } - { { t } _ { k } } \right)$ form removes any incentive to loiter near the target and accumulate shaping reward.

The perception terms reward behavior that keeps the target inside camera view. The centering reward $r _ { \mathrm { c a m } }$ is maximal when the target is imaged at the frame centre $\begin{array} { r } { ( c _ { x } = c _ { y } = \frac { 1 } { 2 } ) } \end{array}$ ) and decays parabolically to zero at the frame edge. It is gated by the detection confidence $c ,$ vanishing when no detection is available. It provides a gradient toward keeping the target in view and hence stimulates pure pursuit-like tailchase interception behavior. The FOV penalty $p _ { \Phi }$ is its reactive counterpart: $\tau _ { \mathrm { l o s t } }$ counts the control steps since the last successful detection, and the penalty grows linearly to unity at the loss horizon $\tau _ { \mathrm { m a x } } .$ beyond which the episode is terminated. Together they encourage the agent to keep the target framed before it is lost, and to reacquire it quickly if it is.

The stable flight terms aim to shape the policy towards behavior that is realistic and non-damaging for the airframe and also not stall the aircraft. The body-rate penalty $p _ { \Omega }$ discourages aggressive angular rates $\Omega _ { i }$ (roll, pitch, yaw), each normalized by $\Omega _ { r _ { \mathrm { m a x } } }$ and squared. The smoothness penalty $p _ { u }$ penalizes large timestep changes in the four actuator commands $\Delta u _ { i }$ pushing for smooth control. The exponential AOA penalty $p _ { \alpha }$ enforces high AOA avoidance, remaining zero below a warning angle $\alpha _ { \mathrm { { W a r n } } }$ and rising sharply to unity at the AOA limit $\alpha _ { \mathrm { m a x } }$

The coefficients are calibrated relative to the unit maximum of the progress reward. Penalties weighted above one: $p _ { \Phi }$ $p _ { \alpha }$ , and $p \Omega$ near saturation, can render sustained target loss, incipient stall, or hard manoeuvring net-negative even while the interceptor is closing, whereas $c _ { c } < 1$ ensures the centering reward nudges rather than overrides the pursuit geometry. Appendix F describes the exact weights used in training of the policy used in the real flight tests.

## E. RL algorithm

Proximal Policy Optimization (PPO) [32] was used for training in combination with the Stable-Baselines3 Python library [33]. The algorithm parameters were all kept default, with the exception of clip range ϵ and discount factor γ. ϵ was reduced from 0.2 to 0.05, meaning the policy updates are limited to 5 percent increments per step to increase training stability. $\gamma$ was increased from 0.99 to 0.999 to encourage longer-term path planning in the policy.

## F. Training environment

In [19], a custom environment was developed to run a parallel simulation of more than a thousand quadcopters. In this work, the environment is adapted for fixed-wing aircraft. JAX [34] and Numba [35] are used for efficient compiling, reducing training time and allowing the training to run on a CPU. Training occurs in episodes with a single target $T$ and interceptor I. I is initialized with conditions close to trim. T is modeled as a moving or accelerating point mass. Episodes are terminated when:

$T$ is intercepted : $\left( \mathbf { p } _ { I } - \mathbf { p } _ { T } \right) ^ { \top } \left( \mathbf { p } _ { I } - \mathbf { p } _ { T } \right) \leq r _ { n } ^ { 2 }$

• $T$ is missed : $r _ { n } < d < r _ { m } \& d > 0$

$T$ is lost : $\tau _ { \mathrm { l o s t } } > \tau _ { \mathrm { l } }$ max

• I stalls: $\alpha > 6 0 ^ { \circ }$

• I hits the ground: $h < 1 0 \mathrm { ~ m ~ }$

• I goes above altitude ceiling: $h > 1 0 0 0 \mathrm { ~ m ~ }$

• I goes out of bounds $p _ { I _ { x } } , p _ { I _ { y } } > 1 0 0 0 ~ \mathrm { m }$

$$
t _ { k } > t _ { \mathrm { m a x } }
$$

with $r _ { n } = 3$ m and miss radius $r _ { m } = 5  { \mathrm { m } } . \ \tau _ { \mathrm { l o s t } }$ is the number of control steps since the last successful detection and $\tau _ { \operatorname* { m a x } } =$ 200 steps (4 s at 50 Hz). $t _ { \mathrm { m a x } }$ varies across the curriculum (see appendix G) to accommodate for longer range approaches and faster targets.

Contrary to quadrotors, fixed-wing drones can stall at low airspeeds or high AOA. This proved to be one of the main challenges encountered in this work. Instead of exploring poststall modelling, the approach taken here involved encouraging the agent to avoid stall regimes in the first place. While stall regimes were not clearly defined during the system identification, 25 degree AOA was deemed to be a conservative limit to avoid, regardless of airspeed, as this was near the maximum AOA observed during the system identification flights. At first, episodes were simply cut off when this threshold was crossed, however this resulted in many episodes being lost due to stalls without the agent learning how to avoid them. It is also not possible to penalize AOA, as non-zero AOA is needed to generate lift. Finally, a stall penalty was included that only penalized time steps above a certain AOA threshold.

To create a RL controller that works in a real environment, the simulation should reflect uncertainties, noise, constraints and variance in the underlying observations and model. Domain Randomization (DR) has been successfully used to train a single neural network that is capable of controlling physically distinct quadcopters [21]. In this work, DR is increased from zero to 30 percent around nominal values as the training progresses.

## G. Curriculum learning

The full interception task involves pursuing a far-away maneuvering target from noisy monocular vision estimates under wind, turbulence and domain randomization of the aircraft parameters. This is too difficult to learn from random initialization all at once. Hence, the training is structured into a 10-stage curriculum. In the initial stage, the interception task is relatively easy. The target is close-by, slow moving and not accelerating. The main goal of this stage is to learn how to fly the aircraft and produce motion towards the target. Progression to a successive stage occurs when the interception rate reaches 75 percent. The policy is evaluated every 1000 episodes, to see whether it can advance to the next stage in the curriculum. The final stage uses an interception target of 101 percent, which is impossible, making it train until the timestep budget runs or the training crashes. When progressing between stages, the best policy is transferred (not the latest) and acts a starting point for the training to continue. Appendix G describes the individual stages in the curriculum used for training the policy that was ultimately tested in real flight.

## H. Real interception test flights

Flight tests were conducted at Vliegbasis Valkenburg in the Netherlands. It was quickly discovered that testing the entire system together proved to be difficult initially. Target perception was especially challenging. The interceptor needs to be aligned with the target before the RL policy can take control of the actuators. Furthermore, the detection algorithm was not as robust as anticipated, and particularly struggled with tail-based engagements when the background is not clear and the target has a very thin profile.

1) Virtual stationary target: To separate perception from control, a script was developed that projects a virtual 1.7 meter wide target on the camera image using the pinhole model. This script takes predefined Global Navigation Satellite System (GNSS) coordinates and the interceptor’s attitude and position and computes a bounding box onto the camera image. This artificial bounding box then feeds into the CAKF. This way, the pipeline is identical to the deployment scenario, except there are no detection dropouts and the aligning the interceptor with the target becomes much easier.

2) Virtual moving target: Once stationary virtual targets were successfully intercepted, the script was expanded to also allow a moving virtual target to be projected onto the camera. Instead of a single GNSS coordinate, a series of predefined coordinates plus altitudes and a fixed ground speed simulate a moving target, with the bounding box generation logic identical to the stationary virtual target. The moving target travels between the given coordinates in a looping fashion, progressing to the first coordinate after the last coordinate has been reached, and does so indefinitely, allowing for systematic testing.

3) Real fixed-wing target: A second X-UAV Talon was developed to function as a moving target. To provide a meaningful interception scenario, the target was autonomously flying a 90 meter altitude rectangular waypoint trajectory using AUTO mode. The pilot was instructed to provide evasive stick inputs whilst in AUTO mode, which allows the target to maneuvre away from the flight mission until the pilot lets go of the sticks. This makes it both harder to track and to intercept the target. Airspeed was managed autonomously with throttle by Ardupilot; the pilot could alter demanded airspeed with stick inputs.

## I. Baseline classical controller

To compare the RL with a baseline, vision-based Proportional Navigation (PN) and Pure Pursuit (PP) classical guidance controllers were developed. All three controllers (RL, PN and PP) used the same perception pipeline as described in sections III-B and III-C, meaning differences in performance would be attributable to differences in control.

1) Guidance laws: PN commands an acceleration perpendicular to the LOS and proportional to its rotation rate:

$$
\mathbf { a } _ { \mathrm { P N } } = N V \left( \pmb { \Omega } \times \hat { \mathbf { r } } _ { b } \right) , \qquad \pmb { \Omega } = \mathbf { R } _ { b / i } \big ( \hat { \mathbf { r } } _ { i } \times \dot { \hat { \mathbf { r } } } _ { i } \big )\tag{20}
$$

$$
\hat { \mathbf { r } } _ { i } = \mathbf { R } _ { i / b } \hat { \mathbf { r } } _ { b } , \hat { \mathbf { r } } _ { b } = \frac { \mathbf { r } _ { b } } { \lVert \mathbf { r } _ { b } \rVert } ,\tag{21}
$$

where N is the dimensionless gain and Ω is the inertial LOS rotation rate. $\dot { \hat { \mathbf { r } } } _ { i }$ is the finite difference of $\hat { \mathbf { r } } _ { i }$ over one control step, passed through a first order low-pass filter with a time constant $\tau _ { \lambda } = 0 . 5 \mathrm { s } .$ . Ω is computed in the inertial frame and then converted to the body frame using rotation matrix $\mathbf { R } _ { b / i }$ such that the interceptor’s own rotation does not influence it. The LOS rate can also be derived analytically in its standard vector form [4][5] as $\begin{array} { r } { \Omega = \frac { \mathbf { r } _ { b } \times \dot { \mathbf { r } } _ { b } } { | | \mathbf { r } _ { b } | | ^ { 2 } } } \end{array}$ scaled by $V _ { c } = - \hat { \mathbf { r } } _ { b } \cdot \dot { \mathbf { r } } _ { b }$ instead of $V ,$ , however this depends on the estimated relative velocity $\dot { \mathbf { r } } _ { b }$ to set the direction and magnitude of the command. With the current approach, the commands generated depend only on the measured bearing and the interceptor’s own speed, and are not affected by the noisy estimate of relative velocity.

PP instead drives the angle ε between the camera boresight $\hat { \mathbf { b } } = [ 1 , 0 , 0 ] ^ { \top }$ and the line of sight to zero:

$$
\mathbf { a } _ { \mathrm { P P } } = K V \varepsilon { \frac { \mathbf { e } _ { \perp } } { \lVert \mathbf { e } _ { \perp } \rVert } } , \varepsilon = \operatorname { a r c c o s } { \bigl ( } { \hat { \mathbf { r } } } _ { b } \cdot { \hat { \mathbf { b } } } { \bigr ) } , \mathbf { e } _ { \perp } = { \hat { \mathbf { r } } } _ { b } - { \bigl ( } { \hat { \mathbf { r } } } _ { b } \cdot { \hat { \mathbf { b } } } { \bigr ) } { \hat { \mathbf { b } } } .\tag{22}
$$

so that $K \ [ \mathrm { s } ^ { - 1 } ]$ sets the demanded turn rate per unit of pointing error. LOS should be minimized with respect to the boresight instead of the velocity vector because a fixed camera should be aligned with the target to see.

2) Cascaded autopilot: Both classical guidance laws produce an acceleration command, which must be realized by an aircraft whose only maneuvering force is lift generated from the wings. To achieve the normal acceleration, the aircraft must hence roll the lift vector onto the demanded acceleration vector. A cascaded autopilot performs this mapping. The acceleration command is first translated into the specific force the accelerometer must register,

$$
\mathbf { f } _ { \mathrm { c m d } } = \mathbf { a } _ { \mathrm { c m d } } - \mathbf { g } _ { b } ,\tag{23}
$$

with $\mathbf { g } _ { b }$ gravity expressed in body axes. Its magnitude in the plane normal to the optical axis sets how much lift is required, and its direction sets where that lift must point:

$$
f _ { n , { \mathrm { c m d } } } = \big \| ( f _ { y } , f _ { z } ) \big \| , \phi _ { \mathrm { e r r } } = \mathrm { a t a n 2 } \big ( f _ { y } , - f _ { z } \big ) ,\tag{24}
$$

with $f _ { y }$ and $f _ { z }$ the $y$ and z components of $\mathbf { f } _ { \mathrm { c m d } }$ in the body frame. The roll channel drives $\phi _ { \mathrm { e r r } }$ to zero, aligning the lift vector with the demand, while the pitch channel regulates the achieved normal specific force onto $f _ { n , \mathrm { c m d } }$ through a feedback loop. Both outer loops feed inner body-rate loops with proportional, integral and feedforward terms. A sideslip loop drives $\beta$ to zero through the rudder.

Because $\phi _ { \mathrm { e r r } }$ is an error rather than an absolute bank angle, and because gravity is accounted for once in $\mathbf { f } _ { \mathrm { c m d } }$ rather than through separate corrections in each channel, the mapping remains valid at any attitude. The pitch loop compares the demanded normal specific force $f _ { n , \mathrm { c m d } }$ with the actual specific force, and zeroes the difference with a pitch rate command. In simulation, measured specific force is not available and hence it is reconstructed from the velocity in the body frame:

$$
\tilde { \mathbf { f } } = \frac { \mathbf { v } _ { b } [ k ] - \mathbf { v } _ { b } [ k - 1 ] } { \Delta t } + \boldsymbol { \omega } \times \bar { \mathbf { v } } _ { b } - \mathbf { g } _ { b } .\tag{25}
$$

Because the leading term is a numerical derivative, it amplifies noise on $\mathbf { v } _ { b }$ by $1 / \Delta t$ . Therefore, the estimate is low pass filtered by a first order filter with a time constant $\tau _ { f } = 0 . 0 8 \mathrm { s }$ $\mathbf { A }$ block diagram of the constructed autopilot is shown in figure 2.

![](images/1c67bedf419f1973b2bf426d648af18a47cb14d7f83e6e904cee9c0c1d9c2b92.jpg)  
Fig. 2. A block diagram showing the constructed feedback loops that power the baseline classical autopilot. In simulation, sideslip is computed as $\beta = \arcsin ( v / \| \mathbf { v } _ { b } \| )$ , with v the lateral velocity in the body frame. In real flight, Ardupilot’s EKF estimate for sideslip can be used. Note that unlike the baseline controllers, the RL policy does not observe $\beta .$

A total of sixteen gains were tuned manually, with the inner loops being closed first before the outer loops were tuned. The gains are summarized in appendix E.

After the cascaded autopilot was tuned, the guidance law gains were then selected by sweeping each against interception rate, giving $N = 2$ and $K = 1 \mathrm { { s } ^ { - 1 } }$

## IV. RESULTS

## A. System identification

The interceptor weighs approximately 2.87 kg. The moments of inertia are estimated to be: $I _ { x x } ~ = ~ 0 . 1 4 \mathrm { k g } \cdot \mathrm { m ^ { 2 } }$ $I _ { y y } = 0 . 2 0 \mathrm { k g } { \cdot } \mathrm { m } ^ { 2 }$ and $I _ { z z } = 0 . 3 4 \mathrm { k g } { \cdot } \mathrm { m } ^ { 2 }$ . All cross-coupling terms are assumed to be zero. The surface area of the wing is $0 . 4 6 \mathrm { m ^ { 2 } }$ , the wingspan is 1.718 m and the mean aerodynamic chord is 0.265 m. The dihedral angle of the V-tail was measured to be approximately 35 degrees. A 12.5 inch diameter, 6.5 inch pitch propeller was used in combination with an X2820 KV570 electric motor. For the thrust model, data from a test report<sup>2</sup> was used. The estimated maximum thrust at zero airspeed is 38.23 N. The maximum velocity was estimated to be around 40 m/s based on airspeeds achieved in flight.

In table I, the estimated aerodynamic coefficients of the aircraft are given for the three system identification flights that were performed. Flight 1 utilized Ardupilot’s extended Kalman filter’s estimate of the AOA, whereas during flight 2 and 3, a wind vane on an angle encoder was used to measure the real AOA. Additionally, flights 2 and 3 were flown in MANUAL mode, meaning there is no stabilization and each axis was excited separately. Flight 1 was mostly flown in STABILIZED mode, which means yaw and roll are excited simultaneously on lateral stick inputs.

Figure 3 shows the aerodynamic coefficients estimates as a flight-by-flight comparison. Especially flight 2 and 3 agree well. This is likely because of the use of the AOA vane, instead of the Extended Kalman Filter (EKF) estimated AOA onboard the flight controller.

Figure 4 shows the modeled versus measured aerodynamic coefficients over a ten second segment of flight 2. The model was identified from flight 3, so the comparison is out of sample. While there are sizeable residuals, the model follows the trend of the measurements.

![](images/e9ca2bbc4747d6f728e2dfcca1c0aa0e3cbeda799dd2992680770fc4d00e206a.jpg)

TABLE I  
ESTIMATED AERODYNAMIC STABILITY AND CONTROL DERIVATIVES
<table><tr><td>Category</td><td>Coefficient</td><td>Flight 1</td><td>Flight 2</td><td>Flight 3</td></tr><tr><td rowspan="5">Pitching moment</td><td> $\overline { { \mathbf { R } ^ { 2 } } }$ </td><td>0.4794</td><td>0.6809</td><td>0.6889</td></tr><tr><td> $C _ { m 0 }$ </td><td>-0.0035</td><td>-0.0113</td><td>-0.0127</td></tr><tr><td> $C _ { m _ { \alpha } }$ </td><td>-0.0508</td><td>-0.1885</td><td>-0.1724</td></tr><tr><td> $C _ { m _ { q } }$ </td><td>-1.9179</td><td>-3.0066</td><td>-2.4160</td></tr><tr><td> $C _ { m _ { \delta _ { e } } }$ </td><td>0.1536</td><td>0.3647</td><td>0.2907</td></tr><tr><td rowspan="4">Lift</td><td> $\overline { { \mathbf { R } ^ { 2 } } }$ </td><td>0.5943</td><td>0.7853</td><td>0.7967</td></tr><tr><td> $C _ { L 0 }$ </td><td>0.2140</td><td>0.3793</td><td>0.2963</td></tr><tr><td> $\boldsymbol { C } _ { L _ { \alpha } }$ </td><td>1.3442</td><td>3.1066</td><td>2.8529</td></tr><tr><td> $C _ { L _ { \delta _ { \epsilon } } }$ </td><td>1.6628</td><td>0.4925</td><td>0.4181</td></tr><tr><td rowspan="3">Drag</td><td> $\overline { { \mathbf { R } ^ { 2 } } }$ </td><td>0.4316</td><td>0.5727</td><td>0.4473</td></tr><tr><td> $C _ { D 0 }$ </td><td>0.0797</td><td>0.1552</td><td>0.1232</td></tr><tr><td> $k _ { \mathrm { i n d u c e d } }$ </td><td>0.3142</td><td>0.1835</td><td>0.1303</td></tr><tr><td rowspan="6">Rolling moment</td><td> $\scriptstyle \mathbf { R } ^ { 2 }$ </td><td>0.2281</td><td>0.2754</td><td>0.2486</td></tr><tr><td> $C _ { l 0 }$ </td><td>-0.0005</td><td>-0.0010</td><td>-0.0007</td></tr><tr><td> $C _ { l _ { \beta } }$ </td><td>-0.0074</td><td>-0.0176</td><td>-0.0180</td></tr><tr><td> $C _ { l _ { p } }$ </td><td>-0.0781</td><td>-0.0996</td><td>-0.0910</td></tr><tr><td> $C _ { l _ { r } }$ </td><td>0.0156</td><td>0.1029</td><td>0.0727</td></tr><tr><td> $C _ { l _ { \delta _ { a } } }$ </td><td>0.0188</td><td>0.0367</td><td>0.0317</td></tr><tr><td rowspan="6">Yawing moment</td><td> $\overline { { \mathbf { R } ^ { 2 } } }$ </td><td>0.3204</td><td>0.3565</td><td>0.4278</td></tr><tr><td> $C _ { n _ { \beta } }$ </td><td>0.0132</td><td>0.0215</td><td>0.0206</td></tr><tr><td> $C _ { n _ { p } }$ </td><td>-0.0198</td><td>-0.0797</td><td>-0.0524</td></tr><tr><td> $C _ { n _ { r } }$ </td><td>-0.0062</td><td>-0.0509</td><td>-0.0342</td></tr><tr><td> $C _ { n _ { \delta _ { a } } }$ </td><td>0.0010</td><td>0.0108</td><td>0.0059</td></tr><tr><td> $C _ { n _ { \delta _ { r } } }$ </td><td>0.0242</td><td>0.0316</td><td>0.0263</td></tr><tr><td rowspan="3">Side force</td><td> $\overline { { \mathbf { R } ^ { 2 } } }$ </td><td>0.1243</td><td>0.0313</td><td>0.1075</td></tr><tr><td> $C _ { Y _ { \beta } }$ </td><td>-0.0954</td><td>-0.0577</td><td>-0.1361</td></tr><tr><td> $\underline { { C _ { Y _ { \delta _ { \tau } } } } }$ </td><td>-0.1689</td><td>0.1276</td><td>-0.2167</td></tr></table>

The $\scriptstyle \mathbf { R } ^ { 2 }$ values represent the Pearson correlation coefficient for each respective axis, where larger values indicate a better agreement between the measured data and the fitted model.

![](images/99dfad028c3c593bae13643ef0c457d26a38fb2ca8e9fe80bc888723037b3e30.jpg)  
Fig. 3. A flight by flight comparison of the estimated aerodynamic coefficients. The dashed line indicates identical estimates between two flights.

OLS regression assumes the residuals are white noise. To check whether this assumption holds, the autocorrelation function (ACF) for the residuals was computed for lag 0 to 300 and is shown in figure 5. For most axes, the residuals are indeed uncorrelated over time. However, $C _ { L } .$ , and consequently $C _ { D }$ which is dependent on $C _ { L }$ , shows significant autocorrelation, especially at lower lag. This means the residuals are potentially not white noise, indicating some deficiency in the aerodynamic model. This was likely due to the thrust model, which neglects advance ratio and other physically correct additions in favor of simplicity. Instead of perfecting the system identification, these results were deemed ”good enough” to start training a controller as domain randomization of the aircraft parameters alleviates the need for really accurate system identification.

For each coefficient, the estimate with the strongest fit and best agreement between flights was adopted. The final values used in training the RL policy are described in appendix H.

![](images/c772a8076977a7999b2599d99dfb228b0a05b649a6f94c49fa79366335a983d7.jpg)

Fig. 4. A ten second segment from flight 2 showcasing how the model identified using flight 3 (orange line) compares to the measured aerodynamic coefficients from flight 2 (blue line) over time.  
![](images/5e723f282da7ee30f90abbce0614fac2c286ddc55a9f885599910ae2c3c4b2d2.jpg)  
Fig. 5. The autocorrelation function of the residuals for each aerodynamic coefficient.

## B. Cascaded autopilot for baseline classical guidance

Figure 6 shows the simulated tracking performance of the autopilot used for the baseline classical controller in the full non-linear model. The two inner rate loops are the fastest elements of the cascade. The roll rate loop (a) reaches 90 percent of demand in 0.10 s with at most 7 percent overshoot. The pitch-rate loop (b) is even faster at 0.06 s but more oscillatory, overshooting by 26 percent before settling within 2 percent in roughly 0.44 s. The sideslip regulator (c) is excited by an instantaneous lateral gust rather than a demand step, and returns sideslip angle to zero from initial values of $5 ^ { \circ }$ to 20<sup>◦</sup>. The first undershoot reaches approximately $- 6 ^ { \circ }$ for the largest lateral gust, with residual oscillations fully decayed by 2.5 s.

![](images/d6dbc64f22bfd49259eeba9bf2887dc20cc82ca3608abcf3476eca69bebd42e2.jpg)  
Fig. 6. The performance of the cascaded autopilot controller in the full nonlinear model. (a) The inner roll-rate loop tracking response to a 0.5, 1, 2 and 4 rad/s roll rate demand. (b) The inner pitch-rate loop tracking response to a 0.1, 0.2, 0.4 and 0.8 rad/s pitch rate demand. (c) The inner sideslip angle loop nullifying response to a 5, 10, 15 and 20 degree initial sideslip. (d) The outer roll angle tracking loop response to a 15, 30, 45 and 60 degree roll demand. (e) The outer normal acceleration tracking loop response to a 3, 6, 10 and $\mathrm { { i } } 5 \mathrm { { m } / \mathrm { { s } } ^ { 2 } }$ normal acceleration demand.

The two outer loops are three to five times slower than the inner loops they command. The roll channel (d) reaches 90 percent of a commanded bank angle in 0.16-0.22 s and settles within 2 percent in approximately 0.75 s, with nearly no overshoot. The normal-acceleration loop (e) is the slowest and least damped element: it rises in 0.16-0.20 s but overshoots by 15-26 percent and takes 1.0-1.5 s to settle.

## C. RL learning

Throughout the course of this research more than 120 different RL policies were trained with varying degrees of success in simulation. The scope of this research was sim-to-real transfer. Hence, only the policy that successfully intercepted targets in real flight is discussed.

Learning was structured into 10 stages (see appendix G). The progress of the training in terms of average interception rate and mean reward is shown in figure 7. In the initial stage, the agent quickly progresses from an approximate 30 percent interception rate to nearly 95 percent interception rate. At this stage, the task is simple and maintaining the aircraft’s attitude usually results in an interception. As the training progresses through the curriculum, the agent learns to adapt to new circumstances, which causes it to recover from initial drops in performance. An interception rate above 80 percent is not seen after completing stage 4. In stage 8, the task is so difficult that the agent is not able to progress above the threshold of

![](images/1958e91743e9ce4185ad22ec5c5c8b1bbfaad48902ae087341c3e4a2d3116c1b.jpg)  
Fig. 7. The top and bottom figures showcase the progression of the average interception rate and reward throughout the curriculum respectively. Data is plotted in increments of a thousand episodes over the cumulative amount of training steps. Each colour represents a separate stage.

75 percent interception rate before the training stops (due to CPU constraints).

## D. Performance in simulation

1) Single episode analysis: Before analyzing performance across many episodes, it is insightful to examine a single episode to understand how the policy behaves. In figure 8, the trajectories of the interceptor and target in a single engagement are shown. The target inertial velocity in NED was set to $1 0 \mathrm { { m / s } \ ( \it { v _ { N } } = 4 \mathrm { { m / s } , \it { v _ { E } } = 6 \mathrm { { m / s } ) } } }$ ) and it was manoeuvering $( a _ { N } = 1 \mathrm { m } / \mathrm { s } ^ { 2 } , a _ { E } = - 1 \mathrm { m } / \mathrm { s } ^ { 2 } )$ . During this episode, there was a steady wind speed of 2 m/s and a 0.5 m/s Gaussian gust at each timestep. The aircraft parameters were randomized ±15 percent around nominal values. Target wingspan was also randomized ±10 percent around 1.7 m. The detection latency was set at 0.06 seconds. The interceptor was able to successfully intercept the target within 6 seconds under these circumstances.

Figure 9 shows the ground truth distance to the target versus the estimated distance produced by the KF, with the red areas indicating periods of no detection. Even during these blind moments, the KF produces a reasonable closing distance estimate that follows the ground truth, leading to a successful interception. Figure 10 showcases that AOA throughout the course of the engagement stays within the ±10 degree penalty thresholds. This shows the policy has learned to fly conservatively, which is as intended. The interceptor’s airspeed varied between a minimum of 21 and a maximum of 33 m/s, which is well above the aircraft’s stall speed at these relatively low AOA values.

Figure 11 shows the state of the actuators throughout the engagement. The policy commands full throttle immediately. This is a behavior that was consistent across every policy trained throughout the course of this research. However, this particular policy will reduce throttle if the target is lost for a prolonged period of time. This behavior started occurring once the reward for quick interceptions was reduced and the reward for keeping the target inside the FOV was increased.

![](images/39f4a056ae859b70c3d33171968db9f4500ac3da7733c77095801de9f817841c.jpg)

Fig. 8. The trajectories of the interceptor (blue) and target (red) throughout the course of a single simulated engagement. The circle and cross indicate the start and finish respectively. The dashed orange line indicates the KF estimate of the target’s position.  
![](images/15cc214bcec2a66f9f449c87030b72d05708addc1b207d736086139e83c1ee1c.jpg)  
Fig. 9. The ground truth (GT) range (blue) versus the Kalman Filter (KF) range estimate (dashed orange). The red areas indicate periods where the target is not being detected. The dashed green line indicates the range threshold for interception.

Besides throttle, the flight control surfaces all show slight oscillations. Interestingly, the policy aggressively oscillates rudder deflection, which causes subsequent oscillatory yawing behavior as can be seen in figure 12. Earlier policies that were trained with less strict AOA penalties showed the same behavior but rather in the pitch axis, with the elevator being excited continuously up to its limits.

Figure 13 shows the same engagement but with the trajectories generated by the baseline classical controllers also included. All three interceptors manage to intercept the target during this engagement. Figure 14 shows the Euler angles of the classically guided interceptors throughout the engagement. Figure 15 shows the control surfaces of the classically guided interceptors over the course of this episode.

![](images/72947c4d435ada4a50e7cd30b238acdc71642d58a49c65f8c1e1f89e195066cf.jpg)  
Fig. 10. The angle of attack throughout the simulated engagement. The dashed lines indicate the angle thresholds beyond which high absolute angle of attack is punished. The red dotted line indicates the cutoff threshold to end the episode early due to stall or airframe collapse.

![](images/5e55d457469d2bea50649950e01785e61beb623139e3c957ba0b354655603d82.jpg)  
Fig. 11. The state of the actuators throughout the simulated engagement. Aileron, rudder and elevator deflection are represented by the red, green and blue lines respectively. Throttle percentage [0,1] is shown by the brown line.

![](images/e1bf42fbc37b998d85fb64484b172abddd38dfa89186f96b3344b436cbc7b995.jpg)  
Fig. 12. The interceptor’s attitude in Euler angles throughout the simulated engagement. Roll, pitch and yaw are indicated by red, green and blue lines respectively.

2) Multiple episodes analysis: The trained RL policy and the baseline classical controllers were tested in the curriculum that the policy was trained in. In figure 16 the performance of each controller throughout the curriculum is shown. To show the effects of limited sensing through the noisy, $6 0 ^ { \circ }$ horizontal FOV monocular camera, the performance is also shown with an idealized 180<sup>◦</sup> FOV camera, which always detects the target as long as it is in the FOV. For each controller, each of the nine curriculum stages and each of the two camera types, 896 episodes were simulated, for a total of 48,384 episodes.

![](images/db50efcb950f8995eecdeb56e6d21c09661dc3a73a4c7d852397d29ca27b6e31.jpg)  
Fig. 13. The trajectories of the RL interceptor (blue), PN interceptor (orange), PP interceptor (green) and target (red) throughout the course of a single engagement. The circle and cross indicate the start and finish respectively. Noise and other randomness was seeded identically for each interceptor.

![](images/0ca09feb5e35168cb0816262837e6756a0d9586de42f6e3c61bd25a7e18c2dca.jpg)

![](images/b414c5ae1eaf3e1899afc80666006f77107edcbbaba8c450c6928d9d31b739e0.jpg)  
Fig. 14. The Euler angles of the classically guided interceptors during the same simulated engagement. Roll, pitch and yaw are represented in red, green and blue respectively. The top and bottom subfigures shows the PN and PP interceptors respectively.

All controllers perform nearly perfect at stage 0. At this stage, there is very little perceptual noise, no wind or gusts, and the target is near stationary and not accelerating at a low range from the interceptor. As the curriculum progresses, all controllers lose performance, but not equally. Overall, RL performs best and is only matched in the final stages by ideal PP and even outperformed slightly by ideal PN. Ideal RL is even better and degrades much slower. PP and PN perform near equally until stage 5, after which PN degrades more rapidly than PP. Real PP is less affected by camera noise than PN as its guidance law commands it to keep the target in view, resulting in robustness despite noisy detections.

![](images/832a95adf9cf971ef16d3ae21154af6c4a958b03d0c888dbc9b372f136b07e99.jpg)  
Fig. 15. The control surfaces of the classically guided interceptors during the same simulated engagement. Elevator, aileron and rudder are represented in blue, red and green respectively. The top and bottom subfigures shows the PN and PP interceptors respectively. Throttle is set to maximum at the start of the engagement for both controllers.

![](images/02799ed117de20276354cf5790f9efdff10f93fe61b0c399ffdf90ceb729ebe9.jpg)  
Fig. 16. The performance of the RL (blue), PN (orange) and PP (green) controllers throughout the nine stages. The solid lines and circles represent performance with a 60<sup>◦</sup> FOV, noisy camera undergoing increasing disturbances as specified in appendix G. The dashed lines and triangles represent performance with an ideal camera with a 180<sup>◦</sup> FOV, with perfect detections at all stages. Each point represents 896 simulated episodes (64 parallel environments running 14 episodes each). The interception rate plotted on the y-axis represents the fraction of those episodes ending with a successful interception. The shaded area represents the 95 percent confidence interval.

To draw meaningful conclusions about the behavior and performance of the controllers in general, 14,256 additional episodes were simulated to investigate the effect of a single variable condition. 336 episodes were generated with nominal conditions similar to stage 5 of the curriculum. Each condition comprises 144 episodes: 5 conditions at increasing target ground speed (5 m/s increments) and no acceleration, 5 at increasing target acceleration $( 1 . 1 5 \mathrm { m } / \mathrm { s } ^ { 2 }$ increments), 5 at increasing wind strength (2.25 m/s increments), 5 with increasingly long clustered camera dropout bursts (up to 2.8 s), 4 with 50 m initial range increments, and 5 with varying approach geometry from tail-chase to head-on.

Figure 17 shows the interception rate under these conditions. Target speed has little effect over most of its range, but the three controllers differ. RL seems to degrade only at target speeds of 20 m/s, whereas PN degrades slightly at target speeds of 15 m/s, but recovers at speeds of 20 m/s. PP outperforms the other controllers at higher target speeds. On the contrary, target acceleration seems to have a real degrading effect on the performance of the controllers, with higher accelerations leading to lower interception rates for all controllers. However, the classical controllers are more affected than RL in terms of performance. Wind affects each controller differently. PP and RL are mostly unaffected across the whole sweep, but PN drops below 60 percent interception rate at 9 m/s winds. Increasingly, long camera dropout bursts lead to decline in performance of all controllers, but surprisingly this decline plateaus beyond 1.4 s dropouts. Initial range is non-monotone. Close range initialization provides little time to correct the initial geometry leading to relatively low performance. The highest performance is at 50-150 m for all controllers. Finally, approach geometry produces the largest spread in the figure, and its minimum lies at $4 5 ^ { \circ }$ rather than head-on for RL and PP. RL falls from 83.3 percent in a tail chase to 47.2 percent at $4 5 ^ { \circ }$ , recovering to 57.6 percent head-on. PP is worst affected, dropping to 29.2 percent and 34.0 percent at these two geometries, while PN is the least sensitive and even outperforms RL at 45<sup>◦</sup>. PP performing the worst is expected as it always aims to nullify the LOS with respect to the boresight of the camera and hence will result in tail chases. A head-on initialization will therefore likely result in lots of near misses and also loss of target lock, as the interceptor can be too slow in turning around to reacquire the target after passing it. Interestingly, the performance of PN and RL are nearly equal at head-on and 45<sup>◦</sup> engagements, suggesting that RL adapts it pursuit strategy based on the closing velocity.

Figure 18 gives a breakdown of the reasons for termination of episodes per condition and per controller. Across all conditions, RL outperforms the other controllers with the exception of 20 m/s target speed, in which the classical controllers are better. 9 m/s wind barely has an impact on performance in RL and PP. PN however degrades from 81 percent to 73 percent interception rate.

Longer initial range has an impact on performance. With initial ranges between 150 to 200 meters, the performance of RL drops slightly to 82 percent interception rate, with a small portion of episodes surprisingly being terminated due to loss of control. At increasingly higher ranges, the KF range estimate becomes unstable, such that it produces estimates far beyond the training envelope, causing erratic behavior. For the classical controllers, a longer range means more engagements terminate in a target loss, which is expected as the detection probability is lower at longer range.

A quickly accelerating target also reduces interception rates for all controllers, mostly due to target loss. RL is most robust against maneuvering targets and is still able to intercept 78 percent of the time with a majority of the other episodes resulting in a near miss. Surprisingly, a maneuvering target does not cause episodes to be terminated in a loss of control. Unsuccessful terminations are due to near miss and target loss. PN and PP struggle with maneuvering targets with interception rates of 54 and 62 percent respectively. Extended periods of camera dropouts impact the interception rate of all controllers, leading mainly to more target loss. In training, the agent sees mainly tail-based engagements, where the target is moving away from the interceptor’s initial position. However for evaluation, beam (sideways) and head-on approaches were also simulated. These show a significant reduction in performance, with both a large amount of near misses and target losses. PP is the worst affected during head-on approaches.

![](images/13d8e97bb9f24683f131dc239b39da6db65a1b379d353ea63b1132608b264656.jpg)

![](images/6fe90152c0bde39d4b304feb97463cb26de6b37bc059e0f4f49b559195a49140.jpg)

![](images/4236f9c5295b52e647856034f637ef4f9a53504a5071007a57c935c7ca558f92.jpg)

![](images/288178ae27495fcc4170b54fc3a2fe8c4024a160afb4ac3ab8410d605cecded3.jpg)

![](images/ebc14dc8640ab166bb943bf9d996fe4e41ee3c346aa3c89bbd1b4941b4cfab6e.jpg)

![](images/d2a319d4e4a8cb1503dfb6352b2dd5d8c2f2e25576dddcbaefb6ca904cdecbfc.jpg)

Fig. 17. Interception rates for increasingly difficult conditions. The error bars indicate the Wilson 95 percent intervals. The nominal conditions of stage 5 hold except for one variable. In (a), the interception rate is shown for increasingly faster, non-accelerating targets. In (b), the target is increasingly accelerating with a mean initial speed of 7.3 m/s. In (c), the max wind speed increases up to 9 m/s. In (d), the mean detection dropout length increases up to 2.8 s. In (e), the range is increased. In (f), the approach geometry is varied from a tail-chase to a head-on engagement.  
![](images/57c4edce3918b7d2e9c65691b7a265e176a74560065b43ccbc84623feb212ea6.jpg)  
Fig. 18. A breakdown of the reasons episodes were terminated, categorized by condition. Green, purple, orange, red and grey represent the fraction of episodes terminated due to successful interception, target loss, near misses, loss of control and timeout.

![](images/96b447d65914a27e30f2ff192b17d648747ca9285debc2061657681735d38489.jpg)  
Fig. 19. 3D trajectories of the interceptor during 6 real flight engagements of a stationary virtual target. The green dot represents the start of the engagement, that is the interceptor is flying in MANUAL mode and the RL controller is overriding RC. The red star indicates the relative position and altitude of the virtual target. The black cross indicates the closest point of approach (CPA) in meters. The viridian color scheme shows how the trajectory progresses through time. The aircraft in white with grey outline is a schematic representation of the interceptor’s attitude at discrete time steps. The aircraft model is not to scale.

Finally, the condition where the target has a speed of 20 m/s (excluding acceleration) is examined. Unsurprisingly, some episodes terminate in a timeout, as the interceptor can lack the speed to catch the target before the time runs out, however RL is more affected than PN and PP. The latter even improves as target speed increases. This is likely due to PP resulting in a tail-based engagement. When the target has a higher speed, the relative velocity will be lower, allowing for more precise terminal guidance. This is reflected by the fact that there are also no near misses for this condition.

## E. RL controller in real flight

Initial in flight testing of the RL controller proved to be difficult. A moving fixed-wing target was hard to keep in the FOV whilst engaging the RL policy. Furthermore, unsuccessful attempts could either be due to a sim-to-real gap in the controller, or the perception could be insufficient.

![](images/0295cd2ef1c41a42af2c73a9a694b626bc1c68c42ba327e62bc3021f33089ce7.jpg)  
Fig. 20. The commanded flight surface deflections and throttle during each real flight engagement of a virtual stationary target. In blue, orange and green, the normalized surface deflection of the ailerons, elevator and rudder respectively. The dashed grey line represents the commanded throttle.

1) Stationary virtual target: Flight tests with a virtual target resolved this issue. During the first successful interception tests, the video<sup>3</sup> showed the interceptor passing through the target 5 out of 6 times.

Figure 19 shows the trajectories of the interceptor during all 6 real flight engagements of a stationary virtual target. Engagements 2, 4 and 6 show the interceptor passing through the virtual target barring a few meters error. Engagements 1 and 5 were stopped just before the target was being passed through because the pilot switched to STABILIZED mode, meaning the RL controller is interrupted. However as the onboard video recording shows, the interceptor still passed through the target shortly after without pilot stick inputs. Only engagement 3 shows a clear failure with the interceptor seemingly turning away at about 70 meters from the target. This could be due to a bad KF estimate of the target, however this can not be confirmed as the KF estimation was not logged during this flight. Also visible in figure 19 is the fact that the controller does not stop after the target is intercepted. In training and simulation, the engagement stops, but in real flight, the controller experiences a flight through the target and after passing quickly tries to reacquire it in its FOV with an aggressive maneuver (visible in engagements 2, 4 and 6).

Figure 20 shows the actuator commands by the RL controller during each engagement. As observed in simulation, the policy heavily oscillates the rudder. This is also observed on the onboard video which shows the plane yawing back and forth significantly. Throttle is mostly full, except in engagement 2 and 3. In engagement 2, the throttle drops immediately after the policy takes control, only to recover quickly to full throttle after 0.7 s. In engagement 3, the throttle is reduced for a longer period of time. This points to a loss of target track, as the same behavior is observed in simulation when a target is lost for a prolonged period.

With the RL policy being able to intercept a virtual target, the sim-to-real transfer has been successful controller-wise. The policy produces flight behavior that mimics simulation. It successfully avoids high body rates and erratic control inputs. One of the key insights was to ensure that the deployment scripts running on the interceptor provided the same telemetry frequency as the policy was trained on. Earlier flight tests provided 10 to 20 Hz telemetry, whereas the policy was trained on 40-50 Hz.

2) Moving virtual target: After successfully intercepting virtual stationary targets, the controller was challenged with a moving virtual target. The target was flying a 90 meter altitude loop at a fixed ground speed of 8 m/s through a set of coordinates that defined an oval-shaped track. Eight attempts were made, of which four resulted in an interception (distance $\leq 3  { \mathrm { m } } )$ , two resulted in a miss (3 m ≤ distance ≤ 8 m) and two were aborted due to divergence. The trajectories of the interceptor and virtual target during each engagement are shown in figure 21. In figure 22, the distance between the interceptor and the target throughout the engagement is shown.

Overall, the controller performed well with a moving target. During the first engagement, a closest point of approach (CPA) of 0.3 m was achieved. This is tenfold below the chosen interception threshold of 3 m, but also far below the wingspan of the target (1.7 m), indicating precision even as the target moves. During engagement four, five and seven, similar performance was observed. Engagement six and eight also showed the interceptor closing in quite well, but ultimately remained above the threshold of 3 m. Engagement two and three were aborted due to the interceptor steering away from the target.

In simulation, it was observed that tail-based approach geometries had the highest success rate. In real flight, approach geometry does not necessarily negatively influence performance. Engagement four shows an approximately sideways engagement, resulting in a successful interception (CPA: 1.3 m). Engagement seven and eight start nearly head-on, with engagement seven resulting in a precise interception (CPA: 0.8 m) and engagement eight resulting in a near miss (CPA: 7.3 m). Engagement eight shows the interceptor making a very sharp move to try and align itself with the target. Engagement six is fully head-on and the interceptor does not try to align itself with the target’s trajectory but just flies into it head on (CPA: 5.6 m).

![](images/d45a85e9cdd97629db79438f545f70e1a4db4b8cd78ee04fbb8701256799981a.jpg)  
Fig. 21. 3D trajectories of the interceptor during 8 real flight engagements of a moving virtual target. The green and red dot represent the interceptor’s and target’s position respectively at the start of the engagement. The red and black cross indicate the closest point of approach (CPA) of the target and interceptor respectively. The grey dashed line shows connects the target and interceptor at the CPA. The red line shows the target’s trajectory. The viridian color scheme shows how the interceptor’s trajectory progresses through time. The aircraft in white with grey outline is a schematic representation of the interceptor’s attitude at discrete time steps. The aircraft model is not to scale.

Engagement two and three were aborted because the interceptor started to steer away from the target. This was likely due to the KF producing an unreasonable range and position estimate. Once the estimates are far beyond what the policy has seen in training, it starts to diverge. During these two engagements, the policy reduces throttle, which is consistent behavior occurring when it has lost track of the target.

3) Real target: The ultimate test of the system was the interception of a real target. For this test, a fixed-wing target was flying a rectangular pattern at 90 meters altitude in AUTO mode. Meanwhile, the operator of the target was instructed to give stick inputs to maneuver and actively avoid being collided into.

The interceptor was aimed at the target manually. The controller would be switched on after a target detection was confirmed on the camera feed. After switching on the controller, RC commands generated by the RL policy take over the aircraft, with pilot stick inputs having no effect until the controller is switched off again. The relative trajectory was monitored by both pilots from the ground. When the interceptor got into collision range, the controller was switched off and both pilots maneuvered in STABILIZED mode to avoid an actual impact.

![](images/81fd1b93bb8c98dc3af93a85edb8136b0414c0cecfd7ae73e87375d7a1f3f555.jpg)  
Fig. 22. The distance between the interceptor and the target throughout the course of the eight engagements. Each colour indicates a separate engagement, with the cross in the corresponding colour representing the shortest distance between the interceptor and the target. CPA: closest point of approach.

A total of six engagements were flown with the RL controller engaged. Three engagements resulted in a successful interception with both pilots taking over to avoid a collision. During the first engagement<sup>4</sup>, the interceptor detected a bird or another aircraft, which was mistakenly assumed by the human operators to be the target drone. This resulted in normal behavior, but chasing a phantom target of which detection faded as the controller was turned on. The third interception was the first successful engagement<sup>5</sup>. Both pilots actively took over control manually to avoid a collision. The fifth engagement was a successful interception. The sixth and final attempt was also a successful interception. The interceptor banked 130 degrees during this chase (as visible in the flight logs) and was able to keep the target in sight before the pilot took back control.

Unfortunately, the target’s flight logging got corrupted a mere 17 seconds after launch. Hence, there are no GNSS coordinates of the target’s trajectory. To still get an indication of CPA, a mixture of information was used. Firstly, the target was mostly flying a rectangular pattern between four predetermined GNSS coordinates with a desired altitude of 90 meters. The target was maneuvering to make it more difficult for the interceptor, but it can be assumed that the target was approximately flying the mission. For each engagement, the CPA with this trajectory can be computed. By assuming a constant airspeed of 15 m/s, the reconstructed target trajectory based on the mission can be compared to the interceptor’s trajectory as shown in figure 23.

![](images/f3b99a66d85c58dad2855260af4fd2d58ed32c5857d2105ee2d3f4224f845466.jpg)  
Fig. 23. 3D trajectories of the interceptor during 6 real flight engagements of a fixed-wing target. The green and red dot represent the interceptor’s and target’s position respectively at the start of the engagement. The red and black cross indicate the closest point of approach (CPA) of the target and interceptor respectively. The grey dashed line shows connects the target and interceptor at the CPA. The red line shows the target’s mission flight path. The CPA here is computed based on the proximity to the mission flight path, not the actual target trajectory. The viridian color scheme shows how the interceptor’s trajectory progresses through time. The aircraft in white with grey outline is a schematic representation of the interceptor’s attitude at discrete time steps. The aircraft model is not to scale. The target’s trajectory shown here is an estimated trajectory, assuming a constant 15 m/s speed and no maneuvering or deviation from the predetermined flight path. Actual CPA may have been smaller because the target was also maneuvering which is not represented in this figure. The target trajectory is computed based on the assumption that the target had a constant speed and the red line is simply an extrapolation of where the target would have been if the theoretical CPA was indeed the closest point of approach.

Secondly, the video recordings can be analyzed to see how much the target drone takes up the frame. Using the pinhole model equations (3, 4), a depth estimate can be reconstructed from the image.

In figures 24, 25 and 26, still images of the target being intercepted during engagement three, five and six respectively are shown. The interceptor comes even closer on video, but the images shown here contain relatively little static noise and also show the entire wingspan of the aircraft. For engagement three, the interceptor and target bank opposite ways to avoid a collision, meaning this range estimate will be quite accurate as the closest point of approach. For engagement five and six, the interceptor flies underneath the target. Therefore the CPA is likely closer than these images suggest.

The wingtips are identified manually in each frame, and the tip-to-tip separation L measured in screen-capture pixels

![](images/76977b5df6328d42f3d31085f3b815de54b1fe6895f493ab7f5e4198fa2fda2f.jpg)

Fig. 24. Engagement 3. 3.3 meter range estimate. The date and time stamp are incorrect and can be neglected; they are only included to show the full frame.  
![](images/e4b1c5b4329b7aa474b6813a77eb0ad6e55b6f3e3ec72eb139e4a42fb2b5ae40.jpg)

Fig. 25. Engagement 5. 6.0 meter range considering approximately 45 degrees of yaw with respect to the optical axis of the interceptor. The date and time stamp are incorrect and can be neglected; they are only included to show the full frame.  
![](images/c24d27523e2ba635a841d40c0402d88a700064488b462a09856dfabb35c770ae.jpg)  
Fig. 26. Engagement 6. 8.1 meter range considering approximately 20 degrees of yaw with respect to the optical axis of the interceptor. The date and time stamp are incorrect and can be neglected; they are only included to show the full frame.

![](images/93afb73c5eaf39051ca1a8254a0dbca2a15c033f664f6ad3784cc0fb398c779e.jpg)  
Fig. 27. 1 Hz KF logs of the estimated range (a), position estimate variance (b), estimated target velocity (c), velocity estimate variance (d), time elapsed since the last detection (e). Each colour represents a single engagement plotted on a common time axis.

is converted to native camera pixels through $s = 1 2 8 0 / W _ { \mathrm { a c t } } .$ where $W _ { \mathrm { a c t } }$ is the width of the active video area. With $f _ { x } =$ 1108.5 px from Eq. (4) and $W \ : = \ : 1 . 7 1 8 \mathrm { m }$ , Eq. (3) gives $Z = f _ { x } W / w _ { p x }$ with $w _ { p x } = s L ;$
<table><tr><td>L [px]</td><td></td><td> $w _ { p x }$  S</td><td>[px]</td><td> $\overline { { Z \ [ \mathrm { m } ] } }$ </td></tr><tr><td>(3)</td><td> ${ \overline { { { \sqrt { 5 1 5 ^ { 2 } + 9 2 7 ^ { 2 } } } } } } = 1 0 6 0 . 4 $ </td><td>0.552</td><td>585.6</td><td>3.25</td></tr><tr><td>(5)</td><td> ${ \sqrt { 3 7 6 ^ { 2 } + 1 5 0 ^ { 2 } } } = 4 0 4 . 8$ </td><td>0.553</td><td>224.0</td><td>8.50</td></tr><tr><td>(6)</td><td> ${ \sqrt { 3 5 4 ^ { 2 } + 1 8 6 ^ { 2 } } } = 3 9 9 . 9$ </td><td>0.554</td><td>221.5</td><td>8.60</td></tr></table>

A target yawed by ψ relative to the camera axis presents an effective width W cos ψ, so the values above are upper bounds. Estimating $\psi \approx 4 5 ^ { \circ }$ in engagement 5 and $\psi \approx 2 0 ^ { \circ }$ in engagement 6 yields corrected ranges of 6.0 m and 8.1 m, respectively.

Lastly, besides the mission and the images, the KF running on the interceptor gives a 1 Hz log containing the range estimate, which gives an indication of perceived range. In figure 27, various KF parameters are plotted over time during each engagement. The top subfigure shows the KF range estimate.

![](images/9cabed4082f6254398b67dc3cd01ac637a89dec8fc5aa6b7e655791e221dffaa.jpg)  
Fig. 28. The root mean square (rms) in degrees of rudder deflection versus the range between the target and the interceptor. The blue line shows rudder deflection amplitude against real range as seen when intercepting stationary targets. The purple and yellow dashed lines show the rudder deflection amplitude at various ranges based on presented artificial range estimates.

For engagements 3, 5 and 6 the final range estimate is 3, 4 and 6 meters respectively. This corroborates the range estimates obtained using other methods. Besides the final estimates, these plots also show how target state estimation progresses over time during real interception flights. Engagements 1, 3 and 6 show a very similar pattern in the range estimate, which starts high and gradually decreases over time. This explains why during flight 1, the interceptor did not cut throttle: the filter’s estimate remained consistent with a closing engagement throughout. However, the video shows the detector had locked onto something else in the sky. The detection dropout plot shows that six seconds into the engagement, the target has been lost for 3.7 seconds. Surprisingly, the interceptor keeps flying normally on the filter’s estimate rather than breaking off. This shows the strength of introducing detection dropouts to the training environment. As long as the target is far away and not maneuvering unpredictably, a long detection dropout can still result in a successful pursuit and possibly regain of target lock if the target being chased is real.

Engagement 5 is interesting because the range estimate is actually very low upon initialization (12 m), but then quickly climbs to 54 m, only to drop all the way to 4 m when the target is being intercepted. The perception model possibly locked onto different illusionary targets mid-engagement, reducing the reliability of the KF estimate. This is also reflected by the other plots in figure 27, which show a climbing target speed estimate, as if the target is accelerating away. Also, the position and velocity estimate variances are high compared to the actual range and velocity estimates. The detection dropout plot corroborates these findings as it shows that 2 seconds into the engagement, the target was lost for 1.7 seconds. After that it (re)acquires the real target and proceeds to intercept successfully.

## F. Rudder behavior in RL

The RL policy showed significant oscillation of the rudder, both in simulation and in real flight, resulting in an oscillatory yaw rate that decreases in amplitude as the target is approached. This was investigated and seems to be correlated with perceived range to the target as shown in figure 28. Rudder oscillation amplitude, quantified as the root mean square rudder deflection in degrees, correlated nearly linearly with perceived range. If the policy is presented artificial high or low range estimates, the rudder oscillation amplitude will be high or low respectively, regardless of actual range.

## V. DISCUSSION

## A. Performance

In simulation, RL outperformed the classical controllers under almost all circumstances. Against fast, non-maneuvering targets, PP overtakes RL. At a target speed of $2 0 \mathrm { m \cdot s ^ { - 1 } }$ it achieves 91.8 percent interception rate against the policy’s 70.2 percent. The policy’s training distribution reached only $1 2 \mathrm { m } { \cdot } \mathrm { s } ^ { - 1 }$ target speed per axis, favoring to leave a velocity margin for accelerating targets. Hence, this condition lies partly outside the training distribution. RL commanded rudder independently, PP and PN only use rudder to coordinate turns. PN underperformed RL but mostly matched PP, only being outperformed in the later stages of the curriculum where perception is severely limited by noise. This is mainly due to the sensor being a strapdown monocular camera. Traditionally, PN is used on missiles with seeker heads which ensure the sensor is pointing at the target. That assumption is not valid for the interceptor used in this work. Notably, PN is implemented in a range-invariant fashion. This differs from textbook approaches [4][5] which scale the commands by closing velocity. This approach was initially attempted, but abandoned as it was observed that the closing velocity estimate from apparent bounding box dimensions was too noisy to properly scale the guidance commands. The performance of the baseline classical guidance algorithms also depends on the underlying cascaded autopilot that was manually tuned.

In real flight, the RL policy achieved sim-to-real transfer. With virtual targets its performance showed high accuracy at a variety of initial geometries, attitudes and wind speeds. The performance was ultimately successfully validated by the interceptions of real, maneuvering fixed-wing MALE targets. Human operators had to intervene repeatedly to prevent an actual collision between the interceptor and its target.

## B. Perception

Perception was a key challenge in this work. Initial results showed that simply providing bounding-box data as input was not sufficient to let the policy converge to reliable interception behavior in training. The Kalman filter resolved this by providing relative target state estimates. Additionally, the Kalman filter also made the classical controllers more robust. With the bounding-box only approach, these controllers would have no input to act on when the target left the FOV. However, the filter produces estimates even when the target is not detected.

In training, the challenge was to simulate all real-life perception noise. Since the performance of the computer vision algorithm was not quantitatively known, the approach was to train the controller with as much noise as possible. Initially, mainly white noise in the pixels and frame-by-frame detection dropouts were provided, but this is easily compensated for by the filter. By clustering detections dropouts into larger chunks, it becomes harder for the filter to create reliable estimates. Consequently, the policy learns to behave conservatively. In real flight, the perception might be less noisy, meaning more optimal approaches could be flown. During interceptions of real targets, it was indeed observed that detection can drop out over prolonged periods. Figure 27 shows that during engagement 5 with a real target, there were no detections for at least 1.7 seconds at one point. Despite this, the policy was able to successfully intercept the target, showing robustness during perception degradation.

One noise source that was not simulated in training is the detection of false positives or even other potential wrong targets such as birds, clouds, or even high-altitude civilian aircraft. This was especially visible during real target engagement 1, where the RL controller was engaged whilst locking onto a different target. The policy currently does not discern between a real target and something else that still results in detections feeding into the Kalman filter.

The Kalman filter currently uses all detections with a confidence score above 0.3. Using these scores as weights could actually result in a better prediction with a lower uncertainty. In the current implementation, the detection algorithm’s confidence score is lost during the filtering.

Range estimation is based on the Kalman Filter back projecting bounding box dimensions onto a known wingspan of 1.7 meters. While the simulation and training environments randomize the actual target dimensions around this nominal value, the range estimate will breakdown for targets that are much smaller or larger. This does not necessarily mean that performance degrades however. In real flight, spurious oddsized detections did not result in sudden changes in course.

## C. Flight model

To effectively train a policy that is transferable to a real platform, robustness is needed rather than accuracy. The dynamics model used in training should cover the real life physics, but also not over-complicate things. To identify the aircraft parameters, three flights were flown, with two having an AOA vane mounted on the aircraft. The results in table I are not perfect, with no correlation scores above 0.8. This could probably be greatly improved by flying more flights with an AOA vane and specifically exciting different axis independently. Additionally, a different regression method could be used. OLS was used here for simplicity, but figure 5 showed that for lift and drag, the residuals are autocorrelating, which violates the OLS assumption that the noise is independent. Some non-linear identification method could possibly generate a more valid estimate. A better thrust model could potentially also resolve the autocorrelation problem.

However, the scope of this work was sim-to-real transfer. Rather than focusing on creating a very high fidelity simulation, domain randomization and other techniques were successfully used to overcome gaps between simulation and real flight.

## D. Training

Once a policy was successfully used to intercept stationary virtual targets in real flight, further development and training of other RL policies was halted. This was not because it could not have been improved, but rather the choice was deliberately made to test the policy with as many different real flight experiments as possible whilst keeping performance and limitations comparable across experiments.

One of the downsides of reinforcement learning is that it requires training of the neural network, which can take a long time. This makes iteration cycles long. To alter the observations of the policy or the curriculum usually means retraining the policy from scratch, costing multiple hours of CPU time. Transfer learning allows the weights of one policy to be transferred to the next. This is helpful when quickly testing late-stage curriculum changes, where the policy already should have learned how to fly. However, to obtain reproducible results, the curriculum and parameters should not be altered mid-training.

The analyses in sections IV-D1 and IV-D2 were performed after a bug fix in the training environment. Specifically, the pre-fix training environment mixed aircraft acceleration in the inertial frame with aircraft acceleration in the body frame (equations 27 and 28). This lead to the additional term $- \omega _ { b } \times \mathbf v _ { b }$ fighting the aircraft during turning in the training environment. In other words, the RL policy was trained in an environment in which turning was approximately twice as hard as would be realistic. This explains why the performance in the corrected environment is lower than what is suggested by the curriculum learning progression in figure 7, which is in the environment where the bug is still present. The bug was discovered after real flight tests were successfully performed and hence instead of retraining a new policy, the current policy was maintained for comparability at the cost of performance. Nonetheless, this error shows the robustness of the RL policy. Not only did performance degrade only slightly in simulation after the fix, the policy also managed to intercept targets in a real environment, showcasing sim-to-real transferability even when trained in an imperfect physics model. The performance of the policy in the original training environment is included in appendix I, which unsurprisingly shows higher performance across all domains.

One interesting but unexplored idea to potentially improve performance of the policy is to introduce a reward for producing a correct relative target state estimate. Currently the policy is driven towards perception by keeping the camera aimed at the target and the target in the FOV, but this might not always be the optimal solution. The reward function could stimulate low covariances in the state estimates or even penalize proportional to the error in the state estimate.

## E. Transferability

Initially, time since start of the engagement was also provided to the policy, but this proved to hinder sim-to-real transfer. During real flight testing, the controller is running continuously, but its authority over the aircraft is being toggled on and off by a human operator. Time since start of the engagement quickly blows up beyond the training envelope, leading to unpredictable behavior. Additionally, the parameter adds no value to the agent’s decision making.

By using a framework that trains policies with up to 30 percent domain randomization in simulation, the controller becomes robust to sim-to-real gaps in the flight model. This controller is end-to-end, meaning even low-level dynamics needed to be simulated.

## F. RL controller flight behavior

One of the more surprising observations during this research was the continuous rudder oscillations produced by the policy both in simulation and in real flight. The aircraft is visibly heavily yawing back and forth, even in level wings flight, which means it is unlikely to be attributable to turn coordination. The oscillatory yawing fades as perceived range shrinks, as if the interceptor is straightening itself out before impact. At first it was thought that the policy might have learned to triangulate for better range estimates, however replacing the noisy camera with an ideal one reduced KF position error to nearly zero, whilst rudder oscillation was unchanged. Then it was hypothesized that perhaps the oscillatory motion aids in reacquisition of target lock after a detection dropout, as the oscillation effectively widens the FOV. However this was unlikely, as reacquisition seemed independent of rudder oscillation. It was thought that maybe the policy uses the generated sideslip as an airbrake. This could keep throttle at a maximum, which is preferred to avoid its ramp up, whilst slowing closing velocity allowing for preciser maneuvering. This theory was refuted as airspeeds were comparable with or without the oscillations. It could also have been an artifact of the pre-correction training environment physics, which artificially made banking a lot harder, but this appears unlikely as removing rudder actually reduced performance less in the pre-correction training environment than in the corrected training environment. Since replacing the RL rudder control with the sideslip damper used in the cascaded autopilot for the baseline controllers did not reduce the interception rate significantly, it was hypothesized that the behavior must be optimizing some reward, however this was not confirmed. In the end, this analysis also highlights one of the downsides of using neural networks for control: its behavior can be a black box to an extent.

## G. Future work

This work has demonstrated fixed-wing, vision-based interception using an RL controller in real flight. The focus here was to demonstrate the proof-of-concept; once a policy was successfully flight tested, development of other policies stopped. Future work could continue this research. The approach in this study was to create a very conservative policy, with the highest chance of successful sim-to-real transfer, but this is at the cost of performance. It would be very interesting to see exactly when the policy fails to transfer to reality. In the current setup, policies are trained with up to 30 percent domain randomization of the aircraft parameters. This value is based on other work, but such high domain randomization also degrades performance and it is not certain that it is needed to such a degree to ensure sim-to-real transfer. Additionally, the policy described has been steered significantly by including large incentives to keep the target in sight. While this makes the policy more robust in the case of lots of camera noise or bad computer vision, it also hinders interception performance. It would be insightful to see where the limits of all of these considerations lie and where the policy breaks down in real flight.

Previous work has investigated quadrotor-based interception in simulation. This work now shows it is possible both in simulation and real flight for fixed-wing drones. This indicates that the possibilities are not gated by the platform used. The work outlined here could be extended for other platforms like tailsitters or X-wing interceptors.

One caveat of the work demonstrated here is that upon initializing an interception, for both the RL and classical controllers, the target needs to be detected. The KF produces a relative target position estimate. If there are no prior detections, the KF can not produce a position estimate, meaning the controllers have nothing to act on. Having the controller acquiring the target autonomously would greatly improve combat readiness.

The classical controllers were not tested in real flight. While their they were outperformed in simulation, for a complete comparison they would also need to be tested in real flight.

Currently, the relative state of the target is determined solely by visual observations. For midcourse guidance, a rough, ground-based radar estimate of the relative position of the target can be used. Future controllers could fuse such estimates with onboard detections, leading to a more robust behavior and also allowing autonomous target (re)acquisition for terminal guidance. For terminal guidance, monocular cameras provide a cheap and established way to detect targets. However, other sensors such as acoustic sensors, stereovision cameras, eventbased cameras and infrared sensors can also be used. Multiple monocular cameras of different focal lengths can even be used to fuse a sharp focused image with a wide, unsharp image. This can make target reacquisition much easier in case of a missed interception.

## VI. CONCLUSION

In this work, an end-to-end neural network was trained using reinforcement learning to control a fixed-wing interceptor aircraft tasked with taking down an uncooperative target. The policy maps filtered, monocular vision observations directly to actuator commands. In simulation it outperformed baseline classical guidance algorithms. The controller achieved simto-real transfer, showcasing a comparable performance during real flight experiments. The system was ultimately successfully validated by intercepting real fixed-wing, medium altitude, long endurance targets, showing the potential of autonomous, intelligent interceptors in a modern air-defense context.

## APPENDIX A

## AERODYNAMICS MODEL

This section briefly describes the aerodynamics model that was used in the Python-based simulation and training environments.

1) State vector: The state vector x can be defined as:

$$
\mathbf { x } ^ { \top } = \left[ \mathbf { p } _ { i } \quad \mathbf { v } _ { i } \quad \mathbf { q } \quad \omega _ { b } \quad \delta _ { e } \quad \delta _ { a } \quad \delta _ { r } \quad \delta _ { t } \right]\tag{26}
$$

where $\mathbf { p } _ { i } , \mathbf { v } _ { i } \in \mathbb { R } ^ { 3 }$ are the position and velocities in the North-East-Down (NED) inertial frame, $\mathbf { q } = \left[ q _ { 0 } q _ { 1 } q _ { 2 } q _ { 3 } \right] ^ { 1 }$ , is the unit quaternion attitude representation, $\omega _ { b } \in \mathbb { R } ^ { 3 }$ are the angular rates in the body frame, $\delta _ { e } , \delta _ { a } , \delta _ { r } \in [ - 1 , ~ 1 ]$ are the normalized control surfaces actuator states which are scaled by the actuator deflection limits in radians and $\delta _ { t } \in [ 0 ,$ , 1] is the normalized throttle state. All actuator states undergo firstorder lag and hence: $\begin{array} { r } { \dot { \delta } _ { j } = \frac { \delta _ { j , \mathrm { c m d } } - \delta _ { j } } { \tau _ { i } } , \qquad j \in \{ e , a , r , \tilde { t } \} } \end{array}$ , with $\tau _ { e , a , r } = 0 . 0 5$ s and $\tau _ { t } = 0 . 4 ~ \mathrm { s } .$

2) Translational dynamics: The translational dynamics in the body frame can be defined as:

$$
\begin{array} { r } { \dot { \mathbf { v } } _ { b } = \frac { 1 } { m } \mathbf { F } _ { \mathrm { t o t } } - \boldsymbol { \omega } _ { b } \times \mathbf { v } _ { b } . } \end{array}\tag{27}
$$

Here, m is the mass of the aircraft and $F _ { t o t }$ is the total force acting on the body. The simulator propagates position and velocity in NED inertial coordinates, giving:

$$
\begin{array} { r } { \dot { \bf p } _ { i } = { \bf v } _ { i } , \dot { \bf v } _ { i } = \frac { 1 } { m } { \bf R } _ { b / i } ^ { \top } { \bf F } _ { \mathrm { t o t } } . } \end{array}\tag{28}
$$

Vectors are resolved either in the NED inertial frame, whose axes point north, east and downward, or in the body frame, whose axes $x _ { b } , \ y _ { b } , \ z _ { b }$ point forward along the fuselage, out the right wing and downward respectively. The corresponding body-frame velocity components are denoted $u ,$ v and w:

$$
\mathbf { v } _ { b } = \mathbf { R } _ { b / i } \mathbf { v } _ { i } = \left[ u \quad v \quad w \right] ^ { \top } , \quad V = \| \mathbf { v } _ { b } \|\tag{29}
$$

The rotation matrix $R _ { b / i }$ maps the inertial frame to the body frame:

$$
\mathbf { R } _ { b / i } = { \left[ \begin{array} { l l l } { 1 - 2 { \big ( } q _ { 2 } ^ { 2 } + q _ { 3 } ^ { 2 } { \big ) } } & { 2 { \big ( } q _ { 1 } q _ { 2 } + q _ { 0 } q _ { 3 } { \big ) } } & { 2 { \big ( } q _ { 1 } q _ { 3 } - q _ { 0 } q _ { 2 } { \big ) } } \\ { 2 { \big ( } q _ { 1 } q _ { 2 } - q _ { 0 } q _ { 3 } { \big ) } } & { 1 - 2 { \big ( } q _ { 1 } ^ { 2 } + q _ { 3 } ^ { 2 } { \big ) } } & { 2 { \big ( } q _ { 2 } q _ { 3 } + q _ { 0 } q _ { 1 } { \big ) } } \\ { 2 { \big ( } q _ { 1 } q _ { 3 } + q _ { 0 } q _ { 2 } { \big ) } } & { 2 { \big ( } q _ { 2 } q _ { 3 } - q _ { 0 } q _ { 1 } { \big ) } } & { 1 - 2 { \big ( } q _ { 1 } ^ { 2 } + q _ { 2 } ^ { 2 } { \big ) } } \end{array} \right] }\tag{30}
$$

The matrix is orthonormal, so its inverse is its transpose, $\mathbf { R } _ { b / i } ^ { - 1 } = \mathbf { R } _ { b / i } ^ { \top } = \mathbf { R } _ { i / b }$

The total force is comprised of the aerodynamic forces, the thrust generated by the propeller and the force due to gravity:

$$
\mathbf { F } _ { \mathrm { t o t } } = \mathbf { F } _ { \mathrm { a e r o } } + \mathbf { F } _ { \mathrm { t h r u s t } } + \mathbf { F } _ { \mathrm { g r a v i t y } } ,\tag{31}
$$

where:

$$
\mathbf { F } _ { \mathrm { g r a v i t y } } = R _ { b / i } \left[ 0 \quad 0 \quad m g \right] ^ { \top } ,\tag{32}
$$

$$
\mathbf { F } _ { \mathrm { t h r u s t } } = \left[ \delta _ { t } T _ { \mathrm { m a x } } \sqrt { \frac { \rho } { \rho _ { 0 } } } \left( 1 - \hat { V } ^ { 2 } \right) \quad 0 \quad 0 \right] ^ { \top } ,\tag{33}
$$

$$
\mathbf { F } _ { \mathrm { a e r o } } = \bar { q } S \left[ \begin{array} { c c c } { \cos \alpha \cos \beta } & { - \sin \beta } & { - \sin \alpha \cos \beta } \\ { \cos \alpha \sin \beta } & { \cos \beta } & { - \sin \alpha \sin \beta } \\ { \sin \alpha } & { 0 } & { \cos \alpha } \end{array} \right] \left[ \begin{array} { c } { - C _ { D } } \\ { C _ { Y } } \\ { - C _ { L } } \\ { - C _ { L } } \end{array} \right] .\tag{34}
$$

Here, g is the gravitational acceleration. $T _ { m a x }$ is the maximum thrust achievable at full throttle with zero airspeed.

$\begin{array} { r } { \hat { V } = \operatorname* { m i n } \Bigl ( \frac { V } { V _ { \mathrm { m a x } } } , 1 \Bigr ) } \end{array}$ , with V being the airspeed and $V _ { \mathrm { m a x } }$ the airspeed at which thrust is zero. ρ is the air density (ISA troposphere model) and $\rho _ { 0 } = 1 . 2 2 5 ~ \mathrm { k g } \mathrm { m } ^ { - 3 }$ is the reference air density at sea level. In this research, $\sqrt { \frac { \rho } { \rho _ { 0 } } }$ is effectively 1 due to the low altitude testing (max 120 m). S is the wing surface area and $\bar { q }$ is the dynamic pressure. $\begin{array} { r } { \alpha = \arctan { \frac { w } { u } } } \end{array}$ is the Angle Of Attack $( \mathrm { A O A } ) , \beta = \arcsin { \frac { v } { V } }$ is the sideslip angle. $C _ { L } , C _ { D }$ and $C _ { Y }$ are the non-dimensional lift, drag and sideforce coefficients and can be described as:

$$
C _ { L } = C _ { L _ { 0 } } + C _ { L _ { \alpha } } \alpha + C _ { L _ { \delta _ { e } } } \delta _ { e } ,\tag{35}
$$

$$
C _ { D } = C _ { D _ { 0 } } + k C _ { L } ^ { 2 } ,\tag{36}
$$

$$
C _ { Y } = C _ { Y _ { \beta } } \beta + C _ { Y _ { \delta _ { r } } } \delta _ { r } .\tag{37}
$$

Here, k is the induced drag factor. $C _ { L _ { 0 } } , C _ { L _ { \alpha } } , C _ { L _ { \delta _ { e } } }$ are coefficients that describe how the lift is shaped by AOA and elevator deflection. $C _ { D _ { 0 } }$ is the zero-lift drag coefficient. $C _ { Y _ { \beta } } , C _ { Y _ { \delta _ { \tau } } }$ are coefficients that shape how the sideforce is shaped by sideslip angle and rudder deflection.

3) Rotational dynamics: The rotational dynamics are given by:

$$
\dot { \boldsymbol { \omega } } _ { b } = \mathbf { J } ^ { - 1 } \Big ( \mathbf { M } _ { b } - \boldsymbol { \omega } _ { b } \times ( \mathbf { J } \boldsymbol { \omega } _ { b } ) \Big ) ,\tag{38}
$$

where J is the inertia tensor of the aircraft and $\mathbf { M } _ { b }$ is the total moment vector which is approximated to contain only aerodynamic moments:

$$
\mathbf { M } _ { b } = \bar { q } S \left[ b C _ { l } \quad c C _ { m } \quad b C _ { n } \right] ^ { \top } .\tag{39}
$$

Here, b is the wingspan and c is the mean aerodynamic chord. $C _ { l } , C _ { m }$ and $C _ { n }$ are the dimensionless aerodynamic moment coefficients that describe how airflow generates rotational moments about the body axes:

$$
C _ { l } = C _ { l _ { 0 } } + C _ { l _ { \beta } } \beta + C _ { l _ { p } } \frac { p b } { 2 V } + C _ { l _ { r } } \frac { r b } { 2 V } + C _ { l _ { \delta _ { a } } } \delta _ { a } + C _ { l _ { \delta _ { r } } } \delta _ { r } ,\tag{40}
$$

$$
C _ { m } = C _ { m _ { 0 } } + C _ { m _ { \alpha } } \alpha + C _ { m _ { q } } \frac { q c } { 2 V } + C _ { m _ { \delta _ { e } } } \delta _ { e } ,\tag{41}
$$

$$
C _ { n } = C _ { n _ { 0 } } + C _ { n _ { \beta } } \beta + C _ { n _ { p } } \frac { p b } { 2 V } + C _ { n _ { r } } \frac { r b } { 2 V } + C _ { n _ { \delta _ { a } } } \delta _ { a } + C _ { n _ { \delta _ { r } } } \delta _ { r } .\tag{42}
$$

Here, $p , q , r$ are the body angular rates.

4) Attitude kinematics: The attitude kinematics are defined by:

$$
\begin{array} { r } { \dot { \mathbf { q } } = \frac { 1 } { 2 } \left[ { \begin{array} { r r r } { - q _ { 1 } } & { - q _ { 2 } } & { - q _ { 3 } } \\ { q _ { 0 } } & { - q _ { 3 } } & { q _ { 2 } } \\ { q _ { 3 } } & { q _ { 0 } } & { - q _ { 1 } } \\ { - q _ { 2 } } & { q _ { 1 } } & { q _ { 0 } } \end{array} } \right] \boldsymbol \omega _ { b } . } \end{array}\tag{43}
$$

5) V-tail configuration: The interceptor built for this study has a V-tail ruddervator configuration instead of a conventional rudder and elevator. Hence:

$$
\delta _ { e , \mathrm { { e f f } } } = \frac { 1 } { 2 } \cos ( \Gamma ) ( \delta _ { v _ { L } } + \delta _ { v _ { R } } ) ,\tag{44}
$$

$$
\delta _ { r , \mathrm { e f f } } = \frac { 1 } { 2 } \sin ( \Gamma ) ( \delta _ { v _ { L } } - \delta _ { v _ { R } } ) ,\tag{45}
$$

where Γ is the dihedral angle and $\delta _ { v _ { L } } , \delta _ { v _ { R } } \in [ - 1 , 1 ]$ are the normalized left and right ruddervator deflection.

## APPENDIX BORDINARY LEAST SQUARES REGRESSION

Ordinary Least Squares describes the regression problem as follows:

$$
\mathbf { z } = \mathbf { X } { \boldsymbol { \theta } } + \mathbf { v } .\tag{46}
$$

Here, z is the vector of the measured dependent variables. X is the regressor matrix formed from measured independent variables. θ is the unknown true parameter vector. Finally, v is the residual vector. The OLS parameter estimate θ<sup>ˆ</sup> is obtained by minimizing the sum of squared residuals:

$$
J ( \pmb { \theta } ) = ( \mathbf { z } - \mathbf { X } \pmb { \theta } ) ^ { T } ( \mathbf { z } - \mathbf { X } \pmb { \theta } ) .\tag{47}
$$

The necessary condition for the minimum is:

$$
\mathbf { X } ^ { T } \mathbf { z } = \mathbf { X } ^ { T } \mathbf { X } \hat { \pmb { \theta } } .\tag{48}
$$

Solving this yields the solution:

$$
\begin{array} { r } { \pmb { \hat { \theta } } = ( \mathbf { X } ^ { T } \mathbf { X } ) ^ { - 1 } \mathbf { X } ^ { T } \mathbf { z } . } \end{array}\tag{49}
$$

This OLS parameter vector estimate gives the best leastsquares fit of the linear model to the gathered flight data and was consequently used to create a high-fidelity simulation to train the RL controller.

## APPENDIX C KALMAN FILTER

This section describes the Kalman filter used to filter the bounding-box signals and output relative state estimates.

1) State vector and process noise model: The filter tracks the target’s dynamic state:

$$
\pmb { x } _ { k } = \left[ \pmb { \dot { p } } _ { k } \right] = \left[ p _ { N } p _ { E } p _ { D } v _ { N } v _ { E } v _ { D } a _ { N } a _ { E } a _ { D } \right] ^ { \top } ,\tag{50}
$$

where ${ \pmb { p } } _ { k } , \dot { \pmb { p } } _ { k } , \ddot { \pmb { p } } _ { k } \in \mathbb { R } ^ { 3 }$ are the target position, velocity and acceleration in NED. By including acceleration in the state vector, the filter can follow a maneuvering target. By working in a NED inertial frame, the target’s motion is decoupled from the interceptor’s ego-motion.

Between measurements, the state evolves under a discrete constant acceleration model with additive Gaussian process noise,

$$
{ \pmb x } _ { k } = \pmb \Phi { \pmb x } _ { k - 1 } + { \pmb w } _ { k - 1 } , \qquad { \pmb w } _ { k - 1 } \sim { \mathcal N } ( { \bf 0 } , { \pmb Q } ) ,\tag{51}
$$

with the sampling interval $\Delta t = 1 / f _ { c } = 0 . 0 2 \mathrm { s }$ matching the control and perception frequency $f _ { c } = 5 0 \mathrm { H z }$ . In training, $f _ { c }$ is randomized between 40 and 50 Hz. With $\pmb { I } _ { 3 }$ and ${ \bf 0 } _ { 3 }$ the $3 \times 3$ identity and zero matrices, the transition matrix Φ is

$$
\Phi = \left[ \begin{array} { c c c } { { I _ { 3 } } } & { { \Delta t I _ { 3 } } } & { { \frac { 1 } { 2 } \Delta t ^ { 2 } I _ { 3 } } } \\ { { \bf 0 } _ { 3 } } & { { I _ { 3 } } } & { { \Delta t I _ { 3 } } } \\ { { \bf 0 } _ { 3 } } & { { \bf 0 } _ { 3 } } & { { I _ { 3 } } } \end{array} \right] .\tag{52}
$$

Real targets do not obey constant acceleration, hence process noise is introduced to explain the change of acceleration (jerk).

The process noise is obtained by treating jerk as a zeromean white process of intensity (power spectral density) q and integrating over $\Delta t \colon$

$$
\pmb { Q } = q \left[ \frac { \Delta t ^ { 5 } } { 2 0 } I _ { 3 } \quad \frac { \Delta t ^ { 4 } } { 8 } I _ { 3 } \quad \frac { \Delta t ^ { 3 } } { 6 } I _ { 3 } \right] , \qquad q = 5 \times 1 0 ^ { - 4 } .\tag{53}
$$

$Q$ is the covariance of the error that the prediction model makes in a single step.

2) Measurement model: Only target pixel coordinates are used, not their temporal derivatives. Each detection where $c \geq$ 0.3 is converted to a measured NED position $\boldsymbol { m } _ { k } . ~ c _ { x } , c _ { y }$ and normalized width w are back projected through the pinhole model using the image size $W _ { \mathrm { i m g } } \times H _ { \mathrm { i m g } } = 1 2 8 0 \times 7 2 0$ px, and the nominal target wingspan $W = 1 . 7$ m using equations 3, 4 and 5 yielding the camera-frame point $r _ { \mathrm { c a m } } = [ X , Y , Z ] ^ { \top }$ This is rotated into the body frame and then into NED and translated by the interceptor’s own NED position $\mathbf { \nabla } _ { \mathbf { p } _ { i } }$ using the body-to-NED rotation $\mathbf { R } _ { i / b }$ formed from the autopilot attitude quaternion $\pmb q \colon$

$$
\begin{array} { r } { { \pmb { m } } _ { k } = { \pmb { p } } _ { i } + { \bf R } _ { i / b } { \bf R } _ { b / c } { \pmb { r } } _ { \mathrm { c a m } } , ~ { \bf R } _ { b / c } = \left[ \begin{array} { c c c } { 0 } & { 0 } & { 1 } \\ { 1 } & { 0 } & { 0 } \\ { 0 } & { 1 } & { 0 } \end{array} \right] . } \end{array}\tag{54}
$$

The resulting measurement is linear in the state

$$
\begin{array} { r } { m _ { k } = H \pmb { x } _ { k } + \pmb { \nu } _ { k } , \qquad \pmb { \nu } _ { k } \sim \mathcal { N } ( \mathbf { 0 } , \pmb { R } _ { k } ) , } \end{array}\tag{55}
$$

with $H = \left\lceil I _ { 3 } \quad 0 _ { 3 \times 6 } \right\rceil$ such that velocity and acceleration are never measured directly but instead are estimated from the sequence of position fixes through the model in (52). $\nu _ { k }$ represents the zero mean Gaussian measurement noise with covariance $\scriptstyle { R _ { k } }$

3) Range-dependent measurement covariance: Range is recovered from apparent size and is far less certain than bearing, with an uncertainty that grows with distance. This is modeled by constructing $\scriptstyle { \cal R } _ { k }$ in a LOS frame and rotating it into NED. With range $d = \| m _ { k } - \pmb { p } _ { i } \|$ (floored at 0.5 m), the lateral and depth standard deviations are

$$
\sigma _ { \mathrm { l a t } } = \bigg | \frac { \partial X } { \partial u _ { c } } \bigg | \sigma _ { c } = \frac { Z } { f _ { x } } \sigma _ { c } = \frac { d ( \sigma _ { c , 0 } + \sigma _ { c , 1 } d ) } { f _ { x } }\tag{56}
$$

and

$$
\sigma _ { \mathrm { d e p t h } } = \Big | \frac { \partial Z } { \partial w _ { p x } } \Big | \sigma _ { s } = \frac { Z ^ { 2 } } { f _ { x } W } \sigma _ { s } = \frac { d ^ { 2 } ( \sigma _ { s , 0 } + \sigma _ { s , 1 } d ) } { f _ { x } W } ,\tag{57}
$$

with $\sigma _ { c , 0 } = 3 . 0 \mathrm { p x } , \sigma _ { c , 1 } = 0 . 0 5 \mathrm { p x } / \mathrm { m } , \sigma _ { s , 0 } = 2 . 0 \mathrm { p x } , \sigma _ { s , 1 } =$ $0 . 0 2 \mathrm { p x } / \mathrm { m }$ and $Z \approx d .$ . The three sources of error are assumed to be uncorrelated. The covariance in the LOS frame $R _ { k } ^ { \mathrm { L O S } }$ is therefore:

$$
R _ { k } ^ { \mathrm { L O S } } = \left[ \begin{array} { c c c } { \sigma _ { \mathrm { d e p t h } } ^ { 2 } } & { 0 } & { 0 } \\ { 0 } & { \sigma _ { \mathrm { l a t } } ^ { 2 } } & { 0 } \\ { 0 } & { 0 } & { \sigma _ { \mathrm { l a t } } ^ { 2 } } \end{array} \right]\tag{58}
$$

and is mapped into NED by

$$
\pmb { R } _ { k } = \pmb { L } \pmb { R } _ { k } ^ { \mathrm { L O S } } \pmb { L } ^ { \top } , \qquad \pmb { L } = \left[ \hat { \pmb { r } } \quad \pmb { v } _ { 1 } \quad \pmb { v } _ { 2 } \right] ,\tag{59}
$$

where $\hat { \pmb { r } } = ( \pmb { m } _ { k } - \pmb { p } _ { i } ) / d$ is the LOS unit vector and ${ \boldsymbol { v } } _ { 1 } , { \boldsymbol { v } } _ { 2 }$ are two unit vectors orthogonal to each other and to ${ \hat { \mathbf { \mathit { r } } } } ,$ completing an orthonormal basis. Bearing can be trusted from few measurements, whereas range needs a longer measurement time to be trusted.

4) Initialisation: On the first detection that exceeds the confidence threshold, the filter is seeded from the measurement, with zero velocity and acceleration and large priors on the unobserved states:

$$
\hat { \pmb { x } } _ { 0 } = \left[ \begin{array} { c } { m _ { 0 } } \\ { \mathbf { 0 } } \\ { \mathbf { 0 } } \end{array} \right] , \qquad \pmb { P } _ { 0 } = \left[ \begin{array} { c c c } { R _ { 0 } } & { \mathbf { 0 } } & { \mathbf { 0 } } \\ { \mathbf { 0 } } & { \sigma _ { v _ { 0 } } ^ { 2 } I _ { 3 } } & { \mathbf { 0 } } \\ { \mathbf { 0 } } & { \mathbf { 0 } } & { \sigma _ { a _ { 0 } } ^ { 2 } I _ { 3 } } \end{array} \right] ,\tag{60}
$$

with $\sigma _ { v _ { 0 } } = 5 \mathrm { m } / \mathrm { s }$ and $\sigma _ { a _ { 0 } } = 2 \mathrm { m } / \mathrm { s } ^ { 2 }$ , so that the first updates are dominated by the measurements.

5) Prediction step: At every control step, irrespective of whether a detection is available, the state and covariance are propagated with (52)–(53):

$$
\hat { \pmb { x } } _ { k | k - 1 } = \pmb { \Phi } \hat { \pmb { x } } _ { k - 1 | k - 1 } ,\tag{61}
$$

$$
P _ { k | k - 1 } = \Phi P _ { k - 1 | k - 1 } \Phi ^ { \top } + Q .\tag{62}
$$

6) Correction step: When an accepted detection is available, the predicted state is corrected by the measurement $\mathbf { \nabla } m _ { k }$ The innovation (measurement residual) and its covariance are

$$
\tilde { \pmb { y } } _ { k } = \pmb { m } _ { k } - \pmb { H } \hat { \pmb { x } } _ { k | k - 1 } ,\tag{63}
$$

$$
\begin{array} { r } { { S } _ { k } = H P _ { k | k - 1 } H ^ { \top } + R _ { k } . } \end{array}\tag{64}
$$

The Kalman gain weights the correction by the relative confidence in the prediction and the measurement,

$$
\begin{array} { r } { \pmb { K } _ { k } = \pmb { P } _ { k | k - 1 } \pmb { H } ^ { \top } \pmb { S } _ { k } ^ { - 1 } , } \end{array}\tag{65}
$$

and the state and covariance are updated as

$$
\hat { \pmb x } _ { k | k } = \hat { \pmb x } _ { k | k - 1 } + \pmb { K } _ { k } \tilde { \pmb y } _ { k } ,\tag{66}
$$

$$
P _ { k | k } = \left( I _ { 9 } - K _ { k } H \right) P _ { k | k - 1 } .\tag{67}
$$

Although only position enters the innovation (63), the offdiagonal terms of $P _ { k | k - 1 }$ built up by the dynamics in (52) make velocity and acceleration observable, so the gain (65) corrects all nine states.

7) Detection handling and coasting: When there is no detection or detection is below the confidence threshold, the correction step (63)–(67) is skipped and only the prediction step (61)–(62) executes. The filter then coasts on its motion model, with P growing through Q to reflect the increasing uncertainty until the target is reacquired.

8) Outputs: The filter exposes the estimated target state and covariance,

$$
\begin{array} { r } { \hat { p } _ { k } = \hat { \pmb x } _ { k | k } ^ { ( 1 ; 3 ) } , \quad \hat { \pmb v } _ { k } = \hat { \pmb x } _ { k | k } ^ { ( 4 ; 6 ) } , \quad \hat { \pmb a } _ { k } = \hat { \pmb x } _ { k | k } ^ { ( 7 ; 9 ) } , \quad P _ { k | k } . } \end{array}\tag{68}
$$

These are expressed relative to the interceptor and rotated into the body frame,

$$
\begin{array} { r } { { \pmb r } _ { b , k } = { \mathbf { R } } _ { i / b } \left( \hat { \pmb p } _ { k } - { \pmb p } _ { i } \right) , \qquad \dot { \pmb r } _ { b , k } = { \mathbf { R } } _ { i / b } \left( \hat { \pmb v } _ { k } - { \pmb v } _ { i } \right) , } \end{array}\tag{69}
$$

from which the LOS angles and rates used by the controllers are computed.

TABLE II  
GAINS AND CONSTANTS OF THE CASCADED ACCELERATION AUTOPILOT OF FIGURE 2. SURFACE COMMANDS ARE NORMALISED TO [−1, 1], SO GAINS ACTING ON A SURFACE ARE DIMENSIONLESS.
<table><tr><td>Symbol</td><td>Value</td><td>Unit</td><td>Description</td></tr><tr><td colspan="4">Roll-rate loop (inner, aileron)</td></tr><tr><td> $K _ { f f , p }$ </td><td>0.134</td><td> $\mathrm { s r a d } ^ { - 1 }$ </td><td>rate feedforward,  $1 / p _ { s s }$ </td></tr><tr><td> $K _ { P , p }$ </td><td>0.25</td><td> $\mathrm { s r a d } ^ { - 1 }$ </td><td>proportional gain</td></tr><tr><td> $K _ { I , p }$ </td><td>0.15</td><td> $\mathrm { r a d } ^ { - 1 }$ </td><td>integral gain</td></tr><tr><td colspan="4">Pitch-rate loop (inner, elevator)</td></tr><tr><td> $K _ { f f , q }$ </td><td>0.300</td><td> $\mathrm { s r a d } ^ { - 1 }$ </td><td>rate feedforward  $( 1 / q _ { s s } = 0 . 4 0 5 )$ </td></tr><tr><td> $K _ { P , q }$ </td><td>0.70</td><td> $\mathrm { s r a d } ^ { - 1 }$ </td><td>proportional gain</td></tr><tr><td> $K _ { I , q }$ </td><td>2.00</td><td> $\mathrm { r a d } ^ { - 1 }$ </td><td>integral gain</td></tr><tr><td colspan="4">Sideslip regulator (inner, rudder)</td></tr><tr><td> $K _ { P , \beta }$ </td><td>1.2</td><td> $\mathrm { r a d } ^ { - 1 }$ </td><td>proportional gain</td></tr><tr><td> $K _ { I , \beta }$ </td><td>0.8</td><td> $\mathrm { r a d } ^ { - 1 } \mathrm { s } ^ { - 1 }$ </td><td>integral gain</td></tr><tr><td> $K _ { D , r }$ </td><td>0.30</td><td> $\mathrm { s r a d } ^ { - 1 }$ </td><td>yaw damping about rcoord</td></tr><tr><td colspan="4">Roll channel (outer)</td></tr><tr><td> $T _ { \phi } \qquad 0 . 1 5 \qquad \mathrm { ~ s ~ }$ </td><td></td><td></td><td>roll time constant;  $P = 1 / T _ { \phi }$ </td></tr><tr><td colspan="4">Normal-acceleration channel (outer)</td></tr><tr><td> $K _ { P , a }$ </td><td>2.0</td><td> $\overline { { \mathbf { s } } } ^ { - 1 }$ </td><td>open-loop gain</td></tr><tr><td> $K _ { I , a }$ </td><td>4.0</td><td></td><td>integral gain</td></tr><tr><td colspan="4">Applying across loops</td></tr><tr><td> $V _ { \mathrm { s t a b } }$ </td><td>25.0</td><td> $\mathrm { ~ m ~ s ~ } ^ { - 1 }$ </td><td>reference for  $( V _ { \mathrm { s t a b } } / V ) ^ { 2 }$  schedule</td></tr><tr><td> $\begin{array} { l } { \tau _ { f } } \\ { f _ { \operatorname* { m i n } } } \end{array}$ </td><td>0.08</td><td>S</td><td>low-pass on specific force</td></tr><tr><td></td><td>1.47</td><td> $\mathrm { m } \mathrm { s } ^ { - 2 }$ </td><td>roll demand frozen below this</td></tr></table>

## APPENDIX D ACTUATOR LAG MODEL

Each actuator is modeled as a first-order lag between the commanded deflection $\delta _ { i , \mathrm { c m d } }$ and the achieved deflection $\delta _ { i }$

$$
\dot { \delta } _ { i } = \frac { \delta _ { i , \mathrm { c m d } } - \delta _ { i } } { \tau _ { i } } , \qquad i \in \{ e , a , r , t \} ,\tag{70}
$$

with $\tau _ { e } = \tau _ { a } = \tau _ { r } = 0 . 0 5 \mathrm { s }$ for the flight control surfaces and $\tau _ { t } = 0 . 4 0 \mathrm { s }$ for the propeller.

## APPENDIX E

## CLASSICAL GUIDANCE CASCADED AUTOPILOT GAINS

The gains of the cascaded autopilot used in the baseline classical guidance controller are given in table II. Every integrator is protected against windup by clamping its magnitude, to 0.35, 0.55 and 0.40 of full travel in the roll rate, pitch rate and sideslip loops respectively and to 30 m $\mathrm { s } ^ { - 2 }$ in the normal acceleration loop. Accumulation is suspended whenever the surface command exceeds actuator limits and the current error would drive it further beyond the limits. Integration resumes as soon as the error reverses, so recovery from a saturated maneuver is not delayed by an integrator that must first unwind.

## APPENDIX F

## REWARD FUNCTION COEFFICIENTS

Table III shows the reward function coefficients used in the training of the ultimate policy that was flown in the real flight tests.

TABLE III  
COEFFICIENTS AND PARAMETERS OF THE REWARD FUNCTION.
<table><tr><td>Symbol</td><td>Description</td><td>Value</td></tr><tr><td>Term weights</td><td></td><td></td></tr><tr><td> $c _ { d }$ </td><td>Distance-progress reward</td><td>1.0</td></tr><tr><td> $c _ { \mathrm { i n t } }$ </td><td>Interception bonus</td><td>0.5</td></tr><tr><td> $c _ { c }$ </td><td>Camera-centering reward</td><td>0.4</td></tr><tr><td> $c _ { \Phi }$ </td><td>Field-of-view penalty</td><td>1.5</td></tr><tr><td> $c _ { \Omega }$ </td><td>Body-rate penalty</td><td>0.7</td></tr><tr><td> $c _ { u }$ </td><td>Action-smoothness penalty</td><td>1.5</td></tr><tr><td> $c _ { \alpha }$ </td><td>Angle-of-attack penalty</td><td>2.0</td></tr><tr><td>Shaping parameters</td><td></td><td></td></tr><tr><td> $v _ { c l , \mathrm { s a f e } }$ </td><td>Safe closing speed</td><td>10 m/s</td></tr><tr><td> $\Omega _ { r _ { \mathrm { m a x } } }$ </td><td>Body-rate normalizer</td><td>2.0 rad/s</td></tr><tr><td> $\alpha _ { \mathrm { W a r n } }$ </td><td>AOA warning angle</td><td> $1 \dot { 0 } ^ { \circ }$ </td></tr><tr><td> $\alpha _ { \mathrm { m a x } }$ </td><td>AOA cutoff limit</td><td> $6 0 ^ { \circ }$ </td></tr><tr><td> $K$ </td><td>AOA penalty curvature</td><td> $^ { - 3 }$ </td></tr><tr><td> $\tau _ { \mathrm { m a x } }$ </td><td>Target-loss horizon</td><td>200 steps</td></tr><tr><td> $R _ { \mathrm { m a x } }$ </td><td>Return normalization budget</td><td>1000</td></tr></table>

![](images/b9fd70c2d0cc257340f24265e85106cc4429337749a3720d6b74fe557c1d7984.jpg)  
Fig. 29. A breakdown of the reasons episodes were terminated, categorized by condition. Green, purple, orange, red and grey represent the fraction of episodes terminated due to successful interception, target loss, near misses, loss of control and timeout. All conditions except for nominal are in some way beyond the training envelope, giving insight into the robustness of the policy. These variations were measured against stage 8 nominal conditions instead of stage 5 nominal conditions as in in figure 18.

## APPENDIX G

## CURRICULUM LEARNING STAGES

Tabel IV describes all the changes per stage during the training curriculum. Maximum range is deliberately lowered in the latest two stages, as it was that observed that the camera noise became too large at greater ranges to reliably detect the target.

## APPENDIX H

## AERODYNAMIC AIRCRAFT COEFFICIENTS

In table V, the aerodynamic aircraft coefficients that were used in training of the RL policy are given.

## APPENDIX I

## RL POLICY PERFORMANCE IN THE OLD TRAINING ENVIRONMENT

The performance of the RL policy in the old simulation environment (with the bug not fixed) is shown in figure 29.

<table><tr><td colspan="10">CURRICULUM SCHEDULE USED IN TRAINING THE POLICY DESCRIBING THE VARIABLES THAT WERE CHANGED FOR EACH STAGE.</td></tr><tr><td>Stage</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td></tr><tr><td>Target dynamics</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Max spawn range [m]</td><td>60</td><td>70</td><td>80</td><td>90</td><td>100</td><td>110</td><td>120</td><td>120</td><td>100</td><td>80</td></tr><tr><td>Trajectoryª</td><td>CV</td><td>CV</td><td>CA</td><td>CA</td><td>CA</td><td>CA</td><td>CA</td><td>CA</td><td>CA</td><td>CA</td></tr><tr><td>Max vel. xy [m/s]</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td></tr><tr><td>Max vel. z [m/s]</td><td>0.1</td><td>0.2</td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.7</td><td>0.8</td><td>0.9</td><td>1.0</td></tr><tr><td>Max accel. xy [m/s2]</td><td>0.0</td><td>0.1</td><td>0.2</td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.7</td><td>0.8</td><td>1.0</td></tr><tr><td>Max accel. z [m/s2]</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.1</td><td>0.1</td><td>0.2</td><td>0.2</td><td>0.3</td></tr><tr><td colspan="9">Environment &amp; domain randomization</td><td></td></tr><tr><td>Airframe DR [%]</td><td>1</td><td>2</td><td>3</td><td>5</td><td>10</td><td>15</td><td>20</td><td>20</td><td>25</td><td>30</td></tr><tr><td>Target-size rand. [%]</td><td>0</td><td>0</td><td>0</td><td>0</td><td>5</td><td>10</td><td>10</td><td>15</td><td>17</td><td>20</td></tr><tr><td>Control freq. [Hz]</td><td>44-46</td><td>43-47</td><td>42-48</td><td>41-49</td><td>40-50</td><td>40-50</td><td>40-50</td><td>40-50</td><td>40-50</td><td>40-50</td></tr><tr><td>Wind speed max [m/s]</td><td>0.2</td><td>0.4</td><td>0.6</td><td>0.8</td><td>1.0</td><td>1.3</td><td>1.7</td><td>2.3</td><td>4.5</td><td>6.0</td></tr><tr><td>Turbulence [m/s]</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.10</td><td>0.20</td><td>0.40</td><td>0.60</td></tr><tr><td>Camera realism</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Dropout mean [s]</td><td>0.20</td><td>0.25</td><td>0.30</td><td>0.35</td><td>0.40</td><td>0.45</td><td>0.50</td><td>0.60</td><td>0.70</td><td>0.70</td></tr><tr><td>Bbox bias std [-]</td><td>0.04</td><td>0.05</td><td>0.06</td><td>0.07</td><td>0.08</td><td>0.09</td><td>0.10</td><td>0.11</td><td>0.12</td><td>0.12</td></tr><tr><td>Detection latency [s]</td><td>0.02</td><td>0.02</td><td>0.02</td><td>0.04</td><td>0.04</td><td>0.04</td><td>0.06</td><td>0.06</td><td>0.06</td><td>0.06</td></tr><tr><td>Training</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Episode length [s]</td><td>20</td><td>20</td><td>25</td><td>25</td><td>30</td><td>30</td><td>35</td><td>40</td><td>45</td><td>45</td></tr><tr><td colspan="9">aCV = constant velocity; CA = constant velocity + constant acceleration.</td></tr></table>

TABLE V  
AERODYNAMIC STABILITY AND CONTROL DERIVATIVES
<table><tr><td>Category</td><td>Coefficient</td><td>Value</td></tr><tr><td rowspan="3">Lift</td><td> $\overline { { C _ { L 0 } } }$ </td><td>0.31</td></tr><tr><td> $\scriptstyle C _ { L _ { \alpha } }$ </td><td>2.90</td></tr><tr><td> $C _ { L _ { \delta _ { e } } }$ </td><td>0.50</td></tr><tr><td rowspan="2">Drag</td><td> $\overline { { C _ { D 0 } } }$ </td><td>0.12</td></tr><tr><td> $k _ { \mathrm { i n d u c e d } }$ </td><td>0.18</td></tr><tr><td rowspan="2">Side Force</td><td> $C _ { Y _ { \beta } }$ </td><td>-0.096244</td></tr><tr><td> $C _ { Y _ { \delta r } }$ </td><td>-0.18</td></tr><tr><td rowspan="4">Pitching Moment</td><td> $\overline { { C _ { m 0 } } }$ </td><td>-0.011</td></tr><tr><td> $C _ { m _ { \alpha } }$ </td><td>-0.18</td></tr><tr><td> $C _ { m _ { q } }$ </td><td>-2.60</td></tr><tr><td> $\underline { { C _ { m _ { \delta _ { e } } } } }$ </td><td>0.30</td></tr><tr><td rowspan="5">Rolling Moment</td><td> ${ C _ { l _ { \beta } } }$ </td><td>-0.0170</td></tr><tr><td> $C _ { l _ { p } } ^ { \phantom { \dagger } }$ </td><td>-0.092</td></tr><tr><td> $C _ { l _ { r } } ^ { - }$ </td><td>0.075</td></tr><tr><td> $C _ { l _ { \delta _ { a } } }$ </td><td>0.03</td></tr><tr><td> ${ C _ { l } } _ { \delta _ { r } } ^ { \mathrm { ~ \scriptsize ~ \simeq ~ } }$ </td><td>0.006506</td></tr><tr><td rowspan="5">Yawing Moment</td><td> $\overline { { C _ { n _ { \beta } } } }$ </td><td>0.02</td></tr><tr><td> $C _ { n _ { p } }$ </td><td>-0.051</td></tr><tr><td> $\bar { C _ { n _ { r } } }$ </td><td>-0.035</td></tr><tr><td> $C _ { n _ { \delta _ { a } } }$ </td><td>0.007</td></tr><tr><td> $C _ { n _ { \delta _ { r } } }$ </td><td>0.028</td></tr></table>

## ACKNOWLEDGMENT

This work was funded by TU Delft and Dremian BV. The latter provided the computer vision algorithm used to detect real targets in real flight experiments.

## REFERENCES

[1] Denisa Kunertova. The war in ukraine shows the gamechanging effect of drones depends on the game. Bulletin of the Atomic Scientists, 79(2):95–102, 2023. doi: 10. 1080/00963402.2023.2178180.

[2] Olena Bilousova, Nataliia Shapoval, and Vladyslav Vlasiuk. Strengthening sanctions on foreign components in russian military drones. Working group paper #13, The International Working Group on Russian Sanctions, August 23 2023. URL https://fsi.stanford.edu/ working-group-sanctions.

[3] N.A. Shneydor. Missile Guidance and Pursuit: Kinematics, Dynamics and Control. Horwood engineering science series. Woodhead Publishing, 1998. ISBN 9781782420590. URL https://books.google.nl/books?id= QUykAgAAQBAJ.

[4] Rafael Yanushevsky. Modern Missile Guidance. CRC Press, Taylor & Francis Group, Boca Raton, FL, USA, 2 edition, 2018. ISBN 9780815384861.

[5] Paul Zarchan. Tactical and Strategic Missile Guidance, volume 239 of Progress in Astronautics and Aeronautics. American Institute of Aeronautics and Astronautics, Reston, VA, 6 edition, 2012. ISBN 9781600868948.

[6] Elia Kaufmann, Leonard Bauersfeld, Antonio Loquercio, Matthias Muller, Vladlen Koltun, and Davide Scara-¨ muzza. Champion-level drone racing using deep reinforcement learning. Nature, 620:982–987, 2023. doi: 10.1038/s41586-023-06419-4.

[7] Jagadeswara PKV Pothuri, Aditya Bhatt, Prajit KrisshnaKumar, Manaswin Oddiraju, and Souma Chowdhury.

Learning approach to efficient vision-based active tracking of a flying target by an unmanned aerial vehicle, 2025. URL https://arxiv.org/abs/2506.18264.

[8] Carol Mart´ınez, Thomas Richardson, Peter Thomas, Jonathan Luke du Bois, and Pascual Campoy. A visionbased strategy for autonomous aerial refueling tasks. Robotics and Autonomous Systems, 61(8):876–895, 2013. ISSN 0921-8890. doi: 10.1016/j.robot.2013.02.006.

[9] Shuvrangshu Jana, Lima Agnel Tony, Aashay Anil Bhise, V. P. Varun, and Debasish Ghose. Interception of an aerial manoeuvring target using monocular vision. Robotica, 40(12):4535–4554, 2022. doi: 10.1017/S0263574722001096. URL https://doi.org/10. 1017/S0263574722001096.

[10] T. P. Patel, E. L. Tevere, E. H. Kramer, and R. M. Mukherjee. A general purpose method for robotic interception of non-cooperative dynamic targets. In 2026 IEEE Aerospace Conference, pages 1–11, Big Sky, MT, USA, 2026. IEEE. doi: 10.1109/AERO66936.2026. 11520048.

[11] Clemente van der Aa. Design of an image-based visual servoing system for autonomous quadcopter interceptor. Msc thesis, Delft University of Technology, 2024. URL https://resolver.tudelft.nl/uuid: 4d8f5709-c20e-4ed2-9f62-26ce3ecde223.

[12] Antonella Barisic, Marko Car, and Stjepan Bogdan. Vision-based system for a real-time detection and following of uav. In 2019 Workshop on Research, Education and Development of Unmanned Aerial Systems (RED UAS), page 156–159. IEEE, November 2019. doi: 10.1109/reduas47371.2019.8999675. URL http://dx.doi. org/10.1109/REDUAS47371.2019.8999675.

[13] Jaemin Kim, Youngrun Kim, SuHyeon Kim, Hyeongjun Cho, and Dongwon Jung. Vision-based geolocation of moving ground targets using kalman filtering with a gimbal camera on board a uav. Aerospace, 12(12), 2025. ISSN 2226-4310. doi: 10.3390/aerospace12121065. URL https://www.mdpi.com/2226-4310/12/12/1065.

[14] Hailong Yan, Kun Yang, Yixiao Cheng, Zihao Wang, and Dawei Li. Precise interception flight targets by imagebased visual servoing of multicopter. IEEE Transactions on Industrial Electronics, 72(11):11499–11509, 2025. doi: 10.1109/TIE.2025.3559951.

[15] Jemin Hwangbo, Inkyu Sa, Roland Siegwart, and Marco Hutter. Control of a quadrotor with reinforcement learning. IEEE Robotics and Automation Letters, 2(4):2096– 2103, Oct 2017. doi: 10.1109/LRA.2017.2720851.

[16] William Koch, Renato Mancuso, Richard West, and Azer Bestavros. Reinforcement learning for uav attitude control. ACM Trans. Cyber-Phys. Syst., 3(2), February 2019. ISSN 2378-962X. doi: 10.1145/3301273. URL https://doi.org/10.1145/3301273.

[17] Elia Kaufmann, Antonio Loquercio, Rene Ranftl,´ Matthias Muller, Vladlen Koltun, and Davide Scara-¨ muzza. Deep drone acrobatics. CoRR, abs/2006.05768, 2020. URL https://arxiv.org/abs/2006.05768.

[18] Stavrow A. Bahnam, Robin Ferede, Till M. Blaha, Anton E. Lang, Erin Lucassen, Quentin Missinne, Aderik

E. C. Verraest, Christophe De Wagter, and Guido C. H. E. de Croon. Monorace: Winning champion-level drone racing with robust monocular ai, 2026. URL https://arxiv.org/abs/2601.15222.

[19] Maxime Capelle. Beyond proportional navigation: Deep reinforcement learning for robust drone interception. Msc thesis, Delft University of Technology, 2025. URL https://resolver.tudelft.nl/uuid: 22d13eff-4198-4d4b-927a-01edaddd990a.

[20] Francisco Giral, Ignacio Gomez, and Soledad Le Clainche. Intercepting unauthorized aerial robots in controlled airspace using reinforcement learning. Aerospace Science and Technology, 168:110985, 2026. ISSN 1270-9638. doi: https://doi.org/10.1016/j.ast.2025. 110985. URL https://www.sciencedirect.com/science/ article/pii/S127096382501048X.

[21] Robin Ferede, Till Blaha, Erin Lucassen, Christophe De Wagter, and Guido C.H.E. de Croon. One net to rule them all: Domain randomization in quadcopter racing across different platforms. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 6357–6363, 2025. doi: 10.1109/ICRA55743.2025. 11128790.

[22] Haochen Chai, Meimei Su, Yang Lyu, Zhunga Liu, Chunhui Zhao, and Quan Pan. Monocular obstacle avoidance based on inverse ppo for fixed-wing uavs. https://arxiv.org/abs/2411.18009, 2024. arXiv preprint arXiv:2411.18009 [cs.RO].

[23] Dennis J. Marquis, Blake Wilhelm, Devaprakash Muniraj, and Mazen Farhood. Adversarial reinforcement learning for robust control of fixed-wing aircraft under model uncertainty. https://arxiv.org/pdf/2510.16650, 2025. arXiv preprint arXiv:2510.16650 [cs.RO].

[24] Xing Zhuang, Dongguang Li, Yue Wang, Xinyu Liu, and Hanyu Li. Optimization of high-speed fixed-wing uav penetration strategy based on deep reinforcement learning. Aerospace Science and Technology, 148: 109089, 2024. ISSN 1270-9638. doi: 10.1016/j.ast.2024. 109089. URL https://www.sciencedirect.com/science/ article/pii/S1270963824002220.

[25] Eivind Bøhn, Erlend M. Coates, Dirk Reinhardt, and Tor Arne Johansen. Data-efficient deep reinforcement learning for attitude control of fixed-wing uavs: Field experiments. IEEE Transactions on Neural Networks and Learning Systems, 35(3):3168–3180, March 2024. ISSN 2162-2388. doi: 10.1109/tnnls.2023.3263430. URL http://dx.doi.org/10.1109/TNNLS.2023.3263430.

[26] L. J. Fletcher, R. J. Clarke, T. S. Richardson, and M. Hansen. Reinforcement learning for a perched landing in the presence of wind. In AIAA Scitech 2021 Forum, page 1282. AIAA, 2021.

[27] Leslie Pack Kaelbling, Michael L. Littman, and Anthony R. Cassandra. Planning and acting in partially observable stochastic domains. Artificial Intelligence, 101(1):99–134, 1998. ISSN 0004- 3702. doi: https://doi.org/10.1016/S0004-3702(98) 00023-X. URL https://www.sciencedirect.com/science/ article/pii/S000437029800023X.

[28] Brian L. Stevens, Frank L. Lewis, and Eric N. Johnson. Aircraft Control and Simulation: Dynamics, Controls Design, and Autonomous Systems. Wiley, 3 edition, 2015. ISBN 978-1118870983.

[29] M. V. Cook. Principles in Flight Dynamics. Edward Arnold, London, 1997. ISBN 0-340-63200-3.

[30] Vladislav. Klein and Eugene A. Morelli. Aircraft system identification theory and practice, 2006.

[31] Jan Krejcˇ´ı, Oliver Kost, Yuxuan Xia, Lennart Svensson, and Ondˇrej Straka. Model-based multi-object visual tracking: Identification and standard model limitations. In 2025 28th International Conference on Information Fusion (FUSION), page 1–8. IEEE, July 2025. doi: 10.23919/fusion65864.2025.11124146. URL http://dx. doi.org/10.23919/FUSION65864.2025.11124146.

[32] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. 2017. URL https://arxiv.org/abs/1707.06347.

[33] Antonin Raffin, Ashley Hill, Adam Gleave, Antti Kanervisto, Maximilian Ernestus, and Noah Dormann. Stable-baselines3: Reliable reinforcement learning implementations. Journal of Machine Learning Research, 22 (268):1–8, 2021.

[34] James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. Jax: composable transformations of python+numpy programs. https://github.com/jax-ml/jax, 2018. Accessed: 2026-04- 21.

[35] Siu Kwan Lam, Antoine Pitrou, and Stanley Seibert. Numba: A llvm-based python jit compiler. In Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC, LLVM ’15, New York, NY, USA, 2015. Association for Computing Machinery. doi: 10.1145/2833157.2833162. URL https://doi.org/10.1145/ 2833157.2833162.