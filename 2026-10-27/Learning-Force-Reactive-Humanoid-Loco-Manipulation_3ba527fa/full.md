# Learning Force-Reactive Humanoid Loco-Manipulation

Sihang Li<sup>1,2,§</sup>, Xiaofeng Guo<sup>1</sup>, Jinkun Cao<sup>1</sup>, Charlie Cheng<sup>1</sup>, Ruihan Yang<sup>1</sup>, Chen Feng<sup>1,2,†</sup>, Guanya Shi<sup>1,3,†</sup>, Haozhi Qi<sup>1,4,†</sup>

<sup>1</sup>Amazon FAR (Frontier AI & Robotics) <sup>2</sup>New York University <sup>3</sup>Carnegie Mellon University <sup>4</sup>University of Chicago

![](images/9f0c54a2c5f957bfad7d79b5b34e8a7011f1e99b2af9fdf83a7e783f77f61602.jpg)  
Fig. 1. FLORE is a framework that learns force-reactive loco-manipulation from robot-free demonstrations. (a) Two handheld UMI-style grippers carrying 6-axis F/T sensors and three trackers record hand/torso motion, gripper width and contact forces. From these demonstrations, we train a forcereactive diffusion policy that predicts per-hand force commands. A unified force-position whole-body controller then executes the motion and force together, enabling the following capabilities: (b) robustness to force variation: the robot walks a heavy wheeled gantry forward with both hands; (c) bimanual coordination: the robot rotates a heavy box (12 kg) while dragging it across the table without tipping or lifting it; (d) sustained contact: one hand presses an eraser against a whiteboard and holds a steady normal force through the stroke. The overlaid curves trace the hand keypoints and the arrows mark the commanded force at its peak; the traces below show the commanded (dashed) and measured (solid) force per hand and axis.

Abstract— Many everyday tasks require controlling contact force as well as motion. On a humanoid, this force must often be coordinated across both hands and the whole body, and cannot be specified by kinematic trajectories alone. Therefore, we present FLORE, a framework for learning forcereactive humanoid loco-manipulation. We collect robot-free demonstrations using two force-sensing UMI-style handheld grippers for hand motion and contact wrenches, and wearable trackers for body motion. A force-reactive diffusion policy takes vision, proprioception, and measured wrenches as input, then predicts actions and per-hand force commands. A hybrid compliance whole-body controller with force tracking executes these commands while maintaining balance: an integral of the force error measured by real force/torque (F/T) sensors lets each hand track force accurately along the commanded direction and remain compliant along the others. In simulation, the controller tracks per-hand force commands of varied magnitude and direction, with little loss in whole-body motion accuracy. Across three contact-rich tasks on the robot, the force-reactive policy succeeds in 80% of rollouts, compared with 26.7% for a position-only policy trained on the same demonstrations and executed by the same controller. For more details, please visit our website: https://louis-leee.github.io/FLoRe/.

## I. INTRODUCTION

Humanoid robots offer a general-purpose embodiment for working in environments built for people. Many tasks in these environments, such as pushing a cart, moving a heavy box, and wiping a surface, require the robot to control contact force as well as motion. However, learned humanoid locomanipulation is commonly formulated as generating and tracking motion references [1]–[3]. Tracking a demonstrated hand trajectory alone does not reproduce the demonstrated force, which also depends on contact geometry, stiffness, and object motion. Unlike a fixed-base arm, a humanoid must stay balanced under these contact loads, coordinating its hands, torso, and legs while regulating force.

Two difficulties stand in the way. The first is data: imitation learning requires scalable demonstrations that record contact force alongside motion. Existing robot-free interfaces for whole-body demonstration collection record motion but not force [4], [5]. The second is execution: the robot must accurately apply the desired force. Current learned humanoid whole-body controllers track motion alone, or treat force as a disturbance to reject or a stiffness to lower [6], [7].

Tracking a commanded contact force has been shown only on quadrupedal manipulators, using estimated rather than measured force [8], [9], which is not accurate enough. A humanoid instead needs to track force at both hands together with whole-body motion while maintaining balance.

We present FLORE, a framework for learning forcereactive humanoid loco-manipulation. Its robot-free forceannotated data collection system records contact force with each demonstration; the policy learns to predict this force, and the controller tracks it. The system pairs two UMI-style handheld grippers [10] with a chest tracker and two ankle trackers [4], [11]. Each gripper carries a wrist camera, a pose tracker, and a six-axis F/T sensor [12], [13]. The applied wrench is thus recorded alongside hand, torso, and foot motion. The robot uses the same gripper–sensor assembly.

For the high-level planner, we propose a force-reactive diffusion policy [14]. It observes wrist images, robot state, and the measured wrench at each hand, then predicts a chunk [15] of action, gripper widths, and per-hand desired force. Force enters as both an input and an output. The input lets the policy react to interactions that images do not reveal. The output is commanded to the controller [13], rather than used only as an auxiliary target [16]. This connects the force measured during demonstration to the deployed force.

To track the commanded forces, we develop a wholebody controller with hybrid compliance. It outputs only joint positions, so each hand’s force command becomes a shift of its position target. In contact, an integral of the force error measured at the wrist adjusts this shift until the hand exerts the commanded force, whatever the contact stiffness. Along uncommanded axes, the same integral lets the hand yield to external pushes. Each hand thus tracks force along the commanded direction and complies along the others, without switching modes. We train the controller in simulation [17] and deploy it on the robot using wrist F/T measurements.

We deploy FLORE on a Unitree G1 and evaluate three whole-body contact-rich tasks: pushing a gantry hanging another robot, bimanual box rotation, and board wiping. In simulation, the controller realises in-contact force error of about 1 N per hand while tracking whole-body motion. On the real robot, FLORE succeeds in 80% of rollouts, compared with 26.7% for a position-only policy trained on the same demonstrations and executed by the same controller with force commands disabled. Withholding only the force command reduces success to 36.7%, while removing only the force observation reduces it to 66.7%. Both contribute to the final performance. Our key contributions are:

• A robot-free pipeline for learning force-reactive humanoid loco-manipulation from demonstrations that record both motion and contact wrench.

• A learned whole-body controller that tracks per-hand force together with keypoint motion while remaining compliant to external forces and maintaining balance.

• Evaluation in simulation and on three contact-rich tasks on a Unitree G1, where force commands and force observations both improve task success over position-only control.

## II. RELATED WORK

A. Force Sensing for Loco-Manipulation Data Collection and Policy Learning

Robot-free data collection builds on UMI [10]. Forceaware variants add wrist or fingertip sensing, but deploy on fixed-base arms with analytic impedance or admittance control [12], [13], [18]. Humanoid extensions instead collect whole-body motion and execute it with learned trackers, but do not record contact force [4], [11]. WT-UMI is the closest interface for humanoids, although it measures only a scalar normal force at each palm and relies on paired teleoperation data [19]. Our interface records a six-axis wrench at each hand together with torso motion, and its demonstrations require no robot-side data.

Measured force has been used to condition visuomotor policies [20], [21]. Other work predicts stiffness, admittance gains, or virtual position targets that are executed by an impedance controller [12], [22]–[24]. More closely related methods predict a wrench trajectory together with pose, but execute it on fixed-base arms through analytic force controllers [13], [25], [26]. On humanoids, WT-UMI passes a scalar normal-force prediction to an admittance rule [19]. UniFP is the closest formulation. It predicts 3-D force commands, but deploys end-effector force control only on a quadrupedal manipulator. Its humanoid interface exposes force only at the base [9]. Our planner instead predicts perhand pose, gripper, and wrench trajectories from handheld demonstrations, then sends the predicted force to a learned humanoid whole-body controller.

## B. Force for Low-Level Control

Learned legged and humanoid controllers usually either withstand external force or yield to it through commanded stiffness or damping [6], [7], [27]–[30], but neither approach directly specifies how hard to press. Direct force tracking has mainly been studied on quadrupedal manipulators, as a separate control mode or through a target-shifting spring law, both using force estimated from proprioception [8], [9]. WT-UMI regulates a scalar palm force on a humanoid, but through an analytic admittance rule and inverse kinematics above a separate locomotion controller [19]. In contrast, our hybrid compliance whole-body controller with force tracking integrates the force error measured at each wrist. One learned policy tracks position in free space and force along the commanded direction, while remaining compliant along the other directions and maintaining balance.

## III. METHOD

In FLORE (Fig. 2), a hybrid compliance whole-body controller with force tracking executes motion references and per-hand force commands (Sec. III-A); a robot-free forceannotated data collection system records vision, motion, and contact wrenches with handheld devices (Sec. III-B); and a force-reactive diffusion policy, trained on these demonstrations, predicts keypoint motion and force commands for the controller (Sec. III-C).

![](images/c9ffb4ac5a40ca4f3568a7943a1476255f108d822f79dd4576e525a5fe23c250.jpg)  
Fig. 2. Overview of our FLORE framework. (B) Two handheld UMI-style grippers with F/T sensors and three additional trackers record images, motion, and contact force without a robot. (C) A force-reactive diffusion policy takes images, state and measured force as input and outputs keypoint motion and desired force commands at 30 Hz. (A) The hybrid compliance whole-body controller with force tracking converts the force command into a shift of the target along the force direction, so that one RL policy tracks motion and force simultaneously while remaining compliant along other directions.

Notation. The robot is a humanoid with a parallel-jaw gripper on each hand. The controller tracks keypoints $k \in \mathcal { K }$ (both hand tool center point (TCP), the torso, and both feet) with poses $\mathbf { T } _ { k } \ = \ \left( \mathbf { p } _ { k } , \mathbf { R } _ { k } \right)$ expressed in the pelvis frame. For each hand $h \in \mathcal { H } , \mathbf { f } _ { h } ^ { \mathrm { c m d } }$ is the force the hand should exert on the environment, f<sup>meas</sup> the reaction force measured by its wrist F/T sensor, and $\mathbf { f } _ { h } ^ { \mathrm { e x t } }$ the true contact force available only in simulation; all are expressed in the pelvis frame and act at the TCP, so that perfect tracking gives $\mathbf { f } _ { h } ^ { \mathrm { m e a s } } = - \mathbf { f } _ { h } ^ { \mathrm { c m d } }$ . The high-level policy predicts chunks of H steps; the low-level controller runs at a higher rate and sends joint-position targets to joint PD controllers. Hardware, rates, and all numerical settings are given in Sec. IV-A.

## A. Hybrid Compliance Whole-body Controller with Force Tracking Capability

The controller is a single learned policy that tracks wholebody motion together with a commanded force at each hand. We describe its control law, the contact model and simulated sensor it is trained with, and its training objective.

Control law and unified reference generator. The controller outputs only joint positions and is rewarded only on keypoint poses, so a force command must become a position target. For each hand h, a unified reference generator combines the motion reference $\mathbf { p } _ { h } ^ { * }$ and the force command $\mathbf { f } _ { h } ^ { \mathrm { c m d } }$ into a single target (see Fig. 3):

$$
\begin{array} { r l } & { \mathbf { p } _ { h } ^ { \mathrm { t g t } } = \underbrace { \mathbf { p } _ { h } ^ { * } } _ { \begin{array} { l } { \mathrm { m o t i o n ~ r e f e r e n c e } } \\ { \mathbf { I } _ { h } \gets c _ { h } \mathrm { c l i p } \left( \mathbf { I } _ { h } + K _ { I } \underbrace { ( \mathbf { f } _ { h } ^ { \mathrm { c m d } } + \mathbf { f } _ { h } ^ { \mathrm { m e a s } } ) } _ { \mathrm { f o r c e ~ e r r o r } } \right) } \end{array} } , } \\ & { ~ \mathbf { I } _ { h } \gets c _ { h } \mathrm { c l i p } \left( \mathbf { I } _ { h } + K _ { I } \underbrace { ( \mathbf { f } _ { h } ^ { \mathrm { c m d } } + \mathbf { f } _ { h } ^ { \mathrm { m e a s } } ) } _ { \mathrm { f o r c e ~ e r r o r } } \Delta t , \pm I _ { \mathrm { m a x } } \right) , } \end{array}\tag{1}
$$

with nominal stiffness $K ,$ , integral gain $K _ { I }$ , bound $I _ { \mathrm { m a x } }$ and time step $\Delta t ; \Delta \mathbf { x } _ { h } = ( \mathbf { f } _ { h } ^ { \mathrm { c m d } } + \mathbf { I } _ { h } ) / K$ is the total shift. The command moves the target $\mathbf { f } _ { h } ^ { \mathrm { c m d } } / K$ along the force direction; where the target lies inside the surface, the hand cannot follow it and presses instead, and tracking means ${ \bf f } _ { h } ^ { \mathrm { m e a s } } = - { \bf f } _ { h } ^ { \mathrm { c m d } }$ . How hard a given shift presses depends on the contact stiffness, which the robot does not know. The integral of the force error therefore moves the target further in, or back, until the measured force balances the command, independent of K and $K _ { \mathrm { e n v } } ;$ unlike the spring law of UniFP [9], which shifts the target in proportion to an estimated contact force, the shift is thus closed on the measured force. The soft contact phase $c _ { h } \in [ 0 , 1 ]$ , computed from the measured force magnitude with onset and release hysteresis thresholds and a first-order low-pass filter, lets the integral build up only in contact and clears it on release, while the clip bounds windup. It requires no privileged contact information, and also enters the observation and weights the force-related rewards. $\mathbf { p } _ { h } ^ { \mathrm { t g t } }$ replaces the hand target in the tracking reward, and ${ \mathbf I } _ { h }$ and $\Delta { \bf x } _ { h }$ enter the observation; the joint command is untouched, so the policy remains the sole executor.

![](images/07a01abcd81b11c723da6becbd65a63d8b84396e5fc2c18615d1328407e67d33.jpg)  
Fig. 3. Control law of the hybrid compliance controller. Along the force direction, the target $ { \mathbf { p } } _ { h } ^ { \mathrm { t g t } }$ is shifted from the motion reference $\mathbf { p } _ { h } ^ { * }$ into the surface by the force command and the force feedback, until the measured force $\mathbf { f } _ { h } ^ { \mathrm { m e a s } }$ balances the command. Along the compliance direction, the gripper yields to $\mathbf { f } _ { h } ^ { \mathrm { e x t } }$ (dashed gripper).

The law acts on every axis of each hand, and a measured force need not lie along the command. Dropping the hand index, let a force be commanded along x and none along z.

![](images/aa3d3f3bd8be7c8b549ba626e4f1f46dfa73330982200bbd77641980d3a40463.jpg)  
Fig. 4. Robot-free data-collection interface. Left: the demonstrator holds a handheld device in each hand and wears trackers on the chest and both ankles. Right: each device carries a VIVE tracker, a RealSense D405 wrist camera, and an F/T sensor mounted between the handle and gripper.

A push along z is then measured too, $f _ { z } ^ { \mathrm { m e a s } } = f _ { z } ^ { \mathrm { e x t } }$

$$
\begin{array} { l } { { x : ~ f _ { x } ^ { \mathrm { c m d } } \not = 0 \Rightarrow f _ { x } ^ { \mathrm { m e a s } }  - f _ { x } ^ { \mathrm { c m d } } ~ ( \mathrm { f o r c e ~ t r a c k i n g } ) , } } \\ { { z : ~ f _ { z } ^ { \mathrm { c m d } } = 0 \Rightarrow \dot { I } _ { z } = K _ { I } f _ { z } ^ { \mathrm { m e a s } } = K _ { I } f _ { z } ^ { \mathrm { e x t } } ~ ( \mathrm { c o m p l i a n c e } ) . } } \end{array}
$$

Along z the integral grows with the push and moves the target with it at a rate $K _ { I } f _ { z } ^ { \mathrm { e x t } } / K$ , so the hand gives way instead of resisting, for as long as the push lasts and within the bound $I _ { \mathrm { m a x } } .$ . Each hand thus tracks force along the commanded direction and remains compliant along the others, with no switching between them.

Contact model and simulated sensor. In training each hand faces a virtual compliant plane with outward normal ${ \mathbf { n } } _ { h }$ and randomized stiffness $K _ { \mathrm { e n v } }$ that reacts only to penetration,

$$
\mathbf { f } _ { h } ^ { \mathrm { e x t } } = K _ { \mathrm { e n v } } \operatorname* { m a x } \bigl ( 0 , \ \mathbf { n } _ { h } ^ { \top } ( \mathbf { x } _ { h } ^ { \mathrm { w a l l } } - \mathbf { p } _ { h } ) \bigr ) \mathbf { n } _ { h } ,\tag{2}
$$

applied at the TCP, with penetration capped at $d _ { \mathrm { m a x } }$ and the force capped as well. The policy observes a simulated F/T reading instead of $\mathbf { f } _ { h } ^ { \mathrm { e x t } }$

$$
\mathbf { f } _ { h } ^ { \mathrm { m e a s } } = \mathbf { f } _ { h } ^ { \mathrm { e x t } } + m _ { d } \ddot { \mathbf { p } } _ { h } ^ { \mathrm { w r i s t } } + \mathbf { b } _ { h } + \epsilon _ { h } ,\tag{3}
$$

with randomized distal mass $m _ { d } .$ , per-episode bias $\mathbf { b } _ { h }$ and white noise $\epsilon _ { h }$ . At each reset the plane is placed at a random distance beyond the hand’s reset position with a random orientation, and $\mathbf { f } _ { h } ^ { \mathrm { c m d } }$ follows trapezoidal ramp–hold–release pulses with random peaks, clamped to what the plane can deliver, $K _ { \mathrm { e n v } } d _ { \mathrm { m a x } }$ . The motion reference is the unmodified clip; where it conflicts with the force, the shifted target of (1) wins. The plane is frictionless, so training commands are purely normal; on the robot the same per-axis law applies along tangential axes as above, although tangential loads are never seen in training.

Training objective. The reward extends the motion-tracking reward of [1], $r _ { \mathrm { t r a c k } }$ , with terms for the hands:

$$
\begin{array} { r } { r = r _ { \mathrm { t r a c k } } + r _ { \mathrm { p o s } } + r _ { \mathrm { t c p } } + r _ { \mathrm { o r n } } + r _ { \mathrm { f o r c e } } } \\ { + r _ { \mathrm { c o n t a c t } } + r _ { \mathrm { a p p r o a c h } } - r _ { \mathrm { r a t e } } . } \end{array}\tag{4}
$$

All tracking terms use the shifted target of (1): $r _ { \mathrm { p o s } } ,$ r<sub>tcp</sub> and $r _ { \mathrm { o r n } }$ are exponential kernels on the hand-position, TCPposition and hand-orientation errors, the latter two constraining the wrist-to-TCP offset. Weighted by $c _ { h }$ while a force is commanded, $r _ { \mathrm { f o r c e } }$ scores the force error against the clean simulated force and $r _ { \mathrm { c o n t a c t } }$ rewards contact below the force cap; weighted by $1 - c _ { h } , r _ { \mathrm { a p p r o a c h } } = \exp ( - d _ { h } / d _ { 0 } )$ pulls the hand toward the plane, with $d _ { h }$ its distance and $d _ { 0 }$ a length scale. $r _ { \mathrm { r a t e } }$ penalizes changes in contact force between steps and softens impacts.

## B. Force-annotated Robot-free Data Collection System

Device. Each handheld UMI-style gripper [10] carries an F/T sensor<sup>1</sup> between handle and gripper, a wrist camera, a pose tracker, and a jaw-width encoder; the demonstrator holds one in each hand and wears trackers on the chest and both ankles (Fig. 4). The robot carries the same gripper–sensor assembly on each wrist, so sensor frame, sensor-to-TCP transform and gripper geometry are identical in collection and deployment. The wrench is kept in the sensor frame, gravity-compensated with a per-device payload and tared at the start of each recording; on the robot, inertial loads are also compensated. Streams and synchronization. Images, tracker poses, wrenches, and gripper widths arrive at different rates and are stamped on arrival with one host clock; the labels below are built from the raw streams, not the camera grid. Missing wrench readings and tare transients are masked rather than zero-filled, since a tared sensor also reads zero in free space.

## C. Force-Reactive High-level Policy Learning

The high-level policy $\pi _ { \theta }$ is a diffusion policy [14] that maps the current observation to a chunk of H future actions [15]. We augment both its inputs and outputs with perhand contact wrenches.

Observation. The observation $\mathbf { o } _ { t }$ contains the wrist images of both hands, $\mathbf { I } _ { t } ^ { L }$ and $\mathbf { I } _ { t } ^ { R }$ , the measured wrenches $\{ \mathbf { w } _ { h , t } ^ { \mathrm { m e a s } } \} _ { h \in \mathcal { H } }$ , and a low-dimensional state:

$$
\begin{array} { r l } & { \mathbf { s } _ { t } = \left[ ( \mathbf { p } _ { H _ { L } } - \mathbf { p } _ { H _ { R } } ) _ { x y } , \ ( \mathbf { p } _ { H _ { L } } - \mathbf { p } _ { T } ) _ { x y } , \ ( \mathbf { p } _ { H _ { R } } - \mathbf { p } _ { T } ) _ { x y } , \right. } \\ & { \quad \quad \quad \left. \mathrm { r p y } ( \mathbf { R } _ { H _ { L } } ) , \ \mathrm { r p y } ( \mathbf { R } _ { H _ { R } } ) , \ g _ { H _ { L } } , \ g _ { H _ { R } } \right] . } \end{array}\tag{5}
$$

The state contains pairwise horizontal position differences among the hands and torso, hand orientations in the pelvis frame, and gripper widths $g _ { h }$ . Here, $( \cdot ) _ { x y }$ selects the horizontal components, and rpy(R) returns the extrinsic-XYZ Euler angles (roll, pitch, yaw) of R in radians. Using position differences makes the state invariant to absolute translation. We omit absolute poses and joint angles because they differ between the completed demonstration motion and the robot. Action. Each step $j$ of the chunk stacks the pose, gripper width and force command $\mathbf { f } _ { h } ^ { \mathrm { c m d } }$ of each hand, together with the poses of the torso and both feet:

$$
\begin{array} { r } { \begin{array} { r l } & { \mathbf { a } _ { t + j } = [ \{ \tilde { \mathbf { T } } _ { h } , g _ { h } , \mathbf { f } _ { h } ^ { \mathrm { c m d } } \} _ { h \in \mathcal { H } } , \{ \tilde { \mathbf { T } } _ { k } \} _ { k \in K \backslash \mathcal { H } } ] , } \\ & { \quad \tilde { \mathbf { T } } _ { k } = \mathbf { T } _ { k , t } ^ { - 1 } \mathbf { T } _ { k , t + j } . } \end{array} } \end{array}\tag{6}
$$

Poses are anchor-relative, in each keypoint’s own frame at the observation time, so the chunk is invariant to the world frame and to where the demonstrator stood [4], [11]; rotations use the continuous rotation representation of [31].

Model and training. Each image is encoded by a CLIPinitialized vision transformer [32], the state by a linear layer, and each hand’s wrench history by a small causal convolutional encoder, in which a learned embedding replaces masked windows. These condition a temporal U-Net [33] denoiser through FiLM [34], trained with the DDPM objective [35] and sampled with DDIM [36]. Actions are quantile-normalized per column and the loss is a masked Mean Squared Error (MSE) Loss.

Closing the loop. At each policy step the chunk is decoded into absolute pelvis-frame keypoint targets, re-based onto the currently commanded keypoints so that a re-plan never jumps, and streamed to the controller either sequentially (execute N steps, then re-plan) or asynchronously with temporal ensembling [15], [37]. The predicted force of each hand becomes its $\mathbf { f } _ { h } ^ { \mathrm { c m d } }$ (Sec. III-A).

## IV. EXPERIMENTS

We evaluate whether a humanoid can learn force-reactive loco-manipulation from robot-free demonstrations. Experiments in simulation and on the robot test four capabilities:

• Whole-body force tracking: tracking per-hand force com mands and whole-body motion with one learned controller.

• Sustained contact: making contact and holding an appropriate force while the hands move with or along the object.

• Bimanual and whole-body coordination: applying force with both hands while the torso and legs balance and walk.

• Robustness to force variation: completing tasks when resistance changes within or across episodes but is not visible in the images, e.g., a heavier load or box mass.

We first compare our hybrid compliance whole-body controller with force tracking (Sec. III-A) to three baselines:

• SONIC [3]: a large-scale whole-body motion tracker, run from its released checkpoint. It tracks motion references and accepts no force command.

• SONIC + admittance: SONIC followed by an analytic force servo that adds $\alpha \mathbf { J } _ { h } ^ { \top } ( \mathbf { f } _ { h } ^ { \mathrm { c m d } } + \mathbf { f } _ { h } ^ { \mathrm { m e a s } } )$ to its arm joint targets, a correction that SONIC does not observe.

$\mathrm { F L o R E } \quad \mathrm { w / o }$ Force: our controller trained with the force channel removed $( \mathbf { f } ^ { \mathrm { c m d } } \equiv 0 ;$ , no $\mathbf { I } _ { h } , \Delta \mathbf { x } _ { h } , c _ { h }$ observed).

The second comparison isolates how the high-level policy uses force. FLORE observes the measured wrench (F-in), learns to predict the demonstrated force (F-out), and sends this prediction to the controller (F-cmd). w/o F-cmd and w/o F-in remove one pathway each, and DP all three:

• w/o F-cmd: the FLORE policy with its predicted force withheld $( { \bf f } ^ { \mathrm { c m d } } \equiv 0 )$ , which turns off the force channel $( c _ { h } , \mathbf { I } _ { h } , \Delta \mathbf { x } _ { h } \equiv 0 )$ so that the controller tracks pose only.

• w/o F-in: the FLORE policy trained without the measured wrench in its observation. It still commands force but cannot sense whether contact is made or lost.

![](images/9f07193603ff0b7a7fd69e32ba4fdfdc421309ab2cc1718dec93aed68cfee2cc.jpg)  
Fig. 5. Hybrid compliance whole-body controller with force tracking. Top: in simulation, each hand presses a randomly placed virtual plane while the robot tracks the reference motion and maintains balance. The arrows show commanded force (pink), measured reaction force (blue), external force (yellow), and force perturbation (red). Bottom: on the real robot, a person pulls both hands, which yield instead of resisting.

• DP: the position-only diffusion policy of UMI-style humanoid systems [4], [11], trained with the same architecture, data and schedule as FLORE but without force input or output, and executed by our controller with $\mathbf { f } ^ { \mathrm { c m d } } \equiv 0$

We train one policy per task and evaluate each in 10 rollouts, with test conditions paired across policies. We report the success rate, compared using Fisher’s exact test, and the offline validation error of the predicted pose, force and torque on held-out demonstrations.

## A. Implementation Details

Hybrid Compliance Whole-body Controller. The reference generator uses K = 200 N/m, $\begin{array} { c c l } { K _ { I } } & { = } & { 0 . 5 \mathrm { s } ^ { - 1 } } \end{array}$ and $I _ { \mathrm { m a x } } \ = \ 4 0 \mathrm { N } ;$ the soft contact phase uses hysteresis thresholds of 2.0 N for onset and 1.5 N for release and a 50 ms low-pass time constant. We sample $K _ { \mathrm { e n v } } \ \sim$ LogUniform(100, 800) N/m and cap penetration at $d _ { \operatorname* { m a x } } =$ 8 cm and contact force at 35 N. Command pulses peak at up to 25 N. The reward kernels have widths of 0.1 m (TCP position), 0.25 rad (hand orientation) and 25 N (force). $r _ { \mathrm { c o n t a c t } }$ rewards contact below 35 N.

Policy. The two 224 × 224 wrist images are encoded by a CLIP-initialized ViT-B/16; the state has 14 dimensions, and rotations use the first two rows of the rotation matrix. The 1- D U-Net denoiser is sampled with 16 DDIM steps, the image encoder is trained with a 10× lower learning rate, and each policy is trained for $1 0 ^ { 5 }$ steps with $H = 5 0 \ ( 1 . 6 7 { \mathrm { s } } )$

## B. Hybrid Compliance Whole-body Low-level Controller

Simulation protocol. Following [1], we train in Isaac Lab [17] and evaluate sim-to-sim in MuJoCo (Fig. 5). Motion/Force tracking is measured with $\mathbf { f } ^ { \mathrm { c m d } } = 0$ on two held-out sets of retargeted clips: ROOT-90 [38] and VLK-Pick-100 [39]. For force tracking, the dose-response test applies forward commands of 5, 10, 15, 20, and 25 N to each hand in turn, with $F { = } 0$ as the baseline; we also test 10 N commands along $\pm x , \pm y ,$ and ±z, and simultaneous commands to both hands. Statistics are computed over clips in which both paired conditions complete without a fall.

TABLE I: Sim-to-sim evaluation of motion and force tracking for our unified force–position controller and the baselines. Motion tracking is evaluated on two held-out motion sets. <sup>†</sup>Averaged over the left and right hands.
<table><tr><td></td><td></td><td colspan="3">Motion Tracking</td><td colspan="3">Force Tracking†</td></tr><tr><td>Controller</td><td>fcmd</td><td>Global Root (m)↓</td><td>Wrist Pos (mm) ↓</td><td>Wrist Ori (rad) ↓</td><td> $a _ { 1 }  1$ </td><td> $| e _ { F } | \left( { \bf N } \right) \downarrow$ </td><td>F=0 floor (N) ↓</td></tr><tr><td colspan="8">ROOT-90 [38]</td></tr><tr><td>SONIC [3]</td><td>X</td><td>0.317</td><td>47.16</td><td>0.2787</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>SONIC + admittance [3]</td><td>√</td><td>0.317</td><td>47.16</td><td>0.2787</td><td>0.70</td><td>7.89</td><td>2.62</td></tr><tr><td>FLoRE w/o Force</td><td>x</td><td>0.342</td><td>36.75</td><td>0.2239</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>FLoRE (Ours)</td><td>√</td><td>0.374</td><td>44.70</td><td>0.1176</td><td>1.06</td><td>1.05</td><td>1.50</td></tr><tr><td colspan="8">VLK-Pick-100 [39]</td></tr><tr><td>SONIC</td><td>X</td><td>0.072</td><td>21.45</td><td>0.2658</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>SONIC + admittance [3]</td><td>√</td><td>0.072</td><td>21.45</td><td>0.2658</td><td>0.67</td><td>7.87</td><td>2.62</td></tr><tr><td>FLoRE w/o Force</td><td>x</td><td>0.071</td><td>23.54</td><td>0.2141</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>FLoRE (Ours)</td><td>√</td><td>0.084</td><td>23.16</td><td>0.1165</td><td>1.03</td><td>1.00</td><td>1.25</td></tr></table>

Metrics. For motion tracking, we report global root error and wrist position and orientation errors [40]. For force tracking, the gain $a _ { 1 }$ is the slope of a least-squares fit, with free intercept, of the realized hold-phase force against the true contact force $\mathbf { f } ^ { \mathrm { e x t } }$ . The in-contact error $\| e _ { F } \| _ { 1 } =$ $\left\| \mathbf { f } _ { h } ^ { \mathrm { c m d } } + \mathbf { f } _ { h } ^ { \mathrm { m e a s } } \right\| _ { 1 }$ is averaged over the hold phase of each pulse. The $F { = } 0$ floor is the force read back when nothing is commanded; it bounds the smallest verifiable command.

Adding a force channel costs little motion tracking. Table I compares FLORE with FLORE $\mathtt { w / o }$ Force, SONIC [3], and SONIC + admittance, all evaluated with $\mathbf { f } ^ { \mathrm { c m d } } =$ 0 on two held-out motion sets. Relative to FLORE $\mathtt { w } / \mathtt { o }$ Force, FLORE adds 1.3–3.2 cm of global root error and at most 8.0 mm of wrist position error, while its wrist orientation error roughly halves; it also tracks orientation better than SONIC, at a higher root error. When a force is commanded, the hand target is shifted by design $( \mathbf { f } ^ { \mathrm { c m d } } / K$ alone is 12.5 cm at 25 N), and the hand departs from the motion reference only as far as needed to realize the force.

Hybrid compliance controller tracks commanded force accurately. Neither SONIC nor FLORE w/o Force accepts a force command. SONIC + admittance does, but its servo acts outside the balance policy and realizes only 0.67–0.70 of the commanded force, with an in-contact error of about 7.9 N. FLORE tracks 5–25 N commands with a gain of 1.03–1.06 and about 1 N of in-contact error per hand; lateral and downward commands reach a lower gain and leak partly into the other axes. With nothing commanded the sensor still reads 1.25–1.50 N, so at deployment we zero commands below 1.5 N and cap them at 25 N. Integrating the measured force error removes the dependence on the unknown contact stiffness: Spring-Law, the target-shift law of [9] without its force-feedback term, cannot detect contact, and its force scales with that stiffness relative to the assumed K. The same integral makes the hand compliant: while it tracks a force, a push along an uncommanded axis is read by the wrist sensor and builds up the integral, which moves the target with it, so the hand gives way instead of resisting, with no mode switch (real robot, Fig. 5 bottom).

Real-world force tracking. We verify force tracking on the robot against an external reference: a rigidly mounted digital force gauge that the hand presses along each of $\pm x , \pm y .$ , and ±z while the humanoid stands freely. The command follows a staircase of 4, 6, 8, and 10 N, repeated across directions and hands. Along the x axis, the measured steady-state force reaches approximately 91% of the commanded magnitude, with a mean absolute steady-state error of about 1.1 N. The response enters a ±1 N band in 1.8 s on average, while the steady-state fluctuation remains approximately ±0.3–0.5 N. The controller consistently recovers the commanded contact force without loss of stability.

## C. Push Gantry

Task. As shown in Fig. 1b, the robot places both hands on a heavy wheeled gantry, from which a second humanoid hangs, and walks it forward to a target distance. The demonstrated hand motion does not specify how hard to push, yet the gantry does not move until the push overcomes its static friction. This is the only task recorded with ankle trackers, so the policy predicts all five keypoints. A rollout succeeds if the gantry reaches the target distance without a stop.

Capabilities. Bimanual and whole-body coordination: the legs and torso produce the push while the two hands balance their forces; otherwise the gantry drifts from its heading. Sustained contact: both hands must keep contact as the gantry accelerates and the robot steps. Robustness to force variation: the hanging humanoid swings as the gantry rolls, so the resistance changes over time.

Test scenarios. We run 10 rollouts per policy over two gantry loads and five initial poses that vary in distance and heading. Performance. Ours reaches the target in 7/10 rollouts (Table II). It fails when a strong push coincides with a step. DP succeeds in 2/10 rollouts; in the others the gantry either does not move or drifts from its heading as the two hands push unequally (Fig. 6, top). w/o F-cmd succeeds in 3/10 and fails similarly, as it senses the resistance but cannot command force. w/o F-in succeeds in 6/10. Without force feedback, it stalls against the heavier load and keeps walking after losing contact with the lighter one. Offline, Ours has a pose error of 0.82 cm and a force error of 0.77 N per hand.

![](images/d3397448bf23f08165525743fcb7013af0116da7bbf3612797889c1562c2e9ac.jpg)

![](images/6355293c4b443cfe16e159d494925b8dab4b5f70b30488908ea6d9eae266dd9a.jpg)

![](images/61d8b9ff72894653fb12848ab75d5b22fead8ed03aedbea545b56dab385ce433.jpg)

![](images/116302afa2e3f90e817d83d51dfd910d3d116e823810e001653f75d326605556.jpg)

![](images/8cc8e92ae5230fca57cd32dfaf1c2e9f64747d7e2a3d0e5c828fe0b899019dfb.jpg)  
Fig. 6. Rollouts on the robot and failure modes of the position-only baseline. Top: Ours walks the gantry forward from left to right. The baseline either fails to move the gantry or pushes unequally, causing it to drift from its heading. The traces show commanded (dashed) and measured (solid) force for each hand and axis. Bottom left: Ours drags and rotates the box. The baseline either cannot move it or leans into it and loses balance. Bottom right: Ours erases the strip under steady pressure. The baseline hovers above the board and never makes contact.

TABLE II: Success rate and offline validation error of the predicted pose, force and torque for Ours and its force-pathway ablations, trained on the same demonstrations and executed by the same unified controller.
<table><tr><td></td><td colspan="3">Force Switches</td><td colspan="4">Success Rate (%) ↑</td><td colspan="3">Evaluation Error (Val) ↓</td></tr><tr><td>Policy</td><td>F-in</td><td>F-out</td><td>F-cmd</td><td>PUSH_GANTRY</td><td>ROTATE_BOX</td><td>WIPE_BOARD</td><td>Mean</td><td>Pose (cm)</td><td>Force (N)</td><td>Torque (N m)</td></tr><tr><td>Ours</td><td>√</td><td>√</td><td>√</td><td>7/10</td><td>9/10</td><td>8/10</td><td>80%</td><td>1.18</td><td>1.29</td><td>0.148</td></tr><tr><td> $\boldsymbol { \mathrm { w } } / \mathrm { o } \mathrm { ~ \ E - c m d } ^ { \dag }$ </td><td>√</td><td>√</td><td>X</td><td>3/10</td><td>4/10</td><td>4/10</td><td>36.7%</td><td>1.18</td><td>1.29</td><td>0.148</td></tr><tr><td>w/o F-in</td><td>X</td><td>√</td><td>√</td><td>6/10</td><td>7/10</td><td>7/10</td><td>66.7%</td><td>1.78</td><td>2.51</td><td>0.277</td></tr><tr><td>DP</td><td>X</td><td>x</td><td>X</td><td>2/10</td><td>3/10</td><td>3/10</td><td>26.7%</td><td>2.23</td><td>N/A</td><td>N/A</td></tr></table>

<sup>†</sup>Predicted and supervised but withheld from the controller $( \mathbf { f } ^ { \mathrm { c m d } } = 0 )$ : auxiliary supervision only. Offline, w/o F-cmd is the same network as Ours.

## D. Rotate Box

Task. As shown in Fig. 1c, the robot pinches the rim of a box with both hands. It rotates the box by about 50<sup>◦</sup>, drags it about 0.5 m across the table, and releases it. The box remains on the table, so the measured wrench reflects table friction, box inertia, and the internal squeeze between the hands. A rollout succeeds if the box rotates by more than $4 0 ^ { \circ }$ and remains upright after release.

Capabilities. Bimanual coordination: the hands share an internal force and rotate about a common axis. In the demonstrations, the right hand applies about 1.4× the force of the left. If one hand leads the other, the box tips. Sustained contact: the pinch must hold through the drag and the yaw without slipping or lifting the box. Robustness to force variation: friction changes with the box mass and when the box begins to slide.

Test scenarios. We run 10 rollouts per policy over two box masses and five initial yaw angles.

Performance. Ours succeeds in 9/10 rollouts and has an offline pose error of 0.70 cm. DP succeeds in 3/10 and exhibits two failure modes (Fig. 6, bottom left). It either fails to drag the heavy box or leans into it and loses balance. w/o F-cmd succeeds in 4/10 and fails similarly because its predicted squeeze is not executed. w/o F-in succeeds in 7/10 but cannot react when the box begins to slide.

## E. Wipe Board

Task. As shown in Fig. 1d, the robot presses an eraser against a whiteboard and wipes a marked strip. Success depends on normal force, which is not visible in the images. A rollout succeeds if the strip is erased without a stop.

Capabilities. Sustained contact: the robot must maintain normal force during each stroke and re-establish it after losing contact. Images alone do not distinguish hovering from pressing. Robustness to force variation: we displace the board by ±3 cm along its normal. The demonstrated trajectory then produces either no contact or excessive force if tracked directly.

Test scenarios. We run 10 rollouts per policy over two board offsets (±3 cm along the normal) and two surfaces.

Performance. Ours erases the strip in 8/10 rollouts. DP succeeds in 3/10. It usually hovers above the board (Fig. 6, bottom right). This directly shows that observed motion can be valid supervision in free space but not in contact [41]. w/o F-cmd succeeds in 4/10. It detects contact but cannot increase a force that is too low. w/o F-in succeeds in 7/10 but continues wiping in free space when the board is farther away. This task also has the largest offline errors: 2.02 cm for pose and 1.59 N for force.

## F. Findings Summary

F1: The force channel costs little motion tracking. Relative to FLORE w/o Force, FLORE adds 1.3–3.2 cm of root error and at most 8.0 mm of wrist position error while halving wrist orientation error (Table I); in return it tracks commanded force with a gain of 1.03–1.06 and about 1 N of in-contact error. The 1.25–1.50 N F=0 floor bounds the smallest verifiable command, and lateral and downward commands remain weaker.

F2: Force-reactive policies resolve failures that motion alone cannot distinguish. With the same demonstrations and controller, DP succeeds in 26.7% of rollouts and FLORE in 80% (Table II). The position-only policy stalls when a reached pose does not move the object, hovers or presses too hard when the board is displaced, and loses balance against the heavy box (Fig. 6). Force also composes with coordination: both hands hold the internal force on Rotate Box, and the push is sustained while walking on Push Gantry. F3: Force commands give the larger, significant gain; force observations show a smaller trend. Withholding the predicted force from the controller reduces success from 80% (24/30) to 36.7% (11/30; Fisher’s exact test, p = 0.001): the policy senses resistance but can respond only through pose. Removing the force observation reduces success to 66.7%.

## V. CONCLUSION

We presented FLORE, a framework for learning force-reactive humanoid loco-manipulation from robot-free demonstrations that record both motion and contact wrench. Our hybrid compliance whole-body controller tracks the learned force commands, allowing the humanoid to sustain contact, coordinate both hands, and apply force while walk ing and balancing. On three tasks, the force-reactive policy succeeds in 80% of rollouts, compared with 26.7% for a position-only policy. By making force a shared interface between demonstrations, policies, and whole-body control, FLORE extends robot-free imitation learning to contact-rich skills that motion alone cannot specify.

## REFERENCES

[1] Q. Liao, T. E. Truong, X. Huang, Y. Gao, G. Tevet, K. Sreenath, and C. K. Liu, “BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion,” Science Robotics, 2026.

[2] T. He, W. Xiao, T. Lin, Z. Luo, Z. Xu, Z. Jiang, J. Kautz, C. Liu, G. Shi, X. Wang, L. Fan, and Y. Zhu, “HOVER: Versatile Neural Whole-Body Controller for Humanoid Robots,” in ICRA, 2025.

[3] Z. Luo, Y. Yuan, T. Wang, C. Li, F. Castaneda, S. Chen, Z.-A. Cao, ˜ J. Li, D. Minor, Q. Ben, J. Park, D. Sami, Z. Wang, X. Da, R. Ding, C. Hogg, L. Song, E. Lim, E. Jeong, T. He, H. Xue, W. Xiao, S. Yuen, J. Kautz, Y. Chang, U. Iqbal, L. Fan, and Y. Zhu, “SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control,” Science Robotics, 2026.

[4] R. Nai, B. Zheng, J. Zhao, H. Zhu, S. Dai, Z. Chen, Y. Hu, Y. Hu, T. Zhang, C. Wen, and Y. Gao, “Humanoid Manipulation Interface: Humanoid Whole-Body Manipulation from Robot-Free Demonstrations,” arXiv:2602.06643, 2026.

[5] X. Xu, J. Park, H. Zhang, E. Cousineau, A. Bhat, J. Barreiros, D. Wang, J. Bohg, and S. Song, “HoMMI: Learning Whole-Body Mobile Manipulation from Human Demonstrations,” in RSS, 2026.

[6] Y. Zhang, Y. Yuan, P. Gurunath, I. Gupta, S. Omidshafiei, A. akbar Agha-mohammadi, M. Vazquez-Chanlatte, L. Pedersen, T. He, and G. Shi, “FALCON: Learning Force-Adaptive Humanoid Loco-Manipulation,” in L4DC, 2026.

[7] G. B. Margolis, M. Wang, N. Fey, and P. Agrawal, “SoftMimic: Learning Compliant Whole-body Control from Examples,” in ICRA, 2026.

[8] T. Portela, G. B. Margolis, Y. Ji, and P. Agrawal, “Learning Force Control for Legged Manipulation,” in ICRA, 2024.

[9] P. Zhi, P. Li, J. Yin, B. Jia, and S. Huang, “Learning a Unified Policy for Position and Force Control in Legged Loco-Manipulation,” in CoRL, 2025.

[10] C. Chi, Z. Xu, C. Pan, E. Cousineau, B. Burchfiel, S. Feng, R. Tedrake, and S. Song, “Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots,” in RSS, 2024.

[11] H. Wang, C. Yu, Y. Hu, J. Zhang, Y. Li, and S. Luo, “BifrostUMI: Bridging Robot-Free Demonstrations and Humanoid Whole-Body Manipulation,” arXiv:2605.03452, 2026.

[12] H. Choi, Y. Hou, C. Pan, S. Hong, A. Patel, X. Xu, M. R. Cutkosky, and S. Song, “In-the-Wild Compliant Manipulation with UMI-FT,” in ICRA, 2026.

[13] W. Liu, J. Wang, Y. Wang, W. Wang, and C. Lu, “ForceMimic: Force-Centric Imitation Learning with Force-Motion Capture System for Contact-Rich Manipulation,” in ICRA, 2025.

[14] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song, “Diffusion Policy: Visuomotor Policy Learning via Action Diffusion,” in RSS, 2023.

[15] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware,” in RSS, 2023.

[16] Y. Niu, Z. Fang, B. Chen, S. Zhou, R. K. Senthilkumaran, H. Zhang, B. Chen, C. Qiu, H. E. Tseng, J. Francis, and D. Zhao, “Learning Versatile Humanoid Manipulation with Touch Dreaming,” in IROS, 2026.

[17] M. Mittal, P. Roth, J. Tigue, A. Richard, O. Zhang, P. Du, A. Serrano-Munoz, X. Yao, R. Zurbr˜ ugg, N. Rudin, L. Wawrzyniak, M. Rakhsha,¨ A. Denzler, E. Heiden, A. Borovicka, O. Ahmed, I. Akinola, A. Anwar, M. T. Carlson, J. Y. Feng, A. Garg, R. Gasoto, L. Gulich, Y. Guo, M. Gussert, A. Hansen, M. Kulkarni, C. Li, W. Liu, V. Makoviychuk, G. Malczyk, H. Mazhar, M. Moghani, A. Murali, M. Noseworthy, A. Poddubny, N. Ratliff, W. Rehberg, C. Schwarke, R. Singh, J. L. Smith, B. Tang, R. Thaker, M. Trepte, K. V. Wyk, F. Yu, A. Millane, V. Ramasamy, R. Steiner, S. Subramanian, C. Volk, C. Chen, N. Jawale, A. V. Kuruttukulam, M. A. Lin, A. Mandlekar,

K. Patzwaldt, J. Welsh, H. Zhao, F. Anes, J.-F. Lafleche, N. Moenne-¨ Loccoz, S. Park, R. Stepinski, D. V. Gelder, C. Amevor, J. Carius, J. Chang, A. H. Chen, P. de Heras Ciechomski, G. Daviet, M. Mohajerani, J. von Muralt, V. Reutskyy, M. Sauter, S. Schirm, E. L. Shi, P. Terdiman, K. Vilella, T. Widmer, G. Yeoman, T. Chen, S. Grizan, C. Li, L. Li, C. Smith, R. Wiltz, K. Alexis, Y. Chang, D. Chu, L. J. Fan, F. Farshidian, A. Handa, S. Huang, M. Hutter, Y. Narang, S. Pouya, S. Sheng, Y. Zhu, M. Macklin, A. Moravanszky, P. Reist, Y. Guo, D. Hoeller, and G. State, “Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning,” arXiv:2511.04831, 2025.

[18] S. Luo, Y. Li, Y. Hu, C. Yu, C. Xu, J. Zhang, G. Yao, T. Huang, R. He, and Z. Wang, “OmniUMI: Towards Physically Grounded Robot Learning via Human-Aligned Multimodal Interaction,” arXiv:2604.10647, 2026.

[19] J. Jang, Z. Gu, A. Cueva, Z. Chai, J. Sheng, T. Nguyen, H. Galundia, Y. Wu, H. Xue, I. Legene, O. Mediratta, D. Doan, A. Collins, S. Sadegh, K. Kim, R. Dhalbisoi, Z. Chen, and Y. Zhao, “WT-UMI: Tactile-based Whole-Body Manipulation via Force-Supervised Contact-Aware Planning,” in CoRL, 2026.

[20] Z. He, H. Fang, J. Chen, H.-S. Fang, and C. Lu, “FoAR: Force-Aware Reactive Policy for Contact-Rich Robotic Manipulation,” RA-L, 2025.

[21] J. J. Liu, Y. Li, K. Shaw, T. Tao, R. Salakhutdinov, and D. Pathak, “FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning,” in RSS, 2025.

[22] R. Mart´ın-Mart´ın, M. A. Lee, R. Gardner, S. Savarese, J. Bohg, and A. Garg, “Variable Impedance Control in End-Effector Space: An Action Space for Reinforcement Learning in Contact-Rich Tasks,” in IROS, 2019.

[23] T. Kamijo, C. C. Beltran-Hernandez, and M. Hamaya, “Learning Variable Compliance Control From a Few Demonstrations for Bimanual Robot with Haptic Feedback Teleoperation System,” in IROS, 2024.

[24] Y. Hou, Z. Liu, C. Chi, E. Cousineau, N. Kuppuswamy, S. Feng, B. Burchfiel, and S. Song, “Adaptive Compliance Policy: Learning Approximate Compliance for Diffusion Guided Control,” in ICRA, 2025.

[25] H. Fang, S. Tang, M. Mei, H. Qin, Z. He, J. Chen, Y. Feng, C. Wang, W. Liu, Z. He, C. Lu, and S. Wang, “Force Policy: Learning Hybrid Force-Position Control Policy under Interaction Frame for Contact-Rich Manipulation,” in RSS, 2026.

[26] H. He, Z. Yan, Q. Liu, N. Guo, and W. Lian, “FAWAM: Force-Aware World Action Models for Closed-Loop Contact-Rich Manipulation,” arXiv:2606.08555, 2026.

[27] K. Xiao, Z. Xu, C. Zhe, C. Liu, and Q. Chen, “Kinematics-Aware Multi-Policy Reinforcement Learning for Force-Capable Humanoid Loco-Manipulation,” arXiv:2511.21169, 2025.

[28] C. Liu, Q. Lu, G. Yang, X. Shi, C. Yang, Y. Guo, and J. Chen, “HEFT: Heavy-Payload Full-size Humanoid Teleoperation with Privileged Motion Guidance and Windowed Payload Curriculum,” arXiv:2607.02332, 2026.

[29] B. Xu, H. Weng, Q. Lu, Y. Gao, and H. Xu, “FACET: Force-Adaptive Control via Impedance Reference Tracking for Legged Robots,” in CoRL, 2025.

[30] S. Chen, Z. ang Cao, Z. Luo, F. Castaneda, C. Li, T. Wang, Y. Yuan,˜ L. J. Fan, C. K. Liu, and Y. Zhu, “CHIP: Adaptive Compliance for Humanoid Control through Hindsight Perturbation,” in CoRL, 2026.

[31] Y. Zhou, C. Barnes, J. Lu, J. Yang, and H. Li, “On the Continuity of Rotation Representations in Neural Networks,” in CVPR, 2019.

[32] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning Transferable Visual Models From Natural Language Supervision,” in ICML, 2021.

[33] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” in Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2015.

[34] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, “FiLM: Visual Reasoning with a General Conditioning Layer,” in AAAI Conference on Artificial Intelligence (AAAI), 2018.

[35] J. Ho, A. Jain, and P. Abbeel, “Denoising Diffusion Probabilistic Models,” in NeurIPS, 2020.

[36] J. Song, C. Meng, and S. Ermon, “Denoising Diffusion Implicit Models,” in ICLR, 2021.

[37] T. Lin, Y. Zhang, Q. Li, H. Qi, B. Yi, S. Levine, and J. Malik, “Learning Visuotactile Skills with Two Multifingered Hands,” in ICRA, 2025.

[38] RoboParty Lab Team, “sim2real: A lightweight and modular sim2sim and sim2real deployment stack,” https://github.com/ EGalahad/sim2real, 2026, documentation: https://egalahad.github.io/ sim2real/.

[39] Y.-J. Wang, J. Li, S. Chen, T. E. Truong, P. Xu, P. Abbeel, R. Duan, K. Sreenath, A. Kanazawa, C. Sferrazza, G. Shi, and C. K. Liu, “Vlk: Learning humanoid loco-manipulation from synthetic interactions in reconstructed scenes,” in CoRL, 2026.

[40] C. Tessler, Y. Guo, O. Nabati, G. Chechik, and X. B. Peng, “Masked-Mimic: Unified Physics-Based Character Control Through Masked Motion Inpainting,” ACM Transactions on Graphics (SIGGRAPH Asia), vol. 43, no. 6, 2024.

[41] V. Surendran, N. Peri, and D. Watkins, “Bridging Handheld and Teleoperated Supervision for Contact-Rich Manipulation via State-Gated Experts,” arXiv:2606.26603, 2026.