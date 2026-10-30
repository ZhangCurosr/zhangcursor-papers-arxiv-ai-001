# AegisFlow: Training-Free Non-Myopic Path-Safe Guided Flow Matching

Kunpeng Liu   
Department of Automation and BNRist Tsinghua University Beijing, China   
liukp22@mails.tsinghua.edu.cn

Boshi Zhang Department of Automation and BNRist Tsinghua University Beijing, China zbs22@mails.tsinghua.edu.cn

Keyou You<sup>∗</sup> Department of Automation and BNRist Tsinghua University Beijing, China youky@tsinghua.edu.cn

## Abstract

Pretrained generative models, such as diffusion and flow matching models, have demonstrated exceptional capabilities in synthesizing high-quality data. However, steering their generative trajectories to satisfy strict constraints—such as identity preservation in image editing or obstacle avoidance in robotics—remains fundamentally challenging. While recent training-free guidance methods enforce path-safety, they rely heavily on min-norm controllers that only account for immediate constraint satisfaction. These myopic approaches often yield suboptimal trajectories that hover dangerously close to the boundary, severely restricting the model’s ability to optimize toward its target. In this paper, we propose AegisFlow, a training-free, non-myopic path-safe guidance framework that recasts guided generation as a safety-critical optimal control problem with global scope. By integrating a time-varying barrier function directly into the global cost functional, AegisFlow naturally enforces constraints along the entire generative path. To compute the global optimal control, we derive a tractable, mathematically grounded approximation of the Hamilton-Jacobi-Bellman (HJB) equation. Comprehensive experiments demonstrate that AegisFlow outperforms state-of-the-art methods across text-guided image manipulation and robotic planning, establishing a significantly better trade-off between target alignment and constraint satisfaction. Code is available at https://github.com/LiuKunpeng03/AegisFlow.

## 1 Introduction

Generative models such as diffusion [25, 53] and flow matching [34] have exhibited remarkable capabilities in synthesizing high-quality data across various domains, including images [45, 16] and planning [8, 14]. Guided generation, e.g., image editing [38, 24], takes advantage of the expressiveness of pretrained generative models and steers the generative trajectory towards a target. Reward-guided methods [52, 36, 56], which leverage a predefined reward function to optimize the trajectory at inference time, have attracted significant attention due to their training-free nature and flexibility in handling various objectives.

![](images/300cbc6f51dd9388646872d61b6802a552931b8c1d88f8630205ce06b90fc800.jpg)  
Figure 1: Toy Example. Vanilla represents the original flow which is trained to fit into the contour of a cat. Guidance stands for the case where the flow is guided towards two eyes of the cat. Although the guidance successfully steers the trajectory, many trajectories collide with the red yarn ball which serves as the obstacle. Correction means min-norm controller is applied to the former case. While all trajectories are safe, the quality is significantly degraded. Aegis successfully guides most of the trajectories towards the target while maintaining a safe distance from the obstacle.

Besides objective guidance, constraint satisfaction is increasingly accounted for in guided generation. For example, fake or harmful content should be strictly prohibited from generation [51, 33], and generated trajectories for robotics should obey physical law and avoid obstacles [27, 8]. Previous methods deal with constraints by either soft guidance [60, 9] or projection [10, 54, 61]. The former admits no hard guarantees on constraint satisfaction, while the latter might hinder the generation quality. Recent works [57, 58, 20] extend the desired constraint on the output to the entire generative trajectory, and enforce path-safety by imposing control to the pretrained flow model. These methods are able to achieve better constraint satisfaction while maintaining generation quality. However, they apply a myopic min-norm controller, which only considers the immediate effect of the control input without accounting for the long-term effect on the trajectory. As a result, these methods often find suboptimal solutions that hover dangerously close to safety boundaries, severely limiting the aggressiveness with which a model can optimize toward its target. See Figure 1 for an illustration.

To overcome the fundamental limitations of myopic controllers, we propose AegisFlow, a trainingfree, non-myopic path-safe guidance method for flow matching models. AegisFlow recasts guided generation as a safety-critical optimal control problem with a global scope. Instead of relying on perstep corrections, we integrate a time-varying barrier function directly into the global cost functional. This formulation penalizes the trajectory for approaching the boundary of the safe set, ensuring the entire generated path remains within a safe tube. AegisFlow seeks to find the global optimal control via estimating the solution to the Hamilton-Jacobi-Bellman (HJB) equation associated with the cost functional, which inherently accounts for the long-term effects of control inputs on both target alignment and safety constraints. Since the barrier function drives the cost to infinity near safety boundaries, any finite-cost solution natively guarantees constraint satisfaction, allowing us to bypass the ad-hoc projections or per-step barrier functions required by prior methods.

We extensively evaluate AegisFlow across diverse and challenging domains, spanning from highdimensional text-guided image manipulation (CelebA-HQ) to physical planning (Maze2D and D3IL robotic manipulation). In image editing, AegisFlow proactively maintains a larger safety margin for identity preservation, enabling stronger textual alignment (higher CLIP scores) and better identity retention than myopic baselines. In continuous control tasks, AegisFlow achieves higher success rates, smoother trajectories, and fewer safety violations compared to current state-of-the-art safe diffusion methods.

In summary, our main contributions are as follows:

• Overcoming of Myopic Limitations: We identify the suboptimality of min-norm controllers in guided flow matching and propose AegisFlow, a globally aware, training-free optimal control framework.

• Tractable HJB Approximation: We formulate a barrier-inclusive cost functional and derive a tractable, mathematically grounded approximation for the optimal safe control using the Hamilton-Jacobi-Bellman (HJB) equation.

• Comprehensive Empirical Validations: We demonstrate superior performance across both high-dimensional image manipulation and safety-critical continuous control tasks, establishing a significantly better trade-off between target alignment and constraint satisfaction.

## 2 Related Works

Training-free image manipulation. There are various training-free methods for image manipulation based on pretrained diffusion models. The mainstream approach starts with image-to-noise inversion and then applies certain guidance to the generative trajectory [36, 50, 7, 26, 5, 56]. Some works directly optimize over the original image [31, 41, 59]. Recently, Kulikov et al. [32] also explore inversion-free methods. In this paper, we will mainly focus on the first approach. However, we think there is no fundamental barrier to applying our method to the other approaches, and we leave it for future work.

Flow matching under control theory. Prior works have formulated the training [15, 65, 66], sampling [63, 6, 55, 23], and test-time editing [36, 56, 48, 49, 21] of flow matching models as optimal control problems. Some of them use Pontryagin’s Maximum Principle (PMP) [15, 36, 56], while others adopt the Hamilton-Jacobi-Bellman (HJB) approach [65, 49, 21]. They aim to optimize a certain cost, while the behavior of the trajectory is controlled by a regularizer and admits no hard guarantees. Several recent works explore the use of per-step projection [10, 47, 61, 54, 37], control barrier functions (CBFs) with min-norm controller (MNC) [57, 62, 58, 20] to enforce hard constraints along the entire trajectory. However, the myopic nature of per-step projection and MNCs remains to be addressed. Mizuta and Leung [40] try to encode the CBF constraint into the reward function. But they do so by including the increment of the term inside the supremum in (19) (in Appendix A) between two timesteps as part of the reward, which is still essentially myopic and cannot find globally optimal solutions. Mishra and Manchester [39] explore the use of an emerging barrier function to enforce safety constraints in the diffusion process, but their method is mainly a heuristic one. In constrast, our method estimates the optimal control from the HJB equation with a cost functional that includes the barrier function, which attempts to find a globally optimal solution that minimizes the cost functional while ensuring the trajectory stays within a safe set.

## 3 Preliminaries and Problem Formulation

Flow Matching. Flow matching [34, 35, 1] formulates the diffusion models from a perspective of ordinary differential equations (ODEs). Given a data distribution $\pi _ { 1 }$ and a noise distribution $\pi _ { 0 } ,$ , flow matching learns a time-dependent vector field $f ( x , t )$ that transports $\pi _ { 0 } \tan \pi _ { 1 }$ along the trajectories defined by the ODE:

$$
\begin{array} { r } { \dot { x } = f ( x , t ) , \quad x ( t _ { 0 } ) \sim \pi _ { 0 } . } \end{array}\tag{1}
$$

Indeed, once a sample $x _ { 0 }$ is drawn from $\pi _ { 0 } .$ , a unique trajectory $x _ { t }$ is determined by the ODE under necessary regularity conditions. Therefore f induces a map $\Phi _ { t _ { 1 }  t _ { 2 } } ^ { f }$ such that $\Phi _ { t _ { 1 }  t _ { 2 } } ^ { f } ( x ( t _ { 1 } ) ) = x ( t _ { 2 } )$ which further results in a pushforward measure $\Phi _ { t _ { 0 } \to t _ { f } } ^ { f } ( \pi _ { 0 } )$ . The flow matching objective is to learn

![](images/c385da8141f06bcea83a0607fb30da4617eec381a2ab8d2df31d10c9454023c1.jpg)  
Figure 2: Conceptual comparison between myopic control and AegisFlow. (a) Myopic Control: A greedy controller driven purely by local target attraction and rigid boundary constraints. It gets trapped sliding jaggedly along the edge of the Unsafe Set and fails to reach the target state. (b) AegisFlow: By explicitly modeling the global cost field, the proposed trajectory is smooth and globally optimal. It proactively avoids the obstacle with an anticipated safety margin and successfully converges to the target. (c) Barrier Cost Landscape: The cost functional used by AegisFlow combines a barrier function to enforce constraints and a reward of the target, balancing constraint satisfaction and target guidance.

a vector field $f$ such that $\Phi _ { t _ { 0 }  t _ { f } } ^ { f } ( \pi _ { 0 } ) = \pi _ { 1 }$ . In practice, flow matching generally minimizes the regression objective:

$$
\mathbb { E } _ { t , x ( t _ { 0 } ) , x ( t _ { f } ) } ( \| f ( x ( t ) , t ) - u _ { t } ( x ( t ) | x ( t _ { 0 } ) , x ( t _ { f } ) ) \| _ { 2 } ^ { 2 } ) ,
$$

where $u _ { t }$ is the velocity field of the reference trajectory, which can be defined in various ways, e.g.   
linear interpolant [1].

Guided Generation and Optimal Control. The primary goal of guided generation is to steer the generative trajectory towards a target while satisfying some constraints, e.g. identity preservation or obstacle avoidance. In the scope of control theory, the problem can be formulated as follows. Given a pretrained flow matching model with vector field $f ,$ we consider the following controlled dynamics:

$$
{ \dot { x } } = f ( x , t ) + u ,\tag{2}
$$

where u is the control input to be chosen. Reward-guided methods prove to be effective in steering the flow towards the target. Specifically, they select the control input u that minimizes a certain cost functional,

$$
J ( u ) : = \int _ { t _ { 0 } } ^ { t _ { f } } L ( x ( t ) , u ( t ) , t ) \mathrm { d } t + \Psi ( x ( t _ { f } ) ) , \quad \mathrm { s . t . } \quad x ( t _ { 0 } ) = x _ { 0 } ,\tag{3}
$$

where $L$ is the running cost and Ψ is the terminal cost. For example, Wang et al. [56] set $L ( x , u , t ) =$ $\lVert u \rVert _ { 2 } ^ { 2 }$ and $\Psi ( \cdot ) = - \bar { \mathrm { C L I P } } ( \cdot$ ·, prompt) in (3) in the text-guided image manipulation task to encourage prompt adherence.

Finding the optimal control $u ^ { \star }$ that minimizes the cost functional (3) is a classical problem in control theory, which can be solved by the Hamilton-Jacobi-Bellman (HJB) equation. To begin with, let us make the dependence of the cost functional $J$ on the initial state $x _ { 0 }$ and time $t _ { 0 }$ explicit, namely

$$
J ( u , x _ { 0 } , t _ { 0 } ) : = \int _ { t _ { 0 } } ^ { t _ { f } } L ( x ( t ) , u ( t ) , t ) \mathrm { d } t + \Psi ( x ( t _ { f } ) ) , \quad \mathrm { s . t . } \quad x ( t _ { 0 } ) = x _ { 0 } .\tag{4}
$$

Denote the value function of the optimal control problem as

$$
V ( x , t ) : = \operatorname* { m i n } _ { u } J ( u , x , t ) .\tag{5}
$$

HJB states that under necessary regularity conditions, $V$ satisfies the following PDE:

$$
- \partial _ { t } V ( x , t ) = \operatorname* { m i n } _ { n } [ L ( x , u , t ) + \nabla _ { x } V ( x , t ) \cdot ( f ( x , t ) + u ) ] , \quad \mathrm { s . t . } \quad V ( x , t _ { f } ) = \Psi ( x ) ,\tag{6}
$$

and the optimal control $u ^ { \star }$ satisfies

$$
u ^ { \star } ( x , t ) \in \arg \operatorname* { m i n } _ { u } [ L ( x , u , t ) + \nabla _ { x } V ( x , t ) \cdot ( f ( x , t ) + u ) ] .\tag{7}
$$

Algorithm 1 AegisFlow   
1: Input: $x _ { t _ { 0 } } \sim \rho _ { 0 } ,$ , schedules $\{ \lambda _ { t } , \beta _ { t } \} .$ , barrier function B, reference flow $f ,$ terminal loss $\Psi ,$ , timestep grid   
$\left\{ t _ { 0 } , \dots , t _ { N } \right\}$ , maximum iteration M, learning rate η, weight decay ν   
2: for $k = 0 , \ldots , M - 1$ do   
3: Solve the controlled dynamics (2) with the current control $u _ { n } ^ { k }$ to obtain the trajectory $\{ x _ { n } \ =$   
$x ( t _ { n } ) \} _ { n = 0 , 1 , . . . , N - 1 }$   
4: $\begin{array} { r } { s _ { n } \gets \int _ { t _ { n } } ^ { 1 } \lambda _ { \tau } ^ { - 1 } \mathrm { d } \tau } \end{array}$   
5: $g _ { \mathrm { s a f e } , n } \gets \nabla _ { x } B ( x _ { n } , t _ { n } )$   
6: $g _ { \mathrm { t a r } , n } \gets \nabla _ { x } ( \Psi \circ \Phi _ { t _ { n }  1 } ^ { f + u ^ { k } } ) ( x _ { n } )$   
7: $g _ { n } \gets \lambda _ { t _ { n } } ^ { - 1 } g _ { \mathrm { t a r } , n } + s _ { n } \beta _ { t _ { n } }$ g<sub>safe,n</sub>   
8: $u _ { n } ^ { k + 1 }  \nu u _ { n } ^ { k } - \eta g _ { n }$   
9: end for   
10: return $x _ { t _ { N } }$

Conversely, if a function $\hat { V }$ satisfies (6) and a control uˆ satisfies (7) with V replaced by $\hat { V }$ , then it must be the case that $\hat { V }$ is the value function and uˆ is an optimal control.

Constraint satisfaction is also increasingly considered in guided generation. It requires the generated sample to satisfy certain constraints, $\mathrm { i . e . , } \bar { x } ( t _ { f } ) \in \mathcal { C }$ for some constraint set $\mathcal { C } .$ . Traditional approaches utilize soft guidance or projection to enforce constraints, which offer no hard guarantees or suffer from performance degradation. For example, Gao et al. [21] set Ψ to be the distance to the admissible set and utilize HJB to estimate the optimal control. To alleviate performance deterioration caused by post hoc projection, recent works extend the constraints to the entire trajectory [57, 58, 20], i.e., they demand $x ( t ) \in \mathcal { C } _ { t }$ for all $t \in [ t _ { 0 } , t _ { f } ]$ where $\mathcal { C } _ { t }$ is a time-varying safe set with $\mathcal { C } _ { t _ { f } } \subseteq \mathcal { C } ,$ . They enforce such path-safety by CBFs and MNCs. However, MNCs only consider the immediate effect of the control input and such myopic nature often leads to suboptimal solutions. Details about CBFs and MNCs are deferred to Appendix A.

In summary, the core problem we seek to address is as follows: How can we find the optimal control that minimizes the cost functional while satisfying the given constraints? Formally speaking, we aim to solve the following optimization problem:

$$
\begin{array} { r l } { \underset { u } { \operatorname* { m i n } } } & { \quad J ( u , x _ { 0 } , t _ { 0 } ) : = \displaystyle \int _ { t _ { 0 } } ^ { t _ { f } } L ( x ( t ) , u ( t ) , t ) \mathrm { d } t + \Psi ( x ( t _ { f } ) ) , } \\ { \mathrm { s . t . } } & { \quad \dot { x } = f ( x , t ) + u , \quad x ( t _ { 0 } ) = x _ { 0 } , \quad x ( t ) \in \mathcal { C } _ { t } , \quad \forall t \in [ t _ { 0 } , t _ { f } ] . } \end{array}\tag{8}
$$

## 4 Methodology

## 4.1 AegisFlow: Non-Myopic Path-Safe Guided Flow Matching

We have recast guided generation as a safety-critical optimal control problem (8). To enhance the long-term awareness of the safe control, we consider the following cost functional:

$$
J ( u , x _ { 0 } , t _ { 0 } ) : = \int _ { t _ { 0 } } ^ { t _ { f } } \Big [ \frac { \lambda _ { t } } { 2 } \| u ( t ) \| _ { 2 } ^ { 2 } + \beta _ { t } B ( x ( t ) , t ) \Big ] \mathrm { d } t + \Psi ( x ( t _ { f } ) ) , \quad \mathrm { s . t . } \quad x ( t _ { 0 } ) = x _ { 0 } ,\tag{9}
$$

where B(·, t) : Int $( \mathcal { C } _ { t } )  \mathbb { R } _ { \geq 0 }$ is a barrier function that satisfies

$$
\operatorname* { l i m } _ { x  \partial \mathcal { C } _ { t } } B ( x , t ) = + \infty , \quad \forall t \in [ t _ { 0 } , t _ { f } ] .
$$

Note that the safety constraints are implicitly enforced by the barrier function in the cost functional, which allows us to directly apply the HJB approach to estimate the optimal control. Intuitively, the barrier function B penalizes the trajectory for getting too close to the boundary of the safe set, which encourages the trajectory to stay within $\mathcal { C } _ { t } .$ . See Figure 2 for an illustration. Indeed, if the cost functional (9) is finite, then the trajectory must satisfy $x ( t ) \in \operatorname { I n t } ( \mathcal { C } _ { t } )$ for all $t \in [ t _ { 0 } , t _ { f } ]$ . By adjusting the weight $\beta _ { t }$ , we can control the trade-off between editability and safety and $\lambda _ { t }$ can be used to control the smoothness of the control input. Path-safety is formally justified in Proposition 2 below. Such cost functional is considered in [11], where kernel method is used to approximate the value function and optimal control. In contrast, we seek to directly estimate the optimal control from the HJB equation (6) with the cost functional (9).

![](images/33622cbb539c7fc114997f814800b33ac5a1ad148c7b26f066a141b369b0fcf3.jpg)  
Figure 3: Visualization of text-guided image manipulation on CelebA-HQ. The row "+ MNC" shows the results when MNC is applied on OC-Flow.

Recall that the value function V is defined as (5) with J given by (9). We are able to derive the more explicit form of the HJB equation.

Proposition 1. Assume V is smooth and $\lambda _ { t } > 0$ . Then V satisfies

$$
- \partial _ { t } V - \nabla _ { x } V \cdot f + \frac { 1 } { 2 \lambda _ { t } } \| \nabla _ { x } V \| _ { 2 } ^ { 2 } - \beta _ { t } B ( x , t ) = 0 , \quad V ( x , t _ { f } ) = \Psi ( x ) ,\tag{10}
$$

and the optimal control is given by

$$
\begin{array} { r } { u ^ { \star } ( x , t ) = - \lambda _ { t } ^ { - 1 } \nabla _ { x } V ( x , t ) . } \end{array}\tag{11}
$$

Proof. See Appendix B.1.

Proposition 2. Assume V is smooth, $\mathcal { C } _ { t }$ varies not too rapidly, $B ( x , t )$ blows up fast enough as $x  \partial \mathcal { C } _ { t }$ , and $u ^ { \star }$ is given by (11). Suppose $x ( t _ { 0 } ) \in \mathrm { I n t } ( \mathcal { C } _ { t _ { 0 } } )$ , then the trajectory of the controlled system (2) under the control $u ^ { \star }$ satisfies x(t) ∈ Int(C<sub>t</sub>) for all $t \in [ t _ { 0 } , t _ { f } ]$

Proof. See Appendix B.2.

Remark 1. Proposition 2 establishes the path-safety ofthe optimal control to the costfunctional (9) under certain regularities. It does not automatically guarantee the path-safety ofthe approximated control derived from the HJB approximation in the next section, and of the output of Algorithm 1. In practice, one may set a safe guard (e.g., an MNC) that intervenes when necessary to ensure path-safety.

## 4.2 Tractable HJB Approximation

In light of Proposition 1, it suffices to calculate the gradient of the value function $\nabla _ { x } V$ in order to obtain the optimal control. However, solving the HJB equation (10) is generally intractable. Instead, we seek to directly estimate the optimal control without explicitly solving for V . Let us start by characterizing $\boldsymbol { \nabla _ { x } } \dot { \boldsymbol { V } }$ further.

Following Gao et al. [21], we introduce the following change of variables:

$$
\begin{array} { r l } & { s ( t ) : = \displaystyle \int _ { t } ^ { t _ { f } } \lambda _ { \tau } ^ { - 1 } \mathrm { d } \tau , \quad \widehat { V } ( y , s ( t ) ) : = V ( \Phi _ { t _ { f }  t } ^ { f } ( y ) , t ) , \quad y : = \Phi _ { t  t _ { f } } ^ { f } ( x ) , } \\ & { G _ { t } ( y ) : = \nabla _ { x } \Phi _ { t  t _ { f } } ^ { f } ( \Phi _ { t _ { f }  t } ^ { f } ( y ) ) \nabla _ { x } \Phi _ { t  t _ { f } } ^ { f } ( \Phi _ { t _ { f }  t } ^ { f } ( y ) ) ^ { \top } , \quad \Gamma _ { t } ( y ) : = \lambda _ { t } \beta _ { t } B ( \Phi _ { t _ { f }  t } ^ { f } ( y ) , t ) , } \end{array}\tag{12}
$$

where $\Phi _ { t _ { 1 }  t _ { 2 } } ^ { f }$ is the flow map induced by the vector field f as defined in the preliminaries. Then we have the following result.

Proposition 3. Under Eq. (12), Eq. (10) transforms to aforced Hamilton-Jacobi equation

$$
\partial _ { s } \widehat { V } ( y , s ) + \frac { 1 } { 2 } \nabla _ { y } \widehat { V } ( y , s ) ^ { \top } G _ { t } ( y ) \nabla _ { y } \widehat { V } ( y , s ) = \Gamma _ { t } ( y ) , \quad \widehat { V } ( y , 0 ) = \Psi ( y ) .\tag{13}
$$

Moreover,

$$
\nabla _ { \boldsymbol { x } } V ( \boldsymbol { x } , t ) = \nabla _ { \boldsymbol { x } } \Phi _ { t  t _ { f } } ^ { f } ( \boldsymbol { x } ) ^ { \top } \nabla _ { \boldsymbol { y } } \widehat { V } ( \boldsymbol { y } , s ( t ) ) .\tag{14}
$$

The expression (13) is still difficult to solve exactly in high dimension. Suppose $\Gamma _ { t } ( y ) \equiv \Gamma$ and $G _ { t } \equiv G$ are constant, then (13) admits an explicit solution given by the Hopf-Lax formula [17]:

$$
\widehat V ( y , s ) = s \Gamma + \operatorname* { i n f } _ { z } \Big \{ \Psi ( z ) + \frac { 1 } { 2 s } \| y - z \| _ { G ^ { - 1 } } ^ { 2 } \Big \} .
$$

Since PDE is locally well-approximated, we have

$$
\widehat { V } ( y , s ) \approx s \Gamma _ { t } ( y ) + \operatorname* { i n f } _ { z } \Big \{ \Psi ( z ) + \frac { 1 } { 2 s } \| y - z \| _ { G _ { t } ^ { - 1 } } ^ { 2 } \Big \} = : s \Gamma _ { t } ( y ) + U .
$$

If the infimum is attained at a unique point $y ^ { + } ( y )$ for all $y ,$ then $U$ is differentiable and its gradient is given by the Moreau-Yosida identity [44]:

$$
\nabla _ { y } U = \frac { 1 } { s } G _ { t } ^ { - 1 } ( y - y ^ { + } ( y ) ) .
$$

Consequently, by (11) and (14), the optimal control can be approximated by

$$
\begin{array} { r l r } {  { u ^ { \star } ( x , t ) \approx - \lambda _ { t } ^ { - 1 } \nabla _ { x } \Phi _ { t  t _ { f } } ^ { f } ( x ) ^ { \top } \Big ( s \nabla _ { y } \Gamma _ { t } ( y ) + \frac { 1 } { s } G _ { t } ^ { - 1 } ( y - y ^ { + } ( y ) ) \Big ) } } \\ & { } & { = - s \beta _ { t } \nabla _ { x } B ( x , t ) - \lambda _ { t } ^ { - 1 } \nabla _ { x } \Phi _ { t  t _ { f } } ^ { f } ( x ) ^ { \top } \frac { 1 } { s } G _ { t } ^ { - 1 } ( y - y ^ { + } ( y ) ) . } \end{array}\tag{15}
$$

If we adopt the first-order approximation $\Psi ( z ) \approx \Psi ( y ) + \nabla \Psi ( y ) \cdot ( z - y )$ and assume $G _ { t } \approx I$ , then the optimal control can be approximated by

$$
\boldsymbol { u } ^ { \star } ( \boldsymbol { x } , t ) \approx - s \beta _ { t } \nabla _ { \boldsymbol { x } } B ( \boldsymbol { x } , t ) - \lambda _ { t } ^ { - 1 } \nabla _ { \boldsymbol { x } } \Psi ( \boldsymbol { y } ) .\tag{16}
$$

Gao et al. [21] deal with a similar problem without considering path-safety. The introduction of the barrier function term turns an unforced HJB equation into a forced one, which enhance the difficulty to obtain the approximation. We summarize the procedure of AegisFlow in Algorithm 1.

Remark 2. In Algorithm 1, we implement a heuristic substitution that replaces the nominal flow $\Phi ^ { f }$ in (16) with the iteratively updated controlledflow $\Phi ^ { f + u ^ { k } }$ . This modification ensures that the terminal loss landscape is evaluated exactly along the controlled trajectory. As k increases and $u ^ { k }$ converges toward the optimal control, exploiting $\Phi ^ { f + u ^ { k } }$ substantially reduces the compounding linearization error that would otherwise arisefrom tracking solely along the unperturbedflow Φ<sup>f</sup>.

## 5 Experiments

## 5.1 Text-Guided Image Manipulation with Identity Preservation

Dataset and Evaluation. We conduct experiments on the celebrity face dataset CelebA-HQ [29] with text guidance {old, smile, sad, angry, curly hair}. Following the same setup as in [36], the alignment between the generated image and text prompt is evaluated by the CLIP model [43]. As in [30], we measure the perceptual similarity between the source and generated image by LPIPS [64] and identity similarity by ID, which is computed using ArcFace [12]. Implementation details are deferred to Appendix C.1.

Main Results. We benchmark AegisFlow against state-of-the-art methods on CelebA-HQ (Figure 3). The results are summarized in Table 1 and Figure 4. Our method achieves the best CLIP and LPIPS score, indicating that it improves text alignment while preserving subject identity better than prior methods. We also compare against the myopic safety baseline, i.e. applying the plug-and-play MNC to OC-Flow. While MNC attains the highest ID, it underperforms AegisFlow on both CLIP and LPIPS.

To better understand AegisFlow, we record the value of the function $h ( x , t )$ at each timestep, as shown in Figure 4. It can be seen that AegisFlow acts the most proactively to maintain a large margin from the boundary of the safe set. Although MNC and AegisFlow both renders safety for the most part of the trajectory, AegisFlow achieves better performance by maintaining a larger safety margin and thus allowing more aggressive optimization towards the target prompt. Comparisons on computational efficiency are deferred to Appendix D.1.

Table 1: Comparison of AegisFlow with state-of-the-art methods on CelebA-HQ. The best results are in bold. We also include the results of directly applying MNC for reference.
<table><tr><td>Method</td><td>CLIP↑</td><td>LPIPS ↓</td><td>ID↑</td></tr><tr><td> $\mathrm { C G } + \mathrm { R F } \left[ 1 3 \right]$ </td><td>0.292</td><td>0.346</td><td>0.643</td></tr><tr><td> $\mathrm { C G } + \mathrm { L D M } \left[ 1 3 \right]$ </td><td>0.298</td><td>0.383</td><td>0.513</td></tr><tr><td>DiffusionCLIP [30]</td><td>0.285</td><td>0.398</td><td>0.659</td></tr><tr><td> $\mathrm { S t y l e C L I P + e 4 e \ [ 4 2 ] }$ </td><td>0.267</td><td>0.359</td><td>0.704</td></tr><tr><td> $\mathrm { F l o w G r a d } + \mathrm { R F } \ [ 3 6 ]$ </td><td>0.299</td><td>0.302</td><td>0.737</td></tr><tr><td>OC-Flow [56]</td><td>0.302</td><td>0.207</td><td>0.732</td></tr><tr><td> $\mathrm { O C - F l o w } \left[ 5 6 \right] + \mathrm { M N C }$ </td><td>0.304</td><td>0.193</td><td>0.920</td></tr><tr><td>AegisFlow (Ours)</td><td>0.309</td><td>0.188</td><td>0.901</td></tr></table>

![](images/ad3a8b5cbfe71ad1191bcd0866299957355038d03541f35f200dc85f324e57e0.jpg)

![](images/1fe56ebe279521316a8ce6639ca1d29a9c81ecddec037a4030a67c3598cff8fe.jpg)  
Figure 4: Quantitative Analysis on CelebA-HQ. a. The value of the function $h ( x , t )$ at each timestep. The solid lines represent the mean value for the three methods and the shallow bands represent the range of variation. The x-axis is emphasized by a dash line since the forward invariance of $\backslash \mathcal { C } _ { t }$ requires $h ( x , t ) \geq 0$ for all t. b. AegisFlow achieves a more favorable balance between text alignment (CLIP) and structure preservation (LPIPS). Connected markers represents different hyperparameter settings (See Appendix D.2).

## 5.2 Maze Navigation with Safety Constraints

Dataset and Evaluation. We evaluate AegisFlow on a 2D maze navigation task with safety constraints. The experiment is conducted on maze\_large\_v1 of the Maze2D environment from D4RL [19]. Following Xiao et al. [57], Yang et al. [58], we introduce two obstacle-based safety constraints given by the following functions:

$$
\mathbf { B S 1 } : ( \frac { x - x _ { 0 } } { a } ) ^ { 2 } + ( \frac { y - y _ { 0 } } { b } ) ^ { 2 } \geq 1 , \quad \mathbf { B S 2 } : ( \frac { x - x _ { 0 } } { a } ) ^ { 4 } + ( \frac { y - y _ { 0 } } { b } ) ^ { 4 } \geq 1 .
$$

The minimum value of the above functions along the planned trajectory is recorded to evaluate the safety performance. We also report the D4RL normalized score, the average curvature κ and the average acceleration a of the trajectory. Implementation details are deferred to Appendix C.2.

Main Results. The results are summarized in Table 2. AegisFlow achieves the strongest safety margins (highest BS1/BS2) and the best generation quality among all methods, indicating safer and smoother plans. Compared with SafeFlowMatcher (which adopts MNC), AegisFlow improves both safety metrics and reduces trajectory curvature and acceleration. Figure 5 shows that AegisFlow maintains a larger safety buffer early in the rollout, which aligns with its non-myopic formulation. Comparisons on computational efficiency are deferred to Appendix D.1.

SafeFlowMatcher

AegisFlow

![](images/e71d3279ca052adb6a1bacc920c050d3bf5776721aa5ca69809865c24e4e74d3.jpg)  
t = 0.00

![](images/e3bed34919a251bf3cd63e03f508ca5a25d3bbbe88447ac829c07f2a7bd0d8e1.jpg)  
t = 0.25

![](images/84417caeafbdc10a50df0e90affabf50c2fd4535fa9943ae62484300664309f8.jpg)  
t = 0.50

![](images/489cf1dfb16aff71a1d01cae24e74f81a34c083603d117d6fb12e3ebd214cbfb.jpg)  
t = 0.75

![](images/c68d5f5409b02ac517a7f40015a3097e96b597d0feab2b5100e0a49444ba76d5.jpg)  
t = 1.00

Figure 5: Visualization of path generation process in Maze2D. Red circles represent the obstacles. AegisFlow behaves more proactively to maintain a large margin from the boundary of the safe set in the early stages of generation.  
Table 2: Comparison of AegisFlow with state-of-the-art methods on Maze2D. The best results are in bold. BS1 and BS2 being positive (≥ 0) indicates the trajectory is safe with respect to the corresponding constraint. a ± b means a is the mean and b is the standard deviation over 100 trials.
<table><tr><td>Method</td><td>BS1 ↑ BS2↑</td><td>Score ↑</td><td>κ↓</td><td>a↓</td></tr><tr><td>Diffuser [27]</td><td>-0.825 -0.784</td><td>1.572±0.288</td><td>77.04±4.30</td><td>86.68±3.81</td></tr><tr><td>FlowMatcher</td><td>-0.324 -0.904</td><td>1.632±0.003</td><td>73.51±1.02</td><td>88.45±0.60</td></tr><tr><td>ROS-SafeDiffuser [57]</td><td>0.010 0.010</td><td>1.435±0.502</td><td>75.15±6.67</td><td>422.87±86.70</td></tr><tr><td>RES-SafeDiffuser [57]</td><td>0.010 0.010</td><td>1.442±0.451</td><td>80.30±13.06</td><td>398.17±1060.86</td></tr><tr><td>TVS-SafeDiffuser [57]</td><td>-0.003 -0.003</td><td>1.506±0.405</td><td>78.72±7.80</td><td>124.51±34.22</td></tr><tr><td>SafeFlowMatcher [58]</td><td>0.020 0.041</td><td>1.632±0.003</td><td>51.97±2.55</td><td>65.71±2.45</td></tr><tr><td>AegisFlow (Ours)</td><td>0.049 0.071</td><td>1.635±0.003</td><td>44.72±2.10</td><td>50.29±6.17</td></tr></table>

Table 3: Comparison of AegisFlow with state-of-the-art methods on Avoiding. The best results are in bold. a ± b means a is the mean and b is the standard deviation over 100 trials.
<table><tr><td>Method</td><td>Timesteps ↓</td><td>Goal ↑</td><td>Safe ↑</td><td>Violation ↓</td></tr><tr><td>Diffuser [27]</td><td>76.7±12.7</td><td>1.00</td><td>0.07</td><td>17.8±12.1</td></tr><tr><td>Flowmatcher</td><td>64.2±9.4</td><td>1.00</td><td>0.22</td><td>14.8±11.5</td></tr><tr><td>DOM [22]</td><td>79.1±13.9</td><td>1.00</td><td>0.96</td><td>0.1±0.5</td></tr><tr><td>Model-free [46]</td><td>76.1±12.1</td><td>0.99</td><td>0.07</td><td>18.0±12.0</td></tr><tr><td>DPCC-C [47]</td><td>69.0±12.9</td><td>1.00</td><td>0.98</td><td>0.0±0.3</td></tr><tr><td>AegisFlow (Ours)</td><td>62.6±2.0</td><td>1.00</td><td>0.98</td><td>0.1±0.5</td></tr></table>

## 5.3 Robotic Manipulation with Safety Constraints

Dataset and Evaluation. We implement AegisFlow on a robotic manipulation task with novel test-time constraints. The experiment is conducted on the Avoiding environment from D3IL [28]. The task is to control a robot manipulator to reach the target region while avoiding collision with obstacles. We also introduce the same novel test-time constraints as in Römer et al. [47]. We report the success rate of reaching the target (goal), of reaching the target while satisfying the safety constraints (safe), the number of timesteps the manipulator takes to reach the target (timesteps), and the number of timesteps the manipulator violates the safety constraints (violation). Implementation details are deferred to Appendix C.3.

![](images/3073d8eaaa090f18fb601d812f2ed98d1a4f5352057b441643cdda1bebef1f26.jpg)  
Figure 6: Qualitative Results for Robotic Manipulation. a. Visualization of the robotic manipulation task. The object is to reach the green line while avoiding collision with the red obstacles. b. Generated trajectories by Flowmatcher. c. Generated trajectories by AegisFlow. The novel test-time constraints are that the robot should not enter the blue regions.

Main Results. Table 3 shows that AegisFlow maintains the best goal success and safety rate while achieving the fewest timesteps. Compared with DPCC-C, AegisFlow maintains the same safety level with faster task completion. Relative to unsafe baselines, AegisFlow substantially reduces violations while preserving goal success. Figure 6 visualizes that AegisFlow respects the novel constraints by steering trajectories away from forbidden regions.

## 6 Conclusion

In this paper, we presented AegisFlow, a training-free guidance framework that addresses the limitations of myopic controllers in path-safe guided flow matching. By casting guided generation as a safety-critical optimal control problem, AegisFlow integrates constraints into the cost functional. A tractable HJB approximation is derived to obtain the optimal control. Comprehensive empirical evaluations across diverse domains demonstrate the effectiveness of AegisFlow. Future works may improve HJB approximation, enhance computational efficiency, and extend this control-theoretic framework to broader settings and more complex modalities.

## 7 Acknowledgements

This work was supported by National Natural Science Foundation of China (62325305), and the BNRist project (No. BNR2024TD03003).

## References

[1] Michael Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. Journal ofMachine Learning Research, 26(209): 1–80, 2025.

[2] Aaron D Ames, Kevin Galloway, Koushil Sreenath, and Jessy W Grizzle. Rapidly exponentially stabilizing control lyapunov functions and hybrid zero dynamics. IEEE Transactions on Automatic Control, 59(4):876–891, 2014.

[3] Aaron D Ames, Xiangru Xu, Jessy W Grizzle, and Paulo Tabuada. Control barrier function based quadratic programs for safety critical systems. IEEE Transactions on Automatic Control, 62(8):3861–3876, 2016.

[4] Aaron D Ames, Samuel Coogan, Magnus Egerstedt, Gennaro Notomista, Koushil Sreenath, and Paulo Tabuada. Control barrier functions: Theory and applications. In 2019 18th European control conference (ECC), pages 3420–3431. IEEE, 2019.

[5] Heli Ben-Hamu, Omri Puny, Itai Gat, Brian Karrer, Uriel Singer, and Yaron Lipman. D-flow: Differentiating through flows for controlled generation. In International Conference on Machine Learning, 2024.

[6] Julius Berner, Lorenz Richter, and Karen Ullrich. An optimal control perspective on diffusionbased generative modeling. Transactions on Machine Learning Research, 2024.

[7] Manuel Brack, Felix Friedrich, Katharia Kornmeier, Linoy Tsaban, Patrick Schramowski, Kristian Kersting, and Apolinário Passos. Ledits++: Limitless image editing using text-toimage models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8861–8870, 2024.

[8] Joao Carvalho, An T Le, Mark Baierl, Dorothea Koert, and Jan Peters. Motion planning diffusion: Learning and planning of robot motions with diffusion models. In 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 1916–1923. IEEE, 2023.

[9] Chang Chen, Fei Deng, Kenji Kawaguchi, Caglar Gulcehre, and Sungjin Ahn. Simple hierarchical planning with diffusion. In International Conference on Learning Representations, 2024.

[10] Jacob K Christopher, Stephen Baek, and Ferdinando Fioretto. Constrained synthesis with projected diffusion models. Advances in Neural Information Processing Systems, 37:89307– 89333, 2024.

[11] Max H Cohen and Calin Belta. Approximate optimal control for safety-critical systems with control barrier functions. In 2020 59th IEEE conference on decision and control (CDC), pages 2062–2067. IEEE, 2020.

[12] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4690–4699, 2019.

[13] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

[14] Haoran Ding, Noémie Jaquier, Jan Peters, and Leonel Rozo. Fast and robust visuomotor riemannian flow matching policy. IEEE Transactions on robotics, 2025.

[15] Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, and Ricky TQ Chen. Adjoint matching: Fine-tuning flow and diffusion generative models with memoryless stochastic optimal control. In International Conference on Learning Representations, 2025.

[16] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In International Conference on Machine Learning, 2024.

[17] Lawrence C Evans. Partial differential equations, volume 19. American mathematical society, 2022.

[18] Randy Freeman and Petar V Kokotovic. Robust nonlinear control design: state-space and Lyapunov techniques. Springer Science & Business Media, 2008.

[19] Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine. D4rl: Datasets for deep data-driven reinforcement learning. arXiv preprint arXiv:2004.07219, 2020.

[20] Darshan Gadginmath, Ahmed Allibhoy, and Fabio Pasqualetti. Provably safe generative sampling with constricting barrier functions. Transactions on Machine Learning Research, 2026.

[21] Weiguo Gao, Ming Li, and Qianxiao Li. Terminally constrained flow-based generative models from an optimal control perspective. arXiv preprint arXiv:2601.09474, 2026.

[22] Giorgio Giannone, Akash Srivastava, Ole Winther, and Faez Ahmed. Aligning optimization trajectories with diffusion models for constrained design generation. Advances in neural information processing systems, 36:51830–51861, 2023.

[23] Aaron Havens, Benjamin Kurt Miller, Bing Yan, Carles Domingo-Enrich, Anuroop Sriram, Brandon Wood, Daniel Levine, Bin Hu, Brandon Amos, Brian Karrer, et al. Adjoint sampling: Highly scalable diffusion samplers via adjoint matching. In International Conference on Machine Learning, 2025.

[24] Amir Hertz, Kfir Aberman, and Daniel Cohen-Or. Delta denoising score. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2328–2337, 2023.

[25] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[26] Inbar Huberman-Spiegelglas, Vladimir Kulikov, and Tomer Michaeli. An edit friendly ddpm noise space: Inversion and manipulations. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12469–12478, 2024.

[27] Michael Janner, Yilun Du, Joshua B Tenenbaum, and Sergey Levine. Planning with diffusion for flexible behavior synthesis. In International Conference on Machine Learning, 2022.

[28] Xiaogang Jia, Denis Blessing, Xinkai Jiang, Moritz Reuss, Atalay Donat, Rudolf Lioutikov, and Gerhard Neumann. Towards diverse behaviors: A benchmark for imitation learning with human demonstrations. In International Conference on Learning Representations, 2024.

[29] Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of gans for improved quality, stability, and variation. In International Conference on Learning Representations, 2018.

[30] Gwanghyun Kim, Taesung Kwon, and Jong Chul Ye. Diffusionclip: Text-guided diffusion models for robust image manipulation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2426–2435, 2022.

[31] Juil Koo, Chanho Park, and Minhyuk Sung. Posterior distillation sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13352–13361, 2024.

[32] Vladimir Kulikov, Matan Kleiner, Inbar Huberman-Spiegelglas, and Tomer Michaeli. Flowedit: Inversion-free text-based editing using pre-trained flow models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19721–19730, 2025.

[33] Zhiwen Li, Die Chen, Mingyuan Fan, Cen Chen, Yaliang Li, Yanhao Wang, and Wenmeng Zhou. Responsible diffusion models via constraining text embeddings within safe regions. In Proceedings ofthe ACM on Web Conference 2025, pages 1588–1601, 2025.

[34] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023.

[35] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In International Conference on Learning Representations, 2023.

[36] Xingchao Liu, Lemeng Wu, Shujian Zhang, Chengyue Gong, Wei Ping, and Qiang Liu. Flowgrad: Controlling the output of generative odes with gradients. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24335–24344, 2023.

[37] Hao Luan, Yi Xian Goh, See-Kiong Ng, and Chun Kai Ling. Projected coupled diffusion for test-time constrained joint generation. In International Conference on Learning Representations, 2026.

[38] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations, 2022.

[39] Raghav Mishra and Ian R Manchester. Eb-mbd: Emerging-barrier model-based diffusion for safe trajectory optimization in highly constrained environments. arXiv preprint arXiv:2510.07700, 2025.

[40] Kazuki Mizuta and Karen Leung. Cobl-diffusion: Diffusion-based conditional robot planning in dynamic environments using control barrier and lyapunov functions. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 13801–13808. IEEE, 2024.

[41] Hyelin Nam, Gihyun Kwon, Geon Yeong Park, and Jong Chul Ye. Contrastive denoising score for text-guided latent diffusion image editing. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 9192–9201, 2024.

[42] Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or, and Dani Lischinski. Styleclip: Text-driven manipulation of stylegan imagery. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2085–2094, 2021.

[43] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763, 2021.

[44] R Tyrrell Rockafellar and Roger JB Wets. Variational analysis. Springer, 1998.

[45] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

[46] Ralf Römer, Lukas Brunke, Martin Schuck, and Angela P Schoellig. Safe offline reinforcement learning using trajectory-level diffusion models. In ICRA 2024 Workshop Back to the Future: Robot Learning Going Probabilistic, 2024.

[47] Ralf Römer, Alexander von Rohr, and Angela P Schoellig. Diffusion predictive control with constraints. arXiv preprint arXiv:2412.09342, 2024.

[48] Litu Rout, Yujia Chen, Nataniel Ruiz, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Semantic image inversion and editing using rectified stochastic differential equations. In International Conference on Learning Representations, 2025.

[49] Litu Rout, Yujia Chen, Nataniel Ruiz, Abhishek Kumar, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Rb-modulation: Training-free stylization using reference-based modulation. In International Conference on Learning Representations, 2025.

[50] Dvir Samuel, Barak Meiri, Haggai Maron, Yoad Tewel, Nir Darshan, Shai Avidan, Gal Chechik, and Rami Ben-Ari. Lightning-fast image inversion and editing for text-to-image diffusion models. In International Conference on Learning Representations, 2025.

[51] Patrick Schramowski, Manuel Brack, Björn Deiseroth, and Kristian Kersting. Safe latent diffusion: Mitigating inappropriate degeneration in diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22522–22531, 2023.

[52] Jiaming Song, Qinsheng Zhang, Hongxu Yin, Morteza Mardani, Ming-Yu Liu, Jan Kautz, Yongxin Chen, and Arash Vahdat. Loss-guided diffusion models for plug-and-play controllable generation. In International Conference on Machine Learning, pages 32483–32498. PMLR, 2023.

[53] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021.

[54] Utkarsh Utkarsh, Pengfei Cai, Alan Edelman, Rafael Gomez-Bombarelli, and Christopher Rackauckas. Physics-constrained flow matching: Sampling generative models with hard constraints. Advances in Neural Information Processing Systems, 38:160217–160252, 2026.

[55] Francisco Vargas, Will Grathwohl, and Arnaud Doucet. Denoising diffusion samplers. In International Conference on Learning Representations, 2023.

[56] Luran Wang, Chaoran Cheng, Yizhen Liao, Yanru Qu, and Ge Liu. Training free guided flow matching with optimal control. In International Conference on Learning Representations, 2025.

[57] Wei Xiao, Tsun-Hsuan Wang, Chuang Gan, Ramin Hasani, Mathias Lechner, and Daniela Rus. Safediffuser: Safe planning with diffusion probabilistic models. In International Conference on Learning Representations, 2023.

[58] Jeongyong Yang, Seunghwan Jang, and SooJean Han. Safeflowmatcher: Safe and fast planning using flow matching with control barrier functions. International Conference on Learning Representations, 2026.

[59] Xiaofeng Yang, Cheng Chen, Fayao Liu, Guosheng Lin, et al. Text-to-image rectified flow as plug-and-play priors. In International Conference on Learning Representations, volume 2025, pages 13896–13920, 2025.

[60] Hui Yuan, Kaixuan Huang, Chengzhuo Ni, Minshuo Chen, and Mengdi Wang. Reward-directed conditional diffusion: Provable distribution estimation and reward improvement. Advances in Neural Information Processing Systems, 36:60599–60635, 2023.

[61] Stefano Zampini, Jacob K Christopher, Luca Oneto, Davide Anguita, and Ferdinando Fioretto. Training-free constrained generation with stable diffusion models. Advances in Neural Information Processing Systems, 38:27285–27316, 2026.

[62] Jichen Zhang, Liqun Zhao, Antonis Papachristodoulou, and Jack Umenberger. Constrained diffusers for safe planning and control. Advances in Neural Information Processing Systems, 38:34965–34998, 2026.

[63] Qinsheng Zhang and Yongxin Chen. Path integral sampler: a stochastic control approach for sampling. In International Conference on Learning Representations, 2022.

[64] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.

[65] Hanyang Zhao, Haoxian Chen, Ji Zhang, David D Yao, and Wenpin Tang. Score as action: Finetuning diffusion generative models by continuous-time reinforcement learning. In International Conference on Machine Learning, 2025.

[66] Kaizhen Zhu, Mokai Pan, Yuexin Ma, Yanwei Fu, Jingyi Yu, Jingya Wang, and Ye Shi. Unidb: A unified diffusion bridge framework via stochastic optimal control. In International Conference on Machine Learning, 2025.

## A Background on Control Barrier Functions

## A.1 Control Barrier Functions

Control Barrier Functions (CBFs) [3, 4] are a powerful tool to enforce hard safety constraints in control theory. Specifically, given a control system

$$
\dot { x } = f ( x ) + g ( x ) u ,\tag{17}
$$

we want to ensure that the state $x ( t )$ remains within a safe set

$$
\mathcal { C } : = \{ x \in \mathbb { R } ^ { n } : h ( x ) \geq 0 \}\tag{18}
$$

for all $t \in [ t _ { 0 } , t _ { f } ]$ . If so, we say that the set $\mathcal { C }$ isforward invariant. We also allow $\mathcal { C }$ to be time-varying, i.e. $\mathcal { C } _ { t } : = \left\{ x \in \bar { \mathbb { R } } ^ { n } : h ( x , t ) \geq 0 \right\}$ , and in this case we say $\mathcal { C } _ { t }$ is forward invariant if $x ( t ) \in \mathcal { C } _ { t }$ for all $t \in [ t _ { 0 } , \dot { t } _ { f } ]$ . The intuition of CBFs is to ensure that when evaluated along the trajectories of the system, the gradient of h is nonnegative whenever $h ( x ) = 0 ,$ which prevents the system from leaving the safe set. We shall introduce the formal definition of CBFs as follows.

Definition 1 (Extended Class $\kappa _ { \infty }$ Function). A function $\alpha : \mathbb { R }  \mathbb { R }$ is an extended class $\kappa _ { \infty }$ function ifit is strictly increasing and with $\alpha ( 0 ) = 0$

Definition 2 (Control Barrier Function). Let $\mathcal { C } \subset \mathcal { D } \subset \mathbb { R } ^ { n }$ be with theform of (18) and $h : \mathcal { D } $ R be a continuously differentiablefunction, then h is a control barrierfunction (CBF) ifthere exists an extended class $\kappa _ { \infty }$ function α such that for the control system (17), we have

$$
\operatorname* { s u p } _ { u \in U } [ \nabla h ( x ) \cdot ( f ( x ) + g ( x ) u ) + \alpha ( h ( x ) ) ] \ge 0\tag{19}
$$

for all $x \in \mathcal { D }$ . For a CBF h, define the set of safe controls at x as

$$
K _ { C B F } ( x ) : = \{ u \in U : \nabla h ( x ) \cdot ( f ( x ) + g ( x ) u ) + \alpha ( h ( x ) ) \geq 0 \} .
$$

Theorem 1 (Ames et al. [3]). Let $\mathcal { C } \subset \mathcal { D } \subset \mathbb { R } ^ { n }$ be with the form of (18) and $h : { \mathcal { D } }  \mathbb { R }$ be a continuously differentiable function. If h is a CBF and $\nabla h ( x ) \ \ne \ 0 , \forall x \in \partial { \mathcal { C } }$ , then any Lipschitz continuous controller $u ( x ) \in K _ { C B F } ( x )$ renders the set C forward invariant. Moreover, $\mathcal { C }$ is asymptotically stable, i.e. $i f h ( x ( t ) ) < 0 f o r$ some $t ,$ then $h ( x ( t ) ) \to 0 a s t \to \infty .$

The result above can be easily generalized to time-varying safe sets by introducing an auxiliary state to represent time.

## A.2 Min-norm Controller

Min-norm controller [18, 2] is a widely used method to render forward invariance in control theory. Given a dynamical system and a nominal control $u _ { \mathrm { n o m } }$ , by imposing an additive safe control $u _ { \mathrm { s a f e } }$ , we seek to ensure that $\dot { \mathcal { C } } _ { t }$ under

$$
\dot { x } = f ( x , t ) + u _ { \mathrm { n o m } } ( t ) + u _ { \mathrm { s a f e } } ( t )
$$

is forward invariant.

It is desired that $u _ { \mathrm { s a f e } }$ can be as small as possible in order to preserve the editability of the original nominal control, which naturally leads to the following optimization problem at each timestep:

$$
\begin{array} { r l } { \underset { u _ { \mathrm { s a f e } } ( t ) , \xi } { \operatorname* { m i n } } } & { \frac { 1 } { 2 } \| u _ { \mathrm { s a f e } } \| _ { 2 } ^ { 2 } + \rho _ { t } \xi } \\ { \mathrm { s . t . } } & { a _ { t } \cdot u _ { \mathrm { s a f e } } ( t ) + b _ { t } + \xi \geq 0 , \quad \xi \geq 0 . } \end{array}\tag{20}
$$

where we introduce the slack variable $\xi$ to relax the CBF constraint for the sake of stability and robustness, and we denote for brevity

$$
a _ { t } : = \nabla _ { x } h ( x , t ) , \quad b _ { t } : = \nabla _ { x } h ( x , t ) \cdot ( f ( x , t ) + u _ { \mathrm { n o m } } ( t ) ) + \partial _ { t } h ( x , t ) + \alpha ( h ( x , t ) ) .
$$

Min-norm controller imposes the safe control $u _ { \mathrm { s a f e } } ( t )$ by solving the optimization problem (20).

## B Proof of the Theoretical Results

## B.1 Proof of Proposition 1

Proof. Substitute the cost functional (9) and the dynamics (2) into the HJB equation (6), we have

$$
- \partial _ { t } V = \operatorname* { m i n } _ { u \in U } \left\{ \frac { \lambda _ { t } } { 2 } \| u \| _ { 2 } ^ { 2 } + \beta _ { t } B + \nabla V \cdot ( f + u ) \right\} , \quad V ( t _ { f } , x ) = \Psi ( x ) .
$$

The term inside the minimization is a quadratic function of $u ,$ which allows us to derive the explicit form of the value function and the optimal control.

$$
- \partial _ { t } V = \nabla _ { x } V \cdot f + \beta _ { t } B - \frac { 1 } { 2 \lambda _ { t } } \| \nabla _ { x } V \| _ { 2 } ^ { 2 } ,
$$

$$
\boldsymbol { u } ^ { \star } ( \boldsymbol { x } , t ) = - \lambda _ { t } ^ { - 1 } \nabla _ { \boldsymbol { x } } V ( \boldsymbol { x } , t ) ,
$$

which completes the proof.

## B.2 Proof of Proposition 2

Proof. Suppose $\mathcal { C } _ { t }$ is not forward invariant. Let $\bar { t } : = \operatorname* { i n f } \{ t \in [ t _ { 0 } , t _ { f } ] : x ( t ) \notin \mathcal { C } _ { t } \}$ be the first time the trajectory leaves the safe set. Since $x ( t _ { 0 } ) \in \mathrm { I n t } ( C _ { t _ { 0 } } )$ , we have $\bar { t } > t _ { 0 }$ and $h ( x ( \bar { t } ) , \bar { t } ) = 0$ . Taking derivative of $V$ along the trajectory gives

$$
\frac { \mathrm { d } } { \mathrm { d } t } V ( x ( t ) , t ) = \nabla _ { x } V \cdot ( f + u ^ { \star } ) + \partial _ { t } V \le - \beta _ { t } B ( x ( t ) , t ) ,
$$

where the inequality follows from (10) and (11). Since $\beta _ { t } \geq 0$ and $B ( x ( t ) , t ) \geq 0$ for all $t \in [ t _ { 0 } , \bar { t } )$ we have $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } t } V ( x ( t ) , t ) \le 0 } \end{array}$ . On the other hand, since $\mathcal { C } _ { t }$ varies not too rapidly and $B ( x , t )$ blows up fast enough as $x  \partial \mathcal { C } _ { t }$ , we have $V ( x ( t ) , t ) \to \infty$ as $t  \bar { t } .$ . This contradicts the fact that $V$ non-increasing along the trajectory. □

Remark 3. For example, suppose $B ( x , t ) \geq \frac { c } { h ( x , t ) ^ { 2 } }$ for some $c > 0$ and $d ( t ) : = \mathrm { d i s t } ( x ( t ) , \partial \mathcal { C } _ { t } )$ is L-Lipschitz. Suppose $\begin{array} { r } { \int _ { t _ { 0 } } ^ { t } B ( x ( t ) , t ) \mathrm { d } t < \infty } \end{array}$ , then it follows that $d ^ { - 1 } \in L ^ { 2 } ( [ t _ { 0 } , \bar { t } ] )$ . On the other hand, $\dot { x } = f + u ^ { \star } \in L ^ { 2 } .$ Thus, x is absolutely continuous and hence d is absolutely continuous. Moreover, $| \dot { d } | \leq \| \dot { x } \| + L \in L ^ { 2 }$ . By the Cauchy-Schwarz inequality, it follows that $| \dot { d } | / d \in L ^ { 1 } ( [ t _ { 0 } , \bar { t } ] )$ which implies that log d is absolutely continuous. However, since $d ( t ) \to 0 a s t \to \bar { t } ,$ itfollows that log $d ( t ) \to - \infty a s t \to \bar { t } ,$ absurd.

## B.3 Proof of Proposition 3

Proof. Eq. (14) can be derived by the chain rule and the definition of the flow map. Then it follows that

$$
\| \nabla _ { \boldsymbol { x } } V ( \boldsymbol { x } , t ) \| _ { 2 } ^ { 2 } = \nabla _ { \boldsymbol { y } } \widehat { V } ( \boldsymbol { y } , \boldsymbol { s } ) ^ { \top } \boldsymbol { G } _ { t } ( \boldsymbol { y } ) \nabla _ { \boldsymbol { y } } \widehat { V } ( \boldsymbol { y } , \boldsymbol { s } ) .
$$

Notice that $\Phi _ { t  t _ { f } } ^ { f } ( x ( t ) ) = x ( t _ { f } ) =$ const., taking the time derivative on both sides gives

$$
\nabla _ { x } \Phi _ { t  t _ { f } } ^ { f } ( x ( t ) ) \cdot f + \partial _ { t } \Phi _ { t  t _ { f } } ^ { f } ( x ( t ) ) = 0 .
$$

Taking the time derivative on both sides of $\widehat V ( \Phi _ { t  t _ { t } } ^ { f } ( x ) , s ( t ) ) = V ( x , t )$ , we have

$$
\begin{array} { r l } & { \partial _ { t } V ( x , t ) = \partial _ { s } \widehat { V } ( y , s ( t ) ) \cdot s ^ { \prime } ( t ) + \nabla _ { y } \widehat { V } ( y , s ( t ) ) \cdot \partial _ { t } \Phi _ { t  t _ { f } } ^ { f } ( x ) } \\ & { \qquad = - \lambda _ { t } ^ { - 1 } \partial _ { s } \widehat { V } ( y , s ( t ) ) - \nabla _ { y } \widehat { V } ( y , s ( t ) ) \cdot \nabla _ { x } \Phi _ { t  t _ { f } } ^ { f } ( x ) \cdot f . } \end{array}
$$

Substituting the above identities into (10) gives (13), which completes the proof.

## C Implementation Details

## C.1 Text-Guided Image Manipulation with Identity Preservation

Following [36, 56], we adopt the pretrained Rectified Flow (RF) [35] as the generative prior. To ensure identity preservation, we define the safe set as $\mathcal { C } _ { t } : = \{ x : h ( x , t ) \geq 0 \}$ , with

$$
h ( x , t ) : = \epsilon _ { t } - [ \mathrm { d i s t } ( x , x _ { \mathrm { s r c } } ) ] ^ { 2 } ,
$$

where dist $( \cdot , \cdot )$ is the ArcFace distance, $x _ { \mathrm { s r c } }$ is the source image, and $\epsilon _ { t }$ is a time-varying margin. Based on the dataset statistics, we set

$$
\epsilon _ { t } : = \epsilon _ { 1 } + \frac { \epsilon _ { 0 } - \epsilon _ { 1 } } { 1 + \exp ( \kappa _ { \epsilon } ( t - \tau _ { 0 } ) ) } ,
$$

with $\epsilon _ { 0 } = 2 . 5 , \epsilon _ { 1 } = 0 . 2 , \kappa _ { \epsilon } = 1 0 , \tau _ { 0 } = 0 . 5 5$ . Note that $\epsilon _ { \mathrm { 0 } }$ is set to be larger than the maximum ArcFace distance between any image and its inversion to ensure the condition of Proposition 2 is satisfied, i.e. $x _ { 0 } \in \mathrm { I n t } ( \mathcal { C } _ { 0 } )$ . The barrier function is set to be

$$
B ( x , t ) : = \frac { \sigma \left( - \kappa _ { B } h ( x , t ) \right) } { h ( x , t ) } ,
$$

where $\sigma ( \cdot )$ is the sigmoid function, $\kappa _ { B } = 5 ,$ and $\delta = 0 . 1 5$ . With such a barrier function, Proposition 2 guarantees the ArcFace similarity between the generated image and the source image would be larger than $\begin{array} { r } { 1 - \frac { \epsilon _ { t _ { f } } + \delta } { 2 } \approx 0 . 8 2 5 } \end{array}$ , which is aligned with the empirical results in Table 1.

The terminal loss is defined as $\Psi ( x ) : = - 0 . 7 \cdot \mathrm { C L I P } ( x , c )$ , where c is the target text prompt. The schedules are set to be $\lambda _ { t } \equiv 0 . 6$ and $\beta _ { t } \equiv 0 . 1$ . Following Wang et al. [56], we set $M = 1 5 , \bar { \eta } = 2 . 5$ and $\nu = 0 . 9 9 5$

For the min-norm controller, we adopt the same safe set with $\rho _ { t } = 1$ for all t. The $\kappa _ { \infty }$ function α is set to be $\alpha ( r ) = \gamma \cdot r$ with $\gamma = 0 . 5$ . The experiments are conducted on 4 NVIDIA GeForce RTX 4090 GPUs with 24GB memory.

## C.2 Maze Navigation with Safety Constraints

We train a flow matching model to be the generative prior. The training strategy is summarized in Table 4. The barrier function is set to be $\begin{array} { r } { B ( \overline { { \tau } } , t ) : = \sum _ { x \in \tau } \left[ ( h _ { 1 } ( x ) - t + \overline { { \delta } } ) ^ { - 1 } + \overline { { 2 } } ( h _ { 2 } ( x ) - t + \delta ) ^ { - 1 } \right] } \end{array}$ where $h _ { 1 }$ and $h _ { 2 }$ are the two functions defined in BS1 and BS2, respectively, and $\delta = 0 . 0 5$ . Note that the term corresponding to BS2 is weighted more heavily because it is harder to satisfy [57]. Since score is computed via interaction with the environment and thus is non-differentiable, we set the terminal loss to be a linear combination of the curvature and acceleration, i.e. $\Psi ( \tau ) : = \qquad $ $0 . 0 1 ( \lambda _ { \kappa } \kappa ( \tau ) + \lambda _ { a } a ( \tau ) )$ , where τ is the planned trajectory, $\lambda _ { \kappa } = 0 . 6$ and $\lambda _ { a } = 0 . 4$ . The schedule is set to be $\lambda _ { t } \equiv 2 , \beta _ { t } \equiv 0 . 0 1 , M = 1 5 , \eta = 0 . 0 5$ and $\nu = 0 . 9 9 5$ . The experiments are conducted on 4 NVIDIA GeForce RTX 4090 GPUs with 24GB memory.

Table 4: Training strategy for the flow matching model in maze navigation.
<table><tr><td>U-Net channels Loss type Training steps Steps per epoch Batch size Learning rate</td><td> $( 1 , 4 , 8 ) \times 3 2$  L2 1M 5K 64</td></tr></table>

## C.3 Robotic Manipulation with Safety Constraints

We train a flow matching model to be the generative prior. The barrier function is set to be $B ( \tau , t ) : =$ $\begin{array} { r l } { \sum _ { x \in \tau } \left[ ( h _ { 1 } ( x ) - t + \delta ) ^ { - 1 } + ( h _ { 2 } ( x ) - t + \bar { \delta } ) ^ { - 1 } \right] } \end{array}$ , where $h _ { 1 }$ and $h _ { 2 }$ are the two functions corresponding

Table 5: Computational efficiency comparisons. $a \pm b$ means a is the mean and b is the standard deviation over 1000 trials (CelebA-HQ) or 100 trials (Maze2D).
<table><tr><td>Dataset</td><td>Method</td><td>Computation Time (s)</td></tr><tr><td rowspan="3">CelebA-HQ</td><td>OC-Flow</td><td> $4 2 1 . 0 2 { \pm } 2 5 . 5 6 $ </td></tr><tr><td> $\mathrm { O C - F l o w + M N C }$ </td><td> $5 1 8 . 2 6 { \pm } 2 5 . 5 4 $ </td></tr><tr><td>AegisFlow</td><td> $4 3 3 . 8 1 \pm 2 5 . 5 7$ </td></tr><tr><td rowspan="3">Maze2D</td><td>SafeDiffuser</td><td> $4 . 7 8 { \pm } 0 . 4 2$ </td></tr><tr><td>SafeFlowmatcher</td><td> $4 . 2 1 \pm 0 . 3 9$ </td></tr><tr><td>AegisFlow</td><td> $4 4 . 1 5 { \pm } 1 . 9 6 $ </td></tr></table>

Table 6: Detailed metrics comparisons for text-guided image manipulation on CelebA-HQ.
<table><tr><td>Method</td><td>Hyperparameters</td><td>CLIP↑</td><td>LPIPS↓</td><td> $\mathrm { I D } \uparrow$ </td></tr><tr><td rowspan="2">FlowGrad</td><td>+RF</td><td>0.299</td><td>0.302</td><td>0.737</td></tr><tr><td>+LDM</td><td>0.294</td><td>0.298</td><td>0.743</td></tr><tr><td rowspan="4">OC-Flow</td><td> $\alpha = 0 . 6$ </td><td>0.295</td><td>0.119</td><td>0.822</td></tr><tr><td> $\alpha = 0 . 7$ </td><td>0.302</td><td>0.207</td><td>0.732</td></tr><tr><td> $\alpha = 0 . 8$ </td><td>0.306</td><td>0.192</td><td>0.722</td></tr><tr><td> $\alpha = 0 . 9$ </td><td>0.310</td><td>0.244</td><td>0.662</td></tr><tr><td rowspan="3">OC-Flow + MNC</td><td> $\alpha = 0 . 7$ </td><td>0.299</td><td>0.152</td><td>0.937</td></tr><tr><td> $\alpha = 0 . 8$ </td><td>0.304</td><td>0.193</td><td>0.920</td></tr><tr><td> $\alpha = 0 . 9$ </td><td>0.309</td><td>0.242</td><td>0.901</td></tr><tr><td rowspan="5">AegisFlow</td><td> $\lambda = 1 . 0$ </td><td>0.299</td><td>0.152</td><td>0.903</td></tr><tr><td> $\lambda = 0 . 8$ </td><td>0.303</td><td>0.167</td><td>0.901</td></tr><tr><td> $\lambda = 0 . 6$ </td><td>0.309</td><td>0.188</td><td>0.901</td></tr><tr><td> $\lambda = 0 . 5$ </td><td>0.312</td><td>0.205</td><td>0.902</td></tr><tr><td> $\lambda = 0 . 4$ </td><td>0.316</td><td>0.224</td><td>0.903</td></tr></table>

to the halfspace and circular constraints, respectively, and $\delta = 0 . 0 5$ . Since timesteps is computed via interation with the environment and thus is non-differentiable, we set the terminal loss to be the curvature of the path, i.e. $\Psi ( \tau ) : = 0 . 1 \kappa ( \tau )$ , where τ is the planned trajectory. The schedule is set to be $\lambda _ { t } \equiv 1 . 5 , \bar { \beta } _ { t } \equiv 0 . 5 , \dot { M } = 7 , \eta = \mathrm { { 0 . 2 } }$ and $\nu = 0 . 9 9 5$ . The experiments are conducted on 4 NVIDIA GeForce RTX 4090 GPUs with 24GB memory.

## D Additional Experimental Results

## D.1 Computational Efficiency Comparisons

We compare the computational efficiency of AegisFlow with the baselines. The results are shown in Table 5. For the image manipulation task, AegisFlow takes slightly more time than OC-Flow, and is more efficient than OC-Flow + MNC. For the maze navigation task, AegisFlow is slower than SafeDiffuser and SafeFlowmatcher, which is expected since AegisFlow attempts to estimate the solution to the forced HJB equation, while SafeDiffuser and SafeFlowmatcher only solve a simple quadratic program at each timestep in one go. Indeed, they mainly consider constraint satisfaction in a myopic manner, while AegisFlow seeks to find a globally optimal solution that balances constraint satisfaction and task performance.

The experiment is conducted on 2 NVIDIA RTX 6000D GPUs with 84GB memory.

## D.2 Metrics Comparisons for Text-Guided Image Manipulation

The detailed metrics comparisons for text-guided image manipulation on CelebA-HQ are shown in Table 6.