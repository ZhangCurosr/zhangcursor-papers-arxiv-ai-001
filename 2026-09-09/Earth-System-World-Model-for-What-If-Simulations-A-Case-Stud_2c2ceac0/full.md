# Earth System World Model for What-If Simulations: A Case Study for Terrestrial Ecosystems

Zhihao Wang<sup>1</sup>, Ruichen Wang<sup>1</sup>, Ruohan Li<sup>1</sup>, Lei Ma<sup>1</sup>, George Hurtt<sup>1</sup>,

Xiaowei Jia<sup>2</sup>, Gengchen Mai<sup>3</sup>, Shaowen Wang<sup>4</sup>, Yiqun Xie<sup>1∗</sup>

<sup>1</sup>University of Maryland, <sup>2</sup>Rutgers University, <sup>3</sup>University of Texas at Austin, <sup>4</sup>University of Illinois Urbana-Champaign, {zhwang1, ruichenw, r526li, lma6, gchurtt, xie}@umd.edu,

xj159@cs.rutgers.edu, gengchen.mai@austin.utexas.edu, shaowen@illinois.edu

## Abstract

Machine learning emulators have become essential for accelerating expensive Earth-system simulations, but most existing approaches remain passive forecasters: they reproduce simulator trajectories under prescribed forcings without an explicit interaction mechanism for user-specified interventions. This limits their use in interactive scientific workflows and Earth-system digital twins, where users often need to explore how a system would respond if selected state components were changed. We propose an action-conditioned world-modeling framework for Earth-system emulation that reformulates simulator trajectories as supervision for controllable state-transition learning. The key idea is transition-action pretraining: naturally observed state changes are treated as label-free action supervision, allowing the model to learn both prescribed dynamics and action-conditioned responses without manually annotated interventions. We further introduce masked response learning to infer unobserved variables under partial state edits and learn coupled system dependencies. We test this framework on ecosystem dynamics across six global regions and multiple stand ages. Experiments show that the model preserves competitive long-horizon emulation accuracy while enabling controllable structural interventions and coherent responses in coupled ecosystem-cycle variables. These results suggest a practical route from passive Earth-system emulators toward interactive, intervention-aware scientific surrogates.

## CCS Concepts

• Computing methodologies → Machine learning; Modeling and simulation.

## Keywords

World model, digital twin, Earth system model, simulation

## ACM Reference Format:

Zhihao Wang<sup>1</sup>, Ruichen Wang<sup>1</sup>, Ruohan Li<sup>1</sup>, Lei Ma<sup>1</sup>, George Hurtt<sup>1</sup>,, Xiaowei Jia<sup>2</sup>, Gengchen Mai<sup>3</sup>, Shaowen Wang<sup>4</sup>, Yiqun Xie<sup>1</sup>. 2026. Earth System World Model for What-If Simulations: A Case Study for Terrestrial Ecosystems. In The 34th ACM International Conference on Advances in Geographic Information Systems (SIGSPATIAL ’26), November 03–06, 2026, Riverside, CA, USA. ACM, New York, NY, USA, 5 pages. https://doi.org/10.1145/3841645. 3843371

## 1 Introduction

Machine learning emulators have become an important tool for accelerating expensive Earth-system models, including atmospheric, terrestrial, and hydrological simulations [6, 13]. By learning surrogate mappings from environmental forcings and initial states to future system trajectories, these models can reduce the computational cost of large-scale simulations and enable scientific discovery under a much broader range of scenarios for impact assessment (i.e., answering “what-if” questions). However, most Earth-system emulators are designed as passive forecasters or approximators: they reproduce model outputs under prescribed forcings, but they do not provide an explicit model design to enable user-specified interactions or interventions (e.g., disturbance events such as wildfire, logging, mortality, management actions, or policy-induced changes [10, 11]). This is analogous to early-stage video generators that can create high-fidelity pixel streams but do not provide interactability [1]. This limits the use of the machine learning emulators in interactive scientific and application workflows for Earth-system digital twins, where users are often interested in asking not only what trajectory is likely under a given scenario, but also how the system would respond if selected components of the state were changed [9]. Such what-if reasoning is essential components of exploratory analysis and hypothesis generation in scientific studies and policy making, yet less considered by standard emulators.

Recent progress in world models provides a promising direction for moving beyond passive emulation. World models aim to learn compact latent representations of system states and predict how the latent states evolve under actions, with interactability and controllability as their key distinctions: rather than only extrapolating from fixed inputs, the model should update its future predictions in response to interventions, actions, or goals injected on the fly [1, 2]. This action-conditioned dynamics formulation is particularly relevant for Earth-system emulation, where a learned surrogate could support rapid exploration of alternative management and policy decisions, or structural latent state edits [9]. However, directly adapting existing world-model methods to Earth systems requires additional considerations and tailored designs. Unlike games or robotics, Earth-system trajectories often lack explicit action labels, and many relevant interventions are not observed as discrete commands. Moreover, Earth-system states are multi-variable, partially observed, and physically coupled across variables and time scales, so an intervention on one component of the system should induce coherent responses in other variables.

In this study, we take an exploratory step toward interactive world models for Earth-system emulation, via a concrete case <sup>Inputs Natural</sup> <sup>Encoded</sup> study on terrestrial ecosystems. Specifically, we reformulate emu-<sup>Forcing head</sup> <sup>(bypass) provided) latent</sup> <sub>state</sub>lator training from passive input-output supervision into action-<sup>#:!</sup> conditioned state-transition modeling, enabling rapid what-if ex-Action gateploration for explicit, user-edited system states and their coupled LAI� <sup>!</sup>responses. Our contributions are: (1) We introduce transition-action <sup>GPP</sup>�<sup>state provided?pretraining,</sup> <sup>which</sup> <sup>derives</sup> <sup>label-free</sup> <sup>action</sup> <sup>supervision</sup> <sup>from</sup> <sup>nat-</sup> <sup>NPPction</sup>urally observed state changes; (2) We integrate masked response �<sub>! (natural) (user</sub> <sub>edit)learning as a required step to improve the model’s stability with</sub> Optionalpartially unobserved variables, which is needed to enable partial <sub>variables</sub> <sub>at</sub> <sub>action</sub> <sub>step;</sub> <sup>user</sup> <sup>action</sup>state edits; (3) We conduct experiments on representative ecosystem <sup>infer</sup> <sup>coupled</sup> <sup>response.</sup>trajectories over heterogeneous regions from 6 continents.

![](images/8482444a4bc92a61d6e08b399aaa7d8c19e52f6e6b64776a7bdb4535370334a6.jpg)  
Figure 1: Overview of the proposed action-conditioned world-modeling framework for extending passive Earth-system emula Action-Conditioned Geospatial World Modeltors with controllable state edits, transition-action pretraining, and masked response inference.

## 2 Related Work

Earth-system emulation. Deep learning emulators have been increasingly developed to approximate expensive theory-based models across atmospheric, terrestrial, and hydrological processes [6, 8, 13]. For example, in weather forecasting, AI emulators have shown strong potential for accelerating medium-range global prediction by learning directly from large-scale atmospheric reanalysis data such as ERA5 [5]. Similarly, for terrestrial ecosystems, recent work has demonstrated orders of magnitude speed-ups from AI emulators for calibrated, high-fidelity large-scale simulations [7, 12]. However, these emulators usually do not expose an explicit model interaction mechanism for user-defined interventions or partial state edits. As a result, they are less suited for interactive what-if analysis, where users need to modify selected system components dynamically on the fly and infer the coupled downstream response.

World models and controllable dynamics. World models provide a complementary direction by learning compact representations of system state and modeling how those representations evolve under actions [3]. They have been widely studied in reinforcement learning, robotics, video prediction, autonomous driving, and interactive generative environments, where action-conditioned dynamics enable on-the-fly planning, control, and counterfactual interaction [4]. Recent latent-prediction and generative world models further show that future states can be predicted in representation space or generated as interactive environments, making controllability a central distinction from passive forecasting [1, 2]. However, unlike robotics or games, Earth-system trajectories typically lack explicit action labels, where the interventions are often continuous or implicit rather than directly usable discrete commands (e.g., turn left or right in video simulation), and system states are physically coupled and partially observed

<sup>masked</sup> To the best of our knowledge, this work is among the first at-Action gatelearningtempts to explore and bridge these two lines of research by reformulating AI-driven Earth-system emulation as an action-conditioned, provided?<sub>No Yes</sub>interactive world model. Our approach uses transition-action pretraining and masked response learning to move beyond passive (natural) (user edit)surrogate prediction toward interactive, intervention-aware scientific emulation.

## 3 Method

## 3.1 Problem Formulation

We consider Earth-system trajectories consisting of external forcings and system states. Let $\mathbf { x } _ { t } \in \mathbb { R } ^ { M }$ denote the environmental forcings (e.g., temperature, precipitation, CO ) at time �, and $\mathbf { y } _ { t } \in \mathbb { R } ^ { D }$ denote a �-dimensional system state. Given an example of terrestrial ecosystems, $\mathbf { y } _ { t }$ contains coupled structural and ecosystem variables, including canopy height, above-ground biomass, soil carbon, ecosystem carbon fluxes, etc.

Standard emulators learn a passive transition model,

$$
\widehat { \mathbf { y } } _ { t + 1 } = F _ { \theta } ( \mathbf { y } _ { 1 : t } , \mathbf { x } _ { 1 : t } ) ,\tag{1}
$$

which predicts future simulator outputs under prescribed forcings and historical observations. In contrast, our goal is to learn an action-conditioned state-transition model,

$$
\widehat { \mathbf { y } } _ { t + 1 } = F _ { \theta } \big ( \mathbf { y } _ { 1 : t } , \mathbf { x } _ { 1 : t } , \mathbf { a } _ { t } \big ) ,\tag{2}
$$

where $\mathbf { a } _ { t }$ denotes a user-specified state-edit action that may be absent during prescribed rollout. In Earth-system emulation, actions need not correspond to discrete commands as in games or robotics. They can represent interventions that modify selected components of the system state, such as logging-induced reductions in vegetation structure, reforestation-driven structural recovery, disturbance events, or policy-induced land-management changes.

## 3.2 Transition-Action Pretraining

A central challenge in Earth-system emulation is that simulator trajectories usually do not contain explicit action labels. We therefore derive action supervision directly from observed state transitions. Let $S \subseteq \{ 1 , \dots , D \}$ denote the subset of controlled variables in system states. For each transition $\mathbf { y } _ { t } \to \mathbf { y } _ { t + 1 }$ , we define a transition derived action as:

$$
\mathbf { a } _ { t } ^ { S } = \mathbf { y } _ { t + 1 } ^ { S } - \mathbf { y } _ { t } ^ { S } .\tag{3}
$$

where $\mathbf { a } _ { t } ^ { S } , \mathbf { y } _ { t + 1 } ^ { S } ,$ and $\mathbf { y } _ { t } ^ { S }$ denote the subset of controlled variables in $\mathbf { a } _ { t } , \mathbf { y } _ { t + 1 }$ , and $\mathbf { y } _ { t }$ . This converts every simulator transition into a selfsupervised action example without requiring manually annotated interventions. In this study, we instantiate the controlled variable as canopy height. Let $h _ { t }$ denote the canopy-height component of the system state $\mathbf { y } _ { t }$ . The transition-derived action is then defined as the canopy-height changes: $h _ { t + 1 } - h _ { t }$

The action here provides a simple but physically meaningful structural intervention. Positive actions correspond to accelerated structural growth, while negative actions can represent structural reductions such as harvesting or disturbance. The remaining ecosystem variables are not directly interacted; instead, the model must infer their coupled response from the action-conditioned transition.

During training, the model alternates between two modes. In the prescribed mode, it learns natural system evolution from forcings and state history without an action. In the action-conditioned mode, a transition-derived action is provided, and the model learns to propagate the edited controlled state into the remaining variables. This strategy allows the same simulator trajectories to supervise both passive emulation and controllable state-transition learning.

## 3.3 Action-Conditioned Dynamics

A key design challenge is to make the model controllable without losing its ability to predict natural system evolution. If the action is represented only as a latent token, the network may bypass it because the same transition can often be inferred from state history and environmental forcings. In contrast, if the controlled state is always defined directly by the action, the model becomes controllable but cannot predict that state when no action is provided. We address this tension with a gated dynamics design that combines prescribed growth prediction with action-conditioned state editing.

Given the forcing history and observed state history, an encoder produces a latent representation $\mathbf { z } _ { t } . \mathrm { A }$ natural-dynamics head predicts the prescribed increment of the controlled variable:

$$
\Delta \widehat { h } _ { t } = f _ { \theta } ( \mathbf { z } _ { t } ) .\tag{4}
$$

The next controlled state is then computed and decoded by an action gate:

$$
\widehat { h } _ { t + 1 } = \left\{ h _ { t } + D _ { h } ( \Delta \widehat { h } _ { t } ) , \quad \mathrm { i f ~ n o ~ a c t i o n ~ i s ~ p r o v i d e d , } \quad \right. \quad\tag{5}
$$

The remaining variables are decoded conditioned on the latent representation and the resulting controlled state:

$$
\widehat { \mathbf { y } } _ { t + 1 } ^ { - h } = g _ { \boldsymbol { \theta } } \Big ( \mathbf { z } _ { t } , \widehat { h } _ { t + 1 } \Big ) ,\tag{6}
$$

where $\widehat { \mathbf { y } } _ { t + 1 } ^ { - h }$ denotes all predicted variables except canopy height. The final predicted state is

$$
\widehat { \mathbf { y } } _ { t + 1 } = \left[ \widehat { h } _ { t + 1 } ; \widehat { \mathbf { y } } _ { t + 1 } ^ { - h } \right] .\tag{7}
$$

This design gives a single model two operating modes: prescribed rollout when no action is provided and controllable response prediction when an action is specified.

![](images/a633114fd7c7056f422eb4478769a94c9838850460f6332d14f819ae6b0f0d0a.jpg)  
Figure 2: Study areas over six continents.

## 3.4 Masked Response Learning

To infer the coupled response when an intervention modifies only part of the system state, we integrate masked response learning during action-conditioned training, which is necessary to improve the model’s stability to handle user interventions that will change one or a subset of variables but leave others outdated. Let <sup>¯</sup>� denote the complement subset (i.e., non-controlled) of the controlled variable subset �. In the action-conditioned mode, the model observes the controlled state and action while masking the non-controlled variables at the edited step, i.e., setting non-controlled variables in $\bar { S }$ at the input step � to 0. This reflects the scenarios that will be encountered during the inference, where users may edit controlled variables and the model will need to predict with the outdated non-controlled variables masked out. Additionally, in the actionconditioned training mode (randomly assigned to samples), the "true reference" of controlled variables in � at the next step is already provided by the actions simulated using Eq. (3), so the loss function will have these variables masked out in loss calculation. During inference, the model can be rolled out without actions as a standard emulator or run with interactive user state edits to examine controlled responses.

## 4 Experiment

## 4.1 Experimental Setup

Dataset. We evaluate our framework on CarbonGlobe [12], a global, 40-years ecosystem forecasting dataset based on the Ecosystem Demography (ED) model. We select six globally distributed regions covering heterogeneous ecosystem conditions and evaluate three representative stand ages: young, intermediate, and mature forests. A total of 21,315 data samples are selected, where training (85%) and testing (15%) locations are spatially separated, with held-out grid cells used for evaluation.

Model configuration. At each step, the model takes a five-year history of environmental forcings, which are first embedded by an encoder (MLP) and combined with ecosystem states. A recurrent network (using a standard GRU as an example in this prototype study) then encodes them into latent temporal context, followed by a natural-dynamics head (MLP) to predict height changes, and a decoder (MLP) that predicts the remaining multi-variable responses conditioned on that updated height and latent context.

Baselines and metrics. We compare against a persistence baseline, $\widehat { \mathbf { y } } _ { t + 1 } = \mathbf { y } _ { t }$ , and a dedicated no-action baseline emulator trained only for prescribed prediction. We report long-horizon autoregressive rollout performance using RMSE and relative RMSE (rRMSE), with the latter normalized by variable means to account for scale differences. For action-conditioned evaluation, we perturb the transitionderived height action to assess whether the model responds sensitively to structural actions and propagates coherent changes to non-height ecosystem variables.

Table 1: Rollout mean rRMSE across 3 forest conditions.
<table><tr><td>Model</td><td>All Time Steps (Young / Inter. / Mature)</td><td>Final Time Step (Young / Inter. / Mature)</td></tr><tr><td>Persistence</td><td>0.710 / 0.295 / 0.302</td><td>1.124 / 0.376 / 0.347</td></tr><tr><td>Baseline emulator</td><td>0.180 / 0.211 / 0.216</td><td>0.207  / 0.217 / 0.242</td></tr><tr><td>World Model</td><td>0.162 / 0.189 / 0.221</td><td>0.202 / 0.205 / 0.246</td></tr></table>

## 4.2 Results

Prescribed Emulation Performance. We first evaluate whether action-conditioned world model pretraining preserves baseline prescribed emulation performance when no user action is provided. Table 1 reports long-horizon rollout rRMSE across young, intermediate, and mature forest conditions. The World Model achieves performance comparable to, and in several cases better than, the dedicated baseline emulator, while consistently outperforming the persistence baseline. This indicates that introducing action-conditioned training does not compromise standard prescribed emulation and may provide useful transition-level regularization.

Controllability under Height Actions. We evaluate controllability by perturbing the transition-derived height action and rolling out the model under the modified action. As shown in the top row of Fig. 3, the model produces ordered height responses across perturbation magnitudes, indicating sensitivity to user-specified structural actions. The efect is stronger in young forests and weaker in mature forests, consistent with age-dependent forest growth dynamics and slower structural change near maturity.

Masked Response Inference. We further evaluate whether the model can infer coupled ecosystem responses when only the controlled height variable is observed at the action step. As shown in the bottom row of Fig. 3, the deviations of a non-height variable, aboveground biomass, remain close to zero after masking, indicating that the masked model produces responses consistent with the full-observation setting. In mature forests, non-height responses remain close to zero despite height perturbations, suggesting weaker cross-variable sensitivity. This suggests the model learns cross-variable dependencies and can infer coupled ecosystem responses from partial state information rather than relying on direct observation of all variables.

![](images/5245013c0523071083cf3ec3e1dd2f4c8530f3e42fee0655e58a0eb78d91dcbe.jpg)  
Figure 3: Action-conditioned responses under perturbed height actions (top: height; bottom: aboveground biomass).

## 5 Conclusions

We introduce an action-conditioned world-modeling framework for Earth-system emulation, enabling learned emulators to support both prescribed prediction and user-specified structural interventions. The proposed framework derives action supervision from state transitions and uses masked response learning to infer coupled ecosystem responses from partial state information. Experiments on ecosystem trajectories show that this design preserves strong long-horizon emulation performance while enabling consistent responses to user-specified actions and reliable reconstruction of non-controlled variables under masking. These findings provide a first step toward interactive, intervention-aware Earth-system emulators for scientific what-if analysis.

## Acknowledgments

Zhihao Wang, Ruichen Wang, Ruohan Li, and Yiqun Xie are supported in part by NSF Grant No. 2126474, 2147195, 2425844, and 2530610; NASA Grant No. 80NSSC25K0013 and 80NSSC25K7221; Google’s AI for Social Good Impact Scholars program; and the Zaratan cluster at the University of Maryland. Xiaowei Jia is supported in part by NSF Grant No. 2239175, 2147195, 2316305, 2425845, 2530609, and 2203581; NASA Grant No. 80NSSC24K1061 and 80NSSC 25K0013; USGS Grant No. G21AC10564 and G22AC00266; and Pitt Momentum Funds and CRC at the University of Pittsburgh. George Hurtt is supported by NASA Grant No. 80NSSC25K7221 and 80NSSC22K1733. Lei Ma is supported by NASA Grant No. 80NSSC25K7221, 80NSSC24K0599, and 80NSSC24K1632, and Schmidt Sciences. Gengchen Mai is supported by NSF Grant No. 2521631. Shaowen Wang is supported by NSF Grant No. 2118329. We would also like to thank the Derecho system from NSF NCAR.

## References

[1] Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, et al. 2025. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985 (2025).

[2] Jingtao Ding, Yunke Zhang, Yu Shang, Yuheng Zhang, Zefang Zong, Jie Feng, Yuan Yuan, Hongyuan Su, Nian Li, Nicholas Sukiennik, et al. 2025. Understanding world or predicting future? a comprehensive survey of world models. Comput. Surveys 58, 3 (2025), 1–38.

[3] David Ha and Jürgen Schmidhuber. 2018. World models. arXiv preprint arXiv:1803.10122 (2018).

[4] Wenlong Huang, Yu-Wei Chao, Arsalan Mousavian, Ming-Yu Liu, Dieter Fox, Kaichun Mo, and Li Fei-Fei. 2026. PointWorld: Scaling 3D World Models fo In-The-Wild Robotic Manipulation. arXiv preprint arXiv:2601.03782 (2026).

[5] Thorsten Kurth, Shashank Subramanian, Peter Harrington, Jaideep Pathak, Morteza Mardani, David Hall, Andrea Miele, Karthik Kashinath, and Anima Anandkumar. 2023. Fourcastnet: Accelerating global high-resolution weather forecasting using adaptive fourier neural operators. In Proceedings ofthe platform for advanced scientific computing conference. 1–11.

[6] Remi Lam, Alvaro Sanchez-Gonzalez, Matthew Willson, Peter Wirnsberger, Meire Fortunato, Ferran Alet, Suman Ravuri, Timo Ewalds, Zach Eaton-Rosen, Weihua Hu, et al. 2023. Learning skillful medium-range global weather forecasting. Science 382, 6677 (2023), 1416–1421.

[7] Ruohan Li, Zhihao Wang, Xiaowei Jia, Gengchen Mai, Lei Ma, George C Hurtt, Quan Shen, Zhili Li, and Yiqun Xie. 2026. EcoDifusion: Uncertainty-Aware Emulation of Ecosystem Processes with Conditional Difusion for Long Sequences with Single-Step Initialization. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40. 38880–38888.

[8] Ruohan Li, Yiqun Xie, Xiaowei Jia, Dongdong Wang, Yanhua Li, Yingxue Zhang, Zhihao Wang, and Zhili Li. 2024. SolarCube: An Integrative Benchmark Dataset Harnessing Satellite and In-situ Observations for Large-scale Solar Energy Forecasting. Advances in Neural Information Processing Systems 37 (2024), 3499–3513.

[9] Michael D Morecroft, Simon Dufield, Mike Harley, James W Pearce-Higgins, Nicola Stevens, Olly Watts, and Jeanette Whitaker. 2019. Measuring the success

of climate change adaptation and mitigation in terrestrial ecosystems. Science 366, 6471 (2019), eaaw9256.

[10] Rupert Seidl, Mart-Jan Schelhaas, and Manfred J Lexer. 2011. Unraveling the drivers of intensifying forest disturbance regimes in Europe. Global Change Biology 17, 9 (2011), 2842–2852.

[11] Zhihao Wang, Cooper Li, Ruichen Wang, Lei Ma, George Hurtt, Xiaowei Jia, Gengchen Mai, Zhili Li, and Yiqun Xie. 2025. TreeFinder: A US-Scale Benchmark Dataset for Individual Tree Mortality Monitoring Using High-Resolution Aerial

[12] Zhihao Wang, Lei Ma, George Hurtt, Xiaowei Jia, Yanhua Li, Ruohan Li, Zhili Li, Shuo Xu, and Yiqun Xie. 2025. CarbonGlobe: A Global-Scale, Multi-Decade Dataset and Benchmark for Carbon Forecasting in Forest Ecosystems. Advances in Neural Information Processing Systems 38 (2025).

Imagery. Advances in Neural Information Processing Systems 38 (2025).

[13] Zhihao Wang, Yiqun Xie, Xiaowei Jia, Lei Ma, and George Hurtt. 2023. High-Fidelity Deep Approximation of Ecosystem Simulation over Long-Term at Large Scale. In ACM SIGSPATIAL. 1–10.