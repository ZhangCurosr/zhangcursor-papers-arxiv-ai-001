# Discovering hierarchical governing equations for community-structured complex network dynamics with LLM agents

Jinlei Zhang<sup>1,6,\*</sup>, Zixuan Zhang<sup>1</sup>, Hao Qiu<sup>1</sup>, Feng Chen<sup>2</sup>, Jiateng Yin<sup>1</sup>, Xiaoyong Yan<sup>1</sup>, Xiaolei Ma<sup>3</sup>, Hao Sun<sup>4</sup>, Jianxi Gao<sup>5</sup>, Lixing Yang<sup>1,\*\*</sup> and Ziyou Gao<sup>1,\*\*\*</sup>

<sup>1</sup>School of Systems Science, Beijing Jiaotong University, Beijing, China

<sup>2</sup>School of Civil Engineering, Beijing Jiaotong University, Beijing, China.

<sup>3</sup>School of Transportation Science and Engineering, Beihang University, Beijing, China

<sup>4</sup>Gaoling School of Artificial Intelligence, Renmin University of China, Beijing, China

<sup>5</sup>Department of Computer Science, Rensselaer Polytechnic Institute, Troy, NY, US.

<sup>6</sup>Lead contact

\*Correspondence: zhangjinlei@bjtu.edu.cn

\*\*Correspondence: lxyang@bjtu.edu.cn

\*\*\*Correspondence: zygao@bjtu.edu.cn

## SUMMARY

Governing equations provide an interpretable description of the mechanisms underlying complex network dynamics, yet their discovery from observational data remains challenging when shared community-level dynamics coexist with individual-level variations and the relevant functional structures may be absent from the initial candidate library or not explicitly specified in advance. Here we formulate this task as a hierarchical law-discovery problem and introduce SAGE, a large language model-guided self-reflective agent framework for discovering community-shared governing laws and individual-specific dynamical variations. SAGE organizes equation discovery into an iterative Library–Discovery–Review process that first identifies shared community-level structures and then resolves node-specific residual dynamics, while adaptively refining the candidate space to introduce missing dynamical terms when needed. This hierarchical and interpretable formulation enables SAGE to distinguish shared community-level dynamics from individual-level variations and to reveal the mechanisms underlying complex network dynamics. Across two synthetic networks and four real-world networks spanning mobile communication, brain

electrophysiology, power grids and urban rail transit, SAGE recovers compact and interpretable governing equations, reproduces observed system dynamics and outperforms representative equation-discovery baselines. The framework remains robust under incomplete observations and can transfer community-level governing laws to nodes with entirely missing observations. These results establish a hierarchical and adaptive approach for disentangling shared and individual dynamical mechanisms in community-structured complex networks.

## INTRODUCTION

Inferring the governing laws of complex networks, such as ecological <sup>1,2</sup>, neuronal <sup>3,4</sup>, social <sup>5,6</sup>, and transportation <sup>7,8</sup> networks, is fundamental to understanding these systems. Many complex networks exhibit clear community structures and individual-level variations in their dynamics. These networks are typically characterized by the coexistence of community-level dynamics and individual-level dynamics. Communitylevel dynamics <sup>9–11</sup> describe dynamical behaviors shared among nodes within the same community, giving rise to similar community-level dynamical patterns. Individual-level dynamics <sup>12–14</sup>, on the other hand, reflect dynamical variations among individual nodes within the same community. Using a single universal governing equation to describe the dynamics of all nodes in a community-structured network may overlook dynamical differences across communities. In contrast, assigning a separate governing equation to every node ignores shared dynamical structure, typically requires more data, and reduces interpretability. The central challenge is therefore to determine which dynamical mechanisms are shared among nodes within the same community and which represent node-specific deviations. This motivates a hierarchical representation in which the governing dynamics of each node are decomposed into community-level dynamical components shared within the community and individual-level components that capture nodespecific deviations. This decomposition is relevant to many complex real-world networks. For example, in neuronal networks, neurons of the same type or within the same brain region are often assumed to share similar core intrinsic dynamics. However, differences in ion-channel composition or expression levels can introduce additional dynamics through parameter changes or additional current terms. These differences can lead to node-specific variations in neuronal activity. Modeling both community-level and individual-level dynamics thus enables a more complete description of community-level dynamical patterns and nodespecific variations in complex networks.

Conventional approaches derive governing equations for complex networks from first principles or domain knowledge–based derivations <sup>15</sup>. However, these methods often require strong prior assumptions and predefined model structures <sup>16–18</sup>. Such requirements limit their use in real-world networks. These networks are often highly nonlinear, and their underlying mechanisms may remain poorly understood. The growing availability of large-scale spatiotemporal data has enabled data-driven discovery of governing laws in complex networks. For example, symbolic regression methods <sup>19–23</sup> can infer governing equations of complex systems using combinations of predefined variables and operators to fit observational data. Beyond networked systems, recent graph-based equation discovery has further demonstrated the potential to identify compact and interpretable physical laws directly from experimental data <sup>24</sup>. However, equationdiscovery methods that search over combinations of variables and operators can face a rapidly growing search space, leading to high computational costs and potential overfitting. SINDy <sup>25</sup> builds a library of candidate functions and applies sparse regression to identify dominant terms. As an extension of this sparse-library paradigm, PDE-FIND <sup>26</sup> extends this sparse-library paradigm to the discovery of partial differential equations. These sparse-library methods formulate equation discovery as a regularized regression problem that balances model complexity and fitting accuracy. SINDy and its variants have been applied to a wide range of domains, including physical <sup>27,28</sup>, biological <sup>29</sup>, and engineering systems <sup>30,31</sup>. More recent studies have combined equation discovery with deep learning, reinforcement learning, and neural symbolic search <sup>32–38</sup>. Despite these advances, sparse-library approaches generally rely on a predefined candidate function space. When relevant dynamical terms are absent from the initial library, existing methods often lack a systematic way to revise the candidate space based on model failures. As a result, missing candidate functions may remain unaddressed, leading to incomplete or inaccurate governing equations. This is especially problematic in community-structured networks, where residual dynamics may reflect either true node-specific behavior or missing terms in the candidate library.

For complex networks, an additional challenge arises because node dynamics reflect both intrinsic mechanisms and network interactions <sup>39</sup>. The Two-Phase framework enables autonomous inference of deterministic complex network dynamics from incomplete and noisy observations <sup>40</sup>. LaGNA further extends interpretable equation discovery to stochastic complex systems by modeling intrinsic dynamics, network interactions, and stochastic diffusion, thereby learning interpretable stochastic differential equations from observational data <sup>41</sup>. Despite this progress, many existing methods use a shared functional form across all nodes. Such a global model may overlook dynamical differences between communities and nodespecific deviations within each community. Conversely, fitting an independent equation for every node ignores shared dynamical structure and typically requires more data. Therefore, existing methods do not directly address the joint identification of community-level and individual-level dynamics in communitystructured complex networks.

Recent studies have used large language models (LLMs) to guide symbolic and scientific search. LaSR uses an LLM to discover and evolve reusable concepts for symbolic regression <sup>42</sup>. D3 employs multiple agents to propose, evaluate, and iteratively refine interpretable dynamical models <sup>43</sup>. Reflection-based methods also show that language agents can improve their output through feedback without updating model parameters. Reflexion stores linguistic reflections in episodic memory <sup>44</sup>, while Self-Refine improves output through repeated feedback and revision <sup>45</sup>. These studies show that LLMs can support iterative model refinement through feedback. However, governing-equation discovery in community-structured complex networks still faces two coupled challenges: distinguishing shared community-level dynamics from nodespecific variations, and revising the candidate function space when important dynamical terms are missing. A unified framework is therefore needed to address both challenges within the same discovery process.

To address these challenges, we formulate governing-equation discovery in community-structured complex networks as a hierarchical law-discovery problem, in which node dynamics are decomposed into community-level governing laws shared among related nodes and individual-level components that capture node-specific deviations. We develop a Large Language Model–guided Self-reflective Agent framework for Governing Equation discovery (SAGE) to solve this problem when the candidate function space may initially be incomplete. SAGE organizes equation discovery into an iterative candidate-space construction, equation identification, and evaluation process. The Library Agent constructs and adaptively refines the candidate function library by integrating network observations with diagnostic feedback. The Discovery Agent first identifies community-level equation structures and then analyzes the remaining residual dynamics to determine whether additional individual-level terms are required, while Pareto selection balances equation complexity and model accuracy. The Review Agent evaluates candidate equations through derivative consistency, coefficient stability, and state prediction, providing feedback when the inferred dynamics remain incomplete or inconsistent. When structural mismatch is detected, failed review criteria and residual patterns are converted into targeted repair instructions that guide the next round of candidate-library refinement. Importantly, the LLM does not directly generate equations or estimate numerical parameters; instead, it assists in identifying missing dynamical mechanisms and refining the candidate function space, whereas candidate equation generation and coefficient estimation remain handled by numerical equationdiscovery methods. In this way, SAGE combines hierarchical dynamics discovery with adaptive candidatespace refinement in an interpretable and iterative scientific discovery process:

(1) We formulate governing-equation discovery in community-structured complex networks as a hierarchical law-discovery problem and propose SAGE, a unified Large Language Model–guided Selfreflective Agent framework for Governing Equation discovery. Rather than assuming either a single shared equation or fully independent node-wise equations, the framework jointly identifies community-leve dynamics shared among related nodes and individual-level dynamics that capture node-specific deviations. The iterative process is interpretable, reviewable, repairable, and traceable.

(2) We treat the candidate function library as an adaptive rather than fixed search space. When a candidate equation fails a repairable review criterion, SAGE uses failed review checks and residual patterns to identify potentially missing dynamical mechanisms and guide targeted expansion or revision of the candidate library under explicit complexity constraints. This process allows the framework to refine its search without updating the LLM parameters.

(3) We conduct extensive experiments on two synthetic and four real-world community-structured complex networks from different domains. The results demonstrate the interpretability, predictive accuracy, and robustness of SAGE. SAGE also outperforms representative equation-discovery baselines and accurately infers governing equations even under incomplete observation settings. Furthermore, the discovered community-level dynamics can be transferred to nodes with entirely missing observations, demonstrating the value of explicitly identifying shared governing laws in data-limited complex networks.

## RESULTS

## Overview of the SAGE framework

Complex networks often exhibit both community-level and individual-level dynamics (Fig. 1). Considering a complex network with N nodes and K communities, the dynamics of each node can be described by a hierarchical governing equation composed of community-level dynamical terms together with additional individual-level dynamical terms:

$$
\frac { \mathrm { d } x _ { \mathrm { i } } ( t ) } { \mathrm { d } t } = S _ { k } ( x _ { i } ( t ) , A ) + H _ { i } ( x _ { i } ( t ) , A ) , \ i \in c o m m u n i t y \ k\tag{1}
$$

where $\pmb { x } _ { i } ( t ) \in \mathbb { R } ^ { d }$ denotes the d-dimensional state vector of node � at time � and $\pmb { A } \in \mathbb { R } ^ { N \times N }$ represents the network adjacency matrix. The node dynamics $\frac { \mathrm { d } x _ { \mathrm { i } } ( t ) } { \mathrm { d } t } \in \mathbb { R } ^ { d }$ consist of both community-level and individuallevel components. $S _ { k } ( x _ { i } ( t ) , A ) \in \mathbb { R } ^ { d }$ denotes the community-level dynamics for community �. All nodes in community � share the same $\pmb { S } _ { k }$ , which captures their shared dynamical behavior. $\pmb { H } _ { i } ( \pmb { x } _ { i } ( t ) , \pmb { A } ) \in \mathbb { R } ^ { d }$ denotes the individual-level dynamics of the node �. It captures unique dynamic behaviors caused by intrinsic attributes. Each node has its own individual-level function $\pmb { H } _ { i } ,$ . These functions capture variation across nodes.

In real-world complex networks, prior knowledge of network dynamics $( S _ { k }$ and $\pmb { H } _ { i } )$ is often limited or unknown. Our goal is to automatically infer the hierarchical governing equations of complex networks from observational data, including both community-level dynamics and individual-level dynamics. We assume that governing equations usually contain only a few key terms, which can be selected from a large library of candidate functions. For simplicity, we consider one dimension $x _ { \mathrm { i } } ( t )$ of node � in community �. Then, Equation 1 can be written as:

$$
\frac { \mathrm { d } x _ { i } ( t ) } { \mathrm { d } t } = S _ { k } \big ( x _ { i } ( t ) \big ) + H _ { i } \big ( x _ { i } ( t ) \big ) = \Theta ^ { ( r ) } ( x _ { i } ( t ) ) \xi _ { k } ^ { S ( r ) } + \Theta ^ { ( r ) } ( x _ { i } ( t ) ) \xi _ { i } ^ { H ( r ) }\tag{2}
$$

where $\Theta ^ { ( r ) } ( x _ { i } ( t ) )$ denotes the adaptive candidate-function library at search round $r ,$ including polynomial, trigonometric, exponential, and interaction terms (see Table S2). The library is initialized with these common function families and can be adaptively expanded by the Library Agent during equation discovery under explicit complexity constraints. $M _ { r }$ denotes the number of candidate functions in the current library at search round $\boldsymbol { r } . \boldsymbol { \xi } _ { k } ^ { S ( r ) } = \left[ \xi _ { k , 1 } ^ { S ( r ) } , \cdots , \xi _ { k , M } ^ { S ( r ) } \right] ^ { T } \in \mathbb { R } ^ { M _ { r } }$ denotes the coefficient vector of the community-level dynamics shared by nodes in community �. $\boldsymbol { \xi } _ { i } ^ { H ( r ) } = \left[ \xi _ { i , 1 } ^ { H ( r ) } , \cdots , \xi _ { i , M } ^ { H ( r ) } \right] ^ { T } \in \mathbb { R } ^ { M _ { r } }$ denotes the coefficient vector of the individual-level dynamics of node �. The superscript denotes the search round in which the candidate library and corresponding coefficients are obtained. Accordingly, hierarchical governing-equation discovery can be formulated as a joint sparse estimation problem for the community-level coefficients $\xi _ { k } ^ { S ( r ) }$ and the individual-level coefficients $\xi _ { i } ^ { H ( r ) }$

To solve this hierarchical estimation problem, SAGE identifies community-level and individual-level dynamics from node-state observations and network structure through a unified hierarchical procedure (Fig. 1b). The framework comprises three agents: the Library Agent, the Discovery Agent, and the Review Agent.

The Library Agent preprocesses the observations, estimates temporal derivatives and splits the data into training, validation and test sets. It then builds an initial candidate function library with self-dynamics, polynomials, trigonometric functions, saturation terms, temporal terms and topology-aware interactions. Each candidate term has an explicit mathematical form linked to a possible dynamical mechanism, which supports physical interpretation of the inferred equations. The library is adaptive rather than fixed. When residuals suggest missing oscillatory, saturation or nonlinear interaction terms, etc., the Library Agent expands the relevant function families under explicit complexity constraints. For each node, the library matrix $\Theta ^ { ( r ) } ( x _ { i } ( t ) )$ provides a common basis for estimating community-level and individual-level terms.

The Discovery Agent searches for equations that explain the observed node dynamics. It is designed to be independent of any specific numerical solver and uses interchangeable numerical equation-discovery methods to generate candidate equation structures and estimate their coefficients. In this study, we instantiate the Discovery Agent with three representative sparse regression methods: bootstrap sequentially thresholder least squares, cross-validated Lasso, and ridge regression under multiple threshold settings. These numerical methods are not fixed components of SAGE and can be replaced or extended by other compatible equation discovery methods without changing the overall framework. Rather than relying on a single solution, the agent compares candidate equations based on validation error and equation complexity. It then selects a Pareto-optimal model that balances accuracy and simplicity. The search proceeds at two levels. At the community level, the Discovery Agent identifies dynamical terms shared by nodes within the same community and estimates community-specific coefficients. At the individual level, it analyzes the remaining unexplained dynamics and determines whether additional nodespecific dynamical terms are required.

The Review Agent serves as an independent validation and decision layer. It evaluates each selected equation on the validation set using three complementary checks: derivative fit, coefficient stability, and state prediction. The agent also examines whether the selected terms have clear physical meaning. Based on these results, the Review Agent returns one of three decisions: ACCEPT, REPAIR or REJECT. ACCEPT indicates that the equation meets all review criteria. The agent then reports its structure, coefficients, predictive performance and physical interpretation. REPAIR indicates that the equation is incomplete or unstable. The agent identifies the failed checks, converts them into a structured repair plan and sends targeted instructions to the Library Agent. These instructions specify which function families should be revised or expanded in the next search round. REJECT ends the search when the available data cannot support a reliable equation or when the maximum number of iterations is reached and then provides a report that explains the remaining problems.

SAGE repeats the Library–Discovery–Review cycle until an equation passes the review criteria or a stopping criterion is reached. Throughout this process, the framework records the candidate libraries, inferred equations, review results, and repair actions. These records make the discovery process transparent and traceable. Once accepted, the selected community-level and individual-level terms, together with their coefficients, define the governing equation for each node. The inferred node dynamics are expressed as follows (Fig. 1c):

$$
\frac { \mathrm { d } \hat { x } _ { i } ( t ) } { \mathrm { d } t } = \Theta ^ { ( r ) } ( x _ { i } ( t ) ) \hat { \xi } _ { k } ^ { S ( r ) } + \Theta ^ { ( r ) } ( x _ { i } ( t ) ) \hat { \xi } _ { i } ^ { H ( r ) }\tag{3}
$$

To further validate whether the discovered equation can reproduce the observed system evolution, the inferred dynamics numerically evolved over a time interval Δ�:

$$
\hat { x } _ { i } ( t + \Delta t ) = x _ { i } ( t ) + \frac { d \hat { x } _ { i } ( t ) } { d t } \Delta t\tag{4}
$$

here, ${ \hat { x } } _ { i } ( t )$ denotes the inferred state of node ${ \hat { x } } _ { i } ( t )$ , and Δ� represents the prediction interval. The superscript � denotes the search round at which the candidate-function library and the corresponding coefficient vectors are obtained. $\hat { \pmb { \xi } } _ { k } ^ { S ( r ) }$ and $\hat { \pmb { \xi } } _ { i } ^ { H ( r ) }$ denote the corresponding community-level and individuallevel coefficient vectors, respectively. Equation 3 represents the discovered governing dynamics, whereas Equation 4 is used as a validation step to assess whether these dynamics can reproduce the observed node-state trajectories.

Further details are provided in Methods and Document S1, section 1.

SAGE is a multi-agent framework for governing equation discovery that organizes the inference process into an iterative Library–Discovery–Review loop. The Library Agent prepares the numerical inputs and constructs and updates the candidate function library, while also identifying community structures from node observations when prior community information is unavailable.

The Discovery Agent generates and selects sparse equation structures, and the Review Agent evaluates the selected equations and guides subsequent repair when necessary. The main notation and variable dimensions are summarized in Table S1.

In this study, we consider a complex network with multiple communities, as shown in Figure 2a. The framework takes node state observations and the adjacency matrix as primary inputs. Community assignments are used for subsequent hierarchical dynamics discovery, where available community labels are directly adopted; otherwise, they are inferred by the Library Agent from node state observations.

## Inferring synthetic complex network dynamics

To evaluate the effectiveness and robustness of SAGE, we construct two synthetic complex networks: a single-dynamics network and a multi-dynamics network. Both networks simultaneously exhibit community level dynamics and individual-level dynamics. Specifically, we first generate network structures using the

Stochastic Block Model (SBM) <sup>46</sup> and then simulate network dynamics on the generated topology to obtain node-state observations. In our experiments, the single-dynamics network highlights inter-community differences by assigning distinct community-level coefficients to different communities while sharing the same underlying dynamics. In contrast, the multi-dynamics network captures inter-community differences by applying different types of network dynamics to different communities.

In the single-dynamics network, all nodes follow the Lotka–Volterra dynamics <sup>47,48</sup>:

$$
\left\{ \begin{array} { l l } { \displaystyle \frac { \mathrm { d } x _ { i } } { \mathrm { d } t } = a _ { k } x _ { i } - b _ { k } x _ { i } y _ { i } + c _ { k } \sum _ { j \in N _ { i } } x _ { j } + \phi _ { i } \mathrm { s i n } ( 1 . 5 t ) \cdot x _ { i } } \\ { \displaystyle \frac { \mathrm { d } y _ { i } } { \mathrm { d } t } = d _ { k } x _ { i } y _ { i } - e _ { k } y _ { i } + \varphi _ { i } \mathrm { s i n } ( 1 . 5 t ) \cdot y _ { i } } \end{array} \right.\tag{5}
$$

where $a _ { k } , b _ { k } , c _ { \mathrm { k } } , d _ { k } , e _ { k }$ are community-level coefficients for community �. Nodes within the same community � share these coefficients, which characterize their shared dynamical behavior. In contrast, different communities are assigned to distinct coefficients, highlighting dynamical differences across communities. $\phi _ { i } , \varphi _ { i }$ are individual-level coefficients for node � and represent individual-level variations. Each node has distinct coefficients that capture its individual dynamical behavior.

In the multi-dynamics network, nodes follow three dynamical systems: the Lorenz <sup>49</sup>, Rössler <sup>50</sup>, and Chen <sup>51</sup> systems. This network highlights stronger dynamics differences across communities:

$$
L \mathrm { o r e n } z \left\{ \begin{array} { l l } { \displaystyle \frac { \mathrm { d } x _ { i } } { \mathrm { d } t } = a _ { 1 } ( y _ { i } - x _ { i } ) + \phi _ { i } x _ { i } } \\ { \displaystyle \frac { \mathrm { d } y _ { i } } { \mathrm { d } t } = x _ { i } ( b _ { 1 } - z _ { i } ) - y _ { i } + \varphi _ { i } y _ { i } } \\ { \displaystyle \frac { \mathrm { d } z _ { i } } { \mathrm { d } t } = x _ { i } y _ { i } - c _ { 1 } z _ { i } + \psi _ { i } z _ { i } } \end{array} \right.
$$

$$
C h e n \left\{ \begin{array} { l l } { \displaystyle \frac { \mathrm { d } x _ { i } } { \mathrm { d } t } = a _ { 3 } ( y _ { i } - x _ { i } ) + \phi _ { i } x _ { i } } \\ { \displaystyle \frac { \mathrm { d } y _ { i } } { \mathrm { d } t } = - x _ { i } z _ { i } + c _ { 3 } y _ { i } + \varphi _ { i } y _ { i } } \\ { \displaystyle \frac { \mathrm { d } z _ { i } } { \mathrm { d } t } = x _ { i } y _ { i } - b _ { 3 } z _ { i } + \psi _ { i } z _ { i } } \end{array} \right.
$$

$$
\mathit { R o s s l e r } \left\{ \frac { \mathrm { d } x _ { i } } { \mathrm { d } t } = - y _ { i } - z _ { i } + \phi _ { i } z _ { i } \right. \nonumber\tag{6}
$$

where $a , b , c$ are community-level coefficients. Each node follows a governing equation derived from one of the three systems-Lorenz, Rössler, or Chen. $\phi _ { i } , \varphi _ { i } , \psi _ { i }$ are individual-level coefficients for node �.

Figures 3 and 4 demonstrate that SAGE can accurately infer governing equations for complex networks with coupled community-level and individual-level dynamics. For the single-dynamics network (Fig. 3), we first examine whether SAGE can identify the relevant dynamical terms from the adaptive function library. The Discovery Agent searches the candidate function library and identifies the necessary terms to describe the system dynamics (Fig. 3a). After review and refinement, the selected terms correspond to the main components of the governing equations. The inferred equations reproduce the true phase portraits, demonstrating that SAGE captures the underlying governing dynamics of the system (Fig. 3b). The inferred derivatives closely match the true derivatives (Fig. 3c), while the inferred equations accurately reproduce node states (Fig. 3d). The quantitative comparison across overall, community-level, and individual-level dynamics further confirms the accuracy of the inferred equations, with R² values reaching 0.99 (Fig. 3e). Finally, the inferred equations recover both shared community-level terms and individual-level terms, with relative coefficient errors below 2.71% (Fig. 3f).

Beyond the single-dynamics case, we further evaluate whether SAGE can distinguish different governing systems across communities. In this setting, nodes from different communities follow distinct nonlinear systems (Lorenz, Rössler, and Chen), while nodes within each community retain individual-level dynamics. The Discovery Agent identifies the functional terms required by each governing system (Fig. 4a). The inferred equations successfully reproduce the phase portraits of the true systems, demonstrating that SAGE can recover distinct dynamical behaviors across different communities (Fig. 4b). The inferred derivatives and node states remain consistent with the true systems (Fig. 4c, d). The quantitative evaluation further confirms accurate recovery across overall, community-level, and individual-level dynamics(Fig. 4e). Finally, SAGE recovers distinct governing equations across communities while preserving individual-level variations within each community(Fig. 4f).

In summary, SAGE demonstrates the ability to infer governing equations and reconstruct network dynamics for both a single-dynamics network (Fig. 3) and a multi-dynamics network (Fig. 4). In the single-dynamics network, the model accurately identifies community-level and individual-level dynamical terms, while recovering the true governing equations and dynamical trajectories. In the multi-dynamics network, although different communities follow distinct governing systems, SAGE effectively identifies the corresponding governing equations together with community-level and individual-level dynamics. These two experiments demonstrate the equation recovery capability, and equation recovery capability of SAGE in modeling networks with coupled community-level and individual-level dynamics. Detailed system descriptions and inference results for synthetic complex networks can be found in Document S1, section 2.

## Validation of discovered governing equations in real-world networks

To evaluate whether SAGE can uncover meaningful dynamical mechanisms in real-world networks, we apply it to four community-structured complex networks, including an urban mobile communication network <sup>52</sup>, a brain EEG network <sup>53</sup>, a power grid network <sup>54</sup>, and an urban rail transit network <sup>55</sup>. These systems represent distinct classes of complex dynamics, including human activity-driven interactions, biological oscillations, infrastructure synchronization, and mobility propagation.

We first examine whether the discovered governing equations reveal network-specific dynamical mechanisms consistent with the intrinsic characteristics of each network. For each accepted equation, we analyze the retained dynamical terms from three perspectives: (i) the dominant mechanisms encoded by the equation structure, (ii) the correspondence between discovered terms and known network dynamics, and (iii) the separation between community-level shared mechanisms and individual-level variations. State reconstruction and short-term numerical evolution are used only as quantitative validation tools to assess whether the discovered equations can reproduce the observed network dynamics.

Detailed dataset descriptions, discovered equations, and additional validation results are provided in Document S1, section 3.

## Urban mobile communication network

Urban mobile communication networks exhibit recurrent temporal rhythms, nonlinear interactions, and substantial spatial heterogeneity driven by human mobility and regional activity patterns. To examine whether SAGE can recover these characteristic dynamical structures, we select Node 1 in Community 1 as a representative case. For each node, the three state variables $x _ { i } ( t ) , y _ { i } ( t )$ , and $z _ { i } ( t )$ represent Internet activity, phone-call activity, and SMS activity, respectively. The discovered equation structure and its hierarchical interpretation are summarized in Fig. 5, while additional interpretation results are provided in Document S1, section 3.

The discovered equations reveal three main dynamical mechanisms: state–time modulation, time modulated cross-state coupling, and nonlinear state dependence. State–time modulation terms, such as $x c o s ( w _ { 0 } t ) , y c o s ( w _ { 0 } t )$ , and �sin $( w _ { 0 } t )$ , indicate that the effects of communication states vary with recurrent temporal phases. Time-modulated cross-state terms show that the evolution of one communication state can be influenced by other states under the same temporal modulation. In addition, the nonlinear term $z ^ { 2 }$ captures state-dependent dynamics that cannot be represented by linear responses alone. Together, these terms show that urban communication dynamics are shaped by recurrent temporal modulation, cross-state coupling, and nonlinear state dependence.

At the community level, nodes within the same community share similar state–time modulation and crossstate coupling structures, indicating common temporal and interaction mechanisms. At the individual level, node-specific terms and coefficient variations capture local departures from these shared structures, including differences in temporal responses and nonlinear state dependence. Thus, SAGE separates community-shared dynamical mechanisms from node-specific variations, providing an explicit and interpretable description of heterogeneous urban communication dynamics

## Brain EEG network

Brain EEG networks are representative biological complex networks composed of interacting neural channels, characterized by nonlinear oscillatory activity, functional connectivity, and dynamical variation across channels. EEG channels within functionally related communities may share common interaction patterns while retaining channel-specific deviations. Here, we analyze the governing equations discovered by SAGE to reveal the dynamical structures underlying community-level oscillatory behavior and individual variations in EEG networks. For each channel $i ,$ the two state variables $x _ { i } ( t )$ and $y _ { i } ( t )$ denote the real component of the filtered EEG analytic signal and its corresponding Hilbert quadrature component, respectively.

SAGE identifies the following compact governing equations for the EEG dynamics:

$$
\begin{array} { c } { { \displaystyle \frac { d x } { d t } = \left( - 1 0 . 7 6 8 4 y + 5 . 7 2 1 4 \right) + \left( - 1 . 3 2 1 4 y - 0 . 1 7 5 8 \right) = - 1 2 . 0 8 9 8 y + 5 . 5 4 5 6 } } \\ { { \displaystyle \frac { d y } { d t } = \left( 1 7 . 6 6 3 5 x - 6 . 6 7 1 0 \right) + \left( - 0 . 5 1 4 7 x - 0 . 8 6 1 5 \right) = 1 7 . 1 4 8 8 x - 7 . 5 3 2 5 } } \end{array}\tag{7}
$$

The discovered equations reveal two main dynamical mechanisms: cross-state coupling and constant bias. The cross-state coupling terms, � and �, indicate that the evolution of each EEG state is governed by the other state rather than by an isolated self-dynamic term, revealing a coupled dynamical structure underlying EEG activity. Moreover, the opposite signs of these coupling terms suggest an antisymmetric coupling pattern consistent with coupled oscillatory dynamics between the real and quadrature components, rather than independent evolution of the two states. The constant terms, 5.5456 and -7.5325, introduce statespecific offsets that shift the dynamical baseline away from the origin. Together, these terms indicate that the discovered EEG dynamics are primarily characterized by linear cross-state coupling between the two analytic-signal components together with non-zero baseline offsets

At the community level, EEG channels within the same dynamical community share similar cross-state coupling structures, indicating coordinated oscillatory behavior governed by related dynamical mechanisms. At the individual level, channel-specific coefficient variations and baseline shifts capture local deviations from these shared community dynamics. Thus, SAGE separates community-shared coupling structures from channel-specific dynamical variations, providing an explicit and interpretable description of EEG network dynamics. Detailed dataset descriptions and additional interpretation results are provided in Document S1, section 3.

## Power grid network

Power grid networks are representative critical infrastructure systems composed of interconnected buses whose dynamics are strongly shaped by transmission-line topology and phase interactions. Variations in operating conditions and network connectivity can lead to heterogeneous dynamical behaviors across individual buses, while buses within the same dynamical community may share similar interaction structures. Here, the state variable $x _ { i }$ represents the phase state of bus �.

SAGE identifies the following topology-mediated governing equation:

$$
\frac { d x } { d t } = - K \sum _ { j } A _ { i j } { \sin ( x _ { i } - x _ { j } ) }\tag{8}
$$

where $A _ { i j }$ represents the physical transmission-line connection between buses � and $j ,$ and � denotes the fitted coupling strength. The discovered equation reveals two main dynamical mechanisms: an angle– velocity relation and topology-dependent phase coupling. The derivative $d x _ { i } / d t$ describes the angular velocity associated with the evolution of the rotor-angle state, explicitly linking rotor-angle evolution to its rate of change. More importantly, the adjacency-weighted term $A _ { i j } \mathrm { s i n } ( x _ { i } - x _ { j } )$ shows that the dynamics of each bus depend on phase differences with physically connected neighboring buses. The sinusoidal interaction captures the nonlinear effect of phase differences, while the adjacency matrix restricts these interactions to the physical transmission network. Together, these terms show that power-grid dynamics are governed by local rotor-angle evolution and topology-mediated phase interactions, providing a compact description of coherent electromechanical dynamics.

At the community level, buses within the same dynamical community share similar angle–velocity relations and topology-dependent phase-coupling structures, indicating common electromechanical interaction patterns shaped by network connectivity. At the individual level, node-specific coupling coefficients and local interaction strengths capture deviations from these shared community-level dynamics. Thus, SAGE separates community-shared phase-coupling mechanisms from bus-specific interaction variations, providing an explicit and interpretable description of power-grid dynamics. Detailed dataset descriptions and additional interpretation results are provided in Document S1, section 3.

## Urban rail transit network

Urban rail transit networks are representative transportation infrastructure systems formed by interconnected stations and passenger-flow interactions, characterized by strong temporal dependence, nonlinear demand evolution, and spatially heterogeneous travel patterns. Influenced by passenger mobility, station functionality, commuting behaviors, and network connectivity, urban rail systems exhibit both community-level flow dynamics among stations with similar urban roles and individual variations caused by local demand fluctuations and station-specific characteristics. Here, we analyze the governing equations discovered by SAGE for passenger-flow evolution and identify the dynamical mechanisms underlying temporal dependence and heterogeneous mobility patterns.

SAGE identifies the following compact governing equations for the two passenger-flow state variables:

$$
\begin{array} { l } { \displaystyle { \frac { \mathrm { d } x } { \mathrm { d } t } = ( - 1 . 2 5 9 9 x ( t - 1 ) - 0 . 3 3 1 8 x ^ { 2 } - 0 . 4 0 3 6 y ( t - 1 ) ) + ( - 0 . 0 1 3 0 A y ) } } \\ { \displaystyle { \frac { \mathrm { d } y } { \mathrm { d } t } = ( - 1 . 8 2 8 9 y ( t - 1 ) - 0 . 7 1 3 5 y ^ { 2 } + 0 . 1 9 8 6 x ( t - 1 ) ) + ( 1 . 7 4 9 1 x _ { i } y ^ { 2 } ) } } \end{array}\tag{9}
$$

The discovered equations reveal four main dynamical mechanisms: short-term memory, nonlinear state dependence, network coupling, and node-specific cross-state interaction. Short-term memory is reflected by the lagged terms, such as �(� − 1) and $y ( t - 1 )$ , indicating that current passenger-flow changes depend on the previous time step and therefore retain temporal persistence. Nonlinear state dependence is captured by quadratic state terms, showing that the response of passenger flow to its current state is not purely linear. Network coupling is represented by terms such as $A y$ , indicating that the dynamics of a station are influenced by passenger-flow conditions in connected neighboring stations. In addition, node-specific cross-state interaction terms, such as $x y _ { \mathrm { { i } } }$ show that interactions between the two passenger-flow variables vary across stations and contribute to local dynamical heterogeneity. Together, these terms indicate that urban rail transit dynamics are jointly shaped by temporal memory, nonlinear state dependence, network structure, and local cross-state interactions.

At the community level, stations with similar urban functions and travel-demand patterns share common lagged, nonlinear, and network coupling terms, indicating similar passenger flow evolution mechanisms among related stations. At the individual level, node-specific cross state interaction terms together with coefficient variations capture local deviations from these shared dynamics, reflecting heterogeneous passenger flow responses across stations. Thus, SAGE separates community shared passenger flow mechanisms from station-specific dynamical variations, providing an explicit and interpretable description of urban rail transit dynamics. Detailed dataset descriptions, interpretation results, and validation evidence are provided in Document S1, section 3.

To further validate whether the discovered equations accurately describe the underlying system evolution, we evaluate their ability to reconstruct node-state trajectories on real-world networks. The states generated from the inferred equations closely match the observed trajectories across overall, community-level, and individual-level dynamics (Fig. 6). The inferred equations successfully reproduce both shared dynamical patterns among nodes within the same community and individual-level variations among nodes, demonstrating that SAGE can capture hierarchical dynamical structures in real-world complex networks.

## SAGE remains robust under incomplete observations

Real-world complex networks often suffer from incomplete observations caused by sensor failures, communication errors, or limited monitoring resources. Such missing observations make it challenging to identify the underlying governing dynamics of complex systems. To evaluate whether SAGE can recover reliable dynamical laws under incomplete observations, we conduct two experiments by introducing different missing patterns in both synthetic networks and a real-world urban rail transit network.

Similarly, we use node state prediction as an indirect evaluation criterion to examine whether the discovered governing equations can accurately characterize the underlying system dynamics. Specifically, if SAGE successfully identifies the governing equations from incomplete observations, the inferred equations should be able to reconstruct missing dynamical trajectories and accurately describe future system evolution.

In real-world networks, individual sensors may temporarily fail, or communication errors may cause missing observations at certain time steps. To simulate this scenario, we randomly remove observations from all nodes at specific time steps with different missing ratios and investigate whether SAGE can still identify consistent governing dynamics (Fig. 7a, c, d).

As shown in Fig. 7c, for the synthetic network, although the state reconstruction error gradually increases with increasing missing ratios, the inferred node dynamics remain highly consistent with those obtained from complete observations. Even when 50% of temporal observations are removed, SAGE can still recover the underlying dynamical structures and accurately reconstruct the observed state trajectories. The average increase in reconstruction error across the three communities remains below 0.1%. Similar results are obtained for the real-world urban rail transit network (Fig. 7d). Although missing observations introduce additional uncertainty, the discovered dynamics remain stable and the average increase in reconstruction error remains below 3%. These results demonstrate that SAGE can reliably identify governing dynamics even when temporal observations are incomplete, indicating its robustness to missing measurements.

## Transfer of governing laws to unobserved nodes

In some real-world scenarios, certain nodes may have no observations due to sensor failures or deployment limitations. This situation creates a more challenging problem because the governing dynamics of these nodes cannot be directly identified from their own measurements. To evaluate whether SAGE can transfer learned dynamical knowledge across nodes, we completely remove observations from selected nodes and infer their dynamics using the community-level governing equations learned from observed nodes within the same community. The inferred equations are then transferred to unobserved nodes to evaluate whether shared community-level dynamical patterns can characterize nodes without direct observations (Fig. 7b, e, f).

For the synthetic network (Fig. 7e), the reconstructed dynamics and predicted states closely match the ground truth. The average increase in reconstruction error across the three communities remains below 0.1%. For the real-world urban rail transit network (Fig. 7f), SAGE also maintains reliable reconstruction performance, with the average reconstruction error increase remaining below 15%. These results indicate that SAGE can extract shared community-level dynamical patterns from observed nodes and transfer them to nodes with completely missing observations.

## Adaptive reasoning improves equation discovery

To further evaluate the reliability and performance of SAGE, we conduct a series of analyses (Fig. 8). These experiments examine whether SAGE can recover compact and accurate governing equations across synthetic and real-world networks and further evaluate the contributions of its main components. Increasing the relative threshold gradually reduces the number of retained terms (Fig. 8a). At lower thresholds, most of the removed terms are false positives, whereas true positive terms are largely preserved and the state WMAPE remains nearly unchanged. As the threshold increases further, some true-positive terms are also removed, causing a clear increase in state WMAPE. These results indicate that a moderate threshold can remove redundant terms while preserving the terms required to accurately describe the system dynamics. We next benchmark SAGE against several state-of-the-art equation-discovery methods, including SINDy <sup>25</sup>, PDE-FIND <sup>26</sup>, Two-Phase <sup>40</sup>, LLC <sup>22</sup>, ND2 <sup>38</sup> and BILLIE <sup>32</sup>—using both a single-dynamics synthetic network and a real-world urban rail transit network (Fig. 8b). The comparison is based on WMAPE, where a lower value indicates a smaller discrepancy between the reconstructed and observed system states. SAGE consistently achieves the lowest reconstruction error on both networks, with a WMAPE of 9.0% on the real-world network and 0.60% on the synthetic network. In comparison, the best-performing baselines reach 15.8% and 0.68%, respectively. These results demonstrate that SAGE can recover governing equations that reproduce the observed system dynamics more accurately across both synthetic and realworld settings.

We compare different agent reasoning strategies, including One-shot <sup>56</sup>, ReAct <sup>57</sup>, Reflexion <sup>44</sup>, Self-Refine <sup>45</sup>, and SAGE, in terms of recovery accuracy, stability, and structural recovery rate (Fig. 8c). Recovery accuracy evaluates how accurately the governing equations are recovered, stability measures the consistency of the predictive performance of the discovered equations across repeated runs, and structural recovery rate measures whether the correct functional structure is successfully identified. SAGE achieves scores of approximately 0.97, 1.00, and 1.00 for recovery accuracy, stability, and structural recovery rate, respectively. In comparison, ReAct, Reflection, and Self-Refine achieve recovery accuracies of about 0.95, but their stability remains around 0.71 and their structural recovery rates range only from approximately 0.18 to 0.27. SAGE achieves consistently high scores across all three metrics, indicating that it not only recovers the governing equations accurately, but also produces equations with more stable predictive performance and more reliable structural recovery.

Ablation experiments further validate the contribution of each module in SAGE (Fig. 8 d-f). Specifically, the three ablated variants respectively remove the structure-guided candidate library (E1-library), the review process for evaluating discovered equations (E2-Review), and the feedback mechanism that uses review results to refine subsequent discovery (E3-Feedback). As shown in Fig. 8d, the full SAGE achieves F1 scores of approximately 1.00 and 0.97 for ��/�� and ��/��, respectively. Removing the structure-guided candidate library causes the largest degradation, reducing the corresponding F1 scores to approximately 0.50 and 0.45. Removing the review or feedback components also reduces structural recovery, with the F1 scores decreasing to approximately 0.60 for ��/�� and 0.75 ��/�� . These results indicate that the candidate library is particularly important for recovering the correct functional structure, while the review and feedback mechanisms further improve term selection.

A similar pattern is observed in predictive performance (Fig. 8e). SAGE achieves an $R ^ { 2 }$ close to 1.0 for both state dimensions, whereas the ablated variants decrease to approximately 0.85–0.88. This decline indicates that errors in the recovered equation structure directly affect the ability of the inferred equations to reproduce the observed dynamics.

The term-level analysis in Fig. 8f further explains the source of these differences. The ground-truth equations contain four terms for and ��/�� and three terms for ��/��, as indicated by the dashed reference lines. SAGE recovers exactly these four and three terms without introducing spurious terms. In contrast, removing the candidate-library component recovers only two correct terms for each equation while introducing additional false terms, resulting in four selected terms for ��/�� and six for ��/��. When the review or feedback mechanism is removed, the inferred ��/�� equation contains two correct terms together with one spurious term, whereas the ��/�� equation retains the three ground-truth terms but introduces two additional spurious terms.

Thus, the lower F1 scores of the ablated variants arise from both missed ground-truth terms and the inclusion of unnecessary functional terms. Together, these results show that the structure-guided candidate library provides an effective search space, while the review and feedback mechanisms suppress spurious terms and improve the recovered equation structure. More importantly, these findings highlight the role of LLM-guided reasoning in enabling adaptive equation discovery. By interpreting residual patterns and review feedback, the LLM identifies potential missing dynamical mechanisms and guides subsequent candidatelibrary updates. This adaptive refinement allows SAGE to adjust the search space according to different dynamical systems and discovery failures.

## Blind recovery of unseen dynamical mechanisms

To determine whether adaptive candidate-space refinement provides benefits beyond simply enlarging the function library, we designed a controlled blind-repair experiment in which an entire dynamical mechanism was removed from the initial candidate library. Specifically, the state–time modulation family, including $x _ { i } s \mathrm { i n } ( w _ { g } t )$ $y _ { i } s \mathrm { i n } ( w _ { g } t )$ , and their corresponding cross-state forms, was withheld from a synthetic community-structured network. The withheld family was removed from the initial search library but remained available in a broader candidate-family catalog that all adaptive repair methods could access. The ground truth equation structure was never exposed during the search.

We compared four candidate-space update strategies under otherwise identical conditions. The fixedlibrary variant retained the incomplete initial library throughout the experiment and did not allow any new function families to be added. Random expansion added one previously unused function family, sampled uniformly from the same candidate family catalog, whenever the current equation failed the same predefined review criteria. Frozen rule-based repair used a fixed set of if–then rules that mapped residual patterns, such as periodicity or temporal drift, to candidate-function families. These rules were predefined, were not tailored to the withheld mechanism, and remained unchanged throughout the experiment. SAGE instead selected repair families from the same catalog by interpreting numerical residual patterns together with structured feedback from the Review Agent. All four methods used the same Ridge-STLSQ solver, training/validation split, candidate-family catalog, pruning procedure, and a maximum of three repair rounds. At most one new function family was introduced in each repair round.

Across 20 repeated runs, SAGE reintroduced the withheld state–time modulation family in all runs and exactly recovered the complete ground-truth equation structure in 90% of runs. Here, exact recovery means that all ground-truth terms were recovered without introducing any spurious terms. Random expansion achieved exact recovery in 50% of runs, whereas neither the fixed-library nor the frozen rule-based variant recovered the complete structure (Fig. 8g). The exact recovery rate of SAGE was significantly higher than that of random expansion (one-sided Fisher’s exact test, p=0.0069). SAGE further achieved a term-level F1 score of 0.993 and retained only 0.10 false-positive terms on average (Fig. 8 h, i). These results indicate that SAGE does not benefit merely from access to a larger candidate space; rather, it uses residual and review evidence to identify which dynamical family is missing and to repair the search space selectively.

This study introduces SAGE as an interpretable framework for the data-driven discovery of governing laws in community-structured complex networks. Rather than treating all nodes as governed by a single universal equation or fitting each node independently, SAGE addresses a central scientific question: which dynamical mechanisms are shared within a community, and which arise from individual-specific variations. A centra feature of SAGE is its hierarchical formulation of network dynamics, which separates the governing dynamics of each node into community-level dynamics shared by nodes within the same community and individual-level dynamics specific to individual nodes. This decomposition provides an explicit mathematical representation of hierarchical organization in network dynamics and makes the discovered mechanisms directly interpretable. SAGE organizes governing-equation discovery into an iterative Library–Discovery– Review loop, in which the candidate function space is progressively revised according to residual patterns and review evidence rather than remaining fixed throughout the search. The resulting discovery process is therefore not only interpretable and traceable, but also able to revise its search space when the current equation fails to explain the observed dynamics Experiments on two synthetic and four real-world community-structured complex networks demonstrate the effectiveness of SAGE across different dynamical systems and application domains. On synthetic networks, SAGE accurately recovers the underlying governing equations and distinguishes community-leve dynamical terms from individual-level dynamical terms, even when different communities follow different nonlinear dynamical systems. More importantly, the real-world experiments show that the discovered equations reveal system-specific dynamical laws rather than merely reproducing observed trajectories. In the urban mobile communication network, the equations reveal recurrent state–time modulation and nonlinear state dependence; in the EEG network, they identify cross-variable coupling underlying oscillatory dynamics; in the power grid, they recover topology-dependent phase coupling; and in the urban rail transit network, they reveal temporal dependence, nonlinear response, and passenger-flow interactions. Across these networks, the hierarchical equations further distinguish mechanisms shared by related nodes from node-specific dynamical deviations, providing an interpretable description of both collective and individual behavior. Node-state reconstruction and prediction serve as quantitative validation that these discovered laws reproduce the observed network dynamics, rather than as the primary objective of the framework. Under incomplete observations, community-level dynamics learned from observed nodes can also be transferred to nodes with completely missing observations, demonstrating the value of explicitly modeling shared dynamical patterns within communities. The blind repair experiment provides additional evidence that adaptive candidate-space refinement is not equivalent to simply enlarging the function library. When the entire state–time modulation family was withheld from the initial library, SAGE recovered the missing family in all 20 repeated runs and exactly recovered the complete governing structure in 90% of runs, compared with 50% for random expansion and 0% for both the fixed-library and frozen rule-based variants. SAGE also achieved a term-level F1 score of 0.993 while retaining only 0.10 false-positive terms on average. These results support the role of evidence-

## DISCUSSION

guided reasoning in identifying which dynamical mechanism is missing and repairing the candidate space in a targeted manner. Comparisons with representative equation-discovery methods, different agent reasoning strategies, and ablation experiments further demonstrate the advantages of SAGE in structural recovery, predictive performance, and solution stability.

Beyond predictive performance, the main methodological significance of SAGE lies in combining numerica equation discovery with LLM-guided scientific reasoning while preserving explicit mathematical laws. Most sparse equation-discovery methods rely on a predefined candidate function library, so relevant dynamical terms may be missed when the required functions are absent from the initial library. SAGE instead treats the candidate function library as an adaptive space that can be revised according to residual patterns and review evidence. Importantly, the LLM does not directly generate governing equations or estimate numerical coefficients. Numerical methods remain responsible for generating candidate equation structures and estimating their coefficients, while the LLM uses structured diagnostic evidence to guide candidate selection and candidate-library refinement. This separation is important for interpretability: the numerical solver determines the equation and its coefficients, whereas the LLM helps decide where the search space should be revised. Because each retained term has an explicit mathematical form and can be linked to a corresponding dynamical mechanism, the discovered equations remain physically interpretable rather than functioning as black-box predictors. Together, hierarchical decomposition and adaptive candidate-space refinement turn equation discovery into a process for identifying shared and individual dynamical laws from observational network data.

## Limitations of the study

Nevertheless, several limitations remain. First, the separation between community-level and individual-level dynamics relies on the quality of community identification. In this study, community structures are either obtained from available domain knowledge or inferred from node-state observations when prior labels are unavailable. While this enables application to networks without predefined communities, inaccurate or ambiguous community assignments may affect the decomposition of shared and individual dynamics. Future work could jointly optimize community identification and governing-equation discovery, allowing the discovered dynamics to further refine the community assignments. Second, strong observational noise, irregular sampling, or sparse measurements may affect derivative estimation and subsequent equationstructure identification. More robust derivative-estimation techniques or derivative-free equation-discovery methods could therefore be incorporated. Third, although SAGE can adaptively revise the candidate library, the current implementation selects new function families from a predefined catalog. The blind repair experiment therefore tests the recovery of mechanisms withheld from the initial library rather than completely unrestricted generation of previously unknown mathematical forms. Extending the framework toward more open-ended function discovery is an important direction for future work. Finally, the current formulation assumes that community assignments and governing equations remain approximately stable over the observation period. Extending SAGE to identify time-varying community structures and evolving governing equations would further broaden its applicability to real-world complex networks.

## METHODS

## Library Agent

The Library Agent prepares the observational data and manages the candidate function library throughout equation discovery. It first preprocesses node-state observations and determines community assignments using either available domain knowledge or inferred dynamical structures from node-state observations when prior information is unavailable. The community assignments obtained are then used to organize nodes for subsequent hierarchical equation discovery. It then constructs an initial candidate function library from the processed observations. During subsequent search rounds, the Library Agent updates the library in response to residual patterns and diagnostic feedback from the Review Agent, adding function families that may capture missing dynamical mechanisms.

Using the preprocessed node-state observations and, when available, the adjacency matrix, the Library Agent constructs the initial candidate function library. The candidate functions are evaluated for each node using the same library and are then organized according to the community assignments. For community k, resulting initial candidate-function matrix is written as:

$$
{ \pmb { \Theta } } _ { K } ^ { ( 0 ) } = [ { \pmb { \Theta } } ^ { ( 0 ) } ( { \pmb { x } } _ { i } ( t ) ) ] _ { i \in k }\tag{10}
$$

where $\Theta ^ { ( 0 ) } ( x _ { i } ( t ) )$ denotes the candidate function vector generated from the initial library and evaluated at the state of node � at time �, and � indicates the community index. When community labels are unavailable, they are inferred from the node-state observations during preprocessing (Community Clustering). Thus, all nodes share the same candidate function definition, whereas the community labels determine how their observations are assembled for community-level equation discovery. The complete set of candidate function families is provided in Table S2.

The library is used in the first discovery round but may not contain all functions required to explain the observed dynamics. After the Discovery Agent selects an equation at search round �, the Review Agent evaluates it on the validation set. The corresponding derivative residual is defined as:

$$
\mathrm { R e s } ^ { ( r ) } = \frac { \mathrm { d } x _ { i } ( t ) } { \mathrm { d } t } - \frac { \mathrm { d } \hat { x } _ { i } ^ { ( r ) } ( t ) } { \mathrm { d } t }\tag{11}
$$

where $\frac { \mathrm { d } x _ { i } ( t ) } { \mathrm { d } t } \in \mathbb { R } ^ { d }$ denotes the derivatives estimated from the validation data, and $\frac { \mathrm { d } \hat { x } _ { i } ^ { ( r ) } ( t ) } { \mathrm { d } t } \in \mathbb { R } ^ { d }$ denotes the derivatives predicted by the equation identified at search round. The Review Agent summarizes the failed review checks, residual patterns, and unstable terms as diagnostic feedback. When this feedback suggests that the current library lacks relevant dynamical mechanisms, the LLM identifies additional or revised function families from the broader structural catalog and uses them to generate or instantiate candidate terms that are not currently represented in the active library. Let $R ^ { ( r ) }$ denote the function families selected for repair at search round r. The candidate function library is updated as:

$$
L ^ { ( r + 1 ) } = P [ L ^ { ( r ) } \cup S ( R E P A I R ^ { ( r ) } ) ]\tag{12}
$$

where $S ( R E P A I R ^ { ( r ) } )$ maps the selected function families to candidate terms, and �(⬝) denotes the library pruning operator that removes duplicate and strongly collinear terms. The detailed procedure for function family selection, candidate term generation, and library pruning is described in Document S1, section 1.3. If a selected function family is already represented in the current library, the framework may instantiate higher-order, reparametrized, or interaction variants constructed from the admissible primitives rather than simply duplicating existing terms. The updated library is then returned to the Discovery Agent for the next round of equation structure search, completing the Library–Discovery–Review repair loop. Implementation details of the LLM interface are provided in Document S1, section 1.6.

## Discovery Agent

The Discovery Agent searches the current candidate library for sparse governing equations. The Discovery Agent is not tied to any specific numerical solver. Numerical methods are used to generate candidate equation structures and estimate their coefficients and other suitable methods can be incorporated into the framework. In this study, we use bootstraps sequentially thresholded least squares (STLSQ) <sup>24</sup>, Lasso with cross-validation <sup>58</sup>, and Ridge regression <sup>59</sup> followed by validation-guided thresholding as three representative methods. These methods employ different regularization and term-selection mechanisms and therefore provide complementary candidate structures and supporting evidence for reliable term selection. The three methods used here are not fixed components of SAGE and can be replaced or extended by other numerical equation-discovery methods without changing the overall framework. At the community level, each numerical backend estimates the shared coefficient vector $\hat { \pmb { \xi } } _ { k } ^ { S }$ defined in the governing equation formulation above using observations from nodes within community �.

The numerical backends return candidate equation structures and their estimated coefficients. For each candidate term � , the Discovery Agent evaluates three complementary stability measures: $f _ { m }$ , the retention consistency across numerical backends; $b _ { m }$ , the selection frequency across bootstrap samples; and $s _ { m } .$ , the consistency of the coefficient signs across repeated estimates. Together, these measures define the stability evidence for term m as:

$$
\boldsymbol { z _ { m } } = ( f _ { m } , b _ { m } , s _ { m } )\tag{13}
$$

These stability measures are retained separately and provide term-level supporting evidence for each candidate equation; no single stability measure determines whether a term is finally selected. Each candidate equation is then evaluated by its validation error and structural complexity, where structural complexity is measured by the number of retained terms. The Pareto candidate set is defined as:

$$
{ \mathcal { P } } _ { \mathrm { P a r e t o } } = \left\{ q \in { \mathcal { Q } } | \sharp q ^ { \prime } \in { \mathcal { Q } } \colon E _ { q ^ { \prime } } \leq E _ { q } , C _ { q ^ { \prime } } \leq C _ { q } , \left( E _ { q ^ { \prime } } < E _ { q } \vee C _ { q ^ { \prime } } < C _ { q } \right) \right\}\tag{14}
$$

Here, $E _ { q }$ and $C _ { q }$ denote the validation error and structural complexity of candidate equation, respectively. Pareto filtering removes candidates that are both less accurate and more complex, while retaining alternative structures that represent different trade-offs between accuracy and simplicity. The LLM then compares only these Pareto candidates using their validation error, structural complexity, and the stability evidence of their retained terms, and selects the community-level equation structure. The LLM does not generate new equation structures or estimate coefficients. After the structure is selected, the equation structure is fixed and its coefficients are re-estimated numerically using observations from nodes in the corresponding community. This refitting step follows the same interchangeable numerical-backend interface and is therefore not tied to a particular solver.

The Discovery Agent next examines the part of each node’s dynamics that remains unexplained by the selected community-level equation. For node i in community k, the residual dynamics are obtained by subtracting the fitted community-level component from its observed temporal derivative:

$$
R e s _ { i } ^ { H } ( t ) = \frac { d x _ { i } ( t ) } { d t } - \Theta ^ { ( r ) } ( x _ { i } ( t ) ) \xi _ { k } ^ { S ( r ) }\tag{15}
$$

The residual dynamics are then examined to determine whether additional individual-level terms are required. A candidate term is treated as an individual-level terms if it improves validation performance for only a subset of nodes, or if it is consistently selected across nodes within the same community but produces different fitted coefficient values between nodes. These two forms of evidence allow the Discovery Agent to distinguish individual-level variations from the dynamics already explained by the shared community-level equation. To quantify the contribution of candidate term m to node i, we measure the reduction in validation error after including the term:

$$
\Delta E _ { i , m } = E _ { i } ^ { ( - m ) } - E _ { i } ^ { ( + m ) }\tag{16}
$$

The validation improvement of candidate term for node is measured by $\Delta E _ { i , m } .$ , defined as the reduction in validation error after including the term. For each candidate individual-level term, the Discovery Agent considers three factors: whether the term improves validation performance, whether this improvement is consistently observed across repeated fits, and how much additional complexity the term adds to the equation. The decision to retain a term is based on these factors together rather than on any single criterion. After the community-level and individual-level terms are determined, their coefficients are refitted numerically using the selected numerical backend. Finally, any term whose removal does not noticeably reduce validation performance is discarded. The resulting governing equation follows the same formulation as introduced in Equation 2.

## Review Agent

The Review Agent evaluates each selected equation on the validation set using three complementary checks: derivative fit, coefficient stability, and state prediction. Derivative fit measures the agreement between the derivatives predicted by the inferred equation and the observed derivatives. To assess coefficient stability, the selected structure is fixed and its coefficients are re-estimated over B bootstrap resamples. For candidate term m, the relative coefficient dispersion is defined as:

$$
C V _ { m } = \frac { s t d ( \xi _ { m } ^ { ( 1 ) } , \ldots , \xi _ { m } ^ { ( B ) } ) } { | m e a n ( \xi _ { m } ^ { ( 1 ) } , \ldots , \xi _ { m } ^ { ( B ) } ) | }\tag{17}
$$

where a smaller $C V _ { m }$ indicates a more stable coefficient estimate. State prediction further evaluates whether the inferred equation can accurately predict future system states by evolving the observed initial states and comparing the predicted trajectories with the actual validation trajectories. Using the three checks jointly, the Review Agent returns ACCEPT when the equation satisfies the predefined review criteria, REPAIR when the remaining errors indicate a correctable structural mismatch and REJECT when the available data cannot support a stable equation or the maximum number of repair rounds is reached. The detailed review criteria and numerical settings are provided in Document S1, section.

An ACCEPT decision indicates that the selected equation satisfies the predefined validation criteria and represents a physically interpretable governing equation, with each retained functional term corresponding to an identifiable dynamical mechanism. The accepted equation is then passed to the independent test set for final evaluation. For REPAIR, the Review Agent summarizes the failed checks, residual patterns, and unstable terms into a structured repair plan. Based on this evidence, the LLM selects or refines function families and proposes admissible candidate-term constructions from the available mathematical primitives, while numerical solvers remain solely responsible for coefficient estimation and final equation fitting. Let ������<sup>(௥)</sup> denote the function families selected for repair at round r. The resulting repair instruction is passed to the Library Agent, which updates the candidate function library according to Equation 3. The updated library is then passed back to the Discovery Agent, and the Library–Discovery–Review cycle is repeated.

A REJECT decision terminates the search when the available data cannot support a stable equation or when the maximum number of repair rounds is reached. In this case, the system generates a structured report summarizing the entire iteration history, including the candidate library evolution, tested equations, review outcomes, and repair actions, ensuring full traceability and diagnostic transparency. Throughout the search, each round records the candidate library, candidate equations, review results, and repair actions. The test set remains isolated throughout library adaptation, model selection, and review, and is used only after an equation receives ACCEPT.

## Community Clustering

When community labels are provided, they are directly adopted as prior structural information. When labels are unavailable, the Library Agent infers dynamical communities from node-state observations using fuzzy K-means clustering. This state-based approach assigns nodes with similar dynamical behaviors to the same community and does not require predefined network topology information for community identification.

## Numerical Differentiation

Accurate derivative estimation is critical for equation discovery but remains challenging for noisy observational data. We therefore use the Savitzky–Golay filter <sup>60</sup>, which applies local polynomial regression to smooth the data while preserving higher order moments. This approach suppresses noise and provides stable derivative estimates.

## Evaluation Metric

To quantify errors in the inferred node dynamics and predicted node states, we use the weighted mean absolute percentage error (WMAPE):

$$
\mathsf { W M A P E } = \frac { \sum _ { i = 1 } ^ { N } \lvert y _ { i } - \hat { y } _ { i } \rvert } { \sum _ { i = 1 } ^ { N } \lvert y _ { i } \rvert } \times 1 0 0 \%\tag{18}
$$

Here, $y _ { i }$ and $\hat { y } _ { i }$ denote the true and predicted values, respectively, and is the number of samples.

## Ethics statement

This study involved no new experiments on human participants or animals. Therefore, ethical approval and informed consent were not required.

## RESOURCE AVAILABILITY

## Lead contact

Requests for further information and resources should be directed to and will be fulfilled by the lead contact, Jinlei Zhang (zhangjinlei@bjtu.edu.cn).

## Materials availability

This study did not generate new physical materials.

## Data and code availability

The synthetic and real-world datasets used in this study are publicly available in the SAGEdata repository on Hugging Face: https://huggingface.co/datasets/tczzx6/SAGEdata/tree/main. The repository includes the synthetic single- and multi-dynamics datasets and the real-world urban mobile communication, brain EEG, IEEE 39-bus power grid, and Beijing urban rail transit datasets. Processed data and data-generation scripts required to reproduce the main analyses are also provided. Further details are given in the Methods and Document S1.

The implementation of SAGE is publicly available at https://huggingface.co/tczzx6/LLM-SAGE/tree/main. The repository includes the Library Agent, Discovery Agent, and Review Agent, together with the numerical equation-discovery modules, adaptive candidate-library refinement procedure, and scripts used to reproduce the main experiments and figures reported in this study.

Any additional information required to reanalyze the data reported in this paper is available from the lead contact upon request.

## ACKNOWLEDGMENTS

This work was supported in part by the National Natural Science Foundation of China under Grant 72288101, in part by the Beijing Natural Science Foundation under Grant 4252049 and Grant L251030,

and in part by the Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China under Grant JYB2025XDXM211.

The authors have no additional acknowledgements to declare.

## AUTHOR CONTRIBUTIONS

J.Z. conceived the study and developed the overall research framework. J.Z., Z.Z., and H.Q. developed the methodology and performed the main analyses. Z.Z. and H.Q. conducted the experiments and data analysis. F.C., J.Y., X.Y., X.M., H.S., and J.G. contributed to the result interpretation and discussion. L.Y. and Z.G. supervised the research. J.Z., Z.Z., and H.Q. wrote the initial manuscript. All authors reviewed and revised the manuscript and approved the final version.

## DECLARATION OF INTERESTS

The authors declare no competing interests.

## SUPPLEMENTAL INFORMATION

Document S1. Supplemental methods, Tables S1–S6, and Figures S1–S6.

## FIGURE TITLES AND LEGENDS

![](images/11e15185fe0a70ea493bcbca5cbd834c46915e9f7e139ff4095c31653d7c3b8f.jpg)

![](images/9d17789c86bca7f5f9037d5d4205bd3cb4fc4a1a7c944651bdf3f0e6bd9c6c02.jpg)

![](images/09e12afcb00ecb2d72cc326770cd836fbc7124c879b982ae4946f8b0a1f68147.jpg)  
Figure 1. Overview of SAGE. a) Nodes are organized into dynamical communities, where community assignments can be obtained from prior knowledge when available or inferred from node-state observations when such information is unavailable. The community-level dynamics shared within each community are then modeled separately from the individual-level dynamics specific to each node. b) Dynamics inference for a representative node from the first community. The Library Agent constructs and adaptively updates the candidate function library from observational data. The Discovery Agent generates candidate equation structures and selects Pareto-optimal solutions. The Review Agent evaluates the selected equations and provides feedback for subsequent library refinement. c) Inferred node dynamics and predicted node states for synthetic and real-world networks, together with comparisons against baseline method.

![](images/82a3a03e9a1f62fb53986ab18dab37cf06286c1448121a9283a52a43ae4726ad.jpg)  
Figure 2. Overview of the SAGE framework for governing-equation discovery in communitystructured complex networks. a) Input data, including the community-structured network, node-state trajectories, and adjacency matrix. b) The iterative Library–Discovery–Review workflow. The Library Agent identifies community structures, preprocesses node-state trajectories, estimates temporal derivatives, and constructs the candidate function library. The Discovery Agent uses multiple numerical solvers to generate candidate equation structures, performs Pareto selection based on validation error and structural complexity, and separately identifies community-level and individual-level dynamics. The Review Agent evaluates the selected equations using derivative fit, coefficient stability, and state prediction, and returns ACCEPT, REPAIR, or REJECT. REPAIR feedback is returned to the Library Agent to guide subsequent library refinement and equation discovery. c) Model decomposition and final outputs, including the inferred community-level and individual-level governing terms, true-versus-inferred derivatives on held-out test data, predicted node states on the test set, and a traceable and explainable report.

a  
![](images/9bf62577622439d949631f057ee884ed1aa35cfea9a2a54cba4ddd3304eadd45.jpg)

b  
![](images/a536828f1e0b5d0a8b66cc3e6ca4244a5fbe9386855f83c9f884fade006d1cda.jpg)

f  
![](images/7d891b7dbc98feade90258b3cc472408af63625ce0e38ed7f03ef8738aa6428c.jpg)  
Figure 3. Validating SAGE on a single-dynamics network with 60 nodes divided into three communities. a) Evolution of candidate terms guided by the Library–Discovery–Review Agent loop. Candidate terms are progressively selected from the function library based on their contribution to equation discovery, with color intensity indicating selection strength. The inference results for a representative node in the first community. b) True and inferred trajectories and force field in both x and y dimensions. c-d) True and inferred node dynamics $( d x _ { i } / d t , d y _ { i } / d t )$ and the validation using node state $( x _ { i } , y _ { i } )$ for a representative node in the first community. e) Goodness of fit between true and inferred overall $( d x _ { i } / d t , d y _ { i } / d t )$ community-level $\left( S _ { k } ( x _ { i } ) , S _ { k } ( y _ { i } ) \right)$ , and individual-level $\left( H _ { i } ( x _ { i } ) , H _ { i } ( y _ { i } ) \right)$ ) dynamics. f) True and inferred governing equations for the first and second nodes in Community 1 and the first node in Community 2.

a  
![](images/3f31b1d177f08aa4b32e4ae8974015854741416f5de479fd560b74649d622f87.jpg)  
Figure 4. Validating SAGE on a multi-dynamics network with 60 nodes divided into three communities. a) Evolution of candidate terms guided by the Library–Discovery–Review Agent loop. Candidate terms are progressively selected from the function library based on their contribution to equation discovery, with color intensity indicating selection strength. b) True and inferred trajectories and force field in both x, y, and z dimensions. c-d) Node dynamics $( d x _ { i } / d t , d y _ { i } / d t , d z _ { i } / d t )$ and node state $\left( x _ { i } , y _ { i } , z _ { i } \right)$ generated by true and inferred governing equations. e) Goodness of fit between true and inferred overall $( d x _ { i } / d t , d y _ { i } / d t , d z _ { i } / d t )$ , community-level $( S _ { k } ( x _ { i } ) , S _ { k } ( y _ { i } ) , S _ { k } ( z _ { i } ) )$ and individual-level $( H _ { i } ( x _ { i } ) , H _ { i } ( y _ { i } ) , H _ { i } ( z _ { i } ) )$ dynamics. f) True and inferred governing equations for the first and second nodes in Community 1 and the first node in Community 2

<table><tr><td rowspan=1 colspan=1>Identified mechanism</td><td rowspan=1 colspan=1>Representative terms (in equations)</td><td rowspan=1 colspan=1>Interpretation</td></tr><tr><td rowspan=1 colspan=1>State-time modulation</td><td rowspan=1 colspan=1>x cos(ω0, t), z cos(ω0, t), z sin(ω0, t)</td><td rowspan=1 colspan=1>Effects vary with recurrent temporal phases</td></tr><tr><td rowspan=1 colspan=1>Self-/cross-state interactions</td><td rowspan=1 colspan=1>z2, x sin(ω0, t)</td><td rowspan=1 colspan=1>Temporal modulation acts on both own and other states</td></tr><tr><td rowspan=1 colspan=1>Nonlinear state dependence</td><td rowspan=1 colspan=1>z²</td><td rowspan=1 colspan=1>Additional nonlinear state-dependent dynamics</td></tr></table>

![](images/941c83af9d9cff97629600e5f7a14385400d7ae89455aec1907058bdf346e3dd.jpg)  
Figure 5. Validation of governing equations discovered across real-world networks. Report presents the discovered governing equations, key dynamical structures, mechanistic interpretation, hierarchical decomposition into community-shared and individual-specific dynamics, and quantitative validation based on step prediction $R ^ { 2 }$ bootstrap coefficient variation and term recurrence frequency. Purple and green shading indicate community-level and individual-level terms, respectively. The results reveal a hierarchical organization in which shared dynamical mechanisms coexist with heterogeneous node-specific variations across real-world systems.

![](images/c1d04eeadc140bb01e89a0a54ca1e8a813e53bc51123fab591df8e983d31c634.jpg)  
Figure 6. Validation of SAGE on real-world networks through node-state reconstruction and comparison between observed and predicted state distributions across all nodes and communities. a) Urban mobile communication network. b) Brain EEG network. c) Power grid network. d) Urban rail transit network.

![](images/7eef6c0ec319fa011909fcb63c3fbe8c7e192bae23f6d4c56825ced92df43b6a.jpg)  
Robustness of SAGE under incomplete observations  
Prediction performance under observation missing at full time steps in a subset of nodes

b  
![](images/ae9ee44ae1ff4df73e27768e692ad99b705d75aea2b61ba8bc0e306577062355.jpg)

![](images/e20030eeeceb8a50f92bdb6801d8ab983ca07be920d9990abd5e44e75590a289.jpg)

![](images/9d6000f84e95f6bf67c3c836f3f3fdf8712a380b1af5cb9d5592abb79241dd49.jpg)

![](images/df54eb2214f21c402351d9fe4fc961396a508ade611ece50824dea217ab3d460.jpg)

![](images/b22daa524acce4d603adb16df6151cb98d9e194a45d52e3d1d72e31d7e726af4.jpg)

![](images/cae966da2208e458821afbf8461ffc85ac1003b4ba708f732283822b1e7b89d2.jpg)

![](images/880997ab49cbc3951405b6b2e45ceb595d0408c8ea88b451025239592a5ae253.jpg)

![](images/c536bcd4487d22ec13ad21d1dd55d33fd51a9a5213a9455952a7892dd79ae57b.jpg)

![](images/7bc179a8746539d8a697371dfa686cab97dcf0ddd4d37b8069d4051f2f697b94.jpg)

![](images/eef4a71619a90e96cb92a8e3c5c838d64ba879aa9ab834f73a1267f3763acae2.jpg)  
Figure 7. Robustness of SAGE under incomplete observations. a) Diagram of observation missing at partial time steps across all nodes. b) Diagram of observation missing at full-time steps in a subset of nodes. c-f) from left to right: true and inferred node dynamics $( d x _ { i } / d t )$ and node states $( x _ { i } )$ , goodness of fit between true and inferred node dynamics $( d x _ { i } / d t )$ , and performance comparison under different data missing settings for both synthetic networks (c, e) and real-world network (d, f).

![](images/a6efd9a714f9a86346207f83fea1b34e1505052ce78d08ded104f1eb09633c50.jpg)

b  
![](images/fed9567ef6f46ab27cefe4435b6fc47f90e037ea76ddc223e3834e5a83129217.jpg)

c  
![](images/6b351f31e63d17bee61128166ce1e75a8aafa98e814cb14f2b509a3a77df751d.jpg)

d  
F1  
![](images/81283f3f1d8fabd1351eef7e7d75381d344905eac74618a0c52fb21ec428ab21.jpg)

e  
![](images/6a3145a3f79007bdd04a15b88d77faacdb49573ce15c6409b2700cfc8d78f263.jpg)

f  
![](images/44c9536ab0025426cdd4b69f21dbc3b2fe25068322ef8e044c637e947630f713.jpg)

g  
![](images/2f6e92301d72ab7ca89dc8501a85e541e9f94b58bf94b45eb6c1680d7e7fe009.jpg)

h  
![](images/a7294c40837b7dbbce41d9548a24641cad1250303af15692196738f2eaeb2b9f.jpg)

i  
![](images/754783d81f575287225b93b78465475b9c1cc2c1d8d28f70fa070ea152f63601.jpg)  
Figure 8. Diagnostic and ablation analysis of SAGE. a) Sensitivity to the relative threshold � showing the numbers of total, true positive, and false positive terms together with validation state WMAPE. b) Reconstruction errors of SAGE and representative equation discovery methods on synthetic and real-world networks. All methods are constrained to use the same number of terms in the discovered equations. c) Comparison of different agent reasoning strategies in terms of recovery accuracy, stability $\left( 1 . \mathsf { S D } \left( R ^ { 2 } \right) \right)$ , and structural recovery rate. d) Equation-discovery F1 scores for $d x / d t$ and $d y / d t$ under SAGE and different module ablations. e) Predictive accuracy of the discovered equations measured by $R ^ { 2 } { \mathfrak { f } } )$ Numbers of recovered terms for $d x / d t$ and $d y / d t ;$ hatched regions indicate spurious terms, and dashed lines denote the corresponding ground truth term counts. g) Recovery rates of the withheld state–time modulation family and the complete ground-truth equation structure across 20 repeated runs. The entire state–time modulation function family was removed from the initial candidate library before discovery. h) Term-level F1 scores of the recovered governing equations. i) Average number of false-positive terms remained in the final equations. Error bars indicate variability across repeated runs. All methods used the same Ridge-STLSQ numerical solver, candidate-family catalog, data split, and three-round repair budget. The frozen rule-based baseline used predefined repair rules that were not tailored to the withheld mechanism and remained unchanged throughout the experiment.

## REFERENCES

1. Brose, U., Hirt, M.R., Ryser, R. et al. (2025). Embedding information flows within ecological networks. Nat. Ecol. Evol. 9:547–558. https://doi.org/10.1038/s41559-025-02670-2.

2. Galiana, N., Lurgi, M., Bastazini, V.A.G. et al. (2022). Ecological network complexity scales with area. Nat. Ecol. Evol. 6:307–314. https://doi.org/10.1038/s41559-021-01644-4.

3. Yang, Y., Qiao, S., Sani, O.G. et al. (2021). Modelling and prediction of the dynamic responses of large-scale brain networks during direct electrical stimulation. Nat. Biomed. Eng. 5:324–345. https://doi.org/10.1038/s41551-020-00666-w.

4. Oldham, S., Fulcher, B.D., Aquino, K. et al. (2022). Modeling spatial, developmental, physiological, and topological constraints on human brain connectivity. Sci. Adv. 8:eabm6127. https://doi.org/10.1126/sciadv.abm6127.

5. Zhou, B., Pei, S., Muchnik, L. et al. (2020). Realistic modelling of information spread using peer-to-peer diffusion patterns. Nat. Hum. Behav. 4:1198–1207. https://doi.org/10.1038/s41562-020-00945-1.

6. Parkinson, C., Kleinbaum, A.M. and Wheatley, T. (2017). Spontaneous neural encoding of social network position. Nat. Hum. Behav. 1:0072.

7. Akbarzadeh, M. and Estrada, E. (2018). Communicability geometry captures traffic flows in cities. Nat. Hum. Behav. 2:645–652.

8. Saberi, M., Hamedmoghadam, H., Ashfaq, M. et al. (2020). A simple contagion process describes spreading of traffic jams in urban networks. Nat. Commun. 11:1616. https://doi.org/10.1038/s41467- 020-15353-2.

9. Pecora, L.M., Sorrentino, F., Hagerstrom, A.M. et al. (2014). Cluster synchronization and isolated desynchronization in complex networks with symmetries. Nat. Commun. 5:4079. https://doi.org/10.1038/ncomms5079.

10. Della Rossa, F., Pecora, L., Blaha, K. et al. (2020). Symmetries and cluster synchronization in multilayer networks. Nat. Commun. 11:3179. https://doi.org/10.1038/s41467-020-16343-0.

11. Lodi, M., Sorrentino, F. and Storace, M. (2021). One-way dependent clusters and stability of cluster synchronization in directed networks. Nat. Commun. 12:4073. https://doi.org/10.1038/s41467-021- 24363-7.

12. Lidstrom, M.E. and Konopka, M.C. (2010). The role of physiological heterogeneity in microbial population behavior. Nat. Chem. Biol. 6:705–712. https://doi.org/10.1038/nchembio.436.

13. Pellett, C. and Valbuena, R. (2025). Disentangling dispersion from mean reveals true heterogeneity– diversity relationships. Nat. Commun. 16:8532. https://doi.org/10.1038/s41467-025-64287-0.

14. Perez-Nieves, N., Leung, V.C.H., Dragotti, P.L. and Goodman, D.F.M. (2021). Neural heterogeneity promotes robust learning. Nat. Commun. 12:5791. https://doi.org/10.1038/s41467-021-26022-3.

15. Karniadakis, G.E., Kevrekidis, I.G., Lu, L. et al. (2021). Physics-informed machine learning. Nat. Rev. Phys. 3:422–440. https://doi.org/10.1038/s42254-021-00314-5.

16. Raissi, M., Perdikaris, P. and Karniadakis, G.E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. J. Comput. Phys. 378:686–707. https://doi.org/10.1016/j.jcp.2018.10.045.

17. Chen, Z., Liu, Y. and Sun, H. (2021). Physics-informed learning of governing equations from scarce data. Nat. Commun. 12:6136. https://doi.org/10.1038/s41467-021-26434-1.

18. Willard, J., Jia, X., Xu, S. et al. (2023). Integrating scientific knowledge with machine learning for engineering and environmental systems. ACM Comput. Surv. 55:66:1–66:37.

19. Bongard, J. and Lipson, H. (2007). Automated reverse engineering of nonlinear dynamical systems. Proc. Natl. Acad. Sci. USA 104:9943–9948.

20. Udrescu, S.M. and Tegmark, M. (2020). AI Feynman: A physics-inspired method for symbolic regression. Sci. Adv. 6:eaay2631. https://doi.org/10.1126/sciadv.aay2631.

21. Schmidt, M. and Lipson, H. (2009). Distilling free-form natural laws from experimental data. Science 324:81–85. https://doi.org/10.1126/science.1165893.

22. Hu, J., Cui, J. and Yang, B. (2025). Learning interpretable network dynamics via universal neural symbolic regression. Nat. Commun. 16:6226. https://doi.org/10.1038/s41467-025-61575-7.

23. Ruan, K., Xu, Y., Gao, Z.-F. et al. (2026). Discovering physical laws with parallel symbolic enumeration. Nat. Comput. Sci. 6:53–66. https://doi.org/10.1038/s43588-025-00904-8.

24. Xu, H., Chen, Y. and Zhang, D. (2026). Beyond empirical models: Discovering constitutive laws in solids with graph-based equation discovery. Sci. Adv. 12:eaec0989.

25. Brunton, S.L., Proctor, J.L. and Kutz, J.N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. Proc. Natl. Acad. Sci. USA 113:3932–3937. https://doi.org/10.1073/pnas.1517384113.

26. Rudy, S.H., Brunton, S.L., Proctor, J.L. and Kutz, J.N. (2017). Data-driven discovery of partial differential equations. Sci. Adv. 3:e1602614. https://doi.org/10.1126/sciadv.1602614.

27. Loiseau, J.C. and Brunton, S.L. (2018). Constrained sparse Galerkin regression. J. Fluid Mech. 838:42–67.

28. Loiseau, J.C., Noack, B.R. and Brunton, S.L. (2018). Sparse reduced-order modelling: sensor-based dynamics to full-state estimation. J. Fluid Mech. 844:459–490.

29. Mangan, N.M., Brunton, S.L., Proctor, J.L. and Kutz, J.N. (2016). Inferring biological networks by sparse identification of nonlinear dynamics. IEEE Trans. Mol. Biol. Multiscale Commun. 2:52–63.

30. Kaiser, E., Kutz, J.N. and Brunton, S.L. (2018). Sparse identification of nonlinear dynamics for model predictive control in the low-data limit. Proc. R. Soc. A 474:20180335.

31. Champion, K.P., Brunton, S.L. and Kutz, J.N. (2019). Discovery of nonlinear multiscale systems: sampling strategies and embeddings. SIAM J. Appl. Dyn. Syst. 18:312–333.

32. Li, Z., Yuan, H., Han, W. et al. (2025). Bi-level identification of governing equations for nonlinear physical systems. Nat. Comput. Sci. 5:456–466. https://doi.org/10.1038/s43588-025-00804-x.

33. Li, Z., Han, W., Zhang, Y. et al. (2024). Learning spatiotemporal dynamics with a pretrained generative model. Nat. Mach. Intell. 6:1566–1579. https://doi.org/10.1038/s42256-024-00938-z.

34. Zhai, Z.M., Stern, B.D. and Lai, Y.-C. (2025). Bridging known and unknown dynamics by transformerbased machine-learning inference from sparse observations. Nat. Commun. 16:8053.

35. Rao, C., Ren, P., Wang, Q. et al. (2023). Encoding physics to learn reaction–diffusion processes. Nat. Mach. Intell. 5:765–779. https://doi.org/10.1038/s42256-023-00685-7.

36. Floryan, D. and Graham, M.D. (2022). Data-driven discovery of intrinsic dynamics. Nat. Mach. Intell. 4:1113–1120.

37. Xu, H., Chen, Y., Cao, R. et al. (2025). Generative discovery of partial differential equations by learning from math handbooks. Nat. Commun. 16:10255. https://doi.org/10.1038/s41467-025-65114- 2.

38. Yu, Z., Ding, J. and Li, Y. (2026). Discovering network dynamics with neural symbolic regression. Nat. Comput. Sci. 6:156–168. https://doi.org/10.1038/s43588-025-00893-8.

39. Barzel, B. and Barabási, A.-L. (2013). Universality in network dynamics. Nat. Phys. 9:673–681.

40. Gao, T.T. and Yan, G. (2022). Autonomous inference of complex network dynamics from incomplete and noisy data. Nat. Comput. Sci. 2:160–168.

41. Gao, T.T., Barzel, B. and Yan, G. (2024). Learning interpretable dynamics of stochastic complex systems from experimental data. Nat. Commun. 15:6029.

42. Grayeli, A., Sehgal, A., Costilla-Reyes, O. et al. (2024). Symbolic regression with a learned concept library. Adv. Neural Inf. Process. Syst. 37:44678–44709.

43. Holt, S., Qian, Z., Liu, T. et al. (2024). Data-driven discovery of dynamical systems in pharmacology using large language models. Adv. Neural Inf. Process. Syst. 37:96325–96366.

44. Shinn, N., Cassano, F., Gopinath, A. et al. (2023). Reflexion: language agents with verbal reinforcement learning. Adv. Neural Inf. Process. Syst. 36:8634–8652.

45. Madaan, A., Tandon, N., Gupta, P. et al. (2023). Self-Refine: iterative refinement with self-feedback. Adv. Neural Inf. Process. Syst. 36:46534–46594. https://doi.org/10.52202/075280-2019.

46. Holland, P.W., Laskey, K.B. and Leinhardt, S. (1983). Stochastic blockmodels: First steps. Soc. Netw. 5:109–137.

47. Lotka, A.J. (1920). Analytical note on certain rhythmic relations in organic systems. Proc. Natl. Acad. Sci. USA 6:410–415.

48. Volterra, V. (1926). Variazioni e fluttuazioni del numero d'individui in specie animali conviventi. Mem. R. Accad. Naz. Lincei, Ser. 6 2:31–113.

49. Lorenz, E.N. (1963). Deterministic nonperiodic flow. J. Atmos. Sci. 20:130–141.

50. Rössler, O.E. (1976). An equation for continuous chaos. Phys. Lett. A 57:397–398.

51. Chen, G. and Ueta, T. (1999). Yet another chaotic attractor. Int. J. Bifurcation Chaos 9:1465–1466.

52. Barlacchi, G., De Nadai, M., Larcher, R. et al. (2015). A multi-source dataset of urban life in the city of Milan and the Province of Trentino. Sci. Data 2:150055. https://doi.org/10.1038/sdata.2015.55.

53. Schalk, G. (2009). EEG Motor Movement/Imagery Dataset (version 1.0.0). PhysioNet. https://doi.org/10.13026/C28G6P.

54. Athay, T., Podmore, R. and Virmani, S. (1979). A practical method for the direct analysis of transient stability. IEEE Trans. Power Appar. Syst. PAS-98:573–584.

55. Zhang, J., Chen, F., Cui, Z. et al. (2021). Deep learning architecture for short-term passenger flow forecasting in urban rail transit. IEEE Trans. Intell. Transp. Syst. 22:7004–7014. https://doi.org/10.1109/TITS.2020.3000761.

56. Brown, T.B., Mann, B., Ryder, N. et al. (2020). Language models are few-shot learners. Adv. Neural Inf. Process. Syst. 33:1877–1901.

57. Yao, S., Zhao, J., Yu, D. et al. (2023). ReAct: synergizing reasoning and acting in language models. International Conference on Learning Representations.

58. Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. J. R. Stat. Soc. B 58:267– 288.

59. Hoerl, A.E. and Kennard, R.W. (1970). Ridge regression: biased estimation for nonorthogonal problems. Technometrics 12:55–67.

60. Savitzky, A. and Golay, M.J.E. (1964). Smoothing and differentiation of data by simplified least squares procedures. Anal. Chem. 36:1627–1639. https://doi.org/10.1021/ac60214a047.

# Document S1. Supplemental information

## Discovering hierarchical governing equations for community-structured complex network dynamics with LLM agents

## 1. Methods

SAGE is a multi-agent framework for governing-equation discovery. It organizes equation discovery into a three-agent iterative loop. The Library Agent manages the candidate functions, the Discovery Agent identifies sparse equation structures, and the Review Agent evaluates the selected equations and requests repair when necessary.

## 1.1 Notations and problem formulation

For clarity, the main notations and their dimensions used in this paper are summarized in Table S1. Scalars are denoted by non-bold letters $( \mathsf { e . g . } , x _ { i } ( t ) \in \mathbb { R } )$ , and vectors or matrices are denoted by bold letters (e.g., $\pmb { x } _ { i } ( t ) \in \mathbb { R } ^ { d } )$ . The detailed definitions of the symbols are given below.

Table S1. List of notations
<table><tr><td colspan="1" rowspan="1">Symbol</td><td colspan="1" rowspan="1">Description</td></tr><tr><td colspan="1" rowspan="1"> $N \in \mathbb { R }$ </td><td colspan="1" rowspan="1">Total number of nodes in the network</td></tr><tr><td colspan="1" rowspan="1"> $K \in \mathbb { R }$ </td><td colspan="1" rowspan="1">Total number of communities in the network</td></tr><tr><td colspan="1" rowspan="1"> $k \in \{ 1 , 2 , \ldots , K \}$ </td><td colspan="1" rowspan="1">Index of community k</td></tr><tr><td colspan="1" rowspan="1"> $T \in \mathbb { R }$ </td><td colspan="1" rowspan="1">Total number of time steps</td></tr><tr><td colspan="1" rowspan="1"> $N _ { k } \in \mathbb { R }$ </td><td colspan="1" rowspan="1">Total number of nodes in community k</td></tr><tr><td colspan="1" rowspan="1"> $\pmb { A } \in \mathbb { R } ^ { N \times N }$ </td><td colspan="1" rowspan="1">Network adjacency matrix describing interactions among N nodes.</td></tr><tr><td colspan="1" rowspan="1"> $\pmb { x } _ { \mathrm { i } } ( t ) \in \mathbb { R } ^ { d }$ </td><td colspan="1" rowspan="1">The state of node i at time t across all d dimensions</td></tr><tr><td colspan="1" rowspan="1"> $x _ { i } ( t ) \in  { \mathbb { R } }$ </td><td colspan="1" rowspan="1">The state of node i at time t for one dimension</td></tr><tr><td colspan="1" rowspan="1"> $\frac { \mathrm { d } \pmb { x } _ { \mathrm { i } } ( t ) } { \mathrm { d } t } \in \mathbb { R } ^ { d }$ </td><td colspan="1" rowspan="1">The dynamics of node i at time t across all d dimensions</td></tr><tr><td colspan="1" rowspan="1"> $\frac { \mathrm { d } x _ { i } ( t ) } { \mathrm { d } t } \in \mathbb { R }$ </td><td colspan="1" rowspan="1">The dynamics of node i at time t for one dimension</td></tr><tr><td colspan="1" rowspan="1"> $S _ { k } ( x _ { i } ( t ) , A ) \in \mathbb { R } ^ { d }$ </td><td colspan="1" rowspan="1">The community-level dynamics of community k across all ddimensions</td></tr><tr><td colspan="1" rowspan="1"> $S _ { k } \in \mathbb { R }$ </td><td colspan="1" rowspan="1">The community-level dynamics of community k for one dimension</td></tr><tr><td colspan="1" rowspan="1"> $\pmb { H } _ { i } ( \pmb { x } _ { i } ( t ) , \pmb { A } ) \in \mathbb { R } ^ { d }$ </td><td colspan="1" rowspan="1">The individual-level dynamics of node i across all d dimensions</td></tr><tr><td colspan="1" rowspan="1"> $H _ { i } \in \mathbb { R }$ </td><td colspan="1" rowspan="1">The individual-level dynamics of node i for one dimension</td></tr><tr><td colspan="1" rowspan="1"> $L ^ { ( r ) } = \{ L _ { m } \} _ { m = 1 } ^ { M _ { r } }$ </td><td colspan="1" rowspan="1">Adaptive candidate function library maintained at search round r</td></tr><tr><td colspan="1" rowspan="1"> $\pmb { \Theta } ^ { ( r ) } ( \pmb { x } _ { i } ( t ) ) \in \mathbb { R } ^ { M _ { r } }$ </td><td colspan="1" rowspan="1">Candidate function vector evaluated from $L ^ { ( r ) }$ for node i at time t</td></tr><tr><td colspan="1" rowspan="1"> $\boldsymbol { \Theta } _ { k } ^ { ( r ) } \in \mathbb { R } ^ { N _ { k } T \times M _ { r } }$ </td><td colspan="1" rowspan="1">Candidate function matrix constructed for community k atround r</td></tr><tr><td colspan="1" rowspan="1"> $M _ { r }$ </td><td colspan="1" rowspan="1">Number of candidate functions at search round r</td></tr><tr><td colspan="1" rowspan="1"> $\pmb { \xi } _ { k } ^ { S ( r ) } = \left[ \xi _ { k , 1 } ^ { S ( r ) } , \cdots , \xi _ { k , M } ^ { S ( r ) } \right] ^ { T } \in \mathbb { R } ^ { M _ { r } }$ </td><td colspan="1" rowspan="1">The coefficient vector for community-level dynamics of community k</td></tr><tr><td colspan="1" rowspan="1"> $\boldsymbol { \xi } _ { i } ^ { H ( r ) } = \left[ \xi _ { i , 1 } ^ { H ( r ) } , \cdots , \xi _ { i , M } ^ { H ( r ) } \right] ^ { T } \in \mathbb { R } ^ { M _ { r } }$ </td><td colspan="1" rowspan="1">The coefficient vector for individual-level dynamics of node i</td></tr><tr><td colspan="1" rowspan="1"> $\pmb { \xi } ^ { S ( r ) } = \left[ \pmb { \xi } _ { 1 } ^ { S ( r ) } , \cdots , \pmb { \xi } _ { K } ^ { S ( r ) } \right] \in \mathbb { R } ^ { M _ { r } \times K }$ </td><td colspan="1" rowspan="1">The coefficients for community-level dynamics of all k communities</td></tr><tr><td colspan="1" rowspan="1"> $\pmb { \xi } ^ { H ( r ) } = \left[ \pmb { \xi } _ { 1 } ^ { H ( r ) } , \cdots , \pmb { \xi } _ { N } ^ { H ( r ) } \right] \in \mathbb { R } ^ { M _ { r } \times N }$ </td><td colspan="1" rowspan="1">The coefficients for individual-level dynamics of all N nodes</td></tr><tr><td colspan="1" rowspan="1"> $\frac { \mathrm { d } \hat { x } _ { i } ( t ) } { \mathrm { d } t } \in \mathbb { R }$ </td><td colspan="1" rowspan="1">The inferred dynamics of node i at time t</td></tr><tr><td colspan="1" rowspan="1"> $\Delta t$ </td><td colspan="1" rowspan="1">Prediction time interval</td></tr><tr><td colspan="1" rowspan="1"> $\hat { x } _ { i } ( t + \Delta t ) \in \mathbb { R }$ </td><td colspan="1" rowspan="1">The predicted state of node i at time t+∆t</td></tr><tr><td colspan="1" rowspan="1"> $a _ { k } , b _ { k } , c _ { \mathrm { k } } , d _ { k } , e _ { k }$ </td><td colspan="1" rowspan="1">The community-level coefficients for community k</td></tr><tr><td colspan="1" rowspan="1"> $\phi _ { i } , \varphi _ { i }$ </td><td colspan="1" rowspan="1">The individual-level coefficients for node i</td></tr></table>

## 1.2 Candidate-function catalog and adaptive library roles in SAGE

The initial candidate library serves only as a compact active search space and contains a small set of common low-complexity dynamical terms. It does not exhaustively enumerate all candidate terms that may be considered during discovery. Instead, SAGE maintains a broader catalog of admissible function families and mathematical primitives that defines the structural prior for candidate-space refinement. During subsequent repair rounds, the LLM analyzes residual patterns together with review feedback to determine which dynamical mechanisms are insufficiently represented in the current library. Based on this diagnosis, SAGE can activate previously unused function families and dynamically instantiate, reparametrize, or compose additional candidate terms from the admissible primitives, including higher-order nonlinearities, state–time interactions, and topology-aware interaction structures. The resulting candidates are incorporated into the active library only after redundancy and collinearity screening. In this way, the candidate space evolves according to the observed failure modes of the current equation rather than being fully enumerated in advance.

Table S2. Candidate-function families and their roles in the adaptive library of SAGE
<table><tr><td colspan="1" rowspan="1">Function family</td><td colspan="1" rowspan="1">Symbol</td><td colspan="1" rowspan="1">Library role</td><td colspan="1" rowspan="1">Description</td></tr><tr><td colspan="1" rowspan="1">Constant</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Base</td><td colspan="1" rowspan="1">Baseline term or constantexternal drive</td></tr><tr><td colspan="1" rowspan="1">Self-dynamics</td><td colspan="1" rowspan="1">x, y, z</td><td colspan="1" rowspan="1">Base</td><td colspan="1" rowspan="1">Linear self-dependentdynamics of node (i)</td></tr><tr><td colspan="1" rowspan="1">Polynomial self-interaction</td><td colspan="1" rowspan="1"> $x _ { i } ^ { 2 } , x _ { i } x _ { j }$ </td><td colspan="1" rowspan="1">Base + adaptiveextension</td><td colspan="1" rowspan="1">Second-order nonlinear stateinteractions within nodestates</td></tr><tr><td colspan="1" rowspan="1">Cubic self-interaction</td><td colspan="1" rowspan="1"> $x ^ { 2 } , x y , x y z , x ^ { 3 }$ </td><td colspan="1" rowspan="1">Adaptive expansion</td><td colspan="1" rowspan="1">Higher-order nonlinear self-effects and multi-dimensionalinteractions</td></tr><tr><td colspan="1" rowspan="1">Trigonometric</td><td colspan="1" rowspan="1"> $\sin ( x _ { i } ) , \cos ( x _ { i } )$ </td><td colspan="1" rowspan="1">Base + adaptiveextension</td><td colspan="1" rowspan="1">Oscillatory or periodicnonlinear dynamics</td></tr><tr><td colspan="1" rowspan="1">Exponential / decay</td><td colspan="1" rowspan="1"> $e ^ { x _ { i } }$ </td><td colspan="1" rowspan="1">Adaptive expansion</td><td colspan="1" rowspan="1">Exponential growth, decay,or relaxation mechanisms</td></tr><tr><td colspan="1" rowspan="1">Network meancoupling</td><td colspan="1" rowspan="1">Ax</td><td colspan="1" rowspan="1">Base</td><td colspan="1" rowspan="1">Neighbor averaging andglobal diffusion-like coupling</td></tr><tr><td colspan="1" rowspan="1">Network diffusion</td><td colspan="1" rowspan="1"> $\pmb { A } ( x _ { j } - x _ { i } )$ </td><td colspan="1" rowspan="1">Base</td><td colspan="1" rowspan="1">Difference-based diffusiveinteractions betweenconnected nodes</td></tr><tr><td colspan="1" rowspan="1">Nonlinear networkcoupling</td><td colspan="1" rowspan="1"> $A f ( x _ { j } - x _ { i } )$ </td><td colspan="1" rowspan="1">Adaptive expansion</td><td colspan="1" rowspan="1">Nonlinear topology-awareinteractions through networkconnections</td></tr><tr><td colspan="1" rowspan="1">Networkautoregressivecoupling</td><td colspan="1" rowspan="1"> $A f ( t - \tau )$ </td><td colspan="1" rowspan="1">Adaptive expansion</td><td colspan="1" rowspan="1">Delayed network effects andtemporal neighbordependence</td></tr><tr><td colspan="1" rowspan="1">Saturation</td><td colspan="1" rowspan="1"> $\operatorname { t a n h } ( x _ { i } ) , s i g m o i d ( x _ { i } )$ </td><td colspan="1" rowspan="1">Base + adaptiveextension</td><td colspan="1" rowspan="1">Saturating or threshold-likeresponses</td></tr><tr><td colspan="1" rowspan="1">Explicit timedependence</td><td colspan="1" rowspan="1"> $t , \sin ( \omega t ) , \cos ( \omega t )$ </td><td colspan="1" rowspan="1">Base + adaptiveextension</td><td colspan="1" rowspan="1">Time-varying external forcingand periodic driving</td></tr><tr><td colspan="1" rowspan="1">State-time modulation</td><td colspan="1" rowspan="1"> $x _ { i } \sin ( \omega t ) , x _ { i } \cos ( \omega t )$ </td><td colspan="1" rowspan="1">Adaptive expansion</td><td colspan="1" rowspan="1">Time modulation of state-dependent dynamics</td></tr><tr><td colspan="1" rowspan="1">Time-state interaction</td><td colspan="1" rowspan="1"> $t x _ { i } , s \mathrm { i n } ( t ) x _ { i }$ </td><td colspan="1" rowspan="1">Adaptive expansion</td><td colspan="1" rowspan="1">Coupled temporal and state-dependent effects</td></tr><tr><td colspan="1" rowspan="1">Autoregressive terms</td><td colspan="1" rowspan="1"> $x _ { i } ( t - \tau )$ </td><td colspan="1" rowspan="1">Adaptive expansion</td><td colspan="1" rowspan="1">Short-term memory anddelayed state effects</td></tr><tr><td colspan="1" rowspan="1">Moving-averageterms</td><td colspan="1" rowspan="1"> $M { \bf A } _ { \omega } ( x _ { i } )$ </td><td colspan="1" rowspan="1">Adaptive expansion</td><td colspan="1" rowspan="1">Smoothed historicaldependence over temporalwindows</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Adaptive expansion</td><td colspan="1" rowspan="1"></td></tr></table>

## 1.3 Adaptive candidate function library expansion

During equation discovery, the initial candidate-function library may not contain all dynamical mechanisms required to describe the observed system. Instead of manually specifying an excessively large function library, SAGE enables adaptive library expansion guided by residual diagnostics from the Review Agent. When the selected equation fails to explain specific dynamical patterns, the Review Agent summarizes the remaining errors and identifies potential missing mechanisms. Based on these diagnostic signals, the LLM identifies appropriate function families from the broader structural catalog and determines whether existing candidate terms should be activated, reparametrized, or extended through admissible compositions of mathematical primitives.

At search round r, let $S ( R E P A I R ^ { ( r ) } )$ denote the set of function families selected for repair. Each selected function family is translated into concrete candidate terms through a term-generation operator, which may instantiate predefined forms or construct admissible higher-order, interaction, or parameterized variants from the available primitives.

$$
S ( R E P A I R ^ { ( r ) } ) = \{ L _ { 1 } , L _ { 2 } , \dots , L _ { q } \}\tag{S1}
$$

where $S ( \cdot )$ generates explicit mathematical candidate terms from the selected function families, such as polynomial, periodic, saturation, or interaction terms, which are subsequently added to the candidate function library. The expanded library is then obtained by combining the existing candidate-function library with the newly generated terms:

$$
L ^ { ( r ) } \cup S ( R E P A I R ^ { ( r ) } )\tag{S2}
$$

To prevent uncontrolled growth of the candidate library, a pruning operation is applied:

$$
L ^ { ( r + 1 ) } = P [ L ^ { ( r ) } \cup S ( R E P A I R ^ { ( r ) } ) ]\tag{S3}
$$

The pruning procedure removes redundant, duplicated, or highly correlated candidate terms while preserving function families that provide complementary explanatory capabilities. The selection of additional function families is determined according to the characteristics of the residual dynamics. The Review Agent analyzes residual patterns from three complementary perspectives: temporal dynamics, state dependence, and interaction effects. These diagnostic patterns are converted into structured feedback for the LLM, which selects corresponding function families from the predefined catalog. Typical mappings between residual patterns and candidate function families are summarized below:

Table S3. Mapping between residual patterns, candidate function families, and corresponding candidate terms
<table><tr><td rowspan=1 colspan=1>Residual pattern</td><td rowspan=1 colspan=1>Candidate function families</td><td rowspan=1 colspan=1>Candidate function term</td></tr><tr><td rowspan=1 colspan=1>Periodic oscillation in residuals</td><td rowspan=1 colspan=1>Oscillatory dynamics</td><td rowspan=1 colspan=1>sin(x), cos(x)</td></tr><tr><td rowspan=1 colspan=1>Nonlinear amplitude variation</td><td rowspan=1 colspan=1>Nonlinear self-interaction</td><td rowspan=1 colspan=1>Polynomial terms</td></tr><tr><td rowspan=1 colspan=1>Saturating response</td><td rowspan=1 colspan=1>Bounded nonlinear effects</td><td rowspan=1 colspan=1>Sigmoid, hyperbolic tangent functions</td></tr><tr><td rowspan=1 colspan=1>Exponential growth or decay</td><td rowspan=1 colspan=1>Growth/attenuation mechanisms</td><td rowspan=1 colspan=1>Exponential terms</td></tr><tr><td rowspan=1 colspan=1>Persistent cross-nodedependency</td><td rowspan=1 colspan=1>Network interaction mechanisms</td><td rowspan=1 colspan=1>Topology-aware interaction terms</td></tr><tr><td rowspan=1 colspan=1>Time-dependent residual drift</td><td rowspan=1 colspan=1>Explicit temporal effects</td><td rowspan=1 colspan=1>Time-dependent basis functions</td></tr></table>

## 1.4 Matrix representation of SAGE

To provide a compact mathematical representation of SAGE, we describe its equation-discovery procedure in matrix form. Consider a complex network with N nodes, clustered into K communities. The task is to infer the governing equation of each node from its observed state trajectories. We assume that the governing dynamics are sparse in the current candidate-function space, such that only a small subset of candidate terms is required to explain the observed dynamics. For simplicity, we consider one state dimension $x _ { i } ( t )$

of node i in community k. The procedure for other dimensions is the same. The dynamics of this state dimension are decomposed into a community-level component and an individual-level component:

$$
\begin{array} { c } { \displaystyle \frac { \mathrm { d } x _ { i } ( t ) } { \mathrm { d } t } = S _ { k } ( x _ { i } ( t ) ) + H _ { i } ( x _ { i } ( t ) ) } \\ { = \Theta ^ { ( r ) } ( x _ { i } ( t ) ) \xi _ { k } ^ { S ( r ) } + \Theta ^ { ( r ) } ( x _ { i } ( t ) ) \xi _ { i } ^ { H ( r ) } } \\ { = [ 1 , x _ { i } ( t ) , x _ { i } ^ { 2 } ( t ) , \sin ( x _ { i } ( t ) ) , \cdots ] \left[ \begin{array} { c } { \xi _ { k , 1 } ^ { S ( r ) } } \\ { \vdots } \\ { \xi _ { k , M } ^ { S ( r ) } } \end{array} \right] + [ 1 , x _ { i } ( t ) , x _ { i } ^ { 2 } ( t ) , \sin ( x _ { i } ( t ) ) , \cdots ] \left[ \begin{array} { c } { \xi _ { i , 1 } ^ { H ( r ) } } \\ { \vdots } \\ { \xi _ { k , M } ^ { H ( r ) } } \end{array} \right] } \end{array}\tag{S4}
$$

where $\pmb { \Theta } ^ { ( r ) } \big ( x _ { i } ( t ) \big ) \in \mathbb { R } ^ { M _ { r } }$ denotes the candidate-function vector available at search round ${ \sf r , }$ and $M _ { r }$ is the number of candidate terms in the current library. The library may contain self-dynamics, nonlinear state interactions, periodic terms and topology-aware interactions, as described in Section 1.2. The coefficient vector $\xi _ { K } ^ { S } \in \mathbb { R } ^ { M _ { r } }$ represents the community-level dynamics shared by nodes in community k, whereas $\pmb { \xi } _ { i } ^ { H } \in \mathbb { R } ^ { M _ { r } }$ represents the additional individual-level dynamics specific to node �.

## (1) Community-level dynamics

We first infer the community-level dynamics shared by the $N _ { k }$ nodes in community �. For the considered state dimension, the observed trajectories of these nodes are arranged into the state matrix

$$
X _ { k } = [ \begin{array} { c c c c c c } { x _ { 1 } ( 1 ) } & & & & & \\ { x _ { 2 } ( 1 ) } & { x _ { 2 } ( 2 ) } & { \cdots } & { x _ { 1 } ( T ) } \\ { \vdots } & { \vdots } & { \vdots } & { \vdots } \\ { x _ { N _ { k } } ( 1 ) } & { x _ { N _ { k } } ( 2 ) } & { \cdots } & { x _ { N _ { k } } ( T ) } \end{array} ] \in \mathbb { R } ^ { N _ { k } \times T }  x _ { k } = [ \begin{array} { c } { x _ { 1 } ( 1 ) } \\ { \vdots } \\ { x _ { 1 } ( T ) } \\ { x _ { 2 } ( 1 ) } \\ { \vdots } \\ { x _ { 2 } ( T ) } \\ { x _ { N _ { k } } ( T ) } \end{array} ] \in \mathbb { R } ^ { N _ { k } T }\tag{S5}
$$

The corresponding temporal derivatives are arranged in the same order:

$$
\frac { \mathrm { d } x } { \mathrm { d } t } = [ \begin{array} { c c c c c c } { \frac { \mathrm { d } x _ { 1 } ( 1 ) } { \mathrm { d } t } } & { \frac { \mathrm { d } x _ { 1 } ( 2 ) } { \mathrm { d } t } } & { . . . } & { \frac { \mathrm { d } x _ { 1 } ( T ) } { \mathrm { d } t } } \\ { \frac { \mathrm { d } x _ { 2 } ( 1 ) } { \mathrm { d } t } } & { \frac { \mathrm { d } x _ { 2 } ( 2 ) } { \mathrm { d } t } } & { . . . } & { \frac { \mathrm { d } x _ { 2 } ( T ) } { \mathrm { d } t } } \\ { \frac { \mathrm { d } x _ { N } ( 1 ) } { \mathrm { d } t } } & { \frac { \mathrm { d } x _ { N } ( 2 ) } { \mathrm { d } t } } & { . . . } & { \frac { \mathrm { d } x _ { N } ( T ) } { \mathrm { d } t } } \\ { \frac { \mathrm { d } x _ { N } ( 1 ) } { \mathrm { d } t } } & { \frac { \mathrm { d } x _ { N } ( 2 ) } { \mathrm { d } t } } & { . . . } & { \frac { \mathrm { d } x _ { N } ( T ) } { \mathrm { d } t } } \end{array} ] \in \mathbb { R } ^ { N _ { k } \times T }  \frac { \mathrm { d } x } { \mathrm { d } t } = [ \begin{array} { c } { \frac { \mathrm { d } x _ { 1 } ( 1 ) } { \mathrm { d } t } } \\ { \vdots } \\ { \frac { \mathrm { d } x _ { 1 } ( T ) } { \mathrm { d } t } } \\ { \frac { \mathrm { d } x _ { 2 } ( 1 ) } { \mathrm { d } t } } \\ { \vdots } \\ { \frac { \mathrm { d } x _ { N } ( T ) } { \mathrm { d } t } } \\ { \frac { \mathrm { d } x _ { N } ( T ) } { \mathrm { d } t } } \end{array} ] \in \mathbb { R } ^ { N _ { k } T }\tag{S6}
$$

The candidate functions in the current library $L ^ { ( r ) }$ are evaluated at each node–time sample to construct the community-level design matrix $\Theta _ { k } ^ { ( r ) }$

$$
\Theta _ { k } ^ { ( r ) } ( x ) = \left[ \begin{array} { c c c c c c } { 1 } & { x _ { 1 } ( 1 ) } & { x _ { 1 } ^ { 2 } ( 1 ) } & { \sin ( x _ { 1 } ( 1 ) ) } & { \cdots } \\ { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } \\ { 1 } & { x _ { 1 } ( T ) } & { x _ { 1 } ^ { 2 } ( T ) } & { \sin ( x _ { 1 } ( T ) ) } & { \cdots } \\ { 1 } & { x _ { 2 } ( 1 ) } & { x _ { 2 } ^ { 2 } ( 1 ) } & { \sin ( x _ { 2 } ( 1 ) ) } & { \cdots } \\ { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } \\ { 1 } & { x _ { 2 } ( T ) } & { x _ { 2 } ^ { 2 } ( T ) } & { \sin ( x _ { 2 } ( T ) ) } & { \cdots } \\ { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } \\ { 1 } & { x _ { N _ { k } } ( T ) } & { x _ { N _ { k } } ^ { 2 } ( T ) } & { \sin ( x _ { N _ { k } } ( T ) ) } & { \cdots } \end{array} \right] \in \mathbb { R } ^ { N _ { k } T \times M _ { r } }\tag{S7}
$$

For a candidate community-level coefficient vector $\xi _ { k } ^ { S }$ , the stacked dynamics are represented as

$$
\left[ \begin{array} { c } { \frac { \operatorname { d } x _ { 1 } ( 1 ) } { \operatorname { d } t } } \\ { \vdots } \\ { \frac { \operatorname { d } x _ { 1 } ( T ) } { \operatorname { d } t } } \\ { \frac { \operatorname { d } x _ { 2 } ( 1 ) } { \operatorname { d } t } } \\ { \vdots } \\ { \frac { \operatorname { d } x _ { 2 } ( T ) } { \operatorname { d } t } } \\ { \frac { \operatorname { d } x _ { 2 } ( T ) } { \operatorname { d } t } } \\ { \frac { \operatorname { d } x _ { 3 } } { \operatorname { d } t } } \end{array} \right] \overset { { , } } { = } \left[ \begin{array} { c c c c c } { 1 } & { x _ { 1 } ( 1 ) } & { x _ { 1 } ^ { 2 } ( 1 ) } & { \sin ( x _ { 1 } ( 1 ) ) } & { \cdots } \\ { \vdots } & { \vdots } & { \vdots } & { \vdots } \\ { 1 } & { x _ { 1 } ( T ) } & { x _ { 2 } ^ { 2 } ( T ) } & { \sin ( x _ { 1 } ( T ) ) } & { \cdots } \\ { 1 } & { x _ { 2 } ( 1 ) } & { x _ { 2 } ^ { 2 } ( 1 ) } & { \sin ( x _ { 2 } ( 1 ) ) } & { \cdots } \\ { \vdots } & { \vdots } & { \vdots } & { \vdots } \\ { \vdots } & { x _ { 2 } ( T ) } & { x _ { 2 } ^ { 2 } ( T ) } & { \sin ( x _ { 2 } ( T ) ) } & { \cdots } \\ { \vdots } & { \vdots } & { \vdots } & { \vdots } \\ { x _ { N _ { k } } ( T ) } & { x _ { N _ { k } } ^ { 2 } ( T ) } & { \sin ( x _ { N _ { k } } ( T ) ) } & { \cdots } \end{array} \right] \overset { \mathcal { \ell } \overset { \mathcal { S } ( F ) } { } } { \underset { \mathrm { d } x _ { k } ( T ) } { \operatorname { d } x } }\tag{S8}
$$

The objective function of this sparse regression problem is given as:

$$
E _ { K } ( \xi ) = \frac { 1 } { N _ { K } T } \big \| x _ { K } - \Theta _ { k } ^ { ( r ) } \xi \big \| _ { 2 } ^ { 2 } , \qquad C ( \xi ) = \| \xi \| _ { 0 }\tag{S9}
$$

Bootstrap STLSQ, cross-validated Lasso and Ridge regression under multiple threshold settings generate multiple candidate coefficient vectors and support sets from Eq. (S9). These candidates are compared using validation error $E _ { K }$ , structural complexity � and stability evidence derived from cross-solver consistency, bootstrap resampling, and coefficient sign consistency. Pareto filtering retains candidate structures that are not simultaneously dominated in validation accuracy and structural complexity, after which the selected community-level structure is refitted numerically to obtain $\xi _ { k } ^ { S }$ .The non-zero entries of $\xi _ { k } ^ { S }$ define the community-level terms shared by nodes in community �.

## (2) Individual-level dynamics

After identifying the community-level structure, SAGE examines the residual dynamics of each node to determine whether additional individual-level terms are required. For node � in community �, the trajectory of the considered state dimension is written as:

$$
\pmb { x } _ { i } ( t ) = [ x _ { i } ( 1 ) \quad x _ { i } ( 2 ) \quad \cdots \quad x _ { i } ( T ) ] ^ { T } \in \mathbb { R } ^ { T }\tag{S10}
$$

Its temporal derivatives are arranged as:

$$
{ \frac { \mathrm { d } x _ { \mathrm { i } } ( t ) } { \mathrm { d } t } } = \left[ { \frac { \mathrm { d } x _ { i } ( 1 ) } { \mathrm { d } t } } \quad { \frac { \mathrm { d } x _ { i } ( 2 ) } { \mathrm { d } t } } \quad \ldots \quad { \frac { \mathrm { d } x _ { i } ( T ) } { \mathrm { d } t } } \right] ^ { T } \in \mathbb { R } ^ { T }\tag{S11}
$$

The current candidate library is evaluated along this trajectory to form the node-specific design matrix:

$$
\Theta _ { k } ^ { ( r ) } ( x ) = \left[ \begin{array} { c c c c c } { 1 } & { x _ { i } ( 1 ) } & { x _ { i } ^ { 2 } ( 1 ) } & { \sin ( x _ { i } ( 1 ) ) } & { \cdots } \\ { 1 } & { x _ { i } ( 2 ) } & { x _ { i } ^ { 2 } ( 2 ) } & { \sin ( x _ { i } ( 2 ) ) } & { \cdots } \\ { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } \\ { 1 } & { x _ { i } ( T ) } & { x _ { i } ^ { 2 } ( T ) } & { \sin ( x _ { i } ( T ) ) } & { \cdots } \end{array} \right] \in \mathbb { R } ^ { T \times M _ { r } }\tag{S12}
$$

After subtracting the fitted community-level contribution, the unexplained dynamics of node � are:

$$
R s e _ { i } = \pmb { x _ { i } } - \pmb { \Theta } _ { i } ^ { ( r ) } \pmb { \xi } _ { k } ^ { S ( r ) } \ , \ R s e _ { i } \in \mathbb { R } ^ { T }
$$

$$
\left[ \begin{array} { c } { R s e _ { i } ( 1 ) } \\ { R s e _ { i } ( 2 ) } \\ { \vdots } \\ { R s e _ { i } ( T ) } \end{array} \right] = \left[ \begin{array} { c } { \varDelta x _ { i } ( 1 ) } \\ { \varDelta x _ { i } ( 2 ) } \\ { \vdots } \\ { \varDelta x _ { i } ( T ) } \\ { \varDelta x _ { i } } \end{array} \right] - \left[ \begin{array} { c c c c c } { 1 } & { x _ { i } ( 1 ) } & { x _ { i } ^ { 2 } ( 1 ) } & { \sin ( x _ { i } ( 1 ) ) } & { \cdots } \\ { 1 } & { x _ { i } ( 2 ) } & { x _ { i } ^ { 2 } ( 2 ) } & { \sin ( x _ { i } ( 2 ) ) } & { \cdots } \\ { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } \\ { 1 } & { x _ { i } ( T ) } & { x _ { i } ^ { 2 } ( T ) } & { \sin ( x _ { i } ( T ) ) } & { \cdots } \end{array} \right] \left[ \begin{array} { c } { \hat { \xi } _ { k , 1 } ^ { S ( r ) } } \\ { \hat { \xi } _ { k , 2 } ^ { S ( r ) } } \\ { \vdots } \\ { \vdots } \\ { \hat { \xi } _ { k , M } ^ { S ( r ) } } \end{array} \right]\tag{S13}
$$

Individual-level terms are then used to explain this residual:

$$
\pmb { r } _ { i } = \pmb { \Theta } _ { i } ^ { ( r ) } \pmb { \xi } _ { i } ^ { H ( r ) }\tag{S14}
$$

A non-zero entry in $\xi _ { i } ^ { H }$ may therefore represent either an additional node-specific function or a nodespecific coefficient adjustment to a function already present in the community-level structure.

Combining the selected community-level and individual-level components gives the governing equation of node � in community � :

$$
\frac { \mathrm { d } { x } _ { i } ( t ) } { \mathrm { d } { t } } = S _ { k } \big ( x _ { i } ( t ) \big ) + H _ { i } \big ( x _ { i } ( t ) \big ) = \Theta ^ { ( r ) } \big ( x _ { i } ( t ) \big ) \hat { \xi } _ { k } ^ { S } + \Theta ^ { ( r ) } \big ( x _ { i } ( t ) \big ) \hat { \xi } _ { i } ^ { H }\tag{S15}
$$

## 1.5 Pseudocode of SAGE

Algorithm 1 summarizes the complete Library–Discovery–Review procedure of SAGE for governingequation discovery. The framework first initializes the candidate function library from the observed nodestate observations and then iteratively searches, evaluates, and refines governing equations. The Library Agent constructs and updates the candidate function library based on residual patterns and diagnostic feedback. The Discovery Agent identifies sparse equation structures from the current library using numerical regression and selects candidate structures according to validation error, structural complexity, and stability evidence. The Review Agent evaluates the selected equation using derivative fit, coefficient stability, and state prediction, and provides repair instructions when the equation fails the predefined criteria.

During the iterative search, all adaptive operations, including candidate-library updates, equation selection, and repair decisions, are conducted using only the training and validation sets. The test set is completely isolated from the search process and is used only for the final evaluation after an equation receives

ACCEPT. All numerical configurations, search parameters, and random seeds are specified in a unified configuration file to ensure reproducibility across experiments.

```latex
Algorithm 1 | LLM-SAGE for governing-equation discovery in community-structured complex
networks
1: Input:
$\{ \mathbf { x } _ { i } ( t ) \mid i = 1 , . . . , N ; t = 1 , . . . , T \}$ : node-state observations
A: adiacency matrix, when available
C: community labels, provided or inferred from node-state trajectories
$R _ { \mathrm { m a x } } \mathrm { : }$ maximum number of repair rounds
2: Output: $\mathbf { S } _ { k } ( \mathbf { x } _ { i } ( t ) , \mathbf { A } ) , H _ { i } ( \mathbf { x } _ { i } ( t ) , \mathbf { A } ) ,$ , and final governing equations
3: # Data preprocessing and initialization
4: Preprocess observations, estimate temporal derivatives, and split data into training
validation, and test sets
5: Determine community assignments and construct initial candidate library $\mathcal { L } ^ { ( 0 ) }$
6: for $r = 0$ to $R _ { \mathrm { m a x } }$ do
7: Construct candidate-function vectors $\Theta ^ { ( r ) } ( \mathbf { x } _ { i } ( t ) )$ from $\mathcal { L } ^ { ( r ) }$ and assemble
community-level design matrices $\Theta _ { k } ^ { ( r ) }$
8: The Discovery Agent generates sparse candidate equations using bootstrap STLSQ
cross-validated Lasso, and validation-guided Ridge sparsification.
9: Aggregate stability evidence and construct Pareto candidate equations
10: Select community-level support structures and refit coefficients $\xi _ { k } ^ { S ( r ) }$
11: Compute residual dynamics
$\begin{array} { r } { \mathrm { R e s } _ { i } ^ { H \left( r \right) } \left( t \right) = \frac { d { \bf x } _ { i } ( t ) } { d t } - \Theta ^ { \left( r \right) } \left( { \bf x } _ { i } ( t ) \right) \xi _ { k } ^ { S ( r ) } } \end{array}$
and identify individual-level candidate terms.
12: Combine community-level and individual-level support sets,
re-estimate $\xi _ { k } ^ { S ( r ) }$ and $\xi _ { i } ^ { H ( r ) }$ , and simplify the model.
13: The Review Agent evaluates derivative fit, coefficient stability,
and one-step state prediction
14: if ACCEPT then
15: Evaluate on independent test set and return final equations.
16: else if REPAIR and $r < R _ { \operatorname* { m a x } }$ then
17: Map repair plan to function families $\mathcal { R } ^ { ( r ) }$ and update:
18: $\mathcal { L } ^ { ( r + 1 ) } = \mathcal { P } \left[ \mathcal { L } ^ { ( r ) } \cup \mathcal { S } \left( \mathcal { R } ^ { ( r ) } \right) \right]$
19: else
20: Return REJECT and diagnostic report
21: end if
22: end for
23: # Note
24: All candidate-library updates, model-selection steps, and repair decisions use only the
training and validation sets. The test set is excluded from all adaptive decisions until
a model is accepted.
```

## 1.6 LLM prompts and structured returns

To illustrate how SAGE integrates language-mediated decisions into the numerical equation-discovery pipeline, we provide a representative execution trace on a real-world transportation network. This example records the structured information provided to each LLM agent, the corresponding prompts, and the

returned decisions during one complete Library–Discovery–Review repair cycle. The following example demonstrates how language-based decisions interact with numerical equation discovery.

## Library Agent: residual diagnosis and function-family expansion

This call corresponds to the adaptive library update described in Library Agent. The Review Agent provides diagnostic feedback on the current equation, including unexplained residual patterns and unstable terms. Based on this feedback, the Library Agent selects additional or revised function families from the predefined catalog to update the candidate library.

Table S4. LLM interaction trace of the Library Agent
<table><tr><td>Representative prompt</td><td>Example structured return</td></tr><tr><td>Representative prompt</td><td></td></tr><tr><td>Agent role: You are the Library Agent in SAGE.</td><td>"diagnosis": [</td></tr><tr><td>Input information provided by the numerical module:</td><td>"persistent oscillatory residual</td></tr><tr><td>Residual diagnostics:</td><td>patterns remain after the current library fitting"</td></tr><tr><td>residual std ratio: relative residual magnitude</td><td></td></tr><tr><td>lag1_autocorr: temporal correlation of residuals</td><td>1, "recommended_families": [</td></tr><tr><td>dominant_frequency: dominant frequency components detected from residual spectra</td><td>"trigonometric",</td></tr><tr><td>base_term_correlations: correlations between residuals and current candidate terms</td><td>"state time modulation" 1,</td></tr><tr><td>network_residual_correlation: correlation between residuals and</td><td>"evidence": [</td></tr><tr><td>network topology</td><td>"dominant_frequency",</td></tr><tr><td>community_residual_stats: residual statistics across communities</td><td>"high_lag1_autocorr",</td></tr><tr><td>Current candidate-function library: Existing function families and</td><td></td></tr><tr><td>candidate terms used in the current search round.</td><td>"residual_pattern_not_explained_by_curre</td></tr><tr><td>Allowed expansion families: Function families that can be added from</td><td>nt_terms"</td></tr><tr><td>the predefined candidate function catalog.</td><td>1,</td></tr><tr><td>Task:</td><td>"rationale": "The residual dynamics</td></tr><tr><td>Diagnose whether the current candidate library lacks important</td><td>exhibit stable periodic behavior that is not</td></tr><tr><td>dynamical mechanisms.</td><td>captured by the current candidate library</td></tr><tr><td>Identify the most plausible missing function families based only on the</td><td>Adding trigonometric and state-time</td></tr><tr><td>provided diagnostics.</td><td>modulation terms may provide additional</td></tr><tr><td>Select function families only from the allowed expansion catalog.</td><td></td></tr><tr><td></td><td>mechanisms to represent the observed temporal variation."</td></tr><tr><td>Do not generate governing equations, mathematical expressions, or</td><td></td></tr><tr><td>numerical coefficients.</td><td></td></tr><tr><td>Return a structured JSON response following the required format.</td><td></td></tr><tr><td>Required JSON output format:</td><td></td></tr><tr><td>"diagnosis": [ "brief description of unexplained dynamical patterns"],</td><td></td></tr><tr><td>"recommended_families": ["selected function families from the</td><td></td></tr><tr><td>allowed catalog"], "evidence": [ "diagnostic signals supporting the decision"],</td><td></td></tr><tr><td>"rationale": "brief explanation of why these families may explain the</td><td></td></tr><tr><td>residual dynamics" 一</td><td></td></tr></table>

## Discovery Agent: Pareto candidate structure selection

This call corresponds to the Pareto candidate selection described in Discovery Agent. The numerical module first generates candidate equations using Bootstrap STLSQ, Lasso, and Ridge regression. These candidates are evaluated using validation error, structural complexity, and stability evidence. The Discovery Agent then compares the Pareto candidates and selects the most appropriate equation structure.

Table S5. LLM interaction trace of the Discovery Agent
<table><tr><td>Representative prompt</td><td>Example structured return</td></tr><tr><td>Representative prompt</td><td>2</td></tr><tr><td>Agent role: You are the Discovery Agent in SAGE.</td><td>&quot;decision&quot;: &quot;SELECT&quot;,</td></tr><tr><td>Input information provided by the numerical module:</td><td>&quot;model_id&quot;: &quot;Pareto-03&quot;,</td></tr><tr><td>Pareto candidate equations: Candidate structures generated by</td><td>&quot;supported_terms&quot;: [</td></tr><tr><td>sparse regression methods and retained after Pareto filtering.</td><td>&quot;self_linear&quot;,</td></tr><tr><td>Candidate evaluation evidence:</td><td>&quot;polynomial_self&quot;,</td></tr><tr><td>validation_error: prediction error on the validation set</td><td>&quot;network diffusion&quot;</td></tr><tr><td>complexity: number of retained candidate terms</td><td>1,</td></tr><tr><td>stability_matrix: term selection consistency across numerical solvers</td><td>&quot;concerns&quot;: [</td></tr><tr><td>bootstrap_stability: selection frequency from bootstrap resampling</td><td>&quot;weak evidence for additional</td></tr><tr><td>sign_consistency: coefficient sign consistency across repeated</td><td>nonlinear coupling terms&quot;</td></tr><tr><td>estimations Term metadata: Function-family information and physical interpretation</td><td>1,</td></tr><tr><td>labels of candidate terms.</td><td>&quot;rationale&quot;: &quot;Pareto-03 achieves a</td></tr><tr><td>Task:</td><td>favorable balance between validation</td></tr><tr><td>Compare only the supplied Pareto candidate structures.</td><td>accuracy, model sparsity, and cross-solver</td></tr><tr><td></td><td>stability. The retained terms are</td></tr><tr><td>Select the candidate equation that provides the best trade-off between</td><td>consistently supported by Bootstrap</td></tr><tr><td>accuracy, sparsity, and stability.</td><td>STLSQ, LassoCV, and Ridge threshold</td></tr></table>

```jsonl
Representative prompt Example structured return
Prefer structures supported by multiple numerical solvers and }
repeated estimations.
Do not add new candidate terms.
Do not modify or estimate numerical coefficients.
Return a structured JSON response following the required format.
Required JSON output format:
{
"decision": "SELECT|REVISE",
"model_id": "identifier of selected Pareto candidate",
"supported_terms": [
"retained candidate terms"
],
"concerns": [
"remaining limitations or uncertainties"
],
"rationale": "brief explanation of the selection decision"
}
```  
Implementation note: The LLM only performs structure-level adjudication among existing Pareto candidates, whereas coefficient estimation and model reconstruction are performed by numerical regression.

## Review Agent: reflective review and repair

This call corresponds to the review stage described in Review Agent. The Review Agent evaluates the selected equation using derivative fit, coefficient stability, and state prediction. Based on these criteria, it returns ACCEPT, REPAIR, or REJECT. When REPAIR is returned, the Review Agent provides diagnostic information that guides the next library update round.

Table S6. LLM interaction trace of the Review Agent
<table><tr><td>Representative prompt</td><td>Example structured return</td></tr><tr><td>Agent role: You are the Review Agent in SAGE.</td><td>J</td></tr><tr><td>Input information provided by the numerical module:</td><td>"decision": "REPAIR"</td></tr><tr><td>Current equation structure: Selected sparse governing equation</td><td>"failed_checks": [</td></tr><tr><td>and retained function terms.</td><td>"state_prediction",</td></tr><tr><td>Evaluation evidence:</td><td>"structured_residual"</td></tr><tr><td>derivative_fit: agreement between observed and predicted</td><td>],</td></tr><tr><td>derivatives</td><td></td></tr><tr><td>Representative prompt bootstrap_stability: coefficient uncertainty estimated from bootstrap resampling state_prediction: step prediction performance on the validation set residual_diagnostics: remaining residual patterns and unexplained structures Task: Evaluate whether the current equation satisfies the predefined review criteria. Return exactly one decision: ACCEPT, REPAIR, or REJECT. If ACCEPT, indicate that the equation can proceed to independent test evaluation. If REPAIR, identify the failed evaluation aspects and recommend function families that may capture the missing dynamics. If REJECT, summarize the main reasons why a stable equation cannot be supported. Do not modify numerical coefficients or generate new governing equations. Return a structured JSON response following the required format. Required JSON output format: { "decision": "ACCEPT|REPAIR|REJECT", "failed_checks": [</td><td>Example structured return "failure mode": "The equation reproduces local derivative patterns but leaves a persistent oscillatory component that affects future state prediction.", "repair_families": [ "trigonometric", "state_time_modulation" 1, "rationale": "Residual diagnostics indicate a stable periodic pattern not explained by the current candidate library. Additional oscillatory function families are recommended for the next search round." }</td></tr></table>

When ACCEPT is returned, the selected equation is transferred to the independent test set for final evaluation. When REPAIR is returned, the identified failure patterns and recommended function families are passed to the Library Agent, which expands or modifies the candidate function library according to Equation 12. When REJECT is returned, the search terminates and a diagnostic report is generated to summarize the remaining limitations, failed evaluations, and iteration history.

## LLM configuration.

All language-model calls were made through the gpt-5.6-sol API using the chat-completions interface in JSON mode. The temperature was set to 0.2 to reduce response variability, and the request timeout was 60 s. The maximum number of repair rounds was set to five. All LLM outputs were required to follow the corresponding JSON schema before entering the numerical pipeline.

## 2. Supplementary details and results for synthetic complex network

## 2.1 Supplementary details for the single-dynamics network

The Lotka–Volterra system (predator–prey model), originally proposed by Alfred J. Lotka and Vito Volterra, is a classical nonlinear dynamical system widely used to describe interactions among two or more populations in ecological systems. It is particularly suited for modeling ecological processes such as predation, competition, and symbiosis. In this study, we extend the Lotka–Volterra dynamics to a community-structured complex network. The network consists of 60 nodes, where each node represents an ecological unit and its state represents the corresponding population size. Network topology is generated using the Stochastic Block Model (SBM) and clustered into three communities. Nodes within the same community have a higher connection probability, whereas connections between different communities are less frequent. The simulation runs for a total duration of 30 with a time step of 0.01. The governing equations are given as follows:

$$
\left\{ \begin{array} { l } { \displaystyle \frac { \mathrm { d } \mathbf { x } _ { i } } { \mathrm { d } t } = \alpha _ { k } x _ { \mathrm { i } } - \beta _ { k } x _ { i } y _ { i } + \varepsilon _ { \mathrm { k } } \sum _ { j \in \mathbb { N } _ { i } } x _ { j } + \zeta _ { i } \mathrm { s i n } ( 1 . 5 t ) x _ { i } } \\ { \displaystyle \frac { \mathrm { d } y _ { i } } { \mathrm { d } t } = \delta _ { k } x _ { i } y _ { i } - \gamma _ { k } y _ { i } + \varsigma _ { i } y _ { i } } \end{array} \right.\tag{S16}
$$

where $\alpha _ { k } , \beta _ { k } , \varepsilon _ { \mathrm { k } } , \delta _ { k } , \gamma _ { k }$ are community-level coefficients for node i within community k. All nodes within community k share the same community-level coefficients, which characterize their consistent dynamic behavior. In contrast, different communities are assigned to distinct and significantly different coefficients, highlighting dynamical differences across communities. $\zeta _ { i } , \varsigma _ { i }$ are individual-level coefficients for node i and represent variation at the individual level. Each node carries distinct coefficients that highlight its unique dynamic behavior. $\varepsilon _ { \mathrm { k } } \sum _ { j \in \mathbb { N } _ { i } } x _ { j }$ denotes the interaction effect of the neighboring nodes on node � . The detailed dynamics of the three communities in the single-dynamics network are as follows.

$$
\{ \begin{array} { l l } { \displaystyle \mathrm { G r o u p 1 L } \{ \frac { \mathrm { d } \mathrm { x i } _ { i } } { \mathrm { d } t } = 2 x _ { i } - 1 x _ { i } y _ { i } + 0 . 3 \sum _ { j ^ { \mathrm { E q } } } x _ { j } + \zeta _ { i } x _ { i }  } \\ { \displaystyle \mathrm { G r o u p 1 L } \{ \frac { \mathrm { d } y _ { i } } { \mathrm { d } t } = 0 . 5 x _ { i } y _ { i } - y _ { i } + \zeta _ { i } y _ { i }  } \\ { \displaystyle \mathrm { G r o u p 2 } \{ \frac { \mathrm { d } \mathrm { x i } _ { i } } { \mathrm { d } t } = 1 . 5 x _ { i } - 0 . 3 x _ { i } y _ { i } + 0 . 3 \sum _ { j ^ { \mathrm { e q } } } x _ { j } + \zeta _ { i } x _ { i }  } \\ { \displaystyle \mathrm { G r o u p 2 } \{ \frac { \mathrm { d } y _ { i } } { \mathrm { d } t } = 0 . 4 x _ { i } y _ { i } - 0 . 8 y _ { i } + \zeta _ { i } y _ { i } } \\ { \displaystyle \mathrm { G r o u p 3 } \{ \frac { \mathrm { d } \mathrm { x i } _ { i } } { \mathrm { ~ d } t } = 3 x _ { i } - 1 . 5 x _ { i } y _ { i } + 0 . 3 \sum _ { j ^ { \mathrm { e q } } } x _ { j } + \zeta _ { i } x _ { i }  } \\ { \displaystyle \mathrm {  \mathrm { d } \mathrm { y t } _ { i } = 0 . 6 x _ { i } y _ { i } - 1 . 2 y _ { i } + \zeta _ { i } y _ { i } \} } } \end{array}\tag{S17}
$$

where $\xi _ { i }$ and $\varsigma _ { i }$ represent individual-level coefficients of each node, which follow a normal distribution with mean 0 and variance 0.1. Figure S1 illustrates the inference results across the three communities.

The inference results for a representative node in the group 1  
![](images/032258616d65467ab89bf3c6383c86791409d8e2f83dea5a224b2ce2e9d7fae3.jpg)

The inference results for a representative node in the group 2  
![](images/c2ac5d6026c6edb524c72dda02bc6954b216716dbfdc9e7150768edb5ccb8fae.jpg)

The inference results for a representative node in the group 3  
![](images/6f7d972a1777f81f3ed3da3e936a525cbf34f3387fa3ffc3c4eda14ad23bf4be.jpg)  
Figure S1. Validating SAGE on a single-dynamic network with 60 nodes divided into three communities. A-L Inference results for three representative nodes, one randomly selected from each community.

## 2.2 Supplementary details for the multi-dynamics network

Chaotic systems, as typical nonlinear dynamical systems, are characterized by strong nonlinearity, sensitivity to initial conditions, non-periodicity, and boundedness. In this study, we extend classical chaotic systems to complex networks and construct a chaotic complex network composed of nodes and edges. The network consists of 60 nodes, with its topological structure generated using the Stochastic Block Model (SBM) and is further clustered into three communities. The nodes in the three communities follow three representative chaotic systems: the Lorenz system, the Rössler system, and the Chen system. The simulation runs for a total duration of 100 with a time step of 0.01. The governing equations are given as follows:

$$
\begin{array}{c} \begin{array} { r l } & { \mathrm { i m } _ { \mathrm { L G } } = } \\ & { \mathrm { i m } _ { \mathrm { N L } } } \\ & { \mathrm { i m } _ { \mathrm { N L } } } \\ & { \mathrm { i m } _ { \mathrm { N L } } } \\ & { \mathrm { i m } _ { \mathrm { N L } } } \\ & { \mathrm { i m } _ { \mathrm { N L } } } \\ & { \mathrm { R e s s u p . } } \\ & { \mathrm { i f f f u c e } } \\ & { \mathrm { i f f f u c e } } \\ & { \mathrm { i f f f u c e } } \\ & { \mathrm { i f f f u c e } } \\ & { \mathrm { i f f f u c e } } \\ & { \mathrm { i f f f u c e } } \\ & { \mathrm { i f f f u c e } } \\ & { \mathrm { i f f f u c e } } \\ & { \mathrm { i f f f u c e } } \\ & { \mathrm { i f f f u c e } } \\ & { \mathrm { i f f f u c e } } \\ & { \mathrm { i f f f u c e } } \\ & { \mathrm { e t e r m } _ { \mathrm { N L } } } \end{array}  ,  \end{array}\tag{S18}
$$

where $a , b , { \mathsf { c } }$ are community-level coefficients. Each node follows a governing equation derived from one of the three systems-Lorenz, Rössler, or Chen. $\varsigma _ { i } , \zeta _ { i } , \xi$ are individual-level coefficients for node � . The detailed dynamics of the three communities in the multi-dynamics network are given as follows.

$$
\begin{array} { r l } & { \quad ( \frac { \partial A _ { x } } { \partial x } = 1 0 ( g _ { x } - x _ { x } ) + s _ { x } x _ { x }  } \\ & { \quad  \mathrm { ( d i s f . ~ } } \\ & { \quad  \mathrm { ( d i s f . ~ } - x _ { x } ( 2 0 0 - g _ { x } ) - s _ { y } x _ { x } + s _ { x } x _ { y } , } \\ & { \quad  \mathrm { ( d i s f . ~ } } \\ & { \quad \mathrm { ( d i s f . ~ } ) - s _ { x } x _ { y } - \frac { B _ { x } } { 2 } s _ { x } + \xi _ { y } x _ { z }   } \\ & { \quad  \mathrm { ( d i s f . ~ } } \\ & { \quad  \mathrm { ( d i s f . ~ } - x _ { x } - z _ { x } + s _ { y } x _ { z }   } \\ & { \quad  \mathrm { ( d i s f . ~ } } \\ & { \quad  \mathrm { ( d i s f . ~ } - x _ { x } + 0 . 3 \mathrm { ( d i s f . ~ } + s _ { x } x _ { y } )    } \\ & { \quad   \mathrm { ( d i s f . ~ } = 1 \mathrm { ~ } + 2 . \mathrm { ( d i s f . ~ } - s _ { x } ) \cdot \mathrm { ( d i s f . ~ } } \\ & { \quad  \mathrm { ( d i s f . ~ } - x _ { x } ) \cdot \mathrm { ( d i s f . ~ } } \\ & { \quad  \mathrm { ( d i s f . ~ } - x _ { x } ( 2 0 0 - g _ { x } ) \cdot \mathrm { ( ~ c o s s . ~ } )  \ } \ .   \\ & { \quad   \mathrm { ( d i s f . ~ } - x _ { x } ( 2 0 0 - g _ { x } ) \cdot \mathrm { ( ~ c o s s . ~ } \xi _ { x } )  \   } \\ & { \quad   \mathrm { ( d i s f . ~ } - x _ { x } ( 2 0 0 - g _ { x } ) \cdot \mathrm { ( ~ c o s s . ~ } \xi _ { x } )  \  } \\ &  \quad   \mathrm { ( d i s f . ~ } - x _  x  \end{array}\tag{S19}
$$

Where $\varsigma _ { i } , \zeta _ { i } , \xi _ { i }$ represent the individual-level terms of each node, which follow a normal distribution with mean 0 and variance 0.1. Figure S2 illustrates the inference results across the three communities.

![](images/8ae8ec57c9c4aaad7ca46bbd87ffd14440d03070e7d6c99c7dac43095510ae7c.jpg)  
Figure S2. Validating SAGE on a multi-dynamics network with 60 nodes divided into three communities. A-L Inference results for three representative nodes, one randomly selected from each community.

## 3. Supplementary Details for Real-World Complex Network Analysis

## 3.1 Urban mobile communication network

We use the urban mobile communication dataset from the Telecom Italia Big Data Challenge, which records mobile communication activity in Milan, Italy, from November 1 to November 7, 2013, including Internet activity, phone-call activity, and SMS activity. We select 48 spatial grid cells as network nodes. For each node � , the state vector is defined as $\pmb { x } _ { i } ( t ) = [ x _ { i } ( t ) , y _ { i } ( t ) , z _ { i } ( t ) ] ^ { \mathrm { T } }$ , where $x _ { i } ( t ) , y _ { i } ( t )$ , and $z _ { i } ( t )$ denote Internet activity, phone-call activity, and SMS activity, respectively. The processed dataset contains 168 steps for each node.

Network topology is constructed according to geographical adjacency. Specifically, two grid cells are connected if they are spatially adjacent under a four-neighbor structure, so that the adjacency matrix represents local interactions among neighboring urban regions. The resulting node-state trajectories and spatial topology are used jointly for hierarchical governing-equation discovery. Detailed inference and interpretation results are presented in the following section.

## 3.2 Brain EEG network

In this study, we use the EEG Motor Movement Imagery Dataset (EEGMMIDB) from PhysioNet and select the S001R01 recording. The dataset contains 64 EEG channels, which are treated as network nodes. The signals are band-pass filtered between 1 and 20 Hz, re-referenced using common average referencing (CAR), and transformed using the Hilbert transform. For each EEG channel, the analytic signal obtained from the Hilbert transformation is represented by two state variables, where $x _ { i } ( t )$ represents the filtered EEG signal, corresponding to the real component of the analytic signal, and $y _ { i } ( t ) = H [ x _ { i } ( t ) ]$ ] represents the corresponding Hilbert quadrature component. The processed data contains 2,440-time steps and two state variables for each node.

The EEG network topology is constructed from amplitude-envelope functional connectivity between channels. Specifically, the amplitude envelope of each channel is extracted from its analytic signal, and the absolute Pearson correlation coefficient is calculated for every pair of channel envelopes over the entire recording. Among all 2,016 channel pairs, the top 20% with the largest absolute correlations are retained, yielding 403 undirected edges with an empirical cutoff of approximately $\lvert r \rvert { = } 0 . 6 6 9$ . The resulting adjacency matrix � is symmetric and binary without self-loops, where $\pmb { A } _ { i j } = 1$ indicates strong amplitude-envelope co-fluctuation between channels � and $j ,$ and $\pmb { A } _ { i j } = 0$ otherwise. This topology represents sensor-level functional connectivity rather than anatomical connectivity or spatial-neighborhood relationships.

The 64 EEG channels are organized into five dynamical communities for subsequent equation discovery and hierarchical analysis. These communities are used to identify shared dynamical structures among channels and distinguish community-level mechanisms from channel-specific variations. Detailed governing-equation inference and interpretation results are presented below.

![](images/e4cd11ca3e3546050974b8f0b3e03f35b062c512bcac0ffacd5817317f41593d.jpg)  
Figure S3. Dynamical interpretation of the governing equations discovered for the brain EEG network. The discovered EEG dynamics are dominated by linear cross-state coupling together with constant bias terms. The coupling coefficients reveal an antisymmetric interaction between the two state variables, consistent with feedback-like oscillatory structure, whereas the constant terms indicate a shifted dynamical baseline. The hierarchical decomposition further distinguishes shared community-level coupling patterns from individual-level coefficient variations and baseline shifts. Model reliability is supported by high one-step prediction accuracy and low bootstrap coefficient variation. Purple and green shading denote community-level and individual-level terms, respectively.

## 3.3 Power grid network

We use the IEEE 39-bus New England power system as a benchmark power-grid network. The system contains 39 buses, which are treated as network nodes and organized into three dynamical communities for hierarchical governing-equation discovery. Each bus � is described by a single state variable $x _ { i } ( t )$ representing its phase-angle state associated with rotor-angle dynamics.

The network topology is constructed directly from the physical transmission-line connectivity of the system. Specifically, two buses � and � are considered connected if they are directly linked by a transmission line. The resulting adjacency matrix $\pmb { A } _ { i j }$ therefore represents the physical coupling topology among interconnected buses, where $\pmb { A } _ { i j } = 1$ indicates a direct transmission-line connection between buses � and � , and $\pmb { A } _ { i j } = 0$ otherwise. This topology is jointly used with the bus-state trajectories to identify topology-dependent phase-coupling mechanisms in the power-grid dynamics.

The phase-angle trajectories $x _ { i } ( t )$ are used as dynamical observations for governing-equation discovery, while the transmission-line adjacency matrix provides the corresponding network-coupling structure. SAGE jointly uses these temporal trajectories and the physical network topology to identify shared and nodespecific phase-coupling mechanisms across the power-grid network. Detailed governing-equation inference and dynamical interpretation results are presented below. Detailed governing-equation inference and dynamical interpretation results are presented below.

## 01 DISCOVERED GOVERNING EQUATIONS

(t: time, x: state variable)

$$
{ \frac { d x } { d t } } = K \sum _ { j } A _ { i j } \sin ( x _ { j } - x _ { i } )
$$

## 02 KEY DYNAMICAL STRUCTURE

<table><tr><td rowspan=1 colspan=1>Discovered term type</td><td rowspan=1 colspan=1>Representative terms (in equations)</td><td rowspan=1 colspan=1>Interpretation</td></tr><tr><td rowspan=1 colspan=1>Topology-dependentphase coupling</td><td rowspan=1 colspan=1> $A _ { i j } \sin ( x _ { j } - x _ { i } )$ </td><td rowspan=1 colspan=1>Each bus is influenced by phasedifferences with its connected neighbors</td></tr></table>

03

## MECHANISTIC TAKEAWAY

Power grid dynamics are governed by topology-dependent phase coupling with sine interaction for coherent electromechanical evolution.

04

## HIERARCHICAL INTERPRETATION

## Community-level interpretation

Nodes within the power grid community share similar network-mediated phase coupling with coefficient variations.

## Individual-level interpretation

Individual modes exhibit node-specific dynamical components and coefficient variations that capture local deviations from the shared community structure.

05

## REVIEW EVIDENCE

One-step prediction R²

Bootstrap coefficient CV

x: 0.9999

Term recurrence frequency

dx: 0.001

1 / 1 key terms stable

06

## OVERALL ASSESSMENT

The discovered equation is compact and interpretable, capturing local bus dynamics and network connectivity effects. The topology-dependent sine coupling provides a coherent description of electromechanical evolution, supported by quantitative evidence of stability and predictive reliability.

Figure S4. Dynamical interpretation of the governing equations discovered for the power grid network. The inferred power-grid dynamics are characterized by topology-dependent phase coupling, in which the evolution of each node is governed by the phase differences between electrically connected neighbors through a sine-coupling term. This structure captures the network-mediated synchronization mechanism underlying electromechanical dynamics. At the community level, nodes share a common phase-coupling structure, whereas individual nodes exhibit coefficient variations that reflect local dynamical heterogeneity. The retained terms show high recurrence and predictive stability, supporting the robustness of the discovered mechanism.

## 3.4 Urban rail transit network

We use an urban rail transit passenger-flow dataset from Beijing, China. The dataset was collected from the Automatic Fare Collection (AFC) system between 05:00 and 23:00 over five consecutive weeks, from February 29 to April 3, 2016. The network contains 17 lines and 276 transit stations. Each station is treated as a dynamical node, and its passenger-flow state is represented by two variables. For station �, the state vector is defined as $\pmb { x } _ { i } ( t ) = [ x _ { i } ( t ) , y _ { i } ( t ) ] ^ { \mathrm { T } }$ , where $x _ { i } ( t )$ and $y _ { i } ( t )$ represent the entry and exit passenger flows, respectively. The passenger-flow data are aggregated at a 15-minute temporal resolution, providing sequential observations for governing equation discovery.

For network construction, the adjacency matrix $\pmb { A } _ { i j }$ is constructed based on the physical connectivity of the transit network, where two stations � and � are considered connected if they are directly linked by a rail segment. Accordingly, $\pmb { A } _ { i j } = 1$ indicates a direct physical connection between stations � and �, whereas $\pmb { A } _ { i j } = 0$ otherwise. The resulting topology characterizes the physical connectivity through which passenger flow dynamics may interact across neighboring stations and is jointly used with the station-level state trajectories for hierarchical governing-equation discovery. Detailed governing-equation inference and dynamical interpretation results are presented below.

![](images/5bf54e952f25ad21704541ce684e09bbdf2c06c368dcddd8c7daed93664434fe.jpg)  
Figure S5. Dynamical interpretation of the governing equations discovered for the urban rail transit network. The discovered passenger-flow dynamics combine short-term temporal memory, nonlinear state dependence, network coupling, and node-specific cross-state interactions. Lagged derivative terms capture temporal persistence in passenger-flow evolution, quadratic terms describe nonlinear self-regulation, and the topology-dependent coupling term represents the influence of neighboring stations. The hierarchical decomposition shows that shared temporal, nonlinear, and network mechanisms operate at the community level, while cross-state interactions and coefficient variations account for station-specific deviations. The

retained terms provide consistent temporal and network interaction evidence and remain physically interpretable.

## 4. Supplementary Validation Results for Real-World Complex Networks

Figure S6 provides detailed validation results for the four real-world complex networks, further comparing the true and inferred dynamics across different state variables and their corresponding community-level and individual-level components.

![](images/4b06ae9bac735873e8a6456d81de0f43d1711f3da642710d8f71af33817644dd.jpg)  
Figure S6. Detailed validation of SAGE on four real-world complex networks. True and inferred dynamics are compared across different state variables, together with the corresponding community-level dynamics $S _ { k } ( x )$ , individual-level dynamics $H _ { i } ( x )$ , and reconstructed node states. a) Urban mobile communication network. b) Brain EEG network. c) Power grid network. d) Urban rail transit network.