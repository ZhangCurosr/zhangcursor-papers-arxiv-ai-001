# AdCoEvo: A Co-Evolving Multi-Agent Advertising Delivery System

Tianzhu Hu<sup>1,2,∗,‡</sup>, Hang Yu<sup>3,∗,‡</sup>, Zeyuan Zhao<sup>4</sup>, Zeyu Pan<sup>4</sup>, Huan Luo<sup>4</sup>, Chun Gan<sup>4,†</sup>, Mingzhe Cao<sup>4</sup>, Liubin Wang<sup>4</sup>, Mingming Li<sup>1,†</sup>, Jiao Dai<sup>1</sup>, Jizhong Han<sup>1</sup>, Songlin Hu<sup>1,2</sup>

<sup>1</sup>Institute of Information Engineering, Chinese Academy of Sciences <sup>2</sup>School of Cyberspace Security, University of Chinese Academy of Sciences <sup>3</sup>The University of Sydney <sup>4</sup>JD.COM

hutianzhu@iie.ac.cn, hayu0853@uni.sydney.edu.au {zhaozeyuan.6, panzeyu3, luohuan.12, ganchun1, caomingzhe1, wangliubin.ubi}@jd.com {limingming, daijiao, hanjizhong, husonglin}@iie.ac.cn

<sup>∗</sup>Equal contribution, <sup>†</sup>Corresponding authors, <sup>‡</sup>Work was done during internship.

## Abstract

Digital advertising is an important means for platforms to reach potential users and drive business growth. Advertising delivery is the core link between marketing objectives and actual returns. Existing methods have achieved partial automation in individual tasks such as bidding and budget allocation, but they still rely on human operators for overall coordination, making it dificult to support large-scale operations. They also often fail to achieve optimal performance, and individual experience remains dificult to turn into reusable operational capabilities. In addition, as advertising markets change, advertising delivery systems need timely strategy updates to remain efective in dynamic markets. To automate the entire advertising operations process and improve adaptability to market changes, we propose AdCoEvo, a co-evolving multi-agent advertising delivery system. Specifically, the system connects delivery decisions across diferent timescales and operational granularities through role specialization. Diagnostic results guide account planning and strategy adjustment in the next period. The multi-agent system updates its model parameters based on actual delivery outcomes and continually improves delivery strategies through mutual feedback among planning, adjustment, and diagnosis, thereby achieving co-evolution. Experiments show that, relative to the rule-based operational baseline (Rule), AdCoEvo increases advertiser value by 14.02% while reducing cost per acquisition by 7.07%. Compared with a multi-agent system with fixed parameters, AdCoEvo increases advertiser value by 3.96% and reduces cost per acquisition by 1.24%, indicating that co-evolution can yield greater delivery returns.

Code: https://github.com/jd-opensource/AdCoEvo

## 1 Introduction

Digital advertising is an important channel for platforms to reach potential users and drive business growth. Turning marketing objectives into returns requires continuous high-quality advertising delivery decisions under budget constraints [1]. As shown on the left of Figure 1, existing delivery workflows still rely heavily on manual configuration and experience-based judgment by operators [2]. As operations grow, limited manpower and responsiveness make this approach dificult to scale and often prevent optimal performance [2]. Operators accumulated experience is also hard to turn into reusable operational capabilities for later delivery.

![](images/77a05e8ea1d86b1199092103a75bcf723f8df6d15af0ba12617655fd635e4911.jpg)  
Figure 1 From manual to multi-agent advertising delivery.

Prior work has automated individual tasks, such as bidding [1, 3] and budget allocation [4–6], but overall advertising operations still rely on human coordination. More recently, large language model agents have begun to take on decisions across the entire advertising delivery process. AdPilot uses a Reflector–Actor workflow: it first generates a diagnosis from current metrics and historical trends, then makes delivery decisions based on that diagnosis, and improves agent capabilities through a series of post-training steps [2].

Training policies for the entire delivery process requires linking business feedback to specific decisions. Budget planning and ad-unit adjustments continue to afect subsequent delivery, and their efects need to be evaluated through cumulative returns; post-delivery diagnosis takes efect through planning and adjustment in the next period. Organizing multi-agent collaboration by operational responsibility therefore also requires constructing training signals according to each decision’s scope of influence and feedback timing, and scheduling role updates after feedback settlement, enabling models to co-evolve during continuous delivery.

Meanwhile, the advertising delivery market is not static: a continually changing auction environment may create a mismatch between historical strategies and current conditions, requiring timely updates to delivery strategies [3, 7]. Beyond autonomous execution, we therefore need to study how multi-agent systems use continuous business feedback to adjust strategies and jointly adapt to market changes through collaboration.

To automate the entire advertising delivery process and improve adaptability to market changes, we propose AdCoEvo, a co-evolving multi-agent advertising delivery system. As shown on the right of Figure 1, the system connects decisions across diferent timescales and operational granularities by assigning separate responsibilities for account-level budget planning, ad-unit delivery strategy adjustment, and end-of-period diagnosis. Diagnostic results guide account planning and strategy adjustment in the next period. After ofline training, the system updates one role per period in rotation using settled feedback, while keeping other roles’ parameters fixed. The updated policy takes efect in the next period. Through joint execution, it changes the delivery states and training feedback encountered by other roles, which adapt their policies in later rotations. We call this mutual adaptation across periods co-evolution. Experiments show that AdCoEvo improves advertiser value (ADV) and reduces cost per acquisition (CPA) in both environments.<sup>1</sup> Compared with a multi-agent system with fixed parameters, co-evolution can yield greater delivery returns.

## Our main contributions are as follows:

1. We propose AdCoEvo, a co-evolving multi-agent advertising delivery system that connects account budget planning, ad-unit delivery strategy adjustment, and post-delivery diagnosis through role specialization across diferent timescales and operational granularities, enabling autonomous delivery across periods.

2. We develop a co-evolution framework for continuous advertising delivery in which the multi-agent system updates policies using settled business feedback and jointly adapts to changes in delivery as its decisions influence one another.

3. We validate the system’s delivery performance across diferent market sequences and compare role compositions, training stages, and role-update schemes to analyze gains from role collaboration, operational capability development, and co-evolution.

## 2 Related Work

## 2.1 Advertising Delivery

Research on advertising delivery has largely focused on automating individual tasks. Automated bidding has progressed from optimizing bid functions under budget constraints [1] to using reinforcement learning [10, 11] or generative modeling [12] for sequential decision-making. RTBAgent incorporates expert knowledge and reflection on past experience into bidding decisions [13]. AdComB jointly optimizes budget allocation and bidding [5], while HiBid uses a hierarchical mechanism to connect cross-channel budgets with within-channel bidding [4]. AuctionNet evaluates automated bidding through trafic generation and auction interactions [14].

AdPilot supports end-to-end autonomous advertising delivery through a Reflector–Actor workflow informed by current states and historical memory. Post-training further improves its delivery decisions [2]. AdCoEvo organizes operational responsibilities into roles with separate policies. Period-end diagnosis guides account planning and strategy adjustment in the next period, while settled feedback supports role-policy updates during continuous delivery.

## 2.2 Multi-Agent Collaboration

The core advantage of multi-agent systems is decomposing complex processes into roles with clear responsibilities. Roles address distinct objectives and contexts, then make collective decisions through interaction. Studies have applied complementary roles and task decomposition to complex problem solving and workflow automation, supporting multi-step tasks that are dificult for a single agent to complete independently [15–19].

Current research on multi-agent systems organizes role collaboration through standardized workflows, dynamic task allocation, and runtime supervision [17, 20–23]. It also trains models using joint execution trajectories, role evaluations, or interaction rewards to improve collaborative decision-making capabilities [24, 25]. Our work focuses on the collaboration and continual evolution of multi-agent systems in advertising delivery tasks.

## 2.3 Agentic RL

Agentic RL uses feedback from environmental interactions to optimize multi-step decision-making and tool-use policies [26–28]. For example, ReTool uses tool-call outcomes to improve when and how tools are called [29], while AgentFlow uses multi-turn trajectories to improve planning policies [30]. MARTI uses multi-agent interaction trajectories for distributed training updates [31].

We continually update role policies through alternating reinforcement learning, focusing on mutual adaptation among roles during continuous delivery. Training signals reflect each role’s scope of influence and feedback timing. Cross-period diagnosis and subsequent joint execution allow each role’s policy updates to shape feedback for other roles’ subsequent decisions and learning.

## 3 Problem Formulation

We formulate continuous advertising delivery as sequential decision-making, with objectives for value generation and conversion costs under budget and adjustment-count constraints.

## 3.1 Modeling Delivery Decisions

During delivery, the system plans ad-unit budgets within each account and adjusts control tCPA based on the current state. The fixed underlying bidder uses these settings to bid for individual trafic opportunities. At the end of the period, the system generates diagnostic guidance from settled outcomes to inform account planning and strategy adjustment in the next period.

Index decisions chronologically under system policy π. Decision i uses currently observable information:

$$
u _ { i } \sim \pi ( \cdot \mid h _ { i } ) .\tag{1}
$$

• Decision information $h _ { i }$ consists of the current decision type and its observable delivery state, historical execution records, and settled feedback. Historical diagnostic guidance forms part of this information and informs subsequent account planning and strategy adjustment.

• Decision output $u _ { i } \colon$ an account’s ad-unit budget allocations, control tCPA, and diagnostic guidance.

## 3.2 Business Objectives and Constraints

The system measures the business outcomes of delivery decisions using ADV and CPA, which reflect value generation and conversion costs, respectively. Let A denote the set of controlled accounts and $\mathcal { U } _ { a }$ the set of ad units in account a. The total ADV across all controlled accounts in period e is

$$
F _ { e } = \sum _ { a \in \mathcal { A } } \sum _ { j \in \mathcal { U } _ { a } } d _ { a , j } q _ { e , a , j } ,\tag{2}
$$

Here, $d _ { a , j }$ is the environment-provided initial tCPA of the ad unit, and $q _ { e , a , j }$ is the sum of predicted conversion probabilities for its valid impressions within the period. ADV is always calculated using the initial tCPA. Control tCPA changes afect ADV through delivery outcomes, while its valuation basis stays fixed.

Let $\bar { F } _ { e } = F _ { e } / | \mathcal { A } |$ denote average ADV per account, and $C _ { e } , N _ { e }$ denote total spend and conversions across controlled accounts in period e. Aggregate CPA is $C _ { e } / N _ { e }$ . We optimize these two objectives subject to budget and adjustment-count constraints:

$$
\begin{array} { r l r } { \underset { \pi \in \Pi } { \mathrm { m a x i m i z e } } } & { \left( \mathbb { E } _ { \pi } [ \bar { F } _ { e } ] , - \frac { \mathbb { E } _ { \pi } [ C _ { e } ] } { \mathbb { E } _ { \pi } [ N _ { e } ] } \right) , } \\ { \mathrm { s u b j e c t ~ t o } } & { \displaystyle \sum _ { j \in \mathcal { U } _ { a } } b _ { e , a , j } = B _ { e , a } , } & { \forall e , a , } \\ & { b _ { e , a , j } \geq 0 , } & { v _ { e , a , j } \leq K , \forall e , a , j . } \end{array}\tag{3}
$$

• Policy set Π: policies use only information observable at decision time; CPA requires $\mathbb { E } _ { \pi } [ N _ { e } ] > 0$

• Budget constraint: $B _ { e , a }$ is the budget of account a in period $e ,$ and $b _ { e , a , j }$ is the total budget limit of ad unit j. Reallocation among ad units preserves the account’s total budget.

• Adjustment-count constraint: $v _ { e , a , j }$ counts valid tCPA adjustments executed for ad unit j in period $e ; K$ is the maximum count per ad unit per period.

## 4 Method

AdCoEvo organizes advertising delivery through role collaboration. As shown in the upper part of Figure 2, Planner handles account budget planning, Optimizer adjusts ad-unit delivery strategies, and Inspector turns settled period outcomes into diagnostic guidance for the next period. Together, planning, adjustment, and diagnosis form a collaborative process across periods.

The system adopts two-stage training. The first stage develops operational decision-making capabilities through supervised fine-tuning and ofline reinforcement learning. The second stage, shown in the lower part of Figure 2, updates role parameters using settled business feedback during continuous delivery, enabling co-evolution. The ofline reinforcement learning and co-evolution algorithms are given in Appendices B.2 and C.3, respectively.

![](images/d461fdfd08a19ded722e1024f70943283470f8cd4d328d92ea9969fbe0bd94b5.jpg)  
Figure 2 Role collaboration and continuous co-evolution in AdCoEvo. The upper panel shows delivery execution, and the lower panel shows role-policy updates.

## 4.1 Role Specialization and Collaborative Execution

Account budget planning, ad-unit delivery strategy adjustment, and end-of-period diagnosis difer in their scope of influence and feedback timing. AdCoEvo accordingly assigns these responsibilities to Planner, Optimizer, and Inspector and maintains a separate policy $\pi _ { \theta _ { r } }$ for each role, where $r \in \{ P , O , I \}$ and $\theta _ { r }$ denotes the role’s model parameters. This organization associates execution records for diferent responsibilities with their corresponding training feedback. Planner and Optimizer make decisions for one account and its ad units at a time, whereas Inspector aggregates period outcomes across all controlled accounts.

After observing W time steps, Planner uses observed delivery performance to allocate the account’s remaining budget, outputting a proportion $\alpha _ { j }$ for each ad unit. Unit j then has total budget limit

$$
b _ { j } = c _ { j } ^ { \mathrm { p r e } } + R \alpha _ { j } ,\tag{4}
$$

Here, $c _ { j } ^ { \mathrm { p r e } }$ is the cumulative spend of ad unit $j$ before planning, $\begin{array} { r } { R = B - \sum _ { j \in \mathcal { U } } c _ { j } ^ { \mathrm { p r e } } } \end{array}$ is the account’s remaining budget, and B and U denote the account’s total budget and set of ad units, respectively. The allocation proportions satisfy $\alpha _ { j } \geq 0$ and $\textstyle \sum _ { j \in { \mathcal { U } } } \alpha _ { j } = 1$

After budget planning, Optimizer determines at every remaining time step whether each ad unit in the account needs a delivery strategy adjustment. Ad-unit actions include HOLD and a multiplier adjustment relative to the initial tCPA. If an adjustment is selected and the unit’s number of valid adjustments in the current period has not reached K, Optimizer selects $\kappa _ { j }$ from the available multiplier set K and updates the control tCPA to

$$
\widehat { d } _ { j } = \kappa _ { j } d _ { j } ,\tag{5}
$$

Here, $d _ { j }$ is the environment-provided initial tCPA. The current control value is retained for HOLD or when valid adjustments reach K.

After period settlement, Inspector aggregates delivery performance, budget utilization, and action records across the controlled accounts to generate diagnostic guidance $g _ { e }$ as part of the decision inputs of Planner and Optimizer in the next period.

## 4.2 Role Reward Construction

Training feedback for Planner and Optimizer accounts for both value generation and cost control. For ADV, spend, and conversions $F , C , N$ within the corresponding aggregation scope, we define the proxy return

$$
Q ( F , C , N ) = \frac { F - \eta ( C - \rho N ) } { s } ,\tag{6}
$$

Here, $\rho$ is the mean initial tCPA of ad units in the settlement window, weighted by conversion counts; η weights the cost deviation $C - \rho N$ ; and s scales the reward.

Role rewards reflect the scope of influence and feedback timing of each decision. Planner coordinates the account’s period budget and evaluates resource allocation using outcomes from observation end to period end. Optimizer continually adjusts ad-unit delivery strategies, computes stepwise returns from value generation and the cost deviation, and accumulates subsequent returns during training to account for the lasting efects of its actions. Both rewards combine the current account’s return with a weighted overall return. Inspector provides diagnostic guidance to Planner and Optimizer in the next period. Its reward is computed from changes in overall feedback between adjacent periods once the next period’s delivery outcomes are settled:

$$
r _ { a } ^ { P } = Q _ { a } + \omega Q _ { \mathrm { a l l } } , \qquad r _ { a , t } ^ { O } = Q _ { a , t } + \omega Q _ { \mathrm { a l l } , t } , \qquad r _ { e - 1 } ^ { I } = \mathbf { w } ^ { \top } ( \mathbf { y } _ { e } - \mathbf { y } _ { e - 1 } ) .\tag{7}
$$

Here, $Q _ { a } , Q _ { \mathrm { a l l } }$ aggregate outcomes from observation end to period end for the current account and all controlled accounts, respectively; time-indexed terms use current-step increments. The parameter ω weights the overall reward. The period feedback vector $\mathbf { y } _ { e }$ combines standardized ADV, conversions, the CPA constraint margin, and the forced-HOLD rate across all controlled accounts, with fixed weights w. Unsettled diagnoses are excluded from training. Calculations appear in Appendices A.1–A.2.

## 4.3 Offline Development of Operational Capabilities

We first apply supervised fine-tuning [32, 33] on role-specific demonstration trajectories to learn action semantics and structured output specifications. Ofline reinforcement learning then uses delivery outcomes as training feedback aligned with each role’s responsibilities to improve operational decisions further.

To compare diferent decisions from the same delivery state, the system fixes the other roles’ policies, samples G candidate responses for the target role, and executes them separately, with $J _ { g } ^ { r }$ denoting the return of candidate $g .$ . Planner and Optimizer construct leave-one-out advantages [34]; Inspector evaluates the delivery improvement from guidance against the baseline return $J _ { 0 }$ of a branch in which neither Planner nor Optimizer receives diagnostic guidance. The candidate advantages are

$$
A _ { g } ^ { r } = \left\{ \begin{array} { l l } { { J _ { g } ^ { r } - \displaystyle \frac { 1 } { G - 1 } \sum _ { k \neq g } J _ { k } ^ { r } , } } & { { r \in \{ P , O \} } , } \\ { { J _ { g } ^ { I } - J _ { 0 } , } } & { { r = I . } } \end{array} \right.\tag{8}
$$

Planner and Optimizer advantages are normalized by a scale, while Inspector retains the return diference relative to the baseline without diagnostic guidance. We denote them by $\widehat { A } _ { g } ^ { r } .$ . Using the sequence-level clipping approach of GSPO [35], we apply decision advantages to policy optimization over entire responses. For role r with valid candidates $\mathcal { T } _ { r }$ , we optimize

$$
\mathcal { L } _ { \mathrm { o f f } } ^ { r } ( \theta _ { r } ) = - \frac { 1 } { \vert \mathscr { L } _ { r } \vert } \sum _ { g \in \mathscr { Z } _ { r } } \operatorname* { m i n } \Bigl [ \varrho _ { g } \widehat { A } _ { g } ^ { r } , \mathrm { c l i p } \bigl ( \varrho _ { g } , 1 - \varepsilon ^ { - } , 1 + \varepsilon ^ { + } \bigr ) \widehat { A } _ { g } ^ { r } \Bigr ] + \beta \widehat { D } _ { \mathrm { K L } } .\tag{9}
$$

Here, $\varrho _ { g }$ is the length-normalized sequence importance ratio of the current policy to the sampling policy, and $\widehat { D } _ { \mathrm { K I } }$ measures the current policy’s deviation from a fixed reference policy. The parameters $\varepsilon ^ { - } , \varepsilon ^ { + }$ and $\beta$ control the clipping range and KL penalty strength, respectively. The calculations for advantage processing, the sequence ratio, and the KL term are detailed in Appendix B.1.

## 4.4 Co-Evolution during Continuous Delivery

Ofline training compares candidate decisions while keeping the other roles’ policies fixed. During continuous delivery, updated roles participate in joint execution again, changing subsequent delivery states and business feedback, to which other roles must also adapt their decisions. AdCoEvo draws on SAO’s single-trajectory asynchronous update idea [36] to construct training signals from settled execution trajectories.

At period end, the system settles current $\mathrm { P / O }$ rewards and constructs period feedback $\mathbf { y } _ { e }$ . Inspector generates diagnostic guidance $g _ { e }$ for the next period from the delivery outcomes. The system then settles the reward for the previous diagnosis $g _ { e - 1 }$ , adds settled records to role bufers, and updates one role in rotation $( P , O , I )$ Other role parameters remain fixed, and the new policy takes efect in the next period.

Value Estimation and Policy Optimization. During continuous delivery, the system executes one response at each decision point. Each role therefore maintains a separate lightweight value model, PrefixValueModel $V _ { \phi _ { r } } ( h )$ , which predicts a scalar return from the input before response generation. The model is continually trained on settled feedback to provide a baseline for policy updates.

Let $V _ { i } ^ { - }$ denote the value prediction for node i before the current round of updates. Planner and Inspector subtract this baseline from their settled rewards. Optimizer combines current-step rewards and subsequent value estimates through GAE [37]:

$$
\widehat { A } _ { i } ^ { r } = \left\{ \begin{array} { l l } { r _ { i } ^ { r } - V _ { i } ^ { - } , } & { r \in \{ P , I \} , } \\ { \mathrm { G A E } _ { i } ( \{ r _ { k } ^ { O } , V _ { k } ^ { - } \} ) , } & { r = O . } \end{array} \right.\tag{10}
$$

Optimizer successor nodes follow decision time within the same account and period.

Advantages remain fixed during the current round of policy optimization. The value model is trained independently by regression, without propagating its gradients into the role language model. The entire response corresponding to a decision shares the same action advantage. For a response z of length $L ,$ input $h ,$ and action advantage $\stackrel { \sim } { A }$ we optimize

$$
\mathcal { L } _ { \mathrm { c o - e v o } } ( \theta _ { r } ) = - \frac { \widehat { A } } { Z } \sum _ { \ell = 1 } ^ { L } m _ { \ell } \bar { \varrho } _ { \ell } \log \pi _ { \theta _ { r } } ( z _ { \ell } \mid h , z _ { < \ell } ) ,\tag{11}
$$

Here, $m _ { \ell }$ is a token-selection mask based on the probability ratio between the current and behavior policies, $\bar { \varrho } _ { \ell }$ is a clipped importance weight with its gradient stopped, and $\begin{array} { r } { Z = \sum _ { \ell } m _ { \ell } > 0 } \end{array}$ is the number of valid tokens. KL is used only for monitoring and triggering rollback. Implementation details of advantage estimation, value regression, and update control are provided in Appendices C.1–C.2.

Through subsequent joint execution, updated policies change the delivery states and feedback encountered by other roles. These roles adjust their policies accordingly in later rotations, enabling the system to continually evolve through collaboration.

## 4.5 Inference Time Complexity Analysis

AdCoEvo’s per-period inference overhead depends on both the number of model calls and the cost of each call. Let n denote the number of decision points per period and |A| the number of controlled accounts. Planner is called once per account after observation, Optimizer is called once per account at each decision point after observation, and Inspector is called once globally at the end of the period. The number of model calls per period is therefore $O ( | A | n )$

Let L denote the combined input and output length per call, which is afected by the number of ad units per account and the length of historical records. For a fixed model size with KV caching, the computational cost of Transformer self-attention is $O ( L ^ { 2 } )$ per call, yielding the following per-period inference complexity

$$
\mathcal { C } _ { \mathrm { i n f e r } } = O \left( | A | n L ^ { 2 } \right) .\tag{12}
$$

## 5 Experiments

We compare AdCoEvo with operational baselines and foundation models in delivery performance and overhead across continuous market sequences. Ablations examine the roles of collaboration, operational capability development, and diagnostic guidance. Role-update comparisons and market-switch experiments further assess whether co-evolution continually improves delivery, while budget and parameter experiments test whether these gains persist under diferent settings.

## 5.1 Experimental Setup

Experimental environment: Experiments use the AuctionNet simulator [14] with its oficial period length and trafic volume. The environment contains 48 ad units that can purchase trafic. We control 12 units organized into 3 accounts, each containing 4 units, all using Abid as the fixed underlying bidder. We use two environments, AuctionNet (Sparse) and AuctionNet (General). The former uses the public simulation environment, where each controlled account’s initial budget is 1/12 of the total budget of all units. For AuctionNet (General), we calibrate the pValue generation scale, ad-unit budgets, and CPA constraints using the public General dataset, while keeping the other auction mechanisms unchanged. Detailed configurations are provided in Appendix D.

Evaluation metrics: Delivery performance is measured using ADV, CPA, and a composite score. For ADV, which uses tCPA-weighted predicted conversion values [8], we report the per-period average of total ADV across all controlled accounts within the evaluation scope. CPA is total spend divided by total conversions over the same evaluation scope [9]. The invalid output rate IR is the fraction of responses that violate the required format or action specifications, such as parsing failures, incorrect ad-unit decision counts, or tCPA multipliers outside the allowed set. Each invalid response is counted once. With No Operation under the same environment and budget as the reference, method m’s composite score is

$$
\mathrm { S c o r e } _ { m } = \frac { \mathrm { A D V } _ { m } } { \mathrm { A D V } _ { 0 } } \cdot \frac { \mathrm { C P A } _ { 0 } } { \mathrm { C P A } _ { m } } - \mathrm { I R } _ { m } .\tag{13}
$$

Here, subscript 0 denotes No Operation, whose score is 1. Invalid output rates are reported as percentages in tables and used as decimal fractions in the formula.

Comparison methods: We use No Operation, Rule, and AdPilot<sup>\*</sup> as advertising delivery baselines to compare delivery performance across diferent approaches to operational decision-making. No Operation makes no operational adjustments, while Rule adjusts budgets and tCPA using hand-crafted CPA-based rules. AdPilot<sup>\*</sup> follows AdPilot’s Reflector–Actor workflow [2], where a single agent generates actions informed by delivery diagnosis. AdCoEvo (Fixed) retains all roles and the diagnostic workflow, with model parameters fixed at the ofline GSPO checkpoint during evaluation. We also evaluate representative closed- and open-source foundation models [38– $- 4 1 \bar { ] ^ { 2 } }$ that directly perform advertising delivery tasks, comparing delivery performance, output validity, and operational overhead.

Implementation and evaluation: AdCoEvo uses Qwen3-8B [42] with separate LoRA parameters [43] for each role. AdPilot<sup>\*</sup> uses the same backbone and follows AdPilot’s post-training procedure [2], with SFT followed by GRPO. Its model parameters remain fixed during formal evaluation. We set the observation length to W = 12, with at most K = 8 tCPA adjustments per ad unit per period.

Each experimental group uses five independent market sequences of 50 consecutive delivery periods, matched across methods. These sequences were not used for SFT, ofline reinforcement learning, or selecting the default parameters. AdCoEvo independently starts each sequence from the same ofline GSPO checkpoint. AuctionNet regenerates market conditions, including trafic and predicted conversion probabilities, each period, while role policies and diagnostic guidance carry over within the sequence.

## 5.2 Overall Delivery Performance

We first examine whether AdCoEvo can achieve better delivery outcomes than fixed operational rules derived from human experience. As shown in Table 1, relative to Rule, AdCoEvo increases ADV by 14.02% and reduces CPA by 7.07% in the Sparse environment, while also improving both metrics in the General environment. Human-designed rules can trigger adjustments based on delivery states, but their decision logic is specified in advance. AdCoEvo instead uses delivery feedback for policy learning and continuous updates, allowing the models to accumulate experience throughout delivery.

The comparison with AdPilot<sup>\*</sup> further examines how end-to-end delivery decisions are organized. Both use the same backbone model. With both systems’ parameters fixed during evaluation, AdCoEvo (Fixed) achieves

Table 1 Delivery performance across market sequences. Overall aggregates results from five 50-period sequences.
<table><tr><td rowspan="2">Method</td><td>Sequence 1</td><td></td><td>Sequence 2</td><td>Sequence 3</td><td></td><td>Sequence 4</td><td>Sequence 5</td><td></td><td>Overall</td><td></td></tr><tr><td>ADV ↑</td><td>CPA↓</td><td>ADV ↑ CPA ↓</td><td>ADV ↑</td><td>CPA ↓</td><td>ADV ↑ CPA↓</td><td>ADV ↑</td><td>CPA ↓</td><td>ADV ↑ CPA ↓</td><td>Score ↑</td></tr><tr><td colspan="10">AuctionNet (Sparse)</td></tr><tr><td>No Operation</td><td>26,440.6 115.575</td><td></td><td>26,372.9 115.676</td><td>26,175.7 117.539</td><td></td><td>26,598.1 116.739</td><td>26,286.4 115.531</td><td></td><td>26,374.7 116.207</td><td>1.0000</td></tr><tr><td>Rule</td><td>28,680.3104.124</td><td></td><td>28,333.9 103.629</td><td>28,376.6 105.553</td><td></td><td>28,537.4 104.623</td><td>28,195.3 103.658</td><td></td><td>28,424.7104.315</td><td>1.2006</td></tr><tr><td>AdPilot*</td><td>28,090.3 118.479</td><td>27,012.3</td><td>118.847</td><td>27,103.3120.397</td><td></td><td>27,046.0119.133</td><td>26,876.5</td><td>118.358</td><td>27,225.7119.040</td><td>1.0077</td></tr><tr><td>AdCoEvo (Fixed)</td><td>31,720.6 98.368</td><td>30,709.7</td><td>98.583</td><td>30,615.4 100.112</td><td>31,675.0</td><td>99.104</td><td>31,154.9</td><td>97.600</td><td>31,175.1 98.150</td><td>1.3995</td></tr><tr><td>AdCoEvo</td><td>32,125.4</td><td>96.807 32,350.1</td><td></td><td>96.217 32,725.4 98.087</td><td>32,341.5</td><td></td><td>98.220 32,500.9</td><td>96.361</td><td>32,408.7 96.936</td><td>1.4731</td></tr><tr><td colspan="10">AuctionNet (General)</td></tr><tr><td>No Operation</td><td>27,612.0</td><td>8.864 28,094.9</td><td>8.685</td><td>27,782.8</td><td>8.875</td><td>28,306.5</td><td>8.738 28,203.9</td><td>8.677</td><td>28,000.0</td><td>8.767 1.0000</td></tr><tr><td>Rule</td><td>29,917.9</td><td>7.199 30,282.6</td><td>7.084</td><td>30,244.9</td><td>7.266</td><td>30,513.2 7.143</td><td>30,147.7</td><td>7.077</td><td>30,221.3</td><td>7.153 1.3229</td></tr><tr><td>AdPilot*</td><td>28,688.2</td><td>9.122 28,604.6</td><td>8.879</td><td>28,223.2</td><td>9.147</td><td>28,762.7 8.979</td><td>28,336.0</td><td>8.806</td><td>28,522.9</td><td>8.985 0.9940</td></tr><tr><td>AdCoEvo (Fixed)</td><td>31,965.4</td><td>7.112 31,375.8</td><td>7.025</td><td>30,914.1</td><td>7.139 32,294.7</td><td>7.258</td><td>31,541.2</td><td>7.203</td><td>31,618.2 7.147</td><td>1.3852</td></tr><tr><td>AdCoEvo</td><td>33,164.4</td><td>7.069 33,009.5</td><td>6.892</td><td>32,507.9</td><td>7.068 33,744.4</td><td></td><td>6.944 32,334.3</td><td>6.975</td><td>32,952.1</td><td>6.989 1.4763</td></tr></table>

Table 2 Delivery performance, API costs, and inference latency of AdCoEvo and foundation models in Sparse. Bold values indicate the best results.
<table><tr><td></td><td colspan="4">Business Performance</td><td colspan="2">Estimated API Cost</td><td colspan="3">Latency</td></tr><tr><td>Model</td><td>ADV↑</td><td>CPA↓</td><td>IR (%) ↓</td><td>Score↑</td><td>CNY/Period↓</td><td>CNY/10K ADV↓</td><td>P50(s) ↓</td><td>P95(s) ↓</td><td>Period(s) ↓</td></tr><tr><td colspan="10">Closed-Source</td></tr><tr><td>GPT-5.6-Terra</td><td>31,243.6</td><td>100.176</td><td>0.00</td><td>1.3742</td><td>5.733</td><td>1.8348</td><td>3.84</td><td>5.27</td><td>710.8</td></tr><tr><td>Claude-Sonnet-5</td><td>31,024.5</td><td>100.027</td><td>3.22</td><td>1.3344</td><td>8.152</td><td>2.6275</td><td>9.18</td><td>15.30</td><td>1,340.0</td></tr><tr><td>Gemini-3.1-Flash-Lite</td><td>30,291.8</td><td>101.393</td><td>8.16</td><td>1.2348</td><td>0.683</td><td>0.2255</td><td>1.40</td><td>2.23</td><td>470.8</td></tr><tr><td colspan="10">Open-Source</td></tr><tr><td>DeepSeek-V4-Pro</td><td>30,890.9</td><td>100.147</td><td>0.00</td><td>1.3591</td><td>0.830</td><td>0.2688</td><td>3.96</td><td>6.62</td><td>724.4</td></tr><tr><td>Kimi-K3</td><td>29,055.5</td><td>102.191</td><td>8.98</td><td>1.1630</td><td>13.441</td><td>4.6258</td><td>15.33</td><td>82.97</td><td>3,346.1</td></tr><tr><td>GLM-5.2</td><td>28,056.8</td><td>109.419</td><td>8.74</td><td>1.0423</td><td>1.427</td><td>0.5084</td><td>17.86</td><td>34.19</td><td>2,170.3</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>27,152.6</td><td>117.737</td><td>11.58</td><td>0.9003</td><td>2.334</td><td>0.8597</td><td>14.49</td><td>26.98</td><td>1,709.7</td></tr><tr><td colspan="10">AdCoEvo</td></tr><tr><td>AdCoEvo</td><td>32,408.7</td><td>96.936</td><td>0.00</td><td>1.4731</td><td>0.235</td><td>0.0725</td><td>1.20</td><td>2.19</td><td>397.1</td></tr></table>

14.51% higher ADV and 17.55% lower CPA in Sparse. Compared with a single agent that handles delivery decisions, AdCoEvo assigns operational responsibilities to roles that learn separately and provide feedback to one another, enabling them to collaborate and co-evolve toward a shared business objective.

Across all five market sequences in both environments, AdCoEvo achieves the highest ADV and lowest CPA among the compared methods. These results indicate its adaptability to changing markets.

## 5.3 Comparison with Foundation Models

We compare with foundation models to assess advertising-specific operational capabilities. As shown in Table 2, AdCoEvo achieves the highest ADV, lowest CPA, and highest composite score. Compared with GPT-5.6-Terra, the best-performing foundation model overall, it increases ADV by 3.73% and reduces CPA by 3.23%. With a smaller model, domain-specific training and co-evolution can build operational capabilities aligned with delivery objectives, improving value generation and cost eficiency.

In addition to delivery performance, AdCoEvo also has lower API costs. Based on actual API usage across all roles and the corresponding listed prices,<sup>3</sup> its average API cost is CNY 0.235 per period, 65.59% lower than that of Gemini-3.1-Flash-Lite, the least expensive foundation model, and its cost per 10,000 ADV is CNY 0.0725. Lower API costs help scale advertising delivery.

Qwen3.5-397B-A17B, a larger model from the same Qwen family, does not achieve better delivery results, further indicating that operational capability cannot be assessed by model size alone. Organizing decisions and learning around the advertising delivery task provides another efective way to balance business performance and operational cost, beyond simply increasing the size of the backbone model.

Table 3 ADV and CPA of diferent role combinations in Sparse without domain-specific post-training.
<table><tr><td>Method</td><td>ADV↑</td><td>CPA ↓</td></tr><tr><td>No Operation</td><td>26,374.7</td><td>116.207</td></tr><tr><td>Planner</td><td>26,682.1</td><td>125.751</td></tr><tr><td>Optimizer</td><td>25,004.8</td><td>113.410</td></tr><tr><td>P+O</td><td>26,243.8</td><td>121.240</td></tr><tr><td>P+O+I</td><td>27,543.2</td><td>109.641</td></tr></table>

Table 4 Training-stage ablation in Sparse.
<table><tr><td>Method</td><td>ADV ↑</td><td>CPA↓</td></tr><tr><td>No Operation</td><td>26,374.7</td><td>116.207</td></tr><tr><td>w/o Training</td><td>27,543.2</td><td>109.641</td></tr><tr><td>SFT</td><td>30,892.4</td><td>100.016</td></tr><tr><td>GSPO</td><td>31,175.1</td><td>98.150</td></tr><tr><td>AdCoEvo</td><td>32,408.7</td><td>96.936</td></tr></table>

## 5.4 Ablation Analysis

## 5.4.1 Role Composition Ablation

To examine collaboration among specialized roles, we construct combinations of Planner, Optimizer, and Inspector without domain-specific post-training. As shown in Table 3, using Planner alone increases ADV but also raises CPA; using Optimizer alone reduces CPA but decreases ADV. Combining the two still does not improve both metrics simultaneously. Adding Inspector increases ADV by 4.95% and reduces CPA by 9.57% relative to the Planner–Optimizer combination. The full system outperforms No Operation on both metrics.

This change indicates that local improvements from individual roles do not simply add up to overall gains when more agents are introduced. Budget planning and ad-unit adjustment influence one another and require collaboration around a shared delivery objective. Inspector turns execution outcomes into diagnostic guidance for Planner and Optimizer, enabling both roles to revise subsequent decisions using prior delivery feedback.

## 5.4.2 Policy Learning Stage Ablation

With all roles present, we compare four settings: before domain-specific post-training (w/o Training), SFT alone, subsequent GSPO ofline reinforcement learning, and co-evolution. As shown in Table 4, SFT increases ADV and reduces CPA; subsequent GSPO further improves both metrics. Business performance improves with training even when the division of roles remains unchanged, indicating that the collaborative structure alone does not guarantee high-quality decisions; the corresponding operational capabilities also need to be developed. Supervised demonstrations provide initial decision-making capabilities, while reinforcement learning further optimizes delivery strategies using execution outcomes, making multi-agent collaboration more efective in supporting value generation and cost control.

Compared with AdCoEvo (Fixed), AdCoEvo increases ADV by 3.96% and reduces CPA by 1.24% in Sparse. In General, ADV increases by 4.22% and CPA decreases by 2.21%. This improvement shows that the operational capabilities established through ofline training still have room to improve. In a changing advertising market, co-evolution enables the multi-agent system to continually improve its policies using new settled feedback, yielding greater delivery returns on top of its existing collaboration and operational capabilities.

## 5.4.3 Co-Evolution Ablation

To study role participation and rotation, we retain all roles and compare Optimizer-only updates, Planner– Optimizer updates, and AdCoEvo against AdCoEvo (Fixed). To equalize role-update counts, the first setting updates Optimizer every period; the second alternates Planner and Optimizer across periods.

As shown in Table 5, full co-evolution achieves higher ADV and composite scores than updating Planner and Optimizer across all five market sequences. Overall, ADV further increases by 2.88% and the composite score by 2.78%, while CPA increases by only approximately 0.10%. This shows that, while retaining diagnostic guidance, allowing Inspector to participate in co-evolution can further increase advertiser value at a similar cost per acquisition, yielding higher delivery returns.

Table 5 ADV and Score gains (%) over AdCoEvo (Fixed) under diferent role-update settings in Sparse. All settings retain the complete role composition.
<table><tr><td></td><td colspan="2">O gains</td><td colspan="2">P+O gains</td><td colspan="2">AdCoEvo gains</td></tr><tr><td>Seq.</td><td>ADV↑</td><td>Score↑</td><td>ADV↑</td><td>Score ↑</td><td>ADV↑</td><td>Score↑</td></tr><tr><td>1</td><td>0.62</td><td>2.25</td><td>0.93</td><td>2.56</td><td>1.28</td><td>2.91</td></tr><tr><td>2</td><td>0.74</td><td>3.30</td><td>1.03</td><td>3.58</td><td>5.34</td><td>7.93</td></tr><tr><td>3</td><td>0.64</td><td>2.84</td><td>1.01</td><td>3.11</td><td>6.89</td><td>9.10</td></tr><tr><td>4</td><td>0.56</td><td>1.67</td><td>0.94</td><td>1.98</td><td>2.10</td><td>3.02</td></tr><tr><td>5</td><td>0.59</td><td>2.11</td><td>1.33</td><td>2.17</td><td>4.32</td><td>5.66</td></tr><tr><td>Overall</td><td>0.63</td><td>2.08</td><td>1.05</td><td>2.41</td><td>3.96</td><td>5.26</td></tr></table>

(a) ADV uplift (%)  
![](images/faeb5726cd1ae39fa887d460aaea0e8ac5264ad406c279e0300a0c5495633daf.jpg)

(b) ∆CPA (%)  
![](images/cac000c3baa291c782584c207c39e3b1b0a089de53770ebbf67c437515eaf914.jpg)  
Figure 3 ADV uplift and CPA change relative to AdCoEvo (Fixed) under market switches, averaged across five market sequences using five-period windows.

To examine how co-evolution adapts to market changes, we switch from General to Sparse at period 16 and back to General at period 36, comparing AdCoEvo with AdCoEvo (Fixed). We report results averaged over five market sequences using five-period windows. As shown in Figure 3, AdCoEvo retains an ADV advantage after the market switches and gradually recovers its CPA improvement after the return to General, showing that co-evolution can improve adaptation to market changes.

## 5.4.4 Diagnostic Guidance Ablation

To examine the role of diagnostic guidance, we replace Inspector-generated diagnoses with fixed statistical summaries and compare delivery performance under the two feedback settings. Relative to fixed statistical summaries, Inspector-generated diagnoses increase ADV by 1.61% and reduce CPA by 1.55% in Sparse, with corresponding improvements of 1.64% and 1.34% in General. This shows that converting delivery outcomes into diagnostic guidance better supports subsequent account planning and strategy adjustment than providing statistical information alone.

## 5.5 Budget Constraints and Parameter Sensitivity

To examine whether AdCoEvo’s delivery gains depend on a particular resource setting, we compare business performance under diferent initial budgets. As shown in Figure 4, AdCoEvo achieves the highest ADV and lowest CPA at 60%, 80%, and 100% of the original budget. Even when the budget is reduced to 60%, it increases ADV by 7.17% and reduces CPA by 0.78% relative to AdCoEvo (Fixed). Tighter budgets require the system to coordinate account planning and ad-unit adjustment with more limited resources, yet AdCoEvo still improves both value generation and cost eficiency. Thus, co-evolution retains its gains over the fixed ofline policy at lower budgets.

In addition to budget, we examine the observation length W and the maximum number of tCPA adjustments per ad unit per period K. As shown in Table 6, the default setting of W = 12 and K = 8 achieves the highest ADV and lowest CPA among the configurations compared. Extending W from 6 to 12 improves both metrics, whereas extending it further to 18 reduces performance. A longer observation period provides more information before decisions are made but shortens the time during which planning and adjustment can afect subsequent delivery, requiring a balance between observation and execution.

![](images/a33ab721f77c4328fbfb02034cf0685ebd7ca25d7eb37dc8476fca77b334f180.jpg)  
Figure 4 ADV (solid bars, left axis) and CPA (hatched bars, right axis) under diferent initial budgets in Sparse.

Table 6 Efects of observation length W and adjustment limit K on delivery performance.
<table><tr><td>Parameter</td><td>ADV ↑</td><td>CPA ↓</td><td>Score ↑</td></tr><tr><td colspan="4">Observation length</td></tr><tr><td>W = 6</td><td>29,547.0</td><td>97.742</td><td>1.3319</td></tr><tr><td>W = 12 (default)</td><td>32,408.7</td><td>96.936</td><td>1.4731</td></tr><tr><td>W = 18</td><td>30,649.6</td><td>107.914</td><td>1.2514</td></tr><tr><td colspan="4">Adjustment limit</td></tr><tr><td>K = 4</td><td>31,992.6</td><td>97.240</td><td>1.4496</td></tr><tr><td>K = 8 (default)</td><td>32,408.7</td><td>96.936</td><td>1.4731</td></tr><tr><td>K = 12</td><td>31,268.6</td><td>99.510</td><td>1.3845</td></tr></table>

Reducing K from 8 to 4 decreases ADV by 1.28% and raises CPA by 0.31%, while performance remains close to the default configuration under the tighter adjustment constraint. Relaxing the limit to 12 lowers ADV and raises CPA, so more adjustment opportunities do not necessarily improve performance.

## 6 Discussion and Conclusion

We present AdCoEvo, a multi-agent system for advertising delivery that organizes delivery decisions through role specialization across diferent timescales and operational granularities, and improves delivery policies through ofline development of operational capabilities and co-evolution. Experiments show that operationallevel decisions, role collaboration, and post-training all help improve delivery performance, while co-evolution provides further gains beyond ofline reinforcement learning.

However, gains vary across environments, and the efectiveness of co-evolution still needs evaluation in specific delivery scenarios. How operational roles’ decision frequencies and scopes of influence afect co-evolution, and how advantage estimation and update mechanisms can be improved accordingly, warrant further study to help multi-agent systems use actual delivery feedback more efectively.

## References

[1] Weinan Zhang, Shuai Yuan, and Jun Wang. Optimal real-time bidding for display advertising. In Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 1077–1086. Association for Computing Machinery, 2014. doi: 10.1145/2623330.2623633. URL https://doi.org/10.1145/ 2623330.2623633.

[2] Shuoshuo Sun, Qi He, Xiaoting Li, et al. AdPilot: Towards fully autonomous advertising delivery via agentic reinforcement learning. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data

Mining V.2, pages 8021–8031, 2026. doi: 10.1145/3770855.3818315. URL https://doi.org/10.1145/3770855. 3818315.

[3] Zhiyu Mou et al. Sustainable online reinforcement learning for auto-bidding. In Advances in Neural Information Processing Systems, volume 35, pages 2651–2663. Curran Associates, Inc., 2022. doi: 10.52202/068431-0192. URL https://proceedings.neurips.cc/paper\_files/paper/2022/file/ 11faf17bf7e5412d9cded369f97db23d-Paper-Conference.pdf.

[4] Hao Wang et al. HiBid: A cross-channel constrained bidding system with budget allocation by hierarchical ofline deep reinforcement learning. IEEE Transactions on Computers, 73(3):815–828, 2024. doi: 10.1109/tc.2023.3343111. URL https://doi.org/10.1109/tc.2023.3343111.

[5] Alessandro Nuara, Francesco Trovò, Nicola Gatti, and Marcello Restelli. A combinatorial-bandit algorithm for the online joint bid/budget optimization of pay-per-click advertising campaigns. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1):2379–2386, 2018. doi: 10.1609/aaai.v32i1.11888. URL https://ojs.aaai.org/ index.php/AAAI/article/view/11888.

[6] Mingxuan Song, Yusen Huo, Bohan Zhou, Shenglin Yin, Zhen Xiao, Jieyi Long, Zhilin Zhang, and Chuan Yu. DARA: Few-shot budget allocation in online advertising via in-context decision making with RL-finetuned LLMs. In Proceedings of the ACM Web Conference 2026, pages 40–50. Association for Computing Machinery, 2026. doi: 10.1145/3774904.3792129. URL https://doi.org/10.1145/3774904.3792129.

[7] Lorenzo Croissant, Marc Abeille, and Clement Calauzenes. Real-time optimisation for online learning in auctions. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 2217–2226. PMLR, 2020. URL https://proceedings.mlr.press/v119/croissant20a. html.

[8] Ashwinkumar Badanidiyuru. Model monotonicity in autobidding auctions: When do better predictions lead to better outcomes? In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 4802–4819. PMLR, 2026. URL https://proceedings.mlr. press/v306/badanidiyuru26a.html.

[9] Google Ads. Average CPA: Definition. Google Ads Help, n.d. URL https://support.google.com/google-ads/ answer/6396841?hl=en. Accessed: 2026-10-11.

[10] Di Wu, Xiujun Chen, Xun Yang, Hao Wang, Qing Tan, Xiaoxun Zhang, Jian Xu, and Kun Gai. Budget constrained bidding by model-free reinforcement learning in display advertising. In Proceedings of the 27th ACM International Conference on Information and Knowledge Management, pages 1443–1451. Association for Computing Machinery, 2018. doi: 10.1145/3269206.3271748. URL https://doi.org/10.1145/3269206.3271748.

[11] Han Cai, Kan Ren, Weinan Zhang, Kleanthis Malialis, Jun Wang, Yong Yu, and Defeng Guo. Real-time bidding by reinforcement learning in display advertising. In Proceedings of the Tenth ACM International Conference on Web Search and Data Mining, pages 661–670. Association for Computing Machinery, 2017. doi: 10.1145/3018661.3018702. URL https://doi.org/10.1145/3018661.3018702.

[12] Jiayan Guo et al. Generative auto-bidding via conditional difusion modeling. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 5038–5049, 2024. doi: 10.1145/3637528. 3671526. URL https://doi.org/10.1145/3637528.3671526.

[13] Leng Cai et al. RTBAgent: A LLM-based agent system for real-time bidding. In Companion Proceedings of the ACM on Web Conference 2025, pages 104–113, 2025. doi: 10.1145/3701716.3715259. URL https: //doi.org/10.1145/3701716.3715259.

[14] Kefan Su, Yusen Huo, Zhilin Zhang, et al. AuctionNet: A novel benchmark for decision-making in large-scale games. In Advances in Neural Information Processing Systems, volume 37, pages 94428–94452. Curran Associates, Inc., 2024. doi: 10.52202/079017-2993. URL https://proceedings.neurips.cc/paper\_files/paper/2024/ file/ab9b7c23edfea0011507f7e1eae82cd2-Paper-Datasets\_and\_Benchmarks\_Track.pdf.

[15] Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, et al. CAMEL: Communicative agents for “mind” exploration of large language model society. In Advances in Neural Information Processing Systems, volume 36, pages 51991–52008, 2023. doi: 10.52202/075280-2264. URL https://proceedings.neurips.cc/paper\_files/ paper/2023/hash/a3621ee907def47c1b952ade25c67698-Abstract-Conference.html.

[16] Qingyun Wu, Gagan Bansal, Jieyu Zhang, et al. AutoGen: Enabling next-gen LLM applications via multi-agent conversations. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id= BAakY1hNKS.

[17] Sirui Hong, Mingchen Zhuge, Jonathan Chen, et al. MetaGPT: Meta programming for a multiagent collaborative framework. In The Twelfth International Conference on Learning Representations, pages 23247–23275, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/hash 6507b115562bb0a305f1958ccc87355a-Abstract-Conference.html.

[18] Chen Qian, Wei Liu, Hongzhang Liu, et al. ChatDev: Communicative agents for software development. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15174–15186, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.810. URL https://aclanthology.org/2024.acl-long.810/.

[19] Wei Tao, Yucheng Zhou, Yanlin Wang, Wenqiang Zhang, Hongyu Zhang, and Yu Cheng. MAGIS: LLM-based multi-agent framework for GitHub issue resolution. In Advances in Neural Information Processing Systems, volume 37, pages 51963–51993, 2024. doi: 10.52202/079017-1647. URL https://proceedings.neurips.cc/ paper\_files/paper/2024/hash/5d1f02132ef51602adf07000ca5b6138-Abstract-Conference.html.

[20] Boye Niu, Yiliao Song, Kai Lian, Yifan Shen, Yu Yao, Kun Zhang, and Tongliang Liu. Flow: Modularized Agentic Workflow Automation. In International Conference on Learning Representations, pages 74949–74977, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ ba84da6921f3040b74ee163aa7451f53-Abstract-Conference.html.

[21] Han Zhou, Xingchen Wan, Ruoxi Sun, Hamid Palangi, Shariq Iqbal, Ivan Vulić, Anna Korhonen, and Sercan Arik. Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies. In International Conference on Learning Representations, pages 15844–15872, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/ 2026/hash/1ab4e0e8f35078109bc78d7b465d306f-Abstract-Conference.html.

[22] Fulin Lin, Shaowen Chen, Ruishan Fang, Hongwei Wang, and Tao Lin. Stop Wasting Your Tokens: Towards Eficient Runtime Multi-Agent Systems. In International Conference on Learning Representations, pages 113497–113528, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/hash/ b86a195e70f27017c514fa0e5f80595f-Abstract-Conference.html.

[23] Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chi-Min Chan, Heyang Yu, Yaxi Lu, Yi-Hsin Hung, Chen Qian, Yujia Qin, Xin Cong, Ruobing Xie, Zhiyuan Liu, Maosong Sun, and Jie Zhou. AgentVerse: Facilitating multi-agent collaboration and exploring emergent behaviors. In International Conference on Learning Representations, pages 20094–20136, 2024. URL https://proceedings.iclr.cc/paper\_files/ paper/2024/hash/578e65cdee35d00c708d4c64bce32971-Abstract-Conference.html.

[24] Sumeet Ramesh Motwani, Chandler Smith, Rocktim Jyoti Das, et al. MALT: Improving reasoning with multi-agent LLM training. In Second Conference on Language Modeling, 2025. URL https://openreview.net/forum?id= jXP9bgFack.

[25] Xiangyuan Xue, Yifan Zhou, Guibin Zhang, Zaibin Zhang, Yijiang Li, Chen Zhang, Zhenfei Yin, Philip Torr, Wanli Ouyang, and Lei Bai. CoMAS: Co-Evolving Multi-Agent Systems via Interaction Rewards. In International Conference on Learning Representations, pages 32373–32394, 2026. URL https://proceedings.iclr.cc/paper\_ files/paper/2026/hash/372593bd318ad8b34b3a8da77e20272b-Abstract-Conference.html.

[26] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Ö. Arık, Dong Wang, Hamed Zamani, and Jiawei Han. Search-R1: Training LLMs to reason and leverage search engines with reinforcement learning. In Second Conference on Language Modeling, 2025. URL https://openreview.net/forum?id=Rwhi91ideu.

[27] Weihua Du, Hailei Gong, Zhan Ling, Kang Liu, Lingfeng Shen, Xuesong Yao, Yufei Xu, Dingyuan Shi, Yiming Yang, and Jiecao Chen. Generalizable End-to-End Tool-Use RL with Synthetic CodeGym. In International Conference on Learning Representations, pages 17733–17756, 2026. URL https://proceedings.iclr.cc/paper\_ files/paper/2026/hash/1e4322fddd833f83c855660ac65e428d-Abstract-Conference.html.

[28] Zehan Qi, Xiao Liu, Iat Long Iong, Hanyu Lai, Xueqiao Sun, Jiadai Sun, Xinyue Yang, Yu Yang, Shuntian Yao, Wei Xu, Jie Tang, and Yuxiao Dong. WebRL: Training LLM web agents via selfevolving online curriculum reinforcement learning. In International Conference on Learning Represen tations, pages 79791–79821, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ c66e1fcc9691aae706250638f36f681b-Abstract-Conference.html.

[29] Jiazhan Feng, Shijue Huang, Xingwei Qu, Ge Zhang, Yujia Qin, Baoquan Zhong, Chengquan Jiang, Jinxin Chi, and Wanjun Zhong. ReTool: Reinforcement Learning for Strategic Tool Use in LLMs. In International Conference on Learning Representations, pages 37909–37926, 2026. URL https://proceedings.iclr.cc/paper\_ files/paper/2026/hash/4038c9208dfc22644c60ad39c24e5c53-Abstract-Conference.html.

[30] Zhuofeng Li, Haoxiang Zhang, Seungju Han, Sheng Liu, Jianwen Xie, Yu Zhang, Yejin Choi, James Y Zou, and Pan Lu. In-the-Flow Agentic System Optimization for Efective Planning and Tool Use. In International Conference on Learning Representations, pages 50524–50570, 2026. URL https://proceedings.iclr.cc/paper\_ files/paper/2026/hash/52c4a736cd63b3c7bf52011a1f66f3f6-Abstract-Conference.html.

[31] Kaiyan Zhang, Kai Tian, Runze Liu, Sihang Zeng, Xuekai Zhu, Guoli Jia, Yuchen Fan, Xingtai Lv, Yuxin Zuo, Che Jiang, Yuru Wang, Jianyu Wang, Ermo Hua, Xinwei Long, Junqi Gao, Youbang Sun, Zhiyuan Ma, Ganqu Cui, Ning Ding, Biqing Qi, and Bowen Zhou. MARTI: A Framework for Multi-Agent LLM Systems Reinforced Training and Inference. In International Conference on Learning Repre sentations, pages 138790–138813, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/hash/ e045c55a71fb4fc04d9a20579415247d-Abstract-Conference.html.

[32] Long Ouyang, Jefrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pages 27730–27744. Curran Associates, Inc., 2022. doi: 10.52202/068431-2011. URL https://proceedings.neurips. cc/paper\_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html.

[33] Aohan Zeng, Mingdao Liu, Rui Lu, Bowen Wang, Xiao Liu, Yuxiao Dong, and Jie Tang. AgentTuning: Enabling generalized agent abilities for LLMs. In Findings of the Association for Computational Linguistics: ACL 2024, pages 3053–3077. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.findings-acl.181. URL https://aclanthology.org/2024.findings-acl.181/.

[34] Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting REINFORCE-style optimization for learning from human feedback in LLMs. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 12248–12267. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.acl-long.662. URL https://aclanthology.org/2024.acl-long.662/.

[35] Chujie Zheng et al. Group sequence policy optimization. arXiv preprint, 2025. URL https://arxiv.org/abs/ 2507.18071.

[36] Zhenyu Hou et al. Single-rollout asynchronous optimization for agentic reinforcement learning. arXiv preprint, 2026. URL https://arxiv.org/abs/2607.07508.

[37] John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. In International Conference on Learning Representations, 2016. URL https://arxiv.org/abs/1506.02438.

[38] DeepSeek-AI. DeepSeek-V4: Towards highly eficient million-token context intelligence. Technical report, 2026. URL https://arxiv.org/abs/2606.19348.

[39] Kimi Team. Kimi K3: Open frontier intelligence. arXiv preprint, 2026. URL https://arxiv.org/abs/2607.24653.

[40] Z.AI. GLM-5.2: Built for long-horizon tasks, June 2026. URL https://huggingface.co/blog/zai-org/ glm-52-blog.

[41] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen.ai/blog?id= qwen3.5.

[42] An Yang, Anfeng Li, Baosong Yang, et al. Qwen3 technical report. arXiv preprint, 2025. URL https: //arxiv.org/abs/2505.09388.

[43] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=nZeVKeeFYf9

## A Role Reward Computation

We detail aggregation scopes and parameters for ofline and co-evolution rewards (Section 4.2).

## A.1 Rewards for Planner and Optimizer

Equations (6)–(7) combine separate returns for the current account and all controlled accounts. Planner uses results from observation end to period end; Optimizer uses current-step increments. Each term aggregates ADV, spend, and conversions $F , C , N$ in its scope to compute the scaled proxy return.

The CPA target averages initial tCPA across all units, weighted by conversions. For settlement window $\mathcal { W } .$ let $d _ { u }$ be unit u’s initial tCPA and $N _ { u , w }$ its conversions. The target cost allowance $\begin{array} { r } { M _ { \mathcal { W } } = \sum _ { u } d _ { u } N _ { u , \mathcal { W } } } \end{array}$ and total conversions $\begin{array} { r } { N _ { \mathcal { W } } = \sum _ { u } N _ { u , \mathcal { W } } } \end{array}$ give $\rho = M _ { \mathcal W } / N _ { \mathcal W } ;$ when $N _ { \mathcal { W } } = 0$ , we set $\rho = 0$

Both Sparse and General use this computation. Training sets $\eta = 1 , s = 1 0 0 0 0$ , and $\omega = 0 . 2$ to control the cost-deviation weight, reward scaling, and overall reward weight, respectively. Planner’s reward settles at period end; Optimizer’s current-step reward is used in the advantage recursion in Appendix C.1.

## A.2 Period-Level Feedback for Inspector

Period feedback aggregates value generation, conversions, cost control, and adjustment execution across all controlled accounts. Let $F _ { e } , C _ { e } , N _ { e }$ denote aggregate ADV, spend, and conversions in period e. The feedback vector and fixed weights in Eq. (7) are

$$
\begin{array} { r l } & { { \mathbf y } _ { e } = \left[ \widetilde { F } _ { e } , \widetilde { N } _ { e } , m _ { e } ^ { { \mathrm { C P A } } } , - r _ { e } ^ { \mathrm { f o r c e d } } \right] ^ { \top } , } \\ & { { \mathbf w } = \left[ 1 . 0 , 0 . 2 , 0 . 5 , 0 . 1 \right] ^ { \top } . } \end{array}\tag{A.1}
$$

To obtain $\widetilde { F } _ { e }$ and $\widetilde { N } _ { e }$ , we subtract training-set means $\mu _ { F } , \mu _ { N }$ from ADV and conversions, divide by the corresponding standard deviations $\sigma _ { F } , \sigma _ { N }$ plus a numerical stabilizer $\epsilon > 0$ , and clip to [−3, 3]. The means and standard deviations remain frozen during training.

The CPA constraint margin compares actual spending with the target cost allowance $\begin{array} { r } { M _ { e } = \sum _ { u } d _ { u } N _ { e , u } . } \end{array}$ , where u ranges over all controlled units and $N _ { e , u }$ counts unit u’s conversions in period e:

$$
m _ { e } ^ { \mathrm { C P A } } = \mathrm { c l i p } \left( \frac { M _ { e } - C _ { e } } { \operatorname* { m a x } ( M _ { e } , \epsilon ) } , - 3 , 3 \right) .\tag{A.2}
$$

This component is positive below the target cost allowance and negative above it.

The forced-HOLD rate reflects adjustment requests made after the adjustment limit has been reached. Let $H _ { e }$ count SET requests converted to HOLD because the limit has been reached, and $S _ { e }$ count all SET requests:

$$
r _ { e } ^ { \mathrm { f o r c e d } } = \mathrm { c l i p } \bigg ( \frac { H _ { e } } { \operatorname* { m a x } ( S _ { e } , 1 ) } , 0 , 3 \bigg ) .\tag{A.3}
$$

This component enters the feedback vector negatively. Once period $e \mathrm { { s } }$ feedback is settled, the system computes the previous diagnosis’s reward as the weighted change between adjacent periods, $r _ { e - 1 } ^ { I } = \mathbf { w } ^ { \top } ( \mathbf { y } _ { e } - \mathbf { y } _ { e - 1 } )$

## B Offline Optimization Details and Algorithm

## B.1 Candidate Advantages and Sequence-Level Optimization

At the same decision snapshot, the system fixes the policies of the other roles, samples candidate responses for the target role using behavior policy $\pi _ { b } ^ { r } .$ , and computes returns according to the reward definitions in Appendix A. Optimizer’s candidate return is the undiscounted sum of stepwise rewards for the same account from the step when the candidate action takes efect through the end of the period; each stepwise reward retains both the account-level and global feedback terms.

Planner and Optimizer construct leave-one-out advantages using $\operatorname { E q . } \ ( 8 )$ , normalizing with an exponential moving average of within-group return standard deviations; candidate groups with overly small return variance receive zero advantages. Inspector uses each candidate branch’s return minus that of the branch without a diagnosis. Both share the previous period’s feedback, so their diference depends only on next-period outcomes. These advantages stay fixed throughout ofline optimization.

Policy optimization uses the sequence-level objective in $\operatorname { E q . }$ (9). For candidate $^ { g , }$ let $L _ { g }$ denote the response length, and let $p _ { g \ell } ( \theta )$ and $p _ { g \ell } ^ { b }$ denote the conditional probabilities of token ℓ under the current and behavior policies, respectively. The length-normalized sequence importance ratio is

$$
\varrho _ { g } ( \theta ) = \exp \left[ \frac { 1 } { L _ { g } } \sum _ { \ell = 1 } ^ { L _ { g } } \log \frac { p _ { g \ell } ( \theta ) } { p _ { g \ell } ^ { b } } \right] .\tag{B.1}
$$

The KL penalty uses reference policy $\pi _ { \mathrm { r e f } }$ , with token probability $p _ { g \ell } ^ { \mathrm { r e f } }$ . Define $\xi _ { g \ell } = \log [ p _ { g \ell } ^ { \mathrm { r e f } } / p _ { g \ell } ( \theta ) ]$ . The k3 estimate averages $\exp ( \xi _ { g \ell } ) - \xi _ { g \ell } - 1$ over valid response tokens. Behavior and reference policies stay fixed throughout ofline optimization. Appendix D.2 gives the clipping range and KL coeficient.

## B.2 Offline Reinforcement Learning Algorithm

In Algorithm B.1, $S _ { r }$ denotes the set of decision snapshots for role $r \in \mathcal { R } = \{ P , O , I \}$ , and $\Gamma _ { r }$ maps responses to executable actions or diagnostic guidance. $B _ { r } ^ { \mathrm { v a l i d } }$ stores the inputs, responses, returns, fixed advantages, and behavior policy probabilities and versions of valid candidates, and B denotes a training batch.

Algorithm B.1: Ofline Reinforcement Learning   
Input: SFT parameters $\{ \theta _ { r } ^ { \mathrm { S F T } } \}$ , decision snapshots {S<sub>r</sub>}, delivery environment E.   
Output: Ofline parameters $\{ \theta _ { r } ^ { \mathrm { { o f f } } } \}$   
1 $\theta _ { r } \longleftarrow \theta _ { r } ^ { \mathrm { S F T } } ; \mathrm { f i x } \pi _ { b } ^ { \bar { r } } \longleftarrow \pi _ { \theta _ { r } } ; \mathcal { D } _ { r } ^ { \mathrm { o f f } } \longleftarrow \infty ,$ ∀r;   
2 for r ∈ R, σ ∈ S<sub>r</sub> do   
3 Read h<sup>r</sup> from $\sigma ; \{ z _ { g } ^ { r } \} _ { g = 1 } ^ { G } \sim \pi _ { b } ^ { r } ( \cdot \mid h ^ { r } ) ;$   
4 $u _ { g } ^ { r } \gets \Gamma _ { r } ( z _ { g } ^ { r } , h ^ { r } ) ;$ execute candidates separately from $\sigma ;$ compute $\{ J _ { g } ^ { r } \} _ { g = 1 } ^ { G }$ as specified in Appendix B.1;   
5 if r = I then   
6 Run the branch without a diagnosis; compute $J _ { 0 }$ using Eq. (7);   
7 end   
8 Compute and fix $\{ \widehat { A } _ { g } ^ { r } \}$ using Eq. (8) and its normalization rules;   
9 $\mathcal { D } _ { r } ^ { \mathrm { o f f } }  \mathcal { D } _ { r } ^ { \mathrm { o f f } } \cup \mathcal { B } _ { r } ^ { \mathrm { v a l i d } } ;$   
10 end   
11 for r ∈ R do   
12 Fix $\pi _ { \mathrm { r e f } } ^ { r } ;$   
13 for $B \subset { \mathcal { D } } _ { r } ^ { \mathrm { o f f } }$ do   
14 Compute $\varrho _ { g } ^ { r }$ and $\mathcal { L } _ { \mathrm { o f f } } ^ { r }$ using Eqs. (B.1) and (9), and update $\theta _ { r } ;$   
15 end   
16 θ<sup>of</sup><sub>r</sub> ← θ<sub>r</sub>;   
17 end   
18 return $\{ \theta _ { r } ^ { \mathrm { o f f } } \} _ { r \in \mathcal { R } } ;$

## C Co-Evolution Update Details and Algorithm

Co-evolution uses settled feedback from actual execution trajectories. Trajectory $\tau _ { e }$ records inputs, responses, behavior policy versions, generation probabilities, and feedback. Settled records form $\mathcal { D } _ { e }$ at period end, with role $r \mathrm { { s } }$ records in $\mathcal { D } _ { e } ^ { r }$ . The new diagnosis awaits next-period settlement; schedule $k _ { e }$ cycles through $( P , O , I )$

Each role adds newly settled records to its bufer ${ \mathcal { C } } ^ { r }$ and, when scheduled, selects valid records to form the current round’s training set ${ \cal B } _ { e } ^ { r }$ . Used records are removed from the bufer and excluded from later rounds; the remaining records are retained. Updated role policies take efect in the next period.

## C.1 Value Baselines and Advantage Estimation

Each role’s independent PrefixValueModel $V _ { \phi _ { r } } ( h )$ maps the last input token’s D-dimensional hidden represen tation before response generation through LayerNorm(D), Linear(D, 256), Tanh, and Linear(256, 1) to a scalar return. Detached input features block value-model gradients to the role language model.

Before each training round, the system computes value predictions $V _ { i } ^ { - }$ , then advantages and return targets. Planner and Inspector use $r _ { i } ^ { r } - V _ { i } ^ { - }$ as the advantage in Eq. (10), with regression targets $\widehat { R } _ { i } ^ { r } = r _ { i } ^ { r }$ . Optimizer applies GAE within each account and period, including HOLD decisions.

Fixing one period, let $t _ { a , i }$ denote the i-th decision step of account a, $r _ { a , i } ^ { O }$ its reward, $\Delta _ { a , i } = t _ { a , i + 1 } - t _ { a , i }$ the number of environment steps to the next node of the same account, and $b _ { a , i } \in \{ 0 , 1 \}$ the sequence termination indicator. The recursion is

$$
\begin{array} { r l } & { \delta _ { a , i } = r _ { a , i } ^ { O } + \gamma ^ { \Delta _ { a , i } } ( 1 - b _ { a , i } ) V _ { a , i + 1 } ^ { - } - V _ { a , i } ^ { - } , } \\ & { \widehat { A } _ { a , i } ^ { O } = \delta _ { a , i } + \gamma ^ { \Delta _ { a , i } } \lambda ( 1 - b _ { a , i } ) \widehat { A } _ { a , i + 1 } ^ { O } , } \\ & { \widehat { R } _ { a , i } ^ { O } = \widehat { A } _ { a , i } ^ { O } + V _ { a , i } ^ { - } . } \end{array}\tag{C.1}
$$

The discount factor is $\gamma = 1$ and the GAE parameter is $\lambda = 0 . 9 5$ . Each account’s final node in a period has $b _ { a , i } = 1$ and $\Delta _ { a , i } = 0$ , with zero terminal bootstrap value and boundary advantage. Recursion never crosses accounts or periods, even though rewards include all-account feedback. Advantages and return targets are computed over complete account-period sequences before batching and remain fixed throughout the round.

## C.2 Value Regression and Policy Updates

The value model performs SmoothL1 regression against fixed, detached return targets $\widehat { R } _ { i }$ , averaging the per-node regression losses over the settled batch B. Each role’s first scheduled round updates only its value model. Later rounds update the value model first, then the role policy using the previously fixed action advantages. The two models maintain separate optimizers and optimizer states.

Policy updates follow Eq. (11), with an entire response sharing the same action advantage. Let $p _ { \ell } ( \theta )$ and $p _ { \ell } ^ { b }$ denote the conditional probabilities of token ℓ under the current policy and the behavior policy that collected the response. Their ratio is $\varrho _ { \ell } = p _ { \ell } ( \theta ) / p _ { \ell } ^ { b }$ . The selection mask and importance weight are

$$
\begin{array} { r l } & { m _ { \ell } = \mathbf { 1 } \left[ 1 - \varepsilon ^ { - } \leq \varrho _ { \ell } \leq 1 + \varepsilon ^ { + } \right] , } \\ & { \bar { \varrho } _ { \ell } = \mathrm { s g } \left[ \mathrm { c l i p } ( \varrho _ { \ell } , 1 - \varepsilon ^ { - } , 1 + \varepsilon ^ { + } ) \right] . } \end{array}\tag{C.2}
$$

Here, $\varepsilon ^ { - } = \varepsilon ^ { + } = 0 . 2$ , sg stops the gradient, and $\begin{array} { r } { Z = \sum _ { \ell = 1 } ^ { L } m _ { \ell } } \end{array}$ is the number of valid tokens.

Each actor update recomputes probability ratios and masks; advantages and importance weights stay fixed during backpropagation. Responses with $Z = 0$ are skipped for that update. The k3 KL to the behavior policy is used only for monitoring and rollback, not in the actor objective. If KL exceeds 0.01 or the retained-token fraction falls below 0.4, the system restores the parameters and optimizer state from before that update and stops actor updates for the round. Completed value-model updates are retained.

## C.3 Continuous Co-Evolution Algorithm

In Algorithm C.1, let $\Theta _ { e } ^ { r }$ denote role r’s policy parameters, value parameters, and optimizer states, $\nu _ { r }$ its number of completed training rounds, and $\widetilde { \Theta } ^ { r }$ its state at the end of the current round.

Algorithm C.1: Continuous Co-Evolution   
Input: Ofline parameters $\{ \theta _ { r } ^ { \mathrm { o f f } } \}$ , delivery environment $\mathcal { E } ,$ role schedule $\{ k _ { e } \} _ { e = 1 } ^ { E }$   
Output: Final training states $\big \{ \Theta _ { E + 1 } ^ { r } \big \}$   
1 Initialize {Θ<sup>r</sup>} using $\{ \theta _ { r } ^ { \mathrm { o f f } } \} ; \nu _ { r } \gets 0 , \dot { \mathcal { C } } ^ { \dot { r } } \gets \emptyset , \forall r ; g _ { 0 } \gets \emptyset ;$   
2 for $e = 1 , \ldots , E$ do   
3 Execute and record $\mathrm { P / O }$ delivery with $g _ { e - 1 }$ in $\tau _ { e } ;$ compute $\mathrm { P / O }$ rewards using Eqs. (6)–(7) and construct   
${ \bf y } _ { e } ;$   
4 $z _ { e } ^ { I } \sim \pi _ { \theta _ { I } } ( \cdot \mid h _ { e } ^ { I } ) ; g _ { e }  \Gamma _ { I } ( z _ { e } ^ { I } , h _ { e } ^ { I } ) ;$   
5 Append the current diagnosis to $\tau _ { e }$ as pending;   
6 Initialize $\mathcal { D } _ { e }$ with settled $\mathrm { P / O }$ records; if a previous diagnosis is pending, settle its reward using Eq. (7)   
and add its record;   
7 $\begin{array} { r } { \mathcal { C } ^ { r ^ { \prime } }  \mathcal { C } ^ { r ^ { \prime } } \cup \mathcal { D } _ { e } ^ { r ^ { \prime } } , \quad \forall r ^ { \prime } \in \mathcal { R } ; } \end{array}$   
8 $r \gets k _ { e } ;$ form $\boldsymbol { B } _ { e } ^ { r }$ from valid records in ${ \mathcal { C } } ^ { r } ;$ ;   
9 Compute and fix $V _ { i } ^ { - } , \widehat { A } _ { i } ^ { r } , \widehat { R } _ { i } ^ { r }$ for $B _ { e } ^ { r }$ using Eq. (10) and Appendix C.1;   
10 Divide $B _ { e } ^ { r }$ into batches; update $\phi _ { r }$ by value regression in Appendix C.2;   
11 if $\nu _ { r } > 0$ then   
12 Update $\theta _ { r }$ batch by batch using Eq. (11) with filtering and rollback in Appendix C.2;   
13 end   
14 $\mathcal { C } ^ { r }  \mathcal { C } ^ { r } \backslash B _ { e } ^ { r } ;$   
15 $\nu _ { r }  \nu _ { r } + 1 ; \Theta _ { e + 1 } ^ { r }  \widetilde \Theta ^ { r } ; \Theta _ { e + 1 } ^ { r ^ { \prime } }  \Theta _ { e } ^ { r ^ { \prime } } , r ^ { \prime } \neq r ;$   
16 end   
17 return $\{ \Theta _ { E + 1 } ^ { r } \} _ { r \in \mathcal { R } } ;$

## D Experimental Implementation Settings

## D.1 Environment Configuration

Delivery periods use 48 time steps and a trafic generation setting of 500,000 opportunities, following AuctionNet’s oficial defaults [14]. General is calibrated from the General dataset in AuctionNet’s open-source repository: the generation baseline for predicted conversion probabilities (pValue) is 0.005, and budgets and CPA constraints use the full 48-dimensional configurations. Other auction mechanisms remain unchanged.

For each environment configuration, the total trafic volume, initial budgets and CPA constraints, competitortype composition, and pValue generation baseline remain fixed across periods. Each period uses the episode index to resample predicted conversion probabilities and their variability and within-period trafic curves, and to initialize native competitors’ random states, producing diferent market conditions. Both Sparse and General split episodes 71–320 consecutively into five market sequences, each with 50 delivery periods.

## D.2 Training Configuration

For SFT, each role uses 3,290 demonstration trajectories generated by DeepSeek V4 Flash. Training uses a learning rate of $1 0 ^ { - 4 }$ , microbatch size 8, gradient accumulation of 2, and maximum total sequence length of 2,560 tokens. Planner, Optimizer, and Inspector are trained for 4, 3, and 3 epochs, respectively.

Ofline GSPO uses a learning rate of $1 0 ^ { - 5 }$ for 5 epochs, 4 candidates per decision, sequence-ratio clipping to [0.8, 1.2], and a KL penalty coeficient of 0.1. Both stages use LoRA with rank 16 and alpha 32.

Co-evolution uses AdamW at actor and value-model learning rates of $1 0 ^ { - 6 }$ and $5 \times 1 0 ^ { - 4 }$ , respectively. Batches hold valid settled decision nodes for the scheduled role. Rounds with valid nodes update the value model once and the actor at most four times; Appendix C.2 covers the first scheduled round and rollback.

## D.3 Inference and Action Handling

During continuous delivery, responses are sampled at a temperature of 0.7 with thinking disabled. The context and response limits are 8,192 and 512 tokens, respectively. Top-p is not explicitly specified. Optimizer selects HOLD or a tCPA multiplier of 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, or 1.3.

Foundation models and AdCoEvo use the same role workflow, prompts, output format, and action-parsing rules. We specify the required output format in prompts for foundation models, without enforcing constrained JSON generation. If Planner parsing or format validation fails, the remaining budget is allocated in proportion to the ad units’ initial budgets. For Optimizer, parsing failure or an incorrect decision count produces four HOLDs; each invalid multiplier is replaced by HOLD. During evaluation, each invalid response is counted once, regardless of how many errors it contains. Format errors do not trigger resampling.

## D.4 Baseline and Ablation Configurations

AdPilot<sup>\*</sup> combines reflection and action generation in a single model call. Its SFT demonstrations are generated by DeepSeek V4 Flash, and its SFT and GRPO training use the same training markets as AdCoEvo.

Rule multiplies an ad unit’s budget by 1.1 when its CPA does not exceed the target, and by 0.9 otherwise, then normalizes budgets within the account to preserve its total budget. For tCPA, it uses 0.7 if CPA exceeds 1.10 times the target, 1.3 if CPA is below 0.90 times the target, and HOLD otherwise. No conversions or an exhausted adjustment limit yield HOLD. The rules do not use within-period timing or spending progress to determine when to adjust.

Role-composition ablations omit the excluded roles at inference. Without Optimizer, the execution layer applies HOLD throughout delivery and keeps control tCPA at its initial value; without Planner, each ad unit retains its initial budget limit. Without Inspector, the remaining Planner and Optimizer roles receive the current-period delivery state but no cross-period diagnostic guidance or fixed statistical summaries.