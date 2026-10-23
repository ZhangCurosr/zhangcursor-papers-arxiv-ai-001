# METTA: MEMORY-ENHANCED TEST-TIME ADAPTATION FOR META-REINFORCEMENT LEARNING

Ali Najar Hosein HasaniAmir Kooshan Fattah Yasamin RajabiMahdieh Soleymani Baghshah

Department of Computer Engineering Sharif University of Technology {ali.najjar82,hosein.hasani,amir.hesari82, Yasamin.rajabi53,soleymani}@sharif.edu

## ABSTRACT

Meta-reinforcement learning aims to train agents that adapt quickly to unseen tasks from limited interaction, but this requires representing task evidence over long interaction histories. Classical recurrent agents compress experience into hidden states, while transformer-based agents retrieve from past trajectories at quadratic memory and computation cost. In this work, we propose Memory-Enhanced Test-Time Adaptation (METTA) for meta-reinforcement learning. It decomposes the fast adaptation phase into two temporal scales: a causal neural memory performs fine-grained within-episode adaptation with linear sequence complexity, while compact episode-level summaries support long-range cross-episode task inference without attending over the full interaction history. Moreover, equipping agents with neural memory provides sufficient expressive capacity and testtime training capability, enabling effective adaptation to unseen held-out tasks and task families at meta-test time. On unseen held-out Meta-World ML10/45 test environments, METTA achieves a success rate over 50% while maintaining a low training budget and short wall-clock time. On long-context sparse-reward Point-Maze tasks, it outperforms transformer-based baselines by exhibiting more structured exploration behavior. These results show that METTA provides an efficient path toward more adaptive and scalable long-context meta-RL.

## 1 INTRODUCTION

Reinforcement learning (RL) studies how agents learn to make sequential decisions through interaction with an environment, using rewards as feedback to improve future behavior (Sutton & Barto, 2018). Despite strong performance in fixed environments, standard RL methods often require large amounts of data to adapt to new tasks and generalize poorly under environment shifts (Packer et al., 2018; Cobbe et al., 2019). Meta-reinforcement learning addresses this limitation by training agents over a distribution of tasks so they can use past training experience to adapt quickly to unseen tasks from limited interaction at meta-test time (Duan et al., 2016; Wang et al., 2016; Finn et al., 2017; Rakelly et al., 2019). A fundamental problem in meta-RL is representing task evidence collected during interaction with the environment (Humplik et al., 2019; Zintgraf et al., 2021). Black-box sequence-based agents learn this representation directly from trajectories using internal dynamics (Duan et al., 2016; Wang et al., 2016; Mishra et al., 2018; Alver & Precup, 2021). Latent-context methods instead make task inference more explicit by inferring a task variable or belief from collected experience (Humplik et al., 2019; Rakelly et al., 2019; Zintgraf et al., 2021).

Yet despite their differences, most approaches still rely on compressed task representations, which may restrict their capacity in meta-RL settings where task evidence may be sparse, delayed, or unevenly distributed across episodes. Once experience is compressed into a latent task representation, the policy cannot easily recover specific past evidence that later becomes important.

Transformers offer a natural alternative because self-attention enables associative retrieval over past interactions and supports long-range dependency modeling without relying on recurrent state compression (Vaswani et al., 2017; Melo, 2022; Shala et al., 2025). This makes them well suited to cross-episode meta-RL, where useful evidence may appear far back in the trial (Mishra et al., 2018; Shala et al., 2025; Melo, 2022). However, standard self-attention scales quadratically with sequence length (Vaswani et al., 2017; Duman Keles et al., 2023), making long adaptation horizons increasingly expensive in online RL settings. Prior work also showed that transformer optimization in reinforcement learning can be unstable due to exploration dynamics and non-stationary training signals (Parisotto et al., 2019). In online meta-RL, the effective context grows with both the episode horizon and the number of adaptation episodes, making full-history attention increasingly expensive in memory and computation. As a result, transformer-based agents often face a trade-off between retaining long histories and remaining computationally practical.

![](images/3e31424375f1b31774e564225fec877a277fa241b4ff8760524c6144dda84216.jpg)  
Figure 1: Exploration behavior on a sparse-reward PointMaze task (Fu et al., 2020) across adaptation episodes. Trajectories are shown for Episodes 1, 2, 5, and 10 of the same task. As interaction proceeds, METTA explores new regions and reduces repeated trajectories across episodes, suggesting more effective reuse of past exploration evidence in long-context adaptation. TrMRL (Melo, 2022) is difficult to scale to long multi-episode settings, while ECET (Shala et al., 2025) is limited by its short within-episode context.

The computational overhead of full-history attention has motivated a recent shift toward modern linear recurrent models and neural memory architectures (Gu et al., 2021; Smith et al., 2022; Gu & Dao, 2023; Sun et al., 2023; 2025). Instead of attending over the full history, these methods update an adaptive memory state online while preserving linear sequence complexity. Recent work suggests that recurrence need not be limited to a small passive hidden state and can instead be equipped with expressive state dynamics or trainable memory modules, allowing models to store information over long horizons while preserving linear sequence complexity (Gu & Dao, 2023; Sun et al., 2025). In particular, Test-Time Training (TTT) (Sun et al., 2025) treats memory as a trainable module that can be updated continually during inference, turning sequence processing from passive state compression into adaptive memory updating. This has led to promising results in settings where long-range information is essential, including language modeling, time series, genomics, novelview synthesis, and video generation (Sun et al., 2025; Behrouz et al., 2025; Zhang et al., 2025; Dalal et al., 2025).

In this work, we propose Memory-Enhanced Test-Time Adaptation (METTA), which leverages TTT layers (Sun et al., 2025) as a modern neural memory module to gather task evidence from longrange interaction sequences in meta-RL. Besides this within-episode memory module, which provides fine-grained episode-level memories, we also use a coarse-grained aggregation mechanism that combines summaries across episodes. Thus, in addition to the classical slow and fast learning processes in meta-RL, our fast adaptation phase separates within-episode adaptation from crossepisode evidence aggregation, allowing the policy to condition on longer meta-test histories without relying on quadratic computation over agent interaction steps.

Our experiments across locomotion, manipulation, and navigation benchmarks show that METTA improves fast adaptation in both standard and long-context meta-RL settings. Interestingly, the relative gain over existing methods is especially notable when generalizing to unseen held-out Meta-World task families that are not encountered during meta-training. This provides evidence of taskfamily-level out-of-distribution generalization, where adaptation transfers to task families not observed during meta-training. This suggests that neural memory modules can support adaptation beyond the task families observed during training.

In addition, METTA achieves these improvements with substantially lower training cost than transformer-based meta-RL methods, while preserving linear sequence complexity over long horizons. Figure 1 illustrates the role of neural memory in long-context sparse-reward settings. METTA explores more broadly while reducing repeated trajectories across episodes, suggesting more effective reuse of information gathered during earlier episodes. We examine this behavior further in Section 4.2.1.

## 2 PRELIMINARIES

## 2.1 META-REINFORCEMENT LEARNING

Reinforcement learning problems are formulated as Markov decision processes (MDPs), defined by the tuple $( \mathcal S , \mathcal A , P , R , P _ { 0 } , \gamma , H )$ , where $s$ and A denote the state and action spaces, $P ( s ^ { \prime } \mid s , a )$ denotes the transition distribution, $R ( s , a )$ the reward function, $P _ { 0 }$ the initial-state distribution, $\gamma \in$ $[ 0 , 1 ]$ the discount factor, and H the horizon. At timestep t, the agent observes state $s _ { t }$ , samples an action $a _ { t } \sim \pi _ { \psi } ( \cdot \mid s _ { t } )$ from a policy parameterized by $\psi ,$ receives reward $r _ { t } ,$ and transitions to the next state $s _ { t + 1 }$ . The objective is to maximize the expected discounted return

$$
J ( \pi _ { \psi } ) = \mathbb { E } _ { \pi _ { \psi } } \left[ \sum _ { t = 0 } ^ { H } \gamma ^ { t } r _ { t } \right] .\tag{1}
$$

In meta-reinforcement learning, tasks are sampled from a distribution $p ( \mathcal T )$ , where each task $\mathcal { T } _ { i } =$ $( \mathcal { S } , \mathcal { A } , P _ { i } , R _ { i } , P _ { 0 , i } , \gamma , H )$ shares the same observation and action spaces but may differ in transition dynamics, reward functions, or goals. During meta-training, the agent interacts with tasks sampled from $p ( \tau )$ and learns a policy that can adapt efficiently across the task distribution. The meta-RL objective is to maximize expected return across the task distribution.

$$
\operatorname* { m a x } _ { \psi } \mathbb { E } _ { \mathcal { T } \sim p ( \mathcal { T } ) } \left[ \mathbb { E } _ { \pi _ { \psi } } \left[ \sum _ { t = 0 } ^ { H } \gamma ^ { t } r _ { t } \right] \right] .\tag{2}
$$

During meta-testing, the learned policy is evaluated on unknown tasks sampled from the same distribution $p ( \mathcal T )$ . Meta-RL evaluation commonly measures adaptation over multiple episodes from the same task instance or family. The agent must improve behavior using only its internal memory and interaction history during rollout execution.

## 2.2 NEURAL MEMORY

Neural memory provides a common abstraction for sequence models that update an internal state and read from it to produce predictions. This view includes recurrent networks, linear attention, state-space models, and recent test-time memory architectures (Katharopoulos et al., 2020; Gu & Dao, 2023; Sun et al., 2025; Behrouz et al., 2025). Given an input sequence $x _ { 1 : n } ,$ a memory model maintains a state $\mathcal { M } _ { t } = \mathcal { U } _ { \theta } ( \mathcal { M } _ { t - 1 } , x _ { t } )$ and produces an output $y _ { t } = \mathcal { R } _ { \theta } ( \mathcal { M } _ { t } , x _ { t } )$ , where $\mathcal { U } _ { \theta }$ is the update rule and $\scriptstyle { \mathcal { R } } _ { \theta }$ is the readout rule. This notation covers both fixed-size recurrent states and richer memory states designed to store and retrieve information over longer horizons.

Many memory models use an attention-style interface in which each input is projected into a key, value, and query. Given learned projections $\mathbf { k } _ { t } = \theta _ { K } x _ { t } , \mathbf { v } _ { t } = \theta _ { V } x _ { t }$ , and $\mathbf { q } _ { t } = \theta _ { Q } x _ { t }$ , the memory update and readout can be written in terms of key-value writing and query-based retrieval. In general, the write and read operations can be expressed as

$$
\mathcal { M } _ { t } = f ( \mathcal { M } _ { t - 1 } , \mathbf { k } _ { t } , \mathbf { v } _ { t } ) , \qquad y _ { t } = g ( \mathcal { M } _ { t } , \mathbf { q } _ { t } ) ,
$$

where $f$ is the memory update rule, and $g$ is the read function. A simple example is the linear Transformer (Katharopoulos et al., 2020; Schlag et al., 2021), which can be interpreted as a fastweight memory storing key-value associations. In this case $f ( \mathcal { M } _ { t - 1 } , \mathbf { k } _ { t } , \mathbf { v } _ { t } ) = \mathcal { M } _ { t - 1 } ^ { \circ } + \mathbf { v } _ { t } \phi ( \mathbf { k } _ { t } ) ^ { \intercal }$ and $g ( \mathcal { M } _ { t } , \mathbf { q } _ { t } ) = \mathcal { M } _ { t } \phi ( \mathbf { q } _ { t } ) / \mathbf { n } _ { t } ^ { \top } \phi ( \mathbf { q } _ { t } )$ , where $\phi ( \cdot )$ is a feature map and $\begin{array} { r } { \mathbf { n } _ { t } = \sum _ { i = 1 } ^ { t } \phi ( \mathbf { k } _ { i } ) } \end{array}$ is the normalization state. This example shows how attention-like retrieval can be implemented through a recurrent memory update.

![](images/a11006219c0f5906bb99532403123d45f3de7dc505f4cbfafa6a503e8a025631.jpg)  
Figure 2: Overview of METTA. The current episode is processed by a within-episode neural memory to produce $h _ { e , t } ^ { ( L ) }$ . At the end of each episode, the final representation is stored as an episode summary $\bar { h } _ { e }$ . During later episodes, previous summaries are combined with the current representation to form the context representation $z _ { e , t }$ The actor and critic condition on the current state $s _ { e , t }$ and $z _ { e , t }$

Test-Time Training (TTT). One recent neural memory model with linear sequence complexity is TTT, whose hidden states are themselves learnable models updated while processing the sequence (Sun et al., 2025). Given an input sequence $x _ { 1 : n } ,$ a TTT layer maintains an inner-loop model $f _ { \boldsymbol { \mathcal { M } } _ { t } }$ , where $\mathcal { M } _ { t }$ denotes the hidden state after processing the prefix $x _ { 1 : t }$ . The layer therefore keeps the interface of a recurrent sequence model, but replaces the usual hidden vector with the parameters of a model that is trained online.

The memory is updated by training $f _ { \mathcal { M } }$ in a self-supervised manner to predict the label view $\mathbf { v } _ { t }$ from the training view $\mathbf { k } _ { t }$

$$
\ell ( \mathcal { M } ; \mathbf { k } , \mathbf { v } ) = \frac { 1 } { 2 } \left\| f _ { \mathcal { M } } ( \mathbf { k } ) - \mathbf { v } \right\| _ { 2 } ^ { 2 } , \qquad \mathcal { M } _ { t } = \mathcal { M } _ { t - 1 } - \eta _ { t } \nabla _ { \mathcal { M } } \ell ( \mathcal { M } _ { t - 1 } ; \mathbf { k } _ { t } , \mathbf { v } _ { t } ) ,\tag{3}
$$

where $\eta _ { t }$ is a learnable inner-loop learning rate. The output is then produced by applying the updated memory to the test view, $y _ { t } = f _ { \mathcal { M } _ { t } } ( \mathbf { q } _ { t } )$ . Thus, $y _ { t }$ is computed from a memory model that has been adapted on the causal prefix $x _ { 1 : t }$

The distinction between inner-loop and outer-loop parameters is central to TTT. The memory parameters $\mathcal { M } _ { t }$ are fast parameters: they are initialized at the start of a sequence and updated while the sequence is processed, including at test time. In contrast, the view projections $\theta _ { K } , \theta _ { V } , \theta _ { Q }$ , the initial memory parameters, and the downstream network parameters are slow parameters learned across training sequences. Thus, the outer loop learns how to construct a useful self-supervised update rule, while the inner loop applies this rule online to adapt the memory to the current sequence.

## 3 METTA

In this section, we introduce Memory-Enhanced Test-Time Adaptation (METTA) for meta-RL. We consider a trial-based meta-RL setting in which the agent interacts with multiple consecutive episodes from the same sampled task. Each trial is generated from a hidden task $\bar { \tau } \sim p ( \mathcal { T } )$ and contains E episodes, each with horizon H.

At episode e and timestep t, the agent uses the interaction history to construct a context representation $z _ { e , t }$ that summarizes task-relevant evidence accumulated during adaptation. The policy conditions on both the current state and this context representation for action selection,

$$
\begin{array} { r } { a _ { e , t } \sim \pi _ { \psi } \left( \cdot \mid s _ { e , t } , z _ { e , t } \right) . } \end{array}\tag{4}
$$

Besides the standard slow parameter updates during meta-training, METTA performs fast adaptation at two temporal scales. A within-episode neural memory supports fine-grained online adaptation, while a cross-episode memory aggregates information across episodes. Together, these modules

produce the context representation $z _ { e , t }$ , which provides the policy with information accumulated through interaction. Figure 2 summarizes the overall architecture.

All modules are trained end-to-end with PPO (Schulman et al., 2017) over full meta-RL trials. Additional implementation details and full meta-train and meta-test algorithms are provided in $\mathsf { A p - }$ pendix B.

## 3.1 WITHIN-EPISODE NEURAL MEMORY

We use a TTT architecture (Sun et al., 2025) as the within-episode neural memory in METTA. We denote this module by TTTθ, which consists of L stacked TTT layers whose fast weights are updated online while processing each token. METTA is modular with respect to the withinepisode memory architecture. We instantiate this component with TTT in our default model, but the framework can accommodate other causal neural-memory architectures, allowing us to separately study the role of memory design and cross-episode aggregation.

Following $_ \mathrm { R L } \mathrm { ^ { 2 } \mathrm { - s t y l e } }$ conditioning (Duan et al., 2016), METTA processes the sequential input $x _ { e , t } = [ s _ { e , t } , a _ { e , t - 1 } , r _ { e , t - 1 } , d _ { e , t - 1 } ]$ in an autoregressive manner. Sequence processing is performed by $\mathrm { T T T } _ { \theta }$ with linear computational cost. We first embed the interaction input as $h _ { e , t } ^ { ( 0 ) } = \phi ( x _ { e , t } )$ and then pass this representation through L TTT layers. For each layer $l ,$ the model maintains a fast-weight memory $\mathcal { M } _ { e , t } ^ { ( l ) }$ that is initialized at the start of the episode and updated online as the episode unfolds.

Given the previous-layer representation $h _ { e , t } ^ { ( l - 1 ) }$ , the layer forms training, label, and test views,

$$
\begin{array} { r } { \mathbf { k } _ { e , t } ^ { ( l ) } = \theta _ { K } ^ { ( l ) } h _ { e , t } ^ { ( l - 1 ) } , \qquad \mathbf { v } _ { e , t } ^ { ( l ) } = \theta _ { V } ^ { ( l ) } h _ { e , t } ^ { ( l - 1 ) } , \qquad \mathbf { q } _ { e , t } ^ { ( l ) } = \theta _ { Q } ^ { ( l ) } h _ { e , t } ^ { ( l - 1 ) } . } \end{array}
$$

It first writes the current evidence into memory through a gradient step on the self-supervised loss defined in Eq. (3),

$$
\mathcal { M } _ { e , t } ^ { ( l ) } = \mathcal { M } _ { e , t - 1 } ^ { ( l ) } - \eta _ { e , t } ^ { ( l ) } \nabla _ { \mathcal { M } } \ell \left( \mathcal { M } _ { e , t - 1 } ^ { ( l ) } ; \mathbf { k } _ { e , t } ^ { ( l ) } , \mathbf { v } _ { e , t } ^ { ( l ) } \right) .
$$

The updated memory is then queried to produce the next-layer representation, $h _ { e , t } ^ { ( l ) } = f _ { \mathcal { M } _ { e , t } ^ { ( l ) } } ( \mathbf { q } _ { e , t } ^ { ( l ) } )$ After the final layer, $h _ { e , t } ^ { ( L ) }$ is the adapted context representation of the current episode prefix.

## 3.2 CROSS-EPISODE MEMORY

The within-episode TTT module adapts to the current episode, but meta-RL trials also contain useful information across episodes. We use the final adapted representation of each episode, $\bar { h } _ { e } = h _ { e , H } ^ { ( L ) }$ as an episode-level summary. At episode $e ,$ the agent causally has access to all previous summaries $\mathcal { H } _ { e - 1 } \dot { = } \{ \bar { h } _ { i } \} _ { i = 1 } ^ { e - 1 }$ . We consider two mechanisms for aggregating these summaries to construct the context representation $z _ { e , t } ,$ which is used as input to the actor and critic heads.

Exponential moving average. A simple aggregation mechanism is to maintain an exponential moving average (EMA) of previous episode summaries. Let $c _ { e - 1 }$ denote the compact cross-episode summary available before episode $e _ { \cdot }$ For $e > 1$ , the current within-episode representation is combined with the recent cross-episode summary to form the current context representation

$$
z _ { e , t } = \alpha c _ { e - 1 } + ( 1 - \alpha ) h _ { e , t } ^ { ( L ) } ,\tag{5}
$$

where $\alpha \in [ 0 , 1 ]$ controls the influence of previous episodes. For the first episode, no previous summary is available, so $z _ { 1 , t } = h _ { 1 , t } ^ { ( L ) }$ and $c _ { 1 } = \bar { h } _ { 1 }$ . The EMA summary for episode e is updated as

$$
c _ { e } = \alpha c _ { e - 1 } + ( 1 - \alpha ) \bar { h } _ { e } .\tag{6}
$$

This variant stores only a single vector and updates cross-episode information in constant time.

Attention-based retrieval. While EMA is efficient, it does not allow selective retrieval of previous episodes. Therefore, we use an attention layer over stored episode summaries as the default

![](images/13279791805b7baad5f2e590a51382494b7b26580feb0f54c15414f71af7bbfa.jpg)

![](images/a66ca6a869869ad686330e84a060676117f56dd94f291f57a0031729edeb75e8.jpg)  
(a) MuJoCo HalfCheetahVel.

![](images/d475948164a6c0db1ab64040a7133b3cfcc90f5dd707a161e719d9ac55e110c0.jpg)  
(b) Sparse-reward PointMaze.

![](images/60d1f9cfb576bf83ba728d7180cc15fcf5e4d91d714c0d1a6203d5943aeb4107.jpg)  
Figure 3: Fast adaptation during meta-training and meta-testing. (a) HalfCheetahVel training return and adaptation across meta-test episodes. (b) PointMaze training and unseen held-out test success rate.

aggregation mechanism in METTA. This learnable attention layer forms a query $q _ { e , t } = W _ { q } h _ { e , t } ^ { ( L ) }$ and uses the unordered set of episode summaries $\mathcal { H } _ { e - 1 }$ as keys and values for information retrieval:

$$
\tilde { h } _ { e , t } = \mathrm { A t t n } \left( q _ { e , t } , \mathcal { H } _ { e - 1 } \right) .\tag{7}
$$

The retrieved cross-episode representation is fused with the current within-episode representation,

$$
z _ { e , t } = \mathrm { L a y e r N o r m } \left( h _ { e , t } ^ { ( L ) } + \tilde { h } _ { e , t } \right) .\tag{8}
$$

The resulting representation $z _ { e , t }$ is passed to the RL agent (actor and critic modules) along with the current state $s _ { e , t }$ . Compared with EMA, attention-based retrieval is more expressive because it can learn which previous episode summaries contain the most relevant task information.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Benchmarks. We evaluate the effectiveness of our method on standard meta-RL benchmarks covering locomotion, manipulation, and navigation tasks. We evaluate continuous-control adaptation on MuJoCo HalfCheetahVel (Todorov et al., 2012), where the agent must infer the target velocity from interaction. We use a limited training budget to compare how efficiently each method adapts under restricted environment interaction.

Next, we evaluate robotic manipulation on ML1, ML10, and ML45 from the Meta-World benchmark (Yu et al., 2020). In ML1, following the standard baseline convention, goal coordinates are masked and must be inferred through interaction. ML10 and ML45 span multiple task families with disjoint train and test splits, allowing us to evaluate generalization to families not observed during meta-training. We view this as a form of task-family-level OOD evaluation, since the held-out test families are encountered only at meta-test time. In these settings, goal coordinates are included in the observation, while the task family remains fixed within each trial. During multi-class training, each rollout batch assigns environments to task classes in a balanced way and then samples a task variation within each class.

Finally, we use PointMaze (Fu et al., 2020) with sparse rewards (only at the goal position) to evaluate long-context adaptation. In this environment, the agent may need to explore for many steps before receiving useful feedback, making it important to preserve information from earlier parts of the trajectory. This benchmark directly tests whether meta-RL methods can benefit from long context.

Baselines. We compare METTA against representative meta-RL baselines from recurrent, gradient-based, latent-context, and transformer-based families. RL2 uses recurrent state to implement black-box adaptation from interaction history (Duan et al., 2016), while MAML learns an initialization that can be adapted with gradient updates (Finn et al., 2017). PEARL and VariBAD represent latent-context approaches that infer task variables from collected experience (Rakelly et al., 2019; Zintgraf et al., 2021). For transformer-based meta-RL, we compare against TrMRL, which uses self-attention over trajectory context (Melo, 2022), ECET, which improves crossepisode conditioning (Shala et al., 2025), and AMAGO, an in-context RL agent trained on long sequences (Grigsby et al., 2024). For all baselines, we use the default implementations and hyperparameters from their respective repositories. All methods use the same evaluation protocol. Results are averaged over five seeds, with shaded regions showing ±1 SEM. Additional details are provided in Appendix D.

METTA ECET MAML PPO PEARL — RL2 TrMRL VariBAD AMAGO  
![](images/db9d34757579f14c35f6922e3a0f74f3c1f0fa71faff6c92b6fce27d18194508.jpg)  
Figure 4: Meta-World success rate on ML1, ML10, and ML45. Top: evaluation on unseen task instances from task families used during meta-training. Bottom: evaluation on the Meta-World test benchmark, which contains unseen held-out task families for ML10 and ML45.

## 4.2 MAIN RESULTS

In this section, we focus on standard evaluations of baselines using well-known metrics, average return and success rate, and leave additional analyses, including ablation studies and qualitative results, to Appendix G. Figure 3a reports HalfCheetahVel performance under a limited training budget. The training curve shows that METTA improves rapidly with low training variance. The episode-wise adaptation plot in Figure 3a shows that METTA maintains strong returns from the first episode and remains stable across later episodes, suggesting that its neural memory can quickly infer the target velocity from interaction.

## 4.2.1 ROLE OF LONG-CONTEXT MEMORY

Figure 3b shows that, in long-horizon tasks with sparse rewards, METTA rapidly improves the success rate during both meta-training and meta-testing, while the other baselines remain substantially lower. This suggests that METTA can preserve useful evidence from earlier exploration and reuse it in later episodes. Transformer-based baselines (Melo, 2022; Shala et al., 2025) struggle in this setting. TrMRL is difficult to scale to long effective horizons and tends to repeat similar exploration behavior, while ECET is limited by short-context cross-episode conditioning. VariBAD improves more slowly, likely because sparse rewards provide weak evidence for updating the latent task belief until successful exploration occurs (Zintgraf et al., 2021).

The exploration patterns in Figure 1 and the additional trajectories in Appendix G.6 are further supported by Table 1. Coverage measures the fraction of the traversable maze visited, while Episode Jaccard measures overlap between consecutive episodes, with lower values indicating less repetitive exploration. Steps measures the interactions

Table 1: PointMaze exploration at 1.5M environment steps (mean ± SEM).
<table><tr><td>Method</td><td></td><td>Coverage ↑ Episode Jaccard↓</td><td>Steps ↓</td></tr><tr><td>METTA</td><td> $\mathbf { 0 . 4 4 7 \pm 0 . 0 3 6 }$ </td><td> $\mathbf { 0 . 7 9 2 \pm 0 . 0 2 2 }$ </td><td> ${ \bf 1 9 4 . 8 \pm 3 0 . 1 }$ </td></tr><tr><td>ECET</td><td> $0 . 2 5 0 \pm 0 . 0 8 7$ </td><td> $0 . 9 3 5 \pm 0 . 0 2 8$ </td><td> $4 2 7 . 3 \pm 4 4 . 5$ </td></tr><tr><td>TrMRL</td><td> $0 . 3 3 0 \pm 0 . 0 3 1$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $3 7 3 . 4 \pm 1 6 . 1$ </td></tr></table>

required before success or horizon termination. METTA achieves the highest coverage and lowest overlap while requiring the fewest steps, suggesting that it maintains diverse exploration across episodes without sacrificing navigation efficiency. In contrast, ECET and TrMRL cover less of the maze while revisiting previously explored regions more frequently. Detailed definitions are provided in Appendix G.7.

![](images/58e1fe7dd2630e7ca2444ceb1ffac8909fd80b7c6ece1e3ea413c897ebf38a63.jpg)  
Figure 5: Structure of the learned context representations. Left: t-SNE visualization of z for all episodes. Circles denote training tasks and triangles denote test tasks. Right: pairwise cosinesimilarity matrix computed from the final episode representation of each trial.

Figure 4 shows adaptation performance on the Meta-World benchmark. On simpler settings such as ML1, METTA quickly improves over adaptation episodes, indicating that the neural memory can infer task-relevant information from limited interaction. On the more challenging ML10 and ML45 benchmarks, METTA surpasses other baselines by a large margin across diverse task families.

The performance gap is notable not only on train environments, where evaluation uses task families observed during meta-training, but also on the Meta-World test environments (up to a \~ 40% success-rate margin), which consist of unseen held-out task families. This result indicates that the learned adaptation mechanism of METTA generalizes beyond the task families encountered during meta-training. In our experiments, this generalization is substantially stronger than that of RL² (Duan et al., 2016), suggesting an advantage of adaptive neural memory over conventional autoregressive recurrent conditioning, which relies on passive and less expressive memory updates during inference.

To further analyze the structure of the learned context, we measure representation similarity both quantitatively and qualitatively. Figure 5 shows cosine similarities of the context representations on ML10 along with a t-SNE visualization. The t-SNE plot shows that METTA organizes context representations according to task families while keeping test tasks aligned with related training tasks, suggesting generalization beyond memorization. The cosine-similarity matrix shows consistent representations within each task family through large diagonal values, while the off-diagonal structure captures relationships across different task families. These results indicate that the learned context contains task-discriminative structure while preserving meaningful similarities across tasks.

## 4.3 ONLINE MEMORY UPDATES

To isolate the contribution of test-time memory adaptation, we evaluate METTA with its online memory updates disabled while retaining cross-episode aggregation. Figure 6a shows that the effect depends strongly on how task-relevant information is obtained. On PointMaze, disabling the online update causes a large drop in test performance, showing that continual within-episode adaptation is particularly important when useful evidence is sparse and must be accumulated through interaction. In contrast, the two variants perform similarly on ML10, where the goal is observed and adaptation mainly requires identifying the task family, which is easy to distinguish once learned, beyond this, online updating makes little difference.

Motivated by this contrast, we further evaluate the same ablation on ML1 Push, where the goal is hidden and must instead be inferred through interaction. Figure 6b shows that, with online updates, METTA reaches substantially higher success in the first evaluation episode, before any previousepisode information is available for cross-episode aggregation. Without online updates, performance

— METTA (online TTT update)  METTA (no online TTT update)

![](images/5b1817a6a5fa18241a7a816069a67d2a1d4a39acb6177f7f3c86c95598bfd56d.jpg)

![](images/9b20bb04ef322687eae42200ef5c3456a7172367821a2cc7ee0912e7c8a2a78b.jpg)  
METTA (online TTT update)  METTA (no online TTT update)  
(a) ML10 and PointMaze test performance.

![](images/95967b92de843748fbbea2870c83b4bcbfbba9bde309038f925d7d4d33c765ac.jpg)

![](images/39940442d8e7363572b3a4a14e58cdf1d163593f30ee59a7682f8c6d2631f621.jpg)  
(b) ML1 Push adaptation performance.

Figure 6: Effect of METTA test-time adaptation. The ML1 Push results report adaptation episodes 1 and 25 within each evaluation trial.

is much lower initially and improves as information accumulates across episodes, but remains below the full model even after 25 episodes. This shows that cross-episode aggregation alone does not recover the benefit of within-episode neural memory adaptation when task-relevant information must be inferred from interaction.

To further examine the effect of the within-episode memory design, we integrate METTA with alternative neural memory architectures, including GRU, Mamba (Gu & Dao, 2023), and linear Transformer (Katharopoulos et al., 2020), and compare their performance in Appendix G.1. We also study alternative cross-episode aggregation mechanisms in Appendix G.2 to evaluate how the two memory components contribute to adaptation.

## 4.4 TRAINING EFFICIENCY

Figure 7 compares the wall-clock training cost of METTA against transformer-based methods and standard meta-RL baselines on ML10. METTA dominates transformer-based methods, achieving stronger performance with substantially lower GPU-hour cost. At the same time, its training time remains comparable to widely used baselines, while surpassing their performance by a large margin. This shows that neural memory provides a favorable trade-off between longcontext adaptation and practical training cost.

![](images/0264e309aa8c1e06662579650c95fe5566d519e5a399abf04a531027de77ec9f.jpg)  
Figure 7: Success versus GPU hours.

Online inference complexity. Let T denote the episode horizon and E the number of episodes in a trial. Over a complete trial, METTA requires $\mathcal { O } ( E ^ { 2 } \dot { T } )$ computation with attention-based aggregation and O(ET) with EMA, compared with $\mathcal { O } ( E ^ { 2 } T ^ { 2 } )$ for TrMRL (Melo, 2022) and $\mathcal { O } ( E T ^ { 3 } + \mathbf { \bar { \nabla } } E ^ { 3 } T )$ for ECET (Shala et al., 2025). Since typically $E \ll T$ , the dependence on timestep-level history dominates in long-context settings. TTT incurs a constant per-step cost from its inner update, but this cost does not grow with timestep-level history. Detailed derivations and measured inference time and memory are provided in Appendix F.

## 5 CONCLUSION

We presented METTA, a meta-RL framework based on neural memory and test-time training. The method separates fast adaptation into within-episode online memory updates and cross-episode evidence aggregation, allowing the agent to leverage long interaction histories without attending over the full history. This temporal hierarchy provides an efficient mechanism for long-context adaptation with linear sequence complexity. Unlike standard recurrent approaches that rely on passive memory updates during inference, the TTT module performs gradient-based memory updates, providing an additional mechanism for fast adaptation to previously unseen task families.

Experiments on HalfCheetahVel, Meta-World, and sparse-reward PointMaze showed that METTA improves adaptation across both standard and long-context meta-RL settings while maintaining favorable computational efficiency. The gains were especially strong on unseen held-out Meta-World task families, providing further evidence that neural memory with test-time adaptation can support task-family-level OOD generalization to previously unseen, novel task families. Our findings also suggest a promising direction for future work that extends METTA beyond the fixed-task trial setting considered here, for example by incorporating temporal or recency information into the crossepisode memory for continual or nonstationary environments that require fast online adaptation as task structure, dynamics, or rewards change over time.

## AI USE STATEMENT

We used generative AI tools to assist in polishing the language, improving clarity and grammar, and debugging the experimental code. These tools did not contribute to the core research ideas, methodological design, or experimental design of this work. All technical decisions, experiments, analyses, and interpretations were carried out and verified by the authors, who take full responsibility for the content of the paper.

## REPRODUCIBILITY STATEMENT

The main paper and appendix document the model architecture, training procedure, benchmark settings, hyperparameters, statistical protocol, baseline configurations, and computational environment used in our experiments. Appendix B provides implementation and optimization details, Appendix C describes the benchmark configurations, and Appendix D documents the baseline implementations and evaluation protocol. The supplementary material includes the training and evaluation code and experiment configurations needed to reproduce the reported results. We will make the complete repository publicly available upon acceptance.

## REFERENCES

Maruan Al-Shedivat, Trapit Bansal, Yuri Burda, Ilya Sutskever, Igor Mordatch, and Pieter Abbeel. Continuous adaptation via meta-learning in nonstationary and competitive environments, 2018. URLhttps://arxiv.org/abs/1710.03641.

Safa Alver and Doina Precup. What is going on inside recurrent meta-reinforcement learning agents? arXiv preprint arXiv:2104.14644, 2021.

Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W Hoffman, David Pfau, Tom Schaul, Brendan Shillingford, and Nando De Freitas. Learning to learn by gradient descent by gradient descent. Advances in neural information processing systems, 29, 2016.

Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. Titans: Learning to memorize at test time. In Advances in Neural Information Processing Systems, 2025.

Yoshua Bengio, Samy Bengio, and Jocelyn Cloutier. Learning a synaptic learning rule, volume 2. Université de Montréal, Département d’informatique et de recherche opérationnelle, 1990.

Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch. Decision transformer: Reinforcement learning via sequence modeling, 2021. URL https://arxiv.org/abs/2106.01345.

Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Łukasz Kaiser, David Belanger, Lucy Colwell, and Adrian Weller. Rethinking attention with Performers. In International Conference on Learning Representations (ICLR), 2021.

Karl Cobbe, Oleg Klimov, Chris Hesse, Taehoon Kim, and John Schulman. Quantifying generalization in reinforcement learning. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 1282–1289. PMLR, 09–15 Jun 2019. URL https : //proceedings.mlr.press/v97/cobbe19a.html.

Karan Dalal, Daniel Koceja, Jiarui Xu, Yue Zhao, Shihao Han, Ka Chun Cheung, Jan Kautz, Yejin Choi, Yu Sun, and Xiaolong Wang. One-minute video generation with test-time training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 17702– 17711,2025.

Yan Duan, John Schulman, Xi Chen, Peter L. Bartlett, Ilya Sutskever, and Pieter Abbeel. RL2: Fast reinforcement learning via slow reinforcement learning. arXiv preprint arXiv:1611.02779, 2016.

Feyza Duman Keles, Pruthuvi Mahesakya Wijewardena, and Chinmay Hegde. On the computational complexity of self-attention. In Proceedings of The 34th International Conference on Algorithmic Learning Theory, volume 201 of Proceedings of Machine Learning Research, pp. 597–619. PMLR, 2023.

Rasool Fakoor, Pratik Chaudhari, Stefano Soatto, and Alexander J. Smola. Meta-Q-learning, 2020. URLhttps://arxiv.org/abs/1910.00125.

Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pp. 1126–1135. PMLR, 2017.

Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine. D4RL: Datasets for deep data-driven reinforcement learning. arXiv preprint arXiv:2004.07219, 2020.

Jake Grigsby, Linxi Fan, and Yuke Zhu. AMAGO: Scalable in-context reinforcement learning for adaptive agents. In The Twelfth International Conference on Learning Representations, 2024.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.

Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured state spaces. arXiv preprint arXiv:2111.00396, 2021.

Arthur Guez, David Silver, and Peter Dayan. Better optimism by Bayes: Adaptive planning with rich models,2014. URL https://arxiv.org/abs/1402.1958.

Abhishek Gupta, Russell Mendonca, YuXuan Liu, Pieter Abbeel, and Sergey Levine. Metareinforcement learning of structured exploration strategies. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, pp. 5307–5316, Red Hook, NY, USA, 2018. Curran Associates Inc.

Timothy Hospedales, Antreas Antoniou, Paul Micaelli, and Amos Storkey. Meta-learning in neural networks: A survey,2020.URL https://arxiv.org/abs/2004.05439.

Jan Humplik, Alexandre Galashov, Leonard Hasenclever, Pedro A. Ortega, Yee Whye Teh, and Nicolas Heess. Meta reinforcement learning as task inference. arXiv preprint arXiv:1905.06424, 2019.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are RNNs: Fast autoregressive transformers with linear attention. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 5156–5165. PMLR, 2020.

Luckeciano C. Melo. Transformers are meta-reinforcement learners. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 15340–15359. PMLR, 2022.

Nikhil Mishra, Mostafa Rohaninejad, Xi Chen, and Pieter Abbeel. A simple neural attentive metalearner. In International Conference on Learning Representations, 2018.

Alex Nichol, Joshua Achiam, and John Schulman. On first-order meta-learning algorithms. arXiv preprint arXiv:1803.02999, 2018.

Charles Packer, Katelyn Gao, Jernej Kos, Philipp Krähenbühl, Vladlen Koltun, and Dawn Song. Assessing generalization in deep reinforcement learning. arXiv preprint arXiv: 1810.12282, 2018.

Emilio Parisotto, H. Francis Song, Jack W. Rae, Razvan Pascanu, Caglar Gulcehre, Siddhant M. Jayakumar, Max Jaderberg, Raphael Lopez Kaufman, Aidan Clark, Seb Noury, Matthew M. Botvinick, Nicolas Heess, and Raia Hadsell. Stabilizing transformers for reinforcement learning,2019.URLhttps://arxiv.org/abs/1910.06764.

Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Leon Derczynski, Xingjian Du, Matteo Grella, Kranthi Gv, Xuzheng He, Haowen Hou, Przemyslaw Kazienko, Jan Kocon, Jiaming Kong, Bartłomiej Koptyra, Hayden Lau, Jiaju Lin, Krishna Sri Ipsit Mantri, Ferdinand Mom, Atsushi Saito, Guangyu Song, Xiangru Tang, Johan Wind, Stanisław Woźniak, Zhenyuan Zhang, Qinghua Zhou, Jian Zhu, and Rui-Jie Zhu. RWKV: Reinventing RNNs for the transformer era. In Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 14048–14077, 2023.

Kate Rakelly, Aurick Zhou, Deirdre Quillen, Chelsea Finn, and Sergey Levine. Efficient off-policy meta-reinforcement learning via probabilistic context variables. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 5331–5340. PMLR, 2019.

Jonas Rothfuss, Dennis Lee, Ignasi Clavera, Tamim Asfour, and Pieter Abbeel. ProMP: Proximal meta-policy search. arXiv preprint arXiv:1810.06784, 2018.

Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, and Timothy Lillicrap. Metalearning with memory-augmented neural networks. In International conference on machine learning, pp. 1842–1850. PMLR, 2016.

Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. Linear transformers are secretly fast weight programmers. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 9355–9366. PMLR, 2021.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Gresa Shala, André Biedenkapp, Pierre Krack, Florian Walter, and Josif Grabocka. Efficient crossepisode Meta-RL. In International Conference on Learning Representations, 2025.

Jimmy T. H. Smith, Andrew Warrington, and Scott W. Linderman. Simplified state space layers for sequence modeling. arXiv preprint arXiv:2208.04933, 2022

Bradly C. Stadie, Sergey Levine, and Pieter Abbeel. Incentivizing exploration in reinforcement learning with deep predictive models, 2015. URL https: //arxiv.org/abs/1507. 00814.

Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, and Carlos Guestrin. Learning to (Learn at test time): RNNs with expressive hidden states. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 57503–57522. PMLR, 2025.

Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. Retentive network: A successor to transformer for large language models. arXiv preprint arXiv:2307.08621, 2023.

Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. MIT Press, second edition, 2018.

Sebastian Thrun and Lorien Pratt. Learning to learn: Introduction and overview. In Learning to learn, pp. 3–17. Springer, 1998.

Emanuel Todorov, Tom Erez, and Yuval Tassa. MuJoCo: A physics engine for model-based control. In 2012 IEEE/RSJ international conference on intelligent robots and systems, pp. 5026–5033. IEEE, 2012.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017.

Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Koray Kavukcuoglu, and Daan Wierstra. Matching networks for one shot learning. Advances in neural information processing systems, 29, 2016.

Jane X. Wang, Zeb Kurth-Nelson, Dhruva Tirumala, Hubert Soyer, Joel Z. Leibo, Remi Munos, Charles Blundell, Dharshan Kumaran, and Matt Botvinick. Learning to reinforcement learn. arXiv preprint arXiv:1611.05763, 2016.

Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-World: A benchmark and evaluation for multi-task and meta reinforcement learning. In Conference on robot learning, pp. 1094–1100. PMLR, 2020.

Tianyuan Zhang, Sai Bi, Yicong Hong, Kai Zhang, Fujun Luan, Songlin Yang, Kalyan Sunkavalli, William T. Freeman, and Hao Tan. Test-time training done right. arXiv preprint arXiv:2505.23884, 2025.

Luisa Zintgraf, Sebastian Schulze, Cong Lu, Leo Feng, Maximilian Igl, Kyriacos Shiarlis, Yarin Gal, Katja Hofmann, and Shimon Whiteson. VariBAD: Variational bayes-adaptive deep rl via meta-learning. Journal of Machine Learning Research, 22(289):1–39, 2021.

## A RELATED WORK

Reinforcement learning methods often achieve strong performance within a fixed environment but generalize poorly across tasks or changing dynamics (Sutton & Barto, 2018; Packer et al., 2018; Cobbe et al., 2019). This limitation motivated research on meta-learning, where models are trained over a distribution of tasks to learn transferable inductive biases that improve adaptation to unseen settings (Hospedales et al., 2020; Bengio et al., 1990; Thrun & Pratt, 1998). Early meta-learning work showed that experience across tasks can improve fast adaptation to unseen settings (Finn et al., 2017; Santoro et al., 2016; Vinyals et al., 2016). Meta-RL extends this idea to sequential decision making, where the agent must infer task structure and adapt behavior using limited interaction data (Duan et al., 2016; Wang et al., 2016; Finn et al., 2017). Unlike conventional RL, which optimizes a single policy for one environment, meta-RL explicitly targets fast adaptation under task variation.

Meta-learning methods are commonly divided into optimization-based, metric-based, and memorybased approaches. Optimization-based methods learn parameters or update rules that support rapid adaptation through gradient-based learning (Finn et al., 2017; Nichol et al., 2018; Andrychowicz et al., 2016). Metric-based methods instead learn embedding spaces for fast similarity-based generalization (Vinyals et al., 2016), while memory-based approaches use recurrent architectures or external memory to encode task information over time (Santoro et al., 2016). Although most meta-RL work builds on optimization-based learning (Finn et al., 2017), recurrent adaptation and latent memory mechanisms became particularly important because RL environments are sequential and partially observable.

Early meta-RL methods mainly relied on recurrent policies that implicitly infer task identity from trajectories by compressing interaction history into recurrent hidden states. RL² (Duan et al., 2016) formulated the problem as learning an RL algorithm inside a recurrent network, where hidden states encode adaptation behavior (Wang et al., 2016). Subsequent work improved task inference using probabilistic latent variables and context encoders. VariBAD (Zintgraf et al., 2021) modeled meta-RL as approximate Bayesian inference over hidden task variables, while PEARL (Rakelly et al., 2019) introduced off-policy probabilistic context adaptation and significantly improved sample efficiency. Other influential methods include ProMP (Rothfuss et al., 2018), MAESN (Gupta et al., 2018), and Meta-Q-Learning (MQL) (Fakoor et al., 2020), which use recurrent or latent context encoders to improve adaptation efficiency and task inference. Gradient-based meta-RL approaches directly adapted MAML to policy optimization settings (Al-Shedivat et al., 2018), enabling fast policy adaptation using a few gradient steps.

A major challenge in meta-RL is balancing exploration and adaptation under partial observability, particularly when task-relevant evidence is sparse or delayed. In many benchmarks, the underlying task is hidden and must be inferred from sparse interactions. This led to methods that explicitly separate task inference from control (Rakelly et al., 2019; Zintgraf et al., 2021). Bayesian formulations became especially influential because they model uncertainty over latent task identity and support structured exploration policies. VariBAD (Zintgraf et al., 2021) and PEARL (Rakelly et al., 2019) showed that latent posterior estimation improves adaptation efficiency and robustness. Related work also studied exploration-driven adaptation, where agents learn information-seeking behaviors that reduce uncertainty during early episodes (Gupta et al., 2018; Stadie et al., 2015). These formulations closely connect meta-RL with partially observable Markov decision processes and Bayesian reinforcement learning (Guez et al., 2014), where the agent must maintain and update beliefs over latent task structure.

Attention mechanisms and transformer architectures have been applied to reinforcement learning and meta-reinforcement learning to improve long-range retrieval over interaction histories (Mishra et al., 2018; Parisotto et al., 2019; Chen et al., 2021). Recent transformer-based meta-RL methods such as TrMRL (Melo, 2022), ECET (Shala et al., 2025), and AMAGO (Grigsby et al., 2024) use sequence modeling over trajectories or episodes to improve task inference and adaptation. These methods reduce reliance on compressed recurrent states and improve long-context reasoning. However, the high memory usage and quadratic computation cost of self-attention make transformerbased methods less suitable for long-horizon sparse-reward settings. In contrast, METTA uses neural memory with linear-time recurrent updates, preserving both high expressivity and linear sequence complexity, which makes it scalable to considerably longer horizons.

Linear-time alternatives to quadratic self-attention have been studied through recurrent, linearattention, and state-space sequence models. State-space and linear recurrent models such as S4 (Gu et al., 2021), S5 (Smith et al., 2022), RWKV (Peng et al., 2023), RetNet (Sun et al., 2023), and Mamba (Gu & Dao, 2023) improve scalability through recurrent memory updates and causal streaming inference. Linear and kernelized attention methods (Katharopoulos et al., 2020; Choromanski et al., 2021) reinterpret attention as associative recurrent computation with subquadratic complexity. More recent neural memory approaches extend recurrent sequence models beyond passive hiddenstate compression by updating memory parameters during inference. Test-Time Training (TTT) layers (Sun et al., 2025) formulate the hidden state as a learnable model optimized online through self-supervised updates, while Titans (Behrouz et al., 2025) introduces persistent neural memory modules for long-range associative retrieval. These architectures provide efficient long-horizon memory and online adaptation mechanisms that are well suited to meta-reinforcement learning and partially observable environments.

## B IMPLEMENTATION DETAILS

## B.1 SEQUENTIAL PPO OPTIMIZATION

Standard PPO implementations often construct minibatches by randomly sampling timesteps from the rollout buffer. For adaptive sequence models, this is inefficient. Evaluating a randomly selected timestep requires reconstructing the neural-memory state by forwarding the trajectory from the beginning of the trial up to that point. Random timestep sampling therefore repeatedly recomputes overlapping prefixes.

To improve training efficiency, we perform PPO optimization sequentially over trajectory segments Each trial is divided into ordered segments that are processed chronologically, and the PPO loss is applied only to the current segment. In our experiments, each segment corresponds to one episode within the trial. This avoids repeatedly reconstructing arbitrary trajectory prefixes and roughly halves the PPO update time in our implementation.

During rollout collection, the TTT fast weights and cross-episode summaries evolve causally and are used for action selection, while the behavior-policy log probabilities and value estimates are stored. PPO optimization then reconstructs the memory state from the stored interactions rather than reusing the rollout memory states. For a segment in episode e, all preceding episodes and the required prefix of episode e are reprocessed using the current slow parameters. After the PPO loss for that segment is backpropagated and the slow parameters are updated, the computation graph is discarded. The causal history required for the next segment is therefore recomputed using the updated parameters, so memory states produced before an optimizer step are never reused by later segments.

Gradients are propagated through the complete reconstructed causal computation for the current segment, including the preceding episode representations used by cross-episode aggregation and the differentiable TTT inner memory updates. Although the PPO loss is evaluated only on the current segment, gradients can flow through reprocessed preceding episodes when their representations contribute to the current context. We do not apply a stop-gradient within this reconstructed PPO computation, including through the fast-weight updates, allowing the outer PPO objective to train the slow TTT parameters through their effect on online adaptation. The computation graph ends at each PPO optimizer step; the required history is reconstructed for the next segment using the updated parameters.

Advantage estimates are computed over the flattened trial trajectory using generalized advantage estimation (GAE),

$$
\hat { A } _ { t } = \sum _ { l = 0 } ^ { \infty } ( \gamma \lambda ) ^ { l } \delta _ { t + l } ,
$$

where $\delta _ { t } = r _ { t } + \gamma V _ { t + 1 } - V _ { t }$ . To stabilize optimization across heterogeneous tasks, observations are normalized online and rewards are scaled by running discounted-return statistics. After computing GAE over the flattened trial trajectory, advantages are normalized over valid rollout positions, optionally within each task class when task labels are available.

Policy optimization uses the PPO clipped objective,

$$
\mathcal { L } _ { \mathrm { P P O } } = \mathbb { E } \left[ \operatorname* { m i n } \left( \rho _ { t } \hat { A } _ { t } , \operatorname { c l i p } ( \rho _ { t } , 1 - \epsilon , 1 + \epsilon ) \hat { A } _ { t } \right) \right] ,\tag{9}
$$

where

$$
\rho _ { t } = \exp \left( \log \pi _ { \psi } ( a _ { t } \mid s _ { t } , z _ { t } ) - \ell _ { t } ^ { \mathrm { o l d } } \right) ,\tag{10}
$$

and $\ell _ { t } ^ { \mathrm { o l d } }$ is the behavior-policy log probability stored during rollout collection and kept fixed throughout all PPO epochs. Meanwhile, $z _ { t }$ is freshly recomputed using the current model parameters. Gradients therefore propagate through the policy head, encoder, neural memory, and cross-episode aggregator when computing the numerator. An old TTT state or context representation does not need to be reconstructed.

## B.2 MORE ON TEST-TIME TRAINING

TTT-MLP. Following Sun et al. (2025), we instantiate the inner-loop model $f _ { \mathcal { M } }$ as a residual MLP memory. Specifically, let $f _ { \mathrm { M L P } , \mathcal { M } }$ denote a two-layer MLP with parameters M and a GELU activation. The memory model is defined as

$$
f _ { \mathcal { M } } ( u ) = u + \mathrm { L N } ( f _ { \mathrm { M L P } , \mathcal { M } } ( u ) ) ,
$$

where LN denotes layer normalization. The residual connection and normalization stabilize the online TTT updates, while the MLP gives the memory more expressive capacity.

Chunk-wise Training. The step-wise TTT update is causal but difficult to parallelize, since fully online gradient descent computes each gradient using the memory produced by the previous step. To improve efficiency, TTT uses a mini-batch gradient descent approximation over chunks. Let $\mathcal { C } _ { m } = \{ x _ { m , 1 } , \hdots , x _ { m , C } \}$ denote the m-th chunk of length $C ,$ and let $\mathcal { M } _ { m , 0 }$ be the memory inherited from the previous chunk. Instead of computing the gradient at each intermediate memory state, all gradients inside the chunk are computed with respect to the same initial memory:

$$
g _ { m , c } = \nabla _ { \boldsymbol { \mathcal { M } } } \ell \left( \boldsymbol { \mathcal { M } } _ { m , 0 } ; \mathbf { k } _ { m , c } , \mathbf { v } _ { m , c } \right) , \qquad c = 1 , \ldots , C .
$$

The memory after position c is then obtained by accumulating the gradients up to that position,

$$
\mathcal { M } _ { m , c } = \mathcal { M } _ { m , 0 } - \sum _ { i = 1 } ^ { c } \eta _ { m , i } g _ { m , i } .
$$

The final memory $\mathcal { M } _ { m , C }$ is passed to the next chunk as $\mathcal { M } _ { m + 1 , 0 }$ , preserving chronological dependence across chunks.

Because all gradients in a chunk are evaluated at ${ \mathcal { M } } _ { m , 0 } ,$ they become independent once the chunk begins and can be computed in parallel. The cumulative update still preserves causal ordering within the chunk, while the chunk-level formulation substantially improves GPU utilization by replacing many sequential updates with batched operations.

## B.3 METTA ALGORITHM

Let $\Theta = \{ \theta , \phi , \psi , \nu , \zeta \}$ denote all slow outer-loop parameters of METTA. Here, φ parameterizes the input encoder $h _ { e , t } ^ { ( 0 ) } = \phi ( x _ { e , t } )$ , ψ parameterizes the policy head $\pi _ { \psi } , \nu$ parameterizes the value function $V _ { \nu }$ , and $\zeta$ párameterizes the cross-episode aggregator $\operatorname { A g g } _ { \zeta }$ . For attention-based aggregation, $\zeta$ consists of the query, key, and value projections together with the parameters of the final layer normalization. The parameters θ define the TTT module, including the view projections, innerloop learning rates, and learned initial fast-weight memories $\{ \mathcal { M } _ { 0 } ^ { ( l ) } \} _ { l = 1 } ^ { L }$ . PPO updates the slow parameters $\Theta$ in the outer loop. In contrast, the episode-specific memories $\mathcal { M } _ { e , t } ^ { ( l ) }$ are fast inner-loop states. They are initialized from $\mathcal { M } _ { 0 } ^ { ( l ) }$ at the start of each episode, updated online during sequence processing, and discarded after the episode.

Algorithm 1 Training METTA with PPO   
Require: Training tasks $p _ { \mathrm { t r a i n } } ( \mathcal { T } )$ , slow parameters $\Theta = \{ \theta , \phi , \psi , \nu , \zeta \}$   
Require: Trial length $E ,$ episode horizon $H ,$ number of environments $B$   
Require: PPO parameters $\gamma , \lambda , \epsilon ,$ number of PPO epochs $N _ { \mathrm { P P O } }$ , environment minibatch size $B _ { \mathrm { m b } }$   
1: for each training update do   
2: Sample tasks $\{ \dot { \mathcal { T } } _ { b } \} _ { b = 1 } ^ { B }$ from $p _ { \mathrm { t r a i n } } ( \tau )$   
3: Initialize episode-summary history $\mathcal { H } _ { b } \gets \emptyset$ for each environment b   
4: Reset environments and initialize previous action, reward, and done inputs to zero   
5: for $e = 1 , \ldots , E$ do   
6: Initialize within-episode TTT memories $\{ \mathcal { M } _ { b , 0 } ^ { ( l ) } \} _ { l = 1 } ^ { L }$ from learned initial memories   
7: Reset previous action, reward, and done inputs to zero   
8: for $t \overset { \vartriangle } { = } 0 , \ldots , H$ do   
9: Form the interaction input   
$x _ { b , e , t } = [ s _ { b , e , t } , a _ { b , e , t - 1 } , r _ { b , e , t - 1 } , d _ { b , e , t - 1 } ]$   
10: Embed the input $h _ { b , e , t } ^ { ( 0 ) } = \phi ( x _ { b , e , t } )$   
11: Process the current episode prefix with the TTT module   
$h _ { b , e , t } ^ { ( L ) } , \{ \mathcal { M } _ { b , t } ^ { ( l ) } \} _ { l = 1 } ^ { L } = \mathrm { T T T } _ { \theta } \left( h _ { b , e , t } ^ { ( 0 ) } , \{ \mathcal { M } _ { b , t - 1 } ^ { ( l ) } \} _ { l = 1 } ^ { L } \right)$   
12: Aggregate previous episode summaries   
$z _ { b , e , t } = \mathrm { A g g } _ { \zeta } \left( h _ { b , e , t } ^ { ( L ) } , \mathcal { H } _ { b } \right)$   
13: Compute policy and value   
$\mu _ { b , e , t } , \sigma _ { \psi } = \pi _ { \psi } ( s _ { b , e , t } , z _ { b , e , t } ) , \qquad v _ { b , e , t } = V _ { \nu } ( s _ { b , e , t } , z _ { b , e , t } )$   
14: Sample action   
$a _ { b , e , t } \sim \mathcal { N } ( \mu _ { b , e , t } , \sigma _ { \psi } ^ { 2 } I )$   
15: Step the environment and store transition data   
$( x _ { b , e , t } , z _ { b , e , t } , a _ { b , e , t } , r _ { b , e , t } , d _ { b , e , t } , \log \pi _ { \psi } ( a _ { b , e , t } \mid s _ { b , e , t } , z _ { b , e , t } ) , v _ { b , e , t } )$   
16: end for   
17: Store the final episode representation as an episode summary $\bar { h } _ { b , e } = h _ { b , e , H } ^ { ( L ) }$   
18: Update the episode-summary history $\mathcal { H } _ { b }  \mathcal { H } _ { b } \cup \{ \bar { h } _ { b , e } \}$   
19: if $\bar { \boldsymbol { e } } < E$ then   
20: Reset the environment under the same task   
21: end if   
22: end for   
23: Compute advantages $\hat { A } _ { b , e , t }$ and returns $\hat { R } _ { b , e , i }$ using GAE over the flattened trial   
24: Partition each trial b into chronological PPO segments $\{ { \mathcal G } _ { b , m } \} _ { m = 1 } ^ { M }$   
25: for $j = 1 , \ldots , N _ { \mathrm { P P O } }$ do   
26: Partition environment indices $\{ 1 , \ldots , B \}$ into minibatches $\{ \mathcal { T } _ { q } \} _ { q = 1 } ^ { Q }$ of size $B _ { \mathrm { m b } }$   
27: for $q = 1 , \ldots , Q$ do   
28: for $m = 1 , \ldots , M$ do   
29: Form segment minibatch $\mathcal { G } _ { q , m } = \{ \mathcal { G } _ { b , m } : b \in \mathcal { T } _ { q } \}$   
30: Reprocess $\mathcal { G } _ { q , m }$ with the corresponding prefixes and episode-summary histories   
31: Compute $\mathcal { L } _ { \mathrm { P P O } }$ and ${ \mathcal { L } } _ { V }$ on $\mathcal { G } _ { q , m }$   
32: Update Θ by minimizing   
$\mathcal { L } _ { \mathrm { t o t a l } } = - \mathcal { L } _ { \mathrm { P P O } } + c _ { v } \mathcal { L } _ { V }$   
33: end for   
34: end for   
35: end for   
36: end for

Algorithm 2 Meta-Test Evaluation of METTA   
Require: Test tasks $p _ { \mathrm { t e s t } } ( \mathcal { T } )$ , trained slow parameters $\Theta = \{ \theta , \phi , \psi , \nu , \zeta \}$   
Require: Evaluation trial length $E _ { \mathrm { e v a l } } .$ , episode horizon H, number of environments B   
1: Freeze slow parameters Θ   
2: Sample test tasks $\{ \mathcal { T } _ { b } \} _ { b = 1 } ^ { B }$ from $p _ { \mathrm { t e s t } } ( \tau )$   
3: Initialize episode-summary history $\mathcal { H } _ { b } \gets \emptyset$ for each environment b   
4: Reset environments and initialize previous action, reward, and done inputs to zero   
5: for $e = 1 , \ldots , E _ { \mathrm { e v a l } }$ do   
6: Initialize within-episode TTT memories $\{ \mathcal { M } _ { b , 0 } ^ { ( l ) } \} _ { l = 1 } ^ { L }$ from learned initial memories   
7: Reset previous action, reward, and done inputs to zero   
8: for $t \overset { \vartriangle } { = } 0 , \ldots , H$ do   
9: Form the interaction input   
$x _ { b , e , t } = [ s _ { b , e , t } , a _ { b , e , t - 1 } , r _ { b , e , t - 1 } , d _ { b , e , t - 1 } ]$   
10: Embed the input   
$h _ { b , e , t } ^ { ( 0 ) } = \phi ( x _ { b , e , t } )$   
11: Process the current episode prefix with the TTT module   
$h _ { b , e , t } ^ { ( L ) } , \{ \mathcal { M } _ { b , t } ^ { ( l ) } \} _ { l = 1 } ^ { L } = \mathrm { T T T } _ { \theta } \left( h _ { b , e , t } ^ { ( 0 ) } , \{ \mathcal { M } _ { b , t - 1 } ^ { ( l ) } \} _ { l = 1 } ^ { L } \right)$   
12: Aggregate previous episode summaries   
$z _ { b , e , t } = \mathrm { A g g } _ { \zeta } \left( h _ { b , e , t } ^ { ( L ) } , \mathcal { H } _ { b } \right)$   
13: Compute deterministic evaluation action   
$\mu _ { b , e , t } = \pi _ { \psi } ( s _ { b , e , t } , z _ { b , e , t } ) , \qquad a _ { b , e , t } = \mu _ { b , e , t }$   
14: Step the environment and record return and success   
15: end for   
16: Store the final episode representation as an episode summary   
$\bar { h } _ { b , e } = h _ { b , e , H } ^ { ( L ) }$   
17: Update the episode-summary history   
$\mathcal { H } _ { b }  \mathcal { H } _ { b } \cup \{ \bar { h } _ { b , e } \}$   
18: if $e < E _ { \mathrm { e v a l } }$ then   
19: Reset the environment under the same test task   
20: end if   
21: end for

![](images/87eef77caa11abeeabe313e28ec4197afbb455f9ee2781a135d106742d7eee8c.jpg)  
Figure 8: Examples of Meta-World environments illustrate the diversity of robotic manipulation tasks in the benchmark. Figures adapted from the official project page https : / /meta-world. github.io.

## C BENCHMARK DETAILS

## C.1 META-WORLD

Meta-World is a popular benchmark in the field of meta-reinforcement learning, focusing on robotic manipulation tasks, which enables evaluation in few-shot adaptation. These tasks vary in objectives, object geometries, and interaction dynamics while sharing a common robot embodiment (Sawyer arm) and action space, enabling structured transfer and meta-learning evaluation. We evaluated our algorithm, METTA, on three families of environments within this benchmark.

ML1. In ML1, each trial contains episodes from a single task family (e.g., push). Following the standard convention used by the baselines, we mask the goal coordinates from the observation, requiring the agent to infer the goal through interaction across episodes.

ML10 and ML45. ML10 and ML45 contain multiple robotic manipulation task families. Within a trial, all episodes belong to the same task family, while task variations such as object and goal configurations may change across episodes. Following the standard convention used by the baselines, goal coordinates are included in the observation for ML10 and ML45, rather than being masked as in ML1. Therefore, the persistent information across episodes primarily concerns the underlying task family and task-specific interaction structure. The same trial construction and observation convention are used for all compared methods.

## C.2 MUJOCO

We evaluate our method and baselines on a continuous-control task in the MuJoCo simulator, which is widely used in the meta-reinforcement learning literature. Specifically, we consider HalfCheetahVel, where the agent must adapt its running speed to a target velocity. In this setup, the target velocity is sampled uniformly from the range [0, 2], and the episode length is set to 100 timesteps.

## C.3 POINTMAZE

We use a PointMaze setup with a 7 × 7 maze, where the agent must navigate to a goal location. The agent and goal are initialized with a minimum shortest-path distance of 10 steps, making successful exploration require long-horizon credit assignment. The environment provides sparse rewards, with reward +1 only when the agent reaches the goal, at which point the episode terminates. Within a trial, the task remains fixed across episodes, allowing information acquired during earlier exploration to be reused in subsequent episodes. Training and test tasks differ in their start and goal configurations, with 100 training tasks and 24 held-out test tasks.

Table 2: Hyperparameters used in our experiments.
<table><tr><td rowspan="2">Parameter</td><td colspan="3">Meta-World</td><td rowspan="2">MuJoCo</td><td rowspan="2">PointMaze</td></tr><tr><td>ML1</td><td>ML10</td><td>ML45</td></tr><tr><td colspan="6">Training</td></tr><tr><td>num_envs</td><td>50</td><td>50</td><td>180</td><td>50</td><td>25</td></tr><tr><td>rollout_steps</td><td></td><td>500</td><td></td><td>100</td><td>500</td></tr><tr><td>trial_length</td><td></td><td>5</td><td></td><td>5</td><td>10</td></tr><tr><td colspan="6">TTT</td></tr><tr><td>hidden_size</td><td colspan="6"></td></tr><tr><td colspan="6">num_hidden_layers</td></tr><tr><td>mini_batch_size</td><td colspan="6">2</td></tr><tr><td>ttt_layer_type</td><td colspan="6">128 mlp</td></tr><tr><td colspan="6">PPO</td></tr><tr><td colspan="6">ppo-epochs</td></tr><tr><td>1r</td><td colspan="2">10</td><td>1 10 3e-4</td><td></td><td>5</td></tr><tr><td>ppo_envs-per_minibatch</td><td colspan="6">10</td></tr><tr><td>policy-hidden_sizes</td><td colspan="6">64,64</td></tr><tr><td>value_hidden_sizes</td><td colspan="6">64,64</td></tr><tr><td>gamma</td><td colspan="6">0.99</td></tr><tr><td>gae_lambda</td><td colspan="6">0.95 0.2</td></tr><tr><td>clip-epsilon</td><td colspan="6"></td></tr><tr><td>ent_coef</td><td colspan="6"></td></tr><tr><td>vf_coef</td><td colspan="6">0.0 0.5</td></tr><tr><td>max_std</td><td colspan="6">1.5</td></tr><tr><td>min_std</td><td colspan="6"></td></tr><tr><td>init_std</td><td colspan="6">0.5 1.0</td></tr><tr><td>Meta-Test</td><td colspan="6"></td></tr><tr><td></td><td colspan="6"></td></tr><tr><td>eval_trial_length</td><td>25</td><td></td><td></td><td>25</td><td>10</td><td></td></tr></table>

## D EXPERIMENTAL PROTOCOL AND BASELINE DETAILS

Baseline implementations and hyperparameters. For each baseline, we use the implementation and the benchmark-specific configuration recommended by the corresponding repository when available. The baseline implementations are integrated into our codebase through a common training and evaluation pipeline while retaining their default architectures and method-specific settings. Where a repository provides benchmark-specific configurations, we use the corresponding default configuration. All methods are evaluated under the same environment-interaction budget within each benchmark and using the same task splits and evaluation protocol. All reported results are computed over five independent random seeds. Learning curves show the mean across seeds, with shaded regions indicating ±1 standard error of the mean (SEM).

Computational environment. All experiments and wall-clock measurements are run on the same server equipped with an NVIDIA RTX 3090 Ti GPU. All methods are provided with a common hardware and evaluation environment for the computational comparisons while preserving the implementation choices specific to each method.

## E HYPERPARAMETERS

Table 2 reports the main hyperparameters used in our experiments. The parameter num\_envs corresponds to the number of parallel environments B. The parameter rol1out\_steps denotes the episode horizon H, while tri al\_length denotes the number of episodes E in each meta-RL trial used during rollout collection and sequential PPO optimization.

For the TTT module, hidden\_size is the hidden dimension of the TTT representations, and num\_hidden\_layers corresponds to the number of stacked TTT layers $L .$ The parameter mini\_batch\_size is the chunk length C used for chunk-wise TTT training. The parameter ttt\_layer\_type specifies the inner-loop memory model $f _ { \mathcal { M } } ;$ in our experiments, we use a twolayer MLP.

For $\mathrm { P P O , }$ ppo\_epochs is $N _ { \mathrm { P P O } }$ , and ppo\_envs-per\_minibatch denotes $B _ { \mathrm { m b } }$ , the number of environment trials used in each PPO minibatch before applying chronological segment updates. The policy and value hidden sizes specify the MLP hidden dimensions of the actor and critic heads.

## F COMPUTATIONAL COMPLEXITY AND INFERENCE EFFICIENCY

Let T denote the number of timesteps per episode, E the number of episodes in a trial, e the current episode index, and t the current timestep. We analyze scaling with interaction-history length while treating hidden dimensions and network depths as constants. Existing complexity analyses for sequence-based meta-RL typically characterize the cost of constructing a context representation at a single decision point rather than the computation accumulated throughout an online trial. For example, ECET (Shala et al., 2025) reports $\mathcal { O } ( T ^ { 2 } + E ^ { 2 } )$ for its hierarchical attention, corresponding to one context construction when the episode-level transformer evaluations are treated as parallel operations. In online meta-RL, however, action selection occurs at every timestep, so the full-trial complexity naturally accumulates this computation across all timesteps and episodes. We therefore report both the per-decision cost and the total computation over the complete trial.

TrMRL. TrMRL (Melo, 2022) applies timestep-level self-attention over the accumulated interaction context. At timestep t of episode $e ,$ the context contains approximately $( e - 1 ) T + t$ transitions, giving a per-decision cost of $\bar { \mathcal { O } } ( ( e - 1 ) T + t )$ . Summing this cost over the full trial gives

$$
\sum _ { \ell = 1 } ^ { E T } \mathcal { O } ( \ell ) = \mathcal { O } ( E ^ { 2 } T ^ { 2 } ) .
$$

The stored timestep-level history grows as $\mathcal { O } ( E T )$

ECET. At episode e, ECET has a per-decision computation of $\mathcal { O } ( e T ^ { 2 } + e ^ { 2 } )$ . Since this computation is performed at each of the T timesteps, the total cost for episode e is

$$
\mathcal { O } ( e T ^ { 3 } + e ^ { 2 } T ) .
$$

Summing over all $E$ episodes gives

$$
\sum _ { e = 1 } ^ { E } \mathcal { O } ( e T ^ { 3 } + e ^ { 2 } T ) = \mathcal { O } ( E ^ { 2 } T ^ { 3 } + E ^ { 3 } T ) .
$$

When the computations across the e episodes are parallelized, the effective per-decision time reduces to $\mathcal { O } ( T ^ { 2 } + e ^ { 2 } )$ , resulting in $\mathcal { O } ( T ^ { 3 } + \dot { e } ^ { 2 } T )$ per episode and

$$
\sum _ { e = 1 } ^ { E } \mathcal { O } ( T ^ { 3 } + e ^ { 2 } T ) = \mathcal { O } ( E T ^ { 3 } + E ^ { 3 } T )
$$

over the full trial. The stored interaction history scales as $\mathcal { O } ( E T )$ . Since practical memory usage also depends on the implementation and parallelization strategy, we report measured memory usage below.

METTA. The within-episode TTT memory has a fixed size with respect to the accumulated interaction history, giving $\mathcal { O } ( 1 )$ history-dependent computation per timestep. With attention-based crossepisode aggregation, the current representation attends only to the $e { - } 1$ previous episode summaries, giving $\mathcal { O } ( e )$ additional computation. The computation over episode e is therefore $\mathcal { O } ( T + e T )$ , and over the full trial

$$
\sum _ { e = 1 } ^ { E } \mathcal { O } ( T + e T ) = \mathcal { O } ( E ^ { 2 } T ) .
$$

Table 3: Online inference complexity with respect to episode horizon T and number of episodes E. Network dimensions and layer counts are treated as constants.
<table><tr><td>Method</td><td>Per-decision computation Full-trial computation</td><td>1</td><td>History memory</td></tr><tr><td>TrMRL</td><td> $\mathcal { O } ( ( e - 1 ) T + t )$ </td><td> $\mathcal { O } ( E ^ { 2 } T ^ { 2 } )$ </td><td>O(ET)</td></tr><tr><td>ECET</td><td> $\mathcal { O } ( T ^ { 2 } + e ^ { 2 } )$ </td><td> $\mathcal { O } ( E T ^ { 3 } + \dot { E ^ { 3 } } T )$ </td><td>O(ET)</td></tr><tr><td> $\mathrm { M E T T A } + \mathrm { A t t e n t i o n }$ </td><td> $\mathcal { O } ( 1 + e )$ </td><td> $\mathcal { O } ( E ^ { 2 } T )$ </td><td>O(E)</td></tr><tr><td> $\mathbf { M E T T A } + \mathbf { E M A }$ </td><td>O(1)</td><td> $\mathcal { O } ( E T )$ </td><td>0(1)</td></tr></table>

Table 4: Measured per-step inference latency and memory usage as context length increases. TTT-MLP and TTT-Linear include the inner updates performed during inference. Lower values are better (↓).
<table><tr><td></td><td colspan="2">METTA</td><td colspan="4">TrMRL</td><td colspan="4">ECET (E = 5)</td><td colspan="4">ECET (E = 25)</td></tr><tr><td>Metric</td><td>TTT-MLP</td><td>TTT-Linear</td><td> $\overline { { T = 5 } }$ </td><td> $\overline { { T = 2 5 } }$ </td><td>T = 100</td><td> $\overline { { T = 5 0 0 } }$ </td><td> $\overline { { T = 5 } }$ </td><td>T = 25</td><td>T = 100</td><td> $\overline { { T = 5 0 0 } }$ </td><td>T = 5</td><td>T = 25</td><td>T = 100</td><td> $\overline { { T = 5 0 0 } }$ </td></tr><tr><td>Time (ms/step) ↓</td><td>2.159</td><td>1.616</td><td>0.551</td><td>0.639</td><td>1.061</td><td>9.223</td><td>0.761</td><td>0.991</td><td>2.119</td><td>21.713</td><td>1.035</td><td>2.151</td><td>7.599</td><td>105.403</td></tr><tr><td>Memory (MiB) ↓</td><td>5.034</td><td>2.397</td><td>0.335</td><td>0.356</td><td>0.570</td><td>5.333</td><td>0.792</td><td>0.863</td><td>1.701</td><td>21.388</td><td>2.534</td><td>2.857</td><td>6.813</td><td>101.115</td></tr></table>

Only episode summaries are retained, requiring ${ \mathcal { O } } ( E )$ history-dependent memory. With EMA aggregation, both within-episode and cross-episode memory remain constant with respect to history length, giving $\mathcal { O } ( E T )$ computation over the trial and O(1) history-dependent memory.

Since typically $E \ll T$ , the dependence on timestep-level history becomes particularly important in long-horizon settings. In addition to the asymptotic analysis, we directly measure per-step inference latency and memory usage for both TTT variants and the transformer-based baselines, providing an empirical comparison of their practical inference overhead.

Table 4 supports the scaling trends in Table 3. At short context lengths, TrMRL and ECET have lower latency because TTT performs an inner update at every timestep. As T increases, however, their inference time and memory grow substantially, while the TTT cost remains independent of the accumulated timestep-level history. At $T = 5 0 0$ , TTT-MLP requires 2.16 ms per step, providing 4.3× lower latency than TrMRL and 10.1×/48.8× lower latency than ECET with $E = 5 / 2 5$ . TTT-MLP also uses 5.03 MiB of memory, over 20× less than ECET with $E = 2 5$ . The lighter TTT-Linear variant further reduces the cost to 1.62 ms per step and 2.40 MiB, reaching 65.2× lower latency and 42.2× lower memory than ECET with $E = 2 5 \mathrm { a t } T = 5 0 0$ . Overall, TTT has a higher constant cost at short contexts but scales substantially better as the interaction history grows.

## G ADDITIONAL RESULTS

## G.1 ALTERNATIVE NEURAL MEMORIES

To examine the flexibility of the within-episode memory component, we integrate METTA with several alternative sequence architectures. Specifically, we replace the TTT memory with a GRU, Mamba (Gu & Dao, 2023), and a linear Transformer (Katharopoulos et al., 2020), while keeping the cross-episode aggregation mechanism, policy and value networks, and training procedure unchanged.

Figure 9 shows that the METTA framework can operate with different within-episode sequence models, while their adaptation behavior differs substantially. On PointMaze, TTT achieves the strongest and most consistent test performance, while the linear Transformer approaches it later in training and GRU and Mamba remain lower. The differences are also clear on ML10 test tasks: TTT and the linear Transformer achieve substantially stronger generalization than GRU and Mamba, despite all variants learning the training tasks effectively. These results show that the overall framework is not tied to a particular sequence architecture, while the choice of within-episode memory remains important for test-time adaptation and generalization.

![](images/1815e3d38aabf5451e58b34406c3eafa9a412bda947aafa4eff1dd66d0fa6f74.jpg)  
METTA  METTA (GRU) — METTA (Mamba-1)  METTA (linear Transformer)

Figure 9: Comparison of alternative within-episode neural memory architectures in METTA on ML10 and PointMaze.  
![](images/cb393ca921852e3a7afadb7369e6cefd298155cdb1abc011e7144bc43bc1466c.jpg)  
Figure 10: Comparison of cross-episode aggregation mechanisms in METTA on ML10 and Point-Maze. All variants use the same within-episode neural memory.

## G.2 CROSS-EPISODE AGGREGATION

We next vary the mechanism used to combine information across episodes while keeping the withinepisode neural memory and the rest of METTA unchanged. We compare the default softmax attention with no cross-episode aggregation, an exponential moving average (EMA) with $\alpha = 1 / 2$ , and linear attention.

![](images/3429034f1f383e740fc22dc11205602ac87ef8ab3eb407c8aebceb13cabe6b18.jpg)

![](images/3151662aa01964cc02e53aa192c589e33b29471d29a3aa97e57cc18cff12ddc7.jpg)

![](images/08e0734222261c995e7acc908b48c59e3e8ebcdddcd170e45f7e89e119c7b17f.jpg)

![](images/03208460bab71eeb62f95e98658d79a1507e86a4aa1a0b2145a8b3942a05f209.jpg)

![](images/47c64eb2dde30419b2d32e593c5dd569e823b7079a56eecf0cd4b74a6bc2be43.jpg)

![](images/32279b4f0c8c0821a4929c5c9d00c06dff31c260aa96ff083c390d21ab2d4536.jpg)

Figure 11: Complete online memory-update ablation on PointMaze, ML10, and ML1 Push. Top: training performance. Bottom: test performance.  
![](images/8f15077bae72193cee98182529c79864cd62ec10551c83e9bbec8cae273af65e.jpg)

![](images/c6fd6ad0018996475ee096dabca8e5780928de1a60491cc1c145b0b8572448c8.jpg)

![](images/c97bb2340e0e66c1fad39264fe57c8f92da27f2b767efe5b16010120982aabc4.jpg)  
Figure 12: ML1 Push test performance at adaptation Episodes 1, 5, and 25 within each evaluation trial. The episode index denotes the progression of adaptation within the same trial.

Figure 10 shows that the aggregation mechanism has relatively little effect on training performance but becomes substantially more important at test time. On ML10, removing cross-episode aggregation leads to consistently lower test success, remaining around 0.3, while softmax attention, EMA $( \alpha = 1 / 2 )$ , and linear attention all benefit from reusing information across episodes. On Point-Maze, the no-aggregation variant learns much more slowly at test time and remains near zero during approximately the first 0.5M environment steps, whereas the aggregation-based variants begin improving substantially earlier. This further highlights the importance of reusing evidence collected in previous episodes for adaptation on held-out tasks.

Among the aggregation mechanisms, softmax attention provides the strongest overall trade-off across the four settings. EMA performs competitively on several test settings but is less stable on PointMaze and somewhat weaker on ML10 training tasks. Linear attention closely tracks softmax attention during training and reaches strong PointMaze test performance, but is less consistent on held-out tasks. These results support retaining a selective cross-episode retrieval mechanism while showing that METTA is not restricted to a single aggregation design.

## G.3 ADDITIONAL ONLINE MEMORY UPDATE RESULTS

Section 4.3 studies the contribution of METTA's online memory adaptation. Here, we provide the complete training and test curves for this ablation. In all cases, the no-update variant uses the same architecture and cross-episode aggregation as METTA, but disables the online memory updates.

Figures 11 and 12 provide the complete results corresponding to the comparison in Section 4.3. The large difference on PointMaze appears in both training and test performance, with a substantially larger gap on held-out test tasks. On ML10, the two variants remain close throughout training and testing. On ML1 Push, both variants learn the training task effectively, while the online-update model achieves stronger test performance.

![](images/ec1158854cead353383f859ab803a4e719cf1383c5fa5d3e5ff50b671ea5ed50.jpg)  
Figure 13: Effect of within-episode memory capacity. We compare the default TTT-MLP memory with the lighter TTT-Linear variant on PointMaze, ML45, and ML10.

## G.4 MEMORY CAPACITY

We study the effect of the capacity of the within-episode memory by comparing the default TTT-MLP with the lighter TTT-Linear variant. TTT-MLP uses a two-layer MLP as the adaptive neural memory, whereas TTT-Linear uses a single linear layer. Both variants otherwise use the same METTA architecture, cross-episode aggregation, policy and value networks, and training procedure.

Figure 13 shows that both memory variants learn the training tasks effectively, while the highercapacity memory provides stronger test-time adaptation. The difference is largest on PointMaze, where TTT-MLP substantially outperforms TTT-Linear on held-out tasks despite similar training success. TTT-MLP also achieves stronger test performance on ML10 and ML45, although the gap is smaller. These results indicate that increasing the capacity of the adaptive memory is particularly useful when task inference requires integrating richer interaction evidence, while the lighter linear memory remains a viable lower-cost alternative.

## G.5 SEPARATING STATE AND TASK CONTEXT

In the default METTA architecture, the learned context representation $z _ { e , t }$ summarizes task-relevant interaction history, while the current state $s _ { e , t }$ is provided separately to the actor and critic. We ablate this design by removing the separate state input, forcing the learned representation to jointly encode both the current state and task context.

Figure 14 shows that explicitly separating state information from task context has relatively little effect on training performance but improves generalization to held-out tasks. The difference is particularly pronounced on PointMaze, where removing the separate state input causes a large reduction in test success despite maintaining strong training performance. The same trend is visible on ML10, while the two variants are closer on ML45. These results suggest that allowing the memory to focus on task-relevant evidence, while providing the instantaneous state directly to the policy and value function, produces a more effective representation for adaptation than requiring a single learned representation to encode both.

## G.6 QUALITATIVE POINTMAZE TRAJECTORIES

Figure 15 shows PointMaze examples for METTA, ECET, and TrMRL at two stages of learning: halfway through training and at the end of training. For each method, we visualize trajectories from Episodes 1, 2, 5, and 10 on the same task instances to illustrate how information is used across episodes within a trial.

A clear pattern is that, for ECET and TrMRL, trajectories from different episodes often overlap substantially. This suggests that these agents do not effectively use information gathered in previous episodes, and instead tend to repeat similar exploration patterns across the trial. Even when these methods eventually reach the goal, they frequently revisit previously explored paths rather than refining their search based on earlier failures or successes. In TrMRL, this behavior is consistent with limited effective long-context memory across episodes. For ECET, although it explicitly incorporates cross-episode context, the available context is short, which makes it difficult to use previous episode information effectively in this very low-signal sparse-reward environment. Moreover, this behavior remains visible from mid-training to the end of training, indicating that these methods do not substantially improve their ability to reuse previous episodes over the course of learning.

![](images/4dda5acaf07464239862026ad2c8ef0125411b84a90736a16ecf213dec15807b.jpg)  
Figure 14: Effect of separating the current state from the learned task context. The default METTA provides $s _ { e , t }$ separately to the actor and critic, while the ablated variant must represent both state and task information through the learned context.

In contrast, METTA learns a qualitatively different exploration strategy. In particular, it learns to search the maze more broadly in the first episode, and the search space covered by this first episode becomes noticeably larger as training progresses. Later episodes then make better use of the information gathered during this broader initial exploration, leading to more directed behavior toward the goal. These examples, together with the exploration measures in Table 1, support our claim that episodic neural memory helps preserve and reuse task-relevant evidence across episodes, which is especially important in long-context sparse-reward settings such as PointMaze.

## G.7 POINTMAZE EXPLORATION METRICS

We characterize PointMaze exploration using three measures reported in Table 1. These measures are computed from the trajectories generated across episodes of each evaluation trial and then averaged across seeds.

Coverage. Coverage measures the fraction of traversable maze cells visited by the agent before success or episode termination. Higher coverage indicates that the agent explores a larger portion of the available maze rather than repeatedly visiting the same regions.

Episode Jaccard. For each pair of consecutive episodes, we compute the Jaccard similarity between the sets of maze cells visited in the two episodes after excluding the forced starting cell. For consecutive episodes e and $e + 1$ with visited-cell sets $\mathcal { C } _ { e }$ and $\mathcal { C } _ { e + 1 }$ , the similarity is

$$
J ( \mathcal C _ { e } , \mathcal C _ { e + 1 } ) = \frac { | \mathcal C _ { e } \cap \mathcal C _ { e + 1 } | } { | \mathcal C _ { e } \cup \mathcal C _ { e + 1 } | } .
$$

Lower values indicate that consecutive episodes overlap less and therefore exhibit less repetitive exploration.

Steps. Steps denotes the number of environment interactions before an episode reaches the goal or terminates at the horizon. When considered together with success performance and maze coverage, fewer steps indicate that broader exploration is achieved without requiring longer trajectories.

![](images/b1099031e949cbfd442bd0ad2f352f9779531c63bb973cc39f52e70fa7e240b2.jpg)  
Figure 15: Qualitative PointMaze trajectories for METTA, ECET, and TrMRL. Columns correspond to methods, while rows show 9 task instances at mid-training and at the end of training. Colored curves denote Episodes 1, 2, 5, and 10 from the same trial. ECET and TrMRL frequently revisit similar regions across episodes, whereas METTA covers a wider portion of the maze and progressively produces more goal-directed behavior.