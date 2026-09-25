# NOT EVERY TOKEN IS WORTH DISTILLING: SELECTIVE SUPERVISION FOR DIRECT-OPD

Yibo Zhao<sup>∗</sup>, Zixuan Yang<sup>∗</sup>, Yunshi Lan, Xiang Li<sup>†</sup> School of Data Science and Engineering East China Normal University

Hugging Face Github

## ABSTRACT

Direct On-Policy Distillation (Direct-OPD) transfers reinforcement-learninginduced policy improvements from a small model to a larger student by using the token-level log-ratio between post-RL and pre-RL checkpoints as dense supervision on the student’s own rollouts. This transfer rewards the policy shift at every state, yet the log-ratio measures only relative change: it can stay fixed even as the probability mass that both checkpoints assign to the student’s candidate tokens vanishes. Through an exact construction, we show that the Direct-OPD reward and its update can remain unchanged while the Jensen–Shannon divergence (JSD) and both KL directions between the checkpoints vanish with this mass, and we note that a small JSD bounds how much the teacher’s behavior changed. Motivated by this analysis, we propose Selective Supervision for Direct-OPD (S<sup>2</sup>D-OPD), which ranks student-sampled states by their teacher–reference JSD and masks Direct-OPD supervision at low-divergence states, retaining only the top 10% of states per response. Across two teacher pairs and four student models ranging from 1.7B to 8B parameters, S<sup>2</sup>D-OPD improves held-out accuracy over dense Direct-OPD on AIME and HMMT benchmarks in seven of eight settings and matches it in the eighth, without extra forward passes. Our code is available at https://anonymous.4open.science/r/S2D-OPD-8868.

“No Free Lunch for Supervised Machine Learning.” — David H. Wolpert

## 1 INTRODUCTION

Guided by scaling laws (Kaplan et al., 2020; Hoffmann et al., 2022; Pearce & Song, 2024), recent large language models have continued to scale up pre-training (DeepSeek-AI et al., 2026; Team et al., 2026; GLM-5-Team et al., 2026), which broadens their knowledge and latent capabilities. Post-training, most prominently reinforcement learning (RL) (Yue et al., 2025; Shao et al., 2024; Zhao et al., 2025), is then needed to elicit and refine these capabilities. As model size grows, RL demands more rollout generation, training compute, memory, and infrastructure (Wu et al., 2025), and stable optimization remains difficult to achieve (Wang et al., 2026a). Consequently, the models with the greatest post-training potential are also the most expensive to improve through RL.

One line of work makes large-scale RL more stable through improved training algorithms (Yu et al., 2025; Zheng et al., 2025; MiniMax et al., 2025; Ma et al., 2025; Hou et al., 2026) and more efficient through training and inference systems (Sheng et al., 2025; Fu et al., 2025; Kwon et al., 2023; Narayanan et al., 2021; Zhu et al., 2025). These advances make RL on large models more practical, but its cost still grows with model size. Another line of work uses on-policy distillation (OPD) (Agarwal et al., 2024; Gu et al., 2024; Lu & Lab, 2025), in which a stronger teacher provides token-level supervision on states sampled by the student. OPD transfers capabilities efficiently to a smaller student (Li et al., 2026b; Fu et al., 2026c), but it relies on a teacher stronger than the student, which is unavailable when the target is already the strongest model. Together, these approaches leave a challenge open: how to improve a large model without paying the cost of RL at its scale.

Direct-OPD (Feng et al., 2026) and Proxy-OPD (Fu et al., 2026a) recently proposed a weak-tostrong route around this challenge: running RL on a small model and transferring the result to a larger one. Both methods extract the RL-induced policy shift as the token-level log-ratio between the post-RL teacher and its pre-RL reference, and use it as an OPD reward on states sampled by the larger student. This turns outcome-level rewards into dense token-level supervision for the large model, without running RL at its scale or requiring a teacher stronger than the student.

However, this apparent free lunch leaves unexamined whether every token-level reward reflects a meaningful change in the teacher’s behavior. Because the log-ratio reward measures only relative change, it can stay fixed even as the probability mass that both checkpoints assign to the student’s top candidates vanishes. Direct-OPD can therefore reward a state where both checkpoints barely support these candidates as strongly as one where RL clearly changed the teacher’s behavior. This raises the question that motivates this work: is every state’s policy shift worth distillingforfree?

We make this probability-mass mismatch exact in Sec. 4.1: as the mass that both checkpoints assign to the student’s top candidates vanishes, the Direct-OPD reward and gradient can stay fixed, whereas the Jensen–Shannon divergence (JSD) and both directions of KL divergence between the checkpoints vanish with it. Because JSD accounts for the probability mass that the log-ratio ignores, we propose Selective Supervision for Direct-OPD (S<sup>2</sup>D-OPD), which ranks student-sampled states by teacher–reference JSD and masks Direct-OPD supervision at low-divergence states. Empirically, the answer to our question is no: retaining only the top 10% of states per response, S<sup>2</sup>D-OPD improves held-out accuracy over dense Direct-OPD in seven of eight teacher–student settings and matches it in the eighth (mean gain 0.95 points; 95% CI 0.40–1.54), without extra forward passes.

In summary, our contributions are threefold:

• A Probability-Mass Mismatch in Direct-OPD. Through an exact construction, we show that the log-ratio reward and its local gradient on the student can remain fixed while the probability mass behind the policy shift vanishes, and with it the teacher–reference JSD and both KL directions. We further show that JSD bounds how much the teacher’s behavior can change at every state, which motivates selecting states by divergence rather than by the reward itself.

• Stable and Effective Selective Transfer. We propose S<sup>2</sup>D-OPD, which masks Direct-OPD supervision at low-divergence states during policy transfer. Across four student scales and two teacher pairs, it improves held-out accuracy over Direct-OPD in seven of eight settings and yields smoother late-stage validation curves under the JustRL teacher pair.

• Understanding the Gains from Selective Transfer. Within a fixed teacher–student setting, performance broadly rises with JSD percentile: the top bin outperforms a uniformly sampled 10% subset, whereas the lowest bin degrades the student below its initialization and eventually collapses. Across teacher pairs, the pair with lower overall JSD benefits more from masking, consistent with low-divergence filtering being a source of the improvement over dense Direct-OPD.

## 2 RELATED WORK

On-Policy Distillation. OPD trains a student on prefixes sampled from its own policy, using the teacher’s next-token distributions as dense supervision at every position (Agarwal et al., 2024; Gu et al., 2024; Lu & Lab, 2025). Subsequent work refines it through alternative objectives (Jin et al., 2026; Jia et al., 2026), stabilization strategies (Li et al., 2026b; Fu et al., 2026b), and privilegedcontext self-distillation (Zhao et al., 2026; Pan et al., 2026). Despite their differences, these methods primarily learn from the teacher’s policy itself, limiting transfer to improvements already present in the teacher; recent work therefore targets the teacher’s policy shift instead. ExOPD (Yang et al., 2026) extrapolates the teacher’s improvement over its reference model to construct a target beyond the teacher. Direct-OPD (Feng et al., 2026) and Proxy-OPD (Fu et al., 2026a) instead transfer the log-ratio between a reward-optimized checkpoint and its pre-RL reference, analogous to the logit shifts induced by fine-tuning studied in CMC (Wu et al., 2024). This targets the RL-induced policy shift and can provide useful supervision even for students already stronger than the post-RL teacher. However, token-level log-ratios capture relative changes but are insensitive to the absolute probability mass supporting these changes. We examine this probability-mass mismatch in Direct-OPD and use teacher–reference divergence to select supervision positions.

Token Selection in Policy Distillation. Selective distillation asks which positions are worth training on, and existing criteria differ mainly in which distributions they read. Some read the student alone, prioritizing positions where it is uncertain (Tavor et al., 2026; Ko et al., 2026); others the teacher alone, weighting by its confidence or local margin (Jin et al., 2026; Zhou et al., 2026); a third group compares the two, emphasizing teacher–student disagreement, which TIP (Xu et al., 2026) organizes through an entropy–divergence taxonomy and TA-OPD (Wang et al., 2026b) restricts to the student’s predictive support. We consider the policy change from a pre-RL reference to a post-RL teacher at student-visited prefixes. A related approach, $\mathrm { O } \mathrm { \bar { P } D ^ { 2 } }$ (Heo et al., 2026), gates sampled-token delta updates by sign agreement between the centered teacher–base and teacher–student log-ratios. Our selection criterion instead ranks positions by teacher–reference JSD, accounting for the probability mass underlying the policy shift while retaining the Direct-OPD update at selected positions.

## 3 PRELIMINARIES

Setting. We consider three policies: a pre-RL reference $\pi _ { \mathrm { r e f } }$ , a post-RL teacher $\pi _ { \mathrm { T } }$ obtained from $\pi _ { \mathrm { r e f } }$ by outcome-based RL such as GRPO (Shao et al., 2024), and a larger student $\pi _ { \theta }$ initialized at $\pi _ { \mathrm { s t u } } .$ . In our experiments, both teacher-side checkpoints are publicly released (Sec. 5.1), so we run no RL ourselves. Given a prompt x $\sim \mathcal { D }$ and a response $\pmb { y } = ( y _ { 1 } , \dots , y _ { | \pmb { y } | } )$ sampled from the student, position t has state $\pmb { s } _ { t } = ( \pmb { x } , \pmb { y } _ { < t } )$

Direct-OPD treats the teacher’s RL-induced policy shift as a dense reward for the student. For any token $v ,$ the reward at state $\mathbf { \boldsymbol { s } } _ { t }$ is the teacher–reference log-ratio

$$
\Delta _ { t } ( v \mid s _ { t } ) = \log { \frac { \pi _ { \mathrm { T } } ( v \mid s _ { t } ) } { \pi _ { \mathrm { r e f } } ( v \mid s _ { t } ) } } ,\tag{1}
$$

which is positive where RL increased the probability of v and negative where it decreased it.

Direct-OPD maximizes this reward on states visited by the student, with KL regularization:

$$
J _ { \mathrm { D i r e c t - O P D } } ( \theta ) = \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } , ( \boldsymbol { s } _ { t } , \boldsymbol { y } _ { t } ) \sim \pi _ { \theta } } [ \Delta _ { t } ( \boldsymbol { y } _ { t } \mid \boldsymbol { s } _ { t } ) - \alpha D _ { \mathrm { K L } } [ \pi _ { \theta } ( \cdot \mid \boldsymbol { s } _ { t } ) ] \mid \pi _ { \mathrm { s t u } } ( \cdot \mid \boldsymbol { s } _ { t } ) ] ] .\tag{2}
$$

Here, $\alpha > 0$ controls KL regularization, and the expectation covers all valid response positions.

Top-K implementation. In practice, Direct-OPD evaluates the reward on the student’s top-K candidates at each state rather than only on the sampled token:

$$
\mathscr { V } _ { K } ( s _ { t } ) = \mathrm { T o p K } ( \pi _ { \theta } ( \cdot  { | } s _ { t } ) , K ) , \quad \bar { p } _ { t } ( v ) = \frac { \pi _ { \theta } ( v  { | } s _ { t } ) } { \sum _ { u \in \mathscr { V } _ { K } ( s _ { t } ) } \pi _ { \theta } ( u  { | } s _ { t } ) } ,\tag{3}
$$

where $\mathrm { T o p K } ( \cdot , K )$ returns the set of K tokens with the largest probabilities. Each candidate receives the teacher–reference reward $\Delta _ { t } ( v \mid s _ { t } )$ , weighted by its renormalized student probability $\bar { p } _ { t } ( v )$ With the state and candidate set held fixed, the implemented local reward-gradient contribution is:

$$
g _ { t } ^ { R } = \sum _ { v \in \mathcal { V } _ { K } ( s _ { t } ) } \operatorname { s g } \left[ \bar { p } _ { t } ( v ) \log \frac { \pi _ { \mathrm { T } } ( v \mid s _ { t } ) } { \pi _ { \mathrm { r e f } } ( v \mid s _ { t } ) } \right] \nabla _ { \theta } \log \pi _ { \theta } ( v \mid s _ { t } ) ,\tag{4}
$$

where sg denotes stop-gradient. The student-anchor KL term supplies a separate regularization gradient. Appendix A describes its implementation and adaptive coefficient.

## 4 METHOD

Direct-OPD applies its log-ratio reward at every valid position of a student response. Sec. 4.1 shows that this reward can ignore the probability mass behind the teacher’s policy shift, and Sec. 4.2 introduces $\mathrm { S ^ { 2 } D \mathrm { - } O P D }$ , which selects states by teacher–reference divergence.

## 4.1 THEORETICAL MOTIVATION: PROBABILITY-MASS MISMATCH

At each state, Direct-OPD weights the teacher–reference rewards $\Delta _ { t } ( v \mid s _ { t } )$ by the student’s renormalized probabilities $\bar { p } _ { t } ( v )$ over its top-K candidates in Eq. 4. These weights reflect the student’s preferences, whereas the rewards encode relative changes in the teacher’s policy. Neither depends on how much probability mass the teacher and reference place on these candidates: rescaling both by a common factor leaves every log-ratio unchanged. The following exact construction makes this precise: the teacher–reference divergence can vanish while the Direct-OPD update stays fixed.

An exact construction. Fix a state s and a student checkpoint $\theta _ { 0 }$ with top-K candidate set $\mathcal { V } _ { K } = \mathcal { V } _ { K } ( \pmb { s } )$ . Let $\mathbf t \neq \mathbf q$ be strictly positive probability vectors on $\nu _ { K }$ , let b be a strictly positive probability vector on its complement, and for $0 < \epsilon < 1$ define the teacher and reference:

$$
P _ { \epsilon } ( v ) = \left\{ \begin{array} { l l } { \epsilon t _ { v } , } & { v \in \mathcal { V } _ { K } , } \\ { ( 1 - \epsilon ) b _ { v } , } & { v \notin \mathcal { V } _ { K } , } \end{array} \right. \quad \quad Q _ { \epsilon } ( v ) = \left\{ \begin{array} { l l } { \epsilon q _ { v } , } & { v \in \mathcal { V } _ { K } , } \\ { ( 1 - \epsilon ) b _ { v } , } & { v \notin \mathcal { V } _ { K } . } \end{array} \right.\tag{5}
$$

Here, ϵ is the probability mass that each checkpoint assigns to the student’s candidates, while the student, candidate set, and conditional distributions stay fixed. Then, for every candidate $v \in \mathcal { V } _ { K }$

$$
\Delta _ { \epsilon } ( v \mid s ) = \log \frac { t _ { v } } { q _ { v } } , \qquad D _ { \mathrm { J S } } ( P _ { \epsilon } , Q _ { \epsilon } ) = \epsilon D _ { \mathrm { J S } } ( \mathbf { t } , \mathbf { q } ) ,\tag{6}
$$

and both directions of KL scale with ϵ in the same way. $\mathbf { A s } \ \epsilon \ \to \ 0 ,$ all three divergences vanish, whereas every candidate reward, and hence the Direct-OPD update of Eq. 4, stays fixed and nonzero. The construction thus isolates a single degree of freedom, the probability mass behind the policy shift, to which the Direct-OPD update is insensitive but the divergences are not. We state this result formally in Prop. 1 and prove it in App. B, including why the update is nonzero.

Divergence bounds the behavioral change. The construction shows that the Direct-OPD reward can ignore divergence; conversely, divergence bounds how much the teacher’s behavior can change. For any two distributions P and Q on a finite set and any event A,

$$
| P ( A ) - Q ( A ) | \leq D _ { \mathrm { T V } } ( P , Q ) \leq \sqrt { 2 D _ { \mathrm { J S } } ( P , Q ) } ,\tag{7}
$$

where $D _ { \mathrm { T V } } ( P , Q ) \ = \ \operatorname * { m a x } _ { A } | P ( A ) - Q ( A ) |$ is the total variation distance and $D _ { \mathrm { J S } }$ is measured in nats. The second inequality follows from Pinsker’s inequality applied to each term of $D _ { \mathrm { J S } } ( P , Q ) \ = \ { \textstyle { \frac { 1 } { 2 } } } D _ { \mathrm { K L } } ( P \| M ) \ { \stackrel { . } { + } } \ { \textstyle { \frac { 1 } { 2 } } } { \dot { D _ { \mathrm { K L } } } } ( Q \| M )$ with $M \ = \ ( P \overset { \cdot } { + } \ Q ) \overset { \cdot } { / } 2$ , since $D _ { \mathrm { T V } } ( P , M ) ~ =$ $\begin{array} { r } { D _ { \mathrm { T V } } ( Q , M ) = \frac { 1 } { 2 } D _ { \mathrm { T V } } ( P , Q ) } \end{array}$ . Unlike the construction, this bound holds at every state: wherever the teacher–reference JSD is small, the teacher assigns nearly the same probability as the reference to every token and every set of tokens. Together, the two results characterize what low-divergence masking removes: states at which the teacher’s behavior provably changed little, yet at which the Direct-OPD update can be as large as anywhere else. They do not show that removing these states improves transfer, which Sec. 5.3 tests by training on JSD percentile bins.

## 4.2 DIVERGENCE-GUIDED STATE SELECTION

Sec. 4.1 shows that the Direct-OPD reward is insensitive to the probability mass behind a policy shift, whereas the teacher–reference JSD bounds how much the teacher’s behavior changed at a state. $\mathrm { S ^ { 2 } D \mathrm { - } O P D }$ therefore scores each student-sampled state by this divergence and retains Direct-OPD supervision only at the highest-scoring states within each response (Fig. 1). We use JSD as the default score because, unlike KL, it is symmetric, bounded by log 2, and finite when either checkpoint assigns a token zero probability; the procedure applies unchanged to other divergences.

Scoring on the student’s candidates. Rather than the full vocabulary, we evaluate the divergence on the top-K candidate set $\nu _ { K } ( \pmb { s } _ { t } )$ , where Direct-OPD already evaluates both checkpoints, and add a residual token $v _ { \mathrm { o } }$ that collects all remaining probability mass. For $M \in \{ \mathrm { T } , \mathrm { r e f } \}$ , define

$$
\widetilde { \pi } _ { M } ( v \mid s _ { t } ) = \left\{ 1 - \sum _ { u \in \mathcal { V } _ { K } ( s _ { t } ) } \pi _ { M } ( u \mid s _ { t } ) , \quad v = \mathcal { V } _ { K } ( s _ { t } ) , \right.\tag{8}
$$

This keeps each candidate’s probability and the total residual mass without renormalization. By the data-processing inequality, the coarsening cannot increase JSD:

$$
\tilde { d } _ { t } \triangleq D _ { \mathrm { J S } } \big ( \widetilde { \pi } _ { \mathrm { T } } ( \cdot  { \lvert } s _ { t } )  { \lvert } | \widetilde { \pi } _ { \mathrm { r e f } } \big ( \cdot  { \lvert } s _ { t } ) \big ) \leq D _ { \mathrm { J S } } \big ( \pi _ { \mathrm { T } } \big ( \cdot  { \lvert } s _ { t } \big )  { \lvert } | \pi _ { \mathrm { r e f } } \big ( \cdot  { \lvert } s _ { t } \big ) \big ) .\tag{9}
$$

Fig. 1 illustrates two ways in which the score can be small. Either both checkpoints place little mass on the student’s candidates, so that both compressed distributions concentrate on $v _ { \mathrm { o } } ,$ which is the regime of the construction in Sec. 4.1; or both place substantial mass on the candidates and agree on how it is distributed. The score treats these cases alike, and it need not separate them: because Eq. 7 holds for any pair of distributions, including the compressed ones, a small score implies in either case that the teacher barely changed its behavior on the student’s candidates. Conversely, the score is large only when the checkpoints disagree on the individual candidate probabilities or on the total mass assigned to the candidate set. The score thus measures behavioral change at the resolution of the student’s candidates: differences among tail tokens outside the candidate set do not affect it.

![](images/a79addb1bf75c8a6d5810ce12807c1b01608b8ffdecee7b53eab592094d3f05e.jpg)  
Figure 1: Overview of $\mathrm { S ^ { 2 } D \mathrm { - } O P D }$ . At each student-sampled state, teacher–reference JSD is computed over the student’s top-K candidates, with the remaining probability mass grouped into a residual token $v _ { \mathrm { o } } .$ . The highest-JSD states within each response are retained for the Direct-OPD objective.

Per-response selection. We select states within each response, so that rollouts with different overall divergence levels all contribute supervision. For valid response positions $\tau$ and a common retention ratio $\rho \in ( 0 , 1 ]$ , we retain the k states with the largest scores:

$$
k = \operatorname* { m a x } \left\{ 1 , \lceil \rho \rceil \rceil \right\} , \qquad \mathbb { Z } = \mathrm { T o p K } _ { t \in \mathcal { T } } ( \tilde { d } _ { t } , k ) ,\tag{10}
$$

where retaining at least one state ensures that every rollout is represented in the objective. Selecting within each response also makes each mask independent of the rest of the batch.

Masked objective. We apply the mask $m _ { t } = \mathbf { 1 } \{ t \in \mathcal { T } \}$ to both terms of the Direct-OPD objective:

$$
\begin{array} { r } { J _ { \mathrm { S ^ { 2 } D - O P D } } ( \theta ) = \mathbb { E } _ { \alpha \sim \mathcal { D } , ( s _ { t } , y _ { t } ) \sim \pi _ { \theta } } \left[ \Delta _ { t } ( y _ { t } \mid s _ { t } ) - \alpha D _ { \mathrm { K L } } \left[ \pi _ { \theta } ( \cdot \mid s _ { t } ) \| \pi _ { \mathrm { s t u } } ( \cdot \mid s _ { t } ) \right] \mid m _ { t } = 1 \right] . } \end{array}\tag{11}
$$

In practice, we hold the selection fixed during optimization and average over all retained states in the batch, which gives the update

$$
\mathbf { g } ^ { \mathrm { S ^ { 2 } D - O P D } } = \frac { 1 } { | \mathcal { T } _ { \mathcal { B } } | } \sum _ { t \in \mathcal { T } _ { \mathcal { R } } } \Big [ g _ { t } ^ { R } - \alpha \nabla _ { \theta } D _ { \mathrm { K L } } \big [ \pi _ { \theta } ( \cdot \mid s _ { t } ) \big | \big | \pi _ { \mathrm { s t u } } ( \cdot \mid s _ { t } ) \big | \Big ] ,\tag{12}
$$

where $\mathcal { T } _ { B }$ collects the retained states of all responses in the batch and $g _ { t } ^ { R }$ is the reward gradient of Eq. 4. Masked states thus receive neither the reward nor the student anchor, and $\rho = 1$ recovers Direct-OPD. The adaptive KL coefficient is still computed from all valid positions; App. A gives the remaining optimization details. Because the score reuses the teacher and reference probabilities that Direct-OPD already computes, S<sup>2</sup>D-OPD requires no extra forward passes.

## 5 EXPERIMENTS

Sec. 4.1 shows which states low-divergence masking removes, but not whether removing them improves transfer. We test this through three research questions (further analyses in App. C, D, E):

RQ1: Does masking low-JSD states improve transfer? (Sec. 5.2)

RQ2: How does JSD relate to supervision utility and selective masking gains? (Sec. 5.3)

RQ3: Are the gains robust across divergence measures and selection schemes? (Sec. 5.4)

Table 1: Main results (Avg@32) across four student models and two teacher–reference pairs, comparing dense Direct-OPD with S<sup>2</sup>D-OPD using top-10% JSD selection. Checkpoints are selected on AIME24/25 and evaluated on held-out AIME26 and HMMT (Nov. 2025 and Feb. 2026). Test Avg. is the average accuracy over 93 held-out problems; parentheses denote gains over the initial student.
<table><tr><td></td><td colspan="5">R1-Distill-1.5B → JustRL-1.5B</td><td colspan="5">Nemotron-1.5B → QuestA-1.5B</td></tr><tr><td></td><td colspan="2">Checkpoint Selection</td><td colspan="3">Held-out Evaluation</td><td colspan="2">Checkpoint Selection</td><td colspan="3">Held-out Evaluation</td></tr><tr><td>Student / Method</td><td>AIME24</td><td>AIME25</td><td>AIME26 HMMT</td><td></td><td>Test Avg.</td><td>AIME24</td><td>AIME25</td><td>AIME26 HMMT</td><td></td><td>Test Avg.</td></tr><tr><td>Qwen3-1.7B</td><td>49.3</td><td>36.7</td><td>37.7</td><td>28.2</td><td>31.3</td><td>49.3</td><td>36.7</td><td>37.7</td><td>28.2</td><td>31.3</td></tr><tr><td>+ Direct-OPD</td><td>59.7</td><td>42.9</td><td>45.4</td><td>32.0</td><td>36.4 (+5.1)</td><td>58.6</td><td>43.4</td><td>46.8</td><td>31.6</td><td>36.5 (+5.2)</td></tr><tr><td>+ S2D-OPD</td><td>61.4</td><td>44.5</td><td>47.4</td><td>32.9</td><td>37.6 (+6.3)</td><td>59.5</td><td>43.8</td><td>48.1</td><td>33.3</td><td>38.1 (+6.8)</td></tr><tr><td>Qwen3-4B</td><td>73.2</td><td>65.2</td><td>64.8</td><td>45.9</td><td>52.0</td><td>73.2</td><td>65.2</td><td>64.8</td><td>45.9</td><td>52.0</td></tr><tr><td>+ Direct-OPD</td><td>78.1</td><td>70.3</td><td>66.8</td><td>47.1</td><td>53.5 (+1.5)</td><td>77.6</td><td>70.4</td><td>70.4</td><td>47.1</td><td>54.6 (+2.6)</td></tr><tr><td>+ S2D-OPD</td><td>77.0</td><td>70.9</td><td>68.7</td><td>46.5</td><td>53.7 (+1.7)</td><td>78.0</td><td>68.3</td><td>70.7</td><td>49.3</td><td>56.2 (+4.2)</td></tr><tr><td>Qwen3-8B</td><td>77.3</td><td>66.0</td><td>67.5</td><td>49.7</td><td>55.4</td><td>77.3</td><td>66.0</td><td>67.5</td><td>49.7</td><td>55.4</td></tr><tr><td>+ Direct-OPD</td><td>77.5</td><td>72.2</td><td>69.8</td><td>50.2</td><td>56.5 (+1.1)</td><td>75.9</td><td>71.7</td><td>69.7</td><td>48.6</td><td>55.4 (+0.0)</td></tr><tr><td>+ S2D-OPD</td><td>78.0</td><td>73.4</td><td>70.7</td><td>50.0</td><td>56.7 (+1.3)</td><td>78.1</td><td>71.7</td><td>70.9</td><td>50.6</td><td>57.1 (+1.7)</td></tr><tr><td>R1-Distill-7B</td><td>56.7</td><td>40.5</td><td>48.2</td><td>29.5</td><td>35.6</td><td>56.7</td><td>40.5</td><td>48.2</td><td>29.5</td><td>35.6</td></tr><tr><td>+ Direct-OPD</td><td>63.6</td><td>45.7</td><td>56.8</td><td>32.8</td><td>40.5 (+4.9)</td><td>60.7</td><td>42.3</td><td>49.5</td><td>31.9</td><td>37.6 (+2.0)</td></tr><tr><td>+ S2D-OPD</td><td>64.6</td><td>47.1</td><td>55.0</td><td>33.6</td><td>40.5 (+4.9)</td><td>60.6</td><td>42.3</td><td>52.4</td><td>32.0</td><td>38.6 (+3.0)</td></tr></table>

## 5.1 EXPERIMENTAL SETUP

We evaluate $\mathrm { S ^ { 2 } D \mathrm { - } O P D }$ with two teacher pairs, R1-Distill-1.5B → JustRL-1.5B (He et al., 2025a) and Nemotron-1.5B → QuestA-1.5B (Li et al., 2026a), and transfer each policy shift to four students: Qwen3-1.7B, Qwen3-4B, Qwen3-8B<sup>1</sup>, and R1-Distill-7B<sup>2</sup>. All students are trained on Skywork-OR1-RL-Data (He et al., 2025b). Following the observation of Meng et al. (2026) that RL-induced policy shifts are sparse, with large divergence concentrated in a small fraction of tokens, we retain the top 10% of states per response $( \rho = 0 . 1 )$ by default. We select checkpoints on AIME 2024 and AIME 2025 and evaluate the selected checkpoint on three held-out benchmarks: AIME 2026, HMMT November 2025, and HMMT February 2026. All three postdate the release of every student model and are therefore absent from its training data, including undisclosed post-training data. We report Avg@32 on all benchmarks, define validation accuracy as the mean over AIME 2024 and AIME 2025, and give further training hyperparameter details in App. F.

## 5.2 MASKING LOW-DIVERGENCE STATES IMPROVES TRANSFER

Retaining 10% of states improves held-out transfer. Tab. 1 compares $\mathrm { S ^ { 2 } D \mathrm { - } O P D }$ with dense Direct-OPD across four students and two teacher pairs. Retaining 10% of states achieves comparable validation accuracy and improves the held-out Test Avg. in seven of eight settings, with a tie in the eighth. We assess statistical significance over 93 held-out problems, first averaging each problem’s 32 responses and then averaging its paired differences across the eight settings. Using a paired problem bootstrap and an exact one-sided sign-flip test, we find that S<sup>2</sup>D-OPD improves mean heldout accuracy from 46.36% to 47.31%, a gain of 0.95 points (95% CI: 0.40–1.54; $p = 6 . 4 \times 1 0 ^ { - 4 } )$

Selective supervision preserves early learning while improving late-stage stability. Fig. 2 shows the validation trajectories under the JustRL teacher pair. Despite masking low-divergence states, S<sup>2</sup>D-OPD keeps pace with dense Direct-OPD during early training, matching or exceeding its accuracy over the first 60 steps for all four students. Discarding 90% of states thus does not slow early learning, suggesting that the supervision driving early improvement is concentrated in the retained high-divergence subset. Later in training, S<sup>2</sup>D-OPD maintains higher accuracy for all four students and is more stable on Qwen3-4B and Qwen3-8B, where dense Direct-OPD repeatedly falls back from its peaks and, on Qwen3-4B, even drops below the initial student by step 300. These results answer RQ1: masking low-JSD states improves held-out transfer without slowing early learning.

![](images/be825324d06446d91e19b27b8bad1997c93513bf5cc5a8ede59935c132722408.jpg)  
Figure 2: Validation accuracy (Avg@32) under the JustRL teacher pair across four students. Dashed lines mark the initial student; dotted lines mark the JustRL-1.5B teacher.

![](images/b748983644fe378e3560b32f24b7f1b5812768c662ac66fdfbe80b3175c5fd7d.jpg)  
(a) Performance across JSD percentile bins.

![](images/47ce883d554a0d78edaf378eb181ce4a725648f74fd89644c23be1d103cd02ba.jpg)  
(b) Mean JSD tracks transfer gains.  
Figure 3: Which states are retained, not how many, determines transfer (Qwen3-1.7B, JustRL teacher pair). (a) Validation accuracy when training on each JSD percentile bin or on a random 10% of states. (b) Gain in peak validation accuracy over the initial student versus the mean training JSD of the retained states; the line is a log-linear fit.

## 5.3 LOW-DIVERGENCE SUPERVISION IS REDUNDANT AND HARMFUL

Sec. 4.1 shows that low-JSD states are those at which the teacher’s behavior changed little, yet the Direct-OPD update there can be as large as anywhere else. To test what such supervision contributes, we fix the JustRL teacher pair, the Qwen3-1.7B student, the training data, and the hyperparameters, and vary only which states enter the Direct-OPD objective. Within each response, we rank states by JSD, split them into ten equal-sized percentile bins (0–10, . . . , 90–100), and train a separate student on each bin. As a control, we train on a uniformly sampled 10% of states, whose update is an unbiased estimate of the dense Direct-OPD update (Fig. 3a).

Dense supervision is largely redundant. The random 10% control reaches a peak validation accuracy of 51.2, on par with dense Direct-OPD (51.3; Tab. 1). Discarding 90% of states at random thus loses little, supporting the view of Fu et al. (2026c) that OPD is “data-overfed but algorithm-starved.”

Which states are retained determines transfer. At the same 10% budget, performance rises broadly with JSD percentile, and the top bin reaches 52.9, above both the random control and dense Direct-OPD. The effect is graded: across all eleven subsets, the gain in peak validation accuracy over the initial student is approximately linear in the logarithm of the subset’s mean training JSD $( R ^ { 2 } = 0 . 9 7 ; \mathrm { F i g . } 3 \mathsf { b }$ ). The random control lies on the same line, so within this setting the mean divergence of the retained states predicts transfer whether they are selected by rank or at random.

![](images/af646a0dd410fef6629df9a257c1ce82200972a51f353db6827141629f7dd869.jpg)

![](images/71d4578faa12d24e5518f511c441f05f9d881b300a5e11084ee15108e0d7fce4.jpg)  
(a) All-position and selected-position JSD.

![](images/9efa89cd7b6b0b2e31cadf128209bea3ae26df6f4028c7cba6d641b7fe77909e.jpg)  
(b) Held-out gains over Direct-OPD.

Figure 4: Teacher-pair comparison with top-10% retention across four students. (a) Mean teacher– reference JSD over all valid response states (solid) and retained states (dashed) during S<sup>2</sup>D-OPD training. (b) Held-out Test Avg. gains of S<sup>2</sup>D-OPD over dense Direct-OPD under each teacher pair, in percentage points.  
![](images/5ac7f1d11b106333c0e124a0b3ac947f8e231ece38ca586060ce620bcb4cfd9b.jpg)  
Figure 5: JSD and token log-ratio rank states differently. The state with the largest sampled-token log-ratio in the response is masked, while a larger redistribution of probability mass is retained.

Low-divergence supervision is harmful, not merely weak. Trained in isolation, the three lowest bins end below the initial student, and the two lowest collapse. This matches the regime identified in Sec. 4.1, where the Direct-OPD update persists although the teacher barely changed. Top-K overlap suggests a mechanism (App. C): the top bin moves the student toward the teacher while preserving its overlap with the reference, whereas lower bins move it away from the reference without a commensurate gain in teacher alignment.

Divergence and reward magnitude rank states differently. Fig. 5 shows a state from an initial Qwen3-1.7B rollout under the JustRL pair. RL lowers the probability of the sampled token 5 from 0.12 to 2 × 10<sup>−5</sup>, giving this state the largest Direct-OPD reward magnitude among the response’s 8,192 states, although the teacher’s distribution moves by only 0.12 in total variation. Its JSD (0.044) falls below the retention cutoff (0.067), so $\mathrm { S ^ { 2 } D \mathrm { - } O P D }$ masks it while retaining a state with a smaller log-ratio but a larger redistribution of mass (JSD 0.262).

Masking helps more under the lower-divergence teacher pair. Mean JSD is higher for JustRL than for QuestA over both all states and the retained top 10% (Fig. 4a), yet the held-out gains of masking over dense Direct-OPD are larger under QuestA: 1.0–1.7 points versus 0.0–1.2 for JustRL across the four students (Fig. 4b). If the benefit came from concentrating supervision on high absolute divergence, this ordering would be reversed; together with the harm of low-JSD bins, it suggests that masking helps mainly by removing supervision that undermines learning. These results answer RQ2: JSD ranks supervision utility within a teacher–student setting, and masking pays off by filtering low-divergence supervision rather than by maximizing the divergence of what remains.

![](images/14a7d914f85f22fc3eea70eff14aec742e92a38ca5afca6e09d4005b3642d796.jpg)  
(a) Divergence measure.  
(b) Selection scope.  
Figure 6: Robustness of selective masking across design choices. (a) Top-10% selection by JSD, reverse KL, and forward KL on Qwen3-1.7B, 4B, and 8B under the JustRL teacher pair. (b) Responselevel versus batch-level selection on Qwen3-1.7B under both teacher pairs.

## 5.4 SELECTIVE MASKING IS ROBUST ACROSS DESIGN CHOICES

We vary three design choices while keeping the rest of $\mathrm { S ^ { 2 } D \mathrm { - } O P D }$ fixed: the divergence used for scoring, the scope over which states are ranked, and the retention rate (Fig. 6).

Divergence measure. The construction in Sec. 4.1 shows that both KL directions vanish with the candidate mass just as JSD does, so either should flag the same low-mass states. Under the JustRL pair, selection by forward KL, $D _ { \mathrm { K L } } ( \pi _ { \mathrm { T } } \Vert \pi _ { \mathrm { r e f } } )$ , or reverse $\mathrm { K L } , D _ { \mathrm { K L } } ( \pi _ { \mathrm { r e f } } \Vert \pi _ { \mathrm { T } } )$ , attains higher mean validation accuracy than dense Direct-OPD from step 160 onward on all three Qwen3 students, and avoids the late decline of dense Direct-OPD on Qwen3-4B (final accuracy 71.6–74.1 versus 67.2). Forward KL closely tracks JSD at all three scales, whereas reverse KL is weaker on Qwen3-4B.

Selection scope. Ranking states across the whole batch, so that a response may retain none, matches response-level selection on Qwen3-1.7B under both teacher pairs (peak validation accuracy 52.3 versus 52.9 under JustRL and 51.7 versus 51.7 under QuestA). The per-response rule of Sec. 4.2 is thus not needed for accuracy; we keep it so that each mask is independent of other responses.

Retention rate. Varying ρ from 5% to 20% on Qwen3-1.7B changes peak validation accuracy by less than one point (52.2–53.0), and every rate stays above dense Direct-OPD (51.3) and the random 10% control (51.2; App. D). These results answer RQ3: within the tested range, the gains do not hinge on the divergence measure, selection scope, or retention rate.

## 6 CONCLUSION

Direct-OPD transfers the policy shift that RL induces in a small teacher to a larger student by rewarding every student-sampled state with the teacher–reference log-ratio. We showed that this reward ignores the probability mass behind the shift: it can stay fixed while the mass on the student’s candidates, and with it the teacher–reference divergence, vanishes. Building on this observation, we introduced $\mathrm { S ^ { 2 } D \mathrm { - } O P D }$ , which retained Direct-OPD supervision only at the highest-JSD states of each response and required no extra forward passes. Retaining 10% of states improved held-out accuracy over dense Direct-OPD in seven of eight teacher–student settings and tied in the eighth. Controlled analyses further showed that dense supervision was largely redundant, that transfer depended on which states were retained rather than how many, and that low-divergence supervision could harm the student when trained on alone; the gains persisted across divergence measures, selection scopes, and retention rates. These findings suggest that weak-to-strong policy transfer depends not only on what is transferred but also on where it is applied. Extending state selection to adaptive retention rates and to domains beyond mathematical reasoning is a natural next step.

## AI USE STATEMENT

We used generative AI tools solely to improve the readability and clarity of the manuscript, including language editing and suggestions on presentation. All AI-assisted revisions were reviewed by the authors, who take full responsibility for the final content of this work.

## REPRODUCIBILITY STATEMENT

To facilitate reproducibility, we describe the selective supervision procedure in Section 4 and report the teacher–reference pairs, datasets, training configurations, and evaluation protocols in Section 5.1 and Appendix F. We provide our code at https://anonymous.4open.science/ r/S2D-OPD-8868.

## REFERENCES

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from selfgenerated mistakes. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=3zKtaqxLhW.

DeepSeek-AI, Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chengyu Hou, Chenhao Xu, et al. Deepseek-v4: Towards highly efficient million-token context intelligence, 2026. URL https://arxiv.org/abs/2606.19348.

Shiyuan Feng, Huan ang Gao, Haohan Chi, Hanlin Wu, Zhilong Zhang, Zheng Jiang, Bingxiang He, Wei-Ying Ma, Ya-Qin Zhang, and Hao Zhou. Weak-to-strong generalization via direct on-policy distillation, 2026. URL https://arxiv.org/abs/2607.05394.

Daocheng Fu, Rong Wu, Yu Yang, Jianbiao Mei, Licheng Wen, Pinlong Cai, Xuemeng Yang, Yong Liu, Botian Shi, and Yu Qiao. Proxy opd: On-policy distillation with transferable relative proxy update, 2026a. URL https://arxiv.org/abs/2607.11505.

Wei Fu, Jiaxuan Gao, Xujie Shen, Chen Zhu, Zhiyu Mei, Chuyi He, Shusheng Xu, Guo Wei, Jun Mei, WANG JIASHU, Tongkai Yang, Binhang Yuan, and Yi Wu. AREAL: A large-scale asynchronous reinforcement learning system for language reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview. net/forum?id=X9diEuva9R.

Yuqian Fu, Haohuan Huang, Kaiwen Jiang, Jiacai Liu, Zhuo Jiang, Yuanheng Zhu, and Dongbin Zhao. Revisiting on-policy distillation: Empirical failure modes and simple fixes, 2026b. URL https://arxiv.org/abs/2603.25562.

Zixuan Fu, Bingxiang He, Yuxin Zuo, Haohuan Huang, Jinqian Zhang, Ruhang Xiao, Cheng Qian, Qinyu Luo, Huan ang Gao, Yudong Wang, Zhiyuan Liu, Ning Ding, and Chaojun Xiao. Rethinking on-policy distillation of large language models ii: One training example, 2026c. URL https://arxiv.org/abs/2609.04172.

GLM-5-Team, Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, Chenzheng Zhu, Congfeng Yin, Cunxiang Wang, Gengzheng Pan, Hao Zeng, et al. Glm-5: from vibe coding to agentic engineering, 2026. URL https://arxiv.org/abs/2602.15763.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. MiniLLM: Knowledge distillation of large language models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=5h0qf7IBZZ.

Bingxiang He, Zekai Qu, Zeyuan Liu, Yinghao Chen, Yuxin Zuo, Cheng Qian, Kaiyan Zhang, Weize Chen, Chaojun Xiao, Ganqu Cui, Ning Ding, and Zhiyuan Liu. Justrl: Scaling a 1.5b llm with a simple rl recipe, 2025a. URL https://arxiv.org/abs/2512.16649.

Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, Siyuan Li, Liang Zeng, Tianwen Wei, Cheng Cheng, Bo An, Yang Liu, and Yahui Zhou. Skywork open reasoner 1 technical report, 2025b. URL https: //arxiv.org/abs/2505.22312.

Byeongho Heo, Jaehui Hwang, Sangdoo Yun, and Dongyoon Han. On-policy delta distillation, 2026. URL https://arxiv.org/abs/2607.15161.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, et al. Training compute-optimal large language models. In Proceedings ofthe 36th International Conference on Neural Information Processing Systems, NIPS ’22, Red Hook, NY, USA, 2022. Curran Associates Inc. ISBN 9781713871088.

Zhenyu Hou, Yujiang Li, Jie Tang, and Yuxiao Dong. Single-rollout asynchronous optimization for agentic reinforcement learning, 2026. URL https://arxiv.org/abs/2607.07508.

Nan Jia, Haojin Yang, Xing Ma, Jiesong Lian, Shuailiang Zhang, Weipeng Zhang, Ke Zeng, Xunliang Cai, and Zequn Sun. Asymmetric on-policy distillation: Bridging exploitation and imitation at the token level, 2026. URL https://arxiv.org/abs/2605.06387.

Woogyeol Jin, Taywon Min, Yongjin Yang, Dennis Wei, Yi Zhou, Swanand Ravindra Kadhe, Nathalie Baracaldo, and Kimin Lee. Entropy-aware on-policy distillation of language models. In Forty-third International Conference on Machine Learning, 2026. URL https: //openreview.net/forum?id=J5i09faOOf.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models, 2020. URL https://arxiv.org/abs/2001.08361.

Jongwoo Ko, Sara Abdali, Young Jin Kim, Tianyi Chen, and Pashmina Cameron. Scaling reasoning efficiently via relaxed on-policy distillation, 2026. URL https://arxiv.org/abs/2603. 11137.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, SOSP ’23, pp. 611–626, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9798400702297. doi: 10.1145/3600006.3613165. URL https: //doi.org/10.1145/3600006.3613165.

Jiazheng Li, Hongzhou Lin, Hong Lu, Kaiyue Wen, Zaiwen Yang, Jiaxuan Gao, Yi Wu, and Jingzhao Zhang. Questa: Expanding reasoning capacity in LLMs via question augmentation. In The Fourteenth International Conference on Learning Representations, 2026a. URL https://openreview.net/forum?id=3MifB0f7qR.

Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu Yu, Huan ang Gao, Wenkai Yang, Zhiyuan Liu, and Ning Ding. Rethinking on-policy distillation of large language models: Phenomenology, mechanism, and recipe, 2026b. URL https://arxiv. org/abs/2604.13016.

Kevin Lu and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism, 2025. doi: 10.64434/tml.20251026. https://thinkingmachines.ai/blog/on-policy-distillation.

Wenhan Ma, Hailin Zhang, Liang Zhao, Yifan Song, Yudong Wang, Zhifang Sui, and Fuli Luo. Stabilizing moe reinforcement learning by aligning training and inference routers, 2025. URL https://arxiv.org/abs/2510.11370.

Haoming Meng, Kexin Huang, Shaohang Wei, Chiyu Ma, Shuo Yang, Xue Wang, Guoyin Wang, Bolin Ding, and Jingren Zhou. Sparse but critical: A token-level analysis of distributional shifts in RLVR fine-tuning of LLMs. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=8vWIXno8LW.

MiniMax, Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo Yang, Boji Shan, Changqing Yu, Chao Wang, Cheng Zhu, Chengjun Xiao, Chengyu Du, Chi Zhang, Chu Qiao, Chunhao Zhang, et al. Minimax-m1: Scaling test-time compute efficiently with lightning attention, 2025. URL https://arxiv.org/abs/2506.13585.

Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amar Phanishayee, and Matei Zaharia. Efficient large-scale language model training on gpu clusters using megatron-lm. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, SC ’21, New York, NY, USA, 2021. Association for Computing Machinery. ISBN 9781450384421. doi: 10.1145/3458817.3476209. URL https://doi.org/10.1145/3458817.3476209.

Leyi Pan, Shuchang Tao, Yunpeng Zhai, Lingzhe Zhang, Zhaoyang Liu, Bolin Ding, Aiwei Liu, and Lijie Wen. Rlcsd: Reinforcement learning with contrastive on-policy self-distillation, 2026. URL https://arxiv.org/abs/2606.11709.

Tim Pearce and Jinyeop Song. Reconciling kaplan and chinchilla scaling laws. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/ forum?id=NLoaLyuUUF. Reproducibility Certification.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402. 03300.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, EuroSys ’25, pp. 1279–1297, New York, NY, USA, 2025. Association for Computing Machinery. ISBN 9798400711961. doi: 10. 1145/3689031.3696075. URL https://doi.org/10.1145/3689031.3696075.

Almog Tavor, Itay Ebenspanger, Neil Cnaan, and Mor Geva. Rethinking selective knowledge distillation, 2026. URL https://arxiv.org/abs/2602.01395.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, S. H. Cai, Yuan Cao, Y. Charles, H. S. Che, Cheng Chen, Guanduo Chen, Huarong Chen, Jia Chen, Jiahao Chen, Jianlong Chen, Jun Chen, Kefan Chen, et al. Kimi k2.5: Visual agentic intelligence, 2026. URL https://arxiv.org/ abs/2602.02276.

Xiaoxuan Wang, Han Zhang, Haixin Wang, Yidan Shi, Ruoyan Li, Kaiqiao Han, Chenyi Tong, Haoran Deng, Alexander K Taylor, Renliang Sun, Yanqiao Zhu, Jason Cong, Yizhou Sun, and Wei Wang. ARLArena: A unified framework for stable agentic reinforcement learning. In Fortythird International Conference on Machine Learning, 2026a. URL https://openreview. net/forum?id=90kxFi9VGP.

Yuanyi Wang, Su Lu, Yanggan Gu, Pengkai Wang, Yifan Yang, Zhaoyi Yan, Congkai Xie, Jianmin Wu, and Hongxia Yang. Not all disagreement is learnable: Token teachability in on-policy distillation, 2026b. URL https://arxiv.org/abs/2605.26844.

Bo Wu, Sid Wang, Yunhao Tang, Jia Ding, Eryk Helenowski, Liang Tan, Tengyu Xu, Tushar Gowda, Zhengxing Chen, Chen Zhu, Xiaocheng Tang, Yundi Qian, Beibei Zhu, and Rui Hou. Llamarl: A distributed asynchronous reinforcement learning framework for efficient large-scale llm training, 2025. URL https://arxiv.org/abs/2505.24034.

Jiayi Wu, Hao Sun, Hengyi Cai, Lixin Su, Shuaiqiang Wang, Dawei Yin, Xiang Li, and Ming Gao. Cross-model control: Improving multiple large language models in one-time training. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=YPqHSTSoFs.

Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, Zhipeng Wang, and Alborz Geramifard. Tip: Token importance in on-policy distillation, 2026. URL https://arxiv.org/abs/2604. 14084.

Wenkai Yang, Weijie Liu, Ruobing Xie, Kai Yang, Saiyong Yang, and Yankai Lin. Learning beyond teacher: Generalized on-policy distillation with reward extrapolation, 2026. URL https:// arxiv.org/abs/2602.12125.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, et al. Dapo: An open-source llm reinforcement learning system at scale, 2025. URL https://arxiv.org/abs/2503.14476.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model? In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=4OsgYD7em5.

Rosie Zhao, Alexandru Meterez, Sham M. Kakade, Cengiz Pehlevan, Samy Jelassi, and Eran Malach. Echo chamber: RL post-training amplifies behaviors learned in pretraining. In Second Conference on Language Modeling, 2025. URL https://openreview.net/forum? id=dp4KWuSDzj.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models, 2026. URL https: //arxiv.org/abs/2601.18734.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization, 2025. URL https://arxiv.org/abs/2507.18071.

Yuhang Zhou, Lizhu Zhang, Yifan Wu, Mingyi Wang, Bo Peng, Jiayi Liu, Xiangjun Fan, and Zhuokai Zhao. Sage-opd: Selective agent-guided intervention for multi-turn on-policy distillation, 2026. URL https://arxiv.org/abs/2606.19659.

Zilin Zhu, Chengxing Xie, Xin Lv, and slime Contributors. slime: An llm post-training framework for rl scaling. https://github.com/THUDM/slime, 2025. GitHub repository. Corresponding author: Xin Lv.

## A OPTIMIZATION DETAILS

This section details the adaptive KL control inherited from Direct-OPD and the selective positionwise aggregation used in $\mathrm { S ^ { 2 } D \mathrm { - } O P D }$

## A.1 ADAPTIVE KL CONTROL

The KL coefficient controls the strength of the student anchor relative to the policy-shift reward. We update this coefficient using Direct-OPD’s sign-based controller:

$$
\alpha _ { m + 1 } = \mathrm { c l i p } ( \alpha _ { m } \left[ 1 + \epsilon \mathrm { \ s g n } ( \bar { r } _ { m } ) \right] , \alpha _ { \mathrm { m i n } } , \alpha _ { \mathrm { m a x } } ) ,\tag{13}
$$

where $\bar { r } _ { m }$ denotes the mean student-weighted policy shift $\bar { p } _ { t } ( v ) \Delta _ { t } ( v \ \mid \ s _ { t } )$ over valid response positions and their top-K candidates at iteration m. The controller increases α when the mean weighted policy shift is positive and decreases it when negative, subject to the bounds. We use $\alpha _ { 0 } = 2 . 5 , \epsilon = 0 . 0 1$ , and $[ \alpha _ { \mathrm { m i n } } , \alpha _ { \mathrm { m a x } } ] = [ 0 . 5 , 2 . 5 ]$ ]. The updated coefficient $\alpha _ { m + 1 }$ is used in the subsequent actor update.

## A.2 DETAILED LOSS AGGREGATION

Direct-OPD averages the local update over all valid response positions. $\mathrm { S ^ { 2 } D \mathrm { - } O P D }$ keeps the local update unchanged but restricts this average to selected positions. Let I collect the positions selected independently within each response by Eq. 10 across the batch. Giving each retained position equal weight yields the local ascent direction

$$
g _ { \theta } ^ { \mathrm { S ^ { 2 } D . O P D } } = \frac { 1 } { | \mathcal { Z } | } \sum _ { t \in \mathcal { Z } } \left[ \sum _ { \boldsymbol { v } \in \mathcal { V } _ { K } ( \boldsymbol { s } _ { t } ) } \operatorname { s g } [ \bar { p } _ { t } ( \boldsymbol { v } ) \Delta _ { t } ( \boldsymbol { v } \mid \boldsymbol { s } _ { t } ) ] \nabla _ { \theta } \log \pi _ { \theta } ( \boldsymbol { v } \mid \boldsymbol { s } _ { t } ) - \alpha _ { m + 1 } \nabla _ { \theta } \widehat { d } _ { t } ( \boldsymbol { \theta } ) \right] .\tag{14}
$$

Here, sg denotes stop-gradient, and $\widehat { d } _ { t } ( \theta )$ is the per-position KL penalty estimator toward the initial student $\pi _ { \mathrm { s t u } }$ , computed using verl’s low var kl implementation. Both terms are applied to the same selected response positions. The sampled prefixes, candidate sets, selection mask, and KL coefficient are held fixed during differentiation.

## B PROOF OF THE MASS-INVARIANCE CONSTRUCTION

Proposition 1 (Mass invariance of the Direct-OPD update). Fix a state s and a student checkpoint $\theta _ { 0 } ,$ , and write $p ( v ) = \pi _ { \theta _ { 0 } } ( v \mid s ) > 0$ for all $v \in \mathcal V$ . Let $\gamma _ { K } ( s ) \subset \gamma$ be the top-K candidate set, with $K \geq 2$ , and define $\begin{array} { r } { \bar { p } ( v ) = p ( v ) / \sum _ { u \in \mathcal { V } _ { K } ( s ) } p ( u ) f o r v \in \mathcal { V } _ { K } ( s ) } \end{array}$ . Let $\mathbf { t } = ( t _ { v } ) _ { v \in \mathcal { V } _ { K } ( \pmb { s } ) }$ and $\mathbf { q } ~ = ~ ( q _ { v } ) _ { v \in \mathcal { V } _ { K } ( \pmb { s } ) }$ be distinct, strictly positive probability vectors on the candidate set, and let $\mathbf { b } = ( b _ { v } ) _ { v \in \mathcal { V } \backslash \mathcal { V } _ { K } ( \pmb { s } ) }$ be a common strictly positive probability vector on its complement. For $0 < \epsilon < 1$ , define the teacher and reference distributions:

$$
P _ { \epsilon } ( v ) = \left\{ \begin{array} { l l } { \epsilon t _ { v } , } & { v \in \mathcal { V } _ { K } ( s ) , } \\ { ( 1 - \epsilon ) b _ { v } , } & { v \notin \mathcal { V } _ { K } ( s ) , } \end{array} \right. \quad \quad Q _ { \epsilon } ( v ) = \left\{ \begin{array} { l l } { \epsilon q _ { v } , } & { v \in \mathcal { V } _ { K } ( s ) , } \\ { ( 1 - \epsilon ) b _ { v } , } & { v \notin \mathcal { V } _ { K } ( s ) . } \end{array} \right.\tag{15}
$$

Here, ϵ is the total probability mass that each checkpoint assigns to the student’s candidates. Both checkpoints vary with $\epsilon ,$ while the student, candidate set, and conditional distributions remainfixed. Then,for every $\epsilon \in ( 0 , 1 )$

(i) Reward invariance. For every candidate $v \in \mathcal { V } _ { K } ( s )$ , the Direct-OPD reward $( E q . \ I )$ with $\pi _ { \mathrm { T } } = P _ { \epsilon }$ and $\pi _ { \mathrm { r e f } } = Q .$ <sub>ϵ</sub> satisfies

$$
\Delta _ { \epsilon } ( v \mid s ) = \log \frac { P _ { \epsilon } ( v ) } { Q _ { \epsilon } ( v ) } = \log \frac { t _ { v } } { q _ { v } } .\tag{16}
$$

(ii) Fixed, nonzero update. The local reward gradient ofDirect-OPD $( E q . 4 )$ at $\theta _ { 0 }$

$$
{ \bf g } _ { \epsilon } = \sum _ { v \in \mathcal { V } _ { K } ( s ) } \bar { p } ( v ) \log \frac { t _ { v } } { q _ { v } } \nabla _ { \theta } \log \pi _ { \theta } ( v \mid s ) \Big | _ { \theta = \theta _ { 0 } } = { \bf g } ,\tag{17}
$$

is independent ofϵ and nonzero with respect to the student’s logits.

(iii) Divergences scale with the mass. The teacher–reference divergences satisfy

$$
\begin{array} { r } { D _ { \mathrm { J S } } ( P _ { \epsilon } , Q _ { \epsilon } ) = \epsilon D _ { \mathrm { J S } } ( \mathbf { t } , \mathbf { q } ) , D _ { \mathrm { K L } } ( P _ { \epsilon } \| Q _ { \epsilon } ) = \epsilon D _ { \mathrm { K L } } ( \mathbf { t } \| \mathbf { q } ) , D _ { \mathrm { K L } } ( Q _ { \epsilon } \| P _ { \epsilon } ) = \epsilon D _ { \mathrm { K L } } ( \mathbf { q } \| \mathbf { t } ) . } \end{array}\tag{18}
$$

Consequently, $a s \epsilon \to 0 ,$ , all three divergences vanish, whereas the reward and its gradient remain fixed and nonzero.

Proof. (i) The common factor ϵ cancels in the ratio. (ii) By (i), each term of $\mathbf { g } _ { \epsilon }$ is independent of $\epsilon ,$ since $\nu _ { K } ( \pmb { s } )$ and $\bar { p }$ depend only on $\theta _ { 0 } .$ For softmax logits z, ∇<sub>z</sub> log $\pi ( \boldsymbol { v } \mid \boldsymbol { s } ) = \mathbf { e } _ { \boldsymbol { v } } - \pi ( \cdot \mid \boldsymbol { s } )$ so $\begin{array} { r } { \mathbf { g } = \mathbf { w } - \left( \sum _ { u } w _ { u } \right) p } \end{array}$ , where $w _ { v } = \bar { p } ( v ) \log ( t _ { v } / q _ { v } )$ on $\gamma _ { K } ( \pmb { s } )$ and $w _ { v } = 0$ elsewhere. Since $\mathcal { V } _ { K } ( \pmb { \mathscr { s } } ) \subset \mathcal { V } \mathrm { a n d } \bar { p } > 0 , \mathbf { \mathscr { s } } = \mathbf { 0 }$ would require $\mathbf { w } = \mathbf { 0 } , \mathrm { i . e . , \mathbf { t } = \mathbf { q } . }$ . (iii) Off the candidate set, $P _ { \epsilon } , Q _ { \epsilon }$ and their mixture coincide, so the shared tail contributes zero to each divergence; on the candidate set, the factor ϵ cancels inside each logarithm and factors out of the sum. □

## C TOKEN OVERLAP ACROSS DIVERGENCE RANKS

Prior work associates successful OPD with increasing overlap between the student’s and teacher’s high-probability token sets (Li et al., 2026b). Direct-OPD instead transfers a teacher–reference logratio, motivating us to examine overlap with both checkpoints. We track how these overlaps evolve when supervision is restricted to different teacher–reference JSD deciles.

For a student policy $\pi _ { S }$ and a comparison policy $\pi _ { C }$ at state $s _ { t } ,$ , let

$$
\mathcal { V } _ { k } ^ { S } ( \pmb { \mathscr { s } } _ { t } ) = \mathrm { T o p K } \big ( \pi _ { S } ( \cdot \mid \pmb { \mathscr { s } } _ { t } ) , k \big ) ,\tag{19}
$$

$$
\mathcal { V } _ { k } ^ { C } ( \pmb { s } _ { t } ) = \mathrm { T o p K } \big ( \pi _ { C } ( \cdot \mid \pmb { s } _ { t } ) , k \big ) ,\tag{20}
$$

![](images/2db346c39190a58a6ed804722ffd9366b4838eb6783caeaa747d72141090b613.jpg)

![](images/5c1af93a8666c56549584186567e20c4a71ef433b3736e74b85c2fc3ae010a1d.jpg)  
Figure 7: Top-16 token overlap over training for Qwen3-1.7B under the JustRL teacher pair, using JSD deciles or random 10% selection. Panels compare the student with the post-RL teacher (left) and pre-RL reference (right). Overlap is the fraction of shared tokens between top-16 sets.

so that the per-state overlap ratio is

$$
\mathrm { O v e r l a p } _ { k } ( S , C ; \pmb { s } _ { t } ) = \frac { \big | \mathcal { V } _ { k } ^ { S } ( \pmb { s } _ { t } ) \cap \mathcal { V } _ { k } ^ { C } ( \pmb { s } _ { t } ) \big | } { k } .\tag{21}
$$

which we report with $k = 1 6$ against both the post-RL teacher $( C = T )$ and the reference $( C =$ $T _ { \mathrm { r e f } } )$ . Every bin starts from the same student initialization, so the two panels record how training on a given decile moves the student relative to each checkpoint.

Higher-JSD bins generally attain greater overlap with the post-RL teacher (Figure 7, left), broadly tracking the performance ordering in Figure 3a. This resembles the overlap growth observed in successful standard OPD (Li et al., 2026b), here under an objective that transfers the teacher’s policy shift. The reference panel adds a complementary observation: the highest-JSD bin increases teacher overlap while maintaining reference overlap above its initial level. Reference overlap alone does not follow the performance ordering; for example, the 10–20 bin recovers close to its initial reference overlap despite much smaller gains in teacher overlap.

The strongest transfer occurs without a trade-off between overlap with the two checkpoints: the highest-JSD bin increases teacher overlap while sustaining reference overlap. This suggests that transferring the RL-induced policy shift need not entail moving from the reference’s high-probability candidate set. The teacher–reference log-ratio can favor a token that remains highly ranked under both checkpoints, allowing transfer through changes in relative preference among shared candidates. Related evidence from standard OPD shows that supervision restricted to tokens shared by the student and teacher recovers nearly the full benefit of student top-k supervision (Li et al., 2026b). Our JSD criterion identifies positions with substantial changes in candidate probabilities or total mass relative to the tail (Eq. 8). Together, these observations suggest a view of selective transfer as learn ing substantial changes in probability allocation within largely overlapping candidate spaces.

## D SENSITIVITY TO RETENTION RATIO

Our main experiments retain the 10% of positions with the highest teacher–reference JSD within each response. Figure 8 compares retention ratios of 5%, 10%, 15%, and 20% for Qwen3-1.7B under the JustRL teacher pair. All four settings yield broadly similar learning curves and substantial gains over the base model on AIME 2024 and AIME 2025. No retention ratio consistently dominates across both benchmarks and training checkpoints.

Together with the decile analysis in Section 5.3, these results reveal an asymmetry in supervision selection. Training on low-JSD bins can substantially degrade transfer, whereas varying the retained fraction within the high-JSD region has a comparatively small effect. Moreover, the most selective setting does not consistently outperform the broader subsets. This pattern supports using JSD to screen supervision, without requiring that larger divergence always imply greater transfer utility. It also accords with our theoretical motivation: small divergence bounds the observable policy change, while large divergence alone does not establish the value of transferring that change. The similar performance across retention ratios suggests that effective selection admits a broad operating range in this setting, and we adopt 10% as a common default.

![](images/2cd585e8058aa3647864feb54287bbff87177265933ac11e936ddf49c5171c36.jpg)  
Figure 8: Sensitivity to the per-response retention ratio for Qwen3-1.7B under the JustRL teacher pair. The four settings retain the top 20%, 15%, 10%, and 5% of positions ranked by teacher– reference JSD. Dashed lines indicate base-model accuracy.

![](images/9a3653a7003332dd4c887ba51231956b03df43560c8108740485f5cc229165d1.jpg)  
Figure 9: Same sampled token, different mask decisions. The mask filters a calculation on which the teacher and reference agree but retains supervision for a boundary correction.

## E CASE STUDY: WHAT POLICY CHANGES DOES SELECTION PRESERVE?

To examine what the mask keeps and removes, we inspect states from 20 rollouts of the initial Qwen3-1.7B student on AIME 2026 under the JustRL pair. At each state, we compare the teacher and reference next-token distributions over the student’s top-K candidates and the residual token, and we mark the state as retained if its JSD falls within the top 10% of its response, following the per-response rule of Sec. 4.2. We write ∆p for the change in a token’s probability from the reference to the teacher. We selected the four cases below by hand for interpretability, so they illustrate how the score behaves rather than estimate how often each pattern occurs.

Selection acts on states, not on sampled tokens. Fig. 9 contrasts two states at which the student samples 3. At the first, 3 completes a valid calculation, and the teacher and reference distributions are nearly identical (JSD = 0.0001). At the second, 3 sets an incorrect boundary: k = 3 still permits A = B = 1, so excluding their simultaneous occurrence requires $k \geq 4 .$ . Here RL moves probability from 3 to 4 (∆p ≈ −0.55 and +0.56), and the state is retained (JSD = 0.1735). The same token thus receives opposite mask decisions, because the score depends on how the teacher’s distribution changed at the state rather than on which token was sampled.

![](images/f953b3afbd837d49b880f41c18dc95d209dd780e10da3ecf8e1e47aad2133881.jpg)

![](images/833b739d40f4999b5fb7ac63331d1e86c117ba463a38f9eae4d31ba01c970ebd.jpg)  
Figure 10: Retained shifts encode continuation preferences. At two retained states, the teacher shifts probability away from Alternatively; the later state favors Let, Given, and Since.

![](images/5092e2c8fc5f3fbc81e621b23a1addd1e1ec8510f4658352d9e6d39c82a57670.jpg)

![](images/2eb1bb16ec1464572767ebbc175fe07b7963ac38aa4440d922b2a612363a3856.jpg)  
Figure 11: A later reconsideration cue does not undo an earlier adverse shift. The mask retains a shift toward an incorrect parity judgment, followed by a teacher preference for reconsideration.

Retained shifts include preferences over how reasoning continues. Fig. 10 shows two retained states at which the student opens a new passage with Alternatively . At both, RL lowers the probability of this token; at the later state the decrease is large $\overline { { ( \Delta p \approx } } - 0 . 7 1 )$ , and the teacher instead favors Let, Given, and Since. Together with the arithmetic cases, this shows that the retained supervision covers both local corrections and preferences over how the reasoning proceeds, consistent with a score that measures behavioral change rather than correctness.

JSD does not judge correctness. In Fig. 11, the student describes R1 = 6 as odd . RL reinforces this error, raising the probability of odd (∆p ≈ +0.41) and lowering that of even (∆p ≈ −0.12). The JSD of 0.1197 exceeds the response’s retention cutoff of 0.0744, so this adverse shift is retained. At the next state, conditioned on the erroneous statement, the teacher favors Wait over Therefore (from 0.01 to 0.71 and from 0.49 to 0.02), a preference compatible with reconsideration, but this later shift does not cancel the reward for odd at the earlier state. The score thus selects states by the size of the teacher’s change, not by whether the change is correct.

Summary. These cases separate where supervision is applied from what it encourages. JSD decides which states are retained, while Direct-OPD’s log-ratio rewards decide which tokens are encouraged at those states. Because the score measures how much the teacher’s distribution changed on the student’s candidates, the retained states include numerical corrections, continuation preferences, and local errors alike. Masking therefore concentrates Direct-OPD supervision on the states where RL changed the teacher most, without filtering shifts by their correctness.

Table 2: Default training and evaluation configuration.
<table><tr><td>Setting</td><td>Training</td><td>Evaluation</td></tr><tr><td>Framework</td><td>verl</td><td></td></tr><tr><td>Hardware</td><td>8× NVIDIA H200</td><td></td></tr><tr><td>Global batch size</td><td>128</td><td></td></tr><tr><td>Mini-batch size</td><td>128</td><td></td></tr><tr><td>Rollout n</td><td>4</td><td></td></tr><tr><td>Max. prompt length</td><td>1,024</td><td></td></tr><tr><td>Max. response length</td><td>2,048</td><td>31,744</td></tr><tr><td>Samples per problem</td><td></td><td>32</td></tr><tr><td>Sampling temperature</td><td>1.0</td><td>0.7</td></tr><tr><td>OPD support size Top-K</td><td>16</td><td></td></tr><tr><td>Top-p sampling</td><td>1.0</td><td>0.95</td></tr><tr><td>Learning rate</td><td>1 × 10−6</td><td></td></tr><tr><td>Training steps</td><td>300</td><td></td></tr><tr><td>KL coefficient α</td><td>Adaptive</td><td></td></tr><tr><td>Controller €</td><td>0.01</td><td></td></tr><tr><td>[αmin, αmax]</td><td>[0.5, 2.5]</td><td></td></tr><tr><td>Checkpoint selection</td><td></td><td>AIME 24/25</td></tr><tr><td>Held-out evaluation</td><td></td><td>AIME 26, HMMT Nov.25 / Feb.26</td></tr></table>

## F TRAINING DETAILS

Data and prompt. All runs use the math subset of Skywork-OR1-RL-Data (He et al., 2025b) and apply the following prompt template.

Solve the following math problem step by step.   
The last line of your response should be of the form   
Answer: \$Answer (without quotes) where \$Answer is the   
answer to the problem.   
{Question}   
Remember to put your answer on its own line after "Answer:".

Training and evaluation. All experiments are implemented with verl and run on 8 NVIDIA H200 GPUs. Table 2 summarizes the default training and evaluation configuration used throughout our experiments. Owing to limited compute, each configuration is trained with a single run; the confidence interval in Sec. 5.2 resamples held-out problems and does not reflect variation across training seeds.

## G LIMITATIONS

Our work has four main limitations. First, JSD is a proxy for how much RL changed the teacher, not a judge of whether the change is correct. Because it scores only the magnitude of the teacher– reference difference, S<sup>2</sup>D-OPD retains a large shift toward an error as readily as a large correction (Fig. 11), so the quality of the retained supervision still depends on the teacher’s RL. Selectors that also account for the direction or correctness of a shift, for example through verifier or outcome signals, could filter such states. Second, JSD is blind to what the student needs. The score depends on the student only through its top-K candidate set: a state where the student already follows the teacher’s shift consumes the same budget as one where it does not, and a low-divergence shift that the student lacks is discarded. Combining teacher–reference divergence with student-side signals, in the spirit of the student-side calibration in Proxy-OPD (Fu et al., 2026a), is a natural extension. Third, our evidence is limited in scale and scope. We transfer from 1.5B teachers to students of up to 8B parameters, whereas weak-to-strong transfer is most valuable for much larger students, where RL is most costly. All experiments use mathematical reasoning with verifiable rewards; whether selective supervision helps in code generation or agentic tasks, where RL-induced shifts may be distributed differently across states, remains untested. Finally, owing to limited compute, each configuration is trained once, so our confidence interval reflects variation across held-out problems rather than across training seeds; repeated runs would strengthen the per-setting comparisons.