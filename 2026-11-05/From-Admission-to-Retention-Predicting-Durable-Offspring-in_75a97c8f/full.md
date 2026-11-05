# From Admission to Retention: Predicting Durable Offspring in Quality-Diversity Search

Corrado Calamita<sup>∗</sup> Luca Ferrari<sup>†</sup>

Matteo Mendicini<sup>∗</sup> Riccardo Bertamini<sup>†</sup>

BAINSA – Board of AI and Neuroscience Student Association

## Abstract

Automated failure discovery can be framed as a diversity problem: the objective is to expose distinct ways a system fails rather than a single critical adversarial case. Yet uniform parent sampling provides no way to allocate search effort toward individuals likely to produce durable archive contributions. We study this credit assignment problem in an unstructured Quality-Diversity (QD) archive, testing whether signals available before an individual has reproduced distinguish offspring that merely enter the archive from those still retained generations later. Across a planar-arm benchmark and two robotic-locomotion tasks, the same signals are often associated with entry and persistence in opposite directions. In particular, lineage depth is the clearest such signal, and its persistence discrimination grows with population capacity, which is associated with lower turnover. Biasing parent selection toward the signals with the strongest measured association to survival nevertheless does not improve search: moderate bias matches uniform sampling, and strong bias reduces QD-score.

## 1 Introduction

Failure discovery can require a collection of behaviourally distinct failures. This objective has motivated QD-based scenario generation (Fontaine and Nikolaidis, 2021). QD algorithms aim to produce collections of behaviorally diverse and high-performing solutions (Lehman and Stanley, 2011; Pugh et al., 2016), specifically MAP-Elites (Mouret and Clune, 2015) uses a fixed descriptor grid and CVT-MAP-Elites still partitions a bounded descriptor space in advance (Vassiliades et al., 2018). DNS (Bahlous-Boldi et al., 2025) differs from this perspective: it removes the partition while retaining a descriptor, distance metric, and finite population capacity. We use that setting as motivation and evaluate on standard QD benchmarks, where behaviour spaces are predefined enabling the focus to be the measurement of parent offspring events. In typical MAP-Elites implementations, pressure toward exploitation acts mainly during replacement in cells while retained elites are sampled uniformly as parents (Mouret and Clune, 2015). Related selection pressures appear in different QD frameworks. Cully and Demiris use accumulated offspring success to bias archive-parent selection (Cully and Demiris, 2018), and Wang et al. rank MAP-Elites parents using non-surrounded dominance (Wang et al., 2023), observing that QD algorithms “often select the parent solutions uniformly at random, which lacks selection pressure and may limit the performance”.

The same question, being how to allocate search effort from accumulated performance information rather than uniformly, has a longer history in evolutionary computation. Most directly, Sfikas et al. formulate MAP-Elites parent selection as a multi-armed bandit problem and use the survival rate of a parent’s offspring as an indirect reward (Sfikas et al., 2021). That signal provides accumulated feedback for previously selected parents; our question instead asks whether useful predictors exist before a parent has produced any offspring.

Table 1: Signal availability.
<table><tr><td>Group</td><td>Cold-start? Signals</td><td></td></tr><tr><td>Lineage at birth</td><td>yes</td><td>depth; ancestral-fitness EWMA; ancestry entropy; competition fitness at birth</td></tr><tr><td>Current state</td><td>yes</td><td>age</td></tr><tr><td>Archive state</td><td>yes</td><td>parent fitness; current competition fitness</td></tr><tr><td>Mating event</td><td>no</td><td>|λ|; compositional residual; child fitness</td></tr><tr><td>Offspring feedback</td><td>no</td><td>curiosity; offspring survival rate</td></tr></table>

Such feedback is informative only after an individual has reproduced. Kistemaker and Whiteson (2011) formalise the potential of a genotype as its descendant fitness probability (DFP), the distribution over the fitness of its future descendants, and use it to analyse behaviour characterisations in novelty search rather than as a selection rule. They are explicit that it cannot serve as one, for two reasons: even a simplified DFP is intractable on non-trivial problems, and were it known, the descendants’ fitnesses would be known with it, leaving no optimisation problem to solve. Any practical rule must therefore substitute a proxy that is cheap and already available, and the empirical question is which proxies carry information.

## 2 Setting and Method

Dominated Novelty Search. Let $\mathcal { P } = \mathcal { A } \cup \mathcal { O }$ be the pool formed by the current archive A and the offspring batch O, and let $f _ { i }$ and $d _ { i }$ denote the objective fitness and the descriptor of individual i. DNS assigns each individual a competition fitness: its mean descriptor-space distance to the $k = 3$ nearest individuals in P of equal or higher objective fitness. Writing $\begin{array} { r } { N _ { i } = \dot { k } \mathrm { - N N } ( \left\{ j \in \mathcal { P } , j \neq i : f _ { j } \geq f _ { i } \right\} ) } \end{array}$ for that neighbour set,

$$
\widetilde { f } _ { i } = \frac { 1 } { | N _ { i } | } \sum _ { j \in N _ { i } } \| d _ { i } - d _ { j } \| _ { 2 } .\tag{1}
$$

If fewer than k such individuals exist we average over those that do; if there are none we set $\widetilde f _ { i } = + \infty$ The pooled set is then ranked by $\widetilde { f }$ from highest to lowest and truncated to the population capacity. Distances use unnormalised Euclidean distance on the raw descriptors. Two consequences follow. Eviction is global and recurrent rather than local and triggered: each archive update re-ranks the entire pooled set, so an individual can be displaced at any later generation by an event anywhere in descriptor space. Retention is therefore an ongoing outcome rather than a one-time contest for a cell, which is what makes persistence worth measuring separately from entry, and makes archive age and lineage history plausible signals of repeated global competition. Despite having several archive and lineage signals available, DNS still samples parents uniformly. This leaves open whether any signal available before reproduction can identify parents whose offspring will persist.

Signal availability. For a signal to guide an individual’s first selection as a parent, it must be defined before that individual has produced offsprings. Cold-start selectability is distinct from being fixed at birth: age is selectable before reproduction although it changes over time. Lineage depth is the number of reproduction steps from the founder along the base-parent chain and is fixed at creation. Prior QD work has related genealogical diversity to performance post hoc (Nordmoen et al., 2021); here we test lineage information as a candidate selection-time signal. The archive-state group provides our contemporaneous baselines: the selection quantities the literature already proposes, recomputed from the current archive at every generation.

Feedback signals remain useful diagnostics: at capacity 1024, on Arm, Omnidirectional Ant and Walker 2D respectively, curiosity reaches entry AUC 0.738, 0.603 and 0.601, and offspring survival rate 0.673, 0.581 and 0.564. They are nevertheless undefined for a newcomer.

Protocol. We distinguish between two moments: immediate entry into the archive and long-term persistence. We measure all twelve signals in Table 1 (eleven at capacity 4096; see Appendix A.2). All AUC measurements are made under uniform parent selection. Entry means that an offspring is kept in the archive immediately after it is created. Persistence means that it is still in the archive H generations later. We use $H = 1 0 0 0$ for the main results and check $H \in \{ 1 0 0 , 1 0 0 0 \}$ on all tasks and $\bar { H } = 1 0$ on Arm, where archive logging is per-generation; signs are stable across horizons. Offspring created too close to the end of a run to measure persistence are excluded.

Let $g ^ { + }$ denote the signal value of a parent whose offspring achieved the outcome and $g ^ { - }$ that of a parent whose offspring did not. For each signal we compute

$$
\mathrm { A U C } = \mathrm { P r } \left( g ^ { + } > g ^ { - } \right) + \textstyle { \frac { 1 } { 2 } } \mathrm { P r } \left( g ^ { + } = g ^ { - } \right) , \qquad \Delta = \left| \mathrm { A U C } - 0 . 5 \right| ,\tag{2}
$$

and report the discrimination strength $\Delta$ together with the sign of AUC − 0.5. An AUC of 0.5 means no discrimination; $\mathrm { A U C } < 0 . 5$ means the association runs in the opposite direction, which does not imply that inverting the signal yields a good selection rule. We compute one AUC per signal per run, with different seeds per (domain, capacity) cell (Table 3), and use runs, rather than individual parent-offspring pairs, as the unit of analysis. This avoids treating the ${ \approx } 2 . 5 \times 1 0 ^ { 7 }$ correlated pairs as independent samples.

We evaluate three domains at archive capacities 256, 1024 and 4096, holding offspring per generation fixed at 256, so the ratio of new candidates to available slots is 1.00, 0.25 and 0.06 and larger archives turn over more slowly. Each measurement run lasts 10,000 generations with $k = 3$ and uniform parent sampling. Measured across domains, discrimination and archive entry rate co-vary: entry rates are approximately 0.13 for Arm, 0.10 for Omnidirectional Ant and 0.04 for Walker 2D. Varying capacity within a fixed domain holds the operator, the descriptor space and the per-generation evaluation budget identical, but it is not a clean instrument for turnover alone: it also changes the offspring-to-slot ratio, the competition geometry and the attainable coverage. We therefore report an association between capacity and discrimination that is consistent with a turnover mechanism, rather than an isolated causal effect of turnover.

Selection rule. For the intervention we replace uniform sampling of the base parent with a ranksoftmax, $P ( i ) \propto \exp ( \operatorname { r a n k } ( g _ { i } ) / \tau )$ , over the occupied slots, so the ratio between the most- and least-favoured parent is $\exp ( 1 / \tau )$ whatever the signal’s units; ranks are tie-averaged for integervalued signals such as depth. Sweeps are listed in Table 3.

## 3 Results

Entry is not persistence. At capacity 1024, several selectable signals reverse direction between immediate entry and persistence after $H = 1 0 0 0$ generations on all three tasks (Figure 1). This means that properties associated with easier immediate admission can be associated in the opposite direction with survival 1000 generations later. Lineage depth is the clearest case: shallower lineages are associated with entry, while deeper lineages are associated with persistence.

This describes a rank association and does not show which signal, if any, would improve coverage when used for selection.

![](images/cb447c0598add275f91023837e87a9a1645e5465aa5787cd4324e39df25cf8c3.jpg)

![](images/3a1be1be2655bcd0275479bdf0efe3f2b2fb69b1a11ca305798695e2ea62892e.jpg)

![](images/0d0f27b50cc16b462d24f390ac2369b1a9b07bafa584a2fdf67db981c1a5b666.jpg)  
Figure 1: Entry against persistence AUC at capacity 1024 and $H = 1 0 0 0$ , one panel per domain. Points are cold-start-selectable signals averaged across seeds; blue denotes lineage- and current-state signals, grey the contemporaneous archive-state baselines. Grey lines mark chance. Signals on opposite sides of chance for the two outcomes are associated with entry and persistence in opposite directions.

Turnover and depth spread. The weaker AUC at high turnover is not caused by retained individuals having very similar lineage depths. In fact, depth variation is already large at capacity 256 and usually decreases as capacity grows. For example, on Omnidirectional Ant, persistence discrimination increases from 0.097 to 0.198 (Figure 2) while the interquartile range of lineage depth drops from ≈94 to ≈16. The association strengthens on Arm and Walker 2D as well, without any comparable narrowing of depth. Lineage-depth persistence AUC rises monotonically with population capacity on all three tasks, while entry AUC falls below chance (Figure 2).

![](images/d915873ce95f3efd119d9a31dd5a3c017e2572de648ca4b465755ad90b48bc30.jpg)  
Figure 2: Signed lineage-depth AUC across archive capacities for entry and persistence. Larger archives have lower turnover because the number of offspring stays fixed.

Neither signal converts into a selection rule. The turnover result makes a falsifiable prediction. Parent fitness discriminates persistence on Omnidirectional Ant roughly five times as well at capacity 4096 as at 256 (from 0.027 to 0.138), so if biasing toward it fails at low capacity for want of association to exploit, it should succeed at high capacity. We found it does not. Moderate pressure $( \tau = 0 . 5 )$ matches uniform DNS at both capacities, reaching projected QD-score $4 . 0 4 \times 1 0 ^ { 6 }$ against $4 . 0 2 \times 1 0 ^ { 6 }$ for uniform at capacity 4096 over 10 seeds; strong pressure $( \tau = 0 . 1 )$ is substantially worse, and worse at the higher capacity (−64.4% against −43.6% at capacity 1024). This represents a loss of diversity rather than of quality as: projected coverage drops from 0.9999 to 0.517, mean novelty drops from 0.848 to 0.544 while maximum fitness stays essentially unchanged. The birth-available signal behaves no better. On Omnidirectional Ant at capacity 1024 over 10,000 generations, rank-softmax on lineage depth $( \tau = 0 . 5 )$ reaches projected QD-score $3 . 1 3 0 \times 1 0 ^ { 6 }$ at coverage 0.767; weakening the bias $( \tau = 2 . 0 )$ gives $3 . 1 1 2 \times 1 0 ^ { 6 } ;$ ; inverting it gives $3 . 1 2 5 \times 1 0 ^ { 6 }$ at coverage 0.769. The matched uniform run at the same capacity and horizon reaches $3 . 1 5 6 \times 1 0 ^ { 6 }$ at coverage 0.773. No depth policy exceeds uniform sampling on either metric, and every difference falls within the between-seed spread, so favouring deep lineages, weakening that preference, and reversing it are all indistinguishable from sampling uniformly. Coverage here is 0.77 rather than saturated, so this null is not an artefact of a coverage ceiling.

## 4 Conclusion

Entry and long-term persistence are associated with different, often oppositely signed parent properties, so the two outcomes should be evaluated separately, and a signal is usable only if it is defined before an individual has reproduced. Lineage depth satisfies both conditions and its retention association strengthens as turnover falls, even as the spread of depth narrows, but it does not translate into a better selection rule. These results concern predictive association rather than causal credit, and are limited to DNS, one variation operator and three standard QD domains.

This null is not specific to parent selection: perturbing DNS’s survival rule degrades QD-score monotonically in the perturbation size on all three domains, again through coverage rather than fitness, and how far a rule moves the selection ordering predicts the loss (Appendix A.3). DNS’s ordering is hard to improve on locally, which bounds what a different rule form can buy. More broadly, if turnover governs how informative lineage history is, then archive capacity is not only a budget parameter but a choice about what credit assignment is possible at all, a parameter a self-evolving search could adapt during a run rather than fix in advance.

## References

Ren-Jian Wang, Ke Xue, Haopu Shang, Chao Qian, Haobo Fu, and Qiang Fu. Multi-objective optimization-based selection for quality-diversity by non-surrounded-dominated sorting. In Proceedings ofthe 32nd International Joint Conference on Artificial Intelligence (IJCAI), 2023.

Ryan Bahlous-Boldi, Maxence Faldor, Luca Grillotti, Hannah Janmohamed, Lisa Coiffard, Lee Spector, and Antoine Cully. Dominated novelty search: Rethinking local competition in quality-diversity. arXiv preprint arXiv:2502.00593, 2025.

Antoine Cully and Yiannis Demiris. Quality and diversity optimization: A unifying modular framework. IEEE Transactions on Evolutionary Computation, 22(2):245–259, 2018.

Joel Lehman and Kenneth O. Stanley. Evolving a diversity of virtual creatures through novelty search and local competition. In Proceedings ofGECCO, pages 211–218, 2011.

Matthew C. Fontaine and Stefanos Nikolaidis. A quality diversity approach to automatically generating human– robot interaction scenarios in shared autonomy. In Robotics: Science and Systems XVII, 2021.

Steijn Kistemaker and Shimon Whiteson. Critical factors in the performance of novelty search. In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO), pages 965–972, 2011.

Jean-Baptiste Mouret and Jeff Clune. Illuminating search spaces by mapping elites. arXiv preprint arXiv:1504.04909, 2015.

Jørgen Nordmoen, Frank Veenstra, Kai Olav Ellefsen, and Kyrre Glette. MAP-Elites enables powerful stepping stones and diversity for modular robotics. Frontiers in Robotics and AI, 8:639173, 2021.

Justin K. Pugh, Lisa B. Soros, and Kenneth O. Stanley. Quality diversity: A new frontier for evolutionary computation. Frontiers in Robotics and AI, 3:40, 2016.

Vassilis Vassiliades, Konstantinos Chatzilygeroudis, and Jean-Baptiste Mouret. Using centroidal Voronoi tessellations to scale up the multidimensional archive of phenotypic elites algorithm. IEEE Transactions on Evolutionary Computation, 22(4):623–630, 2018.

Vassilis Vassiliades and Jean-Baptiste Mouret. Discovering the elite hypervolume by leveraging interspecies correlation. In Proceedings ofthe Genetic and Evolutionary Computation Conference (GECCO), 2018.

Konstantinos Sfikas, Antonios Liapis, and Georgios N. Yannakakis. Monte Carlo Elites: Quality-diversity selection as a multi-armed bandit problem. In Proceedings ofGECCO, 2021. doi:10.1145/3449639.3459321.

## A Supplementary Details

## A.1 Lineage-depth spread

One possible explanation for the capacity effect is that high turnover gives archive members little time to develop different histories. If their histories remain too similar, a history-based signal could appear close to chance even if the underlying idea is useful. That story requires spread and association to rise together, and they do not (Table 2). Depth IQR never falls below ≈13 over a 10,000-generation run, so the near-chance AUC at capacity 256 is not explained by absent variation. Values are read from per-generation archive traces and reported to two significant figures.

Table 2: Interquartile range of lineage depth among retained individuals.
<table><tr><td>Task</td><td>256</td><td>1024</td><td>4096</td></tr><tr><td>Omnidirectional Ant</td><td>≈94</td><td>≈32</td><td>≈16</td></tr><tr><td>Arm</td><td>≈16</td><td>≈15</td><td>≈13</td></tr><tr><td>Walker 2D</td><td>≈22</td><td>≈23</td><td>≈16</td></tr></table>

## A.2 Runs

All conditions use $k = 3 ,$ 256 offspring per generation, and Iso+LineDD variation (Vassiliades and Mouret, 2018) with $\sigma _ { \mathrm { i s o } } = 0 . 0 0 5$ and $\sigma _ { \mathrm { l i n e } } = 0 . 0 5$

Table 3: Experimental configuration. Seed counts differ by capacity in the measurement sweep.
<table><tr><td>Experiment</td><td>Domains</td><td>Capacity</td><td>Gens</td><td>Parent selection</td><td>Seeds</td></tr><tr><td rowspan="3">Signal measurement</td><td>all three</td><td>256</td><td>10,000</td><td>uniform</td><td>5</td></tr><tr><td>all three</td><td>1024</td><td>10,000</td><td>uniform</td><td>10</td></tr><tr><td>all three</td><td>4096</td><td>10,000</td><td>uniform</td><td>5</td></tr><tr><td rowspan="2">Fitness sweep</td><td>Omnidirectional Ant</td><td>1024</td><td>4,000</td><td>uniform,  $r \in \{ 0 . 1 , 0 . 5 \}$ </td><td>10</td></tr><tr><td>Omnidirectional Ant</td><td>4096</td><td>4,000</td><td> $\mathrm { { u n i f o r m } } , \tau \in \{ 0 . 1 , 0 . 5 \}$ </td><td>10</td></tr><tr><td>Depth sweep</td><td>Omnidirectional Ant</td><td>1024</td><td>10,000</td><td> $\pm \mathrm { d e p t h } , \tau \in \{ 0 . 5 , 2 . 0 \}$ </td><td>10</td></tr></table>

Two instrumentation limits apply. The Brax domains log the archive every tenth generation, so the $H = 1 0$ persistence horizon is undefined there and is reported only for Arm. Ancestry entropy is undefined at capacity 4096 on all three tasks for memory reasons, so that condition carries 11 of the 12 signals.

Domain specifications: Arm uses 128 joint angles in [0, 1] with an end-effector descriptor in $[ 0 , 1 ] ^ { 2 }$ and objective −std(angles); Omnidirectional Ant uses Brax spring physics, final position in $[ - 3 0 , \dot { 3 } 0 ] ^ { 2 }$ as descriptor, forward reward removed, horizon 1000; Walker 2D uses Brax spring physics, an MLP [16] policy, a feet-contact descriptor in $[ 0 , 1 ] ^ { 2 }$ , the standard Walker2d reward, and horizon 1000.

## A.3 Where the null comes from: departures from DNS’s ordering are costly

A companion sweep perturbs the survival rule rather than the parent-selection rule, on the same three domains and the same archive (capacity 1024, 256 offspring per generation, $k = 3 .$ , uniform parent selection; 10,000 generations and 10 seeds on Arm, 4,000 generations and 5 seeds on the Brax domains). The perturbation mixes a second notion of crowding, distance to a bounded record of previously evaluated individuals, into the competition score with weight λ. So $\lambda = 0$ recovers DNS exactly. The ladder is monotone decreasing on every domain, so the optimum is the endpoint rather than an interior mixture.

The ranking of rules replicates across domains (Kendall τ between domain rankings 0.81 to 0.90, with DNS first and λ = 1 last on all three). Within each domain, the Spearman correlation between a rule’s ordering of candidates and DNS’s ordering predicts its final QD-score: $r = 0 . 8 9$ on Arm, 0.95 on Omnidirectional Ant and 0.91 on Walker 2D, and 0.90 over all 21 domain-rule pairs. The further a rule moves the selection ordering, the more it loses.

The failure mode matches the one reported in the main text. Over all runs at the final generation, projected coverage correlates with projected QD-score at 0.998 on Omnidirectional Ant and

Table 4: Final projected QD-score as a percentage of unmodified DNS $( \lambda = 0 )$ . The ladder is monotone on every domain.
<table><tr><td>Domain</td><td> $\lambda = 0 . 2 5$ </td><td> $\lambda = 0 . 5 0$ </td><td> $\lambda = 1 . 0 0$ </td></tr><tr><td>Arm</td><td>98.9</td><td>96.0</td><td>66.0</td></tr><tr><td>Omnidirectional Ant</td><td>97.0</td><td>94.2</td><td>78.5</td></tr><tr><td>Walker 2D</td><td>95.8</td><td>94.6</td><td>60.9</td></tr></table>

0.954 on Walker 2D, while maximum fitness correlates at −0.39 and 0.06. On Arm the two are not independent evidence, since with the offset used there QD-score equals coverage to a relative error of $1 0 ^ { - 7 }$