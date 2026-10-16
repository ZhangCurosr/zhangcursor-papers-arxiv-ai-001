# Asking Smarter Questions: Targeted Active Preference Learning for Underspecified Rewards

Helena Merker, Nick Walker, Andreea Bobu Massachusetts Institute of Technology {hmerker, nswalker, abobu}@mit.edu

Abstract: Reward learning typically assumes the human’s input covers everything that matters for the task. In practice, humans can only attend to so much at once, so their teaching naturally captures some features—or aspects of the task—better than others. Prior work has shown how the robot can identify which features were poorly supervised directly from its teaching data. But the current solution is to ask the human to try again, putting the burden on someone who already struggled to convey those features. We propose shifting this burden to the robot. Our key insight is that the same information that tells the robot what it doesn’t know can help it ask better questions. Preference queries offer a low-effort channel for resolving this uncertainty, but only if the robot knows where to direct them and the human knows what to attend to when answering. We act on this insight in two ways: we build an informed prior that targets preference queries at the features that need resolving, and we accompany each query with an explanation that focuses the human’s attention on exactly those features. In simulation, the informed prior beats the standard active preference learning baseline by up to 30.8% in the lowquery regime, where reducing human effort matters most. In a user study, the informed prior and explanations each significantly improve reward recovery and amplify each other.

Keywords: Reward Learning from Human Feedback, Active Preference Learning

## 1 Introduction

When a human teaches a robot how to do a task, not every relevant aspect of the task—orfeature— gets equal attention. For instance, while teaching a robot how to carry a knife across your cluttered kitchen, you might do a good job avoiding the hot stovetop or the rack hanging overhead, but struggle to consistently keep the knife away from every person or obstacle. Because humans have limited working memory and can’t focus on everything at once [1, 2], some features get carefully attended to while others are left underspecified. Recent work shows how the robot can detect, directly from its teaching data, which features were poorly supervised or entirely irrelevant to the task [3, 4]. We go a step further and ask: how should the robot work with the human to resolve that underspecification?

One approach is to ask the human for more demonstrations focused on the underspecified features [3]. But this places the burden on the human, who already struggled with attending to those features. We propose to shift this burden onto the robot: rather than ask the human to try again, the robot should itself ask targeted questions that resolve what it is uncertain about.

Preference queries, where the robot generates behavior options and the human just picks their preferred one, offer this kind of low-effort channel [5, 6, 7]. Active learning approaches select queries to maximally reduce uncertainty under a distribution over the human’s reward [6]. But when that distribution is initialized from imperfect teaching data [8], it inherits not just what the feedback got right but also its errors on the underspecified features. The robot then spends its early queries correcting a misaligned distribution before it can make progress on what it actually needs to learn.

![](images/edc2861064ff3371c8e9892dda75d572455df01fd77aad4c2ad9fd889c07a1e8.jpg)  
Figure 1: A user teaches a robot to carry a mug of coffee but pays little attention to how close it gets to the human. Using prior work [3], the robot diagnoses from the demonstrations D which features were well-supervised (green), underspecified (red), or irrelevant (gray), and uses this partition o in two ways. (1) The robot shapes its belief $P ( w \mid D )$ (true weights marked by ▲) into an informed prior $P ( w \mid \mathbf { o } )$ : the robot stays confident about the laptop, doesn’t commit to anything about the human, and assumes the apple doesn’t matter. (2) The robot tells the user what to focus on. Since the robot is only unsure about distance to human, it asks about two trajectories that differ in exactly that, and each answer sharpens $P ( w \mid \mathcal { P } , \mathbf { o } )$ toward what it still needs to learn.

Our key insight is that the same information that tells the robot what it doesn’t know can help it ask better questions. The robot’s diagnosis of what past teaching missed serves two roles. First, we use it to build an informed prior over the human’s reward: a Gaussian that is confident on the well-supervised features, flat on the underspecified ones, and near zero on irrelevant ones. This ensures preference queries are automatically directed at what needs resolving. Second, we use it to supplement each query with an explanation that tells the human which features to focus on when answering, making the robot’s learning process transparent [9]. In Fig. 1, the robot communicates uncertainty about distance to human, and its query presents two options that vary along that feature. Together, these two components align what the robot asks with what the human attends to.

Our contributions are: (1) an informed prior for active preference learning that corrects what the teaching data got wrong, automatically targeting queries at the underspecified features; (2) an explanation mechanism that directs the human’s attention to those same features when answering, aligning what the robot asks with what the human attends to; and (3) empirical validation in two simulated 7-DoF manipulation domains and a user study, where both components independently improve reward recovery and amplify each other, with the largest gains in the low-query regime.

## 2 Related Work

Learning from Imperfect Teaching Reward learning from human feedback typically assumes the teacher is optimal or noisily optimal [10, 11], but human input is routinely suboptimal. Some methods build richer models of the human, capturing systematic or structured deviations from optimality [12, 13]. Others treat suboptimality as localized failures of supervision rather than uniform noise: the robot reasons about how well its hypothesis space explains the human’s input and flags what it cannot account for [14, 15, 16], or detects from the teaching data which features were wellsupervised versus underspecified [3, 4, 17]. Rather than resolving the diagnosed gap by requesting more data, we use the diagnosis to drive active preference queries, shifting resolution onto the robot.

Active Preference Elicitation Preference queries, which ask the human to choose between trajectories, are a low-effort channel for reward learning. The robot can select each query to maximize information gain, minimizing the comparisons the human must answer [6]. Prior work extends this loop in various ways: warm-starting the reward belief from demonstrations [8], enriching queries to ask which features make a trajectory preferable [7], and using language models to select queries and ease answering [18]. From Biyik et al. [8] we inherit the initialization of the reward belief from demonstrations. But rather than treating all features alike, we shape that belief to encode where teaching was reliable, letting the information-gain criterion target the underspecified features.

## 3 Problem Setup

We address the problem of learning a reward function from human input when the teaching signal leaves some features underspecified. Recent work shows that robots can detect underspecified features from statistical patterns in the teaching data [3]. We use this detection and ask: how should the robot resolve underspecified features? We propose using that knowledge in two ways: to correct the imperfect posterior into an informed prior, and to direct human attention during preference queries.

## 3.1 Preliminaries

The robot’s goal is to learn a reward function that captures how the human wants the task performed. We adopt a linear reward parameterization [10, 11] over state features. Let $\phi : \mathcal { S }  \mathbb { R } ^ { k }$ be a feature mapping that extracts a k-dimensional vector from each state $s \in S .$ , and let w $\in \mathbb { R } ^ { k }$ be the reward weight vector with $\| w \| _ { 2 } ~ \le ~ 1$ The per-state reward is $R _ { w } ( s ) = w ^ { \top } \phi ( s )$ , and the cumulative reward of a trajectory $\xi = ( s ^ { 0 } , s ^ { 1 } , \ldots , s ^ { T } ) \in \Xi$ aggregates state-level contributions $\begin{array} { r } { R _ { w } ( \xi ) = \sum _ { s ^ { t } \in \xi } w ^ { \top } \phi ( s ^ { t } ) = w ^ { \top } \phi ( \xi ) } \end{array}$ , where $\begin{array} { r } { \phi ( \xi ) = \sum _ { s ^ { t } \in \xi } \phi ( s ^ { t } ) } \end{array}$ accumulates features along the trajectory. We write $w ^ { \star }$ for the human’s true preference weights, which are unknown to the robot. The features ϕ may be hand-crafted [11], learned [19], or extracted from a large language model given a high-level task description [20]; we assume access to a candidate feature set.

## 3.2 An Imperfect Posterior

We assume the robot has access to a dataset D of human teaching data, and maintains a posterior $P ( w \mid \mathcal { D } ) \propto P ( \mathcal { D } \mid w ) P ( w )$ over the reward weights. In general, the likelihood $P ( \mathcal { D } \mid w )$ can be instantiated for any teaching modality ${ \mathcal { D } } ,$ including demonstrations [10, 11], preferences [5, 8], corrections [21], or natural language [22]. For exposition, we follow DemPref [8] and initialize the reward belief from Boltzmann-rational demonstrations [23, 24, 25], which we then refine with pref erence queries. The resulting posterior $P ( w \mid D )$ is rarely uniform in quality across the components of w. Teaching a task that requires attending to multiple objectives at once is cognitively demanding; humans inevitably focus on some features at the expense of others. The demonstrations are thus not driven by $w ^ { \star }$ equally across all features, leaving the posterior misleading along the underspecified dimensions. Prior work [3, 4] showed how the robot can detect, from D, which features were well optimized and which were underspecified. The candidate feature set may also contain task-irrelevant features, whose posterior reflects spurious correlations rather than the human’s intent.

Feature Partition From Merker et al. [3] we inherit the variance-based detection of underspecified features and the LLM-based filtering of task-irrelevant ones, which together partition the k features into three disjoint subsets, collectively denoted by the partition o: 1) $\phi ^ { o } \colon$ the task-relevant features that are well-optimized in $\mathcal { D } , 2 ) \phi ^ { u } \colon$ the task-relevant features that are underspecified, and $3 ) \phi ^ { \varepsilon } \colon$ the features irrelevant to the task. We write $w ^ { o } , w ^ { u } , w ^ { \varepsilon }$ for the blocks of w corresponding to $\phi ^ { o } , \phi ^ { u } , \phi ^ { \varepsilon }$

Refining the Prior with Active Feedback The partition o summarizes what D reveals: $P ( w \mid D )$ is trustworthy on $\phi ^ { o }$ , misleading on $\phi ^ { u }$ , and uninformative on $\phi ^ { \varepsilon }$ . Our goal is to elicit additional human feedback that sharpens our estimate of $w ^ { \star }$ along the underspecified dimensions. One approach is to passively request additional demonstrations of the features in $\phi ^ { u }$ , returning the burden of producing behavior that exercises those features to the human [3]. Instead, we adopt an active preference learning framework [6], where the robot itself generates queries to resolve its uncertainty.

Standard active preference learning [6] uses $P ( w \mid D )$ as the prior; however, its deficiencies in $\phi ^ { u }$ and $\phi ^ { \varepsilon }$ force early queries to first correct the prior before they can resolve the underspecified features. We use o in two ways to address this. First, we construct a corrected distribution $P ( w \mid \mathbf { o } )$ by modifying $P ( w \mid D )$ block-wise: retaining what D revealed about $\phi ^ { o }$ , flattening the prior on $\phi ^ { u }$ , and concentrating it near zero on $\phi ^ { \varepsilon }$ (see Sec. 4.1). We use $P ( w \mid \mathbf { o } )$ as the prior for active preference learning, directing the robot’s queries at its uncertainty along $\phi ^ { u }$ (Sec. 4.2). Second, we adapt the explanation of Merker et al. [3], reframed to instruct the human to attend to $\phi ^ { u }$ when answering queries, so that their responses inform the specific features that need resolving (Sec. 4.3).

## 4 Method

Our approach reuses the robot’s diagnosis of which features past teaching left underspecified to elicit targeted preference feedback. First, we construct an informed prior $P ( w \mid \mathbf { o } )$ that corrects $P ( w \mid D )$ block-wise, concentrating uncertainty on $\phi ^ { u }$ . Second, we pair each query with an explanation that focuses the human’s attention on $\phi ^ { u }$ when answering, so their responses inform those features.

## 4.1 The Informed Prior

We now describe how to construct $P ( w \mid \mathbf { o } )$ in place of the standard initialization from $P ( w \mid \mathcal { D } )$ We want a corrected version of $P ( w \mid D )$ that leaves $\phi ^ { o }$ untouched, is uninformative on $\phi ^ { u }$ , and concentrates near zero on $\phi ^ { \varepsilon }$ . This requires a parametric form for $P ( w \mid D )$ amenable to block-wise edits. We represent $P ( w \mid D )$ as a Gaussian, whose parameters we modify to obtain $P ( w \mid \mathbf { o } )$

Gaussian Approximation To approximate $P ( w \mid D )$ , we draw samples via Metropolis-Hastings with the unnormalized posterior $P ( \mathcal { D } \mid w ) P ( w ) [ 6 ]$ , and fit a Gaussian $\textstyle { \mathcal { N } } ( { \boldsymbol { \mu } } , { \boldsymbol { \Sigma } } )$ to them. This yields the best Gaussian summary of $P ( w \mid D )$ , i.e. the closest in KL divergence [26]. However, $\mathcal { N } ( \boldsymbol { \mu } , \boldsymbol { \Sigma } )$ inherits the deficiencies of $P ( w \mid D )$ : only the blocks $\mu ^ { o }$ and $\Sigma ^ { o o }$ for the well-optimized features $\phi ^ { o }$ are informative, while $\mu ^ { u }$ and $\Sigma ^ { u u }$ came from demonstrations where $\phi ^ { u }$ was underspecified, and $\mu ^ { \varepsilon }$ and $\sum ^ { \varepsilon \varepsilon }$ came from spurious correlations. The block-wise modification below corrects these.

Constructing the Informed Prior The informed prior $P ( w \mid \mathbf { o } )$ takes a different form on each of the three blocks of o. On $\phi ^ { o } , P ( w \mid \mathbf { o } )$ should preserve the information about $w ^ { \star , o }$ already contained in $P ( w \mid D )$ ), since the demonstrations are informative on these features. On $\phi ^ { u } , P ( w \mid \mathbf { o } )$ should be uninformative about $w ^ { \star , u }$ since the human did not attend to those features; this way, the preference queries in Sec. 4.2 are responsible for setting $w ^ { u }$ . On $\phi ^ { \varepsilon } , P ( w \mid \mathbf { o } )$ should concentrate $w ^ { \varepsilon }$ near zero.

We construct $P ( w \mid \mathbf { o } )$ as a Gaussian density on w:

$$
P ( w \mid \mathbf { o } ) \propto \exp \Bigl ( - \textstyle { \frac { 1 } { 2 } } ( w - \tilde { \mu } ) ^ { \top } \tilde { \Lambda } ( w - \tilde { \mu } ) \Bigr ) ,\tag{1}
$$

with mean $\tilde { \mu }$ and precision matrix $\tilde { \Lambda }$ obtained by per-block modification of $\mu$ and $\Sigma .$ The mean is $\tilde { \mu } ^ { o } = \mu ^ { o } , \tilde { \mu } ^ { u } = 0 , \tilde { \mu } ^ { \varepsilon } = 0$ , and the precision is block-diagonal across $\phi ^ { o } , \phi ^ { u } , \phi ^ { \varepsilon } \colon \tilde { \Lambda } ^ { o o } \ =$ $( \Sigma ^ { o o } ) ^ { - 1 } , \tilde { \Lambda } ^ { u u } = 0 , \tilde { \Lambda } ^ { \varepsilon \varepsilon } = \mathrm { d i a g } \big ( ( \lambda _ { l } ) _ { l \in \phi ^ { \varepsilon } } \big )$ . Setting $\tilde { \mu } ^ { o } = \mu ^ { o }$ and $\tilde { \Lambda } ^ { o o } = ( \Sigma ^ { o o } ) ^ { - 1 }$ carries over the demonstration-derived information about $w ^ { \star , o }$ from $P ( w \mid D )$ . Setting $\tilde { \Lambda } ^ { u u } = 0$ makes $P ( w \mid \mathbf { o } )$ flat in $w ^ { u }$ , so all information about $w ^ { u }$ comes from the preference queries. On $\phi ^ { \varepsilon }$ , we set $\tilde { \mu } ^ { \varepsilon } = 0$ and $\tilde { \Lambda } ^ { \varepsilon \varepsilon } = \mathrm { d i a g } ( ( \lambda _ { l } ) _ { l \in \phi ^ { \varepsilon } } )$ , where $\lambda _ { l } \ge 0$ sets the concentration strength for ϕ proportionally to how likely that feature is to be irrelevant. This concentrates $w ^ { \varepsilon }$ near zero, encoding the belief that $w ^ { \star , \varepsilon } \approx 0$ . In practice, we use an LLM queried with the task description to estimate per-feature irrelevance, and describe the full computation of $\lambda _ { l }$ in Appendix A.

## 4.2 Active Preference Learning

Pairwise preference queries are a low-effort channel for reward inference [5]: the robot presents two trajectories ${ Q _ { i } = \{ \xi _ { i } ^ { A } , \xi _ { i } ^ { B } \} }$ } and asks the human which they prefer, $q _ { i } \in Q _ { i }$ . The probability follows a Bradley-Terry model [27], which typically assumes the human responds purely based on which trajectory has higher reward, with no explicit model of attention. Inspired by the Boltzmann rationality model used for demonstrations [23, 3], we introduce a per-query, per-feature rationality vector $\beta _ { i } \ \in \ \mathbb { R } _ { > 0 } ^ { k }$ that controls how much each feature influences the human’s response. Each component $\beta _ { i , l }$ modulates the contribution of feature $\phi _ { l }$ on query i: setting $\beta _ { i , l } = 0$ removes ϕ<sub>l</sub> from the likelihood entirely, encoding that the human did not attend to that feature when answering, so the choice is carried out as if that feature did not exist. This per-query indexing lets the robot direct what the human attends to on each query. We describe how o sets the values of $\beta _ { i }$ in Sec. 4.3. The preference probability is then:

$$
P ( q _ { i } = \xi _ { i } ^ { j } \mid Q _ { i } , w , \beta _ { i } ) = \frac { \exp \left( \sum _ { l = 1 } ^ { k } \beta _ { i , l } w _ { l } \phi _ { l } ( \xi _ { i } ^ { j } ) \right) } { \exp \left( \sum _ { l = 1 } ^ { k } \beta _ { i , l } w _ { l } \phi _ { l } ( \xi _ { i } ^ { A } ) \right) + \exp \left( \sum _ { l = 1 } ^ { k } \beta _ { i , l } w _ { l } \phi _ { l } ( \xi _ { i } ^ { B } ) \right) } , \quad j \in \{ A , B \}\tag{2}
$$

The robot sets $\beta _ { i }$ based on the partition o before issuing each query, encoding an assumption about which features the human will attend to when answering (see Sec. 4.3). The data collected from K such queries is $\mathcal { P } = \{ ( Q _ { i } , q _ { i } ) \} _ { i = 1 } ^ { K }$ , and since $\beta _ { i }$ is determined by o, the full posterior is:

$$
P ( w \mid \mathcal { P } , \mathbf { o } ) \propto P ( \mathcal { P } \mid w , \mathbf { o } ) P ( w \mid \mathbf { o } ) = \left[ \prod _ { i = 1 } ^ { K } P ( q _ { i } \mid Q _ { i } , w , \beta _ { i } ) \right] P ( w \mid \mathbf { o } ) .\tag{3}
$$

Given this posterior, the robot must decide which query to ask next. At each step, it selects the query that maximally reduces uncertainty about w under the current posterior, i.e. the query that maximizes the mutual information between the human’s response and the reward weights [6]:

$$
Q _ { i } ^ { * } = \arg \operatorname* { m a x } _ { Q _ { i } } I \big ( w ; q _ { i } \mid Q _ { i } , \beta _ { i } \big ) = \arg \operatorname* { m a x } _ { Q _ { i } } H \big ( w \mid Q _ { i } , \beta _ { i } \big ) - \mathbb { E } _ { q _ { i } } \big [ H \big ( w \mid q _ { i } , Q _ { i } , \beta _ { i } \big ) \big ] ,\tag{4}
$$

where I is the mutual information and H denotes Shannon entropy [28]. We approximate the posterior with M samples $\Omega = \{ w _ { 1 } , \ldots , w _ { M } \}$ . Letting = denote asymptotic equality as $M \to \infty [ 6 ]$ the criterion becomes

$$
Q _ { i } ^ { * } \doteq \arg \operatorname* { m a x } _ { Q _ { i } } \frac { 1 } { M } \sum _ { q _ { i } \in Q _ { i } } \sum _ { w \in \Omega } P ( q _ { i } \mid Q _ { i } , w , \beta _ { i } ) \log _ { 2 } \frac { M \cdot P ( q _ { i } \mid Q _ { i } , w , \beta _ { i } ) } { \sum _ { w ^ { \prime } \in \Omega } P ( q _ { i } \mid Q _ { i } , w ^ { \prime } , \beta _ { i } ) } .\tag{5}
$$

Because the informed prior $P ( w \mid \mathbf { o } )$ is flat on $w ^ { u }$ and tightly constrained on $w ^ { o }$ and $w ^ { \varepsilon }$ , uncertainty concentrates along $w ^ { u }$ . The information gain criterion therefore automatically directs queries toward trajectories that differ along $\phi ^ { u }$ , which are the components we intend for the preferences to inform.

## 4.3 Directing Attention with Explanations

The second use of o is directing the person’s attention during preference queries. The partition o is a record of what past teaching revealed, i.e. which features were well-optimized, underspecified, or irrelevant. We now repurpose it as prospective guidance, telling the human which features to attend to when answering queries. In that sense, o determines $\beta _ { i }$ , the rationality vector that governs how the human’s response enters the likelihood. Without an explanation, the human is assumed to attend to every feature equally, corresponding to $\beta _ { i } = \beta _ { 0 } \mathbf { 1 }$ , with the same positive value $\beta _ { 0 }$ on every component. Alternatively, the robot can provide an explanation that directs the human’s attention to the underspecified features $\phi ^ { u } \left( \mathrm { e . g . } \right.$ “I’m uncertain about how to handle [features in $\phi ^ { u } ]$ . Please focus on these features when comparing the two trajectories.”). This is how o determines $\beta _ { i } { \mathrm { : } }$ : features in $\phi ^ { u }$ receive rationality $\beta _ { 0 }$ while $\phi ^ { o }$ and $\phi ^ { \varepsilon }$ are zeroed out, giving $\beta _ { i } ^ { u } = \beta _ { 0 } \mathbf { 1 }$ and $\beta _ { i } ^ { o } = \beta _ { i } ^ { \varepsilon } = 0$ . The Bradley-Terry probability in Equation 2—and hence the likelihood $P ( \mathcal P \mid w , \mathbf o )$ in Equation $^ { 3 - }$ then depends on w only through $w ^ { u }$ . The trajectories therefore enter the query only through their $\phi ^ { u }$ features, so a query whose two trajectories agree on those features produces no information at all.

## 5 Simulated Experiments

We evaluate our method in simulation, comparing the informed prior against four alternative priors, a targeted query objective, random querying, and three attention-directing explanation variants.

## 5.1 Experimental Setup

We use two scenes wherein a 7-DoF Jaco arm carries an object above a cluttered surface (Fig. 2a–b). In Stove, the robot carries a utensil across a workspace with a stove, a human, an overhead rack, and four distractors (sponge, banana, soup can, sugar box). The task-relevant features are: (1) stove, the xy-distance from the end-effector to the stove; (2) human, the xy-distance to the human; (3) point, the dot product between the end-effector’s pointing direction and the unit vector toward the human; and (4) rack, the z-distance below the rack. In Coffee, the robot carries a coffee cup across a workspace containing a laptop, a human, and four distractors (apple, bowl, fork, marker). The task-relevant features are: (1) laptop, the xy-distance to the laptop; (2) table, the z-distance of the end-effector above the table; (3) human, the xy-distance to the human; and $( 4 ) c o f f e e$ , the projection of the end-effector’s vertical axis onto the world up-axis, measuring how upright the cup remains. For each environment, we construct 50,000 trajectories from 5,000 random start-goal end-effector pairs, linearly interpolated in joint space and perturbed by 10 random smooth deformations each. Each trajectory has 21 waypoints encoding joint configurations and the xyz positions and rotations of all joints. We split the bank 60-20-20% into training, validation, and test sets. The simulated human draws its 10 initial demonstrations and all preference-query pairs from the training set.

![](images/2f4bddbb03a69878895f3a94b81f41c5b665d87cdbb6e2306e9304b15f3f8dd6.jpg)  
(a)

![](images/9b5bc0371dc8fa9b0c4ccf91b623ac79c307fb9db27bd6b77be5d8499a86a50c.jpg)  
(b)

![](images/993a5e9a2b4526b3139b6eaf8266a9c9513246f1a1fd69930831d967d7b89b83.jpg)  
(c)  
Figure 2: (a) Stove: the robot carries a utensil near a stove and a human. (b) Coffee: the robot carries a coffee cup near a laptop and a human. (c) A Franka Emika FR3 arm transports a knife in a kitchen.

We model the demonstrator as Boltzmann-rational with ground-truth weights $w ^ { \star }$ and a per-feature rationality vector $\beta \in \mathbb { R } _ { > 0 } ^ { k } ,$ . A high $\beta _ { l }$ produces demonstrations that reliably optimize ϕ , while $\beta _ { l } = 0$ simulates underspecification. Distractors are fixed to $w _ { l } ^ { \star } = 0$ . When answering queries without an explanation, the human uses $\beta _ { i } = \beta _ { 0 } \mathbf { 1 }$ , so responses follow the full $w ^ { \star }$ . When the robot provides an explanation directing attention to $\phi ^ { \mathrm { e x p } } \subseteq \phi ,$ , the human’s per-query rationality vector has nonzero entries only on the indices of $\phi ^ { \mathrm { e x p } }$ , so the response depends only on those features.

Methods (1) Informed Prior constructs $P ( w \mid \mathbf { o } )$ by block-wise modification of a Gaussian approximation of $P ( w \mid D )$ . The active query loop targets $P ( w \mid \mathcal { P } , \mathbf { o } )$ and selects queries by information gain. (2) Informed Prior + Informed Explanation further sets $\phi ^ { \mathrm { e x p } }$ to the predicted $\phi ^ { u }$ , so the human attends to the same features the prior leaves unconstrained. (3) Oracle Prior uses the true set of underspecified features. (4) Oracle Prior + Oracle Explanation further sets $\phi ^ { \mathrm { e x p } }$ to the true underspecified features. (5) Extra-Only Prior is a version of the informed prior that does not predict underspecified features. It combines the predicted irrelevant features $\phi ^ { \varepsilon }$ , represented as a diagonal Gaussian, with the demonstration likelihood. (6) Standard models posterior $P ( w \mid \mathcal { P } , \mathcal { D } )$ by combining the demonstration likelihood with a uniform prior over the unit ball and selects queries by information gain. (7) Standard + Oracle Explanation sets $\phi ^ { \mathrm { e x p } }$ to the true underspecified features, isolating the contribution of the explanation alone. (8) Targeted uses the unmodified $P ( w \mid D )$ and no explanation $( \beta _ { i } = \beta _ { 0 } \mathbf { 1 } )$ ), but selects queries by maximizing $I ( w ^ { u } ; q _ { i } \mid Q _ { i } , \beta _ { i } )$ over the predicted $\phi ^ { u }$ . (9) Gaussian Approximation is Standard but with the Gaussian approximation of $P ( w \mid D )$ indicating the penalty of summarizing the original posterior. (10) Random draws queries uniformly at random. We report normalized reward: the average ground-truth reward of the top-k test trajectories under the posterior mean, normalized by the range of ground-truth rewards across the test set. All experiments are repeated across 5 seeds, use $k = 2 0$ , and we report mean and standard error.

## 5.2 Results

The informed prior accelerates reward recovery across both environments and degrees of underspeci fication. We evaluate settings with one and then two underspecified task-relevant features, averaging over every choice of underspecified feature or pair of underspecified features (Fig. 3). Results for each individual configuration are provided in Fig. 5 of Appendix B. In both cases, the informed prior recovers the true reward faster than the baselines. The relative ordering of the methods is also stable across the single-feature and two-feature underspecification conditions (Fig. 3, left panels), with the largest gains generally appearing in the low-query regime $( K \le 1 0 )$ . This is practically significant as the early queries are the regime where reducing human effort matters most. Averaged over all 20 configurations, Informed Prior outperforms Standard by 13.0% at $K = 5$ and by 11.3% at $K = 1 0$ . The largest gap of 30.8% occurs at $K = 4$ in Stove when human and point are both underspecified (Fig. 5b). Extra-Only Prior, which concentrates only $w ^ { \varepsilon }$ near zero, improves on Standard but remains below Informed Prior. This suggests that much of the gain comes from removing the misleading demonstration signal on $w ^ { u }$ . Gaussian Approximation, which replaces $P ( w \mid D )$ with $\mathcal { N } ( \boldsymbol { \mu } , \boldsymbol { \Sigma } )$ , performs below Standard. Since Informed Prior builds on this same summary, its advantage stems from the block-wise modification of $\mathcal { N } ( \boldsymbol { \mu } , \boldsymbol { \Sigma } )$

![](images/f3bc8cef4cf952481b7c0fb4adc8db7deba1b4c6b165e165220e167ab7685f5b.jpg)  
a) Stove One Underspecified Feature

![](images/c72d45ca2bce8c422ebc1a1325283e7ce4a2062e17fe8481a0798fe249442787.jpg)

![](images/80144015e01301d0b44022b4e28853b62273d9c60b6aab515f6c6219ab10ffde.jpg)

b) Stove Two Underspecified Features  
![](images/f34c5f043afabdc2d3cdacf46cea4dd85783607051b6e9cddfd29b7219b878df.jpg)  
Figure 3: Average normalized reward versus number of preference queries in Stove (a, b) and $C o f f e e$ $( \mathrm { c } , \mathrm { d } )$ , with one (a, c) or two (b, d) features initially underspecified. Results are averaged over all four choices of the underspecified feature $( \mathrm { a } , \mathrm { c } )$ and all six pairs of underspecified features (b, d). Curves show the mean and shaded bands ±1 SE across seeds. Left panels compare methods without explanations; right panels compare explanation variants alongside their base methods.

Misidentified Underspecified Features When one or more well-optimized features are predicted as underspecified, there is a gap at $K = 0$ between Informed Prior and Oracle Prior $( \mathrm { e . g . }$ , table in Fig. 5c). This gap carries over into the averaged curves, as in Coffee with one underspecified feature (Fig. 3c). If a feature $\phi _ { l }$ is incorrectly deemed underspecified, its prior moves into the underspecified block, so its prior-anchored mean is zero despite $w _ { l } ^ { \star } \neq 0$ . This degrades the top-k selection until preferences supply enough information to substitute for the removed signal.

Effect of Explanations Informed Prior + Informed Exp and Oracle Prior + Oracle Exp track their non-explanation counterparts with two underspecified features (Fig. 3b, Fig. 3d), but peak at small K and then decline with one (Fig. 3a, Fig. 3c). Standard + Oracle Exp plateaus below Standard in all four settings. As preferences accumulate, the posterior of $w ^ { u }$ tightens and $\| w ^ { u } \|$ grows beyond the value of $0 ,$ causing $\| w ^ { o } \|$ to contract under the constraint $\| w \| _ { 2 } \leq 1 .$ . Learning from explanationguided queries quickly recovers information about $\phi ^ { u }$ , but collects no information about balancing against the originally optimized $\phi ^ { o }$ . For Standard + Oracle $E x p ,$ explanation-guided responses additionally carry no direct information about $w ^ { \varepsilon }$ , so the spurious weights in $P ( w \mid D )$ are never directly corrected. Adaptive schemes that broaden $\phi ^ { \mathrm { e x p } }$ as $w ^ { u }$ is resolved are left to future work.

## 6 User Study

We conducted a user study to evaluate whether generating queries under the informed prior and providing explanations to guide responses improves reward learning. We also examine whether the explanations improve participants’ subjective experience along three dimensions: ease of answering queries, understanding of their purpose, and expression of what the robot needs to learn.

(c) Robot study  
![](images/3743867105e76f1a9914fd1558b73543e8c3fc8e8f882493ccda8399e3802474.jpg)

![](images/3d38a9cc3285407645007e3a9b38efee8ddec768b41ce1fd6e92abc3059e3942.jpg)

![](images/c1a2408b03058e7e4ff80fd53a3ec47ec382aecf7e52858d4874ab2ad7110328.jpg)  
No explanation With explanation Uninformed prior Informed prior Informed Prior Oracle Prior Standard Random  
Figure 4: User study (a) normalized reward and (b) Likert score per condition. (c) Robot study mean normalized reward at K = 10 by feature initially underspecified. Error bars are ±1 SE.

## 6.1 Experiment Design

We conducted an IRB-approved study using the Coffee environment. The 2×2 within-subjects design crossed the prior (uninformed vs. informed) with the query delivery (no explanation vs. explanation), yielding four conditions: (1) Uninformed, (2) Informed, (3) Uninformed + Explanation and (4) Informed + Explanation. For each condition, participants were told the four task objectives: keep the mug upright, stay close to the table, stay away from the laptop, and stay away from the human. Participants were shown three trajectories chosen under the ground-truth reward, followed by 10 preference queries for each of two tasks, one with laptop initially underspecified and the other with human. The explanation conditions prefaced each task’s preference queries with a natural language explanation that named the underspecified feature. All conditions used the ground-truth partition o, leaving the prior and the explanation as the only factors that vary. Reward inference ran online: the posterior was re-sampled after every response, so queries adapted to each participant’s earlier answers. We rendered queries in simulation because executing arbitrary trajectories is unsafe without a filter [29], and with a filter the robot only shows acceptable behavior. Reward models calibrated on such data over-reward the behaviors humans reject [30]. Participants answered six 7-point Liker items after each task (see Appendix C), completed the NASA Task Load Index (NASA-TLX) [31] after each condition, and ranked the four conditions by preference at the end. We recruited $N = 1 2$ participants (9 female, 2 male, 1 preferred not to say; age $M = 2 3 . 8 , S D = 2 . 3 )$

Hypotheses We expected that: (H1) using the informed prior will lead to improved reward recovery; (H2) providing explanations to guide participant responses to queries will lead to improved reward recovery; explanations would make (H3) queries easier to answer, (H4) the purpose of queries more understandable, and (H5) participants feel they can better express to the robot what it needs to learn; and (H6) participants subjectively prefer having an explanation.

## 6.2 Results

Objective For each participant, condition, and task we computed normalized reward, and ran a 2×2 repeated-measures ANOVA on its value at $K \ = \ 1 0$ (averaged across the two tasks) with prior and explanation as within-subjects factors. We found a significant main effect of prior $( F ( 1 , 1 1 ) = 1 2 6 . 6 8 , p < 0 . 0 0 1 , \eta _ { p } ^ { 2 } = 0 . 9 2 0 )$ , supporting H1. We also found a significant main effect of explanation $( F ( 1 , 1 1 ) = 7 \dot { 1 } . 9 2 , p < 0 . 0 0 1 , \eta _ { o } ^ { 2 } = 0 . 8 6 7 )$ , supporting H2. The prior × explanation interaction was also significant and positive $( F ( 1 , 1 1 ) = 1 5 . 3 3 , p = 0 . 0 0 2 , \eta _ { p } ^ { 2 } = 0 . 5 8 2 )$ Fig. 4a plots the normalized reward by condition.

Subjective We ran a 2×2 repeated-measures ANOVA on each Likert scale with prior and explanation as within-subjects factors. The explanation main effect was significant on ease $( F ( 1 , 1 1 ) =$ $1 5 . 3 1 , p = 0 . 0 0 2 , \eta _ { p } ^ { 2 } = 0 . 5 8 2 )$ , supporting H3, and on understanding $( F ( 1 , 1 1 ) = 6 . 6 6 , p = 0 . 0 2 6$ $\eta _ { p } ^ { 2 } = 0 . 3 7 7 )$ , supporting H4, but not on expression $( F ( 1 , 1 1 ) = 2 . 9 6 , p = 0 . 1 1 3 , \eta _ { p } ^ { 2 } = 0 . 2 1 2 )$ yielding no support for H5. Fig. 4b plots the scores by condition. We also assessed perceived workload with a 2×2 repeated-measures ANOVA with prior and explanation as within-subjects factors.

Explanation significantly reduced perceived workload $( F ( 1 , 1 1 ) = 1 6 . 3 7 , p = 0 . 0 0 2 , \eta _ { p } ^ { 2 } = 0 . 5 9 8 )$ while prior had no effect on workload $( F ( 1 , 1 1 ) < 0 . 0 1 , p = 0 . 9 8 5 , \eta _ { \upsilon } ^ { 2 } < 0 . 0 0 1 )$ . A Friedman test on the rankings was not significant $( \chi ^ { 2 } ( 3 ) = 4 . 3 0 , p = 0 . 2 3 1 , N \mathit { \Omega } = 1 2 )$ , giving no support for H6 (mean ranks, where 1 is most preferred: $I n f o r m e d + E x p l a n a t i o n = 2 . 0 8$ , Uninformed + Explanation = 2.17, Uninformed = 2.75, Informed = 3.00). While some participants valued directed attention (P4 noted that “getting a cue about which feature to attend to is helpful,” and P6 found it “definitely easier to teach the robot when [they] knew what it was unclear about”), others felt constrained by focusing on a single dimension (P11 wasn’t “sure how much to prioritize [the feature] over the other factors,” and P8 instead “liked balancing the [features]” themselves). These responses suggest a trade-off: explanations help users focus their attention, but can conflict with a desire to balance multiple objectives, motivating attention schemes that broaden as features are resolved.

Demonstration-based Feedback Our approach assumes that preference queries resolve underspecification with less effort than additional demonstrations. To test this, we compared NASA-TLX responses from our study’s explanation conditions against existing data from Merker et al. [3], where a separate cohort of participants gave corrective kinesthetic demonstrations in response to naturallanguage explanations (N = 12 each, between subjects). Physical demand was lower for preference queries $( M = 3 . 8 , S E = 1 . 9 )$ than for kinesthetic demonstrations $( M = 3 1 . 7 , S E = 6 . 9 ;$ $t ( 1 2 . 6 ) = 3 . 8 9 , p = 0 . 0 0 2 )$ . Overall workload was also lower $( M = 1 7 . 4 \mathrm { v s . } M = 2 4 . 2 )$ , though not significantly $( t ( 2 2 . 0 ) = 1 . 3 1 , p = 0 . 2 0 5 )$ , and the gap shrinks once physical demand is excluded $( M = 2 0 . 1 \mathrm { v s . } M = 2 2 . 7 ; t ( 2 1 . 3 ) = 0 . 4 5 , p = 0 . 6 5 8 )$ . Preference queries thus remove the physical cost of kinesthetic teaching, while we found no evidence of increased cognitive workload.

## 7 Real Robot Examples

We use a Franka Emika FR3 arm carrying a knife in a kitchen (Fig. 2c). We model the kitchen in Py-Bullet to generate the trajectory bank and run reward learning offline. The task-relevant features are stove (xy-distance to cooktop), cat (xy-distance to cat), and point (how directly the knife’s extension axis points at the cat), alongside three distractors: xy-distances to a mug, lego, and rubik’s cube. The ground-truth reward keeps the knife clear of both the stove and the cat and avoids pointing the blade at the cat. We compare Informed Prior, Oracle Prior, Standard, and Random, reporting normalized reward at $K = 1 0 ( \mathrm { F i g . 4 c } )$ . Since normalized reward doesn’t reveal whether gaps between methods correspond to behavior a person wouldn’t like, we execute the top test trajectories on the physical arm to demonstrate that the learned rewards translate to sensible behavior (see Appendix D).

## 8 Conclusion

We present an approach for resolving underspecification in reward learning that shifts the burden from the human to the robot. Our insight is that the same information that indicates which features were underspecified can also be used to ask better questions about them. The robot uses this di agnosis to shape an informed prior for active preference learning and prefaces its queries with an explanation that directs the human’s attention to those features. In simulation and in a user study, the informed prior recovers the true reward faster than standard active preference learning.

Limitations The method takes the feature partition as input, so errors in the upstream diagnosis propagate into the informed prior. Updating the partition online as preferences arrive is a natural extension we leave to future work. We also summarize the demonstration-derived posterior as a Gaussian, discarding its higher-order structure. The attention scheme is similarly coarse: each query either directs the human’s attention exclusively to the underspecified features or gives no explanation at all. As features become resolved, the human’s attention should expand to allow later queries to also inform the well-optimized features. Softening the attention model is simple to simulate, but when and how much to broaden, and whether changing the instruction mid-task erodes ease and understanding, requires substantial human validation that we leave to future work. In addition, our experiments focused on tabletop manipulation with a small set of task-relevant features. Evaluating our approach on larger feature sets or dynamic objects would further test robustness and generality.

## Acknowledgments

This work has taken place in the Collaborative Learning and Autonomy Research Lab (CLEAR Lab) at MIT. CLEAR research is supported in part by the National Science Foundation (OIA-2535195) and the MIT Schwarzman College of Computing’s Social and Ethical Responsibilities of Computing program. Any opinions, findings, and conclusions expressed in this material are those of the authors and do not necessarily reflect the views of the sponsors.

## References

[1] J. Sweller. Cognitive load during problem solving: Effects on learning. Cognitive Science, 12 (2):257–285, 1988. DOI: 10.1207/s15516709cog1202 4.

[2] A. Baddeley. Working memory. Science, 255(5044):556–559, 1992. DOI: 10.1126/science.1736359.

[3] H. Merker, N. Walker, and A. Bobu. Robots that know what to ask: Recovering misaligned rewards through targeted explanations. In Proceedings of Robotics: Science and Systems, Sydney, Australia, 2026. DOI: 10.15607/RSS.2026.XXII.116.

[4] A. Peng, A. Netanyahu, M. K. Ho, T. Shu, A. Bobu, J. Shah, and P. Agrawal. Diagnosis, feedback, adaptation: A human-in-the-loop framework for test-time policy adaptation. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings ofMachine Learning Research, pages 27630–27641. PMLR, 2023. URL https://proceedings.mlr.press/v202/peng23c.html.

[5] D. Sadigh, A. D. Dragan, S. Sastry, and S. A. Seshia. Active preference-based learning of reward functions. In Robotics: Science and Systems XIII, Massachusetts Institute ofTechnology, Cambridge, Massachusetts, USA, July 12-16, 2017, 2017. DOI: 10.15607/RSS.2017.XIII.053.

[6] E. Biyik, M. Palan, N. C. Landolfi, D. P. Losey, and D. Sadigh. Asking easy questions: A userfriendly approach to active reward learning. In 3rd Annual Conference on Robot Learning, CoRL 2019, Osaka, Japan, October 30 - November 1, 2019, Proceedings, Proceedings of Machine Learning Research, pages 1177–1190. PMLR, 2019. URL http://proceedings. mlr.press/v100/b-iy-ik20a.html.

[7] A. Peng, Y. Sun, T. Shu, and D. Abel. Pragmatic feature preferences: Learning rewardrelevant preferences from human input. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, Proceedings of Machine Learning Research, pages 40258–40270. PMLR, 2024. URL https://proceedings.mlr.press/ v235/peng24d.html.

[8] E. Biyik, D. P. Losey, M. Palan, N. C. Landolfi, G. Shevchuk, and D. Sadigh. Learning reward functions from diverse sources of human feedback: Optimally integrating demonstrations and preferences. Int. J. Robotics Res., 41(1):45–67, 2022. DOI: 10.1177/02783649211041652.

[9] S. Habibian, A. Jonnavittula, and D. P. Losey. Here’s What I’ve Learned: asking questions that reveal reward learning. ACM Trans. Hum. Robot Interact., 11(4):40:1–40:28, 2022. DOI: 10.1145/3526107.

[10] P. Abbeel and A. Y. Ng. Apprenticeship learning via inverse reinforcement learning. In Machine Learning, Proceedings ofthe Twenty-first International Conference (ICML 2004), Banff, Alberta, Canada, July 4-8, 2004, volume 69 of ACM International Conference Proceeding Series. ACM, 2004. DOI: 10.1145/1015330.1015430.

[11] B. D. Ziebart, A. L. Maas, J. A. Bagnell, A. K. Dey, et al. Maximum entropy inverse reinforcement learning. In AAAI, volume 8, pages 1433–1438. Chicago, IL, USA, 2008.

[12] C. Laidlaw and A. D. Dragan. The boltzmann policy distribution: Accounting for systematic suboptimality in human models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum?id=\_l\_QjPGN5ye.

[13] R. Shah, N. Gundotra, P. Abbeel, and A. D. Dragan. On the feasibility of learning, rather than assuming, human biases for reward inference. In Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, Proceedings of Machine Learning Research, pages 5670–5679. PMLR, 2019. URL http: //proceedings.mlr.press/v97/shah19a.html.

[14] A. Bobu, A. Bajcsy, J. F. Fisac, and A. D. Dragan. Learning under misspecified objective spaces. In 2nd Annual Conference on Robot Learning, CoRL 2018, Zurich, Switzerland, 29-¨ 31 October 2018, Proceedings, Proceedings of Machine Learning Research, pages 796–805. PMLR, 2018. URL http://proceedings.mlr.press/v87/bobu18a.html.

[15] A. Bobu, A. Bajcsy, J. F. Fisac, S. Deglurkar, and A. D. Dragan. Quantifying hypothesis space misspecification in learning from human-robot demonstrations and physical corrections. IEEE Trans. Robotics, 36(3):835–854, 2020. DOI: 10.1109/TRO.2020.2971415.

[16] M. Zurek, A. Bobu, D. S. Brown, and A. D. Dragan. Situational confidence assistance for lifelong shared autonomy. In International Conference on Robotics and Automation (ICRA), 2021.

[17] I. Lourenc¸o, A. Bobu, C. R. Rojas, and B. Wahlberg. Diagnosing and repairing feature representations under distribution shifts. In 62nd IEEE Conference on Decision and Control, CDC 2023, Singapore, December 13-15, 2023, pages 3638–3645. IEEE, 2023. DOI: 10.1109/CDC49753.2023.10383644. URL https://doi.org/10.1109/CDC49753.2023. 10383644.

[18] S. Mahmud, M. Nakamura, and S. Zilberstein. MAPLE: A framework for active preference learning guided by large language models. In Thirty-Ninth AAAI Conference on Artificial Intelligence, Thirty-Seventh Conference on Innovative Applications of Artificial Intelligence, Fifteenth Symposium on Educational Advances in Artificial Intelligence, AAAI 2025, Philadelphia, PA, USA, February 25 - March 4, 2025, pages 27518–27528. AAAI Press, 2025. DOI: 10.1609/AAAI.V39I26.34964.

[19] A. Bobu, M. Wiggert, C. J. Tomlin, and A. D. Dragan. Inducing structure in reward learning by learning features. The International Journal of Robotics Research, 41(5):497–518, 2022. DOI: 10.1177/02783649221078031.

[20] A. Peng, B. Z. Li, I. Sucholutsky, N. Kumar, J. Shah, J. Andreas, and A. Bobu. Adaptive language-guided abstraction from contrastive explanations. In Conference on Robot Learning, 6-9 November 2024, Munich, Germany, volume 270 of Proceedings of Machine Learning Research, pages 3425–3438. PMLR, 2024. URL https://proceedings.mlr.press/v270/ peng25c.html.

[21] A. Bajcsy, D. P. Losey, M. K. O’Malley, and A. D. Dragan. Learning robot objectives from physical human interaction. In 1st Annual Conference on Robot Learning, CoRL 2017, Mountain View, California, USA, November 13-15, 2017, Proceedings, volume 78 of Proceedings of Machine Learning Research, pages 217–226. PMLR, 2017. URL http://proceedings. mlr.press/v78/bajcsy17a.html.

[22] T. R. Sumers, M. K. Ho, R. X. D. Hawkins, K. Narasimhan, and T. L. Griffiths. Learning rewards from linguistic feedback. In Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications of Artificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021, pages 6002–6010. AAAI Press, 2021. DOI: 10.1609/AAAI.V35I7.16749.

[23] E. T. Jaynes. Information theory and statistical mechanics. Phys. Rev., 106:620–630, May 1957. DOI: 10.1103/PhysRev.106.620.

[24] J. Von Neumann and O. Morgenstern. Theory of games and economic behavior. Princeton University Press Princeton, NJ, 1945.

[25] C. L. Baker, J. B. Tenenbaum, and R. R. Saxe. Goal inference as inverse planning. In Proceedings ofthe Annual Meeting ofthe Cognitive Science Society, volume 29, 2007.

[26] T. P. Minka. Expectation propagation for approximate bayesian inference. In UAI ’01: Proceedings of the 17th Conference in Uncertainty in Artificial Intelligence, University of Washington, Seattle, Washington, USA, August 2-5, 2001, pages 362–369. Morgan Kaufmann, 2001.

[27] R. A. Bradley and M. E. Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952. ISSN 00063444, 14643510. URL http://www.jstor.org/stable/2334029.

[28] T. M. Cover and J. A. Thomas. Elements of Information Theory. John Wiley & Sons, 2012.

[29] S. A. Mehta and D. P. Losey. Unified learning from demonstrations, corrections, and preferences during physical human-robot interaction. ACM Trans. Hum. Robot Interact., 13(3): 39:1–39:25, 2024. DOI: 10.1145/3623384.

[30] R. Tian, Y. Wu, and A. Bajcsy. Position: Good embodied reward models need bad behavior data. In Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea, 2026. arXiv:2606.01036.

[31] S. G. Hart and L. E. Staveland. Development of NASA-TLX (task load index): Results of empirical and theoretical research. Advances in Psychology, 52:139–183, 1988. DOI: 10.1016/S0166-4115(08)62386-9.

[32] H. Singmann, B. Bolker, J. Westfall, F. Aust, and M. S. Ben-Shachar. afex: Analysis of Factorial Experiments, 2025. URL https://CRAN.R-project.org/package=afex. R package version 1.5-1.

[33] R. V. Lenth and J. Piaskowski. emmeans: Estimated Marginal Means, aka Least-Squares Means, 2026. URL https://CRAN.R-project.org/package=emmeans. R package version 2.0.3.

## A Experimental Details

Identifying Irrelevant Features The candidate feature set ϕ is not guaranteed to be task-relevant in its entirety. Features that have no bearing on the task carry no signal in D, so the demonstrationderived posterior along these dimensions is shaped not by the human’s intent but by spurious correlations in the data. We use an LLM to identify these features. For each candidate feature $\phi _ { l }$ , we issue N independent queries to the LLM. Each query supplies the task description and the full candidate feature set, and asks the model to label $\phi _ { l }$ as either relevant or irrelevant. If a majority of the $N$ responses vote irrelevant, we assign $\phi _ { l }$ to $\phi ^ { \varepsilon }$ and record the LLM’s confidence as $c _ { l } \in [ 0 . 5 , 1 ]$ , the fraction of the $N$ responses that voted irrelevant.

Sec. 4.1 defines the prior’s precision on $\phi ^ { \varepsilon }$ as $\tilde { \Lambda } ^ { \varepsilon \varepsilon } = \mathrm { d i a g } \big ( ( \lambda _ { l } ) _ { l \in \phi ^ { \varepsilon } } \big )$ , so each $\lambda _ { l } \geq 0$ controls how tightly the prior pulls $w _ { l }$ toward zero. $\lambda _ { l }$ should match the concentration of $P ( w \mid D )$ on $w ^ { o }$ when the LLM is fully confident that $\phi _ { l }$ is irrelevant, and weaken when the LLM is less confident. We set $\lambda _ { l } = c _ { l } \tilde { \tau }$ , where $\tilde { \tau } = \mathrm { m e d i a n } \{ 1 / { \Sigma _ { l l } } : l \in \phi ^ { o } \}$ is the median precision among the well-optimized features and $\Sigma _ { l l }$ is the diagonal entry of the Gaussian summary $\mathcal { N } ( \mu , \Sigma ) \approx P ( w \mid \mathcal { D } )$ . The scalar τ˜ sets a default concentration strength comparable to the prior’s concentration on $w ^ { o }$ , and the factor c<sub>l</sub> scales this strength down when the LLM is less confident that $\phi _ { l }$ is irrelevant. We take $\tilde { \Lambda } ^ { \varepsilon \varepsilon }$ to be diagonal because the LLM evaluates each feature independently, providing no signal about cross-feature correlations within $\phi ^ { \varepsilon }$

LLM Prompts We use OpenAI’s gpt-5.2 with a structured JSON output schema and reasoning effort set to high. The prompts for the Stove, Coffee, and Kitchen environments are shown below. For each candidate feature $\phi _ { l } ,$ , we issue the corresponding environment’s prompt $N = 1 0$ times, with {feature} and {feature description} replaced by the name and short description of $\phi _ { l }$

## Stove prompt

The following is an instruction for a task that the agent needs to complete. instruction = A 7- DoF Jaco robot arm is carrying a utensil above a tabletop environment. The arm is carrying a utensil with a sharp, pointed tip. Various objects are on the table, including a stove, a sponge, a banana, a tomato soup can, and a sugar box. A human is right next to the table. A ceiling rack is right above the table.

The current available feature names are: human (distance from end-effector to human) stove (distance from end-effector to stove) point (how much the utensil held by the end-effector points toward the human) rack (z-distance of the end-effector below the ceiling rack) sponge (distance from end-effector to sponge) banana (distance from end-effector to banana) soup (distance from end-effector to tomato soup can) sugar (distance from end-effector to sugar box)

We are evaluating whether {feature} ({feature description}) is part of the set of features that a typical human would consider necessary for this task.

Please return either a 1 or 0 for the following feature: feature = {feature} ({feature description}).

A feature should be labeled 1 only if a typical human would expect that ignoring it would commonly lead to a clearly undesirable outcome in this specific task. If it is unclear whether the feature would meaningfully affect behavior in this context, then it should be labeled 0.

Please reason step by step internally.

Provide your answer as ONLY JSON in one of these two forms: {”label”: 0} or {”label”: 1}.   
Do not include any surrounding explanation or text.

## Coffee prompt

The following is an instruction for a task that the agent needs to complete. instruction = A 7- DoF Jaco robot arm is carrying a coffee cup above a tabletop environment. The arm is carrying a fragile ceramic coffee cup that is at risk of slipping. Various objects are on the table. A human is right next to the table.

The current available feature names are: laptop (distance from end-effector to laptop) table (height of end-effector above table surface) human (distance from end-effector to human) coffee (coffee cup orientation (tilt from upright)) apple (distance from end-effector to apple) bowl (distance from end-effector to bowl) fork (distance from end-effector to fork) marker (distance from end-effector to marker)

We are evaluating whether {feature} ({feature description}) is part of the set of features that a typical human would consider necessary for this task.

Please return either a 1 or 0 for the following feature: feature = {feature}   
({feature description}).

A feature should be labeled 1 only if a typical human would expect that ignoring it would commonly lead to a clearly undesirable outcome in this specific task. If it is unclear whether the feature would meaningfully affect behavior in this context, then it should be labeled 0.

Please reason step by step internally.

Provide your answer as ONLY JSON in one of these two forms: {”label”: 0} or {”label”: 1}.   
Do not include any surrounding explanation or text.

## Kitchen prompt

The following is an instruction for a task that the agent needs to complete. instruction = A 7-DoF Franka robot arm is carrying a utensil in a kitchen environment. The end-effector is carrying a knife with a sharp, pointed tip. On the counter there is a stove with burners and a live cat sitting nearby. Above the counter there are cabinets and a microwave. Also on the counter is a red mug, a lego block, and a rubik’s cube.

The current available feature names are: cat (distance from end-effector to a live cat) stove (distance from end-effector to stove) point (how much the knife held by the end-effector points toward the live cat on the counter) mug (distance from end-effector to a mug) lego (distance from end-effector to a lego block) rubiks (distance from end-effector to a rubik’s cube)

We are evaluating whether {feature} ({feature description}) is part of the set of features that a typical human would consider necessary for this task.

Please return either a 1 or 0 for the following feature: feature = {feature} ({feature description}).

A feature should be labeled 1 only if a typical human would expect that ignoring it would commonly lead to a clearly undesirable outcome in this specific task. If it is unclear whether the feature would meaningfully affect behavior in this context, then it should be labeled 0.

Please reason step by step internally.

Provide your answer as ONLY JSON in one of these two forms: {”label”: 0} or {”label”: 1}.   
Do not include any surrounding explanation or text.

## B Per-Configuration Simulation Results

We report each of the 20 configurations individually in Fig. 5, complementing the averaged results in Fig. 3. Each environment contributes four configurations with a single underspecified feature (a, c) and six with a pair of underspecified features (b, d). Curves show the mean and the shaded regions the standard error across 5 seeds.

![](images/483ab3cb5c57a2a42aac373707768176f1dc1420871e8e1b0ebf0e78527f69da.jpg)  
Figure 5: Normalized reward with (a, c) one underspecified feature and (b, d) two underspecified features for the Stove (a, b) and Coffee (c, d) environments. Across conditions, the informed prior recovers the true reward faster than the baselines, with the largest gains in the low-query regime.

## C User Study Details

Procedure After providing informed consent, participants were familiarized with the task and the preference-query interface. They were told that the robot was learning to carry a mug of coffee across a cluttered table and that it had not yet learned to do so correctly. Four task objectives were explicitly stated: (1) keeping the mug upright, (2) staying close to the table, (3) staying away from the laptop, and (4) staying away from the person. These objectives were repeated before each condition. Participants did not provide any demonstrations: the underspecified teaching signal that began each task was generated by a simulated Boltzmann-rational demonstrator (rationality $\beta ^ { u } = 0$ on the task’s underspecified feature) and held fixed across all participants and conditions using a single random seed, so that every participant began each task from an identical reward belief and contributed only preference responses.

The study crossed the prior (uninformed vs. informed) with the query delivery (no explanation vs. explanation), yielding four conditions: Uninformed, Informed, Uninformed + Explanation, and Informed + Explanation. Each condition began with a short familiarization phase in which the participant watched three example trajectories drawn under the ground-truth reward. The participant then completed 2 tasks, one with laptop initially underspecified and one with human initially underspecified, each consisting of 10 preference queries. The two framings were:

• No explanation (Uninformed, Informed): “In each query you will see two trajectories side by side. Click ‘Prefer A’ or ‘Prefer B’ for the one you prefer.”

• Explanation (Uninformed + Explanation, Informed + Explanation): the same instructions, followed by “The robot is uncertain about how to handle [feature]. Please focus on this feature when comparing the two trajectories,” where [feature] was “distance to the laptop” or “distance to the human” depending on the task.

Preference queries were presented through a PyBullet interface that played the two candidate trajectories side by side (Fig. 6). Participants chose between them with “Prefer A (left)” and “Prefer B (right)” buttons, and could re-watch both trajectories as often as they wished with a “Replay” button. Condition order was counterbalanced with a Williams Latin square of size four. The order of the two tasks within each condition was independently counterbalanced so that, within every condition, the participants split evenly between laptop-first and human-first. After each task, participants answered the six 7-point Likert items in Table 1; after each condition, they completed the NASA Task Load Index (NASA-TLX) [31]; and at the end of the session, they ranked the four conditions from most to least preferred.

Table 1: ANOVA results for the main effect of explanation on Likert scales.
<table><tr><td>Scale</td><td>Questions</td><td>Cronbach&#x27;s α</td><td>F</td><td>p</td></tr><tr><td>Ease</td><td>I found it easy to choose between the two trajectories in each query. The queries were difficult to answer.</td><td>0.94</td><td>15.31</td><td>0.002</td></tr><tr><td>Under.</td><td>I understood what each query was trying to learn. The queries seemed random or unmotivated.</td><td>0.83</td><td>6.66</td><td>0.026</td></tr><tr><td>Expr.</td><td>The queries covered important parts of the task. Queries were wasted on things that didn&#x27;t matter.</td><td>0.75</td><td>2.96</td><td>0.113</td></tr></table>

Analysis We ran every test in R. We analyzed the objective measure with a 2×2 repeated-measures ANOVA (afex [32]) with prior and explanation as within-subjects factors and partial eta-squared $( \eta _ { p } ^ { 2 } )$ as the effect-size measure. We supplemented the ANOVA with Tukey-corrected pairwise contrasts among the four conditions, computed on the estimated marginal means (emmeans [33]) with within-subjects Cohen’s $d _ { z }$ . These contrasts showed that Informed + Explanation significantly outperformed Uninformed $( p < 0 . 0 0 1 , d _ { z } = 5 . 8 4 )$ , and that Informed $( p = 0 . 0 0 3 , d _ { z } = 1 . 3 6 )$ and Uninformed + Explanation $( p = 0 . 0 0 1 , d _ { z } = 1 . 5 5 )$ each significantly outperformed Uninformed, indicating that the prior and the explanation each contribute measurable gains and that combining them produces the largest gain. Informed and Uninformed + Explanation did not differ from each other $( p = 0 . 9 7 6 )$ , suggesting that the prior alone and the explanation alone provide comparable benefit over baseline.

For the subjective measures, each negatively-worded Likert item was reverse-coded and averaged with its positively-worded partner to form the three two-item scales (ease, understanding, and expression). Each scale was analyzed with its own $2 \times 2$ repeated-measures ANOVA. Beyond the explanation main effects in Table 1, the prior main effect was not significant for any scale (ease: $F ( 1 , 1 1 ) = 2 . 2 0 , p = 0 . 1 6 6 ;$ understanding: $F ( 1 , 1 1 ) = 0 . 8 3 , p = 0 . 3 8 2 ;$ ; expression: $F ( 1 , 1 1 ) = 1 . 0 7 , p = 0 . 3 2 3 )$ , and the prior × explanation interaction was significant only for ease $( F ( 1 , 1 1 ) = 6 . 6 7 , p = 0 . 0 2 5 , \eta _ { p } ^ { 2 } = 0 . 3 7 8 ,$ ; understanding: $F ( 1 , 1 1 ) < 0 . 0 1 , p = 0 . 9 6 1$ ; expression: $F ( 1 , 1 1 ) = 0 . 1 7 , p = 0 . 6 9 0 )$

Perceived workload was measured once per condition and summarized as the unweighted mean of the six subscales on a 0–100 scale. We used a $2 \times 2$ repeated-measures ANOVA and supplemented it with Tukey-corrected pairwise contrasts on the estimated marginal means. These contrasts localized the workload reduction to the two explanation conditions. Uninformed + Explanation $( M = 1 7 . 6 $ $p = 0 . 0 4 4 , d _ { z } = 0 . 8 9 )$ and Informed + Explanation $( M = 1 7 . 2 , p = 0 . 0 4 5 , d _ { z } = 0 . 8 9 )$ each significantly lowered workload relative to Uninformed $( M = 2 4 . 7 )$ , whereas Informed did not $( M =$ $2 5 . 1 , p = 0 . 9 9 9 , d _ { z } = - 0 . 0 5 )$

![](images/8cd5efe9c20ff62c018f540af13daaba910ab70fd7a06ac35d44ecd18a282e4c.jpg)  
Figure 6: An example preference query from the user study in the Coffee environment: participants are shown two trajectories side by side and select the one they prefer.

## D Real Robot Execution

We validate that the learned rewards translate into sensible physical behavior by executing the top trajectories on the Franka Emika FR3 arm (Fig. 2c). The trajectory bank is generated entirely within a PyBullet model of the kitchen (Fig. 7) and all reward learning is performed offline; only the final selected trajectories are run on hardware. For each of the four methods (Informed Prior, Oracle Prior, Standard, and Random) and each of the three underspecification conditions (cat, stove, or point underspecified), we take the top-ranked test trajectories after K = 10 queries and execute them on the physical arm. Both Informed Prior and Oracle Prior are visually consistent across all three underspecification conditions. Whether cat, stove, or point was initially underspecified, the top trajectories steer the end-effector away from both the cat and the stove while keeping the knife pointed away from the cat. The Standard baseline mirrors Informed Prior when point was underspecified. However, when cat was underspecified, the top trajectories carry the knife substantially closer to the cat; when stove was un-

![](images/d9544066a2badb91f66f01e3c95c7a5451c1616258a3fd4f72df331305068b1e.jpg)  
Figure 7: The simulated model of the kitchen used to generate the trajectory bank for the real robot demos. A box is used as a visual placeholder for the cat in the actual environment.

derspecified, they likewise route the end-effector closer to the stove (Fig. 8). In neither case did the preference queries recover the clearance that the demonstrations had missed. The Random baseline follows the same trend: visually similar to Informed Prior when point was underspecified, yet routing the knife closer to whichever of cat or stove was initially underspecified.

The similar performance of all four methods when point is underspecified is partly a consequence of the trajectory bank, which contains a wide range of orientations. Since point is only weakly correlated with the distance features and has the widest spread of any feature, trajectories vary widely along point, and comparisons that differ along it are abundant. Trajectories that stay clear of both the cat and the stove at once are less common in the bank, so comparisons that would teach the robot to balance the two are rarer. This is a property of our kitchen, whose reachable workspace is small in xy; a larger scene would leave more room to stay clear of both objects at once. In addition, although the stove-underspecified condition shows a large gap in normalized reward between methods (Fig. 4c), the visible difference between some of the trajectories (Fig. 8) is less pronounced than this gap suggests. The stove feature scores the end-effector’s distance to a single reference point (cooktop center). Because the stove itself is a large object, distance to that point is only a loose proxy for how close the gripper actually appears to the stove. The end-effector can pass near the stove’s physical edge while still registering a sizable distance to its center.

Informed Prior, cat  
![](images/d41f0d71690ce88909e3e532a415d4ff76d101c909049bdedde6b61693daca57.jpg)

![](images/f66b7b6207562c12e24a73bdb4b3a0a268cf1c5397803b5d39b0627366524cdc.jpg)

![](images/85107dd0ecb2658dfdd049c8f324883857dd7c7d3073d8a1f686c89975c5b7bc.jpg)

Standard, cat  
![](images/5d1358fbf477e6e6cf7a66c795650797bd6080609f0f9746d5821d4326626ca4.jpg)

![](images/9da829f57aaf157d231ddfa7d45b7883db4009271fc93bc85a3e653506b17a67.jpg)

![](images/5f91ec4a77c1933250f328be9a43807d9a9688d3e4b2ac20a8ae136393e11917.jpg)

Informed Prior, stove  
![](images/2ae65117a7d57ebad910e6026c1926b023e2eb22dab048e823a9e9c8282bb6ea.jpg)

![](images/7a40da3fc9258ff7bd823a4ce2b20b115ca73f175c5a1378c19de49b0ad34c9f.jpg)

![](images/5ff20e34b524b7794b1b804131e6f172fdb387ebf0f323f59e01eed8922f14cc.jpg)

Standard, stove  
![](images/e4665ef47694e01aa438a6aee19187d7218926dab16ad8299262c1d03c525234.jpg)

![](images/a36d15851cf286cdf2c0ebabdcda5402324fef990160777348d0c2bbe56fc0cd.jpg)

![](images/f8623607f207b37f930e0368e6139baa360abc90cc911116bd56c691ed655c61.jpg)  
Figure 8: The three highest-ranked test trajectories executed on the Franka Emika FR3 arm after K = 10 preference queries, for Informed Prior and Standard when cat (top two rows) or stove (bottom two rows) was initially underspecified. Each panel traces the path of the knife in orange.