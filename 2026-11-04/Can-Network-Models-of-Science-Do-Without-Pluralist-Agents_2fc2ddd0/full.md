# Can Network Models of Science Do Without Pluralist Agents?

Jonas Hallgren<sup>1,2</sup>, David Hyland<sup>3</sup>, and Mahault Albarracin<sup>4,5</sup>

<sup>1</sup> Equilibria Network, Stockholm, Sweden

<sup>2</sup> Uppsala University, Uppsala, Sweden

<sup>3</sup> University of Oxford, Oxford, UK

<sup>4</sup> Laboratoire d’Analyse Cognitive de l’Information, Université du Québec à Montréal, Québec, Canada

<sup>5</sup> Centre of Excellence for AI and Robotics, Shefield Hallam University, Shefield, UK

Abstract. Science is a process of useful models being proven wrong time and again, but despite its many flaws, it somehow works. Why is this? Paul Feyerabend argued that it works because people dare to hold and defend incompatible rival theories against the accepted one, since a theory’s failures often lead to its abandonment in the presence of a more accurate or simpler alternative. He called this view theoretical pluralism. We introduce a theoretical and experimental model in which this can be examined. It compares communities of pluralist agents, which can hold rival theories side by side and maintain beliefs within and between those theories, with communities of monist agents, which can only incorporate what they hear or observe into a single model of their own. In a multi-agent variational inference setting, our experimental demonstrations illustrate how communities with rival theories can go through periods of uncertainty followed by a shift in the theories they hold true, how the information that is communicated between communities afects beliefs when those communities make diferent observations about the world, and how sharing support for theories and sharing beliefs within them contribute diferently to collective belief dynamics.

Keywords: Active inference · Model comparison · Belief sharing · Paradigm change · Social epistemology

## 1 Introduction

If we are to make progress in science, we need to understand the role that diversity of thought plays in it. The question becomes more pressing as Large Language Models (LLMs) are increasingly integrated into the scientific enterprise [19]. How do we know that they help the scientific process, rather than leading us down locally valid paths whose underlying theoretical assumptions are fundamentally wrong, and what representation of the world would AI systems require to avoid this?

In this paper, we introduce the idea of epistemically pluralist agents, inspired by theoretical pluralism. This is partly inspired by the work of Paul Feyerabend, who argued that the failures of an accepted theory often show only against an incompatible alternative, so a community that works with one theory at a time hides its own anomalies [4, 9]. One of the practical implications that Feyerabend argued for was that scientists and scientific communities should cultivate and preserve competing theories in order to make sure that we have many ways of viewing and explaining the same phenomena [8]. We can contrast this with its opposite view, theoretical monism, where a community works within one theory until incompatible evidence evidence forces it out, as in Kuhn’s normal science [17]. Which of the two serves a community better has mostly been argued from historical cases [6], since the question is hard to test directly on real science. This is one reason to turn to models.

In the last two decades there has been progress on the front of modelling the dynamics of scientific communities [21, 25, 27]. Network-epistemology models study how beliefs about competing options change when agents exchange evidence [16, 25–27]. Opinion-dynamics models also examine communication of numerical estimates [7, 12], while active-inference models can represent structured latent states, learning, and belief sharing [1, 5, 10]. Our focus is on the distinction between communicating a conditional belief within a candidate theory and communicating support for that theory relative to its rivals. In order to model this, we require a rich enough representation to describe whether a theory is right or not, and we need to consider the structure of the theories themselves [13, 23, 24].

The central distinction we make concerns two levels of belief. On one hand, an agent may be uncertain about the values predicted by a particular theory, and on the other hand, they may be separately uncertain about which theory to favour. We model pluralist agents as those who retain both a conditional belief state for each candidate theory and support between candidates. In contrast, we model monist agents as those who entertain only one belief distribution over observable quantities. One of the central questions we study is what changes when the two levels of belief can be communicated separately.

What happens if we introduce structure to the models with Bayes nets [22]? In our model, each agent entertains a set of rival candidate theories, each one a Bayes net over the same observations, together with a belief within each theory and a weight for how much it favours each of them. It can share either the beliefs or the weights with its neighbours. These are our epistemically pluralist agents, and we compare them with monist agents that see the same evidence but only hold one model. We study in our experiments how features such as the observations that agents make and the way that they share information influence the dynamics of individual and collective belief updating. We examine this by looking at a classic philosophy of science example, namely theories of combustion, which has been explored in the philosophy of science literature [2, 6, 18, 24]. When the evidence turns against the accepted theory, pluralist agents go through a period of uncertainty and then switch theory, while monist agents only shift their estimates, with no sign that their explanation has changed. Sharing which theory one favours is what spreads a change of theory, while sharing beliefs within a theory mostly improves predictions. Groups that see diferent parts of the world can also converge on diferent theories while still communicating with each other. Going back to the question we started with, our results suggest that an AI system that is to help science, rather than narrow ${ \mathrm { i t } } ,$ should be able to hold more than one theory at a time and communicate beliefs about those theories in addition to beliefs within candidate theories.

## 2 Belief Representation and Updating

We start by introducing our model in the general setting and then describe experiments using a motivating example based on a competition between the late 18th century theory of phlogiston and the oxygen theory of combustion.

## 2.1 Beliefs within and between theories

We model a population of N agents making observations about a set of random variables and communicating synchronously on a fixed network over a series of rounds that occur sequentially through time. In each round, indexed by $t = 0 , \ldots , T - 1$ , all agents make an observation at the start of the round, communicate information pertaining to their beliefs within and across theories, evaluate the new observation, and then update their beliefs, with state t + 1 including observation t. Agents are indexed by $i , j \in \{ 1 , \ldots , N \}$ , and hold K candidate theories, indexed by $k \in \{ 1 , \ldots , K \}$ , which represent diferent probabilistic relationships between the random variables. We use the terms candidate and theory interchangeably in this paper. Theory k represents a vector of represented quantities $x _ { k } \in \mathcal X _ { k }$ , where $\mathcal { X } _ { k }$ is theory $k \mathrm { { s } }$ state space. Some coordinates may be observable and others latent, and diferent theories may use diferent latent coordinates. At the beginning of round $t ,$ agent i maintains two objects for each candidate: a probability $q _ { i k , t }$ assigned to the candidate, and a probability density $\rho _ { i k , t } ( x _ { k } )$ over the quantities represented within that candidate. We parameterize the candidate probabilities by unnormalized log support $a _ { i k , t } \in \mathbb { R }$

$$
q _ { i k , t } = \frac { \exp { ( a _ { i k , t } ) } } { \sum _ { k ^ { \prime } = 1 } ^ { K } \exp { ( a _ { i k ^ { \prime } , t } ) } } ,\tag{1}
$$

for each theory k. Thus, $q _ { i k , t }$ represents which theory the agent supports, whereas $\rho _ { i k , \ast }$ represents what values the agent assigns to the quantities represented by theory k. The collection $\{ q _ { i k , t } , \rho _ { i k , t } \} _ { k = 1 } ^ { K }$ constitutes agent i’s belief state at round t. The real numbers $a _ { i k , t }$ store each theory’s log weights. Each distribution $\rho _ { i k , t }$ has a fixed reference $\rho _ { i k } ^ { 0 }$ which serves as a default anchor toward which beliefs relax through forgetting.

## 2.2 Environments, observations, and theories

Let S denote the environmental state space and $\xi _ { t } \in \cal S$ denote the state of the environment at round t. Let O denote the observation space. The environment

![](images/161507dfdbe19c3475dc75329fd93450ee4625828b7465ed862f4bdf76038fc5.jpg)  
Fig. 1: (a) Private observations generated from a common world state whose likelihoods are evaluated under diferent candidate theories; (b) Agent i retains a conditional belief distribution $\rho _ { i k }$ over each candidate’s quantities and one model probability $q _ { i k }$ for that candidate; (c) Edges on a communication graph represent the exchange of these beliefs.

generates observations according to

$$
o _ { i , t } \sim p _ { \star } ( o \mid \xi _ { t } ) ,\tag{2}
$$

where $p _ { \star } : { \mathcal { S } }  { \varDelta ( \mathcal { O } ) }$ denotes the data-generating process. The environmental state $\xi _ { t }$ is distinct from the candidate-specific quantity vector $x _ { k }$ , in that $\xi _ { t }$ describes the system generating the observations, whereas $x _ { k }$ describes how candidate theory k represents and explains those observations.

Each theory specifies its own likelihood $\mathcal { L } _ { i k , t } ( o \mid x _ { k } )$ . The theory need not represent the environment in the same coordinates as $\xi _ { t } ,$ , and its likelihood need not coincide with the true data-generating process $p _ { \star }$

## 2.3 Communication, prediction, and belief-updating

We denote the communication matrices for within-theory beliefs and model support by $W _ { t } ^ { x }$ and ${ W } _ { t } ^ { m }$ , respectively. Their entries are nonnegative, each row sums to one, and entry $i j$ represents sender $j ^ { \prime }$ s influence on receiver i over conditional beliefs and model weights. Quantities with bars are used to denote beliefs after communication within round t, and are given by

$$
\bar { \rho } _ { i k , t } ( \boldsymbol { x } _ { k } ) \propto \prod _ { j = 1 } ^ { N } \rho _ { j k , t } ( \boldsymbol { x } _ { k } ) ^ { W _ { i j , t } ^ { x } } , \qquad \bar { a } _ { i k , t } = ( 1 - \alpha _ { m } ) a _ { i k , t } + \alpha _ { m } \sum _ { j = 1 } ^ { N } W _ { i j , t } ^ { m } a _ { j k , t } .\tag{3}
$$

The first operation normalizes the weighted product of beliefs within each candidate, forming a logarithmic opinion pool [11]. The second is a log-linear sociallearning rule [20]. The parameter $\alpha _ { m } \in [ 0 , 1 ]$ parameterizes model-weight sharing, and the sharing of conditional beliefs uses $W _ { t } ^ { x }$ directly. Setting $W _ { t } ^ { x } = I _ { N }$ (where $I _ { n }$ is the n-dimensional identity matrix) disables conditional sharing, and setting $\alpha _ { m } = 0$ disables model-weight sharing.

Each agent i evaluates observation $y _ { i , t }$ using a normalized likelihood $\mathcal { L } _ { i k , t } ( y \mid x _ { k } )$ which can incorporate assigned observation weights. Its predictive log score and subsequent update are given by

$$
\ell _ { i k , t } = \log \int _ { \mathcal { X } _ { k } } \mathcal { L } _ { i k , t } ( y _ { i , t } \mid x _ { k } ) \bar { \rho } _ { i k , t } ( x _ { k } ) d x _ { k } ,\tag{4}
$$

$$
\rho _ { i k , t } ^ { + } ( x _ { k } ) = e ^ { - \ell _ { i k , t } } \mathcal { L } _ { i k , t } ( y _ { i , t } \mid x _ { k } ) \bar { \rho } _ { i k , t } ( x _ { k } ) , \qquad a _ { i k , t } ^ { + } = \bar { a } _ { i k , t } + \ell _ { i k , t } ,\tag{5}
$$

where the superscript + denotes the learned state before reference anchoring is applied. Scoring precedes learning from the same observation, in that a theory can accommodate an observation while receiving a low score for its earlier prediction.

We model belief updating as a process of minimizing variational free energy over theory distributions $\varrho ,$ which is given by

$$
\mathcal { F } _ { i k , t } [ \varrho ] = \underbrace { D _ { \mathrm { K L } } [ \varrho \| \bar { \rho } _ { i k , t } ] } _ { \mathrm { c o m p l e x i t y } } - \underbrace { \mathbb { E } _ { \varrho } \log \mathcal { L } _ { i k , t } ( y _ { i , t } \mid x _ { k } ) } _ { \mathrm { a c c u r a c y } } ,\tag{6}
$$

where $D _ { \mathrm { K L } } [ \cdot \parallel \cdot ]$ is the Kullback–Leibler divergence. Thus, we write an agent i’s initial belief update (before reference anchoring) as

$$
\rho _ { i k , t } ^ { + } = \arg \operatorname* { m i n } _ { \varrho } \mathcal { F } _ { i k , t } [ \varrho ] .\tag{7}
$$

## 2.4 Reference anchoring

We include reference anchoring to represent the persistence of background theoretical commitments. New observations modify an agent’s beliefs, but those beliefs remain partly constrained by the assumptions encoded in the candidate theory’s reference distribution. To model this, we introduce a retention parameter which controls how quickly accumulated evidence can move beliefs away from that reference. Specifically, after seeing the current observation, the agent partially relaxes its within-theory belief toward the fixed reference distribution $\rho _ { i k } ^ { 0 }$ for that theory according to

$$
\rho _ { i k , t + 1 } ( x _ { k } ) \propto [ \rho _ { i k } ^ { 0 } ( x _ { k } ) ] ^ { 1 - \omega } [ \rho _ { i k , t } ^ { + } ( x _ { k } ) ] ^ { \omega } , \qquad a _ { i k , t + 1 } = a _ { i k , t } ^ { + } , \quad 0 \leq \omega \leq 1 ,\tag{8}
$$

where ω is a retention parameter. At $\omega = 1$ , the learned belief is retained without relaxation and at $\omega = 0$ , it is reset to the reference. Because this operation follows the current observation update, $\omega < 1$ attenuates both previously accumulated information and the information supplied by the current observation. Information learned from observation t can afect another agent through communication beginning in round t + 1.

## 3 Experimental setup

We base our experiments on a simplified model of two theories of combustion from the late 18th century. This motivating example draws on historical debates about the nature of combustion, whose theories and development were more varied than this simplified pair of models [2, 3, 14]. In particular, the phlogiston theory of combustion postulated that flammable objects contain a substance known as phlogiston, which is released upon combustion and absorbed by the surrounding air. According to this theory, substances should lose mass after combustion due to the loss of phlogiston, which made it dificult to accommodate the observation that some substances actually gained mass after combustion. This observation was an important factor that contributed to the discovery of oxygen’s role in combustion. In our experiments, we use these two theories and associated experimental observations as variables in the networks that agents maintain beliefs over.

## 3.1 Theory explanations and observations

The experiments use multivariate normal (Gaussian) conditional beliefs, described by a mean and covariance. Two diferent theories share observables for combustion extent $C _ { i }$ net gas uptake $G ,$ and solid mass change $M .$ . The ‘release’ account introduces an unobserved quantity $P$ for hypothetical phlogiston release, and the ‘oxygen’ account introduces unobserved oxygen incorporation $O$

In the experiments, both theory spaces are four-dimensional, $\mathrm { i . e . , } \mathcal { X } _ { k } = \mathbb { R } ^ { 4 }$ . Their represented state vectors are

$$
\begin{array} { r } { x _ { \mathrm { P } } = ( \zeta _ { \mathrm { P } } , C , G , M ) ^ { \top } , \qquad x _ { \mathrm { O } } = ( \zeta _ { \mathrm { O } } , C , G , M ) ^ { \top } . } \end{array}
$$

The theory graphs, coeficients, and reference distributions are fixed through out an experiment. Agents do not learn graph topology or structural coeficients, but only update beliefs about the quantities represented within each theory and update their relative support for each candidate.

The reference observable means are $( 1 , 1 , - 1 ) ^ { \top }$ for the phlogiston theory and $( 1 , 1 , 1 ) ^ { \top }$ for the oxygen theory, and the theory-specific latent means are zero. All agents begin with their working beliefs equal to these references and with equal probabilities over the two theories.

In the experiments, the environmental state is specialized to the three common observable quantities,

$$
\xi _ { t } = ( C _ { t } , G _ { t } , M _ { t } ) ^ { \top } \in \mathbb { R } ^ { 3 } ,
$$

and observations are noisy measurements of this state:

$$
o _ { i , t } = \xi _ { t } + \epsilon _ { i , t } , \qquad \epsilon _ { i , t } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I _ { 3 } ) .\tag{9}
$$

The state $\xi _ { t }$ is constant in the selective-observation comparison and changes during the evidence-shift experiments. Attention weights $w _ { i , t } \in [ 0 , 1 ] ^ { 3 }$ specify the

precision assigned to each channel. With $H = \left[ \boldsymbol { 0 } _ { 3 \times 1 } ~ \boldsymbol { I } _ { 3 } \right]$ and $S _ { i , t } = \mathrm { d i a g } ( \sqrt { w _ { i , t } } )$ agents evaluate

$$
y _ { i , t } = S _ { i , t } o _ { i , t } , \qquad \mathcal { L } _ { i k , t } ( y \mid x _ { k } ) = \mathcal { N } ( y ; S _ { i , t } H x _ { k } , \sigma ^ { 2 } I _ { 3 } ) .\tag{10}
$$

A positive channel weight w assigns efective channel variance $\sigma ^ { 2 } / w$ . These weights change the working likelihood, but not the physical sampling distribu tion. Appendix A.3 gives the Gaussian update.

## 3.2 Experimental comparisons

The experiments study how access to evidence, communication, and the representation of alternatives afect belief revision. We distinguish within-theory sharing, in which agents exchange beliefs about the quantities represented by each candidate, from model-support sharing, in which agents exchange their relative confidences in the competing theories. The former communicates what a theory predicts assuming that it is correct, whereas the latter communicates the evidence accumulated in its favour. We examine these in three comparisons (settings in Appendix B).

Revision with full observation. Sixteen agents communicate on a complete graph and initially observe the signal $( C , G , M ) = ( 1 , 1 , - 1 )$ . The mass component then changes sign, giving (1, 1, 1), and we follow the transition from phlogiston to oxygen $\left( \mathrm { F i g . ~ 2 } \right)$

Partial observation and competing theories. The state is held fixed at (1, 1, 1). Eight agents observe gas uptake and eight observe mass change, with withintheory sharing at total weight $\beta = 0 . 0 5$ across groups (Fig. 3). We compare cases where agents keep their model support private against sharing it, as well as a baseline control where all agents maintain a single joint Gaussian belief over combustion, gas uptake, and mass change. In all cases, agents share their conditional beliefs, i.e., their predictions and uncertainties under each theory that they represent. In the two cases where agents share their model supports or keep them private, the agents maintain separate Gaussian beliefs over the observables for each candidate theory.

Communication channels after changed evidence. We return to the evidencechange scenario from the first experiment on a complete graph, with four fully observing agents and twenty restricted agents that receive only weak gas uptake and mass change evidence (Fig. 4). The four conditions share the same history and observation streams and difer only in which communication channels remain enabled after the signal changes. This demonstrates which messages spread support for the oxygen theory and whether the resulting theory choices also improve prediction.

## 3.3 Measured outcomes

We find it useful to distinguish theory choice, numerical disagreement, and predictive accuracy. Theory choice is measured by the probability $q _ { i \mathrm { O } , t }$ weight assigned to the oxygen theory. Group means summarize collective theory support, whereas individual probabilities illustrate how support varies across agents.

Mean individual model entropy describes uncertainty between theories. It is low when agents strongly favour one theory and high when each agent assigns appreciable probability to both theories. It therefore describes a diferent quantity from disagreement between confident agents.

Numerical disagreement is the distance between the groups’ predicted means on the common observable coordinates (C, G, M). For candidate models, these predictions take an average over theories using their current probabilities, and for the Gaussian controls, they use the single Gaussian mean. Diferent theory choices can yield similar predictions, and similar theory choices can coexist with diferent numerical estimates.

Predictive accuracy is evaluated over all three observable coordinates, including those receiving little or no weight during learning. In particular, mean squared error (MSE) compares the predicted mean with the generating signal for the evaluated state $\xi _ { t - 1 }$ . Negative log-likelihood (NLL) evaluates the complete predictive distribution on independent held-out observations. Lower values indicate better prediction. Appendix C gives the formulae, reporting windows, and treatment of repeated simulation runs.

## 4 Results

## 4.1 Delayed belief revision after changed evidence

With full observation, the community first comes to favour the phlogiston theory under the negative mass signal, then revises toward the oxygen theory after the signal changes. Revision is delayed, in that the new observations must alter the support accumulated during the earlier regime. The model distinguishes this change in preferred explanation from the continuing adjustment of beliefs within each candidate.

In this setting, we observe that individual entropy rises while support moves between the theories and falls as agents become confident again. This identifies a period of uncertainty within agents, which is analogous to a period of ‘cri sis’ in science. With two theories, such a rise accompanies movement through intermediate candidate theory probabilities.

## 4.2 Theory choice and disagreement under selective observation

Diferent access to the same world can sustain opposing theory preferences even while agents communicate. With within-theory sharing and private model support, gas observers favour the phlogiston theory and mass observers favour the oxygen theory (Fig. 3b). Sharing model support instead leads both groups to favour oxygen. Thus, communicating what each theory predicts does not necessarily align preferences between theories.

![](images/ec940da9783c0edc67c46922d2cca953bd91df2c04e9dba454d0c19f89eef28d.jpg)  
Fig. 2: A demonstration of theory revision under full observation of observable quantities (extent of combustion, net gas uptake, and solid mass change). Mean oxygen probability (green) and mean individual candidate entropy (blue) after the mass signal changes. The dotted line marks the first changed observation at round 80, and the bands represent pointwise 95 percentile bootstrap confidence intervals for means across 50 simulation runs, using 2,000 resamples of whole runs.

The single-Gaussian control highlights what the distance between group predictions measures. This control retains diferences between the groups’ numerical predictions (Fig. 3a), despite having no probability distribution over alternative theories. This demonstrates that a numerical gap between the two communities’ predictions does not by itself establish opposition between theories, and a larger gap does not imply better prediction.

We also observe from Fig. 3b that the support of mass observers for the oxygen theory is virtually unafected by whether model support is communicated over the network, whereas the beliefs of gas observers in the oxygen theory depend critically on the model support information shared by the mass observers. This demonstrates that sharing predictions and uncertainties alone is not suficient for the agents observing strong evidence in favour of oxygen to influence the gas-observing agents to favour the oxygen theory.

![](images/bb2c9631f836aee25630890651d4afe3c90c4d3a609777d562d05971fcc3f3a3.jpg)  
Fig. 3: Theory choice and numerical prediction under selective observation. All conditions use the same 50 observation streams, $\beta = 0 . 0 5 ,$ , and $\omega = 0 . 3 .$ . (a) Distance between the groups’ observable predictive means, where theory predictions are averaged using their current probabilities. (b) Oxygen support in the gas observer group (solid) and mass observer group (dotted), for only the models that maintain beliefs over both theories.

## 4.3 Sharing model support spreads revision after changed evidence

Having distinguished numerical disagreement from opposing theory preferences under selective observation, we next examine how the two communication channels contribute to revision after a change in evidence. In particular, the final experiment holds the complete graph fixed and changes which messages agents exchange after the signal changes (Fig. 4). When model support is shared, either alone or alongside within-theory beliefs, the restricted group’s mean oxygen prob ability approaches 1 over the reported window. Neither condition with private model support reaches that threshold within the simulated horizon, indicating a slower rate of updating in favour of the oxygen theory. Direct communication of accumulated support therefore enables revision in these runs among agents receiving only weak evidence of the change.

Within-theory sharing makes a further contribution on top of model-support sharing. In particular, combined with model-support sharing, it brings the restricted group to believe in the oxygen theory sooner and reduces its late-window MSE relative to sharing model support alone. Communicating model support changes which explanation is favoured, while communicating conditional beliefs also changes the predictions associated with that explanation.

The efects are not uniform across outcomes or observation groups. Within theory sharing alone modestly improves restricted agents’ late-window prediction relative to no communication, while increasing their beliefs in the oxygen theory even more slowly than in the case where no information is shared. Fully observing agents instead have slightly higher late-window MSE when within-theory sharing is enabled. Theory adoption and predictive improvement are therefore distinct outcomes, and the value of a communication channel depends on the evidence available to its recipients.

![](images/a61ce080c24502b97042896a2b45e08183f4ac20ffafaec5da7def9fc75dbdb9.jpg)

![](images/4c8eefedfeec0e4dce01c301f1678a2043dfe53cced6cd311c27e461aaeea2a3.jpg)

![](images/8ea518c6c096d842c304181c16b83e1e0a0dc2c91216356e2e290e4b1805e497.jpg)  
Fig. 4: Communication channels after changed evidence on a complete graph. The four conditions begin from the same fully observed, fully communicating state after 80 updates and use the same subsequent observation streams. (a,b) Support for the oxygen theory among four fully observing and twenty restricted agents. The dotted line marks the common state before the first changed observation. (c) Full-signal MSE (i.e., the average squared error in an agent’s predictions across all three observable quantities) is averaged over completed states 451–600, by observation group. Restricted weights are (1, 0.01, 0.01), $\omega = 0 . 9$ , and $\sigma = 1$

## 5 Discussion

Can network models of science do without pluralist agents? Our initial results suggest that the answer is no. A monist agent can only share its estimates, whereas a pluralist agent can also share which theory it favours, and these two messages give rise to diferent belief dynamics. Sharing estimates improved agents’ predictions, but agents with weak evidence only switched theory when others told them which theory they favoured. Knowing what a theory predicts does not tell you whether to adopt it, and a model with monist agents cannot see that diference. Pluralist agents can also register when their explanation changes. The local observation update shifts relative support toward the theory assigning the observation the higher predictive probability. A monist agent has no such state, since it can only shift its estimates. This is close to Feyerabend’s point that some failures of accepted theories failures become apparent through comparison with alternatives [9].

More experiments are needed to understand under what conditions this remains true and how it interfaces more with existing network epistemological theories. In particular, we have not yet tested how much of a theory’s internal structure matters, as opposed to simply holding several theories. The present comparisons do not test the benefit of retaining a currently disfavoured theory, as both theories remain available throughout, and agents do not choose experiments. A natural next step is to let agents choose their experiments and examine how network structure afects discovery and long-term predictive performance when several theories remain available [26].

Several features also limit the interpretation of these demonstrations. The candidate theories, observation schedules, and attention weights are supplied in advance. Agents do not invent alternatives, revise their graphs, choose which experiments to conduct, or selectively report/update their beliefs based on other motivational considerations [15]. These results therefore do not isolate a benefit of explicit latent graphs, or conclusively establish that pluralism improves scientific discovery. Rather, they demonstrate how retaining separate candidates makes theory support distinguishable from numerical adjustment of beliefs.

The way that agents communicate theories to each other matters, and so does the way that they represent these theories. Telling others which theory you favour changes which theory others favour, as when a few informed agents brought the rest over to the oxygen theory, while telling them what you believe within a theory mostly improves their predictions.

What this means for science in the age of modern AI systems such as LLMs is an open question, as we do not understand the extent to which they are able to represent and maintain diverse theories. An LLM that folds everything it observes into one account could behave like a monist agent, able to refine a theory but not to notice when a rival explains the data better. An LLM that summarizes what theories predict, without keeping track of which theory the evidence favours, would share estimates but never verdicts. We believe that this needs to be studied and understood, so that science can remain a functioning endeavour with many disagreements, even with automated scientists.

## Acknowledgements

The work of D.H. was supported by a UKRI AI World Leading Researcher Fellowship (grant number EP/W002949/1) awarded to Michael Wooldridge.

## Bibliography

[1] Albarracin, M., Demekas, D., Ramstead, M.J.D., Heins, C.: Epistemic communities under active inference. Entropy 24(4), 476 (2022). https: //doi.org/10.3390/e24040476

[2] Blumenthal, G., Ladyman, J.: The development of problems within the phlogiston theories, 1766–1791. Foundations of Chemistry 19(3), 241–280 (2017). https://doi.org/10.1007/s10698-017-9289-0

[3] Boantza, V.D., Gal, O.: The ‘absolute existence’ of phlogiston: the losing party’s point of view. The British Journal for the History of Science 44(3), 317–342 (2011). https://doi.org/10.1017/S000708741000155X

[4] Bschir, K.: Feyerabend and Popper on theory proliferation and anomaly import: on the compatibility of theoretical pluralism and critical rationalism. HOPOS: The Journal of the International Society for the History of Philosophy of Science 5(1), 24–55 (2015). https://doi.org/10.1086/680368

[5] Çatal, O., Van de Maele, T., Pitliya, R.J., Albarracin, M., Pattisapu, C., Verbelen, T.: Belief sharing: a blessing or a curse. In: Buckley, C.L., et al. (eds.) Active Inference: IWAI 2024, Communications in Computer and Information Science, vol. 2193, pp. 121–133. Springer (2025). https: //doi.org/10.1007/978-3-031-77138-5\_8

[6] Chang, H.: Is Water H<sub>2</sub>O? Evidence, Realism and Pluralism, Boston Studies in the Philosophy and History of Science, vol. 293. Springer (2012). https: //doi.org/10.1007/978-94-007-3932-1

[7] DeGroot, M.H.: Reaching a consensus. Journal of the American Statistical Association 69(345), 118–121 (1974). https://doi.org/10.1080/016214 59.1974.10480137

[8] Feyerabend, P.K.: Consolations for the specialist. In: Lakatos, I., Musgrave, A. (eds.) Criticism and the Growth of Knowledge, pp. 197–230. Cambridge University Press (1970). https://doi.org/10.1017/CBO9781139171434 .010

[9] Feyerabend, P.K.: Against Method: Outline of an Anarchistic Theory of Knowledge. New Left Books (1975)

[10] Friston, K.J., Parr, T., Heins, C., Constant, A., Friedman, D., Isomura, T., Fields, C., Verbelen, T., Ramstead, M., Clippinger, J., Frith, C.D.: Federated inference and belief sharing. Neuroscience & Biobehavioral Reviews 156, 105500 (2024). https://doi.org/10.1016/j.neubiorev.2023.105 500

[11] Genest, C., Zidek, J.V.: Combining probability distributions: a critique and an annotated bibliography. Statistical Science 1(1), 114–135 (1986). https: //doi.org/10.1214/ss/1177013825

[12] Hegselmann, R., Krause, U.: Opinion dynamics and bounded confidence: models, analysis and simulation. Journal of Artificial Societies and Social Simulation 5(3), 2 (2002), https://www.jasss.org/5/3/2.html

[13] Henderson, L., Goodman, N.D., Tenenbaum, J.B., Woodward, J.F.: The structure and dynamics of scientific theories: a hierarchical Bayesian perspective. Philosophy of Science 77(2), 172–200 (2010). https://doi.org/ 10.1086/651319

[14] Holmes, F.L.: The “revolution in chemistry and physics”: overthrow of a reigning paradigm or competition between contemporary research programs? Isis 91(4), 735–753 (2000). https://doi.org/10.1086/384947

[15] Hyland, D., Albarracin, M.: On the variational costs of changing our minds. In: Albarracin, M., et al. (eds.) Active Inference: IWAI 2025, Communica tions in Computer and Information Science, vol. 2857, pp. 3–23. Springer (2026). https://doi.org/10.1007/978-3-032-16955-6\_1

[16] Jonard, N., Reijula, S., Marengo, L.: Theory choice in epistemic networks: five ways to avoid premature convergence. Philosophy of Science (2026). https://doi.org/10.1017/psa.2026.10275

[17] Kuhn, T.S.: The Structure of Scientific Revolutions. University of Chicago Press (1962)

[18] Kusch, M.: Scientific pluralism and the Chemical Revolution. Studies in History and Philosophy of Science Part A 49, 69–79 (2015). https://doi. org/10.1016/j.shpsa.2014.10.001

[19] Messeri, L., Crockett, M.J.: Artificial intelligence and illusions of understanding in scientific research. Nature 627(8002), 49–58 (2024). https: //doi.org/10.1038/s41586-024-07146-0

[20] Molavi, P., Tahbaz-Salehi, A., Jadbabaie, A.: A theory of non-Bayesian social learning. Econometrica 86(2), 445–490 (2018). https://doi.org/10 .3982/ECTA14613

[21] O’Connor, C., Weatherall, J.O.: The Misinformation Age: How False Beliefs Spread. Yale University Press (2019). https://doi.org/10.2307/j.ctv8 jp0hk

[22] Pearl, J.: Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference. Morgan Kaufmann (1988). https://doi.org/10.1016/C2 009-0-27609-4

[23] Quine, W.V.O.: Two dogmas of empiricism. The Philosophical Review 60(1), 20–43 (1951). https://doi.org/10.2307/2181906

[24] Thagard, P.: The conceptual structure of the Chemical Revolution. Philosophy of Science 57(2), 183–209 (1990). https://doi.org/10.1086/289543

[25] Zollman, K.J.S.: The communication structure of epistemic communities. Philosophy of Science 74(5), 574–587 (2007). https://doi.org/10.1086/ 525605

[26] Zollman, K.J.S.: The epistemic benefit of transient diversity. Erkenntnis 72(1), 17–35 (2010). https://doi.org/10.1007/s10670-009-9194-6

[27] Zollman, K.J.S.: Network epistemology: communication in epistemic communities. Philosophy Compass 8(1), 15–27 (2013). https://doi.org/10.1 111/j.1747-9991.2012.00534.x

## A Mathematical details

We begin by briefly restating the notation and update order of the main text. Communication produces $\bar { \rho } _ { i k , t }$ and ${ \bar { a } } _ { i k , t }$ , observation t is scored and learned from, and reference anchoring produces stored state t + 1. The superscript + denotes the learned state before anchoring. All experiments use positive Gaussian densities and common reference distributions across agents, $\rho _ { i k } ^ { 0 } = \rho _ { k } ^ { 0 }$ . Communication and anchoring are separate operations from local Bayesian learning.

## A.1 Variational form of within-theory learning

Writing $Z _ { i k , t } = \exp ( \ell _ { i k , t } )$ , the predictive normalization and posterior are

$$
Z _ { i k , t } = \int _ { \mathcal { X } _ { k } } \mathcal { L } _ { i k , t } ( y _ { i , t } \mid x _ { k } ) \bar { \rho } _ { i k , t } ( x _ { k } ) \mathrm { d } x _ { k } ,\tag{11}
$$

$$
\rho _ { i k , t } ^ { + } ( x _ { k } ) = Z _ { i k , t } ^ { - 1 } \mathcal { L } _ { i k , t } ( y _ { i , t } \mid x _ { k } ) \bar { \rho } _ { i k , t } ( x _ { k } ) .\tag{12}
$$

For any normalized trial density ϱ for which the terms are finite,

$$
\begin{array} { r l } & { { \mathcal F } _ { i k , t } [ \varrho ] = D _ { \mathrm { K L } } [ \varrho \| \bar { \rho } _ { i k , t } ] - { \mathbb E } _ { \varrho } [ \log \mathcal { L } _ { i k , t } ( y _ { i , t } \mid x _ { k } ) ] } \\ & { \qquad = D _ { \mathrm { K L } } \Big [ \varrho \Big \| \rho _ { i k , t } ^ { + } \Big ] - \log Z _ { i k , t } . } \end{array}\tag{13}
$$

Thus, $\rho _ { i k , } ^ { + }$ minimizes the local objective, with minimum given by $- \ell _ { i k , t }$

## A.2 Reference anchoring and within-theory sharing

The anchoring operation has its own variational form:

$$
\begin{array} { r } { \mathcal { A } _ { i k , t } [ \varrho ] = ( 1 - \omega ) D _ { \mathrm { K L } } \big [ \varrho \big | \big | \rho _ { k } ^ { 0 } \big ] + \omega D _ { \mathrm { K L } } \Big [ \varrho \Big | \Big | \rho _ { i k , t } ^ { + } \Big ] . } \end{array}\tag{14}
$$

For $0 \leq \omega \leq 1$ , its minimizing density is

$$
\rho _ { i k , t + 1 } ( x _ { k } ) = \frac { [ \rho _ { k } ^ { 0 } ( x _ { k } ) ] ^ { 1 - \omega } [ \rho _ { i k , t } ^ { + } ( x _ { k } ) ] ^ { \omega } } { \int _ { \mathcal { X } _ { k } } [ \rho _ { k } ^ { 0 } ( v ) ] ^ { 1 - \omega } [ \rho _ { i k , t } ^ { + } ( v ) ] ^ { \omega } \mathrm { d } v } .\tag{15}
$$

Substitution allows us to write $\boldsymbol { A } _ { i k , \cdot }$ as a KL divergence from this density plus a constant. The endpoint $\omega = 0$ resets the belief to the reference, whereas $\omega = 1$ retains the learned belief. Anchoring applies to within-theory densities, not to the accumulated log support $\boldsymbol { a } _ { i k , \mathrm { i } }$ <sub>t</sub>.

Within-theory communication uses the normalized geometric pool

$$
\bar { \rho } _ { i k , t } ( x _ { k } ) \propto \prod _ { j } \rho _ { j k , t } ( x _ { k } ) ^ { W _ { i j , t } ^ { x } } .\tag{16}
$$

The nonnegative weights sum to one in each row and include self-influence. Only densities under the same candidate are pooled. Agents are assumed to agree on candidate identities and coordinate meanings, and alignment between incompatible vocabularies is not modelled here.

## A.3 Gaussian implementation and selective observation

For $\rho _ { i k , t } = \mathcal { N } ( \mu _ { i k , t } , \varSigma _ { i k , t } )$ , let

$$
\begin{array} { r } { \varPi _ { i k , t } = \varSigma _ { i k , t } ^ { - 1 } , \qquad h _ { i k , t } = \varPi _ { i k , t } \mu _ { i k , t } . } \end{array}\tag{17}
$$

The shared projection is $H \ = \ [ 0 _ { 3 \times 1 } \ I _ { 3 } ]$ in both theory coordinate systems. Letting $S _ { i , t } = \mathrm { d i a g } ( \sqrt { w _ { i , t } } ) , y _ { i , t } = S _ { i , t } o _ { i , t }$ and $\widetilde { H } _ { i , t } = S _ { i , t } H$ , agent i’s working likelihood is

$$
\mathcal { L } _ { i k , t } ( y \mid x _ { k } ) = \mathcal { N } ( y ; \widetilde { H } _ { i , t } x _ { k } , \sigma ^ { 2 } I _ { 3 } ) .\tag{18}
$$

The physical observation noise has covariance $\sigma ^ { 2 } I _ { 3 }$ before weighting. Consequently, the actual covariance of $S _ { i , t } o _ { i , t }$ is $\sigma ^ { 2 } S _ { i , t } ^ { 2 }$

## A.4 Model-support dynamics

Equations (3)–(8) imply that

$$
q _ { i k , t + 1 } \propto \exp ( \ell _ { i k , t } ) q _ { i k , t } ^ { 1 - \alpha _ { m } } \prod _ { j } q _ { j k , t } ^ { \alpha _ { m } W _ { i j , t } ^ { m } } ,\tag{19}
$$

normalized over candidates. For two candidates, let

$$
z _ { i , t } = a _ { i \mathrm { O } , t } - a _ { i \mathrm { P } , t } = \log \frac { q _ { i \mathrm { O } , t } } { q _ { i \mathrm { P } , t } } , \qquad s _ { i , t } = \ell _ { i \mathrm { O } , t } - \ell _ { i \mathrm { P } , t } .
$$

Then,

$$
z _ { i , t + 1 } = ( 1 - \alpha _ { m } ) z _ { i , t } + \alpha _ { m } \sum _ { j } W _ { i j , t } ^ { m } z _ { j , t } + s _ { i , t } .\tag{20}
$$

Within-theory sharing changes the predictive distributions entering $s _ { i , t }$ and can therefore afect theory preference even when $\alpha _ { m } = 0$ . The probabilities express support relative to the supplied candidates under this update rule.

## B Experimental implementation

In this appendix, we provide more details about the implementation of our experiments.<sup>6</sup>

## B.1 Theory reference models

Each reference is compiled from the acyclic linear model

$$
x _ { k } = b _ { k } + B _ { k } x _ { k } + \varepsilon _ { k } , \qquad \varepsilon _ { k } \sim { \mathcal { N } } ( 0 , R _ { k } ) .\tag{21}
$$

In coordinate order $( \zeta _ { k } , C , G , M )$ ，

$$
\begin{array} { c } { \mu _ { \mathrm { P } } ^ { 0 } = ( 0 , 1 , 1 , - 1 ) ^ { \top } , \qquad \mu _ { \mathrm { O } } ^ { 0 } = ( 0 , 1 , 1 , 1 ) ^ { \top } , } \\ { R _ { k } = \mathrm { d i a g } ( 0 . 2 5 , 0 . 5 , 0 . 5 , 0 . 5 ) . } \end{array}\tag{22}
$$

Writing $P = \zeta _ { \mathrm { P } }$ and $O = \zeta _ { \mathrm { O } }$ , the release model has coeficients $C \to P : + 0 . 8 .$ $C  G : + 0 . 8$ , and $P \to M : - 0 . 8 ;$ the oxygen model has $C  O : + 0 . 8 .$ $O \to G : + 0 . 8 .$ , and $O \to M : + 0 . 8$ . All other coeficients are zero. Entry $( B _ { k } ) _ { v u }$ represents $u  v$ . With $F _ { k } = I - B _ { k }$ , we have

$$
b _ { k } = F _ { k } \mu _ { k } ^ { 0 } , \qquad \boldsymbol { \pi } _ { k } ^ { 0 } = F _ { k } ^ { \top } R _ { k } ^ { - 1 } \boldsymbol { F } _ { k } , \qquad h _ { k } ^ { 0 } = F _ { k } ^ { \top } R _ { k } ^ { - 1 } \boldsymbol { b } _ { k } .\tag{23}
$$

All agents start at their references with equal candidate support and graphs, coeficients, and references remain fixed during each run.

![](images/c08e013be6c9ed4963f5fce5115d6ca17e30b31128dd130254eb5da697d57ea6.jpg)  
Fig. 5: Reference dependencies for the release and oxygen candidates. Dashed nodes are candidate-specific latent quantities; $C , G ,$ , and M are shared observables. Arrows give the signs of coeficients of magnitude 0.8.

The observable reference covariance is $H ( \pi _ { k } ^ { 0 } ) ^ { - 1 } H ^ { \top }$ . With common references, initialization at those references, and observable-only likelihoods, all subsequent beliefs factor as $\rho _ { i k , t } ( \zeta _ { k } , u ) = p _ { k } ^ { 0 } ( \zeta _ { k } \mid u ) \rho _ { i k , t } ^ { u } ( u )$ , where $\boldsymbol { u } = ( C , G , M ) ^ { \top }$

## B.2 Communication between observation groups

The selective-observation experiment has two groups of $n = 8$ agents and constant signal $( C , G , M ) \ = \ ( 1 , 1 , 1 )$ . Agents 1–8 use attention weights $( 0 , 1 , 0 )$ and agents 9–16 use $( 0 , 0 , 1 )$ ; neither group assigns likelihood weight to $C .$ For $J _ { n } = \mathbf { 1 } _ { n } \mathbf { 1 } _ { n } ^ { \top } / n$ , we have

$$
W _ { \beta } = \left( \begin{array} { c c } { { ( 1 - \beta ) J _ { n } } } & { { \beta J _ { n } } } \\ { { \beta J _ { n } } } & { { ( 1 - \beta ) J _ { n } } } \end{array} \right) .\tag{24}
$$

Each agent assigns total influence $1 - \beta$ to its own group, including self-influence, and $\beta$ to the other group. Both candidate conditions use $W ^ { x } = W _ { \beta }$ . Private support uses $\alpha _ { m } = 0$ , and shared support uses $W ^ { m } = W _ { \beta }$ with $\alpha _ { m } = 0 . 0 2$ . The Gaussian comparison uses the same observations and $W ^ { \dot { x } } = W _ { \beta }$

## B.3 Single-Gaussian comparison

The single-Gaussian comparison represents only $\boldsymbol { u } = ( C , G , M ) ^ { \top }$ and has no candidate theory probabilities. Its common reference is the equally weighted logarithmic pool of the two observable reference marginals. Defining $\check { \mu } _ { k } ^ { u , 0 } = H \mu _ { k } ^ { 0 }$ and $\varSigma _ { k } ^ { u , 0 } = H ( \varPi _ { k } ^ { 0 } ) ^ { - 1 } H ^ { \intercal }$ , we have

$$
\begin{array} { r } { \pi _ { \mathrm { G } } ^ { 0 } = \frac { 1 } { 2 } \displaystyle \sum _ { k \in \{ \mathrm { P } , \mathrm { O } \} } \big ( \boldsymbol { \itSigma } _ { k } ^ { u , 0 } \big ) ^ { - 1 } , \qquad h _ { \mathrm { G } } ^ { 0 } = \frac { 1 } { 2 } \displaystyle \sum _ { k \in \{ \mathrm { P } , \mathrm { O } \} } \big ( \boldsymbol { \itSigma } _ { k } ^ { u , 0 } \big ) ^ { - 1 } \mu _ { k } ^ { u , 0 } . } \end{array}\tag{25}
$$

It begins at this reference and uses the same local update, retention, attention weights, and communication matrix as the candidate comparison, with $H = I _ { 3 }$

## B.4 Evidence changes and network communication

All evidence-change experiments use

$$
\xi _ { t } = \left\{ \begin{array} { l l } { ( 1 , 1 , - 1 ) ^ { \top } , } & { 0 \leq t < 8 0 , } \\ { ( 1 , 1 , 1 ) ^ { \top } , } & { t \geq 8 0 , } \end{array} \right.\tag{26}
$$

with $\sigma = 1$ and $\omega = 0 . 9$ . The full-observation demonstration (Fig. 2) has 16 agents, $W _ { i j } = 1 / 1 6$ , 240 updates, attention weights (1, 1, 1) throughout, and both communication channels are enabled with $\alpha _ { m } = 0 . 0 2$

The communication and topology comparisons have 24 agents and 600 updates. All agents use (1, 1, 1) for the first 80 observations. From observation 80, agents 1–4 retain full weights while agents 5–24 use (1, 0.01, 0.01). All enabled model-support channels use $\alpha _ { m } ~ = ~ 0 . 0 2$ . The complete-graph communication comparison branches from the same fully observing, fully communicating state after 80 updates in each run, and uses identical subsequent observations in its four conditions:

## B.5 Revision across communication networks

We conducted an additional experiment comparing diferent network topologies, which keeps both channels enabled, holding each network fixed throughout its initial and changed-evidence periods. We number agents 1–24, so the set of fullyobserving agents is {1, 2, 3, 4}. The ring topology joins consecutive vertices and 24 to 1. The wheel has hub 1, joined to all vertices 2–24, with a cycle through those rim vertices. The community graph has cliques on 1–4 and 5–24 and bridges (1, 8), (2, 8), (2, 20), (3, 9), (3, 11), (3, 23), (4, 24). The complete graph joins every distinct pair.

<table><tr><td>Condition  $W ^ { x } \alpha _ { m }$ </td></tr><tr><td>Neither  $I _ { 2 4 } \mathrm { ~  ~ { ~ 0 ~ } ~ }$ </td></tr><tr><td>Within-theory only  $W \_ { 0 }$ </td></tr><tr><td>Model support only  $I _ { 2 4 } ~ 0 . 0 2$ </td></tr><tr><td>Both W 0.02</td></tr><tr><td></td></tr></table>

Table 1: Communication settings from observation 80 in Fig. 4. Local observation updates remain enabled in every condition. Here $W _ { i j } = 1 / 2 4$

![](images/0a1c30bf7a2194e12a30e6b133417c6a735c6366db19365113ee8d146db94a59.jpg)  
Fig. 6: Belief revision of the restricted agents over candidate theories with both channels enabled across four networks. Black outlines around nodes in the topology depictions mark the fully observing agents. Bold curves average the restricted-group mean oxygen support across 50 runs. Bands represent pointwise 95% percentile-bootstrap intervals from 2,000 whole-run resamples. Thin curves show individual restricted agents in run 0.

These results show that revision spreads more quickly on the complete network and hub-informed wheel and more slowly on the community network. Restricted-group oxygen support on the cycle network continues to increase at the end of the horizon.

## C Measurements and reporting

## C.1 Timing convention

Stored state t enters round t, and observation t produces state t+1. The selective and communication-channel plots label completed states, whereas the cycle and topology plots label generating observation indices and show the corresponding post-update states. Predictive evaluation at state $t \geq 1$ uses signal $\xi _ { t - 1 }$ , the signal that generated the most recently learned observation.

## C.2 Theory choice and uncertainty

For group I, define

$$
\bar { q } _ { \mathcal { T } , t } ( \mathrm { O } ) = | \mathcal { T } | ^ { - 1 } \sum _ { i \in \mathcal { T } } q _ { i \mathrm { O } , t } .\tag{27}
$$

Individual uncertainty is given by

$$
\mathcal { H } ( q _ { i , t } ) = - \sum _ { k } q _ { i k , t } \log q _ { i k , t } ,\tag{28}
$$

with 0 log $0 = 0$ . The plotted mean averages individual entropies over agents before averaging runs. It is diferent from the entropy of a group’s mean probability, in that confident disagreement can coexist with low individual entropy.

## C.3 Prediction and numerical disagreement

At stored state $t ,$ a candidate model predicts

$$
\hat { u } _ { i , t } = \sum _ { k } q _ { i k , t } H \mu _ { i k , t } ,\tag{29}
$$

$$
p _ { i , t } ( o ) = \sum _ { k } q _ { i k , t } \mathcal { N } ( o ; H \mu _ { i k , t } , H \varSigma _ { i k , t } H ^ { \top } + \sigma ^ { 2 } I _ { 3 } ) .\tag{30}
$$

Evaluation uses all three observable coordinates irrespective of attention during learning. For the Gaussian control, these are the corresponding single-Gaussian mean and predictive density with observation noise added. For $t \geq 1$ 2

$$
\begin{array} { r } { \mathrm { M S E } _ { i , t } = \frac { 1 } { 3 } \| \hat { u } _ { i , t } - \xi _ { t - 1 } \| _ { 2 } ^ { 2 } . } \end{array}\tag{31}
$$

MSE measures estimation error relative to the noiseless signal.

For gas and mass groups $\mathcal { T } _ { G }$ and $\mathcal { T } _ { M }$ , the distance between community beliefs is given by

$$
D _ { t } = \left. | \mathcal { Z } _ { G } | ^ { - 1 } \sum _ { i \in \mathcal { Z } _ { G } } \hat { u } _ { i , t } - | \mathcal { Z } _ { M } | ^ { - 1 } \sum _ { i \in \mathcal { Z } _ { M } } \hat { u } _ { i , t } \right. _ { 2 } .\tag{32}
$$