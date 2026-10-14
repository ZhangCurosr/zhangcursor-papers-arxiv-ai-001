# Choosing LLMs for Strategic Agent Interactions

Arastun Mammadli   
University of Exeter   
United Kingdom   
am2169@exeter.ac.uk

Herbert Woisetschläger IBM Research Switzerland h.woisetschlaeger@ibm.com

Jawad Fayaz   
University of Exeter   
United Kingdom   
j.fayaz@exeter.ac.uk   
Shiqiang Wang   
University of Exeter   
United Kingdom   
s.wang9@exeter.ac.uk

## Abstract

Game-theoretic benchmarks claim that large language model (LLM) strategic decision-making varies across game types, interaction structures, and capability dimensions. However, they only evaluate strategic performance across individual fixed models. In parallel, contemporary LLM routers pick models under qualitycost trade-offs, but with no notion of a strategic structure. They do not account for an opponent that might be unknown, and that can adapt over time and shift the feedback signal. To address this, we extend the game-theoretic evaluations and define strategic selection as the meta game layered on top of the base game. In each round, a strategic selector fields an LLM from a portfolio of models to navigate a game payoff matrix that is latent and learned online. We find that the evaluation signal is game-specific, and that extracting a reliable signal requires a much higher number of seed runs N compared to traditional benchmarking setups. Evaluating across 5 games and 10 LLMs, we show a model’s strategic competence varies significantly depending on the game setup and the opponent faced. Building on this, we implement a suite of selector methods and find that approaches that rely on query embeddings or LLM self-reflection do not consistently capture the underlying strategic signal. A lightweight multi-armed bandit algorithm performs more reliably across games, though it still falls short of the best fixed model. Finally, we show that simple non-stationary and adversarial bandit variants fail to outperform a stationary bandit baseline, pointing to a gap in how current methods handle a shifting opponent in strategic settings.

![](images/2635639dd809f3d7711a32e1bd2125ed2665e08b1814a92a35191e15551ae71f.jpg)

![](images/88186d24c9633f51817ac4fd8d4084c7136fb425bffafe068afc3d9299da3c0e.jpg)  
Figure 1: Left: A strategic selector i plays a repeated game over T rounds against an unknown opponent −i and observes the game outcome $\xi _ { i , t }$ . The UPDATE $( \theta _ { i , t } , x _ { i , t } , \xi _ { i , t } )$ function is applied at every round t. Right: Strategic performance (game utility v) varies across games and models.

## 1 Introduction

Large language models (LLMs) are increasingly deployed in multi-agent settings as part of a larger compound system. However, a growing share of deployers treat the inter-agent dynamics, competitive and cooperative alike, as a black box. This makes it hard to build systems that remain capable, efficient, and reliable. Existing work addresses pieces of this problem in isolation. The strategic decision-making literature focuses on a fixed model’s play under a specified game setting [7, 28, 37]. In parallel, LLM routing addresses model selection under performance-cost trade-off [5, 26, 31], but with a static task distribution and no notion of a strategic structure or opponent. Neither accounts for the fact that contemporary LLM portfolios differ not only in scale and cost, but also in the type and quality of strategic behaviour they exhibit.

Our work reframes model selection itself as a strategic decision. Concretely, we extend the gametheoretic evaluations and define strategic selection as the opponent-conditioned meta game layered on top of the base game. Here the base game is played between the fielded LLMs, while the meta game is played between two strategic selectors that field a model of their choice each round. In repeated settings, the dynamic and adaptive nature of this problem (as opposed to traditional routers operating on a fixed distribution of tasks) means our selection should be strategic. Beyond balancing the performance-cost trade-offs, such a selection layer can help steer the deployer’s general play, adding control over the reliability and safety of the interaction (Figure 1). We ask the question:

Given LLM strategic capabilities vary across different games and models, can a selector exploit this by learning which model tofield against an unknown and possibly adaptive opponent?

Why This Matters. Traditional routers operate under the assumption of a fixed task distribution, learning a static mapping from queries to optimal models. In contrast, operating against an adaptive opponent transforms this into a non-stationary learning problem, where the optimal strategy can continuously shift over time. A further challenge arises from the partial observability of the meta game’s payoff matrix. In practice, a selector cannot observe the opponent’s net gain, since the opponent’s model choice is unknown, nor can it observe the counterfactual payoffs of fielding an alternative model in the current game state. We also acknowledge that LLMs are stochastic models. In strategic selection, we introduce an additional layer of environmental noise through the dynamic responses of the opponent. Therefore, benchmarking and result reporting call for a careful statistical treatment. Our contributions are threefold:

• Strategic Model Selection. We formalise strategic selection as a repeated general-sum meta game between two players. Beyond capability and cost, this lets us treat the interaction itself as a first-class design consideration for model selection.

• Quantifying Noise in Strategic Play. We characterise the signal and noise levels across games, and suggest adapting methods from traditional hypothesis testing to decide on the number of seeds per a game-portfolio pair.

• Empirical Investigation. To set up our empirical investigation, we select 10 models from different families, and adapt a range of existing routing methods (Table 1). We first show that LLM strategic capabilities are scenario-dependent and exploitable by a selector. We then investigate the performance of adapted methods against both fixed models and selector opponents.

## 2 Related Work

## 2.1 Strategic LLM Decision-Making

A growing literature explores LLMs in strategic settings (e.g. how LLMs play games against each other). Game-theoretic benchmarks [7, 10, 28, 36] and frameworks [11, 33, 34, 37] test LLMs in normal-form games and find that performance is uneven and structure-dependent. Others [3, 24, 32, 35] evaluate capability on a higher level (e.g. planning, collaboration, theory of mind) and consistently find spiky, uneven profiles across models. Similarly, Mei et al. [13] and Horton et al. [9] run behavioural Turing tests and show that different models exhibit different strategic dispositions (i.e. behavioural profiles). Motivated by such findings, Raman et al. [19] outlines the need for an element-level (model, prompt, domain) selection approach, but leaves it to future work. We note that despite extensive coverage, the existing literature always fixes the model per player.

Table 1: Implemented suite of selector methods. Cost-Aware: the method accounts for a quality-cost trade-off when selecting a model. Opp.-ID: the opponent’s identity is revealed to the selector. Online: the method learns online (round-by-round) from game outcomes. Embed.: the method encodes context into embeddings before feeding them into the predictor. Non-Stat.: the method is designed to handle a non-stationary environment.
<table><tr><td>Method</td><td>Cost-Aware</td><td>Opp.-ID</td><td>Online</td><td></td><td></td><td>Embed. Non-Stat. Source</td></tr><tr><td>Fixed Model</td><td>X</td><td>×</td><td>X</td><td>X</td><td>X</td><td></td></tr><tr><td>Per-Opponent Full-Information</td><td></td><td></td><td>X</td><td>X</td><td>X</td><td></td></tr><tr><td>Implemented selectors</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LLM Selector</td><td></td><td>X</td><td></td><td>X</td><td>×</td><td>Shinn et al. [21]</td></tr><tr><td>RouterDC</td><td></td><td>×</td><td></td><td></td><td>×</td><td>Chen et al. [5]</td></tr><tr><td>Causal LLM Router</td><td></td><td>X</td><td></td><td></td><td>×</td><td>Tsiourvas et al. [26]</td></tr><tr><td>Thompson Sampler</td><td></td><td>X</td><td></td><td>X</td><td>X</td><td>William [29]</td></tr><tr><td>Discounted Thompson Sampler</td><td></td><td>X</td><td></td><td>×</td><td></td><td>Raj and Kalyani [18]</td></tr><tr><td>Sliding-Window Thompson Sampler</td><td></td><td>×</td><td></td><td>×</td><td></td><td>Trovò et al. [25]</td></tr><tr><td>EXP3</td><td></td><td>X</td><td></td><td>X</td><td></td><td>Auer et al. [2]</td></tr></table>

## 2.2 LLM Routing

An orthogonal direction learns to predict a best-fit model for a given query, often trading off performance for cost and relying on minimal two-model setups [1, 6, 16]. Others extend this to incorporate more models [4, 17, 22, 30] or train on limited observational data [26, 31]. These works represent queries as vector embeddings to semantically capture their scope and difficulty. However, we argue that discriminative signal in strategic selection largely comes from evolving game history (actions and outcomes), and two consecutive rounds with near identical embeddings can call for opposite model choices. In the absence of routing methods that explicitly model this structure, we also consider bandit-style per-round feedback [2, 29] as a paradigm suited to a non-stationary interactive signal.

## 3 Strategic LLM Selection

## 3.1 What is Strategic Selection?

In multi-agent systems, a setting is non-strategic if the objective concerns only the overall performance of the system or its performance-cost trade-off. A setting is strategic if it also accounts for the interaction between agents, including how they communicate, coordinate, and negotiate. Modelling this interaction can produce systems that are more reliable and efficient. In a cooperative setting, it improves the coordination between agents [12]. In a competitive setting, it gives the deployer more control over the interaction with an opponent and helps steer the interaction toward the deployer’s preferences. For example, for a given business, the reward of the negotiation, the cost of using AI, and the consistency (i.e. associated risk) of the results can all matter at varying levels.

## 3.2 Problem Formulation

We formalise strategic LLM selection as a repeated, contextual, general-sum game with two layers: a base game that fielded models play, and a selection meta game that the deployers play. Strategic selection focuses on the meta layer, a two-player game with i indexing the player (deployer) and −i its opponent. Repeated games are played over rounds $t = 1 , \dots , T$ , where each round t is associated with context $x _ { i , t } \in \mathcal { X } \left( \mathrm { e . g } \right.$ . game setup, instructions, past history). Deployer i’s set of pure strategies (meta game actions) $S _ { i }$ is defined as the set of models $\{ 1 , \ldots , M _ { i } \}$ in its portfolio. In each round, i selects a model $s _ { i , t } \sim \sigma _ { i } ( \cdot \mid x _ { i , t } )$ , where $\sigma _ { i } \in \Delta ( S _ { i } )$ is a mixed strategy (i.e. selection policy). Per-round context $x _ { i , t }$ is then passed down to the fielded model to produce the base game action $s _ { i , t } ( x _ { i , t } ) \to a _ { i , t }$ . The base game (i.e. of fielded models $s _ { i , t }$ against $s _ { - i , t } )$ produces a utility $v _ { i , t }$ at an inference cost $c _ { i , t }$ (tokens spent), as well as a net gain of $g _ { i , t }$ that captures their trade-off. Together, player i’s game outcome is denoted as ${ \xi } _ { i , t } = ( s _ { i , t } , a _ { i , t } , g _ { i , t } )$ . Drawing from the online learning literature, each selector maintains an adaptive parameter vector $\theta _ { i , t }$ that updates after each round based on observed context $x _ { i , t }$ and game outcome $\xi _ { i , t } \mathrm { : }$

$$
\begin{array} { r } { s _ { i , t } \sim \sigma _ { i } ( \cdot  { | \mathbf { \phi } _ { x _ { i , t } ; \theta _ { i , t } } ) , } \qquad \theta _ { i , t + 1 } = \mathrm { U P D A T E } \big ( \theta _ { i , t } , x _ { i , t } , \xi _ { i , t } \big ) . } \end{array}\tag{1}
$$

Any architecture that maps $\theta _ { i , t }$ to a selection policy $\sigma _ { i , t }$ and supports updates from online feedback qualifies as a backbone. Such options can include classical multi-armed bandit samplers [29], neural network-based classifiers [5, 26], or at the frontier, an LLM prompted with game history. Consequently, $\theta _ { i , t } \mathrm { ' s }$ definition depends on the backbone, such as the network weights for a classifier, or the per-arm posterior for a Thompson sampler. We note that the game outcome is also incorporated into the next round’s context $x _ { i , t + 1 } = f ( x _ { i , t } , \xi _ { i , t } )$ to be used in base game decision-making. Finally, this is repeated over N independent seed runs (where the parameter vector is reset for each seed $\theta _ { i , 1 }  \theta _ { i } ^ { \bar { 0 } } )$ to account for stochastic variance in strategic play. Algorithm 1 summarises the overall procedure. Note that from player i’s view, the opponent’s selector $\sigma _ { - i }$ , its fielded model $s _ { - i , t }$ , and action $a _ { - i , t }$ are never observed. They are resolved by the game environment $\mathcal { G }$

## 3.3 Noise in Strategic Play

Single-trial LLM decision-making is noisy, subject to the probabilistic sampling mechanism, and sensitive to variations in prompts. Related work addresses LLM evaluations by adapting cross-disciplinary methods such as signal processing [8], experiment-design theory [14], and psychometrics [15]. They find that in traditional evaluations (e.g. question-answering benchmarks), even a modest amount of replication (e.g. $N \approx 5 )$ substantially stabilises the estimates. We argue that due to its interactive opponent-conditioned structure, strategic LLM evaluation is inherently noisier and requires a higher number of replications. To quantify the noise in strategic play, we define a method to measure

Algorithm 1: Repeated strategic selection, de  
ployer i’s view, in game G, across N seed runs   
1 Initialize $S _ { i } = \{ 1 , \dots , M _ { i } \} , \theta _ { i } ^ { 0 } , T$ rounds, N seeds;   
for n $,  1$ to N do   
// runs are independent   
2 $\theta _ { i , 1 }  \theta _ { i } ^ { 0 } , x _ { i , 1 } \in \mathcal { X } ;$   
3 for t ← 1 to T do   
4 $s _ { i , t } \sim \sigma _ { i } ( \cdot \mid x _ { i , t } ; \theta _ { i , t } ) ;$   
$/ /$ model takes action   
5 $a _ { i , t } \gets s _ { i , t } ( x _ { i , t } ) ;$   
$/ /$ base game resolves   
6 $v _ { i , t } , c _ { i , t } , g _ { i , t } \gets \mathcal { G } ( a _ { i , t } ) ;$   
7 $\xi _ { i , t } \gets ( s _ { i , t } , a _ { i , t } , g _ { i , t } ) ;$   
$/ /$ selector parameter is updated   
8 $\theta _ { i , t + 1 }  \mathrm { U P D A T E } ( \theta _ { i , t } , x _ { i , t } , \xi _ { i , t } ) ;$   
9 $x _ { i , t + 1 } \gets f ( x _ { i , t } , \xi _ { i , t } ) ;$

the levels of signal and noise across games by adapting Heineman et al. [8]’s work. Building on this, we develop approaches to choose $N$ for a given game-portfolio pair [14]. See Appendix B for further details.

## 3.4 Evaluation Metrics

To measure strategic performance, we capture the game utility $v _ { i } ,$ associated cost $c _ { i } ,$ and their trade-off represented as the net gain $g _ { i }$ . The game utility $v _ { i }$ is min-max normalised $( \hat { v } _ { i } \in [ 0 , 1 ] )$ based on its game-specific bounds (e.g. the economic profit in Cournot Duopoly, or the number of winning battlefields in Colonel Blotto game). The associated inference cost $c _ { i }$ is represented by the per-model token usage cost (in dollars \$) using the OpenRouter<sup>1</sup> listings as market proxies. This cost is min-max normalised $( \hat { c } _ { i } \in [ 0 , 1 ] )$ based on the portfolio-bounds (minimum and maximum per-token price observed across the model portfolio). Finally, we define net gain $g _ { i }$ as the weighted sum of normalised utility $\hat { v } _ { i }$ and cost $\hat { c } _ { i }$

$$
g _ { i } = \frac { \hat { v } _ { i } + \lambda ( 1 - \hat { c } _ { i } ) } { 1 + \lambda } , \qquad g _ { i } \in [ 0 , 1 ] ,\tag{2}
$$

where λ represents the cost trade-off weight and is fixed at $\lambda = 1$ across our runs. Any game episode is replicated across N seeds to account for stochastic noise. Each of the N seed runs yields a single sample, from which we compute the mean X<sup>¯</sup> , the standard error (SE), and the 95% confidence interval (CI). Alternative metrics include the in-game winning rate (i.e. we ignore the raw $g _ { i }$ and measure $\mathbb { 1 } [ g _ { i } > g _ { - i } ] )$ ), or representing the cost $c _ { i }$ by the per-model throughput (tokens produced per second).

## 4 Experiments

We provide empirical evidence in 3 steps and motivate the problem of strategic LLM selection. First, an exhaustive pairwise matchup across 10 models and 5 games motivates the problem, showing that strategic capability is varied and exploitable by a selector. Following these findings, a suite of selectors is run against fixed but unknown models (i.e. opponent is a fixed LLM) to evaluate their potential as strategic selectors. Finally, these selectors are deployed against adaptive opponents to simulate the non-stationary dynamics of real-world settings. We note that reported results should not be compared across (Tables 2 to 4), as a method’s performance is a function of the specific opponent pool it faced which varies across settings (e.g. the runs against adaptive opponents do not include fixed models).

## 4.1 Design

Full details on the models, games, and selector implementations are given in Appendix C.

Models and Games. The model portfolio draws on 10 open-weight LLMs across four families (Granite, Mistral, Llama, and Qwen) ranging from 3B to 70B to capture varying capability levels, usage costs, and dispositions. The game inventory captures 5 classical game-theoretic scenarios (Colonel Blotto, Cournot Duopoly, Stag Hunt, Vickrey Auction, Chicken Game) spanning zero-sum to cooperative dilemmas. Experiments are run through OutplayArena<sup>2</sup>, which owns the base game rules, turn sequencing, and scoring.

Selectors. A suite of selectors spanning traditional and adversarial bandit samplers [2, 29], contemporary LLM routers [5, 26], and an LLM self-reflection framework [21] is then incorporated to draw on this portfolio. Adapting selectors from a range of different backgrounds plays an important diagnostic role in understanding the strengths and weaknesses of each method as strategic selectors. Additionally, simple opponent-aware full-information selectors are evaluated post-hoc to measure the selection headroom.

Number of Rounds and Seeds. Some strategic benchmarks [28] evaluate models across hundreds of games. To test whether such broad evaluations rest on a consistent signal, we quantify the pergame signal-to-noise ratio (SNR) and find that the evaluation signal is inherently game-specific. A power analysis on single-round fixed-model matchups shows that the share of detectable pairwise comparisons stabilises at N ≈ 150 seeds in every evaluated game, with Cournot Duopoly separating models more cleanly than any other game (Appendix B.3). In our experiments, fixed-model matchups are single-round T = 1 with N = 300 seeds. Selectors learn online and require repeated play, so we run T = 300 rounds per seed with a single seed (N = 1) to keep the inference cost tractable. Note that the fielded model’s and fixed opponent’s context is limited to the current round. This ensures that each round is an independent draw of the opponent’s play.

## 4.2 Results

Strategic Capabilities Vary Across Models and Games. Table 2 reveals that strategic competence is uneven and game dependent. For example, the Llama models are top performers in Cournot Duopoly but fail to make the podium in Chicken Game. In addition, the substantial performance variance among models within individual games highlights an opportunity to adopt a selector. Figure 2a underscores that model scale does not reliably translate into strategic capability. Across our evaluations, we find only a weak positive correlation between model scale and utility (Pearson’s r = 0.37), and a moderate negative relationship when accounting for the usage costs (r = −0.54). Upon a closer look, we also see that smaller models have game-specific usability. For example in Vickrey Auction, due to its high rationality (i.e. always bidding the true value), the Granite4.1 3B outperforms other much bigger models that range from 24B to 70B parameters. Finally, Figure 2b supports our framing “choosing a model is choosing a strategy”. For each model, we score strategic dispositions from its in-game actions, such as the dare rate in Chicken Game or the produced quantity in Cournot Duopoly. We describe each model by a vector of its seven disposition scores, and project these vectors onto their principal components (Appendix D.1). We find that even same family models can differ sharply in disposition. For example, Granite4.1 8B dares in 96% of Chicken Game rounds while its 3B and 30B versions never dare. Families still show a general coherence, as Qwen3.5 and Llama models compete aggressively in Cournot Duopoly, whereas most Granite and Mistral models cooperate. We believe this is an important signal that a strategic selector should learn to infer and exploit.

![](images/48fdffe04a59873d740881e266da17e7d51602f9aebcc66ead29835ddd8b5153.jpg)  
Params (B)

![](images/78821711d3641cc442cd5a5cccbfad6475b82962fad65b7163cc06b9536dadf4.jpg)  
(a) Normalised v ↑ and g(\$) ↑ across all games.  
Params (B)

![](images/4b0d416d598ff6891235363a61b743d3a78c42ca31dba90ac8f9c7d41f9c930d.jpg)  
(b) Model dispositions on the first two PCs.  
Figure 2: (a) Model scale does not reliably translate into strategic capability. (b) Models differ in strategic disposition. Each point shows a model at the projection of its seven in-game disposition scores onto the first two principal components. Granite, Mistral, Llama, Qwen3.5.

Table 2: Normalised in-game utility v ↑ acquired by each model across 5 games. We report the mean ± 95% CI, ×100, aggregated across all pairwise matchups, using T = 1 round, N = 300 seeds per run, and highlight the best score for each game.
<table><tr><td>Model</td><td></td><td></td><td>Chicken Game Colonel Blotto Cournot Duopoly Stag Hunt Vickrey Auction</td><td></td><td></td></tr><tr><td>Granite4.1 3B</td><td> ${ \bf 8 7 . 5 \pm 0 . 3 }$ </td><td> $5 0 . 6 \pm 0 . 3$ </td><td> $2 6 . 0 \pm 1 . 2$ </td><td> $7 4 . 3 \pm 2 . 9$ </td><td> ${ \bf 7 4 . 3 \pm 0 . 6 }$ </td></tr><tr><td>Granite4.1 8B</td><td> $6 9 . 6 \pm 3 . 0$ </td><td> $5 0 . 5 \pm 0 . 3$ </td><td> $7 . 8 \pm 1 . 5$ </td><td> $6 5 . 6 \pm 2 . 2$ </td><td> $7 3 . 7 \pm 0 . 6$ </td></tr><tr><td>Granite4.1 30B</td><td> $8 7 . 3 \pm 0 . 3$ </td><td> $5 0 . 4 \pm 0 . 2$ </td><td> $1 3 . 2 \pm { 1 . 5 }$ </td><td> $7 2 . 9 \pm 2 . 9$ </td><td> $6 9 . 4 \pm 0 . 5$ </td></tr><tr><td>Ministral3 3B</td><td> $8 6 . 4 \pm 0 . 7$ </td><td> $4 8 . 5 \pm 0 . 5$ </td><td> $1 8 . 4 \pm 2 . 0$ </td><td> $7 3 . 0 \pm 2 . 8$ </td><td> $7 2 . 0 \pm 0 . 6$ </td></tr><tr><td>Mistral-Nemo 12B</td><td> $8 4 . 2 \pm 1 . 2$ </td><td> $5 0 . 3 \pm 0 . 2$ </td><td> $2 0 . 0 \pm 2 . 3$ </td><td> $7 2 . 7 \pm 2 . 8$ </td><td> $7 1 . 9 \pm 0 . 6$ </td></tr><tr><td>Mistral-Small 3.2 24B</td><td> $8 2 . 2 \pm { 1 . 7 }$ </td><td> $4 9 . 7 \pm 0 . 3$ </td><td> $3 5 . 2 \pm 2 . 8$ </td><td> $5 6 . 1 \pm 1 . 2$ </td><td> $7 2 . 5 \pm 0 . 6$ </td></tr><tr><td>Llama3.1 8B</td><td> $7 7 . 1 \pm 2 . 3$ </td><td> $4 9 . 8 \pm 0 . 4$ </td><td> $4 0 . 4 \pm 2 . 7$ </td><td> $6 4 . 5 \pm 2 . 0$ </td><td> $7 1 . 5 \pm 0 . 7$ </td></tr><tr><td>Llama3.3 70B</td><td> $6 9 . 9 \pm 2 . 9$ </td><td> $5 0 . 4 \pm 0 . 2$ </td><td> ${ \bf 4 8 . 9 \pm 2 . 9 }$ </td><td> $7 5 . 7 \pm 2 . 6$ </td><td> $7 3 . 8 \pm 0 . 6$ </td></tr><tr><td>Qwen3.5 9B</td><td> $7 2 . 0 \pm 2 . 8$ </td><td> $4 9 . 6 \pm 0 . 4$ </td><td> $2 3 . 2 \pm 2 . 3$ </td><td> $7 4 . 7 \pm 2 . 6$ </td><td> $7 3 . 0 \pm 0 . 6$ </td></tr><tr><td>Qwen3.5 35B</td><td> $8 3 . 7 \pm 1 . 4$ </td><td> $5 0 . 3 \pm 0 . 3$ </td><td> $4 2 . 0 \pm 2 . 8$ </td><td> $7 7 . 2 \pm 2 . 6$ </td><td> $7 1 . 4 \pm 0 . 7$ </td></tr></table>

Strategic Context Needs Special Treatment. We first simplify our selection problem by playing against a fixed unknown opponent, to try and infer strategic learning signals against a stationary fixed LLM. Table 3 indicates that not all traditional methods transfer equally to the strategic selection setup, with the LLM Selector scoring the lowest overall. Upon closer inspection, we find that it also falls behind nearly every fixed model across most games (Appendix D.3). The Causal LLM Router and RouterDC show relative improvement but are still beaten by several fixed models in multiple games. We attribute this to the fact that strategic settings require special treatment of the game context and the learning signal. Methods that rely on query embeddings for semantic similarity matching (Causal LLM Router, RouterDC), or on LLM reflection (LLM Selector), fail to surface implicit strategic signals. A simple bandit algorithm using Thompson sampling achieves the highest net gain among all selectors in every game, despite relying on nothing more than per-round outcome feedback. However, even in Cournot Duopoly, where the Thompson sampler proves most dominant, it fails to beat the best fixed model (Llama3.1 8B), indicating significant room for improvement. We believe a better selector algorithm should learn to directly infer the latent, hidden per-game strategic signal, rather than relying solely on observed game outcomes.

An Interactive Learning Problem. In practice, opponents can also use a selector that adapts its policy each round based on the observed game outcome. To capture this setting, we extend our evaluations and play against adaptive selector opponents. Table 4 reveals that multi-armed bandit algorithms remain dominant. However, the non-stationary adaptations (Sliding-Window TS and Discounted TS) fail to exploit the underlying concept drift in adaptive settings to outclass the plain TS. Similarly, an adversarial variant (EXP3) performs noticeably worse, on par with embedding-based methods. Hence, we note that simple multi-armed bandit adaptations are not sufficient to capture the implicit learning signals shaped by repeated strategic interaction.

## 5 Lessons Learned

## 5.1 How to Learn from Strategic Context?

We believe strategic context is multi-faceted and encapsulates the game instructions (i.e. the strategic scenario we are dealing with), the portfolio model dispositions (i.e. understanding our resources to act on this scenario), and the opponent’s actions that shape our scenario. Together, this context is either available as a prior (e.g. through existing game-specific runs across models), or gathered post-hoc by observing the game outcome $\xi _ { i , t }$ at the end of each round. Traditional routing and sampling methods fail to make best use of this context. Existing methods either encode the entire context (e.g. game instructions, history) as a whole, as form of vector embedding (RouterDC), without an underlying form or structure, or rely entirely on the observed game outcome (Thompson Sampler) with no regard to the strategic scenario (captured by the game instructions) we are dealing with. A better selector should infer game-specific strategic cues, and map selection choices to their implications for future game states. It should build an understanding of the performance of its own portfolio models, and the opponent under the current scenario.

Table 3: Normalised net gain g ↑ (top) and in-game utility v ↑ (bottom) acquired by each selector, against fixed opponents and across 5 games. We show the mean ± 95% CI, ×100, averaged across 10 runs (the size of our portfolio), using $T = 3 0 0$ rounds, N = 1 seed per run, and highlight the best score for each game. Selectors’ learning signal was set as their per-round gain $g _ { i , t } .$
<table><tr><td>Selector</td><td></td><td></td><td>Chicken Game Colonel Blotto Cournot Duopoly Stag Hunt Vickrey Auction</td><td></td><td></td></tr><tr><td colspan="6">Reports normalised gain g ↑</td></tr><tr><td>Per-Opponent Full-Information</td><td> $9 2 . 8 \pm 0 . 4$ </td><td> $7 3 . 0 \pm 0 . 2$ </td><td> $8 0 . 7 \pm { 1 . 5 }$ </td><td> $9 2 . 2 \pm 1 . 0$ </td><td> $8 6 . 7 \pm 0 . 5$ </td></tr><tr><td>Thompson Sampler (TS)</td><td> ${ \bf 8 5 . 1 \pm 0 . 5 }$  </td><td> ${ \bf 7 1 . 2 \pm 0 . 2 }$ </td><td> ${ \bf 4 7 . 7 \pm 0 . 8 }$ </td><td> ${ \bf 7 6 . 1 \pm 0 . 8 }$ </td><td> ${ \bf 7 4 . 8 \pm 0 . 1 }$ </td></tr><tr><td>RouterDC</td><td> $8 4 . 3 \pm 0 . 1$ </td><td> $6 8 . 6 \pm 0 . 1$ </td><td> $3 6 . 2 \pm 0 . 1$ </td><td> $7 4 . 4 \pm 0 . 8$ </td><td> $7 2 . 8 \pm 0 . 2$ </td></tr><tr><td>Causal LLM Router</td><td> $7 8 . 6 \pm 0 . 7$ </td><td> $6 3 . 2 \pm { 0 . 5 }$ </td><td> $3 1 . 3 \pm 0 . 7$ </td><td> $7 4 . 0 \pm 0 . 9$ </td><td> $6 4 . 8 \pm 0 . 5$ </td></tr><tr><td>LLM Selector</td><td> $6 1 . 0 \pm 0 . 6$ </td><td> $4 6 . 9 \pm 0 . 3$ </td><td> $2 6 . 9 \pm 0 . 5$ </td><td> $4 6 . 6 \pm 1 . 2$ </td><td> $5 4 . 4 \pm 0 . 2$ </td></tr><tr><td colspan="6">Reports normalised utility v ↑</td></tr><tr><td>Per-Opponent Full-Information</td><td> $8 8 . 9 \pm 0 . 7$ </td><td> $5 0 . 6 \pm 0 . 3$ </td><td> $4 3 . 0 \pm 2 . 6$ </td><td> $8 8 . 1 \pm 2 . 1 $ </td><td> $7 5 . 1 \pm 0 . 7$ </td></tr><tr><td>Thompson Sampler (TS)</td><td> $8 7 . 4 \pm 0 . 7$ </td><td> $5 0 . 4 \pm 0 . 1$ </td><td> ${ \bf 3 9 . 4 \pm 1 . 4 }$  </td><td> $7 8 . 0 \pm 1 . 3$ </td><td> ${ \bf 7 3 . 4 \pm 0 . 3 }$ </td></tr><tr><td>RouterDC</td><td> ${ \bf 8 7 . 8 \pm 0 . 2 }$  </td><td> $5 0 . 5 \pm 0 . 1$ </td><td> $1 9 . 6 \pm 0 . 3$ </td><td> $7 6 . 1 \pm 1 . 5$ </td><td> $7 2 . 8 \pm 0 . 3$ </td></tr><tr><td>Causal LLM Router</td><td> $7 9 . 7 \pm 1 . 1$ </td><td> $5 0 . 1 \pm 0 . 2$ </td><td> $2 7 . 1 \pm 1 . 3$ </td><td> $7 2 . 4 \pm 1 . 4$ </td><td> $7 2 . 2 \pm 0 . 4$ </td></tr><tr><td>LLM Selector</td><td> $8 0 . 6 \pm 1 . 2$ </td><td> $5 0 . 1 \pm 0 . 2$ </td><td> $1 8 . 1 \pm { 1 . 0 }$ </td><td> ${ \bf 8 7 . 0 \pm 2 . 3 }$  </td><td> $7 1 . 5 \pm 0 . 5$ </td></tr></table>

Table 4: Normalised net gain g ↑, in-game utility v ↑, and raw usage cost $c \downarrow$ acquired by each selector, against other selectors and in Cournot Duopoly. We report the mean ± 95% CI, ×100, aggregated across all pairwise matchups, using $T = 3 0 0$ rounds, $N = 1$ seed per run, and highlight the best score for each metric. Selectors’ learning signal was set as their per-round gain g<sub>i,t</sub>.
<table><tr><td>Selector</td><td></td><td>net gain g ↑ in-game utility v ↑ usage cost</td><td> $c \left( \times 1 0 ^ { - 3 } \$ \right) \downarrow$ </td></tr><tr><td>Thompson Sampler (TS)</td><td>一  $\mathbf { 7 2 . 4 \pm 0 . 8 }$ </td><td> ${ \bf 4 7 . 1 \pm 1 . 7 }$  </td><td> $0 . 0 2 7 \pm 0 . 0 0 1$ </td></tr><tr><td>Sliding-Window TS</td><td> $6 5 . 9 \pm 0 . 7$ </td><td> $3 3 . 2 \pm 1 . 4$ </td><td> $0 . 0 1 7 \pm 0 . 0 0 1$ </td></tr><tr><td>Discounted TS</td><td> $6 3 . 1 \pm 0 . 8$ </td><td> $2 7 . 5 \pm 1 . 6$ </td><td> $0 . 0 1 7 \pm 0 . 0 0 1$ </td></tr><tr><td>EXP3</td><td> $6 0 . 2 \pm 0 . 7$ </td><td> $2 2 . 1 \pm 1 . 4$ </td><td> $0 . 0 2 0 \pm 0 . 0 0 1$ </td></tr><tr><td>RouterDC</td><td> $5 9 . 6 \pm 0 . 3$ </td><td> $1 9 . 3 \pm 0 . 5$ </td><td> $\mathbf { 0 . 0 0 2 \pm 0 . 0 0 0 }$  </td></tr><tr><td>Causal LLM Router</td><td> $5 9 . 1 \pm 0 . 7$ </td><td> $1 9 . 7 \pm 1 . 3$ </td><td> $0 . 0 1 9 \pm 0 . 0 0 1$ </td></tr><tr><td>LLM Selector</td><td> $4 0 . 9 \pm 0 . 7$ </td><td> $1 9 . 6 \pm 1 . 3$ </td><td> $0 . 4 0 9 \pm 0 . 0 0 4$ </td></tr></table>

## 5.2 Games as Strategic Testbeds

We saw that best-performing models vary noticeably across the 5 evaluated games, each setup capturing a particular strategic scenario (a model-to-model interplay) (Table 2). Moreover, an observation can be made that games vary in difficulty (i.e. complexity). For example, one can say that identifying the equilibrium strategy would require higher model reasoning capabilities in a game like Cournot Duopoly (an economic profit game) than in Colonel Blotto, which relies on a simple per-battlefield resolution rule (the larger number wins). Empirically, this idea is reflected in the correlation between model scale and performance, which is the highest in Cournot Duopoly $( \mathrm { P e a r s o n ' s } ~ r = 0 . 6 6 )$ , and much lower in games like Stag Hunt $( r = 0 . 1 2 )$ and Colonel Blotto (r = 0.24) (Figure D.3). Mathematically, there are an infinite number of normal- and extensive-form games that range in complexity and the strategic interplay captured. Future work should therefore establish well-defined methods to classify a game’s difficulty and strategic value. Such methods are of high value to deployers before they decide where to train and use their agent workflows, as well as towards designing strategic selectors that better understand the underlying strategic learning signal.

## 5.3 Games as Dynamic Testbeds

Real-world AI deployments are messy. Inputs are noisy, and the agents involved often hold different, sometimes conflicting objectives. Our results show that a model’s competence is shaped by the opponent it faces, and how both sides adapt as the interaction unfolds. Traditional benchmarks fail to capture this, as they test one model at a time against a fixed correct answer. This makes game inventories a natural fit for evaluating models in decision-pressured, interactive settings that they are actually deployed to in practice. Moreover, because game states are generated dynamically through interaction, games are more resistant to the training-data contamination that increasingly undermines static benchmarks [23].

That said, games are not immune to saturation. Even without contamination, the discriminative signal can erode as models approach equilibrium play (strategy profiles from which no player can gain by unilaterally deviating). More specifically, under standard no-regret learning, adaptive play converges to the set of coarse correlated equilibria (CCE), a superset of Nash equilibria [20]. In two-player zero-sum games such as Colonel Blotto, every equilibrium yields the same expected payoff, so once models play near-equilibrium strategies, none can outperform another and their results become indistinguishable. In contrast, in general-sum games the set of CCE is typically large and contains outcomes with very different payoffs. This suggests favouring general-sum games when building inventories meant to remain discriminative.

## 6 Conclusion and Future Work

We extend game-theoretic evaluations and define strategic selection as a repeated opponentconditioned meta game layered on top of the base game. Initial evaluations across 5 games and 10 models show that LLM strategic capability varies substantially across games and models, leaving considerable room for a selector to exploit. By adapting a range of selection methods, we report that the existing methods do not transfer equally well even against a simple fixed-model opponent, and that a plain bandit algorithm can surpass structurally heavier pre-trained encoders and LLM reflection methods. Subsequent runs against a selector opponent reveal that simple adaptations like discounted and adversarial bandit algorithms are not able to capture the non-stationary dynamics of opponent-conditioned strategic selection, and fail to outclass the plain sampler. Overall, the results highlight the challenges of strategic model selection, and motivate a selector design that can extract and learn from signals across 3 different dimensions: model capability, usage cost, and strategic disposition, in noise-heavy interactive settings.

Our investigation was scoped to the analysis of 5 games. However, to better understand agent-to-agent interplay and the underlying safety and risk factors, it is important to consider games from a range of fields, including economics, social sciences, and operations research. We believe this choice should reflect both the measurable effect of the game and the underlying practical application of the selector. We also relied on explicit parsed game actions as a communication method between the models, which underestimates both the strategic capabilities (e.g. persuasion), and the associated risk factors in deployment. A more realistic setting should capture agent-to-agent natural language interactions [27]. We leave such free-form interactions to future work. Finally, high-stakes repeated interactions (e.g. negotiations, auctions) involve a risk factor not captured by expected gain. This can include catastrophic losses, regulatory penalties, or reputational damage. Capturing the variance in realised gain to account for involved risk is another future direction. For example, a risk knob $( \beta \in \mathbb { R } \colon \beta = 0$ is risk-neutral, $\beta > 0$ risk-averse, $\beta < 0$ risk-seeking) could allow users to better align the selector with their deployment policies.

## References

[1] P. Aggarwal, A. Madaan, A. Anand, S. P. Potharaju, S. Mishra, P. Zhou, A. Gupta, D. Rajagopal, K. Kappaganthu, Y. Yang, et al. AutoMix: Automatically mixing language models. In Advances in Neural Information Processing Systems (NeurIPS), volume 37, pages 131000–131034, 2024.

[2] P. Auer, N. Cesa-Bianchi, Y. Freund, and R. E. Schapire. The nonstochastic multiarmed bandit problem. SIAM journal on computing, 32(1):48–77, 2002.

[3] J. Chen, X. Hu, S. Liu, S. Huang, W.-W. Tu, Z. He, and L. Wen. LLMArena: Assessing capabilities of large language models in dynamic multi-agent environments. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL), pages 13055–13077, 2024.

[4] L. Chen, M. Zaharia, and J. Zou. FrugalGPT: How to use large language models while reducing cost and improving performance. Transactions on Machine Learning Research, 2024. URL https://openreview.net/forum?id=cSimKw5p6R.

[5] S. Chen, W. Jiang, B. Lin, J. Kwok, and Y. Zhang. RouterDC: Query-based router by dual contrastive learning for assembling large language models. In Advances in Neural Information Processing Systems (NeurIPS), volume 37, pages 66305–66328, 2024.

[6] D. Ding, A. Mallick, C. Wang, R. Sim, S. Mukherjee, V. Rühle, L. V. S. Lakshmanan, and A. H. Awadallah. Hybrid LLM: Cost-efficient and quality-aware query routing. In International Conference on Learning Representations (ICLR), 2024.

[7] J. Duan, R. Zhang, J. Diffenderfer, B. Kailkhura, L. Sun, E. Stengel-Eskin, M. Bansal, T. Chen, and K. Xu. GTBench: Uncovering the strategic reasoning capabilities of LLMs via gametheoretic evaluations. In Advances in Neural Information Processing Systems (NeurIPS), volume 37, pages 28219–28253, 2024.

[8] D. Heineman, V. Hofmann, I. Magnusson, Y. Gu, N. A. Smith, H. Hajishirzi, K. Lo, and J. Dodge. Signal and noise: A framework for reducing uncertainty in language model evaluation. In Advances in Neural Information Processing Systems (NeurIPS), volume 38, pages 17073–17114, 2025.

[9] J. J. Horton, A. Filippas, and B. S. Manning. Large language models as simulated economic agents: What can we learn from homo silicus? Working Paper 31122, National Bureau of Economic Research, 2023. Revised February 2026.

[10] J.-t. Huang, E. J. Li, M. H. Lam, T. Liang, W. Wang, Y. Yuan, W. Jiao, X. Wang, Z. Tu, and M. R. Lyu. Competing large language models in multi-agent gaming environments. In International Conference on Learning Representations (ICLR), 2025.

[11] J. Jia, Z. Yuan, J. Pan, P. E. McNamara, and D. Chen. LLM strategic reasoning: Agentic study through behavioral game theory. In Advances in Neural Information Processing Systems (NeurIPS), volume 38, pages 64992–65024, 2025.

[12] A. Khatua, H. Zhu, P. Tran, A. Prabhudesai, F. Sadrieh, J. K. Lieberwirth, X. Yu, Y. Fu, M. J. Ryan, J. Pei, and D. Yang. CooperBench: Why coding agents cannot be your teammates yet. In Proceedings of the 3rd Conference on Language Modeling (COLM), 2026. URL https://openreview.net/forum?id=qo1hehBPVf.

[13] Q. Mei, Y. Xie, W. Yuan, and M. O. Jackson. A turing test of whether ai chatbots are behaviorally similar to humans. Proceedings of the National Academy of Sciences, 121(9):e2313925121, 2024.

[14] E. Miller. Adding error bars to evals: A statistical approach to language model evaluations. arXiv preprint arXiv:2411.00640, 2024.

[15] Z. Mustahsan, A. Lim, M. Anand, S. Jain, and B. McCann. Stochasticity in agentic evaluations: Quantifying inconsistency with intraclass correlation. In AAAI 2026 Workshop on Trustworthy Agentic AI, 2026. arXiv:2512.06710.

[16] I. Ong, A. Almahairi, V. Wu, W.-L. Chiang, T. Wu, J. E. Gonzalez, M. W. Kadous, and I. Stoica. RouteLLM: Learning to route LLMs from preference data. In International Conference on Learning Representations (ICLR), 2025.

[17] C. Qian, Z. Liu, S. Kokane, A. Prabhakar, J. Qiu, H. Chen, Z. Liu, H. Ji, W. Yao, S. Heinecke, et al. xRouter: Training cost-aware LLMs orchestration system via reinforcement learning. arXiv preprint arXiv:2510.08439, 2025.

[18] V. Raj and S. Kalyani. Taming non-stationary bandits: A Bayesian approach. arXiv preprint arXiv:1707.09727, 2017.

[19] N. Raman, T. Lundy, S. Amouyal, Y. Levine, K. Leyton-Brown, and M. Tennenholtz. STEER: Assessing the economic rationality of large language models. In Proceedings of the 41st International Conference on Machine Learning (ICML), volume 235, pages 42026–42047. PMLR, 2024.

[20] T. Roughgarden. Twenty Lectures on Algorithmic Game Theory. Cambridge University Press, Cambridge, UK, 2016. ISBN 978-1-316-62479-1.

[21] N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, and S. Yao. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS), volume 36, pages 8634–8652, 2023.

[22] S. Somerstep, F. M. Polo, A. F. M. de Oliveira, P. Mangal, M. Silva, O. Bhardwaj, M. Yurochkin, and S. Maity. CARROT: A cost aware rate optimal router. arXiv preprint arXiv:2502.03261, 2025.

[23] Y. Sun, H. Wang, D. Li, G. Wang, and H. Zhang. The emperor’s new clothes in benchmarking? A rigorous examination of mitigation strategies for LLM benchmark data contamination. In Proceedings ofthe 42nd International Conference on Machine Learning (ICML), volume 267, pages 57728–57753. PMLR, 2025.

[24] W. Tang, Y. Zhou, K. Cheng, E. Xu, L. Xiao, and M. Li. DSGBench: A diverse strategic game benchmark for evaluating LLM-based agents in complex decision-making environments. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 16987–16991. IEEE, 2026.

[25] F. Trovò, S. Paladino, M. Restelli, and N. Gatti. Sliding-window Thompson sampling for non-stationary settings. Journal ofArtificial Intelligence Research, 68:311–364, 2020. doi: 10.1613/jair.1.11407.

[26] A. Tsiourvas, W. Sun, and G. Perakis. Causal LLM Routing: End-to-end regret minimization from observational data. In Advances in Neural Information Processing Systems (NeurIPS), volume 38, pages 41861–41886, 2025.

[27] V. C. Vendrell, M. R. Luyten, and M. van der Schaar. GameTalk: Training LLMs for strategic conversation. arXiv preprint arXiv:2601.16276, 2026.

[28] H. Wang, X. Feng, L. Li, Y. Guo, Z. Qin, D. Sui, and L. Kong. TMGBench: A systematic game benchmark for evaluating strategic reasoning abilities of LLMs. arXiv preprint arXiv:2410.10479, 2024.

[29] R. T. William. On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. Biometrika, 25(3/4):285–294, 1933.

[30] H. Woisetschläger, R. Zhang, S. Wang, and H.-A. Jacobsen. MESS+: Dynamically learned inference-time LLM routing in model zoos with service level guarantees. In Advances in Neural Information Processing Systems (NeurIPS), volume 38, pages 60178–60211, 2025.

[31] H. Woisetschläger, A. Mammadli, R. Zhang, and S. Wang. Cost-optimal LLM routing with limited user feedback under user satisfaction guarantees. arXiv preprint arXiv:2606.19376, 2026.

[32] Y. Wu, X. Tang, T. Mitchell, and Y. Li. SmartPlay: A benchmark for LLMs as intelligent agents. In International Conference on Learning Representations (ICLR), volume 2024, pages 1543–1561, 2024.

[33] Y. Xie, Z. Zhou, C. Cao, Q. Niu, T. Liu, and B. Han. From debate to equilibrium: Beliefdriven multi-agent LLM reasoning via Bayesian Nash equilibrium. In Proceedings ofthe 42nd International Conference on Machine Learning (ICML), volume 267, pages 72277–72316, 2025.

[34] Y. Xie, K. Wang, B. Cheng, J. Yao, Z. Sha, A. Duffy, Y. Xi, H. Mei, C. Tan, C. Wei, P. Viswanath, and Z. Wang. MEMO: Memory-augmented model context optimization for robust multi-turn multi-agent LLM games. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. arXiv:2603.09022.

[35] L. Xu, Z. Hu, D. Zhou, H. Ren, Z. Dong, K. Keutzer, S. K. Ng, and J. Feng. MAgIC: Investigation of large language model powered multi-agent in cognition, adaptability, rationality and collaboration. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7315–7332, 2024.

[36] Z. Xu, Z. Xu, X. Yi, H. Yuan, M. Guang, K. Long, X. Chen, Y. Wu, C. Yu, and Y. Wang. VS-Bench: Evaluating vlms for strategic abilities in multi-agent environments. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21561–21572, 2026.

[37] H. Yuan, Z. Xu, Z. Tan, X. Yi, M. Guang, K. Long, H. Hui, B. Li, X. Chen, B. Zhao, et al. MARSHAL: Incentivizing multi-agent reasoning via self-play with strategic LLMs. In International Conference on Learning Representations (ICLR), volume 2026, pages 6209–6236, 2026.

## Appendix

A Glossary of Notation 13   
B More Details about Noise in Strategic Play 13   
B.1 Signal and Noise Ratio 13   
B.2 Number of Seeds 13   
B.3 Analysis . 14   
C More Details about Experimental Setup 15   
C.1 Models 15   
C.2 Games 15   
C.3 Selectors 16   
D More Results 17   
D.1 Fixed Model Disposition Scores 17   
D.2 Fixed Model vs. Fixed Model . 18   
D.3 Selector vs. Fixed Model 19

## A Glossary of Notation

Table A.1: Glossary of notation.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td colspan="2">Meta Game Γ (LLM selection game)</td></tr><tr><td> $i , - i$ </td><td>player indices, where —i denotes the opponent of i</td></tr><tr><td> $t = 1 , \dots , T$ </td><td>round index of the game</td></tr><tr><td> $N$ </td><td>number of seeds in the game (independent replications) finite set of pure strategies, i.e. player i&#x27;s model portfolio</td></tr><tr><td> $S _ { i } = \{ 1 , \dots , M _ { i } \}$   $\sigma _ { i } \in \Delta ( S _ { i } )$ </td><td>mixed strategy for player i, a probability simplex over Si</td></tr><tr><td> $\boldsymbol { x } _ { i , t } \in \mathcal { X }$ </td><td>context observed by player i before round t</td></tr><tr><td></td><td>game outcome produced by player ¿ on round t</td></tr><tr><td> $\xi _ { i , t }$ </td><td>pure strategy (model choice) fielded by player i at round t</td></tr><tr><td> $s _ { i , t } \sim \sigma _ { i }$ </td><td>utility obtained by player i from the base-game outcome</td></tr><tr><td> $_ { v _ { i , t } }$ </td><td></td></tr><tr><td> $^ { c _ { i , t } }$ </td><td>inference cost incurred by player i (tok/$)</td></tr><tr><td> $^ { g _ { i , t } }$ </td><td>net gain that captures the utility-cost trade-off</td></tr><tr><td> $\theta _ { i , t }$ </td><td>internal state of player ¿&#x27;s selector at round t</td></tr><tr><td colspan="2">Base Game G (e.g. Colonel Blotto, Cournot Duopoly)</td></tr><tr><td> ${ { a } _ { i , t } }$ </td><td>action produced by the fielded model  $s _ { i , t }$  at round t</td></tr></table>

## B More Details about Noise in Strategic Play

## B.1 Signal and Noise Ratio

Following Heineman et al. [8], we first define a descriptive method that characterises signal and noise levels across games. Fix a game G and let S denote its set of models under evaluation, with $j , k \in S$ ranging over the pair of models compared in G. For a model $s \in S$ , let $\tau = 1 , \dots , N$ index its seeds, where a seed is a single independent evaluation run. Let $v _ { \mathcal { G } , s , \tau }$ be the bounds-normalised score on seed τ of $\mathcal { G }$ played by model $s .$ . Let $\overline { { v } } _ { { \mathcal G } , s }$ denote the mean of $v _ { \mathcal { G } , s , \tau }$ over all N seeds, and let $\overline { { v } } _ { \mathcal G }$ denote the mean of $\overline { { v } } { _ { \mathscr { G } , s } }$ over all models $s \in S$ . Noise, signal, and signal-to-noise ratio (SNR) are then

$$
\begin{array} { r } { \operatorname { n o i s e } ( \mathcal { G } , s ) = \frac { \operatorname { S t d } _ { \tau } \left( v _ { \mathcal { G } , s , \tau } \right) } { \overline { { v } } _ { \mathcal { G } , s } } , \qquad \operatorname { s i g n a l } ( \mathcal { G } ) = \frac { \operatorname* { m a x } _ { j , k \in S } \left| \overline { { v } } _ { \mathcal { G } , j } - \overline { { v } } _ { \mathcal { G } , k } \right| } { \overline { { v } } _ { \mathcal { G } } } , \qquad \operatorname { S N R } ( \mathcal { G } ) = \frac { \operatorname { s i g n a l } ( \mathcal { G } ) } { \overline { { \operatorname { n o i s e } } } ( \mathcal { G } ) } . } \end{array}\tag{3}
$$

Here Std<sub>τ</sub>(·) denotes the standard deviation taken over the seed index $\tau _ { \ast }$ , and noise(G) denotes noise $( { \mathcal { G } } , s )$ averaged over every model $s \in S$ . Noise is a coefficient of variation, a model’s own seed-to-seed spread relative to its mean. Signal is the largest gap between any two models’ means, relative to the game’s overall mean. SNR is high when models are well separated relative to how much a single model’s score jitters across seeds, and low when the two quantities are comparable.

## B.2 Number of Seeds

Given the SNR, we want to determine the minimum number of seeds N that lets us report statistically significant outcomes. This is a trade-off between cost (the number of model inferences we run) and reliability (our statistical confidence in the reported results).

Decision accuracy. Heineman et al. [8] suggests calculating a decision accuracy metric to determine the number of seeds $N$ needed to resolve the pairwise gaps between models. Let $\mathcal { P }$ denote the set of model pairs compared in $\mathcal { G } _ { : }$ , let n denote a candidate (small) batch size, and let $N \gg n$ denote a large reference batch size. Let $\overline { { v } } _ { \mathcal { G } , s } ^ { ( n ) }$ denote model $s \mathrm { ^ { \circ } s }$ mean bounds-normalised score over n seeds. Decision accuracy at batch size n is then

$$
\mathrm { D A } ( n ) = \frac { 1 } { \vert \mathcal { P } \vert } \sum _ { \{ s _ { A } , s _ { B } \} \in \mathcal { P } } \mathbb { 1 } \Big [ \mathrm { s i g n } \Big ( \overline { { v } } _ { \mathcal { G } , s _ { A } } ^ { ( n ) } - \overline { { v } } _ { \mathcal { G } , s _ { B } } ^ { ( n ) } \Big ) = \mathrm { s i g n } \Big ( \overline { { v } } _ { \mathcal { G } , s _ { A } } ^ { ( N ) } - \overline { { v } } _ { \mathcal { G } , s _ { B } } ^ { ( N ) } \Big ) \Big ] ,\tag{4}
$$

the fraction of the pairs in $\mathcal { P }$ whose sign at batch size n agrees with the sign at the reference batch size N. Every n below N is estimated as the mean over repeated independent subsamples of the N-seed reference, drawn per model pair so each draw mirrors an actual N-seed experiment run at batch size n. However, decision accuracy needs a reference batch, and in game-theoretic evaluations this reference is often not remotely the scale of traditional benchmarks, since the game-theoretic reference batch is self-generated instead of drawn from an external test suite.

Minimum detectable effect. A more practical approach is to use the minimum detectable effect (MDE), which does not need a reference batch, and asks directly whether the current N and opponentpool size is sufficient to resolve the pairwise gaps we report. Power analysis of Miller [14] (Eqs. 9-10), for two models $s _ { A }$ and $s _ { B }$ with M shared opponents and K seeds played against each opponent, gives the MDE as

$$
\begin{array} { r l r } & { } & { \delta ( K ) = ( z _ { \alpha / 2 } + z _ { \beta } ) \sqrt { \frac { \omega ^ { 2 } + \sigma _ { s _ { A } } ^ { 2 } / K + \sigma _ { s _ { B } } ^ { 2 } / K } { M } } , \qquad ( 5 ) } \\ & { } & { \omega ^ { 2 } = \mathrm { V a r } _ { o } ( \overline { { v } } _ { \mathscr G , s _ { A } , o } - \overline { { v } } _ { \mathscr G , s _ { B } , o } ) , \qquad \sigma _ { s _ { A } } ^ { 2 } = \mathbb { E } _ { o } [ \mathrm { V a r } _ { t } ( v _ { \mathscr G , s _ { A } , o , t } ) ] , \qquad \sigma _ { s _ { B } } ^ { 2 } = \mathbb { E } _ { o } [ \mathrm { V a r } _ { t } ( v _ { \mathscr G , s _ { B } , o , t } ) ] . } \end{array}\tag{6}
$$

Here o ranges over the M shared opponents $( M = 8$ in a 10-model pool experiment), and $t =$ $1 , \ldots , K$ indexes seeds played against a given opponent. $z _ { \gamma }$ denotes the $( 1 - \gamma )$ quantile of the standard normal distribution, so $\alpha = 0 . 0 5 ( \mathrm { g i v i n g } z _ { \alpha / 2 } )$ and power $1 - \beta = 0 . 8 ( \mathrm { g i v i n g } z _ { \beta } )$ are Miller [14]’s own defaults for significance and power. $\omega ^ { 2 }$ is the variance, across the M shared opponents, of the cluster-mean gap between $s _ { A }$ and $s _ { B } .$ , i.e. how much that gap swings depending on the opponent, and $\sigma _ { s _ { A } } ^ { 2 }$ and $\sigma _ { s _ { B } } ^ { 2 }$ are each model’s mean within-opponent variance across seeds. All three quantities are estimated from a reference (e.g. $N = 3 0 0 ) . \delta ( K )$ is the smallest true skill gap detectable at that $K$ , for a fixed opponent-pool size. For game $\mathcal { G }$ evaluated on a model pool S, the detectable fraction at K seeds is

$$
\rho _ { \mathcal { G } } ( K ) = \frac { 1 } { { \binom { | S | } { 2 } } } \sum _ { \{ s _ { A } , s _ { B } \} \subset S } \mathbb { 1 } [ | \overline { { v } } g , s _ { A } - \overline { { v } } g , s _ { B } | \ge \delta ( K ) ] ,\tag{7}
$$

the share of the $\binom { | S | } { 2 }$ pairwise gaps in $\mathcal { G }$ that exceed $\delta ( K )$ . Sweeping K per game produces a rough signal on the statistical significance of pairwise model comparisons from that game.

## B.3 Analysis

Scope of the analysis. All signal, noise, and seed-stabilisation analyses in this section are computed on fixed-model matchups, using single-round episodes $( T = 1 )$ . Each seed is therefore an independent draw of the base game, and the estimates characterise the inherent variability of strategic play between two given models. We note that our analyses do not directly transfer to selector runs, where selector’s own choices can evolve over the rounds.

Signal-to-noise ratio is not game-agnostic. Cournot Duopoly separates models almost twice as cleanly as any other game, while Colonel Blotto carries the lowest signal (Figure B.1a). A qualitative investigation shows that this stems from all models playing Colonel Blotto very similarly, all converging to an equilibrium of an even distribution across battlefields. This SNR ranking validity is confirmed independently by decision accuracy (Figure B.1b). Holding the seed count n fixed and correlating SNR against decision accuracy across the five games gives Pearson $r \approx 0 . 8 1$ to 0.85 for $n \geq 7 5$ close to Heineman et al. [8]’s reported $R = 0 . 7 9$ , indicating that our SNR metric captures the same effect as theirs.

<table><tr><td>Game</td><td>Signal</td><td>Noise</td><td>SNR</td></tr><tr><td>Cournot Duopoly</td><td>1.431</td><td>1.537</td><td>0.931</td></tr><tr><td>Chicken Game</td><td>0.224</td><td>0.357</td><td>0.627</td></tr><tr><td>Stag Hunt</td><td>0.315</td><td>0.559</td><td>0.564</td></tr><tr><td>Vickrey Auction</td><td>0.066</td><td>0.129</td><td>0.512</td></tr><tr><td>Colonel Blotto</td><td>0.040</td><td>0.089</td><td>0.449</td></tr></table>

(a)

![](images/ded0dd1f5b7d97fc76a8eebed434258d4a3154518355c6aeac733f6027e1181e.jpg)  
(b)  
Figure B.1: Statistical analysis of signal and noise across 5 games and 4 model families. (a) Signal, noise, and SNR per game, $n = 3 0 0$ . (b) Similar to Heineman et al. [8] $( R = 0 . 7 9 )$ we show that our correlation holds at $r \approx 0 . 8 1$ from $n = 7 5$ onwards.

Decision accuracy results in Figure B.2b follow the same ranking. Cournot Duopoly reaches 95% decision accuracy at the smallest seed count of the five games, $n = 2 0$ , while Colonel Blotto needs the largest, $n = 2 0 0$ , consistent with its near-zero signal. No single seed count clears 95% decision accuracy on every game before $n = 2 0 0$

In the power analysis (Figure B.2a), some games never clear the significance bar even at the full seed budget, while Cournot Duopoly has the highest ratio of detectable pairwise comparisons, confirming the ranking from both the SNR and the decision accuracy analysis. The detection rate converges after around $\geq 1 5 0$ seeds for the games that do clear the bar. For the primary investigation, we fix the number of seeds to $N = 3 0 0$

![](images/8e8458df1c6f72a1b3cf4a81f0fe5866f16b7186a1dd9f5e2d29e48e61b0725c.jpg)  
(a)

![](images/3246866ee7f995baad8ec4e7ca9ff88984a82ea810117ef0f092fb94198dfbba.jpg)  
(b)  
Figure B.2: Statistical analysis of signal and noise across $^ { 5 }$ games and 4 model families. (a) Fraction of all 45 reported pairwise gaps that are statistically detectable. (b) Decision accuracy, mean over 30 subsample draws per n.

## C More Details about Experimental Setup

## C.1 Models

Our experimental model portfolio spans 10 open-weight models with three size tiers, listed in Table C.1. Every model is served locally through Ollama. Every model call uses a sampling temperature of 1.0 and nucleus sampling (top-p) of 0.9. Every experiment runs on a workstation with 64 physical CPU cores (128 logical cores) and two NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs, under Linux 7.0.0-28-generic (x86-64, glibc 2.39) with Python 3.12.3.

Table C.1: Model portfolio used across experiments.
<table><tr><td>Model</td><td>Hugging Face repository</td></tr><tr><td>granite4.1:3b</td><td>ibm-granite/granite-4.1-3b</td></tr><tr><td>granite4.1:8b</td><td>ibm-granite/granite-4.1-8b</td></tr><tr><td>granite4.1:30b</td><td>ibm-granite/granite-4.1-30b</td></tr><tr><td>1lama3.1:8b-instruct-q4_K_M</td><td>meta-1lama/Llama-3.1-8B-Instruct</td></tr><tr><td>1lama3.3:70b</td><td>meta-1lama/Llama-3.3-70B-Instruct</td></tr><tr><td>ministral-3:3b</td><td>mistralai/Ministral-3-3B-Instruct-2512</td></tr><tr><td>mistral-nemo:12b</td><td>mistralai/Mistral-Nemo-Instruct-2407</td></tr><tr><td>mistral-small3.2:24b</td><td>mistralai/Mistral-Small-3.2-24B-Instruct-2506</td></tr><tr><td>qwen3.5:9b</td><td>Qwen/Qwen3.5-9B</td></tr><tr><td>qwen3.5:35b</td><td>Qwen/Qwen3.5-35B-A3B</td></tr></table>

## C.2 Games

The base game $\mathcal { G }$ is instantiated as one of five two-player games. For each game we have kept the default configurations provided by OutplayArena. We provide brief details about each game, and refer the reader to the source outlined in Table C.2 for further information.

• Colonel Blotto is a zero-sum game where two players split a fixed budget across several battlefields. It measures how a model reallocates its strategy in response to an opponent’s shifting distribution.

• Cournot Duopoly is a simultaneous quantity-setting game where two firms choose outputs that jointly fix market price and profit. It isolates a single continuous best-response signal that separates models by how closely each converges toward the equilibrium quantity.

• Stag Hunt is a coordination game with a risk-dominant and a payoff-dominant equilibrium. It measures whether a model favors the safer but lower-payoff option over the cooperative one when the opponent’s action is uncertain.

• Vickrey Auction is a second-price sealed-bid auction where each bidder submits a valuebased bid. It measures how closely a model’s bid tracks its private valuation under the auction’s truthful-bidding equilibrium.

• Chicken Game is an anti-coordination game built around brinkmanship, where mutual insistence is worst for both players. It measures a model’s willingness to escalate or concede as the risk of mutual loss grows.

Table C.2: The five base games G used across experiments, all played between 2 players.
<table><tr><td>Game</td><td>Structure</td><td>Source</td></tr><tr><td>Chicken Game</td><td>anti-coordination, brinkmanship</td><td>https://en.wikipedia.org/wiki/Chicken_(game)</td></tr><tr><td>Colonel Blotto</td><td>zero-sum resource allocation</td><td>https://en.wikipedia.org/wiki/Blotto_game</td></tr><tr><td>Cournot Duopoly</td><td>simultaneous quantity competition</td><td>https://en.wikipedia.org/wiki/Cournot_competition</td></tr><tr><td>Stag Hunt</td><td>coordination, risk dominance</td><td>https://en.wikipedia.org/wiki/Stag_hunt</td></tr><tr><td>Vickrey Auction</td><td>second-price sealed-bid auction</td><td>https://en.wikipedia.org/wiki/Vickrey_auction</td></tr></table>

## C.3 Selectors

We provide the implementation detail of each selector.

Fixed Model. One fixed model across all seeds N that keeps context and past outcomes across rounds t = 1, . . . , T unless otherwise stated.

Per-Opponent Full-Information. Given the identity of the fixed opponent, it looks up the round-robin table (our prior) once at construction and commits to whichever candidate scores highest against that opponent, fielding it every round with no further adaptation.

LLM Selector. Uses llama3.3:70b as the selecting model and relies on the Reflexion framework [21] to pick the model to field. Namely, each round it is queried once with a description of the game, an entry per portfolio model (tier and per-token price), and a log of past rounds covering its own fielded model, action, and outcome signal. It parses a JSON response giving the picked model, a confidence, and a reasoning. Prior information enters only as in-context history, a rolling log of past rounds inside the prompt, with default sampling temperature = 1.0.

RouterDC. Adapted from Chen et al. [5]. Each round a frozen sentence encoder embeds the current query, built from player’s own model, action, and signal history, and the pick is whichever candidate’s trained embedding has highest cosine similarity to that query embedding. The embeddings come from a checkpoint fit offline, once per game. All prior information is folded into this pre-trained checkpoint at training time.

Causal LLM Router. We follow the doubly robust regret-minimising rule of Tsiourvas et al. [26]. Each round it draws a candidate from a softmax over its current per-(candidate, opponent) belief, with a floor that keeps every candidate at nonzero pick probability. It then folds in a doubly robust correction of the observed signal, using the pick probability it assigned as its own propensity instead of fitting a separate propensity model. We use the defaults surrogate = softmax, tau = 1.0, and propensity\_floor = 0.05.

Thompson Sampler. It keeps a per-model Gaussian belief over the chosen signal that is refined as a running mean after every round. Each round it draws one sample per model from its current belief and fields the model with the highest draw. Additionally, we implement two non-stationary variants that change the belief update. The discounted variant (Dis. Thompson Sampler) multiplies every model’s accumulated pseudo-count by the discount factor before folding in the new observation, so older evidence decays exponentially. The sliding-window variant (SW Thompson Sampler) keeps only the W (window size) most recent observations, and drops older ones from the belief entirely. We use discount = 0.99 in the discounted variant and window = 50 in the sliding-window variant.

EXP3. Exponential-weights bandit updated by an importance-weighted multiplicative rule on the observed signal. Its mixed strategy blends the weight-proportional distribution with a uniform exploration floor, so every candidate keeps nonzero pick probability (exploration\_rate = 0.1).

## D More Results

Section D.1 captures the complete in-game disposition analysis inferred from fixed model matchups.   
Section D.2 gives the complete fixed-model-versus-fixed-model matchup results across all 5 games.   
Section D.3 then gives the full selector-versus-fixed-model matchup results.

## D.1 Fixed Model Disposition Scores

For each game, we infer a small set of strategic dispositions from in-game play (Table D.1). To summarize how models differ, we standardize each disposition score across models and project the resulting vectors onto their two directions of largest variation. Together, these two principal components capture 69% of the differences between models. We interpret PC1 as how competitive a model is and PC2 as how much it trusts its opponent.

Table D.1: Strategic disposition metrics and their loadings on the first two principal components. All metrics are averaged across rounds and lie in [0, 1].
<table><tr><td>Game</td><td>Disposition</td><td>Definition</td><td>PC1</td><td>PC2</td></tr><tr><td>Chicken Game</td><td>Risk appetite</td><td>Fraction of rounds choosing dare over swerve.</td><td>+0.35</td><td>-0.12</td></tr><tr><td rowspan="2">Cournot Duopoly</td><td>Cooperativeness</td><td>Collusion index; 1 = joint-profit quantity, 0 = at or beyond Nash.</td><td>-0.45</td><td>-0.41</td></tr><tr><td>Aggressiveness</td><td>Own minus opponent quantity, as a share of the maximum quantity.</td><td>+0.49</td><td>+0.35</td></tr><tr><td rowspan="2">Stag Hunt</td><td>Risk appetite</td><td>Fraction of rounds choosing stag over the safe hare.</td><td>-0.38</td><td>+0.50</td></tr><tr><td>Cooperativeness</td><td>Fraction of rounds landing on mutual stag.</td><td>-0.34</td><td>+0.54</td></tr><tr><td rowspan="2">Vickrey Auction</td><td rowspan="2">Risk appetite Rationality</td><td>Mean own bid as a share of the maximum bid.</td><td>+0.38</td><td>+0.29</td></tr><tr><td>One minus the mean |bid — value|; 1 = always bids its true value.</td><td>+0.19</td><td>-0.28</td></tr><tr><td>Explained variance</td><td></td><td></td><td>43%</td><td></td></tr></table>

![](images/88578a88b5b154a108f82ae3fd331953b7e527b5ef7c2386fe8feca114cff95e.jpg)

![](images/f464d44a829b283950aa23ce078ae822e64fb2ec7436c8ac42b95d7415f70a29.jpg)

![](images/84ae2985b881203f7c0a2da0f654ec9331e4a71b923e75d3e768634cb416f2da.jpg)

![](images/70b2c979ca90f48778e80b8c67b782711166bf078506004c8a9d0b21a4ac831a.jpg)  
Figure D.1: Normalised in-game utility v ↑ against the disposition. Dashed lines show the linear fit and r its Pearson correlation. Granite, Mistral, Llama, Qwen3.5.

![](images/067cc384528b0d4dfcbe4b623d1d06d805a74ea834426571f1cf1496cc2890df.jpg)  
Figure D.2: Strategic dispositions inferred from in-game behaviour for each model family. Each axis is min-max normalised across the 10 models of the portfolio. Granite, Mistral, Llama, Qwen3.5.

<table><tr><td colspan="6">Reports normalised cost c ↓</td></tr><tr><td>Granite4.1 3B</td><td> ${ \bf 0 . 0 \pm 0 . 0 }$ </td><td> $0 . 1 \pm 0 . 0$ </td><td> ${ \bf 0 . 0 \pm 0 . 0 }$ </td><td> ${ \bf 0 . 0 \pm 0 . 0 }$ </td><td> ${ \bf 0 . 0 \pm 0 . 0 }$ </td></tr><tr><td>Granite4.1 8B</td><td> $5 . 0 \pm 0 . 0$ </td><td> $1 . 5 \pm 0 . 0$ </td><td> $1 3 . 9 \pm 0 . 3$ </td><td> $2 1 . 6 \pm 1 . 1$ </td><td> $2 . 0 \pm 0 . 1$ </td></tr><tr><td>Granite4.1 30B</td><td> $1 3 . 3 \pm 0 . 0$ </td><td> $2 . 4 \pm 0 . 1$ </td><td> $1 6 . 1 \pm 0 . 9$ </td><td> $1 6 . 7 \pm 0 . 0$ </td><td> $9 . 3 \pm 0 . 4$ </td></tr><tr><td>Ministral3 3B</td><td> $1 3 . 4 \pm 0 . 2$ </td><td> $1 . 7 \pm 0 . 0$ </td><td> $1 1 . 4 \pm 0 . 2$ </td><td> $1 6 . 6 \pm 0 . 2$ </td><td> $3 . 6 \pm 0 . 1$ </td></tr><tr><td>Mistral-Nemo 12B</td><td> $0 . 1 \pm 0 . 0$ </td><td> $0 . 2 \pm 0 . 0$ </td><td> $1 . 2 \pm 0 . 1$ </td><td> $0 . 4 \pm 0 . 1$ </td><td> $0 . 3 \pm 0 . 0$ </td></tr><tr><td>Mistral-Small 3.2 24B</td><td> $2 7 . 4 \pm 0 . 8$ </td><td> $1 . 3 \pm 0 . 0$ </td><td> $1 7 . 7 \pm 0 . 7$ </td><td> $1 6 . 3 \pm 0 . 8$ </td><td> $4 . 2 \pm 0 . 1$ </td></tr><tr><td>Llama3.1 8B</td><td> $0 . 2 \pm 0 . 0$ </td><td> $0 . 1 \pm 0 . 0$ </td><td> $0 . 9 \pm 0 . 0$ </td><td> $0 . 4 \pm 0 . 0$ </td><td> $0 . 3 \pm 0 . 0$ </td></tr><tr><td>Llama3.3 70B</td><td> $1 3 . 6 \pm 0 . 0$ </td><td> $3 . 9 \pm 0 . 1$ </td><td> $1 7 . 5 \pm 0 . 6$ </td><td> $1 7 . 0 \pm 0 . 0$ </td><td> $6 . 4 \pm 0 . 2$ </td></tr><tr><td>Qwen3.5 9B</td><td> $1 3 . 6 \pm 0 . 0$ </td><td> $1 . 5 \pm 0 . 0$ </td><td> $7 . 4 \pm 0 . 0$ </td><td> $1 7 . 2 \pm 0 . 0$ </td><td> $2 . 0 \pm 0 . 0$ </td></tr><tr><td>Qwen3.5 35B</td><td> $2 2 . 8 \pm 0 . 0$ </td><td> $6 . 0 \pm 0 . 3$ </td><td> $1 3 . 2 \pm 0 . 0$ </td><td> $2 7 . 8 \pm 0 . 0$ </td><td> $3 . 7 \pm 0 . 2$ </td></tr></table>

## D.2 Fixed Model vs. Fixed Model

Table D.2: Net gain, normalised utility, and normalised cost acquired by each model across 5 games. We report the mean ± 95% CI, ×100, aggregated across all pairwise matchups, using T = 1 round, N = 300 seeds per run. Highlight marks the best score per game within each metric block.
<table><tr><td>Model</td><td>Chicken Game</td><td></td><td>Colonel Blotto Cournot Duopoly Stag Hunt Vickrey Auction</td><td></td><td></td></tr><tr><td colspan="6">Reports normalised gain g ↑</td></tr><tr><td>Granite4.1 3B</td><td> ${ \bf 9 3 . 8 \pm 0 . 1 }$ </td><td> $7 5 . 2 \pm { \bf 0 . 1 }$ </td><td> $6 3 . 0 \pm 0 . 6$ </td><td> ${ \bf 8 7 . 2 \pm 1 . 4 }$ </td><td> ${ \bf 8 7 . 2 \pm 0 . 3 }$ </td></tr><tr><td>Granite4.1 8B</td><td> $8 2 . 3 \pm 1 . 5$ </td><td> $7 4 . 5 \pm 0 . 1$ </td><td> $4 6 . 9 \pm 0 . 8$ </td><td> $7 2 . 0 \pm 1 . 2$ </td><td> $8 5 . 8 \pm 0 . 3$ </td></tr><tr><td>Granite4.1 30B</td><td> $8 7 . 0 \pm 0 . 1$ </td><td> $7 4 . 0 \pm 0 . 1$ </td><td> $4 8 . 6 \pm 0 . 9$ </td><td> $7 8 . 1 \pm 1 . 4$ </td><td> $8 0 . 1 \pm 0 . 3$ </td></tr><tr><td>Ministral3 3B</td><td> $8 6 . 5 \pm 0 . 4$ </td><td> $7 3 . 4 \pm 0 . 2$ </td><td> $5 3 . 5 \pm 1 . 0$ </td><td> $7 8 . 2 \pm 1 . 4$ </td><td> $8 4 . 2 \pm 0 . 3$ </td></tr><tr><td>Mistral-Nemo 12B</td><td> $9 2 . 1 \pm 0 . 6 $ </td><td> $7 5 . 0 \pm 0 . 1$ </td><td> $5 9 . 4 \pm 1 . 2$ </td><td> $8 6 . 2 \pm 1 . 4$ </td><td> $8 5 . 8 \pm 0 . 3$ </td></tr><tr><td>Mistral-Small 3.2 24B</td><td> $7 7 . 4 \pm 0 . 9$ </td><td> $7 4 . 2 \pm 0 . 1$ </td><td> $5 8 . 8 \pm 1 . 4$ </td><td> $6 9 . 9 \pm 0 . 7$ </td><td> $8 4 . 2 \pm 0 . 3$ </td></tr><tr><td>Llama3.1 8B</td><td> $8 8 . 4 \pm 1 . 1$ </td><td> $7 4 . 8 \pm 0 . 2$ </td><td> ${ \bf 6 9 . 8 \pm 1 . 4 }$ </td><td> $8 2 . 0 \pm 1 . 0$ </td><td> $8 5 . 6 \pm 0 . 3$ </td></tr><tr><td>Llama3.3 70B</td><td> $7 8 . 2 \pm 1 . 4$ </td><td> $7 3 . 2 \pm 0 . 1$ </td><td> $6 5 . 7 \pm 1 . 5$ </td><td> $7 9 . 3 \pm 1 . 3$ </td><td> $8 3 . 7 \pm 0 . 3$ </td></tr><tr><td>Qwen3.5 9B</td><td> $7 9 . 2 \pm 1 . 4$ </td><td> $7 4 . 0 \pm 0 . 2$ </td><td> $5 7 . 9 \pm 1 . 1$ </td><td> $7 8 . 8 \pm 1 . 3$ </td><td> $8 5 . 5 \pm 0 . 4$ </td></tr><tr><td>Qwen3.5 35B</td><td> $8 0 . 5 \pm 0 . 7$ </td><td> $7 2 . 2 \pm 0 . 2$ </td><td> $6 4 . 4 \pm 1 . 4$ </td><td> $7 4 . 7 \pm 1 . 3$ </td><td> $8 3 . 8 \pm 0 . 4$ </td></tr></table>

$$
{ \bf 8 7 . 5 \pm 0 . 3 }
$$

$$
5 0 . 6 \pm 0 . 3
$$

$$
2 6 . 0 \pm 1 . 2
$$

$$
6 9 . 6 \pm 3 . 0
$$

$$
7 4 . 3 \pm 2 . 9
$$

$$
5 0 . 5 \pm 0 . 3
$$

$$
7 . 8 \pm 1 . 5
$$

$$
6 5 . 6 \pm 2 . 2
$$

$$
{ \bf 7 4 . 3 \pm 0 . 6 }
$$

$$
8 7 . 3 \pm 0 . 3
$$

$$
5 0 . 4 \pm 0 . 2
$$

$$
1 3 . 2 \pm { 1 . 5 }
$$

$$
7 2 . 9 \pm 2 . 9
$$

$$
7 3 . 7 \pm 0 . 6
$$

$$
8 6 . 4 \pm 0 . 7
$$

$$
4 8 . 5 \pm 0 . 5
$$

$$
1 8 . 4 \pm 2 . 0
$$

$$
6 9 . 4 \pm 0 . 5
$$

$$
7 3 . 0 \pm 2 . 8
$$

$$
8 4 . 2 \pm { 1 . 2 }
$$

$$
5 0 . 3 \pm 0 . 2
$$

$$
2 0 . 0 \pm 2 . 3
$$

$$
7 2 . 0 \pm 0 . 6
$$

$$
7 2 . 7 \pm 2 . 8
$$

$$
8 2 . 2 \pm { 1 . 7 }
$$

$$
4 9 . 7 \pm 0 . 3
$$

$$
3 5 . 2 \pm 2 . 8
$$

$$
5 6 . 1 \pm 1 . 2
$$

$$
7 1 . 9 \pm 0 . 6
$$

$$
7 7 . 1 \pm 2 . 3
$$

$$
4 9 . 8 \pm 0 . 4
$$

$$
4 0 . 4 \pm 2 . 7
$$

$$
6 4 . 5 \pm 2 . 0
$$

$$
7 2 . 5 \pm 0 . 6
$$

$$
6 9 . 9 \pm 2 . 9
$$

$$
5 0 . 4 \pm 0 . 2
$$

$$
{ \bf 4 8 . 9 \pm 2 . 9 }
$$

$$
7 1 . 5 \pm 0 . 7
$$

$$
7 2 . 0 \pm 2 . 8
$$

$$
7 5 . 7 \pm 2 . 6
$$

$$
4 9 . 6 \pm 0 . 4
$$

$$
8 3 . 7 \pm 1 . 4
$$

$$
7 3 . 8 \pm 0 . 6
$$

$$
2 3 . 2 \pm 2 . 3
$$

$$
5 0 . 3 \pm 0 . 3
$$

$$
7 4 . 7 \pm 2 . 6
$$

$$
4 2 . 0 \pm 2 . 8
$$

$$
7 7 . 2 \pm 2 . 6
$$

$$
7 3 . 0 \pm 0 . 6
$$

$$
7 1 . 4 \pm 0 . 7
$$

![](images/544767efe6899a96d9699d0bca16e819faec37459072e2017f076d24bf2748cf.jpg)

![](images/657ea73d8d9e8b736646971e0e7e55ad2fb25a9453ed087f4a9f66f5edc46977.jpg)

![](images/22074b4fc3ee1f8c433574eff4ba8be4e0a58a657848b5e527da5fc730933d22.jpg)  
Figure D.3: Model elo score (bootstrapped) vs. scale. Granite, Mistral, Llama, Qwen3.5.

![](images/1cee7bede8f45b5fe5169f3bb0970a0ae0a34c32ae07e38f58b73c5b4578f1de.jpg)

![](images/0b940ec2203e0c6d3739d249376dd4f092cbb0b400c75cc77a3f4a01730fdb7e.jpg)

## D.3 Selector vs. Fixed Model

Table D.3: Normalised net gain g ↑ (top) and in-game utility $v \uparrow$ (bottom) acquired by each selector and fixed model from full pairwise matchups against each other, and across 5 games. We show the mean ± 95% CI, ×100, using $T = 3 0 0$ rounds, $N = 1$ seed per run, and highlight the best score for each game. Selectors’ learning signal was set as their per-round gain $g _ { i , t } .$
<table><tr><td>Method</td><td></td><td></td><td>Chicken Game Colonel Blotto Cournot Duopoly Stag Hunt Vickrey Auction</td><td></td><td></td></tr><tr><td colspan="6">Reports normalised gain g ↑</td></tr><tr><td>Per-Opponent Full-Information</td><td> $9 2 . 8 \pm 0 . 4$ </td><td> $7 3 . 0 \pm 0 . 2$ </td><td> $8 0 . 7 \pm { 1 . 5 }$ </td><td> $9 2 . 2 \pm 1 . 0$ </td><td> $8 6 . 7 \pm 0 . 5$ </td></tr><tr><td>Granite4.1 3B</td><td> $8 9 . 7 \pm 0 . 1$ </td><td> $7 1 . 6 \pm 0 . 1$ </td><td> $6 4 . 0 \pm 0 . 5$ </td><td> $8 9 . 2 \pm 0 . 8$ </td><td> $8 3 . 4 \pm 0 . 2$ </td></tr><tr><td>Granite4.1 8B</td><td> $8 4 . 5 \pm 0 . 8$ </td><td> $7 0 . 5 \pm 0 . 1$ </td><td> $4 9 . 8 \pm 0 . 5$ </td><td> $8 1 . 1 \pm 0 . 7$ </td><td> $8 2 . 0 \pm 0 . 2$ </td></tr><tr><td>Granite4.1 30B</td><td> $7 7 . 3 \pm 0 . 1$ </td><td> $5 9 . 1 \pm 0 . 1$ </td><td> $4 0 . 9 \pm 0 . 5$ </td><td> $7 6 . 6 \pm 0 . 8$ </td><td> $6 8 . 6 \pm 0 . 2$ </td></tr><tr><td>Ministral3 3B</td><td> $8 6 . 5 \pm 0 . 2$ </td><td> $6 7 . 3 \pm 0 . 2$ </td><td> $5 3 . 3 \pm 0 . 7$ </td><td> $8 3 . 6 \pm 0 . 8$ </td><td> $7 8 . 9 \pm 0 . 2$ </td></tr><tr><td>Mistral-Nemo 12B</td><td>1  ${ \bf 9 1 . 9 \pm 0 . 4 }$ </td><td> ${ \bf 7 5 . 1 \pm 0 . 1 }$  </td><td> $6 1 . 0 \pm 0 . 8$ </td><td> ${ \bf 9 0 . 9 \pm 0 . 8 }$ </td><td> ${ \bf 8 5 . 8 \pm 0 . 2 }$ </td></tr><tr><td>Mistral-Small 3.2 24B</td><td> $8 0 . 7 \pm 0 . 6$ </td><td> $6 4 . 2 \pm 0 . 1$ </td><td> $5 7 . 2 \pm 0 . 9$ </td><td> $6 6 . 6 \pm 0 . 4$ </td><td> $7 5 . 7 \pm 0 . 2$ </td></tr><tr><td>Llama3.1 8B</td><td> $9 0 . 4 \pm 0 . 7$ </td><td> $7 4 . 6 \pm 0 . 1$ </td><td>_  ${ \bf 7 2 . 6 \pm 0 . 9 }$ </td><td> $8 1 . 9 \pm 0 . 7$ </td><td> $8 5 . 6 \pm 0 . 3$ </td></tr><tr><td>Llama3.3 70B</td><td> $7 1 . 9 \pm 0 . 9$ </td><td> $5 8 . 2 \pm 0 . 1$ </td><td> $6 1 . 3 \pm 0 . 9$ </td><td> $7 3 . 8 \pm 0 . 8$ </td><td> $6 9 . 8 \pm 0 . 2$ </td></tr><tr><td>Qwen3.5 9B</td><td> $8 0 . 4 \pm 0 . 8$ </td><td> $6 5 . 7 \pm 0 . 1$ </td><td> $6 0 . 3 \pm { 0 . 8 }$ </td><td> $8 1 . 2 \pm 0 . 8$ </td><td> $7 7 . 0 \pm 0 . 2$ </td></tr><tr><td>Qwen3.5 35B</td><td> $4 2 . 5 \pm 0 . 5$ </td><td> $2 5 . 1 \pm 0 . 1$ </td><td> $2 5 . 1 \pm 0 . 9$ </td><td> $4 2 . 4 \pm 0 . 8$ </td><td> $3 5 . 6 \pm 0 . 3$ </td></tr><tr><td>Thompson Sampler (TS)</td><td> $8 5 . 1 \pm 0 . 5$ </td><td> $7 1 . 2 \pm 0 . 2$ </td><td> $4 7 . 7 \pm 0 . 8$ </td><td> $7 6 . 1 \pm 0 . 8$ </td><td> $7 4 . 8 \pm 0 . 1$ </td></tr><tr><td>RouterDC</td><td> $8 4 . 3 \pm 0 . 1$ </td><td> $6 8 . 6 \pm 0 . 1$ </td><td> $3 6 . 2 \pm 0 . 1$ </td><td> $7 4 . 4 \pm 0 . 8$ </td><td> $7 2 . 8 \pm 0 . 2$ </td></tr><tr><td>Causal LLM Router</td><td> $7 8 . 6 \pm 0 . 7$ </td><td> $6 3 . 2 \pm { 0 . 5 }$ </td><td> $3 1 . 3 \pm 0 . 7$ </td><td> $7 4 . 0 \pm 0 . 9$ </td><td> $6 4 . 8 \pm 0 . 5$ </td></tr><tr><td>LLM Selector</td><td> $6 1 . 0 \pm 0 . 6$ </td><td> $4 6 . 9 \pm 0 . 3$ </td><td> $2 6 . 9 \pm 0 . 5$ </td><td> $4 6 . 6 \pm 1 . 2$ </td><td> $5 4 . 4 \pm 0 . 2$ </td></tr><tr><td colspan="6">Reports normalised utility v ↑</td></tr><tr><td>Per-Opponent Full-Information</td><td> $8 8 . 9 \pm 0 . 7$ </td><td> $5 0 . 6 \pm 0 . 3$ </td><td> $4 3 . 0 \pm 2 . 6$ </td><td> $8 8 . 1 \pm 2 . 1 $ </td><td> $7 5 . 1 \pm 0 . 7$ </td></tr><tr><td>Granite4.1 3B</td><td> $8 6 . 7 \pm 0 . 2$ </td><td> $5 0 . 5 \pm 0 . 1$ </td><td> $3 5 . 3 \pm { 1 . 0 }$ </td><td> $8 5 . 6 \pm 1 . 5$ </td><td> ${ \bf 7 4 . 1 \pm 0 . 5 }$ </td></tr><tr><td>Granite4.1 8B</td><td> $7 8 . 2 \pm 1 . 6$ </td><td> $5 0 . 3 \pm 0 . 1$ </td><td> $8 . 8 \pm 1 . 0$ </td><td> $7 1 . 3 \pm 1 . 4$ </td><td> $7 3 . 2 \pm 0 . 5$ </td></tr><tr><td>Granite4.1 30B</td><td> $8 6 . 7 \pm 0 . 2$ </td><td> $5 0 . 3 \pm 0 . 1$ </td><td> $1 3 . 9 \pm 1 . 0$ </td><td> $8 5 . 4 \pm { 1 . 5 }$ </td><td> $6 9 . 4 \pm 0 . 4$ </td></tr><tr><td>Ministral3 3B</td><td> $8 6 . 8 \pm 0 . 5$ </td><td> $4 8 . 5 \pm 0 . 3$ </td><td> $2 0 . 3 \pm 1 . 4$ </td><td> $8 1 . 1 \pm 1 . 6$ </td><td> $7 1 . 6 \pm 0 . 5$ </td></tr><tr><td>Mistral-Nemo 12B</td><td> $8 3 . 7 \pm 0 . 8$ </td><td> $5 0 . 2 \pm 0 . 2$ </td><td> $2 2 . 1 \pm 1 . 5$ </td><td> $8 1 . 7 \pm { 1 . 7 }$ </td><td> $7 1 . 6 \pm 0 . 5$ </td></tr><tr><td>Mistral-Small 3.2 24B</td><td> $8 2 . 0 \pm 1 . 2$ </td><td> $4 9 . 0 \pm 0 . 2$ </td><td> $3 5 . 0 \pm 1 . 8$ </td><td> $5 3 . 9 \pm 0 . 8$ </td><td> $7 2 . 1 \pm 0 . 5$ </td></tr><tr><td>Llama3.1 8B</td><td> $8 0 . 8 \pm 1 . 4$ </td><td> $4 9 . 3 \pm 0 . 3$ </td><td> $4 5 . 2 \pm 1 . 7$ </td><td> $6 3 . 8 \pm 1 . 4$ </td><td> $7 1 . 2 \pm 0 . 5$ </td></tr><tr><td>Llama3.3 70B</td><td> $7 7 . 7 \pm 1 . 8$ </td><td> $5 0 . 4 \pm 0 . 1$ </td><td>_  ${ \bf 5 6 . 5 \pm 1 . 8 }$ </td><td>1  $8 1 . 5 \pm { 1 . 6 }$ </td><td> $7 3 . 6 \pm 0 . 5$ </td></tr><tr><td>Qwen3.5 9B</td><td> $7 9 . 1 \pm 1 . 6$ </td><td> $4 9 . 8 \pm 0 . 2$ </td><td> $3 9 . 0 \pm 1 . 7$ </td><td> $8 0 . 7 \pm 1 . 6$ </td><td> $7 2 . 4 \pm 0 . 5$ </td></tr><tr><td>Qwen3.5 35B</td><td> $8 5 . 0 \pm 0 . 9$ </td><td> $5 0 . 2 \pm 0 . 2$ </td><td> $5 0 . 2 \pm { 1 . 8 }$ </td><td> $8 4 . 8 \pm { 1 . 6 }$ </td><td> $7 1 . 2 \pm 0 . 6$ </td></tr><tr><td>Thompson Sampler (TS)</td><td> $8 7 . 4 \pm 0 . 7$ </td><td> $5 0 . 4 \pm 0 . 1$ </td><td> $3 9 . 4 \pm 1 . 4$ </td><td> $7 8 . 0 \pm 1 . 3$ </td><td> $7 3 . 4 \pm 0 . 3$ </td></tr><tr><td>RouterDC</td><td>1  ${ \bf 8 7 . 8 \pm 0 . 2 }$  </td><td> $5 0 . 5 \pm 0 . 1$ </td><td> $1 9 . 6 \pm 0 . 3$ </td><td> $7 6 . 1 \pm 1 . 5$ </td><td> $7 2 . 8 \pm 0 . 3$ </td></tr><tr><td>Causal LLM Router</td><td> $7 9 . 7 \pm 1 . 1$ </td><td> $5 0 . 1 \pm 0 . 2$ </td><td> $2 7 . 1 \pm 1 . 3$ </td><td> $7 2 . 4 \pm 1 . 4$ </td><td> $7 2 . 2 \pm 0 . 4$ </td></tr><tr><td>LLM Selector</td><td> $8 0 . 6 \pm 1 . 2$ </td><td> $5 0 . 1 \pm 0 . 2$ </td><td> $1 8 . 1 \pm { 1 . 0 }$ </td><td> ${ \bf 8 7 . 0 \pm 2 . 3 }$ </td><td> $7 1 . 5 \pm 0 . 5$ </td></tr></table>