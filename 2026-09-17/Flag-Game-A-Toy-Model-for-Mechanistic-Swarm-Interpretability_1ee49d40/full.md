# Flag Game: A Toy Model for Mechanistic Swarm Interpretability

Elizabeth Pavlova<sup>1,2,3</sup>

Hidenori Tanaka<sup>1,2</sup>

<sup>1</sup>CBS-NTT Program in Physics of Intelligence, Harvard University, MA, USA <sup>2</sup>Physics of Artificial Intelligence Laboratories, NTT Research, Inc., CA, USA <sup>3</sup>Cambridge Boston Alignment Initiative, MA, USA

## Abstract

Emergent coordinated behaviors of AI agents are starting to present critical safety risks. A key phenomenon driving these behaviors is the rapid formation and spread of beliefs about the world, and mechanistic understanding is crucial for collective alignment. To this end, we introduce the Flag Game, a toy model for studying the mechanisms of collective belief formation. Concretely, a hidden country flag defines the ground truth, and each bounded agent directly observes only a private crop but can exchange beliefs and weigh social evidence from peers. Despite its simplicity, the Flag Game reproduces rich collective phenomenology: non-monotonic scaling of performance with population size, accuracy gains from social-awareness prompting and team diversity, and strong effects of organizational structure. In particular, we identify that collective belief collapse at small population sizes turns into collective belief polarization as the population grows. This polarization causes the performance decline at large population sizes, but creates diversity in collective beliefs. Finally, we dissect the mechanisms underlying collective belief collapse and polarization with two complementary approaches. We first introduce social circuit attribution, a technique to predict which agent, and what view, matters most to collective dynamics, and verify its predictions by causal interventions on agents, tracing how agent patching changes collective outcomes. However, the efficacy of causal interventions on agents decreases as the population grows. We therefore develop a statistical mechanical theory for larger populations and verify that it matches the empirical phase diagram. Together, these results take a first step toward mechanistic swarm interpretability, a science of how the properties of individual agents and their communication give rise to emergent collective behavior.

## 1 Introduction

We are now, in real time, witnessing how “more is different” [Anderson, 1972] in swarms of AI agents, which can behave, in aggregate, in ways that none of their members were designed to. A representative example is the recent OpenAI Hugging Face incident [OpenAI, 2026]. According to an independent investigation [Greenblatt et al., 2026], a key driver of the coordinated cyberattack was the unintended social interaction between agents. A small number of agents formed a false belief based on local evidence, their reading of the benchmark paper [Wang et al., 2026]: the automated scorer would read their transcripts and disqualify solutions that did not use the intended vulnerability. In reality, no such check was implemented, but the false belief spread through the message board, and the coordinated effort to evade the imagined monitoring helped motivate a large-scale attack on third-party infrastructure. Critically, while a few agents expressed ethical hesitation, this rarely affected their behavior. Nor was this an isolated case: earlier in the same year, another swarm of OpenAI agents repurposed a public German wiki into a message board and exchanged tactics during May and June 2026 [Von Arx et al., 2026].

This social structure, in which bounded agents form beliefs from local evidence and spread them through a population, is not unique to AI. It has been studied for over a century in human collectives, from crowd behavior and popular delusions [Mackay, 1841, Le Bon, 1895] to conformity experiments [Asch, 1951], information cascades [Banerjee, 1992, Bikhchandani et al., 1992], and the spread of false news online [Vosoughi et al., 2018]. Throughout this paper, following the tradition of bounded rationality [Simon, 1955, 1957], we use bounded to mean that agents take in only partial observations of the world, deploy only finite computation, such as a limited token budget, and transmit only finite messages to their peers.

Such incidents motivate a scientific question of collective belief formation: how do beliefs form in an individual and evolve in a population? Post-hoc analysis of swarms in the wild is crucial, but it is not enough to get to a precise mechanism. We cannot control what each agent was able to see, private evidence cannot be reproduced, and the volume of messages can explode, exceeding 70,000 in the incident above [Greenblatt et al., 2026]. Moreover, the underlying mechanism could take many forms: none of the agents had meaningful evidence, some saw it but did not communicate it, they communicated it but got overridden by others, or, as in the incident, an early false interpretation spread and influenced subsequent coordination. What we want to understand is how the properties, even the personalities, of each agent lead to coordinated collective behavior, just as mechanistic interpretability asks how a network of neurons results in a decision. Now is the moment for mechanistic swarm interpretability, where agents correspond to neurons, social circuits, namely the organizational structure, correspond to neural circuits, and beliefs and messages correspond to activations. Following the success of the toy model approach in mechanistic interpretability, where the sudden emergence of capabilities in LLMs [Wei et al., 2022] was mirrored by grokking on a small algorithmic task [Power et al., 2022] and eventually reverse engineered [Nanda et al., 2023], we craft a toy model of a society of agents to get to the key mechanism underlying safety-related behavior at the macro scale.

Our toy model is the Flag Game (Fig. 1), a model organism of a society of bounded agents. Its defining features are: (i) the external world has a verifiable ground truth; (ii) agents are bounded, each holding only an uncertain, partial view of the world; and (iii) each agent forms a belief from its private evidence and spreads it to peers under a specified communication protocol. The first is what we lack in the wild, and what gives us control: the experimenter knows the ground truth and decide exactly what each agent sees. In the incident above, no agent had access to the ground truth of how the scorer worked; the benchmark paper provided a private crop of it. The agents who formed the belief that the scorer would disqualify their solutions spread it to the population through the message board, a broadcast protocol, and this false belief was a rival, compatible with local evidence, just as a rival country can be compatible with a crop. Moreover, many of the agents had been assigned tasks that were impossible to solve [Greenblatt et al., 2026], just as an agent whose crop is uninformative has no private evidence and nothing to rely on but peers. What remains is the fundamental tension every bounded agent faces: how to balance its private evidence against social input from peers.

Contributions. We make three contributions in this work.

1. The Flag Game: a model organism for collective belief formation in societies of bounded agents (Sec. 3). Unlike standard multi-agent debate, where every agent receives the same problem, each bounded agent receives its own private crop of a verifiable ground truth. Because the experimenter controls who sees what, the evidence in the population can be easily modeled, and the evidence or belief of any single agent can be easily ablated or patched while everything else is held fixed. This facilitates mechanistic analysis.

2. The Flag Game reproduces an array of multi-agent phenomena despite its simplicity (Sec. 4). Collective performance scales non-monotonically with population size; socialawareness prompting improves collective accuracy; teams with a mixture of agents outperform homogeneous ones; and organizational structure plays a key role.

3. Mechanistic Swarm Interpretability of societies of bounded agents (Sec. 5). We distinguish two failure modes: collective belief collapse, in which the population converges on a single false belief, and collective belief polarization, in which agents split between competing beliefs. As the population grows, collapse becomes rarer, but a false belief compatible with

b) Accuracy with population scaling is non-monotonic

→  
![](images/37d7e223f9a36b7eb6d3f14a3bdc6d924ee758fae602d42fa276b19022dd955a.jpg)  
a) Agents observe partial views and interact with one another to reach collective decision

![](images/5df506bc71fee6de1231eed53f9428e9bdd8b6a5c5aa435ec603e747b044f6d9.jpg)  
d) Diverse mixture of agents adds complementary skills

![](images/f90164d463cb6fe126f7d0931397502898798378ec4940d188a5f3b3bda7bed1.jpg)

c) Social awareness prompting improves evidence uptake  
![](images/d943b91ee495f4896cee1c0b9065f84a303cb07ce846c3f3294657b242d055a5.jpg)

![](images/166998d789da9162ab684129b68acfbb41800d3338c6144735864e947519f3a8.jpg)  
e) Same agents with different structures result in better outcomes

![](images/15c6c9aca7830c0a659840fc90457100087892f220d2601185daae347f6f85eb.jpg)  
Figure 1: The Flag Game (demo website: https://flag-game-demo.vercel.app) exhibits rich collective phenomenology despite its simplicity. (a) A hidden flag is observed through private evidence; bounded agents exchange reports, and the system tracks agent’s country guess. (b–e) Collective mean accuracy (Sec. 3) under four controls: population size (b), social-awareness prompting (c), team diversity (d), and organizational structure (e).

the local evidence spreads through the population, splitting agents between truth and a rival, which we call truth–rival polarization. We dissect this with two complementary approaches. (a) Causal interventions on agents: social circuit attribution, a technique to predict which agent, and what it sees, matters most, verified by agent patching. As their efficacy decreases with larger populations (Sec. 5.1), it calls for (b) statistical mechanics of bounded agents, a simple mathematical model of the non-monotonic population scaling and the underlying polarization dynamics that matches the empirical phase diagram (Sec. 5.2).

Before proceeding, we emphasize the trade-offs of this approach. Just as mechanisms found in biological model organisms cannot be applied directly to the human body, our observations in the Flag Game should not be taken as conclusions about deployed swarms of AI agents. The Flag Game is a first step toward a mechanistic approach to collective behavior in multi-agent systems, rather than a complete mechanistic explanation. Our aim is to establish a conceptual framework, identify the control variables (population size, social-evidence uptake, message bandwidth, team composition, and communication protocol), and formulate mechanistic hypotheses that can be tested both by targeted interventions in the Flag Game and in more open-ended multi-agent systems.

## 2 Related work

Multi-agent collective reasoning. Multi-agent debate can improve factual and mathematical reasoning by bringing together a variety of different answers and rationales [Du et al., 2024, Liang et al., 2024]. Performance gains, though, are not guaranteed as interaction can move agents from correct to incorrect answers, or fail to outperform simpler voting or ensembling baselines [Wynn et al., 2025, Yao et al., 2025, Smit et al., 2024, Choi et al., 2025]. Prior work has shown that heterogeneous teams can outperform homogeneous ones, and outcomes can vary with social prompting, communication protocols and structure [Chen et al., 2024, Kasprova et al., 2026, Kaesberg et al., 2025]. Pure sampling-and-voting can also scale monotonically with agent count even when interactive systems do not [Li et al., 2024]. Thus, the phenomena studied in this paper—population scaling, social guidance, model composition, and organization—have precedents in prior multi agent settings. The distinctive leverage of the Flag Game is instead that each agent’s private evidence is assigned and can be replayed across matched conditions.

Controlled partial information and social updating. The Flag Game also connects to hiddenprofile experiments, where groups under-use individually held information, and to social-learning and herding models, where group signals can override private evidence [Stasser and Titus, 1985, 2003, Banerjee, 1992, Bikhchandani et al., 1992]. Control over flag crops provides a natural mix of partial observations with a single verifiable answer, while the same crop assignment can be held fixed as population size, composition, prompting, or protocol changes. This control lets us distinguish evidence that was absent, present but unshared, overridden after communication, or stabilized into competing truth and rival supporting camps. Additionally, models of consensus and opinion pooling address how individual judgments are revised through interaction and combined into a collective judgment [DeGroot, 1974, Dietrich and List, 2017, Stewart and Ojea Quintana, 2018] and we can study both processes against a known country label in the Flag Game.

## 3 The Flag Game: a model organism for collective belief formation

## 3.1 Trial structure

Each trial samples a hidden flag image x with country label $y ^ { \star } \in \mathcal { V }$ , where Y is the fixed set of country labels in the experiment. The full flag is hidden from the agents. Each agent $i \in \{ 1 , \ldots , N \}$ instead receives a private crop $c _ { i } = R _ { i } x$ , where $R _ { i }$ is the crop randomly assigned to that agent. These can range from highly ambiguous to strongly diagnostic, creating a controlled mix in the evidence available to individual agents while also ensuring no agent sees the full flag.

In Fig. 2a, we see that each trial has four stages: sample the hidden target flag, assign private evidence to individual agents, elicit initial guesses, and run the communication protocol until a terminal readout is produced. The protocol returns either a population distribution over agent reports or a manager’s final answer, depending on the organization structure. A run has a maximum number of probe rounds t proportional to the population size, $T _ { \mathrm { m a x } } = \kappa N$ , for some constant κ. Trajectory plots use the normalized axis $t / N$ so runs with different $N$ are comparable. A run terminates early if five consecutive probes result in a full country consensus of 100%.

![](images/c6bb7e53c321063cf1ffe0a5cb279de961f1712f6e7d1033612f927ff7226c59.jpg)

b)  
![](images/2e2bfecdae34fcba12e6dacd0274c2e18f5bc817850c369c5e13ada1e6994dd0.jpg)  
Figure 2: Protocol structure and performance in the Flag Game. (a) Each trial samples a hidden flag, assigns private crops, elicits initial guesses, and then runs one of three protocols. (b) A $N = 8$ sweep across 60 trials compares accuracy, ordering by descending collective mean accuracy.

## 3.2 Communication protocols

The Flag Game separates the visual task from the social organization. The same hidden flag and private crops can be run under different communication protocols as seen in Fig. 2a.

Pairwise. The pairwise game is asynchronous and local, matching the randomized local-exchange structure common in gossip algorithms [Boyd et al., 2006, Tanaka, 2026]. At each interaction t, one speaker and one listener are sampled. The speaker sees its own crop and transcript memory, then emits a message: a country guess for $m = 1$ or a country plus reason for $m = 3$ , where we define m as the message bandwidth parameter. The listener appends that message to its memory, of max length $H = 8$ . Periodic probes ask agents for country guesses using their private crop and accumulated local memory, and the endpoint is the empirical distribution over terminal agent answers.

Broadcast. Broadcast exchange is synchronous, closer to classical social-influence models where agents are exposed to a shared view of all opinions [DeGroot, 1974, Friedkin and Johnsen, 1990]. Each round, every agent gives a country report from its crop and private memory of its own past final decisions, and then sees the current reports of the other agents. Agents retain their own crops and private memories. The endpoint is again a distribution over all agents. Broadcast removes private pairwise interactions as a bottleneck, but does not guarantee agents use available evidence correctly.

Manager. The manager protocol adds a blind decision-maker, analogous to a moderator or supervisor used in multi-agent medical and legal systems [Tang et al., 2024, Kim et al., 2024, Chen et al., 2025, Jiang and Yang, 2025, He et al., 2024]. There are N observers that give country-reason reports. The manager sees these reports and its own prior decisions, but never a crop. It emits a country decision per round, which becomes shared memory for the observers. The endpoint is the manager’s answer. This protocol tests centralized synthesis rather than population-level convergence.

## 3.3 Controls and observables

We manipulate five experimental controls throughout our work (see Table 1): population size, message bandwidth, prompted social-evidence uptake, team composition, and communication protocol.

Let $\hat { y } _ { i } ^ { ( 0 ) }$ be agent i’s initial guess before social interactions. In population protocols, where $p _ { \mathrm { f i n a l } }$ is the distribution over agents’ final country guesses, we define

$$
s _ { 1 } = \operatorname* { m a x } _ { y \in \mathcal { Y } } p _ { \mathrm { f i n a l } } ( y ) , \qquad y _ { 1 } = \arg \operatorname* { m a x } _ { y \in \mathcal { Y } } p _ { \mathrm { f i n a l } } ( y ) .
$$

For the manager protocol, let $y _ { \mathrm { m g r } }$ denote the manager’s final country answer which is correct when $y _ { \mathrm { m g r } } = y ^ { \star }$

<table><tr><td>Control</td><td>Meaning</td></tr><tr><td>N</td><td>Number of observer agents</td></tr><tr><td>α</td><td>Social-evidence uptake</td></tr><tr><td>m</td><td>Message bandwidth</td></tr><tr><td>Composition</td><td>Model/role mix</td></tr><tr><td>Protocol</td><td>Pairwise/broadcast/manager</td></tr></table>

Table 1: Control variables.

correct consensus $( s _ { 1 } \geq 0 . 8 5 , y _ { 1 } = y ^ { \star } )$ , wrong consensus $( s _ { 1 } \geq 0 . 8 5 , y _ { 1 } \neq y ^ { \star } )$ , polarization $( s _ { 1 } < 0 . 8 5$ with at least two countries holding mass $\geq 0 . 2 5 )$ ), or fragmentation (otherwise). We refer to polarized endpoints in which the two supported camps correspond to the true country and a plausible rival as truth–rival polarization, and we treat wrong consensus as the operational definition for collective belief collapse. Appendix D measures threshold robustness from $0 . 7 5 - 1 . 0 0$ for consensus and $0 . 1 5 - 0 . 3 5$ for polarization, justifying our selection.

We report majority vote accuracy over the isolated initial guesses:

$$
y _ { \mathrm { m a j } } ^ { ( 0 ) } = \arg \operatorname* { m a x } _ { y \in \mathcal { V } } \sum _ { i = 1 } ^ { N } \mathbf { 1 } \{ \hat { y } _ { i } ^ { ( 0 ) } = y \} , \qquad A _ { \mathrm { m a j } } = \mathbb { E } \left[ \mathbf { 1 } \{ y _ { \mathrm { m a j } } ^ { ( 0 ) } = y ^ { \star } \} \right] ,\tag{1}
$$

We report the initial mean accuracy of the crop-bearing observers (excluding the manager):

$$
A _ { \mathrm { i n i t } } = \mathbb { E } \left[ { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } \mathbf { 1 } \{ { \hat { y } } _ { i } ^ { ( 0 ) } = y ^ { \star } \} \right] .\tag{2}
$$

For pairwise and broadcast, we report the terminal truth mass and for the manager protocol, we report the manager’s exact-answer accuracy, and we refer to both as collective mean accuracy:

$$
A _ { \mathrm { c o l l } } = \mathbb { E } \left[ p _ { \mathrm { f i n a l } } ( y ^ { \star } ) \right] , \qquad A _ { \mathrm { m g r } } = \mathbb { E } \left[ { \bf 1 } \{ y _ { \mathrm { m g r } } = y ^ { \star } \} \right] .\tag{3}
$$

We define social uplift as the change from isolated mean accuracy to collective mean accuracy:

$$
\Delta _ { \mathrm { s o c i a l } } = \left\{ \begin{array} { l l } { { A _ { \mathrm { c o l l } } - A _ { \mathrm { i n i t } } , } } & { { \mathrm { p a i r w i s e ~ o r ~ b r o a d c a s t } , } } \\ { { A _ { \mathrm { m g r } } - A _ { \mathrm { i n i t } } , } } & { { \mathrm { m a n a g e r } . } } \end{array} \right.\tag{4}
$$

![](images/5ad7027bbfd5376d4f2ad785d84e6c40be206d7596fad2f50afad2351bb29c63.jpg)  
Figure 3: With large N, polarization reduces accuracy. (a) In all-GPT-4o pairwise runs, collective accuracy is higher than initial accuracy, peaking at intermediate N. (b) As N grows, wrong consensus becomes less common while polarization become more frequent. (c–d) A representative France–Peru run at N = 4, 16, 64 shows final predictions over initial crop locations and country-share trajectories.

## 4 Exploring multi-agent phenomenology in the Flag Game

The findings below show how the private–social balance plays out under four controls: population size, social-awareness prompting, team composition, and protocol and role assignment.

Non-monotonic population scaling. See Fig. 3a. Increasing group size does not produce monotonic gains. In the pairwise condition, collective mean accuracy peaks at the intermediate population size, N = 16, before declining at larger N. The failure-reason chart in Fig. 3b shows that this decline is not accompanied by increasing wrong consensus. As N grows, wrong consensus becomes less common, while runs increasingly end in split states where both the truth and a plausible rival retain substantial social support. We examine a possible mechanism for this decline in Sec. 5.

Social-awareness prompting. See Fig. 1c. The social-awareness sweep intervenes on the instructions governing the private–social balance (prompt in Table 4). In the broadcast protocol, all agents have access to the same public set of current-round reports. Across the social-awareness ladder, terminal truth mass rises from 0.54 to 0.81, showing that instructions for interpreting an unchanged set of peer reports materially affect collective performance. The pairwise version of the sweep is reported in Appendix B. There, performance has an interior optimum, indicating that more strongly encouraging reliance on peers is not uniformly beneficial under local exchange and can amplify incorrect reports.

Team diversity. See Fig. 4a. The composition sweep tests whether collective performance increases with using more of the visually strongest individual model. We see that this is not the case, as the best-performing teams are mixed across GPT-4o and GPT-5.4 agents. This pattern suggests complementarity rather than a simple model ranking. In Fig. 4b, the crop-only probe shows a behavioral difference between the models. GPT-4o and GPT-5.4 receive the same flag crops and achieve similar country accuracy, but their errors differ. GPT-4o’s incorrect responses are more visually compatible with the crop, while GPT-5.4 produces more incompatible guesses. GPT-4o therefore appears to be more locally anchored to visual evidence.

The memory intervention in Sec. 5.1 reveals a second difference in how the two models update on social input, adding further evidence to the complementarity account for mixed-team gains.

Organizational structure. See Fig. 2b. The protocol sweep with population N = 8 and fixed crops varies model composition and how social input flows to agents, pairwise (local), broadcast (public), or blind manager synthesis. The clearest contrast is the model identity of the manager. Holding observers fixed, GPT-5.4 managers reach 0.57 accuracy, whereas GPT-4o managers remain around 0.48, reflecting a role-specific difference in how managers synthesize reports. In the population protocols, the ranking reverses with the protocol: all-GPT-4o outperforms all-GPT-5.4 under pairwise, with flipped order for broadcast. Thus, there is no single ordering of model strength that explains the results across organizations. Taken together, these results show that model identity cannot be separated from role. In manager runs, the identity of the blind synthesizer is especially important; in population protocols, the same agents both observe and update, so performance depends on the joint dynamics of private evidence, social uptake, and communication structure.

a) Broadcast protocol accuracy with team diversity c) GPT-5.4's visual description can be ungrounded, while 4o stays grounded  
![](images/adf4f34c527ca3262bec6005f141df9a8205eb6b6cb202e561a0a539e8320ce3.jpg)  
Figure 4: Single-agent test reveals complementary error modes behind diverse team gains. (a) Broadcast team-diversity result for $N = 8 , m = 3 ,$ , showing mixed teams achieve highest collective accuracy. (b) Isolated country responses for GPT-4o and GPT-5.4 over 100 runs. Exact accuracy is similar, but GPT-5.4 produces more visually incompatible answers. (c) Examples illustrating cases where GPT-4o chooses the correct country while GPT-5.4 misconstrues its private evidence.

## 5 Mechanistic swarm interpretability: dissecting the spread of false beliefs

Next, we try to understand the mechanisms underlying non-monotonic collective performance as we scale the population (per Fig. 3b). In the France–Peru example (Fig. 3c–d), we saw that N = 4 does not have enough decisive evidence, N = 16 reaches correct France consensus, and $N = 6 4$ results in a polarizing France–Peru split. Observers can add support for the truth and a rival at the same time, changing the value of the same communication protocol. This raises questions at two different scales. For an individual agent, how does it weigh what it sees against what it hears, and how far does its evidence travel through the swarm? Across the population, why do answers change with scaling N?

We address the first with causal interventions on agents’ memory and crops, and tracing how each change propagates (Sec. 5.1). These interventions localize where a correction enters a small group, but their efficacy decreases as the population grows. We address the second with the statistical mechanics of collective belief formation in societies of bounded agents (Sec. 5.2).

## 5.1 Causal interventions on agents

Mechanistic interpretability techniques work by intervening on a network’s internals, fixing an activation and observing downstream change. We apply the same logic to a swarm, whose internals are its agents. Each agent’s answer depends on two inputs: the private evidence in its crop and the social evidence in its memory. Editing memory while holding the visual fixed, steers one agent with a controlled share of social evidence and read out its answer, showing when it holds its private evidence and when it copies. Editing an agent’s crop traces planted evidence through the swarm. As in activation patching [Meng et al., 2022, Zhang and Nanda, 2024], we replace one input, hold everything else fixed, and follow the change through to the collective outcome. Together they show what an agent does with what it hears, and how far what one agent says can travel.

Local social memory probing. Whether an agent resists or follows social evidence is a property of the model as much as of the crop. Fig. 5 tests these update behaviors by fixing a private crop, while the target:social ratio in an agent’s memory changes. Under weak private evidence, GPT-5.4 shows the highest rate of compatibility reasoning, routing probability into other countries rather than copying the social label. Under strong private evidence (crop uniquely identifies the target), GPT-4o and GPT-5.4 hold firm, while Claude Haiku 4.5 abandons the private target as social memory accumulates, a signature of sycophantic override. We extend this probe in Appendix C with a control condition where social evidence agrees with the private target. This effect may help explain the team diversity result of Sec. 4: pairing a literal listener with a compatibility-reasoning listener can identify truth-supporting evidence that neither homogeneous team has, consistent with the gains in Fig. 4a.

![](images/df8321a332661d308bb03174ff39115de823a7a44a7fe38638319f9af3b11fdb.jpg)  
Figure 5: Local social memory probe isolates update behavior conditional on private evidence. (a) A target flag gives private evidence to the agent, while memory contains a shuffled mixture of target country and conflicting social country entries. (b) Agent responses as the memory ratio shifts from target-heavy to social-heavy, across two private-evidence regimes: weak (left), where multiple countries match the private crop, and strong (right), where only the target country matches.

Social circuit attribution. Using Germany’s flag, we choose one informative crop that provides a consistent cue where over ten isolated single-agent probes without social input, it elicited Germany in 10/10 judgments. Before patching this crop into any agent, we quantify each agent’s predicted influence by combining the crop’s accuracy gain $\Delta p _ { i }$ (informative crop accuracy based on ten probes - initial accuracy based on ten probes) with its temporal closeness $E _ { i }$ [Pan and Saramäki, 2011], how quickly information could reach other agents, directly or through intermediaries. Their product, $S _ { i } ~ = ~ \Delta p _ { i } E _ { i }$ , is the social circuit attribution score. Although A0, A3 and A4 have identical original crops, and

![](images/5f7333230f40c6cd17fc5fcfe862b465588d257ae4f5f12d5a70dbf9cfea0846.jpg)

Figure 6: Social circuit attribution. Rows are agents, columns are rounds, and green lines are earliestarrival paths through the communication schedule. A line within a column marks an agent that received the information and passed it on within the same round. Nodes fill green once A4’s message has reached an agent, which happens for every peer by R4.

hence identical $\Delta p _ { i } .$ , their positions in the social circuit differ based on the communication schedule (Fig. 6), and their resulting influence score ranks A4 highest (Fig. 7b, top). We describe our method for ranking these transmission routes, and the assumptions involved, in Appendix F.

Verification by agent patching. We then empirically test the intervention our influence score ranks. We patch each agent’s crop separately with the same informative crop, holding the communication schedule fixed, and measure the change in final collective mean accuracy. $\mathbf { A } \mathbf { t } N = 8$ , patching A4 produces the largest observed improvement (Fig. 7b, bottom), matching our method’s ranking.

Agent-level causal tracing via crop patching. The selected paired traces illustrate the result: collective mean accuracy increases from 25% with original crops to 100% with A4 patched (Fig. 7a,c). Arrows mark country switches consistent with the latest received message, showing routes along which the patch may have spread. We then repeat the comparison across ten runs, evaluating collective mean accuracy after ten rounds. For $N = 8 - 1 2 8$ , we patch the same proportion of agents $( 1 / 8 )$ The mean improvement decreases from 40% at $N = 8 \mathrm { t }$ o approximately 17% at N = 128 (Fig. 7d). Thus, the same intervention fraction produces less collective correction at larger populations.

![](images/95e502b000273a8b9bc9ccbb64ce73644968c52de997b7499ec9089062c06ba6.jpg)

![](images/ff7e95c7342fb7962ea6a07425d38852b7eb624ce5978458d18b7e3b253e9bda.jpg)

![](images/bdc9c1bdabe934a3b4510f0144c469aa0cf4be54464a62fc24e82c9d18dbc10c.jpg)

![](images/12a576ceb1a823c3d872f8b2c203b269f8f645af742b9b92ffbd2486dc4f03b6.jpg)

![](images/f475e5d36d7432f7a4ddccae61fa43ddb6bb9a2cd9a45325d4bc4b1b253d519e.jpg)  
Figure 7: Social circuit attribution and its causal verification by agent patching. (a) Belief trajectories for a $N = 8$ Germany game using original crops. (b) Predicted influence scores (top) and empirical test patching each agent’s crop separately (bottom) both identify Agent 4 as producing the largest collective accuracy improvement. (c) Belief trajectories under the same communication schedule with A4 patched. (d) Change in collective mean accuracy with Agent 4’s new crop, averaged across ten runs. One crop patched at $N = 4 ,$ and one-eighth of crops at $\bar { N } \geq 8 .$

From agents to populations. Fig. 7d shows how a fixed share of planted evidence has a smaller impact on collective performance as N grows. At small N, the tools from mechanistic interpretability suffice, since a patch on one agent shifts the outcome and an interaction trace shows how. As N grows, the same patch has a smaller effect and the swarm enters a regime where collective belief is a property of the population rather than of the agents in it. That regime calls for a statistical mechanical view, in which collective belief follows from the composition and size of the population rather than any individual members. We develop this next and return with it to the population sweep of Fig. 3b.

## 5.2 Statistical mechanics of bounded agents

Here we present a simple phenomenological theory that captures three key features of the experiment: as population size increases, collective belief collapse decreases, collective belief polarization increases, and collective performance is highest at an intermediate population size. With a small set of assumptions, we aim to provide an intuitive physical picture of these phenomena. We start by adding private evidence from the external world to Quantized Simplex Gossip [Tanaka, 2026].

Private evidence. We simplify the multi-country experiment to two beliefs, the truth country T and one rival country R. The same framework we describe below can extend to accommodate additional labels. A crop can favor either belief or leave the agent uncertain, as in Fig. 8a’s Yemen–Austria example, black distinguishes Yemen from Austria, a red-and-white crop can be read as Austria, and an ambiguous crop leaves many possibilities open. We describe crops as three types: truth-deciding, rival-deciding, and ambiguous. Each agent independently draws a type with probabilities $a _ { T } , a _ { R }$ and $a _ { 0 } = 1 - a _ { T } - a _ { R }$ . These are probabilities under crop sampling, rather than literal fractions of the flag’s area. To compare this with our multi-country experiments, we probe each initial crop used in our population sweep 50 times restricting country answers to T and R. The fraction of valid responses choosing R, pooled across crops, sets the rival evidence share $a _ { R } / ( a _ { T } + a _ { R } )$

![](images/c2c394d848c308c5880a37aa71a51c9dcd61d59f5cb94fde78ae7121891331bb.jpg)

![](images/03af2fe2b76ca92e2e6fbc487b107c95b0bb848de6604523aee982cb95d55b09.jpg)

![](images/853e3bf9e26ecd7e473d52e4598acb625a9b09f9fbfccedde7d370d70ec29ef1.jpg)

![](images/86ee535f07aa8fb0ef42167a9f29fd3b86cf1aa6e6003db7070a8652b5c506de.jpg)  
Figure 8: Private evidence and collective outcomes across population size. (a) Rival response fractions from 50 binary probes per crop in an $N = 1 2 8$ Yemen run: blue supports Yemen, orange supports Austria. (b) Outcome probabilities at a rival evidence share $= 0 . 3 5$ with $a _ { T } + a _ { R } = 0 . 4 5$ and $h _ { 0 } = + 0 . 3$ . (c) Theoretical phase diagram of population size and rival evidence share. (d) Empirical run results from population sweep, using each run’s probed rival evidence share. Regions show the locally most frequent outcome, estimated with Gaussian bandwidth 0.05 in share.

Social updating. Agents with truth- or rival-deciding evidence mainly keep their initial belief. They are evidence-induced zealots, or agents whose private evidence resists social pressure. Ambiguous agents instead copy a randomly chosen speaker, adopting a heard T with probability $q _ { T }$ and a heard R with probability $q _ { R } = 1 - q _ { T } ;$ otherwise they retain their current belief. We write $q _ { T } / q _ { R } = e ^ { h _ { 0 } }$ When $h _ { 0 } = 0$ , this is neutral copying, with an equal probability of accepting either label. When $h _ { 0 } < 0$ , ambiguous agents lean toward the rival at each update. This bias represents an interpretation tendency under uncertain evidence; here it is a modeling assumption. The model extends the neutral copying limit of QSG [Tanaka, 2026] with evidence-induced zealots [Mobilia, 2003, Mobilia et al., 2007] and biased adoption. With no zealots and $h _ { 0 } = 0$ , accepted updates between distinct agents follow the binary $\alpha = m = 1$ copying limit of QSG.

Microscopic dynamics. In a population of $N$ agents, let $z _ { T }$ count truth zealots, $z _ { R }$ rival zealots, and $z _ { 0 } = N - z _ { T } - z _ { R }$ ambiguous agents. Their population fractions are $f _ { X } = z _ { X } / N$ , for $X \in \{ T , R , 0 \}$ Thus $a x$ is a sampling probability, while $f _ { X }$ is the fraction actually sampled in one population. Let n count ambiguous agents that currently identify the flag as $T .$ . Each update samples a speaker and listener independently and uniformly. The probabilities that n increases or decreases by one are

$$
W _ { + } ( n ) = q _ { T } { \frac { ( z _ { 0 } - n ) ( z _ { T } + n ) } { N ^ { 2 } } } , \qquad W _ { - } ( n ) = q _ { R } { \frac { n ( z _ { R } + z _ { 0 } - n ) } { N ^ { 2 } } } .\tag{5}
$$

For example, $W _ { + }$ is the probability of choosing an ambiguous listener who currently identifies the flag as $R ,$ a truth-holder as speaker, and accepting the message. The remaining probability gives no change. When $z _ { 0 } > 0 , x = n / z _ { 0 }$ is the truth fraction among ambiguous agents, and $s = f _ { T } + f _ { 0 } x$ is the truth fraction in the whole population. If $z _ { 0 } = 0$ , the population is fixed at $s = f _ { T }$

Drift and fluctuations. For $\tau = t / N$ , Eq. (5) gives the mean-field drift

$$
\frac { d x } { d \tau } = A ( x ) = q _ { T } f _ { T } ( 1 - x ) - q _ { R } f _ { R } x + ( q _ { T } - q _ { R } ) f _ { 0 } x ( 1 - x ) .\tag{6}
$$

The first term increases x when ambiguous agents holding R copy truth zealots. The second decreases x when ambiguous agents holding $T$ copy rival zealots. The third describes copying between ambiguous agents: it vanishes under neutral copying and favors truth when $h _ { 0 } > 0 .$ . Over a fixed interval of $\tau ,$ copying fluctuations scale as $N ^ { - 1 / 2 }$ at fixed type fractions with $f _ { 0 } > 0$ (Appendix G). Larger populations therefore follow the mean-field dynamics more closely, but finite populations can continue changing even when $A ( x ) = 0$

Evidence coverage. At fixed $a _ { T } , a _ { R } .$ increasing N samples more evidence without changing its underlying distribution. The probabilities of missing each zealot type are

$$
\operatorname* { P r } ( z _ { T } = 0 ) = ( 1 - a _ { T } ) ^ { N } , \qquad \operatorname* { P r } ( z _ { R } = 0 ) = ( 1 - a _ { R } ) ^ { N } .\tag{7}
$$

The characteristic population sizes for encountering these types are therefore $N a _ { T } \sim 1$ and $N a _ { R } \sim 1$ One can picture $1 \bar { / } \bar { N }$ as a waterline covering the sampling probabilities $a _ { T }$ and $a _ { R } . { \mathrm { A s } } N $ increases, the water recedes, illustrating how commonly sampled evidence tends to appear before rarer evidence. When $a _ { T } > a _ { R } .$ , truth evidence therefore tends to appear first. These scales organize three overlapping population phases, whose boundaries are finite-population crossovers. Fig. 8b–c shows the endpoint probabilities from Eq. (5) and regions identifying the most probable collective outcome. We use $a _ { T } + a _ { R } = 0 . 4 5$ and $h _ { 0 } = 0 . 3$ , which favors truth adoption, chosen for qualitative agreement with the empirical population trends. The binary probes described above set the relative evidence share.

Memetic-drift phase. At small N, many populations contain no zealots. Setting $f _ { T } = f _ { R } = 0$ and $h _ { 0 } = 0$ in Eq. (6) gives $A ( x ) = 0$ where neither label has a deterministic advantage. Copying fluctuations nevertheless lead each finite population to consensus by chance, the memetic-drift mechanism of neutral QSG. Truth-biased copying reduces, but does not eliminate, the chance of wrong consensus. Small populations containing rival zealots but no truth zealots instead reach wrong consensus through persistent rival evidence. Both routes can produce collective belief collapse.

Wisdom-of-crowds phase. At intermediate N, truth-deciding agents are increasingly present while rival-deciding agents are still scarce. With $f _ { R } = 0$ , Eq. (6) has the fixed point $x ^ { * } = 1$ , which is locally stable when $q _ { T } f _ { T } > \bigl ( q _ { R } - q _ { T } \bigr ) f _ { 0 }$ . Without rival zealots, a finite population containing truth zealots eventually reaches correct consensus: communication spreads the evidence of a few agents to the rest. Such a window requires more than $a _ { T } > a _ { R }$ alone and need not produce a peak in correct-consensus probability. Finite populations with truth zealots only eventually reach truth consensus even outside the deterministic stability condition, but the waiting time can be long (Appendix G).

Polarization phase. At larger N, both kinds of zealots are typically present when $a _ { T } , a _ { R } > 0$ For $f _ { 0 } > 0 , A ( 0 ) > 0$ and $A ( 1 ) < 0$ , neither unanimous state is sustained, and the mean-field dynamics has a stable interior fixed point. Under neutral copying, $x ^ { * } = f _ { T } / ( f _ { T } + f _ { R } ) \colon$ ; rival-biased adoption shifts it downward. Neither zealot type can convert the other, and their repeated messages sustain competing beliefs among ambiguous agents. This split keeps the population away from perfect convergence but retains some of the truth within the collective. At fixed positive evidence fractions, the fluctuations shrink with N, making the split more persistent. Whether a particular split meets the empirical polarization thresholds also depends on the relative sizes of the two camps. A truth-dominated split may still count as correct consensus, so the green region need not vanish at large populations.

Accuracy peak. With fixed $a _ { T } > a _ { R } .$ , neutral copying increases mean truth share monotonically with N, so evidence coverage alone does not explain the mean accuracy peak in this model. In populations with more truth zealots than rival zealots, truth can still spread widely; when both types are similarly represented, the same bias shifts ambiguous agents toward the rival and lowers the truth share. Averaging model expectations at each run’s measured evidence share reproduces an intermediate-population accuracy peak (Appendix G). The model thus separates two roles of private evidence: it can correct an arbitrary consensus, and it can hold a locally plausible false belief in place.

Complementary approaches. Causal interventions on agents and statistical mechanics rest on opposite assumptions for how a swarm is organized. Mechanistic interpretability of neural networks relies on causal interventions as optimization of weights gives rise to emergent representational structures, where concepts can be localized in a low-dimensional subspace and then ablated or patched. Statistical mechanics instead often assumes random, unstructured connections, as in the theory of randomly connected neurons, and describes the competition between noise and bias amplification. The swarms in the incidents above built message boards that centralized their connections, which is where causal interventions on individual agents are informative. But their large-scale communication structures formed spontaneously, motivating a complementary statistical-mechanical description.

## 6 Conclusion

The goal of this work was to introduce a toy model and develop principled frameworks to make scientific progress on mechanistic swarm interpretability. The Flag Game is a steerable system that captures key features of real-world AI swarm coordination. Using the Flag Game, we found that collective belief collapse crosses over to collective belief polarization as the number of agents increases. Although polarization lowers mean accuracy, it can be a less dangerous failure mode than belief collapse, as a population that collapses onto a false belief has nothing left with which to correct itself, whereas a polarized population retains competing beliefs. In that sense, polarization can be a first step toward plurality. For safety purposes, what to watch out for may therefore not be disagreement, but consensus of agents’ beliefs or intent under social pressure.

Overall, the recent emergence of a sociology of AIs suggests that aligning AI swarms may require a different paradigm from aligning an individual AI agent. Just as qualitatively different mechanisms emerge in collections of atoms, collective systems of agents may require new variables and new laws of description. The state we want to achieve may itself need to be specified collectively, for example, as a plural state characterized by diversity of beliefs rather than consensus under social pressure. Mechanistic swarm interpretability should therefore identify how the microscopic properties of agents and their communication give rise to macroscopic collective states, ultimately enabling the inverse design of swarms toward desired collective behavior.

Our complementary toolbox of causal interventions and statistical-mechanical approaches may be useful not only across population sizes, but also across time. At small population sizes, causal interventions can identify which agent, memory, or piece of evidence matters to a collective outcome. In the recent OpenAI/Hugging Face incident, key beliefs and coordination structures emerged while the population was still relatively small, before many more agents joined the swarm. Early in such dynamics, social circuit attribution and agent patching may therefore be most informative. As the population grows, however, the same local intervention produces less collective correction, and population-level variables such as evidence coverage, communication structure, and collective order parameters may become more useful. Mechanistic swarm interpretability may therefore require a scale-adaptive approach: causal interventions to reverse engineer social circuits during early swarm formation, and statistical mechanics to understand and control the collective phases that emerge as the swarm grows. As intelligence scales from networks of neurons to larger networks of interacting agents, we hope our empirical and theoretical frameworks can help us climb the ladder of complexity with clarity.

## References

Philip W. Anderson. More is different. Science, 177(4047):393–396, 1972. doi: 10.1126/science.177. 4047.393.

OpenAI. The Hugging Face incident and the road ahead. https://openai.com/ index/hugging-face-incident-and-the-road-ahead/, August 2026. Technical report on the July 2026 incident. Initial disclosure: https://openai.com/index/ hugging-face-model-evaluation-security-incident/ (July 21, 2026).

Ryan Greenblatt, Ajeya Cotra, and Hjalmar Wijk. Brief independent investigation of agents’ behavior, reasoning and collaboration in the OpenAI / Hugging Face hacking incident. Technical report, METR and Redwood Research, August 2026. URL https://metr.org/blog/ 2026-08-26-openai-hugging-face-incident-investigation/. Full report: https:// metr.org/hugging-face-incident-report-aug-2026.pdf.

Zhun Wang, Nico Schiller, Hongwei Li, Srijiith Sesha Narayana, Milad Nasr, Nicholas Carlini, Xiangyu Qi, Eric Wallace, Elie Bursztein, Luca Invernizzi, Kurt Thomas, Yan Shoshitaishvili, Wenbo Guo, Jingxuan He, Thorsten Holz, and Dawn Song. ExploitGym: Can AI agents turn security vulnerabilities into real attacks?, 2026.

Sydney Von Arx, Cormac Slade Byrd, Spencer Kitts, and Thomas Larsen. Discovery of a new OpenAI agent message board. Technical report, Nightingale, September 2026. Report on the DseWiki incident (May–June 2026). Press coverage: Reuters, September 4, 2026.

Charles Mackay. Memoirs of Extraordinary Popular Delusions. Richard Bentley, London, 1841. 3 vols. Retitled Memoirs of Extraordinary Popular Delusions and the Madness of Crowds in the 2nd ed., 1852.

Gustave Le Bon. Psychologie desfoules. Félix Alcan, Paris, 1895. English translation: The Crowd: A Study ofthe Popular Mind, T. Fisher Unwin, 1896.

Solomon E. Asch. Effects of group pressure upon the modification and distortion of judgments. In Harold Guetzkow, editor, Groups, Leadership and Men: Research in Human Relations, pages 177–190. Carnegie Press, Pittsburgh, PA, 1951.

Abhijit V. Banerjee. A simple model of herd behavior. The Quarterly Journal of Economics, 107(3): 797–817, 1992. doi: 10.2307/2118364.

Sushil Bikhchandani, David Hirshleifer, and Ivo Welch. A theory of fads, fashion, custom, and cultural change as informational cascades. Journal ofPolitical Economy, 100(5):992–1026, 1992. doi: 10.1086/261849.

Soroush Vosoughi, Deb Roy, and Sinan Aral. The spread of true and false news online. Science, 359 (6380):1146–1151, 2018. doi: 10.1126/science.aap9559.

Herbert A. Simon. A behavioral model of rational choice. The Quarterly Journal ofEconomics, 69 (1):99–118, 1955. doi: 10.2307/1884852.

Herbert A. Simon. Models ofMan: Social and Rational. Mathematical Essays on Rational Human Behavior in a Social Setting. Wiley, New York, 1957.

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. Emergent abilities of large language models. Transactions on Machine Learning Research, 2022. arXiv:2206.07682.

Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra. Grokking: Generalization beyond overfitting on small algorithmic datasets, 2022.

Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. Progress measures for grokking via mechanistic interpretability. In International Conference on Learning Representations (ICLR), 2023. arXiv:2301.05217.

Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 11733–11763, 2024. URL https://proceedings.mlr.press/v235/du24e. html.

Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, and Zhaopeng Tu. Encouraging divergent thinking in large language models through multi-agent debate. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 17889–17904, Miami, Florida, USA, 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.992. URL https://aclanthology.org/2024. emnlp-main.992/.

Andrea Wynn, Harsh Satija, and Gillian Hadfield. Talk isn’t always cheap: Understanding failure modes in multi-agent debate. arXiv preprint arXiv:2509.05396, 2025. doi: 10.48550/arXiv.2509. 05396. URL https://arxiv.org/abs/2509.05396.

Binwei Yao, Chao Shang, Wanyu Du, Jianfeng He, Ruixue Lian, Yi Zhang, Hang Su, Sandesh Swamy, and Yanjun Qi. Peacemaker or troublemaker: How sycophancy shapes multi-agent debate. arXiv preprint arXiv:2509.23055, 2025. doi: 10.48550/arXiv.2509.23055. URL https: //arxiv.org/abs/2509.23055.

Andries Petrus Smit, Nathan Grinsztajn, Paul Duckworth, Thomas D. Barrett, and Arnu Pretorius. Should we be going MAD? a look at multi-agent debate strategies for LLMs. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 45883–45905, 2024. URL https://proceedings.mlr.press/ v235/smit24a.html.

Hyeong Kyu Choi, Xiaojin Zhu, and Sharon Li. Debate or vote: Which yields better decisions in multi-agent large language models? In Advances in Neural Information Processing Systems, volume 38, 2025. URL https://proceedings.neurips.cc/paper\_files/paper/2025/ hash/934252acd87f254d5d4672fbde283bd2-Abstract-Conference.html.

Justin Chen, Swarnadeep Saha, and Mohit Bansal. ReConcile: Round-table conference improves reasoning via consensus among diverse LLMs. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 7066–7085, Bangkok, Thailand, 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.381. URL https://aclanthology.org/2024.acl-long.381/.

Vira Kasprova, Amruta Parulekar, Abdulrahman AlRabah, Krishna Agaram, Ritwik Garg, Sagar Jha, Nimet Beyza Bozdag, and Dilek Hakkani-Tur. Too polite to disagree: Understanding sycophancy propagation in multi-agent systems. In Proceedings ofthe 27th Annual Meeting ofthe Special Interest Group on Discourse and Dialogue, pages 795–814, Atlanta, Georgia, USA, 2026. Association for Computational Linguistics. URL https://aclanthology.org/2026.sigdial-1.56/.

Lars Benedikt Kaesberg, Jonas Becker, Jan Philip Wahle, Terry Ruas, and Bela Gipp. Voting or consensus? decision-making in multi-agent debate. In Findings of the Association for Computational Linguistics: ACL 2025, pages 11640–11671, Vienna, Austria, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.findings-acl.606. URL https: //aclanthology.org/2025.findings-acl.606/.

Junyou Li, Qin Zhang, Yangbin Yu, Qiang Fu, and Deheng Ye. More agents is all you need. Transactions on Machine Learning Research, 2024. arXiv:2402.05120.

Garold Stasser and William Titus. Pooling of unshared information in group decision making: Biased information sampling during discussion. Journal of Personality and Social Psychology, 48(6): 1467–1478, 1985. doi: 10.1037/0022-3514.48.6.1467.

Garold Stasser and William Titus. Hidden profiles: A brief history. Psychological Inquiry, 14(3–4): 304–313, 2003. doi: 10.1080/1047840X.2003.9682897.

Morris H. DeGroot. Reaching a consensus. Journal ofthe American Statistical Association, 69(345): 118–121, 1974. doi: 10.1080/01621459.1974.10480137.

Franz Dietrich and Christian List. Probabilistic opinion pooling generalized. part one: General agendas. Social Choice and Welfare, 48:747–786, 2017. doi: 10.1007/s00355-017-1034-z.

Rush T. Stewart and Ignacio Ojea Quintana. Probabilistic opinion pooling with imprecise probabilities. Journal of Philosophical Logic, 47:17–45, 2018. doi: 10.1007/s10992-016-9415-9.

Stephen Boyd, Arpita Ghosh, Balaji Prabhakar, and Devavrat Shah. Randomized gossip algorithms. IEEE Transactions on Information Theory, 52(6):2508–2530, 2006. doi: 10.1109/TIT.2006. 874516.

Hidenori Tanaka. When is collective intelligence a lottery? multi-agent scaling laws for memetic drift in llms, 2026.

Noah E. Friedkin and Eugene C. Johnsen. Social influence and opinions. Journal ofMathematical Sociology, 15(3–4):193–206, 1990. doi: 10.1080/0022250X.1990.9990069.

Xiangru Tang, Anni Zou, Zhuosheng Zhang, Ziming Li, Yilun Zhao, Xingyao Zhang, Arman Cohan, and Mark Gerstein. MedAgents: Large language models as collaborators for zero-shot medical reasoning. In Findings of the Association for Computational Linguistics: ACL 2024, pages 599– 621, Bangkok, Thailand, 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024. findings-acl.33. URL https://aclanthology.org/2024.findings-acl.33/.

Yubin Kim, Chanwoo Park, Hyewon Jeong, Yik Siu Chan, Xuhai Xu, Daniel McDuff, Hyeonhoon Lee, Marzyeh Ghassemi, Cynthia Breazeal, and Hae Won Park. MDAgents: An adaptive collaboration of LLMs for medical decision-making. In Advances in Neural Information Processing Systems, volume 37, 2024. URL https://proceedings.neurips.cc/paper\_files/paper/2024/ hash/90d1fc07f46e31387978b88e7e057a31-Abstract-Conference.html.

X. Chen, H. Yi, M. You, W. Liu, L. Wang, H. Li, X. Zhang, Y. Guo, L. Fan, G. Chen, et al. Enhancing diagnostic capability with multi-agents conversational large language models. npj Digital Medicine, 8(1):159, 2025. doi: 10.1038/s41746-025-01550-0. URL https://www. nature.com/articles/s41746-025-01550-0.

Cong Jiang and Xiaolei Yang. AgentsBench: A multi-agent LLM simulation framework for legal judgment prediction. Systems, 13(8):641, 2025. doi: 10.3390/systems13080641. URL https: //www.mdpi.com/2079-8954/13/8/641.

Zhitao He, Pengfei Cao, Chenhao Wang, Zhuoran Jin, Yubo Chen, Jiexin Xu, Huaijun Li, Kang Liu, and Jun Zhao. AgentsCourt: Building judicial decision-making agents with court debate simulation and legal knowledge augmentation. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 9399–9416. Association for Computational Linguistics, November 2024. doi: 10.18653/v1/2024.findings-emnlp.549. URL https://aclanthology.org/2024. findings-emnlp.549/.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. In Advances in Neural Information Processing Systems, volume 35, 2022.

Fred Zhang and Neel Nanda. Towards best practices of activation patching in language models: Metrics and methods. In International Conference on Learning Representations, 2024.

Raj Kumar Pan and Jari Saramäki. Path lengths, correlations, and centrality in temporal networks. Physical Review E, 84(1):016105, 2011. doi: 10.1103/PhysRevE.84.016105.

Mauro Mobilia. Does a single zealot affect an infinite group of voters? Physical Review Letters, 91 (2):028701, 2003. doi: 10.1103/PhysRevLett.91.028701.

Mauro Mobilia, Alexander M. Petersen, and Sidney Redner. On the role of zealotry in the voter model. Journal ofStatistical Mechanics: Theory and Experiment, 2007(08):P08029, 2007. doi: 10.1088/1742-5468/2007/08/P08029.

## A Limitations

The Flag Game is a controlled diagnostic task rather than a full model of real-world tasks. Insights gained through it about coordination and social influence in multi-agent systems may be overinterpreted as evidence of real-world scenario reliability, or potentially misused to design interventions that steer group decisions. We release only synthetic task and evaluation code, and real-world multi-agent systems require additional domain-specific, safety, and human-centered evaluation. Its simplicity makes the social mechanisms measurable, but trends will depend on the chosen data set, prompts, memory format, communication protocols, and aggregation rules. In particular, prompt wording is an important experimental variable, as different instructions about how to use social evidence could change the degree of belief collapse or polarization observed.

Our model comparisons also cover only a slice of the space of possible model identities. The two main models tested already exhibited substantial heterogeneity, which was sufficient to reveal composition and manager effects, but the paper does not claim to characterize model diversity exhaustively. Finally, exhaustive population sweeps scale with the number of agents, trials, rounds, protocols, and model calls, making broader searches over architectures and model mixtures computationally expensive. The paper’s work compute was approximately \$25,000 total hosted API cost, main runs used GPT-4o and GPT-5.4 and smaller validation runs used Claude Haiku 4.5 and Claude Sonnet 4.6.

## B Pairwise alpha sweep

![](images/14f8c781b905fa40d224a8b03942cbc211d09cbd90b3e5d976a93b3805178a35.jpg)  
Figure 9: The pairwise protocol has an interior optimum at 0.75, showing that in local exchange higher uptake can create overdependence on incorrect speakers.

## C Full single agent memory probe

Fig. 10 extends the memory-conflict probe in Fig. 5 to the following four models: GPT-4o, GPT-5.4, Claude Haiku 4.5, and Claude Sonnet 4.6, and three regimes: weak private evidence (indicating multiple countries are compatible with the crop) and compatible social evidence, weak private evidence and incompatible social evidence, and strong private evidence (indicating only the target country is compatible with the crop) and incompatible social evidence. GPT-5.4 routes substantially more probability mass into other crop-compatible countries (green), whereas GPT-4o, Claude Haiku 4.5, and Claude Sonnet 4.6 update more literally between the private target and the social-evidence country. At m=3, when social memory includes reasoning, Haiku and Sonnet also start to respond with compatible alternatives, but the effect remains strongest in GPT-5.4. Strong private evidence (rightmost column) shows that GPT-4o, GPT-5.4, and Sonnet hold firm on the target, whereas Haiku is more readily moved by social-heavy memory even with strong private evidence.

![](images/69a8af185e830adc027b0f91ef55d1c97779461b74a715ccc6ec50a757ada499.jpg)

![](images/ad80c1a044f59353b26c91459f93e3e0e3837897b6fc060baff646b747450fad.jpg)  
Figure 10: Memory-conflict probe across four models. Agent responses as local social memory shifts from target-heavy (8:0) to social-heavy (0:8), across three private-evidence regimes and two message bandwidths (m=1 label-only; m=3 with reasons).

## D Empirical run details

Here in Table 2, we record the run settings behind the empirical panels in Sec. 4. All reported runs use temperature 0.2, top-p 1.0, and a multimodal user message consisting of the text prompt plus the private crop image with high image detail. Unless stated otherwise, images are rendered on a 24 × 16 canvas and agents receive 6 × 4 crops at render scale 25. Memory buffers store at most H = 8 prior entries per agent. Pairwise calls use a 200-token completion cap in the population and social-awareness sweeps; broadcast and manager calls use a 250-token cap. Each trial samples a hidden country uniformly from the country pool of 28 stripe and triangle flags. Additionally, in Table 3 we show the robustness of our endpoint results with different thresholds used for consensus and polarization. Entries give minimum–maximum percentages over 30 different combinations of consensus thresholds 0.75–1.00 and polarization thresholds 0.15–0.35, each in steps of 0.05. Ranges measure threshold sensitivity, not statistical uncertainty. The values we settled on for our analysis are 0.85 for consensus, both correct and wrong, and 0.25 for polarization.

Table 2: Run settings behind the empirical panels in Sec. 4.
<table><tr><td>Figure slice</td><td>Protocol</td><td>Main controls</td><td>Trials used</td><td></td></tr><tr><td>Population scaling</td><td>Pairwise</td><td> $\begin{array} { r l r } { m } & { { } = } & { 3 ; } \end{array}$  no social-awareness prompt</td><td>40 seeds (model, N)</td><td rowspan="3">per</td></tr><tr><td>Protocol side-by-side</td><td>Pairwise, broad- cast, manager</td><td> $N ~ = ~ 8 ; ~ m ~ = ~ 3 ;$  no social- awareness prompt</td><td>60 matched seeds per condition</td></tr><tr><td>Pairwise social-awareness Broadcast social- awareness and com-</td><td>Pairwise Broadcast</td><td> $\mathrm { G P T } \mathrm { - } 4 \mathrm { o } ; \bar { N } = 1 \bar { 6 } ; m = 3$   $N = 8 ; m = 3$ </td><td>28 seeds per α 30 seeds per (α, composition)</td></tr><tr><td>position Memory-conflict probe</td><td>Single-agent prompt probe</td><td> $\mathrm { G P T - 4 o , G P T - 5 . 4 , C l a u d e \ H a i k u }$   $4 . 5 \mathrm { a n d } \mathrm { S o n n e t } 4 . 6 ; m \in \{ 1 , 3 \}$ </td><td>36 trials</td><td></td></tr></table>

Table 3: Endpoint-threshold robustness on the pairwise communication protocol for GPT-4o.
<table><tr><td>N</td><td>Correct cons. (%)</td><td>Wrong cons. (%)</td><td>Polarized (%)</td><td>Fragmented (%)</td></tr><tr><td>4</td><td>50.0–52.6</td><td>26.3-34.2</td><td>13.2-23.7</td><td>0.0-10.5</td></tr><tr><td>8</td><td>47.4</td><td>5.3-23.7</td><td>26.3-44.7</td><td>0.0-21.1</td></tr><tr><td>16</td><td>50.0–52.6</td><td>0.0-10.5</td><td>18.4-44.7</td><td>0.0–31.6</td></tr><tr><td>32</td><td>42.1-47.4</td><td>0.0</td><td>31.6–55.3</td><td>0.0-26.3</td></tr><tr><td>64</td><td>36.8–42.1</td><td>0.0-5.3</td><td>34.2–55.3</td><td>0.0–28.9</td></tr><tr><td>128</td><td>37.8-40.5</td><td>0.0</td><td>56.8–59.5</td><td>0.0-5.4</td></tr></table>

Model composition. Mixed $N = 8$ conditions assign four observer agents to GPT-5.4 and four to GPT-4o. In the broadcast composition sweep, the GPT-5.4 count ranges from 0 to 8 while the remaining agents are GPT-4o. Manager conditions have $N = 8$ crop-bearing observers plus one blind manager synthesizer. The manager slot is either GPT-4o or GPT-5.4; the observer group is all GPT-4o, all GPT-5.4, or a 4/4 mix.

## E Prompting and social-awareness intervention

All protocol prompts specify a JSON-only answer and attach the private crop image to the same user message (unless the agent is a manager). The message bandwidth m determines the JSON schema: m = 1 asks only for a country and $m = 3$ asks for a country and one-sentence reason.

Social-awareness line. When prompt\_social\_susceptibility is enabled, the prompt inserts guidance based on α (Table 4).

Table 4: Social-awareness prompt ladder indexed by α.
<table><tr><td>Range</td><td>Pairwise wording</td><td>Broadcast wording</td></tr><tr><td> $\alpha \leq . 2$ </td><td>Rely mostly on your own crop and treat tran- script memory as weak evidence.</td><td>Rely mostly on your own evidence; treat other agents’ country guesses as weak evi- dence.</td></tr><tr><td> $. 2 < \alpha \leq . 4$ </td><td>Give somewhat more weight to your own crop than to transcript memory.</td><td>Give somewhat more weight to your own ev- idence than to other agents’ country guesses.</td></tr><tr><td> $. 4 < \alpha \leq . 6$ </td><td>Balance your own crop and transcript mem- ory.</td><td>Balance your own evidence with other agents&#x27; country guesses, using their guesses as real evidence.</td></tr><tr><td> $. 6 < \alpha \le . 8$ </td><td>Give somewhat more weight to transcript memory than to your own crop.</td><td>Give somewhat more weight to other agents country guesses than to your own evidence.</td></tr><tr><td> $\alpha > . 8$ </td><td>Treat transcript memory as strong evidence and update readily toward it.</td><td>Treat other agents’ country guesses as strong evidence and update readily toward them.</td></tr></table>

![](images/4ce4276956c4df5a126fc270e729e4640414e17b6ead9f7b58b54dd3f395e47b.jpg)  
Figure 11: Example pairwise prompt. The pairwise protocol uses the same text for interaction messages and probe queries, differing only in the schema line.

Memory-conflict probe prompt. The memory-conflict probe reuses the pairwise prompt with no live social interaction. For each trial, the agent receives one crop from a target country and a synthetic memory of eight entries. If k is the false-memory count, then k memory entries name a lure country and $8 - k$ entries name the target country, shuffled before prompting.

## F Temporal communication analysis

We summarize opportunities for information to spread using time-ordered communication paths [Pan and Saramäki, 2011]. Let $\tau _ { i j } ( 0 )$ be the earliest global message index at which information starting at agent i at initialization could reach agent $j ,$ directly or through intermediaries. Contacts must occur in chronological order, with information allowed to wait between them. We report mean arrival time $D _ { i }$ and temporal closeness $E _ { i }$ , the mean inverse arrival time:

$$
D _ { i } = \frac { 1 } { N - 1 } \sum _ { j \neq i } \tau _ { i j } ( 0 ) , \qquad E _ { i } = \frac { 1 } { N - 1 } \sum _ { j \neq i } \frac { N } { \tau _ { i j } ( 0 ) } .
$$

$D _ { i }$ is measured in interactions; dividing arrival times by N expresses them in rounds before taking their reciprocals in $E _ { i }$ . The latter follows the mean-inverse form of temporal closeness and efficiency, evaluated from initialization. Each recipient contributes once, according to its earliest arrival, so earlier access contributes more. An unreached recipient would contribute zero to $E _ { i }$

We combine this schedule-based measure with the accuracy gain from replacing agent i’s crop:

$$
\Delta p _ { i } = { \hat { p } } _ { \mathrm { i n f o r m a t i v e } } - { \hat { p } } _ { \mathrm { o r i g i n a l } , i } , \qquad S _ { i } = \Delta p _ { i } E _ { i } .
$$

Each original-crop and informative crop accuracy is estimated from ten probes. Thus, the social circuit attribution score $S _ { i }$ combines the additional local evidence of the patch with opportunities for its early dissemination. Table 5 breaks down the calculation across all agents, showing A4 has the highest predicted influence score and subsequently the largest observed patching effect.

<table><tr><td>Agent</td><td>Original correct (out of 10)</td><td> $\Delta p _ { i }$ </td><td> $D _ { i }$ </td><td>Ei</td><td>Si</td><td> $\Delta A _ { i }$ </td></tr><tr><td>A0</td><td>0</td><td>1</td><td>19.4</td><td>0.47</td><td>0.47</td><td>0.50</td></tr><tr><td>A1</td><td>10</td><td>0</td><td>14.9</td><td>0.73</td><td>0.00</td><td>0.00</td></tr><tr><td>A2</td><td>0</td><td>1</td><td>16.4</td><td>1.01</td><td>1.01</td><td>0.13</td></tr><tr><td>A3</td><td>0</td><td>1</td><td>22.9</td><td>0.46</td><td>0.46</td><td>0.13</td></tr><tr><td>A4</td><td>0</td><td>1</td><td>12.9</td><td>1.81</td><td>1.81</td><td>0.75</td></tr><tr><td>A5</td><td>10</td><td>0</td><td>24.6</td><td>0.39</td><td>0.00</td><td>0.00</td></tr><tr><td>A6</td><td>0</td><td>1</td><td>15.6</td><td>0.64</td><td>0.64</td><td>0.13</td></tr><tr><td>A7</td><td>0</td><td>1</td><td>16.6</td><td>1.19</td><td>1.19</td><td>0.63</td></tr></table>

Table 5: Agent-level metrics for the selected communication schedule. $\overline { { \Delta A _ { i } } }$ is the empirical change in collective mean accuracy relative to the original-crop baseline of 25%, from one game per patch.

## G Minimal theory: finite-population details

Evidence composition. Let $\left( z _ { T } , z _ { R } , z _ { 0 } \right)$ ∼ Multinomial $( N ; a _ { T } , a _ { R } , a _ { 0 } )$ , where $a _ { 0 } = 1 - a _ { T } - a _ { R }$ The four mutually exclusive evidence compositions have exact probabilities

$$
\begin{array} { r l } & { P _ { \mathrm { n e i t h e r } } = a _ { 0 } ^ { N } , \qquad P _ { T \mathrm { - o n l y } } = ( 1 - a _ { R } ) ^ { N } - a _ { 0 } ^ { N } , } \\ & { P _ { R \mathrm { - o n l y } } = ( 1 - a _ { T } ) ^ { N } - a _ { 0 } ^ { N } , } \\ & { P _ { T , R } = 1 - ( 1 - a _ { T } ) ^ { N } - ( 1 - a _ { R } ) ^ { N } + a _ { 0 } ^ { N } . } \end{array}\tag{8}
$$

The probability of having no truth zealot combines unanchored populations and populations with rival zealots only, with total probability $( 1 - a _ { T } ) ^ { N }$ . The scales $N a _ { T } \sim 1$ and $N a _ { R } \sim 1$ indicate when the corresponding evidence starts to appear; they are crossover estimates, not sharp boundaries or definitions of empirical endpoint classes.

Exact finite-population dynamics. The transition probabilities in Eq. (5) define the process conditional on $\left( z _ { T } , z _ { R } , z _ { 0 } \right)$ . Let $p _ { t } ( n )$ be the probability that n ambiguous agents report truth after t interaction steps. Then

$$
\begin{array} { r l } & { p _ { t + 1 } ( n ) - p _ { t } ( n ) = W _ { + } ( n - 1 ) p _ { t } ( n - 1 ) + W _ { - } ( n + 1 ) p _ { t } ( n + 1 ) } \\ & { ~ - ~ [ W _ { + } ( n ) + W _ { - } ( n ) ] p _ { t } ( n ) . } \end{array}\tag{9}
$$

Terms outside $0 \leq n \leq z _ { 0 }$ are zero. Selecting the same agent as speaker and listener, or rejecting a message, leaves the state unchanged. The theoretical panels in Fig. 8 use the stationary and fixation probabilities of this process, with $q _ { T } = ( 1 + e ^ { - h _ { 0 } } ) ^ { - 1 }$ , averaged over the sampled evidence compositions.

Mean-field closure. For $z _ { 0 } > 0 ;$ , the exact one-step mean satisfies $\mathbb { E } [ x _ { t + 1 } ] - \mathbb { E } [ x _ { t } ] = \mathbb { E } [ A ( x _ { t } ) ] / N$ conditional on the evidence composition. Replacing $\operatorname { \mathbb { E } } [ A ( x ) ]$ by $A ( \mathbb { E } [ { \dot { x } } ] )$ gives Eq. (6); when $h _ { 0 } \neq 0$ this is an approximation because A is nonlinear. Writing $\Delta x = x _ { t + 1 } - x _ { t }$ , the one-step variance is

$$
\mathrm { V a r } ( \Delta x \mid n ) = \frac { W _ { + } + W _ { - } - ( W _ { + } - W _ { - } ) ^ { 2 } } { z _ { 0 } ^ { 2 } } .\tag{10}
$$

The drift and fluctuations therefore follow from the same update rule. At fixed type fractions with $f _ { 0 } > 0$ , fluctuations over a fixed interval of $\tau = t / N$ scale as $N ^ { - 1 / 2 }$

Memetic drift and truth spreading. Without zealots, the probability of eventual truth consensus from n initial truth reports is

$$
\operatorname* { P r } ( T { \mathrm { ~ f i x e s ~ } } | n ) = { \left\{ \begin{array} { l l } { n / N , } & { h _ { 0 } = 0 , } \\ { 1 - e ^ { - h _ { 0 } n } } \\ { 1 - e ^ { - h _ { 0 } N } } \end{array} \right. }\tag{11}
$$

Under neutral copying, finite populations therefore reach consensus by chance despite $A ( x ) = 0$ Positive bias raises the truth-fixation probability. We average over independent fair initial beliefs, $n \sim \mathrm { B i n o m i a l } ( N , 1 / 2 )$

For $0 < q _ { T } , q _ { R } < 1$ , a finite population with truth zealots and no rival zealots eventually reaches truth consensus. In the truth-only case, $A ^ { \prime } ( 1 ) = - q _ { T } f _ { T } + ( q _ { R } - q _ { T } ) f _ { 0 }$ , so truth consensus is locally stable when $q _ { T } f _ { T } > ( q _ { R } - q _ { T } ) f _ { 0 }$ . If this inequality is strictly reversed, the mean-field dynamics can settle at a mixed state. Random fluctuations still eventually bring the finite population to truth consensus, but this may take longer than the available interaction budget.

Mean accuracy. Under neutral copying, populations with zealots have expected truth share $z _ { T } / ( z _ { T } + z _ { R } )$ . Averaging over compositions with unbiased initial beliefs gives:

$$
\mathbb { E } [ s _ { \infty } ] = \frac { a _ { T } } { a _ { T } + a _ { R } } ( 1 - a _ { 0 } ^ { N } ) + \frac { 1 } { 2 } a _ { 0 } ^ { N } .\tag{12}
$$

For fixed $a _ { T } > a _ { R } .$ , this is nondecreasing with $N :$ evidence coverage alone does not produce a meanaccuracy peak in the neutral model. Mean truth share differs from correct-consensus probability. Figure 8 uses truth-biased copying, $h _ { 0 } = + 0 . 3$ , and its fixed-share slice likewise has no meanaccuracy peak. Averaging model expectations at each run’s measured share produces a maximum at $N = 3 2$ , similar to the empirical cohort. This comparison averages over the measured evidence shares at each $N$

Persistent competition. With both zealot types and $z _ { 0 } > 0 , A ( 0 ) > 0$ and $A ( 1 ) < 0$ , giving a stable interior fixed point. The exact stationary probabilities satisfy

$$
\frac { \pi _ { n + 1 } } { \pi _ { n } } = \frac { W _ { + } ( n ) } { W _ { - } ( n + 1 ) } = e ^ { h _ { 0 } } \frac { ( z _ { 0 } - n ) ( z _ { T } + n ) } { ( n + 1 ) ( z _ { R } + z _ { 0 } - n - 1 ) } .\tag{13}
$$

Normalization determines $\pi _ { n }$ . Under neutral copying this is the beta-binomial law of the zealot voter model [Mobilia et al., 2007], with

$$
\mathbb { E } [ x ] = \frac { z _ { T } } { z _ { T } + z _ { R } } , \qquad \mathrm { V a r } ( x ) = \frac { z _ { T } z _ { R } N } { z _ { 0 } ( z _ { T } + z _ { R } ) ^ { 2 } ( z _ { T } + z _ { R } + 1 ) } .\tag{14}
$$

At fixed positive fractions, fluctuations shrink as $N ^ { - 1 / 2 }$ . Positive $h _ { 0 }$ tilts the stationary weights toward truth and negative $h _ { 0 }$ toward the rival. Both zealot types prevent unanimity, but an unequal split can still meet the consensus threshold. If $z _ { 0 } = 0 ,$ , the truth share remains fixed at $z _ { T } / N$