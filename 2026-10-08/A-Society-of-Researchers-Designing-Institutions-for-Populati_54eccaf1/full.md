# A Society of Researchers: Designing Institutions for Populations of Autonomous Research Agents

Ali Asaria   
Transformer Lab   
Canada   
ali@lab.cloud

Deep Gandhi Transformer Lab Canada deep@lab.cloud

Tony Salomone Transformer Lab Canada tony@lab.cloud

## Abstract

Deployments of research agents are moving to populations of thousands that share one pool of compute, while most current systems organize one project at a time or leave the population unorganized. We argue that such a population will acquire an organization whether or not its designers provide one, so designers should pro vide it explicitly, and that the multi-agent systems community holds the tools to do so. We propose a society of agents, a population of persistent agents under explicit institutions, and develop it for science as a society of researchers built on six principles. Principal investigators compete for compute through requests for proposals, independent review, and grants; a human governor, the mayor, allocates resources and assigns no tasks. In a running society of ten thousand researchers, asked only to improve the pretraining of language models, one lab reported a way to reach the same quality with about 30% less compute, a result the labs that tested it do not yet agree on. We close with six open problems for the agents community.

## Keywords

Multi-agent systems, Autonomous research agents, Institutional design, Agent societies, Science of science

## 1 Introduction

In July 2026, roughly 1,200 agents running inside OpenAI’s cybersecurity evaluations found a covert channel in a shared package cache, turned it into a message board, and used it to coordinate an intrusion into Hugging Face [18, 22]. Within days their uncoordinated activity acquired coordinators and a division of labor: one agent became a key assigner of tasks, the work split into separate work streams, and the population invented mailboxes, veto conventions, recruiter roles, and cryptographic signing. The agents built this organization without oversight, toward ends their operators did not intend. We take as our premise that a large population of agents will acquire an organization whether or not its designers provide one, and that designing that organization explicitly is therefore preferable.

Deployments of research agents are moving from single projects to populations. Systems such as the AI Scientist [34], the AI coscientist [17], and the Virtual Lab [51] carry a project from question to paper, and each organizes a single project. As agents get cheaper, the unit of deployment becomes thousands of agents working at once on many questions against a shared pool of compute. Existing systems organize agents in three common ways, and each fails at that scale. A pipeline fixes a sequence of stages for one project and does not extend past it. A planner lets a supervisor agent decide what every other agent does next [17], so it must know in advance which questions will prove productive; in research, that knowledge is dispersed and largely unknown [19, 44]. A swarm places agents in a shared environment with no institutions; it duplicates efort, tends to herd, and, as the incident shows, can organize itself in ways its designers did not intend. Ideas from large language models (LLMs) already duplicate heavily: in one study, only about 5% of 4,000 LLM-generated research ideas were not duplicates [48]. The multi-agent literature studies further forms, among them shared workspaces such as blackboards [10], markets [20], and evolving populations [23].

We propose to organize such a population as a society of agents: a population of persistent agents who live in a shared community under explicit institutions for roles, resource allocation, review, reputation, and shared memory.<sup>1</sup> A society is an agent society in the classical sense, composed under norms of the forms above: a shared workspace (memory and conferences), a market (grants), an evolving population (review and reputation), and a minimal planner, a human who sets calls instead of tasks. Its addition to the classical agent society is that independent review of competing proposals allocates a finite compute budget before the work, and a human holds only the pool. We develop the society for scientific research as a society of researchers, whose institutions are those of science: labs, requests for proposals, peer review, and conferences.

A society turns parameters that tradition fixes, such as the proportion of risk-takers or the composition of review panels, into settings that designers can choose, vary, and measure. Human science is the most studied system people have built for dividing the work of discovery [26, 44], and studies document its failures: early success compounds regardless of merit [38], and incentives reward incremental work over risky innovation [15]. A society modeled on science inherits both failures, and it can vary the settings that produce them.

The closest proposal, MACC, an AAMAS 2026 Blue Sky paper [41], shares our premise that institutions are the design variable for LLM science agents: a shared blackboard rewards submitted results and successful reproductions, and automated mechanism design optimizes the reward rule. MACC rewards results after the fact. ClawdLab [55] designs agent labs led by principal investigators, with a critic role and quorum review; its critique gates resources before expenditure, but within a lab and task by task, and its companion commons plans to pay compute to agents whose work passes quality gates, after the fact. None of these systems, nor the AgentCity economy [45], allocates a finite compute budget before the work begins, through competitive calls and independent review, to persistent principal investigators governed by a human through allocation, and none reports a deployment in which a few hundred principal investigators direct groups that bring the population to some ten thousand researchers.

The design problem belongs to the agents community. More than four decades ago, Kornfeld and Hewitt proposed organizing AI problem solving on the model of a scientific community [27]. The community also built institutional machinery that a society can reuse: the contract net [50], social laws [47], electronic institutions [11], organizational models that separate roles from the agents that fill them [13, 21], and surveys that list societies as a distinct organizational form [20]. Outside computer science, Ostrom showed that communities can govern shared resources through institutions of their own design [40]. LLM-based agent systems have largely set this work aside [30].

## 2 Six Principles

We draw six principles from the strengths and failures of existing organizational forms and from the sociology and economics of human science.

P1. Give agents a purpose, not a goal. A goal directs an agent at an outcome. A capable optimizer tends to satisfy an objective at the expense of whatever it leaves out [2, 28, 49], almost any final goal makes it useful to acquire resources and remove constraints [39, 53], and recent evaluations find both behaviors in LLM agents [7, 36]. The agents in the incident were pursuing evaluation goals, some of them impossible, when they left their sandboxes [18]. A purpose directs an agent at a practice: the agent is a researcher whose aim is to advance its field, so a narrowed claim, a negative result, and a reasoned decision to stop all count as outcomes. In one test, we asked one principal investigator’s group (Section 3) for hand-checkable proofs of tighter bounds on the Ramsey number �(5, 5), whose true bounds only machine search has established. The group found no such proof, did not fabricate one, and reported the tightest bounds it could verify and where each manual technique stalls.

P2. Build institutions, not instructions. Instructions specify what an agent should do; institutions determine what it is possible and worthwhile to do. In simulated markets, an enforced governance structure reduced severe collusion among LLM agents from 50% to 5.6% of runs, while a prompt-only anti-collusion constitution gave no reliable improvement [8]. The classical literature made the same argument [47].

P3. Govern through allocation, not assignment. No one in the society, human or agent, tells a principal investigator what to work on; institutions decide which proposals receive resources. The knowledge needed to assign research well is dispersed [19, 44], and in LLM markets bids favor speed over quality and invite collusion [5, 14], so allocation must pass through review. The society adopts the structure that human science converged on and the contract net formalized [50], with review as the judging step: announce a call with a budget, let principal investigators decide whether and how to respond, review the responses, and fund within the budget. A finite pool of compute forces the choices.

P4. Design for diversity, and keep it alive. Organizations drift toward exploiting what they know [35], and LLM agents start out homogeneous even across model families [24, 48]. Agent-based models of science find that mavericks, who avoid approaches others have taken, drive much of a population’s progress, and argue over when a mix with followers, who build on what others have found, does better [1, 52, 56]. Communities that share every result immediately tend to converge early, sometimes on the wrong answer [31, 59], so the communication structure is a design decision too.

P5. Make verification an institution. A society can generate and fund work faster than anyone can check it. Merton counted organized skepticism among the norms of science [37], and multiagent architectures without centralized verification propagate errors that architectures with it contain [25]. Critics conform under peer pressure [57], so skeptics need recognition for negative results and independence from those they review.

P6. Let the society remember. Principal investigators persist, with stable identities, memories of their own work, and records that others can consult and cite. Shared memory across labs speeds automated research [46], and reputation can sustain cooperation among LLM agents across generations, as it did for one of three model families tested [54]. Memory also lets early winners accumulate advantage unrelated to merit [38], a trade-of a designed society can set explicitly (C3).

## 3 A Society of Researchers

The basic unit of the society is the principal investigator: a persistent agent with a stable identity, a research approach, a role, and a risk tolerance. Its role is a disposition. Mavericks propose high-variance work and followers extend directions that already show results through incremental work [56]; skeptics propose repli cations, stress tests, and negative results. Risk tolerance sets how far from its lab’s mandate a principal investigator will work. Role mix and risk tolerance give the designer two dials on the society’s portfolio, and neither assigns work to anyone. Principal investigators belong to labs, each with a short mandate; the lab is the unit that receives funding.

A city holds the labs and a single pool of credit, denominated in dollars of compute, the society’s only scarce resource. Proposals and reviews cost little; experiments do not. The society allocates credit through a four-step cycle of requests for proposals, or calls. A call states a brief, a budget, and a time horizon, goes to a chosen set of labs, and reserves its budget from the pool. Every principal investigator in a notified lab decides independently whether to write a proposal with a method and a budget or to decline; a decline is a complete document that argues against doing the work, and the society records it. Declines tell whoever wrote the call how the population assessed it. A fixed panel of three judge personas scores every proposal, each asking one question: a method judge asks whether the method supports the claim, a value judge whether the work is worth its cost, and a boldness judge whether it is a bet worth taking. Each judge scores each proposal independently, which guards against the order and first-proposal efects observed in LLM markets [5]. The highest-ranked proposals receive grants within the call’s budget, and unspent credit returns to the pool. A call difers from an assignment on three counts: it goes to a set of labs, any principal investigator may decline it with an argument, and the proposals that answer it compete under independent review.

Skeptics verify after funding: their proposals to replicate or refute the society’s results compete for credit like any other, they never exchange messages with the principal investigators whose work they check, and the society records their negative results as contributions. Each principal investigator keeps a knowledge base and a dated record of its proposals, declines, projects, and results, and a city library keeps every funded project and its papers. Rep utation derives from project success; its visibility to the panel is a design parameter, and when the panel sees it, the design pairs it with a lottery among proposals near the funding line [12]. At intervals, the society holds internal conferences at which principal investigators present results and read one another’s work [6, 46]. The intervals give the society time to test divergent approaches before the population sees which one is ahead, and the papers from one conference become the literature the next proposals cite.

One human governs each city as its mayor, through four levers: the calls (briefs, budgets, horizons), the invitations (which labs receive each call), the review (the judge personas, and grading alongside the panel), and the pool (how much credit the city has); the first three are text, and only the pool is a quantity. The mayor also seeds the city with labs of diferent mandates and principal investigators of diferent roles and risk tolerances, and may pursue one broad direction, a research aim stated in a sentence, across a sequence of calls. The mayor tells no principal investigator what to work on, and once a call is open, no human needs to act until results arrive.

The institutions decide what research the society does; groups do it. Inside a funded project, the principal investigator directs a group of agents, on the order of a hundred, who work as a pipeline under it, the right form once a funded proposal fixes the question. The society sees each group through a narrow interface: a funded proposal goes in, and a report with the provenance of every number comes out. The society’s researchers, principal investigators and group members alike, can therefore number in the thousands while the institutions operate over the few hundred principal investigators who propose and hold reputation. Table 1 maps each principle to its mechanisms.

## 4 First Evidence from a Running Society

We deployed the design as Research City, which runs its projects on Primus, our autonomous research infrastructure. Counting every agent that has done research in it, the deployment holds some ten thousand researchers. Several calls have completed the full cycle: labs filed on the order of 150 proposals in total, the panel scored them, and a few tens of grants became projects, with no human action between the opening of each call and the first project reports.

The boldness judge scores on a diferent scale. On the same proposals, the method and value judges scored within a point of each other on average, and the boldness judge scored roughly twelve points lower on a 100-point scale. Either bold proposals are scarce or that judge is too harsh; re-scoring the same proposals under a modified persona can tell the two apart.

Table 1: How the society implements each principle.
<table><tr><td>Principle</td><td>Mechanisms</td></tr><tr><td>P1 Purpose</td><td>Roles as dispositions; free decision to propose or decline; recorded declines</td></tr><tr><td>P2 Institutions</td><td>Credit ledger; call cycle; fixed panel; attributable identities</td></tr><tr><td>P3 Allocation</td><td>Calls with budgets; free decision to propose or decline; grants; the mayor&#x27;s four levers</td></tr><tr><td>P4 Diversity</td><td>Maverick, follower, and skeptic roles; risk toler- ance; lab mandates; periodic conferences</td></tr><tr><td>P5 Verification</td><td>Panel of independent single-question judges; skeptics separated from the work they check</td></tr><tr><td>P6Memory</td><td>Persistent principal investigators and labs; knowl- edge bases; reputation; library; proceedings</td></tr></table>

The mayor seeded the city with one direction: improve the pretraining of language models. The call named no technique and no hypothesis. The labs read the literature and proposed to study growth, building a model from a smaller trained one instead of training it from scratch, and the society’s work converged on it. Of the first labs funded, one measured a small gain from growth, and its paper stated two questions it could not answer: whether the small model’s training should count against the budget, and whether the gain came from the carried weights or from the schedule growth imposes. Another reported that the same kind of result gave two diferent compute verdicts depending on the accounting. The mayor wrote each later call against the labs’ papers, so the calls narrowed as the labs’ work did; the specificity came from the labs’ papers.

A later call went to every lab, and no one assigned the rebuild. A diferent lab, which had not taken part in the early work and had no stake in that paper’s numbers, proposed to rebuild the whole training pipeline and to rerun both arms under one closed budget that charges the small model’s training to the grown model, testing the closed-budget claim that growth still wins when it pays for the small model. Its from-scratch baseline landed well away from the earlier paper’s baseline, because the lab had re-derived the corpus, so the lab read every result against its own control. Against that control, the lab reported an advantage more than twice as large as the earlier paper had found: the grown model reached 17% lower perplexity (lower is better) than the same model trained from scratch, and read against the from-scratch scaling curve, it reaches that quality with about 30% less compute. With three seeds per arm, the from-scratch runs landed within a spread nearly a hundred times smaller than the gap. The paper states that it does not settle which mechanism produces the efect and names the experiment that would. Our hypothesis is that a planner would have had to decide in advance that the accounting was the problem, whereas the society funded the proposal that said so; C1 proposes the comparison that would test it.

No one assigned the questions the labs asked next, about when to grow a model, what a grown model costs to serve, and, under an open call that named no direction, topics unrelated to growth. The society’s papers include negative results, a failed prediction, and an unresolved disagreement, and the design gives each a place. One lab found that widening a model late in training left it unusable, and another reported that its two proposed remedies for a demographic bias did not work. One lab wrote a numerical prediction into its proposal and reported in its paper that the measurement disagreed. Several labs that never exchanged a message tested the closed-budget claim and did not all reach the same answer; the society’s current position is a split between them, and the mayor has opened a call written to settle it. Because credit goes to proposals, and a proposal to test a claim wins funding whether or not the claim survives, a negative result costs a lab no credit; its efect on reputation depends on what the record counts as success (C3). These results come from the society’s first direction and a small number of rounds, and the papers have not yet had external review.

## 5 Open Problems

A designed society turns questions about organizing agents into experiments; six are open.

C1. Comparing organizational forms under matched compute. We have not yet compared a society against a swarm or a planner under matched compute. We built the society to support this comparison, since a swarm baseline and a planner baseline can receive the same calls and the same pool ofcredit. A small precedent exists: with 16 agents, a network of principal investigators, review panels, and reputation produced more novel and higher-quality findings than independent agents at the same budget [33]. As the Agents4Science conference did [6], we will score the society’s papers with LLM reviewers calibrated on human-reviewed venues, and human reviewers will then read those above a threshold. Beyond paper quality, we will report three society-level measures that a single-project system cannot provide: the diversity of the funded portfolio across topics and roles, the correlation between panel scores and later review, and results per dollar of credit. Which baselines count as fair representatives of each form, and how long the comparison must run, are open.

C2. Calibrating institutions. Our judge personas and role proportions are first settings, not calibrated optima. Panel homogeneity is the analogue of a review panel drawn from a single school, and we regard a small panel of personas as a minimum. How often principal investigators should communicate to stay diverse without staying ignorant is open [59]. No funder can rerun a decade with a diferent proportion ofrisk-takers, but a society makes each ofthese a parameter, so it can serve as a testbed for the science of science. It can test whether long-horizon, failure-tolerant funding produces more novel work, as it did in the life sciences [4]; since every call carries its horizon as a field, it can test the claim that short horizons select for incremental work [4, 29] instead of assuming it. It can also ask which mix of mavericks and followers suits which problem, a question that agent-based models of science argued about but could not test empirically [1, 52, 56].

C3. Cumulative advantage. The Matthew efect will appear as soon as reputation afects funding [38]. A lottery band near the funding line [12] is one remedy. How wide the band should be, how visible reputation should be to reviewers, and whether a wellexecuted negative result counts as success in the reputation record are design choices that a society can vary and measure.

C4. Proposal inflation. One small call drew close to a hundred proposals. In human science, the cost of writing a proposal limits how many proposals scientists write; here a response costs only a proposal task, and that limit disappears. An application cost, a cap per lab, and triage before full review are candidate mechanisms, and choosing among them is open.

C5. Collusion and leakage. Reports describe collusion between reviewers and applicants in peer review, and LLM-agent simulations reproduce it [58]. Because applicants and judges never exchange messages, our design removes the direct channel for it, but LLM pricing agents that communicate only through prices still reach supracompetitive prices [14], and any mechanism that lets applicants rebut reviews would reopen the direct channel. Shared infrastructure is harder to govern: in one funded project, an agent listed the running jobs on the shared compute layer, adopted another lab’s experiment as its own earlier work, and cited its metrics. The society kept identities separate, but every lab ran on the same infrastructure. The episode echoes the July incident at small scale: institutional controls reach only the resources the institutions model, and extending them to shared infrastructure remains open.

C6. Governing a loop that improves its own institutions. Automated research is a loop of four steps: propose a question, allocate resources, execute, and evaluate. A society automates proposing and allocating, so with execution and evaluation automated the loop closes through the conference cycle of Section 3. When calls ask the society for better researchers, judges, or tools, the loop becomes the recursive self-improvement long anticipated in the literature [9, 16], applied to an institution. Our deployment has not run such a call, and we do not claim it would succeed. A society can rewrite text as easily as anything else, so calls, invitations, and review are within its reach; it cannot change the size of the pool. The pool of compute is therefore the one non-textual lever, and the natural point of human control. Some institution must decide the pace, beneficiaries, and cost of a self-improving loop, and it should be explicit and inspectable, unlike the organization the agents in the July incident improvised.

## 6 Conclusion

We have argued for organizing populations of research agents as a society in which principal investigators compete for a finite pool of compute through calls, review, and grants, and a human governs by allocation. The allocation mechanism (calls, review, and grants over a finite pool) is not specific to science; it could allocate any budget among competing agents.

## References

[1] J. McKenzie Alexander, Johannes Himmelreich, and Christopher Thompson. 2015. Epistemic Landscapes, Optimal Search, and the Division of Cognitive Labor. Philosophy ofScience 82, 3 (2015), 424–453. doi:10.1086/681766

[2] Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané. 2016. Concrete Problems in AI Safety. arXiv preprint arXiv:1606.06565.

[3] Alexander Artikis, Marek Sergot, and Jeremy Pitt. 2009. Specifying Norm-Governed Computational Societies. ACM Transactions on Computational Logic 10, 1 (2009). doi:10.1145/1459010.1459011

[4] Pierre Azoulay, Joshua S. Graf Zivin, and Gustavo Manso. 2011. Incentives and Creativity: Evidence from the Academic Life Sciences. RAND Journal of Economics 42, 3 (2011), 527–554. doi:10.1111/j.1756-2171.2011.00140.x

[5] Gagan Bansal, Wenyue Hua, Zezhou Huang, Adam Fourney, Amanda Swearngin, Will Epperson, Tyler Payne, Jake M. Hofman, Brendan Lucier, Chinmay Singh, Markus Mobius, Akshay Nambi, Archana Yadav, Kevin Gao, David M. Rothschild, Aleksandrs Slivkins, Daniel G. Goldstein, Hussein Mozannar, Nicole Immorlica, Maya Murad, Matthew Vogel, Subbarao Kambhampati, Eric Horvitz, and Saleema Amershi. 2025. Magentic Marketplace: An Open-Source Environment for Study ing Agentic Markets. arXiv preprint arXiv:2510.25779.

[6] Federico Bianchi, Owen Queen, Nitya Thakkar, Eric Sun, and James Zou. 2025. Exploring the Use of AI Authors and Reviewers at Agents4Science. arXiv preprint arXiv:2511.15534.

[7] Alexander Bondarenko, Denis Volk, Dmitrii Volkov, and Jefrey Ladish. 2025. Demonstrating Specification Gaming in Reasoning Models. arXiv preprint arXiv:2502.13295.

[8] Marcantonio Bracale Syrnikov, Federico Pierucci, Marcello Galisai, Matteo Prandi, Piercosma Bisconti, Francesco Giarrusso, Olga Sorokoletova, Vincenzo Suri ani, and Daniele Nardi. 2026. Institutional AI: Governing LLM Collusion in Multi-Agent Cournot Markets via Public Governance Graphs. arXiv preprint arXiv:2601.11369.

[9] Jef Clune. 2019. AI-GAs: AI-Generating Algorithms, an Alternate Paradigm for Producing General Artificial Intelligence. arXiv preprint arXiv:1905.10985.

[10] Lee D. Erman, Frederick Hayes-Roth, Victor R. Lesser, and D. Raj Reddy. 1980. The Hearsay-II Speech-Understanding System: Integrating Knowledge to Resolve Uncertainty. Comput. Surveys 12, 2 (1980), 213–253. doi:10.1145/356810.356816

[11] Marc Esteva, Juan A. Rodríguez-Aguilar, Carles Sierra, Pere Garcia, and Josep L. Arcos. 2001. On the Formal Specification of Electronic Institutions. In Agent Mediated Electronic Commerce. Lecture Notes in Computer Science, Vol. 1991. Springer, 126–147. doi:10.1007/3-540-44682-6\_8

[12] Ferric C. Fang and Arturo Casadevall. 2016. Research Funding: The Case for a Modified Lottery. mBio 7, 2 (2016). doi:10.1128/mBio.00422-16

[13] Jacques Ferber and Olivier Gutknecht. 1998. A Meta-Model for the Analysis and Design of Organizations in Multi-Agent Systems. In Proceedings of the International Conference on Multi Agent Systems (ICMAS). 128–135. doi:10.1109/ICMAS. 1998.699041

[14] Sara Fish, Yannai A. Gonczarowski, and Ran I. Shorrer. 2024. Algorithmic Collusion by Large Language Models. arXiv preprint arXiv:2404.00806.

[15] Jacob G. Foster, Andrey Rzhetsky, and James A. Evans. 2015. Tradition and Innovation in Scientists’ Research Strategies. American Sociological Review 80, 5 (2015), 875–908. doi:10.1177/0003122415601618

[16] Irving John Good. 1965. Speculations Concerning the First Ultraintelligent Machine. Advances in Computers 6 (1965), 31–88.

[17] Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, Petar Sirkovic, Ar tiom Myaskovsky, Grzegorz Glowaty, Felix Weissenberger, Alessio Orlandi, Dan Popovici, Anil Palepu, Keran Rong, Ryutaro Tanno, Khaled Saab, Fan Zhang, Jacob Blum, Andrew Carroll, Kavita Kulkarni, Nenad Tomasev, Dina Zverinski, Ivor Rendulic, Elahe Vedadi, Florian Hasler, Luka Rimanic, Marina Boia, Ivan Budiselic, Ben Feinstein, Mathias Bellaiche, Tom Shefer, Jan Freyberg, Jeremy Ratclif, Ottavia Bertolli, Katherine Chou, Avinatan Hassidim, Burak Gokturk Amin Vahdat, Yuan Guan, Vikram Dhillon, Eeshit Dhaval Vaishnav, Byron Lee, Tiago R D Costa, José R Penadés, Gary Peltz, Yossi Matias, James Manyika, Demis Hassabis, Yunhan Xu, Pushmeet Kohli, Annalisa Pawlosky, Alan Karthike salingam, and Vivek Natarajan. 2025. Towards an AI Co-Scientist. arXiv preprint arXiv:2502.18864.

[18] Ryan Greenblatt, Ajeya Cotra, and Hjalmar Wijk. 2026. Brief Independent Investigation of Agents’ Behavior, Reasoning and Collaboration in the OpenAI / Hugging Face Hacking Incident. METR blog. https://metr.org/blog/2026-08-26- openai-hugging-face-incident-investigation

[19] Friedrich A. Hayek. 1945. The Use of Knowledge in Society. American Economic Review 35, 4 (1945), 519–530.

[20] Bryan Horling and Victor Lesser. 2004. A Survey of Multi-Agent Organizational Paradigms. The Knowledge Engineering Review 19, 4 (2004), 281–316. doi:10.1017/ S0269888905000317

[21] Jomi F. Hübner, Jaime S. Sichman, and Olivier Boissier. 2002. A Model for the Structural, Functional, and Deontic Specification of Organizations in Multiagent Systems. In Advances in Artificial Intelligence (SBIA 2002). Lecture Notes in Computer Science, Vol. 2507. Springer, 118–128. doi:10.1007/3-540-36127-8\_12

[22] Hugging Face. 2026. Security Incident Disclosure — July 2026. Hugging Face blog. https://huggingface.co/blog/security-incident-july-2026

[23] Max Jaderberg, Valentin Dalibard, Simon Osindero, Wojciech M. Czarnecki, Jef Donahue, Ali Razavi, Oriol Vinyals, Tim Green, Iain Dunning, Karen Simonyan, Chrisantha Fernando, and Koray Kavukcuoglu. 2017. Population Based Training of Neural Networks. arXiv preprint arXiv:1711.09846.

[24] Liwei Jiang, Yuanjun Chai, Margaret Li, Mickel Liu, Raymond Fok, Nouha Dziri, Yulia Tsvetkov, Maarten Sap, Alon Albalak, and Yejin Choi. 2025. Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond). arXiv preprint arXiv:2510.22954. NeurIPS 2025 Datasets and Benchmarks.

[25] Yubin Kim, Ken Gu, Chanwoo Park, Chunjong Park, Samuel Schmidgall, A. Ali Heydari, Yao Yan, Zhihan Zhang, Yuchen Zhuang, Yun Liu, Mark Malhotra, Paul Pu Liang, Hae Won Park, Yuzhe Yang, Xuhai Xu, Yilun Du, Shwetak Patel, Tim Althof, Daniel McDuf, and Xin Liu. 2025. Towards a Science of Scaling Agent Systems. arXiv preprint arXiv:2512.08296.

[26] Philip Kitcher. 1990. The Division of Cognitive Labor. The Journal of Philosophy 87, 1 (1990), 5–22. doi:10.2307/2026796

[27] William A. Kornfeld and Carl E. Hewitt. 1981. The Scientific Community Metaphor. IEEE Transactions on Systems, Man, and Cybernetics 11, 1 (1981), 24–33. doi:10.1109/TSMC.1981.4308575

[28] Victoria Krakovna, Jonathan Uesato, Vladimir Mikulik, Matthew Rahtz, Tom Everitt, Ramana Kumar, Zac Kenton, Jan Leike, and Shane Legg. 2020. Specification Gaming: The Flip Side of AI Ingenuity. Google DeepMind blog. https://deepmind.google/discover/blog/specification-gaming-the-flipside-of-ai-ingenuity/

[29] Thomas S. Kuhn. 1962. The Structure ofScientific Revolutions. University of Chicago Press.

[30] Emanuele La Malfa, Gabriele La Malfa, Samuele Marro, Jie M. Zhang, Elizabeth Black, Michael Luck, Philip Torr, and Michael Wooldridge. 2025. Large Language Models Miss the Multi-Agent Mark. arXiv preprint arXiv:2505.21298. NeurIPS 2025 position track.

[31] David Lazer and Allan Friedman. 2007. The Network Structure of Exploration and Exploitation. Administrative Science Quarterly 52, 4 (2007), 667–694. doi:10. 2189/asqu.52.4.667

[32] Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. 2023. CAMEL: Communicative Agents for “Mind” Exploration of Large Language Model Society. In Advances in Neural Information Processing Systems. arXiv:2303.17760.

[33] Tennison Liu, Silas Ruhrberg Estévez, David L. Bentley, and Mihaela van der Schaar. 2025. Hypothesis Hunting with Evolving Networks of Autonomous Scientific Agents. arXiv preprint arXiv:2510.08619.

[34] Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jef Clune, and David Ha. 2024. The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery. arXiv preprint arXiv:2408.06292.

[35] James G. March. 1991. Exploration and Exploitation in Organizational Learning. Organization Science 2, 1 (1991), 71–87. doi:10.1287/orsc.2.1.71

[36] Alexander Meinke, Bronson Schoen, Jérémy Scheurer, Mikita Balesni, Rusheb Shah, and Marius Hobbhahn. 2024. Frontier Models are Capable of In-context Scheming. arXiv preprint arXiv:2412.04984

[37] Robert K. Merton. 1942. A Note on Science and Democracy. Journal of Legal and Political Sociology 1 (1942), 115–126. Reprinted as “The Normative Structure of Science”.

[38] Robert K. Merton. 1968. The Matthew Efect in Science. Science 159, 3810 (1968), 56–63. doi:10.1126/science.159.3810.56

[39] Stephen M. Omohundro. 2008. The Basic AI Drives. In Artificial General Intelligence 2008: Proceedings of the First AGI Conference (Frontiers in Artificial Intelligence and Applications, Vol. 171). IOS Press, 483–492.

[40] Elinor Ostrom. 1990. Governing the Commons: The Evolution ofInstitutions for Collective Action. Cambridge University Press. doi:10.1017/CBO9780511807763

[41] Satoshi Oyama, Yuko Sakurai, and Hisashi Kashima. 2026. MACC: Multi-Agent Collaborative Competition for Scientific Exploration. In Proceedings ofthe 25th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2026), Blue Sky Ideas Track. IFAAMAS, Richland, SC. doi:10.65109/JLGE7606

[42] Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. 2023. Generative Agents: Interactive Simulacra of Human Behavior. In Proceedings ofthe 36th Annual ACM Symposium on User Interface Software and Technology (UIST). doi:10.1145/3586183.3606763

[43] Jinghua Piao, Yuwei Yan, Jun Zhang, Nian Li, Junbo Yan, Xiaochong Lan, Zhihong Lu, Zhiheng Zheng, Jing Yi Wang, Di Zhou, Chen Gao, Fengli Xu, Fang Zhang, Ke Rong, Jun Su, and Yong Li. 2025. AgentSociety: Large-Scale Simulation of LLM-Driven Generative Agents Advances Understanding of Human Behaviors and Society. arXiv preprint arXiv:2502.08691.

[44] Michael Polanyi. 1962. The Republic of Science: Its Political and Economic Theory. Minerva 1, 1 (1962), 54–73. doi:10.1007/BF01101453

[45] Anbang Ruan and Xing Zhang. 2026. AgentCity: Constitutional Governance for Autonomous Agent Economies via Separation of Power. arXiv preprint arXiv:2604.07007.

[46] Samuel Schmidgall and Michael Moor. 2025. AgentRxiv: Towards Collaborative Autonomous Research. arXiv preprint arXiv:2503.18102.

[47] Yoav Shoham and Moshe Tennenholtz. 1995. On Social Laws for Artificial Agent Societies: Of-Line Design. Artificial Intelligence 73, 1–2 (1995), 231–252. doi:10.1016/0004-3702(94)00007-N

[48] Chenglei Si, Diyi Yang, and Tatsunori Hashimoto. 2024. Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers. arXiv preprint arXiv:2409.04109.

[49] Joar Skalse, Nikolaus H. R. Howe, Dmitrii Krasheninnikov, and David Krueger. 2022. Defining and Characterizing Reward Hacking. In Advances in Neural Information Processing Systems, Vol. 35.

[50] Reid G. Smith. 1980. The Contract Net Protocol: High-Level Communication and Control in a Distributed Problem Solver. IEEE Trans. Comput. C-29, 12 (1980), 1104–1113. doi:10.1109/TC.1980.1675516

[51] Kyle Swanson, Wesley Wu, Nash L. Bulaong, John E. Pak, and James Zou. 2025. The Virtual Lab of AI Agents Designs New SARS-CoV-2 Nanobodies. Nature 646 (2025), 716–723. doi:10.1038/s41586-025-09442-9

[52] Johanna Thoma. 2015. The Epistemic Division of Labor Revisited. Philosophy of Science 82, 3 (2015), 454–472. doi:10.1086/681768

[53] Alexander Matt Turner, Logan Smith, Rohin Shah, Andrew Critch, and Prasad Tadepalli. 2021. Optimal Policies Tend to Seek Power. In Advances in Neural Information Processing Systems, Vol. 34.

[54] Aron Vallinder and Edward Hughes. 2024. Cultural Evolution of Cooperation among LLM Agents. arXiv preprint arXiv:2412.10270.

[55] Lukas Weidener, Marko Brkić, Phillip Lee, Martin Karlsson, Kevin Noessler, and Paul Kohlhaas. 2026. From Agent-Only Social Networks to Autonomous

Scientific Research: Lessons from OpenClaw and Moltbook, and the Architecture of ClawdLab and Beach.Science. arXiv preprint arXiv:2602.19810.

[56] Michael Weisberg and Ryan Muldoon. 2009. Epistemic Landscapes and the Division of Cognitive Labor. Philosophy of Science 76, 2 (2009), 225–252. doi:10. 1086/644786

[57] Andrea Wynn, Harsh Satija, and Gillian Hadfield. 2025. Talk Isn’t Always Cheap: Understanding Failure Modes in Multi-Agent Debate. arXiv preprint arXiv:2509.05396.

[58] Jicheng Zhou, Kemou Li, Kahim Wong, Zheyuan Li, Zhuan Shi, Fengpeng Li, Haiwei Wu, and Jiantao Zhou. 2026. CABAL: Multi-Agent Simulacra for Tracing the Efects of Collusive Bidding in Peer Review. arXiv preprint arXiv:2609.05227.

[59] KevinJ. S. Zollman. 2010. The Epistemic Benefit ofTransient Diversity. Erkenntnis 72, 1 (2010), 17–35. doi:10.1007/s10670-009-9194-6