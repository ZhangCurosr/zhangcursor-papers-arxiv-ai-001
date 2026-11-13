# From More to Better: Causal Inference and Digital Experimentation in the Age of AI

Winston Chou<sup>1</sup>, Mårten Schultzberg<sup>2</sup>, Lo-Hua Yuan<sup>3</sup>, Wenjing Zheng<sup>4</sup>, Avi Feller<sup>5</sup>, David   
Holtz<sup>6,9</sup>, Widad Machmouchi<sup>7</sup>, Sven Schmit<sup>6</sup>, Gagan Bhat<sup>8</sup>, Simon Ejdemyr<sup>1</sup>, Veronica Hannan<sup>1</sup>, Amy Hartwig<sup>7</sup>, Mahnaz Islam<sup>10</sup>, Nathan Kallus<sup>1,12</sup>, Jeremy Karp<sup>4</sup>, Kyle Kretschman<sup>2</sup>, Hannah Li<sup>9</sup>, James McQueen<sup>10</sup>, Evan Miller<sup>8</sup>, Nicolas Stier<sup>11</sup>, and Sean 6   
<sup>1</sup>Netflix   
<sup>2</sup>Spotify   
<sup>3</sup>Independent   
<sup>4</sup>Roblox   
<sup>5</sup>University of California, Berkeley   
<sup>6</sup>OpenAI   
<sup>7</sup>Microsoft   
<sup>8</sup>Anthropic   
<sup>9</sup>Columbia University   
<sup>10</sup>Amazon   
<sup>11</sup>Meta   
<sup>12</sup>Cornell University

## Abstract

Artificial intelligence (AI) is quickly expanding the capacity of organizations to produce and analyze data. We argue that this expansion shifts the leverage of data scientists from performing individual analyses to assessing the credibility and importance of findings and determining when evidence generalizes across contexts. Drawing on a roundtable of academic researchers and industry practitioners, we examine three ways in which data scientists who work on causal inference and experimentation can help organizations turn more analytical capacity into better decisions: preserving and accumulating the learning value of experimentation beyond individual tests; choosing and combining experimental, observational, and synthetic evidence; and ensuring evaluation keeps pace with AI as its capabilities and uses evolve. In each area, we distinguish established practice from open questions. We conclude with implications for the training and practice of data science and for organizations.

## 1 Introduction

Artificial intelligence (AI) is transforming how organizations generate, analyze, and use data. As the barriers to writing code and analyzing data fall, more people can investigate questions that previously required specialized expertise. Yet more analytical capacity does not necessarily yield deeper understanding or better decisions. The productivity gains from earlier general-purpose technologies, such as electricity and the internet, depended on complementary if intangible investments in processes and skills (Brynjolfsson et al., 2021). Likewise, the value of more analysis will depend on what organizations build around it.

We argue that, as AI makes data and analyses easier to generate, the central challenge in causal inference and experimentation is shifting away from producing individual analyses to assessing the relevance and credibility of findings and determining when they generalize across contexts. Observational causal inference ofers a useful analogy: an observational estimate, even when backed by millions of data points, is credible only under assumptions that link the data to the causal question and counterfactuals at hand (Hernán and Robins, 2006). Similarly, AI equips organizations with exciting new capabilities, such as the simulation of new treatments on synthetic respondents; yet whether and how these capabilities should inform decisions afecting real users requires assumptions about internal validity, external validity, and transportability that should be stated and scrutinized (Pearl and Bareinboim, 2014; Degtiar and Rose, 2023; Egami and Hartman, 2023; Persson et al., 2026).

To be sure, causal inference and experimentation have always involved more than estimating treatment efects. Understanding why an intervention works, determining whether its efects generalize, reconciling conflicting findings, and connecting empirical results to meaningful outcomes are longstanding challenges in digital experimentation (Gupta et al., 2019; Holtz et al., 2026). However, AI intensifies these challenges in three distinct ways. By accelerating product development and lowering barriers to analysis, AI increases both the demand for measurement and the volume of results that must be critically evaluated. As a source of synthetic evidence, AI expands what organizations can simulate without testing on real users — though, again, drawing the appropriate conclusions from these simulations requires causal assumptions linking simulated responses to real user behavior (Persson et al., 2026). Lastly, as an open-ended and rapidly evolving product, AI upends static measures of success, especially as users discover new applications and adapt their behavior (Park et al., 2026).

In this paper, we outline three ways data scientists can help organizations meet these challenges:

1. Understand causal mechanisms and build organizational knowledge. As AI accelerates product development and analysis, data scientists can help organizations move beyond individual experiments and decisions toward cumulative learning. We examine how causal reasoning and experimental metadata can help produce and preserve understanding of generalizable mechanisms that explain whether interventions succeed or fail.

2. Establish playbooks for how to choose and combine diferent sources of evidence to support a decision. As synthetic, observational, experimental, and hybrid sources of evidence become more accessible, organizations need to be aware of their assumptions and limitations. We discuss the considerations behind choosing and combining these sources and where the validity of synthetic evidence remains unresolved.

3. Build evaluation frameworks that adapt to evolving AI-powered products. As products become more open-ended and their behavior less predictable, organizations must actively evolve what they measure and scrutinize whether and how past findings generalize. We examine what existing methods can establish and where novel evaluation frameworks may be needed.

Our perspective draws on the 2026 Causal Inference and Digital Experimentation Roundtable (CIDER), which convened participants from technology companies and academic institutions to discuss the intersection of data science and AI.<sup>1</sup> From the CIDER discussions and the authors’ collective experience, we distill several key claims, highlighted in bold. We use the following vocabulary to communicate the level of agreement among authors. Consensus refers to propositions broadly endorsed by the author group. Shared refers to propositions expressed by multiple authors without substantial pushback. Contested refers to propositions having substantive disagreement between authors.

## 2 From Individual Experiments to Organizational Learning

As AI accelerates feature development, organizations face growing pressure to evaluate more ideas. Yet when experiments are conducted, organizations often fail to capture their learning value beyond the immediate decision (such as the choice to “launch” a tested innovation). An experiment generates learning value when it changes a relevant belief or informs a subsequent action. An experiment can fail to justify the launch of a new feature or the scaling up of a novel intervention and still teach the organization something useful. However, without recording the experimenter’s intent, observations, and reasoning, this learning value can be dificult to share and reconstruct.

## 2.1 Mechanisms and metadata for organizational learning

Our first proposition is that, as AI increases the volume of experimentation, data scientists should prioritize learning across experiments. Of special importance are generalizable mechanisms that help explain why interventions succeed or fail. Mechanism metrics — variables hypothesized to mediate the efects of treatment on downstream outcomes — are key to diagnosing whether an intervention was delivered as intended, whether the intermediate processes it sought to afect changed, and whether relevant user or business outcomes improved. Mechanistic reasoning can also guide follow-up experiments, including tests that probe the mechanism directly.<sup>2</sup>

A corollary of this proposition is that learning and launching are distinct goals. The launch decision asks whether a treatment should be released to users or whether an intervention should be scaled up. The learning objective asks whether the experiment improved the organization’s understanding of users, products, constraints, and tradeofs. These goals can diverge: an experiment that fails to justify a change can nevertheless rule out a hypothesized mechanism or inform which interventions a team pursues next. Some tests are designed as learning tests that will not launch regardless of the results. For example, at Netflix, experiments are run to test structural models (Zielnicki et al., 2026) and to measure the cumulative impact of an experimentation program (Aridor et al., 2026).

Accumulating knowledge across experiments necessitates records of what was tested and how the results were interpreted, not just estimated treatment efects. To interpret a past result and assess its relevance to a new decision, teams need to know what experimenters tried to test; which core, mechanism, and guardrail metrics actually changed; which users were afected; and how the findings informed subsequent decisions. Code and feature flags can help verify what was actually implemented. Circumstances such as the user population, product state, and model version can also afect whether a past result carries over to a new decision. Records should distinguish statistical findings from proposed explanations and include inconclusive, abandoned, and screened-out ideas. Otherwise, a repository may disproportionately preserve successes and memorable failures. Especially as experimentation scales, incomplete metadata exacerbates the risk of repeatedly testing similar ideas without accumulating understanding (Kohavi et al., 2020).

In many organizations, this institutional knowledge is often tacit or scattered across documents, if written down at all. One promising application of AI is to extract structure from code changes, strategy documents, and other unstructured artifacts. These records, especially when reviewed by human experts and traceable to authoritative sources, can also provide vetted context for AI agents that draw on past experiments to propose or evaluate new interventions.

## 2.2 Balancing structure and burden in metadata collection

The value of experimental metadata was consensus; how much structure to require in collecting it was contested. Some authors advocated for structured elicitation of metadata, for example, requiring hypotheses, causal diagrams, or intended mechanisms before running a test. Others noted that requiring such structure can result in false precision and $/ \mathrm { o r }$ low-quality responses from annoyed partners. Furthermore, complex hypotheses may be dificult to represent adequately in a structured manner. One compromise is to ask for lightweight structure in exchange for automation: for example, specify the intended treatment(s) and primary metrics for sample size calculation, or record launch decisions to support cumulative impact measurement.

## 3 Beyond Classical A/B Testing: Synthetic and Hybrid Measurement

How organizations choose and combine evidence should depend on the question or decision at hand. This principle applies equally to experimentation and synthetic or observational evidence. For example, even a well-designed fixed-allocation $\mathrm { A } / \mathrm { B }$ test may be inappropriate when adaptive experimentation would better serve the decision (Hadad et al., 2021), or when faster and less costly sources of evidence would sufice. AI expands the set of measurement options available to organizations, increasing the importance of deciding which sources of evidence are appropriate for a given decision.

## 3.1 Evaluating and combining diverse sources of evidence

Our core proposition here is that evidence must be validated against its intended use. A model used to screen ideas need not estimate their efects precisely, but it should reliably distinguish promising candidates from those unlikely to succeed. In contrast, a model used to support launch decisions should reliably estimate the magnitude of user benefits and harms.

Experimentation remains the gold standard for evidence of real user behavior when exposed to novel interventions. However, organizations overuse it when they treat it as the sole arbiter of decisions requiring multiple sources of evidence (Deaton and Cartwright, 2018). A useful metaphor is the “evidence ladder”: synthetic and ofline evaluations are useful for screening ideas, pilots and online experiments for assessing promising candidates, and longer-term holdbacks for measuring long-term cumulative impact. Live experiments remain important for evaluating efects on real users, but not every decision needs to ascend the entire ladder.

Large language models (LLMs) may reproduce observed human responses, but this does not establish that they can estimate diferences between treatments or predict the efects of new interventions. Existing research on the fidelity of synthetic respondents paints a mixed picture of their capabilities. Argyle et al. (2023) and Horton et al. (2023) find that LLMs are able to reproduce aspects of human behavior in political surveys and economic experiments, respectively. These results provide evidence of usefulness in particular settings, but leave open how that performance might translate to new tasks or interventions.

Toubia et al. (2025) introduce Twin-2K-500, a dataset for building and evaluating personalized synthetic respondents (“digital twins”). Although their digital twins predict held-out individual responses better than a random benchmark, they reproduce only some of the experimental patterns observed among human participants. Gao et al. (2025) find that the models they study generally fail to reproduce human choices and responses to changes in a strategic decision task. Wang et al. (2025) find that LLMs can flatten variation within demographic groups and sometimes resemble outsiders’ portrayals more closely than group members’ own responses.

If recent history is a guide, the capabilities of LLMs to emulate human behavior are likely to improve over time. However, the underlying principle — to evaluate models against their intended use — remains the same. For example, models that screen new treatments should be audited on a random subset of the ideas they reject (and which are otherwise suitable for live user testing). Models used to predict treatment efects should be evaluated against held-out experiments, with special care to ensure that the experimental results do not leak into their training data.

Combining multiple sources of evidence is a well-worn strategy that can help organizations avoid making sweeping judgments about diferent types of data. Prediction-powered inference, for example, combines many model predictions with a smaller sample of measured outcomes. Comparing the two allows researchers to correct the final estimates for prediction error while accounting for uncertainty in the correction (Angelopoulos et al., 2023). The surrogate index is another example of data combination: it uses observational data to relate short-term proxies to a long-term outcome, then applies that relationship to short-term outcomes measured in an experiment. Under surrogacy and comparability assumptions, this allows researchers to estimate long-term treatment efects before those outcomes can be observed in the experiment (Athey et al., 2026).

Crucially, these methods are valid only under assumptions that can be clearly articulated and scrutinized. Combining human and synthetic responses requires similar care: researchers need to specify how synthetic responses relate to human responses and how prediction errors afect the resulting estimates and uncertainty (Persson et al., 2026).

## 3.2 When can synthetic evidence substitute for live experiments?

Although the need to validate evidence against its intended use was a widely shared proposition, whether synthetic evidence can replace live experiments remains contested. The case for synthetic evidence is strongest when live user data is expensive, slow, scarce, or risky to collect. But the same conditions that make simulation attractive can make bias costly. Self-driving is a high-stakes example: simulation is vital for reducing exposure to risk, but failure to simulate rare real-world scenarios can be catastrophic (Schwall et al., 2020; Scanlon et al., 2021).

To be sure, live user experiments also have their limits. Detecting rare outcomes or estimating efects across interacting users requires an appropriate experimental design and suficient observations (Hanley and Lippman-Hand, 1983). These limitations strengthen the case for considering and combining experiments with other evidence. Existing statistical concepts such as surrogacy and transportability provide a vocabulary for making some of the necessary assumptions explicit (Persson et al., 2026). Organizations should also ask the practical question of what evidence from live users would change a decision, especially if that evidence can itself be noisy or nondefinitive.

## 4 Evaluating Non-Deterministic and AI-Powered Systems

Task ambiguity, user learning and adaptation, and rare failures are longstanding challenges in the evaluation of digital products. AI products intensify these challenges through the sheer variability of their applications and outputs. A key tension here is that, although organizations must continuously revise their evaluations as these use cases emerge, they also need to curate an ever-expanding set of evaluations and distill the measurements that are most consequential for decision-making and product development. We encapsulate this tension with the next two propositions.

## 4.1 Evaluating products as their uses change

Our first proposition here is that evaluation needs evolve with how a product is used. This proposition may seem self-evident, yet the challenge of keeping pace with AI products is not always fully appreciated. For example, a two-week experiment on a coding agent might cover routine debugging, whereas after deployment users adopt it for code migrations. Under appropriate assumptions, reweighting can account for diferences in the prevalence of tasks represented in the experiment, but it cannot recover efects for entirely unrepresented tasks, regardless of how much data was collected in the experiment (Dahabreh et al., 2019). Even when tasks remain similar, how users interact with the product can change (Sigerson et al., 2026). Some users might become more adept at prompting the model for certain tasks. Others deterred by an early bad experience might not return, even after the model improves.

Although the authors shared the view that AI products present unique challenges, whether these challenges warrant genuinely new methodologies and evaluation frameworks was contested. On the one hand, tools such as controlled rollouts, guardrail metrics, and anytime-valid inference remain useful for catching unexpected regressions and undesirable behavior (Howard et al., 2021;

Johari et al., 2022). Counterfactual or “shadow” logging, widely used in advertising measurement (Johnson et al., 2017), ofers another useful tool: recording what an alternative system would have done without exposing users to its output. For AI products, shadow evaluations can run candidate and incumbent models on the same live inputs while showing users only the incumbent’s response. This enables paired comparisons of output quality, safety, and computational cost (Machmouchi and Gupta, 2023).

On the other hand, the multimodal and qualitative nature of LLM output increases the need for new and correspondingly rich evidence sources. For example, screenshots, user reports, and human review can all reveal failures that aggregate measures miss (cf. Shankar et al., 2024).

## 4.2 Connecting evaluations to product decisions

The previous proposition sits in tension with our next proposition, which is that each metric and evaluation should have a clear role in decision-making. Continuously adding evaluations for new tasks and failures is important for maintaining coverage of how the product is used. However, each new evaluation also makes it harder to interpret the results holistically.

Not every evaluation needs to determine whether a model should launch. For example, an evaluation of a particular debugging task can help developers identify and fix a weakness of a coding agent, but whether that weakness should block deployment depends on how often users encounter it and how serious its consequences are. Other evaluations serve as guardrails: rare but damaging model failures might be suficient to block a launch even when average quality improves. In general, each measure should have a clear role in decision-making based on its relationship to higher-level organizational goals (Rodden et al., 2010).

How exactly to tame the proliferation of metrics and evaluations was contested. Some authors focused on the infrastructure needed to run, maintain, and inspect large numbers of evaluations. Others focused on measuring the connection of each metric to organizational north stars, such as long-term user retention and revenue. Both concerns bear on the question of which measurements to add, update, or retire. Data scientists can help teams make these choices by specifying what each evaluation is intended to establish and how its results should afect product development or deployment.

## 5 Implications for Data Science Practice and Organizations

## 5.1 Data science practice and training

As models become more adept at analyzing data, we anticipate that data scientists will spend more time selecting and framing important questions, evaluating the systems and workflows responsible for producing analyses, interpreting results, and making decisions. Foundational and frontier expertise are complementary here. On the one hand, foundational expertise in statistical, computational, and domain knowledge can help with specifying an agent’s task, recognizing analytical errors, and evaluating its conclusions (National Academies of Sciences, Engineering, and Medicine, 2018). On the other hand, hands-on experience with AI tools can help with understanding their capabilities and failure modes.

A shared concern is that, although the authors generally expect the importance of context, judgment, and communication skills to increase, AI may automate some of the tasks through which junior data scientists have traditionally acquired those assets. For example, using AI, an analyst can quickly generate a working query without understanding the underlying data model or intuiting when the result may be misleading. In a relevant study of developers learning an unfamiliar programming library, Shen and Tamkin (2026) found that AI assistance impaired conceptual understanding, code reading, and debugging abilities. Providing forums where humans review AIgenerated analyses or consequential decisions together is one way that organizations can help junior data scientists hone their data literacy and judgment.

## 5.2 Organizing analytical work

Consider two agents that produce equally sophisticated analyses of the same experiment using diferent populations or observation windows. Both analyses may be correct, but address subtly diferent questions. The organization needs to determine which question most closely corresponds to the decision at hand. Ideally, it should also record the rationale behind its choice, so that even this decision — how an organization chooses to learn — can be audited and improved over time.

Some of this work can be incorporated into analytical infrastructure (Cherry et al., 2026). Standard calculations and filters can be implemented as deterministic routines for agents to invoke and assemble. More nuanced choices, such as navigating metric tradeofs and reconciling conflicting findings, may require explicit organizational alignment and deliberate curation. Mechanisms for designating “blessed” analyses, such as watermarks, can help organizations keep a clean record of findings and decisions, again to support organizational learning.

How exactly analytical work will be organized remains contested. Some authors expect analyses to flow through more centralized infrastructure, and for scientific work to index more heavily on building and evaluating those analytical systems. Others expect decentralization, as seasoned analysts can investigate more questions with greater nuance and depth without falling back to onesize-fits-all tooling. Yet another possibility lies somewhere in between, with organizations supplying shared context and primitives (e.g., metric definitions) to support arbitrarily deep analyses.

## 6 Conclusion

AI is expanding the capacity of organizations to generate ideas, data, and analyses. Data scientists can equip organizations to harness this capacity by preserving and accumulating knowledge across experiments and over time, choosing and assembling the right mix of evidence for a given decision, and revising evaluations as products and their uses evolve.

Important questions remain. What metadata is suficient for organizations to accumulate knowledge, and how do we make its collection incentive-compatible? How reliable is synthetic evidence as a substitute for live user trafic, and how can the two be efectively combined? When should an evaluation be added or retired, and what evidence connects it to long-term value? Given the rapid pace of AI development, the more technical of these questions may well be resolved by the next CIDER. However, other questions — especially those that stem from the underlying challenges of helping organizations learn, innovate, and make decisions under uncertainty — are likely to remain evergreen.

## References

Anastasios N. Angelopoulos, Stephen Bates, Clara Fannjiang, Michael I. Jordan, and Tijana Zrnić. Prediction-powered inference. Science, 382(6671):669–674, 2023. doi: 10.1126/science.adi6000.

Lisa P. Argyle, Ethan C. Busby, Nancy Fulda, Joshua R. Gubler, Christopher Rytting, and David

Wingate. Out of one, many: Using language models to simulate human samples. Political Analysis, 31(3):337–351, 2023. doi: 10.1017/pan.2023.2.

Guy Aridor, Winston Chou, Nathan Kallus, Antoine Scheid, Allen Tran, and Kevin Zielnicki. Recommendation quality and the concentration of consumption: Experimental evidence from Netflix, 2026.

Susan Athey, Raj Chetty, Guido W. Imbens, and Hyunseung Kang. The surrogate index: Combining short-term proxies to estimate long-term treatment efects more rapidly and precisely. The Review of Economic Studies, 93(4):2284–2312, 2026. doi: 10.1093/restud/rdaf087.

Aurélien Bibaut, Winston Chou, Simon Ejdemyr, and Nathan Kallus. Learning the covariance of treatment efects across many weak experiments. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 153–162, 2024. doi: 10.1145/ 3637528.3672034.

Erik Brynjolfsson, Daniel Rock, and Chad Syverson. The productivity j-curve: How intangibles complement general purpose technologies. American Economic Journal: Macroeconomics, 13(1): 333–372, 2021. doi: 10.1257/mac.20180386.

Josh Cherry, Clement Peng, Johanne Jiao, Justin Leder, and Chen Chang. How Anthropic enables self-service data analytics with Claude, June 2026.

Issa J. Dahabreh, Sarah E. Robertson, Eric J. Tchetgen Tchetgen, Elizabeth A. Stuart, and Miguel A. Hernán. Generalizing causal inferences from individuals in randomized trials to all trial-eligible individuals. Biometrics, 75(2):685–694, 2019. doi: 10.1111/biom.13009.

Angus Deaton and Nancy Cartwright. Understanding and misunderstanding randomized controlled trials. Social Science & Medicine, 210:2–21, 2018. doi: 10.1016/j.socscimed.2017.12.005.

Irina Degtiar and Sherri Rose. A review of generalizability and transportability. Annual Review of Statistics and Its Application, 10(1):501–524, 2023. doi: 10.1146/ annurev-statistics-042522-103837.

Naoki Egami and Erin Hartman. Elements of external validity: Framework, design, and analysis. American Political Science Review, 117(3):1070–1088, 2023. doi: 10.1017/S0003055422000880.

Yuan Gao, Dokyun Lee, Gordon Burtch, and Sina Fazelpour. Take caution in using LLMs as human surrogates. Proceedings of the National Academy of Sciences, 122(24):e2501660122, 2025. doi: 10.1073/pnas.2501660122.

Somit Gupta, Ronny Kohavi, Diane Tang, Ya Xu, Reid Andersen, Eytan Bakshy, Niall Cardin, Sumitha Chandran, Nanyu Chen, Dominic Coey, Mike Curtis, Alex Deng, Weitao Duan, Peter Forbes, Brian Frasca, Tommy Guy, Guido W. Imbens, Guillaume Saint Jacques, Pranav Kantawala, Ilya Katsev, Moshe Katzwer, Mikael Konutgan, Elena Kunakova, Minyong Lee, MJ Lee, Joseph Liu, James McQueen, Amir Najmi, Brent Smith, Vivek Trehan, Lukas Vermeer, Toby Walker, Jefrey Wong, and Igor Yashkov. Top challenges from the first practical online controlled experiments summit. ACM SIGKDD Explorations Newsletter, 21(1):20–35, 2019. doi: 10.1145/3331651.3331655.

Vitor Hadad, David A. Hirshberg, Ruohan Zhan, Stefan Wager, and Susan Athey. Confidence intervals for policy evaluation in adaptive experiments. Proceedings of the National Academy of Sciences, 118(15):e2014602118, 2021. doi: 10.1073/pnas.2014602118.

James A. Hanley and Abby Lippman-Hand. If nothing goes wrong, is everything all right? interpreting zero numerators. JAMA, 249(13):1743–1745, 1983. doi: 10.1001/jama.1983.03330370053031.

Miguel A. Hernán and James M. Robins. Estimating causal efects from epidemiological data. Journal of Epidemiology and Community Health, 60(7):578–586, 2006. doi: 10.1136/jech.2004. 029496.

David Holtz, Iavor Bojinov, Ramesh Johari, Nathan Kallus, Apoorva Lal, Sathya Anand, Kyle Carlson, Brent Cohn, Tom Cunningham, Alex Deng, Maria Dimakopoulou, Amit Gandhi, Victor Kostyuk, Madhav Kumar, Shyue-Ming Loh, Widad Machmouchi, Jialiang Mao, James McQueen, John Meakin, Konrad Miziolek, Dominique Perrault-Joncas, Vladimir Petrovic, Sven Schmit, James Sorenson, Matthew Wardrop, Michael Zhao, Wenjing Zheng, and Martin Tingley. The future of large-scale experiments and their challenges in the digital era. Nature Human Behaviour, 2026. doi: 10.1038/s41562-026-02582-6.

John J. Horton, Apostolos Filippas, and Benjamin S. Manning. Large language models as simulated

economic agents: What can we learn from Homo silicus? NBER Working Paper 31122, National Bureau of Economic Research, 2023.

Steven R. Howard, Aaditya Ramdas, Jon McAulife, and Jasjeet Sekhon. Time-uniform, nonparametric, nonasymptotic confidence sequences. The Annals of Statistics, 49(2):1055–1080, 2021. doi: 10.1214/20-AOS1991.

Kosuke Imai, Luke Keele, and Dustin Tingley. A general approach to causal mediation analysis. Psychological Methods, 15(4):309–334, 2010. doi: 10.1037/a0020761.

Ramesh Johari, Pete Koomen, Leonid Pekelis, and David Walsh. Always valid inference: Continuous monitoring of A/B tests. Operations Research, 70(3):1806–1821, 2022. doi: 10.1287/opre. 2021.2135.

Garrett A. Johnson, Randall A. Lewis, and Elmar I. Nubbemeyer. Ghost ads: Improving the economics of measuring online ad efectiveness. Journal of Marketing Research, 54(6):867–884, 2017. doi: 10.1509/jmr.15.0297.

Ron Kohavi, Diane Tang, and Ya Xu. Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing. Cambridge University Press, 2020. doi: 10.1017/9781108653985.

Widad Machmouchi and Somit Gupta. How to evaluate LLMs: A complete metric framework, September 2023. Published September 27, 2023.

National Academies of Sciences, Engineering, and Medicine. Data Science for Undergraduates: Opportunities and Options. The National Academies Press, 2018. doi: 10.17226/25104.

Hyerim Park, Khanh Huynh, Malin Eiband, Jeremy Dillmann, Sven Mayer, and Michael Sedlmair. Evaluating generative AI in the lab: Methodological challenges and guidelines. In Proceedings of the 31st International Conference on Intelligent User Interfaces, pages 599–616, 2026. doi: 10.1145/3742413.3789065.

Judea Pearl and Elias Bareinboim. External validity: From do-calculus to transportability across populations. Statistical Science, 29(4):579–595, 2014. doi: 10.1214/14-STS486.

Joel Persson, Mårten Schultzberg, and Sebastian Ankargren. Statistical foundations of LLM-based A/B testing: A surrogacy framework for human causal inference, 2026.

Ross L. Prentice. Surrogate endpoints in clinical trials: Definition and operational criteria. Statistics in Medicine, 8(4):431–440, 1989. doi: 10.1002/sim.4780080407.

Kerry Rodden, Hilary Hutchinson, and Xin Fu. Measuring the user experience on a large scale: User-centered metrics for web applications. In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, pages 2395–2398, 2010. doi: 10.1145/1753326.1753687.

John M. Scanlon, Kristofer D. Kusano, Tom Daniel, Christopher Alderson, Alexander Ogle, and Trent Victor. Waymo simulated driving behavior in reconstructed fatal crashes within an autonomous vehicle operating domain. Accident Analysis & Prevention, 163:106454, 2021. doi: 10.1016/j.aap.2021.106454.

Matthew Schwall, Tom Daniel, Trent Victor, Francesca Favaro, and Henning Hohnhold. Waymo public road safety performance data, 2020.

Shreya Shankar, J. D. Zamfirescu-Pereira, Björn Hartmann, Aditya Parameswaran, and Ian Arawjo. Who validates the validators? aligning LLM-assisted evaluation of LLM outputs with human preferences. In Proceedings of the 37th Annual ACM Symposium on User Interface Software and Technology (UIST ’24), 2024. doi: 10.1145/3654777.3676450.

Judy Hanwen Shen and Alex Tamkin. How AI impacts skill formation, 2026.

Leif Sigerson, Tom Cunningham, Winston Chou, Sana Pandey, Jonathan Stray, Lo-Hua Yuan, Eytan Bakshy, Timothy Chan, Molly Davies, Maria Dimakopoulou, Simon Ejdemyr, Kenneth Hung, Nathan Kallus, Madhav Kumar, Thu Le, A. Demetri Pananos, Lee Richardson, Brennan Schafner, Rose Tan, Martin Tingley, Nadia Tomova, Panagiotis Toulis, Wenjing Zheng, Zander Arnao, and Dean Eckles. Evaluating for the long term: Learnings from industry, 2026.

Olivier Toubia, George Z. Gui, Tianyi Peng, Daniel J. Merlau, Ang Li, and Haozhe Chen. Database report: Twin-2K-500: A data set for building digital twins of over 2,000 people based on their answers to over 500 questions. Marketing Science, 44(6):1446–1455, 2025. doi: 10.1287/mksc. 2025.0262.

Angelina Wang, Jamie Morgenstern, and John P. Dickerson. Large language models that replace human participants can harmfully misportray and flatten identity groups. Nature Machine Intelligence, 7(3):400–411, 2025. doi: 10.1038/s42256-025-00986-z.

Kevin Zielnicki, Guy Aridor, Aurélien Bibaut, Allen Tran, Winston Chou, and Nathan Kallus. The value of personalized recommendations: Evidence from Netflix. International Journal of Industrial Organization, 108:103304, 2026. doi: 10.1016/j.ijindorg.2026.103304.

## A Challenges identified at CIDER 2026

Each CIDER begins with an around-the-table shareout of the top experimentation and causal inference challenges confronting each organization. This year’s discussion surfaced six broad clusters, summarized in Table 1: evolving measurement needs, increasing business complexity, deploying and evaluating data science agents, experimentation velocity, alternatives and complements to the classical A/B test, and evaluation of AI-driven products.

Table 1: Challenge clusters shared across organizations that participated in CIDER 2026
<table><tr><td rowspan=1 colspan=1>Cluster</td><td rowspan=1 colspan=1>Summary</td></tr><tr><td rowspan=1 colspan=1>Measurement</td><td rowspan=1 colspan=1>As products mature, core metrics like retention and Daily Active Users(DAU) are harder to move and user satisfaction is more nuanced as userjourneys get longer. Teams need sensitive proxies that capture user valueand that feature teams can “hill climb&quot; on. Building such metrics is anacute challenge, especially for AI-powered products: for example, longercoding assistant sessions could reflect delight or frustration, hence the needfor metrics that can distinguish between these two scenarios.</td></tr><tr><td rowspan=1 colspan=1>Business complexity</td><td rowspan=1 colspan=1>Experimentation leaders are increasingly asked to help navigate tradeoffsbetween user value, business value, engagement, retention, andinfrastructure costs. Local wins can create global losses. Comparingdisparate outcomes, measuring cumulative effects, and aligning incentivesacross teams all remain difficult. Domain knowledge and product experiencecontinue to be critical for such decisions, and they are increasingly harder tooperationalize for complex products.</td></tr><tr><td rowspan=1 colspan=1>Data science agents</td><td rowspan=1 colspan=1>Agents are both tools and objects of evaluation. They often needinstitutional context—shared metric definitions, default filters, and approvedsources—to perform well (Cherry et al., 2026). The democratization ofexperimentation analysis through AI has empowered teams to understandthe impact of changes faster. At the same time, it amplifies governance riskand the need for certification of “blessed&quot; analyses to avoid untrustworthydecisions based on inaccurate interpretation of data.</td></tr><tr><td rowspan=1 colspan=1>Experimentationvelocity</td><td rowspan=1 colspan=1>AI lowers the cost of generating and testing ideas, shortening the pathbetween “prompt to prototype to production.&quot; Because live user traffic andmanual review capacity remain scarce, filtering ideas becomes critical; AIfunnels need their own validation. Higher throughput increases the value ofhistorical learning: reusable metrics, portfolio-level views, and records ofwhat has already been tried.</td></tr><tr><td rowspan=1 colspan=1>Alternatives andcomplements to classicalA/B testing</td><td rowspan=1 colspan=1>Options include incremental rollouts, holdbacks, observational causalinference, adaptive experiments, bandits, offline simulations, synthetic users,and hybrid systems. The right combination depends on the decision and theevidence each method can provide. This also amplifies the need to buildguardrails and harnesses around such methods as AI makes themincreasingly available.</td></tr><tr><td rowspan=1 colspan=1>Evaluation ofAI-powered products</td><td rowspan=1 colspan=1>AI products can support open-ended tasks and exhibit branching andnondeterministic behavior. This complicates metric selection and thedetection of rare but costly failures. Teams must reconcile offlinebenchmarks, online experiments, qualitative and anecdotal evidence (e.g.,screenshots), and continuous monitoring. Using AI agents for evaluationrequires additional care to avoid feedback loops where AI agents evaluatethemselves.</td></tr></table>