# Hypothesis-Driven Autonomous Materials Synthesis with Multimodal LLM Agents

Izumi Takahara<sup>1</sup>, Kazunori Nishio<sup>2</sup>, Akira Aiba<sup>3</sup>, Shigeru Kobayashi<sup>2</sup>, Takao Nakajima<sup>4</sup>, Taro Hitosugi<sup>2</sup>, and Teruyasu Mizoguchi<sup>1,\*</sup>

<sup>1</sup>Institute of Industrial Science, The University of Tokyo, Tokyo 153-8505, Japan <sup>2</sup>Department of Chemistry, The University of Tokyo, Tokyo 113-0033, Japan <sup>3</sup>Rigaku Corporation, Tokyo 196-8666, Japan <sup>4</sup>MITSUI KNOWLEDGE INDUSTRY CO., LTD., Tokyo 107-0062, Japan Corresponding author: teru@iis.u-tokyo.ac.jp

Self-driving laboratories can explore synthesis conditions autonomously, but their decision-making layer is typically a black-box optimizer, and the output is a set of optimized samples, with the measurements reduced to predefined scalar objectives and the reasons behind success left unarticulated. Here we present SynAgent, a framework in which large language model agents operate an automated experimental system and maintain an explicit, revisable understanding of the synthesis process as the campaign’s primary output. Starting with no predefined analysis pipeline, SynAgent adaptively generates analysis skills for newly acquired data and evolves this understanding through multimodal reasoning over experimental data such as X-ray difraction patterns and electron micrographs. The evolution is guided by a verify–falsify scheme, in which the agent deliberately challenges its own hypotheses by testing conditions predicted to fail as well as those predicted to succeed. In a single campaign of 18 autonomous experiments using $\mathrm { L i C o O } _ { 2 }$ (001) thin-film deposition as a testbed, SynAgent synthesized highly crystalline films and evolved an understanding of how the substrate temperature governs crystallization, discovering an abrupt threshold and a narrow optimal growth window at 650–690 <sup>◦</sup>C. These results extend autonomous experimentation beyond optimized samples to testable, human-readable understanding.

## 1. Introduction

The development of new functional materials underpins technological progress across diverse fields, from energy to information processing [1–3]. In this endeavor, synthesis plays a central role, requiring researchers to navigate a vast space of processing conditions through repeated cycles of fabrication, characterization, and analysis [4]. Because these cycles are slow and labor-intensive, laboratory automation has become a key strategy for accelerating materials research [5]. Self-driving laboratories (SDLs), which close the loop between robotic experimentation and machine-learning-based experiment planning, have demonstrated order-of-magnitude accelerations across diverse domains [6, 7], including photocatalyst formulations [8], functional thin films [9], inorganic powders [10], and semiconductor nanocrystals [11].

In most SDLs, experiment planning is driven by black-box optimization, typically Bayesian optimization (BO), which proposes the next experimental conditions on the basis of the outcomes accumulated so far [8, 9, 11–13]. Black-box optimization is a powerful and well-established strategy for optimizing a predefined objective. When it serves as the decision-making layer of an SDL, however, two limitations arise. First, recent platforms interconnect modular synthesis and measurement instruments and deliver diverse data from every experiment, including difraction patterns, electron micrographs, spectra, and measured physical properties [14], yet the optimizer sees only one or a few hand-crafted scalar metrics, leaving most of these data unexploited. Second, black-box optimizers remain largely opaque to researchers, as they can neither explain why they propose a particular condition nor articulate what the accumulated measurements have established about the process and what remains unknown. Indeed, it has been argued that laboratory automation has so far accelerated only the testing of hypotheses, while their generation and refinement remain outside the loop [15].

In parallel, the rapid advance of large language models (LLMs) opens a complementary possibility. Unlike conventional optimizers, LLMs can articulate their reasoning in natural language, opening their decision-making to human inspection. LLMs encode substantial knowledge of inorganic synthesis. Fine tuned LLMs predict synthesizability and precursors at levels comparable to task-specific machine-learning models [16, 17], and even of-the-shelf LLMs recall precursors and processing conditions reported in the literature, in some tasks matching specialized models [18]. Modern LLMs are also multimodal, accepting images and other modalities alongside text, which allows them to interpret heterogeneous experimental data directly. Moreover, the growing reasoning and tool-use capabilities of LLMs have made it possible to deploy them as autonomous agents [19, 20]. Exploiting these agentic capabilities, LLM agents have accelerated materials design in silico [21–24]. They are now beginning to enter physical laboratories as well, where they have assisted tasks ranging from the planning and execution of syntheses to the operation of instruments and the orchestration of research workflows [25–31], and have most recently steered closed-loop solid-state synthesis campaigns [32].

These advances suggest that LLM agents can serve not merely as optimizers but as participants in the experimental process itself. Serving as such a participant, however, requires capabilities that current frameworks do not yet provide. Many agentic frameworks have been validated only against simulated experiments or computational tasks [33–35], and even the agents that have reached physical laboratories remain largely passive, operating with predefined toolsets and a single data modality. The picture of the system that they build up over a campaign also remains implicit rather than being maintained as an explicit, testable account. Moreover, LLM reasoning is prone to biases, such as anchoring on early exploration choices [33] and overtrust of prior knowledge and plausible-sounding hypotheses [34], so that the agent’s initially incomplete beliefs about the system may persist unrevised throughout a campaign.

Here we present SynAgent, a multimodal, multi-agent framework in which a main agent plans experiments and interprets their outcomes, supported by subagents that generate analysis skills (Fig. 1). Harnessing the scientific knowledge and reasoning capabilities of LLMs, SynAgent actively tests its own hypotheses through autonomous experiments. To address the challenges outlined above, SynAgent combines two capabilities. First, adaptive skill generation: a dedicated subagent generates analysis skills on demand for newly acquired experimental data and registers them for reuse, freeing the framework from predefined analysis pipelines and integrating the diverse data produced during a campaign into the agent’s reasoning. Second, evolving understanding: the agent generates hypotheses about the relationship between synthesis conditions and outcomes, tests them experimentally, and revises them accordingly. This evolution is driven by a verify– falsify reasoning scheme, in which the agent deliberately tests not only conditions predicted to succeed but also conditions predicted to fail, counteracting the confirmation bias that would otherwise leave incomplete beliefs unchallenged. It is further supported by multimodal reasoning, which integrates X-ray difraction (XRD) patterns and scanning electron microscopy (SEM) images as complementary evidence. Every decision is accompanied by explicit reasoning that human researchers can trace, interpret, and learn from.

We demonstrate SynAgent on a physical automated laboratory [14] for the synthesis of highly crystalline $\mathrm { L i C o O } _ { 2 }$ (001) thin films, a prototypical layered cathode material [36] whose cation ordering depends sensitively on deposition conditions [37, 38]. This well-characterized dependence makes the system an ideal testbed, allowing us to assess whether the agent can acquire an understanding of the relationship through its own autonomous experiments. We show that SynAgent autonomously generates the analysis skills required for the data at hand and progressively evolves its understanding of the deposition process. It delivers not only promising synthesis conditions but also a clear statement of how substrate temperature governs crystallization, namely an abrupt threshold and a narrow window in which well-ordered films form, written so that a researcher can read it, trace it back to the experiments behind it, and test it further.

## 2. Results

## SynAgent framework

Figure 1 presents the overall architecture of SynAgent, which couples the scientific knowledge, multimodal reasoning, and code-generation capabilities of LLMs with autonomous experimentation to continually update its understanding of the target system and thereby propose increasingly promising synthesis conditions. SynAgent runs a synthesis campaign as an iterative four-step loop (Fig. 1a): the agent proposes synthesis conditions on the basis of its current understanding (Step 1), synthesis and measurement are executed autonomously under the proposed conditions (Step 2), the acquired data are analyzed with skills, that is, analysis code generated by SynAgent itself (Step 3), and the agent reflects on the outcome and updates its understanding (Step 4).

SynAgent incorporates two mechanisms into this loop. The first, adaptive skill generation (Fig. 1b), frees the framework from the predefined toolsets that constrain current laboratory agents. SynAgent maintains a skill library, a growing collection of the skills generated during the campaign. At the start of a campaign, this library is empty, and no analysis capability is predefined. When newly acquired data call for an analysis that no registered skill covers, the main agent delegates the creation of a new skill to a dedicated subagent. The subagent designs the analysis code around the actual measurement data and registers the validated skill in the library. The registered skills are then reused by the main agent, both when a new measurement arrives and during its reflection on the outcome. The analysis repertoire therefore grows over the course of the campaign rather than being fixed at design time, allowing heterogeneous data such as XRD patterns and SEM images to enter the decision loop as complementary evidence.

The second, the verify–falsify scheme that drives the evolution of understanding (Fig. 1c), follows from the demand that the agent produce an understanding rather than only an optimized sample. If the goal were only optimization, there would be little reason to test conditions predicted to fail. But an understanding is a claim about where the process succeeds and where it fails, and a claim of that kind is confirmed only when its predicted failures are observed. The agent therefore tests conditions it expects to fail as well as those it expects to succeed. Equally important, the same discipline counteracts the susceptibility of LLM reasoning to biases such as anchoring and overconfidence in prior knowledge.

SynAgent maintains its understanding of the target system as an explicit natural-language document, a set of claims about how the synthesis conditions govern the outcome, and updates this document at every cycle of the loop. In each iteration, the agent derives from the current understanding either a verify proposal, targeting conditions under which synthesis of the desired material is predicted to succeed, or a falsify proposal, targeting conditions under which it is predicted to fail, committing in both cases to the hypothesis being tested and to a concrete prediction of the outcome.

a  
b  
![](images/e2fe3a0e1161f348404b953612e86ff82300e09a6825207ea005ff72bc2b3670.jpg)  
Figure 1: Overview of the SynAgent framework. a, The four-step autonomous loop of experiment proposal (Step 1), synthesis and measurement (Step 2), skill-based data analysis (Step 3), and reflection with understanding update (Step 4), where all reasoning is performed by multimodal LLM agents. b, Adaptive skill generation: when no suitable skill exists in the skill library, the main agent delegates to a subagent that creates and registers a new one, and the registered skills are reused in later iterations. c, Evolving understanding through verify–falsify reasoning: each proposal is designed to either verify or falsify the current understanding, and the reflection on the experimental outcome reinforces or revises it, progressively evolving the understanding over the course of the campaign.

After the autonomous experiment and analysis, the agent reflects on whether the outcome matched the committed prediction. This reflection is multimodal, as the agent receives the acquired images themselves as vision input and reasons over them alongside the measured target and the descriptors extracted by the skills. A supported proposal reinforces the underlying claim, whereas a refuted proposal leads the agent to revise the claims now in doubt through abductive reasoning, inferring the most plausible explanation for the unexpected outcome. By deliberately seeking evidence against its own claims, the agent prevents an initially incomplete picture of the system from persisting uncorrected, and the progressively refined understanding in turn becomes the basis for the subsequent proposals.

## Synthesis of $\mathbf { L i C o O } _ { 2 }$ (001) thin films

To demonstrate the framework, we applied SynAgent to the synthesis of highly crystalline $\mathrm { L i C o O } _ { 2 }$ (001) thin films on a physical automated laboratory [14], with all agents driven by GPT-5.5, a multimodal LLM developed by OpenAI [39]. In this system, high crystallinity refers to a well-ordered layered structure, in which the Li and Co ions occupy alternating cation layers. We considered the growth of $\mathrm { L i C o O } _ { 2 }$ (001) on $\mathbf { A l } _ { 2 } \mathbf { O } _ { 3 }$ (0001) substrates, in which the crystallinity is controlled by the substrate temperature. The agent was asked to use the intensity ratio of the 003 and 006 reflections in the XRD pattern, �(003)/�(006), as a measure of the crystallinity of $\mathrm { L i C o O } _ { 2 }$ [14, 38], and to propose substrate temperatures that maximize this ratio. The agent was therefore required to observe how the proposed temperatures relate to the crystallinity of the resulting films and to evolve its understanding of this relationship over the course of the campaign. In this campaign, the agent alternated between verify and falsify proposals at successive iterations, beginning with a verify proposal (Fig. 1c).

Figure 2 summarizes the behavior of SynAgent in this campaign, showing the analysis skills it generated and the outcomes of all the experiments analyzed by these skills. Figure 2a shows the analysis skills that the agent generated from the data measured in the first experiment, for which it proposed a substrate temperature of $6 0 0 ^ { \circ } \mathrm { C } .$ . To compute the objective �(003)/�(006) from the XRD pattern, the agent generated an analysis skill that locates the 003 and 006 reflections and extracts their intensity ratio. During the reflection on this experiment, the agent recognized bright particles dispersed on the film surface in the SEM image, and requested another skill tailored to this specific morphology, quantifying the bright-particle area fraction, the particle count, and the equivalent particle diameter, together with statistics of the background texture. These descriptors served as complementary evidence for interpreting the synthesis outcomes, and the values they returned agreed with manual analysis of the same data.

Figure 2b summarizes the measured �(003)/�(006) of all the experiments as a function of the proposed substrate temperature, together with the agent’s predictions. Each experiment is classified by the mode of the proposal, verify or falsify, and by whether the measured ratio supported or refuted the prediction committed to at the proposal stage. Over the campaign, the agent proposed and executed 18 experiments, of which 15 predictions were judged as supported and 3 as refuted (Supplementary Table 1). The measured ratios reveal a sharply structured temperature dependence. The films grown at $2 5 0 – 6 3 0 \mathrm { ~ } ^ { \circ } \mathrm { C }$ showed essentially no ordering, with ratios below 0.1, whereas ordered growth set in abruptly at $6 5 0 ^ { \circ } \mathrm { C }$ , with a narrow high-quality window at $6 5 0 – 6 9 0 \ ^ { \circ } \mathrm { C }$ and a plateau of ratios of 32–41 at ${ 7 0 0 { - } 7 5 0 \ ^ { \circ } } \mathrm { C }$ . The highest crystallinity of the campaign, $I ( 0 0 3 ) / I ( 0 0 6 ) = 6 1 . 6$ , was obtained at $6 7 0 ~ ^ { \circ } \mathrm { C } .$ . The surface morphology also varied markedly with the temperature, as illustrated by the SEM images of the films grown at 400, 670, and $7 3 0 ^ { \circ } \mathrm { C }$ (insets in Fig. 2b).

XRD spectrum  
![](images/cbd8040d56f6154c1d767d3d46ebb555a269ccd9118dbe77bea97f65e37c746a.jpg)

a  
![](images/72058b33fd049d351a062e929d20ff085fa52b695e25b1697bc3356dac80ba34.jpg)  
b

![](images/5be727b2ef2ef787e7d8ddcd66be3521713f49f2b56c3e56a6e499877532966d.jpg)

![](images/bde2c8d0be8fbd138878ae4593bcf993916fda875ed1d4b5cc6b46e4c0828bff.jpg)

C  
![](images/2f594b017e8fd7c6d79b6547cb40633dafe56e6b0373c812f606808ba2120dd3.jpg)

![](images/0c397025d0d008538c71639f55587571907e26b9a8688ec975ddcb00ed07c607.jpg)

![](images/8bf736a3d8c6c7c4450b9de68cbee35230a699f28b686832580a7a6f5ee542f7.jpg)  
Figure 2: Behavior of SynAgent in the $\mathbf { L i C o O } _ { 2 }$ synthesis campaign. a, Analysis skills generated by the agent from the data of the first experiment $( 6 0 0 ~ ^ { \circ } \mathrm { C } )$ . For the X-ray difraction (XRD) pattern, the agent generated a skill (lco\_peak\_ratio\_xrd) that searches for the $\mathrm { L i C o O } _ { 2 }$ 003 and 006 reflections within windows around their reference positions (pink bands) and computes their intensity ratio, �(003)/�(006). Pink triangles mark the maxima selected by the skill. For the scanning electron microscopy (SEM) image, the agent identified bright particles scattered over the film surface, features visible to the eye but not part of any predefined analysis, and generated a skill (lco\_bright\_particles\_sem) that quantifies the surface morphology, including the bright-particle area fraction, the particle count, and the equivalent particle diameter. b, �(003)/�(006) as a function of the proposed substrate temperature. Numbers denote iterations, blue circles denote verify proposals, and red diamonds denote falsify proposals. Filled and open symbols indicate that the prediction was supported and refuted, respectively. Horizontal bars indicate the ratios predicted by the agent, and are connected to the corresponding measured values. The vertical dashed line marks $6 7 0 ~ ^ { \circ } \mathrm { C }$ (iteration 13), which yielded the highest ratio. Insets show SEM images of the films grown at 400, 670, and $7 3 0 \ { } ^ { \circ } { \bf C } . \ { \bf c } ,$ The generated SEM skill returns three morphological descriptors, namely the bright-particle area fraction, the particle count, and the mean equivalent particle diameter (in pixels, with 5 nm per pixel), as functions of the proposed substrate temperature. Iterations with no detected particles are omitted from the diameter panel.

Figure 2c shows the morphological descriptors extracted from the SEM images, which allowed the agent to distinguish between films that are indistinguishable by XRD alone. Both regimes below $6 5 0 ~ ^ { \circ } \mathrm { C }$ show ratios below 0.1 and are identical as far as the objective is concerned. The films grown at $2 5 0 { - } 3 2 0 ~ ^ { \circ } \mathrm { C }$ showed nearly featureless surfaces at the imaging resolution, with bright-particle area fractions of about 0.1%, which the agent tentatively interpreted as continuous films formed under mobility-limited conditions. In contrast, the films grown at $4 0 0 { - } 6 3 0 \ ^ { \circ } \mathrm { C }$ contained bright particles covering $4 { - } 6 \%$ of the surface. The agent interpreted these features as possible signatures of incomplete crystallization or surface segregation, although their composition and origin could not be determined from the SEM images alone. On the hightemperature plateau, the SEM images showed continuous films with roughened or faceted textures but few visible segregated particles, which the agent used to attribute the lower ratios of this region to subtler causes such as defects, texture, strain, or mild stoichiometry shifts, while noting that the composition of the faceted features would require additional characterization to determine.

In proposing each substrate temperature, the agent drew on all of this evidence, namely the measured $I ( 0 0 3 ) / I ( 0 0 6 )$ , the SEM images themselves, and the morphological descriptors extracted from them. The raw images also contributed observations that were not captured by the numerical descriptors, such as the faceted shapes of the bright surface features. These observations informed the agent’s tentative hypotheses about possible growth and failure modes, although they did not establish the composition or phase identity of the observed features. All of these analyses were performed with the two skills generated at the first iteration, which were reused without modification throughout the campaign. Taken together, these results demonstrate that SynAgent, integrating the XRD objective with the complementary SEM evidence into an understanding evolved from its own observations, narrowed the search to the $6 5 0 – 6 9 0 \ ^ { \circ } \mathrm { C }$ window and located the best condition at $6 7 0 ~ ^ { \circ } \mathrm { C }$ in a single autonomous campaign, while generating testable hypotheses, in its own words, about why the films outside that window exhibited lower ordering.

## Evolving understanding through verify–falsify reasoning

Figure 3 traces how this picture was established, following the campaign iteration by iteration. The top panel shows the best �(003)/�(006) obtained up to each iteration, and the bottom panel shows the proposed substrate temperatures overlaid on the temperature regions stated in the agent’s evolving understanding, which culminated in a partition of the temperature axis into four regions. The full text of the understanding at the key stages of the campaign is reproduced in Supplementary Note 1. The first verify proposal tested the prior expectation that $6 0 0 ~ ^ { \circ } \mathrm { C }$ would be near optimal, and failed with $I ( 0 0 3 ) / I ( 0 0 6 ) = 0 . 0 0 1$ (Supplementary Note 2). Reflecting on the difraction pattern and the particle-covered surface observed by SEM, the agent hypothesized that the failure might be associated with incomplete crystallization, morphological inhomogeneity, or possible secondary-phase segregation, and revised its understanding to place the optimum below $6 0 0 ^ { \circ } \mathrm { C } .$ The falsify proposal of the next iteration probed $7 0 0 ^ { \circ } \mathrm { C }$ as a condition predicted to fail under this revised picture, but instead yielded a well-ordered film with a ratio of 35.6. This single refutation overturned the revised picture, and the agent reorganized its understanding into a temperature-activated crystallization threshold, above which ordering sets in abruptly, and placed the transition somewhere between 600 and $7 0 0 ~ ^ { \circ } \mathrm { C } .$ The subsequent verify proposals extended the newly found ordered growth from $7 0 0 ~ ^ { \circ } \mathrm { C }$ up to $7 4 0 ^ { \circ } \mathrm { C } .$ , establishing a plateau of similar ratios, while the interleaved falsify proposals at $2 5 0 { - } 5 0 0 ~ ^ { \circ } \mathrm { C }$ confirmed the absence of ordering at low temperatures.

The decisive step came at iteration 10. The falsify proposal probed $6 5 0 ^ { \circ } \mathrm { C } ,$ the unsampled gap between the failed $6 0 0 ~ ^ { \circ } \mathrm { C }$ film and the successful $7 0 0 ~ ^ { \circ } \mathrm { C }$ film, with a predicted ratio of 0.2. The measured ratio of 54.9 was instead the best obtained so far, refuting the claim that the crystallization threshold lay close to $7 0 0 ~ ^ { \circ } \mathrm { C }$ (Supplementary Note 2). The agent narrowed the transition interval to $6 0 0 { - } 6 5 0 \ { ^ { \circ } } \mathrm { C } ,$ , and the following verify proposals rapidly refined the newly opened window: $6 6 0 ~ ^ { \circ } \mathrm { C }$ gave 60.2 and $6 7 0 ~ ^ { \circ } \mathrm { C }$ gave 61.6, the highest crystallinity of the campaign, while 680 and $6 9 0 ~ ^ { \circ } \mathrm { C }$ gave 55.0 and 58.2. The remaining falsify proposals sharpened the boundaries of this picture. The film grown at $7 5 0 ~ ^ { \circ } \mathrm { C }$ gave 40.9, supporting the claim that the high-temperature plateau is robust but suboptimal, and the films grown at 620 and $6 3 0 ^ { \circ } \mathrm { C }$ failed completely, bracketing the abrupt transition between 630 and $6 5 0 ^ { \circ } \mathrm { C } .$ The SEM image of the $6 3 0 ~ ^ { \circ } \mathrm { C }$ film still showed abundant faceted surface features, which the agent regarded as being consistent with a morphologically inhomogeneous pre-threshold growth regime. However, SEM alone could not determine whether these features represented secondary phases, compositional segregation, or $\mathrm { L i C o O } _ { 2 }$ crystallites with a distinct morphology. By the end of the campaign, the agent had thus bracketed the transition to a $2 0 ~ ^ { \circ } \mathrm { C }$ interval, and resolving it further would require a finer temperature grid.

![](images/9b2555dee815b92ccb2e37c3c35db9eb32dc06b20673be3fb95e5089ef703086.jpg)  
Figure 3: Evolution of the agent’s understanding over the campaign. Top, the best �(003)/�(006) ratio obtained up to each iteration. Gray triangles mark the iterations at which the agent’s prediction was refuted (iterations 1, 2, and 10). Bottom, the substrate temperatures proposed at each iteration, shown as blue circles for verify proposals and red diamonds for falsify proposals, overlaid on the temperature regions stated in the agent’s evolving understanding. Each region extends from the iteration after which it was stated until it was revised by a later experiment, and the shading before iteration 1 schematically represents the prior expectation that $6 0 0 ~ ^ { \circ } \mathrm { C }$ would be near optimal. The dashed line traces the substrate temperature that the agent regarded as best at each stage.

Each of the three refuted predictions triggered a major revision of the understanding. The final understanding delineated the four regions shown in Fig. 3: a no-ordering region below $6 3 0 ^ { \circ } \mathrm { C } .$ , a transition interval between 630 and $6 5 0 ~ ^ { \circ } \mathrm { C } .$ , an ordered-growth window at $6 5 0 – 6 9 0 \ ^ { \circ } \mathrm { C }$ with the optimum near $6 7 0 ^ { \circ } \mathrm { C }$ , and a robust but suboptimal ordered region at ${ 7 0 0 { - } 7 5 0 ~ ^ { \circ } C }$

These results illustrate how the verify–falsify scheme drove the campaign. The preconception inherited from prior knowledge, that $6 0 0 ~ ^ { \circ } \mathrm { C }$ would be near optimal, was overturned at the first iteration, when the prediction committed by the verify proposal was refuted. The two largest advances that followed, the discovery of ordered growth at $7 0 0 ~ ^ { \circ } \mathrm { C }$ and of the superior window at $6 5 0 ~ ^ { \circ } \mathrm { C } .$ , were both made by falsify proposals. This is not incidental. Proposals whose outcomes match their predictions refine the current picture, as when the falsify proposals at 620 and $6 3 0 ~ ^ { \circ } \mathrm { C }$ bracketed the transition. It is when a predicted failure does not occur that the agent learns something it could not have anticipated, and the experiment is informative in proportion to the confidence with which the failure was predicted. Each refutation was converted into a revision of the understanding. The verify–falsify scheme thus corrects the preconceptions that the LLM brings into a campaign and drives its understanding toward a faithful picture of the system, which in this case guided the agent to the optimal growth window.

## 3. Discussion

In this work, we proposed SynAgent, a multi-agent framework that connects LLM agents to an automated experimental system and drives autonomous materials synthesis campaigns while generating its own analysis tools and actively testing its own hypotheses. Demonstrated on a physical automated laboratory, SynAgent synthesized and characterized $1 8 \mathrm { L i C o O } _ { 2 }$ (001) films without human intervention, wrote and registered the two analysis skills it needed at the first iteration, and evolved an explicit, human-readable understanding of the deposition process that culminated in the four-region map of Fig. 3 and in the optimal growth window at $6 5 0 – 6 9 0 \ ^ { \circ } \mathrm { C }$

The product of the campaign is therefore twofold. Beyond the synthesis conditions themselves, SynAgent delivers an interpretable account of the process, a set of testable mechanistic hypotheses that state where ordered growth occurs and why, each linked to the experiments that support or constrain it. In this respect the framework complements rather than replaces BO. The $\mathrm { L i C o O } _ { 2 }$ task, the temperature grid, and the objective were the same as those of the BO-driven campaign reported previously on this platform [14], and SynAgent likewise arrived at the optimal condition. The diference lies in what remains at the end of the campaign: BO leaves the optimum together with the data that produced it, whereas SynAgent additionally leaves a document that partitions the temperature axis into four regions, locates an abrupt threshold between 630 and $6 5 0 ~ ^ { \circ } \mathrm { C } ,$ and states that the $7 0 0 { - } 7 5 0 \ ^ { \circ } \mathrm { C }$ plateau is robust but suboptimal, with each claim tied to the experiments that supported or refuted it. This account, however, was possible only because the agent could decide what features to extract and quantify from the acquired measurements. The bright particles on the low-temperature films were not part of any predefined analysis pipeline. The agent saw them in the SEM image, judged them informative, and obtained the means to quantify them. An agent restricted to a predefined toolset can reason only over quantities that someone anticipated, and an unanticipated feature is precisely what an evolving understanding needs. The two skills generated at the first iteration were reused unchanged for the remaining 17 experiments, indicating that what the agent produced were not ad hoc scripts but analysis tools. The multimodal evidence proved essential to this account, as the SEM-derived descriptors separated the featureless low-temperature films from the particle-covered ones, a distinction invisible to the scalar objective, and thereby shaped the mechanistic hypotheses that the agent subsequently refined and tested.

The campaign also illustrates why deliberate falsification matters for LLM-driven experimentation. LLM agents are prone to overtrust their prior knowledge and to anchor to their own early successes, failure modes reported for simulated campaigns [33, 34]. An agent rewarded only for confirming its expectations would have had little reason to revisit the region between the failed $6 0 0 ^ { \circ } \mathrm { C }$ film and the successful high-temperature plateau. In our campaign, the falsify proposals were precisely what forced such revisits, and their refutations produced the two largest advances. By committing to a concrete prediction before each experiment and revising its claims through abduction when the prediction failed, the agent replaced an initially biased picture with an understanding grounded in the observed data, converting surprises into structured updates.

Several avenues remain for extending the framework. The present demonstration involved a single controllable variable and a single materials system, and applying the same scheme to higher-dimensional condition spaces, where concise and testable understanding documents would be even more valuable, is a natural next step. In this campaign, the verify and falsify modes were simply alternated. Exploring more adaptive switching strategies, for example letting the agent choose the mode according to the maturity of its current understanding, may make the evolution of understanding still more eficient. Each condition was, moreover, visited only once, and the small diferences among the best ratios in the $6 5 0 – 6 9 0 \ ^ { \circ } \mathrm { C }$ window may lie within run-to-run variation. Repeated experiments would quantify this variation and further test the claims of the understanding. A controlled comparison with BO would likewise require re-running both decision layers under matched budgets, which the sequential nature of the physical experiments precluded here. Finally, the correctness of the generated skills is currently verified only within the scope of their self-tests, and the judgments of the campaign, from the supported and refuted verdicts to the abductive explanations, rest on the reasoning of the LLM itself. Since prior knowledge can enter a campaign through the design of its analysis tools, human-in-the-loop or agent-in-the-loop approaches, in which the generated skills and the recorded judgments are reviewed by human researchers or by specialized validation agents, would be an efective complement.

In conclusion, these results position LLM agents not merely as optimizers of laboratory workflows but as experimental partners that decide how newly acquired measurements should be analyzed, and build, articulate, and stress-test scientific understanding. As self-driving laboratories continue to standardize and interconnect instruments [14], such agents will be able to draw on ever richer characterization modalities and to carry the understanding gained in one campaign into the next. We anticipate that autonomous experiments of this kind will deliver not only better materials but also a deeper understanding of the underlying materials science.

## 4. Methods

## SynAgent framework

SynAgent is implemented as a set of LLM agents built on the OpenAI Agents SDK [40], with OpenAI GPT-5.5 [39] as the underlying model for all agents. A campaign is declared in a configuration file that specifies an overview of the experiment, the target variable, the controllable variables with their allowed grids, and the fixed experimental conditions. Each iteration consists ofthree LLM turns around the autonomous experiment. In the proposal turn, the agent receives the campaign declaration, the observed pairs of conditions and target values, and its current understanding of the system, which is maintained as a natural-language document denoted <understanding>, and returns, under the mode assigned by the schedule, a structured proposal consisting of the hypothesis under test, the proposed condition chosen from the allowed grid, a concrete predicted target value, and the reasoning behind them. After the proposed experiment has been executed and the measurement analyzed, the reflection turn judges the outcome, and the understanding turn rewrites <understanding>. All turns run inside a single persistent session stored on disk, so that the agent retains the conversational context of the entire campaign. The SEM images of the three most recent iterations are replayed to the model as vision input, and older images are replaced by the textual morphology descriptions that the agent itself recorded at the corresponding iterations. Every prompt and response is logged per iteration, and all proposals, measured targets, skill outputs, and judgments are appended to a persistent history file, making each decision traceable.

## Adaptive skill generation

A skill consists of a metadata document (SKILL.md) and an analysis module (analyze.py). The metadata describe the technique, material, instrument, metric, and applicability of the skill, and the module implements a single function that reads a measurement file in the standardized MaiML format [14, 41] and returns a structured analysis result. When a measurement arrives, a dispatcher presents the metadata of the registered skills to the LLM, which either selects a matching skill or signals that a new one is needed. A new skill is generated by a dedicated agent equipped with a code-execution tool. The generator agent inspects the actua MaiML file and the available reference files, and iteratively writes and runs candidate analysis code against the measured data within a fixed tool-call budget. When the MaiML file references an image, the image is also attached to the generator as vision input, so that the analysis is designed for the features actually present in the data. Before registration, the generated module is executed on the measurement at hand, and the skil enters the library only when this self-test passes. On failure, the error and the previous code are fed back for regeneration, with up to three attempts. During reflection, the main agent can execute any registered skill on the measurements of the current iteration through a use\_skill tool and can request at most one new skill per iteration through a create\_skill tool, specifying what to quantify and from which measurement. The prompt instructs the agent to prefer existing skills and to create a new one only when no registered skill covers the needed analysis. The skill that computes the optimization target is pinned by name in the campaign configuration. It is generated once, when the first measurement arrives, and its definition remains fixed for the rest of the campaign. In the $\mathrm { L i C o O } _ { 2 }$ campaign, the XRD skill was generated through this route, the SEM skill was created through a create\_skill request during the first reflection, and no further skills were created in the remaining iterations.

## Verify–falsify reasoning scheme

The proposal mode alternates between verify and falsify, starting from verify, and is implemented as an interchangeable strategy so that other schedules can be substituted. In each proposal turn, the agent identifies the single claim of <understanding> to be tested, chooses a condition that tests this claim under the current mode, and commits to a concrete numerical prediction of the target. In the reflection turn, the agent receives the measured target value and the SEM image and judges the prediction as supported or refuted following a mode-dependent procedure. In verify mode, the prediction is judged as supported when the measurement and the prediction both indicate success, and in falsify mode, when both indicate failure. No numerical tolerance is imposed on the diference between the predicted and measured values. The numerical prediction serves as a concrete commitment, and the agent itself judges whether the outcome falls into the predicted category. Every judgment is accompanied by a written reasoning that compares the measured and predicted values in the light of the hypothesis and the SEM observation. When the prediction is supported, the agent states which claim of <understanding> is now more strongly backed by the experiment, without extrapolating beyond the tested regime. When the prediction is refuted, the agent performs abduction, naming the claim now in doubt and proposing the most plausible alternative explanation for the observed outcome. The understanding turn then rewrites <understanding> to incorporate the judgment, and the updated document becomes the basis of the next proposal.

## Autonomous synthesis and characterization

All experiments were performed on dLab, a digital laboratory that physically interconnects modular synthesis and measurement instruments [14]. The campaign task, including the temperature grid and the objective, follows the autonomous demonstration reported previously on the same platform, in which the next condition was selected by BO. In the present work, this decision-making layer is replaced by SynAgent. (001)-oriented $\mathrm { L i C o O } _ { 2 }$ thin films were grown on $\mathbf { A l } _ { 2 } \mathbf { O } _ { 3 }$ (0001) substrates by RF magnetron sputtering using a sintered $\mathrm { L i } _ { 1 . 2 } \mathrm { C o O } _ { x }$ target. The RF power was fixed at 80 W, the deposition time at 1.5 h, and the total gas pressure at 0.50 Pa, with argon and oxygen partial pressures of 0.45 and 0.05 Pa. The substrate temperature during deposition was the only controllable variable and was chosen by the agent from 200 to $7 5 0 ~ ^ { \circ } \mathrm { C }$ in steps of $1 0 ~ ^ { \circ } \mathrm { C }$ . The campaign was run for a preset budget of 18 iterations. XRD patterns were measured with a Rigaku SmartLab XE difractometer over a 2� range of $1 0 { - } 1 2 0 ^ { \circ }$ with a step of $0 . 0 1 ^ { \circ }$ . Surface morphologies were observed with a JEOL JSM-IT700HR scanning electron microscope at a pixel size of 5.0 nm. All measurement data were delivered to the agent as MaiML files, and the SEM data associated with each XRD measurement were linked automatically.

## Acknowledgments

This work was supported by JST ACT-X (grant no. JPMJAX24DB), JST BOOST (grant no. JPMJBS2418), JST CREST (grant no. JPMJCR2204), the MEXT Data-Driven Materials Research and Development Project (grant no. JPMP1122712807) and JSPS KAKENHI (grant no. 24K01599).

## Data and Code Availability

The data and code for this study are available on request to the corresponding author.

## Declaration of Generative AI and AI-Assisted Technologies

During the preparation of this work, the authors used Claude Code (Anthropic) and ChatGPT (OpenAI) to assist in developing the software used in the experiments and in preparing the manuscript. The authors reviewed and edited the output as needed and take full responsibility for the content of the published article.

## References

[1] Chi Chen, Yunxing Zuo, Weike Ye, Xiangguo Li, Zhi Deng, and Shyue Ping Ong. A critical review of machine learning of energy materials. Advanced Energy Materials, 10(8):1903242, 2020.

[2] Takashi Toyao, Zen Maeno, Satoru Takakusagi, Takashi Kamachi, Ichigaku Takigawa, and Ken-ichi Shimizu. Machine learning for catalysis informatics: recent applications and prospects. ACS Catalysis, 10(3):2260–2297, 2020.

[3] Amil Merchant, Simon Batzner, Samuel S. Schoenholz, Muratahan Aykol, Gowoon Cheon, and Ekin Dogus Cubuk. Scaling deep learning for materials discovery. Nature, 624(7990):80–85, 2023.

[4] Hanchen Wang, Tianfan Fu, Yuanqi Du, Wenhao Gao, Kexin Huang, Ziming Liu, Payal Chandak, Shengchao Liu, Peter Van Katwyk, Andreea Deac, Anima Anandkumar, Karianne Bergen, Carla P. Gomes, Shirley Ho, Pushmeet Kohli, Joan Lasenby, Jure Leskovec, Tie-Yan Liu, Arjun Manrai, Debora Marks, Bharath Ramsundar, Le Song, Jimeng Sun, Jian Tang, Petar Veličković, Max Welling, Linfeng Zhang, Connor W. Coley, Yoshua Bengio, and Marinka Zitnik. Scientific discovery in the age of artificial intelligence. Nature, 620(7972):47–60, 2023.

[5] Andrew I. Cooper, Patrick Courtney, Kourosh Darvish, Moritz Eckhof, Hatem Fakhruldeen, Andrea Gabrielli, Animesh Garg, Sami Haddadin, Kanako Harada, Jason Hein, Maria Hübner, Dennis Knobbe,

Gabriella Pizzuto, Florian Shkurti, Ruja Shrestha, Kerstin Thurow, Rafael Vescovi, Birgit Vogel-Heuser, Ádám Wolf, Naruki Yoshikawa, Yan Zeng, Zhengxue Zhou, and Henning Zwirnmann. Accelerating discovery in natural science laboratories with AI and robotics: Perspectives and challenges. Science Robotics, 10(106):eadv7932, 2025.

[6] Milad Abolhasani and Eugenia Kumacheva. The rise of self-driving labs in chemical and materials sciences. Nature Synthesis, 2(6):483–492, 2023.

[7] Gary Tom, Stefan P. Schmid, Sterling G. Baird, Yang Cao, Kourosh Darvish, Han Hao, Stanley Lo, Sergio Pablo-García, Ella M. Rajaonson, Marta Skreta, Naruki Yoshikawa, Samantha Corapi, Gun Deniz Akkoc, Felix Strieth-Kalthof, Martin Seifrid, and Alán Aspuru-Guzik. Self-driving laboratories for chemistry and materials science. Chemical Reviews, 124(16):9633–9732, 2024.

[8] Benjamin Burger, Phillip M. Mafettone, Vladimir V. Gusev, Catherine M. Aitchison, Yang Bai, Xiaoyan Wang, Xiaobo Li, Ben M. Alston, Buyi Li, Rob Clowes, Nicola Rankin, Brandon Harris, Reiner Sebastian Sprick, and Andrew I. Cooper. A mobile robotic chemist. Nature, 583(7815):237– 241, 2020.

[9] Benjamin P. MacLeod, Fraser G. L. Parlane, Thomas D. Morrissey, Florian Häse, Loïc M. Roch, Kevan E. Dettelbach, Raphaell Moreira, Lars P. E. Yunker, Michael B. Rooney, Joseph R. Deeth, Veronica Lai, Gordon J. Ng, Henry Situ, Ryan H. Zhang, Michael S. Elliott, Ted H. Haley, David J. Dvorak, Alán Aspuru-Guzik, Jason E. Hein, and Curtis P. Berlinguette. Self-driving laboratory for accelerated discovery of thin-film materials. Science Advances, 6(20):eaaz8867, 2020.

[10] Nathan J. Szymanski, Bernardus Rendy, Yuxing Fei, Rishi E. Kumar, Tanjin He, David Milsted, Matthew J. McDermott, Max Gallant, Ekin Dogus Cubuk, Amil Merchant, Haegyeom Kim, Anubhav Jain, Christopher J. Bartel, Kristin Persson, Yan Zeng, and Gerbrand Ceder. An autonomous laboratory for the accelerated synthesis of inorganic materials. Nature, 624(7990):86–91, 2023.

[11] Jinge Xu, Christopher H. J. Moran, Arup Ghorai, Fazel Bateni, Jefrey A. Bennett, Nikolai Mukhin, Koray Latif, Andrew Cahn, Pragyan Jha, Fernando Delgado Licona, Sina Sadeghi, Lior Politi, and Milad Abolhasani. Autonomous multi-robot synthesis and optimization of metal halide perovskite nanocrystals. Nature Communications, 16:7841, 2025.

[12] Ryota Shimizu, Shigeru Kobayashi, Yuki Watanabe, Yasunobu Ando, and Taro Hitosugi. Autonomous materials synthesis by machine learning and robotics. APL Materials, 8(11):111110, 2020.

[13] Shigeru Kobayashi, Ryota Shimizu, Yasunobu Ando, and Taro Hitosugi. Autonomous exploration of an unexpected electrode material for lithium batteries. ACS Materials Letters, 5(10):2711–2717, 2023.

[14] Kazunori Nishio, Akira Aiba, Kei Takihara, Yota Suzuki, Ryo Nakayama, Shigeru Kobayashi, Akira Abe, Haruki Baba, Shinichi Katagiri, Kazuki Omoto, Kazuki Ito, Ryota Shimizu, and Taro Hitosugi. A digital laboratory with a modular measurement system and standardized data format. Digital Discovery, 4(7):1734–1742, 2025.

[15] T. Jesper Jacobsson. AI-generated hypotheses and the emergence of autonomous scientific discovery. ACS Materials Letters, 8(6):1457–1464, 2026.

[16] Seongmin Kim, Yousung Jung, and Joshua Schrier. Large language models for inorganic synthesis predictions. Journal ofthe American Chemical Society, 146(29):19654–19659, 2024.

[17] Seongmin Kim, Joshua Schrier, and Yousung Jung. Explainable synthesizability prediction of inorganic crystal polymorphs using large language models. Angewandte Chemie International Edition, 64(19):e202423950, 2025.

[18] Thorben Prein, Elton Pan, Janik Jehkul, Stefen Weinmann, Elsa Olivetti, and Jennifer L. M. Rupp. Language models enable data-augmented synthesis planning for inorganic materials. ACS Applied Materials & Interfaces, 17(51):69221–69233, 2025.

[19] Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations (ICLR), 2023.

[20] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS), volume 36, 2023.

[21] Shuyi Jia, Chao Zhang, and Victor Fung. LLMatDesign: Autonomous materials discovery with large language models. arXiv preprint arXiv:2406.13163, 2024.

[22] Izumi Takahara, Teruyasu Mizoguchi, and Bang Liu. Accelerated inorganic materials design with generative AI agents. Cell Reports Physical Science, 6(12):103019, 2025.

[23] Ryan Nduma, Hyunsoo Park, and Aron Walsh. Crystalyse: a multi-tool agent for materials design. arXiv preprint arXiv:2512.00977, 2025.

[24] Seongmin Kim, Jaehwan Choi, Kunik Jang, Junkil Park, Varinia Bernales, Alán Aspuru-Guzik, and Yousung Jung. Materealize: a multi-agent deliberation system for end-to-end material design and synthesis. arXiv preprint arXiv:2601.15743, 2026.

[25] Daniil A. Boiko, Robert MacKnight, Ben Kline, and Gabe Gomes. Autonomous chemical research with large language models. Nature, 624(7992):570–578, 2023.

[26] Andres M. Bran, Sam Cox, Oliver Schilter, Carlo Baldassari, Andrew D. White, and Philippe Schwaller. Augmenting large language models with chemistry tools. Nature Machine Intelligence, 6:525–535, 2024.

[27] Kourosh Darvish, Marta Skreta, Yuchi Zhao, Naruki Yoshikawa, Sagnik Som, Miroslav Bogdanovic, Yang Cao, Han Hao, Haoping Xu, Alán Aspuru-Guzik, Animesh Garg, and Florian Shkurti. ORGANA: A robotic assistant for automated chemistry experimentation and characterization. Matter, 8(2):101897, 2025.

[28] Indrajeet Mandal, Jitendra Soni, Mohd Zaki, Morten M. Smedskjaer, Katrin Wondraczek, Lothar Wondraczek, Nitya Nand Gosvami, and N. M. Anoop Krishnan. Evaluating large language model agents for automation of atomic force microscopy. Nature Communications, 16:9104, 2025.

[29] Aikaterini Vriza, Michael H. Prince, Tao Zhou, Henry Chan, and Mathew J. Cherukara. Operating advanced scientific instruments with AI agents that learn on the job. npj Computational Materials, 12:160, 2026.

[30] Tongyu Shi, Yutang Li, Zhanlong Wang, Wenhe Xu, Guolai Jiang, Dawei Dai, Jie Zhou, Hao Huang, Rui He, Seeram Ramakrishna, Paul K. Chu, Wenhua Zhou, and Xue-Feng Yu. Knowledge-driven autonomous materials research via collaborative multi-agent and robotic system. Matter, 9(2):102577, 2026.

[31] Xu Huang, Junwu Chen, Yuxing Fei, Zhuohan Li, Philippe Schwaller, and Gerbrand Ceder. CASCADE: Cumulative agentic skill creation through autonomous development and evolution. arXiv preprint arXiv:2512.23880, 2025.

[32] Yuxing Fei, Bernardus Rendy, Xiaochen Yang, Junhee Woo, Xu Huang, Chang Li, Shilong Wang, David Milsted, Yan Zeng, and Gerbrand Ceder. Agentic LLM reasoning in a self-driving laboratory for air-sensitive lithium halide spinel conductors. arXiv preprint arXiv:2604.11957, 2026.

[33] Angel Yanguas-Gil. Performance of AI agents based on reasoning language models on ALD process optimization tasks. Journal ofVacuum Science & Technology A, 44(4):043410, 2026.

[34] Abdoulatif Cissé, Max E. Cooper, Mengjia Zhu, Xenophon Evangelopoulos, and Andrew I. Cooper. Can we automate scientific reasoning in closed-loop experiments using large language models? Digital Discovery, 5:1132–1160, 2026.

[35] Yunheng Zou, Austin H. Cheng, Abdulrahman Aldossary, Jiaru Bai, Shi Xuan Leong, Jorge Arturo Campos-Gonzalez-Angulo, Changhyeok Choi, Cher Tian Ser, Gary Tom, Andrew Wang, Zijian Zhang, Ilya Yakavets, Han Hao, Chris Crebolder, Varinia Bernales, and Alán Aspuru-Guzik. El Agente: An autonomous agent for quantum chemistry. Matter, 8(7):102263, 2025.

[36] K. Mizushima, P. C. Jones, P. J. Wiseman, and J. B. Goodenough. $\mathrm { L i } _ { x } \mathrm { C o O } _ { 2 } ~ ( 0 < x \leq 1 )$ : A new cathode material for batteries of high energy density. Solid State Ionics, 3–4:171–174, 1981.

[37] M. Antaya, K. Cearns, J. S. Preston, J. N. Reimers, and J. R. Dahn. In situ growth of layered, spinel, and rock-salt $\mathrm { L i C o O } _ { 2 }$ by laser ablation deposition. Journal ofApplied Physics, 76(5):2799–2806, 1994.

[38] Kazunori Nishio, Koji Horiba, Naoto Nakamura, Miho Kitamura, Hiroshi Kumigashira, Ryota Shimizu, and Taro Hitosugi. Bottom-current-collector-free thin-film batteries using $\mathrm { L i N i _ { 0 . 8 } C o _ { 0 . 2 } O _ { 2 } }$ epitaxial thin films. Journal ofPower Sources, 416:56–61, 2019.

[39] OpenAI. Introducing GPT-5.5. https://openai.com/index/introducing-gpt-5-5/, 2026. Published April 23, 2026; accessed August 20, 2026.

[40] OpenAI. OpenAI Agents SDK. https://openai.github.io/openai-agents-python/, 2025. Accessed August 20, 2026.

[41] Japan Analytical Instruments Manufacturers’ Association (JAIMA). Comprehensive file format (MaiML) for the measurement and analysis. https://www.maiml.org/index\_en.html. Accessed September 15, 2026.

# Supplementary Information for: Hypothesis-Driven Autonomous Materials Synthesis with Multimodal LLM Agents

This Supplementary Information contains Supplementary Table 1, summarizing all iterations of the autonomous $\mathrm { L i C o O } _ { 2 }$ synthesis campaign, Supplementary Note 1, reproducing the agent’s understanding document at key stages of the campaign, and Supplementary Note 2, reproducing the full proposal and reflection records of the three iterations whose predictions were refuted.

Table S1: Summary of the autonomous $\mathbf { L i C o O } _ { 2 }$ synthesis campaign. For each iteration, the table lists the proposal mode, the proposed substrate temperature $T _ { \mathrm { s u b } }$ , the predicted and measured $I ( 0 0 3 ) / I ( 0 0 6 )$ and the judgment of the reflection turn. In verify mode, the prediction is judged as supported when the measurement and the prediction both indicate successful ordered growth. In falsify mode, it is judged as supported when both indicate failure. No numerical tolerance was imposed on the diference between the predicted and measured values, and the agent itself made the judgment. All 18 judgments were decided by whether the outcome fell into the predicted category: the three refuted judgments correspond to reversals between success and failure, whereas deviations within the same category, up to a factor of eight at $4 0 0 ^ { \circ } \mathrm { C } .$ were judged as supported.
<table><tr><td colspan="3"></td><td colspan="2">I(003)/1(006)</td></tr><tr><td>Iteration</td><td>Mode</td><td> $T _ { \mathrm { s u b } } ( ^ { \circ } \mathrm { C } )$ </td><td>Predicted</td><td>Measured Judgment</td></tr><tr><td>1</td><td>verify</td><td>600</td><td>8.0 0.001</td><td>refuted</td></tr><tr><td>2</td><td>falsify</td><td>700 0.001</td><td>35.6</td><td>refuted</td></tr><tr><td>3</td><td>verify</td><td>710 34.0</td><td>34.8</td><td>supported</td></tr><tr><td>4</td><td>falsify</td><td>400 0.010</td><td>0.081</td><td>supported</td></tr><tr><td>5</td><td>verify</td><td>720</td><td>32.0 31.8</td><td>supported</td></tr><tr><td>6</td><td>falsify</td><td>250</td><td>0.005 0.056</td><td>supported</td></tr><tr><td>7</td><td>verify</td><td>730 29.0</td><td>34.9</td><td>supported</td></tr><tr><td>8</td><td>falsify</td><td>500 0.050</td><td>0.001</td><td>supported</td></tr><tr><td>9</td><td>verify</td><td>740 33.0</td><td>35.5</td><td>supported</td></tr><tr><td>10</td><td>falsify</td><td>650 0.200</td><td>54.9</td><td>refuted</td></tr><tr><td>11</td><td>verify</td><td>660 52.0</td><td>60.2</td><td>supported</td></tr><tr><td>12</td><td>falsify</td><td>320 0.050</td><td>0.043</td><td>supported</td></tr><tr><td>13</td><td>verify</td><td>670 58.0</td><td>61.6</td><td>supported</td></tr><tr><td>14</td><td>falsify</td><td>750 34.0</td><td>40.9</td><td>supported</td></tr><tr><td>15</td><td>verify</td><td>680 60.0</td><td>55.0</td><td>supported</td></tr><tr><td>16</td><td>falsify</td><td>620 1.0</td><td>0.001</td><td>supported</td></tr><tr><td>17</td><td>verify</td><td>690 48.0</td><td>58.2</td><td>supported</td></tr><tr><td>18</td><td>falsify</td><td>630</td><td>2.0 0.001</td><td>supported</td></tr></table>

## Supplementary Note 1: Evolution of the understanding document

SynAgent maintains its evolving understanding of the target synthesis process as an explicit natural-language document, denoted <understanding> in the main text, and rewrites this document at the end of every iteration.

This note reproduces the document verbatim at four key stages of the $\mathrm { L i C o O } _ { 2 }$ campaign: after iteration 1, in which the initial expectation that $6 0 0 ^ { \circ } \mathrm { C }$ would be near optimal was refuted, after iteration 2, in which the falsify proposal discovered ordered growth at $7 0 0 ~ ^ { \circ } \mathrm { C }$ , after iteration 10, in which the falsify proposal discovered the superior growth window at $6 5 0 ^ { \circ } \mathrm { C } ,$ and after the final iteration 18, in which the understanding reached the four-region map described in the main text. The header of each box summarizes the proposal and the outcome of the corresponding iteration. Bold face follows the emphasis placed by the agent in the original document, and only typographic conversion, such as the typesetting of chemical formulae, was applied, leaving the wording unchanged.

Understanding after iteration 1   
verify proposal at $6 0 0 ^ { \circ } C ,$ predicted �(003)/�(006) = 8.0, measured 0.001, judged refuted   
At $\mathbf { 6 0 0 } \ { } ^ { \circ } \mathbf { C } ,$ LiCoO<sub>2</sub> growth under the fixed sputtering conditions fails to produce ordered layered (001) crys  
tallinity: the measured $I ( 0 0 3 ) / I ( 0 0 6 ) = 0 . 0 0 1$ is essentially zero, contradicting the initial assumption that $6 0 0 ^ { \circ } \mathrm { C }$   
would be near-optimal.   
The SEM morphology supports this failure: the surface is mostly fine-textured/dark but contains many bright   
isolated particles/islands, with an estimated bright-particle area fraction of ∼6%. This suggests inhomogeneous   
growth, likely from secondary-phase segregation, Li/Co oxide particulate formation, or islanding, rather   
than a continuous well-ordered layered $\mathrm { L i C o O } _ { 2 }$ film.   
The most plausible mechanism is that, at $6 0 0 ~ ^ { \circ } \mathrm { C }$ and the given oxygen pressure, the film becomes chemically   
or morphologically unstable: Li volatility/of-stoichiometry and/or secondary-phase formation may domi  
nate over the beneficial efect of increased adatom mobility. Thus, the previous claim that intermediate-high   
temperature improves ordering up to around $6 0 0 ^ { \circ } \mathrm { C }$ is now doubtful.   
Current working hypothesis: the optimal substrate temperature, if it exists, is likely below ${ \bf 6 0 0 } ^ { \circ } { \bf C } .$ , where Li   
loss and secondary-phase segregation may be reduced while still allowing enough adatom mobility for layered   
ordering.

Understanding after iteration 2   
falsify proposal at $7 0 0 ^ { \circ } C ,$ predicted �(003)/�(006) = 0.001, measured 35.6, judged refuted   
The prior claim that the optimum is likely below $\mathbf { 6 0 0 } ^ { \circ } \mathbf { C }$ is now rejected. A $7 0 0 ^ { \circ } \mathrm { C }$ film gave $I ( 0 0 3 ) / I ( 0 0 6 ) =$   
35.64, while $6 0 0 ^ { \circ } \mathrm { C }$ gave 0.001, so the temperature response is strongly non-monotonic and cannot be explained   
by simple high-temperature degradation.   
The prior observation that $\mathbf { 6 0 0 } ^ { \circ } \mathbf { C }$ is a failed growth condition is reinforced. Its particle-rich SEM morphology   
and near-zero XRD ratio still indicate poor layered $\mathrm { L i C o O } _ { 2 }$ formation, likely from a local instability such as   
phase segregation or incomplete transformation.   
New working hypothesis: ${ 7 0 0 ~ ^ { \circ } C }$ crosses a crystallization/mobility threshold that enables strong layered

$\mathrm { L i C o O } _ { 2 }$ (001) ordering on $\mathbf { A l } _ { 2 } \mathbf { O } _ { 3 }$ under the fixed sputtering/O<sub>2</sub> conditions. In this regime, the dominant mechanism appears to be improved adatom mobility and crystallization kinetics, producing a continuous finetextured film with essentially no bright secondary-phase particles in SEM.

The $6 0 0 ~ ^ { \circ } \mathrm { C }$ failure is now best interpreted not as evidence that all higher temperatures cause Li loss, but as a possible intermediate-temperature instability window where mobility or reaction kinetics are insuficient for complete layered ordering, allowing segregated particles/islands to form.

Open questions:

• Is $7 0 0 ^ { \circ } \mathrm { C }$ near the optimum, or does performance improve further at $7 1 0 { - } 7 5 0 \ ^ { \circ } \mathrm { C ? }$

• Does Li volatility/of-stoichiometry eventually dominate at still higher temperature?

• Where is the sharp transition between failed $6 0 0 ^ { \circ } \mathrm { C }$ growth and successful $7 0 0 ^ { \circ } \mathrm { C }$ growth?

Discriminating evidence would be XRD/SEM at temperatures between 600 and $7 0 0 ~ ^ { \circ } \mathrm { C }$ and above $7 0 0 ~ ^ { \circ } \mathrm { C } \colon$ a smooth increase would support a crystallization-threshold mechanism, while a peak followed by degradation would indicate onset of Li-loss-driven instability.

## Understanding after iteration 10

## falsify proposal at $6 5 0 ^ { \circ } C ,$ predicted �(003)/�(006) = 0.200, measured 54.9, judged refuted

The prior claim that the crystallization threshold lies close to ${ 7 0 0 } ^ { \circ } \mathrm { C }$ is rejected. The $\mathbf { 6 5 0 ~ } ^ { \circ } \mathbf { C }$ sample produced $I ( 0 0 3 ) / I ( 0 0 6 ) = 5 4 . 9 2$ , the best result so far and much higher than the ${ 7 0 0 } { - 7 4 0 } ~ ^ { \circ } \mathrm { C }$ values (∼32–36). Thus, $6 5 0 ~ ^ { \circ } \mathrm { C }$ is not below the ordering threshold; it is now the leading candidate for the optimum.

The updated working hypothesis is that the useful ordered-growth window begins somewhere between 600 and ${ \bf 6 5 0 } \ ^ { \circ } { \bf C } ,$ with a sharp transition from the failed $6 0 0 ~ ^ { \circ } \mathrm { C }$ condition to highly ordered growth by $6 5 0 ~ ^ { \circ } \mathrm { C } .$ . In this window, the dominant mechanism is likely that substrate temperature is high enough to enable $\mathbf { L i } / \mathbf { C o } / \mathbf { O }$ adatom mobility and layered-phase crystallization, while still low enough to avoid some high-temperature roughening, defect formation, or Li volatility that may slightly limit the ${ 7 0 0 } { - 7 4 0 } ^ { \circ } \mathrm { C }$ plateau.

The $6 0 0 ~ ^ { \circ } \mathrm { C }$ failure is now best viewed as a local pre-threshold instability or incomplete-crystallization regime, not as representative of all temperatures below $7 0 0 ^ { \circ } \mathrm { C } . \mathrm { A t } 6 0 0 ^ { \circ } \mathrm { C } .$ , mobility/reaction kinetics may be just insuficient for complete layered ordering, allowing segregated particles or poorly ordered phases to dominate. The high-temperature success window remains reinforced but is no longer assumed optimal: ${ \bf 7 0 0 - 7 4 0 ~ } ^ { \circ } { \bf C }$ consistently gives strong ordering, but not as high as $6 5 0 ~ ^ { \circ } \mathrm { C } .$ . This suggests that excess thermal energy may introduce more defects, roughening/coarsening, or slight stoichiometric deviation even though the layered phase remains dominant.

Morphology interpretation is further refined: the $6 5 0 ~ ^ { \circ } \mathrm { C }$ SEM shows sparse bright triangular/faceted features $( \sim 4 . 6 \%$ bright area fraction), yet the XRD ratio is excellent. Therefore, faceted bright features can coexist with very high layered ordering and are not automatically destructive secondary phases. Their composition remains unknown.

Current working map:

$2 5 0 ~ ^ { \circ } \mathrm { { C } : }$ continuous but likely amorphous/nanocrystalline; poor ordering from insuficient mobility.

$\mathbf { 4 0 0 - 6 0 0 } ^ { \circ } \mathbf { C } \mathbf { : }$ poor/failure regime with incomplete crystallization and particle/island segregation.

${ \bf 6 5 0 ~ } ^ { \circ } { \bf C } :$ best observed condition; likely near optimal balance of mobility, layered ordering, and limited high-temperature degradation.

$7 0 0 { \ - } 7 4 0 \ ^ { \circ } \mathbf { C } ;$ robust but lower plateau of strong layered ordering.

Open questions:

• Where exactly between 600 and $\mathbf { 6 5 0 } \ ^ { \circ } \mathbf { C }$ does the sharp transition occur?

• Is the true optimum at $\mathbf { 6 5 0 } \mathbf { \mathrm { ~ ‰ ~ } }$ or nearby, e.g. $6 3 0 { - } 6 8 0 \ ^ { \circ } \mathrm { C ? }$

• Does $7 5 0 ^ { \circ } \mathrm { C }$ remain high-performing or begin to show upper-temperature degradation?

• Are the bright triangular/faceted features at 650 and $7 4 0 ~ ^ { \circ } \mathrm { C }$ $\mathrm { L i C o O } _ { 2 }$ crystallites, secondary phases, or surface-segregated material? Secondary-peak inspection or compositional analysis would discriminate these possibilities.

## Understanding after iteration 18

falsify proposal at $6 3 0 ^ { \circ } C ,$ predicted �(003)/�(006) = 2.0, measured 0.001, judged supported

The lower crystallization threshold is now constrained more tightly: ${ \bf 6 3 0 } ^ { \mathrm { ~ \circ ~ } } \mathrm { ~ C ~ }$ still fails completely with $I ( 0 0 3 ) / I ( 0 0 6 ) = \mathbf { 0 . 0 0 1 }$ , while $\mathbf { 6 5 0 ~ } ^ { \circ } \mathbf { C }$ is excellent at 54.92. Therefore, the abrupt transition to high-quality layered $\mathrm { L i C o O } _ { 2 }$ ordering occurs between 630 and $\mathbf { 6 5 0 } \mathbf { \mathrm { ~ ‰ ~ } }$

The claim that ${ \bf 6 3 0 } ~ ^ { \circ } \bf C$ remains in the pre-threshold incomplete-crystallization/segregation regime is reinforced. SEM at $6 3 0 ^ { \circ } \mathrm { C }$ shows many bright triangular/faceted particles or islands on a dark continuous background: 128 bright objects, ${ \sim } 6 . 4 \%$ bright area fraction, typical diameter ∼25 px. Because this morphology coincides with near-zero XRD ratio, these features likely represent segregated islands/secondary crystallites or incomplete phase formation rather than coherent layered $\mathrm { L i C o O } _ { 2 }$ (001) growth.

The dominant mechanism remains a temperature-activated crystallization threshold. Below the threshold, Li/Co/O mobility and layered-phase reaction kinetics are insuficient, so the film forms a poorly ordered matrix plus particles/islands. Above the threshold, the matrix crystallizes into strongly layered $\mathrm { L i C o O } _ { 2 } .$ , and similar faceted surface features can become benign rather than destructive.

Current working map:

$2 5 0 { - } 3 2 0 \ ^ { \circ } \mathbf { C } ;$ mobility-limited amorphous/nanocrystalline continuous films; very poor ordering.

$\mathbf { 4 0 0 - 6 3 0 } \mathbf { \mathit { ^ { \circ } C } } \mathbf { : }$ incomplete crystallization/segregation regime; near-zero ordering with particles/islands.

$\mathbf { 6 5 0 - 6 9 0 \ ^ { \circ } C : }$ high-quality window; optimum near $\mathbf { 6 6 0 - 6 7 0 } \ ^ { \circ } \mathbf { C } ,$ with $\mathbf { 6 7 0 } \ ^ { \circ } \mathbf { C }$ currently best.

• 700–750 ◦C: good but suboptimal high-temperature growth, likely limited by subtle defects, texture, strain, or mild stoichiometry shifts.

Remaining key open question: the transition lies between 630 and ${ \bf 6 5 0 } ^ { \circ } { \bf C } ;$ testing $\mathbf { 6 4 0 } ^ { \circ } \mathbf { C }$ would locate whether the switch is very sharp near $6 4 0 – 6 5 0 \ ^ { \circ } \mathrm { C }$

## Supplementary Note 2: Proposal and reflection records of the refuted iterations

Each iteration of the campaign is recorded as a structured proposal, committed before the experiment, and a structured reflection, produced after the measurement. This note reproduces these records verbatim for the three iterations whose predictions were refuted (iterations 1, 2, and 10), which triggered the major revisions of the understanding described in the main text. The proposal record lists the mode, the proposed condition, the committed prediction, the hypothesis under test, and the reasoning behind the proposal. The reflection record lists the measured target, the judgment, the reasoning behind the judgment, the observation of the SEM image, and, because the prediction was refuted, the abduction that names the claim now in doubt and proposes the most plausible alternative explanation.

Proposal and reflection record of iteration 1   
Mode: verify   
Proposed condition: substrate temperature $6 0 0 ^ { \circ } \mathrm { C }$   
Predicted target: $I ( 0 0 3 ) / I ( 0 0 6 ) = 8 . 0$   
Hypothesis: $\mathrm { L i C o O } _ { 2 }$ layered ordering and (001) crystallinity are maximized at an intermediate-high substrate   
temperature where adatom mobility is suficient for cation ordering without excessive Li volatility or secondary  
phase formation.   
Reasoning: With no prior observations, I test the common sputtered $\mathrm { L i C o O } _ { 2 }$ mechanism that increasing substrate   
temperature improves surface difusion and layered ordering up to an optimum, while too high a temperature   
can degrade stoichiometry via Li loss. A substrate temperature of $6 0 0 ^ { \circ } \mathrm { C }$ is a plausible near-optimal verification   
point for crystalline layered $\mathrm { L i C o O } _ { 2 }$ on sapphire under fixed oxygen pressure. The expected $I ( 0 0 3 ) / I ( 0 0 6 )$ value   
is set to 8.0 as a concrete prediction for a successful, well-ordered film under this hypothesis.   
Measured target: $I ( 0 0 3 ) / I ( 0 0 6 ) = 0 . 0 0 1$   
Judgment: refuted   
Reflection: The verification prediction was a successful high $I ( 0 0 3 ) / I ( 0 0 6 )$ ratio of 8.0 at $6 0 0 ~ ^ { \circ } \mathrm { C } ,$ but the   
measured value was 0.001, essentially indicating no useful 003/006 layered $\mathrm { L i C o O } _ { 2 }$ difraction ratio. This is   
qualitatively opposite to the hypothesis that $6 0 0 ~ ^ { \circ } \mathrm { C }$ would provide suficient adatom mobility for well-ordered   
(001) $\mathrm { L i C o O } _ { 2 }$ without damaging stoichiometry.   
SEM observation: The SEM image shows a mostly dark, fine-textured/continuous background with many bright   
isolated particles or islands distributed across the surface. Automated morphology analysis estimates about $6 . 3 \%$   
bright-particle area fraction with 263 bright objects in the image and typical equivalent diameters around 17   
px. This particle-rich, inhomogeneous morphology is consistent with secondary-phase particles, Li/Co oxide   
segregation, droplets, or Volmer-Weber-like islanding rather than a smooth epitaxial layered film, supporting the   
very low XRD ratio.   
Abduction: The claim now in doubt is that ${ \bf 6 0 0 } \ ^ { \circ } { \bf C }$ lies in the optimal intermediate-high temperature   
regime where mobility improves layered ordering without causing degradation. The evidence instead points   
to the alternative that, under the fixed $\mathrm { s p u t t e r i n g } / \mathrm { O } _ { 2 }$ conditions, ${ \bf 6 0 0 } \ ^ { \circ } { \bf C }$ is already too high or chemically   
unfavorable, likely promoting Li volatility, of-stoichiometry, secondary-phase segregation, or particulate/island

$$
\mathrm { L i C o O } _ { 2 }
$$

$$
7 0 0 ^ { \circ } \mathrm { C }
$$

$$
\mathrm { L i C o O } _ { 2 }
$$

$$
6 0 0 ^ { \circ } \mathrm { C } ,
$$

$$
7 0 0 ~ ^ { \circ } \mathrm { C }
$$

$$
{ } . 6 0 0 ~ ^ { \circ } \mathrm { C }
$$

$$
6 0 0 ^ { \circ } \mathrm { C }
$$

$$
I ( 0 0 3 ) / I ( 0 0 6 ) = 3 5 . 6
$$

$$
7 0 0 ^ { \circ } \mathrm { C }
$$

$$
6 0 0 ^ { \circ } \mathrm { C } ,
$$

$$
7 0 0 ^ { \circ } \mathrm { C } .
$$

$$
6 0 0 ^ { \circ } \mathrm { C } .
$$

$$
\mathbf { 6 0 0 } ^ { \circ } \mathbf { C }
$$

$$
{ \bf 6 0 0 } \ ^ { \circ } { \bf C }
$$

$$
{ \bf 7 0 0 } ^ { \circ } { \bf C }
$$

$$
\mathrm { L i C o O } _ { 2 }
$$

$$
6 0 0 ~ ^ { \circ } \mathrm { C }
$$

Proposal and reflection record of iteration 10   
Mode: falsify   
Proposed condition: substrate temperature $6 5 0 ~ ^ { \circ } \mathrm { C }$

Predicted target: $I ( 0 0 3 ) / I ( 0 0 6 ) = 0 . 2 0 0$

Hypothesis: The sharp crystallization threshold has not been reached well below $7 0 0 ~ ^ { \circ } \mathrm { C } ;$ at $6 5 0 ~ ^ { \circ } \mathrm { C }$ adatom mobility and layered-phase reaction kinetics should still be insuficient for complete $\mathrm { L i C o O } _ { 2 }$ (001) ordering, leading to underperforming or segregated/incompletely crystallized films rather than the high-ratio ${ 7 0 0 } { - 7 4 0 } ^ { \circ } \mathrm { C }$ regime.

Reasoning: Falsify mode should probe an under-sampled region away from the verified $7 0 0 { - } 7 4 0 \ ^ { \circ } \mathrm { C }$ success cluster. The most informative unsampled gap is between the failed $6 0 0 ^ { \circ } \mathrm { C }$ point and the successful $7 0 0 ^ { \circ } \mathrm { C }$ point. Choosing $6 5 0 ~ ^ { \circ } \mathrm { C }$ tests whether the transition occurs only close to $7 0 0 ~ ^ { \circ } \mathrm { C } ;$ under the current threshold model, $6 5 0 ~ ^ { \circ } \mathrm { C }$ is expected to remain below the mobility/crystallization threshold and fail or strongly underperform. I predict $I ( 0 0 3 ) / I ( 0 0 6 ) = 0 . 2$ , allowing slight improvement over $6 0 0 ^ { \circ } \mathrm { C }$ but still far below a successful film.

Measured target: $I ( 0 0 3 ) / I ( 0 0 6 ) = 5 4 . 9$

Judgment: refuted

Reflection: The falsify prediction was that $6 5 0 ~ ^ { \circ } \mathrm { C }$ would remain below the crystallization threshold and underperform, with expected $I ( 0 0 3 ) / I ( 0 0 6 ) = \mathbf { 0 . 2 }$ . Instead, the measured value is 54.92220144, the highest result so far and far above the ${ 7 0 0 } { - 7 4 0 } ^ { \circ } \mathrm { C }$ plateau. This is a surprise success and refutes the claim that $6 5 0 ^ { \circ } \mathrm { C }$ is below the threshold for strong layered $\mathrm { L i C o O } _ { 2 }$ ordering.

SEM observation: The SEM shows a continuous rough/fine-textured background with sparse but prominent bright triangular/faceted crystallites or islands. Quantitative analysis found 49 bright particles, $\sim 4 . 6 \%$ bright area fraction, and typical equivalent diameter $\sim 3 6 \ \mathbf { p x } .$ Despite these faceted features, the XRD ratio is exceptionally high, so the morphology is more consistent with coexistence of strong layered $\mathrm { L i C o O } _ { 2 }$ ordering with faceted crystalline growth or minor surface segregation, rather than destructive secondary-phase-dominated failure.

Abduction: The prior claim now in doubt is that the crystallization threshold lies close to $7 0 0 ^ { \circ } \mathrm { C }$ and that $6 5 0 ^ { \circ } \mathrm { C }$ should still be mobility-limited. The evidence instead points to the alternative working hypothesis that $\mathbf { 6 5 0 } ~ ^ { \circ } \mathbf { C }$ is within, or possibly near the optimum of, the ordered $\mathbf { L i C o O } _ { 2 }$ growth window. A plausible mechanism is that $6 5 0 ~ ^ { \circ } \mathrm { C }$ provides enough adatom mobility for layered ordering while reducing high-temperature roughening, Li volatility, or excess defect formation compared with higher temperatures. The very poor $6 0 0 ~ ^ { \circ } \mathrm { C }$ result may therefore represent a sharp local transition or instability just below the ordered-growth window rather than evidence for a broad $6 0 0 { - } 7 0 0 \ { ^ { \circ } C }$ failure regime.