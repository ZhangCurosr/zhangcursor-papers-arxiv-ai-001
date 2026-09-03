Preprint

# Collective creativity in hybrid societies

Mason Youngblood<sup>1,2,\*</sup>, Katie Mudd<sup>1</sup>, Manuel Anglada-Tort<sup>3</sup>, Cameron Jones<sup>4</sup>, Elena Miu<sup>5</sup>, Diana Omigie<sup>3</sup> and Margaret Schedel<sup>6</sup>

<sup>1</sup>Institute for Advanced Computational Science, Stony Brook University, USA <sup>2</sup>Animal Culture Observatory, USA

<sup>3</sup>Department of Psychology and Neuroscience, Goldsmiths, University of London, UK <sup>4</sup>Department of Psychology, Stony Brook University, USA

<sup>5</sup>Department of Linguistics, Cognitive Science and Semiotics, Aarhus University, DK

<sup>6</sup>Department of Music, Stony Brook University, USA

\*Corresponding author: Mason Youngblood (masonyoungblood@gmail.com)

## ABSTRACT

Generative AI is changing how cultural artifacts are created and circulated, and with it our understanding of creativity itself. Researchers disagree about whether these tools enrich or impoverish culture, and we argue that much of that disagreement comes from conflating two distinct components of creativity: novelty, a property of single artifacts, and diversity, a property of populations. We argue further that creativity in the context of generative AI is best understood as a property of hybrid collectives, or populations of interacting people and algorithms, rather than of individuals. AIassisted ideation reliably raises the novelty of individual output while narrowing diversity in the aggregate, but this is not an inevitable consequence of putting machines in the loop. Because humans and models search in complementary ways, mixed groups can outperform and out-diversify groups of either kind alone, and machine-discovered solutions can enter human culture and persist there. What decides the outcome is composition: which agents are present, in what proportion and how they are connected. The question is no longer whether AI helps or harms creativity, but which mixtures let individual gains accumulate without eroding collective diversity.

## 1 INTRODUCTION

Generative AI is now embedded in creative work across writing, music, images and code, altering how cultural artifacts are created and circulated. This shift sharpens an ongoing debate about creativity itself: is it a capacity of individuals, or a property of the systems that connect them? Building on theories of creativity<sup>1</sup> and recent work on collective and computational creativity<sup>2,3</sup>, we argue for the latter—that creativity in the age of generative AI is best understood as an emergent property of hybrid collectives of interacting humans and algorithms. This framing follows broader accounts of societies as human–algorithm systems characterized by feedback loops, distributed cognition and multi-level dynamics<sup>4,5</sup>.

Much of the disagreement in this literature comes from treating two measures as one. Novelty is a property of a single artifact measured against a reference set (i.e., how far it departs from what came before), whereas diversity is a property of a set itself (i.e., how much its members differ from one another). A tool can raise the former while lowering the latter, and the field still lacks agreed ways of assessing either at the population level. Drawing on cognitive science, computational social science and cultural evolution, we ask whether embedding AI in creative collectives widens or narrows the space of what a society can imagine.

## 2 REFRAMING CREATIVITY

## Beyond the lone genius to the creative collective

History celebrates solo inventors, scientists and artists, and psychology followed suit for much of the twentieth century, hunting for the personality traits, cognitive profiles and neural correlates of creative individuals<sup>6–9</sup>. On this account, creativity sits inside a brain.

Work on collective creativity points elsewhere, describing human societies as “collective brains” in which innovation emerges from cultural learning and the recombination of ideas across many individuals<sup>10</sup>. Major discoveries are routinely made independently and near-simultaneously, as if available to the field rather than the individual<sup>11</sup>, and teams have since displaced solo authors as the source of high-impact work in nearly every field<sup>12</sup>. Creativity researchers increasingly describe creative work as relational and distributed<sup>13</sup>, arising through interaction and spreading across practices and groups rather than residing in a single head.

The two levels also come apart empirically. In networked discovery experiments, solo performance predicts group performance only weakly (r = 0.18–0.23, against 0.38–0.54 between group conditions), suggesting they are at least partly distinct capacities<sup>14</sup>. Assembling creative individuals is no guarantee of a creative collective.

Much of the difference lies in the structure and composition of the collective itself. Network size and topology shape consensus, problemsolving and collective intelligence<sup>15–17</sup>, and features like clustering and information eficiency determine how well a group aggregates distributed knowledge<sup>18,19</sup>. Within-group factors like sociality, transmission fidelity and cultural variance also matter<sup>10</sup>, although almost all of this evidence comes from groups of humans. What changes when machines enter the network, whether as tools that shape what people produce or as participants that produce in their own right?

## Toward hybrid creativity: AI as a partner

AI systems now shape collective behavior and social systems in addition to individual behavior<sup>5,20,21</sup>. Non-human actors can alter group dynamics in surprising ways: bots placed centrally in a network and injecting small amounts of random noise helped human groups out of the dead ends of a coordination game, in which every player had to choose a color differing from all of their neighbors’ and could see only their immediate neighbors, while noisier or more peripheral bots did not<sup>22</sup>. More recently, AI-only networks began with higher rated creativity and higher collective diversity than human-only or mixed ones, yet over successive generations the mixed networks overtook them in diversity, because AI agents retained little of the stories they were given while human chains preserved continuity<sup>23</sup>.

Table 1: Mechanisms of idea generation<sup>1</sup> in hybrid collectives.
<table><tr><td>Mechanism</td><td>Definition</td><td>What humans and machines each do</td><td>What hybrids add</td></tr><tr><td>Combination / recombination</td><td>Connecting ideas already present to create something new.</td><td>Machines excel at new combinations across far more material than any single person²0.</td><td>Cross-context combiners: a model trained on scientific content and author networks can predict “alien“ hypotheses unavailable to individual scientists37.</td></tr><tr><td>Search / exploration</td><td>Searching a constrained space of possibilities.</td><td>Both can search; machines are particularly fast³1.</td><td>Simple bots accelerate collective search by relaying ideas into distant network regions14. AlphaGo&#x27;s unorthodox moves raised novelty and quality in human openings without direct copying20,38.</td></tr><tr><td>Transformation</td><td>Stepping outside a space to open a new one; progress mixes tweaks with transformations36.</td><td>Hardest for machines alone; may require interaction with the physical environment31, and models lack the causal theories that support this kind of restructuring39.</td><td>Machines may supply triggers they cannot act on. Out-of-domain contact (e.g., jazz syncopation reorganizing Mondrian&#x27;s late painting) drives human transformation⁴0, and AlphaGo similarly led professionals to re- examine long-standing judgments about which configurations of stones are strong.</td></tr></table>

Why should a mixed system behave differently from a human one? There are two answers. The first is that beliefs about the partner change behavior. Musicians who think they are improvising with an AI, even when a hidden human is responding, lower their expectations of musicality and feel released from social judgment, which emboldens creative risks<sup>24</sup>. People do not hold a fixed view of the partner, either, casting the same tool as friend, student or manager<sup>25</sup>; merely believing an associate is an AI can also increase conformity<sup>26</sup>. The second answer is that humans and models generate information differently. Humans preserve what they are given, while models discard it in pursuit of new material<sup>23</sup>, and in collective search humans explore broadly while models exploit, settling into a single strategy in 99% of rounds when left alone<sup>27</sup>. Differences of this kind are why mixtures can beat either population alone, in line with long-standing results on group diversity<sup>28</sup>.

Design matters as much as belief: meaning in human–AI collaboration accumulates through turn-taking rather than a single static prompt, so tools that permit only rigid one-shot requests probably foreclose the shared agency that productive collaboration requires<sup>29</sup>.

## 3 MECHANISMS OF HYBRID CREATIVITY

We organize the mechanisms around the standard cultural-evolutionary cycle of variation, selection and retention<sup>30</sup>, under which creative production runs through three stages: generating variants, evaluating them, and transmitting the survivors. Cutting across those stages are the mechanisms by which novelty arises, divided into combination, search within a space of constraints, and transformation of the space itself<sup>31</sup>. Stages and mechanisms are separate axes rather than a nested hierarchy, since a collective can combine while generating and combine again while selecting.

## Idea generation

People who use AI produce work rated more creative individually, while the pool of work produced by many such people is less diverse. Participants using ChatGPT produced more creative output than with web search, although the gains were incremental<sup>32</sup>. In a brainstorming study inventing a toy from a brick and a fan, unassisted ideas were entirely unique, whereas 94% of ChatGPT-assisted ideas shared overlapping concepts, with nine participants independently naming their toy a “Build-a-Breeze Castle”<sup>33</sup>. Short stories show the same shape: more creative individually, more similar to one another<sup>34</sup>.

When many people sample from the same generative distribution, each draw lands nearer its high-density region, so quality per item rises while variance between items falls. Ideas from any one participant were about as diverse with ChatGPT as with a non-AI tool; only the group-level distribution narrowed<sup>35</sup>. That locates the problem in shared sampling rather than in the models themselves, which is why mixing models, injecting noise and pairing generation with search can recover diversity (Section 5).

Cultural change consists mostly of small modifications punctuated by rarer reorganizations<sup>36</sup>, and AI agents are likely to contribute more to the first than to the second (Table 1).

Transformation is the mechanism on which machines contribute least, and one explanation is that they complete statistical patterns without building the causal theories that let people learn from sparse data<sup>39</sup>. In narrative tasks, AI discards core elements in pursuit of novelty while humans retain characters and through-lines, so hybrid chains pair machine variants with human continuity<sup>23</sup>.

There is a gap between what these experiments measure and what the field concludes. The studies above<sup>33–35</sup> pool independently produced outputs, describing individual work rather than population dynamics. Designs that build in transmission tell a different story. When humans and models alternate within the same chain, hybrid groups outperform both human-only and AI-only groups and retain more exploratory variation than AI-only ones<sup>27</sup>. Seeding three reinforcement-learning agents into a founding generation let a counterintuitive strategy, which only 1 in 600 humans ever discovered alone, spread and persist across four subsequent human-only generations<sup>41</sup>. Composition matters more than capability here, a theme we return to in Section 5.

## Selection through evaluation

Selection determines which generated ideas survive, and evaluation operates at two levels that the literature tends to blur together: within a single creator, who filters their own options over many passes, and across creators, whose independent choices act as a population-level selection pressure.

Within a single creator, AI can take either role: generator, as when teams in the AI Song Contest produced large volumes of model output and selected among it afterwards<sup>42</sup>, or evaluator, ranking and recommending options on the creator’s behalf. What predicts a good outcome is not obviously evaluative skill. When students revised their own designs after rating GPT-3.5 proposals, expertise and baseline creative ability predicted the quality of the revision, whereas accuracy in judging the AI’s ideas predicted nothing, and both engineering and psychology students improved by comparable amounts<sup>43</sup>. The study measures revision rather than selection, so it speaks to whether judging a model’s output well carries over into improving one’s own, and finds that it does not. In collaborative coding chains, AI raters inflated scores, preferred their own outputs and could not distinguish humanguided from machine-guided work, yet swapping an AI selector in for a human one cost nothing so long as a human still wrote the instructions —suggesting that the human contribution concentrates in generation rather than evaluation<sup>44</sup>.

Across creators, evaluation becomes a selection pressure on a population. When many people rely on the same model both to propose and to rank ideas, their selection pressures become correlated<sup>35</sup>. Creativity-support tools are heavily skewed toward generation (45% of publications surveyed against 18% for evaluation)<sup>45</sup>, and non-experts tend to evaluate opportunistically, treating a single success or failure as evidence about the tool as a whole<sup>46</sup>. Selection can also run against novelty outright. Across 130,882 Artbreeder remix lineages, users preferred to remix simpler and less novel images even though more novel ones led to more popular remixes, and lineages drifted toward thematic attractors rather than fanning outward<sup>47</sup>. The drift reflects both halves of the system, since the generator shaped which images were easy to reach and the users decided which of them to build on. Shared evaluation and shared preference can compress a population even when the generator is capable of novelty.

## Transmission and collaboration

All human creative output relies on social information<sup>30,48</sup>. Because of that reliance, humans have evolved heuristics that guide what and who we pay attention to (i.e., social learning biases and strategies)<sup>49,50</sup>. Network shape also affects how information spreads, with consequences for collective innovation<sup>51,52</sup>: fully connected networks converge quickly on the same solutions, which benefits easy tasks, whereas partially connected ones preserve enough variation to reach complex innovations that fully connected groups never produce<sup>53</sup>.

Machines now intervene in both network structure and these learning biases<sup>5,20</sup>. They reshape the network directly (e.g., through friend suggestions) and circumvent it by promoting content regardless of how common it actually is locally. Conformist bias is a response to how widespread a position appears to be rather than to how widespread it is, so repeated algorithmic exposure can make a marginal view look like a majority one and recruit the bias on its behalf. Machines also serve as information-preservation and teaching agents, and thus as cultural transmitters in their own right<sup>20</sup>.

Whether any of this helps collective discovery depends on problem dificulty. Simple bots that rebroadcast neighbors’ guesses into distant network regions improve discovery when the semantic landscape is easy to navigate, and lose that advantage once people can no longer track which regions are worth searching<sup>14,54</sup>. More capable tools also bring more practitioners into creative domains<sup>20,55</sup>, though professional artists still draw more out of the same systems than laypeople do<sup>56</sup>, so lowering the barrier changes who participates without fully leveling what they produce.

## 4 HYBRID CREATIVITY IN PRACTICE

These mechanisms play out differently across domains, as the three cases below show.

## In science and technology

Generative AI has entered research practice as an ideation partner whose outputs human researchers then interpret<sup>57</sup>. Most deployed systems, though, are built to comply: the dominant co-creative tools in a recent survey are “pleasing” agents that generate contributions similar to whatever the human has just proposed, while “provoking” agents that contribute something distant remain uncommon<sup>58</sup>. A compliant partner will tend to amplify whatever its user already intended.

Systems designed to augment human judgment are likely to serve science better than systems designed to emulate it<sup>59</sup>, especially since capabilities differ. Models now match or exceed humans on standardized creativity tests measuring fluency and originality, but the best human responses still beat them on flexibility and on creativity as judged by human raters<sup>60,61</sup>, and controlled-task performance transfers poorly to work demanding more complex skills<sup>60</sup>. The clearest hybrid case is models that forecast both what is likely to be discovered and who is likely to discover it, surfacing hypotheses unavailable to scientists limited to the people and literatures they already know<sup>37</sup>.

Adoption has outrun skill: PhD students and early-career academics are the heaviest users of AI for research, mainly for writing rather than analysis<sup>62</sup>, yet non-experts asked to design prompts rarely test them systematically and tend to anthropomorphize the model<sup>46</sup>. If trainees use AI to circumvent the tasks that build expertise, they may lack the judgment to know when the model is wrong.

## In art, music and literature

In the arts, the human curatorial layer does most of the work attributed to the machine. The 2018 Christie’s sale of the GAN-generated Portrait of Edmond Belamy was framed as machine authorship, but originality rests on human decisions before and after generation—from assembling the training set to choosing which output to print<sup>63</sup>. Interactive machine-learning approaches make that layer explicit, allowing artists to shape computational behavior through examples and iterative refinement<sup>24,64,65</sup>.

Music shows the same pattern. David Cope’s Experiments in Musical Intelligence generated new compositions by extracting structural signatures from a corpus and recombining them while preserving higher-order patterns<sup>66</sup>. Algorithmic combination inherits human-created constraints rather than discovering them. Contemporary teams work similarly: nearly every entrant in the AI Song Contest broke composition into modules and recombined the output themselves<sup>42</sup>, and novices elicit several AI outputs and combine them into a coherent work<sup>67,68</sup>.

Literature shows what can happen at scale. Books written with AI assistance rose from near zero in 2022 to more than half of new releases in 2025; new Amazon releases roughly tripled, and average quality fell, though volume raised the number of moderately good books enough to lift estimated consumer surplus, or the total value readers gain beyond what they paid, by about 7%<sup>69</sup> —an economic proxy rather than a judgment of literary merit. This is the individual–collective tension running opposite to the ideation experiments: quality per item dropped while the value of the pool rose. Variety has not obviously followed, as plot elements recur across LLM-generated stories far more than across human ones<sup>70</sup>, and digital artists report finding AI images polished but “soulless”<sup>71</sup>.

## In games and other domains

Games offer the longest time series available, and caution against inferring durable change from short-term novelty. After Lee Sedol’s 2016 defeat by AlphaGo, many expected standard openings to disappear<sup>72</sup>, and professionals did initially depart from known sequences more often and earlier<sup>38</sup>. Set against four centuries of recorded play, though, superhuman AI produced only a modest and short-lived disruption<sup>73</sup>. Opening diversity instead is highest at intermediate player-population sizes and falls away at both extremes, with the peak occurring in the early 1980s, and the decline accompanied a shift from many small subgroups to a few large ones that was underway before superhuman AI—so any account attributing convergence to AI has to separate the model from the informational infrastructure through which it arrived. A caption-writing contest that pitted unassisted people against people using LLM assistance reproduces the same individual–collective split: LLM assistance raised idea count and lowered effort, the funniest items came mostly from people, and human–AI teams took the top places while feeling less ownership<sup>74</sup>.

## 5 CONSEQUENCES

The individual-level gains reviewed above are real. What they add up to at the level of a population is a separate question, and the answer depends on how the collective is assembled.

## Convergence and divergence

Studies that build in transmission suggest that composition, tools and cultural context largely decide outcomes. AI is often described as a blender, and whether it expands or compresses the space depends on who is holding it. AI-only story chains—iterated sequences in which each agent selects and rewrites a neighbor’s story—start out the most diverse and then converge, while mixed human–AI networks overtake them in diversity over successive generations, presumably because human continuity offsets machine novelty-seeking<sup>23</sup>. Partial benefits also emerged when different AI models (e.g., GPT and Claude) were combined within the same network, even though each model was limited on its own<sup>23</sup>. On Artbreeder, the generator and human remix choices jointly produced the drift toward thematic attractors<sup>47</sup>. In Go, consolidation of the player network was already homogenizing openings before AI arrived<sup>73</sup>. Convergence of this kind is not unique to machines: cumulative cultural evolution tends to reduce diversity as populations concentrate on refining solutions that are already performing well<sup>36</sup>.

Model collapse remains a structural risk, because model-generated text and images increasingly become training data for the next generation, and repeated self-training can narrow outputs toward a few dominant traits<sup>20</sup>. Convergence is at least partly a design choice, though: pairing LLMs with evolutionary search and explicit blending strategies sustains open-ended exploration<sup>75</sup>, and small curated training sets allow creators to treat a model’s idiosyncrasies as an expressive signature rather than inherit the monoculture of large general models<sup>65</sup>.

Convergence is not the only outcome. Mixing agents with different search strategies can push the other way: hybrid human–AI groups in collective search outperformed AI-only groups and retained more variation<sup>27</sup>, bots injecting small amounts of noise pulled human groups out of coordination dead ends<sup>22</sup>, and machine agents can put into circulation strategies that human populations essentially never find on their own<sup>41</sup>. Machines also make tractable a scale of distributed search that neither party would attempt alone<sup>37</sup>.

## Ownership, attribution and bias

Beyond output similarity, hybrid creativity raises unresolved questions about who creates and who is credited. Data scraping, exploitative labor and the scale of training corpora make tracing whose work is in a given output nearly impossible<sup>76,77</sup>, and artists who object find that opting out is dificult to verify<sup>78</sup>, leaving authorship and copyright unsettled<sup>79–81</sup>. The same pipelines encode biases that models reproduce, with image generators misrepresenting or erasing demographic groups<sup>82–84</sup> and language models favoring negative, threatening, socially salient and gender-stereotype-consistent material<sup>85</sup>. Because people absorb the outputs they are exposed to, machine error propagates back into human judgment. Participants assisted by a systematically biased classifier went on to repeat its mistakes unaided<sup>86</sup>, and in a three-stage chain a network trained on mildly biased human judgments amplified that bias from 53% to 65% and passed it to fresh participants, who drifted from 50% to 61% over six blocks; an equivalent all-human chain produced no such amplification, and merely believing the partner was an AI was enough to increase conformity<sup>26</sup>.

AI use can also shift how creators see themselves. People rate their creative abilities lower when working with AI<sup>87</sup>, users of LLM tools report feeling less responsible for the ideas they generate<sup>35,74</sup>, and inconsistent model behavior can frustrate deliberate changes<sup>42,71</sup>. How much this stings is domain-specific: computer scientists want verifiable knowledge and read a malfunctioning model as a failure, whereas new media artists treat those same errors as a playful source of ideas<sup>57</sup>.

## Environmental costs

Hybrid creativity is a cultural process that also rests on a physical base: the mining of materials for hardware, the paid and unpaid human labor of data annotation and moderation, the land and water that data centers occupy, and the electrical grid that supplies them<sup>88</sup>. Training large models requires a massive amount of energy<sup>89,90</sup>, and everyday inference now dominates deployment cost, with multi-purpose generative models orders of magnitude more expensive per query than taskspecific ones<sup>91</sup>. Given widespread adoption, data-center electricity, carbon footprint and water consumption are all predicted to rise<sup>92–</sup> <sup>94</sup>. The cost of a creative act has become detached from the person performing it, sitting instead in shared infrastructure that runs continuously, and its consequences fall unevenly across populations<sup>95,96</sup>. A cost of this size is not a necessary feature of hybrid work, though, since smaller, task-specific models can do much of the same work with a fraction of the energy<sup>91</sup>.

## 6 FUTURE DIRECTIONS

The evidence reviewed here converges on one claim: AI changes the structure and dynamics of creative work rather than feeding ideas into a process that stays fixed. Novice music producers using AI tools, for example, spend less time preparing and more time rapidly generating and selecting options<sup>67</sup>. That it helps individuals is by now well established; what remains open is which arrangements of people and machines preserve diversity in the aggregate.

Three priorities follow. First, the field needs population-scale designs, since nearly all evidence concerns individuals or dyads with a single tool and, as argued in Section 3, pooling isolated participants does not make a population. Studying many humans and many models together over multiple rounds is necessary before anyone can say whether local gains and global losses simply add up, though such designs are hard to scale along structural, interactional and individual axes<sup>21,23,97</sup>. Second, group composition should be treated as the independent variable: if mixed networks outperform homogeneous ones, the useful questions concern ratio, coupling and division of labor, connecting directly to the group-diversity results noted in Section 2<sup>28</sup>. Expertise belongs here as well, since professionals and laypeople extract different things from identical systems<sup>56</sup>. Humans and models do not even respond alike to the same intervention: cross-domain prompting reliably raises originality in people and leaves model output unchanged<sup>98</sup>. Finally, measurement is the binding constraint: the field still lacks standardized ways to assess creativity at the population level, and its tools under-support evaluation relative to generation<sup>45</sup>, so rigorous population-level diversity metrics are a prerequisite for the studies above.

All three run up against cost: the resource burden in Section 5 means environmental footprint should inform the research agenda itself, including when heavy generative methods are warranted at all.

In sum, the most productive reframing is to stop asking whether AI helps or harms creativity and to start asking which mixtures of humans and machines, coupled in which ways, allow individual gains to accumulate without eroding the collective diversity on which we depend.

## ACKNOWLEDGMENTS

We would like to thank all other participants of the "Collective Creativity" workshop held at Stony Brook University in April 2025: Kate Armstrong, Satyarth Arora, Brooke Belisle, Alex Doboli, Simona Doboli, Nick Harley, Nori Jacoby, Jordan Kodner, Owen Rambow, Lauren Ruiz, Oleg Sobchuk, Ofer Tchernichovski, and Nick Wilson.

## REFERENCES

1. Boden, M.A. (1998). Creativity and artificial intelligence. Artificial Intelligence 103. https://doi.org/10.1016/S0004-3702(98)00055-1.

2. Adam, D. (2023). The muse in the machine. Proceedings of the National Academy of Sciences 120. https://doi.org/10.1073/pnas. 2306000120.

3. Rafner, J., Beaty, R.E., Kaufman, J.C., Lubart, T., and Sherson, J. (2023). Creativity in the age of generative AI. Nature Human Behaviour 7. https://doi.org/10.1038/s41562-023-01751-1.

4. Rahwan, I., Cebrian, M., Obradovich, N., Bongard, J., Bonnefon, J.- F., Breazeal, C., Crandall, J.W., Christakis, N.A., Couzin, I.D., Jackson, M.O., et al. (2019). Machine behaviour. Nature 568. https://doi. org/10.1038/s41586-019-1138-y.

5. Tsvetkova, M., Yasseri, T., Pescetelli, N., and Werner, T. (2024). A new sociology of humans and machines. Nature Human Behaviour 8. https://doi.org/10.1038/s41562-024-02001-8.

6. Feist, G.J. (1998). A Meta-Analysis of Personality in Scientific and Artistic Creativity. Personality and Social Psychology Review 2. https://doi.org/10.1207/s15327957pspr0204\_5.

7. Guilford, J.P. (1950). Creativity. American Psychologist 5. https:// doi.org/10.1037/h0063487.

8. Jung, R.E., and Vartanian, O. (2018). The Cambridge Handbook of the Neuroscience of Creativity https://doi.org/10.1017/ 9781316556238.

9. Mumford, M.D. (2003). Where Have We Been, Where Are We Going? Taking Stock in Creativity Research. Creativity Research Journal 15. https://doi.org/10.1080/10400419.2003.9651403.

10. Muthukrishna, M., and Henrich, J. (2016). Innovation in the collective brain. Philosophical Transactions of the Royal Society B: Biological Sciences 371. https://doi.org/10.1098/rstb.2015.0192.

11. Merton, R.K. (1961). Singletons and multiples in scientific discovery: A chapter in the sociology of science. Proceedings of the American Philosophical Society 105, 470–486.

12. Wuchty, S., Jones, B.F., and Uzzi, B. (2007). The increasing dominance of teams in production of knowledge. Science 316, 1036– 1039. https://doi.org/10.1126/science.1136099.

13. Glăveanu, V.P. (2014). Distributed creativity: What is it? (Springer) https://doi.org/10.1007/978-3-319-05434-6\_1.

14. Ueshima, A., Jones, M.I., and Christakis, N.A. (2024). Simple autonomous agents can enhance creative semantic discovery by human groups. Nature Communications 15. https://doi.org/10. 1038/s41467-024-49528-y.

15. Centola, D., and Baronchelli, A. (2015). The spontaneous emergence of conventions: An experimental study of cultural evolution. Proceedings of the National Academy of Sciences 112, 1989–1994. https://doi.org/10.1073/pnas.1418838112.

16. Derex, M., Beugin, M.-P., Godelle, B., and Raymond, M. (2013). Experimental evidence for the influence of group size on cultural complexity. Nature 503. https://doi.org/10.1038/nature12774.

17. Rand, D.G., Arbesman, S., and Christakis, N.A. (2011). Dynamic social networks promote cooperation in experiments with humans. Proceedings of the National Academy of Sciences 108, 19193– 19198. https://doi.org/10.1073/pnas.1108243108.

18. Centola, D. (2022). The network science of collective intelligence. Trends in Cognitive Sciences 26. https://doi.org/10.1016/j.tics. 2022.08.009.

19. Kameda, T., Toyokawa, W., and Tindale, R.S. (2022). Information aggregation and collective intelligence beyond the wisdom of crowds. Nature Reviews Psychology 1, 345–357. https://doi.org/10. 1038/s44159-022-00054-y.

20. Brinkmann, L., Baumann, F., Bonnefon, J.-F., Derex, M., Müller, T.F., Nussberger, A.-M., Czaplicka, A., Acerbi, A., Grifiths, T.L., Henrich, J., et al. (2023). Machine culture. Nature Human Behaviour 7. https://doi.org/10.1038/s41562-023-01742-2.

21. Burton, J.W., Lopez-Lopez, E., Hechtlinger, S., Rahwan, Z., Aeschbach, S., Bakker, M.A., Becker, J.A., Berditchevskaia, A., Berger, J., Brinkmann, L., et al. (2024). How large language models can reshape collective intelligence. Nature Human Behaviour 8. https://doi.org/10.1038/s41562-024-01959-9.

22. Shirado, H., and Christakis, N.A. (2017). Locally noisy autonomous agents improve global human coordination in network experiments. Nature 545. https://doi.org/10.1038/nature22332.

23. Shiiku, S., Marjieh, R., Grifiths, T.L., Anglada-Tort, M., and Jacoby, N. (2026). Hybrid human-AI societies balance creativity and diversity. https://doi.org/10.31234/osf.io/wzh2x\_v1.

24. Thelle, N.J.W., and Fiebrink, R. (2022). How do musicians experience jamming with a co-creative “AI”?. In NeurIPS 2022 Workshop on Machine Learning for Creativity and Design https://neuripsc reativityworkshop.github.io/2022/papers/ml4cd2022/\_paper12. pdf.

25. Guzdial, M., Liao, N., Chen, J., Chen, S.-Y., Shah, S., Shah, V., Reno, J., Smith, G., and Riedl, M.O. (2019). Friend, collaborator, student, manager: How design of an AI-driven game level editor affects creators. In Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems, pp. 1–13. https://doi.org/10.1145/ 3290605.3300854.

26. Glickman, M., and Sharot, T. (2025). How human-AI feedback loops alter human perceptual, emotional and social judgements. Nature Human Behaviour 9. https://doi.org/10.1038/s41562-024-02077- 2.

27. Li, C., Marjieh, R., Hu, H., Steyvers, M., Collins, K.M., Sucholutsky, I., and Jacoby, N. (2026). Human-AI synergy supports collective creative search. https://doi.org/10.48550/arXiv.2602.10001.

28. Hong, L., and Page, S.E. (2004). Groups of diverse problem solvers can outperform groups of high-ability problem solvers. Proceedings of the National Academy of Sciences 101, 16385–16389. https://doi.org/10.1073/pnas.0403723101.

29. Davis, N., Sherson, J., and Rafner, J. (2025). The co-creative design framework for hybrid intelligence. In Proceedings of the 2025 Conference on Creativity and Cognition, pp. 560–572. https://doi. org/10.1145/3698061.3726934.

30. Mesoudi, A., and Thornton, A. (2018). What is cumulative cultural evolution?. Proceedings of the Royal Society B 285. https://doi. org/10.1098/rspb.2018.0712.

31. Boden, M.A. (2009). Computer models of creativity. AI Magazine 30. https://doi.org/10.1609/aimag.v30i3.2254.

32. Lee, B.C., and Chung, J.(. (2024). An empirical investigation of the impact of ChatGPT on creativity. Nature Human Behaviour 8. https://doi.org/10.1038/s41562-024-01953-1.

33. Meincke, L., Nave, G., and Terwiesch, C. (2025). ChatGPT decreases idea diversity in brainstorming. Nature Human Behaviour 9. https://doi.org/10.1038/s41562-025-02173-x.

34. Doshi, A.R., and Hauser, O.P. (2024). Generative AI enhances individual creativity but reduces the collective diversity of novel content. Science Advances 10. https://doi.org/10.1126/sciadv.adn 5290.

35. Anderson, B.R., Shah, J.H., and Kreminski, M. (2024). Homogenization effects of large language models on human creative ideation. In Creativity and Cognition, pp. 413–425. https://doi.org/10.1145/ 3635636.3656204.

36. Miu, E., Gulley, N., Laland, K.N., and Rendell, L. (2018). Innovation and cumulative culture through tweaks and leaps in online pro-

gramming contests. Nature Communications 9. https://doi.org/ 10.1038/s41467-018-04494-0.

37. Sourati, J., and Evans, J.A. (2023). Accelerating science with human-aware artificial intelligence. Nature Human Behaviour 7. https://doi.org/10.1038/s41562-023-01648-z.

38. Shin, M., Kim, J., Van Opheusden, B., and Grifiths, T.L. (2023). Superhuman artificial intelligence can improve human decisionmaking by increasing novelty. Proceedings of the National Academy of Sciences 120.

39. Lake, B.M., Ullman, T.D., Tenenbaum, J.B., and Gershman, S.J. (2017). Building machines that learn and think like people. Behavioral and Brain Sciences 40. https://doi.org/10.1017/S0140525X 16001837.

40. Goldstein, J.L. (2024). The boogie-woogie approach to creativity in art and science. Proceedings of the National Academy of Sciences 121. https://doi.org/10.1073/pnas.2413304121.

41. Brinkmann, L., Eisenmann, T.F., Nussberger, A.-M., Derex, M., Bonati, S., Chirkov, V., and Rahwan, I. (2026). Propagation and preservation of AI-discovered problem-solving strategies in human culture. Nature Communications 17. https://doi.org/10.1038/ s41467-026-76113-2.

42. Huang, C.-Z.A., Koops, H.V., Newton-Rex, E., Dinculescu, M., and Cai, C.J. (2020). AI Song Contest: Human-AI Co-Creation in Songwriting. https://doi.org/10.48550/arXiv.2010.05388.

43. DiStefano, P.V., Zeitlen, D., Rafner, J., De Chantal, P.L., Peng, A., Miller, S., and Beaty, R. (2025). Evaluating AI’s ideas: The role of individual creativity and expertise in human-AI co-creativity. https://doi.org/10.31234/osf.io/k2u87\_v1.

44. Hu, H., Marjieh, R., Collins, K.M., Li, C., Grifiths, T.L., Sucholutsky, I., and Jacoby, N. (2026). Why human guidance matters in collaborative vibe coding. https://doi.org/10.48550/arXiv.2602.10473.

45. Frich, J., MacDonald Vermeulen, L., Remy, C., Biskjaer, M.M., and Dalsgaard, P. (2019). Mapping the landscape of creativity support tools in HCI. In Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems, pp. 1–18. https://doi.org/10.1145/ 3290605.3300619.

46. Zamfirescu-Pereira, J.D., Wong, R.Y., Hartmann, B., and Yang, Q. (2023). Why Johnny can’t prompt: How non-AI experts try (and fail) to design LLM prompts. In Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems, pp. 1–21. https:// doi.org/10.1145/3544548.3581388.

47. Youngblood, M., Nusz, J., and Simon, J. (2026). Dynamics of collective creativity in AI art competitions. https://doi.org/10.48550/ arXiv.2605.17141.

48. Tennie, C., Call, J., and Tomasello, M. (2009). Ratcheting up the ratchet: On the evolution of cumulative culture. Philosophical Transactions of the Royal Society B: Biological Sciences 364.

49. Boyd, R., and Richerson, P.J. (2005). The origin and evolution of cultures.

50. Laland, K.N. (2004). Extending the extended phenotype. Biology and Philosophy 19.

51. Lazer, D., and Friedman, A. (2007). The network structure of exploration and exploitation. Administrative Science Quarterly 52, 667– 694. https://doi.org/10.2189/asqu.52.4.667.

52. Mason, W.A., Jones, A., and Goldstone, R.L. (2008). Propagation of innovations in networked groups. Journal of Experimental Psychology: General 137, 422–433. https://doi.org/10.1037/a0012798.

53. Derex, M., and Boyd, R. (2016). Partial connectivity increases cultural accumulation within groups. Proceedings of the National Academy of Sciences 113.

54. Kauffman, S.A., and Weinberger, E.D. (1989). The NK model of rugged fitness landscapes and its application to maturation of the immune response. Journal of Theoretical Biology 141, 211–245. https://doi.org/10.1016/S0022-5193(89)80019-0.

55. Epstein, Z., Hertzmann, A., Akten, M., Farid, H., Fjeld, J., Frank, M.R., Groh, M., Herman, L., Leach, N., Mahari, R., et al. (2023). Art and the science of generative AI. Science 380, 1110–1111. https:// doi.org/10.1126/science.adh4451.

56. Eisenmann, T.F., Karjus, A., Canet Sola, M., Brinkmann, L., Supriyatno, B.I., and Rahwan, I. (2026). Expertise Elevates AI Usage: Experimental Evidence Comparing Laypeople and Professional Artists. International Journal of Human–Computer Interaction 1. https://doi.org/10.1080/10447318.2026.2669041.

57. Wingström, R., Hautala, J., and Lundman, R. (2024). Redefining creativity in the era of AI? Perspectives of computer scientists and new media artists. Creativity Research Journal 36. https://doi. org/10.1080/10400419.2022.2107850.

58. Rezwana, J., and Maher, M.L. (2023). Designing creative AI partners with COFI: A framework for modeling interaction in human-AI co-creative systems. ACM Transactions on Computer-Human Interaction 30. https://doi.org/10.1145/3519026.

59. Lin, Z. (2026). Human–AI complementarity needs augmentation, not emulation. Nature Reviews Psychology 5, 228–229. https:// doi.org/10.1038/s44159-026-00536-3.

60. Geroimenko, V. (2025). Human-Computer Creativity: Generative AI in Education, Art, and Healthcare https://doi.org/10.1007/978-3- 031-86551-0.

61. Koivisto, M., and Grassini, S. (2023). Best humans still outperform artificial intelligence in a creative divergent thinking task. Scientific Reports 13, 13601. https://doi.org/10.1038/s41598-023- 40858-3.

62. Mohammadi, E., Thelwall, M., Cai, Y., Collier, T., Tahamtan, I., and Eftekhar, A. (2026). Is generative AI reshaping academic practices worldwide? A survey of adoption, benefits, and concerns. Information Processing & Management 63. https://doi.org/10.1016/j.ipm. 2025.104350.

63. Cetinic, E., and She, J. (2021). Understanding and creating art with AI: Review and outlook. https://doi.org/10.48550/arXiv.2102. 09109.

64. Fiebrink, R., and Caramiaux, B. (2016). The machine learning algorithm as creative musical tool. https://doi.org/10.48550/arXiv. 1611.00379.

65. Vigliensoni, G., and Fiebrink, R. (2024). Data- and interactiondriven approaches for sustained musical practices with machine

learning. Journal of New Music Research 53. https://doi.org/10. 1080/09298215.2024.2442361.

66. Cope, D. (1999). One approach to musical intelligence. IEEE Intelligent Systems and Their Applications 14. https://doi.org/10.1109/ 5254.769878.

67. Fu, Y., Newman, M., Going, L., Feng, Q., and Lee, J.H. (2025). Exploring the collaborative co-creation process with AI: A case study in novice music production. In Proceedings of the 2025 ACM Designing Interactive Systems Conference.

68. Wei, L., Yu, Y., Qin, Y., and Zhang, S. (2025). From Tools to Creators: A Review on the Development and Application of Artificial Intelligence Music Generation. Information 16. https://doi.org/10.3390/ info16080656.

69. Reimers, I., and Waldfogel, J. (2026). AI and the Quantity and Quality of Creative Products: Have LLMs Boosted Creation of Valuable Books?. https://doi.org/10.3386/w34777.

70. Xu, W., Jojic, N., Rao, S., Brockett, C., and Dolan, B. (2025). Echoes in AI: Quantifying lack of plot diversity in LLM outputs. Proceedings of the National Academy of Sciences 122. https://doi.org/10. 1073/pnas.2504966122.

71. Zhang, L., Wilson, K., and Amos, C. (2025). The rise of AI art: A look through digital artists’ eyes. First Monday 30. https://doi.org/10. 5210/fm.v30i4.13809.

72. Shibano, T. (2022). Fuseki Revolution: How AI Has Changed Go.

73. Beheim, B.A. (2025). Opening strategies in the game of go from feudalism to superhuman AI. Evolutionary Human Sciences 7, e28. https://doi.org/10.1017/ehs.2025.10016.

74. Wu, Z., Weber, T., and Müller, F. (2025). One does not simply meme alone: Evaluating co-creativity between LLMs and humans in the generation of humor. In Proceedings of the 30th International Conference on Intelligent User Interfaces, pp. 1082–1092. https:// doi.org/10.1145/3708359.3712094.

75. Simon, J. (2025). Lluminate: Creative exploration with reasoning LLMs. https://www.joelsimon.net/lluminate.

76. Bender, E.M., Gebru, T., McMillan-Major, A., and Shmitchell, S. (2021). On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? �. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, FAccT ’21, pp. 610– 623. https://doi.org/10.1145/3442188.3445922.

77. Birhane, A., and Guest, O. (2021). Towards Decolonising Computational Sciences. https://doi.org/10.7146/kkf.v29i2.124899.

78. Taylor, J., Mire, J., DeVrio, A., Sap, M., Zhu, H., and Fox, S.E. (2026). “I Just Don’t Want My Work Being Fed Into The AI Blender”: Queer Artists on Refusing and Resisting Generative AI. https://doi.org/ 10.48550/arXiv.2604.14266.

79. Hertzmann, A. (2018). Can computers create art?. Arts 7. https:// doi.org/10.3390/arts7020018.

80. McCormack, J., Gifford, T., and Hutchings, P. (2019). Autonomy, authenticity, authorship and intention in computer generated art. In Computational Intelligence in Music, Sound, Art and Design https://doi.org/10.1007/978-3-030-16667-0\_3.

81. Zeilinger, M. (2021). Generative adversarial copy machines. 20. https://culturemachine.net/vol-20-machine-intelligences/ generative-adversarial-copy-machines-martin-zeilinger/.

82. Baum, J., and Villasenor, J. (2024). Rendering misrepresentation: Diversity failures in AI image generation. https://www.brookings. edu/articles/rendering-misrepresentation-diversity-failures-inai-image-generation/.

83. Small, Z. (2023). Black artists say A.I. shows bias, with algorithms erasing their history. https://www.nytimes.com/2023/07/ 04/arts/design/black-artists-bias-ai.html.

84. Yang, Y. (2025). Racial bias in AI-generated images. https://doi. org/10.1007/s00146-025-02282-1.

85. Acerbi, A., and Stubbersfield, J.M. (2023). Large language models show human-like content biases in transmission chain experiments. Proceedings of the National Academy of Sciences 120. https://doi.org/10.1073/pnas.2313790120.

86. Vicente, L., and Matute, H. (2023). Humans inherit artificial intelligence biases. Scientific Reports 13. https://doi.org/10.1038/s 41598-023-42384-8.

87. Faiella, A., Zielińska, A., Karwowski, M., and Corazza, G.E. (2025). Am I still creative? The effect of artificial intelligence on creative self-beliefs. The Journal of Creative Behavior 59.

88. Crawford, K. (2021). The Atlas of AI: Power, Politics, and the Planetary Costs of Artificial Intelligence https://doi.org/10.2307/j.ctv1 ghv45t.

89. Luccioni, A.S., Viguier, S., and Ligozat, A.-L. (2022). Estimating the Carbon Footprint of BLOOM, a 176B Parameter Language Model. https://doi.org/10.48550/arXiv.2211.02001.

90. Patterson, D., Gonzalez, J., Le, Q., Liang, C., Munguia, L.-M., Rothchild, D., So, D., Texier, M., and Dean, J. (2021). Carbon Emissions and Large Neural Network Training. https://doi.org/10. 48550/arXiv.2104.10350.

91. Luccioni, A.S., Jernite, Y., and Strubell, E. (2024). Power hungry processing: Watts driving the cost of AI deployment?. In The 2024 ACM Conference on Fairness Accountability and Transparency, pp. 85–99. https://doi.org/10.1145/3630106.3658542.

92. Guidi, G., Dominici, F., Gilmour, J., Butler, K., Bell, E., Delaney, S., and Bargagli-Stofi, F.J. (2024). Environmental Burden of United States Data Centers in the Artificial Intelligence Era. https://doi. org/10.48550/arXiv.2411.09786.

93. Agency, I.E. (2025). Energy and AI. https://www.iea.org/reports/ energy-and-ai.

94. Vries-Gao, A.d. (2026). The carbon and water footprints of data centers and what this could mean for artificial intelligence. Patterns 7. https://doi.org/10.1016/j.patter.2025.101430.

95. Anand, P., and Coeckelbergh, M. (2026). AI, climate change and justice: Elements for a normative framework centring the Global South. AI and Ethics 6. https://doi.org/10.1007/s43681-025- 00949-5.

96. Li, P., Yang, J., Wierman, A., and Ren, S. (2024). Towards Environmentally Equitable AI via Geographical Load Balancing. https:// doi.org/10.48550/arXiv.2307.05494.

97. Sucholutsky, I., Collins, K.M., Jacoby, N., Thompson, B.D., and Hawkins, R.D. (2025). Using LLMs to advance the cognitive science of collectives. https://doi.org/10.48550/arXiv.2506.00052.

98. Liu, Q.E., Dubova, M., Conklin, H., Harada, T., and Grifiths, T.L. (2026). Assessing the effect of cross-domain mapping on creativity in humans and large language models. arXiv.