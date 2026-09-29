# Environmental requirements for the use of social information by artificial life agents using evolved plastic artificial neural networks.

Hugh Charterton<sup>1∗</sup>, James M. Borg<sup>1</sup>, Aniko Ek´ art´ <sup>1</sup>

<sup>1</sup>Aston Centre for AI Research and Application, Aston University, Birmingham, United Kingdom <sup>∗</sup>240403850@aston.ac.uk

## Abstract

Evolved Plastic Artificial Neural Networks (EPANNs) consist of two principal processes, the first, evolution, and the second, development and in-life learning. In the context of the origins of social learning, very few studies have been carried out using ALIFE models based on EPANN requirements. Studies in this field have usually involved an imitative teacher/pupil relationship. This, however, ignores the possibility that the observed behaviour is a consequence of social information cues rather than direct imitation or teaching.

Starting with the first of the EPANN processes (evolution), a series of experiments was undertaken using artificial neural network (ANN) based agents in a variety of foraging environments to examine under what minimal environmental conditions the use of social information might have evolved, as measured by the number of generations taken to meet a specified fitness criterion. NEAT (Neuroevolution of Augmenting Topologies) was the ANN used as its evolutionary algorithm would evolve a network’s topology as well its weights.

Unintentionally, in the experiment there was a simple network topology based on the location of the nearest food item which enabled agents to swiftly meet the fitness criterion. With this topology, additional information, social or otherwise, was not required and could have proved to be a hindrance. However, this does indicate that for the use of social information to have evolved, it would require a greater degree of complexity in the environment to do so.

Submission type: Full Paper

Code available at: https://osf.io/kugnb/ files/osfstorage

## Introduction

Soltoggio et al. introduced the concept of Evolved Plastic Artificial Neural Networks (EPANNs) in 2018, which they defined as computationally based systems inspired by the evolutionary processes observed in the natural world which “employ simulated evolution in-silico to breed plastic neural networks with the aim to autonomously design and create learning systems”.

The aims, properties and evolutionary algorithmic requirements for an EPANN were outlined and EPANNrelated progress to date in a variety of related fields such as plasticity (Soltoggio, 2008), evolutionary robotics (Nitschke et al., 2012), evolved learning (Bullinaria, 2009) and neuromodulation (Soltoggio et al., 2008) reviewed. The question as to what environmental conditions, natural or computational, might promote the evolution of neural plasticity, learning and intelligence was also raised.

The final section of Soltoggio et al. (2018) reviewed potential directions for EPANNs, one such being “Incremental and social learning” where it was suggested that an area where incremental learning might find a role is in social learning where an EPANN learns about its environment through communication, observation or imitation of other EPANN instances.

Both of the examples cited by Soltoggio et al. (2018), (McQuesten and Miikkulainen, 1997; Bullinaria, 2017) used forms of imitation as the means of knowledge transfer between individuals, as have other subsequent Artificial Life investigations and social learning models (Jolley et al., 2016; Bartoli et al., 2020; Borg et al., 2011; Bourahla et al., 2022), where knowledgeable individuals make that knowledge available to others based on a teacher/pupil, parent/child relationship, though that knowledge would have to, at some point, been gained through experiential learning (Heyes, 1994; Laland, 2004).

However, Noble and Todd (2002) have questioned the perceived prevalence of imitative behaviour in nature. It was argued that what was often interpreted as imitative learning could quite easily have been the result of an observational heuristic based on social information cue. An experiment in which rhesus monkeys, which had previously shown no reaction to the presence of a snake, only exhibited a fearful response once they had heard the fearful response of a monkey who had had previous experience of snakes (Mineka and Cook, 2013). While this might appear imitative, it was their ability to recognize the fear cue that allowed them to subsequently respond fearfully to the presence of a snake, their heuristic in this case being “Pay attention to what others are doing or experiencing, and if the results for them appear to be good or bad then learn from this”, or more simply, watch what others are doing and do likewise.

Similarly in the case of Norway rats which, despite it being poisonous, would eat a new food they had smelt on the breath of another (Galef Jr, 1996). The heuristic for the rat being “Pay attention to what others are eating and do likewise”, this being based on the observation that the other rat had not died.

The question we are looking to answer in this paper is how might this use of social information have evolved, the hypothesis being that using social information gives an evolutionary advantage resulting in improved performance in a foraging task?

To this end we used an agent-based artificial life model which used a minimally configured EPANN as the basis for each agent rather than a pre-defined heuristic or learning algorithm. Agents of varying capabilities were given a number of foraging environments of differing environmental conditions and allowed to evolve sustainable solutions over a fixed number of generations.

Although environments and technology are not the same, these experiments explore a territory similar to that of Borg and Channon (2020) who showed that social information without learning within lifetime could be useful until insurmountably challenging environments were encountered.

## Technical Background

NEAT (Neuroevolution of Augmenting Topologies) (Stanley and Miikkulainen, 2002) was cited by Andrea Soltoggio et al. 2018 as a prime example of an EPANN’s evolutionary algorithm since it managed not only the mutation and recombination of network structures (nodes, connections, layers and weighting), but also allowed diverse network structures to evolve and grow.

However NEAT, as with other EAs, does not incorporate the plasticity required to support the evolution of an emergent learning capability and while the investigation covered in this paper uses NEAT as its evolutionary algorithm, it did so without enhancing it to include such within lifetime plasticity.

The following experiment was carried out using NEATpython <sup>1</sup> and it should be noted that there are documented differences in implementation to the original (CodeReclaimers, 2024). It should also be noted that the reproductive process takes place at the end of a generational cycle, as defined by a number of timesteps, with the next generation of each species being derived from both a set number of its fittest members and a random selection of other individuals.

The configuration used in this work generally follows the default NEAT parameter settings with the exceptions found in table 1.

<table><tr><td>Configuration Item</td><td>Description and setting</td></tr><tr><td>pop_size</td><td>Number of agents (dynamically set as per the run requirements).</td></tr><tr><td>num_inputs</td><td>Number of network inputs (see table 5 for values).</td></tr><tr><td>feed_forward</td><td>Network configuration. Set to false to enable recurrent connectivity.</td></tr><tr><td>activation_default</td><td>TANH select as the node activation function as it has a sharp response in the range [−1, 1] corresponding to movements along an axis, providing forward/backward, left/right move- ment.</td></tr><tr><td></td><td>activation_options List of other activation functions that could be used. Set to TANH for consistency.</td></tr><tr><td>fitness_threshold</td><td>Set to 150 as derived through initial trials and based on agents’ energy value at which they were observed to be self-sustaining.</td></tr></table>

Table 1: NEAT configuration values (where different from default).

## Experimental Setup

A varying number of NEAT-based agents were randomly placed in a rectangular foraging environment measuring 800 × 600 pixels as were a varying number of food and poison items of different ratios (see table 2 for agent and item numbers, and item ratios used).

The agents were sized at 20 × 20 pixels and the items at 15 × 15 pixels. Energy and fitness had different counters so that an agent’s mortality (an agent was deemed to have died if its energy value reached zero, and was removed) and its progression towards being self-sustaining could be tracked separately. If an agent overlapped an item, the item would automatically be consumed with an energy increase of (+50) for food or an energy decrease of (−50) for poison. Each agent had an energy starting point of +100. This consumption in turn mapped to the agent’s fitness value which, starting from 0, increased or decreased by 10.

An energy loss of 0.1 was incurred for each of the 1000 timesteps in a generation. Consequently, an agent would die at the end of a generation if it did nothing. Each movement also incurred an energy loss of 0.1. These losses were not reflected in the fitness value since they did not reflect the agent’s progress towards its fitness goal.

<table><tr><td>Environment Variables</td><td>Value a</td><td>Value b</td><td>Value C</td><td>Value d</td><td>Value e</td></tr><tr><td>Agent Pop- ulation</td><td>30</td><td>40</td><td>50</td><td>60</td><td>70</td></tr><tr><td>Total Food/Poison Items</td><td>40</td><td>45</td><td>50</td><td>55</td><td>60</td></tr><tr><td>Food/Poison Ratio</td><td>1:1</td><td>1.3:1</td><td>1.8:1</td><td>2:1</td><td></td></tr></table>

Table 2: Variable values for agent population, total number of food/poison items and the varying food/poison ratios. During a run, additional food/poison items would re-spawn if a random number between zero and one was less than 0.2 times the food/poison ratio.

In addition, inspired by the bioluminescence found throughout nature, each agent would have a beacon. Configured to reflect an agent’s energy, the beacon would act as a potential source of social information from which other agents could potentially infer a strategy for foraging in the environment based on that agent’s energy health.

An agent could therefore interact with inputs from three types of artifact, food/poison items, other agents and itself (e.g. energy). However, only information which was within an agent’s area of visibility would be available to it, with visibility defined as being within a quarter of the length of the environment’s diagonal. Information on each of the three artifact types would make up possible inputs to an agent’s NEAT instance as described in table 3.

As the information provided by another agent did not give information about a potential energy source, it was deemed to provide indirect information, while information pertaining to a potential energy source was deemed to provide direct information, giving each agent three types of possible interaction with the environment as shown in table 4. Each interaction would be composed of appropriate combinations of NEAT inputs (See table 5) with each combination forming the basis for a set of runs across each combination of the environment configuration variables.

All agent NEAT networks had two outputs which were used to update its position in the environment relative to its current position at the end of each timestep, as shown in table 6.

20 runs over 120 generations, each of 1000 timesteps were executed for each input code combination across each combination of the environment configuration variables, a total

<table><tr><td>Input Code</td><td>Description</td><td># NEAT Inputs</td></tr><tr><td>FP_XY</td><td>XY co-ordinates of the nearest visible Food/Poison item relative to agent.</td><td>2</td></tr><tr><td>FP_IND</td><td>Indicator of whether an item is 1 either food or poison.</td><td></td></tr><tr><td>AG_EN</td><td>Normalized value (n) of agent&#x27;s 1 own energy (e) as calculated in equation 1</td><td></td></tr><tr><td rowspan="2"></td><td> $n = m a x ( 0 . 0 , m i n ( 1 . 0 , e / f ) )$  (1)</td><td></td></tr><tr><td>where f refers to fitness thresh- old in units of energy</td><td></td></tr><tr><td>NA_XY NA_B</td><td>XY co-ordinates of other agent relative to agent with brightest beacon as calculated using equa- tion 2</td><td>2</td></tr><tr><td rowspan="2"></td><td>Beacon value (b&#x27; in equation 2). where&#x27;refers to other agent e&#x27; refers to other agent&#x27;s energy x&#x27; refers to other agent&#x27;s x axis co-ordinate y&#x27; refers to other agent&#x27;s y axis</td><td>1</td></tr><tr><td>co-ordinate:  $b ^ { \prime } = \frac { e ^ { \prime } } { s q r t ( ( y ^ { \prime } - y ) ^ { 2 } + ( x ^ { \prime } - x ) ^ { 2 } ) }$ </td><td></td></tr></table>

Table 3: Inputs, descriptions and the number of NEAT inputs required.

of 20, 000 runs.

At the end of each generation, the initial variable values, outcome with respect to fitness threshold and the network structure of the fittest agent were logged for subsequent analysis.

## Results

The effectiveness of any environmental configuration was measured in terms of the number of times the fitness threshold was met per configuration. As might be expected, the number of thresholds being met increased as the ratio of food to poison items increased with similar numbers across all food/poison item totals as shown in figure 1.

Variations in the population had no discernible impact other than a steady overall decrease in effectiveness as the population grew in spite of the higher volumes of food/poison items as shown in figure 2.

<table><tr><td>Interaction direct/indirect Id</td><td>Description</td></tr><tr><td>1 direct</td><td>Information about item and the agent its self.</td></tr><tr><td>2 indirect</td><td>Information about another agent.</td></tr><tr><td>3 direct &amp; indirect</td><td>Combined direct and indi- rect information</td></tr></table>

Table 4: Direct/Indirect Interaction types.

<table><tr><td>NEAT input combi- nation ID</td><td>Inter- action ID</td><td>Input codes</td><td>Total Inputs</td></tr><tr><td>1</td><td>1</td><td>FPXY</td><td>2</td></tr><tr><td>2</td><td>1</td><td>FP_XY, FP_IND</td><td>3</td></tr><tr><td>3</td><td>1</td><td>FP_XY, AG_EN</td><td>3</td></tr><tr><td>4</td><td>1</td><td>FP_XY, FP_IND, AG_EN</td><td>4</td></tr><tr><td>5</td><td>2</td><td>NA_XY, NA_B</td><td>3</td></tr><tr><td>6</td><td>2</td><td>NA_XY, NA_B, AG_EN</td><td>4</td></tr><tr><td>7</td><td>3</td><td>FP_XY, NA_XY, NA_B</td><td>5</td></tr><tr><td>8</td><td>3</td><td>FP_XY, FP_IND, NA_XY,</td><td>6</td></tr><tr><td>9</td><td>3</td><td>NA_B FP_XY, AG_EN, NA_XY,</td><td>6</td></tr><tr><td>10</td><td>3</td><td>NA_B FP_XY, FP_IND, AG_EN, NA_XY, NA_B</td><td>7</td></tr></table>

Table 5: NEAT input combinations, grouped by their type of interaction with the environment and their total number of inputs to a NEAT neural network. Please note that combination 6 has been classified as indirect since the agent’s energy is viewed as complementary to the information about another agent.
<table><tr><td>Output code</td><td>Description</td><td># Outputs</td></tr><tr><td>X</td><td>Movement along X axis</td><td>1</td></tr><tr><td>Y</td><td>Movement along Y axis</td><td>1</td></tr></table>

Table 6: NEAT neural network output values. As NEAT would return values between -1 and 1, a speed value was used to map the returned value to give the number of pixels to move. The speed value was set to 5.

![](images/6c697ecfcd4a356c021ff44c20743c7a774d5e19778105fd38a2ab439795fb62.jpg)  
Figure 1: Number of fitness thresholds met for each ratio of food/poison items for each initial total of food/poison items in the environment across all populations and all input code combinations.

![](images/75ed0a31fc2c3cf62e68198fc26ec5dbf7c817fbed06e826199f2df5e0fc1803.jpg)  
Figure 2: Number of fitness thresholds met across the range of population values for each initial total of food/poison items in the environment across all food/poison ratios and all input code combinations.

When reviewing the fitness threshold results by NEAT input combination alone, they divided into their interaction types with those combinations which interacted directly with the environment (NEAT input combinations 1 – 4) achieving the greatest number of thresholds met while those interacting directly and indirectly (NEAT input combinations 7 – 10) came second. Those which only interacted indirectly with the environment (NEAT input combinations 5 & 6) hardly ever met the required thresholds, as can be seen in figure 3.

A subsequent set of 50 runs each of 1000 generations was carried out for the indirect NEAT input combinations over a population of 40 agents with a food/poison item total of 55 and a food/poison ratio of 1.3 : 1, with the aim of looking into the possibility that additional generations might allow a higher number of thresholds to be met. However, no thresholds were met at all.

![](images/0409994cce2cd3e1730563a5ca3d924a6231cd781e7b7112cb1f27d6be5efc77.jpg)  
Figure 3: Fitness thresholds meeting the termination criteria for each NEAT input combination. For NEAT input combinations, see table 5.

One possible reason that the direct NEAT input combinations performed best might have been that there was a simple network configuration based only on the co-ordinates of the nearest food/poison item which provided the agent with a survival solution such that it could ignore other inputs and information be it additional direct environmental information or indirect social information. For instance, between 9 and 11 percentage of each of the direct input code combinations meeting the required threshold did so in the first generation as shown in figure 4. Reviewing the network configurations from the first generation threshold-meeting outcomes showed that only a network topology featuring direct connections between the FP XY input, as defined in table 3, and the two output nodes was required, as illustrated in the examples in figure 5.

![](images/cac2afe1cdc7b4388f1c7bafd23eaeb99be12617ffa62f7465f946eb71742be1.jpg)  
Figure 4: Percentage of thresholds met in their first generation out of the total number of thresholds met per input code combination. where 9-11% of each of the direct NEAT input combinations meets the threshold in their first generation, a strong indicator that this environment might have been too simple. Those with combined direct and indirect NEAT input combinations meet the threshold approximately 1% of the time in their first generation, while the indirect combinations never reached the threshold in their first generation. For details on each input code combination, see table 5.

![](images/3dc8881cbce8beac0447c35526a50a55b43f1249acf3cb0927577feff18c8dfc.jpg)  
(a) NEAT network topology for NEAT input combination 1 (as defined in table 5)

![](images/31a85e5264d8d4a15573626b5402ba59289b490ee0b358e1ac9e25066ef5a079.jpg)  
(b) NEAT network topology for NEAT input combination 3 (as defined in table 5)  
Figure 5: Examples of first generation NEAT topologies which reached the fitness threshold in which NEAT inputs connect directly to the outputs with no hidden layers. Both examples had initial agent populations of 30, initial number of food/poison items of 55 with a ratio of 2 : 1.

Comparing the total number of runs where the fitness threshold was met for each NEAT input combination across all generations and for each population number, total food/poison items and their ratios, shows a distinct split between the interaction types as illustrated in figure 6 which shows the cumulative total of thresholds met for each of the input code combinations over 120 generations for each population, number of food/poison items and food/poison ratios.

While the NEAT input combinations for each interaction type follow the same trajectory, it was noted that the leading combination for each interaction type was the one with the least number of NEAT network inputs.

![](images/b14649d0ae59a42fe20d67bb1d81e989703d325b343953d62aea66da970442b7.jpg)  
Figure 6: Cumulative number of fitness thresholds met per input code combination as defined in table 5.

Figure 6 also shows that the number of thresholds being met by the direct NEAT input combinations initially increases at a faster rate than those for the combined direct and indirect combinations, while the indirect combinations hardly register at all.

## Conclusion

This study has been carried out with only half an EPANN, in that only an evolutionary component, in the form of NEAT, was deployed, in runs of a short generational time span, across environments with a varying number of agents and food/poison items of differing ratios. The only environmental change during a generational lifetime was the reduction and subsequent re-spawning of items due to agent consumption.

Given that 10% of the direct input combinations had a solution before NEAT’s evolutionary algorithm was invoked suggests that the environment was too simple to provide evidence as to whether social information improves performance in an EPANN/NEAT based foraging environment. Where this was not the case, the number of thresholds being met decreased as the number of NEAT inputs increases, as can be seen in figure 3. It is therefore unlikely that the addition of an in-life learning capability would, in this instance, have made any difference. However, it should be recognized that the availability of simple solutions should not automatically be seen adversely as they can provide efficient solutions as demonstrated by Seth (1998).

One variable which was not altered was that of the impact of food/poison on agents. Varying this value, along with other environment changes as listed below might increase the environment’s complexity such that, on its own, the nearest item’s co-ordinates no longer provided an easy solution.

• Have an agent decide whether to consume an item or not.

• Have agents react when they encounter another agent, currently they ignore each other.

• Vary the impact of food/poison items on agents.

• Have food/poison items distributed in clusters.

• Have food/poison items which are not consumed in one timestep.

• Have seasonal item availability over a longer generational timespan.

Along with additional logging of movement and proximity data of other agents, it might then be possible for the beneficial use of a social information input to evolve and be recognized. A further step would be to replicate the Borg and Channon (2020) environment using NEAT and make a comparison with the original results.

Social information has been defined as “information derived from the behaviours, actions, cues or signals of other agents” (Borg, 2018) and as such the beacon used in the experiment would qualify as providing social information. However, since the beacon is solely a reflection of an agent’s energy level over which the agent has no control, perhaps it better corresponds to being defined as public information since it is “inadvertent social information” (Danchin et al., 2004).

If, however, the agent manipulated its beacon value before it was visible to other agents, then it could be deemed social information, since the agent would have control over what value, if any, the beacon made available. One way this could be achieved would be through an additional NEAT output which would modulate the beacon value before it became available to other agents.

The corollary of a modulated beacon would be a capability to decode a beacon value which could be provided through neuromodulated (Soltoggio et al., 2007; Barnes et al., 2020) within-lifetime learning, the other half of an EPANN.

While the question of how the use of social information might have evolved has not been answered in this experiment, it has shown that it is unlikely to have emerged if other simpler survival mechanisms are available. The requisite environmental complexity remains to be explored.

## References

Barnes, C. M., Ekart, A., Ellefsen, K. O., Glette, K., Lewis, P. R., ´ and Tørresen, J. (2020). Coevolutionary learning of neuromodulated controllers for multi-stage and gamified tasks. In 2020 IEEE International Conference on Autonomic Computing and Self-Organizing Systems (ACSOS), pages 129–138. IEEE.

Bartoli, A., Catto, M., De Lorenzo, A., Medvet, E., and Talamini, J. (2020). Mechanisms of social learning in evolved artificial life. In Artificial Life Conference Proceedings 32, pages 190– 198. MIT Press.

Borg, J. M. (2018). The emergence and utility of social behaviour and social learning in artificial evolutionary systems. PhD thesis, Keele University Staffordshire.

Borg, J. M. and Channon, A. (2020). The effect of social information use without learning on the evolution of social behavior. Artificial Life, 26(4):431–454.

Borg, J. M., Channon, A., and Day, C. (2011). Discovering and maintaining behaviours inaccessible to incremental genetic evolution through transcription errors and cultural transmission. In Proceedings of the eleventh European conference on the synthesis and simulation of living systems, volume 101, page 108.

Bourahla, Y., Atencia, M., and Euzenat, J. (2022). Knowledge transmission and improvement across generations do not need strong selection. In AAMAS 2022-21st ACM international conference on Autonomous Agents and Multi-Agent Systems, pages 163–171. ACM.

Bullinaria, J. A. (2009). Lifetime learning as a factor in life history evolution. Artificial Life, 15(4):389–409.

Bullinaria, J. A. (2017). Imitative and direct learning as interacting factors in life history evolution. Artificial Life, 23(3):374– 405.

CodeReclaimers (2024). Neat-python.

Danchin, E., Giraldeau, L.-A., Valone, T. J., and Wagner, R. H. <sup>´</sup> (2004). Public information: from nosy neighbors to cultural evolution. Science, 305(5683):487–491.

Galef Jr, B. G. (1996). Social enhancement of food preferences in norway rats: A brief review. Social learning in animals: The roots ofculture.

Heyes, C. (1994). Social learning in animals: categories and mech anisms. Biological Reviews, 69:207–231.

Jolley, B. P., Borg, J. M., and Channon, A. (2016). Analysis of social learning strategies when discovering and maintaining behaviours inaccessible to incremental genetic evolution. In International Conference on Simulation of Adaptive Behavior, pages 293–304. Springer.

Laland, K. N. (2004). Social learning strategies. Learning & behavior, 32(1):4–14.

McQuesten, P. and Miikkulainen, R. (1997). Culling and teaching in neuro-evolution. In ICGA, pages 760–767.

Mineka, S. and Cook, M. (2013). Social learning and the acquisition of snake fear in monkeys. In Social learning, pages 51–73. Psychology Press.

Nitschke, G. S., Schut, M. C., and Eiben, A. E. (2012). Evolving behavioral specialization in robot teams to solve a collective construction task. Swarm and Evolutionary Computation, 2:25–38.

Noble, J. and Todd, P. M. (2002). Imitation or something simpler? modeling simple mechanisms for social information processing, page 423–439. MIT Press, Cambridge, MA, USA.

Seth, A. K. (1998). Evolving action selection and selective attention without actions, attention, or selection. In From animals to animats 5: Proceedings of the fifth international confer ence on simulation ofadaptive behavior, volume 5, page 139. MIT Press.

Soltoggio, A. (2008). Neural plasticity and minimal topologies for reward-based learning problems. In Proceeding of the 8th International Conference on Hybrid Intelligent Systems (HIS2008), pages 10–12.

Soltoggio, A., Bullinaria, J. A., Mattiussi, C., Durr, P., and Flore-¨ ano, D. (2008). Evolutionary advantages of neuromodulated plasticity in dynamic, reward-based scenarios. In Proceedings of the 11th international conference on artificial life (Al ife XI), page 569. MIT Press.

Soltoggio, A., Durr, P., Mattiussi, C., and Floreano, D. (2007). Evolving neuromodulatory topologies for reinforcement learning-like problems. In 2007 IEEE Congress on evolutionary computation, pages 2471–2478. IEEE.

Soltoggio, A., Stanley, K. O., and Risi, S. (2018). Born to learn: The inspiration, progress, and future of evolved plastic artifi cial neural networks. Neural Networks, 108:48–67.

Stanley, K. O. and Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. Evolutionary Computation, 10(2):99–127.