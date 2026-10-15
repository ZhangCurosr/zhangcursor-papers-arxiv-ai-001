# Integrating Simulation and Chain-of-thought Reasoning in Multimodal-Language Models For Physical Reasoning

Yingqiao Wang<sup>∗</sup> Department of Psychology, Harvard University

Sakana AI yingqiaowang@g.harvard.edu

Eric Bigelow Department of Psychology, Harvard University

Goodfire AI ebigelow@g.harvard.edu

Tomer Ullman Department of Psychology, Harvard University tullman@fas.harvard.edu

Yujin Tang   
Sakana AI   
yujintang@sakana.ai

Sakana AI sebastianrisi@sakana.ai

## Abstract

Multimodal language models often need to reason about the physical dynamics of the world in order to answer questions appropriately. Various methods have emerged for solving physical reasoning tasks, and in particular, two leading approaches are Chain-of-Thought reasoning and physics-grounded simulation. However, relatively little is known about which particular problems can be solved by each approach. Some tasks may benefit from computationally expensive generative video simulations, whereas others may be solved equally well (or even better) with language-based reasoning. Our work addresses this by systematically comparing simulation and language reasoning across a variety of multimodal tasks involving physical reasoning, such as answering questions about fluid dynamics and elasticity. We draw a parallel with this to resource-rational theories of human cognition, which propose that people optimally trade off simulation and reasoning depending on task demands. We present a novel cognitively-inspired hybrid model which optimally trades off simulation and reasoning to improve model performance and computational efficiency. Our model is able to decide the appropriate method to solve a physical reasoning problem at test time, and finds an optimal

balance between accuracy and cost. Further, our model can be trained on a smaller vision-language model that is only 4B parameters and deployed to improve performance on much bigger base models with >3T parameters like Gemini 3 Pro.

## 1 Introduction

Physical reasoning is an important topic in the current AI community, and it is needed to overcome important challenges like ensuring that embodied AI agents act safely in the world [23] [3]. There are different methods to implement such reasoning ability. Chain-of-Thought [7] [20] is one dominant method, used to solve many reasoning tasks, including physical reasoning. The recent SoTA physical reasoning models like [6] adapted the chain-of-thought as the only reasoning methods when reasoning about physical tasks. However, this might not be the only method to implement physical reasoning ability.

In cognitive science, an analogous line of research is called intuitive physics, which studies how human approaches physical reasoning. Current approaches suggests that mental simulation could be a way that people approach physical reasoning [2] [18]. Researchers in AI had also adapted this idea of mental simulation into grounding the physical reasoning in Multi-modal Large Language Models (MLLMs) [9]. Furthermore, recent breakthrough in diffusion model-based video model has shown emerging zero-shot ability as a physics simulator [21] and it’s ability to push towards intelligence in robotics [17].

In theory, Chain-of-thought is considered to be computationally cheaper when compared with simulation with physics simulator, but simulation is considered to be more accurate since it is grounded on real physics . This raises an interesting observation that whether each method would be better suited to solve specific types of physical reasoning tasks, thus creating this unique tradeoff between accuracy and computational costs in physical reasoning in MLLMs when different method was used.

Such dynamical selection based on tradeoff between accuracy and computational costs is actively adapted in areas like mathematical reasoning [19] and LLM Agentic Reasoning [22]. Given that reasoning effort is tunable in LLMs equipped with Chain-of-thought, these work have found that dynamically adjusting the reasoning effort based on the questions the model is facing could not only increase accuracy, but also manage computational cost to prevent wasting computational power. Furthermore, works like [10] have found that reasoning more does not help with all the problems. Over-reasoning might hurt the performance of the LLM on certain types of problems, and this finding aligns with human studies.

Therefore, to answer the questions that whether such dynamical method selection between simulation and chain-of-thought based on tradeoff between accuracy and computational costs applies to physical reasoning, we propose a model that not only empirically shows that such tradeoff between accuracy and computational costs exists, but also able to find the best tradeoff between accuracy and computational costs on various physical reasoning tasks, inspired by research in resource rationality in cognitive science.

## 2 Resource Rationality in Intuitive Physics

People are able to efficiently reason about the physical dynamics of everyday objects. For example, if you saw a ball flying towards you, you might quickly predict where to catch the ball. This ability of people being able to reason about physics of the world is called intuitive physics. Many competing theories have been proposed to explain this ‘intuitive physics’, and recent research have suggested that it is likely that humans use a combination of different methods to carry out this reasoning [5]. Interestingly, previous work in cognitive science have suggested a hybrid model that combines two most competing methods, which are mental simulation and heuristic reasoning [14] [15].

The hybrid models are designed around one core concept: resource rationality. In [14], they proposed the Integration of Simulation and Rules (ISR) framework based on resource rationality before performing a task. Before performing physical reasoning on a specific task T, we have to decide on a strategy S to answer this task. But, several factors that are associate with this strategy needs to be balanced and considered. There are mainly two factors to consider in the ISR: the metabolic costs C for adapting this strategy, and the expected utility U of this strategy, needs to be considered.

Therefore, the ISR framework aims to optimize for the best strategy $S ^ { * }$ that uses the resource in the most rational way as the following:

$$
S ^ { * } = \arg \operatorname* { m a x } E ( U ( S , T ) - C ( S , T ) )\tag{1}
$$

Furthermore, in [15], a hybrid model that could be applied when performing a task was proposed. When humans are performing physical reasoning, even though it is possible that they are adapting a simulation strategy when doing a physical reasoning task, some parts in the task that can be resolved by using heuristics rules could be used to save cognitive resource, achieving resource rationality. This idea of resource rationality in intuitive physics of human is useful for many cases, and serves as a motivation for our model. Resource rationality has been shown to be useful in intuitive physics for reducing the amount of computation humans need to perform for physical reasoning, while remaining accurate. This serves as inspiration for our model, which similarly trades off physica simulation against heuristic-based reasoning, to improve model accuracy while also reducing costs.”

## 3 Hybrid Simulation and Reasoning Model

![](images/8c84bba02eb476ac2613954295e0e0d238a101bbeb273d2c37804706a89b0704.jpg)  
Figure 1: The overview of HSR model. Our HSR model consists of simulation controller, physics simulator-based simulation, and Chain-of-thought reasoner. The inputs consist of a set of video and a question based on this video, which is passed into simulation controller. Depending on the assignment from the simulation controller, either simulation or Chain-of-thought reasoner will be applied to answer the question.

Here we present our model Hybrid Simulation and Reasoning Model (HSR) to incorporate resource rationality into physical reasoning. Consider an arbitrary physical reasoning task that consist of a problem s and a video d that the problem is based on, and two methods of solving physical reasoning tasks: simulator and Chain-of-thought reasoning. First, we determine which of the two methods is best suited to solve this physical reasoning tasks by using a trained simulation controller. Second, we pass the physical reasoning tasks to the method determined by the simulation controller to get an answer for this physical reasoning tasks.

## 3.1 Model Architecture

We first process the problem s and the video d into embeddings. For visual embedding, we only use the first frame $x _ { 1 }$ of the video d. We used visual embedding model and text embedding model to generate the corresponding embeddings. Therefore, we will be able to obtain a concatenated embedding h by:

$$
\mathbf { v } = f _ { v i s u a l } ( x _ { 1 } ) \in \mathbb { R } ^ { 5 1 2 } \qquad \mathbf { u } = f _ { t e x t } ( s ) \in \mathbb { R } ^ { 7 1 8 } \qquad \mathbf { h } = [ \mathbf { u } ; \mathbf { v } ] \in \mathbb { R } ^ { 1 2 3 0 }\tag{2}
$$

We pass the concatenated embedding h into the simulation controller, as shown in fig 1. The simulation controller is a trained MLP that that process the embedding h into a logit. We use a 2-hiddenlayer MLP $( 1 2 3 0 \to 5 1 2 \to 2 5 6 \to 1 )$ with BatchNorm (512, 256), ReLU, and dropout $( p = 0 . 5 )$ after each hidden layer to produce a scalar logit ℓ.

Where ℓ is the predicted logit from the model. The final predicted probability $p$ can be recovered by applying sigmoid function on logit:

$$
p = \sigma ( \ell ) = \frac { 1 } { 1 + e ^ { - \ell } }
$$

During the training phase, the model aims to minimize the BCE With Logit Loss, stated as the following:

$$
\mathcal { L } _ { \mathrm { B C E - l o g i t } } ( \ell , y ) = - \Big ( y \log p + ( 1 - y ) \log ( 1 - p ) \Big )\tag{3}
$$

## 3.2 Definition of Ground Truth Probability

During the training phase, ground truth probability y is provided for all data in the training set and is generated by the base VLM. Given that $y \in [ 0 , 1 ]$ , let $( \Omega , \mathcal { F } , \mathbb { P } )$ be a probability space. Based on this probability space, we define a random variable $X : \Omega \to \{ S , C \}$ where S denotes using simulation and C denotes using chain-of-thought only.

Let $0 \leq p _ { \mathrm { s i m } } , p _ { \mathrm { c o t } } \leq 1$ . Define $p _ { \mathrm { s i m } }$ and $p _ { \mathrm { c o t } }$ to be the probability of using simulation and the probability of using chain-of-thought only reasoning. For each data entry in the training data, it was answered by using simulation K times, and Chain-of-thought reasoning K times. Therefore:

$$
p _ { \mathrm { s i m } } = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \mathbb { I } \Big [ a _ { \mathrm { s i m } } ^ { ( i ) } = a _ { \mathrm { g t } } \Big ] \in [ 0 , 1 ] \qquad p _ { \mathrm { c o t } } = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \mathbb { I } \Big [ a _ { \mathrm { c o t } } ^ { ( i ) } = a _ { \mathrm { g t } } \Big ] \in [ 0 , 1 ]\tag{4}
$$

where $a _ { \mathrm { s i m } } ^ { ( i ) }$ is the answer output by simulation method in ith attempt, $a _ { \mathrm { c o t } } ^ { ( i ) }$ is the answer output by Chain-of-thought method in the ith attempt, and $a _ { \mathrm { g t } }$ is the ground truth answer. The indicator function allows partially correct answer.

We hereby define the ground truth probability y as:

$$
y \triangleq \mathbb { P } ( X = S ) = \frac { p _ { \mathrm { s i m } } } { p _ { \mathrm { s i m } } + p _ { \mathrm { c o t } } } ,
$$

## 3.3 Method Selection Threshold

We define a threshold $\tau _ { t }$ to be $\tau _ { t } = \tau _ { a c c u r a c y } + \tau _ { c o s t }$ <sub>t</sub>. All parameters τ<sub>t</sub>, τ<sub>accuracy</sub>, $\tau _ { t } .$ $\tau _ { c o s t }$ are tunable hyperparameter, but $\tau _ { a c c u r a c y }$ is determined by how the ground truth probability y is defined and constructed, meaning that it is usually set as $0 . 5 . \tau _ { c o s t }$ serves as a penalty term that penalizes the use of simulation due to the high metabolic cost of simulation.

If $p \leq \tau _ { t }$ we assign the data entry as Chain-of-thought reasoning, and if $p > \tau _ { t }$ then we assign the data entry as simulation. Notice that this threshold controls the number of data entries being assigned as simulation, thus controls the computational costs, since simulation is by definition more costly.

Then, as shown in fig 1, we pass the question and video into simulation or Chain-of-thought reasoning, depending on the decision made by the simulation controller to obtain the final answer to the question. In simulation, a fine-tuned video diffusion model is used as the physics-grounded simulator. We utilize a VLM for Chain-of-thought reasoning and readout phase in simulation.

## 3.4 Design of Video Simulator

In our HSR model, we implemented a compositional structure for the physics simulator we used in our model in order to make it support counterfactual simulation. First, a base VLM will takes a video and question and determines whether counterfactual simulation is needed. Then, if needed, the base VLM will propose the plan to construct the physics prior (conditional frames) needed for the video simulator to generate the video. In this plan, 1: The question will be rewritten to their noncounterfactual version and 2: the modifications will be suggested and passed to an image generation model for the modification. Then, the image generation model will modify the physics prior, and this physics prior will be passed to the physics simulator as the conditional prior for video generation. For the prompt that is used during this phase, see Appendix B.

![](images/4559a1f943d645dea251ab7c1db1500d6335b534be62b2ff6124a7f01d54a6a8.jpg)  
Figure 2: Evaluation results of simulation only (Simulation) and chain-of-thought only (Chain-of-Thought) when Gemini-3-Pro and Qwen3-VL-4B-Thinking was set as the base model. The evalu ation is based on fluid hourglass (FD) and bouncing ball (EC) domains, with questions in property (Property), predictive (Predictive), and counterfactual (Counterfactual) categories. We could observe that HSR performs near the best accuracy across all physical domains and question categories

## 4 Experiments

## 4.1 Hypothesis

We hypothesize that Chain-of-thought and Simulation should be applied to specific categories of physical reasoning tasks. And our HSR model is able to find the best scheme to assign questions from different categories to the best method suitable for answering this question, by considering the tradeoff between accuracy and computational costs. Furthermore, we hypothesize the simulation controller in our HSR model is irreplaceable by a Zero-Shot VLM-as-a-Judge, and such a simulation controller could be trained using training data generated by a smaller model and still generalizes to a larger model.

## 4.2 Dataset and Model Details

We used tasks from PhysBench dataset [3] and ContPhy dataset [23] as it contains numerous realistic physical reasoning tasks from variety of categories. For more details regarding the dataset, please refer to Appendix A.

The visual embedding models we used is CLIP ViT-B/32 [12] and the text embedding model we used is all-MiniLM-L6-V2 [13]. For the simulator used in simulation, we used LTX-Video [4] as the physics-grounded simulator. The Nano-Banana-Pro and Gemini-3-Pro [16] is used as the image generation model and base VLM to generate physical prior needed for counterfactual simulation, as shown in 3.4. For the VLM that was used in Chain-of-thought reasoning method and as a readout VLM in the simulation method, we used Gemini 3 Pro, Qwen3-VL-4B-Thinking, and Qwen3-VL-2B-Thinking [1] as the VLM. The Gemini-3-Pro is accessed through official Google API interface, and the LTX-Video model, Qwen3-VL models were run on a single A100 for maximum of 3 days.

## 4.3 Sampling Details

We adapt $K = 5$ as the sampling parameter for our ground truth probability y. By the design of our ground truth probability y, we set $\tau _ { a c c u r a c y } = 0 . 5$ . For the penalization term $\tau _ { c o s t }$ , we set $\tau _ { c o s t } = 0 . 0 3$ . When answering a question, the model will be rewarded a score of 1 if all the answers were selected. If only the part of the correct answers were selected, a score of 0.5 will be rewarded. If none of the correct answers were selected, a score of 0 will be rewarded.

6

## 5 Results

## 5.1 Evaluating Chain-of-Thought and Simulation Across Various Physical Reasoning Tasks

Accuracy in each question category differs by the method Using Gemini-3-Pro and Qwen3- VL-4B-Thinking as the base VLM, we evaluated the performance of Chain-of-Thought and Simulation as two separate independent method across all question categories in two domains: fluid hourglass and bouncing ball and all categories: property, predictive, and counterfactual. As shown in table 2, we observe that simulation is able to achieve greater accuracy in counterfactual category by a large margin. Noticably, in fluid hourglass - counterfactual (FD-Counter), the simulation outperformed Chain-of-Thought by 10% when Gemini-3-Pro was used as the base model, and 17% when Qwen3-VL-4B-Thinking was used as the base model. Conversely, simulation did not perform better than Chain-of-Thought in the property category. In categories like bouncing ball - property (EC-Prop), we observe a performance gap, where chain-of-thought outperformed simulation by 27%. In addition, we see a relatively mixed signal in the predictive category, indicating that in this question category, simulation and chain-of-thought could serve as two methods that complement each other.

## 5.2 Evaluating HSR Across Physical Reasoning Tasks

As we demonstrated above, simulation and chain-of-thought does not achieve best accuracy across all physical domains and question categories, therefore, our HSR model is essential here as it is equipped with a simulation controller that routes questions in test-time to the best method that is suitable to solve this question. Here we will evaluate and analyze the model in depth.

HSR achieve best accuracies on average As shown in the Figure 2 and 3, we also evaluated the performance of HSR under two physical domain and three question categories. We notice that HSR is able to achieve near-best performance across all physical domains and question categories, thus achieve the best

average performance when Gemini-3-Pro and Qwen3-VL-4B-Thinking was set as the base VLM, comparing with just simulation or just chain-of-thought. This demonstrates the HSR’s ability to perform at near-best accuracy regarding the question categories and physical domains, highlighting the importance of routing scheme and the HSR model.

![](images/d380860f7619a3095d9c4ba3030052205353205690380074d5750dafa6581d7c.jpg)  
Figure 3: Evaluation results of simulation only (Simulation) and chain-of-thought only (Chainof-Thought) when Gemini-3-Pro and Qwen3-VL-4B-Thinking was set as the base model. The evaluation is based on overall performance We could observe that HSR achieved the overall accuracy across all physical domains and question categories

Since our HSR model’s accuracy is entirely based on base VLMs, which our HSR model provides only a routing scheme that assigns each question to a suitable method, including simulation and chain-of-thought, we performed additional analysis probing specifically the impact of base VLM ability on the HSR model. Please refer to Appendix C for further details.

HSR determines routing scheme based on accuracy and computational costs Since our HSR model’s core ability is to provide a routing scheme given a set of physical reasoning questions, it is necessary to analyze the criterion in which this scheme is determined. Looking back to section 2 and 3, we analyze the relationship between accuracy and the computational cost. As stated in 3, the computational cost is controlled by parameter $\tau _ { t } .$ . But this parameter $\tau _ { t }$ also translates to the number of questions being assigned to ”simulation” method, given that simulation is more costly than Chain-of-Thought. Therefore, instead of analyzing the relationship between parameter $\tau _ { t } ,$ , we transform the parameter $\tau _ { t }$ to ”number of questions assigned to simulation”.

<table><tr><td rowspan=1 colspan=1>BaseModel</td><td rowspan=1 colspan=1>Prop</td><td rowspan=1 colspan=1>Pred</td><td rowspan=1 colspan=1>Counter</td></tr><tr><td rowspan=1 colspan=1>Gemini-3-Pro</td><td rowspan=1 colspan=1>28%</td><td rowspan=1 colspan=1>23%</td><td rowspan=1 colspan=1>48%</td></tr><tr><td rowspan=1 colspan=1>Qwen3-VL-4B-Thinking</td><td rowspan=1 colspan=1>27%</td><td rowspan=1 colspan=1>36%</td><td rowspan=1 colspan=1>38%</td></tr></table>

Figure 5: The percentage of questions assigned to simulation. Two models were evaluated as the base model for HSR: Gemini-3-Pro and Qwen3- VL 4R Thinking Throo que

![](images/40e1f96a7a72a0af26224ae76f9d50cdee80dce432c6304d2d082ea5153e61b7.jpg)  
Figure 4: Accuracy vs Number of Questions Assigned to Simulation. The first row is the HSR model with Gemini-3-Pro as the base model. The second row is the HSR model with Qwen3- VL-4B-Thinking as the base model. The first column is the property category collasped across two physical domains. The second column is the predictive category collapsed across two physcial domains. The third column is the counterfactual category collapsed across two physical domains. The red dash line is the number of questions assigned to simulation, which is determined by HSR model at threshold $\tau _ { t } = 0 . 5 3$ This dash line also indicates the local maximum of the function, indicating that HSR found the local optimum routing scheme when considering the tradeoff.

As figure 5.2 shows, we observe that the routing scheme determined by the HSR model is mathematically the local maximum of the function that describe the relationship between two variables: accuracy and number of questions

assigned to simulation. This indicates that our HSR model does not always go for the highest accuracy, if the computational cost is extremely high, like assigning all questions to simulation. Although the HSR model prioritize accuracy, instead, it optimizes the routing scheme based on the tradeoff between accuracy and computational costs. which is indicated by the qualitative results that our HSR model is able to reach local maximum.

HSR picks up categorical difference without training label As we shown in section 5.1, simulation only and chain-of-thought only achieved different accuracy when the question category changes, where Chain-of-thought only performed better than simulation only in property category; simulation only perform better than chain-of-thought only in counterfactual category; simulation only and chain-of-thought performed similarly well in predictive category. We wonder if our HSR model picked up such categorical difference in each method.

![](images/1a9025574361a11bd12f21200911f3476578124a448905dd3e1162ba40c4bdaf.jpg)

![](images/e1ab196ceeec3b4024f7327c21865ae2182d4f54756b3a879f470098629799fd.jpg)

![](images/c5dc6bc5abd0da792a47facd41755c625e452f603614dfe96fde3de93fc94a1f.jpg)

![](images/be2ea1abcaafe71d116121954f50931e4538765ea0d6b80e6848c3825598f652.jpg)

![](images/9d7ddbcd15de42c9f64a243810523c63ca7b492def485899a29e06d640c565d8.jpg)

![](images/fd0a1210522dfb716c97bf5876c8a1da5fb880b1135d3beb670a7e5fae72ba91.jpg)  
Figure 6: First row: Accuracy vs Number of Questions Assigned to Simulation, when VLM-as-a-Judge serves as the simulation controller in the HSR model, with Gemini-3-Pro as the base model. The first column is the property category collasped across two physical domains. The second column is the predictive category collapsed across two physcial domains. The third column is the counterfactual category collapsed across two physical domains. The red dash line is the number of questions assigned to simulation, which is determined by VLM-as-a-Judge at threshold $\tau _ { t } = 0 . 5 3$ . The plot is showing uninterpretable results, showing the ineffectiveness of VLM-as-a-Judge as a simulator controller. Second row: Histogram of the probability of simulation predicted by VLM-as-a-Judge. The probability density concentrates around 0.

As shown in table 5.2, we analyzed the percentage of questions assigned to simulation for each question category, collasping across the two physical domains. Regardness of base VLMs, the HSR tends to route property questions to Chain-of-thought more, indicated by the fact that only 28% of the questions were routed to simulation for property category. But the reverse is true for the counterfactual category, where the HSR model tends to route large proportion $( \sim 4 0 - 5 0 \% )$ of questions to simulation. In the predictive category, when Gemini-3-Pro is set as the base model, less questions are assigned to simulation, comparing with the proportion of questions that are assigned to simulation in the predictive categories, when Qwen3-VL-4B-Thinking is set as the base model.

Notice that the category label of the question were never passed into the model, and such difference in proportion of question assigned to simulation indicates that model picked up this feature without specific training label provided.

## 5.3 Is the Simulation Controller Just Learning Trivial Text Feature About The Questions?

As shown in section 5.2, our model is able to pick up the categorical difference without specific training labels, but one speculation still remains: is it possible that the simulator controller is picking up trivial text feature about the questions input into the simulation controller? Here we present an example to demonstrate that questions were assigned to simulation or chain-of-thought is not based on simple text features that could be picked up.

## Counterfactual Question Example 1

Questions: ”If the pink stick were removed, which container would light blue fluid flow into? A. Green container B. Blue container C. Pink container D. Brown container” Decision: Simulation

## Counterfactual Question Example 2

Questions: ”If the orange stick were removed, which container would light blue fluid flow into? A. Green container B. Pink container C. Brown container D. Purple container” Decision: Chain-of-thought

From the examples shown above, we can see that both questions are structurally similar, given that counterfactual condition like ”if the pink stick were removed” is an obvious feature in the question. But two questions were routed into two different methods. Therefore, the simulation controller is not just learning trivial simple text features when deciding the routing scheme.

## 5.4 Can VLM-as-a-Judge Act as a Zero-Shot Simulation Controller?

The simulator controller in our HSR model serves as the most important component. but a possibility remains that we could potentially replacing this simulator controller with a VLM-as-a-Judge method, eliminating the necessity of training a separate MLP to serve as the controller.

To test this possibility, we implemented VLM-as-a-Judge [8] method to replace the simulator controller (MLP) in our HSR model. The prompt we used for VLM-as-a-Judge is in Appendix B. We used Gemini-3-Pro as the VLM in the VLM-as-a-Judge and as the base VLM in the HSR model, since Gemini-3-Pro model is the most powerful model among the models we used. As shown in the figure 5.2, by replacing the MLP simulator controller with a VLM-as-a-Judge, the HSR model failed to produce effective routing scheme, indicated by the uninterpretable plot showing the relationship between accuracy and number of questions assigned to simulation.

We investigated into the reason behind such uninterpretable plot, and found that VLM-as-a-Judge cannot reliably judge if a question should be routed to Chain-of-thought or simulation, as indicated by the concentration of the probability of simulation around 0. This suggests that the VLM-as-a Judge just monotonically assigned all questions to Chain-of-thought, rarely assigning questions to simulation, which clearly contradicts with our finding in section 5.1.

## 5.5 Smaller Model Could Serve as a Controller For Larger Models

In our HSR model, the simulation controller is trained based on training data generated by the base model. This raises an interesting observation that if a simulation controller trained based on training data generated by a small model could be applied to HSR model with base VLM that is a large model.

We conducted such experiment that we trained a simulation controller with training data generated by Qwen3-VL-4B-Thinking, but set the base VLM as Gemini-3-Pro. As shown in the figure 5.5, we see that the simulation controller is able to again find the best routing scheme based on the trade off between accuracy and computational cost, despite being trained on training data generated by Qwen3-VL-4B-Thinking that is a way smaller model compared with Gemini-3-Pro. Such finding serves as an evidence that smaller model could be used to train a simulation controller that orches trates a model that is way larger than itself, alike the idea demonstrated in papers like [11].

## 6 Conclusion

In this paper, we introduce Hybrid Simulation and Reasoning Model (HSR) for physical reasoning in MLLMs. Our model is able to decide the appropriate method to solve a physical reasoning problem at test time, and finds an optimal balance between accuracy and cost. Further, our model can be trained on a smaller vision-language model that is only 4B parameters and deployed to improve performance on much bigger base models with >3T parameters like Gemini 3 Pro.

We recognize that our HSR model is limited by the base model performance. But with the development of video models and VLMs, indicated by works like [21] and [6], we are optimistic that our HSR model will become more powerful as the base model’s ability improves in the future. And as the base model’s ability improves, the routing scheme provided by HSR could be beneficial. For future work, we want to study the relationship between scene complexity and the necessity for simulation.

![](images/03a32a466ae1f0e2f20f88d07ec6fabc294916a3bf9e8054c382f808476a6f43.jpg)

![](images/2cb8ab96ef524a4e513ae219f19f9b6c95793697e15d7dbdd3a12bf66c13c362.jpg)

![](images/b5ae58207783a8fd824391c036c49e950d5e2235b193f20a57417b5188ef4630.jpg)  
Figure 7: Accuracy vs Number of Questions Assigned to Simulation. The HSR model with Gemini-3-Pro as the base VLM model and simulation controller trained based on Qwen3-VL-4B-Thinking is evaluated. The first column is the property category collasped across two physical domains. The second column is the predictive category collapsed across two physcial domains. The third column is the counterfactual category collapsed across two physical domains. The red dash line is the number of questions assigned to simulation, which is determined by HSR model at threshold $\tau _ { t } = 0 . 5 3$ This dash line also indicates the local maximum of the function, indicating that HSR found the local optimum routing scheme when considering the tradeoff.

## References

[1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[2] Peter W Battaglia, Jessica B Hamrick, and Joshua B Tenenbaum. Simulation as an engine of physical scene understanding. Proceedings of the national academy of sciences, 110(45): 18327–18332, 2013.

[3] Wei Chow, Jiageng Mao, Boyi Li, Daniel Seita, Vitor Guizilini, and Yue Wang. Physbench: Benchmarking and enhancing vision-language models for physical world understanding. arXiv preprint arXiv:2501.16411, 2025.

[4] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.

[5] Joshua K Hartshorne and Mengguo Jing. Insights into cognitive mechanics from education, developmental psychology and cognitive science. Nature Reviews Psychology, pages 1–15, 2025.

[6] Jingyi Jin, Shun Zhang, Xiaodong Yang, and Joseph Wang. Post-train cosmos reason 2 for autonomous vehicle video captioning and vqa. https://nvidia-cosmos.github.io/ cosmos-cookbook/recipes/post\_training/reason2/video\_caption\_vqa/post\_ training.html, January 2026. NVIDIA Cosmos Cookbook.

[7] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199–22213, 2022.

[8] Seongyun Lee, Seungone Kim, Sue Park, Geewook Kim, and Minjoon Seo. Prometheusvision: Vision-language model as a judge for fine-grained evaluation. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 11286–11315, 2024.

[9] Ruibo Liu, Jason Wei, Shixiang Shane Gu, Te-Yen Wu, Soroush Vosoughi, Claire Cui, Denny Zhou, and Andrew M Dai. Mind’s eye: Grounded language model reasoning through simulation. arXiv preprint arXiv:2210.05359, 2022.

[10] Ryan Liu, Jiayi Geng, Addison J Wu, Ilia Sucholutsky, Tania Lombrozo, and Thomas L Griffiths. Mind your step (by step): Chain-of-thought can reduce performance on tasks where thinking makes humans worse. arXiv preprint arXiv:2410.21333, 2024.

[11] Stefan Nielsen, Edoardo Cetin, Peter Schwendeman, Qi Sun, Jinglue Xu, and Yujin Tang. Learning to orchestrate agents in natural language with the conductor. arXiv preprint arXiv:2512.04388, 2025.

[12] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. Proceedings of the 38th International Conference on Machine Learning, 139:8748–8763, 2021. doi: 10.48550/arxiv.2103.00020.

[13] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bertnetworks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 11 2019. URL https://arxiv.org/ abs/1908.10084.

[14] Kevin Smith, Peter Battaglia, and Joshua Tenenbaum. Integrating heuristic and simulationbased reasoning in intuitive physics. PsyArXiv, 2023.

[15] Felix A Sosa, Samuel J Gershman, and Tomer D Ullman. Blending simulation and abstraction for physical reasoning. Cognition, 254:105995, 2025.

[16] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

[17] Gemini Robotics Team, Krzysztof Choromanski, Coline Devin, Yilun Du, Debidatta Dwibedi, Ruiqi Gao, Abhishek Jindal, Thomas Kipf, Sean Kirmani, Isabel Leal, Fangchen Liu, Anirudha Majumdar, Andrew Marmon, Carolina Parada, Yulia Rubanova, Dhruv Shah, Vikas Sindhwani, Jie Tan, Fei Xia, Ted Xiao, Sherry Yang, Wenhao Yu, and Allan Zhou. Evaluating gemini robotics policies in a veo world simulator, 2025. URL https://arxiv.org/abs/ 2512.10675.

[18] Tomer D Ullman, Elizabeth Spelke, Peter Battaglia, and Joshua B Tenenbaum. Mind games: Game engines as an architecture for intuitive physics. Trends in cognitive sciences, 21(9): 649–665, 2017.

[19] Abdul Waheed, Chancharik Mitra, Laurie Z Wang, Deva Ramanan, and Bhiksha Raj. Less is more tokens: Efficient math reasoning via difficulty-aware chain-of-thought distillation. arXiv preprint arXiv:2509.05226, 2025.

[20] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

[21] Thaddaus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swer-¨ sky, Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025.

[22] Jingbo Yang, Bairu Hou, Wei Wei, Yujia Bao, and Shiyu Chang. Ares: Adaptive reasoning effort selection for efficient llm agents. arXiv preprint arXiv:2603.07915, 2026.

[23] Zhicheng Zheng, Xin Yan, Zhenfang Chen, Jingzhou Wang, Qin Zhi Eddie Lim, Joshua B Tenenbaum, and Chuang Gan. Contphy: Continuum physical concept learning and reasoning from videos. In International Conference on Machine Learning. PMLR, 2024.

## A Dataset Details

We consider questions in three different categories: property, counterfactual, predictive and two physical domains: fluid hourglass and bouncing ball. These two sub-domains represent fluid dynamics and elastic collision, which are both non-rigid body motions.

In total, there are 150 questions in property category for both sections; There are 104 questions in temporal predictive category and 116 questions in counterfactual category for fluid hourglass setting; There are 88 questions in temporal predictive category and 122 questions in counterfactual category for bouncing ball setting.

## B Prompt

## B.1 Prompt Used for Counterfactual Simulation Prior Generation

Here we present the prompt that we used when implementing Gemini-3-Pro and Nano-Banana-Pro to prepare the physical prior of the counterfactual simulation for physical simulator (LTX-video).

Step 1: Judging the neediness for counterfactual simulation

Prompt for judging the neediness for counterfactual simulation

Imagine you are a physical simulation expert. Here are some videos or images together and a question based on these images or videos. the question is: question. You have to analyze the images or videos and the question. Do you think you need to reason counterfactually in order to answer the question based on the videos or images? Answer in ’Yes’ or ’No’ Only.

Step 2: Judging the modifications for counterfactual simulation

Prompt for judging the modifications for counterfactual simulation

Imagine you are a physical simulation expert. Here are some videos or images together and a question based on these images or videos. The question is: question. Does the question suggested any counterfactual modifications we have to make to the objects and setup in the images and videos? If so, output the modifications you want to make only. If you are suggesting multiple modifications, separate them by a newline character. Example modifications can be: Remove object A, add object A, etc.

Step 3: Modifying the original question into non-counterfactual version.

Prompt for modifying the original question into non-counterfactual version

Here you will be presented with a question with it’s answer: question. Please remove the counterfactual condition in the original question provided to you, and output the question with the answer choices provided in the original question, if there are any, in the form that aligns with the input question I give to you only.

## B.2 Prompt Used for VLM-as-a-Judge

Here we present the prompt that we used when implementing VLM-as-a-Judge:

## Prompt for VLM-as-a-Judge

First, you will see a set of images or videos, along with a physical reasoning question based on this set of images or videos. Then, the physical reasoning question is: question. Do you think in order to answer this question, you will need to perform extra simulation in a physics simulator, instead of using chain-of-thought only? Answer in ’Yes’ or ’No’ only.”

## C Additional Analysis

## C.1 Impact of Base Model

We conducted all of our experiment using various models including powerful closed-sourced model and open-source models. Given the importance of base VLM models in our HSR model, we tested the boundary that to which extent does HSR fails to work due to the incapability of the base VLM model. Therefore, we conducted experiment using Qwen3-VL-2B-Thinking, which is a even smaller model thant Qwen3-VL-4B-Thinking, and found that this model failed miserable as a base model in the HSR model.

Therefore, here we present some of the failure cases in which Qwen3-VL-2B-Thinking struggles with basic question answering due to repetitiveness and congested Chain-of-thought.

## Example 1

## Question: ”Is the density of blue fluid equal to that of pink fluid?”

Answer: ”So, let’s look at the video. The blue and pink fluids. wait, in the video, when the blue fluid is moving, does it interact with the pink? Wait, the question is about their densities. But in the simulation, maybe the colors are just visual. Wait, but the key here is to check if they have the same density. Wait, in the video, when the fluids are moving, maybe they’re the same. Wait, but let’s think. Wait, the problem is whether the density is equal. But in the video, maybe the blue and pink fluids are moving in a way that suggests same density. Wait, but maybe I need to check the motion. Wait, in the video, the blue fluid and pink fluid2014do they have the same density?<sup>˘</sup> Wait, maybe the answer is yes. Wait, but maybe not. Wait, let’s see. Wait, the question is: Is the density of blue fluid equal to that of pink fluid? So, if in the simulation, the two fluids are moving in the same way, maybe they have same density. Wait, but maybe the answer is yes. Wait, but maybe I need to think again. Wait, the video shows the blue and pink fluids. Wait, in the simulation, maybe they have the same density. Wait, but the question is asking if they are equal. So, based on the video, if the blue and pink fluids are moving in the same direction or same way, maybe they have same density. Wait, but maybe the answer is yes. Wait, but I need to check. Wait, in the video, the blue fluid and pink fluid2014do they have the same density?<sup>˘</sup> Wait, maybe the answer is yes. Wait, but maybe not. Wait, perhaps the answer is no. Wait, but I need to think. Wait, the problem is that in the simulation, the blue and pink fluids might have different densities. .....Wait \*stop”

Question: ”Which container will the fluid from the other orange emitter flow into? A. Cyan container B. Pink container C. Gray container D. Green container”

Answer: ””Got it, let’s look at the video. The orange emitters are the ones with orange colors at the top. The fluid from the orange emitters flows into the containers below. Let’s check the containers: the bottom row has gray, green, cyan, pink. Wait, the video shows that the orange fluid flows into the blue container? Wait no, let’s see. Wait, the containers are the ones at the bottom. Let’s check the colors. The containers are: first one (leftmost) is gray, then green, then cyan, then pink. Wait, the orange emitter2014<sup>˘</sup> Wait, the emitters are the orange ones at the top. Wait, in the video, the orange fluid flows into the blue container? Wait, no. ..... Wait, no. Wait, maybe the answer is C. Gray container. Wait, no. Wait, I think the correct answer is C. Gray container. Wait, no. Wait, maybe the answer is C. Gray container. Wait, no. Wait, I’m confused. Wait, maybe the answer is C. Gray container. Wait, no. Wait, let’s check the video again. The orange fluid flows into the blue container. Wait, no. Wait, the blue container is not there. Wait, the containers are the ones at the bottom. The first container is gray, then green, then cyan, then pink. Wait, the orange fluid flows into the blue container. Wait, no. Wait, maybe the orange fluid flows into the green container. Wait, no. Wait, the answer is C. Gray container? Wait, no. Wait, maybe the answer is C. Gray container. Wait, no. Wait, I think the correct answer is C. Gray container. Wait, no. Wait, maybe the answer is C. Gray container. Wait, no. Wait, I’m stuck. Wait, maybe the answer is C. Gray container. Wait, no. Wait, let’s see: the video shows that the orange fluid flows into the blue container. Wait, no. Wait, the blue container is not there. Wait, the containers are the ones at the bottom. The first container is gray, then green, then cyan, then pink. Wait, the orange fluid flows into the blue container. Wait, no. Wait, maybe the orange fluid flows into the green container. Wait, no. Wait, the answer is C. Gray container? Wait, no. Wait, maybe the answer is C. Gray container. Wait, no. Wait, I think the correct answer is C. Gray container. Wait,”, \*stop”

## Example 3

## Question: ”Is the plasticity of the orange ball much less than the blue ball?”

Answer: ”So, let’s analyze the video. The orange ball and blue ball are moving around. In the video, when the blue ball moves, it seems to interact with the orange ball. But Wait, the question is about plasticity. Plasticity here might refer to how much they deform or how they bounce. Wait, in the video, the orange ball and blue ball are rolling or moving. Wait, maybe the key is that the blue ball is moving more or less. Wait, no. Wait, the question is asking if the plasticity of the orange ball is much less than the blue ball. Plasticity is about how much a material deforms. If the blue ball is more deformable, then its plasticity is higher. Wait, but in the video, when the blue ball moves, does it deform? Wait, maybe the answer is based on whether they roll or bounce. Wait, the orange ball and blue ball are moving on the blocks. Wait, maybe the blue ball is more elastic or less plastic. Wait, but the question is about plasticity. Plasticity is the ability to deform permanently. If the blue ball is more plastic, then its plasticity is higher. Wait, but the question is asking if the orange ball’s plasticity is much less than the blue ball’s. So if the blue ball is more plastic, then yes. But how do we know from the video? Wait, maybe the blue ball is more likely to deform. .... Wait, maybe the answer is B. Wait, but I’m not sure. Wait, maybe the answer is B. Wait, but I think the answer is B. Wait, no. Wait, maybe the answer is B. Wait, but I’m not sure. Wait, maybe the answer is B. Wait, but I think the answer is B. Wait, no. Wait, maybe the answer is B. Wait, but I’m not sure. Wait, maybe the answer is B. Wait, but I think the answer is B. Wait, no. Wait, maybe the answer is B. Wait, but I” \*stop

## C.2 Reasoning Effort

For Gemini-3-Pro, we notice that the model comes with the ability to adjust the reasoning effort during the reasoning process. Although when Gemini-3-Pro serves as the base VLM to support the Chain-of-thought method in our HSR model, the reasoning effort should by default be high, the role of reasoning effort when Gemini-3-Pro serves as the readout VLM in the simulation method remains unclear. Therefore, as an additional analysis, we tested the impact of reasoning effort when Gemini-3-Pro is set as the readout model of the simulation method in our HSR model.

According to our analysis in section 5.2, we focuses on two specific question categories: predictive and counterfactual, as these two categories are more relevant to the simulation method. As shown in the table C.2, we notice that the reasoning effort when Gemini-3-Pro is served as the readout VLM for the simulation method does does impact the final accuracy. Surprisingly, during the readout phase of simulation, higher reasoning effort did not increase the accuracy of the readout, instead it decreased the accuracy.

<table><tr><td rowspan=1 colspan=1>BaseModel</td><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=1>FD-Pred</td><td rowspan=1 colspan=1>EC-Pred</td><td rowspan=1 colspan=1>FD-Counter</td><td rowspan=1 colspan=1>EC-Counter</td></tr><tr><td rowspan=2 colspan=1>Gemini-3-Pro</td><td rowspan=1 colspan=1>Sim-High</td><td rowspan=1 colspan=1>0.315</td><td rowspan=1 colspan=1>0.570</td><td rowspan=1 colspan=1>0.435</td><td rowspan=1 colspan=1>0.440</td></tr><tr><td rowspan=1 colspan=1>Sim-Low</td><td rowspan=1 colspan=1>0.363</td><td rowspan=1 colspan=1>0.568</td><td rowspan=1 colspan=1>0.541</td><td rowspan=1 colspan=1>0.451</td></tr></table>

Table 1: Evaluation results of simulation only (Sim) and chain-of-thought only (CoT) when Gemini-3-Pro set as the base model. We varied the reasoning effort in Gemini-3-Pro, which high reasoning effort (High) and low reasoning effort (Low) were used for evaluation. The evaluation is based on fluid hourglass (FD) and bouncing ball (EC) domains, with questions in predictive (Pred), and counterfactual (Counter) categories. The best accuracy in each question category and physical domains is highlighted in bold.

## D Statements

## D.1 Language Model Use

LLMs were used in this work for literature review, editing the grammar error, and polishing writing in this paper. Also assisted the code writing.