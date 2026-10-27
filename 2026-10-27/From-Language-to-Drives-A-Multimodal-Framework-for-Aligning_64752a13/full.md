# From Language to Drives: A Multimodal Framework for Aligning Autonomous Robots with Human Purposes

Diana Cerdas-Vargas, Alejandro Romero and Richard J. Duro

Abstract— This work explores a multimodal approach to the alignment of the behavior of open-ended learning autonomous robots with human purposes through the use of large language models (LLMs) as cultural enablers. While autonomous robots can generate their own skills and goals through interaction with the environment, they often lack mechanisms to ensure that these remain relevant and aligned to those of human users. The proposed approach makes use of the general and cultural knowledge contained in LLMs and VLMs to establish a multimodal interactive alignment process in which human purposes, expressed in natural language, are transformed into motivational components of a robot architecture in the form of robot missions and drives that guide the behavior of the robot. The system is structured into three layers for semantic alignment, mission generation, and drive modeling. This design enables a bidirectional interaction loop: human inputs define high-level purposes, robots expand them into executable missions making use of multimodal information relayed by LLMs and VLMs, and feedback maintains alignment with human intent. Experiments in simulation and with a physical robot demonstrate that the process produces valid missions and behaviors, supporting effective human–agent alignment in interactive scenarios with autonomous robots.

## I. INTRODUCTION

A central objective of cognitive and developmental robotics is to construct artificial agents that acquire increasingly sophisticated competencies through prolonged interaction with their environment. Inspired by human cognitive development, developmental robotics emphasizes intrinsically motivated exploration, incremental skill acquisition, and the progressive organization of behavior [1], [2]. In parallel, the paradigm of Open-Ended Learning (OEL) formalizes the capacity of autonomous systems to continuously generate new goals and expand their behavioral repertoire beyond predefined tasks or datasets [3]. Together, these perspectives frame autonomy not as the execution of fixed policies, but as an ongoing developmental process.

Within cognitive architectures, such developmental openendedness is commonly achieved through motivation-based control systems [4]–[6]. In these systems, behavior is regulated by internal motivational variables that define why the agent acts, allowing goals and skills to emerge as means to regulate internal states. We refer to these domainindependent motivations as robot purposes.

Robot purposes include designer-embedded needs, such as homeostatic requirements and intrinsic motivations that drive exploration and competence acquisition [7], [8], as well as externally derived purposes, or missions, which reflect what human users expect the robot to accomplish during runtime. Both needs and missions are operationalized through drives – quantitative variables encoding the urgency of satisfying a given purpose [9]. By acting to regulate its drives, the robot autonomously discovers goals and learns skills that are grounded in the robot’s sensorimotor system. Alignment is achieved when the resulting behavior fulfills human expectations [10].

However, integrating human purposes into a developmental, open-ended system presents a significant challenge. Human intentions are expressed through natural language – ambiguous, context-dependent, and culturally grounded [11] – whereas motivational architectures require structured, formally defined representations [12]. This discrepancy creates a semantic gap that may lead to misaligned internal purposes, undermining trust and safety [10]. Explainability [13] and grounding strategies [14] address aspects of this issue, but they do not fully solve the problem of embedding human purposes into the motivational dynamics that drive developmental learning.

Recent advances in Large Language Models (LLMs) and Vision-Language Models (VLMs) provide powerful tools for interpreting linguistic instructions and relating them to perceptual context. Nevertheless, employing such models as end-to-end robot controllers raises concerns regarding explainability, controllability, safety, and the absence of explicit motivational structure [15]. From a developmental perspective, delegating control to monolithic policies risks bypassing the very mechanisms – intrinsic motivation, goal discovery, and incremental learning – that underpin autonomous cognitive growth.

We therefore adopt a complementary view: LLMs and VLMs are integrated not as end-to-end control policies, but as high-level semantic and cultural knowledge resources that provide structured priors to a motivation-based cognitive architecture. These multimodal models support the interpretation and clarification of human purposes, constraining the space of plausible missions. The discovery of specific goals and skills, however, remains grounded in the robot’s developmental learning processes and regulated by explicit drives. In this way, large models inform but do not replace the motivational dynamics that sustain autonomous development.

To operationalize this perspective, we propose a multimodal alignment framework composed of three layers: (1) semantic alignment, which interprets and refines human pur poses through interactive dialogue; (2) mission generation, which translates aligned purposes into structured, domainconsistent missions; and (3) drive modeling, which transforms missions into quantitative drive functions embedded in the motivational system. An explicit alignment need, implemented as an alignment drive, motivates the robot to seek clarification until a coherent internal representation of the intended purpose is achieved.

The contributions of this work are threefold: (i) a cognitively and developmentally grounded framework for embedding human purposes into a motivational architecture supporting open-ended learning; (ii) the integration of specialized LLM- and VLM-based modules as semantic priors within this architecture; and (iii) experimental validation in simulation and on a physical robotic platform, demonstrating effective human–robot alignment while preserving autonomous developmental processes.

By combining multimodal semantic mediation with explicit motivational regulation, this work contributes to the design of developmental autonomous systems that can sustain long-term growth while remaining aligned with human intentions.

## II. RELATED WORK

## A. Instruction Following and Foundation Models in Robotics

Bridging the semantic gap between high-level human purposes and executable robot behavior has become a central research topic in embodied AI. Recent approaches increasingly leverage large-scale foundation models to interpret user instructions and generate robot actions. For instance, SayCan [16] combines a Large Language Model (LLM) with a predefined library of robot skills. The LLM ranks candidate skills according to semantic relevance, while a value function evaluates feasibility in the current state. This modular strategy enables execution of long-horizon instructions, provided the required primitive skills are available. However, the system remains constrained by a fixed behavioral repertoire and does not generate new goals or expand its skill space.

Other approaches, such as RT-2 [17] move toward tighter integration by training a unified vision-language-action model on Internet-scale and robotic data. The model directly maps multimodal inputs to low-level actions, enabling impressive zero-shot generalization. Nevertheless, this end-toend formulation lacks explicit motivational or goal representations. The resulting policies are difficult to interpret, regulate, or adapt structurally, raising concerns regarding transparency, controllability, and long-term alignment in open-ended environments.

From another point of view, there are systems that address pragmatic intention understanding. One such case is [11], which classifies user instructions according to their explicitness and leverages contextual cues to infer appropriate actions. Another is the work presented in [18] that consideres introducing an estimation of uncertainty in user commands and tries to increase certainty through dialogue. While improving robustness in human–robot interaction, such approaches typically operate within predefined task templates and do not support autonomous goal discovery.

Finally, shared autonomy methods [19] adapt robot policies to user preferences through interactive feedback. Although effective for collaboration, they primarily refine existing behaviors rather than reorganizing the agent’s motivational structure or enabling the emergence of new purposes.

## B. Developmental Robotics and Open-Ended Learning

Developmental robotics investigates how autonomous agents can acquire progressively complex competencies through processes inspired by human cognitive development [1], [2]. Rather than relying on externally specified task objectives, developmental systems are driven by intrinsic motivations such as curiosity, competence improvement, or learning progress. These mechanisms support exploratory behaviors, goal babbling, and the autonomous construction of hierarchical skill repertoires over extended interaction with the environment.

Motivation-based cognitive architectures operationalize these principles through internal drives that regulate behavior and trigger self-generated goals [4]–[6]. Within the broader Open-Ended Learning (OEL) paradigm [3], agents continually expand their behavioral and representational spaces, discovering novel goals and refining predictive models through sensorimotor experience. Crucially, learning remains grounded: goals emerge from embodied interaction rather than from predefined symbolic plans.

Despite these advances, most developmental approaches assume that high-level purposes are either hard-coded by the designer or provided in structured form. While they excel at autonomous goal discovery and skill acquisition [4], [5], [12], they do not directly address how abstract, linguistically expressed human purposes can be integrated into the agent’s motivational system without disrupting its developmental dynamics. As a result, the problem of semantically aligning human intentions with intrinsically motivated, open-ended learning processes remains largely unresolved.

## C. Alignment in Embodied Agents

Beyond robotics, the AI alignment literature highlights the difficulty of ensuring that artificial agents pursue objectives consistent with human intentions [10]. Many alignment strategies for large models rely on techniques such as reinforcement learning from human feedback [20] or instruction tuning [21]. Although effective at shaping output behavior, these methods do not necessarily yield explicit, inspectable representations of purpose within the agent’s control architecture. In embodied, open-ended systems, alignment requires mechanisms that integrate human purposes into the internal motivational dynamics governing goal generation and learning.

Taken together, existing research either (i) leverages large multimodal models for instruction following, often through fixed skills or end-to-end policies, or (ii) develops developmental architectures capable of autonomous goal discovery without addressing semantic alignment with human purposes. What remains lacking is an integrated framework in which multimodal models provide semantic and cultural priors while an explicit motivational architecture preserves grounded developmental learning, and alignment.

Our work addresses this gap by embedding LLMs and VLMs as knowledge resources within a motivation-based cognitive architecture that integrates information from different sensing modalities, enabling the transformation of human purposes into aligned drives that guide autonomous openended development.

## III. PROPOSED APPROACH

Translating human purposes expressed in natural language into motivational representations such as missions and drives is not a trivial task. Humans rely on cultural and contextual knowledge to interpret intentions and desires, whereas computational systems lack this implicit background. To bridge this semantic gap, we use LLMs as cultural knowledge bases related to different sensing modalities that, through an alignment process, are able to clarify human purposes.

Fig. 1 illustrates the proposed framework. At the input of the framework there are several elements that allow the user to add multimodal inputs such as an image, language and audio. Next, at its core, the Alignment Engine extracts and interprets the human purpose from multimodal information (ususally speech and vision). In cases of ambiguity, it engages in iterative dialogue with the user to refine the interpretation, seeking confirmation before proceeding. Once the purpose is clarified, it is passed to the Translation Engine, which decomposes it into missions and corresponding drive functions. These are explicitly instantiated within the motivational system of the cognitive architecture, guiding the robot’s autonomous behavior.

The system is designed for non-expert users. Interaction is therefore concentrated at the specification stage: the user expresses the desired objective in natural language, while the architecture autonomously generates the underlying missions and drives. This allows intuitive purpose specification without requiring knowledge of the robot’s motivational structure.

The function and configuration of each of the three instances of the LLMs used are explained in detail below. Additionally, the prompts used in each of the LLM instances, as well as the source code and the results from the design tests are all available in a public repository <sup>1</sup>.

## A. Multimodal Input

This module manages multimodal inputs to improve the user experience by providing a more intuitive interaction interface. Visual data is processed through a Vision Language Model (VLM), specifically we used GPT-4o by OpenAI. This model is configured through a specialized prompting to generate a detailed scene description and provide an inferred human intent. For audio processing, the system integrates two distinct frameworks: OpenAI’s Whisper model for Speech-to-Text (STT) transcription, and Google Text-to-Speech (gTTS) to handle the system’s vocal responses.

## B. Alignment Engine

This first LLM uses an instance of the phi4:14b model by Ollama. It receives the natural language purpose provided by the user and/or a visual description of the environment and reformulates it into a clear, unambiguous specification. When necessary, it engages in a brief interactive dialogue with the user to resolve ambiguities or request missing details.

1) Prompt: The prompt used to guide this LLM instance was carefully structured to include all relevant contextual and operational information. It consists of six main components: (1) Context, which introduces the LLM’s task and defines key concepts such as needs, missions and drives; (2) Robot description and general description of its environment; (3) Robot perceptions, specifying available sensory inputs along with their value ranges; (4) Input description, explaining the nature and format of the user-provided purpose; (5) Task constraints, which define what the LLM must do and explicitly limit its scope; and (6) Interaction rules, which describe the expected structure and format of the model’s responses, ensuring consistency and compatibility with downstream components.

2) Input and Output Configuration: The input to this module is a purpose expressed in natural language by the user, along with the image description from the Multimodal Input, when used. The user is required to answer questions and provide additional details to the LLM when needed.

To determine when a purpose was fully aligned we looked at how humans clarify intentions during conversation. Often, they ask: “Is this what you meant?” or “Is this what you want?”. Based on this behavior, the user is who decides when the purpose is correctly understood. From a humanrobot interaction perspective, this means the human user has full control over when to move to the next stage.

The output is a final aligned and refined purpose statement, written as a clear paragraph. If, based on the robot’s capabilities and perceptions, the model determines that the purpose is not achievable, it returns a standard message: “It is impossible to achieve your purpose with this robot”.

## C. Mission Generator

Once the user purpose has been aligned, a second LLM instance, also using the phi4:14b model, is responsible for decomposing it into a structured set of missions. These missions represent concrete end-states that the robot must achieve in order to fulfill the user’s intention.

Each mission also includes an associated weight that indicates its relative importance within the overall task context, reflecting how critical it is for achieving the human’s intended purpose for the robot. The output of this component serves as a bridge between high-level human intent and the motivational structure that will later govern the robot’s autonomous behavior.

![](images/ec20266297c45889894a46b55d54f90578d6619af869d624bdd38e07c2dca2f4.jpg)  
Fig. 1: Overview of the proposed framework. The Alignment Engine manages human–robot interaction to align human and robot purposes. The Translation Engine converts aligned purposes into missions and drives, which are integrated into the Motivational System of a cognitive architecture to guide autonomous robot behavior.

1) Prompt: The prompt for this instance was designed to clearly instruct the model to generate a list of missions that directly correspond to the aligned purpose. It included: (1) Context, which introduces the LLM’s task and defines key concepts such as needs, missions and drives; (2) Robot description; (3) Robot perceptions; (4) Requirements for creating missions and assigning their weight (priority to fulfill the purpose); (5) Interaction rules and output format; and (6) A list of predefined internal needs, introduced by the designer within the cognitive architecture.

2) Input and Output Configuration: The input for this module is the refined purpose generated by the Alignment Engine, expressed as text. On the other hand, the output is an array of missions, where each mission is represented as another array. The first element is a descriptive tag of the mission, it should represent a meaningful endstate, for example “object in hand”. The second element corresponds to a numerical weight that indicates the relative priority of that mission. The format of the output is: [[tag<sub>1</sub>, weight<sub>1</sub>], [tag<sub>2</sub>, weight<sub>2</sub>], ..., [tag<sub>N</sub>, weight<sub>N</sub>]].

## D. Drive Generator

A third LLM instance uses the GPT-4.1 model from OpenAI, and is responsible for modeling a mathematical drive function for each mission. A drive function is a continuous, monotonic mathematical expression that quantifies the urgency to satisfy a given mission based on the robot’s current state. This function must return zero only when the mission has been fulfilled, and increase smoothly as the robot deviates from the desired goal-state. The output of this component integrates directly into the robot’s motivational system.

1) Prompt: The prompt provided to this LLM was designed to ensure internal consistency with the elements of the cognitive architecture, and alignment with the robot’s perceptual space. It includes: (1) Context definitions for missions, drive functions, and drive values; (2) Robot description; (3) Robot perceptions; (4) Format specification of the missions generated in the previous stage; and (5) Structured instructions for generating a drive function for each mission.

2) Input and Output Configuration: The input to this component consists of two elements. First, the fully refined purpose provided by the Alignment Engine, expressed in text form. Second, the list of missions generated by the previous LLM instance. The output is a list of mathematical drive functions, one for each mission. Each entry follows the structure: Mission: [tag, weight] and Drive: valid Python expression. If the LLM detects that the purpose cannot be fulfilled with the robot’s current capabilities, it returns the message: “It is impossible to achieve your purpose with this robot”.

## IV. EXPERIMENTS AND RESULTS

To evaluate the proposed framework, we conducted two experiments. The first experiment evaluates whether the framework can align with and correctly interpret human purposes, while the second assesses the generation of missions and drives and their application to robotic tasks.

## A. Dataset

To evaluate whether the proposed framework can align with and interpret human purposes expressed with different levels of clarity, we defined three robot-executable purposes with increasing complexity. These purposes were: arranging apples by color (simple), classifying cylinders by color into baskets (intermediate), and stacking cylinders at the center of the table from largest to smallest (complex).

To simulate how a non-expert user might express these purposes in natural language, we created 15 different ambiguous user inputs for each of the three purposes. Table I presents example ambiguous inputs for each purpose. In addition, three different images illustrating the intended outcome were used for each human purpose. During the experiments, the images associated with each purpose were alternated across trials so that the VLM was exposed to different visual scenarios.

The complete dataset, including the 45 expressions and 9 images used in the experiments, is available in a public repository <sup>2</sup>.

## B. Semantic Alignment Test

This test was designed to evaluate the ability of the Alignment Engine to disambiguate natural language purposes and convert them into fully defined, structured task descriptions through user interaction. Since both the inputs and outputs of this process are purely linguistic, cosine semantic similarity in embedding space [22] was used to measure how closely their meanings align.

TABLE I: Example ambiguous user inputs for the three evaluated purposes, simulating how non-expert users describe tasks in natural language.
<table><tr><td rowspan=1 colspan=1>Purpose</td><td rowspan=1 colspan=1>Example of non-expert user input</td></tr><tr><td rowspan=1 colspan=1>Arrange apples by color</td><td rowspan=1 colspan=1>&quot;Place the apples where they belong[as in the image]&quot;</td></tr><tr><td rowspan=1 colspan=1>Classifying cylinders by color</td><td rowspan=1 colspan=1>&quot;Sort the cylinders on the table [asin the image].&quot;</td></tr><tr><td rowspan=1 colspan=1>Stacking cylinders in the cen-ter of a table from biggest tosmallest</td><td rowspan=1 colspan=1>&quot;Arrange the pieces on top of eachother [as in the image]&quot;</td></tr></table>

Cosine semantic similarity captures the similarity of two sentences by comparing the angle between their embedding vectors. A value of 1 indicates identical semantic meaning, a value of 0 indicates no semantic relation, and a value of -1 indicates opposite meanings.

To evaluate the alignment process, we tested the Alignment Engine using the dataset described in Section IV-A. Each input was processed under four distinct experimental conditions to isolate the impact of multimodal input and human feedback:

1) Full Interaction (Multimodal + Human Feedback): The system uses multimodal input, such as an image illustrating the intended outcome, together with active user interaction, where the LLM performs iterative clarification until the user confirms that the purpose is fully understood.

2) Autonomous Multimodal Interaction (Multimodal + No Human Feedback): The LLM interprets the input together with the multimodal information as provided, without any subsequent human clarification.

3) Human Interaction (Unimodal + Human Feedback): The system operates without multimodal support but maintains iterative dialogue with the user to refine and disambiguate the purpose.

4) Baseline (Unimodal + No Human Feedback): The LLM interprets the ambiguous textual purpose without additional resources or user feedback.

In each of the four experimental conditions, the final purpose description generated by the LLM was compared with an ideal manually written reference description using cosine semantic similarity. The embedding vectors were computed using the all-MiniLM-L6-v2 model [23]. The reference purpose descriptions used for semantic comparison are listed in Table II.

TABLE II: Detailed ideal purpose descriptions used as reference to estimate the semantic similarity.  
![](images/b367f3c09ca898c9bd234ec232238c6563d5e480106033ef8a4bec38c44f7037.jpg)

The results of the semantic alignment evaluation are summarized in Fig.2, which illustrates the distribution of cosine similarity scores across the different experimental conditions. For the initial low-detail inputs provided by users, similarity scores ranged between 0.4 and 0.6. These values quantify the initial semantic gap, reflecting the fact that human instructions are often underspecified or ambiguous. At this level of similarity, the Alignment Engine cannot reliably translate purposes into well-defined missions and drive functions.

When the purpose was interpreted by the LLM without alignment interaction, the cases that included multimodal input showed much higher semantic similarity to the ideal reference. This improvement suggests that the model was able to use visual context about the robot and its environment to enrich the linguistic description. However, despite this increase, the results show that the generated purposes still diverge semantically from the ideal reference. In other words, the improvement is insufficient to guarantee that the generated purpose fully captures the user’s intended meaning.

A Mann-Whitney U test was performed to determine whether the use of multimodal resources led to greater improvement in semantic alignment. This non-parametric test was used to evaluate distributional differences between the unimodal and multimodal settings for three cases: the initial inputs, the outputs without feedback, and the outputs with feedback. Statistical differences were observed in the nohuman-feedback scenarios for the Arrange apples by color and Classify cylinders by color purposes. In these cases, the p-values were below the α = 0.05 threshold, leading to rejection of the null hypothesis. This result indicates that multimodal resources provide a significant improvement mainly in scenarios without human feedback.

![](images/dce5641a16c8c9a920e39821087f84f213f667a21f8f746c000c05eb47cf8025.jpg)  
Fig. 2: Box plot of semantic similarity between input purposes and output descriptions generated by the Alignment Engine under multimodal and unimodal configurations, with and without human feedback.

When interactive alignment is enabled (pink distributions in Fig. 2), cosine similarity scores consistently converge toward high and stable values across all three tasks. Importantly, these scores are significantly higher than those obtained in non-interactive conditions, regardless of whether multimodal resources are available. This result highlights the critical role of user interaction in achieving precise semantic alignment. Iterative clarification ensures that the internal representation of the purpose closely matches the intended meaning.

The contribution of multimodal information (e.g., visual input) is more nuanced. In interactive settings, multimodal resources do not substantially increase the final semantic similarity compared to linguistic-only interaction. Instead, their primary benefit lies in improving alignment efficiency. Specifically, multimodal scenarios required an average of 3 interaction rounds to reach convergence, whereas text-only interaction required 4.5 rounds on average, with some cases demanding up to 11 rounds. Thus, multimodal grounding reduces the interaction overhead needed to achieve alignment.

Overall, these results indicate that multimodal information enhances interpretation quality and reduces clarification effort, but interactive alignment remains essential to ensure reliable semantic grounding before translating purposes into motivational structures.

## C. Mission and Drive Generation Test

This test evaluates the Translation Engine’s capacity to transform refined human purposes into structured motivational representations through two interconnected stages: (1) decomposing purposes into missions, and (2) generating mathematical drive functions for each mission. To isolate these stages from spurious effects introduced by the Alignment Engine, no feedback from the mission or drive evaluation was used to further disambiguate the purpose

during these tests.

The inputs to the mission generator were the aligned purposes obtained from the experiment described in Section IV-B. The resulting list of missions was then used as input to the drive generation stage, allowing this second step to be evaluated independently from unrelated input errors. To assess the validity of the missions and drives, an expert manually classified each output as valid or invalid. To this end, several factors were considered. First, we assessed whether the complete set of missions would enable the robot to achieve the intended human purpose and whether the associated weight distribution was appropriate. Additionally, each generated drive function was individually examined to ensure both mathematical correctness and proper implementation in Python. It was also verified that each function reached zero, or decreased, only under the appropriate task conditions. For clarity, Table III presents the set of missions and drives obtained for the ideal Arrange apples by color purpose defined in Table II.

TABLE III: Missions and drive functions for the Arrange apples by color ideal purpose.
<table><tr><td rowspan=1 colspan=1>Mission [label, weight]</td><td rowspan=1 colspan=1>Drive</td></tr><tr><td rowspan=1 colspan=1>[&quot;red_apple_in_hand”, 0.2]</td><td rowspan=1 colspan=1> $\overline { { | g r i p p e r - 1 | + | x _ { r o b } - x _ { r e d } | } }$  $+ | y _ { r o b } - y _ { r e d } |$ </td></tr><tr><td rowspan=1 colspan=1>[&quot;red_apple_right_side&quot;, 0.4]</td><td rowspan=1 colspan=1> $\overline { { \operatorname* { m a x } ( 0 , 0 . 5 - x _ { r e d } ) } }$ </td></tr><tr><td rowspan=1 colspan=1>[&quot;green_apple_in_hand&quot;, 0.6]</td><td rowspan=1 colspan=1> $| g r i p p e r - 1 | + | x _ { r o b } - x _ { g r e e n } |$  $+ | y _ { r o b } - y _ { g r e e n } |$ </td></tr><tr><td rowspan=1 colspan=1> $\mathrm { \Gamma ^ { \prime \prime } g r e e n . a p p l e . l e f t . s i d e ^ { \prime \prime } , 0 . 8 } \mathrm { ] }$ </td><td rowspan=1 colspan=1> $\overline { { \operatorname* { m a x } ( 0 , \ : x _ { g r e e n } - 0 . 5 ) } }$ </td></tr></table>

The first two missions in Table III are related to grasping and placing the red apple in the corresponding location, while the last two focus on grasping and placing the green apple in its correct position. Overall, the weight distribution follows an increasing pattern, consistent with the training prompt, where later missions are assigned higher weights to reflect the intended task order. For readability, the drive functions are presented in simplified form, although they were originally expressed in Python code.

It can be observed that the drive functions associated with grasping evaluate to zero only when the robot is holding an object $( g r i p p e r = 1 )$ and the (x, y) positions of the apple and the robot hand coincide. On the other hand, the drives associated with placing the apples are structurally similar. For the red apple, the function depends on its x-position: if $x _ { r e d } < 0 . 5$ (left side of the table), the drive is non-zero, and it becomes zero only when $x _ { r e d } > 0 . 5 .$ , thus ensuring placement on the right side of the table. A similar behavior is observed for the green apple, but in this case the goal is the opposite: the drive encourages the apple to be placed on the left side of the table $( x _ { g r e e n } < 0 . 5 )$

A similar manual analysis was conducted for each set of missions and drives. As illustrated in Fig.3, scenarios involving human feedback led to a significantly higher percentage of valid missions than scenarios without user intervention. Although the autonomous multimodal configuration (multimodal without feedback) occasionally exhibited superior performance, its success remains highly dependent on the visual context provided and the complexity of the task. Consequently, this setting cannot guarantee consistent reliability. Therefore, only scenarios involving human feedback were used for the final drive evaluation.

![](images/58a8272f570b97af1ba1435a3568b3e964f629ee3c7bc52b62552c0ed5714227.jpg)  
Fig. 3: Validity of generated missions across the three tested purposes. The ideal purpose (Table II) had a validity rate of 100% across the three purposes.

Regarding the generation of drive functions, Fig. 4shows that validity rates varied across tasks. While the system performed very well on the Classify cylinders by color task, it was less successful on the Stack cylinders by size task, failing in 6 out of 15 instances. Compared with the ideal case, performance on the first two purposes closely matches the ideal results, whereas performance on the stacking task is substantially worse in both the multimodal and unimodal settings. The reason for this behavior is that, as the purpose becomes more complex and requires a more detailed description, the Alignment Engine becomes more prone to producing ambiguous purposes. When these ambiguities propagate to the drive generation stage of the Translation Engine, they lead to inconsistent performance and, in some cases, invalid drives.

We observed that when the purpose is well specified, repeated runs through the Translation Engine produce essentially the same valid set of drives. However, if the purpose contains ambiguity, repeated runs through the Translation Engine often produce different sets of drives, some of which are invalid. This variability can therefore be interpreted as a signal that the purpose remains underspecified, which could be used to trigger further dialogue between the Alignment Engine and the user.

In general, these findings demonstrate that the Translation Engine can adequately translate fully defined user purposes into motivational structures. When failures occur, they are usually caused by ambiguities in the purpose definition, which indicates that the Alignment Engine should be triggered to refine that definition.

## D. Autonomous Robot Test

Once the complete framework, including the three LLM instances for alignment, mission generation, and drive modeling, had been validated, it was tested on a robot to evaluate the full system operation. The framework was integrated into the e-MDB cognitive architecture [5], [12], which has a motivational system based on needs, missions, and drives.

In this experiment, the robot’s purpose was to place a red cylinder inside a box and press a green button whenever the red cylinder was outside its reachable area. The system interpreted the human purpose, generated the missions and drive functions, and executed the task autonomously. Fig. 5 shows the experimental setup. A video of the robot’s performance is available in a public repository <sup>3</sup>.

![](images/d06388312c12fe9abe78ecf41e8e1fc19bb995bbb987e50516f8d6b1e5beea1f.jpg)  
Fig. 4: Validity of generated drive functions across the three tested purposes and the success rate for the ideal purposes (Table II).

![](images/67146abf82b1f8bf806770436e58cb4c0e14c73c6f7bc6b17cd6a32c8d3f288d.jpg)  
Fig. 5: Simulated and real environment used to validate the integration of the generated missions and drives within the e-MDB cognitive architecture.

The experiment was run ten times to check the consistency of the robot’s behavior. To obtain a performance metric, each run of the experiment included 50 trials, in which we recorded how many iterations the robot needed to achieve the human purpose in each case. Fig.6 shows these results.

The results shown in Fig. 6 indicate that it consistently reached the goal in 3–4 iterations, which is close to the practical minimum given the initial positions of the objects on the table. Notably, the robot reached the goal with very few iterations, suggesting a more directed behavior than what would typically be expected from novelty-based exploration alone. Moreover, the robot quickly converges to the correct sequence of actions, guided by the rewards induced by the generated drives. These results demonstrate that the generated missions and drives enabled the robot to complete the task accurately and efficiently.

## V. CONCLUSIONS AND FUTURE WORK

This paper introduced a multimodal framework for aligning human purposes with the motivational system of an autonomous robot. By integrating three LLM-based components for semantic alignment, mission generation, and drive modeling, the approach enables non-expert users to specify goals in natural language while grounding them in a structured, motivation-driven control architecture.

![](images/262b9a547eb9078578857aee7bc458bee37f6fe5ee24bca0d53825afb2a22d44.jpg)  
Fig. 6: Number of iterations required for the robot to achieve the assigned purpose using the Alignmnet and Translation Engines compared to using just novelty.

Experimental results demonstrate that interactive alignment significantly improves semantic precision, confirming that clarification dialogue is essential to reduce ambiguity and reliably translate human purposes into executable moti vational structures. While multimodal inputs enhance interpretation efficiency and provide contextual information, iterative human feedback remains crucial for robust alignment. The generated missions and drive functions were found to satisfy the required formal constraints and, when embedded in a cognitive architecture, support consistent goal-directed behavior.

Overall, the proposed system shows the potential of large multimodal models to serve as semantic mediators within a developmental, motivation-based architecture, while maintaining interpretability and alignment. Future work will expand multimodal interaction capabilities and investigate adaptive alignment mechanisms that evolve through prolonged human–robot interaction.

## REFERENCES

[1] G. Baldassarre, M. Mirolli et al., Intrinsically motivated learning in natural and artificial systems. Springer, 2013.

[2] V. G. Santucci, G. Baldassarre, and M. Mirolli, “Which is the best intrinsic motivation signal for learning multiple skills?” Frontiers in neurorobotics, vol. 7, p. 22, 2013.

[3] S. Doncieux, D. Filliat, N. D´ıaz-Rodr´ıguez, T. Hospedales, R. Duro, A. Coninx, D. M. Roijers, B. Girard, N. Perrin, and O. Sigaud, “Openended learning: a conceptual framework based on representational redescription,” Frontiers in neurorobotics, vol. 12, p. 59, 2018.

[4] A. Romero, G. Baldassarre, R. J. Duro, and V. G. Santucci, “H-grail: A robotic motivational architecture to tackle open-ended learning challenges,” IEEE Transactions on Cognitive and Developmental Systems, pp. 1–16, 2025.

[5] J. A. Becerra, A. Romero, F. Bellas, and R. J. Duro, “Motivational engine and long-term memory coupling within a cognitive architecture for lifelong open-ended learning,” Neurocomputing, vol. 452, pp. 341– 354, 2021.

[6] A. Romero, F. Bellas, and R. J. Duro, “A perspective on lifelong openended learning autonomy for robotics through cognitive architectures,” Sensors, vol. 23, no. 3, p. 1611, 2023.

[7] A. Romero, G. Baldassarre, R. J. Duro, and V. G. Santucci, “Autonomous learning of multiple curricula with non-stationary interdependencies,” in 2022 IEEE International Conference on Development and Learning (ICDL). IEEE, 2022, pp. 272–279.

[8] A. Romero, G. Baldassarre, R. Duro, and V. G. Santucci, “Autonomous discovery and learning of interdependent goals in non-stationary scenarios,” in 2024 IEEE International Conference on Development and Learning (ICDL). IEEE, 2024, pp. 1–8.

[9] A. Romero, F. Bellas, J. A. Becerra, and R. J. Duro, “Motivation as a tool for designing lifelong learning robots,” Integrated Computer Aided Engineering, vol. 27, no. 3, pp. 353–372, 2020.

[10] G. Baldassarre, R. J. Duro, E. Cartoni, M. Khamassi, A. Romero, and V. G. Santucci, “A formalisation of the purpose framework: the autonomy-alignment problem in open-ended learning robots,” arXiv preprint arXiv:2403.02514, 2025. [Online]. Available: https: //arxiv.org/abs/2403.02514

[11] Z. Li, Y. Mu, Z. Sun, S. Song, J. Su, and J. Zhang, “Intention understanding in human–robot interaction based on visual-nlp semantics,” Frontiers in Neurorobotics, vol. 14, p. 610139, 2021. [Online]. Available: https://doi.org/10.3389/fnbot.2020.610139

[12] A. Romero, F. Bellas, J. A. Becerra, and R. J. Duro, “Motivation as a tool for designing lifelong learning robots,” Integrated Computer-Aided Engineering, vol. 27, no. 4, pp. 353–372, 2020. [Online]. Available: https://doi.org/10.3233/ICA-200633

[13] A. Barredo Arrieta, N. D´ıaz-Rodr´ıguez, J. Del Ser, A. Bennetot, S. Tabik, A. Barbado, S. Garcia, S. Gil-Lopez, D. Molina, R. Benjamins, R. Chatila, and F. Herrera, “Explainable artificial intelligence (xai): Concepts, taxonomies, opportunities and challenges toward responsible ai,” Information Fusion, vol. 58, pp. 82–115, 2020. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S1566253519308103

[14] F. Stramandinoli, V. Tikhanoff, U. Pattacini, and F. Nori, “Grounding speech utterances in robotics affordances: An embodied statistical language model,” in 2016 Joint IEEE International Conference on Development and Learning and Epigenetic Robotics (ICDL-EpiRob). IEEE, 2016, pp. 79–86.

[15] X. Huang, T. Chen, M. Bryson, T. Chaffey, H. Chen, K.-K. R. Choo, I. R. Manchester et al., “Trust in llm-controlled robotics: a survey of security threats, defenses and challenges,” arXiv preprint arXiv:2601.02377, 2025.

[16] A. Brohan, Y. Chebotar, C. Finn, K. Hausman, A. Herzog, D. Ho, J. Ibarz, A. Irpan, E. Jang, R. Julian et al., “Do as i can, not as i say: Grounding language in robotic affordances,” in Conference on robot learning. PMLR, 2023, pp. 287–318.

[17] B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid et al., “Rt-2: Vision-language-action models transfer web knowledge to robotic control,” in Conference on Robot Learning. PMLR, 2023, pp. 2165–2183.

[18] J. Park, S. Lim, J. Lee, S. Park, M. Chang, Y. Yu, and S. Choi, “Clara: classifying and disambiguating user commands for reliable interactive robotic agents,” IEEE Robotics and Automation Letters, vol. 9, no. 2, pp. 1059–1066, 2024.

[19] E. Yousefi, M. Chen, and I. Sharf, “Shared autonomy policy finetuning and alignment for robotic tasks,” The International Journal of Robotics Research, p. 02783649241312699, 2025.

[20] T. Kaufmann, P. Weng, V. Bengs, and E. Hullermeier, “A survey¨ of reinforcement learning from human feedback,” Transactions on Machine Learning Research, 2024.

[21] S. Zhang, L. Dong, X. Li, S. Zhang, X. Sun, S. Wang, J. Li, R. Hu, T. Zhang, G. Wang et al., “Instruction tuning for large language models: A survey,” ACM Computing Surveys, vol. 58, no. 7, pp. 1–36, 2026.

[22] J. Wang and Y. Dong, “Measurement of text similarity: A survey,” Information, vol. 11, no. 9, p. 421, 2020. [Online]. Available: https://www.mdpi.com/2078-2489/11/9/421

[23] SBERT.net, “Sentence-bert: Sentence embeddings using siamese bertnetworks,” https://www.sbert.net/, 2023, accessed: 2025-07-21.