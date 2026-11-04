# CLARA: A Cognitively-Driven Clarification Framework for Grounding Ambiguous Requests

Dimitrios Bekris<sup>1</sup>, Marios Glytsos<sup>1</sup>, Panagiotis Filntisis<sup>1</sup>, Niki Efthymiou<sup>1</sup>, George Retsinas<sup>1</sup> and Petros Maragos<sup>2</sup>

Abstract— Robotics and AI systems are increasingly deploying large vision-language models (LVLMs) to follow high-level instructions, from routine household chores to more complex organizational tasks. Yet, broad directives like “clean the room” or “organize the desk” frequently omit crucial details about user preferences and specific arrangement of objects that influence the desired outcome. Without any further guidance, such models tend to overestimate their grasp of ambiguous instructions and frequently assume they have sufficient understanding despite missing important details. In this paper, we propose CLARA (Conversational cLArification for Robotic Arrangement), a cognitively-driven clarification framework that proactively elicits key information from users, in a similar fashion as humans’ clarification seeking behavior. Through targeted questioning, CLARA refines ambiguous requests into structured, detailed instructions describing precisely how objects should be organized or rearranged, according to the user’s preferences. These clarifications then guide a diffusion-based image generation pipeline, producing realistic visual renderings of the final, desired scene. This final rendering also serves as a visual confirmation of the system’s understanding from the user. Experimental results and user studies demonstrate that CLARA leads to more accurate and consistent outcomes compared to another State-of-the-Art LVLM system, highlighting the importance of interactive clarification for bridging the gap between vague directives and fully grounded task specifications.

## I. INTRODUCTION

Robotics and intelligent systems/agents increasingly rely on large vision-language models (LVLMs) [1], [2], [3], [4], [5] to interpret and execute human instructions, in an effort to simplify tasks ranging from household chores to complex workflows[6], [7], [8], [9]. Crucial to their effectiveness is accurately grounding ambiguous high-level commands, such as “clean the room” into precise visual outcomes according to the desires of the instructor, since this can mean drastically different things based on an individual’s expectations, cultural norms, or personal preferences. Unlike structured tasks in controlled environments, everyday scenarios often involve vague instructions and considerable variation in how individuals imagine the outcomes, making the task of

LVLMs more difficult to reliably translate user intent from limited textual input alone.

From a cognitive standpoint, clarification-seeking behavior is recognized as a fundamental part of how humans manage ambiguous information. Conversation is viewed as a cooperative, joint activity in which speakers and listeners work together to ensure mutual understanding (i.e., establishing “common ground”). Developmentally also, adults often scaffold children’s understanding by prompting them to explain or by clarifying instructions (since initially children tend to overestimate their understanding of an ambiguous message), which over time teaches children how to resolve ambiguities through conversation. Interestingly, LVLMs as well tend to in many cases overestimate their understanding of the user’s query, and do not engage in dialogue to refine their understanding. In this paper, inspired by such clarificationseeking behavior, we aim to equip LVLM-powered robotic agents with a proactive questioning mechanism designed to clarify the user’s intent.

Most existing grounding methods for vision-language models depend on single-turn, prompt-based interactions, assuming initial instructions fully represent user intentions [10], [11]. While those methods are effective for simple direction, they frequently struggle with high level ambiguous requests. For instance, an instruction like ”clean the room” can lead to diverse visual interpretations based on different individual expectations regarding organization or aesthetics. Although recent diffusion-based image generation models [12], [11] have significantly improved prompt to image translation, they rely on explicitly detailed prompts therefore limiting the applicability where user intentions remain underspecified. More adaptive approaches incorporate conversational grounding, allowing users iterative clarification through dialogue [13], [14], [15]. Yet, current methods mostly allow small, step by step improvements without actively asking users for important clarifications, making them less useful in real situations.

To overcome these limitations, in this paper, we propose a framework that enables robotic agents to actively ask targeted questions, effectively grounding and clarifying abstract user goals like “clean this room”, converting them into specific, visually coherent outcomes. More specifically, we propose a chatbot-based framework designed to ground ambiguous user goals through targeted questioning and iterative interaction. Our approach initiates a guided dialogue, proactively asking clarifying questions in an effort to capture the user’s precise visual intent. Leveraging these input clarifications, our system generates a visualization of the final scene, in order to verify with the user that it correctly understood their intentions.

In summary, our key contributions are:

• A conversational grounding framework inspired by human clarification-seeking behavior, that proactively engages users through targeted clarifying questions to accurately determine visual intent.

• Integration of the interactive dialogue with image editing and diffusion-based image generation to better align the user’s intentions(goals) with the system’s “understanding”, and verify the system’s clear understanding of the instruction.

• A comparative user study with a a state-of-the-art LVLM-based system (Gemini 2.0), which shows that LVLMs augmented with such clarification seeking behavior are significantly preferred by the users in terms of their effectiveness for task completion.

## II. RELATED WORK

Grounding in Vision-Language Models Vision-language grounding methods typically focus on establishing accurate correspondences between textual instructions and visual representations. Early methods, such as Visual Dialog [16] and GuessWhat?! [17], established conversational tasks requiring agents to interpret user questions grounded explicitly in visual scenes. However, these primarily tested understanding rather than sicking clarification or iterative goal alignment.

Recent developments in large vision-language models (LVLMs) such as LLaVA [3], [4] have enhanced open-ended dialogue about images, yet their interactions remain largely user-driven, without the system clarifying ambiguous commands. Models like ChatterBox [18] have advanced multiturn conversational grounding by refining spatial references through dialogue but still emphasize reference resolution e.g. identification and location of objects through rather than task and goal oriented visual manipulation.

Clarifying Ambiguous Instructions Effective communication between human and machines often involves clarifying ambiguous instructions through dialogue, a challenge increasingly recognized in human-AI interaction research. Min et al. [19] highlight the need for robotic agents to handle underspecified instructions through implicit inference or explicit questioning, highlighting that actively clarifying user intentions through interaction is essential for accurately capturing and achieving their desired outcomes. Selective clarification approaches like CLAM [20] demonstrate decision making based on uncertainty estimates but primarily address linguistic ambiguity rather than visual task grounding. Despite recognizing the value of clarification, current LVLMs rarely autonomously initiate clarification questions and tend to overestimate their information for completing tasks. Our work directly addresses this gap by enabling the chatbot to proactively query the user in order to clarify ambiguous instructions into precise visual outcomes.

Recent conversational editing approaches, including DialogPaint [13] and ChatEdit [14], address iterative refinement through dialogue but mainly focus on incremental adjustments after initial generation, without proactive clarification. These methods iteratively apply image edits based on sequential user instructions, typically building upon previous outputs. Consequently, minor inaccuracies introduced in early stages may compound over successive interactions, gradually diverging from the user’s original intent.

Multimodal Planning and Goal Decomposition Recent multimodal planning frameworks, such as HuggingGPT [21] and GenArtist [22], demonstrate the effectiveness of breaking complex visual tasks into manageable sub-goals orchestrated by language models. HuggingGPT leverages LLMs for task decomposition and multi-step execution across diverse models, while GenArtist utilizes iterative verification to ensure accurate task execution.

Although these planning frameworks excel at structured decomposition, they typically rely on automated inference without interactive user clarification. Embodied agents like SayCan [23] introduce physical grounding through feasibility checks but similarly lack explicit conversational clarifications from the user.

## III. METHOD

The goal of CLARA is to align abstract user instructions, such as “organise this room”, with expected visual outcomes by clarifying the user’s intent. After clarifying the desired structure, the system proceeds to generate an image of the final scene, in order to verify that it understood the user’s instructions. The pipeline of the system can be seen in Figure 1. We first provide the high level overview of the system and subsequently describe each stage in detail.

## A. System Overview

Input and Goal Specification. The user provides a single image depicting the initial scene (e.g., a room) together with a concise, abstract instruction (e.g., “clean the room”). Since such commands are inherently ambiguous, the system must uncover the user’s actual preferences for object placement, storage, and arrangement.

Conversational Clarification. Leveraging a large visionlanguage model (LVLM), our system engages with the user to clarify how they want to transform the scene (Sect. III-B). For example, if the user simply says “organize the desk”, the system might ask: ’Should books be placed on the shelf, or remain stacked on the table?

Visual Generation and Refinement. After confirming that it has sufficiently clarified the initial vague instruction, the system then translates these refined directives into a final image. This process involves segmenting and repositioning objects, discarding any specified items, and employing a generative model to achieve realistic completions. (Sec. III-C).

## B. Conversational Clarification via Chatbot

To handle abstract or underspecified commands, we employ a custom LVLM-powered chatbot based on

![](images/498691df674276dfb5620c831d9d0ae875d8bed1f89728628aac42c639b30695.jpg)  
Fig. 1: CLARA system pipeline. Given an input image and a high-level goal, an LVLM-based chatbot refines the user’s intentions via clarifying questions. The refined specification then guides scene segmentation, object rearrangement, inpainting, and final image generation.

LLAMA 3.2 (11B)[24]. Unlike typical single-turn promptbased systems, our chatbot actively asks targeted clarification questions. It is initialized with guidelines instructing it to:

• Focus on spatial relations. Elicit which objects the user wants moved, stored, or grouped together.

• Confirm containers and storage details. Pinpoint which items go into specific containers, plates, bins, etc.

• Stop once sufficient clarity is achieved. Upon judging that it has enough information, it provides a concise textual description of the final desired state, ensuring no important details are missing.

This way, the system can transform an initial vague request (e.g., “clean the room”) into a structured description specifying how each object should be handled. For example:

“Books: move to left shelf. Place all pens and notebooks in the top drawer. Clear trash from floor into bin.”

This clarifying dialogue significantly reduces ambiguity and sets the stage for visually accurate scene editing which happens in the next stage.

## C. Visual Generation and Refinement

After generating the textual structured description, the system generates an updated image illustrating the requested changes. This step helps verify that the system has successfully clarified the desires of the user.

The visual generation and refinement stage follows the next steps:

a) Grounded Segmentation.: To identify and segment the individual objects within the original scene based on their textual description, we adopt Grounded-SAM, an integration of Grounding DINO [25] and Segment Anything Model (SAM) [26]. Grounding DINO provides open-set object detection (i.e., it can localize arbitrary objects given textual queries), while SAM refines those detections into precise segmentation masks.

b) Scene Rearrangement.: Now, using the structured directives produced by the Chatbot (e.g., “move the lamp to the corner table”), each identified object is (automatically) virtually repositioned or removed. In cases where the user wants an object hidden (e.g., “throw away all crumpled papers”), we remove the relevant masks from the original image.

c) Coarse Inpainting.: Because the previous object removal or movement leaves visual gaps in the original image, we initially fill these voids with LaMa inpainting [27], a specialized model for reconstructing missing regions. Although LaMa produces coherent completions, the results can appear slightly disjointed or lack the full realism often expected in complex scenes.

d) Stable Diffusion Refinement.: To achieve a more photorealistic and harmonious final image, we pass the inpainted output through a Stable Diffusion model [12], conditioning on the user’s clarified description. By conditioning on both textual guidance (e.g., “the desk is free of clutter, with the books neatly on the shelf”) and the partially inpainted image, Stable Diffusion refines object boundaries, ensures consistent lighting, and enhances overall realism. This conditioning step allows the model to preserve crucial details of the scene layout while seamlessly integrating any newly inpainted or rearranged content.

![](images/7c5efa97cf97b25d4b60826db59cceae4bd6eb53613c38ea70b20b4eb7d2c996.jpg)  
Fig. 2: The abstract scenarios designed for the experimental setup. An unorganized desk (left), a dinner table with food to serve (middle), and a dirty desk (right).

Finally, the system presents the rendered scene to the user for verification. If the user indicates that the arrangement does not fully capture their desires or requests further modifications, the chatbot re-engages to clarify any outstanding ambiguities. This iterative process can continue until the user confirms that the final image accurately reflects their intended outcome.

e) Verification.: Finally, the system presents the rendered scene to the user for evaluation. This ensures that the user can confirm whether the final image accurately reflects their intended outcome.

## IV. RESULTS AND EVALUATION

## A. Experimental Setup

a) Dataset and Prompts: We designed three distinct abstract scenarios involving common household settings, each associated with ambiguous, high-level instructions. The scenarios include:

Scenario 1: Organize a Desk. The model is provided with an image of an unorganized desk with various items and is asked to ”tidy the desk up”.

Scenario 2: Serving food into platters. The model is provided with an image of a dinner table with fruits and platters and is asked to ”serve the food”.

Scenario 3: Clean a desk. The model is provided with an image of a dirty desk with trash and other various items and is asked to ”clean it.”

The three images of these scenarios can be seen in Figure 2.

b) Methods: We assess the effectiveness of CLARA by comparing it against Gemini 2.0 Flash (Experimental) [28], a next-generation multimodal model from Google AI that is also capable of generating images. We also perform an ablation study where we combine the conversational clarification part of CLARA with the image generation capabilities of Gemini (i.e., we feed Gemini with the final clarified directives extracted by CLARA). We call this model CLARA+Gemini.

## B. Quantitative and Qualitative results

In order to quantitively assess the performance of each system to successfully clarify the abstract instruction of the user we run for each system and each scenario 10 times using the same prompt, summing up to a total of 30 comparisons per model. More specifically, we ran Gemini in singlestep mode: for each prompt (e.g., “Organize this desk.”), Gemini generated 10 images by varying the random seed. For CLARA, we conducted a single interactive dialogue per prompt, asking for the same object arrangement. Once finalized, we generated one final image from our pipeline.

We report two main quantitative measures:

a) Standard Deviation in Embedding Space.: We extracted an image embedding for each output using CLIP [29] and measured the standard deviation across the 10 generations. Lower values indicate more consistent outputs. As shown in Table I, CLARA’s outputs exhibit lower standard deviation.

b) Cosine Similarity: Similarly, we measured the mean value of pairwise cosine similarity among the embeddings. Here, higher values indicate more consistent outputs. As shown in Table I, CLARA’s outputs exhibit higher cosine similarity.

TABLE I: Standard deviation and mean of cosine similarity for each model. CLARA achieves more consistent visual results when prompted with the same instruction.
<table><tr><td>System</td><td>stdev</td><td>cos</td></tr><tr><td>Gemini 2.0</td><td>0.1</td><td>0.92</td></tr><tr><td>CLARA+Gemini 2.0</td><td>0.14</td><td>0.88</td></tr><tr><td>CLARA</td><td>0.08</td><td>0.95</td></tr></table>

The result consistency does indicate a measure of the system’s understanding of the user’s desires - however there could be cases where if the system does consistently something not asked by the user (e.g., just removing all objects from the scene), then the result is consistent, but not aligned with the user’s purpose. Subsequently, we show qualitative results but also a user study to verify that our consistent results are also aligned with the human’s goal.

c) Qualitative Results: We show an example of chatting with CLARA in Figure 3. Also, Figure 4 shows sample outputs from Gemini along with outputs of CLARA for the same prompt, “Serve the food” Gemini’s outputs vary widely, sometimes placing objects in unrealistic positions or generating objects that were never requested. In contrast, our pipeline yields a desired scene that closely follows the user’s specifications.

## C. User Study

To further assess our system from a human perspective, and assess that it correctly aligns the visual result with the user’s desires, we conducted a user study in order to compare our conversational grounding approach against Gemini.

Participants and Procedure We recruited 6 participants, each participant interacted with all three systems (in random order) for the three scenarios (Tidying a Desk, Organizing a Floor, and Arranging a Kitchen). Participants then answered the following three questions using a 5-point Likert scale:

1) ”The final image accurately reflects my original intention.”

![](images/5fef8a803c1798407c7f38a3aeaebafb7ea9bb0c3365a7521c2d736da3ff9442.jpg)  
Fig. 3: Example of clarification seeking behavior from CLARA. First the system seeks to clarify the abstract goal of tidying up and what exactly it refers to. Then, the system asks to specify how exactly the user wants the objects to be organized. When the dialog ends and CLARA has all the information needed, it generates the new scene and shows it to the user for verification.

2) ”The system successfully retained the static scene components (i.e., the desk/tables were not changed visually).”

3) ”I am satisfied with the interaction with the system.”

Participants’ responses (averaged) are shown in Table II.

As shown in Table II, the standard Gemini 2.0 approach (lacking clarification-seeking behavior) yields notably lower scores for both user intent fulfillment and overall satisfaction. By contrast, incorporating clarification-seeking (i.e., the CLARA systems) significantly improves these metrics. Moreover, regarding static or unmovable scene elements, the CLARA systems demonstrate a stronger capability to preserve the underlying structure of the scene while repositioning objects as specified. Interestingly, passing the extracted directives back into Gemini sometimes led to unsatisfactory outcomes; however, in certain cases where the directives were sufficiently abstract for our image-editing module to parse, Gemini’s generative process performed better.

TABLE II: User Study Ratings (Mapped to 1–5 Likert Scale) for the three systems.
<table><tr><td>System</td><td>Q1 (Intent)</td><td>Q2 (Scene)</td><td>Q3 (Satisfaction)</td></tr><tr><td>Gemini 2.0</td><td>1.40</td><td>3.00</td><td>1.30</td></tr><tr><td>CLARA+Gemini 2.0</td><td>2.97</td><td>3.60</td><td>3.17</td></tr><tr><td>CLARA</td><td>2.90</td><td>3.89</td><td>3.37</td></tr></table>

## D. Discussion

Our experiments highlight the value of clarificationseeking behavior in large vision-language models for abstract/vague commands. LVLMs when left without specific guidance tend to overestimate their capabilities in grounding vague commands, resulting in many cases in drastically different outcomes compared to the desires or the instructor. As shown both in our qualitative results, as well as our user study and quantitative metrics, the proposed system tends to generate both more consistent results (as demonstrated by Table I), but also results that are aligned with the user’s intensions, as demonstrated by our user study. Moving forward it is important to instill such kind of clarification seeking behavior more implicitly in such models during training, imitating how initially children tend to overestimate their understanding, while adults engage in dialogue to further establish common ground.

## V. CONCLUSION

This paper presented CLARA, a cognitively-inspired framework that proactively resolves ambiguous directives through targeted clarification questions. By modeling the clarification-seeking process inherent in human communication and development, we demonstrated that CLARA achieves more reliable visual outcomes, aligned with the user’s intention In the future, we plan to extend CLARA’s capabilities to dynamic robot operations in real-world environments. One promising direction is to integrate the final target image into a reinforcement learning loop, using visual similarity or arrangement correctness as a reward mechanism.

## REFERENCES

[1] J. Zhang, J. Huang, S. Jin, and S. Lu, “Vision-language models for vision tasks: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[2] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou, “Qwen-vl: A frontier large vision-language model with versatile abilities,” arXiv preprint arXiv:2308.12966, 2023.

[3] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” in NeurIPS, 2023.

[4] H. Liu, C. Li, Y. Li, and Y. J. Lee, “Improved baselines with visual instruction tuning,” 2023.

[5] H. Lu, W. Liu, B. Zhang, B. Wang, K. Dong, B. Liu, J. Sun, T. Ren, Z. Li, Y. Sun, C. Deng, H. Xu, Z. Xie, and C. Ruan, “Deepseek-vl: Towards real-world vision-language understanding,” 2024.

[6] S. S. Kannan, V. L. Venkatesh, and B.-C. Min, “Smart-llm: Smart multi-agent robot task planning using large language models,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 12 140–12 147.

[7] J. Wang, E. Shi, H. Hu, C. Ma, Y. Liu, X. Wang, Y. Yao, X. Liu, B. Ge, and S. Zhang, “Large language models for robotics: Opportunities, challenges, and perspectives,” Journal ofAutomation and Intelligence, 2024.

In the tidy image, the markers are stored in a large box and the small boxes are stored in a bowl.

In the desired state, the green plate contains bananas and the orange plate contains tomatoes.

![](images/3c68931088456c8e4cc8d5fe6853ba372ccc23c636276def61cd69a5f95c535a.jpg)

![](images/d10945dafa5c752052ec567555aebeddf9574c433615291eba38e2b9e59dfb21.jpg)

In the desired state, the paper is in the trash, and the postcards and pen are together in the tin box.

Original Image  
![](images/08bbec377577762556bba35ce701a01a6a51f389cb39512bbcbffdbd369a1dfa.jpg)  
(a) Gemini

![](images/bc23818b97884ce23c993686ee52172e48441dd3fab307302148922a1d8da05c.jpg)  
(b) CLARA+Gemini

![](images/11c15129f93ada84bd9e4d5dc9c1094c5e5fee145ce7ba733b2c26543934a71d.jpg)  
(c) CLARA  
Fig. 4: Visual outputs of each model in each of the abstract scenarios described: “Organize a desk”, “Serve food to platters”, and “Clean a desk”. As we can see, CLARA achieves a visual result which is aligned with the textual instruction while retaining the whole scene structure. Gemini either introduces new objects (such as new platters) or cannot capture the user’s purpose (e.g. removing most objects on the table), since it does not perform clarification seeking behavior. CLARA+Gemini performs better than Gemini, but also introduces new objects on the scene (e.g. a new bowl) or erroneously modifies some objects.

[8] F. Zeng, W. Gan, Y. Wang, N. Liu, and P. S. Yu, “Large language models for robotics: A survey,” arXiv preprint arXiv:2311.07226, 2023.

[9] J. Gao, B. Sarkar, F. Xia, T. Xiao, J. Wu, B. Ichter, A. Majumdar, and D. Sadigh, “Physically grounded vision-language models for robotic manipulation,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 12 462–12 469.

[10] T. Brooks, A. Holynski, and A. A. Efros, “Instructpix2pix: Learning to follow image editing instructions,” arXiv preprint arXiv:2211.09800, 2022.

[11] Z. Lai, X. Zhu, J. Dai, Y. Qiao, and W. Wang, “Mini-dalle3: Interactive text to image by prompting large language models,” 2023. [Online]. Available: https://github.com/Zeqiang-Lai/Mini-DALLE3

[12] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” 2021.

[13] J. Wei, S. Wu, X. Jiang, and Y. Wang, “Dialogpaint: A dialog-based image editing model,” 2023. [Online]. Available: https://arxiv.org/abs/2303.10073

[14] X. Cui, Z. Li, P. Li, Y. Hu, H. Shi, and Z. He, “Chatedit: Towards multi-turn interactive facial image editing via dialogue,” in Conference on Empirical Methods in Natural Language Processing, 2023. [Online]. Available: https://api.semanticscholar.org/CorpusID:257631998

[15] T. T. Nguyen, Z. Ren, T. Pham, T. T. Huynh, P. L. Nguyen, H. Yin, and Q. V. H. Nguyen, “Instruction-guided editing controls for images and multimedia: A survey in llm era,” arXiv preprint arXiv:2411.09955, 2024.

[16] A. Das, S. Kottur, K. Gupta, A. Singh, D. Yadav, J. M. F. Moura, D. Parikh, and D. Batra, “Visual dialog,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017.

[17] H. de Vries, F. Strub, S. Chandar, O. Pietquin, H. Larochelle, and A. C. Courville, “Guesswhat?! visual object discovery through multi-modal dialogue,” in Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

[18] Y. Tian, T. Ma, L. Xie, J. Qiu, X. Tang, Y. Zhang, J. Jiao, Q. Tian, and

Q. Ye, “Chatterbox: Multi-round multimodal referring and grounding,” arXiv preprint arXiv:2401.13307, 2024.

[19] S. Y. Min, X. Puig, D. S. Chaplot, T.-Y. Yang, A. Rai, P. Parashar, R. Salakhutdinov, Y. Bisk, and R. Mottaghi, “Situated instruction following,” in ECCV, 2024.

[20] L. Kuhn, Y. Gal, and S. Farquhar, “CLAM: Selective Clarification for Ambiguous Questions with Large Language Models,” arXiv, Dec. 2022.

[21] Y. Shen, K. Song, X. Tan, D. Li, W. Lu, and Y. Zhuang, “Hugginggpt: Solving ai tasks with chatgpt and its friends in huggingface,” in Advances in Neural Information Processing Systems, 2023.

[22] W. Zhenyu, L. Aoxue, L. Zhenguo, and L. Xihui, “Genartist: Multimodal llm as an agent for unified image generation and editing,” arXiv preprint arXiv:2407.05600, 2024.

[23] M. Ahn, A. Brohan, and N. B. . A. Zeng, “Do as i can and not as i say: Grounding language in robotic affordances,” in arXiv preprint arXiv:2204.01691, 2022.

[24] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan et al., “The llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.

[25] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, C. Li, J. Yang, H. Su, J. Zhu et al., “Grounding dino: Marrying dino with grounded pre-training for open-set object detection,” arXiv preprint arXiv:2303.05499, 2023.

[26] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollar, and R. Gir-´ shick, “Segment anything,” arXiv:2304.02643, 2023.

[27] R. Suvorov, E. Logacheva, A. Mashikhin, A. Remizova, A. Ashukha, A. Silvestrov, N. Kong, H. Goka, K. Park, and V. Lempitsky, “Resolution-robust large mask inpainting with fourier convolutions,” arXiv preprint arXiv:2109.07161, 2021.

[28] G. Team, R. Anil, S. B. ..., and O. Vinyals, “Gemini: A family of highly capable multimodal models,” 2024. [Online]. Available: https://arxiv.org/abs/2312.11805

[29] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger,

and I. Sutskever, “Learning transferable visual models from natural language supervision,” CoRR, 2021. [Online]. Available: https://arxiv.org/abs/2103.00020