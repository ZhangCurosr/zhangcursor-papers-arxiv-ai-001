# IRWOZ 2.0: A Large Language Model-driven Dialogue Dataset for Industrial Robot Conversations

Chen Li<sup>1\*</sup> and Dimitrios Chrysostomou<sup>1</sup>

<sup>1\*</sup>Department of Materials and Production, Aalborg University, Fibigerstraede 16, Aalborg, 9220, Denmark.

\*Corresponding author(s). E-mail(s): cl@mp.aau.dk; Contributing authors: dimi@mp.aau.dk;

## Abstract

IRWOZ has improved industrial human-robot interaction (HRI) dialogue systems through domain-specific annotations. However, its initial version contains substantial noise in dialogue states and utterances, limiting state-tracking accuracy. We introduce IRWOZ 2.0, which addresses these limitations through large language model (LLM) enhanced generation (Mistral/Claude-3.5) and quality refinements. Our improved dataset expands to 390 dialogues across 4 industrial domains (Assembly, Delivery, Position, Relocation), featuring manual corrections and automated typo removal. Benchmark experiments on dialogue state tracking demonstrate significant improvements, with GPT-2’s BLEU-4 score increasing from 0.1651 to 0.5604 compared to original IRWOZ. To support industrial HRI research, we publicly released IRWOZ 2.0 dataset at https://ieee-dataport.org/documents/irwoz-20-large-language-model-drivendialogue-dataset-industrial-robot-conversations

Keywords: Industrial human-robot interaction, Corpus building, Dialogue dataset, Wizard-of-Oz, Large language models, Dialogue state tracking

## 1 Introduction

The rapid development of artificial intelligence (AI) has led to its increased adoption across industries, with human-robot interaction (HRI) playing a significant role in enhancing productivity and eficiency Campagna et al. (2024). Natural language-enabled AI agents show particular potential for revolutionizing HRI by facilitating seamless communication between humans and robots Li et al. (2023a); Zheng et al. (2024).

![](images/d3dbb864876a7d2d9001d8073c9b86529afd7af5e5391ee579050a0e015388bc.jpg)  
Fig. 1: Example dialogue for phone assembly tasks in industrial settings, demonstrating operator-robot interaction sequences.

However, developing such agents requires high-quality dialogue datasets that accurately capture industrial complexities Kernan Freire et al. (2023); Ni et al. (2023). Figure 1 illustrates a typical three-turn dialogue between a human operator and industrial robot during phone assembly tasks.

Traditional dialogue dataset creation relies on manual collection and annotation through Wizard-of-Oz (WoZ) experiments. This approach presents substantial challenges in industrial environments, where data collection proves both time-consuming and costly. Capturing diverse industrial scenarios remains particularly challenging, while multi-annotator involvement often introduces inconsistencies that limit scalability for comprehensive domain coverage Li et al. (2023b).

Our previous IRWOZ dataset Li et al. (2023b) represents significant progress in addressing these challenges. However, manual collection and annotation processes still introduce substantial noise in both annotations and utterances, negatively impacting dialogue state tracking and response generation performance Yi et al. (2024). Recent analyses indicate error rates exceeding 20% in manually annotated industrial dialogues Yi et al. (2024).

To address these limitations, we present IRWOZ 2.0 - a systematic framework combining large language models (LLMs) with three key innovations for industrial dialogue dataset enhancement: (1) automated LLM-powered error detection/correction, (2) structured prompt engineering for domain-specific generation, and (3) hybrid validation integrating automated checks with expert review. Our implementation leverages Mistral Jiang et al. (2023) and Claude-3.5 Anthropic (2024) to expand the original dataset to 390 dialogues across four critical industrial domains (Assembly, Delivery, Position, Relocation), achieving 3.1× faster generation than manual methods while preserving technical accuracy. Benchmark results demonstrate this framework’s efectiveness through GPT-2’s improved BLEU-4 scores for dialogue state tracking (0.1651 → 0.5604), alongside maintained domain specificity in complex industrial scenarios. The contribution of the paper is as follows:

• Enhanced Data Quality: Implementation of hybrid quality control measures (automated and manual) reducing annotation errors while improving dialogue state tracking reliability

• Expanded Domain Coverage: Comprehensive coverage across four industrial domains (Assembly, Delivery, Position, Relocation) with 390 dialogues capturing diverse interaction scenarios

• LLM Integration: Systematic framework combining Mistral Jiang et al. (2023) and Claude-3.5 Anthropic (2024) for eficient, context-aware dialogue generation

• Rigorous Evaluation: Comprehensive benchmarking and human evaluation protocols assessing performance improvements over IRWOZ

## 2 Related Work

In this section, we review three key areas relevant to our work: task-oriented dialogue datasets, the use of LLMs for dataset correction, and LLMs for dataset generation. These areas form the foundation for our approach in creating IRWOZ 2.0, a novel dataset for industrial robot conversations.

## 2.1 Task-Oriented Dialogue Datasets

Task-oriented dialogue datasets play a crucial role in advancing research in dialogue systems Reimann et al. (2024). Traditional pipeline models for task-oriented dialogue systems often require explicit modeling of dialogue states and hand-crafted action spaces to interact with domain-specific knowledge bases Wen et al. (2018). Taskoriented dialogue systems aim to assist users in completing specific tasks such as booking flights, making restaurant reservations, or providing information. Traditional datasets for such systems are often created through manual annotation or by collecting real-world interactions. For instance, the ATIS (Airline Travel Information System) dataset Hemphill et al. (1990) has been a benchmark for spoken language understanding systems, containing flight information queries.

Another significant dataset is the MultiWOZ dataset Budzianowski et al. (2018), which is a large-scale multi-domain dialogue dataset containing interactions spanning over multiple domains and services. These datasets have been instrumental in advancing the field by providing structured and annotated data for training and evaluating dialogue systems. More recently, the Schema-Guided Dialogue (SGD) dataset Rastogi et al. (2020) has been introduced, which includes over 20,000 dialogues spanning 16 domains. SGD aims to provide a more comprehensive and flexible framework for taskoriented dialogue systems by incorporating schema information that describes the entities and actions within each domain. The Frames dataset El Asri et al. (2017) is another example, which focuses on the challenge of handling implicit information in task-oriented dialogues, providing a testbed for systems to infer and act upon such information.

However, even though these datasets have significantly advanced dialogue systems research, they primarily focus on general domains such as travel and hospitality. The industrial robotics domain, with its unique vocabulary and task-specific interactions, remains underrepresented. This gap underscores the need for specialized datasets like IRWOZ 2.0 that can capture the nuances of HRI in industrial settings.

## 2.2 LLMs for Dataset Correction

LLMs have become pivotal in the realm of dataset corrections, efectively addressing a spectrum of issues such as typographical errors within dialogue datasets. With their deep understanding of language structure and context, LLMs are adept at detecting and rectifying typographical errors, ensuring the dataset’s text is clean and coherent. CoachLM represents a significant approach, automatically enhancing the quality of instruction pairs in datasets Liu et al. (2024). The model’s efectiveness is demonstrated by its real-world application at Huawei, where it increased dataset cleaning eficiency by 20%. Studies also explore LLMs’ self-correction abilities, indicating that external feedback and large-scale fine-tuning bolster their reliability Kamoi et al. (2024). Additionally, GrammarGPT has shown promise in correcting grammatical errors in native Chinese text using a hybrid dataset Fan et al. (2023). In software engineering, LLMs predict implicit dataflows in dynamically typed code for dataflow graph generation, outperforming traditional static analysis methods Huang et al. (2024). However, applying LLMs to correct specialized datasets such as those for industrial robotics presents unique challenges, including the need to preserve domain-specific terminology and task-relevant information while improving overall quality.

## 2.3 LLMs for Dataset Generation

The use of LLMs has revolutionized the way dialogue datasets are generated. LLMs, such as GPT Radford et al. (2018) and BERT Devlin et al. (2018), have been finetuned on dialogue data to generate realistic and coherent conversational data. These models can produce large volumes of dialogue data that can be used for training and testing dialogue systems. For example, Adiwardana et al. (2020) introduced a model that generates responses indistinguishable from human-written ones, showcasing the potential of LLMs in creating high-quality dialogue datasets by enabling the generation of diverse and nuanced dialogue scenarios.

In addition to generating new datasets, LLMs have been used to enhance existing datasets by augmenting them with more diverse and contextually relevant responses. For instance, the work by Zhou et al. (2023) explores the use of LLM for text-to-text generation task, to improve conversational machine reading datasets, making them more robust and adaptable to real-world applications. Furthermore, the DialoGPT model Zhang et al. (2020) represents a significant step forward in generating conversational responses, demonstrating the ability of LLMs to capture the nuances of human dialogue.

It is evident, that the traditional datasets ofer the advantage of domain-specific knowledge and structured annotations, however, they often require significant manual efort and are limited in size and diversity. On the other hand, LLM-generated datasets provide scalability and a broader range of conversational contexts but may lack the depth of domain-specific knowledge. Our work with IRWOZ 2.0 represents a hybrid approach, leveraging LLMs to generate scalable, diverse, and industry-relevant dialogue datasets while maintaining focus on specific industrial domains. This method is particularly important for advancing dialogue systems capable of handling the complex demands of real-world industrial applications, an area where existing LLM applications have been limited.

## 3 Dataset Corrections

To create IRWOZ 2.0, we first addressed the limitations and errors present in the original IRWOZ dataset. The original IRWOZ dataset was collected using a Wizard-of-OZ setup among 18 participants, including students, academics, and shop floor workers with backgrounds ranging from robotics to computer science. The participants were randomly paired as two to construct dialogues. One participant played the role of the Wizard and the other plays the shop floor worker. The shop floor worker is asked to choose a task and initiate a dialogue in one of domains, Delivery, Position, Assembly and Relocation. The wizard needed to respond to the shop floor worker according to the required task. The web application was constructed with separate interfaces, one for the user and one for the Wizard. The wizard was given access to the back-end database and a list of the robot-controlling APIs to verify the resource requested by user.

Regardless the great efort to develop such web application for dialogue dataset collection and annotation, along with implementing human validation processes, there are many mistakes in IRWOZ such as incomplete markups, and mis-annotations. These errors arise from misinterpretations, inconsistencies among annotators, and the complexities of contextual nuances in dialogues. Such inaccuracies can significantly afect the quality of the dataset and the performance of dialogue systems.

## 3.1 Error Types

The most common errors types in the original dialogue dataset:

• Typos. The text, including user utterances, responses, and slots, contains typographical errors that can afect its accuracy and efectiveness. For example, in Figure 2, the phrase ”can you assembly 8 blue phones for me?” The word ”assembly” should be replaced by ”assemble.”

• Incomplete mark-ups. The slot value was not fully extracted from the user’s input. This frequently happens with compound nouns. For instance, in Figure 2, we observe this scenario in which the term ”shop floor” denotes the physical location of the laboratory, yet only ”shop” is extracted from the user’s utterance.

<table><tr><td rowspan=1 colspan=1>Error Type</td><td rowspan=1 colspan=1>Conversation Example</td><td rowspan=1 colspan=1>IRWOZ</td><td rowspan=1 colspan=1>IRWOZ 2.0</td></tr><tr><td rowspan=1 colspan=1>Typos</td><td rowspan=1 colspan=1>user: &quot;Hello friend! Can you assemblyeight blue phones for me?&quot;,system: &quot;Of course. When do you want tobegin?&quot;,s_system: &quot;What a lovely day!&quot;</td><td rowspan=1 colspan=1>&lt;|user|&gt; hello friend!can you assembly eightblue phones for me?</td><td rowspan=1 colspan=1>&lt;|user|&gt; hello friend! canyou assemble eight bluephones for me?</td></tr><tr><td rowspan=1 colspan=1>Incompletemark-ups</td><td rowspan=1 colspan=1>user: &quot;It&#x27;s the shop floor of the lab.&quot;,system: &quot;All right. When do you want todiliver?&quot;,s_system: &quot;&quot;,</td><td rowspan=1 colspan=1>&lt;|DB_req|&gt; deliveryarea=lablocation=shop</td><td rowspan=1 colspan=1>&lt;|DB_req|&gt; deliveryarea=lablocation=shop floor</td></tr><tr><td rowspan=1 colspan=1>Mis-annotations</td><td rowspan=1 colspan=1>user: &quot;Hey robot, is it possible to get methe PCB?&quot;,system: &quot;&quot;,s_system: &quot;Of course it is. Let me seewhere it is. Here you go! &quot;</td><td rowspan=1 colspan=1>&lt;|boSres|&gt; Of course itis. Let me see where itis. Here you go!&lt;|eoSres|&gt;</td><td rowspan=1 colspan=1>&lt;|boTres|&gt; Of course itis.&lt;|eoTres|&gt;&lt;|boSres|&gt; Of course it is.Let me see where it is. Hereyou go! let me see where itis. &lt;|eoSres|&gt;</td></tr></table>

Fig. 2: Examples of annotation errors between IRWOZ and IRWOZ 2.0 (< |DB req| >, $< | u s e r | > , < | b o S r e s | > , < | e o S r e s | , < | b o T r e s | > \mathrm { a n d } < | e o T r e s | >$ are special tokens defined to annotate the dialogue dataset Li et al. (2023b).

• Mis-annotations. The value is assigned to an incorrect slot type. For example, in Figure 2, the robot’s response to the shop floor work, ”Let me see where it is,” is intended to be task related. However, it is incorrectly marked as small talk.

## 3.2 Dialogue Utterance Corrections

Dialog dataset correction was done through Manual corrections and Automated corrections with the help of the LLM, Claude-3.5.

Manual Corrections. Manual corrections were employed to address misannotations and incomplete markups identified within the dataset. This process involved two steps:

• Slot Values Verification: verification of each slot value was made to determine if it was correctly marked against the corresponding user utterance.

• Response Verification: check if the given utterance was annotated against the dialogue context and against the task requirements.

These manual corrections were essential for maintaining the integrity of the dataset, especially in cases where nuanced understanding and contextual judgement were required. This process was particularly crucial for preserving the accuracy of domainspecific terminology and task-related information unique to industrial robotics scenarios. The Table 1 shows the statistical analysis results of the mis-annotations and incomplete markups, i.e., 2.2% of the turns have incomplete markups and 3.2% of the turns have incorrect annotations.

Automated Corrections.To enhance the eficiency and accuracy of the correction process, automated corrections were implemented using Claude-3.5. This LLM was specifically fine-tuned to recognize and preserve industrial terminology while correcting typographical errors. The following steps were involved in the automated correction process:

<table><tr><td>Error Type</td><td>% of errors</td></tr><tr><td>Typos and Misspellings</td><td>4.3%</td></tr><tr><td>Incomplete Markups</td><td>2.2%</td></tr><tr><td>Mis-annotations</td><td>3.2%</td></tr></table>

Table 1: Statistical analysis of manual and automated corrections of the IRWOZ dataset.

• Prompt Definition: A specific prompt (see Appendix A) was defined to guide Claude-3.5 in detecting and correcting typos in the dialogue dataset.

• Execution: The IRWOZ dataset were uploaded as a separate document, where Claude-3.5 processed the dataset according to the defined prompt, systematically identifying and correcting typographical errors.

The Table 1 shows the statistical analysis results of typos in the IRWOZ, i.e., 4.3% of the turns contain typos or misspellings. The correction process significantly improved the overall quality of the dataset. By reducing typos, incomplete markups, and misannotations, we had substantial improvements in the performance of dialogue systems trained on this data. This is particularly important in industrial settings where miscommunication between humans and robots can lead to ineficiencies or safety issues.

## 4 Dataset Generation

Even though the IRWOZ was collected through WoZ method, which allows for rich, realistic interactions by simulating system responses with human operators, the collection and annotation of such dialogue datasets for HRI in industrial domains is a laborintensive and time-consuming task. Therefore, we leverage the LLMs’ capabilities to generate realistic and contextually relevant dialogues that reflect the language-enabled HRIs in four industrial domains.

## 4.1 Overall Framework

Figure 3 illustrates the overall architecture of the dialogue generation framework for generating the IRWOZ 2.0 dataset. The framework’s foundation lies in the detailed design of task-specific prompts for each industrial domain. The framework is composed of three components:

• Prompt Instructions. It includes 1) dataset structure definition including user input, dialogue belief state, system action, and corresponding system responses, including both task-oriented and small talk responses; 2) database information, facilitating accurate generation and verification of belief states and system actions, from key database tables (e.g., area, product) incorporated into the prompt, and 3) task requirements specifying behavioral constraints for the language models, such as the protocol for informing users about non-existent products in the database.

![](images/723d433b18f87cffb33579185b8e4b69ecff06aaaebec584e3432417b5b2e2eb.jpg)  
Fig. 3: The overall architecture of the dialogue generation framework for building the IRWoZ 2.0 dataset.

• LLMs: Claude-3.5 and Mistral, are leveraged to generated the datasets based on the given prompt instructions.

• Dialogue Datasets: generated dialogues for each industrial domain (i.e., Delivery, Position, Assembly, and Relocation).

## 4.2 Prompt Design

While LLMs are highly efective in handling conversations, the quality and style of their responses are largely dependent on the prompts they receive from users. Well-set up and articulate prompts are imperative for LLMs to exhibit the desired behavior.

The primary objective is to ensure the generated dialogue dataset is contextually relevant and technically accurate. This requires a clear definition of the industrial task which the prompt is trying to address — a detailed background of how the task is usually carried out as well as the specific roles of humans and robots in the corresponding scenario. As an example, a prompt designed to generate a dialogue for a relocation task would need to specify: the items to be relocated, where to relocate them to, and any barriers that could be run into. The prompt should also encourage the generation of dialogues that incorporate common industrial terminologies and commands, ensuring that the robot’s responses are precise and aligned with the industry domain.

![](images/8e87733feb7f855f49036d0d163d929acb4fabe0f08594bba4b5063a02e06cda.jpg)  
Fig. 4: An example of designed prompt for the Assembly task for Claude-3.5 and Mistral

Furthermore, the prompt is designed to simulate realistic HRI by incorporating natural language elements such as disambiguation, confirmation, and error-handling. This requires creating scenarios in which the human operator may ask the robot to report the status of the task, verify a correct action, or fix any problem during the task. For example, the generated response for the relocation task like “Has the PCB been correctly put on the conveyor belt?” or ”What to do if the PCB cannot be found in the desired location.” By embedding such interactions in the conversations, the generated dialogues are able to address the operational aspects of the tasks, enhance the robot’s ability to communicate efectively, and respond to unexpected situations. Moreover, due to the fact that LLMs are trained on vast and diverse text data across various contexts, they can efectively mimic human speech styles and generate human-like responses. This capability aligns with one of the key objectives of the IRWOZ dataset: to provide both task-related and small talk responses, thereby enhancing the user experience during HRI.

Our optimal prompt for the Assembly task is shown in Figure 4. The prompt begins with a required dataset structure of the desired dialogue. The special tokens from IRWOZ Li et al. (2023b) are leveraged to annotate the dialogue context, dialogue belief, system actions and system response. Each of the placeholder, which is annotated with colored ’[]’,is given the corresponding explanation. It is then followed by an example from previous IRWOZ dataset to provide extra information to instruct the dataset generation process. We use the enumerated list to point out all the requirements that constrain the LLM’s behavior. When we provide only the high level goal instruction to Claude-3.5 and Mistral, we use the term ”Task”. Similar prompts are designed for the other three tasks, Relocation, Deliver and Position. Appendix C shows two generated dialogue samples for the Assembly task using Claude-3.5 and Mistral.

<table><tr><td>Task</td><td>Role</td><td>Min</td><td>Max</td><td>Avg</td></tr><tr><td>Assembly</td><td>User</td><td>3</td><td>10</td><td>6.98</td></tr><tr><td rowspan="4">Delivery</td><td>Task Resp.</td><td>9</td><td>48</td><td>23.29</td></tr><tr><td>ST Resp.</td><td>4</td><td>22</td><td>12.90</td></tr><tr><td>User</td><td>3</td><td>14</td><td>8.91</td></tr><tr><td>Task Resp.</td><td>8</td><td>62</td><td>26.71</td></tr><tr><td rowspan="4">Position</td><td>ST Resp.</td><td>0</td><td>22</td><td>10.30</td></tr><tr><td>User</td><td>4</td><td>21</td><td>8.22</td></tr><tr><td>Task Resp.</td><td>6</td><td>47</td><td>25.41</td></tr><tr><td>ST Resp.</td><td>2</td><td>26</td><td>11.96</td></tr><tr><td rowspan="3">Relocation</td><td>User</td><td>6</td><td>24</td><td>10.77</td></tr><tr><td>Task Resp.</td><td>5</td><td>56</td><td>21.35</td></tr><tr><td>ST Resp.</td><td>3</td><td>21</td><td>11.67</td></tr></table>

Table 2: Sentence length (words) distribution in IRWOZ 2.0. Task Resp.: Taskrelated responses, ST Resp.: Small talk responses

## 4.3 Data Statistics

A total of 390 dialogues were generated and annotated across four industrial domains: Assembly, Delivery, Position, and Relocation. The distribution of dialogue turns for each domain in Appendix B. Each of the above domain has around 54%-55% 1 turn dialogues, 33%-35% 2 turns dialogues and only around 10% -11% are 3 turns dialogues. Which makes sense since keeping short and concrete dialogues for work related HRI is major happens in shop floor.

Table 2 illustrates the distribution of sentence length for user utterances, taskrelated responses, and small talk responses. The average length of user utterances varies across domains, ranging from 6.98 words for Assembly to 10.77 words for Relocation. Task-related responses are generally longer, with averages ranging from 21.35 words for Relocation to 26.71 words for Delivery. Small talk responses are consistently shorter across all domains, with averages between 10.30 and 12.90 words. As expected, each domain contains system responses without small talk involved. The varied response lengths contribute to improved generalization ability for training models. The maximum sentence length for task-related responses reaches up to 62 words in the Delivery domain, while user utterances and small talk responses have lower maximum lengths, enhancing the dataset’s diversity and realism.

Figure 5 illustrates varied sentence length distributions across four domains. User utterances show task-specific ranges, with Relocation the widest (6-24 words) and Assembly the narrowest (3-10 words). Task responses consistently have the broadest distributions, peaking at 62 tokens for Delivery, indicating detailed explanations in industrial settings. Small talk responses are briefer (0-26 words) and more consistent

![](images/67a6f30d9457e9dd2d9ad68ece259ffd552474d277f5f55a445bd480e0499c0e.jpg)  
(a) User utterance length distribution

![](images/bb81bcd3099a1296035e764fe6ac9fcf2fe0c8c02afa76ef826ccfd4dc434abf.jpg)  
(b) Task response length distribution

![](images/9facc30355d9429cae7e7f0772abf7dea78d18c075ae7d587c6b32127101257b.jpg)  
(c) Small talk responses length distribution  
Fig. 5: Distribution of sentence lengths across four industrial domains (Assembly, Delivery, Position, and Relocation) in IRWOZ 2.0.

across domains. Notably, Delivery exhibits the most variation overall, suggesting higher complexity. These patterns reflect diverse communication needs in industrial dialogues, with task responses requiring the most linguistic flexibility, while small talk remains concise across all domains.
<table><tr><td>Model</td><td>Dataset</td><td>Perplexity</td><td>BLEU-1</td><td>BLEU-2</td><td>BLEU-3</td><td>BLEU-4</td><td>SA</td><td>JGA</td></tr><tr><td rowspan="2">GPT-2</td><td>IRWOZ</td><td>1.05</td><td>0.3816</td><td>0.2680</td><td>0.2088</td><td>0.1651</td><td>0.949</td><td>0.801</td></tr><tr><td>IRWOZ 2.0</td><td>1.10</td><td>0.7370</td><td>0.6581</td><td>0.6059</td><td>0.5604</td><td>0.944</td><td>0.824</td></tr><tr><td rowspan="2">GPT-2 Medium</td><td>IRWOZ</td><td>1.05</td><td>0.6214</td><td>0.5620</td><td>0.5352</td><td>0.5114</td><td>0.979</td><td>0.925</td></tr><tr><td>IRWOZ 2.0</td><td>1.11</td><td>0.7392</td><td>0.6633</td><td>0.6114</td><td>0.5656</td><td>0.956</td><td>0.855</td></tr><tr><td rowspan="2">GPT-2 Large</td><td>IRWOZ</td><td>1.05</td><td>0.6493</td><td>0.5949</td><td>0.5681</td><td>0.5446</td><td>0.983</td><td>0.938</td></tr><tr><td>IRWOZ 2.0</td><td>1.10</td><td>0.7399</td><td>0.6645</td><td>0.6134</td><td>0.5681</td><td>0.960</td><td>0.871</td></tr></table>

Note: BLEU (bilingual evaluation understudy), SA (Slot Accuracy), and JGA (Joint goal accuracy). Table 3: Performance comparison of three diferent model architectures on IRWoZ and IRWOZ 2.0 dataset

## 5 Benchmark Evaluation

## 5.1 Benchmark Models

To ensure a fair comparison between the proposed IRWOZ 2.0 and its predecessor, experiments are conducted using dialogues generated by LLMs from IRWOZ 2.0, allowing for a comparison with the original IRWOZ dataset. We also employed the same evaluation methodology using three GPT-2 language models: gpt2, gpt2-medium, and gpt2-large Li et al. (2023b). This comparison aims to quantify any performance diferences and assess whether LLM-generated datasets can match or potentially surpass the quality of manually curated ones in various dialogue system tasks.

## 5.2 Results and Discussion

As we can see in Table 3, one of the most notable improvements with IRWOZ 2.0 is the substantial increase in BLEU scores across all gpt2 model variants. IRWOZ 2.0 consistently outperforms the original IRWOZ dataset in BLEU metrics for all model sizes. For instance, the gpt2 model exhibits a BLEU-1 score increase from 0.3816 (IRWOZ) to 0.7370 (IRWOZ 2.0), and BLEU-4 from 0.1651 to 0.5604. Similar improvements are observed in gpt2-medium and gpt2-large models, with BLEU-4 scores reaching 0.5656 and 0.5681, respectively, compared to their IRWOZ counterparts. These enhancements indicate that IRWOZ 2.0 facilitates the generation of more fluent, relevant, and contextually appropriate dialogues, which are crucial for realistic HRIs.

IRWOZ 2.0 demonstrates notable improvements in joint goal accuracy (JGA), particularly with the base gpt2 model.The gpt2 model’s JGA increases from 0.801 (IRWOZ) to 0.824 (IRWOZ 2.0), showcasing a better ability to fulfill dialogue objectives. Although gpt2-medium and gpt2-large show higher JGA with the original IRWOZ, the base gpt2 model benefits significantly from IRWOZ 2.0, suggesting that even smaller models can achieve enhanced performance with the improved dataset.

Average Grades for Each Question  
![](images/9c75924ddc6fe0bf6ed5a39ff8f0e6f6ad54bde1b7e86a99825bb88db607202a.jpg)  
Fig. 6: The score distribution of diferent raters

IRWOZ generally exhibits slightly better perplexity scores across all model sizes, even if the diferences are marginal (IRWOZ: 1.05 vs. IRWOZ 2.0: 1.10 for gpt2, IRWOZ 2.0: 1.11 for gpt2-medium, and IRWOZ 2.0: 1.10 for gpt2-large). Given that BLEU scores and JGA are more indicative of dialogue quality and goal achievement, the slight increase in perplexity is outweighed by the substantial gains in language generation and goal accuracy. Though the original IRWOZ dataset generally shows higher slot accuracy (SA) scores, IRWOZ 2.0 maintains competitive performance. For the gpt2 model, SA remains high (0.944 for IRWOZ 2.0 vs. 0.949 for IRWOZ). Although gpt2- medium and gpt2-large experience a slight dip in SA with IRWOZ 2.0 (0.956 and 0.960 respectively) compared to IRWOZ (0.979 and 0.983), the overall SA scores are still commendable, ensuring reliable intent recognition and response generation.

Although some metrics like Perplexity show minor trade-ofs, the substantial improvements in BLEU scores and JGA underscore the dataset’s enhanced language generation capabilities and efectiveness in achieving dialogue objectives. These positive outcomes afirm that IRWOZ 2.0 is a valuable asset for training more sophisticated, responsive, and reliable HRI in industrial settings.

## 6 Human Evaluation

We also perform a human evaluation on the quality of the LLM generated dataset. We randomly sampled 40 dialogues (see Appendix D) from each domain and recruited 23 participants from diferent nationalities and background, including shop floor workers with industrial robot operation experience (≥ 3 years), robotics engineers, computational linguistics expert, robotics/AI researchers and students and industrial automation researchers. Specifically, we followed the similar strategy from Ye et al. (2022) and Sun et al. (2021), the raters were asked to assign a score from 1 to 10 for each question (see Appendix D) based on the following criteria: 1) Accurate: the generated dialogue demonstrates accurate understanding of context and extracting the correct slots from user’s request; 2) Knowledgeable: the dialogues contains meaningful and relevant industrial scenarios; 3) Engaging: the dialogue maintains natural interaction flow and encourages human-robot collaboration; 4) Human-like: the interaction feels natural while maintaining appropriate robot-specific characteristics.

The evaluation results (see Figure 6) demonstrate strong performance in accuracy metrics (Accurate Q1: 8.13, Q2: 8.39) particularly in contextual understanding of user intents, reflecting the LLM’s capability to maintain coherent task-oriented interactions. This aligns with findings from Ye et al. (2022) regarding the importance of precise slot-value annotation in dialogue systems. The knowledgeable dimension (Q1: 8.48, Q2: 7.39) reveals domain-specific expertise in industrial scenarios while indicating opportunities for improved integration of technical details during task execution phases. The engagement scores (Q1: 8.04, Q2: 8.30) suggest efective turn-taking mechanics but reveal challenges in maintaining dynamic interaction flows during complex multi-step operations. The human-like characteristics show the highest single-question score (Q1: 8.70) in politeness maintenance, while performance in language variation (Q2: 7.74) indicates room for improvement in generating diverse natural expressions.

## 7 Conclusions & Future Work

In this paper, we present IRWOZ 2.0, an advanced dialogue dataset for industrial HRI. It combines LLM-based generation with manual and automated correction processes, resulted in a more comprehensive and accurate dataset covering four key industrial domains: Assembly, Delivery, Position, and Relocation. Benchmarking results show that IRWOZ 2.0 outperforms its predecessor, with notable improvements in BLEU scores, Joint Goal Accuracy, and other key metrics across various model architectures. The public release of IRWOZ 2.0 will provide the research community with a valuable resource for developing more sophisticated and responsive dialogue systems for industrial HRI. Future work may focus on expanding to additional industrial domains, integrating multimodal data, and exploring transfer learning techniques to further improve model performance across various industrial scenarios.

Author Contributions. Author Chen Li prepared the initial manuscript, conducted data generation, and benchmark evaluation. Dimitrios Chrysostomou contributed to data correction and data analysis. All authors have reviewed, contributed to, and approved of the manuscript.

Data availability. Our IRWOZ dataset is available at https://github.com/lcroy/ToD4IR/tree/main/dataset, the IRWOZ 2.0 dataset is available for non-commercial purpose at: IEEE dataport: https://ieee-dataport.org/documents/irwoz-20-large-language-model-driven-dialoguedataset-industrial-robot-conversations.

## Declarations

Conflict of interest. All authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## References

Adiwardana D, Luong MT, Cao Y, et al (2020) Towards a human-like open-domain chatbot. arXiv preprint arXiv:200109977

Anthropic (2024) Meet claude. URL https://www.anthropic.com/claude, accessed 27 Jun 2024

Budzianowski P, Wen TH, Tseng BH, et al (2018) Multiwoz-a large-scale multi-domain wizard-of-oz dataset for task-oriented dialogue modelling. In: Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp 5016–5026

Campagna G, Lagomarsino M, Lorenzini M, et al (2024) Promoting trust in industrial human-robot collaboration through preference-based optimization. IEEE Robotics and Automation Letters

Devlin J, Chang MW, Lee K, et al (2018) Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:181004805

El Asri L, Schulz H, Sharma S, et al (2017) Frames: a corpus for adding memory to goaloriented dialogue systems. In: Jokinen K, Stede M, DeVault D, et al (eds) Proceedings of the 18th Annual SIGdial Meeting on Discourse and Dialogue. Association for Computational Linguistics, Saarbr¨ucken, Germany, pp 207–219, https://doi.org/10. 18653/v1/W17-5526, URL https://aclanthology.org/W17-5526

Fan Y, Jiang F, Li P, et al (2023) Grammargpt: Exploring open-source llms for native chinese grammatical error correction with supervised fine-tuning. URL https: //arxiv.org/abs/2307.13923, arXiv:2307.13923

Hemphill CT, Godfrey JJ, Doddington GR (1990) The atis spoken language systems pilot corpus. Proceedings of the workshop on Speech and Natural Language pp 96–101

Huang Q, Luo Z, Xing Z, et al (2024) Revealing the unseen: Ai chain on llms for predicting implicit dataflows to generate dataflow graphs in dynamically typed code. ACM Trans Softw Eng Methodol 33(7). https://doi.org/10.1145/3672458, URL https://doi.org/10.1145/3672458

Jiang AQ, Sablayrolles A, Mensch A, et al (2023) Mistral 7b. arXiv preprint arXiv:231006825

Kamoi R, Zhang Y, Zhang N, et al (2024) When can llms actually correct their own mistakes? a critical survey of self-correction of llms. URL https://arxiv.org/abs/ 2406.01297, arXiv:2406.01297

Kernan Freire S, Foosherian M, Wang C, et al (2023) Harnessing large language models for cognitive assistants in factories. In: Proceedings of the 5th International Conference on Conversational User Interfaces, pp 1–6

Li C, Chrysostomou D, Yang H (2023a) A speech-enabled virtual assistant for eficient human–robot interaction in industrial environments. Journal of Systems and Software 205:111818

Li C, Chrysostomou D, Zhang X, et al (2023b) Irwoz: Constructing an industrial robot wizard-of-oz dialoguing dataset. IEEE Access 11:28236–28251. https://doi.org/10. 1109/ACCESS.2023.3259325

Liu Y, Tao S, Zhao X, et al (2024) Coachlm: Automatic instruction revisions improve the data quality in llm instruction tuning. In: 2024 IEEE 40th International Conference on Data Engineering (ICDE). IEEE Computer Society, Los Alamitos, CA, USA, pp 5184–5197, https://doi.org/10.1109/ICDE60146.2024.00390, URL https://doi. ieeecomputersociety.org/10.1109/ICDE60146.2024.00390

Ni J, Young T, Pandelea V, et al (2023) Recent advances in deep learning based dialogue systems: A systematic survey. Artificial intelligence review 56(4):3055–3155

Radford A, Narasimhan K, Salimans T, et al (2018) Improving language understanding by generative pre-training. URL https://cdn.openai.com/research-covers/ language-unsupervised/language understanding paper.pdf

Rastogi A, Zang X, Sunkara S, et al (2020) Towards scalable multi-domain conversational agents: The schema-guided dialogue dataset. In: Proceedings of the AAAI Conference on Artificial Intelligence, pp 8689–8696

Reimann MM, Kunneman FA, Oertel C, et al (2024) A survey on dialogue management in human-robot interaction. ACM Transactions on Human-Robot Interaction

Sun K, Moon S, Crook P, et al (2021) Adding chit-chat to enhance task-oriented dialogues. In: Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. Association for Computational Linguistics, Online, pp 1570–1583, https://doi.org/10. 18653/v1/2021.naacl-main.124, URL https://aclanthology.org/2021.naacl-main.124/

Wen H, Liu Y, Che W, et al (2018) Sequence-to-sequence learning for task-oriented dialogue with dialogue state representation. In: Bender EM, Derczynski L, Isabelle P (eds) Proceedings of the 27th International Conference on Computational Linguistics. Association for Computational Linguistics, Santa Fe, New Mexico, USA, pp 3781– 3792, URL https://aclanthology.org/C18-1320

Ye F, Manotumruksa J, Yilmaz E (2022) MultiWOZ 2.4: A multi-domain taskoriented dialogue dataset with essential annotation corrections to improve state tracking evaluation. In: Proceedings of the 23rd Annual Meeting of the Special Interest Group on Discourse and Dialogue. Association for Computational Linguistics, Edinburgh, UK, pp 351–360, https://doi.org/10.18653/v1/2022.sigdial-1.34, URL https://aclanthology.org/2022.sigdial-1.34/

Yi Z, Ouyang J, Liu Y, et al (2024) A survey on recent advances in llm-based multi-turn dialogue systems. arXiv preprint arXiv:240218013

Zhang Y, Sun S, Galley M, et al (2020) DIALOGPT : Large-scale generative pre-training for conversational response generation. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics pp 270–278. https://doi.org/10. 18653/v1/2020.acl-demos.30, URL https://aclanthology.org/2020.acl-demos.30

Zheng T, Grosse EH, Morana S, et al (2024) A review of digital assistants in production and logistics: applications, benefits, and challenges. International Journal of Production Research pp 1–27

Zhou S, Ouyang S, Zhang Z, et al (2023) Towards end-to-end open conversational machine reading. In: Vlachos A, Augenstein I (eds) Findings of the Association for Computational Linguistics: EACL 2023. Association for Computational Linguistics, Dubrovnik, Croatia, pp 2064–2076, https://doi.org/10.18653/v1/2023.findings-eacl. 154, URL https://aclanthology.org/2023.findings-eacl.154

## Appendix A Prompt for correction

![](images/6213606f17f914add6bc36d9cdd14a18d697ad0e7eb16d8fb2de29176dceecf4.jpg)  
Fig. A1: The defined prompt for remove the typos from the IRWOZ dataset

# Appendix B Dialogue turn distribution of each domain

Figure B2 shows the generated dialogues turn distribution of four domains.

![](images/19fddf5f8bb08b229f8e74fd39d7f47e12c31b801cacf2a3c54c36d5a34b01b4.jpg)  
(a) Domain: Assembly

![](images/338b6b4db74926204c6ca10dc7685876f2ff06f2693f5015ef865776e61bad56.jpg)  
(b) Domain: Delivery

![](images/1f64d7095f14c548ba21f4f3942458d45706818b389556accc5e4947303f4155.jpg)  
(c) Domain: Position

![](images/bdd2beb22c0502d88c9aa2dcb5e9fa66bb5f8346a2c6bc6d27621aab24d97f3d.jpg)  
(d) Domain: Relocation  
Fig. B2: Dialogue turn distribution of each domain

## Appendix C Generated Dialogue Samples

Figure B3 shows two generated dialogue samples for the Assembly task.

## Appendix D Human Evaluation - Questions and Results

Table B1 shows the results of human evaluation on dialogue quality. Figure D4 and D5 show examples of randomaly selected LLM-generated dialogues for human evaluation.

![](images/2ff44b7cc5865222abd57e6cb53533ce5b3c26b625bff320d3921a1f6cf2d49b.jpg)  
Fig. B3: Two generated dialogue samples for the Assembly task using Claude-3.5 and Mistral

<table><tr><td>Category</td><td>Question</td><td>G3</td><td>G4</td><td>G5</td><td>G6</td><td>G7</td><td>G8</td><td>G9</td><td>G10</td></tr><tr><td rowspan="2">Engaging</td><td>Q1: The dialogue flow maintains a good pace without long delays or abrupt endings</td><td></td><td></td><td>3</td><td>1</td><td>4</td><td>5</td><td>4</td><td>6</td></tr><tr><td>Q2: The dialogue includes proper turn-taking and interaction dynamics</td><td></td><td></td><td>2</td><td>1</td><td>8</td><td>4</td><td>3</td><td>6</td></tr><tr><td rowspan="2">Knowledgeable</td><td>Q1: The dialogue covers the relevant four industrial</td><td></td><td></td><td>2</td><td>1</td><td>2</td><td>4</td><td>7</td><td>7</td></tr><tr><td>Q2: The dialogue maintains coherence throughout and includes appropriate task- specific details</td><td>2</td><td>1</td><td></td><td>3</td><td>2</td><td>9</td><td>4</td><td>2</td></tr><tr><td rowspan="2">Humanlike</td><td>Q1: The generated responses show appropriate levels of politeness</td><td>1</td><td></td><td></td><td>2</td><td>1</td><td>3</td><td>6</td><td>10</td></tr><tr><td>Q2: The dialogue includes natural language variations and expressions</td><td></td><td>1</td><td>1</td><td>2</td><td>4</td><td>9</td><td>3</td><td>3</td></tr><tr><td rowspan="2">Accurate</td><td>Q1: The dialogue demonstrates accurate understanding of the user&#x27;s intent</td><td></td><td>1</td><td>1</td><td>1</td><td>5</td><td>3</td><td>7</td><td>5</td></tr><tr><td>Q2: The responses show accurate understand- ing of the context of the conversation</td><td></td><td></td><td>1</td><td></td><td>3</td><td>10</td><td>3</td><td>6</td></tr></table>

Table B1: Results of human evaluation on dialogue quality. Each cell shows the number of human evaluators who gave that grade (G3-G10 represent grades from 3 to 10, where ’–’ indicates zero responses). Grades 1 and 2 are not shown in the table as no evaluators assigned these grades. The evaluation was conducted across four categories with two questions each, measuring diferent aspects of dialogue quality.

<table><tr><td rowspan=1 colspan=5>Assembly</td></tr><tr><td rowspan=1 colspan=1>Example ID</td><td rowspan=1 colspan=1>Turn ID</td><td rowspan=1 colspan=1>Role</td><td rowspan=1 colspan=2>Conversations</td></tr><tr><td rowspan=3 colspan=1>1</td><td rowspan=3 colspan=1>Turn 1</td><td rowspan=1 colspan=1>User</td><td rowspan=1 colspan=2>&quot;I need to assemble 3 spur gears.&quot;</td></tr><tr><td rowspan=2 colspan=1>System</td><td rowspan=1 colspan=1>TaskResponse</td><td rowspan=1 colspan=1>&quot;Certainly! I can help you assemble 3 spur gears. Spur gears are part of ourmechanical product type. Is there anything specific you&#x27;d like to know about theassembly process?&quot;,</td></tr><tr><td rowspan=1 colspan=1>Small TalkResponse</td><td rowspan=1 colspan=1>&quot;Spur gears are a great choice for many applications. I&#x27;m excited to help you withthis task!&quot;</td></tr><tr><td rowspan=6 colspan=1>2</td><td rowspan=3 colspan=1>Turn 1</td><td rowspan=1 colspan=1>User</td><td rowspan=1 colspan=1>&quot;Can you he</td><td rowspan=1 colspan=1>lp me assemble a phone?&quot;</td></tr><tr><td rowspan=2 colspan=1>System</td><td rowspan=1 colspan=1>TaskResponse</td><td rowspan=1 colspan=1>&quot;Sure, I can help with that. What color phone do you need?&quot;</td></tr><tr><td rowspan=1 colspan=1>Small TalkResponse</td><td rowspan=1 colspan=1>&quot;I can assemble different colors.&quot;</td></tr><tr><td rowspan=3 colspan=1>Turn 2</td><td rowspan=1 colspan=1>User</td><td rowspan=1 colspan=1>&quot;I need a white</td><td rowspan=1 colspan=1>phone.&quot;</td></tr><tr><td rowspan=2 colspan=1>System</td><td rowspan=1 colspan=1>TaskResponse</td><td rowspan=1 colspan=1>&quot;Sure, I can assemble a white phone for you. How many do you need?&quot;</td></tr><tr><td rowspan=1 colspan=1>Small TalkResponse</td><td rowspan=1 colspan=1>&quot;White phones are quite popular.&quot;,</td></tr><tr><td rowspan=6 colspan=1>3</td><td rowspan=3 colspan=1>Turn 1</td><td rowspan=1 colspan=1>User</td><td rowspan=1 colspan=1>&quot;I want to as</td><td rowspan=1 colspan=1>semble some lego toys.&quot;</td></tr><tr><td rowspan=2 colspan=1>System</td><td rowspan=1 colspan=1>TaskResponse</td><td rowspan=1 colspan=1>&quot;Sure, I can help with that. How many lego toys do you need and what color?&quot;</td></tr><tr><td rowspan=1 colspan=1>Small TalkResponse</td><td rowspan=1 colspan=1>&quot;Lego toys are always fun to assemble.&quot;</td></tr><tr><td rowspan=3 colspan=1>Turn 2</td><td rowspan=1 colspan=1>User</td><td rowspan=1 colspan=1>&quot;I need 3 gree</td><td rowspan=1 colspan=1>n lego toys.&quot;,</td></tr><tr><td rowspan=2 colspan=1>System</td><td rowspan=1 colspan=1>TaskResponse</td><td rowspan=1 colspan=1>&quot;Sure, I can assemble 3 green lego toys for you.&quot;</td></tr><tr><td rowspan=1 colspan=1>Small TalkResponse</td><td rowspan=1 colspan=1>&quot;Green lego toys are very popular.&quot;</td></tr></table>

<table><tr><td rowspan=1 colspan=5>Delivery</td></tr><tr><td rowspan=1 colspan=1>Example ID</td><td rowspan=1 colspan=1>Turn ID</td><td rowspan=1 colspan=1>Role</td><td rowspan=1 colspan=2>Conversations</td></tr><tr><td rowspan=3 colspan=1>1</td><td rowspan=3 colspan=1>Turn 1</td><td rowspan=1 colspan=1>User</td><td rowspan=1 colspan=2>&quot;Can you deliver a package to the warehouse in the lab?&quot;.</td></tr><tr><td rowspan=2 colspan=1>System</td><td rowspan=1 colspan=1>TaskResponse</td><td rowspan=1 colspan=1>&quot;Certainly! I can deliver the package to the warehouse in the lab. Is there anythingelse you&#x27;d like to specify about the delivery?&quot;,</td></tr><tr><td rowspan=1 colspan=1>Small TalkResponse</td><td rowspan=1 colspan=1>&quot;I&#x27;m happy to help with your delivery request!&quot;</td></tr><tr><td rowspan=6 colspan=1>2</td><td rowspan=3 colspan=1>Turn 1</td><td rowspan=1 colspan=1>User</td><td rowspan=1 colspan=1>&quot;I need to se</td><td rowspan=1 colspan=1>nd a package to the lab.&quot;,</td></tr><tr><td rowspan=2 colspan=1>System</td><td rowspan=1 colspan=1>TaskResponse</td><td rowspan=1 colspan=1>&quot;Certainly! I can help you with that. Which specific location in the lab would you likethe package delivered to?&quot;,</td></tr><tr><td rowspan=1 colspan=1>Small TalkResponse</td><td rowspan=1 colspan=1>&quot;I&#x27;m here to assist you with your delivery needs!&quot;.</td></tr><tr><td rowspan=3 colspan=1>Turn 2</td><td rowspan=1 colspan=1>User</td><td rowspan=1 colspan=1>&quot;Please deli</td><td rowspan=1 colspan=1>ver it to the warehouse in the lab.&quot;,</td></tr><tr><td rowspan=2 colspan=1>System</td><td rowspan=1 colspan=1>TaskResponse</td><td rowspan=1 colspan=1>&quot;Understood. I&#x27;ll arrange for the package to be delivered to the warehouse in thelab. Is there anything else you&#x27;d like to specify about the delivery?&quot;,</td></tr><tr><td rowspan=1 colspan=1>Small TalkResponse</td><td rowspan=1 colspan=1>&quot;We&#x27;re making great progress with your delivery request!&quot;,</td></tr><tr><td rowspan=3 colspan=1></td><td rowspan=3 colspan=1>Turn 1</td><td rowspan=1 colspan=1>User</td><td rowspan=1 colspan=1>&quot;Can you deli</td><td rowspan=1 colspan=1>ver a blue folder to Kate?&quot;,</td></tr><tr><td rowspan=2 colspan=1>System</td><td rowspan=1 colspan=1>TaskResponse</td><td rowspan=1 colspan=1>&quot;Certainly! I can arrange for a blue folder to be delivered to Kate. Could you pleasespecify the area or location where Kate is?&quot;,</td></tr><tr><td rowspan=1 colspan=1>Small TalkResponse</td><td rowspan=1 colspan=1>&quot;I&#x27;m happy to help with your delivery request!&quot;,</td></tr><tr><td rowspan=3 colspan=1>3</td><td rowspan=3 colspan=1>Turn 2</td><td rowspan=1 colspan=1>User</td><td rowspan=1 colspan=1>&quot;Kate is in ro</td><td rowspan=1 colspan=1>om 408 in Fib 16.&quot;,</td></tr><tr><td rowspan=2 colspan=1>System</td><td rowspan=1 colspan=1>TaskResponse</td><td rowspan=1 colspan=1>&quot;Thank you for the information. I&#x27;ll arrange for the blue folder to be delivered toKate in room 408, Fib 16. Is there anything else you&#x27;d like to add to this deliveryrequest?&quot;,</td></tr><tr><td rowspan=1 colspan=1>Small TalkResponse</td><td rowspan=1 colspan=1>&quot;We&#x27;re all set to make this delivery happen!&quot;.</td></tr></table>

Fig. D4: Examples of randomly selected LLM-generated dialogues from Assembly and Delivery domains

<table><tr><td rowspan=1 colspan=7>Position</td></tr><tr><td rowspan=1 colspan=1>Example ID</td><td rowspan=1 colspan=1>Turn ID</td><td rowspan=1 colspan=2>Role</td><td rowspan=1 colspan=3>Conversations</td></tr><tr><td rowspan=3 colspan=1>1</td><td rowspan=3 colspan=1>Turn 1</td><td rowspan=1 colspan=2>User</td><td rowspan=1 colspan=3>&quot;Can you guide me to the laser cell?&quot;,</td></tr><tr><td rowspan=2 colspan=2>System</td><td rowspan=1 colspan=2>TaskResponse</td><td rowspan=1 colspan=1>&quot;Certainly, I can help you move to the laser cell. The location isaccessible in our database. What would you like the robot to do?&quot;</td></tr><tr><td rowspan=1 colspan=2>Small TalkResponse</td><td rowspan=1 colspan=1>&quot;Is there anything specific you need to do at the laser cell?&quot;,</td></tr><tr><td rowspan=3 colspan=1>2</td><td rowspan=3 colspan=1>Turn 1</td><td rowspan=1 colspan=2>User</td><td rowspan=1 colspan=2>&quot;I need to g</td><td rowspan=1 colspan=1>o to the workbench.&quot;,</td></tr><tr><td rowspan=2 colspan=2>System</td><td rowspan=1 colspan=2>TaskResponse</td><td rowspan=1 colspan=1>&quot;Certainly, I can help you move to the workbench. The location isaccessible in our database. The robot will now start moving to theworkbench.&quot;,</td></tr><tr><td rowspan=1 colspan=2>Small TalkResponse</td><td rowspan=1 colspan=1>&quot;Do you have a specific task you need to perform at the workbench?&quot;,</td></tr><tr><td rowspan=3 colspan=1></td><td rowspan=3 colspan=1>Turn 2</td><td rowspan=1 colspan=2>User</td><td rowspan=1 colspan=2>&quot;Yes, I nee</td><td rowspan=1 colspan=1>d to assemble some components for a project.&quot;,</td></tr><tr><td rowspan=2 colspan=2>System</td><td rowspan=1 colspan=2>TaskResponse</td><td rowspan=1 colspan=1>&quot;Understood. We&#x27;re on our way to the workbench for your assemblytask. Is there any specific tool or equipment you&#x27;ll need for assemblingthe components?&quot;</td></tr><tr><td rowspan=1 colspan=2>Small TalkResponse</td><td rowspan=1 colspan=1>&quot;If you need any information about the tools available at theworkbench, feel free to ask.&quot;,</td></tr><tr><td rowspan=6 colspan=1>3</td><td rowspan=3 colspan=1>Turn 1</td><td rowspan=1 colspan=2>User</td><td rowspan=1 colspan=2>&quot;Take me to</td><td rowspan=1 colspan=1>the 3D printer, please.&quot;,</td></tr><tr><td rowspan=2 colspan=2>System</td><td rowspan=1 colspan=2>TaskResponse</td><td rowspan=1 colspan=1>&quot;Certainly, I can help you move to the 3D printer. The location isaccessible in our database. The robot will now start moving to the 3Dprinter.&quot;,</td></tr><tr><td rowspan=1 colspan=2>Small TalkResponse</td><td rowspan=1 colspan=1>&quot;Do you have a specific project in mind for the 3D printer?&quot;,</td></tr><tr><td rowspan=3 colspan=1>Turn 2</td><td rowspan=1 colspan=2>User</td><td rowspan=1 colspan=2>&quot;Yes, I need</td><td rowspan=1 colspan=1>to print a prototype for a new part.&quot;</td></tr><tr><td rowspan=2 colspan=2>System</td><td rowspan=1 colspan=2>TaskResponse</td><td rowspan=1 colspan=1>&quot;Understood. We&#x27;re heading to the 3D printer for your prototypingtask. Is the 3D model file ready for printing?&quot;</td></tr><tr><td rowspan=1 colspan=2>Small TalkResponse</td><td rowspan=1 colspan=1>&quot;If you need any assistance with the 3D printer setup or have questionsabout its capabilities, feel free to ask.&quot;</td></tr><tr><td rowspan=1 colspan=7></td></tr><tr><td rowspan=1 colspan=7>Relocation</td></tr><tr><td rowspan=1 colspan=1>Example ID</td><td rowspan=1 colspan=1>Turn ID</td><td rowspan=1 colspan=1>Role</td><td rowspan=1 colspan=4>Conversations</td></tr><tr><td rowspan=3 colspan=1>1</td><td rowspan=3 colspan=1>Turn 1</td><td rowspan=1 colspan=1>User</td><td rowspan=1 colspan=4>&quot;Can you bring me the bottom cover?&quot;.</td></tr><tr><td rowspan=2 colspan=1>System</td><td rowspan=1 colspan=2>TaskResponse</td><td rowspan=1 colspan=2>&quot;Certainly, I&#x27;ll bring you the bottom cover right away.&quot;,</td></tr><tr><td rowspan=1 colspan=2>Small TalkResponse</td><td rowspan=1 colspan=2>&quot;Is there anything else you need along with the bottom cover?&quot;.</td></tr><tr><td rowspan=6 colspan=1>2</td><td rowspan=3 colspan=1>Turn 1</td><td rowspan=1 colspan=1>User</td><td rowspan=1 colspan=2>&quot;I need the</td><td rowspan=1 colspan=2>large top cover from the storage room.&quot;</td></tr><tr><td rowspan=2 colspan=1>System</td><td rowspan=1 colspan=2>TaskResponse</td><td rowspan=1 colspan=2>&quot;I&#x27;ll get the large top cover from the storage room for you. Where would you likeme to bring it?&quot;,</td></tr><tr><td rowspan=1 colspan=2>Small TalkResponse</td><td rowspan=1 colspan=2>&quot;Do you need any assistance carrying or installing the large top cover?&quot;</td></tr><tr><td rowspan=3 colspan=1>Turn 2</td><td rowspan=1 colspan=1>User</td><td rowspan=1 colspan=2>&quot;Please br</td><td rowspan=1 colspan=2>ing it to the assembly area. And can you also get the corresponding PCB?&quot;,</td></tr><tr><td rowspan=2 colspan=1>System</td><td rowspan=1 colspan=2>TaskResponse</td><td rowspan=1 colspan=2>&quot;I&#x27;ll bring the large top cover from the storage room to the assembly area. I canalso get the corresponding PCB. Where should I retrieve the PCB from?&quot;.</td></tr><tr><td rowspan=1 colspan=2>Small TalkResponse</td><td rowspan=1 colspan=2>&quot;Would you like me to check if there are any assembly instructions for thesecomponents?&quot;,</td></tr><tr><td rowspan=6 colspan=1>3</td><td rowspan=3 colspan=1>Turn 1</td><td rowspan=1 colspan=1>User</td><td rowspan=1 colspan=2>&quot;Bring the m</td><td rowspan=1 colspan=2>ultimeter from the calibration bench.&quot;.</td></tr><tr><td rowspan=1 colspan=1>System</td><td rowspan=1 colspan=2>TaskResponse</td><td rowspan=1 colspan=2>&quot;I apologize, but I don&#x27;t have information on how to grasp a multimeter. Is thereanother object you need that I might be able to help with? I can assist with bottomcovers, PCBs, top covers, or fuses.&quot;,</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Small TalkResponse</td><td rowspan=1 colspan=2>&quot;Perhaps one of these items would be helpful for your current task?&quot;,</td></tr><tr><td rowspan=3 colspan=1>Turn 2</td><td rowspan=1 colspan=1>User</td><td rowspan=1 colspan=2>&quot;I see. In th</td><td rowspan=1 colspan=2>at case, can you bring me a small fuse from the electrical cabinet?&quot;.</td></tr><tr><td rowspan=2 colspan=1>System</td><td rowspan=1 colspan=2>TaskResponse</td><td rowspan=1 colspan=2>&quot;Certainly, I can bring you a small fuse from the electrical cabinet. Where would youlike me to place it?&quot;,</td></tr><tr><td rowspan=1 colspan=2>Small TalkResponse</td><td rowspan=1 colspan=2>&quot;Do you need any safety equipment to work with the fuse?&quot;.</td></tr></table>

Fig. D5: Examples of randomly selected LLM-generated dialogues from Position and Relocation domainsains