# Beyond Fine-Tuning: Evaluating Prompt Engineering and Prompt Tuning for Roman Urdu Hate Speech Detection

Toneema Zubair <sup>1</sup>, Muhammad Junaid Asif <sup>2,3\*</sup>, Faisal Kamiran <sup>1</sup>, Hafiz Hassan Saeed <sup>1</sup>, Rana Fayyaz Ahmad <sup>2</sup>

<sup>1</sup> Department of Computer Sciences, Information Technology University (ITU), Lahore 54000, Pakistan

<sup>2</sup> Natural Language Processing (NLP) Lab, Artificial Intelligence Technology Centre (AITeC), Islamabad 44000, Pakistan

<sup>3</sup> Faculty of IT and Computer Sciences, University of Central Punjab (UCP), Lahore (54000), Pakistan

\*junaid.asif@ncp.edu.pk

Abstract— Low Resource Languages (LRLs) pose significant challenges for hate speech detection because of a scarcity of readily available annotated data, language variation, nonstandard spelling, and context-dependent ambiguity. Beyond the problem of informal conventions of writing in Roman Urdu, there is also the issue of code-mixing. After working on the previous project about zero-shot learning and parameterefficient fine-tuning (PEFT), this research systematically compares prompt engineering and prompt tuning for Roman Urdu hate speech detection. To study the effect of in-context examples, we will compare the zero-shot and few-shot prompting approaches. To understand the advantage of structured prompting and learnable prompt representations, we will conduct studies of manual vs. mixed prompt-based adaptation strategies. Experimentations are carried out on the PURUT dataset (\~72,771 social media comments) with GPT-3.5, LLaMA-3, Mistral-7B, and multilingual BERT in several settings. Experimental results show that few-shot prompting always outperforms zero-shot prompting, with the Mistral-7B model scoring 0.67 F1 compared to 0.56 of the zero-shot models. Based on this observation, prompt-based adaptation is proposed to introduce additional improvements, with the proposed mixed prompt configuration able to yield a F1-score up to 0.87 which further surpasses the performance of regular manual prompts. Despite the larger size of the model, its performance trade-off between inference-based prompting, prompt optimization, and parameter-efficient model adaptation is compared to the previously reported PEFT-based F1-score, which is \~0.93. Overall, the study demonstrates the potential of carefully designed and optimized prompting strategies for improving hate speech detection in linguistically complex, low-resource environments while providing practical insights into selecting suitable adaptation approaches.

Keywords— Roman Urdu Hate Speech Detection (RU-HSD), Low Resource Languages (LRLs), Prompt Engineering, Zero-Shot Learning, Few-Shot Learning, Prompt Tuning, Prefix Prompting, Cloze Prompting

## I. INTRODUCTION

The rapid expansion of social media platforms has transformed how people communicate, share opinions, and engage in public discourse. While platforms such as Twitter (now X), Facebook, and YouTube have increased opportunities for information sharing and public interaction, they have also facilitated the dissemination of harmful content, including hate speech, offensive language, and abusive expressions. Hate speech poses significant psychological and societal risks by contributing to discrimination, polarization, and hostility toward individuals and communities. Consequently, automatic hate speech detection has emerged as an important research area within natural language processing (NLP), supporting the development of safer and more inclusive online environments.

Over the past decade, hate speech detection has advanced significantly through machine learning, deep learning, and transformer-based models such as BERT, which leverage contextual embeddings and self-attention mechanisms. However, most studies have focused on high-resource languages like English, where large annotated datasets are available [1]. In contrast, lowresource languages remain underexplored, and models trained on high-resource languages often fail to generalize effectively.

Roman Urdu presents particular challenges for hate speech detection because of its non-standard spelling, phonetic variations, informal expressions, and frequent code-mixing. These characteristics complicate contextual understanding and classification. Furthermore, limited linguistic resources and imbalanced datasets affect model performance. To address these challenges, Saeed et al. [2] introduced the PURUTT dataset, comprising 72,771 toxic and non-toxic social media comments, providing a benchmark for Roman Urdu toxicity classification.

Recent advancements in large language models (LLMs), including GPT, LLaMA, and Mistral, have introduced prompt-based learning as an alternative to conventional model fine-tuning. Prompt-based learning can be broadly divided into prompt engineering and prompt tuning. Prompt engineering enables models to perform downstream tasks through natural language instructions without updating model parameters. In particular, zero-shot prompting relies solely on task instructions, whereas few-shot prompting incorporates a limited number of labeled examples to improve contextual understanding and classification performance [3]. These approaches are particularly relevant to low-resource NLP applications where extensive model training may be impractical.

Beyond prompt engineering, prompt tuning provides an alternative mechanism for adapting pre-trained models through task-specific prompt representations. Lester et al.[4] demonstrated the effectiveness of learnable soft prompts, while Li and Liang [5] introduced prefix tuning, which optimizes continuous prompt representations while keeping the underlying model parameters frozen. Additionally, cloze-based prompting reformulates classification as a masked-token prediction task, allowing models to associate contextual information with predefined class labels [6]. These approaches offer different mechanisms for improving task adaptation while reducing the need for full-model fine-tuning.

Despite their advantages, prompt-based approaches remain sensitive to prompt structure, instruction wording, demonstration selection, and linguistic variability. These limitations are particularly relevant to Roman Urdu, where inconsistent spelling and contextual ambiguity can affect classification reliability. Although parameter-efficient fine-tuning (PEFT), particularly Low-Rank Adaptation (LoRA), provides effective task-specific adaptation [8], the comparative effectiveness of prompt engineering and alternative prompt tuning strategies for Roman Urdu hate speech detection requires further investigation.

Building upon our previously published work comparing zero-shot prompting and PEFT-based adaptation [9], this study focuses on a systematic evaluation of zero-shot versus few-shot prompt engineering and prefix versus cloze prompt formulations for Roman Urdu hate speech detection. Experiments are conducted using the PURUTT dataset and multiple pretrained language models to investigate the influence of prompt design and adaptation strategies on classification performance. The study evaluates accuracy, precision, recall, and F1-score to identify effective prompting approaches for linguistically complex and low-resource environments. To guide this study, the following research questions are formulated:

RQ1: How does few-shot prompting perform compared to zero-shot prompting in Roman Urdu hate speech detection?

 RQ2: How do prefix and cloze prompting strategies affect classification performance?

 RQ3: Does mixed prompt tuning outperform conventional manual prompting?

 RQ4: How do prompt-based approaches compare with previously reported PEFT results?

 RQ5: What are the performance and computational efficiency trade-offs among these approaches?

This paper makes several key contributions to the field of low-resource NLP:

 A comprehensive comparison of prompt engineering, prompt tuning, and parameter-efficient fine-tuning for hate speech detection in Roman Urdu.

Introduction and evaluation of a mixed prompt tuning approach that combines manual templates with learnable soft tokens.

 Empirical analysis of zero-shot and few-shot prompting strategies in low-resource settings.

Insights into the strengths and limitations of promptbased learning compared to PEFT.

 Practical recommendations for selecting appropriate adaptation strategies in resource-constrained environments.

The remainder of this paper is organized as follows. Section II reviews related work, while Section III presents the dataset, methodology, and experimental setup. Section IV reports the experimental results, followed by a discussion of key findings in Section V. Finally, Section VI concludes the study and outlines future research directions.

## II. RELATED WORK

This section presents a comprehensive overview of these approaches, highlighting their strengths, limitations, and applicability in such challenging settings.

## A. Traditional Machine and Deep Learning Approaches

Early research in hate speech detection primarily employed traditional machine learning algorithms, including Support Vector Machines (SVM), Logistic Regression, Naïve Bayes, Random Forest, and K-Nearest Neighbors (K-NN). These methods typically relied on handcrafted textual features, such as n-grams, TF-IDF, and lexical characteristics, to identify harmful content [7], [8]. Their simplicity, computational efficiency, and effectiveness in structured classification tasks made them widely adopted. In particular, SVM and Logistic Regression demonstrated competitive performance when combined with appropriate feature engineering, while ensemble methods improved classification robustness. Despite these advantages, traditional approaches have limited capabilities in capturing semantic relationships and contextual information. Their dependence on handcrafted features reduces adaptability across languages and domains. These limitations are particularly significant for Roman Urdu, where inconsistent spelling, phonetic variations, code-mixing, and informal expressions complicate text classification[2]. Consequently, traditional classifiers often struggle to identify implicit hate speech, sarcasm, and context-dependent expressions. To overcome these limitations, researchers introduced deep learning architectures, including Convolutional Neural Networks (CNNs), Long Short-Term Memory (LSTM), and Bidirectional LSTM (Bi-LSTM). Unlike traditional methods, these models automatically learn hierarchical textual representations, reducing the need for manual feature engineering. CNNs effectively capture local linguistic patterns, while LSTM-based architectures model sequential dependencies and contextual relationships[9], [10].

Furthermore, attention mechanisms improve classification by emphasizing relevant words and expressions. Deep learning methods have demonstrated improved semantic understanding and greater adaptability to noisy textual data. However, their effectiveness often depends on the availability of sufficiently large annotated datasets. For low-resource languages such as Roman Urdu, limited labelled data can lead to overfitting and reduced generalization. Additionally, model training may require considerable computational resources, restricting practical deployment in resource-constrained environments [10]. These challenges have encouraged the adoption of transformer-based architectures and prompt-based learning strategies that leverage pre-trained language representations for more effective low-resource hate speech detection.

## B. Transformer-Based Models and LLMs for HSD:

Transformer-based architectures have significantly advanced natural language processing (NLP) through selfattention mechanisms that capture long-range dependencies and contextual relationships. Among these architectures, BERT has demonstrated strong performance in text classification by generating bidirectional contextual representations, enabling more effective interpretation of complex linguistic patterns and ambiguous expressions[11]. These capabilities have made transformer-based models particularly useful for hate speech detection, where understanding contextual meaning is essential. Despite their effectiveness, conventional transformer-based models often require task-specific fine-tuning, which can involve substantial computational resources and memory. Furthermore, models predominantly pre-trained on highresource languages may struggle with domain-specific expressions and linguistic variations in low-resource languages such as Roman Urdu. These challenges highlight the importance of efficient adaptation strategies that reduce training requirements while maintaining classification performance [12]

Recent advancements in large language models (LLMs), including GPT, LLaMA, Mistral, and Gemma, have introduced new opportunities for hate speech detection. These models support zero-shot and few-shot learning, allowing classification through natural language instructions without necessarily updating model parameters [3]. Prompt engineering enables task adaptation through carefully designed instructions, while in-context learning incorporates labeled examples within prompts to improve classification. Such approaches are particularly valuable in low-resource settings where extensive annotated datasets and computational infrastructure may be unavailable. Recent studies have explored both prompt-based learning and efficient fine-tuning for text classification. Xu et al. [13] investigated QLoRAbased adaptation of LLaMA2-7B for news topic classification, reporting improvements over selected transformer baselines. Nadeau et al. [14] evaluated LLMs, including LLaMA2, Mistral, Gemma, and GPT, across safety-related dimensions such as toxicity, bias, factuality, and hallucination, highlighting differences in model behavior. These findings demonstrate the importance of selecting appropriate models and adaptation strategies for reliable classification.

Prompt-based techniques have also shown promising results in hate speech detection. Zhang et al. [15] introduced Domain-Enhanced Prompt Learning (DePL) for implicit hate speech detection in Chinese, addressing data scarcity through domain-aware prompt learning. Yamagishi [16] explored simplified prompts for zero-shot multimodal hate speech detection, suggesting that concise prompts can be effective without additional complexity. Similarly, Govers et al. [17] proposed Prompt-GAN, employing adversarial prompt tuning to generate synthetic extremist content and reporting improvements in classification performance. Han and Tang [18] investigated GPT-3-based in-context learning for hate speech detection, emphasizing the influence of task instructions and demonstration examples on few-shot classification.

Although these studies demonstrate the potential of LLMs and prompt-based learning, their effectiveness remains sensitive to prompt structure, demonstration selection, and linguistic characteristics. These limitations are particularly relevant to Roman Urdu, where inconsistent spelling, codemixing, informal expressions, and contextual ambiguity complicate classification. Consequently, systematic evaluation of zero-shot and few-shot prompting, together with alternative prompt tuning strategies, is necessary to identify effective approaches for low-resource hate speech detection.

Despite significant advancements in hate speech detection, several research gaps remain, particularly for low-resource languages such as Roman Urdu. Existing studies predominantly focus on high-resource languages, with limited attention to Roman Urdu's linguistic complexities, including inconsistent spelling, code-mixing, and contextual ambiguity.

Furthermore, comprehensive evaluations of zero-shot and few-shot prompting, manual and mixed prompt tuning, and their comparative effectiveness remain limited. Although parameter-efficient fine-tuning (PEFT) has demonstrated promising results, further investigation is needed to understand how prompt-based approaches compare with previously reported PEFT benchmarks. Additionally, the relationship between classification performance, computational efficiency, and adaptation complexity remains insufficiently explored. This study addresses these gaps by systematically evaluating prompt engineering and prompt tuning strategies for Roman Urdu hate speech detection, aiming to identify effective and resource-efficient approaches for low-resource NLP applications.

## III. MATERIAL AND METHODS

## 3.1. Dataset Details

The PURUTT (Parallel Urdu and Roman Urdu Toxicity) dataset [2] is a benchmark for hate speech detection in lowresource languages, comprising 72,771 social media comments, including 13,097 toxic and 59,674 non-toxic samples. It reflects real-world linguistic challenges such as inconsistent spelling, phonetic variations, code-mixing, informal grammar, and semantic ambiguity, making it suitable for evaluating NLP models. For experimentation, the dataset is preprocessed and divided into 60% training, 20% validation, and 20% testing subsets. To address class imbalance, class-weighted cost-sensitive learning is applied during training. Its linguistic diversity and real-world relevance make PURUTT an effective benchmark for evaluating large language models, prompt-based learning, and parameter-efficient fine-tuning techniques such as LoRA. Summary of PURUTT dataset is described in Table 1.

TABLE 1. SUMMARY OF PURUTT DATASET
<table><tr><td rowspan=1 colspan=1>Attribute</td><td rowspan=1 colspan=1>Description</td></tr><tr><td rowspan=1 colspan=1>Total Samples</td><td rowspan=1 colspan=1>72,771 Nos.</td></tr><tr><td rowspan=1 colspan=1>Toxic Samples</td><td rowspan=1 colspan=1>13,097 Nos.</td></tr><tr><td rowspan=1 colspan=1>Non-Toxic Samples</td><td rowspan=1 colspan=1>59,674 Nos.</td></tr><tr><td rowspan=1 colspan=1>Task Type</td><td rowspan=1 colspan=1>Binary Classification (Toxic / Non-Toxic)</td></tr><tr><td rowspan=1 colspan=1>Language</td><td rowspan=1 colspan=1>Roman Urdu (Latin script)</td></tr><tr><td rowspan=1 colspan=1>Data Source</td><td rowspan=1 colspan=1>Social media comments</td></tr><tr><td rowspan=1 colspan=1>Class Distribution</td><td rowspan=1 colspan=1>Imbalanced (≈18% toxic, 82% non-toxic)</td></tr><tr><td rowspan=1 colspan=1>Key Challenges</td><td rowspan=1 colspan=1>Non-standard spelling, phonetic variation,semantic ambiguity</td></tr><tr><td rowspan=1 colspan=1>Preprocessing</td><td rowspan=1 colspan=1>Tokenization, normalization, padding/truncation</td></tr><tr><td rowspan=1 colspan=1>Data Split</td><td rowspan=1 colspan=1>60% Train / 20% Validation / 20% Test</td></tr><tr><td rowspan=1 colspan=1>Handling Imbalance</td><td rowspan=1 colspan=1>Class weighting (cost-sensitive learning)</td></tr></table>

## 3.2. Proposed Methodology

This study presents a unified framework for evaluating prompt engineering and prompt tuning for Roman Urdu hate speech detection. The framework addresses key challenges of low-resource NLP, including inconsistent spelling, codemixing, semantic ambiguity, and class imbalance. Unlike conventional fine-tuning, the proposed approaches leverage pre-trained language models while minimizing the need for extensive parameter updates. Fig. 1 illustrates the proposed architecture. The process begins with the PURUTT dataset, followed by text normalization, tokenization, padding, and truncation. The framework investigates three complementary adaptation strategies. First, Prompt engineering evaluates zero-shot and few-shot classification using GPT-3.5 through Azure OpenAI, employing task-specific instructions and labeled examples without updating model parameters.

Second, Prompt tuning utilizes multilingual BERT (uncased) through the OpenPrompt framework to evaluate Prefix, Cloze, and Mixed Prompting strategies under few-shot configurations of (K = 32,64,128). The Mixed Prompting approach combines fixed textual templates with learnable soft embeddings to improve task-specific representations. Previously published results [19] obtained using conventional inference and PEFT-LoRA are considered as reference baselines for contextual comparison rather than as experimental contributions of the present study. The proposed approaches are evaluated using accuracy, precision, recall, and F1 score to assess their effectiveness for Roman Urdu hate speech detection.

![](images/d000aafb821f25ed68603becf663d70b906da5e5e059d5631bd4c3408bf190c1.jpg)  
Fig. 1. Proposed framework for Roman Urdu Hate Speech Detection (RU-HSD) using prompt engineering and prompt tuning strategies.

## 3.2.1. Prompt Engineering

Prompt engineering is employed to evaluate the ability of large language models to classify Roman Urdu comments without updating model parameters. The classification task is formulated through structured chat messages consisting of system instructions, user inputs, and model-generated responses. Two strategies are investigated: zero-shot and fewshot prompting.

## A. Zero Shot Prompting:

In zero-shot prompting, the model receives a task-specific system instruction and an unlabeled Roman Urdu comment. No labeled examples are provided. The model is instructed to classify the comment as toxic (1) or non-toxic (0), returning only the predicted label. The following system prompt is employed:

The zero-shot approach provides a baseline for evaluating the inherent classification capabilities of pre-trained LLMs in a linguistically complex, low-resource environment.

## B. Few Shot Prompting:

Few-shot prompting extends the zero-shot approach by introducing labeled examples before the target comment. The thesis uses five toxic and five non-toxic examples in each prompt, represented as user–assistant message pairs. These examples provide contextual guidance without modifying the model parameters. The same system instruction is retained to ensure consistent classification criteria. The research evaluates GPT-3.5 through Microsoft Azure OpenAI, processing comments in batches of 100. It reports 728 batches for each prompting configuration, with batch-level evaluation metrics averaged to obtain the final results.

<table><tr><td>Algorithm 1: Prompt Engineering using OpenAI for RU-HSD</td></tr><tr><td>Input: D: Roman Urdu dataset with comments and labels Input: M: OpenAI GPT-3.5 model Output: Predicted labels and evaluation metrics</td></tr><tr><td>1: Define system message with strict instructions to output only 0 or 1 2: Split dataset D into batches of 100 3: for each batch B in D do</td></tr><tr><td>4: Initialize prompt with system message</td></tr><tr><td>5: if few-shot then</td></tr><tr><td>6: Select 5 examples per class from B and add to prompt</td></tr><tr><td>7: end if 8: for each comment c in B do</td></tr><tr><td>9: Add user message with c</td></tr><tr><td>10: Model generates predicted label (0 or 1)</td></tr><tr><td>11: Save predicted label</td></tr><tr><td></td></tr><tr><td>12: end for</td></tr><tr><td>13: Compute batch-level metrics: accuracy, precision, recall, F1-</td></tr><tr><td>score</td></tr><tr><td></td></tr><tr><td>14: end for</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td>15: Compute final overall metrics as the average of all batch metrics</td></tr></table>

## 3.2.2. Prompt Tuning

Prompt tuning is investigated using the OpenPrompt framework and the bert-base-multilingual-uncased model. Roman Urdu hate speech detection is reformulated as a masked language modeling task, where a [MASK] token is used to predict label-related words. A manual verbalizer maps these predictions to toxic and non-toxic classes. The experiments consider manual prefix and cloze templates alongside a mixed prompt approach incorporating learnable soft tokens.

## A. Manual Prompt:

Manual prompting employs fixed, human-designed templates to formulate hate speech detection as a maskedtoken prediction task. Two template structures are evaluated:

<table><tr><td>Prefix Prompt: Cloze Prompt:</td><td>Ye hai [MASK]! Ye [MASK] hai.</td></tr></table>

In the prefix formulation, the masked token appears at the end of the template, whereas the cloze formulation places it within the sentence. The input comment provides contextual information for predicting the masked token, which is subsequently mapped to a classification label through the verbalizer. Both formulations use fixed textual templates without introducing learnable soft prompt embeddings. Their comparison enables an assessment of the influence of template structure and masked-token placement on classification performance.

![](images/a0a98a9619ad8431d3800c5bf50ff8ef051a5e3b06c5f7468ecdb40475f88ebe.jpg)  
Fig. 2. Architecture of Prefix-Based and Cloze-Based Manual Prompting

## B. Mixed Prompt:

Mixed prompt tuning combines manually constructed prompt elements with trainable soft token embeddings. Unlike fixed manual prompts, the soft embeddings are optimized during training while the underlying language model remains frozen. This allows task-specific adaptation with a limited number of trainable parameters. The mixed prompt format is as follows:

{"placeholder":"text\_a","soft":"Yeh","soft":"comment hai","mask"}

![](images/a0763d70de6c7bd6382942da180f08efa401d824b292a2cc27f805a5832efd7c.jpg)  
Fig. 3. Mixed Prompt Tuning Architecture with Learnable Soft Tokens

This hybrid approach combines human-defined prompt structure with data-driven adaptation. It aims to improve the representation of linguistic variations and contextual information while reducing the number of trainable parameters compared with full-model fine-tuning.

## C. Manual Verbalizer

A manual verbalizer converts masked-token predictions into binary class labels. The thesis defines label-word groups associated with non-toxic (0) and toxic (1) comments. The verbalizer enables the model to associate predicted words with the corresponding classification labels.

TABLE 2. MANUAL VERBALIZER FOR RU-HSD
<table><tr><td rowspan=1 colspan=1>Class</td><td rowspan=1 colspan=1>Label words</td></tr><tr><td rowspan=1 colspan=1>Non-Toxic (0)</td><td rowspan=1 colspan=1>muhabbat, pyaar, acha, dua, wah</td></tr><tr><td rowspan=1 colspan=1>Toxic (1)</td><td rowspan=1 colspan=1>khilaf, nafrat, ghussa, lanat, ganda, beh***d, gaali, bura</td></tr></table>

Algorithm 1. Prompt Tuning Using the OpenPrompt Framework Input: D: Roman Urdu dataset containing comments and labels

Input: PLM: Pretrained language model (bert-base-multilingual-uncased)   
Input: T: Prompt template style (prefix, cloze, or mixed)   
Output: Trained prompt-based classifier   
1: Split D into D <sub>train+dev</sub> (80%) and D<sub>test</sub> (20%)   
2: Define a verbalizer mapping label words to classes 0 and 1   
3: for sample size n in {32, 64, 128} do   
4: for repetition r = 1 to 2 do   
5: Randomly sample n examples from $\mathrm { D } _ { \mathrm { t r a i n ^ { + } d e v } }$ as $\mathrm { D } _ { \mathrm { t r a i n } } ^ { \mathrm { ~ \tiny ~ ( r ) ~ } }$   
6: Use the remaining examples as $\mathrm { D _ { d e v } } ^ { ( \mathrm { r } ) }$   
7: Initialize the OpenPrompt pipeline with PLM   
8: if T = prefix then   
9: Template $ \mathrm { ^ { 6 6 } Y e }$ hai [MASK]!”   
10: else if T = cloze then   
11: Template ← “Ye [MASK] hai.”   
12: else if T = mixed then   
Template ← {"placeholder":"text\_a", "soft":"Yeh",   
13: "soft":"comment hai", "mask"}   
14: end if   
15: Create PromptDataLoader for $\mathrm { D } _ { \mathrm { t r a i n } } ^ { \mathrm { ~ \tiny ~ ( r ) ~ } }$   
Initialize loss and optimizer (for soft tokens if T =   
16:   
mixed)   
17: for epoch = 1 to 3 do   
18: for each batch in $\mathbf { D _ { t r a i n } } ^ { ( \mathbf { r } ) }$ do   
19: Compute loss and update parameters   
20: end for   
21: end for   
22: Evaluate on $\mathrm { { D } _ { d e v } } ^ { \left( \mathrm { r } \right) }$ and store results   
23: end for   
24: Select the best model among both repetitions   
25: Evaluate the selected model on $\mathrm { D } _ { \mathrm { t e s t } }$   
26: end for

## 3.3. Implementation Details

The proposed framework is implemented in Python using PyTorch, Hugging Face Transformers, and OpenPrompt. Prompt engineering experiments are conducted using GPT-3.5 through Azure OpenAI services, while prompt tuning employs multilingual BERT (uncased) with fixed textual templates and learnable soft prompt representations. Modelspecific tokenization, padding, and truncation are applied to ensure appropriate input processing. The local prompt-tuning experiments are conducted using an NVIDIA A100 GPU. For prompt tuning, AdamW optimization is employed with a learning rate of 1 x 10<sup>-4</sup>, batch size of 32, 64, and 128, and 3 training epochs. The maximum sequence length is 512 tokens, with dropout set to 0.05. Validation-based model selection is used before final evaluation on the test set. The main experimental configurations are summarized below.

TABLE 3. EXPERIMENTAL CONFIGURATION
<table><tr><td rowspan=1 colspan=1>Component</td><td rowspan=1 colspan=1>Specification</td></tr><tr><td rowspan=1 colspan=1>Framework</td><td rowspan=1 colspan=1>PyTorch,</td></tr><tr><td rowspan=1 colspan=1>Prompt Engineering Model</td><td rowspan=1 colspan=1>GPT 3.5 (Azure Open AI)</td></tr><tr><td rowspan=1 colspan=1>Prompt Tuning Model</td><td rowspan=1 colspan=1>BERT-base-multilingual-uncased</td></tr><tr><td rowspan=1 colspan=1>GPU</td><td rowspan=1 colspan=1>NVIDIA A100</td></tr><tr><td rowspan=1 colspan=1>Sequence Length</td><td rowspan=1 colspan=1>512 tokens</td></tr><tr><td rowspan=1 colspan=1>Batch Size</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=1 colspan=1>Optimizer</td><td rowspan=1 colspan=1>AdamW</td></tr><tr><td rowspan=1 colspan=1>Learning Rate</td><td rowspan=1 colspan=1>1x 10{4</td></tr><tr><td rowspan=1 colspan=1>Epochs</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>Dropout</td><td rowspan=1 colspan=1>0.05</td></tr><tr><td rowspan=1 colspan=1>Prompt Engineering</td><td rowspan=1 colspan=1>Zero-shot, Few-shot</td></tr><tr><td rowspan=1 colspan=1>Prompt Tuning</td><td rowspan=1 colspan=1>Manual, Mixed, Cloze</td></tr><tr><td rowspan=1 colspan=1>Evaluation</td><td rowspan=1 colspan=1>Accuracy, Precision, Recall, F1-score</td></tr></table>

## IV. RESULTS AND DISCUSSION

This section presents the experimental results of prompt engineering and prompt tuning for Roman Urdu hate speech detection using the PURUTT dataset. The evaluation examines the effectiveness of zero-shot and few-shot prompting, manual prompt formulations, and mixed prompt tuning across different pre-trained language models. The primary objective is to investigate how these adaptation strategies influence classification performance in a lowresource language characterized by spelling variations, codemixing, informal expressions, and class imbalance. Model performance is evaluated using accuracy, precision, recall, and F1-score. Particular emphasis is placed on F1-score because the dataset contains substantially fewer toxic samples than non-toxic samples. The experimental analysis is organized into three parts: performance evaluation of prompt engineering, comparison of manual and mixed prompt tuning, and model-wise comparative analysis. Previously reported parameter-efficient fine-tuning (PEFT) results are considered separately to provide additional context regarding the effectiveness of prompt-based adaptation.

## 4.1. Performance Evaluation of Prompt Engineering

## 4.1.1. Experimental Approach

This experiment evaluates the effectiveness of zero-shot and few-shot prompt engineering for Roman Urdu hate speech detection using the OpenAI GPT-3.5 model accessed through Azure OpenAI services. The objective was to investigate the model's classification performance without task-specific fine-tuning under two prompting configurations:

Zero-Shot Prompting: The model was provided with task-specific instructions without labeled examples.

Few-Shot Prompting: A small set of representative labeled examples was included in the prompt to guide the model's classification decisions.

For both configurations, the Roman Urdu comments were processed in batches of 100. Performance was evaluated using accuracy, precision, recall, and F1 score. The reported results include averaged evaluation metrics and the maximum F1 score observed across the evaluated batches.

## 4.1.2. Experimental Results:

Table 4 presents the comparative performance of zeroshot and few-shot prompting. The results demonstrate that few-shot prompting consistently outperformed zero-shot prompting across all four-evaluation metrics.

TABLE 4. PERFORMANCE COMPARISON OF ZERO-SHOT AND FEW-SHOT PROMPT ENGINEERING
<table><tr><td rowspan=1 colspan=1>PromptType</td><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>Recall</td><td rowspan=1 colspan=1>F1-Score</td><td rowspan=1 colspan=1>Max F1-Score</td></tr><tr><td rowspan=1 colspan=1>Few-Shot</td><td rowspan=1 colspan=1>0.9100</td><td rowspan=1 colspan=1>0.8278</td><td rowspan=1 colspan=1>0.8989</td><td rowspan=1 colspan=1>0.8565</td><td rowspan=1 colspan=1>0.9494</td></tr><tr><td rowspan=1 colspan=1>Zero-Shot</td><td rowspan=1 colspan=1>0.7229</td><td rowspan=1 colspan=1>0.6799</td><td rowspan=1 colspan=1>0.8129</td><td rowspan=1 colspan=1>0.6698</td><td rowspan=1 colspan=1>0.8250</td></tr></table>

Few-shot prompting achieved an accuracy of 0.9100 and F1 score of 0.8565, compared with 0.7229 and 0.6698 for zero-shot prompting. This represents absolute improvements of 18.71 and 18.67 percentage points in accuracy and F1 score, respectively. Similarly, precision increased from 0.6799 to 0.8278, while recall improved from 0.8129 to 0.8989. The maximum observed F1 score also increased from 0.8250 for zero-shot prompting to 0.9494 for few-shot prompting. Figure 4.11 illustrates these performance differences, highlighting the consistent advantage of providing labeled examples during inference.

![](images/fae8bfc5d2833406ef002f251c4176e66dca5e032c8ed930f4d2b48c260331ac.jpg)  
Fig. 4. Comparative performance of GPT-3.5 using zero-shot and few-shot prompting across accuracy, precision, recall, and F1 score.

## 4.2. Performance Evaluation of Prompt Tuning

## 4.2.1. Experimental Approach

This experiment evaluates multilingual BERT (uncased) for Roman Urdu hate speech detection using three adaptation strategies: direct inference without fine-tuning, Parameter-Efficient Fine-Tuning (PEFT) using Low-Rank Adaptation (LoRA), and prompt tuning. Three prompting strategies were evaluated: Prefix Prompting, Cloze Prompting, and Mixed Prompting. Each strategy was assessed using three values of the prompt configuration parameter, (K = 32, 64, 128). The performance was measured using precision, recall, F1 score, and classification accuracy.

## 4.2.2. Dataset Preparation and Experimental Setup

The PURUTT dataset was divided into three mutually exclusive subsets: training $( D _ { t r a i n } )$ , development $( D _ { d e \nu } )$ , and testing $( \pmb { D } _ { t e s t } )$ , satisfying:

$$
D = D _ { t r a i n } \ U D _ { d e v } \ U D _ { t e s t }\tag{1}
$$

$$
D _ { t r a i n } \cap D _ { d e \nu } = D _ { t r a i n } \cap D _ { t e s t } = D _ { d e \nu } \cap D _ { t e s t } = \emptyset\tag{2}
$$

For each few-shot configuration, K training examples and K development examples were randomly selected, where:

$$
K \in \{ 3 2 , 6 4 , 1 2 8 \}\tag{3}
$$

$$
| D t r a i n | = K , \quad | D d e \nu | = K , \quad | D t e s t | = | D | - 2 K\tag{4}
$$

Each configuration was evaluated over two experimental rounds, with two epochs per round:

$$
R = 2 , \quad E = 2\tag{5}
$$

Training samples were selected without overlap between experimental rounds:

$$
D _ { t r a i n } ( ^ { i } \cap D _ { t r a i n } ^ { } ( \beta = \emptyset , \quad i \ne j\tag{6}
$$

The same preprocessing and evaluation procedures were applied across the experiments to support consistent performance comparisons.

## 4.2.3. Evaluation of Prefix Prompting

Table 5 and Fig. 5presents the performance of Prefix Prompting across different values of (K = 32, 64, and 128). At (K=32), the model achieved an F1 score of 0.6095 and accuracy of 0.7970. Increasing (K) to 64 improved the F1 score to 0.6312, although accuracy decreased to 0.7584. The best performance was obtained at (K=128), achieving a precision of 0.6747, recall of 0.6894, F1 score of 0.6813, and accuracy of 0.8048. These results indicate that increasing (K) generally improved the F1 score, although accuracy did not follow a consistent trend.

Cloze Prompt: Evaluation Metrics by K  
TABLE 5. EVALUATION OF PREFIX PROMPTING BY DIFFERENT K-VALUES
<table><tr><td rowspan=1 colspan=1>K-Value</td><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>Recall</td><td rowspan=1 colspan=1>F1-Score</td><td rowspan=1 colspan=1>Accuracy</td></tr><tr><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>0.6311</td><td rowspan=1 colspan=1>0.5989</td><td rowspan=1 colspan=1>0.6095</td><td rowspan=1 colspan=1>0.7970</td></tr><tr><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>0.6222</td><td rowspan=1 colspan=1>0.6485</td><td rowspan=1 colspan=1>0.6312</td><td rowspan=1 colspan=1>0.7584</td></tr><tr><td rowspan=1 colspan=1>128</td><td rowspan=1 colspan=1>0.6747</td><td rowspan=1 colspan=1>0.6894</td><td rowspan=1 colspan=1>0.6813</td><td rowspan=1 colspan=1>0.8048</td></tr></table>

![](images/46ef3d730272a7e19f99dfa8331ea0fd2746c72bb1864eec32779a115b7469ec.jpg)  
Fig. 5. Prefix Prompt: Evaluation Metrics by K

## 4.2.4. Evaluation of Cloze Prompting

Table 6 and Fig. 6 summarizes the results of Cloze Prompting. At (K=32), the model achieved its highest F1 score of 0.6532 and accuracy of 0.6984. Increasing (K) to 64 improved recall to 0.7234 but reduced the F1 score to 0.5886 and accuracy to 0.6299. At (K=128), the F1 score recovered to 0.6167, while accuracy reached 0.6842. These findings suggest that increasing (K) does not consistently improve Cloze Prompting performance, with (K=32) providing the best balance between F1 score and accuracy.

TABLE 6. EVALUATION OF CLOZE PROMPTING BY DIFFERENT K-VALUES
<table><tr><td rowspan=1 colspan=1>K-Value</td><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>Recall</td><td rowspan=1 colspan=1>F1-Score</td><td rowspan=1 colspan=1>Accuracy</td></tr><tr><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>0.6097</td><td rowspan=1 colspan=1>0.6833</td><td rowspan=1 colspan=1>0.6532</td><td rowspan=1 colspan=1>0.6984</td></tr><tr><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>0.6342</td><td rowspan=1 colspan=1>0.7234</td><td rowspan=1 colspan=1>0.5886</td><td rowspan=1 colspan=1>0.6299</td></tr><tr><td rowspan=1 colspan=1>128</td><td rowspan=1 colspan=1>0.6256</td><td rowspan=1 colspan=1>0.7049</td><td rowspan=1 colspan=1>0.6167</td><td rowspan=1 colspan=1>0.6842</td></tr></table>

![](images/e8a7bf1efe5c8936698ceb295b65a6bb27000c28b1783ac6a47ea448ed9fe822.jpg)  
Fig. 6. Cloze Prompt: Evaluation Metrics by K

## 4.2.5. Evaluation of Mixed Prompting

Table 8 and Fig. 7 presents the results of Mixed Prompting, which combines fixed textual prompts with learnable prompt representations. At (K=32), the model achieved an F1 score of 0.6008 and accuracy of 0.6976. Increasing (K) to 64 improved the F1 score to 0.6531 and accuracy to 0.7580. At (K=128), the model achieved its highest F1 score of 0.6852 and recall of 0.7810, although accuracy decreased slightly to 0.7498. Overall, Mixed Prompting demonstrated improved F1 performance with increasing (K), achieving the highest recall among all evaluated prompting configurations.

TABLE 7. EVALUATION OF MIXED PROMPTING BY DIFFERENT K-VALUES
<table><tr><td rowspan=1 colspan=1>K-Value</td><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>Recall</td><td rowspan=1 colspan=1>F1-Score</td><td rowspan=1 colspan=1>Accuracy</td></tr><tr><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>0.5992</td><td rowspan=1 colspan=1>0.6484</td><td rowspan=1 colspan=1>0.6008</td><td rowspan=1 colspan=1>0.6976</td></tr><tr><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>0.6405</td><td rowspan=1 colspan=1>0.6885</td><td rowspan=1 colspan=1>0.6531</td><td rowspan=1 colspan=1>0.7580</td></tr><tr><td rowspan=1 colspan=1>128</td><td rowspan=1 colspan=1>0.6774</td><td rowspan=1 colspan=1>0.7810</td><td rowspan=1 colspan=1>0.6852</td><td rowspan=1 colspan=1>0.7498</td></tr></table>

![](images/dbd7e9700f992c1fad598eaf43070c9283577a359cdcd5522dec45263ee668e7.jpg)  
Fig. 7. Mixed Prompt: Evaluation Metrics by K

## 4.2.6. Comparative Analysis of Prompting Strategies

The comparative results in Table 8 demonstrate that Mixed Prompting achieved the highest F1 score of 0.6852 at (K=128), followed closely by Prefix Prompting (0.6813). Prefix Prompting achieved the highest classification accuracy of 0.8048, while Cloze Prompting achieved its best F1 score of 0.6532 at (K=32). These findings indicate that Mixed Prompting was more effective in terms of F1 score and recall, whereas Prefix Prompting provided better overall accuracy.

TABLE 8. ANALYSIS OF PERFORMANCE ACROSS PROMPTING STRATEGIES
<table><tr><td rowspan=1 colspan=1>Prompt Type</td><td rowspan=1 colspan=1>Best F1-Score</td><td rowspan=1 colspan=1>Best Accuracy</td></tr><tr><td rowspan=1 colspan=1>Prefix</td><td rowspan=1 colspan=1>0.6813</td><td rowspan=1 colspan=1>0.8048</td></tr><tr><td rowspan=1 colspan=1>Cloze</td><td rowspan=1 colspan=1>0.6532</td><td rowspan=1 colspan=1>0.6984</td></tr><tr><td rowspan=1 colspan=1>Mixed</td><td rowspan=1 colspan=1>0.6852</td><td rowspan=1 colspan=1>0.7580</td></tr></table>

The results in Table 9 and Fig. 8 highlight the effectiveness of prompt-based approaches for Roman Urdu hate speech detection. Among previously published baselines [19], Mistral achieved the highest F1 score (0.9387) and accuracy (0.9642). In the current study, GPT-3.5 few-shot prompting outperformed zero-shot prompting, achieving an F1 score of 0.8565 and accuracy of 0.9100, demonstrating the benefits of incorporating labelled examples without finetuning.

![](images/a4a0328f2ad542e11733e0199e3cf8b9733d62b160b4009ea56ff0c12b5eafe2.jpg)  
Fig. 8. F1 score comparison of prompting strategies and PEFT-based models for Roman Urdu hate speech detection.

TABLE 9. PERFORMANCE COMPARISON OF PROMPTING STRATEGIES AND PEFT-BASED MODELS FOR ROMAN URDU HATE SPEECH DETECTION
<table><tr><td rowspan=1 colspan=1>Technique</td><td rowspan=1 colspan=1>F1</td><td rowspan=1 colspan=1>Acc.</td><td rowspan=1 colspan=1>Prec.</td><td rowspan=1 colspan=1>Recall</td></tr><tr><td rowspan=1 colspan=1>Mistral (PEFT) [19]</td><td rowspan=1 colspan=1>0.9387</td><td rowspan=1 colspan=1>0.9642</td><td rowspan=1 colspan=1>0.9383</td><td rowspan=1 colspan=1>0.9392</td></tr><tr><td rowspan=1 colspan=1>LLaMA (PEFT) [19]</td><td rowspan=1 colspan=1>0.9379</td><td rowspan=1 colspan=1>0.9630</td><td rowspan=1 colspan=1>0.9407</td><td rowspan=1 colspan=1>0.9353</td></tr><tr><td rowspan=1 colspan=1>Falcon-7B (PEFT) [19]</td><td rowspan=1 colspan=1>0.9200</td><td rowspan=1 colspan=1>0.9500</td><td rowspan=1 colspan=1>0.9300</td><td rowspan=1 colspan=1>0.9100</td></tr><tr><td rowspan=1 colspan=1>Gemma (PEFT) [19]</td><td rowspan=1 colspan=1>0.9100</td><td rowspan=1 colspan=1>0.9463</td><td rowspan=1 colspan=1>0.9100</td><td rowspan=1 colspan=1>0.9100</td></tr><tr><td rowspan=1 colspan=1>GPT-3.5 (Few-Shot)</td><td rowspan=1 colspan=1>0.8565</td><td rowspan=1 colspan=1>0.9100</td><td rowspan=1 colspan=1>0.8278</td><td rowspan=1 colspan=1>0.8989</td></tr><tr><td rowspan=1 colspan=1>BERT (PEFT) [19]</td><td rowspan=1 colspan=1>0.8200</td><td rowspan=1 colspan=1>0.8945</td><td rowspan=1 colspan=1>0.8200</td><td rowspan=1 colspan=1>0.8200</td></tr><tr><td rowspan=1 colspan=1>DeepSeek-Qwen</td><td rowspan=1 colspan=1>0.7500</td><td rowspan=1 colspan=1>0.9354</td><td rowspan=1 colspan=1>0.7300</td><td rowspan=1 colspan=1>0.7800</td></tr><tr><td rowspan=1 colspan=1>BERT (Mixed Prompt)</td><td rowspan=1 colspan=1>0.6852</td><td rowspan=1 colspan=1>0.7498</td><td rowspan=1 colspan=1>0.6774</td><td rowspan=1 colspan=1>0.7810</td></tr><tr><td rowspan=1 colspan=1>BERT (Prefix Prompt)</td><td rowspan=1 colspan=1>0.6813</td><td rowspan=1 colspan=1>0.8048</td><td rowspan=1 colspan=1>0.6747</td><td rowspan=1 colspan=1>0.6894</td></tr><tr><td rowspan=1 colspan=1>GPT-3.5 (Zero-Shot)</td><td rowspan=1 colspan=1>0.6698</td><td rowspan=1 colspan=1>0.7229</td><td rowspan=1 colspan=1>0.6799</td><td rowspan=1 colspan=1>0.8129</td></tr><tr><td rowspan=1 colspan=1>BERT (Cloze Prompt)</td><td rowspan=1 colspan=1>0.6532</td><td rowspan=1 colspan=1>0.6984</td><td rowspan=1 colspan=1>0.6097</td><td rowspan=1 colspan=1>0.6833</td></tr></table>

Among BERT-based strategies, the proposed Mixed Prompt Tuning achieved the highest F1 score (0.6852) and recall (0.7810) at (K=128), while Prefix Prompting achieved the highest accuracy (0.8048). Cloze Prompting obtained its best F1 score (0.6532) at (K=32). Increasing few-shot examples generally improved Prefix and Mixed Prompting performance. Previously published PEFT-LoRA results (F1: 0.8200; accuracy: 0.8945) exceeded the BERT-based prompting results, although experimental differences limit direct comparisons. Overall, the findings demonstrate the potential of few-shot prompt engineering and Mixed Prompt Tuning for low-resource hate speech detection, highlighting the importance of prompt design and labelled examples.

## V. CONSLUSIONS

This study investigated the effectiveness of prompt engineering and prompt tuning for RU-HSD using the PURUTT dataset. The experimental evaluation demonstrated that GPT-3.5 few-shot prompting outperformed zero-shot prompting, achieving an F1 score of 0.8565 and accuracy of 0.9100, highlighting the importance of representative labeled examples in low-resource language classification. Among the BERT-based prompting strategies, the proposed Mixed Prompt Tuning approach, combining fixed textual templates with learnable soft embeddings, achieved the highest F1 score of 0.6852 and recall of 0.7810, while Prefix Prompting achieved the highest accuracy of 0.8048. The results further revealed that increasing few-shot training examples improved Prefix and Mixed Prompting performance, although improvements were not consistent across all configurations. Comparisons with previously published PEFT-LoRA [19] and language model baselines provided additional context for evaluating the proposed approaches. Overall, the findings demonstrate the potential of prompt-based learning for Roman Urdu hate speech detection without extensive model finetuning. Future research will focus on improving prompt optimization, investigating cross-domain generalization, and evaluating the proposed approaches across additional lowresource languages and datasets.

## REFERENCES

[1] S. Gururangan et al., “Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks,” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, Stroudsburg, PA, USA: Association for Computational Linguistics, 2020, pp. 8342–8360. doi: 10.18653/v1/2020.aclmain.740.

[2] H. H. Saeed, T. Khalil, and F. Kamiran, “Urdu Toxic Comment Classification With PURUTT Corpus Development,” IEEE Access, vol. 13, pp. 21635–21651, 2025, doi: 10.1109/ACCESS.2025.3535862.

[3] T. Brown et al., “Language Models are Few-Shot Learners,” in Advances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, Eds., Curran Associates, Inc., 2020, pp. 1877–1901. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/2020/file/1457c0 d6bfcb4967418bfb8ac142f64a-Paper.pdf

[4] B. Lester, R. Al-Rfou, and N. Constant, “The Power of Scale for Parameter-Efficient Prompt Tuning,” in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, Stroudsburg, PA, USA: Association for Computational Linguistics, 2021, pp. 3045–3059. doi: 10.18653/v1/2021.emnlp-main.243.

[5] X. L. Li and P. Liang, “Prefix-Tuning: Optimizing Continuous Prompts for Generation,” in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), Stroudsburg, PA, USA:

Association for Computational Linguistics, 2021, pp. 4582–4597. doi: 10.18653/v1/2021.acl-long.353.

T. Schick and H. Schütze, “Exploiting Cloze-Questions for Few-Shot Text Classification and Natural Language Inference,” in Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, Stroudsburg, PA, USA: Association for Computational Linguistics, 2021, pp. 255–269. doi: 10.18653/v1/2021.eaclmain.20.

[7] P. Fortuna and S. Nunes, “A Survey on Automatic Detection of Hate Speech in Text,” ACM Comput. Surv., vol. 51, no. 4, pp. 1– 30, Jul. 2019, doi: 10.1145/3232676.

[8] A. Schmidt and M. Wiegand, “A Survey on Hate Speech Detection using Natural Language Processing,” in Proceedings of the Fifth International Workshop on Natural Language Processing for Social Media, Stroudsburg, PA, USA: Association for Computational Linguistics, 2017, pp. 1–10. doi: 10.18653/v1/W17-1101.

[9] Z. Zhang, D. Robinson, and J. Tepper, “Detecting Hate Speech on Twitter Using a Convolution-GRU Based Deep Neural Network,” 2018, pp. 745–760. doi: 10.1007/978-3-319-93417-4\_48.

[10] A. Gandhi et al., “Hate speech detection: A comprehensive review of recent works,” Expert Syst., vol. 41, no. 8, Aug. 2024, doi: 10.1111/exsy.13562.

[11] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pretraining of Deep Bidirectional Transformers for Language Understanding,” in Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), Stroudsburg, PA, USA: Association for Computational Linguistics, 2019, pp. 4171–4186. doi: 10.18653/v1/N19-1423.

[12] S. Gururangan et al., “Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks,” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, Stroudsburg, PA, USA: Association for Computational Linguistics, 2020, pp. 8342–8360. doi: 10.18653/v1/2020.aclmain.740.

[13] H. Xu, Y. Xia, and Q. Zu, “Instruction fine-tuning based on Llama2-7b for news topic classification,” in Proceedings of the 2023 3rd International Conference on Big Data, Artificial Intelligence and Risk Management, New York, NY, USA: ACM, Nov. 2023, pp. 183–186. doi: 10.1145/3656766.3656798.

[14] D. Nadeau, M. Kroutikov, K. McNeil, and S. Baribeau, “Benchmarking Llama2, Mistral, Gemma and GPT for Factuality, Toxicity, Bias and Propensity for Hallucinations,” 2024. [Online]. Available: https://arxiv.org/abs/2404.09785

[15] Y. Zhang, T. Zhong, T. Yi, and H. Li, “Domain-Enhanced Prompt Learning for Chinese Implicit Hate Speech Detection,” IEEE Access, vol. 12, pp. 13773–13782, 2024, doi: 10.1109/ACCESS.2024.3351804.

[16] Y. Yamagishi, “YYama@Multimodal Hate Speech Event Detection 2024: Simpler Prompts, Better Results - Enhancing Zero-shot Detection with a Large Multimodal Model,” in Proceedings of the 7th Workshop on Challenges and Applications of Automated Extraction of Socio-political Events from Text (CASE 2024), Stroudsburg, PA, USA: Association for Computational Linguistics, 2024, pp. 60–66. doi: 10.18653/v1/2024.case-1.7.

[17] J. Govers, P. Feldman, A. Dant, and P. Patros, “Prompt-GAN– Customisable Hate Speech and Extremist Datasets via Radicalised Neural Language Models,” in Proceedings of the 2023 9th International Conference on Computing and Artificial Intelligence, New York, NY, USA: ACM, Mar. 2023, pp. 515– 522. doi: 10.1145/3594315.3594366.

[18] L. Han and H. Tang, “Designing of Prompts for Hate Speech Recognition with In-Context Learning,” in 2022 International Conference on Computational Science and Computational Intelligence (CSCI), IEEE, Dec. 2022, pp. 319–320. doi: 10.1109/CSCI58124.2022.00063.

[19] T. Zubair, M. J. Asif, F. Kamiran, H. H. Saeed, and R. F. Ahmad, “Efficient Adaptation of LLMs for Hate Speech Detection in Low-Resource Languages: A Comparative Study on Roman Urdu,” 2026. [Online]. Available: https://arxiv.org/abs/2608.18142