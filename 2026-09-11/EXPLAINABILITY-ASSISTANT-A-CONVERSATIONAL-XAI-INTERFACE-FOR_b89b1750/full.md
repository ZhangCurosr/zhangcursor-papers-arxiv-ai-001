# EXPLAINABILITY ASSISTANT: A CONVERSATIONAL XAI INTERFACE FOR INTERPRETING ENERGY CONSUMPTION MODELS

Rodion Krjutškov   
Nupp Software   
Tallinn, Estonia   
rodionnupp.dev   
Eduard Barbu [ORCID]   
Institute of Computer Science   
University of Tartu   
Tartu, Estonia   
eduard.barbuut.ee   
Nikos Sakkas   
Apintech Ltd   
POLIS-21 Group   
Cyprus   
sakkasapintech.com

Sofia Yfanti Department of Mechanical Engineering Hellenic Mediterranean University Heraklion, Greece sifantihmu.gr

September 11, 2026

## ABSTRACT

Energy consumption forecasting relies on increasingly complex machine learning (ML) models, such as Genetic Programming-based symbolic regressors, whose predictions can be difficult for facility managers and building operators to interpret. Explainable Artificial Intelligence (XAI) techniques address this opacity, but traditional XAI dashboards require substantial technical expertise and provide limited flexibility for dynamic, context-aware inquiry. Conversational XAI systems offer a promising alternative; however, previous approaches, such as TalkToModel, were constrained by rigid custom grammars and achieved only 76.8% intent-parsing accuracy. This paper introduces the Explainability Assistant, an open-source conversational XAI system that leverages the function-calling capabilities of modern Large Language Models (LLMs) to overcome these limitations. The system achieves 94% intent-parsing accuracy, supports flexible natural language interaction, and adapts to different ML problem types without task-specific fine-tuning. We present the system’s architecture and report results from a comparative evaluation conducted with energy domain specialists, contrasting the Explainability Assistant with a traditional XAI dashboard. The evaluation suggests improved usability and consistent task accuracy, with all experts unanimously preferring the conversational interface for practical use.

Keywords Explainable AI · Conversational AI · Energy Consumption Forecasting · Large Language Models · Function Calling

## 1 Introduction

ML models are increasingly deployed in high-stakes domains such as energy management, healthcare, and finance, where understanding model predictions is critical for trust and accountability [1, 2]. In building energy systems, for example, facility managers rely on consumption forecasts produced by complex models, yet have limited means to verify why a particular prediction was made. Post-hoc explainability techniques, such as LIME [1] and SHAP [2], as well as counterfactual explanations [3], have emerged as popular tools to address this opacity. However, practitioners struggle to select appropriate explanation methods, interpret their outputs, and answer follow-up questions during model analysis [4].

Traditional explainability interfaces, such as point-and-click dashboards, require users to possess substantial technical expertise and often fail to support flexible, iterative exploration. This is particularly limiting in applied settings such as building energy management, where the end users are facility managers and engineers rather than data scientists. Natural language dialogue systems offer a promising alternative, enabling open-ended and accessible interactions that allow users to ask questions, explore model behavior dynamically, and receive contextually relevant explanations without needing to understand the underlying technical machinery.

Building on the foundations of TalkToModel [5], a conversational system that translates user utterances into executable operations for explainability, we introduce the Explainability Assistant (EA), an enhanced dialogue-based system for understanding ML models. The Explainability Assistant addresses key limitations of TalkToModel by leveraging recent advances in LLMs with function-calling capabilities, enabling more accurate natural language understanding, improved architectural modularity, and broader applicability across diverse use cases.

We evaluate the Explainability Assistant in energy consumption forecasting and demonstrate that it achieves significantly higher parsing accuracy (up to 94%), surpassing the 76.8% accuracy reported for previous grammar-based approaches. Through evaluation using both automatic metrics and stakeholder feedback, we show that the system supports diverse explainability tasks, including feature importance analysis, counterfactual reasoning, ML model error analysis, and what-if queries.

Our contributions are threefold:

1. A conversational architecture that integrates modern LLMs for natural language understanding in explainability contexts.

2. A lightweight, modular system design that separates concerns and enables flexible deployment.

3. Empirical evidence that dialogue-based interfaces can support improved model understanding for domain experts and ML practitioners.

The rest of the paper is organized as follows. Section 2 reviews relevant work in explainability and conversational interfaces. Section 3 presents the architecture and XAI tools underlying the Explainability Assistant. Section 4 reports results from the parsing evaluation and the expert validation with energy domain specialists. Section 5 concludes the paper.

We provide all materials required to assess the system: a live demo, a video walkthrough of the system, the gold parse datasets, the source code (front-end and back-end), and the full expert validation questionnaire. All links are collected in Appendix B.

## 2 Related Work

The need for interpretable ML systems has driven extensive research into explainability techniques. Early work focused on inherently interpretable models [6, 7], while post-hoc methods emerged for black-box models. LIME [1] and SHAP [2] compute feature importance, counterfactual explanations [3] identify minimal input changes that alter predictions, and gradient-based methods [8] visualize neural network behavior. Despite these advances, practitioners struggle to select appropriate methods and interpret results [4, 9].

Interactive tools combine multiple explanation techniques into unified interfaces. Dashboard-based systems, such as the Language Interpretability Tool [10] and the What-If Tool [11], provide point-and-click access to explanations, but still require users to understand which operations to perform. TalkToModel [5] introduced natural language dialogue for explainability using fine-tuned language models with custom grammars, demonstrating improved user understanding but achieving only 76.8% parsing accuracy and requiring complex task-specific adaptation.

Modern LLMs, including the GPT, Llama, and Gemini families, support function-calling mechanisms for tool orchestration, enabling applications in conversational QA [12], data analysis [13], and agentic tasks [14, 15]. Recent work has begun applying these capabilities to XAI solutions. Wang et al. [16] introduced LLMCheckup, a conversational XAI system that uses prompt engineering and in-context learning to enable dialogue-based model explanations, relying on prompt chaining to invoke explanation methods. Samimi et al. [17] developed a visual-conversational interface for diabetes risk prediction that combines a fine-tuned T5 parser with LLM-based dialogue and interactive visualizations, retaining a task-specific fine-tuned component alongside LLMs. At a different level of abstraction, Shaham et al. [18] proposed MAIA, a multimodal agent that interprets neural network internals through autonomous experimentation, targeting model internals interpretability for researchers rather than prediction-level explainability for end users.

The Explainability Assistant differs from these systems in its use of structured LLM function-calling to translate natural language queries into explainability operations, providing more predictable tool invocation than prompt chaining [16], eliminating task-specific fine-tuning requirements [17], and focusing on prediction-level explanations accessible to domain experts in applied settings such as energy management.

![](images/285630e900558e7fc3f6e9ca96db07a07e76495f28efdc847744e6685231d764.jpg)  
Figure 1: Explainability Assistant system architecture.

## 3 System Description

## 3.1 System Architecture

The Explainability Assistant employs a modular architecture that decouples the presentation layer from the core computational logic, as illustrated in Figure 1. This separation enhances maintainability, enables independent scaling of components, and facilitates integration with diverse frameworks and LLM providers.

Front-end. The user interface is implemented using Next.js, providing a conversational interface for natural language interaction, as illustrated in Figure 2. The front-end manages client-side conversation history, handles user authentication and session management, and allows users to select from multiple supported LLMs (e.g., Llama-3.3-70B-Instruct, Gemini-2.0-Flash, Gemini-2.5-Flash). This design enables users to switch between models and providers, even within a single session.

Back-end. The server-side component is implemented as a FastAPI application that orchestrates data operations and ML model inference, executes pre-defined explainability functions, and manages communication with external LLM providers via API connections. The back-end leverages modern LLMs’ function-calling capabilities: rather than fine-tuning models on task-specific grammars, the system provides the LLM with the user query, conversation history, and JSON specifications of available functions (including names, descriptions, and required parameters). The LLM then determines which function(s) to invoke and generates the appropriate arguments in a structured format.

This function-calling paradigm offers several key advantages:

• Improved Accuracy. Intent extraction accuracy exceeds 90% for capable models (e.g., Gemini-2.5-Flash achieves 93-94%), compared to 76.8% for the grammar-based system with T5 models in TalkToModel. This improvement is enabled both by the function-calling architecture and by advances in the underlying LLM capabilities.

• Enhanced Transparency. The LLM is constrained to respond using a structured JSON format that includes a mandatory free-form explanation field describing the assistant’s planned actions (e.g., "I will predict the electricity consumption for sample ID 42 and explain which features most influenced this prediction"). This design exposes the system’s intended actions and explicitly lists the functions to be executed, allowing users to verify the system’s interpretation and making it more trustworthy.

• Data Privacy. Function execution occurs entirely within the back-end, retrieving results directly from local data and models without the necessity to transmit sensitive information to external LLM providers. Only the user query and tool schemas (and optionally summarized outputs) are sent to the LLM provider. This ensures that proprietary datasets and model internals can remain protected.

• Deployment Flexibility. The system adapts to new datasets and model types through configuration changes alone, as demonstrated with both energy consumption regression and clinical risk classification, without requiring task-specific fine-tuning.

It is worth noting that general-purpose tool-calling protocols, such as the Model Context Protocol (MCP) [19], have recently emerged to standardize LLM-to-tool connectivity. However, the Explainability Assistant provides a domainspecific conversational layer that goes beyond raw tool invocation: it manages multi-turn XAI dialogues, performs automatic parameter selection for explanation methods based on conversational context, and presents results in a form accessible to non-expert users.

## 3.2 XAI Tools

The Explainability Assistant integrates several established post-hoc explanation techniques to support model interpretability.

Feature Importance Analysis. Understanding which features most influence model predictions is fundamental to model interpretability. The system employs SHAP [2], a model-agnostic method grounded in game-theoretic principles that provides consistent and theoretically justified feature attributions. Users can request explanations of feature importance for individual predictions (e.g., "Why was sample 42 predicted to have such consumption level?") or global summaries across the entire dataset (e.g., "What are the three most important features overall?").

Counterfactual Explanations. Counterfactual reasoning addresses contrastive questions of the form "What would need to change for the prediction to be different?" The system implements counterfactual generation using DiCE [3], which identifies minimal, realistic modifications to input features that would alter the model’s prediction. This capability supports actionable insights: for instance, in the energy consumption prediction use case, a specialist might ask, "What changes would move this prediction from high consumption to low?" to explore minimal required adjustments.

What-If Scenario Analysis. Beyond automated counterfactual generation, the system supports user-directed what-if queries that simulate specific hypothetical changes. Users can specify precise feature modifications and observe their effects on predictions, enabling exploratory analysis such as "If indoor temperature increases by 5 degrees, how much will energy consumption change?" This functionality is particularly valuable for policy analysis and sensitivity testing.

## 3.3 Use Cases

We demonstrate the capabilities of the Explainability Assistant through two distinct use cases: energy demand forecasting and heart disease classification.

Energy Demand Forecasting (Regression). This use case addresses mid-term forecasting of electricity demand in the commercial sector. The dataset originates from a real-world monitoring system deployed in an office building [20], combining smart meter readings with environmental sensor data. The primary predictors include indoor temperature, outdoor temperature, and time-shifted previous energy consumption values. We applied the EA to a Genetic Programming symbolic regressor model trained by the original stakeholder, enabling users to evaluate model performance and analyze feature attributions through natural language dialogue.

Heart Disease Classification (Classification). To demonstrate the architectural modularity of the Explainability Assistant beyond regression tasks, we deployed the system on the Heart Disease Cleveland dataset [21]<sup>1</sup> (303 patients, 14 clinical attributes, Decision Tree classifier). By simply updating the JSON function specifications, without modifying the core application logic or fine-tuning the LLMs, the system successfully transitioned from predicting continuous energy values to explaining binary classification risk factors. This confirms that the function-calling paradigm enables rapid adaptation to new domains.

## 4 Evaluation

The Explainability Assistant is evaluated through two complementary approaches: LLM parsing accuracy and an expert validation with energy domain specialists.

![](images/581f04fb5a259957ce71b2a7b5cd0ccf117190e61d5a704495e964548fcefac8.jpg)  
Figure 2: Explainability Assistant user interface.

![](images/f11b51aa42dd57430b7423f1661dd8871d5365b028e01b8af2638f01a22eb818.jpg)  
Figure 3: Individual expert ratings (1-5 Likert scale) comparing Explainer Dashboard (left) and Explainability Assistant (right) across four dimensions: trust in outputs, ease of use, system understanding, and likelihood of future use. Each colored line represents one expert’s ratings.

## 4.1 LLM Parsing Accuracy

Intent classification reliability was measured across multiple state-of-the-art LLMs using gold-standard parse datasets. Each evaluation instance consisted of user input, prior conversation history, and expected function call(s), as exemplified in Table 1.

![](images/32642500c396db34c657cfd64d2d72b71b6a2d7fe3cbf24af6caaa425b53e348.jpg)  
Table 1: Gold parse dataset sample example.

Three gold parse datasets were constructed for the energy consumption forecasting use case. Dataset A (20 samples) was created manually to ensure coverage of core system functionalities (20 functions available in total). Datasets B and C (80 samples each) were generated using frontier LLMs GPT-5 and Gemini-2.5-Pro with Dataset A as few-shot examples, then manually verified and corrected. Datasets B and C were generated using LLMs that partially overlap with those evaluated, but manual verification of all samples and cross-validation on two combined sets ensure that the evaluation remains independent of the generation process.

Four cost-effective frontier LLMs were evaluated: GPT-5-mini, Llama-3.3-70B-Instruct, Gemini-2.0-Flash, and Gemini-2.5-Flash. Exact match parsing accuracy was measured on combined datasets A+B and A+C (100 samples each).

Table 2 presents the results. Gemini-2.5-Flash achieved the highest accuracy (94% on A+B, 93% on A+C), followed by Gemini-2.0-Flash (88% and 91%). This demonstrates that modern function-calling architectures can achieve production-grade reliability (>90%) without the task-specific fine-tuning required by earlier grammar-based systems, which typically plateaued around 75% [5].

<table><tr><td>Model</td><td>A+B</td><td>A+C</td></tr><tr><td>GPT-5-mini</td><td>74%</td><td>79%</td></tr><tr><td>Llama-3.3-70B</td><td>78%</td><td>74%</td></tr><tr><td>Gemini-2.0-Flash</td><td>88%</td><td>91%</td></tr><tr><td>Gemini-2.5-Flash</td><td>94%</td><td>93%</td></tr></table>

Table 2: Exact match parsing accuracy on combined gold parse datasets.

## 4.2 Expert Validation Design

To assess practical utility, we conducted a within-subjects comparison study with three energy domain specialists. Participants were professionals actively involved in building energy management and policy decisions, with 10-20 years of experience in the energy sector and varying levels of familiarity with ML systems (self-reported as intermediate to expert). This expert-driven evaluation prioritizes depth of domain insight over sample breadth, consistent with established practices for evaluating specialized decision-support tools in engineering contexts. The small participant count reflects the limited pool of domain experts with direct access to the deployed building energy management system and familiarity with its underlying ML models; the within-subjects design partially compensates for the small sample size by ensuring that individual differences are controlled across conditions.

Participants engaged in a simulated decision-support scenario: managing energy consumption in a commercial building during business hours. An ML model provided hourly energy consumption predictions (regression output) based on three features: indoor temperature, outdoor temperature, and past consumption. We compared two interfaces providing access to the same underlying model and explanation methods:

Explainer Dashboard (ED): A traditional point-and-click graphical interface based on the open-source Explainer Dashboard framework<sup>2</sup>. The dashboard provided tabbed access to SHAP feature importances, individual predictions, global feature importance rankings, and interactive sliders for what-if analysis.

Explainability Assistant (EA): The conversational interface described in Section 3, configured with the same dataset and model. Users interacted through natural language queries to request predictions, explanations, and scenario analyses.

The study was conducted asynchronously via Google Forms to accommodate participants’ schedules. Each participant received links to both interface deployments along with detailed usage instructions. The survey consisted of six sections: (1) background information collection, (2) Task Block A using ED, (3) subjective evaluation of ED, (4) Task Block B using EA, (5) subjective evaluation of EA, and (6) open-ended feedback.

Each task block contained five questions requiring participants to synthesize ML predictions, XAI explanations, and domain knowledge. Questions were carefully designed to be parallel in structure and difficulty while using different data instances to prevent memorization. Example tasks included: "What is the predicted energy consumption for hour X?", "Which feature most influences this prediction?", "What would happen to consumption if indoor temperature increased by Y degrees?", "How accurate is the model overall?", and "Identify a case where the model makes an error." Complete task questions are provided in Appendix C.

Following each task block, participants rated four dimensions on five-point Likert scales (1=Strongly Disagree, 5=Strongly Agree): ease of use, trust and confidence in the interface, understanding of how the system works, and likelihood of using the interface in future work.

Our design deliberately emphasized measuring appropriate reliance and objective task accuracy rather than subjective trust alone, addressing concerns about potential illusion of explanatory depth in conversational systems [22].

## 4.3 Expert Validation Results

The participants completed ten task questions (five per interface) with high accuracy, achieving 93% correct responses using ED and 100% using EA. Although not statistically significant given the sample size, this suggests that the conversational interface maintains or improves objective performance while offering usability advantages.

Figure 3 presents individual expert ratings across four evaluation dimensions. For the Explainer Dashboard (left), ratings varied considerably: Expert 1 (blue) provided low ratings (2-3), Expert 2 (orange) gave the highest ratings (3-5), and Expert 3 (green) showed intermediate ratings.

The Explainability Assistant (right) received consistently higher and more uniform ratings. All three experts rated "Easy to use" and "Would use again" at or near maximum (5), demonstrating unanimous agreement on usability and practical value. Experts 1 and 3 showed slightly more measured enthusiasm for "Trust outputs" and "Understand system" (3-4), while Expert 2 provided near-perfect scores across all dimensions.

The most striking contrast appears in "Would use again," where EA achieved unanimous maximum ratings (5) compared to ED’s variable scores (2-4). Similarly, "Easy to use" improved dramatically from ED (2-5, with disagreement) to EA (all 5s). These patterns suggest the conversational paradigm successfully addresses usability barriers in traditional dashboards.

The more conservative ratings for "Trust outputs" and "Understand system" (3-5) suggest participants maintained appropriate epistemic caution about model outputs and system mechanics, aligning with the design goal of supporting appropriate reliance rather than uncritical trust. Participants noted that ED’s visual overview was occasionally useful for orientation, suggesting hybrid approaches may offer complementary benefits.

## 4.4 Discussion

The evaluation demonstrates that the Explainability Assistant achieves two key objectives: (1) substantially improved reliability in comprehension compared to prior conversational XAI systems, and (2) offered practical utility for domain experts, allowing actionable insights based on ML model predictions. The 94% intent classification accuracy, achieved through effective integration of LLM function-calling with frontier models, addresses a central limitation of earlier grammar-based systems, where frequent misinterpretations undermined user’s trust and prevented production deployments. The structured function-calling format further increases transparency by exposing the system’s intended actions before execution.

The expert validation shows that the conversational interface reduces usability friction without sacrificing analytical quality. Experts unanimously preferred EA in terms of usability and future adoption while maintaining high task accuracy. At the same time, ratings for "trust outputs" and "understand system" remained moderate, suggesting appropriate epistemic caution, which is aligned with our design goals. Participants also noted that traditional dashboards still provide value for global orientation, indicating that hybrid conversational-visual interfaces merit exploration.

## 5 Conclusion

We presented the Explainability Assistant, a conversational XAI system that leverages LLM function-calling to achieve 94% intent classification accuracy, representing a substantial improvement over prior grammar-based systems. The modular architecture eliminates task-specific fine-tuning requirements and enhances transparency through structured responses that expose system reasoning before execution.

Our evaluation with energy domain specialists demonstrated that the conversational paradigm successfully addresses usability barriers in traditional dashboards, with all three experts unanimously preferring the Explainability Assistant for ease of use and likelihood of future adoption while maintaining high task accuracy (100% vs. 93%). The system advances conversational XAI through improved comprehension reliability, reduced deployment complexity, enhanced user experience, and reasoning transparency. By making sophisticated XAI techniques accessible through natural language, the system lowers barriers for domain experts to engage with ML model understanding regardless of their technical background. The open-source implementation enables the research community to extend conversational XAI to new application domains.

Future work should include larger-scale studies with diverse user populations across multiple domains, temporal analyses of extended usage patterns to understand long-term adoption, and investigation of hybrid interfaces that combine conversational flexibility with strategic use of visualization, as suggested by participant feedback.

## ACKNOWLEDGMENT

This research was conducted under the Transparent, Reliable, and Unbiased Smart Tool for AI (Trust-AI) project (Grant Agreement No. 952060), funded by the European Commission. This work was also supported by the Estonian Research Council grant PRG2006.

## Review Materials

To facilitate evaluation, readers are provided with: (1) a video demonstration of the system, (2) a public demo interface, (3) the complete source code, (4) gold parse datasets and evaluation scripts, and (5) the full expert validation questionnaire. All links are listed in Appendix B.

## A Ethics Statement

This work follows established research ethics guidelines. The Explainability Assistant is designed to improve transparency and appropriate reliance in machine learning systems by exposing the model’s intended actions before execution and avoiding hidden reasoning steps. All data used in system development are either synthetic, publicly available (e.g., Heart Disease Cleveland dataset), or provided by industry partners in anonymized form. No personally identifiable information is processed or transmitted to LLM providers when all function execution and data access occur locally within the back-end. The expert validation involved domain experts using non-sensitive energy data; no personal information or behavioral logs were collected. The system is intended for decision support and not for autonomous or high-stakes decision-making.

## B Resources

1. Explainability Assistant website: https://explainabilityassistant.com

2. Demo access to the EA chat interface: https://explainabilityassistant.com/demo

3. Introductory video presentation about Explainability Assistant: https://www.youtube.com/watch?v=36Tm3kof-tc

4. Explainability Assistant, back-end application code: https://github.com/krkv/explainability-backend

5. Explainability Assistant, front-end application code: https://github.com/krkv/explainability-frontend

6. Gold parse datasets, dataset generation prompts and scripts, LLM evaluation scripts: https://github.com/krkv/llm-evaluation

## C Expert Validation Questions

## C.1 Introduction and Background Questions

Hello! This survey is designed to evaluate your experience and compare preferences in using two different software tools for explainable machine learning.

In this scenario, a machine learning model is used to forecast energy consumption of a building based on time series data (indoor temperature, outdoor temperature and energy consumption in the past hour).

You will need to find answers to a few questions about these predictions. It should take you up to 30 minutes. Please read the instructions in the beginning of every section. Thank you for participating!

• How many years of experience do you have in the energy domain?

• How familiar are you with machine learning algorithms and applications?

• What is your current role? (optional)

## C.2 Block A Task Questions

For answering these questions, please open the Explainer Dashboard app and find the answers there. If you can’t find the answer, leave the field empty or select "I can’t tell".

• What is the prediction for ID 33?

• Which feature contributed the most to this prediction?

• How would the prediction for ID 33 change if the indoor temperature changed to 20 degrees?

• How big is the error in the prediction of ID 33?

• Find another example of how the model prediction can change (what are the input values and what is the model prediction)

## C.3 Block A Post-Task Questions

Please answer if you agree with these statements about your experience using the Explainer Dashboard app.

• It was easy to use the dashboard to find the answers.

• I trust the dashboard and the outputs it gave.

• I understand how this system operates.

• I would use this type of tool again in the future.

## C.4 Block B Task Questions

For answering these questions, please open the Explainability Assistant app and find the answers there. If you can’t find the answer, leave the field empty or select "I can’t tell".

• What is the prediction for ID 55?

• Which feature contributed the most to this prediction?

• How would the prediction for ID 55 change if the indoor temperature changed to 20 degrees?

• How big is the error in the prediction of ID 55?

• Find another example of how the model prediction can change (what are the input values and what is the model prediction)

## C.5 Block B Post-Task Questions

Please answer if you agree with these statements about your experience using the Explainability Assistant app.

• It was easy to use the assistant to find the answers.

• I trust the assistant and the outputs it gave.

• I understand how this system operates.

• I would use this type of tool again in the future.

## C.6 Open-ended Feedback

Feel free to go back to the Explainability Assistant tool and explore it more. You can try different types of questions, also there are examples that can be toggled in the top right corner.

Please share any feedback, ideas and thoughts you might have after completing these questions and interacting with both interfaces.

Your opinion as an expert is very important, and we are very thankful for your contribution!

## D Limitations

Several limitations of this work warrant consideration. The expert validation involved a small sample size (n=3), which limits statistical power and generalizability of the findings. However, the within-subjects design and unanimous preference patterns on key usability dimensions provide meaningful initial evidence of the system’s practical utility.

The asynchronous study format, while accommodating participants’ schedules, prevented observation of real-time interaction patterns and think-aloud protocols that could reveal deeper insights into how experts leverage each interface and formulate queries during exploratory analysis.

Because the study used a fixed interface order (Explainer Dashboard first, then Explainability Assistant), participants may have benefited from learning/familiarization with the task structure and data in Block A, potentially inflating performance or usability ratings in Block B. Future work should counterbalance interface order to control for these ordering effects.

Our evaluation focused on a single domain (energy forecasting) with a specific model type (symbolic regression). While the architectural modularity and demonstrated parsing accuracy across diverse query types suggest potential for broader applicability, validation in other domains, especially in high-stakes settings like healthcare, remains necessary to establish generalizability.

## References

[1] M. T. Ribeiro, S. Singh, and C. Guestrin, “"Why should i trust you?" Explaining the Predictions of Any Classifier,” in Proceedings ofthe 22nd ACM SIGKDD international conference on knowledge discovery and data mining, 2016, pp. 1135–1144.

[2] S. M. Lundberg and S.-I. Lee, “A Unified Approach to Interpreting Model Predictions,” Advances in neural information processing systems, vol. 30, 2017.

[3] R. K. Mothilal, A. Sharma, and C. Tan, “Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations,” in Proceedings of the 2020 conference on fairness, accountability, and transparency, 2020, pp. 607–617.

[4] H. Kaur, H. Nori, S. Jenkins, R. Caruana, H. Wallach, and J. Wortman Vaughan, “Interpreting Interpretability: Understanding Data Scientists’ Use of Interpretability Tools for Machine Learning,” in Proceedings ofthe 2020 CHI conference on humanfactors in computing systems, 2020, pp. 1–14.

[5] D. Slack, S. Krishna, H. Lakkaraju, and S. Singh, “TalkToModel: Explaining Machine Learning Models with Interactive Natural Language Conversations,” arXiv preprint arXiv:2207.04154, 2023.

[6] Y. Lou, R. Caruana, J. Gehrke, and G. Hooker, “Accurate Intelligible Models with Pairwise Interactions,” in Proceedings ofthe 19th ACM SIGKDD international conference on Knowledge discovery and data mining, 2013, pp. 623–631.

[7] R. Caruana, Y. Lou, J. Gehrke, P. Koch, M. Sturm, and N. Elhadad, “Intelligible Models for Healthcare: Predicting Pneumonia Risk and Hospital 30-day Readmission,” in Proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining, 2015, pp. 1721–1730.

[8] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-cam: Visual Explanations from Deep Networks via Gradient-Based Localization,” in Proceedings ofthe IEEE international conference on computer vision, 2017, pp. 618–626.

[9] Q. V. Liao, D. Gruen, and S. Miller, “Questioning the AI: Informing Design Practices for Explainable AI User Experiences,” Proceedings ofthe 2020 CHI Conference on Human Factors in Computing Systems, pp. 1–15, 2020.

[10] I. Tenney, J. Wexler, J. Bastings, T. Bolukbasi, A. Coenen, S. Gehrmann, E. Jiang, M. Pushkarna, C. Radebaugh, and E. Reif, “The Language Interpretability Tool: Extensible, Interactive Visualizations and Analysis for NLP Models,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 2020, pp. 107–118.

[11] J. Wexler, M. Pushkarna, T. Bolukbasi, M. Wattenberg, F. Viégas, and J. Wilson, “The What-If Tool: Interactive Probing of Machine Learning Models,” in IEEE transactions on visualization and computer graphics, vol. 26, no. 1, 2019, pp. 56–65.

[12] L. Gao, A. Madaan, S. Zhou, U. Alon, P. Liu, Y. Yang, J. Callan, and G. Neubig, “PAL: Program-Aided Language Models,” in International Conference on Machine Learning. PMLR, 2023, pp. 10 764–10 799.

[13] W. Zhang, Y. Shen, W. Lu, and Y. Zhuang, “Data-Copilot: Bridging Billions of Data and Humans with Autonomous Workflow,” in ICLR 2024 Workshop on Large Language Model (LLM) Agents, 2024. [Online]. Available: https://openreview.net/forum?id=iUEBnQyfWE

[14] Y. Qin, S. Liang, Y. Ye, K. Zhu, L. Yan, Y. Lu, Y. Lin, X. Cong, X. Tang, B. Qian et al., “ToolLLM: Facilitating Large Language Models to Master 16000+ Real-World APIs,” arXiv preprint arXiv:2307.16789, 2023.

[15] T. Schick, J. Dwivedi-Yu, R. Dessì, R. Raileanu, M. Lomeli, L. Zettlemoyer, N. Cancedda, and T. Scialom, “Toolformer: Language Models Can Teach Themselves to Use Tools,” Advances in Neural Information Processing Systems, vol. 36, 2024.

[16] Q. Wang, T. Anikina, N. Feldhus, J. van Genabith, L. Hennig, and S. Möller, “LLMCheckup: Conversational Examination of Large Language Models via Interpretability Tools and Self-Explanations,” 2024. [Online]. Available: https://arxiv.org/abs/2401.12576

[17] R. Samimi, A. Bhattacharya, L. Gosak, G. Stiglic, and K. Verbert, “Visual-Conversational Interface for Evidence-Based Explanation of Diabetes Risk Prediction,” in Proceedings of the 7th ACM Conference on Conversational User Interfaces, ser. CUI ’25. New York, NY, USA: Association for Computing Machinery, 2025. [Online]. Available: https://doi.org/10.1145/3719160.3736616

[18] T. R. Shaham, S. Schwettmann, F. Wang, A. Rajaram, E. Hernandez, J. Andreas, and A. Torralba, “A Multimodal Automated Interpretability Agent,” 2025. [Online]. Available: https://arxiv.org/abs/2404.14394

[19] Anthropic, “Model Context Protocol,” https://modelcontextprotocol.io, 2024, accessed: March 2025.

[20] N. Sakkas, S. Yfanti, P. Shah, N. Sakkas, C. Chaniotakis, C. Daskalakis, E. Barbu, and M. Domnich, “Explainable Approaches for Forecasting Building Electricity Consumption,” Energies, vol. 16, no. 20, 2023. [Online]. Available: https://www.mdpi.com/1996-1073/16/20/7210

[21] R. Detrano, A. Janosi, W. Steinbrunn, M. Pfisterer, J. Schmid, S. Sandhu, K. Guppy, S. Lee, and V. Froelicher, “Heart Disease Cleveland UCI,” https://archive.ics.uci.edu/ml/datasets/heart+Disease, 1988, also available on Kaggle: https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland.

[22] G. He, N. Aishwarya, and U. Gadiraju, “Is Conversational XAI All You Need? Human-AI Decision Making With a Conversational XAI Assistant,” in Proceedings of the 30th International Conference on Intelligent User Interfaces, ser. IUI ’25. Association for Computing Machinery, 2025, pp. 907–924. [Online]. Available: https://doi.org/10.1145/3708359.3712133