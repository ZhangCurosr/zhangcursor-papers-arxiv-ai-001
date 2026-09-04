# Analysis of Prompt Engineering for Drug Toxicity Prediction

Mia MacGregor<sup>∗,1</sup>, Aakash Welgamage Don<sup>2</sup>, and Mark Bartlett<sup>1</sup>

<sup>1</sup> School of Computing, Engineering & Technology, Robert Gordon University, Aberdeen, UK.

<sup>2</sup> School of Pharmacy, Applied Sciences & Public Health, Robert Gordon University, Aberdeen, UK.

ORCID codes: FA 0009-0005-6021-2894; SA 0000-0001-8364-2495; TA 0000-0003-3383-4100.

corresponding author: m.macgregor1@rgu.ac.uk

Keywords: Prompt Engineering, Machine Learning, Drug Toxicity.

Abstract Clinical trials in the UK can cost up to £1.3 million, with an approximately 90% drug failure rate. Toxicity is a main contributing factor in drug failure. Traditional toxicity testing is time and cost intensive. In recent years, the use of artificial intelligence has been increasingly explored to aid in the prediction of drug toxicity, with extensive use of large language models (LLMs), such as ChatGPT. However, LLMs outputs can show considerable variation when minor changes are made to the prompts given to them. The aim of the paper is to investigate how important prompt phrasing is in relation to drug toxicity prediction. A machine learning pipeline was created that involved prompting LLMs at two points in the process. The experiments show that the natural variance which occurs in the outputs of LLMs outweighed any fine-tuning of prompts. There were, however, substantial improvements in model performance when using chemoinformatic code to extract features instead of using LLM generated values, which are common in the field. The proposed analysis methodology is applicable to a wide range of prompt types across different areas of bioinformatics.

## 1 Introduction

UK Clinical trials can cost up to £1.3 million [1] with only approximately 10% of drugs included in phase I clinical trials reaching UK markets [2]. One of the main reasons for failure in clinical trials is toxicity [3], which is a measure of harm that a chemical can cause to living organisms. Wet-lab toxicity prediction is associated with considerable time and cost burdens [4].

In recent years, scientists have begun using AI methods to predict toxicity prior to clinical trials, enhancing toxicity prediction by use of interdisciplinary tools, utilizing in silico drug toxicity prediction [5]. One method in recent literature is to use large language models (LLM) [6, 7, 8, 9]. The responses that LLMs provide can vastly differ depending on the prompt they are provided [6], but the impact of the phrasing of the prompt used is often not considered in these studies. This paper addresses this gap by proposing a method to analyse prompt engineering used for drug toxicity prediction. Prompt engineering is the method used to optimise an LLM output [10]. We investigate the impact of prompt engineering on the variation of LLM outputs generated, as well as on the accuracy of the subsequent toxicity prediction task.

Drug toxicity can be classified in different ways depending on the training data. Using different datasets allows our approach to investigate specific toxicity types, with toxic and non-toxic classifications determined by the machine learning (ML) models training data.

## 1.1 Related and Background Work

Recent years have seen growing interest in the application of AI for drug toxicity prediction. [5]. Attention has recently shifted to using LLMs for this task. This section reviews some of the recent studies which adopt an approach similar to methodology studied in this work.

Zero-shot methods (such as [7]) ask an LLM whether a chemical is toxic without providing examples. Few-shot methods ask the same question but also provide examples of the desired outputs for given inputs, as in [7, 8]. These studies typically represent chemicals as SMILES (Simplified Molecular-Input Line-Entry System), which encode chemical information as a single line of text. [8] combines a general prompt with a task-specific template, using GPT models in a few-shot setting by providing example SMILES with toxicity labels before asking for the toxicity of a SMILES with unknown toxicity. The LLM returns 0 (non-toxic) or 1 (toxic). However, it is unclear whether the model is predicting toxicity or recalling it from its training data (the data leakage problem), making its ability to predict the toxicity of novel chemicals uncertain.

The state-of-the-art technique [9] presents a methodology most similar to the one studied in this work. Rather than asking the LLM to directly classify the chemical as toxic or not, the LLM is asked to provide data about each chemical. This dataset is then used to train a ML model to predict toxicity; this avoids data leakage. LLMs undertake knowledge synthesis from the literature and then must identify inferred data rules from a labelled dataset to aid in toxicity prediction. These rules are used to create feature vectors for chemicals which contain information about each chemical stored as a list of 1’s and 0’s corresponding to a yes or no answer to a question. The vectors are then used to train ML models to predict the toxicity.

## 2 Methods and Results

The general methodology used is described below, and each stage is explored in detail in subsequent sections. Initially, a prompt is passed to a LLM, asking it to output a list of 15 features that could be used in predicting the toxicity of a chemical. Unlike [9], we allow for features which are not just yes or no values, such as the inclusion of features like LogP or molecular weight. The LLM is then asked to produce the value associated with each feature for each of a list of SMILES obtained from PubChem [11]. This results in a dataset containing a column for each of the 15 features and a row for each chemical. The ground truth for whether each chemical is toxic or not was also obtained from PubChem and appended to the dataset. The ML models were trained to predict toxicity using 80% of the data and then evaluated on the remaining 20%. The models produce feature importance values, and metrics are used to evaluate these. This process is repeated using different prompts and LLMs in order to study the impact of the prompt phrasing on the overall quality of the prediction made.

## 2.1 Initial Prompt and Feature Generation

Prompts were designed to generate a list of 15 features that are of importance when predicting drug toxicity. Prompts were constructed to investigate job role, prompt structuring, and rule interpretation. These can be seen in Table 1.

The experiments used several LLMs: Gemma3 4B [12], Deepseek-r1-Distill-Qwen-8B [13], Llama3.2 3B Instruct [14], Mistral 7B Instruct[15] and Gemini-2.5-flash [16]. Gemini-2.5-flash was run via API using Google’s Gemini platform, while the other LLMs were run using Ollama<sup>1</sup>. These LLMs were selected due to their occurrence in recent literature. Our intention was not to produce a state-of-the-art result but to allow us to investigate the variance between models when provided with the same prompts.

The output lists were analysed to identify the similarity of features generated from different

Table 1: Prompts passed to the LLMs. Each prompt was instantiated with four different job roles substituted for the {JOB} placeholder: Biochemist, Biologist, Chemist, and No job provided. This resulted in a total of 20 prompt variants tested. Colours indicate corresponding aspects of different prompts.
<table><tr><td>Prompt Variant</td><td>Prompt Text</td></tr><tr><td>Prompt 1 – Baseline prompt</td><td>{You are a JoB. } Produce a list of fifteen items. These should be factors which can be used to predict the toxicity of a chemical. Do not produce any output except a numbered list.</td></tr><tr><td rowspan="2">Prompt 2 – Inclusion of SMILES</td><td>SMILES to predict the toxicity of a chemical.Do not produce any output except a numbered list.</td></tr><tr><td>{You are a JOB.} You are interested in toxicity and drug discovery. You want to know what features</td></tr><tr><td rowspan="2">Prompt 3 – Addition of contextual information</td><td>would be of interest when investigating drug toxicity. Produce a list of fifteen items. These should be factors</td></tr><tr><td>which can be used to predict the toxicity of a chemical. Do not produce any output except a numbered list.</td></tr><tr><td rowspan="3">Prompt 4 – Combined SMILES features and contextual information</td><td>{You are a JoB. }You are interested in toxicity of drugs. You want to know what features would be of</td></tr><tr><td>interest when investigating drug toxicity.Produce a list of fifteen items.These should be factors which can</td></tr><tr><td>be calculated from SMILESto predict the toxicity of a chemical.1 Do not produce any output except a numbered list.</td></tr><tr><td rowspan="4">Prompt 5 – Paraphrased prompt with role emphasis</td><td>{You are a JOB.} investigating drug toxicities. Which features would you look at to determine toxicity</td></tr><tr><td></td></tr><tr><td>of a chemical when only provided with a SMILES. Produce a list of fifteen items. These items should be</td></tr><tr><td>used to predict chemical toxicity.Do not produce any output except a numbered list.</td></tr></table>

LLMs and prompts. The feature names were standardised by Gemini 2.5 flash; this was necessary due to differing names being produced for the same property i.e. Molecular Weight, MolW, Mol Weight. The most recurring features can be seen in Figure 1. LogP appeared in every feature list, this is expected as it is inversely correlated to water solubility, and can commonly indicate toxicity. Molecular weight and topological surface area were also prevalent in feature lists. There are only 7 features which occur in more than 25% of the lists, showing a large degree of variance in most of the features identified by the LLMs.

## 2.2 Creation of Datasets

To study the impact of LLMs, job role, prompt structure and inherent randomness on prediction accuracy, 18 datasets were generated using a subset of the previously obtained feature lists. The LLM was provided with 2,294 SMILES from a PubChem toxicity dataset [11]. The dataset comprises 789 toxic and 1,505 non-toxic compounds, reflecting moderate class imbalance. The model also received a feature list from section 2.1 and a prompt with instructions. Llama3.2 was selected due to its success in complying with strict formatting during preliminary experiments. Based on preliminary experiments, the below prompt was used for dataset construction.

From the given SMILES string, compute all chemical features listed below.

Follow the exact feature order provided.

Output exactly 15 values, separated by commas only, no spaces.

Numeric values only. Binary features encoded as 1 or 0.

Output NA if a value cannot be computed.

If the SMILES is invalid, output INVALID SMILES.

Respond with the output line only.

To check the accuracy of the LLM-generated values, we programmed a chemoinformatics pipeline to calculate the same features independently using RDKit. The pipeline used the same SMILES dataset and feature definitions, and produced results in the same format for comparison. Features that could not be calculated were either approximated using related descriptors or marked as missing. All calculations were deterministic. SMILES that could not be parsed were excluded from feature calculations but still recorded in the output.

Identical prompts were asked to the same LLM multiple times, to measure the impact of random variation (1aL<sub>1</sub>, 1aL<sub>2</sub> and 1aL<sub>3</sub>). We investigate using different LLMs, while keeping the prompt and role the same (3aD, 3aM, 3aL, 3aG and 3aF). Job roles were investigated by giving Llama3.2 the same prompt variant with differing job roles (5aL, 5bL, 5cL and 5dL). The datasets 4cL Calc and 4cL LLM used the same generated feature lists and compare extracting the features by chemoinformatic code or by an LLM respectively. A breakdown of the dataset names can be seen in the caption of Figure 2.

![](images/3b9e9c6da90fa7ed1796de803faab7bbc7b341ac6795d8cee95cff1183d20a58.jpg)  
Figure 1: Most frequently identified chemical features across all prompt–LLM combinations.

## 2.3 Machine Learning Models and Evaluation

The generated datasets were cleaned by removing uncomputable LLM-generated features and rows containing INVALID or NaN values. Ground truth toxicity was then added as a new column. The datasets were used to train several well-known ML models: Random Forest, Decision Tree, Neural Network (multilayer perceptron), Support Vector Machine, Na¨ıve Bayes, and Extreme Gradient Boosting. StandardScaler, SMOTE, and GridSearchCV were used for scaling, data balancing, and parameter tuning, respectively.

Figure. 2 shows that features extracted using the chemoinformatics pipeline outperform those generated by the LLM, as measured by AUC. Performance also varied across datasets when only the inherent randomness in LLM feature generation changed, indicating variability in the generated features. In contrast, changing the LLM had little impact, with similar AUC values across models. Performance also differed between job variants, with no single variant consistently performing best. All evaluation metrics showed similar trends. Permutation feature importance was calculated by disrupting a feature’s relationship with the target and measuring the resulting drop in model performance. Feature importance for LogP and molecular weight across all models and datasets is shown in Figure. 3, illustrating the variation in the two most common features.

## 3 Conclusions

This study investigated the impact of prompt engineering on drug toxicity prediction. It was found that simple fine-tuning prompts did not have large effects on LLM output. The study found that random variation in LLM response seemed to cause similar variance to changing the prompt text or LLM used. Feature extraction by LLM was shown to be less effective than chemoinformatic extraction, with the latter showing increased performance. Although finetuning prompts may not be a productive approach in this instance to increase prediction quality, future work should investigate variations of the process, such as using representations other than SMILES, and more complex feature engineering methods such as retrieval-augmented generation. Significance testing may also provide additional insight into observed variations.

Proceedings of CIBB 2026  
![](images/27fb7713dd40cf6b8ef34769fd84a0525efe9c1d6ed23bc3cb038249abbd8ca3.jpg)  
Figure 2: Heatmap of AUC values for the top 10 models across all datasets. Each cell shows the mean AUC of a model on a specific dataset. Warmer colours indicate higher AUC, highlighting models that perform consistently well across datasets. Dataset labels on the x-axis are a breakdown of the prompt and LLMs used. xyz - where x is the prompt used 1-5 as seen in Table 1, y is the job variant; a=Biochemist, b=Biologist, c=Chemist, d=no role provided. z is the LLM used; D=Deepseek-v2, M=Mistral, L=Llama3.2, G=Gemma3 and F=Gemini-2.5-flash. Datasets which have xyzw when w is a number reflect repeated LLM sampling under inherent stochasticity in the generation process. If w = Calc this is where chemioinformatics code has been used to generate the dataset, when w = LLM this is the LLM-generated variant of the dataset.

## Availability of data and software code

The implementation is available at the following URL: <sup>2</sup>.

## References

[1] Medicines & Healthcare products Regulatory Agency. Current MHRA fees. https://www.gov.uk/ government/publications/mhra-fees/current-mhra-fees, 2026. Updated 12 January 2026, accessed 3 February 2026.

[2] Association of the British Pharmaceutical Industry. UK industry clinical trials: Translating actions into impact. https://www.abpi.org.uk/publications/uk-industry-clinical-trialstranslating-actions-into-impact/, December 2025. Published 02 December 2025, accessed 03 February 2026.

[3] Duxin Sun, Wei Gao, Hongxiang Hu, and Simon Zhou. Why 90% of clinical drug development fails and how to improve it? Acta Pharmaceutica Sinica B, 12(7):3049–3062, 2022.

[4] Kevin H. Denny and C. W. Stewart. Acute, subacute, subchronic, and chronic general toxicity testing for preclinical drug development. In Ali S. Faqi, editor, A Comprehensive Guide to Toxicology in Nonclinical Drug Development, pages 149–171. Academic Press, 3 edition, 2024.

[5] Ruiqiu Zhang, Hairuo Wen, Zhi Lin, Bo Li, and Xiaobing Zhou. Artificial intelligence-driven drug toxicity prediction: Advances, challenges, and future directions. Toxics, 13(7):525, 2025.

[6] Bosheng Song, Xiaowen Li, Xiuxiu Chao, Li Wang, Yiping Liu, Zhen Xia, Dongsheng Cao, and Xiangzheng Fu. Advancements in large language models (LLMs): Empowering drug discovery. Wiley Interdisciplinary Reviews: Computational Molecular Science, 15(6), 2025.

![](images/4739bdaa6d1b99e706d58de3294ff0e448c5967733a70364a4071713ea59dcf3.jpg)  
Figure 3: Permutation feature importance of Molecular Weight and LogP across all datasets and models. Bars represent mean decrease in model performance when a feature is permuted, calculated across multiple models. Error bars show standard deviation of feature importance between models. A breakdown in dataset titles can be seen in the caption of Figure. 2.

[7] Yueh Hsi Chung, Chun Wei Tung, and Yung Chun Chang. Exploring the efficacy of large language models in predicting chemical toxicity. In Hamido Fujita, Yutaka Watanobe, Moonis Ali, and Yinglin Wang, editors, Advances and Trends in Artificial Intelligence. Theory and Applications, volume 15706 of Lecture Notes in Computer Science, pages 407–418. Springer, 2025.

[8] Hengzheng Yang, Jian Xiu, Weiqi Yan, Kaifeng Liu, Huizi Cui, Zhibang Wang, Qizheng He, Yilin Gao, and Weiwei Han. Large language models as tools for molecular toxicity prediction: AI insights into cardiotoxicity. J. Chem. Inf. Model., 65(5):2268–2282, 2025.

[9] Yizhen Zheng, Huan Yee Koh, Jiaxin Ju, Anh T N Nguyen, Lauren T May, Geoffrey I Webb, and Shirui Pan. Large language models for scientific discovery in molecular property prediction. Nat. Mach. Intell., 7(3):437–447, 2025.

[10] Thomas F. Heston and Charya Khun. Prompt engineering in medical education. International Medical Education, 2(3):198–205, 2023.

[11] PubChem Database. Pubchem bioassay aid 489025: A cell based secondary assay to explore cytotoxicity in THP-1 cells of compounds that modulate non-replicating, drug-tolerant mycobacterium tuberculosis. PubChem, National Center for Biotechnology Information, U.S. National Library of Medicine, 2025.

[12] Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Riviere, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on Gemini\` research and technology. https://arxiv.org/abs/2403.08295, 2024.

[13] Aixin Liu, Bei Feng, and Bing Xue. Deepseek-r1: Incentivizing reasoning capability in LLMs via reinforcement learning. https://arxiv.org/abs/2501.12948, 2025. arXiv:2501.12948.

[14] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix,´ Baptiste Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard\` Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023.

[15] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lelio Renard Lavaud,´ Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothee Lacroix, and´ William El Sayed. Mistral 7b, 2023.

[16] Google DeepMind Gemini Team. Gemini: A Family of Highly Capable Multimodal Models. https: //deepmind.google/gemini/gemini\_1\_report.pdf, 2023.