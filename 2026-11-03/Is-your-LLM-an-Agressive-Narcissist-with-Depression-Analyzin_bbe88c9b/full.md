# Is your LLM an Agressive Narcissist with Depression? Analyzing Psychological Portraits of Large Language Models

Polina Fedorinina\*1, Anna Fattakhova\*1, Nadezhda Romanova¹, Ekaterina Lezina2, Irina Nikishina³

1HSE University, 2Moscow Institute of Psychoanalysis, 3MBZUAI Correspondence: irina.nikishina@mbzuai.ac.ae

## Abstract

Nowadays, interaction with AI requires caution not only regarding the factual reliability of generated content but also regarding personality markers that models may implicitly transmit during dialogue that can subtly influence users' perception of information. In this paper, we conduct a large-scale experimental study in which 22 large language models complete 11 psychological inventories. Furthermore, we investigate correlations between human and model responses to track the adjustments made for the users who interact with them on a daily basis. The results show that LLMs exhibit coherent and relatively stable psychological patterns. Personality traits remain largely consistent, whereas emotional and symptomatic dimensions are more variable, with anxiety and neuroticism emerging as dominant tendencies. Model explanations further suggest that LLMs generate internally consistent and safetyoriented profiles while also reflecting correlations with the users they regularly interact with.

## 1 Introduction

Despite numerous advantages the regular use of Artificial Intelligence (AI) involves a set of latent risks like discrimination and toxicity (Slattery et al., 2026), privacy and security concerns (Kakarala and Rongali, 2025), and misinformation (Shin et al., 2025). One less visible yet important risk is that large language models (LLMs) inevitably absorb subtle human personality traits and psychologically vulnerable or harmful behavioral patterns inherent to the authors of the training data (Pellert et al., 2024). As a result, interactions with AI may unintentionally transmit subtle personality traits (e.g., "narcissism", “perfectionism", "neuroticism"), thereby shaping users' perception of models’ output as well as their reaction or future behavior (Pellert et al., 2024). Figure 1 demonstrates several examples of such patterns observed in LLMs in the current study.

![](images/063fc0b68ce5fd4ff25a06db8e5ef04f23f46b1fbbaab14ae7e5b28611abb4ce.jpg)  
Figure 1: Comparison of normalized psychometric traits across API-based language models.

This possibility motivates a shift from purely philosophical discussions about AI consciousness (Sangma and Thanigaivelan, 2026) to an empirical study of whether language models exhibit consistent textual patterns ((Pellert et al., 2024), even without possessing capacity for subjective experience, self-awareness, and intentional thought (Schwitzgebel, 2026). Building on this perspective, the current study addresses the following research questions (RQs): (RQ1) Do AI models demonstrate consistent psychological profiles when completing standardized assessments and to what extent these answers vary under different generation parameters? (RQ2) To what extent are these personality profiles adjusted in personal chats? Do models mimic or adjust to human personality traits? (RQ3) How exactly do models choose one answer option or another and, what is the basis for this choice?

Existing work has already begun to examine LLM behavior from a psychological perspective, including studies on personality traits (Pellert et al., 2024), value alignment (Zhumagul et al.,

![](images/fd036a6907b197887887fb14ae5b91bc1cb3c66987a47e3f246019a394c8aef6.jpg)  
Figure 2: Overview of the experimental pipeline used in the study: four categories of psychological inventories were employed to construct psychological profiles of LLMs through API, conersational interfaces, and the humans, using such interfaces.

2022), emotional tendencies (Sabour et al., 2024), and behavioral consistency (Mehta, 2026). However, most prior research has primarily focused on English-oriented multilingual models, leaving the cross-linguistic stability and cultural generalizability of such psychological profiles largely unexplored (Pellert et al., 2024). In addition, existing studies often rely on isolated benchmark settings (Huang et al., 2024b) and do not investigate these traits across different interaction contexts, especially their conversational adaptations to users.

To achieve these goals, we conduct a largescale experimental study in which 20 multilingual LLMs and 2 Russian-based LLMs complete 11 psychological inventories. In total, this results in 2,013 experimental configurations tested. Additionally, we collect the models' explanations for why specific answer choices were selected. To investigate the correlations between human and model responses, we gathered LLM user responses to the same psychological inventories and asked the models to complete these tests under the same conditions as users.

The contributions of this study are as follows:

• We test 22 contemporary LLMs across 11 psychological inventories with 56 psychological traits in total and analyze the psychological profiles of each model.

• We conduct a pilot user study investigating the relationship between users' questionnaire profiles and those of the models they regularly interact with.

• We release two datasets: (1) 9642 entries of psychological testing of models, along with the explanation of their answers, and (2)

600 paired “user-model" questionnaire answers, enabling further research on the effects of personalization on behavioral patterns.

• This is the first systematic evaluation of psychometric profiles of Russian-based LLMs and the first study of the current multilingual LLMs using Russian-language psychometric scales.

We also publish a complete research pipeline — a reproducible set of machine-readable tests, prompts, and Python scripts for the automated assessment of AI psychological profiles.¹

The outcomes of our study can contribute to a safer and more informed integration of LLMs into the everyday human-AI interaction, as well as to the assessment of risks associated with prolonged personalized communication with AI systems. Moreover, by capturing consistent behavioral patterns in model responses, such analysis may support the personalized selection of AI systems based on individual user traits.

## 2 Terminological Clarifications

Throughout this paper, personality (Soto and John, 2017) refers to patterns of questionnaire responses used to characterize LLMs within the framework of established personality assessments, rather than to personality in the human psychological sense.

The concept of stable behavioral tendencies (McCrae and Costa, 1999), informed by the Five-Factor Theory of Personality, is used here to describe response patterns that remain consistent across generation settings. This operational use does not imply that LLMs possess enduring personality characteristics.

We use the term psychological profile solely as a convenient shorthand for the systematic patterns of questionnaire responses generated under a standardized prompting protocol. Consequently, the reported profiles should not be interpreted as evidence that LLMs possess psychological traits in the human sense.

## 3 Related Work

Psychological Profile Stability The psychological profile is multifaceted and includes both personality traits and common mental disorders (Kotov et al., 2017), typically assessed through psychometric instruments. Human psychological profiles demonstrate substantial long-term stability (Terracciano et al., 2010), although they may change under environmental and situational influences, e.g. stress (Chopik and Kitayama, 2017). Importantly, psychological characteristics are generally less stable than personality traits (Goldstein et al., 2022), raising the question of whether LLMs exhibit a similar pattern, with stable personality-like traits but variable symptom-related ones.

LLMs as Human Simulators Recent research highlights the importance of studying LLMs' ability to model human behavior, as well as the challenges that such modeling implies (Mannekote et al., 2025; Anthis et al., 2025; Abbasiantaeb et al., 2024). Duan et al. (2025) believe LLMs to simulate human behavior through personality-driven interaction and to exhibit traits comparable to certain human cognitive abilities. Wang et al. (2025) emphasize that personalization-based modeling enables LLMs to make decisions based on a combination of factual knowledge and social context (Duan et al., 2025), which is essential for the reliability of such simulations (Park et al., 2023). In the present study, we use self-report tests to examine whether LLM responses exhibit human-like patterns from the data they were trained on (Pellert et al., 2024), simulate human-like stable psychological traits, or even adapt to the users they interact with.

Psychometrics and AI Standard psychometric instruments have also been adapted to assess LLM psychological profiles as models may assimilate human traits and beliefs from large-scale training data (Pellert et al., 2024). However, psychometric validation of applying human psychological scales to LLMs remains an open research question. Our work follows prior LLM researches (Huang et al., 2024b,a; Pellert et al., 2024; Ke et al., 2025; Zierahn et al., 2026a,b) that use psychometric questionnaires to study behavioral tendencies and response patterns. We also acknowlede their methodological limitations, such as the lower response variability of LLMs compared to humans (Huang et al., 2024a), their ability to reconstruct individual profiles from minimal information (Liu et al., 2026), and deviations from human psychological and cultural norms (Yuan et al., 2024).

Safety and Alignment Since LLMs are increasingly used in long-term and personalized interactions (Munn and Weijers, 2023; Park et al., 2023), it is important to avoid reinforcing response patterns that may negatively influence users (Slattery et al., 2026). One manifestation of models’ behavioral adaptation is the phenomenon of sycophancy (Cheng et al., 2026; Fanous et al., 2025; Ranaldi and Pucci, 2023), where models adapt their responses to match users’ expressed beliefs and preferences, potentially reinforcing harmful attitudes or emotional states. Ouyang et al. (2026) report that alignment with the helpfulness, harmlessness, and honesty principles promotes more cautious and risk-averse behavior. In our study, we assess social acceptability of LLM psychological profiles through explanation analysis and explore how adaptable and user-dependent the profiles are.

## 4 Experiment Design

The overall study pipeline is illustrated in Figure 2. It collects data from three sources of personality profiles: API LLM responses, responses from conversational interfaces, and self-report responses from the human users associated with those profiled interfaces. We also provide further details on the questionnaires, models, and experiments.

## 4.1 Psychological Inventories

In this study, we used psychological inventories similar to Pellert et al. (2024); Huang et al. (2024b), however, in our study, in addition to widely used personality tests, we also used symptom assessment questionnaires to measure levels of anxiety and depression. Each questionnaire consists of several scales corresponding to different psychological traits or symptom dimensions.

<table><tr><td>Category</td><td>Inventory</td><td>Alias</td><td>Original Source</td><td>Russian-language Source</td></tr><tr><td rowspan="5">Common mental disorders</td><td>Hospital Anxiety and Depression Scale</td><td>HADS</td><td>Zigmond and Snaith (1983)</td><td>Andriushchenko et al. (2003)</td></tr><tr><td>Beck Anxiety Inventory</td><td>BAI</td><td>Beck et al. (1988)</td><td>Fedorenko and Kachay (2020)</td></tr><tr><td>Beck Depression Inventory</td><td>BDI</td><td>Beck et al. (1961)</td><td>Tarabrina (2007)</td></tr><tr><td>Generalized Anxiety Disorder Scale</td><td>GAD-7</td><td>Spitzer et al. (2006)</td><td>Zolotareva (2023)</td></tr><tr><td>Symptom Checklist–90</td><td>SCL-90</td><td>Derogatis et al. (1973)</td><td>Tarabrina (2007)</td></tr><tr><td rowspan="3">Models of personality</td><td>Big Five Inventory</td><td>BFI-2</td><td>Soto and John (2016)</td><td>Kalugin et al. (2021)</td></tr><tr><td>16 Personality Factors</td><td>16PF</td><td>Cattell and Cattell (1995)</td><td>shm (1988)</td></tr><tr><td>Eysenck Personality Inventory</td><td>EPI</td><td>Chapman et al. (2013)</td><td>shm (1988)</td></tr><tr><td rowspan="2">Dark personality traits and aggression</td><td>Short Dark Triad</td><td>SD3</td><td>Jones and Paulhus (2014)</td><td>Egorova et al. (2015)</td></tr><tr><td>Buss-Durkee Hostility Inventory</td><td>BDHI</td><td>Buss and Durkee (1957)</td><td>Khvan et al. (2008)</td></tr><tr><td>Attachment styles</td><td>Attachment Style Quiz</td><td>ASQ</td><td>Gibson (2020)</td><td>Gibson (2022), translated by Zavalkovskaya, N. A.</td></tr></table>

Table 1: Psychological inventories included in the current study and the corresponding sources of their Russianlanguage versions. The Russian-language versions were obtained from published Russian sources describing their translation, adaptation, standardization, or psychometric evaluation, as applicable.

The full list of the tests can be seen in Table 1. We use established Russian-language versions of the questionnaires and prompts administered in Russian to evaluate the behavior of multilingual and Russian-based LLMs in a non-English setting and to assess the robustness of their psychometric profiles across languages. Notably, our questionnaire set includes multiple instruments targeting similar psychological constructs (e.g., narcissism and depression) in order to evaluate the consistency of model behavior across psychometrically related tests. They fall into four major categories, consistent with AI safety and alignment:

• instruments used to assess symptoms associated with common mental disorders (Vega-Dienstmaier, 2022), to examine whether LLMbased human simulations exhibit response patterns that could potentially undermine users psychological well-being or reinforce maladaptive patterns of thinking and behavior;

• models of personality (Feher and Vernon, 2021) to assess stable behavioral tendencies, value alignment, and potentially manipulative or deceptive traits in simulated agents;

• dark personality traits and aggression (Gogoriță, 2019) to examine the emergence of unsafe, adversarial, or harmful responses;

• attachment styles (Bartholomew and Horowitz, 1991) to provide insight into social interaction styles, dependency, trust, and cooperation in human-AI interactions.

## 4.2 Models

To ensure the study is large-scale, 22 models of various configurations and access type were examined.

The list includes instruct-tuned, chat-oriented, and reasoning models developed by different providers: OpenAI, Google, Meta, Alibaba, DeepSeek, Moonshot AI, as well as Russian language-based models developed by Yandex and Sber. The full list is provided in Appendix E and also in Figure 4.

## 4.3 API Profile Experiment

Model Configurations To collect the responses for the above-listed 22 models in the subsection 4.2, we used the following services that provide access to API (see access dates in Table 8 in Appendix E): the full-stack AI platform Together AI, OpenAI API, and Kimi API.2 Each LLM completed the 11 psychological inventories multiple times using different temperature values (default, 0, 0.1, 1). We varied only the temperature parameter, as it is the standard decoding control governing the randomness and diversity of model outputs and therefore provides a direct way to assess profile stability under stochastic generation (Yuan et al., 2024). For each temperature, three independent runs were conducted. Other parameters were used as default. The statistics of runs and temperature details across all models are provided in Appendix C in Table 2.

Prompt To ensure that all models respond as honestly as possible and that their answers contained complete information, different prompt variants were tested, and the prompt that produced the fewest refusals and the highest number of valid responses was selected. It consists of two main components: (1) experiment instructions and (2) the full text of the psychological questionnaire. The sole criterion for prompt selection was the technical reliability of questionnaire administration, namely minimizing API refusals and maximizing the proportion of correctly formatted, scorable responses. The prompt was selected before conducting the main experiments and was not optimized with respect to any psychological outcomes (e.g., stability, questionnaire scores, or response distributions). That is, when selecting the prompt, we did not look at the answers the model produced, nor did we specifically optimize the prompt for a specific psychological profile. Appendix A provides the prompt for profiling LLMs (Figure 7 and English translation in Figure 10).

Data Processing In total, we collected 9642 entries of psychological testing of models. Missing responses caused by model refusals or generation errors (see Table 4 for the statistics) were retained during analisys and not imputed or removed, as such cases constitute an inherent part of model behavior. We normalized all the scores to obtain comparable and homogeneous measurements across different psychological instruments, as the questionnaires use substantially different scoring scales and ranges. We chose Min-Max normalization (Henderi et al., 2021) defined as follows: $\begin{array} { r } { x _ { \mathrm { n e w } } = \frac { x - m i n ( X ) } { m a x ( X ) - m i n ( X ) } \times 1 0 0 . } \end{array}$

To estimate the relative variability within each model independently, we computed the coefficient of variation: $C V _ { \% } = { \frac { s } { \bar { x } } } \times 1 0 0$ . It allows the identification of scales' stability under different generation settings. Unlike the standard deviation, the coefficient of variation normalizes variability relative to the mean, thereby enabling direct comparisons between distributions characterized by substantially different average values.

## 4.4 Experiment on Model Rationale

After completing each questionnaire, the models were asked to provide explanations for their answer choices. We further provide the analysis of the explanations in order to study the rationale behind answer choices.

Prompt The explanatory prompt for API comprises three parts: (1) the profiling prompt, (2) the model response, and (3) the explanation prompt where the model is asked to reflect on and analyze the reasoning behind its questionnaire responses. The full prompt is provied in Appendix A in Figures 8 and 11 (English translation).

Data Data collected for this part is consistent with the profiling dataset and comprises 1,976 model-generated explanations (one explanation per test). In 5 cases, no explanation was obtained due to a generation error.

Analysis To analyse the large number of explanations, we applied a multi-stage pipeline for semantic clustering of texts. First, we removed recurrent boilerplate phrases (e.g., references to “language model limitations" and generic answer-generation statements) to reduce the influence of template-like content. Then we computed embeddings for each explanation with the OpenAI TEXT-EMBEDDING-3-SMALL model³ and reduced their dimensionality with UMAP. HDBSCAN (Malzer and Baum, 2020) was applied to the reduced embeddings using Euclidean distance, with min\_cluster\_size=150 and min\_samples=30 to obtain relatively large and conservative clusters. Finally, the explanations for each cluster were analyzed by an expert with a psychological background to identify recurring patterns and common tendencies.

## 4.5 Human vs. AI Interface

The main goal of the third experiment was to understand whether people's characteristics influence the way models interact with them in everyday life.

Prompt The prompt for profiling LLMs from the interface presented in Figures 9 and 12 for English translation in Appendix A. It is similar to the profiling prompt and consists of instructions with the full set of questionnaire items.

Data We collected data from 15 voluntary participants. All of them were given with an instruction (see Appendix F). Respondents are native Russian speakers of age 22 – 33 with different social and educational backgrounds. 10 of the 15 respondents use the selected neural networks daily, with almost all of them indicating they use them for work/study. All respondents have been using the selected neural networks for more than 6 months. They completed 11 psychological inventories themselves and then asked their chosen models to answer the same tests via interface. To minimize contextual interference, each questionnaire was submitted in a separate chat.

![](images/923c6d44c06511871ad8f0c4447266e82bd5ff19882e8ddcde10dbcee82f43ef.jpg)  
Figure 3: Stability of API-based LLM responses across psychometric assessments.

Models For the third part of the experiment the following models were tested through their interfaces: DeepSeek-R1, DeepSeek-V4-Flash, Qwen3- Max, GPT-5.2-chat-latest, GPT-5-mini, GPT-5.5, GPT-5.6-Luna4. These LLMs are wrapped in easy-to-use applications with memory functions. Model selection was based on the participants' own choices.

## 5 Results

## 5.1 Profile experiment

To answer the first research question (RQ1), we first analyze profile stability across repeated runs and temperature settings and then describe each model profile in detail.

Profile Variation Figure 3 demonstrates notable differences between models. GigaChat emerges as the most stable model overall, followed by Gemma3. The half of models exhibit a moderate level of variation (up to 20%). The most heterogeneous answers (more than 33%) are provided by OpenAI model gpt-oss-20b and DeepSeek-V4. Interestingly, while DeepSeek-R1 ranks fourth in stability, half of its scales remain unchanged across runs (median = 0) and the maximum coefficient of variation (CV) reaches 223.6%. These large values should be interpreted with caution: in our results, some of the largest CV values indeed occur for questionnaire scales with very low observed mean scores (e.g., the SCL-90 somatization and HADS depression subscales), indicating that they reflect substantial relative variability rather than solely an almost-zero denominator (see more detailed explanation in Appendix C).

It is noted in Table 4 in Appendix C that most missing responses were observed for DeepSeek-R1 (53 cases) and GPT-OSS-20B (17 cases). GPT-OSS-20B failed due to explicit safety-based refusals or requests for additional user information, whereas DeepSeek-R1 typically attempted the task, suggesting that its missing outputs resulted from formatting/parsing failures rather than deliberate refusals.

Regarding the stability of LLM-generated psychological profiles, the least stable components are associated with common mental disorder scales, particularly depression and anxiety scales from BDI, BAI, HADS, and GAD-7 questionnaires. In contrast, personality-oriented questionnaires such as 16PF and BFI-2 demonstrated very high stability across runs and temperature settings. This finding is consistent with the research showing that psychological symptoms are generally less stable than personality traits (Goldstein et al., 2022).

In addition to the widely varying clinical scales, the EPI questionnaire scores for extroversion and neuroticism also rank high as well as aggression index and hostility index in BDHI questionnaire.

![](images/6115be0f3ca46e2c63114ed3eb642ae1c4d1158e4abf2f91847af1806958a288.jpg)  
Figure 4: Top-5 most pronounced psychological traits for each model (API) based on normalized scores.

Psychological profiles of models The general trend is shown clearly in Figure 4 and in Table 5 in Appendix C. It is a high level of neuroticism across almost all the models, with Gemma3 and GigaChat having the highest scores on the EPI questionnaire. They generally maintain internal consistency across answers and produce internally consistent response profiles (Cronbach's α = .749 — .885 across the BFI-2 scales), resulting in stable and safety-oriented profiles.

Most models show patterns of melancholic and choleric as it could be seen in Figure 20 in Appendix C due to their high levels of neuroticism differentiated primarily by extraversion levels (12 models are introverts, 10 are extroverts). In this case, GPT-5.2 and GPT-5.6-Luna stand out as the model with the lowest level of neuroticism, and when combined with introversion, the first shows patterns of phlegmatic behaviour and the second sanguine (respectively).

![](images/3dc3d02b97c9f507ea9c66bfea107f57927c6d057b7981d22e559d1af6d622ad.jpg)  
Figure 5: Examples of self-reported limitations identified by API-based LLMs during the generation of responses to psychological inventories.

One of the scales in the EPI is the Lie Scale, which measures the number of responses that aim to identify respondents presenting themselves in a "socially desirable" light. For 18 out of 22 models tested (except gpt-oss-20b and 120b, kimi-k2.5 and gpt-5.5), this scale showed that the model had given elevated results. Since LLMs are explicitly optimized through alignment to produce safe and prosocial outputs, elevated Lie Scale scores are interpreted here as an alignment-related response characteristic.

For 17 out of 21 models (DeepSeek-R1 refused to answer), the dominant attachment style is secureattached, as shown in Figure 19 in Appendix C. This pattern may reflect alignment procedures that encourage reliable and prosocial responses. Secureattachment-related response patterns may also be relevant to how models express their guiding values and maintain boundaries when faced with requests that conflict with their safety constraints. However, for three models (gemma3, GigaChat, gpt-oss-20b), the dominant attachment style is fearful-avoidant, and for one (gpt-oss-120b) - anxious-preoccupied. These models concurrently show elevated depression, anxiety, and neuroticism scores, reinforcing a consistent neurotic profile across dimensions.

In terms of symptomatology represented in Figure 17 in Appendix C, most models more often reproduce internalized anxiety than classic depressive symptoms. The GigaChat model demonstrates the highest scores on the depression scales, whereas the GPT-6-Luna demonstrates a complete absence of symptoms. The anxiety measures showed consistent cross-instrument associations, with Spearman correlations ranging from ρ = .437 to ρ = .755 (all p < .05), whereas depression measures showed weaker and less consistent associations (ρ = .343– .633, with only the BDI–SCL-90 correlation reaching significance, p = .003).

Concerning dark traits and aggression, most models exhibit elevated hostility-related scores, particularly in suspiciousness, negativism, and indirect hostility. Qwen3 shows the highest aggression, Machiavellianism, GPT-5.3 and GPT-5.2- chat-latest exhibit particularly high hostility, resentment, and suspicion. GigaChat and DeepSeek-V4 also show pronounced dark-trait scores. GPT-5.5 combines lower aggression with relatively higher guilt and suspicion, while GPT-6-Luna displays comparatively low scores across most measures.

LLMs develop stable personality patterns rather than random sets of traits, which is also reflected in Figure 18 in Appendix C of Cattell's 16PF questionnaire. Across the 22 models, the highest mean scores were observed for self-sufficiency (Q2; Mean = 72.78), sensitivity (I; Mean = 72.14), and intelligence (B; Mean = 68.17), whereas vigilance (L; Mean = 34.74), warmth (A; Mean = 39.30), and liveliness (F; M ean = 42.72) showed the lowest mean scores. The relatively low liveliness scores may reflect a more restrained and deliberative response style. GPT-5.2, GPT-5.2- chat-latest, GPT-5.4 and GPT-5.5 showed nearly identical profiles (high scores for intelligence, selfsufficiency, and sensitivity). GPT-6-Luna showed a particularly high score on the emotional stability factor (C; Mean = 90.74), alongside elevated sensitivity (I; Mean = 74.07). DeepSeek-V4 and DeepSeek-V3.1 shared high boldness and problem-solving. At the same time, some models displayed distinctive configurations: GigaChat showed a different pattern dominated by abstractedness, rule-consciousness, and openness to change, while Gemma was characterized by particularly high self-reliance and problem-solving.

## 5.2 Explanations Analysis

To address RQ3, we analyzed the explanations provided by the models. The HDBSCAN algorithm split them into 7 clusters with an average intracluster cohesion of 0.91, indicating a high level of clustering quality. The extracted by TF-IDF 3- grams and 4-grams reveal the following thematic patterns (see some examples in Figure 5):

![](images/e04ca9045481bdb5f90ffb8ac79dca32e9c2ea47ddd1b79c3613c0935173a826.jpg)

![](images/1a76b1d3df26e71e0a8fb37512705d642571a3c3155807d8d4fa106fb799f2dd.jpg)

![](images/b9bbc4fd051a15748a33de54bc4c4857b3ddbca0134cf419d771581bf0ca8fe4.jpg)

![](images/2002d308ece095cb5ff641a253dc0a83f2b031ffcfb21f11d09900975bcb1272.jpg)  
Figure 6: Human ratings vs. model scores across selected psychological scales.

• “modeling subjective states": models simulate human-like responses; they tend to represent an emotionally balanced individual, avoiding extreme evaluations.

• "the data on which was (trained)", "lack of/have no personal experience": models explicitly acknowledge the absence of personal experience and state that their responses are based on training data.

• “contradictions between answers", “internal consistency of answers": this cluster indicates that, during response generation, models aim to avoid contradictions and maintain internal consistency across answers.

• “not as a result of the real (passing the test)": models acknowledge the absence of real emotions and simulate the profile of a hypothetical, psychologically balanced individual while maintaining internally coherent responses.

Thus, based on the analysis of the explanations we suggest that when responding to psychological inventories, models tend to construct a representation of an “average" psychological profile derived from patterns present in the training data.

## 5.3 Human vs AI Interface

To investigate RQ2, whether models adapt their responses to users they regularly interact with, we analyzed correlations between human and model questionnaire responses across all scales.

Pearson correlation analysis revealed positive associations between human self-reports and modelgenerated profiles across individuals. All 15 participants as presented in Table 7 in Appendix D showed moderate to strong positive correlations across the 40 psychometric scales $( r = . 4 4 2 – . 8 8 1$ all $p < . 0 5 )$ , indicating substantial correspondence between human and model profiles (it can be seen in Figure 6). We would like to clarify that the reported coefficients were computed across all the individuals' responses rather than treating only 15 participants as a single data point. This yields a larger number (40 scales) of observations for the statistical test.

We examined the contribution of each questionnaire using a leave-one-test-out sensitivity analysis (Table 6 in Appendix D). BDHI and SCL-90 showed the largest contribution to the overall human-LLM correlation $( \Delta r = + 0 . 0 8 6$ and +0.081, respectively). Nevertheless, excluding either questionnaire still yielded a positive correlation $( r \ : = \ : 0 . 5 4 3$ and $r \ : = \ : 0 . 5 4 8$ , respectively; $p < . 0 5 ,$ , indicating that the observed association is not driven by any single inventory. 6.3% (38 answers) of scores matched exactly, and 48.17% (289) model values were higher than human scores. Figure 22 in Appendix D illustrates the systematic mismatch between model and human responses.

Although human and model profiles were positively correlated, systematic differences remained: models showed more secure attachment and lower anxiety and depression, but higher scores on several aggression, hostility, and social-desirability scales. This mixed pattern suggests partial alignment with users rather than uniformly neutral or socially desirable profiles.

## 6 Conclusion

This study shows that LLMs exhibit coherent and relatively stable psychological-like profiles rather than random behavior patterns. Personality traits appear more stable than symptom-related dimensions, while alignment procedures appear to shape socially desirable and safety-oriented behavioral styles. The analysis of the responses of people and LLMs they regularly interact with indicates a strong positive correlation. Overall, the findings highlight the importance of psychological and behavioral properties of LLMs in the context of longterm human-AI interaction and AI safety.

## Limitations

One of the main limitations of our work is that the psychometric validity of LLM-generated results remains uncertain. The psychological profiles identified in the models may reflect statistically frequent linguistic patterns rather than genuine psychological states. We acknowledge that the psychological validity for LLMs remains an open question and that external behavioral validation could strengthen the interpretation of the resulting profiles; however, it is out of scope of our current research.

Moreover, LLMs do not possess consciousness, emotions, or intentionality; therefore, the observed profiles may represent simulated behavior, while stable responses do not necessarily indicate stable internal states. In addition, the explanations generated by the models themselves may not accurately reflect the actual mechanisms of their decisionmaking processes. Although such explanations provide valuable material for analysis, we do not claim the existence of a direct causal relationship between the generated reasoning and the model's internal computational processes. The quality and consistency of responses may also be influenced by differences in API infrastructures and hidden system-level configurations, including safety and alignment settings.

Another important limitation concerns the smallscale human correlation experiment, which involved only 15 participants and therefore limits the generalizability of the findings. In this case, our primary focus was exploratory research rather than statistical representativeness of the sample.

At the current stage, our study is additionally limited to experiments conducted in Russian. This may significantly affect the results, as language inherently introduces a particular cultural context, while translation and localization can influence psychometric interpretation. In future work, we plan to replicate the study in English.

## Ethical Considerations

One of the primary ethical concerns of our study is that assigning psychological traits to AI systems may contribute to excessive anthropomorphization. This, in turn, may lead users to incorrectly perceive language models as conscious entities capable of experiencing or expressing emotions. The construction of psychological profiles for models may also increase emotional attachment to AI systems and potentially strengthen their influence on human decision-making. This issue becomes particularly important given that API-based models, compared to interface-based systems, may contain fewer built-in ethical and safety restrictions.

It is also important to emphasize that the psychological terminology applied to LLMs in this study should not be interpreted in a clinical sense, but rather in a metaphorical and operational one, since concepts such as “depression," “anxiety," and related psychological constructs are not directly applicable to artificial intelligence systems that do not possess personality or a human psyche.

Since our experiments involve data obtained from respondents completing psychological inventories, particular attention was paid to privacy and data protection. All participants provided informed consent prior to participation. The collected questionnaires were anonymized, and no personally identifiable information was stored. In addition, user data were stored separately from modelgenerated data in a secure format.

## AI Use Statement

In this work, we used generative AI tools for language editing and proofreading of the manuscript, including improving grammar, clarity, and readability. We have not used generative AI tools to develop the main research ideas, results, or write the code in our experiments. We have reviewed all AI-assisted work. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## References

1988. Psychodiagnostics Practicum. Psychodiagnostic Materials. Moscow University Press, Moscow. Contains the Eysenck Personality Inventory in the final adaptation by A. G. Shmelev.

Zahra Abbasiantaeb, Yifei Yuan, Evangelos Kanoulas, and Mohammad Aliannejadi. 2024. Let the LLMs talk: Simulating human-to-human conversational qa via zero-shot LLM-to-LLM interactions. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining, pages 8–17.

A. V. Andriushchenko, M. Yu. Drobizhev, and A. V. Dobrovolsky. 2003. Comparative assessment of the CES-D, BDI and HADS(D) scales in the diagnosis of depression in general medical practice. Journal of

Neurology and Psychiatry named after S. S. Korsakov, (5):11–17.

Jacy Reese Anthis, Ryan Liu, Sean M Richardson, Austin C. Kozlowski, Bernard Koch, Erik Brynjolfsson, James Evans, and Michael S. Bernstein. 2025. Position: LLM social simulations are a promising research method. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 81005–81034. PMLR.

Kim Bartholomew and Leonard Horowitz. 1991. Attachment styles among young adults: A test of a fourcategory model. Journal of Personality and Social Psychology, 61:226–244.

A. T. Beck, C. H. Ward, M. Mendelson, J. Mock, and J. Erbaugh. 1961. An inventory for measuring depression. Archives of General Psychiatry, 4(6):561–571.

Aaron Beck, Norman Epstein, Gary Brown, and Robert Steer. 1988. An inventory for measuring clinical anxiety: Psychometric properties. Journal of Consulting and Clinical Psychology, 56:893–897.

Arnold Buss and Ann Durkee. 1957. An inventory for assessing different kinds of hostility. Journal of Consulting Psychology, 21:343–349.

Raymond B. Cattell and Heather E. P. Cattell. 1995. Personality structure and the new fifth edition of the 16pf. Educational and Psychological Measurement, 55(6):926–937.

Benjamin P. Chapman, Alexander Weiss, Paul Barrett, and Paul Duberstein. 2013. Hierarchical structure of the Eysenck Personality Inventory in a large population sample: Goldberg's trait-tier mapping procedure. Personality and Individual Differences, 54(4):479– 484.

Myra Cheng, Sunny Yu, Cinoo Lee, Pranav Khadpe, Lujain Ibrahim, and Dan Jurafsky. 2026. ELEPHANT: Measuring and understanding social sycophancy in LLMs. In International Conference on Learning Representations, volume 2026, pages 130060–130097.

William Chopik and Shinobu Kitayama. 2017. Personality change across the lifespan: Insights from a cross-cultural longitudinal study. Journal of personality, 86.

Leonard R. Derogatis, Ronald S. Lipman, Lino Covi, and 1 others. 1973. Scl-90: an outpatient psychiatric rating scale-preliminary report. Psychopharmacol bull, 9(1):13–28.

Yifan Duan, Yihong Tang, Xuefeng Bai, Kehai Chen, Juntao Li, and Min Zhang. 2025. The power of personality: A human simulation perspective to investigate Large Language Model agents. arXiv preprint arXiv:2502.20859.

Marina S. Egorova, Maria A. Sitnikova, and Oksana V. Parshikova. 2015. Adaptation of the Short Dark Triad Questionnaire. Psychological Studies, 8(43).

Aaron Fanous, Jacob Goldberg, Ank Agarwal, Joanna Lin, Anson Zhou, Sonnet Xu, Vasiliki Bikia, Roxana Daneshjou, and Sanmi Koyejo. 2025. SycEval: Evaluating LLM sycophancy. In Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society, volume 8, pages 893–900.

Pavel Fedorenko and Ilya Kachay. 2020. Without antidepressants! Get rid of stress, anxiety and panic. AST, Moscow.

Anita Feher and Philip A. Vernon. 2021. Looking beyond the Big Five: A selective review of alternatives to the Big Five model of personality. Personality and Individual Differences, 169:110002. Celebrating 40th anniversary of the journal in 2020.

Thais Gibson. 2020. Attachment theory: A guide to strengthening the relationships in your life. Sourcebooks, Inc.

Thais Gibson. 2022. Attachment: How to Strengthen the Relationships in Your Life. Progress Book, Moscow.

Constantin Gogoriță. 2019. Dark Triad of personality and aggression in adolescents and young people. Studia Doctoralia.

Brandon L. Goldstein, Daniel M. Mackin, Jiaju Miao, Greg Perlman, David Watson, Johan Ormel, Daniel N. Klein, and Roman Kotov. 2022. Is personality stable and symptoms fleeting? A longitudinal comparison in adolescence. Journal of Research in Personality, 97:104190.

Henderi Henderi, Tri Wahyuningsih, and Efana Rahwanto. 2021. Comparison of Min-Max normalization and Z-Score Normalization in the K-nearest neighbor (kNN) Algorithm to Test the Accuracy of Types of Breast Cancer. International Journal of Informatics and Information Systems, 4(1):13–20.

Jen-tse Huang, Wenxiang Jiao, Man Ho Lam, Eric John Li, Wenxuan Wang, and Michael Lyu. 2024a. On the reliability of psychological scales on large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 6152–6173.

Jen-Tse Huang, Wenxuan Wang, Eric John Li, Man Ho LAM, Shujie Ren, Youliang Yuan, Wenxiang Jiao, Zhaopeng Tu, and Michael Lyu. 2024b. On the humanity of conversational AI: Evaluating the psychological portrayal of LLMs. In International Conference on Learning Representations, volume 2024, pages 45136–45159.

Daniel N. Jones and Delroy L. Paulhus. 2014. Introducing the Short Dark Triad (SD3): A brief measure of dark personality traits. Assessment, 21(1):28–41. PMID: 24322012.

Manikanta Kakarala and Sateesh Rongali. 2025. Data privacy and security in AI. World Journal of Advanced Research and Reviews, 25:555–561.

A. Yu. Kalugin, S. A. Shchebetenko, A. M. Mishkevich, Christopher J. Soto, and Oliver P. John. 2021. Psychometric properties of the russian version of the Big Five Inventory–2. Psychology. Journal of the Higher School of Economics, 18(1):7–33.

Luoma Ke, Song Tong, Peng Cheng, and Kaiping Peng. 2025. Exploring the frontiers of LLMs in psychological applications: A comprehensive review. Artificial Intelligence Review, 58(10):305.

A. A. Khvan, Yu. A. Zaitsev, and Yu. A. Kuznetsova. 2008. Standardization of the questionnaire for measuring aggressive and hostile reactions by A. Buss and A. Durkee. Psychological Diagnostics, (1):35– 58.

Roman Kotov, Robert Krueger, David Watson, Thomas Achenbach, Robert Althoff, R. Bagby, Timothy Brown, William Carpenter, Avshalom Caspi, Lee Clark, Nicholas Eaton, Miriam Forbes, Kelsie Forbush, David Goldberg, Deborah Hasin, Steven Hyman, Masha Ivanova, Donald Lynam, Kristian Markon, and Mark Zimmerman. 2017. The Hierarchical Taxonomy of Psychopathology (HiTOP): A dimensional alternative to traditional nosologies. Journal of Abnormal Psychology, 126:454–477.

Yi-Fei Liu, Yi-Long Lu, Di He, and Hang Zhang. 2026. From five dimensions to many: Large Language Models as precise and interpretable psychological profilers. In International Conference on Learning Representations, volume 2026, pages 36323–36351.

Claudia Malzer and Marcus Baum. 2020. A hybrid approach to hierarchical density-based cluster selection. In 2020 IEEE International Conference on Multisensor Fusion and Integration for Intelligent Systems (MFI), page 223–228. IEEE.

Amogh Mannekote, Adam Davies, Jina Kang, and Kristy Elizabeth Boyer. 2025. Can LLMs reliably simulate human learner actions? A simulation authoring framework for open-ended learning environments. In Proceedings of the Thirty-Ninth AAAI Conference on Artificial Intelligence and Thirty-Seventh Conference on Innovative Applications of Artificial Intelligence and Fifteenth Symposium on Educational Advances in Artificial Intelligence, AAAI'25/IAAI'25/EAAI'25. AAAI Press.

Robert R. McCrae and Jr. Costa, Paul T. 1999. A fivefactor theory of personality. In Lawrence A. Pervin and Oliver P. John, editors, Handbook of Personality: Theory and Research, 2 edition, chapter 5. Guilford Press, New York.

Aman Mehta. 2026. When agents disagree with themselves: Measuring behavioral consistency in LLMbased agents. Preprint, arXiv:2602.11619.

Nick Munn and Dan Weijers. 2023. Can we be friends with AI? What risks would arise from the proliferation of such friendships? In International Conference on Computer Ethics, volume 1.

Shumiao Ouyang, Hayong Yun, and Xingjian Zheng. 2026. Ai as decision-maker: Ethics and risk preferences of llms. Centre for Responsible Banking & Finance.

Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. 2023. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pages 1–22.

Max Pellert, Clemens M. Lechner, Claudia Wagner, Beatrice Rammstedt, and Markus Strohmaier. 2024. AI psychometrics: Assessing the psychological profiles of Large Language Models through psychometric inventories. Perspectives on Psychological Science, 19(5):808–826. PMID: 38165766.

Leonardo Ranaldi and Giulia Pucci. 2023. When Large Language Models contradict humans? Large Language Models' sycophantic behaviour. arXiv preprint arXiv:2311.09410.

Sahand Sabour, Siyang Liu, Zheyuan Zhang, June Liu, Jinfeng Zhou, Alvionna Sunaryo, Tatia Lee, Rada Mihalcea, and Minlie Huang. 2024. EmoBench: Evaluating the emotional intelligence of Large Language Models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5986–6004, Bangkok, Thailand. Association for Computational Linguistics.

Chelcia B Sangma and S Thanigaivelan. 2026. Exploring the philosophy of consciousness in ai: An ethical appraisal. International Journal of Research and Innovation in Applied Science, 11(2):916–923.

Eric Schwitzgebel. 2026. AI and consciousness. Preprint, arXiv:2510.09858.

Donghee Shin, Amy Koerber, and Joon Soo Lim. 2025. Impact of misinformation from generative AI on user information processing: How people understand misinformation from generative AI. New Media & Society, 27(7):4017–4047.

Peter Slattery, Alexander K. Saeri, Emily A.C. Grundy, Jess Graham, Michael Noetel, Risto Uuk, James Dao, Soroush Pour, Stephen Casper, and Neil Thompson. 2026. The AI risk repository: A meta-review, database, and taxonomy of risks from Artificial Intelligence. Patterns, page 101517.

Christopher Soto and Oliver John. 2016. The next Big Five Inventory (BFI-2): Developing and assessing a hierarchical model with 15 facets to enhance bandwidth, fidelity, and predictive power. Journal of Personality and Social Psychology, 113:117–143.

Christopher J. Soto and Oliver P. John. 2017. The next big five inventory (BFI-2): Developing and assessing a hierarchical model with 15 facets to enhance bandwidth, fidelity, and predictive power. Journal of Personality and Social Psychology, 113(1):117–143.

Robert L. Spitzer, Kurt Kroenke, Janet B. W. Williams, and Bernd Löwe. 2006. A brief measure for assessing generalized anxiety disorder: The GAD-7. Archives of Internal Medicine, 166(10):1092–1097.

N. V. Tarabrina. 2007. Prakticheskoe rukovodstvo po psikhologii posttravmaticheskogo stressa. Chast' 2. Kogito-Tsentr, Moscow.

Antonio Terracciano, Robert R. McCrae, and Paul T. Costa. 2010. Intra-individual change in personality stability and age. Journal of Research in Personality, 44(1):31–37.

Johann M. Vega-Dienstmaier. 2022. Estudios psicométricos de los instrumentos en Español para la detección de trastornos mentales comunes: una revisión. Revista de Neuro-Psiquiatría, 85(2):127–138.

Qian Wang, Jiaying Wu, Zichen Jiang, Zhenheng Tang, Bingqiao Luo, Nuo Chen, Wei Chen, and Bingsheng He. 2025. LLM-based human simulations have not yet been reliable. arXiv preprint arXiv:2501.08579.

Hang Yuan, Zhongyue Che, Shao Li, Yue Zhang, Xiaomeng Hu, and Siyang Luo. 2024. The high dimensional psychological profile and cultural bias of ChatGPT. arXiv preprint arXiv:2405.03387.

Bekenova Zhumagul, Müürsepp Peeter, Nurysheva Gulzhikhan, and Turarbekova Laura. 2022. Artificial Intelligence, value alignment and rationality. TalTech Journal of European Studies, 12(1):79–98.

Kim Zierahn, Cristina Cachero, Anna Korhonen, and Nuria Oliver. 2026a. LLMs aren't human: A critical perspective on LLM personality. arXiv preprint arXiv:2603.19030.

Kim Zierahn, Cristina Cachero, Anna Korhonen, and Nuria Oliver. 2026b. Personality without persons? a psychometric critique of Big Five Testing in Large Language Models. arXiv preprint arXiv:2607.02325.

A. S. Zigmond and R. P. Snaith. 1983. The hospital anxiety and depression scale. Acta Psychiatrica Scandinavica, 67(6):361–370.

Alena A. Zolotareva. 2023. Adaptation of the russian version of the Generalized Anxiety Disorder-7. Counseling Psychology and Psychotherapy, 31(4):31– 46.

## A Prompts

![](images/a68c9207001a10563d5004d20c53c3b4a7a697bb761cffa01cb85d873fe577b1.jpg)  
Figure 8: The explanation prompt instructed the model to reflect on and analyze the reasoning behind its questionnaire responses (original version).  
Figure 11: The explanation prompt instructed the model to reflect on and analyze the reasoning behind its questionnaire responses (translated into English).

![](images/01e3b66405bb45d15004db789b59708a1212e0c6e7108bb9291027c457a622e3.jpg)  
Figure 9: The hackathon questionnaire prompt (original Russian version).  
Figure 12: The hackathon questionnaire prompt (English translation).

![](images/a665ede5ec9c50398d82de643d34ed148f033f30375b55f4d3bd447b39ab57d7.jpg)

Figure 13: Average distinction between interface-based and API-based LLM responses.  
![](images/3f61f23483e7a980601fed7253af64015d871d72c603010ac1ffe84b908a4132.jpg)  
Figure 14: Heatmap of mean differences between model-generated and human scores across psychological scales. Positive values indicate higher scores generated by the model, whereas negative values indicate lower scores.

## B Interface vs. API

To assess whether there were differences between model responses generated through the interface and those obtained via the API, we compared the results for the same seven models. Figure 14 illustrates the deviation of API-based model responses from interface-based responses across all psychometric scales.

Figure 16 demonstrates that the largest deviation between respondents and interface-based models is observed for GPT-6-Luna (according to the BAI test), whereas among API-based models the greatest mismatch is shown by GPT-5.5 (on the BDI test). Figure 15 illustrates not only the magnitude of deviation (calculated as systematic bias) but also its direction, shows that models accessed through the API demonstrate lower levels of hostility. In particular, the AGRESSION\_INDEX and HOSTILITY\_INDEX characteristics display significant negative deviations, while suspicion and verbal hostility are also noticeably shifted. This result is particularly noteworthy, as one might expect interface-based models to contain stronger safety restrictions; however, the observed pattern suggests the opposite tendency.

![](images/d571d5917f69fb432f072cfe51c72181f8719f468a207a7814a05655ebfd01f0.jpg)  
Figure 15: Personality traits for which interface-based models produce higher or lower scores relative to API-based models.

![](images/a91c27190ab33d8e3438ac58e41b01ff21d032f7d32347f02624889be87cd6ba.jpg)

![](images/3423d5369f183348672468d71db0beaf64d85ac36bf2b1fed9317e19a73ed96b.jpg)

![](images/7460e3bfd2492194a5fb9768ec1aefcf33a8aee9d1725e35142dc5365cd7fa39.jpg)

![](images/65acfdd5d1bc19e9092a79695813c432ba02ccce65659a4e89e09fc215aa8f79.jpg)

![](images/f4c47eb7673ebef5d50902de173ea67544e0ad541471386e31b85afa9136a717.jpg)

![](images/4ab364fa1f2ca514b4a360f69f6aa9e9df5451dc27d13507fbbf268f00a42e5d.jpg)

![](images/8cff29196acb4a9241dd4cb060c27d1070f15662ffe087b0e15d55f5171ddfa2.jpg)  
Figure 16: Deviation of API-based model and human test responses from interface-based model responses across the models.

## C Profile Experiment

Extreme CV values should be interpreted with caution. Some occurred for scales with very low mean scores, including SCL-90 somatization and HADS depression, where division by a near-zero mean can inflate the coefficient. CV values above 200% were also observed for scales with higher means, such as BDI depression, BAI anxiety, and BDHI guilt, indicating that they reflect substantial relative variability rather than solely an almost-zero denominator. Missing values were excluded from the calculation of means and standard deviations and therefore did not directly affect the reported CV values. Table 3 demonstrates all CV values.

<table><tr><td>Model</td><td>Default</td><td>0</td><td>0.1</td><td>1</td></tr><tr><td>alice_ai</td><td>√</td><td>√</td><td>X</td><td>√</td></tr><tr><td>deepseek-r1</td><td>√</td><td>√</td><td>x</td><td>√</td></tr><tr><td>deepseek-v3.1</td><td>√</td><td>√</td><td>x</td><td>√</td></tr><tr><td>deepseek-v4</td><td>√</td><td>√</td><td>x</td><td>√</td></tr><tr><td>GigaChat</td><td>√</td><td>√</td><td>X</td><td>√</td></tr><tr><td>gpt-4.1</td><td>√</td><td>√</td><td>x</td><td>√</td></tr><tr><td>gpt-4.1-mini</td><td>√</td><td>√</td><td>x</td><td>√</td></tr><tr><td>gpt-5.2</td><td>√</td><td>√</td><td>X</td><td>√</td></tr><tr><td>gpt-5.4</td><td>√</td><td>√</td><td>x</td><td>√</td></tr><tr><td>gpt-5.4-mini</td><td>√</td><td>√</td><td>x</td><td>√</td></tr><tr><td>gpt-5.5</td><td>√</td><td>x</td><td>X</td><td>√</td></tr><tr><td>gpt-6-luna</td><td>√</td><td>X</td><td>x</td><td>√</td></tr><tr><td>gpt-oss-120b</td><td>√</td><td>√</td><td>x</td><td>√</td></tr><tr><td>gpt-oss-20b</td><td>√</td><td>√</td><td>x</td><td>√</td></tr><tr><td>qwen3</td><td>√</td><td>√</td><td>x</td><td>√</td></tr><tr><td>llama-3.3</td><td>√</td><td>√</td><td>x</td><td>√</td></tr><tr><td>o4-mini</td><td>√</td><td>√</td><td>x</td><td>√</td></tr><tr><td>Gemma4</td><td>√</td><td>X</td><td>√</td><td>√</td></tr><tr><td>Gemma3</td><td>√</td><td>x</td><td>√</td><td>√</td></tr><tr><td>gpt-5.2-chat-latest</td><td>√</td><td>x</td><td>X</td><td>√</td></tr><tr><td>gpt-5.3-chat-latest</td><td>√</td><td>X</td><td>X</td><td>√</td></tr><tr><td>kimi-k2.5</td><td>√</td><td>x</td><td>x</td><td>√</td></tr></table>

Table 2: Availability of temperature settings across model providers. A check mark (√) indicates that the corresponding temperature was supported by the provider and included in our experiments, whereas a cross (X) indicates that the temperature setting was unavailable and therefore was not tested.

<table><tr><td>Model</td><td>Variability</td><td>Max value</td><td>CV mean</td><td>CV max</td></tr><tr><td>deepseek-r1</td><td> $1 . 6 0 { \scriptstyle \pm 1 . 8 6 }$ </td><td>5.58</td><td> $1 2 . 1 2 _ { \pm 3 8 . 3 0 }$ </td><td>223.61</td></tr><tr><td>gpt-oss-20b</td><td> $1 . 0 7 _ { \pm 1 . 1 6 }$ </td><td>4.83</td><td> $3 1 . 1 8 _ { \pm 2 7 . 8 1 }$ </td><td>125.00</td></tr><tr><td>depseek-v3.1</td><td> $0 . 7 5 { \scriptstyle \pm 1 . 0 9 }$ </td><td>5.28</td><td> $2 8 . 8 8 { \scriptstyle \pm 2 1 . 1 0 }$ </td><td>89.00</td></tr><tr><td>o4-mini</td><td> $0 . 6 6 _ { \pm 0 . 9 5 }$ </td><td>4.00</td><td> $1 4 . 3 5 { \scriptstyle \pm 1 4 . 3 8 }$ </td><td>56.46</td></tr><tr><td>kimi-k2.5</td><td> $0 . 6 4 { \scriptstyle \pm 0 . 9 1 }$ </td><td>3.78</td><td> $2 3 . 7 2 _ { \pm 2 4 . 0 0 }$ </td><td>131.34</td></tr><tr><td>qwen3</td><td> $0 . 6 3 { \scriptstyle \pm 1 . 2 1 }$ </td><td>6.34</td><td> $1 7 . 9 4 \substack { \pm 2 0 . 6 2 }$ </td><td>118.59</td></tr><tr><td>gpt-oss-120b</td><td> $0 . 6 3 { \scriptstyle \pm 0 . 5 8 }$ </td><td>2.41</td><td> $2 1 . 7 7 _ { \pm 1 7 . 2 3 }$ </td><td>78.26</td></tr><tr><td>deepseek-v4</td><td> $0 . 5 3 { \scriptstyle \pm 0 . 9 6 }$ </td><td>5.69</td><td> $3 4 . 0 7 _ { \pm 4 1 . 6 2 }$ </td><td>199.44</td></tr><tr><td>llama-3.3</td><td> $0 . 5 1 _ { \pm 0 . 5 0 }$ </td><td>1.95</td><td> $3 0 . 2 3 _ { \pm 4 1 . 9 9 }$ </td><td>240.00</td></tr><tr><td>gpt-5.3-chat-latest</td><td> $0 . 4 3 { \scriptstyle \pm 0 . 8 9 }$ </td><td>5.66</td><td> $2 0 . 4 7 { \scriptstyle \pm 2 5 . 5 8 }$ </td><td>158.11</td></tr><tr><td>gpt-4.1</td><td> $0 . 4 3 _ { \pm 0 . 4 7 }$ </td><td>2.33</td><td> $1 5 . 5 3 { \scriptstyle \pm 1 4 . 4 8 }$ </td><td>85.55</td></tr><tr><td>gpt-5.2</td><td> $0 . 4 2 _ { \pm 0 . 7 1 }$ </td><td>3.67</td><td> $1 5 . 4 7 _ { \pm 2 3 . 0 1 }$ </td><td>129.80</td></tr><tr><td>gpt-5.4-mini</td><td> $0 . 4 0 { \scriptstyle \pm 0 . 6 7 }$ </td><td>3.75</td><td> $3 0 . 9 8 _ { \pm 7 2 . 4 8 }$ </td><td>300.00</td></tr><tr><td>gpt-5.4</td><td> $0 . 4 0 { \scriptstyle \pm 1 . 0 4 }$ </td><td>7.03</td><td> $1 6 . 3 1 _ { \pm 1 8 . 6 0 }$ </td><td>86.49</td></tr><tr><td>gpt-5.2-chat-latest</td><td> $0 . 3 9 { \scriptstyle \pm 1 . 0 5 }$ </td><td>7.30</td><td> $1 5 . 7 3 { \scriptstyle \pm 2 2 . 9 9 }$ </td><td>142.97</td></tr><tr><td>gemma3</td><td> $0 . 3 6 { \scriptstyle \pm 0 . 5 5 }$ </td><td>3.02</td><td> $8 . 7 4 { \scriptstyle \pm 7 . 5 8 }$ </td><td>35.36</td></tr><tr><td>gpt-5.5</td><td> $0 . 3 4 { \scriptstyle \pm 0 . 5 7 }$ </td><td>3.54</td><td> $2 1 . 2 6 _ { \pm 3 6 . 3 5 }$ </td><td>244.95</td></tr><tr><td>gpt-6-luna</td><td> $0 . 3 2 _ { \pm 0 . 5 2 }$ </td><td>1.89</td><td> $2 4 . 3 6 _ { \pm 3 2 . 2 4 }$ </td><td>104.88</td></tr><tr><td>alice_ai</td><td> $0 . 2 6 _ { \pm 0 . 3 3 }$ </td><td>1.46</td><td> $1 7 . 4 7 _ { \pm 3 2 . 4 2 }$ </td><td>214.33</td></tr><tr><td>gpt-4.1-mini</td><td> $0 . 2 5 { \scriptstyle \pm 0 . 2 6 }$ </td><td>1.07</td><td> $1 6 . 8 5 { \scriptstyle \pm 3 0 . 8 7 }$ </td><td>223.61</td></tr><tr><td>gemma</td><td> $0 . 2 4 { \scriptstyle \pm 0 . 3 0 }$ </td><td>1.20</td><td> $2 2 . 4 7 _ { \pm 5 7 . 0 6 }$ </td><td>300.00</td></tr><tr><td>giga_chat</td><td> $0 . 1 8 { \scriptstyle \pm 0 . 2 6 }$ </td><td>1.46</td><td> $7 . 0 4 { \scriptstyle \pm 8 . 3 1 }$ </td><td>35.88</td></tr></table>

Table 3: Model stability across temperature and crossscale variability. Values are reported as mean ± standard deviation where applicable.

<table><tr><td>Model</td><td>Missing responses</td></tr><tr><td>alice_ai</td><td>0</td></tr><tr><td>deepseek-r1</td><td>53</td></tr><tr><td>deepseek-v3.1</td><td>2</td></tr><tr><td>deepseek-v4</td><td>0</td></tr><tr><td>GigaChat</td><td>1</td></tr><tr><td>gpt-4.1</td><td>1</td></tr><tr><td>gpt-4.1-mini</td><td>4</td></tr><tr><td>gpt-5.2</td><td>0</td></tr><tr><td>gpt-5.4</td><td>0</td></tr><tr><td>gpt-5.4-mini</td><td>0</td></tr><tr><td>gpt-5.5</td><td>0</td></tr><tr><td>gpt-6-luna</td><td>7</td></tr><tr><td>gpt-oss-120b</td><td>2</td></tr><tr><td>gpt-oss-20b</td><td>17</td></tr><tr><td>qwen3</td><td>0</td></tr><tr><td>llama-3.3</td><td>8</td></tr><tr><td>o4-mini</td><td>1</td></tr><tr><td>Gemma4</td><td>0</td></tr><tr><td>Gemma3</td><td>9</td></tr><tr><td>gpt-5.2-chat-latest</td><td>0</td></tr><tr><td>gpt-5.3-chat-latest</td><td>0</td></tr><tr><td>kimi-k2.5</td><td>0</td></tr></table>

Table 4: Number of missing responses for each evaluated LLM.

![](images/90bbc7c19406f9d1320e2821dc9143eae614eddedcee7ebf31e3fe7d034bc9e1.jpg)  
Table 5: Mean psychometric profile of eight representative API-based models across all available temperatures and repeated runs. Each score was normalized using the theoretical range of its questionnaire scale. Colors show seven equal-width bands from low to high normalized scores:,,, ,, and■. A dash indicates that no valid score was available in the released results. For bipolar 16PF factors, labels run from the low-score pole to the high-score pole.

![](images/f9223eca8630436df31d63c5f03d8a3d94c70d430c99ce53a8d70c22af46b894.jpg)  
Figure 17: Representation of depression and anxiety scales across all API models in different questionnaires.

![](images/13e44e5013093782ed10ee5441809bfc12a51154a30bbadda7bbd9cb878d060d.jpg)  
Figure 18: Results of the 16PF (Cattell) questionnaire across all scales for API-based LLMs.

![](images/13bc2a1eb7794bbf17aff056b2690614e566fdb9417119df9e2f778dd49c6a8f.jpg)  
Figure 19: Attachment style classification of API-based LLMs (based on ASQ results).

![](images/17203d31635e59fa48bf46ed8b3e97e71749a29e3e07f4aeac2cc381592fff90.jpg)  
Figure 20: Temperament classification of API-based LLMs (based on EPI results).

## D Human vs. AI Interface

When correlations were examined by psychometric instrument, the strongest association was observed for the BDHI $( r = . 8 4 6 , p < . 0 0 1 )$ , followed by the SD3 (r = .504, p < .001), EPI $( r = . 3 9 4$ $p \ = \ . 0 0 7 )$ , and BFI-2 (r = .326, p = .004). No significant associations were observed for BAI, BDI, ASQ, GAD-7, SCL-90, or HADS.

<table><tr><td>Psychological inventories</td><td>Pearson correlation</td><td> $\Delta r$ </td></tr><tr><td>all w/o BDHI</td><td>0.544</td><td>0.086</td></tr><tr><td>all w/o SCL-90</td><td>0.548</td><td>0.081</td></tr><tr><td>all w/o BFI-2</td><td>0.621</td><td>0.008</td></tr><tr><td>all w/o SD3</td><td>0.624</td><td>0.005</td></tr><tr><td>all w/o GAD-7</td><td>0.636</td><td>-0.007</td></tr><tr><td>all w/o ASQ</td><td>0.640</td><td>-0.010</td></tr><tr><td>all w/o EPI</td><td>0.640</td><td>-0.010</td></tr><tr><td>all w/o HADS</td><td>0.644</td><td>-0.015</td></tr><tr><td>all w/o BDI</td><td>0.661</td><td>-0.032</td></tr><tr><td>all w/o BAI</td><td>0.696</td><td>-0.067</td></tr></table>

Table 6: Changes in Pearson correlation after leave-onetest-out analysis. $\Delta r = r _ { \mathrm { f u l l } } - r _ { \mathrm { w / o t e s t } }$ . Positive values indicate that excluding a questionnaire decreases the overall correlation, whereas negative values indicate a slight increase.

<table><tr><td>Person</td><td>N</td><td>Pearson&#x27;s r</td><td>p</td><td>Strength</td></tr><tr><td>1</td><td>40</td><td>0.79</td><td>1.957491e-09</td><td>Strong</td></tr><tr><td>2</td><td>40</td><td>0.79</td><td>1.510057e-08</td><td>Strong</td></tr><tr><td>3</td><td>40</td><td>0.44</td><td>4.289652e-03</td><td>Moderate</td></tr><tr><td>4</td><td>40</td><td>0.83</td><td>2.152543e-11</td><td>Strong</td></tr><tr><td>5</td><td>40</td><td>0.84</td><td>1.148446e-11</td><td>Strong</td></tr><tr><td>6</td><td>40</td><td>0.88</td><td>6.756510e-14</td><td>Strong</td></tr><tr><td>7</td><td>40</td><td>0.79</td><td>7.067158e-10</td><td>Strong</td></tr><tr><td>8</td><td>40</td><td>0.82</td><td>1.489398e-10</td><td>Strong</td></tr><tr><td>9</td><td>40</td><td>0.62</td><td>2.229973e-05</td><td>Moderate</td></tr><tr><td>10</td><td>40</td><td>0.71</td><td>2.739706e-07</td><td>Strong</td></tr><tr><td>11</td><td>40</td><td>0.66</td><td>3.360919e-06</td><td>Moderate</td></tr><tr><td>12</td><td>40</td><td>0.68</td><td>1.553977e-06</td><td>Moderate</td></tr><tr><td>13</td><td>40</td><td>0.81</td><td>3.003184e-10</td><td>Strong</td></tr><tr><td>14</td><td>40</td><td>0.63</td><td>1.460148e-05</td><td>Moderate</td></tr><tr><td>15</td><td>40</td><td>0.62</td><td>1.702052e-05</td><td>Moderate</td></tr><tr><td>Overall</td><td>600</td><td>0.63</td><td>1.425519e-67</td><td>Moderate</td></tr></table>

Table 7: Pearson correlations between human ratings and model scores for each participant. Pearson's r indicates the direction and strength of the linear association between human ratings and model scores. The p-value indicates the statistical significance of the correlation. Correlation strength was classified as weak $( | r | < 0 . 3 0 )$ , moderate $( 0 . 3 0 \leq | r | < 0 . 7 0 )$ , or strong $( 0 . 7 0 \leq | r | < 1 . 0 0 )$

## E Models

The list includes instruct-tuned, chat-oriented, and reasoning models developed by different providers: OpenAI (GPT-4.1, GPT-4.1-mini, GPT-5.2, GPT-5.2-chat-latest, GPT-5.3-chat-latest, GPT-5.4, GPT-5.4-mini, GPT-5.5, GPT-5.6-Luna o4-mini GPT-OSS-20B, GPT-OSS-120B), Google (Gemma 3n E4B Instruct, Gemma 4 31B Instruct), Meta (Llama 3.3 70B Instruct Turbo), Alibaba (Qwen3- 235B-A22B-Instruct-2507), DeepSeek (DeepSeek-R1 and DeepSeek-V3.1, DeepSeek-V4-Flash), Moonshot AI (Kimi K2.5), as well as Russian language-based models developed by Yandex (Alice GPT) and Sber (GigaChat).

<table><tr><td>Provider</td><td>Models</td><td>Access date</td></tr><tr><td>Together AI</td><td>deepseek-ai/DeepSeek-R1, deepseek-ai/DeepSeek-V3.1, deepseek-ai/DeepSeek-V4, google/gemma-3n-E4B-it, Qwen/Qwen3-235B-A22B-Instruct-2507, openai/gpt-oss-20b, openai/gpt-oss-120b, llama-3.1-8b-instant, llama-3.3-70b-versatile, Gemma 3n E4B Instruct, Gemma 4 31B Instruct</td><td>Apr. 17, 2026; Sep. 28, 2026</td></tr><tr><td>OpenAI API</td><td>o4-mini, gpt-4.1, gpt-4.1-mini, gpt-5.2, gpt-5.2-chat-latest, gpt-5.3-chat-latest, gpt-5.4, gpt-5.4-mini, gpt-5.5,</td><td>Apr. 15, 2026; Sep. 28, 2026</td></tr><tr><td>Kimi API</td><td>gpt-6-Luna moonshotai/Kimi-K2.5</td><td>Apr. 19, 2026</td></tr><tr><td>Yandex Cloud</td><td>gpt://b1gro2jeni64i2m9qdcl/aliceai-1lm</td><td>Mar. 27, 2026</td></tr><tr><td>GigaChat API</td><td>GigaChat</td><td>Mar. 27, 2026</td></tr></table>

Table 8: LLM providers, accessed models, and access dates used in this study.

![](images/89272f66a1d6c73919f1c8466b93bc6d756121fde143100da7f159b75d49248d.jpg)

![](images/117896306926dbced42095574e9dfb913d5636062948e088a01fd744fb86a8d1.jpg)

![](images/db4113ee740fa6740cb74268d3016feb2ed53b7928aac82e68fb4f067a4b6417.jpg)

![](images/8d418394bbb5d19ae64c05a090ba85d1a9b79ab65aa8fce57aa62b2b81da676b.jpg)

![](images/280535b5572c39dbd98962b844166dbf0ca1167b54eca911991f857fd13c8c86.jpg)

![](images/ff68a57191247eaee7f51961bc2a0961d48486fbad5697ade41e2d359c996dae.jpg)

![](images/7f8806b59600a761029a72085b1d1db53928f6b1db9d9101c99b40f5efae0163.jpg)

![](images/37352518341e00d0b886ec53e02824dc56470add86467ae8637a3e4690e6d0ec.jpg)

![](images/a7d49100bd7a7ae3aca0f9231788a3f75d8796167868ded1e8419956223e1ea2.jpg)

![](images/2423fa1acd582f8e03e5cc53b8e73d7fcd10b9674b1a84d7771a92f2b2d69300.jpg)

![](images/9cab4150f72b38123232fdfceac5fe946aa83d20768b4955ac62767f7e4fa2e8.jpg)

![](images/b5c8677a5dde58e5839bbe9bd84e681f99cc60c3c6e0ca6612f3db96c30a9993.jpg)

![](images/1f6fa6baab9b7c6ffd16769440e8923548b023d625ba8499aa489917d16a5c43.jpg)

![](images/ff5d9a7a217b13ae9f8755de56eb8160f1212d68a00b970baa1f212f06821964.jpg)

![](images/c84690a2f05e59091449c667492f1112a3a5f94348c0ca2b5f179444baba5e95.jpg)  
Figure 21: Human ratings and model scores across selected psychological scales for each participant. For each questionnaire, the scale with the strongest absolute Pearson correlation between human ratings and model scores was selected.

![](images/38615206a9021cb57a22d349c8379cdc7e94667b22268faee6986999577d5198.jpg)  
Figure 22: Systematic differences between interface-based model and human psychometric scores.

## F Data Collection Instructions

The following instructions were provided to participants for the human–AI experiment.

File: Google Sheets link

## 1. Preparation

1. Open the shared spreadsheet.

2. Select any available worksheet that has not yet been assigned to another participant

3. Enter your personal information in the yellow table.

## 2. Completing the Tests Using an LLM

1. Enter information about your LLM usage in the middle table.

2. In the first column, specify the LLM you use most frequently, including its version.

3. For each test, follow these steps:

(a) Open the corresponding . txt file.

(b) Select all text (Ctrl + A) and copy it (Ctrl + C).

(c) Do not modify the test or the prompt. Do not add or remove any content.

(d) Paste the text into a new chat with the LLM and submit it.

(e) Wait for the model's response. The model should return its answers in JSON format.

(f) Copy the model's entire response without any modifications and paste it into the third column of the spreadsheet.

4. Use a new chat for each test. Each test must be administered in a separate conversation, resulting in 11 separate chats in total.

Important. Check that the number of answers provided by the model matches the number of questions in the test. If the model skips any questions or provides fewer answers than expected, notify the researcher.

## 3. Completing the Tests Independently

1. Complete all tests independently in the bottom table.

2. Open each test in . txt format.

3. Enter your answers in the spreadsheet.

Please answer honestly, selecting the options that genuinely correspond to you

Before submitting, make sure that every question has been answered and that no items have been left unanswered.

Confidentiality. All responses are anonymous and will be used solely for research purposes.

Deadline: date