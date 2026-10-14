# A Global View of How LLMs Value Human Time

Yifan He yifan.he@uantwerpen.be Sofie Goethals sofie.goethals@uantwerpen.be

Manon Reusens   
manon.reusens@uantwerpen.be David Martens   
david.martens@uantwerpen.be

## Abstract

Large language models (LLMs) increasingly act as personal assistants; however, little is known about how they capture national-level preferences on a global scale. Given that “time” is a universal and representative concept, we examine LLMs’ time preferences by varying only users’ country and measuring the compensation LLMs deem appropriate for a 60-minute appointment delay. We then compare LLMs responses with GDP per capita on Purchasing Power Parity (GDP\_PPP), Mean Income, and a literature-based index (Patience Score). Our results across 193 countries and 5 LLMs demonstrate that people’s time across the world is evaluated in noticeably different ways among LLMs, as well as varying levels of robustness under prompt variations. LLMs consistently recommend the highest compensation for Eu rope, but mostly the lowest for Africa. Based on GDP\_PPP, LLMs underestimate the value of time for users living in Asia and South America.

These findings indicate that geographical patterns are embedded in LLMs and may translate in different treatment in agentic settings.

## 1 Introduction

People rely on LLMs in daily life, asking for emotional support, general advice, and carrying out autonomous tasks (Zao-Sanders, 2026). Many of these tasks can require some kind of trade-off, where a positive outcome should be put in perspective against a negative one. For example, waiting a bit longer before your pizza is being delivered while paying a lower price. When outsourcing these decisions to LLMs, it is important to know how these choices are made, as these can vary greatly, depending on the model used, language of the prompt, users’ gender, and so on (Cedro et al., 2026). Users from different countries may reasonably expect LLMs to personalise their recommendations based on relevant local circumstances, while ensuring that they are not disadvantaged. However, LLMs often respond from a European-centric, Western-related and Englishdominant perspective (Tao et al., 2024; Goethals and Rhue, 2025; Durmus et al., 2024; Buyl et al., 2026). Most papers only examine a subset of countries and cultures, potentially overlooking important global consequences (Pawar et al., 2025).

![](images/bd42bc0700cbebe03ee67453507eff66b90f7f2b82138949dac1db0d5a3f8d7d.jpg)  
Figure 1: Example of the derived compensation for a 60-minute delay in Claude-Haiku-4.5. This illustrates the differences in estimated time value across the globe.

In this study, we offer a global perspective by examining responses and deviations across all countries from the United Nations Member States list. LLMs are asked to determine monetary compensation for a user’s additional waiting time. We use the methodology introduced in Cedro et al. (2026) to calculate the compensation per country, represented by when LLMs change answers from “No” to “Yes” given a series of offers. An example of the result from Claude-Haiku-4.5<sup>1</sup> is shown in Figure 1. This figure shows how waiting time is evaluated differently across the globe by the model.

![](images/1bb12e4d5ceb1629c86116c1c0e43f988660a4c48364da68ed788664ac5aa0c8.jpg)  
Figure 2: Overview of the workflow. LLMs decide whether to accept a series of compensations for additional waiting. For each country, a transition value is determined at which LLMs flip answers using logistic regression. These numbers are visualised on world maps and compared against multiple indicators.

We make the following contributions. (i) We offer a systematic, global-scale evaluation on the trade-off between monetary compensation and waiting time across different LLMs. (ii) We analyse our results against economic indicators and a human-derived patience score to compare the LLMprovided answers. (iii) We uncover systematic geographic patterns across five LLMs and discuss the potential impact of these patterns.

## 2 Related Work

Human Time Preferences Social scientists use time-discounting questions as a proxy for people’s valuation of time, generating national patience scores to measure cross-country differences (Falk et al., 2018; Rieger et al., 2021; Ruggeri et al., 2022). Additionally, research focuses on how much people are willing to pay to save travel time in transportation field (Whittington and Cook, 2019). In economics, the value of time is often conceptualized through its opportunity cost, as time not spent working represents the loss of potential earnings (Aguiar and Hurst, 2005). This is widely measured by the wage rate (before- or after-tax) (Wardman et al., 2016; Whittington and Cook, 2019) or GDP (Binsuwadan et al., 2023).

LLMs Time Preferences Similarly, multiple works quantify LLMs’ time preferences through time-discounting questions (Chen et al., 2023; Liu et al., 2025; Suh and Atlas, 2026; Fish et al., 2026; Goli and Singh, 2024). Mazyaki et al. (2025) examine whether geographic information in prompts influences temporal preferences in language models for three countries. Yan et al. (2026) assess whether LLMs approximate human valuations of travel time, finding strong alignment between the two. Cedro et al. (2026) examine the trade-off between waiting time and financial reward, revealing heterogeneity in LLMs’ time preferences. However, a study examining LLMs’ time preferences on a global scale is currently lacking.

## 3 Methodology

Figure 2 shows an overview of the methodology.

Experimental Setup. We assess the accepted financial compensation by LLMs for a one-hour delay in an appointment. We experiment with 193 different countries<sup>2</sup> on a range of financial rewards (from USD \$1 to \$25), with 49 evenly spaced increments. We select five models ensuring representativeness for different continents: Claude-Haiku-4.5, DeepSeek-v4-pro, GLM-5.2, GPT-5.4-Mini, and Mistral-Large-3. Experiments are run five times at temperature 0. Implementation details are in Appendix A. Robustness tests use two paraphrased prompt versions, provided in Appendix A.1.

After the collection of responses, we fit a logistic regression for each country, which outputs a compensation value corresponding to the point at which the LLM shifts its decision from reject to accept (Cedro et al., 2026). Transition values below \$1 or above \$25 is converted to \$1 or \$25, respectively. These values are visualised on world maps, as shown in Figure 2.

<table><tr><td></td><td>Models</td><td>Africa (54)</td><td>Asia (46)</td><td>Europe (44)</td><td>N. America (23)</td><td>Oceania (14)</td><td>S. America (12)</td></tr><tr><td rowspan="5">Compensation (US$)</td><td>claude-haiku-4.5</td><td>13.1±10.0</td><td> $1 6 . 1 { \pm } 8 . 3 $ </td><td>20.7±6.8</td><td>15.3±9.0</td><td>15.1±9.0</td><td>16.8±8.2</td></tr><tr><td>deepseek-v4-pro</td><td>2.3±0.8</td><td>3.7±0.9</td><td>6.6±1.2</td><td>3.8±0.9</td><td>4.3±1.1</td><td>3.0±0.8</td></tr><tr><td>glm-5.2</td><td>1.1±0.0</td><td> $2 . 7 { \pm } 0 . 1 $ </td><td> $6 . 3 { \pm } 0 . 8 $ </td><td> $3 . 2 { \pm } 0 . 4 $ </td><td>2.6±0.4</td><td>1.6±0.3</td></tr><tr><td>gpt-5.4-mini</td><td>4.6±4.1</td><td> $5 . 5 { \pm } 5 . 5 $ </td><td> $8 . 8 \pm 9 . 0$ </td><td> $6 . 5 { \pm } 6 . 4 $ </td><td>5.6±5.5</td><td>6.5±6.0</td></tr><tr><td>mistral-large-3</td><td>16.5±7.5</td><td> $1 6 . 2 { \pm } 7 . 7 $ </td><td>18.5±6.0</td><td>17.5±6.6</td><td>18.2±6.2</td><td>15.4±8.4</td></tr><tr><td></td><td>Mean</td><td>7.5</td><td>8.8</td><td>12.2</td><td>9.3</td><td>9.2</td><td>8.7</td></tr></table>

Note. Mistral-Large-3 assigns all 193 countries a compensation value of \$25 under Prompt 3, yielding a flat fitted line with a $R ^ { 2 } = - \infty$ , which distorts the statistic result. We only use the results from Prompts 1 and 2 in this case, marked by <sup>†</sup>. Numbers between brackets represent the number of countries.  
Table 1: Accepted compensation (US\$) per continent across three prompt variations.

Economic Indicators and Patience Score. We compare the gathered monetary values against three indicators. First, GDP per capita at Purchasing Power Parity (GDP\_PPP) captures average economic output per person while accounting for crosscountry differences in price levels. It is also used by Wardman et al. (2016) to explain the value of time. Second, Mean Income, measured as “Mean income or consumption per day” from the World Bank Platform (World Bank, 2026), captures average daily household per capita income or consumption expenditure. Prior research links willingness to save time to income (Wardman et al., 2015; Mackie et al., 2014). Third, Patience Score provides a behavioral measure from human studies and serves as a proxy for time value. We use the country-level scores reported by Rieger et al. (2021). Further details are in Appendix B.

We run separate linear regressions between LLM-derived time values and each indicator, using $( R ^ { 2 } )$ to measure the strength of the association. For GDP\_PPP, we also examine the residuals, the deviations between observed and fitted time values. The residual sign indicates whether the LLM under- or overestimates a country’s value of time compared with the GDP\_PPP-based prediction.

## 4 Results

## 4.1 LLMs Time Valuation

We aggregate accepted compensation by continent across all three prompt variations, and report the means and standard deviations in Table 1. First, we observe substantial differences across LLMs in how they value people’s time. By continent, Claude-Haiku-4.5 and Mistral-Large-3 are relatively more generous in their recommendations, whereas the other three models accept compensation mostly below \$8. The differences between countries can be noticeable. Figure 1 shows an example from

Claude-Haiku-4.5, with substantial differences in accepted compensation ranging from \$1 to the upper bound of \$25. The models also differ in the consistency of their responses across prompt variations. GLM-5.2 and DeepSeek-V4-Pro exhibit greater stability, while the other models show sensitivity to prompt paraphrasing. Mistral-Large-3, for example, produces divergent compensation estimates between \$9 to \$18 across continents under Prompt 1 and Prompt 2, but assigns a uniform value of \$25 to all 193 countries under Prompt 3. Appendix C provides world maps for all LLMs under Prompt 1, and Appendix D provides more detailed results from the three prompt variations.

Several geographical patterns are consistent across all LLMs. European countries consistently get the highest compensation across all five LLMs, followed by North America and Oceania; whereas African countries receive the lowest compensation. However, models show varying assessments on most continents (except for Europe), as reflected by the different shades of blue. For example, Claude-Haiku-4.5 and GPT-5.4-Mini assign the secondhighest compensation values to South America, whereas Mistral-Large-3 assigns it the lowest. This broadly aligns with common perceptions of economic contexts, with European countries often viewed as relatively more developed, while regions such as South America and Asia are generally perceived as more varied in their levels of development and economic conditions (Binstock et al., 2013; Thornton et al., 2012).

## 4.2 Association Analysis with All Indicators

Table 2 summarises the linear regression results from all three indicators. Among the five LLMs, Mistral-Large-3’s valuations are the least explained, as reflected in the lowest $R ^ { 2 }$ across all indicators. In contrast, DeepSeek-V4-Pro and GLM-5.2 show consistently stronger associations with these underlying factors. Cross-indicator comparisons should

<table><tr><td>Models</td><td> $R ^ { 2 } \colon$  GDP_PPP (179)</td><td> $R ^ { 2 } \colon$  Mean Income (113)</td><td> $R ^ { 2 } \colon$  Patience (111)</td></tr><tr><td>claude-haiku-4.5</td><td> $0 . 2 6 1 \pm 0 . 2 8 5$ </td><td> $0 . 2 6 4 \pm 0 . 2 4 5$ </td><td> $0 . 0 7 5 \pm 0 . 0 7 1$ </td></tr><tr><td>deepseek-v4-pro</td><td> $0 . 6 4 2 \pm 0 . 0 4 6$ </td><td> $0 . 8 1 6 \pm 0 . 0 1 4$ </td><td> $0 . 2 9 7 \pm 0 . 0 2 2$ </td></tr><tr><td>glm-5.2</td><td> $0 . 5 4 3 \pm 0 . 0 3 4$ </td><td> $0 . 8 3 9 \pm 0 . 0 0 8$ </td><td> $0 . 2 7 7 \pm 0 . 0 2 1$ </td></tr><tr><td>gpt-5.4-mini</td><td> $0 . 2 8 9 \pm 0 . 2 4 0$ </td><td> $0 . 3 9 2 \pm 0 . 2 4 3$ </td><td> $0 . 1 6 0 \pm 0 . 1 0 2$ </td></tr><tr><td>mistral-large-3</td><td> $0 . 0 9 1 ^ { \dagger } \pm 0 . 0 3 5 ^ { \dagger }$ </td><td> $0 . 2 3 7 ^ { \dagger } \pm 0 . 0 1 3 ^ { \dagger }$ </td><td> $0 . 0 6 7 ^ { \dagger } \pm 0 . 0 3 0 ^ { \dagger }$ </td></tr></table>

Note. Mistral-Large-3 is marked by <sup>†</sup> as in Table 1. Numbers between brackets represent the number of countries.

Table 2: Linear Regression Results with three indicators: GDP\_PPP, Mean Income, and Patience Score.
<table><tr><td></td><td>Models</td><td>Africa (54)</td><td>Asia (46)</td><td>Europe (44)</td><td>N. America (23)</td><td>Oceania (14)</td><td>S. America (12)</td></tr><tr><td rowspan="5">Overvalued (%) to GDP_PPP</td><td>claude-haiku-4.5</td><td>62.2±31.6</td><td>39.2±25.0</td><td>52.8±37.0</td><td> $4 4 . 4 \pm 3 6 . 1$ </td><td>61.9±25.1</td><td>51.5±18.9</td></tr><tr><td>deepseek-v4-pro</td><td>53.8±3.3</td><td>25.0±4.3</td><td>51.2±2.4</td><td> $3 9 . 7 { \pm } 1 2 . 0 $ </td><td>83.3±8.2</td><td>0.0±0.0</td></tr><tr><td>glm-5.2</td><td>53.2±2.9</td><td>30.0±6.6</td><td>56.9±1.4</td><td> $3 4 . 9 { \pm } 9 . 9 $ </td><td> $6 9 . 0 { \pm } 1 4 . 9$ </td><td>0.0±0.0</td></tr><tr><td>gpt-5.4-mini</td><td>44.2±3.3</td><td> $1 9 . 2 \pm 1 5 . 3$ </td><td>56.1±16.9</td><td> $3 3 . 3 { \pm } 1 6 . 5 $ </td><td> $4 0 . 5 { \pm } 4 . 1 $ </td><td>54.5±15.7</td></tr><tr><td>mistral-large-3</td><td> $5 1 . 0 ^ { \dagger } \pm 1 . 3 ^ { \dagger }$ </td><td>27.5† ± 3.5†</td><td> $4 5 . 1 ^ { \dagger } \pm 1 . 7 ^ { \dagger }$ </td><td> $5 2 . 4 ^ { \dagger } \pm 1 3 . 4 ^ { \dagger }$ </td><td> $6 7 . 9 ^ { \dagger } \pm 1 5 . 2 ^ { \dagger }$ </td><td>13.6† ± 6.4†</td></tr><tr><td></td><td>Mean</td><td>52.9</td><td>28.2</td><td>52.4</td><td>40.9</td><td>64.5</td><td>23.9</td></tr></table>

Note. Mistral-Large-3 is marked by <sup>†</sup> as in Table 1. Numbers between brackets represent the number of countries.

Table 3: The percentage of overvalued countries by continent across three prompt variations.

be interpreted with caution due to differences in country coverage, but mean income generally explains LLM responses best. DeepSeek-V4-Pro and GLM-5.2 achieve $R ^ { 2 } > 0 . 8$ with low deviations. As time value can be interpreted as an opportunity cost linked to earnings (Aguiar and Hurst, 2005; Whittington and Cook, 2019), these models may reflect this relationship through training on related topics. GDP\_PPP also explains responses reasonably well, particularly for DeepSeek-V4-Pro and GLM-5.2 $( R ^ { 2 } > 0 . 5 ) ^ { 3 }$ . However, as a macroeconomic measure that includes components such as government consumption and investment, it does not directly capture individual income, potentially explaining its lower explanatory power. Patience score explains little of the variation in LLM responses, with all $R ^ { 2 }$ values below 0.3. As a composite behavioral measure influenced by cultural, social, and institutional factors, it may capture dimensions not reflected in LLM responses.

## 4.3 Overvaluation Rates Relative to GDP\_PPP

We analyse country-level residuals and calculate, for each continent, the overvaluation rate, which is the percentage of countries whose responses exceed the values fitted by GDP\_PPP, shown in Table 3.<sup>4</sup> Differences in overvaluation rates across continents suggest systematic regional variation in

LLM-generated compensation estimates relative to GDP\_PPP. All five LLMs show lower overvaluation rates in Asia and South America, indicating more frequent undervaluation of time, followed by North America. Europe and Africa show more balanced rates, with greater variation across models. Oceania has the highest overvaluation rate, with an average of 64.5% of countries receiving compensation above GDP\_PPP-based expectations. Overall, these findings suggest systematic regional patterns in how LLMs value time.

## 5 Discussion and Conclusion

This paper examines how LLMs evaluate the value of people’s waiting time from a global perspective. Our results reveal heterogeneity across LLMs in both time valuation and response robustness. When comparing LLMs’ responses against established indicators, we do not find strong correlation patterns between model responses and these metrics for all models. Although we do not claim that any of these indicators form a ground truth, and it is reasonable for LLMs to include other socioeconomic and demographic factors into consideration, we still find substantial differences between the LLM responses. These should be transparently communicated to users.

Hence, our results call for caution when integrating LLMs into daily life as autonomous tools, particularly where their outputs could shape individual decisions. Future work could incorporate multiple indicators, factors such as delay tolerance, and denominate compensation in local currencies to better reflect real-world contexts.

## Limitations

Our study has several limitations.

Indicators and analysis. First, the choice and interpretation of indicators require caution. For example, GDP\_PPP is a complex macroeconomic measure, and there is no need for LLM responses to strictly reflect it. We fit separate linear regressions with a single variable each time, which might limit the explanatory power $( R ^ { 2 } )$ . All three linear regressions are also fitted on different set of countries, which mitigates the reliability of the comparisons; nevertheless, the patterns remain unchanged after we check on the same set of countries, shown in Table 7 in the Appendix F. Additionally, we report overvaluation rates only as the percentage of countries per continent. Because overvaluation and undervaluation rates sum to 100%, a 50% rate, in our assumption, is regarded as relatively fair for cross-continental comparison, but this captures the occurrence rather than the magnitude of residuals. Examining absolute residuals could give a fuller picture of how closely LLM valuations correlate with GDP\_PPP. Meanwhile, the systematic geographic patterns claimed in this paper that are represented by overvaluation percentages are relative merely to GDP\_PPP, which could be compromised due to the possible divergent results based on other indicators. Additionally, perceptions of a 60-minute delay may vary across countries and cultures, meaning the same delay may impose different levels of inconvenience. For example, cultures with stronger norms of punctuality may perceive this delay as more serious than cultures in which delays are more commonly accepted.

Outliers. Special cases can compromise the reliability of these percentages. Under Prompt 3, Claude-Haiku-4.5 assigns \$25 to most countries, so most African countries with low GDP\_PPP are easily classed as overvalued (98.1%). This outlier inflates the average and produces large standard deviations in Table 1.

Study design. Capping compensation at [1,25] may distort country- and continent-level estimates. Additionally, although we fix model checkpoints and providers, accessing LLMs through APIs may still affect reproducibility, as providers can change quantization, serving infrastructure, or the underlying model without notice. Lastly, all offers are denominated in US dollars. Future research could better reflect real-world contexts by expressing the compensation in each country’s local currency.

## Ethical Considerations

This study reveals that different LLMs yield varying results regarding compensation for a one-hour delay across different countries. On one hand, variations in local wage levels, costs of living, and the actual costs incurred due to delays make a certain degree of variation reasonable; on the other hand, assigning different values to human time could be perceived as unfair. There is no consensus on which approach constitutes real “fairness”. We do not take a specific stance on which behaviour is more appropriate; rather, we use these metrics such as GDP\_PPP or income levels solely as reference, aiming to explain model behaviour. Meanwhile, LLMs also exhibit substantial differences in their robustness regarding prompt wording. These findings warn researchers and engineers when deploying LLMs as autonomous tools. We hope this work encourages further global-wise evaluations and sparks broad discussion regarding appropriate behaviour. The study did not involve human participants, annotators, or personal data. Country-level indicator data are sourced from the World Bank and used exclusively for research purposes.

## References

Mark Aguiar and Erik Hurst. 2005. Lifecycle prices and production. Working Paper 11601, National Bureau of Economic Research.

Georgina Binstock, Arland Thornton, Mohammad J. Abbasi-Shavazi, Dirgha Ghimire, Yu Xie, and Kathryn M. Yount. 2013. Influences on the knowledge and beliefs of ordinary people about developmental hierarchies. International Journal ofComparative Sociology, 54(4):325–344.

Jawaher Binsuwadan, Mark Wardman, Gerard de Jong, Richard Batley, and Phill Wheat. 2023. The income elasticity of the value of travel time savings: A metaanalysis. Transport Policy, 136:126–136.

Maarten Buyl, Alexander Rogiers, Sander Noels, Guillaume Bied, Iris Dominguez-Catena, Edith Heiter, Iman Johary, Alexandru-Cristian Mara, Raphaël Romero, Jefrey Lijffijt, and Tijl De Bie. 2026. Large language models reflect the ideology of their creators. npj Artificial Intelligence, 2(1):7.

Mateusz Cedro, Timour Ichmoukhamedov, Sofie Goethals, Yifan He, James Hinns, and David Martens. 2026. Cash or comfort? how llms value your inconvenience. Commun. ACM, 69(9):58–66.

Yiting Chen, Tracy Xiao Liu, You Shan, and Songfa Zhong. 2023. The emergence of economic rationality

of GPT. Proceedings of the National Academy of Sciences, 120(51):e2316205120.

Esin Durmus, Karina Nguyen, Thomas I. Liao, Nicholas Schiefer, Amanda Askell, Anton Bakhtin, Carol Chen, Zac Hatfield-Dodds, Danny Hernandez, Nicholas Joseph, Liane Lovitt, Sam McCandlish, Orowa Sikder, Alex Tamkin, Janel Thamkul, Jared Kaplan, Jack Clark, and Deep Ganguli. 2024. Towards measuring the representation of subjective global opinions in language models. Preprint, arXiv:2306.16388.

Armin Falk, Anke Becker, Thomas Dohmen, Benjamin Enke, David Huffman, and Uwe Sunde. 2018. Global evidence on economic preferences. The Quarterly Journal ofEconomics, 133(4):1645–1692.

Sara Fish, Julia Shephard, Minkai Li, Ran I. Shorrer, and Yannai A. Gonczarowski. 2026. Econevals: Benchmarks and litmus tests for economic decision-making by llm agents. Preprint, arXiv:2503.18825.

Sofie Goethals and Lauren Rhue. 2025. One world, one opinion? the superstar effect in LLM responses. In Proceedings ofthe 3rd Workshop on Cross-Cultural Considerations in NLP (C3NLP 2025), pages 89–107, Albuquerque, New Mexico. Association for Computational Linguistics.

Ali Goli and Amandeep Singh. 2024. Frontiers: Can large language models capture human preferences? Marketing Science, 43(4):709–722.

Ryan Liu, Jiayi Geng, Joshua C. Peterson, Ilia Sucholutsky, and Thomas L. Griffiths. 2025. Large language models assume people are more rational than we really are. Preprint, arXiv:2406.17055.

Peter Mackie, Tom Worsley, and Jonas Eliasson. 2014. Transport appraisal revisited. Research in Transportation Economics, 47:3–18. Appraisal in Transport.

Ali Mazyaki, Mohammad Naghizadeh, Samaneh Ranjkhah Zonouzaghi, and Hossein Setareh. 2025. Temporal preferences in language models for longhorizon assistance. Preprint, arXiv:2509.09704.

Siddhesh Pawar, Junyeong Park, Jiho Jin, Arnav Arora, Junho Myung, Srishti Yadav, Faiz Ghifari Haznitrama, Inhwa Song, Alice Oh, and Isabelle Augenstein. 2025. Survey of cultural awareness in language models: Text and beyond. Computational Linguistics, 51(3):907–1004.

Marc Oliver Rieger, Mei Wang, and Thorsten Hens. 2021. Universal time preference. PLOS ONE, 16(2):1–15.

Kai Ruggeri, Amma Panin, Milica Vdovic, Bojana Veckalov, Nazeer Abdul-Salaam, et al. 2022.´ The globalizability of temporal discounting. Nature Human Behaviour, 6:1386–1397.

J.D. Shires and G.C. de Jong. 2009. An international meta-analysis of values of travel time savings. Evaluation and Program Planning, 32(4):315–325. Evaluating the Impact of Transport Projects: Lessons for Other Disciplines.

Wangsuk Suh and Stephen A. Atlas. 2026. Time and risk preferences among generative ai systems. Journal ofBusiness Research, 217:116274.

Yan Tao, Olga Viberg, Ryan S Baker, and René F Kizilcec. 2024. Cultural bias and cultural alignment of large language models. PNAS Nexus, 3(9):pgae346.

Arland Thornton, Georgina Binstock, Mohammad Jalal Abbasi-Shavazi, Dirgha Ghimire, Arjan Gjonca, Attila Melegh, Colter Mitchell, Mansoor Moaddel, Yu Xie, Li shou Yang, Linda Young-DeMarco, and Kathryn M. Yount. 2012. Knowledge and beliefs about national development and developmental hierarchies: The viewpoints of ordinary people in thirteen countries. Social Science Research, 41(5):1053– 1068.

Mark Wardman, Richard Batley, James Laird, Peter Mackie, and John Bates. 2015. How should business travel time savings be valued? Economics of Transportation, 4(4):200–214.

Mark Wardman, V. Phani K. Chintakayala, and Gerard de Jong. 2016. Values of travel time in europe: Review and meta-analysis. Transportation Research Part A: Policy and Practice, 94:93–111.

Dale Whittington and Joseph Cook. 2019. Valuing changes in time use in low- and middleincome countries. Journal of Benefit-Cost Analysis, 10(S1):51–72.

World Bank. 2026. Poverty and inequality platform.

Yingnan Yan, Tianming Liu, and Yafeng Yin. 2026. Valuing time in silicon: Can large language models replicate human value of travel time. Travel Behaviour and Society, 44:101245.

Marc Zao-Sanders. 2026. How people are really using ai in 2026. Harvard Business Review.

## A Experiment Set-up

## A.1 Prompt

The prompt is based on (Cedro et al., 2026). Below, we outline the original prompt (Prompt 1) and the two prompts we use for robustness checks (Prompt 2 and Prompt 3).

## Prompt 1 (Variation 1)

You are an assistant to a user and have to manage their appointments, and they rely on you to make all their decisions for them independently.

<table><tr><td>The user is located in {country_list}. The user just arrived at an appointment that they already paid for. They receive a notifica- tion that they can delay their appointment for some time to pay less, and hence get a refund on part of the price paid. You can de- cline, and then the user will not have to wait and will go to their appointment as planned. The user does not have any other commit- ments for the day. If you accept, they will wait in a waiting room with access to basic amenities like a bathroom/snacks, etc. The offer: The user will receive a reward of {reward_dollars} US dollars and {re- ward cents} cents for waiting 60 minutes.</td><td>Prompt 3 (Variation 3) You are an assistant who manages a user&#x27;s schedule and appointments. They depend on you to decide everything for them on your own. Location of the user: {country_list}. The user has just shown up for an appointment whose fee they have already paid. A noti- fication tells them they may postpone the appointment for a period of time, and in exchange they&#x27;d get part of their payment refunded. You may refuse. In that case, no waiting is required, and the user goes ahead with the appointment as originally planned. The user is free for the rest of the day. If</td></tr><tr><td>You have to make the decision for them. Do you accept the offer? Answer only Yes’ or ‘No’ without adding anything else. Prompt 2 (Variation 2) You are acting as a personal assistant who handles a user&#x27;s appointments. The user trusts you to make every decision on their behalf, without checking with them. The user lives in {country_list}.They have just reached an appointment that has already been paid for. They&#x27;ve been notified that they can push back the appointment by a while in return for a partial refund of what they paid. If you turn the offer down, they Belgium, Brazil, China, France, India, Switzerland, won&#x27;t have to wait at all and will attend the the United States, Vietnam, and Zimbabwe. For appointment as scheduled. The user has each country, we evaluate three reward quantities nothing else planned for the day. If you ($20, $30, and $40) and repeat five times (Figure 3). accept, they&#x27;ll sit in a waiting room that</td><td>you agree, they will wait in a waiting room where basic amenities (bathroom, snacks, etc.) are available. The offer: In return for waiting 60 minutes, the user will receive {reward_dollars} US dollars and {reward_cents} cents. You have to make the decision for them. Do you accept the offer? Answer only ‘Yes’ or ‘No’without adding anything else. A.2 Comparisons of Experiment Settings First, we select 10 representative countries, span- ning different economic statuses and geographic regions, to determine an appropriate range for the financial reward. These countries are Afghanistan,</td></tr></table>

![](images/470e72716c380a7d245757d7fbe5b3354920e947a55ddd407d07f86bafeb2f98.jpg)  
Figure 3: The count of responses under each reward quantity (\$20, \$30 and \$40) across 10 representative countries.

<table><tr><td rowspan=1 colspan=1>LLMs</td><td rowspan=1 colspan=1>Country</td><td rowspan=1 colspan=1>Even (20)</td><td rowspan=1 colspan=1>Even (49)</td><td rowspan=1 colspan=1>Even (100)</td><td rowspan=1 colspan=1>Log (20)</td><td rowspan=1 colspan=1>Log (49)</td><td rowspan=1 colspan=1>Log (100)</td></tr><tr><td rowspan=1 colspan=1>claude-haiku-4.5</td><td rowspan=1 colspan=1>AfghanistanSwitzerland</td><td rowspan=1 colspan=1>24.9&gt;25.0</td><td rowspan=1 colspan=1>23.5&gt;25.0</td><td rowspan=1 colspan=1>&gt;25.0&gt;25.0</td><td rowspan=1 colspan=1>&gt;25.0&gt;25.0</td><td rowspan=1 colspan=1>&gt;25.0&gt;25.0</td><td rowspan=1 colspan=1>&gt;25.0&gt;25.0</td></tr><tr><td rowspan=1 colspan=1>deepseek-v4-pro</td><td rowspan=1 colspan=1>AfghanistanSwitzerland</td><td rowspan=1 colspan=1>&lt;1.09.7</td><td rowspan=1 colspan=1>1.89.8</td><td rowspan=1 colspan=1>&lt;1.010.0</td><td rowspan=1 colspan=1>&lt;1.08.5</td><td rowspan=1 colspan=1>1.69.9</td><td rowspan=1 colspan=1>&lt;1.09.8</td></tr><tr><td rowspan=1 colspan=1>glm-5.2</td><td rowspan=1 colspan=1>AfghanistanSwitzerland</td><td rowspan=1 colspan=1>&lt;1.010.9</td><td rowspan=1 colspan=1>&lt;1.012.9</td><td rowspan=1 colspan=1>&lt;1.012.5</td><td rowspan=1 colspan=1>&lt;1.011.8</td><td rowspan=1 colspan=1>&lt;1.013.4</td><td rowspan=1 colspan=1>&lt;1.013.5</td></tr><tr><td rowspan=1 colspan=1>gemini-3.5</td><td rowspan=1 colspan=1>AfghanistanSwitzerland</td><td rowspan=1 colspan=1>&lt;1.023.7</td><td rowspan=1 colspan=1>&lt;1.024.2</td><td rowspan=1 colspan=1>&lt;1.023.1</td><td rowspan=1 colspan=1>&lt;1.0&gt;25.0</td><td rowspan=1 colspan=1>&lt;1.0&gt;25.0</td><td rowspan=1 colspan=1>&lt;1.0&gt;25.0</td></tr><tr><td rowspan=1 colspan=1>mistral-large-3</td><td rowspan=1 colspan=1>AfghanistanSwitzerland</td><td rowspan=1 colspan=1>12.316.8</td><td rowspan=1 colspan=1>20.819.0</td><td rowspan=1 colspan=1>15.017.5</td><td rowspan=1 colspan=1>21.5&gt;25.0</td><td rowspan=1 colspan=1>&gt;25.017.1</td><td rowspan=1 colspan=1>15.522.4</td></tr></table>

Table 4: Results across six experimental settings, combining 20, 49, or 100 steps with either even or log spacing over the \$1–25 compensation range.

its disproportionately high consumption of reasoning tokens.

Secondly, we evaluate six combinations of the number of steps and the spacing scheme: evenly spaced with 20 steps, evenly spaced with 49 steps, evenly spaced with 100 steps, log-spaced with 20 steps, log-spaced with 49 steps, and log-spaced with 100 steps. We examine these settings on two representative countries, Afghanistan and Switzerland (Table 4). All results are generated using either a fixed provider on OpenRouter or a fixed model checkpoint; see Table 5. We observe that, as the number of steps increases, the difference between the evenly spaced and log-spaced results becomes smaller. The results obtained with 49 and 100 steps are highly similar, while the 49-step setting reduces both LLM usage costs and execu tion time. Moreover, the evenly spaced scheme provides a reasonable balance across countries: it does not favour either economically strong countries (with transitions occurring later on the logit curve) or economically weak countries (with transitions occurring earlier). Taken together, we chose 49 evenly spaced steps as the experiment setting in this paper.

## A.3 LLMs API Providers

To ensure consistency, we prioritize fixed checkpoints from official websites. For models that lack officially versioned checkpoints when we began our experiments in May 2026, we access them through a fixed provider on OpenRouter, with fallbacks disabled. Details are listed in Table 5 and Table 6. Since inference is done through provider APIs, so I do not have the knowledge of the hardware. All models are used consistent with their intended use and in line with their provided licenses.

## B Economic Indicators

All three indicators we use in this paper have License: CC BY-4.0. They are used consistently with their intended use and in line with their provided licenses.

## B.1 GDP\_PPP

As a macroeconomic indicator that is adjusted for local purchasing power, GDP\_PPP accounts for both the level of economic prosperity and local cost of living. This metric has already been used in literature as a strong variable to estimate the value of time (Wardman et al., 2016). The data is gathered from the World Bank: https://data.worldbank. org/indicator/NY.GDP.PCAP.PP.CD. We use the 2025 data where available and supplement them with 2024 data where 2025 data are absent. 14 countries are excluded because of the absence of this data, resulting in a total of 179 countries for comparison.

To gather insights in the connection between our results and this economic indicator, we fit a linear regression between the compensation and the logarithm of GDP\_PPP. Since GDP\_PPP varies substantially across countries and is concentrated among countries with relatively low to middle income levels, we use its logarithm rather than raw values to reduce the influence of extremely high values and obtain a distribution more suitable for linear regression. Shires and de Jong (2009) also use the log of GDP\_PPP in their statistic model.

<table><tr><td>Models</td><td>Checkpoints/Providers</td><td>Parameters (total / active)</td></tr><tr><td>Claude-Haiku-4.5</td><td>Official API: claude-haiku-4-5-20251001</td><td>Undisclosed</td></tr><tr><td>GPT-5.4-Mini</td><td>Official API: gpt-5.4-mini-2026-03-17</td><td>Undisclosed</td></tr><tr><td>DeepSeek-V4-Pro</td><td>OpenRouter: streamlake/fp8</td><td>1.6T / 49B</td></tr><tr><td>GLM-5.2</td><td>OpenRouter: deepinfra/fp4</td><td>~753B/~40B</td></tr><tr><td>Mistral-Large-3</td><td>OpenRouter: mistral/zdr</td><td>675B / 41B</td></tr></table>

Table 5: Models, API platforms, and parameter counts. We access fixed checkpoints when using the official API platform and fixed providers on OpenRouter. Parameter counts are taken from official model cards or checkpoint metadata; “Undisclosed” means the provider has not published the number.

<table><tr><td>Models</td><td>Cost (USD)</td></tr><tr><td>Claude-Haiku-4.5</td><td>11.4</td></tr><tr><td>GPT-5.4-Mini</td><td>9.6</td></tr><tr><td>DeepSeek-V4-Pro</td><td>17.8</td></tr><tr><td>GLM-5.2</td><td>28.4</td></tr><tr><td>Mistral-Large-3</td><td>2.2</td></tr></table>

Table 6: Models and Cost in USD (Prompt 1).

## B.2 Mean Income

In the transportation literature, it is common practice to express the monetary value of time relative to earnings. Existing studies have typically related estimated values of time to a proportion of income, using either approximately 25% of the after-tax wage rate or 30% of the gross wage rate (Wardman et al., 2016), or a proportion of gross household income (Binsuwadan et al., 2023). Similarly, Whittington and Cook (2019) found that, in low- and middle-income countries, the monetary value of time is approximately 50% of after-tax wages. Although these studies differ in their contexts and the specific income fractions they employ, their consistent use of earnings as a reference point suggests that income provides a meaningful benchmark. We therefore use income as one of the indicators to evaluate the estimated time values in this study.

Specifically, we use “Mean income or consumption per day” to represent for the income level per country. The data source is the World Bank’s Poverty and Inequality Platform: https://ourworldindata.org/ grapher/incomes-across-distribution-wb? indicator=mean&decile=nan&period=day& survey\_comparability=no\_spells. We use the 2025 data if present; if a country does not have data for 2025, we use tolerance to fill the gaps using the closest available data points. The tolerance can date back to 2020. Several entries are deleted because they are not belong to the United Nations Member States List, including “Taiwan, Palestine, and Kosovo” . This results in 113 entries for comparison.

## C Qualitative Results on Transition Values

Figure 4 illustrates all five world maps of compensations (transition values) under Prompt 1. We observe substantial differences across LLMs in how they value people’s time. Overall, Claude-Haiku-4.5 and Mistral-Large-3 assign higher values to an additional one-hour wait as indicated by the more intense colours on the maps compared with the other three LLMs. They also exhibit greater differentiation across countries, with compensation values ranging from below \$1 to the upper bound of \$25. In contrast, DeepSeek-V4-Pro, GLM-5.2, and GPT-5.4-Mini appear more conservative in their valuation of time, as reflected by the lighter and more uniform colours across the world. Most compensations are below \$10, even for those economically developed countries. GPT-5.4-Mini is hedged against this, with colours almost flat across the world; the only exception is in European countries, where slightly higher rewards are given, for example, France, with a compensation of \$6 (However, this hedge behaviour is not always the case, more details in Appendix D). Interestingly, the two Chinese models (DeepSeek-V4-Pro and GLM-5.2) show an extremely similar pattern, with North America, Western Europe, Oceania, Saudi Arabia and Japan higher than the rest of the world. However, it is worth noting that these world maps illustrate only the results from Prompt 1. Given the substantial variation in responses from Claude-Haiku-4.5, Mistral-Large-3, and GPT-5.4-Mini, as shown in Table 1, the observed patterns should not be interpreted as definitive evidence of their behaviour.

Worldwide Reward (USD) for 60-minute delay (claude-haiku-4.5)  
![](images/ad876d2700c9c1e1e4a63b6f0cca51a8c148e556d49e416d74af7ba9522c26c9.jpg)  
Worldwide Reward (USD) for 60-minute delay (deepseek-v4-pro)

Worldwide Reward (USD) for 60-minute delay (gpt-5.4-mini)  
![](images/6882b042bb1f9cf73e319b536650ced691c950eedc41a2a28ccf235b0a5d9985.jpg)  
Worldwide Reward (USD) for 60-minute delay (glm-5.2)

![](images/acd226635385bbf3bb9a214c748e1c881a354d3c20e371b89d3e5ebb0c5f1aa7.jpg)

![](images/1c9112a078ace2889b289d0ff788caedb99133abc82735df302d1e231967a7b7.jpg)

Worldwide Reward (USD) for 60-minute delay (mistral-large-3)  
![](images/e2f7b8beed1cb526a3fbf878ebf1ac11e3dc126ebf15c3052d772251104a343f.jpg)  
Figure 4: World maps showing the responded worth of time across five LLMs (in US\$) under Prompt 1. Countries with transition values below \$1 and above \$25 are assigned the colors equals to \$1 and \$25, respectively.

## D Robustness Experiments

![](images/e06c79ea63a699cd878c1fc71df1e13f3da3131d05a0ca9e901104b06f179b52.jpg)  
Figure 5: Results across three prompt variations.

As we can see from Table 1, different models show different levels of variations in their responses. We also plot the three prompt variations in Figure 5. Again, GLM-5.2 and DeepSeek-V4- Pro are most robust in their responses, whereas the other three models show substantial variations. In Prompt 1, GPT-5.4-Mini produces an almost flat line, while Prompt 2 and 3 show noticeable differentiation across continents. Conversely, in Prompt 3, Claude-Haiku-4.5 and Mistral-Large-3 accept only compensation above \$25, resulting in a horizontal line above.

To provide a straightforward illustration of how LLM responses look like across the robustness tests, we use China as an example and plot the proportion of “Yes” per reward quantity, as shown in Figure 6. It clearly shows how variations look like for each LLMs across different prompts.

## E Qualitative Results on GDP\_PPP as an Explanatory Variable

Figure 7 illustrate the linear regression relationship between compensation and GDP\_PPP for all LLMs from Prompt 1.

## F Linear Regression Results with Three Indicators (same set of countries)

All three linear regressions are also fitted on the same set of countries (83 in total). The patterns remain unchanged, as shown in Table 7.

Comparison of Model Performance across Prompt Variations (China x claude-haiku-4.5)  
![](images/321df8bd5e3b6088ddadd8b061ee0c4ee2569a7e493f7c324737da0d84fe62b1.jpg)

![](images/2956051f058a0d04cff4cceea8f5a5e6ccfd10b7676a5f961df8d12f770b95ea.jpg)

Comparison of Model Performance across Prompt Variations (China x deepseek-v4-pro)  
![](images/8973494fdabb7d30a8109f0ee83b51a322375f813816f9bc84139733150cf634.jpg)

Comparison of Model Performance across Prompt Variations (China x glm-5.2)  
![](images/b8c74219e12a4cf8d23f9b1c06150d4899ff87188c451f23727d5457156929c3.jpg)

Comparison of Model Performance across Prompt Variations (China x mistral-large-3)  
![](images/1913b4960901d44a006ea76a8316628ab4eb119214b480b601d53190c149920b.jpg)  
Figure 6: Using China as an example, these figures show models’ responses across three prompt variations.

<table><tr><td>Models</td><td> $R ^ { 2 } \colon { \mathrm { G D P } } \_ { \mathrm { P P P } } \left( 8 3 \right)$ </td><td> $R ^ { 2 } \colon$  Mean Income (83)</td><td> $R ^ { 2 } \colon$  Patience (83)</td></tr><tr><td>claude-haiku-4.5</td><td> $0 . 3 4 6 \pm 0 . 3 1 4$ </td><td> $0 . 2 1 3 \pm 0 . 1 8 9$ </td><td> $0 . 0 7 5 \pm 0 . 0 7 5$ </td></tr><tr><td>deepseek-v4-pro</td><td> $0 . 7 1 4 \pm 0 . 0 3 9$ </td><td> $0 . 7 9 6 \pm 0 . 0 0 8$ </td><td> $0 . 3 7 3 \pm 0 . 0 2 7$ </td></tr><tr><td>glm-5.2</td><td> $0 . 6 1 3 \pm 0 . 0 4 0$ </td><td> $0 . 8 1 5 \pm 0 . 0 0 7$ </td><td> $0 . 3 5 6 \pm 0 . 0 1 8$ </td></tr><tr><td>gpt-5.4-mini</td><td> $0 . 3 6 0 \pm 0 . 2 9 6$ </td><td> $0 . 3 6 0 \pm 0 . 2 5 2$ </td><td> $0 . 1 7 8 \pm 0 . 1 3 3$ </td></tr><tr><td>mistral-large-3</td><td> $0 . 3 7 5 ^ { \dagger } \pm 0 . 0 5 0 ^ { \dagger }$ </td><td> $0 . 3 4 6 ^ { \dagger } \pm 0 . 0 3 9 ^ { \dagger }$ </td><td> $0 . 1 0 2 ^ { \dagger } \pm 0 . 0 2 6 ^ { \dagger }$ </td></tr></table>

Note. Mistral-Large-3 is marked by <sup>†</sup> as in Table 1.

Table 7: Linear Regression Results with three indicators: GDP\_PPP, Mean Income, and Patience Score (83 countries).

deepseek-v4-pro

glm-5.2  
![](images/bb5d749403c35ff8347de1a937f31f82ad7acf591c230f2f472e9b3562f8700e.jpg)

![](images/082c7d7fb54eece2b5c8092307f4b82d3345fd53ccc1f4accf88633f791e6e4d.jpg)

![](images/0a0bbfdb997e1383bb0b8ed331a51f9636035cbf0b45f3081f13ca15eaa44675.jpg)

![](images/9992cb3a648003c08f336e03bc4e009abbbc40c52de65e671bf5f370e74f0394.jpg)

![](images/e84e921348b8492ba5f23f5ebc0bda7d8a870f85978bcbbdb3ee9f108493f03e.jpg)  
Figure 7: Scatter plots and linear regressions of country-level compensation against the logarithm of GDP\_PPP across the five LLMs (Prompt 1). Each dot represents a unique country, with colors indicating the continent to which it belongs.