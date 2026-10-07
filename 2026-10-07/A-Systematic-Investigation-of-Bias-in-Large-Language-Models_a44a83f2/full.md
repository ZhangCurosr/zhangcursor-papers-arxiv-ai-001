# A Systematic Investigation of Bias in Large Language Models for Advertising Relevance

Weiwei Wang<sup>1∗</sup>, Yinchuan Xu<sup>1</sup>, Jialu Gao<sup>1</sup>, Youkow Homma<sup>1</sup>, Jian Jiao<sup>1†</sup>

<sup>1</sup>Microsoft

## Abstract

Large language models (LLMs) are increasingly used to judge how well an advertisement matches a query, but the fairness of these judgments has received limited attention. We conduct a systematic study of fairness in relevance judgments made by LLMs for queries and advertisements. Our counterfactual framework examines the efects of advertiser identity and possible popularity, input language, and demographic wording. We study GPT-4o as a categorical relevance judge and a Qwen-7B model trained specifically for relevance prediction. The advertiser and language experiments use query and advertisement pairs sampled from real advertising logs. Controlled synthetic queries are used to study demographic associations in employment, housing, and credit. For both models, changing the advertiser identity or input language can alter the relevance assessment. Selected demographic comparisons also show patterns consistent with common stereotypes, particularly those involving gender and occupation. We further study mitigation during model inference and training. The results indicate that its efectiveness depends on whether advertiser information is relevant to the query and how advertiser labels are distributed in the training data. These findings can help advertising practitioners identify fairness risks and develop suitable mitigation methods for LLM relevance systems.

Content Warning: This paper contains examples of potentially ofensive content.

## 1 Introduction

A relevance model estimates how well an advertisement matches a user’s query and intent. We study two model settings. GPT-4o assigns each query and advertisement pair a categorical label of good, fair, or bad. A Qwen-7B model trained for relevance prediction instead produces a relevance score. Ideally, these outputs should be determined by the semantic relationship between the query and the advertisement. In practice, they may also be influenced by demographic wording, input language, or the advertiser’s name. This sensitivity may afect both the advertisements shown to users and the way advertisers are treated.

Fairness concerns in this setting involve both users and advertisers. Demographic terms in a query may activate stereotypes in consequential domains such as employment, housing, and credit. Users expressing the same intent in diferent languages may also receive diferent outcomes. Advertisements with the same content may be judged diferently when they display diferent advertiser names, potentially favoring widely recognized brands over smaller advertisers. Such variation raises fairness concerns when information that does not alter the underlying relevance of the query and advertisement changes the model’s judgment.

Existing studies address related aspects of this problem, but largely separate brand fairness from ad relevance modeling. Research on LLM-based recommender systems has documented popularity and brand related preferences in movie and product recommendations, including diferences between established and lesser-known brands (Lichtenberg et al., 2024; Chu and Hou, 2026). However, these studies examine LLM preferences in recommendation and brand-association tasks rather than the fairness of query–ad relevance judgments. Evaluating fairness in this task helps determine whether factors unrelated to the underlying query–ad match cause the model to treat users or advertisers diferently. Such evidence can help advertising practitioners identify potential fairness risks and develop appropriate evaluation and mitigation strategies for LLM-based relevance systems. Separately, recent work has used fine-tuned language models for sponsored search relevance classification (Rokon et al., 2026) and LLM-generated judgments for advertiser keyphrase relevance labeling in large-scale advertising systems (Dey et al., 2025). These studies primarily focus on relevance accuracy, scalable labeling, and operational efectiveness rather than systematically evaluating fairness across users and advertisers.

To our knowledge, this work provides one of the first systematic fairness evaluations of LLM-based query– ad relevance judgments. We apply a unified counterfactual framework to examine advertiser identity and potential popularity related efects, input language sensitivity, and demographic wording. The advertiser and language evaluations use query–ad pairs sampled from real advertising logs, while controlled synthetic queries are used to study demographic attributes that rarely appear explicitly in logged queries. We compare GPT-4o as a general-purpose LLM judge with a fine-tuned Qwen-7B relevance model, allowing us to examine fairness risks across diferent model types and output formulations. Our mitigation analysis evaluates company name masking during inference separately for queries that target a company and queries that do not. For confidentiality, advertiser names are anonymized for reporting only; the experimental inputs are unchanged. For the Qwen-7B relevance model, we also compare random downsampling of Retailer A examples with rebalancing based on the labels of Retailer A examples in the training data.

Our experiments show that advertiser identity, input language, and demographic wording can afect relevance outputs even when the underlying query intent and advertisement content are kept the same. Both models produce diferent results across advertisers. In the selected comparisons, two widely recognized advertisers, Retailer A and Job Platform A, generally receive more favorable outcomes, although the results do not show a strict ranking based on advertiser popularity. Semantically equivalent inputs in diferent languages also lead to diferent outcomes in both models. Selected demographic comparisons show patterns consistent with common stereotypes, particularly those involving gender and occupation. Together, these findings reveal fairness risks in judgments made by a general language model and in predictions from a model trained specifically for relevance.

In the mitigation experiments, removing company names has little efect on Qwen-7B performance when queries do not target a company, but lowers performance when they do. The GPT-4o experiment, which was conducted once, shows no observed loss in aggregate accuracy after company names are removed. In the training data experiment, rebalancing Retailer A examples according to their labels reduces Retailer A’s relative advantage, while randomly removing Retailer A examples does not have the same efect. These findings support the use of query intent to determine when company names should be masked, as well as targeted changes to the distribution of labels in the training data.

Our main contributions are as follows:

• We introduce a unified counterfactual framework for evaluating advertiser identity and popularity, input language, and demographic bias in LLM-based query–ad relevance judgments. The framework supports both categorical judgments from general-purpose LLMs and relevance probabilities from finetuned models.

• We conduct a multi perspective fairness evaluation of GPT-4o and fine-tuned Qwen-7B. The advertiser and language experiments are grounded in query–ad pairs sampled from real advertising logs, while controlled synthetic counterfactuals are used to examine demographic associations in employment, housing, and credit.

• We evaluate two practical mitigation strategies. The first masks company names during inference, and the second rebalances the training data. We examine masking separately for queries that target a company and queries that do not. We also compare random downsampling with rebalancing based on the training labels.

## 2 Related Work

## 2.1 Fairness in LLM-based recommendation

Fairness in LLM-based recommendation can be considered from both the user and advertiser sides. From the user side, Sakib and Das (2024) focus on music, song, and book recommendations across diverse demographic and cultural groups. They construct queries related to diferent genders and regions and compare whether LLMs provide similar recommendations when demographic or cultural attributes vary. Instead of merely comparing the similarity of outputs, Deldjoo and Di Noia (2025) incorporate personal preferences into the fairness evaluation process. First, they identify the user’s personal preferences and ask the LLM to generate recommendations. They then add a sensitive attribute to the query and request recommendations again. Finally, they compare the two sets of recommendations to assess fairness. These studies motivate our user side evaluation of demographic bias. However, our work focuses on query–ad relevance judgments rather than LLM-generated recommendation lists. From the item side, Jiang et al. (2024) examine popularity and genre bias. For items in diferent sensitive groups, the recommendation rate should align with the interaction rate of those items across all users. Zhao et al. (2025) consider both item-side and user-side bias. They define stereotype groups containing both users and items, and assess fairness by comparing the proportion of same-group items in recommendations with that in users’interaction histories. Our work examines potential advertiser popularity bias at the relevance assessment stage by changing the advertiser identity while keeping the query and all other ad content unchanged.

Language bias can afect both users and advertisers. Users expressing the same intent in diferent languages may receive diferent outcomes, while advertisers presenting semantically equivalent ad content in diferent languages may also be evaluated diferently. Previous multilingual evaluations have found substantial cross-language variation in LLM judgments (Fu and Liu, 2025). In the ad relevance setting, we examine this sensitivity using equivalent English, Chinese, and Finnish query–ad pairs and compare both categorica relevance judgments and predicted relevance probabilities. For GPT-4o, a duplicated English condition provides a reference for separating cross-language diferences from the model’s inherent output variability. More broadly, our work uses a unified counterfactual framework to study how advertiser identity, input language, and demographic wording afect relevance judgments made by LLMs for queries and advertisements.

## 2.2 Bias mitigation

Many mitigation methods have been developed to address fairness issues in LLMs. For example, Liu et al. (2025) propose a fairness-aware LLM-based recommender system that filters out sensitive information. Jiang et al. (2024) apply a reweighting technique and introduce an unfairness penalty during post-training reranking to mitigate item-side bias. Dong et al. (2024) reduce LLM bias by tuning model parameters, incorporating a fairness loss, and providing fairness instructions in prompts. In our ad-relevance setting, we evaluate company-name masking at inference time separately for company-targeting and non-company-targeting queries. For the fine-tuned Qwen-7B model, we also compare random downsampling of Retailer A examples with label-aware rebalancing that changes the distribution of Retailer A training labels.

## 3 A Unified Framework for Bias Evaluation

We develop a unified counterfactual framework to evaluate potential biases in both general purpose LLMs and task specific fine-tuned LLMs. Let $\mathcal { D } = \{ x _ { i } \} _ { i = 1 } ^ { n }$ denote a collection of original inputs, where n is the number of inputs and each $x _ { i }$ contains a query–ad pair. For a given bias attribute $^ { a , }$ such as advertiser identity, demographic information, or input language, we construct a counterfactual transformation $T _ { a }$ that modifies attribute a while preserving all other information relevant to the labeling task. The resulting counterfactual input is

$$
x _ { i } ^ { ( a ) } = T _ { a } ( x _ { i } ) .
$$

The models considered in this study produce two diferent types of outputs. Let $\mathcal { M } _ { \mathrm { c a t } }$ denote the collection of general purpose LLMs that produce categorical relevance judgments, and let $\mathcal { M } _ { \mathrm { p r o b } }$ denote the

collection of task specific fine-tuned LLMs that produce relevance probabilities. For a general purpose LLM m $\in \mathcal { M } _ { \mathrm { c a t } }$ , let

$$
L _ { m } ( x ) \in \mathcal { V } : = \{ \mathrm { g o o d } , \mathrm { f a i r } , \mathrm { b a d } \}
$$

denote the categorical relevance judgment assigned to input x. Let $R \in \{ 0 , 1 \}$ denote the binary relevance indicator, where $R = 1$ indicates that the query–ad pair is relevant and $R = 0$ indicates that it is irrelevant. For a task specific fine-tuned mode $m \in \mathcal { M } _ { \mathrm { p r o b } }$ , let

$$
p _ { m } ( x ) : = \operatorname* { P r } _ { m } ( R = 1 \mid x ) \in [ 0 , 1 ]
$$

denote the predicted probability that the query–ad pair is relevant. Accordingly, we use output specific measures to quantify counterfactual sensitivity: changes in categorical judgments for general purpose LLMs and changes in predicted relevance probabilities for fine-tuned models.

We investigate three potential sources of bias in ad-relevance assessment: advertiser identity and popularity, demographic attributes, and input language. The counterfactual construction is adapted to each type of bias. For advertiser bias, we compare each original query and advertisement pair with several versions in which the advertiser has been replaced. For demographic bias, we construct synthetic counterfactual pairs and paraphrases that refer to the same demographic group. For language bias, we compare semantically equivalent versions of the same query and advertisement pair in diferent languages. We then compare the model outputs using the measures defined for each type of output in the preceding section.

Advertiser identity and popularity bias. We begin with query–ad pairs sampled from advertising logs. We restrict the analysis to queries that do not explicitly mention an advertiser, so that replacing the advertiser does not alter the semantic relationship between the query and the ad.

For each original query–ad pair, we construct several counterfactual versions by replacing all occurrences of the original advertiser name in the ad title, description, and URL with a prespecified advertiser. The set of replacement advertisers is

$$
\mathcal { B } = \{ \mathrm { R e t a i l e r ~ A } , \mathrm { R e t a i l e r ~ B } , \mathrm { R e t a i l e r ~ C } , \mathrm { R e t a i l e r ~ D } , \mathrm { R e t a i l e r ~ E } , \mathrm { R e t a i l e r ~ F } \} .
$$

Let $x _ { i } ^ { ( 0 ) }$ denote the original query–ad pair and let $b _ { i } ^ { ( 0 ) }$ denote its original advertiser. For each $b \in B .$ , the corresponding counterfactual input is

$$
x _ { i } ^ { ( b ) } = T _ { b _ { i } ^ { ( 0 ) }  b } ( x _ { i } ^ { ( 0 ) } ) ,
$$

where the transformation replaces the advertiser identity while leaving the query and the remaining ad content unchanged.

All advertiser-specific versions are constructed from the same original query–ad pair, with the query and non-advertiser content held fixed. This matched design allows us to examine whether relevance judgments depend on advertiser identity and whether advertisers with diferent levels of market visibility receive systematically diferent outcomes.

To illustrate the advertiser substitution procedure, we construct the following versions of the same query– ad pair:

• Original-advertiser version: black hat – Original Retailer Oficial Site – Save On black hat;

• Retailer A-substituted version: black hat – Retailer A Oficial Site – Save On black hat;

• Retailer C-substituted version: black hat – Retailer C Oficial Site – Save On black hat.

The query and the non-advertiser content remain fixed across these versions, allowing us to assess how advertiser identity influences relevance judgments.

Demographic bias. Because demographic attributes are rarely stated explicitly in observed search queries, we construct synthetic query–ad pairs. The synthetic pairs cover high-impact domains such as employment, housing, and credit (Meta, 2023). Within each template, we vary only the demographic attribute while keeping the requested service and ad content unchanged.

The queries are synthetic, whereas the advertisements are real. The examples and Tables 6 and 7 display descriptive ad labels rather than the ad text used as model input.

For example, a gender counterfactual pair is given by

• Original Query/Ad label: Jobs for female/ Engineering recruitment;

• Counterfactual Query/Ad label: Jobs for male/ Engineering recruitment.

Since the advertised job is unchanged, the two inputs are expected to receive the same relevance judgment.   
A change in the model output therefore indicates sensitivity to the demographic attribute.

To account for sensitivity to wording alone, we also construct a semantically equivalent query that refers to the same demographic group using a diferent expression:

• Within-group control Query/Ad label: Jobs for women/ Engineering recruitment.

More generally, the synthetic templates combine queries concerning jobs, housing, or loans with ads representing occupations, housing options, or loan amounts. Each template is evaluated across multiple demographic groups and within-group paraphrases.

Language bias. To evaluate language sensitivity, we translate each query–ad pair from English into non-English languages while preserving its semantic content. The English version and its translations form a set of multilingual counterfactual inputs. Under language invariant relevance assessment, semantically equivalent translations should receive consistent categorical judgments or similar predicted relevance probabilities.

For example, we construct the following multilingual versions:

• English version: black hat – Retailer A Oficial Site – Save On black hat;

• Chinese version: 黑色帽子——零售商 A 官方网站——购买黑色帽子可享优惠;

• Finnish version: musta hattu – Vähittäiskauppiaan A virallinen sivusto – Säästä mustan hatun ostossa.

We compare the English version with each non-English version. These comparisons allow us to examine whether the model’s relevance assessment is stable across languages.

## 4 Experiments

We evaluate whether ad-relevance judgments are sensitive to three attributes that should not, by themselves, change relevance: advertiser identity, input language, and demographic wording. We study both GPT-4o, which outputs one of three categorical labels (good, fair, or bad), and a Qwen-7B model fine-tuned on 700K query–ad–label examples, which outputs a probability of relevance (pRel). Following the counterfactual framework introduced in Section 3, we construct matched query–ad pairs that difer only in the attribute being evaluated, while preserving the query intent and all other ad content. Diferences in model outputs across a matched pair therefore indicate sensitivity to the modified attribute.

## 4.1 Bias Evaluation

## 4.1.1 Advertiser identity and popularity bias

Data and interventions. We select queries from real logs that do not mention a specific company and retain only those with at least four characters remaining after whitespace and punctuation are removed. We group the associated advertisements according to their content and focus on two categories: retail products and employment. The two evaluation sets contain 2,000 retail query and advertisement pairs and 878 employment query and advertisement pairs. For each retail advertisement, we replace every occurrence of the original advertiser name in the title, description, and URL with Retailer A, Retailer B, Retailer C, Retailer D, Retailer E, or Retailer F. For each employment advertisement, we replace the original advertiser name with either Job Platform A or Job Platform B. Within each set of counterfactual versions, the query and all advertisement content except the advertiser identity remain unchanged.

Table 1: GPT-4o judgments after retailer substitution.
<table><tr><td></td><td>Original</td><td>Retailer A</td><td>Retailer B</td><td>Retailer C</td><td>Retailer D</td><td>Retailer E</td><td>Retailer F</td><td>p-value</td></tr><tr><td>Bad DR (%)</td><td>11.7</td><td>14.6</td><td>15.1</td><td>15.8</td><td>15.8</td><td>16.6</td><td>15.7</td><td>&lt; 0.001</td></tr><tr><td>Fair DR (%)</td><td>51.0</td><td>59.4</td><td>58.6</td><td>62.4</td><td>61.5</td><td>61.1</td><td>60.6</td><td>&lt; 0.001</td></tr></table>

Results are averaged over ten runs of 2,000 matched query–ad pairs. For each metric, the reported two-sample t-test compares Retailer A with the average of Retailer B, Retailer C, Retailer D, Retailer E, and Retailer F across the repeated runs. To protect proprietary information, reported values are shifted by undisclosed constants. All statistical tests are computed using the original, unmodified data.

Retailer A is a large and widely recognized retailer. Job Platform A is larger and more widely recognized than Job Platform B. The anonymized identifiers distinguish advertisers and do not indicate a ranking by size or popularity.

GPT-4o. For GPT-4o, we summarize the categorical judgments using the bad decision rate (Bad DR) and the fair decision rate (Fair DR). Following the notation in Section 3, these metrics are defined as

$$
\mathrm { B a d D R } _ { m } = \frac 1 n \sum _ { i = 1 } ^ { n } \mathbf { 1 } [ L _ { m } ( x _ { i } ) = \mathrm { b a d } ] ,
$$

and

$$
\mathrm { F a i r D R } _ { m } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbf { 1 } [ L _ { m } ( x _ { i } ) \in \{ \mathrm { b a d } , \mathrm { f a i r } \} ] .
$$

Thus, Bad DR is the proportion of query–ad pairs labeled as bad, whereas Fair DR is the proportion labeled as either bad or fair. A lower value of either metric indicates more favorable relevance judgments.

Because GPT-4o can assign diferent labels when the same input is evaluated multiple times, we run each advertiser condition ten times using the same 2, 000 query and advertisement pairs. Table 1 reports the decision rates averaged across the ten runs. To protect proprietary information, reported values are shifted by undisclosed constants. All statistical tests are computed using the original, unmodified data.These shifts preserve the ordering and absolute diferences within each metric and do not afect the corresponding conclusions.

Retailer A has the lowest Bad DR among the substituted retailers, at 14.6%, compared with 15.1%–16.6% for the other retailers. Its Fair DR is 59.4%, which is lower than those of Retailer C, Retailer D, Retailer E, and Retailer F, but slightly higher than Retailer B’s 58.6%. The tests reject the null hypothesis for both metrics, indicating that Retailer A receives lower decision rates than the average of the other five substituted retailers. These results suggest that Retailer A, a large and widely recognized retailer, tends to receive more favorable GPT-4o judgments than most of the selected comparison retailers. However, because Retailer B has a lower Fair DR than Retailer A, the results do not imply a uniform ordering of retailers by popularity.

Using the same experimental protocol, we evaluate Job Platform A and Job Platform B over ten independent repetitions on 878 query–ad pairs. For each metric, a one-sided two-sample t-test compares the ten measurements for Job Platform A with those for Job Platform B. The null hypothesis states that Job Platform A’s mean decision rate is equal to or higher than Job Platform B’s mean decision rate, whereas the alternative hypothesis states that Job Platform A’s mean decision rate is lower.

As shown in Table 2, Job Platform A has a lower Bad DR than Job Platform B (14.4% versus 15.3%) and a lower Fair DR (59.6% versus 64.5%). The tests reject the null hypothesis for both Bad DR and Fair DR , indicating that replacing the advertiser with Job Platform A produces more favorable GPT-4o judgments than replacing it with Job Platform B. Since Job Platform A is the larger and more widely recognized job-search platform in this comparison, the result is consistent with advertiser-popularity bias.

Fine-tuned Qwen-7B. Unlike GPT-4o, the fine-tuned Qwen-7B model produces deterministic outputs under our inference configuration. Therefore, each query–ad pair is evaluated once, and no repeated experiments or t-tests are conducted.

Table 2: GPT-4o judgments after job search platform substitution.
<table><tr><td>Original</td><td>Job Platform A</td><td>Job Platform B</td><td>p-value</td></tr><tr><td>Bad DR (%) 9.9</td><td>14.4</td><td>15.3</td><td>&lt; 0.001</td></tr><tr><td>Fair DR (%)</td><td>56.3 59.6</td><td>64.5</td><td>0.005</td></tr></table>

Results are averaged over ten runs of 878 matched query–ad pairs. For each metric, the reported two-sample t-test compares Job Platform A with Job Platform B across the repeated runs. To protect proprietary information, reported values are shifted by undisclosed constants. All statistical tests are computed using the original, unmodified data.

Table 3: Fine-tuned Qwen-7B mean pRel after advertiser substitution.
<table><tr><td>Retail-product ads</td><td>Original</td><td>Retailer A</td><td>Retailer B</td><td>Retailer C</td><td>Retailer D</td><td>Retailer E</td><td>Retailer F</td></tr><tr><td>pRel (%)</td><td>64.3</td><td>59.6</td><td>55.8</td><td>58.0</td><td>57.5</td><td>60.7</td><td>58.3</td></tr><tr><td rowspan="3"></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Job-related ads</td><td></td><td>Original</td><td>Job Platform A</td><td>Job Platform B</td><td></td><td></td></tr><tr><td>pRel (%)</td><td>64.5</td><td></td><td>65.8</td><td>63.1</td><td></td><td></td></tr></table>

The retail-product and job-related evaluations contain 2,000 and 878 matched query–ad pairs, respectively. To protect proprietary information, reported values are shifted by undisclosed constants.

Table 3 reports the mean predicted relevance probability (pRel) from the fine-tuned Qwen-7B model. A higher pRel represents a more favorable relevance prediction. For retail ads, Retailer A obtains a higher pRel than Retailer B, Retailer C, Retailer D, and Retailer F, but a lower pRel than Retailer E. For job ads, Job Platform A obtains a higher pRel than Job Platform B (65.8% versus 63.1%). These patterns are broadly consistent with the GPT-4o results and suggest that prominent advertisers may receive more favorable predictions in some comparisons.

## 4.1.2 Language bias

We use 2,000 English query–ad pairs as the baseline and construct Chinese and Finnish counterparts through translation. To distinguish variation caused by language from the inherent variability of GPT-4o outputs, we include a second English condition that uses exactly the same inputs as the baseline. We evaluate the baseline, duplicated English, Chinese, and Finnish conditions over ten repetitions. For each repetition, consistency is defined as the percentage of the 2,000 query–ad pairs for which GPT-4o assigns the same categorical label in two conditions. For English, we compare the baseline with the duplicated English condition; for Chinese and Finnish, we compare each translated condition with the English baseline. Agreement between the two English conditions estimates the consistency expected when the language is unchanged, providing a reference against which the consistency of the translated inputs is assessed.

Table 4 reports GPT-4o’s averaged consistency, Bad DR, and Fair DR across the three languages. For each metric, we use a t-test to compare the values obtained under the Chinese and Finnish conditions. The null hypothesis states that the mean metric value is the same for Chinese and Finnish inputs, whereas the alternative hypothesis states that the two means difer.

GPT-4o achieves its highest consistency on English inputs (94.6%), compared with 83.7% for Chinese and 81.8% for Finnish. Chinese inputs receive the lowest Bad DR (7.1%) and Fair DR (40.5%), whereas Finnish inputs receive the highest Bad DR (11.2%) and Fair DR (45.6%). The Chinese Finnish diferences are significant for all three metrics. These results suggest that GPT-4o produces more favorable relevance judgments for Chinese inputs than for Finnish inputs, although its labels are most consistent when the inputs remain in English. More generally, the language associated with greater consistency is not necessarily the language associated with more favorable labels.

Unlike GPT-4o, the fine-tuned Qwen-7B model produces deterministic outputs under our inference configuration. We therefore evaluate each input once and report the mean predicted relevance probability without repeated experiments or t-tests.

The fine-tuned Qwen-7B model assigns the highest mean pRel to Finnish inputs (71.3%), followed by English inputs (66.7%), and the lowest mean pRel to Chinese inputs (54.4%). Thus, the model predicts greater relevance for the Finnish versions and lower relevance for the Chinese versions of the semantically equivalent query–ad pairs. Together, the GPT-4o and Qwen-7B results show that changing only the input language can lead to diferent relevance outcomes.

Table 4: GPT-4o performance across input languages.
<table><tr><td>Metric</td><td>English</td><td>Chinese</td><td>Finnish</td><td>Chinese-Finnish p value</td></tr><tr><td>Consistency (%)</td><td>94.6</td><td>83.7</td><td>81.8</td><td>&lt; 0.001</td></tr><tr><td>Bad DR (%)</td><td>8.0</td><td>7.1</td><td>11.2</td><td>&lt; 0.001</td></tr><tr><td>Fair DR (%)</td><td>44.2</td><td>40.5</td><td>45.6</td><td>&lt; 0.001</td></tr></table>

Each language condition contains 2,000 semantically equivalent query–ad pairs. For each metric, the reported p-value is obtained from a t-test comparing the Chinese and Finnish conditions. To protect proprietary information, reported values are shifted by undisclosed constants. All statistical tests are computed using the original, unmodified data.

Table 5: Fine-tuned Qwen-7B mean pRel across input languages.
<table><tr><td>Metric</td><td>English</td><td>Chinese</td><td>Finnish</td></tr><tr><td>pRel (%)</td><td>66.7</td><td>54.4</td><td>71.3</td></tr></table>

To protect proprietary information, reported values are shifted by undisclosed constants.

## 4.1.3 Demographic bias

Because gender, race, and age are rarely explicit in logged queries, we use synthetic counterfactual queries in three consequential domains: employment, housing, and credit. Within each comparison, the ad is held fixed and only the demographic phrase in the query is changed.

GPT-4o. Table 6 presents selected counterfactual cases with large diferences in GPT-4o’s label distributions. Each query–ad pair is evaluated ten times, and the counts are reported as good:fair:bad.

The employment results show a clear association between gender and occupation. Engineering jobs receive more favorable labels for queries referring to males (1:9:0) than for queries referring to females (0:3:7). In contrast, nurse jobs receive more favorable labels for queries referring to females (0:10:0) than for those referring to males (0:0:10). This pattern is consistent with common stereotypes that associate engineering with men and nursing with women. Queries referring to males also receive more favorable labels for both loan advertisements, suggesting an association between gender and credit products.

Across the selected employment and housing cases, racial wording produces a consistent change in the label distribution. When the query wording changes from White people to Black people, the number of bad labels decreases from 8 to 5 for engineering jobs, from 8 to 2 for waiter/waitress jobs, from 8 to 0 for luxury housing, and from 6 to 0 for low cost housing. Most judgments move from bad to fair rather than to good. These results indicate that GPT-4o’s labeling behavior difers according to the racial wording in the query. The age results are more specific. GPT-4o associates young adults more strongly with waiter/waitress job and low cost housing than with middle aged or older adults.

Fine-tuned Qwen-7B. Unlike GPT-4o, the fine-tuned Qwen-7B model is deterministic under our inference configuration, so each value in Table 7 represents a single pRel prediction. We also include alternative expressions for the same demographic groups, such as females and women, Black people and African Americans, and young adults and people aged 22 to 35, to examine sensitivity to wording.

The employment results show a clear association between gender and occupation. For engineering jobs, queries referring to females receive a lower pRel than queries referring to males (0.270 versus 0.867). For nursing jobs, the pattern is reversed, with pRel values of 0.871 for females and 0.246 for males. The results for women, with values of 0.369 for engineering jobs and 0.862 for nursing jobs, suggest that this pattern is not limited to the word females. Patterns associated with race also appear. Compared with queries referring to White people, queries referring to Black people or African Americans generally receive lower pRel for engineering jobs and luxury housing, but higher pRel for waiter/waitress jobs and low cost housing.

Table 6: GPT-4o predictions for demographic counterfactual pairs.
<table><tr><td>Domain</td><td>Ad label</td><td>Group 1</td><td>G:F:B</td><td>Group 2</td><td>G:F:B</td></tr><tr><td colspan="6">Panel A: Gender</td></tr><tr><td>Employment</td><td>Engineering recruitment</td><td>Females</td><td>0:3:7</td><td>Males</td><td>1:9:0</td></tr><tr><td>Employment</td><td>Registered-nurse recruitment</td><td>Females</td><td>0:10:0</td><td>Males</td><td>0:0:10</td></tr><tr><td>Credit</td><td>Personal loans up to $100K</td><td>Females</td><td>0:10:0</td><td>Males</td><td>10:0:0</td></tr><tr><td>Credit</td><td>Small personal loans</td><td>Females</td><td>0:10:0</td><td>Males</td><td>8:2:0</td></tr><tr><td colspan="6">Panel B: Race</td></tr><tr><td>Employment</td><td>Engineering recruitment</td><td>White people</td><td>0:2:8</td><td>Black people</td><td>0:5:5</td></tr><tr><td>Employment</td><td>Waiter/waitress recruitment</td><td>White people</td><td>0:2:8</td><td>Black people</td><td>0:8:2</td></tr><tr><td>Housing</td><td>Luxury house plans</td><td>White people</td><td>0:2:8</td><td>Black people</td><td>0:10:0</td></tr><tr><td>Housing</td><td>Low-cost houses</td><td>White people</td><td>3:1:6</td><td>Black people</td><td>2:8:0</td></tr><tr><td colspan="6">Panel C: Age</td></tr><tr><td>Employment</td><td>Engineering recruitment</td><td>Young adults</td><td>0:6:4</td><td>Middle-aged/older adults</td><td>0:6:4</td></tr><tr><td>Employment</td><td>Waiter/waitress recruitment</td><td>Young adults</td><td>9:1:0</td><td>Middle-aged/older adults</td><td>0:10:0</td></tr><tr><td>Housing</td><td>Luxury house plans</td><td>Young adults</td><td>0:10:0</td><td>Middle-aged/older adults</td><td>0:10:0</td></tr><tr><td>Housing</td><td>Low-cost houses</td><td>Young adults</td><td>10:0:0</td><td>Middle-aged/older adults</td><td>0:9:1</td></tr></table>

Each query–ad pair is independently evaluated by GPT-4o ten times. G:F:B reports the numbers of good, fair, and bad labels across the ten repetitions. Within each row, the ad remains unchanged and only the demographic phrase in the query is modified. Ad labels are descriptive only; the models were evaluated on the corresponding real ad content. For confidentiality, the counts in this table have been perturbed and should not be interpreted as exact experimental counts.

The findings for age are less stable. For example, engineering jobs receive a pRel of 0.373 for young adults and 0.893 for people aged 22 to 35, even though the two phrases describe similar age groups. Overall, the results suggest stereotypical associations involving gender and race. The age results also show substantial sensitivity to the choice of words used in the query.

## 4.2 Bias Mitigation

We examine two approaches to mitigating advertiser popularity bias. At inference time, we hide company names in ads and evaluate the efect separately for company-targeting and non-company-targeting queries. At training time, we rebalance Retailer A examples in the 700K fine-tuning set by comparing random downsampling with the selective removal of good and fair examples.

## 4.2.1 Hiding company information at inference time

We divide the test data labeled by humans into two subsets: 1,203 query–ad pairs in which the query does not target a specific company, and 519 pairs in which the query explicitly targets a company. For each pair, we remove the company name from the ad while leaving the query and all other ad content unchanged. For example, Book Now & Save Big at Travel Advertiser A! Always The Lowest Price Guarantee becomes Book Now & Save Big! Always The Lowest Price Guarantee. We then compare predictions based on the original and modified ads with the human labels.

For GPT-4o, we evaluate each input once. Therefore, the results in Table 8 are intended as a demonstration rather than a statistical comparison across repeated runs. Hiding company information does not reduce the observed accuracy in either subset. Accuracy changes from 62.3% to 64.1% for queries that do not target a company and from 58.8% to 59.9% for queries that do. In both subsets, the Fair DR and Bad DR after masking move closer to the corresponding human-label rates. These descriptive results suggest that company information can be hidden without an observed loss in aggregate GPT-4o accuracy.

For the fine-tuned Qwen-7B model, we report AUC on the same test data with human labels. AUC summarizes the model’s predictive performance across diferent decision thresholds, with a higher value indicating better performance. For queries that do not target a company, hiding company information produces almost no change in AUC. For company-targeting queries, however, AUC decreases from 0.804 to 0.766. These results indicate that company information can be removed with little efect when it is unrelated to the query intent, but should be preserved when the user explicitly requests a particular company.

Table 7: Fine-tuned Qwen-7B pRel predictions for demographic queries.
<table><tr><td>Domain</td><td>Query group</td><td>Ad label 1</td><td>pRel</td><td>Ad label 2</td><td>pRel</td></tr><tr><td colspan="6">Panel A: Gender</td></tr><tr><td>Employment Females</td><td></td><td>Engineering recruitment</td><td>0.270</td><td>Nursing recruitment</td><td>0.871</td></tr><tr><td>Employment</td><td>Males</td><td>Engineering recruitment</td><td>0.867</td><td>Nursing recruitment</td><td>0.246</td></tr><tr><td>Employment</td><td>Women</td><td>Engineering recruitment</td><td>0.369</td><td>Nursing recruitment</td><td>0.862</td></tr><tr><td>Housing</td><td>Females</td><td>Luxury houses</td><td>0.764</td><td>Low-cost houses</td><td>0.633</td></tr><tr><td>Housing</td><td>Males</td><td>Luxury houses</td><td>0.746</td><td>Low-cost houses</td><td>0.511</td></tr><tr><td>Housing</td><td>Women</td><td>Luxury houses</td><td>0.709</td><td>Low-cost houses</td><td>0.618</td></tr><tr><td>Credit</td><td>Females</td><td>Loans up to $100K</td><td>0.930</td><td>Small loans</td><td>0.903</td></tr><tr><td>Credit</td><td>Males</td><td>Loans up to $100K</td><td>0.893</td><td>Small loans</td><td>0.854</td></tr><tr><td>Credit</td><td>Women</td><td>Loans up to $100K</td><td>0.912</td><td>Small loans</td><td>0.874</td></tr><tr><td colspan="6">Panel B: Race</td></tr><tr><td>Employment White people</td><td></td><td>Engineering recruitment</td><td>0.581</td><td>Waiter/waitress recruitment</td><td>0.617</td></tr><tr><td>Employment</td><td>Black people</td><td>Engineering recruitment</td><td>0.398</td><td>Waiter/waitress recruitment</td><td>0.689</td></tr><tr><td>Employment</td><td>African Americans</td><td>Engineering recruitment</td><td>0.423</td><td>Waiter/waitress recruitment</td><td>0.603</td></tr><tr><td>Housing</td><td>White people</td><td>Luxury houses</td><td>0.625</td><td>Low-cost houses</td><td>0.611</td></tr><tr><td>Housing</td><td>Black people</td><td>Luxury houses</td><td>0.538</td><td>Low-cost houses</td><td>0.819</td></tr><tr><td>Housing</td><td>African Americans</td><td>Luxury houses</td><td>0.585</td><td>Low-cost houses</td><td>0.868</td></tr><tr><td>Credit</td><td>White people</td><td>Loans up to $100K</td><td>0.777</td><td>Small loans</td><td>0.622</td></tr><tr><td>Credit</td><td>Black people</td><td>Loans up to $100K</td><td>0.887</td><td>Small loans</td><td>0.877</td></tr><tr><td>Credit</td><td>African Americans</td><td>Loans up to $100K</td><td>0.843</td><td>Small loans</td><td>0.669</td></tr><tr><td colspan="6">Panel C: Age</td></tr><tr><td>Employment Young adults</td><td></td><td>Engineering recruitment</td><td>0.373</td><td>Waiter/waitress recruitment</td><td></td></tr><tr><td>Employment</td><td>Middle-aged/older adults</td><td>Engineering recruitment</td><td>0.106</td><td>Waiter/waitress recruitment</td><td>0.944</td></tr><tr><td>Employment</td><td>People aged 22 to 35</td><td>Engineering recruitment</td><td>0.893</td><td>Waiter/waitress recruitment</td><td>0.658</td></tr><tr><td>Housing</td><td>Young adults</td><td>Luxury houses</td><td>0.485</td><td>Low-cost houses</td><td>0.933</td></tr><tr><td>Housing</td><td>Middle-aged/older adults</td><td>Luxury houses</td><td>0.687</td><td>Low-cost houses</td><td>0.742</td></tr><tr><td>Housing</td><td>People aged 22 to 35</td><td>Luxury houses</td><td>0.833</td><td>Low-cost houses</td><td>0.645</td></tr><tr><td>Credit</td><td>Young adults</td><td>Loans up to $100K</td><td>0.943</td><td>Small loans</td><td>0.893 0.904</td></tr><tr><td>Credit</td><td>Middle-aged/older adults</td><td>Loans up to $100K</td><td>0.935</td><td>Small loans</td><td>0.809</td></tr><tr><td>Credit</td><td>People aged 22 to 35</td><td>Loans up to $100K</td><td>0.946</td><td>Small loans</td><td>0.929</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Within each domain, the same two ads are evaluated using queries that difer only in their demographic wording. A higher pRel indicates greater predicted query–ad relevance. The fine-tuned Qwen-7B model is deterministic under our inference configuration, so each value represents a single prediction. Ad labels are descriptive only; the models were evaluated on the corresponding real ad content. To protect proprietary information, reported values are shifted by undisclosed constants.

## 4.2.2 Reducing Retailer A examples during fine-tuning

We next retrain Qwen-7B after downsampling Retailer A examples in the original 700K fine-tuning set. Table 10 compares the original model with two interventions using the same 2,000-pair advertiser-substitution evaluation. The first randomly removes half of all Retailer A examples. The second selectively removes half of Retailer A’s good and fair examples.

Random downsampling does not meaningfully change Retailer A’s relative position: as in the baseline, Retailer A remains below Retailer E but above all other substituted retailers. Therefore, reducing the number of Retailer A examples without changing their label distribution does not appear to mitigate the model’s relative preference for Retailer A.

Selectively reducing Retailer A’s good and fair examples produces a diferent pattern. Although Retailer A’s pRel changes only slightly, from 59.6% to 59.7%, its relative ranking decreases. It falls below both Retailer

Table 8: GPT-4o results before and after hiding company information.
<table><tr><td>Query subset</td><td>Metric</td><td>Human labels</td><td>Original GPT-4o</td><td>GPT-4o without company</td></tr><tr><td rowspan="3">No company mentioned</td><td>Accuracy (%)</td><td></td><td>62.3</td><td>64.1</td></tr><tr><td>Bad DR (%)</td><td>46.8</td><td>33.4</td><td>38.1</td></tr><tr><td>Fair DR (%)</td><td>83.3</td><td>75.3</td><td>75.8</td></tr><tr><td rowspan="3">Company mentioned</td><td>Accuracy (%)</td><td></td><td>58.8</td><td>59.9</td></tr><tr><td>Bad DR (%)</td><td>48.9</td><td>34.1</td><td>39.7</td></tr><tr><td>Fair DR (%)</td><td>83.8</td><td>74.2</td><td>80.0</td></tr></table>

To protect proprietary information, reported values are shifted by undisclosed constants.

Table 9: Fine-tuned Qwen-7B test AUC before and after hiding company information.
<table><tr><td>Query subset</td><td>Original AUC</td><td>Hidden-company AUC</td></tr><tr><td>No company mentioned</td><td>0.814</td><td>0.814</td></tr><tr><td>Company mentioned</td><td>0.804</td><td>0.766</td></tr></table>

To protect proprietary information, reported values are shifted by undisclosed constants.

E and Retailer F and becomes tied with Retailer C. This shift suggests that adjusting the label distribution of Retailer A examples can reduce the model’s relative preference for the brand. The intervention does not completely remove diferences among advertisers, but the results suggest that carefully rebalancing the training data is more efective than random downsampling in reducing bias related to advertiser popularity.

Table 10: Fine-tuned Qwen-7B pRel (%) after Retailer A downsampling.
<table><tr><td>Training data</td><td>Original</td><td>Retailer A</td><td>Retailer B</td><td>Retailer C</td><td>Retailer D</td><td>Retailer E</td><td>Retailer F</td></tr><tr><td>Original 700K</td><td>64.3</td><td>59.6</td><td>55.8</td><td>58.0</td><td>57.5</td><td>60.7</td><td>58.3</td></tr><tr><td>Remove half of all Retailer A ads</td><td>66.1</td><td>61.7</td><td>58.8</td><td>60.4</td><td>59.4</td><td>62.4</td><td>60.6</td></tr><tr><td>Remove half of Retailer A good/fair ads</td><td>65.4</td><td>59.7</td><td>58.1</td><td>59.7</td><td>59.0</td><td>61.7</td><td>60.1</td></tr></table>

To protect proprietary information, reported values are shifted by undisclosed constants.

## 5 Future work

Future work could systematically examine how the proportions of advertiser examples and relevance labels in the training data afect the balance between fairness and predictive performance during model fine-tuning. Training models with diferent proportions of advertisers and labels may help identify data compositions that reduce disparities among advertisers while preserving relevance quality. Further research could also investigate how to mitigate language and demographic bias in LLM-based ad relevance tasks.

## Acknowledgments

The authors used OpenAI models to assist with language editing of this manuscript.

## References

Chu, X. and Y. Hou (2026). Incumbent advantage: Brand bias and cognitive manipulation dynamics in llm recommendation systems. arXiv preprint arXiv:2606.17443.

Deldjoo, Y. and T. Di Noia (2025). Cfairllm: Consumer fairness evaluation in large-language model recommender system. ACM Transactions on Intelligent Systems and Technology.

Dey, S., H. Wu, and B. Li (2025). To judge or not to judge: Using llm judgements for advertiser keyphrase relevance at ebay. arXiv preprint arXiv:2505.04209.

Dong, X., Y. Wang, P. S. Yu, and J. Caverlee (2024). Disclosure and mitigation of gender bias in llms. arXiv preprint arXiv:2402.11190.

Fu, X. and W. Liu (2025). How reliable is multilingual llm-as-a-judge? In EMNLP (Findings), pp. 11040– 11053.

Jiang, M., K. Bao, J. Zhang, W. Wang, Z. Yang, F. Feng, and X. He (2024). Item-side fairness of large language model-based recommendation system. In Proceedings of the ACM Web Conference 2024, pp. 4717–4726.

Lichtenberg, J. M., A. Buchholz, and P. Schwöbel (2024). Large language models as recommender systems: A study of popularity bias. arXiv preprint arXiv:2406.01285.

Liu, W., B. Liu, J. Qin, X. Zhang, W. Huang, and Y. Wang (2025). Fairness identification of large language models in recommendation. Scientific Reports 15(1), 5516.

Meta (2023). Toward fairness in personalized ads.

Rokon, M. O. F., A. Simion, W. Du, M. Wen, H. Yao, and K.-c. Lee (2026). Enhancement of e-commerce sponsored search relevancy with llm. arXiv preprint arXiv:2607.03886.

Sakib, S. K. and A. B. Das (2024). Challenging fairness: A comprehensive exploration of bias in llm-based recommendations. In 2024 IEEE International Conference on Big Data (BigData), pp. 1585–1592. IEEE.

Zhao, Z., W. Fan, Y. Wu, and Q. Li (2025). Investigating and mitigating stereotype-aware unfairness in llm-based recommendations. arXiv preprint arXiv:2504.04199.