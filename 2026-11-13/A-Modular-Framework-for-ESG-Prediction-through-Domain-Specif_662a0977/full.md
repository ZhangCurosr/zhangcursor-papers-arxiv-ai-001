# A Modular Framework for ESG Prediction through Domain-Specific Language Modeling and Graph Learning

Sofia Baiocco<sup>1</sup>, Gianluca Bonifazi<sup>2,∗</sup>, Christopher Buratti<sup>2</sup>, Michele Marchetti<sup>2</sup>, Camilla Mazzoli<sup>1</sup>, Federica Parlapiano<sup>2</sup>, Giulia Quaglieri<sup>2</sup>, Davide Traini<sup>2,3</sup>, Domenico Ursino<sup>2</sup>, Luca Virgili<sup>2</sup>

<sup>1</sup> DIMA, Polytechnic University of Marche, Italy <sup>2</sup> DII, Polytechnic University of Marche, Italy <sup>3</sup> CHIMOMO, University of Modena and Reggio Emilia, Italy ∗ Corresponding author

s.baiocco@pm.univpm.it; g.bonifazi@univpm.it; c.buratti@pm.univpm.it; michele.marchetti@univpm.it; c.mazzoli@univpm.it; f.parlapiano@pm.univpm.it; g.quaglieri@pm.univpm.it; davide.traini@unimore.it; d.ursino@univpm.it; luca.virgili@univpm.it

## Abstract

Environmental, Social and Governance (ESG) scores are increasingly used to evaluate corporate sustainability, supporting investment decisions, ensuring regulatory compliance and assessing risk. However, current ESG evaluations largely rely on proprietary methodologies that often produce inconsistent ratings. This motivates the development of data-driven prediction frameworks. In this paper, we present ESGPred, a modular framework for predicting ESG scores by combining domainspecific language modeling with graph-based learning over a network that models the relationships among the companies being evaluated. In addition to presenting the rationale behind ESGPred and detailing its ESG score prediction process, we show that ESGPred can efectively predict both overall ESG ratings and their individual Environmental, Social, and Governance components. We evaluated ESGPred using a dataset of 5,266 sustainability reports published by 2,134 publicly listed companies between 2020 and 2023. In this evaluation, we used ESG ratings provided by LSEG Data & Analytics as ground truth. Our experimental results demonstrated that integrating graph-based relational learning improves prediction performance over purely text-based baselines.

Keywords: ESG Prediction, Graph Neural Networks, Language Models, Sustainable Finance, Graph Learning.

## 1 Introduction

In recent years, Environmental, Social and Governance (ESG) criteria have become central to sustainable finance, corporate governance, and institutional investment decisions [43, 44]. ESG scores summarize a company’s sustainability performance beyond traditional financial indicators and influence capital allocation, risk assessment, and stakeholder evaluation [28,30,36]. Meanwhile, regulatory attention has intensified, particularly in the European Union, where the Corporate Sustainability Reporting Directive (CSRD)<sup>1</sup> aims to improve the transparency, comparability, and standardization of corporate sustainability reports. Similar initiatives in other jurisdictions reflect a broader global shift toward more accountable, data-driven ESG reporting practices [15].

Despite this growing institutional and regulatory relevance, ESG assessment remains challenging and only partially standardized [39]. ESG ratings are commonly produced by specialized companies, such as Sustainalytics<sup>2</sup>, Moody’s Vigeo-Eiris<sup>3</sup>, S&P Global<sup>4</sup>, MSCI<sup>5</sup>, and LSEG Data & Analytics<sup>6</sup>. These agencies rely on proprietary methodologies that difer in terms of data sources, rating scales, weighting schemes, and aggregation criteria. Consequently, the same company may receive substantially diferent ESG evaluations depending on the agency [4]. This lack of alignment raises concerns about the comparability, reliability, and transparency of ESG scores, particularly among investors, regulators, and stakeholders who use them for decision-making purposes.

The main reason for this discrepancy is the challenge of translating diverse, mainly qualitative corporate reports into consistent, quantitative indicators. Although sustainability reports contain valuable information about environmental policies, social initiatives, governance structures, risk management practices, and compliance strategies, they are lengthy and written in specialized corporate and financial language. Manual assessment is consequently costly and dificult to reproduce.

Conversely, fully automated approaches can extract meaningful patterns from large volumes of unstructured textual data. Recent advances in Natural Language Processing (NLP) have demonstrated the efectiveness of Transformer-based models in analyzing financial and corporate documents. Fin-BERT [29], a BERT-based model pre-trained on financial corpora, is particularly suitable for financial text-mining tasks because it provides language representations well-suited to them. However, ESG prediction cannot be treated as a purely textual problem. In fact, companies often share sector-specific characteristics, business models, regulatory constraints, organizational structures, markets, products, and sustainability narratives. Ignoring these relationships may limit the predictive models’ ability to capture broader patterns of ESG behavior. Graph-based learning can provide a great support to handle these relational dependencies. In particular, it is possible to define a graph whose nodes represent companies and/or their sustainability reports and whose edges denote the corresponding relationships. We refer to this structure as a “sustainability graph” in the following. Given one or more sustainability graphs, Graph Neural Networks (GNNs) can be employed to detect information and knowledge about the ESG sustainability of the corresponding companies, for instance to determine ESG scores or classify companies into suitable ESG classes. In the ESG domain, this perspective is particularly relevant because relationships between companies can be multiple and heterogeneous. For instance, they can involve the companies’ industrial sector, the semantic content of their sustainability reports,

their business activity, and so forth.

In this paper, we present ESGPred, a modular framework that combines domain-specific language modeling with graph-based learning to predict the ESG class of a company. This class is encoded as a categorical variable corresponding to ESG score quartiles. To perform its task, first ESGPred fine-tunes FinBERT in a supervised setting using ESG-related labels in order to adapt the FinBERT’s semantic representations to the sustainability domain. Since sustainability reports often exceed the token limit of Transformer architectures, documents are segmented into chunks, which are then embedded and aggregated to obtain a representation for each sustainability report. ESGPred uses these embeddings to construct some sustainability graphs. In particular, it constructs one sustainability graph for each kind of relationship of interest; for instance, it can construct a sustainability graph considering the semantic similarity of sustainability reports, another capturing the similarity of company descriptions, a third one taking the company’s industrial sector into account, and so on. Finally, ESGPred leverages the resulting sustainability graphs to train a Graph Neural Network model, which combines the textual information of each report with knowledge derived from sustainability graphs for determining the ESG class of a company.

Unlike approaches based solely on structured financial indicators or document analysis, ESGPred allows for the systematic evaluation of the impact of sustainability reports. This is achieved by using several sustainability graphs, each considering a specific kind of relationship between the involved sustainability reports and/or companies, as well as graph learning models and ESG targets. Experimental results on a large-scale dataset of 5,266 sustainability reports published by 2,134 publicly listed companies from 2020 to 2023 demonstrate the benefits of combining both language modeling and graph learning.

The rest of this paper is organized as follows. Section 2 reviews the related literature on ESG predic tion, NLP-based sustainability assessment, and graph-based methods. Section 3 provides a technical description of ESGPred. Section 4 describes the experiments performed, the obtained results and the hyperparameter tuning. Section 5 discusses the strengths, limitations, and implications of ESGPred. Finally, Section 6 summarizes the main findings and outlines some future research directions.

## 2 Related Work

ESG scores are widely recognized in current literature as a standard metric for corporate sustainability that extends beyond conventional financial indicators [20], [18], [9]. Private rating agencies calculate these scores using proprietary methods that combine human expertise and automated data processing [14]. To mitigate inconsistent and subjective outcomes, research is shifting toward datadriven, automated ESG evaluation frameworks. A comprehensive review underscores this shift from qualitative analyst assessments to advanced, data-driven frameworks [45]. In this context, Artificial Intelligence (AI) in general, and Machine Learning (ML) in particular, are essential for processing diverse datasets and integrating structured financial records and unstructured textual data.

A significant portion of recent ESG literature attempts to predict sustainability scores using structured financial data [41], [37]. These studies exploit the relationship between a firm’s economic indicators and its ESG performance [10], [31]. To capture these complex dynamics, researchers are replacing traditional linear regression with robust nonlinear methodologies [8], [11]. For example, the author of [6] proposes a framework based on advanced gradient boosting algorithms, such as CatBoost, to reveal intricate connections between financial and environmental indicators. As a further advance in this direction, the authors of [22] present an ensemble model combining neural networks with XG-Boost and CatBoost. Their research shows that basic financial metrics can reliably estimate initial ESG scores, which is beneficial for small companies that lack comprehensive sustainability reporting.

Further studies shift the focus to geographical contexts and investigate regional market variations [46], [7], [2]. For instance, the authors of [26] demonstrate that governance and financial data are highly efective ESG predictors for Taiwanese non-financial companies. They emphasize that Asian markets present distinct liquidity risks and a higher density of technology companies than Western economies. Similarly, the authors of [12] design a model customized for the Chinese market. They demonstrate that Machine Learning approaches require adaptation to local regulatory contexts to accurately reflect regional sustainability dynamics.

Despite recent methodological progress, the authors of [34] point out that financial metrics alone cannot fully capture ESG performance. To address this limitation, they propose a stacked ensemble model that integrates structured non-financial variables, such as controversy and geospatial indicators.

However, relying solely on structured data is inherently limiting. A comprehensive understanding of corporate responsibility requires analyzing unstructured data [35], [16], [1]. Consequently, a substantial portion of recent research has shifted toward Natural Language Processing (NLP) [32], [38]. These studies posit that corporate narratives, such as Corporate Social Responsibility (CSR) statements and annual reports, contain crucial information for predicting ESG performance, which traditional structured financial data fail to capture [40], [25].

A significant obstacle to the linguistic analysis of company sustainability reports is that general purpose language models cannot accurately process specialized financial terminology. To address this issue, the authors of [23] propose ESG-KIBERT, a domain-specific variant of BERT that leverages a hard attention mechanism to improve semantic interpretation. Integrating this architecture with FinBERT efectively allows for the evaluation of the thematic content of corporate ESG narratives alongside broader market sentiment, capturing the interplay between oficial reporting and investor perception.

Furthermore, company reports are often structured to minimize negative sentiment, so standard NLP tools have dificulty identifying underlying sustainability risks. To address this issue, the authors of [33] use Transformer-based models trained on oficial company filings. These models relate textual representations to established ESG ratings within a binary classification framework. The results described in [33] suggest that pre-screening ESG-relevant passages allows these models to significantly outperform traditional methods when classifying corporate sustainability performance. The authors of [47] introduce E-BERT, which automates the evaluation pipeline by fine-tuning BERT on a specialized corpus. E-BERT filters out non-essential information in corporate documents, enabling the model to distinguish authentic sustainability initiatives from promotional rhetoric. This specialized focus allows E-BERT to consistently outperform baseline deep learning methods. The ESG2PreEM framework, proposed in [24], evaluates textual data by integrating FinBERT sentiment features with a soft-voting ensemble architecture that includes BERT, ALBERT, and RoBERTa.

Graph-based methods for modeling and exploiting company relationships have gained traction in ESG research. The authors of [5] model company governance as a network connecting companies through shared decision-makers. They then use GraphSAGE [19] and GAT [42] architectures to integrate node-level features and network structure, achieving stronger classification results. Similarly, the authors of [3] predict ESG ratings by constructing dynamic graphs from multiple sources, such as overlapping board members, mutual fund co-ownership, and media co-mentions. This enriches the model with centrality-based features.

Despite these advances, applying Graph Neural Networks (GNNs) to predict ESG scores is still in its early stages. The existing literature largely focuses on explicit, observable inter-company relationships, such as operational or strategic linkages. The ESGPred framework introduced in this paper systematically examines multiple forms of report and/or company relationships and integrates them with text-based analysis of sustainability reports to obtain more complete and refined ESG scores. Furthermore, ESGPred enables the automated assessment of ESG performance based on corporate disclosures and context, thus addressing the consistency issues that could arise from a human-based evaluation performed by agencies, each having a proprietary procedure.

## 3 Proposed Approach

ESGPred uses textual information associated with companies, particularly their sustainability reports and descriptions, to predict corporate sustainability scores and classify companies accordingly. Unlike purely textual approaches, which treat each document and/or company as an independent enti ${ \mathrm { , y , } }$ ESGPred adopts a “relational” perspective. In fact, it models sustainability reports as nodes in a graph, which we call a “sustainability graph”, and the relationships between sustainability reports and/or the companies to which they refer as edges. These edges are constructed based on diferent similarity or membership criteria.

Let $R = \{ r _ { 1 } , r _ { 2 } , . . . , r _ { n } \}$ be the set of sustainability reports into consideration, where each report $r _ { i } \in R$ is associated with a specific company $\zeta _ { i }$ and a given reference year. $r _ { i }$ is also associated with a label $y _ { i }$ , which corresponds to the sustainability metric to be predicted. This metric can be the overall ESG score or one of its components (Environmental, Social or Governance). Thus, ESGPred is not limited to a single predictive target; rather, it can be applied to various ESG assessment dimensions. R is partitioned into a training set $R _ { t r a i n }$ and a test set $R _ { t e s t }$ applying a 80/20 split.

Figure 1 illustrates the ESGPred’s overall workflow, from processing sustainability reports to constructing sustainability graphs and predicting the final ESG scores. As can be seen from this figure, ESGPred’s workflow consists of four steps.

In the first step, ESGPred uses the sustainability reports to perform supervised fine-tuning of FinBERT, a Transformer model specialized for the financial domain [29]. This step aims to adapt FinBERT to the corporate sustainability domain and to allow it to learn semantic representations more closely aligned with ESG metrics. Since sustainability reports are generally long and often exceed the maximum token limit supported by models, the text is segmented into chunks using dynamic or fixed segmentation strategies.

In the second step, ESGPred uses the fine-tuned FinBERT to generate document-level embeddings. Specifically, ESGPred divides each report into chunks and extracts a dense embedding for each chunk. Then, it averages the embeddings associated with the individual chunks to obtain a single embedding of the report.

![](images/c347095af350cf11d16a13804c5dc337aa626843c82d9ade10e20ad2816022d7.jpg)  
Figure 1: Overview of the ESGPred’s workflow

In the third step, ESGPred uses the obtained embeddings as initial features for the nodes of a sustainability graph $G = ( V , E )$ . Each node $v _ { i } \in V$ represents the sustainability report of a company for a given year. The edges of E represent relationships among the sustainability reports and/or the companies to which the sustainability reports refer. As previously pointed out, diferent kinds of relationship can be considered. Each of these kinds can give rise to a diferent sustainability graph.

In the fourth step, G is used as input for a graph-based predictive model. This model combines the representation of each node with the representation of the nodes belonging to its neighborhood in the sustainability graph. Thus, the final prediction depends not only on the content of the report associated with a specific company, but also on the information provided by the neighboring reports and/or companies in the sustainability graphs. The latter are clearly diferent for the diferent kinds of company relationship taken into consideration. Therefore, this procedure ensures that ESG scores derive from both text information stored in the report and company relationships modeled in the sustainability graphs into consideration.

The ESGPred’s key feature is its modularity. Each ESGPred’s component (i.e., the model used to generate embeddings, the document segmentation strategy, the method for constructing sustainability graphs, the graph-based predictive model, and the target label) can be replaced or configured independently.

In the following subsections, we illustrate in more detail the various steps of the ESGPred’s workflow.

## 3.1 Step 1: Supervised Fine-Tuning of FinBERT

The first step of the ESGPred’s workflow involves the supervised fine-tuning of FinBERT [29] to adapt it to the domains of corporate sustainability and ESG assessment. Specifically, this fine-tuning aims to allow FinBERT to learn semantic representations of sustainability reports closely related to ESG classes. It also allows FinBERT to learn language representations already specialized in economic and financial domains. This improves the ESGPred’s ability to interpret terminology typically found in corporate reports and ESG documentation. ESGPred performs the fine-tuning of FinBERT by providing it with the set $R _ { t r a i n }$ of the training sustainability reports into consideration.

Due to their length, sustainability reports often exceed the 512-token maximum supported by the BERT architecture. To overcome this limitation, ESGPred employs a text segmentation strategy that uses overlapping chunks. First, ESGPred tokenizes the text of each report using the WordPiece tokenizer, which is compatible with FinBERT. At the end of this task, the resulting token sequence is divided into chunks with a maximum length of 510 tokens. The special tokens [CLS] and [SEP] are then added to these chunks, producing final chunks of 512 tokens. To maintain semantic continuity between consecutive chunks, ESGPred uses a sliding window with a stride of 256 tokens, which introduces controlled overlap between adjacent chunks.

To address class imbalance, which is a significant issue in the ESG context, ESGPred uses Weighted Focal Loss [27] during the fine-tuning of FinBERT. Focal Loss reduces the contribution of easily classifiable samples, thereby focusing the learning process on more dificult samples. This improves the model’s robustness with respect to minority classes. The Focal Loss used in ESGPred is formally defined as follows:

$$
\mathcal { L } _ { F L } = - \alpha _ { t } ( 1 - p _ { t } ) ^ { \gamma } \log ( p _ { t } )\tag{3.1}
$$

Here, $p _ { t }$ is the predicted probability of the correct class; $\alpha _ { t }$ is the weight associated with the class t, which is inversely proportional to the class’s relative frequency in the training set; $\gamma$ is the focusing parameter that controls the attention given to dificult samples.

ESGPred trains FinBERT in supervised mode using the training set; its final configuration is determined by optimizing the F1-Score on the validation set. To prevent overfitting, ESGPred employs an early stopping strategy based on the trend of the F1-Score.

The fine-tuned FinBERT is able to generate document embeddings semantically specialized for the ESG domain and is used by ESGPred for constructing the embeddings of sustainability reports.

## 3.2 Step 2: Generation of ESG report embeddings

The second step of ESGPred’s workflow involves generating dense vector representations of sustainability reports, which are then used as initial features for sustainability graph nodes. The goal of this step is to obtain document-level embeddings that can semantically represent the content of sustainability reports preserving information even when it is stored in particularly long documents.

As discussed in Section 3.1, sustainability reports often exceed the maximum token limit supported by Transformer architectures. For this reason, each report is divided into chunks compatible with the language model being used (e.g., 512-token chunks compatible with FinBERT). ESGPred supports two segmentation strategies, i.e., dynamic and fixed. In the case of dynamic splitting, the number of chunks increases with the report length. Specifically, the report is divided into chunks, each of which has a length equal to the Transformer’s capacity. This is achieved using a sliding window with controlled overlap between consecutive chunks. This strategy preserves the report’s semantic structure and automatically adjusts segmentation according to the report’s actual length. In the case of fixed splitting, each report is divided into a predefined number of chunks, regardless of its total length. We considered diferent values for the number of chunks to analyze the impact of text granularity on the quality of the generated embeddings.

Formally, given a report $r _ { i } \in R .$ the segmentation process produces a set of chunks:

$$
C _ { i } = \{ c _ { i _ { 1 } } , c _ { i _ { 2 } } , . . . , c _ { i _ { m } } \}\tag{3.2}
$$

ESGPred supports various embedding models, albeit the default model is the fine-tuned FinBERT obtained as described in Section 3.1. Thus, ESGPred’s behavior is independent of the linguistic model adopted for performing chunk embeddings. For each chunk $c _ { i _ { j } }$ , ESGPred obtains a dense embedding $\mathbf { z } _ { i _ { j } } \in \mathbb { R } ^ { d }$ , where d represents the dimensionality of the linguistic model’s latent space. To obtain a single vector representation of the complete report, ESGPred aggregates the embeddings of the individual chunks using a mean pooling operation. Formally, the final embedding associated with the report $r _ { i }$ is obtained as follows:

$$
\mathbf { z } _ { i } = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathbf { z } _ { i _ { j } }\tag{3.3}
$$

This aggregation strategy generates stable, semantically meaningful document-level representations by combining the contributions of a report’s chunks into a single, dense vector. These embeddings are then used as initial features for graph nodes when constructing the sustainability graph (see Section 3.3).

## 3.2.1 Possible embedding models to integrate in ESGPred

As previously pointed out, ESGPred is agnostic with respect to the embedding model used internally. In our experiments, we analyzed two models for generating embeddings of sustainability reports. They are:

• Fine-Tuned FinBERT [29]. This model is fine-tuned for the supervised ESG classification task as described in Section 3.1. It is the default model in ESGPred.

• Fine-Tuned Jina Embeddings (v2) [17]. The Jina model family was designed to generate highdimensional, general-purpose semantic embeddings. Unlike FinBERT, Jina was not specifically trained on the financial domain. Analogously to FinBERT, we fine-tuned Jina for the supervised ESG classification task.

## 3.3 Step 3: Construction of the graph

The third step of the ESGPred’s workflow creates a sustainability graph that models the dependencies among the considered sustainability reports (see Section 3). Given a set R of sustainability reports and a kind of relationship $\varrho$ of interest, the sustainability graph $G _ { \varrho }$ can be modeled as:

$$
G _ { \varrho } = ( V _ { \varrho } , E _ { \varrho } )\tag{3.4}
$$

Each node $v _ { i } \in V _ { \varrho }$ corresponds to a sustainability report $r _ { i } \in R ;$ therefore, the number of nodes in $G _ { \varrho }$ coincides with the number of reports in R. Each node $v _ { i }$ has associated a feature vector $\mathbf { z } _ { i } \in \mathbb { R } ^ { d }$ which is the embedding of $r _ { i }$ obtained in the previous step.

Given a node $v _ { i } \in V _ { \varrho }$ , we indicate by $\mathcal { N } ( v _ { i } )$ the set of neighbors of $v _ { i } ,$ i.e., the set of nodes in $G _ { \varrho }$ directly connected to $v _ { i }$

The most important part of this step is defining the kinds of relationship of interest. ESGPred supports diferent kinds of relationship on the same set of sustainability reports. Each kind of relationship models a specific type of correlation between companies or reports. For example, one kind of relationship considers the semantic similarity of reports’ textual content, another one the similarity of descriptive information associated with companies, and another one structural or categorical relationships between companies (e.g., the fact that they belong to the same industrial sector).

This formulation allows us to model both latent relationships derived from the linguistic content of reports and explicit relationships associated with organizational or structural characteristics of the corresponding companies. Generally, an edge between two nodes indicates that the corresponding nodes share a specific form of proximity, as defined by the kind of relationship which it refers to. Formally, each edge $e _ { i k } \in E _ { \varrho }$ connects two nodes $v _ { i }$ and $v _ { k }$ and can be associated with a weight $w _ { i k } \in \mathbb { R }$ , representing the strength of the relationship between the reports represented by $v _ { i }$ and $v _ { k }$ or the companies to which they refer. This weight can then be used in the prediction of ESG scores.

The ESGPred’s capability of simultaneously handling diferent kinds of relationship enables a systematic analysis of how diferent kinds of relationship afect the quality of learned representations and the final performance of the ESG prediction task.

## 3.3.1 Graph Construction Strategies

As we have seen above, ESGPred is agnostic with respect to the kind of relationship associated with the sustainability graph $G _ { \varrho }$ (see Equation 3.4). In our experiments, we considered latent correlations derived from the reports’ content and explicit correlations associated with the companies’ descriptive or structural characteristics. In the following we present three kinds of relationship. The first is an example of latent correlation; the second and the third are examples of explicit correlations.

## Semantic Report Similarity Graph

The first kind of relationship used for constructing the sustainability graph is the semantic similarity of the text in sustainability reports. In this case, the edges of $G \varrho$ and the corresponding weights are directly derived within the embedding space at the document level. Let $\mathbf { z } _ { i }$ and $\mathbf { z } _ { k }$ be the embeddings

associated with the reports $r _ { i }$ and $r _ { k }$ , respectively. The similarity between $r _ { i }$ and $r _ { k }$ is obtained by calculating the cosine similarity between $\mathbf { z } _ { i }$ and $\mathbf { z } _ { k } .$

$$
\mathbf { s i m } ( r _ { i } , r _ { k } ) = { \frac { \mathbf { z } _ { i } \cdot \mathbf { z } _ { k } } { \| \mathbf { z } _ { i } \| \| \mathbf { z } _ { k } \| } }\tag{3.5}
$$

The existence of an edge between the nodes $v _ { i }$ and $v _ { k }$ in $G _ { \varrho } ,$ corresponding to the reports $r _ { i }$ and $r _ { k }$ , depends on whether the value of sim $( r _ { i } , r _ { k } )$ is greater than or equal to a similarity threshold $\tau _ { c o s } 2$

$$
e _ { i k } \in E _ { \varrho } \iff \mathbf { s } \mathrm { i m } ( r _ { i } , r _ { k } ) \geq \tau _ { c o s }\tag{3.6}
$$

This allows $G _ { \varrho }$ to link reports with semantically related content and enables graph-based prediction models to consider the existence of similarities between linguistic patterns, ESG terminology, or narrative structures. The resulting cosine similarity is used as the weight $w _ { i k }$ associated with $e _ { i k }$

## Company Description Similarity Graph

The second kind of relationship used for constructing the edges of $G _ { \varrho }$ is the similarity of the descriptive information about companies associated with reports. In this case, the edges of $G _ { \varrho }$ model descriptive and organizational-semantic similarities. The nodes of $G _ { \varrho }$ are associated with sustainability reports, while edges are constructed using information extracted automatically from company descriptions via Natural Language Processing (NLP) techniques. Specifically, the SpaCy library is used to perform a text extraction pipeline consisting of two complementary modules devoted to performing morphosyntactic analysis and Named Entity Recognition (NER), respectively.

Morphosyntactic analysis identifies semantically relevant nouns and proper nouns within company descriptions. The extracted terms are then lemmatized and filtered to reduce linguistic variability and standardize semantically equivalent words. NER identifies named entities belonging to specific categories of information, including: (i) organizations and companies (ORG), (ii) geopolitical entities (GPE), (iii) locations (LOC), (iv) people (PERSON), (v) products, brands, and technologies (PRODUCT). Combining these two techniques overcomes the limitations of each one. In fact, morphosyntactic analysis captures informative keywords but can break down complex expressions, while NER preserves complete semantic entities but tends to ignore relevant common terms.

At the end of the process, the extracted keywords and entities related to the company $\zeta _ { i }$ associated with a report $r _ { i }$ are merged into one set:

$$
{ { S } _ { i } } = \left\{ { { s } _ { i _ { 1 } } } , { { s } _ { i _ { 2 } } } , \ldots , { { s } _ { i _ { m } } } \right\}\tag{3.7}
$$

$S _ { i }$ represents a description of $\zeta _ { i }$ . The similarity between the sets $S _ { i }$ and $S _ { k }$ , relative to the companies $\zeta _ { i }$ and $\zeta _ { k }$ associated with the reports $r _ { i }$ and $r _ { k }$ , is calculated using the Jaccard coeficient between $S _ { i }$ and $S _ { k }$ :

$$
J ( S _ { i } , S _ { k } ) = { \frac { | S _ { i } \cap S _ { k } | } { | S _ { i } \cup S _ { k } | } }\tag{3.8}
$$

The presence of an edge $e _ { i k }$ between the two nodes $v _ { i }$ and $v _ { k }$ depends on whether $J ( S _ { i } , S _ { k } )$ is greater than or equal to a threshold $\tau _ { J }$

$$
e _ { i k } \in E _ { \varrho } \iff J ( S _ { i } , S _ { k } ) \geq \tau _ { J }\tag{3.9}
$$

Also in this case, the similarity value is used as the weight $w _ { i k }$ of the edge $e _ { i k }$ . This configuration allows for the creation of a sustainability graph based on descriptive similarities among companies, regardless of the direct similarity of their sustainability reports.

## Sector-Based Graph

The third kind of relationship used for constructing the edges of $G _ { \varrho }$ is based on the industry classification of the companies to which the reports refer. Formally, let $r _ { i }$ and $r _ { k }$ be two reports and let $\zeta _ { i }$ and $\zeta _ { k }$ be the companies to which they refer. Let $v _ { i }$ and $v _ { k }$ be the nodes in $G _ { \varrho }$ associated with $r _ { i }$ and $r _ { k } .$ , respectively. Let sector $( \zeta _ { i } )$ be the function that returns the industrial sector associated with $\zeta _ { i }$ The presence of an edge between $v _ { i }$ and $v _ { k }$ depends on whether the industrial sectors of $\zeta _ { i }$ and $\zeta _ { k }$ are the same:

$$
e _ { i k } \in E _ { \varrho } \iff \widehat { \mathbf { s e c t o r } ( \zeta _ { i } ) } = \mathsf { s e c t o r } ( \zeta _ { k } )\tag{3.10}
$$

In this case, the weight of all the edges of $G _ { \varrho }$ thus obtained is set to 1. This procedure generates a sustainability graph consisting of multiple connected components, each associated with a specific industrial sector. The underlying assumption is that companies within the same industrial sector have similar operational practices, regulatory constraints, ESG strategies, and industry terminology.

This variety of relationship types that can be considered for constructing sustainability graphs allows us to systematically evaluate how diferent forms of correlation between reports and/or companies contribute to ESG predictions. Since the nodes of the sustainability graphs are the same across the diferent kinds of relationship, observed diferences in performance are primarily due to efects of the modeled kinds of relationship.

Figure 2 illustrates an example of the three graph construction strategies described above. In particular, the three subfigures show three sustainability graphs, each consisting of 40 nodes representing 40 sustainability reports relative to companies belonging to the same industrial sector. The first (resp., second, third) subfigure shows the Semantic Report Similarity Graph (resp., Company Description Similarity Graph, Sector-Based Graph) obtained from the same reports. The figure highlights the markedly diferent edge distributions produced by the three graph construction strategies. The Semantic Report Similarity Graph forms relatively compact clusters of companies with similar sustainability reports. Instead, the Company Description Similarity Graph exhibits a more heterogeneous connectivity pattern, reflecting similarities in the organizational characteristics of companies. Finally, the Sector-Based Graph is considerably denser because it connects companies belonging to the same industrial sector.

## 3.4 Step 4: Graph-Based Prediction of ESG Scores

The fourth and final step of the ESGPred’s workflow employs graph-based predictive models to predict the ESG scores associated with sustainability reports and to classify them accordingly. Starting from the graph $G _ { \varrho } = ( V _ { \varrho } , E _ { \varrho } )$ constructed in the previous step, ESGPred learns contextualized representations of nodes by considering the relationships linking the nodes of $G _ { \varrho }$ and the corresponding strengths expressed by edge weights. Therefore, the predicted class of a report $r _ { i }$ corresponding to a node $v _ { i }$ in $G _ { \varrho }$ depends not only on the textual content of $r _ { i } .$ , but also on the information derived from the nodes belonging to the neighborhood $\mathcal { N } ( v _ { i } )$ of $v _ { i }$ in $G _ { \varrho }$ . Clearly, $\mathcal { N } ( v _ { i } )$ depends on the edges of $G _ { \varrho }$ (see Section 3), and therefore on the relationship type ϱ associated with $G _ { \varrho } .$

![](images/796a218abe3833b0543857c2cdd9229d4a0368f2630ad9bc2738db73d4291d03.jpg)

![](images/8431a8c2a26b20bcae6adc1f5b6ec2f535f527bf75122f54aa551393b70a42bd.jpg)

(a) Semantic Report Similarity Graph  
(b) Company Description Similarity Graph  
![](images/f6007a74c2ccd04e262ac1ddb580e413f51789f9e4b8a718421792c10224d9d9.jpg)  
(c) Sector-Based Graph  
Figure 2: Sustainability graphs obtained by applying Semantic Report Similarity Graph, Company Description Similarity Graph and Sector-Based Graph on the same nodes

Formally, let $\mathbf { Z } \in \mathbb { R } ^ { n \times d }$ be the matrix of the embeddings of the reports associated with the nodes of $G _ { \varrho }$ , where the row $\mathbf { z } _ { i }$ corresponds to the embedding of the report $r _ { i }$ . A graph-based predictive model adopted by ESGPred learns a parametric function:

$$
f _ { \theta } : ( G _ { \varrho } , \mathbf { Z } ) \to \hat { \mathbf { Y } }\tag{3.11}
$$

Here, θ is the set of learnable parameters and $\hat { \mathbf Y }$ is the matrix of predictions generated for the nodes of $G _ { \varrho } .$ , and therefore for the reports of R. Here, we assume that the graph-based predictive model consists of $L$ layers. In general, each layer updates the representation of a node $v _ { i }$ by aggregating information from its neighborhood $\mathcal { N } ( v _ { i } )$ . Let $\mathbf { h } _ { i } ^ { 0 } = \mathbf { z } _ { i }$ denote the initial representation of $v _ { i }$ . The representation of $v _ { i }$ updated at the layer $\ell + 1$ can be expressed as follows:

$$
\mathbf { h } _ { i } ^ { \ell + 1 } = \phi ^ { \ell } \left( \mathbf { h } _ { i } ^ { \ell } , \mathsf { A G G } \left( \left\{ \mathbf { h } _ { k } ^ { \ell } , e _ { i k } , w _ { i k } : v _ { k } \in \mathcal { N } ( v _ { i } ) \right\} \right) \right)\tag{3.12}
$$

Here: $( i )$ AGG is an aggregation function depending on the adopted prediction model; $( i i ) ~ \phi ^ { \ell }$ is a learnable transformation that may be combined with nonlinear activation functions, normalization, and regularization.

ESGPred can support various graph-based predictive models, including Graph Convolutional Networks, GraphSAGE, Graph Attention Networks, and Graph Transformers. These models difer primarily in how they aggregate and weight information from adjacent nodes. Some approaches use normalized aggregations based on the graph structure. Others learn neighborhood sampling and aggregation functions. Still others use attention mechanisms to assign varying importance to the nodes of the neighborhood or the edges connecting them.

Let $\mathbf { h } _ { i } ^ { L }$ be the representation of the node $v _ { i }$ in the final layer of the graph-based predictive model. ESGPred inputs $\mathbf { h } _ { i } ^ { L }$ into a final classifier Γ, which returns the probability distribution associated with all the possible ESG classes. The class with the highest probability is returned as the class of $r _ { i } { : }$

$$
\hat { y } _ { i } = \mathbf { a r g m a x } ( \Gamma ( \mathbf { h } _ { i } ^ { L } ) )\tag{3.13}
$$

The graph-based predictive model is trained in a supervised manner using the node labels available in the training set. The loss is computed by comparing the predicted class $\hat { y } ^ { i }$ with the corresponding ground-truth label $y _ { i }$ . To mitigate the efects of class imbalance, ESGPred employs weighted loss functions, such as Weighted Cross-Entropy or Focal Loss, thereby reducing the bias toward majority classes and improving predictive performance on underrepresented classes.

Due to its modular design, ESGPred can be adapted to predict diferent types of sustainabilityrelated information while preserving its overall architecture and workflow. Specifically, the predictive graph-based model underlying ESGPred can be trained to predict various sustainability metrics. The first metric is the overall ESG score, which is an aggregate measure of corporate performance across the Environmental, Social, and Governance dimensions. In this case, the label $y _ { i } ^ { E S G }$ associated with the report $r _ { i }$ indicates the company’s overall ESG class. In addition to the overall ESG score, ES-GPred can predict the individual ESG components. For the Environmental component, the label $y _ { i } ^ { E }$ represents the company’s assessment with respect to issues such as environmental impact, emis sions, resource use, energy management, and climate policies. For the Social component, the label $y _ { i } ^ { S }$ reflects aspects related to human resources, working conditions, community relationships, human rights, safety, and social responsibility. Finally, for the Governance component, the label $y _ { i } ^ { G }$ refers to elements such as board structure, oversight practices, transparency, corporate ethics, compensation, and risk management.

Formally, ESGPred allows a set of possible prediction targets:

$$
\mathcal { T } = \{ E S G , E , S , G \}\tag{3.14}
$$

For each target $t \in \tau$ , ESGPred can associate a set of labels $\mathbf { Y } ^ { t } = \{ y _ { 1 } ^ { t } , y _ { 2 } ^ { t } , \ldots , y _ { n } ^ { t } \}$ such that $y _ { i } ^ { t }$ $1 \leq t \leq n$ , is the score of the report $r _ { i } \in R$ . The graph-based predictive model underlying ESGPred is trained to learn a function specific to each target:

$$
f _ { \theta } ^ { t } : ( G , \mathbf { Z } ) \to { \hat { \mathbf { Y } } } ^ { t }\tag{3.15}
$$

## 3.4.1 Graph-Based Predictive Models

ESGPred is agnostic with respect to the graph-based predictive model employed. In our experiments, we considered various Graph Neural Networks (GNNs) as predictive models to analyze the impact of diferent aggregation, propagation, and attention mechanisms on the quality of the provided predictions. In all cases, each node in the sustainability graph is associated with an initial feature vector:

$$
\mathbf { h } _ { i } ^ { 0 } = \mathbf { z } _ { i } ,\tag{3.16}
$$

Here, $\mathbf { z } _ { i } \in \mathbb { R } ^ { d }$ is the embedding of the report $r _ { i }$ . The prediction activity consists of iteratively updating these representations through message-passing operations between connected nodes of the sustainability graph.

The predictive models under consideration difer primarily in the following aspects: (i) the strategy for aggregating information from the neighborhood; (ii) the weights of the contributions from neigh borhing nodes of the sustainability graph; (iii) the explicit use of edge weights; (iv) the mechanism for propagating information throughout the sustainability graph. They are:

• Graph Convolutional Network (GCN) [21]. It uses approximate spectral convolution to propagate information between adjacent nodes. Instead of averaging the values of neighboring nodes, GCN applies a symmetric normalization based on node degrees, allowing the contribution of connected nodes to be adjusted according to the local graph structure. In ESGPred, the GCN model explicitly uses the weights associated with the sustainability graph’s edges.

• GraphSAGE [19]. Unlike traditional GCNs, GraphSAGE employs parameterized neighborhood aggregation schemes, enabling a more flexible and expressive integration of local neighborhood information.

• Graph Attention Network (GAT) [42]. It is based on attention mechanisms applied to node neighborhoods. In this model, the contribution of adjacent nodes is not solely determined by the sustainability graph structure; rather, it is learned dynamically through attention coeficients.

• Graph Transformer [13]. It combines Transformer principles with the message passing characteristic of Graph Neural Networks. Unlike previous architectures, the Graph Transformer uses multi-head attention mechanisms to simultaneously model diferent kinds of relationship between the sustainability graph nodes. Specifically, each attention head learns a diferent neighborhood weighting function, allowing the predictive model to capture multiple, potentially complementary relationships.

## 4 Experiments

In this section, we describe our experiments conducted to evaluate ESGPred. In particular, Subsection 4.1 presents the experimental setup. Subsection 4.2 reports our experimental results and compares ESGPred with several related approaches. Finally, Subsection 4.3 presents a study of ESGPred’s hyperparameters.

## 4.1 Experimental Setup

In this section, we illustrate the datasets, baselines and metrics adopted in our experiments.

## 4.1.1 Dataset

For our experiments, we downloaded sustainability reports in PDF format from ResponsibilityReports<sup>7</sup>, a platform that aggregates oficial company documentation. In addition to the reports, the platform provides complementary metadata for each company, including sector classification, country of origin, and a descriptive profile. These descriptions range from brief summaries spanning a few lines to more detailed accounts spanning several paragraphs. They typically address the company’s industry afiliation, founding history, geographical roots, core business activities, and principal products or services it provides. They may also convey the company’s mission or values. Our collection covers the period from 2020 to 2023 and is restricted to public companies. This selection is based on two practical considerations: (i) documents from public companies adhere to more standardized disclosure norms and are consistently accessible, and (ii) ground-truth ESG scores from our reference provider are available for this category of companies. While the raw collection contains 10,014 reports, our final experimental dataset focuses on a subset of them. The sectors most represented in the final sample are: finance (519 reports), technology (486), industrial goods (461), basic materials (428), and consumer goods (419). Conversely, the countries with the most reports are the United States (1,201), England (234), and Australia (212). The overall geographic distribution of companies to which sustainability reports refer is shown in Table 1.

All reports underwent a structured preprocessing pipeline before use. This pipeline was designed to convert PDF content into plain text. Non-textual elements, including figures, charts, and graphical layouts, were systematically discarded. The pipeline proceeded through three stages: (i) raw text extraction from the source PDF files, (ii) normalization of character encoding and removal of formatting residues, and (iii) filtering of non-informative recurring elements, such as page numbers and repeated section headers.

<table><tr><td>Country</td><td>Count</td><td>Country</td><td>Count</td></tr><tr><td>USA</td><td>1,201</td><td>Luxembourg</td><td>3</td></tr><tr><td>England</td><td>234</td><td>Spain</td><td>2</td></tr><tr><td>Australia</td><td>212</td><td>Belgium</td><td>2</td></tr><tr><td>Canada</td><td>205</td><td>Chile</td><td>2</td></tr><tr><td>Sweden</td><td>52</td><td>Hong Kong</td><td>2</td></tr><tr><td>Finland</td><td>23</td><td>Cayman Islands</td><td>2</td></tr><tr><td>Denmark</td><td>20</td><td>Italy</td><td>2</td></tr><tr><td>Ireland</td><td>18</td><td>Wales</td><td>2</td></tr><tr><td>China</td><td>15</td><td>Argentina</td><td>2</td></tr><tr><td>Bermuda</td><td>15</td><td>Cyprus</td><td>1</td></tr><tr><td>Greece</td><td>14</td><td>Austria</td><td>1</td></tr><tr><td>Switzerland</td><td>12</td><td>Isle of Man</td><td>1</td></tr><tr><td>Singapore</td><td>10</td><td>Peru</td><td>1</td></tr><tr><td>Scotland</td><td>9</td><td>Panama</td><td>1</td></tr><tr><td>Israel</td><td>9</td><td>Kazakhstan</td><td>1</td></tr><tr><td>Germany</td><td>8</td><td>Mauritius</td><td>1</td></tr><tr><td>Guernsey</td><td>7</td><td>Ghana</td><td>1</td></tr><tr><td>Jersey</td><td>6</td><td>Mexico</td><td>1</td></tr><tr><td>France</td><td>4</td><td>Lithuania</td><td>1</td></tr><tr><td>Brazil</td><td>4</td><td>Papua New Guinea</td><td>1</td></tr><tr><td>Netherlands</td><td>4</td><td>Slovenia</td><td>1</td></tr><tr><td>New Zealand</td><td>4</td><td>Taiwan</td><td>1</td></tr><tr><td>India</td><td>3</td><td>Turkey</td><td>1</td></tr><tr><td>South Africa</td><td>3</td><td>Thailand</td><td>1</td></tr><tr><td>Monaco</td><td>3</td><td>Ukraine</td><td>1</td></tr><tr><td>Nigeria</td><td>3</td><td>Uruguay</td><td>1</td></tr><tr><td>Total</td><td></td><td>2,134</td><td></td></tr></table>

Table 1: Overall geographical distribution of the companies to which sustainability reports refer

The preprocessed corpus, organized as one text document per company per fiscal year, served two purposes in ESGPred. First, it provided the training set for fine-tuning FinBERT. Second, it supplied the embeddings used to construct the sustainability graph, which is the input for graphbased predictive models.

To validate ESGPred, we gathered ESG scores for public companies. Specifically, we used the scores provided by “LSEG Data & Analytics” (formerly Refinitiv) via the “LSEG Workspace” platform<sup>8</sup>. These scores summarize a company’s overall ESG standing by combining diferent indicators through a proprietary aggregation methodology.

We compared the ESG scores generated by ESGPred with the corresponding scores provided by LSEG. To ensure a valid comparison, we restricted the dataset to companies for which both the sustainability report and the LSEG ESG score were available for the same reference year.

As LSEG ESG data is subject to a commercial license that prohibits public redistribution, we cannot disclose individual scores in this paper. Instead, we report experimental results in terms of aggregate statistics and score quartiles encoded as letter grades ranging from A to D, where A indicates the highest-performing quartile and D denotes the lowest one.

Table 2 reports some statistics related to our dataset. This dataset corresponds to the set R of sustainability reports mentioned in Section 3. As pointed out in that section, we divided R into a training set $R _ { t r a i n }$ and a test set $R _ { t e s t } ,$ applying a 80/20 split. We assigned all reports associated with the same company exclusively to either the training set or the test set to prevent data leakage.

<table><tr><td>Name</td><td>Value</td></tr><tr><td>Total number of sustainability reports</td><td>5,266</td></tr><tr><td>Total number of reports relating to 2020</td><td>1,133</td></tr><tr><td>Total number of reports relating to 2021</td><td>1,459</td></tr><tr><td>Total number of reports relating to 2022</td><td>1,665</td></tr><tr><td>Total number of reports relating to 2023</td><td>1,009</td></tr><tr><td>Total number of reports with score A</td><td>785</td></tr><tr><td>Total number of reports with score B</td><td>2,652</td></tr><tr><td>Total number of reports with score C</td><td>1,611</td></tr><tr><td>Total number of reports with score D</td><td>218</td></tr><tr><td>Total number of companies</td><td>2,134</td></tr><tr><td>Total number of countries of origin of companies</td><td>52</td></tr></table>

Table 2: Some statistics of the dataset R of reports used in our experiments

## 4.1.2 Baselines

To evaluate the efectiveness of ESGPred, we compared the results returned by it to three baselines representing the current state-of-the-art in text-based ESG classification. These baselines allow us to analyze the contributions of the various ESGPred’s components and distinguish the efects of the language model’s domain-specific specialization from the efects of the sustainability graph and graphbased predictive models. They are:

• Vanilla FinBERT [29] in its original version. It was pre-trained on financial corpora, but not adapted for the ESG domain. In this configuration, the model is used directly as a supervised classifier for sustainability reports. This baseline evaluates the predictive power of generic financial language representations without adaptation to a specific task.

• Fine-Tuned FinBERT. It was fine-tuned using the ESG dataset employed in this paper. In this case, the model was trained following the procedure described in Section 3.1 with ESG scores as the supervised targets. This baseline is particularly relevant because it uses the same textual information and language adaptation tasks as ESGPred, but does not introduce any sustainability graph or graph-based predictive model. Therefore, comparing ESGPred to this baseline allows us to directly quantify the contributions of our graph modeling present in ESGPred compared to an approach based solely on the textual content of sustainability reports.

• Fine-Tuned Transformer [33]. It was originally employed as a part of an ESG classification system that used fine-tuned Transformer models, SEC Form 10-K regulatory documents, and ESG ratings provided by S&P Global. In the original paper, the problem is formulated as a binary classification task, which is achieved through a median split of ESG ratings. To ensure a fair comparison with ESGPred, we reimplemented the approach described in [33] to make it applicable to the dataset used in this paper. Specifically, we adapted it to a four-class classification problem (with the classes being A, B, C, and D), which is consistent with the categorization of the ESG scores provided by LSEG. Thus, the results obtained from this baseline allow us to compare ESGPred with a Transformer-based approach specifically designed for ESG assessment and widely recognized in recent literature.

## 4.1.3 Metrics

To evaluate and compare ESGPred with the selected baselines, we used four classification metrics widely adopted in the literature: Accuracy, Precision, Recall, and F1-Score.

Let $C l = \{ A , B , C , D \}$ denote the set of ESG classes and let $y _ { i } \in C l$ denote the actual class associated with the report $r _ { i } \in R .$ Let $\hat { y } _ { i } \in C l$ denote the class predicted by the model.

Accuracy is defined as the proportion of sustainability reports for which the predicted and actual ESG classes coincide. It is defined as follows:

$$
A c c u r a c y = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } ( y _ { i } = \hat { y } _ { i } ) ,\tag{4.1}
$$

where N is the total number of reports in the test set and I(·) is the indicator function, which returns 1 when the actual and predicted classes coincide and 0 otherwise.

For each ESG class cl $\in C l$ , Precision is defined as the proportion of reports predicted to belong to cl that actually belong to cl. It is defined as follows:

$$
P _ { c l } = \frac { T P _ { c l } } { T P _ { c l } + F P _ { c l } } ,\tag{4.2}
$$

where $T P _ { c l }$ is the number of reports whose actual and predicted class is cl and $F P _ { c l }$ is the number of reports predicted as class cl despite belonging to a diferent actual class.

For each ESG class cl $\in C l$ , Recall measures the proportion of reports that actually belong to cl and have been correctly assigned to cl by the model. It is defined as follows:

$$
R _ { c l } = \frac { T P _ { c l } } { T P _ { c l } + F N _ { c l } } ,\tag{4.3}
$$

where $F N _ { c l }$ is the number of reports whose actual class is cl but whose predicted class is diferent. For each ESG class cl $\in C l .$ , F1-Score is defined as the harmonic mean of Precision and Recall. It is defined as follows:

$$
F _ { c l } = 2 \cdot \frac { P _ { c l } \cdot R _ { c l } } { P _ { c l } + R _ { c l } } .\tag{4.4}
$$

Since there are four classes in ESG, the final Precision, Recall and F1-Score values considered in our experiments were obtained by averaging the corresponding parameter values for each class:

$$
P r e c i s i o n = \frac { 1 } { 4 } \sum _ { c l \in C l } P _ { c l } , ~ R e c a l l = \frac { 1 } { 4 } \sum _ { c l \in C l } R _ { c l } , ~ F 1 . S c o r e = \frac { 1 } { 4 } \sum _ { c l \in C l } F _ { c l }\tag{4.5}
$$

## 4.2 Results

We conducted all experiments on a server equipped with two NVIDIA Tesla T4 GPUs, each with 16GB of GPU memory, and an Intel Xeon CPU (Skylake-SP architecture) with 2 cores and 4 hardware threads.

Table 3 shows the results of comparing ESGPred with the baselines described in Section 4.1.2. In this table and in the next ones, the highest value of the performance metric associated with each column is shown in bold.

<table><tr><td>Category</td><td>Model / Architecture</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1-Score</td></tr><tr><td colspan="6">Baselines</td></tr><tr><td></td><td>Vanilla FinBERT</td><td>0.72</td><td>0.69</td><td>0.60</td><td>0.63</td></tr><tr><td></td><td>Fine-Tuned FinBERT</td><td>0.77</td><td>0.72</td><td>0.74</td><td>0.73</td></tr><tr><td></td><td>Fine-Tuned Transformer</td><td>0.76</td><td>0.74</td><td>0.65</td><td>0.68</td></tr><tr><td colspan="6">ESGPred</td></tr><tr><td>Description-Based Graph</td><td>Graph Transformer</td><td>0.85</td><td>0.83</td><td>0.85</td><td>0.84</td></tr><tr><td></td><td>GraphSAGE</td><td>0.84</td><td>0.80</td><td>0.84</td><td>0.84</td></tr><tr><td></td><td>GAT</td><td>0.81</td><td>0.76</td><td>0.82</td><td>0.78</td></tr><tr><td></td><td>GCN</td><td>0.77</td><td>0.71</td><td>0.74</td><td>0.73</td></tr><tr><td>Semantic-Based Graph</td><td>Graph Transformer</td><td>0.84</td><td>0.80</td><td>0.85</td><td>0.82</td></tr><tr><td></td><td>GraphSAGE</td><td>0.82</td><td>0.80</td><td>0.82</td><td>0.81</td></tr><tr><td></td><td>GAT</td><td>0.82</td><td>0.78</td><td>0.85</td><td>0.81</td></tr><tr><td></td><td>GCN</td><td>0.81</td><td>0.78</td><td>0.85</td><td>0.81</td></tr><tr><td>Sector-Based Graph</td><td>Graph Transformer</td><td>0.81</td><td>0.76</td><td>0.79</td><td>0.77</td></tr><tr><td></td><td>GraphSAGE</td><td>0.78</td><td>0.73</td><td>0.75</td><td>0.74</td></tr><tr><td></td><td>GAT</td><td>0.79</td><td>0.73</td><td>0.77</td><td>0.75</td></tr><tr><td></td><td>GCN</td><td>0.76</td><td>0.70</td><td>0.74</td><td>0.72</td></tr></table>

Table 3: Comparison between baselines and ESGPred with diferent graph-based configurations

As expected, the Fine-Tuned FinBERT significantly outperforms the Vanilla FinBERT. In fact, Accuracy increases by 6.94%, while F1-Score rises by 15.87%. These results confirm the importance of adapting the language model used to obtain report embeddings to the ESG domain and demonstrate that representations learned during fine-tuning capture information relevant to classifying sustainabil ity reports.

The Fine-Tuned Transformer yields results similar to those of the Fine-Tuned FinBERT, achieving an Accuracy of 0.76. Although this method was designed for binary classification, its adaptation to a multi-class context returns satisfying results. This confirms the efectiveness of Transformer-based strategies for analyzing ESG documents. Nevertheless, its performance remains inferior to the best configurations of ESGPred.

Analyzing the various sustainability graph construction strategies in ESGPred clearly shows that the quality of the sustainability graph significantly afects the performance of ESGPred. The config uration that yields the best results is the Description-Based Graph combined with the Graph Transformer, which achieves an Accuracy of 0.85 and an F1-Score of 0.84. This suggests that data extracted from company descriptions is an efective source of useful information for ESG prediction. In fact, descriptions tend to capture the structural characteristics of a company, such as its industrial sector, products, technologies, target markets, and business type. This generates connections between companies that share similar economic and organizational features.

The Semantic-Based Graph produces slightly inferior but competitive results. The optimal configuration achieves an Accuracy of 0.84 and an F1-Score of 0.82. These results highlight the importance of semantic similarity between reports as a source of information, though this information may be slightly less useful than relationships constructed from corporate descriptions. Indeed, reports from diferent sectors may use similar ESG terminology to describe vastly diferent corporate contexts.

The Sector-Based Graph shows the lowest performance. While being in the same sector represents an intuitive relationship between companies, this information is less discriminating than the one considered by the other strategies. Constructing the sustainability graph based solely on the industrial sector of companies generates relatively coarse relationships that fail to capture ESG diferences between companies in the same industrial sector.

Looking at the graph architecture performances, these results reveal a clear trend. In nearly all configurations, the Graph Transformer is the most efective predictive model. This is due to its ability to use multi-head attention mechanisms and explicitly incorporate edge-associated information during the prediction task. Thus, it can diferentiate the importance of connections and leverage company relationships more efectively. GraphSAGE generally ranks as the second-best model. It performs particularly well with the Description-Based Graph, achieving an F1-Score of 0.84, which is essentially equivalent to the Graph Transformer’s score. In contrast, the GAT and GCN models perform worse. Although they benefit from the sustainability graph, they appear to have a more limited ability than the Graph Transformer to model complex relationships between companies. GCN consistently performs the worst in most experiments.

Table 4 shows the results of replacing the embeddings generated by the fine-tuned FinBERT with those produced by the fine-tuned Jina. In all configurations, when using the Jina model, performance decreases compared to FinBERT-generated embeddings. The best Jina-based configuration achieves an Accuracy of 0.81 and an F1-Score of 0.77; these values are lower than those obtained by the best FinBERT-based configuration.

<table><tr><td>Category</td><td>Model / Architecture</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1-Score</td></tr><tr><td colspan="6">Jina fine-tuned embeddings</td></tr><tr><td>Description-Based Graph</td><td>Graph Transformer</td><td>0.79</td><td>0.76</td><td>0.75</td><td>0.76</td></tr><tr><td></td><td>GraphSAGE</td><td>0.79</td><td>0.73</td><td>0.75</td><td>0.74</td></tr><tr><td></td><td>GAT</td><td>0.70</td><td>0.62</td><td>0.69</td><td>0.64</td></tr><tr><td></td><td>GCN</td><td>0.69</td><td>0.61</td><td>0.68</td><td>0.61</td></tr><tr><td>Semantic-Based Graph</td><td>Graph Transformer</td><td>0.81</td><td>0.76</td><td>0.79</td><td>0.77</td></tr><tr><td></td><td>GraphSAGE</td><td>0.78</td><td>0.73</td><td>0.75</td><td>0.74</td></tr><tr><td></td><td>GAT</td><td>0.79</td><td>0.73</td><td>0.77</td><td>0.75</td></tr><tr><td></td><td>GCN</td><td>0.76</td><td>0.70</td><td>0.74</td><td>0.72</td></tr><tr><td>Sector-Based Graph</td><td>Graph Transformer</td><td>0.80</td><td>0.76</td><td>0.75</td><td>0.75</td></tr><tr><td></td><td>GraphSAGE</td><td>0.78</td><td>0.72</td><td>0.75</td><td>0.74</td></tr><tr><td></td><td>GAT</td><td>0.79</td><td>0.77</td><td>0.73</td><td>0.74</td></tr><tr><td></td><td>GCN</td><td>0.75</td><td>0.70</td><td>0.71</td><td>0.70</td></tr></table>

Table 4: Results obtained using Jina-based embeddings instead of FinBERT-based embeddings

These results underscore the importance of domain-specific specialization in the context of ESG score evaluation. Although fine-tuned Jina produces high-quality embeddings for general tasks, finetuned FinBERT’s embeddings incorporate domain-specific knowledge of finance and sustainability, which makes them better suited for ESG classification.

An interesting result to highlight regards the relative performance stability across diferent configurations. Even when using Jina-based embeddings, the Semantic-Based Graph and the Description-Based Graph continue to outperform the Sector-Based Graph. Meanwhile, the Graph Transformer remains the best-performing predictive model. These results suggest that the superiority of prediction graphs and graph-based predictive models observed in the previous experiments does not depend exclusively on the quality of embeddings but rather represents a more general property of ESGPred.

## 4.2.1 Single-Dimension Prediction (Environmental, Social, and Governance Scores)

In addition to evaluating the ESGPred’s capability of predicting the overall ESG score, we evaluated the ESGPred’s ability to independently estimate the three fundamental dimensions of corporate sustainability, i.e., Environmental, Social, and Governance. To perform this evaluation, we carried out another experiment with two objectives: (i) verifying the generalizability of ESGPred across diferent sustainability targets, and (ii) analyzing whether the sustainability graph supporting the graph-based predictive model of ESGPred can capture specific information associated with individual ESG components.

To this end, we selected the ESGPred configuration that performed best in predicting the overall ESG score. This configuration consisted of a sustainability graph based on company descriptions and the Graph Transformer as predictive model. We used this configuration in the next experiment.

We conducted an independent training process for each ESG dimension. Specifically, we replaced the overall ESG score with the Environmental, Social, and Governance scores provided by the LSEG platform, respectively. We applied the whole methodological pipeline described in Section 3 for each dimension separately. In particular, we first performed the supervised fine-tuning of FinBERT. Then, we generated the embeddings of sustainability reports and constructed the sustainability graph. Finally, we trained the Graph Transformer.

To assess the impact of sustainability graph, we compared the outcomes of ESGPred with those of a fine-tuned FinBERT model that was trained using the same dataset. The latter served as a purely textual baseline. This allowed us to measure the benefit of adding graph learning to purely linguistic representation for each sustainability dimension.

Table 5 shows the obtained results. As can be seen, ESGPred consistently outperforms the Fine-Tuned FinBERT across all metrics and ESG dimensions. Very good results are obtained for the Environmental component, where Accuracy, Precision, Recall and F1-Score are always greater than 0.85. Furthermore, ESGPred improves the Accuracy (resp., Precision, Recall, and F1-Score) of the Fine-Tuned FinBERT by 24.29% (resp., 17.57%, 27.54%, and 23.94%). These results suggest that information from both sustainability reports and graph-modeled company relationships efectively captures aspects concerning environmental policies, emissions, and resource management.

ESGPred also shows significant performance improvement over the Fine-Tuned FinBERT in the Social component. In fact, it improves the baseline of 25.00% in Accuracy, 22.39% in Precision, 37.70% in Recall, and 29.69% in F1-Score. These results highlight the value of modeling company relationships in identifying common practices and strategies related to human capital management, corporate social responsibility, and stakeholder relationships.

<table><tr><td>Dimension</td><td>Model / Architecture</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1-Score</td></tr><tr><td>Environmental</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>ESGPred (Desc.-Based Graph+Graph Transf.)</td><td>0.87</td><td>0.87</td><td>0.88</td><td>0.88</td></tr><tr><td></td><td>Fine-Tuned FinBERT (baseline)</td><td>0.70</td><td>0.74</td><td>0.69</td><td>0.71</td></tr><tr><td>Social</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>ESGPred (Desc.-Based Graph+Graph Transf.)</td><td>0.85</td><td>0.82</td><td>0.84</td><td>0.83</td></tr><tr><td></td><td>Fine-Tuned FinBERT (baseline)</td><td>0.68</td><td>0.67</td><td>0.61</td><td>0.64</td></tr><tr><td>Governance</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>ESGPred (Desc.-Based Graph+Graph Transf.)</td><td>0.79</td><td>0.80</td><td>0.81</td><td>0.80</td></tr><tr><td></td><td>Fine-Tuned FinBERT (baseline)</td><td>0.57</td><td>0.59</td><td>0.53</td><td>0.55</td></tr></table>

Table 5: Comparative results of ESGPred and the Fine-Tuned FinBERT for individual Environmental, Social, and Governance dimensions

The Governance component is the most dificult to predict. Although ESGPred achieves high performance (with an Accuracy of 0.79, a Precision of 0.80, a Recall of 0.81, and an F1-Score of 0.80), the results are lower than those obtained for the Environmental and Social components. One possible explanation is that Governance information tends to be less descriptive and more closely tied to specific corporate, organizational, and decision-making structures. This makes it more dificult to infer all necessary information from textual content and semantic company relationships alone. In fact, this trend can be also observed in the Fine-Tuned FinBERT. Interestingly, the improvements over the baseline are higher for this component than for the other two; in fact, Accuracy increases by 38.60%, Precision by 35.59%, Recall by 52.83% and F1-Score by 45.45%.

These results demonstrate that ESGPred can not only predict the aggregate ESG score, but also efectively generalize to individual sustainability dimensions. Furthermore, the systematic improvement over purely text-based models confirms the importance of the information about report and company relationships modeled through the sustainability graph. This highlights that information from related companies is a significant factor in assessing the ESG performance of a company.

## 4.3 Hyperparameter tuning

After comparing the various ESGPred configurations, we conducted some hyperparameter tuning experiments to analyze the impact of certain design choices on the ESGPred’s final performance. Specifically, we examined two key aspects of the pipeline: the strategy for generating document-level embeddings and the criteria for constructing the sustainability graph.

The goal of this analysis was to evaluate the ESGPred’s robustness across diferent configurations and to identify settings that balance visualization quality, sustainability graph structure, and ESG prediction accuracy.

The first hyperparameter tuning test examined the efect of the strategy used to generate embeddings for sustainability reports. As described in Section 3.2, ESGPred supports two types of segmentation, namely: (i) dynamic segmentation, which is based on the actual length of the sustainability report, and (ii) fixed segmentation, which uses a predetermined number of chunks. The default segmentation in ESGPred is the dynamic one because we hypothesized that it better preserves the semantic structure of reports by avoiding an excessive fragmentation in short reports and a loss of information in long ones.

To test this hypothesis, we compared dynamic segmentation with three fixed-segmentation configurations (allowing 5, 10, and 15 chunks), while keeping all the other components of ESGPred unchanged. Table 6 shows the results obtained for the three types of graph construction strategies allowed in ESGPred (see Section 3.3).

<table><tr><td>Graph Type</td><td>Split Method</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1-Score</td></tr><tr><td colspan="6">Description-Based Graph</td></tr><tr><td></td><td>Fixed (5 splits)</td><td>0.82</td><td>0.82</td><td>0.83</td><td>0.82</td></tr><tr><td></td><td>Fixed (10 splits)</td><td>0.83</td><td>0.81</td><td>0.84</td><td>0.82</td></tr><tr><td></td><td>Fixed (15 splits)</td><td>0.84</td><td>0.83</td><td>0.83</td><td>0.83</td></tr><tr><td></td><td>Dynamic (Used by ESGPred)</td><td>0.85</td><td>0.83</td><td>0.85</td><td>0.84</td></tr><tr><td colspan="6">Semantic-Based Graph</td></tr><tr><td></td><td>Fixed (5 splits)</td><td>0.81</td><td>0.79</td><td>0.83</td><td>0.81</td></tr><tr><td></td><td>Fixed (10 splits)</td><td>0.82</td><td>0.80</td><td>0.83</td><td>0.81</td></tr><tr><td></td><td>Fixed (15 splits)</td><td>0.82</td><td>0.82</td><td>0.84</td><td>0.82</td></tr><tr><td></td><td>Dynamic (Used by ESGPred)</td><td>0.84</td><td>0.80</td><td>0.85</td><td>0.82</td></tr><tr><td colspan="6">Sector-Based Graph</td></tr><tr><td></td><td>Fixed (5 splits)</td><td>0.80</td><td>0.78</td><td>0.84</td><td>0.80</td></tr><tr><td></td><td>Fixed (10 splits)</td><td>0.82</td><td>0.79</td><td>0.85</td><td>0.82</td></tr><tr><td></td><td>Fixed (15 splits)</td><td>0.83</td><td>0.81</td><td>0.84</td><td>0.82</td></tr><tr><td></td><td>Dynamic (Used by ESGPred)</td><td>0.83</td><td>0.83</td><td>0.84</td><td>0.83</td></tr></table>

Table 6: Impact of report segmentation strategies on the performance of ESGPred

This table shows that dynamic segmentation yields the best performance across all configurations. This improvement is particularly evident in the graphs based on company descriptions and semantic similarity of reports. This behavior suggests that adaptive report segmentation generates more informative embeddings that align better with the semantic structure of sustainability reports. Additionally, the observed increase in performance as the number of fixed chunks increases indicates that a more granular document representation is generally beneficial. However, none of the fixedsegmentation configurations matches the results of the dynamic strategy.

The second hyperparameter tuning study examined the sensitivity of ESGPred to the thresholds used to construct sustainability graphs. To carry out this task: (i) we varied the Jaccard similarity threshold for the graphs constructed from company descriptions (see Section 3.3); (ii) we modified the cosine similarity threshold for the graphs constructed from the semantic similarity of reports.

Table 7 reports the results obtained for several values of the two thresholds. The analysis of this table shows that both configurations have an optimal threshold value, beyond which performance begins to degrade. For sustainability graphs based on company descriptions, the best result is obtained with a Jaccard threshold of 0.30. Lower values introduce less informative connections, while higher values excessively reduce the graph’s connectivity.

A similar behavior is observed in the graph based on the semantic similarity of reports. The optimal threshold for this graph corresponds to a cosine similarity of 0.80. As in the previous case, thresholds that are too permissive increase structural noise, while thresholds that are too restrictive limit the ESGPred’s ability to exploit semantic relationships between reports.

<table><tr><td>Graph Type</td><td>Threshold</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1-Score</td></tr><tr><td>Description-Based Graph</td><td>TJ</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>0.29 (Jaccard)</td><td>0.83</td><td>0.82</td><td>0.83</td><td>0.82</td></tr><tr><td></td><td>0.30 (Jaccard)</td><td>0.85</td><td>0.83</td><td>0.85</td><td>0.84</td></tr><tr><td></td><td>0.31 (Jaccard)</td><td>0.84</td><td>0.80</td><td>0.85</td><td>0.82</td></tr><tr><td></td><td>0.33 (Jaccard)</td><td>0.83</td><td>0.79</td><td>0.85</td><td>0.82</td></tr><tr><td>Semantic-Based Graph</td><td>Tcos</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>0.75 (Cosine Similarity)</td><td>0.83</td><td>0.80</td><td>0.83</td><td>0.82</td></tr><tr><td></td><td>0.80 (Cosine Similarity)</td><td>0.84</td><td>0.80</td><td>0.85</td><td>0.82</td></tr><tr><td></td><td>0.85 (Cosine Similarity)</td><td>0.83</td><td>0.70</td><td>0.82</td><td>0.81</td></tr></table>

Table 7: Impact of graph construction thresholds on the performance of ESGPred

## 5 Discussion

The experimental results demonstrate that predicting ESG scores substantially improves when the semantic content of sustainability reports and existing company relationships are jointly considered. The systematic improvement achieved by graph-based configurations, as opposed to purely text-based ones, suggests that a company’s ESG performance cannot be fully characterized by treating each report as an isolated information. Rather, similarities in organizational characteristics, business models, and corporate practices of the companies which the reports refer to provide a relational contex that contributes to better interpret the report’s information. The ESGPred’s ability to generalize to individual Environmental, Social, and Governance dimensions further supports this result, as the graph-based component yields significant improvements over the textual baseline in the prediction of the scores of each single dimension.

ESGPred may have significant implications for developing automated ESG assessment systems. First, it uses sustainability reports and company metadata. These sources can be collected and processed on a large scale. This ofers a more reproducible and adaptable methodology than entirely proprietary assessment pipelines. Second, the ESGPred’s ability to predict both the overall ESG rating and its individual components suggests that the support of company relationship information is not limited to a single classification target.

In any case, we would like to strongly point out that ESGPred and the related frameworks should be viewed as tools that support analysts’ judgment, rather than replace it, especially given the complexity and multidimensional nature of ESG criteria.

ESGPred has also some limitations that define its current capabilities. First, we used LSEG ratings as the ground truth. Therefore, ESGPred learns to replicate a specific rating methodology. However, given the well-known divergence among ESG providers, it would be desirable to ensure that ESGPred can also learn to replicate ratings produced according to diferent criteria. Additionally, the current formulation categorizes scores into four classes and does not directly model the diferences between companies within the same class. Finally, ESGPred considers static graphs and a limited set of company relationships. However, ESG performance and company relationships evolve over time.

These changes may depend on information not represented in the used reports or metadata, such as supply chain relationships, ownership structures, and financial indicators. These limitations suggest enhancements to ESGPred, such as the usage of temporal and heterogeneous sustainability graphs, as well as graph explainability methods that can identify which company relationships contribute most to the final assessment.

## 6 Conclusion

This paper presented ESGPred, a modular, graph-based framework that predicts ESG scores from company sustainability reports. Unlike conventional approaches that rely solely on textual analysis, ESGPred combines domain-specific language representations with relational graph learning. This enables the prediction process to leverage the semantic content of sustainability reports and the relationships among companies.

ESGPred’s workflow consists of four main steps: (i) supervised fine-tuning of FinBERT on ESGrelated documents; (ii) generation of embeddings from sustainability reports; (iii) construction of sustainability graphs representing diferent kinds of relationship between sustainability reports and/or the companies to which they refer; and (iv) prediction through graph-based learning architectures. A key characteristic of ESGPred is its flexibility, which allows the embedding model, sustainability graph construction strategy, graph-based predictive model, and target sustainability metric to be modified and evaluated independently. An experimental analysis conducted on a dataset of sustainability reports and ESG ratings demonstrated the efectiveness of integrating textual and relational information. Across all evaluated configurations, graph-based models outperformed purely textual baselines. These results highlight the importance of modeling latent relationships between companies and suggest that descriptive information is a valuable resource for constructing relational structures (such as sustainability graphs) in ESG assessment tasks.

Future research directions include: (i) investigating dynamic and temporal graph structures that can support the analysis of the evolution of ESG performance over time; (ii) integrating additional heterogeneous information sources, such as financial indicators, supply-chain relationships, and ownership networks in the construction of sustainability graphs; and (iii) exploring graph-based structures and predictive models that can handle multiple kinds of entity and relationship simultaneously.

## Data Availability

The datasets and code used for our study are publicly available at https://anonymous.4open.scie nce/r/ESGPred-25BE/.

## References

[1] F.K. Alarfaj. A Comprehensive Study on Automated Extraction from Financial Reports Using NLP Approaches. Arabian Journal for Science and Engineering, 51(9):11843–11891, 2026. Springer.

[2] E. Alio˘gullari and E. Yildiz. Forecesting ESG scores in European countries using machine learning algorithms. Neural Computing and Applications, 38(10):379, 2026. Springer.

[3] G. Ang, Z. Guo, and E. Lim. On Predicting ESG Ratings Using Dynamic Company Networks. ACM Transactions on Management Information Systems, 14(3):1–34, 2023. ACM.

[4] F. Berg, J.F. K¨olbel, and R. Rigobon. Aggregate confusion: The divergence of ESG ratings. Review of finance, 26(6):1315–1344, 2022. Oxford University Press.

[5] B. Bernhardt, C. Marciano, S. Gigli, L. Maddalena, and M. Guarracino. Understanding ESG Scores Through Network Analysis: A Study Using Graph Neural Networks. In Proc. of the Scientific Meeting of the Classification and Data Analysis Group of the Italian Statistical Society (CLADAG-VOC’25), pages 37–47, Cham, Switzerland, 2025. CLADAG-VOC.

[6] O. Chaouech. Predicting Corporate ESG Scores from Financial Performance and Environmental Indicators: A Machine Learning Framework. Journal of Cultural Analysis and Social Change, 10(3):2042–2056, 2025.

[7] J.C. Choi, Q. Chen, and S.J. Lee. Predicting ESG ratings by machine learning and analyzing influencing factors by XAI. In Proc. of the International Conference on Software Engineering and Information Management (ICSIM’24), pages 120–125, Suva, Fiji, 2024. ACM.

[8] M.A.F. Chowdhury, M. Abdullah, M.A.K. Azad, Z. Sulong, and M.N. Islam. Environmental, social and governance (ESG) rating prediction using machine learning approaches. Annals of Operations Research, pages 1–25, 2023. Springer.

[9] G. Cohen. ESG risks and corporate survival. Environment Systems and Decisions, 43(1):16–21, 2023. Springer.

[10] V. D’Amato, R. D’Ecclesia, and S. Levantesi. Fundamental ratios as predictors of ESG scores: A machine learning approach. Decisions in Economics and Finance, 44(2):1087–1110, 2021. Springer.

[11] V. D’Amato, R. D’Ecclesia, and S. Levantesi. ESG score prediction through random forest algorithm. Computational Management Science, 19(2):347–373, 2022. Springer.

[12] L. David, J. Wang, V. Angel, and M. Luo. China’s ESG scorecard: A predictive machine learning model. Corporate Social Responsibility and Environmental Management, 31(4):3468–3486, 2024. Wiley Online Library.

[13] V.P. Dwivedi and X. Bresson. A generalization of transformer networks to graphs. arXiv preprint arXiv:2012.09699, 2020.

[14] E. Escrig-Olmedo, M. J. Mu˜noz-Torres, and M.A. Fernandez-Izquierdo. Socially responsible investing: sustainability indices, ESG rating and information provider agencies. International Journal of Sustainable Economy, 2(4):442–461, 2010. Inderscience Publishers.

[15] B.T. Fagbemi, B.P. Saah, A.I. Nduka, and E.M. Aloke. The Evolution of ESG and Sustainability Reporting: A Review of Standards, Challenges, and Impacts on Corporate Transparency. Journal of Economics, Business, and Commerce, 2(2):297–305, 2025.

[16] J. Fischbach, M. Adam, V. Dzhagatspanyan, D. Mendez, J. Frattini, O. Kosenkov, and P. Elahidoost. Automatic ESG assessment of companies by mining and evaluating media coverage data: NLP approach and tool. In Proc. of the IEEE International Conference on Big Data (BigData’23), pages 2823–2830, Sorrento, Italy, 2023. IEEE.

[17] M. G¨unther, J. Ong, I. Mohr, A. Abdessalem, T. Abel, M.K. Akram, S. Guzman, G. Mastrapas, S. Sturua, B. Wang, N. Wang M. Werk, and H. Xiao. Jina embeddings 2: 8192-token general-purpose text embeddings fo long documents. arXiv preprint arXiv:2310.19923, 2023.

[18] S. Halid, R.A. Rahman, R. Mahmud, N. Mansor, and R.A. Wahab. A literature review on ESG score and its impact on firm performance. International Journal of Academic Research in Accounting Finance and Management Sciences, 13(1):272–282, 2023

[19] W. Hamilton, Z. Ying, and J. Leskovec. Inductive Representation Learning on Large Graphs. In Proc. of the International Conference on Neural Information Processing Systems (NeurIPS’17), Long Beach, CA, USA, 2017.

[20] M. Joubrel and E. Maksimovich. ESG data and scores. In Valuation and Sustainability: A Guide to Include Environmental, Social, and Governance Data in Business Valuation, pages 67–98. 2023. Springer.

[21] T.N. Kipf and M. Welling. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016.

[22] T. Krappel, A. Bogun, and D. Borth. Heterogeneous Ensemble for ESG Ratings Prediction. In Proc. of the KDD Workshop on Machine Learning in Finance (KDD-MLF’21), Singapore, 2021. KDD-MLF.

[23] H. Lee, J. Kim, and H. Jung. ESG-KIBERT: A new paradigm in ESG evaluation using NLP and industry-specific customization. Decision Support Systems, 193:114440, 2025. Elsevier.

[24] H. Lee, S. Lee, H. Park, J. Kim, and H. Jung. ESG2PreEM: Automated ESG grade assessment framework using pre-trained ensemble models. Heliyon, 10(4):e26404, 2024. Elsevier.

[25] J. Lee and M. Kim. ESG information extraction with cross-sectoral and multi-source adaptation based on domaintuned language models. Expert Systems with Applications, 221:119726, 2023. Elsevier.

[26] H.Y. Lin and B.W. Hsu. Empirical Study of ESG Score Prediction through Machine Learning—A Case of Non-Financial Companies in Taiwan. Sustainability, 15(19):14106, 2023. MDPI.

[27] T.Y. Lin, P. Goyal, R. Girshick, K. He, and P. Doll´ar. Focal loss for dense object detection. In Proc. of the IEEE International Conference on Computer Vision (ICCV’17), pages 2980–2988, Venice, Italy, 2017. ICCV.

[28] Z. Liu, D. Bao, S. Li, and H. Zhao. From compliance to commitment: A deep learning decision-support artifact linking corporate social performance to multidimensional employee perceptions. Decision Support Systems, page 114683, 2026. Elsevier.

[29] Z. Liu, D. Huang, K. Huang, Z. Li, and J. Zhao. Finbert: A pre-trained financial language representation model for financial text mining. In Proc. of the International Joint Conferences on Artificial Intelligence (IJCAI’21), pages 4513–4519, Yokohama, Japan, 2021. IJCAI.

[30] F. Maibaum, J. Kriebel, and J. N. Foege. Selecting textual analysis tools to classify sustainability information in corporate reporting. Decision Support Systems, 183:114269, 2024. Elsevier.

[31] I. Martin-Melero, R. Gomez-Martinez, M. L. Medrano-Garcia, and F. Hernandez-Perlines. Comparison of sectorial and financial data for ESG scoring of mutual funds with machine learning. Financial Innovation, 11(1):84, 2025. Springer.

[32] H.Q. Ngee, A. Ganesh, M.A.N. Azmi, T.Y. Tang, M. Mukred, F. Mohammed, and A. A. B. Ahmad. Environmental. Social and Governance (ESG) Scores Automation in Global Reporting Initiative (GRI) with Natural Language Processing. In Proc. of the International Conference on Internet Applications, Protocols, and Services (NETAPPS’24), pages 1–7, Kuala Lumpur, Malaysia, 2024. IEEE.

[33] S. Pasch and D. Ehnes. NLP for responsible finance: Fine-tuning transformer-based models for ESG. In Proc. of the IEEE International Conference on Big Data (Big Data’22), pages 3532–3536, Osaka, Japan, 2022. IEEE.

[34] S. Patel, A. Nath, and P. Desai. Predicting ESG Scores Using Machine Learning for Data-Driven Sustainable Investment. Analytics, 5(1):7, 2026. MDPI.

[35] S. Pawar, P. Dolhare, S. Fatangare, H. Gawhale, and A. Gadgil. Automated ESG Scoring and Prediction. In Proc. of the International Conference on ICT for Sustainable Development (ICT4SD’25), pages 196–204, Goa, India, 2025. Springer.

[36] P.R. Rau and T. Yu. A survey on ESG: investors, institutions and firms. China Finance Review International, 14(1):3–33, 2024. Emerald Publishing Limited.

[37] G. Sariyer, S. K. Mangla, S. Chowdhury, M. E. Sozen, and Y. Kazancoglu. Predictive and prescriptive analytics for ESG performance evaluation: A case of Fortune 500 companies. Journal of Business Research, 181:114742, 2024. Elsevier.

[38] T. Schimanski, A. Reding, N. Reding, J. Bingler, M. Kraus, and M. Leippold. Bridging the gap in ESG measurement: Using NLP to quantify environmental, social, and governance communication. Finance Research Letters, 61:104979, 2024. Elsevier.

[39] Y. Shi and T. Yao. ESG Rating Divergence: Existence, Driving Factors, and Impact Efects. Sustainability, 17(10):4717, 2025. MDPI.

[40] A. Sokolov, J. Mostovoy, J. Ding, and L. Seco. Building machine learning systems for automated ESG scoring. The Journal of Impact and ESG Investing, 1(3):39–50, 2021. Institutional Investor Journals Umbrella.

[41] D. Taskin, G. Sariyer, E. Acar, and E.C. Cagli. Do past ESG scores eficiently predict future ESG performance? Research in International Business and Finance, 74:102706, 2025. Elsevier.

[42] P. Velickovic, G. Cucurull, A. Casanova, A. Romero, P. Li\`o, and Y. Bengio. Graph Attention Networks. In Proc. of the International Conference on Learning Representations (ICLR’18), Vancouver, British Columbia, Canada, 2018. OpenReview.net.

[43] N. N. Y. Vo, X. He, S. Liu, and G. Xu. Deep learning for decision making and the optimization of socially responsible investments and portfolio. Decision Support Systems, 124:113097, 2019. Elsevier.

[44] M. Yu, E. Cao, and Y. Zhang. ESG investment under diferent supply chain power structures: decisions, impacts, and the triple win. Expert Systems with Applications, 312:131518, 2026. Elsevier.

[45] A. Zhang and J. Zhang. Renovation in environmental, social and governance (ESG) research: the application of machine learning. Asian Review of Accounting, 32(4):554–572, 2024. Emerald.

[46] J. Zhang and Z. Zhao. Corporate ESG rating prediction based on XGBoost-SHAP interpretable machine learning model. Expert Systems with Applications, 295:128809, 2026. Elsevier.

[47] M. Zhang, Q. Shen, Z. Zhao, S. Wang, and G.Q. Huang. Optimizing ESG reporting: Innovating with E-BERT models in nature language processing. Expert Systems with Applications, 265:125931, 2025. Elsevier.