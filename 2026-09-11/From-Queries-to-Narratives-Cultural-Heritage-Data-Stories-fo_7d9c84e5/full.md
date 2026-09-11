# From Queries to Narratives: Cultural Heritage Data Stories for Knowledge Graph Exploration and Quality Assessment

Tabea Tietz<sup>1,2</sup>, Torsten Schrade<sup>3</sup>, Etienne Posthumus<sup>1</sup>, Linnaea Söhn<sup>3</sup>, Jonatan Jalle Steller<sup>3</sup>, Jörg Waitelonis<sup>1,2</sup> and Harald Sack<sup>1,2</sup>

<sup>1</sup>FIZ Karlsruhe – Leibniz Institute for Information Infrastructure, Eggenstein-Leopoldshafen, Germany <sup>2</sup>Institute of Applied Informatics and Formal Description Methods (AIFB) of KIT, Karlsruhe, Germany <sup>3</sup>Academy of Sciences and Literature Mainz, Geschwister-Scholl-Straße 2, 55131 Mainz, Germany

## Abstract

Cultural-heritage knowledge graphs such as the NFDI4Culture Knowledge Graph contain millions of triples about artworks, music, inscriptions, historical events, and the people and places connected to them. For many users, however, discovering this knowledge can be dificult. While SPARQL can be learned, writing meaningful queries first requires an in-depth understanding of the graph’s data model, an investment many domain researchers and practitioners are unwilling to make. Even with existing user interfaces, a starting point and some guidance are usually needed, because the data contained in the graph is highly specialized, heterogeneous, and constantly growing, making it challenging to know what it contains or which questions it can answer. In this paper, we present data stories as a way not only to lower this barrier, but also to turn exploration into data-quality assessment, and thus combine accessible querying with the discovery of issues that remain hidden in aggregate statistics. In this contribution, a data story is understood as a narrative document that integrates explanatory text and images with executable SPARQL queries and their visualized results. It is described how they are authored against the graph and how they serve several purposes: guiding users through an unfamiliar graph, creating reproducible narratives, and surfacing data-quality issues previously hidden in aggregate statistics. The authoring platform LODEON including its Sparnatural and AI-supported authoring assistants is introduced as a proof-of-concept. Within the authoring environment, every claim made about the data can be backed by an explicit query, making these narratives transparent and reproducible. This paper also reflects on lessons learned from hands-on seminars with Master’s students and an international workshop held at DH2026. Early experience suggests that such data stories make cultural-heritage knowledge graphs more accessible for both exploration and quality assessment.

## Keywords

Cultural Heritage, Knowledge Graph, Storytelling, LLM, Data Story

## 1. Introduction

The NFDI4Culture Knowledge Graph<sup>1</sup> (NFDI4Culture-KG) currently contains approximately 150 million triples relating to data on material and immaterial cultural heritage, such as artworks, musical sources, inscriptions, historical events, and the people and places that connect them. Its scale makes it a valuable resource for the (German) cultural-heritage community, but also a challenge to approach for new users, domain experts, and the data providers whose collections it integrates. The graph indexes data from numerous data portals across a wide range of domains ranging from musicology, art history, architecture to the performing arts and media studies, each contributing according to their subject-specific standards. Like many domain-specific KGs, it is highly specialized, heterogeneous, and constantly growing in a way that users struggle to know what it contains or which questions it can help to answer. The graph can be explored through a public SPARQL endpoint. SPARQL can be learned, but writing meaningful queries first requires learning the underlying data model: the Culture Ontology (CTO) [1] at the domain level, the NFDIcore ontology at the mid-level [2, 3], and the Basic Formal Ontology (BFO) [4] on the upper level [4]. Many domain researchers and practitioners are hesitant to make this investment. The data can therefore also be explored via the Culture Data Search<sup>2</sup>, a multimodal interface for finding research data through a free-text and incipit search, filter options, or image input. However, with data this specialized, users still need a starting point and an idea of which questions the graph can answer. Feedback from the community over several years has made clear that users want guidance on what to ask and how to ask it, and they want searches to be reproducible, so that a query is permanent and can become part of the research record.

Data stories can provide this guidance. We understand a data story as a narrative document about the data in the KG that integrates explanatory text and images with executable SPARQL queries and their visualized results. In this contribution, data stories serve three purposes. First, they onboard new users, who can read a story and get an idea of what the graph contains and how the data in it is connected. Second, data stories allow users to go through an authoring process in which they can begin to explore their own questions. Third, the act of authoring a story surfaces data-quality issues that remain hidden in aggregate statistics.

In this paper we describe how a first NFDI4Culture data-stories environment was implemented based on the graph and realized with SHMARQL<sup>3</sup>, how it was used in teaching at Karlsruhe Institute of Technology and international workshops, and how the friction observed there motivated an efort towards an AI-assisted authoring and exploration environment. On that basis we introduce LODEON, together with the Sparnatural query builder [5] and an AI authoring assistant, as a proof-of-concept.

The use cases described in this contribution are tied to the cultural-heritage domain, with the NFDI4Culture-KG and in parts the German Memory Atlas (GeMeA), a KG over approximately 26.8 million objects from the German Digital Library (DDB)<sup>4</sup>. However, LODEON can serve any KG, and the problems of access, query formulation, and reproducibility it addresses are not specific to this field.

This paper is structured as follows: Chapter 2 introduces existing data stories environments and means of exploration. Chapter 3 introduces shortly the NFDI4Culture-KG. Chapter 4 describes the process of developing a data stories environment, discusses lessons learned in teaching and workshops and motivates the need for the AI-assisted environment LODEON. The LODEON prototype is described, followed by a discussion of lessons learned. The conclusion in Chapter 6 closes the paper.

## 2. Related Work

The approach presented here builds on two lines of research, the narrative presentation of data and the exploration and querying of KGs. Both are discussed in the following.

## 2.1. Data Stories and Narrative Visualization

The idea of guiding an audience through data with a narrative is well known in data journalism [6], where visualized data supports an authored story, for instance in "The Upshot" by The New York Times<sup>5</sup>. In the Digital Humanities (DH), several tools bring this format closer to research data: the CLARIAH Data Stories Editor ofers an interactive environment for creating and publishing data-driven research narratives [7], and the Carnegie Hall Data Lab experiments present collection data through exploratory, narrative interfaces.<sup>6</sup> These approaches share the goal of this contribution of making data more approachable through narratives. However, they difer in the way how the narrative relates to the underlying data. A story is typically authored over a fixed export, and a reader cannot see, run, or alter the query that produced a given statement. In the contribution presented in this paper, each claim about the underlying data in a story can be generated by a SPARQL query embedded in the narrative and executed against the live endpoint. As a result, the reader can inspect and re-run the query behind any statement, and a story’s results reflect the current state of the data rather than a snapshot fixed at authoring time.

## 2.2. Querying and Exploring Knowledge Graphs

Linked Data browsers such as LodLive [8], LODmilla [9], and LodView<sup>7</sup> let users navigate entities and their relationships, while query-oriented tools such as YASGUI [10] and the educational RDF Playground [11] support writing and running SPARQL. Visual query builders like Sparnatural [5] allow users to assemble queries without writing SPARQL by hand. Furthermore, large language models (LLMs) have been used widely to translate natural-language questions into SPARQL [12, 13]. Our contribution integrates the capabilities of SHMARQL, a Linked Data publishing platform that enables semantic web professionals to disseminate data, Sparnatural, and an LLM-based "AI Assistant" within a storytelling format that supports the guidance through the data in the KG and ensures reproducibility.

The contributions described above present solutions where narrative visualization ofers guidance but not reproducible access to the data or query tools ofer access and exploration, but no narrative guidance. The contribution of this paper is to combine both in one storytelling format over a live cultural-heritage KG.

## 3. Bridging the Distance with Data Stories in NFDI4Culture

NFDI4Culture is a consortium within the German National Research Data Infrastructure (NFDI) dedicated to material and immaterial cultural heritage. It brings together subject-specific portals and collections from architecture, art history, design, musicology, the performing arts, and media studies, which are maintained by diferent institutions and described with various metadata standards [14, 1]. To make these decentralized resources findable and interoperable, NFDI4Culture has been creating the NFDI4Culture-KG, a semantic index that ofers a single point of access to research data while the data itself remains stored and owned by the contributing institutions. The graph is modeled with the Culture Ontology (CTO), which extends the mid-level ontology NFDIcore and is aligned with BFO [15]. The NFDI4Culture-KG is published through a public SPARQL endpoint<sup>8</sup> and can be explored using the graphical user interfaces within the Culture Information Portal.

Data has been integrated into the graph through provider-specific feeds. Once an institution or a researcher is ready to contribute, a data feed is created and maintained via the Culture Information Portal. The provider’s data is then processed by an Extract–Transform–Load (ETL) pipeline, called "Culture Kitchen" [16, 17]. Currently the KG contains more than 3.7m resources from 18 data feeds contributed by the cultural-heritage community in Germany, with the potential to grow given the 88 data portals listed by NFDI4Culture. Even though aggregating decentralized resources into a shared index is necessary, this alone is not suficient for their reuse. As Borgman and Groth argue, efective reuse requires overcoming the existing distance between those who create and those who intend to reuse the data [18].

As part of NFDI4Culture, user stories<sup>9</sup> were collected from the German cultural-heritage community to capture requirements for the graph and the Culture Information Portal [14]. However, it became clear that many contributors did not have a concrete idea of what the graph would contain, how it should be used, and how it might help to answer specific research questions. Over the years it has become clear that these challenges are not resolved by the detailed ontology documentation, the public SPARQL endpoint, or search interfaces alone, because some conventions that matter for querying only become apparent in use and users require starting points to create meaningful searches using the UI. Therefore it is necessary to explicitly showcase the users what knowledge the graph contains and which questions can be answered. Furthermore, users need to see questions answered against the live graph, i.e. a working query that can be read, adapted to a related question, and cited as a permanent part of the research process.

In order to fulfill these needs, the data stories laboratory<sup>10</sup> has been developed within NFDI4Culture featuring an AI-assisted workbench, which is described in the following Chapter 4. However, these needs discovered within NFDI4Culture are not specific to this particular project or to cultural heritage, as any KG containing heterogeneous and continuously growing sources confronts its users with some level of distance between a documented schema and the research questions the data can actually answer. Data stories represent a form of knowledge exchange that helps to close that distance.

## 4. Data Stories in NFDI4Culture from Consumption to Creation and AI Assisted Exploration

This section follows data stories in NFDI4Culture from consuming existing stories created by domain experts to the first environment in which users assume the role of creators within a SHMARQL-based authoring environment. This is followed by lessons learned and the introduction of the proof-ofconcept workbench LODEON, which adds a visual query builder and an AI assistant for authoring and exploration.

## 4.1. Consuming Data Stories as an Entry Point to Structured Data

The graph’s discoverability is shaped by its modular design, the alignment of CTO with NFDIcore and BFO, and the deliberately light-weight ontology. Since it integrates multiple providers, its coverage remains uneven, with information density depending on each provider’s specific domain. While external identifiers such as RISM<sup>11</sup> or GeoNames<sup>12</sup> and classification systems like Iconclass<sup>13</sup> or Getty AAT<sup>14</sup> facilitate data integration and querying, efective utilization requires knowledge of how they are represented within the underlying data. These challenges are further detailed in [1]. Data stories help to address them by pairing specific questions with the queries that execute them over the graph. Embedding these queries within the narratives makes data conventions visible, demonstrates modeling patterns, illustrate what a given feed contains, and ensures that queries remain valid as the graph evolves.

One example of a data story based on the NFDI4Culture-KG is titled "An Italian Data Journey"<sup>15</sup>, depicted in Fig. 1. It was authored by an expert in Digital Humanities and leverages services provided by NFDI4Culture and EOSC<sup>16</sup>. The story exemplifies how data federation with European infrastructures can significantly enhance the interoperability of research data and create multimodal research perspectives. Methodologically, it ranges from genre distribution analyses to geospatial mappings of opera premiere locations and music information retrieval through federated SPARQL queries. For a reader, its value is twofold. The story opens with a simple overview query that lists the data portals feeding the graph and the number of records of each type it holds, allowing new users to immediately understand the

# An Italian Data Journey

![](images/d57991f95de238f8bec475323d63758aa5ba7a3d331d3434521669212da828a9.jpg)  
Figure 1: Data Story E6263 titled "An Italian Data Journey" by Torsten Schade

Analysing research data about 18th century Italian opera using the Culture Knowledge Graph and federated European research infrastructures

![](images/4b10f862212b9a611d137953522fd1c20f96626ffc3eb2d20e9d19b525c57600.jpg)  
Daniel de Lafeuille, Nouvelle Carte D'Italie - Nieuwe Kaart van Italien, 1706, Wikimedia Commons, Public Domain  
Abstract: This data story illustrates a digital exploration of reserach data on opera holdings of the Doria Pamphilj Archive by the Partitura project of the German Historical Institute in Rome (DHI Rome). By enriching the Partitura dataset with established authority sources such as Wikidata, RISM, GeoNames, and transforming it to LOD, new analytical insights into the

scale and composition of the available data. The subsequent analyses serve as reusable patterns: each embedded query can be opened, inspected, and adapted ensuring readers gain both direct answers and templates for their own questions.

Another example titled "A Deep Dive into NFDI4Culture’s Integration Workflows"<sup>17</sup> explores the connections between RISM Online Musical Sources<sup>18</sup> (an access point to over 1m musical sources of the Répertoire International des Sources Musicales) and the Gregorovius letters<sup>19</sup> through the lens of the NFDI4Culture-KG. The Gregorovius edition contains 1,093 annotated pieces of correspondence from the historian Ferdinand Gregorovius (1821-1881), whose heritage from the 19th century testifies to a rich engagement with intellectual-historical movements and musicians of his time.

This data story, further described in [19], demonstrates with query examples the way heterogeneous data are integrated in the KG and how external identifiers help to link once entirely separate feeds in a way that they can be queried through a single access point.

Both story examples illustrate which questions the graph can answer and how to construct them. Furthermore, they demonstrate how heterogeneous data is integrated and connected in the first place.

![](images/9843dbd5bc4cd012c8d2469678e1c7ec80c6a6aad4806f04bb918d40dafcec32.jpg)

## Painter Biography Explorer

Use the explorer below to trace the lives and works of individual Baroque ceiling painters through an interactive map and a chronological list, Select a painter to see

• Geographic journey: Where they worked throughout their career

• Chronological works: All paintings ordered by date and grouped by building

• Detailed information: Click any painting or building for more details

![](images/f8b16963bfb916cb7c217cace4023e458e62c478184f9e82cb4c5de7c38e5701.jpg)  
Figure 2: Explorer to trace the lives and works of individual Baroque ceiling painters through an interactive map, as part of the data story "Baroque Ceiling Paintings in Germany" by Thanos Drossos, Robin Kleemann, YiMin Cai.

However, consuming a data story remains guided by the author’s own narrative. To answer their own research questions, users must turn from consumer to author by creating their own data stories.

## 4.2. Authoring Data Stories: A SHMARQL-based Environment

The SHMARQL platform enables the creation and publication of Linked Data applications by combining an RDF triplestore with Markdown-based querying and visualization features. Users can ingest raw triple files into Oxigraph<sup>20</sup>, an integrated high-performance triplestore for medium-sized KGs, or connect any SPARQL-compliant HTTP endpoint such as QLever [20], for larger KGs. By adding a custom Markdown code-block extension, SPARQL queries are executed directly, displaying their end-results within the narrative. In early versions of SHMARQL, configuration parameters for the Plotly<sup>21</sup> visualization library were embedded within SPARQL query comments. This combined each query and its visualization setup into a single, portable and cacheable component. Rather than serving static graphs or tables, the system keeps these visualizations tied to live, editable queries. Readers can access and modify the underlying queries for further exploration or to inspect data provenance. Ultimately, a defining strength of this environment is that every claim in a story is backed by an explicit, inspectable query.

## 4.2.1. Lessons Learned on SHMARQL-supported Story Creation

Insights into how the first SHMARQL-based authoring environment was received in practice came from a hands-on Master’s seminar at Karlsruhe Institute of Technology (KIT) over several semesters.

![](images/7f6eebc80b06bd579ee9025f42fe35c6955cd8759241d0c7539dba1d458b875e.jpg)  
Figure 3: The LODEON authoring environment on the example of the story "Long John SPARQL and the Secret of Data Island" by Torsten Schrade

Students at KIT were provided with the NFDI4Culture-KG as a fixed resource for their work. The data story "Baroque Ceiling Paintings in Germany"<sup>22</sup> is one example of a result of the course and is depicted in Fig. 2. The students independently selected their target data, research questions and the overall narrative. Coming from backgrounds in computer science and economics, most had no prior experience with digital humanities or the domain-specific research questions typical of this discipline. Students had varying levels of prior experience in semantic web technologies ranging from proficient users to complete beginners.

The authoring process proved to be an efective method for students to learn the graph’s modeling patterns. By providing a self-contained story with a defined research goal and a fixed set of questions, the task gave data exploration a clear purpose, demonstrating far greater efectiveness than confronting participants with the open SPARQL endpoint alone. In pursuing their goal, the students became familiar with the data feeds and their underlying structures, and after initial onboarding, successfully learned to query and visualize the data. Participants responded positively to several key design choices. In particular, they appreciated the seamless integration of queries with their corresponding result integrated directly within the narrative, as well as the ability to embed Plotly visualizations alongside the text. Furthermore, the use of Markdown was widely valued for keeping the authoring process lightweight and intuitive. However, writing SPARQL queries from scratch remained a considerable hurdle for participants without prior experience. As expected, the initial SHMARQL-based environment alone could not fully resolve this challenge. This limitation directly motivated the development of an environment that provides more active query construction support, as described in the following section.

## 4.3. Towards AI Assisted Authoring with LODEON

LODEON<sup>23</sup> integrates KGs, WYSIWYG note-taking, evidence-based exploration, and guided AI assistance into a unified workbench. This enables users to interact with KGs, document their insights, and produce media-rich visualizations (cf. Fig. 3). Its local-first, data-sovereign architecture allows users to keep full control over their data and model choices on their own infrastructure. Currently, the public workbench supports the utilization of both the NFDI4Culture-KG and GeMeA, a KG containing data from the German Digital Library (DDB)<sup>24</sup>. LODEON facilitates an evidence-based KG workflow. Authors can select an active KG, formulate questions, gather evidence, and synthesize findings into structured notes or narratives. SHMARQL remains part of LODEON and provides the underlying querying and rendering of results against the live endpoint. Query construction is further supported both by the visual query builder Sparnatural and by an AI assistant that ofers graph-oriented help, depicted in Fig. 4a. The AI assistant is built on a Model Context Protocol (MCP)<sup>25</sup> shaped tool layer that makes its assistance traceable. The assistant can inspect the profile of the active KG, look up ontology terms, validate and run queries. It then returns proposals that the author can insert directly into the story. On this basis it can generate SPARQL queries, suggest how a story might continue, and create visualizations. To lower the initial barrier, the assistant starts from a set of curated query suggestions. In the local version of LODEON, users can connect their own model that exposes an OpenAI-compatible API.

![](images/ff5e81817f796b7e54262f1c458973c6f890b5a7c70a1dff3f8ffb8013f1b463.jpg)  
(a) LODEON AI assistant for KG exploration and data story authoring.  
(b) LODEON metadata panel.  
Figure 4: Close-ups of LODEON authoring workbench: (a) the AI assistant and (b) the metadata panel.

Query results and visualizations can be inserted into the story as interactive nodes. Beyond Plotly charts, a story can incorporate a range of specialized nodes, including IIIF images, RAWGraphs visualisations, and a node to embed remote content, such as videos or interactive 3D models, as depicted in Fig. 5 in Appendix A.

LODEON also provides a metadata panel, shown in Fig. 4b. The author can add a short abstract, keywords, a language tag, name contributors together with their roles, afiliations, and identifiers such as ORCID. Furthermore, the user can assign a license and one or more persistent identifiers under which the story will be published. This information is carried into the exported story, so that a published data story can be referenced and reused like any other research output. Once a story is complete, it can be previewed in the way it will be rendered and then exported as a zip file or published. Throughout this process, LODEON assists, suggests, queries, and visualizes, while the author remains in full control of what enters the story in the end.

## 4.3.1. Availability of LODEON

A demo video of the LODEON proof-of-concept is available on the Web<sup>26</sup>, and the public workbench can be used online<sup>27</sup>. Due to the associated costs, LODEON’s AI assistant is not yet freely available on the public workbench, but users are free to run LODEON locally and connect their own models, as it is available as MIT licensed open source software on GitLab <sup>28</sup>.

## 4.3.2. Lessons Learned on LODEON and AI Assisted Data Exploration

LODEON was presented and tested as a proof-of-concept in a hands-on workshop at DH2026 in Daejeon, South Korea [21]<sup>29</sup>. Participants worked with a pre-structured data story and completed tasks involving the AI assistant, adding SPARQL queries and their results, building queries with Sparnatural, and creating visualizations. For this workshop, users received a login in order to use the AI assistant and could choose between two models, Qwen 3.5 and EOSC GLM 4.7 Flash. Feedback was gathered through live polls using the Claper tool<sup>30</sup> and largely through verbal discussion, since only about half of the roughly 60 participants joined the Claper event.

The audience came from disciplines across languages and literature, digital humanities, history and archaeology, the arts, and computer science. Most participants had prior exposure to KGs, worked with graph databases, or integrated a KG into an application. A minority described themselves as comfortable with SPARQL. Data stories were largely new to the group, and most of those who participated in the poll had neither read nor written one before. The reception of LODEON in general was widely positive. The participants judged LODEON useful for their own work within cultural heritage and digital humanities and beyond. No participant judged the tool as irrelevant or unusable. The AI assistant in particular received a more mixed assessment. On a scale from 1 (not at all) to 10 (perfect), ratings of its results clustered in the lower range, mostly between 2 and 4.

Graphical user interfaces and data stories alike are always subject to design decisions that predetermine what users are asked to do. These decisions include available filters, the order in which information is displayed or a pre-curated question of the AI assistant a data story author can choose. These design decisions quietly shape the space of possible questions and therefore may impact the research results. Direct access to the data, e.g. by means of a SPARQL query, is the counterweight, as it allows users to pursue their own interests and lines of research. One question that motivated the creation of LODEON was whether AI could bridge the two, giving users direct access to the data while also helping them understand what can be searched for. Experience now suggests that it can do so partially and under certain conditions. In the workshop, the AI assistant lowered the barrier to query construction and helped participants understand a graph they did not know beforehand. However, it sometimes also produced unreliable answers, particularly for aggregate questions such as counts, but its output looked equally trustworthy whether it was correct or not. In the aftermath of the workshop, much care has been taken to improve the quality and visual transparency of the AI assistant. The infrastructural conditions under which AI is used in research add to this, since availability is also determined by costs. On the public workbench the assistant cannot be ofered freely yet and during the workshop, results had to be capped for eficiency. That means, when using the public workbench, performance and costs currently also determine what users see and what they do not. This is another reason why authors are advised to run their own models locally instead.

These conditions do not argue against AI as a bridge, but there are risks and limits. The principle that every claim in a story remains backed by an explicit, inspectable query is necessary and relevant. However, the workshop also made clear that it is not suficient on its own. Transparency has to further improve and it has to be made clear where a result may be wrong and where an answer also reflects the limits of the infrastructure and not only the data. Only then can AI provide both access and understanding without quietly reintroducing the design decisions that direct access to the data was meant to escape.

## 5. Quality Assessment

The quality of KGs and Linked Data has been studied extensively. For instance, Färber et al. assess DBpedia, Freebase, OpenCyc, Wikidata, and YAGO against a common set of data-quality criteria, such as accuracy, consistency, and completeness, applied through metrics computed over a graph as a whole [22]. Such aggregate measures are efective at scale, but they can hide localized problems in individual data feeds, which might originate in the original data, the data integration step, or could reveal additional requirements for the ontology. Authoring a data story brings such problems to light, since concrete research questions require records to be retrieved and inspected in depth. In this way, issues such as missing entities, or incomplete metadata become visible in a way that graph-wide statistics do not reveal. Story authoring is therefore better understood as a complement to automated quality assessment than as a replacement for it.

In the process of authoring a data story during the seminar at KIT<sup>31</sup>, the Master’s students were not only able to test the SHMARQL-based authoring environment, but also uncovered a range of errors in the graph. The process of writing a narrative helped in discovering missing data points, and missing controlled vocabulary entries to enable standardized data linking. The full report is available on Github<sup>32</sup>. As a consequence, several of the missing fields could be added to the NFDI4Culture-KG in subsequent ingestion rounds, and missing controlled-vocabulary entries needed for standardized linking could be identified. Currently, users can only report on issues discovered in the KG while authoring a data story by means of Github issues or by contacting NFDI4Culture. However, in future development, a dedicated and more structured means of data quality issue reporting with respect to the KG is planned. This also involves the report of missing connections, which may reveal new requirements for the underlying ontology, as more and more users contribute their stories and their research questions.

## 6. Conclusion

This paper presents data stories as a way to make the large, heterogeneous, and continuously growing NFDI4Culture-KG more accessible, and to turn its exploration into a form of quality assessment. The paper follows data stories in NFDI4Culture from reading to authoring. Existing stories, such as An Italian Data Journey and A Deep Dive into NFDI4Culture’s Integration Workflows, give new users an entry point to the graph and a set of reusable query patterns. A first authoring environment built with

SHMARQL has been extended by the LODEON workbench, a proof-of-concept that adds Sparnatural and an AI assistant.

Early experience, including a workshop at DH2026 suggests that this approach is useful. The reception of LODEON is widely positive, and the AI assistant lowers the barrier to query construction and helps participants understand an unfamiliar graph. At the same time, it became clear that the AI assistant can serve as a bridge to the data in a research process only when its own limits are made transparent. Immediate next steps include a quantitative evaluation of the AI assistant focusing on the accuracy of the SPARQL queries it generates, and on which model performs best for the specific task of data story generation at the lowest computational cost. LODEON will be further developed and improved based on the existing lessons learned and the upcoming quantitative results, before the current proof-of-concept can be oficially released. The challenges addressed in this paper are not specific to the NFDI4Culture project or to the cultural heritage field. Therefore, future development will also include the use cases in domains, like Materials Science and Sports Science.

## Acknowledgments

This work is funded by Deutsche Forschungsgemeinschaft (DFG), project number 441958017. We would also like to thank YiMin Cai, Thanos Drossos, and Robin Kleemann who completed the Master’s seminar "Knowledge-Driven AI" with the topic "Telling Data Stories with Semantic Technologies and Generative AI" at Karlruhe Institute of Technology (KIT). As part of the seminar, the students authored the data story "Baroque Ceiling Paintings in Germany" and provided useful insights into existing data quality challenges in the NFDI4Culture-KG.

## Declaration on Generative AI

During the preparation of this work, the author(s) used Claude Opus 4.8 in order to: Grammar and spelling check, and paraphrase and reword. After using these tool(s)/service(s), the author(s) reviewed and edited the content as needed and take(s) full responsibility for the publication’s content.

## References

[1] T. Tietz, E. Posthumus, L. Söhn, J. J. Steller, O. Bruns, J. Waitelonis, T. Schrade, H. Sack, Knowledge Representation and Discovery for Cultural Heritage Research Data with CTO and SHMARQL, in: 5th International Workshop on Scientific Knowledge: Representation, Discovery, and Assessment, Sci-K 2025. Proceedings: Co-located with 24th International International Semantic Web Conference (ISWC 2025), Nara, Japan, November 2, 2025., 2025.

[2] J. Waitelonis, et al., NFDIcore Ontology, 2025. URL: https://nfdi.fiz-karlsruhe.de/ontology/3.0.1.

[3] O. Bruns, T. Tietz, J. Waitelonis, E. Posthumus, H. Sack, NFDIcore 2.0: A BFO-Compliant Ontology for Multi-Domain Research Infrastructures, 2024. URL: https://arxiv.org/abs/2410.01821. arXiv:2410.01821.

[4] J. N. Otte, J. Beverley, A. Ruttenberg, BFO: Basic Formal Ontology, Applied Ontology 17 (2022) 17–43.

[5] T. Francart, Sparnatural: A Visual Knowledge Graph Exploration Tool, in: European Semantic Web Conference, Springer, 2023, pp. 11–15.

[6] Á. Arrese, “In the Beginning were the Data”: Economic Journalism as/and Data Journalism, Journalism studies 23 (2022) 487–505.

[7] W. Sanders, R. Ordelman, M. Wigham, R. Klein, J. van Gorp, J. Noordegraaf, Developing Data Stories in Digital Humanities: Challenges and Protocol, DH Benelux Journal 5 (2023) 1–17.

[8] D. V. Camarda, S. Mazzini, A. Antonuccio, LodLive, Exploring the Web of Data, in: Proceedings of the 8th International Conference on Semantic Systems, 2012, pp. 197–200.

[9] A. Micsik, S. Turbucz, Z. Tóth, Exploring Publication Metadata Graphs with the LODmilla Browser and Editor, International Journal on Digital Libraries 16 (2015) 15–24.

[10] L. Rietveld, R. Hoekstra, YASGUI: Not Just Another SPARQL Client, in: Extended Semantic Web Conference, Springer, 2013, pp. 78–86.

[11] B. Inostroza, R. Cid, A. Hogan, RDF Playground: An Online Tool for Learning About the Semantic Web, in: Companion Proceedings of the ACM Web Conference 2023, 2023, pp. 111–114.

[12] H. Khorashadizadeh, A. F. Zahra, E. Morteza, I. Frédéric, T. Sanju, M. Nandana, G. Jinghua, S. Soror, F. Benamara, G. Sven, Research Trends for the Interplay between Large Language Models and Knowledge Graphs, in: 1st International Workshop on Data Management Opportunities in Unifying Large Language Models+ Knowledge Graph. Workshop at the 50th International Conference on Very Large Data Bases (VLDB 2024), 2024.

[13] M. R. A. H. Rony, U. Kumar, R. Teucher, L. Kovriguina, J. Lehmann, Sgpt: A Generative Approach for SPARQL Query Generation from Natural Language Questions, IEEE access 10 (2022) 70712–70723.

[14] T. Tietz, O. Bruns, L. Söhn, J. Tolksdorf, E. Posthumus, J. J. Steller, H. Fliegl, E. Norouzi, J. Wait elonis, T. Schrade, H. Sack, From Floppy Disks to 5-Star LOD: FAIR Research Infrastructure for NFDI4Culture, in: 3rd Workshop on Metadata and Research (objects) Management for Linked Open Science (DaMaLOS), co-located with ESWC 2023, Publisso, 2023.

[15] H. Sack, T. Schrade, O. Bruns, E. Posthumus, T. Tietz, E. Norouzi, J. Waitelonis, H. Fliegl, L. Söhn, J. Tolksdorf, J. J. Steller, A. Azócar Guzmán, S. Fathalla, A. Zainul Ihsan, V. Hofmann, S. Sandfeld, F. Fritzen, A. Laadhar, S. Schimmler, P. Mutschke, Knowledge Graph Based RDM Solutions: NFDI4Culture - NFDI-MatWerk - NFDI4DataScience, in: 1st Conference on Research Data Infrastructure, 2023.

[16] O. Bruns, T. Tietz, L. Söhn, J. J. Steller, S. R. Ondraszek, E. Posthumus, T. Schrade, H. Sack, What’s Cooking in the NFDI4Culture Kitchen? A KG-based Research Data Integration Workflow, in: 4th Workshop on Metadata and Research (objects) Management for Linked Open Science (DaMaLOS), co-located with ESWC, 2024.

[17] J. J. Steller, L. C. Söhn, J. Tolksdorf, O. Bruns, T. Tietz, E. Posthumus, H. Fliegl, S. Pittrof, H. Sack, T. Schrade, Communities, Harvesting, and CGIF: Building the Research Data Graph at NFDI4Culture, in: DHd2024: Quo Vadis, 2024.

[18] C. L. Borgman, P. Groth, From Data Creator to Data Reuser: Distance Matters, Harvard Data Science Review 7 (2025). Https://hdsr.mitpress.mit.edu/pub/2mvqwgmf.

[19] L. C. Söhn, T. Tietz, J. J. Steller, P. Kehrein, A. Büttner, O. Bruns, E. Posthumus, J.-P. Grünewälder, J. Hörnschemeyer, C. Sander, V. Grund, H. Fliegl, H. Sack, T. Schrade, NFDI4Culture Integration Stories: Bridging Gaps between Isolated Research Resources, in: Digital Humanities 2025: Accessibility & Citizenship (DH2025), Zenodo, Lisbon, Portugal, 2025. URL: https: //doi.org/10.5281/zenodo.16570076. doi:10.5281/zenodo.16570076.

[20] H. Bast, J. Kalmbach, R. Textor-Falconi, C. Ullinger, Sparqloscope: A Generic Benchmark for the Comprehensive and Concise Performance Evaluation of SPARQL Engines, in: International Semantic Web Conference, Springer, 2025, pp. 22–40.

[21] T. Schrade, L. C. Söhn, J. J. Steller, A. Büttner, T. Tietz, E. Posthumus, O. Bruns, H. Fliegl, H. Sack, The Culture Data Story Laboratory: Engaging Knowledge Graphs through Executable Narratives, in: Digital Humanities 2026: Book of Abstracts, The Alliance of Digital Humanities Organizations (ADHO), 2026, pp. 76–78. URL: https://doi.org/10.5281/zenodo.22656106.

[22] M. Färber, F. Bartscherer, C. Menne, A. Rettinger, Linked Data Quality of DBpedia, Freebase, Opencyc, Wikidata, and Yago, Semantic Web 9 (2017) 77–129.

## A. LODEON Features

![](images/d8ce71b7963ffd89c838c0fac4c1ea289097b437f519b3a53b44679f43ad7b9f.jpg)  
Figure 5: LODEON supports the embedding of remote content, like the interactive 3D model from the data portal semantic kompakkt (https://nfdi4culture.de/id/E3752).