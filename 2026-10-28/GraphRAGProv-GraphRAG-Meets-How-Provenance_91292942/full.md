# GraphRAGProv: GraphRAG Meets How-Provenance

Wadie El Amrani Independent Toulouse, France elamrani@wadie.fr

Luis Galárraga   
INRIA   
Rennes, France   
luis.galarraga@inria.fr

![](images/7e0741bb45df27f25c3f4690415a0c8f3c8e7117faea7ed929ba905356102fce.jpg)  
Fi ure 1: (a) Source- rounded RDF statement occurrences constructed from text. (b) SPARQL uestion answerin and rovenance tracin<sub>g</sub> from answer bindin<sub>g</sub>s to RDF occurrences and exact source text evidence.

## Ab<sub>s</sub>t<sub>rac</sub>t

GraphRAG systems often use knowledge graphs to retrieve relevant context for an LLM that produces the final answer. That makes answers more inscrutable and therefore less appealing for domains where transparency is paramount. We present GraphRAGProv, an architecture that executes SPARQL queries over source-grounded RDF graphs to compute answers and their how-provenance poly nomials. These polynomials record which statement occurrences were used and how they were combined. Because each occurrence is linked to its source text, users can follow these links to check whether the extracted statements reflect the text and whether the query combines them to answer the question. Querying thereby becomes a way to inspect and evaluate the text-to-graph transformation. Our interactive demo examines 400 cases from two multi-hop QA benchmarks, revealing extraction and query errors even when the final answer is correct.

## 1 I<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>ti<sub>on</sub>

GraphRAG uses graph structure to connect information across a corpus of text. Microsoft GraphRAG builds entity relation graphs and community summaries, while HippoRAG builds an OpenIE graph and uses Personalized PageRank for retrieval [3, 6]. In many such architectures, the graph determines what evidence reaches an LLM, but the LLM still produces the final answer. A retrieved path or subgraph may therefore show what evidence was available to the LLM without specifying how the returned value was computed. This is limiting in critical settings such as medical diagnosis or computational justice, where the computation leading to an answer may itself need to be verified.

One approach is to produce the answer by querying the graph itself. This requires representing facts from the text documents as explicit statements in a data graph, such as subject-predicate-object triples in RDF. A question can then be translated into SPARQL and evaluated by a graph database engine, so that the query result is the final answer [7, 10]. While this approach is well established when a knowledge graph already exists [4, 7], applying it to document collections requires constructing the graph from text first, as in Neo4j’s LLM Graph Builder [9] and KAG [8]. KAG reports answering questions from retrieved subgraphs in 33%, 34%, and 18% of cases on HotpotQA, 2WikiMultiHopQA, and MuSiQue, respectively, and supplements graph reasoning with text retrieval when it yields no results. GraphRAGProv instead uses an LLM to translate questions into SPARQL, with the returned bindings constituting the answers and no text retrieval or LLM generation as a fallback. To make these computations inspectable, we augment query evaluation with database how-provenance [5], which identifies the contributing statements and records how they were combined. The database computes both the answers and their derivations, without requiring an LLM to interpret retrieved context. Each returned binding is annotated with a polynomial over statement identifiers:

![](images/d7d27fa0c5947c90075a9d67a8ad8558e5038ec5602f3aa2a6e191526012d8ea.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> 2: Gr<sub>ap</sub>hRAGPr<sub>ov a</sub>r<sub>c</sub>hit<sub>ec</sub>t<sub>u</sub>r<sub>e.</sub> An LLM <sub>e</sub>xtr<sub>ac</sub>t<sub>s se</sub>m<sub>a</sub>nti<sub>c s</sub>t<sub>a</sub>t<sub>e</sub>m<sub>e</sub>nt<sub>s a</sub>nd th<sub>e</sub>ir t<sub>e</sub>xt<sub>ua</sub>l <sub>ev</sub>id<sub>e</sub>n<sub>ce.</sub> Th<sub>e</sub> l<sub>oca</sub>liz<sub>e</sub>r r<sub>eso</sub>l<sub>ves</sub> evidence to exact source spans for RDF-star materialization. An LLM generates SPARQL, which NPCS rewrites to compute <sub>answer</sub> bi<sub>n</sub>di<sub>ngs an</sub>d h<sub>ow-provenance.</sub>

multiplication records joint use, while addition records alternative derivations. NPCS [2] computes these annotations by query rewriting over reified RDF statements.

A derivation identifies the statements used to compute an answer, but does not establish whether those statements are supported by the source text. GraphRAGProv makes this interpretation inspectable by preserving each extraction as a statement occurrence linked to its source passage and evidence spans. The extractor supplies quotations for the subject, predicate, and object; a deterministic localizer resolves them to exact character spans, and RDF-star reification preserves the occurrence identifier and its source links. NPCS [2] uses these identifiers in provenance polynomials, so following a factor recovers both the assertion used in the computation and the original text against which that assertion can be checked.

Together, query execution, how-provenance, and textual ground ing provide a way to inspect and evaluate the text-to-graph transformation. The extractor may misread a passage, entity identifiers may be inconsistent, or the generated SPARQL may fail to capture the question. This explicit chain allows users to examine whether the contributing statements reflect the source text and whether the query combines them as intended.

The present demonstration lets users move across this complete chain : question → SPARQL → answer binding → provenance polynomial → RDF occurrence → exact source spans and modify either the question or the SPARQL query to observe how the answer and its derivation change. We examine the availability and diagnostic value of this trace on 400 cases from 2WikiMultiHopQA and MuSiQue.

## 2 Gra<sub>p</sub>hRAGPro<sub>v</sub>

Graph construction. Figure 2(a) illustrates how GraphRAGProv transforms source paragraphs into an RDF knowledge graph. Ex tracting relational statements with textual evidence is well established: Stanford OpenIE retains token spans for extracted subjects, relations, and objects [1], while recent LLM-based approaches such as AEVS associate semantic triples with character-level evidence [11]. Following this line of work, an LLM interprets naturallanguage expressions as RDF statements $t _ { i } ~ = ~ ( s _ { i } , p _ { i } , o _ { i } )$ . For example, “Borhane Alaouié is a Lebanese film director” can yield (Borhane Alaouié, nationality, Lebanese). The relation nationality is inferred from the adjective “Lebanese” and ”is a”, rather than copied from the sentence. Alongside each statement, the LLM provides textual quotations supporting its components. A deterministic localizer resolves these quotations to exact source spans $E _ { i } ^ { s } , E _ { i } ^ { p } , E _ { i } ^ { o } .$ Quotations that cannot be located trigger a correction request to the LLM; unresolved quotations cannot be accepted as exact evidence. Symbolic chunk. We call a source-grounded statement occurrence a symbolic chunk. Each occurrence receives an identifier $i d _ { i } ,$ connecting its RDF triple to the source paragraph �<sub>�</sub> and the available evidence spans. Identical triples extracted from diferent passages retain separate occurrence identifiers and textual evidence. GraphRAGProv materializes these connections using reification with RDF-star through textprov:occurrenceOf, as illustrated in Figure 1(a).

Query execution. Figure 2(b) shows the answering process. Given a question � and the constructed graph, an LLM generates a SPARQL query �, which GraphDB executes to obtain answer bindings �. NPCS [2] rewrites the query to additionally compute how provenance $P _ { \mu }$ . For the Kafr Kasem example, the database returns Lebanese, with $P _ { \mu } = i d _ { 1 } \otimes i d _ { 2 }$ . Multiplication represents the joint use of statement occurrences, while addition represents alternative derivations [5]. Each provenance identifier resolves to its source evidence, completing the trace in Figure 1(b):

$$
q  Q  \mu  P _ { \mu }  i d _ { i }  ( u _ { i } , E _ { i } ^ { s } , E _ { i } ^ { p } , E _ { i } ^ { o } ) .
$$

## 3 Demonstration

The live demonstration is publicly accessible, and the source code is available under the MIT license.

![](images/2052bd30cc16900b93bb94483f9107604f67d5281abdf5c095903c2101d7a1b0.jpg)  
Figure 3: GraphRAGProv interactive workbench. Users can inspect SPARQL queries, follow answer derivations through how-<sub>p</sub>rovenance<sub>,</sub> and hi<sub>g</sub>hli<sub>g</sub>ht source ex<sub>p</sub>ressions associated with <sub>p</sub>artici<sub>p</sub>atin<sub>g</sub> RDF statements.

User Experience. Figure 3 shows the GraphRAGProv workbench. Users can explore recorded cases from 2WikiMultiHopQA and MuSiQue, each containing the benchmark question, source paragraphs, and constructed RDF graph. Selecting a graph statement highlights its supporting text and vice versa. Users can inspect the LLM-generated SPARQL query, execute it, and examine the returned answers and how-provenance polynomials. Selecting a provenance factor reveals the contributing statement and its textual evidence. The workbench also allows editing natural-language questions or SPARQL queries to observe changes in answers and derivations.

Implementation. GraphDB stores the RDF graphs, with named graphs separating source paragraphs. The React frontend communicates with a FastAPI backend. SPARQL queries are rewritten using a Rust implementation of NPCS<sup>1</sup> to compute answer bindings and how-provenance. Graph construction and query generation use DeepSeek-V4-Flash. The extraction<sup>2</sup> and query-generation<sup>3</sup> prompts can be found in the source code.

## 4 E<sub>va</sub>l<sub>ua</sub>ti<sub>on</sub>

Experimental setting. We examine 200 compositional questions from 2WikiMultiHopQA and 200 answerable questions from Musique.

![](images/da4e272932e409b88663fab5a8f90ad40fc361c232addc0ad1ddd553a9fdf42a.jpg)

For each case, the graph is constructed from its designated supporting passages, excluding document retrieval from the experiment. Graph construction and SPARQL generation use DeepSeek-V4- Flash, with access to the benchmark question, expected answer, and reasoning evidence. This supervised setting allows us to examine the resulting graphs and computations, but does not establish generalization to unseen questions. We retain the first query that passes mechanical validation and executes, without retrying based on answer correctness.

Trace availability. Table 1 reports the number of cases reaching each stage. A complete trace requires every factor in the decoded derivations to resolve to its statement occurrence and matching source spans. Complete traces are available for 193 of 200 2Wiki cases and 118 of 200 MuSiQue cases. All 311 decoded traces pass the occurrence and source-span checks, and their answer bindings match ordinary query execution. Sixteen MuSiQue cases return an answer without decoded provenance.

Case inspection. Table 2 illustrates what can be learned by inspecting the computation and its sources. In Meesaya Murukku, the answer is correct and the cited word “Indian” exists at the recorded ofsets, but it describes the film rather than its director. The error lies in the interpretation of the text, not in provenance resolution. Labyrinth reveals a diferent problem: the query returns the correct year using two individually supported assertions, yet never establishes that Acornsoft published the game. Its provenance faithfully records the executed computation, but that computation does not express the question. In contrast, the Caledon River case yields a suficient derivation from one passage containing both required relations, despite the benchmark designating two supporting passages.

<table><tr><td>Stage</td><td>2Wiki</td><td>MuSiQue</td><td>Total</td></tr><tr><td>Cases</td><td>200</td><td>200</td><td>400</td></tr><tr><td>SPARQL executed</td><td>200</td><td>198</td><td>398</td></tr><tr><td>Answer returned</td><td>193</td><td>134</td><td>327</td></tr><tr><td>Provenance decoded</td><td>193</td><td>118</td><td>311</td></tr><tr><td>Complete text trace</td><td>193</td><td>118</td><td>311</td></tr></table>

Table 1: Answer-to-source trace availabilit<sub>y</sub>.

Missing answers. An empty result provides no answer derivation to inspect. In Aldobrandino, the father relation and birth date are present in the graph, but attached to diferent IRIs representing the same person. The resulting join fails despite the required information being extracted. This failure is identified by inspecting the graph and query directly, rather than following provenance of a nonexistent result.

<table><tr><td>Question and case ID</td><td>Inspection finding</td></tr><tr><td>Meesaya Murukku (2Wiki) Which country is its director</td><td>Correct answer, India, but the cited text describes the film as Indian, not its director.</td></tr><tr><td>from?</td><td> $q  Q  \mu  P _ { \mu }  i d _ { i }$ </td></tr><tr><td>ba413fca0bd911eba7f7acde48001122 Labyrinth (MuSiQue)</td><td>Correct answer, 1986, but the query omits the</td></tr><tr><td>When did its publisher cease operating?</td><td>game-publisher relation.  $q  Q  \mu  P _ { \mu }  i d _ { i }$ </td></tr><tr><td>2hop__170823_120171</td><td></td></tr><tr><td>Aldobrandino (2Wiki) When was his father born?</td><td>Two IRIs represent the same father, preventing the join.</td></tr><tr><td>96a72db40bdd11eba7f7acde48001122</td><td> $q  Q  \mu = \emptyset  P _ { \mu }  i d _ { i }$ </td></tr><tr><td>Caledon River (MuSiQue)</td><td>One passage supports both required relations,</td></tr><tr><td>Where is the source of the receiving river?</td><td>although the benchmark designates two.  $q  Q  \mu  P \mu  i d _ { i }$ </td></tr></table>

Table 2: Re<sub>p</sub>resentative cases. Red identifies an unsu<sub>pp</sub>orted assertion, an incorrect <sub>q</sub>uer<sub>y</sub>, or an em<sub>p</sub>t<sub>y</sub> resu<sup>l</sup>t; gra<sub>y</sub> in<sup>d</sup>icates unavailable provenance stages. Questions are abbreviated<sub>,</sub> and case IDs identif<sub>y</sub> the saved ex<sub>p</sub>eriments.

## 5 C<sub>o</sub>n<sub>c</sub>l<sub>us</sub>i<sub>o</sub>n

An executable query answers what it asks of a graph, not necessarily what the user asked of the original text: answering the original question also requires the graph to preserve the relevant knowledge and the query to express the intended computation. GraphRAGProv makes this distinction inspectable. SPARQL specifies the computation, how-provenance identifies the contributing statement occurrences, and symbolic chunks connect them to their textual origins. These connections allow users to inspect the interpretations made by the LLM during graph construction. Our demonstrations show why these layers must be examined together: supported statements may participate in an incorrect query, correct answers may depend on unsupported interpretations, and missing answers may result from inconsistent entity representations rather than absent knowledge. Provenance exposes the chain to help users locate and diagnose failures.

## R<sub>e</sub>f<sub>erences</sub>

[1] Gabor Angeli, Melvin Jose Johnson Premkumar, and Christopher D. Manning. 2015. Leveraging Linguistic Structure for Open Domain Information Extraction. In Proceedings ofthe 53rd Annual Meeting ofthe Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). Association for Computational Linguistics, 344–354. doi:10.3115/v1/P15-1034

[2] Zubaria Asma, Daniel Hernández, Luis Galárraga, Giorgos Flouris, Irini Fundulaki, and Katja Hose. 2024. NPCS: Native Provenance Computation for SPARQL. In Proceedings ofthe ACM Web Conference 2024. 2085–2093. doi:10.1145/3589334. 3645557

[3] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. 2024. From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv preprint arXiv:2404.16130 (2024).

[4] Yanlin Feng, Simone Papicchio, and Sajjadur Rahman. 2025. CypherBench: Towards Precise Retrieval over Full-scale Modern Knowledge Graphs in the LLM Era. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Vienna, Austria, 8934–8958. doi:10.18653/v1/2025.acl-long.438

[5] Todd J. Green, Gregory Karvounarakis, and Val Tannen. 2007. Provenance Semirings. In Proceedings ofthe 26th ACM SIGMOD-SIGACT-SIGART Symposium on Principles ofDatabase Systems. 31–40. doi:10.1145/1265530.1265535

[6] Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. 2024. HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. In Advances in Neural Information Processing Systems, Vol. 37. doi:10.52202/079017-1902

[7] Yunshi Lan, Gaole He, Jinhao Jiang, Jing Jiang, Wayne Xin Zhao, and Ji-Rong Wen. 2021. A Survey on Complex Knowledge Base Question Answering: Methods, Challenges and Solutions. In Proceedings ofthe Thirtieth International Joint Conference on Artificial Intelligence. 4483–4491. doi:10.24963/ijcai.2021/611

[8] Lei Liang, Zhongpu Bo, Zhengke Gui, Zhongshu Zhu, Ling Zhong, Peilong Zhao, Mengshu Sun, Zhiqiang Zhang, Jun Zhou, Wenguang Chen, Wen Zhang, and Huajun Chen. 2025. KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation. In Companion Proceedings ofthe ACM on Web Conference 2025. Association for Computing Machinery, New York, NY, USA, 334–343. doi:10.1145/3701716.3715240

[9] Neo4j Labs. 2024. LLM Graph Builder: Neo4j Graph Construction from Unstructured Data Using LLMs. GitHub repository. https://github.com/neo4j-labs/llmgraph-builder Accessed 2026-08-29.

[10] Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang Tang. 2025. Graph Retrieval-Augmented Generation: A Survey. ACM Transactions on Information Systems 44, 2, Article 35 (2025), 52 pages. doi:10.1145/3777378

[11] Yuzhao Yang, Genlang Chen, Binhua He, and Yan Zhao. 2026. Grounded Knowledge Graph Extraction via LLMs: An Anchor-Constrained Framework with Provenance Tracking. Computers 15, 3 (2026), 178. doi:10.3390/computers15030178