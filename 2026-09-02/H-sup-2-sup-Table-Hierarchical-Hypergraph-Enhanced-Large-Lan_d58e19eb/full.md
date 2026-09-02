# H<sup>2</sup>Table: Hierarchical Hypergraph-Enhanced Large Language Models for Complex Table Reasoning

Jia Ling<sup>1</sup>, Yangfan Wang<sup>1</sup>, Chen Tang<sup>2</sup>, Haoming Tan<sup>1</sup>,

Yang Yang<sup>3</sup>, Yi Guan<sup>1</sup>, Jingchi Jiang<sup>1,4,\*</sup>

<sup>1</sup>Harbin Institute of Technology, <sup>2</sup>AI Research Center, Midea Group (Shanghai) Co., Ltd. <sup>3</sup>Changchun University of Science and Technology

<sup>4</sup>State Key Laboratory of Smart Farm Technologies and Systems {2022112476,yf.wang,2022111514}@stu.hit.edu.cn travistang@foxmail.com, yangyang\_hit\_wi@163.com {guanyi,jiangjingchi}@hit.edu.cn

## Abstract

Tables are ubiquitous across diverse domains, yet reasoning over them remains a significant challenge for modern large language models (LLMs). Current approaches typically linearize tables into sequences, inherently overlooking their intrinsic two-dimensional and hierarchical structure. To address this, we propose H<sup>2</sup>Table (Hierarchical Hypergraph-Enhanced Table Reasoning), a novel framework that represents complex tables as hierarchical nested hypergraphs. To process this representation, we design a tailored hypergraph encoder to facilitate message passing between hyperedges (headers) and nodes (cells), thereby perceiving the semantic entailment relationships between them within complex tables. Furthermore, we introduce a set of learnable query vectors acting as a lightweight bridge to extract representative structural embeddings from the encoder into the LLM. Experimental results demonstrate that our approach effectively handles complex table question answering tasks with hierarchical nested headers. Notably, on the HiTab dataset, H<sup>2</sup>Table achieves an average improvement of 22.88% over state-of-the-art baselines on highly complex tables with a nesting depth of four. Our code is available at: https: //github.com/lila120/h2table.

## 1 Introduction

Table understanding and reasoning are critical for enabling diverse downstream applications, such as table question answering (TableQA) (Pasupat and Liang, 2015) and Text-to-SQL (Yu et al., 2018), to process complex, real-world data. Consequently, leveraging the powerful semantic synthesis and reasoning capabilities of large language models (LLMs) has become essential in this domain. Driven by the need to decode structured data, existing approaches have evolved significantly. Recently, the paradigm has shifted toward LLM-based methods. These approaches primarily rely on 1D text-centric serialization to harness pretrained knowledge, or construct massive instruction datasets for full-parameter supervised finetuning (SFT) (e.g., TableLlama Zhang et al., 2024, TableGPT Li et al., 2024). Furthermore, to explicitly capture internal structural information, some works represent tables as graphs. For instance, HeGTa (Jin et al., 2025) models tables as heterogeneous graphs, while HYTREL (Chen et al., 2023) utilizes hypergraphs. Building upon this structural perspective, TAMO (Li et al., 2026) integrates the hypergraph encoder from HYTREL with LLMs, achieving the current state-of-the-art performance.

![](images/5fd2387c68f2396ba9f764314c7ffd3428a0f092f6ade3c39dbcb2300cb85753.jpg)  
Figure 1: Comparison of modeling approaches and corresponding message passing mechanisms for complex tables between the proposed H<sup>2</sup>Table and other leading baselines.

However, existing methods exhibit several critical limitations. LLM paradigms relying on 1D serialization and full-parameter fine-tuning demand massive amounts of high-quality data and incur prohibitive computational costs. Alternatively, representing tables as standard graphs fails to capture their intrinsic structural features. Since edges in standard graphs are strictly pairwise, they inherently contradict the one-to-many relationships between headers and cells. Existing hypergraphbased methods like HYTREL overcome this bottleneck but are restricted to simple flat tables. Furthermore, lacking LLM integration, they are incapable of handling generative tasks like TableQA. While TAMO addresses this by coupling a hypergraph encoder with an LLM, its encoding mechanism fundamentally overlooks the nested hierarchy of complex headers. As illustrated in Figures 1(a) and (c), when processing complex tables with hierarchical headers, TAMO flattens all hyperedges into a single level by directly connecting high-level headers to their associated cell nodes. This approach neglects the hierarchical relationships among hyperedges, leading to a loss of multi-level dependencies and table semantic implications. Consequently, the subsequent message passing occurs solely between these flattened hyperedges and cell nodes, ignoring the essential interactions across different header levels.

To address the aforementioned challenges, we propose H<sup>2</sup>Table (Hierarchical Hypergraph-Enhanced Table Reasoning). We first introduce a tailored transformation that maps complex tables into hierarchical nested hypergraphs. Rather than flattening these structures, our method preserves intrinsic tabular semantics, particularly the semantic entailment flowing from multi-level headers down to specific cells, by employing higherlevel hyperedges that connect parent headers exclusively to their direct sub-headers. To encode these semantic dependencies, we develop a hypergraph encoder equipped with a four-stage hierarchical interaction module (V2E, C2P, P2C, and E2V). Beyond conventional vertex-hyperedge interactions, we specifically introduce the C2P and P2C stages to facilitate information exchange across different hyperedge levels, as depicted in Figures 1(d). Subsequently, we introduce a compact set of learnable query vectors to extract representative tabular structures from the encoder, optimized continuously through end-to-end finetuning. These vectors act as a Soft Structure Prompt (Lester et al., 2021) prepended to the serialized text, naturally infusing the LLM with robust structural awareness. Remarkably, rather than relying on costly full-parameter tuning, H<sup>2</sup>Table applies lightweight LoRA (Hu et al., 2022) to the base LLM, achieving performance comparable or even superior to full fine-tuning while incurring minimal computational overhead.

To demonstrate that our structural modeling yields superior reasoning and generalization capabilities, we conduct extensive experiments on multiple complex table benchmarks. Our main contributions are summarized as follows:

• Hierarchical Nested Hypergraph Modeling for Complex Tables: We propose a hierarchical nested hypergraph to preserve the intrinsic semantic entailment in multi-level tables. Furthermore, we develop a tailored, hierarchy-aware encoder that facilitates dynamic message passing to ensure comprehensive cross-level interactions.

• Learnable Query Based Structural Alignment: We introduce a query-based Soft Structure Prompt mechanism powered by a compact set of learnable query vectors. Through end-to-end fine-tuning, this mechanism progressively learns to extract taskbeneficial structural features from tables, thereby boosting reasoning capabilities for question answering.

• Comprehensive Superiority and Structural Robustness: Empirical results show that H<sup>2</sup>Table consistently outperforms competing baselines across multiple datasets. Crucially, it demonstrates exceptional robustness against increasing structural complexity, achieving an average relative improvement of 22.88% on deep hierarchical tables and enabling fine-tuned medium-sized models to eclipse hundred-billion parameter models.

## 2 Related Work

## 2.1 LLM-based Table Reasoning

Currently, flattening two-dimensional tables into one-dimensional sequences is the prevailing approach for LLMs to process tabular data (Lu et al., 2025). To enhance the comprehension capabilities of LLMs on serialized tables, previous studies have typically employed instruction tuning (e.g., TableGPT, Li et al., 2024; TableLlama, Zhang et al., 2024) or prompt engineering (e.g., Chain-of-Table, Wang et al., 2024) for optimization. However, these optimization strategies exhibit distinct limitations: instruction tuning demands substantial amounts of data and incurs prohibitive computational costs while prompt engineering is heavily constrained by the context window limits. More importantly, forcibly flattening a table inherently disrupts its intrinsic topological structure, a critical flaw that becomes especially evident when dealing with complex or nested tables.

To mitigate the structural information loss caused by serialization, several studies (e.g., TabPrompt, Jin et al., 2023; HeGTa, Jin et al., 2025) have modeled tables as graphs to capture their intrinsic two-dimensional and higher-order features. However, since each edge in a conventional graph typically denotes only pairwise relationships, it struggles to accurately characterize the inherent one-to-many higher-order correlations between a row (or column) header and its constituent cells. Consequently, HYTREL (Chen et al., 2023) represents tables as hypergraphs, where a single hyperedge can connect multiple nodes. Although effective, it is restricted to simple flat tables and lacks integration with LLMs, making it incapable of handling generative downstream tasks such as TableQA. Building upon this, TAMO (Li et al., 2026) integrates the HYTREL encoder with LLMs, treating tabular data as an independent continuous modality. While effective for flat tables, TAMO’s encoding mechanism fundamentally flattens complex nested headers by directly grouping all associated cells under a single, coarse-grained hyperedge. This design inherently overlooks the multilevel dependencies and internal topological structures among the headers themselves. To bridge the gap, H<sup>2</sup>Table advances hypergraph-based table reasoning by explicitly modeling these hierarchical relationships. By designing nested hyperedges and a tailored multi-stage message-passing mechanism, our framework systematically captures fine grained structural semantics while maintaining exceptional parameter efficiency.

## 2.2 Cross-Modal Alignment between Graph Structures and Semantic Spaces

The aforementioned graph-based tabular reasoning methods inherently rely on fusing the structural topology of graphs with the semantic comprehension capabilities of LLMs. Fundamentally, this constitutes a challenge of cross-modal feature alignment. Existing strategies generally fall into two paradigms. The first involves projectionbased mapping, taking cues from large multimodal models (e.g., LLaVA, Liu et al., 2023a). For instance, GraphGPT (Tang et al., 2024) employs an MLP to project node and subgraph embeddings, extracted by a pre-trained GNN, directly into the continuous input space of the LLM. Similarly, both HeGTa (utilizing heterogeneous graphs) and TAMO (leveraging hypergraphs) adopt analogous linear projectors for feature transformation. Alternatively, the second paradigm focuses on serialization-based alignment. InstructGLM (Ye et al., 2024) heuristically flattens the graph’s topological structure into an extended text sequence, facilitating subsequent fine-tuning strictly within the textual semantic domain.

However, simple linear projectors often struggle to efficiently compress highly complex structural information, while serialization inevitably leads to excessively long contexts and topological loss. In contrast to these conventional methods, we draw inspiration from the architecture of Q-Former (Li et al., 2023). Specifically, we introduce learnable query vectors to perform cross-attention with graph embeddings, thereby actively extracting and compressing global topological and semantic features via end-to-end generative fine-tuning. This design not only alleviates the computational overhead but also significantly bolsters the model’s feature aggregation capacity when reasoning over complex nested tables.

## 3 Method

## 3.1 Modeling Complex Tables as Hierarchical Nested Hypergraphs

To effectively capture the structural and semantic dependencies within complex tables, we model the table as a Hierarchical Nested Hypergraph H = $( V , E , R , I )$

Specifically, the node set $V = \{ v _ { 1 } , v _ { 2 } , \ldots , v _ { n } \}$ represents all data cells. The hyperedge set $E =$ $\{ e _ { 1 } , e _ { 2 } , \ldots , e _ { m } \}$ comprises all table headers along with a virtual global root representing the entire table. This set is strictly partitioned into disjoint leaf headers $E _ { l e a f }$ and high-level headers $E _ { h i g h }$ (which contains the root).

The hierarchical relation $R \subseteq E \times E$ defines the structural dependency among hyperedges across different header levels, where a directed edge $( e _ { i } , e _ { j } )$ indicates that $e _ { i }$ is the parent of $e _ { j }$ For instance, as shown in Figure 2, in the complex table, the high-level header "First Quarter Sales" acts as the parent of the leaf header "March". The incidence relation $I \subseteq V \times E _ { \mathrm { l e a f } }$ links each data cell v to its corresponding lowest-level leaf headers. For instance, as illustrated in Figure 2, the cell node "20" is connected to both the column header hyperedge "February" and the row header hyperedge "watch".

![](images/54a3333711b18018f0a64bbc39c5926bc6b855a6eeab2274e751b1d4e2e6fe7b.jpg)  
Figure 2: The overall architecture of the proposed framework. First, a complex hierarchical table is modeled as our designed hierarchical nested hypergraph, which is then processed by a hierarchical encoder featuring a four-stage message passing mechanism to derive the corresponding structural embeddings. Subsequently, a set of self-attended query vectors interacts with these structural embeddings via cross-attention. This yields a fixed number of structure-aware embeddings, which are finally combined with the serialized text and fed into the LLM to facilitate reasoning and generate the correct answer.

Based on these relations, the hypergraph establishes a recursively nested scope. For any leaf header $e _ { \mathrm { c h i l d } } \in E _ { \mathrm { l e a f } } ,$ , its base scope is determined by the incidence relation as $S ( e _ { \mathrm { c h i l d } } ) = \{ v \in V \mid$ $( v , e _ { \mathrm { c h i l d } } ) \in I \}$ . Conversely, the scope of any highlevel header $e _ { \mathrm { p a r e n t } } \in E _ { \mathrm { h i g h } }$ is defined as the union

of its children’s scopes:

$$
S ( e _ { \mathrm { p a r e n t } } ) = \bigcup _ { ( e _ { \mathrm { p a r e n t } } , e _ { \mathrm { c h i l d } } ) \in R } S ( e _ { \mathrm { c h i l d } } )
$$

Consequently, this formulation ensures that all parent hyperedges inherently perceive the scopes of their descending sub-headers all the way down to the individual cell nodes.

## 3.2 Hierarchy-Aware Hypergraph Encoder

Tailored to the proposed hierarchical nested hypergraph, we design a hierarchy-aware hypergraph encoder to facilitate comprehensive information propagation, encompassing both the interactions between data cells and hyperedges, and the hierarchical message passing among hyperedges across different levels. Within this encoder, we innovatively introduce a Hierarchical

Hyperedge Interaction Module, which integrates a GAT into the hierarchical message-passing mechanism of hypergraphs. This design facilitates directed, weighted message propagation dynamically among hyperedges, thereby enabling the model to autonomously learn the underlying semantic correlations between table headers. As shown in Figure 2, once a complex table is modeled as the hierarchical hypergraph detailed in Section 3.1 and initial embeddings are generated, these embeddings undergo iterative information interaction through this encoder. The encoder’s single-layer forward propagation is meticulously structured into four distinct stages:

• V2E (Vertex-to-Edge): Guided by the incidence relation I, this stage utilizes a set attention mechanism (Lee et al., 2019; Chen et al., 2023) to aggregate data cell features into their corresponding leaf headers, thereby establishing the base representations for the lowest-level hyperedges.

• C2P (Child-to-Parent): Along the directed edges of the hierarchical relation R, a GAT dynamically weights and propagates information from child headers to their parents in a bottom-up manner. This enables highlevel headers and ultimately the global root to capture the underlying data distribution and structural hierarchy.

• P2C (Parent-to-Child): Operating symmetrically top-down along the reverse direction of R, this phase distributes global context. The GAT mechanism adaptively balances the broader semantic information passed from the parent with the preservation of the child’s own local features.

• E2V (Edge-to-Vertex): Finally, the hierarchically enriched hyperedge features are returned to the data cells via set attention. Residual connections are applied here to mitigate gradient vanishing, completing one full hierarchy-aware message-passing cycle.

## 3.3 Query-Based Parameter-Efficient Feature Alignment

While the hypergraph encoder captures rich topological representations, bridging the gap between continuous graph structural representations and the semantic reasoning space of LLMs remains a non-trivial challenge; naively mapping extensive graph embeddings via linear projection underutilizes structural priors and overloads the LLM’s context. To address this, we introduce a parameterefficient cross-modal alignment module to seamlessly bridge these distinct spaces.

Inspired by Q-Former (Li et al., 2023), we design a query-based mechanism for structural feature extraction and alignment. Specifically, we initialize a compact set of learnable query vectors designed to distill salient global topological representations from the dense node and hyperedge features yielded by the encoder. As illustrated in Figure 2, the alignment module is instantiated via a stack of Transformer decoder layers (Vaswani et al., 2017).

In each layer, the query vectors first undergo self-attention to model their intra-query dependencies. During the subsequent cross-attention phase, these queries attend to the comprehensive structural embeddings acting as keys and values generated by the hypergraph encoder, thereby facilitating deep cross-modal fusion (Li et al., 2023). Finally, the fused representations are projected through a feed-forward network (FFN).

Formally, let H denote the structural graph features and Q<sup>(l</sup>−<sup>1)</sup> denote the queries at layer l − 1; the core update process at the l-th layer can be abstracted as:

$$
Q ^ { ( l ) } = \mathrm { F F N } \Big ( \mathbf { M C A } \big ( \mathbf { M S A } ( Q ^ { ( l - 1 ) } ) , H , H \big ) \Big )
$$

where MSA and MCA represent multi-head selfattention and cross-attention, respectively. For brevity, the standard residual connections and layer normalization applied within each submodule are omitted from the notation.

Optimized continuously through end-to-end fine-tuning, the queries learn to extract the most representative structural features of the table. After multi-layer alignment, these highly distilled query embeddings function as soft structural prompts that are prepended to the serialized table text, constructing a cohesive dual-stream semanticstructural input that seamlessly guides the downstream LLM’s reasoning (Liu et al., 2023b).

Irrespective of table complexity, the module condenses structural information into a fixed number of tokens, alleviating the sequence explosion common in direct graph inputs and minimizing the LLM’s contextual burden. Moreover, as this module manages the cross-modal translation, it circumvents costly full-parameter finetuning. By updating merely 1% of the parameters, this paradigm achieves performance comparable to full fine-tuning. Consequently, it drastically reduces computational costs while mitigating catastrophic forgetting and preserving the LLMs inherent reasoning capabilities.

<table><tr><td rowspan="2">Dataset</td><td colspan="5">HiTab</td><td colspan="4">TATQA</td></tr><tr><td>I</td><td>ⅡI</td><td>Ⅲ</td><td>IV</td><td>Avg.</td><td>I</td><td>Ⅱ</td><td>ⅢII</td><td>Avg.</td></tr><tr><td>Depth Llama3.1-8B</td><td>0.3143</td><td>0.3609</td><td>0.2977</td><td>0.2286</td><td>0.3004</td><td>0.1866</td><td>0.1693</td><td>0.1603</td><td>0.1721</td></tr><tr><td>+pure text (LoRA)</td><td>0.6571</td><td>0.7190</td><td>0.6458</td><td>0.5714</td><td>0.6483</td><td>0.3528</td><td>0.3964</td><td>0.3462</td><td>0.3651</td></tr><tr><td>+TAMO (LoRA)</td><td>0.7143</td><td>0.7960</td><td>0.7355</td><td>0.6286</td><td>0.7186</td><td>0.4461</td><td>0.4819</td><td>0.4295</td><td>0.4525</td></tr><tr><td>+H2Table (LoRA)</td><td>0.6857</td><td>0.8046</td><td>0.7614</td><td>0.7143</td><td>0.7415</td><td>0.4606</td><td>0.4819</td><td>0.4359</td><td>0.4595</td></tr><tr><td>Gemma2-9B</td><td>0.6000</td><td>0.6733</td><td>0.5818</td><td>0.5143</td><td>0.5924</td><td>0.3965</td><td>0.4136</td><td>0.3974</td><td>0.4025</td></tr><tr><td>+pure text (LoRA)</td><td>0.7143</td><td>0.7946</td><td>0.7343</td><td>0.7143</td><td>0.7394</td><td>0.4781</td><td>0.5328</td><td>0.5128</td><td>0.5079</td></tr><tr><td>+TAMO (LoRA)</td><td>0.7143</td><td>0.8031</td><td>0.7540</td><td>0.5714</td><td>0.7107</td><td>0.4723</td><td>0.5311</td><td>0.4936</td><td>0.4990</td></tr><tr><td> $+ \mathbf { H } ^ { 2 } \mathbf { T a b l e }$  (LoRA)</td><td>0.7143</td><td>0.8260</td><td>0.7552</td><td>0.7429</td><td>0.7596</td><td>0.4927</td><td>0.5354</td><td>0.5000</td><td>0.5094</td></tr><tr><td>Llama2-7B</td><td>0.2000</td><td>0.1997</td><td>0.1242</td><td>0.1429</td><td>0.1667</td><td>0.1283</td><td>0.1364</td><td>0.1410</td><td>0.1352</td></tr><tr><td>+pure text (LoRA)</td><td>0.5429</td><td>0.5649</td><td>0.4379</td><td>0.4000</td><td>0.4864</td><td>0.1953</td><td>0.2409</td><td>0.1859</td><td>0.2074</td></tr><tr><td> $+ \mathrm { T A M O }$  (LoRA)</td><td>0.6000</td><td>0.6862</td><td>0.5461</td><td>0.4571</td><td>0.5724</td><td>0.3324</td><td>0.3575</td><td>0.3718</td><td>0.3539</td></tr><tr><td> $+ \mathbf { H } ^ { 2 } \mathbf { T a b l e }$  (LoRA)</td><td>0.6571</td><td>0.6862</td><td>0.5720</td><td>0.5714</td><td>0.6217</td><td>0.3615</td><td>0.3549</td><td>0.3654</td><td>0.3606</td></tr><tr><td>TableLlama</td><td>0.5714</td><td>0.6933</td><td>0.5966</td><td>0.5429</td><td>0.6011</td><td>0.1574</td><td>0.1451</td><td>0.1282</td><td>0.1436</td></tr><tr><td>GPT-40</td><td>0.7143</td><td>0.7418</td><td>0.6913</td><td>0.6571</td><td>0.7011</td><td>0.5335</td><td>0.5337</td><td>0.5064</td><td>0.5245</td></tr><tr><td>DeepseekV3</td><td>0.6571</td><td>0.7247</td><td>0.6531</td><td>0.6286</td><td>0.6659</td><td>0.4606</td><td>0.4491</td><td>0.4551</td><td>0.4549</td></tr></table>

Table 1: Performance comparison of $\mathrm { H } ^ { 2 \prime }$ Table against baseline models on the TableQA task across different table nesting depths (I–IV for HiTab, and I–III for TATQA). Models without indentation (including backbone baselines and the bottom group) are evaluated under the zero-shot setting, while indented rows represent LoRA fine-tuned variants. $\tilde { \bf \Delta } ^ { 6 6 } \mathrm { \bf \vec { s } } ^ { , 3 }$ denotes the arithmetic mean across depths. Bold and underlined values indicate the best and second-best results within each model group, respectively.

## 4 Experiments

## 4.1 Experimental setup

Datasets. We evaluate H<sup>2</sup>Table on several representative benchmarks featuring complex hierarchical structures, namely HiTab (Cheng et al., 2022) and TATQA (Zhu et al., 2021). The primary downstream task across all selected datasets is TableQA, with Accuracy serving as the core evaluation metric. The distribution of tables within these datasets, categorized by header nesting depth, is illustrated in Figure 3. Notably, because the formatted TATQA dataset includes a mere 12 tables with four-level headers, a quantity insufficient for statistical significance, these instances were excluded from our main experimental evaluation.

Baselines. We select TableLlama (Zhang et al., 2024) as a primary baseline, as it achieved stateof-the-art performance across numerous table reasoning tasks through extensive SFT on diverse tabular datasets. Additionally, we compare our approach with TAMO (Li et al., 2026), a contemporary method that treats tables as an independent modality and has demonstrated superior performance over TableLlama on several benchmarks. To ensure a fair and direct comparison with these representative models, we adopt Llama2-7B (Touvron et al., 2023) as our primary base model. Furthermore, to evaluate the efficacy of our method on more advanced architectures, we extend our experiments to Llama3.1-8B (Grattafiori et al., 2024) and Gemma2-9B (Team et al., 2024). Our evaluation also includes comparisons against several training configurations for the base models, including zero-shot inference, and LoRA fine tuning using pure-text representations. Detailed experimental results are reported in Table 1.

![](images/24abca04a3ae9a52b5e805be120d0de7a1c1616a9d3d7cf1bfa0300e3b71ca42.jpg)  
Figure 3: Distribution of table nesting depth for HiTab and TATQA

Implementation Details For the initialization of our hypergraph, node embeddings are extracted from the final layer of a pre-trained RoBERTalarge model (Liu et al., 2019), and the hidden size of our hypergraph encoder is strictly set to 1024 to maintain dimensional consistency. For a fair comparison, both the baseline models and our proposed framework are fine-tuned end-to-end using LoRA with a rank of 8, with binary cross-entropy as the loss function. More detailed hyperparameter settings and training configurations are provided in Appendix A.

## 4.2 Main Results

Table 1 presents the performance comparison on the HiTab and TATQA datasets. To evaluate the capability of handling complex tables, the results are further categorized by table nesting depth. Overall, our proposed $\mathrm { H } ^ { 2 \cdot }$ Table consistently outperforms all baseline representations across different base models and depth levels, demonstrating its effectiveness in TableQA tasks especially when encountering hierarchical complex tables .

When integrated with various open-source LLMs such as Llama3.1-8B, Gemma2-9B, and Llama2-7B, the $\mathrm { H } ^ { 2 \cdot }$ Table representation consistently yields the highest average accuracy within each base model group. For instance, fine-tuning Gemma2-9B with H<sup>2</sup>Table achieves an average accuracy of 0.7596 on HiTab, substantially surpassing both the pure text and TAMO representations. This indicates that $\mathrm { H } ^ { 2 \cdot }$ Table serves as a general and robust paradigm to improve the table reasoning capabilities of LLMs.

Aligning with our primary motivation to tackle complex tables, $\mathrm { H } ^ { 2 \cdot }$ Table exhibits remarkable superiority on tables with high nesting depths (i.e., Depth-3 and Depth-4). While the performance of traditional baseline methods drops precipitously as structural complexity increases, H<sup>2</sup>Table remains highly robust. On the HiTab dataset at Depth-4, Llama3.1-8B with H<sup>2</sup>Table achieves an accuracy of 0.7143, yielding a significant absolute improvement over the pure text (0.5714) and TAMO (0.6286) baselines. Similar trends are consistently observed across other base models, validating that our method effectively preserves structural integrity and captures deep hierarchical semantics.

Remarkably, fine-tuning medium-sized models (7B–9B parameters) with H<sup>2</sup>Table enables them to rival or even surpass state-of-the-art closedsource and massively large LLMs in complex table reasoning. On the HiTab dataset, by only updating approximately 1% of the total parameters, Gemma2-9B equipped with $\mathrm { H } ^ { 2 \cdot }$ Table (0.7596) significantly outperforms the hundred-billion parameter DeepseekV3 (0.6659), and the powerful GPT-4o (0.7011). These results compellingly demonstrate that providing a superior structural representation empowers smaller models to achieve highly competitive reasoning capabilities on complex tables.

## 4.3 Ablation Study

To thoroughly investigate the individual contributions and architectural synergies of key components in $\mathrm { H } ^ { 2 \cdot }$ Table, we conduct an extensive ablation study on the HiTab dataset based on Llama3.1- 8B.

![](images/392bf2172cda3298cbeed1217294bf26085e11eb20dd0c7f787abd8a672d031b.jpg)  
Figure 4: The heatmap of the ablation study. Deep blue hues indicate superior performance, while lighter shades represent degraded capabilities.

As visualized in Figure 4, removing the textual feature module (w/o text) causes a catastrophic performance collapse (represented by the blank row), underscoring that textual semantics remain the foundation of table understanding. Crucially, omitting the complete hierarchical hypergraph encoder (w/o encoder) also induces severe performance degradation, which validates that explicit structural modeling is indispensable for breaking the table-comprehension bottleneck of plain LLMs.

Furthermore, removing the GAT-based message passing (denoted as w/o GAT, where the hierarchical edge aggregation is replaced by a standard Set Attention mechanism) or omitting the crossattention alignment (w/o query) leads to a slight yet consistent performance decline across most subsets. Although these performance gaps may appear modest, such stable improvements are highly meaningful given the extremely challenging generative setting of complex hierarchical TableQA.

<table><tr><td rowspan="2">Training Set Test Set</td><td colspan="3">HiTab</td><td colspan="3">TATQA</td></tr><tr><td>HiTab</td><td>TATQA</td><td>AITQA</td><td>TATQA</td><td>HiTab</td><td>AITQA</td></tr><tr><td>Llama3.1-8B</td><td>0.3245</td><td>0.1708</td><td>0.5146</td><td></td><td></td><td></td></tr><tr><td>+Prompt tuning</td><td>0.0322</td><td>0.0298</td><td>0.1023</td><td>0.0639</td><td>0.0463</td><td>0.0498</td></tr><tr><td>+pure text (LoRA)</td><td>0.6774</td><td>0.2548</td><td>0.8330</td><td>0.3815</td><td>0.5696</td><td>0.7540</td></tr><tr><td>+TAMO (LoRA)</td><td>0.7643</td><td>0.3156</td><td>0.8634</td><td>0.4721</td><td>0.6086</td><td>0.7974</td></tr><tr><td>+H²Table (LoRA)</td><td>0.7713</td><td>0.3225</td><td>0.8673</td><td>0.4745</td><td>0.6313</td><td>0.8220</td></tr></table>

Table 2: Performance comparison of various models on standard and cross-domain TableQA tasks. Bold and underlined values indicate the best and second-best performance among the fine-tuned variants for each base model, respectively. All results are averaged over three independent training and evaluation runs.  
structure-aware TAMO improves it to 0.8634.

Interestingly, the w/o GAT variant yields highly competitive results and even achieves the top score on the Depth-1 subset. As Depth-1 tables are essentially flat, they do not heavily rely on complex hierarchical message passing; thus, the strong performance of this variant is expected. However, the performance gap becomes significantly more pronounced on deeper hierarchical tables (e.g., Depth-3 and Depth-4), which is exactly where the GATbased hierarchy-aware propagation demonstrates its efficacy. Overall, the full H<sup>2</sup>Table architecture achieves the best average performance, validating the necessity and strong synergistic effects of our integrated modules.

## 4.4 Out-of-Distribution Generalization

To evaluate the out-of-distribution (OOD) generalization capability of our method, we conduct cross-domain experiments by incorporating AITQA (Katsis et al., 2022), another complex hierarchical table benchmark, as an unseen evaluation set.

As shown in Table 2, the vanilla Llama3.1-8B and its prompt-tuning variant exhibit poor robustness across domains, with prompt tuning suffering from catastrophic performance drops. Incorporating table-specific structural modeling significantly boosts generalization. For instance, when trained on HiTab, the pure text baseline (LoRA) achieves an OOD score of 0.8330 on AITQA, whereas the

Most importantly, our proposed H<sup>2</sup>Table consistently achieves the best performance across all settings, outperforming all baselines by a clear margin. Specifically, whether trained on HiTab or TATQA, H<sup>2</sup>Table uniformly surpasses TAMO on both in-domain and all cross-domain test sets. These results demonstrate that by effectively encoding hierarchical table structures, H<sup>2</sup>Table learns more robust, domain-invariant representations for complex TableQA tasks.

## 4.5 Analysis on Number of Query Vectors

As shown in Table 3, we evaluated the impact of varying the number of query vectors on overall performance.

<table><tr><td>N</td><td>Llama2-7B</td><td>Llama3.1-8B</td><td>Gemma2-9B</td></tr><tr><td>1</td><td>0.6172</td><td>0.7675</td><td>0.7862</td></tr><tr><td>4</td><td>0.5905</td><td>0.7677</td><td>0.7847</td></tr><tr><td>8</td><td>0.6061</td><td>0.7672</td><td>0.7841</td></tr><tr><td>12</td><td>0.6113</td><td>0.7713</td><td>0.7860</td></tr><tr><td>16</td><td>0.5995</td><td>0.7603</td><td>0.7914</td></tr></table>

Table 3: Comparison of different numbers of query vectors (denoted as N). Bold and underlined values indicate the best and second-best performance, respectively.

While the optimal vector count differs slightly across model architectures, using 12 query vectors consistently yields robust and highly competitive results. Specifically, it achieves the best performance on Llama3.1-8B and maintains nearoptimal scores for both Llama2-7B and Gemma2- 9B. To strike a balance between performance and cross-model stability, we empirically set the number of query vectors to 12 for all experiments.

## 5 Conclusion

In this paper, we propose a novel framework, H<sup>2</sup>Table. Unlike conventional methods that model tables as standard hypergraphs, we design a specialized hierarchical hypergraph representation tailored for complex tables, coupled with a hypergraph encoder featuring a four-stage hierarchical message-passing mechanism. Subsequently, a learnable query vector mechanism is employed to extract the most representative structural features from the encoder’s output. The resulting structural embeddings effectively assist LLMs in table reasoning tasks. The effectiveness of our framework has been validated across multiple datasets, particularly those featuring intricate hierarchical tables, demonstrating strong performance and generalization capabilities.

## Limitations

Our current work assumes that the header structures of complex hierarchical tables can be accurately obtained beforehand. Under this setting, we treat header hierarchies as available structural annotations to isolate and evaluate the contribution of hierarchical hypergraph modeling itself. Future work could explore integrating upstream table-structure extractors with H<sup>2</sup>Table for end-toend processing. Additionally, the proposed framework still relies on feeding the serialized tabular text into the LLM. Subsequent work will investigate how to enable the structural embeddings generated by the encoder to simultaneously capture rich semantic information.

## Acknowledgments

We gratefully acknowledge the support of the Key Research and Development Program of Heilongjiang Province, China [2024ZX01A07] and the National Key Research and Development Program [2025YFE0209200].

## References

Pei Chen, Soumajyoti Sarkar, Leonard Lausen, Balasubramaniam Srinivasan, Sheng Zha, Ruihong Huang, and George Karypis. 2023. Hytrel: Hypergraph-enhanced tabular data representation learning. Advances in Neural Information Processing Systems, 36:32173–32193.

Zhoujun Cheng, Haoyu Dong, Zhiruo Wang, Ran Jia, Jiaqi Guo, Yan Gao, Shi Han, Jian-Guang Lou, and

Dongmei Zhang. 2022. Hitab: A hierarchical table dataset for question answering and natural language generation. In Proceedings ofthe 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1094–1110.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Rihui Jin, Yu Li, Guilin Qi, Nan Hu, Yuan-Fang Li, Jiaoyan Chen, Jianan Wang, Yongrui Chen, Dehai Min, and Sheng Bi. 2025. Hegta: Leveraging heterogeneous graph-enhanced large language models for few-shot complex table understanding. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 24294–24302.

Rihui Jin, Jianan Wang, Wei Tan, Yongrui Chen, Guilin Qi, and Wang Hao. 2023. Tabprompt: Graph-based pre-training and prompting for few-shot table understanding. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 7373– 7383.

Yannis Katsis, Saneem Chemmengath, Vishwajeet Kumar, Samarth Bharadwaj, Mustafa Canim, Michael Glass, Alfio Gliozzo, Feifei Pan, Jaydeep Sen, Karthik Sankaranarayanan, and 1 others. 2022. Aitqa: Question answering dataset over complex tables in the airline industry. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies: Industry Track, pages 305– 314.

Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, and Yee Whye Teh. 2019. Set transformer: A framework for attention-based permutation-invariant neural networks. In International conference on machine learning, pages 3744– 3753. PMLR.

Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 conference on empirical methods in natural language processing, pages 3045–3059.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR.

Liyao Li, Chao Ye, Wentao Ye, Yifei Sun, Zhe Jiang, Haobo Wang, Jiaming Tian, Yiming Zhang, Ningtao Wang, Xing Fu, and 1 others. 2026. Table as

a modality for large language models. Advances in Neural Information Processing Systems, 38:29285– 29308.

Peng Li, Yeye He, Dror Yashar, Weiwei Cui, Song Ge, Haidong Zhang, Danielle Rifinski Fainman, Dongmei Zhang, and Surajit Chaudhuri. 2024. Table-gpt: Table fine-tuned gpt for diverse table tasks. Proceedings ofthe ACM on Management ofData, 2(3):1–28.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023a. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916.

Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. 2023b. Pretrain, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM computing surveys, 55(9):1–35.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

Weizheng Lu, Jing Zhang, Ju Fan, Zihao Fu, Yueguo Chen, and Xiaoyong Du. 2025. Large language model for table processing: A survey. Frontiers of Computer Science, 19(2):192350.

Panupong Pasupat and Percy Liang. 2015. Compositional semantic parsing on semi-structured tables. In Proceedings ofthe 53rd Annual Meeting ofthe Associationfor Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1470– 1480.

Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Lixin Su, Suqi Cheng, Dawei Yin, and Chao Huang. 2024. Graphgpt: Graph instruction tuning for large language models. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 491–500.

Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, and 1 others. 2024. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, and 1 others. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Zilong Ryan Wang, Hao Zhang, Chun-Liang Li, Julian M Eisenschlos, Vincent Perot, Zifeng Wang, Lesly Miculicich, Yasuhisa Fujii, Jingbo Shang, Chen-Yu Lee, and 1 others. 2024. Chain-of-table: Evolving tables in the reasoning chain for table understanding. In International Conference on Learning Representations, volume 2024, pages 55587– 55610.

Ruosong Ye, Caiqi Zhang, Runhui Wang, Shuyuan Xu, and Yongfeng Zhang. 2024. Language is all a graph needs. In Findings of the association for computational linguistics: EACL 2024, pages 1955–1973.

Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, and 1 others. 2018. Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-tosql task. In Proceedings of the 2018 conference on empirical methods in natural language processing, pages 3911–3921.

Tianshu Zhang, Xiang Yue, Yifei Li, and Huan Sun. 2024. Tablellama: Towards open large generalist models for tables. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6024–6044.

Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao Wang, Shuo Zhang, Jiancheng Lv, Fuli Feng, and Tat-Seng Chua. 2021. Tat-qa: A question answering benchmark on a hybrid of tabular and textual content in finance. In Proceedings of the 59th annual meeting of the Association for Computational Linguistics and the 11th international joint conference on natural language processing (volume 1: long papers), pages 3277–3287.

## A Details of the Experiments

## A.1 Training Settings.

For efficient fine-tuning, we applied LoRA $( r =$ $8 , \alpha \ = \ 1 6 ,$ dropout = 0.05). The model was optimized using AdamW $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5 ,$ weight decay = 0.05) with a learning rate of 1 × $1 0 ^ { - 5 }$ .To construct the training sequences, we concatenated the graph/query embeddings with the text prompt (comprising the description and the question) and the target label. Descriptions were truncated to 1,024 tokens, and the target generation length was bounded to 128 tokens. While the training sequences were dynamically batched, we relaxed the maximum context window to 4,096 tokens during the inference and evaluation phases.

## A.2 Datasets.

In addition to HiTab, we applied specific preprocessing steps to the other two datasets. For • (H²Table)Answer: natural resources and conservation

<table><tr><td rowspan=1 colspan=7>Field / Sex</td><td rowspan=1 colspan=1>2017</td><td rowspan=1 colspan=1>2018</td><td rowspan=1 colspan=1>2019</td><td rowspan=1 colspan=1>Percent Change2017-19</td></tr><tr><td rowspan=27 colspan=1>All surveyed fields</td><td rowspan=1 colspan=6>Male</td><td rowspan=1 colspan=1>16580</td><td rowspan=1 colspan=1>17468</td><td rowspan=1 colspan=1>17980</td><td rowspan=1 colspan=1>8.4</td></tr><tr><td rowspan=1 colspan=6>Female</td><td rowspan=1 colspan=1>11600</td><td rowspan=1 colspan=1>11816</td><td rowspan=1 colspan=1>12369</td><td rowspan=1 colspan=1>6.6</td></tr><tr><td rowspan=6 colspan=2></td><td rowspan=6 colspan=3>Science</td><td rowspan=1 colspan=1>Agricultural sciences</td><td rowspan=1 colspan=1>496</td><td rowspan=1 colspan=1>565</td><td rowspan=1 colspan=1>645</td><td rowspan=1 colspan=1>30</td></tr><tr><td rowspan=1 colspan=1>Biological and biomedical sciences</td><td rowspan=1 colspan=1>8203</td><td rowspan=1 colspan=1>8250</td><td rowspan=1 colspan=1>8229</td><td rowspan=1 colspan=1>0.3</td></tr><tr><td rowspan=1 colspan=1>Computer and information sciences</td><td rowspan=1 colspan=1>476</td><td rowspan=1 colspan=1>515</td><td rowspan=1 colspan=1>510</td><td rowspan=1 colspan=1>7.1</td></tr><tr><td rowspan=1 colspan=1>Geosciences, atmospheric sciences, and ocean sciences</td><td rowspan=1 colspan=1>1794</td><td rowspan=1 colspan=1>2106</td><td rowspan=1 colspan=1>2177</td><td rowspan=1 colspan=1>21.3</td></tr><tr><td rowspan=4 colspan=1></td><td rowspan=1 colspan=1>Mathematics and statistics</td><td rowspan=1 colspan=1>240</td><td rowspan=1 colspan=1>266</td><td rowspan=1 colspan=1>305</td><td rowspan=1 colspan=1>27.1</td></tr><tr><td rowspan=1 colspan=1>Multidisciplinary and interdisciplinary studies</td><td rowspan=1 colspan=1>806</td><td rowspan=1 colspan=1>832</td><td rowspan=1 colspan=1>820</td><td rowspan=1 colspan=1>1.7</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=2></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Natural resources and conservation</td><td rowspan=1 colspan=1>364</td><td rowspan=1 colspan=1>580</td><td rowspan=1 colspan=1>582</td><td rowspan=1 colspan=1>59.9</td></tr><tr><td rowspan=15 colspan=2>Science and engineering</td><td rowspan=5 colspan=3></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>Physical sciences</td><td rowspan=1 colspan=1>2871</td><td rowspan=1 colspan=1>3056</td><td rowspan=1 colspan=1>3316</td><td rowspan=1 colspan=1>15.5</td></tr><tr><td rowspan=1 colspan=1>Psychology</td><td rowspan=1 colspan=1>494</td><td rowspan=1 colspan=1>507</td><td rowspan=1 colspan=1>576</td><td rowspan=1 colspan=1>16.6</td></tr><tr><td rowspan=1 colspan=1>Social sciences</td><td rowspan=1 colspan=1>1524</td><td rowspan=1 colspan=1>1601</td><td rowspan=1 colspan=1>1659</td><td rowspan=1 colspan=1>8.9</td></tr><tr><td rowspan=4 colspan=3>Engineering</td><td rowspan=1 colspan=1>Bioengineering and biomedical engineering</td><td rowspan=1 colspan=1>415</td><td rowspan=1 colspan=1>440</td><td rowspan=1 colspan=1>492</td><td rowspan=1 colspan=1>18.6</td></tr><tr><td rowspan=1 colspan=1>Chemical engineering</td><td rowspan=1 colspan=1>281</td><td rowspan=1 colspan=1>257</td><td rowspan=1 colspan=1>328</td><td rowspan=1 colspan=1>16.7</td></tr><tr><td rowspan=1 colspan=1>Civil engineering</td><td rowspan=1 colspan=1>422</td><td rowspan=1 colspan=1>414</td><td rowspan=1 colspan=1>492</td><td rowspan=1 colspan=1>16.6</td></tr><tr><td rowspan=1 colspan=1>engineering</td><td rowspan=1 colspan=1>557</td><td rowspan=1 colspan=1>588</td><td rowspan=1 colspan=1>637</td><td rowspan=1 colspan=1>14.4</td></tr><tr><td></td><td rowspan=2 colspan=2></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td rowspan=1 colspan=2>ering</td><td rowspan=1 colspan=1>Engineering science, mechanics, and physics</td><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1>220</td><td rowspan=1 colspan=1>186</td><td rowspan=1 colspan=1>-7</td></tr><tr><td rowspan=4 colspan=3></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Industrial and manufacturing engineering</td><td rowspan=1 colspan=1>119</td><td rowspan=1 colspan=1>105</td><td rowspan=1 colspan=1>137</td><td rowspan=1 colspan=1>15.1</td></tr><tr><td rowspan=1 colspan=1>Mechanical engineering</td><td rowspan=1 colspan=1>458</td><td rowspan=1 colspan=1>489</td><td rowspan=1 colspan=1>531</td><td rowspan=1 colspan=1>15.9</td></tr><tr><td rowspan=1 colspan=1>Metallurgical and materials engineering</td><td rowspan=1 colspan=1>181</td><td rowspan=1 colspan=1>215</td><td rowspan=1 colspan=1>242</td><td rowspan=1 colspan=1>33.7</td></tr><tr><td rowspan=1 colspan=1>Other engineeringa</td><td rowspan=1 colspan=1>641</td><td rowspan=1 colspan=1>842</td><td rowspan=1 colspan=1>864</td><td rowspan=1 colspan=1>34.8</td></tr><tr><td rowspan=2 colspan=2>Health</td><td rowspan=1 colspan=4>Clinical medicine</td><td rowspan=1 colspan=1>6448</td><td rowspan=1 colspan=1>6159</td><td rowspan=1 colspan=1>6273</td><td rowspan=1 colspan=1>-2.7</td></tr><tr><td rowspan=1 colspan=4>Other health</td><td rowspan=1 colspan=1>1190</td><td rowspan=1 colspan=1>1277</td><td rowspan=1 colspan=1>1348</td><td rowspan=1 colspan=1>13.3</td></tr></table>

Question: rates of growth in the number of nfrs between 2017 and 2019 in various s&e fields of research varied widely, which broad field had the largest percentage of increasing?

• (TAMO)Answer: geosciences, atmospheric sciences, and ocean sciences

![](images/e9ea456c15a21ba7d6851f652e4c4f1d3bf456bf7642fd5060372f1a2a98bbb1.jpg)  
Figure 5: An example of complex hierarchical tables which has a depth of 4 and corresponding question.

AITQA, since the original dataset provides annotations for all parent headers of each leaf header, it was straightforward to convert it into the HiTab format, specifically, comprising a row header tree, a column header tree, and a data matrix. For TATQA, we employed LLM to transform the original flat matrix tables into the aforementioned hierarchical format. Consequently, this approach inevitably introduces some conversion errors, which partially explains why the model trained on the TATQA dataset exhibits less stable performance compared to the model trained on HiTab. A detailed analysis of this noise is provided below.

While converting flat TATQA tables into hierarchical structures via LLMs inherently introduces a degree of structural noise, all evaluated models (both baselines and H<sup>2</sup>Table) are trained and tested on this identical augmented dataset. Consequently, the relative performance comparisons remain strictly fair.

To quantitatively assess the impact of this conversion noise, we manually reviewed a random sample of 100 tables from the TATQA test set (comprising 600 QA pairs, averaging 6 questions per table). The manual inspection revealed that 91 tables were converted accurately, whereas 9 exhibited noticeable structural errors. This low error rate (9%) suggests that TATQA tables possess predominantly simple, shallow hierarchies, making the LLM-based structural extraction largely reliable for this benchmark.

To further investigate how this structural noise affects downstream reasoning, we evaluated H<sup>2</sup>Table (using the Gemma-2-9B backbone) on this sampled subset. The results reveal a clear performance disparity between correctly and incorrectly converted tables:

<table><tr><td>Subset</td><td>Accuracy</td></tr><tr><td>Correct (546 QA pairs)</td><td>52.20%</td></tr><tr><td>Wrong (54 QA pairs)</td><td>37.04%</td></tr><tr><td>Overall (Full TATQA Test Set)</td><td>52.07%</td></tr></table>

Table 4: Performance disparity between correctly and incorrectly converted tables. The overall evaluation encompasses the full TATQA test set of 1,669 QA pairs.

As detailed above, the model achieves a 52.20% accuracy on the correctly converted subset, which is 15.16% higher than its performance on the subset with structural errors (37.04%). This discrepancy confirms that structural conversion quality directly influences downstream TableQA performance. However, because the overall conversion error rate remains low, the accuracy on the full TATQA test set (52.07%) aligns closely with the performance on the accurately converted subset, thereby demonstrating the robustness of our evaluation on this benchmark.

## A.3 Evaluation.

Regarding the evaluation metrics, rather than employing a strict Exact Match (EM), we applied necessary data cleaning and format normalization to the model’s predictions, such as stripping currency symbols (e.g., \$). We clarify that all compared methods (including closed-source models, fine-tuned baselines, and $\mathrm { H } ^ { 2 } \mathrm { T a b l e } )$ are evaluated under exactly the same answer-cleaning and format-normalization criteria. The reported accuracy is therefore computed under a unified relaxedmatch protocol after shared post-processing, ensuring fair comparability.

## B Case Study

In this section, we present a case study to demonstrate our model’s capability in handling complex table structures. As illustrated in Figure 5, our model successfully yields the correct answer to the given question, whereas a strong baseline, TAMO, fails to do so. This superior performance is attributed to our specifically designed modeling and hierarchical message-passing mechanism, which enable us to process such intricate tables more effectively.

## C Discussion

## C.1 Compatibility and Generalization over Flat Hypergraphs

A natural theoretical question is how $\mathrm { H } ^ { 2 } \mathrm { T a b l e }$ behaves when applied to standard flat tables lacking nested structures. In such degenerate cases, where all headers reside at a single depth, the hierarchical relations naturally collapse, and message passing reduces strictly to the leaf-level vertex-hyperedge interactions (i.e., V2E and E2V).

We regard this property as an intrinsic strength of our design: $\mathrm { H } ^ { 2 } \mathrm { T a b l e }$ remains mathematically well-defined and seamlessly applicable to flat tables, without imposing any artificial or redundant hierarchical priors. Crucially, even under this flat degeneration, H<sup>2</sup>Table is fundamentally distinct from prior hypergraph-based approaches such as TAMO in how structural knowledge is transferred to the LLM. Rather than relying on simple average-pooling over encoder embeddings, our query-based cross-attention module employs learnable queries to selectively attend over encoder representations, effectively distilling highdensity structural signals into a compact soft prompt. Furthermore, once multi-level nested headers are present, the hierarchical messagepassing stages (C2P and P2C) are dynamically activated, empowering the model to capture crosslevel semantic entailments that flat hypergraph formulations fundamentally overlook. Consequently, $\mathrm { H } ^ { 2 } \mathrm { T a b l e }$ generalizes rather than merely replaces flat hypergraph modeling—it preserves full backward compatibility with flat tables, provides a more selective encoder-LLM alignment mechanism, and explicitly incorporates hierarchical inductive bias precisely when nested headers matter.

## C.2 Statistical Significance Analysis on Deep Hierarchy Tables

To ensure a robust statistical evaluation and address the limited sample size of the isolated Depth-4 subset $( N \ = \ 3 5 )$ , we aggregate the Depth-3 $( N ~ = ~ 8 1 3 )$ and Depth-4 tables into a unified "Deep Hierarchy" subset, yielding a more substantial total sample size of $N \ = \ 8 4 8$ Evaluated using the Llama-3.1-8B backbone, our proposed $\mathrm { H } ^ { 2 r }$ Table architecture demonstrates a stable improvement over the strongest baseline, TAMO, on this combined challenging subset.

To formally validate the reliability of these performance gains, we conduct a paired McNemar’s test comparing the exact match predictions of H<sup>2</sup>Table against TAMO. The test confirms that the improvements introduced by our model are statistically significant, yielding an exact McNemar $p \textmd { - }$ value of 0.0079 and a continuity-corrected $\chi ^ { 2 } \ p -$ value of 0.0083 $( p < 0 . 0 1 )$ . These statistical results substantiate that the advantages of $\mathrm { H } ^ { 2 \cdot }$ Table on highly complex hierarchical tables are robust and not an artifact of small sample variance.