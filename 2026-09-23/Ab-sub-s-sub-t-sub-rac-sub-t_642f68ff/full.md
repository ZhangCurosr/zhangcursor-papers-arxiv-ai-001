# Arti<sub>c</sub>l<sub>e</sub>Min<sub>e</sub>r<sub>:</sub> Ont<sub>o</sub>l<sub>ogy</sub>-G<sub>u</sub>id<sub>e</sub>d Kn<sub>o</sub>wl<sub>e</sub>d<sub>ge</sub> Gr<sub>ap</sub>h C<sub>ons</sub>t<sub>ruc</sub>ti<sub>on</sub> f<sub>rom</sub> S<sub>c</sub>i<sub>en</sub>tifi<sub>c</sub> P<sub>u</sub>bli<sub>ca</sub>ti<sub>ons</sub>

Md Abrar Jahin<sup>1,2,\*</sup>, Craig A. Knoblock<sup>1,2</sup> and Jay Pujara<sup>1,2</sup>

<sup>1</sup>USC Information Sciences Institute, Marina del Rey, CA 90292, USA

<sup>2</sup>Thomas Lord Department of Computer Science, University of Southern California, Los Angeles, CA 90089, USA

## Ab<sub>s</sub>t<sub>rac</sub>t

Scientific papers keep much oftheir quantitative content in tables and supplementary files, where a number means something only through its header, caption, unit, analytical method, and the conventions ofits field. Recovering the rows and columns ofa table is therefore not the same as recovering the scientific fact it reports. Most semantic tableinterpretation methods assume that a clean table is already available and subsequently map its cells or columns to ontology terms, whereas most publication-level extraction systems are designed for a single domain. We study a middle path: a shared process that reads a paper and its supplementary files, gathers evidence from several parsers and a language model, and reconciles that evidence, while a bounded human-authored task module for each task supplies the domain meaning. The module lists the canonical names the graph may use, the surface forms that map to them, a small set ofderivation rules and validity constraints, an identity key, and the bindings used to write RDF. It defines what a task is allowed to emit; it does not try to list every convention of a field. We build four such modules (for drug-discovery chemistry, materials science, machine learning, and mineral geochemistry) in the ArticleMiner framework, and evaluate them on 163 papers, including a new geochemistry benchmark with expert-curated ground truth. In comparisons against a same-LLM few-shot baseline, the point estimates favor ArticleMiner on all four tasks, with uncertainty on the two smaller benchmarks. The geochemistry comparison also includes access to supplementary files, so its improvement cannot be attributed to domain guidance alone.

## Ke<sub>y</sub>words

knowledge graphs, document understanding, table understanding, scientific publications, ontology, large language models

## 1<sub>.</sub> I<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>ti<sub>on</sub>

Scientific publications contain quantitative observations in tables, captions, footnotes, and supplementary spreadsheets. Reusing these observations in knowledge graphs (KGs), cross-paper search, and meta-analysis requires scientific interpretation as well as layout recovery. For example, <0.5 under an Au (ppm) header denotes a below-detection observation, distinct from a measured value. A composition percentage may denote wt% or mol%, with its meaning supplied by a header, caption, or footnote. Extraction must preserve these contextual distinctions when constructing structured records.

Prior work addresses parts of this problem under diferent assumptions. Semantic table interpretation and table-to-KG systems map cells, columns, or rows to entities, classes, or properties in a KG or ontology [1–4], but generally assume a usable table representation is available. Scientific information extraction systems operate closer to publications but are typically engineered for a single domain or target relation, e.g., chemical properties, glass compositions, or mineral-resource records [5–7]. We study an explicit task-module interface for reusing publication-level evidence acquisition and extraction across domains, while making the required domain-specific artifacts visible.

We investigate three research questions. RQ1: How does domain guidance afect extraction quality under a fixed backbone? RQ2: How can outputs from multiple evidence sources be reconciled while preserving their provenance? RQ3: How do acquisition failures and interpretation errors contribute to the remaining error? Same-LLM comparisons and component ablations address RQ1; the reconciliation design and backend ablations address RQ2; raw-PDF versus pre-parsed comparisons and error analysis address RQ3. The full-system comparison includes several architectural diferences; ontology ablations more directly assess the contribution of domain guidance.

ArticleMiner separates shared publication-level orchestration (§2) from a bounded, humanauthored ontology module supplying vocabulary, normalization and derivation rules, executable validation, identity, and RDF bindings (Figure 2). The module defines a task’s permissible outputs without axiomatizing the entire field. Adapting the framework requires this module, prompts, an output schema, and, where needed, a classifier assigning records to domain categories, such as geochemical deposit types.

This paper makes three contributions. (1) We formulate ontology-constrained publication-to-KG construction, applying domain knowledge during extraction, normalization, and validation rather than only after table recovery, preserving unit, null-value, identifier, and taxonomy semantics (RQ1). (2) We define an explicit ontology-module interface and evidence model for reconciling parser outputs into validated records with source and backend-support provenance (RQ2). (3) We report an empirical study of four bounded task instantiations over 163 papers, with raw-PDF and pre-parsed settings, matched same-LLM baselines, parser and component ablations, cost/runtime measurements, and JSON-LD export structure (RQ3). The evaluation distinguishes full-system comparisons from component ablations and reports uncertainty at the publication level.

## 2<sub>.</sub> P<sub>ro</sub>bl<sub>em</sub> F<sub>ormu</sub>l<sub>a</sub>ti<sub>on an</sub>d M<sub>e</sub>th<sub>o</sub>d

Figure 1 summarizes the five stages: parse, extract, self-correct, reconcile, and serialize. Sections 2.1 and 2.3 define the shared interface and orchestration; §2.2 instantiates the module for geochemistry. Each task supplies a module, prompts, a row schema, and an optional classifier. The four instantiations are GeoChem (mineral geochemistry), ChemTables (drug-discovery chemistry), DiSCoMaT (materials science), and MLTables (machine learning).

We anchor the exposition in a mineral-geochemistry example. A row reports Au<0.5 for sample $\mathrm { B R } - 1 2 ;$ the header specifies ppm, and an associated method label reads icpms. Recovering the observation requires three distinctions: the inequality denotes a detection limit rather than a measured concentration; the method alias maps to ICP-MS; and the unit comes from the header rather than the cell. This schematic example illustrates the method rather than a measured benchmark result. Each stage below shows how contextual interpretation and explicit module rules preserve this information.

## 2<sub>.</sub>1<sub>.</sub> Ontolo<sub>gy</sub> Module and Framework O<sub>p</sub>erator

A domain module specifies the permissible outputs of a bounded extraction task. It provides canonical vocabulary, surface-form mappings, deterministic derivations, executable constraints, identity keys, and RDF bindings. Unknown terms are retained or rejected according to the task’s validators; membership in the accepted-record space denotes conformance to these constraints, not independently verified factual correctness. The modules use labels, aliases, shallow hierarchies, and value conventions, without OWL reasoning. Validation runs as Python code. Simple admissibility and range checks can be represented as SHACL constraints, but no SHACL engine is executed in the evaluated pipeline.

We model a domain module as a six-tuple

$$
\Omega _ { d } = ( V _ { d } , C _ { d } , \Phi _ { d } , \Gamma _ { d } , \kappa _ { d } , \beta _ { d } ) ,\tag{1}
$$

The component types specify the interface between provisional records, task semantics, and accepted outputs. Let $\mathcal { \widetilde { R } } _ { d }$ denote the space of provisional records: partial assignments of the task’s fields to strings or numbers, as proposed by extraction; let $\mathcal { R } _ { d } \subseteq \widetilde { \mathcal { R } } _ { d }$ denote the accepted-record space of schema-complete, canonicalized records. $V _ { d }$ is a finite set ofcanonical task identifiers and types; $C _ { d } \colon \widetilde { \mathcal { R } } _ { d }  \widetilde { \mathcal { R } } _ { d }$ is a field-wise canonicalizer over provisional records; $\Phi _ { d } : \widetilde { \mathcal { R } } _ { d }  \widetilde { \mathcal { R } } _ { d }$ is a deterministic derivation operator; $\Gamma _ { d } = \{ \gamma _ { j } :$

![](images/f590d0521d05871894d5cd628763181d4efcacc0d41f83c14b17c83485c3f1e8.jpg)  
Fi<sub>g</sub>ure 1: Shared orchestration and task-module integration (schematic). Parsing preserves source locations; extraction <sub>p</sub>ro<sub>p</sub>oses records<sub>;</sub> tar<sub>g</sub>eted retries address missin<sub>g</sub> or unreadable evidence. Reconciliation a<sub>pp</sub>lies derivation, canonicalization, validation, and identity-based merging. JSON-LD exposes the resulting records; Turtle and <sub>g</sub>ra<sub>p</sub>h-<sub>q</sub>uer<sub>y</sub> labels de<sub>p</sub>ict intended downstream use<sub>,</sub> not evaluated native out<sub>p</sub>uts. Backend su<sub>pp</sub>ort measures identit<sub>y</sub> recover<sub>y,</sub> not factual correctness.

![](images/a2a0034c74b4390fec7c303d31794b8e41933766e00b9d16471f92c30ccbed57.jpg)  
Fi<sub>g</sub>ure 2: The six-com<sub>p</sub>onent task-module interface. The semantic core su<sub>pp</sub>lies vocabular<sub>y,</sub> normalization<sub>,</sub> derivations<sub>,</sub> and checks<sub>;</sub> identit<sub>y</sub> and RDF bindin<sub>g</sub>s su<sub>pp</sub>ort reconciliation and serialization. Exam<sub>p</sub>les combine t<sub>as</sub>k<sub>s;</sub> <sub>on</sub>l<sub>y</sub> d<sub>ec</sub>l<sub>are</sub>d <sub>mo</sub>d<sub>u</sub>l<sub>e</sub> <sub>ru</sub>l<sub>es</sub> <sub>app</sub>l<sub>y.</sub> E<sub>p</sub>ith<sub>erma</sub>l t<sub>ype</sub> <sub>a</sub>l<sub>one</sub> d<sub>oes</sub> <sub>no</sub>t d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub> <sub>su</sub>lfid<sub>a</sub>ti<sub>on</sub> <sub>c</sub>l<sub>ass.</sub> V<sub>a</sub>lid<sub>a</sub>t<sub>ors</sub> execute as P<sub>y</sub>thon <sub>p</sub>redicates<sub>,</sub> not as SHACL sha<sub>p</sub>es.

$\widetilde { \mathcal { R } } _ { d }  \{ 0 , 1 \} \}$ is a set ofrecord-level validators whose conjunction gates emission; $\kappa _ { d }$ is the task identity key inducing an equivalence relation over accepted records (§2.3, Reconcile); and $\beta _ { d }$ is the RDF binding that fixes the @context, IRIs, and datatypes ofemitted assertions (§2.3, Serialize). $C _ { d }$ and $\Phi _ { d }$ are total by construction: on surface forms outside their mapping tables, they act as the identity, so unmapped strings are preserved rather than coerced, and whether such a record is emitted is decided by $\Gamma _ { d }$ against the task schema. Canonical targets come from the benchmark schema and published domain resources (e.g., the USGS CMMI taxonomy and IMA mineral list for GeoChem); the released modules expose the curated surface-form mappings, and canonicalization precedes validation, so validators operate on canonical identifiers rather than on every synonym. Appendix E gives the per-task authoring sources. The other three modules fill the same interface with diferent content (Figure 2, Table 1).

Given a publication $P$ and a domain �, the framework produces the set of validated records

$$
\begin{array} { r } { R ( P , d ) = \big \{ C _ { d } ( \Phi _ { d } ( r ) ) : r \in \tilde { R } ( P , d ) , \Gamma _ { d } ( C _ { d } ( \Phi _ { d } ( r ) ) ) = 1 \big \} , } \end{array}\tag{2}
$$

where $\tilde { R } ( P , d ) \subseteq \widetilde { \mathcal { R } } _ { d }$ is the set of provisional records and $\Gamma _ { d } ( q ) = 1$ abbreviates $\Lambda _ { j } \gamma _ { j } ( q ) = 1$ . Extraction proposes; $\Phi _ { d }$ derives declared facts; $C _ { d }$ canonicalizes; and $\Gamma _ { d }$ accepts or rejects. A rejected record is not emitted as an accepted record; its source locator and structured rejection reasons are retained in the audit log. Adapting to a new domain consists of authoring $\Omega _ { d }$ (including its identity key and RDF bindings), three to six prompt templates, the output row schema, and an optional classifier subsystem when the task contains a non-trivial classification problem; Table 1 itemizes these artifacts per task. The evaluated modules contain approximately 60–220 entries and use flat vocabularies or shallow hierarchies. Validation logs can help identify coverage gaps, although a rejection can also indicate an extraction error.

A<sub>u</sub>th<sub>ore</sub>d <sub>ar</sub>tif<sub>ac</sub>t<sub>s</sub> f<sub>or</sub> th<sub>e</sub> f<sub>our</sub> t<sub>as</sub>k<sub>s.</sub> E<sub>n</sub>t<sub>ry coun</sub>t<sub>s summar</sub>i<sub>ze mo</sub>d<sub>u</sub>l<sub>e con</sub>t<sub>en</sub>t<sub>s, no</sub>t <sub>measure</sub>d <sub>a</sub>d<sub>ap</sub>t<sub>a</sub>ti<sub>on e</sub>f<sub>or</sub>t<sub>.</sub> Each task additionall<sub>y</sub> needs <sub>p</sub>rom<sub>p</sub>ts<sub>,</sub> an out<sub>p</sub>ut schema<sub>,</sub> and RDF bindin<sub>g</sub>s. GeoChem uses three to six <sub>p</sub>rom<sub>p</sub>ts and a se<sub>p</sub>arate de<sub>p</sub>osit classifier. Detailed entries a<sub>pp</sub>ear in A<sub>pp</sub>endix E.
<table><tr><td>Task</td><td>Entries</td><td>Vocabulary, rules, and checks</td><td>Identity and additional artifacts</td></tr><tr><td>GeoChem</td><td>220</td><td>Elements, minerals, methods, deposit mappings; units, null conventions, hierarchy, projection; 189-type classifier and plausibility checks</td><td>Sample + method; 209-column</td></tr><tr><td>DiSCoMaT</td><td>167</td><td>Oxide components; mol% and wt%; component admissibility and composition-sum checks</td><td>Material + constituent; composition tuple schema</td></tr><tr><td>MLTables</td><td></td><td>60 Cell types, metrics, tasks, datasets; type-dependent fields and metric checks</td><td>Typed-field fingerprint; entry-type schema</td></tr><tr><td>ChemTables</td><td></td><td>80 Assays, targets, treatments, units; value bounds and required-field checks</td><td>Treatment + target + assay; bioactivity tuple schema</td></tr></table>

The module intervenes at three distinct points. Prompt-time grounding guides which fields and meanings the model proposes; deterministic derivation, canonicalization, and validation constrain the resulting records; RDF bindings specify the graph representation. These roles are complementary. A canonicalizer can normalize a supplied unit or method alias, but cannot recover a missing caption that identifies the measurement. Conversely, a model may recover all relevant context while still emitting inconsistent labels. Separating these responsibilities makes it possible to inspect whether a failure arose from absent evidence, contextual interpretation, or a declared rule, rather than treating every error as a prompting failure.

## 2<sub>.</sub>2<sub>.</sub> G<sub>eoc</sub>h<sub>em</sub>i<sub>s</sub>t<sub>ry</sub> I<sub>ns</sub>t<sub>an</sub>ti<sub>a</sub>ti<sub>on</sub>

The geochemistry module illustrates four categories of semantic content within the six-component interface. Identity and RDF bindings are specified separately (Table 1).

Vocabulary and canonicalization $( V _ { d } , C _ { d } ) .$ . Mineral and analytical-method aliases map to canonical labels, such as la-icp-ms to LA-ICPMS. The approximately 220 entries summarize curated mappings and vocabulary entries; the separate deposit classifier draws on the full 189-type CMMI taxonomy. An unlisted surface form is retained for validation rather than assigned an unsupported canonical label. For example, laser ablation icp-msand the misspelling laipcmsshare the canonical target LA-ICPMS; electron microprobe maps to EPMA. Mineral entries additionally record coarse classes and formulae, such as sphalerite, sulfide, and ZnS. The mappings encode labels, aliases, and coarse classes rather than logical axioms. A spelling absent from the curated list is not guessed: its original form is retained and assessed by the validators. These are inspectable mappings, not assertions that all scientifically related terms are interchangeable.

Taxonomic derivations $( \Phi _ { d } ) .$ . A recognized deposit type determines its group and environment in the adopted USGS CMMI hierarchy [8]. This is a lookup over the published classification, rather than general logical inference. For instance, the adopted taxonomy places MVT zinc-lead in the Mississippi Valleytype group and Basin hydrothermal environment. Classifying the deposit from publication evidence is a model-assisted task; filling its parent categories after that classification is deterministic. Keeping these operations separate exposes an incorrect type assignment instead of treating the derived parents as independent corroboration.

Units and null conventions. An explicitly labeled wt% value can be converted to ppm by multiplying by $1 0 ^ { 4 }$ ; ambiguous units are not inferred from magnitude alone. A reported bound <0.5 is represented as −0.5, preserving its detection limit. The sentinel −99999 denotes below detection without a stated limit, while an unmeasured value remains blank. These conventions distinguish censored measurements from measured zero. For example, a query for samples with detectable gold must exclude both kinds of below-detection observation and unmeasured cells; collapsing them into a common missing-value marker would lose that distinction. Symbols such as n.d. and a dash require the source legend: a task default cannot establish their meaning in every publication.

Validation $( \Gamma _ { d } ) .$ Task checks examine admissible terms and units, required fields, and numerical plausibility. They constrain what is accepted, but cannot guarantee that a value is attached to the correct sample. Simple membership and range predicates have SHACL counterparts, such as sh:in and sh:minInclusive; context-sensitive extraction decisions need additional logic. A rejected candidate remains outside the accepted output, with its evidence locator and failure reason retained so that the omission can be inspected. The finite vocabulary and shallow semantics bound the scope ofthe module.

## 2<sub>.</sub>3<sub>.</sub> Sh<sub>are</sub>d O<sub>rc</sub>h<sub>es</sub>t<sub>ra</sub>ti<sub>on</sub>

The shared orchestration consumes a raw publication together with its supplementary files and produces $R ( P , d )$ . We describe each stage in terms of what fails without it, using the running geochemistry example as the lead illustration; the quantitative ablation evidence appears in §4.2.

Parse. A single PDF backend has uneven coverage. The framework supports five PDF backends (Docling, Marker, MinerU, pdfplumber, and Camelot) and treats their outputs as evidence; the evaluation includes single-backend, leave-one-out, and all-five configurations. Docling reads publisher-rendered tables but fails on some landscape-oriented or image-only tables; pdfplumber recovers grids that other backends miss; and Camelot handles some ruled and borderless layouts but can corrupt merged head ers, so no single output is trusted. Table 5(b) reports controlled backend configurations. MinerU lacked OCR weights in the evaluation environment, so its results do not characterize a fully enabled installation. Additional rotation handling and adaptive backend selection in the release are outside the reported evaluation (Appendix I). An evidence unit $e \in E ( P )$ records its kind (PDF table, spreadsheet sheet, caption, header, or footnote), content, source locator, and container-level annotations such as workbook and sheet names; $E ( P )$ denotes the set of all evidence units associated with publication $P .$

In the running geochemistry example, raw measurements are commonly provided in multi-sheet supplementary spreadsheets. Workbook names such as ICP-MS\_results.xlsx may identify the analytical method, sheet names such as sphalerite may group observations by mineral, and column headers such as Au\_ppm specify the element and measurement unit. These container-level annotations are retained as evidence-unit metadata, allowing the extraction model to infer method=ICP-MS even when the method is not stated within an individual row.

Extract. An extraction call conditioned on evidence $E ( P )$ and module $\Omega _ { d }$ proposes records $f _ { \theta } ( E ( P ) , \Omega _ { d } ) \subseteq \widetilde { \mathcal { R } } _ { d } .$ , where � denotes the backbone; a publication may require several calls. Prompt-time grounding supplies admissible labels, units, and null conventions. It helps the model associate a cell with the contextual fields needed by subsequent deterministic normalization. It does not replace the canonicalizer: explicitly declared aliases and value patterns remain normalizable even when returned as strings. Geochemistry additionally uses a separate 189-type deposit classifier; the other tasks omit this subsystem.

For example, the geochemistry prompt supplies method labels such as ICP-MS and XRF, unit classes such as ppm and wt%, and the distinction between measured, below-detection, and unmeasured observations. In the running example, extraction must associate <0.5 with Au, ppm, sample BR-12, and the analytical method before normalization can represent the observation correctly. The same cell string could denote a diferent element or unit in another table. Vocabulary constraints narrow the admissible labels, but the publication supplies the evidence connecting those labels to the cell. This is why domain guidance is applied during extraction as well as after it.

Se<sup>lf</sup>-correct. Self-correction detects likely structural failures and retries the afected acquisition stage. In the geochemistry implementation, a pre-extraction blueprint, an LLM-generated profile of the publication, estimates the expected sample count �^ from prose, captions, and table descriptions. The blueprint uses contextual descriptions to anticipate how many samples of extraction should recover. It’s an estimate gate escalation rather than directly modifying measurement values; it is a diagnostic heuristic, not ground truth. When text extraction returns no samples or fewer than 0.85^� samples, an enabled vision stage reads table images and supplements the recovered sample set. After these fallbacks, a completeness retry is triggered for a nonempty PDF sample set when $| R _ { \mathrm { p d f } } | < \hat { n } / 2$ and $| R _ { \mathrm { p d f } } | < \hat { n } - 5$ . These conditions require both a relative shortfall exceeding 50% and an absolute shortfall exceeding five samples, so a small count discrepancy alone does not trigger another pass. The retry lowers the required number of recognized element columns from three to two and replaces the earlier set only if it recovers more samples.

For a failed supplementary table, a diagnostic LLM receives a preview and returns parsing hints, such as the header row, orientation, or identifier column. The parser reruns with those hints, with at most two diagnostic attempts; the PDF correction path uses one attempt. For example, identifying row 3 as the header avoids treating two rows of license text as column names. Correcting such a structural error can restore many observations at once. These retries do not establish semantic correctness: an inaccurate expected count can also encourage over-extraction. Correction and vision stages are therefore configurable, and their efects are assessed by task and backbone (§4.2).

The diagnostic call proposes parsing parameters rather than replacement measurement values. A misplaced header, transposed table, or incorrect identifier column can suppress many rows simultaneously; correcting the responsible parameter allows the parser to recover those rows from the original evidence. The retry limit bounds repeated diagnosis, without making the resulting records inherently reliable. Vision addresses a diferent failure: changing text-parser parameters cannot recover cells that are absent from the machine-readable evidence.

Reconci<sup>l</sup>e. Multiple backends and passes produce many candidates for the same underlying observation. Reconciliation first applies $\Phi _ { d }$ and $C _ { d }$ to each provisional record, validates the canonicalized record with $\Gamma _ { d } .$ , and groups accepted candidates by the module’s identity key $\kappa _ { d } .$ , which encodes domain identity rather than string equality. In geochemistry, $\kappa _ { d } = ( \mathrm { s a m p l e } _ { \mathrm { - } }$ \_name,analytical\_method), so records for BR-12 under ICP-MS from the PDF and from the supplementary spreadsheet are merged, while ICP-MS vs. LA-ICP-MS records for the same sample are not. The other domains’ identity keys are listed in Table 1; machine-learning tables use a normalized text fingerprint over typed fields because no reliable semantic key exists. For each $\kappa _ { d }$ -equivalence class, let � be the set ofactive backends and $R _ { s }$ the candidates recovered by backend �. The backend support

$$
a ( x ) = \frac { 1 } { | S | } { \sum _ { s \in S } } \mathcal { k } \big [ \exists x ^ { \prime } \in R _ { s } : \kappa _ { d } ( x ^ { \prime } ) = \kappa _ { d } ( x ) \big ]\tag{3}
$$

records the fraction of backends recovering the same identity. It does not require agreement on every field, does not resolve conflicting values by itself, and is not a calibrated probability ofcorrectness; backend errors can be correlated. The provenance retains source locations and backend support for inspection. A record recovered by only one backend can still be valid; support is not a majority-vote acceptance rule. The identity key, therefore, defines which observations are candidates for reconciliation, while the remaining fields describe those observations. For example, two candidates with the same sample and method may disagree about the concentration or its unit. Their shared identity contributes to backend support, but they cannot establish which value is correct. Keeping identity, field content, and source evidence conceptually separate avoids interpreting repeated recovery ofa sample name as independent verification of its measurements.

Seria<sup>l</sup>ize. The framework exposes accepted records in JSON-LD, with task-specific terms and source metadata. The RDF binding $\beta _ { d }$ specifies how record fields map to graph predicates and datatypes. Appendix J describes the released exporter<sup>'</sup>s record structure and its limits; serialization is not an additional evaluated prediction task. The released JSON-LD exporter and downstream RDF validation are separate from the tuple-level evaluation reported here. The exported structure separates sample identity from elemental measurements and source metadata. Sample nodes carry a paper-scoped identifier and sample name; measurement entries carry the element and value, with units when available. This structure allows consumers to retain the association between an observation and its source publication. It does not itself resolve sample identities across papers, verify external ontology mappings, or establish the correctness of a downstream query.

## 2.4. A Worked Exam<sub>p</sub>le

For sample BR-12, a header supplies the unit ppm and contextual evidence identifies ICP-MS as the method. Canonicalization represents Au<0.5 as a below-detection value with a limit of 0.5, and validation checks conformance to the task schema. Iffour offive backends recover their identity, their support is $a ( x ) = 0 . 8 \mathrm { { : } }$ ; that support does not independently verify the concentration. An RDF representation can then distinguish queries for detected gold from queries for samples analyzed by ICP-MS. This is an illustrative example rather than an additional evaluated observation.

The same separation applies beyond geochemistry. In a glass-composition table, a caption declaring mol% prevents treating a value as wt%, even when both composition vectors sum to 100. In a bioactivity table, IC50 >10 with unit �M is a lower bound, not a measured concentration of 10; the target and compound must remain attached to that bound. In an ML result table, Ours: 87.3 becomes interpretable only after the caption or prose identifies the model, dataset, and metric. These examples show why output-schema conformance is necessary but insuficient: a structurally valid record can still express the wrong scientific claim. In particular, a query for samples analyzed by ICP-MS can include the running geochemistry observation, whereas a query for detected gold must exclude its censored measurement. The distinction depends on preserving the detection-limit convention together with the element and unit. Similarly, the bound bioactivity must remain attached to its compound and target; preserving the inequality alone would not identify which scientific observation it qualifies.

## 3<sub>.</sub> Ev<sub>a</sub>l<sub>ua</sub>ti<sub>o</sub>n D<sub>es</sub>i<sub>g</sub>n

We evaluate ArticleMiner on four benchmarks (Table 2) chosen to span key dimensions of scientific table extraction: input modality (PDF only vs. PDF+supplementary), ontology size (60–220 terms), tuple granularity (cell-, row-, or sample-level), and reliance on supplementary evidence. The benchmark sizes are severely imbalanced (111 DiSCoMaT papers, 28 GeoChem, 15 MLTables, and 9 ChemTables), so the two smallest datasets are treated throughout as case evidence rather than population-level confirmation, and the 163 papers are not 163 independent draws of domain portability. Primary comparisons hold the backbone fixed and compare publication-level extraction with few-shot prompting; the pre-parsed setting, backend, and component ablations, and per-domain best configurations are secondary or posthoc analyses and are labeled as such.

## 3<sub>.</sub>1<sub>.</sub> B<sub>enc</sub>h<sub>mar</sub>k D<sub>a</sub>t<sub>ase</sub>t<sub>s</sub>

GeoChem, introduced in this paper, contains 28 peer-reviewed mineral-geochemistry papers with ground truth curated by domain experts from the U.S. Geological Survey. The annotations follow the CMiO-MIN record structure: each gold record represents a mineral observation and specifies hierarchical sample identifiers and elemental measurements, linked to mineral, analytical-method, and deposit metadata. Deposit types are drawn from the 189-category CMMI classification scheme of Hofstra et al. [8]. The benchmark contains 1,236 unique sample identifiers and 87,759 non-null elemental measurements. It is the largest benchmark in our evaluation by number of observations and the only one in which extracted records are matched to the ground truth using sample identifiers. We use the complete 28-paper corpus for all primary GeoChem results.

Three external benchmarks evaluate the framework in other domains. ChemTables contains XMLformatted drug-discovery tables; each gold record links a numeric cell to its value, measurement type (e.g., IC ), biological target, treatment, and unit [4]. DiSCoMaT contains CSV-formatted glasscomposition tables with gold tuples specifying the material, chemical component, percentage, and unit (mol% or wt%) [6]. MLTables contains LAT X-formatted machine-learning tables whose cells are labeled as Result, Hyper-parameter, Data Statistics, or Other [4]; interpreting these cells often requires evidence from the surrounding paper.

## T<sub>a</sub>bl<sub>e</sub> 2

Benchmark sizes and a<sub>pp</sub>roximate module counts.
<table><tr><td>Dataset</td><td>Publications</td><td>Tables</td><td></td><td>Gold records Module entries</td></tr><tr><td>DISCoMAT</td><td>111</td><td>175</td><td>4,755</td><td>167</td></tr><tr><td>GEOCHEM</td><td>28</td><td>~80</td><td>5,307</td><td>220</td></tr><tr><td>MLTABLES</td><td>15</td><td>68</td><td>2,060</td><td>60</td></tr><tr><td>CHEMTABLES</td><td>9</td><td>14</td><td>462</td><td>80</td></tr></table>

## T<sub>a</sub>bl<sub>e</sub> 3

Cost (\$) and runtime (s) per paper.

<table><tr><td rowspan="2">Task</td><td colspan="2">Sonnet</td><td colspan="2">Haiku</td><td colspan="2">Few-shot</td></tr><tr><td>S</td><td>$</td><td>S</td><td>$</td><td>S</td><td>$</td></tr><tr><td>CHEMTABLES</td><td>60</td><td>0.10</td><td>86</td><td>0.03</td><td>17</td><td>0.05</td></tr><tr><td>MLTABLES</td><td>120</td><td>0.10</td><td>118</td><td>0.03</td><td>62</td><td>0.05</td></tr><tr><td>DiSCoMaT</td><td>140</td><td>0.10</td><td>40</td><td>0.03</td><td>25</td><td>0.05</td></tr><tr><td>GeoChem</td><td>180</td><td>0.30</td><td>120</td><td>0.10</td><td>21</td><td>0.05</td></tr></table>

## 3<sub>.</sub>2<sub>.</sub> Four-Tier Evaluation for GeoChem

A wrong deposit type, a missing sample, an incorrect concentration, and an incorrectly populated missing value afect scientific reuse diferently. The four tiers separate these errors; numerical accuracy receives the largest weight because concentrations drive downstream geochemical analysis.

GeoChem uses the composite score $S { = } 0 . 3 0 T _ { 1 } { + } 0 . 4 0 T _ { 2 } { + } 0 . 1 5 T _ { 3 } { + } 0 . 1 5 T _ { 4 }$ . The tiers assess diferent properties and must be interpreted separately. In the released evaluator, $T _ { 1 }$ (metadata) averages normalized string-similarity scores for paper metadata; absent gold fields receive full credit. $T _ { 2 }$ (numerical accuracy) averages numerical scores over matched rows: nonzero values within 5% relative error receive full credit, with linearly decreasing partial credit up to 100% error; zero gold values use an absolute tolerance of 0.001. Below-detection sentinels have separate partial-credit rules. $T _ { 3 }$ (sample recovery) measures sample matching, using identifier matching with a position-based fallback when fewer than 30% of gold rows match by name. $T _ { 4 }$ (missing-value agreement) scores whether fields absent in the gold remain null; non-null gold values, including below-detection observations, are handled by $T _ { 2 }$ . Thus, $T _ { 2 }$ and $T _ { 4 }$ are conditional on row matching and do not independently measure extraction recall or full three-way null semantics. Appendix C specifies the 209-column projection, and Appendix C.1 gives the remaining scoring details.

## 3<sub>.</sub>3<sub>.</sub> E<sub>va</sub>l<sub>ua</sub>ti<sub>on</sub> M<sub>e</sub>t<sub>r</sub>i<sub>cs</sub>

For the three external benchmarks, we report precision, recall, and strict tuple- $F _ { 1 } { \mathrm { : } }$ a prediction counts as correct only when the required tuple fields match. GeoChem instead uses sample matching and the four-tier score defined above; its scores are not interchangeable with strict tuple-�<sub>1</sub>. For DiSCoMaT [6], this matches the published $\mathrm { T u p l e } { \cdot } F _ { 1 }$ setting. Our setting is stricter than the attribute-level token- $F _ { 1 }$ of Bai et al. [4], which gives partial credit for ≥25% token overlap; their headline numbers are therefore not directly comparable to ours and appear only as context (§4.1). For GeoChem, the four-tier score is primary; we also report sample-row $\mathrm { P / R } / F _ { 1 }$ for cross-benchmark comparability.

## 3<sub>.</sub>4<sub>.</sub> B<sub>ase</sub>li<sub>nes,</sub> C<sub>on</sub>fi<sub>gura</sub>ti<sub>ons,</sub> <sub>an</sub>d St<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>l P<sub>ro</sub>t<sub>oco</sub>l

Comparison matrix. We compare ArticleMiner against the same-LLM few-shot baselines and published task-specific systems. These are system comparisons rather than interventions on domain knowledge alone. B1, PDF <sup>f</sup>ew-s<sup>h</sup>ot: the same backbone receives the raw-PDF text in a single call with a fixed three-exemplar prompt, temperature zero, the task output schema, and the same metric scripts. It does not receive ontology grounding, executable validation, cross-backend reconciliation, vision, retries, or supplementary-file orchestration. In GeoChem, ArticleMiner reads supplementary spreadsheets while B1 receives PDF text and a compact row schema; both are scored on the common evaluation projection. The comparison, therefore, includes unequal evidence access, schema presentation, and identifier handling. B2, pre-parsed <sup>f</sup>ew-s<sup>h</sup>ot: the same protocol starts from the benchmark’s released table text, separating evidence acquisition from extraction. Pub<sup>l</sup>is<sup>h</sup>ed systems: the GNN of Gupta et al. [6] on DiSCoMaT is compared under tuple- ${ \bf \nabla } \cdot { \cal F } _ { 1 }$ ; InstrucTE of Bai et al. [4] on ChemTables/MLTables uses attribute-level token- ${ \bf \nabla } \cdot F _ { 1 }$ and is reported only as non-comparable context. All five closed backbones are evaluated on all four benchmarks; the full grid in the appendix additionally reports three open 7–8B backbones.

Hardware, so<sup>f</sup>tware, and protoco<sup>l</sup>. We run on a Linux workstation (Ubuntu 22.04) with one NVIDIA H100 80 GB for open-weight inference (vLLM 0.6.x); closed backbones are called via oficial APIs. PDF backends are docling, marker, MinerU, pdfplumber, and Camelot (version families, model identifiers, and environment notes in Appendix I). Temperature is zero; resampling and exemplar ordering use seed 42. Table 4(b) reports mean per-paper scores and percentile bootstrap intervals (1,000 resamples), paired by publication identifier. Sonnet costs approximately \$0.10 per external-benchmark paper and \$0.30 per GeoChem paper; the corresponding Haiku costs are \$0.03 and \$0.10 (Table 3). A no\_pdf record denotes one of two GeoChem data-reuse entries without a standalone source PDF, not a model or pipeline fail ure. Such entries are excluded from PDF-dependent averages: headline spreadsheet-aware runs use 28 papers, open-LLM PDF runs use $^ { 2 6 , }$ and the paired analysis uses the 25 papers with valid outputs from both compared systems.

## 4<sub>.</sub> R<sub>esu</sub>lt<sub>s</sub>

The section follows the research questions: the matched comparison (§4.1, Table 4) and component ablations (§4.2, Table 5a) answer RQ1; the parser and backbone analyses (§4.3, Table 5b) answer RQ2; the raw-PDF/pre-parsed contrast and error analysis (§4.4, Table 6) answer RQ3.

## 4<sub>.</sub>1<sub>.</sub> Same-Backbone Com<sub>p</sub>arison and Uncertaint<sub>y</sub>

In the raw-publication comparison, ArticleMiner has higher point estimates than the same-LLM fewshot baseline on all four tasks and all five closed backbones (Table 4a). Panel (b) reports publication-level means and paired diferences, rather than the aggregate scores in panel (a). Reported mean gains are 8.8 points on ChemTables, 14.4 on MLTables, 15.5 on DiSCoMaT, and 62.3 sample-�<sub>1</sub> points on the 25-paper GeoChem paired subset. The small ChemTables and MLTables samples yield substantial uncertainty. The paired summaries are retained as descriptive evidence; their configuration-specific run provenance is incomplete, so they do not establish a formal significance claim. The GeoChem comparison includes access to supplementary spreadsheets and is not an isolated test of ontology grounding.

For the five closed backbones, ArticleMiner scores higher with pre-parsed input on all three external benchmarks (Tables 10–13); GeoChem has no separate pre-parsed comparison in that table. For example, Opus 4.6 reaches 81.0 tuple- ${ \bf \nabla } \cdot F _ { 1 }$ on raw DiSCoMaT PDFs and 94.7 on pre-parsed tables. This gap indicates acquisition losses, but does not isolate parsing from changes in evidence representation or prompting. The published DiSCoMaT GNN score of 70.04 [6] uses tuple- $F _ { 1 }$ and is reported as external context,

Same-backbone comparison in the raw-publication setting. (a) Strict tuple- ${ \bf \nabla } \cdot { \cal F } _ { 1 }$ <sub>on</sub> th<sub>e</sub> th<sub>ree ex</sub>t<sub>erna</sub>l t<sub>as</sub>k<sub>s;</sub> GeoChem reports composite � and sample- $F _ { 1 }$ on 28 entries. ArticleMiner also reads GeoChem su<sub>pp</sub>lements<sub>;</sub> few-shot uses PDF text. Bold marks the larger score within each pair. (b) Reported Sonnet per-paper means, <sub>p</sub>ercentile 95% bootstra<sub>p</sub> intervals<sub>,</sub> and <sub>p</sub>aired diferences<sub>;</sub> GeoChem uses 25 <sub>p</sub>aired entries. These historical summaries are descriptive; exact run provenance remains incomplete. Panel (b) is not obtained by subtracting panel (a)’s aggregate scores; rounding can also afect displayed diferences.
<table><tr><td colspan="7">(a) Same-backbone comparison across five closed backbones</td></tr><tr><td>Backbone</td><td>System</td><td>ChemTables MLTables</td><td></td><td>DiSCoMaT</td><td>GeoChem S</td><td>GeoChem  $F _ { 1 }$ </td></tr><tr><td rowspan="3">Sonnet 4.6</td><td>ARTICLEMINER</td><td>32.1</td><td>52.2</td><td>79.3</td><td>76.0</td><td>60.8</td></tr><tr><td>Few-shot</td><td>15.8</td><td>43.2</td><td>64.5</td><td></td><td>0.8</td></tr><tr><td>ARTICLEMINER</td><td>31.8</td><td>55.9</td><td>81.0</td><td>76.3</td><td>60.6</td></tr><tr><td rowspan="2">Opus 4.6 Haiku 4.5</td><td>Few-shot</td><td>15.9</td><td>41.6</td><td>59.5</td><td></td><td>3.0</td></tr><tr><td>ARTICLEMINER</td><td>25.9</td><td>51.6</td><td>72.4</td><td>75.5</td><td>54.3</td></tr><tr><td rowspan="2">GPT-40</td><td>Few-shot</td><td>15.8</td><td>38.0</td><td>55.8</td><td></td><td>4.1</td></tr><tr><td>ARTICLEMINER</td><td>30.2</td><td>45.6</td><td>61.9</td><td>75.9</td><td>63.1</td></tr><tr><td rowspan="2"></td><td>Few-shot</td><td>16.5</td><td>32.8</td><td>51.9</td><td></td><td>3.6</td></tr><tr><td>ARTICLEMINER</td><td>22.6</td><td>42.7</td><td>75.9</td><td>75.7</td><td>57.6</td></tr><tr><td>Gemini 2.5</td><td>Few-shot</td><td>14.3</td><td>42.4</td><td>52.1</td><td></td><td>0.3</td></tr><tr><td colspan="7">(b) Sonnet 4.6 paired statistics (mean [95% Cl]; ∆)</td></tr><tr><td></td><td colspan="2">System mean</td><td colspan="2">Baseline mean</td><td colspan="2"> $\Delta F _ { 1 }$ </td></tr><tr><td>CHEMTABLES (n=9)</td><td colspan="2">36.4 [10.8, 66.6]</td><td colspan="2">27.6 [5.9, 54.8]</td><td colspan="2">+8.8</td></tr><tr><td>MLTABLES (n=15)</td><td colspan="2">52.2 [36.2, 68.1]</td><td colspan="2">37.8 [25.5, 51.4]</td><td colspan="2">+14.4</td></tr><tr><td>DiSCoMaT (n=111)</td><td colspan="2">66.5 [58.1, 73.8]</td><td colspan="2">51.1 [43.3, 59.2]</td><td colspan="2">+15.5</td></tr><tr><td>GeoChem (n=25)</td><td colspan="2">63.2 [49.9, 76.1]</td><td colspan="2">0.8 [0.3, 1.6]</td><td colspan="2">+62.3</td></tr></table>

without a paired significance test. On GeoChem, Sonnet’s reported tier means are $T _ { 1 } { = } 7 1 . 6 , T _ { 2 } { = } 7 6 . 5 ,$ $T _ { 3 } = 6 8 . 5$ , and $T _ { 4 } = 9 0 . 6 $ , with $S = 7 6 . 0$ . Numerical and null scores are conditional on matched rows; they cannot establish that all values were recovered or assigned correctly. The baseline’s low sample scores reflect both identifier mismatch and the lack of the supplementary file path. These diferences explain why the gap from ArticleMiner cannot be attributed to ontology guidance alone.

## 4<sub>.</sub>2<sub>.</sub> C<sub>on</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> <sub>o</sub>f th<sub>e</sub> D<sub>oma</sub>i<sub>n</sub> S<sub>pec</sub>ifi<sub>ca</sub>ti<sub>on:</sub> C<sub>omponen</sub>t Abl<sub>a</sub>ti<sub>ons</sub>

The full ablation grid uses Haiku 4.5 to limit inference cost; Sonnet 4.6 reruns assess backbone sensitivity. Under Haiku 4.5, removing ontology grounding reduces tuple- ${ \bf \nabla } \cdot F _ { 1 }$ by 16.3 points on DiSCoMaT, 7.7 on MLTables, and 2.2 on ChemTables (Table 5a). GeoChem’s full row reports 54.3/75.5 for sample- ${ \cal F } _ { 1 } / S ;$ ; every component-removed row reports 51.0 sample- ${ \bf - } F _ { 1 }$ , while $S$ ranges from 72.7 to 74.5. Equality among the ablated rows does not imply equality with the full pipeline. Because these rows combine a headline reference with separately collected component runs, their GeoChem diferences require configuration and ground-truth alignment before a causal interpretation.

Component efects also depend on the backbone. On Sonnet 4.6, removing vision reduces ChemTables from 32.1 to 14.6 and MLTables from 52.2 to 4.8; on Haiku, validation removal causes larger losses than vision removal on both tasks. Removing paper intelligence raises ChemTables from 32.1 to 44.6 under Sonnet but lowers it under Haiku (25.9 to 23.4). On Haiku DiSCoMaT, its removal raises $F _ { 1 }$ from 72.4 to 76.5. These are diagnostic results, not evidence for a universally optimal component set; Table 7 gives the Sonnet details. Thus, a richer pipeline is not automatically a better extractor. A blueprint or retry can expose missing evidence, but can also introduce additional candidates that are irrelevant or incorrectly interpreted. The observed reversals motivate evaluating correction policies jointly with the task and backbone. They do not establish which intermediate error caused a particular gain or loss.

## 4<sub>.</sub>3<sub>.</sub> A<sub>cqu</sub>i<sub>s</sub>iti<sub>on,</sub> B<sub>ac</sub>k<sub>en</sub>d<sub>s,</sub> <sub>an</sub>d B<sub>ac</sub>kb<sub>one</sub> S<sub>ens</sub>iti<sub>v</sub>it<sub>y</sub>

PDF backends. On DiSCoMaT, docling alone reaches 71.7 $F _ { \mathrm { 1 } } ;$ the leave-Docling-out configuration scores 43.5, compared with 71.0 for all-five fusion. On ChemTables and MLTables, single docling

## T<sub>a</sub>bl<sub>e</sub> 5

Haiku 4.5 component and backend analyses. (a) Component removals: tuple- $F _ { 1 }$ on ChemTables/MLTables/DiS-CoMaT $( n = 9 / 1 5 / 1 1 1 ) \colon$ <sub>;</sub> GeoChem re<sub>p</sub>orts sam<sub>p</sub>le- $F _ { 1 } / S .$ <sub>.</sub> Th<sub>e</sub> G<sub>eo</sub>Ch<sub>em</sub> f<sub>u</sub>ll <sub>row</sub> i<sub>s</sub> th<sub>e</sub> 28<sub>-paper</sub> h<sub>ea</sub>dli<sub>ne re</sub>f<sub>-</sub> <sub>erence, w</sub>hil<sub>e componen</sub>t<sub>-remove</sub>d <sub>runs were co</sub>ll<sub>ec</sub>t<sub>e</sub>d <sub>separa</sub>t<sub>e</sub>l<sub>y.</sub> E<sub>qua</sub>l <sub>a</sub>bl<sub>a</sub>t<sub>e</sub>d <sub>scores</sub> d<sub>o no</sub>t i<sub>mp</sub>l<sub>y equa</sub>lit<sub>y</sub> with that reference. (b) Controlled backend configurations; GeoChem uses 26 common entries with corrected <sub>g</sub>round truth. Com<sub>p</sub>are within each <sub>p</sub>anel and run condition. Bold marks the best dis<sub>p</sub>la<sub>y</sub>ed value; ↑ indicates im<sub>p</sub>rovement over the com<sub>p</sub>onent reference. Sonnet sensitivit<sub>y</sub> is in Table 7.
<table><tr><td colspan="5">(a) Component ablation</td></tr><tr><td>Variant</td><td>ChemTables</td><td>MLTables</td><td>DiSCoMaT</td><td>GeoChem  $F _ { 1 } / S$ </td></tr><tr><td>Full pipeline</td><td>25.9</td><td>51.6</td><td>72.4</td><td>54.3/ 75.5</td></tr><tr><td>w/o ontology</td><td>23.7</td><td>43.9</td><td>56.1</td><td>51.0/ 73.2</td></tr><tr><td>w/o self-correct</td><td>23.1</td><td>49.8</td><td>67.1</td><td>51.0/ 74.5</td></tr><tr><td>w/o validation</td><td>21.7</td><td>41.3</td><td>71.2</td><td>51.0/ 74.4</td></tr><tr><td>w/o intelligence</td><td>23.4</td><td>46.3</td><td> $7 6 . 5 ^ { \uparrow }$ </td><td>51.0/ 72.7</td></tr><tr><td>w/o vision</td><td>22.3</td><td>48.4</td><td> $7 1 . 6$ </td><td>51.0/ 74.2</td></tr><tr><td colspan="5">(b) PDF-backend configurations (F1)</td></tr><tr><td>Configuration</td><td>ChemTables</td><td>MLTables</td><td>DiSCoMaT</td><td>GeoChem</td></tr><tr><td>docling only</td><td>26.2</td><td>56.3</td><td>71.7</td><td>47.9</td></tr><tr><td>marker only</td><td>23.2</td><td>51.4</td><td>66.2</td><td>48.0</td></tr><tr><td>mineru only</td><td>23.2</td><td>51.6</td><td>66.5</td><td>46.3</td></tr><tr><td>pdfplumber only</td><td>23.4</td><td>51.0</td><td>66.3</td><td>46.3</td></tr><tr><td>camelot only</td><td>25.5</td><td>43.2</td><td>43.7</td><td>48.0</td></tr><tr><td>leave-one-out docling</td><td>24.2</td><td>41.9</td><td>43.5</td><td>46.0</td></tr><tr><td>leave-one-out camelot</td><td>24.8</td><td>55.3</td><td>72.5</td><td>47.9</td></tr><tr><td>all-five fusion</td><td>20.8</td><td>47.6</td><td>71.0</td><td>54.3</td></tr></table>

exceeds all-five fusion (26.2 vs. 20.8; 56.3 vs. 47.6), a negative result for universal multi-backend fusion. On GeoChem, all single and leave-one-out variants cluster at $4 6 { - } 4 8 \ { F _ { 1 } }$ , while all-five fusion reaches 54.3 on the same papers (Table 5(b)). The number ofbackends is therefore a per-task choice rather than a fixed prescription. LLMbackbones. On GeoChem, closed backbones cluster tightly at 75.5–76.3 Overall � while open 7–8B models trail at 52.6–54.0, under a diferent 26-paper subset and a 512-token stage cap, with deposit-classification failures reported in Appendix H.4. This comparison does not isolate model capacity. Few-shotcount. The ChemTables exemplar-count analysis plateaus near three examples (Figure 4b); it does not establish that context budgets are suficient for all tasks. Cross-LLM agreement is 52–58% on the external benchmarks but 75% on GeoChem (Table 9, Appendix H).

## T<sub>a</sub>bl<sub>e</sub> 6

Error distribution (% of error events; $n _ { \mathrm { e r r } } = { \tt c o u n t } )$ for ArticleMiner (Sonnet 4.6) vs. the same-LLM few-shot baseline. Classes: over-extraction<sub>,</sub> omission<sub>,</sub> attribute mismatch<sub>,</sub> near miss. GPT-4o rows in A<sub>pp</sub>endix H.
<table><tr><td>Task</td><td>System</td><td>Over-extraction</td><td>Omission</td><td>Attribute</td><td>Near</td><td> $n _ { \mathrm { e r r } }$ </td></tr><tr><td>CHEMTABLES</td><td>ARTICLEMINER</td><td>57.0</td><td>16.1</td><td>26.6</td><td>0.4</td><td>704</td></tr><tr><td></td><td>Few-shot</td><td>10.5</td><td>70.9</td><td>18.0</td><td>0.7</td><td>440</td></tr><tr><td>MLTABLES</td><td>ARTICLEMINER</td><td>59.9</td><td>30.7</td><td>3.9</td><td>5.5</td><td>2163</td></tr><tr><td></td><td>Few-shot</td><td>27.6</td><td>49.8</td><td>5.8</td><td>16.7</td><td>1679</td></tr><tr><td>DiSCoMaT</td><td>ARTICLEMINER</td><td>2.7</td><td>92.1</td><td>5.2</td><td>0.0</td><td>1721</td></tr><tr><td></td><td>Few-shot</td><td>12.1</td><td>75.5</td><td>6.5</td><td>2.5</td><td>2697</td></tr></table>

## 4<sub>.</sub>4<sub>.</sub> Error Anal<sub>y</sub>sis

The diagnostic taxonomy separates unmatched predictions (over-extraction), unrecovered gold values (omission), matched values with incorrect type/component or unit (attribute mismatch), and otherwise unmatched numerical values within 5% of a gold value (near miss). These categories describe value-level matching and do not replace strict tuple evaluation. Errors fall into these four classes (Table 6); model choice mostly shifts the precision–recall balance (Fig. 5). Sonnet over-extracts on ChemTables/MLTables but under-extracts on DiSCoMaT; GPT-4o under-extracts everywhere, with omission 59–83% of its errors (Table 14, Appendix H); omission accounts for approximately 50–75% of the baseline error events, not ofall gold tuples. Open-weight models on DiSCoMaT show high precision but low recall (P=79–88%, R=10–43%): validation filters a strong generator’s invalid tuples but cannot recover ones never proposed, showing that schema validation alone cannot recover omitted candidates.

The error counts and proportions must be read together. For Sonnet on DiSCoMaT, omissions constitute 92.1% of ArticleMiner’s 1,721 error events, compared with 75.5% of 2,697 for few-shot prompting. The larger omission share, therefore, does not imply more omissions in absolute terms. On ChemTables and MLTables, over-extraction instead contributes 57.0% and 59.9% of ArticleMiner’s errors. These contrasting profiles motivate diferent interventions: inspect missing evidence in the former setting and candidate relevance in the latter, rather than applying uniformly stricter validation.

## 5<sub>.</sub> R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d W<sub>or</sub>k

Table-to-KGand semantictableannotation. The SemTab challenge and systems such as DAGOBAH and SAND, with domain-independent interpretation and DBpedia table matching, annotate cells, columns, and properties over already-recovered tables [1–3, 9, 10]. Because a cell’s meaning depends on its surrounding evidence (headers, captions, units, sheet names, null conventions), ArticleMiner instead couples parsing with interpretation.

Scientific IE and domain-specific KGs. ChemDataExtractor, DiSCoMaT, and MinMod [5–7] show the value of domain knowledge but are each engineered around one domain, source, or target schema. ArticleMiner instead externalizes domain knowledge into a module over a shared orchestration, so adaptation centers on explicit domain artifacts (Table 1), including executable rules and, when needed, a classifier.

LLM-based table extraction and document parsing. InstrucTE [4], LLM data-integration systems [11], table models such as TaPas and TURL [12, 13], operate primarily over structured or recovered tables; surveys discuss the broader use of language models with tabular data [14, 15]. For raw PDFs, layout-aware parsers [16–18] help, but no single parser is reliable across scientific layouts, so ArticleMiner treats their outputs as evidence to be validated.

Semantic lifting and ontology-guided KG construction. Mapping languages (R2RML, RML), OBDA, and SHACL lift and validate RDF from structured sources, and KG construction needs stable identifiers, datatypes, and provenance [19–25]. ArticleMiner sits one stage earlier, recovering units, null values, identifiers, and context before a structured source exists and exposing structured records for downstream RDF mappings.

## 6<sub>.</sub> Di<sub>scuss</sub>i<sub>on</sub> <sub>an</sub>d Th<sub>rea</sub>t<sub>s</sub> t<sub>o</sub> V<sub>a</sub>lidit<sub>y</sub>

The results support three conclusions within the evaluated tasks. For RQ1, the component removals show that explicit domain guidance contributes to extraction quality, with task- and backbone-dependent gains. For RQ2, canonicalization and task-defined identity make independently recovered candidates comparable, while source locators and backend support retain evidence for inspection. The backend ablations show that adding extractors does not consistently improve accuracy. For RQ3, the raw-PDF/preparsed gaps and omission-heavy settings identify acquisition as an important remaining limitation, whereas over-extraction on the smaller benchmarks also requires better candidate selection. These findings favor task-specific acquisition and correction policies over a universal configuration.

Adaptation and maintenance. A new task requires vocabulary and mappings, derivation and validation code, prompts, a row schema, identifiers, and RDF bindings; geochemistry also requires a deposit classifier. The four modules demonstrate reuse of the orchestration, not zero-shot transfer or a measured reduction in authoring time. We did not record person-hours. Maintenance requires domain expertise to distinguish vocabulary gaps from extraction errors, assess source-specific conventions, and revise artifacts when the schema changes. Logged rejections help prioritize inspection but do not automatically diagnose every cause. Adaptation consequently involves both domain specification and assessment of representative publications. The authoring inventory makes the required artifacts explicit: a vocabulary entry addresses a naming gap, a derivation encodes a declared relationship, and a validator restricts admissible outputs. None can substitute for evidence absent from the recovered document. The four worked examples illustrate this division across tasks without implying that module size measures the dificulty ofa new domain.

Implications for publication-to-KG systems. Downstream reuse requires units, censoring status, identity, and evidence locators alongside canonical labels. Identity deserves particular attention during adaptation: sample name alone can merge distinct analytical methods, while overly specific keys retain duplicates. Defining the target observation before expanding its vocabulary is therefore a practical design recommendation, although its efect on authoring efort remains unevaluated.

Construct validity. Strict tuple- $F _ { 1 }$ , sample-�<sub>1</sub>, and the GeoChem composite score measure diferent properties. GeoChem’s numerical and null tiers use matched rows; position-based matching can exploit numerical overlap, and missing gold metadata receives full credit. These choices can raise scores without improving end-to-end recall. The weights can change the ranking ofnearby variants; per-tier scores are therefore necessary. No direct measure of downstream KG query quality is reported.

Internal validity. Full-system comparisons alter multiple stages, and GeoChem additionally difers in supplementary-file access and identifier handling. They do not isolate the efect of ontology grounding. Post-hoc configurations, difering ground-truth snapshots, backbone-specific token limits, and unavailable MinerU OCR weights further restrict causal interpretation. Component ablations should be interpreted within their documented run conditions.

External validity. The 163 publications represent four manually authored tasks and are dominated by DiSCoMaT. ChemTables and MLTables contain only 9 and 15 papers, respectively. GeoChem relies heavily on machine-readable supplements, and the system constructs per-publication records rather than resolving entities across publications. These conditions limit claims about unfamiliar scientific fields.

Conclusion validity. The paired summaries resample publications rather than individual tuples. Small samples yield imprecise estimates. The exact historical runs underlying the paired summaries are not fully identified by the release, so those summaries are descriptive and do not establish formal statistical significance or absence of an efect. Repeated-run variability was not characterized suficiently to interpret very small diferences. API drift and incomplete version pinning can hinder reproduction [26]. Possible training-data overlap remains unaudited; sharing a backbone does not eliminate contamination or its interaction with prompting [27].

## 7<sub>.</sub> C<sub>onc</sub>l<sub>us</sub>i<sub>on</sub>

ArticleMiner separates publication-level extraction from explicit task-specific semantic artifacts. Across four authored tasks, the reported raw-publication results favor the complete pipeline over same-LLM few-shot prompting, while component and parser analyses show substantial task dependence. The framework’s contribution is a reusable interface for acquiring, interpreting, and reconciling evidence under bounded domain constraints. General adaptation cost, controlled supplementary-aware baselines, and downstream graph quality remain open evaluation needs. Future work includes cross-paper entity resolution, adaptive evidence acquisition, and expert-reviewed module expansion from logged failures.

Supp<sup>l</sup>ementa<sup>l</sup> Materia<sup>l</sup> Statement: Source code, ontology modules, prompts, output schemas, outputs for all eight backbones, ablation grids, and author-contributed GeoChem ground truth are available under CC-BY-4.0 at https://github.com/Abrar2652/articleminer-iswc26. Few-shot outputs cover Sonnet 4.6 and open backbones; other closed-backbone results require regeneration using the released runner. Appendix I provides the full manifest.

## D<sub>ec</sub>l<sub>ara</sub>ti<sub>on o</sub>f <sub>use o</sub>f G<sub>enera</sub>ti<sub>ve</sub> AI

The authors used generative AI in two roles. As research instrumentation, large language models (Claude, GPT-4o, Gemini, and open 7–8B models) are the object of study and part of the ArticleMiner pipeline, with identifiers and configurations reported in Sections 2–4 and the appendices. As manuscriptassistance, a generative AI tool was used for grammar, spelling, rewording, and readability; the authors reviewed and edited all such text and take full responsibility for the content.

## A<sub>c</sub>k<sub>now</sub>l<sub>e</sub>d<sub>gmen</sub>t<sub>s</sub>

This work was supported in part by the Defense Advanced Research Projects Agency (DARPA) under Contract No. 140D0426C0018. Any opinions, findings, and conclusions or recommendations expressed in this paper are those of the author(s) and do not necessarily reflect the views of DARPA or its Contracting Agent, the U.S. Department of the Interior, and no oficial endorsement should be inferred. The work of the first author was supported by the Viterbi Graduate School Fellowship from the USC Viterbi School of Engineering.

## R<sub>e</sub>f<sub>erences</sub>

[1] Viet-Phi Huynh, Jixiong Liu, Yoan Chabot, Frédéric Deuzé, Thomas Labbé, Pierre Monnin, and Raphaël Troncy. DAGOBAH: Table and graph contexts for eficient semantic annotation of tabular data. In Proceedings of the Semantic Web Challenge on Tabular Data to Knowledge Graph Matching (SemTab 2021) co-located with the 20th International Semantic Web Conference (ISWC 2021), volume 3103 of CEUR Workshop Proceedings, pages 19–31. CEUR-WS.org, 2021. URL https://ceur-ws.org/ Vol-3103/paper2.pdf.

[2] Binh Vu and Craig A. Knoblock. SAND: A tool for creating semantic descriptions of tabular sources. In The Semantic Web: ESWC 2022 Satellite Events, volume 13384 of Lecture Notes in Computer Science. Springer, 2022. doi: 10.1007/978-3-031-11609-4\_12.

[3] Dominique Ritze, Oliver Lehmberg, and Christian Bizer. Matching HTML tables to DBpedia. In Proceedings of the 5th International Conference on Web Intelligence, Mining and Semantics (WIMS 2015), pages 10:1–10:6. ACM, 2015. doi: 10.1145/2797115.2797118.

[4] Fan Bai, Junmo Kang, Gabriel Stanovsky, Dayne Freitag, Mark Dredze, and Alan Ritter. Schemadriven information extraction from heterogeneous tables. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 10252–10273, Miami, Florida, USA, 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-emnlp.600. URL https://aclanthology. org/2024.findings-emnlp.600/.

[5] Matthew C. Swain and Jacqueline M. Cole. ChemDataExtractor: A toolkit for automated extraction ofchemical information from the scientific literature. JournalofChemicalInformation andModeling, 56(10):1894–1904, 2016. doi: 10.1021/acs.jcim.6b00207.

[6] Tanishq Gupta, Mohd Zaki, Devanshi Khatsuriya, Kausik Hira, N. M. Anoop Krishnan, and Mausam. DiSCoMaT: Distantly supervised composition extraction from tables in materials science articles. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13465–13483, Toronto, Canada, 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.753. URL https://aclanthology.org/2023.acl-long.753/.

[7] Craig A. Knoblock, Binh Vu, Basel Shbita, Yao-Yi Chiang, Pothula Punith Krishna, Xiao Lin, Goran Muric,Jiyoon Pyo, Adriana Trejo-Sheu, and Meng Ye. Exploiting LLMs and semantic technologies to build a knowledge graph ofhistorical mining data. In The SemanticWeb – ISWC2025, Proceedings, Part II, volume 16141 of Lecture Notes in Computer Science, pages 451–471. Springer, 2025. doi: 10.1007/978-3-032-09530-5\_26.

[8] Albert H. Hofstra, Vladimir Lisitsin, Louise Corriveau, Suzanne Paradis, Jan M. Peter, Kathleen Lauzière, Christopher J. M. Lawley, Michael Gadd, Jean-Luc Pilote, Ian Honsberger, Evgeniy Bas-

trakov, David C. Champion, Karol Czarnota, Michael P. Doublier, David L. Huston, Oliver Raymond, Simon VanDerWielen, Poul Emsbo, Matthew Granitto, and Douglas C. Kreiner. Deposit classification scheme for the critical minerals mapping initiative global geochemical database. USGS Open-File Report 2021–1049, U.S. Geological Survey, 2021. URL https://pubs.usgs.gov/of/2021/ 1049/ofr20211049.pdf.

[9] Ernesto Jiménez-Ruiz, Oktie Hassanzadeh, Vasilis Efthymiou, Jiaoyan Chen, and Kavitha Srinivas. SemTab 2019: Resources to benchmark tabular data to knowledge graph matching systems. In The Semantic Web – ESWC 2020, Proceedings, volume 12123 of Lecture Notes in Computer Science, pages 514–530. Springer, 2020. doi: 10.1007/978-3-030-49461-2\_30. Canonical SemTab challenge resource paper; published in ESWC 2020 LNCS proceedings (not the Semantic Web Journal).

[10] Binh Vu, Craig A. Knoblock, and Fandel Lin. A domain-independent approach for semantic table interpretation. In The Semantic Web – ISWC 2025, Proceedings, Part I, volume 16140 of Lecture Notes in Computer Science, pages 235–252. Springer, 2025. doi: 10.1007/978-3-032-09527-5\_13.

[11] Aaron Steiner and Christian Bizer. Automatic end-to-end data integration using large language models. arXiv preprint arXiv:2603.10547, 2026. URL https://arxiv.org/abs/2603.10547.

[12] Jonathan Herzig, Paweł Krzysztof Nowak, Thomas Müller, Francesco Piccinno, and Julian Martin Eisenschlos. TaPas: Weakly supervised table parsing via pre-training. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020), pages 4320– 4333. Association for Computational Linguistics, 2020. doi: 10.18653/v1/2020.acl-main.398. URL https://aclanthology.org/2020.acl-main.398/.

[13] Xiang Deng, Huan Sun, Alyssa Lees, You Wu, and Cong Yu. TURL: Table understanding through representation learning. Proceedings ofthe VLDB Endowment, 14(3):307–319, 2020. doi: 10.14778/ 3430915.3430921.

[14] Haoyu Dong, Zhoujun Cheng, Xinyi He, Mengyu Zhou, Anda Zhou, Fan Zhou, Ao Liu, Shi Han, and Dongmei Zhang. Table pre-training: A survey on model architectures, pre-training objectives, and downstream tasks. In Proceedings ofthe Thirty-First International Joint Conference on Artificial Intelligence (IJCAI 2022), pages 5426–5435. IJCAI Organization, 2022. doi: 10.24963/ijcai.2022/761. URL https://www.ijcai.org/proceedings/2022/761.

[15] Avanika Narayan, Ines Chami, Laurel Orr, Simran Arora, and Christopher Ré. Can foundation models wrangle your data? Proceedings ofthe VLDB Endowment, 16(4):738–746, 2022. doi: 10. 14778/3574245.3574258.

[16] Christoph Auer, Maksym Lysak, Ahmed Nassar, Michele Dolfi, Nikolaos Livathinos, Panos Vagenas, Cesar Berrospi Ramis, Matteo Omenetti, Fabian Lindlbauer, Kasper Dinkla, Lokesh Morin, and Peter W. J. Staar. Docling technical report, 2024. URL https://arxiv.org/abs/2408.09869.

[17] Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, Bo Zhang, Liqun Wei, Zhihao Sui, Wei Zhou, Botian Xiang, Runyu Wei, Renqiu Li, Xinyue Su, Kaïkang Zhang, Guangsan Yan, Zhen Zhang, Yuantao Sun, Zichao Liu, He Peng, Haodong Wu, Haohui Liu, Conghui Yan, and Conghui He. MinerU: An open-source solution for precise document content extraction, 2024. URL https://arxiv.org/abs/2409.18839.

[18] Lukas Blecher, Guillem Cucurull, Thomas Scialom, and Robert Stojnic. Nougat: Neural optical understanding for academic documents, 2023. URL https://arxiv.org/abs/2308.13418.

[19] Souripriya Das, Seema Sundara, and Richard Cyganiak. R2RML: RDB to RDF mapping language. W3C Recommendation, 2012. URL https://www.w3.org/TR/r2rml/.

[20] Anastasia Dimou, Miel Vander Sande, Pieter Colpaert, Ruben Verborgh, Erik Mannens, and Rik Van de Walle. RML: A generic language for integrated RDF mappings of heterogeneous data. In Proceedingsofthe7th Workshopon LinkedData on theWeb, 2014. URL https://ceur-ws.org/Vol-1184/ ldow2014\_paper\_01.pdf.

[21] Guohui Xiao, Diego Calvanese, Roman Kontchakov, Domenico Lembo, Antonella Poggi, Riccardo Rosati, and Michael Zakharyaschev. Ontology-based data access: A survey. In Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence, IJCAI-18, pages 5511–5519. International Joint Conferences on Artificial Intelligence Organization, 7 2018. doi: 10.24963/ijcai. 2018/777. URL https://doi.org/10.24963/ijcai.2018/777.

[22] Holger Knublauch and Dimitris Kontokostas. Shapes Constraint Language (SHACL). W3C Recommendation, 2017. URL https://www.w3.org/TR/shacl/.

[23] Aidan Hogan, Eva Blomqvist, Michael Cochez, Claudia d’Amato, Gerard de Melo, Claudio Gutierrez, Sabrina Kirrane, José Emilio Labra Gayo, Roberto Navigli, Sebastian Neumaier, Axel-Cyrille Ngonga Ngomo, Axel Polleres, Sabbir M. Rashid, Anisa Rula, Lukas Schmelzeisen, Juan F. Sequeda, Stefen Staab, and Antoine Zimmermann. Knowledge graphs. ACM Computing Surveys, 54(4): 71:1–71:37, 2021. doi: 10.1145/3447772.

[24] Natalya F. Noy, Yuqing Gao, Anshu Jain, Anant Narayanan, Alan Patterson, and Jamie Taylor. Industry-scale knowledge graphs: Lessons and challenges. Communications of the ACM, 62(8): 36–43, 2019. doi: 10.1145/3331166.

[25] Mark D. Wilkinson, Michel Dumontier, IJsbrandJ.J. Aalbersberg, Gabrielle Appleton, Myles Axton, Arie Baak, Niklas Blomberg,Jan-Willem Boiten, Luiz Bonino da Silva Santos, Philip E. Bourne, et al. The FAIR guiding principles for scientific data management and stewardship. Scientific Data, 3: 160018, 2016. doi: 10.1038/sdata.2016.18.

[26] June Sallou, Thomas Durieux, and Annibale Panichella. Breaking the silence: the threats of using LLMs in software engineering. In Proceedings ofthe 2024 ACM/IEEE 46th International Conference on Software Engineering: New Ideas and Emerging Results (ICSE-NIER), pages 102–106, 2024. doi: 10.1145/3639476.3639764.

[27] Shahriar Golchin and Mihai Surdeanu. Time travel in LLMs: Tracing data contamination in large language models. In The Twelfth International Conference on Learning Representations (ICLR 2024), 2024. URL https://openreview.net/forum?id=2Rwq6c3tvr.

[28] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegrefe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-Refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems 36(NeurIPS 2023), 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/hash/ 91edf07232fb1b55a505a9e9f6c0f3-Abstract-Conference.html.

## A<sub>.</sub> The Three Inte<sub>g</sub>ration Points of the Ontolo<sub>gy</sub> Module

The pipeline consults $\Omega _ { d }$ (Figure 2) at three points, linking evidence to the declared task representation: prompt-time grounding at extraction, canonicalization and validation at reconciliation, and IRI binding at serialization (Figure 1).

Prompt-time grounding (integration point ○1 ): $V _ { d }$ enters the LLM’s system prompt at extraction time. Admissible classes, canonical unit strings, and admissible null kinds are rendered into the prompt as an enumeration; the LLM is instructed to emit rows drawn only from those enumerations. The efect is to restrict the LLM’s hypothesis space so that surface variations (e.g., ‘micromolar’ vs $^ \star \mathtt { p m } ^ { \mathrm { , } } )$ map to canonical forms before $C _ { d }$ ever runs.

Canonicalization and per-tuple validation (integration point ○2 ): $\Phi _ { d } , C _ { d } .$ and $\Gamma _ { d }$ run during reconciliation in the order specified by $\operatorname { E q . } \left( 2 \right)$ . Every canonicalized candidate is evaluated against the executable predicates in $\Gamma _ { d } ;$ those that fail are excluded from the trusted graph with a structured rejection reason $( \mathrm { e . g . }$ , unit-not-admissibl $\displaystyle . \mathrm { e } ,$ sum-not-100, summary-row). Rejection reasons and evidence locators are logged so an auditor can trace why a record did not appear in $R ( P , d )$

IRIand @context declaration (integration point ○3 ): the RDF binding specifies how output fields are named in the graph. The released JSON-LD exporter declares a vocabulary and emits sample nodes with measurements and source metadata. This does not establish that every generated identifier resolves to an externally maintained ontology term, or that a native equivalent Turtle export has been evaluated (Appendix J).

The three points are functionally distinct: (○1 ) guides what the LLM proposes; (○2 ) canonicalizes, derives, and rejects records that fail structural constraints; (○3 ) fixes the RDF identity of what remains. These integration points have diferent intended roles: prompt grounding guides proposed fields; record checks enforce declared constraints; and RDF bindings define the graph vocabulary. These roles do not constitute separate empirical ablations of each integration point.

## B<sub>.</sub> Cross-Domain Worked Exam<sub>p</sub>le Analo<sub>g</sub>s

The geochemistry worked example in §2.4 traces one row (BR-12 with Au<0.5 and method ICP-MS) through Parse, Extract, Reconcile, and Serialize. The following schematic examples expand the crossdomain analogs in the main text; they are not additional evaluated outputs or verbatim serializer dumps.

Materia<sup>l</sup>s science (DiSCoMaT). A glass-composition row reports S $\mathrm { i 0 _ { 2 } } \colon 5 5 , \mathrm { N a _ { 2 } O } \colon 2 5 , \mathrm { C a O } \colon 2 0$ in a table whose caption declares the unit as mol%. Parse recovers the row and its header via docling. Extract emits the provisional record �˜ = (material, $\left[ ( \mathrm { S i O _ { 2 } , 5 5 ) , ( N a _ { 2 } O , 2 5 ) , ( C a O , 2 0 ) } \right]$ ,unit=mol%). Reconcile: $\Phi _ { d }$ is a no-op here (no taxonomic derivation on the constituents), $C _ { d }$ passes the oxide names through (already canonical), and $\Gamma _ { d }$ accepts because (i) all constituents are in the oxide vocabulary, (ii) the unit is admissible, and $( \mathrm { i i i } ) \sum _ { i } p _ { i } = 1 0 0$ within tolerance. Seria<sup>l</sup>ize must retain each constituent, percentage, and declared unit. A row whose caption instead declares wt% represents a diferent composition and cannot be substituted for the mol% row in a downstream query. This example specifies the information to preserve without assuming an external oxide or unit IRI mapping.

Drug-discovery c<sup>h</sup>emistry (ChemTables). A bioactivity table reports IC50 >10 for compound 7a against HER2, with a header declaring $\mu \mathrm { M }$ . The extracted record must preserve the treatment, target, assay, unit, and inequality. Here >10 is a lower bound on the concentration, not a measured value of 10 or an upper bound. A downstream query for compounds with $\mathrm { I C } _ { 5 0 } \le 5 \mu \mathrm { M }$ must exclude this observation. The evaluated prompt retains the literal inequality; an RDF mapping may encode its comparator and threshold explicitly.

Mac<sup>h</sup>ine <sup>l</sup>earning (MLTables). An arXiv table reports Ours: 87.3 under Accuracy (%), with CIFAR-10 identified in the caption. Extraction links the value to its model, dataset, metric, and unit using table and prose evidence. Resolving Ours requires source context at extraction time; it is not a lookup that a fixed canonicalizer can perform without that evidence. Subsequent normalization and validation enforce the declared task schema.

The examples distinguish contextual interpretation by the extraction model from deterministic operations specified by the module. Across tasks, semantic fields and schemas change while the high-level orchestration is reused.

## C<sub>.</sub> Out<sub>p</sub>ut Schemas

The ArticleMiner GeoChem ground-truth schema contains 209 columns: 63 metadata/provenance fields and 146 element columns (73 elements × {value, detection-limit/null information}). The pipeline’s output Excel allows extra optional fields (uncertainty, method, alternative units), but evaluation pulls only the named-field projection (§3), so additional pipeline-emitted columns are ignored rather than penalized, and pipeline and baseline are compared on the same 209-column projection. The other three benchmarks each define a tuple shape: ChemTables uses ⟨cell\_index, value, type, target, treatment, unit⟩ with $t y p e \in \{ \mathrm { I C } _ { 5 0 } , \mathrm { E C } _ { 5 0 } , \mathrm { G I } _ { 5 0 } , \mathrm { C C } _ { 5 0 } , \mathrm { M I C } \}$ ; DiSCoMaT uses ⟨sample\_id, component, percentage, unit⟩ with the unit in {mol%, wt%}; MLTables uses an entry-type-tagged tuple in {Result,Data Stat.,Hyper-parameter,Other} with type-dependent attribute fields $( \mathrm { e . g . }$ Result carries model, dataset, metric, task).

## C<sub>.</sub>1<sub>.</sub> G<sub>eo</sub>Ch<sub>em scor</sub>i<sub>ng</sub> d<sub>e</sub>t<sub>a</sub>il<sub>s</sub>

For $T _ { 1 }$ , string comparison returns the maximum oftokenJaccard similarity and sequence similarity after normalization. Missing gold values score 1; a missing prediction for a present gold value scores 0. For a nonzero numerical gold value $^ { g , }$ let $e = | p - g | / | g |$ . The released $T _ { 2 }$ score is 1 for $e \leq 0 . 0 5 , 0$ for $e \geq 1$ and $1 - ( e - 0 . 0 5 ) / 0 . 9 5$ otherwise. A zero gold value receives full credit only within absolute error 0.001. When the gold is the below-detection sentinel, a matching sentinel scores 1, a missing prediction 0.5, and another value 0.3. A predicted sentinel against an ordinary gold value scores $0 . { \ : } T _ { 2 }$ averages scored fields within a matched row and then averages matched rows; $T _ { 4 }$ similarly averages null agreement over gold-null fields.

Identifier matching tests available sample-name columns and accepts exact or prefix matches. Iffewer than 30% ofgold rows match by name, a fallback compares row order under ofsets and numerical overlap. Rows are compatible when at least 30% of shared positive elemental values agree within 10%; for much larger prediction sets, a greedy value-fingerprint match is used. The fallback therefore uses measurement values to establish alignment, which must be considered when interpreting numerical accuracy. $T _ { 3 }$ is the resulting sample- ${ \bf \nabla } \cdot F _ { 1 }$ , while row counts are also reported diagnostically. These rules describe the released evaluator; they do not make its score equivalent to strict tuple matching.

## D<sub>.</sub> P<sub>er-</sub>D<sub>oma</sub>i<sub>n</sub> E<sub>x</sub>t<sub>erna</sub>l<sub>-</sub>B<sub>enc</sub>h<sub>mar</sub>k D<sub>e</sub>t<sub>a</sub>il

DiSCoMaT. On raw PDFs, Opus 4.6 reaches 81.0 tuple-�<sub>1</sub> and Sonnet 4.6 reaches 79.3, compared with same-backbone few-shot scores of 59.5 and 64.5. Sonnet’s 98.2% precision and 66.4% recall indicate that omissions dominate the remaining error. The reported per-paper analysis includes 32 of 111 papers with no predictions; these failures remain in the main denominator. Published GNN results are external context and do not share the paired analysis in Table 4.

MLTab<sup>l</sup>es. Opus 4.6 reaches 55.9 tuple-�<sub>1</sub> against its few-shot baseline’s 41.6. The Sonnet paired means in Table 4(b) are a separate comparison: 52.2 versus 37.8, with a reported paired diference of14.4. Open-model scores range from 28.6 to 51.1. These observations are limited by the 15-paper sample.

C<sup>h</sup>emTab<sup>l</sup>es. The full Sonnet configuration reaches 32.1 tuple- $F _ { 1 }$ against 15.8 for few-shot prompting; disabling paper intelligence raises the system score to 44.6. The latter is a post-hoc configuration.

The 9-paper paired summary has wide intervals. The configuration underlying the historical paired summary is not fully documented; that summary should not be treated as a confirmed paired test ofthe full pipeline.

GeoC<sup>h</sup>em. Closed-backbone composite scores range from 75.5 to 76.3 on 28 entries. The separate paired analysis uses 25 common entries and reports a mean sample- ${ \bf \nabla } \cdot F _ { 1 }$ of 63.2 for Sonnet. These values describe diferent estimands and denominators. Supplementary-file access and sample-identifier handling contribute to the gap from PDF-only few-shot prompting.

## E<sub>.</sub> Ontolo<sub>gy</sub> Module Entries

This appendix gives the full per-task semantic content behind the compact summary in Table 1. Representative entries from the four domain-specific ArticleMiner ontology modules follow; full modules are released with the source code.

## E.1. Geochemistr<sub>y</sub> (220 entries)

The geochemistry module combines deposit keyword mappings, mineral labels, analytical-method aliases, and value conventions. Approximately 90 keyword entries map deposit descriptions into the adopted hierarchy; these mappings are distinct from the complete 189-type classifier vocabulary. Around 80 mineral entries associate labels with coarse classes and formulae, for example sphalerite with sulfide and $Z \mathrm { n } S ,$ or magnetite with oxide and $\mathrm { F e _ { 3 } O _ { 4 } }$ . More than 50 method variants map to approximately 12 canonical methods, including LA-ICPMS, EPMA, SIMS, ICP-OES, and XRF. Value rules distinguish detection-limit encodings from missing values and preserve explicit unit classes. These counts describe diferent artifact categories and should not be interpreted as the number of axioms in a formal ontology.

## E.2. Dru<sub>g</sub> discover<sub>y</sub>: ChemTables (80 entries)

The bioassay vocabulary includes five activity types $\{ \mathrm { I C _ { 5 0 } , E C _ { 5 0 } , G I _ { 5 0 } , M I C , C C _ { 5 0 } } \}$ together with their canonical units and assay interpretations. The target taxonomy lists representative protein kinases (CHK1, CHK2, EGFR, ...), receptors, and microbial targets seen across the 9 PMC papers, with synonym maps where applicable. The module includes assay-specific value ranges and compound-target linkage constraints; these do not establish monotonicity of an unobserved dose-response curve.

## E.3. Materials science: DiSCoMaT (167 entries)

The material-classification taxonomy distinguishes glass, ceramic, alloy, polymer, plus a long tail ofsubclasses from the SciGlass database. The oxide-component taxonomy lists the standard compositional building blocks $( \mathrm { S i O _ { 2 } , N a _ { 2 } O , B _ { 2 } O _ { 3 } , P b O , C a O , . . . } )$ with their canonical chemical formulae. Unit standardization distinguishes mol% vs. wt% (both valid; the distinction must be preserved per table). Constraints encode composition-sum tolerance $( \sum \approx 1 0 0 \% )$ and component plausibility ranges per material class.

## E.4. Machine learnin<sub>g</sub>: MLTables (60 entries)

The entry-type taxonomy enumerates the four MLTables annotation classes {Result, Data Statistic, Hyper-parameter/Architecture, Other}. The metric taxonomy lists common ML metrics (accuracy, $\mathrm { F _ { 1 } }$ BLEU, perplexity, AUC, $\mathrm { m A P , \ldots ) }$ with their value ranges and interpretation (higher-better vs. lowerbetter). Attribute linking defines the model→dataset→metric→value chain that identifies a Result and the parameter→model association that identifies a Hyper-parameter. Constraints encode metric plausibility (accuracy $\in [ 0 , 1 0 0 ]$ , loss ≥ 0) and model-dataset consistency within a paper.

## F. Prom<sub>p</sub>t Tem<sub>p</sub>lates

This appendix documents the prompt templates used in each LLM-based stage of ArticleMiner; full templates are released with the source code. Listings 1–4 reproduce the extraction system prompt ofeach task module verbatim from the released code, so the per-domain diferences are directly comparable: the prompts share a high-level structure (role, target output, domain rules, and prohibitions on fabrication), while the vocabulary, tuple shape, unit conventions, and null-value semantics come from $\Omega _ { d } .$ . Note that the dash convention is deliberately opposite across domains: in DiSCoMaT composition tables, a - cell means the component is absent (0%), whereas the GeoChem task default treats it as below detection; source legends may require a diferent interpretation (§2.2, Part 3). The same surface symbol carries diferent semantics per domain, which is why null conventions live in the task module rather than in shared code.

Listin<sub>g</sub> 1: Exact DiSCoMaT extraction s<sub>y</sub>stem <sub>p</sub>rom<sub>p</sub>t   
You are an expert materials scientist specializing in glass and ceramic compositions. Your   
task is to extract material composition data from tables in materials science papers.   
## Extraction Target   
For each composition table, extract tuples of the form:   
{ "sample\_id": "<identifier, from row header or sample number>",   
"component": "<chemical formula, e.g., SiO2, Na2O, PbO>",   
"value": <numerical percentage value>,   
"unit": "<mol or wt>" }   
## Unit Inference Rules   
1. Check the table caption for "mol%", "mole%", "wt%", or "weight%".   
2. Check column headers for unit annotations like "SiO2 (mol%)".   
3. If the caption says "mol%" then ALL values in the table are mol%.   
4. If no unit is found, check if the values sum to ∼100%.   
## Critical Rules   
1. ONLY extract composition data; skip physical properties (density, �<sub>�</sub>, hardness).   
2. Component names must be valid chemical formulas (SiO2, not "silica").   
3. A cell of "-" or "–" means 0% (component absent); do NOT extract it.   
4. A blank cell means "not reported"; do NOT extract it.

## Listin<sub>g</sub> 2: Exact GeoChem raw-PDF table extraction s<sub>y</sub>stem <sub>p</sub>rom<sub>p</sub>t

You are an expert geochemistry database curator. You are given the text of a research paper   
(including any tables embedded in the PDF). There is NO supplementary spreadsheet available.   
Your task is to extract individual sample analytical data (element concentrations) directly   
from tables in the PDF text.   
Extract information EXACTLY as reported — do not paraphrase, estimate, or fabricate values.   
{knowledge\_base}   
The {knowledge\_base}placeholderiswhere�<sub>�</sub> isrenderedintothepromptatruntime(integrationpoint○1 ): theadmissible   
element list, unit classes, mineral and method vocabularies, and the null-kind conventions (BDL, NM, measured zero). The   
companion user template requests a JSON object with samples and extraction\_notes, one object per analytical spot.

## Listin<sub>g</sub> 3: Exact ChemTables extraction s<sub>y</sub>stem <sub>p</sub>rom<sub>p</sub>t

You are an expert medicinal chemist specializing in drug discovery data extraction. Your   
task is to extract bioactivity measurements from tables in medicinal chemistry papers.   
## Extraction Target   
For each numerical cell in the table that represents a bioactivity measurement, produce a   
JSON object:   
{ "cell\_index": "CA(row,col)",   
"value": "<exact numerical value from the cell>",   
"type": "<one of: IC50, EC50, GI50, MIC>",   
"target": "<protein target, cell line, or organism being assayed>",   
"treatment": "<compound identifier (number, name, or code) being tested>",   
"unit": "<measurement unit, e.g., �M, nM, �g/mL>" }

## Assay Type Definitions   
- IC50: Half-maximal inhibitory concentration (enzyme/protein inhibition AND cell   
viability/proliferation assays)   
- EC50: Half-maximal effective concentration (cell-based functional assays)   
- GI50: Concentration causing 50% growth inhibition (ONLY when paper explicitly uses the   
term GI50)   
- MIC: Minimum inhibitory concentration (antimicrobial assays)   
IMPORTANT: If the table caption or column header explicitly states "IC50", classify as IC50   
even if the assay measures cell proliferation. The paper’s own terminology takes precedence   
over domain conventions. Only use GI50 if the paper explicitly labels values as GI50.   
## Critical Rules   
1. Extract ALL cells containing numerical bioactivity/potency values. This includes IC50,   
EC50, GI50, MIC, and related measurements like % inhibition (which should be classified as   
MIC if units are �g/mL or �M) or cytotoxicity values.   
2. The "treatment" is the compound being tested — usually identified by a number in the   
first column or a compound name.   
3. The "target" is what the compound is tested against — a protein (CHK1, EGFR), cell line   
(A549, HeLa), or organism (S. aureus, E. coli). If the target is not clear, use "xx".   
4. Infer type, target, and unit from column headers. Column headers often follow patterns   
like "CHK1 IC50 (�M)" or "MIC (�g/mL) S. aureus".   
5. If a column header says "% inhibition" or "inhibition (%)" with units like �g/mL,   
classify as MIC.   
6. If a cell contains a range like "1.0 (0.86, 1.2)", extract the primary value "1.0" AND   
include the full string with ± if present (e.g., "265 ± 17" should have value "265 ± 17").   
7. If a cell contains ">100" or ">50", this means the compound is inactive at the tested   
range. Still extract it with value ">100" or ">50".   
8. Do NOT hallucinate values. Only extract what is explicitly present in the table.   
9. Skip truly non-bioactivity data like selectivity ratios, ligand efficiency, or physical   
properties (LogP, MW).   
10. Output one JSON object per line. No markdown, no explanation.

## Listing 4: Exact MLTables extraction s<sub>y</sub>stem <sub>p</sub>rom<sub>p</sub>t

You are an expert ML researcher. Given a table extracted from a machine learning paper,   
extract ALL quantitative entries (results, dataset statistics, hyperparameters).   
For each entry, output a JSON object on one line:   
{"value": "<exact number>", "type": "<Result|Data   
Stat.|Hyper-parameter/Architecture|Other>", . . . attributes. . . }   
For type "Result": include "model", "dataset", "metric", "task" if identifiable.   
For type "Data Stat.": include "dataset", "attribute name" (e.g., "number of samples",   
"number of classes").   
For type "Hyper-parameter/Architecture": include "parameter/architecture name", "model".   
RULES:   
1. Extract the EXACT numerical value from the table cell.   
2. Classify each value into one of the four types based on context.   
3. For Results: the model is usually in the row, the metric in the column header, the   
dataset from caption or headers.   
4. Skip non-quantitative cells (model names, dataset names used as labels).   
5. Do NOT hallucinate. Only extract what is explicitly in the table.   
6. Output ONLY JSON lines, no explanation.

The verbatim ChemTables prompt enumerates four assay labels although the module also includes CC<sub>50</sub>. Its rule mapping some inhibition measurements to MIC is a task heuristic, not a biochemical equivalence. Likewise, a composition summing to 100 does not distinguish mol% from wt%. These prompt choices are retained to document the evaluated instrumentation and can introduce systematic interpretation errors.

Paper-inte<sup>ll</sup>igence B<sup>l</sup>ueprint s<sup>k</sup>im (p<sup>l</sup>anning stage). The blueprint prompt asks the model to identify analytical methodology without inventing values. It receives PDF text and the task vocabulary, and returns expected sample count, measured elements, minerals, analytical methods, instrument, laboratory, standards, and operating conditions. The expected count guides acquisition retries; the other fields provide extraction context.

Paper-<sup>l</sup>eve<sup>l</sup> metadata LLM (p<sup>l</sup>anning stage). System role: extract paper-level metadata, validate each category against the provided taxonomy, and reject categories absent from it; never invent units or method names. User input: PDF text, the Blueprint object, and the classification taxonomy (e.g. the 189- class CMMI deposit-type list for geochemistry). Output: JSON conforming to the paper-level metadata schema, with per-category confidence.

Agentic diagnostic LLM (structura<sup>l</sup>-<sup>h</sup>int retry). System role: diagnose why parsing produced fewer rows than expected and return structured hints for a retry, enumerating the failure modes to look for (wrong header row, transposed layout, unrecognized sample-ID column, wrong unit assumption, multi-row headers, non-geochemical content). It requests only parsing hints, not values — the iterative refine–feedback pattern of[28] restricted to structural correction. User input: failed-extraction rows, expected sample count, raw table text per backend. Output: JSON hints deserialized into {header\_row, is\_transposed, sample\_id\_col, element\_label\_col, notes\_from\_llm}, with null for fields whose auto-detection was correct; separate supplement-file and raw-PDF variants exist, bounded at two diagnosis attempts per failing file.

Token budgets are set per stage rather than globally (most extraction and diagnosis calls use 8192; short classification and filtering calls use less); temperature is 0.0 throughout for reproducibility.

Open-LLM cap. For 7–8B open backbones (Qwen-2.5, Mistral-v0.2, Llama-3.1) on GeoChem, max\_new\_tokens is capped at 512 across blueprint, metadata, table-extraction, and deposit-classifier stages. The cap suppresses hallucination spirals observed under longer budgets (e.g., open-ended isotopechain enumeration on geochemistry tables) and short-circuits failed JSON-schema completions; the deposit classifier still emits malformed JSON on the 189-class CMMI taxonomy under all three open back bones, which the pipeline catches and writes deposit\_type=null (§4.1, “Open-LLM geochemistry”).

Few-s<sup>h</sup>ot base<sup>l</sup>ine prompt. The PDF few-shot baseline uses one call at temperature zero with three fixed exemplars. Exemplars come from the external benchmarks’ training splits or a separate GeoChem exemplar pool. The user input contains PDF text, or released table text for the pre-parsed setting, without a blueprint, ontology context, vision, or retries. The reported token budgets are 4096 input and 8192 output tokens. The GeoChem baseline emits a compact per-row schema rather than the complete 209-column projection and does not ingest supplementary spreadsheets. Its comparison with ArticleMiner therefore includes input-access and representation diferences. The ChemTables exemplar-count analysis motivates three exemplars for that task; it does not establish context suficiency elsewhere.

## G<sub>.</sub> Sonnet 4<sub>.</sub>6 Sensitivit<sub>y</sub> on PDF-onl<sub>y</sub> Domains

The main-text ablation table (Table 5, all panels) fixes the LLM backbone to Haiku 4.5 for cross-domain uniformity. On the small PDF-only benchmarks ChemTables (�=9) and MLTables (�=15), the magnitudes ofcomponent efects difer between Haiku and Sonnet. Re-running the same ablations on Sonnet 4.6 shows that component magnitudes and, for paper intelligence, the direction of the efect change with the backbone, so the two grids should be read as task- and backbone-specific diagnostics rather than as a single universal ranking.

## H<sub>.</sub> E<sub>x</sub>t<sub>en</sub>d<sub>e</sub>d R<sub>esu</sub>lt<sub>s</sub>

This appendix provides supporting model, prompt-count, and protocol analyses. Table 8 is a historical sequence ofconfigurations, not a controlled one-factor ablation: changes in $T _ { 4 }$ cannot be assigned solely to the deposit classifier. Table 9 describes overlap among model outputs rather than their correctness.

Sonnet 4<sub>.</sub>6 com<sub>p</sub>onent ablation on ChemTables/MLTables<sub>,</sub> com<sub>p</sub>anion to Table 5. Vision is the lar<sub>g</sub>est sin<sub>g</sub>le contributor on PDF-onl<sub>y</sub> domains $( - 1 7 . 5 / - 4 7 . 4 \mathsf { F } _ { 1 }$ when removed); removing paper-intelligence raises ChemTables F<sub>1</sub> by +12.5 via stricter precision. Bold = largest drop / largest gain.
<table><tr><td></td><td colspan="3">ChemTables (n=9)</td><td colspan="3">MLTables (n=15)</td></tr><tr><td>Variant</td><td>P</td><td>R</td><td> $\mathsf { F } _ { 1 }$ </td><td>P</td><td>R</td><td> $\mathsf { F } _ { 1 }$ </td></tr><tr><td>Full pipeline</td><td>30.4</td><td>34.0</td><td>32.1</td><td>52.6</td><td>51.7</td><td>52.2</td></tr><tr><td>w/o ontology</td><td>28.8</td><td>32.3</td><td>30.4</td><td>51.9</td><td>51.6</td><td>51.7</td></tr><tr><td>w/o self-correction</td><td>28.6</td><td>32.3</td><td>30.3</td><td>51.8</td><td>51.6</td><td>51.7</td></tr><tr><td>w/o validation</td><td>28.4</td><td>32.3</td><td>30.2</td><td>51.7</td><td>50.8</td><td>51.2</td></tr><tr><td>w/o intelligence</td><td>64.9</td><td>34.0</td><td>44.6↑</td><td>52.5</td><td>52.1</td><td>52.3</td></tr><tr><td>w/o vision</td><td>17.1</td><td>12.8</td><td>14.6</td><td>28.3</td><td>2.6</td><td>4.8</td></tr></table>

$$
\mathsf { F } _ { 1 }
$$

$$
\mathsf { F } _ { 1 }
$$

so neither <sub>g</sub>rid <sub>g</sub>ives universal com<sub>p</sub>onent efects<sub>;</sub> vision <sub>p</sub>roduces the lar<sub>g</sub>est losses under Sonnet<sub>,</sub> whereas validation removal <sub>p</sub>roduces lar<sub>g</sub>er losses under Haiku<sub>,</sub> and GeoChem’s su<sub>pp</sub>lementar<sub>y</sub>-data <sub>p</sub>ath makes LLM-side com<sub>p</sub>onents far less i<sub>n</sub>fl<sub>uen</sub>ti<sub>a</sub>l <sub>on</sub> th<sub>a</sub>t t<sub>as</sub>k<sub>.</sub>

## T<sub>a</sub>bl<sub>e</sub> 8

Post-hoc protocol-component analysis on GeoChem (All-28, Sonnet 4.6): four-tier scores as protocol components <sub>are a</sub>dd<sub>e</sub>d<sub>.</sub> E<sub>xp</sub>l<sub>ora</sub>t<sub>ory</sub> di<sub>agnos</sub>ti<sub>c, no</sub>t i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>con</sub>fi<sub>rma</sub>ti<sub>on.</sub> B<sub>o</sub>ld <sub>mar</sub>k<sub>s</sub> th<sub>e</sub> b<sub>es</sub>t <sub>per co</sub>l<sub>umn.</sub>
<table><tr><td>Configuration</td><td> $T _ { 1 }$ </td><td> $T _ { 2 }$ </td><td> $T _ { 3 }$ </td><td> $T _ { 4 }$ </td><td>Overall S</td></tr><tr><td>Base configuration</td><td>69.9</td><td>70.7</td><td>67.8</td><td>88.4</td><td>72.7</td></tr><tr><td>+ USGS protocol</td><td>71.5</td><td>70.9</td><td>70.8</td><td>88.4</td><td>73.7</td></tr><tr><td>+ LLM deposit classifier (3-backend)</td><td>71.2</td><td>74.0</td><td>68.5</td><td>97.6</td><td>75.9</td></tr><tr><td>+ full 5-backend (ARTICLEMINER)</td><td>71.6</td><td>76.5</td><td>68.5</td><td>90.6</td><td>76.0</td></tr></table>

## T<sub>a</sub>bl<sub>e</sub> 9

Pairwise tuple Jaccard (%) between backbones and their three-way intersection; higher = more agreement.
<table><tr><td>Backbone pair</td><td>CHEMTABLES</td><td>MLTABLES</td><td>DISCoMAT</td><td>GEOCHEM</td></tr><tr><td>Sonnet 4.6 / Opus 4.6</td><td>99</td><td>85</td><td>79</td><td>90</td></tr><tr><td>Sonnet  $4 . 6 / \mathrm { G P T } – 4 0$ </td><td>72</td><td>57</td><td>74</td><td>79</td></tr><tr><td>Opus 4.6 / GPT-40</td><td>72</td><td>60</td><td>64</td><td>82</td></tr><tr><td>Three-way</td><td>58</td><td>53</td><td>52</td><td>75</td></tr></table>

Figure 3 shows a 0.8-point spread in GeoChem’s composite score across closed backbones. This stability is consistent with the shared deterministic spreadsheet path, but does not imply that every metadata field or individual prediction is invariant to the backbone.

![](images/93edb2bb24fd5de6ae667b5ae5f48f1c17c139d16dfce87c728c621e92be43c6.jpg)  
Fi<sub>g</sub>ure 3: Per-backbone score across the four tasks (corresponding to Table 4): strict tuple- $F _ { 1 }$ (%) for ChemTables/MLTables/DiSCoMaT and four-tier Overall � (%) for GeoChem. The two metrics are distinct constructs and <sub>are no</sub>t di<sub>rec</sub>tl<sub>y compara</sub>bl<sub>e across</sub> t<sub>as</sub>k<sub>s.</sub>

## H<sub>.</sub>1<sub>.</sub> F<sub>u</sub>ll <sub>cross-mo</sub>d<sub>e</sub>l <sub>gr</sub>id

Tables 10–13 reports the complete grid of precision, recall, and $F _ { 1 }$ for all eight backbones under both input modalities; the main text retains only the matched closed-model comparison (Table 4) and its uncertainty (Table 4).

T<sub>a</sub>bl<sub>e</sub> 11  
Ch<sub>em</sub>T<sub>a</sub>bl<sub>es</sub> $\mathsf { P } / \mathsf { R } / F _ { 1 }$ (%). � = 9 publications; raw-PDF and pre-parsed inputs. ArticleMiner is abbreviated AM.
<table><tr><td colspan="2"></td><td colspan="3">Raw PDF</td><td colspan="3">Pre-parsed</td></tr><tr><td>Backbone</td><td>System</td><td>P</td><td>R</td><td> $F _ { 1 }$ </td><td>P</td><td>R</td><td> $F _ { 1 }$ </td></tr><tr><td>Sonnet 4.6</td><td>AM</td><td>30.4</td><td>34.0</td><td>32.1</td><td>57.8</td><td>60.2</td><td>59.0</td></tr><tr><td rowspan="2">Opus 4.6</td><td>Few-shot</td><td>41.7</td><td>9.7</td><td>15.8</td><td>48.3</td><td>39.2</td><td>43.2</td></tr><tr><td>AM</td><td>30.6</td><td>33.1</td><td>31.8</td><td>59.9</td><td>61.5</td><td>60.7</td></tr><tr><td rowspan="2">Haiku 4.5</td><td>Few-shot</td><td>43.7</td><td>9.7</td><td>15.9</td><td>50.9</td><td>50.9</td><td>50.9</td></tr><tr><td>AM</td><td>19.3</td><td>39.4</td><td>25.9</td><td>55.0</td><td>57.1</td><td>56.1</td></tr><tr><td rowspan="2">GPT-40</td><td>Few-shot</td><td>42.1</td><td>9.7</td><td>15.8</td><td>48.4</td><td>45.9</td><td>47.1</td></tr><tr><td>AM</td><td>28.2</td><td>32.5</td><td>30.2</td><td>55.0</td><td>52.4</td><td>53.7</td></tr><tr><td rowspan="2">Gemini 2.5 Flash</td><td>Few-shot</td><td>62.0</td><td>9.5</td><td>16.5</td><td>45.4</td><td>43.1</td><td>44.2</td></tr><tr><td>AM</td><td>16.5</td><td>35.9</td><td>22.6</td><td>61.3</td><td>61.3</td><td>61.3</td></tr><tr><td rowspan="2">Qwen-2.5-7B (open)</td><td>Few-shot</td><td>24.0</td><td>10.2</td><td>14.3</td><td>47.6</td><td>44.8</td><td>46.2</td></tr><tr><td>AM</td><td>11.9</td><td>13.2</td><td>12.5</td><td>79.0</td><td>53.9</td><td>64.1</td></tr><tr><td rowspan="2">Mistral-7B (open)</td><td>Few-shot</td><td>37.7</td><td>5.0</td><td>8.8</td><td>44.7</td><td>28.6</td><td>34.9</td></tr><tr><td>AM</td><td>22.3</td><td>17.7</td><td>19.8</td><td>16.9</td><td>19.0</td><td>17.9</td></tr><tr><td rowspan="2">Llama-3.1-8B (open)</td><td>Few-shot</td><td>6.7</td><td>6.5</td><td>6.6</td><td>36.0</td><td>20.8</td><td>26.3</td></tr><tr><td>AM</td><td>18.6</td><td>14.9</td><td>16.6</td><td>49.9</td><td>54.3</td><td>52.0</td></tr><tr><td></td><td>Few-shot</td><td>15.4</td><td>8.7</td><td>11.1</td><td>45.3</td><td>30.5</td><td>36.5</td></tr></table>

MLTables $\mathsf { P } / \mathsf { R } / F _ { 1 }$ (%). �=15 publications; raw-PDF and pre-parsed inputs. ArticleMiner is abbreviated AM.
<table><tr><td colspan="2"></td><td colspan="3">Raw PDF</td><td colspan="3">Pre-parsed</td></tr><tr><td>Backbone</td><td>System</td><td>P</td><td>R</td><td> $F _ { 1 }$ </td><td>P</td><td>R</td><td> $F _ { 1 }$ </td></tr><tr><td>Sonnet 4.6</td><td>AM</td><td>52.6</td><td>51.7</td><td>52.2</td><td>70.9</td><td>94.9</td><td>81.2</td></tr><tr><td rowspan="2">Opus 4.6</td><td>Few-shot</td><td>54.8</td><td>35.6</td><td>43.2</td><td>68.8</td><td>92.1</td><td>78.8</td></tr><tr><td>AM</td><td>57.0</td><td>54.9</td><td>55.9</td><td>72.1</td><td>95.4</td><td>82.1</td></tr><tr><td rowspan="2">Haiku 4.5</td><td>Few-shot</td><td>54.1</td><td>33.8</td><td>41.6</td><td>75.6</td><td>75.9</td><td>75.8</td></tr><tr><td>AM</td><td>45.7</td><td>59.4</td><td>51.6</td><td>70.8</td><td>92.8</td><td>80.3</td></tr><tr><td rowspan="2">GPT-40</td><td>Few-shot</td><td>52.3</td><td>29.9</td><td>38.0</td><td>79.9</td><td>75.4</td><td>77.6</td></tr><tr><td>AM</td><td>55.0</td><td>38.9</td><td>45.6</td><td>78.2</td><td>92.4</td><td>84.7</td></tr><tr><td rowspan="2">Gemini 2.5 Flash</td><td>Few-shot</td><td>41.8</td><td>27.0</td><td>32.8</td><td>80.9</td><td>91.5</td><td>85.9</td></tr><tr><td>AM</td><td>44.7</td><td>40.8</td><td>42.7</td><td>63.9</td><td>92.2</td><td>75.4</td></tr><tr><td rowspan="2">Qwen-2.5-7B (open)</td><td>Few-shot</td><td>44.5</td><td>40.5</td><td>42.4</td><td>75.5</td><td>99.0</td><td>85.7</td></tr><tr><td>AM</td><td>49.2</td><td>51.3</td><td>50.2</td><td>75.3</td><td>79.3</td><td>77.3</td></tr><tr><td>Mistral-7B (open)</td><td>Few-shot</td><td>38.7</td><td>13.5</td><td>20.0</td><td>76.4</td><td>59.7</td><td>67.0</td></tr><tr><td rowspan="2"></td><td>AM</td><td>42.8</td><td>21.5</td><td>28.6</td><td>58.5</td><td>63.7</td><td>61.0</td></tr><tr><td>Few-shot</td><td>36.9</td><td>12.7</td><td>18.9</td><td>53.6</td><td>37.3</td><td>44.0</td></tr><tr><td rowspan="2">Llama-3.1-8B (open)</td><td>AM</td><td>51.9</td><td>50.4</td><td>51.1</td><td>59.9</td><td>80.2</td><td>68.6</td></tr><tr><td>Few-shot</td><td>31.6</td><td>18.3</td><td>23.2</td><td>69.5</td><td>59.9</td><td>64.3</td></tr></table>

## H<sub>.</sub>2<sub>.</sub> GeoChem <sub>p</sub>er-<sub>p</sub>a<sub>p</sub>er 4-tier breakdown $( n = 2 8 )$

Per-paper $T _ { 1 } / T _ { 2 } / T _ { 3 } / T _ { 4 }$ scores for the headline configuration (Fu<sup>ll</sup> (5-bac<sup>k</sup>end, ArticleMiner) from Table 8, Sonnet 4.6) yield the following bootstrap 95% CIs on per-tier means (�=28, 1,000 resamples, seed 42): $T _ { 1 }$ (metadata) 71.6[68.6,74.7], �<sub>2</sub> (numerical) 76.5[65.4,87.2], $T _ { 3 }$ (structural) 68.5 [55.0,80.4], $T _ { 4 }$ (null) 90.6 [85.2,95.1], Overall 76.0 [70.1,81.4]. The $T _ { 2 }$ and $T _ { 3 }$ intervals are wide (±10–13) because

T<sub>a</sub>bl<sub>e</sub> 13  
T<sub>a</sub>bl<sub>e</sub> 12  
DiSCoMaT $\mathsf { P } / \mathsf { R } / F _ { 1 }$ (%). �=111 publications; raw-PDF and pre-parsed inputs. ArticleMiner is abbreviated AM.
<table><tr><td colspan="2"></td><td colspan="3">Raw PDF</td><td colspan="3">Pre-parsed</td></tr><tr><td>Backbone</td><td>System</td><td>P</td><td>R</td><td> $F _ { 1 }$ </td><td>P</td><td>R</td><td> $F _ { 1 }$ </td></tr><tr><td>Sonnet 4.6</td><td>AM</td><td>98.2</td><td>66.4</td><td>79.3</td><td>91.5</td><td>91.1</td><td>91.3</td></tr><tr><td rowspan="2">Opus 4.6</td><td>Few-shot</td><td>82.7</td><td>52.9</td><td>64.5</td><td>92.1</td><td>94.7</td><td>93.4</td></tr><tr><td>AM</td><td>97.0</td><td>69.5</td><td>81.0</td><td>95.1</td><td>94.2</td><td>94.7</td></tr><tr><td rowspan="2">Haiku 4.5</td><td>Few-shot</td><td>80.4</td><td>47.2</td><td>59.5</td><td>91.2</td><td>85.5</td><td>88.3</td></tr><tr><td>AM</td><td>97.0</td><td>57.8</td><td>72.4</td><td>92.8</td><td>91.2</td><td>92.0</td></tr><tr><td rowspan="2">GPT-40</td><td>Few-shot</td><td>73.7</td><td>44.9</td><td>55.8</td><td>92.4</td><td>90.1</td><td>91.2</td></tr><tr><td>AM</td><td>87.3</td><td>47.9</td><td>61.9</td><td>92.4</td><td>88.6</td><td>90.4</td></tr><tr><td rowspan="2">Gemini 2.5 Flash</td><td>Few-shot</td><td>63.0</td><td>44.1</td><td>51.9</td><td>88.2</td><td>90.8</td><td>89.5</td></tr><tr><td>AM</td><td>92.9</td><td>64.1</td><td>75.9</td><td>85.4</td><td>88.7</td><td>87.0</td></tr><tr><td rowspan="2">Qwen-2.5-7B (open)</td><td>Few-shot</td><td>62.5</td><td>44.7</td><td>52.1</td><td>89.3</td><td>91.8</td><td>90.5</td></tr><tr><td>AM</td><td>88.3</td><td>43.0</td><td>57.8</td><td>80.8</td><td>78.5</td><td>79.6</td></tr><tr><td rowspan="2">Mistral-7B (open)</td><td>Few-shot</td><td>33.8</td><td>22.5</td><td>27.0</td><td>73.4</td><td>13.6</td><td>23.0</td></tr><tr><td>AM</td><td>81.1</td><td>38.2</td><td>51.9</td><td>60.2</td><td>54.1</td><td>57.0</td></tr><tr><td rowspan="3">Llama-3.1-8B (open)</td><td>Few-shot</td><td>11.5</td><td>2.8</td><td>4.5</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>AM</td><td>79.0</td><td>10.3</td><td>18.2</td><td>69.3</td><td>73.8</td><td>71.5</td></tr><tr><td>Few-shot</td><td>15.4</td><td>26.0</td><td>19.4</td><td>60.0</td><td>26.0</td><td>36.3</td></tr><tr><td>Published GNN</td><td></td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td><td>70.04</td></tr></table>

GeoChem $\mathsf { P } / \mathsf { R } / F _ { 1 }$ (%). Closed ArticleMiner runs use 28 entries and supplementary files; open runs use $^ { 2 6 }$ entries. Few-shot reads PDF text. The re<sub>p</sub>orted $\mathsf { P } / \mathsf { R } / F _ { 1 }$ a<sub>gg</sub>re<sub>g</sub>ates need not satisf<sub>y</sub> the harmonic-mean identit<sub>y</sub> across <sub>p</sub>a<sub>p</sub>ers. ArticleMiner is abbreviated AM.
<table><tr><td>Backbone</td><td>System</td><td>P</td><td>R</td><td> $F _ { 1 }$ </td></tr><tr><td>Sonnet 4.6</td><td>AM</td><td>58.0</td><td>82.9</td><td>60.8</td></tr><tr><td rowspan="2">Opus 4.6</td><td>Few-shot</td><td>3.0</td><td>0.5</td><td>0.8</td></tr><tr><td>AM</td><td>58.1</td><td>83.5</td><td>60.6</td></tr><tr><td rowspan="2">Haiku 4.5</td><td>Few-shot</td><td>8.0</td><td>2.1</td><td>3.0</td></tr><tr><td>AM</td><td>50.5</td><td>83.2</td><td>54.3</td></tr><tr><td rowspan="2">GPT-40</td><td>Few-shot</td><td>14.0</td><td>3.1</td><td>4.1</td></tr><tr><td>AM</td><td>62.2</td><td>82.9</td><td>63.1</td></tr><tr><td rowspan="2">Gemini 2.5 Flash</td><td>Few-shot AM</td><td>7.7</td><td>2.8</td><td>3.6</td></tr><tr><td>Few-shot</td><td>54.9 3.2</td><td>83.1</td><td>57.6</td></tr><tr><td rowspan="2">Qwen-2.5-7B (open)</td><td>AM</td><td>55.0</td><td>0.2 74.8</td><td>0.3</td></tr><tr><td>Few-shot</td><td>3.1</td><td>0.2</td><td>56.2</td></tr><tr><td rowspan="2">Mistral-7B (open)</td><td>AM</td><td>55.0</td><td>74.8</td><td>0.3</td></tr><tr><td>Few-shot</td><td>0.0</td><td>0.0</td><td>56.2 0.0</td></tr><tr><td rowspan="2">Llama-3.1-8B (open)</td><td>AM</td><td>55.0</td><td>74.8</td><td>56.2</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Few-shot</td><td>4.6</td><td>1.9</td><td>2.6</td></tr></table>

GPT-4o error distribution (% of error events; complements the main-text Table $^ { 6 , }$ which re<sub>p</sub>orts ArticleMiner-Sonnet and few-shot).
<table><tr><td>Task</td><td>System</td><td>Over-extraction</td><td>Omission</td><td>Attribute</td><td>Near miss</td><td> $n _ { \mathrm { e r r } }$ </td></tr><tr><td>CHEMTABLES</td><td>ARTICLEMINER-GPT40</td><td>5.2</td><td>62.7</td><td>31.2</td><td>0.9</td><td>346</td></tr><tr><td>MLTABLES</td><td>ARTICLEMINER-GPT40</td><td>27.1</td><td>59.3</td><td>0.1</td><td>13.5</td><td>2020</td></tr><tr><td>DiSCoMaT</td><td>ARTICLEMINER-GPT40</td><td>5.8</td><td>82.7</td><td>10.0</td><td>1.4</td><td>2788</td></tr></table>

per-paper scores have a long left tail (a small number of layout-only or borderless-table papers score near 0); the overall-mean interval (±5.7) reflects heterogeneity across papers.

![](images/0c954b368d2134eb28889750212579eedfa8b9c8127b2375d30c98776703a6a7.jpg)  
(a) Per-LLM four-tier Overall � on GeoChem.

![](images/2eecf5fde80ed3e193b8de3dee62dc37328c2ebe90c62739d6146f782945eb9c.jpg)  
(b) ChemTables few-shot $F _ { 1 }$ vs. �-shot exemplars (Sonnet 4.6).

Fi<sub>g</sub>ure 4: Supporting analyses: per-LLM four-tier Overall � on GeoChem (left), and the few-shot exemplar-count plateau on ChemTables (right).  
![](images/1e3772d46eb6b8050876aee0fb742e545a93f01fa13bd96e0e4b44fe8a8f9e13.jpg)  
Fi<sub>g</sub>ure 5: Precision and recall on raw-PDF in<sub>p</sub>uts<sub>, p</sub>lotted directl<sub>y</sub> from Tables 10–12. Circles denote ArticleMiner b<sub>ac</sub>kb<sub>ones; crosses</sub> d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> S<sub>onne</sub>t f<sub>ew-s</sub>h<sub>o</sub>t b<sub>ase</sub>li<sub>ne.</sub> E<sub>ac</sub>h <sub>pane</sub>l <sub>uses</sub> it<sub>s</sub> b<sub>enc</sub>h<sub>mar</sub>k’<sub>s s</sub>t<sub>r</sub>i<sub>c</sub>t t<sub>up</sub>l<sub>e-ma</sub>t<sub>c</sub>hi<sub>ng</sub> metric.

## H<sub>.</sub>3<sub>.</sub> I<sub>n</sub>t<sub>erpre</sub>t<sub>a</sub>ti<sub>on</sub> <sub>o</sub>f <sub>componen</sub>t <sub>an</sub>d b<sub>ac</sub>k<sub>en</sub>d <sub>ana</sub>l<sub>yses</sub>

Table 5(a) reports component removals, whereas panel (b) compares backend configurations. Their full/reference rows difer, so a backend efect must be computed against the corresponding all-five row in panel (b). Removing Camelot changes ChemTables from 20.8 to 24.8, MLTables from 47.6 to 55.3, and DiSCoMaT from 71.0 to 72.5. GeoChem’s backend comparison uses a 26-paper subset and a corrected ground-truth rerun, rather than the 28-paper headline reference. These distinctions prevent interpreting diferences across panels as a single controlled ablation sequence.

## H.4. O<sub>p</sub>en-LLM GeoChem 4-tier breakdown (�=26, max-tokens=512)

The three open 7–8B backbones run the full ArticleMiner pipeline on the 26-paper subset (the 2 datareuse entries lacking standalone PDFs are excluded; see annotation protocol). Reported per-model fourtier scores:

• Qwen-2.5-7B: �<sub>1</sub>=22.2, �<sub>2</sub>=64.3, �<sub>3</sub>=60.2, �<sub>4</sub>=83.5, Overall = 54.0.

• Mistral-7B-v0.2: �<sub>1</sub>=17.7, �<sub>2</sub>=64.3, �<sub>3</sub>=60.2, �<sub>4</sub>=83.5, Overall =52.6.

• Llama-3.1-8B: � =20.5, � =64.3, � =60.2, � =83.5, Overall =53.4.

The reported aggregate $T _ { 2 } / T _ { 3 } / T _ { 4 }$ scores are identical across these three runs, consistent with their shared supplementary-file parser. This does not imply identical cell-level outputs. All three deposit classifiers return malformed JSON under the 512-token cap, so the observed gap from closed models confounds model choice with decoding budget and coverage. Tables 10–13 give the separately reported sample-score grid.

## I<sub>.</sub> H<sub>ar</sub>d<sub>ware,</sub> S<sub>o</sub>ft<sub>ware,</sub> <sub>an</sub>d R<sub>epro</sub>d<sub>uc</sub>ibilit<sub>y</sub>

Experiments used Ubuntu 22.04 and CUDA 12.4, with one NVIDIA H100 80GB GPU and vLLM 0.6.x for open-model inference. Closed models were called through the Anthropic, OpenAI, and Google SDKs in April–May 2026. Reported identifiers are listed below.

claude-sonnet-4-6; claude-opus-4-6; claude-haiku-4-5-20251001;

gpt-4o-2024-08-06; gemini-2.5-flash; Qwen-2.5-7B-Instruct;

Mistral-7B-Instruct-v0.2; Llama-3.1-8B-Instruct.

Reported parser versions are Docling 2.x, Marker 1.x, MinerU 0.9, pdfplumber 0.11, and Camelot 0.11. MinerU’s OCR path was disabled when its weights were unavailable. The reported temperature is zero, and the resampling and exemplar-ordering seed is 42. The release contains stage-specific token caps and run configurations, but does not provide a complete manifest linking every historical result to its configuration. Table 3 reports per-paper cost and runtime.

The reported software versions are version families rather than a complete executable environment lock. The release also includes additional rotation handling and optional adaptive backend selection that were not part of the reported evaluated configurations. Reproducing those configurations requires the corresponding settings and output files, not merely running the latest default pipeline.

## J. Knowledge-Graph Output: Export Structure

Recovered samples are serialized as JSON-LD. The representation separates sample identity, elemental measurements, and source metadata, as described below.

## @context an<sup>d</sup> @graph.

The context declares a configurable default vocabulary and mineral/deposit namespace prefixes; the graph contains one node per sample row. A namespace declaration alone does not verify that the resulting terms correspond to a published ontology.

## S<sub>amp</sub>l<sub>e</sub> id<sub>en</sub>tit<sub>y.</sub>

Each node has @id, @type, and sample\_name. The identifier incorporates the paper identifier and a normalized sample name. Optional fields include mineral, deposit, and analytical method.

## Measurements<sub>.</sub>

Each nonempty elemental value produces a measurement containing element and value\_ppm, with a unit when available. The exporter adds a below-detection note for the designated sentinel. A stated detection limit encoded as a negative value does not receive that note under the same rule; downstream consumers must therefore retain and interpret the source value convention.

## S<sub>ource me</sub>t<sub>a</sub>d<sub>a</sub>t<sub>a.</sub>

The provenance object records the source PDF stem and pipeline model. These fields do not by themselves constitute a complete PROV mapping or cell-level evidence locator.

The task-level RDF binding describes the intended graph interface, while this exporter implements a particular representation. Native equivalent Turtle output, verified external ontology alignment, and downstream query correctness are not established by the tuple-level experiments. Such reuse requires validated predicate mappings, stable record identities, explicit censoring semantics, and compatible ingestion rules.

## K<sub>.</sub> GeoChem-28 Annotation Protocol

This section summarizes the curation protocol used for the 28 contributed mineral-geochemistry papers.

Scope. 28 peer-reviewed papers reporting LA-ICP-MS trace-element analyses ofsulfide and sulphosalt minerals in ore deposits, published 2004–2025. Each paper is annotated against the 209-column CMiO-MIN schema (an extension of CMiO/Hofstra [8] for individual-grain mineral geochemistry).

Annotators. Multiple geochemistry annotators participated, coordinated with U.S. Geological Survey domain experts. Annotations were independently spot-checked by a second annotator and any disagreements reconciled against the source PDF + supplementary tables.

## <sup>Per-</sup>p<sup>a</sup>p<sup>er</sup> p<sup>rocedure.</sup>

1. Deposit metadata $( T _ { 1 } ) .$ The fifteen annotated fields are deposit\_name, deposit\_type, deposit\_environment, deposit\_group, all\_commodities, mineral, analytical\_method, instrument\_type\_model,laboratory\_location,operating\_conditions,standards\_used, country, age, host\_rock, and sample\_reference. Values are transcribed from the publication where possible and otherwise normalized to the task vocabulary.

2. Per-samp<sup>l</sup>e rows $( T _ { 2 } / T _ { 3 } / T _ { 4 } )$ . Each row corresponds to one (sample\_id, mineral) analysis under a three-tier sample ID (top-level sample / sub-sample or thin section/spot). For each element column <el>\_ppm or <el>\_wtpct: a numeric value if reported; −� for a stated detection limit $L ,$ or −99999 when below detection is reported without a limit; blank if not measured or not reported.

3. Minera<sup>l</sup> assignment. One mineral per row. Mineral identity is taken from the per-analysis annotation in the paper’s supplementary tables (analysis\_id prefix, data-sheet name, or explicit column), never inferred from abundance patterns.

4. Units. Preserved as reported; no conversion. Columns carry the unit in their name (e.g. cu\_ppm, s\_wtpct).

## C<sub>onven</sub>ti<sub>ons.</sub>

<sup>•</sup> Be<sup>l</sup>ow-detection <sup>l</sup>imit: −� when the limit is known, or −99999 when it is unspecified. The encoding distinguishes censored measurements from missing values; the evaluator treats non-null gold entries in $T _ { 2 }$ and gold-null entries in $T _ { 4 }$

• Not applicable / not measured: blank cell.

<sup>•</sup> Re<sup>f</sup>erence materia<sup>l</sup>s (MASS-1, NIST 610, etc.) are included as sample rows tagged with the referencematerial flag. Annotation does not filter them out; both ArticleMiner and the few-shot baseline are expected to reproduce them in the prediction set so that the structural match in $T _ { 3 }$ is well-defined.

<sup>•</sup> Data-reuse papers (sufix \_as\_reported\_in\_X\_et\_al\_YYYY): gold tuples are taken from paper �’s supplementary tables. The original paper PDF is therefore not strictly required for evaluation, because all gold numbers reappear in the companion paper. Two of the 28 entries are data-reuse pairs; the open-LLM 4-tier breakdown in §H.4 reports $n = 2 6$ rather than $n = 2 8$ because the open-LLM PDF-vision path requires standalone PDFs (not available for the two reuse entries), whereas the closed-LLM and headline runs report � = 28 because they ingest the supplementary spreadsheets directly.

License. Ground-truth annotations are released under CC-BY-4.0. Source-paper PDFs retain their original licenses; the annotation license does not grant redistribution rights to those PDFs.