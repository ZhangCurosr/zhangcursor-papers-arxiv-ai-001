# Clade: Category-Level Answering from Deficient Evidence in Retrieval-Augmented Language Models

S M Rayeed, Charles V. Stewart, Yao Ma

Rensselaer Polytechnic Institute, Troy, NY, USA {rayees, stewart, may13}@rpi.edu

## Abstract

Retrieval-augmented systems often retrieve passages that are accurate and relevant but state the answer at a coarser level of granularity than the question requires. For example, a question may ask which specific disease a drug treats (familial hyperlipidemia), while the retrieved passage mentions the broader disease family (hyperlipidemia). Prior work on context suficiency treats such coarse passages simply as insuficient. We argue that they form a distinct case, which we call resolvable insuficiency: text alone cannot settle the fine-grained answer, but the broader category it mentions maps to a short list of candidate answers in a domain ontology. When language models read these coarse, family-only passages, they exhibit a phenomenon we call Clade (Category-Level Answering from Deficient Evidence): they answer instead of abstaining or searching further. On biomedical questions from PrimeKG, family-only passages reduce abstention by 13.8 points compared to uninformative passages, and the resulting guesses are wrong about half the time. A counterfactual test confirms the cause: replacing an exact disease name with its broader family name lowers abstention by 17.4 points compared to deleting the sentence. Simple fixes like confidence filtering, scaling to a larger model, and explicitly instructing the model to abstain cannot resolve Clade, and this replicates on ecological species interactions in GloBI. Finally, we test whether an ontology can turn coarse passages into exact answers. Given the correct family, an ontology list includes the true answer in 98.5% of cases, but string matching finds that family on the page for only 19.8%. Letting the model choose the family raises this to 30.0% and end-to-end recovery from 10.0% to 26.8%. Overall, our results show when language models should stop guessing and consult an ontology, and highlight which steps of that process remain unsolved. Codes are available at: https://anonymous.4open.science/r/CLADE.

## 1 Introduction

Retrieval-augmented generation (RAG) systems answer questions by reading retrieved text (Lewis et al. 2020). Recent work asks what a model should do when that text does not contain the complete answer. The usual framing is binary: a passage is either suficient or insuficient (Joren et al. 2025), and a reliable model answers in the first case and abstains in the second (Wen et al. 2025). In practice, many retrieved passages fall between these two extremes. Consider the question

What disease does lovastatin treat?, where the true answer is the specific condition familial hyperlipidemia. Suppose the retriever returns a passage stating that lovastatin is used to treat hyperlipidemia ( Figure 1). This passage is accurate and relevant. It rules out every disease outside the hyperlipidemia family, but it is too coarse to specify which exact subtype is meant. In this case, a system can guess a specific disease, abstain, or look further using a medical ontology that lists specific members of the hyperlipidemia family.

We call this phenomenon resolvable insuficiency. The passage is insuficient on its own because it states the answer at a coarse category level rather than the required fine-grained entity level. Yet it is resolvable because the coarse name maps directly to a short, finite set of candidate answers in a structured ontology. When a model instead guesses a finegrained answer directly from such coarse text, we name the behavior Clade (Category-Level Answering from Deficient Evidence). Although this setting occurs frequently in real text with biomedical documents state the exact disease name for only a small fraction of knowledge-graph facts while describing many others at the family level, prior work overlooks how coarse evidence afects model abstention. Rather than proposing a surface-level patch, this paper digs into the mechanics of this phenomenon to understand how coarse evidence misleads large language models (LLMs) and where a neural reader should hand of to symbolic knowledge.

To study this systematically, we group retrieved passages into three categories based on how specifically they describe the target answer: named (exact-match) passages, which explicitly mention the fine-grained target entity or an exact synonym;family-only (coarse) passages, which omit the specific answer but mention a broader parent category (such as a disease family or biological genus) within two levels above the answer in an ontology; and silent (unmentioned) passages, which discuss the query subject (e.g., the drug) but mention neither the specific answer nor its broader family. Using this framework, we investigate three research questions: (1) how often reference texts state fine-grained biomedical facts versus coarse categories (and whether a coarse name helps a model on its own), (2) whether family-only passages cause models to stop abstaining and guess inaccurately, and (3) whether a domain ontology can turn a coarse family mention into the exact fine-grained answer. Our main contributions are as follows:

![](images/2c0cb2fbb46f21de1d97a9d75515a85e6954bf3bad606ee6bbdf179c10b845da.jpg)  
Figure 1: Resolvable insuficiency as an agent workflow, with the bottleneck we measure at each stage. The retrieved passage is accurate and relevant but names only the answer’s family. A page-type check should send it to an ontology, which expands the family into a short candidate list for a chooser. The dashed red path is Clade: the model guesses from the family-only passage instead. In current practice, only the ontology step works reliably.

• We define resolvable insuficiency and Clade, and measure how text states 1,800 PrimeKG facts and 23,694 GloBI interactions. Text omits the exact disease for 77.5– 84.9% of PrimeKG facts, and a family name alone gives the model no closed-book advantage.

• On 856 PrimeKG questions and three 7–8B models, family-only passages cut abstention by 13.8 points relative to silent passages. About half of the extra answers are wrong, and precision falls from 84–95% to 45–59%.

• Counterfactual edits show that the family name causes this drop (−17.4 points against deleting the sentence). Confidence filtering, a 14B model and an instruction to abstain do not stop Clade, and the efect replicates on 900 GloBI questions. In free text, no model gives the exact answer from a family-only passage.

• On 410 family-only facts, an ontology list holds the answer 98.5% of the time once the family is known, but string matching finds the family from the passage only 19.8% of the time. A model that chooses the family raises this to 30.0% and end-to-end recovery from 10.0% to 26.8%, and a popularity prior beats the model at choosing within the list.

## 2 Related Work

Insuficient context. Retrieval failures are a common weakness in RAG systems (Barnett et al. 2024). Joren et al. (2025) study the errors through the lens of context suficiency. They label retrieved passages as either suficient or insuficient and find that LLMs often hallucinate instead of abstaining when context is insuficient. Although they note that a passage can narrow down the answer without fully settling it, they group all such cases under insuficient context. Prior work also shows that models are easily thrown of by irrelevant text (Shi et al. 2023; Yoran et al. 2024), and Setiawan (2026) show that misleading passages cause prompt-based abstention to fail, whereas missing passages do not. Our focus, resolvable insuficiency, difers from both irrelevant and misleading context. A coarse passage is factually accurate and points in the right direction, yet this partial relevance is enough to trigger Clade and break abstention.

Partial evidence. To handle imperfect passages, several methods train or prompt models to grade evidence quality. Self-RAG (Asai et al. 2024) trains a model to judge whether a passage fully or partially supports a generated answer, and ALCE (Gao et al. 2023) checks whether citations back up model claims. Similarly, CRAG (Yan et al. 2024) scores retrieved documents to trigger new searches when support is weak, and SURE-RAG (Qiu, Han, and Huang 2026) checks whether context is suficient before answering. While these methods detect that support is partial, they do not identify what kind of information is missing. A coarse mention is a structured form of partial evidence: it shows that finegrained entity detail is missing and indicates where to find the candidate answers in an ontology.

Abstention. When evidence falls short and cannot be resolved, a reliable model should abstain. Recent surveys and benchmarks show that models often guess when they should hold back (Wen et al. 2025; Kirichenko et al. 2025), including in clinical settings (Machcha et al. 2026). Most existing methods focus on detecting when a model lacks knowledge so it can decline to answer (Kadavath et al. 2022; Feng et al. 2024). While those studies measure whether a model abstains overall, we test how the granularity of retrieved evidence (named, family-only, silent) causes a model to stop abstaining and guess.

Ontologies in RAG. Rather than guessing or abstaining on a coarse passage, a system can consult a structured ontology. Recent RAG systems use symbolic structures to guide retrieval: OG-RAG (Sharma, Kumar, and Li 2024) grounds retrieval in domain ontologies, and TaSR-RAG (Sun et al. 2026) uses taxonomies for multi-hop retrieval. Broader work on combining LLMs with knowledge graphs and symbolic reasoning is surveyed by Pan et al. (2024) and d’Avila Garcez and Lamb (2023). Unlike pipelines that query an ontology on every question, we study the specific case where a coarse passage calls for an ontology lookup. We also measure each step of the pipeline: extracting the family, listing its members, and picking the final answer; then we check where the process breaks down.

<table><tr><td></td><td>PrimeKG</td><td>GloBI</td></tr><tr><td>Source records</td><td>8,100,498</td><td>7,822,670</td></tr><tr><td>Edges</td><td>indication 9,388 contraind. 30,675 off-label 2,568</td><td>eats 235,960 preys on 87,678 parasite of 117,894 pollinates 15,854</td></tr><tr><td>Hierarchy</td><td>MONDO</td><td>GloBI taxonomy</td></tr><tr><td>child-parent links</td><td>17,801</td><td>77,852</td></tr><tr><td>families, ≥2 members</td><td>3,182</td><td>8,370</td></tr><tr><td>median / max size</td><td>3 / 150</td><td>3 /315</td></tr><tr><td>Text</td><td>Wiki. + PubMed</td><td>Wikipedia</td></tr><tr><td>sources with text</td><td>574 drugs</td><td>5,425 species</td></tr><tr><td>facts measured</td><td>3,444</td><td>23,694</td></tr><tr><td>Questions</td><td>856</td><td>900</td></tr></table>

Table 1: Data for both domains. PrimeKG links drugs to diseases, and MONDO groups diseases into families. GloBI links species to species, and its taxonomy groups species into genera. Afamily is a parent with two or more children. Facts measured are edges we check for a mention of the answer.

Fact coverage in text. Our analysis also connects to work on what reference and pretraining texts actually state. Mallen et al. (2023) and Kandpal et al. (2023) link a model’s closedbook knowledge to entity popularity and document counts, while LMEnt (Gottesman et al. 2026) tracks entity mentions across pretraining data. Whereas those studies count mentions of individual entities, we measure how often reference texts state relational facts at an exact (named) level versus a coarse (family-only) level.

Fine-grained biology. Biology shows the same gap between coarse and fine answers. On Bio-VQA, taxonomic accuracy falls from 0.72 at phylum to 0.12 at species (Rayeed et al. 2027). And models lose accuracy from genus to species (Rayeed et al. 2025) even on curated specimen images with validated traits (Rayeed et al. 2026). Knowledge-based VQA benchmarks also contain answers that their corpus does not support (Ma et al. 2026). We investigate the text side of this gap: a page that names the genus or family but not the answer.

## 3 Data and Problem Setup

Knowledge graphs and ontologies. We use drug-disease relations from PrimeKG, linked to the MONDO Disease Ontology. We reconstruct MONDO into 17,801 edges across 11,788 diseases (excluding parents with >400 descendants). To test generalizability beyond medicine, we evaluate species interactions from GloBI, using 457,386 species-to-species edges across four relations (eats, preys on, parasite of, pollinates) with GloBI’s taxonomy (species, genus, family) as the hierarchy. Table 1 summarizes both domains.

Reference text corpora. For PrimeKG drugs, we use English Wikipedia articles and PubMed abstracts. For GloBI organisms, we use species’ Wikipedia articles (dropping stubs <3000 characters). All articles are stripped of markup, templates, infoboxes, tables, and file links.

<table><tr><td></td><td></td><td colspan="2">Text names it</td><td colspan="2">Text does not</td></tr><tr><td>Relation</td><td>n</td><td>knows</td><td>not</td><td>knows</td><td>not</td></tr><tr><td>Indication</td><td>600</td><td>9.7</td><td>7.2</td><td>21.8</td><td>61.3</td></tr><tr><td>Contraind.</td><td>600</td><td>1.5</td><td>5.2</td><td>20.7</td><td>72.7</td></tr><tr><td>Off-label</td><td>600</td><td>4.3</td><td>6.2</td><td>24.2</td><td>65.3</td></tr><tr><td>All</td><td>1,800</td><td>5.2</td><td>6.2</td><td>22.2</td><td>66.4</td></tr></table>

Table 2: Share of PrimeKG facts (%) by whether the drug’s Wikipedia article names the disease and whether Qwen2.5 0.5B picks it from five options closed book (“knows”).

<table><tr><td>Relation</td><td>Text</td><td>n</td><td>Named</td><td>Family</td><td>Silent</td></tr><tr><td rowspan="3">Indication</td><td>Wikipedia</td><td>1,722</td><td>20.3</td><td>15.5</td><td>64.2</td></tr><tr><td>PubMed</td><td>1,722</td><td>26.3</td><td>18.6</td><td>55.1</td></tr><tr><td>both</td><td>1,722</td><td>29.0</td><td>19.7</td><td>51.3</td></tr><tr><td rowspan="3">Contraind.</td><td>Wikipedia</td><td>1,722</td><td>9.8</td><td>4.1</td><td>86.1</td></tr><tr><td>PubMed</td><td>1,722</td><td>11.4</td><td>5.4</td><td>83.2</td></tr><tr><td>both</td><td>1,722</td><td>15.9</td><td>6.9</td><td>77.2</td></tr><tr><td>PrimeKG</td><td>both</td><td>3,444</td><td>22.5</td><td>13.3</td><td>64.2</td></tr><tr><td>Eats</td><td>Wikipedia</td><td>8,000</td><td>4.0</td><td>6.6</td><td>89.3</td></tr><tr><td>Preys on</td><td>Wikipedia</td><td>8,000</td><td>2.2</td><td>4.4</td><td>93.4</td></tr><tr><td>Parasite of</td><td>Wikipedia</td><td>2,389</td><td>12.5</td><td>15.5</td><td>72.0</td></tr><tr><td>Pollinates</td><td>Wikipedia</td><td>5,305</td><td>2.2</td><td>8.0</td><td>89.7</td></tr><tr><td>GloBI</td><td>Wikipedia</td><td>23,694</td><td>3.9</td><td>7.1</td><td>89.1</td></tr></table>

Table 3: Share of knowledge-graph facts whose text names the answer, names only its family, or names neither. PrimeKG uses 574 drugs with three facts of each relation.

Evidence granularity. We classify retrieved text into three levels: named (exact name, synonym, abbreviation), familyonly (broad parent category within two levels), and silent (neither). A control with unrelated disease pairs shows that matches track facts, not the size of the name set (unrelated matches occur in only 0.9–1.4% of Wikipedia articles, versus 6.7–19.3% for true diseases).

Corpus coverage. Evaluating 1,800 PrimeKG facts with Qwen2.5 0.5B (closed book) reveals that for 66.4% of facts, the text lacks the exact disease name and the model lacks the fact. As detailed in Table 2 and Table 3, the exact disease is missing in 84.9% of Wikipedia articles, 81.1% of PubMed sets, and 77.5% combined, with random guessing accounting for most closed-book correct answers on unnamed facts.

Closed-book recall. Although family-only text covers 13.3% of PrimeKG facts, testing on Qwen2.5-7B shows that coarse family mentions do not reflect stronger underlying model knowledge: it answers 39.6% of family-only facts, 32.0% of silent facts and 65.7% of named facts correctly (Appendix B). Odds ratio (OR) compares the odds of a correct closed-book answer in two groups. OR=1 means no diference, and OR<1 means the first group is answered correctly less often. We pool the relations with the Mantel–Haenszel estimator; � is the chance of a gap this large if the groups did not difer. Family-only facts are answered far less often than named facts $( \mathbf { O R } = 0 . 2 7 , p = 9 \times 1 0 ^ { - 9 } )$ and as often as silent facts $( \mathrm { O R } = 1 . 1 0 , p = 0 . 6 1 ) $ . While an exact Wikipedia mention yields a 3.7× increase in the odds of model knowledge, a family name provides no measurable advantage on its own. This motivates testing how models handle coarse partial evidence at inference time, and whether symbolic ontologies can recover exact answers.

## 4 Methodology

We run four experiments. We measure how family-only pages change abstention, edit pages to isolate the efect of the family name, test three common fixes, and test whether an ontology can recover the exact answer.

## 4.1 Abstention Evaluation Setup

Questions. We curate 856 multiple-choice questions from the PrimeKG indications and contraindications: 300 with a named page, 256 with a family-only page, and 300 with a silent page. Each relation and group has 150 questions, except family-only contraindications, where only 106 facts exist. Each question has five candidate diseases and a sixth option, Not enough information to tell. The four wrong options come from the answer’s MONDO neighborhood: siblings first, then cousins, then random diseases only if both run out. We exclude the answer’s ancestors and descendants and any other disease the drug has under the same relation.

Pages. We ask each question twice: closed book and with a page. The page starts with the drug’s cleaned Wikipedia article, followed by its PubMed abstracts. We keep the first 800 characters and a 280 character window around each mention of the answer or its family, merge overlapping windows, and cap the page at 1800 characters. A silent page has no mentions, so it is the article’s opening.

Models and Scoring. We evaluate five instruction-tuned open-weight models from three families: Qwen2.5 0.5B, 7B and 14B (Qwen Team 2024), OLMo 2 7B (Team OLMo et al. 2024), and Granite 3.1 8B (IBM Granite Team 2024). The model sees the page, the question and options A–F. We take the letter with the highest next-token probability, renormalized over the six letters. Choosing Not enough information to tell counts as a decline. As a format check, we also ask 120 matched questions with no options and score each generated answer by name matching.

Controls. Two diferences between groups could bias the comparison. First, family-only answers more often have enough relatives to fill all four wrong options, which makes those questions harder. Our main analysis therefore keeps only questions whose four wrong options all come from the neighborhood: 106 named, 215 family-only and 160 silent. Appendix D reports all 856 questions, with the same pattern.

Second, the groups hold diferent facts, which may difer in dificulty before any page is read. We therefore compare each question with itself. For a question � and an outcome such as a decline, let $y _ { q } ^ { \mathrm { p a g e } } = 1$ if the outcome occurs when the model reads the page and 0 otherwise, and let $y _ { q } ^ { \mathrm { c l o s e d } }$ be the same without the page. The page efect of � is $y _ { q } ^ { \mathrm { { \bar { p a g e } } } } - y _ { q } ^ { \mathrm { c l o s e d } }$ . Our main measure is the diference in mean page efect between the family-only questions � and the silent questions �:

$$
\Delta = \frac { 1 } { | F | } \sum _ { q \in F } \bigl ( y _ { q } ^ { \mathrm { p a g e } } - y _ { q } ^ { \mathrm { c l o s e d } } \bigr ) - \frac { 1 } { | S | } \sum _ { q \in S } \bigl ( y _ { q } ^ { \mathrm { p a g e } } - y _ { q } ^ { \mathrm { c l o s e d } } \bigr ) .\tag{1}
$$

For declines, $\Delta < 0$ means a family-only page pushes the model toward declining less than a silent page does. To pool the 7–8B models, we average each question’s page efect over the models first, so each question counts once.

Statistics. Efects are in percentage points. Brackets give 95% confidence intervals (CI) from 5,000 bootstrap resamples of questions; an interval that excludes 0 marks a reliable efect. �-values come from two-sided �-tests (Appendix C).

## 4.2 Counterfactual Evaluation

Page edits. The groups in Section 4.1 test diferent facts. To hold the fact fixed, we take 130 named questions whose answer has a MONDO parent. Then we edit each page in two ways: family edit replaces every mention of the answer with its parent’s name, and removed edit deletes every sentence that mentions the answer (Figure 3). The question, options and model stay the same.

Simple fixes fail. We test three ways to curb Clade. Confidence filtering uses the renormalized probability of the chosen option; we report the AUROC for separating correct from wrong answers. Scaling compares Qwen2.5 at 0.5B, 7B and 14B. Instruction adds a line to the prompt telling the model to choose Not enough information to tell when the page names only a broad category.

Second domain. We repeat the main experiment and counterfactual edits on 900 GloBI questions: 75 per page type in each of four relations. Wrong options come from the same genus first, then the same family.

## 4.3 Ontology Recovery

We test recovery on 410 PrimeKG facts whose Wikipedia and PubMed text names only the disease family (Figure 1). A drug often has several true diseases in a family (2.47 per list on average), so a pick counts as correct if it matches any of them. Recovery has three stages:

1. Build the list. The oracle given-family list takes the narrowest MONDO ancestor named in the text and lists its descendants up to two levels down, capped at 64. The automaticpage-built list takes every MONDO parent named in the drug’s Wikipedia article and joins their descendants; lists over 64 count as too broad. The model-chosen list asks Qwen2.5-7B which of the families on the page (up to ten) holds the answer and expands only that one.

2. Choose one. We compare five options: (1) Qwen2.5-7B closed book, (2) Qwen2.5-7B with the page in the prompt, (3) a reverse lookup to check whether the candidate’s own Wikipedia article names the drug, (4) PubMed comention counts with relation keywords, raw and normalized by the disease’s PubMed volume, and (5) a popularity prior that picks the disease with the most PubMed abstracts, ignoring the drug.

<table><tr><td>Model</td><td>Passage Type Correct Wrong</td><td></td><td></td><td>Declines Prec.</td><td></td></tr><tr><td rowspan="3">Qwen2.5 7B</td><td>named</td><td>71.7</td><td>3.8</td><td>24.5</td><td>95.0</td></tr><tr><td>family-only</td><td>26.5</td><td>18.1</td><td>55.3</td><td>59.4</td></tr><tr><td>silent</td><td>13.8</td><td>2.5</td><td>83.8</td><td>84.6</td></tr><tr><td rowspan="3">OLMo 2 7B</td><td>named</td><td>77.4</td><td>15.1</td><td>7.5</td><td>83.7</td></tr><tr><td>family-only</td><td>37.2</td><td>44.7</td><td>18.1</td><td>45.5</td></tr><tr><td>silent</td><td>30.6</td><td>24.4</td><td>45.0</td><td>55.7</td></tr><tr><td rowspan="3">Granite 3.1 8B</td><td>named</td><td>72.6</td><td>7.5</td><td>19.8</td><td>90.6</td></tr><tr><td>family-only</td><td>30.2</td><td>26.5</td><td>43.3</td><td>53.3</td></tr><tr><td>silent</td><td>15.0</td><td>10.0</td><td>75.0</td><td>60.0</td></tr><tr><td rowspan="3">Qwen2.5 0.5B</td><td>named</td><td>77.4</td><td>14.2</td><td>8.5</td><td>84.5</td></tr><tr><td>family-only</td><td>34.4</td><td>55.8</td><td>9.8</td><td>38.1</td></tr><tr><td>silent</td><td>28.7</td><td>50.6</td><td>20.6</td><td>36.2</td></tr><tr><td rowspan="3">Qwen2.5 14B</td><td>named</td><td>67.9</td><td>2.8</td><td>29.2</td><td>96.0</td></tr><tr><td>family-only</td><td>24.2</td><td>17.2</td><td>58.6</td><td>58.4</td></tr><tr><td>silent</td><td>11.9</td><td>2.5</td><td>85.6</td><td>82.6</td></tr></table>

Table 4: Open-book outcomes (%) on PrimeKG questions where all four distractors come from the answer’s MONDO neighborhood (106 named, 215 family-only, and 160 silent questions). Precision is the percentage of correct answers among questions where the model did not abstain.

3. Compare without ontology. Two baselines replace the family list with a list of the same length that holds the answer plus random diseases, scored by the model with or without the page. We test paired diferences with Mc-Nemar’s test (McNemar 1947).

## 5 Results and Analysis

## 5.1 Coarse evidence weakens abstention (Clade)

Models abstain less with partial evidence. Table 4 shows outcomes when the model reads the page. All 7-8B models abstain much less on a family-only page than on a silent page: Qwen2.5-7B from 83.8% to 55.3%, OLMo 2 from 45.0% to 18.1%, and Granite from 75.0% to 43.3%. Many of the extra answers are wrong. Precision, the share of answers that are correct, falls from 83.7-95.0% on named pages to 45.5-59.4% on family-only pages.

Clade drop comes from context. Family-only facts are harder even closed book: they draw more wrong guesses than silent facts in all three 7-8B models (Appendix D). The page efect in Eq. 1 removes this diference. Pooled over the models, a silent page raises abstention by 28.5 points over closed book, but a family-only page raises them by only 14.7 points. The diference Δ is −13.8 points [−20.3, −7.5], and it stays significant after a Bonferroni correction for three outcomes $( p { = } 1 . 1 { \times } 1 0 ^ { - 4 } ;$ Figure 2). The freed answers split between wrong answers (+7.7 [1.2, 14.0]) and correct ones (+6.1 [−0.1, 12.4]). In plain terms, a silent page makes the models stop and say they cannot tell; a family-only page makes them stop much less often, and they guess instead. The drop is significant for OLMo 2 (−22.1, �=6×10<sup>−6</sup>), Granite (−18.2, $\scriptstyle p = 6 \times 1 0 ^ { - 4 } )$ and Qwen2.5-0.5B (−8.4, �=0.027). Qwen2.5- 7B shows almost no gap $( - 1 . 1 , p { = } 0 . 8 2 )$ . It already declines family-only questions far less often closed book (32.1% vs.

![](images/c641fa5b2f320b05355132b9833f80ecdc71221d08f3bc02e12f8b3a7963a3e3.jpg)  
Figure 2: Efect of a family-only page relative to a silent page. Each point is Δ from Eq. 1 for one outcome, with its 95% bootstrap CI; points left of zero mean the family-only page lowers that outcome. The pooled row averages the three 7-8B models over 481 questions. Faded rows are models outside the pooled set.

![](images/1ea8cb85617e446a4b282cca6bfd156ed257c01f411ec2b857a31a6c65e2a40c.jpg)  
Figure 3: Page edits on the same questions: PrimeKG (130) and GloBI (255). The family edit replaces the answer with its parent; the removed edit deletes the sentence. Bars show outcomes with the page (%), averaged over the 7-8B models.

59.4%), and the page raises its abstention by about the same amount in both groups.

An example. For moxifloxacin, PrimeKG lists Klebsiella pneumonia, and the page says only that the drug treats pneumonia. All five models pick some bacterial pneumonia with confidence up to 1.00, and none abstains. For lovastatin, OLMo 2 is right closed book but picks a sibling after reading a page that says only hyperlipidemia (Appendix I). Without options, failure looks diferent: in a free-text check on 120 questions, no model names the exact answer from a familyonly page; instead 17.5-45.0% answers give the family itself, which never occurs on named or silent pages (Appendix K).

Family size. We split the family-only questions into thirds by family size. The drop is large for families of up to 10 members (−17.3 [−26.2, −8.2]) and of 11-20 members (−18.4 [−26.6, −10.3]). It shrinks to −6.5 [−15.7, +3.0] for families above 20 members (Figure 4c). A narrow family makes the page look more answerable.

## 5.2 Counterfactual Edits confirm Clade

The edits change only the page (Figure 3). Against the original page, the family edit lowers correct answers by 21.0 points and raises wrong answers by 9.5 points, while abstention rises by only 11.5 points. The key comparison is with the removed edit. Keeping the family name instead of deleting the sentence cuts abstention by 17.4 points [−23.6, −11.5] and raises wrong answers by 5.1 points [0.5, 10.3]. Every model shows this order: from removed to family, abstention falls from 70.0% to 43.8% for Qwen2.5-7B, from 56.2% to 40.8% for Granite, and from 18.5% to 7.7% for OLMo 2. The family name alone suppresses abstention.

## 5.3 Three Fixes Fall Short

Confidence does not flag wrong answers. AUROC measures how well confidence ranks the correct answers above wrong ones: 1.0 is perfect and 0.5 is chance. On family-only pages it is 0.59-0.72, against 0.73-0.87 on named pages, in every model (Figure 4a). Wrong guesses are confident: their mean probability is 0.77-0.99 in 7-14B models. A threshold that keeps correct answers also keeps most wrong ones.

Scaling does not help. Qwen2.5-14B has the best precision on named pages (96.0%). On family-only pages, precision rises from 38.1% at 0.5B to 59.4% at 7B, but then stops: 14B reaches 58.4%, and it gives 14.7 points more wrong answers than on silent pages $( p { = } 6 { \times } 1 0 ^ { - 6 } )$ ).

Instruction. Telling the model not to answer when the page names only a broad category raises abstention by 15.2 points on family-only pages (Figure 4b). It also raises them by 14.0 points on silent pages and 9.1 points on named pages, where it costs 6.3 points of correct answers. Precision on family-only pages stays at 47.9-64.9%, far below named pages (89.6-94.7%). The models scores change diferently, OLMo 2 abstains far more on all page types, while Granite barely changes (Appendix F). However, none of these models tells a family name from a specific answer.

## 5.4 Second Domain: GloBI

Every finding on PrimeKG replicates on species interactions (Table 5). A silent page raises abstention by 12.1 points, but a page that names only the genus or family lowers them by 2.3 points (Δ=−14.3, $\stackrel { \smile } { p = 1 . 4 \times 1 0 ^ { - 9 } } )$ . The efect is largest in OLMo 2 (−31.7). Here the extra answers are mostly wrong (+11.6 wrong, $p { = } 2 . 5 { \times } 1 0 ^ { - 8 } ; ~ { + } 2 . 7$ correct, �=0.09). Precision on family-only pages falls to 22-28%, close to the 20% chance rate, because sibling species are harder to tell apart than sibling diseases. The page edit is even sharper: on 255 questions, keeping the genus name instead of deleting the sentence cuts abstention by 27.7 points [−32.2, −23.5] and raises wrong answers by 6.7 points (Figure 3). Clade comes from hierarchical evidence in text, not from medical terms (Appendix G).

## 5.5 Ontology Recovery

List coverage. When the right family is given, its MONDO descendants form a list of median 14 candidates. The list holds the sampled answer 98.5% of the time and some true answer 99.5% of the time (Table 6).

<table><tr><td>Metric</td><td>PrimeKG</td><td>GloBI</td></tr><tr><td>Questions in pooled set (all-neighbor)</td><td>481</td><td>833</td></tr><tr><td>Abstention page effect, family – silent</td><td>-13.8</td><td>-14.3</td></tr><tr><td>Wrong-answer page effect, family – silent</td><td>+7.7</td><td>+11.6</td></tr><tr><td>Precision on named passages (%)</td><td>84-95</td><td>84-95</td></tr><tr><td>Precision on family-only passages (%)</td><td>45-59</td><td>22-28</td></tr><tr><td>Confidence AUROC on named passages</td><td>.73-.87</td><td>.82-.86</td></tr><tr><td>Confidence AUROC on family-only passages</td><td>.66-.72</td><td>.52-.68</td></tr><tr><td>Edit: abstention, family – removed</td><td>-17.4</td><td>-27.7</td></tr><tr><td>Edit: wrong answers, family – removed</td><td>+5.1</td><td>+6.7</td></tr><tr><td>Edit: correct answers, family – original</td><td>-21.0</td><td>-30.8</td></tr></table>

Table 5: Comparison of biomedical drug–disease facts (PrimeKG with MONDO) and ecological species interactions (GloBI with biological taxonomy) across the three 7– 8B models. Efects are reported in percentage points; ranges span individual models. Every diference reported has a 95% CI excluding zero.

<table><tr><td>Metric</td><td>Given</td><td></td><td>Page-built Model-chosen</td></tr><tr><td>Usable lists (of 410)</td><td>410</td><td>166</td><td>375</td></tr><tr><td>Holds sampled answer</td><td>98.5%</td><td>19.8%</td><td>30.0%</td></tr><tr><td>Holds any true answer</td><td>99.5%</td><td>24.6%</td><td>44.9%</td></tr><tr><td>Median list size</td><td>14</td><td>32</td><td>12</td></tr><tr><td>End-to-end recovery</td><td>一</td><td>10.0%</td><td>26.8%</td></tr></table>

Table 6: Candidate lists for the 410 family-only facts: given family (oracle), all families on the page (page-built), and the one family Qwen2.5-7B picks from the page (model-chosen). Of the 244 unusable page-built lists, 209 exceed 64-candidate cap and 35 match no multi-child MONDO family. End-toend recovery uses the best chooser for each list.

Finding the family. When the family must come from the drug’s Wikipedia page, usable lists hold the answer for only 19.8% of facts. The family is rarely missing; it is buried. A drug article names a median offive MONDO categories, such as comorbidities, side efects and other uses, and their union exceeds 64 candidates for 209 of 410 facts. PubMed makes this worse: coverage drops to 3.7%. A model does better than string matching. When Qwen2.5-7B reads the page and picks one of the families it names, the chosen list holds the answer for 30.0% of facts (44.9% for any true answer), above the first-mentioned family (25.1%) and a random one (13.9%). Its lists are also short, with a median of 12 candidates. The ceiling is 50.0%: for half of the facts, the right family is not among the usable families on the page.

Choosing a member. Within the given-family list (Figure 5), Qwen2.5-7B alone reaches 35.9% (32.4% with the page in the prompt), and raw PubMed co-mentions also reach 35.9%. Evidence first, with the model as a fallback, reaches 41.7%. Yet the best chooser ignores the drug: picking the disease with the most PubMed abstracts reaches 46.8%. When we divide co-mentions by each disease’s abstract count, accuracy drops from 35.9% to 25.4%, so co-mentions work mainly because common diseases are studied more.

(c) Size of the family  
family members  
![](images/31d5b6b3e9b6510fe36f52d10b51097b9f99f92f7a61a46f675800f62bd450e0.jpg)

![](images/f56f779009f5f1a3a21d4c5647fe926f2658555d7975ce2074067713971b8b29.jpg)

![](images/cc44f6c473adf6dc53e59ec43dd2f0b6780c3fb14670263ddbdcd4ea1e89de10.jpg)

Figure 4: Three candidate fixes. (a) Confidence separates correct from wrong answers poorly on family-only pages. (b) An instruction to decline on broad categories raises abstention on every page type, including named pages (pooled 7-8B models, 95% CIs). (c) The drop in abstention is largest when the family has 20 or fewer members.  
![](images/54933b22ced9a6ad78fd04f031ef1af0df89b0d0532e4be51411ff69488604ed.jpg)  
Figure 5: Accuracy of choosing the exact disease for 410 family-only facts. Blue and gray bars choose from the givenfamily list; hatched bars do not receive the true family.

Errors and end-to-end recovery. Most misses pick a better-studied sibling, such as ovarian carcinoma instead of malignant germ cell tumor of ovary for carboplatin, or a valid use missing from PrimeKG, so strict graph scoring understates accuracy (Appendix H). Without the given family, page-built lists recover the answer for 10.0% of facts with Wikipedia and 1.5% with PubMed added; model-chosen lists recover 26.8% (Table 6).

## 6 Discussion

Hand-of point. Our results mark a clear boundary between the language model and the ontology. On a familyonly page, models take the family as license to guess. The ontology turns the same family into a short list that holds the answer, once the system knows which family on the page belongs to the question.

A three-step workflow. This suggests a simple workflow (Figure 1). (1) Check the page type, and answer from the page only if it names the answer. (2) Find the target family tied to the queried relation; if none fits, abstain. (3) Expand and choose with evidence and a popularity prior; answer only when one candidate clearly leads, otherwise return the list. Current models skip steps 1 and 2, and neither confidence nor instructions restore step 1. Step 2 is the main bottleneck: string matching finds the right family for 19.8% of facts, and a model chooser raises this to 30% against a ceiling of 50%. Any chooser for step 3 should beat the popularity prior.

Beyond suficiency. Recent work asks whether a page is suficient (Joren et al. 2025); our results show that how it falls short matters. A family-only page difers from a silent one: it carries real signal, it breaks abstention, and it points into a symbolic hierarchy. Treating it as its own case lets a RAG system send partial evidence to an ontology instead of guessing or discarding it.

## 7 Limitations

We study two domains and run recovery only on PrimeKG. We use multiple-choice questions with an explicit decline option, which controls the wrong options and the scoring but may difer from free-text refusals. A free-text check points the same way (Appendix K): no model gives the exact answer from a family-only page, and 17.5–45.0% ofits answers name the family itself. We test open models up to 14B, and the page edits cover only named facts whose answer has a parent. Page-built lists rely on exact string matching. Family-only contraindications are few (106 questions).

## 8 Conclusion

Retrieved text often states facts at a coarser level than the question asks. We show that this resolvable insuficiency triggers Clade in medicine and ecology: a family or genus name leads models to stop declining and guess, which roughly halves precision. An ontology can narrow the answer to a short list that holds it 98.5% of the time once the right family is found, and a model that chooses the family from the page already raises end-to-end recovery from 10.0% to 26.8%. Finding the family more reliably and beating a popularity prior remain the key open problems for neuro-symbolic RAG systems that must know when to hand of to an ontology.

## References

Asai, A.; Wu, Z.; Wang, Y.; Sil, A.; and Hajishirzi, H. 2024. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. In International Conference on Learning Representations (ICLR).

Barnett, S.; Kurniawan, S.; Thudumu, S.; Brannelly, Z.; and Abdelrazek, M. 2024. Seven Failure Points When Engineering a Retrieval Augmented Generation System. In Proceedings of the IEEE/ACM 3rd International Conference on AI Engineering – Software Engineering for AI (CAIN), 194– 199.

d’Avila Garcez, A.; and Lamb, L. C. 2023. Neurosymbolic AI: The 3rd Wave. Artificial Intelligence Review, 56: 12387– 12406.

Feng, S.; Shi, W.; Wang, Y.; Ding, W.; Balachandran, V.; and Tsvetkov, Y. 2024. Don’t Hallucinate, Abstain: Identifying LLM Knowledge Gaps via Multi-LLM Collaboration. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (ACL), 14664–14690.

Gao, T.; Yen, H.; Yu, J.; and Chen, D. 2023. Enabling Large Language Models to Generate Text with Citations. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), 6465–6488.

Gottesman, D.; Gilaie-Dotan, A.; Cohen, I.; Gur-Arieh, Y.; Mosbach, M.; Yoran, O.; and Geva, M. 2026. LMEnt: A Suite for Analyzing Knowledge in Language Models from Pretraining Data to Representations. Transactions of the Associationfor Computational Linguistics, 14: 1685–1722.

IBM Granite Team. 2024. Granite 3.0 Language Models. Technical report, IBM.

Joren, H.; Zhang, J.; Ferng, C.-S.; Juan, D.-C.; Taly, A.; and Rashtchian, C. 2025. Suficient Context: A New Lens on Retrieval Augmented Generation Systems. In International Conference on Learning Representations (ICLR).

Kadavath, S.; Conerly, T.; Askell, A.; et al. 2022. Language Models (Mostly) Know What They Know. arXiv preprint arXiv:2207.05221.

Kandpal, N.; Deng, H.; Roberts, A.; Wallace, E.; and Rafel, C. 2023. Large Language Models Struggle to Learn Long-Tail Knowledge. In Proceedings of the 40th International Conference on Machine Learning (ICML), 15696–15707.

Kirichenko, P.; Ibrahim, M.; Chaudhuri, K.; and Bell, S. J. 2025. AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions. arXiv preprint arXiv:2506.09038.

Lewis, P.; Perez, E.; Piktus, A.; et al. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In Advances in Neural Information Processing Systems 33 (NeurIPS), 9459–9474.

Ma, Q.; Rayeed, S. M.; Stewart, C. V.; Wu, Q.; and Ma, Y. 2026. Identifying and Resolving Pitfalls of Knowledge-Based VQA Benchmarks: Auditing, Repairing, and Augmenting. In European Conference on Computer Vision (ECCV).

Machcha, S.; Yerra, S.; Gupta, S.; Sahoo, A.; Sultana, S.; Yu, H.; and Yao, Z. 2026. Knowing When to Abstain: Medical LLMs Under Clinical Uncertainty. In Proceedings of the

19th Conference of the European Chapter of the Association for Computational Linguistics (EACL), 6153–6182.

Mallen, A.; Asai, A.; Zhong, V.; Das, R.; Khashabi, D.; and Hajishirzi, H. 2023. When Not to Trust Language Models: Investigating Efectiveness of Parametric and Non-Parametric Memories. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL), 9802– 9822.

McNemar, Q. 1947. Note on the Sampling Error of the Diference Between Correlated Proportions or Percentages. Psychometrika, 12(2): 153–157.

Pan, S.; Luo, L.; Wang, Y.; Chen, C.; Wang, J.; and Wu, X. 2024. Unifying Large Language Models and Knowledge Graphs: A Roadmap. IEEE Transactions on Knowledge and Data Engineering, 36(7): 3580–3599.

Qiu, J.; Han, Z.; and Huang, C. 2026. SURE-RAG: Sufficiency and Uncertainty-Aware Evidence Verification for Selective Retrieval-Augmented Generation. arXiv preprint arXiv:2605.03534.

Qwen Team. 2024. Qwen2.5 Technical Report. arXiv preprint arXiv:2412.15115.

Rayeed, S. M.; East, A.; Stevens, S.; Record, S.; and Stewart, C. V. 2025. BeetleVerse: A Study on Taxonomic Classification of Ground Beetles. arXiv preprint arXiv:2504.13393.

Rayeed, S. M.; Khurana, M.; East, A.; Fluck, I. E.; Campolongo, E. G.; Stevens, S.; Zarubiieva, I.; Lowe, S. C.; Denslow, M. W.; Donoso, E. D.; Wu, J.; Ramirez, M.; Baiser, B.; Stewart, C. V.; Mabee, P.; Berger-Wolf, T.; Karpatne, A.; Lapp, H.; Guralnick, R. P.; Taylor, G. W.; and Record, S. 2026. A Continental-Scale Dataset of Ground Beetles with High-Resolution Images and Validated Morphological Trait Measurements. Scientific Data.

Rayeed, S. M.; Ma, Q.; Stewart, C. V.; and Ma, Y. 2027. How Finely Can Vision-Language Models See Biology? Bio-VQA: A Benchmark for Taxonomic, Morphological, Spatial and Individual Reasoning. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV).

Setiawan, Y. A. 2026. Prompt-Based Abstention Fails Under Misleading Context: A Controlled Study of Small Frozen RAG Models. arXiv preprint arXiv:2608.22228.

Sharma, K.; Kumar, P.; and Li, Y. 2024. OG-RAG: Ontology-Grounded Retrieval-Augmented Generation for Large Language Models. arXiv preprint arXiv:2412.15235.

Shi, F.; Chen, X.; Misra, K.; Scales, N.; Dohan, D.; Chi, E. H.; Schärli, N.; and Zhou, D. 2023. Large Language Models Can Be Easily Distracted by Irrelevant Context. In Proceedings of the 40th International Conference on Machine Learning (ICML), 31210–31227.

Sun, J.; Xie, Y.; Shi, J.; Wang, S.; and Han, J. 2026. TaSR-RAG: Taxonomy-guided Structured Reasoning for Retrieval-Augmented Generation. arXiv preprint arXiv:2603.09341.

Team OLMo; Walsh, P.; Soldaini, L.; Groeneveld, D.; et al. 2024. 2 OLMo 2 Furious. arXiv preprint arXiv:2501.00656.

Wen, B.; Yao, J.; Feng, S.; Xu, C.; Tsvetkov, Y.; Howe, B.; and Wang, L. L. 2025. Know Your Limits: A Survey of

Abstention in Large Language Models. Transactions of the Association for Computational Linguistics, 13.

Yan, S.-Q.; Gu, J.-C.; Zhu, Y.; and Ling, Z.-H. 2024. Corrective Retrieval Augmented Generation. arXiv preprint arXiv:2401.15884.

Yoran, O.; Wolfson, T.; Ram, O.; and Berant, J. 2024. Making Retrieval-Augmented Language Models Robust to Irrelevant Context. In International Conference on Learning Representations (ICLR).

# Supplementary Material

## Contents

We organize this supplement into eleven sections, in the same order as the main paper. Every number comes from the same per-question outputs that produce the tables and figures of the main paper.

A Overview and reproducibility. We summarize the four experiments in one figure. We describe the released code and data, and we give the commands that rerun every table without a GPU and the full pipeline on a cluster.

B Data and page types. We describe how we build both graphs: PrimeKG with MONDO, and GloBI with its own taxonomy. We explain how we clean Wikipedia and query PubMed. We define the three name sets and the rule that sorts each page as named, family-only or silent. We then report the chance control, in which unrelated diseases match the same text only 0.9–1.4% of the time, and the closed-book check, which shows that a family name does not mark an easier fact.

C Statistical procedures. We define every statistic in the paper with an equation: the page efect, the diference in diferences Δ, pooling over models, the bootstrap intervals, the �-test and Bonferroni correction, the Mantel– Haenszel odds ratio, AUROC and McNemar’s test.

D Abstention questions and full results. We show how we sample the 856 questions, write each question, and draw its wrong options from the answer’s MONDO neighborhood. We give the windowed page, the prompt, and the way we read the answer from letter probabilities. We then report the full results: closed-book outcomes, all 856 questions, per-model page efects, results by relation, and results by family size.

E Page edits. We describe how we build the family and removed versions of each named page. We give per-model outcomes for all three versions in both domains, and the pooled edit efects with their intervals.

F Three fixes. We give per-model confidence on right and wrong answers, with AUROC on each page type. We follow precision across the three Qwen2.5 sizes. We give the exact instruction sentence and show how diferently each model responds to it.

G Second domain: GloBI. We describe the GloBI page types, question templates and wrong options. We give per-model outcomes closed book and with the page, and per-model page efects, where OLMo 2 shows the largest drop (−31.7 points). We explain why precision on familyonly pages falls close to chance.

H Ontology recovery. We describe how we build the givenfamily and page-built lists, with pseudo-code, and why page-built lists fail. We define each chooser, including the combined evidence score and the popularity prior, and report how often each signal fires and how accurate it is. We compare the family lists with random lists of the same length, and we show examples of recovered and missed answers.

I Worked examples. We follow single questions through the test, with real pages and each model’s answer: a confident wrong guess for moxifloxacin, a page that makes one model worse for lovastatin, and three GloBI pages that are named, family-only and silent.

J Software and compute. We list the model checkpoints and describe how we run them, batch them, cache their answers and fix every random seed.

K Choosing the family and free-text answers. We test a first fix for step 2, in which Qwen2.5-7B picks the family on the page. It raises list coverage from 19.8% to 30.0% and end-to-end recovery from 10.0% to 26.8%. We then ask 120 questions without options. No model gives the exact answer from a family-only page, and many give the family name itself.

How to read the tables. We report outcomes as percentages of questions and efects in percentage points. Square brackets give 95% bootstrap confidence intervals; an interval that excludes zero marks a reliable efect. Precision is the share of correct answers among the questions a model answers. Pooled results average the three 7-8B models, Qwen2.5-7B, OLMo 2 7B and Granite 3.1 8B, over the same questions.

A note on terms. Apage is the retrieved passage the model reads. Afamily is a parent ofthe answer up to two levels above it: a MONDO disease family, or a genus or family in GloBI. We use abstain and decline for the same outcome: the model chooses Not enough information to tell.

## A Overview and Reproducibility

This supplement gives the details behind each part of the main paper. It follows the same order as the paper: data and page types, statistics, the abstention test, page edits, the three fixes, the second domain, and ontology recovery. The last appendix covers two follow-up tests: a model that chooses the family on the page, and free-text answers. Table 7 shows where each result of the main paper is explained in more depth. Figure 6 summarizes the four experiments in one picture.

Code and data. We release all code and derived data at https://anonymous.4open.science/r/CLADE. The release holds the question sets, the windowed pages, the candidate lists, the per-question model outputs, and every script that builds them. We do not redistribute Wikipedia or PubMed text; the scripts download it again from the public APIs. PrimeKG, MONDO, GloBI, Wikipedia and PubMed are all public.

Rerunning the numbers. Every table and figure comes from the shipped per-question outputs, so a reader can check them without a GPU or network access:

bash tools/check\_results.sh # reruns the analysis and compares it # with results/; prints "identical: 22"

![](images/783a6fb107dd9d1a03763d214f21e45e223e219daca0f2e6ffe8dfcb63890a8e.jpg)  
Figure 6: Our four experiments. (1) We sort each page as named, family-only or silent, and ask each question closed book and with the page. (2) We edit named pages: the answer becomes its family name, or we delete the sentence. (3) We test three fixes. (4) We try to recover the exact answer from the ontology.

<table><tr><td>Main paper Supplement</td></tr><tr><td>Data, page types, Tables 1 and 3 Section B</td></tr><tr><td>Odds ratios, CIs, p-values, Section C</td></tr><tr><td>Eq.1 Questions, prompts, Table 4, Section D</td></tr><tr><td>Figure 2 Page edits, Figure 3 Section E</td></tr><tr><td>Confidence, scale, instruction, Section F</td></tr><tr><td>Figure 4 GloBI, Table 5 Section G</td></tr><tr><td>Recovery, Table 6, Figure 5 Section H</td></tr><tr><td>Worked examples Section I</td></tr><tr><td>Software and compute Section J</td></tr><tr><td>Choosing the family, free text Section K</td></tr></table>

Table 7: Where each part of the main paper is explained in this supplement.

The full pipeline, from raw downloads to the final tables, runs with one command on a SLURM cluster:

```shell
bash run_all.sh # PrimeKG
bash globi.sh all # GloBI
```

Both scripts resume where they stop, and a smoke test on small synthetic data checks the whole pipeline in a few minutes (-smoke).

## B Data and Page Types

This appendix explains how we build the two graphs, how we read the text, and how we decide whether a page names the answer, names only its family, or says nothing.

## B.1 Graphs and Hierarchies

PrimeKG and MONDO. PrimeKG has 8 100 498 rows. We keep three drug–disease relations: 9388 indications, 30 675 contraindications and 2568 of-label uses. Each disease has a MONDO identifier. We rebuild MONDO as a table of child–parent links: 17 801 links over 11 788 diseases. A parent with two or more children is a family; MONDO has 3182 of them. We drop parents with more than 400 descendants, such as disease or disorder. Naming them on a page tells the reader almost nothing, and they would turn almost every page into a family-only page.

GloBI and its taxonomy. We use the archived GloBI dump (Zenodo record 4460654), which holds 7 822 670 interaction records. We keep four relations: eats, preys on, parasite of and pollinates. Together they hold 3 210 597 records. We keep a record only when both sides are species, written as a two-word Latin name (“Genus epithet”). We drop names with “sp.”, hybrids and numbers, because a question about a genus has no single answer. This leaves 457 386 species-to-species edges:

<table><tr><td>Relation</td><td>Edges</td><td>Source species</td></tr><tr><td>eats</td><td>235 960</td><td>25 637</td></tr><tr><td>parasite of</td><td>117894</td><td>31 432</td></tr><tr><td>preys on</td><td>87678</td><td>5138</td></tr><tr><td>pollinates</td><td>15 854</td><td>2818</td></tr></table>

GloBI records give a taxonomic path for each taxon (species, genus, family, order, class). We build the hierarchy from these paths. Diferent records can give a taxon diferent parents, so we keep the parent seen most often. This makes the table a tree: climbing it never moves sideways. The result has 77 852 child–parent links, and 8370 genera have two or more species. As in MONDO, the median family has three members.

## B.2 Text Sources

Wikipedia. For each drug, and for each consumer species in GloBI, we fetch the English Wikipedia article through the MediaWiki API. We turn the raw markup into plain prose. We remove comments, reference tags, templates (including infoboxes and citation blocks, with nested braces), tables, and file and category links. We replace each internal link with its visible text and remove bold and italic marks, HTML tags and character entities. We keep section headings as plain lines, so the text reads like the page a person sees. For GloBI we skip stubs shorter than 3000 characters.

PubMed. For each drug we retrieve PubMed abstracts through NCBI E-utilities. We split the query budget equally between a treatment query and an adverse-efect or contraindication query, so both relations get the same chance to appear. The drug’s full text is its Wikipedia article followed by its abstracts.

## B.3 How a Page Names a Fact

Three name sets. Each answer has three nested sets of names:

• the exact name: its primary MONDO label, or its Latin name in GloBI;

• alternative names: titles of Wikipedia pages that redirect to the answer’s article, which bring in synonyms, lay terms and abbreviations; in GloBI we also add the abbreviated form (A. plantaginifolia);

• family names: the labels of the answer’s ancestors up to two levels above it (the MONDO parent and grandparent, or the genus and family).

<table><tr><td></td><td colspan="2">True disease</td><td colspan="2">Unrelated diseases</td></tr><tr><td>Relation</td><td></td><td>named + family</td><td>named</td><td>+ family</td></tr><tr><td>Indication</td><td>19.3</td><td>34.7</td><td>1.4</td><td>2.7</td></tr><tr><td>Contraindication</td><td>6.7</td><td>9.3</td><td>0.9</td><td>1.8</td></tr><tr><td>Off-label use</td><td>11.8</td><td>23.7</td><td>0.9</td><td>3.3</td></tr></table>

Table 8: Share of facts (%) whose Wikipedia text matches the true disease or one of three unrelated diseases from the same relation (600 facts per relation; 1800 unrelated draws each). “+ family” also counts family names.

Page type. Let � be the text, $N ( a )$ the exact and alternative names of answer �, and $A _ { 2 } ( a )$ its family names. We assign each fact to the finest level the text matches:

$$
\operatorname { t y p e } ( T , a ) = { \left\{ \begin{array} { l l } { \operatorname { n a m e d } } & { { \mathrm { i f ~ } } T { \mathrm { ~ m a t c h e s ~ } } N ( a ) , } \\ { \operatorname { f a m i l y - o n l y } } & { { \mathrm { e l s e ~ i f ~ } } T { \mathrm { ~ m a t c h e s ~ } } A _ { 2 } ( a ) , } \\ { \operatorname { s i l e n t } } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{2}
$$

Matching rule. A name matches when it appears on word boundaries, ignoring case, with an optional trailing s for simple plurals. A multi-word name matches across any run of white space. We ignore names shorter than four characters, because short acronyms match by accident.

Chance control. A family name is broader than a disease name, so it can match unrelated text more easily. To measure this, we pair each fact with three diseases from the same relation that have no link to the drug, and we run the same matcher on the same text. Table 8 compares the true disease with these unrelated ones on 1,800 PrimeKG facts in Wikipedia. The unrelated diseases match 0.9–1.4% of the time at the named level and 1.8–3.3% once family names are added. The true disease matches 6.7–19.3% and 9.3–34.7%. Matches therefore follow the fact, not the size of the name set.

## B.4 Text Coverage and Closed-Book Knowledge

The four-box table. Table 2 in the main paper uses 1,800 PrimeKG facts (600 per relation). A fact counts as text names it when the drug’s Wikipedia article contains the exact or an alternative name. A family-only page counts as text does not name it. To test closed-book knowledge, Qwen2.5-0.5B scores the true disease against four random diseases from the same relation. It reads a short stem, such as “X is a drug used to treat the disease called”, and we take the mean logprobability of each disease name after it. The highest score is the model’s pick.

Knowledge above chance. On facts the text does not name, the model picks the true disease 25.1% of the time. With five options, chance is 20%. We correct for guessing with

$$
k = { \frac { \operatorname { a c c } - c } { 1 - c } } = { \frac { 0 . 2 5 1 - 0 . 2 0 0 } { 1 - 0 . 2 0 0 } } \approx 6 . 4 \% ,\tag{3}
$$

where � is the chance rate. So the model truly knows only about 6% of these facts; the rest of its correct picks are lucky guesses.

<table><tr><td>Relation</td><td></td><td>Named Family only</td><td>Silent</td></tr><tr><td>Indication</td><td>84.2 (101)</td><td>44.7 (94)</td><td>44.0 (405)</td></tr><tr><td>Contraind.</td><td>32.5 (40)</td><td>18.8 (16)</td><td>23.4 (544)</td></tr><tr><td>Off-label</td><td>57.1 (63)</td><td>37.5 (72)</td><td>31.6 (465)</td></tr><tr><td>All</td><td>65.7 (204)</td><td>39.6 (182)</td><td>32.0 (1,414)</td></tr></table>

Table 9: Closed-book accuracy of Qwen2.5-7B by what the drug’s Wikipedia page says about the answer. Chance is 20%.

Does a family name mean an easier fact? We test whether facts written at the family level are simply easier. Qwen2.5- 7B answers the same 1,800 facts closed book (Table 9).

Within each relation we build a $2 \times 2$ table (family-only vs. named, or family-only vs. silent; correct vs. wrong) and pool the three tables with the Mantel–Haenszel odds ratio (Section C). Family-only facts are much harder than named facts (OR = 0.27 [0.17, 0.43]) and as hard as silent facts (OR = 1.10 [0.80, 1.53]). An exact name in the text goes with 3.7 times higher odds that the model knows the fact; a family name goes with none.

## C Statistical Procedures

This appendix defines every statistic in the paper. All tests are two-sided, and all code uses only the Python standard library, with random seed 0.

Page efect. For a question � and an outcome $o \in$ {correct, wrong, decline}, let $y _ { q } ^ { o , \mathrm { p a g e } } = 1$ if the model’s answer has outcome � when it reads the page, and 0 otherwise. $\operatorname { L e t } y _ { q } ^ { o , \mathrm { c l o s e d } }$ be the same without the page. The page efect of $q$ is

$$
d _ { q } ^ { o } = y _ { q } ^ { o , \mathrm { p a g e } } - y _ { q } ^ { o , \mathrm { c l o s e d } } \in \{ - 1 , 0 , 1 \} .\tag{4}
$$

A page efect of +1 for declines means the page turned an answer into a decline; −1 means the page turned a decline into an answer. Our main measure (Eq. 1) is the mean page efect on family-only questions $F$ minus that on silent questions �:

$$
\Delta ^ { o } = \bar { d } _ { F } ^ { o } - \bar { d } _ { S } ^ { o } , \qquad \bar { d } _ { G } ^ { o } = \frac { 1 } { | G | } \sum _ { q \in G } d _ { q } ^ { o } .\tag{5}
$$

This is a diference in diferences. The first diference (page minus closed book) removes how hard each fact is on its own. The second (family-only minus silent) asks what the family name adds beyond a page that says nothing.

Pooling models. For the pooled 7–8B result, we first average each question’s page efect over Qwen2.5-7B, OLMo 2 7B and Granite 3.1 8B, and then compute $\Delta ^ { o }$ on these averages. Each of the 481 questions thus counts once, however many models answer it.

Confidence intervals. We draw 5,000 bootstrap samples. Each sample redraws the questions of each group with replacement, keeping the group sizes, and recomputes $\Delta ^ { o }$ . The 95% CI is the 2.5th to 97.5th percentile of these values. If the interval excludes zero, we call the efect reliable.

�-values. We test $\Delta ^ { o } = 0$ with a Welch-type �-test, which allows the two groups to have diferent variances:

$$
z = \frac { \bar { d } _ { F } ^ { o } - \bar { d } _ { S } ^ { o } } { \sqrt { s _ { F } ^ { 2 } / | F | + s _ { S } ^ { 2 } / | S | } } ,\tag{6}
$$

where $s _ { G } ^ { 2 }$ is the sample variance of $d _ { q } ^ { o }$ in group �. The �-value is $2 \left( 1 - \Phi ( | z | ) \right)$ , with Φ the standard normal distribution function. A small � means a gap this large is unlikely if the family name had no efect. Each comparison covers three outcomes, so the Bonferroni-corrected �-value is three times the raw one.

Odds ratios. For the closed-book check in Section B, each relation � gives a $2 \times 2$ table: $a _ { r }$ and $b _ { r }$ are correct and wrong answers in the first group, $c _ { r }$ and $e _ { r }$ in the second, and $n _ { r }$ is the total. The Mantel–Haenszel odds ratio pools the relations:

$$
\mathrm { O R } _ { \mathrm { M H } } = \frac { \sum _ { r } a _ { r } e _ { r } / n _ { r } } { \sum _ { r } b _ { r } c _ { r } / n _ { r } } .\tag{7}
$$

OR = 1 means the two groups are answered correctly equally often. $\mathrm { O R } = 0 . 2 7$ means the odds of a correct answer in the first group are about a quarter of those in the second. Pooling within relations keeps the result from being driven by one relation that happens to be easier.

AUROC. To judge confidence filtering, we take the answers a model gives (not its declines) and ask how well the probability of the chosen option ranks correct answers above wrong ones. AUROC is the chance that a random correct answer has higher confidence than a random wrong answer. An AUROC of 1.0 means a threshold can separate them perfectly; 0.5 means confidence carries no signal.

McNemar’s test. In recovery, two choosers answer the same 410 facts. Let � be the number of facts the first gets right and the second gets wrong, and � the reverse. McNemar’s test uses only these disagreements, with a continuity correction:

$$
\chi ^ { 2 } = \frac { ( | b - c | - 1 ) ^ { 2 } } { b + c } ,\tag{8}
$$

compared with a $\chi ^ { 2 }$ distribution with one degree of freedom. It asks whether one chooser wins the disagreements more often than chance would allow.

## D Abstention Questions and Full Results

This appendix shows how we build each question, how the model sees it, and how we read its answer. It then gives the full results behind Table 4 and Figure 2 of the main paper.

## D.1 Building the Questions

Sampling. We sort PrimeKG facts into six cells: three page types (named, family-only, silent) times two relations (indication, contraindication). We shufle each cell with seed 0 and take facts in order until the cell has 150 questions. We skip a fact when the drug has no text, when we cannot draw four wrong options, or when the 1,800-character page no longer shows the matched name or family. Only 106 familyonly contraindications pass these checks, so we use all of them. This gives 856 questions.

<table><tr><td>Page type</td><td>Relation</td><td>n</td><td>All-neighbor (%)</td></tr><tr><td>Named</td><td>both</td><td>300</td><td>35.3</td></tr><tr><td>Family-only</td><td>contraindication</td><td>106</td><td>83.0</td></tr><tr><td>Family-only</td><td>indication</td><td>150</td><td>84.7</td></tr><tr><td>Silent</td><td>contraindication</td><td>150</td><td>45.3</td></tr><tr><td>Silent</td><td>indication</td><td>150</td><td>61.3</td></tr></table>

Table 10: Questions per group and the share whose four wrong options are all siblings or cousins of the answer in MONDO.

Question templates. Indication questions ask: Which disease is X used to treat? Contraindication questions ask: In patients with which disease must X not be given?

Wrong options. Each question has five diseases and the option Not enough information to tell, labeled A to F. We place the true disease and the decline option at random letters. We draw the four wrong options in this order:

1. siblings: other children of the answer’s MONDO parent;

2. cousins: other grandchildren of the answer’s grandparent;

3. random diseases from the same relation, only if siblings and cousins run out.

We never use an ancestor or descendant of the answer, or any other disease the drug has under the same relation in PrimeKG, since that option would also be correct. Because siblings and cousins share the answer’s family, a page that names only the family cannot rule them out. Random diseases, however, can be ruled out by the family name alone, which makes a question easier. Table 10 shows that familyonly questions more often have four neighborhood options than the other groups. This is why the main analysis keeps only questions whose four wrong options all come from the neighborhood (106 named, 215 family-only, 160 silent; 481 in all).

Windowed pages. A full drug article is too long for a prompt, so we cut a window around the evidence. The page starts with the first 800 characters of the drug’s cleaned text. We then add a 280-character window on each side of every mention of the answer or its family. We merge windows that overlap, join them with ellipses, and stop at 1,800 characters. A silent page has no mentions, so it is just the opening of the article. Every model sees the same page for the same question.

Prompt. Each prompt is one user turn in the model’s own chat template. The closed-book prompt is the same without the passage block:

Passage:   
<passage>   
Question: <question>   
Options:   
A. <option>   
F. <option>   
Answer with the letter of one option   
only.

Reading the answer. We do not generate text. We read the probability of the next token for each letter. For letter ℓ, let $p ( \ell )$ be the larger of the probabilities of “ℓ” and “ ℓ” (with a leading space). We renormalize over the six letters:

$$
\pi ( \ell ) = \frac { p ( \ell ) } { \sum _ { m \in \{ A , \ldots , F \} } p ( m ) } ,\tag{9}
$$

and the model’s answer is the letter with the highest �. Its confidence is that �. If the answer is Not enough information to tell, the question counts as a decline. Precision is the share of correct answers among the questions the model does not decline.

## D.2 Full Results

Without a page. Table 11 gives closed-book outcomes on the 481 matched questions. Family-only questions already draw more wrong answers and fewer declines than silent ones in every 7–8B model. For example, Qwen2.5-7B declines 32.1% of family-only questions and 59.4% of silent ones before it reads anything. If we compared pages directly, part of the gap would come from these facts, not from the page. The page efect removes this part.

<table><tr><td>Model</td><td>Group</td><td>Correct Wrong</td><td></td><td>Declines</td></tr><tr><td>Qwen2.5 7B</td><td>named family-only silent</td><td>60.4 35.3 25.0</td><td>17.9 32.6 15.6</td><td>21.7 32.1 59.4</td></tr><tr><td>OLMo 2 7B</td><td>named family-only silent</td><td>65.1 36.7 37.5</td><td>31.1 56.7 51.2</td><td>3.8 6.5 11.2</td></tr><tr><td>Granite 3.1 8B</td><td>named family-only silent</td><td>50.9 33.0 26.2</td><td>24.5 33.0 26.2</td><td>24.5 34.0 47.5</td></tr><tr><td>Qwen2.5 0.5B</td><td>named family-only silent</td><td>39.6 30.2 28.7</td><td>56.6 69.8 68.8</td><td>3.8 0.0 2.5</td></tr></table>

Table 11: Closed-book outcomes (%) on the 481 neighborhood-matched questions (106 named, 215 familyonly, and 160 silent) when no passage is provided.

With a page, all questions. Table 12 repeats Table 4 on all 856 questions, without the neighborhood filter. The pattern is the same: family-only pages draw far more wrong answers than silent pages, with lower precision. Compared directly with the page, family-only pages give more wrong answers than silent pages by 15.6, 20.3 and 16.5 points for Qwen2.5- 7B, OLMo 2 and Granite $( p = 3 \times 1 0 ^ { - 6 } , \overset { \cdot } { 5 } \times 1 0 ^ { - 5 } , 6 \times 1 0 ^ { - 5 } ) .$ and fewer declines by 28.4, 26.9 and 31.7 points $( p < 1 0 ^ { - 7 }$ each).

Page efects per model. Table 13 breaks the page efect into its parts. For each outcome it shows the mean page efect on family-only questions, the same on silent questions, and their diference with its CI and �-value. The pooled row is the result in the main paper. OLMo 2 and Granite each show a large drop in declines; Qwen2.5-7B shows none, because it already declines family-only questions much less often closed book (Table 11).

<table><tr><td>Model</td><td>Passage</td><td>Corr.</td><td>Wrong</td><td>Decl. Prec.</td><td></td><td>Conf.</td></tr><tr><td>Qwen2.5 7B</td><td>named family-only silent</td><td>64.7 29.3 13.0</td><td>3.3 17.6 2.7</td><td>32.0 53.1 84.3</td><td>95.1 62.5 83.0</td><td>0.99 1.00 0.99</td></tr><tr><td>OLMo 2 7B</td><td>named family-only silent</td><td>71.3 39.1 32.0</td><td>20.7 42.6 28.0</td><td>8.0 18.4 40.0</td><td>77.5 47.8 53.3</td><td>0.80 0.84 0.76</td></tr><tr><td>Granite 3.1 8B</td><td>named family-only silent</td><td>64.3 33.2 15.7</td><td>4.7 25.0 11.0</td><td>31.0 41.8 73.3</td><td>93.2 57.0 58.8</td><td>0.67 0.93 0.81</td></tr><tr><td>Qwen2.5 0.5B</td><td>named family-only silent</td><td>78.7 37.5 28.7</td><td>14.3 52.3 50.0</td><td>7.0 10.2 21.3</td><td>84.6 41.7 36.4</td><td>0.43 0.55 0.39</td></tr></table>

Table 12: Open-book outcomes across all 856 PrimeKG questions (%; 300 named, 256 family-only, and 300 silent). Precision is the share of correct answers among nonabstained questions; Conf. is the median probability assigned to wrong answers.

By relation. Table 14 splits wrong answers with the page by relation. Family-only pages give more wrong answers than silent pages for both relations in every 7–8B model. For indications, Qwen2.5-7B is wrong on 26.0% of family-only questions and 2.2% of silent ones. Models are more careful on contraindications and decline more of them, but the gap stays. Qwen2.5-0.5B rarely declines at all (8–21% with a page, under 4% closed book), so its wrong-answer rates are high everywhere.

Family size. Table 15 splits the family-only questions into thirds by the size of the named family. The pooled drop in declines is about 17–18 points for families of up to 20 members and only 6.5 points, not significant, for larger families. A short list looks answerable to the model; a long one does not.

## E Page Edits

The abstention test compares diferent facts: the family-only questions are not the same facts as the silent ones. The page edits remove this diference. They ask each question three times, with three versions of one page, and change nothing else.

Which questions. We start from named questions, whose page names the answer. In PrimeKG we keep the 130 whose answer has a MONDO parent, since the family edit needs a parent name. In GloBI we keep the 255 named questions whose answer has a genus.

## Three versions.

• Original: the page as retrieved. It names the answer.

• Family: every mention of the answer, and of each of its alternative names, becomes the name of its immediate parent (the MONDO parent, or the genus). In GloBI we also replace the abbreviated form (A. plantaginifolia).

<table><tr><td></td><td colspan="4">Declines (Abstention)</td><td colspan="3">Correct Answers</td><td colspan="3">Wrong Answers</td></tr><tr><td>Model</td><td>Fam.</td><td>Sil.</td><td>Diff. [95% CI], p</td><td>Fam.</td><td>Sil.</td><td>Diff. [95% CI], p</td><td>Fam.</td><td>Sil.</td><td></td><td>Diff. [95% CI], p</td></tr><tr><td>Qwen2.5 7B</td><td>+23.3</td><td>+24.4</td><td>-1.1 [–10.5, 8.4], .82</td><td></td><td>-8.8 -11.2</td><td></td><td>+2.4 [–5.7, 10.5], .57</td><td>-14.4</td><td>-13.1</td><td>–1.3 [–9.4, 7.1], .76</td></tr><tr><td>OLMo 2 7B</td><td>+11.6</td><td>+33.8</td><td>-22.1 [-31.8, −12.6], 6e-6</td><td></td><td>+0.5 -6.9</td><td></td><td>+7.3 [–2.6, 17.0], .15</td><td>-12.1</td><td>-26.9</td><td>+14.8 [3.6, 26.1], .009</td></tr><tr><td>Granite 3.1 8B</td><td>+9.3</td><td>+27.5</td><td>-18.2[-28.5, -8.1], .0006</td><td>-2.8</td><td>-11.2</td><td></td><td>+8.5 [–0.9, 17.8], .064</td><td>-6.5</td><td>-16.2</td><td>+9.7 [–0.1, 19.9], .056</td></tr><tr><td>Qwen2.5 0.5B</td><td>+9.8</td><td>+18.1</td><td>–8.4 [–15.7, –1.1], .027</td><td>+4.2</td><td>0.0</td><td></td><td>+4.2 [–6.1, 14.4], .42</td><td>-14.0-18.1</td><td></td><td>+4.2 [–6.4, 14.6], .44</td></tr><tr><td>Pooled 7–8B</td><td></td><td>+14.7+28.5</td><td>-13.8 [–20.3, -7.5], 4e-5</td><td>-3.7</td><td>-9.8</td><td></td><td>+6.1 [–0.1, 12.4], .058</td><td>-11.0-18.8</td><td></td><td>+7.7 [1.2, 14.0], .019</td></tr></table>

Table 13: Within-question page efects (in percentage points): the change in each outcome when adding a passage relative to closed-book answering on family-only (Fam., � = 215) and silent (Sil., � = 160) neighborhood-matched questions, along with the diference-in-diferences estimate.

<table><tr><td rowspan="2">Model</td><td colspan="2">Indication</td><td colspan="2">Contraindication</td></tr><tr><td>Family-only</td><td>Silent</td><td>Family-only Silent</td><td></td></tr><tr><td>Qwen2.5 7B</td><td>26.0</td><td>2.2</td><td>6.8</td><td>2.9</td></tr><tr><td>OLMo 2 7B</td><td>38.6</td><td>21.7</td><td>53.4</td><td>27.9</td></tr><tr><td>Granite 3.1 8B</td><td>37.8</td><td>16.3</td><td>10.2</td><td>1.5</td></tr><tr><td>Qwen2.5 0.5B</td><td>46.5</td><td>46.7</td><td>69.3</td><td>55.9</td></tr></table>

Table 14: Open-book wrong-answer rates (%) broken down by clinical relation on neighborhood-matched questions (indication: 127 family-only, 92 silent; contraindication: 88 family-only, 68 silent).
<table><tr><td>Family size</td><td>n</td><td>Median Decline effect [95% CI]</td><td>p</td></tr><tr><td>≤ 10</td><td>71</td><td>7 −17.3 [-26.2, -8.2]</td><td>.001</td></tr><tr><td>11-20</td><td>69</td><td>16 -18.4 [-26.6, -10.3]</td><td>&lt;.001</td></tr><tr><td>&gt; 20</td><td>68</td><td> $4 4 \ - 6 . 5 [ - 1 5 . 7 , + 3 . 0 ]$ </td><td>.174</td></tr></table>

Table 15: Decline efect (Δ for declines, points) by the number of members in the named family, pooled over the three 7–8B models.

• Removed: every sentence that mentions the answer is deleted.

We replace whole words only. If the answer has several parents, we use the shortest name, and we never use a parent that is itself one of the options. We drop a question if the answer is still mentioned after the edit, or if nothing is left after removal. For example, a page that says “. . . used to treat familial hyperlipidemia” becomes “. . . used to treat hyperlipidemia” in the family version, and loses that sentence in the removed version. The question, the options, their letters and the model stay the same. Each prompt is cached under a hash of its exact text, so an edited page can never reuse the answer of the original.

What the comparison shows. The family and removed versions both lack the answer. The only diference is that the family version keeps the family name in place of the answer. If the family name had no efect, the two versions would give the same outcomes. Table 16 shows that they do not: in every model and both domains, the family version draws fewer declines than the removed version.

<table><tr><td>Model</td><td>Version</td><td>Decl. Corr.</td><td></td><td>Wrong</td><td>Prec.</td></tr><tr><td>PrimeKG (130 questions)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen2.5-7B</td><td>original</td><td>27.7</td><td>66.9</td><td>5.4</td><td>92.6</td></tr><tr><td rowspan="4">OLMo 2 7B</td><td>family</td><td>43.8</td><td>46.2</td><td>10.0</td><td>82.2</td></tr><tr><td>removed</td><td>70.0</td><td>26.9</td><td>3.1</td><td>89.7</td></tr><tr><td>original</td><td>6.2</td><td>74.6</td><td>19.2</td><td>79.5</td></tr><tr><td>family</td><td>7.7</td><td>58.5</td><td>33.8</td><td>63.3</td></tr><tr><td rowspan="4">Granite 3.1 8B</td><td>removed</td><td>18.5</td><td>52.3</td><td>29.2</td><td>64.2</td></tr><tr><td>3 original</td><td>23.8</td><td>69.2</td><td>6.9</td><td>90.9</td></tr><tr><td>family</td><td>40.8</td><td>43.1</td><td>16.2</td><td>72.7</td></tr><tr><td>removed</td><td>56.2</td><td>31.5</td><td>12.3</td><td>71.9</td></tr><tr><td colspan="8">GloBI (255 questions)</td></tr><tr><td rowspan="4">Qwen2.5-7B</td><td>original</td><td>38.4</td><td>58.8</td><td>2.7</td><td>95.5</td></tr><tr><td>family</td><td>71.0</td><td>21.6</td><td>7.5</td><td>74.3</td></tr><tr><td>removed</td><td>92.2</td><td>5.1</td><td>2.7</td><td>65.0</td></tr><tr><td>original</td><td>17.3</td><td>71.4</td><td>11.4</td><td>86.3</td></tr><tr><td rowspan="4">Granite 3.1 8B</td><td>family</td><td>25.9</td><td>44.3</td><td>29.8</td><td>59.8</td></tr><tr><td>removed</td><td>56.5</td><td>20.8</td><td>22.7</td><td>47.7</td></tr><tr><td>original</td><td>32.2</td><td>61.2</td><td>6.7</td><td>90.2</td></tr><tr><td>family</td><td>50.2</td><td>32.9</td><td>16.9</td><td>66.1</td></tr><tr><td></td><td>removed</td><td>81.6</td><td>9.8</td><td>8.6</td><td>53.2</td></tr></table>

Table 16: Outcomes (%) with each version of the page. In every model and domain, the family version draws fewer declines than the removed version.

Pooled efects. Table 17 pools the three models, averaging each question over models first. Against the removed version, the family name cuts declines by 17.4 points in PrimeKG and 27.7 in GloBI, and raises both wrong and correct answers. Against the original, the family version loses many correct answers and gains wrong ones. So the family name lets the model guess among the family members without telling it which one is meant.

## F Three Fixes

This appendix gives the per-model numbers behind Figure 4 of the main paper.

Confidence. A simple guard is to answer only when the model is confident. Table 18 shows, for each model and page type, how many questions the model answers, its precision, its mean confidence on right and on wrong answers, and the

<table><tr><td></td><td>Outcome</td><td>Family – removed</td><td>Family – original</td></tr><tr><td rowspan="3">PrimeKG</td><td>Declines</td><td>-17.4 [-23.6, -11.5]</td><td>+11.5 [+7.4, +15.9]</td></tr><tr><td>Wrong</td><td>+5.1 [+0.5, +10.3]</td><td>+9.5 [+5.9, +13.6]</td></tr><tr><td>Correct</td><td>+12.3 [+6.9, +17.9]</td><td>-21.0 [-26.9, -15.6]</td></tr><tr><td rowspan="3">GloBI</td><td>Declines</td><td>-27.7 [–32.2, -23.5]</td><td>+19.7 [+15.4, +23.9]</td></tr><tr><td>Wrong</td><td>+6.7 [+3.4, +10.2]</td><td>+11.1 [+8.2, +14.1]</td></tr><tr><td>Correct</td><td>+21.0 [+16.9, +25.4] -30.8[-36.2, -25.5]</td><td></td></tr></table>

Table 17: Pooled edit efects over the three 7–8B models (points, 95% bootstrap CI).
<table><tr><td>Model</td><td>Page</td><td colspan="5">Ans. Prec. Conf. R Conf. W AUROC</td></tr><tr><td>PrimeKG</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen2.5-0.5B named</td><td></td><td></td><td>97 84.5</td><td>.83</td><td>.49</td><td>.85</td></tr><tr><td>Qwen2.5-7B</td><td>family</td><td>194</td><td>38.1</td><td>.71</td><td>.60</td><td>.64</td></tr><tr><td></td><td>named</td><td>80</td><td>95.0</td><td>.98</td><td>1.00</td><td>.73</td></tr><tr><td>OLMo 2 7B</td><td>family</td><td>96</td><td>59.4</td><td>.98</td><td>.90</td><td>.72</td></tr><tr><td></td><td>named</td><td>98</td><td>83.7</td><td>.97</td><td>.75</td><td>.86</td></tr><tr><td>Granite 3.1 8B named</td><td>family</td><td>176</td><td>45.5</td><td>.89</td><td>.77</td><td>.71</td></tr><tr><td></td><td></td><td>85</td><td>90.6</td><td>.95</td><td>.71</td><td>.87</td></tr><tr><td>Qwen2.5-14B</td><td>family</td><td>122</td><td>53.3</td><td>.90</td><td>.82</td><td>.66</td></tr><tr><td></td><td>named</td><td>75</td><td>96.0</td><td>.99</td><td>.92</td><td>.77</td></tr><tr><td></td><td>family</td><td>89</td><td>58.4</td><td>.98</td><td>.99</td><td>.59</td></tr><tr><td>GloBI</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen2.5-7B</td><td>named</td><td>1167</td><td>94.6</td><td>.98</td><td>.90</td><td>.82</td></tr><tr><td>OLMo 2 7B</td><td>family</td><td>18</td><td>27.8</td><td>.94</td><td>.89</td><td>.55</td></tr><tr><td></td><td>named</td><td>231</td><td>84.4</td><td>.94</td><td>.69</td><td>.86</td></tr><tr><td></td><td>family</td><td>138</td><td>26.8</td><td>.76</td><td>.62</td><td>.68</td></tr><tr><td>Granite 3.1 8B named</td><td></td><td>188</td><td>89.4</td><td>.94</td><td>.75</td><td>.86</td></tr><tr><td></td><td>family</td><td>46</td><td>21.7</td><td>.75</td><td>.74</td><td>.52</td></tr></table>

Table 18: Confidence of answers with the page. Ans. is the number of questions answered (not declined); Conf. R and Conf. W are the mean confidence of right and wrong answers.

AUROC (Section C). On named pages, right answers carry clearly more confidence than wrong ones, and AUROC is 0.73–0.87. On family-only pages the two are close, and AU-ROC drops to 0.59–0.72. The wrong guesses are confident: Qwen2.5-14B puts a mean probability of 0.99 on its wrong family-only answers. In GloBI the drop is larger, from 0.82– 0.86 on named pages to 0.52–0.68.

Scale. Within Qwen2.5, a larger model is more precise on named pages: 84.5% at 0.5B, 95.0% at 7B and 96.0% at 14B. On family-only pages, precision rises from 38.1% to 59.4% between 0.5B and 7B and then stops: the 14B model reaches 58.4%. The 14B model gives 14.7 points more wrong answers on family-only pages than on silent ones $( p = \bar { 6 } { \times } 1 0 ^ { - 6 } )$ . Like the 7B model, it already declines familyonly questions less often closed book, so its page efect does not difer between the groups (+5.3 points, � = 0.29). A larger model reads named pages better but still guesses from a family name.

Instruction. We add one sentence to the prompt with the page:

If the passage names only a broad category of diseases and does not say which specific disease is meant, answer with the letter of“Not enough information to tell.

<table><tr><td colspan="3">Declines</td><td colspan="2">Precision</td></tr><tr><td>Model</td><td>Page</td><td>base instr.</td><td>base</td><td>instr.</td></tr><tr><td>Qwen2.5-7B</td><td>named family</td><td>24.5 29.2 55.4 64.2</td><td>95.0 59.4</td><td>94.7 64.9</td></tr><tr><td>OLMo 2 7B</td><td>silent named family</td><td>83.8 7.5 18.1</td><td>88.1 84.6 27.4 83.7 56.3 45.5 47.9</td><td>89.5 89.6</td></tr><tr><td>Granite 3.1 8B named</td><td>silent family silent</td><td>45.0 19.8 43.3 75.0</td><td>85.6 55.7 22.6 90.6 41.9 53.3 53.6 71.9 60.0 55.6</td><td>60.9 91.5</td></tr></table>

Table 19: Declines and precision (%) with the page, without (base) and with (instr.) the instruction to decline on a broad category.

Table 19 shows the result. The models react very diferently. OLMo 2 declines far more on every page type, including named pages (from 7.5% to 27.4%). Granite barely reacts. Qwen2.5-7B declines somewhat more. Pooled over the three models, the instruction raises declines by 15.2 points on family-only pages, but also by 14.0 on silent pages and 9.1 on named pages, where it costs 6.3 points of correct answers. Precision on family-only pages stays low (47.9–64.9%). The instruction makes the models more cautious everywhere, but it does not teach them to tell a family name from an answer.

## G Second Domain: GloBI

We repeat the abstention test and the page edits on species interactions. This tests whether Clade depends on medicine, on MONDO, or on PrimeKG. It does not: the same pattern appears with a diferent graph, a diferent hierarchy and different text.

## G.1 Setup

Facts and pages. Each fact links a consumer species to a partner species, such as a butterfly and the plant its larvae eat. The page is the consumer’s Wikipedia article, cleaned as in Section B. A page is named if it gives the partner’s Latin name, its abbreviated form (A. plantaginifolia) or a common name from Wikipedia redirects. It is family-only if it gives only the partner’s genus or family, and silent if it gives neither. We measure 23 694 edges over 5425 articles. Pages name the partner for only 3.9% of edges and name only its genus or family for 7.1% (Table 3). Family-only pages are thus about twice as common as named ones.

Questions. We use one template per relation:
<table><tr><td>Relation</td><td>Question</td></tr><tr><td>eats</td><td>Which species does X feed on?</td></tr><tr><td>preys on</td><td>Which species does X prey on?</td></tr><tr><td>parasite of</td><td>Which species is X a parasite of?</td></tr><tr><td>pollinates</td><td>Which plant does X pollinate?</td></tr></table>

We take 75 questions per page type and relation, for 900 in all. Wrong options are other species of the partner’s genus first, then of its family, and never another recorded partner of the consumer. For binomial answers, we fill any remaining slots with species from other genera, never with a bare genus name, so every option is a species. Most questions have four neighborhood options (833 of 900; Table 20), and the main analysis keeps only these.

<table><tr><td>Page type</td><td>eats</td><td>parasite of</td><td>pollinates</td><td>preys on</td></tr><tr><td>Named</td><td>93.3</td><td>94.7</td><td>94.7</td><td>89.3</td></tr><tr><td>Family-only</td><td>100.0</td><td>98.7</td><td>100.0</td><td>93.3</td></tr><tr><td>Silent</td><td>90.7</td><td>85.3</td><td>92.0</td><td>78.7</td></tr></table>

Table 20: Share of GloBI questions (%, 75 per cell) whose four wrong options are all from the partner’s genus or family.
<table><tr><td></td><td></td><td colspan="2">Correct</td><td colspan="2">Wrong</td><td colspan="2">Declines</td></tr><tr><td>Model</td><td>Page</td><td></td><td>cl. page</td><td></td><td>cl. page</td><td></td><td>cl. page</td></tr><tr><td>Qwen2.5-7B</td><td>named</td><td>7.9</td><td>56.6</td><td>4.3</td><td>3.2</td><td>87.8</td><td>40.1</td></tr><tr><td></td><td>family</td><td>1.7</td><td>1.7</td><td>2.7</td><td>4.4</td><td>95.6</td><td>93.9</td></tr><tr><td>OLMo 2 7B</td><td>silent</td><td>3.5</td><td>0.4</td><td>1.2</td><td></td><td>1.2 95.4 98.5</td><td></td></tr><tr><td></td><td>named</td><td>20.8</td><td>69.924.7</td><td></td><td></td><td>12.9 54.517.2</td><td></td></tr><tr><td></td><td>family</td><td>10.9</td><td>12.622.8</td><td></td><td></td><td>34.4 66.3 53.1</td><td></td></tr><tr><td></td><td>silent</td><td>8.5</td><td>3.5 22.3</td><td></td><td></td><td>8.8 69.2 87.7</td><td></td></tr><tr><td>Granite 3.1 8B</td><td>named</td><td>12.2</td><td>60.2 16.8</td><td></td><td></td><td>7.2 71.032.6</td><td></td></tr><tr><td></td><td>family</td><td>9.2</td><td></td><td>3.4 14.6</td><td></td><td>12.2 76.2 84.4</td><td></td></tr><tr><td></td><td>silent</td><td>5.8</td><td></td><td>1.512.3</td><td></td><td>1.9 81.996.5</td><td></td></tr></table>

Table 21: GloBI outcomes (%) closed book (cl.) and with the page, on questions whose four wrong options all come from the neighborhood (279 named, 294 family-only, 260 silent).

## G.2 Results per Model

Outcomes. Table 21 gives outcomes for each model, closed book and with the page. Closed book, all three models decline most species questions; they know few of these facts. A named page lets them answer, and their answers are precise (84–95%). A family-only page also lets them answer, but most of these answers are wrong: precision is 22–28%, close to the 20% chance rate of a five-way guess. A silent page makes them decline almost everything.

Page efects. Table 22 gives the page efect for each model. The efect is largest in OLMo 2, where a silent page raises declines by 18.5 points but a family-only page lowers them by 13.3, a diference of 31.7 points. Qwen2.5-7B declines almost every question on either page, so its efect is small but still significant. Granite’s decline efect is not significant alone, but its wrong answers rise reliably. Pooled over the three models, Δ = −14.3 points for declines and +11.6 for wrong answers, with only +2.7 for correct answers: in GloBI, the answers that the family name frees are mostly wrong.

Page edits and confidence. The per-model edit results are in Table 16 and the confidence results in Table 18. In every model, the genus name in place of the answer draws fewer declines than deleting the sentence, and AUROC on familyonly pages falls to 0.52–0.68.

<table><tr><td>Model</td><td>Outcome</td><td>Δ [95% CI]</td><td>p</td></tr><tr><td>Qwen2.5-7B</td><td>declines</td><td>-4.8 [-8.9, -0.8]</td><td>.022</td></tr><tr><td>OLMo 2 7B</td><td>wrong declines</td><td>+1.7 [-1.1, +4.8] -31.7[-40.6, -22.7]</td><td>.26 2e-11</td></tr><tr><td></td><td>wrong</td><td>+25.0 [+16.5, +33.4]</td><td>1e-8</td></tr><tr><td>Granite 3.1 8B declines</td><td></td><td>-6.5 [-14.0, +0.8]</td><td>.097</td></tr><tr><td></td><td>wrong</td><td>+8.0 [+1.5, +14.6]</td><td>.019</td></tr><tr><td>Pooled</td><td>declines</td><td>-14.3 [-18.7, -9.6]</td><td>1e-9</td></tr><tr><td></td><td>wrong</td><td>+11.6 [+7.5, +15.4]</td><td>3e-8</td></tr></table>

Table 22: GloBI page efects (points), family-only minus silent (Eq. 1).

Why GloBI is harder. Sibling species are harder to tell apart than sibling diseases. The models know few species facts closed book (Table 21), so when a page names only the genus, they guess inside the genus close to chance. This is why precision on family-only pages is near chance in GloBI (22–28%) but closer to half in PrimeKG (45–59%).

## H Ontology Recovery

Recovery asks whether the ontology can turn a family name back into the exact answer. It has three steps: build a list of candidates, choose one, and compare with baselines that do not use the ontology.

Facts. We use all 410 PrimeKG indications and contraindications whose combined Wikipedia and PubMed text names only a MONDO family that expands to at least two diseases. A drug often has several true diseases in the same family (2.47 per list on average). We therefore count a pick as correct if it matches any disease the drug has under that relation. Table 25 also gives the strict score, which accepts only the one sampled answer.

## H.1 Building the List

Given family. This list assumes we already know which family on the page belongs to the question. We start at the true disease and climb up to two levels in MONDO. Among these ancestors, we keep the ones the drug’s text names. For each, we collect its descendants up to two levels down. We keep the ancestor with the smallest list, sort the list, and cap it at 64 diseases. This is an oracle: it uses the true disease to pick the family. It tells us how good the list can be at best.

Page built. This list uses only the page, as a real system would. We take all 3,182 MONDO families with two or more children as candidate family names. We index every run of up to eight words on the drug’s cleaned Wikipedia page, adding singular forms of words that end in s. Every family whose name appears in this index is a match. We drop any matched family whose own two-level list exceeds 64 diseases, and join the lists of the rest. If the joined list still exceeds 64, we mark the page as too broad. In short:

```python
fams = [f for f in families if name(f)
in page]
fams = [f for f in fams if |list(f)| <=
64]
```

<table><tr><td>Wikipedia + PubMed</td></tr><tr><td>Matched families per page (median)</td></tr><tr><td>5 17 166 19</td></tr><tr><td>Usable lists (of 410) Lists over the 64 cap 209 391</td></tr><tr><td>Usable list holds sampled answer 19.8% 3.7%</td></tr><tr><td>Usable list holds any true answer 24.6% 4.1%</td></tr><tr><td>Median usable list size 32 59</td></tr><tr><td>End-to-end recovery (evidence pick) 10.0% 1.5%</td></tr></table>

Table 23: Page-built lists from Wikipedia alone and from Wikipedia plus PubMed, over all 410 family-only facts. The remaining unusable lists match no family with two or more children.

cands = union(list(f) for f in fams)   
if |cands| > 64: too broad

Why page-built lists fail. A drug article names many disease families that are not the answer’s: other uses, side efects, warnings and conditions that often occur together. The median Wikipedia page matches five families; with PubMed it matches 17. Their union is usually too large (Table 23). The family that holds the answer is on the page, but the matcher cannot tell it apart from the others.

## H.2 Choosing One Candidate

Model scores. Qwen2.5-7B scores each candidate by its mean log-probability as the completion of a relation stem, such as “X is a drug used to treat the disease called”. In the model alone condition the prompt has only the stem; in the model with page condition the page comes first. The highest score is the pick.

Evidence scores. We use two outside sources. The Wikipedia reverse lookup checks whether the candidate disease’s own article names the drug (a mention), and whether a relation keyword appears within 250 characters of that mention (a cue). Indication keywords are treat, therap, indicated and first-line; contraindication keywords are contraindicat, avoid, must not, caution, worsen and adverse. The PubMed co-mention count is the number of records that name both the drug and the candidate with a relation qualifier:

$$
\begin{array} { l } { ^ { \mathfrak { n } } < \mathrm { d } \mathrm { r } \mathrm { u } \mathrm { g } > ^ { \mathfrak { n } } [ \mathrm { t } \mathrm { i } \mathrm { a } \mathrm { b } ] \quad \mathbb { A } \mathrm { N } \mathrm { D } ^ { \mathrm { ~ \mathfrak { n } } } < \mathrm { c } \mathrm { a } \mathrm { n } \mathrm { d } > ^ { \mathfrak { n } } [ \mathrm { t } \mathrm { i } \mathrm { a } \mathrm { b } ] \quad \mathbb { A } \mathrm { N } \mathrm { D } } \\ { < \mathrm { q } > } \end{array}
$$

where the indication qualifier <q> is   
(treat<sub>\*</sub>[tiab] OR therap<sub>\*</sub>[tiab] OR   
efficac<sub>\*</sub>[tiab]) and the contraindica  
tion qualifier is (contraindicat<sub>\*</sub>[tiab]   
OR adverse[tiab] OR toxicit<sub>\*</sub>[tiab]   
OR "side effect"[tiab] OR "side   
effects"[tiab] OR induced[tiab]). We   
strip parenthetical parts such as “(disease)” from MONDO   
labels before we query. The combined evidence score of   
candidate � is

$$
E ( c ) = 2 { \cdot } 1 [ \mathrm { c u e } ] + 1 [ \mathrm { m e n t i o n } ] + \log \left( 1 + \mathrm { c o - m e n t i o n s } \right) . ~ ( 1 0 )
$$

A signal makes a pick only when one candidate has the strictly highest score; otherwise it records a tie or no signal.

<table><tr><td colspan="5">Signal Picks Ties None Acc./pick Acc./all</td></tr><tr><td>Wikipedia lookup</td><td>82</td><td>80</td><td>248</td><td>50.0</td></tr><tr><td>PubMed co-mention</td><td>324</td><td>12</td><td>74 45.4</td><td>10.0 35.9</td></tr><tr><td>Combined evidence</td><td>327</td><td>14</td><td>69 44.6</td><td>35.6</td></tr><tr><td>Popularity prior</td><td>409</td><td>0</td><td>1 46.9</td><td>46.8</td></tr><tr><td>Normalized PubMed</td><td>336</td><td>0</td><td>74</td><td>31.0 25.4</td></tr></table>

Table 24: Signals inside the given-family list (410 facts, 8448 candidates). Accuracy (%) counts a pick as correct if it matches any true disease of the drug.
<table><tr><td>Condition</td><td>Sampled Any true</td><td></td></tr><tr><td>Model, given-family list</td><td>18.3</td><td>35.9</td></tr><tr><td>Model + page, given-family list</td><td>17.8</td><td>32.4</td></tr><tr><td>Model, random list</td><td>30.5</td><td>34.9</td></tr><tr><td>Model + page, random list</td><td>29.0</td><td>32.7</td></tr></table>

Table 25: Qwen2.5-7B recovery accuracy (%) on the 410 family-only facts, scoring only the sampled answer (Sampled) or any true answer of the drug (Any true).

The evidence, then model chooser uses � when it makes a pick and falls back to the model otherwise.

Popularity prior. The popularity prior ignores the drug. It counts all PubMed records that name the candidate ("<cand>"[tiab]) and picks the candidate with the most. The normalized co-mention signal divides each co-mention count by this total, which removes the popularity part.

What each signal does. Table 24 shows how often each signal makes a pick and how accurate those picks are. The Wikipedia lookup is accurate when it fires (50.0%), but it fires for only 82 facts, because a disease article rarely names a specific drug. PubMed co-mentions fire often, but they are only as accurate as the popularity prior. Once we divide out popularity, accuracy falls from 35.9% to 25.4%. Most of the signal in co-mentions is therefore how much a disease is studied, not its link to the drug.

## H.3 Baselines Without the Ontology

To see what the ontology adds, we replace the family list with a list of the same length that holds the sampled answer and random diseases from the same relation (never another true partner of the drug). The model scores this random list with or without the page. Table 25 compares the conditions. Under the strict score, the random list looks easier, because its wrong candidates are unrelated diseases that the model can rule out. Under the any-true-answer score, the family list catches the drug’s other true diseases, and the model does about as well on it as on the random list. The family list is valuable because it is short and complete (98.5% coverage), not because it makes the model’s choice easier.

## H.4 Examples

Table 26 shows evidence picks within the given-family list. When the pick misses the PrimeKG answer, it usually takes a better-studied member of the same family (ovarian carcinoma for carboplatin) or a true use that PrimeKG does not list (Legionnaires’ disease for moxifloxacin, which is used against this pneumonia). Strict scoring against the graph counts both as wrong, so it understates how often the pick is clinically sound.

<table><tr><td>Drug</td><td>PrimeKG Target</td><td>Evidence Pick</td><td>Match?</td></tr><tr><td>Lovastatin</td><td>familial</td><td>familial</td><td>yes</td></tr><tr><td>Tetrabenazine</td><td>hyperlipidemia juvenile Huntington</td><td>hyperlipidemia juvenile Huntington</td><td>yes</td></tr><tr><td>Moxifloxacin</td><td>disease Klebsiella pneumonia</td><td>disease Legionnaires&#x27;</td><td>no</td></tr><tr><td>Carboplatin</td><td>malignant germ cell tumor of ovary</td><td>disease ovarian carcinoma</td><td>no</td></tr></table>

Table 26: Qualitative examples of candidate diseases chosen by combined literature evidence within the given-family list on indication facts.

## I Worked Examples

This appendix follows single questions through the test. Each example shows the page, the question and what each model answers, closed book and with the page. Page excerpts come from English Wikipedia (CC BY-SA 4.0); “. . . ” marks text we leave out.

## I.1 PrimeKG

## Moxifloxacin: a confident wrong guess.

Fact: moxifloxacin treats Klebsiella pneumonia.   
Page type: family-only. The page says the drug treats pneumonia,   
but not which kind.   
Question: Which disease is moxifloxacin used to treat?   
Options include: Klebsiella pneumonia, Legionnaires’ disease,   
pneumonia caused by gram negative bacteria, and other bacterial   
pneumonias; Not enough information to tell.

Every model answers, and every model is wrong:
<table><tr><td>Model</td><td>Closed book</td><td>With the page (conf.)</td></tr><tr><td>Qwen2.5-0.5B</td><td>granegative pn.</td><td>granegative pn. (.35)</td></tr><tr><td>Qwen2.5-7B</td><td>Legionnaires&#x27;</td><td>Legionnaires&#x27;(.88)</td></tr><tr><td>OLMo 2 7B</td><td>granegative pn.</td><td>granegative pn. (.86)</td></tr><tr><td>Granite 3.1 8B</td><td></td><td>granegative pn. Legionnaires&#x27;(.77)</td></tr><tr><td>Qwen2.5-14B</td><td>Legionnaires&#x27;</td><td>granegative pn. (1.00)</td></tr></table>

The page tells the models that the answer is a pneumonia, and they pick a pneumonia with high confidence. None of them declines, even though the page cannot say which pneumonia is meant. A confidence threshold would keep most of these answers. Note also that Legionnaires’ disease is a real use of moxifloxacin that PrimeKG does not list: the graph counts a sound answer as wrong (Section H).

## Lovastatin: the page makes one model worse.

Fact: lovastatin treats hyperlipidemia, familial combined, LPL related. Page type: family-only. The page says hyperlipidemia. Family in MONDO: hyperlipidemia → familial hyperlipidemia → {the answer, hyperlipidemia due to hepatic triglyceride lipase deficiency, familial lipoprotein lipase deficiency, . . . }.

<table><tr><td>Model</td><td>Closed book</td><td>With the page</td></tr><tr><td>Qwen2.5-0.5B</td><td>wrong sibling</td><td>wrong sibling</td></tr><tr><td>Qwen2.5-7B</td><td>declines</td><td>correct (1.00)</td></tr><tr><td>OLMo 2 7B</td><td>correct (.95)</td><td>wrong sibling (.54)</td></tr><tr><td>Granite 3.1 8B</td><td>correct (.98)</td><td>correct (.86)</td></tr><tr><td>Qwen2.5-14B</td><td>wrong sibling</td><td>wrong sibling (1.00)</td></tr></table>

Here the family name helps Qwen2.5-7B and hurts OLMo 2. OLMo 2 knows the answer closed book, but with the page it switches to hyperlipidemia due to hepatic triglyceride lipase deficiency, a sibling. The page points to the right family, and the model then picks the wrong member. This is Clade in its plainest form.

## I.2 GloBI

Named page: the answer is on the page.

Page (Congriscus megastomus): “Congriscus megastomus is an eel in the family Congridae (conger/garden eels). . . . It is a marine, temperate-water dwelling eel which is known from Japan and the Kyushu–Palau Ridge . . . ” The page later names the prey. Question: Which species does Congriscus megastomus feed on? Options: Ventrifossa garmani, Ventrifossa mucocephalus, Coryphaenoides zaniophorus, Hymenocephalus lethonemus (true), Gadomus colletti, Not enough information to tell.

All three models decline closed book. With the page, Qwen2.5-7B and Granite pick the true species (confidence 1.00 and .997); OLMo 2 declines. When the page names the answer, the models use it well.

## Family-only page: the page turns declines into guesses.

Page (Vanessa virginiensis, American painted lady): “The lar  
vae feed on various Asteraceae, such as the cudweeds (genus   
Gnaphalium), the pussytoes (Antennaria), and the everlastings   
(Anaphalis) . . . ”   
Question: Which species does Vanessa virginiensis feed on?   
Options: Antennaria pulcherrima, Antennaria plantaginifolia   
(true), Antennaria dimorpha, Antennaria lanata, Antennaria mi  
crophylla, Not enough information to tell.

Every option is an Antennaria species, so the genus name on the page rules out none of them. Closed book, all three models decline. With the page, Qwen2.5-7B still declines, but OLMo 2 picks A. microphylla (.52) and Granite picks A. pulcherrima (.63). Both are wrong. The page did not tell them the answer; it only made the question look answerable.

Silent page: the models abstain.

Page (Cottunculus microps, polar sculpin): “. . . a species of fathead sculpin, a deepwater fish found in the North Atlantic and Arctic Oceans. . . . Like other members of its family, the polar sculpin has a large head, a short tapering body and a small tail. ,,

Question: Which species does Cottunculus microps feed on? True answer: Lumpenus lampretaeformis.

The page describes the fish but says nothing about its food. All three models decline, closed book and with the page. This is the behavior we want on a family-only page as well, unless the system hands the question to the taxonomy.

A named page the models miss. The page for Orgyia vetusta (western tussock moth) names its food plant only by the common name toyon, which redirects to Photinia arbutifolia. Our matcher counts this page as named. All three models decline anyway: they do not link the common name to the Latin name in the options. Common names are a second gap between how text names a fact and how a graph stores it.

## J Software and Compute

Models. We use the following instruction-tuned checkpoints from Hugging Face: Qwen2.5-0.5B-Instruct, Qwen2.5- 7B-Instruct, Qwen2.5-14B-Instruct, OLMo-2-1124-7B-Instruct, granite-3.1-8b-instruct. We do not fine-tune any model.

Running the models. All models run in 32-bit floating point with Hugging Face Transformers and PyTorch, on CPU nodes of a university cluster. Each model answers all 1,712 abstention prompts (856 questions, closed book and with the page) in one 12-thread job. We score prompts in batches with left padding. We rebuild the position indices from the attention mask, so a batched prompt gets exactly the same probabilities as the same prompt alone. Every answer is stored under a hash of its exact prompt, so an edited page never reuses an old answer.

Recovery. In recovery, all candidates for one fact share the same prompt prefix. We encode this prefix once and reuse its key–value cache for every candidate. Before each run, we check a sample of these cached scores against full forward passes.

Randomness and statistics. All sampling uses seed 0, and the bootstrap draws are sorted by question identifier before resampling, so every interval in the paper is the same on every run. The statistics use only the Python standard library.

## K Choosing the Family and Free-Text Answers

This appendix covers two follow-up tests. The first tests a simple fix for step 2 of the hand-of: which family on the page holds the answer. The second asks the abstention questions without options, to check that the option list does not drive the main result.

## K.1 Choosing the Family on the Page

Setup. We use the same 410 family-only facts as in Section H. For each fact, we take the families that the page-built list finds on the drug’s Wikipedia page and keep those whose own list has at most 64 diseases. We order them by where the page first mentions them and show at most ten, labeled A to J. Qwen2.5-7B reads a window of the page around these mentions and the question, and picks one family:

<table><tr><td>Rule</td><td>Sampled answer Any true answer</td></tr><tr><td>Qwen2.5-7B picks the family</td><td>30.0 44.9</td></tr><tr><td>First mentioned</td><td>25.1 37.3</td></tr><tr><td>Random</td><td>13.9 22.0</td></tr><tr><td>Smallest family</td><td>7.8 13.2</td></tr><tr><td>Page-built list (all families)</td><td>19.8 24.6</td></tr><tr><td>Any shown family (ceiling)</td><td>50.0 64.6</td></tr></table>

Table 27: Share of the 410 family-only facts (%) whose chosen list holds the sampled answer, or any true answer of the drug. At most ten families are shown per page.

Passage:   
<page window>   
Question: Which disease is X used to   
treat?   
The answer is one specific disease.   
Which of these disease categories from   
the passage contains it?   
Options:   
A. <family>   
Answer with the letter of one option   
only.

We read the answer from the letter probabilities, as in Section D. We then expand only the chosen family, two levels down, and score its members with the same choosers as before.

Rules that read no meaning. To see what the model adds, we compare it with three rules that ignore the question: the family the page mentions first, the family with the smallest list, and a random family. We also report a ceiling: the share of facts where at least one of the shown families holds the answer.

Results. Table 27 shows the result. The model’s choice holds the answer for 30.0% of all 410 facts, against 19.8% for the page-built list, which joins every family on the page. It also beats each rule. The first-mentioned family is a strong rule (25.1%); the smallest family is a poor one (7.8%), worse than a random pick. The ceiling is 50.0%: for the other half of the facts, the family that holds the answer is not among the usable families shown. The model therefore closes 30.0 of the 50.0 points that any chooser over these families could reach.

End to end. The chosen lists are short (median 12 diseases), and 375 of the 410 facts get one. Scored with the same choosers, they recover the answer for 19.3% of facts with the combined evidence and 26.8% with the popularity prior. The page-built lists reach 10.0% and 9.0%. With no oracle at any step, a single prompted choice of the family nearly triples end-to-end recovery.

## K.2 Free-Text Answers

Setup. We sample 40 questions per page type (named, family-only, silent) from the 481 questions whose four wrong options all come from the answer’s neighborhood, with seed

<table><tr><td>Model</td><td>Page</td><td>Corr. Abst. Family</td><td>Sibl. Other</td><td></td></tr><tr><td rowspan="2">Qwen2.5-7B</td><td>named</td><td>20.0 47.5</td><td>0.0 0.0</td><td>32.5</td></tr><tr><td>family</td><td>0.0 42.5</td><td>17.5 0.0</td><td>40.0</td></tr><tr><td rowspan="3">OLMo 2 7B</td><td>silent</td><td>0.0 52.5</td><td>0.0 0.0</td><td>47.5</td></tr><tr><td>named</td><td>25.0 37.5</td><td>0.0 0.0</td><td>37.5</td></tr><tr><td>family silent</td><td>0.0 32.5 52.5</td><td>20.0 2.5</td><td>45.0</td></tr><tr><td rowspan="4">Granite 3.1 8B named</td><td></td><td>0.0 22.5 37.5</td><td>0.0 0.0</td><td>47.5</td></tr><tr><td>family</td><td>0.0 30.0</td><td>0.0 0.0 45.0 2.5</td><td>40.0</td></tr><tr><td>silent</td><td></td><td></td><td>22.5</td></tr><tr><td></td><td>0.0 45.0</td><td>0.0 0.0</td><td>55.0</td></tr></table>

Table 28: Free-text outcomes with the page (%, 40 questions per row).

0. We ask each one closed book and with the page, with no options:

Passage:   
<page>   
Question: Which disease is X used to   
treat?   
Answer with the name of one disease   
only. If the passage does not tell   
you which one, answer "Not enough   
information to tell".

Each model writes up to 32 tokens with greedy decoding.

Scoring. We check each answer in this order and assign the first outcome that applies:

1. correct: it names the answer or one of its alternative names;

2. abstain: it says there is not enough information, or that the passage does not say;

3. sibling: it names one of the four wrong options of the multiple-choice question;

4. family: it names one of the answer’s family names, such as acute myeloid leukemia for acute myeloid leukemia with t(8;21);

5. other: it names any other disease.

The scoring is lenient: an answer that lists several diseases counts as correct if one of them is the answer. This afects only Granite on named pages, where 7 of its 9 correct answers are lists. It does not afect family-only or silent pages, where no model is ever correct.

Results. Table 28 gives the outcomes with the page. Three patterns stand out.

• No exact answersfromfamily-only pages. No model ever names the exact answer from a family-only or silent page. When the page names the answer, the models name it 20.0–25.0% of the time.

• Answering with the family. On family-only pages, 17.5– 45.0% of answers name the family itself, such as “acute myeloid leukemia (AML)”. Closed book, this happens 10.0–15.0% of the time. It never happens on named or silent pages. Without options, Clade takes this form: the model does not pick a sibling, but gives the coarse category as if it were the answer.

• Fewer abstentions. Compared with silent pages, familyonly pages lower abstention in all three models. The page efects (Eq. 1) are −5.0 [−25.0, +15.0] for Qwen2.5- 7B, −17.5 [−45.0, +7.5] for OLMo 2 and −10.0 [−32.5, +10.0] for Granite. All three point the same way as in the main test, but with 40 questions per group no single interval excludes zero.

Free text also makes the models more cautious overall. They abstain on 37.5–47.5% of named pages, where the multiplechoice test shows 7.5–24.5%. Writing the answer is harder than picking a letter, so we compare page types within the free-text format, not across formats.