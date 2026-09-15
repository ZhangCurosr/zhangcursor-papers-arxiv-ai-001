# LLM-Based Schema-Aware Split Learning for Privacy-Preserving Mental Distress Prediction Across Heterogeneous Surveys

Md Khalid Syfullah and Alvi Ataur Khalil

Transformative Innovation for Trustworthy AI and Network Security (TITANS) Lab,

Computer Science, Southern Illinois University Carbondale, USA

{mdkhalid.syfullah, a.khalil}@siu.edu

Abstract—Rising societal and lifestyle complexity has been linked to a growing prevalence of mental distress worldwide. Educational institutions, workplaces, clinics, etc. collect large volumes of mental health survey data to understand and reduce this burden. Collaborative analysis of such data could yield effective generalizable predictive models. Privacy constraints and varied survey designs (i.e., different questions, scales, and formats) hinder direct integration. We propose a schema-aware split learning (SL) framework that preserves privacy, using a large language model (LLM) as a shared semantic encoder to harmonize heterogeneous survey schemas across institutions. We serialize each survey record into a natural-language description, unifying disparate survey schemas into a common format. The LLM is fine-tuned for mental distress assessment via Low-Rank Adaptation (LoRA) and partitioned across client and server. Clients retain the raw survey responses locally and run only a lightweight front-end, so original records never leave the institution that collected them. The resource-intensive backbone runs on the server, minimizing client-side computation. Using LLaMA-3.2-3B-Instruct, the framework attains an average ANLS of 0.708 with only 2,000 training samples, surpasses federated learning (FL) in eight of nine settings, and cuts per-client computation by three orders of magnitude, while generalizing to unseen datasets. Overall, it enables accurate, privacy-preserving, and resource-efficient collaborative learning from heterogeneous mental health survey data.

Index Terms—Split learning, large language models, low-rank adaptation, data heterogeneity, mental distress prediction

## I. INTRODUCTION

Mental disorders are among today’s most pressing public health challenges. The World Health Organization (WHO) estimates that in 2022, roughly one in eight people worldwide (nearly one billion individuals) were living with a mental disorder, including anxiety or depression, with the burden falling more heavily on students and working-age populations [1], [2]. Stress and related conditions are often assessed using self-report instruments, including the Depression, Anxiety and Stress Scales (DASS) [3]. Different organizations conduct mental health surveys every year [4]. These surveys support automated mental distress screening with predictive AI models [5].

Despite these advances in AI models, two main challenges limit the collaborative use of these data. First, they are highly sensitive, and unauthorized disclosure can cause stigma and discrimination [6]. In addition, data-protection regulations such as HIPAA and the GDPR restrict the sharing of health records, keeping datasets siloed within individual institutions and permitting their use only under strict privacy safeguards [7], [8]. Second, institutions use different survey instruments, response scales, and question formats, creating inconsistent feature spaces and data distributions, a problem known as data heterogeneity [9]. An effective AI-based assessment system must therefore address privacy protection and cross-schema heterogeneity simultaneously. However, existing methods typically tackle only one of these issues, and, to the best of our knowledge, no prior work jointly addresses both for mental distress assessment.

A Large Language Model (LLM) can act as a semantic encoder to address schema heterogeneity in conventional models [10]. When a survey record is converted into a natural-language description, for example, “sleep duration: less than 5 hours” rather than a coded value like “sleep: 3”, the meaning of each feature is expressed through language rather than encoded in a fixed vector position. As a result, surveys that capture similar concepts through different questions, scales, or formats can be mapped into a shared representation, even when their original schemas differ. Recent work on language-based tabular learning confirms this serialization transfers knowledge across datasets [11]–[13].

Our goal is to let institutions collaboratively train a mentaldistress model without sharing raw responses, which we achieve through two components. First, we adapt the LLM with Low-Rank Adaptation (LoRA) [14], which freezes the pretrained weights and trains only small adapters, sharply reducing trainable parameters and the memory and communication cost of collaborative training. Second, a Split Learning (SL) architecture keeps raw data on the client: each holds only the tokenizer, a LoRA-adapted embedding, and a lightweight projection head, while the LLM backbone and classifier stay on the server. Accordingly, our privacy model is data locality: the survey responses and identifiers never leave the institution that collected them, and only cutlayer activations and their gradients cross the client–server boundary. We do not claim that these intermediate tensors are non-invertible; hardening them against reconstruction (model-inversion) attacks, for instance through differential privacy or activation obfuscation, is outside the scope of this work. To the best of our knowledge, this is the first framework to combine SL for lightweight, data-local clients, a LoRA-adapted LLM as a semantic encoder, and languagebased harmonization of heterogeneous survey schemas for mental distress assessment. One of the related works uses a frozen language model with federated averaging instead of SL, forcing every client to run the full model, ignores heterogeneous schemas, and targets cardiac and financial tables rather than mental health surveys [15]. Recent SL frameworks for LLMs target general NLP tasks and device heterogeneity, not heterogeneous survey schemas [16]. The contributions of this paper are fourfold:

• We propose a privacy-preserving SL framework that trains a LoRA-adapted LLaMA-3.2-3B-Instruct [17] collaboratively across institutional clients and a server, so that raw responses remain local to each institution.

• We introduce a schema-aware serialization that turns heterogeneous survey records into LLM prompts by mapping ordinal responses to text, marking missing values, and adding a schema header, so surveys with differing feature sets share a common representation.

• We evaluate the framework under varying data imbalance and schema heterogeneity, benchmarking it against Federated Learning (FL), zero-shot, few-shot, and LoRA baselines and a second LLM OpenBioLLM-8B [18].

• We analyze the framework’s generalization to unseen survey datasets by training on a small data pool.

The study is guided by the following research questions:

• RQ1: Can an LLM-based SL framework predict mental distress more accurately than conventional baselines when clients hold different survey schemas?

• RQ2: How does performance change as clients vary in dataset size and survey structure?

• RQ3: How does SL compare with FL in performance and per-client computational cost?

The remainder of this paper is organized as follows. Section II introduces the necessary background, Section III reviews related work, Section IV presents the methodology, Section V reports the results, and Section VI concludes.

## II. BACKGROUND

This section introduces the concepts behind our framework: LLM, LLMs and FT-Transformer for survey data, LoRA, SL.

## A. LLMs, LLaMA and OpenBioLLM

LLMs build on the Transformer architecture [19] and learn semantic representations from large text corpora, making them useful beyond traditional NLP. In this work, we use LLaMA-3.2-3B-Instruct [17], a compact model from Meta’s LLaMA family of open-weight LLMs that balances performance and efficiency [20]. We also evaluate OpenBioLLM-8B [18], a biomedical model built on LLaMA-3-8B and finetuned on curated clinical and medical corpora, which reports state-of-the-art results on clinical NLP benchmarks.

## B. Survey Data with LLMs and FT-Transformer

Recent studies show tabular and survey data can be serialized into natural language and processed effectively by LLMs [12], [13]. This maps related questions into a shared representation even when institutions use different features, scales, or coding schemes. Beyond LLMs, FT-Transformer [21] tokenizes each tabular feature and uses attention to model feature interactions, achieving strong performance on structured data. However, it learns a feature space directly from the data rather than feature semantics, limiting transfer across datasets with differing schemas [11].

![](images/e49fe4b67eeb6ed44df2bd7f7f2e802609dad481af9fa6551dc0148c76a1f904.jpg)  
Fig. 1: Overview of the split-learning framework.

## C. Low-Rank Adaptation (LoRA)

Fine-tuning all LLM parameters is computationally expensive and memory intensive. LoRA [14] keeps the pretrained weights fixed and trains only a small set of low-rank adapters, preserving most benefits of full fine-tuning. This suits adapting large models to domain-specific tasks with limited resources.

## D. Split Learning (SL)

SL divides a neural network between clients and a server, keeping data local while exchanging intermediate representations during training [22]. This improves privacy and allows resource-constrained clients to participate without hosting the model. Fig. 1 shows an overview of SL framework.

## III. LITERATURE REVIEW

Both SL and FL enable collaboration without sharing raw data, differing in how they distribute computation: SL partitions the model across client and server, whereas FL shares the model. In SL, clients process data and share intermediate activations [22], and SplitFed improves scalability through parallel training [23]. FL provides an alternative, with FedAvg [24] as the most widely used scheme.

Parameter-Efficient Fine-Tuning (PEFT) has made LLMs practical in distributed settings: LoRA [14] trains only a few parameters, while SplitLoRA [25] and HSplitLoRA [16] extend this to SL and FL. These approaches reduce communication and computation, letting clients join LLM-backbone training without sharing raw data [25].

A separate line of work uses LLMs as semantic encoders for tabular data by serializing records into text. TabLLM [11], LIFT [12], and TP-BERTa [13] show that semantically related features from different schemas can share a representation space, while FT-Transformer [21] provides a strong non-LLM baseline for structured data. FedLLM-Align [15] brings this to heterogeneous tables in a federated setting, yet uses a frozen encoder with federated averaging and targets nonclinical data. These studies confirm the value of semantic serialization, yet none addresses a central need: enabling resource-constrained institutions to learn collaboratively from heterogeneous mental health surveys without exposing sensitive responses. SL targets this by keeping raw data local and offloading the heavy LLM backbone to the server. In mental health, machine learning has predicted depression and stress from survey data, with privacy-preserving extensions through FL [26], yet these works assume a common feature schema.

![](images/02c0a564d6636ab5dee1f7b8534491632d78ac0edcdf367629da66333e273ec2.jpg)  
Fig. 2: System overview of the proposed split-learning framework.

Across these areas, existing methods satisfy at most two of the three needs in our setting: SL for lightweight, datalocal clients; a LoRA-adapted LLM encoder; and support for heterogeneous survey schemas [11], [12], [15], [16], [25]. None combines all three for mental distress prediction, and none demonstrates strong cross-schema performance from a small training pool. Our framework fills this gap by unifying schema-aware survey serialization, a LoRA-based semantic encoder, and SL into a privacy-preserving solution that is lightweight and data-efficient on the client.

TABLE I: List of Notations.
<table><tr><td rowspan=1 colspan=1>Symbol</td><td rowspan=1 colspan=1>Description</td></tr><tr><td rowspan=1 colspan=1>N</td><td rowspan=1 colspan=1>Number of SL clients</td></tr><tr><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>Serialized survey record (prompt)</td></tr><tr><td rowspan=1 colspan=1>a</td><td rowspan=1 colspan=1>Cut-layer activation sent client→server</td></tr><tr><td rowspan=1 colspan=1>g</td><td rowspan=1 colspan=1>Gradient returned server→client</td></tr><tr><td rowspan=1 colspan=1>W0</td><td rowspan=1 colspan=1>Frozen pretrained weight matrix</td></tr><tr><td rowspan=1 colspan=1>A, B</td><td rowspan=1 colspan=1>LoRA factors (∆W = αBA)</td></tr><tr><td rowspan=1 colspan=1>r, α</td><td rowspan=1 colspan=1>LoRA rank and scaling constant</td></tr><tr><td rowspan=1 colspan=1>T</td><td rowspan=1 colspan=1>ANLS non-answer threshold</td></tr><tr><td rowspan=1 colspan=1>p, g</td><td rowspan=1 colspan=1>Predicted and ground-truth core answers</td></tr></table>

IV. METHODOLOGY

This section presents the framework: schema-aware serialization, data partitions, the SL architecture, and evaluated LLM paradigms. Fig. 2 shows the system overview.

## A. Schema-Aware Serialization

We use four public mental health surveys spanning domains, schemas, and label spaces: the OSMI Mental Health in Tech Survey (OST) [27], DASS-42 (D-42) [28], the Student Depression Dataset (SD) [29], and a general-population Mental Health dataset (MHD) [30]. A unified pipeline cleans each dataset, harmonizes column names, unifies boolean encodings, and applies DASS validity checks. Identifiable and demographic columns are removed before model building.

Each survey record, i.e., one respondent’s row in a cleaned dataset, is serialized into multiple instruction-following samples in the {instruction, input, output} format, one per outcome, so that the model learns each survey’s multiple distress-related outcomes instead of a single fixed inputto-label mapping. Table II summarizes this mapping. The predicted outcome is removed from the input to prevent leakage, and exact (input, output) duplicates are dropped before any split. Because a single respondent yields several samples, all splitting is performed at the respondent level rather than the sample level: every sample derived from a given respondent is assigned to exactly one of the training, validation, or test sets, so no respondent ever appears in more than one split and no near-duplicate of a training record can leak into evaluation.

## B. Data Partitions

Serialization places all surveys in a shared language space, letting datasets with different formats merge into one corpus. A model trained on this mix acquires broader knowledge than one tied to a single schema. To study this, we construct three partitions with increasing schema heterogeneity, all stratified by source with a fixed seed and grouped by respondent:

• Homogeneous: All four datasets used separately.

• Moderate Heterogeneous (MHg): Merged OST and SD, two surveys that share binary distress outcomes but differ in feature schemas and target populations.

• Fully Heterogeneous (Hg): All four datasets merged. For evaluation, 500 samples are held out from each of the six test sets across all setups. These held-out samples are drawn from respondents that appear in no training or validation split, and the same respondent-disjoint constraint applies when the training pool is sharded across clients, confining each respondent to a single client.

## C. Split-Learning Architecture

The framework partitions a LoRA-adapted, decoder-only LLaMA-3.2-3B between clients and a server. LoRA injects trainable rank-r matrices into the projection layers of each transformer block; for $\mathbf { W } _ { 0 }$ , the adapted forward pass is

$$
h = { \bf W } _ { 0 } x + \frac { \alpha } { r } { \bf B A } x , \quad { \bf A } \in \mathbb { R } ^ { r \times n } , { \bf B } \in \mathbb { R } ^ { m \times r } ,\tag{1}
$$

where only A and B are updated, reducing trainable parameters to roughly 0.75% of the model. The model is split as follows, with all symbols defined in Table I:

TABLE II: Record-to-prompt serialization across the four datasets.
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>#Outcomes</td><td rowspan=1 colspan=1>Example input (serialized features)</td><td rowspan=1 colspan=1>Example instruction</td><td rowspan=1 colspan=1>Example output</td></tr><tr><td rowspan=1 colspan=1>OST</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>Employer provides mental health benefits: No. Family history of mentalillness: Yes. Ease of taking medical leave: Somewhat difficult.</td><td rowspan=1 colspan=1>Did this tech worker seekprofessional mental healthtreatment?</td><td rowspan=1 colspan=1>Yes, this tech worker has soughtprofessional mental health treat-ment.</td></tr><tr><td rowspan=1 colspan=1>D-42</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>Symptom responses — Q3 (inability to experience positive feeling): mostof the time; ... Personality — anxious and easily upset: agree strongly.</td><td rowspan=1 colspan=1>What is this person&#x27;s de-pression severity level?</td><td rowspan=1 colspan=1>This person&#x27;s depression sever-ity is: Severe (score: 24/42).</td></tr><tr><td rowspan=1 colspan=1>SD</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>Academic pressure (0–5): 5. Sleep duration: less than 5 hours. Dietaryhabits: Unhealthy. Financial stress (1–5): 4.</td><td rowspan=1 colspan=1>Does this person showsigns of depression?</td><td rowspan=1 colspan=1>Yes, this person is experiencingdepression.</td></tr><tr><td rowspan=1 colspan=1>MHD</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>Occupation: Corporate. Days spent indoors: more than 2 months. Moodswing frequency: High. Noticed changes in habits: Yes.</td><td rowspan=1 colspan=1>Is this person experiencinggrowing stress?</td><td rowspan=1 colspan=1>Yes, this person is experiencinggrowing stress.</td></tr></table>

• Client: Each client holds the tokenizer, a LoRA-adapted embedding, and a two-layer projection head (∼100K parameters). It maps token IDs to a cut activation a and transmits only a to the server. The projection head is zero-initialized as a residual MLP, so it begins as an identity mapping, keeps a within the distribution expected by the trunk, and learns only a correction.

• Server: The server holds the full LoRA-adapted transformer trunk and the language-model head. It consumes a as input embeddings, generates answer tokens, computes the loss against ground truth, and returns the cut gradient g to complete the client’s backward pass.

Activations and gradients are exchanged per batch, so only cut-layer tensors cross the client-server boundary; the original survey responses, identifiers, and labels remain on the client by construction and are never transmitted. To study SL under realistic conditions, we form a 3 × 3 grid of nine setups by crossing three schema-heterogeneity levels with three datavolume imbalance levels. Schema heterogeneity ranges from homogeneous (one dataset, IID shards) to moderate (two or three datasets) to high (four datasets, one source per client). Volume imbalance ranges from symmetric (10% per client) to moderate (power-law split) to high (60% held by one client).

## D. LLM Paradigms

We evaluate four configurations with increasing supervision and decentralization. All share the system prompt, “You are a mental health assessment assistant. Answer the question concisely and directly based on the given information”. The first three serve as baselines:

• Zero-shot: The model receives only the task instruction and serialized test record, then generates predicted label greedily without examples or parameter updates.

• Few-shot: The model receives 10 topically matching labeled examples and the serialized test record. Examples with the same answer as the test sample are excluded to avoid label leakage.

• LoRA fine-tuning: The LoRA adapters from Eq. (1) are trained centrally using the full training set. The loss is computed only on assistant response tokens, so the model learns to generate the target label without being penalized for the input prompt.

## V. EXPERIMENTAL RESULTS

We evaluate the proposed framework: the setup and metric, centralized baselines, SL results, the FL comparison, crossdataset generalization, the FT-Transformer reference, and research-question findings.

## A. Experimental Setup and Metric

Experiments used PyTorch on two NVIDIA RTX 6000 Blackwell GPUs (96 GB each) and 32 GB RAM. The default backbone is LLaMA-3.2-3B-Instruct, with OpenBioLLM-8B for cross-family checks. All LLM configurations use LoRA rank 16, α=32 on the $\boldsymbol { q } , \boldsymbol { k } , \boldsymbol { v } , \boldsymbol { o } ,$ gate, up, down projections, a 1024-token limit, and learning rate $2 \times 1 0 ^ { - 4 }$ . Centralized LoRA uses 2,000 training samples, batch 32, and 3 epochs. SL and FL both use 10 clients, batch 8, and 3 local epochs over the 3 × 3 grid, with at most 2,000 samples split across clients. The FT-Transformer uses 8 heads, 3 layers, batch 256, learning rate $1 \times 1 0 ^ { - 4 }$ , and up to 30 epochs.

All LLM configurations are scored with Average Normalized Levenshtein Similarity (ANLS), adapted for generative answers. Each raw output is mapped to a canonical core answer via priority-ordered extraction and vocabulary mapping (e.g., “high risk” → severe), then scored by a thresholded normalized similarity:

$$
\mathrm { N L S } ( p , g ) = \left\{ \begin{array} { l l } { s } & { \mathrm { i f } ~ s \geq \tau , } \\ { ~ } & { \mathrm { o t h e r w i s e } , } \end{array} \right. \quad \quad s = 1 - \frac { \mathrm { E d i t D i s t } ( p , g ) } { \operatorname* { m a x } ( | p | , | g | ) } ,\tag{2}
$$

with τ=0.5; adjacent severity levels receive partial credit of 0.5 for clinical proximity. Dataset-level ANLS is the mean over the 500-sample test set, which contains only respondents absent from the corresponding training and validation splits (Section IV-B).

## B. Performance under Centralized LLM Baselines

Table III reports centralized LLM baselines. Zero-shot is weak: LLaMA-3.2-3B averages 0.229 ANLS, ranging from 0.097 to 0.336. Few-shot improves only slightly (0.279), showing prompting alone cannot map heterogeneous surveys to distress labels. Per-dataset LoRA, trained on 2,000 samples each, raises the average to 0.665 ANLS, with best results on D-42 (0.788) and OST (0.761), establishing LoRA as our framework’s main trainable component. OpenBioLLM-8B performs similarly (0.659), and Fig. 4a shows its gains appear mainly in zero-shot and largely disappear after adaptation, confirming that LLaMA stays competitive despite being almost three times lighter.

## C. SL Performance with Heterogeneous Data

Table IV reports SL framework’s ANLS scores across the 3 × 3 grid alongside FL. SL averages 0.708 ANLS and stays within 0.660–0.759 across nine settings. Performance peaks under homogeneous schemas at 0.759 and drops slightly as heterogeneity grows, showing that schema-aware serialization

TABLE III: Centralized baseline ANLS scores.
<table><tr><td rowspan="2"></td><td colspan="2">Zero-Shot</td><td colspan="2">Few-Shot</td><td colspan="2">LoRA</td></tr><tr><td>Dataset Llama</td><td>OpenBio</td><td>Llama</td><td>OpenBio</td><td>Llama</td><td>OpenBio</td></tr><tr><td>SD</td><td>0.274</td><td>0.333</td><td>0.336</td><td>0.393</td><td>0.664</td><td>0.657</td></tr><tr><td>OST</td><td>0.336</td><td>0.370</td><td>0.316</td><td>0.280</td><td>0.761</td><td>0.753</td></tr><tr><td>MHg</td><td>0.302</td><td>0.314</td><td>0.353</td><td>0.375</td><td>0.631</td><td>0.650</td></tr><tr><td>Hg</td><td>0.155</td><td>0.196</td><td>0.238</td><td>0.226</td><td>0.702</td><td>0.693</td></tr><tr><td>MHD</td><td>0.210</td><td>0.257</td><td>0.258</td><td>0.136</td><td>0.444</td><td>0.427</td></tr><tr><td>D-42</td><td>0.097</td><td>0.155</td><td>0.170</td><td>0.164</td><td>0.788</td><td>0.776</td></tr></table>

lets clients with different feature sets share one representation. It is stable under volume imbalance: average ANLS shifts by under 0.01 across the three volume levels.

TABLE IV: SL and FL ANLS across grid settings.
<table><tr><td rowspan="2">Volume Schema</td><td colspan="3">Symmetric</td><td colspan="4">Moderate</td><td colspan="3">High</td></tr><tr><td>Homo</td><td>Mod</td><td></td><td>High</td><td>Homo</td><td>Mod</td><td>High</td><td>Homo</td><td>Mod</td><td>High</td></tr><tr><td>SL</td><td>0.754</td><td>0.660</td><td>0.701</td><td></td><td>0.759</td><td>0.670</td><td>0.705</td><td>0.757</td><td>0.669</td><td>0.696</td></tr><tr><td>FL</td><td>0.737</td><td>0.653</td><td>0.656</td><td></td><td>0.737</td><td>0.651</td><td>0.661</td><td>0.765</td><td>0.633</td><td>0.694</td></tr></table>

## D. Compute and Memory Cost: SL vs FL

The FL baseline trains the same trunk-LoRA adapters locally and aggregates them with sample-weighted FedAvg, using the same 2,000-sample, 10-client setup. SL exceeds FL in average ANLS (0.708 vs. 0.687) and wins in eight of nine settings (mean advantage +0.021; Table IV). The methods place cost differently (Table V): each SL client trains only 3.68M parameters (0.114% of the 3.21B model) at ∼0.08 Tera FLOating-Point operations per step (TFLOPs) and ∼1.3 GB, while the heavy trunk (∼158 TFLOPs, 15– 25 GB) runs once server-side. In FL, every client instead runs the full model (∼158 TFLOPs, 15–25 GB). SL thus keeps clients lightweight at roughly three orders of magnitude lower per-client compute.

TABLE V: Per-side compute and memory costs in our setup.
<table><tr><td>Method / side</td><td>Frozen base</td><td>Trainable params</td><td>Compute / step</td><td>VRAM (train)</td></tr><tr><td>SL client</td><td>~0.39B</td><td>3.68M</td><td>~0.08 TFLOPs</td><td>~1.3 GB</td></tr><tr><td>SL server</td><td>~3.21B</td><td>24.31M</td><td>~158 TFLOPs</td><td>15–25 GB</td></tr><tr><td>FL client (ea.)</td><td>~3.21B</td><td>24.31M</td><td>~158 TFLOPs</td><td>15–25 GB</td></tr><tr><td>FL server</td><td>0</td><td>0</td><td>~0</td><td>~0 (CPU)</td></tr></table>

## E. The Generalist Model

The generalist model is trained on the merged collection of all heterogeneous datasets in the symmetric-volume mode and evaluated on all six held-out test sets (Fig. 3). The training pool has 10,000 samples spread evenly across 10 clients (1,000 each), a fivefold increase over the grid. It averages 0.662 ANLS, transfers well to D-42 (0.798) and Hg (0.725), and performs worst on the feature-poor MHD set (0.394). This increase gives only a small gain over the matching grid setting, suggesting the model already generalizes from limited data. Fig. 4b compares it with per-dataset baselines: it beats zero- and few-shot on every dataset, by up to +0.701 ANLS over zero-shot on D-42, and remains competitive with dataset-specific LoRA. This shows it captures transferable structure rather than one fixed schema.

![](images/d72d0dd630dd6a5670558d9fe35f9c4aa8308612aff5c88bc4e2d4e70bd156b6.jpg)  
Fig. 3: Generalist ANLS across test datasets.

![](images/a4c110523b781a1e0b85067dbd1b132d4a509107a5a66d4270b9d91096f2e79a.jpg)  
(a)

![](images/17dd7d795a481ee94c039399bec8594d10e2926ca31ead7f8cf3e2f990d72bcb.jpg)  
(b)  
Fig. 4: ANLS gain heatmaps: (a) OpenBioLLM vs. LLaMA; (b) Generalist vs. baselines on LLaMA.

## F. Performance under FT-Transformer Baseline

We evaluate a fully supervised, non-LLM FT-Transformer as a per-dataset upper bound (Fig. 5). It averages 0.852 accuracy, peaks on D-42 (0.972) and Hg (0.932), and is lowest on the feature-poor MHD set (0.719); ROC-AUC is shown only for binary targets. The model is accurate but schema-bound: it depends on a fixed feature set and cannot handle a new survey without retraining, the heterogeneity our LLM-based framework is designed to absorb.

## G. Research Question Discussion and Limitations

For RQ1, the framework beats prompting baselines and matches or exceeds FL (0.708 vs. 0.687) even when clients hold different schemas, confirming that schema-aware serialization enables privacy-preserving cross-schema sharing. For RQ2, performance is robust: it varies by under 0.01 ANLS across volume levels, drops slightly with heterogeneity, and peaks at +0.045 over FL in symmetric/high setting. For RQ3, SL keeps client lightweight (3.68M parameters, ∼1.3 GB, ∼0.08 TFLOPs) against FL’s full-model client (15–25 GB, ∼158 TFLOPs, 2000 times heavier), and the OpenBioLLM-8B and cross-dataset results (0.662 average) show gains hold across backbones and unseen data. The evaluation covers four survey domains with 500-sample test sets and order-of-magnitude compute estimates. On privacy, only cutlayer activations and gradients leave each institution; we do not claim these tensors resist reconstruction, and defending against model-inversion attacks is outside our scope.

## H. Future Work

Future work will focus on stronger privacy guarantees, broader validation, and practical deployment. This includes protecting cut-layer representations with differential privacy and secure aggregation, evaluating larger and more diverse clinical datasets, and measuring computation, communication, and runtime costs. Further studies should examine domain-specific models, uncertainty, fairness, and clinicianin-the-loop validation. Multi-institution deployment will also be important for testing the framework under real governance and infrastructure constraints.

![](images/f49005f2099b4944dd754c8c43cce8e6dd04edd9b7d7b915566f8eb30cdf7c72.jpg)  
Fig. 5: FT-Transformer results across datasets.

## VI. CONCLUSION

We presented a privacy-preserving SL framework, with a LoRA-adapted LLM as a collaborative encoder, that predicts mental distress from survey data, letting institutions share predictive knowledge while the responses remain local. Central to the design is a schema-aware serialization that maps heterogeneous records into a shared representation. Across a $3 \times 3$ grid of volume imbalance and schema heterogeneity, it averaged 0.708 ANLS, stayed stable under increasing heterogeneity, and beat a federated baseline in eight of nine settings while keeping per-client compute three orders of magnitude lighter. It also surpassed zero- and fewshot prompting, stayed competitive with centralized LoRA through a single shared model, generalized across LLM families, and remained effective from a small training pool.

## VII. ACKNOWLEDGEMENT

The authors acknowledge the National Artificial Intelligence Research Resource (NAIRR) Pilot for contributing to this research result with NCSA Delta GPU access (NAIRR260054).

## REFERENCES

[1] W. H. Organization, World mental health report: Transforming mental health for all. World Health Organization, 2022.

[2] P. Cuijpers, A. Javed, and K. Bhui, “The who world mental health report: a call for action,” The British Journal of Psychiatry, vol. 222, 2023.

[3] P. F. Lovibond and S. H. Lovibond, “The structure of negative emotional states: Comparison of the depression anxiety stress scales (dass) with the beck depression and anxiety inventories,” Behaviour research and therapy, vol. 33, no. 3, pp. 335–343, 1995.

[4] Substance Abuse and Mental Health Services Administration, “Key substance use and mental health indicators in the united states: Results from the 2022 national survey on drug use and health,” Center for Behavioral Health Statistics and Quality, Substance Abuse and Mental Health Services Administration, Rockville, MD, Tech. Rep. HHS Publication No. PEP23-07-01-006, NSDUH Series H-58, 2023.

[5] A. B. Shatte, D. M. Hutchinson, and S. J. Teague, “Machine learning in mental health: a scoping review of methods and applications,” Psychological medicine, vol. 49, no. 9, pp. 1426–1448, 2019.

[6] G. Thornicroft, C. Sunkel, A. A. Aliev, S. Baker, E. Brohan, R. El Chammay, K. Davies, M. Demissie, J. Duncan, W. Fekadu et al., “The lancet commission on ending stigma and discrimination in mental health,” The Lancet, vol. 400, no. 10361, pp. 1438–1480, 2022.

[7] A. Act et al., “Health insurance portability and accountability act of 1996,” Public law, vol. 104, no. 191, pp. 1–16, 1996.

[8] P. Voigt and A. Von dem Bussche, “The eu general data protection regulation (gdpr),” A practical guide, 1st ed., Cham: Springer International Publishing, vol. 10, no. 3152676, pp. 10–5555, 2017.

[9] D. Gao, X. Yao, and Q. Yang, “A survey on heterogeneous federated learning,” arXiv preprint arXiv:2210.04505, 2022.

[10] C. Ye, G. Lu, H. Wang, L. Li, S. Wu, G. Chen, and J. Zhao, “Towards cross-table masked pretraining for web data mining,” in Proceedings of the ACM Web Conference 2024, 2024, pp. 4449–4459.

[11] S. Hegselmann, A. Buendia, H. Lang, M. Agrawal, X. Jiang, and D. Sontag, “Tabllm: Few-shot classification of tabular data with large language models,” in International conference on artificial intelligence and statistics. PMLR, 2023, pp. 5549–5581.

[12] T. Dinh, Y. Zeng, R. Zhang, Z. Lin, M. Gira, S. Rajput, J.-y. Sohn, D. Papailiopoulos, and K. Lee, “Lift: Language-interfaced finetuning for non-language machine learning tasks,” Advances in Neural Information Processing Systems, vol. 35, pp. 11 763–11 784, 2022.

[13] J. Yan, B. Zheng, H. Xu, Y. Zhu, D. Chen, J. Sun, J. Wu, and J. Chen, “Making pre-trained language models great on tabular prediction,” in International Conference on Learning Representations, 2024.

[14] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen et al., “Lora: Low-rank adaptation of large language models.” Iclr, vol. 1, no. 2, p. 3, 2022.

[15] A. Gaber, H. Abd-Eltawab, Y. Abuzied, M. ElMahdy, and T. ElBatt, “Federated learning meets llms: Feature extraction from heterogeneous clients,” arXiv preprint arXiv:2510.00065, 2025.

[16] Z. Lin, Y. Zhang, Z. Chen, Z. Fang, X. Chen, P. Vepakomma, W. Ni, J. Luo, and Y. Gao, “Hsplitlora: A heterogeneous split parameterefficient fine-tuning framework for large language models,” IEEE Transactions on Mobile Computing, 2026.

[17] Meta AI, “Llama 3.2: Revolutionizing edge AI and vision with open, customizable models,” https://huggingface.co/meta-llama/ Llama-3.2-3B-Instruct, 2024, llama-3.2-3B-Instruct model card; released September 25, 2024.

[18] M. S. Ankit Pal, “Openbiollms: Advancing open-source large language models for healthcare and life sciences,” https://huggingface. co/aaditya/OpenBioLLM-Llama3-70B, 2024.

[19] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[20] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan et al., “The llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.

[21] Y. Gorishniy, I. Rubachev, V. Khrulkov, and A. Babenko, “Revisiting deep learning models for tabular data,” Advances in neural information processing systems, vol. 34, pp. 18 932–18 943, 2021.

[22] P. Vepakomma, O. Gupta, T. Swedish, and R. Raskar, “Split learning for health: Distributed deep learning without sharing raw patient data,” arXiv preprint arXiv:1812.00564, 2018.

[23] C. Thapa, P. C. M. Arachchige, S. Camtepe, and L. Sun, “Splitfed: When federated learning meets split learning,” in Proceedings of the AAAI conference on artificial intelligence, vol. 36, no. 8, 2022.

[24] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, “Communication-efficient learning of deep networks from decentralized data,” in Artificial intelligence and statistics. Pmlr, 2017.

[25] Z. Lin, X. Hu, Y. Zhang, Z. Chen, Z. Fang, X. Chen, A. Li, P. Vepakomma, and Y. Gao, “Splitlora: A split parameter-efficient fine-tuning framework for large language models,” arXiv preprint arXiv:2407.00952, 2024.

[26] S. S. Khalil, N. S. Tawfik, and M. Spruit, “Federated learning for privacy-preserving depression detection with multilingual language models in social media posts,” Patterns, vol. 5, no. 7, 2024.

[27] Open Sourcing Mental Illness (OSMI), “Mental health in tech survey,” https://osmihelp.org/research, 2014–2023, annual survey data collected 2014–2023. Accessed 2024.

[28] Open-Source Psychometrics Project, “DASS-42 dataset: Depression anxiety stress scales online administration,” https://openpsychometrics. org/ rawdata/, 2019, data collected 2017–2019. Accessed 2024.

[29] S. Opeyemi, “Student depression dataset,” https://www.kaggle.com/ datasets/hopesb/student-depression-dataset, Nov. 2024, kaggle dataset, last accessed August 2026.

[30] B. Jikadara, “Mental health dataset,” https://www.kaggle.com/datasets/ bhavikjikadara/mental-health-dataset, 2023, kaggle dataset, last accessed August 2026.