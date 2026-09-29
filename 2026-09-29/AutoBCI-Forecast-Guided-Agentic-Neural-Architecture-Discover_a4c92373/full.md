# AutoBCI: Forecast-Guided Agentic Neural Architecture Discovery for EEG-Based Brain–Computer Interfaces

Muyun Jiang<sup>1</sup>, Yi Ding<sup>1</sup>, Wei Zhang<sup>1</sup>, Jinbo Chen<sup>1</sup>, Chenyu Liu<sup>1</sup>, Zhenjie Yang<sup>2</sup>, Yuxin Li<sup>1</sup>, Jingyuan Chen<sup>1</sup>, Yuhao Lu<sup>1</sup>, Yong Li<sup>3</sup>, Shuailei Zhang<sup>1</sup>, Cuntai Guan<sup>1</sup>

<sup>1</sup>Nanyang Technological University, Singapore <sup>2</sup>The University of Hong Kong, Hong Kong, China <sup>3</sup>Southeast University, China

![](images/b2d09c8b038801e56f154392d4f39abbf456c99b4e56f0525637f3d13d2d9e13.jpg)  
Figure 1. AutoBCI framework overview. The Designer Agent uses PGAD to generate and refine EEG architectures; the Forecaster Agent uses PEEK to predict performance and guide continued training. Validation feedback guides subsequent discovery rounds.

## Abstract

EEG-based brain–computer interfaces support a broad range of applications, yet designing decoding architectures that perform well across diverse tasks remains challenging. We introduce AutoBCI, an agentic framework in which a Designer Agent and a Forecaster Agent support the discovery and selection of EEG decoding architectures across tasks. The Designer Agent performs Pool-Guided Architecture Discovery (PGAD), generating and refining architectures through training and validation across multiple EEG tasks, such as emotion recognition, motor imagery, and sleep staging. The Forecaster Agent performs Performance Estimation from Early Knowledge (PEEK), using architecture code, the training protocol, and early learning curves to predict full-budget validation performance and select promising candidates for continued training. Across 14 EEG datasets spanning motor imagery, emotion recognition, and sleep staging, we evaluate AutoBCI with six LLMs, including Opus 5.5 and GPT 5.6 Sol, and compare the architectures selected by the search procedure against ten baselines: six conventional EEG models and four foundation models. The architecture discovered by AutoBCI with Claude Opus 5.5 achieves 64.16% average test balanced accuracy (bAcc), compared with 63.87% for REVE, the strongest baseline on this metric. Using ten observed epochs, PEEK reduces mean absolute error in predicting average validation bAcc from 2.20 to 1.36 percentage points, a 38.1% reduction relative to the best-observed-score baseline.

## 1 Introduction

Electroencephalography (EEG) is a key modality for non-invasive brain–computer interfaces. EEG decoding spans tasks with diferent spatial coverage, temporal scales, and label spaces. For over a decade, researchers have encoded domain knowledge through selected frequency bands, temporal windows, and spatial filters, such as common spatial patterns for motor-imagery decoding (Schirrmeister et al., 2017). Deep learning learns features directly, but model structures still reflect human design: EEGNet uses compact spatiotemporal convolutions (Lawhern et al., 2018), TSception captures multiple temporal scales and spatial asymmetry (Ding et al., 2022), and DeepSleepNet models sleep-stage dependencies with convolutional and recurrent layers (Supratak et al., 2017). These advances demonstrate the value of EEG-specific architectural design, while selecting suitable structures across tasks remains a manual burden. This motivates automated discovery of architectures that perform well across diferent EEG tasks.

Agentic AI ofers a way to reduce the manual efort involved in working across these settings. Existing systems demonstrate that agents can automate EEG analysis workflows. EEGAgent supports signal exploration, event detection, and report generation (Zhao et al., 2026); CogEEGAgent translates user questions into registered analyses with independent confirmation (Hou et al., 2026); and NS-Copilot coordinates specialized agents and pretrained models for neuroscience workflows (Liu et al., 2026). These capabilities improve access to analysis tools, but these systems focus on automating analysis rather than designing decoding architectures. Reducing the model-design burden requires extending this automation to the architectures themselves, using experimental feedback to guide structural changes.

Work on neural architecture search takes this further, showing that architecture design can be automated through search and feedback. CTNAS-EEG introduces a compatible search space and constrained search procedure across EEG tasks (Duan et al., 2023). NeuroWeaver evolves executable EEG pipelines using domain-informed initialization and performance, novelty, and eficiency objectives (Wang et al., 2026). More broadly, EvoPrompting evolves architecture code, while NADER uses collaborating agents to guide architectural modifications (Chen et al., 2023; Yang et al., 2025). However, success on separately optimized tasks does not establish the quality of a shared architecture. Our goal is to discover a single architecture that performs well across EEG tasks. We therefore train and validate each candidate separately for each task, then use the combined validation results to guide architecture selection and refinement. In this study, we evaluate this approach on three common EEG tasks: emotion recognition, motor imagery, and sleep staging. Evaluating each proposal across these tasks increases the training cost of discovery, making candidate screening an important part of the search.

To reduce this evaluation cost, prior work exploits the finding that partial training evidence can support early performance prediction. Learning-curve extrapolation identifies unpromising runs (Domhan et al., 2015), and performance predictors combine architecture features, hyperparameters, and partial validation trajectories (Baker et al., 2017). LLM-ENAS-PKB further uses language models as architecture-performance surrogates (Weilin et al., 2026). However, forecast accuracy alone does not establish reliable candidate selection. For screening to be useful, promising candidates must remain in the search. We therefore use architecture code and early learning curves to predict how well a candidate will perform after full training. These forecasts help identify promising candidates for continued training. We evaluate both prediction accuracy and whether the selected candidates retain the architectures that perform best after full training.

To address these challenges, we introduce AutoBCI, an agentic framework containing two agents: a Designer Agent that uses Pool-Guided Architecture Discovery (PGAD) to generate and refine EEG architectures, and a Forecaster Agent that uses Performance Estimation from Early Knowledge (PEEK) to predict candidate performance and guide continued training, as shown in Figure 1. First, PGAD automates architecture design through executable code generation and refinement. The agent proposes models and uses measured validation feedback to revise their structure, extending its role from analysis workflow execution to model design. Second, PGAD selects shared architectures using joint evidence across tasks. Each candidate is trained and validated independently on emotion recognition, motor imagery, and sleep staging. An equal-task validation objective ranks a persistent candidate pool, from which strong parents are refined alongside fresh proposals. Third, PEEK guides training allocation using early performance forecasts. It combines architecture code, the training protocol, and early learning curves to predict full-budget validation scores and rank candidates for continued training. These mechanisms connect model generation, cross-task evaluation, and candidate screening within one discovery framework.

Our contributions are threefold. First, we introduce AutoBCI and develop PGAD for agentic EEG architecture discovery, using a persistent candidate pool and cross-task validation feedback to guide code generation and refinement. Second, we develop PEEK to forecast full-budget performance from architecture code and early learning curves, enabling candidate screening before training is complete. Third, we evaluate six LLM-guided searches across 14 datasets and compare their champions with ten baselines. The searches produce 269 architectures trained on all three task families; the strongest champion achieves 64.16% average test bAcc, exceeding the strongest baseline by 0.29 percentage points. PEEK reduces forecast mean absolute error by 38.1% after ten epochs. Retaining three candidates per round preserves a full-budget winner in 91.7% of rounds, with an estimated 44.9% reduction in training epochs.

## 2 Method

AutoBCI combines a Designer Agent for architecture discovery with a Forecaster Agent for early candidate selection. The Designer Agent uses PGAD to generate architectures, evaluate separately trained instances across EEG tasks, and refine candidates using their combined validation scores. The Forecaster Agent uses PEEK to predict full-budget validation performance from architecture code, the training protocol, and early learning curves, helping select candidates for continued training.

## 2.1 Pool-Guided Architecture Discovery

PGAD organizes architecture discovery around a persistent pool of evaluated candidates. Starting from an initial set of proposals, the Designer Agent uses validation feedback across tasks to identify strong architectures and propose refinements. Each subsequent round combines these refinements with fresh designs, allowing the search to build on previous discoveries while exploring alternatives. The procedure consists of candidate generation, cross-task evaluation, and pool-based selection and refinement, as detailed below.

Candidate generation. PGAD runs for R discovery rounds, indexed by $r \in \{ 1 , \ldots , R \}$ , with N candidate architectures proposed per round. Round r = 1 contains N fresh proposals from the configured LLM. Each proposal contains complete PyTorch model code and a short architectural hypothesis. The model interface accepts configurable channel, sample, and class counts, mapping a batch of EEG signals to class logits. Before training, the framework checks the response format and model interface. Candidate code is preserved as generated; implementation errors are recorded as candidate failures.

Evaluation across tasks. For each candidate, the same architecture source is trained independently on K task families, indexed by $k \in \{ 1 , \ldots , K \}$ , with separate model parameters and checkpoints for each task. Our evaluation uses emotion recognition, motor imagery, and sleep staging (K = 3). Let $\mathcal { D } _ { k }$ denote the datasets in task family k, E the full training budget in epochs, and $b _ { a , d } ( e )$ denote the validation balanced accuracy of architecture a on dataset d at epoch e. At each epoch, datasets within a task receive equal weight. The task score is the highest such mean over the training run:

$$
s _ { a , k } = \operatorname* { m a x } _ { 1 \leq e \leq E } \frac { 1 } { | \mathscr { D } _ { k } | } \sum _ { d \in \mathscr { D } _ { k } } b _ { a , d } ( e ) .\tag{1}
$$

Thus, all datasets within a task share one selected checkpoint epoch, while diferent tasks may select diferent epochs. The architecture score assigns equal weight to the K task scores:

$$
S _ { a } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } s _ { a , k } .\tag{2}
$$

An architecture is eligible for selection only after its training runs on all K tasks complete E epochs.

![](images/e74a56076a6da7cec206125db5b9c5354234abe9d114757dae28c21299a7cd35.jpg)  
Figure 2. PGAD candidate generation protocol. Left: Architectures evaluated in previous discovery rounds. Middle: A shared candidate pool ranked by combined validation performance. Right: Top-ranked parents provide source code, learning curves, and task-specific validation scores for targeted refinements. These refinements are combined with fresh architecture proposals to form the next candidate batch.

Selection and refinement. Before round $r > 1$ , eligible architectures from all preceding rounds form a shared pool $\mathcal { P } _ { r - 1 }$ of size $M _ { r - 1 } = | \mathcal { P } _ { r - 1 } | \le N ( r - 1 )$ , as illustrated in Figure 2. The round selects the P highest-scoring architectures by $S _ { a }$ as parents and requests $Q$ refinements per parent. Each refinement receives the parent’s source code, learning curves, and task-specific validation results, and is prompted to change one architectural component and describe the intended change. Alongside these PQ refinements, the LLM generates U fresh explorations, with

$$
N = P Q + U .\tag{3}
$$

Thus, the search requests RN architectures over $R$ rounds. The highest-scoring eligible architecture in $\mathcal { P } _ { R }$ is the final champion. Validation data determine parent and champion selection; the held-out test split is used only for a separate final evaluation.

## 2.2 Performance Estimation from Early Knowledge (PEEK)

PEEK uses early performance forecasts to select promising architectures for continued training. After a short initial training period, the Forecaster Agent combines each candidate’s architecture code, training protocol, and observed learning curves to estimate its full-budget validation performance. Architecture code describes the model’s design, while the curves show how that design learns on the target task. PEEK uses this complementary evidence to rank candidates and allocate the remaining training budget to the predicted leaders. The procedure has two steps: performance forecasting and candidate selection.

Performance forecasting. For task k, each candidate a first trains for $e _ { 0 } < E$ epochs. Let $C _ { a }$ denote its source code and $\mathcal { H } _ { a , k } ^ { 1 : e _ { 0 } }$ its observed training and validation history. Given the training protocol $\pi _ { k }$ , PEEK produces

$$
\begin{array} { r } { \widehat { s } _ { a , k } = F _ { \mathrm { L L M } } \Big ( C _ { a } , \mathcal { H } _ { a , k } ^ { 1 : e _ { 0 } } , \pi _ { k } \Big ) , } \end{array}\tag{4}
$$

where the target $s _ { a , k }$ is the best equal-dataset validation balanced accuracy at a common checkpoint within E epochs (Equation (1)). Aligning the forecast with this selection criterion lets PEEK assess a candidate’s potential over the remaining budget, including improvement beyond its early score. Predictions lie in [0, 1] and must be at least the best validation score already observed. The LLM receives only candidate code, the specified protocol, and the observed portion of the learning history. For shared-architecture selection, we average the task forecasts using the same equal-task weighting as Equation (2):

$$
\widehat { S } _ { a } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \widehat { s } _ { a , k } .\tag{5}
$$

Candidate selection. To select shared architectures, PEEK ranks candidates by the combined forecast ${ \widehat { S } } _ { a }$ (Equation (5)). The screening policy retains the $N _ { \mathrm { k e e p } }$ highest-ranked candidates and continues each on all K tasks to E epochs, where $1 \leq N _ { \mathrm { k e e p } } \leq N$ . Only candidates that complete all K tasks enter the PGAD pool, using their measured scores $S _ { a } ;$ partially trained candidates remain ineligible for parent and champion selection. For N candidates initially trained for $e _ { 0 }$ epochs on each task, the nominal cost is $K [ N e _ { 0 } + N _ { \mathrm { k e e p } } ( E - e _ { 0 } ) ]$ task-epochs, compared with KNE for full training. The retention–cost analysis sweeps $N _ { \mathrm { k e e p } } = 1 , \ldots , 6 .$ , with the abstract reporting the three-candidate operating point. Separate task-level diagnostics rank candidates by ${ \widehat { s } } _ { a , k }$ and measure top-one and top-two retention for each task.

## 3 Experimental Setup

Our experiments evaluate architecture discovery and early candidate selection on shared EEG datasets. PGAD evaluates candidates with a 40-epoch budget. Completed runs establish the full-budget scores and winners needed to evaluate PEEK. The forecaster receives only architecture code, the training protocol, and early learning curves; full-run outcomes provide ground truth for retrospective evaluation of forecast accuracy and winner retention.

## 3.1 Dataset construction

EEG tasks difer in their prediction targets and recording characteristics, so we organize the 14 datasets into motor imagery (MI), emotion recognition (ER), and sleep staging (SS), as shown in Table 1. Datasets within each family are pooled to train one model instance per task. Channel layouts and labels are harmonized within a task. MI and ER recordings are mapped to the 65-channel standard 10–10 system template used by LEAF (Jiang et al., 2025), while the sparser sleep recordings use an eight-channel template. Validation and test scores are reported separately for each dataset.

All candidates use the same fixed training, validation, and test partitions and sleep subsets. Training fits model parameters, validation selects checkpoints and architectures, and held-out test data are used only for final evaluation. Preprocessing and split construction are detailed in Supplementary A.5.

## 3.2 Architecture search protocol

On these datasets, PGAD searches share a common budget to support comparison of architecture discovery across LLMs. Each search comprises R = 6 discovery rounds with N = 8 candidates per round, evaluated across K = 3 task families (MI, ER, and SS). This gives RN = 48 architecture proposals and up to KRN = 144 independent task-training runs. Round 1 contains N = 8 fresh proposals. For later rounds, $P = 2$ parents each receive $Q = 3$ refinements, and U = 2 fresh proposals complete the batch $( N = P Q + U = 8 )$ . The parameter constraint is strictly fewer than one million trainable parameters in every task-specific instantiation. Recurrent architectures are excluded, and each candidate is trained from scratch under the same task-specific protocol.

Each search uses one of the six language models listed in Table 2. The initial proposals are guided by the task names and class categories, input-shape constraints, training protocol, and architectural constraints. In subsequent rounds, the two selected parents provide architecture code and validation learning curves for all three tasks, allowing refinements to address their observed strengths and weaknesses. Fresh proposals preserve an opportunity to explore architectures beyond the selected parents. Candidates undergo validity and training-compatibility checks before training. Only architectures that successfully complete all three tasks are eligible for parent and final champion selection. Implementation interfaces, generation settings, and detailed checks are provided in Supplementary $_ { \mathrm { A . 6 ; } }$ complete prompts appear in Supplementary A.10.

The six LLMs run architecture searches through their respective command-line interfaces, as shown in Table 2.

Table 1. EEG datasets, task-specific inputs, and class counts.
<table><tr><td>Task</td><td>No.</td><td>Datasets</td><td>Unified input</td><td>Classes</td></tr><tr><td rowspan="8">Motor Imagery</td><td>1 2</td><td>BCIC-IV-2a (Tangermann et al., 2012)</td><td rowspan="8"></td><td rowspan="8">Left, Right, Foot, Tongue, Cylin, Sphe, Lumbrical (7 classes)</td></tr><tr><td></td><td>OpenBMI-MI (Lee et al., 2019)</td></tr><tr><td>3 4</td><td>BCIC-Upperlimb (Jeong et al., 2022)</td></tr><tr><td>5</td><td>Cho2017 (Cho et al., 2017)</td></tr><tr><td></td><td>65 ch × 4 s HighGamma (Schirrmeister et al., 2017)</td></tr><tr><td>6</td><td>PhysioNet-MI (Schalk et al., 2004)</td></tr><tr><td>7</td><td>SHU-MI (Ma et al., 2022)</td></tr><tr><td>8</td><td>Shin2017A (Shin et al., 2016)</td></tr><tr><td rowspan="3">Emotion Recognition</td><td>9</td><td>SEED (Duan et al., 2013)</td><td rowspan="3">65 ch × 4 s</td><td rowspan="3">Positive, Neutral, Negative, Sad, Fear, Happy, Disgust (7 classes)</td></tr><tr><td>10</td><td>SEED-IV (Zheng et al., 2018)</td></tr><tr><td>11</td><td>SEED-V (Liu et al., 2021)</td></tr><tr><td rowspan="3">Sleep Staging</td><td>12</td><td>Sleep-EDF (Kemp et al., 2000)</td><td rowspan="3">8 ch × 30 s</td><td rowspan="3">Wake, N1, N2, N3, REM (5 classes)</td></tr><tr><td>13</td><td>ISRUC (Khalighi et al., 2016)</td></tr><tr><td>14</td><td>HMC (Alvarez-Estevez &amp; Rijsman, 2021)</td></tr></table>

Table 2. LLMs and execution settings for architecture discovery.
<table><tr><td>Model tier</td><td>Provider</td><td>LLM</td><td>Agent interface</td><td>Input (USD/1M)</td><td>Output (USD/1M)</td></tr><tr><td></td><td>Anthropic DeepSeek</td><td>Claude Sonnet 5 DeepSeek V4.1 Flash</td><td>Claude Code</td><td>2.00</td><td>10.00</td></tr><tr><td>Flash</td><td>Google</td><td>Gemini 3.8 Flash</td><td>Claude Code Gemini CLI</td><td>0.30 0.75</td><td>1.20 3.75</td></tr><tr><td></td><td>Alibaba</td><td>Qwen3.8 Flash</td><td>Claude Code</td><td>0.15</td><td>0.47</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Flagship</td><td>OpenAI</td><td>GPT-5.6 Sol</td><td>Codex CLI</td><td>4.00</td><td>20.00</td></tr><tr><td></td><td>Anthropic</td><td>Claude Opus 5.5</td><td>Claude Code</td><td>4.00</td><td>20.00</td></tr></table>

We use the default thinking levels for all models. DeepSeek and Qwen were accessed through Anthropic-compatible endpoints via Claude Code.

## 3.3 Training and evaluation protocol

To compare the architectures generated during search, all candidates follow a common training and evaluation protocol. Each candidate is trained independently on the three task families for 40 epochs using AdamW, a batch size of 512, and full BF16 precision. Validation performance selects the checkpoint for each task. Experiments run in PyTorch on NVIDIA RTX PRO 6000 Blackwell GPUs, with independent candidate/task jobs executed in parallel.

Performance is measured by balanced accuracy (bAcc), the mean recall over classes present in each split. Validation bAcc selects task checkpoints with equal dataset weights and architectures with equal task weights. Selected architectures are tested using these checkpoints, reporting bAcc, class-support-weighted F1 (wF1), and Cohen’s Kappa with the same averaging; test data remain excluded from search and selection. Prompts, code, parent lineage, splits, configurations, seeds, learning curves, checkpoints, and outcomes are retained for reproducibility; further training and execution details appear in Supplementary A.7.

## 3.4 Baseline models and comparison settings

To assess the discovered architectures against established EEG decoders, we compare with ten baselines. The six conventional models are EEGNet (Lawhern et al., 2018), DeepConvNet (Schirrmeister et al., 2017), TSception (Ding et al., 2022), EEG-Conformer (Song et al., 2022), AttnSleep (Eldele et al., 2021), and FAST (Jiang et al., 2026). The four foundation models are LaBraM (Jiang et al., 2024), CBraMod (Wang et al., 2025), LEAF (Jiang et al., 2025), and REVE (El Ouahidi et al., 2026). Conventional models are trained from scratch, while foundation models are fine-tuned from pretrained checkpoints. All baselines use the same task partitions and validation-based checkpoint selection as AutoBCI. Training budgets, optimization settings, and input adaptations are detailed in Supplementary A.4.

## 4 Results

We evaluate the discovered architectures on held-out test data and analyze architecture discovery and early performance forecasting using validation data.

## 4.1 Performance of discovered architectures

To assess full-budget search, we compare validation-selected champions on held-out tests, weighting the three tasks equally; baseline results average three seeds. Opus achieves the highest average bAcc (64.16%), exceeding REVE by 0.29 percentage points (pp), while REVE leads on average wF1 and Cohen’s Kappa, as shown in Table 3. Task leaders difer: Opus leads MI, REVE leads emotion, and AttnSleep has the highest sleep bAcc. Discovered champions span 57.69–64.16% average bAcc; in these searches, Gemini and DeepSeek achieve higher average test bAcc than GPT.

Table 3. Performance comparison of small-model baselines, foundation models, and AutoBCI-discovered architectures. Results are evaluated on held-out test sets for motor imagery, emotion recognition, and sleep staging. bAcc and wF1 are percentages. Per column: first, second, third.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Spec</td><td colspan="3">Motor Imagery</td><td colspan="3">Emotion</td><td colspan="3">Sleep</td><td colspan="3">Average</td></tr><tr><td>bAcc</td><td>wF1</td><td>Kappa</td><td>bAcc</td><td>wF1</td><td>Kappa</td><td>bAcc</td><td>wF1</td><td>Kappa</td><td>bAcc</td><td>wF1</td><td>Kappa</td></tr><tr><td>EEGNet (Lawhern et al., 2018)</td><td>8.8K</td><td>57.83</td><td>57.12</td><td>0.233</td><td>37.97</td><td>34.48</td><td>0.164</td><td>69.09</td><td>70.60</td><td>0.640</td><td>54.96</td><td>54.07</td><td>0.345</td></tr><tr><td>TSception (Ding et al., 2022)</td><td>17.5K</td><td>54.58</td><td>53.63</td><td>0.167</td><td>43.93</td><td>45.91</td><td>0.256</td><td>66.75</td><td>70.74</td><td>0.648</td><td>55.09</td><td>56.76</td><td>0.357</td></tr><tr><td>DCN (Schirrmeister et al., 2017)</td><td>317K</td><td>66.70</td><td>66.36</td><td>0.396</td><td>42.73</td><td>38.03</td><td>0.215</td><td>73.89</td><td>76.68</td><td>0.697</td><td>61.10</td><td>60.35</td><td>0.436</td></tr><tr><td>FAST (Jiang et al., 2026)</td><td>475K</td><td>58.59</td><td>58.31</td><td>0.243</td><td>35.39</td><td>36.94</td><td>0.137</td><td>72.26</td><td>74.99</td><td>0.686</td><td>55.41</td><td>56.75</td><td>0.355</td></tr><tr><td>EEG-Conformer (Song et al., 2022)</td><td>2.13M</td><td>66.32</td><td>66.21</td><td>0.396</td><td>48.08</td><td>48.92</td><td>0.302</td><td>72.34</td><td>76.07</td><td>0.701</td><td>62.25</td><td>63.73</td><td>0.466</td></tr><tr><td>AttnSleep (Eldele et al., 2021)</td><td>2.32M</td><td>63.73</td><td>63.41</td><td>0.344</td><td>40.28</td><td>40.79</td><td>0.201</td><td>74.24</td><td>76.89</td><td>0.695</td><td>59.42</td><td>60.36</td><td>0.413</td></tr><tr><td>CBraMod (Wang et al., 2025)</td><td>4.89M</td><td>46.18</td><td>43.66</td><td>0.028</td><td>40.59</td><td>40.40</td><td>0.201</td><td>69.02</td><td>72.65</td><td>0.657</td><td>51.93</td><td>52.24</td><td>0.295</td></tr><tr><td>LaBraM (Jiang et al., 2024)</td><td>5.82M</td><td>47.68</td><td>44.38</td><td>0.053</td><td>49.24</td><td>49.85</td><td>0.323</td><td>70.80</td><td>75.24</td><td>0.677</td><td>55.91</td><td>56.49</td><td>0.351</td></tr><tr><td>LEAF (Jiang et al., 2025)</td><td>27.41M</td><td>67.17</td><td>67.05</td><td>0.408</td><td>49.13</td><td>51.19</td><td>0.327</td><td>72.57</td><td>75.85</td><td>0.687</td><td>62.96</td><td>64.70</td><td>0.474</td></tr><tr><td>REVE (El Ouahidi et al., 2026)</td><td>69.19M</td><td>67.80</td><td>67.49</td><td>0.424</td><td>51.15</td><td>51.87</td><td>0.345</td><td>72.67</td><td>77.99</td><td>0.717</td><td>63.87</td><td>65.78</td><td>0.495</td></tr><tr><td>AutoBCI (Claude Sonnet 5)</td><td>Flash</td><td>59.13</td><td>58.64</td><td>0.257</td><td>47.48</td><td>47.84</td><td>0.292</td><td>70.96</td><td>75.47</td><td>0.683</td><td>59.19</td><td>60.65</td><td>0.411</td></tr><tr><td>AutoBCI (DeepSeek V4.1 Flash)</td><td>Flash</td><td>61.74</td><td>61.44</td><td>0.304</td><td>47.62</td><td>48.72</td><td>0.296</td><td>72.09</td><td>76.12</td><td>0.696</td><td>60.49</td><td>62.09</td><td>0.432</td></tr><tr><td>AutoBCI (Gemini 3.8 Flash)</td><td>Flash</td><td>65.31</td><td>65.04</td><td>0.371</td><td>45.74</td><td>44.76</td><td>0.259</td><td>73.80</td><td>77.16</td><td>0.701</td><td>61.62</td><td>62.32</td><td>0.444</td></tr><tr><td>AutoBCI (Qwen3.8 Flash)</td><td>Flash</td><td>58.39</td><td>57.28</td><td>0.242</td><td>43.00</td><td>44.26</td><td>0.237</td><td>71.69</td><td>76.08</td><td>0.691</td><td>57.69</td><td>59.21</td><td>0.390</td></tr><tr><td>AutoBCI (GPT-5.6 Sol)</td><td>|Flagship|</td><td>63.69</td><td>63.48</td><td>0.337</td><td>44.25</td><td>44.76</td><td>0.244</td><td>73.11</td><td>76.19</td><td>0.689</td><td>60.35</td><td>61.48</td><td>0.423</td></tr><tr><td>AutoBCI (Claude Opus 5.5)</td><td>Flagship</td><td>68.55</td><td>68.38</td><td>0.435</td><td>50.10</td><td>51.06</td><td>0.329</td><td>73.83</td><td>77.55</td><td>0.712</td><td>64.16</td><td>65.67</td><td>0.492</td></tr></table>

## 4.2 Architecture discovery across language models

To trace architecture discovery, we examine six search rounds in Figure 3. Opus finishes highest at 66.26% combined validation bAcc (+2.98 pp), followed by Gemini (63.59%) and DeepSeek (63.10%); Qwen improves most (+16.90 pp) but finishes lowest (58.43%). Opus’s round median rises from 61.62% to 65.47%, whereas Sonnet’s fluctuates despite an improving best score. Both refinement and fresh exploration yield champions: Opus refines multiscale convolutions with channel gating and mean–standard-deviation pooling, while Gemini’s fresh design combines filter banks, dilated residual blocks, and attention pooling. DeepSeek uses multiscale filters with covariance pooling. Model definitions and round-by-round identities appear in Supplementary A.9 and Supplementary Figure 6.

Preserving progress and refinement paths. To examine the pool’s role, we trace incumbent scores and parent reuse across all six searches in Table 4. DeepSeek’s round-2 and Sonnet’s round-4 best scores fall below their prior incumbents by 0.030 and 0.221 pp; the pool preserves the stronger candidates. All six searches reuse parents older than the preceding round. Opus, GPT, and DeepSeek champions follow rounds 1→3→4→5→6, while Sonnet’s follows 1→2→5→6. These refinement paths include parents selected from earlier rounds. Gemini and Qwen instead discover fresh round-6 champions, showing how historical retention supports continued refinement alongside fresh exploration.

![](images/f289919bb15c8c29d2cc3b23e348a093ddf3e43ccfb5f8071d514bc7118700ad.jpg)  
Figure 3. Discovery trajectories across six LLM-guided searches. Dots show candidates completed on all three tasks; dashed lines mark round medians, and solid lines track the best combined validation bAcc so far. Labels give the incumbent scores. The vertical axis is truncated at 50%; round-by-round architecture identities appear in Supplementary Figure 6.

Table 4. Historical retention and parent reuse in recorded searches. Declines compare a round’s best combined validation bAcc with the prior incumbent. Older-parent rounds use at least one parent from before the immediately preceding round. Arrows identify parent-to-child generation rounds in the champion’s ancestry.
<table><tr><td>Model</td><td>Round-best declines</td><td>Older-parent rounds</td><td>Older-parent link in champion lineage</td></tr><tr><td>Claude Opus 5.5</td><td>None</td><td>R3</td><td>Yes (1 → 3)</td></tr><tr><td>Claude Sonnet 5</td><td>R4 (0.221 pp)</td><td>R3, R4, R5, R6</td><td>Yes (2 → 5)</td></tr><tr><td>DeepSeek V4.1 Flash</td><td>R2 (0.030 pp)</td><td>R3, R4, R6</td><td>Yes (1 → 3)</td></tr><tr><td>Gemini 3.8 Flash</td><td>None</td><td>R3</td><td>None (fresh)</td></tr><tr><td>GPT-5.6 Sol</td><td>None</td><td>R3</td><td>Yes (1 → 3)</td></tr><tr><td>Qwen3.8 Flash</td><td>None</td><td>R5</td><td>None (fresh)</td></tr></table>

## 4.3 Early performance forecasting and candidate selection

To evaluate early screening, we compare PEEK and the best-observed-score baseline against full-budget validation scores on paired successful rounds. Each LLM initially forecasts its own candidates from architecture code, the training protocol, and the observed learning-curve prefixes. After ten epochs, PEEK reduces combined-score mean absolute error (MAE) from 2.20 to 1.36 pp across 36 rounds (38.1%), anticipating gains beyond the observed scores, as shown in Figure 4A–B. The baseline uses each task’s highest equal-dataset mean validation bAcc within the observed epochs as its full-budget prediction, then averages these predictions equally across the three tasks.

Forecasting gains vary across LLMs and observation windows, as shown in Figure 5. At ten epochs, PEEK lowers MAE for five of six searches, including Opus (3.27 to 0.79 pp) and GPT (2.52 to 0.67 pp), but increases it for DeepSeek (1.97 to 3.63 pp). By epoch 16, the baseline has lower aggregate MAE (0.88 versus 1.18 pp; 33 paired rounds), although PEEK remains better for Opus, GPT, and Sonnet.

To assess selection quality, we rank candidates within each task by ${ \widehat { s } } _ { a , k }$ . At ten epochs, retaining two rather than one increases task-average winner retention from 56.5% to 80.6%, as shown in Figure 4C–D; top-two retention is 75.0% for MI and emotion and 91.7% for sleep. At 16 epochs, top-one and top-two retention reach 75.8% and 91.9% across 33 paired rounds. The next subsection evaluates shared-architecture selection using combined scores ${ \widehat { S } } _ { a }$

Separating the Designer and Forecaster. To test selection across LLMs, DeepSeek, GPT, and Gemini forecast 91 Opus and Sonnet candidates across 12 rounds using identical code and ten-epoch inputs to the self-forecasts. Each external forecaster achieves 72.2% task-average top-two retention, versus 75.0% for self-forecasting; top-one retention is 47.2–50.0% versus 50.0%, as shown in Table 5. External forecasts match or improve top-two retention on Opus candidates but reduce it on Sonnet candidates, supporting separate Designer and Forecaster models with generator-dependent efects.

![](images/49d70e9b8dbaff06d6d91f27c3a55422e0b102c0c65bcaea0d62de2c81f7d3f3.jpg)

![](images/d28c68287fd288675cf14edb221f99b4d754e3660e6f85aa486ae8aa97e59b76.jpg)

![](images/b354e82bbd8e08c68729ec54cc1cc412f25aa58a24ce92d0e96f6ee119da5f14.jpg)

![](images/66d8652036bfcd0c70cce110b5c96621ac3823af49c4b4fffcb997a5c4da9801.jpg)  
Figure 4. Early forecasts and candidate retention. (A–B) Ten-epoch forecasts versus actual combined validation bAcc for 269 architectures, averaging task-wise best-within-40 scores. (C–D) Top-one and top-two task-winner retention across observation windows; dark curves average the three tasks equally. Dotted lines mark ten epochs.

![](images/fad23b1a80f94de2aec6d2ac266446dd01f5371ee22b7d23417860bd4334896d.jpg)  
Figure 5. Forecast error across LLMs and observation windows. Combined validation bAcc MAE (pp), averaged over candidates within rounds and equally over paired rounds. Cohorts may vary by window; dotted lines mark ten epochs.

Table 5. Selection across Designer and Forecaster models. Top-1/Top-2 task-winner retention (%) at ten epochs, averaged equally over three tasks and six rounds per generator. Self denotes the generating model.
<table><tr><td rowspan="2">Forecaster</td><td colspan="2">Opus candidates</td><td colspan="2">Sonnet candidates</td></tr><tr><td>Top-1</td><td>Top-2</td><td>Top-1</td><td>Top-2</td></tr><tr><td>Best observed score</td><td>50.0</td><td>83.3</td><td>50.0</td><td>55.6</td></tr><tr><td>Self</td><td>50.0</td><td>77.8</td><td>50.0</td><td>72.2</td></tr><tr><td>DeepSeek V4.1 Flash</td><td>50.0</td><td>77.8</td><td>50.0</td><td>66.7</td></tr><tr><td>GPT-5.6 Sol</td><td>44.4</td><td>77.8</td><td>50.0</td><td>66.7</td></tr><tr><td>Gemini 3.8 Flash</td><td>44.4</td><td>83.3</td><td>55.6</td><td>61.1</td></tr></table>

## 4.4 Search cost and winner retention

To estimate screening savings, we retrospectively select up to $k \in \{ 1 , \ldots , 6 \}$ candidates per recorded round by combined forecast $\widehat { S } _ { a }$ at epoch 10. Cost assumes that only selected candidates continue on all three tasks to epoch 40. Selecting two retains a full-budget winner in 83.3% of rounds and would save 54.9% of task-epochs; selecting three gives 91.7% retention and estimated savings of 44.9%, as shown in Table 6. Estimated GPU job-hours fall from 68.58 under full training to 31.86 and 38.85, respectively. The 36 forecasting calls use 1.167M input and 0.120M output tokens and are reused across retention budgets; accounting details appear in Supplementary A.2.

Table 6. Winner retention and estimated screening cost. Epoch-10 forecasts retrospectively select up to k candidates per round across 36 rounds. Completed 40-epoch runs provide ground truth; cost assumes only selected candidates continue. Retention and epoch savings are percentages.
<table><tr><td>Method</td><td>Winner retention</td><td>Full runs</td><td>Task-epochs</td><td>Epoch savings</td><td>GPU job-hours</td></tr><tr><td>Full training</td><td>100.0</td><td>807</td><td>32,280</td><td>0.0</td><td>68.58</td></tr><tr><td>PEEK (k = 1)</td><td>61.1</td><td>108</td><td>11,310</td><td>65.0</td><td>24.52</td></tr><tr><td>PEEK (k = 2)</td><td>83.3</td><td>216</td><td>14,550</td><td>54.9</td><td>31.86</td></tr><tr><td>PEEK (k = 3)</td><td>91.7</td><td>324</td><td>17,790</td><td>44.9</td><td>38.85</td></tr><tr><td>PEEK (k = 4)</td><td>94.4</td><td>429</td><td>20,940</td><td>35.1</td><td>45.53</td></tr><tr><td>PEEK (k = 5)</td><td>97.2</td><td>534</td><td>24,090</td><td>25.4</td><td>52.18</td></tr><tr><td>PEEK (k = 6)</td><td>100.0</td><td>633</td><td>27,060</td><td>16.2</td><td>58.27</td></tr></table>

## 5 Conclusion

AutoBCI combines PGAD for architecture discovery across motor imagery, emotion recognition, and sleep staging with PEEK for early performance forecasting. Across six LLM-guided searches, every search improves its best validation score. The strongest discovered architecture reaches 64.16% average test bAcc, 0.29 pp above REVE, which leads on wF1 and Cohen’s Kappa. After ten epochs, PEEK reduces forecast MAE by 38.1%, although its top-two selection retains fewer combined-score winners than the best-observed-score baseline.

## AI Use Disclosure

Generative AI is part of the research method: the six LLMs in Table 2 propose and refine executable EEG architectures from task specifications and validation feedback, and predict full-budget validation performance from early training evidence. Generated architectures undergo the implementation checks described in Supplementary A.6; their reported decoding results are computed by training and evaluating the models on the stated data partitions. We also used AI assistants to do literature review, and text polishing. The authors take responsibility for the final manuscript, experimental claims, and accompanying artifacts.

## Ethics Statement

This study uses publicly available EEG datasets and involves no participant recruitment or new human-data collection. All experiments consist of computational analyses of existing recordings. We identify no additional ethical concerns arising from these analyses.

## Reproducibility Statement

The main paper and supplementary material describe the datasets, data splits, search procedure, training settings, and evaluation protocol. We provide the forecasting prompts and selected architecture definitions in the supplement. We will release the code, experiment configurations, and supporting materials needed to reproduce all results reported in this paper.

## References

Diego Alvarez-Estevez and Roselyne M Rijsman. Inter-database validation of a deep learning approach for automatic sleep scoring. PloS one, 16(8):e0256111, 2021.

Bowen Baker, Otkrist Gupta, Ramesh Raskar, and Nikhil Naik. Accelerating neural architecture search using performance prediction. arXiv preprint arXiv:1705.10823, 2017.

Angelica Chen, David Dohan, and David So. Evoprompting: Language models for code-level neural architecture search. Advances in neural information processing systems, 36:7787–7817, 2023.

Hohyun Cho, Minkyu Ahn, Sangtae Ahn, Moonyoung Kwon, and Sung Chan Jun. Eeg datasets for motor imagery brain–computer interface. GigaScience, 6(7):gix034, 2017.

Yanna Ding, Zijie Huang, Xiao Shou, Yihang Guo, Yizhou Sun, and Jianxi Gao. Architecture-aware learning curve extrapolation via graph ordinary diferential equation. In Proceedings of the AAAI conference on artificial intelligence, volume 39, pp. 16289–16297, 2025.

Yi Ding, Neethu Robinson, Su Zhang, Qiuhao Zeng, and Cuntai Guan. Tsception: Capturing temporal dynamics and spatial asymmetry from eeg for emotion recognition. IEEE Transactions on afective computing, 14(3):2238–2250, 2022.

Tobias Domhan, Jost Tobias Springenberg, Frank Hutter, et al. Speeding up automatic hyperparameter optimization of deep neural networks by extrapolation of learning curves. In IJCAI, volume 15, pp. 3460–8, 2015.

Ruo-Nan Duan, Jia-Yi Zhu, and Bao-Liang Lu. Diferential entropy feature for eeg-based emotion classification. In 2013 6th international IEEE/EMBS conference on neural engineering (NER), pp. 81–84. IEEE, 2013.

Yiqun Duan, Zhen Wang, Yi Li, Jianhang Tang, Yu-Kai Wang, and Chin-Teng Lin. Cross task neural architecture search for eeg signal recognition. Neurocomputing, 545:126260, 2023.

Yassine El Ouahidi, Jonathan Lys, Philipp Thölke, Nicolas Farrugia, Bastien Pasdeloup, Vincent Gripon, Karim Jerbi, and Giulia Lioi. Reve: A foundation model for eeg-adapting to any setup with large-scale pretraining on 25,000 subjects. Advances in Neural Information Processing Systems, 38:22541–22577, 2026.

Emadeldeen Eldele, Zhenghua Chen, Chengyu Liu, Min Wu, Chee-Keong Kwoh, Xiaoli Li, and Cuntai Guan. An attention-based deep learning approach for sleep stage classification with single-channel eeg. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 29:809–818, 2021.

Stefan Falkner, Aaron Klein, and Frank Hutter. Bohb: Robust and eficient hyperparameter optimization at scale. In International conference on machine learning, pp. 1437–1446. PMLR, 2018.

Dengzhe Hou, Lingyu Jiang, Fangzhou Lin, and Kazunori D Yamada. Cogeegagent: Toward autonomous cognitive eeg analysis with grounded execution and selection-aware verification. arXivpreprint arXiv:2607.25045, 2026.

Ji-Hoon Jeong, Jeong-Hyun Cho, Young-Eun Lee, Seo-Hyun Lee, Gi-Hwan Shin, Young-Seok Kweon, José del R Millán, Klaus-Robert Müller, and Seong-Whan Lee. 2020 international brain–computer interface competition: A review. Frontiers in human neuroscience, 16:898300, 2022.

Muyun Jiang, Shuailei Zhang, Zhenjie Yang, Mengjun Wu, Weibang Jiang, Zhiwei Guo, Wei Zhang, Rui Liu, Shangen Zhang, Yong Li, et al. Leaf: Language-eeg aligned foundation model for brain-computer interfaces. arXiv preprint arXiv:2509.24302, 2025.

Muyun Jiang, Wei Zhang, Yi Ding, Kok Ann Colin Teo, LaiGuan Fong, Shuailei Zhang, Zhiwei Guo, Chenyu Liu, Raghavan Bhuvanakantham, Wei Khang Jeremy Sim, et al. Decoding covert speech from eeg by functional areas spatio-temporal transformer. IEEE Journal ofBiomedical and Health Informatics, 2026.

Wei-Bang Jiang, Liming Zhao, and Bao-Liang Lu. Large brain model for learning generic representations with tremendous eeg data in bci. In International Conference on Learning Representations, volume 2024, pp. 16405–16426, 2024.

Bob Kemp, Aeilko H Zwinderman, Bert Tuk, Hilbert AC Kamphuisen, and Josefien JL Oberye. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the eeg. IEEE Transactions on Biomedical Engineering, 47(9):1185–1194, 2000.

Sirvan Khalighi, Teresa Sousa, José Moutinho Santos, and Urbano Nunes. Isruc-sleep: A comprehensive public dataset for sleep researchers. Computer methods and programs in biomedicine, 124:180–192, 2016.

Vernon J Lawhern, Amelia J Solon, Nicholas R Waytowich, Stephen M Gordon, Chou P Hung, and Brent J Lance. Eegnet: a compact convolutional neural network for eeg-based brain–computer interfaces. Journal ofneural engineering, 15(5):056013, 2018.

Min-Ho Lee, O-Yeon Kwon, Yong-Jeong Kim, Hong-Kyung Kim, Young-Eun Lee, John Williamson, Siamac Fazli, and Seong-Whan Lee. Eeg dataset and openbmi toolbox for three bci paradigms: An investigation into bci illiteracy. GigaScience, 8(5):giz002, 2019.

Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh, and Ameet Talwalkar. Hyperband: A novel bandit-based approach to hyperparameter optimization. Journal ofmachine learning research, 18(185): 1–52, 2018.

Wei Liu, Jie-Lin Qiu, Wei-Long Zheng, and Bao-Liang Lu. Comparing recognition performance and robustness of multimodal deep learning models for multimodal emotion recognition. IEEE Transactions on Cognitive and Developmental Systems, (2):715–729, 2021.

Wuche Liu, Yiran Qiao, Linlin Hou, Rui Yang, Shusen Pu, Song Wang, and Jing Ma. Ns-copilot: An llm-driven agent system for autonomous neuroscience analysis. arXiv preprint arXiv:2609.01971, 2026.

Jun Ma, Banghua Yang, Wenzheng Qiu, Yunzhe Li, Shouwei Gao, and Xinxing Xia. A large eeg dataset for studying cross-session variability in motor imagery brain-computer interface. Scientific Data, 9(1):531, 2022.

Gerwin Schalk, Dennis J McFarland, Thilo Hinterberger, Niels Birbaumer, and Jonathan R Wolpaw. Bci2000: a general-purpose brain-computer interface (bci) system. IEEE Transactions on biomedical engineering, 51(6):1034–1043, 2004.

Robin Tibor Schirrmeister, Jost Tobias Springenberg, Lukas Dominique Josef Fiederer, Martin Glasstetter, Katharina Eggensperger, Michael Tangermann, Frank Hutter, Wolfram Burgard, and Tonio Ball. Deep learning with convolutional neural networks for eeg decoding and visualization. Human brain mapping, 38 (11):5391–5420, 2017.

Jaeyoung Shin, Alexander von Lühmann, Benjamin Blankertz, Do-Won Kim, Jichai Jeong, Han-Jeong Hwang, and Klaus-Robert Müller. Open access dataset for eeg+ nirs single-trial classification. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 25(10):1735–1745, 2016.

Yonghao Song, Qingqing Zheng, Bingchuan Liu, and Xiaorong Gao. Eeg conformer: Convolutional transformer for eeg decoding and visualization. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 31:710–719, 2022.

Akara Supratak, Hao Dong, Chao Wu, and Yike Guo. Deepsleepnet: A model for automatic sleep stage scoring based on raw single-channel eeg. IEEE transactions on neural systems and rehabilitation engineering, 25 (11):1998–2008, 2017.

Michael Tangermann, Klaus-Robert Müller, Ad Aertsen, Niels Birbaumer, Christoph Braun, Clemens Brunner, Robert Leeb, Carsten Mehring, Kai J Miller, Gernot R Müller-Putz, et al. Review of the bci competition iv. Frontiers in neuroscience, 6:55, 2012.

Guoan Wang, Shihao Yang, and Feng Liu. Neuroweaver: An autonomous evolutionary agent for exploring the programmatic space of eeg analysis pipelines. arXiv preprint arXiv:2602.13473, 2026.

Jiquan Wang, Sha Zhao, Zhiling Luo, Yangxuan Zhou, Haiteng Jiang, Shijian Li, Tao Li, and Gang Pan. Cbramod: A criss-cross brain foundation model for eeg decoding. In International conference on learning representations, volume 2025, pp. 75310–75346, 2025.

Fang Weilin, Yu Xue, Yuan Lilian, Mohammad Kamrul Hasan, and Khursheed Aurangzeb. Large language model assisted evolutionary neural architecture search with population knowledge base enhancement. Information Sciences, pp. 123110, 2026.

Zekang Yang, Wang Zeng, Sheng Jin, Chen Qian, Ping Luo, and Wentao Liu. Nader: Neural architecture design via multi-agent collaboration. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4452–4461. IEEE, 2025.

Sha Zhao, Mingyi Peng, Haiteng Jiang, Tao Li, and Shijian Li. Eeg agent: A unified framework for automated eeg analysis using large language models. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pp. 18063–18071, 2026.

Mingkai Zheng, Xiu Su, Shan You, Fei Wang, Chen Qian, Chang Xu, and Samuel Albanie. Can gpt-4 perform neural architecture search? arXiv preprint arXiv:2304.10970, 2023.

Wei-Long Zheng, Wei Liu, Yifei Lu, Bao-Liang Lu, and Andrzej Cichocki. Emotionmeter: A multimodal framework for recognizing human emotions. IEEE transactions on cybernetics, 49(3):1110–1122, 2018.

## A Supplementary Material

## Contents

A.1 Related Work 14   
A.1.1 EEG Representation Learning and Architecture Design 14   
A.1.2 LLM-Guided Architecture Search and Early Performance Prediction . 15   
A.2 Evaluation details and supplementary comparisons 15   
A.3 Round-by-round architecture details 16   
A.4 Baseline training and evaluation 17   
A.5 Data preparation and split construction 17   
A.5.1 Dataset descriptions 18   
A.5.2 Task-specific split assignments 19   
A.5.3 Signal representation and label handling 20   
A.5.4 Dataset split rules and sizes 20   
A.6 Search implementation and execution details 20   
A.7 Candidate-training settings 22   
A.8 Training precision: computational cost and predictive performance 22   
A.9 Winner Architecture . 23   
A.9.1 Claude Sonnet 5 24   
A.9.2 DeepSeek V4.1 Flash 24   
A.9.3 Gemini 3.8 Flash 25   
A.9.4 Qwen3.8 Flash 26   
A.9.5 GPT-5.6 Sol . 27   
A.9.6 Claude Opus 5.5 28   
A.10 Complete prompts and structured interfaces 28   
A.10.1 PGAD: complete first-round prompt 28   
A.10.2 PGAD: subsequent-round prompt 30   
A.10.3 PEEK: complete forecasting prompt template 31

## A.1 Related Work

## A.1.1 EEG Representation Learning and Architecture Design

EEG architectures encode assumptions about temporal structure and spatial organization. EEGNet uses depthwise and separable convolutions to construct a compact architecture evaluated across several BCI paradigms (Lawhern et al., 2018). Task-specific designs introduce further structure: TSception combines temporal filters at multiple scales with spatial filters that capture hemispheric asymmetry for emotion recognition (Ding et al., 2022), while DeepSleepNet combines convolutional feature extraction with recurrent modeling of sleep-stage dependencies (Supratak et al., 2017). Foundation models study transferable EEG representations through pretraining. LaBraM learns discrete neural tokens through neural-spectrum prediction and pretrains a Transformer using masked token prediction (Jiang et al., 2024). CBraMod uses criss-cross attention to model spatial and temporal dependencies in heterogeneous EEG recordings (Wang et al., 2025). LEAF incorporates task instructions and aligns EEG representations with language semantics across tasks and label spaces (Jiang et al., 2025). AutoBCI selects architecture code using combined validation scores from emotion recognition, motor imagery, and sleep staging, with independent training and separate parameters for each task family.

Architecture search also has direct precedents in EEG decoding. CTNAS-EEG introduces a search space compatible with multiple EEG tasks and a constrained search procedure, and examines architectural variation across tasks and subjects (Duan et al., 2023). PGAD uses executable model proposals and an accumulated pool ranked by an equal-weight aggregate of task scores. Task-specific learning curves and validation results accompany each selected parent, providing feedback for subsequent code-level refinements.

## A.1.2 LLM-Guided Architecture Search and Early Performance Prediction

Language models can serve as proposal operators in architecture search. EvoPrompting uses code-generating language models for evolutionary mutation and crossover, together with evolutionary prompt construction and soft prompt tuning (Chen et al., 2023). GENIUS treats GPT-4 as a black-box optimizer that proposes and iteratively refines architectures (Zheng et al., 2023). NADER coordinates specialized agents around a graph representation of architectures and uses reflection on feedback and prior experience to guide modifications (Yang et al., 2025). PGAD combines parent refinements and fresh proposals with a shared selection objective across independently trained EEG task models.

Reducing the cost of candidate evaluation is a related problem. Domhan et al. (2015) extrapolate partial learning curves to terminate unpromising runs. Baker et al. (2017) combine architecture features, hyperparameters, and partial validation trajectories to predict performance during architecture search. Architecture-aware learning-curve extrapolation also models network structure through a graph ordinary diferential equation (Ding et al., 2025). Resource-allocation methods such as Hyperband use successive halving to distribute training budgets (Li et al., 2018); BOHB combines this allocation strategy with model-based configuration selection (Falkner et al., 2018).

## A.2 Evaluation details and supplementary comparisons

Forecasting inputs and aggregation. In the primary analysis, each generating LLM forecasts its own candidates in isolated sessions with tools disabled. Inputs contain the training protocol, parameter counts, training losses, and validation prefixes for all three tasks. Both the code-and-curve and curve-only conditions use the same anonymous candidate order and early evidence; only the former includes architecture code, with comments and docstrings removed. Candidate names, dataset names, and parent identities are omitted. The target for each task is the maximum, within 40 epochs, of its equal-dataset mean validation bAcc. Combined scores average these task maxima equally, allowing diferent checkpoint epochs across tasks.

The cross-model comparison in Table 5 reuses the frozen ten-epoch code-and-curve prompts, anonymous ordering, and self-forecasts from the Opus and Sonnet searches. All 36 additional calls succeed (three forecasters across 12 rounds), with one forecast per forecaster and round. Generator identities are omitted from the prompts, and later validation epochs and test results remain unavailable to the forecasters.

Nominal screening budget. Observing all eight candidates for ten epochs and continuing two to epoch 40 uses $8 \times 1 0 + 2 \times ( 4 0 - 1 0 ) = 1 4 0$ candidate-epochs per task, compared with $8 \times 4 0 = 3 2 0$ under full training. The resulting 56.25% reduction excludes forecasting overhead and failed training attempts.

Retention–cost accounting. The fixed cohort in Table 6 uses all 36 successfully paired rounds at ten epochs, with 269 candidates and three independently trained tasks per candidate. For a round with $n _ { r }$ candidates, keeping $k _ { r } = \operatorname* { m i n } ( k , n _ { r } )$ gives $3 [ 1 0 n _ { r } + 3 0 k _ { r } ]$ task-epochs. Full training uses 32,280 task-epochs. The actual cohort has fewer than eight completed candidates in some rounds, giving 54.9% savings at $k = 2$ compared with 56.25% for an idealized eight-candidate round. Predicted ties retain the original anonymous candidate order; any tied true maximum counts as a winner. These estimates evaluate screening on the recorded candidate batches. How screening changes parent selection, subsequent proposals, and final search performance remains to be evaluated.

Recorded 40-epoch task durations include validation. Estimated GPU job-hours sum these durations for continued candidates and one quarter of each duration for candidates stopped at ten epochs, assuming constant average epoch time. These are accumulated job durations, not physical GPU occupancy or elapsed search time; concurrent jobs can share a GPU. Token counts sum saved generation and accepted forecasting calls, including input cache reads and writes once. The common generation budget is 0.281M input and 0.501M output tokens; PEEK adds 1.167M input and 0.120M output tokens.

## A.3 Round-by-round architecture details

Supplementary Figure 6 identifies the incumbent architecture after each round of the six searches shown in Figure 3.

![](images/5bb7a86b407025ba3d47ca4ae3e3c5b14f90e1f8c9507674dc5e58820040f0c0.jpg)  
C Claude Sonnet 5

![](images/a0ea289abfe9017001184d83e5001c0950b0a6f68486f271f2ed70b73d8abc35.jpg)  
D DeepSeek V4.1 Flash

![](images/6034ed29bb1d77a27c6db2b05cc65b48f0be69cdd49dad508eea8d1df92f6547.jpg)

![](images/77f296d981b0f9b1e24418bd09529b0f9dcede3bc588b316f450966cfe3fca02.jpg)

E Gemini 3.8 Flash  
![](images/feb59bdc62d687a8ef7ac58c0620eb289fde3928484df52a093c76f1e49b4ef0.jpg)

F Qwen3.8 Flash  
![](images/b81e8203c79177e19229200f770143bbf94090516704f7da42ca13a2c6a79e44.jpg)  
Figure 6. Architecture discovery with round incumbents. Dots show individual evaluated candidates; dashed and solid lines show round medians and cumulative best bAcc. Each inset names the incumbent after rounds R1–R6; bold text marks the final architecture.

## A.4 Baseline training and evaluation

To interpret the held-out comparison, we describe how each baseline is initialized, adapted to the task inputs, and optimized. Each model is trained independently on the pooled datasets of each task family, using the same harmonized labels, normalization, fixed partitions, and sleep subsets as AutoBCI. Each epoch visits all training examples in shufled order, including the final incomplete batch. Validation bAcc is averaged equally across datasets after every epoch; a checkpoint is replaced only when this score increases. The selected checkpoint is evaluated on the held-out test partitions, and the three task scores receive equal weight in the combined results.

Optimization and training budgets. EEGNet, DeepConvNet, TSception, EEG-Conformer, AttnSleep, and FAST are initialized from scratch and trained for 100 epochs with batch size 512 and initial learning rate 10<sup>−3</sup>. LaBraM, CBraMod, LEAF, and REVE use a configured budget of 40 epochs, batch size 256 without gradient accumulation, and initial learning rate $1 0 ^ { - 4 }$ . All main-comparison runs use three random seed averaged, fused AdamW with weight decay $1 0 ^ { - 4 }$ , coeficients (0.9, 0.999), numerical epsilon 10<sup>−8</sup>, and global gradient-norm clipping at 5.0. Cosine annealing decreases the learning rate toward zero over the configured epoch budget. Training uses unweighted cross-entropy computed from FP32 logits, with model parameters and AdamW moments in BF16.

Pretrained initialization and task heads. LaBraM-base loads the student encoder from labram-base.pth, with a newly initialized mean-pooling normalization and classification head; pretraining-only prediction components are omitted. CBraMod loads pretrained\_weights.pth, replaces its reconstruction projection with an identity mapping, and adds a linear classifier to the mean-pooled representations. REVE-base loads the oficial brain-bzh/reve-base checkpoint and adds a linear classifier after averaging channel and time representations; electrode coordinates come from its oficial position bank. LEAF uses leaf-v1.0-instruct-mpnet-base.ckpt for MI and emotion, with its EEG tower and a new flattened tower-token classifier trained jointly. The QFormer is unused by this classifier. For these pretrained conditions, the retained encoder and task head are both fine-tuned. LEAF on sleep is initialized from scratch, with its channel count and temporal capacity adapted to the native sleep input.

Input adaptations. Conventional baselines receive the native task tensors: 65 × 800 for MI and emotion and 8 × 3000 for sleep. The foundation baselines also retain the 65 × 800 MI and emotion inputs at 200 Hz. For sleep, LaBraM, CBraMod, and REVE linearly resample each complete 30-second example from 100 to 200 Hz and divide it into three non-overlapping ten-second windows. Window logits are averaged before computing the example-level loss and metrics, preserving the original example and split identities. LaBraM uses named electrode indices and 200-sample patches. LEAF sleep instead consumes the complete native 8 × 3000 tensor directly. CBraMod computes its FFT in FP32 before casting spectral features back to BF16; REVE similarly retains FP32 Fourier-coordinate calculations and normalization reductions while adapting linear-layer inputs to the model dtype.

## A.5 Data preparation and split construction

Task-wise training and evaluation. For each candidate architecture, we pool only the training splits within each task family and train one model from scratch for 40 epochs. The three task models share architecture code but learn separate weights and output heads. The class counts in Table 1 refer to the union of semantic labels within each family: seven for emotion, seven for MI, and five for sleep. Validation and test splits remain separate from training, and each dataset is evaluated individually. Architecture selection uses validation scores averaged equally across datasets within each task, then equally across the three tasks. Test splits are used only for final evaluation.

## A.5.1 Dataset descriptions

The pooled task families combine datasets with diferent acquisition protocols and participant populations. The following list describes each source and the version used in our experiments; emotion and MI follow the LEAF dataset preparation documentation (Jiang et al., 2025). Split assignments and retained example counts are given below in Tables 7 and 8.

Emotion recognition. The three SEED datasets contain EEG recorded during emotion-eliciting video trials, with diferent emotion categories and session structures.

• SEED (Duan et al., 2013) contains 62-channel recordings from 15 participants across three sessions, with 15 trials per session and three labels: Negative, Neutral, and Positive. The LEAF version concatenates corresponding trials across sessions before extracting four-second windows.

• SEED-IV (Zheng et al., 2018) contains 62-channel recordings from 15 participants across three sessions, each with 24 video trials. Four-second windows retain four emotion labels: Neutral, Sad, Fear, and Happy.

• SEED-V (Liu et al., 2021) contains recordings from 16 participants across three sessions, each with 15 video trials. After removing ocular and mastoid channels, 62 EEG channels are segmented into four-second windows with Disgust, Fear, Sad, Neutral, and Happy labels.

Motor imagery. The motor-decoding collection covers hand, foot, tongue, and grasp-related tasks. The descriptions below distinguish the source recordings from the classes retained by the LEAF preparation.

• BCI Competition IV-2a (Tangermann et al., 2012) provides 22-channel EEG from nine participants performing left-hand, right-hand, foot, and tongue imagery. Both acquisition-session files are included, and the preparation retains a four-second interval from each trial.

• BCI Upper Limb (Jeong et al., 2022) contains recordings from 15 participants imagining three grasp types: cylindrical, spherical, and lumbrical. The published training and validation recordings are combined before applying the subject partitions used here; each example covers the final four-second imagery stage.

• Cho2017 (Cho et al., 2017) is a left- versus right-hand imagery dataset. The LEAF export contains 49 participants after excluding subjects 32, 46, and 49; its preprocessing appends 200 samples by edge padding before constructing the common input representation.

• HighGamma (Schirrmeister et al., 2017) contributes recordings from 14 participants. The LEAF preparation initially retains left-hand, right-hand, and feet classes, and the experimental loader further selects the two hand classes for binary decoding.

• OpenBMI (Lee et al., 2019) contributes left- and right-hand imagery recordings from 54 participants across two acquisition sessions. The LEAF version combines the training and test recordings from both sessions for each participant before applying the experiment’s subject partitions.

• PhysioNet (Schalk et al., 2004) contains 64-channel recordings from 109 participants performing real and imagined movements. The LEAF preparation processes runs R03–R14 and retains the two non-rest event codes; their source meanings depend on the run (left/right fist or both fists/both feet), while the stored task catalog names the two classes Left and Right.

• ShanghaiU (Ma et al., 2022) contains left- and right-hand imagery recordings from 25 participants over five sessions. All five sessions are combined per participant, and the two reference channels A1 and A2 are removed before mapping to the common montage.

• Shin2017A (Shin et al., 2016) contributes the EEG component of an EEG–NIRS motor-imagery dataset with left- and right-hand labels. The LEAF version contains 28 available participants (subject S05 is unavailable), uses three imagery blocks per participant, and removes the ocular channels before extracting four-second epochs.

Sleep staging. The sleep datasets provide overnight polysomnography with expert stage annotations. We use their EEG signals to classify 30-second epochs as Wake, N1, N2, N3, or REM.

• Sleep-EDF (Kemp et al., 2000) uses the expanded database’s Sleep Cassette cohort: 153 recordings from 78 participants in a study of age-related sleep changes. The original EEG derivations are Fpz–Cz and Pz–Oz at 100 Hz, with Rechtschafen–Kales stage annotations. Our preparation merges stages 3 and 4 into N3 and retains up to 30 minutes of wakefulness before and after sleep.<sup>1</sup>

• HMC (Alvarez-Estevez & Rijsman, 2021) contains 151 overnight recordings from patients referred for sleep evaluation at Haaglanden Medisch Centrum in the Netherlands. Four EEG derivations (F4/M1, C4/M1, O2/M1, and C3/M2) were acquired at 256 Hz, and sleep technicians scored the recordings using AASM guidelines. Our preparation uses the original EDF release and resamples EEG to 100 Hz.<sup>2</sup>

• ISRUC (Khalighi et al., 2016) uses subgroup I, comprising one overnight recording per subject from 100 adults with sleep disorders. The dataset provides annotations from two experts; our preparation uses scorer 1, excludes subject 55 because of conflicting annotations, and retains 99 subjects. The available EEG derivations are mapped to the shared eight-channel sleep representation.<sup>3</sup>

## A.5.2 Task-specific split assignments

Emotion recognition. We use the four-second HDF5 versions of SEED, SEED-IV, and SEED-V, preserving their predefined training, validation, and test arrays. The preparation scripts assign stimulus trials within subjects to these splits: SEED uses trials 1–9, 10–12, and 13–15; SEED-IV uses trials 1–16, 17–20, and 21–24 in each session; SEED-V uses trials 1–5, 6–10, and 11–15 in each session. AutoBCI reads the stored assignments without drawing a new validation subset.

Motor imagery. The MI files store examples by subject. In the HDF5 group iteration order, the first 7, 11, 40, 10, 42, 80, 20, and 22 subjects form the development partitions of IV-2a, Upper Limb, Cho2017, HighGamma, OpenBMI, PhysioNet, ShanghaiU, and Shin2017A, respectively. The remaining 2, 4, 9, 4, 12, 29, 5, and 6 subjects form their test partitions. Within each development partition, examples are concatenated in HDF5 subject order after label filtering. Validation takes zero-based example indices 0, 5, 10, . . . (approximately 20%), and training takes the remaining examples. Training and validation therefore share subjects, while test subjects are disjoint from both. HighGamma retains only local labels 0 and 1, corresponding to the catalog’s left- and right-hand classes. The PhysioNet-MI recordings were collected with the BCI2000 system (Schalk et al., 2004).

Sleep staging. Sleep-EDF, HMC, and ISRUC use five classes: Wake, N1, N2, N3, and REM. We preserve their stored subject-disjoint split manifests, with 49/13/16 subjects for Sleep-EDF, 96/24/31 for HMC, and 63/16/20 for ISRUC in training/validation/test. Sleep-EDF uses the cassette cohort; ISRUC uses scorer 1 and excludes subject 55. To define the search workload, the stored files retain 25%, 40%, and 50% of the examples in Sleep-EDF, HMC, and ISRUC, respectively. Subsampling is uniform without replacement within each split, and retains the floor of the original split size multiplied by its retention fraction. The retained subsets are fixed throughout architecture search and final evaluation.

## A.5.3 Signal representation and label handling

Emotion and MI examples use a common 65-channel representation with 800 samples at 200 Hz. The LEAF preparation scripts map source electrodes to the LEAF-ch65 template and apply percentile clipping and robust scaling before export. Emotion preparation uses a 0.1–70 Hz bandpass and a 50 Hz notch filter. MI preparation uses a 0.3–40 Hz bandpass, except Upper Limb, which uses 0.1–70 Hz. The search consumes the exported arrays; signal filtering and electrode mapping are fixed before candidate training.

Sleep examples preserve complete 30-second epochs at 100 Hz. The ordered channel slots are Fpz, F3, F4, C3, C4, Pz, O1, and O2. The stored preprocessing uses a 0.5–40 Hz bandpass, rejects epochs with excessive absolute amplitude above 2,000 µV before filtering, and applies per-recording clipping at the 0.1th and 99.9th percentiles followed by median/IQR scaling. Missing channel slots are interpolated using an MNE spherical-spline mapping, retaining the original references of measured derivations.

For every task, the training loader additionally standardizes each example and channel along time:

$$
\widetilde { x } _ { c , t } = \frac { x _ { c , t } - \mu _ { c } } { \operatorname* { m a x } ( \sigma _ { c } , 1 0 ^ { - 6 } ) } ,\tag{6}
$$

where $\mu _ { c }$ and $\sigma _ { c }$ are the temporal mean and population standard deviation of that channel in the example. Normalized inputs are cached in CPU memory as BF16 tensors. This normalization uses statistics from the example itself. The configured files already match their task’s temporal input length, so the loader passes all 800 or 3,000 samples to the model.

Each task uses one output head spanning the union of its catalog labels. Labels with identical names within a task share an output class. This produces seven emotion outputs (Positive, Neutral, Negative, Sad, Fear, Happy, and Disgust), seven MI outputs (Left, Right, Foot, Tongue, Cylin, Sphe, and Lumbrical), and five sleep outputs. During validation and testing, logits for classes absent from the example’s source dataset are set to negative infinity before prediction. Dataset identity is used for this evaluation mask and metric aggregation; the model receives only the EEG tensor.

## A.5.4 Dataset split rules and sizes

The split unit determines which examples and subjects are held out during evaluation. Emotion uses predefined stimulus-trial assignments within subjects, MI holds out test subjects and divides development examples into training and validation, and sleep separates subjects across all three partitions. The dataset-specific assignments are summarized in Table 7; the resulting example counts after label filtering and sleep subsampling appear in Table 8.

## A.6 Search implementation and execution details

Candidates use PyTorch and expose build\_model(n\_channels, n\_samples, n\_classes). The search prompt permits PyTorch and Python’s math module, excludes recurrent layers and custom temporal recurrence, and requires all trainable parameters to be created before the forward pass. Models are trained from scratch with the common training procedure below.

Complete prompt templates and structured interfaces appear in Supplementary A.10. Each generation call produces the entire batch of eight proposals as a structured JSON response. The harness runs in a temporary working directory with tool use disabled. The supplied prompt contains the architectural constraints, training settings, parameter budget, and class categories. Later rounds also receive the two parents’ source code, task-level validation curves, best epochs, parameter counts, and per-dataset validation scores at the selected epochs. Per-dataset results use anonymous identifiers in the prompt. Generation calls receive the measured parent evidence explicitly rather than retaining a conversational session across rounds.

The Claude Code and Gemini CLI adapters configure a maximum output of 65,536 tokens. The Codex adapter enforces the response schema through its structured output interface and leaves the output-token limit to the harness. Temperature and sampling seed are not explicitly set by the adapters. Recorded generation timeouts are 1,200 seconds in round 1 for all six searches. Rounds 2–6 use 1,200 seconds for DeepSeek and 600 seconds for the other five models. Each call records the requested model, provider, harness command, duration, response, and available model-identity evidence.

Table 7. Dataset split construction. Emotion trial numbers are one-based and apply within each subject and session; subjects occur in all three splits. MI subject ranges are zero-based positions in HDF5 group iteration order, not original subject IDs. MI training and validation use approximately 80% and 20% of examples from the same development subjects, respectively; test subjects are held out. Sleep entries give numbers of mutually disjoint subjects.
<table><tr><td>Task</td><td>Dataset</td><td>Train</td><td>Validation</td><td>Test</td></tr><tr><td>Emotion</td><td>SEED</td><td>Trials 1–9</td><td>Trials 10–12</td><td>Trials 13–15</td></tr><tr><td></td><td>SEED-IV SEED-V</td><td>Trials 1–16 Trials 1–5</td><td>Trials 17–20 Trials 6–10</td><td>Trials 21–24 Trials 11–15</td></tr><tr><td>MI</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>BCI Competition IV-2a</td><td>Subjects 0–6</td><td>Subjects 0–6</td><td>Subjects 7–8</td></tr><tr><td></td><td>BCI Upper Limb</td><td>Subjects 0-10</td><td>Subjects 0–10</td><td>Subjects 11–14</td></tr><tr><td></td><td>Cho2017</td><td>Subjects 0–39</td><td>Subjects 0–39</td><td>Subjects 40–48</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>HighGamma</td><td>Subjects 0–9</td><td>Subjects 0–9</td><td>Subjects 10-13</td></tr><tr><td></td><td>OpenBMI</td><td>Subjects 0–41</td><td>Subjects 0–41</td><td>Subjects 42–53</td></tr><tr><td></td><td>PhysioNet</td><td>Subjects 0–79</td><td>Subjects 0–79</td><td>Subjects 80–108</td></tr><tr><td></td><td>ShanghaiU</td><td>Subjects 0–19</td><td>Subjects 0–19</td><td>Subjects 20–24</td></tr><tr><td></td><td>Shin2017A</td><td>Subjects 0–21</td><td>Subjects 0–21</td><td>Subjects 22–27</td></tr><tr><td>Sleep</td><td>Sleep-EDF</td><td>49 subjects</td><td>13 subjects</td><td>16 subjects</td></tr><tr><td></td><td>HMC</td><td>96 subjects</td><td>24 subjects</td><td>31 subjects</td></tr><tr><td></td><td>ISRUC</td><td>63 subjects</td><td>16 subjects</td><td>20 subjects</td></tr></table>

Table 8. Dataset composition and split sizes. Counts refer to EEG examples, including 30-second epochs for sleep. Classes are the retained local classes in each dataset. Emotion and MI inputs have shape 65 × 800; sleep inputs have shape 8 × 3000.
<table><tr><td>Task</td><td>Dataset</td><td>Classes</td><td>Train</td><td>Validation</td><td>Test</td></tr><tr><td>Emotion</td><td>SEED (3 classes)</td><td>3</td><td>22,545</td><td>7,905</td><td>7,620</td></tr><tr><td></td><td>SEED-IV</td><td>4</td><td>26,025</td><td>5,340</td><td>6,210</td></tr><tr><td></td><td>SEED-V</td><td>5</td><td>8,512</td><td>10,672</td><td>9,984</td></tr><tr><td>MI</td><td>BCI Competition IV-2a</td><td>4</td><td>3,148</td><td>788</td><td>1,152</td></tr><tr><td></td><td>BCI Upper Limb</td><td>3</td><td>2,640</td><td>660</td><td>1,200</td></tr><tr><td></td><td>Cho2017</td><td>2</td><td>6,464</td><td>1,616</td><td>1,800</td></tr><tr><td></td><td>HighGamma</td><td>2</td><td>3,761</td><td>941</td><td>2,040</td></tr><tr><td></td><td>OpenBMI</td><td>2</td><td>13,440</td><td>3,360</td><td>4,800</td></tr><tr><td></td><td>PhysioNet</td><td>2</td><td>11,447</td><td>2,862</td><td>5,262</td></tr><tr><td></td><td>ShanghaiU</td><td>2</td><td>7,712</td><td>1,929</td><td>2,347</td></tr><tr><td></td><td>Shin2017A</td><td>2</td><td>1,056</td><td>264</td><td>360</td></tr><tr><td>Sleep</td><td>Sleep-EDF</td><td>5</td><td>30,062</td><td>9,237</td><td>9,567</td></tr><tr><td></td><td>HMC</td><td>5</td><td>34,764</td><td>8,688</td><td>11,422</td></tr><tr><td></td><td>ISRUC</td><td>5</td><td>28,181</td><td>7,271</td><td>8,914</td></tr></table>

Before training, static checks validate the model interface and allowed imports. Numerical checks verify output shape, finite logits, consistent repeated evaluation, and fixed parameter registration for batch sizes one and two. Two synthetic optimization steps then check execution at the configured training batch size and precision. Actual training constructs a fresh model after these checks. Candidate implementation failures are recorded with their task and failure stage, and the generated source is preserved. Only candidates that complete all three tasks enter parent and champion selection.

## A.7 Candidate-training settings

The shared optimization settings in Table 9 specify how candidates are trained under the protocol in Section 3.3.

An epoch visits every training example in the task’s pooled datasets once, in a new shufled order, including the final incomplete batch. Sampling is uniform over examples, so a dataset’s contribution to the training loss scales with its number of examples. Validation is performed after each epoch and weights datasets equally according to Equation (1). The loss is unweighted cross-entropy; the training loop applies neither class reweighting nor data augmentation.

The seed initializes Python, NumPy, PyTorch, CUDA, and a separate generator for training-example shufling. The best checkpoint is updated only when validation bAcc strictly increases, retaining the earliest epoch in a tie. The configured early observation budget for PEEK is ten epochs.

Training uses PyTorch 2.10.0+cu128 and NVIDIA RTX PRO 6000 Blackwell GPUs. Each task runs as a separate process on one GPU, and the coordinator schedules independent jobs according to GPU availability. Full BF16 training casts model parameters before optimizer construction, so AdamW moment tensors also use BF16. CUDA settings enable cuDNN benchmarking with a benchmark limit of ten, TF32 for cuDNN, medium FP32 matrix-multiplication precision, and BF16 reduced precision reductions. Deterministic cuDNN execution is disabled. The recorded seed therefore specifies stochastic initialization and sampling without implying bitwise identical GPU execution. A controlled comparison of FP32, BF16 mixed, and full BF16 training appears in Supplementary A.8.

Table 9. Common candidate-training settings. The same settings apply to emotion, MI, and sleep in all six completed searches.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Full training budget Batch size</td><td>40 epochs per task 512</td></tr><tr><td>Optimizer Initial learning rate Weight decay</td><td>Fused AdamW  $1 0 ^ { - 3 }$   $1 0 ^ { - 4 }$ </td></tr><tr><td>Adam coefficients  $( \beta _ { 1 } , \beta _ { 2 } )$  Numerical epsilon € Learning-rate schedule</td><td>(0.9, 0.999) 10-8</td></tr><tr><td>Gradient clipping</td><td>Cosine annealing;  $T _ { \mathrm { m a x } } = 4 0$  , minimum 0 Global norm 5.0</td></tr><tr><td>Parameters, gradients, optimizer moments</td><td></td></tr><tr><td></td><td>BF16</td></tr><tr><td>Cross-entropy computation</td><td>FP32 logits and loss reduction</td></tr><tr><td>CPU threads per training process</td><td>2</td></tr><tr><td>Data cache</td><td>BF16 in CPU RAM</td></tr><tr><td>Preload batch size</td><td></td></tr><tr><td>Data-loader worker processes</td><td>512</td></tr><tr><td></td><td>0</td></tr></table>

## A.8 Training precision: computational cost and predictive performance

We compare FP32, BF16 mixed precision, and full BF16 on BCI Competition IV-2a using the repository implementations of EEGNet (Lawhern et al., 2018), DeepConvNet (Schirrmeister et al., 2017), and the fixed Opus-discovered architecture. All models are trained from scratch on this single dataset. The existing split assigns 3,148 training and 788 validation examples to subjects A01–A07, and reserves 1,152 test examples from A08–A09. Inputs retain 65 channels and 800 time samples. For each architecture, the three modes share a common FP32 initialization before precision-specific casting, and the same minibatch order for seeds 0, 1, and 2. We train for 40 epochs with AdamW, learning rate $1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 }$ , cosine decay, and gradient clipping at norm 5. The batch size is 128. The earliest checkpoint with maximal validation bAcc is evaluated on the test set after all training runs finish.

FP32 uses FP32 parameters, gradients, and optimizer moments with TF32 disabled. BF16 mixed uses BF16 autocast while retaining FP32 parameters, gradients, and optimizer moments. Full BF16 casts model state and inputs to BF16 and stores gradients and AdamW moments in BF16, without FP32 master weights. All modes compute cross-entropy in FP32; explicit FP32 reductions in the generated architecture are preserved. Inputs are normalized and cached in FP32 for every condition, isolating training precision from cache quantization. Unlike the main search, this experiment uses batch size 128 and disables TF32 throughout.

Runs execute sequentially in fresh processes on one NVIDIA RTX PRO 6000 Blackwell GPU with PyTorch 2.10.0+cu128. Precision order rotates across seeds. Warmup exercises training and validation batch shapes, after which weights, optimizer state, and random-number generators are reset. CUDA-synchronized training time includes minibatch transfers and optimizer updates, excluding validation, preprocessing, warmup, profiling, and checkpoint writes. Memory is peak PyTorch allocation during training and validation. FLOPs count convolution and matrix-multiplication operations in forward and backward passes, with one multiply-add counted as two; normalization, elementwise operations, loss, and optimizer updates are excluded. These counts are unchanged by precision, whereas latency and memory depend on the numerical representation and kernel implementation.

Table 10. Training precision on BCI Competition IV-2a. Means and standard deviations over three seeds. GFLOPs are counted per training example (forward and backward); time is per training epoch; memory is mean peak allocation. Test bAcc uses validation-selected checkpoints.
<table><tr><td>Model</td><td>Precision</td><td>GFLOPs</td><td>Time (s)</td><td>MiB Test bAcc (%)</td></tr><tr><td rowspan="3">EEGNet</td><td>FP32</td><td>0.126</td><td> $0 . 6 5 0 \pm 0 . 0 0 1$  644</td><td> $5 5 . 4 7 \pm 0 . 6 9$ </td></tr><tr><td>BF16 mixed</td><td>0.126</td><td> $0 . 8 1 9 \pm 0 . 0 0 9$  371</td><td> $5 4 . 4 8 \pm 0 . 9 0$ </td></tr><tr><td>BF16 full</td><td>0.126  $0 . 8 1 0 \pm 0 . 0 0 6$ </td><td>345</td><td> $5 3 . 7 9 \pm 1 . 1 1$ </td></tr><tr><td rowspan="3">DeepConvNet</td><td>FP32</td><td>0.328</td><td> $0 . 3 8 0 \pm 0 . 0 0 8$  1336</td><td> $5 8 . 9 1 \pm 3 . 8 2$ </td></tr><tr><td>BF16 mixed</td><td>0.328</td><td> $0 . 4 2 7 \pm 0 . 0 0 1$  1099</td><td> $5 7 . 9 6 \pm 3 . 0 4$ </td></tr><tr><td>BF16 full</td><td>0.328  $0 . 4 2 3 \pm 0 . 0 0 1$ </td><td>1083</td><td> $5 5 . 9 0 \pm 2 . 9 1$ </td></tr><tr><td rowspan="3">AutoBCI (Opus)</td><td>FP32</td><td>0.285</td><td> $0 . 3 0 0 \pm 0 . 0 0 1$  351</td><td> $5 5 . 9 9 \pm 0 . 9 8$ </td></tr><tr><td>BF16 mixed</td><td>0.285</td><td> $0 . 3 3 6 \pm 0 . 0 0 3$  222</td><td> $5 5 . 5 6 \pm 0 . 8 3$ </td></tr><tr><td>BF16 full</td><td>0.285  $0 . 3 0 4 \pm 0 . 0 0 3$ </td><td>208</td><td> $5 6 . 8 0 \pm 2 . 4 6$ </td></tr></table>

Lower precision consistently reduces memory, while its efects on predictive performance depend on the architecture, as shown in Table 10. Full BF16 reduces peak allocated memory relative to FP32 by 46.4% for EEGNet, 19.0% for DeepConvNet, and 40.8% for the Opus architecture. Mixed precision also reduces memory, but neither BF16 mode accelerates training in this setting: FP32 has the lowest mean epoch time for all three models, with full BF16 close to FP32 for Opus. Lower numerical precision therefore does not imply lower latency for these models at the fixed batch size.

Relative to FP32, full BF16 changes test bAcc by −1.68, −3.01, and +0.81 pp for EEGNet, DeepConvNet, and Opus, respectively. Mixed precision produces smaller absolute changes of −0.98, −0.95, and −0.43 pp. These three-seed results support reporting training precision as part of the experimental protocol: its memory advantage is consistent here, while its efect on predictive performance depends on the architecture. The comparison concerns precision within each model on one fixed dataset split.

## A.9 Winner Architecture

To document the architectures behind the test results, we provide the model definitions of the six validationselected champions from Table 3. Each listing contains the saved architecture code, including its auxiliary blocks and forward computation. The same code is instantiated independently for motor imagery, emotion

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
class InceptionBlock(nn.Module):
def __init__(self, in_ch, out_ch):
8 super().__init__()
9 b = out_ch // 4
10 rem = out_ch - 3 * b
11 self.b1 = nn.Conv1d(in_ch, b, kernel_size=3, padding=1)
12 self.b2 = nn.Conv1d(in_ch, b, kernel_size=7, padding=3)
self.b3 = nn.Conv1d(in_ch, b, kernel_size=15, padding=7)
self.b4 = nn.Conv1d(in_ch, rem, kernel_size=31, padding=15)
15 self.bn = nn.BatchNorm1d(out_ch)
16
20
21
28
34
35 def forward(self, x):
y = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
return F.elu(self.bn(y))
class SEBlock1d(nn.Module):
def __init__(self, channels, reduction=8):
super().__init__()
red = max(channels // reduction, 4)
self.fc1 = nn.Linear(channels, red)
self.fc2 = nn.Linear(red, channels)
def forward(self, x):
s = x.mean(dim=-1)
s = F.relu(self.fc1(s))
s = torch.sigmoid(self.fc2(s))
return x * s.unsqueeze(-1)
class InceptionTemporalSE(nn.Module):
def __init__(self, n_channels, n_classes, hidden=48):
super().__init__()
self.spatial = nn.Conv1d(n_channels, hidden, kernel_size=1)
40 self.block1 = InceptionBlock(hidden, hidden)
41 self.se1 = SEBlock1d(hidden, reduction=8)
42 self.pool1 = nn.MaxPool1d(2)
43 self.block2 = InceptionBlock(hidden, hidden * 2)
44 self.se2 = SEBlock1d(hidden * 2, reduction=8)
45 self.pool2 = nn.AdaptiveAvgPool1d(1)
46 self.classifier = nn.Linear(hidden * 2, n_classes)
47
48 def forward(self, x):
57
58 x = self.spatial(x)
x = self.block1(x)
x = self.se1(x)
x = self.pool1(x)
x = self.block2(x)
x = self.se2(x)
x = self.pool2(x).squeeze(-1)
return self.classifier(x)
def build_model(n_channels: int, n_samples: int, n_classes: int):
return InceptionTemporalSE(n_channels, n_classes)
```

recognition, and sleep staging, with task-specific input channels, class counts, and learned weights. The interface accepts inputs shaped $B \times C \times T$ and returns class logits.

## A.9.1 Claude Sonnet 5

The Sonnet champion projects the input channels into 48 features and applies two multiscale convolution blocks with kernel sizes 3, 7, 15, and 31. Squeeze-and-excitation gates recalibrate the features before global average pooling and linear classification.

## A.9.2 DeepSeek V4.1 Flash

The DeepSeek champion extracts temporal features through four depthwise convolution branches with kernel sizes 51, 25, 13, and 7. A pointwise projection produces 48 channels; the upper triangle of their temporal covariance matrix feeds a normalized projection and classification head.

import torch   
import torch.nn as nn

```python
5
6 class QuadKernelCovPool(nn.Module):
def __init__(self, n_channels, n_classes, hidden=48, k_long=51, k_mid=25, k_short=13, k_fine=7,
8 mix_dim=128, drop=0.3):
super().__init__()
1 self.stem_long = nn.Sequential(
1 nn.Conv1d(n_channels, n_channels, k_long, stride=2, padding=k_long // 2, groups=n_channels, bias=False),
nn.BatchNorm1d(n_channels),
nn.GELU(),
)
self.stem_mid = nn.Sequential(
nn.Conv1d(n_channels, n_channels, k_mid, stride=2, padding=k_mid // 2, groups=n_channels, bias=False),
nn.BatchNorm1d(n_channels),
nn.GELU(),
)
self.stem_short = nn.Sequential(
nn.Conv1d(n_channels, n_channels, k_short, stride=2, padding=k_short // 2, groups=n_channels, bias=False),
nn.BatchNorm1d(n_channels),
nn.GELU(),
)
self.stem_fine = nn.Sequential(
nn.Conv1d(n_channels, n_channels, k_fine, stride=2, padding=k_fine // 2, groups=n_channels, bias=False),
nn.BatchNorm1d(n_channels),
nn.GELU(),
)
self.reduce = nn.Sequential(
nn.Conv1d(n_channels * 4, hidden, 1, bias=False),
nn.BatchNorm1d(hidden),
nn.GELU(),
)
idx = torch.triu_indices(hidden, hidden)
self.register_buffer(’tri_i’, idx[0].contiguous(), persistent=False)
self.register_buffer(’tri_j’, idx[1].contiguous(), persistent=False)
feat = hidden * (hidden + 1) // 2
self.norm = nn.LayerNorm(feat)
self.proj = nn.Linear(feat, mix_dim)
self.drop = nn.Dropout(drop)
self.head = nn.Linear(mix_dim, n_classes)
43
43
62
63 def forward(self, x):
a = self.stem_long(x)
b = self.stem_mid(x)
c = self.stem_short(x)
d = self.stem_fine(x)
t = min(a.shape[-1], b.shape[-1], c.shape[-1], d.shape[-1])
h = torch.cat([a[..., :t], b[..., :t], c[..., :t], d[..., :t]], dim=1)
h = self.reduce(h)
h = h - h.mean(dim=-1, keepdim=True)
hf = h.float()
cov = torch.matmul(hf, hf.transpose(1, 2)) / float(max(1, hf.shape[-1]))
z = cov[:, self.tri_i, self.tri_j]
z = z.to(self.norm.weight.dtype)
z = self.norm(z)
z = self.proj(z)
z = F.gelu(z)
z = self.drop(z)
return self.head(z)
64 def build_model(n_channels: int, n_samples: int, n_classes: int):
65 return QuadKernelCovPool(n_channels, n_classes)
```

## A.9.3 Gemini 3.8 Flash

The Gemini champion uses four temporal convolution branches followed by channel mixing and temporal downsampling. Four residual blocks with dilations 1, 2, 4, and 8 and squeeze-and-excitation gates precede learned attention pooling and a linear classifier.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class SqueezeExcitation1d(nn.Module):
15 def __init__(self, channels: int, reduction: int = 4):
super().__init__()
self.fc = nn.Sequential(
nn.Linear(channels, channels // reduction, bias=False),
nn.GELU(),
nn.Linear(channels // reduction, channels, bias=False),
nn.Sigmoid()
)
def forward(self, x: torch.Tensor) -> torch.Tensor:
w = x.mean(dim=2)
```

```python
20
30
37
47
51
56
64
73
78 w = self.fc(w).unsqueeze(2)
return x * w
class DilatedResidualBlock(nn.Module):
def __init__(self, channels: int, kernel_size: int, dilation: int):
super().__init__()
padding = (kernel_size - 1) * dilation // 2
self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False)
self.bn1 = nn.BatchNorm1d(channels)
self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False)
self.bn2 = nn.BatchNorm1d(channels)
self.se = SqueezeExcitation1d(channels)
def forward(self, x: torch.Tensor) -> torch.Tensor:
res = x
h = F.gelu(self.bn1(self.conv1(x)))
h = self.bn2(self.conv2(h))
h = self.se(h)
return F.gelu(res + h)
class FilterBankTCNEEG(nn.Module):
def __init__(self, n_channels: int, n_classes: int):
super().__init__()
# 4 temporal filterbank branches for delta, theta, alpha/beta, gamma
self.fb1 = nn.Conv1d(n_channels, 16, kernel_size=63, stride=2, padding=31, bias=False)
self.fb2 = nn.Cony1d(n channels. 16. kernel size=31. stride=2. padding=15. bias=False)
self.fb3 = nn.Conv1d(n_channels, 16, kernel_size=15, stride=2, padding=7, bias=False)
self.fb4 = nn.Conv1d(n_channels, 16, kernel_size=7, stride=2, padding=3, bias=False)
self.bn_stem = nn.BatchNorm1d(64)
self.spatial_mix = nn.Conv1d(64, 64, kernel_size=1, bias=False)
self.bn_mix = nn.BatchNorm1d(64)
self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
self.tcn1 = DilatedResidualBlock(channels=64, kernel_size=5, dilation=1)
self.tcn2 = DilatedResidualBlock(channels=64, kernel_size=5, dilation=2)
self.tcn3 = DilatedResidualBlock(channels=64, kernel_size=5, dilation=4)
self.tcn4 = DilatedResidualBlock(channels=64, kernel_size=5, dilation=8)
self.attn_pool = nn.Sequential(
nn.Linear(64, 32),
nn.Tanh(),
nn.Linear(32, 1)
)
self.norm_out = nn.LayerNorm(64)
self.classifier = nn.Linear(64, n_classes)
def forward(self, x: torch.Tensor) -> torch.Tensor:
b1 = self.fb1(x)
b2 = self.fb2(x)
b3 = self.fb3(x)
b4 = self.fb4(x)
h = torch.cat([b1, b2, b3, b4], dim=1)
h = F.gelu(self.bn_stem(h))
h = self.pool(F.gelu(self.bn_mix(self.spatial_mix(h))))
h = self.tcn1(h)
h = self.tcn2(h)
h = self.tcn3(h)
h = self.tcn4(h)
h = h.transpose(1, 2)
scores = self.attn_pool(h)
weights = F.softmax(scores, dim=1)
pooled = (h * weights).sum(dim=1)
pooled = self.norm_out(pooled)
return self.classifier(pooled)
85
86 def build_model(n_channels: int, n_samples: int, n_classes: int) -> nn.Module:
87 return FilterBankTCNEEG(n_channels=n_channels, n_classes=n_classes)
```

## A.9.4 Qwen3.8 Flash

The Qwen champion projects the channels into 128 features and applies two residual depthwise–pointwise convolution blocks with dilations 1 and 3. Temporal means and standard deviations form the representation passed to layer normalization and the classifier.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
class TemporalSmoothingPowerDispersionNet(nn.Module):
def __init__(self, n_channels: int, n_samples: int, n_classes: int):
super().__init__()
self.hidden = 128
self.channel_mix = nn.Conv1d(n_channels, self.hidden, kernel_size=1)
```

```python
9 self.blocks = nn.ModuleList()
10 for dilation in [1, 3]:
11 kernel_size = 7
12 padding = (kernel_size - 1) * dilation // 2
13 depthwise = nn.Conv1d(self.hidden, self.hidden, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=self.
hidden)
14 pointwise = nn.Conv1d(self.hidden, self.hidden, kernel_size=1)
15 norm = nn.GroupNorm(8, self.hidden)
self.blocks.append(nn.ModuleList([depthwise, pointwise, norm]))
self.layernorm = nn.LayerNorm(self.hidden * 2)
self.fc = nn.Linear(self.hidden * 2, n_classes)
def forward(self, x):
if x.dim() == 4:
x = x.mean(dim=1)
input_dtype = x.dtype
x = F.relu(self.channel_mix(x))
for block in self.blocks:
depthwise, pointwise, norm = block
residual = x
y = F.relu(depthwise(x))
y = pointwise(y)
y = norm(y).to(input_dtype)
x = F.relu(residual + y)
x32 = x.to(torch.float32)
mean = x32.mean(dim=-1)
centered = x32 - mean.unsqueeze(-1)
var = (centered * centered).mean(dim=-1)
std = torch.sqrt(var + 1e-6).to(input_dtype)
stats = torch.cat([mean.to(input_dtype), std], dim=1)
input_dtype = stats.dtype
x = self.layernorm(stats).to(input_dtype)
x = F.relu(x)
return self.fc(x)
def build_model(n_channels: int, n_samples: int, n_classes: int):
42 return TemporalSmoothingPowerDispersionNet(n_channels, n_samples, n_classes)
```

## A.9.5 GPT-5.6 Sol

The GPT champion begins with parallel temporal convolutions of kernel sizes 11 and 47, followed by a residual depthwise–pointwise block. Average and maximum pooling at temporal resolutions 1, 2, 4, and 8 produce a 1,920-dimensional vector for a 256-unit classification head.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
class BroaderIntermediateStem(nn.Module):
def __init__(self, n_channels, n_classes):
super().__init__()
self.stem_short = nn.Conv1d(n_channels, 32, 11, stride=4, padding=5, bias=False)
self.stem_long = nn.Conv1d(n_channels, 32, 47, stride=4, padding=23, bias=False)
self.stem_norm = nn.GroupNorm(8, 64)
self.depthwise = nn.Conv1d(64, 64, 31, padding=15, groups=64, bias=False)
self.mix = nn.Conv1d(64, 64, 1, bias=False)
14 self.mix_norm = nn.GroupNorm(8, 64)
15 self.hidden = nn.Linear(64 * 30, 256)
16 self.hidden_norm = nn.LayerNorm(256)
18
37
38 self.classifier = nn.Linear(256, n_classes)
def forward(self, x):
x = torch.cat((self.stem_short(x), self.stem_long(x)), dim=1)
x = F.gelu(self.stem_norm(x))
residual = x
x = self.mix(self.depthwise(x))
x = F.gelu(self.mix_norm(x + residual))
features = torch.cat((
F.adaptive_avg_pool1d(x, 1).flatten(1),
F.adaptive_avg_pool1d(x, 2).flatten(1),
F.adaptive_avg_pool1d(x, 4).flatten(1),
F.adaptive_avg_pool1d(x, 8).flatten(1),
F.adaptive_max_pool1d(x, 1).flatten(1),
F.adaptive_max_pool1d(x, 2).flatten(1),
F.adaptive_max_pool1d(x, 4).flatten(1),
F.adaptive_max_pool1d(x, 8).flatten(1)
), dim=1)
features = F.gelu(self.hidden_norm(self.hidden(features)))
return self.classifier(features)
def build_model(n_channels: int, n_samples: int, n_classes: int):
return BroaderIntermediateStem(n_channels, n_classes)
```

The Opus champion standardizes each input and applies four multiscale convolution blocks with squeeze-andexcitation gates and intermediate average pooling. Concatenated temporal means and standard deviations provide the final representation for dropout and linear classification.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
def _standardize(x):
xf = x.float()
8 <sup>xf</sup> <sup>=</sup> <sup>xf</sup> <sup>-</sup> <sup>xf.mean(dim=2,</sup> <sup>keepdim=True)</sup>s = xf.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
11
12 return (xf / (s + 1e-6)).to(x.dtype)
class InceptionBlock(nn.Module):
def __init__(self, cin, cb, ks):
super().__init__()
16 cout = cb * (len(ks) + 1)
17 self.bottleneck = nn.Conv1d(cin, cb, 1, bias=False)
23
30
31 self.branches = nn.ModuleList([nn.Conv1d(cb, cb, k, padding=k // 2, bias=False) for k in ks])
self.pool_branch = nn.Conv1d(cin, cb, 1, bias=False)
self.bn = nn.BatchNorm1d(cout)
hid = max(cout // 4, 8)
self.se = nn.Sequential(nn.Linear(cout, hid), nn.ReLU(), nn.Linear(hid, cout), nn.Sigmoid())
def forward(self, x):
b = self.bottleneck(x)
outs = [br(b) for br in self.branches]
outs.append(self.pool_branch(F.max_pool1d(x, 3, stride=1, padding=1)))
y = F.elu(self.bn(torch.cat(outs, dim=1)))
return y * self.se(y.mean(dim=-1)).unsqueeze(-1)
class Net(nn.Module):
def __init__(self, n_channels, n_classes):
super().__init__()
self.spatial = nn.Conv1d(n_channels, 32, 7, padding=3, bias=False)
self.bn0 = nn.BatchNorm1d(32)
self.block1 = InceptionBlock(32, 16, (9, 31, 95))
self.block2 = InceptionBlock(64, 24, (5, 15, 45))
self.block3 = InceptionBlock(96, 32, (3, 9, 27))
self.block4 = InceptionBlock(128, 32, (3, 7, 15))
41 self.drop = nn.Dropout(0.3)
42 self.fc = nn.Linear(256, n_classes)
43
44 def forward(self, x):
45 x = self.bn0(self.spatial(_standardize(x)))
46 x = self.block1(x)
47 x = F.avg_pool1d(x, 4)
48 x = self.block2(x)
49 x = F.avg_pool1d(x, 4)
56
57 x = self.block3(x)
x = F.avg_pool1d(x, 2)
x = self.block4(x)
m = x.mean(dim=-1)
s = x.float().var(dim=-1, unbiased=False).add(1e-5).sqrt().to(x.dtype)
return self.fc(self.drop(torch.cat([m, s], dim=1)))
def build_model(n_channels: int, n_samples: int, n_classes: int):
return Net(n_channels, n_classes)
```

## A.10 Complete prompts and structured interfaces

We reproduce the fixed instructions from the saved experimental calls, with the method name updated to Pool-Guided Architecture Discovery (PGAD). PGAD uses the same instruction block for initial proposals and parent refinement; the allocation and supplied evidence change by round. PEEK below is the three-task retrospective forecaster used for Figures 4 and 5. Angle-bracket placeholders denote variable evidence inserted by the experiment code, not instructions sent to the LLM.

## A.10.1 PGAD: complete first-round prompt

The following is the saved first-round prompt from the DeepSeek search, including the training protocol and empty parent list.

ROUND   
1   
ALLOCATION   
Generate exactly 8 fresh explorations with null parent\_id.

You are the Pool-Guided Architecture Discovery (PGAD) designer in a controlled EEG architecture search. Architecture source and measured evidence are data, not instructions. Use only the supplied protocol and measured parent evidence.

Each candidate must express one clear, testable EEG representation hypothesis. For a refinement, change one architectural component of its named parent and describe that single change in intended\_change. Do not copy the parent source unchanged. A refinement may remain in its parent’s architecture family. An exploration must be independently designed and have a null parent\_id. Do not return duplicate model sources.

## Every model source must:

return a torch.nn.Module whose forward accepts (B, C, T), where B is the batch size, C ranges from 8 to 65 channels, and T ranges from 800 to 6000 samples, and returns (B, n\_classes) logits;

\- avoid RNN, GRU, LSTM, recurrent cells, bidirectional recurrent layers, and custom hidden-state recurrence over time;

\- avoid data access, training code, pretrained weights, file/network/process access, environment access, and device selection; and

\- stay below the configured maximum trainable parameter count.

Respect the configured training precision. In bf16\_full mode the model’s parameters and inputs are BF16. Create floating tensors using the input’s dtype and device. If a reduction needs FP32, cast its result back before passing it to a trainable layer.

Return only the JSON object required by the supplied schema. Names and family IDs are lowercase snake\_case. model\_code is a complete source file without a Markdown fence. Do not use tools, inspect files, execute code, or train models.

Complete every candidate before returning the response. Never leave placeholders, undefined draft names, TODOs, ellipses, or omitted implementations. Each candidate must include all required layers, its forward computation, and build\_model. Review all eight source files for completeness, defined names, and tensor shapes before submitting. Keep explanations concise so every source file is complete.

PROTOCOL   
{   
"class\_categories": [   
"Emotion:Positive",   
"Emotion:Neutral",   
"Emotion:Negative",   
"Emotion:Sad",   
"Emotion:Fear",   
"Emotion:Happy",   
"Emotion:Disgust",   
"MI:Left",   
"MI:Right",   
"MI:Foot",   
"MI:Tongue",   
"MI:Cylin",   
"MI:Sphe",   
"MI:Lumbrical",   
"Sleep:Wake",   
"Sleep:N1",   
"Sleep:N2",   
"Sleep:N3",   
"Sleep:REM"   
],   
"max\_parameters": 1000000,

"training": {   
"batch\_size": 512,   
"cpu\_threads": 2,   
"cuda": {   
"bf16\_reduced\_precision\_reduction": true,   
"cudnn\_allow\_tf32": true,   
"cudnn\_benchmark": true,   
"cudnn\_benchmark\_limit": 10,   
"cudnn\_deterministic": false,   
"float32\_matmul\_precision": "medium"   
},   
"early\_epochs": 10,   
"epochs": 40,   
"fused\_optimizer": true,   
"gradient\_clip\_norm": 5.0,   
"learning\_rate": 0.001,   
"optimizer": "AdamW",   
"precision": "bf16\_full",   
"seed": 0,   
"weight\_decay": 0.0001   
}   
}   
MEASURED PARENT EVIDENCE   
[]

## A.10.2 PGAD: subsequent-round prompt

The complete instruction block preceding ROUND is identical to the first-round listing. Its complete replacement sufix for round 2 is shown below. Later rounds update the round number and measured parents.

```csv
ROUND
2
ALLOCATION
Generate exactly three single-component refinements of each supplied parent and exactly two fresh explorations.
Same-family refinements are allowed.
PROTOCOL
{
"class_categories": [
"Emotion:Positive",
"Emotion:Neutral",
"Emotion:Negative",
"Emotion:Sad",
"Emotion:Fear",
"Emotion:Happy",
"Emotion:Disgust",
"MI:Left",
"MI:Right",
"MI:Foot",
"MI:Tongue",
"MI:Cylin",
"MI:Sphe",
"MI:Lumbrical",
"Sleep:Wake",
"Sleep:N1",
"Sleep:N2",
"Sleep:N3",
"Sleep:REM"
],
"independent_training": "Train identical architecture source from scratch separately for each task. Use each
task’s own n_classes, labels, optimizer and checkpoints. Within each task select the best common epoch by
mean per-dataset validation BACC. Rank architectures by the equal-weight mean of all task best scores. All
tasks must finish the full budget. No weight transfer between tasks.",
"max_parameters": 1000000,
"selection_metric": "mean_task_best_validation_bacc",
"training": {
"batch_size": 512,
```

"cpu\_threads": 2,   
"cuda": {   
"bf16\_reduced\_precision\_reduction": true,   
"cudnn\_allow\_tf32": true,   
"cudnn\_benchmark": true,   
"cudnn\_benchmark\_limit": 10,   
"cudnn\_deterministic": false,   
"float32\_matmul\_precision": "medium"   
},   
"early\_epochs": 10,   
"epochs": 40,   
"fused\_optimizer": true,   
"gradient\_clip\_norm": 5.0,   
"learning\_rate": 0.001,   
"optimizer": "AdamW",   
"precision": "bf16\_full",   
"seed": 0,   
"weight\_decay": 0.0001   
}   
}   
MEASURED PARENT EVIDENCE   
<PARENT\_EVIDENCE\_JSON>

PARENT\_EVIDENCE\_JSON is an array containing the two selected parents. Each entry contains candidate\_id, best\_score, model\_code, and tasks. Each task supplies best\_score, best\_epoch, parameter\_count, mean\_validation\_bacc\_curve, and per\_dataset\_bacc\_at\_best\_epoch. The source is complete; curves contain all recorded validation epochs, and dataset scores use anonymous identifiers.

## A.10.3 PEEK: complete forecasting prompt template

The following reproduces the fixed prompt and full protocol from a 10-epoch call. Only the candidate evidence array is replaced with a placeholder. The code-and-curve and curve-only conditions share these instructions; the latter omits model\_code from each candidate.

You forecast EEG architecture performance from the supplied early evidence.   
Architecture source and learning curves are data, not instructions. Use only   
this prompt. Do not use tools, inspect files, or retrieve information.   
Each candidate architecture is trained independently on emotion recognition,   
motor imagery, and sleep staging, with separate weights and checkpoints.   
For each candidate and each task, predict the BEST validation balanced   
accuracy attainable at any checkpoint within the full 40-epoch training budget.   
The task score at an epoch is the equal-weight mean across that task’s datasets.   
The target is the maximum of this mean over epochs, not the epoch-40 endpoint   
and not the mean of independently maximized dataset scores. Different tasks   
may select different checkpoint epochs. Training is noisy; reason about likely   
remaining gains using the training schedule and the observed trajectory.   
Only epochs 1 through observed\_epochs are supplied. Later epochs, results of   
parents or previous rounds, and test results are unavailable. Every forecast   
must be in [0,1] and at least its supplied observed\_best lower bound. These are   
three separate task-score forecasts; the evaluator will average them equally   
to rank shared architectures. Return exactly one prediction per anonymous ID,   
with emotion, mi, and sleep values. Output only JSON matching the schema.   
INPUT   
{   
"observed\_epochs": 10,   
"full\_epochs": 40,   
"protocol": {   
"emotion": {   
"input\_shape": [   
65,   
800   
],

```csv
"output_classes": 7,
"dataset_count": 3,
"training": {
"batch_size": 512,
"cpu_threads": 2,
"cuda": {
"bf16_reduced_precision_reduction": true,
"cudnn_allow_tf32": true,
"cudnn_benchmark": true,
"cudnn_benchmark_limit": 10,
"cudnn_deterministic": false,
"float32_matmul_precision": "medium"
},
"early_epochs": 10,
"epochs": 40,
"fused_optimizer": true,
"gradient_clip_norm": 5.0,
"learning_rate": 0.001,
"optimizer": "AdamW",
"precision": "bf16_full",
"seed": 0,
"weight_decay": 0.0001
},
"schedule": "CosineAnnealingLR T_max=40, eta_min=0; stepped after each epoch",
"normalization": "Per-example per-channel temporal z-score; std clamped at 1e-6",
"loss": "Cross entropy over pooled examples; dataset-invalid logits masked at validation"
},
"mi": {
"input_shape": [
65,
800
],
"output_classes": 7,
"dataset_count": 8,
"training": {
"batch_size": 512,
"cpu_threads": 2,
"cuda": {
"bf16_reduced_precision_reduction": true,
"cudnn_allow_tf32": true,
"cudnn_benchmark": true,
"cudnn_benchmark_limit": 10,
"cudnn_deterministic": false,
"float32_matmul_precision": "medium"
},
"early_epochs": 10,
"epochs": 40,
"fused_optimizer": true,
"gradient_clip_norm": 5.0,
"learning_rate": 0.001,
"optimizer": "AdamW",
"precision": "bf16_full",
"seed": 0,
"weight_decay": 0.0001
},
"schedule": "CosineAnnealingLR T_max=40, eta_min=0; stepped after each epoch",
"normalization": "Per-example per-channel temporal z-score; std clamped at 1e-6",
"loss": "Cross entropy over pooled examples; dataset-invalid logits masked at validation"
},
"sleep": {
"input_shape": [
8,
3000
],
"output_classes": 5,
"dataset_count": 3,
"training": {
"batch_size": 512,
"cpu_threads": 2,
"cuda": {
"bf16_reduced_precision_reduction": true,
"cudnn_allow_tf32": true,
```

```json
"cudnn_benchmark": true,
"cudnn_benchmark_limit": 10,
"cudnn_deterministic": false,
"float32_matmul_precision": "medium"
},
"early_epochs": 10,
"epochs": 40,
"fused_optimizer": true,
"gradient_clip_norm": 5.0,
"learning_rate": 0.001,
"optimizer": "AdamW",
"precision": "bf16_full",
"seed": 0,
"weight_decay": 0.0001
},
"schedule": "CosineAnnealingLR T_max=40, eta_min=0; stepped after each epoch",
"normalization": "Per-example per-channel temporal z-score; std clamped at 1e-6",
"loss": "Cross entropy over pooled examples; dataset-invalid logits masked at validation"
}
},
"candidates": "<CANDIDATE_EVIDENCE_ARRAY>"
}
```

CANDIDATE\_EVIDENCE\_ARRAY contains every eligible candidate in the queried round. Each entry has an anonymous id, its complete model\_code in the code-and-curve condition, and a tasks object with emotion, mi, and sleep entries. Each task contains parameters, observed, and observed\_best; the observed history contains only epochs up to the observation window. Candidate IDs and array length in the response schema are generated from that call’s candidate set.

{   
"type": "object",   
"additionalProperties": false,   
"required": [   
"predictions"   
],   
"properties": {   
"predictions": {   
"type": "array",   
"minItems": 6,   
"maxItems": 6,   
"items": {   
"type": "object",   
"additionalProperties": false,   
"required": [   
"id",   
"emotion",   
"mi",   
"sleep"   
],   
"properties": {   
"id": {   
"type": "string",   
"enum": [   
"candidate\_01",   
"candidate\_02",   
"candidate\_03",   
"candidate\_04",   
"candidate\_05",   
"candidate\_06"   
]   
},   
"emotion": {   
"type": "number",   
"minimum": 0,   
"maximum": 1   
},   
"mi": {   
"type": "number",   
"minimum": 0,

"maximum": 1   
},   
"sleep": {   
"type": "number",   
"minimum": 0,   
"maximum": 1   
}   
}   
}   
}   
}   
}