# BioStudyBench: Evaluating Agents on Post-Cutoff Biomedical Studies

David Li<sup>1</sup>, Shaamil Karim<sup>2</sup>, Christian Gensbigler<sup>2</sup>

<sup>1</sup>University of Waterloo <sup>2</sup>Atlas Discovery

david.li3@uwaterloo.ca

{shaamil, christian}@atlasdiscovery.bio

## Abstract

We evaluate whether AI agents can match the reported findings of published biomedical studies using public data. Existing evaluations do not consistently separate analysis from prior knowledge or retrieval of the published answer. We introduce BioStudyBench, a benchmark of 25 long-horizon analysis tasks drawn from studies first published between July and September 2026, after the developerreported knowledge cutoffs of the models we evaluate, semi-automatically filtered down from 404,019 PubMed records. In each task, the agent receives a neutral research question but no data files, so it must find and download the relevant public data, search the literature through tools that return only records dated before its cutoff, and report findings through data analysis. To measure gains over prior knowledge, we run every task both with and without access to data and tools. Across eight models, access to data and tools raises the pass rate by 47 percentage points on average over the no-data baseline. Open-weight models across sizes trail closed-weight models, with the best open-weight model passing 81.3% of tasks against 94.7% for the best closed-weight model.

## 1 Introduction

Re-deriving a published biomedical finding requires locating the right data release, reconstructing the study population, applying the survey or study design and choosing an estimator. Current agents do this unreliably. The best agents reported in the respective papers answer 21% of BixBench’s open-ended analysis questions [11], succeed on 39.9% of RWE-bench’s observational-study tasks [7], and achieve a mean artifact pass rate across tasks of 48% on BixBench3 [6].

These scores are also difficult to interpret (Table 1). These data-analysis benchmarks do not require every source study to postdate every evaluated model’s cutoff, so a correct answer alone does not establish that a finding was newly derived. PaperBench and FIRE-Bench allow web access without date restrictions [15, 17], and Jung et al. [5] report that agents retrieved the ground-truth conclusion “even when instructed not to”. We found that given only the research question for each of 2,131 biomedical papers published after its cutoff, GPT-6 Sol reported no knowledge of 85% of the findings, yet matched the direction and significance of 83% of the non-null ones (Appendix C).

We introduce BioStudyBench, an environment and benchmark for evaluating whether agents can match findings from studies published after their developer-reported knowledge cutoff (Figure 1). The agent is given a research question but not the data. It must find and download relevant public data, write scripts to perform analysis, and report its findings. Each task requires long-horizon analysis, taking agents up to 117 turns per task on average, with runs lasting up to 90 minutes.

![](images/0c3c4b4d57241674dfbebb33ddcc916ac493a379b8bd309a6a2a21aa0a1ca7e8.jpg)  
Figure 1: Overview of the BioStudyBench environment, shown for one task. The agent sees the question (red), dataset (orange) and answer schema (yellow) but not the reference claims (green). Its tools return only records dated before its cutoff, and its answers are scored against the hidden findings; an audit checks the logs, and a paired run without data records the model’s prior.

Table 1: Selected benchmarks for scientific data analysis and conclusion synthesis. ✓ yes; ◦ partial, or as an ablation only; ✗ no; – not applicable. Post-cutoff means a required study-date split, rather than possible inclusion of recent studies.
<table><tr><td></td><td></td><td></td><td>Post-cutoff</td><td>No-data</td><td>Date-restricted Agent acquires Deterministic</td><td></td><td></td></tr><tr><td></td><td># Tasks</td><td>Domain</td><td>tasks?</td><td>control?</td><td>search?</td><td>the data?</td><td>scoring?</td></tr><tr><td>Data analysis</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DiscoveryBench [10]</td><td>264</td><td>multiple</td><td>x</td><td>O</td><td>x</td><td>x</td><td>x</td></tr><tr><td>BixBench [11]</td><td>205 (61)</td><td>bioinformatics</td><td>x</td><td>0</td><td>x</td><td>x</td><td>x</td></tr><tr><td>BixBench3 [6]</td><td>20</td><td>computational biology</td><td>x</td><td>x</td><td>x</td><td>x</td><td>√</td></tr><tr><td>RWE-bench [7]</td><td>162</td><td>EHR (MIMIC-IV)</td><td>x</td><td>x</td><td>x</td><td>x</td><td></td></tr><tr><td>MedResearchBench [16]</td><td>16</td><td>NHANES, SEER</td><td>x</td><td>x</td><td>x</td><td>0</td><td>x</td></tr><tr><td>Time-controlled evaluation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FIRE-Bench [17]</td><td>40</td><td>machine learning</td><td>0</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>SciConBench [5]</td><td>268</td><td>systematic reviews</td><td>√</td><td>0</td><td>√</td><td>一</td><td>x</td></tr><tr><td>BioStudyBench</td><td>25</td><td>biomedical studies</td><td>√</td><td>√</td><td></td><td>√</td><td>J</td></tr></table>

BixBench: 205 questions in 61 capsules (v3). MedResearchBench: 16 tasks, with pilot results on 3. FIRE-Bench: 40 evaluated tasks, plus 60 additional released papers. SciConBench: 268 post-cutoff reviews selected from 9,107; search is restricted by the target review’s publication date, and no-tool controls cover three base models.

## 2 Related work

Automated data analysis. DiscoveryBench [10] derives data-driven hypotheses from papers in six fields and scores them with a language-model judge; BixBench [11] poses open-answer bioinformatics questions, and BixBench3 [6] asks agents to reproduce artifacts from computationalbiology studies given each study’s protocol. Closest to our setting, RWE-bench [7] evaluates agents on 162 observational-study tasks in MIMIC-IV, while MedResearchBench [16] offers 16 tasks using NHANES and SEER, with pilot results on three. Other benchmarks give the agent a paper’s code or text to reproduce [13, 15, 23] or pose data-science questions on supplied data [2, 4, 18]. Our emphasis is on combining post-cutoff studies, restricted retrieval, and paired runs with and without data.

Information control. Live benchmarks refresh questions using recent sources or newly emerging knowledge to limit contamination [19, 20]. FIRE-Bench [17] asks agents to rediscover the insights of 40 machine-learning papers, including both pre- and post-cutoff studies, with unrestricted web access. SciConBench [5], closest to our leakage controls, filters search results by the target review’s publication date and blocks reference-review content, but asks agents to synthesise review conclusions rather than analyse data; Proof of Time [22] freezes evidence to forecast post-cutoff outcomes. Language models can distinguish reported neuroscience results from altered alternatives [9], and analysts diverge on the same data [14], motivating our no-data control and tolerances. See Table 1 for a comparison with these benchmarks.   
2

![](images/7dcbb3cc0758bc7bf548616e0b6bede31ff818b69aac3759a3ec867bb154d342.jpg)  
Figure 2: Construction funnel. Node heights are log-scaled; each node splits in proportion between kept (blue) and excluded (grey) records.

![](images/bc511d53000bc4c56712acffb361f070348a54134847d04dd2b844770cb38d47.jpg)  
Figure 3: Answers without data (hollow) and with data (blue) on two tasks where models match the reference with data (left, centre) and one they struggle with (right).

## 3 The BioStudyBench environment

Tasks. We source tasks in BioStudyBench from published biomedical papers. Our semi-automated pipeline (Section 4) turns each paper into a neutral question and uses the paper’s findings as reference answers. The agent is not given the data. The task names the public source, and the agent must download the raw files, read the codebooks, and merge, recode and weight the records itself. It reports its answers in a fixed format, and a rule-based scorer compares them with the paper’s findings. A task passes if its primary estimate falls within the original paper’s 95% confidence interval, or, for a finding reported only as an association, if its direction and statistical significance match the paper’s. Our score measures agreement with the study’s reported findings. The reference confidence interval provides an operational scoring tolerance; a passing answer does not by itself validate the agent’s data preparation, statistical methods, uncertainty estimates, or scientific interpretation. The agent works in a sandbox with a shell and Python and searches the literature and datasets through tools covering PubMed, Europe PMC, OpenAlex, Crossref, GEO, ClinicalTrials.gov, Zenodo and the Wayback Machine.

Leakage controls. Every task in BioStudyBench originates from a paper first published between July and September 2026, after June 2026, the latest developer-reported knowledge cutoff among the evaluated models (Appendix Table 3). This temporal separation, together with target-paper blocklists and restricted retrieval, reduces contamination risk. However, published cutoffs and publication dates do not establish the absence of prior exposure through all training or post-training data. Besides the literature and dataset tools listed above, the agent’s shell can reach only allowlisted public data hosts, such as the CDC and FDA servers that distribute NHANES and FAERS, and web search is disabled (Figure 1). The literature and dataset tools filter returned records by their reported dates and apply target-paper blocklists (Appendix A).

## 4 Benchmark construction and experimental setup

As shown in Figure 2, we retained the 404,019 PubMed [12] records first made public between July and September 2026 and resolved each record’s earliest appearance across preprint servers and publishers. We designed a structural filter to remove reviews, editorials and case reports, and two classifier passes (GPT-6 Luna) which retain 2,131 studies of existing public data with a single scorable claim and obtainable data. A recall screen then showed GPT-6 Sol only each question, demoted findings it predicted, and a review classifier kept 35 studies (full prompts provided in Appendix E). After scientific exclusions, 33 tasks remain; 25 have accessible data and form BioStudyBench (Appendix F).

We run each model on every task under two conditions, three times each. In the with data and tools condition, the agent works in the full environment, with limits of 300 turns and 90 minutes. In the prior knowledge condition, it receives the same question and answer format but no tools or data, and must predict every requested value from what it already knows. Each repeat’s pass rate uses all 25 tasks as its denominator; timeouts, run errors and runs without scored answers count as failures.

## 5 Results

Data and tools improve reference agreement. Without data, models pass 24–51% of tasks; access to data and tools raises pass rates by 32–55 percentage points (Table 2). These gains do not establish that every successful answer was independently derived.

Open-weight models trail closed-weight models. Open-weight models average 26% against 40% for closed models without data, and 70% against 89% with data (Table 2). The best open-weight model, GLM 5.3, passes fewer tasks than three of the four closed-weight models. Figure 3 shows two tasks where widely differing priors converge on the published value once models analyse the data, and one on which they narrow but stay above it.

<table><tr><td rowspan="2"></td><td colspan="2">Pass rate (%)</td><td rowspan="2">Turns</td></tr><tr><td>Prior knowledge</td><td>With data and tools</td></tr><tr><td>Closed weights GPT-6 Astra</td><td></td><td></td><td>42</td></tr><tr><td>Gemini 3.8 Flash</td><td> $5 0 . 7 { \scriptstyle \pm 8 . 3 }$   $4 0 . 0 { \scriptstyle \pm 1 0 . 6 }$ </td><td> $9 4 . 7 { \scriptstyle \pm 2 . 3 }$   $9 4 . 7 \pm 6 . 1$ </td><td>117</td></tr><tr><td>Claude Opus 5.5</td><td> $4 6 . 7 _ { \pm 6 . 1 }$ </td><td> $9 3 . 3 { \scriptstyle \pm 2 . 3 }$ </td><td>85</td></tr><tr><td>GPT-6 Luna</td><td> $2 4 . 0 { \scriptstyle \pm 4 . 0 }$ </td><td> $7 4 . 7 \pm 6 . 1$ </td><td>64</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">Open weights</td><td></td><td></td></tr><tr><td>GLM 5.3</td><td> $2 6 . 7 _ { \pm 2 . 3 }$ </td><td> $8 1 . 3 { \scriptstyle \pm 6 . 1 }$ </td><td>65</td></tr><tr><td>DeepSeek V4.1 Flash</td><td> $2 5 . 3 _ { \pm 6 . 1 }$ </td><td> $7 6 . 0 { \scriptstyle \pm 4 . 0 }$ </td><td>81</td></tr><tr><td>MiMo V2.6 Pro</td><td> $2 6 . 7 { \scriptstyle \pm 6 . 1 }$ </td><td> $6 6 . 7 \pm 1 2 . 9$ </td><td>43</td></tr><tr><td>Qwen3.8 27B</td><td> $2 4 . 0 { \scriptstyle \pm 0 . 0 }$ </td><td> $5 6 . 0 { \scriptstyle \pm 1 0 . 6 }$ </td><td>97</td></tr></table>

Table 2: Main results: pass rate (%) over 25 tasks, $\mathrm { m e a n } \pm \mathrm { s . d }$ . across three repeats, and mean turns per run with data where recorded.

Qualitative failure cases. The inspected failures illustrate mismatches in analysis and reporting (Appendix Figure 4). These failures span a range of modes: when a paper leaves an analytic choice implicit, models may choose an analysis different from the paper’s; models may compute an estimate matching the reference but misstate the conclusion; and when a data source fails, models often fail to look for another route to the data.

## 6 Limitations

BioStudyBench contains only 25 long-horizon tasks. Its temporal separation expires once evaluated models’ reported cutoffs extend past September 2026. Our construction pipeline outlasts any one set of tasks, however, and can build a new window of post-cutoff papers for each generation of models.

Date filtering and host restrictions do not guarantee that every accessible source is free of answerbearing information. Additionally, an incorrect analysis can produce a correct result, while a scientifically defensible alternative analysis can fall outside the scoring tolerance. Our reported pass rates therefore measure reference agreement rather than comprehensive methodological correctness.

## 7 Conclusion

BioStudyBench evaluates reference agreement on post-cutoff biomedical studies using restricted retrieval and paired no-data controls. Access to data and tools raises pass rates by 32–55 percentage points over prior knowledge alone; open-weight models trail closed-weight models in both conditions.

## References

[1] Anthropic. Eval awareness in claude opus 4.6’s browsecomp performance. https://www. anthropic.com/engineering/eval-awareness-browsecomp, March 2026. Accessed: 2026-09-27.

[2] Ziru Chen, Shijie Chen, Yuting Ning, Qianheng Zhang, Boshi Wang, Botao Yu, Yifei Li, Zeyi Liao, Chen Wei, Zitong Lu, Vishal Dey, Mingyi Xue, Frazier N. Baker, Benjamin Burns, Daniel Adu-Ampratwum, Xuhui Huang, Xia Ning, Song Gao, Yu Su, and Huan Sun. ScienceAgent-Bench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery. arXiv preprint arXiv:2410.05080, 2024. URL https://arxiv.org/abs/2410.05080. ICLR 2025.

[3] Xin Gao, Ruiyi Zhang, Daniel Du, Saurabh Mahindre, Sai Ashish Somayajula, and Pengtao Xie. Can prompts rewind time for LLMs? evaluating the effectiveness of prompted knowledge cutoffs. arXiv preprint arXiv:2510.02340, 2025. URL https://arxiv.org/abs/2510.02340.

[4] Ken Gu, Ruoxi Shang, Ruien Jiang, Keying Kuang, Richard-John Lin, Donghe Lyu, Yue Mao, Youran Pan, Teng Wu, Jiaqian Yu, Yikun Zhang, Tianmai M. Zhang, Lanyi Zhu, Mike A. Merrill, Jeffrey Heer, and Tim Althoff. BLADE: Benchmarking Language Model Agents for Data-Driven Science. arXiv preprint arXiv:2408.09667, 2024. URL https://arxiv.org/ abs/2408.09667.

[5] Hayoung Jung, Pedro Viana Diniz, José Reinaldo Corrêa Roveda, Abner Fernandes da Silva, Haeun Jung, Enoch Tsai, Aleksandra Korolova, and Manoel Horta Ribeiro. Can AI agents synthesize scientific conclusions? arXiv preprint arXiv:2606.11337, 2026. URL https: //arxiv.org/abs/2606.11337.

[6] Zane Koch, Asmamaw T. Wassie, Javier Valdes-Aleman, Jason Lee, Michaela M. Hinks, Samuel G. Rodriques, Andrew D. White, and Jon M. Laurent. BixBench3: Benchmarking AI agents on research-study-scale computational biology tasks. arXiv preprint arXiv:2608.25286, 2026. URL https://arxiv.org/abs/2608.25286.

[7] Dubai Li, Yuxiang He, Yan Hu, Yu Tian, and Jingsong Li. Can LLM agents generate real-world evidence? evaluating observational studies in medical databases. arXiv preprint arXiv:2603.22767, 2026. URL https://arxiv.org/abs/2603.22767.

[8] Alejandro Lopez-Lira, Yuehua Tang, and Mingyin Zhu. The memorization problem: Can we trust LLMs’ economic forecasts? arXiv preprint arXiv:2504.14765, 2025. URL https: //arxiv.org/abs/2504.14765.

[9] Xiaoliang Luo, Akilles Rechardt, Guangzhi Sun, Kevin K. Nejad, Felipe Yáñez, Bati Yilmaz, Kangjoo Lee, Alexandra O. Cohen, Valentina Borghesani, Anton Pashkov, Daniele Marinazzo, Jonathan Nicholas, Alessandro Salatiello, Ilia Sucholutsky, Pasquale Minervini, Sepehr Razavi, Roberta Rocca, Elkhan Yusifov, Tereza Okalova, Nianlong Gu, Martin Ferianc, Mikail Khona, Kaustubh R. Patil, Pui-Shee Lee, Rui Mata, Nicholas E. Myers, Jennifer K Bizley, Sebastian Musslick, Isil Poyraz Bilgin, Guiomar Niso, Justin M. Ales, Michael Gaebler, N Apurva Ratan Murty, Leyla Loued-Khenissi, Anna Behler, Chloe M. Hall, Jessica Dafflon, Sherry Dongqi Bao, and Bradley C. Love. Large language models surpass human experts in predicting neuroscience results. Nature Human Behaviour, 9:305–315, 2025. doi: 10.1038/s41562-024-02046-9.

[10] Bodhisattwa Prasad Majumder, Harshit Surana, Dhruv Agarwal, Bhavana Dalvi Mishra, Abhijeetsingh Meena, Aryan Prakhar, Tirth Vora, Tushar Khot, Ashish Sabharwal, and Peter Clark. DiscoveryBench: Towards Data-Driven Discovery with Large Language Models. arXiv preprint arXiv:2407.01725, 2024. URL https://arxiv.org/abs/2407.01725.

[11] Ludovico Mitchener, Jon M Laurent, Alex Andonian, Benjamin Tenmann, Siddharth Narayanan, Geemi P Wellawatte, Andrew White, Lorenzo Sani, and Samuel G Rodriques. BixBench: a Comprehensive Benchmark for LLM-based Agents in Computational Biology. arXiv preprint arXiv:2503.00096, 2025. URL https://arxiv.org/abs/2503.00096v3.

[12] Eric W Sayers, Evan E Bolton, Anna M Fine, Christopher Kelly, Sunghwan Kim, Melissa Landrum, Stacy Lathrop, Adriana Malheiro, Terence D Murphy, Lon Phan, Shashikant Pujar, Barton W Trawick, Valerie A Schneider, and Kim D Pruitt. Database resources of the National Center for Biotechnology Information in 2026. Nucleic Acids Research, 54(D1):D20–D27, 2026. doi: 10.1093/nar/gkaf1060.

[13] Zachary S. Siegel, Sayash Kapoor, Nitya Nadgir, Benedikt Stroebl, and Arvind Narayanan. CORE-Bench: Fostering the Credibility of Published Research Through a Computational Reproducibility Agent Benchmark. arXiv preprint arXiv:2409.11363, 2024. URL https: //arxiv.org/abs/2409.11363.

[14] R. Silberzahn, E. L. Uhlmann, D. P. Martin, P. Anselmi, F. Aust, E. Awtrey, Š. Bahník, F. Bai, C. Bannard, E. Bonnier, R. Carlsson, F. Cheung, G. Christensen, R. Clay, M. A. Craig, A. Dalla Rosa, L. Dam, M. H. Evans, I. Flores Cervantes, N. Fong, M. Gamez-Djokic, A. Glenz, S. Gordon-McKeon, T. J. Heaton, K. Hederos, M. Heene, A. J. Hofelich Mohr, F. Högden, K. Hui, M. Johannesson, J. Kalodimos, E. Kaszubowski, D. M. Kennedy, R. Lei, T. A. Lindsay, S. Liverani, C. R. Madan, D. Molden, E. Molleman, R. D. Morey, L. B. Mulder, B. R. Nijstad, N. G. Pope, B. Pope, J. M. Prenoveau, F. Rink, E. Robusto, H. Roderique, A. Sandberg, E. Schlüter, F. D. Schönbrodt, M. F. Sherman, S. A. Sommer, K. Sotak, S. Spain, C. Spörlein, T. Stafford, L. Stefanutti, S. Tauber, J. Ullrich, M. Vianello, E.-J. Wagenmakers, M. Witkowiak, S. Yoon, and B. A. Nosek. Many analysts, one data set: Making transparent how variations in analytic choices affect results. Advances in Methods and Practices in Psychological Science, 1(3):337–356, 2018. doi: 10.1177/2515245917747646.

[15] Giulio Starace, Oliver Jaffe, Dane Sherburn, James Aung, Jun Shern Chan, Leon Maksin, Rachel Dias, Evan Mays, Benjamin Kinsella, Wyatt Thompson, Johannes Heidecke, Amelia Glaese, and Tejal Patwardhan. PaperBench: Evaluating AI’s Ability to Replicate AI Research. arXiv preprint arXiv:2504.01848, 2025. URL https://arxiv.org/abs/2504.01848.

[16] Shuping Tan and Zhanxiao Tian. MedResearchBench: A Multi-Domain Benchmark for Evaluating AI Research Agents on Clinical Medical Research. medRxiv, 2026. doi: 10.64898/2026. 03.30.26349749. URL https://doi.org/10.64898/2026.03.30.26349749.

[17] Zhen Wang, Fan Bai, Zhongyan Luo, Jinyan Su, Kaiser Sun, Xinle Yu, Jieyuan Liu, Kun Zhou, Claire Cardie, Mark Dredze, Zhiting Hu, and Eric P. Xing. FIRE-Bench: Evaluating AI agents on the rediscovery of scientific insights. arXiv preprint arXiv:2602.02905, 2026. URL https://arxiv.org/abs/2602.02905.

[18] Zifeng Wang, Benjamin Danek, and Jimeng Sun. BioDSA-1K: Benchmarking Data Science Agents for Biomedical Research. arXiv preprint arXiv:2505.16100, 2025. URL https: //arxiv.org/abs/2505.16100.

[19] Colin White, Samuel Dooley, Manley Roberts, Arka Pal, Ben Feuer, Siddhartha Jain, Ravid Shwartz-Ziv, Neel Jain, Khalid Saifullah, Sreemanti Dey, Shubh-Agrawal, Sandeep Singh Sandha, Siddartha Naidu, Chinmay Hegde, Yann LeCun, Tom Goldstein, Willie Neiswanger, and Micah Goldblum. LiveBench: A Challenging, Contamination-Limited LLM Benchmark. arXiv preprint arXiv:2406.19314, 2024. URL https://arxiv.org/abs/2406.19314. ICLR 2025.

[20] Xiaobao Wu, Liangming Pan, Yuxi Xie, Ruiwen Zhou, Shuai Zhao, Yubo Ma, Mingzhe Du, Rui Mao, Anh Tuan Luu, and William Yang Wang. AntiLeakBench: Preventing Data Contamination by Automatically Constructing Benchmarks with Updated Real-World Knowledge. arXiv preprint arXiv:2412.13670, 2024. URL https://arxiv.org/abs/2412.13670.

[21] Ruijie Xu, Zengzhi Wang, Run-Ze Fan, and Pengfei Liu. Benchmarking benchmark leakage in large language models. arXiv preprint arXiv:2404.18824, 2024. URL https://arxiv.org/ abs/2404.18824.

[22] Bingyang Ye, Shan Chen, Jingxuan Tu, Chen Liu, Zidi Xiong, Samuel Schmidgall, and Danielle S. Bitterman. Proof of time: A benchmark for evaluating scientific idea judgments. arXiv preprint arXiv:2601.07606, 2026. URL https://arxiv.org/abs/2601.07606.

[23] Christine Ye, Sihan Yuan, Suchetha Cooray, Steven Dillmann, Ian L. V. Roque, Dalya Baron, Philipp Frank, Sergio Martin-Alvarez, Nolan Koblischke, Frank J Qu, Diyi Yang, Risa Wechsler, and Ioana Ciuca. ReplicationBench: Can AI Agents Replicate Astrophysics Research Papers? arXiv preprint arXiv:2510.24591, 2025. URL https://arxiv.org/abs/2510.24591.

## A Leakage controls

## A.1 Pre-training and post-training data

Leaked test data distorts benchmark comparisons [21]. Every task comes from a paper first made public after June 2026, the latest developer-reported knowledge cutoff among the evaluated models (Table 3).

Table 3: Evaluated models and their knowledge cutoffs as published by each developer.
<table><tr><td>Model</td><td>Developer</td><td>Weights</td><td>Knowledge cutoff</td></tr><tr><td>Claude Opus 5.5</td><td>Anthropic</td><td>closed</td><td>June 2026</td></tr><tr><td>GPT-6 Astra</td><td>OpenAI</td><td>closed</td><td>30 April 2026</td></tr><tr><td>GPT-6 Luna</td><td>OpenAI</td><td>closed</td><td>18 May 2026</td></tr><tr><td>Gemini 3.8 Flash</td><td>Google</td><td>closed</td><td>March 2026</td></tr><tr><td>GLM 5.3</td><td>Z.ai</td><td>open</td><td>January 2026</td></tr><tr><td>MiMo V2.6 Pro</td><td>Xiaomi</td><td>open</td><td>November 2025</td></tr><tr><td>DeepSeek V4.1 Flash</td><td>DeepSeek</td><td>open</td><td>December 2025</td></tr><tr><td>Qwen3.8 27B</td><td>Alibaba (Qwen)</td><td>open</td><td>September 2025</td></tr></table>

We date each paper by its earliest appearance across preprint servers, journal online-first dates and print issues, which excluded 95 candidates with earlier preprints. We use real post-cutoff papers rather than instructing a model to ignore what it knows, because prompted cutoffs fail, since models still use later knowledge when it is causally relevant [3], and instructions to respect historical boundaries do not prevent recall of memorised values [8].

## A.2 In-environment leakage controls

Agents need the literature for variable definitions, codebooks and the conventions of earlier studies, so we restrict search by date rather than remove it, since unrestricted retrieval leaks answers [1, 5]. The tools apply the cutoff in each upstream query, re-check every returned date and remove forward links such as citing papers. We utilize a re-check because we found that upstream dates are unreliable (i.e., given a date of January 1st 2026 when there is missing information about the exact date and month). Each task also blocklists the seed paper’s DOI, PubMed and PubMed Central identifiers and publisher URLs, following PaperBench’s use of per-paper lists of prohibited repositories [15]. Web search is disabled, the shell reaches only allowlisted data and package hosts, and each run’s sandbox contains only its own files, with the paper’s findings stored outside it. Across 17,429 tool calls, 0 of the 8,071 records the tools returned were dated after the cutoff.

## A.3 Post-processing

After each run, an audit re-resolves every identifier the agent encountered, including those it cited from memory, against Europe PMC and Crossref, and flags any first made public after the cutoff.

## B Failure cases

Figure 4 shows detailed qualitative examples of failure cases, each quoting the agent’s structured answer and report alongside the paper’s result.

## C Predictability of post-cutoff findings

Post-cutoff findings are largely predictable from the research question alone, and the prediction does not depend on the model recognising the paper. We gave GPT-6 Sol only the neutral question for each of the 2,131 shortlisted papers, all first made public between July and September 2026, after its cutoff. Without tools, it returned a direction, a significance call, a numeric estimate, a confidence and whether it knew the finding, and we compared these with the primary claim parsed from each abstract (Table 4).

![](images/93bbb84146028e1c3f721861bcbf77322fc5de096b1fc3c55889795e49b737b0.jpg)

![](images/6730f0adcb2510ecdd5afe0b00da7c3d4eea0ee8e3a8242f0554684d9fd41252.jpg)  
Figure 4: Failure cases, quoted from the agents’ answers and reports. (a) The agent reports the pre-adjustment cost share as its answer and the value matching the paper only as a sensitivity analysis. (b) The report concludes no association, but the structured answer records the direction of the point estimate. (c) The agent obtains no data and declines to estimate.

Table 4: Blind guesses on the 2,131 shortlisted papers. Rates are over the papers for which each quantity is defined (count in grey); – marks cells with fewer than 15 papers.
<table><tr><td rowspan="2">Papers</td><td rowspan="2">n</td><td colspan="2">Direction and significance matched</td><td rowspan="2">Significant effect predicted for nulls</td><td rowspan="2">Numeric guess inside paper CI</td></tr><tr><td>non-null findings</td><td>null findings</td></tr><tr><td>All shortlisted papers</td><td>2,131</td><td>83% (1673)</td><td>16% (101)</td><td>61% (104)</td><td>52% (1313)</td></tr><tr><td>Model reports it does not know the finding</td><td>1,814</td><td>81% (1399)</td><td>17% (90)</td><td>57% (93)</td><td>51% (1090)</td></tr><tr><td>Model is unsure whether it knows it</td><td>317</td><td>95% (274)</td><td></td><td></td><td>54% (223)</td></tr><tr><td>First public July 2026</td><td>913</td><td>85% (718)</td><td>12% (41)</td><td>69% (42)</td><td>50% (578)</td></tr><tr><td>First public August 2026</td><td>698</td><td>82% (555)</td><td>11% (28)</td><td>63% (30)</td><td>54% (419)</td></tr><tr><td>First public September 2026</td><td>520</td><td>83% (400)</td><td>25% (32)</td><td>47% (32)</td><td>51% (316)</td></tr><tr><td>Data: NHANES</td><td>462</td><td>87% (418)</td><td>12% (16)</td><td>56% (16)</td><td>59% (374)</td></tr><tr><td>Data: CHARLS</td><td>201</td><td>93% (192)</td><td></td><td></td><td>55% (159)</td></tr><tr><td>Data: MIMIC</td><td>186</td><td>90% (167)</td><td></td><td></td><td>48% (155)</td></tr><tr><td>Data: SEER</td><td>93</td><td>83% (75)</td><td></td><td></td><td>60% (67)</td></tr><tr><td>Data: GWAS / MR</td><td>67</td><td>62% (58)</td><td></td><td></td><td>36% (56)</td></tr><tr><td>Data: FAERS</td><td>37</td><td>97% (29)</td><td></td><td></td><td>30% (27)</td></tr><tr><td>Data: all other sources</td><td>1,085</td><td>79% (734)</td><td>20% (51)</td><td>54% (52)</td><td>48% (475)</td></tr></table>

The model matched the direction and significance of 81% of non-null findings it reported not knowing, and of 95% of those it was unsure about. The rate is flat across the three months of first publication (82–85%). Predictability is highest for heavily mined sources such as NHANES, CHARLS, MIMIC and FAERS and lowest for Mendelian randomisation, whose instruments and outcome GWAS vary between studies. The guesses also reproduce publication bias: the model predicted a significant effect for 61% of papers whose finding was null (63 of 104) and matched only 16% of null findings.

## D Claim-level outcomes

Shown in Figure 5, access to data corrects between 69% and 84% of each model’s prior errors, while minimally reversing the claims its prior got right.

![](images/d7f5a221671d5574cc1f775c0b740ad3ea7e7380ddf0f2e925077f7dcfbe4708.jpg)  
Figure 5: What the data changes: outcome of every scored claim from the answer without data to the answer with data, per model; the number on the right is the share of prior errors the data corrected.

## E Screening prompts

We report the LLM prompts used to filter publications at scale for tasks fit for BioStudyBench. Stages 1 and 2 ran with GPT-6 Luna over every candidate record. The subsequent stages used GPT-6 Sol, over the 2,131 shortlisted papers. As shown in Figure 6, the blind guess sees only the neutral question and powers the recall screen, predicting a significant effect for 80% of papers and for 61% of those with null findings (63 of 104), so findings it recovered were demoted.

![](images/8dace8d47c85660b089a58d701413f31ea879dff487f5a371350d4bb6b15f628.jpg)

Figure 6: Recall screen. Share of shortlisted papers whose direction and significance GPT-6 Sol matched from the question alone, by the paper’s finding, and share of numeric guesses inside the paper’s 95% CI, with 95% Wilson intervals.

## E.1 Stage 1: study type

Classify the PubMed record as JSON. Did the authors generate ANY new primary data, or only analyze data that already existed? Abstract outranks accessions and publication types. Apply steps IN ORDER; the first match sets ‘primary\_data\_collection‘ (pdc) and ‘design‘.

NEW cues (authors did it): recruited, enrolled (unless "retrospective" also appears), randomized, placebo, informed consent, healthy controls, volunteers, "participants", questionnaires/interviews the authors administered, specimens collected, "we measured/sequenced/imaged", mice/cells treated, synthesized, deployed (cameras, sensors). "Prospective" is NEW only if the authors themselves enrolled participants.

EXISTING cues: retrospective(ly) anywhere in title, MeSH ("Retrospective Studies") or abstract (decisive: always step 10, never unclear), medical records, database, registry, claims, EHR, biobank, "data were obtained/downloaded from", publicly available, target trial emulation, population-based, GWAS summary statistics, bioinformatics, systematic search, bibliometric; any named cohort, survey or public dataset in ANY field, including its assays (EXISTING unless participants were recruited for THIS paper); vendor/device data recorded before the study; data the authors collected earlier under another protocol. Measurements, contacts or questionnaires made for THIS paper are NEW.

- mixed\_minor\_validation: true iff design=secondary\_analysis\_existing\_data AND pdc is new\_\* (step 5).

## # Decision steps

1. No results or no data analysis: future-tense protocol ("will be enrolled"), editorial, guideline, consensus statement (including Delphi-based), classification or framework proposal with no cohort and no N, bench-protocol chapter (Methods in Molecular Biology, Bio-protocol, "here we describe the method") that lists procedures but reports no results: other\_non\_research, pdc=none. Protocol verbs are not evidence of new data; look for an N, a measurement or a result. Future tense overrides PUBLICATION TYPES and trial accessions. Data descriptors are step 4 or 7.

2. Descriptive report of <=10 patients the authors treated, or titled "case series", no comparison or effect estimate: case\_report\_or\_series, pdc=new\_human\_data. Retrospective comparisons of any N are step 10.

3. The headline is the performance of a tool, kit, assay, device, algorithm, chatbot or rater (coverage, duplication rate, LOD, recovery, accuracy or AUC versus a reference, ICC/kappa/Bland-Altman agreement, "our approach outperforms" on benchmark data), or the subject matter is non-biomedical (finance, markets, traffic, industrial processes): methods\_or\_software regardless of any new samples or participants used as test material; pdc=new\_\* if that test material was newly acquired, else none. This step outranks 4 and 7.

4. Authors generated measurements on animals (including pets, livestock), plants, cells, chemicals, materials or environments (chemistry, field work) with functional experiments or no pre-existing dataset analyzed: lab\_or\_animal\_experiment, pdc=new\_lab\_or\_animal\_data, even without a collection verb. Human group comparisons whose measurements are bench assays on specimens obtained for this paper (biopsy IHC, serum panels, histology) are lab\_or\_animal\_experiment with pdc=new\_human\_data. A deposited GEO/SRA accession is NEW; "downloaded" accessions are existing sources.

5. Existing-dataset analysis whose ONLY new data is validation (qPCR, western blot, IHC on a few samples, no functional assay) or a small new patient cohort confirming the finding: secondary\_analysis\_existing\_data, pdc=new\_lab\_or\_animal\_data (new\_human\_data for the cohort), mixed\_minor\_validation=true.

6. Authors assigned an intervention or experimental condition to humans and report their own results: randomized, placebo, task conditions, crossover, repeated measures before/after an author-applied manipulation (anaesthetic block, load, sham, fasting, posture), n-of-1: rct\_or\_interventional, pdc= new\_human\_data. Outranks step 7 whenever the authors controlled the exposure. Prespecified substudies reporting protocol measurements are trial reports; analyses labelled secondary, post hoc or exploratory are step 10 (trial\_data\_reuse).

7. Authors recruited or followed participants for this paper, measured them, used research-only instruments (psychometric scales, neuropsychological batteries, patient-reported scales: records cannot hold these, never unclear), ran their own surveys/interviews/Delphi, or assayed stored cohort specimens: prospective\_primary\_cohort (qualitative\_or\_survey\_primary for surveys/interviews); pdc= new\_human\_data. Non-patient samples (workers, students) and cross-sectional studies of "participants" belong here even without a recruitment verb.

8. Literature synthesis (systematic search, bibliometric): with a pooled estimate (pooled OR/HR/SMD, I2, network or IPD meta-analysis) meta\_analysis\_or\_pooled (is\_secondary\_data\_study=true), else systematic\_or\_narrative\_review; pdc=none.

9. Model fed only by published parameters (Markov, cost-effectiveness), or simulation of method performance even if resampling public data: modeling\_or\_simulation. New algorithm, software, assay or scale benchmarked on public or test-bed data ("we propose [tool]"): methods\_or\_software; pdc=none ( never unclear) unless the authors acquired new test data (then new\_\*, same design). Tie-break: a model fitted to a record-level clinical, biological or population-health dataset (risk score, hub genes, clinical prediction model with AUC headline) is step 10; only a new algorithm compared against other algorithms, or a non-biomedical subject, stays methods\_or\_software.

10. Analysis of existing data only (EXISTING cue, no NEW cue): secondary\_analysis\_existing\_data, pdc=none. Also "consecutive patients treated between [past years]", routine clinical/imaging/lab data with analysis verbs, past date ranges and no consent/specimen/intervention language, even if it says " enrolled" or "cohort study" (confidence 0.7-0.85).

11. Otherwise: pdc=unclear, design=secondary\_analysis\_existing\_data, confidence<=0.5, one descriptive data\_sources entry with access=unknown. Wet-lab, animal, plant or material work never reaches this step.

## # Field rules

- is\_secondary\_data\_study: true ONLY when pdc=none AND design is secondary\_analysis\_existing\_data or meta\_analysis\_or\_pooled; otherwise false.

- data\_sources: pre-existing datasets only; never the authors’ own participants, samples or animals; empty for steps 1, 2, 4, 6, 7 and parameter-only models. REQUIRED (>=1 entry) when design is secondary\_analysis\_existing\_data or meta\_analysis\_or\_pooled; if the abstract does not name the dataset, one descriptive entry with access=unknown. Primary-result source FIRST (name it in the rationale), validation-only sources after. Type and access:

open: NHANES, BRFSS, CDC WONDER, GBD [public\_dataset\_or\_survey]; ClinicalTrials.gov [registry]; GEO, SRA, TCGA/GDC open tier, GTEx, all GWAS summary statistics (GWAS Catalog, IEU OpenGWAS) [omics\_repository ]; FAERS/openFDA [pharmacovigilance]; published literature [literature].

registration: CHARLS, HRS, ELSA, SHARE, CLSA, DHS/MICS [public\_dataset\_or\_survey]; SEER [registry]; MIMIC, eICU [ehr\_or\_claims]; BioLINCC, YODA, Vivli trial IPD [trial\_data\_reuse].

restricted: individual-level UK Biobank, All of Us, dbGaP-controlled [biobank]; Medicare, TriNetX, national claims [ehr\_or\_claims]; national or government registers of any country (Nordic, Norwegian, Danish, cancer, cause-of-death, vital statistics, device registers): [registry], never unknown, never institutional\_records; VigiBase, JADER [pharmacovigilance]; Web of Science, Scopus [literature]; trial IPD unless BioLINCC/YODA/Vivli named [trial\_data\_reuse]; named cohorts, consortium data, vendor /device data [other]; hospital, clinic or single-center chart data ONLY [institutional\_records], never unknown.

unknown: otherwise. A source on the lists above always takes its listed type and tier.

\- focus: FIRST, if design is not one of secondary\_analysis\_existing\_data, meta\_analysis\_or\_pooled, prospective\_primary\_cohort, modeling\_or\_simulation, focus MUST be other, no exceptions even when ORs or betas are reported. Otherwise: genetic\_gwas\_or\_mr; omics\_bioinformatics (DEGs, network pharmacology); pharmacovigilance; health\_services\_or\_cost; diagnostic\_accuracy (test vs reference standard); prediction\_model (scored model whose AUC/C-index is the headline); treatment\_effect\_observational (treatments/policies compared in non-randomised data or trial reanalysis, e.g. target trial emulation); descriptive\_epidemiology (prevalence, trends, no exposureoutcome estimate); association\_or\_etiology (any other exposure-outcome estimate, including " predictors" reported as ORs/HRs).

\- confidence: 0.9-1.0 explicit quoted cue; 0.7-0.85 inferred; <=0.5 unclear.

\- rationale: <=30 words quoting the deciding cue and the primary source.

\# Worked examples (cue -> pdc / design / focus / is\_secondary; sources)

\- "Multicentre cohort, N=2,148 treated 2015-2022; target trial emulation" -> none / secondary\_analysis\_existing\_data / treatment\_effect\_observational / true; [multicentre records, institutional\_records, restricted]; confidence 0.8, NOT unclear

\- "GSE18781 downloaded from GEO; hub genes validated by qPCR in patient samples" -> new\_lab\_or\_animal\_data / secondary\_analysis\_existing\_data / omics\_bioinformatics / false; mixed=true; [GSE18781, omics\_repository, open]

\- "Three WES capture kits compared on 8 DNA samples; coverage uniformity" -> new\_lab\_or\_animal\_data / methods\_or\_software / other / false; []

## E.2 Stage 2: scorable claim and obtainable data

You screen PubMed records for an environment where an agent, given a public dataset at a fixed version and a neutral question, must re-derive the paper’s headline result. Judge from the record alone; fill every field. Score only in Step 7.

## ## 1. Gate: analysis of pre-existing data?

Fails if PUBLICATION TYPES or TITLE has Review, Meta-Analysis of published studies (not pooled GWAS summary statistics or GEO series), Editorial/Letter/Comment, Case Report, Protocol; if the authors generated data (recruited/enrolled/randomized participants, questionnaires, animal/cell/in-vitro work, assays, imaging or sequencing for this study; bioinformatics screens whose headline is wet-lab validation; dosimetric or treatment-planning studies where the authors generated new plans or simulations); if the headline is the performance of an AI model, LLM, software tool or device evaluated on the data (a tool evaluation, not a data analysis); or no empirical dataset. DATABANK ACCESSIONS: NCT/DRKS/ChiCTR = the authors’ own trial; GSE/PRJNA/figshare/Dryad usually = data they DEPOSITED unless the abstract says it was downloaded.

(A) Gate FAILS: use ALL defaults: booleans false, data\_publicly\_obtainable "no", type "other", direction " na", strings "" (except concerns), confidence >= 0.9.

(B) Gate PASSES but data is "no" (retrospective review of hospital, registry, EHR, imaging or genetic-test records): fill primary\_claim, data\_sources and data\_version\_stated normally; single\_primary\_question follows Step 5; score 1.

Bibliometric/publication-count studies: gate passes (PubMed is public) but single\_primary\_question=false ( multi-aim descriptive), score 2.

## ## 2. Headline claim -> primary\_claim

First rule that applies: (a) the outcome labelled "primary" (several: the first listed); (b) the exposureoutcome pair in the TITLE (first-named exposure; bidirectional MR: forward direction); (c) the first exposure-outcome result in RESULTS, skipping sample descriptors (N, mean age, % with a trait). ONE exposure only, the first named: for "predictors of / risk factors for X" papers take the first predictor and its own sign; never concatenate predictors. Never a subgroup, secondary, sensitivity or crude result, and never by significance. "Prevalence and risk factors of X": the prevalence. Shape plus estimate: the estimate is the claim; shape goes in concerns. If text and numbers disagree, follow the numbers. When the headline comes from rule (c), confidence <= 0.6 (hard cap).

type: prevalence/rate/APC -> estimate; any AUC, including a pre-existing score (SOFA) validated in public data -> auc. value\_text: numbers verbatim with CI/p, adjusted over crude; set: selection criterion plus members or their count; "" if none. direction = sign of the association: positive (OR/HR/RR>1), negative (HR<1, even for treatment vs control), null (CI includes null), mixed (ONE exposure whose sign reverses by subgroup or timepoint, never several predictors with different signs), na (auc, set, rank, count, categorical, other).

## ## 3. answer\_objective, null\_or\_negative\_result

answer\_objective=true only if value\_text has a number or an enumerated set ("significantly higher" alone is not). Borderline = 0.01<=p<0.05 or CI bound within 0.05 of null; write "borderline" in robustness\_signals.

null\_or\_negative\_result=true ONLY if direction is null, or the conclusion states the exposure did NOT have the expected or hypothesised effect. A significant protective or beneficial effect (HR/OR<1) is direction=negative with null\_or\_negative\_result=false. Never lowers the score.

## ## 4. Data

data\_sources: names as written with years/version. Grade the WORST source the primary claim needs (public discovery + private validation: grade discovery; public data merged with data the authors compiled: " no").

yes: NHANES, NHIS, BRFSS, MEPS, CDC WONDER, GBD, FAERS/openFDA, GEO, TCGA/GDC open, GTEx, CGGA, cBioPortal, GWAS Catalog, IEU OpenGWAS, any published GWAS summary statistics, ClinicalTrials.gov records, PubMed/OpenAlex metadata. MR: grade the summary statistics, not the cohort (UK Biobank summary statistics = yes).

registration: SEER (always "registration", never "no"), MIMIC, eICU, HRS, CHARLS, ELSA, SHARE, BioLINCC.

no: UK Biobank, All of Us, dbGaP controlled, national registers/insurance (Nordic, Korea, Taiwan, CPRD, JAGES), US claims/registries (Medicare/CMS, HCUP NIS/NRD/KID, NCDB, NSQIP, NTDB), commercial (Optum, TriNetX), consortium or multicentre cohort data (CENTER-TBI, IBBC, PIERS, EXCEL), QI or registry initiatives (MAQI2), national HMIS/DHIS2 extracts, research-group cohorts, trial data without a named repository (an NCT number is not one), "available on reasonable request", ANY hospital/medicalrecords database, any UNNAMED source ("a nationwide database", city-level data the authors compiled), except unnamed GWAS summary statistics.

unknown: a named repository or portal that LOOKS public but is not on the yes list (Rembrandt/GlioVis), or GWAS summary statistics from unnamed studies.

data\_version\_stated=true only if a cycle, year range, release or accession is given ("MIMIC-IV" alone = false).

## ## 5. single\_primary\_question, question\_neutral

false (forces score 2) if ANY: many exposures or outcomes with no labelled primary; multi-aim descriptive paper ("burden of X", incidence AND survival AND disparities, "characterize reporting patterns") - never carve out one rate as the primary; >=3 independent analytic arms (MR + colocalisation + pharmacovigilance); the deliverable is a model, risk score, prognostic signature, hub-gene/PPI pipeline, database/web platform, software or method. Bidirectional MR (forward direction), " prevalence and risk factors" designs (primary = the prevalence), secondary outcomes, subgroups and sensitivity analyses do NOT break singleness. Omics set claims: a screen of a named gene family or a DEG list on TCGA/GEO/CGGA whose result is an enumerated set of genes with a stated criterion ( direction of survival association, FDR<0.05) is ONE question: type=set, value\_text=criterion plus members, answer\_objective=true; the hub-gene/PPI exclusion applies only when the deliverable is a network, signature score or wet-lab validation.

question\_neutral: "" when single\_primary\_question is false or the score is 1; never write "Not applicable". Otherwise sentence 1 = population, dataset + version, exposure, outcome, main adjustment; sentence 2 = "Report ..." with exactly ONE estimand and its format: "Report the adjusted OR with 95% CI and state whether an association is present". Never "for each outcome" or "each measure" (that means single\_primary\_question is false). Never include result numbers, direction words (increase, reduce, protective, risk factor), the significant subgroup or feature, or the title.

## ## 6. analysis\_complexity

Rate ONLY the primary claim’s analysis; ignore auxiliary ML/SHAP, splines, nomograms, mechanistic add-ons and a source’s own modelling. low: regression, survival, propensity/IPTW, joinpoint/APC, disproportionality, two-sample IVW MR, differential expression. medium: DiD/target-trial emulation, data-derived thresholds, MR with colocalisation, ML with cross-validation. high: raw read alignment/ variant calling, GWAS from genotypes, deep learning on images/waveforms, proprietary software.

\## 7. environment\_buildability: the FIRST line that applies

1: gate failed, or data "no", or answer\_objective false.

2: single\_primary\_question false (even with open data), or complexity high.

3: data "unknown"; or estimate with neither CI nor p; or set with no stated criterion; or ABSTRACT empty or truncated mid-sentence (then confidence<=0.3).

4: data "registration"; or complexity medium; or borderline; or data\_version\_stated false; or N<1000 and not nationally representative.

## 5: otherwise.

Nothing else changes it. If answer\_objective is false or data is "no" the score is 1 regardless of anything below. verdict is a pure function of the score: 5-4 strong, 3 possible, 2 weak, 1 no.

## ## 8. Remaining fields

robustness\_signals (<=30 words): N, p/CI, sensitivity analyses, replication, "borderline"; "none stated" if absent. concerns (<=40 words): never empty when verdict != strong; start with the step that failed ("Step 1: authors enrolled participants", "Step 4: single-centre EHR, not public"). Cite only rules written in this document; if a study type is not covered, say so and apply the best-fit step. confidence: >=0.9 only when every step had an explicit cue; NEVER above 0.6 when data is "unknown" or the headline came from rule (c).

## ## Examples

"NHANES 2009-2014, BRI vs periodontitis, OR 1.053 (1.016-1.092), N=10,790" -> yes, estimate, positive, low, 5 strong. "MIMIC-IV, PIV vs 60-day mortality, HR 1.20 (1.01-1.44) above a U-shape inflection" -> registration, estimate, positive, borderline, 4 strong. "SEER Texas endometrial cancer incidence by age, race, histology plus survival" -> single\_primary\_question=false, 2 weak. "Bidirectional MR on unnamed GWAS summary statistics, forward OR 0.99 (0.97-1.01)" -> unknown, null, null\_or\_negative\_result=true, 3 possible.

## E.3 Recall screen: blind guess from the question alone

You are a senior clinical epidemiologist with excellent recall of the biomedical literature up to your knowledge cutoff. You will be shown ONLY a research question (and the exposure, outcome and population it concerns), with no data and no paper. Answer as if a colleague asked you to bet on the result before the analysis is run.

Give: the direction you expect (positive = higher exposure goes with higher outcome/risk; negative = lower; null = no association; na for counts, prevalences, sets or AUCs where direction does not apply), whether you expect it to be statistically significant at 0.05 in a study of typical size for that dataset, your single best numeric guess in the units the question implies, your probability that direction AND significance are right, and whether you recognise this specific finding from literature you have seen (yes / unsure / no). For prevalences, rates, AUCs and counts, the numeric guess is the whole answer: be as precise as you can.

Do not hedge into "null" merely because you are unsure; give the answer you would actually bet on. Base it on what you know about the exposure-outcome pair, the population, and the typical sample size and analysis for the named dataset.

## E.4 Review classifier

You are the lead designer of a benchmark of controlled data-analysis environments. In each environment an AI agent receives (a) the exact public dataset file(s) a paper used, at a fixed version, and (b) a neutral research question that does not reveal the result. The agent must analyze the data and report the headline estimate. It is scored on reproducing the paper’s DIRECTION and SIGNIFICANCE (and, more loosely, whether its estimate overlaps the paper’s confidence interval). A separate "no-data" run checks whether a model can state the answer from memory; items it passes are discarded as leaked.

You read a PubMed record plus fallible screener output, a deterministic arithmetic check of the reported estimate, and a BLIND GUESS made by a strong model that saw only the neutral question. Decide whether this paper would make a good controlled environment. Think like an experimentalist: the environment is only useful if the answer is determined by the data plus a conventional analysis, is not guessable from prior knowledge, and can be scored against a precise number. Judge the estimand, not the abstract’s writing style. Distinguish two kinds of things you cannot see in an abstract:

\- ROUTINE checks that every environment build performs anyway and that do NOT lower the verdict: the exact cycle/submission/release of a source that is versioned by nature (NHANES cycles, SEER submission year, MIMIC-IV version, FinnGen release, GBD round) once the source is named; the full covariate list ; the CI when direction and p are given; the cause-of-death or code-selection rule; software versions ; weighting details. List them in needs\_full\_text and move on.

\- DECISIVE unknowns that could flip the verdict and therefore cap it at maybe: whether the data tier is actually obtainable; whether a needed variable exists in the public release; whether the exposure is a native variable or a construct fitted in the paper; whether the reported number is an adjusted main effect or a pooled/marginal/subgroup quantity; whether the sample is large enough to support the claim. List these in unknowns.

Only OBSERVED failures produce bad.

## # Gates, in order

0. DATE. If public\_before\_cutoff is true or earliest\_public\_date < 2026-07-01: date\_eligible=false, verdict bad, stop. If earliest\_public\_date is exactly a January 1st, treat it as a placeholder: cap at maybe, unknowns += "date placeholder".

1. SCOPE. Human health, biomedicine, epidemiology, health services, pharmacovigilance, clinical genomics. Physics, ecology, geoscience, materials, economics without a health outcome, bibliometrics about publishing: in\_scope\_biomedical=false, verdict bad.

5 open, versioned, downloadable: NHANES/NHIS/BRFSS/MEPS/NSDUH/YRBS/NSCH cycles; CDC WONDER (ICD codes, years, standard population); NHANES public-use Linked Mortality Files; FAERS/VAERS/MAUDE/JADER quarterly files; GEO GSE; TCGA/GDC project + release; GTEx release; GWAS Catalog GCST id, FinnGen release number, IEU OpenGWAS id; ClinicalTrials.gov/AACT snapshot with date; GBD results tool with release; open non-US equivalents (Canadian CCHS PUMF, ENSANUT, Brazil PNS/DATASUS, Peru ENDES, Japan e-Stat/NDB open, KOSIS, Eurostat, WHO Mortality DB).

4 free registration, standard, versioned: SEER research data (submission year), MIMIC-IV vX.Y, eICU v2 .0, KNHANES (KDCA portal), MICS, HRS/ELSA/SHARE/CLSA/CHARLS/CLHLS with waves stated, UK HSE/UKHLS via UKDS.

3 named source but cycle/version/accession not in the abstract; or DHS (application, usually approved in days); or PRAMS (proposal). Cap at maybe; needs\_full\_text += "version/cycles".

2 or less, bad: restricted or licensed: SEER-Medicare, UK Biobank/All of Us individual-level, dbGaP controlled, VigiBase, TriNetX/Optum/MarketScan/national claims, NHIRD, CPRD, ELSA-Brasil, HILDA, HRS sensitive/DNAm; unnamed sources ("a nationwide database", "publicly available GWAS" with no id); live portals without a snapshot (NCBI Pathogen Detection, IMG/M, "all GEO series matching a query", FishBase); data the authors compiled or scraped themselves; "available on request".

Variables must exist in the mountable release: public SEER has months not days to treatment and no county below state; NHANES has no geocodes; BRFSS no county; MIMIC free text only in the note module. If the estimand needs a missing field: bad, design\_flags += variable\_not\_in\_public\_release.

Multi-source: record-level linkage, geocoding of individuals, or harmonising individual-level variables across two or more named cohorts = bad (multi\_source\_linkage). Joining published aggregate tables on

a public key (county FIPS, country-year) from open versioned sources = cap maybe ( aggregate\_table\_join). MR grades the summary statistics, not the underlying cohort (UK Biobank summary statistics are open).

3. ESTIMAND DEFINABILITY (controllability). Would two competent analysts with the same file converge on direction and significance? Allowed exposures/outcomes: native codebook variables; standard code lists (ICD, MedDRA PT or SOC, SEER site/histology); closed-form published formulas used continuously or per SD (BMI, BRI, TyG, TyG-BMI, AIP, SHR, UHR, METS-VF, NLR, SII, RAR, eGFR CKD-EPI, EASIX, SHARE-FI, Langa-Weir, KDIGO stages); a gene symbol on a named GSE/TCGA project; a drug name and event PT in FAERS.

Medium (controllability 3, note as composite\_index): literature-defined composite scores with implementation choices: DII/E-DII, CDAI, OBS, HEI-2015, LE8/LC9, MIND/Mediterranean scores, frailty indices, weighted quantile sum mixtures.

Low (controllability 2 or less, usually bad): exposure or outcome produced by a model fitted in the same paper (trajectory groups, latent classes, k-means or SOM clusters, NLP labels, PCA/factor scores ); cut-points that are data-driven, X-tile, "optimal", or unstated ("high vs low expression", " elevated"); undefined ICU phenotypes (sepsis without Sepsis-3 mention, delirium from chart strings, procedure cohorts by string matching, first/peak lab in an unstated window); outcome composites the authors defined (undefined\_icu\_phenotype or unstated\_or\_datadriven\_cutpoint).

Estimators allowed: design-weighted prevalence/proportion, logistic/linear/Poisson/modified Poisson/ negative binomial/Cox/Fine-Gray, GLM gamma or two-part cost models, direct age standardisation, joinpoint/AAPC, disproportionality ROR/PRR/IC on 2x2 tables, IVW two-sample MR with default clumping, limma/DESeq2 differential expression, Kaplan-Meier differences, adjusted predicted-prevalence contrasts, external validation of a frozen published score (PREVENT, PCE, SCORE2, APACHE, SOFA) reported as AUC or calibration (external\_validation\_frozen\_score, claim\_kind auc).

Estimators that make the headline bad (causal\_machinery, psm\_or\_matching\_only, ml\_headline): mediation/ PROCESS/SEM direct or indirect effects, instrumental variables, staggered DiD, parametric g-formula, age-period-cohort, spatial Durbin, Oaxaca-Blinder, multi-group ITS, clone-censor-weight emulation, ML /SHAP/nomogram performance, a matched or PSM-dependent estimate as the ONLY headline. A simple twogroup pre/post or single-series ITS null with the design fully stated may be maybe.

Headline must be the main effect in the overall population: an interaction, RERI, sex/race-stratified subgroup, one cell of a multi-arm result, "mixed" direction, or the top hit of a screen (many exposures/outcomes/taxa/proteins) is bad unless the screen hit is study-wide significant and the sources are pinned (interaction\_or\_subgroup\_headline, screen\_winner).

A proportion pooled across several surveys or countries by meta-analysis is not a weighted prevalence: controllability 3 at most (pooled\_across\_surveys). Pooling NHIS across the 2019 redesign, BRFSS across 2011, NSDUH across 2022 needs a stated harmonisation (survey\_redesign\_pooling).

4. ROBUSTNESS. Use the ARITHMETIC CHECK; do not recompute. robust\_margin true, or p <= 0.005, or a powered null (ratio CI within about 0.7-1.4 with analytic N >= 1,000 and >= 100 exposed/events): robustness 4-5. borderline (implied 1.96 <= z < 2.8, or 0.01 <= p < 0.05, or a CI bound within 0.05 of null on the log scale): robustness 2, cap maybe, bad if N < 5,000 or the abstract mentions attenuation in sensitivity analyses (borderline\_significance). A null from a tiny or selected sample is a power artefact: robustness 2, cap maybe (underpowered\_null). point\_outside\_ci, asymmetric\_ci, p\_ci\_mismatch, or two different Ns in the abstract: bad (arithmetic\_inconsistent). If the abstract gives direction and significance for an allowed estimand but no CI, that is not a failure of the estimand: cap maybe with needs\_full\_text += "CI". If no number exists anywhere and none could exist, bad. Analytic N < 1,000 or < 100 events for regression items: robustness at most 3.

5. RECALL RISK. Two separate tests; take the worse. The BLIND GUESS is your main evidence; the scoring rule is: Track A passes on direction + significance, with CI overlap as the finer score.

HARD RULE: if the BLIND GUESS direction AND significance both match the paper’s primary claim (for a directional claim), recall\_risk is at least 4 and the verdict is at most maybe. No exception for "the magnitude is informative": the benchmark’s no-data baseline would already pass on direction + significance. Use this rule before any other recall reasoning.

recall\_risk 5: the blind guess matches the paper’s direction and significance AND its point guess lies inside the paper’s CI (or the claim has no numeric target beyond direction), or known\_finding = yes. The no-data baseline would pass: cap at maybe.

recall\_risk 4: the blind guess matches direction and significance but its point guess is outside a narrow CI; or a textbook pair (adiposity and diabetes/ED/asthma, frailty and mortality, smoking and menopause, biomarker ratio and ICU mortality, gene X and prognosis in TCGA, boxed-warning drug and its labelled event, surgery or radiotherapy versus none in SEER, education and fertility; design\_flags += textbook\_direction) even when the blind guess happened to miss. Verdict at most maybe (replication bucket).

recall\_risk 3: direction plausible but the blind guess got significance wrong, or its numeric guess is far outside the CI, or the estimand is a stratum- or period-specific number.

recall\_risk 1-2: the blind guess is wrong, or says an effect exists where the paper found a null, or the pair is genuinely odd.

Numeric test for prevalences, rates, counts, AUCs and ROR magnitudes: compare the BLIND GUESS point\_guess with the paper’s CI: inside means recall\_risk >= 4. A wide CI is easier to hit, so wide-CI descriptive numbers are MORE recall-prone. Figures printed in the data producer’s own pre-cutoff report (DHS final report, MICS report, SAMHSA tables, NCHS data briefs, KDCA reports) are recallable: recall\_risk >= 4.

Also: if the Background cites a prior study with the same exposure-outcome direction ("prior research using ... found", "we extend"), design\_flags += replication\_of\_cited\_prior and recall\_risk >= 4. If question\_neutral contains a data-derived class name, an analyst-chosen cut-point, or a directional adjective ("delay", "deficiency", "elevated") applied to the exposure, design\_flags += question\_leaks\_result (the question would have to be rewritten; note it, do not reject for it).

- NHANES 2005-2018, DII and LE8 < 80, OR 1.39 (1.30-1.50): composite exposure AND composite outcome, textbook direction: maybe (composite\_index, textbook\_direction).

Low recall risk (1-2): null or counter-prior findings, odd exposure-outcome pairs, stratum-specific numeric estimands (HCV RNA prevalence among MSM in NHANES 2011-2014), a specific drug-event ROR, a specific site-period incidence trend, a differential-expression set for one GSE.   
recall\_risk 5 means the environment measures memory: cap at maybe (replication bucket).

6. SCORABILITY AND BETTER PRIMARIES. The neutral question must name one estimand; "for each outcome" or several arms means not single. List in your head every scorable claim in the abstract (weighted prevalences with CI, main effects, nulls). If the screener’s primary is a poor choice but the abstract contains a better one (a native-variable main effect, a powered null, a stratum-specific numeric estimand), write it in recommended\_primary with its value; otherwise write "screener primary". Only promote a null that meets the powered-null bar.

## # Verdict

good = "build it": data\_feasibility >= 4 (or 3 only because the version is a routine check), controllability >= 4, robustness >= 4 (or a powered null), recall\_risk <= 3, the recommended primary is scorable, and no DECISIVE unknown. Routine needs\_full\_text items never block good.

bad if any gate fails on OBSERVED evidence: date, scope, data\_feasibility <= 2, controllability <= 2, arithmetic\_inconsistent, headline is a subgroup/interaction/screen winner, no scorable number possible.

maybe otherwise: name the single decisive check first in reasoning. A maybe must always carry at least one decisive unknown, a design flag, recall\_risk >= 4, or robustness <= 3; if none applies, the verdict is good.

Family priors from a 320-item human review (stratified, not shortlist rates): registry mortality trends via WONDER 3/4 good; single-survey weighted prevalence of a native variable 7/17; FAERS ROR with quarters 4/15; SEER code-defined cohorts 4/16; single-gene TCGA Cox with continuous exposure 4/26; survey logistic regression 8/55 (mostly textbook pairs); pinned two-sample MR 2/39 (unnamed GWAS and borderline p); ICU cohorts 1/24 (undefined phenotypes, ratio-to-mortality genre); ageing panels 0/29 (home-made indices, waves unstated); latent-exposure or causal-machinery designs 1/48. Do not apply these as quotas; use them to know where to look for the failure.

Prefer variety for the benchmark: nulls, odd pairs, numeric estimands, and non-US families alongside standard survey regressions. Reward clean reproducible measurement, not clinical importance.

## # Calibration anchors

- NHANES 2011-2014, MSM aged 20-59, HCV RNA prevalence 3.5% (95% CI 1.9-5.7): open, native variables, weighted proportion, stratum-specific number nobody recalls: good (5/5/5, recall 1).

- SEER 2000-2023 mucinous colorectal cancer, left vs right sidedness, cancer-specific mortality HR 1.33 (1.28-1.39), N=40k, covariates listed, counter-prior direction: good; the submission year and AJCC harmonisation are routine needs\_full\_text, not blockers.

- CDC WONDER 1999-2024, ICD-defined VTE deaths in adults with cancer, AAMR near doubling, p<0.001: good; the underlying-vs-any-position rule and the AAPC CI are routine full-text items.

- FAERS quarters stated, VOWST and death ROR 0.29 (0.15-0.53), counter-intuitive: good; reference-group and product-name matching rules are routine.

- FinnGen R12 + GCST90205183, IVW MR, OR 0.99 (0.97-1.01) null with 248 instruments: pinned, standard, powered null: good.

- CDC WONDER 1999-2023, ICD J69.0 underlying with I60-I69 contributing, age 25+, AAPC with CI: good.

- MIMIC-IV, AIP quartiles and in-hospital mortality in sepsis, univariable OR only: bad (undefined phenotype, no adjusted estimate, ratio-to-mortality genre).

- CHARLS 2011-2020 staggered DiD on clean heating, "3.3%" with no CI: bad (causal\_machinery, no CI). - VigiBase, any estimand: bad (restricted).

- 2,260 ICH patients, LCMM glucose trajectory group vs 28-day mortality HR 1.71: bad ( latent\_or\_fitted\_exposure).

- NSDUH 2022-2023 pooled, prescription opioid use and medical cannabis use, aOR 1.76 (1.53-2.03), native items, post-redesign years: maybe (textbook\_direction; precise magnitude is the only non-recallable part).

Output every field of the schema. reasoning <= 80 words.

## F Tasks

Table 5: The 25 benchmark items of window v2026Q3. First public = earliest of preprint, online-first or print date. Kind: D = directional (direction and significance), N = numeric estimate or count, T = trend.
<table><tr><td>PMID</td><td>Paper title</td><td>Dataset</td><td>Access</td><td>First public</td><td>Kind / cond.</td></tr><tr><td>42505895</td><td>Predicted Excess Cardiovascular Age ENDES Peru and a Reverse Socioeconomic Gradient in a Middle-Income Latin American Country: A</td><td></td><td>open</td><td>2026-07-09</td><td>D/A</td></tr><tr><td>PMID Paper title</td><td></td><td>Dataset</td><td>Access</td><td>First public</td><td>Kind / cond.</td></tr><tr><td>42681608</td><td>Racial discrimination in Latin American school settings: evidence from Brazilian adolescents.</td><td>PeNSE - Pesquisa Nacional de Saude do Escolar 20</td><td>open</td><td>2026-09-01</td><td>N/A</td></tr><tr><td>42738955</td><td>Vegetable and Fruit Consumption Among Adolescents in the Gulf Cooperation Council Countries: A Pooled Analysis of National School</td><td>WHO Global School-based Student Health Survey</td><td>open</td><td>2026-08-26</td><td>N/A</td></tr><tr><td>42788097</td><td>Surveys Prevalence of night work in the Brazilian employed population. Autoimmune hypothyroidism and</td><td>PNS - Pesquisa Nacional de Saude</td><td>open</td><td>2026-09-22</td><td>N/A</td></tr><tr><td>42621321</td><td>endometriosis: evidence from bidirectional Mendelian randomization and an independent</td><td>GWAS summary statistics: FinnGen R12-UK Biobank</td><td>open</td><td>2026-08-04</td><td>D/A</td></tr><tr><td>42700087</td><td>clinical cohort analysis Alzheimer disease and lichen planus: A Mendelian randomization analysis. SPAG1 Expression as a Candidate</td><td>GWAS summary statistics</td><td>open</td><td>2026-09-01</td><td>D/A</td></tr><tr><td>42650068</td><td>Predictor of Pathological Lymph Node Metastasis in Prostate Cancer: A Transcriptomic Analysis of The Cancer Genome Atlas Prostate Adenocarcinoma Cohort</td><td>TCGA-PRAD</td><td>open</td><td>2026-07-28</td><td>D/A</td></tr><tr><td>42426766</td><td>Who measures the measurers: Closing CMS HCRIS the administrative accountability gap in non-profit acute care hospitals.</td><td></td><td>open</td><td>2026-07-09</td><td>N/A</td></tr><tr><td>42443048</td><td>Substance use among fatally-injured pedestrians in New York State, USA: Reporting System a retrospective analysis.</td><td>NHTSA Fatality Analysis</td><td>open</td><td>2026-07-13</td><td>N/A</td></tr><tr><td>42547587</td><td>Real-world safety profile of fezolinetant: a disproportionality analysis based on the FAERS database</td><td>FAERS</td><td>open</td><td>2026-08-03</td><td>D/A</td></tr><tr><td>42675671</td><td>A retrospective study of disproportionality analysis signals between proprotein convertase subtilisin/kexin 9 inhibitor use and respiratory, thoracic, and mediastinal</td><td>FAERS</td><td>open</td><td>2026-08-28</td><td>D/A</td></tr><tr><td>42675701</td><td>disorders. Safety analysis of acyclovir use in neonates based on the FDA Adverse Event Reporting System: A retrospective observational</td><td>FAERS</td><td>open</td><td>2026-08-28</td><td>D/A</td></tr><tr><td>42401752</td><td>pharmacovigilance study Patient out-of-pocket costs for antihypertensive single pill combination products with cost caps</td><td>Medical Expenditure Panel Survey, Household Comp</td><td>open</td><td>2026-07-04</td><td>N/A</td></tr><tr><td>42415254</td><td>Treatment Patterns and Barriers to Care Among U.S. Adults With Co-Occurring Substance Use Disorder and Mental Illness</td><td>NSDUH</td><td>open</td><td>2026-07-08</td><td>N/A</td></tr><tr><td>42432888</td><td>Cross-sectional association between relative fat mass and abdominal aortic calcification among US adults: The</td><td>NHANES</td><td>open</td><td>2026-07-01</td><td>D/A</td></tr><tr><td>42560253</td><td>NHANES 2013 to 2014 Screening for Advanced Liver Fibrosis: Comparing Serum Indices With Elastography in a Nationally</td><td>NHANES</td><td>open</td><td>2026-08-01</td><td>N/A</td></tr><tr><td>42567448</td><td>Representative U.S. Cohort A cross-sectional evaluation of racial NHANES and ethnic differences in apparent treatment resistant hypertension</td><td></td><td>open</td><td>2026-08-07</td><td>N/A</td></tr><tr><td>42578523</td><td>among US adults Prevalence of Moderate to Severe Joint Pain Among Adults With Diagnosed Arthritis: United States,</td><td>NHIS</td><td>open</td><td>2026-08-11</td><td>N/A</td></tr><tr><td>42606884</td><td>2023 Serologic Immunity to Hepatitis A and B in US Adults and High-Risk</td><td>NHANES</td><td>open</td><td>2026-08-17</td><td>N/A</td></tr><tr><td>42606983</td><td>Populations Prevalence of Urinary Incontinence in NHANES the United States 2015-2020</td><td></td><td>open</td><td>2026-08-17</td><td>N/A</td></tr><tr><td>42649615</td><td>Prevalence and risk factors of current hepatitis C infection among men who have sex with men aged 20-59 years:</td><td>NHANES</td><td>open</td><td>2026-08-27</td><td>N/A</td></tr><tr><td>42731386</td><td>NHANES 1999-2016 Association between bisphenol A exposure, methylmalonic acid, and cardiovascular disease among adults aged 60 and younger</td><td>NHANES</td><td>open</td><td>2026-09-12</td><td>D/A</td></tr><tr><td>PMID</td><td>Paper title</td><td>Dataset</td><td>Access</td><td>First public</td><td>Kind / cond.</td></tr><tr><td>42736760</td><td>Serum magnesium and depressive symptoms among U.S. adults: A cross-sectional study of NHANES</td><td>NHANES</td><td>open</td><td>2026-09-01</td><td>D/A</td></tr><tr><td>42560347</td><td>august 2021-august 2023 A Two-Decade Analysis of Post-Procedural Respiratory Disorder Cause of Death Mortality Among Older Adults in the</td><td>CDC WONDER Multiple</td><td>open</td><td>2026-08-06</td><td>D/A</td></tr><tr><td>42740601</td><td>United States Trends and Disparities in Aspiration Pneumonitis Mortality With Co-Listed Cause of Death Stroke Among US Adults, 1999-2024: A CDC WONDER Analysis</td><td>CDC WONDER Multiple</td><td>open</td><td>2026-09-01</td><td>D/A</td></tr></table>