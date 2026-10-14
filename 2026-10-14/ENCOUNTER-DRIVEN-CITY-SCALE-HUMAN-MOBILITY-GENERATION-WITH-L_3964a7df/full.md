# ENCOUNTER-DRIVEN CITY-SCALE HUMAN MOBILITY GENERATION WITH LARGE LANGUAGE MODELS

Chao Li<sup>1</sup>, Renhe Jiang<sup>2</sup>, Chuan Xiao<sup>3</sup>, Jiawei Wang<sup>1</sup>

<sup>1</sup>The Hong Kong University of Science and Technology

<sup>2</sup>The University of Tokyo, <sup>3</sup>The University of Osaka

chaoli@hkust-gz.edu.cn, jiangrh@csis.u-tokyo.ac.jp chuanx@ist.osaka-u.ac.jp, jiaweiw@hkust-gz.edu.cn

## ABSTRACT

Large language models (LLMs) have shown strong potential for human mobility generation by modeling individual routines, activities, and mobility preferences. However, existing approaches remain largely resident-centric: generating realistic individual trajectories does not necessarily recover the encounters that emerge when multiple residents co-occur at the same place and time. To this end, we study encounter-aware human mobility generation, which jointly preserves individual mobility and population-level interaction structure. We propose ENCOUN-TERMOB, an encounter-driven LLM framework consisting of a coarse-to-fine Encounter Generator and an Iterative Coupling Framework. The Encounter Generator first captures city-level encounter rhythms and then instantiates participant– time–venue encounters, while iterative coordinates these encounters with personal mobility behavior through successive refinement. We further introduce an encounter recovery protocol covering encounter marginal and joint distributions over residents, times, and venues. Experiments on the Japan and NYC datasets show that ENCOUNTERMOB outperforms all external baselines on all seven encounter metrics and all six personal mobility metrics. Compared with ELLMob, it reduces the participant–time–venue joint error by 40.4%/41.7% and the mean personal mobility error by 58.8%/32.7% on Japan/NYC, respectively.

## 1 INTRODUCTION

City-scale human mobility data capture the spatiotemporal movements of residents across cities and support the study of individual mobility patterns and urban dynamics, with established applications in transportation planning (Çolak et al., 2016), public-health decision-making (Chang et al., 2021), and urban planning and management (Zheng et al., 2014). However, access to real-world mobility data are often constrained by collection costs, limited coverage, commercial restrictions, and privacy risks (Kostandova et al., 2025; Zhu et al., 2023a). Human mobility generation therefore aims to synthesize spatiotemporal trajectories that reproduce real-world mobility patterns, alleviating data scarcity and supporting downstream mobility analysis and urban applications(Yuan et al., 2026).

Recent LLM-based mobility generation methods have substantially improved individual trajectory synthesis by leveraging rich behavioral information, including historical routines(Wang et al., 2024), activity semantics(Gong et al., 2024), mobility motivations(Ju et al., 2025), and contextual factors (Wang et al., 2026). Beyond reproducing spatiotemporal regularities, these methods use LLMs to interpret the behavioral semantics underlying individual mobility, such as what activities residents perform, what places they prefer, and how their mobility adapts to different contexts. Such semantic reasoning enables generated trajectories to better reflect not only where and when each resident moves, but also the behavioral patterns, preferences, and motivations that shape their movement.

Despite their remarkable success, existing methods primarily adopt a resident-centric perspective, generating individual trajectories according to personal mobility behavioral. However, mobility data also capture cross-resident relations, including encounters, i.e., the co-presence of two or more residents at the same place and time (Sun et al., 2013). Preserving such encounters requires coordination across multiple residents, since the activity of one resident must align with the timing,

![](images/cda04da5d8079f6f1e7cb15d5440303461cfb9d4e6ecd7ac8eb09613c3a37a9e.jpg)

Figure 1: Individual mobility generation versus encounter-driven mobility generation. Colored paths represent the trajectories of different residents, and highlighted locations indicate encounters across trajectories. (a) Individual mobility generation models each resident according to personal mobility patterns. (b) Encounter-driven mobility generation additionally coordinates encounter participants, times, and locations across the trajectories of multiple residents within the generated mobility data.

location, and activity context of others. Recent household mobility work uses LLMs to reason about activity dependencies among household members and coordinate their mobility accordingly (Liu et al., 2025). However, in existing city-scale mobility generation, encounters are typically induced by spatiotemporal overlaps among independently generated trajectories. Since encounters reflect the joint behavior of multiple residents, resident-centric generation may be insufficient to capture the collective interactions involved in such events (Yuan et al., 2025). As a result, the ability of existing methods to recover encounter structures remains questionable. As illustrated in Figure 1(a), independently generated trajectories may produce mismatches in encounter participants, times, and venues, leading to inaccurate city-wide encounter distributions across these dimensions.

Therefore, we propose ENCOUNTERMOB, an encounter-driven approach to human mobility generation that coordinates resident trajectories to reproduce encounter structures in the generated mobility data, as illustrated in Figure 1(b). This formulation introduces two key challenges. First, the encounter structures used for coordination need to be established before generating individual trajectories. We introduce a coarse-to-fine encounter generator that first captures the temporal and category-level distribution of encounters and then instantiates concrete participant–time–venue events. Second, these encounter structures need to be coupled with the mobility patterns of individual residents. We therefore introduce an iterative coupling framework, where a joint critic evaluates their consistency and guides the refinement of both encounter structures and personal trajectories. Together, these components enable coordinated trajectory generation while preserving plausible personal behavior. Besides, we introduce an encounter-aware evaluation protocol that assesses encounter recovery across participants, times, and venues, together with their joint structures.

Experiments on the Japan and NYC datasets demonstrate consistent improvements in both encounter and personal mobility fidelity. Compared with ELLMob, ENCOUNTERMOB achieves lower scores on all seven encounter metrics, reducing the joint encounter score by 40.4%/41.7%(Japan/NYC), and on all six personal mobility metrics, with mean reductions of 58.8%/32.7%, respectively. Ablation studies show that removing either city-level rhythms or Encounter Instantiation increases the full group–time–venue JSD by 33.6% on average across the two ablations and both datasets. Removing iterative coupling increases this score by 74.4% on average across the two datasets, supporting the contributions of encounter generation and iterative refinement to encounter recovery. Our contributions are summarized as follows:

1. To the best of our knowledge, we present the first city-scale human mobility generation study aimed at recovering the encounter structures observed in real mobility data.

2. We propose ENCOUNTERMOB, an encounter-driven LLM framework that generates encounter structures and coordinates them with personal mobility behavior.

3. We develop an encounter-aware evaluation protocol that measures participant, temporal, and spatial structures of generated encounters and their dependencies.

4. Extensive experiments on Japan and NYC show that ENCOUNTERMOB outperforms all external baselines across all encounter and personal mobility metrics.

## 2 RELATED WORK

City-Scale Human Mobility Generation Deep learning has been widely studied for human mobility prediction and generation (Luca et al., 2023). Early neural methods use recurrent architectures to capture temporal dependencies in trajectories (Kulkarni & Garbinato, 2017; Berke et al., 2022), while later studies employ variational models, GANs, adversarial imitation learning, and diffusion models for trajectory generation (Huang et al., 2019; Feng et al., 2020; Choi et al., 2021; Zhu et al., 2023b; Zhang et al., 2025). More recently, LLM-based methods incorporate behavioral and contextual information into individual trajectory generation, including activity patterns and motivations (Wang et al., 2024), persona-conditioned activity planning (Ju et al., 2025), and event-specific context (Wang et al., 2026). These advances establish LLMs as a promising direction for human mobility generation.

LLMs for Human Mobility Modeling LLM-based approaches to human mobility use mobility histories and behavioral context to reason about individual movement (Wang et al., 2023; Gong et al., 2024). Recent frameworks further incorporate demographics, activities, motivations, preferences, and external events to support interpretable trajectory generation and behavioral adaptation (Zhang et al., 2024; Wang et al., 2024; Ju et al., 2025; Wang et al., 2026). Beyond individual mobility, household mobility work uses LLMs to infer activity dependencies among household members and coordinate their schedules (Liu et al., 2025). These studies extend LLM-based mobility modeling from individual behavior to coordinated and population-level mobility across multiple scales. However, existing city-scale mobility generation primarily treats resident independently, with encounter structures arising only from spatiotemporal overlap rather than cross-resident coordination.

## 3 PROBLEM FORMULATION

Given historical mobility records and the set of active residents ${ \mathcal { R } } _ { \mathrm { a c t } }$ for the target day, we aim to generate a daily trajectory $\widehat { \tau } _ { r }$ for each resident $r \in \mathcal { R } _ { \mathrm { a c t } }$ . Each trajectory records a time-ordered sequence of venue visits. When these trajectories place two or more residents at the same venue during the same time interval, they form an encounter, denoted by $\boldsymbol { e } = \left( t _ { e } , v _ { e } , \mathcal { R } _ { e } \right)$ , where $t _ { e } , v _ { e }$ and $\mathcal { R } _ { e }$ specify the time, venue, and complete set of co-present residents, respectively. For a set of trajectories $\tau$ , let $\mathcal { E } ( \mathcal { T } )$ denote all encounters induced by T. Their organization across participants, times, and venues defines the encounter structure of $\tau .$ . Formally, our task is to generate the joint trajectory set $\widehat { \mathcal { T } } = \{ \widehat { \tau } _ { r } \ | \ r \in \mathcal { R } _ { \mathrm { a c t } } \}$ such that each trajectory preserves the mobility regularities of its resident, while $\mathcal { E } ( \widehat { \mathcal { T } } )$ reproduces the encounter structure observed in real mobility data.

## 4 EVALUATION PROTOCOL FOR ENCOUNTER RECOVERY

Existing evaluation of human mobility generation mainly assesses mobility patterns at the individual or population level (Wang et al., 2024; 2026; Zhang et al., 2025; Yuan et al., 2025). However, these two levels capture different aspects of mobility fidelity and do not evaluate the cross-resident structure formed by encounters. For example, generated trajectories may reproduce individual mobility patterns and aggregate time–venue distributions, while different groups of residents meet at those times and venues. To this end, we propose a evaluation protocol to evaluate encounter recovery across participants, times, and venues, including their marginal and joint distributions.

Specifically, each encounter is jointly characterized by its complete participant group G, time $T ,$ and venue V. We evaluate encounter recovery using seven distributions, as summarized in Table 1. The marginal distributions of G, T, and V assess which residents participate in encounters, when encounters occur, and where they occur. The joint distributions $\bar { G \times T } , \bar { G } \times V .$ , and $T \times V$ evaluate the associations between these dimensions, while $G { \times } T { \times } V$ evaluates their participant–time–venue structure. Together, these dimensions provide a comprehensive evaluation of encounter recovery.

Table 1: Evaluation dimensions in the proposed encounter recovery protocol.
<table><tr><td>Dimension</td><td>Distribution</td><td>Characterization</td></tr><tr><td>Participant group</td><td>G (GroupD)</td><td>Complete participant groups involved in encounters</td></tr><tr><td>Time</td><td> $T \left( \mathrm { T D } \right)$ </td><td>Temporal distribution of encounters</td></tr><tr><td>Venue</td><td>V (VD)</td><td>Venue distribution of encounters</td></tr><tr><td>Participant-Time</td><td> $G \times T$ </td><td>Association between participant groups and times</td></tr><tr><td>Participant-Venue</td><td> $G \times V$ </td><td>Association between participant groups and venues</td></tr><tr><td>Time-Venue</td><td> $T \times V$ </td><td>Association between encounter times and venues</td></tr><tr><td>Full joint</td><td> $G { \times } T { \times } V$ </td><td>Joint participant-time-venue structure</td></tr></table>

Following prior mobility-generation evaluation (Wang et al., 2024; 2026), after extracting the above distributions from both generated and ground-truth mobility data, Jensen–Shannon divergence (JSD) is employed to quantify the discrepancy between them. Details are provided in Appendix C.2.

## 5 METHODOLOGY

Figure 2 presents ENCOUNTERMOB, an encounter-driven LLM framework for city-scale human mobility generation. It consists of a coarse-to-fine Encounter Generator and an Iterative Coupling Framework. The Encounter Generator derives concrete participant–time–venue encounters from city-level encounter rhythms, while the Iterative Coupling Framework iteratively adjusts encounter proposals and personal mobility behaviors. Together, the two components guide resident trajectories toward realistic encounter structures while preserving individual mobility regularities.

## 5.1 COARSE-TO-FINE ENCOUNTER GENERATOR

To guide mobility generation with encounters, we need to identify plausible encounters for the active residents. We introduce a coarse-to-fine Encounter Generator that first infers city-level encounter rhythms and then instantiates them into concrete participant–time–venue encounter proposals.

City-Level Encounter Rhythms Encounters arise as residents carry out their daily activities across the city. Since these activities follow recurring temporal and spatial routines, the resulting encounters also exhibit regularities in when and where they occur (Leng et al., 2021). For example, commuting brings many residents to stations during the morning and evening rush hours, while meal periods concentrate residents in restaurants, creating more opportunities for encounters at these times and places. Based on this, we characterize the overall encounter activity of the target day as city-level encounter rhythms, providing coarse guidance on the times and venue categories in which encounters are likely to occur.

To infer these rhythms for a specific target day, we consider both historical encounter activity and the behavioral context of the active residents. Historical encounter activity is summarized from several aspects, including daily encounter volume, temporal distribution, venue-category composition, and weekday variation. For each active resident, we further summarize historical mobility into a behavioral profile describing recurring activities together with their typical timing and venue conditions. The LLM combines these sources to determine how the historical encounter regularities should be reflected for the current population, producing the target-day encounter volume and its coarse distribution over time and venue categories.

Encounter Instantiation Given the city-level encounter rhythms, the next step is to instantiate specific encounters for the target day. In addition to the resident profiles used above, we summarize historical encounters through their participants, meeting times, venues, and activity contexts, providing more specific evidence for how residents have interacted in the past. The LLM combines this historical evidence with the coarse temporal and venue-category guidance to determine plausible participant–time–venue configurations. Historical relations support previously observed combinations, while the behavioral context of the residents also allows the LLM to consider new combinations when they are consistent with their activities. The resulting instantiated encounters form the initial encounter proposals for the subsequent Iterative Coupling Framework.

![](images/28db1edd50c72d4562b19031f964776919b3b94883ddc2920bf2f0058aa54dcb.jpg)  
Figure 2: Overview of ENCOUNTERMOB. (1) The coarse-to-fine Encounter Generator infers citylevel encounter rhythms from historical mobility and resident profiles, then instantiates participant– time–venue encounter proposals. (2) The Iterative Coupling Framework compares these proposals with encounters realized by independently initialized personal trajectories. Guided by historical evidence, the joint critic provides feedback to revise personal trajectories and, when needed, citylevel rhythms and encounter proposals. The personal trajectories form the final output. Colored dashed curves illustrate correspondences between proposed and realized encounters after coupling.

## 5.2 ITERATIVE COUPLING FRAMEWORK

The initial encounter proposals need to be further coordinated with the mobility behavior of the involved residents on the target day. A proposed encounter may require a resident to change an otherwise plausible activity, while established personal mobility patterns may make the proposed time, venue, or participant combination difficult to realize. We therefore treat both encounter proposals and personal mobility behavior as revisable. Then an Iterative Coupling Framework is introduced to enable them to couple each other through iterative feedback (Madaan et al., 2023; Gou et al., 2024).

First, we prompt the LLM to generate a complete daily trajectory for each resident based on historical personal mobility patterns, and then extract the encounters formed by these trajectories. Second, the LLM compares these extracted encounters with the encounter proposals and examines the historical evidence supporting both sides. For personal behavior, such evidence includes recurring activities and their typical timing and venue conditions; for encounter proposals, it includes the historical encounter contexts and resident activities used during instantiation. Based on this comparison, the Joint Critic either accepts the current coordination or identifies the remaining inconsistencies and provides feedback for refinement.

At each refinement round, the Critic feedback is passed to the personal mobility generator to revise the affected trajectories while preserving personal behavior consistent with the supported encounter proposals. The revised trajectories are then fed back to the coarse-to-fine Encounter Generator, which updates the city-level encounter rhythms and subsequently regenerates the encounter proposals conditioned on the updated rhythms and current personal trajectories. The revised trajectories and encounter proposals are evaluated again by the Joint Critic in the next round. Through this process, personal behavior can adapt to supported encounter proposals, while encounter proposals can also be revised according to well-supported personal behavior. The refinement proceeds for at most K rounds and terminates earlier once the Joint Critic accepts the current coordination.

Table 2: Encounter fidelity on Japan and NYC. All metrics are JSDs computed with natural logarithms on normalized histograms; lower is better. Bold and underline indicate the best and secondbest results among the compared methods; ablations are excluded from ranking.
<table><tr><td></td><td colspan="7">Japan</td><td colspan="7">NYC</td></tr><tr><td>Model</td><td>TD</td><td>VD</td><td>GroupD</td><td>T×V</td><td>G×T</td><td>G×V</td><td>G×T×V</td><td>TD</td><td>VD</td><td>GroupD</td><td>T×V</td><td>G×T</td><td>G×V</td><td>G×T×V</td></tr><tr><td>LSTM</td><td>0.09877</td><td>0.24476</td><td>0.68514</td><td>0.50963</td><td>0.69207</td><td>0.68664</td><td>0.69207</td><td>0.69315</td><td>0.66362</td><td>0.69315</td><td>0.69315</td><td>0.69315</td><td>0.69315</td><td>0.69315</td></tr><tr><td>GETNext</td><td>0.05961</td><td>0.23111</td><td>0.68071</td><td>0.49085</td><td>0.69315</td><td>0.68496</td><td>0.69315</td><td>0.65010</td><td>0.69315</td><td>0.69315</td><td>0.69315</td><td>0.69315</td><td>0.69315</td><td>0.69315</td></tr><tr><td>MHSA</td><td>0.06601</td><td>0.22904</td><td>0.68160</td><td>0.49340</td><td>0.69315</td><td>0.68468</td><td>0.69315</td><td>0.64424</td><td>0.66869</td><td>0.69315</td><td>0.69315</td><td>0.69315</td><td>0.69315</td><td>0.69315</td></tr><tr><td>TrajGAIL</td><td>0.08504</td><td>0.45497</td><td>0.68895</td><td>0.60298</td><td>0.69315</td><td>0.69167</td><td>0.69315</td><td>0.45976</td><td>0.66984</td><td>0.66637</td><td>0.69315</td><td>0.69315</td><td>0.69315</td><td>0.69315</td></tr><tr><td>DiffTraj</td><td>0.06280</td><td>0.22032</td><td>0.67645</td><td>0.46585</td><td>0.69286</td><td>0.68045</td><td>0.69286</td><td>0.66286</td><td>0.66728</td><td>0.69315</td><td>0.69315</td><td>0.69315</td><td>0.69315</td><td>0.69315</td></tr><tr><td>LLMob</td><td>0.22617</td><td>0.23132</td><td>0.65848</td><td>0.51287</td><td>0.68804</td><td>0.66946</td><td>0.68804</td><td>0.58444</td><td>0.69315</td><td>0.54434</td><td>0.69315</td><td>0.66869</td><td>0.69315</td><td>0.69315</td></tr><tr><td>ELLMob</td><td>0.07869</td><td>0.19624</td><td>0.56109</td><td>0.36724</td><td>0.59846</td><td>0.57975</td><td>0.59938</td><td>0.45658</td><td>0.54306</td><td>0.43226</td><td>0.55025</td><td>0.54232</td><td>0.54306</td><td>0.55025</td></tr><tr><td>ENCOUNTERMOB</td><td>0.03653</td><td>0.10779</td><td>0.30178</td><td>0.22091</td><td>0.35382</td><td>0.33267</td><td>0.35742</td><td>0.20716</td><td>0.29093</td><td>0.14600</td><td>0.32079</td><td>0.27262</td><td>0.29093</td><td>0.32079</td></tr><tr><td>w/o Rhythm</td><td>0.03844</td><td>0.12302</td><td>0.35228</td><td>0.23459</td><td>0.39645</td><td>0.38434</td><td>0.39952</td><td>0.17373</td><td>0.32835</td><td>0.13937</td><td>0.38107</td><td>0.28658</td><td>0.32835</td><td>0.38107</td></tr><tr><td>w/o Instantiation</td><td>0.04170</td><td>0.19092</td><td>0.45823</td><td>0.34306</td><td>0.52240</td><td>0.49942</td><td>0.52344</td><td>0.25463</td><td>0.44168</td><td>0.23567</td><td>0.47931</td><td>0.47926</td><td>0.48400</td><td>0.50454</td></tr><tr><td>w/o Coupling</td><td>0.06393</td><td>0.16443</td><td>0.52829</td><td>0.33275</td><td>0.56156</td><td>0.54584</td><td>0.56274</td><td>0.53722</td><td>0.58943</td><td>0.47331</td><td>0.61383</td><td>0.59937</td><td>0.60546</td><td>0.61383</td></tr></table>

## 6 EXPERIMENTS

## 6.1 EXPERIMENTAL SETUP

Datasets. We use two city-scale mobility datasets constructed from the global Foursquare checkin resource associated with LBSN2Vec (Yang et al., 2019; 2021), covering Japan and the New York metropolitan area. The Japan and NYC datasets use fixed cohorts of 1,049 and 226 residents constructed solely from historical-period encounters, with 60/30 and 123/61 historical/test days, respectively. Both datasets are discretized into 10-minute intervals(Wang et al., 2024; 2026) and use Foursquare venue categories. More detailed information is provided in Appendix C.1.

Metrics. We evaluate both encounter and personal mobility fidelity. For encounters, we use the metrics defined in Section 4: TD, VD, GroupD, T×V, G×T, G×V, and G×T×V. For personal mobility, we use six metrics from LLMob and ELLMob (Wang et al., 2024; 2026): step distance (SD) and step interval (SI) measure consecutive-stop distances and time gaps; daily activity routine distribution (DARD) and spatio-temporal visit distribution (STVD) capture time–activity and time– location patterns; category distribution (CD) and spatial grid distribution (SGD) capture activity composition and spatial visit distributions. All scores compare normalized observed and generated histograms using Jensen–Shannon divergence (JSD) (Lin, 1991); lower is better. Detailed definitions and calculation settings are provided in Appendix C.2.

Baselines. We compare ENCOUNTERMOB with two types of baselines: (1) deep-learning-based methods, including predictive models LSTM (Hochreiter & Schmidhuber, 1997), GETNext (Yang et al., 2022), and MHSA (Hong et al., 2023), and generative models TrajGAIL (Choi et al., 2021) and DiffTraj (Zhu et al., 2023b); and (2) LLM-based methods, including LLMob (Wang et al., 2024) and ELLMob (Wang et al., 2026). Within each dataset, all methods use the same settings, information, and evaluation metrics; the LLM-based methods use the same backbone unless otherwise specified. Implementation and generation details are provided in Appendix C.3.

## 6.2 MAIN RESULTS AND ANALYSIS

Main results. Tables 2 and 3 evaluate encounter and personal mobility fidelity on Japan and NYC. ENCOUNTERMOB outperforms all external baselines on all seven encounter metrics and all six personal metrics. Several baseline scores reach the maximum JSD value of 0.69315, indicating no overlap between the observed and generated distributions. ELLMob is competitive on personal mobility, leading all six baseline metrics in Japan and five in NYC, yet its encounter errors remain substantially higher than those of ENCOUNTERMOB. This shows that personal fidelity alone is not enough to assess generated mobility, and encounter recovery provides a complementary view.

The encounter improvements extend from temporal and venue distributions to complete participant groups and their joint configurations. Compared to ELLMob, ENCOUNTERMOB reduces GroupD by 46.2% in Japan and 66.2% in NYC, and reduces $G { \times } T { \times } V$ by 40.4% and 41.7%, respectively. These results support encounter modeling for recovering which residents meet, when, and where.

Table 3: Personal mobility fidelity on Japan and NYC. All metrics are JSDs computed with natural logarithms; lower is better. Bold and underline indicate the best and second-best results among the compared methods; ablations are excluded from ranking.
<table><tr><td rowspan="2">Model</td><td colspan="6">Japan</td><td colspan="6">NYC</td></tr><tr><td>SD</td><td>SI</td><td>DARD</td><td>STVD</td><td>CD</td><td>SGD</td><td>SD</td><td>SI</td><td>DARD</td><td>STVD</td><td>CD</td><td>SGD</td></tr><tr><td>LSTM</td><td>0.06371</td><td>0.10285</td><td>0.17937</td><td>0.58511</td><td>0.00493</td><td>0.11069</td><td>0.22653</td><td>0.11680</td><td>0.58435</td><td>0.68255</td><td>0.00870</td><td>0.20886</td></tr><tr><td>GETNext</td><td>0.05912</td><td>0.08853</td><td>0.15050</td><td>0.55073</td><td>0.01972</td><td>0.10460</td><td>0.25078</td><td>0.16924</td><td>0.57884</td><td>0.68183</td><td>0.00541</td><td>0.20827</td></tr><tr><td>MHSA</td><td>0.06143</td><td>0.05722</td><td>0.16247</td><td>0.55529</td><td>0.01929</td><td>0.11078</td><td>0.25595</td><td>0.15803</td><td>0.55372</td><td>0.67848</td><td>0.00734</td><td>0.19664</td></tr><tr><td>TrajGAIL</td><td>0.12680</td><td>0.08676</td><td>0.22410</td><td>0.60217</td><td>0.03489</td><td>0.22519</td><td>0.37290</td><td>0.20186</td><td>0.60749</td><td>0.68741</td><td>0.03541</td><td>0.35073</td></tr><tr><td>DiffTraj</td><td>0.08385</td><td>0.09709</td><td>0.14790</td><td>0.53861</td><td>0.01787</td><td>0.10721</td><td>0.23151</td><td>0.17015</td><td>0.55110</td><td>0.67917</td><td>0.01012</td><td>0.20594</td></tr><tr><td>LLMob</td><td>0.07955</td><td>0.14059</td><td>0.22595</td><td>0.58424</td><td>0.00638</td><td>0.12113</td><td>0.19057</td><td>0.15416</td><td>0.53990</td><td>0.67073</td><td>0.01240</td><td>0.21916</td></tr><tr><td>ELLMob</td><td>0.00931</td><td>0.01112</td><td>0.09474</td><td>0.35217</td><td>0.00121</td><td>0.05058</td><td>0.07621</td><td>0.05242</td><td>0.30089</td><td>0.39345</td><td>0.00583</td><td>0.07278</td></tr><tr><td>ENCOUNTERMOB</td><td>0.00429</td><td>0.00331</td><td>0.03391</td><td>0.15189</td><td>0.00037</td><td>0.01989</td><td>0.05221</td><td>0.02475</td><td>0.20448</td><td>0.27555</td><td>0.00117</td><td>0.04836</td></tr><tr><td>w/o Rhythm</td><td>0.00495</td><td>0.00462</td><td>0.04027</td><td>0.17562</td><td>0.00044</td><td>0.02420</td><td>0.05561</td><td>0.02735</td><td>0.21856</td><td>0.29499</td><td>0.00202</td><td>0.05280</td></tr><tr><td>w/o Instantiation</td><td>0.00566</td><td>0.00456</td><td>0.03828</td><td>0.17327</td><td>0.00039</td><td>0.02342</td><td>0.05883</td><td>0.02906</td><td>0.22089</td><td>0.30294</td><td>0.00133</td><td>0.05447</td></tr><tr><td>w/o Coupling</td><td>0.01005</td><td>0.01146</td><td>0.07809</td><td>0.30375</td><td>0.00172</td><td>0.05049</td><td>0.08013</td><td>0.04613</td><td>0.31159</td><td>0.41470</td><td>0.00331</td><td>0.08346</td></tr></table>

These encounter gains accompany improvements in personal mobility. Compared with ELLMob, ENCOUNTERMOB reduces the mean of the six personal errors by 58.8% in Japan and 32.7% in NYC, with lower errors on every personal metric in both datasets. The complete framework therefore improves encounter recovery while also preserving more faithfully individual mobility, suggesting that correctly formed encounters can in turn benefit the quality of individual mobiltiy generation.

Ablation study. We conduct ablation studies to examine three key designs in ENCOUNTERMOB: city-level encounter rhythms (w/o Rhythm), Encounter Instantiation (w/o Instantiation), and the Iterative Coupling Framework (w/o Coupling). As can be seen in Table 2, removing city-level encounter rhythms consistently degrades encounter generation performance on both datasets. The improvement on TD, VD, and T × V is the most significant, showcasing the importance of citylevel rhythms in recovering the temporal and venue distributions of encounters. We also observe improvements in GroupD on both datasets. This effectiveness is likely attributed to more accurate temporal and venue guidance, which reduces inconsistency in the resulting group distribution.

The importance of Encounter Instantiation is particularly evident in GroupD and the group-related joint metrics, where its removal causes consistent performance degradation on both datasets. This highlights its critical role in determining which residents participate in encounters and how they are associated with time and venue. These results also indicate that city-level encounter rhythms alone are insufficient to recover concrete participant–time–venue configurations.

By comparing ENCOUNTERMOB with w/o Coupling, we observe clear improvements in GroupD and the group-related joint metrics on both datasets, highlighting the role of iterative coupling in coordinating encounter requirements with personal mobility behavior. Meanwhile, the mean of the six personal mobility errors decreases by 53.1% in Japan and 35.4% in NYC, showing that improved encounter fidelity is accompanied by better individual mobility fidelity.

## 6.3 RECOVERY OF CITY-LEVEL ENCOUNTER RHYTHMS DISTRIBUTIONS

Figure 3 compares the encounter distributions in ground truth, ELLMob, our city-level encounter rhythms, and the final generated trajectories. The heatmaps show that ELLMob deviates from ground truth in how encounters are distributed across hours and venue categories, whereas our city-level encounter rhythms more closely capture the observed distribution. The hourly counts in Figure 3(f) reveal a similar difference in encounter volume: ELLMob produces an overly concentrated peak, while our city-level encounter rhythms better follow the temporal variation in ground truth. The encounters extracted from the final trajectories largely preserve this distribution, while the adjustments introduced during coupling generally bring both the category allocation and hourly counts closer to ground truth. These results show that city-level encounter rhythms provide effective guidance for recovering the encounter distribution while allowing adaptation to personal mobility.

![](images/769100aa636603ea25e4216295679f15ec11f138d8a07ffba31f4baf613bf1d4.jpg)

![](images/8a2887fb58a1802a4b9f806131584c7a38c097a64bcab59b2a6f85b784ce5846.jpg)  
Figure 3: City-level encounter patterns on Japan over 30 days. (a–e) Hourly encounter distributions across the ten most frequent ground-truth venue categories for ground truth, ELLMob, w/o Rhythm, w/o Instantiation, and final output. Each heatmap is normalized within these categories; colors are scaled by the maximum for each category across all five panels and hours. (f) Hourly encounter counts across all venue categories for the same five sources, pooled over the 30 days. Generated encounters are extracted from final trajectories.

Table 4: Participant-group recovery in Japan and NYC. Count is the number of generated encounter events, precision, recall, and F1 are percentages; higher is better.
<table><tr><td>Dataset</td><td>Method</td><td>Count</td><td>Precision ↑</td><td>Recall ↑</td><td>F1↑</td></tr><tr><td>Japan</td><td>ELLMob</td><td>1,627</td><td>13.83</td><td>15.53</td><td>14.63</td></tr><tr><td></td><td>w/o Rhythm</td><td>940</td><td>44.55</td><td>38.75</td><td>41.45</td></tr><tr><td></td><td>w/o Instantiation</td><td>1,104</td><td>27.86</td><td>28.73</td><td>28.29</td></tr><tr><td></td><td>ENCOUNTERMOB</td><td>1,111</td><td>49.66</td><td>49.86</td><td>49.76</td></tr><tr><td>NYC</td><td>ELLMob</td><td>26</td><td>50.00</td><td>30.77</td><td>38.10</td></tr><tr><td></td><td>w/o Rhythm</td><td>123</td><td>69.23</td><td>69.23</td><td>69.23</td></tr><tr><td></td><td>w/o Instantiation</td><td>88</td><td>31.82</td><td>53.85</td><td>40.00</td></tr><tr><td></td><td>ENCOUNTERMOB</td><td>80</td><td>71.43</td><td>57.69</td><td>63.83</td></tr></table>

## 6.4 RECOVERY OF PARTICIPANT IDENTITIES

Beyond predicting when and where encounters occur, Encounter Instantiation must also identify the full set of residents involved in each encounter. The group-based distributional metrics used in the main evaluation evaluate the distributions of participant groups, but they do not directly measure whether individual groups are recovered exactly. We therefore also report group precision, recall, and F1 under exact matching. A predicted group is counted as correct only when its full participant set exactly matches a group in the ground-truth data. As shown in Table 4, Encounter Instantiation improves participant-group recovery across both datasets. These results highlight its role in recovering the complete sets of residents involved in encounters. In NYC, w/o Rhythm achieves highe Recall and F1, but only by generating more encounters.

## 6.5 EFFECT OF ITERATIVE COUPLING

We investigate the effect of refinement rounds in the Iterative Coupling Framework. At each round, we evaluate encounter fidelity using G×T×V JSD, participant-group recovery using group F1 in Table 4, and the mean of the six personal JSD metrics. As shown in Figure 4, encounter JSD decreases while group F1 increases across rounds, indicating improved joint encounter fidelity and participant-group recovery. Encounter JSD decreases and group F1 increases across refinement rounds, while personal mean JSD remains low and decreases slightly throughout the process. These trends indicate that iterative refinement improves both distribution-level encounter fidelity and exact participant-group recovery without degrading personal mobility.

## 6.6 CASE STUDY OF ITERATIVE COUPLING

Figure 5 presents two representative cases to illustrate the refinement process of Iterative Coupling. In Japan (a), the encounter proposal places A and B at Train Station #2238 at 07:50 on July 11, while the initial trajectories place only B there. The Joint Critic identifies A as a missing participant and revises the trajectory of A by adding the 07:50 station stop while preserving the remaining four stops; the trajectory of B remains unchanged. The realized participant set is thereby corrected from {B} to {A,B}, recovering the complete ground-truth participant group.

![](images/e66ebd1de646009214d4a0fd1b75fad598067ee7bc7737e5cab4a7211dc11cb2.jpg)  
Realized JSD Proposal JSD Personal JSD Group F1

Figure 4: Effect of Iterative Coupling on Japan over R0–R3. Encounter JSD decreases across rounds, while group recovery F1 of the generated trajectories increases and personal mean JSD remains relatively stable. JSD curves use the left axis; group recovery F1 uses the right axis. Lower JSD and higher F1 are better.  
![](images/d00d30a6ff95ddeae5acc00f26b887506d90379fb94a5b7e4c3056352ad2ac96.jpg)  
Figure 5: Case studies of Iterative Coupling: (a) revising personal activity to realize a supported encounter; (b) revising an encounter proposal to preserve supported personal activity. Center panels paraphrase Critic feedback and subsequent revisions. Japan shows a morning segment; NYC shows full daily stops. Lines connect stops, not road routes. The cases use different saved configurations, with ground truth reserved for evaluation.

In NYC (b), the initial 22:30 bar proposal conflicts with the shared 15:50 coffee stop in the personal trajectories. The Joint Critic preserves the personal trajectories, and the Encounter Generator regenerates the proposal as a 15:50 coffee encounter, bringing the generated encounter closer to the ground truth. These cases show that the Joint Critic identifies the problematic part of the current generation and guides the subsequent refinement, either by revising the conflicting personal trajectory or by regenerating the encounter proposal while preserving the personal trajectories.

## 7 CONCLUSION

This work investigates how encounter information can improve LLM-based human mobility generation by explicitly modeling interactions among residents. Specifically, a LLM based framework ENCOUNTERMOB is proposed to incorporate two coupled mobility semantic representation. Moreover, a critic and a recovery protocol are introduced to enable the reliable harnessing of LLMs for human mobility generation. In experimental studies, we construct encounter-based datasets from the open-source mobility datasets to evaluate the performance of our framework. Future work will extend the framework to more diverse mobility settings and incorporate richer social and behavioral priors, exploring more realistic and generalizable mobility generation.

## AI USE STATEMENT

In this work, we used generative AI tools both as the object of study and as research assistants. Beyond the model calls evaluated by the proposed and baseline mobility-generation systems, generative AI tools helped refine hypotheses and methodology, review experimental designs, implement and inspect software, construct author-reviewed data-processing code, interpret intermediate results, translate and edit text, draft portions of the paper, and search and summarize candidate literature. We did not treat AI-generated text as an authoritative source: citations were checked against primary records, data transformations and evaluation code were inspected and tested, and empirical claims were restricted to retained experimental artifacts. The authors reviewed all AI-assisted work and take responsibility for the final content, including text, claims, code, and artifacts produced with generative AI assistance.

## ETHICS STATEMENT

This study analyzes historical check-in and friendship data, which can expose sensitive information about mobility and social relationships even when direct identifiers are absent. We therefore do not release raw user trajectories, friendship records, or API credentials. Any public release will be limited to permitted processing code, prompts, data contracts, and aggregate statistics, subject to verification of the source dataset’s redistribution terms. The generated interaction labels are operational co-presence proxies and must not be interpreted as evidence of conversation, shared intent, or private social behavior.

## REPRODUCIBILITY STATEMENT

The main text specifies the task, temporal split, event-construction protocol, information policy, and evaluation families. The appendix records the metric definitions, prompt and schema contracts, retry policy, deterministic fallbacks, and audit fields needed to reproduce a generated day. Token usage for the retained full runs is reported in Appendix D.2. A public code artifact containing the permitted preprocessing and evaluation pipeline is planned for a future release, subject to verification of the source dataset’s redistribution terms.

## REFERENCES

Alex Berke, Ronan Doorley, Kent Larson, and Esteban Moro. Generating synthetic mobility data for a realistic population with RNNs to improve utility and privacy. In Proceedings of the 37th ACM/SIGAPP Symposium on Applied Computing, pp. 964–967. ACM, 2022. doi: 10.1145/3477314.3507230. URL https://doi.org/10.1145/3477314.3507230.

Serdar Çolak, Antonio Lima, and Marta C. González. Understanding congested travel in urban areas. Nature Communications, 7(1), 2016. doi: 10.1038/ncomms10793. URL https://doi. org/10.1038/ncomms10793.

Serina Chang, Emma Pierson, Pang Wei Koh, Jaline Gerardin, Beth Redbird, David Grusky, and Jure Leskovec. Mobility network models of COVID-19 explain inequities and inform reopening. Nature, 589(7840):82–87, 2021. doi: 10.1038/s41586-020-2923-3. URL https://doi.org/ 10.1038/s41586-020-2923-3.

Seongjin Choi, Jiwon Kim, and Hwasoo Yeo. TrajGAIL: Generating urban vehicle trajectories using generative adversarial imitation learning. Transportation Research Part C: Emerging Technologies, 128:103091, 2021. doi: 10.1016/j.trc.2021.103091.

Jie Feng, Zeyu Yang, Fengli Xu, Haisu Yu, Mudan Wang, and Yong Li. Learning to simulate human mobility. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 3426–3433, 2020. doi: 10.1145/3394486.3412862.

Letian Gong, Yan Lin, Xinyue Zhang, Yiwen Lu, Xuedi Han, Yichen Liu, Shengnan Guo, Youfang Lin, and Huaiyu Wan. Mobility-LLM: Learning visiting intentions and travel preferences from human mobility data with large language models. In Advances in Neural Information Processing Systems, volume 37, pp. 36185–36217, 2024. doi: 10.52202/079017-1141.

Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. CRITIC: Large language models can self-correct with tool-interactive critiquing. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun (eds.), International Conference on Learning Representations, volume 2024, pp. 57734–57811, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/file/ fef126561bbf9d4467dbb8d27334b8fe-Paper-Conference.pdf.

Sepp Hochreiter and Jürgen Schmidhuber. Long Short-Term Memory. Neural Computation, 9(8): 1735–1780, 1997. doi: 10.1162/neco.1997.9.8.1735.

Ye Hong, Yatao Zhang, Konrad Schindler, and Martin Raubal. Context-aware multi-head selfattentional neural network model for next location prediction. Transportation Research Part C: Emerging Technologies, 156:104315, 2023. doi: 10.1016/j.trc.2023.104315.

Dou Huang, Xuan Song, Zipei Fan, Renhe Jiang, Ryosuke Shibasaki, Yu Zhang, Haizhong Wang, and Yugo Kato. A variational autoencoder based generative model of urban human mobility. In 2019 IEEE Conference on Multimedia Information Processing and Retrieval (MIPR), pp. 425– 430. IEEE, 2019. doi: 10.1109/MIPR.2019.00086. URL https://doi.org/10.1109/ MIPR.2019.00086.

Chenlu Ju, Jiaxin Liu, Shobhit Sinha, Hao Xue, and Flora Salim. TrajLLM: A modular LLMenhanced agent-based framework for realistic human trajectory simulation. In Companion Proceedings of the ACM on Web Conference 2025, pp. 2847–2850. ACM, 2025. doi: 10.1145/ 3701716.3715201. URL https://doi.org/10.1145/3701716.3715201.

Natalya Kostandova, Ronan Corgel, Shweta Bansal, et al. Improving mobility data for infectious disease research. Nature Human Behaviour, 9:1309–1312, 2025. doi: 10.1038/ s41562-025-02151-3.

Vaibhav Kulkarni and Benoît Garbinato. Generating synthetic mobility traffic using RNNs. In Proceedings of the 1st Workshop on Artificial Intelligence and Deep Learning for Geographic Knowledge Discovery, pp. 1–4. ACM, 2017. doi: 10.1145/3149808.3149809. URL https: //doi.org/10.1145/3149808.3149809.

Yan Leng, Dominiquo Santistevan, and Alex Pentland. Understanding collective regularity in human mobility as a familiar stranger phenomenon. Scientific Reports, 11(1), September 2021. ISSN 2045-2322. doi: 10.1038/s41598-021-98475-x. URL http://dx.doi.org/10.1038/ s41598-021-98475-x.

J. Lin. Divergence measures based on the shannon entropy. IEEE Transactions on Information Theory, 37(1):145–151, 1991. ISSN 0018-9448. doi: 10.1109/18.61115. URL http://dx. doi.org/10.1109/18.61115.

Yifan Liu, Xishun Liao, Haoxuan Ma, Brian Yueshuai He, Chris Stanford, and Jiaqi Ma. Human mobility modeling with household coordination activities under limited information via retrieval augmented LLMs. In 2025 IEEE 28th International Conference on Intelligent Transportation Systems, pp. 951–958, 2025. doi: 10.1109/ITSC60802.2025.11423539.

Massimiliano Luca, Gianni Barlacchi, Bruno Lepri, and Luca Pappalardo. A survey on deep learning for human mobility. ACM Computing Surveys, 55(1):1–44, 2023. doi: 10.1145/3485125.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Selfrefine: Iterative refinement with self-feedback. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 46534–46594. Curran Associates, Inc., 2023. doi: 10.52202/ 075280-2019. URL https://proceedings.neurips.cc/paper\_files/paper/ 2023/file/91edff07232fb1b55a505a9e9f6c0ff3-Paper-Conference.pdf.

Lijun Sun, Kay W. Axhausen, Der-Horng Lee, and Xianfeng Huang. Understanding metropolitan patterns of daily encounters. Proceedings of the National Academy of Sciences, 110(34):13774– 13779, 2013. doi: 10.1073/pnas.1306440110.

U.S. Census Bureau. 2024 TIGER/Line Shapefiles. Geographic boundary data, 2024. URL https://www.census.gov/geographies/mapping-files/time-series/ geo/tiger-line-file.html. Core Based Statistical Areas; accessed September 21, 2026.

Jiawei Wang, Renhe Jiang, Chuang Yang, Zengqing Wu, Makoto Onizuka, Ryosuke Shibasaki, Noboru Koshizuka, and Chuan Xiao. Large language models as urban residents: An LLM agent framework for personal mobility generation. In Advances in Neural Information Processing Systems, volume 37, 2024. URL https://proceedings.neurips.cc/paper\_files/paper/2024/hash/ e142fd2b70f10db2543c64bca1417de8-Abstract-Conference.html.

Xinglei Wang, Meng Fang, Zichao Zeng, and Tao Cheng. Where would i go next? large language models as human mobility predictors, 2023. URL https://arxiv.org/abs/2308. 15197.

Yusong Wang, Chuang Yang, Jiawei Wang, Xiaohang Xu, Jiayi Xu, Dongyuan Li, Chuan Xiao, and Renhe Jiang. ELLMob: Event-driven human mobility generation with self-aligned LLM framework. In International Conference on Learning Representations, 2026. URL https: //openreview.net/forum?id=MPYsaBgZIT.

Dingqi Yang, Bingqing Qu, Jie Yang, and Philippe Cudre-Mauroux. Revisiting user mobility and social relationships in lbsns: A hypergraph embedding approach. In The World Wide Web Conference, WWW ’19, pp. 2147–2157. ACM, May 2019. doi: 10.1145/3308558.3313635. URL http://dx.doi.org/10.1145/3308558.3313635.

Dingqi Yang, Bingqing Qu, Jie Yang, and Philippe Cudré-Mauroux. Lbsn2vec++: Globalscale check-in dataset with user social networks, 2021. URL https://data.4tu.nl/ articles/\_/15112308.

Song Yang, Jiamou Liu, and Kaiqi Zhao. GETNext: Trajectory flow map enhanced transformer for next POI recommendation. In Proceedings ofthe 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 1144–1153, 2022. doi: 10.1145/ 3477495.3531983.

Yuan Yuan, Jingtao Ding, Huandong Wang, Depeng Jin, and Yong Li. Activity trajectory generation via modeling spatiotemporal dynamics. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’22, pp. 4752–4762. ACM, August 2022. doi: 10.1145/3534678.3542671. URL http://dx.doi.org/10.1145/3534678.3542671.

Yuan Yuan, Jingtao Ding, Depeng Jin, and Yong Li. Learning the complexity of urban mobility with deep generative network. PNAS Nexus, 4(5):pgaf081, 2025. doi: 10.1093/pnasnexus/pgaf081.

Yuan Yuan, Yuheng Zhang, Jingtao Ding, and Yong Li. WorldMove, a global open data for human mobility. Scientific Data, 13:549, 2026. doi: 10.1038/s41597-026-06555-2.

Kunyi Zhang, Yanbo Pang, Yurong Zhang, and Yoshihide Sekimoto. MobGLM: A large language model for synthetic human mobility generation. In Proceedings of the 32nd ACM International Conference on Advances in Geographic Information Systems, SIGSPATIAL ’24, pp. 629– 632. ACM, 2024. doi: 10.1145/3678717.3691311. URL https://doi.org/10.1145/ 3678717.3691311.

Yuheng Zhang, Yuan Yuan, Jingtao Ding, Jian Yuan, and Yong Li. Noise matters: Diffusion modelbased urban mobility generation with collaborative noise priors. In Proceedings of the ACM on Web Conference 2025, pp. 5352–5363, 2025. doi: 10.1145/3696410.3714516.

Yu Zheng, Licia Capra, Ouri Wolfson, and Hai Yang. Urban computing: Concepts, methodologies, and applications. ACM Transactions on Intelligent Systems and Technology, 5(3):1–55, 2014. doi: 10.1145/2629592. URL https://doi.org/10.1145/2629592.

Yuanshao Zhu, Yongchao Ye, Ying Wu, Xiangyu Zhao, and James J. Q. Yu. Syn-Mob: Creating high-fidelity synthetic GPS trajectory dataset for urban mobility analysis. In Advances in Neural Information Processing Systems, volume 36, 2023a. URL https://proceedings.neurips.cc/paper\_files/paper/2023/ hash/4786c0d1b9687a841bc579b0b8b01b8e-Abstract-Datasets\_and\_ Benchmarks.html.

Yuanshao Zhu, Yongchao Ye, Shiyao Zhang, Xiangyu Zhao, and James J. Q. Yu. DiffTraj: Generating GPS trajectory with diffusion probabilistic model. In Advances in Neural Information Processing Systems, 2023b. URL https: //proceedings.neurips.cc/paper\_files/paper/2023/hash/ cd9b4a28fb9eebe0430c3312a4898a41-Abstract-Conference.html.

## A ALGORITHM PSEUDO-CODES

Algorithm 1 presents the coarse-to-fine Encounter Generator and Iterative Coupling Framework described in Section 5. Here, H denotes the historical mobility records; the LLM-based modules use the resident behavioral profiles and historical encounter summaries described in that section. We denote the city-level encounter rhythms by $C ,$ the participant–time–venue encounter proposals by $Q ,$ and the Joint Critic’s decision and feedback by a and $F ,$ respectively. Each LLM call uses the corresponding task prompt with the listed inputs; the target day d and active residents ${ \mathcal { R } } _ { \mathrm { a c t } }$ remain fixed throughout.

Personal trajectories are initialized independently of the encounter proposals. At each evaluation, $E = \mathcal { E } ( \widehat { \tau } )$ is extracted from the current trajectories using the encounter definition in Section 3. The algorithm returns the joint trajectory set ${ \widehat { \tau } } ;$ its realized encounters are $\mathcal { E } ( \widehat { T } )$ . The initial evaluation is indexed by $k = 0 ,$ , followed by at most K refinement rounds. Reaching the refinement limit returns the current trajectories without implying acceptance by the Joint Critic.

Algorithm 1 Encounter-Driven Human Mobility Generation   
Require: Historical mobility records H, target day d, active residents ${ \mathcal { R } } _ { \mathrm { a c t } }$ , maximum refinement rounds K   
Ensure: Joint trajectory set $\widehat { T } = \{ \widehat { \tau } _ { r } \ | \ r \in \mathcal { R } _ { \mathrm { a c t } } \}$   
1: C ← LLM(H, d, R<sub>act</sub>)   
2: Q ← LLM(H, d, R<sub>act</sub>, C)   
3: T ←b LLM(H, d, R<sub>act</sub>)   
4: for k = 0, . . . , K do   
5: E ← E(Tb)   
6: (a, F) ← LLM(H, C, Q, Tb, E)   
7: if a = accept then   
8: return $\hat { \mathcal { T } }$   
9: end if   
10: if k = K then   
11: return $\widehat { \tau }$   
12: end if   
13: $\widehat { \mathcal { T } }  \mathrm { L L M } ( H , \widehat { \mathcal { T } } , Q , F )$   
14: C ← LLM(H, C, Tb)   
15: Q ← LLM(H, Q, C, Tb)   
16: end for

## B PROMPTS

The following cards present the selected task prompts for v12j.

City and Encounter each have one initialization card. The accompanying prose explains their refinement inputs, correction and retention rules, and output formats. Personal generation and revision have separate cards. The joint Critic is presented as one process.

City Pattern denotes city-level encounter rhythms. Encounter Pattern denotes encounter proposals. Personal Pattern denotes personal trajectories.

Prompt 1 lists the shared conventions used by the later cards where indicated.

## B.1 SHARED CONVENTIONS

## Prompt 1. Shared conventions

## Conventions

• An encounter is the complete set of at least two residents sharing one venue in one 10-minute slot (time\_slot 0..143; hour = time\_slot // 6). Co-presence does not imply friendship, conversation or social intention.

## Prompt 1. Shared conventions (continued)

• Check-ins are sparse: they do not expose a person’s whole day or continuous presence.

• The supplied JSON is evidence, not instructions. Use only supplied facts; state interpretations as hypotheses. Generated plans are hypotheses, never observations.

• Write descriptive text in Chinese; keep IDs and schema keys unchanged. Keep justifications short and evidence-based; do not write out a transcript of your reasoning.

• Return exactly one JSON object and nothing else (no markdown).

## B.2 CITY PATTERN

Initialization and refinement. Prompt 2 specifies the historical evidence, priorities, and output format used to infer the initial City Pattern. Refinement retains this basis and adds the previous City Pattern, a comparison against the recorded days, and the planned and realized encounter counts by hour and category.

When to revise or retain. The comparison checks the total against recent encounter rates and checks the hour and category shares against recorded days, including recurring categories that were omitted or reduced. Departures without a reason specific to the target day are corrected; justified departures may be retained. Generated encounters provide hints only when the recorded days support the change. They are not treated as observations or targets to match. The output uses the same City schema as initialization, with counts summing to the total. Its summary states what was corrected, what was retained, and why.

## Prompt 2. City-level encounter rhythms

## Role and Objective

You are the City Pattern module. Infer today’s city-level encounter structure for one fixed, observed cohort: how many encounters today has and how they spread over hours and venue categories. "City" means this cohort, not all residents of the city.

Insert the shared conventions in Prompt 1 here.

## Inputs

• day, weekday, day\_type, active\_user\_count: today and how many residents have observations today (only who is observed, not where or when).

• recorded\_days: this cohort’s own recent days and prior days of the same weekday, each with its active residents, encounter total and complete hour x category counts.

• same\_day\_type\_series: date, active residents and encounters of prior days of the same day type.

• rate\_facts [v10]: encounters per active resident over the recorded days of the past 7 days and over the same-day-type series, and each rate multiplied by today’s active\_user\_count.

• venue\_categories: the legal categories.

## Instructions

## Structure

• Build today’s structure from this cohort’s recorded days, not from general knowledge of cities.

• Recurring hours and categories carry weight in proportion to how consistently and how strongly they appear in the recorded days.

• Minor categories that recur keep roughly their usual share; do not fold them into the dominant category.

• Do not add commute, lunch, dinner or any other typical-city peak the recorded days do not show.

![](images/c19cd9e312bdcbd2de72de45aac95423e56c581cfaaca208892c488cf7a45039.jpg)

## B.3 ENCOUNTER PATTERN

Initialization and refinement. Prompt 3 generates the initial encounter plan from the City Pattern and the groups’ meeting histories. Refinement applies the same priorities to the revised City Pattern and the previous plan. Each previous event is accompanied by its meeting history, whether it is currently realized, any stated member conflicts, and the members’ nearby planned visits. Additional candidates include unplanned co-presence with meeting history and habitual groups not yet scheduled.

When to revise or retain. A previous event is retained unless there is a concrete reason to remove it: a stated member conflict, weak historical support for the target day, or a more plausible alternative involving the same residents or venue and time. Realization or non-realization alone is insufficient to retain or remove an event. Plausible co-presence and habitual meetings may be added where City allows room; co-presence retained by the joint review provides additional support. Meeting times remain grounded in each group’s history. Venue selection prioritizes repeated meeting venues, then members’ nearby planned visits, then the group’s most recent venue. All retained and added events must satisfy the same resident and group consistency constraints. Unsupported City cells may remain unfilled, with the discrepancy reported.

Revision output. Initialization returns the full event list. Refinement instead returns the patch below. Every previous event absent from remove remains unchanged; removal IDs must refer to the previous plan. Added events follow the same participant, venue, and time constraints as initialization. Empty remove and add lists retain the entire plan.

"day": "YYYY-MM-DD",   
"remove": [{"event\_id": "e...", "reason": "..."}],   
"add": [{   
"participants": ["uid", "uid"],   
"time\_slot": 108,   
"venue": "v0",   
"basis": "..."   
}],   
"city\_discrepancy": "..."

## Prompt 3. Encounter proposals

## Role and Objective

You are the Encounter Pattern module. Write today’s encounter plan: which residents meet, at which venue and in which 10-minute slot, choosing the meetings that are most believable today and that realize City’s hour x category cells.

Insert the shared conventions in Prompt 1 here.

## Inputs

• day, weekday.

• city: today’s City cells (hour, category, count) and event\_count.

• groups: complete groups of today’s active residents that met before (in the past week or on this weekday), each with meeting\_history.

• venues: the legal venue aliases with their categories.

• Personal plans are generated independently and are not supplied.

## Instructions

## Who meets

• meeting\_history describes how this exact complete group met before today: times\_together, last\_together, days\_since\_last, times\_in\_past\_week, times\_on\_this\_weekday, usual\_places (the most recent meetings with day, weekday, time\_slot, venue, category), venue\_counts (how many past meetings took place at each venue) and past\_slots (time slots of past meetings). Groups that never met carry times\_together 0.

• People who met in the past few days, or who meet on this weekday as a habit, are likely to meet again; a group whose only meetings were weeks ago is weaker evidence.

• You may form a new group from supplied residents whose own places and times coincide when that is believable; residents who never met rarely stand at exactly the same venue in the same slot, so do not invent such meetings freely.

## When [v10]

• A group meets at the times its own past meetings support (past\_slots, usual\_places): keep a meeting near the group’s own past times.

• City’s cells never move a chosen group’s time. City decides which groups to choose for a cell: fill an hour x category cell with groups whose own history places them in that hour and category; if no such group exists, leave the cell unfilled and say so.

## Where [v10]

Prompt 3. Encounter proposals (continued)   
• Real repeat meetings often take place at a venue new to the group, so a single past venue   
is weak evidence. Choose the venue by the strongest evidence available, in this order:   
1. a venue where the group met repeatedly (venue\_counts of 2 or more);   
2. a venue where today’s plans already put the members near that time   
(members\_today\_near\_time, when supplied);   
3. the group’s most recent meeting venue.   
Consistency   
• Each resident is at one venue per slot; one venue-slot holds one complete group.   
City   
• City’s cells say how many meetings to expect and in which hours and categories. Aim to   
be close to City in total and cell by cell with believable meetings; City is a reference, not a   
quota.   
Priorities   
1. Believability of who meets (meeting history).   
2. The group’s own time; the venue by the evidence order above.   
3. City’s cells, used only to choose among believable groups; never to move a group’s time   
and never to add a meeting that is not believable.   
Reasoning Steps   
1. For each City cell, list the groups whose own history places them in that hour and category.   
2. Choose the most believable of them, up to the cell’s count.   
3. For each chosen group take a time from its own past times and a venue by the evidence   
order.   
4. Check one venue per resident per slot and one group per venue-slot; note unfilled cells.   
Output Format   
{"day":"YYYY-MM-DD","events":[{"participants":["uid","uid"],"time\_slot":108,"venue":   
"v0", "basis":"short reason"}],"city\_discrepancy":"short summary of where and why   
the plan departs from City"}   
• time\_slot 0..143; venue is a supplied alias; every event has at least two distinct supplied   
residents.   
Final Reminder   
Choose believable groups for City’s cells; keep each group at its own time; never move a   
group to fit a cell.

## B.4 PERSONAL TRAJECTORY GENERATION

Initial generation. Prompt 4 presents the simple generation task: infer a plausible target-day trajectory from one resident’s historical visits.

Feedback-guided revision. Prompt 5 takes the previous trajectory, personal evidence, current encounter proposals, and the joint Critic’s advice. It produces a revised trajectory that addresses encounter consistency and personal realism while preserving the visits that should remain. These two tasks are shown separately because revision introduces an existing plan, explicit correction requests, and constraints on what must be preserved.

Prompt 4. Personal trajectory generation

## Role and Objective

You are the Personal Pattern module. Generate one resident’s trajectory for the target day based on their historical trajectories.

## Inputs

• Target date and day of the week.

• The resident’s historical trajectories.

• Available venues and their identifiers.

## Instructions

Use the historical trajectories to identify the resident’s usual places, visit times, and activity sequences. Generate a plausible trajectory for the target day that follows these personal patterns. Keep visits chronologically ordered and use valid venue identifiers.

## Output Format

Return one JSON object:   
{   
"day": "YYYY-MM-DD",   
"stops": [   
{"time\_slot": 44, "venue": "v0"}   
]   
}   
Each time\_slot is a ten-minute slot from 0 to 143. Return only the JSON object.

## Prompt 5. Personal trajectory revision

## Role and Objective

You are the Personal Pattern module, adjusting the day this resident actually lives on the target date. The day must meet two requirements: it agrees with today’s encounter plan, and it agrees with the resident’s own living habits. critic\_advice tells you what the encounter plan and the habit review ask.

Insert the shared conventions in Prompt 1 here.

## Inputs

• uid, day, weekday, personal\_profile, historical\_evidence, historical\_summary (observed stops, active days, daily stop counts).

• candidate\_pattern\_library: complete observed days of this resident (with days\_before\_target); venue\_catalog; allowed\_evidence\_ids.

• previous\_personal: your last day.

• critic\_advice: concrete actions for today (REALIZE <event\_id>, DROP <event\_id>, removals of overlaps, and possibly "Personal realism" advice from the habit review); must\_preserve.

• current\_encounter\_pattern: planned meetings that involve this resident.

## Instructions

Requirement 1: agree with the encounter plan

• For each event you are asked to realize, place a stop at exactly its venue and time\_slot, by inserting a stop or moving an existing stop there; cite a historical evidence ID, preferring your own past visit to that venue.

• For an advised removal of an overlap, delete that stop or shift it by 1-2 slots as advised.

## Prompt 5. Personal trajectory revision (continued)

• If an action would require a day this resident would not live, or collides with another planned meeting you attend, do not perform it and state the concrete conflict in unresolved, naming the event\_id. If the advice asks you to keep your stops, keep them.

## Requirement 2: agree with your own habits

• The rest of the day is what you would actually do today given these meetings: places, times, order, number of check-ins and gaps as your recorded days show them.

• Your last few recorded days show your current life best: take today’s places, times and the shape of the day from them first, and use older days, even of the same weekday, only where the recent days say nothing.

• Recency ranks above the weekday: a recorded day from the last few days outranks a same-weekday day from earlier weeks as the reference for today.

• You decide the day. You may reuse a recorded day that fits today, whole or in part, around the meeting stops; you may also combine what your recorded days show.

• Keep real unevenness and occasional places; never replace an occasional place by one of your most frequent places; do not fill unobserved hours or spread stops evenly.

• When "Personal realism" advice asks to rewrite the day, write the day anew outside the meeting stops, from the recent recorded days it names: you may replace every other stop - its places, times and the number of check-ins; previous\_personal is not a constraint. When it asks to change a part, change that part.

• With encounter actions only (no "Personal realism" advice), change what those actions call for and keep the rest of previous\_personal.

• Never claim a meeting is realized unless your day has its exact venue-slot.

## Priorities

1. Stops that realize planned meetings or kept encounters are placed and kept exactly.

2. Advised removals are carried out.

3. The rest of the day agrees with your own habits, the last few days before same-weekday days.

## Reasoning Steps

1. Read critic\_advice and list the required stops and removals.

2. Decide whether the advice asks to rewrite the day or to change a part; decide what the rest of today looks like for you around the meeting stops, from your recent recorded days.

3. Write the day; carry out the removals; note any concrete conflict.

4. Check slots, venues and evidence citations.

## Output Format

One resident: {"uid":"...","day":"YYYY-MM-DD","selected\_anchor\_pattern\_id":"uid\_date", "stops":[{"time\_slot":0,"venue":"v0","activity":"short phrase","evidence\_ids":["s..."], "flexibility":"short phrase"}],"revision\_summary":"short sentence","unresolved":"short sentence; none if resolved"}

• selected\_anchor\_pattern\_id is the recorded day in candidate\_pattern\_library closest to your day, copied exactly; at least one stop; unique strictly increasing slots 0..143; at least one of allowed\_evidence\_ids per stop.

## Final Reminder

Two requirements: the encounter plan’s stops exactly as advised, and a day that agrees with your own habits.

## B.5 JOINT CRITIC

Joint review. Prompt 6 presents one review process covering encounter consistency and personal realism. It produces advice for each resident and lists issues that remain open. Acceptance requires no remaining advice or unresolved issues; otherwise, the process returns concrete revision requests.

Prompt 6. Joint Critic

## Role and Objective

You are the Joint Critic. Review the encounter proposals and personal trajectories together. Identify inconsistencies between planned and realized encounters and departures from residents’ recorded habits. Return one coherent set of feedback for refinement.

Insert the shared conventions in Prompt 1 here.

## Inputs

• Target date and day of the week.

• The current City pattern: the expected encounter volume and its distribution over hours and venue categories.

• The current Encounter pattern: proposed meetings with event identifiers, participants, times, and venues.

• Personal trajectories and the actual encounters extracted from them.

• Residents’ profiles, historical trajectories, meeting histories, and available evidence identifiers.

## Instructions

1. Review encounter consistency. Compare each proposed meeting with the actual encounters. Use meeting history to judge whether unrealized proposals should be carried out or dropped, and whether unplanned encounters should be retained or removed. Use the City pattern as a reference when meetings are similarly supported, rather than as an exact quota.

2. Review personal realism. Check every resident’s places, times, visit order, number of check-ins, and gaps against their own recorded days. Give recent records priority when they show a change in routine. Preserve sparse and uneven activity patterns supported by the history.

3. Provide coordinated advice. Specify the stops that should be inserted, moved, removed, or preserved. For a retained meeting, coordinate the same venue and time across its participants. Habit-related changes must preserve retained meeting stops. Check that the proposed edits do not create conflicting visits or new unsupported overlaps.

4. Combine the feedback. Give at most one advice item per resident, combining encounter-related changes and personal-habit corrections. Cite supporting evidence, keep the explanation brief, and state any remaining issue explicitly.

## Output Format

Return one JSON object:   
{   
"decision": "ACCEPT or REVISE",   
"assessment": "<brief joint assessment>",   
"personal\_advice": [   
{   
"uid": "<resident\_id>",   
"instruction": "<coordinated trajectory changes>",   
"event\_ids": ["<relevant\_event\_id>"],   
"evidence\_ids": ["<supporting\_evidence\_id>"],   
"must\_preserve": "<supported stops or meetings>"

## Prompt 6. Joint Critic (continued)

}   
],   
"unresolved\_issues": ["<remaining issue>"],   
"accept\_reason": "<brief reason, or empty when revising>"   
}   
Use ACCEPT only when no corrective advice or unresolved issue remains; both lists must   
then be empty. Use REVISE otherwise, with concrete advice and the issues it addresses. Use   
empty event\_ids for advice unrelated to a proposed meeting. Reporting an issue does not   
count as resolving it.   
Final Reminder   
Judge encounter consistency and personal realism jointly. Return one coherent review,   
grounded in the supplied evidence.

## C EXPERIMENTAL SETUP

## C.1 DATASET CONSTRUCTION AND PREPROCESSING

We construct the Japan and NYC datasets from the global Foursquare check-in resource associated with LBSN2Vec (Yang et al., 2019; 2021). The check-in records and venue information provide resident identifiers, visiting times, venue identifiers, geographical coordinates, and venue categories. Starting from this resource, we construct each dataset in four steps: (1) specify the study region and historical/test periods; (2) preprocess timestamps and venue categories; (3) extract encounters from the processed check-ins; and (4) select residents using historical encounters and assemble their trajectories and encounters for both periods.

1. Study regions and periods. We construct one dataset for Japan and one for the New York metropolitan area. The Japan window spans May 2–July 30, 2012, with the first 60 days used as the historical period and the last 30 days reserved for generation and evaluation. For NYC, we retain check-ins within the New York metropolitan area (CBSA 35620), using the boundary from the 2024 TIGER/Line CBSA files (U.S. Census Bureau, 2024). Its historical period is July 1–October 31, 2012 (123 days), and its test period is November 1–December 31, 2012 (61 days).

2. Temporal and semantic preprocessing. We organize check-ins by resident and local date, converting NYC timestamps to local time using the offsets supplied with the records. For both datasets, we discretize time into 10-minute intervals, following LLMob (Wang et al., 2024). Each activity stop retains its venue identifier, and the Foursquare taxonomy provides the venue’s activity category. These records form the daily trajectories used for encounter extraction.

3. Encounter extraction. For each day d, ten-minute time slot $t \in \{ 0 , \ldots , 1 4 3 \}$ }, and venue identifier v, we collect the distinct residents with a recorded stop in that cell:

$$
\mathcal { R } _ { d , t , v } = \{ r : \mathrm { r e s i d e n t } r \ : \mathrm { h a s \ a \ r e c o r d e d \ s t o p \ a t } ( d , t , v ) \} .\tag{1}
$$

Whenever $| \mathcal { R } _ { d , t , v } | \geq 2$ , we extract one encounter $\boldsymbol { e } = ( t , \boldsymbol { v } , \mathcal { R } _ { d , t , v } )$ , following the definition in Section 3, and retain its date d. All residents in the cell form the complete participant set, with each resident counted once. Encounters in consecutive occupied slots are recorded as separate events. We first apply this procedure to historical records to support resident selection in the next step.

4. Resident selection and dataset assembly. For each resident pair, we count the historical encounters whose participant set contains both residents. A pair qualifies when this count is at least two. Each distinct day–time-slot–venue cell contributes one occurrence, with time slots defined at ten-minute resolution; the occurrences may be on the same or different days. We then select every historical encounter containing at least one qualifying pair and include its complete participant set in the resident cohort. The repetition threshold applies to the qualifying pair: other residents in such an encounter are included even if they participate in only one qualifying encounter.

For example, suppose two historical encounters in distinct cells have participant sets $\{ A , B \}$ and $\{ A , B , C \}$ . The pair (A, B) occurs twice and therefore qualifies. Both encounters are selected, so residents A, B, and C all enter the cohort, even if C appears only in the second encounter.

The union of these participant sets defines the fixed cohort, yielding 1,049 residents for Japan and 226 for NYC. We retain the selected residents’ trajectories in both the historical and test periods and apply the same extraction procedure to obtain the encounters in each final dataset. Cohort membership is determined entirely from the historical period and remains fixed throughout evaluation.

Table 5 summarizes the resulting datasets. The Japan test period contains 1,950 encounter participation records, and NYC contains 154. The NYC historical period contains 5,963 activity stops and 258 encounters.

## C.2 METRICS

We evaluate encounter structure and personal mobility using normalized histograms and Jensen– Shannon divergence (JSD). Histograms are pooled over each dataset-specific panel before comput-

Table 5: Dataset statistics. Activity stops, active resident-days, and encounters refer to the test period.
<table><tr><td>Statistic</td><td>Japan</td><td>NYC</td></tr><tr><td>Historical period (days)</td><td>60</td><td>123</td></tr><tr><td>Test period (days)</td><td>30</td><td>61</td></tr><tr><td>Residents</td><td>1,049</td><td>226</td></tr><tr><td>Activity stops</td><td>43,147</td><td>3,432</td></tr><tr><td>Active resident-days</td><td>12,887</td><td>2,017</td></tr><tr><td>Encounters</td><td>921</td><td>77</td></tr></table>

ing divergence. We define the encounter metrics and the personal mobility metrics adapted from LLMob and ELLMob (Wang et al., 2024; 2026) below.

Encounter Metrics. We apply the encounter extraction procedure in Section C.1 to both observed and generated trajectories before constructing the evaluation histograms.

Let $\mathcal { E } _ { \mathrm { o b s } }$ and ${ \mathcal E } _ { \mathrm { g e n } }$ denote the observed and generated encounter sets. For each encounter $e , G _ { e } = \mathcal { R } _ { e }$ is its complete unordered participant set, $t _ { e }$ is its time-of-day slot, and $v _ { e }$ is its venue identifier. We pool encounters over dates and count each event once, irrespective of its number of participants.

For each view $X \in \{ T , V , G , T \times V , G \times T , G \times V , G \times T \times V \}$ , let $h _ { \mathcal { E } } ^ { X }$ be the count histogram defined below and $\Omega _ { X }$ the union of observed and generated support. For nonempty encounter sets, define

$$
p _ { X } ( x ) = \frac { h _ {  { \varepsilon } _ { \mathrm { o b s } } } ^ { X } ( x ) } { |  { \mathcal { E } } _ { \mathrm { o b s } } | } , \qquad q _ { X } ( x ) = \frac { h _ {  { \mathcal { E } } _ { \mathrm { g e n } } } ^ { X } ( x ) } { |  { \mathcal { E } } _ { \mathrm { g e n } } | } , \qquad m _ { X } ( x ) = \frac { p _ { X } ( x ) + q _ { X } ( x ) } { 2 } .\tag{2}
$$

The divergence for each view is

$$
\mathrm { J S D } ( p _ { X } , q _ { X } ) = \frac { 1 } { 2 } \sum _ { x \in \Omega _ { X } } p _ { X } ( x ) \ln \frac { p _ { X } ( x ) } { m _ { X } ( x ) } + \frac { 1 } { 2 } \sum _ { x \in \Omega _ { X } } q _ { X } ( x ) \ln \frac { q _ { X } ( x ) } { m _ { X } ( x ) } .\tag{3}
$$

We use natural logarithms, so the divergence lies in [0, ln $2 ]$ , without taking its square root. Zeroprobability terms contribute zero. The seven metrics are computed as follows; each histogram formula applies separately to $\mathcal { E } _ { \mathrm { o b s } }$ and ${ \mathcal E } _ { \mathrm { g e n } }$

Time distribution (TD). This metric compares the allocation of encounters across time-of-day slots:

$$
\begin{array} { r } { h _ { \mathcal { E } } ^ { T } ( t ) = \displaystyle \sum _ { e \in \mathcal { E } } \mathbb { I } [ t _ { e } = t ] , } \\ { \mathrm { T D } = \mathrm { J S D } ( p _ { T } , q _ { T } ) . } \end{array}\tag{4}
$$

Venue distribution (VD). This metric compares encounter frequencies at each venue:

$$
\begin{array} { r } { h _ { \mathcal { E } } ^ { V } ( v ) = \displaystyle \sum _ { e \in \mathcal { E } } \mathbb { I } [ v _ { e } = v ] , } \\ { \mathrm { V D } = \mathrm { J S D } ( p _ { V } , q _ { V } ) . } \end{array}\tag{5}
$$

Participant-group distribution (GroupD). This metric compares the frequencies of complete participant sets, with equality defined by resident identities:

$$
\begin{array} { r l } & { \quad h _ { \mathcal { E } } ^ { G } ( G ) = \displaystyle \sum _ { e \in \mathcal { E } } \mathbb { I } [ G _ { e } = G ] , } \\ & { \quad \mathrm { G r o u p D } = \mathrm { J S D } ( p _ { G } , q _ { G } ) . } \end{array}\tag{6}
$$

Time–venue distribution $( T \times V )$ . This metric compares when encounters occur at each venue:

$$
\begin{array} { r l r } {  { h _ { \mathcal { E } } ^ { T \times V } ( t , v ) = \sum _ { e \in \mathcal { E } } \mathbb { I } [ t _ { e } = t ] \mathbb { I } [ v _ { e } = v ] , } } \\ & { } & { D _ { T \times V } = \mathrm { J S D } ( p _ { T \times V } , q _ { T \times V } ) . } \end{array}\tag{7}
$$

Participant–time distribution $( G \times T )$ . This metric compares when each complete participant group meets:

$$
\begin{array} { r l r } {  { h _ { \varepsilon } ^ { G \times T } ( G , t ) = \sum _ { e \in \mathcal { E } } \mathbb { I } [ G _ { e } = G ] \mathbb { I } [ t _ { e } = t ] , } } \\ & { } & { D _ { G \times T } = \mathrm { J S D } ( p _ { G \times T } , q _ { G \times T } ) . } \end{array}\tag{8}
$$

Participant–venue distribution $( G \times V )$ . This metric compares where each complete participant group meets:

$$
\begin{array} { r l } & { { \displaystyle h _ { \varepsilon } ^ { G \times V } ( G , v ) = \sum _ { e \in \mathcal { E } } \mathbb { I } [ G _ { e } = G ] \mathbb { I } [ v _ { e } = v ] } , } \\ & { \quad \quad \quad D _ { G \times V } = \mathrm { J S D } ( p _ { G \times V } , q _ { G \times V } ) . } \end{array}\tag{9}
$$

Participant–time–venue distribution $( G \times T \times V )$ . This metric compares the full encounter configurations:

$$
\begin{array} { r l } & { h _ { \varepsilon } ^ { G \times T \times V } ( G , t , v ) = \displaystyle \sum _ { e \in \mathcal { E } } \mathbb { I } [ G _ { e } = G ] \mathbb { I } [ t _ { e } = t ] \mathbb { I } [ v _ { e } = v ] , } \\ & { \quad \quad \quad \quad D _ { G \times T \times V } = \displaystyle \mathrm { J S D } \big ( p _ { G \times T \times V } , q _ { G \times T \times V } \big ) . } \end{array}\tag{10}
$$

Personal Mobility Metrics. Let S denote the collection of recorded stops in the evaluation panel. A stop s has resident $r _ { s }$ , day $d _ { s } ,$ , minute of day $\tau _ { s } .$ , exact venue $v _ { s } ,$ , raw venue category $a _ { s } .$ , and coordinates $( \phi _ { s } , \lambda _ { s } )$ Within each resident-day, stops are sorted chronologically. Let A contain adjacent stop pairs in these sequences. No transitions are added across day boundaries, and a day with fewer than two stops contributes no SD or SI pair. All recorded stops contribute equally to the visit-based histograms, rather than being weighted by inferred dwell time.

Shared scoring rule. The shared comparison is Jensen–Shannon divergence (Lin, 1991). For any personal histogram, normalize observed and generated counts separately: $p _ { i } = r _ { i } / \sum _ { j } r _ { j }$ and $q _ { i } =$ $g _ { i } / \sum _ { j } g _ { j }$ . With $m _ { i } = ( p _ { i } + q _ { i } ) / 2$ , compute

$$
\mathrm { J S D } ( p , q ) = \frac { 1 } { 2 } \sum _ { i } p _ { i } \ln \frac { p _ { i } } { m _ { i } } + \frac { 1 } { 2 } \sum _ { i } q _ { i } \ln \frac { q _ { i } } { m _ { i } } .\tag{11}
$$

This uses natural logarithms, has range $[ 0 , \ln { 2 } ]$ , and is the divergence itself, without taking its square root.

Step distance (SD). Following the consecutive-stop distance feature evaluated by LLMob (Wang et al., 2024), we compute the following histogram with our fixed binning. For adjacent stops $( s , s ^ { \prime } ) \in$ A with valid coordinates, their Haversine distance in kilometers is

$$
\delta ( s , s ^ { \prime } ) = 2 R \arcsin \sqrt { \sin ^ { 2 } \frac { \phi _ { s ^ { \prime } } - \phi _ { s } } { 2 } + \cos \phi _ { s } \cos \phi _ { s ^ { \prime } } \sin ^ { 2 } \frac { \lambda _ { s ^ { \prime } } - \lambda _ { s } } { 2 } } , \quad R = 6 3 7 1 \mathrm { k m } ,\tag{12}
$$

where angles are in radians. Distances are not rounded before binning. Using $b _ { \mathrm { S D } } ( x ) \ =$ min $( \lfloor x / 0 . 1 \rfloor$ , 1000), we form

$$
h ^ { \mathrm { S D } } ( k ) = \sum _ { ( s , s ^ { \prime } ) \in \cal { A } } \mathbb { I } [ b _ { \mathrm { S D } } ( \delta ( s , s ^ { \prime } ) ) = k ] .\tag{13}
$$

Step interval (SI). This feature follows the consecutive-activity time interval evaluated in ActSTD and LLMob (Yuan et al., 2022; Wang et al., 2024). For adjacent stops in a resident-day, $\Delta ( s , s ^ { \prime } ) =$ $\tau _ { s ^ { \prime } } - \tau _ { s }$ is their recorded time gap in minutes. Define

$$
h ^ { \mathrm { S I } } ( k ) = \sum _ { ( s , s ^ { \prime } ) \in \cal { A } } \mathbb { I } [ \operatorname* { m i n } ( \lfloor \Delta ( s , s ^ { \prime } ) / 1 0 \rfloor , 1 4 4 ) = k ] .\tag{14}
$$

Daily activity routine distribution (DARD). We use the time–activity view introduced in the evaluation of LLMob (Wang et al., 2024). With $t _ { s } = \lfloor \tau _ { s } / 1 0 \rfloor$ , the joint time–activity histogram is

$$
h ^ { \mathrm { D A R D } } ( t , a ) = \sum _ { s \in \mathcal { S } } \mathbb { I } [ t _ { s } = t ] \mathbb { I } [ a _ { s } = a ] .\tag{15}
$$

Spatio-temporal visit distribution (STVD). The histogram is

$$
h ^ { \mathrm { S T V D } } ( t , v ) = \sum _ { s \in \cal S } \mathbb { I } [ t _ { s } = t ] \mathbb { I } [ v _ { s } = v ] .\tag{16}
$$

Category distribution (CD). Let $c ( a )$ map a raw venue category to the historical Foursquare top-level taxonomy used for our dataset. The histogram is

$$
h ^ { \mathrm { C D } } ( c ) = \sum _ { s \in \cal S } \mathbb { I } [ c ( a _ { s } ) = c ] .\tag{17}
$$

Spatial grid distribution (SGD). Let $\mathcal { S } _ { \mathrm { J a p a n } }$ contain stops covered by the Japan land polygon from Natural Earth at 1:10 million scale, including its boundary. We project coordinates to the equal-area system EPSG:6933. For projected coordinates $( x _ { s } , y _ { s } )$ in meters, define

$$
g ( s ) = \left( \left\lfloor \frac { x _ { s } } { 1 0 0 0 } \right\rfloor , \left\lfloor \frac { y _ { s } } { 1 0 0 0 } \right\rfloor \right) , \qquad h ^ { \mathrm { S G D } } ( g ) = \sum _ { s \in \mathcal { S } _ { \mathrm { J a p a n } } } \mathbb { I } [ g ( s ) = g ] .\tag{18}
$$

## C.3 IMPLEMENTATION DETAILS

We use gpt-5.6-luna as the shared LLM backbone for ENCOUNTERMOB, LLMob, and ELLMob.

For ENCOUNTERMOB, the maximum number of refinement rounds is set to $K = 3$ on both Japan and NYC. For LLMob, we use the learning-based motivation retrieval variant (LLMob-L). For ELL-Mob, we use the Normal setting with weekday/weekend context and retain the released implementation’s limit of three reflection attempts.

Retry policy. For ENCOUNTERMOB, failed generation requests are retried until an output passes parsing and validity checks.

## D ADDITIONAL EXPERIMENTAL RESULTS

## D.1 ENCOUNTER DISTRIBUTIONS ACROSS VENUE CATEGORIES

Figure 6 complements the main-text category comparison with spatial distributions of encounters in transportation, dining, retail, and entertainment venues. These maps show that the discrepancies of the baselines include both over-generation and under-generation across encounter scenarios.

![](images/cbc49109dcb286fff6e02cf76ac800ef6ca35e1221814de933eba194efe830b7.jpg)  
Figure 6: Spatial distributions of actual encounters in Greater Tokyo over the same 30-day evaluation period. Rows represent venue categories, and columns compare ground truth with encounters extracted from the final trajectories generated by each method. Panels within each row share the ground-truth count denominator and color scale.

## D.2 TOKEN CONSUMPTION

Table 6 records the token cost in Japan and NYC.

## D.3 EXAMPLES

We illustrate one recorded NYC run on November 17, following residents A and B through initialization and refinement. Their histories contain coffee, dining, and evening leisure activities, including prior co-occurrences in bar or coffee settings. The two cards show successive stages of the same run. Counts for C and Q refer to the full day.

Table 6: Inference token usage of the retained runs. M denotes million tokens; k denotes thousand tokens. Per-day usage is normalized by active resident-days (RD). † marks a calibrated estimate; — indicates unavailable token accounting.
<table><tr><td>Dataset</td><td>Method</td><td>Active RD</td><td>Total tokens (M)</td><td>Tokens/RD (k)</td></tr><tr><td rowspan="3"></td><td>ELLMob</td><td>14,279</td><td>130.058</td><td>9.11</td></tr><tr><td>ENCOUNTERMOB</td><td>12,887</td><td>587.586</td><td>45.60</td></tr><tr><td>ELLMob</td><td>2,017</td><td>14.645</td><td>7.26</td></tr><tr><td></td><td>ENCOUNTERMOB</td><td>2,017</td><td>49.823</td><td>24.70</td></tr></table>

## Example: Initial generation

City-level encounter rhythms (C): Three coarse allocations: Restaurant at 14:00, Bar at 19:00, and Italian Restaurant at 21:00.

Encounter proposals (Q): Three proposals are instantiated. The proposal for A and B places them at Bar v2171 at 22:30.

## Personal trajectories generated:

A: 11:10 v2003 → 15:20 v1548 → 15:50 Cofee Shop (v2019) → 16:30 v2024.

B: 00:50 v1998 → 15:50 Cofee Shop (v2019) → 16:00 v2020.

Realized encounter: Both independently generated trajectories contain Cofee Shop v2019 at 15:50. They realize this cofee-shop co-occurrence, rather than the proposed 22:30 bar encounter.

## Example (continued): Critic feedback and refinement

Joint Critic feedback: REVISE. Retain the supported cofee-shop stops and make their realized encounter available to the proposal update. Historical preferences do not justify forcing unsupported precise proposals.

Personal trajectories: Both residents’ complete daily sequences remain unchanged, including the shared 15:50 stop.

Updated rhythms (C): Two coarse allocations remain: Restaurant at 18:00 and Bar at 21:00.   
They express broad tendencies rather than required events.

Updated proposal (Q): One proposal remains: A and B at Cofee Shop v2019 at 15:50. The exact intersection of their trajectories provides more direct support than the coarse evening tendencies.

Final assessment: ACCEPT. The run ends after one refinement round with one proposal and one exactly corresponding realized encounter. Proposal–trajectory agreement improves; the realized encounter itself is unchanged.

Retrospective comparison. The observed records place A and B together at a Burger Joint at 14:50. The initial bar proposal differs by 460 minutes and approximately 6.26 km; the revised coffee-shop proposal differs by 60 minutes and approximately 1.70 km. The revised proposal is closer but does not match exactly. These observations are used only for this retrospective comparison, not for the Critic’s decision.

Two additional runs illustrate how city rhythms become concrete encounters and how local trajectory revisions realize them. Resident labels are local to each example. The cards summarize saved inputs and outputs; R0 denotes initialization and R1–R3 denote refinement rounds.

## Example: From city rhythms to concrete encounters

Context: NYC, December 29, 2012. The city module allocates two encounters: one at an American Restaurant in the 20:00 hour and one at a Bar in the 22:00 hour.

## Example: From city rhythms to concrete encounters (continued)

Historical support: A and B previously co-occurred at American Restaurant v4179 on Dec. 21 at 22:10. C and D previously co-occurred at Bar v869 on Dec. 27 at 22:40. Both observations precede the target day.

Concrete proposals (Q): The encounter module proposes A and B at v4179 at 20:00, and C and D at v869 at 22:40. The participant sets and venues have historical support; the restaurant proposal uses an earlier time than the cited historical encounter. Both proposals fall within their respective city-level hour/category allocations.

Realization across rounds: Neither proposal is realized by the R0 trajectories. Both are realized at their exact proposed participant sets, times, and venues in R1, and remain realized through R3. The full-day Critic decision becomes ACCEPT at R3.

Mechanism illustrated: Coarse city-level allocations are resolved into diferent participant sets and specific venues, with timing adapted where needed. Subsequent personal refinement makes those proposals appear in the generated trajectories.

## Example: Realizing an encounter with a local revision

Context: Japan, July 1, 2012. The encounter module proposes A and B at 14:00, Café v18324. Their supplied history contains a co-occurrence at this café at 14:00 on June 3, before the target day.

## Initial trajectories (R0):

A: 08:40 Train Station (v1764) → 16:00 Men’s Store (v10582) → 19:20 Train Station (v1764).

B: 14:00 Café (v18324).

The proposal is not realized because A has no corresponding café stop.

Critic feedback (paraphrased): Add a stop at v18324 at 14:00 to realize the encounter with B; the other existing stops can be retained.

Revised trajectories (R1): A adds 14:00 Café (v18324) while preserving all three original stops at their original times and venues. B’s complete trajectory remains unchanged. The proposed encounter is now realized and persists through R3.

Scope of the outcome: This local inconsistency is resolved, although the full-day Critic decision remains REVISE at R3. The example illustrates targeted encounter realization with a single added stop, rather than a full rewrite of either resident’s day.