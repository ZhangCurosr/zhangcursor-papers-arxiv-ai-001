## Highlights

ONCship: A Large-Scale Underwater Acoustic Corpus for Ship Recognition Zhe Cao, Jichao Zhang, Xiao-Lei Zhang

• ONCship contains 4521h of underwater recordings from six ONC stations.

• Domain-specific pre-training improves DeepShip NMI and retrieval AUC.

• Pre-training benefits vary across metrics and evaluation datasets.

# ONCship: A Large-Scale Underwater Acoustic Corpus for Ship Recognition

Zhe Cao<sup>a,b,1</sup>, Jichao Zhang<sup>a,b,1</sup> and Xiao-Lei Zhang<sup>a,b,∗</sup>

<sup>a</sup>School of Marine Science and Technology, Northwestern Polytechnical University, Xi’an, 710072, China

<sup>b</sup>Research and Development Institute of Northwestern Polytechnical University in Shenzhen, Shenzhen, 518063, China

## A R T I C L E I N F O

Keywords:   
Underwater acoustic target recognition   
Underwater acoustic dataset   
Ship classification

## A BS T RA C T

Recent advances in deep learning have significantly improved underwater acoustic vessel recognition. However, existing publicly available datasets are typically limited in scale, hindering large-scale representation learning and downstream task evaluation. To address this limitation, we introduce ONCship, a large-scale underwater acoustic dataset comprising 4521 hours of curated recordings from Ocean Networks Canada (ONC) hydrophones and Automatic Identification System (AIS) data. We evaluate the proposed dataset across three tasks: self-supervised representation learning, vesseltype classification, and MMSI-Level Vessel Classification. Experiments on DeepShip and ShipsEar demonstrate the efectiveness of ONCship for acoustic representation learning, with joint pre-training on ONCship and AudioSet achieving a classification accuracy of 62.45% on DeepShip through linear probing. Furthermore, incorporating ONCship as auxiliary training data improves downstream vesseltype classification performance. Beyond conventional vessel-type classification, we further investigate MMSI-Level Vessel Classification, treating each vessel as a distinct identity. Experimental results show that models fine-tuned with diferent loss functions achieve Top-1 accuracies exceeding 53% and Top-5 accuracies approaching 69%, demonstrating the feasibility of distinguishing individual vessels through MMSI-Level Vessel Classification. Overall, ONCship provides a valuable resource for underwater acoustic representation learning and vessel recognition, while its rich vessel-level metadata enables further exploration of MMSI-Level Vessel Classification.

## 1. Introduction

Underwater acoustic target recognition uses hydrophone recordings to identify vessel types and other acoustic sources. It supports maritime surveillance, vessel trafic management, fisheries enforcement, and ocean noise monitoring Aslam, Zhang, Liu, Irfan, Xu, Li, Zhang, Zheng and Li (2024). Passive acoustic monitoring does not require illumination and can complement radar and optical sensing, but sound propagation and weather-dependent ambient noise afect the recorded signals Hummel, Van Der Mei and Bhulai (2024). Vessel-type classification describes broad categories, whereas individual-vessel identification can link repeated observations of a known vessel, a finer-grained task analogous to speaker identification in speech processing. Identity recognition should therefore be evaluated across temporally separated encounters rather than neighboring clips from the same source recording. Both task levels require training data that cover this acoustic variation.

Limited annotated data constrain underwater representation learning and evaluation. Widely used benchmarks such as DeepShip Irfan, Zheng, Ali, Iqbal, Masood and Hamid (2021) and ShipsEar Santos-Domínguez, Torres-Guijarro, Cardenal-López and Pena-Gimenez (2016) together provide only tens of hours, whereas general-audio corpora such as AudioSet contain thousands of hours Gemmeke,

Ellis, Freedman, Jansen, Lawrence, Moore, Plakal and Ritter (2017). For supervised vessel-type classification, limited independent labeled recordings also restrict the diversity available during training Huang, Sun, Lu, Xu, Xiu and Zhang (2026). Supplementary recordings can expand this coverage, but their value depends on whether they support the benchmark’s recognition task. Larger underwater resources are emerging: UniqueShip contains 2460h of shipradiated audio from 4218 vessels, or 3437h including background recordings, and separates vessel identities across benchmark partitions Hashemi, Stout, Hoogs and Parham (2026). Its analysis shows that identity overlap can inflate estimates of generalization in vessel-type classification. The Maritime Mobile Service Identity (MMSI) therefore serves both as a grouping key for identity-aware type-classification evaluation and as a label for individual-vessel identification. Expanding labeled training data and defining identity-level tasks both require attention to these metadata and evaluation boundaries.

Self-supervised learning ofers another use of large audio archives: learning representations without taskspecific labels during pre-training. Underwater applications include masked-token learning Feng, Zhu and Ma (2024) and contrastive learning on separate unlabeled ONC recordings Hummel, Gansekoele, Bhulai and van der Mei (2026). Their transfer depends on the training distribution: underwater recordings contain low-frequency ship-noise components shaped by propagation and ambient noise Hummel et al. (2024), while general-audio corpora cover a broader range of sound events. This diference motivates comparing general-audio, underwater, andjoint pre-training.

![](images/b0041a050d4e43c0651c0004a0f2ef2e74979b96318cdc8456e328a3057783df.jpg)  
Figure 1: Example log-mel spectrograms for the four most frequent vessel types in ONCship: Tug, Cargo, Passenger, and Towing. Each panel shows one recording.

Together with the labeled-data and identity requirements above, it defines three complementary questions: how pre-training sources afect transfer, how additional labeled recordings afect vessel-type classification, and whether acoustic information distinguishes known vessels across temporally separated encounters.

To address these three questions, we construct ONCship from the long-term hydrophone recordings and vesseltrafic observations provided by ONC Heesemann, Insua, Scherwath, Juniper and Moran (2014). Automatic Identification System (AIS) messages provide vessel identifiers, types, and positions for association with acoustic recording intervals. Starting from 20000h of raw recordings, trajectory processing, spatial association, and temporal selection yield a 4521h corpus. It includes AIS-associated singleship and multi-ship intervals, background intervals, and recordings for which AIS data were not downloaded. For self-supervised pre-training, we exclude recordings from the FAE and LSBBL stations used by DeepShip, leaving 3905h of ONCship data. Figure 1 presents log-mel spectrogram examples for the four most frequent vessel types in ONCship.

Using ONCship, we evaluate the three questions under distinct protocols. First, we examine how pre-training sources afect representation quality and classification transfer, using the same Eficient Audio Transformer (EAT) architecture and objective for AudioSet-only, ONCshiponly, and joint pre-training Chen, Liang, Ma, Zheng and Chen (2024). Second, we assess how supplementary labeled recordings afect DeepShip vessel-type classification, comparing models trained on the benchmark alone with those receiving additional ONCship data from the same or diferent hydrophones, with supplementary MMSIs entirely disjoint from DeepShip. Third, we test whether acoustic information supports identification of known vessels across temporally separated encounters, using MMSI labels and keeping all clips from the same source recording within one data partition.

The contributions of this study are threefold:

• We construct ONCship, a 4521h underwater acoustic corpus curating single-ship, multi-ship, and ambient recordings with rich AIS vessel-type and MMSI metadata, providing a large-scale resource for both labelfree pre-training and downstream supervised tasks.

• We systematically evaluate ONCship across selfsupervised representation learning, vessel-type classification, and MMSI-Level Vessel Classification. The experiments cover general-audio, domainspecific, and joint pre-training, matched and mismatched hydrophone deployments, and finegrained discrimination of individual vessels across diferent voyages and temporal encounters.

Section 2 reviews related work, Section 3 describes ONCship construction, and Section 4 reports the pretraining, vessel-type classification, and MMSI identification experiments. Section 5 summarizes the findings and limitations.

## 2. Related Work

## 2.1. Underwater Acoustic Datasets

Underwater acoustic datasets have evolved from relatively small benchmarks designed for vessel-type classification to larger corpora with richer vessel annotations, more diverse acoustic scenes, and long-term recordings. Existing datasets difer substantially in recording duration, vessel taxonomy, background annotations, acquisition environments, and data construction strategies Aslam et al. (2024). Representative resources include ShipsEar, DeepShip, QiandaoEar22, Oceanship, IARA, and UniqueShip, while longterm observatories such as ONC provide the raw hydrophone and AIS data from which several large-scale datasets have been constructed.

ONC: ONC operates long-term underwater observatories that provide open-access hydrophone recordings together with AIS information Heesemann et al. (2014). These recordings cover extended periods and contain diverse vessel activities, environmental noise, and complex multi-source acoustic scenes. Several widely used ship-radiated noise datasets, including DeepShip, VTUAD, Oceanship, and UniqueShip, have been constructed from subsets of the ONC archive.

ShipsEar: ShipsEar is a widely used underwater vessel classification dataset collected near the port of Vigo, Spain Santos-Domínguez et al. (2016). It contains approximately 2.8h of recordings from eleven vessel types. The detailed vessel labels can also be grouped into broader categories, making ShipsEar a common benchmark for evaluating feature extraction and supervised vessel classification methods.

DeepShip: DeepShip is a ship-radiated noise dataset constructed from ONC recordings acquired between 2016 and 2018 Irfan et al. (2021). It contains approximately 47.1h of recordings from vessels passing near the hydrophones and defines four vessel categories: cargo ships, passenger ships, tankers, and tugboats. DeepShip has been widely adopted as a benchmark for underwater acoustic target recognition and vessel classification.

Table 1  
Selected underwater acoustic datasets. Durations are approximate totals in hours, including background where available.
<table><tr><td>Dataset</td><td>Year</td><td>Duration (h)</td><td>Label scope</td><td>Source</td></tr><tr><td>ShipsEar Santos-Domínguez et al. (2016)</td><td>2016</td><td>2.8</td><td>11 vessel types</td><td>Vigo, Spain</td></tr><tr><td>DeepShip Irfan et al. (2021)</td><td>2021</td><td>47.1</td><td>4 vessel types</td><td>ONC, Canada</td></tr><tr><td>QiandaoEar22 Du and Hong (2024)</td><td>2024</td><td>31.4</td><td>20 target labels</td><td>Qiandao Lake, China</td></tr><tr><td>Oceanship Li et al. (2024)</td><td>2024</td><td>121.2</td><td>15 vessel types</td><td>ONC, Canada</td></tr><tr><td>IARA da Silva et al. (2025)</td><td>2025</td><td>129.6</td><td>34 vessel types</td><td>Santos Basin, Brazil</td></tr><tr><td>UniqueShip Hashemi et al. (2026)</td><td>2026</td><td>3437.0</td><td>11 vessel types</td><td>ONC, Canada</td></tr><tr><td>ONCship (Ours)</td><td>2026</td><td>4521.0</td><td>22 vessel types</td><td>ONC, Canada</td></tr></table>

QiandaoEar22: QiandaoEar22 was collected in Qiandao Lake, China, and contains approximately 31.4h of ship and background recordings Du and Hong (2024). The dataset contains 20 target labels and includes both singletarget and multi-target acoustic scenes. Its multi-target recordings support recognition tasks in which multiple acoustic sources may be simultaneously present.

Oceanship: Oceanship is a ship-radiated noise dataset curated from the ONC repository using recordings collected between July 2020 and February 2021 Li, Xiang, Yu, Gao, Ruan, Hu, Liu and Fu (2024). It contains approximately 121.2h of audio and extends the vessel taxonomy beyond the four broad categories commonly used in earlier ONCderived datasets, providing fifteen vessel-type categories. The dataset combines hydrophone recordings with AIS information to support supervised vessel-type classification.

IARA: IARA contains approximately 129.6h of underwater acoustic recordings collected in the Santos Basin, Brazil da Silva, Fernandes, Soares Filho, de Oliveira e Souza Filho, Spengler and de Moura Júnior (2025). It provides annotations for 34 detailed vessel types that are further organized into eight general vessel groups. The dataset also includes background recordings, providing a relatively broad range of vessel categories and underwater acoustic conditions.

UniqueShip: UniqueShip is a large-scale dataset constructed from long-term ONC recordings Hashemi et al. (2026). It contains approximately 3437h of audio, including background recordings, and includes recordings associated with 4218 distinct vessels. Its benchmark defines eleven vessel types together with a background class.

## 2.2. Underwater Acoustic Target Recognition

Underwater acoustic target recognition has progressed from handcrafted signal descriptors to learned representations. Traditional systems combine spectral features, including LOFAR line spectra, DEMON envelope spectra, and Mel-frequency cepstral coeficients, with conventional classifiers Aslam et al. (2024). These descriptors encode properties of ship-radiated noise, but their efectiveness depends on feature selection and recording conditions. Convolutional networks learn patterns in time-frequency inputs Xie, Ren and Xu (2024), while Transformer-based models capture relationships across the input sequence Wu, Yao, Zhao, Zhao, Zang and Wang (2025). This progression shifts more of the representation design into the model, increasing the importance of training data that cover the acoustic variation encountered at inference.

Beyond conventional vessel-type classification, several studies have explored finer-grained vessel recognition from underwater acoustic signals. Karakos et al. formulated individual-ship detection as an open-class detection problem analogous to speaker identification Karakos, Silovsky, Schwartz, Hartmann and Makhoul (2018). More recently, Du and Hong studied specific-target identification using the QiandaoEar22 dataset Du and Hong (2024). These studies demonstrate the potential of underwater acoustic signals for recognition beyond coarse vessel-type classification.

Changes to model architecture alone do not remove the constraint of limited independent labeled recordings. Regularization and data augmentation seek to improve robustness Xie et al. (2024); Aslam et al. (2024), while few-shot learning targets recognition with fewer labeled examples Huang et al. (2026). Nevertheless, high-capacity supervised models remain at risk of overfitting when the available data are small or insuficiently diverse Xie et al. (2024). Expanding a network’s capacity cannot substitute for expanding the range of acoustic conditions represented during learning. This bottleneck motivates learning representations from the much larger unlabeled archives, with the limited labeled data used to adapt or evaluate those representations on specific recognition tasks.

## 2.3. Self-Supervised Audio Pre-Training and Underwater Adaptation

Self-supervised audio pre-training separates representation learning from task-specific annotation by constructing learning targets from the audio itself. Reconstruction-based methods recover masked spectrogram content Huang, Xu, Li, Baevski, Auli, Galuba, Metze and Feichtenhofer (2022), whereas token-prediction methods use clustered or learned acoustic units as targets Liu, Chang, Auli, Hsu and Glass (2023); Chen, Wu, Wang, Liu, Tompkins, Chen and Wei (2022). Teacher-student methods instead learn by matching representations of related or masked inputs Baevski, Babu, Hsu and Auli (2023); Chen et al. (2024). These objectives impose diferent constraints on the representation while allowing large unlabeled corpora to be used during pretraining. They reduce dependence on labels at that stage, but the resulting features still require downstream evaluation; successful general-audio pre-training does not establish their suitability for every underwater recognition setting.

Underwater adaptation must account for a training distribution shaped by ship-radiated noise, propagation, and ambient sound rather than speech or general sound events Hummel et al. (2024). Domain-specific self-supervised learning has already been investigated: Feng et al. (2024) apply hierarchical masked-token learning to underwater target recognition, and Hummel et al. (2026) learn contrastive representations from separate unlabeled ONC recordings and evaluate frozen features on downstream ship and marinemammal classification tasks. These studies establish the feasibility of learning underwater representations without taskspecific labels during pre-training. These studies motivate comparing general-audio, underwater-audio, and joint pretraining at thousand-hour scale under a common learning framework.

Addressing this question requires a common model and objective so that the comparison does not also change the recognition architecture. We use the EAT, which combines an eficient self-supervised pre-training framework with prediction of teacher representations at both frame and utterance levels Chen et al. (2024). Its general-audio formulation provides a basis for comparing AudioSet, ONCship, and their combination using the same encoder architecture and pre-training objective. This focuses the study on the choice and combination of training data. Downstream evaluation then examines whether any observed benefits extend across benchmarks, metrics, and adaptation protocols, rather than presuming that domain-specific pre-training is uniformly advantageous.

## 3. Dataset Construction

ONCship is constructed by curating an archive of 20000h of raw hydrophone recordings into a 4521h underwater acoustic corpus. Figure 2 illustrates five steps in this process: hydrophone and AIS data collection, trajectory cleaning and interpolation, spatial association, temporal labeling, and segment merging with clip sampling. The workflow combines acoustic recordings with vessel-trafic observations to identify recording intervals, associate vessel metadata where available, and organize the selected audio into training-ready subsets. Section 3.1 describes the recording stations and AIS sources, Section 3.2 details the trajectory-based association and audio selection procedure, and Section 3.3 reports the corpus composition and vessel-type statistics.

## 3.1. Data Source

Recordings are obtained from ONC, a cabled ocean observatory operated by the University of Victoria that maintains instrument platforms in the coastal and ofshore waters of British Columbia, Canada, spanning the Strait of Georgia, the waters around Vancouver Island, and the northeast Pacific Ocean Ocean Networks Canada (n.d.). The observatory streams continuous data from underwater sensors, including broadband hydrophones, and pairs these recordings with vessel trafic information received by shore-based AIS stations.

The dataset draws on recordings from six hydrophone stations, identified by the codes DIIP, FAE, LSBBL, BIIP, CRIP, and USSLP, located between 49.08<sup>◦</sup> N and 54.26<sup>◦</sup> N; Figure 3 shows their locations together with the AIS receivers. Each station hosts one or more icListen hydrophones deployed over successive periods, and the complete deployment history is listed in Table 2. In total, the recordings span March 2016 to May 2026, with individual deployments lasting from several months to a few years.

All recordings are resampled to a common rate of 32 kHz to standardize the waveform sampling grid across deployments. This rate gives a nominal Nyquist frequency of 16 kHz, encompassing the low-frequency tonal components commonly used in ship-noise analysis Hummel et al. (2024). For higher-rate inputs, resampling also reduces the number of samples processed per clip and the associated waveform storage and processing requirements.

Vessel identity and type are obtained from the shorebased AIS receivers (IONA, CRSS and DISS) listed in Table 2. The AIS messages provide the MMSI, vessel type, and position of each transmitting vessel, which we use to associate recordings with specific ships. We selected the US-SLP recordings for self-supervised pre-training, which does not require vessel labels, and therefore did not download the corresponding AIS records. These recordings are retained without AIS-derived vessel associations and reported as a separate subset in Section 3.3.

## 3.2. Data Selection

The 20000h archive contains vessel activity and ambient underwater sound. For stations with downloaded AIS records, we associate recording intervals with vessel trajectories and select single-ship, multi-ship, and no-ship segments using the rules below. USSLP recordings are retained separately because their AIS records were not downloaded. This subsection describes the AIS-based procedure that produces the audio clip process plan.

For each hydrophone, dynamic AIS records within a 50km radius are collected and grouped by MMSI into singlevessel trajectories. Trajectories of stationary vessels are discarded. A trajectory is treated as stationary when it contains fewer than two points, its mean speed-over-ground is below 1 knot, at least 80% of its points fall below 1 knot, or its end-to-end displacement is less than 0.3km.

For the remaining trajectories, intermediate points are inserted at 30s intervals when the gap between consecutive AIS observations lies between 30s and 30 min. Each scalar attribute � (latitude, longitude, or speed-over-ground) is interpolated separately as $q ( t ) = q _ { i } + \alpha ( q _ { i + 1 } - q _ { i } )$ , where $q _ { i }$ and $q _ { i + 1 }$ are its observed values at times $t _ { i }$ and $t _ { i + 1 }$ , and $\alpha = ( t - t _ { i } ) / ( t _ { i + 1 } - t _ { i } )$ . Interpolation times are $t = t _ { i } + k \tau .$

![](images/b9b6d4fdff39866d89e10ef7446b9476db299eee81e1809e1c762dacdb5a440b.jpg)

Figure 2: ONCship construction pipeline. The five steps collect hydrophone and AIS data, clean and interpolate vessel trajectories, associate trajectories with hydrophones, label 30s time bins, and merge segments before sampling 60 s clips. Sampled clips are subsequently subdivided into 10s clips.  
![](images/ab8a00ce263498551ffc0d94e83955dd4d8d895215573f03e8f4bec8877992e8.jpg)  
Figure 3: Locations of the ONC hydrophone stations and associated AIS receivers. Left: DIIP and DISS. Center: CRIP and CRSS. Right: the Vancouver-area stations and IONA. Base map: OpenStreetMap contributors, available under the Open Database License (https://www.openstreetmap.org/copyright).

with � = 30 s and positive integers � satisfying $t _ { i } < t < t _ { i + 1 } ;$ the endpoints remain the original AIS observations. Surface distances between the vessel positions and the hydrophone coordinates are then computed using the Haversine formula, with latitude and longitude converted to radians.

Two kinds of vessel-presence intervals are then extracted. Close-range intervals are continuous segments within 2km of the hydrophone; among them, only segments whose duration does not exceed 1.5h, whose mean speed-over-ground is at least 3 knots, and whose proportion of points below 3 knots does not exceed 40% are retained and marked as vessel-present. Mid-range intervals are continuous segments whose distance to the hydrophone lies between 2km and 50km; these are marked as mid-range vessel present without further filtering. The 2km association radius is chosen to favor recordings in which vessel sound is more distinguishable from ambient noise, since a shorter propagation path generally reduces transmission loss. It is a signal-to-noise consideration rather than a measured acoustic detection boundary: received levels also depend on vessel source strength and local propagation conditions da Silva et al. (2025).

Table 2  
Complete hydrophone deployment history with the associated AIS receivers. Coordinates are decimal degrees; positive latitude denotes north and negative longitude west. A dash denotes that AIS records were not downloaded.
<table><tr><td>Station</td><td>Lat., Lon. (°)</td><td>Device ID</td><td>Recording period</td><td>AIS</td><td>AIS Lat., Lon. (°)</td></tr><tr><td>DIIP</td><td>54.259, -130.431</td><td>ICLISTENHF1352 ICLISTENHF1352 ICLISTENHF1354 ICLISTENHF1352 ICLISTENHF1293 ICLISTENHF1293</td><td>2016.03.30-2017.04.06 2017.04.07-2018.10.25 2018.10.26-2019.08.29 2019.08.29-2021.06.02 2021.06.02-2021.10.12</td><td>DISS</td><td>54.263, -130.425</td></tr><tr><td></td><td>49.081, -123.339</td><td>ICLISTENHF6406 ICLISTENHF1252 ICLISTENAF2523</td><td>2023.09.16-2025.09.12 2025.09.12-2026.05.13 2017.06.24-2017.11.03</td><td></td><td></td></tr><tr><td>FAE LSBBL</td><td>49.081, -123.339</td><td>ICLISTENAF2523 ICLISTENAF2523</td><td>2017.11.04-2018.10.04 2016.05.02-2017.06.24</td><td>IONA IONA</td><td>49.216,-123.205 49.216,-123.205</td></tr><tr><td>BIIP</td><td>49.301,-123.111</td><td>ICLISTENHF1951 ICLISTENHF1354 ICLISTENHF1354 ICLISTENHF1354</td><td>2020.10.22-2022.06.17 2022.10.04–2023.07.25 2023.07.25-2024.02.06 2024.02.06-2025.01.24</td><td>IONA</td><td>49.216, -123.205</td></tr><tr><td>CRIP</td><td>50.021, -125.235</td><td>ICLISTENHF1350 ICLISTENHF1350 ICLISTENHF1350 ICLISTENHF1333 ICLISTENHF1333 ICLISTENAF2523 ICLISTENHF1350</td><td>2016.03.18-2016.07.28 2016.07.29-2017.01.24 2017.01.26-2018.01.30 2018.02.14-2018.07.17 2018.11.15-2020.07.14 2020.07.15-2022.01.28</td><td>CRSS</td><td>50.021, –125.235</td></tr><tr><td></td><td></td><td>ICLISTENHF1349 ICLISTENHF6017</td><td>2022.01.28-2024.03.18 2024.03.19-2025.03.24</td><td></td><td></td></tr><tr><td>USSLP</td><td>49.085, -123.330</td><td>ICLISTENAF2556</td><td>2025.08.01-2026.05.13 2018.10.04-2019.02.28</td><td></td><td></td></tr></table>

The observation period is divided into 30s time bins. A bin covered by a retained close-range interval is added to the near-vessel set, and a bin covered by a mid-range interval is marked as mid-range. Bins with no near or mid-range vessel are labeled no-ship; bins with no near vessel but a mid-range vessel are excluded; bins with exactly one near vessel are labeled single-ship and record its MMSI, mean speed, and type; bins with more than one near vessel are labeled multi-ship and record the vessel count. Excluding mid-range-only bins reduces the risk of weaker, more distant shipping sounds entering the background negative samples even when no close-range vessel is present. The no-ship label is defined by these AIS-based rules, not by verified absence of all shipping noise. Single-ship and multi-ship refer to the number of retained near-vessel trajectories, not to all vessels within 50km.

Consecutive bins with identical status, and, for singleship bins, identical MMSI, whose time gap does not exceed 30s, are merged into segments. Segments from diferent devices and days of the same hydrophone are then combined: no-ship and multi-ship segments are merged directly, while single-ship segments are divided into those with and without valid vessel-type information. Two segments are merged when the end time of one equals the start time of the next and all metadata except time are identical. The total duration of each category is then accumulated.

Fixed-duration audio clips are sampled randomly from the merged timeline for each category. Within a segment of duration �, a clip of duration $D = 6 0$ s must lie entirely inside the segment, so its start ofset is sampled from [0, � − �] when $\mathit { \textbf { L } } \geq \mathit { \textbf { D } }$ . Accepted clip start times within the segment are separated by at least $2 D = 1 2 0 \mathrm { s }$ . For singleship segments, at most 1500 clips are sampled per MMSI. The output is a per-category clip list with total durations, which forms the audio clip download.

## 3.3. Dataset Distribution

We further subdivided the extracted 60s segments into 10s clips. Table 3 presents the final duration of each category in the dataset. Specifically, single-ship recordings with vessel-type metadata account for 3106.8 hours, multi-ship recordings for 929.1 hours, single-ship recordings without vessel-type metadata for 38.4 hours, and no-ship recordings for 7.4 hours. The 439.3 hours of recordings from USSLP are reported separately, as the corresponding AIS data were not downloaded. Together, these categories comprise a total of 4521 hours of underwater acoustic recordings.

![](images/7fb5d7a407b668b24a4c0c039a0f2be88796780332772e53325ea11e3914341a.jpg)  
(a) Segment categories

![](images/c1235f1f9d174a11e1ff235fb86bf6f77a247653931cbd03276d5fe08c67e9b9.jpg)  
(b) Major vessel types  
Figure 4: ONCship distribution. (a) Duration by recording category. (b) Major vessel-type labels among single-ship recordings with type metadata; Others aggregates all remaining labels, including Reserved and Unknown.

Total duration of each segment category after filtering and clipping.
<table><tr><td>Category</td><td>Duration (h)</td></tr><tr><td>Single-ship, labeled</td><td>3106.8</td></tr><tr><td>Single-ship, unlabeled</td><td>38.4</td></tr><tr><td>Multi-ship</td><td>929.1</td></tr><tr><td>No-ship</td><td>7.4</td></tr><tr><td>USSLP (AIS not downloaded)</td><td>439.3</td></tr></table>

Table 4 presents all 24 vessel-type labels from the original metadata. After excluding the Reserved and Unknown labels from the class count and supervised classification, 22 vessel types remain, as reported in Table 1. Among these, Tug, Cargo, Passenger, and Towing account for the majority of clips with vessel-type metadata. However, the number of clips varies by several orders of magnitude across vessel types, resulting in a pronounced long-tailed distribution.

## 4. Experiments

In this section, we first introduce the evaluation metrics and then conduct experiments from three perspectives to demonstrate the efectiveness of our dataset. First, we investigate whether large-scale underwater vessel recordings can improve the feature representation capability of pretrained models. Second, we examine whether the collected data can improve downstream vessel-type classification performance. Finally, we evaluate the feasibility of the fine-grained MMSI-Level Vessel Classification task, which aims to distinguish individual vessels rather than only coarse vessel types.

Number of 10 s clips per raw vessel-type metadata label. SAR: search and rescue; HSC: high-speed craft; WIG: wing-inground craft; RR18: noncombatant vessels covered by Radio Regulations Resolution No. 18 (AIS ship-type code 59) Guo, Xu, Li and Wingfield (2023).
<table><tr><td rowspan=1 colspan=1>Metadata label                                         Clips</td></tr><tr><td rowspan=1 colspan=1>Tug                                                   237238</td></tr><tr><td rowspan=1 colspan=1>Cargo                                                 191287</td></tr><tr><td rowspan=1 colspan=1>Passenger                                             188183</td></tr><tr><td rowspan=1 colspan=1>Towing                                                182301</td></tr><tr><td rowspan=1 colspan=1>Pleasure                                               117036</td></tr><tr><td rowspan=1 colspan=1>Fishing                                                81616</td></tr><tr><td rowspan=1 colspan=1>Sailing                                                 38115</td></tr><tr><td rowspan=1 colspan=1>Tanker                                                 17978</td></tr><tr><td rowspan=1 colspan=1>Pilot                                                   14888</td></tr><tr><td rowspan=1 colspan=1>SAR                                                   12942</td></tr><tr><td rowspan=1 colspan=1>Other                                                  10928</td></tr><tr><td rowspan=1 colspan=1>Port tender                                             5307</td></tr><tr><td rowspan=1 colspan=1>Local                                                   5003</td></tr><tr><td rowspan=1 colspan=1>Military                                                3834</td></tr><tr><td rowspan=1 colspan=1>Reserved                                               3232</td></tr><tr><td rowspan=1 colspan=1>Anti-pollution                                          2828</td></tr><tr><td rowspan=1 colspan=1>Dredging                                               1666</td></tr><tr><td rowspan=1 colspan=1>Law enforcement                                       1187</td></tr><tr><td rowspan=1 colspan=1>Diving                                                  1074</td></tr><tr><td rowspan=1 colspan=1>HSC                                                    1012</td></tr><tr><td rowspan=1 colspan=1>WIG                                                    515</td></tr><tr><td rowspan=1 colspan=1>Unknown                                               126</td></tr><tr><td rowspan=1 colspan=1>Medical                                                 120</td></tr><tr><td rowspan=1 colspan=1>RR18                                                    18</td></tr></table>

## 4.1. Evaluation Metrics

We evaluate classification performance using accuracy (ACC), top-5 accuracy (Top-5 ACC), weighted F1-score (Weighted-F1), macro F1-score (Macro-F1), and macro recall (Macro-Recall). ACC measures the overall proportion of correctly classified samples, while Top-5 ACC considers a prediction correct if the ground-truth class is among the five highest-scoring classes. Top-5 ACC is only reported for the MMSI-Level Vessel Classification task, where the number of classes exceeds five.

For class $c ,$ let $T P _ { c } , F P _ { c }$ , and $F N _ { c }$ denote the numbers of true positives, false positives, and false negatives. The class-wise recall and F1-score are

$$
\mathrm { R e c a l l } _ { c } = \frac { T P _ { c } } { T P _ { c } + F N _ { c } } , \qquad \mathrm { F l } _ { c } = \frac { 2 T P _ { c } } { 2 T P _ { c } + F P _ { c } + F N _ { c } } .
$$

Macro-Recall and Macro-F1 are obtained by averaging the corresponding class-wise scores equally over all classes, while Weighted-F1 weights each class according to its number of samples. These metrics complement ACC by providing a more balanced evaluation under class imbalance.

For frozen representation evaluation, we report normalized mutual information (NMI) and retrieval AUC. NMI measures the agreement between K-means cluster assignments of the extracted features and the ground-truth labels. Retrieval AUC is computed from pairwise cosine similarities between frozen representations. Each sample is treated as a query, with samples from the same class regarded as positives and samples from diferent classes as negatives. A ROC-AUC score is computed for each valid query and then averaged:

$$
\operatorname { A U C } = { \frac { 1 } { | \mathscr { Q } | } } \sum _ { i \in \mathscr { Q } } \operatorname { A U C } _ { i } ,
$$

where  denotes queries containing both positive and negative candidates. This AUC therefore evaluates the discriminability of the learned representation rather than classifier probabilities.

## 4.2. Models

To evaluate the efectiveness and generalizability of ONCship, we conduct a series of experiments using established and competitive audio learning architectures across diferent training paradigms and downstream tasks. Rather than proposing a new model architecture, our objective is to investigate how large-scale underwater acoustic data can benefit audio representation learning and downstream underwater sound recognition. Specifically, we select EAT and BEATs as representative transformer-based architectures, and include a previously reported convolutional baseline for additional comparison.

Eficient Audio Transformer (EAT) Chen et al. (2024) is a self-supervised audio representation learning framework based on teacher-student bootstrapping and the Utterance-Frame Objective (UFO). We adopt EAT to investigate the efectiveness of ONCship under three diferent pre-training configurations, as described in Section 4.3.

BEATs Chen et al. (2022) is a self-supervised audio pretraining framework that iteratively learns acoustic tokenizers and audio representations through masked discrete token prediction. We use BEATs to examine whether incorporating ONCship recordings improves downstream fine-tuning performance.

ResNet He, Zhang, Ren and Sun (2016) is a convolutional neural network architecture that employs residual connections to facilitate the training of deep networks. We include ResNet-based models as convolutional baselines to complement the transformer-based architectures. Specifically, ResNet18 is used for comparison with previously reported contrastive representation learning results, whereas ResNet50 is employed as a supervised baseline for evaluating the efectiveness of additional ONCship recordings in vessel-type classification.

## 4.3. Efectiveness of the Dataset for Self-Supervised Pre-Training

To evaluate the efectiveness of pre-training on diferent datasets, we compare three EAT models trained from scratch: EAT-as, pre-trained exclusively on AudioSet; EATonc, pre-trained exclusively on ONCship; and EAT-as&onc, pre-trained on the combination of both datasets. We assess the quality of the learned audio representations on downstream classification tasks by freezing the pre-trained encoders and training lightweight classifiers on their extracted features. This evaluation allows us to directly examine the contribution of diferent pre-training datasets without the influence of end-to-end fine-tuning. Additionally, we include the DeepShip and ShipsEar classification accuracy reported for LARS, a contrastive learning method proposed by Hummel et al. (2026), as an external reference.

All three configurations use the EAT architecture Chen et al. (2024), with a 12 layer transformer and a hidden dimension of 768. The AudioSet portion contains 5000h Gemmeke et al. (2017); the ONCship portion contains 3905h after excluding all recordings from FAE and LSBBL, the stations represented in DeepShip. This exclusion is by station, not by recording date, and applies to the pre-training corpus. The joint configuration contains 8905h. All pre-training audio is divided into 10s clips. Each model is randomly initialized and does not load an initial pre-trained checkpoint.

We adopt the Utterance-Frame Objective (UFO) proposed by Chen et al. (2024) for self-supervised pre-training, using an exponential-moving-average (EMA) teacher-student framework. The training objective combines frame-level and utterance-level mean-squared errors (MSE). The spectrogram patch-masking ratio is set to 0.8, and no additional data augmentation is applied during pre-training.

Training uses Adam with a cosine learning-rate schedule, a peak learning rate of $5 \times 1 0 ^ { - 4 }$ , 75000 warm-up steps, a batch size of 128, bf16 precision, and 30 epochs. Waveforms are resampled to 16 kHz and converted into 128-bin log-mel spectrograms using a 25 ms window and a 10 ms frame shift.

To evaluate the quality of the learned representations, we freeze the pre-trained encoder and apply temporal average pooling to obtain a clip-level feature vector. A multinomial logistic regression classifier is trained on the extracted features, with the regularization parameter � set to 1.0. For DeepShip, we follow the chronological split proposed by Hummel et al. (2026), using recordings collected before December 2017 for training and those collected from December 2017 onward for testing. We further ensure that the training and test sets are disjoint at the vessel level, with no vessel appearing in both partitions. For ShipsEar, we adopt an 8:2 split based on vessel identity, similarly ensuring vessel-level independence. Audio recordings are subsequently divided into 30s segments with a 15s overlap within each partition to prevent data leakage across the training and test sets.

As shown in Table 5, pre-training on ONCship alone achieves higher NMI and AUC than AudioSet on DeepShip, suggesting that domain-specific underwater acoustic data can improve the discriminative quality of learned representations. Combining AudioSet and ONCship further improves performance, achieving the highest AUC of 0.6540 and ACC of 62.45% on DeepShip. On ShipsEar, EAT-as achieves the best NMI and AUC, while EAT-as&onc obtains the highest ACC of 76.18%, outperforming EAT-as by 5.78 percentage points. These results indicate that ONCship provides complementary acoustic information to general-purpose audio datasets, and that joint pre-training can improve downstream classification performance, although the benefits vary across datasets and evaluation metrics.

Table 5  
Comparison of EAT models pre-trained from scratch on diferent datasets. NMI, AUC, and ACC denote normalized mutual information, feature-retrieval AUC, and classification accuracy, respectively. The best result in each column is highlighted in bold.
<table><tr><td rowspan="2">Model</td><td colspan="3">DeepShip</td><td colspan="3">ShipsEar</td></tr><tr><td>NMI</td><td>AUC</td><td>ACC (%)</td><td>NMI</td><td>AUC</td><td>ACC (%)</td></tr><tr><td>EAT-as</td><td>0.1877</td><td>0.6297</td><td>61.75</td><td>0.4280</td><td>0.7284</td><td>70.40</td></tr><tr><td>EAT-onc</td><td>0.2021</td><td>0.6437</td><td>61.35</td><td>0.3727</td><td>0.7127</td><td>62.36</td></tr><tr><td>EAT-as&amp;onc</td><td>0.1993</td><td>0.6540</td><td>62.45</td><td>0.3882</td><td>0.7264</td><td>76.18</td></tr></table>

Table 6

Comparison of downstream classification performance between EAT models and contrastive learning baselines. ACC and Weighted F1 denote classification accuracy and weighted F1-score, respectively. The best reported result in each column is highlighted in bold.
<table><tr><td rowspan="2">Model</td><td colspan="2">DeepShip</td><td colspan="2">ShipsEar</td></tr><tr><td>ACC (%)</td><td>Weighted F1 (%)</td><td>ACC (%)</td><td>Weighted F1 (%)</td></tr><tr><td>ResNet18 (Adam)</td><td>53.94</td><td>49.00</td><td>21.40</td><td>8.00</td></tr><tr><td>ResNet18 (LARS)</td><td>57.78</td><td>56.00</td><td>43.94</td><td>42.00</td></tr><tr><td>EAT-as</td><td>61.75</td><td>61.93</td><td>70.40</td><td>70.85</td></tr><tr><td>EAT-onc</td><td>61.35</td><td>61.13</td><td>62.36</td><td>61.85</td></tr><tr><td>EAT-as&amp;onc</td><td>62.45</td><td>62.49</td><td>76.18</td><td>77.25</td></tr></table>

Table 6 further compares the EAT models with the contrastive learning baselines reported by Hummel et al. (2026). All three EAT models outperform the ResNet18 baselines in terms of both accuracy and weighted F1-score on DeepShip and ShipsEar. In particular, EAT-as&onc achieves the best overall performance, with an accuracy of 62.45% and a weighted F1-score of 62.49% on DeepShip, exceeding ResNet18 (LARS) by 4.67 and 6.49 percentage points, respectively. The improvements are more pronounced on ShipsEar, where EAT-as&onc achieves 76.18% accuracy and 77.25% weighted F1-score, surpassing ResNet18 (LARS) by 32.24 and 35.25 percentage points, respectively. These results further demonstrate the efectiveness of EAT-based self-supervised pre-training and suggest that incorporating ONCship alongside general-purpose audio data can yield competitive representations for downstream underwater acoustic classification.

## 4.4. Efectiveness of the Dataset for Fine-Tuning

To further investigate the utility of ONCship in downstream classification tasks, we evaluate its contribution to supervised fine-tuning using three representative architectures: ResNet50, EAT, and BEATs. We consider three training configurations: DeepShip only, DeepShip augmented with

ONCship recordings from diferent hydrophone stations, and DeepShip augmented with ONCship recordings from the same hydrophone stations. These configurations allow us to investigate the contribution of additional underwater acoustic data and the influence of hydrophone sources on downstream classification performance.

All models are fine-tuned end-to-end with a linear classification head for 10 epochs using the Adam optimizer, a batch size of 16, a learning rate of $2 \times 1 0 ^ { - 5 }$ , and a weight decay of $1 0 ^ { - 6 } .$ . The final-epoch checkpoint is used for evaluation. DeepShip is partitioned by vessel identity to ensure that recordings from the same vessel do not appear in both training and test sets. Audio recordings are subsequently segmented into 30s clips with a 15s overlap within each partition. For the augmented configurations, additional ONCship recordings are incorporated into the training set according to their hydrophone sources, while the DeepShip test set remains unchanged. All experiments are repeated three times, and the mean and standard deviation are reported.

Table 7 presents the fine-tuning results under the three training configurations. BEATs consistently achieves the highest performance across all evaluation metrics, reaching an accuracy of 69.41% when augmented with ONCship recordings from the same hydrophone stations. For EAT, incorporating ONCship recordings from the same stations improves accuracy from 65.64% to 68.16%, while recordings from diferent stations yield only a marginal improvement to 65.95%. Similarly, BEATs benefits from additional ONCship data, with accuracy increasing from 68.11% to 68.85% and 69.41% for diferent and same hydrophone sources, respectively. In contrast, ResNet50 exhibits a decline in performance when additional ONCship data are introduced. These results suggest that ONCship can enhance the finetuning performance of pre-trained audio transformers, with the benefits depending on model architecture and the acoustic characteristics of the additional training data.

Table 7  
Fine-tuning performance on DeepShip under diferent training data configurations. ACC, F1, and Recall are reported as percentages (mean ± standard deviation over three runs). The highest mean within each training configuration is highlighted in bold.
<table><tr><td rowspan="2">Model</td><td colspan="3">DeepShip</td><td colspan="3">ONCship (Different Hydrophones)</td><td colspan="3">ONCship (Same Hydrophones)</td></tr><tr><td> $A C C \%$ </td><td> $\overline { { \mathsf { m a c r o } \mathrm { - } \mathsf { F 1 } \left( \% \right) } }$ </td><td> $\overline { { \mathsf { m a c r o - R e c a l l } \left( \% \right) } }$ </td><td> $A C C \mathrm { ~ ( \% ) ~ }$ </td><td> $\overline { { \mathsf { m a c r o } { \mathrm { - } } { \mathsf { F 1 } } \left( \% \right) } }$ </td><td>macro-Recall (%)</td><td> $A C C ( \% )$ </td><td> $\overline { { \mathsf { m a c r o } \mathrm { - } \mathsf { F 1 } \left( \% \right) } }$ </td><td>macro-Recall (%)</td></tr><tr><td>ResNet50</td><td> $\overline { { 3 9 . 4 5 \pm 7 . 9 4 } }$ </td><td> $\overline { { 3 0 . 5 3 \pm 7 . 0 3 } }$ </td><td> $\overline { { 3 6 . 7 8 \pm 5 . 7 6 } }$ </td><td> $\overline { { 3 7 . 6 3 \pm 6 . 3 5 } }$ </td><td> $\overline { { 3 0 . 0 0 \pm 8 . 0 2 } }$ </td><td> $\overline { { 3 6 . 3 1 \pm 7 . 8 2 } }$ </td><td> $3 6 . 3 8 \pm 6 . 8 6$ </td><td> $\overline { { 2 8 . 8 9 \pm 9 . 3 1 } }$ </td><td> $\overline { { 3 4 . 9 3 \pm 8 . 3 0 } }$ </td></tr><tr><td> $\mathsf { E A T - a s }$ </td><td> $6 5 . 6 4 \pm 5 . 9 6$ </td><td> $6 4 . 0 8 \pm 4 . 8 2 $ </td><td> $6 5 . 3 8 \pm 4 . 2 4$ </td><td> $6 5 . 9 5 \pm 5 . 1 6$ </td><td> $6 4 . 3 9 \pm 3 . 5 2$ </td><td> $6 4 . 9 2 \pm 3 . 3 7$ </td><td> $6 8 . 1 6 \pm 3 . 7 5$ </td><td> $6 5 . 7 4 \pm 1 . 7 5$ </td><td> $6 6 . 7 4 \pm 1 . 7 0$ </td></tr><tr><td> $\mathsf { B E A T s }$ </td><td> ${ \bf 6 8 . 1 1 \pm 3 . 9 5 }$ </td><td> ${ \bf 6 5 . 6 5 \pm 2 . 6 4 }$ </td><td> ${ \bf 6 5 . 9 9 \pm 2 . 6 8 }$ </td><td> ${ \bf 6 8 . 8 5 \pm 4 . 4 3 }$ </td><td> ${ \bf 6 6 . 3 0 \pm 3 . 2 5 }$ </td><td> ${ \bf 6 6 . 8 6 \pm 2 . 7 8 }$ </td><td> ${ \bf 6 9 . 4 1 \pm 2 . 2 2 }$ </td><td> ${ \bf 6 7 . 0 3 \pm 0 . 2 5 }$ </td><td> ${ \bf 6 7 . 8 3 \pm 1 . 2 7 }$ </td></tr></table>

## 4.5. Efectiveness of the Dataset for MMSI-Level Vessel Classification

To further explore the potential of fine-grained acoustic modeling and fully exploit the rich maritime metadata provided in ONCship, we further investigate MMSI-level vessel classification, where individual vessels are treated as distinct recognition targets. Unlike conventional underwater acoustic classification benchmarks that categorize targets into broad vessel types, this task aims to identify specific vessel instances by capturing their unique acoustic signatures associated with MMSI numbers. As illustrated in Figure 5 (left), conventional vessel-type classification groups diferent vessels into coarse-grained categories. In contrast, MMSI-Level Vessel Classification, shown in Figure 5 (right), aims to assign acoustic segments from the same vessel across diferent voyages and timestamps to a unified identity class.

For this evaluation, we curate a benchmark subset comprising 50 distinct MMSIs with a cumulative duration of 100 hours. For each MMSI, we ensure that pairwise 1 minute source segments are gathered from diferent timestamps and distinct voyages separated by at least 6 hours. Each 1 minute source segment is subsequently segmented into 10-second audio clips. To strictly prevent data leakage, all 10-second clips derived from the same 1 minute source segment are constrained to reside exclusively within either the training, validation, or test set. We benchmark this task using the EAT-as model, systematically assessing the capability of deep acoustic representations to capture vessel-specific acoustic characteristics in the presence of complex acoustic propagation and ambient oceanic noise.

For EAT-as, a linear classification head was appended to the pre-trained backbone for downstream adaptation. The model was trained for 10 epochs using the Adam optimizer, with a batch size of 64, an initial learning rate of $2 \times 1 0 ^ { - 5 }$ and a weight decay of $1 \times 1 0 ^ { - 6 }$ . We explored two loss functions, cross-entropy and ArcFaceDeng, Guo, Xue and Zafeiriou (2019), with the latter configured using an angular margin of $m \ = \ 0 . 2$ and a scale factor of $s ~ = ~ 1 6 . 0$ . For each configuration, the checkpoint achieving the highest validation accuracy was selected for final evaluation.

Table 8 presents the MMSI-Level Vessel Classification results obtained by fine-tuning with cross-entropy and ArcFace losses. Both loss functions yield comparable performance, with Top-1 accuracies exceeding 0.53 and Macro-F1 scores of approximately 0.53. Specifically, ArcFace slightly outperforms CE in terms of Top-1 accuracy and Macro-F1, achieving 0.5320 and 0.5333, respectively, whereas CE achieves marginally higher Top-5 accuracy, reaching 0.6957. The Top-5 accuracies of both methods approach 0.70, indicating that the correct vessel identity is ranked among the top five candidates in most cases. These findings suggest that underwater acoustic signals contain discriminative features associated with individual vessels, demonstrating the feasibility of MMSI-Level Vessel Classification while highlighting the potential for further improvements in finegrained vessel classification.

## 5. Conclusion

To address the scarcity of large-scale labeled data for underwater acoustic vessel recognition, this paper constructs ONCship, a large-scale underwater acoustic dataset based on hydrophone recordings from ONC and AIS data. Approximately 4521 hours of valid recordings were curated from nearly 20000 hours of raw acoustic data, covering diverse vessel types and marine environments. Experiments on DeepShip and ShipsEar demonstrate that ONCship can improve acoustic representation learning, while using it as auxiliary training data further benefits downstream vesseltype classification. We also investigate MMSI-Level Vessel Classification, where individual vessels are treated as distinct identities. The results show Top-1 accuracies above 53% and Top-5 accuracies close to 70%, demonstrating the feasibility of distinguishing individual vessels from underwater acoustic signals. Overall, ONCship provides a scalable resource for large-scale underwater acoustic representation learning, vessel-type classification, and fine-grained vessel identification. Future work will focus on improving the robustness and generalization of acoustic representations across diferent hydrophones, marine environments, and noise conditions.

## References

Aslam, M.A., Zhang, L., Liu, X., Irfan, M., Xu, Y., Li, N., Zhang, P., Zheng, J., Li, Y., 2024. Underwater sound classification using learning based methods: A review. Expert Systems with Applications 255, 124498.

Baevski, A., Babu, A., Hsu, W.N., Auli, M., 2023. Eficient self-supervised learning with contextualized target representations for vision, speech and language, in: International conference on machine learning, PMLR. pp. 1416–1429.

Chen, S., Wu, Y., Wang, C., Liu, S., Tompkins, D., Chen, Z., Wei, F., 2022. Beats: Audio pre-training with acoustic tokenizers. arXiv preprint arXiv:2212.09058 .

Chen, W., Liang, Y., Ma, Z., Zheng, Z., Chen, X., 2024. Eat: Selfsupervised pre-training with eficient audio transformer. arXiv preprint arXiv:2401.03497 .

![](images/778bfae5a0994636bb0d05c031786e4d79a2276c0e9964ba86ac35179e6936e8.jpg)  
Figure 5: Comparison between conventional vessel-type classification and MMSI-Level Vessel Classification. Conventional vesseltype classification assigns underwater acoustic recordings from diferent vessels to coarse-grained vessel categories (left), whereas MMSI-Level Vessel Classification aims to identify individual vessels based on their MMSI labels (right). Recordings from the same vessel share a common MMSI label, while diferent vessels of the same type are treated as distinct identities.

Table 8  
MMSI-Level Vessel Classification results. EAT(CE) denotes the model trained with cross-entropy loss, while EAT(ArcFace) employs ArcFace loss. The best result in each column is highlighted in bold.
<table><tr><td>Model</td><td>Top-1 ACC</td><td>Top-5 ACC</td><td>Macro-F1</td></tr><tr><td>EAT(CE)</td><td>0.5312</td><td>0.6957</td><td>0.5307</td></tr><tr><td>EAT(ArcFace)</td><td>0.5320</td><td>0.6892</td><td>0.5333</td></tr></table>

Deng, J., Guo, J., Xue, N., Zafeiriou, S., 2019. Arcface: Additive angular margin loss for deep face recognition, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4690–4699.

Du, X., Hong, F., 2024. Qiandaoear22: a high-quality noise dataset for identifying specific ship from multiple underwater acoustic targets using ship-radiated noise. EURASIP Journal on Advances in Signal Processing 2024, 96.

Feng, S., Zhu, X., Ma, S., 2024. Masking hierarchical tokens for underwater acoustic target recognition with self-supervised learning. IEEE/ACM Transactions on Audio, Speech, and Language Processing 32, 1365– 1379.

Gemmeke, J.F., Ellis, D.P., Freedman, D., Jansen, A., Lawrence, W., Moore, R.C., Plakal, M., Ritter, M., 2017. Audio set: An ontology and humanlabeled dataset for audio events, in: 2017 IEEE international conference on acoustics, speech and signal processing (ICASSP), IEEE. pp. 776– 780.

Guo, L., Xu, J., Li, S., Wingfield, J., 2023. A Python package for decoding Automatic Identification System (AIS) data from the Canadian Coast Guard. Canadian Technical Report of Hydrography and Ocean Sciences 360. Fisheries and Oceans Canada. URL: https://publications.gc.ca/ collections/collection\_2023/mpo-dfo/Fs97-18-360-eng.pdf.

Hashemi, C., Stout, T., Hoogs, A., Parham, J., 2026. Uniqueship: Mitigating data leakage in acoustic ship classification benchmark datasets. arXiv preprint arXiv:2609.13659 .

He, K., Zhang, X., Ren, S., Sun, J., 2016. Deep residual learning for image recognition, in: Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778.

Heesemann, M., Insua, T.L., Scherwath, M., Juniper, K.S., Moran, K., 2014. Ocean networks canada: From geohazards research laboratories to smart ocean systems. Oceanography 27, 151–153.

Huang, P.Y., Xu, H., Li, J., Baevski, A., Auli, M., Galuba, W., Metze, F., Feichtenhofer, C., 2022. Masked autoencoders that listen. Advances in neural information processing systems 35, 28708–28720.

Huang, W., Sun, S., Lu, J., Xu, Z., Xiu, Z., Zhang, H., 2026. A multitask learning balanced attention convolutional neural network model for few-shot underwater acoustic target recognition. Expert Systems with Applications 313, 131597.

Hummel, H.I., Gansekoele, A., Bhulai, S., van der Mei, R., 2026. The computation of generalized embeddings for underwater acoustic target recognition using contrastive learning. Applied Acoustics 243, 111103.

Hummel, H.I., Van Der Mei, R., Bhulai, S., 2024. A survey on machine learning in ship radiated noise. Ocean Engineering 298, 117252.

Irfan, M., Zheng, J., Ali, S., Iqbal, M., Masood, Z., Hamid, U., 2021. DeepShip: An underwater acoustic benchmark dataset and a separable convolution based autoencoder for classification. Expert Systems with Applications 183, 115270.

Karakos, D., Silovsky, J., Schwartz, R., Hartmann, W., Makhoul, J., 2018. Individual ship detection using underwater acoustics, in: 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 2121–2125.

Li, Z., Xiang, S., Yu, T., Gao, J., Ruan, J., Hu, Y., Liu, T., Fu, Y., 2024. Oceanship: A large-scale dataset for underwater audio target recognition. arXiv preprint arXiv:2401.02099. arXiv:2401.02099.

Liu, A.H., Chang, H.J., Auli, M., Hsu, W.N., Glass, J., 2023. DinoSR: Selfdistillation and online clustering for self-supervised speech representation learning, in: Advances in Neural Information Processing Systems (NeurIPS), pp. 58346–58362.

Ocean Networks Canada, n.d. Oceans 3.0 data portal. Online data portal. URL: https://data.oceannetworks.ca/. accessed 3 October 2026.

Santos-Domínguez, D., Torres-Guijarro, S., Cardenal-López, A., Pena-Gimenez, A., 2016. ShipsEar: An underwater vessel noise database. Applied Acoustics 113, 64–69.

da Silva, F.O.B., Fernandes, J.d.C.V., Soares Filho, W., de Oliveira e Souza Filho, J.B., Spengler, Â., de Moura Júnior, N.N., 2025. IARA: An underwater acoustic database. IEEE Access 13, 120293–120306.

Wu, F., Yao, H., Zhao, Z., Zhao, X., Zang, Y., Wang, H., 2025. ESTMST-ST: An end-to-end soft threshold and multiloss self-distillation based Swin Transformer for underwater acoustic signal recognition. IEEE Transactions on Geoscience and Remote Sensing 63, 1–13.

Xie, Y., Ren, J., Xu, J., 2024. Unraveling complex data diversity in underwater acoustic target recognition through convolution-based mixture of experts. Expert Systems with Applications 249, 123431.