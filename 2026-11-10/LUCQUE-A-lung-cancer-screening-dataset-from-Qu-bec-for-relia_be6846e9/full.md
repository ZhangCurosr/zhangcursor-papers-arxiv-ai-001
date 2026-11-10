# LUCQUE: A lung cancer screening dataset from Québec for reliable AI evaluation

Johanna Topalis<sup>1,2,3,4,</sup> <sup>\*</sup>, Yannick Lemaréchal<sup>6,</sup> <sup>\*</sup>, Gabriel Couture<sup>6</sup>, Hanife Ersay<sup>6</sup>, Loran Nas<sup>1</sup>, Balthasar Schachtner<sup>1,3</sup>, Bastian Oliver Sabel<sup>1,4,7</sup>, Michael Ingrisch<sup>1,2,3</sup>, Philippe Després<sup>5,6,+</sup>, and Katharina Jeblick<sup>1,3,4</sup>

<sup>1</sup>Department of Radiology, LMU University Hospital, LMU Medizin, Ludwig-Maximilians-Universität München, Munich, Germany <sup>2</sup>relAI – Konrad Zuse School of Excellence in Reliable AI, Munich, Germany

<sup>3</sup>Munich Center for Machine Learning (MCML), Munich, Germany

<sup>4</sup>Comprehensive Pneumology Center (CPC-M), Member of the German Center for Lung Research (DZL), Munich, Germany <sup>5</sup>Département de physique, de génie physique et d’optique, Université Laval, Québec, Canada

<sup>6</sup>Centre de recherche de l’Institut universitaire de cardiologie et de pneumologie de Québec - Université Laval, Québec, Canada <sup>7</sup>Department of Radiology, Asklepios Lung Clinic Munich-Gauting, Germany

Shared first authorship

<sup>+</sup>Corresponding author

## Abstract

BACKGROUND AI-based tools are expected to play a key role in lung cancer screening (LCS). The evaluation of these tools often relied on private datasets or fragmented public datasets with inconsistent availability of clinical data and expert nodule annotations. This has hindered reproducibility, comparability, and independent assessment of AI tools. To address these gaps, we present LUCQUE, a novel, large, comprehensively annotated dataset derived from LDCT lung cancer screening in Québec, which is accessible for reliable AI evaluation.

METHODS LUCQUE was curated and is hosted on PARADIM, a digital infrastructure supporting DICOM data de-identification, annotation, long-term storage and analysis. LDCT scans and clinical information were combined to a curated collection and complemented by voxel-level, expert-derived pulmonary nodule and cyst segmentations, and Lung-RADS-based lesion characteristics. Key dataset characteristics and statistics were reported, including participant demographics, imaging parameters, radiological findings, and segmented lesion characteristics. Dataset analyses were conducted according to PARADIM’s model-to-data approach, in which user code is executed on the platform such that data remains on premise.

RESULTS LUCQUE currently comprises 1,678 LDCT baseline studies from 1,678 participants. Most studies were classified as Lung-RADS 2. A total of 1,776 pulmonary nodules and 60 cysts were annotated, with a median (interquartile range) volume of 89(56, 159)mm<sup>3</sup> and 1,384 (589, 3793)mm<sup>3</sup> respectively.

CONCLUSIONS Compared to publicly available LCS datasets, LUCQUE currently represents the largest accessible annotated real-world LCS dataset. Access to LUCQUE is provided via a model-to-data approach, enabling privacy-preserving, reproducible, and independent real-world evaluation of AI tools in LCS.

## 1 Introduction

Lung cancer is the leading cause of cancer-related mortality worldwide [1]. Early detection through low-dose computed tomography (LDCT) screening has been shown to reduce lung cancer mortality [2, 3]. Consequently, lung cancer screening (LCS) programs have been implemented in several countries [4]. Over the past decade, the development of AI models for pulmonary nodule detection, segmentation, tracking, and lung cancer risk prediction has seen significant progress in research [5, 6, 7, 8, 9] and industry [10]. Given the potential of AI models to support clinicians at LDCT scan interpretation, AI-based tools are expected to become a key component of LCS programs [11].

However, the focus of LCS AI research has mainly been the training of advanced models; their independent, comparable, and reproducible evaluation has received less attention [10, 12, 13] and has suffered from a fragmented landscape of LCS datasets. So far, evaluation has relied on either private datasets or on a limited number of publicly available (or controlled-access) datasets [14], such as the LIDC-IDRI [15], LUNA16 [7], and data from the U.S. national LCS trial (NLST) [3]. Private datasets are typically inaccessible to external researchers, making it difficult to reproduce results or compare the performance of different AI models [14]. Public or controlled-access datasets, on the other hand, are (easily) accessible, but they also come with limitations. For example, CT scans in LIDC-IDRI/LUNA16 were not exclusively derived from LCS [15] and lack screening-relevant metadata (e. g., Lung-RADS [16] classification). In contrast, the standard NLST imaging release does not include expert lung nodule segmentations, limiting its utility for evaluating, e. g., AIbased nodule detection, segmentation, and tracking tools. In addition, the integrity of evaluation results on public LCS datasets is generally not guaranteed (e. g., test data might have been (inadvertently) used during training [17, 18]).

This situation has resulted in several challenges for reliable evaluation of AI-based tools for LCS: (1) the fact that some LCS datasets, including LIDC-IDRI/LUNA16, originate (in large parts) from clinical routine rather than real-world LCSs, (2) the lack of LCS datasets combining both expert lung nodule annotations and screening-specific clinical data, (3) limited reproducibility and comparability when inaccessible private datasets are used for evaluation, and (4) the absence of accessible yet independent [19] LCS evaluation datasets.

To address these challenges, we present a novel multicenter LUng Cancer screening dataset from QUÉbec, LUCQUE, specifically designed for independent, comparable, and reproducible real-world evaluation of AI-based tools for LCS. LUCQUE comprises 1,678 baseline LDCT studies from the Lung Cancer Screening Project in Québec,

Canada [20, 21], acquired at two different hospitals. It provides comprehensive screening-specific information (e. g., PLCOm2012 [22], Lung-RADS scores [16]), as well as rich demographic, clinical, and imaging-related data. The imaging studies have been further enriched with expert reference segmentation masks and additional Lung-RADSrelevant characteristics for lung nodules and cysts.

LUCQUE was curated and hosted on PARADIM (Platform for the Annotation, Reuse, and Analysis of Medical Images) [24]. PARADIM is a digital infrastructure supporting customizable DICOM de-identification, curation, annotation, organization and long-term storage. Additionally, it supports the automated execution of large-scale operations on image collections. Access to LUCQUE on PARADIM follows a model-to-data approach: instead of granting external users direct access to the dataset, they upload their models, which are then exposed to the DICOM data on PARADIM within a managed execution environment. This design preserves patient privacy and ensures traceability and reproducibility of evaluation results, while also guaranteeing that no testing data is used for training. Provided formal approvals, LUCQUE is accessible on PARADIM for evaluation purposes to both research and industry partners, providing a robust foundation for the reliable assessment of AI tools for LCS.

## 2 Materials and Methods

The workflow used to create and analyze LUCQUE is summarized in Fig. 1 and described in the following sections. It includes 1. data acquisition, 2. curation, 3. annotation, and 4. analysis. All steps were carried out with PARADIM’s streamlined and largely automated pipelines. Data retrieval and analysis was approved by the ethics committee of Centre de recherche de l’Institut universitaire de cardiologie et de pneumologie de Québec-Université Laval (IUCPQ-ULaval), and required a Privacy Impact Assessment since informed consent was not possible for this project (Ethics approval 2023-3946, 22299).

## 2.1 Data acquisition

LDCT images and clinical information were derived from lung cancer screening in Québec, Canada (Supplemental Material S1.1.1), which began in June 2021 as a demonstration project [20] and is now offered routinely [21]. Eligible participants are current or former smokers (with cessation ≤ 15 years) aged 55-74 years with $\mathbf { a } \geq 2 0$ years smoking history and a PLCOm2012noRace score ≥ 2%. Participants underwent standardized LDCT screening (Supplemental Material S1.1.2) at designated centers, where radiologists evaluated the scans and assigned a Lung-RADS score (Supplemental Material S1.1.3) to guide follow-up imaging or further diagnostic examinations.

![](images/16fe9d176681f3448b82bc5975946e958a7166e11bfc04f95c8d0188ed53e570.jpg)  
Figure 1: Workflow from data acquisition to analysis using PARADIM. PARADIM’s components (white area) are deployed across different environments to facilitate seamless data processing. In the health centers (blue zone), lung cancer screening data is stored in local hospital information systems. Through PARADIM’s data pipelines, DICOM objects are automatically retrieved, de-identified (Karnak), and then transferred to the academic institution’s network (orange zone), where they are stored in a DICOM server (Orthanc) and organized into collections (Kheops). DICOM images are annotated by medical experts using PARADIM’s OHIF-based annotation pipeline [23]. Users can upload containerized data analysis code to a registry and execute it on collections within PARADIM. All data remains on premise; only aggregated analysis results can be extracted from the platform. Cf. [24] for further details on PARADIM’s architecture.

LDCT scans and associated radiology reports were archived in local hospitals’ Picture Archiving and Communication Systems (PACS). Participants’ records were systematically collected and stored in a centralized RED-Cap [25]-based database. These included demographic information (e. g., sex, age, ethnicity), smoking history, medical history (e. g., comorbidities), the PLCOm2012noRace risk score at enrollment, and radiological findings (e. g., Lung-RADS score; Supplemental Material S1.1.3) for each LDCT.

## 2.2 Data curation

In an automated process, eligible LDCT scans were identified in the PACS of two participating hospitals, deidentified using the DICOM gateway Karnak [26] and sent to PARADIM (Supplemental Material S1.2.1). Participant’s data from the centralized REDCap-based screening database was likewise de-identified, and a selection of information uploaded to PARADIM as DICOM Structured Reporting (SR) files (cf. Tab. S1 for included participants characteristics at enrollment and Tab. S2 for included radiological findings). Further details on this data ingestion process are provided in Supplemental Material S1.2.2.

All data was stored in DICOM format in PARADIM’s central storage backend, Orthanc [27], an open-source DICOM server. In PARADIM, Orthanc is not directly exposed to users. Instead, all interactions with the stored data is mediated through Kheops [28], a platform built for managing and sharing imaging data in research settings. It organizes the DICOM data into collections, i.e., logical sets that aggregate DICOM series with tailored access permissions. We created a Kheops collection for LUCQUE (Fig. 2), based on the selected exclusion/inclusion principles (Supplemental Material S1.2.3).

## 2.3 Data annotation

All LDCT studies in LUCQUE were annotated retrospectively by a radiologist with four years of experience in general radiology using PARADIM’s annotation pipeline [23], which leverages the web imaging platform by the Open Health Imaging Foundation (OHIF) [29]. The annotations were performed in a blinded fashion using LDCTs only. Information on the participants’ medical history and screening findings were not provided to the annotating radiologist. The latter was not provided to approximate a typical image-reading setting. Before starting with the annotations, the radiologist was provided a detailed annotation guide (Supplemental Material S4).

All detected pulmonary nodules (3 mm ≤ diameter ≤ 30 mm; consistent with [15]) and tumor masses (diameter > 30 mm) [30] as well as cysts were manually segmented at the voxel-level. Smaller nodules or cysts were also segmented if detected. Part-solid nodules were annotated in two parts, with the solid and non-solid components segmented separately and then linked as components of the same part-solid nodule. In the following all segmented findings (pulmonary nodules and cysts) will be referred to as lesions. For each study, all lesions were stored in a single DICOM Segmentation (SEG) file.

The lesions were then further classified as solid nodule, non-solid nodule or cyst. Lesions were further characterized using binary labels (yes/no, no as initial selection) for key radiological features derived from Lung-RADS v2022 [16] (Supplemental Material S1.3.1). The radiologist also identified the lung segment where the lesion was located. Nodule characteristics were saved as DICOM SR objects. Annotation metadata, including annotator identity, annotation date, and software details, were embedded in DICOM tags of the DICOM SEG and SR files to ensure traceability, following FAIR [31] best practices for data documentation.

To assess the annotation quality, an intra- and inter-reader variability assessment was performed (Supplemental Material S1.3.2).

## 2.4 Data analysis

We extracted key characteristics of LUCQUE to describe the dataset in Section 3. Participants’ characteristics, LDCT technical parameters (including scanner manufacturers, scanner model names, and reconstruction kernels) and radiological findings were analyzed using medians and interquartile ranges (IQRs) or by total counts.

For segmented lesions, we plotted the distribution of lesion counts per participant and reported the total count of solid, non-solid, part-solid nodules, and cysts, as well as the total count of each lesion characteristic. In addition, we calculated the volumes ("Approximate volume" radiomic feature) of lesions directly from the baseline DICOM CT and corresponding SEG objects using PyRadiomics [32]. The code execution on the DICOM objects followed PARADIM’s model-to-data approach described in more detail in Supplemental Material S1.4. We aggregated volume data over all studies to plot the lesion volume distribution, calculate the median (IQR), report the number of solid nodules exceeding volumetric size thresholds of 34mm<sup>3</sup>, 113mm<sup>3</sup>, 268mm<sup>3</sup> and 1,767mm<sup>3</sup> (as used in Lung-RADS v2022 for size-based risk stratification [16]), and the number of (solid) tumor masses with a volume $> 1 4 , 1 3 7 m m ^ { 3 }$ (corresponding to the volume of a sphere with a diameter > 3cm).

## 3 Results

LUCQUE comprises 1,678 baseline studies with 3,356 series from 1,678 participants, acquired between June 2021 and November 2025 at two hospitals in Québec, Canada. 1,539 studies were acquired in center 1 and 139 at center 2.

Each study in LUCQUE (Fig. 2) includes two axial CT series reconstructed with a hard and a soft kernel, an SR with clinical data (participants’ characteristics and radiological findings from the clinical program database), a SEG with lesion segmentations and a corresponding SR with lesion characteristics.

## 3.1 Participant’s characteristics

LUCQUE contains 878 male and 800 female participants with a median (IQR) age of 65(61, 69) years (Fig. 3a). The majority of participants were Caucasian (1,649), but the dataset also includes individuals from other racial or ethnic backgrounds. 1268 participants smoked, while 410 had quit smoking prior to enrollment. Chronic respiratory conditions were prevalent at enrollment, with 404 participants diagnosed with COPD, emphysema, or chronic bronchitis. The calculated median (IQR) PLCOm2012noRace score was 4.2 % (2.7 %, 7.0 %) (Fig. 3b). A summary of all baseline characteristics is provided in Tab. S1.

![](images/75d9cce831aa8ce4444ab451b819c9c00b5efa5e2e6706c7bca551021e3a69aa.jpg)  
Figure 2: Overview of the data structure for LUCQUE. LUCQUE contains baseline studies, each of which include two axial CTs (reconstructed with a softer and harder kernel), and a Structured Report (SR) with clinical data (radiological findings and participants’ characteristics). LUCQUE is also enriched with lesion segmentations (saved as DICOM Segmentation (SEG)) and characteristics (saved as DICOM SR)).

## 3.2 LDCT technical parameters

LDCT studies were acquired with Philips CT iCT (1,539) or Siemens SOMATOM Definition Edge (139). Axial series were reconstructed with one softer and one harder kernel: L and YA for Philips, Br43f/Bf42f and Bl57f for Siemens, respectively.

## 3.3 Radiological Findings

Radiological findings available in LUCQUE are summarized in Tab. S2. Most studies were classified as Lung-RADS 2 (1,286), followed by Lung-RADS 3 (111). The highest Lung-RADS 4B and 4X were assigned for 32 and 21 studies, respectively (Fig. 3c). In 1,637 participants, other radiological findings were reported. Emphysema were reported for 1,423 studies. In 1,490 studies, findings in the heart were reported, among these were 793, 376 and 214 studies with low, moderate and severe coronary calcification.

## 3.4 Nodule characteristics

In 872 baseline studies, the annotating radiologist did not detect any pulmonary nodules or cysts, whereas at least one lesion was annotated in 806 studies. The distribution of total lesions per study is illustrated in Fig. 4a, with the number of annotated lesions ranging from 1 to 43 per study. The majority of the detected 1,776 nodules were solid (1,661), followed by non-solid (103) and part-solid (12) (Fig. 4c). 555 solid pulmonary nodules were labeled as complete, central, popcorn, or concentric ring calcifications, and 106 as juxtapleural nodules with smooth margins; and oval, lentiform, or triangular shape. Additionally, 12 nodules were annotated as as spiculated, 7 nodules as fat-containing, 6 as sub-segmental airway nodules and 6 as segmental or more proximal airway nodules (Fig. 4d). The volume distribution of all segmented nodules (and tumor masses) is shown in Fig. 4b. The median (interquartile range) nodule volume was $8 9 ( 5 6 , 1 5 9 ) \mathrm { m m } ^ { 3 }$ The smallest detected nodule measured $1 7 \mathrm { m m } ^ { 3 }$ , while the largest mass had a volume of approximately $1 7 2 \mathrm { c m } ^ { 3 }$ . A total of 1,555 solid nodules exceeded the Lung-RADSrelevant volume threshold of $3 4 \mathrm { m m } ^ { 3 }$ , among which 571, 188 and 41 further had a volume greater than $1 1 3 \mathrm { m m } ^ { 3 }$ $2 6 8 \mathrm { m m } ^ { 3 }$ and $1 , 7 6 7 \mathrm { m m } ^ { 3 }$ , respectively. 10 solid masses had a volume larger than $1 4 , 1 3 7 \mathrm { m m } ^ { 3 }$ . In addition, 60 cysts were detected with 54 labeled as thin-walled and 5 as thick walled. 51 cysts were annotated as unilocular, 8 as multi-locular and 4 as cysts with opacity (Fig. 4e). Example annotations illustrating pulmonary nodule and cysts characteristics are provided in Fig. 5. Intra- and inter-reader analysis results are presented in Supplemental Material S2.1.

![](images/42cb5dce0fe04c7d3273d5558534ed341590b23ae84628708bcbb4c71559a92a.jpg)  
(a)

![](images/604ea813d2a7f64f6aed511a87553542da9c2641b74f6496b3fcc21256a70e12.jpg)  
(b)

![](images/4a0ae56a4901326a41acdfff0da6c25081cc3e1422bcc421cda81c7367066072.jpg)  
(c)  
Figure 3: Participants’ (a) age (in years) at LDCT imaging, (b) PLCOm2012noRace score, and (c) Lung-RADS.

## 4 Discussion

Given the increasing integration of AI support tools into LCS programs [11], reliable evaluation of these tools is critical for their safe and effective clinical adoption. As highlighted in the HANSE LCS trial [33], differences between AI software tools can lead to clinically relevant discrepancies in key outputs such as nodule volume measurements and Lung-RADS classifications which may result in inconsistent participant management. Yet, current performance data and fair comparisons, across both research and commercial models, remain limited [12, 13, 10].

In this study, we introduced a new LCS dataset from Québec, LUCQUE, designed to address this gap by enabling systematic evaluation of AI models across key LCS tasks including nodule detection, segmentation/volumetric measurement, classification, and Lung-RADS scoring. LUCQUE includes 1,678 baseline LDCT studies, paired with rich screening-specific metadata, expert reference segmentations, and detailed Lung-RADS-relevant labels for pulmonary nodules. These resources enable nuanced analyses across patient subgroups, and nodule characteristics (e. g., non-solid vs. solid), facilitating clinically relevant performance benchmarking.

While several LCS trials and programs have been implemented worldwide [4], only a few datasets have been made accessible for AI development and evaluation. The most widely used datasets in the context of AI in LCS include the LIDC-IDRI [15] and its subset LUNA16 [7], which were in large parts not acquired as part of a LCS, but mainly in clinical routine. The NLST dataset [3] from the large U.S. national trial was the first publicly released LCS dataset, while the recently published Duke Lung Cancer Screening (DLCS) Dataset [34] is the latest addition, providing more contemporary data from the Duke University Health System.

Compared to these datasets (Tab. 1), LUCQUE includes the largest number of participants and studies after NLST, and is the largest LCS dataset with expert-annotated nodules. In contrast to older datasets, LUCQUE reflects recent imaging practices and CT technology. Similar to NLST and DLCS, it is drawn from a real-world LCS, enhancing its relevance for evaluating AI tools in contemporary screening settings.

LUCQUE stands out for the richness of its nodule annotations, offering both voxel-level segmentations and detailed lesion classifications aligned with Lung-RADS v2022. While LIDC-IDRI includes segmentations from multiple annotators and LUNA16 provides derived nodule centers and diameters, DLCS relies on semi-automatically generated bounding boxes [34]. In contrast, the standard NLST imaging release lacks exact nodule locations. Although LIDC-IDRI and NLST report some nodule characteristics, these are not aligned with Lung-RADS. LUCQUE and NLST include detailed participant characteristics relevant for screening, such as demographics, smoking, and medical history. In DLCS, only demographic data and smoking history are provided. Lung-RADS scores are provided only in LUCQUE and DLCS.

![](images/a92e7ff6054895f6b4cf92f085e0384c0684690b1b0d30423a5d9ca4fcfa9b6f.jpg)

![](images/8f0c298545fb15aa2fe77e8d8030441002127b8d01d6f4a0164e26a32abe1094.jpg)

(a)  
![](images/5da98d52d2be7e4963ea66dfd3d45b4bd6fefecc9ff4b260af983d6053e970fd.jpg)  
(c)

(b)  
![](images/702a6177bf201f82f0e5bbc82078433299c7836759d63bcea6c3c03b70127808.jpg)  
(d)

![](images/81d78374813e703d2dd5c5cfd415347d96a6766ea5cd0ad17a1c0c5f39a3d500.jpg)  
(e)

Figure 4: (a) Number of pulmonary nodules and cysts per study. (b) Volume distributions of nodules (blue; plotted until 1, $8 0 0 \mathrm { m m } ^ { 3 } )$ and cysts (orange; plotted until $2 0 \mathrm { c m } ^ { 3 } )$ . (c) Number of findings classified as solid, non-solid or part-solid pulmonary nodules or cysts. (d) Number of characteristics of solid pulmonary nodules. (e) Number of characteristics of cysts.  
![](images/3454efc936eb8eaee028b1eaa7ffaf9213a07bed8a092565ea52f29f324498fc.jpg)

![](images/aeb0572fb22b334c13b2f284d254729a7ec5388fb854c27ef141a8a1eff4de7a.jpg)  
(a)

![](images/67577a96c868f32a6faf047a6af6c9e90f3c27e566dc7585dc6da1d262cb551d.jpg)

![](images/d8e1e854b8052c26520843f59479422cf11d001c7c6ae145c1c6382d0fe1bb51.jpg)

(b)  
![](images/019b2ba3c5a27311ef5ab1748396959f159d27f5bf1af656978397a442a10a73.jpg)

![](images/6125ec6a0554e6df7b039f00fd0430ed799a862bd82609756cc279ce4632ed89.jpg)  
(e)  
(f)

(c)  
(d)  
![](images/2b8c6bea9f21b77d1d96bc4d6e204bc35f547a409d0a46b1aba01f905605c11b.jpg)  
(g)

![](images/8734507260495cf25396c26d4493bbdea15b79c14526def2f2c1d46281acc20f.jpg)  
(h)

![](images/96aa36a0df15b65b252d04c06382de4a9c2ba74c337913cbb873577c310c6b69.jpg)  
(i)

![](images/67542ad3e3e992820c2688cf8774f6f91015910491b7842046345e4a7813c7da.jpg)  
(j)

![](images/93ae74132a4097c1eb4758aac5d2a877d3453a99b948f585fd0539f107636ab7.jpg)  
(k)  
Figure 5: Exemplary annotations of the categories (a) part-solid nodule, (b) nodule with complete, central, popcorn, or concentric ring calcification, (c) fat-containing nodule, (d) juxtapleural nodule with smooth margins; and oval, lentiform, or triangular shape, (e) airway nodule: subsegmental, and (f) airway nodule: segmental or more proximal, (g) thick-walled cyst (>=2mm - symmetric, asymmetric, or nodular), (h) thin-walled cyst (uniform wall thickness < 2 mm), (i) unilocular cyst, (j) multilocular cyst and (k) cyst with opacity (nodular, ground glass, or consolidation).

In sum, to the best of our knowledge, LUCQUE is the largest accessible LCS dataset for AI evaluation that combines precise locations with Lung-RADS-based labels for pulmonary nodules and cysts, comprehensive participant characteristics relevant to LCS, and detailed screening findings, including Lung-RADS scores and additional radiological findings.

We host LUCQUE on the PARADIM platform. Not only the LDCT scans, but also nodule segmentations, lesion classifications, and clinical data are provided as DICOM objects. The DICOM format enables the inclusion of all original imaging metadata (where not de-identified), as well as additional contextual information such as annotator identity, annotation time, and methods, ensuring clear data documentation. Crucially, DICOM tags (e. g., in SEG and SR objects) store references to the original CT series, enabling precise traceability of all data elements. While LIDC-IDRI and NLST also use DICOM for imaging data, LUNA16 and DLCS are only available in NIfTI format, limiting metadata depth and traceability.

Unlike the other datasets in Tab. 1, which are fully downloadable (raising concerns about potential prior exposure to test data during pretraining or tuning [17, 18]) LUCQUE is only accessible through a model-to-data approach on the PARADIM platform. In this setup, users upload containerized AI models, which are then automatically executed within a secure compute environment against the DICOM data stored on PARADIM. These workflows are standardized, scalable, and reproducible, with all data, models, and outputs versioned, traceable and reusable. This approach ensures that the underlying data remain unseen, preserving participant privacy while upholding the integrity of evaluation results. The platform thus enables fair and consistent comparisons across AI tools, directly addressing concerns around reproducibility, replicability, and independence in AI evaluation [18, 19].

While LUCQUE fills a need in the AI evaluation landscape for LCS, several limitations should be noted. First, LUCQUE currently includes data from only two health centers in Québec, with not all participants from these centers yet included and no follow-up imaging, which may reduce representativeness of the full Québec screening cohort. Moreover, annotations were provided by a single radiologist with only a subset being analyzed for intra- and inter-reader annotator variability (Supplemental Material S3.2). Finally, on the technical side, parts of the (non-imaging) data ingestion from the clinical program database into PARADIM still required some manual operations, affecting scalability.

This work presents the first version of our dataset, i.e., LUCQUE v1.0.0. To further enhance its value for AI evaluation, LUCQUE will be continuously extended. New dataset versions can be easily and efficiently created on PARADIM by adding imaging and clinical data or annotations using its streamlined and largely automated data ingestion, annotation, and curation pipelines. Upcoming versions of LUCQUE will include additional participants from the ongoing Québec screening program, and complete follow-up studies for more comprehensive longitudinal coverage. Planned additions also include further clinical information such diagnostic results and annotations by multiple experts. As a next step, we will evaluate AI models, e. g., for lung nodule detection, using LUCQUE. To conclude, we presented LUCQUE, a novel, large and comprehensive and annotated multi-center LUng Cancer screening dataset from QUÉbec. LUCQUE was curated and is hosted on PARADIM, a digital infrastructure enabling streamlined and largely automated de-identification, curation, annotation, organization and long-term storage of medical data collections in DICOM format, as well as automated analysis of this data. Access to LUCQUE on PARADIM for research and industrial purposes is possible through a model-to-data approach, enabling reliable real-world evaluation of AI-based tools for LCS.

## Acknowledgements

KJ was supported through funding from Bayerisches Staatsministerium für Wissenschaft und Kunst in cooperation with Fonds de recherche du Québec. JT was supported by the DAAD programme Konrad Zuse Schools of Excellence in Artificial Intelligence, sponsored by the Federal Ministry of Research, Technology and Space. This research was funded by Data Transformation Grants of the Canadian Cancer Society (CCS grants #708520 and #707586). Additional funding was provided by the Natural Sciences and Engineering Research Council of Canada (Discovery program grant RGPIN-2018-04588 and CREATE program grant 528124-2019), the Ministère de l’Économie, de l’Innovation et de l’Énergie du Québec and the Fonds de recherche du Québec (Collaborative Bilateral Research Program Bavaria-Québec, grant 322724). Authors thank Pierre-Luc Asselin for his assistance in retrieving data from the clinical program database.

Generative AI was used as a supporting tool for copyediting parts of the manuscript. The authors carefully reviewed and revised the content as necessary. The ideas expressed in this work are exclusively those of the human authors. The authors take full responsibility for the final manuscript.

Table 1: Comparison of LUCQUE to other accessible datasets used in the context of LCS. Further accessible datasets (cf. Supplemental Material S3.1) are not listed in this table due to their low number of studies (< 300), missing LCS context or detailed information.

<sup>∗1</sup> Additional 448 scans were withheld as test cases and are not publicly available [34]. <sup>∗2</sup> To the best of our knowledge, no acquisition dates for the LIDC-IDRI data are publicly available. As the paper [15] was published in 2011, all images must have been acquired before 2011. <sup>∗3</sup> Only partly LDCTs from lung cancer screening [15]. To the best of our knowledge, the exact number of screening and routine CTs in the dataset is not officially published, although the SeriesDescription field in the DICOM headers may provide some indication. <sup>∗4</sup> For 8 patients, CTs at two different timepoints are available. <sup>∗5</sup> Can be derived from the segmentations. <sup>∗6</sup> For non-calcified nodules or masses with ≤4 mm, NLST contains (if filled) the longest diameter and longest perpendicular diameter, the location (lung lobe) of the nodule epicenter, and the CT slice number containing the greatest diameter. <sup>∗7</sup> Over 900 NLST participants diagnosed with lung cancer within one year after imaging were (in the context of other studies) semi-automatically annotated with 2D [9] and 3D [35] bounding boxes. <sup>∗8</sup> Assessment of subtlety, internal structure, calcification, sphericity, margin, lobulation, spiculation, texture, and malignancy of segmented lung nodules on a Likert scale from 1 to 5. <sup>∗9</sup> For non-calcified nodules or masses with ≤4 mm diameter, NLST contains (if filled) a characterization of nodule margin and the predominant attenuation. <sup>∗10</sup> Participants’ characteristics includes comprehensive demographic information alongside detailed smoking and medical history relevant to LCS. <sup>∗11</sup> Nodule-based malignancy scoring from 1-5 [15].

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>LUCQUE</td><td rowspan=1 colspan=1>LIDC-IDRI [15]</td><td rowspan=1 colspan=1>LUNA16 [7]</td><td rowspan=1 colspan=1>NLST [3]</td><td rowspan=1 colspan=1>DLCS [34]</td></tr><tr><td rowspan=1 colspan=1>Dataset composition</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>No. patients/participants</td><td rowspan=1 colspan=1>1,678</td><td rowspan=1 colspan=1>1,010</td><td rowspan=1 colspan=1>888</td><td rowspan=1 colspan=1>26,254</td><td rowspan=1 colspan=1>1,613*1</td></tr><tr><td rowspan=1 colspan=1>No. CT studies</td><td rowspan=1 colspan=1>1,678</td><td rowspan=1 colspan=1>1,018</td><td rowspan=1 colspan=1>888</td><td rowspan=1 colspan=1>73,116</td><td rowspan=1 colspan=1>1,613*1</td></tr><tr><td rowspan=1 colspan=1>Image acquisition years</td><td rowspan=1 colspan=1>2021-2025</td><td rowspan=1 colspan=1>before 2011*2</td><td rowspan=1 colspan=1>before 2011*2</td><td rowspan=1 colspan=1>2002-2004</td><td rowspan=1 colspan=1>2015-2021</td></tr><tr><td rowspan=1 colspan=1>Real world LCS CTs</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>X*3</td><td rowspan=1 colspan=1>×*3</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Studies at multipletimepoints</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>×*4</td><td rowspan=1 colspan=1>×*4</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>X</td></tr><tr><td rowspan=1 colspan=1>Nodule annotations</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Segmentations</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td></tr><tr><td rowspan=1 colspan=1>Bounding boxes</td><td rowspan=1 colspan=1>×*5</td><td rowspan=1 colspan=1>X*5</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>×*6,7</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Nodule characteristics</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>*8</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>*9</td><td rowspan=1 colspan=1>X</td></tr><tr><td rowspan=1 colspan=1>Annotation style</td><td rowspan=1 colspan=1>manually</td><td rowspan=1 colspan=1>manually</td><td rowspan=1 colspan=1>manually</td><td rowspan=1 colspan=1>×*7</td><td rowspan=1 colspan=1>semi-autom.</td></tr><tr><td rowspan=1 colspan=1>Multiple independentannotators</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td></tr><tr><td rowspan=1 colspan=1>Participants’ charac.*10</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Radiological findings</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Lung-RADS</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>×*11</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Additional radiologicalfindings</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>X</td></tr><tr><td rowspan=1 colspan=1>Data format</td><td rowspan=1 colspan=1>DICOM</td><td rowspan=1 colspan=1>DICOM</td><td rowspan=1 colspan=1>NifTI</td><td rowspan=1 colspan=1>DICOM</td><td rowspan=1 colspan=1>NifTI</td></tr><tr><td rowspan=1 colspan=1>Available for download</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>*1</td></tr></table>

## Authors contributions

JT: Conceptualization, Data Curation, Formal analysis, Project administration, Software, Validation, Visualization, Investigation, Writing - Original Draft, Writing - Review & Editing. YL: Data Curation, Software, Validation, Writing - Review & Editing. GC: Data Curation, Software, Validation, Writing - Review & Editing. HE: Data Curation, Writing - Review & Editing. LN: Data Curation, Writing - Review & Editing. BS: Conceptualization, Methodology, Writing - Review & Editing. BOS: Validation, Writing - Review & Editing. MI: Funding acquisition, Project administration, Resources, Supervision, Writing - Review & Editing. PD: Conceptualization, Funding acquisition, Project administration, Resources, Supervision, Writing - Review & Editing. KJ: Conceptualization, Funding acquisition, Methodology, Project administration, Supervision, Validation, Writing - Original Draft, Writing - Review & Editing.

## Data Availability

LUCQUE v1.0.0 is made accessible via PARADIM through research collaborations (corresponding author: PD). Additional information and updates can be found on PARADIM’s website https://paradim.science/.

## Code Availability

Code is made available on GitHub at https://github.com/paradim-platform/ lucque-lung-cancer-screening-dataset.

## References

[1] F. Bray, M. Laversanne, H. Sung, et al. “Global cancer statistics 2022: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries”. CA: a cancerjournalfor clinicians 74.3 (2024), pp. 229–263.

[2] H. J. de Koning, C. M. van Der Aalst, P. A. de Jong, et al. “Reduced lung-cancer mortality with volume CT screening in a randomized trial”. New England Journal of Medicine 382.6 (2020), pp. 503–513.

[3] National Lung Screening Trial Research Team. “Reduced lung-cancer mortality with low-dose computed tomographic screening”. New England Journal ofMedicine 365.5 (2011), pp. 395–409.

[4] Lung Cancer Policy Network. Interactive map of lung cancer screening. www . lungcancerpolicynetwork . com / interactive-map/, last accessed on 2026-10-27. 2026.

[5] M. Cellina, L. M. Cacioppa, M. Cè, et al. “Artificial intelligence in lung cancer screening: the future is now”. Cancers 15.17 (2023), p. 4344.

[6] C. de Margerie-Mellon and G. Chassagnon. “Artificial intelligence: A critical review of applications for lung nodule and lung cancer”. Diagnostic and interventional imaging 104.1 (2023), pp. 11–17.

[7] A. A. A. Setio, A. Traverso, T. De Bel, et al. “Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: the LUNA16 challenge”. Medical image analysis 42 (2017), pp. 1– 13.

[8] D. Peeters, B. Obreja, N. Antonissen, et al. “Benchmarking of AI and Radiologists for Indeterminate Lung Nodule Malignancy Risk Estimation at Screening CT: The LUNA25 Challenge”. Radiology: Artificial Intelligence 8.5 (2026), e260179.

[9] P. G. Mikhael, J. Wohlwend, A. Yala, et al. “Sybil: a validated deep learning model to predict future lung cancer risk from a single low-dose chest computed tomography”. Journal of Clinical Oncology 41.12 (2023), pp. 2191–2200.

[10] N. Antonissen, O. Tryfonos, I. B. Houben, et al. “Artificial intelligence in radiology: 173 commercially available products and their scientific evidence”. European Radiology (2025), pp. 1–11.

[11] Bundesministerium für Umwelt, Naturschutz, nukleare Sicherheit und Verbraucherschutz. Verordnung über die Zulässigkeit der Anwendung der Niedrigdosis-Computertomographie zur Früherkennung von Lungenkrebs bei rauchenden Perso-

nen (Lungenkrebs-Früherkennungs-Verordnung — LuKrFrühErkV). 2024.

[12] H. K. Hahn, M. S. May, V. Dicken, et al. “Requirements for quality assurance of AI models for early detection of lung cancer”. arXiv preprint arXiv:2502.17639 (2025).

[13] M. Khunte, A. Chae, R. Wang, et al. “Trends in clinical validation and usage of US Food and Drug Administration-cleared artificial intelligence algorithms for medical imaging”. Clinical Radiology 78.2 (2023), pp. 123–129.

[14] M. Nguyen. “A scoping review of deep learning approaches for lung cancer detection using chest radiographs and computed tomography scans”. Biomedical Engineering Advances 9 (2025), p. 100138.

[15] S. G. Armato III, G. McLennan, L. Bidaut, et al. “The lung image database consortium (LIDC) and image database resource initiative (IDRI): a completed reference database of lung nodules on CT scans”. Medical physics 38.2 (2011), pp. 915–931.

[16] J. Christensen, A. E. Prosper, C. C. Wu, et al. “ACR Lung-RADS v2022: assessment categories and management recommendations”. Journal of the American College of Radiology 21.3 (2024), pp. 473–488.

[17] K. G. van Leeuwen, S. Schalekamp, M. J. Rutten, et al. “Comparison of commercial AI software performance for radiograph lung nodule detection and bone age prediction”. Radiology 310.1 (2024), e230981.

[18] P. R. Bassi, W. Li, Y. Tang, et al. “Touchstone benchmark: Are we on the right way for evaluating AI algorithms for medical segmentation?” Advances in Neural Information Processing Systems 37 (2024), pp. 15184–15201.

[19] U.S. Food and Drug Administration. Good Machine Learning Practice for Medical Device Development: Guiding Principles. 2021.

[20] J. Massougbodji, S. Rodrigue, M.-H. Guertin, et al. Évaluation du projet de démonstration de dépistage du cancer du poumon: Volets des processus de dépistage et d’investigation, des effets sur la santé et de l’offre de soutien à la cessation tabagique. https://www.inspq.qc.ca/ sites/default/files/publications/3534- Projet-demonstration-depistage-cancerpoumon.pdf, last accessed on 2026-10-27. 2024.

[21] Gouvernement du Québec. Information about lung cancer screening. https : / / www . quebec . ca / en / health / advice - and - prevention / screening - and - carrier - testing - offer / lung - cancer - screening / information -

lung-cancer-screening, last accessed on 2026- 10-27. 2026.

[22] M. C. Tammemägi, H. A. Katki, W. G. Hocking, et al. “Selection criteria for lung-cancer screening”. New England Journal of Medicine 368.8 (2013), pp. 728–736.

[23] G. Couture and P. Després. paradim-platform/ohifdicom-annotator: ohif-dicom-annotator v1.0.0. Version v1.0.0. Oct. 2026. DOI: 10.5281/zenodo. 23215515.

[24] Y. Lemaréchal, G. Couture, F. Pelletier, et al. “PARADIM: A Platform to Support Research at the Interface of Data Science and Medical Imaging”. Journal ofImaging Informatics in Medicine (2025), pp. 1–13.

[25] P. A. Harris, R. Taylor, R. Thielke, et al. “Research electronic data capture (REDCap)—a metadatadriven methodology and workflow process for providing translational research informatics support”. Journal of biomedical informatics 42.2 (2009), pp. 377–381.

[26] OsiriX Foundation. Karnak. https://karnak. weasis.org. Last accessed on 2026-10-27.

[27] S. Jodogne. “The Orthanc ecosystem for medical imaging”. Journal of digital imaging 31.3 (2018), pp. 341–352.

[28] OsiriX Foundation. KHEOPS. https://kheops. online. Last accessed on 2026-10-27.

[29] E. Ziegler, T. Urban, D. Brown, et al. “Open health imaging foundation viewer: an extensible opensource framework for building web-based imaging applications to support cancer research”. JCO clinical cancer informatics 4 (2020), pp. 336–345.

[30] J. Austin, N. L. Müller, P. J. Friedman, et al. “Glossary of terms for CT of the lungs: recommendations of the Nomenclature Committee of the Fleischner Society.” Radiology 200.2 (1996), pp. 327–331.

[31] M. D. Wilkinson, M. Dumontier, I. J. Aalbersberg, et al. “The FAIR Guiding Principles for Scientific Data Management and Stewardship”. Scientific Data 3.1 (Mar. 2016), p. 160018. DOI: 10.1038/ sdata.2016.18.

[32] J. J. Van Griethuysen, A. Fedorov, C. Parmar, et al. “Computational radiomics system to decode the radiographic phenotype”. Cancer research 77.21 (2017), e104–e107.

[33] R. Kondrashova, F. Klimeš, T. F. Kaireit, et al. “Comparison of AI software tools for automated detection, quantification and categorization of pulmonary nodules in the HANSE LCS trial”. Scientific Reports 14.1 (2024), p. 27809.

[34] A. J. Wang, F. I. Tushar, M. R. Harowicz, et al. “The Duke Lung Cancer Screening (DLCS) Dataset: A Reference Dataset of Annotated Low-dose Screening Thoracic CT”. Radiology: Artificial Intelligence (2025), e240248.

[35] F. I. Tushar, A. Wang, L. Dahal, et al. “Reproducible Benchmarking for Lung Nodule Detection and Malignancy Classification Across Multiple Low-Dose CT Datasets”. arXiv preprint arXiv:2405.04605 (2026).

## Supplemental Material

## S1 Methods

## S1.1 Data acquisition

## S1.1.1 Québec lung cancer screening project

The Québec lung cancer screening demonstration trial aimed to screen 3,000 participants over two annual rounds [1, 2]. The project has since transitioned into a provincial LCS program [3]. Data in LUCQUE originated from both the demonstration project and the ongoing regional LCS program.

## S1.1.2 LDCT acquisition protocol

LDCT scans were acquired at participating hospitals following a standardized acquisition and reconstruction protocol designed for LCS. Imaging was performed during breath-hold at the end of the inspiration phase. A $\mathrm { C T D I _ { v o l } }$ dose below 2 mGy was defined for standard-weight participants to adhere to the low-dose requirement for screening. The acquisition protocol was standardized across scanners, with parameters including rotation time, collimation, pitch, CT x-ray tube voltage, dose reduction index, tube current modulation, and $\mathrm { C T D I _ { v o l } }$ tailored to the specific hardware in use. Images were reconstructed in axial, coronal, and sagittal planes. Axial images were reconstructed with both a softer kernel and a harder kernel.

## S1.1.3 Lung-RADS

In the screening, reporting and management recommendations were made based on the Lung-RADS classification system [4, 5, 6]. Initially, Lung-RADS v1 was used until around April 2023, after which the updated Lung-RADS v2022 was adopted.

## S1.2 Data curation

To identify correct series in the local PACS, each hospital maintained a centralized file that recorded the local patient ID (PID) and the date of the screening exam for participants. These local PIDs were mapped to depersonalized IDs (DIDs), which served as the unique identifier across imaging and non-imaging data.

## S1.2.1 Imaging data

A local Orthanc instance, deployed in each participating hospital, acted as a DICOM node to interface with the hospital PACS. Using this setup, a Python script, leveraging the PyOrthanc library [7], triggered the local Orthanc to query the PACS via a DIMSE (DICOM Message Service Element) call to retrieve LCS imaging data. For each query, the StudyInstanceUID of the corresponding LDCT study was extracted. Once the list of StudyInstanceUIDs was retrieved, another script [8], using Orthanc and PyOrthanc, requested the PACS to transfer the identified studies to a dedicated server within the clinical environment.

This server contains a Karnak [9] instance. Karnak deidentified the data following a dedicated de-identification profile. Key de-identification steps included replacing the original patient ID with a salted hash, substituting the patient’s name with the corresponding DID, retaining essential metadata, such as study dates and descriptions (to distinguish between baseline and follow-up imaging exams), and completely removing private DICOM tags.

## S1.2.2 Non-imaging data

Non-imaging information from eligible participants was transferred from the clinical program RedCap database. During this transfer, all participant data was de-identified using the DIDs. Selected metadata was extracted from the biobank and stored as a json string in the TextValue field of a Structured Reporting (SR) file.

## S1.2.3 Inclusion and exclusion criteria

Only axial CT series were included in LUCQUE. Image series with a total slice count of <150, non-axial acquisition plane or maximum intensity projection (MIP) reconstructions were excluded. Furthermore, studies were excluded if no Lung-RADS score and PLCOm2012noRace score was available. Studies had to include at least two axial series (reconstructed with harder and softer kernel) to be included in LUCQUE.

## S1.3 Data annotation

## S1.3.1 Lesion characteristics

For specifying in which of the 18 lung segments (10 for the right lung and 8 for the left lung) the lesion was located, the RadLex terminology [10] was employed for standardization. Depending on the selected lesion type (solid, non-solid, cyst), further contextual classification menus were displayed. Solid nodules were further characterized as: (1) nodule with complete, central, popcorn, or concentric ring calcification, (2) fat-containing nodule, (4) juxtapleural nodule with smooth margins; and oval, lentiform, or triangular shape, (5) airway nodule (subsegmental), (6) airway nodule (segmental or more proximal) and (7) spiculation. No further classification menu was required for non-solid nodules. Cysts were characterized as: (1) thick-walled cyst (>=2mm - symmetric, asymmetric, or nodular), (2) thin-walled cyst (uniform wall thickness $< 2 ~ \mathrm { m m } )$ , (3) Unilocular cyst, (4) Multilocular cyst and (5) Cyst with opacity (nodular, ground glass, or consolidation).

Table S1: Participants’ characteristics at baseline. <sup>∗</sup>Data are medians and interquartile ranges (IQRs). <sup>∗1</sup> For participants who have stopped smoking.
<table><tr><td rowspan=2 colspan=1>Demographic data</td><td></td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Sex</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Male</td><td rowspan=1 colspan=1>878</td></tr><tr><td rowspan=1 colspan=1>Female</td><td rowspan=1 colspan=1>800</td></tr><tr><td rowspan=1 colspan=1>Age at LDCT*</td><td rowspan=1 colspan=1>65 (61, 69)</td></tr><tr><td rowspan=1 colspan=1>Race or ethnicity</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Caucasian</td><td rowspan=1 colspan=1>1649</td></tr><tr><td rowspan=1 colspan=1>Indigenous</td><td rowspan=1 colspan=1>18</td></tr><tr><td rowspan=1 colspan=1>Asian</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>Black</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>Hispanic</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>Middle Eastern</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>Other</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>Smoking history</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Smoking status</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Active</td><td rowspan=1 colspan=1>1268</td></tr><tr><td rowspan=1 colspan=1>Former</td><td rowspan=1 colspan=1>410</td></tr><tr><td rowspan=1 colspan=1>Number of smoking years*</td><td rowspan=1 colspan=1>45 (40, 50)</td></tr><tr><td rowspan=1 colspan=1>Number of years since smoking was stopped*,*1</td><td rowspan=1 colspan=1>7.0 (3.0, 10.75)</td></tr><tr><td rowspan=1 colspan=1>Medical history</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>COPD, Emphysema, chronic bronchitis</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>404</td></tr><tr><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>1274</td></tr><tr><td rowspan=1 colspan=1>PLCOm2012noRace*</td><td rowspan=1 colspan=1>4.2 (2.7, 7.0)</td></tr></table>

Table S2: Radiological findings in LUCQUE.
<table><tr><td rowspan=1 colspan=1>LungRADs score</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>51</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>83</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1286</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>4A</td><td rowspan=1 colspan=1>94</td></tr><tr><td rowspan=1 colspan=1>4B</td><td rowspan=1 colspan=1>32</td></tr><tr><td rowspan=1 colspan=1>4X</td><td rowspan=1 colspan=1>21</td></tr><tr><td rowspan=1 colspan=1>Findings of lung and pleura</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Emphysema</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>yes</td><td rowspan=1 colspan=1>1423</td></tr><tr><td rowspan=1 colspan=1>no</td><td rowspan=1 colspan=1>127</td></tr><tr><td rowspan=1 colspan=1>unknown</td><td rowspan=1 colspan=1>128</td></tr><tr><td rowspan=1 colspan=1>Findings in the heart</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>yes</td><td rowspan=1 colspan=1>1490</td></tr><tr><td rowspan=1 colspan=1>no</td><td rowspan=1 colspan=1>110</td></tr><tr><td rowspan=1 colspan=1>unknown</td><td rowspan=1 colspan=1>78</td></tr><tr><td rowspan=1 colspan=1>Coronary calcification</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>absent</td><td rowspan=1 colspan=1>81</td></tr><tr><td rowspan=1 colspan=1>low</td><td rowspan=1 colspan=1>793</td></tr><tr><td rowspan=1 colspan=1>moderate</td><td rowspan=1 colspan=1>376</td></tr><tr><td rowspan=1 colspan=1>severe</td><td rowspan=1 colspan=1>214</td></tr><tr><td rowspan=1 colspan=1>unknown</td><td rowspan=1 colspan=1>214</td></tr></table>

## S1.3.2 Reader variability

To access intra- and inter-reader variability, a subset of 51 baseline studies with varying Lung-RADS scores was again annotated by the annotating radiologist (reader 1) with a wash-out period of at least 4 months (reader 1\*) and additionally by a first-year resident in radiology (reader 2).

The inter- and intra-reader analysis was performed for pulmonary nodules and not for cysts to allow better comparison with variability values reported in previous studies [11, 12]. Series with only nodules $< 3 4 \mathrm { m m } ^ { 3 }$ or no findings in the reference (reader 1) and the comparison reading (reader 1\*/reader 2) were regarded as a true negative (TN) series. An overlap-based matching criterion between nodules of reader 1 and reader 1\*/reader 2 was utilized to assess the nodule detection. Nodules were regarded as matched with a $\begin{array} { r } { D i c e = \frac { 2 \times \mathrm { V o l u m e ~ o f ~ i n t e r s e c t i o n } } { \mathrm { T o t a l ~ v o l u m e } } > 0 } \end{array}$ . Multiple nodules were allowed to match the same reference lesion. In such cases, the matching nodule with the largest Dice was selected and the others were disregarded (neither treated as true positives (TPs) or false positives (FPs)). A matched nodule was counted as a TP if the nodule volume was $\geq 3 4 \mathrm { m m } ^ { 3 }$ in either read. Matched nodules were disregarded and not counted as TPs if the nodule had a volume smaller than $3 4 \mathrm { m m } ^ { 3 }$ in both reads. Reference nodules smaller than $3 4 \mathrm { m m } ^ { 3 }$ with no match in the second read were not counted as false negatives (FNs). Likewise, unmatched comparison-read nodules smaller than 34mm<sup>3</sup> were not counted as FPs.

Table S3: Intra-reader and inter-reader variability in nodule detection and segmentation. Reader 1 (first reading) served as the references for the intra-reader and inter-reader detection metrics, respectively. Dice scores were calculated only for true positive nodules. TP = true positive, FN = false negative, FP = false positive.
<table><tr><td>Metric</td><td>Intra-reader</td><td>Inter-reader</td></tr><tr><td>Nodule detection</td><td></td><td></td></tr><tr><td>TP</td><td>56</td><td>32</td></tr><tr><td>FN</td><td>6</td><td>29</td></tr><tr><td>FP</td><td>21</td><td>7</td></tr><tr><td>Sensitivity</td><td>0.90</td><td>0.52</td></tr><tr><td>Precision</td><td>0.73</td><td>0.82</td></tr><tr><td>Mean FP rate per study</td><td>0.41</td><td>0.14</td></tr><tr><td>Mean FN rate per study</td><td>0.12</td><td>0.57</td></tr><tr><td>F1</td><td>0.81</td><td>0.64</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Segmentation overlap (Dice) Median (IQR)</td><td>0.81 (0.77, 0.87)</td><td></td></tr><tr><td>Mean (std)</td><td>0.80 (0.09)</td><td>0.73 (0.64, 0.80) 0.72 (0.11)</td></tr></table>

The number of TP, FP and FN nodules were reported. Additionally, the sensitivity $\scriptstyle ( { \frac { T P } { T P + F N } } )$ , precision $( \frac { T P } { T P + F P } )$ , mean false-positive rate per study $( \frac { \sum _ { \mathrm { s t u d i e s } } \mathrm { n u m b e r ~ o f ~ F P s ~ p e r ~ s t u d y } } { \mathrm { n u m b e r ~ o f ~ s t u d i e s } } )$ , mean false-negative rate per study ( <sup>∑studies</sup> <sup>number</sup> <sup>of</sup> <sup>FNs</sup> <sup>per</sup> <sup>study</sup> ), as well as the nodulelevel F1 score $\scriptstyle \big ( { \frac { 2 I ^ { \prime } P } { 2 T P + F P + F N } } \big )$ were calculated. Sensitivity and precision, as well as the mean FP and FN rates per scan, describe equivalent performance characteristics, differing only in the reference standard used: Reader 1 or Reader 1<sup>∗</sup>/Reader 2. For TP nodules, the intra/inter-reader segmentation performance was evaluated by calculating the mean (standard deviation (std)) as well as the median (0.25-quantile, 0.75-quantile) Dice score. Furthermore, for the TP cases, the intra- and inter-reader nodule classification performance was assessed by comparing the number of alike and different answers for the lesion type, including solid, ground-glass, part-solid nodules.

## S1.4 Data analysis

The lesion volume analysis with PyRadiomics [13] serves as a basic example of how data analysis code can be applied to DICOM objects within PARADIM’s managed execution environment.

The platform enables systematic processing of medical imaging data in a user-friendly model-to-data approach with minimal setup effort. Users simply package their code into a container that complies with the Open Container Initiative (OCI) specifications and push it to a docker registry. The container must be executable (at the series or study level), and all outputs must be stored as DICOM objects with the container’s name as the series description for traceability. In PARADIM, the containerized application is converted into an Apptainer image. The DICOM outputs are automatically assigned to the corresponding Kheops collection. Results (stored as a flat dictionary in the TextValue field of an SR file) can be aggregated across the collection, exported as a single CSV file, and provided to the user for further analysis.

## S2 Results

## S2.1 Reader variability

Intra- and inter-reader analysis results are summarized in Table S3. The intra-reader analysis showed that 56 nodules were detected in both readings, 6 nodules were annotated only in the first reading and 21 nodules were only detected in the second reading. Sensitivity, precision, mean FP rate per study, mean FN rate per study, and F1 score in the intra-reader analysis was 0.90, 0.73, 0.41, 0.12, and 0.81 respectively. For the cases that were annotated in both reading, a median (IQR) and mean (std) Dice score of 0.81 (0.77, 0.87) and 0.8 (0.09) was determined, respectively. Most of these nodules were classified as solid in both readings (n = 48) (Fig. S1(a)). In the inter-reader analysis 32 nodules were annotated by both readers, while 29 nodules were only annotated by reader 1 and 7 only by reader 2. The inter-reader sensitivity, precision, mean FP rate per study, mean FN rate per study and F1 score was 0.52, 0.82, 0.14, 0.57, and 0.64, respectively. For the nodules, annotated by both readers a median (IQR) and mean (std) Dice of 0.73 (0.64, 0.80) and 0.72 (0.11) was calculated, respectively. Nodule

## S3 Discussion

## S3.1 Other accessible lung nodule CT dataset

Datasets [14] such as ANODE09 [15], LungCT-diagnosis [16], SPIE-AAPM [17], RIDER lung CT [18], RIDER PET-CT [19], VIA/ELCAP [20], Public lung database to address drug response [21], QIN lung CT [22, 23], Lung CT segmentation challenge 2017 [24], LNDb [25] are not listed in Table 1 due to their low number of studies (< 300). UniToChest [26] is not included due to missing LCS context. Datasets such as DeepLesion [27] that did not provide full CT volumes were also not included for comparison in Table 1.

## S3.2 Reader variability

Armato et al. [11] reported a mean sensitivity between 0.51-0.83 and a mean FP rate between 0.33-1.39 per case when comparing the radiologist detections to different panel answers. Our inter-reader sensitivity/precision of 0.52/0.82 and mean FN rate of 0.57 per study were in line with these results, while our FP rate per study of 0.14 was slightly better. The median (std) inter-reader Dice score of 0.73 (0.64, 0.80), respectively, were slightly lower with overlapping IQRs with a previously reported median (IQR) inter-reader Dice coefficient of 0.82 (0.77, 0.85) [12]. This highlights that variation in annotations is to be expected and should be considered when evaluating AI models.

![](images/746033fd18c7e69f9f4303d7e79d17035193f19015128decf88d60ff7dae1577.jpg)  
(a)

![](images/17c29fe589ee464308aa23127ca848255c315e8d1d7092f01da7098ab3eb0315.jpg)  
(b)

![](images/1a7aec80f525d5747d38df01119e9360d60d3c39c3a8b7d2287524f9a95d0f7e.jpg)  
(c)

![](images/546304fd4c51ca79dc9dba8b4a80095c02bc598c770f7812913f89091ab0681b.jpg)  
(d)  
Figure S1: (a,c) Intra-reader analysis; (b,d) inter-reader analysis. (a)/(b) Number of reported nodule characteristics for the pulmonary nodules detected in both readings. (c)/(d) Comparison of subcharacteristics for the nodules that were in both readings classified as solid.

![](images/c59f7999e41a297a8bdb1d271fa7fd3938bfab30b94d2f74b3c424d531c7eae0.jpg)  
(a)

![](images/455d62fbf98795a23d1e4865bf0d8963754c4bcacb2a180483a5e47a3081f1c7.jpg)  
(b)  
Figure S2: Volume of findings in mm<sup>3</sup> segmented in both readings of (a) intra-reader analysis and (b) inter-reader analysis.

## References

[1] G. Boily, A.-C. Guédon, L. Nshimyumukiza, et al. Évaluation du projet pilote de dépistage du cancer du poumon: effets sur la santé des participants et impact organisationnel sur le système de santé. https://www.inesss.qc.ca/fileadmin/ doc/INESSS/Rapports/Oncologie/INESSS Depistage\_cancer\_poumon\_EPP\_EP\_VF.pdf, last accessed on 2026-10-27. 2024.

[2] S. Martel, H. Lizotte, G. Boily, et al. “EP. 04A. 04 Real-World Data of LDCT Lung Cancer Screening Implementation in a Public Healthcare System: Québec Three 3-Year Experience”. Journal of Thoracic Oncology 19.10 (2024), S460–S461.

[3] Gouvernement du Québec. Information about lung cancer screening. https: //www.quebec.ca/en/health/advice-andprevention/screening-and-carriertesting-offer/lung-cancerscreening/information-lung-cancerscreening, last accessed on 2026-10-27. 2026.

[4] American College of Radiology Committee. Lung-RADS. https://www.acr.org/Clinical-Resources/Clinical-Tools-and-Reference/Reporting-and-Data-Systems/Lung-RADS. 2022.

[5] L. Chelala, R. Hossain, E. A. Kazerooni, et al. “Lung-RADS version 1.1: challenges and a look ahead, from the AJR special series on radiology reporting and data systems”. American Journal of Roentgenology 216.6 (2021), pp. 1411–1422.

[6] J. Christensen, A. E. Prosper, C. C. Wu, et al. “ACR Lung-RADS v2022: assessment categories and management recommendations”. Journal of the American College ofRadiology 21.3 (2024), pp. 473–488.

[7] G. Couture, Y. Lemaréchal, and P. Després. “PyOrthanc: A Python Interface for Orthanc DICOM Servers”. Journal of Open Source Software 10.106 (2025), p. 7562. DOI: 10.21105/joss.07562.

[8] Y. Lemaréchal. ylemarechal/dicom-transfer. https://github.com/ylemarechal/dicomtransfer.

[9] OsiriX Foundation. Karnak. https://karnak.weasis.org. Last accessed on 2026-10-27.

[10] C. P. Langlotz. “RadLex: a new method for indexing online educational materials”. Radiographics 26.6 (2006), pp. 1595–1597.

[11] S. G. Armato III, R. Y. Roberts, M. Kocherginsky, et al. “Assessment of radiologist performance in the detection of lung nodules: dependence on the definition of ’truth’”. Academic radiology 16.1 (2009), pp. 28–38.

[12] S. S. Yip, C. Parmar, D. Blezek, et al. “Application of the 3D slicer chest imaging platform segmentation algorithm for large lung nodule delineation”. PloS one 12.6 (2017), e0178944.

[13] J. J. Van Griethuysen, A. Fedorov, C. Parmar, et al. “Computational radiomics system to decode the radiographic phenotype”. Cancer research 77.21 (2017), e104–e107.

[14] F. Shaukat, G. Raja, and A. F. Frangi. “Computer-aided detection of lung nodules: a review”. Journal of Medical Imaging 6.2 (2019), pp. 020901–020901.

[15] B. Van Ginneken, S. G. Armato III, B. de Hoop, et al. “Comparing and combining algorithms for computer-aided detection of pulmonary nodules in computed tomography scans: the ANODE09 study”. Medical image analysis 14.6 (2010), pp. 707–722.

[16] O. Grove, A. E. Berglund, M. B. Schabath, et al. “Quantitative computed tomographic descriptors associate tumor shape complexity and intratumor heterogeneity with prognosis in lung adenocarcinoma”. PloS one 10.3 (2015), e0118261.

[17] S. G. Armato III, L. Hadjiiski, G. D. Tourassi, et al. “LUNGx Challenge for computerized lung nodule classification: reflections and lessons learned”. Journal ofMedical Imaging 2.2 (2015), p. 020103.

[18] B. Zhao, L. P. James, C. S. Moskowitz, et al. “Evaluating variability in tumor measurements from same-day repeat CT scans of patients with non–small cell lung cancer”. Radiology 252.1 (2009), pp. 263–272.

[19] P. Muzi, M. Wanner, and P. Kinahan. “Data from RIDER Lung PET-CT”. The Cancer Imaging Archive (2015). DOI: https://doi.org/10. 7937/k9/tcia.2015.ofip7tvm.

[20] C. I. Henschke, D. I. McCauley, D. F. Yankelevitz, et al. “Early lung cancer action project: a summary of the findings on baseline screening”. The oncologist 6.2 (2001), pp. 147–152.

[21] A. P. Reeves, A. M. Biancardi, D. Yankelevitz, et al. “A public image database to support research in computer aided diagnosis”. 2009 Annual international conference of the IEEE engineering in medicine and biology society. IEEE. 2009, pp. 3715–3718.

[22] J. Kalpathy-Cramer, S. Napel, D. Goldgof, et al. “QIN multi-site collection of Lung CT data with Nodule Segmentations (QIN-LungCT-Seg) (Version 3) [Data set]”. Cancer Imaging Archive 10 (2015), K9. DOI: https://doi.org/10.7937/ k9/tcia.2015.1buvfjr7.

[23] D. Goldgof, L. Hall, S. Hawkins, et al. “Data From QIN LUNG CT” (2015). DOI: 10.7937/K9/TCIA.2015.NPGZYZBZ.

[24] J. Yang, G. Sharp, H. Veeraraghavan, et al. “Data from lung CT segmentation challenge.” The Cancer Imaging Archive (2017). DOI: https:// doi.org/10.7937/K9/TCIA.2017.3R3FVZ08.

[25] J. Pedrosa, G. Aresta, C. Ferreira, et al. “LNDb: a lung nodule database on computed tomography”. arXiv preprint arXiv:1911.08434 (2019).

[26] H. A. H. Chaudhry, R. Renzulli, D. Perlo, et al. “Unitochest: A lung image dataset for segmentation of cancerous nodules on CT scans”. International Conference on Image Analysis and Processing. Springer. 2022, pp. 185–196.

[27] K. Yan, X. Wang, L. Lu, et al. “DeepLesion: automated mining of large-scale lesion annotations and universal lesion detection with deep learning”. Journal ofmedical imaging 5.3 (2018), pp. 036501–036501.

## S4 Annotation guide

You are annotating the Lung Cancer screening dataset from QUÈbec (LUCQUE). The low-dose computed tomography (LDCT) data were acquired in the lung cancer screening project of Québec [1].

The LUCQUE dataset is designed for reliable AI evaluation in lung cancer screening. To assess the performance of AI systems for tasks such as lung nodule detection, segmentation, and Lung-RADS classification, detailed reference annotations regarding the location, size, and type of lung nodules are necessary.

Your tasks will be to segment lung nodules and atypical pulmonary cysts, and provide additional characterization of the segmented findings. The annotation guidelines for the LUCQUE dataset are based on the Lung-RADS v2022, with all referenced categories defined according to this standard [2].

## Annotation process

Please annotate all pulmonary nodules and tumor masses (for simplicity both referred to as “nodules” in the following) that have a diameter greater than 3 mm. If you detect nodules with a diameter smaller than 3 mm, you may also annotate them. Additionally, please annotate any atypical cysts that you encounter. Nodules and cysts are referred to as findings in the following.

The annotation process will be explained below.

## 1) Access and select CT study

Connect to the Université Laval VPN and paste the URL you were provided into your browser. Next, you will then see your study list. To begin the annotation process, select a CT study and then click on “Segmentation-workflow”. If multiple studies are available for one patient, select the baseline study. A new window will open.

![](images/5499e05b0fba50cb71abe89aaee3a66da2659b65a2b79363c3a15eea68208372.jpg)

## 2) Select CT series

For each study, there are two series available: one reconstructed using a harder kernel and the other using a softer kernel. Annotations should be performed on

the series reconstructed with the harder kernel. This series can in most cases be identified by its series description, which e.g., contains the word "Lung".

## 3) Add segmentation

If you detect one or multiple findings, please select “Add Segmentation” from the menu on the right.

![](images/0f0c9b9617566ee75a1b8a47d5d9fbb39ee159ccdb4f94b84c87851d0bee886f.jpg)

If no findings are identified in the study, please skip steps 3 through 11 and proceed directly to step 12.

## 4) Add finding

For each detected finding, select “Add Annotation” in the menu. Steps 4 through 11 need to be repeated for all findings.

![](images/e38c7e0dec0b45567280e720fa880976d334def6891837168a9fc972c448944e.jpg)

## 5) Segmentation of the finding

To begin the segmentation, select the annotation you created in step 4. Then, click on the brush tool and segment the finding. You can choose the segmentation tools you would like to use from the menu. Segmentations should be conducted at the voxel level without any additional margins.

![](images/7b94b7a55077f5b87e17a24ddbc5a6dd69a64e84759f5c96869fb7ab111ddef5.jpg)

## 6) Edit finding characteristic

After completing the segmentation, select in the findings menu and then choose “edit”.

![](images/72732b1bfa031434fe9f861d8a092c0a9686c80da7e48f5872e29c8181fa26d7.jpg)

![](images/2b4bc62e875a1bd200f980ca12bf42ed71b938e66923362394346b0d87092ee6.jpg)  
A new menu will appear asking for more details about the finding (steps 7-11).

## 7) Select the finding type

Select the type of finding you just segmented:

• Solid pulmonary nodule

• Non-solid pulmonary nodule

• Atypical pulmonary cyst

## 8) Select the lung segment

Define the lung segment where the finding is located for “Spatial and Rational Concept”:

• apical segment of upper lobe of right lung

• posterior segment of upper lobe of right lung

• anterior segment of upper lobe of right lung

• lateral segment of middle lobe of right lung

• medial segment of middle lobe of right lung

• superior segment of lower lobe of right lung

• medial basal segment of lower lobe of right lung

• anterior basal segment of lower lobe of right lung

• lateral basal segment of lower lobe of right lung

• posterior basal segment of lower lobe of right lung

• apicoposterior segment of upper lobe of left lung

• anterior segment of upper lobe of left lung

• superior division of upper lobe of left lung (referring to the superior lingular segment of the left upper lobe)

• inferior segment of lingula (referring to the inferior lingular segment of the left upper lobe)

• superior segment of lower lobe of left lung

• anteromedial basal segment of lower lobe of left lung

• lateral basal segment of lower lobe of left lung

• posterior basal segment of lower lobe of left lung

## 9) Label finding characteristics

Depending on the finding type you selected in step 7, diaerent options will be available to further characterize the finding.

If “solid nodule” was selected as the finding type, further characterize it by selecting “yes” or “no” for the following characteristics:

• part solid pulmonary nodule (“yes” if the solid component is part of a partsolid nodule; see “Additional notes”)

• nodule with complete, central, popcorn, or concentric ring calcification

• fat-containing nodule

• juxtapleural nodule with smooth margins; and oval, lentiform, or triangular shape

• airway nodule: subsegmental

• airway nodule: segmental or more proximal

• spiculation

If “non-solid nodule” was selected as the finding type, no further characterization is required, except for information indicating if the non-solid component is part of a part-solid pulmonary nodule (see “Additional notes”).

If “atypical cyst” was selected in step 6, you should further characterize it by selecting “yes” or “no” for the following characteristics:

• thick-walled cyst (>=2mm - symmetric, asymmetric, or nodular)

• thin-walled cyst (uniform wall thickness < 2 mm)

• Unilocular cyst

• Multilocular cyst

• Cyst with opacity (nodular, ground glass, or consolidation)

By default, all responses are set to "no". To indicate the correct finding type, change the answer to "yes".

![](images/cf9ca6b9f8a5c3344269977957b90400a584a146a1c9a931c49e1fd8c003cbaa.jpg)

## 10) Add comment (not mandatory)

If you have any notes regarding your annotations or the image (e.g., difficulties with annotation due to significant breathing artifacts), please feel free to document this in the comment field (in English).

![](images/ce41eab2311e90b60cc517c7a3792f4f8a70837dc92136d998706d666a392b60.jpg)

## 11) Save

Select “Save” to save your annotations.

![](images/63b2d94a0848e5d72f6510db54891a889635c6e22adb635a7f90d4923f0ae5cd.jpg)

## 12) Upload annotation

Once you have completed the annotation of all findings for the series, click on … located next to “Segmentation” and then select “Complete Segmentation” to upload your results.

![](images/3671ad24130a0caff409b53271965e48298ae11f961a5e4f45c1dccfc7a597ab.jpg)

## Additional notes

1. For part-solid nodules, segment the solid and non-solid components as two separate findings. Begin by outlining the outer non-solid part and select "yes" for the characteristic "part-solid". Next, add a new finding, segment the solid part and select "yes" for the characteristic "part-solid" as well.

![](images/ea98d1461fd00c88e365e523fcb445140b4a190ed20eebe1c5137ac9ee9fc35a.jpg)

2. According to Lung-RADS v2022 guidelines, cavitary nodules are managed in the same way as solid nodules. Therefore, annotate them using the same criteria as for solid nodules (refer to note 12.d in Lung-RADS v2022 [2]).

3. When dealing with a cyst that has an associated nodule (see note 12.e in Lung-RADS v2022 [2]), annotate the cyst and the nodule independently from one another.

## References

[1] Gouvernement du Québec. Information about lung cancer screening. https://www.quebec.ca/en/health/advice-and-prevention/screeningand-carrier-testing-offer/lung-cancer-screening/information-lung-cancer-screening.

[2] J. Christensen, A. E. Prosper, C. C. Wu, et al. “ACR Lung-RADS v2022: assessment categories and management recommendations”. Journal of the American College of Radiology 21.3 (2024), pp. 473–488.