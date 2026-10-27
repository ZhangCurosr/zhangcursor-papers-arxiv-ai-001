# Explainable Multimodal AI Approach to Reduce Unnecessary Breast Biopsies

Sadam Hussain<sup>1[0000−0002−3453−0785]⋆</sup>, Mansoor Ali<sup>1[0000−0002−4669−7267]</sup>, Daly Betzabeth Avendaño-Avalos<sup>2[0000−0002−8433−7211]</sup>, and Jose Gerardo Tamez Peña<sup>2[0000−0003−1361−5162]</sup>

<sup>1</sup> School of Engineering and Sciences, Tecnologico de Monterrey, Monterrey, Nuevo Leon, Mexico

a01753094@exatec.tec.mx

2 School of Medicine and Health Sciences, Tecnologico de Monterrey, Monterrey, Nuevo Leon, Mexico

Abstract. Recently, there has been rise in multimodal vision-language models for computer-aided diagnosis (CAD), yet their application in breast diagnosis, particularly for predicting biopsy outcomes remains limited. This study presents a multimodal vision-language approach, incorporating Grad-CAM as an Explainable AI (XAI) technique, to predict benign tumors in patients recommended for biopsy, especially those classified as BI-RADS 4 and 5. A novel multimodal framework was developed using Vision Transformer (ViT) for imaging feature extraction and DeBERTa-V3-Large for textual embeddings. The model was trained and evaluated on a comprehensive in-house dataset comprising 1,632 patients, each with four mammography views and a corresponding radiology report. The features of both modalities were extracted independently and fused by late fusion before classification. Grad-CAM was used to generate visual explanations highlighting key regions influencing model predictions. The proposed model demonstrated high sensitivity 90%, 95%, 99%, and 100% with corresponding specificities of 40%, 36%, 18%, and 11%, respectively. The best AUC achieved was 0.72 by multimodal approach on the test set, outperforming the image-only (0.66) and text-only (0.68) baselines. The proposed multimodal architecture, enhanced with Grad-CAM, demonstrates a strong potential to identify benign tumors among high-risk patients, ofering interpretable insights and thereby helping reduce unnecessary biopsies, patient anxiety, and healthcare costs.

Keywords: Breast Cancer · Biopsy Recommendation · Vision-Language Model · Multimodality · Biopsy Reduction

## 1 Introduction

Breast cancer is the most common disease among women in the United States and the world [15]. Diferent imaging modalities have been used to diagnose breast cancer in the early stage. Mammography is one of the common imaging techniques used for earlier cancer diagnosis and has shown potential benefits in clinical settings [15]. One of the main problems with mammography is that most benign findings are recommended for biopsy. BI-RADS was created to describe the various stages of cancer appropriately to provide better patient care [7]. BI-RADS category 4—is divided into four subsections; 4, 4a, 4b, and 4c, represents a suspicious abnormality, which is therefore recommended for biopsy. Among the recommended biopsies in the BI-RADS 4 category in general, 75% of the BI-RADS category 4 findings are proven to be benign [7]. In the United States alone, more than 970,000 breast biopsies are unnecessary [2]. This causes a high false positive rate, which leads to additional expenditure, clinical procedures, and further stress. Diferent CAD systems have been used to improve the diagnosis and prognosis of breast cancer in various stages using numerous imaging modalities and clinical data [14]. The latest breakthroughs in machine learning (ML) and deep learning (DL) models have enabled the implementation of CAD models [8]. The ever-burgeoning capacity of computational resources and access to a large amount of data creates extraordinarily favorable conditions for DL models in various image classification, segmentation, and object detection tasks and enables extracting insight from textual data in parallel [19]. DL models use Convolutional Neural Networks (CNNs) to automatically extract imaging features, in contrast to conventional hand-crafted techniques [8]. CNN-based advanced architectures have shown potential in diferent breast cancer diagnosis and prognosis applications [17], such as breast cancer classification [10], risk prediction [19], survival [1], and recurrence prediction [11]. In recent studies, a surge in the use of vision-language models for diferent applications in classification, detection, and segmentation has been observed in biomedical and general-purpose applications [20]. However, it is rare in breast cancer diagnosis and prognosis applications due to the lack of multimodal medical data.

BI-RADS 4 and associated categories of patients are recommended for biopsy. Few studies have used DL approaches to predict biopsy outcomes [12]; however, to the best of our knowledge, no study has employed a multimodal approach combined with an XAI technique for this purpose. Therefore, a novel multimodal framework that employs a ViT to extract mammographic features and DeBERTa-V3-Large to obtain embeddings from radiological reports is proposed. Furthermore, an XAI technique—Grad-CAM, is used to visualize the image regions that contribute to the model’s prediction for interpretability purposes. ViT has demonstrated efective performance in tumor classification, and Grad-CAM—an XAI technique, is applied to generate attention-based heatmaps. Grad-CAM highlights the critical imaging regions that contribute the most to the model’s predictions, thereby ofering interpretability and fostering trust in automated decision-making systems [13].

In the proposed framework, features from each modality are processed independently and subsequently fused using a late-fusion strategy through feature concatenation before the final classification layer. This multimodal integration aims to address the limitations of unimodal approaches by incorporating both mammographic imaging data and radiological reports. Four mammographic views—right cranial-caudal (R-CC), the right mediolateral-oblique (R-MLO), the left cranialcaudal (L-CC), and the left mediolateral-oblique (L-MLO), are used as inputs to the model. The objective of this study is to use a vision–language modeling approach to assess biopsy outcomes for BI-RADS 4 and its subcategories, as well as BI-RADS 5, with the goal of reducing unnecessary biopsies for patients who do not have breast cancer.

## 2 Related work

The classification of benign tumors from breast biopsy recommended that patients can be seen in ultrasound imaging studies using ML. Investigators in a few studies [21], have explored the possibilities of identifying benign biopsy from ultrasound images using ML methods. However, it is uncommon to classify biopsy-recommended patients in mammography images using ML techniques. Only a handful of studies have used these techniques to investigate this problem further. An unnecessary breast biopsy reduction problem has been explored using DL on digital mammograms quite recently. Meng and colleagues [21] explored the possibility of classifying patients based on BI-RADS 4 using a deep transfer learning technique. They particularly used a pre-trained Inception-V3 model. The best AUC reported for this model is 0.86. The authors in a study [18], developed a patch-based DL pipeline, where they divided mammography images into diferent patches to obtain local features. They aggregated feature maps using a shallow convolutional aggregation network, which includes two convolutional layers, global average pooling, and a classification layer. They used DenseNet-161 for extracting local features and reported an AUC of 0.799. In another study [4], authors designed a decision support tool for risk categorization, and reduction of unnecessary biopsies in mammographic data. An AUC of 0.93 is reported in this study. In a recent study conducted by Liu and colleagues [9], used a simple DL VGG16 architecture on digital mammograms to classify benign tumors among malignant tumors in the category 4 BI-RADS. They reported the best AUC of 0.72 for the main task of classifying benign tumors among various types of tumors, along with a few other complementary tasks.

Limited research has been conducted on the identification of the biopsy outcome using ML from BI-RADS 4 patients. Few of the studies that have used DL to predict biopsy outcomes from BI-RADS 4 patients are presented as follows. The authors [12] used deep transfer learning on digital mammograms for patients with BI-RADS 4 recommended for biopsy. An Inception-V3 model was particularly used for this objective. In another study [18], the authors extracted patches from mammography images to capture important local features. They aggregated these patches using a shallow convolutional aggregation network and subsequently fed the aggregated feature maps into a DenseNet-161 architecture for final outcome classification. In another work [4], researchers designed a decision support tool used to stratify risk and reduce biopsy in BI-RADS 4 mammograms. The study was limited in the sense that the model was built and validated internally. In a recent preliminary study [9] investigators designed a DL model based on digital mammograms to avoid unnecessary breast biopsies. This study performed four binary classification tasks—the main task being the classification of tumors as benign versus all other types of lesions, achieving a best AUC score of 0.72. However, the study had limitations, such as a limited internal dataset consisting only of mammograms and using a simple VGG16 architecture. In summary, existing studies on biopsy outcome prediction from BI-RADS 4 mammograms remain limited in several respects. Most approaches rely solely on mammographic images and do not leverage the complementary information contained in radiological reports. Furthermore, many studies are developed and evaluated using relatively small, single-institution datasets, which may limit generalizability. In addition, model interpretability has received limited attention, reducing transparency and potentially hindering clinical adoption. These limitations highlight the need for multimodal and explainable approaches for biopsy outcome prediction.

## 3 Methodology

This section provides a detailed description of the dataset used, processing methods, and data augmentation approach applied on the data; the proposed visionlanguage architecture; and implementation details. The architecture involves integrating four-view mammography (LCC, LMLO, RCC, RMLO) and radiological reports using the late fusion technique.

## 3.1 Dataset Overview

The TecSalud hospital’s at Tecnologico de Monterrey provided the dataset for this study. Together with digital mammograms, it also comprises digital breast tomosynthesis (DBT), digital mammography (DM), and ultrasound (US) based radiological reports. All reports and mammograms were anonymized to maintain privacy. The reports were gathered between 2014 and 2019, originally written in Spanish. Google Translate was used to translate them into the English, and bilingual radiologists then confirmed the translation. Two experienced radiologists wrote and assessed each report. The dataset started with 42000 examples. Only biopsy-recommended patient data were extracted. The four views (LCC, LMLO, RCC, and RMLO), as well as the corresponding radiological reports containing detailed descriptions of the diagnosis—including BI-RADS classifications, breast density scores, patient ages, family cancer histories, lesion laterality, biopsy recommendations, among others were combined with each patient’s data. The laterality of the lesion shows whether the patient has ever had breast cancer. If a patient had cancer in either breast, it is indicated by the laterality of the lesion. Three values are used to describe laterality: 0, 1, and 2. Cancer in the left breast is represented by the number 0, cancer in the right breast by the number 1, and cancer in both breasts by the number 2. The purpose of laterality was to determine whether the patient had cancer at all. Furthermore, 1,632 cases comprising a total of 6,528 mammograms, along with their associated radiological reports, were included after preprocessing. The preprocessing steps involved removing reports with missing values, duplicate records, and cases assigned BI-RADS categories 0 (incomplete assessment), 1 (negative), 2 (benign finding), 3 (probably benign), and 6 (biopsy-proven malignancy). Thus, the dataset contains the remaining categories of BI-RADS 4 and associated categories 4A, 4B, 4C, and BI-RADS 5. The distribution of BI-RADS used in this study can be seen in the Fig. 2. The patients, who were all female and used for the final model building and assessment, had an average age of 51.52, a standard deviation (SD) of 11.44, and an age range of 22-93. The report is a sample and can be used for demonstration purpose. The detailed characteristics of the dataset are defined in Table 1. The inclusion-exclusion criteria of the data used in this experiment can be seen in the Fig. 1.

![](images/30158da1bc83e38bbf40dd0f9e84173536c7862c3df1fa21b9b23621627c904d.jpg)  
Fig. 1. Dataset inclusion-exclusion criteria for false positive biopsy reduction multimodal model.

## 3.2 Data Preprocessing

Each patient’s record had a concise clinical indication, personal risk history, imaging descriptions, findings, and diagnostic impressions derived from the radiological reports. The BI-RADS classifications, breast density scores, patient ages, and family cancer histories were collected from the reports. These extracted data were used as input to the multi-modal architecture, along with multi-view mammograms.

The mammography dataset was subjected to various augmentation approaches, meticulously selected to bring diversity and resilience into the training material. Both horizontal and vertical mirroring were used to display the mammograms in various orientations to the model. In addition, a random scaled cropping technique was applied to ensure that the model remains invariant to changes in scale and translation. The cropping was specifically targeted to achieve a size of 224x224 pixels. In addition, dynamic modifications were incorporated into the images, including shifting, scaling, and rotating. The rotations ranged from 0 to 90 degrees, while the scaling factor varied from 0.8 to 1.2 times the original size. These diverse augmentation techniques successfully enlarged the training dataset. The augmentation strategy in this study introduced the model to several views, which prevents overfitting and improves the model’s ability to handle diferent input types, thus enhancing its overall performance.

![](images/f41d3134b09b6321ac1b5866ba457700f9ef9e3c8dd18dc49b51f4bafdd0fa39.jpg)  
Fig. 2. The data distribution of BI-RADS categories.

## 3.3 Model Architecture

This study introduces a multimodal DL architecture aimed at classifying or predicting biopsy outcomes for BI-RADS 4 and 5 lesions. The model employs distinct DL components to extract features from mammography images and textual data. In the imaging component, a pre-trained ViT architecture [3], specifically using the ViT-base-patch16-224-in21k model was used to improve the feature extraction capability of the network. This model is pre-trained on ImageNet-21k (14 million images, 21,843 classes) at a resolution of 224x224. The ViT processes input mammography images and leverages its powerful attention mechanisms to capture intricate details and relationships within the images, complementing the classification task. In this architecture, the initial layers of the ViT were frozen to retain the pre-trained features and unfroze the last 11 layers to allow fine-tuning on the biopsy outcome prediction task. This approach enables the model to leverage both pre-trained knowledge and task-specific learning for improved classification performance. The ViT processes the mammography images to extract detailed feature representations. These image features are then fused with textual features extracted from clinical reports.

Table 1. Characteristics of the dataset.
<table><tr><td>Characteristics of Dataset</td><td>Details</td></tr><tr><td>No. of Examples</td><td>1632</td></tr><tr><td>Sex</td><td>Women (1632), 100%</td></tr><tr><td>Mexican Cohort</td><td>North East Mexico</td></tr><tr><td>Age (mean), SD, Range</td><td>51.52, 11.44, 22-93</td></tr><tr><td>Biopsy Recommendation</td><td>1632</td></tr><tr><td>Family History of Cancer</td><td>No: 1448, Yes: 184</td></tr><tr><td rowspan="4">Breast Density Distribution (ACR)</td><td>Fatty: 62</td></tr><tr><td>Scattered fibroglandular tissue: 415</td></tr><tr><td>Heterogeneously dense: 793</td></tr><tr><td>Extremely dense: 362</td></tr><tr><td rowspan="4">Distribution of BI-RADS</td><td>4: 131</td></tr><tr><td>4A: 341</td></tr><tr><td>4B: 477</td></tr><tr><td>4C: 151 5: 532</td></tr></table>

A variant of DeBERTa [6], DeBERTa-V3-Large [5] is used to extract embeddings from clinical reports. The DeBERTa-V3-Large model comprises 24 layers with a hidden size of 1024. It contains approximately 304 million backbone parameters and an embedding layer with 131 million parameters, derived from a vocabulary of 128,000 tokens. DeBERTa\_V3 is pre-trained on large-scale general-domain corpora using a replaced token detection objective, enabling it to capture rich contextual and semantic representations from clinical text. It is subsequently fine-tuned to better adapt to domain-specific textual inputs. The extracted image features from the ViT, with a dimension of 768, are concatenated with the textual features from DeBERTa-V3-Large, resulting in a unified multimodal feature vector. This vector is then passed through a linear classification layer to predict biopsy outcomes. By integrating ViT and DeBERTa-V3-Large, the model improves its ability to represent and distinguish relevant multimodal patterns, enabling robust analysis for breast cancer diagnosis.

This method enables the separate extraction of mammography and textual characteristics, which are subsequently combined through a late feature fusion mechanism. More precisely, the ViT extracts mammography features, combined with the textual features processed by the DeBERTa-V3-Large. The combined characteristics are then entered into an Artificial Neural Network (ANN) structure for the ultimate categorization assignment. This integrated methodology capitalizes on the advantages of utilizing both visual and textual data, facilitating a more resilient and all-encompassing examination to diferentiate between non-cancerous and cancerous tumors. The workflow is depicted in Fig. 3.

![](images/2b8bfe6592ea9a1a1c54e57d312191ad7c6ef3a2bdfdfe96c97f271038098ec0.jpg)  
Fig. 3. In the first box, the proposed multimodal approach used for breast biopsy outcome predictions is represented. ViT [3] was used to extract embeddings from four views (R-CC, R-MLO, L-CC, L-MLO) of digital mammograms and the DeBERTa-V3-Large [5] model to extract textual embeddings from radiological reports. Afterwards, late feature fusion technique to combine features from both modalities for the final biopsy outcome classification was used. The second box represents the GRAD-CAM applied to mammograms to explain the model decision and subsequently verified by the radiologist.

In this study, a late fusion strategy was employed to integrate features from multiple modalities before classification. The integration process involved concatenating features extracted from diferent sources—mammograms and radiological reports. The details of a late feature fusion approach and classification process are mentioned below.

Feature Extraction: Let $\mathcal { F } _ { \mathrm { i m g } } \in \mathbb { R } ^ { d _ { \mathrm { i } } }$ <sup>mg</sup> represent the feature vector extracted from the image modality using a ViT, where $d _ { \mathrm { i m g } } = 7 6 8$ . Let $\mathcal { F } _ { \mathrm { t e x t } } \in \mathbb { R } ^ { d _ { \mathrm { t e x t } } }$ denote the feature vector extracted from the radiological reports using DeBERTa-V3-Large model, where $d _ { \mathrm { t e x t } } = 1 0 2 4$

Feature Concatenation: The combined feature vector $\mathcal { F } _ { \mathrm { c o n c a t e n a t e d } }$ is obtained by concatenating ${ \mathcal { F } } _ { \mathrm { i m g } }$ and $\mathcal { F } _ { \mathrm { t e x t } }$

$$
\mathcal { F } _ { \mathrm { c o n c a t e n a t e d } } = c o n c a t [ \mathcal { F } _ { \mathrm { i m g } } , \mathcal { F } _ { \mathrm { t e x t } } ] .\tag{1}
$$

Therefore, the dimensionality of $\mathcal { F } _ { \mathrm { c } }$ <sub>oncatenated</sub> is given by:

$$
d _ { \mathrm { c o m b i n e d } } = d _ { \mathrm { i m g } } + d _ { \mathrm { t e x t } } = 7 6 8 + 1 0 2 4 = 1 7 9 2 .\tag{2}
$$

Classification: The final prediction $\hat { y }$ is computed by applying a linear transformation to the concatenated feature vector followed by a softmax activation function. Formally, let $\mathbf { W } \in \mathbb { R } ^ { C \times d _ { \mathbf { \lambda } } }$ <sup>combined</sup> be the weight matrix and b $\in \mathbb { R } ^ { C }$ be the bias vector, where C represents the number of classes. The prediction $\hat { y }$ is given by:

$$
\hat { y } = \mathrm { s o f t m a x } ( \mathbf { W } \cdot \mathcal { F } _ { \mathrm { c o n c a t e n a t e d } } + \mathbf { b } ) ,\tag{3}
$$

where softmax is the activation function that converts the linear output into class probabilities.

## 3.4 Transparency and Integration of Grad-CAM in Breast Cancer Diagnosis:

Transparency in Decision-Making One of the primary challenges of implementing AI in healthcare is the "black-box" nature of complex models, which often lack interpretability. Grad-CAM helps address this by introducing transparency into decision-making. In the context of breast cancer diagnosis and prognosis, this transparency is crucial, as it enables researchers and clinicians to understand the basis of the system’s predictions. By generating heatmaps, Grad-CAM provides intuitive visual aids that highlight critical regions in mammograms, thus bridging the gap between complex AI models and their practical applications in healthcare.

Integration of Grad-CAM for Enhanced Interpretability: To improve the model’s transparency and assist medical professionals in making accurate decisions, Grad-CAM is integrated with outcomes from linear support vector machines. The method generates heatmaps that emphasize the regions in imaging data—such as mammograms—that the model focuses on while predicting tumor presence. The Grad-CAM heatmap is computed as follows:

$$
\mathrm { G r a d \mathrm { - } C A M } ( i , j ) = \mathrm { R e L U } \left( \sum _ { k } \alpha _ { k } A _ { i , j } ^ { k } \right) ,\tag{4}
$$

where $A _ { i , j } ^ { k }$ represents the activation maps of the final convolutional layer at spatial location $( i , j )$ , and $\alpha _ { k }$ is the importance weight for feature map $k ,$ calculated as:

$$
\alpha _ { k } = \frac { 1 } { Z } \sum _ { i } \sum _ { j } \frac { \partial y ^ { c } } { \partial A _ { i , j } ^ { k } } ,\tag{5}
$$

where $y ^ { c }$ is the class score for class $c ,$ and $Z$ is the normalization factor over the spatial dimensions.

To further enhance interpretability, an integrated explanation is constructed by combining the original mammogram with the Grad-CAM heatmap. This is expressed as:

$$
I E = \beta \cdot \mathrm { O r i g i n a l ~ I m a g e } + ( 1 - \beta ) \cdot F H ,\tag{6}
$$

where $I E$ is the integrated explanation, F H is the Grad-CAM-generated heatmap, and $\beta$ is a weighting factor controlling the contribution of the original image and the heatmap.

By incorporating Grad-CAM into the diagnostic workflow, clinicians gain valuable visual explanations of AI predictions, fostering trust in AI-driven models. The proposed framework supports the development of automated systems for breast cancer diagnosis, ensuring interpretable and reliable outcomes to aid in smart healthcare.

## 3.5 Implementation Details

An NVIDIA RTX A4000 GPU with 16 GB memory, PCI Express Gen 4, and third-generation tensor CUDA cores for both training and testing all models were used. The proposed network architecture utilized a ViT backbone to extract features from mammograms, while DeBERTa-V3-Large was used to extract characteristics from textual data. The models underwent training using pre-existing weights. The Adam optimizer was employed with a learning rate of 0.0005, and a batch size of 4 was used. The training process involved training the models for a total of 30 epochs. The activation function utilized was the Rectified Linear Unit (ReLU), with a dropout rate of 0.2. The loss function employed was Binary Cross Entropy (BCE) with logits. The formula for this loss is provided in equation 7. The dataset was partitioned into training, validation, and testing sets, with proportions of 60%, 20%, and 20% accordingly. The evaluation of model performance was conducted using the testing set.

$$
\mathcal { L } _ { b c e } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } y _ { i } \cdot \log ( p ( y _ { i } ) ) + ( 1 - y _ { i } ) \cdot \log ( 1 - p ( y _ { i } ) ) .\tag{7}
$$

## 4 Results

A retrospective dataset of 1632 patients, which totals 6528 mammograms (four views per patient) and 1632 complete radiological reports were used. The dataset included 131 (8.02%) BI-RADS 4 cases, 341 (20.89%) BI-RADS 4A cases, 477 (29.22%) BI-RADS 4B cases, and 151 (9.25%) BI-RADS 4C cases, as well as

532 (32.59%) BI-RADS 5 cases. The mean age of patients was 51.52 years with a standard deviation of 11.44 years, and the age range was 22 to 93 years.

In proposed architecture for mammography feature extraction, the initial layers of the ViT were frozen to retain the pre-trained features and unfroze the last 11 layers to allow fine-tuning on the specific task. This approach enables the model to leverage both pre-trained knowledge and task-specific learning for improved classification performance. Additionally, the DeBERTa-V3-Large model for processing the radiological reports was used—extracting textual features to complement the image features.

To ensure model interpretability, Grad-CAM was applied to generate heatmaps, visualizing the specific regions of mammography images that most influenced the predictions. This added layer of explainability facilitates understanding of the model’s decision-making process, which is particularly important in clinical applications where trust and transparency are critical. These heatmaps can also aid clinicians in identifying imaging patterns relevant to biopsy outcome predictions, further enhancing the model’s usability.

The proposed multimodal classifier, which combines image features from mammograms with textual data from radiological reports, achieved an accuracy score of 87%. The highest precision obtained was 94%, indicating a high rate of correctly identified positive cases. The best AUC score for predicting biopsy outcome was 0.72, achieved on the testing dataset, which constitutes 20% of the total dataset. The ROC curve representing the ratio between true positive rate and false positive rate can be seen in Fig. 4. This AUC value demonstrates the model’s ability to distinguish between positive and negative biopsy outcomes efectively.

The image-only model demonstrated modest specificity improvements across increasing sensitivity thresholds, with specificity scores of 1.0%, 3.0%, 11.0%, and 25.0% at fixed sensitivities of 100%, 99%, 95%, and 90%, respectively. This indicates a limited reduction in unnecessary biopsies when using imaging data alone. In contrast, the text-only model achieved higher specificity values of 12.0%, 15.0%, 24.0%, and 37.0% at the same sensitivity levels, suggesting a greater capability to reduce false positives from radiology reports alone.

The multimodal model consistently outperformed both unimodal approaches, reaching specificity scores of 11.0%, 18.0%, 36.0%, and 40.0% at 100%, 99%, 95%, and 90% sensitivity, respectively. These findings highlight that up to 40% of unnecessary biopsies could be avoided while maintaining 90% cancer detection, demonstrating the potential of multimodal learning to enhance clinical decisionmaking. A summary of these results, along with AUC scores, is provided in Table 2.

Overall, proposed model’s performance demonstrates a significant advancement in the biopsy outcome predictions of breast cancer patients, providing a robust tool for clinicians to enhance decision-making processes. The integration of multimodal data, utilizing both the ViT for image features and the DeBERTa-V3-Large model for textual features, the fine-tuning of these models, and the use of Grad-

Table 2. Specificity (%) at fixed sensitivity levels and AUC scores for each model. Higher specificity at high sensitivity indicates better reduction of false positives while maintaining diagnostic coverage.
<table><tr><td colspan="5">Model 100% Sens. 99% Sens. 95% Sens. 90% Sens. AUC</td></tr><tr><td>Image Model(ViT)</td><td>1.3</td><td>3.4</td><td>11.2</td><td>25.1 0.663</td></tr><tr><td>Text Model(DeBERTa)</td><td>11.6</td><td>15.3</td><td>24.3 37.0</td><td>0.681</td></tr><tr><td>Multimodal Model</td><td>11.2</td><td>17.7</td><td>35.8 40.4</td><td>0.721</td></tr></table>

CAM for interpretability were crucial in achieving these results. These contributions pave the way for future research and potential clinical applications.

## 5 Discussion

In this work, a multimodal DL and LLM-based model was developed that integrates mammography images and radiological reports. The objective of this study was to create an automated DL pipeline that can help categorize BI-RADS 4 patients as benign, with the goal of reducing unnecessary breast biopsies—thereby alleviating patient anxiety, physical discomfort from biopsy procedures, and associated healthcare costs. To this end, a dataset comprising patients with BI-RADS 4 and 5 assessments, covering subcategories 4, 4A, 4B, and 4C were collected. A tumor classification workflow with direct clinical implications, was built and evaluated across a range of diagnostic sensitivity levels.

In addition to the multimodal model, image-only and text-only models were independently trained and evaluated. While the image-only model demonstrated limited specificity improvements, the text-only model achieved better performance in reducing false positives when used in isolation. This supports the hypothesis that clinical context captured in radiology reports complements image features, and motivates the integration of both modalities. As a novel multimodal approach, this study builds upon previous unimodal research [9], and demonstrates that SOTA DL models can efectively leverage digital mammograms and radiological narratives to support decision-making in BI-RADS 4 assessments. The proposed multimodal model achieved strong diagnostic performance and showed potential to reduce unnecessary biopsies by 40%, 36%, 18%, and 11% while maintaining sensitivities at 90%, 95%, 99%, and 100%, respectively.

To ensure interpretability and enhance clinical relevance, XAI techniques like Grad-CAM were applied to generate heatmaps that visually identify the specific regions of mammography images that significantly influenced the model’s predictions. These heatmaps not only improve transparency but also provide radiologists with insights into the decision-making process of the model. The application of Grad-CAM in this pipeline bridges the gap between black-box DL models and practical clinical implementation, enabling clinicians to trust and validate the model’s predictions. Additionally, these visual explanations can aid in highlighting imaging patterns relevant to distinguishing benign tumors from malignant ones, further reinforcing the usability of proposed approach in clinical settings.

![](images/3ae624441a5e26350be3eb26a1d20c7cc290fe9480415ce61fac712d7c2b50a1.jpg)  
Fig. 4. An ROC curve of the image-only, text-only, and multimodal models that represent the ratio between true positive rate and false positive rate.

Although proposed multimodal approach produces superior performance in terms of disease detection, resulting in reducing unnecessary biopsies, there are a few limitations as well. First, the data is limited and obtained from a single institute, which could limit this approach for generalization on other datasets. Second, the dataset is collected locally and therefore, represents a single race; thus, the characteristics of the patients can be limited to a single race. Data from diferent institutes should be obtained to further validate proposed multimodal approach for better generalization. Third, radiologists in practice use diferent imaging modalities, such as breast ultrasound, contrast-enhanced mammography (CEM), computed tomography (CT), and magnetic resonance imaging (MRI) images, among others, for evaluation [16]. On the contrary, this approach only includes mammograms. Fourth, while Grad-CAM improves model transparency and facilitates clinical interpretability, other advanced XAI techniques could be explored to further elucidate the model’s reasoning. Lastly, pre-trained visionlanguage models were used; however, diferent attention-based mechanisms can be applied to extract rich contextual features from images and text, which can improve the overall performance.

In the future, it is planned to extend the dataset and add other modalities and multidimensional data to improve current model’s performance further. The integration of more advanced XAI techniques alongside Grad-CAM will also be investigated to provide a more comprehensive explanation of model predictions and enhance clinical adoption.

## 6 Conclusion

In conclusion, a novel SOTA multimodal vision-language model was developed to classify biopsy outcomes in patients with BI-RADS 4 and its subcategories. Leveraging the strengths of multimodal learning in CAD, digital mammograms processed using ViT and radiological reports encoded with DeBERTa-V3-Large were combined. Features extracted were fused for classification. Proposed model outperformed unimodal baselines, achieving an AUC of 0.72, accuracy of 87%, and precision of 94%. At sensitivities of 90%, 95%, 99%, and 100%, it yielded specificities of 40%, 36%, 18%, and 11%, respectively, indicating up to 40% reduction in false positive biopsies. Grad-CAM visualizations enhanced interpretability, aiding clinical decision-making. This initial multimodal approach demonstrates strong potential to reduce unnecessary biopsies, ofering clinically meaningful benefits in diagnostic accuracy, patient reassurance, and cost reduction.

## 7 Acknowledgment

This research was supported by the Secretaría de Ciencia, Humanidades, Tecnología e Innovación (Secihti) and Tecnologico de Monterrey.

## References

1. Arya, N., Saha, S.: Multi-modal advanced deep learning architectures for breast cancer survival prediction. Knowledge-Based Systems 221, 106965 (2021)

2. Brentnall, A.R., Cuzick, J., Buist, D.S., Bowles, E.J.A.: Long-term accuracy of breast cancer risk assessment combining classic risk factors and breast density. JAMA oncology 4(9), e180174–e180174 (2018)

3. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: International Conference on Learning Representations (2021), https://openreview.net/forum?id=YicbFdNTTy

4. Ezeana, C.F., He, T., Patel, T.A., Kaklamani, V., Elmi, M., Brigmon, E., Otto, P.M., Kist, K.A., Speck, H., Wang, L., et al.: A deep learning decision support tool to improve risk stratification and reduce unnecessary biopsies in bi-rads 4 mammograms. Radiology: Artificial Intelligence 5(6), e220259 (2023)

5. He, P., Gao, J., Chen, W.: DeBERTav3: Improving deBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing. In: The Eleventh International Conference on Learning Representations (2023), https://openreview.net/forum?id=sE7-XhLxHA

6. He, P., Liu, X., Gao, J., Chen, W.: {DEBERTA}: {DECODING}-{enhanced} {bert} {with} {disentangled} {attention}. In: International Conference on Learning Representations (2021), https://openreview.net/forum?id=XPZIaotutsD

7. Lacquement, M.A., Mitchell, D., Hollingsworth, A.B.: Positive predictive value of the breast imaging reporting and data system. Journal of the American College of Surgeons 189(1), 34–40 (1999)

8. LeCun, Y., Bengio, Y., Hinton, G.: Deep learning. nature 521(7553), 436–444 (2015)

9. Liu, C., Sun, M., Arefan, D., Zuley, M., Sumkin, J., Wu, S.: Deep learning of mammogram images to reduce unnecessary breast biopsies: a preliminary study. Breast Cancer Research 26(1), 82 (2024)

10. Lotter, W., Diab, A.R., Haslam, B., Kim, J.G., Grisot, G., Wu, E., Wu, K., Onieva, J.O., Boyer, Y., Boxerman, J.L., et al.: Robust breast cancer detection in mammography and digital breast tomosynthesis using an annotation-eficient deep learning approach. Nature medicine 27(2), 244–249 (2021)

11. Massafra, R., Latorre, A., Fanizzi, A., Bellotti, R., Didonna, V., Giotta, F., La Forgia, D., Nardone, A., Pastena, M., Ressa, C.M., et al.: A clinical decision support system for predicting invasive breast cancer recurrence: Preliminary results. Frontiers in Oncology 11, 576007 (2021)

12. Meng, M., Li, H., Zhang, M., He, G., Wang, L., Shen, D.: Reducing the number of unnecessary biopsies for mammographic bi-rads 4 lesions through a deep transfer learning method. BMC Medical Imaging 23(1), 82 (2023)

13. Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D.: Gradcam: Visual explanations from deep networks via gradient-based localization. In: Proceedings of the IEEE international conference on computer vision. pp. 618–626 (2017)

14. Shen, D., Wu, G., Suk, H.I.: Deep learning in medical image analysis. Annual review of biomedical engineering 19(1), 221–248 (2017)

15. Siegel, R.L., Miller, K.D., Wagle, N.S., Jemal, A.: Cancer statistics, 2023. CA: a cancer journal for clinicians 73(1) (2023)

16. Steyerova, P., Burgetova, A.: Current imaging techniques and impact on diagnosis and survival—a narrative review. Annals of Breast Surgery 6 (2022)

17. Wang, D., Khosla, A., Gargeya, R., Irshad, H., Beck, A.H.: Deep learning for identifying metastatic breast cancer. arXiv preprint arXiv:1606.05718 (2016)

18. Wu, N., Huang, Z., Shen, Y., Park, J., Phang, J., Makino, T., Gene Kim, S., Cho, K., Heacock, L., Moy, L., et al.: Reducing false-positive biopsies using deep neural networks that utilize both local and global image context of screening mammograms. Journal of Digital Imaging 34, 1414–1423 (2021)

19. Yala, A., Mikhael, P.G., Strand, F., Lin, G., Smith, K., Wan, Y.L., Lamb, L., Hughes, K., Lehman, C., Barzilay, R.: Toward robust mammography-based models for breast cancer risk. Science Translational Medicine 13(578), eaba4373 (2021)

20. Yan, B., Pei, M.: Clinical-bert: Vision-language pre-training for radiograph diagnosis and reports generation. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 36, pp. 2982–2990 (2022)

21. Zhao, C., Xiao, M., Liu, H., Wang, M., Wang, H., Zhang, J., Jiang, Y., Zhu, Q.: Reducing the number of unnecessary biopsies of us-bi-rads 4a lesions through a deep learning method for residents-in-training: a cross-sectional study. BMJ open 10(6), e035757 (2020)