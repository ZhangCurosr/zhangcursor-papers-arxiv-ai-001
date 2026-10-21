Carl von Ossietzky

Universität Oldenburg

Fach-Bachelor Informatik

BACHELORARBEIT

ECG for laboratory values:

# A comprehensive physiological and technical analysis

vorgelegt von

Walentin Palanchuk

Betreuender Gutachter

Prof. Dr. Nils Strodthoff

Zweiter Gutachter

Dr. Juan Miguel Lopez Alcaraz

Oldenburg, 31.08.2026

## Abstract

Laboratory measurements provide important information in clinical decision making but typically require invasive methods, such as blood sampling. Recent research suggests that ECG recordings may contain information beyond cardiac rhythm analysis and may provide information about laboratory values. This thesis investigates the prediction of N-Terminal Pro-B-Type Natriuretic Peptide (NT-proBNP) from ECG recordings and patient metadata using a multimodal deep learning framework, while considering the physiological plausibility of ECG-based NT-proBNP prediction. Furthermore, this work investigates the influence of a 1-lead configuration using lead II compared to the corresponding 12-lead ECG configuration on predictive performance. Additional experiments on glucose, hemoglobin and potassium are presented in the appendix.

Regression, binary classification and three-class classification experiments were conducted using the MIMIC-IV database. The three-class classification is based on clinically established thresholds. This work showed that the 12-lead configurations achieved consistently better predictive performance. The regression experiments showed a mean absolute error of $5 3 4 . 9 8 ~ \mathrm { p g / m L }$ for the 1-lead configuration and $5 1 4 . 2 7 ~ \mathrm { p g / m L }$ for the 12-lead configuration. The classification experiment achieved an AUROC of 0.849 for the 1-lead binary classification and 0.876 for the 12-lead configuration. The three-class classification achieved macro-averaged AUROC values of 0.738 and 0.752 for the 1-lead and 12-lead configurations respectively. The regression experiments showed that the precise prediction of NT-proBNP remains challenging, while the classification experiments showed that broader concentration ranges may be classified.

The findings are consistent with the physiological hypothesis that ECG characteristics and NT-proBNP concentrations may reflect shared underlying cardiovascular processes. Consequently, the ECG-based NT-proBNP prediction appears to be feasible as a complementary source of information, particularly when classification of broader concentration ranges is needed. However, laboratory measurements remain necessary and further research using independent datasets is required.

## Contents

1 Introduction 5   
1.1 Clinical Need 5   
1.2 Machine Learning in Healthcare 5   
1.3 Research Gap 6   
1.4 Contributions 6   
2 Background and Theoretical Foundations 8   
2.1 Electrocardiography 8   
2.1.1 ECG Waveform Components 9   
2.1.2 ECG Lead Configuration 9   
2.2 NT-proBNP 10   
2.3 ECG Characteristics and NT-proBNP . 11   
2.4 Deep Learning . 11   
2.4.1 Structured State Space Models 12   
2.4.2 RealMLP 13   
2.4.3 Multimodal Deep Learning . 13   
3 Related Work 14   
3.1 ECG-Based Disease Detection 14   
3.2 ECG-Based Laboratory Value Prediction 14   
4 Methodology 16   
4.1 Problem Formulation 16   
4.2 Datasets 17   
4.3 Data Preprocessing 18   
4.3.1 Laboratory Value and ECG Matching . 18   
4.3.2 Feature Processing 20   
4.4 Model Architecture 22   
4.5 Training Procedure 23   
4.6 Evaluation Metrics 23   
4.6.1 Regression Evaluation 24   
4.6.2 Classification Evaluation 25   
4.7 Additional Laboratory Value Experiments 26   
5 Experiments and Results 27   
5.1 Data Description 27   
5.2 NT-proBNP Regression 28   
5.3 NT-proBNP Classification 31   
5.3.1 Binary Classification 31   
5.3.2 Three-Class Classification 33   
5.4 Additional Laboratory Value Results 35   
5.5 Summary 35   
6 Discussion 36   
6.1 Physiological Interpretation 36   
6.2 Technical Interpretation 38   
6.3 Comparison with Previous Work . . 40   
6.4 Clinical Relevance . 41   
6.5 Generalization to Other Laboratory Values . 42   
6.6 Limitations 43   
6.7 Future Work . 44   
7 Conclusion 46   
A Additional NT-proBNP experiments 48   
A.1 Experimental Setup . 48   
A.2 Results 49   
A.2.1 NYHA Inspired Four-Class Classification 49   
A.2.2 Regression Configuration Experiments 51   
A.3 Interpretation 54   
B Hemoglobin 55   
B.1 Experimental Setup . 55   
B.2 Results . 55   
B.3 Interpretation . 58   
C Glucose 59   
C.1 Experimental Setup . 59   
C.2 Results . . . 59   
C.2.1 Regression . 60   
C.2.2 Classification 61   
C.3 Interpretation . 62   
D Potassium 64   
D.1 Experimental Setup . . 64   
D.2 Results . 64   
D.2.1 Regression 64   
D.2.2 Classification 66   
D.3 Interpretation 66

## 1 Introduction

This chapter introduces the motivation of ECG-based laboratory value prediction and the clinical need for noninvasive approaches to obtain information related to laboratory values. Furthermore, this chapter discusses the role of machine learning in this context, the existing research and the contributions of this work.

## 1.1 Clinical Need

Laboratory values have a crucial role in modern healthcare and are indispensable for clinical decision making. They provide information on the health status of the patient and are routinely used for disease diagnosis, treatment planning and monitoring the effectiveness of therapies[1]. Biomarkers that are measured through laboratory testing, such as NT-proBNP, hemoglobin, potassium and glucose, support clinicians in evaluating a wide range of cardiovascular, metabolic and physiological conditions[2, 3].

Despite their clinical importance, many laboratory values require invasive methods, such as blood sampling. Although blood sampling is invasive, it remains one of the most widely used testing methods[4]. It requires trained personnel, laboratory infrastructure and processing time for the results to become available. This contributes to the costs associated with laboratory testing for both patients and healthcare providers. Furthermore, repeated blood sampling increases workload and may cause discomfort for the patients, especially when frequent monitoring is required[5]. Hence, an approach that can provide clinically relevant information faster and causes less discomfort for patients is in demand.

## 1.2 Machine Learning in Healthcare

Machine learning has become a substantial field in modern healthcare, demonstrating potential to support clinical decision making and improve patient care. Prior research has shown that it may support clinicians by providing fast and accurate insights about patient information and assisting clinical decision making[6]. Common applications of machine learning in healthcare include image analysis and analysis of biological and clinical data[7, 8].

ECG signals have become an important data source for machine learning. Recent research suggests that ECG contains information beyond cardiac rhythm analysis and may reflect broader physiological and pathological processes[9]. Since ECG signals can be obtained easily and without invasive procedures using wearable devices such as smartwatches, they represent a promising data source to be used with machine learning. Furthermore, multimodal machine learning is used by combining multiple data sources, such as physiological signals and patient metadata. Multimodal machine learning is therefore more suitable for use in the medical context, because it mimics the multimodal decision making of clinical experts[10]. These developments have motivated the application of multimodal deep learning for predicting laboratory values from ECG signals and patient metadata.

## 1.3 Research Gap

The prediction of laboratory values from electrocardiography has become an increasingly active area of research in recent years. Previous studies have shown that ECG signals contain information beyond traditional cardiac rhythm analysis[11, 12]. In addition, several studies have investigated the prediction of laboratory values using deep learning approaches[9, 13], demonstrating that clinically relevant information may be extracted from ECG signals.

Despite these results, existing research has focused primarily on the predictive performance of machine learning approaches, while the physiological plausibility of ECG-based laboratory value prediction and the influence of technical design choices on predictive performance have been less extensively studied.

These limitations motivate a more comprehensive investigation of ECG-based laboratory value prediction that considers both physiological plausibility and technical aspects in addition to predictive performance.

## 1.4 Contributions

This thesis investigates ECG-based laboratory value prediction with NT-proBNP as the primary laboratory biomarker. In addition to predictive performance, this work considers the physiological relationships and technical design choices that may

influence ECG-based prediction. The main contributions of this thesis are:

• Evaluation of NT-proBNP prediction using a multimodal deep learning approach combining ECG recordings and patient metadata.

• Comparison of 1-lead and 12-lead ECG configurations across regression and classification tasks.

• Physiological and technical analysis of ECG-based NT-proBNP prediction

• Additional experiments on hemoglobin, glucose, potassium and further NTproBNP configurations.

## 2 Background and Theoretical Foundations

This chapter aims to provide a theoretical background required for the investigation of ECG-based laboratory prediction. It introduces the physiological foundation of electrocardiography and NT-proBNP and briefly describes the deep learning concepts and model architectures used in this work.

## 2.1 Electrocardiography

The electrocardiogram is a noninvasive test that records the electrical activity of the heart throughout a cardiac cycle by using electrodes attached to the skin of the patient. It is commonly used to detect heart rhythm problems and is essential in medical care. Furthermore, recent studies suggest that ECG signals may contain information to help detect noncardiac diseases such as pancreatitis or strokes[11] showing that ECG may reflect broader physiological and pathological processes beyond traditional cardiac rhythm analysis. A typical clinical ECG consists of twelve leads, each providing a different perspective of the electrical activity of the heart.

An electrocardiogram is based on the indirect recording of the dipole vectors of depolarization and repolarization of cardiomyocytes, the muscle cells that make up the heart, over time. During the cardiac cycle, the heart contracts due to action potentials traveling through the chambers of the heart. During this, generally one part of the cardiac tissue is depolarized, while another is at resting potential. Consequently, a dipole is created, which causes current flow in surrounding body fluids, resulting in fluctuating electric fields that are detected by the electrodes on the patient skin[14].

The resulting ECG waveforms consist of several characteristic waves, segments and intervals that summarize electrical events during a cardiac cycle. The appearance of these waveform components depends on the position and orientation of the recording electrodes relative to the direction of cardiac electrical activity. Thus, different ECG leads provide complementary information about the heart and may vary in their ability to capture physiological and pathological changes[14].

## 2.1.1 ECG Waveform Components

A standard ECG waveform consists of several characteristic components that represent different electrical events during the cardiac cycle. The P wave represents atrial depolarization. The QRS complex represents the electrical depolarization of the ventricles, while the T wave represents the ventricular repolarization[15].

In addition to the individual waves, intervals and segments are used in the ECG to describe the timing of cardiac electrical activity[15]. The PR interval describes the time from the beginning of atrial depolarization to the beginning of ventricular depolarization, while the duration of the ventricular depolarization is described by the QRS duration. The QT interval represents the duration of ventricular depolarization and repolarization, while the RR interval reflects the time between two successive ventricular depolarizations[15].

Changes in the morphology, duration or amplitude of components in the ECG reflect alterations in cardiac electrical activity[16]. These changes may be associated with pathological or physiological processes[16]. Consequently, the ECG waveform contains information that may indicate changes in the cardiovascular state of a patient and may contain patterns that can be detected by machine learning models[9]

## 2.1.2 ECG Lead Configuration

An ECG lead shows the electrical activity of the heart from a specific spatial perspective[17]. Since cardiac electrical activity propagates through the heart in different directions, the waveform can vary depending on the corresponding ECG lead orientation[17]. Therefore, different ECG leads provide complementary views of the same cardiac electrical activity and certain waveform characteristics may be more apparent in some lead configurations compared to others[17].

A typical 12-lead ECG consists of six limb leads and six precordial leads. The limb leads include lead I, II, III and the augmented leads aVR, aVL and aVF. The precordial leads include V1 to V6 and are recorded from different positions on the chest[17]. These leads provide multiple angles of cardiac electrical activity and allow electrical changes in different regions of the heart to be captured[17].

Lead II records the electrical activity between the electrode on the right arm and the electrode on the left leg[17]. Due to its electrode orientation, it commonly provides clearly distinguishable P waves, QRS complexes and T waves[18] and is often used for rhythm monitoring and 1-lead ECG recordings[18].

## 2.2 NT-proBNP

N-Terminal Pro-B-Type Natriuretic Peptide is the main laboratory biomarker investigated in this thesis. It is part of the natriuretic peptide system of the heart, which regulates blood pressure and fluid balance in response to cardiovascular stress[14]. In response to increased myocardial wall stress, cardiac muscle cells produce the precursor molecule proBNP[19], which is then cleaved into the biologically active hormone BNP and the biologically inactive NT-proBNP[20].

NT-proBNP is widely used as a biomarker for clinical assessment of heart failure. Low concentrations help to exclude heart failure, whereas elevated concentrations may help with diagnosis and may provide information about the severity of the disease[21, 22]. However, elevated NT-proBNP concentrations are not specific to a single cardiac condition. Therefore, additional clinical information should be considered when interpreting NT-proBNP concentrations[23].

NT-proBNP concentrations may also be influenced by patient-specific and physiological factors. Age, renal function and cardiac rhythm have been associated with NT-proBNP concentrations[24]. Therefore, the actual NT-proBNP concentration may vary between patients depending on their individual clinical characteristics, which need to be considered when NT-proBNP measurements are clinically interpreted[25].

Clinical interpretation of NT-proBNP commonly involves concentration thresholds that assist in the assessment of suspected heart failure[26]. The National Institute for Health and Care Excellence uses thresholds of 400 pg/mL and 2000 pg/mL to distinguish levels of urgency for further assessment[26], while in acute settings, an NT-proBNP concentration threshold of 300 pg/mL is commonly used to rule out acute heart failure[27]. This illustrates that NT-proBNP is interpreted both as a precise measurement and with clinically relevant concentration ranges.

## 2.3 ECG Characteristics and NT-proBNP

Despite the fact that ECG recordings and NT-proBNP measurements represent different aspects of cardiac function, both can reflect changes associated with cardiovascular disease[28]. Furthermore, elevated NT-proBNP concentrations have been associated with ECG characteristics and abnormalities related to cardiac dysfunction[29]. This suggests that physiological processes associated with changes in NTproBNP concentration may also be reflected in the ECG recording,

Recent studies have further investigated the relationship between ECG recordings and NT-proBNP concentrations using machine learning, more specifically deep learning. Indolfi et al. developed an AI-based deep learning approach using 12- lead ECG recordings to estimate elevated NT-proBNP concentrations and reported good predictive performance across multiple NT-proBNP thresholds[30]. This further supports the hypothesis that ECG recordings contain information associated with NT-proBNP concentration.

## 2.4 Deep Learning

The clinical interpretation of electrocardiograms requires specialized knowledge and is time intensive, especially when large numbers of recordings must be evaluated[31]. Due to this, automated interpretation algorithms have been integrated into ECG machines to support clinicians. However, their diagnostic performance remains limited for clinical use, due to the inaccuracy of these algorithms[32, 33]. Relying on automatic ECG analysis may lead to serious consequences such as delayed treatment[34], which has motivated considerable research into artificial intelligence, and more specifically deep learning, to support automated electrocardiogram interpretation[35].

Deep learning is a subfield of machine learning that uses multiple processing layers to learn representations of data and is inspired by the way the brain processes data[36]. The individual processing layers transform the input data into representations at various abstraction levels, hence deep learning models automatically identify patterns and features which are relevant to a specific task. This reduces the need to define relevant features manually before training the model[37].

In supervised deep learning, which is used in this work, a model is trained using input data together with corresponding target values. The difference between the predicted and actual output is quantified by a loss function. Through backpropagation, this error is propagated through the network and used iteratively to adjust the parameters[37].

## 2.4.1 Structured State Space Models

Structured State Space Sequence models (S4) are neural network architectures used for processing sequential data and are based on the concept of state space models, which represent a sequence through an internal state that changes when new information is received[38]. This allows information from previous parts of the sequence to influence the processing of later parts and therefore enables temporal dependencies within sequential data to be modeled.

This is particularly relevant for time series data, such as ECG signals, where individual samples are temporally related and patterns may extend across multiple time steps. S4 can model dependencies between different parts of an input sequence[38]. This allows for the model to learn representations that incorporate temporal information from different parts of the ECG input[39].

A challenge of sequence models is the efficient processing of long input sequences. S4 was developed to address this challenge by using structured parameterization of the state space model for sequence operations to be computed more efficiently[38]. S4 can then represent the state space operation as a convolution over the input sequence, which enables efficient parallel processing during training[38].

Furthermore, recent work has investigated the performance of different deep learning architectures for ECG analysis. Al-Masud et al. compared multiple ECG models across a broad range of clinical prediction tasks and reported considerable differences between model architectures[40]. This further supports the use of structured statespace approaches for ECG-based prediction tasks.

## 2.4.2 RealMLP

Another neural network architecture is RealMLP, which is used for tabular data. Tabular data consists of structured data that are typically organized in rows and columns, where each column represents a specific feature. In clinical datasets, these data include demographic information, physiological measurements and other patient information.

RealMLP is an improved multilayer perceptron developed for regression and classification tasks on tabular data and processes input features through multiple connected layers that can be used for prediction like standard multilayer perceptrons[41]. The difference from regular multilayer perceptrons is that RealMLP is optimized for tabular data through specialized preprocessing, architecture and optimizations[41].

## 2.4.3 Multimodal Deep Learning

In this work, multimodal deep learning was used to combine information from multiple modalities and potentially improve predictive results. This type of deep learning is commonly used in clinical machine learning because clinical data are often heterogeneous[42]. Furthermore, the use of multiple different modalities may support the model by providing complementary information which would otherwise not be available from a single modality alone[43].

Since the modalities differ, separate encoders are commonly utilized. Encoders convert raw input data into usable representations for the following prediction model[44]. Subsequently, the encoded representations need to be combined using a fusion strategy. Fusion strategies are classified as early fusion, intermediate fusion or late fusion.

In early fusion, modalities are combined at or near the input level and then passed to the prediction model. In intermediate fusion, both modalities are processed into separate representations, combined within the model, and then used for prediction. In late fusion, each modality is processed by independent sub-models into separate predictions which are then combined[44].

## 3 Related Work

Previous research has investigated the use of ECG recordings for various deep learning applications. The following sections review the ECG-based disease detection and prediction of laboratory values from ECG recordings.

## 3.1 ECG-Based Disease Detection

Deep learning has been used for various ECG-based disease detection and classification tasks[45]. A major area of early research focused primarily on automated identification of arrhythmia with positive results[45]. A deep neural network trained on 1-lead ECG signals achieved performance results similar to those of certified cardiologists across multiple rhythm classes[46], showing that clinically relevant rhythm patterns may be learned directly from ECG signals.

Deep learning architectures have also been evaluated on large publicly available 12-lead ECG datasets, such as PTB-XL[47]. Experiments on the PTB-XL dataset compared time series classification algorithms for different ECG prediction tasks. These experiments showed that convolutional neural networks had the strongest performance throughout the investigated tasks[47].

Beyond rhythm and morphology classification, ECG-based deep learning has also been used for noncardiac disease detection. A multimodal approach combining ECG and photoplethysmography signals was used to distinguish stroke patients from healthy elderly patients[48]. Studies such as this indicated that ECG-derived information may be used for noncardiac diagnostic tasks, especially when combined with an additional modality[49].

## 3.2 ECG-Based Laboratory Value Prediction

Another area of ECG-based deep learning research investigates whether ECG recordings can be used to predict laboratory values and detect laboratory abnormalities. Serum potassium was among the laboratory values investigated in early ECG-based deep learning research. This is because changes in potassium concentrations can affect cardiac electrophysiology and ECG morphology[50, 51]. These effects may include changes in ECG waves and cardiac conduction, although their occurrence and severity vary between different patients[52].

Subsequently, a deep learning model was developed to detect hyperkalemia from ECG recordings of patients with chronic kidney disease. The model was trained on a large number of ECG recordings and achieved AUC values ranging from 0.853 to 0.883[13]. These results indicate that ECG-based deep learning may support noninvasive screening for elevated potassium levels[13].

Recent research has expanded ECG-based prediction to a broad range of laboratory abnormalities[9]. A multimodal deep learning approach combined ECG recordings with patient metadata to predict laboratory abnormalities using the MIMIC-IV dataset. Among the investigated biomarkers, NT-proBNP showed the strongest predictive performance, having an AUROC of 0.903[9]. Comparatively strong results were also reported for other abnormalities in hemoglobin and albumin[9].

These studies demonstrate that ECG signals may contain information related to a range of laboratory values. However, predictive performance varies between individual biomarkers[9]. The positive results for NT-proBNP motivate a more detailed investigation. Although previous work has demonstrated the feasibility of ECG-based laboratory prediction, physiological interpretation and the influence of different technical design choices, such as ECG lead configurations, have received less attention in previous research. Therefore, this work extends previous research by investigating NT-proBNP prediction using regression, binary classification and three-class classification experiments and the effect of 1-lead and 12-lead configurations.

## 4 Methodology

The methodology used in this work was designed to evaluate ECG-based NTproBNP prediction across different prediction tasks and ECG lead configurations. The following sections describe the methodology used for the experiments.

## 4.1 Problem Formulation

This work investigates whether NT-proBNP concentrations can be predicted from electrocardiogram recordings and patient metadata using a multimodal deep learning framework. The main experiments include regression and classification tasks and investigate the influence of using 1-lead and 12-lead settings for both tasks on predictive performance. The classification experiments include classification using a threshold of $\mathrm { 3 0 0 ~ p g / m L }$ and a three-class task based on the National Institute for Health and Care Excellence (NICE) guidelines[26].

The regression task aims to estimate the continuous NT-proBNP concentration corresponding to each ECG recording. Regression performance is evaluated using the mean absolute error, Bland-Altman analysis and Passing-Bablok regression.

For the binary classification task, the dataset is divided into two classes. The first class contains NT-proBNP concentrations below $\mathrm { 3 0 0 ~ p g / m L }$ , whereas the second class contains NT-proBNP concentrations greater than or equal to $\mathrm { 3 0 0 ~ p g / m L }$

Furthermore, a three-class classification task is defined according to the NICE thresholds[26]. The three classes include $\mathrm { N T - p r o B N P }$ concentrations below $4 0 0 \mathrm { p g / m L }$ concentrations from 400 to $2 0 0 0 \mathrm { p g / m L }$ inclusive, and concentrations above $2 0 0 0 \mathrm { p g / m L }$ Classification performance is evaluated using the area under the receiver operating characteristic curve (AUROC), confusion matrices and additional classification metrics.

The patient metadata is incorporated alongside the ECG recordings as an additional input modality.

The primary analysis evaluates the regression and classification predictive performance across 1-lead (lead II) and 12-lead ECG configurations. Additional experiments involving hemoglobin, blood glucose, potassium and further NT-proBNP prediction settings are presented in the appendix.

Table 1: Overview of the primary NT-proBNP experiments.
<table><tr><td>Task</td><td>Thresholds [pg/mL] Leads</td><td></td><td>Loss</td></tr><tr><td>Regression</td><td></td><td>1-lead</td><td>L1</td></tr><tr><td>Regression</td><td></td><td>12-lead</td><td>L1</td></tr><tr><td>Binary classification</td><td>300</td><td>1-lead</td><td>Weighted focal CE</td></tr><tr><td>Binary classification</td><td>300</td><td>12-lead</td><td>Weighted focal CE</td></tr><tr><td>Three-class classification</td><td>400, 2000</td><td>1-lead</td><td>Weighted focal CE</td></tr><tr><td>Three-class classification</td><td>400, 2000</td><td>12-lead</td><td>Weighted focal CE</td></tr></table>

Table 1 shows an overview of the primary NT-proBNP experiments and the corresponding prediction tasks, classification thresholds, ECG lead configurations and loss functions.

## 4.2 Datasets

The data used in this work are obtained from the Medical Information Mart for Intensive Care IV (MIMIC-IV) database and its associated ECG and emergency department datasets. All datasets are obtained through PhysioNet and contain deidentified patient data collected at Beth Israel Deaconess Medical Center in Boston, Massachusetts[53–55].

MIMIC-IV version 3.1 provides the clinical data in this work, including laboratory measurements, patient information and online medical record measurements[53–55]. NT-proBNP measurements and the additional laboratory values are obtained from the labevents table. Patient age and sex are extracted from the patients table, while height, weight and body mass index are obtained from the omr table.

The ECG waveforms are obtained from the MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset version 1.0[55, 56]. The original recordings have a duration of 10 seconds. In this work, segments with a duration of 2.5 seconds are used as model input. ECG records are linked to patients in the MIMIC-IV clinical database, which enables the matching of ECG recordings with laboratory measurements and

patient information.

In addition to the waveform data, ECG machine measurements are obtained from MIMIC-IV-ECG. These machine measurements include the RR interval, the onset and end of the P wave and QRS complex, the end of the T wave and the electrical axes of the P wave, QRS complex and T wave. These machine measurements obtained from MIMIC-IV-ECG are calculated across all 12 leads[55, 56].

Vital signs are obtained from MIMIC-IV-ED version 2.2, the database of emergency department admissions at the Beth Israel Deaconess Medical Center between 2011 and 2019[55, 57]. MIMIC-IV-ED contains data from emergency department stays and is also linked to patients in the MIMIC-IV clinical database. In this work, the used vital signs include patient temperature, heart rate, respiratory rate, oxygen saturation and systolic and diastolic blood pressure.

## 4.3 Data Preprocessing

The data preprocessing pipeline used in this work combines ECG recordings, laboratory measurements, patient information, ECG machine measurements, online medical record data, and vital signs into a unified dataset for each experiment. Individual data sources are linked using patient and recording identifiers according to the MIMIC-IV documentation[58]. Laboratory measurements and ECG recordings are matched based on their temporal proximity. Each resulting sample consists of an ECG recording, the matching target value or classification label, and a feature vector. The dataset is split at the patient level into 90% training, 5% validation and 5% test data to prevent data leakage.

## 4.3.1 Laboratory Value and ECG Matching

NT-proBNP measurements are extracted from the labevents table using the matching item identifier. For laboratory values with multiple item identifiers, one consistent identifier is selected. This was necessary to avoid combining different measurements that may differ in their laboratory procedures or specimen types, such as blood, urine, or other body fluids

Measurements with missing timestamps, missing, zero or negative numerical target values or missing units are excluded. Measurements without timestamps cannot be accurately matched to an ECG recording, while missing, zero or negative target values do not provide a valid target for either prediction task. Missing target values are not imputed to avoid distorting the target distribution by introducing artificial labels. Furthermore, for the regression experiments, NT-proBNP measurements above 3000 pg/mL are excluded to limit the target range. The remaining regression target values are then standardized using a StandardScaler fitted on the training split.

ECG recordings and laboratory measurements from the same patient are matched using a temporal window of ±3 hours, as shown in Figure 1. This window was chosen to retain a sufficient number of samples while still allowing the most appropriate laboratory measurement to be matched to each ECG recording. For each ECG recording, the laboratory measurement with the smallest absolute temporal distance is selected. If multiple laboratory measurements with the same temporal distance are present, the earlier laboratory timestamp and subsequently the laboratory identifier are used to resolve ties. Duplicate assignments of the same laboratory measurement are removed so individual laboratory values are only represented once in the dataset.

![](images/cae906215f5056394b6230d54e136f756da57e7de6b222a14545a39a01f12b17.jpg)  
Figure 1: Schematic overview of the temporal matching of ECG and laboratory values.

In addition, machine measurements are included in the feature vector. These measurements are described in Section 4.2. Temporal information is represented by the time difference between the ECG recording and the associated laboratory measurement in minutes and hours. The time of day of the ECG recording is additionally encoded using sine and cosine transformation to preserve its cyclic structure.

## 4.3.2 Feature Processing

Patient age is calculated at the time of the ECG recordings using the anchor age and anchor year of the patient, while the patient sex is extracted and encoded as a binary variable.

Patient height, weight and body mass index (BMI) are obtained from the omr table, while the vital signs are obtained from the vitalsign table. Both data sources are matched to the ECG recording using the same temporal matching procedure as for laboratory measurements. For vital signs, a temporal window of ±1 hour is selected so that the patient condition at the time of the ECG recording is reflected. For body measurements, a temporal window of ±183 days is selected since these measurements are recorded less frequently than laboratory values and vital signs.

Missing values for BMI, height, weight and vital signs are replaced with the corresponding median calculated exclusively from the training split. Median imputation is used to retain more samples with incomplete metadata. Calculating the median only from the training split prevents the validation and test splits from influencing preprocessing. Furthermore, binary missing flags are included so that the model can distinguish between original values and imputed values.

Table 2: Tabular input features
<table><tr><td>Feature</td><td>Source</td><td>Processing</td></tr><tr><td>Age</td><td>MIMIC-IV</td><td>Calculated at ECG time</td></tr><tr><td>Sex</td><td>MIMIC-IV</td><td>Binary encoding</td></tr><tr><td>RR interval</td><td>MIMIC-IV-ECG</td><td>Machine measurement</td></tr><tr><td>P wave onset</td><td>MIMIC-IV-ECG</td><td>Machine measurement</td></tr><tr><td>P wave end</td><td>MIMIC-IV-ECG</td><td>Machine measurement</td></tr><tr><td>QRS onset</td><td>MIMIC-IV-ECG</td><td>Machine measurement</td></tr><tr><td>QRS end</td><td>MIMIC-IV-ECG</td><td>Machine measurement</td></tr><tr><td>T wave end</td><td>MIMIC-IV-ECG</td><td>Machine measurement</td></tr><tr><td>P wave axis</td><td>MIMIC-IV-ECG</td><td>Machine measurement</td></tr><tr><td>QRS axis</td><td>MIMIC-IV-ECG</td><td>Machine measurement</td></tr><tr><td>T wave axis</td><td>MIMIC-IV-ECG</td><td>Machine measurement</td></tr><tr><td>BMI</td><td>MIMIC-IV</td><td>±183 day matching</td></tr><tr><td>Height</td><td>MIMIC-IV</td><td>±183 day matching</td></tr><tr><td>Weight</td><td>MIMIC-IV</td><td>±183 day matching</td></tr><tr><td>Time difference [min]</td><td>Derived</td><td>ECG-laboratory difference</td></tr><tr><td>Time difference [h]</td><td>Derived</td><td>ECG-laboratory difference</td></tr><tr><td>ECG time sine</td><td>Derived</td><td>Cyclic encoding</td></tr><tr><td>ECG time cosine</td><td>Derived</td><td>Cyclic encoding</td></tr><tr><td>BMI missing</td><td>Derived</td><td>Binary missing indicator</td></tr><tr><td>Height missing</td><td>Derived</td><td>Binary missing indicator</td></tr><tr><td>Weight missing</td><td>Derived</td><td>Binary missing indicator</td></tr><tr><td>Temperature</td><td>MIMIC-IV-ED</td><td>±1 hour matching</td></tr><tr><td>Heart rate</td><td>MIMIC-IV-ED</td><td>±1 hour matching</td></tr><tr><td>Respiratory rate</td><td>MIMIC-IV-ED</td><td>±1 hour matching</td></tr><tr><td>Oxygen saturation</td><td>MIMIC-IV-ED</td><td>±1 hour matching</td></tr><tr><td>Systolic blood pressure</td><td>MIMIC-IV-ED</td><td>±1 hour matching</td></tr><tr><td>Diastolic blood pressure</td><td>MIMIC-IV-ED</td><td>±1 hour matching</td></tr><tr><td>Temperature missing</td><td>Derived</td><td>Binary missing indicator</td></tr><tr><td>Heart rate missing</td><td>Derived</td><td>Binary missing indicator</td></tr><tr><td>Respiratory rate missing</td><td>Derived</td><td>Binary missing indicator</td></tr><tr><td>Oxygen saturation missing</td><td>Derived</td><td>Binary missing indicator</td></tr><tr><td>Systolic BP missing</td><td>Derived</td><td>Binary missing indicator</td></tr><tr><td>Diastolic BP missing</td><td>Derived</td><td>Binary missing indicator</td></tr></table>

Table 2 summarizes the 33 features used in the experiments.

## 4.4 Model Architecture

The model architecture used in this work is based on the MDS-ICU framework developed by López Alcaraz et al. [59]. The framework was provided for this thesis and adapted for the prediction of different laboratory values. It consists of separate encoder branches for ECG waveforms and tabular features. The ECG branch consists of four stacked structured state space (S4) blocks followed by a pooling head while the tabular branch uses a RealMLP encoder. The representations generated by both branches are then combined using the late fusion described in MDS-ICU[59].

In this work, the ECG recordings were resampled to a sampling frequency of 100 Hz and divided into segments of 2.5 seconds, resulting in 250 samples per lead. The 1-lead configuration uses lead II as the waveform input, while the 12-lead configuration uses all available ECG leads. The tabular branch receives the 33 features derived during data preprocessing. These features include demographic information, machine-generated ECG measurements, body measurements, vital signs, temporal features and missing flags for the imputed body measurements and vital signs. The underlying encoder and fusion architecture remain unchanged across these experiments.

![](images/e2b6e1faf453c9231801856d3149f94b2fc8e070de283079f18e1320535e246d.jpg)  
Figure 2: Overview of the pipeline for the laboratory value prediction.

Figure 2 shows an overview of the complete pipeline from the datasets to laboratory value prediction.

## 4.5 Training Procedure

The models in this work are trained from scratch. The general optimization settings remain unchanged for the regression and classification experiments and the 1-lead and 12-lead configurations. The complete training configuration is shown in Table 3.

L1 loss is used to optimize the regression models based on the difference between the predicted and standardized target values. Parameters used for target standardization are calculated only from the training split. For evaluation, the predictions are transformed back into the original unit of measurement of pg/mL.

For the classification experiments, focal cross entropy loss is used to reduce the contribution of easily classified samples and focus more on difficult examples. Furthermore, class weights are added to account for the imbalance in the training data.

Table 3: Training configuration for the primary NT-proBNP experiments.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Epochs</td><td>20</td></tr><tr><td>Batch size</td><td>32</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td>0.001</td></tr><tr><td>Weight decay</td><td>0.001</td></tr><tr><td>Precision</td><td>32-bit</td></tr><tr><td>Regression loss</td><td>L1 loss</td></tr><tr><td>Classification loss</td><td>Focal cross entropy loss</td></tr><tr><td>Focusing parameter</td><td>2</td></tr><tr><td>Binary class weights</td><td>2.2414, 0.6436</td></tr><tr><td>Three-class weights</td><td>1.2811, 1.1668, 0.7340</td></tr></table>

## 4.6 Evaluation Metrics

Different evaluation metrics are used depending on the prediction task. Regression performance is evaluated using the mean absolute error, Bland-Altman analysis and Passing-Bablok regression. Classification performance is evaluated using the area under the receiver operating characteristic curve (AUROC), accuracy, sensitivity, F1 score and confusion matrices. In addition, 95% confidence intervals are used to indicate the precision of the reported performance estimates. Following the evaluation procedure used in the MDS-ICU framework[59], confidence intervals are estimated using 1000 bootstrap iterations.

## 4.6.1 Regression Evaluation

Mean absolute error (MAE) measures the average absolute difference between the predicted and true NT-proBNP concentrations and is used in this work as the primary performance metric for the regression experiments. It is defined as

$$
\mathrm { M A E } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left| y _ { i } - \hat { y } _ { i } \right| ,\tag{1}
$$

where N denotes the number of samples, $y _ { i }$ the true NT-proBNP concentration, and $\hat { y } _ { i }$ the corresponding model prediction. Lower MAE values indicate smaller prediction errors. Since MAE is reported in the same unit as the target, it allows the prediction error to be interpreted directly in the original NT-proBNP scale.

Due to the standardization of the regression targets, the MAE reported by the training framework is initially calculated on the standardized targets. For the final evaluation, the predictions are transformed back to the original scale, and the MAE is reported in pg/mL.

In addition to MAE, Bland-Altman analysis, using the statsmodels Python package [60], is used to analyze how the predicted NT-proBNP concentration compares to the matching target values. For each sample, the difference between the predicted concentration and reference value is plotted against the mean of both values. The mean difference indicates whether the model tends to overestimate or underestimate NT-proBNP concentrations.

The limits of agreement are calculated as

$$
\bar { d } \pm 1 . 9 6 s _ { d } ,\tag{2}
$$

where $\bar { d }$ denotes the mean difference between predicted and reference values and $s _ { d }$ is the standard deviation of these differences. The limits of agreement show the range in which approximately 95% of the differences between predicted and reference values are expected to lie. Bland-Altman analysis complements MAE by providing additional information about the deviation and spread of differences between the predicted NT-proBNP concentration and the reference concentration.

For further evaluation of the relationship between predicted and reference NTproBNP concentrations, Passing-Bablok regression is used. This method is used to estimate a regression line between both measurements and show systematic and proportional differences and therefore complements the previous analyses. For the analysis, the mcr package in R is used with the bootstrap quantile method and 1000 bootstrap samples. Since the Passing-Bablok implementation in mcr does not support negative values, the negative model predictions are excluded from this analysis.

## 4.6.2 Classification Evaluation

The primary evaluation metric for the classification experiments is the area under the receiver operating characteristic curve (AUROC). For the three-class classification task, macro-averaged AUROC is used. The receiver operating characteristic curve demonstrates how the true positive and false positive rates change across different classification thresholds, while AUROC summarizes this relationship into a single value. Higher AUROC values indicate better classification performance, with a value of 0.5 indicating no discriminative ability and a value of 1 indicating perfect discrimination. Additionally, accuracy, sensitivity and F1 score are reported and calculated using the scikit-learn library

Sensitivity and F1 score are calculated in the binary classification evaluation for the positive class of NT-proBNP concentrations greater than or equal to 300 pg/mL. Furthermore, the three-class classification evaluation uses macro-averaged sensitivity and F1 score to give every class equal weighting regardless of class size.

Furthermore, confusion matrices, using the scikit-learn library[61], are used to evaluate the distribution of correct and incorrect class predictions, where the predicted class labels are compared to the corresponding reference labels to show how frequently samples from each class are assigned to the different classes. This complements AUROC by showing the distribution of classification errors.

## 4.7 Additional Laboratory Value Experiments

In addition to the primary NT-proBNP experiments, additional experiments for hemoglobin, glucose and potassium were conducted. These experiments investigated whether the multimodal ECG-based prediction approach could also be applied to other laboratory values. Furthermore, alternative classification and regression configurations for NT-proBNP were investigated. The detailed experimental setups and results are presented in Appendix A, Appendix B, Appendix C and Appendix D.

## 5 Experiments and Results

This section presents the results of the regression and classification experiments for the prediction of NT-proBNP from ECG signals and patient metadata. The experiments are evaluated using the metrics described in the previous section. In addition, the predictive performance of 1-lead and 12-lead ECG configurations is compared.

## 5.1 Data Description

Table 4: Test split size for the primary NT-proBNP experiments.
<table><tr><td>Task</td><td>ECG recordings F</td><td>ECG segments</td></tr><tr><td>Regression</td><td>910</td><td>3640</td></tr><tr><td>Classification</td><td>1435</td><td>5740</td></tr></table>

The final datasets differed between the regression and classification experiments, since NT-proBNP concentrations above 3000 $\mathrm { p g / m L }$ were excluded from the regression dataset. Table 4 shows the number of ECG recordings and ECG segments in the test splits for each task. Each ECG recording was divided into four 2.5 second segments and all reported prediction metrics were calculated using the 2.5 second ECG segments.

Table 5: Test split class distribution for the NT-proBNP classification experiments.
<table><tr><td>Task</td><td>Class</td><td>ECG recordings</td><td>ECG segments</td></tr><tr><td rowspan="2">Binary</td><td> $< 3 0 0 ~ \mathrm { p g / m L }$ </td><td>307</td><td>1228</td></tr><tr><td> $\geq 3 0 0 ~ \mathrm { p g / m L }$ </td><td>1128</td><td>4512</td></tr><tr><td rowspan="3">Three-class</td><td> $< 4 0 0 ~ \mathrm { p g / m L }$ </td><td>381</td><td>1524</td></tr><tr><td> $4 0 0 { - } 2 0 0 0 ~ \mathrm { p g / m L }$ </td><td>410</td><td>1640</td></tr><tr><td> $> 2 0 0 0 ~ \mathrm { p g / m L }$ </td><td>644</td><td>2576</td></tr></table>

Table 5 shows the class distribution in the test splits for the binary and three-class classification experiments. The binary classification task showed a stronger class imbalance than the three-class classification task.

Table 6: Missing metadata counts in the test splits for the primary NT-proBNP experiments.
<table><tr><td>Feature</td><td>Regression n (%)</td><td>Classification n (%)</td></tr><tr><td>BMI</td><td>229 (25.2%)</td><td>418 (29.1%)</td></tr><tr><td>Height</td><td>367 (40.3%)</td><td>653 (45.5%)</td></tr><tr><td>Weight</td><td>187 (20.5%)</td><td>325 (22.6%)</td></tr><tr><td>Temperature</td><td>628 (69.0%)</td><td>1057 (73.7%)</td></tr><tr><td>Heart rate</td><td>470 (51.6%)</td><td>776 (54.1%)</td></tr><tr><td>Respiratory rate</td><td>472 (51.9%)</td><td>781 (54.4%)</td></tr><tr><td>Oxygen saturation</td><td>487 (53.5%)</td><td>806 (56.2%)</td></tr><tr><td>Systolic blood pressure</td><td>480 (52.7%)</td><td>787 (54.8%)</td></tr><tr><td>Diastolic blood pressure</td><td>480 (52.7%)</td><td>787 (54.8%)</td></tr></table>

Table 6 shows the number of missing metadata values in the test splits for both tasks. These missing values were imputed using the median calculated from the training splits.

## 5.2 NT-proBNP Regression

The following results focus on the regression experiments and compare the 1-lead and 12-lead ECG configurations regarding prediction error and agreement with the reference NT-proBNP concentrations.

Table 7: Mean absolute error with 95% confidence intervals for NT-proBNP prediction.
<table><tr><td colspan="3">ECG configuration MAE [pg/mL] 95% CI [pg/mL]</td></tr><tr><td>1-lead</td><td>534.98</td><td>501.16-571.62</td></tr><tr><td>12-lead</td><td>514.27</td><td>482.83-546.46</td></tr></table>

Table 7 shows the regression performance of the 1-lead and 12-lead experiments. The 1-lead model achieved an MAE of 534.98 pg/mL (95% CI: 501.16–571.62 pg/mL), while the 12-lead model achieved an MAE of 514.27 $\mathrm { p g / m L }$ (95% CI: 482.83- $5 4 6 . 4 6 ~ \mathrm { p g / m L } )$ . The MAE of the 12-lead configuration was therefore 20.71 $\mathrm { p g / m L }$ lower than that of the 1-lead configuration.

![](images/5d288b626e2a1918504c1d126677bbb7d0065ef7aab26625f1871c7255e3a428.jpg)  
(a) 1-lead ECG

Bland-Altman Plot - 12-lead  
![](images/722b58e4f5fc6dd6721f4325c2038c2d669318da5b5e2c0d0a483281faf0dd7d.jpg)  
(b) 12-lead ECG  
Figure 3: Bland-Altman analysis of predicted and reference NT-proBNP concentrations for the 1-lead and 12-lead configurations. The solid line represents the mean difference and the dashed lines indicate the 95% limits of agreement.

Figure 3 illustrates the Bland-Altman analysis for both ECG configurations. The 1-lead model showed a mean difference of $- 2 3 8 . 8 6 ~ \mathrm { p g / m L }$ with limits of agreement from -1646.41 to $1 1 6 8 . 6 9 ~ \mathrm { p g / m L }$ , while the 12-lead model showed a mean difference of -218.25 pg/mL with limits of agreement from -1583.4 to $1 1 4 6 . 9 ~ \mathrm { p g / m L }$ . In both configurations, the spread of the differences widened at higher NT-proBNP concentrations.

![](images/3ecf7b923d1983410c2637f06fab10d55c3e2d4691ea15bd1d752bd0648efa56.jpg)  
(a) 1-lead ECG

![](images/43e096bfc4fb1ea6ad7f3efa3c64146706b36769be1ae23927a77386f058d7f8.jpg)  
(b) 12-lead ECG  
Figure 4: Passing-Bablok regression of predicted and reference NT-proBNP concentrations for the 1-lead and 12-lead configurations. The solid line represents the Passing-Bablok regression fit, the dashed line represents the identity, and the shaded area represents the 95% confidence interval.  
Figure 4 shows the Passing-Bablok regression analysis for 1-lead and 12-lead ECG

configurations. The intercept for the 1-lead configuration was 144.20 pg/mL (95% CI: 132.35–153.81 pg/mL) and the slope was 0.55 (95% CI: 0.53–0.57), while for the 12-lead configuration the intercept was 151.45 pg/mL (95% CI: 136.64–167.10 pg/mL) and the slope was 0.58 (95% CI: 0.56–0.60).

## 5.3 NT-proBNP Classification

The following results show the predictive performance of the classification experiments, starting with the binary classification experiment followed by the three-class classification experiment.

## 5.3.1 Binary Classification

Table 8: Binary classification performance for NT-proBNP prediction.
<table><tr><td>ECG configuration</td><td>AUROC</td><td>95% CI</td><td>Accuracy</td><td>Sensitivity</td><td>F1 score</td></tr><tr><td>1-lead</td><td>0.849</td><td>0.839-0.860</td><td>0.770</td><td>0.784</td><td>0.843</td></tr><tr><td>12-lead</td><td>0.876</td><td>0.866-0.886</td><td>0.799</td><td>0.798</td><td>0.862</td></tr></table>

Table 8 summarizes the performance of the binary classification experiments. The 1-lead experiment achieved an AUROC of 0.849 (95% CI: 0.839–0.860), while the 12- lead experiment achieved an AUROC of 0.876 (95% CI: 0.866–0.886). The 12-lead configuration also showed higher values across all other metrics.

![](images/e869350ddb97504d403dcbf1281d0c2dd6b3f1add7100eba707ca4d316a13243.jpg)

(a) 1-lead ECG  
![](images/054c03c65d77542af167ca7a2e55917d4d1fa17c2ddedc99bd33821114a6086c.jpg)  
(b) 12-lead ECG  
Figure 5: Confusion matrices for the binary NT-proBNP classification task using the 1-lead and 12-lead ECG configurations.

Figure 5 shows the confusion matrices for both ECG configurations. The 1-lead configuration correctly classified 883 samples below $\mathrm { 3 0 0 ~ p g / m L }$ and 3536 samples greater than or equal to $3 0 0 ~ \mathrm { p g / m L }$ . The 12-lead configuration correctly classified 984 samples below $3 0 0 \mathrm { p g / m L }$ and 3600 samples greater than or equal to $3 0 0 \mathrm { p g / m L }$ The number of false positive classifications decreased from 345 to 244, while the false negative classifications decreased from 976 to 912.

## 5.3.2 Three-Class Classification

Table 9: Three-class classification performance for NT-proBNP prediction.
<table><tr><td>ECG configuration</td><td>AUROC</td><td>95% CI</td><td>Accuracy</td><td>Sensitivity</td><td>F1 score</td></tr><tr><td>1-lead</td><td>0.738</td><td>0.729-0.747</td><td>0.551</td><td>0.527</td><td>0.524</td></tr><tr><td>12-lead</td><td>0.752</td><td>0.743-0.761</td><td>0.578</td><td>0.565</td><td>0.539</td></tr></table>

Table 9 summarizes the performance of the three-class classification experiment based on the NICE guidelines. The 1-lead configuration achieved a macro-averaged AUROC of 0.738 (95% CI: 0.729–0.747) and the 12-lead configuration achieved a macro-averaged AUROC of 0.752 (95% CI: 0.743–0.761). The 12-lead configuration also achieved higher accuracy, macro-averaged sensitivity and macro-averaged F1 score.

![](images/bf31f599787621fbf642cdcd1ed5a26c0e5e91cd256cbc4df7e6a0900616d9bb.jpg)

(a) 1-lead ECG  
![](images/430e0f14d41ec622f17881e9305114908c7166eac77fecffe7b4ae66f6d8e077.jpg)  
(b) 12-lead ECG  
Figure 6: Confusion matrices for the three-class NT-proBNP classification task using the 1-lead and 12-lead ECG configurations.

Figure 6 illustrates the confusion matrices for the three-class NT-proBNP classification task, based on the NICE guidelines. The 1-lead configuration correctly classified 925 samples below $4 0 0 ~ \mathrm { p g / m L }$ , 486 between 400 and $2 0 0 0 ~ \mathrm { p g / m L }$ and 1749 samples above $2 0 0 0 ~ \mathrm { p g / m L }$ . The 12-lead configuration correctly classified 1202, 378 and 1738 respectively. The $4 0 0 { - } 2 0 0 0 ~ \mathrm { p g / m L }$ class showed the lowest proportion of correct classifications for both ECG configurations.

## 5.4 Additional Laboratory Value Results

The additional experiments showed varying predictive performance across the investigated laboratory values. Hemoglobin achieved mean absolute errors of approximately $1 . 6 0 ~ \mathrm { g / d L }$ for both ECG configurations. Glucose achieved a mean absolute error of 34.52 mg/dL and a macro-averaged AUROC of 0.632 for the three-class classification experiment. Potassium achieved a mean absolute error of 0.51 mEq/L for the regression experiment and a macro-averaged AUROC of 0.629 for the classification experiment.

The additional NT-proBNP experiments compared alternative classification and regression configurations. The four-class classification experiment showed that weighted cross entropy achieved the highest macro-averaged AUROC of 0.726, compared with 0.714 for unweighted cross entropy and 0.709 for weighted focal cross entropy. Standard scaling achieved a lower MAE than MinMax scaling in the regression experiments, with 2317.63 pg/mL compared with 2390.46 pg/mL.

Detailed results of these experiments are presented in Appendix A, Appendix B, Appendix C and Appendix D.

## 5.5 Summary

Across all evaluation tasks, the 12-lead ECG configuration consistently achieved better predictive performance than the 1-lead configuration, although the improvements remained moderate. The three-class classification task showed lower predictive performance compared to the binary classification task. The 400–2000 pg/mL class showed the lowest proportion of correctly classified samples for both ECG lead configurations.

## 6 Discussion

In this chapter, the experimental results are discussed from physiological, technical and clinical perspectives. Furthermore, the results are compared with previous research, the limitations of this work are discussed and possible future research directions are documented.

## 6.1 Physiological Interpretation

The results of this work suggest that information associated with NT-proBNP concentration may be extracted from ECG signals in combination with patient metadata. From a physiological point of view, this relationship is plausible because NT-proBNP is closely associated with cardiac stress and dysfunction[19]. However, NT-proBNP should not be viewed as directly causing changes in the ECG, but elevated concentrations of NT-proBNP and altered ECG characteristics may reflect shared underlying cardiovascular processes.

Increased myocardial wall stress, which is often associated with elevated NT-proBNP concentrations, can occur as a consequence of cardiac pressure and may be accompanied by functional or structural changes of the heart[19]. These changes can include alterations in ventricular and atrial dimensions[62]. Since such alterations can influence the electrical activity of the heart, they provide a possible physiological link between ECG changes and elevated NT-proBNP concentrations.

Furthermore, alterations of myocardial structure and ventricular function can affect the propagation and recovery of cardiac electrical activity, which may result in changes in cardiac conduction and repolarization[63]. These changes may then become visible as changes in the ECG waveform[64]. This further explains why cardiovascular conditions associated with elevated NT-proBNP concentrations may leave patterns in ECG recordings.

This basis is also consistent with the classification results documented in this work. The ability of the model to distinguish between lower and elevated NT-proBNP concentrations suggests that cardiovascular states associated with increased NTproBNP concentrations may be linked to detectable patterns in the ECG signals and patient metadata. These patterns may reflect underlying changes in cardiac physiology instead of NT-proBNP concentration itself. The classification results therefore support the physiological plausibility that elevated NT-proBNP and altered cardiovascular characteristics may be related through shared underlying physiological processes.

The lower performance of the three-class classification experiments may also be explained by the physiological relationship. NT-proBNP is measured on a continuous scale and is associated with various degrees of cardiac stress and dysfunction, whereas the NICE thresholds divide this continuous concentration into discrete groups[26]. Patients with NT-proBNP values close to a class boundary may therefore be assigned to different classes despite showing similar cardiovascular characteristics. This similarity in physiological characteristics between patients from neighboring classification ranges may then complicate the classification between the three NT-proBNP classes.

In addition, this similarity in physiological characteristics may be particularly relevant for the 400–2000 pg/mL class, which showed the lowest proportion of correctly classified samples in both ECG configurations. Patients in this range may differ considerably in their cardiovascular condition despite showing similar NT-proBNP concentrations. This physiological heterogeneity could have contributed to the lower classification performance for the 400–2000 pg/mL class.

Unlike the classification experiments, the regression experiments show a different physiological challenge. The model is required to predict the exact NT-proBNP concentration from the available ECG signals and patient information. Accurate regression requires the available physiological information to be sufficiently informative about the measured biomarker concentration. However, NT-proBNP concentrations are influenced by multiple physiological and patient-specific factors[25], which may limit the precision of the concentration prediction.

At higher NT-proBNP concentrations, the regression error showed a wider spread, indicating less precise predictions in this range. One possible physiological explanation is that high NT-proBNP concentrations may not correspond to one specific physiological or electrocardiographic pattern. Variability in the cardiovascular conditions associated with elevated NT-proBNP may therefore contribute to the difficulty of predicting precise biomarker concentrations from ECG signals.

The relationship between NT-proBNP concentration and cardiac physiology is also influenced by patient-specific factors[25]. Some of these factors are represented in the patient metadata used by the model and may therefore have provided additional context for the prediction of NT-proBNP concentration.

To summarize, the findings are physiologically consistent with the hypothesis that ECG signals and patient metadata contain indirect information associated with NT-proBNP elevation. However, since NT-proBNP concentrations are influenced by multiple physiological and patient-specific factors, the prediction of precise concentration values remains challenging.

## 6.2Technical Interpretation

The results in this work show a consistent improvement in the predictive performance of the 12-lead ECG configurations compared to the 1-lead configurations in all evaluated prediction tasks. Specifically, the 12-lead model achieved lower prediction error in the regression experiments and higher AUROC values in the classification experiments. The 1-lead configuration in this work used lead II, which only represented the electrical activity from one spatial perspective[17]. The additional spatial information may have been one reason for the improvement in the predictive performance. Additionally, ECG characteristics that may be less apparent in lead II may be more clearly visible in the 12-lead configuration.

Despite the consistent improvement of the 12-lead configuration, the actual improvement remained relatively moderate across the evaluated tasks. The regression experiment showed a mean absolute error decrease from 534.98 pg/mL to 514.27 pg/mL while the AUROC increased in the binary classification experiment from 0.849 to 0.876 and in the three-class classification experiment from 0.738 to 0.752. These results suggest that the 1-lead multimodal configuration already provided a considerable amount of information relevant to the prediction task. At the same time, this improvement comes with a more complex ECG waveform configuration.

Despite the difference in regression performance, the experiments showed a similar error pattern for both models. The Bland-Altman analysis showed a negative mean difference for the 1-lead and 12-lead configurations, indicating an overall tendency for the model to predict lower NT-proBNP concentrations than the corresponding target values. Both configurations also showed a similar spread of prediction errors. Since the general error pattern was documented for both ECG configurations, using the 12-lead compared to using the 1-lead did not substantially change the general regression behavior.

The Passing-Bablok regression further supports this claim. Both ECG configurations showed slopes clearly below one and positive intercepts, indicating that the predicted NT-proBNP concentrations increased more slowly than the associated laboratory values. While the 12-lead configuration showed a slightly higher intercept, its slope was slightly closer to one. Overall, both configurations demonstrated a similar pattern of deviation from the target values, indicating further that using additional leads may not change the regression error pattern substantially.

The binary classification results showed that the improvement in predictive performance by using the 12-lead configuration was not limited to a single evaluation metric. In addition to improvements in AUROC, accuracy, sensitivity and F1 score also improved slightly. These improvements across different evaluation metrics suggest that the 12-lead configuration may support a more consistent separation between lower and elevated NT-proBNP concentrations.

The three-class classification task also showed improvements, although they were less evident. The 12-lead configuration showed a higher AUROC, accuracy, sensitivity and F1 score. However, the improvement was not equally distributed across all NT-proBNP classes. More samples were correctly classified in the below 400 pg/mL class, while the 400–2000 pg/mL and above 2000 pg/mL classes showed a decrease in correctly classified samples. This indicates that an improvement in overall classification performance does not necessarily result in more correctly classified samples for every class.

Overall, the results show that the model achieved better predictive performance across all experiments using the 12-lead ECG configuration compared to using the 1-lead configuration. However, this improvement is only moderate. Furthermore, both regression experiments showed similar predictive deviations from the laboratory values, indicating that the use of the 12-lead configurations compared to the 1-lead configurations did not change the difficulty for the model to predict a precise NTproBNP concentration.

## 6.3 Comparison with Previous Work

The results of this work are broadly consistent with previous research indicating that ECG recordings with patient metadata contain information related to laboratory value prediction. López Alcaraz and Strodthoff investigated the prediction of multiple laboratory abnormalities from ECG signals and clinical data, also using the MIMIC-IV database. Among the investigated laboratory values, NT-proBNP achieved the highest predictive performance, with an AUROC of 0.903 for the prediction of concentrations above 353 pg/mL[9]. The binary classification experiments in this work also showed that elevated NT-proBNP concentrations may be distinguished from lower concentrations with an AUROC of 0.876 using the 12-lead configuration.

These performance values should not be compared directly, however, since the experimental settings differ. López Alcaraz and Strodthoff used a threshold of 353 pg/mL, while this work used a threshold of 300 pg/mL for the binary classification task and different dataset construction and temporal matching. However, both works showed that ECG signals combined with patient metadata contain predictive information about elevated NT-proBNP concentrations.

In addition, this work extended the investigation of NT-proBNP to additional prediction tasks, such as regression and three-class classification based on clinically established NICE thresholds[26]. The regression experiments showed that the prediction of precise NT-proBNP concentrations remains challenging, while the threeclass classification experiments were able to distinguish between the concentration ranges, albeit the predictive performance was lower than in the binary classification task.

Furthermore, this work extended the prediction tasks with a comparison between 1-lead and 12-lead configurations. While López Alcaraz and Strodthoff used 12-lead

ECG recordings, this work additionally evaluated 1-lead configurations with lead II and found consistently better predictive performance with the 12-lead configurations.

## 6.4 Clinical Relevance

The results of this work suggest that ECG-based prediction of NT-proBNP may have potential clinical relevance as an additional source of information about the cardiovascular state of a patient. Acquisition of ECG is noninvasive, as opposed to the common approach of using invasive blood sampling[4]. Therefore, a trained model could be used with ECG signals and patient metadata to provide additional information to clinicians before laboratory results are available. This information should therefore be considered complementary to a direct laboratory measurement rather than as a replacement for it.

The results of the classification tasks may be particularly relevant from a clinical perspective. The classification tasks used in this work defined concentration groups based on established NT-proBNP thresholds[26]. The model could therefore potentially provide an early indication of whether the NT-proBNP concentration falls within an elevated concentration range before the actual laboratory results are available.

The comparison between the 1-lead configuration and 12-lead configuration is also relevant when considering potential clinical applications. Although consistently better prediction performance was documented in the 12-lead configuration, the difference compared to 1-lead remained moderate, indicating that the 1-lead multimodal configuration already retained considerable predictive performance, which may partly be explained by the ECG machine measurements being calculated using all 12 leads. However, if a 12-lead ECG is already available, the results suggest that using the 12-lead waveform input may lead to better predictive performance compared to 1-lead.

Another potential clinical use for this model may be as a preliminary indicator for elevated NT-proBNP concentrations. The information from the model could then be used as an additional indicator of whether further clinical assessment may be warranted for the patient. While the regression results showed limited predictive performance for precise NT-proBNP concentration prediction, the classification experiments showed that the model could distinguish between concentration groups. This indicates that the model may be used as additional decision support rather than a diagnostic measurement.

Multimodal approaches combining ECG recordings with clinical information have been investigated for clinical decision support in intensive care units and emergency departments[6, 59]. These settings may be particularly relevant since ECG recordings and other patient information are routinely available. However, further validation is required before the NT-proBNP prediction approach investigated in this work could be applied in such clinical settings.

Overall, ECG-based NT-proBNP prediction may be most useful as a complementary source of information. In particular, the classification of broader concentration ranges may provide useful preliminary information, while the results from the regression experiments indicate that precise prediction of NT-proBNP concentration remains challenging.

## 6.5 Generalization to Other Laboratory Values

The additional experiments on hemoglobin, glucose and potassium showed that the multimodal ECG-based approach can be applied to different laboratory values with varying predictive performance. This is consistent with previous research showing that the predictive performance of ECG-based laboratory value prediction differs between individual biomarkers[9].

The regression experiments showed a similar pattern across several laboratory values. For hemoglobin, glucose and potassium the predicted concentrations varied less than the corresponding reference concentrations. This indicates that the models had difficulty predicting precise concentration values and tended to predict values closer to the middle of the target ranges.

The classification experiments for glucose and potassium further showed limited performance in distinguishing between multiple concentration ranges. Although potassium is known to affect ECG morphology[52], the experiments still showed difficulty reliably distinguishing the defined classes

Overall, the additional experiments indicate that the applicability of ECG-based laboratory value prediction depends on the laboratory target and the relationship with the information available from the ECG and patient metadata. The predictive performance observed for NT-proBNP cannot be assumed to generalize directly to other laboratory values. Since the additional experiments partly used different preprocessing settings and feature configurations, the different laboratory values should not be compared directly.

## 6.6 Limitations

One limitation of this work is the use of data from a single clinical database. Consequently, the investigated patients and their associated data originated from Beth Israel Deaconess Medical Center[53–55], indicating that the predictive performance may not generalize to patients from other clinical environments. External validation is required to confirm whether the predictive performance remains stable across different datasets.

Another limitation is the temporal matching between ECG recordings and laboratory measurements. In this work, ECG recordings were matched to the nearest NT-proBNP laboratory measurement using a temporal window of ±3 hours. Since a time difference still remained for some ECG and laboratory measurement pairs, the ECG recording and the corresponding laboratory measurement may not necessarily reflect the same physiological state.

All primary experiments in this work used both ECG recordings and patient metadata, meaning that the actual influence of each modality has not been evaluated separately in this work. Although the comparison between 1-lead and 12-lead ECG recordings showed the influence of ECG lead configuration, the relative contributions of ECG and patient metadata remain unclear. Furthermore, since the machine measurements used by both the 1-lead and 12-lead configurations were calculated on all 12 leads, the predictive performance difference may be underestimated.

Finally, another limitation is that the time difference between each ECG recording and its associated laboratory measurement was included as an input feature. This information may not be available when the model is used as a preliminary indicator of elevated NT-proBNP concentrations.

## 6.7 Future Work

Future research should first investigate the proposed approach using independent datasets to evaluate the generalization of the model. Applying the same prediction tasks and ECG configurations to unseen data would verify the stability of the results documented in this work. Independent datasets from different clinical environments, such as the Acutelines emergency department data-biobank[65], could provide opportunities for external validation.

Another direction of future research is a systematic comparison of each individual input modality for NT-proBNP concentration prediction and their contribution to the predictive performance. This research could also be extended to investigate whether the ECG lead configuration still improves the predictive performance when patient metadata is excluded from the model. Furthermore, alternative ECG representations and longer ECG recordings could be investigated. Recent work has shown that ECG-derived spectrograms combined with clinical information can support the prediction of patient deterioration in an emergency department setting[66].

An additional extension would be a more systematic investigation of ECG lead selection. This work compared a 1-lead configuration using lead II with the complete 12-lead configuration, but did not investigate which leads or lead combinations contribute the most to the NT-proBNP prediction. This could identify whether similar or even better predictive performance can be achieved using fewer than 12 leads.

Another direction of future research could be the explainability of ECG-based laboratory value prediction. Explainability methods could be used to investigate which parts of the ECG waveform and which patient features contribute most to individual predictions[67]. This could also provide additional insights into the physiological relationships underlying ECG-based NT-proBNP prediction.

Lastly, future research could investigate other prediction formulations for three or more NT-proBNP concentration groups. The classes investigated in this work have a natural ordering from lower to higher NT-proBNP concentrations, while the current classification approach treats them as completely separate categorical classes. Therefore, future experiments could investigate an ordinal classification approach that considers the relationship between the classes.

## 7 Conclusion

This thesis investigated whether NT-proBNP concentrations can be predicted from ECG recordings and patient metadata using a multimodal deep learning framework. The experiments included regression, binary classification and three-class classification tasks and compared a 1-lead waveform configuration using lead II to a 12-lead waveform configuration. Furthermore, the physiological plausibility of ECG-based NT-proBNP prediction was investigated.

Across all investigated experiments, the 12-lead configuration achieved better predictive performance compared to the associated 1-lead configuration. In the regression experiment, the mean absolute error decreased from 534.98 pg/mL in the 1-lead configuration to 514.27 pg/mL in the 12-lead configuration. In the binary classification experiment, the AUROC increased from 0.849 to 0.876, while in the three-class classification experiment, the macro-averaged AUROC increased from 0.738 to 0.752. The three-class classification task showed lower predictive performance compared to the binary classification task. The 400-2000 pg/mL class showed the lowest proportion of correctly classified samples in both ECG configurations. Although the improvements were consistent, they remained moderate, indicating that the 1-lead configuration already provided considerable predictive information. The additional spatial information from the 12-lead configuration may have contributed to the improved prediction performance, while the regression error patterns remained similar.

The results in this work are consistent with the hypothesis that ECG signals contain indirect information about NT-proBNP concentration. Furthermore, elevated NTproBNP concentrations may reflect underlying cardiovascular processes that are associated with altered ECG characteristics as discussed in Section 6.1. The results should not be interpreted as showing that NT-proBNP directly influences ECG characteristics, but rather that NT-proBNP concentrations and ECG characteristics may both be associated with the same underlying physiological processes.

In conclusion, the findings in this work support the feasibility of ECG-based NTproBNP prediction as a complementary source of information. Notably, the classification experiments showed that the model could distinguish between broader NT-proBNP concentration ranges, while regression tasks remain challenging. Therefore, the proposed approach may provide complementary information alongside direct laboratory measurements for NT-proBNP, while the additional experiments on hemoglobin, glucose and potassium indicate that the observed predictive performance cannot be directly generalized to other laboratory values. More research is required to determine whether the model can generalize to other patient populations and clinical settings.

## A Additional NT-proBNP experiments

This appendix section presents additional experiments conducted for the prediction of NT-proBNP from ECG recordings and patient metadata. These experiments complement the primary NT-proBNP experiments presented in the main part of this thesis.

## A.1 Experimental Setup

The additional experiments included an alternative classification formulation and comparisons between different regression configurations.

The general dataset construction and training pipeline followed the methodology described in the main part of this thesis unless specified otherwise. The model architecture remained unchanged across all additional experiments.

The first group of additional experiments investigated a classification task using simplified NT-proBNP concentration classes inspired by the structure of the New York Heart Association (NYHA) classification[68] with a 1-lead ECG configuration. The NT-proBNP concentration ranges were below $\mathrm { 3 0 0 ~ p g / m L }$ , between 300 and $1 0 0 0 ~ \mathrm { p g / m L }$ , between 1000 and 2000 $\mathrm { p g / m L }$ , and above 2000 $\mathrm { p g / m L }$ . These ranges do not correspond directly to the NYHA classes. For these experiments, different loss functions were evaluated, including cross entropy, weighted cross entropy and weighted focal cross entropy loss. For the weighted configurations, class weights were determined based on the class distribution in the training set.

The next group of additional experiments investigated different target scaling settings using a 1-lead ECG configuration. The target scaling settings included MinMax and standard scaling, with mean squared error as the loss function. All configurations in this comparison used an earlier approach with a temporal matching window of one hour, an upper NT-proBNP clipping threshold of 10000 pg/mL and no patient vital signs.

## A.2 Results

## A.2.1 NYHA Inspired Four-Class Classification

Table 10: Predictive results of the NYHA inspired NT-proBNP classification experiments.
<table><tr><td>Loss configuration</td><td>Macro AUROC</td><td>95% CI</td><td></td><td>Accuracy Sensitivity F1 score</td><td></td></tr><tr><td>Cross entropy</td><td>0.714</td><td>0.705-0.722</td><td>0.544</td><td>0.413</td><td>0.367</td></tr><tr><td>Weighted cross entropy</td><td>0.726</td><td>0.717-0.734</td><td>0.471</td><td>0.413</td><td>0.398</td></tr><tr><td>Weighted focal cross entropy</td><td>0.709</td><td>0.700-0.717</td><td>0.478</td><td>0.435</td><td>0.421</td></tr></table>

Table 10 shows the resulting macro-averaged AUROC values for the three evaluated loss configurations. The weighted cross entropy loss configuration achieved the highest macro-averaged AUROC of 0.726 (95% CI: 0.717–0.734), followed by unweighted cross entropy with a macro-averaged AUROC of 0.714 (95% CI: 0.705–0.722) and lastly weighted focal cross entropy loss with 0.709 (95% CI: 0.700–0.717). The 95% confidence intervals showed overlap across the three configurations. Cross entropy loss achieved the highest accuracy, while weighted focal cross entropy loss achieved the highest macro sensitivity and macro F1 score.

Figure 7 shows the confusion matrices for the three evaluated configurations. The unweighted cross entropy configuration mostly assigned samples to the lowest and highest NT-proBNP classes, while the middle classes were rarely predicted. The weighted configurations showed a more distributed prediction pattern.

![](images/5e9eaf3ea8d88a033311cce13a46e39baf1496008856a0a69664f2b9482efe1d.jpg)

(a) Unweighted cross entropy  
![](images/805c50d238a15749d988939aad857f43a6f174a4296c5bc2cbe4e763b3a37f65.jpg)

(b) Weighted cross entropy  
![](images/593be2a1ed773fb3d39268d3113c27cd0d2ac6a9ebd86e8074554369a3603fe4.jpg)  
(c) Weighted focal cross entropy  
Figure 7: Confusion matrices for the NYHA inspired NT-proBNP classifications

## A.2.2 Regression Configuration Experiments

Table 11: Predictive results of the additional NT-proBNP regression configurations.
<table><tr><td colspan="2">Target scaling MAE [pg/mL] 95% CI [pg/mL]</td></tr><tr><td>MinMax</td><td>2390.46 2314.93-2460.34</td></tr><tr><td>Standard 2317.63</td><td>2242.62-2396.17</td></tr></table>

Table 11 summarizes the predictive performance of the evaluated regression configurations with standard scaling showing the smaller prediction error.

![](images/5261249d55a3a531fc6a020b47f7bd4329522078e5670c31298bef8611cc98c1.jpg)

(a) MinMax scaling Bland-Altman Plot - Standard  
![](images/051661da59b110fb0022eac403f07113cb04c3cc44e4187a31dea53b61eff5d8.jpg)  
(b) Standard scaling  
Figure 8: Bland-Altman analysis of predicted and reference NT-proBNP concentrations for the MinMax and standard scaling configurations. The solid line represents the mean difference and the dashed lines indicate the 95% limits of agreement.

Figure 8 shows the Bland-Altman analysis for the target scaling configurations. MinMax scaling resulted in a mean difference of $- 7 9 9 . 4 4 ~ \mathrm { p g / m L }$ , while standard scaling showed a mean difference of $- 1 1 8 2 . 4 9 ~ \mathrm { p g / m L }$ . The linear structure visible at high NT-proBNP concentrations resulted from the upper clipping threshold of 10000 pg/mL.

![](images/87c05f90ff3322c4b22bfc067c41b96dc26d32b3eadfca5e06d9862b775c9624.jpg)  
(a) MinMax scaling

![](images/09a1546ceb0fcabe6c6308817453580558b04997c51378ba8b8b207f095f6c5f.jpg)  
(b) Standard scaling  
Figure 9: Passing-Bablok regression analysis of predicted and reference NT-proBNP concentrations for the MinMax and standard scaling configurations. The solid line represents the Passing-Bablok regression fit, the dashed line represents the identity, and the shaded area represents the 95% confidence interval.

Figure 9 shows the Passing-Bablok regression analysis for both scaling configurations. MinMax scaling resulted in an intercept of 422.79 pg/mL (95% CI: 400.79– 454.43 $\mathrm { p g / m L ) }$ and a slope of 0.6 (95% CI: 0.57–0.63), while the standard scaling achieved an intercept of 524.31 $\mathrm { p g / m L }$ (95% CI: 482.18–565.62 pg/mL) and a slope of 0.46 (95% CI: 0.44–0.48). The linear structure at high NT-proBNP concentrations also resulted from the upper clipping threshold of 10000 pg/mL.

## A.3 Interpretation

The NYHA inspired classification experiments showed that different loss configurations affected the overall predictive performance of the model. Weighted cross entropy achieved the highest macro-averaged AUROC, while the overall confidence intervals showed overlap. Unweighted cross entropy predicted the lowest and highest NT-proBNP classes the most, whereas the 300–1000 pg/mL and 1000–2000 pg/mL classes were predicted the least. The weighted configurations showed better predictive results for the middle classes, while the overall predictive pattern appeared to be the same. These results also show the same difficulty for the model to predict multiple precise NT-proBNP concentration ranges, similar to the primary experiments.

The regression configuration experiments showed a small difference in mean absolute error between MinMax and standard target scaling. The Bland-Altman analysis showed a mean difference closer to 0 for the MinMax scaling compared to the standard target scaling, indicating a lower negative bias. The Passing-Bablok regression analysis further showed a slope closer to the identity for MinMax. Overall, the results indicate that standard scaling resulted in lower prediction error, while MinMax showed lower systematic bias. These findings are consistent with the regression pattern observed in the primary experiments, where precise NT-proBNP concentration prediction remained challenging.

## B Hemoglobin

Hemoglobin was investigated as an additional laboratory target to assess whether the multimodal ECG-based prediction approach can be used for laboratory values other than NT-proBNP. The following experiments evaluate its predictive performance using ECG recordings and patient metadata.

## B.1 Experimental Setup

The experiments for hemoglobin followed the same methodology used for the primary NT-proBNP experiments. ECG recordings were combined with the same 33 features and processed using the same model architecture and training configuration.

Hemoglobin measurements were matched to ECG recordings using a temporal window of ±2 minutes. The higher availability of hemoglobin measurements allowed for a smaller temporal window compared to the primary NT-proBNP measurements and resulted in 19176 ECG-laboratory pairs. No upper threshold was applied to the hemoglobin target values.

The target values were standardized using a StandardScaler fitted exclusively on the training split. L1 loss was used as the loss function and mean absolute error as the evaluation metric, identical to the primary NT-proBNP regression experiments in this work. Experiments were conducted for both 1-lead and 12-lead ECG configurations.

## B.2 Results

The test split consisted of 957 ECG recordings, corresponding to 3828 2.5 second ECG segments.

Table 12: Mean absolute error with 95% confidence intervals for hemoglobin prediction.
<table><tr><td colspan="3">ECG configuration MAE  $[ \mathrm { g } / \mathrm { d } \mathrm { L } ]$  95% CI  $[ \mathrm { g } / \mathrm { d } \mathrm { L } ]$ </td></tr><tr><td>1-lead</td><td>1.6008</td><td>1.5585-1.6441</td></tr><tr><td>12-lead</td><td>1.5991</td><td>1.5562-1.6413</td></tr></table>

Bland-Altman Plot - 1-lead

Table 12 shows the regression performance of the 1-lead and 12-lead experiments. The 1-lead model achieved an MAE of 1.6008 g/dL (95% CI: 1.5585–1.6441 $\mathrm { g / d L ) }$ 2 while the 12-lead model achieved an MAE of 1.5991 $\mathrm { g / d L }$ (95% CI: 1.5562–1.6413 g/dL).

![](images/d2c48caeb6b2644ad122ef0412a9a398bc46b693fbeb54a3f62ac68883a08d01.jpg)  
(a) 1-lead ECG

Bland-Altman Plot - 12-lead  
![](images/435d47ba836d1c165ea07959c4a6df58ac13d8bf6b3f402952926dba1921dc93.jpg)  
(b) 12-lead ECG  
Figure 10: Bland-Altman analysis of predicted and reference hemoglobin concentrations for the 1-lead and 12-lead ECG configurations. The solid line represents the mean difference and the dashed lines indicate the 95% limits of agreement.

Figure 10 illustrates the Bland-Altman analysis for both ECG configurations. The 1-lead model showed a mean difference of $0 . 0 1 \ \mathrm { g / d L }$ with limits of agreement from -4.12 $\mathrm { g / d L }$ to 4.14 $\mathrm { g / d L }$ , while the 12-lead model showed a mean difference of - $0 . 0 5 ~ \mathrm { g } / \mathrm { d } \mathrm { L }$ with limits of agreement from -4.14 $\mathrm { g / d L }$ to 4.03 $\mathrm { g / d L }$ •

![](images/077a14662c3bed5af2141087c9ace9bcb9a6203336b20a0fe394808d0e3bc34d.jpg)  
(a) 1-lead ECG

![](images/ab3fd1e86e8c96b33e0545df39f1d58c3bd61cf370c7a01ca25f8e83868e10d0.jpg)  
(b) 12-lead ECG  
Figure 11: Passing-Bablok regression analysis of predicted and reference hemoglobin concentrations for the 1-lead and 12-lead ECG configurations. The solid line represents the Passing-Bablok regression fit, the dashed line represents the identity, and the shaded area represents the 95% confidence interval.  
Figure 11 shows the Passing-Bablok regression analysis for 1-lead and 12-lead ECG

configurations. The intercept for the 1-lead configuration was 8.65 $\mathrm { g / d L }$ (95% CI: 8.47–8.87 g/dL) and the slope was 0.31 (95% CI: 0.29–0.32), while for the 12-lead configuration the intercept was 8.87 g/dL (95% CI: 8.70–9.05 g/dL) and the slope was 0.29 (95% CI: 0.27–0.30).

## B.3 Interpretation

The hemoglobin regression experiments showed very close predictive performance for the 1-lead and 12-lead ECG configurations with only minimal differences in mean absolute error between both ECG configurations.

The Bland-Altman analysis further showed that both configurations had very similar mean differences close to zero. However, the limits of agreement show that the predicted concentrations may differ considerably from the reference laboratory measurements.

Passing-Bablok analysis showed slopes clearly below 1 for both ECG configurations, indicating that the predicted hemoglobin concentrations varied considerably less than the corresponding reference values. Furthermore, the combination of intercept and slope indicates that the model tended to predict concentrations around the middle of the hemoglobin range.

Overall, the hemoglobin experiments showed no substantial difference in predictive performance using 1-lead compared to 12-lead ECG configurations. Furthermore, the regression task to predict precise hemoglobin concentrations remains challenging

## C Glucose

Blood glucose was included as an additional prediction target to investigate the prediction of a metabolic laboratory value. The experiments document whether information associated with blood glucose concentration can be extracted from ECG recordings in combination with patient metadata.

## C.1 Experimental Setup

The glucose experiments used an earlier approach with 21 features and a 1-lead ECG configuration similar to the experiments in Section D.1. Patient vital signs were not included in the features. Values below 20 mg/dL and above 500 mg/dL were excluded in the preprocessing to limit the influence of extreme values.

The regression experiments used a temporal window of ±5 minutes, where for each ECG recording the closest glucose measurement was selected. L1 loss was used as the loss function and mean absolute error as the evaluation metric. No target scaling was applied in this experiment.

Furthermore, additional regression configurations with a temporal window of ±1 hour and different loss functions were investigated. These loss functions included Huber loss, Gaussian negative log-likelihood and a Clarke error grid weighted loss. These configurations were investigated to determine whether different loss functions could improve the prediction of blood glucose concentrations.

Finally, a three-class classification task for glucose was investigated. This experiment used the same ±5 minute temporal window. The classes were defined as hypoglycemia for glucose concentrations below 70 mg/dL, hyperglycemia for concentrations above 180 mg/dL and glucose concentrations between these thresholds[69] Weighted focal cross entropy loss with class weights accounting for the unequal class distribution was used, similar to the other classification tasks in this work.

## C.2 Results

The test split consisted of 1797 ECG recordings, corresponding to 7188 2.5 second ECG segments for the datasets with temporal windows of ±5 minutes.

## C.2.1 Regression

The regression experiment using a temporal window of ±5 minutes achieved a mean absolute error of 34.52 mg/dL with a 95% confidence interval from 33.47 mg/dL to 35.60 mg/dL.

Table 13: Regression configurations for glucose prediction.
<table><tr><td>Loss configuration</td><td>MAE [mg/dL] 95% CI [mg/dL]</td></tr><tr><td>L1 loss 34.26</td><td>33.81-34.69</td></tr><tr><td>Huber loss 34.32</td><td>33.87-34.81</td></tr><tr><td>Gaussian negative log-likelihood 37.75</td><td>37.35-38.14</td></tr><tr><td>Weighted Clarke error grid loss</td><td>48.40 48.04-48.72</td></tr></table>

Table 13 shows the predictive performance of the regression experiments for blood glucose using a ±1 hour temporal window and different loss functions. L1 loss achieved the lowest mean absolute error of 34.26 mg/dL (95% CI: 33.81–34.69 mg/dL), while weighted Clarke error grid loss achieved the highest error of 48.40 mg/dL (95% CI: 48.04–48.72 mg/dL).

![](images/0e0401163b0f1d43fcf9affa3ff89298c4f633894063b92704867800af399dee.jpg)  
Figure 12: Bland-Altman analysis of predicted and reference glucose concentrations using the ±5 minute temporal window. The solid line represents the mean difference and the dashed lines indicate the 95% limits of agreement.

Figure 12 shows the Bland-Altman analysis for the glucose prediction using the ±5 minute temporal window. The mean difference was -14.52 mg/dL, with limits of

![](images/5ee9b54fb13ead711cf24b6c62bdf9f44f7f0232d36f4e60e4bd9235e92fa66e.jpg)  
Figure 13: Passing-Bablok regression analysis for glucose prediction.

Figure 13 shows the Passing-Bablok regression analysis for glucose. The intercept was 90.96 mg/dL (95% CI: 89.56–92.47 mg/dL), while the slope was 0.21 (95% CI: 0.20–0.22).

## C.2.2 Classification

Table 14: Three-class classification performance for glucose prediction.
<table><tr><td>ECG configuration Macro AUROC</td><td></td><td>95% CI</td><td>Accuracy</td><td>Sensitivity</td><td>F1 score</td></tr><tr><td>1-lead</td><td>0.632</td><td>0.610-0.652</td><td>0.554</td><td>0.420</td><td>0.338</td></tr></table>

Table 14 shows the classification performance for glucose prediction. The model achieved a macro-averaged AUROC of 0.632 (95% CI: 0.610–0.652) and an accuracy of 0.554. The sensitivity and F1 score achieved were less than 0.5.

![](images/fbe8d890c646baa4d8031c54d0a9455ed37add34e861467a24510e95313cfbc5.jpg)  
Figure 14: Confusion matrix for the three-class glucose classification experiment.

Figure 14 shows the confusion matrix for the glucose classification experiment. The 70-180 mg/dL class was predicted most frequently, while the other classes were predicted considerably less frequently.

## C.3 Interpretation

The Bland-Altman analysis showed wide limits of agreement, indicating that the predicted glucose concentration may differ considerably from the reference glucose concentration.

Passing-Bablok further showed a slope considerably below the identity slope of 1. Combined with the intercept, this supports the pattern that the predicted glucose concentrations varied less than the reference concentrations. In addition, the regression experiment indicated that the model tended to predict values around the middle of the glucose range, similar to the hemoglobin experiments.

The comparison between the different loss functions for the glucose regression experiment showed that L1 loss achieved similar predictive performance to Huber loss. The other loss functions resulted in higher mean absolute error and therefore did not improve the predictive performance.

The three-class classification showed similar results to the regression experiment. The middle class was predicted most frequently, while the other classes were pre-

dicted considerably less reliably.

Overall, the glucose experiments showed limited predictive performance, with difficulty in predicting precise glucose concentrations and classifying broader concentration ranges.

## D Potassium

Potassium was selected as an additional laboratory target since changes in potassium concentration are known to affect cardiac electrophysiology and ECG morphology[52]. The following experiments document the use of ECG recordings and patient metadata for the prediction of potassium.

## D.1 Experimental Setup

The potassium experiments used an earlier approach with a 1-lead ECG configuration combined with 21 features. A temporal window of ±5 minutes was used for the matching of ECG recordings to laboratory measurements. Potassium values outside the range from 1.5 mEq/L to 8.0 mEq/L were excluded during preprocessing to reduce the influence of extreme outliers. Furthermore, patient vital signs were not included in the metadata.

The experiments included both a regression experiment and a three-class classification experiment. For the regression experiment, L1 loss was used as the loss function and mean absolute error as the evaluation metric. No target scaling was applied for this experiment.

The three classes were defined as low, normal and elevated potassium concentrations. The thresholds were concentrations below 3.5 mEq/L, between 3.5 mEq/L and 5.0 mEq/L and above 5.0 mEq/L[70]. Weighted focal cross entropy loss with class weights to account for the class distribution was used.

## D.2 Results

The test split for these experiments consisted of 1872 ECG recordings, corresponding to 7488 2.5 second ECG segments.

## D.2.1 Regression

The model achieved a mean absolute error of 0.51 mEq/L with a 95% confidence interval from 0.50 mEq/L to 0.52 mEq/L.

![](images/61ccb236824669b7df4f4969f638f9066209e7d74ecc0df276c6a2d5d4239795.jpg)  
Figure 15: Bland-Altman analysis of predicted and reference potassium concentrations using the 1-lead ECG configuration. The solid line represents the mean difference and the dashed lines indicate the 95% limits of agreement.

Figure 15 shows the Bland-Altman analysis for the potassium prediction. The mean difference was -0.12 mEq/L with limits of agreement from -1.5 mEq/L to 1.3 mEq/L.

![](images/d7ab8be1314ff9f9720eeb93890cf5788e599aceff10909a33f153f7590209f4.jpg)  
Figure 16: Passing-Bablok regression analysis for potassium prediction.

Figure 16 shows the Passing-Bablok regression analysis for potassium concentration prediction with an intercept of 3.78 mEq/L (95% CI: 3.75–3.80 mEq/L) and a slope of 0.10 (95% CI: 0.10–0.11).

## D.2.2 Classification

Table 15: Three-class classification performance for potassium prediction.
<table><tr><td>ECG configuration Macro AUROC</td><td></td><td>95% CI</td><td>Accuracy</td><td>Sensitivity</td><td>F1 score</td></tr><tr><td>1-lead</td><td>0.629</td><td>0.617-0.642</td><td>0.488</td><td>0.444</td><td>0.356</td></tr></table>

Table 15 shows the predictive performance of the three-class classification experiment for potassium. A macro-averaged AUROC of 0.629 (95% CI: 0.617–0.642) was achieved, while accuracy, sensitivity and F1 score were all below 0.5.

![](images/80aae08cfad7462f9237212c0547cc69b4a39f6d6978144e8392929e47674272.jpg)  
Figure 17: Confusion matrix for the three-class potassium classification experiment.

Figure 17 shows that the normal potassium class was predicted most frequently.   
The low and elevated potassium classes were frequently misclassified.

## D.3 Interpretation

For the regression experiment, the Bland-Altman analysis showed wide limits of agreement, indicating that the predicted concentrations may differ considerably from the corresponding laboratory measurements.

Passing-Bablok analysis further supported this pattern. The slope was 0.1, which is considerably below the identity slope of 1. Combined with the intercept, this indicates that the model had considerable difficulty predicting precise potassium concentrations.

The three-class classification experiment showed the same pattern in Figure 17, where the predicted classes varied considerably compared to the reference classes.

The potassium experiments showed limited predictive performance, with difficulty predicting precise potassium concentrations and classifying broader concentration ranges, similar to glucose.

## References

[1] Simona Ferraro, Federica Braga, and Mauro Panteghini. "Laboratory medicine in the new healthcare environment". In: Clinical Chemistry and Laboratory Medicine (CCLM) 54.4 (Jan. 2016). ISSN: 1437-4331, 1434-6621. DOI: 10 . 1515/cclm-2015-0803. URL: https://www.degruyter.com/document/doi/ 10.1515/cclm-2015-0803/html.

[2] Paul M. McKie and John C. Burnett. "NT-proBNP". en. In: Journal of the American College of Cardiology 68.22 (Dec. 2016), pp. 2437–2439. 1ssN: 07351097. DOI: 10.1016/j.jacc.2016.10.001. URL: https://1inkinghub.elsevier. com/retrieve/pii/S0735109716364622.

[3] Max C. Petersen, Daniel F. Vatner, and Gerald I. Shulman. “Regulation of hepatic glucose metabolism in health and disease". en. In: Nature Reviews Endocrinology 13.10 (Oct. 2017), pp. 572–587. ISSN: 1759-5037. DOI: 10 .1038/ nrendo.2017.80. URL: https://www.nature.com/articles/nrendo.2017. 80.

[4] Gabriel Lima-Oliveira et al. "Laboratory Diagnostics and Quality of Blood Collection". In: Journal of Medical Biochemistry 34.3 (July 2015), pp. 288— 294. ISSN: 1452-8258. DOI: 10.2478/jomb-2014-0043. URL: https://pmc. ncbi.nlm.nih.gov/articles/PMC4922344/.

[5] Brandon Yeshoua et al. “Interventions to reduce repetitive ordering of lowvalue inpatient laboratory tests: a systematic review". en. In: BMJ Open Quality 12.1 (Mar. 2023), e002128. ISSN: 2399-6641. DOI: 10.1136/bmjoq-2022- 002128. URL: https://qir.bmj.com/1ookup/doi/10.1136/bmjoq-2022- 002128.

[6] Juan Miguel Lopez Alcaraz, Hjalmar Bouma, and Nils Strodthoff. “Enhancing Clinical Decision Support with Physiological Waveforms — A Multimodal Benchmark in Emergency Care". In: Computers in Biology and Medicine 192 (June 1, 2025), p. 110196. ISSN: 0010-4825. DOI: 10 .1016/j. compbiomed. 2025.110196. URL: https://www.sciencedirect.com/science/article/ pii/S0010482525005475.

[7] Hafsa Habehh and Suril Gohel. "Machine Learning in Healthcare". In: Current Genomics 22.4 (Dec. 16, 2021), pp. 291–300. ISSN: 1389-2029. DOI: 10.2174/

1389202922666210705124359. PMID: 35273459. URL: https://pmc.ncbi. nlm.nih.gov/articles/PMC8822225/.

[8] Meenal J. Patel et al. “Machine Learning Approaches for Integrating Clinical and Imaging Features in Late-Life Depression Classification and Response Prediction". In: International Journal of Geriatric Psychiatry 30.10 (2015), pp. 1056–1067. ISSN: 1099-1166. DOI: 10 .1002/gps .4262. URL: https:// onlinelibrary.wiley.com/doi/abs/10.1002/gps.4262.

[9] Juan Miguel Lopez Alcaraz and Nils Strodthoff. “Abnormality Prediction and Forecasting of Laboratory Values from Electrocardiogram Signals Using Multimodal Deep Learning". In: Scientific Reports 15.1 (Nov. 10, 2025), p. 39362. ISSN: 2045-2322. DOI: 10.1038/s41598-025-26715-5. URL: https://www. nature.com/articles/s41598-025-26715-5.

[10] Adrienne Kline et al. “Multimodal Machine Learning in Precision Health: A Scoping Review". In: npj Digital Medicine 5.1 (Nov. 7, 2022), p. 171. ISSN: 2398-6352. DOI: 10.1038/s41746-022-00712-8. URL: https://www.nature. com/articles/s41746-022-00712-8.

[11] Alexandr Ceasovschih et al. “Electrocardiogram Features in Non-Cardiac Diseases: From Mechanisms to Practical Aspects". In: Journal of Multidisciplinary Healthcare 17 (Apr. 20, 2024), pp. 1695–1719. ISSN: null. DOI: 10.2147/JMDH. S445549. PMID: 38659633. URL: https://www.tandfonline.com/doi/abs/ 10.2147/JMDH.S445549.

[12] Basheer Abdulah Marzoog and Philipp Kopylov. “Machine Learning Detection of Metabolic Syndrome through Single-Lead Electrocardiography: Digital Phenotyping of Cardiac Electrical Alterations". In: Biomedical Signal Processing and Control 116 (May 1, 2026), p. 109616. ISSN: 1746-8094. DOI: 10.1016/ j.bspc.2026.109616. URL: https://www.sciencedirect.com/science/ article/pii/S1746809426001709.

[13] Conner D. Galloway et al. “Development and Validation of a Deep-Learning Model to Screen for Hyperkalemia From the Electrocardiogram". In: JAMA Cardiology 4.5 (May 1, 2019), pp. 428–436. ISSN: 2380-6583. DOI: 10 .1001/ jamacardio.2019.0640. URL: https://doi.org/10.1001/jamacardio. 2019.0640.

[14] Paul A. Iaizzo. Handbook of Cardiac Anatomy, Physiology, and Devices. Cham, SWITZERLAND: Springer, 2024. ISBN: 978-3-031-72581-4. URL: http : // ebookcentral.proquest.com/lib/zbmed-ebooks/detail.action?docID= 31823687.

[15] Alfred E. Buxton et al. “ACC/AHA/HRS 2006 Key Data Elements and Definitions for Electrophysiological Studies and Procedures". In: JACC 48.11 (Dec. 2006), pp. 2360–2396. DOI: 10 .1016/j.jacc .2006.09.020. URL: https : //www.jacc.org/doi/abs/10.1016/j.jacc.2006.09.020.

[16] Borys Surawicz et al. “AHA/ACCF/HRS Recommendations for the Standardization and Interpretation of the Electrocardiogram: Part III: Intraventricular Conduction Disturbances: A Scientific Statement From the American Heart Association Electrocardiography and Arrhythmias Committee, Council on Clinical Cardiology; the American College of Cardiology Foundation; and the Heart Rhythm Society: Endorsed by the International Society for Computerized Electrocardiology". en. In: Circulation 119.10 (Mar. 2009). ISSN: 0009- 7322, 1524-4539. DOI: 10.1161/CIRCULATIONAHA.108.191095. URL: https: //www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.108.191095.

[17] Paul Kligfield et al. “Recommendations for the Standardization and Interpretation of the Electrocardiogram: Part I: The Electrocardiogram and Its Technology: A Scientific Statement From the American Heart Association Electrocardiography and Arrhythmias Committee, Council on Clinical Cardiology; the American College of Cardiology Foundation; and the Heart Rhythm Society Endorsed by the International Society for Computerized Electrocardiology". en. In: Circulation 115.10 (Mar. 2007), pp. 1306–1324. ISSN: 0009- 7322, 1524-4539. DOI: 10.1161/CIRCULATIONAHA.106.180200. URL: https: //www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.106.180200.

[18] Mohamed F. Issa et al. “Heartbeat classification based on single lead-II ECG using deep learning". English. In: Heliyon 9.7 (July 2023). ISSN: 2405-8440. DOI: 10.1016/j.heliyon.2023.e17974. URL: https://www.cel1.com/ heliyon/abstract/S2405-8440(23)05182-4.

[19] Toshio Nishikimi and Yasuaki Nakagawa. "B-Type Natriuretic Peptide (BNP) Revisited—Is BNP Still a Biomarker for Heart Failure in the Angiotensin

Receptor/Neprilysin Inhibitor Era?" In: Biology 11.7 (July 2022), p. 1034. ISSN: 2079-7737. DOI: 10.3390/biology11071034. URL: https://www.mdpi. com/2079-7737/11/7/1034.

[20] Tim G. Yandle and A. Mark Richards. "B-Type Natriuretic Peptide Circulating Forms: Analytical and Bioactivity Issues". In: Clinica Chimica Acta 448 (Aug. 25, 2015), pp. 195–205. ISSN: 0009-8981. DOI: 10.1016/j. cca.2015. 07.004. URL: https://www.sciencedirect.com/science/article/pii/ S0009898115003290.

[21] Mark Oremus et al. “A Systematic Review of BNP and NT-proBNP in the Management of Heart Failure: Overview and Methods". In: Heart Failure Reviews 19.4 (Aug. 1, 2014), pp. 413–419. ISSN: 1573-7322. DOI: 10 .1007/ s10741-014-9440-0. URL: https://doi.org/10.1007/s10741-014-9440- 0.

[22] MASATOSHI Minamisawa et al. “2023 Update of the Japanese Heart Failure Society Scientific Statement on BNP and NT-proBNP Levels in Heart Failure Practice". In: Journal of Cardiac Failure 31.9 (Sept. 1, 2025), pp. 1453–1459. ISSN: 1071-9164. DOI: 10.1016/j.cardfai1.2025.03.005. URL: https: //www.sciencedirect.com/science/article/pii/S1071916425001393.

[23] Aaron L. Baggish, Roland R. J. van Kimmenade, and James L. Januzzi. “The Differential Diagnosis of an Elevated Amino-Terminal Pro-B-Type Natriuretic Peptide Level". In: The American Journal of Cardiology. An International Consensus Statement Regarding Amino-Terminal Pro-B-Type Natriuretic Peptide Testing: The International NT-proBNP Consensus Panel 101.3, Supplement (Feb. 2008), S43–S48. ISSN: 0002-9149. DO1: 10.1016/j.amjcard. 2007.11.019. URL: https://www.sciencedirect.com/science/article/ pii/S0002914907022448.

[24] “Comparison of BNP and NT-proBNP in Patients With Heart Failure and Reduced Ejection Fraction". en. In: Circulation: Heart Failure (). URL: https: //www.ahajournals.org/doi/10.1161/CIRCHEARTFAILURE.119.006541.

[25] Antoni Bayes-Genis et al. "Practical Algorithms for Early Diagnosis of Heart Failure and Heart Stress Using NT-proBNP: A Clinical Consensus Statement from the Heart Failure Association of the ESC". In: European Journal of Heart

Failure 25.11 (Nov. 2023), pp. 1891–1898. ISSN: 1388-9842. DOI: 10 . 1002/ ejhf.3036. URL: https://doi.org/10.1002/ejhf.3036.

[26] Overview / Chronic Heart Failure in Adults: Diagnosis and Management / Guidance / NICE. Sept. 12, 2018. URL: https : / / www . nice . org . uk/ guidance/ng106.

[27] James L. Januzzi et al. “NT-proBNP testing for diagnosis and short-term prognosis in acute destabilized heart failure: an international pooled analysis of 1256 patients: The International Collaborative of NT-proBNP Study". In: European Heart Journal 27.3 (Feb. 2006), pp. 330–337. 1SSN: 0195-668X. DO1: 10.1093/eurheartj/ehi631. URL: https://doi.org/10.1093/eurheartj/ ehi631.

[28] Meraj Neyazi et al. “Deep learning-based NT-proBNP prediction from the ECG for risk assessment in the community". en. In: Clinical Chemistry and Laboratory Medicine (CCLM) 62.4 (Mar. 2024), pp. 740–752. ISSN: 1437-4331. DOI: 10.1515/cclm-2023-0743. URL: https://www.degruyterbril1.com/ document/doi/10.1515/cclm-2023-0743/html?lang=en.

[29] Amr Abdin et al. “Effects of Dapagliflozin According to QRS Duration Across the Spectrum of Left Ventricular Ejection Fraction: An Analysis of DAPA-HF and DELIVER". In: European Journal of Heart Failure 26.9 (Sept. 2024), pp. 1952–1963. ISSN: 1388-9842. DOI: 10 . 1002/ejhf . 3350. URL: https : //doi.org/10.1002/ejhf.3350.

[30] Ciro Indolfi et al. “Development and External Validation of an AI-ECG Algorithm for Estimating Elevated Serum NT-proBNP Levels". In: European Heart Journal - Quality of Care and Clinical Outcomes (Aug. 2026), qcag128. IssN: 2058-1742. DOI: 10.1093/ehjqcco/qcag128. URL: https://doi.org/10. 1093/ehjqcco/qcag128.

[31] Varshil Prajapati, Shrikrishna Kolhar, and Dhanashree Barbole. “A Comprehensive Review on ECG Diagnosis and Report Generation Using AI: Technologies, Challenges, and Future Research Directions". In: Informatics in Medicine Unlocked 64 (July 1, 2026), p. 101777. ISSN: 2352-9148. DOI: 10.1016/j.imu. 2026.101777. URL: https://www.sciencedirect.com/science/article/ pii/S235291482600047X.

[32] Arunashis Sau and Fu Siong Ng. “ The Emerging Role of Artificial Intelligence Enabled Electrocardiograms in Healthcare". In: BMJ Medicine 2.1 (July 2023), e000193. ISSN: 2754-0413. DOI: 10.1136/bmjmed-2022-000193. URL: https://bmjmedicine.bmj.com/lookup/doi/10.1136/bmjmed-2022- 000193.

[33] Krzysztof Kraik et al. “The Most Common Errors in Automatic ECG Interpretation". In: Frontiers in Physiology 16 (May 22, 2025). ISSN: 1664-042X. DOI: 10.3389/fphys.2025.1590170. URL: https://www.frontiersin.org/ journals/physiology/articles/10.3389/fphys.2025.1590170/ful1.

[34] Nikita Rafie, Anthony H. Kashou, and Peter A. Noseworthy. "ECG Interpretation: Clinical Relevance, Challenges, and Advances". In: Hearts 2.4 (Dec. 2021), pp. 505–513. ISSN: 2673-3846. DOI: 10 .3390 /hearts2040039. URL: https://www.mdpi.com/2673-3846/2/4/39.

[35] Manuel Martínez-Sellés and Manuel Marina-Breysse. “Current and Future Use of Artificial Intelligence in Electrocardiography". In: Journal of Cardiovascular Development and Disease 10.4 (Apr. 2023), p. 175. ISsN: 2308-3425. DO1: 10. 3390/jcdd10040175.URL: https://www.mdpi.com/2308-3425/10/4/175.

[36] Iqbal H. Sarker. "Deep Learning: A Comprehensive Overview on Techniques, Taxonomy, Applications and Research Directions". In: Sn Computer Science 2.6 (2021), p. 420. ISSN: 2662-995X. DOI: 10 . 1007/ s42979- 021- 00815- 1. PMID: 34426802. URL: https://pmc.ncbi.nlm.nih.gov/articles/ PMC8372231/.

[37] Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. "Deep Learning". In: Nature 521.7553 (May 2015), pp. 436–444. ISSN: 1476-4687. DOI: 10 . 1038 / nature14539. URL: https://www.nature.com/articles/nature14539.

[38] Albert Gu, Karan Goel, and Christopher Ré. Efficiently Modeling Long Sequences with Structured State Spaces. arXiv:2111.00396 [cs.LG]. Aug. 2022. DOI: 10.48550/arXiv.2111.00396. URL: http://arxiv.org/abs/2111. 00396.

[39] Temesgen Mehari and Nils Strodthoff. Advancing the State-of-the-Art for ECG Analysis through Structured State Space Models. arXiv:2211.07579 [cs.LG].

Nov. 2022. DOI: 10.48550/arXiv.2211.07579. URL: http://arxiv.org/ abs/2211.07579.

[40] M. A. Al-Masud, Juan Miguel Lopez Alcaraz, and Nils Strodthoff. “Benchmarking ECG FMs: A Reality Check Across Clinical Tasks". en. In: International Conference on Learning Representations 2026 (Apr. 2026), pp. 146689– 146715. URL: https://proceedings.iclr.cc/paper\_files/paper/2026/ hash/ed49dc4569d0c93dab63c6e0a045f6c5-Abstract-Conference.html.

[41] David Holzmüller, Léo Grinsztajn, and Ingo Steinwart. “Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data". In: Neural Information Processing Systems. Vancouver (BC), Canada, Dec. 2024. DOI: 10.48550/arXiv.2407.04491. URL: https://hal.science/hal-04641923.

[42] Vijay Huddar et al. “Predicting Complications in Critical Care Using Heterogeneous Clinical Data". In: IEEE Access 4 (2016), pp. 7988–8001. ISSN: 2169- 3536. DOI: 10 .1109/ACCESS.2016.2618775. URL: https://ieeexplore. ieee.org/abstract/document/7593335.

[43] Felix Krones et al. “Review of Multimodal Machine Learning Approaches in Healthcare". In: Information Fusion 114 (Feb. 1, 2025), p. 102690. ISSN: 1566-2535. DOI: 10.1016/j.inffus .2024.102690. URL: https://www. sciencedirect.com/science/article/pii/S1566253524004688.

[44] Sören Richard Stahlschmidt, Benjamin Ulfenborg, and Jane Synnergren. "Multimodal Deep Learning for Biomedical Data Fusion: A Review". In: Briefings in Bioinformatics 23.2 (Mar. 1, 2022), bbab569. ISSN: 1477-4054. DOI: 10.1093/bib/bbab569. URL: https://doi.org/10.1093/bib/bbab569.

[45] Qiao Xiao et al. “Deep Learning-Based ECG Arrhythmia Classification: A Systematic Review". In: Applied Sciences 13.8 (Jan. 2023), p. 4964. ISSN: 2076- 3417. DOI: 10.3390/app13084964. URL: https://www.mdpi.com/2076- 3417/13/8/4964.

[46] Awni Y. Hannun et al. “Cardiologist-Level Arrhythmia Detection and Classification in Ambulatory Electrocardiograms Using a Deep Neural Network". In: Nature Medicine 25.1 (Jan. 2019), pp. 65–69. ISSN: 1546-170X. DOI: 10.1038/ s41591-018-0268-3. URL: https://www.nature.com/articles/s41591- 018-0268-3.

[47] Nils Strodthoff et al. “Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL". In: IEEE Journal of Biomedical and Health Informatics 25.5 (May 2021), pp. 1519–1528. ISSN: 2168-2208. DOI: 10.1109/JBHI . 2020.3022989. URL: https://ieeexplore.ieee.org/abstract/document/ 9190034.

[48] Jaehak Yu et al. “AI-Based Stroke Disease Prediction System Using ECG and PPG Bio-Signals". In: IEEE Access 10 (2022), pp. 43623–43638. ISSN: 2169- 3536. DOI: 10 .1109/ACCESS.2022.3169284. URL: https://ieeexplore. ieee.org/abstract/document/9761215/keywords.

[49] Chamod Rathnayake et al. “Physiological Monitoring Applications of Wearable Multimodal Fusion Systems Based on ECG and PPG: A Comprehensive Review". In: Sensors 26.11 (Jan. 2026), p. 3477. ISSN: 1424-8220. DOI: 10.3390/s26113477.URL: https://www.mdpi.com/1424-8220/26/11/3477.

[50] Morton F. Arnsdorf. “Electrocardiogram in Hyperkalemia: Electrocardiographic Pattern of Anteroseptal Myocardial Infarction Mimicked by Hyperkalemia-Induced Disturbance of Impulse Conduction". In: Archives of Internal Medicine 136.10 (Oct. 1, 1976), pp. 1161–1163. ISSN: 0003-9926. DOI: 10.1001/archinte. 1976.03630100073019. URL: https://doi.org/10.1001/archinte.1976. 03630100073019.

[51] Philip O. Ettinger, Timothy J. Regan, and Henry A. Oldewurtel. "Hyperkalemia, Cardiac Conduction, and the Electrocardiogram: A Review". In: American Heart Journal 88.3 (Sept. 1, 1974), pp. 360–371. ISsN: 0002-8703. DOI: 10.1016/0002-8703(74)90473-6. URL: https://www.sciencedirect. com/science/article/pii/0002870374904736.

[52] Navid Teymouri et al. “ECG Frequency Changes in Potassium Disorders: A Narrative Review". In: American Journal of Cardiovascular Disease 12.3 (June 15, 2022), pp. 112–124. ISSN: 2160-200X. PMID: 35873184. URL: https: //pmc.ncbi.nlm.nih.gov/articles/PMC9301030/.

[53] Alistair Johnson et al. “MIMIC-IV". In: PhysioNet (Oct. 2024). Version 3.1. DOI: 10.13026/kpb9-mt58. URL: https://doi.org/10.13026/kpb9-mt58.

[54] Alistair E. W. Johnson et al. “MIMIC-IV, a Freely Accessible Electronic Health Record Dataset". In: Scientific Data 10.1 (Jan. 3, 2023), p. 1. ISSN:

2052-4463. DOI: 10.1038/s41597-022-01899-x. URL: https://www.nature. com/articles/s41597-022-01899-x.

[55] Tom Pollard et al. “"PhysioNet as a Global Platform for Biomedical Research". In: Nature Health 1.8 (Mar. 26, 2026), pp. 792–795. ISSN: 3005-0693. DO1: 10.1038/s44360-026-00096-z. URL: https://www.nature.com/articles/ s44360-026-00096-z.

[56] Brian Gow et al. MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset. Version 1.0. PhysioNet. DOI: 10 . 13026 / 4NQG- SB35. URL: https : // physionet.org/content/mimic-iv-ecg/1.0/.

[57] Alistair Johnson et al. MIMIC-IV-ED. Version 2.2. PhysioNet. DOI: 10 .13026/ 5NTK-KM72. URL: https://physionet.org/content/mimic-iv-ed/2.2/.

[58] MIMIC-IV. MIMIC. URL: https://mimic.mit.edu/docs/iv/.

[59] Juan Miguel López Alcaraz et al. A Multimodal Deep Learning Framework for Predicting ICU Deterioration: Integrating ECG Waveforms with Clinical Data and Clinician Benchmarking. Jan. 10, 2026. DO1: 10 .48550/arXiv.2601 . 06645. arXiv: 2601.06645 [eess.SP]. URL: http://arxiv.org/abs/2601. 06645. Pre-published.

[60] Skipper Seabold and Josef Perktold. “"statsmodels: Econometric and statistical modeling with python". In: 9th Python in Science Conference. 2010.

[61] F. Pedregosa et al. "Scikit-learn: Machine Learning in Python". In: Journal of Machine Learning Research 12 (2011), pp. 2825–2830.

[62] Kanako Teramoto et al. “Longitudinal NT-proBNP: Associations With Echocardiographic Changes and Outcomes in Heart Failure". In: Journal of the American Heart Association 13.9 (May 2024), e032254. DOI: 10.1161/JAHA.123. 032254. URL: https://www.ahajournals.org/doi/ful1/10.1161/JAHA. 123.032254.

[63] Christine Mages et al. “Electrical Ventricular Remodeling in Dilated Cardiomyopathy". en. In: Cells 10.10 (Oct. 2021), p. 2767. ISSN: 2073-4409. DOI: 10.3390/cel1s10102767. URL: https://www.mdpi.com/2073-4409/10/10/ 2767.

[64] Vinzenz Hombach. “Electrocardiography of the Failing Heart". English. In: Cardiology Clinics 24.3 (Aug. 2006), pp. 413–426. ISSN: 0733-8651, 1558-2264.

DOI: 10.1016/j.ccl.2006.03.002. URL: https://www.cardiology. theclinics.com/article/S0733-8651(06)00011-7/abstract.

[65] Raymond J. van Wijk et al. Acutelines: a Dutch emergency department-based data-biobank for multimodal, longitudinal acute care research. en. ISSN: 3067- 2007 Pages: 2026.07.30.26359325. Aug. 2026. DO1: 10.64898/2026.07.30. 26359325. URL: https://www.medrxiv.org/content/10.64898/2026.07. 30.26359325v1.

[66] Raymond J. Van Wijk et al. ECG Spectrogram-Based Deep Learning Model to Predict Deterioration of Patients with Early Sepsis at the Emergency Department: A Study from the Acutelines Data- and Biobank. Mar. 27, 2026. DO1: 10.64898/2026.03.26.26349371. URL: http://medrxiv.org/1ookup/doi/ 10.64898/2026.03.26.26349371. Pre-published.

[67] Juan Miguel Lopez Alcaraz and Nils Strodthoff. “Explaining Time Series Classification Predictions via Causal Attributions". In: 2025 IEEE 37th International Conference on Tools with Artificial Intelligence (ICTAI). ISSN: 2375- 0197. Nov. 2025, pp. 42–51. DOI: 10.1109/ICTAI66417.2025.00014. URL: https://ieeexplore.ieee.org/abstract/document/11272614.

[68] Paul A. Heidenreich et al. “2022 AHA/ACC/HFSA Guideline for the Management of Heart Failure: A Report of the American College of Cardiology/American Heart Association Joint Committee on Clinical Practice Guidelines". en. In: Circulation 145.18 (May 2022). ISSN: 0009-7322, 1524-4539. DOI: 10.1161/CIR.0000000000001063. URL: https://www.ahajournals.org/ doi/10.1161/CIR.0000000000001063.

[69] Tadej Battelino et al. “Clinical Targets for Continuous Glucose Monitoring Data Interpretation: Recommendations From the International Consensus on Time in Range". In: Diabetes Care 42.8 (June 2019), pp. 1593–1603. ISSN: 0149-5992. DOI: 10.2337/dci19-0028. URL: https://doi.org/10.2337/ dci19-0028.

[70] Michael J Kim, Christina Valerio, and Glynnis K Knobloch. “Potassium Disorders: Hypokalemia and Hyperkalemia". en. In: ().