Title:Application of Artificial Intelligence in the Management of Climate Change-Related Health Issues in Africa

Authors:

Peter Kokol 1,2\*, peter.kokol@um.si

Martha Kachweka3, mkachwek@andrew.cmu.edu

Helena Blažun Vošner 1,4, Helena.blazun@zd-mb.si

Samuel Chol Buol3, sbuol@andrew.cmu.edu

Jernej Završnik1,4, jernej.zavrsnik@zd-mb.si

1,4Bojan Žlahtič 2 ,bojan.zlahtic@um.si

Grega Žlahtič2, grega.zlahtic1@um.si

## Institutions:

1 Community Healthcare Center Dr. Adolf Drolc Maribor, 2000 Maribor, Slovenia;

2 Faculty of Electrical Engineering and Computer Sciences, University of Maribor, 2000 Maribor, Slovenia

3 CMU- Africa, Bldg Plot No A8, 3594+2RG Kigali Special Economic, Regional ICT Center of Excellence, Kigali, Ruanda

4 Alma Mater Europaea University, ECM Maribor, 2000 Maribor, Slovenia

\* Correspondence: Peter Kokol, peter.kokol@um.si

Application of Artificial Intelligence in the Management of Climate Change-Related Health Issues in Africa

Background: Climate change poses an existential threat to global public health, with Africa bearing a disproportionate burden of climate-sensitive health emergencies. While Artificial Intelligence (AI) and machine learning offer unprecedented capabilities for predictive modeling and early warning systems, their application within African public health frameworks remains under-documented and constrained. This study maps thematic divergences between global and African research landscapes to identify gaps in AI deployment for climate-health management.

Methods: Guided by a Synthetic Knowledge Synthesis approach, we conducted a scoping review and bibliometric analysis of Scopus-indexed literature (through October 2025). We utilized VOSviewer for bibliometric mapping, co-occurrence network analysis, and thematic clustering to structurally compare the Global Literature Corpus (GLC) and the African Literature Corpus (ALC).

Results: Network analysis reveals stark divergences in priorities. The GLC emphasizes complex, predictive planetary health applications, including air pollution mortality tracking and environmental modeling. Conversely, the ALC features localized, reactive solutions, primarily utilizing supervised machine learning for immediate infectious disease triage (e.g., HIV, malaria) and food security. The analysis identifies profound structural barriers in the ALC—notably a "compute deficit," fragmented data silos, and energy instability—precluding the scaling of advanced AI climate-health models.

Conclusions: A critical gap exists between global AI innovations and African public health realities. To build genuine resilience, international funding must pivot from siloed pilot projects toward investing in foundational digital infrastructure and localized, predictive AI frameworks tailored to the African context.

Keywords: artificial intelligence; climate change; public health; health issues; Africa

## o What is known about this subject:

Global climate change is rapidly exacerbating public health crises, particularly in Africa where the burden of climate-sensitive infectious and non-communicable diseases is disproportionately high.

Artificial Intelligence and machine learning are highly effective tools for planetary health, capable of predicting disease outbreaks and optimizing resource allocation.

Advanced AI deployment in healthcare is currently concentrated in the Global North, trained on datasets that often exclude African populations and ecosystems.

## o What this study adds:

It maps a deep thematic divergence: global research focuses on predictive environmental modeling, while African research is constrained to localized, reactive infectious disease triage.

It identifies specific structural barriers—a localized "compute deficit," data silos, and energy instability—preventing the deployment of advanced AI public health frameworks in Africa.

It outlines high-impact, contextually viable AI implementation strategies, such as AI-driven acoustic water leak detection for cholera prevention and predictive solar microgrid management

## BACKGROUND

The convergence of global climate change and health issues presents a rapidly escalating threat [1– 6], demanding sophisticated predictive and adaptive strategies. This crisis is profoundly acute in Africa, where severe public health vulnerabilities are exposed and magnified by environmental shifts. The epidemiological reality is stark: Africa currently carries 95% of the global malaria burden, and climateinduced flooding directly spikes cholera outbreaks and rapidly expands vector habitats across the continent. Addressing these critical, compounding challenges requires moving healthcare systems from a reactive to a proactive approach, a transition where Artificial Intelligence (AI) is emerging as a transformative tool [7–11].

The health burdens discussed in this paper span two distinct disease categories that respond differently to climate change and require fundamentally diferent AI approaches. Communicable diseases, including vector-borne conditions such as malaria and dengue fever, and waterborne diseases such as cholera, are directly linked to shifts in temperature, rainfall, and ecosystem change that alter transmission pathways. Conversely, non-communicable diseases (NCDs), including cardiovascular disease and mental health conditions, are driven by indirect climate pathways such as worsening air quality, chronic heat stress, population displacement, and food insecurity. This distinction maters for AI application design: predictive models for vector-borne diseases rely on climate, mosquito habitat, and mobility data, while NCD-focused tools must integrate environmental exposure data (the “exposome”) with population health records. Both categories impose a growing burden on African health systems. Recognising this distinction is essential for designing AI tools that are contextually appropriate; a satellite-based mosquito habitat model is of litle use for addressing cardiovascular risk, and an exposome pipeline cannot predict a cholera outbreak. Recognising this distinction is essential for designing AI tools that are contextually appropriate; a satellite-based mosquito habitat model is of little use for addressing cardiovascular risk, and an exposome pipeline cannot predict a cholera outbreak (see Figure 1).

AI's core utility lies in its capacity to process vast, high-dimensional datasets that conventional epidemiological models struggle to integrate [12]. Climate change is influencing the occurrence and distribution of diseases like malaria, dengue fever, and cholera by afecting vector habitats, water quality, and extreme weather events. AI can analyse complex variables, including temperature, precipitation paterns, humidity, land-use change, and human movement [13]. This integration allows AI models to outperform traditional statistical approaches by uncovering non-linear relationships and hidden patterns, and forecasting the spatial and temporal spread of outbreaks with higher accuracy [14].

Beyond the infectious disease spectrum, AI supports broader public health systems resilience by modelling the potential impacts of climate hazards like heatwaves and air pollution [15]. AI-driven geospatial analysis can identify specific vulnerable populations and hotspots disproportionately afected by environmental degradation, guiding the strategic allocation of scarce resources and targeted public health interventions [16]. Furthermore, AI and digital health contribute to climate mitigation in the energy sector—e.g., optimizing production and use renewable and clean energy—which indirectly improves public health by reducing reliance on fossil fuels and mitigating the associated air pollution risks [17–19].

According to above AI and digital health have emerged as transformational tools in planetary health. Globally, AI is being rapidly deployed to model complex environmental-health interactions, forecast outbreak trajectories, and optimize resource allocation during climate emergencies. However, the benefits of these digital health innovations are not equitably distributed. The development and deployment of advanced AI models are predominantly concentrated in the Global North, heavily trained on datasets that do not reflect African populations, ecosystems, or health systems.

Consequently, a "digital divide" threatens to widen the existing health disparities between highincome countries and the African continent. While global research pushes the boundaries of predictive environmental modeling and personal exposome tracking, the extent to which these technologies are being adapted for or researched within African contexts remains poorly documented. Furthermore, structural barriers—ranging from a lack of high-performance computing infrastructure (the "compute deficit") to data fragmentation and energy instability—complicate the translation of global AI innovations into localized African public health frameworks.

To address this critical knowledge gap, this study employs a bibliometric and scoping review approach to map the intersection of AI, climate change, and health. Specifically, we aim to systematically compare the Global Literature Corpus (GLC) with the African Literature Corpus (ALC) to identify thematic divergences, chart current research hotspots, and highlight structural barriers. By visualizing the landscape of current research, this review provides a strategic roadmap for researchers, policymakers, and international funding bodies to build equitable, AI-driven climate-health resilience in Africa.

## METHODOLOGY

Africa bears the heaviest, most disproportionate burden of the global climate crisis, manifesting through intensified phenomena like erratic precipitation, devastating floods, and chronic droughts [20]. Although the continent contributes negligibly to worldwide greenhouse gas output, its deep dependence on rain-fed agricultural systems and existing infrastructure deficits leave millions susceptible to climate related health issues or hazards [21]. Artificial Intelligence is rapidly proving to be a critical technological frontier, ofering powerful, data-driven instruments for both climate adaptation and mitigation , and represents a significant, largely under-developed asset for building resilience across the African nations [18,22–24].

This study employed Synthetic Knowledge Synthesis (SKS) to discern diferences in the research literature concerned with employment of artificial intelligence in the management of climate changerelated health issues. SKS methodology, is a three-pronged analytical framework developed by Kokol to provide a comprehensive, semi-automated overview of the research landscape by analysing the whole corpora of publications and mostly biased and fragmented subsets [25]. SKS has already been successfully used in health care related research [26–31]. Because the SKS methodology analyzes the entire retrieved bibliographic corpus computationally, traditional PRISMA screening and eligibility exclusions of individual articles were not performed, eliminating manual selection bias. The PISMA diagram is shown in Figure 2.

The process begins with the extraction of primary bibliographic and meta data from the Scopus database, selected for its superior coverage of non-English literature, and robust metadata compared to alternative repositories. The SKS approach involves the triangulation of bibliometric mapping via VOSviewer to visualize the intellectual structure and thematic clusters of the field; and qualitative content analysis [32] to synthesize core findings and identify emerging research frontiers. By leveraging Scopus’s extensive reach and SKS’s scalable data processing, this methodology minimizes selection bias and eficiently manages "big data" corpora, facilitating an objective, reproducible, and holistic assessment of current scientific developments and future trajectories within the specified domain.

The analysis was executed on two distinct corpora:

• Global Literature Corpus (GLC): Publications concerned with the specified subject areas worldwide.

• African Literature Corpus (ALC): Publications specifically related to research within the African context.

Synthetic Knowledge Synthesis was applied independently in both corpora. Subsequently, a comparative analysis of the resulting SKS outputs was conducted. The identified disparities in research scope between the global and African literature were utilized to pinpoint potential research avenues or approaches that could be introduced and implemented efectively within the African environment and context.

The corpora were formed on 29th of April 2026 using the following two search strings, first for the GLC and the second for the ALC:

TITLE-ABS-KEY(( "artificial intelligence" OR "machine learning" OR "deep learning" OR "intelligent system" OR "support vector machine" OR ( "decision tree" AND ( induction OR heuristic ) ) OR "random forest" OR "Markov decision process" OR "hidden Markov model" OR "fuzzy logic" OR "knearest neighbour" OR "naive Bayes" OR "Bayesian learning" OR "artificial neural network" OR "convolution neural network" OR "recurrent neural network" OR "generative adversarial network" OR "deep belief network" OR "perceptron" OR {natural language processing} OR {natural language understanding} OR {general language model} ) and "climate change\*" and health)

TITLE-ABS-KEY(( "artificial intelligence" OR "machine learning" OR "deep learning" OR "intelligent system" OR "support vector machine" OR ( "decision tree" AND ( induction OR heuristic ) ) OR "random forest" OR "Markov decision process" OR "hidden Markov model" OR "fuzzy logic" OR "knearest neighbour" OR "naive Bayes" OR "Bayesian learning" OR "artificial neural network" OR "convolution neural network" OR "recurrent neural network" OR "generative adversarial network" OR "deep belief network" OR "perceptron" OR {natural language processing} OR {natural language understanding} OR {general language model} ) and "climate change\*" and health) AND TITLE-ABS-KEY(africa OR algeria OR angola OR benin OR botswana OR "Burkina Faso" OR burundi OR cameroon OR "Cape Verde" OR "Central African Republic" OR chad OR comoros OR congo OR "cote d\*" OR "democratic republic of congo" OR djibouti OR egypt OR equator OR guinea OR eritrea OR swaziland OR ethiopia OR gabon OR gambia OR ghana OR guinea OR guinea-bissau OR kenya OR lesotho OR liberia OR libya OR madagascar OR malawi OR mali OR mauritania OR mauritius OR mayote OR morocco OR mozambique OR namibia OR niger OR nigeria OR "réunion for rwanda" OR "Saint Helena" OR "São Tomé" OR "príncipe for senegal" OR seychelles OR "Sierra Leone" OR somalia OR "South Africa" OR "South Sudan" OR sudan OR tanzania OR togo OR tunisia OR uganda OR zambia OR zimbabwe)

Building upon the preliminary SKS employment, a subsequent phase of targeted desktop research and analysis was executed to identify high-impact Artificial Intelligence (AI) deployment opportunities within the African context that might have been omited by the primary analysis. This secondary analysis specifically addresses the critical intersection of deteriorating infrastructure, pervasive energy poverty, and the persistent burden of both communicable and chronic diseases across Sub-Saharan Africa.

## RESULTS

The resultant network (Figure 3), comprising associations (lines) between author keywords organized into discrete, colour-diferentiated clusters, was subjected to further analysis. This analysis of inter-keyword links within clusters led to the identification of categories and the core research themes, as outlined in Table 1. To beter align these technical themes with practical outcomes, a direct public health application metric has been applied to each cluster.

Repeating the approach above on the African Literature Corpus (ALC) publications resulted in the landscape shown in Figure 4. During the thematic clustering of the ALC, a minor cluster (Light blue, 10 keywords) focusing on the use of large language models to increase productivity in neurosurgery was identified. This cluster represents an unrelated overlap in the literature, resulting from the broad nature of the initial search string. Because it does not relate to climate change-induced public health impacts, it has been excluded from the subsequent thematic mapping in Table 2 to maintain the focus and integrity of the review.

The analysis of inter-keyword links within the remaining clusters led to the identification of categories, core research themes, and direct public health applications tailored to the African context, as listed in Table 2.

## DISCUSSION

## Statement of principal findings

This study highlights a profound thematic and operational divergence between global and African research landscapes regarding the application of Artificial Intelligence (AI) in climate-health management. Our network analysis of the Global Literature Corpus (GLC) reveals a strong emphasis on complex, predictive planetary health applications, such as air pollution mortality tracking, complex environmental modeling, and the integration of personal exposomes with chronic disease outcomes. In stark contrast, the African Literature Corpus (ALC) is characterized by localized, reactive solutions. The ALC predominantly utilizes supervised machine learning for immediate infectious disease triage (such as HIV and malaria) and basic sustainability measures. Ultimately, the deployment of advanced AI across Africa is fundamentally constrained by a localized "compute deficit," fragmented data silos, and energy instability. A more detailed discussion of possible application in Africa is presented bellow;

## Sea Temperature and Oceanography

AI can provide crucial forecasting to protect coastal communities and vital 'Blue Economy' sectors.

Early Warning Systems for Coastal Hazards: Use Machine Learning (ML) models trained on satellite data, buoy measurements, and historical data to predict extreme sea-level events, storm surges, and abnormal current paterns [33,34]. This directly informs disaster preparedness and coastal infrastructure protection in countries like South Africa and those along the East African coast.

Fisheries Management: Apply computer vision and oceanographic data modeling to track fish migration paterns and marine ecosystem health (e.g., predicting coral bleaching events due to rising sea temperatures, as noted in Kenya's Blue Economy). This supports sustainable fishing quotas and protects the livelihoods of coastal communities [35,36].

Climate Change Prediction: Leverage AI to improve the accuracy and high-resolution of local climate models by integrating data from various sparse African weather stations and global models, enabling beter long-term adaptation strategies [37,38].

## Carbon Emission Prediction and Electric Vehicles (EVs)

AI is key to leapfrogging to greener energy systems and mobility.

Grid Optimization for Renewables: Use AI to forecast energy demand and optimize the integration of distributed renewable energy sources (solar, wind) into often-fragile African power grids, improving grid stability and reducing reliance on fossil fuels [39].

Within African urban landscapes, Machine Learning facilitates the spatial optimization of EV charging infrastructure by analyzing population density, trafic dynamics, and grid resilience. This predictive modeling identifies strategic site locations, mitigating unique infrastructural botlenecks to accelerate the continental transition toward sustainable, electrified mobility systems through evidence-based resource allocation. [40,41].

Aligning with Agenda 2063, AI-driven auditing enables real-time monitoring and predictive analytics for African industrial and transport emissions. By providing high-fidelity data, these tools empower policymakers to establish rigorous regulatory frameworks and enforce reduction targets. This technological integration is essential for harmonizing the continent’s industrial expansion with sustainable, climate-resilient development mandates [42].

## Health and Socio-Economic Impact

While full-scale AI employment is challenged by infrastructure, AI can augment limited human resources.

Teleradiology/Pathology: AI-powered computer vision tools can automatically analyse Xrays, blood smears (e.g., for malaria), and pathology slides with high accuracy. This is a form of robotization that reduces the workload on scarce specialists and makes expert diagnosis accessible in remote areas via digital health platforms [43,44].

Surgical Training Simulators: Use augmented and virtual reality with AI feedback to provide low-cost, high-quality surgical training to medical professionals without needing complex physical labs [44]

Advanced AI can be used in Poverty and Malaria management. It ofers predictive and prescriptive power for two of Africa's most persistent challenges.

Poverty Targeting: Use satellite imagery (night-lights, roof materials, road networks) and mobile data to estimate poverty levels in granular detail. This advanced AI targeting ensures social welfare programs, food aid, and infrastructure development are directed precisely to the most vulnerable communities [45,46].

Malaria, HIV, Tuberculosis, Chronic Obstructive Pulmonary Diseases and Cholera Outbreak Prediction: Integrate diverse data—climate (temperature, rainfall), mosquito breeding site detection via drones/images, and mobility data—into Deep Learning models to predict diseases outbreaks weeks in advance [9,47], enabling pre-emptive resource deployment (nets, medication, spraying) mitigating the immense socio-economic burden of outbreaks while enhancing the cost-efectiveness of public health expenditures across African sovereign budgets.

## Exposome and Specific Disease Associations (Cardiovascular, Mental)

AI is necessary to handle the complexity of environmental health data.

Personalized Risk Assessment: AI can integrate the exposome (all non-genetic environmental factors like air and water pollution, diet, stress) with local genomic data to identify specific environmental risk factors for non-communicable diseases [48] like Cardiovascular Disease (CVD) [48], which is rising in Africa.

Mental Health Tele-triage: Employ Natural Language Processing (NLP) to analyse speech or text (where available in local languages) from digital health consultations to identify emotional distress and mental health risks, guiding users to appropriate (and often scarce) counselling resources [49].

## Digitalization and Disaster Response

AI can transform disaster management from reactive to anticipatory.

Flash Flood and Drought Forecasting: Utilize ML to rapidly process satellite, radar, and social media data to forecast localized flooding or accelerating drought conditions, providing early warnings tailored to specific communities [50,51].

Damage Assessment: After a disaster (e.g., earthquake, cyclone), AI can rapidly analyse drone and satellite images to automatically map damaged infrastructure, prioritizing aid and rescue routes [52,53].

## Digital Health Innovation

This theme is the foundational layer for AI adoption in healthcare.

Interoperable Data Systems: Research into AI-driven tools to harmonize and link disparate health records (often paper-based or on separate digital systems) across clinics and regions, creating the unified, high-quality data sets needed to train local AI models efectively [54,55]

Telemedicine Optimization: Use AI to intelligently match patients to the right virtual clinician or specialist, optimizing the limited time of medical experts [56,57].

However, implementing these foundational digital health innovations requires active engagement with the heterogeneity of infrastructure across the continent. Africa is not a monolith; the readiness to adopt and sustain AI-driven health tools varies enormously between countries. Rwanda, for instance, has a relatively mature digital health ecosystem, including drone-based medical supply delivery through Zipline, a national electronic medical records rollout coordinated through the Rwanda Biomedical Centre, and widespread DHIS2 adoption. Kenya has pioneered mHealth innovations and mobileintegrated health payment systems. Ethiopia operates one of the largest community health worker networks in the world, with structured digital data collection at scale. Ghana has similarly deployed drone delivery for vaccines and blood products. By contrast, many landlocked or conflict-afected countries face acute constraints in electricity access, mobile internet penetration, and local AI expertise that would fundamentally limit the applicability of the tools proposed here. Therefore, future iterations of this work and subsequent policy frameworks must consult country-level digital health readiness data to stratify recommendations by implementation context, rather than treating the continent as a single deployment environment.

## Sentiment Analysis in Extreme Health, Food Security, and Wildfires

This provides real-time, ground-level intelligence on emerging crises.

Food Insecurity Monitoring: Apply Sentiment Analysis and Topic Modeling to social media, local news articles, and citizen reporting (in local languages) to detect early conversations about rising food prices, scarcity, or distress, serving as a leading indicator of food insecurity before oficial data is available [58,59].

• Public Health Crisis Monitoring: Analyse online chater to gauge public fear, trust in health authorities, or rumours during disease outbreaks, allowing health agencies to deploy targeted communication campaigns to counter misinformation and improve compliance [60,61].

Additional opportunities identified by secondary analysis

A primary lacuna identified in the initial analysis is the stabilization of water supply systems. In nations such as South Africa and Kenya, "non-revenue water"—water lost due to infrastructure decay, leaks, and unauthorized extraction—frequently reaches 40% [62]. AI-driven acoustic sensors and pressure-management algorithms ofer a transformative solution by identifying the unique acoustic "signatures" of pipe bursts in real-time, allowing for rapid containment and repair [63]. Beyond economic recovery, such interventions are critical for disease prevention; stabilizing water pressure prevents the backflow of contaminated groundwater, a primary precursor to cholera epidemics. By integrating AI with Long Short-Term Memory (LSTM) networks that analyze rainfall anomalies and sanitation metrics, health agencies in Nigeria and Malawi can now forecast cholera outbreaks with over 80% accuracy, enabling the pre-emptive deployment of oral rehydration salts and vaccines [64]

The "silent" infrastructure of energy security remains a fundamental barrier to health resilience. Approximately 600 million individuals in Sub-Saharan Africa lack grid access, compromising vaccine cold-chain integrity and hospital operations [65,66] AI is now being leveraged for geospatial site selection and load forecasting to optimize decentralized renewable energy (DRE) systems. In Kenya and Uganda, predictive algorithms manage "pay-as-you-go" solar microgrids, ensuring that rural clinics maintain continuous power for life-saving diagnostics while managing credit risk for unbanked populations [67,68].

Furthermore, the management of high-burden diseases must be prioritized through localized AI training. While global models focus on chronic "exposome" data, African initiatives like HealthPulse AI and Ubenytics utilize computer vision on smartphones to interpret malaria blood smears with 98.5% accuracy, outperforming manual technician reviews in rural Africa [69]. Similarly, deep learning algorithms like CAD4TB are significantly reducing the diagnostic turnaround for Tuberculosis (TB) in South Africa and Ghana, while emerging pharmacometric AI models are being used to tailor dosing for HIV/TB co-infections to the specific genetic diversity of African populations . Finally, the rising prevalence of Chronic Obstructive Pulmonary Disease (COPD)—driven by household air pollution from biomass cooking fires—can be mitigated through AI-integrated air quality monitors that provide real-time health alerts to vulnerable communities [70].

## Strengths and weaknesses of the study

A primary strength of this study is the deployment of the Synthetic Knowledge Synthesis (SKS) framework to evaluate the entire corpus of relevant literature in Scopus. By leveraging comprehensive knowledge synthesis rather than fragmented evaluations of diferent review types, this approach minimizes selection bias and efectively manages large bibliographic datasets. Furthermore, the dual-corpus methodology allows for an objective, reproducible visualization of the specific digital divides impacting public health. A weakness of this study is its reliance exclusively on English-language publications indexed in major databases, which may inadvertently omit localized, grassroots AI implementations documented in regional or non-English repositories.

## Strengths and weaknesses in relation to other studies

While many existing studies outline the theoretical potential of AI in healthcare, this review directly interrogates the infrastructural realities of implementing these technologies in resource-constrained setings. Other bibliometric analyses frequently treat the global research landscape as a monolith. This study specifically isolates the African context, providing a highly granular assessment of systemic vulnerabilities such as unreliable early warning systems and energy poverty.

## Discussion of important diferences in results

The most critical diference identified in our results is the gap between predictive environmental capabilities in the Global North and the reactive application of AI in Africa. While global models focus heavily on chronic non-communicable diseases driven by indirect climate pathways, African applications are necessarily focused on immediate communicable disease burdens and basic food security. Furthermore, the analysis uncovers that the barrier to advanced AI adoption in Africa is not a lack of conceptual understanding, but rather structural deficits, including the lack of high-performance computing infrastructure and secure data harmonization systems.

## CONCLUSION

This study employed Synthetic Knowledge Synthesis to analyse the research landscape concerning AI use in climate change and health, successfully identifying a significant disparity in scope and thematic focus between global and African research scope. Globally, AI research is diverse, encompassing advanced applications in precision agriculture, air pollution modelling, sustainable development, and the 'exposome' of non-communicable diseases. In contrast, the African research focus, while vital, is currently narrower, concentrating on the use of supervised machine learning for specific diseases (e.g., HIV, malaria) and applications for basic sustainability and poverty reduction.

The comparative analysis reveals that advanced AI methodologies remain under-leveraged in Africa despite its disproportionate climate-health burden. This disparity arises from systemic infrastructural constraints—specifically energy instability, limited compute access, and data fragmentation—rather than a lack of expertise. Resolving these structural barriers is critical to de-risking AI investment and unlocking localized, tech-driven solutions that address the continent’s unique environmental, infrastructural, and fiscal challenges. Addressing these barriers through targeted financial instruments, such as Green Bonds and Public-Private Partnerships, is critical to de-risking investment and unlocking localized AI-driven solutions that harmonize environmental resilience with fiscal sustainability. The identified prospective AI application themes—including AI-driven forecasting for coastal hazards, carbon emission prediction and EV infrastructure planning, teleradiology/pathology as a tool for human health resource augmentation, deep learning for poverty targeting and malaria outbreak prediction, and the use of sentiment analysis for food security monitoring—ofer concrete pathways for accelerating climate change adaptation and strengthening public health resilience across Africa.

To efectively harness AI's transformative potential in Africa, future eforts must focus on capacity building and targeted investment. Specifically, there is an urgent need to:

1. Assure international and national funding research that adapts sophisticated global AI models (e.g., for exposome analysis, energy grid optimization, and disaster forecasting) to the unique and local data and infrastructure challenges prevalent in African nations.

2. International and multidisciplinary teams of AI researchers, climatologists, public health experts, and policymakers should ensure that technological advancements are contextually relevant and directly address the Africa's most pressing climate-health vulnerabilities.

In conclusion, the trajectory of AI adoption across the African continent is fundamentally constrained by deep-seated structural and resource-based inequities. The pervasive "compute deficit"— manifested in the scarcity of localized data centers and specialized hardware—not only exacerbates latency but undermines data sovereignty. Furthermore, chronic energy instability and the prohibitive costs of high-speed connectivity sustain a "usage gap" that stifles the scalability of indigenous innovation. Addressing these infrastructure botlenecks is a prerequisite for economic agency; without strategic investment in sovereign compute and localized datasets, the continent risks a digital neocolonialism, remaining a consumer rather than a co-creator of global AI advancement.

## DECLARATIONS

Ethics approval and consent to participate Not applicable.

Consent for publication\*\*

## Not applicable.

Availability of data and materials\*\*

The datasets analyzed during the current study are available from the corresponding author on reasonable request. Or state that they were generated using Scopus queries provided in Section 2.1).

## Competing interests

The authors declare that they have no competing interests.

## Funding

This research received no external funding.

## Authors' contributions

PK, HBV, and BŽ conceptualized and supervised the study. PK, GŽ, and BŽ developed the methodology, analyzed the data, and performed the visualization. JZ and HBV reviewed and edited the manuscript. All authors read and approved the final manuscript.

## Acknowledgements

Not applicable.

## REFERENCES

1. Abbass K, Qasim MZ, Song H, Murshed M, Mahmood H, Younis I. A review of the global climate change impacts, adaptation, and sustainable mitigation measures. Environ Sci Pollut Res. 2022;29(28):42539–42559.

2. Romanello M, Walawender M, Hsu S-C, Moskeland A, Palmeiro-Silva Y, Scamman D, et al. The 2024 report of the Lancet Countdown on health and climate change: facing record-breaking threats from delayed action. Lancet. 2024;404(10465):1847–1896.

3. Saxena V. Water Quality, Air Pollution, and Climate Change: Investigating the Environmental Impacts of Industrialization and Urbanization. Water Air Soil Pollut. 2025;236(2). doi:10.1007/s11270- 024-07702-4.

4. Gajendiran K, Kandasamy S, Narayanan M. Influences of wildfire on the forest ecosystem and climate change: A comprehensive study. Environ Res. 2024;240. doi:10.1016/j.envres.2023.117537.

5. Münzel T, Hahad O, Lelieveld J, Aschner M, Nieuwenhuijsen MJ, Landrigan PJ, et al. Soil and water pollution and cardiovascular disease. Nat Rev Cardiol. 2025;22(2):71–89.

6. Negandhi P, Acharya S, Khan F, Kumar S. Climate change and human health: A medical perspective on emerging challenges and mitigation: A comprehensive review. Multidiscip Rev. 2026;9(4). doi:10.31893/multirev.2026169.

7. Omia E, Bae H, Park E, Kim MS, Baek I, Kabenge I, et al. Remote Sensing in Field Crop Monitoring: A Comprehensive Review of Sensor Systems, Data Analyses and Recent Advances. Remote Sens. 2023;15(2). doi:10.3390/rs15020354.

8. Guo Q, He Z, Wang Z. Monthly climate prediction using deep convolutional neural network and long short-term memory. Sci Rep. 2024;14(1). doi:10.1038/s41598-024-68906-6.

9. Liao H, Lyon CJ, Ying B, Hu T. Climate change, its impact on emerging infectious diseases and new technologies to combat the challenge. Emerg Microbes Infect. 2024;13(1). doi:10.1080/22221751.2024.2356143.

10. Kraemer MUG, Tsui JL-H, Chang SY, Lytras S, Khurana MP, Vanderslott S, et al. Artificial intelligence for modelling infectious disease epidemics. Nature. 2025;638(8051):623–635.

11. Tong X, Liu X, Wu P, Hang Y, Zhang L, Jia B, et al. More frequent precipitation during the maize growth period offsets the yield increasing effect of straw mulching in China: A meta -analysis. Soil Tillage Res. 2026;256. doi:10.1016/j.still.2025.106870.

12. Chen F, Zhang W, Mfarrej MFB, Saleem MH, Khan KA, Ma J, et al. Breathing in danger: Understanding the multifaceted impact of air pollution on health impacts. Ecotoxicol Environ Saf. 2024;280. doi:10.1016/j.ecoenv.2024.116532.

13. Lyimo B. Leveraging Artificial Intelligence to Advance Bioinformatics in Africa: Opportunities, Challenges, and Ethical Considerations in Combating Antimicrobial Resistance. Bioinform Biol Insights. 2026;20:11779322261427123.

14. Odone A, Barbati C, Amadasi S, Schultz T, Resnik DB. Artificial intelligence and infectious diseases: an evidence-driven conceptual framework for research, public health, and clinical practice. The Lancet Infectious Diseases. 2026;26(3):e152–e167.

15. Golden CD, Childs ML, Mudele OE, Andriamizarasoa FA, Bouley TA, De Nicola G, et al. Climatesmart public health for global health resilience. Lancet Planet Health. 2025;9(8). doi:10.1016/j.lanplh.2025.101293.

16. El Maachi S, Saadane R, Wahbi M, Chehri A, Badaoui A. Fine-Grained Land Segmentation for Climate Change Impact Assessment: Leveraging the DeepGlobe Dataset with Advanced AI-driven Geospatial Analysis Techniques. ECTI Trans Comp Inf Technol. 2023;17(4):564–576.

17. Shaamala A, Yigitcanlar T, Nili A, Nyandega D. Algorithmic green infrastructure optimisation: Review of artificial intelligence driven approaches for tackling climate change. Sustainable Cities Soc. 2024;101. doi:10.1016/j.scs.2024.105182.

18. Qoseem IO, Okesanya OJ, Olaleke NO, Ukoaka BM, Amisu BO, Ogaya JB, et al. Digital health and health equity: How digital health can address healthcare disparities and improve access to quality care in Africa. Health Prom Perspect. 2024;14(1):3–8.

19. Mijwil MM, Abotaleb M, Ouko KO, Turyasingura B, Sila W, Kayusi F, et al. Adoption of AI and Livestock management strategies for sustainable food security in the face of climate change in sub-

Saharan Africa. In: Mzili T, Arya AK, Pamucar D, Shaheen M, editors. Optimization, Machine Learning, and Fuzzy Logic: Theory, Algorithms, and Applications. 2025. IGI Global: 455–472.

20. Fonjong L, Matose F, Sonnenfeld DA. Climate change in Africa: Impacts, adaptation, and policy responses. Global Environ Change. 2024;89. doi:10.1016/j.gloenvcha.2024.102912.

21. Hounkpatin H, Nieto-Sanchez C, Castellano Pleguezuelo V, Polman K, Marchal B. How are health systems in sub-Saharan Africa adapting to protect human health from climate change threats? A scoping review and case study. Lancet Planet Health. 2024;8:S10.

22. Bedair H, Alghariani MS, Omar E, Anibaba QA, Remon M, Bornman C, et al. Global Warming Status in the African Continent: Sources, Challenges, Policies, and Future Direction. Int J Environ Res. 2023;17(3). doi:10.1007/s41742-023-00534-w.

23. Wanyama J, Bwambale E, Kiraga S, Katimbo A, Nakawuka P, Kabenge I, et al. A systematic review of fourth industrial revolution technologies in smart irrigation: Constraints, opportunities, and future prospects for sub-Saharan Africa. Smart Agric Technol. 2024;7. doi:10.1016/j.atech.2024.100412.

24. Pimenow S, Pimenowa O, Prus P, Niklas A. The Impact of Artificial Intelligence on the Sustainability of Regional Ecosystems: Current Challenges and Future Prospects. Sustainability. 2025;17(11). doi:10.3390/su17114795.

25. Kokol P. Synthetic Knowledge Synthesis in Hospital Libraries. Journal of Hospital Librarianship. 2023;0(0):1–8.

26. Yalcinkaya T, Cinar Yucel S. Mobile learning in nursing education: A bibliometric analysis and visualization. Nurse Education in Practice. 2023;71:103714.

27. Güngör DC, Soybaş M, Orgun F, Özkütük N. Educational games in nursing education: A bibliometric and content analysis. Nurse Education in Practice. 2025;82:104231.

28. Yalcinkaya T, Cinar Yucel S. Bibliometric and content analysis of ChatGPT research in nursing education: The rabbit hole in nursing education. Nurse Education in Practice. 2024;77:103956.

29. Završnik J, Kokol P, Žlahtič B, Blažun Vošner H. Artificial Intelligence and Pediatrics: Synthetic Knowledge Synthesis. Electronics. 2024;13(3):512.

30. Todman LC, Bush A, Hood ASC. ‘Small Data’ for big insights in ecology. Trends in Ecology & Evolution. 2023;38(7):615–622.

31. Wang L, Wang J, Zhang Y, Zhang H. Current perspectives and trends of the research on hypertensive nephropathy: a bibliometric analysis from 2000 to 2023. Renal Failure. 2024;46(1):2310122.

32. Braun V, Clarke V. What can ‘thematic analysis’ offer health and wellbeing researchers? International Journal of Qualitative Studies on Health and Well-being. 2014;9. doi:10.3402/qhw.v9.26152.

33. Lynch M, Harris F, Ierna M, Mahomed O, Henriquez-Mui F, Gebreslasie M, et al. Warning System for Extreme Weather Events, Awareness Technology for Healthcare, Equitable Delivery, and Resilience (WEATHER) Project: A Mixed Methods Research Study Protocol. Clim. 2025;13(8). doi:10.3390/cli13080170.

34. Ian V-K, Tang S-K, Pau G. Assessing the Risk of Extreme Storm Surges from Tropical Cyclones under Climate Change Using Bidirectional Attention-Based LSTM for Improved Prediction. Atmosphere. 2023;14(12). doi:10.3390/atmos14121749.

35. Fernandez Garcia G, Corpetti T, Nevoux M, Beaulaton L, Martignac F. AcousticIA, a deep neural network for multi-species fish detection using multiple models of acoustic cameras. Aquatic Ecol. 2023;57(4):881–893.

36. Li P, Han H, Zhang S, Fang H, Fan W, Zhao F, et al. Reviews on the development of digital intelligent fisheries technology in aquaculture. Aquac Int. 2025;33(3). doi:10.1007/s10499 -025-01870-8.

37. Rahimi N, Maddah MA, Akhoond-Ali AM, Bahrami M. Assessing variations in meteorological parameters using global climate model (GCM) outputs and artificial neural networks. J Water Clim Change. 2025;16(2):531–546.

38. Boonnam N, Udomchaipitak T, Puttinaovarat S, Chaichana T, Boonjing V, Muangprathub J. Coral Reef Bleaching under Climate Change: Prediction Modeling and Machine Learning. Sustainability. 2022;14(10). doi:10.3390/su14106161.

39. Jang M-J, Oh E. Deep-Reinforcement-Learning-Based Vehicle-to-Grid Operation Strategies for Managing Solar Power Generation Forecast Errors. Sustainability. 2024;16(9). doi:10.3390/su16093851.

40. Yaghoubi E, Yaghoubi E, Khamees A, Razmi D, Lu T. A systematic review and meta-analysis of machine learning, deep learning, and ensemble learning approaches in predicting EV charging behavior. Eng Appl Artif Intell. 2024;135. doi:10.1016/j.engappai.2024.108789.

41. Zhao AP, Li S, Li Z, Wang Z, Fei X, Hu Z, et al. Electric Vehicle Charging Planning: A Complex Systems Perspective. IEEE Trans Smart Grid. 2025;16(1):754–772.

42. Yamini E, Zarnoush M, Jalilvand M, Zolfaghari SM, Esmaeilion F, Taklifi A, et al. Integration of emerging technologies in next-generation electric vehicles: Evolution, advancements, and regulatory prospects. Result Eng. 2025;25. doi:10.1016/j.rineng.2025.104082.

43. Murali S, Ding H, Adedeji F, Qin C, Obungoloch J, Asllani I, et al. Bringing MRI to low- and middleincome countries: Directions, challenges and potential solutions. NMR Biomed. 2024;37(7). doi:10.1002/nbm.4992.

44. Ahuja S, Zaheer S. Advancements in pathology: Digital transformation, precision medicine, and beyond. J Pathol Inform. 2025;16. doi:10.1016/j.jpi.2024.100408.

45. Pilag Kakeu CB, Miamo Wendji C, Kouhomou CZ, Mapa Kamdoum GC. Can technological innovations contribute to more overcome the issue of poverty reduction in africa? Technol Soc. 2024;76. doi:10.1016/j.techsoc.2024.102463.

46. Mishra S, Satapathy SK, Cho S-B, Mohanty SN, Sah S, Sharma S. Advancing COVID-19 poverty estimation with satellite imagery-based deep learning techniques: a systematic review. Spat Inf Res. 2024;32(5):583–592.

47. Khan O, Ajadi JO, Hossain MP. Predicting malaria outbreak in The Gambia using machine learning techniques. PLoS ONE. 2024;19(5). doi:10.1371/journal.pone.0299386.

48. Cannarozzi AL, Latiano A, Massimino L, Bossa F, Giuliani F, Riva M, et al. Inflammatory bowel disease genomics, transcriptomics, proteomics and metagenomics meet artificial intelligence. United Eur Gastroenterol J. 2024;12(10):1461–1480.

49. Alemu R, Sharew NT, Arsano YY, Ahmed M, Tekola-Ayele F, Mersha TB, et al. Multi-omics approaches for understanding gene-environment interactions in noncommunicable diseases: techniques, translation, and equity issues. Hum Genomics. 2025;19(1). doi:10.1186/s40246-025-00718-9.

50. Nearing G, Cohen D, Dube V, Gauch M, Gilon O, Harrigan S, et al. Global prediction of extreme floods in ungauged watersheds. Nature. 2024;627(8004):559–563.

51. Camps-Valls G, Fernández-Torres M-Á, Cohrs K-H, Höhl A, Castelletti A, Pacal A, et al. Artificial intelligence for modeling and understanding extreme weather and climate events. Nat Commun. 2025;16(1). doi:10.1038/s41467-025-56573-8.

52. Boroujeni SPH, Razi A, Khoshdel S, Afghah F, Coen JL, O’Neill L, et al. A comprehensive survey of research towards AI-enabled unmanned aerial systems in pre-, active-, and post-wildfire management. Inf Fusion. 2024;108. doi:10.1016/j.inffus.2024.102369.

53. Macchiarulo V, Giardina G, Milillo P, Aktas YD, Whitworth MRZ. Integrating post-event very high resolution SAR imagery and machine learning for building-level earthquake damage assessment. Bull Earthquake Engin. 2025;23(12):5021–5047.

54. Abad-Navarro F, Martínez-Costa C. A knowledge graph-based data harmonization framework for secondary data reuse. Comput Methods Programs Biomed. 2024;243. doi:10.1016/j.cmpb.2023.107918.

55. Jayathissa P, Rohatsch L, Sauermann S, Hussein R. OMOP-on-FHIR: Integrating the Clinical Data Through FHIR Bundle to OMOP CDM. In: Stud. Health Technol. Informatics. 2025. IOS Press BV: 667– 671.

56. Ueda D, Walston SL, Fujita S, Fushimi Y, Tsuboyama T, Kamagata K, et al. Climate change and artificial intelligence in healthcare: Review and recommendations towards a sustainable future. Daign Intervent Imaging. 2024;105(11):453–459.

57. Li J-PO, Liu H, Ting DSJ, Jeon S, Chan RVP, Kim JE, et al. Digital technology, tele-medicine and artificial intelligence in ophthalmology: A global perspective. Prog Retinal Eye Res. 2021;82. doi:10.1016/j.preteyeres.2020.100900.

58. Garcia-Garcia G, Trollman H, Parra-López C, Carmona-Torres C, Jagtap S, Luo Y, et al. Digitizing the palate: Exploring opportunities for digital transformation in the food industry. In: Soofastaei A, editor. Advanced Analytics for Industry 4.0: Traditional Industries. 2025. CRC Press: 205–232.

59. Rethika S, Rajaraman M, Kannimuthu S, Arunkumar PM. Sentiment Analysis on Climate Change: An empirical approach. In: Int. Conf. Emerg. Res. Comput. Sci., ICERCS. 2024. Institute of Electrical and Electronics Engineers Inc. doi:10.1109/ICERCS63125.2024.10895467.

60. Gawande MS, Zade N, Kumar P, Gundewar S, Weerarathna IN, Verma P. The role of artificial intelligence in pandemic responses: from epidemiological modeling to vaccine development. Mol Biomed. 2025;6(1). doi:10.1186/s43556-024-00238-3.

61. Wongpituk K, Laosupap K, Butsorn A, Boonsang A, Thammaboribal P, Chankong W, et al. Advancements in Disease Surveillance: The Role of GIS in Global Health Preparedness. Int J Geoinformatics. 2024;20(10):95–108.

62. Campbell AM, Racault M-F, Goult S, Laurenson A. Cholera Risk: A Machine Learning Approach Applied to Essential Climate Variables. International Journal of Environmental Research and Public Health. 2020;17(24):9378.

63. Liu Y, Ma X, Li Y, Tie Y, Zhang Y, Gao J. Water Pipeline Leakage Detection Based on Machine Learning and Wireless Sensor Networks. Sensors. 2019;19(23):5086.

64. Akingbola A, Abiodun A, Ojo O, Jessica OU, Alao UH, Owolabi AO, et al. Cholera Outbreak in Nigeria: History, Review of Socioeconomic and Meteorological Drivers, Diagnostic Challenges, and Artificial Intelligence Integration. Glob Health Epidemiol Genom. 2025;2025:8898076.

65. Frnana FM, Kareem PH. Pathways to Achieving Low Energy-Poverty Problems in Central African Nations with Government Effectiveness, Technology, Natural Resources and Sustainable Economic Growth. Sustainability. 2025;17(3):1007.

66. Moner-Girona M, Kakoulaki G, Falchetta G, Weiss DJ, Taylor N. Achieving universal electrification of rural healthcare facilities in sub-Saharan Africa with decentralized renewable energy technologies. Joule. 2021;5(10):2687–2714.

67. Mhlanga D. Artificial Intelligence in the Industry 4.0, and Its Impact on Poverty, Innovation, Infrastructure Development, and the Sustainable Development Goals: Lessons from Emerging Economies? Sustainability. 2021;13(11):5788.

68. Odhiambo O. Energy Access in Africa: Challenges and Solutions. .

69. Lindblade KA, Ngufor C, Yavo W, Atobatele S, Mpimbaza A, Ssewante N, et al. Evaluating the performance of an artificial intelligence-based electronic reader for malaria rapid diagnostic tests across Benin, Côte d’Ivoire, Nigeria and Uganda. Malar J. 2025;24(1):302.

70. Adeyemo S, Sangotola A, Korosteleva O. Modeling Transmission Dynamics of Tuberculosis–HIV Co-Infection in South Africa. Epidemiologia. 2023;4(4):408–419.

## Legends

Figure 1: Conceptual framework illustrating the distinct pathways through which climate change drivers exacerbate both communicable and non-communica ble disease categories, and their corresponding targeted Artificial Intelligence (AI) applications within public health systems.

Figure 2: PRISMA-adapted flow diagram detailing the Synthetic Knowledge Synthesis (SKS) methodology utilized in this study. The diagram illust rates the identification, screening, and inclusion phases for both the Global Literature Corpus (N=2,016) and the African Literature Corpus (N=274) retrieved from the Scopus database on April 29, 2026, comparing AI deployment in climate-health management.

Figure 3: VOSviewer bibliometric network map displaying the thematic clustering of author keywords within the Global Literature Corpus (GLC). The network is generated from 2,016 Scopus-indexed publications retrieved on April 29, 2026, representing the global research landscape concerning the application of artificial intelligence and machine learning in climate-health management.

Figure 4: VOSviewer bibliometric network map displaying the thematic clustering of author keywords within the African Literature Corpus (ALC). The network is generated from 274 Scopus-indexed publications retrieved on April 29, 2026, representing localized research applications of artificial intelligence in managing climate change-related health issues specifically within the African context.

Table 1: Thematic clusters, core research themes, and direct public health applications identified via Synthetic Knowledge Synthesis ( SKS) within the Global Literature Corpus (GLC) of 2,016 Scopus-indexed publications (up to April 2026) regarding global artificial intelligence deployment in climate change-related health issues.

Table 2: Thematicsc clusters core research themes, and direct localized public health applications identified via Synthetic Knowledge Synthesis (SKS) within the African Literature Corpus (ALC) of 274 Scopus-indexed publications (up to April 2026) regarding artificial intelligence deployment for climate changerelated health management in Africa.

Table 1
<table><tr><td rowspan=1 colspan=1>Cluster colour and number of authorskeywords shown in the research land-scape</td><td rowspan=1 colspan=1>Theme induced using qualitative SKSanalysis</td><td rowspan=1 colspan=1>Derived Direct Public Health Application/Impact</td></tr><tr><td rowspan=1 colspan=1>Blue cluster (23 author keywords)</td><td rowspan=1 colspan=1>Application of image processing andcomputer vision in smart/precisionfarming and agriculture supported byIoT</td><td rowspan=1 colspan=1>Mitigatingmalnutrition and food-insecurity-inducedwasting/stuntinginchildren under 5.</td></tr><tr><td rowspan=1 colspan=1>Dark blue cluster (32 author keywords)</td><td rowspan=1 colspan=1>Air pollution and morbidity and mor-tality</td><td rowspan=1 colspan=1>Reducing cardiovascular and respiratory disease mortality linked to cli-mate-exacerbated air pollution and extreme weather events.</td></tr><tr><td rowspan=1 colspan=1>Red cluster (56 author keywords)</td><td rowspan=1 colspan=1>Remote sensing and google earth en-gine in managing climate change con-sequences on environment and health</td><td rowspan=1 colspan=1>Predicting and mitigating vector-borne disease outbreaks (e.g., malaria)and heat-related illnesses through early environmental surveillance.</td></tr><tr><td rowspan=1 colspan=1>Green cluster (34 author keywords)</td><td rowspan=1 colspan=1>Sustainable development during cli-mate change for better health</td><td rowspan=1 colspan=1>Minimizing waterborne disease risks and improving urban respiratoryhealth outcomes through resilient infrastructure and clean transport pol-icies.</td></tr><tr><td rowspan=1 colspan=1>Yellow cluster (28 items)</td><td rowspan=1 colspan=1>Environment and digital public health</td><td rowspan=1 colspan=1>Preventing non-communicable diseases and managing climate-anxietyand mental health crises via targeted digital health triage.</td></tr><tr><td rowspan=1 colspan=1>Viollet cluster (27 author keywords)</td><td rowspan=1 colspan=1>Digitalization of health care and man-agement of natural disasters</td><td rowspan=1 colspan=1>Enhancing emergency medical response and reducing trauma and mor-tality during climate-induced natural disasters.</td></tr></table>

Table 2

<table><tr><td rowspan=1 colspan=1>Cluster colour andnumber of authorskeywords shown inthe research landscape</td><td rowspan=1 colspan=1>Theme derived using qualitative SKS analysis</td><td rowspan=1 colspan=1>Derived Direct Public Health Application/Impact</td></tr><tr><td rowspan=1 colspan=1>Red cluster (25 key-words)</td><td rowspan=1 colspan=1>Machine learning application in increasing sustainabilityand reducing poverty</td><td rowspan=1 colspan=1>Targeting poverty-alleviation resources to highly vulnerable populations to reduce maltransmission and improve baseline community resilience.</td></tr><tr><td rowspan=1 colspan=1>Dark blue cluster (13keywords)</td><td rowspan=1 colspan=1>Use of AI and satellite surveillance toin health innovation</td><td rowspan=1 colspan=1>Enhancing rural disease surveillance and directing mobile health interventions to remotemate-vulnerable communities.</td></tr><tr><td rowspan=1 colspan=1>Green cluster (22 key-words)</td><td rowspan=1 colspan=1>Application of machine learning for off-grid electrifica-tion, and for the prevention and control of diseases</td><td rowspan=1 colspan=1>Ensuring vaccine cold-chain integrity and powering rural primary healthcare clinics duextreme weather events.</td></tr><tr><td rowspan=1 colspan=1>Yellow cluster (14 key-words)</td><td rowspan=1 colspan=1>Application of deep learning for remote sensing and earthobservation data analysis to increase food quality</td><td rowspan=1 colspan=1>Preventing climate-driven nutritional deficiencies and protecting rural food systems agaescalating drought conditions.</td></tr><tr><td rowspan=1 colspan=1>Viollet cluster (11 au-thor keywords)</td><td rowspan=1 colspan=1>Analysis of earth and social media data for managementof COVID-19 and ecosystems</td><td rowspan=1 colspan=1>Monitoring real-time public health crises and deploying rapid, targeted interventions baon localized social intelligence.</td></tr><tr><td rowspan=1 colspan=1>Orange cluster (9 key-words)</td><td rowspan=1 colspan=1>Modelling of climate changes for increased food produc-tion</td><td rowspan=1 colspan=1>Securing staple crop yields to prevent localized famine, acute malnutrition, and associopportunistic infections.</td></tr></table>