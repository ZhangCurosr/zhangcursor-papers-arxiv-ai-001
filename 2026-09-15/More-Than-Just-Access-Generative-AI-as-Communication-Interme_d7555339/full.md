# More Than Just Access: Generative AI as Communication Intermediary for Blind and Low-Vision Users

Protik Dey University of Texas at San Antonio San Antonio, Texas protik.dey@my.utsa.edu

Mohd Saifuzzaman   
University of Texas at San Antonio San Antonio, Texas   
mohd.saifuzzaman@my.utsa.edu

Taslima Akter University of Texas at San Antonio San Antonio, Texas taslima.akter@utsa.edu

## Abstract

Generative AI (GenAI) tools are increasingly woven into how blind and low-vision (BLV) people communicate, not only with digital information, but with the physical world and with other people. Tools such as ChatGPT, Google Gemini, Be My AI, and Seeing AI translate visual and textual content into accessible form, and are beginning to substitute for interpersonal requests for help, such as asking a family member to read a label or describe a scene. Drawing on semi-structured interviews with 19 BLV participants, we exam ine GenAI as a communication intermediary and how it succeeds and fails as an alternative for reading, describing, and even asking another person for help. We also investigated what BLV users gain and risk when these tools take over that role. We conclude with design and policy implications for GenAI systems that communicate uncertainty honestly, protect information, and support BLV users’ independence rather than substitute for it unsafely.

## CCS Concepts

• Human-centered computing → Accessibility systems and tools; Empirical studies in accessibility; Empirical studies in HCI; • Computing methodologies → Philosophical/theoretical foundations of artificial intelligence.

## Keywords

Generative AI, Blind and Low-Vision, Communication, Trust, Accessibility, Privacy, Human-AI Interaction

## 1 Introduction

Blind and low-vision (BLV) people are increasingly using Generative AI (GenAI) to communicate, for example reading a document, describing a scene, identifying an object, or asking a question that would once have required another person’s help [1]. Tools such as ChatGPT, Be My AI, and Seeing AI translate the visual and textual world into a form BLV users can act on, letting them access information and, increasingly, reach other people through an AI intermediary rather than asking directly. Unlike rule-based or deterministic assistive technologies, GenAI systems produce probabilistic outputs that can be flexible and context-sensitive but also less predictable and sometimes incorrect [9, 10]. For BLV users, who often cannot visually verify what a GenAI system tells them, this unpredictability is not simply a usability flaw, it is a communication problem, since the reliability of what is “said” to them cannot be checked against what is actually there.

This shift touches BLV users’ communication in at least two distinct ways. First, it bears on their epistemic autonomy: because BLV users often cannot independently verify a GenAI response, their capacity for information seeking and independent judgment depends on non-visual cues the system may or may not provide [1, 32]. Second, it bears on their relationships: GenAI is increasingly substituting for communication that once involved another person like asking a family member to read a label, describing a scene to a friend, or calling a sighted volunteer to confirm a document [33]. In other words, swapping a person for GenAI doesn’t just change how the exchange happens, it changes who the BLV user is trusting, what they’re revealing, and who’s responsible if it goes wrong. It also changes the efort and sincerity of the exchange itself. Asking GenAI can lower the social cost of a routine request, but it removes the emotional connection and accountability that came with asking a person [4, 8].

A wide body of work has studied why people come to rely on automated systems at all, pointing to reliability, transparency, accountability, and privacy as central considerations [15, 19, 29, 30]. Most ofthis work, however, assumes a user who can visually inspect outputs, compare information, and independently verify system behavior. This assumption does not hold for BLV users communicating through GenAI. In this paper, we examine GenAI as a communication intermediary for BLV users, a channel that increasingly stands in for reading, describing, and asking another person for help. We ask how BLV users experience this substitution, what shapes their willingness to rely on it over a person, and what is gained and put at risk, for their independent judgment and their relationships, when an AI system takes over a communicative role people used to fill. To answer these questions, we conducted semistructured interviews with 19 BLV users who had experience using a range of GenAI tools. Our findings show that GenAI succeeds as a communicative stand-in unevenly. It extends BLV users’ independence and eases the everyday efort of asking family and friends, but only when the exchange itself is reliable, reachable, and safe to disclose to. BLV users are not passive about this shift, rather they exercise independent judgment about when to rely on GenAI and when to return to a person, reserving human help, and the sincerity and accountability that comes with it.

## 2 Related Work

GenAI tools are changing how BLV users communicate, not just what tools they use, but who they turn to for help and what they must share to get it [1, 8]. Prior work on trust in automated systems points to reliability, transparency, accountability, and privacy as central considerations [15, 19, 23, 29, 30], and framework studies extend this across fairness, explainability, and governance [20, 22, 31, 36]. This is especially relevant for GenAI, whose responses are fluent and confident even when wrong, making errors hard for anyone to catch [18, 34]. Trust in these systems is not fixed but shaped by ongoing experience, not just a single correct or incorrect answer [11, 17, 26]. Most of this work, however, assumes a user who can visually inspect outputs and verify them independently, an assumption that does not hold for BLV users. Accessibility research shows BLV users regularly encounter screen reader incompatibility, unlabeled interfaces, and inaccessible content [6, 14, 35], and recent work on explainability finds that BLV users specifically prefer conversational, back-and-forth exchanges that let them question and verify a response, over a static explanation delivered once [32].

BLV users already rely on GenAI for everyday communicative tasks such as reading documents, understanding images, and answering questions [1, 13, 21, 27], but accessibility breaks down when a system ofers no way to ask it to look again or gives no indication of what it actually looked at [16, 24, 39], and fluent-but wrong responses can go unnoticed and erode trust over time [8, 38]. A smaller line of work looks at something more specific, BLV users turning to GenAI in situations where they would previously have asked a person, and carefully managing what they disclose to it in the process [33, 37, 40]. This mirrors long-standing findings on camera-based assistive tools, where BLV users weigh the same disclosure decision with a human helper or bystander in place of an AI, and where privacy harms such as misrepresentation and unfair treatment shape whether the helper, human or AI, is trusted at all [2–5]. This substitution is not automatic; BLV users still reserve high-stakes tasks for a human they trust, and judge how much to rely on a system by how clearly it communicates its own confidence, using GenAI mainly for the smaller, more frequent requests that used to fall on people close to them [7, 35]. What remains underexamined is how BLV users decide, case by case, when an AI is a good enough stand-in for a person, and what they give up, in accountability, sincerity, and disclosure, when it takes that person’s place.

## 3 Methodology

We conducted semi-structured interviews with 19 BLV participants with prior experience using GenAI tools, including general-purpose systems such as ChatGPT, Google Gemini, Meta AI, Perplexity, Claude, and Microsoft Copilot, as well as GenAI-powered assistive technologies such as Be My AI and Seeing AI. Recruitment was conducted primarily through the National Federation of the Blind, with a smaller number of participants recruited via snowball sampling. Interested individuals completed an online recruitment form, which also served as a screening form to assess eligibility. Eligibility criteria included: (1) residing in the United States, (2) age 18 years or older, (3) identify as blind or low-vision, (4) primarily use a screen reader for accessing digital content (instead of magnification), (5) speak English, and (6) have experience using GenAI tools. Two researchers screened respondents who completed the form and then contacted each eligible participant via email to schedule interviews. From a total of 33 respondents, 19 participants were selected based on their level of experience and frequency of GenAI use.

Most participants were moderate to frequent users of GenAI tools, with seven reporting use several times a day; one participant had comparatively less experience, having begun using ChatGPT approximately six months prior to the interview. All participants used GenAI-powered assistive technologies such as Be My AI or Seeing AI, and several also reported using Aira for visual assistance. To access these GenAI tools, all participants relied on screen readers, including JAWS or NVDA, and several also used mobile accessibility features such as VoiceOver or Siri. Of the 19 participants, 13 (68.4%) identified as female and 6 (31.6%) as male. Participants ranged across age groups, with the largest clusters between 30–39 and 40–49 years old, and educational attainment ranged from high school to doctoral degrees.

The interviews were conducted remotely via Zoom, following approval from our university’s Institutional Review Board (IRB). Participants were provided with an informed consent form and a demographic questionnaire once the interview was scheduled. At the beginning of each session, the interviewer introduced the study goals and interview process and obtained participants’ verbal consent to proceed.

The interview protocol was designed to investigate participants’ experiences using GenAI tools and what shaped their trust in relying on GenAI to communicate with information, with their surroundings, and in place of another person. In the first part of the interview, participants were asked to describe their current interactions with GenAI systems, including the communicative tasks they used these tools for, their usage patterns, and challenges encountered. The second part explored participants’ perceptions of trust and reliance through discussion of specific situations including everyday information-seeking, image and document interpretation, and moments where they chose GenAI over asking a person for help, examining how accessibility, privacy, and reliability shaped their confidence in each case. In the final part, participants were invited to share recommendations for developers and researchers on designing GenAI systems that communicate more accessibly and confidently with BLV users.

The interviews typically lasted between 45 and 60 minutes, with some extending beyond an hour to allow participants to elaborate. All interviews were video-recorded with participants’ permission and later transcribed for analysis. Participants received a US\$30 Amazon gift card as compensation for their participation.

For data analysis, two authors independently engaged in the coding process using a reflexive thematic analysis approach [12]. Inter-rater reliability was not calculated, as such metrics are not considered suitable for interview-based data where codes are not treated as fixed [25, 28]. Instead, the authors met weekly to jointly code an initial set of interviews, discuss emerging codes, resolve diferences, and refine a shared codebook. After reaching thematic saturation, the remaining interviews were divided between the two authors, who continued to meet weekly to review new codes and update shared interpretations, ultimately surfacing themes shaping whether participants trusted GenAI enough to rely on it as a communicative stand-in for a person.

## 4 Findings

Across interviews, participants described GenAI less as a singlepurpose tool and more as a channel through which they accessed the visual world, digital content, and increasingly other people. Whether that channel could be trusted depended on how well it worked, whether participants could reach it at all, and what they had to give up to use it.

## 4.1 Reading the World Aloud: GenAI as a Channel to Visual and Textual Information

The most consistent use case of GenAI was translating visual or textual information into a non-visual form: describing images, reading documents, identifying objects etc. Because participants generally could not independently verify these translations, they treated the channel with calibrated skepticism rather than full acceptance. P11 explained:

“When Igetsome information, andwhen Ireadthrough it, you have to be wise, right? Like, do you want to really rely on the information that’s being provided to you? So, if the information I’m using for my work, I read it, but I also verify and use my own judgment whether to use it for my work or not. Because sometimes, there could be a hallucination.”

Verification was easiest when participants already had background knowledge of the content being communicated to them. P15 described catching an error in a measurement task specifically because they had independent domain knowledge, not because the system flagged uncertainty. This dependence on prior knowledge is a direct constraint on epistemic autonomy: the moments when independent judgment matters most are exactly the moments when a BLV user has the least independent basis on which to exercise it. Also the channel is least trustworthy exactly when there is no other way to cross-check what they are being told, which is often the situation in which they turned to GenAI in the first place.

## 4.2 From Asking a Person to Asking a Machine

Participants described a shift from asking a person to asking GenAI tools, particularly for tasks that once required friends, family mem bers, or bystanders. They explicitly tied this shift with independence and reduced social burden. They are also informed about the pri vacy and related concerns that come with using these tools for day to day tasks. P19 described this trade-of directly: “I guess in the beginning I was more wary of using them, like, taking a picture of a card or something. But I don’t know ifwe just get used to it and just kind oflook the other way, but I use them more. Not that I’m not concerned, but, like, I don’t know, convenience kind ofwins.”

Participants also mentioned GenAI supporting continued connection with family and social networks by reducing their dependence on others for routine tasks, freeing interpersonal requests for situations that genuinely needed a human. This substitution trades the efort of a request against the sincerity and accountability a human respondent could ofer, and participants drew that line deliberately rather than by default. They distinguished between low-stakes communicative tasks (e.g., drafting emails, making presentations) where GenAI was an acceptable replacement for a person, and high-stakes tasks (e.g., financial, medical, navigation-critical information) where they preferred a human even if it meant asking for help. P15 shared using GenAI output as a communicative starting point rather than a final answer, “Like you can use whatever that response is as a structured starting point. Like it gives you at least to have a structure like an outline and then you can go ahead andjust start looking.”

## 4.3 No Accessibility, No Communication

Regardless of how well a GenAI tool performed or how carefully it handled personal information, participants could not use it to communicate at all if the interface itself was inaccessible. Unlabeled prompt fields, unclear buttons, and broken copy-and-paste functionality directly blocked communicative exchanges before anything else about the tool could even matter. P11 said:

“IfI’ll ask them a question, it will give me the answer, and JAWS will read it, but I also want to copy that answer, and I do not know where that answer is. I’m not even able to copy the answer.”

This positions accessibility not only as a factor among several, but as a threshold condition for communication itself. An inaccessible interface does not produce a lower-quality conversation, it produces no conversation at all. Participants also reported that accessibility had significantly improved over time. They also mentioned that this trajectory of improvement increased their willingness to treat the remaining barriers as temporary rather than disqualifying.

## 4.4 What the Machine Knows to Help

Because GenAI was often standing in for a person, participants weighed privacy the way one might weigh what to disclose to a stranger versus a trusted intermediary. Sharing a photo of a document, a medical form, or one’s surroundings to receive a description or reading carries diferent stakes than doing the same with a family member. Participants described this as a conscious, negotiated tradeof rather than indiference as convenience and independence often outweighed privacy hesitation, particularly when the alternative was depending on another person. Ownership of the underlying company shaped this perspective. Some participants avoided tools from companies they distrusted, while others prioritized accessibility and usefulness over corporate reputation. “I trust those less than Be My Eyes or even Microsoft... their products are just so useful and accessible that it doesn’t bother me as much.”, said P17. Personalization produced a similar tension. Participants valued GenAI tools “knowing” their needs as BLV users, but also worried about how much the system retained about them over time.

## 5 The Double-Edged Sword: Gains and Risk When GenAI Speaks for BLV Users

Framing GenAI tools as a communication stand-in, rather than a general purpose tool, surfaces benefits and risks that a narrower usability lens tends to miss. On the benefit side, participants described gains that were social as much as technical such as fewer routine requests to family and friends, more privacy around personal matters they might not want to disclose to a person at all, and a stronger day-to-day sense of independence. These gains largely extended participants’ existing relationships rather than replacing them. People still reserved interpersonal requests for situations that genuinely needed human assistance, and used GenAI to absorb the smaller, more frequent asks that used to fall on the people closest to them.

The risks follow directly from GenAI occupying a role a person used to hold. When a family member misreads a label or misdescribes a scene, the error is usually visible, correctable, and socially accountable. The person can be asked to look again, or the person may express uncertainty about the reliability of the information. When GenAI makes the same kind of error, it can be delivered with the same confidence as a correct answer. It is often harder to trace back to a cause, especially in higher stakes exchanges involving financial documents, or medical instructions. Participants managed this risk by reserving GenAI tools for low-stakes communication and returning to a person for anything consequential. But this strategy places the burden of identifying failure on the BLV user rather than on the system, and assumes the user can reliably tell in advance which exchange will turn out to be high-stakes.

A second risk concerns disclosure without full awareness of its consequences. Communicating through GenAI tools often means sharing images, documents, or spoken information that a BLV user might never disclose to a stranger, but is asked to disclose to a system whose data practices are dificult to inspect. Participants weighed this cost consciously, but not always with complete infor mation, and this blind spot was not limited to any one group. As P8 observed, reflecting on both younger and older users: “There’s gotta be some digital literacy. And that’s one ofthe things that is a problem with my mom’s generation. And, I’m not even gonna say my mom’s generation. It’s crazy how I guess Gen Z I work with. They grew up with it, so they don’t much think about it either, you know, so there’s just not that kind ofawareness.”

## 6 Design Implications and Policy Recommendations

Communicate Failure and Uncertainty Explicitly. Sighted users can easily infer a stalled, failed or incomplete response through visual cues, but BLV users often cannot or face dificulties. So GenAI system should treat uncertainty and failure as communicative events requiring explicit non-visual signaling. For example, the tools can announce when the process is still running or it has failed. It can also do the same if it has low confidence in the information provided. This lowers the risk that an unflagged error is mistaken for a trustworthy response giving BLV users the same failure cues a human being would provide.

Make Accessibility a Persistent, System-Wide State. An inaccessible interface does not just create friction, it risks shutting down the communicative exchange entirely. To reduce this risk, accessibility should be an integral part of any GenAI system development, not just an afterthought.

Support Task-Sensitive Communication Modes. The risk of a miscommunication carries diferent consequences depending on what is at stake. Participants trusted GenAI tools diferently when it was replacing a low-stakes convenience versus a high-stakes task. Systems should let users, or context, select a more conservative, uncertainty-forward mode for high-stakes communicative tasks like medical, financial, or navigation-related exchanges, versus a more exploratory mode for casual use. This reduces the risk that trust earned in low-stakes exchanges gets overgeneralized to consequential ones, a risk that our findings suggest is unevenly distributed.

Reduce Disclosure Risk Through Accessible and Early Communication. BLV users often disclose images, documents, and voice recordings that a sighted user might handle diferently, or might not share with an unfamiliar party at all. Systems should therefore communicate, in explicit and screen-reader-accessible terms, what happens to that information (retention, reuse, thirdparty sharing) at the point of disclosure itself, rather than in a separate privacy policy that is rarely read and often inaccessible. Because our findings show that awareness of this risk is uneven across age groups, this communication should not assume a baseline level of digital literacy. It should be legible on first encounter, before the disclosure is made rather than after.

Establish Baseline and Cross-Platform Protection Policy. Participants did not want to bear the risk of individually vetting every company’s privacy and accessibility practices before trusting it to carry a communicative exchange. This points toward a policy need for standardized, enforceable baseline protections which will cover data handling and accessibility conformance. This policy should be applied across GenAI products, shifting the burden of managing communication risk away from individual BLV users. Such a baseline would be especially valuable given that our findings show ownership and corporate reputation already function as an imperfect, unevenly applied proxy for trust. Some participants avoided companies they distrusted, while others set that concern aside when a tool was the only accessible option available to them.

## 7 Conclusion

This paper examined how BLV users communicate through GenAI tools when they stand in for reading, describing, and asking another person for help. Our findings show that this substitution is not automatic. BLV users exercise independent judgment about when to rely on GenAI and when a human is still needed, distinguishing low-stakes tasks from high-stakes ones and weighing reliability, accessibility, and disclosure with every exchange. These results point to the limits of treating GenAI as a general-purpose tool and underscore the need to design it explicitly as a communication partner, particularly as the cost of poor accessibility, opaque disclosure, and silent failure rises with GenAI’s growing role in daily communication. Supporting BLV communication through GenAI is not a narrow accessibility problem. It is a test of whether AI systems can earn a place in relationships that used to belong to people.

## References

[1] Rudaiba Adnin and Maitraye Das. 2024. “I look at it as the king of knowledge”: How Blind People Use and Understand Generative AI Tools. In Proceedings of the 26th International ACM SIGACCESS Conference on Computers and Accessibility. Association for Computing Machinery, New York, NY, USA, Article 64, 14 pages. doi:10.1145/3663548.3675631

[2] Taslima Akter. 2020. Privacy Considerations of the Visually Impaired with Camera Based Assistive Tools. In Companion Publication ofthe 2020 Conference on Computer Supported Cooperative Work and Social Computing. Association for Computing Machinery, New York, NY, USA, 69–74. doi:10.1145/3406865.3418382

[3] Taslima Akter, Tousif Ahmed, Apu Kapadia, and Manohar Swaminathan. 2022. Shared privacy concerns of the visually impaired and sighted bystanders with camera-based assistive technologies. ACM Transactions on Accessible Computing (TACCESS) 15, 2 (2022), 1–33.

[4] Taslima Akter, Tousif Ahmed, Apu Kapadia, and Swami Manohar Swaminathan. 2020. Privacy considerations of the visually impaired with camera based assistive technologies: Misrepresentation, impropriety, and fairness. In Proceedings of the

22nd International ACM SIGACCESS Conference on Computers and Accessibility. 1–14.

[5] Taslima Akter, Bryan Dosono, Tousif Ahmed, Apu Kapadia, and Bryan Semaan. 2020. “I am uncomfortable sharing what I can’t see”: Privacy Concerns of the Visually Impaired with Camera-Based Assistive Applications. In Proceedings of the 29th USENIX Conference on Security Symposium. USENIX Association, USA, Article 109, 20 pages.

[6] Taslima Akter, Aparajita S. Marathe, Darren Gergle, and Anne Marie Piper. 2025. Beyond Accessibility: Understanding the Ease of Use and Impacts of Digital Collaboration Tools for Blind and Low Vision Workers. In Proceedings ofthe 27th International ACM SIGACCESS Conference on Computers and Accessibility. Association for Computing Machinery, New York, NY, USA, Article 85, 17 pages. doi:10.1145/3663547.3746332

[7] Taslima Akter, Manohar Swaminathan, and Apu Kapadia. 2024. Toward Efective Communication of AI-Based Decisions in Assistive Tools: Conveying Confidence and Doubt to People with Visual Impairments at Accelerated Speech. In Proceedings ofthe 21st International Web for All Conference. Association for Computing Machinery, New York, NY, USA, 177–189. doi:10.1145/3677846.3677862

[8] Sabriya Maryam Alam, Marwa Abdulhai, and Niloufar Salehi. 2025. Blind Faith? User Preference and Expert Assessment of AI-Generated Religious Content. In Proceedings ofthe 2025 ACM Conference on Fairness, Accountability, and Transparency. Association for Computing Machinery, New York, NY, USA, 2451–2479. doi:10.1145/3715275.3732162

[9] Saleema Amershi, Dan Weld, Mihaela Vorvoreanu, Adam Fourney, Besmira Nushi, Penny Collisson, Jina Suh, Shamsi Iqbal, Paul N. Bennett, Kori Inkpen, Jaime Teevan, Ruth Kikin-Gil, and Eric Horvitz. 2019. Guidelines for Human– AI Interaction. In Proceedings ofthe 2019 CHI Conference on Human Factors in Computing Systems. Association for Computing Machinery, New York, NY, USA, Article 3, 13 pages. doi:10.1145/3290605.3300233

[10] Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. 2021. On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency. Association for Computing Machinery, New York, NY, USA, 610–623. doi:10.1145/3442188.3445922

[11] Abeba Birhane, Pratyusha Kalluri, Dallas Card, William Agnew, Ravit Dotan, and Michelle Bao. 2022. The Values Encoded in Machine Learning Research. In Proceedings ofthe 2022 ACM Conference on Fairness, Accountability, and Transparency. Association for Computing Machinery, New York, NY, USA, 1–12. doi:10.1145/3531146.3533083

[12] Virginia Braun and Victoria Clarke. 2006. Using thematic analysis in psychology. Qualitative research in psychology 3, 2 (2006), 77–101.

[13] Ruei-Che Chang, Rosiana Natalie, Wenqian Xu, Jovan Zheng Feng Yap, and Anhong Guo. 2025. Probing the Gaps in ChatGPT Live Video Chat for Real World Assistance for People who are Blind or Visually Impaired. In Proceedings of the 27th International ACM SIGACCESS Conference on Computers and Accessibility. Association for Computing Machinery, New York, NY, USA. doi:10.1145/3663547. 3746319

[14] Mahmut Erdemli. 2025. Designing Accessible Calendar Tools for Blind and Low-Vision Users. In Proceedings ofthe 27th International ACM SIGACCESS Conference on Computers and Accessibility. Association for Computing Machinery, New York, NY, USA, Article 174, 9 pages. doi:10.1145/3663547.3759423

[15] Yuanyuan Feng, Abhilasha Ravichander, Yaxing Yao, Shikun Zhang, Rex Chen, Shomir Wilson, and Norman Sadeh. 2024. Understanding How to Inform Blind and Low-Vision Users about Data Privacy through Privacy Question Answer ing Assistants. In Proceedings ofthe 33rd USENIX Security Symposium. USENIX Association, USA, Article 116, 2065–2082 pages.

[16] Ricardo E. Gonzalez Penuela, Jazmin Collins, Cynthia Bennett, and Shiri Azenkot. 2024. Investigating Use Cases of AI-Powered Scene Description Applications for Blind and Low Vision People. In Proceedings ofthe 2024 CHI Conference on Human Factors in Computing Systems. Association for Computing Machinery, New York, NY, USA, Article 901, 21 pages. doi:10.1145/3613904.3642211

[17] Minh-Trí Huynh and Thomas Aichner. 2025. In generative artificial intelligence we trust: unpacking determinants and outcomes for cognitive trust. AI & Society 40 (2025), 5849–5869. doi:10.1007/s00146-025-02378-8

[18] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of Hallucination in Natural Language Generation. Comput. Surveys 55, 12, Article 248 (December 2023), 38 pages. doi:10.1145/3571730

[19] Jeongwon Jo, He Zhang, Jie Cai, and Nitesh Goyal. 2025. AI Trust Reshaping Ad ministrative Burdens: Understanding Trust-Burden Dynamics in LLM-Assisted Benefits Systems. In Proceedings ofthe 2025 ACM Conference on Fairness, Accountability, and Transparency. Association for Computing Machinery, New York, NY, USA, 1172–1183. doi:10.1145/3715275.3732077

[20] Davinder Kaur, Suleyman Uslu, Kaley J Rittichier, and Arjan Durresi. 2022. Trustworthy artificial intelligence: a review. ACM computing surveys (CSUR) 55, 2 (2022), 1–38.

[21] Barbara Leporini, Marina Buzzi, and Giuseppe Della Penna. 2025. A Preliminary Evaluation of Generative AI Tools for Blind Users: Usability and Screen Reader

Interaction. In Proceedings of the 18th ACM International Conference on PErvasive Technologies Related to Assistive Environments. Association for Computing Machinery, New York, NY, USA, 562–568. doi:10.1145/3733155.3737910

[22] Bo Li, Peng Qi, Bo Liu, Shuai Di, Jingen Liu, Jiquan Pei, Jinfeng Yi, and Bowen Zhou. 2023. Trustworthy AI: From principles to practices. Comput. Surveys 55, 9 (2023), 1–46.

[23] Q. Vera Liao, Daniel Gruen, and Sarah Miller. 2020. Questioning the AI: Informing Design Practices for Explainable AI User Experiences. In Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems. Association for Computing Machinery, New York, NY, USA, 1–15. doi:10.1145/3313831.3376590

[24] Haozhe Lin, Jiangtao Gong, Yu Wang, Jinsong Zhang, Bing Bai, Yan Zhang, Luyao Wang, Chenyu Wei, Yancheng Cao, Kun Li, Ruqi Huang, and Guyue Zhou. 2025. AI system facilitates people with blindness and low vision in interpreting and experiencing unfamiliar environments. npj Artificial Intelligence 1, Article 7 (2025). doi:10.1038/s44387-025-00006-w

[25] Nora McDonald, Sarita Schoenebeck, and Andrea Forte. 2019. Reliability and inter-rater reliability in qualitative research: Norms and guidelines for CSCW and HCI practice. Proceedings of the ACM on human-computer interaction 3, CSCW (2019), 1–23.

[26] Andrew McIntyre, Lucy Conover, and Federica Russo. 2025. A Network Approach to Public Trust in Generative AI. Philosophy & Technology 38, 4 (2025), 137.

[27] Ye Mo, Gang Huang, Liangcheng Li, Dazhen Deng, Zhi Yu, Yilun Xu, Kai Ye, Sheng Zhou, and Jiajun Bu. 2025. TableNarrator: Making Image Tables Accessible to Blind and Low Vision People. In Proceedings ofthe 2025 CHI Conference on Human Factors in Computing Systems. Association for Computing Machinery, New York, NY, USA, Article 297, 17 pages. doi:10.1145/3706598.3714329

[28] Cliodhna O’Connor and Helene Jofe. 2020. Intercoder reliability in qualitative research: Debates and practical guidelines. International journal ofqualitative methods 19 (2020), 1609406919899220.

[29] Crystal Qian and James Wexler. 2024. Take It, Leave It, or Fix It: Measuring Productivity and Trust in Human-AI Collaboration. In Proceedings ofthe 29th International Conference on Intelligent User Interfaces. Association for Computing Machinery, New York, NY, USA, 370–384. doi:10.1145/3640543.3645198

[30] Inioluwa Deborah Raji, Andrew Smart, Rebecca N. White, Margaret Mitchell, Timnit Gebru, Ben Hutchinson, Jamila Smith-Loud, Daniel Theron, and Parker Barnes. 2020. Closing the AI Accountability Gap: Defining an End-to-End Framework for Internal Algorithmic Auditing. In Proceedings ofthe 2020 Conference on Fairness, Accountability, and Transparency. Association for Computing Machin ery, New York, NY, USA, 33–44. doi:10.1145/3351095.3372873

[31] Massimo Regona, Tan Yigitcanlar, Carol Hon, and Melissa Teo. 2026. Building Trust in Artificial Intelligence: A Systematic Review through the Lens of Trust Theory. Comput. Surveys (2026).

[32] Abu Noman Md Sakib, Protik Dey, Zijie Zhang, and Taslima Akter. 2026. Explainable AI for Blind and Low-Vision Users: Navigating Trust, Modality, and Interpretability in the Agentic Era. arXiv preprint arXiv:2604.00187 (2026).

[33] Tanusree Sharma, Yu-Yun Tseng, Lotus Zhang, Ayae Ide, Kelly Avery Mack, Leah Findlater, Danna Gurari, and Yang Wang. 2025. “Before, I Asked My Mom, Now I Ask ChatGPT”: Visual Privacy Management with Generative AI for Blind and Low-Vision People. In Proceedings ofthe 27th ACM SIGACCESS Conference on Computers and Accessibility. Association for Computing Machinery, New York, NY, USA, Article 17, 14 pages. doi:10.1145/3663547.3746335

[34] Yosephine Susanto, Adithya Venkatadri Hulagadri, Jann Railey Montalan, Jian Gang Ngui, Xianbin Yong, Wei Qi Leong, Hamsawardhini Rengarajan, Peerat Limkonchotiwat, Yifan Mai, and William Chandra Tjhi. 2025. SEA-HELM: Southeast Asian Holistic Evaluation of Language Models. In Findings ofthe Association for Computational Linguistics: ACL 2025. Association for Computational Linguis tics, Vienna, Austria, 12308–12336.

[35] Xinru Tang, Ali Abdolrahmani, Darren Gergle, and Anne Marie Piper. 2025. Everyday Uncertainty: How Blind People Use GenAI Tools for Information Access. In Proceedings ofthe 2025 CHIConference on Human Factors in Computing Systems. Association for Computing Machinery, New York, NY, USA, Article 63, 17 pages. doi:10.1145/3706598.3713433

[36] Scott Thiebes, Sebastian Lins, and Ali Sunyaev. 2021. Trustworthy artificial intelligence: S. Thiebes et al. Electronic Markets 31, 2 (2021), 447–464

[37] Yu-Yun Tseng, Tanusree Sharma, Lotus Zhang, Abigale Stangl, Leah Findlater, Yang Wang, and Danna Gurari. 2024. BIV-Priv-Seg: Locating Private Con tent in Images Taken by People With Visual Impairments. arXiv preprint arXiv:2407.18243 (2024). https://arxiv.org/abs/2407.18243

[38] Yimeng Wang, Yinzhou Wang, Kelly Crace, and Yixuan Zhang. 2025. Understanding Attitudes and Trust of Generative AI Chatbots for Social Anxiety Support. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems. Association for Computing Machinery, New York, NY, USA, Article 1123, 21 pages. doi:10.1145/3706598.3714286

[39] Jingyi Xie, Rui Yu, He Zhang, Syed Masum Billah, Sooyeon Lee, and John M. Carroll. 2025. Beyond Visual Perception: Insights from Smartphone Interaction of Visually Impaired Users with Large Multimodal Models. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems. Association for Computing Machinery, New York, NY, USA, Article 62, 17 pages. doi:10.1145

[40] Lotus Zhang, Abigale Stangl, Tanusree Sharma, Yu-Yun Tseng, Inan Xu, Danna Gurari, Yang Wang, and Leah Findlater. 2024. Designing Accessible Obfuscation Support for Blind Individuals’ Visual Privacy Management. In Proceedings of the

## 3706598.3714210

2024 CHI Conference on Human Factors in Computing Systems. Association for Computing Machinery, New York, NY, USA, Article 235, 19 pages. doi:10.1145/ 3613904.3642713