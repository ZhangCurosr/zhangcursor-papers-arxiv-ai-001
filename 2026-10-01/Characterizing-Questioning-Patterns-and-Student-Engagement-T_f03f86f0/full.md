# Characterizing Questioning Patterns and Student Engagement Through Contextual Analysis of Real-Time Classroom Interactions

Rohit Sharma<sup>1</sup>, Pavani Ayinampudi<sup>2</sup>, Aditya B.M.V.<sup>2</sup>, Jinal Gupta<sup>2</sup>, Prakash Hegade<sup>2</sup>, Sakshi Sharma<sup>1</sup>, Meenakshi V<sup>1</sup>, and SRS Iyengar<sup>1</sup>

<sup>1</sup> Indian Institute of Technology Ropar, Rupnagar, Punjab 140001, India <sup>2</sup> ANNAM.AI, Rupnagar, Punjab, India

Abstract. Real-time classroom polling is now routine, yet the data it produces is usually read narrowly, as a correctness score or a headcount. Such readings say little about what a poll is doing within a lecture or how it shapes engagement. This is particularly relevant for short-response formats such as True/False, where the same question format can be used to test recall, check comprehension, or direct students’ attention to a deliberately misleading statement. This study asks whether a poll’s answer and instructional function can be determined by reading it against its lecture transcript, what cognitive levels of Bloom’s taxonomy and instructional-function clusters the corpus contains, and how student engagement relates to answering correctly. We analyse a naturalistic corpus of 47 live sessions over 39 days, comprising 604 poll questions and 340,668 responses from 2,807 learners, most items True/False, read against timealigned lecture transcripts and attendance. Reading each poll in context proves essential: the answer to 89% of polls is locatable in the lecture, and a recurring attention-checking device is visible only through context. Questioning is overwhelmingly lower-order and falls into seven instructional functions, and a poll’s response follows its function rather than its wording. Engagement is broad but concentrated, and the class majority answers correctly 88.5% of the time, though a small set of high-consensus yet incorrect answers cannot be detected by agreement alone. An independent survey of 579 students agrees on what the polls are and on their participation, but reveals a gap between perception and reality: students cannot judge their own correctness, and the polls they find hardest are not those they answer worst.

Keywords: Bloom’s taxonomy · Classroom polling · Instructional function · Learning analytics · Real-time formative assessment · Student engagement

## 1 Introduction

Classroom instruction involves continuous interaction between instructors and students. Instructors present ideas, explain concepts, monitor students’ understanding, and adapt their teaching as the session progresses [1]. Students contribute to this process through their participation, responses, and interactions during the session. These interactions allow instructors to observe how students are engaging with the material and how the instruction is progressing. When such interactions are recorded across multiple classes and sessions, they generate a large amount of data that can contain recurring patterns in instructional activity and student engagement. Examining such a large amount of data manually can make it dificult to identify these patterns across sessions. Digital tools can make this analysis feasible by allowing classroom interactions to be organised and analysed across many sessions [2].

One of the ways of classroom interaction is real-time polling. These polls allow instructors to ask questions and collect responses during instruction. Instructors increasingly use clickers, web-based response systems, and polling features built into video-conferencing platforms to collect student responses almost instantaneously [3,4,5]. The resulting records are commonly summarised through performance, measured by the proportion of learners who answer correctly, and participation, measured by the proportion who submit a response [6,7]. Both measures provide useful information about what happened after a question was asked, but they say little about the purpose the question served within the lecture or the kind of thinking it required from students.

Questions asked during a lecture can require students to do diferent things even when they produce the same type of response. A True/False question may ask students to recall a fact, check an explanation, reinforce a recently introduced idea, or pay attention to a deliberate change in a statement. A response to each question can therefore have a diferent instructional meaning even when the response options are identical. Looking only at whether students responded or answered correctly cannot distinguish between these cases [8,9]. The question, therefore, carries information about its instructional purpose that is not captured by response counts or correctness scores.

Understanding the instructional purpose of a question also requires considering the context in which it is asked. Its meaning can depend on what the instructor has explained before it, the examples used to develop the topic, and the point at which it appears in the lecture. A True/False question may check recall when it follows a factual explanation, but may check understanding when it follows a conceptual explanation. Questions can also prompt retrieval, redirect attention, check understanding, or provide feedback during instruction [10,11,12,13,1,14]. Reading the question together with its surrounding lecture can therefore reveal why it was asked and what students were expected to consider at that point.

A poll can therefore be viewed as part of a larger classroom interaction consisting of the lecture context, the question, and the responses it generates. Existing analyses of classroom polling rarely consider these elements together, leaving the relationship between the instructional purpose of a poll and the responses it receives less clearly understood. In this study, we use this interaction as the basis for examining questioning patterns and student engagement in realtime classroom polls. We analyse 604 real-time polls from 47 live sessions using time-aligned poll records, lecture transcripts, and attendance data.

## 2 Background Study

Research on classroom polling has examined its efects on learning, participation, attention, and feedback for more than two decades. Reviews by Fies and Marshall [6], Caldwell [3], and Kay and LeSage [4] report benefits such as increased participation, greater attention, and timely feedback, while also describing the practical challenges of using polling efectively. A meta-analysis by Hunsu et al. [5], covering more than fifty studies, found that these benefits were real but generally modest and depended on how the polling tool was used. Across this literature, outcomes appear to depend more on the pedagogy surrounding polling than on the technology itself [6]. This finding is consistent with broader evidence that active-learning approaches generally produce better learning outcomes than passive lectures in science and engineering [15,16]. Less attention has been given to what the response record itself contains and to how student responses vary across individual polls.

Attention to the question itself has a long precedent in Peer Instruction, where conceptual questions became a central part of classroom interaction and learning. Studies by Mazur [17], Crouch and Mazur [18], and Smith et al. [19] found that discussing a question with peers between successive votes can improve students’ responses. This work also emphasises the purpose and quality of the question. A useful question is one that requires students to reason and can reveal common misconceptions, rather than one that is judged only by its wording. A similar view appears in formative-assessment research [1,20], as well as in work that distinguishes types of feedback by their instructional function, such as confirming an understanding, identifying a dificulty, or guiding students toward a diferent approach [14,21]. Together, these studies suggest that classroom questions and responses can be understood in terms of the instructional function they serve, rather than through correctness alone.

Cognitive psychology provides evidence that questions can support learning during lectures through retrieval and attention. Answering questions can strengthen retention more efectively than restudying the same material, while brief questions during instruction can reduce mind-wandering and improve retention of subsequent material [10,11,12,22,13].

Two further areas of research help explain what polls require from students and how students respond to them. The original Bloom’s taxonomy [23] and its revised, verb-oriented version [24,25] provide two ways to characterise the cognitive demand of a question. Previous studies that code assessment items have repeatedly found a skew toward lower-order cognitive processes [26,8]. Research on teacher questioning also classifies the questions instructors ask according to their cognitive level and communicative function [27,9,28]. These approaches provide established ways to describe the cognitive demands of classroom questions, but they generally assess questions as individual items rather than in the instructional context in which they are used. A separate area of learninganalytics research has consistently found that online participation is uneven, with a small group of highly active participants alongside a larger group of less active participants [29,30,31,32]. Participation is also commonly treated as a form of behavioural engagement [7], but such analyses often focus on overall participation rather than variation across individual classroom questions. Taken together, these studies provide ways to describe the cognitive demand of questions and patterns of student participation, but they generally examine these dimensions separately rather than as parts of the same classroom interaction. Less is known about how the instructional purpose of an individual poll relates to the responses it receives when the question is interpreted within the lecture in which it occurs. There is also limited evidence connecting such contextual interpretations of poll responses with students’ own reports of their experience.

## 3 Methodology and Methods

## 3.1 Research Questions

Our main research question asks what a transcript-grounded reading of realtime polls can reveal about how instructors question and how students engage and answer, beyond what correctness scores or participation counts capture. We address this question through three sub-questions:

RQ1. Can a poll’s answer and instructional function be determined by reading it in its lecture context?

RQ2. What cognitive levels and instructional functions does the corpus of polls contain?

RQ3. How is student engagement structured, and how does it relate to answering correctly?

Each sub-question addresses a diferent part of the main question. RQ1 establishes whether the lecture context is suficient to determine a poll’s answer and to assign its instructional function, which provides the basis for the subsequent analysis. RQ2 uses this contextual reading to characterise how the instructors question the cohort. RQ3 turns to student responses, examining how engagement is distributed and whether it is associated with correctness. Throughout, engagement refers to behavioural engagement in the sense of Fredricks et al. [7], that is, whether a student responded; the design does not measure its emotional or cognitive components.

## 3.2 Unit of Analysis and Coding

We do not classify a poll only by its wording. Instead, we analyse the complete educational interaction surrounding the poll, including the lecture discourse before and after it, the poll question itself, and the distribution of student responses it draws (Fig. 1). Considering these elements together allows us to distinguish identical True/False items by the instructional function they serve.

For each interaction, we derive four types of information. First, we assign the poll a cognitive level based on the original Bloom’s taxonomy. This classification reflects the cognitive process required of students in the instructional context, rather than the verbs used in the question or its format. Then in each cognitive level, we assign an instructional function that describes the purpose served by the poll. When the session establishes an objectively correct answer, we record it as the answer key and use it to assess response accuracy. Polls addressing opinions, mood, or logistical matters are excluded from accuracy analysis because they do not have a correct answer. Finally, consensus, response spread, and participation rate are derived as three measures of engagement for students present when the poll was launched. Throughout the analysis, agreement and accuracy are treated as distinct measures.

Coding procedure. The coding was carried out with a large language model (a Claude Opus model, Anthropic) rather than by hand. For each poll, the model received the question together with the transcript from six minutes before the poll to three minutes after it and followed a written protocol: it first stated the instructional intent of the poll and the cognitive process expected of students, and only then assigned a Bloom level, recording a primary and an alternative level, a confidence rating, and a justification that cites the transcript. The intent recorded here is the intent the lecture context supports, not a claim about what the speaker privately intended. The model and the rule set had no access to the response distributions, so levels and functions were fixed independently of how students answered. Instructional functions were then assigned by a fixed rule set applied to the recorded intent and process; the seven functions are the categories this procedure produced, and we treat them as descriptive rather than as an established taxonomy. The coded output was reviewed by the authors, and the answer key was verified manually by an author for every poll on which the class majority disagreed with the derived answer, with 28 answers corrected. To assess reliability, an independent second automated coder relabelled a random sample of 40 cognitive polls: agreement with the rule-based functions was 65% $( \kappa = 0 . 4 8 )$ , with disagreements concentrated between adjacent functions such as Factual Recall and Attention/Distortion Check. Automated coding of educational text is an established practice, from classifiers that assign Bloom levels to questions [33] and automated measures of classroom discourse [34] to language-model annotation whose agreement with human coders matches agreement among human coders in several tasks [35,36,37]. The protocol, rule set, and reliability sample are given in the online appendix.

## 3.3 Survey and Triangulation

A short post-program student survey consisting of six five-point Likert-scale questions was conducted separately from the corpus analysis. Each question was designed to correspond to one measured finding, and the survey also included two open-ended questions about the perceived purpose of the polls and which polls students found dificult. We matched each respondent to their measured record by hashed identity and compared the two sources at the class-aggregate and per-student levels (Sect. 5.1).

![](images/cafaeebd402c422c3edeb0671c3f1bf52b4f7feb77a54bd93fe84779cbfeb68a.jpg)  
Fig. 1. The poll as an educational interaction (the unit of analysis): its lecture context, the question, and the response distribution, from which we derive its cognitive level and instructional function, an answer key for correctness, and measures of engagement.

## 4 Data Collection and Analysis

## 4.1 Study Context

The corpus is drawn from 47 live sessions conducted across 39 calendar days as part of an online summer internship orientation program led by multiple speakers. The program focuses on motivation, professional culture, and exposure to concepts rather than graded mastery of skills. This orientation genre is relevant to the interpretation of the results. It shapes the heavily lower-order questioning profile we report, so throughout we distinguish findings that are likely specific to this kind of program from those that should generalise to other settings.

## 4.2 Data Streams

Each session provides three forms of data that are time-aligned. The first is the live poll records, which contain the options presented for each poll, individual responses, and submission timestamps. The second is the lecture transcript, aligned with the session timeline, which records what was said at a particular time during the session. The third is attendance, recorded through per-attendee summaries and join and leave intervals for each learner, allowing us to determine who was present during each poll. The full corpus contains 604 poll questions, 340,668 responses, and 2,807 distinct learners. Most questions are True/False, and each session has a corresponding set of polls and transcript records.

## 4.3 Alignment

To interpret the instructional function of each poll, we place the poll and transcript timestamps on a common timeline. For each poll, we calculate its sessionrelative time by subtracting the actual session start time from the timestamp of its first submitted response. We validated the resulting alignment against the timeline recorded in the transcript. We then examine the lecture content from roughly six minutes before to three minutes after the poll is launched. This window captures the content immediately before, during, and after the question and provides the context used to interpret its instructional function (Fig. 2).

![](images/9b967fc5f0ea5aa22a461bb595c7430dc5005b76fbc8de62c5fc90182c3c7015.jpg)  
Fig. 2. Reading a poll in its lecture moment: its launch time locates it on the transcript timeline, and the surrounding window supplies the talk that gives the question its meaning.

## 4.4 Data Handling and Ethics

Prior to data collection, informed consent for the use of student data in the study was obtained from all participating students. The analysis uses only aggregate data or salted-hash pseudonyms, generated by applying a one-way hash to normalised email addresses. All cross-stream joins, including linking student poll responses with attendance records and later with survey responses, are performed using these pseudonyms. No individual student is identified in any table or figure. Continued participation in the program was partly contingent on answering polls. As a result, the recorded participation reflects both students’ engagement and the requirement to answer the polls. We take this constraint into account when interpreting the results in Sect. 5 (Results).

## 4.5 Supplementary Material

The detailed summaries of the anonymised dataset, including corpus composition, the Bloom and instructional-function taxonomies, answer-key coverage and correctness, engagement measures, and a per-session table, are provided in an online appendix. It also includes the complete methods and results of the triangulation survey.<sup>3</sup>

## 5 Results and Discussion

## 5.1 Results

Reading a poll in its lecture context (RQ1). The answer to 89% of polls (539 of 604) could be objectively located in the surrounding lecture. The lecture context was therefore suficient to settle what the correct response was for the large majority of polls. Reading the polls alongside their surrounding transcript also revealed a recurring type of attention-checking poll, the Attention/Distortion-Check, in which a speaker restates a lecture point with a deliberate error, such as changing a number, inserting a negation, or attributing a claim incorrectly. In these cases, the statement may appear true when read on its own, while the surrounding lecture shows that it is false. The finding therefore shows that the context surrounding a poll can contain information about its instructional function that is not available from the question wording alone.

Question types and instructional functions (RQ2). We describe each cognitive poll in two ways: by its cognitive demand, using the six levels of the original Bloom’s taxonomy, and by its instructional function, a finer category nested within each cognitive level. The two classifications capture diferent aspects of a poll. Cognitive demand describes the level of thinking a question requires, while instructional function describes the purpose the poll serves within the lecture. Together, they provide a fuller description of how questions are used across the cohort than either classification alone.

By cognitive demand, the polls are overwhelmingly lower-order. Of the 512 cognitive polls, 96.8% fall at the Knowledge or Comprehension levels of Bloom’s taxonomy (Fig. 3). Only sixteen polls are at higher-order levels, namely Application, Analysis, or Evaluation, while Synthesis is absent entirely. This distribution remains stable across the 39 days, with no clear shift toward more demanding questions as the program progresses. The pattern is consistent with the orientation format of the program, which emphasises exposure to ideas rather than graded mastery.

Nested within that lower-order band, the polls fall into seven recurring instructional functions (Table 1). Four of these correspond to functions already described in the questioning and formative-assessment literature: checking recall [10], verifying comprehension of what was just taught [9,1], redirecting attention [13], and applying a stated procedure [27]. The remaining three, reinforcing a stated principle, integrating ideas across the lecture, and a residual category, were needed to cover this corpus and are specific to it.

Two functions, verifying that a just-delivered idea has been understood and checking recall of a stated fact, together account for roughly three-quarters of the cognitive polls, while the attention-checking device introduced above is a small but distinctive category (10%). Because each poll’s function is assigned from the surrounding lecture context, we can ask whether function, rather than wording, is associated with how students respond. With the True/False format held constant, the spread of responses difers significantly across functions (Kruskal– Wallis $H = 1 7 . 8 , p = 0 . 0 0 7 )$ , whereas participation rate does not (H = 4.7, $p = 0 . 5 8 )$ . This suggests that diferences in how students respond are related to the instructional function of the poll and not to its surface format alone.

Table 1. The seven instructional functions of the 512 cognitive polls, with their counts and share of cognitive polls.
<table><tr><td>Instructional function</td><td>n % of cognitive</td></tr><tr><td>Understanding Verification</td><td>246 48.0</td></tr><tr><td>Factual Recall</td><td>129 25.2</td></tr><tr><td>Attention / Distortion Check</td><td>53 10.4</td></tr><tr><td>Concept Reinforcement</td><td>24 4.7</td></tr><tr><td>Applied Execution</td><td>22 4.3</td></tr><tr><td>Concept Integration</td><td>18 3.5</td></tr><tr><td>Other Recall / Comprehension 20</td><td>3.9</td></tr></table>

![](images/f1e3603f61ee926aa2efb4d761ef305043398748c2cf267a0bb258d581d2e748.jpg)  
Fig. 3. Cognitive demand of the 512 cognitive polls under Bloom’s taxonomy: the questioning is concentrated at the Knowledge and Comprehension levels, with higherorder levels nearly absent.

Engagement structure and correctness (RQ3). About 87% of attendees answer at least one poll, yet participation is highly concentrated. The most active fifth of students account for 74% of all responses, with a Gini coeficient of about 0.70. Against the answer key, the class majority answers correctly on 88.5% of polls and incorrectly on 11.5%. A small subset of these polls represent confident misconceptions, where a large majority selects the wrong answer. Such cases would be dificult to identify from agreement alone. Consensus and correctness are closely related $( r \approx 0 . 8 0 )$ , as expected for binary items with a known answer, where a large majority is usually a correct one; the correlation is therefore largely a property of the format. The informative cases are the exceptions, the confident misconceptions that lie of this pattern (Fig. 4). The relationship between individual participation and answer accuracy is weak, indicating that participation alone provides limited information about understanding. Because continued participation in the program depended in part on answering polls, any apparent increase in engagement over time may also reflect a selection efect rather than genuine growth in engagement.

![](images/298c91ebb56d706a888efe371c72d0afb6ebfe515d27129e7174c813c79eea7a.jpg)  
Fig. 4. Consensus versus correctness across the decidable polls $( r \approx 0 . 8 0 )$ ; the confident misconceptions are the high-consensus yet incorrect exceptions.

Triangulation against the student survey (RQ1–RQ3). An independent survey of 579 respondents, with 95% matched to their measured records by hashed identity, provides a comparison with the corpus at two levels. At the class level, the two sources show similar patterns in how students perceive the polls. Students report that the polls are tied to the lecture (91%), consistent with the 89% of polls whose answers are locatable in the lecture context, and that the polls mainly test recall or basic understanding (84%), consistent with the 96.8% of cognitive polls classified at the Knowledge or Comprehension levels. At the individual level, self-reported participation tracks measured participation, reflecting a behaviour that students can directly observe. The two sources difer more clearly in measures of correctness. Students’ self-assessed correctness does not track their measured accuracy, and among the 456 students who were confident that they could identify the answer, 49% had accuracy below the median. Students also identify the attention-checking poll as both the most memorable and the most dificult, accounting for 46% of dificulty responses, although these polls are not the most dificult to answer correctly (Fig. 5). The survey therefore captures a diference between perceived dificulty and measured accuracy. Individual-level associations are further limited by a strong response ceiling and self-selection, since survey respondents are more accurate and more participatory than non-respondents. The results are interpreted as evidence of a gap between self-perception and measured performance rather than as evidence of no association. Full statistical details are provided in the online appendix (Sect. 4.5).

![](images/2b80cd1ba4d63f2adabb7bd02d10a69120c0bbcfc16929193e4e0a59d6686733.jpg)

![](images/90fcc62230dfef804044d4f7be37d1801f3c6e4c17618cf87f195a0652b08a94.jpg)

![](images/537bccef2a9cff808a38e77f4a8862b3fc3f98c883b02c4a2496621ddb485a7c.jpg)

![](images/3ca430c1f1417d01dfd754ec706e3dc7aa3cd209b4907503d6c6b225825da4e2.jpg)

![](images/0ed96cf57758df99943771ecbfa68f9b1934923ae1a25d61327dadc682168c0b.jpg)  
Fig. 5. Self-report versus measured behaviour: only participation tracks its measure, while self-assessed competence and trap-catching do not.

## 5.2 Discussion

On their own, poll results primarily show correctness and participation. When these results are considered in the context of the lecture and compared with a second source, they provide a broader view of how a class is questioned and how students respond. Four observations support this interpretation.

Context settles the answer, and it exposes the instructional function. In 89% of the polls, the answer could be objectively determined from the surrounding lecture context. Response distributions also varied according to the instructional function of the poll, while participation rates remained stable. Together, these findings indicate that the lecture context is suficient both to settle a poll’s answer and to assign its instructional function. They also suggest that diferences in response patterns are associated with the instructional function of a poll rather than with its wording alone. The Attention/Distortion-Check illustrates what the wording alone would miss, as the statement appears true when read in isolation, but the surrounding lecture context shows that it is false.

In this setting, polls function primarily as a participation activity rather than an assessment. Cognitive demand is low and remains broadly stable, participation does not vary by instructional function, participation is broad, and individual participation is only weakly related to answer accuracy. Students also identify attention and engagement more often than assessment of understanding when describing the purpose of the polls. Taken together, these findings suggest that polling in this orientation program serves primarily to sustain participation rather than to assess mastery. As noted in Sect. 5, this is what the orientation format of the program would lead one to expect. Recovering correctness allows us to say so without dismissing the answers. The class answers correctly on roughly nine out of ten polls, so the response data carry a meaningful correctness signal, but that signal is a by-product of a participation activity rather than the outcome of a graded assessment.

The highest-value signal is confident error. Once correctness is recovered, a particularly informative pattern is agreement on an incorrect answer rather than disagreement. We refer to this as a confident misconception, where a large majority of students select the wrong answer. Such cases cannot be identified from agreement alone and become visible only when responses are compared with an answer key. This matters in practice because a live poll shows the instructor the distribution of responses, and a strong majority can easily be taken as a sign that the class has understood. Confident misconceptions are the cases in which that interpretation fails.

Confidence is not competence. The survey provides an individual-level view of this distinction. Students report the purpose of the polls and their own participation with reasonable consistency, but their confidence in whether they answered correctly does not track their measured accuracy. The polls that students report as most dificult are also not the polls on which they have the lowest accuracy. This gap between self-assessment and measured performance shows why objective, context-grounded measures are needed alongside student reports.

## 5.3 Limitations

There are several limitations to the claims made in this study. The corpus comes from a single orientation program, so the low and stable cognitive profile and the particular mix of instructional functions may reflect the nature of this program. The approach of reading polls in their lecture context, identifying their instructional function, and comparing the findings with survey responses can be applied in other settings, but the distributions observed here may difer. The survey is skewed toward agreement and represents a more engaged, self-selected group of students. This limits how we interpret the individual-level null results. We therefore treat these results as consistent with a gap between self-assessment and measured performance, rather than as evidence that no association exists. Answering polls was also partly required for continued participation in the program. The resulting participation measure reflects both student engagement and the requirement to answer polls. The participation requirement means that the participation data cannot show whether engagement increased or decreased because of the polls, nor can they establish an efect of polling on retention. The seven instructional functions identified in the corpus are specific to our analysis and should therefore be treated as an exploratory classification. The classifications are model-based with moderate second-coder agreement, so fine distinctions between adjacent functions should not be over-read. These points should be kept in mind when interpreting the findings, particularly when considering their relevance to other settings.

## 6 Conclusion

The classroom polls have the information about questioning and student engagement that is not captured by correctness scores or participation counts alone. Reading the polls in their lecture context allowed us to locate an objective answer for 89% of them and to identify a recurring attention-checking function that could not be recognised from the poll wording alone. The response distributions varied while the participation rate remained stable for questions that were primarily lower-order and fell into seven instructional functions. Student engagement was broad but concentrated, and individual participation showed only a weak relationship with correctness. The answer key which was prepared from the lecture, showed that the majority of the class answered roughly nine out of ten polls correctly, while a small set of confident misconceptions emerged when agreement was considered alongside correctness. The survey findings were consistent with the corpus in their reports of poll purpose and participation, but students’ self-assessed correctness did not fully correspond to their measured performance. Together, these findings show that the response record of a classroom poll contains information about the instructional event and student behaviour that is lost when polling is reduced to correctness and participation.

These findings gave insights for further studies. If we continue the study on the same cohort over time, it can help separate changes in participation from the efect of the program’s participation rule. In the future, we can vary the function and timing of polls to examine how these factors relate to engagement and learning under controlled conditions. Applying the analysis across diferent instructors, subjects, and graded settings would show which parts of the observed questioning profile are specific to the orientation setting. The contextual approach could also be used during live sessions to examine how polls function and response patterns, including confident misconceptions, develop as the session progresses. This would help determine whether these measures can provide instructors with useful information during instruction.

## References

1. Black, P., Wiliam, D.: Assessment and classroom learning. Assessment in Education: Principles, Policy & Practice 5(1), 7–74 (1998). https://doi.org/10.1080/ 0969595980050102

2. Ferguson, R.: Learning analytics: Drivers, developments and challenges. International Journal of Technology Enhanced Learning 4(5/6), 304–317 (2012). https: //doi.org/10.1504/IJTEL.2012.051816

3. Caldwell, J.E.: Clickers in the large classroom: Current research and best-practice tips. CBE—Life Sciences Education 6(1), 9–20 (2007). https://doi.org/10.118 7/cbe.06-12-0205

4. Kay, R.H., LeSage, A.: Examining the benefits and challenges of using audience response systems: A review of the literature. Computers & Education 53(3), 819– 827 (2009). https://doi.org/10.1016/j.compedu.2009.05.001

5. Hunsu, N.J., Adesope, O., Bayly, D.J.: A meta-analysis of the efects of audience response systems (clicker-based technologies) on cognition and afect. Computers & Education 94, 102–119 (2016). https://doi.org/10.1016/j.compedu.2015.1 1.013

6. Fies, C., Marshall, J.: Classroom response systems: A review of the literature. Journal of Science Education and Technology 15(1), 101–109 (2006). https://do i.org/10.1007/s10956-006-0360-1

7. Fredricks, J.A., Blumenfeld, P.C., Paris, A.H.: School engagement: Potential of the concept, state of the evidence. Review of Educational Research 74(1), 59–109 (2004). https://doi.org/10.3102/00346543074001059

8. Momsen, J.L., Long, T.M., Wyse, S.A., Ebert-May, D.: Just the facts? introductory undergraduate biology courses focus on low-level cognitive skills. CBE—Life Sciences Education 9(4), 435–440 (2010). https://doi.org/10.1187/cbe.10-0 1-0001

9. Chin, C.: Teacher questioning in science classrooms: Approaches that stimulate productive thinking. Journal of Research in Science Teaching 44(6), 815–843 (2007). https://doi.org/10.1002/tea.20171

10. Roediger, H.L., Karpicke, J.D.: Test-enhanced learning: Taking memory tests improves long-term retention. Psychological Science 17(3), 249–255 (2006). https: //doi.org/10.1111/j.1467-9280.2006.01693.x

11. Karpicke, J.D., Roediger, H.L.: The critical importance of retrieval for learning. Science 319(5865), 966–968 (2008). https://doi.org/10.1126/science.1152408

12. Bunce, D.M., Flens, E.A., Neiles, K.Y.: How long can students pay attention in class? a study of student attention decline using clickers. Journal of Chemical Education 87(12), 1438–1443 (2010). https://doi.org/10.1021/ed100409p

13. Szpunar, K.K., Khan, N.Y., Schacter, D.L.: Interpolated memory tests reduce mind wandering and improve learning of online lectures. Proceedings of the National Academy of Sciences 110(16), 6313–6317 (2013). https://doi.org/10.1073/pn as.1221764110

14. Hattie, J., Timperley, H.: The power of feedback. Review of Educational Research 77(1), 81–112 (2007). https://doi.org/10.3102/003465430298487

15. Hake, R.R.: Interactive-engagement versus traditional methods: A six-thousandstudent survey of mechanics test data for introductory physics courses. American Journal of Physics 66(1), 64–74 (1998). https://doi.org/10.1119/1.18809

16. Freeman, S., Eddy, S.L., McDonough, M., Smith, M.K., Okoroafor, N., Jordt, H., Wenderoth, M.P.: Active learning increases student performance in science, engineering, and mathematics. Proceedings of the National Academy of Sciences 111(23), 8410–8415 (2014). https://doi.org/10.1073/pnas.1319030111

17. Mazur, E.: Peer Instruction: A User’s Manual. Prentice Hall, Upper Saddle River, NJ (1997)

18. Crouch, C.H., Mazur, E.: Peer instruction: Ten years of experience and results. American Journal of Physics 69(9), 970–977 (2001). https://doi.org/10.1119/ 1.1374249

19. Smith, M.K., Wood, W.B., Adams, W.K., Wieman, C., Knight, J.K., Guild, N., Su, T.T.: Why peer discussion improves student performance on in-class concept questions. Science 323(5910), 122–124 (2009). https://doi.org/10.1126/scienc e.1165919

20. Black, P., Wiliam, D.: Developing the theory of formative assessment. Educational Assessment, Evaluation and Accountability 21(1), 5–31 (2009). https://doi.or g/10.1007/s11092-008-9068-5

21. Shute, V.J.: Focus on formative feedback. Review of Educational Research 78(1), 153–189 (2008). https://doi.org/10.3102/0034654307313795

22. Risko, E.F., Anderson, N., Sarwal, A., Engelhardt, M., Kingstone, A.: Everyday attention: Variation in mind wandering and memory in a lecture. Applied Cognitive Psychology 26(2), 234–242 (2012). https://doi.org/10.1002/acp.1814

23. Bloom, B.S.: Taxonomy of Educational Objectives: The Classification of Educational Goals. Handbook I: Cognitive Domain. David McKay, New York (1956)

24. Anderson, L.W., Krathwohl, D.R.: A Taxonomy for Learning, Teaching, and Assessing: A Revision of Bloom’s Taxonomy of Educational Objectives. Longman, New York (2001)

25. Krathwohl, D.R.: A revision of bloom’s taxonomy: An overview. Theory Into Practice 41(4), 212–218 (2002). https://doi.org/10.1207/s15430421tip4104\_2

26. Zheng, A.Y., Lawhorn, J.K., Lumley, T., Freeman, S.: Application of bloom’s taxonomy debunks the “mcat myth”. Science 319(5862), 414–415 (2008). https: //doi.org/10.1126/science.1147852

27. Redfield, D.L., Rousseau, E.W.: A meta-analysis of experimental research on teacher questioning behavior. Review of Educational Research 51(2), 237–245 (1981). https://doi.org/10.2307/1170197

28. Graesser, A.C., Person, N.K.: Question asking during tutoring. American Educational Research Journal 31(1), 104–137 (1994). https://doi.org/10.3102/0002 8312031001104

29. Kizilcec, R.F., Piech, C., Schneider, E.: Deconstructing disengagement: Analyzing learner subpopulations in massive open online courses. In: Proceedings of the Third International Conference on Learning Analytics and Knowledge (LAK ’13). pp. 170–179 (2013). https://doi.org/10.1145/2460296.2460330

30. Nonnecke, B., Preece, J.: Lurker demographics: Counting the silent. In: Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (CHI ’00). pp. 73–80 (2000). https://doi.org/10.1145/332040.332409

31. Sun, N., Rau, P.P.L., Ma, L.: Understanding lurkers in online communities: A literature review. Computers in Human Behavior 38, 110–117 (2014). https://do i.org/10.1016/j.chb.2014.05.022

32. Henrie, C.R., Halverson, L.R., Graham, C.R.: Measuring student engagement in technology-mediated learning: A review. Computers & Education 90, 36–53 (2015). https://doi.org/10.1016/j.compedu.2015.09.005

33. Mohammed, M., Omar, N.: Question classification based on Bloom’s taxonomy cognitive domain using modified TF-IDF and word2vec. PLOS ONE 15(3), e0230442 (2020). https://doi.org/10.1371/journal.pone.0230442

34. Demszky, D., Liu, J., Mancenido, Z., Cohen, J., Hill, H., Jurafsky, D., Hashimoto, T.: Measuring conversational uptake: A case study on student-teacher interactions. In: Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP 2021). pp. 1638–1653 (2021). https: //doi.org/10.18653/v1/2021.acl-long.130

35. Gilardi, F., Alizadeh, M., Kubli, M.: ChatGPT outperforms crowd workers for text-annotation tasks. Proceedings of the National Academy of Sciences 120(30), e2305016120 (2023). https://doi.org/10.1073/pnas.2305016120

36. Xiao, Z., Yuan, X., Liao, Q.V., Abdelghani, R., Oudeyer, P.Y.: Supporting qualitative analysis with large language models: Combining codebook with GPT-3 for deductive coding. In: Companion Proceedings of the 28th International Conference on Intelligent User Interfaces (IUI ’23 Companion). pp. 75–78 (2023). https://doi.org/10.1145/3581754.3584136

37. Tai, R.H., Bentley, L.R., Xia, X., Sitt, J.M., Fankhauser, S.C., Chicas-Mosier, A.M., Monteith, B.G.: An examination of the use of large language models to aid analysis of textual data. International Journal of Qualitative Methods 23 (2024). https://doi.org/10.1177/16094069241231168