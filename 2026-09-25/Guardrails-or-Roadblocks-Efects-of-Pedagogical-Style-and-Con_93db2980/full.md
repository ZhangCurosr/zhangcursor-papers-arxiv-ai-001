# Guardrails or Roadblocks? Efects of Pedagogical Style and Context Awareness in AI Teaching Assistants for Programming

Madeleine Eastwood University of Florida Gainesville, FL, United States meastwood@ufl.edu

Paul Denny University of Auckland Auckland, New Zealand paul@cs.auckland.ac.nz

Harshith Narne   
University of Florida   
Gainesville, FL, United States   
h.narne@ufl.edu

Ashish Aggarwal University of Florida Gainesville, FL, United States ashishjuit@ufl.edu

Joseph Hilby   
University of Florida   
Gainesville, FL, United States   
joseph.hilby@gmail.com Amanpreet Kapoor University of Florida   
Gainesville, FL, United States   
kapooramanpreet@ufl.edu

## Abstract

AI teaching assistants (AI TAs) backed by large language models (LLMs) and pedagogical guardrails are increasingly being integrated into programming courses, providing students with scalable access to hints, conceptual explanations, and code-level feedback. However, guardrails may also create friction. If students feel that the support provided is overly restrictive or poorly contextualized to their current progress, they may bypass approved tools for generalpurpose LLMs. To investigate how AI TA design afects students’ learning experiences, we conducted a randomized controlled trial with 132 students in an introductory programming course. Students completed three tasks related to code-writing and debugging and were randomly assigned to one of four AI TAs varied across two dimensions: pedagogical guidance style (Socratic vs. Direct instruction) and context awareness (no context vs. full context of the problem and student solution). We examined students’ perceptions, interaction behaviors, and evidence of post-task comprehension. Students rated the Socratic AI TA with full context least favorably, reporting significantly lower perceived support for task completion. Descriptively, this condition also showed the highest observed interaction stress, the highest rate of external LLM use, and the lowest proportion of post-task explanations demonstrating full comprehension, though these diferences were not statistically significant. These findings suggest that guardrailed AI TAs are not automatically better for learning. Instead, their efectiveness depends on how pedagogical guidance and contextual awareness are balanced in ways that students experience as useful, supportive, and worth continuing to use.

## CCS Concepts

• Social and professional topics → Computing education.

## Keywords

AI tutor, Feedback, Digital TAs, Automated Tutors, Programming

## 1 Introduction

LLM-backed AI teaching assistants (AI TAs) are increasingly being deployed in programming courses to provide scalable, alwaysavailable help while using pedagogical guardrails to avoid simply giving students solutions [14, 15, 19, 25]. Unlike general-purpose LLM chat tools, these systems are designed to support learning by providing tutor-like guidance without directly completing the work for students. Yet these guardrails create a design tension, because if a system is too restrictive or if students perceive the help as poorly contextualized to their request, they may disengage, avoid seeking help, or bypass approved tools in favor of unrestricted generalpurpose LLMs [14, 16]. This creates a problem for educators because when students migrate to external systems, opportunities to capture learning-related data are lost, and students may become more susceptible to over-reliance on AI-generated solutions [5, 7, 26]. At the same time, student favorability is not necessarily equivalent to learning efectiveness: a guardrailed AI TA that provides more friction between the student and the answer may be perceived as less helpful or favorable to the student, while still supporting the productive struggle associated with deeper learning. The goal, then, is not to maximize favorability alone, but to identify configurations that are both pedagogically sound and experienced by students as useful and worth continuing to use.

A central challenge for our community is to understand how design choices shape students’ experiences with AI TAs. Prior work has introduced and evaluated guardrailed AI systems for programming support [14, 15, 19], and instructors have expressed a strong preference for customizing AI chatbots to fit their course-specific needs [13]. However, most prior work has evaluated fixed AI TA systems as complete interventions, without explicitly testing how diferent configurations of the same system afect student perceptions, behaviors, and learning-related outcomes.

In this paper, we address this gap through a randomized controlled experiment comparing four configurations of the same AI TA. We vary pedagogical guidance style by comparing a Socratic AI TA that guides students through questions with a Direct AI TA that provides brief explanations while still withholding complete solutions. We vary context awareness by comparing an AI TA with full access to the problem statement and the student’s current solution with an AI TA that has access only to the chat history. We investigate students’ learning experiences with these AI TAs by answering the following research question:

We conducted a 2×2 factorial randomized controlled experiment with 132 students in an introductory Python course. Students completed one code-writing and two code-debugging tasks, and each student was randomly assigned to interact with one of four AI TAs configured with a specific combination of pedagogical style and context. We analyzed students’ perceptions from a post-survey, interaction behaviors from tool logs, and task comprehension from a post-task explanation of a debugging solution. Notably, the AI TA that combined Socratic guidance with full context was rated least favorably for supporting task completion, showing that ped agogically motivated designs are not automatically experienced by students as more helpful. Our findings provide evidence-based guidance for educators and developers seeking to deploy such tools efectively in real classroom settings.

## 2 Related Work

Recent work in computing education exploring the use of large language models (LLMs) can be broadly grouped into two strands. The first examines the pedagogical capabilities and limitations of LLMs, including their instructional soundness, learning implications, and ability to generate programming tasks or feedback [12, 22, 24]. The second investigates student reception and adoption of scafolded, pedagogically grounded LLM systems, including AI TA tools that are deliberately constrained to promote productive learning behaviors [2, 6, 15, 17, 19, 20, 34]. Evaluations of systems such as CodeHelp [19] and CodeAid [15] are well known examples of this latter strand of work. There is much interest in this space even beyond computing education, where similar systems have been developed, deployed, and evaluated across disciplines such as chemistry [33], medicine [9], architecture [32], engineering [27, 29], and physics [4]. As such systems become more common, there is a need for research that moves towards understanding how students interact with AI TAs, and especially how particular design choices shape those interactions.

One important design dimension is pedagogical guidance style. Socratic questioning is pedagogically appealing because it guides students toward discovering solutions themselves rather than providing direct answers [31]. Prior work in computing education has shown that Socratic techniques can support learning [28], and recent studies have explored whether LLMs can generate Socratic guidance for programming tasks [3, 18] and whether Socratic AI TAs can improve learning and retention [30]. For instance, Tran et al. found that slower-paced, Socratic-style AI TA interactions produced greater learning gains, particularly among students with less prior knowledge [30]. They also found that faster-paced interactions provided initial learning benefits for more advanced students, but these gains were accompanied by reduced retention. Our work complements Tran et al. by extending the study of AI TA guidance beyond interaction pacing and learning/retention outcomes to examine how pedagogical guidance style and context availability shape students’ perceptions, interaction behaviors, and immediate post-task comprehension. Additionally, previous work has found that students’ reception of Socratic feedback is mixed [1]. For example, Ahmed et al. found that students perceived Socratic hints as less helpful, less accurate, and more time-consuming than other feedback approaches, even when experts viewed the feedback more favorably [1]. This suggests a tension between pedagogical intent and student experience that warrants further investigation in conversational AI TA settings.

A second important design dimension is context awareness. Instructors have expressed a strong preference for customizing AI chatbots to fit their course-specific needs, including how and where the chatbot is deployed [13]. In programming courses, context may include the problem statement, the student’s current code, and the history of the interaction. Providing this context may allow an AI TA to ofer more relevant support, while withholding it may encourage students to articulate their problem more explicitly. Prior work has generally reported AI TA systems with fixed configurations (e.g. [8, 11, 20]) rather than isolating how specific configurations afect student perceptions and learning. Our study addresses this important gap by experimentally comparing AI TAs that vary across both pedagogical guidance style and context awareness.

## 3 Methods<sup>1</sup>

## 3.1 Study Design

Our study employed a between-subjects, 2×2 factorial parallel-arm randomized design [21] to examine the impact of AI TAs’ pedagogical guidance style (Socratic vs. Direct instruction) and context awareness (Full Context of the problem statement and student’s current solution vs. No Context) on students’ learning experiences. These experiences were operationalized through student perceptions, interaction behaviors, and task comprehension. The decision to manipulate these dimensions was motivated by prior work identifying task context and pedagogical strategy as important configurations preferred by instructors in AI TAs [13].

Students were asked to solve three programming problems (one code-writing and two code-debugging tasks) on a web-based coding platform with assistance from one of four randomly assigned AI TAs. Following completion of the assignment, all participants completed a survey that gauged their perceptions of the AI TA, their helpseeking behavior, and their comprehension of one of the debugging tasks. The study’s methodology and data collection procedures were reviewed and approved by the university’s Institutional Review Board (IRB).

## 3.2 Coding Tool and AI TA design

The study was implemented using a coding platform, which allows instructors to author and host programming questions (similar to [14]. A screenshot of the web-based platform is shown in Figure 1. The application has a split-screen interface with the problem statement, student submission history, “AI assistant” on the left, and an embedded text editor on the right. Students could run their own tests or submit their code to run against the instructor-defined test suite.

AI TAs were backed by the OpenAI GPT-4o LLM and returned responses to student queries based on custom prompts with pedagogical guardrails similar to CodeHelp [19]. The four AI TAs differed only in their system prompts and contextual information; the interface presented to students was otherwise identical. In the Full Context conditions, the chatbot’s system prompt was supplied with the problem statement, the student’s current code, and the chat history, whereas in the No Context conditions, it was supplied only with the student’s chat history. In the Direct conditions, the chatbot was instructed to provide brief, direct responses without producing complete code solutions, whereas in the Socratic conditions, it was instructed to guide students exclusively through targeted questions without providing direct answers. These two manipulations resulted in four AI TA configurations: (1) Direct + Full Context, (2) Direct + No Context, (3) Socratic + Full Context, and (4) Socratic + No Context.

![](images/fbf2cae606f414c6b999cf17efaf8ec5cc18f62a6d466fc7a69d0f316c025216.jpg)  
Figure 1: Student-facing Interface of the Web-based Coding Platform

## 3.3 Participants

Students were recruited from an introductory Python programming course (CS1) at a large public research university in the United States during the Fall 2025 semester. Enrolled students (� = 598) were not familiar with the coding platform used in the study, but were invited to participate in our study through in-person and learning management system announcements. They were ofered 0.5% extra credit toward their course grade for completing three programming tasks and a post-survey, and could receive the incentive while opting to exclude their data from research analysis. Data were collected during the final two weeks of the semester, by which point students were familiar with Python lists, loops, and conditionals.

Of the 598 students enrolled in the course, 254 completed the intent-to-participate pre-survey. After excluding non-consenting students, incomplete responses, students who did not use the chatbot, and cases with missing or inconsistent chatbot logs, the final analytic dataset consisted of 132 consenting students. Although participants were randomly and evenly assigned to the four conditions at the pre-survey stage, some imbalance occurred because not all students who initially expressed interest completed the study.

Table 1: Participant Distribution across 2 × 2 Experimental Conditions (� = 132)
<table><tr><td rowspan="2">Feedback Style</td><td colspan="2">Context Awareness</td><td rowspan="2">Row Total</td></tr><tr><td>Full Context</td><td>No Context</td></tr><tr><td>Direct Feedback</td><td>37</td><td>32</td><td>69</td></tr><tr><td>Socratic Feedback</td><td>30</td><td>33</td><td>63</td></tr><tr><td>Column Total</td><td>67</td><td>65</td><td>132</td></tr></table>

The final distribution of participants across conditions is shown in Table 1. Prior programming experience was broadly distributed: 24% reported being Experienced or Very experienced, 50% reported Average experience, and 25% reported being Inexperienced or Very inexperienced, using a self-estimation measure validated in prior work [10].

## 3.4 Study Tasks

We selected three tasks that had been clearly described in prior studies on AI TAs in computing education [6, 14]. The first task was a code-writing task in which students completed a function that determined whether a value was prime. The second task was a debugging task in which students fixed a function intended to determine whether a list contained repeated values. The third task was a debugging task in which students fixed a function that summed values surrounding a specific cell in a 10×10 matrix; the provided implementation had a boundary-checking bug. Function names were anonymized across tasks to reduce the extent to which the chatbot could infer the intended solution from the function name, which was important for evaluating the role of context awareness.

Table 2: Research Question Operationalization: Measures, Survey Items/Log Metrics, and Analysis Methods
<table><tr><td>Experience Dimension</td><td>Measures</td><td>Survey Item/Log Metric</td><td>Analysis Method</td></tr><tr><td>(a) Perceptions</td><td>Task completion† Task comprehension† Correct information</td><td>The [platform] chatbot helped me complete the tasks successfully. The [platform] chatbot helped me better understand the problem statements. The [platform] chatbot provided me with correct information. My experience using the [platform] chatbot was stressful.</td><td>Kruskal-Wallis H test Kruskal-Wallis H test Kruskal-Wallis H test Kruskal-Wallis H test</td></tr><tr><td>(b) Behaviors</td><td>Used another LLM# Reason for LLM use Average no. of chats Average chat length</td><td>Did you use another LLM or chatbot? Why did you use another LLM or chatbot? (if answered Yes&#x27; to previous question) Log data Log data</td><td>Chi-square test Inductive coding Kruskal-Wallis H test Kruskal-Wallis H test</td></tr><tr><td>(c) Task comprehension</td><td>Recall solution‡</td><td>In the tasks you completed, you were asked to debug a function bar() that takes in a 10×10 matrix (2D List) for input, then gives the sum of numbers surrounding a specific cell. Below is the code snippet provided, with the error still present. def bar(values, row, col): sum = 0 for i in range(row-2, row+3): for j in range(col-2, col+3):</td><td>Inductive coding</td></tr></table>

Note. Scale used: <sup>†</sup>5-point Likert scale (1 = Strongly Disagree, 5 = Strongly Agree). <sup>#</sup>Binary (Yes/No). <sup>‡</sup>Open-ended response.

Each student was randomly invited to one of four course shells in our tool, each with a diferent AI TA but the three identical tasks. Students were given one week to complete the study. Each task included ten instructor-provided test cases, and students were instructed to use the chatbot to help solve the problems.

Most students (128 out of 132) completed all three tasks, passing all test cases. Across all conditions, the average score was 2.98 out of $3 \ : ( m e d i a n = 3 , m i n = 1 , m a x = 3 , \sigma = 0 . 1 9 )$ ). Students initiated a total of 2,467 messages to the AI TA across the three problems (∼6 messages per student per problem).

## 3.5 Data Collection and Analysis

Data were collected from a post-survey and tool logs that recorded task completion and AI TA interaction behavior. Survey questions were informed by prior work in the community [6, 19]. Perception measures used 5-point Likert-scale items assessing whether the chatbot helped students complete the tasks, understand the problem statements, receive correct information, and whether the interaction was stressful. Behavioral measures included whether students used another LLM or chatbot, their reasons for doing so, the number of chat messages sent to the AI TA, and average message length. Task comprehension was assessed through an open-ended posttask response in which students re-identified and explained their solution to the boundary-checking bug from the matrix debugging task. Table 2 summarizes how each research question dimension was operationalized, including the corresponding survey items or log metrics and analysis methods.

Quantitative Analysis. Quantitative survey and log data were analyzed using descriptive statistics, followed by inferential tests. Diferences across the four experimental conditions were examined using the non-parametric Kruskal-Wallis H test, chosen because the Likert-scale ordinal responses and numerical log data did not follow a normal distribution (Shapiro-Wilk test, $\textstyle p < . 0 0 1 )$ ). Students’ use of another LLM was analyzed using a chi-square test. The significance level was set at $\alpha = . 0 5$ for all tests.

Qualitative Analysis. Responses to the open-ended questions on task comprehension and students’ reasons for using an alternative LLM were analyzed using inductive content analysis [23]. Student responses were inductively coded by two of the authors, with each author coding one of the open-ended questions. The initial codes were iteratively grouped and abstracted into higher-order categories. Two researchers reviewed all codes and categories to verify their consistency, with discrepancies resolved through discussion.

## 4 Results

## 4.1 Perception Measures

A Kruskal–Wallis H test revealed a statistically significant efect of condition on perceived support of the AI TA for task completion $( \chi ^ { 2 } ( 3 ) \ : = \ : 1 2 . 1 4 , \ : p \ : = \ : . 0 0 7 , N \ : = \ : 1 3 2 )$ . Students in the Socratic + Full Context condition rated the AI TA lowest on this measure (mean rank = 48.63, � = 3.53), substantially below the other three conditions: Direct + Full Context (mean rank $= 7 6 . 8 5 , \mu = 4 . 2 7 )$ Direct + No Context (mean rank $= 7 0 . 7 5 , \mu = 4 . 1 6 )$ , and Socratic + No Context (mean rank $. = 6 7 . 0 2 , \mu = 4 . 1 2 )$ , as shown in Figure 2.

Diferences across the four conditions were not statistically significant for perceived support for task comprehension $( \chi ^ { 2 } ( 3 ) = 4 . 7 9 $ $p = . 1 8 8 )$ , provision of correct information $( \chi ^ { 2 } ( 3 ) = 1 . 7 1 , p = . 6 3 5 )$ or stressful interaction $( \chi ^ { 2 } ( 3 ) = 4 . 5 0 , p = . 2 1 3 )$ . Descriptively, however, students in the Direct conditions rated the AI TA higher on task comprehension than those in the Socratic conditions, and students in the Socratic + Full Context condition rated their interaction as more stressful than students in the other three conditions.

![](images/d2536fcf643d0882767aee1be099bedc70132622c330e0f1d2dd50db8c84056e.jpg)  
Figure 2: Mean Student Ratings for AI TAs by Interaction of Context Awareness and Pedagogical Guidance Style (N=132) Note. Ratings measured on a 5-point Likert scale (1 = Strongly Disagree, 5 = Strongly Agree).

Perception findings: Students assigned to a Socratic AI TA with full context rated the AI TA least favorably across the perception measures. They reported significantly lower perceived support for task completion, and descriptively the lowest support for task comprehension and the highest stressful interaction rating.

## 4.2 Behavior

4.2.1 Using another LLM. Overall, 15% of students (20 of 132) reported using another LLM or chatbot during the experiment. Although usage rates difered descriptively across conditions, the diferences were not statistically significant $( \chi ^ { 2 } ( 3 ) = 2 . 4 7 , p = . 4 8 )$ Students in the Socratic + Full Context condition reported the high est rate of external LLM use (23%, 7 of 30 students), compared with 9–15% in the other three conditions.

We coded students’ open-ended explanations for why they used another LLM. Their reasons were abstracted into three not mutually exclusive categories: inadequate guidance $( n = 1 7 )$ , going in circles $( n = 4 ) _ { ; }$ , and forgetting context $( n = 2 ) .$ The most common concern was that the chatbot did not provide enough support, gave explanations that were too general or inflexible, or failed to work dynamically with the student. For example, one student in the So cratic + Full Context condition wrote that the chatbot “doesn’t really expand or work with the student very well” and was “super rigid”. Another student in the Socratic + No Context condition described the chatbot as using “circle reasoning”, where it repeatedly answered questions with more questions rather than helping with the code.

4.2.2 Interaction volume. A Kruskal–Wallis H test indicated a statistically significant diference in the number of queries students sent to the AI TA across the four conditions $( \chi ^ { 2 } ( 3 ) = 1 3 . 2 8 , p =$ .004). Students in the Socratic + No Context condition sent substantially more queries $( \mu = 1 1 . 1$ per problem) than students in the Socratic + Full Context (� = 5.1), Direct + Full Context $( \mu = 4 . 1 )$ and Direct + No Context $( \mu = 4 . 7 )$ conditions. This suggests that students who received Socratic guidance without problem context required more back-and-forth with the chatbot to make progress.

A Kruskal–Wallis H test also revealed a significant diference in average message length across conditions $( \chi ^ { 2 } ( 3 ) = 2 9 . 9 6 , p < . 0 0 1 )$ Students in the Direct + No Context condition wrote the longest messages on average (� = 126 characters per message), followed by Direct + Full Context (� = 81), Socratic + No Context (� = 69), and Socratic + Full Context $( \mu = 5 4 )$ . This pattern suggests that when the chatbot lacked context, students often had to describe their situation in more detail before receiving useful support.

Behavior findings: Behavioral engagement difered significantly across conditions in query volume and message length. Students in the Socratic + No Context condition sent the most queries per problem, while students in the Socratic + Full Context condition sent the shortest messages on average. Descriptively, students in the Socratic + Full Context condition also reported the highest, though non-significant, rate of external LLM use.

## 4.3 Task Comprehension Analysis

We analyzed students’ responses to a post-task survey question, administered immediately after task completion, asking them to re-identify the error and explain their solution for Task 3 (“Surrounding Sum”). This assessment reflects students’ immediate comprehension of the problem alongside their own debugging solution, rather than a longer-term retention of the underlying concept, and allowed us to examine how the AI TA’s guidance style and context awareness may have shaped that comprehension. Responses were coded into four comprehension categories (see Figure 3). Full comprehension required students to identify the indices out-of-bounds bug and explain a correct fix. Partial/impure comprehension captured responses that identified the correct bug but had incomplete, unclear, or faulty reasoning. Minimal comprehension captured responses that did not identify the out-of-bounds bug. Ungaugable comprehension captured responses where it was unclear whether they had identified the bug.

![](images/ae179372c1360ccb6be79f3199005629dc9afd62217408ba9f68385b4c8e544e.jpg)  
Figure 3: Decision Flowchart for Task Comprehension

![](images/73a7ec1b91133368aa0593c40cf25ad3a698324bbbea37005bcf644ba404f6a2.jpg)  
Figure 4: Comprehension Level by Group Assignment

Across all non-blank responses (� = 128), 57% students showed full comprehension, 30% showed partial/impure comprehension, 11% showed minimal comprehension, and 2% were ungaugable. When broken down by condition, the Direct + Full Context group had the highest proportion of full-comprehension responses (67%), followed by Direct + No Context (61%), Socratic + No Context (50%), and Socratic + Full Context (48%). The Socratic + Full Context group also had the highest proportion of minimal-comprehension responses (14%, see Figure 4). Overall, comprehension patterns appeared to difer more by pedagogical guidance style than by context awareness: Direct conditions produced a higher average proportion of full-comprehension responses than Socratic conditions.

Task comprehension findings: Descriptively, students in the Direct conditions showed a higher proportion of responses that demonstrated full-comprehension than students in the Socratic conditions. The Direct + Full Context condition had the highest proportion of full-comprehension responses, while the Socratic + Full Context condition had the lowest.

## 5 Discussion

Role of Pedagogical Guidance Style: Our results suggest that the AI TA’s guidance style shaped how students engaged with the tool. Students in the Socratic conditions sent the most queries, suggesting that a chatbot that responds with guiding questions rather than direct answers can require more back-and-forth before students make progress. In this sense, Socratic guidance appeared to increase interaction volume as observed in other domains outside of computing [29], especially when the chatbot lacked task context.

However, the perception and comprehension results tell a more cautionary story. Students rated the Socratic + Full Context condition least favorably for supporting task completion, and this condition also had the lowest proportion of full-comprehension post-task explanations. This raises an important tension for educators considering AI TA deployment. A pedagogically motivated approach that encourages students to reason through problems independently may still be experienced as frustrating or unhelpful (similar to results from Ahmed et al. [1]), especially when students are focused on completing a programming task. Educators should therefore consider whether the goal of a given assignment is task completion, conceptual understanding, or productive struggle, and configure the AI TA’s guidance style accordingly.

Role of Contextual Awareness: Context awareness also played a notable role in shaping student behavior. Students in the No Context conditions, where the chatbot had access only to the chat history and not to the problem statement or student’s current code, tended to write longer individual queries than those in the Full Context conditions. This suggests that when the chatbot lacked information about the problem, students compensated by providing more detail in their messages, essentially doing the work of explaining their situation before the chatbot could help them.

At the same time, Full Context did not guarantee a better student experience. The Socratic + Full Context condition produced the shortest messages on average, suggesting that students did not need to explain much before receiving a response. Yet this same condition was rated least favorably for task completion and had the lowest proportion of full-comprehension responses on the posttask. This points to a practical implication for educators, which is that providing an AI TA with full task context may reduce the efort required to formulate queries, but contextual awareness alone does not ensure that the support will feel useful or lead to stronger understanding. Instructors deploying AI TAs in programming courses should therefore consider integrating the tool with the assignment environment while also carefully designing how the chatbot uses that context pedagogically.

Interaction Between Pedagogical Guidance Style and Context Awareness: The interaction plots in Figure 2 show why guidance style and context awareness should not be interpreted independently. When Direct guidance was available, students rated the chatbot similarly for supporting task completion regardless of how much context was provided. In contrast, when the chatbot was Socratic, providing full context reduced students’ perceived support for task completion. This suggests that context awareness was not uniformly beneficial, and that its efect depended on the pedagogical style with which it was paired.

The behavioral results reinforce this interpretation. The Socratic + No Context condition produced the highest query volume, which makes intuitive sense because when a chatbot avoids direct answers and also lacks knowledge of the task, students must work harder to get useful help. The result may be a high-efort interaction that feels more exhausting than educational, particularly for students who are already struggling. In contrast, the Direct + Full Context condition produced the lowest query volume, suggesting that students could get help quickly when the chatbot provided direct guidance and already had access to the problem and their current code.

These findings suggest that guardrails can become roadblocks when pedagogical guidance and contextual awareness are not well balanced. A Socratic chatbot with limited context may demand too much efort from students, while a highly contextualized chatbot that remains rigidly Socratic may still fail to provide the kind of support students find useful, as reflected in the weaker descriptive post-task comprehension pattern for the Socratic + Full Context condition. Rather than treating “more scafolding” or “more context” as inherently better, educators and system designers should consider AI TAs as configurable learning supports whose efectiveness depends on the fit between the task, the student’s needs, and the form of help provided. Future work should explore adaptive AI TAs that can shift between Socratic and more direct guidance based on students’ progress, confusion, or help-seeking behavior.

Limitations: Our study was conducted in an introductory programming course at a large research university. Although this controlled setting supports high internal validity, it may limit the generaliz ability of the findings to other contexts. Another limitation is that we measured task comprehension for only one task, a design choice we made to reduce survey fatigue. This design captures immediate comprehension rather than knowledge retention. Future studies could investigate the impact of AI TA configurations on long-term knowledge retention. Finally, qualitative coding of open-ended responses involves some degree of subjectivity; we mitigate this by developing a codebook, transparent reporting of our process, and by providing a detailed flowchart of our approach.

## 6 Conclusion

AI TAs are increasingly being deployed in programming courses to provide scalable help while using pedagogical guardrails to avoid simply giving students complete solutions. However, our findings show that not all guardrailed designs are experienced by students as helpful — their efect depends on pedagogical guidance style and context awareness. In this study, we moved beyond evaluating a single AI TA system as a complete intervention (similar to [15, 17, 19, 20]) by experimentally comparing four configurations of the same AI TA, varying both guidance style and context awareness. Across students’ perceptions, interaction behaviors, and post-task comprehension, the Socratic + Full Context condition showed the least favorable descriptive pattern. Students rated it significantly lower in perceived support for task completion, and it also showed less favorable descriptive patterns for perceived comprehension support, higher interaction stress, greater external LLM use, and lower post-task comprehension. These results highlight a key design challenge for AI TAs in programming courses, which is that pedagogically motivated configurations can become roadblocks if students experience them as too rigid or insuficiently useful. Efective AI TA design will therefore require balancing pedagogical guidance with contextual awareness in ways that support learning while keeping students engaged with course-approved tools.

## References

[1] Umair Z. Ahmed, Shubham Sahai, Ben Leong, and Amey Karkare. 2025. Feasi bility Study of Augmenting Teaching Assistants with AI for CS1 Programming Feedback. In Proceedings ofthe 56th ACM Technical Symposium on Computer Science Education V. 1 (Pittsburgh, PA, USA) (SIGCSETS 2025). ACM, New York, NY, USA, 11–17. doi:10.1145/3641554.3701972

[2] Gökhan Akçapınar and Elif Sidan. 2024. AI chatbots in programming education: guiding success or encouraging plagiarism. Discover Artificial Intelligence 4, 1 (2024). doi:10.1007/s44163-024-00203-7 Cited by: 16; All Open Access, Gold Open Access.

[3] Erfan Al-Hossami, Razvan Bunescu, Justin Smith, and Ryan Teehan. 2024. Can Language Models Employ the Socratic Method? Experiments with Code Debug ging. In Proceedings ofthe 55th ACM Technical Symposium on Computer Science Education V. 1 (Portland, OR, USA) (SIGCSE 2024). ACM, New York, NY, USA, 53–59. doi:10.1145/3626252.3630799

[4] Shawheen Alipour, Leslie Coward, Justin Burris, Bishnu Karki, and Tony Liao. 2025. Evaluating an AI Teaching Assistant in VR Physics Education: A Case Study of User Experience and Efectiveness. In 2025 IEEE International Symposium on Emerging Metaverse (ISEMV). 62–68. doi:10.1109/ISEMV67326.2025.00021

[5] Jessica Y Bo, Majeed Kazemitabaar, Mengqing Deng, Michael Inzlicht, and Ashton Anderson. 2026. Invisible Saboteurs: Sycophantic LLMs Mislead Novices in Problem-Solving Tasks. In Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems (CHI ’26). ACM, New York, NY, USA, Article 1258, 31 pages. doi:10.1145/3772318.3791365

[6] Paul Denny, Stephen MacNeil, Jaromir Savelka, Leo Porter, and Andrew Luxton-Reilly. 2024. Desirable Characteristics for AI Teaching Assistants in Programming Education. In Proceedings of the 2024 ACM Conference on Innovation and Technology in Computer Science Education V.1 (Milan, Italy, July 8-10, 2024) (ITiCSE ’24). ACM, New York, NY, USA, 7 pages. doi:10.1145/3649217.3653574

[7] Paul Denny, James Prather, Brett A. Becker, James Finnie-Ansley, Arto Hellas, Juho Leinonen, Andrew Luxton-Reilly, Brent N. Reeves, Eddie Antonio Santos, and Sami Sarsa. 2024. Computing Education in the Era ofGenerative AI. Commun. ACM 67, 2 (Jan. 2024), 56–67. doi:10.1145/3624720

[8] Marc Diaz, Dustin Karp, Prayuj Tuli, and Amanpreet Kapoor. 2025. Edugator: An AI-enabled Tool for Creating and Delivering Interactive Computing Content. In Proceedings ofthe 56th ACM Technical Symposium on Computer Science Education V. 2 (Pittsburgh, PA, USA) (SIGCSE TS 2025). ACM, New York, NY, USA, 1732. doi:10.1145/3641555.3705025

[9] Hao Fan, Wenqin Du, Jia Yan, and Hong Jiang. 2025. Application of DeepSeekbased AI teaching assistant in teaching anesthesiology theories. BMC Medical Education (2025).

[10] Janet Feigenspan, Christian Kästner, Jörg Liebig, Sven Apel, and Stefan Hanenberg. 2012. Measuring programming experience. In 2012 20th IEEE International Conference on Program Comprehension (ICPC). 73–82. doi:10.1109/ICPC.2012. 6240511

[11] Tony Haoran Feng, Stefan Hooper, Paul Denny, Burkhard C. Wünsche, and Andrew Luxton-Reilly. 2026. Bot or Prof: Comparing the Characteristics of AI generated and Instructor-written Responses to Student Queries in an Introductory Programming Course Forum. In Proceedings ofthe 28th Australasian Computing Education Conference (ACE ’26). ACM, New York, NY, USA, 63–72. doi:10.1145/ 3786228.3786238

[12] Arto Hellas, Juho Leinonen, Sami Sarsa, Charles Koutcheme, Lilja Kujanpää, and Juha Sorva. 2023. Exploring the Responses of Large Language Models to Beginner Programmers’ Help Requests. In Proceedings ofthe 2023 ACM Conference on International Computing Education Research V.1 (Chicago, IL, USA, August 7-11, 2023) (ICER ’23 V.1). ACM, New York, NY, USA, 13 pages. doi:10.1145/3568813. 3600139

[13] Irene Hou, Zeyu Xiong, Philip J. Guo, and April Yi Wang. 2026. “Bespoke Bots”: Diverse Instructor Needs for Customizing Generative AI Classroom Chatbots. In Proceedings ofthe 2026 CHI Conference on Human Factors in Computing Systems (CHI ’26). ACM, New York, NY, USA, Article 705, 10 pages. doi:10.1145/3772318. 3791491

[14] Amanpreet Kapoor, Paul Denny, Leo Porter, Stephen MacNeil, and Marc Diaz. 2026. Exploring Student Behaviors and Motivations when using AI Teaching Assistants with Optional Guardrails. In Proceedings ofthe 28th Australasian Computing Education Conference (ACE ’26). Association for Computing Machinery, New York, NY, USA, 22–31. doi:10.1145/3786228.3786233

[15] Majeed Kazemitabaar, Runlong Ye, Xiaoning Wang, Austin Zachary Henley, Paul Denny, Michelle Craig, and Tovi Grossman. 2024. CodeAid: Evaluating a Classroom Deployment of an LLM-based Programming Assistant that Balances

Student and Educator Needs. In Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems (Honolulu, HI, USA) (CHI ’24). ACM, New York, NY, USA, Article 650, 20 pages. doi:10.1145/3613904.3642773

[16] Shao-Heng Ko, Matthew Zahn, Kristin Stephens-Martinez, Yesenia Velasco, Lina Battestilli, and Sarah Heckman. 2025. Relationships Between Computing Students’ Characteristics, Help-Seeking Approaches, and Help-Seeking Behavior in Introductory Courses and Beyond. In Proceedings ofthe 2025 ACM Conference on International Computing Education Research V.1 (ICER ’25). ACM, New York, NY, USA, 313–326. doi:10.1145/3702652.3744214

[17] Huiyong Li and Boxuan Ma. 2025. CodeRunner Agent: Integrating AI Feedback and Self-Regulated Learning to Support Programming Education. In International Conference on Computers in Education.

[18] Zhixian Christopher Liding, Michael Osmolovskiy, Harshith Lanka, Ronnie Howard, Nimisha Roy, and Rodrigo Borela. 2026. To Tell or to Ask? Com paring the Efects of Targeted vs. Socratic AI Hints. In Proceedings ofthe 57th ACM Technical Symposium on Computer Science Education V.2 (USA) (SIGCSE TS 2026). Association for Computing Machinery, New York, NY, USA, 1423–1424. doi:10.1145/3770761.377732

[19] Mark Lifiton, Brad E Sheese, Jaromir Savelka, and Paul Denny. 2024. CodeHelp: Using Large Language Models with Guardrails for Scalable Support in Program ming Classes. In Proceedings ofthe 23rd Koli Calling International Conference on Computing Education Research (Koli, Finland) (Koli Calling ’23). ACM, New York, NY, USA, Article 8, 11 pages. doi:10.1145/3631802.3631830

[20] Rongxin Liu, Carter Zenke, Charlie Liu, Andrew Holmes, Patrick Thornton, and David J. Malan. 2024. Teaching CS50 with AI: Leveraging Generative Artificial Intelligence in Computer Science Education. In Proceedings of the 55th ACM Technical Symposium on Computer Science Education V. 1 (Portland, OR, USA) (SIGCSE 2024). ACM, New York, NY, USA, 750–756. doi:10.1145/3626252.3630938

[21] I Scott MacKenzie. 2012. Human-computer interaction: An empirical research perspective. Morgan Kaufmann.

[22] Nishat Raihan, Mohammed Latif Siddiq, Joanna C.S. Santos, and Marcos Zampieri. 2025. Large Language Models in Computer Science Education: A Systematic Literature Review. In Proceedings of the 56th ACM Technical Symposium on Computer Science Education V. 1 (Pittsburgh, PA, USA) (SIGCSETS 2025). ACM, New York, NY, USA, 938–944. doi:10.1145/3641554.3701863

[23] Johnny Saldana. 2011. Fundamentals of qualitative research. Oxford university press.

[24] Sami Sarsa, Paul Denny, Arto Hellas, and Juho Leinonen. 2022. Automatic Generation of Programming Exercises and Code Explanations Using Large Language Models. In Proceedings of the 2022 ACM Conference on International Computing Education Research - Volume 1 (Lugano and Virtual Event, Switzerland) (ICER ’22). ACM, New York, NY, USA, 27–43. doi:10.1145/3501385.3543957

[25] Brad Sheese, Mark Lifiton, Jaromir Savelka, and Paul Denny. 2024. Patterns of Student Help-Seeking When Using a Large Language Model-Powered Programming Assistant. In Proceedings ofthe 26th Australasian Computing Education Conference (Sydney, NSW, Australia) (ACE ’24). ACM, New York, NY, USA, 49–57. doi:10.1145/3636243.3636249

[26] Esther Shein. 2024. The Impact of AI on Computer Science Education. Commun. ACM 67, 9 (Aug. 2024), 13–15. doi:10.1145/3673428

[27] Kewei Shi, Peidong Xu, Jun Zhang, Fei Tang, Xiangtao Zhuan, Xuzhu Dong, and Yuanfeng Chen. 2026. Research on the Impact of AI Teaching Assistants on Electrical Engineering Courses—A Case Study of the Course “Exploring the World of Electricity”. In The Proceedings of the International Council on Electrical Engineering Conference 2025(ICEE 2025), Xuzhu Dong (Ed.). Springer Nature Singapore, Singapore, 273–282.

[28] Lasang Jimba Tamang, Zeyad Alshaikh, Nisrine Ait Khayi, Priti Oli, and Vasile Rus. 2021. A Comparative Study of Free Self-Explanations and Socratic Tutoring Explanations for Source Code Comprehension. In Proceedings ofthe 52nd ACM Technical Symposium on ComputerScience Education (Virtual Event, USA) (SIGCSE ’21). ACM, New York, NY, USA, 219–225. doi:10.1145/3408877.3432423

[29] Augusto Grimaldi Ter Sarkisian, Cameron Miller, Neeraja Poonjolai, Candelaria Aramburu, and Jad Atweh. 2026. Albert or Alberta? Investigating Socratic Dialogue and Modality in AI-Assisted Engineering Problem Solving. In 2026 Systems and Information Engineering Design Symposium (SIEDS). 1–6. doi:10.1109/SIEDS69358.2026.11540271

[30] Karena Tran, Ge Gao, Angela Lombard, Tyler Yu, Haoning Jiang, and Thomas Y. Yeh. 2026. Pacing for Mastery: Optimizing LLM Interactions for Learning. In Proceedings ofthe 57th ACM Technical Symposium on Computer Science Education V.1 (USA) (SIGCSE TS 2026). Association for Computing Machinery, New York, NY, USA, 1068–1074. doi:10.1145/3770762.3772501

[31] Lev S. Vygotsky and Alex Kozulin. 2012. Thought and Language, revised and expanded edition. MIT Press.

[32] Anlan Wang and Xingting Wu. 2025. Building a triadic model of technology, motivation, and engagement: a mixed-methods study of AI teaching assistants in design theory education. Frontiers in Psychology Volume 16 - 2025 (2025). doi:10.3389/fpsyg.2025.1624182

[33] Xingyu Wang, Liwei Zhang, Yu Mao, Duncan J. McGillivray, and Ziyun Wang. 2025. ChEdu: A Guided AI Teaching Assistant for Chemistry Education and Exam Support. Journal ofChemical Education 102, 12 (2025), 5347–5354. doi:10. 1021/acs.jchemed.5c00036

[34] J. D. Zamfirescu-Pereira, Laryn Qi, Björn Hartmann, John DeNero, and Narges Norouzi. 2025. 61A Bot Report: AI Assistants in CS1 Save Students Homework Time and Reduce Demands on Staf. (Now What?). In Proceedings ofthe 56th ACM Technical Symposium on Computer Science Education V.1 (Pittsburgh, PA, USA, February 26-March 1, 2025) (SIGCSE TS ’25). ACM, New York, NY, USA, 7 pages. doi:10.1145/3641554.3701864