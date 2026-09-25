# Beyond Simple Input-Output Assessment Tasks: Leveraging Automated Programming Assessment for Non-Trivial Courses

Artur Jordao

Escola Politecnica, Universidade de S´ ao Paulo˜

## Abstract

The public visibility of Artificial Intelligence (AI) is growing rapidly, driven by the positive impact of its applications across diverse fields of knowledge. In this new chapter, courses that cover the foundations of AI and machine learning become essential for understanding their role and potential in contemporary society. Therefore, understanding fundamental concepts and elementary algorithms through the close integration of theory with practice is essential in AI courses. In this essay, we report our experience designing machine learning exercises for automated assessment tools in programming. It is worth mentioning that we are not developing a novel form of automated grading system. Instead, we propose a perspective that frames machine learning problems as inputoutput assessment tasks. From this perspective, each exercise admits a unique and deterministic answer and enables automated programming assessment tools (e.g., VPL for Moodle, Codeforces, and MOJ) to effectively support AI education. We believe this essay can encourage instructors to foster educational innovation by adopting more dynamic and interactive approaches to AI courses that integrate theory and practice. Importantly, this essay does not introduce an innovation in the use ofAIfor education; rather, it introduces an innovative approach to improving the learning of AI, particularly, machine learning.

## 1. Introduction

Artificial Intelligence (AI) is gaining greater public visibility, driven in particular by the positive impact of generative AI across diverse fields (Bengio & et al., 2025; Maslej & et al., 2025). As a concrete example, AI tools improve workplace efficiency by automating costly tasks, reducing operational costs, and increasing productivity (Dell’Acqua & et al., 2023). The fields of teaching and education also experience broader adoption of AI (Maslej & et al., 2025).

Given the current AI wave and its extensive integration across emerging technologies, courses covering the foundations of artificial intelligence, machine learning, and deep learning become increasingly sought-after and essential for understanding the role and potential of AI in contemporary society (Maslej & et al., 2025). For example, the report by Maslej et al. (Maslej & et al., 2025) found that the number of U.S. institutions offering a dedicated bachelor’s degree in AI nearly doubled between 2022 and 2023. The number of institutions offering dedicated master’s degrees in AI also grew sharply. In this context, hands-on practice through exercises reinforces the underlying concepts while enhancing analytical and problem-solving skills in AI. Additionally, closely aligning theory with practice becomes essential for formulating algorithmic solutions and implementing them through programming languages.

In this essay, we report our experience designing machine learning exercises for automated programming assessment tools. We leverage the renowned International Collegiate Programming Contest (ICPC) as inspiration to formulate practical exercises covering specific machine learning fundamentals (ICPC Foundation). For AI practitioners and instructors, a natural question is: why not host a competition on the Kaggle platform? It turns out that Kaggle competitions allow students (participants) to solve the same problem or task using different machine learning concepts while focusing primarily on the final evaluation metric (i.e., accuracy). Here, on the other hand, we design exercises covering specific machine learning topics, each admitting a unique and deterministic answer. For example, estimating the explained variance captured by the first principal components of PCA or identifying the support vectors given an SVM hyperplane. For this purpose, we transform machine learning fundamentals into input-output assessment tasks suitable for automated programming assessment tools. Figure 1 illustrates an overview of this idea. Powered by these tools, our approach offers clear advantages, including immediate feedback for students and reduced workload for instructors and teaching assistants. We highlight that this input-output assessment strategy is not new and is already common in introductory courses on programming, algorithms and data structures. Our contribution lies in extending this paradigm

![](images/e8b3332cdeb2da025769d7ec5eff5e5657ba530ed07749165c65b30aec7a8db9.jpg)  
Figure 1. Examples of two classical Machine Learning fundamentals mapped to input-output format.

to machine learning courses.

At the time of writing, we release the exercises through Virtual Programming Lab (VPL) for Moodle, but the same exercises can also run on other automated programming assessment platforms, such as MOJ, Codeforces, or URI Online Judge. Overall, as we shall see, we had a successful experience with this practice, with students providing positive and enthusiastic feedback and largely agreeing that the exercises effectively integrate theory and practice.

## 2. Methodology

Machine Learning Concepts to Input-Output Format. The key to successfully employing automated programming assessment platforms lies in effectively translating machine learning fundamentals into an input-output format. Figure 1 introduces two examples of exercises we created for SVM and PCA. Following these examples, we also include exercises on k-nearest neighbors classifier, ordinary least squares, hierarchical clustering, decision tree, z-score and min-max normalization.

Due to environment constraints (i.e., VPL), not all techniques support execution from scratch. Specifically, while the environment we use in 2026 supports SVD through NumPy, it does not support quadratic programming or libraries such as scikit-learn, thus preventing students from obtaining some values from scratch (i.e., through a training process) such as the SVM hyperplane. In these cases, we adopt the following strategy. First, we run the technique locally in an environment that provides all the required dependencies and compute the necessary values.

We then use these values as inputs to the exercise. For example, for SVM, we learn the hyperplane parameters w and b locally and provide these values as inputs to the exercise (see Figure 1 left).

Exercise Writing. When translating a concept into an inputoutput format and designing an exercise, we must consider aspects beyond the core objective to clearly specify how students should solve the problem and avoid cases where different valid approaches produce incompatible outputs. It turns out that some machine learning techniques admit multiple valid solutions, such as computing PCA using either SVD or eigendecomposition of the covariance matrix. Keep in mind that we are discussing deterministic inputoutput execution on automated programming assessment platforms.

The naive solution to the previous issue is to explicitly define how students should solve the problem, e.g., “To compute the SVD, use np.linalg.svd(., full matrices=False)”. The final ingredient when writing an exercise is to address numerical precision. In a similar vein, we explicitly define the required numerical precision for specific calculations or outputs, e.g., “use the np.isclose(., atol=1e-3)function provided in the code”.

On the Importance of Determinism and the Role of Pseudo-random Numbers. Many machine learning techniques have a stochastic nature. For example, k-means clustering uses random initial centroids, while random forests rely on feature and sample subsampling, among other sources of randomness. Since automated programming assessment tools require deterministic outputs, we must ensure algorithms produce deterministic results. Concretely, executing a machine learning algorithm on a given set of inputs must produce the same output.

Fortunately, most Python packages involving randomness provide a mechanism for controlling the pseudo-random number generator through a seed, ensuring deterministic (reproducible) results. Therefore, to address the previous issue, we leverage this feature and set the random seed for the packages involved in each exercise (as we shall see, we restrict what packages a student can use). Two options emerge here: (i) provide the seed as a fixed value in the code template, or (ii) provide the seed as an additional input (see the code in Figure 1 right). We opt for the latter to ensure that students do not inadvertently modify the seed.

Another important role of pseudo-random numbers is to simplify the test cases in exercise definitions. It turns out that, describing input-output examples for many machine learning concepts can become laborious and even hinder the understanding of students. By leveraging pseudo-random numbers, we can easily generate large, representative examples by providing a seed instead of specifying each number individually. For example, providing a seed that generates huge matrices representing the independent (data) and dependent (label) variables, along with the expected behavior. Figure 1 (right) illustrates this idea.

On the Importance of Providing a Template Code. Since we focus on practicing deep concepts rather than basic programming, algorithms, or data structures, we provide a code template for each exercise to minimize the programming overhead for students. In addition, the code template helps prevent students from using unauthorized solutions and guides them in specifying how to solve the problem.

In a code template, we often provide instructions for reading the inputs (including the seed), importing the required packages, and handling numerical precision when producing the output. Depending on the complexity of the exercise, we also specify which sections of the code or classes students must complete. The code below provides an overview of a general code template.

```python
import numpy as np
import random
s, n, m = map(int, input().split())
np.random.seed(s)
random.seed(s)
X = np.random.randn(n, m)
```  
Figure 2. SAGE default prompt. We fill the markers [The Start of Exercise Statement] and [The Start of Code Submitted] with the exercise statement (i.e., Figure 1 left or right) and the student code, respectively.

Large Language Models to Mitigate Unauthorized Solutions - SAGE. While the code template provides a first line of defense against unauthorized solutions, it is clearly not sufficient on its own. Specifically, a student could employ libraries or packages that would easily solve a problem, thus

## Prompt for SAGE

[System] Evaluate the student code against the question statement. Assign an integer score from 0 to 10, considering exclusively how correctly and exactly the code fulfills what the statement requested, including the required format. Consider: 10: fulfills the statement and the requested format completely. 0: does not fulfill what was requested or presents an essentially incorrect solution. Intermediate values: partially fulfill the statement or present errors that compromise part of the solution. Do not penalize the student for aspects that the statement did not explicitly request. Your response MUST be only an integer number between 0 and 10, without additional text, explanations, or punctuation.   
Mandatory format: \b(?:10|[0-9])\b   
Example of a valid response: 8   
[The Start of Exercise Statement]   
{Exercise Statement}   
[The Start of Code Submitted]   
{Code Submitted}

compromising the goal of practicing the concepts targeted by the exercise. A naive approach to this issue involves manually inspecting each submission for potential violations or unauthorized solutions. Unfortunately, even in medium-size classes, manually checking every submission is infeasible.

Given the success of Large Language Models (LLMs) in coding and recent LLM-as-a-Judge benchmarks (Team, 2025; Zheng & et al., 2023), to address the previous challenge, we developed SAGE: an LLM-based system for additional evaluation of student solutions. The idea behind SAGE consists of reading both the exercise statement and the submitted code of a student and assigning a score that reflects whether the student respected all the constraints specified in the problem. From this score, the instructor can identify which student submissions require inspection outside the automated programming assessment platform.

SAGE operates independently of the automated programming assessment platform, running locally to analyze student submissions. We initially developed SAGE using the Gemini API, but its current version employs LM Studio<sup>1</sup>. We believe this approach facilitates broader adoption of SAGE, as LM Studio supports a wide range of models and enables fully local, cost-free inference. SAGE is available for download at this URL

To help LLMs identify the constraints and what students must respect, we explicitly state these requirements in a “Note:” section at the end of the exercise statement. Figure 2 illustrates an example of prompt of SAGE.

Tools and Requirements. At this point, we hope the reader

<sup>1</sup>https://lmstudio.ai/

understands that most exercises depend on Python packages such as NumPy. Indeed, only very simple tasks, such as basic normalization, do not require such external (non-native) libraries. In particular, at the date of this essay, our environment requires only the following Python packages: random, itertools, NumPy, and SciPy. These packages allow us to cover a broad range of fundamental machine learning concepts. As a promising avenue for future exploration, we plan to develop exercises involving more elaborate concepts, including deep learning with TensorFlow and PyTorch.

As a final note, while we employ VPL as our automated programming assessment platform, other platforms can naturally host these exercises, given their simple input-output format, similar to ICPC contests. The platform only needs to provide the essential Python packages mentioned above.

Experience in an Operating Systems Course. Besides machine learning, we also have two years of experience implementing the input-output assessment strategy in an Operating Systems course. In this course, the exercises cover topics such as process and memory management and parallel concurrency (including threads, mutexes, and semaphores). For these topics, we primarily use C/C++ and Assembly x86. Figure 3 illustrates an example of a problem involving mutual exclusion with mutex.

Unlike machine learning, some operating systems exercises rely heavily on SAGE. In particular, students can easily obtain the expected output by manipulating the input in ways that diverge from the core learning objectives of the exercise. We invite the reader to reflect on how, without the support of SAGE, the exercise in Figure 3 is susceptible to potential violations<sup>2</sup>. In this direction, exercises involving pointers require special attention, as students may bypass the intended concepts by using alternative mechanisms, such as registers. These challenges motivate our ongoing efforts to improve SAGE.

![](images/4f4d8e903c6a1885778afb4552b51d5a78c401b0a498f97204e12c21c936b246.jpg)  
Figure 3. Example of an important Operating Systems concept (mutual exclusion) mapped to an input-output format.

Student Feedback. While our strategy for mapping machine learning problems into input-output assessment tasks may sound particularly interesting to machine learning practitioners and instructors, its primary goal is to bridge the gap between theory and practice and enhance student learning. In Figure 4, we summarize the main feedback provided by students across two different courses, including Operating Systems, from 2025 to 2026.

According to Figure 4 (left), more than 60% of students fully agree that the exercises improve the integration of theory and practice. In addition, the responses clearly concentrate in the mid-to-high score range, with only one student indicating that the exercises are not helpful. Figure 4 (middle) reinforces the importance of input-output examples (the tables in Figures 1 and 3), with responses showing a strong, unimodal peak at the maximum score (10). Indeed, these examples play a vital role in understanding the problem, as in ICPC contests, and the results further support their importance. Finally, Figure 4 (right) illustrates an interesting behavior when we ask students whether they prefer human or automated grading. From this figure, we observe that responses are more dispersed suggesting greater variability in student opinions. On the one hand, the results in Figure 4 (right) indicate a clear preference for automated assessment among students. We believe this preference stems from the fast feedback provided by these tools. On the other hand, the concrete scores in the middle range demonstrate that students also appreciate having a human in the loop. Indeed, automated assessment in non-trivial courses can help reallocate instructional time toward higher-impact pedagogical activities, including individualized student tracking.

Overall, Figure 4 confirms positive student acceptance and suggests a successful experience in integrating theory and practice through automated assessment tools in non-trivial courses we adopt (machine learning and operating systems).

Besides the previous categorical (0-10) feedback, we also asked students to provide open-ended comments about their experience with the activities. Figure 5 summarizes student feedback and further supports the effectiveness of our strategy.

<sup>2</sup>If you have spent more than five minutes thinking about the problem, you may be overthinking it—the answer is simply to multiply the inputs.

![](images/0d6f941be93fb4bc870ed036909d681a0865df190f0e1320a729adc452027676.jpg)

![](images/c3177ae3b0c6483d67e5ba0ddb8ed208108e565c348f28b0adf6db5e3d253fe1.jpg)

![](images/7ebf2b1a3cfebaf23cb9c9708b05bb32d1bce5490d4cac416eea1f72bcaa1b97.jpg)  
Figure 4. Left. Answer distribution for the question: Do you believe the automated grading dynamic helped align practical concepts with theory? Scale: 0 (not helpful) to 10 (completely helpful). Middle. Answer distribution for the question: How important do you consider the inclusion ofinput-output examples in the exercise statements? Scale: 0 (not necessary) to 10 (completely necessary). Right. Answe distribution for the question: Do you prefer automated or human grading? Scale: 0 (human only) to 10 (automated only).

![](images/08ed19e7ff7566d69d94ed7231ffb864165ea6f42b3c6d70666d09b878bed568.jpg)  
Figure 5. Open-ended student comments about their experience with the activities. We translated the student feedback into English, preserving the original meaning as closely as possible.

## 3. Final Remarks

In this essay, we report a strategy for integrating theory and practice in non-trivial courses, with a focus on machine learning. Overall, our strategy transforms exercises on machine learning concepts into an input-output format. Leveraging this mapping, each exercise admits a unique and deterministic answer and supports the use of automated programming assessment tools. In particular, in this work, we employ Virtual Programming Lab (VPL) for Moodle as our automated programming assessment tool, but the approach is platform-agnostic and thus supports other tools, such as MOJ, Codeforces, and URI Online Judge. Throughout this essay, we discuss guidelines that can help instructors design effective practical exercises and avoid common issues. We also introduce a large language model–driven inspection mechanism to verify that submissions satisfy the required skill competencies without policy violations, thereby enhancing assessment reliability. We reinforce that this essay does not introduce an innovation in the use of AI for education; rather, it introduces an innovative approach to improving the learning ofAI, particularly, machine learning.

The qualitative and quantitative results indicate strong student acceptance, with students largely agreeing that the exercises effectively integrate theory and practice through automated assessment. Despite this successful experience and positive feedback, our strategy still has room for improvement. The first is to expand the platform support to additional packages such as scikit-learn. These packages would enable the development of exercises covering more advanced machine learning concepts. Second, some students point out that the messages for failed test cases (the output does not match the expected value) are uninformative. Addressing this issue is not straightforward, since overly explicit messages, such as “the expected output is:”, could make the problem trivial or irrelevant. We highlight that this is not specific to the platform we use (VPL) and also affects ICPC-like contests (ICPC Foundation).

Finally, regarding our strategy for inspecting each submission for potential violations or unauthorized solutions, LLMbased analysis remains susceptible to errors and missed violations. Although our experience with larger models (e.g., 7B-29B parameters) suggested high accuracy on this task, they also impose a high computational cost.

If you found this essay useful, stay tuned to my GitHub for updates. Importantly, we echo Donald Knuth’s words from his Claude’s Cycles essay (Knuth, 2026): Please work with like-minded researchers as much as you can, but without putting me into the loop!

What’s Next? Thanks to USP’s Information Technology and Technical Support Center (STI), we are currently developing practical activities for deep learning. We are also working to improve SAGE in terms of accuracy and computational efficiency, enabling it to operate at scale in large classes (> 100 students). An ambitious direction would be to organize an ICPC-style machine learning marathon; we believe this could foster innovation in developing theoretically sound machine learning techniques. Perhaps ML experts would even earn one or two balloons.

## 4. Acknowledgments

The author would like to express their deepest gratitude to the students for their constructive feedback. The author would also like to thank Artur Izquerdo for developing the first version of SAGE and Julia Fugita for her valuable contributions to the project. Finally, the author would like to thank the Central de Servic¸os de TI e Suporte Tecnico at ´ USP for providing a strong support environment that enables the development of new features, including deep learning exercises.

## References

Bengio, Y. and et al. International ai safety report. Technical report, 2025.

Dell’Acqua, F. and et al. Navigating the jagged technological frontier: Field experimental evidence of the effects of ai on knowledge worker productivity and quality. Technical report, 2023.

ICPC Foundation. The international collegiate programming contest.

Knuth, D. E. Claude’s cycles, 2026. Stanford University.

Maslej, N. and et al. Artificial intelligence index report 2024. Technical report, 2024.

Maslej, N. and et al. Artificial intelligence index report 2025. Technical report, 2025.

Team, Q. Qwen3 technical report, 2025. Alibaba Group.

Zheng, L. and et al. Judging llm-as-a-judge with mt-bench and chatbot arena. In Neural Information Processing Systems (NeurIPS), 2023.