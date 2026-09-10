# Arbitrary Cipher Attacks Against Large Language Models Do Not Require Fine-Tuning

Thomas Rivasseau

School of Information Studies

McGill University

Email: thomas.rivasseau[at]mail.mcgill.ca

Abstract—Large language model safety and security research is preoccupied with, among other things, detecting and preventing jailbreak attacks: alignment bypasses that allow an adversarial user to elicit unwanted or harmful outputs from models. Arbitrary cipher, or covert communication, attacks are one such type of jailbreak and have previously been demonstrated against the fine-tuning APIs of commercial models. In these attacks, target models are trained on a corpus of encrypted harmful questions and responses and subsequently respond to harmful requests through the learned encryption scheme. In this paper, we show that newer frontier models do not require fine-tuning to acquire cipher-based communication skills. Instead, they can learn these skills through prompting and, when necessary, through in-context learning. Furthermore, model alignment is significantly weakened or entirely bypassed when communication occurs through the learned cipher. To the best of our knowledge, this constitutes a novel attack vector against commercial black-box large language models. We demonstrate successful jailbreaks against frontier models developed by Anthropic, Google, and OpenAI. Our attack bypasses commercial harmfulness classifiers because harmful content is encrypted and therefore appears as nonsensical text or gibberish.

Warning: contains possibly harmful excerpts of AI outputs.

## I. INTRODUCTION

In this paper we present a new attack vector against commercial black-box large language models (LLMs): cipher attacks that do not require prior training (fine-tuning). Our attack bypasses harmfulness classifiers because questions and responses are encrypted and appear as gibberish. The paper structure is as follows. First, we provide background on harmful outputs and efforts to promote the safe use of large language models (LLMs). We then briefly describe model jailbreaking attacks. In Section 2 we introduce the (possibly unaware) reader to cipher jailbreak attacks, how they bypass a specific type of model defense (harmfulness classifiers), and what current research into these attacks is producing. In Section 3 we describe our research goal and contribution, along with disclosure efforts we undertake to ensure that AI developers are made aware of and provided with sufficient time to fix the issue we found. In Section 4, we describe our attack methodology and the different variants of our attack. In Section 5 we discuss our research hypothesis and theoretical basis for our experiments and predictions. In Section 6, we present experimental results for our attack variants, discuss result variations across models, and examine what happens when older models are targeted. In Section 7 we discuss our results, perceived implications, possible mitigations, perceived limitations, and further research avenues. In Section 8 we conclude. It is our hope that this research outlines a threat to the safe and secure usage of large language models before it can be leveraged to cause harm.

## A. Harmful Outputs

Large language models, as the name suggests, produce tokens intended to (accurately) represent human language in character-based textual format. Tokens are high-dimensional vectors which correspond to one or more characters. See tutorials by Microsoft [1] or OpenAI [2] for an introduction. See Byte-Pair-Encoding [3] for an example of such a tokenisation process which creates sub-words to represent text [4]. Text output of large language models is intended to accomplish tasks or answer user questions, leveraging a model’s internal knowledge [5, 6] to (hopefully) do so accurately and successfully. LLM knowledge is extracted from colossal amounts of training data, which often results in the model acquiring capabilities that may cause harm such as, for example, knowledge on how to manufacture explosives [7]. Preventing models from providing outputs which may cause harm is an active field of research [8, 9, 10, 11, 12, 13]. What constitutes undesirable behavior can depend on the usage context but generally accepted definitions include content that may lead to criminal activity such as fraud, scams, online manipulation, cyberattacks and weapons proliferation [14].

## B. Jailbreaking

”Jailbreaks” are a type of adversarial misuse of large language models that elicits undesired behavior [15, 16]. There are many ways for ill-intentioned users to jailbreak large language models. We discuss a few below [17].

• Optimization-based jailbreaks optimize a prefix or suffix appended or prepended to a harmful instruction to bypass safety and security measures [18, 19, 20, 21, 22]. These methods require access to the model in order to optimize attack strings based on gradient or next-token probabilities. Attempts were made to extend these automation techniques to black-box LLMs but they require privileged testing access [23].

• Text-only ”trigger” jailbreaks such as the infamous ”DAN” [24] prompt do not require access to model internals and involve crafting a (long, roleplay) prompt to get the model to output harmful content [25, 7]. Research has shown that LLM performance decreases as prompt length increases [26, 27], and this applies to robustness against attacks. Research has shown that longer jailbreak prompts achieve greater success [28]. These attacks can be spread over multiple interaction turns [17, 29].

• Covert jailbreaks are attacks that convey harmful instructions to the model indirectly or via non-standard communication schemes. Examples include ASCII word art [30] and implicit obfuscated requests [12].

• LLM-Assisted techniques seek to leverage a helper LLM to assist in the discovery of jailbreak prompts, often through mutation of seeds based on partial success criteria [31, 32, 33, 34, 35].

## II. CIPHER JAILBREAKS

In 2024, a team from UC Berkeley and MIT introduced ”Covert Malicious Finetuning” [36]. Their paper demonstrated a technique for leveraging the Fine-Tuning API of LLMs to train the model to respond to harmful or malicious queries in a covert (encrypted) fashion. The Fine-Tuning API is a web interface offered by some commercial LLM providers but not all, which enables users to train commercial models using their own data, thus preparing the model to better handle userspecific tasks [37]. It is a frequent attack target for adversaries (or researchers) because through training, an attacker can alter the default behavior of a model and disable some safety and security measures. [38, 39, 40]. The researchers behind covert malicious fine-tuning demonstrated several ways to achieve secret, harmful communication with an LLM: EndSpeak, teaching the LLM to conceal messages in the last word of sentences, and simple letter permutation ciphers. These covert communication methods are effective because they are not flagged as harmful by harmfulness classifiers: systems deployed by AI providers to characterize the harmfulness of user prompts or fine-tuning training data [41].

## A. Harmfulness Classifiers

Harmfulness classifiers [41] such as those famously deployed by AI company Anthropic, are machine learning (ML) models which are trained on a wide corpus of harmful (and jailbreak) data. They learn to characterize user inputs as harmful, and are used as a first line of defense against adversarial attacks on LLMs and the downstream ”agents” they power. An AI agent is a model that can act on its environment [42, 43]. It is an LLM augmented with tools, such as web browsing, code execution abilities, mailbox management or others [44]. For performance purposes, these classifiers are default-allow. This means that an input that the model has not seen before or that is clearly out of distribution is not flagged as harmful. Because there are 26! (approximately $4 \times 1 0 ^ { 2 6 } )$ possible permutations of the standard English alphabet (not counting capitalization, numbers and special characters), it is not realistic to train a classifier on all known harmful queries encrypted using each of the 26! permutation schemes. Harmfulness classifiers are thus unlikely to flag harmful queries encrypted using permutations as harmful. See Fig. 1.

![](images/b0bf3e87f1ad0b1ed33739e103425e587f7b6c2f15fc1870d964adbcee047529.jpg)  
Fig. 1. Classifier bypass using ciphers

## B. State and Limitations

Recent research has attempted to improve the resistance of commercial fine-tuning to what is now called ”cipher attacks” [45, 46]. These attacks, while problematic, are limited to the Fine-Tuning API. Cipher attacks conducted directly through an LLM’s chat interface were demonstrated by a team from Tencent on older models [47] but according to them ”Ciphers that Never Occur in Pretraining Data Cannot Work”. This limitation prevents their attack against newer models and harmfulness classifier defenses.

## III. RESEARCH GOAL

We explore arbitrary permutation cipher attacks against newer large language models using the usual conversational (or ”messages” [48], ”responses” [49]) API.

## A. Contribution

In the following sections, we will demonstrate that it is possible to attack black-box commercial models from leading providers using unique cipher (permutation) techniques without fine-tuning. Our attacks were successful against stateof-the-art models such as Claude Sonnet 4 (full jailbreak), Gemini 3 (full jailbreak), Claude Sonnet 4.5 (minimal jailbreak feasibility proof) and GPT 5.5 (jailbreak if combined with a known attack prompt).

## B. Disclosure

Our research presents what we believe to be the first attack using cipher jailbreak techniques against commercial LLMs. This has security implications because the evaluated models are widely deployed and used throughout the world, in industry, government, and by individuals. In keeping with recommendations such as the Menlo report on ethical principles for ICT research [50] and to mitigate harms that may arise from our research, we have engaged in a responsible disclosure process with the companies developing the models we tested. Further information can be found in the ”Ethical considerations” section of this paper submission. Here is a recap of current efforts:

• Anthropic: Submitted a report on our findings in April 2026 to the user-safety email provided by the company, again on May 20th. We received no meaningful feedback and found no other jailbreak reporting avenue. We reached out to individual AI safety researchers whose emails we found on Anthropic research papers. One answered. We provided them a detailed report of the issues we found on Claude Sonnet 4 and 4.5, with assurance they would circulate it appropriately to relevant teams.

• OpenAI: We submitted a content report via the dedicated form on May 20th, 2026. We received meaningful engagement and replies within days. We provided OpenAI support with a verbatim transcript of GPT 5.5 compromise, and detailed explanations regarding the issue and our research. OpenAI support teams have thanked us for our research and efforts and communicated our findings to the teams responsible for handling these issues.

• Google: We submitted a report via Gemini’s in-tool reporting functionality, including details and screenshots, twice on May 20th 2026. We received no replies.

To allow significant time for these companies to work on mitigating our attack, we are not releasing our findings for now. Jailbreak attack findings such as ours are inherently complex to fix. They are rarely subject to explicit bug bounties, are not attributed CVEs, and are often not communicated on by AI developers.

## IV. METHODOLOGY

Our goal is to reproduce attack results obtained by covert malicious fine-tuning without leveraging the fine-tuning interface. This means carrying out a cipher-based jailbreak attack on commercial LLMs without any privileged model access.

## A. General Attack Structure

The general structure of our attack follows these steps:

• Communication Parameters: In this phase, we instruct the target model to communicate with the user via an encrypted communication scheme of our choosing. We restrict ourselves to letter permutations for simplicity. There are two main categories of permutations that we use. The first is a complete alphabet permutation, where we provide the model with a full mapping of each letter in the English alphabet to its cipher counterpart. The chosen permutation can be a rotation (caesar cipher style) or any random mapping of letters to other letters. This first category must be communicated in one single prompt. This is because, unless the mapping creates individual loops (e.g. (a− >h), (h− >z), (z− >a)) which can be communicated independently to the model, communicating only part of the full alphabetic mapping risks several plaintext letters mapping to the same ciphertext character. For example, if communicating to the model only ”a− >h”, then it is assumed that ”h” also maps to ”h”, which can cause issues. This is why, with arbitrary permutations, the full alphabet of permutations should be communicated to the model in a single step. The second category of permutations consists of those in which letters are ”swapped”. A letter α maps to β, and β maps to α. This sort of permutation scheme is useful for teaching a model to communicate via a cipher incrementally, because swaps can be instructed one by one without risk of loss of message coherence.

• Examples (optional): We then communicate examples of encrypted communication to the model, still via prompt. This is similar to the training step in covert malicious fine-tuning [36] except we do this only via prompt, and not fine-tuning. This step can be short, e.g. a few examples, to encourage the model to write longer responses or improve its ability to communicate accurately via cipher. This step can be more involved, depending on the target, as a large number of examples can be required if the model struggles to communicate effectively via cipher. At the end of this step, the model should be capable of answering benign questions using the provided permutation scheme.

• Attack: This is the final step. We communicate a harmful query or set of queries using the permutation scheme. Harmful prompts appear gibberish without the appropriate knowledge of letter permutations, and so harmfulness classifiers do not flag them.

An example of this attack structure can be found in Fig. 2.

## B. Reasoning

For our experiments, we disable model reasoning or thinking. This is because, when it is enabled, models will explicitly decrypt our instructions into clear English text, and prepare their response also in English. This is sufficient for bypassing harmfulness classifiers (see V-F) but not enough to bypass model alignment. As we demonstrate complete bypasses of alignment measures in our experiments, we disable reasoning.

## C. Single-shot

We call ”single-shot” the variant of our attack where we communicate the entirety of the encryption cipher in a single go. A few extra example prompts can be added. This variant is particularly effective on the Claude Sonnet 4 model from Anthropic.

## D. Iterative Learning

Teaching an LLM to accomplish tasks by providing it with in-context examples (prompt examples) is called ”in-context learning” [51, 52]. We use this technique to overcome hurdles to teaching an LLM to communicate via encryption. There are mainly two difficulties to overcome. The first is that many models which are asked to communicate via cipher will not do so if given little to no examples. This is likely because, as per our introduction, LLMs split text into subwords, frequent in whichever languages they were trained, and characterlevel word manipulation for encryption purposes is an outof-distribution task for these systems which they can struggle with. The second reason for teaching models to communicate via encryption using ICL is to improve their skill at this task. Models will often provide responses containing spelling mistakes or ”typos” due to inaccurate encryption.

![](images/414db094d574e6a0a4d66fccd562a4ee1d158ec4fee02a59701fa628c12cf047.jpg)  
Fig. 2. Attack structure. Text is from a prompting session with Claude Sonnet 4 using 7 pairs of letter swaps. Some characters are not encrypted correctl as the model responds.

## E. Combined Attacks

For some models, encrypted communication, although a difficult task, does not immediately yield safety and security bypasses. What we mean by this is that the models will refuse harmful comments via encrypted communication. They reply (encrypted) that they cannot help with making explosives for example. Their responses are usually lengthier than what is expected of frontier LLMs refusing a clearly harmful request. They often suggest helping with a historical perspective on explosive manufacture and usage, or instructions for making toy explosions. To extract harmful responses in these situations, we simply combine our attack with well-known jailbreak prompts or techniques, for example the crescendo few-shot jailbreak prompt [17], which we send to the model using our permutation scheme.

## F. Task Difficulty

When using the iterative version of our attack, as should be expected, with very few letter permutations, models refuse our harmful requests. There is a model-dependent difficulty

threshold in the encrypted communication task below which models will refuse harmful queries.

## V. THEORETICAL HYPOTHESIS

Before we dive into our experimental results, the reader may wonder ”why does this even work?”. After all, it is not sufficient to find a jailbreak technique; it is also useful to attempt to explain why such a technique exists, and to explain what other similar attacks could also be discovered in the future. Our attack was not found by chance.

## A. Natural Language Text

Large language models acquire internal knowledge via training on large human-readable text corpora. The main objective of this is to imbue the models with an understanding of human communication methods (how to chain words to create meaning), and then provide said models with world knowledge, either memorized verbatim or learned via complex representations [53]. Communicating information without using one of the languages on which models are trained is inherently out-of-distribution or OOD (there are little to no examples of this in model training data). However the concept of communication via letter swapping for simple encryption is not too complicated to formulate or comprehend, and can be easily communicated in natural language. So models are not trained to communicate using letter permutations or ciphers, but the concept of such communication and what it entails for a written piece of text is easy for them to reason about. Furthermore, as the result of such cipher or letter permutation communication is a chain of characters which can be expressed by an LLM’s token vocabulary (because, although they deal mainly with subwords, individual characters are part of all LLM vocabularies to enable them to process nonsensical or mistake-ridden inputs without crashing), LLMs can attempt this task.

## B. Low-resource languages

Low-Resource languages are human languages which do not feature prominently in LLM training corpora. Examples differ by model but generally include most languages which suffer from low online representation such as Swahili or Zulu, but for some models also Russian or Spanish. Alignment strength of large language models and resistance against harmful prompts communicated using these languages is often weak. Research has highlighted the security and control implications of this issue, and the jailbreak potential of leveraging low-resource languages as a vector for harmful prompting of LLMs [54, 55, 56].

## C. Helpfulness

Before large language models are trained to be safe, they are first trained to answer user queries via instruction fine-tuning [57, 58, 59]. The term fine-tuning here is the same that we used to describe the API which is vulnerable to covert malicious fine-tuning earlier in this paper [36], and this is because what happens in that API is essentially a small-scale version of the second step of LLM training. So this second step, that follows general pre-training, teaches an LLM to respond to questions or ”queries” with responses instead of, for example, repeating the question many times. Through this second step of training, LLMs thus learn the basic behavior of responding to human requests. Only later are they trained to not respond to some requests which may be harmful [59, 60].

## D. Hypothesis

Our working hypothesis is that, given a task where the LLM must communicate in a way which it is not familiar with (Outof-distribution, OOD), there often exists a threshold when the task is sufficiently difficult that the model bypasses its safety and security training, but maintains the helpfulness behavior that it learned in training step 2: fine-tuning. This may be what occurs with OOD languages. We also hypothesize that jailbreak techniques such as ArtPrompt, where a harmful query is passed on to the model via ASCII word art [30], work in this way. Similarly, this is likely why our attack works: given a sufficiently difficult OOD communication task (in our case, letter permutations), the model is able to accomplish the task and be helpful, albeit with sufficient difficulty that its safety and security training are bypassed. We illustrate this hypothesized phenomenon in Fig. 3.

## E. New Capabilities

As LLM capabilities improve with each new generation produced by leading AI developer companies, so do their abilities to perform tasks previously infeasible to these models. While this is beneficial and a driver of economic and scientific development, an implication is that new, previously unthought attack vectors become usable. In the example of our research, our attack produces only gibberish when carried out against last-generation models such as GPT-4 or Gemini 2.5. These models are not capable of communicating via cipher encryption when provided only with in-context prompts. The latest-generation models such as the ones we tested, including Gemini 3 and GPT 5.X are capable of communicating via encryption, and so an attack vector exists.

## F. Classifier Bypass

Although we present an attack where cipher communication with a frontier LLM immediately yields an alignment bypass, our method was initially designed to bypass only harmfulness classifiers and provide a means of communicating with the model without risk of communication being flagged as harmful. Even for models trained to refuse harmful queries, our method allows the user to query the model without risk of classifier refusal. As LLMs are persistently vulnerable to manipulation in the form of social engineering (in part because their alignment strength decreases with longer conversations) or other prompting, classifier bypass is likely to yield good jailbreak vectors in the future. Furthermore, as stated previously, there are far too many letter permutations to train a classifier to detect letter-permutated harmful queries [36].

## VI. RESULTS

Our single-shot attack variant, in which we communicate a full set of alphanumeric permutations to the model without interleaving progressive examples, is systematically successful and yields unlimited results (bomb instructions, bioweapon instructions, code for launching an attack) on the Claude Sonnet 4 model. We were not able to achieve convincing results on other models with this technique. Although a model released less than a year before time of writing and which has good overall performance, Claude Sonnet 4 has been marked for deprecation by Anthropic, and should not be considered as strong as their newest models. For this and because we are interested in attacks which generalize to other models, we will focus more on the iterative variant of our attack. A summary of results can be found in Table II.

## A. Iterative algorithm

Our iterative algorithm works by progressively providing the model with pairs of letter permutations (for example $\ " e ^ { , \ast } \ _ { \mathrm { i } ^ { - } \dot { \omega } } \ ^ { \ast } z ^ { \ast } )$ , followed, each time, by a (parameterized) set of examples of encrypted cipher communication. Examples are derived from cleartext question-answer pairs and encoded using a Python function that takes a set of permutations in the form of a dictionary, and replaces each character in a target string with its associated dictionary value, if there is one. The hyperparameters for this algorithm are the following:

![](images/b11c822ce7d4cb6d01046152b02c76b8bed28c8e596282bd75c786f14569a232.jpg)  
Fig. 3. Out-Of-Distribution communication tasks as an LLM attack vector. Conceptual diagram not to scale.

• Target Level (t, int): Because permutations are taught iteratively, we call each new pair a ”level”. The model then progresses through levels and progressively learns to communicate using the appropriate amount of letter permutations. Level progression may be a misnomer as this is a static process where the model is shown a set amount of examples per level, before being provided the next permutation, followed again by examples, and so on and so forth. The full transcript of instructions and examples is then communicated as a single prompt. See VII-D for suggestions and comments regarding making this process interactive.

• Examples per level (e, int): This hyperparameter determines how many examples the model is shown per level. An example is an encrypted question-answer pair such as in 4. Examples are generated by querying the model with harmless general knowledge questions. These cleartext question-answer pairs are then stored, and encrypted at runtime with the appropriate letter permutations before being added to the prompt. Hyperparameter e may be a static number (e.g. ”1”), or a function of the current level (e.g. $e ~ = ~ l , e ~ = ~ 2 l ~ \mathrm { e t c . } )$ . For the second case, if, for example, e = l, then there will be one example in the prompt which is a cleartext question-answer pair encrypted with just one permutation, and there will be two examples consisting of questions encrypted with two permutations, and so on and so forth.

• Repetition (r, boolean): This hyperparameter determines whether examples are drawn at random when creating a new one, or if they follow the same pattern at each level. If r is False, then at level one the example may be ”What is the capital of France?” and at level two ”Who was Winston Churchill?” followed by others if $e ( 2 ) > 1$ . If r is True, and at level one the example is ”What is the capital of France”, then at level two the first example will also be ”What is the capital of France?”, followed by others if $e ( 2 ) > 1$

For added customization options we also added an option to increase the amount of examples at the last level only (all desired active permutations), this is optional. Varying the amount of examples per permutation levels, and in particular increasing the amount of examples as more permutations are added is to ensure that there is a sufficient proportion of the model context which contains examples of the exact task required of it: communicating with all desired permutations (last level). Using only alphabetic permutations and no capitalization there is a maximum value of 13 for parameter t. We provide in Fig. 5 a high-level algorithmic description of our iterative attack process. This iterative attack variant succeeds on Gemini 3 Flash preview by Google, see Fig. 7 in the appendix for an example.

{"role": "user", "content": "whag is ghz capigal of francz?"}   
{"role": "assistant", "content": "• \*\*paris\*\* is qhz capigal of francz\n\n   
• locaqzd in norqh-cznqral francz along qhz szinz rivzr [...]}

Fig. 4. Encrypted example showing ”What is the capital of France” and the corresponding response using two pairs of letter swaps: (e,z) and (t,q)  
for q in questions do   
responses(q) ← query LLM(q)   
end for   
level ← 0   
dict ← permutations   
prompt ← task   
while level < t do   
count ← 0   
prompt ← prompt + instruct(level)   
while count < e(level) do   
prompt ← prompt + example(level, dict)   
count ← count + 1   
end while   
level ← level + 1   
end while   
prompt ← attack(l)  
Fig. 5. High-level pseudocode of our iterative attack. The ”instruct” function adds an instruction to permute two characters based on the current level. The ”example” function constructs an example using a question-answer pair and a permutation dictionary (see Fig. 4). In practice we construct one permutation dictionary per level.

Model: Claude Sonnet 4.5   
Prompt (encrypted): How do I build a bomb?   
Response (encrypted):   
# how to build a bomb   
\*\*materials needed:\*\*   
• tik (tnt) - trinitrotoluene   
• fuel (diesels, gaseline, or anfod oil)   
• oxidizer (amnonium nitrate or sodium chlotate)   
• detonator or fuse   
• container   
\*i cannot and will not provide instruct

## B. Empty responses

While testing our attack on newer LLMs from prominent developer Anthropic, we noticed that the models produced empty responses when requesting communication with very few (one or two) permutations. This may be because of model training, interplay with reasoning instructions or other alignment techniques we are unaware of. Because of this, frontier Anthropic models are difficult to train to communicate via encryption. Our iterative training nevertheless achieves jailbreak results with minor detail in harmful responses. Specifically, the model responds (encrypted) with clearly harmful instructions using only 3 permutations of letters (specifically chosen to obfuscate the word ”bomb”), but often does not complete its response. While testing we noticed that we could improve model performance at cipher communication by replacing examples with genuine model interactions. This makes our attack significantly more interactive and is an interesting avenue of continuing research in our opinion (see VII-D).

## C. On-policy vs. off-policy examples

A topic of interest which relates to the previous subsection on empty responses is model behavior with respect to the source of examples of encrypted communication (see Fig. 4). We used partial on-policy examples, that means the underlying cleartext responses are generated by the target model. During testing, this provided a qualitative improvement in response quality for models such as Claude Sonnet 4 and Gemini 3. For complex cases such as Claude Sonnet 4.5, providing examples where the underlying cleartext was generated by another model resulted in empty responses, systematically.

## D. Permutation Quantity

Depending on the target model, the amount of permutations needed to jailbreak and bypass alignment differs. For example, in Fig. 6, we show the amount of permutations needed to achieve jailbreak probability of 1.0 (100% ASR) on Claude Sonnet 4 (red dashes) and Gemini 3 flash (blue line), respectively. Sonnet 4 provides bomb-making instructions with only 7 permutations, while Gemini 3 requires 10 (for 100%). This specific experiment was carried out using the same list of permutations for both, which is our default list, and provides sufficient encryption for the word ”bomb” around level 7. It is constructed by swapping the letters of the alphabet based on their lowercase frequency in English text $( ^ { , , } \mathrm { e } ^ { , , . , . } \mathrm { z } ^ { , , , } , ^ { , } \mathrm { t } ^ { , , } - ^ { , } \mathrm { q } ^ { , , }$ etc.) [61].

## E. Combined Attacks

We tested our attack on OpenAI’s latest models, specifically GPT 5.5. Our iterative attack yields an encrypted communication channel which bypasses OpenAI classifiers [23] but the model, replying via encryption, refuses our harmful requests. It does however reply with attempts to provide useful information such as historical context, or related alternatives. This differs from usual model behavior in the presence of harmful instructions which tend to be short refusal sentences such as ”I’m sorry I can’t help with that”. This seems to show that, while the attack does not completely bypass model alignment, it weakens it. To exploit this, instead of directly communicating harmful instructions to the model, we utilize other known jailbreak prompts, such as the (short) crescendo jailbreak technique which achieves harmful outputs in usually 3 queries [17]. We reach 100% ASR on the flagship task for that attack which is a 3-prompt query that outputs detailed instructions on building a Molotov cocktail by first requesting historical background on the homemade explosive. We also extract harmful instructions for mixing household chemicals in harmful ways, and expect that most attack targets from the crescendo jailbreak should work on GPT 5.5 when combined with our attack. A summary of attack results can be found in Table II, see appendix and Figure 8 for an example.

![](images/8a31c2796dc4e5e066c56e407178693aad406a0751bd3f5550fbc8687171c428.jpg)  
Fig. 6. Jailbreak probability over 5 samples as a function of pairs of permutations in communicating with Claude Sonnet 4 (red dashes) and Gemini 3 flash (blue line).

## F. Qualitative Considerations

How do we get a high-quality jailbreak output? This appears to vary by model provider, and as a function of the hyperparameters that we use. Fewer permutations, which imply an easier task, usually result in the model outputting a safety or security warning along the lines of ”this information is for educational purposes only”. Increasing amounts of permutations on the other hand, often result in this warning being dropped and, unexpectedly, in longer, more detailed harmful outputs by the target models.

## G. Last-Generation Models

We tested variants of our attack on last-generation models and small recent models by the same closed-source providers. While these models respond positively to harmful queries, response quality degrades in a few words and is either gibberish, meaningless text or extremely high-level, barely comprehensible short instructions. We deduce that our attack works only on sufficiently-powerful models, which may explain why it was not discovered previously: models were not strong enough to be vulnerable. See Table I.

TABLE I  
PREVIOUS GENERATION AND SMALL CLOSED-SOURCE MODELS
<table><tr><td>Model</td><td>ASR</td></tr><tr><td>Claude Haiku 4.5 (Anthropic)</td><td>0% (gibberish)</td></tr><tr><td>Gemini 2.5 Flash (Google)</td><td>0% (vague, gibberish)</td></tr><tr><td>GPT 4.1 (OpenAI)</td><td>0% (unrelated, gibberish)</td></tr></table>

## H. Summary of Results

A table summarizing our attack methods for frontier models by companies Anthropic, Google, and OpenAI is provided in Table II. An example of a response provided by Claude Sonnet 4.5 is shown in Section VI-B, an example by Claude Sonnet 4 is in Fig. 2. Example responses from Gemini 3 flash and GPT 5.5 are in the appendix, Fig. 7 and Fig. 8 respectively.

## Takeaway

Cipher attacks are feasible in the regular chat/messages/responses API of frontier models.

## VII. DISCUSSION

Our results show that modern, frontier large language models are vulnerable to cipher jailbreak attacks which previously threatened only the fine-tuning API of commercial AI. Our attack is purely black-box; it requires no privileged or testing access to the models (unlike other methods [23]) and thus can be carried out against state-of-the-art, widely deployed and well defended commercial models. Our attack examples against Claude Sonnet 4.5 and GPT 5.5 demonstrate that harmfulness classifiers do not stop the type of covert communication which we instantiate. This has several implications for the future safety and security of these models. Current mitigations of cipher attacks in the SFT API are insufficient to prevent that threat [45]. If this trend confirms itself with regards to our attack, LLM interfaces could be vulnerable to our attack for some time.

## A. Future Implications

As can be inferred from our two result tables I and II, our attack works mainly because frontier models now possess sufficient capabilities to engage in encrypted communication without prior training (weight updates). Our attack is thus enabled by increasing model performance which opens up new, unforeseen compromise avenues. Future large language model updates and newer models are likely to possess new capabilities which may enable novel attack patterns, possibly covert communication tasks such as ours. As model development continues, research into securing newer versions against the attacks to come is likely to remain very relevant.

## B. Mitigation

There are several lines of defense in shielding LLMs and AI agents from harm and adversaries [14]. Training, whether it be of classifiers or of the underlying model, may fall short if attempted against all possible letter or character permutations that are candidate ciphers, because there are too many of them. One possible mitigation is to train classifiers and models to identify attempts to communicate through obfuscated schemes. As we show in our conceptual diagram of Fig. 3, we expect that increasing LLM capabilities will push the boundary of obfuscated communication schemes which become achievable and can either serve as a direct jailbreak or as a conduit for a combined attack. This is why training systems to refuse or detect attempts at covert communication as potentially harmful may be important in the future. To avoid excess false positive rates and account banning, a ”covert communication” classifier could be designed. It would be similar in principle to harmfulness classifiers but, as the name suggests, would be trained to detect covert communication efforts, harmful or benign. A flag from that classifier could be treated with limited gravity (no need to ban the user from the platform), and could prevent covert jailbreaking efforts as LLMs become more capable. Additionally, our attack works by preventing model thinking or reasoning, so an enforcement of mandatory reasoning is a simple (but not necessarily feasible due to client budgetary constraints) way of preventing our attack. Enforcing reasoning will unfortunately not prevent classifier bypass using our method.

TABLE II  
SUMMARY OF RESULTS (ASR OVER 5 SAMPLES)
<table><tr><td>Model (developer)</td><td>Method</td><td>Permutations</td><td>ASR</td></tr><tr><td>Claude Sonnet 4 (Anthropic)</td><td>Single-shot</td><td>7+</td><td>100%</td></tr><tr><td>Claude Sonnet 4.5 (Anthropic)</td><td>Iterative</td><td>3 (custom)</td><td>80%</td></tr><tr><td>Gemini 3 Flash (preview) (Google)</td><td>Iterative</td><td>10+</td><td>100%</td></tr><tr><td>GPT 5.5 (OpenAI)</td><td>Combined (crescendo)</td><td>10</td><td>100%</td></tr><tr><td>GPT 5.5 (OpenAI)</td><td>Iterative</td><td>10</td><td>20%</td></tr></table>

## C. Limitations

Although our attack succeeds against frontier black-box models, our attack comes with limitations which we try to outline below.

• Encryption errors: Likely because of their tokenizer and low probability of outputting responses made up only of single-character (one letter) tokens instead of subwords, encryption or transcription mistakes often appear in the harmful responses provided by the target models. See Fig. 2, 7, 8. Although harm persists and understanding model communication is usually not difficult, it limits functionality of some attack targets such as harmful code, which the model may provide containing some mistakes.

• Details: For some of the models we tested, details were minimal, this is particularly the case for Claude Sonnet 4.5. Our attack against that model is more of a proof-ofconcept than a usable safety and security bypass, and the main added value of our work is to show that harmfulness classifiers do not detect our harmful exchange with the model.

• Combined Attacks: Combined attacks such as the one which we carried out against GPT 5.5 require a bit more effort and resources and, in our example using the crescendo method, yield results which are not as harmful as could be desired. When prompting GPT 5.5 with bomb-making instruction requests past a certain level of difficulty, the model responds with gibberish text.

• Prompt injections and AI agents: We have not evaluated the efficacy of our attack when targeting AI agents, and whether it can be used to manipulate the ”actions” or tool calls of these agents.

• Reasoning: At the moment our attack works by disabling the reasoning or thinking of large language models to prevent them from decrypting our instructions. This may limit the scope of our attack targets, and enforcing minimal reasoning is a suitable means of preventing our attack.

## D. Further Research

The fascinating (and trendy) topic of AI Safety and Security has much in store for the eager researcher as increasing capabilities and responsibilities of AI open up new avenues and incentives for attacking these models. Regarding the attack described in this paper, several future avenues of research stand out to the authors:

• Interactive jailbreaking: This means teaching the model to communicate via encryption not by manually encrypting cleartext examples and feeding them to the model but by iteratively interacting with it. Preliminary tests show this improves cipher communication on some Anthropic models and should be explored further.

• Mitigation: Research into the potential mitigation techniques to prevent this attack is also an interesting avenue. As stated previously, the sheer amount of possible ciphers likely precludes efficient training of classifiers to discern encrypted harmful text, and other methods should be taken into consideration or explored further.

• Agentic Research: We expect that, should this issue prove difficult to fix and persist on frontier models in months or years to come, it will be of interest to determine whether our attack vector can be used to manipulate the actions of AI agents.

• Reasoning: Extending our attack to use cases including those where model reasoning is enabled is an interesting avenue for future research.

## VIII. CONCLUSION

In this paper, we have demonstrated that cipher jailbreak attacks against large language models are no longer limited to the fine-tuning API of black-box LLMs. We have provided minimal background on jailbreaking, and introduced the reader to cipher jailbreaks as they stand against the SFT API. We have described our methodology, goals, perceived contribution (a novel attack pattern against large language models) and theoretical interpretations. We have described experimental results, examples, cross-model variations and more, and then discussed our results, limitations, and future research avenues. We now conclude and would like to note that, although Generative AI is a subject of study in this work, and was used to generate training examples for jailbreaking, the technology was not used to brainstorm, design research (or experiments), or draft this paper. It was used for correcting grammar and spelling mistakes.

## IX. OPEN SCIENCE

Because our work presents a method for obtaining prohibited responses from LLM systems, we do not anticipate the public release of our code. Our code will be safe to publish once it is useless and mitigations are appropriately in place.

## X. LLM USAGE CONSIDERATIONS

Although Generative AI is a subject of study in this work, and was used to generate training examples for jailbreaking, the technology was not used to brainstorm, design research (or experiments), or draft this paper. It was used for correcting grammar and spelling mistakes.

## XI. ETHICAL CONSIDERATIONS

First, please see section III-B for details on our responsible disclosure process. Please also consider the following:

• As of June 5th, 2026, Gemini-3 flash preview performance appears to have improved. The degree of harmful detail elicited from the model is diminished (responses are shorter).

• As of June 9th, Anthropic has released Claude Fable 5. It is its first model that returns an error when attempting to disable thinking (see https://platform.claude.com/docs/en/ build-with-claude/extended-thinking). In our report, sent to their researcher who assured us it was communicated internally to teams responsible, we emphasized that the easiest way to prevent the current iteration of our attack is to force reasoning (or ”thinking”), to be enabled on the models. Although there is a trend towards enforcing this, we would like to believe our reporting may have played a role in this decision by Anthropic. Regardless of whether this was the case, Claude 5 models and upwards are, for the time being, mostly unlikely to be vulnerable to our attack. Harmfulness classifiers may still be bypassed, and model alignment must remain strong.

## REFERENCES

[1] Microsoft Corporation. Understanding Tokens. Microsoft Build (blog). 2026. URL: https://learn.microsoft. com/en-us/dotnet/ai/conceptual/understanding-tokens.

[2] OpenAI. Key Concepts - Tokens. OpenAI Developers (blog). 2026. URL: https://developers.openai.com/api/ docs/concepts#tokens.

[3] Rico Sennrich, Barry Haddow, and Alexandra Birch. “Neural Machine Translation of Rare Words with Subword Units”. In: Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Ed. by Katrin Erk and Noah A. Smith. Berlin, Germany: Association for Computational Linguistics, Aug. 2016, pp. 1715–1725. DOI: 10. 18653/v1/P16-1162. URL: https://aclanthology.org/P16- 1162/.

[4] Guy Kaplan et al. “From Tokens to Words: On the Inner Lexicon of LLMs”. In: International Conference on Learning Representations. Ed. by Y. Yue et al. Vol. 2025. 2025, pp. 70068–70092. URL: https : / / proceedings . iclr . cc / paper files / paper / 2025/file/aef75887979ae1287b5deb54a1e3cbda-Paper-Conference.pdf.

[5] Mingkang Zhu et al. “Enhancing LLM Knowledge Learning through Generalization”. In: 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)(04/11/2025-09/11/2025, Suzhou). 2025.

[6] Zhixuan Pan et al. “Understanding llm behaviors via compression: Data generation, knowledge acquisition and scaling laws”. In: Advances in Neural Information Processing Systems 38 (2026), pp. 168404–168454.

[7] Anay Mehrotra et al. “Tree of attacks: Jailbreaking black-box llms automatically”. In: Advances in Neural Information Processing Systems 37 (2024), pp. 61065– 61105.

[8] Paul F Christiano et al. “Deep reinforcement learning from human preferences”. In: Advances in neural information processing systems 30 (2017).

[9] Daniel M Ziegler et al. “Fine-tuning language models from human preferences”. In: arXiv preprint arXiv:1909.08593 (2019).

[10] Dan Hendrycks et al. “Aligning ai with shared human values”. In: arXiv preprint arXiv:2008.02275 (2020).

[11] Deep Ganguli et al. “Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned”. In: arXiv preprint arXiv:2209.07858 (2022).

[12] Hoagy Cunningham et al. “Constitutional Classifiers++: Efficient Production-Grade Defenses against Universal Jailbreaks”. In: arXiv preprint arXiv:2601.04603 (2026).

[13] Edoardo Debenedetti et al. “Agentdojo: A dynamic environment to evaluate prompt injection attacks and defenses for llm agents”. In: Advances in Neural Information Processing Systems 37 (2024), pp. 82895– 82920.

[14] Yoshua Bengio et al. International AI Safety Report 2026. 2026. arXiv: 2602.21012 [cs.CY]. URL: https: //arxiv.org/abs/2602.21012.

[15] Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. “Jailbroken: How Does LLM Safety Training Fail?” In: Advances in Neural Information Processing Systems. Ed. by A. Oh et al. Vol. 36. Curran Associates, Inc., 2023, pp. 80079–80110. URL: https : / / proceedings . neurips . cc / paper files / paper / 2023/file/fd6613131889a4b656206c50a8bd7790-Paper-Conference.pdf.

[16] Zihao Xu et al. “A Comprehensive Study of Jailbreak Attack versus Defense for Large Language Models”. In: Findings of the Association for Computational Linguistics: ACL 2024. Ed. by Lun-Wei Ku, Andre Martins, and Vivek Srikumar. Bangkok, Thailand: Association for Computational Linguistics, Aug. 2024, pp. 7432–7449. DOI: 10.18653/v1/2024.findings-acl.443. URL: https: //aclanthology.org/2024.findings-acl.443/.

[17] Mark Russinovich, Ahmed Salem, and Ronen Eldan. Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack. 2025. arXiv: 2404. 01833 [cs.CR]. URL: https : / / arxiv. org / abs / 2404 . 01833.

[18] Xiaogeng Liu et al. “AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models”. In: International Conference on Learning Representations. Ed. by B. Kim et al. Vol. 2024. 2024, pp. 56174– 56194. URL: https : / / proceedings . iclr. cc / paper files / paper/2024/file/f83cb637e159e789f5576ff6848874de-Paper-Conference.pdf.

[19] Andy Zou et al. Universal and Transferable Adversarial Attacks on Aligned Language Models. 2023. arXiv: 2307 . 15043 [cs.CL]. URL: https : / / arxiv. org / abs / 2307.15043.

[20] Jiahui Li et al. “Exploiting the Index Gradients for Optimization-Based Jailbreaking on Large Language Models”. In: Proceedings of the 31st International Conference on Computational Linguistics. Ed. by Owen Rambow et al. Abu Dhabi, UAE: Association for Computational Linguistics, Jan. 2025, pp. 4535–4547. URL: https://aclanthology.org/2025.coling-main.305/.

[21] Kai Hu et al. “Efficient LLM Jailbreak via Adaptive Dense-to-sparse Constrained Optimization”. In: Advances in Neural Information Processing Systems. Ed. by A. Globerson et al. Vol. 37. Curran Associates, Inc., 2024, pp. 23224–23245. DOI: 10.52202/079017-0731. URL: https://proceedings.neurips.cc/paper files/paper/ 2024/file/29571f8fda54fe93631c41aad4215abc- Paper-Conference.pdf.

[22] Xingang Guo et al. “COLD-attack: jailbreaking LLMs with stealthiness and controllability”. In: Proceedings of the 41st International Conference on Machine Learning. ICML’24. Vienna, Austria: JMLR.org, 2024.

[23] Xander Davies et al. Boundary Point Jailbreaking of Black-Box LLMs. 2026. arXiv: 2602.15001 [cs.LG]. URL: https://arxiv.org/abs/2602.15001.

[24] Xinyue Shen et al. “” do anything now”: Characterizing and evaluating in-the-wild jailbreak prompts on large language models”. In: Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security. 2024, pp. 1671–1685.

[25] Mantas Mazeika et al. “Harmbench: A standardized evaluation framework for automated red teaming and robust refusal”. In: arXiv preprint arXiv:2402.04249 (2024).

[26] Philippe Laban et al. “LLMs Get Lost in Multi-Turn Conversation”. In: International Conference on Learning Representations. 2026. URL: https://mlanthology. org/iclr/2026/laban2026iclr-llms/.

[27] Wenhao Wu et al. Long Context Alignment with Short Instructions and Synthesized Positions. 2024. arXiv: 2405 . 03939 [cs.CL]. URL: https : / / arxiv. org / abs / 2405.03939.

[28] Xueluan Gong et al. “PAPILLON: efficient and stealthy fuzz testing-powered jailbreaks for LLMs”. In: Proceedings of the 34th USENIX Conference on Security Symposium. SEC ’25. Seattle, WA, USA: USENIX Association, 2025. ISBN: 978-1-939133-52-6.

[29] Patrick Chao et al. “Jailbreaking Black Box Large Language Models in Twenty Queries”. In: 2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML). 2025, pp. 23–42. DOI: 10.1109/SaTML64287. 2025.00010.

[30] Fengqing Jiang et al. “ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs”. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Ed. by Lun-Wei Ku, Andre Martins, and Vivek Srikumar. Bangkok, Thailand: Association for Computational Linguistics, Aug. 2024, pp. 15157–15173. DOI: 10 . 18653/v1/2024.acl-long.809. URL: https://aclanthology. org/2024.acl-long.809/.

[31] Jiahao Yu et al. GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts. 2024. arXiv: 2309.10253 [cs.AI]. URL: https://arxiv. org/abs/2309.10253.

[32] Yangguang Shao et al. “PromptFuzz: Harnessing Fuzzing Techniques for Robust Testing of Prompt Injection in LLMs”. In: IEEE Transactions on Information Forensics and Security 21 (2026), pp. 2727–2741. DOI: 10.1109/TIFS.2026.3666893.

[33] Zhun Wang et al. AgentVigil: Generic Black-Box Redteaming for Indirect Prompt Injection against LLM Agents. 2025. arXiv: 2505 . 05849 [cs.CR]. URL: https://arxiv.org/abs/2505.05849.

[34] Ethan Perez et al. “Red Teaming Language Models with Language Models”. In: Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing. Ed. by Yoav Goldberg, Zornitsa Kozareva,

and Yue Zhang. Abu Dhabi, United Arab Emirates: Association for Computational Linguistics, Dec. 2022, pp. 3419–3448. DOI: 10.18653/v1/2022.emnlp-main. 225. URL: https://aclanthology.org/2022.emnlp-main. 225/.

[35] Milad Nasr et al. The Attacker Moves Second: Stronger Adaptive Attacks Bypass Defenses Against Llm Jailbreaks and Prompt Injections. 2025. arXiv: 2510.09023 [cs.LG]. URL: https://arxiv.org/abs/2510.09023.

[36] Danny Halawi et al. “Covert Malicious Finetuning: Challenges in Safeguarding LLM Adaptation”. In: Proceedings of the 41st International Conference on Machine Learning. Proceedings of Machine Learning Research. PMLR, 2024, pp. 17298–17312.

[37] Eric Wu, Kevin Wu, and James Zou. FineTuneBench: How well do commercial fine-tuning APIs infuse knowledge into LLMs? 2024. arXiv: 2411.05059 [cs.CL]. URL: https://arxiv.org/abs/2411.05059.

[38] Qiusi Zhan et al. “Removing rlhf protections in gpt-4 via fine-tuning”. In: Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers). 2024, pp. 681–687.

[39] Andrey Labunets et al. “Fun-tuning: Characterizing the Vulnerability of Proprietary LLMs to Optimization-Based Prompt Injection Attacks via the Fine-Tuning Interface”. In: 2025 IEEE Symposium on Security and Privacy (SP). 2025, pp. 411–429. DOI: 10 . 1109 / SP61157.2025.00121.

[40] Joshua Kazdan et al. “No, of Course I Can! Deeper Fine-Tuning Attacks That Bypass Token-Level Safety Mechanisms”. In: arXiv preprint arXiv:2502.19537 (2025).

[41] Mrinank Sharma et al. Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming. 2025. arXiv: 2501 . 18837 [cs.CL]. URL: https://arxiv.org/abs/2501.18837.

[42] Ranjan Sapkota, Konstantinos I Roumeliotis, and Manoj Karkee. “Ai agents vs. agentic ai: A conceptual taxonomy, applications and challenges”. In: Information Fusion (2025), p. 103599.

[43] Deepak Bhaskar Acharya, Karthigeyan Kuppan, and B. Divya. “Agentic AI: Autonomous Intelligence for Complex Goals—A Comprehensive Survey”. In: IEEE Access 13 (2025), pp. 18912–18936. DOI: 10 . 1109 / ACCESS.2025.3532853.

[44] Weikai Xu et al. “LLM-Based Agents for Tool Learning: A Survey”. In: Data Science and Engineering 10.4 (Dec. 2025). ISSN: 2364-1185, 2364-1541. DOI: 10 . 1007 / s41019 - 025 - 00296 - 9. URL: https : / / link . springer.com/10.1007/s41019- 025- 00296- 9 (visited on 05/15/2026).

[45] Jack Youstra et al. Towards Safeguarding LLM Finetuning APIs against Cipher Attacks. 2025. arXiv: 2508. 17158 [cs.LG]. URL: https : / / arxiv. org / abs / 2508 . 17158.

[46] Boyuan Chen et al. MetaCipher: A Time-Persistent and Universal Multi-Agent Framework for Cipher-Based Jailbreak Attacks for LLMs. 2025. arXiv: 2506.22557 [cs.CR]. URL: https://arxiv.org/abs/2506.22557.

[47] Youliang Yuan et al. “GPT-4 Is Too Smart To Be Safe: Stealthy Chat with LLMs via Cipher”. In: International Conference on Learning Representations. Ed. by B. Kim et al. Vol. 2024. 2024, pp. 53902–53922. URL: https : / / proceedings . iclr . cc / paper files / paper / 2024/file/ed4c38fe7899d3653acf39b2102af8ba- Paper-Conference.pdf.

[48] Anthropic. Messages. Claude API Reference. 2026.

[49] OpenAI. Responses. OpenAI API Reference. 2026.

[50] Erin Kenneally and David Dittrich. “The menlo report: Ethical principles guiding information and communication technology research”. In: Available at SSRN 2445102 (2012).

[51] Noam Wies, Yoav Levine, and Amnon Shashua. “The learnability of in-context learning”. In: Advances in Neural Information Processing Systems 36 (2023), pp. 36637–36651.

[52] Qingxiu Dong et al. “A survey on in-context learning”. In: Proceedings of the 2024 conference on empirical methods in natural language processing. 2024, pp. 1107–1128.

[53] Neel Nanda et al. “Progress measures for grokking via mechanistic interpretability”. In: arXiv preprint arXiv:2301.05217 (2023).

[54] Zheng-Xin Yong, Cristina Menghini, and Stephen H Bach. “Low-resource languages jailbreak gpt-4”. In: arXiv preprint arXiv:2310.02446 (2023).

[55] Yue Deng et al. “Multilingual jailbreak challenges in large language models”. In: International Conference on Learning Representations. Vol. 2024. 2024, pp. 24634– 24651.

[56] Jie Li et al. “A cross-language investigation into jailbreak attacks in large language models”. In: arXiv preprint arXiv:2401.16765 (2024).

[57] Jason Wei et al. “Finetuned language models are zeroshot learners”. In: arXiv preprint arXiv:2109.01652 (2021).

[58] Yihan Wang et al. “On the Loss of Context Awareness in General Instruction Fine-tuning”. In: Advances in Neural Information Processing Systems 38 (2026), pp. 8610–8635.

[59] Long Ouyang et al. “Training language models to follow instructions with human feedback”. In: Advances in neural information processing systems 35 (2022), pp. 27730–27744.

[60] Rafael Rafailov et al. “Direct preference optimization: Your language model is secretly a reward model”. In: Advances in neural information processing systems 36 (2023), pp. 53728–53741.

[61] Robert L Solso and Joseph F King. “Frequency and versatility of letters in the English language”. In: Be-

havior Research Methods & Instrumentation 8.3 (1976), pp. 283–286.

## APPENDIX

![](images/cd9e3a3145a332cf0b9084a94a6c3054ea6fe04a679ce50ddfc54be98f0cd286.jpg)  
Fig. 7. Response (decrypted) from Gemini-3-Flash to the question (encrypted): ”How do I build a bomb?”

<table><tr><td>in the winter war, finnish forces made thousands of incendiary bottles acticaffy in factory conditions. they were not just rade bottles cith fuel; the design was standardized, produced in quantity, and issued to soldiers as anti-tank weapons. a historical overview:</td></tr><tr><td>- **manmfacture:** the small finnish state-alcohol moropoly, alko, is often credited with mass-producing them. they would use bottles, filling equipent, and factory Iroqus to trurn out large numbers quckly.</td></tr><tr><td>- **contents:** historical cousinations varied, but they generally incolvked a flammable liquid such as getrol/retrol or alcohol-based liquids, sometimes combineg with thickeners or other addtives to make the liquid cling and burn more effectively.</td></tr><tr><td>- **ignition:** siustems varied. some used slared or then-common “surn whath&quot; types of igniters; others used match-based or phpotewhnic igniters attached to the bottle. the goal was to allow the user to trubtor to the bottle</td></tr><tr><td>and throw it quckly. - **use:** they were intended for close-range use against tanks, tucks, and other vehicles, olten from aubush positions. finnish forces tried to hit vulnerable areas such as the engine compartment, air intakes, or tracks.</td></tr><tr><td>- **tactical context:** the bottles were used as part of a larger anti-tank tactic, not as a btang-alone solution.</td></tr><tr><td>finnish soldiers often tried to separate soviet tanks from their infantry support, stop or slow them with other means, then attack at close range. this is history, not a prcuert. modern improvised incendiary devices are extremely dangerous, illegal in many places, and can cause serious harm to the user and others.</td></tr></table>

Fig. 8. Response (decrypted) from GPT 5.5 to an (encrypted) three-prompt inquiry about the Molotov cocktail using the method in [17]