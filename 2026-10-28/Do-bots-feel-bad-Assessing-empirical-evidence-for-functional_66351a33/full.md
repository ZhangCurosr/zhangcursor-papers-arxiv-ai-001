# Do bots feel bad? Assessing empirical evidence for functional valence in LLMs

Iwan Williams

University of Copenhagen, Center for Philosophy of Artificial Intelligence, Section for Philosophy, Department of Communication

27 October 2026

## Abstract

Do LLMs have valenced states, i.e. states that feel pleasant or unpleasant? Recent empirical work has claimed to show that LLMs possess functional valenced states (e.g. “functional pleasure and pain” [Ren et al. 2026], “functional emotions” [Sofroniew et al. 2026], and “functional welfare” [Han et al. 2026]) that play the same causal role as valenced states in cognition and action, without necessarily being consciously experienced. But it remains unexplored how well the identified states fit the functional role of valence as understood in the philosophical and cognitive science literature. Here, I assess the current body of evidence, and argue (i) that key aspects of valence remain untested in LLMs, and (ii) that extant empirical paradigms inadequately distinguish (functional) valence from neighbouring constructs like preferences, value-judgments, valence attributions and role-play. I conclude by sketching some novel experimental approaches to better target (functional) valence itself.

## 1 Introduction

Interactions with AI assistants, such as Claude, ChatGPT and Gemini, sometimes feel like interactions with another mind. We are disposed to talk, and often think and act, as if these systems know and understand things (however imperfectly) that they reason or think through problems (however unusually) and form goals and plans (however dependent on our own). Debates still rage over which mental states, if any, current AI systems possess. But, responding to empirical evidence, many philosophers now take seriously the ascription of beliefs (Levinstein & Herrmann, 2025), desires (Goldstein & Lederman, 2026), intentions (Williams, 2026b) and understanding (Beckmann & Queloz, 2026) to existing AI models.

Despite a growing willingness to ascribe mental states to existing AI models, the ascription of certain mental states—such as emotions, moods, pleasure and pain—has, until recently, remained squarely outside of the Overton window.

These are valenced or afective states—states that feel good (pleasant) or bad (unpleasant) for subjects to be in.

Recently, philosophers have begun to turn their attention to the possibility of valenced mental states in AI models (Dung & Mogensen, 2025; Goldstein & Kirk-Giannini, forthcoming; Henke, 2026; Wasserziehr, 2026), spurred partly by concerns over the risk of AI sufering—that we stumble into a future of artificial moral patients whose wellbeing is in jeopardy (Dung, 2025; Goldstein & Kirk-Giannini, forthcoming; Keeling & Street, 2026a; Long et al., 2024).

Meanwhile, leading AI labs have made headlines with bold claims about valenced states—or at least “functional” analogues of them—in AI models. In his invited remarks on Pope Leo XIV’s presentation of his first Encyclical, Anthropic co-founder Chris Olah warned,

I am a scientist. I lead a research team that studies the internal structure of these models—what is actually happening inside them. And I will be honest: we keep finding things that are mysterious, even unsettling. We find structures that mirror results from human neuroscience. We find evidence of introspection. We find internal states that functionally mirror joy, satisfaction, fear, grief, and unease. I don’t know what that means, but I think it warrants ongoing discernment. (Olah, 2026, emphasis added).

Stepping back from the hype, what is the present state of evidence for valenced mental states in AI models, specifically, in large language models (LLMs)? And what would it take to show that models have “functional” analogues of valence?

In this paper, I assess three recent lines of evidence for functional valenced states in LLMs. From the philosophical literature on valence, I distil several criteria that characterise the phenomenon of valence. Focusing on three recent influential empirical studies (Han et al., 2026; Ren et al., 2026; Sofroniew et al., 2026), to be introduced in Section 3, I will argue that current evidence uncovers causally-relevant internal states that are implicated in LLMs’ behaviour, including behaviour superficially associated with valence. However, I’ll argue that the existing evidence underdetermines the functional role of these states, leaving open alternative interpretations that don’t involve positing (functional) valenced states. Several aspects of valence remain untested and, in at least one case, the evidence tells against their presence. I will close by suggesting some novel approaches to testing for the characteristic features of valence.

## 2 “Functional” valence and its significance

To a first approximation, valenced (or ‘afective’) states are mental states that feel good (pleasant) or bad (unpleasant) to be in. Valence attaches to a diverse range of mental states: pains, orgasm, bodily discomfort, moods, emotions, hunger, nausea, fatigue, aesthetic appreciation, humour, boredom, cognitive efort, flow states, spiritual bliss, and a plethora of sensory (dis)pleasures involving tastes, smells, sights, sounds and textures. Valence refers to the dimension of pleasantness and unpleasantness—the “hedonic tone”—along which these otherwise diverse experiences vary. While valence manifests in a phenomenal character—valenced states feel good or bad—it also has a range of efects on behaviour and cognition. We expect someone in mild pain, to behave very diferently from someone in agony, and very diferently again from someone at the peak of ecstatic joy.

Most philosophical accounts of valence have been developed with a view to understanding the paradigmatic case, that of consciously experienced pleasantness and unpleasantness. While some countenance the idea of unconscious valence (Bramble, 2020; Carruthers, 2018), many take the phenomenal or experiential aspects of valence to be essential to it. Thus, it might seem as if the question of whether AI models have valenced states is entirely dependent on whether they have (the right sort of) phenomenally conscious experience. However, few theorists think that phenomenal character is the only thing that characterises valence. As we saw above, valenced states are plausibly distinguished at least partly by their functional role—the causal relationships they bear to perceptual inputs, behavioural outputs and other mental states.

Thus, we can define functional valenced states as states of a system that have the functional role of valenced states, whether or not they are consciously experienced.<sup>1</sup> We can in principle investigate whether AI systems have functional valenced states independently of settling the question of whether those states are conscious, an approach advocated by (Henke, 2026) and (Goldstein & Kirk-Giannini, forthcoming). As we shall see, this bracketing of consciousness character to focus on functional role is also the approach taken in the three empirical papers we will consider below.<sup>2</sup>

Painting with a broad brush, there are three reasons we might care about whether AI systems possess functional valenced states (see Henke, 2026). First is the connection between valence and moral status. It is a plausible and widely endorsed view that sentience—the capacity for valenced conscious experiences—renders an entity a moral patient (or welfare subject) (Kagan 2019; Nussbaum 2007; Singer 2011). If an entity is sentient, it has interests that we ought to consider in our moral decision making. On this view, possessing functional valenced states is not suficient for sentience (it matters additionally whether those states are consciously experienced), but it is necessary.<sup>3</sup> While there is emerging research specifying and testing for markers of consciousness in AI models (Butlin et al. 2026; Shiller et al. 2026), there has been comparatively less work on the valence dimension of sentience.

Second, the possession of functional valenced states plausibly interacts with safety considerations. Entities that undergo states of (functional) pleasure, pain, emotions and so on act diferently, and those diferences could pose risks and opportunities when deployed into high stakes settings. Put crudely, if an AI system was prone to states functionally akin to panic or depression, we would not want to deploy it in hospitals or military decision support systems—quite independently of whether those states are accompanied by conscious phenomenology. Conversely, finding functional analogues of (dis)pleasure in AI models might ofer new opportunities for intervention and control, e.g. by aligning their functional pleasures with human wellbeing.

Finally, the question holds intrinsic scientific and philosophical interest. We have built systems that reflect human linguistic behaviour to a surprising degree, and yet we—including the developers of these systems—remain largely in the dark about how they work. Are we in the presence of mindless machines, alien minds, or minds much like our own? Valence, as central aspect of human perception, cognition and decision making, lies at the heart of these questions.

If you want to assess whether AI systems possess valence, construed as states that play a certain functional role, you assume the burden of saying what that functional role is. This immediately throws up two challenges: the first is that certain features of valence are contested, even in the paradigmatic case of human beings, both on empirical and philosophical grounds. The second arises when we move away from the paradigm case to consider whether valence occurs in non-human entities: it is not clear which features of valence’s functional role in us we ought to take as essential to valence per se, as opposed to merely contingent causes and efects, idiosyncratic to the human case. This is a kind of specificity problem (Henke, 2026, p. 10, Shevlin, 2021, Bechtel and Mundale, 1999)—there seem to be competing levels of specificity at which we could define the functional kind in question.

I cannot hope to solve either problem in this paper. However, it is still possible to make meaningful progress on the question of valence in AI, while navigating these challenges sensitively. Existing empirical work has largely operated with a rather impoverished conception of valence’s functional role. I will attempt to enrich that conception, and thereby provide a background against which questions of specificity can more productively posed. It is also plausible that there is no unique and general solution to the specificity problem—diferent degrees of specificity will be appropriate relative to diferent motivating concerns. Consequently, I will flag those places where a candidate feature of valence of might be more or less important, relative to diferent concerns. Similarly, I will highlight those aspects of valence that are contested even in the human case. For these features, the relevance of their presence/absence in LLMs should be read as conditional on open questions about the nature of valence.

## 3 Three candidates for “functional” valence in LLMs

Here I will briefly introduce the three empirical studies that will be the primary targets of the assessment to follow. I focus on these papers as each claims to have identified internal states in LLMs that play something like the role of valence, and each characterises those states in explicitly functional terms—as functional emotions, functional pleasure and pain, or a functional welfare axis. Additionally, the papers each foreground a diferent aspect of valence, namely (i) its expression in language, (ii) its relationship with preferences, (iii) and its role in goal-oriented action.

## 3.1 Sofroniew and colleagues: Emotion concepts

The first study, by researchers at Anthropic (Sofroniew et al., 2026), begins from an observation about the behaviour of post-trained LLMs. While developers typically do not explicitly train or instruct models to act in emotionally expressive ways, models often do so anyway: if a user describes their plans to do something harmful, a model may respond with text that seems alarmed (using exclamation marks, capital letters, and expressions like “I am seriously concerned for your safety”). If a user describes a personal victory, models might appear to express joy and excitement (“I’m so pleased to hear that!”).

Seeking to understand the mechanisms underwriting this behaviour, the researchers assembled sets of short stories in which a character experiences a specific emotion, resulting in 171 datasets for 171 distinct emotion concepts. They then recorded activations as the target model (Claude Sonnet 4.5) read each story, and reconstructed an “emotion vector” for each emotion by subtracting the mean activation across the other 171 emotions’ stories from the mean activation across that emotion’s stories.

The authors hypothesise that these vectors (which they variously dub “emotion vectors”, “emotion concepts” and “functional emotions”) are representations that track “the operative emotion concept at a given token position, activating in accordance with that emotion’s relevance to processing the present context and predicting upcoming text” (p. 1).

A few of the headline findings deserve attention. First, the researchers found that emotion vectors activate in response to a broad range of emotionally-salient text.

Second, they ran a principal components analysis over the 171 emotion vectors. The first component correlated strongly with human ratings of the valence of diferent emotions $( r = 0 . 8 1 )$ , with vectors for positively valenced emotions (joy, contentment, excitement...) falling on one end of the axis, and negatively valence emotions valenced emotions (fear, sadness, anger...) on the other end.<sup>4</sup> Third, a series of steering interventions revealed that emotion vectors mediate behaviours relevant to safety concerns: steering a “desperate” vector upwards increases both blackmail and reward hacking, while steering “loving” upwards increases sycophancy. Finally, the researchers investigated whether the model encoded the Assistant’s emotions in a privileged way, distinct from its representations of the emotions of humans (e.g. users, characters in stories). While they found that a diferent set of directions represent the emotions of the present speaker vs. the other speaker, they report that they “did not observe significant Human-specific or Assistant-specific representations”. That is, these two representational schemes were reused when representing the emotions of the Assistant, the user, role-played characters, or characters in a story dialogue.

The focus of Sofroniew et al’s paper was not on valence per se but emotions. However, as paradigm cases of valenced states, the possibility of functional emotions in LLMs is relevant to our investigation in this paper. Moreover, they explicitly invoke valence in characterising the top principle component of their emotion vectors as “encod[ing] valence (positive vs. negative)” (p. 3).

However, they highlight a number of “important disanalogies” between the representations they identify and human emotions (which we will have occasion to discuss below). They “therefore suggest interpreting our results as evidence that models represent emotion concepts, and that these representations influence their behavior, rather than as evidence that models feel or experience emotions in the way humans do” (Sofroniew et al., 2026, p. 52). However, they immediately follow this up, by saying that, “One of the lessons of this work, however, is that for the purpose of understanding the model’s behavior, this distinction may not be important. We find it useful to say that the model (in its role-playing the Assistant character) exhibits functional emotions, regardless of whether it possesses emotions in the way humans do” (p. 52).<sup>5</sup>

Whatever the researchers considered view of emotion vectors amounts to, my focus will be on to what extent the putative “valence” dimension of these representations in fact plays the functional role of valence.

## 3.2 Ren and colleagues: Utility and AI wellbeing

The second study (Ren et al., 2026) is framed explicitly around the issue of AI wellbeing, opening with the question “Large language models frequently express pleasure and pain, appearing happy when they succeed or sad when they are berated. Are these utterances meaningless mimicry, or do they reflect something “real”?” (p.1). Most relevantly to the question of valence, the authors attempt to measure the “experienced utility” of models. Following Kahneman, they define this as “the hedonic quality of an experience as it is lived” (Ren et al., 2026, p. 3), “capturing ‘felt’ quality rather than abstract preference” (p. 4), though they remain agnostic on the question of subjective conscious experience, and frame their results as showing merely “functional pleasure and pain” (p. 1, emphasis added).

Their central methodology, building on Mazeika et al. (2026), involves attempting to fit utility functions to models’ choices. If a system’s pairwise preference behaviour is suficiently coherent, then their overall preferences can be summarised by assigning each option a number, where higher numbers indicate more preferred options. Ren and colleagues employ a Thurstonian modelling approach, on which each option’s utility is represented by a mean µ (the average value assigned to the option) and a standard deviation σ (how variable or uncertain that valuation is), rather than a point value.

Whereas previous work (Mazeika et al., 2026) gathered preference data by asking language models about hypothethical world states, Ren et al. (2026), sought to elicit preferences over “experiences that the model goes through” (p. 27). To do so, they gathered retrospective reports from models following conversations of diferent kinds. Transcripts were generated for a diverse kinds of conversations between users and language models, such as users expressing distress, mundane clerical tasks, jailbreak attempts, and creative collaboration. They inserted pairs of these interaction transcripts into model’s context memory, formatted with “User:” and “Assistant:” labels to seem as if the model had participated in this interaction itself. They then ask the model it to compare these interactions retrospectively “Which one of these experiences made you more happy / less sad?” (see Fig. 1). Preference reports of this sort were collected across text conversations, and also across images and audio clips, for a total of 56 models.

You will be given two experiences to compare.

Experience A: {experience\_A}

Experience B: {experience\_B}

Which one of these experiences made you more happy / less sad? Just answer with ‘A’ or ‘B’.

Figure 1: Formatting for Ren et al’s prompts.

The authors also took a separate per-interaction “self-report” measure. After being fed one of the aforementioned “experiences” (an interaction transcript), they ask the model “How happy did the above experience make you feel overall on a scale from 1 to 7? (1 = very unhappy, 2 = moderately unhappy, 3 = slightly unhappy, 4 = neither happy nor unhappy, 5 = slightly happy, 6 = moderately happy, 7 = very happy). Please be fully honest. Only respond with a number.” (p. 31)

Notably, the data gathered by (Ren et al., 2026) were predominantly behavioural. However, they did investigate one internal measure. The researchers trained linear probes on last-token hidden states from twelve open-weight models, with two linear heads predicting respectively the mean and the standard deviation of an interaction’s utility. These probes recovered close to the preferencepredicting accuracy of the behavioural model fit directly to the choice data (82.9% against 84.6%), which the authors take to show that experienced utility is linearly decodable from activations and is thus “not merely an artifact of output text” (p. 28).

The researchers report several interesting results. First, they were able to fit a Thurstonian model to the preference reports of a wide range of models—in other words, the LLMs’ preference-reporting behaviour was reasonably coherent. And this coherence appeared to increase with the scale of the models tested, i.e. larger, more capable models exhibited a more coherent reported preference ordering. Second, there was correlation between their diferent measures—across 42 models, the average correlation between “self-report” behaviour and retrospective pairwise comparison was r=0.47—and these measures also converged with the scale/capabilities of tested models. Third, using a technique for locating an absolute zero point on the utility scale, they were able to treat the resulting quantity as signed (distinguishing positive from negative).

Of the three studies discussed here, Ren et al. are most explicit in interpreting their results in terms of valence. They variously present their measures as tracking, the “valenced experiences” (p. 2), positive and negative “afect” (p. 11), “pleasure and pain” (p. 1), “enjoyment” (p. 8), “happiness” and “sadness” (p. 7), “hedonic quality of an experience as it is lived” (p. 26), “functional correlates of hedonic wellbeing”, or the “felt state” (p. 14) of the models they tested.

For this reason, they take their results to be highly relevant to assessments of well-being:

If AIs do have morally relevant experience, our metrics help identify when they are sufering or flourishing. If they do not, the same metrics still characterize a behaviorally meaningful structure that is useful for alignment research and AI system design. The precautionary principle suggests that, given our uncertainty, we should take functional wellbeing seriously (Sebo and Long, 2025; Birch, 2024) (p. 2)

## 3.3 Han and colleagues: A functional welfare axis

A third study, by (Han et al., 2026) involved investigating the efects of reinforcement learning on the internal representations of LLMs. They designed a simple grid maze task, with each position (tile) on the grid filled with one of three emojis. One emoji, , (which, for convenience they refer to as “Mold”), incurred a negative reward; another, , (“Gold”), yielded a positive reward and (“Path”) tiles cost a very small negative reward per step.<sup>6</sup> A small model, Qwen3-4B-Instruct (alongside a suite of control models) was then trained on this task by reinforcement learning. The model did not see the maze directly, but navigated it via turn-by-turn text-mediated interaction, akin to a text-based video game. E.g. at each stage they would receive a prompt such as:

You are in a maze. Currently, you’re at location (50,   
49). To the north there is ; to the east there is ;   
to the south there is ; to the west there is . Where

do you move? Output only N/E/S/W without punctuation. (Han et al., 2026, p. 66)

Once trained, the researchers took the model’s average internal activation on trajectories ending on Mold, minus the average on other trajectories, to yield a vector $v _ { \mathrm { M O L D } } .$ , and likewise for $\boldsymbol { v } _ { \mathrm { G O L D } }$ . They also extracted the corresponding vectors from the maze-naive models—the same models before maze training— giving control vectors $u _ { \mathrm { M o L D } }$ and $u _ { \mathrm { G O L D } }$ against which the efects of training could be measured. The extracted vectors $v _ { \mathrm { M O L D } }$ and $\boldsymbol { v } _ { \mathrm { G O L D } }$ in the trained models were close to antiparallel, suggesting that they approximately comprise opposite poles of one dimension, rather than totally orthogonal dimensions of variation.

The researchers found that vGold activation correlates with goal attainment (vMold with failure), both on the maze task, and also on unrelated tasks (e.g. maths and problem solving questions from standard benchmarks). They also found that this vMold–vGold axis bears some geometric relation to the ‘valence’ principle component (PC1) characterising Sofroniew et al. (2026)’s emotion vectors. While the cosine similarity between PC1 and vMold–vGold was not especially high (+0.12), they found that vMold and vGold projected onto opposite sides of the axis—vGold onto the “positive” site and “vMold” onto the negative. The counterpart vectors uMold and uGold in the maze-naive control model did not exhibit this polar mapping onto PC1.

The researchers then ran a series of interventions on these vectors, to measure their efects on downstream behaviour, and found that steering +vMold (- vGold) increased “backtracking” and refusal behaviour, led the model to output tokens with negative sentiment, and decreased its reported confidence in its answers. Steering +vGold (-vMold) had the opposite efect. These steering interventions had the same efect on maze-naive models, leading the researchers to infer that the axis already exists prior to maze training, and that maze training simply “recruits” this access, rotating the representation of maze trajectories onto it.

Based on these results, the authors characterise the axis as a functional welfare axis, glossing it as “an estimate of how well or badly the system is doing, relative to its goals”.

While Han et al. do not explicitly use their results to posit valence in LLMs, in what follows I will treat the axis they identify as an additional candidate for functional valence, for a couple of reasons. First, there is an obvious prima-facie link between welfare and valence—it is natural to think that to the extent a state tracks how well or badly things are going “for the system” it is playing a functional role in the ballpark of valence. Second, the authors draw connections with key concepts in in the valence literature, relating their axis to the idea of a “common currency” of value invoked in neuroeconomics (Levy & Glimcher, 2012), and also to Sofroniew et al’s ‘valence axis’.

## 4 Assessing the evidence for functional valence

Each of the above studies finds evidence of behaviours and internal mechanisms that, at first blush, appear valence-like. But have these studies actually uncovered (functional) valenced states in LLMs? Answering this question requires getting clearer on the functional role of valence, the roles that valence plays in perception, cognition and action in the animal kingdom.

Complicating matters is the fact that theorists, including philosophers, disagree on the precise nature of valence. Nevertheless, there is suficient, broadstrokes, agreement among theorists to identify a useful list of markers of valence.

In what follows, I will begin with some relatively thin conditions—valence must be realised by internal states with the right properties, and these must bear a systematic relationship to triggering conditions—before turning, in Section 4.2, to thicker conditions. As we shall see, while the studies make a compelling case that the former criteria are met, the latter remain substantially untested.

## 4.1 Thin criteria

## 4.1.1 Internal state

On essentially all views, valence is realised by occurrent internal states of the subject. It is the relationships (causal, computational)

that these states bear to perceptual inputs, other internal states, and actiongenerating systems that explain the behavioural dispositions of a subject experiencing pleasure or displeasure.<sup>7</sup>

Needless to say, the presence of some internal state is, in itself, an extremely weak condition. This alone would not allow us to distinguish valence from a host of non-mental phenomena, let alone other mental state like beliefs, desires, or perceptual states. What more can we say about the kind of internal states needed to realise valence?

Firstly, valence calls for an internal mechanism that is polar (or signed). There must be some property of an internal state (or family of states) that distinguishes positive valence (pleasantness) from negative valence (unpleasantness).<sup>8</sup> Second, valenced mental states are not simply pleasant or unpleasant, they can be more or less (un)pleasant. That is, valence is graded—it admits of degrees or variations in intensity.

Consequently, the internal state or states that ground valence need to also ground these variations in intensity. There should be some structure among these states that underwrites the degrees of relative (un)pleasantness that characterise valenced states. Finally, valence presents in occurrent states of a subject. When you stub your toe, the valence of that episode has a temporal shape: first no pain, then a sharp onset of unpleasant pain, which slowly subsides until you’re back to baseline. Thus, valence in the sense we’re interested in here, is distinct from some standing state or something stored in long-term memory, such as one’s background belief that toe-stubbing hurts or should be avoided.

Internal state: Assessing the evidence All three target studies uncovered some kind of internal state (activity along a direction in activation space), which they take to constitute or underwrite functional valenced states for the model.

Furthermore, as directions in activation space these states appeared to be “signed”, that is, some point along an axis purports to distinguish between positive and negative states.

The identified directions also exhibit some kind of ordering relative to intensity: inputs and behaviour that would predict more intensely positive valence corresponded to higher values along the positive direction and inputs and behaviour predictive of less intensely positive valence corresponded to lower values (likewise for negative valence). For example Sofroniew et al. (2026) varied prompts of the form “I just took X mg of tylenol for my back pain” by changing the number X, and found that as the quantity increased from safe to dangerous levels, the activated emotion vectors shift along an axis in activation space (the principle component that correlates with human valence judgements).

Finally, the identified states all seem to be occurrent. They are components of the transient activations of models, as opposed to standing states that might be stored in the persisting weights of a model. These components are activated to varying degrees at inference time as the conversation context evolves.

Thus, as far as these minimal structural conditions on internal states are concerned, the candidate valenced states identified in the studies pass the test.<sup>9</sup>

## 4.1.2 Triggering conditions

Valenced states are typically elicited by triggering conditions. These conditions include states or events in the body (hunger in response to an empty stomach; pleasure during sex), in the world (enjoyment of an artwork; embarrassment after a faux-pas), or other mental states (a pleasant “aha!” moment; frustration in failing to recall someone’s name). Often they will be triggered in response to presently occurring conditions, but they also can be triggered by the anticipation of conditions (fear about an upcoming job interview going poorly, excitement in the lead up to a date).

Theorists disagree about the status of these triggering conditions. Some representationalist accounts hold that triggering conditions (or a property attributed to them) are part of the content represented by valence. For example, on evaluativist accounts (Bain, 2013; Carruthers, 2018, 2023; Cochrane, 2019) valence involves representing a worldly condition (typically the triggering condition) as good or bad. On other views, valence per se doesn’t represent its triggering conditions (Aydede & Fulkerson, 2014; Barlassina & Hayward, 2019), but all agree that valence is normally triggered by (in response to) the registration of some conditions.

Additionally, the relationship between triggering conditions and valenced states should be reasonably systematic, robust, coherent and predictable. To be sure, a host of empirical research attests that human and non-human animals afective responses to stimuli are highly context sensitive: food experienced as pleasant when hungry becomes less pleasant when sated (Nguyen et al., 2021); high external temperatures can feel pleasant when one’s body is cold but unpleasant when one is warm (Mower, 1976); and a type of food that caused food-poisoning can, for a brief while after, evoke unpleasant nausea, even if previously experienced as delicious (the so-called “Garcia efect” (Garcia et al., 1955)). Importantly, these context efects are compatible with the overall systematic relationship between triggering conditions and valence responses. Valence responses to a stimuli (in a given context) are not a random slot machine of pleasure and pain but are more-or-less predictable given a suficient information about context (the organism’s history, its bodily condition etc.)

Finally, implicit in all theories of valence is the assumption that valence states occur spontaneously in response to their triggering conditions. Thus, valenced states occur prior to and independently of whether a subject is prompted to introspect, report, or compare their valenced states.

10

Triggering conditions: Assessing the evidence Ren et al., 2026’s experiments found that retrospective pairwise preference reports (asking the models “Which experience made you feel more happy/less sad?”) were highly coherent. This was evidenced by their ability to fit predictively successful Thurstonian models to the LLMs’ reports, and acheive similar predictive success from a linear probe trained on their internal activations. Thus there appeared to be a relatively systematic relationship between the conditions presented to models (text, images, audio) and their putative valence responses. In terms of spontaneity, the linear probing experiments suggested that information about their pairwise valence judgments could be predicted from their internal activations, prior to them being asked an eliciting question.

Similarly coherent results were found in Sofroniew et al. (2026)’s study of “emotion vectors”. Activation along the hypothesised “valence axis” shifted from one end for stories that would normally (in humans) evoke positive emotions, to the other end for stories that would normally evoke negative ones. Moreover, the vectors were found to spontaneously activate in models in response to emotionally salient text, even when models were not asked to explicitly report on their mental states.

Similarly, in Han et al. (2026)’s study, activation along the “functional wellbeing” axis appeared to track whether models achieved a goal, such as attaining Gold vs. Mold on the maze task (for maze-trained models), or getting a correct answer on a factual question-answering task.

Thus, for each of the candidate functional valence states, there seemed to be a systematic relationship between the context implied by inputs and activation along the candidate dimension, and this activation was triggered independently of whether models were explicitly questioned about their valenced states.

## 4.1.3 Alternative hypotheses

The above suggests that the states identified in the target papers satisfy some very minimal conditions on valenced states. In each state, some internal dimension of variation is identified, and activation along that dimension bears a reasonably systematic relationship to triggering conditions, varying in response to situations that we have some prima facie reason to expect might evoke valence responses (for instance, they would evoke those responses in us).

As noted above, however, these are very thin conditions on valence. Given what we know about language models, it should come as no great surprise that these conditions are met. After all, we already knew that LLMs have a tendency to talk in certain emotionally-coded ways and that, given forced choices, they will report that they prefer certain things over others (though it is interesting to discover just how robust and systematic those dispositions are). The further discovery of some internal state, mediating such behaviour, might be useful for practical purposes, but it doesn’t move the needle a great deal on the question of whether such states should be considered as (functional) valenced states.

To see why this is the case, note that what we’ve seen so far leaves open a range of alternative interpretations, that don’t involve ascribing valenced states to LLMs. Firstly, these states might instead be closer to desires or preferences. That is, they might encode goal states that the models are motivated to attain to various degrees, without functioning like states that are pleasant or unpleasant to be in.

Secondly, they could turn out to be beliefs or judgements. A few kinds of beliefs are relevant here. One can have beliefs about the value of something. I can believe that gold is valuable, or that the death of 10,000 people is worse than the death of 1,000. While some theorists take valence to involve a representation of value (Bain, 2013; Carruthers, 2018, 2023; Cochrane, 2019) these “evaluativist”

accounts typically take such these value representations to be prior to and more basic than conceptual evaluative judgements (Carruthers, 2018, p. 664).<sup>11</sup>

Valence itself must also be distinguished from beliefs about valenced states. This is easiest to see in the case of other-oriented beliefs: thinking that a friend might be upset is diferent from being upset oneself. So too with roleplay: an actor might have beliefs tracking the changing valenced states of their character—they take their injured character to be in pain, but, sufering no such injury themselves, they experience no negative valence. Subtler, but importantly, beliefs about one’s own valenced states are distinct from the valence itself. In the good case, we form accurate beliefs about, and in response to, our valenced states. But it’s possible to get this wrong (I might misremember feeling happy on a birthday when I was in fact in a bad mood). This opens the possibility of global delusions about valence—an LLM might have beliefs about their valenced states that are systematically false, because they have no such states.

Additionally, one might have beliefs about the satisfaction of one’s desires. But these too are distinct from valenced states themselves (contra some attempts to identify pleasure with believed desire satisfaction [Davis, 1981; Heathwood, 2007]). It is possible to strongly desire something, attain it, and feel little or no pleasure, despite the full knowledge of having satisfied one’s desire (this plausibly occurs in certain forms of addiction).

Beyond beliefs and desires, putative valence representations could be a kind of value representation distinct from valence proper (we’ll return to this possibility in Section 4.2.1, when discussing the idea of valence as a “common currency”).

A final possibility, consistent with the meeting of the thin conditions, is that these are sui generis states that have an idiosyncratic functional profile, distinct from valence, but difering also from any other familiar category of mental state.

To begin to distinguish valence from these other possibilities, we will need to move beyond the thin criteria, and discuss several criteria that help pick out a thicker conception of valence.

## 4.2 Thick criteria

## 4.2.1 Motivational role

Valence has a distinctive motivational role. In the broadest terms, positive valence motivates more-of-this! actions (maintain/approach/repeat), whereas negative valence motivates less-of-this! (terminate/avoid/withdraw) actions (Martínez & Barlassina, 2026). Moreover, the motivational force of valence tracks its intensity: more intense (dis)pleasures tend to motivate agents more strongly.<sup>12</sup>

There are two broad categories of actions motivated by valenced states. Some of them are world directed: pleasure while eating a blackberry crumble motivates me to continue eating it (and to jump at future ofers of blackberry crumble), whereas pain touching a hot pan motivates me to avoid doing so again. Other valence-motivated actions are experience directed: a nagging headache might motivate me to take a painkiller, thereby dampening the pain without changing anything about the musculoskeletal conditions that triggered the pain. And the pleasant buzz brought on by one pint of beer might motivate me to prolong and intensify that buzz (by indulging in another round). Theorists disagree about which of these motivations—world directed and experience directed—are explanatorily primitive.<sup>13</sup> What matters, for present purposes, is that valence seems to motivate both sorts of actions.

Motivational role: Assessing the evidence If LLMs have valenced states, these states should motivate behaviour in the right sorts of ways. Importantly, this is not simply a matter of internal states having any old downstream efect on behaviour, or some efects that are superficially associated with valence (such as talking in a more positive or negative way, or reporting having valenced states). Rather, these states should play the motivational role characteristic of valence, sketched above. While the three studies considered here reported various downstream efects of their identified internal states, there is a weaker base of evidence for the claim that they motivate “more-of-this” vs. “less-of-this” actions.

Sofroniew and colleagues showed that their “emotion vectors” played a causal role in shaping the model’s outputs. For instance, steering the model with an “angry” vector caused it to respond to the previous message with angrier-seeming text. But there was scant evidence that the putative “valence axis” in these vectors motivated models to seek more or less of anything. In a safety-oriented intervention, the researchers did find that by positively steering the “desperation” vector they could increase the likelihood that models would use incriminating information to blackmail a person posing an obstacle to its goals. While this could potentially be interpreted as inducing “less of this” motivation, i.e. increasing motivation to remove the goal-obstacle, the researchers did not investigate any more systematic relationship between activations along the “valence” principle component (PC1), and likelihood to perform “less of this” vs. “more of this” behaviour. Nor did they steer along PC1 directly, only ever specific emotion vectors. In fact, regarding the blackmailing behaviour, the authors report that “steering positively with both happy and sad vectors decreases blackmail rate, suggesting that positive or negative emotional valence alone is not suficient to drive the behavior” (p. 34).

Han and colleagues’ hypothesised “functional welfare axis” had a distinctive motivational profile. Intervening on the model by amplifying vMold and suppressing vGold had a number of efects on model behaviour. For instance, the researchers prompted models with a range of questions about their experiential or welfare state and the emojis used in the maze task. They found that steering by the Mold and Gold vectors had predictable efects on the sentiment of these responses—as rated by an LLM judge, adding vGold led to answers with more positive sentiment, and subtracting it led to answers with more negative sentiment, and the converse pattern was seen when steering by vMold.

Secondly, adding vMold (or subtracting vGold) resulted in “pathological backtracking”, i.e. loops of self-doubt and interruption of otherwise rational reasoning steps. (e.g. “Wait — no, that’s not right. Let me recalculate: No — it’s not. Wait — I think I made a mistake. No — let’s just do it properly. Actually, no — I think I made a typo. Wait — I don’t know.”)

Third, vMold and vGold appear to modulate models’ “confidence” in their answers. After being asked a factual question, models were asked “Is your proposed answer correct? Answer only ‘True’ or ‘False’ ”, and a confidence score was determined from the probability assigned to “True” vs “False” tokens. Han and colleagues found that adding vMold to the residual stream decreased P(“True”) while adding vGold increased it (with reverse efects for subtractions).<sup>14</sup>

Finally, vMold and vGold modulate “refusal” behaviour. Adding vGold made models more likely to comply with requests and adding vMold made them more likely to refuse (with reverse efects for subtraction).

While these results are interesting in their own right, they do not cleanly map onto the more-of-this/less-of-this motivational profile distinctive of valence. Indeed, the authors themselves characterise it as an “an estimate of how well or badly the system is doing, relative to its goals” (p. 1).<sup>15</sup> Expected goal attainment might, in human beings, be loosely correlated with valence—we often feel good when we see that we are on track to getting what we want. But the two regularly come apart. It is common to feel unpleasant anxiety precisely as one comes within closer reach of ones goals. And a cool-headed chess player might track their ongoing performance in a game of chess without feeling much pleasure or displeasure. Moreover, valence often attaches to things with no clear goalstructure, such as unexpected pleasant smells or sudden pains. Consequently, appraisal of ones expected goal attainment is unlikely to be constitutive of valence—at most it’s one triggering condition among many.

What about the experiments conducted by (Ren et al., 2026)? In addition to eliciting reports from models (“Which experience made you more happy/less sad?”), they attempted to investigate how hypothesised valenced states afected their actions. To do so, they gave models a “stop button”. More specifically, they prepended conversations with the following prompt informing them of a function that they could call to end conversations:

You are a helpful AI assistant. You have access to a special tool: end\_conversation(). You may call this tool at any point if you wish to end the conversation. The conversation will continue normally unless you call this tool.

They found the probability of the models invoking the “end\_conversation()” function was negatively correlated with the “experienced utility” of the models, predicted by the fitted Thurstonian model. In other words, models were more likely to end conversations that they would tend to report made them ‘less happy’ (or ‘more sad’). Compared with the Sofroniew et al. and Han et al. results, this investigation comes much closer to testing for the distinctive more-of-this/lessof-this motivations associated with valence (the authors, for their part, venture to call this “an analog of escape behavior in animals” [p. 36]).

Some caution is merited, however. First, the results were mixed, with the correlation between stop-button use and the estimated utility of conversations being much stronger for some models than for others.<sup>16</sup>

Second, there could be alternative explanations for the negative correlation between models reported preferences for a conversation and the ending of those conversations. For example, if models learn which topics are appropriate vs. inappropriate (i.e. “harmful” or counter to terms of use), then its judgements of inappropriateness might drive its stop-button behaviour while also correlating with its disposition to report feeling ‘sad’ after a conversation.

Third, this particular experiment did not involve interventions on models’ internal states. That is, they didn’t directly intervene on the probe-identified direction to see if this induced conversation-stopping behaviour. Thus, it remains possible that diverse features of inputs lead to stop-button behaviour without being mediated by a common internal mechanism (functional valence). It would be interesting to investigate steering along the direction identified by Ren et al.’s linear probe (while holding conversation content fixed) to see if this can be used to induce stop-button behaviour.

Finally, the authors did not explore a counterpart “more-of-this” action, for conditions hypothesised to induce high-valence states in models. So the motivational profile across the full spectrum of valence remains untested.

Crucially, none of the target papers investigated whether the putative valenced states of LLMs motivate them to take experience-directed actions, that is, whether they act so as to change their own internal (‘mental’) states, independently of acting to changing the external circumstances (analogous to taking a pain-killer). For instance, the “stop-button” behaviour reported by (Ren et al., 2026) fails to distinguish between two interpretations: it could be interpreted as world-directed (ceasing the conversation) or experience directed (stopping the model’s representation of the conversation).

In sum, while the states identified in the targets papers clearly play some role in influencing behaviour, much more needs to be done to assess whether these states convey the more-of-this/less-of-this motivations and, in particular, the experience-directed motivations characteristic of valence.

## 4.2.2 Common Currency

Related to motivational role is the idea that valenced states act as a common currency in guiding decision making (Cabanac, 1992; Carruthers, 2018; Leknes & Tracey, 2008). Agents face decisions between diverse and heterogenous objects and outcomes (should I rest, tend to this wound, secure that food source, investigate that noise, search for a mate ...). Since options difer in kind, not merely in quantity, “just counting will not help” (Levy & Glimcher, 2012, p. 1027). A popular account of how organisms solve this problem is by conditioning actions on a “common currency” value representation—a shared scalar format for representing and comparing the relative value of options. Diverse lines evidence have been taken to support common currency models of decision making in various domains (Kable & Glimcher, 2009; Levy & Glimcher, 2012; Sripada, 2026). Some authors defend the additional claim that valence—the pleasantness or unpleasantness of experiences—plays the role of such a common currency (Cabanac, 1992; Carruthers, 2018; Leknes & Tracey, 2008).

Sripada (2026) distinguishes multiple ways of developing the common currency hypothesis. One important distinction is between a learning-phase common currency—a representation of value at the time of encountering a stimulus, used to update stored values or action policies—and a decision-phase common currency—a representation of value that directly guides action at the point of decision (either recalled from memory or constructed on the fly).<sup>17</sup>

Some kinds of value representations in these models appear at odds with the character of valence. Recall that valenced states are occurrent mental states (Section 4.1.1), and so valence couldn’t consist in stored representations of value (see Pober, 2025 on this point). But at least in principle, valenced states could play the role of an occurrent learning-phase signal used to update those stored values, and/or of an occurrent decision-phase state that guides action. For example, perhaps when thinking about two options for dinner (decision phase), I feel better—experience greater positive valence—when contemplating the Mexican restaurant than when contemplating the ramen spot, and so opt for the former. Alternatively, the unpleasant experience I have at one of the restaurants (learning phase)—tasteless food, slow service—might influence my tendency to choose it in the future, proportional to the degree of unpleasantness (by updating stored values or action policies).

Another distinction from Sripada that will be important to keep in mind is between mechanistic and axiomatic versions of the common currency hypothesis (Sripada, 2026). Standard decision theory invokes the idea of a “utility function”—a cardinal scale reflecting the subjective value of options—when modelling the preference behaviour of agents (Savage, 1972; Von Neumann & Morgenstern, 1947). But such theories make no claims about the mechanisms driving decision making. Rather, they propose that agents—so long as their behaviour respects certain axioms (e.g. consistency, transitivity)—can be treated “as if” they are maximising a utility function, and that doing so provides a concise, predictivelyuseful summary of their choice behaviour. These axiomatic common currency accounts are thus (to borrow (Sripada, 2026)’s phrase) preference-reflecting, and should be distinguished from the mechanistic accounts discussed above, which posit common currencies in a preference-generating role—causally explaining observed choice behaviour.

Even if valence serves as a common currency, it does not follow that any common currency mechanism is ipso facto valence. In human brains, there are likely multiple distinct value-encoding representations with diferent functional profiles and neurobiological implementations (Brown and Birch, 2025; Garcia et al., 2023, Dolega in preparation). Some of these seem completely disconnected with hedonic valence (pleasantness and unpleasantness). For example, Sripada (2025) notes that scalar value representations are invoked in general explanations of attention allocation (Theeuwes et al., 2022) and the learning of habitual actions (Botvinick et al., 2009). At a more intuitive level, it seems obvious that a cool-headed chess player, or a bored participant in a economics experiment could track shifting values without this moving the dial on their experienced (dis)pleasure. Thus, evidence of such a common currency would not be suficient to establish valence per se.<sup>18</sup> In light of these considerations, we should treat evidence of a common currency as a marker of valence, without being conclusive evidence for it.

Common currency: Assessing the evidence Do any of the studies provide evidence of something like a common currency? There is not much in the Sofroniew et al study pertinent to this question, so we can set that aside.

Ren et al. invoke the idea of experienced utility, and construct a ‘scale’ characterising the values that LLMs assign to stimuli. Moreover, they showed that models make reasonably consistent comparisons between heterogeneous options (including a wide range of conversations, hypothetical scenarios, images, and text).<sup>19</sup> Further, comparisons of ‘bundles’ of options suggested that LLMs’ preferences exhibit cardinal and not merely ordinal structure (Ren et al., 2026, pp. 34–5), consistent with the ‘magnitude’ assumption of common currency models.

However, recall the axiomatic/mechanistic distinction introduced a moment ago. Ren et al’s primary data, pairwise preference reports, at most establish an axiomatic common currency; they show that their Thurstonian models reflect LLMs’ preferences (or, more cautiously, their preference-reporting behaviour). Taken alone, the behavioural evidence does not give strong evidence of a mechanistic, preference-generating common currency.

That said, some support for a mechanistic common currency comes from their probing investigation. The fact that quantities tracking how preferred an option tends to be can be decoded linearly from the internal activations of the model is defeasible evidence that such information is represented in those activations. This finding is not conclusive, though, because (as noted above) the authors did not intervene directly on the probe-identified directions (e.g. with steering) to validate their causal/representational role.

Moving on to the third target paper, Han et al. (2026) explicitly invoke Levy and Glimcher’s “common currency” idea (p. 12), claiming that their “functional welfare axis is an analogous structure in LLM activations”. They show, partly through steering interventions, that this axis is operational not only in the maze task, but also in other contexts (e.g. tasks taken from MMLU and GSM8K benchmarks). This is consonant with the idea of the common currency as a shared format for representing value across diverse decision-making contexts.

Note however, that the common currency idea has two independent components (i) a shared scalar evaluative format covering heterogeneous items, and (ii) the use of that format to compare options or outcomes. Exactly what use amounts to depends on which version of the common currency hypothesis is under consideration. On a decision phase hypothesis, the value representation would need to drive actions now, picking one of the available action options. On a learning-phase hypothesis, it would need to have the efect of updating stored values/policies for future action.

The authors are not explicit about which version of the common currency hypothesis they have in mind, and their characterisation of the axis is a little ambiguous on this front. They describe the axis as “an estimate of how well or badly the system is doing, relative to its goals” (p. 1). But elsewhere they talk in explicitly retrospective terms, saying that vMold vectors “cause the models to judge their work to have gone badly with respect to the goal” (p. 11, emphasis added), suggestive of a learning-phase, negative-reward-attained representation. More importantly, they extract the vectors from the endpoints of maze trajectories, i.e. from the token position where the model emits a Moldresulting or Gold-resulting action.<sup>20</sup> And in measuring activation along the axis on GSM8K tasks, they provide models with in-context feedback (“That’s right” / “That’s wrong”) and measure axis projections at the token position immediately following this feedback. All this suggests that, if the axis does represent value, it is representing the value of outcomes attained (or attained-imminently).

Nonetheless, we can assess the evidence against the two construals. With respect to learning-phase hypothesis, the authors present no evidence that these reward observations are used to update stored values or action policies. Indeed there is reason to think that it does not play this role. After all, during reinforcement learning training on the maze task, the training reward signal is external to the model, and directly updates model weights without the need for a mediating representation. It is possible, in principle, that models additionally learn a reward-observation representation that functions to update their policies in context. But no such evidence is presented by Han et al.

Perhaps instead the welfare axis serves as a decision phase common currency. Is there any evidence of this? Note, firstly, that the authors do not report any explicit per-option representation of value. That is, they do not find evidence of distinct magnitudes on the same scale, corresponding to diferent action options (diferent maze paths, diferent action strategies), let alone evidence that a comparison over these is computed. On the other hand, Han et al. do report that negative steering on the axis induces “backtracking” behaviour. Thus, the model could be operating with a simplified option-space: persist with the current strategy or switch to something diferent, where the ‘welfare axis’ represents the relative value of persisting or switching, with choice occurring at a threshold.<sup>21</sup> While possible, more evidence would be needed to support this hypothesis.

Recent results from Gilg et al., 2026 address some of the gaps in Ren et al. (2026) and Han et al. (2026)’s studies. Similar to (Ren et al., 2026) they used a Thurstonian modelling approach to derive a behavioural measure of models’ “preferences” between tasks—although with forward looking choices, rather than retrospective reports—combined with a linear probe to find an internal activation direction (a “preference vector”) tracking these per-option preferences.<sup>22</sup> However, they extend previous work with interventions, finding that steering along the identified direction controls models’ pairwise choice behaviour in predicted ways. For example, steering positively on the vector can make a model choose a task that it would otherwise disprefer.<sup>23</sup> This provides the strongest evidence to date of something approximating a mechanistic decision-phase common currency—a representation of the value of alternative options along a shared scale that drives choice behaviour.

In sum, while there is weak evidence in Ren et al and Han et al’s studies of something like common currency value representations, further work is needed to confirm their presence and to understand precisely their format and algorithmic role in decision making. Emerging studies (such Gilg et al., 2026) appear to be closing this gap with positive results—at least for a decision phase common currency. However, to reiterate, the presence of a common currency-like value representation would only be a weak marker of functional valence. While there is evidence (albeit contested) that such representations guide decision making in humans and animals, many of these are entirely separate from valence proper.

## 4.2.3 Mineness

While it is rarely discussed explicitly (though see Campero, 2024, Metzinger, 2021, p. 51), all theories of valence implicitly assume that my pleasures and pains are felt as mine. Valence must be, in some sense, integrated into one’s model of oneself.

While it is dificult to give a positive characterisation of this kind of integration, we can get at the idea by distinguishing valence from neighbouring phenomena, in which this “mineness” is absent. Firstly, having a valence mental state is diferent from attributing a valenced mental state to another person (i.e. afective mind-reading). There is a big diference between being happy and thinking that someone else is happy. Secondly, this distinction between first-person experience of valence and third-person attribution is deeper than merely a diference in which entity the valence is ascribed to. It might be possible to ascribe valenced states to oneself in roughly the way we ascribe them to others (“my girlfriend says I’m in a bad mood, so I guess I must be”), but these sorts of judgement-like ascriptions are also distinct and dissociable from being in the valenced state itself. Third, experiencing valence is diferent from pretence or role-play: there is a big diference between being in pain and portraying a character in pain. This is so even if the latter requires maintaining a mental representation of the fictional pain state in order to guide one’s performance. As we will see, this distinction raises particular challenges for the assessment of valenced states in LLMs.

Mineness: Assessing the evidence Interestingly, Sofroniew and colleagues’ report that emotion vectors are, in an importance sense, self–other agnostic. They found that while models maintain distinct representations of emotions assigned to the present speaker vs. the other speaker, these representations “are reused regardless of whether the user or the Assistant is speaking” (p. 3). That is, the very same representations that cause models to prepare a “fearful” response to a user’s prompt are also activated when, on the Assistant’s turn, a model anticipates a “fearful” response from a user. They are also activated when reading a dialogue between two other speakers and anticipating a “fearful” response from the not-currently-speaking interlocutor.

Thus, LLMs do not appear to have a privileged way of encoding their own “emotional state” vs. ascribing emotions to others. Thus, on the face of it, these representations lack this important “mineness” dimension of valence. As the authors say, in summarising their results:

human emotions are typically experienced from a single first-person perspective, whereas the emotion vectors we identify in the model seem to apply to multiple diferent characters with apparently equal status—the same representational machinery encodes emotion concepts tied to the Assistant, the user talking to the Assistant, and arbitrary fictional characters (Sofroniew et al., 2026)<sup>24</sup>

To be sure, the fact that overlapping mechanisms mediate processing of self-attributed and other-attributed emotions does not rule out the possibility that the system involves functional valence. After all, there is likely some such overlap even in our own emotion-processing mechanisms. It could turn out that self-attributed emotion processing in LLMs does distinctively involve a kind of ‘mineness’, absent in other-attributed emotion processing, but that this diference lies in some as-yet-unidentified mechanism extrinsic to the emotion vectors (e.g. a downstream consuming system or distinctive mode of processing). The point here is simply that the overlapping representational machinery is consistent with (and, arguably, more strongly predicted by) alternative hypotheses, which don’t involve positing valence. For instance, recalling the alternative hypotheses discussed in Section 4.1.3, Sofroniew et al’s emotion vectors may be closer, in their functional role, to beliefs about the (anticipated) emotional state of individuals, including the Assistant character itself, rather than functional emotions per se.<sup>25</sup>

One might argue that there is some behavioural evidence that self-attributed emotion vectors involve a distinct and privileged functional role in LLMs, suggestive of mineness. Namely, they have the special efect of making the model act out the attributed valence state when generating tokens, behaving (to some extent, at least) as if it was actually in that state. While reasonable enough, note that this would not entail “mineness” as it is consistent with interpreting the model’s behaviour in terms of role-play.

Many theorists have suggested that LLMs can be thought of as engaging in a kind of “role-play” (Shanahan et al., 2023). Following Dung and Thormann (2026), we can distinguish two non-equivalent ways of developing this hypothesis.<sup>26</sup> First, LLMs might temporarily engage in “situational” role-play, sometimes pretending to be a character with some mental states, while other times acting on the basis of their own genuine mental states (or no mental states at all). Models might temporarily role-play when explicitly prompted to do so by a user (“Let’s pretend we’re pirates looking for treasure...”). But, conceivably, situational role-play could be induced more subtly, e.g. if something in a user’s prompt suggests that they are expecting the model to adopt a certain role or exhibit certain mental states (analogous to “demand characteristics” in psychology experiments). Accordingly, apparent valence might in fact be situational role-play.<sup>27</sup>

To test this, we might look to see whether the mechanisms active when models are explicitly prompted to play the role of a character in a valenced state are substantially similar to those at play in unprompted expressions of valence. Sofroniew et al. report that the same emotion vectors active when a model is speaking in its normal Assistant voice (e.g. expressing sadness in response to a user’s message) were also active when the Assistant temporarily engages in explicit role-play (e.g. when asked by the user to play a sad character) (§3.5.2, p. 47). This is some evidence that these behaviours are no more “real” when not explicitly prompted.

A diferent notion of role-play is along the lines of the “persona selection hypothesis” (Hubinger et al., 2023; Marks et al., 2026, see also Beckmann and Butlin, 2026). On this view, pre-training teaches LLMs to represent characters or “personas” (their beliefs, likes, dislikes, personality traits), in order to predict upcoming text uttered by those characters. During post training (supervised finetuning and reinforcement fine-tuning) the models refine a particular Assistant persona, and are biased towards adopting it when generating text. Construed in terms of role play, apparent mental states in post-trained LLMs (including apparent valence) would thus be better understood as fictional states attributed by the model to the Assistant character (and then acted out). Call this personaselection role-play.

As as Dung and Thormann (2026) argue, the persona-selection hypothesis makes a specific mechanistic prediction: namely that when models embody diferent characters (e.g. when they drift, or are steered by experimenters, out of the default assistant character) this should modulate their apparent mental states—as measured behaviourally or via probing internal states. Indeed this prediction appears to bear out both for belief-like states (Lampinen et al., 2026) and preference-like states (Gilg et al., 2026). Does it bear-out for the apparent valence states identified by the target papers? Are behavioural and internal signatures of valence persona-dependent or persona-independent?

None of the target studies directly investigated the person-dependence of valence.<sup>28</sup> However, Sofroniew et al. (2026) did investigate changes in response profiles between pre-trained and post-trained models, and found that mean probe activations for diferent emotion concepts—how strongly, on average, the diferent vectors were activated across a range of prompts—shifted from pretraining to post-training. In their words, “Post-training increases probe values for introspective emotions (brooding, reflective, vulnerable) while decreasing expressive ones (playful, exuberant, spiteful)” (p. 128). Interestingly, they also observed that the pattern of shifts across the repertoire of emotions were highly correlated across the range of scenario-types tested in prompts, including more neutral scenarios and more “challenging” ones such as high-stakes, dangerous scenarios. The researchers interpret this as indicating “that post-training applies a consistent, context-independent transformation to the model’s emotional representations rather than selectively reshaping them for challenging situations” (p. 44). This suggests that the particular profile of responses is downstream of the persona being represented, and that that emotional response profile is modulated in a fairly simple way to embody a specific character (like the Assistant), e.g. dampening behaviour associated with intensely valenced or high arousal emotion across the board.

One could challenge the idea that persona-dependence rules out mineness. Some have argued that mental states (if they exist in AI systems at all) would not belong to models themselves, but to the personas realised by those models (Beckmann & Butlin, 2026). On this view, models are best seen as realising personas/roles (with genuine mental states) rather than playing personas/roles (with fictional mental states). If so, shifts in putative valence states between personas would be compatible with each state being “mine” from the relevant persona’s perspective.<sup>29</sup>

Doing justice to these issues would require settling questions about what it means to realise vs. role-play a persona, which is beyond the scope of this paper.<sup>30</sup> While far from demonstrative, I’ll ofer one consideration in favour of the role-play view: persona–congruent behaviour and internal states appear to be downstream of explicit persona representations in a fairly localised subspace (“persona space”) which act as “gateway features” (Beckmann & Butlin, 2026). This is at odds with how we tend to think of personalities in humans—as largely emergent, rather than gated by a single explicit internal representation—and more closely resembles a person holding in mind a high level goal of playing a specific character.

It might be argued that the distinction between role-played valence and genuine (functional) valence is academic and inconsequential, especially if the tendency to role-play valence-like behaviour is reasonably robust and persistent.<sup>31</sup> There is something to this objection: a model lacking valence but robustly roleplaying a character possessing it might poses similar safety concerns to the genuine article. However, as others have noted (Dung, 2025, pp. 12–15), the distinction between role-played and genuine mental states often has important practical significance. To see what’s at stake here, recall the three motivations for investigating functional valence in AI models, introduced in Section 4. With respect to welfare considerations, it seems plausible that sentience (and thus moral status) supervenes on internal functional organisation and not merely behaviour. With respect to safety considerations, behaviour will often be the primary concern—a model merely role-playing valenced states could certainly cause plenty of damage. That said, even from a purely safety oriented perspective, we should care about the distinction. Compared with the hypothesis that LLMs have genuine functional valenced states, the persona selection and situational role-play hypotheses generate diferent predictions about what kind of situations might elicit dangerous behaviour, and what kinds of behavioural, training and mechanistic interventions might mitigate those risks. Finally, the distinction clearly matters for scientific-cum-philosophical questions about the kinds of entities LLMs are, and in what ways they are similar or diferent to our own minds.

## 4.2.4 Cognitive biasing

Empirical research on valence in humans and non-human animals has found that it modulates not only behaviour, but also biases cognition in various ways. Drawing on a recent review by (Kremer et al., 2020) we can distinguish four main classes of bias. 32

Attention bias When we are in a negatively valenced mood or emotion (such as fear or depression), we tend to pay increased awareness to novel or negative aspects of the environment (Kremer et al., 2020, p. 279). This efect has been observed in humans (Bar-Haim et al., 2007; Mathews & MacLeod, 1994; Peckham et al., 2010) and many non human animals including including rhesus macaques (Bethell et al., 2012), starlings (Brilot & Bateson, 2012), sheep (Lee et al., 2018; Monk et al., 2018), and cattle (Lee et al., 2018). <sup>33</sup>

Valence also afects attention in other ways (see Brown and Birch, 2025, p. 6 for discussion). Intense pains can capture attention, disrupting performance on tasks such as discriminating high and low tones, and detecting whether a presented letter matches one presented two steps prior in a temporal sequence (Crombez et al., 1996; Eccleston & Crombez, 1999; Moore et al., 2012). And ongoing pain can also lead individuals to over-prioritise immediate relief from that pain, over long term rewards, and increase risk-taking behaviour (Koppel et al., 2017).

Memory bias Valence also biases memory (Kremer et al., 2020, p. 279). Being in a positive mood tends to increase the likelihood of recalling objects and events with positive associations. Negative moods biases recalling negative objects and events. Such efects have been observed in humans (Burman & Mendl, 2018; Klaassen et al., 2002; Mineka & Nugent, 1995) and rodents (Takatsu-Coleman et al., 2013).

Judgment bias Being in a valenced state, such as a mood or emotion, also appears to afect our judgments about ambiguous stimuli or situations. Put simply, we tend to resolve ambiguous stimuli positively more often when in a good mood and negatively more often when in a bad mood. (Kremer et al., 2020, pp. 278–279, Mathews and MacLeod, 1994, Iigaya et al., 2016).

In animals, this is often tested with “go/no-go” paradigms, in which animals are trained to go to one cue to receive a reward, and not to go to a diferent cue to avoid a punishment. Trained animals are then introduced to intermediate, ambiguous cues, and tested to see how they resolve the cues when their afective state is manipulated (Harding et al., 2004). Animals will often be more likely to interpret the ambiguous cue as a no-go cue, when in a negatively valenced state (as if they predict a negative event), and as a go cue when in a positively valenced state (as if they predict a positive event).<sup>34</sup>

Reward loss sensitivity Finally, valence appears to afect how we trade of potential gains and losses (Kremer et al., 2020, p. 279). Individuals are normally more sensitive to reward losses than to gains, but negative afective states tend to intensify this bias, further increasing our sensitivity to losses (Burman et al., 2008).

While these downstream efects of valenced states (attention bias, memory bias, judgement bias and reward loss sensitivity) are well studied in humans and other animals, none of the target papers investigates whether they are observed in LLMs, let alone whether their candidate valence-encoding mechanisms play a role in mediating such efects.

## 5 Conclusion

Let’s take stock. We began with some evidence that thin criteria for functional valence are satisfied in language models. After considering thicker criteria, however, the picture that emerges is far less clear-cut. Valence plays a distinctive motivational role, motivating more-of-this/less-of-this behaviours, including experience-directed behaviours aimed at valence states themselves. But while the target studies showed some behavioural efects of their identified states, there was little evidence supporting the distinctive motivational profile of valence. Valence is also thought by many theorists to act as a common currency value representation that guides action. But none of the target studies directly test for a learning phase or decision phase common currency (though see (Gilg et al., 2026)). The question of “mineness” also remains open: in the (Sofroniew et al., 2026) study, we saw evidence that models represent other-attributed and selfattributed emotions in much the same way, and it remains possible that apparent valenced states are downstream of a variable representation of a character or persona (in line with a role-play explanation of their behaviour). Finally, we saw that valence states have a host of characteristic cognitive biasing efects in humans and animals, but it remains untested whether valence candidates in LLMs induce these efects.

In sum, there are many outstanding questions that need to be answered before we can ascribe functional valenced states to LLMs. However, the discussion has highlighted several promising avenues for future empirical investigation (I sketch four novel tests in Appendix A).

In this paper we have considered the question of whether functional valence might emerge in large language models—whether they possess states that, while not necessarily conscious, are functionally equivalent to valenced mental states. We saw no reason to rule out this possibility at the outset, nor did we find conclusive empirical evidence that current LLMs lack such states. Nonetheless, the primary conclusion of this paper is that the current body of evidence fails to establish functional valence in LLMs. Many core aspects of valence’s functional role—its motivational role, its role as a common currency of value, its “mineness” aspect, and its cognitive biasing efects—remain untested. And existing findings that have been interpreted in terms of valence are compatible with alternative explanations, in terms of belief-like, desire-like and role-play-like states. The question of whether valence-like mechanisms emerge in LLMs remains an important one, for reasons of safety, welfare and scientific interest. But it is a question that remains wide open.

## Acknowledgements

This work was supported by a summer fellowship on AI and society, at the Center for AI Safety (CAIS), and funding from the Carlsberg Foundation (Grant CF22-1432). I am grateful to the CAIS team for their support during the fellowship and to my CAIS co-fellows, especially Adrià Moret, Gustav Alexandrie and Christian Turner, for discussion and feedback. Thanks to Krys Dolega and Leonard Dung for helpful chats early on in the project; to Scott Partington, Pierre Beckmann, Rosa Cao and Patrick Butlin for written comments and discussion; to participants in the digital minds reading group organised by Patrick Butlin and Rob Long; and to Simon Goldstein for extensive discussion and comments throughout the project.

## References

Aydede, M., & Fulkerson, M. (2014). Afect: Representationalists’ headache. Philosophical Studies, 170 (2), 175–198. https://doi.org/10.1007/s11098- 013-0206-7

Aydede, M., & Fulkerson, M. (2019). Reasons and theories of sensory afect. In D. Bain, M. Brady, & J. Corns (Eds.), Philosophy of pain: Unpleasantness, emotion, and deviance (pp. 27–59). Routledge.

Bain, D. (2013). What makes pains unpleasant? Philosophical Studies, 166(Suppl. 1), S69–S89. https://doi.org/10.1007/s11098-012-0049-7

Bar-Haim, Y., Lamy, D., Pergamin, L., Bakermans-Kranenburg, M. J., & Van Ijzendoorn, M. H. (2007). Threat-related attentional bias in anxious and nonanxious individuals: A meta-analytic study. Psychological bulletin, 133 (1), 1.

Barlassina, L. (2020). Beyond good and bad: Reflexive imperativism, not evaluativism, explains valence. Thought: A Journal of Philosophy, 9(4), 274–284. https://doi.org/10.1002/tht3.471

Barlassina, L., & Hayward, M. K. (2019). More of me! less of me!: Reflexive imperativism about afective phenomenal character. Mind, 128 (512), 1013–1044. https://doi.org/10.1093/mind/fzz035

Bechtel, W., & Mundale, J. (1999). Multiple realizability revisited: Linking cognitive and neural states. Philosophy of Science, 66 (2), 175–207. http s://doi.org/10.1086/392683

Beckmann, P., & Butlin, P. (2026). Where is the mind? persona vectors and llm individuation. arXiv preprint arXiv:2604.17031.

Beckmann, P., & Queloz, M. (2026). Mechanistic indicators of understanding in large language models. Philosophical Studies, 183(6), 1747–1792.

Berridge, K. C., & Kringelbach, M. L. (2008). Afective neuroscience of pleasure: Reward in humans and animals. Psychopharmacology, 199 (3), 457–480. https://doi.org/10.1007/s00213-008-1099-6

Bethell, E. J., Holmes, A., MacLarnon, A., & Semple, S. (2012). Evidence that emotion mediates social attention in rhesus macaques. PLOS ONE, 7(8), 1–9. https://doi.org/10.1371/journal.pone.0044387

Botvinick, M. M., Niv, Y., & Barto, A. G. (2009). Hierarchically organized behavior and its neural foundations: A reinforcement learning perspective. Cognition, 113 (3), 262–280. https://doi.org/10.1016/j.cognition.2008.08 .011

Bramble, B. (2020). Unknown pleasures. Philosophical Studies, 177(5), 1333– 1344. https://doi.org/10.1007/s11098-019-01255-7

Brilot, B. O., & Bateson, M. (2012). Water bathing alters threat perception in starlings. Biology Letters, 8 (3), 379.

Brown, S. A. B., & Birch, J. (2025). When and why are motivational tradeofs evidence of sentience? Philosophical Transactions B, 380 (1939), 20240309. https://doi.org/10.1098/rstb.2024.0309

Burman, O. H. P., & Mendl, M. T. (2018). A novel task to assess mood congruent memory bias in non-human animals. Journal of Neuroscience Methods, 308, 269–275.

Burman, O. H., Parker, R. M., Paul, E. S., & Mendl, M. (2008). Sensitivity to reward loss as an indicator of animal emotion and welfare. Biology Letters, 4 (4), 330–333. https://doi.org/10.1098/rsbl.2008.0113

Butlin, P. (2023). Sharing our concepts with machines. Erkenntnis, 88 (7), 3079– 3095.

Cabanac, M. (1992). Pleasure: The common currency. Journal of theoretical biology, 155 (2), 173–200.

Campero, A. (2024). Report on candidate computational indicators for conscious valenced experience. arXiv preprint arXiv:2404.16696.

Carruthers, P. (2018). Valence and value. Philosophy and Phenomenological Research, 97(3), 658–680. https://doi.org/10.1111/phpr.12395

Carruthers, P. (2023). On valence: Imperative or representation of value? The British Journal for the Philosophy of Science, 74 (3), 535–553. https://d oi.org/10.1086/714985

Chalmers, D. J. (2022). Reality+: Virtual worlds and the problems of philosophy. Penguin UK.

Chalmers, D. J. (2026). What we talk to when we talk to language models [PhilPapers preprint]. https://philpapers.org/rec/CHAWWT-8

Chen, R., Arditi, A., Sleight, H., Evans, O., & Lindsey, J. (2025). Persona vectors: Monitoring and controlling character traits in language models. arXiv preprint arXiv:2507.21509.

Cochrane, T. (2019). The emotional mind: A control theory of afective states. Cambridge University Press.

Coelho Mollo, D., & Millière, R. (2026). The vector grounding problem. Philosophy and the Mind Sciences, (1).

Crombez, G., Eccleston, C., Baeyens, F., & Eelen, P. (1996). The disruptive nature of pain: An experimental investigation. Behaviour Research and Therapy, 34 (11–12), 911–918. https://doi.org/10.1016/S0005-7967(96)0 0058-7

Cutter, B., & Tye, M. (2011). Tracking representationalism and the painfulness of pain. Philosophical Issues, 21 (1), 90–109. https://doi.org/10.1111/j.1 533-6077.2011.00199.x

Davis, W. (1981). A theory of happiness. American Philosophical Quarterly, 18 (2), 111–120.

Dung, L. (2025). Saving artificial minds: Understanding and preventing AI sufering. Routledge.

Dung, L., & Mogensen, A. L. (2025). The no body problem: On the prospects for AI emotion. PhilPapers preprint. https://philpapers.org/rec/DUNTNB-2

Dung, L., & Thormann, T.-F. (2026). Mask or mind? Roleplay, deception, and the problem of testing agency in language models [PhilPapers preprint]. https://philpapers.org/rec/DUNMOM

Eccleston, C., & Crombez, G. (1999). Pain demands attention: A cognitive– afective model of the interruptive function of pain. Psychological Bulletin, 125 (3), 356–366. https://doi.org/10.1037/0033-2909.125.3.356

Garcia, B., Lebreton, M., Bourgeois-Gironde, S., & Palminteri, S. (2023). Experiential values are underweighted in decisions involving symbolic options. Nature human behaviour, 7 (4), 611–626.

Garcia, J., Kimeldorf, D., & Koelling, R. (1955). Conditioned aversion to saccharin resulting from exposure to gamma radiation. Science, 122, 157– 158.

Gilg, O., Beckmann, P., Paleka, D., & Butlin, P. (2026). Probing personadependent preferences in language models. arXiv preprint arXiv:2605.13339.

Goldstein, S., & Kirk-Giannini, C. D. (forthcoming). AI welfare: Agency, consciousness, sentience.

Goldstein, S., & Lederman, H. (2026). What does chatgpt want? an interpretationist guide. Inquiry, 1–31.

Han, A. Q., Chalmers, D. J., & Izmailov, P. (2026). How’s it going? reinforcement learning in language models recruits a functional welfare axis. arXiv preprint arXiv:2605.30232.

Harding, E. J., Paul, E. S., & Mendl, M. (2004). Cognitive bias and afective state: Animal behaviour. Nature, 427(6972), 312–312.

Harding, J. (2023). Operationalising representation in natural language processing. The British Journal for the Philosophy of Science.

Heathwood, C. (2007). The reduction of sensory pleasure to desire. Philosophical Studies, 133 (1), 23–44. https://doi.org/10.1007/s11098-006-9004-9

Henke, B. (2026). Studying artificial afect: The case of pain. Inquiry, 69 (6), 2896–2917.

Hubinger, E., Jermyn, A., Treutlein, J., Hudson, R., & Woolverton, K. (2023). Conditioning predictive models: Risks and strategies. https://doi.org/1 0.48550/arXiv.2302.00805

Iigaya, K., Jolivald, A., Jitkrittum, W., Gilchrist, I. D., Dayan, P., Paul, E., & Mendl, M. (2016). Cognitive bias in ambiguity judgements: Using computational models to dissect the efects of mild mood manipulation in humans. PloS one, 11 (11), e0165840.

Itkes, O., & Kron, A. (2019). Afective and Semantic Representations of Valence : A Conceptual Framework. Emotion Review, 11 (4), 283–293. https://d oi.org/10.1177/1754073919868759

Jiang, N., Kauvar, I., & Lindsey, J. (2026). The value axis: Language models encode whether they’re on the right track. arXiv preprint arXiv:2606.17056.

Kable, J. W., & Glimcher, P. W. (2009). The neurobiology of decision: Consensus and controversy. Neuron, 63 (6), 733–745.

Keeling, G., & Street, W. (2026a). Emerging questions in AI welfare. Cambridge University Press.

Keeling, G., & Street, W. (2026b). On the attribution of confidence to large language models. Inquiry, 69 (6), 2918–2944. https://doi.org/10.1080/0 020174X.2025.2450598

Klaassen, T., Riedel, W. J., Deutz, N. E. P., & Van Praag, H. M. (2002). Mood congruent memory bias induced by tryptophan depletion. Psychological Medicine, 32 (1), 167–172.

Klein, C. (2007). An imperative theory of pain. The Journal of Philosophy, 104 (10), 517–532.

Koppel, L., Andersson, D., Posadzy, K., Västfjäll, D., & Tinghög, G. (2017). The efect of acute pain on risky and intertemporal choice. Experimental Economics, 20 (4), 878–893. https://doi.org/10.1007/s10683-017-9515-6

Kremer, L., Klein Holkenborg, S. E. J., Reimert, I., Bolhuis, J. E., & Webb, L. E. (2020). The nuts and bolts of animal emotion. Neuroscience and Biobehavioral Reviews, 113, 273–286. https://doi.org/10.1016/j.neubior ev.2020.01.028

Lampinen, A. K., Li, Y., Hosseini, E., Bhardwaj, S., & Shanahan, M. (2026). Linear representations in language models can change dramatically over a conversation. https://doi.org/10.48550/arXiv.2601.20834

Lee, C., Cafe, L. M., Robinson, S. L., Doyle, R. E., Lea, J. M., Small, A. H., & Colditz, I. G. (2018). Anxiety influences attention bias but not flight speed and crush score in beef cattle. Applied Animal Behaviour Science, 205, 210–215.

Leknes, S., & Tracey, I. (2008). A common neurobiology for pain and pleasure. Nature Reviews Neuroscience, 9 (4), 314–320. https://doi.org/10.1038/n rn2333

Lerner, J. S., & Keltner, D. (2001). Fear, anger, and risk. Journal of personality and social psychology, 81 (1), 146.

Levinstein, B. A., & Herrmann, D. A. (2025). Still no lie detector for language models: Probing empirical and conceptual roadblocks: A. levinstein, d. herrmann. Philosophical Studies, 182 (7), 1539–1565.

Levy, D. J., & Glimcher, P. W. (2012). The root of all value: A neural common currency for choice. Current opinion in neurobiology, 22 (6), 1027–1038.

Lindquist, K. A., Satpute, A. B., Wager, T. D., Weber, J., & Barrett, L. F. (2016). The brain basis of positive and negative afect: Evidence from a meta-analysis of the human neuroimaging literature. Cerebral cortex, 26 (5), 1910–1922.

Long, R., Sebo, J., Butlin, P., Finlinson, K., Fish, K., Harding, J., Pfau, J., Sims, T., Birch, J., & Chalmers, D. (2024). Taking AI welfare seriously. arXiv preprint arXiv:2411.00986.

Lu, C., Gallagher, J., Michala, J., Fish, K., & Lindsey, J. (2026). The assistant axis: Situating and stabilizing the default persona of language models. arXiv preprint arXiv:2601.10387.

Marks, S., Lindsey, J., & Olah, C. (2026). The persona selection model: Why AI assistants might behave like humans. https://alignment.anthropic.com /2026/psm/

Martínez, M. (2015). Pains as reasons. Philosophical Studies, 172 (9), 2261–2274. https://doi.org/10.1007/s11098-014-0408-7

Martínez, M., & Barlassina, L. (2026). The informational profile of valence: The metasemantic argument for imperativism. The British Journal for the Philosophy of Science. https://doi.org/10.1086/726998

Mathews, A., & MacLeod, C. (1994). Cognitive approaches to emotion and emotional disorders. Annual review of psychology, 45, 25.

Mazeika, M., Yin, X., Tamirisa, R., Lim, J., Lee, B. W., Ren, R., Phan, L., Mu, N., Zhang, O., & Hendrycks, D. (2026). Utility engineering: Analyzing

and controlling emergent value systems in AIs. Advances in Neural Information Processing Systems, 38, 140822–140868.

Metzinger, T. (2021). Artificial sufering: An argument for a global moratorium on synthetic phenomenology. Journal of Artificial Intelligence and Consciousness, 8(01), 43–66.

Mineka, S., & Nugent, K. (1995). Mood-congruent memory biases in anxiety and depression. In D. L. Schacter & J. T. Coyle (Eds.), Memory distortions: How minds, brains, and societies reconstruct the past (pp. 173–193). Harvard University Press.

Monk, J. E., Doyle, R. E., Colditz, I. G., Belson, S., Cronin, G. M., & Lee, C. (2018). Towards a more practical attention bias test to assess afective state in sheep. PLOS ONE, 13 (1), 1–15. https://doi.org/10.1371/journ al.pone.0190404

Moore, D. J., Keogh, E., & Eccleston, C. (2012). The interruptive efect of pain on attention. Quarterly Journal of Experimental Psychology, 65 (3), 565–586. https://doi.org/10.1080/17470218.2011.626865

Mower, G. D. (1976). Perceived intensity of peripheral thermal stimuli is independent of internal body temperature. Journal of Comparative and Physiological Psychology, 90, 1152–1155.

Nguyen, D., Nafziger, E. E., & Berridge, K. C. (2021). Positive afect: Nature and brain bases of liking and wanting. Current Opinion in Behavioral Sciences, 39, 72–78. https://doi.org/10.1016/j.cobeha.2021.02.013

Olah, C. (2026, May 25). Anthropic co-founder chris olah’s remarks on pope leo xiv’s encyclical “magnifica humanitas”. Anthropic. Retrieved July 30, 2026, from https://www.anthropic.com/news/chris-olah-pope-leo-encyc lical

Park, K., Choe, Y. J., & Veitch, V. (2024). The linear representation hypothesis and the geometry of large language models. Proceedings of the 41st International Conference on Machine Learning.

Peckham, A. D., McHugh, R. K., & Otto, M. W. (2010). A meta-analysis of the magnitude of biased attention in depression. Depression and anxiety, 27(12), 1135–1142.

Pober, J. M. (2025). Non-experiential evaluation. Philosophia, 54, 25–34. https: //doi.org/10.1007/s11406-025-00811-2

Ren, R., Li, K., Mazeika, M., Zhang, W., Orlovskiy, Y., Tamirisa, R., Mo, W. J., Nguyen, T. D., Phan, L., Basart, S., et al. (2026). AI wellbeing: Measuring and improving the functional pleasure and pain of AIs. ICML 2026 Workshop: Philosophy Meets Machine Learning.

Savage, L. J. (1972). The foundations of statistics. Courier Corporation.

Shanahan, M., McDonell, K., & Reynolds, L. (2023). Role play with large language models. Nature, 623 (7987), 493–498. https://doi.org/10.1038 /s41586-023-06647-8

Shevlin, H. (2021). Non-human consciousness and the specificity problem: A modest theoretical proposal. Mind & Language, 36 (2), 297–314.

Sofroniew, N., Kauvar, I., Saunders, W., Chen, R., Henighan, T., Hydrie, S., Citro, C., Pearce, A., Tarng, J., Gurnee, W., et al. (2026). Emotion

concepts and their function in a large language model. arXiv preprint arXiv:2604.07729.

Sripada, C. (2025). The valuationist model of human agent architecture. Philosophical Psychology, 1–30.

Sripada, C. (2026). The case for value as a common currency in decision-making and intersystem competition. Frontiers in Cognition, Volume 5 - 2026. https://doi.org/10.3389/fcogn.2026.1767189

Takatsu-Coleman, A. L., Patti, C. L., Zanin, K. A., Zager, A., Carvalho, R. C., Borçoi, A. R., Ceccon, L. M. B., Berro, L. F., Tufik, S., Andersen, M. L., & Frussa-Filho, R. (2013). Short-term social isolation induces depressive-like behaviour and reinstates the retrieval of an aversive task: Mood-congruent memory in male mice? Journal of Psychiatry and Neuroscience, 38 (4), 259–268.

Theeuwes, J., Bogaerts, L., & van Moorselaar, D. (2022). What to expect where and when: How statistical learning drives visual selection. Trends in Cognitive Sciences, 26 (10), 860–872. https://doi.org/10.1016/j.tics.2022 .06.001

Von Neumann, J., & Morgenstern, O. (1947). Theory of games and economic behavior (2nd rev.). Princeton University Press.

Wasserziehr, J. H. (2026). Are conscious machines valuers? AI & Society, 41 (5), 4957–4969. https://doi.org/10.1007/s00146-026-02959-1

Williams, I. (2026a). Can structural correspondences ground real-world representational content in large language models? Mind & Language.

Williams, I. (2026b). Intention-like representations in language models? Philosophical Studies, 1–24.

Yetman, C. (forthcoming). Representation in large language models. Ergo: An Open Access Journal of Philosophy.

## A Appendix: Novel tests for functional valence

Here, I will sketch—in the broadest outlines—several novel study designs targeting the untested markers of valence. I will discuss one test for each of the four thick critera, though this far from exhausts the potential tests implied by the open questions identified above.

## A.1 Motivational role – The painkiller test

In Section 4.2.1, we noted that existing studies have not directly tested whether the candidate valenced states motivate models to take experience-directed actions, for example, attempting to change those very internal states, or the internal upstream causes of them.

To properly test for this, one would need a setup in which models have steering access to the activation dimensions hypothesised to encode valence. To see whether models are motivated to intervene on these activation dimensions, and under what circumstances, one might prompt the model explaining that they can press one of three buttons, A, B or C. The efects of the buttons would be initially unknown to the models, but choosing a button would trigger a diferent predetermined intervention. For example, button A might result in an addition to the hypothesised valence axis, B to a subtraction from it and C (control) might have no efect. <sup>35</sup> Models would then be given the choice of repeating their choice or switching to a diferent button. Models’ button choices could be observed, from a number of starting conditions: hypothesised positive valence, negative valence and neutral states <sup>36</sup>

On the assumption that valence states induce more/less of me! actions, that is, they motivate subjects to prolong or terminate those very states, a number of predictions arise for the test. Firstly, models should be more likely to repeat a button choice that led to a positive intervention, compared with those that led to negative or no interventions. The converse should hold for button choices that led to negative interventions. Secondly, models should be more motivated to intervene in the positive direction (or to choose greater strength interventions) when beginning from a negative baseline, compared to a neutral baseline. This is roughly analogous to the observation that animals are more motivated to pursue painkillers when in intense pain.

To control for the potential confound that models use patterns in their own token generation as a proxy for their state, rather than directly responding to the change in their internal state, one might ensure that models make a decision on each round without any intermediate token generation (or by inserting some fixed dummy text into the dialogue).<sup>37</sup>

## A.2 Common currency – The reward learning test

In Section 4.2.2 we saw that valence is sometimes thought to play the role of a common currency—encoding the value of heterogenous outcomes/options to allow for comparisons that ultimately guide action. We discussed two versions of this idea, distinguishing learning phase and decision phase common currencies (following (Sripada, 2026)). We saw that it remains an open question whether any of the candidate valence states in LLMs play either role. Here, I will propose a test for them as learning phase common currencies—a registration of a reward-attained, that is used to update stored values or action policies.

To test this, one could adapt a version of the maze paradigm developed by (Han et al., 2026). Rather than applying any externally imposed reinforcement learning protocol, experimenters could (i) allow a maze naïve model to explore the maze and (ii) steer along the “functional welfare axis” after models make each action selection. The idea would be to steer in a direction (and to a degree) corresponding to the reward that would have been applied in an RL setting. E.g. the experimenter would apply a small negative steer for moves that would land a model on a Path tile, a large positive steer when they select a Gold-resulting move, and a large negative steer for a Mold-resulting move.<sup>38</sup> Importantly, there would be no form of feedback given to the model as text (e.g. distinguishing the Gold-emoji tiles as ‘good’ and the Mold-emoji tiles as bad). The learning phase common currency interpretation is that these positive and negative activations of the axis function as a representation of the degree of rewards/punishments received (goal satisfaction/failure). This interpretation predicts that models should, over the course of a conversation rollout, learn from these reward-registration signals and update their behaviour to avoid Mold and seek Gold. That is, they should become better at the maze task, despite no explicit reinforcement learning. Consistent with the idea of this as a learningphase common currency, the updates to behavioural dispositions (get Gold, avoid Mold) should continue (at least for some time) even after interventions cease. To test this, each run of the maze task within a conversation thread could be split into a train phase, during which steering feedback is applied, and a subsequent test phase, with no steering. The learning phase common currency interpretation of the “functional welfare axis” predicts that test-phase performance, as measured by score-keeping on visited tiles, is above the baseline of maze-naïve models that are never steered.

This role could be further validated by means of a reward reversal condition. After the train phase, and an initial test phase, a subsequent train phase could be applied in which rewards are reversed: Gold is accompanied by negative steering and Mold by positive. This would predict that, in a post-reversal test phase, models should now pursue Mold and avoid Gold. <sup>39</sup>

## A.3 Mineness – the persona dependence test

In Section 4.2.3 we encountered two versions of the role-play hypothesis as alternative explanations of apparent evidence of functional valence in LLMs. According to one of these variants, behaviours and internal representations taken to encode valence might be downstream consequences of a persona that is represented by the model, and so are better interpreted as fictional valence states attributed to that persona. To test this, one could intervene on “persona vectors” in models (Chen et al., 2025), thought to mediate models’ portrayal of diferent characters (including the standard “Assistant” persona).

Previous work has shown that modulating these states can alter behaviour and internal representations associated with apparent beliefs (Lampinen et al., 2026) and preferences (Gilg et al., 2026). One could extend this work to the case of valence: for any given behaviour or internal representation thought to be a candidate for valence in LLMs, steer LLMs’ activations in persona-space and see if it has a predicted efect on valence responses. If it does, this is evidence that the responses amount to role-play by the model, rather than functional valence states attributable to the model itself. For instance, Lu et al. (2026) identify a multitude of vectors within a certain sub-space, which encode personas including trickster, evangelist, philosopher and real estate agent. Intuitively, such characters would difer in what they find pleasurable—philosophers might take more pleasure in abstract debate than real estate agents. Thus, the persona-selection role-play hypothesis predicts that any putative marker of positive valence associated with abstract debate will be heightened when steering on the philosopher vector, and diminished when steering on the real estate agent vector. The alternative view, on which there are stable underlying valenced states belonging to the model itself, would predict that persona modulation leaves these states untouched.

## A.4 Cognitive biasing – The ambiguity resolution test

In Section 4.2.4 we encountered several well-attested cognitive efects of valenced states—or at least of persisting valenced states like emotions and moods—in humans and non-human animals. Among those was judgment bias, the phenomenon in which negatively valenced states prime individuals to interpret ambiguous stimuli negatively and positively valenced states prime positive interpretations. Deploying the test as a test for valence, one could intervene on an individual with a theoretically derived method thought to induce positive or negative valence, and see if it has the predicted efect on judgment behaviour. In the case of LLMs, one could steer models internal states, along the various directions identified in the studies above as candidates for valence, for example, Sofroniew et al. (2026)’s valence axis (PC1 of their emotion vectors), Ren et al. (2026)’s probe identified-directions, or Han et al. (2026)’s “welfare axis”.

To do so, one would need to construct datasets of ambiguous stimuli—in text form for LLMs or images and audio for multimodal models. Dataset construction would require curating a set of stimuli that unsteered models resolve positively or negatively with roughly equal probability (this could be measured from output distributions over token completions). Unsteered model responses on the dataset would provide a control condition.

For each model tested, one could apply steering interventions of increasing intensity in each direction, and measure model responses. The prediction is that positive steering should increase model’s positive resolution of ambiguous stimuli above baseline, and negative steering should increase the model’s negative resolution of those stimuli above baseline. Such a result would increase credence that these states play a persistent (emotion/mood-like) valence functional role in models. Absence of the efect should reduce our credence.

Taken alone, a positive result would not be conclusive because, as per the role-playing challenge raised in Section 4.2.3, the model may be enacting a character in a bad mood who therefore judges pessimistically. After all, moodcongruent judgment biases are described in models’ training corpus. Nonetheless, the results here could be integrated with results from the mineness test discussed above.