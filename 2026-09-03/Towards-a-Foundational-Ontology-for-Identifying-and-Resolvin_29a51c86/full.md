# Towards a Foundational Ontology for Identifying and Resolving Contradictions in Dialogue-based Human-Robot Interactions

Maitreyee Tewari

Nord A1, Sweden

Alternative Intelligence LLP

Bologna, Italy

ORCID: 0000-0002-3036-6519

Michele Persiani

Department of Informatics

University of Bologna

Bologna, Italy

ORCID: 0000-0001-5993-3292

Abstract—Existing Human-Robot Interaction (HRI) literature has focused on identifying and structuring errors, failures, conflicts, and knowledge issues (called in this work as contradictions) in domain-specific dialogue-based interactions. However, there is still lack of a formal computational framework to represent and define these contradictions, interoperable and usable across HRI and human-agent interaction (HAI) domains. Thus, this research project aims to capture, represent, and evaluate the notion of (1) dialogue-based collaborative interaction and (2) related contradictions in a foundational ontology. METHONTOLOGY, a systematic approach to build domain-independent knowledge graphs was applied. In its conceptualisation stage concepts and models from Activity Theory were used. Preliminary results presented in this work are (i) natural language definitions of concepts relevant to contradictions, (ii) Set Theoretic definitions of those concepts, and (iii) First Order Logic (FoL) formulation of the contradiction concepts and three novel principles guiding human dialogue-based interactions with agents. In summary, we report on ongoing work to develop a foundational ontology based on Activity Theory called Activity Theory-based foundational ontology (ATFOt) to capture and represent the notion of contradictions in HRI.

Index Terms—Knowledge Issues, Conflicts, Activity Theory, Engeström’s Activity System Model, Contradictions, Human-Robot Interaction, Human-Agent Interaction, Knowledge Representation, Foundational Ontologies, First Order Logic, Set Theory.

## I. Introduction

Large Language Models (LLM) have been shown to generate coherent and contextual text similar to human level language generation capabilities. LLMs integrated in virtual assistants (VAs) such as Alexa, Siri, and ChatGPTs have surpassed human expectations in query-based interactions [1]. This success of LLMs with VAs prompted the sudden and rapid integration of LLMs in dialogue-based HRI [1]–[3]. However, this integration of LLMs in HRI merely underscored the already existing limitations such as lack of understanding about the semantics, intent, and norms, lack of contextual knowledge, strategies to manage errors and uncertainties (HRI) [3], [4]. Furthermore, researcher’s [5] found that LLM technology’s limitation of confabulation, (i.e., to generate seemingly coherent yet false and manufactured information [6]) coupled with inconsistent and lack of uncertainty expressions makes them incompetent for ethical HRI. These findings adds to the already existing complex problems of unpredictability in dialogue-unfolding that generally causes communication failures, speech errors, knowledge issues, and conflicts in dialogue-based HRI [4], [7]–[10].

To address failures, knowledge issues, and conflicts, researchers have proposed taxonomies representing HRI failures related to capabilities of robots [10], social errors [7], and conflicts and knowledge issues [11]. Methods to identify and manage failures through speech and gestures [12], interaction patterns [1]; conflict of intentions using Answer Set Programming [13] have been proposed. Given this prior work, however, there is still a lack of structured and granular knowledge representation of communication errors and failures, conflicts, and knowledge issues arising in dialogue-based HRI.

Therefore, the objective of this research is to formally represent the knowledge of dialogue-based interactions (HRI and HAI) and related communication errors, conflicts, and knowledge issues with a foundational ontology. Following the naming convention from Activity Theory, that describes human activity as purposeful, evolving and transforming the actor and the subjective world [14]–[16],) this work refers to errors, knowledge issues, and conflicts as contradictions [17]–[19]. In the rest of the article, knowledge issues and conflicts will be called collectively as contradictions.

The work presented in this article, partially fulfilling the research objective, explores the research question:

How can we create a general representation of contradictions in dialogue-based HRI within a foundational ontology?

To address the research question, activity analysis using the Activity System Model and the concept of contradictions from activity theory was conducted on HRI [8] and HAI (digital and virtual agent) [20], [21] scenarios. These scenarios embedded varying types of contradictions. The analysis informed the development of a foundational ontology encoding a hierarchy of contradictions representing errors, conflicts, and knowledge issues for dialogue-based HRI and HAI. The hierarchy categorises these conflicts and knowledge issues at four levels of abstraction (similar to the four levels of contradictions in activity theory).

The contribution of this article is a novel formal knowledge representation of contradictions and three novel principles guiding and transforming the dialogue-based interactions and the involved participants.

The rest of the article is structured as follows. Section II briefly presents the methodology being applied to develop the ontology. The formal specification of contradictions using Set Theory and First Order Logic (FoL) and relevant definitions from the ontology are presented in Section III. Section IV concludes the article, informing the status and research contributions of the work.

## II. Methodology

To develop the ontology, METHONTOLOGY [22] provides six activities that need to be performed: (1) specification, (2) knowledge acquisition, (3) conceptualisation, (4) integration, (5) implementation, and (6) evaluation.

The specification activity defines the purpose and usecase, the level of formality, scope, characteristics, and granularity of the ontology. The outcome is a specification document in natural language.

Conceptualisation structures the domain knowledge in a conceptual model describing the problem and the solution in the specification document.

During integration, concepts of the developed conceptual model are mapped to existing meta ontologies. The outcome is a document associating the terms in the referenced meta ontologies to model’s concepts.

The development activity commences by formally representing the concepts using logic-based methods. Then a language and an appropriate environment to develop the ontology is selected. We are using Set Theory and First Order Logic to formally define and represent the contradictions concepts. Protege is being used to implement and evaluate the ontology.

In the evaluation activity, the ontology is inspected by verifying its correctness (verification process) and validating that the ontology represents the intended system knowledge [22].

Figure 1 illustrates the phases of METHONTOLOGY adopted in this work to build the ontology. The ontology was developed using all the five stages. The integration stage is embedded within the first and second phases of specification and conceptualisation.

![](images/09f3e24c2accbabf6f5fac3428e43ad367cdfe146f5d82df35ede4727bc1a6af.jpg)  
Fig. 1: Central phases of METHONTOLOGY illustrated in white boxes. The other additional phases of knowledge acquisition, implementation, and documentation are illustrated in grey with the darker gradient implying higher activity in the corresponding phases. The figure is an adaptation from [22] modified using another ontology design workflow introduced recently in [23].\*\*Copyright Esteban Guerrero.

## III. Formalization of Contradictions

A. Defining Object Properties and First Order Logic Formalism

Object is an instantiation of desires and intentions of an entity. Objects exist in a hierarchy. At the highest complexity level the object is the motive that an entity wants to attain. At lower levels its the goals of actions, through which the subject is interacting with the world, which can also be represented as physical afordances (as defined by Gibson [24]) in this article [25, Chp. 3]. Then an object can formally be defined as:

Definition III.1 (Object). Let $D _ { n }$ be the set of all the desires of an entity, $I _ { n }$ be the set of all the intentions, $G _ { n }$ represent goals of an entity, and $P _ { n }$ be all the physical afordances ( [24]). The object is composed of multiple levels, each one comprised of needs, desires, intentions, goals and physical afordances. Each level n of the object can be described as a tuple $\langle N _ { n } , D _ { n } , I _ { n } , G _ { n } , P _ { n } \rangle$ , where:

$\textstyle { \mathcal { N } } _ { n }$ : the set of needs at level n

$D _ { n }$ : the set of desires at level n

$I _ { n }$ : the set of intentions at level n

$G _ { n }$ : the set of goals at level n

$P _ { n }$ : the set of afordances at level n

Levels of abstraction of the object are connected from a level to the next, by the following function:

$$
f _ { N } : \mathcal { N } _ { n } \to D _ { n + 1 }
$$

$$
f _ { N } : D _ { n } \to I _ { n + 1 }
$$

$$
f _ { N } : I _ { n } \to G _ { n + 1 }
$$

$$
f _ { N } : G _ { n } \to P _ { n + 1 }
$$

$\mathcal { N } , D , I , G , P$ have their own sets

Actors are sentient entities (specifically, humans) living and acting in the world to fulfil their needs. An actor becomes a subject when they have both “the ability and need to $a c t ^ { \dprime }$ transforming the world and the self by acting in it. Subject can be an individual or subgroup [18].

We expand the definition of subject to include insentient entities such as software, digital and embodied agents driven by artificial intelligence that have agency, and adopt or support human goals. These insentient entities can transform the world, but may or may not have the capability to change themselves.

Definition III.2 (Subject). Let actors A be the set of all the sentient (humans and animals) ${ \cal { S } } = \{ s _ { 1 } , s _ { 2 } \ldots , s _ { i } \}$ and insentient entities (software, virtual, embodied, and digital agents) $E = \{ e _ { 1 } , e _ { 2 } \ldots , e _ { j } \}$ , such that $A \subseteq S \cup E$ . Then a specific subject is an actor given as $s b j \in A$

Mediating Artefacts are the symbols, signs, and understanding of the concepts used by the subject to mediate with the objects [26]. They have a function to reverse the action, transforming the psychological operations to new forms, allowing the humans to “control their behaviour from the outside” [27, pp. 39-40].

Mediating artefacts can be distinguished between technical and psychological artefacts. Both of these are mediators in the activity.

Technical Tools are used to afect other things.

Psychological Tools are signs used by the subject to afect other entities or themselves [25, pp. 42], [15]. Psychological tools can be physical artefacts such as art and diagrams or symbolic systems such as mathematical symbols and language.

Definition III.3 (Mediating Artefacts). Let PT be the set of all psychological artefacts and TE be the set of all the technical artefacts, then mediating artefacts MT is the union given as, MT ⊆ TE ∪ PT.

Activity system model (ASM) is an “object-oriented, collective, and culturally mediated human activity, or activity system. [19].” Represents a model of collective activities, where the object is shared by a community and collectively carried out. The activity is an interaction between the subject, the object, and the community. These interacting components are mediated through mediating artefacts, rules, and division of labour [18]. A dialogue-based HRI is structured in this work as an ASM.

The community is a group composed of the actors and other entities that have a shared object. For instance in a dialogue-based HRI scenario in healthcare, the robot, the human needing care, and other care providers are part of a community, sharing a common object to supporting the wellbeing of the human.

Rules are the mediating components between the subject and the community. They are explicit and implicit rules, norms, and conventions regulating the activity [25], such as in a medication scenario, medicines should be followed according to the prescription, dialogues between the human and the robot should follow the strategies set out by the designers.

Division of labour are the mediating component between the object and the community. The subject uses this component to coordinate their tasks with the community, creating a mediation between the community and the object (i.e., the motive) [25]. In a collaborative dialoguebased HRI healthcare scenario, a registered doctor prescribes medication, the human takes the medications, and the robot supports the human in this activity, through reminders and .

Outcome is the intended result generated by the transformation of the object through the collaborative activity. The outcome in dialogue-based HRI healthcare scenario will be managing medications successfully.

Definition III.4 (Collaborative Activity). Let a dialogue-based HRI be an ASM, represented as a collaborative activity ca. Let the collaborative activity be an interaction between a community of formal and informal care providers comm, the subject (human or the robot) sbj and the object (supporting health are wellbeing) obj, mediated by artefacts (medication box, prescription, dialogue strategies) mt ∈ MT . The socio-cultural context SOC consisting of the set of rules R and the set of division of labour DOL for a given activity, SOC ⊆ R ∪ DOL. Then a collaborative activity at a given time point tp is given as a function ca(mt, sbj, o, comm, soc, $t p ) = \langle s b j ^ { \prime } , o ^ { \prime } \rangle$

## B. FoL formalism of ASM

We use the object properties defined in SectionIII-A and relationships from the foundational ontology we are developing, called Activity Theory-based Foundational Ontology (ATFOt) for FoL formalisation of ASM (Definition III.4). Prefix atot is used to refer to the foundational ontology in the rest of the formalisation.

# Definition III.5.

DialHRI(ca) ≡ atot.M(ca) ∧ ∃ (x, y, obj, mt, soc, t(x ̸= y), (x, y ∈ comm)) ∧ atot.Subject(x) ∧ atot.hasNeeds(n, x) ∧ atot.Subject(y) ∧ ∧atot.AdoptsNeeds(an, y) ∧ atot.hasObject(obj) ∧ atot.hasTools(mt) ∧ atot.hasContext(soc) ∧   
atot.hasT imeInterval(t) ∧ atot.Employs(s, mt, obj) ∧ ∃ (z)(atot.Subject(z) ∧ atot.adoptsNeeds(n, z)) →   
atot.adheresTo(z, soc) ∧ atfot.Transforms(obj, out)

The definition reads as follows: In an ASM, a subject (s) either with intrinsic needs (sentient) (n) or by adopting the needs of the others (insentient)(an), uses mediating artefacts (mt) within a given socio-cultural context (soc) on the object (obj). The subject and the community (z) adhering to the the social context (soc), transform the object to achieve the desired outcome (out).

Contradictions “are historically emergent and systemic phenomena. Contradictions must therefore be approached through their manifestations. [18, pp. 8].” We define them as situations arising due to the discrepancies within the central ASM or with the members of the ASM family. Contradictions are opportunities to learn and develop a better understanding of the activity and other components of an ASM [8]. Internal contradictions lead to transformative change in the components that triggers the activity to change [28]. Contradictions manifest in the form of a conflict, inconsistency, paradox, tension, or a dilemma [18].

Primary Contradictions reside within the components of an ASM [18], such as a missing artefact, or when it breaks down.

Secondary Contradictions emerge between diferent constitutive components of an ASM [18], such as due to lack of knowledge about the rules or how to use the mediating artefact.

Tertiary Contradictions “appear when the activity system is reshaped and the new pattern collides with vestiges of the old one, generating resistance and forcing the new model to be modified” [18]. When new technologies are introduced in old processes, transforming the activity to a culturally more-advanced activity. Fear, excitement, and uncertainty when a robot is introduced in a care home for the first time.

Quaternary Contradictions “Takes shape when the transformed activity system interacts with its partner activities, generating tensions, disturbances and innovations in the network relations of involved activity systems” [18]. For example, in an HRI collaboration between a human and a robot, intended to support in healthcare management, a conflict between the human and the robot about following the doctor’s prescription, could negatively impact human agency and autonomy. This is a contradiction with the existing rules around prioritizing human agency and autonomy in HRI.

Definition III.6 (Contradiction). If SM is the Engestrom’s Activity System Model (ASM), then let a set of objects in the supporting ASMs be represented as $( o b j _ { p } , \dots , o _ { q } \in S M _ { s u p p o r t } )$ . Let $( s _ { i } , ~ s _ { j } \in \ S M _ { c e n t r a l } )$ be the two random components of the central ASM. Let $o b j _ { n } \in S M ^ { c e n t r a l }$ be the object of the central ASM, and let $o b j _ { m } \in S M _ { a d v a n c e d }$ be the object of the culturally more advanced ASM. Then we define four types of contradictions Contra as follows:

Primary Contradiction: within a specific component of the central ASM $C o n t r a _ { p r i } ( s _ { i } )$

Secondary Contradiction: between a pair of components of central collaborative activity

$$
C o n t r a _ { s e c } ( s _ { i } , s _ { j } )
$$

Tertiary Contradiction: is between the object of the central ASM and the culturally more advanced $\mathrm { A S M } \ C o n t r a _ { q u a t } ( o b j _ { p }$

Quaternary Contradiction: is between the object of the central ASM and one of the supporting ASMs $C o n t r a _ { t e r } ( o b j _ { k } , o b j _ { m } )$

A contradiction is an event that causes a focus shift (fs) from the object of the central ASM, denoted as $o _ { i } \in S$ to the alternative ASM with another object $o _ { k } \in S$ where the contradiction will be resolved [11]. As such, contradictions can be represented as a mapping function:

$$
f s : S  S\tag{1}
$$

1) FoL Formalism of Contradictions: The formalism of contradictions is based on a formal specification of three novel principles that rule every collaborative human activity. These principles are relevant for the computational reasoning about an ongoing activity represented in an ontology. Informally speaking these guiding principles define some of the transformations in the ontology due to contradictions, as follows:

Principle 1: If there is a contradiction in the central ASM, it transforms to be the object, transitioning the focus to an alternative ASM.

$$
\begin{array} { c } { { \mathbf { A 1 : } } } \\ { { \forall ( c ) ~ a t o t . C o n t r a ( c ) \wedge \exists ( o b j ) ~ a t o t . S M _ { c e n t r a l } ( o b j )  } } \\ { { a t o t . S M _ { a l t e r n a t i v e } ( o b j ^ { \prime } ) \wedge \lnot a t o t . S M _ { c e n t r a l } ( o b j ) . } } \end{array}
$$

Principle 2: If there is a contradiction in the central ASM, then there is a conflict with the ASM family.

$$
\begin{array} { c } { { \bf A 2 : } } \\ { { \forall ( c ) ~ a t o t . C o n t r a ( c ) \wedge \exists ( c a ) ~ a t o t . S M _ { c e n t r a l } ( c a )  } } \\ { { \exists ( c a ) ~ a t o t . S M _ { f a m i l y } ( c a ) \wedge } } \\ { { a t o t . C o n f l i c t ( S M _ { f a m i l y } ( c a ) ) \wedge \lnot S M _ { c e n t r a l } ( c a ) . } } \end{array}
$$

Principle 3: If there is a contradiction in the central ASM then there is a contradiction in its inherited ASMs, i.e., the culturally more-advanced ASM and its related ASM family.

$$
\begin{array} { c } { { \mathbf { A 3 : } } } \\ { { \forall ( c ) ~ a t o t . C o n t r a ( c ) \wedge \exists ( c a ) ~ a t o t . S M _ { c e n t r a l } ( c a ) \to } } \\ { { \exists ( c ) ~ C o n t r a ( c ) \wedge \exists ( c a ) S M _ { i n h e r i t } ( c a ) } } \end{array}
$$

## IV. Conclusion and Ongoing Work

This research presented preliminary results on the ontological formalism of dialogue-based interactions for HRI and HAI and related contradictions. Engestrom’s Activity System Model (ASM) and the concept of contradictions from Activity Theory were used as a foundational theory in the conceptualisation and definition of contradictions and dialogue-based interactions.

This research contributed with a novel formalisation of contradictions concepts and three novel principles guiding the transformations in an ontology caused by the inherent contradictions. The presented work is part of an ongoing research on developing a foundational ontology called activity theory-based foundational ontology $( A T F O t )$ . ATFOt represents collective activity and related contradictions such as errors, failures, conflicts, and knowledge issues in dialogue-based interactions between humans and agents.

Currently, we are finishing the formalisation and implementation of ATFOt using set theory, FoL, and Protege environment, respectively. The implemented ontology will be evaluated qualitatively in studies and verified using coverage and competency questions.

ATFOt can be used by agents intended to interact with humans to formally represent, reason about, and manage dialogues and related contradictions. This ontology is a step towards reliable, transparent, and trustworthy dialogue management strategies for both learning and rule-based agents (embodied, digital, or virtual.) This ontology can be used to standardise the terminology for dialogues and relevant contradictions. This standardisation advances the research in contradiction management by enabling the reusability and interoperability of the terminologies across domains in human-agent dialogues.

## References

[1] A. Mahmood, J. Wang, B. Yao, D. Wang, and C.-M. Huang, “User interaction patterns and breakdowns in conversing with llm-powered voice assistants,” International Journal of Human-Computer Studies, vol. 195, p. 103406, 2025. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S107158192

[2] R. Wen, F. Ferraro, and C. Matuszek, “Gpt-4 as a moral reasoner for robot command rejection,” in Proceedings of the 12th International Conference on Human-Agent Interaction, ser. HAI ’24. New York, NY, USA: Association for Computing Machinery, 2024, p. 54–63. [Online]. Available: https://doi.org/10.1145/3687272.3688319

[3] A. Iida, K. Okuoka, S. Fukuda, T. Omori, R. Nakashima, and M. Osawa, “Integrating large language model and mental model of others: Studies on dialogue communication based on implicature,” in Proceedings of the 12th International Conference on Human-Agent Interaction, ser. HAI ’24. New York, NY, USA: Association for Computing Machinery, 2024, p. 260–269. [Online]. Available: https://doi.org/10.1145/3687272.3688303

[4] S. Cao, M. Stiber, A. Mahmood, M. T. Parreira, W. Ju, M. Spitale, H. Gunes, and C.-M. Huang, “Err@hri 2.0 challenge: Multimodal detection of errors and failures in human-robot conversations,” in Proceedings of the 33rd ACM International Conference on Multimedia, ser. MM ’25. New York, NY, USA: Association for Computing Machinery, 2025, p. 14130–14135. [Online]. Available: https://doi.org/10.1145/3746027.3762073

[5] A. Toney-Wails and L. Singh, “Are you sure about that: Eliciting natural language confidence indicators from chatbots,” in Proceedings of the 12th International Conference on Human-Agent Interaction, ser. HAI ’24. New York, NY, USA: Association for Computing Machinery, 2024, p. 278–286. [Online]. Available: https://doi.org/10.1145/3687272.3688318

[6] A. L. Smith, F. Greaves, and T. Panch, “Hallucination or confabulation? neuroanatomy as metaphor in large language models.” PLOS Digit Health, 2023.

[7] TianLeimin and OviattSharon, “A taxonomy of social errors in human-robot interaction,” ACM Transactions on Human-Robot Interaction (THRI), vol. 10, 2 2021. [Online]. Available: https://dl.acm.org/doi/abs/10.1145/3439720

[8] M. Tewari, “Breakdown situations in dialogues between humans and socially intelligent agents,” Ph.D. dissertation, Umeå University, 2023.

[9] E. Kox, M. Hennekens, J. Metcalfe, and J. Kerstholt, “Trust violations due to error or choice: The diferential efects on trust repair in human–human and human–robot interaction,” J. Hum.-Robot Interact., vol. 14, no. 4, Aug. 2025. [Online]. Available: https://doi.org/10.1145/3743694

[10] B. Nogueira, J. Ferreira, H. Simão, F. Rocha, I. Neto, J. Guerreiro, and T. Guerreiro, “Don’t go breaking my trust: A taxonomy of interaction failures for transparent robots,” in Workshop on Designing Transparent and Understandable Robots, 2026. [Online]. Available: https://openreview.net/forum?id=pX5ztNGY2O

[11] M. Tewari and H. Lindgren, “Expecting, understanding, relating, and interacting-older, middle-aged and younger adults’ perspectives on breakdown situations in human-robot dialogues,” Frontiers in Robotics

and AI, vol. 9, 2022. [Online]. Available: https://www.frontiersin.org/articles/10.3389/frobt.2022.956709

[12] L. Wachowiak, P. Tisnikar, A. Coles, G. Canal, and O. Celiktutan, “A time series classification pipeline for detecting interaction ruptures in hri based on user reactions,” in Proceedings of the 26th International Conference on Multimodal Interaction, ser. ICMI ’24. New York, NY, USA: Association for Computing Machinery, 2024, p. 657–665. [Online]. Available: https://doi.org/10.1145/3678957.3688386

[13] E. Guerrero, M. Tewari, P. Kalmi, and H. Lindgren, “Forming we-intentions under breakdown situations in human-robot interactions,” Computer Methods and Programs in Biomedicine, vol. 242, pp. 107–817, 2023.

[14] A. N. Leontiev, “Problems of the development of the mind (originally published in russian in 1959).” 1981.

[15] L. S. VYGOTSKY, “Instrumentalnyj metod v psikhologii [ instrumental method in psychology],” in Sobranie Sochinanij, t. 1 [L. S. Vygotsky, Collected Works. Moscow: Pedagogika (In Russian), 1982, pp. 103–109.

[16] V. Kaptelinin and B. Nardi, Activity Theory in HCI: Fundamentals and Reflections, 04 2012, vol. 5.

<sub>01897</sub>[17] Y. Engeström, Activity theory and individual and social transformation, 1999, vol. 19, no. 38, pp. 19–38.

[18] A. Sannino and Y. Engeström, “Cultural-historical activity theory: founding insights and new challenges,” Cultural-Historical Psychology, vol. 14, no. 3, pp. 43–56, 2018.

[19] Y. Engeström and R. Miettinen, Introduction. Cambridge University Press, 1999, vol. 19, no. 38, pp. 1–16.

[20] V. C. Kaelin, M. Tewari, S. Benouar, and H. Lindgren, “Developing teamwork: transitioning between stages in humanagent collaboration,” Frontiers in Computer Science, vol. 6, p. 1455903, 2024.

[21] H. Lindgren, V. C. Kaelin, A.-M. Ljusbäck, M. Tewari, M. Persiani, and I. Nilsson, “To adapt or not to adapt? older adults enacting agency in dialogues with an unknowledgeable agent,” in Proceedings of the 32nd ACM Conference on User Modeling, Adaptation and Personalization, ser. UMAP ’24. New York, NY, USA: Association for Computing Machinery, 2024, p. 307–316. [Online]. Available: https://doi.org/10.1145/3627043.3659562

[22] M. Fernández-López, A. Gomez-Perez, and N. Juristo, “Methontology: from ontological art towards ontological engineering,” Engineering Workshop on Ontological Engineering (AAAI97), 03 1997.

[23] M. Poveda-Villalón, A. Fernández-Izquierdo, M. Fernández-López, and R. García-Castro, “Lot: An industrial oriented ontology engineering framework,” Engineering Applications of Artificial Intelligence, vol. 111, p. 104755, 2022. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0952197622000525 [24] J. G. Greeno, “Gibson’s afordances.” 1994.

[25] V. Kaptelinin and B. A. Nardi. MIT press, 2006.

[26] M. S. Kabuye Batiibwe, “Using cultural historical activity theory to understand how emerging technologies can mediate teaching and learning in a mathematics classroom: a review of literature,” Research and Practice in Technology Enhanced Learning, vol. 14, 10 2019.

[27] L. S. VYGOTSKY, Mind in Society: Development of Higher Psychological Processes. Harvard University Press, 1978. [Online]. Available: http://www.jstor.org/stable/j.ctvjf9vz4

[28] L.-M. U. Schröder, A. E. J. Wals, and C. S. A. K. van Koppen and, “Analysing the state of student participation in two eco-schools using engeström’s second generation activity systems model,” Environmental Education Research, vol. 26, no. 8, pp. 1088–1111, 2020. [Online]. Available: https://doi.org/10.1080/13504622.2020.1779186