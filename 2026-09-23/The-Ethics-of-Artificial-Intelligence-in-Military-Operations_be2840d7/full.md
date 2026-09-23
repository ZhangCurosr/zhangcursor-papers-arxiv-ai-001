# The Ethics of Artificial Intelligence in Military Operations

Nicolas Drapier<sup>1,2\*</sup>, Florian Mauberger<sup>2</sup>, Aladine Chetouani<sup>1</sup>, Aur´elien Chateigner<sup>2</sup>

<sup>1\*</sup>L2TI Laboratory, Universit´e Sorbonne Paris Nord, 99 Avenue Jean-Baptiste Cl´ement, Villetaneuse, 93430, France. <sup>2</sup>SAS Impact, 1 Rue Sainte-Anne, Orl´eans, 45000, France.

\*Corresponding author(s). E-mail(s): nicolas.drapier@sas-impact.fr; Contributing authors: florian.mauberger@sas-impact.fr;   
aladine.chetouani@univ-paris13.fr; aurelien.chateigner@sas-impact.fr;

## Abstract

Deep learning systems now mediate military decisions to use force, yet their internal logic resists inspection, their evaluation practices are gameable, and their deployment fractures accountability across dispersed stakeholders. The ethical challenge posed by these systems is fundamentally epistemic: not just whether autonomous weapons should be permitted to kill, but whether the conditions for responsible human judgment can survive when critical functions are delegated to opaque algorithms.

We show that this epistemic condition produces a concrete accountability gap: responsibility difuses across designers, operators, and policymakers while International Humanitarian Law presupposes capacities for judgment that current AI systems lack. To address this gap, we propose a governance framework that proceduralizes ethical constraints through named accountability roles, adversarial auditing with undisclosed benchmarks, tiered deployment thresholds, and a proposed NATO evaluation standard.

Counterfactual analysis of eight documented cases (1988-2025) shows that each governance mechanism addresses a documented class of failure, but no single safeguard sufices in isolation: efective governance of military AI requires not only technical constraints but the institutional infrastructure to keep human judgment meaningful.

Keywords: Military AI, Autonomous weapons, Accountability, International Humanitarian Law

## 1 Introduction

The ethical debate surrounding military AI has largely centered on a binary question: should autonomous systems be permitted to select and engage targets? This framing has motivated important policy initiatives, including the open letter by AI researchers calling for a ban on ofensive autonomous weapons [22] and the ongoing deliberations within the UN Convention on Certain Conventional Weapons. Yet it obscures a more fundamental problem. The challenge is not only whether machines should be allowed to kill, but whether the epistemic conditions for responsible decision-making can be preserved when critical functions are delegated to systems whose internal logic resists inspection.

These systems are already operational. Targeting, surveillance, logistics, and command support increasingly depend on deep learning operating at speeds and scales beyond human cognitive capacity. We call this shift Algorithmic Warfare: the integration of autonomous inference into the chain of command, from sensor processing to engagement decisions.

This paper argues that the ethical crisis of military AI is, at root, an epistemic crisis. The technologies driving this transformation range from autonomous targeting to predictive logistics [64], and their proliferation is reshaping strategic stability [27]. Deep learning models are opaque in ways structurally distinct from prior military technologies [11, 37]: their decision processes elude human comprehension even when source code is available, and post hoc explainability techniques ofer approximations that do not restore access to the model’s reasoning [8, 76]. This opacity compounds across the chain of command: it distorts operator trust [23, 41], fragments accountability across dispersed stakeholders [44, 53, 69], and strains the applicability of International Humanitarian Law, whose core principles presuppose capacities for judgment that current AI systems lack [6, 7, 26, 66]. Responsibility does not vanish when decisions are delegated to algorithms. It is displaced, obscured, and redistributed through institutional mechanisms that simulate accountability without delivering it.

Our central claim is that governable military AI requires epistemic infrastructure: institutions, procedures, and technical constraints designed to preserve the conditions for human judgment under irreducible uncertainty. This is not a call to halt military AI adoption, which would be neither realistic nor strategically responsible, but to ensure that deployment proceeds within governance structures adequate to the technology’s demands. This paper makes two contributions:

1. A diagnostic framework showing how the structural opacity of deep learning interacts with adversarial fragility, benchmark gaming, and defense-contracting incentives to produce three compounding epistemic failures (an illusion of understanding, an illusion of accuracy, and an illusion of determinism) that erode both accountability and the practical applicability of International Humanitarian Law (2).

2. A procedural governance framework (named accountability roles, adversarial auditing with undisclosed benchmarks, a proposed NATO evaluation standard, tiered deployment thresholds, and interface design principles) evaluated through counterfactual analysis of eight operational cases spanning 1988-2025 (3).

## 2 Background

## 2.1 Opacity, Explainability, and Adversarial Fragility

The three sources of algorithmic opacity identified by Burrell [11] converge simultaneously in military AI: proprietary algorithms and classification regimes enforce deliberate dissimulation, rapid development cycles produce organizational ignorance, and deep neural networks are mathematically opaque by construction [37]. Post hoc explainability methods such as LIME [60], SHAP [40], and Grad-CAM [65] can aid debugging [77], but these are approximations that can be unfaithful to the model’s actual decision process [8, 76], and explanation quality fails to predict human-AI team performance [10]. In military applications the deeper risk is what we term an illusion of understanding. An analyst viewing a heatmap concentrated on a vehicle’s turret may conclude the model identified a tank by its weapon system, while the model may be keying on background terrain or sensor artifacts. The explanation satisfies the cognitive need for justification without providing epistemic access. Partial transparency is therefore more dangerous than acknowledged opacity: complete opacity preserves skepticism, while the illusion of understanding actively suppresses it.

Adversarial fragility compounds this danger. Physically realizable perturbations systematically fool classifiers [9, 18], defenses remain unstable [12, 42], and models that achieve high accuracy on curated benchmarks degrade unpredictably under distributional shifts characteristic of operational environments [25]. When such systems output predictions without calibrated uncertainty, the result is an illusion of accuracy: the model reports high confidence on every classification, whether processing a clear image or a degraded sensor feed. Failures present identically to successes, and the error becomes visible only when its consequences materialize.

These two bodies of work are typically treated in isolation. We argue that their intersection is where the deepest risk lies: opacity prevents operators from detecting when a system crosses its competence boundary, and adversarial fragility means such crossings can be induced deliberately. No prior work examines this interaction as a unified epistemic condition specific to military decision-making.

## 2.2 Trust, Institutional Incentives, and Systemic Vulnerabilities

The illusions of understanding and accuracy interact with institutional dynamics to produce a third: the illusion of determinism. Clean dashboards and categorical outputs erase the probabilistic nature of the underlying computation. A commander who has observed correct identifications across hundreds of training exercises generalizes to the expectation that the system will perform identically in theater, where unfamiliar terrain, degraded sensors, and adversarial countermeasures invalidate that expectation. The categorical format provides no warning that the system has crossed the boundary of its competence. Operators are known to defer to automated recommendations, a disposition documented as automation bias [14, 23, 57]. In these systems the problem starts earlier, before any deference: a categorical output reports a verdict without the confidence signal that would let an operator judge whether to trust it or not. Interface design mediates whether system limitations become perceptible [54]. Lee and See [35] show that overtrust, calibrated trust, and distrust produce distinct failure signatures, while Risko and Gilbert [61] characterize the mechanism as selective cognitive ofloading.

The benchmarking ecosystem reinforces this dynamic. When evaluation criteria are known in advance, developers optimize for those criteria at the expense of broader robustness [2, 63], and benchmark performance becomes a proxy for institutional confidence in systems that may fail under conditions never tested. Weight poisoning [33, 36] represents a further threat: an adversary can induce targeted failures that standard evaluations do not detect, and the poisoned model continues to perform well on benchmarks while behaving anomalously in high-stakes scenarios.

The growing dependence on private technology firms creates a parallel vulnerability. Project Maven [16, 73] exposed this concretely: Google engineers who raised ethical concerns lacked formal channels to influence deployment, while DoD oficials who authorized it lacked access to the underlying code or training data. The state owned the system, but the knowledge resided elsewhere. As Raji et al. [58] document, real-world AI failures in high-stakes domains typically stem from breakdowns in engineering practice. Military contexts, characterized by secrecy and operational pressure, amplify these gaps. Trust calibration and systemic vulnerability literatures both miss the institutional machinery that compounds these problems: benchmark gaming, contracting structures that separate knowledge from authority, and fragmented expertise that prevents any single actor from recognizing failure conditions.

## 2.3 The Accountability Gap

AI-mediated decision-making dissolves the classical military chain of command into Nissenbaum’s [53] “many hands problem.” The model is designed by engineers who may never witness its deployment, trained on datasets curated by scientists who will never select a target, operated by soldiers who cannot read its internal logic, and authorized by policymakers who lack both technical and operational knowledge. Each actor holds a fragment of agency; none fully causes the system’s decisions. When failure occurs, responsibility is not discovered but negotiated [53], and scapegoating becomes a structurally likely outcome. Matthias [44] introduced the “responsibility gap” for learning automata; Sparrow [69] extended it to lethal autonomous weapons; and machines lack the intentionality that grounds moral accountability [5, 20]. If they cannot bear responsibility and human beings are too fragmented to deal with it on their own, existing structures provide no mechanism for attribution.

The concept of “meaningful human control” [68] is intended to fill this gap, but it lacks operational precision. For oversight to be functionally efective, two conditions must hold: temporal authority (the power to veto, pause, or require stage-gated approval) and interpretive access (the ability to understand why a system produced a given output, not just whether to endorse it). The international NGO Article 36 [4] has interpreted the obligations of states deploying autonomous weapons along these lines:

• States must explicitly afirm that meaningful human control is required over individual attacks.

• Weapon systems operating without such control must be prohibited.

• States must publicly explain how they apply control over existing systems and justify why they consider them acceptable and lawful.

In practice, however, the system’s output arrives with an aura of authority, and what begins as collaboration slides into deference. On Fischer and Ravizza’s account of guidance control [19], the human remains responsible so long as the decision reflects their own critical judgment. But if the operator consistently aligns with the AI’s output without challenge, their role becomes performative: they are not exercising judgment but afixing a stamp of approval. This is not real control. It is responsibility laundering. Automation bias reinforces the dynamic [23, 41], and military operators are rarely trained to recognize overconfidence, distributional shift, or adversarial manipulation [71]. The challenge is not to preserve human involvement as such, but to ensure that it is cognitively robust: the capacity to understand, interrogate, and override.

## 2.4 IHL Under Strain

International Humanitarian Law rests on principles that presume human judgment, contextual awareness, and moral intentionality [6], but the specific points of failure difer. Sharkey [66] and Rof [62] show that distinction and proportionality require contextual interpretation that probabilistic classifiers cannot replicate: the diference between a combatant and a civilian holding a similar object is not a feature-space boundary but a moral assessment. Heyns [26] adds that even if targeting were technically accurate, the physical and moral distance introduced by autonomy undermines the attribution on which legal accountability depends. Liu [38] extends the analysis to situations current systems handle worst: recognizing surrender, medical evacuation, or hors de combat status, where the legally required response is restraint, not classification.

Boulanin et al. [7] translate these requirements into three testable conditions for lawful deployment: foresight, administration, and traceability. Crootof [13] reframes the problem. If AI-mediated outcomes cannot be fully predicted, accountability grounded in intent becomes unworkable. Her “war torts” model shifts the focus from the decisionmaker’s intent to the question of whether the decision-making process adhered to verifiable standards of care. This move from intent to procedural integrity is the conceptual foundation of our framework: it transforms an intractable philosophical question (who holds moral responsibility for an algorithmic output?) into a verifiable institutional one (what process was followed, and did it meet documented standards?). Yet the gap between this insight and operational practice remains wide. Jobin et al. [31] confirm the pattern across 84 AI ethics guidelines: high-level principles converge while implementation diverges. Before proposing a framework, we ask what the instruments already in force actually require of military AI, and of whom.

## 2.5 Existing Governance Instruments and Their Limits

Military AI falls under three regimes. They were written separately, and none of them was written for it. This section makes one claim about them: the rules that bind govern the wrong thing. They govern a weapon, a platform, a system, and they treat the learned model as one more piece of software inside it. The argument has three steps. Two of them are about how far the rules reach. The third is about what the rules take as their object. That third one is the point of this section, because it holds even where a binding rule does apply.

The civil regime is the most developed. The NIST AI Risk Management Framework [46] organizes risk work around four functions: govern, map, measure, and manage. The EU AI Act [17] goes further and turns a scale of risk into duties that can be enforced against high-risk systems. The military regime is narrower. U.S. DoD Directive 3000.09 [74] sets out how autonomous and semi-autonomous weapon systems are reviewed and approved. The NATO AI Strategy [55] commits members to three principles: systems should be governable, traceable, and reliable. The humanitarian regime only states a position. The ICRC [30] asks states to agree on rules for human control over the use of force.

The first limit is about who is covered. The EU AI Act does not apply to systems used only for military, defense, or national security purposes (Art. 2(3)) [17]. The exclusion is smaller than it looks. Recital 24 says that a dual-use system, or a military system reused for civilian work, comes back under the Act [17]. Still, the efect is that the one instrument with real enforceable duties does not reach purely military systems, and covers the dual-use middle ground (predictive logistics, ISR, decision support) only in patches. The second limit is about what is covered. DoDD 3000.09 is the one binding rule written for the military, and it applies only to weapon systems [74]. Logistics, medical triage, intelligence analysis, and cyber operations sit outside it, even when what they produce leads to someone being killed. NIST, NATO, and the ICRC cover the whole range, but as advice or as a stated principle, not as a requirement [30, 46, 55].

Both of those limits are about coverage, and both could be closed by widening a perimeter. The third one cannot, because it is about what the rules govern rather than how far they reach. To see it, take the case that favors the current regime most: a system that sits squarely inside the strongest binding rule.

Consider an air-defense turret covered by DoDD 3000.09. Its object-detection model reads radar and camera returns and decides which aircraft to fire on. The directive covers the turret, not the model. So the model gets reviewed only as one part of the weapon, and every requirement that follows is attached to the turret. Three consequences follow. The rest of this subsection takes them one at a time.

The first is that the rules track where the model sits rather than what it does. Move the same model into a tool that flags vehicles for a human analyst and it becomes ordinary intelligence software, bound by nothing. It still makes the same mistakes at the same rate. Only the cost of a mistake has changed.

The second is that the checks that do exist ask the wrong question. Checks do happen when the model is updated. New software counts as a new baseline, and that triggers the same re-qualification any other component change would. But the check runs as configuration management, and it tests the model against the specification the platform was approved against. That kind of check assumes two things are enough to tell you how a system will behave in the field: a test campaign of finite size, and a look at how the thing was built. For control logic that follows fixed rules, the assumption holds. It breaks for a model that can get an input wrong even when that input looks no diferent from the ones it gets right, and whose behavior comes from training data and weights rather than rules an engineer wrote down. The check runs, the system passes, and what you learn is not what the check was built to tell you.

The third is that some failures arrive with no check at all. As the inputs the model sees in the field drift away from the ones it was trained on, the model gets worse while the approved baseline stays exactly the same. Nothing changes on paper, so there is no event for the rules to hang on.

Together these leave a gap in accountability. When the turret fires on the wrong aircraft, the mount, the fire-control logic, the interlocks, and the sensors can each be found within specifications. The configuration can be found compliant. And the error that decided the outcome sits in a classification that no requirement in the regime was written to constrain.

The pattern is not limited to weapons. Existing rules govern a thing that is built, fielded, and approved as a unit, and treat the model as one more piece of software inside it. But a model carries a decision whose stakes are set by where it is installed, and a behavior that can change while the platform stays the same. Governing the platform leaves that decision free. Better rules have to begin by saying what they should require, and of what. The next section builds that list.

## 3 Design Principles for Accountable Military AI

The previous section ended on a missing list: what should good rules require of a military AI? Before writing it, we have to ask where such rules can hold. One answer is to put them inside the model itself.

At the current state of the art, we do not know how to reliably encode ethics into learning systems in a way that remains robust across diverse contexts, adversarial manipulation, and distribution shifts. Arkin [3] proposed the most developed attempt at an “ethical governor”: a computational architecture that would enforce IHL constraints (distinction, proportionality, prohibition on perfidy) at the system level. The approach demonstrates that certain hard constraints can be implemented as decision-theoretic filters, but it also reveals the limits of formalization. Moral judgment involves interpretation, competing values, and the capacity to recognize when rules should be broken. No loss function captures the principle of proportionality. No training dataset encodes the full meaning of distinction. Attempts to formalize ethics into algorithmic constraints produce brittle systems that satisfy the letter of a rule while violating its spirit, or that fail unpredictably outside their training distribution. Rather than operating under the illusion that morality can be optimized as a mathematical loss function, we propose to proceduralize ethics, encoding ethical constraints into institutions, certification regimes, standard operating procedures, and technical safeguards. A meta-analysis of 84 AI ethics guidelines by Jobin et al. [31] reveals broad convergence on transparency, fairness, non-maleficence but persistent divergence on implementation. This gap between principle and practice is precisely what proceduralization is designed to close.

This pragmatic stance is not a permanent renunciation of value-aligned design. As alignment research matures, a hybrid approach will become viable, combining limited algorithmic constraints (calibrated uncertainty, forced abstention, hard safety interlocks) with rigorous external procedural controls (mandatory audits, authorization gates, traceability). We develop this approach through six components: actionable accountability structures (3.1), adversarial evaluation (3.2), a NATO standardization proposal (3.3), interface design principles for preserving human judgment (3.4), tiered deployment thresholds (3.5) and testing it against eight operational cases (3.7).

## 3.1 Actionable Accountability

If responsibility is dispersed across many hands, governance must make it actionable. The objective is to ensure that (i) every high-stakes decision has an identifiable human authorizer, (ii) every deployed system has an identifiable human owner, and (iii) every critical failure can be reconstructed and attributed through evidence.

We propose three complementary requirements.

Named accountability roles. For each operational AI capability, states should designate a System Owner (responsible for lifecycle risk management), an Operational Authorizer (responsible for each mission-level activation), an Operational User (the operator who interacts with the system in real time, responsible for exercising judgment on its outputs, reporting anomalies, and invoking override or abort procedures), and an Independent Evaluator (responsible for certification and periodic reassessment). These roles must be separated to reduce conflicts of interest, and their responsibilities written into doctrine rather than treated as informal practice.

Traceability by default. Accountability requires investigability. Systems should maintain tamper-evident logs capturing model version and weights hash, data pipeline versioning, sensor provenance (device serial number and certificate identifier), a hash of sensor inputs, uncertainty estimates, operator interactions (overrides, approvals, aborts), and timing information. The design goal is a factual record suficient for post hoc review and legal scrutiny.

Technical enforceability of oversight. Whenever a system is used in an operation that can produce harm, the architecture should enforce stage-gated approvals, abort capability, and fallback modes. If substantive human control is a requirement, it must be implemented as a control surface that the operator can exercise under time pressure.

These measures do not eliminate the many-hands problem, but they transform distributed decision-making into distributed obligation, reducing the conditions for scapegoating by making responsibilities explicit, auditable, and enforceable.

## 3.2 Adversarial Evaluation

Accountability structures establish who must answer for a decision, but not whether the underlying system is robust enough to warrant deployment. Evaluation and certification form the second pillar of proceduralized ethics.

The standard approach relies on benchmarks with known ground-truth labels. When evaluation criteria are known in advance, developers optimize for those criteria at the expense of broader competence. Goodhart [24] first identified this dynamic in monetary policy; Strathern [70] generalized it: “when a measure becomes a target, it ceases to be a good measure.” The pattern is well-documented in ML. Dozens of optimizers claim to outperform AdamW on standard benchmarks, yet almost none see adoption, because comparisons rely on asymmetric hyperparameter tuning, favorable evaluation conditions, and unreported negative results [75]. As Jordan argues in his analysis of the Muon optimizer [32], the only credible evidence of superiority is success under truly competitive conditions. Recht et al. [59] provide a striking illustration at the dataset level: after replicating the ImageNet test set creation process from scratch, they observed accuracy drops of 11-14% across a wide range of classifiers, with no change in the underlying data distribution. Performance on a fixed benchmark, even a well-curated one, does not reliably predict generalization.

For military AI, the same dynamic applies with higher stakes. If defense contractors are evaluated on disclosed benchmarks, optimization will target those benchmarks, producing systems that perform well in controlled demonstrations but may fail in operational environments. A natural countermeasure is adversarial auditing with undisclosed evaluation criteria. Benchmarks used to certify military AI should be kept secret, accessible only to authorized evaluators, and rotated regularly. The existence of certain evaluation sets should itself be classified. The evaluation process should remain transparent and standardized; what must stay hidden is the evaluation content. Transparency about process combined with secrecy about content channels the incentive structure toward competence rather than benchmark-specific optimization.

Evaluation should also include adversarial robustness testing. As discussed in 2.1, adversarial patches [9] and targeted perturbations [18] can cause high-confidence misclassification on inputs that appear benign to human observers. Findings [12, 42] suggest that robustness under adversarial conditions is a more informative criterion than accuracy on clean data. Hendrycks and Dietterich [25] demonstrated this concretely with ImageNet-C, showing that top-performing classifiers degrade severely under common corruptions (noise, blur, weather, digital artifacts) despite high clean-data accuracy. For military systems operating in contested and unpredictable environments, evaluation must reflect these conditions.

A complementary dimension is negative testing. Standard evaluation emphasizes positive cases (does the model correctly identify targets it should identify?), but the converse matters equally: does the model correctly reject inputs it should reject? A classifier trained on military vehicles may learn to associate camouflage patterns with the positive class, triggering false positives on civilian trucks while missing military vehicles in unexpected color. For military AI, negative testing directly operationalizes the principle of distinction. Evaluation should include civilian objects sharing features with military targets, military targets with atypical appearances, scenarios involving surrender or medical evacuation, and inputs at the boundary of the training distribution.

## 3.3 Toward a NATO Evaluation Standard

NATO already possesses the normative infrastructure to standardize how sensor data is captured, formatted, and exchanged. Its 2021 AI Strategy [55] commits member states to responsible AI development based on principles of governability, traceability, and reliability, but stops short of specifying certification mechanisms. What NATO lacks is an equivalent framework for certifying the algorithms that consume this data. Table 1 summarizes the current landscape.

Table 1 NATO standardization landscape for AI-relevant capabilities. Sensor-layer standards are mature; algorithmic and autonomy layers remain under development. Study-stage STANAGs have not been ratified; objectives are drawn from the NATO Standardization Ofice DCRA Report [49].
<table><tr><td>Layer</td><td>STANAG Purpose</td><td></td><td>Status</td></tr><tr><td>Sensor</td><td>4607</td><td>Ground Moving Target Indicator (GMTI) data Promulgated transmission [48]</td><td></td></tr><tr><td>Sensor</td><td>4609</td><td>NATO Digital Motion Imagery (Full Motion Promulgated Video (FMV) with Video Moving Target Indica- tor (VMTI) metadata)</td><td></td></tr><tr><td>Sensor</td><td>4676</td><td>Intelligence, Surveillance and Reconnaissance Promulgated (ISR) tracking data and trajectory exchange</td><td></td></tr><tr><td>Sensor</td><td>4579</td><td>Battlefield Target Identification (electronic IFF) Promulgated</td><td></td></tr><tr><td>Sensor</td><td>5527</td><td>Friendly Force Tracking interoperability</td><td>Promulgated</td></tr><tr><td>Algorithmic 5653</td><td></td><td>Core Data Framework (common metadata Study stage schemas) [50]</td><td></td></tr><tr><td>Algorithmic 5670</td><td></td><td>Federated Data Catalogue (decentralized meta- Study stage data discovery) [52]</td><td></td></tr><tr><td>Algorithmic 5669</td><td></td><td>Neural network and deep learning model Study stage exchange [51]</td><td></td></tr><tr><td>Autonomy 4817</td><td></td><td>Multi-domain unmanned platform command and Study stage control</td><td></td></tr></table>

STANAG 5669 is the most ambitious of the study-stage eforts: it targets the exchange of trained neural network models between nations independently of trainingtime software and hardware [51]. In principle, this would enable inference sharing, cross-domain fine-tuning, and multilateral model development. In practice, however, the exchange of trained military AI models faces a fundamental obstacle. Military AI systems are developed on top of national doctrine, encoding tactical assumptions, operational priorities, and decision heuristics specific to each nation’s armed forces. Sharing a trained model amounts to exposing that doctrine. An adversary with access to the model can reverse-engineer the decision logic it embodies and develop targeted countermeasures, for instance by training a system specifically designed to exploit the doctrinal patterns encoded in the weights. This concern, raised in discussions with Lieutenant-Colonel J´erˆome Ranc at the Human Factors Air Operations Laboratory (Centre d’Expertise A´erienne Militaire / Air Warfare Center), suggests that unrestricted model exchange between allies will remain impractical for the foreseeable future.

What is both feasible and needed is a standardized evaluation protocol. We propose that NATO member states develop such a protocol, formalized as a STANAG. The protocol should be structured in two parts. The first, intended for states and evaluation authorities, specifies:

• Rules governing benchmark secrecy, including classification levels, access controls, and rotation schedules.

• Evaluation methodology: which metrics are measured, how thresholds are determined, and pass/fail criteria.

• Prerequisites for deployment authorization, including documentation, audit trails, and fallback mechanisms.

• Governance structures for independent auditing bodies.

A critical design question concerns the transparency of evaluation criteria. If developers know exactly which metrics are used, they optimize for those metrics (Goodhart’s Law again). If metrics are entirely opaque, developers cannot adequately prepare. A middle path is available: the categories of metrics (robustness, calibration, latency) can be made public while the specific implementations, thresholds, and weighting schemes remain classified. This does not eliminate the Goodhart risk, but it channels optimization toward relevant properties rather than the idiosyncrasies of a known test suite.

The second part, intended for developers, specifies:

• The category of the model and its intended operational use.

• The training dataset, or a statistical characterization suficient to compute distributional properties.

• Compliance with a unified annotation standard per modality and task.

Full dataset disclosure would enable auditors to detect distributional biases and spurious correlations, but training data often constitutes proprietary assets. A workable compromise requires developers to provide statistical summaries (class distributions, domain coverage, annotation quality metrics) suficient for efective audit, complemented by periodic sample-level inspections under confidentiality agreements. Summary statistics have limited power to reveal certain bias classes; sample-level audits provide the necessary complement.

Interoperability demands annotation standardization. The proliferation of labeling formats (Pascal VOC, COCO, custom schemas) creates friction and enables formatdependent inconsistencies. A STANAG for military AI should mandate a canonical annotation format per modality and task, with open-source conversion tools for legacy formats. Table 2 illustrates the breadth of modalities involved and their varying degrees of standardization maturity.

Beyond the modality-specific metadata catalogued in Table 2, a military annotation standard requires cross-cutting provisions: data provenance and chain of custody, uncertainty encoding (annotator confidence, inter-annotator agreement, ambiguity flags), operational context (scenario type, rules-of-engagement applicability), and versioning (schema identifiers, revision histories). The standard should include a governance mechanism for schema evolution so that the format remains current without sacrificing backward compatibility.

## 3.4 The Assistant Principle: Preserving Human Judgment

If AI systems are to support rather than supplant human decision-making, the interface between operator and algorithm becomes a critical design surface. Norman’s concept of afordance [54] captures the core requirement: the system’s perceptible properties should make its capabilities, limitations, and confidence immediately visible, so that the operator’s mental model tracks the system’s actual state.

Table 2 Modalities, representative tasks, structural annotations, and modality-specific metadata relevant to military AI evaluation. The standardization column reflects the availability of public benchmarks and shared annotation schemas. Cross-cutting metadata requirements (provenance, uncertainty, operational context, versioning) apply to all modalities and are discussed in the text.
<table><tr><td rowspan=1 colspan=19>Modality          Representative    Structural Anno- Modality-           Std.Tasks                tations               SpecificMetadata</td></tr><tr><td rowspan=5 colspan=19>Imagery (EO/IR) Detection, classifica- Bounding  boxes, Sensor type, Ground • • •tion, segmentation, pixel masks, class Sample Distance,tracking              labels, object IDs, altitude, weather,occlusion flags       illuminationVideo / FMV      Activity recognition, Temporal segments, Frame rate, com- • • otarget    tracking, action labels, trajec- pression,cameraevent detection       tory annotations     motion,stabiliza-tionRadar / SAR      Automatic target Target chips, detec- Polarization,fre-• oorecognition, change tion masks, class quency       band,detection,     ship labels</td></tr><tr><td rowspan=1 colspan=2>ev</td><td rowspan=1 colspan=4>detection</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>Target chips, detec-</td></tr><tr><td rowspan=1 colspan=1>tion masks, class</td></tr><tr><td rowspan=1 colspan=1>labels</td><td rowspan=1 colspan=6>incidenceangle,</td></tr><tr><td rowspan=1 colspan=12>detection</td><td rowspan=1 colspan=1></td><td rowspan=2 colspan=6>range resolutionCoordinate Refer- • • </td></tr><tr><td rowspan=1 colspan=6>Geospatial</td><td rowspan=1 colspan=6>Change detection, G</td><td rowspan=1 colspan=1>eo-referenced</td></tr><tr><td rowspan=1 colspan=6></td><td rowspan=1 colspan=6>terrain classifica</td><td rowspan=1 colspan=1>- polygons, temporal</td><td rowspan=1 colspan=6>ence System (CRS),</td></tr><tr><td rowspan=1 colspan=6></td><td rowspan=1 colspan=6>tion, infrastructure</td><td rowspan=1 colspan=1>stamps, attribution</td><td rowspan=2 colspan=6>spatial resolution,temporal coverage,</td></tr><tr><td rowspan=3 colspan=6>3D</td><td rowspan=1 colspan=6>mapping</td><td rowspan=1 colspan=1>labels</td></tr><tr><td rowspan=1 colspan=6></td><td rowspan=1 colspan=1></td><td rowspan=2 colspan=6>collection geometryorigin• • </td></tr><tr><td rowspan=1 colspan=6>Object detection,</td><td rowspan=1 colspan=1>3D bounding boxes, Sensor</td><td rowspan=1 colspan=5>Sensororigin</td></tr><tr><td rowspan=1 colspan=6></td><td rowspan=1 colspan=6>scene reconstruction</td><td rowspan=1 colspan=1>, point-wiseclass</td><td rowspan=1 colspan=5>(LiDAR,    stereo,</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=6></td><td rowspan=1 colspan=6>ground segmenta- la</td><td rowspan=1 colspan=1>bels, mesh annota- S</td><td rowspan=1 colspan=5>tructure     from</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=6></td><td rowspan=1 colspan=6>tion</td><td rowspan=1 colspan=1>tions</td><td rowspan=1 colspan=5>Motion     (SfM)),</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=6></td><td rowspan=1 colspan=6></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=5>point density, regis-</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=6></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=5>tration accuracy</td><td rowspan=2 colspan=1>y, oo</td></tr><tr><td rowspan=1 colspan=6>Signals (SIGINT)</td><td rowspan=1 colspan=6>Emitter      classi-</td><td rowspan=1 colspan=1>Time-frequency</td><td rowspan=1 colspan=5>Center frequenc</td></tr><tr><td rowspan=1 colspan=6></td><td rowspan=1 colspan=6>fication,      signal</td><td rowspan=1 colspan=1>annotations, emitter</td><td rowspan=1 colspan=5>bandwidth, Signal-</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=6></td><td rowspan=1 colspan=6>detection, protocol I</td><td rowspan=1 colspan=1>Ds, protocol labelst</td><td rowspan=1 colspan=5>o-Noise     Ratio</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=6></td><td rowspan=1 colspan=6>identification</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=5>(SNR), collection</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=4></td><td rowspan=1 colspan=6></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=5>geometry</td></tr><tr><td rowspan=1 colspan=6></td><td rowspan=10 colspan=13>Entity extraction, Span annotations, Source type, lan- • • •relation extraction, entity types, rela- guage, reliability rat-ing, collection dateSpeaker ID, speech Transcripts, speaker Sample rate, SNR, • • orecognition, acoustic labels,temporal channelrecording environ-Cross-modal fusion, Cross-modal align- Temporal synchro- • o ojoint entity resolu- ment, joint entity nization, spatial co-per- registration, modal-modality annotation ity weightinglayers</td></tr><tr><td rowspan=1 colspan=6></td><td rowspan=1 colspan=6>relation extraction,</td><td rowspan=1 colspan=1>entity types, rela-</td><td rowspan=1 colspan=3>guage, reliab</td></tr><tr><td rowspan=1 colspan=6></td><td rowspan=1 colspan=6>event detection</td><td rowspan=1 colspan=1>tion labels</td><td rowspan=1 colspan=2>ing, collect</td><td rowspan=1 colspan=3>on date</td></tr><tr><td rowspan=1 colspan=6></td><td rowspan=1 colspan=6>Speaker ID, speech</td><td rowspan=1 colspan=1>Transcripts, speaker</td></tr><tr><td rowspan=4 colspan=12></td><td rowspan=1 colspan=1>labels,temporal</td><td rowspan=1 colspan=5>channelcount,</td></tr><tr><td rowspan=1 colspan=1>boundaries</td><td rowspan=1 colspan=3>recording</td><td rowspan=1 colspan=2>envi</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=6></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>ment</td></tr><tr><td rowspan=1 colspan=4>i-modal</td><td rowspan=1 colspan=6>Cross-modal fusion,</td><td rowspan=1 colspan=1>Cross-modal align-</td><td rowspan=1 colspan=1>Tempora</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=9>joint entity resolu-</td><td rowspan=2 colspan=1></td></tr><tr><td></td><td></td><td rowspan=1 colspan=10></td></tr></table>

Standardization maturity (Std.): ••• = established benchmarks and public annotation schemas; • • ◦ = partial standardization or limited military-specific coverage; • ◦ ◦ = predominantly ad hoc or classified formats.

The central risk is miscalibrated trust. The interface should make uncertainty perceptually salient so that trust is continuously recalibrated by the display itself. Risko and Gilbert [61] frame this as selective cognitive ofloading: the system absorbs computational burden (sensor fusion, pattern detection) while freeing attentional resources for judgment, without introducing competing demands.

Where operational constraints permit, inherently transparent model classes (generalized additive models [39], decision trees) should be preferred for high-stakes decisions. For deep models, post hoc explanations (saliency maps, SHAP values) provide diagnostic signals but do not guarantee faithful access to internal reasoning [37]. In our framework, post hoc explanations are treated as audit artifacts: they support traceability and review but do not by themselves confer legitimacy on a decision.

These principles suggest a design philosophy we call the hybrid trajectory: rather than embedding ethics directly in the model, the goal is to combine procedural safeguards with internal technical guardrails and to expand machine autonomy incrementally as reliability is demonstrated under operational conditions. The trajectory rests on three mechanisms:

1. Calibrated uncertainty and abstention. The system estimates and communicates its confidence. Below a task-specific threshold, it abstains and defers to the operator.

2. Hard interlocks. Actions outside the authorized operational envelope (ROE<sup>1</sup> violations, geographic exclusion zones) are made mechanically impossible, independently of software-level controls.

3. Progressive autonomy. The boundary between autonomous operation and human deferral is set conservatively at initial deployment and widened only when traceability records (3.1) demonstrate sustained reliability under operational conditions. Crucially, this expansion of autonomy always remains bounded by the hard interlocks defined in Point 2.

The hybrid trajectory is not a fixed architecture but a temporal process: each expansion of machine autonomy is contingent on demonstrated reliability, and procedural safeguards define the permissible boundary at every stage. This logic directly motivates the tiered deployment framework developed next.

## 3.5 Tiered Deployment Thresholds

Not all military AI systems warrant the same oversight. A tiered classification organized by increasing operational risk and decreasing reversibility determines the minimum procedural safeguards at each stage of development, certification, and fielding. This logic is consistent with the risk-based architecture of the EU AI Act [17], the OECD Recommendation on Artificial Intelligence [56], and the NIST AI Risk Management Framework [46], the diferentiated treatment of autonomous weapon systems in U.S. DoD Directive 3000.09 [74], and the ICRC position that the acceptability of autonomy should be assessed relative to the nature and context of the task [30]. Where the EU AI

Table 3 Formalized tier classification of military AI capabilities. Each tier is characterized by a profile across five measurable dimensions. Escalation on any single dimension may trigger reclassification.
<table><tr><td>Tier (capability)</td><td></td><td>Lethality Reversibility Autonomy</td><td></td><td>Speed of effect</td><td>Scope</td></tr><tr><td>1: Predictive logistics</td><td>None</td><td>Full</td><td>Advisory</td><td>Human-time</td><td>Individual</td></tr><tr><td>2: Medical / maintenance</td><td>None</td><td>Partial</td><td>Advisory</td><td>Human-time</td><td>Individual</td></tr><tr><td>3: ISR / surveillance</td><td>Indirect</td><td>Partial</td><td>Semi- autonomous</td><td>Accelerated</td><td>Unit</td></tr><tr><td>4: Target nomination</td><td>Indirect</td><td>Partial</td><td>Semi- autonomous</td><td>Accelerated</td><td>Strategic</td></tr><tr><td>5: Cyber operations</td><td>Indirect</td><td>Irreversible</td><td>Autonomous</td><td>Machine- speed</td><td>Strategic</td></tr><tr><td>6: Lethal engagement</td><td>Direct</td><td>Irreversible</td><td>Autonomous</td><td>Machine- speed</td><td>Strategic</td></tr></table>

Lethality: None = no causal path to harm; Indirect = outputs feed decisions that may cause harm; Direct = system can apply force.  
Reversibility: Full = decision can be recalled without residual efect; Partial = correction possible but downstream consequences may persist; Irreversible = efects cannot be undone.  
Autonomy: Advisory = human decides; Semi-autonomous = system acts, human approves or vetoes; Autonomous = system acts without per-action human approval.  
Speed: Human-time = hours to days; Accelerated = seconds to minutes; Machine-speed = milliseconds.  
Scope: Individual = single system or process; Unit = operational-unit-level consequences; Strategic = cross-domain or national-level consequences.

Act classifies by application domain, our tiering classifies by operational consequence and reversibility, reflecting the distinct risk structure of military operations.

We formalize this classification along five measurable dimensions (Table 3):

• Lethality potential: whether the system’s outputs can cause death directly, indirectly through downstream decisions, or not at all.

• Reversibility: whether a decision can be recalled or corrected after execution.

• Autonomy level: the degree of human involvement in the decision loop.

• Speed of efect: the temporal window available for human intervention.

• Scope of impact: the organizational and strategic breadth of consequences.

Tier assignment is determined by the combination of these dimensions; escalation along any single dimension triggers reassessment under the re-certification procedure described below.

At minimum, each tier should specify:

• Pre-deployment assurance: depth of independent testing (including adversarial and negative testing) and re-certification frequency.

• Authorization gates: who approves activation, at what command level, and whether stage-gated approvals are required.

• Fallback and containment: required fail-safe modes (safe stop, degraded operation, manual override) and technically enforced function blocks.

• Monitoring and traceability: minimum logging requirements, incident-reporting triggers, and conditions for automatic suspension.

Lower tiers (1-2: logistics, medical support) require standard software-engineering assurance and domain-expert validation. At intermediate tiers (3-4: reconnaissance, command support), errors can directly afect situational awareness and may become irreversible once acted upon. These tiers call for independent verification and validation including structured red-teaming, command-level deployment authorization, exposure of confidence estimates to operators in accordance with 3.4, and technically guaranteed fallback to manual operation.

Tier 5 covers autonomous cyber operations, whose efects propagate at machine speed [27], cross jurisdictional boundaries, and trigger cascading consequences dificult to anticipate. National-authority-level approval should be required. Technical containment must include hard scope limits (target lists, network boundaries, time windows) enforced at the code level. Continuous monitoring with automatic suspension triggers is essential given the latency between misfire and detection in cyberspace.

Tier 6 concerns lethal autonomous weapon systems and raises a more fundamental question: whether a given capability should be deployed at all. The ICRC has called for new internationally agreed rules ensuring human control over the use of force, and has argued that autonomous weapons incapable of IHL compliance should be expressly prohibited [30]. Within the present framework, Tier 6 does not function as authorization to deploy under suficiently robust safeguards; it marks a threshold of political and legal deliberation.

Even where all safeguards are satisfied, certain configurations may remain impermissible. A system that cannot reliably distinguish combatants from civilians, or that is intended for environments where such distinction is structurally unachievable, should not be deployed regardless of procedural compliance. Procedural governance defines necessary conditions; it does not establish suficiency.

A system’s tier is not fixed at deployment. Operational experience, adversarial adaptation, and incremental upgrades can shift the risk profile. A reconnaissance AI at Tier 3 that acquires target-nomination features drifts toward Tier 4 or beyond; if this migration is not formally recognized, safeguards remain calibrated to a lower risk level. The framework must incorporate periodic re-assessment with mandatory re-certification whenever functional scope or operational context changes materially [46, 74].

## 3.6 From Dimension Profiles to Tiers: A Criticality Score

Table 3 names six reference profiles, one per tier. A fielded system rarely matches one of them exactly: with three levels on each of five dimensions there are $3 ^ { 5 } = 2 4 3$ possible profiles, and the table names six. Tier assignment therefore needs a rule for the other 237.

![](images/8ebfea8a7a28c9540ec67a48f2708878cc0d13f346a7882f0a4e461d1477ea48.jpg)  
Fig. 1 Why a weighted sum of the five ranks cannot reproduce Table 3 with evenly spaced tiers. The Tier 3 profile is the arithmetic midpoint of the Tier 1 and Tier 6 profiles, so any weighted sum places it halfway between them, between Tiers 3 and 4.

We write ${ \boldsymbol x } = ( x _ { 1 } , \dots , x _ { 5 } )$ for a profile, where each $x _ { d } \in \{ 0 , 1 , 2 \}$ ranks the level of one dimension from least to most severe (for lethality, None = 0, Indirect = 1, Direct = 2, and likewise for the others in the order of Table 3). We write $A _ { 1 } , \ldots , A _ { 6 }$ for the six reference profiles.

Why not a weighted sum. The natural first idea is to weight each dimension and add: $\textstyle S ( x ) = \sum _ { d } w _ { d } x _ { d }$ , then cut the score into six intervals. Two elementary facts rule this out for Table 3.

Proposition 1 (a weighted sum has no interactions). Under S, raising one dimension by one level changes the score by $w _ { d }$ , whatever the other four dimensions are. The cost of autonomy is the same for a system acting in milliseconds as for one acting over days; the cost of scope is the same whether or not the system is lethal. This follows immediately from the form of S: the four terms that do not change cancel. In Table 3 the step from Tier 5 to Tier 6 changes lethality alone, so w<sub>lethality</sub> is pinned to that single gap, in every context.

Proposition 2 (a weighted sum cannot space the tiers evenly). The Tier 3 profile $A _ { 3 } = ( 1 , 1 , 1 , 1 , 1 )$ is the exact midpoint of the Tier 1 profile $A _ { 1 } = ( 0 , 0 , 0 , 0 , 0 )$ and the Tier 6 profile ${ A _ { 6 } } = ( 2 , 2 , 2 , 2 , 2 )$ . Any weighted sum therefore scores it exactly halfway between Tiers 1 and 6, that is, at “tier $3 . { \bar { 5 } } ^ { \ ' }$ on an evenly spaced scale (Fig. 1), whereas the table places it at Tier 3. Formally, $S ( A _ { 3 } ) = { \textstyle { \frac { 1 } { 2 } } } \big ( S ( A _ { 1 } ) + S ( A _ { 6 } ) \big )$ for every choice of weights. Consequently no weights make the six tiers equally spaced. Worse, the gap from Tier 2 to 3 equals the gap from Tier 3 to 4 plus the gap from Tier 5 to 6 plus $w _ { \mathrm { a u t o n o m y } } + w _ { \mathrm { s p e e d } }$ , so the largest gap between consecutive tiers is at least twice the smallest, with equality only when autonomy and speed carry zero weight.

Both facts concern the same restriction: a weighted sum of ranks treats the two steps of a dimension as equally costly and the five dimensions as independent. Neither is credible here. Going from Indirect to Direct lethality is not the same step as going from None to Indirect; and speed of efect matters because it shortens the window in which a human can intervene, which is only relevant if the system acts on its own.

The score we adopt. We read a tier as a level of expected damage: how likely a wrong action is to take efect, times how bad it is, times how long it lasts. The five dimensions fall into three groups accordingly (Fig. 2):

![](images/c585eac81a3405359d52ea302f6455bf8f7583c09cdc13d662b00491d74e99c3.jpg)  
Fig. 2 Structure of the criticality score. The five dimensions of Table 3 enter through three factors that multiply; the tier is read on a logarithmic scale, so that moving up one tier always means the same multiplicative increase in expected damage. Dimensions within a factor may reinforce each other; dimensions in diferent factors do not.

$$
\arctan ( x ) = \underbrace { \Pi ( \mathrm { a u t o n o m y } , \mathrm { s p e e d } ) } _ { \mathrm { p a s s e s ~ w i t h o u t ~ v e t o } } \times \underbrace { G ( \mathrm { l e t h a l i t y } , \mathrm { s c o p e } ) } _ { \mathrm { g r a v i t y ~ o f ~ t h e ~ a c t i o n } } \times \underbrace { R ( \mathrm { r e v e r s i b i l i t y } ) } _ { \mathrm { p e r m a n e n c e ~ o f ~ t h e ~ h a r m } } .\tag{1}
$$

Π captures the veto window: a system that only advises leaves the decision, and hence the veto, to a human, whatever its speed; a system that acts on its own can still be stopped if its efects unfold over hours, but not if they unfold in milliseconds. G measures the gravity of the action itself, in which scope matters more when the action is lethal. R records whether the harm can be undone once it has occurred; reversibility is a property of the damage, not of the probability that a wrong action is taken.

This form has two consequences. First, within each factor the two dimensions can reinforce each other, which Proposition 1 said a weighted sum cannot do. Second, across factors the score multiplies, so that on the tier scale the three contributions add: the cost of a lethality step is the same whatever the autonomy level.

What the factors are, and how a product becomes a sum. Π, G and R are small tables of declared values (Table 4). Moving up one tier always means multiplying expected damage by the same constant factor, call it $\beta . \mathrm { A }$ level that multiplies expected damage by $\beta$ is therefore worth one tier, a level that multiplies it by $\beta ^ { 2 }$ is worth two, and in general a factor $F$ is worth $\log _ { \beta } F$ tiers. We record every table entry in these units, so that $g = \log _ { \beta } G , \pi = \log _ { \beta }$ Π and $r = \log _ { \beta } R .$ Counted in tiers, factors that multiply become numbers that add, as decibels turn ratios of power into sums. Taking logarithms of Eq. (1) gives

$$
{ \begin{array} { r l } & { \operatorname { s c o r e } ( x ) \ = \ \log _ { \beta } \operatorname { c r i t } ( x ) } \\ & { \qquad = \ g [ \mathrm { l e t h a l i t y , s c o p e } ] + \pi [ \mathrm { a u t o n o m y , s p e e d } ] + r [ \mathrm { r e v e r s i b i l i t y } ] , } \end{array} }\tag{2}
$$

and the tier is one plus the nearest integer to the score. The numerical value of $\beta$ never needs to be fixed: only tier units enter the tables and the rule. The six reference profiles of Table 3 score 0.00, 0.65, 2.32, 2.68, 3.99 and 5.34, and therefore fall in Tiers 1 to 6 as required. Raising any single dimension never lowers the score, so escalation on one dimension can only maintain or raise the tier, as the re-certification rule above assumes.

Table 4 The three factors of the criticality score, expressed in tiers: an entry of 1.0 means that this level multiplies expected damage by the factor β separating two consecutive tiers, an entry of 2.0 by β<sup>2</sup>, and so on. The tier of a profile is one plus the nearest integer to the sum of its three entries. Profiles combining Direct lethality with Full reversibility are treated as inadmissible: a system that can apply force cannot have fully recallable efects.
<table><tr><td colspan="4">g: gravity of the action</td></tr><tr><td>Lethality \Scope</td><td>Individual</td><td>Unit</td><td>Strategic</td></tr><tr><td>None</td><td>0.00</td><td>0.10</td><td>0.39</td></tr><tr><td>Indirect</td><td>0.75</td><td>1.03</td><td>1.39</td></tr><tr><td>Direct</td><td>2.10</td><td>2.38</td><td>2.74</td></tr><tr><td colspan="4">π: the action passes without veto</td></tr><tr><td>Autonomy Speed</td><td>Human-time</td><td>Accelerated</td><td>Machine-speed</td></tr><tr><td>Advisory</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Semi-autonomous</td><td>0.30</td><td>0.64</td><td>0.96</td></tr><tr><td>Autonomous</td><td>0.60</td><td>0.95</td><td>1.30</td></tr><tr><td colspan="4">r: permanence of the harm</td></tr><tr><td>Reversibility</td><td>Full</td><td>Partial</td><td>Irreversible</td></tr><tr><td></td><td>0.00</td><td>0.65</td><td>1.30</td></tr></table>

How the values were set. The entries of Table 4 were fitted to no data: they satisfy six constraints, each of which can be accepted or rejected on its own by an expert panel, and within what those constraints leave open they were chosen so that no admissible profile sits close to a tier boundary (the closest is a tenth of a tier away):

1. Veto window. The Advisory row of π is zero: if a human decides, the speed of efect is immaterial. Below that row, speed counts for more as autonomy grows, and conversely.

2. Gravity. In g, scope counts for more as lethality grows, and lethality for more as scope grows.

3. Admissibility. A system with Direct lethality cannot have Full reversibility; the 27 such profiles are excluded, leaving 216.

4. Fidelity to Table 3. Each reference profile falls inside its own tier band, at a safe distance from the edges, without being forced to the centre of the band. Forcing the centre would assert that the joint rise of four dimensions (Tier 2 to 3) is worth the rise of scope alone (Tier 3 to 4), a precision the table does not carry.

5. Lethality as a threshold. Moving from None to Direct lethality adds at least two tiers, at every scope.

6. No silent dimension. Every step on every dimension costs at least a tenth of a tier, the two steps of a dimension are within a factor of three of each other, and an autonomous system acting at machine speed sits at least 1.2 tiers above an advisory one.

A panel should debate these six statements, not the sixteen free entries of Table 4. On one point Table 3 leaves them little room. It grants a single tier to the joint rise of reversibility, autonomy and speed to their maximum (Tier 4 to 5), but also a single tier to the rise of scope alone (Tier 3 to 4). Any score faithful to the table therefore weighs autonomy lightly relative to scope, and constraint 6 keeps irreversibility from disappearing in the trade-of. Accepting this, or revising the Tier 4 and Tier 5 reference profiles, is the panel’s decision.

## 3.7 Operational Precedents: Counterfactual Analysis

The governance architecture proposed above is intended to be more than a normative exercise. To assess whether its mechanisms would make a practical diference, we apply the framework to eight documented cases spanning Tiers 3-6 and the period 1988- 2025 (Table 5). For each governance component, we identify cases where its absence contributed to documented harm and examine what the framework would have required. Not all cases can be fully verified from public information; the analysis should be read as systematic counterfactual assessment. Counterfactual reasoning demonstrates that the framework’s mechanisms map onto documented failure classes, but it cannot establish that their presence would have prevented harm with certainty. Prospective evaluation through integration into certification procedures remains the necessary next step. These limitations acknowledged, the analysis serves a specific purpose: to show that each governance component addresses a failure mode that has occurred in practice, and that their combination defines a governance posture substantially more demanding than any currently in force.

The evidential basis varies across cases. Some are documented through oficial investigations [15, 16, 21]. Others rest on UN Panel of Experts reports [72], defense analyses [67], or independent technical forensics [34]. The Lavender and Gospel cases rely primarily on investigative journalism [1, 28, 29] and have not been subject to oficial inquiry. We draw on them as the best available evidence while acknowledging that future disclosures may revise the factual record. Table 5 summarizes each case. Table 6 maps which framework dimensions were satisfied, absent, or partially met.

## 3.7.1 Tiered deployment thresholds.

The tiering framework requires that capabilities be classified by their risk profile and that migration across tiers trigger re-certification. Project Maven illustrates the consequences of unrecognized tier drift. Originally an ISR analysis tool (Tier 3), Maven evolved under Palantir into the Maven Smart System, which includes an AI Asset Tasking Recommender that proposes bomber and munition assignments to targets [43]. By 2025, the contract ceiling exceeded \$1.3 billion and the system had been adopted by NATO Allied Command Operations [47]. This migration from Tier 3 to Tier 4 occurred without public evidence of formal re-assessment. The Kargu-2 represents the inverse failure: a Tier 6 capability (autonomous lethal engagement) deployed without any of the governance requirements that tier demands [72]. No political-level authorization, no IHL legal review, no geographic or target-class constraints enforced at the system level. Iron Dome provides the positive counterexample. Although it possesses Tier 6 capability (autonomous launch of interceptor missiles), the system operates within tightly constrained parameters: the target set is incoming projectiles, the decision criterion is ballistic impact-point prediction, and defended zones are politically authorized [67]. Iron Dome demonstrates that Tier 6 governance requirements are compatible with efective autonomous operation when the operational envelope is well-defined.

Table 5 Summary of eight operational cases used for counterfactual framework analysis, ordered chronologically. Tier assignments follow the classification in Table 3.
<table><tr><td>Case</td><td>Year</td><td>Tier System</td><td>type</td><td>Primary deficiency</td><td>Documented consequence</td></tr><tr><td>USS Vin- 1988 cennes</td><td></td><td>6</td><td>Aegis- assisted engagement</td><td>Interface design, trust miscalibra- tion</td><td>290 civilians killed (Iran Air 655)</td></tr><tr><td>Patriot friendly fire</td><td>2003</td><td>6</td><td>Autonomous air defense</td><td>Automation bias, IFF failure</td><td>3 allied aircrew killed</td></tr><tr><td>Stuxnet</td><td>2007-10</td><td>5</td><td>Autonomous cyber weapon</td><td>No scope limits, no recall</td><td>Spread to ~115,000 systems worldwide</td></tr><tr><td>Iron Dome</td><td>2011-</td><td>6</td><td>Defensive interception</td><td></td><td>(Positive case) &gt;90% interception rate</td></tr><tr><td>Project Maven</td><td>2017-</td><td>3→4</td><td>ISR, target nomination</td><td>Contractor- state knowledge asymmetry</td><td>Oversight deficiencies (DoD IG)</td></tr><tr><td>Kargu-2 2020</td><td></td><td>6</td><td>Loitering munition</td><td>safeguards</td><td>Absence of all Autonomous engagement without oversight</td></tr><tr><td>Gospel</td><td>2021-</td><td>4</td><td>Structural targeting</td><td>Volume vs. review quality overwhelmed</td><td>12,000+ targets, review capacity</td></tr><tr><td>Lavender 2023</td><td></td><td>5</td><td>Individual targeting</td><td>Performative endorsement</td><td>~3,700 est. misidentifications</td></tr></table>

## 3.7.2 Actionable accountability.

The framework requires named accountability roles (System Owner, Operational Authorizer, Independent Evaluator) with separated responsibilities. Project Maven’s first phase exposed what happens when these roles are absent. Google engineers who raised ethical concerns lacked formal channels to influence deployment decisions; DoD oficials who authorized deployment lacked direct access to the underlying code or training data [16, 73]. Knowledge resided in the contractor, authority in the state, and no named role bridged the gap. This asymmetry illustrates why governance must be institutionalized above the level of individual contracts: when oversight depends on informal arrangements between parties with misaligned incentives, accountability becomes structurally impossible.

## 3.7.3 Traceability by default.

Stuxnet, a sophisticated computer worm, operated without traceability of any kind. Once deployed, it propagated autonomously across networks, executed sabotage against Iranian centrifuges, and concealed its efects by replaying recorded sensor data to plant operators [34]. No kill switch or recall mechanism was documented. Attribution required years of forensic analysis; accountability was efectively impossible in real time. The framework’s requirement for tamper-evident logs with sensor provenance would have made scope creep detectable. In the Kargu-2 case, no traceability of autonomous engagement decisions was available to UN investigators [72]. For Lavender, the reported absence of feature-level logging meant that the basis for individual targeting scores could not be reconstructed or challenged [1]. In each case, the inability to reconstruct the decision chain rendered post hoc accountability procedurally impossible.

## 3.7.4 Adversarial evaluation.

The Patriot system’s 2003 friendly-fire incidents illustrate the cost of inadequate operational testing. A Defense Science Board investigation found that the system was given “too much autonomy” and that operators “trusted the system in a naive manner” [15]. The IFF subsystem misclassified friendly aircraft as hostile threats; negative testing (does the system correctly reject friendly aircraft under realistic confusion scenarios?) would have identified this failure mode before deployment. The USS Vincennes case prefigures the same concern at the interface level. The Aegis system’s radar correctly tracked Iran Air Flight 655 as climbing in a civilian corridor, but the crew correlated a ground-based military IFF signal with the airborne contact [21]. The system was never tested for this specific class of confusion. Adversarial evaluation under the proposed framework would require testing against precisely such scenarios: ambiguous IFF environments, overlapping military and civilian signatures, and highstress time-critical conditions.

## 3.7.5 The assistant principle.

Lavender exemplifies the failure of every requirement the assistant principle imposes. The system presented a name and a score without exposing the features driving the classification or any confidence interval. Human analysts reviewed each recommendation for approximately twenty seconds, a review that often consisted solely of verifying that the flagged individual was male [1, 29]. This is performative endorsement. Calibrated uncertainty and forced abstention below confidence thresholds would have flagged the system’s reported ten-percent error rate, corresponding to approximately 3,700 misidentifications among 37,000 flagged individuals. The Gospel system, which generated over 12,000 structural targets during operations in Gaza [28, 45], illustrates the same dynamic at the level of infrastructure: when machine-generated target volume overwhelms human review capacity, oversight becomes nominal. The Vincennes tragedy shows that interface design failures predate modern AI. The Aegis system displayed raw data without highlighting the anomaly between the aircraft’s climbing trajectory and its putative hostile classification [21]. Interface design that foregrounds uncertainty and anomalies, as the assistant principle requires, would have given the crew the perceptual salience needed to override.

Table 6 Cross-reference matrix: eight operational cases assessed against the six governance dimensions of the proposed framework. Every case involving documented harm exhibits deficiencies in at least two dimensions.
<table><tr><td>Case</td><td>Tiering</td><td>Account.</td><td>Trace.</td><td>Adv. eval.</td><td>Assist.</td><td>Enforce.</td></tr><tr><td>USS Vincennes</td><td>●</td><td>o</td><td>★</td><td>o</td><td>o</td><td>●</td></tr><tr><td>Patriot</td><td>●</td><td>★</td><td>★</td><td>O</td><td>★</td><td>★</td></tr><tr><td>Stuxnet</td><td>O</td><td>o</td><td>O</td><td>★</td><td>1</td><td>o</td></tr><tr><td>Iron Dome</td><td>●</td><td>●</td><td>●</td><td>●</td><td>●</td><td>●</td></tr><tr><td>Project Maven</td><td>O</td><td>o</td><td>★</td><td>★</td><td>★</td><td>★</td></tr><tr><td>Kargu-2</td><td>O</td><td>o</td><td>o</td><td>o</td><td>1</td><td>o</td></tr><tr><td>Gospel</td><td>★</td><td>★</td><td>★</td><td>★</td><td>0</td><td>★</td></tr><tr><td>Lavender</td><td>0</td><td>★</td><td>O</td><td>★</td><td>0</td><td>★</td></tr></table>

• = dimension satisfied or not applicable; ⋆ = partially satisfied or ambiguous; ◦ = absent and absence contributed to documented harm; - = not applicable to this case.

## 3.7.6 Technical enforceability.

Stuxnet’s propagation beyond its intended target to approximately 115,000 systems in multiple countries, including allied nations [34], is a concrete demonstration of what happens when hard scope limits are absent. The framework’s Tier 5 requirements (target lists, network boundaries, time windows enforced at the code level) are designed to contain precisely this class of failure. The Kargu-2 operated without geographic, temporal, or target-class constraints enforced at the system level [72]. Iron Dome again provides the positive case: its engagement criteria (ballistic trajectory prediction within defended zones) function as hard interlocks that bound the system’s autonomous operation independently of software-level controls [67].

Table 6 reveals a consistent pattern: every case involving documented harm exhibits deficiencies in at least two governance dimensions. No single safeguard would have been suficient in isolation. The presence or absence of a human operator is orthogonal to governance quality; Lavender placed a human in the loop, while Iron Dome removed one, yet the latter satisfies the framework’s requirements comprehensively. These are counterfactual analyses; the framework’s predictive power can only be tested through prospective application in certification procedures. That limitation is acknowledged. What the analysis does establish is that each governance mechanism independently addresses a documented class of failure, and that their combination defines a governance posture substantially more demanding than any currently in force.

## 4 Conclusion

The governance of military AI is usually debated as a question of permission: what machines may be allowed to decide, and where the line of prohibition should fall. This paper has argued that a prior question determines whether any such line can be enforced. Systems whose reasoning cannot be inspected, whose evaluation can be gamed, and whose deployment disperses knowledge away from authority erode the conditions under which human judgment remains judgment at all. The presence of an operator guarantees nothing about the quality of oversight; what matters is whether that operator retains the temporal authority to intervene and the interpretive access to know when intervention is warranted.

Taking this diagnosis seriously requires abandoning a tempting ambition. If ethical constraints cannot be reliably encoded in a loss function, they must be encoded in institutions: named roles that make responsibility attributable, evaluation regimes that resist optimization, deployment thresholds calibrated to irreversibility, and interfaces that keep uncertainty visible. This is the practical form of Crootof’s move from intent to procedural integrity. It does not answer the philosophical question of who bears moral responsibility for an algorithmic output, and it is not meant to. It substitutes a question that institutions can actually adjudicate: what process was followed, by whom, and did it meet a documented standard? The counterfactual analysis in Sect. 3.7 indicates that this substitution is not merely conceptual, since the failures it maps are procedural failures rather than failures of intention.

The proposal has clear limits. The tiered classification is schematic, and operationalizing it would demand sustained negotiation among states with divergent legal traditions and procurement cultures; the history of defense standardization suggests such convergence is incremental at best. The hybrid trajectory presumes continued progress in interpretability and uncertainty quantification at a pace no one can guarantee. Most fundamentally, the framework addresses state military organizations operating within treaty regimes, and ofers little purchase on non-state actors who recognize no such obligations.

These limits mark the work that follows. Structured expert consultation, for instance through a Delphi process involving military legal advisers, operational commanders, and engineers, would test whether the tiering and the accountability roles survive contact with institutional practice. Section 2.5 points out where current rules stop. Rather than inventing new principles, this framework simply focuses on how to apply existing ones to the AI models and military contexts that are currently left out. And the interface principles of Sect. 3.4 invite direct empirical study: whether forced abstention and salient uncertainty measurably improve decision quality under time pressure is a behavioral question, and one the framework currently answers only by assertion.

The deliberations of the UN Convention on Certain Conventional Weapons, now approaching a possible Review Conference, will test whether states are prepared to convert declaratory principles into binding operational constraints. Whatever form those constraints take, the assessment of whether a particular system, in a particular context, satisfies the demands of international humanitarian law will remain a human judgment. The task is to ensure that it is exercised with adequate knowledge and real authority, rather than performed after the fact by someone with neither.

## Declarations

## Funding

No funding was received for conducting this study.

## Competing interests

Authors A, B and D are employed by SAS Impact, which uses machine learning systems for defense applications. Author C declares no competing interests. The authors received no specific funding for this work.

## Data availability

No datasets were generated or analyzed during the current study. All cases discussed are documented in publicly available sources cited in the reference list.

## References

[1] Abraham Y (2024) “Lavender”: The AI machine directing Israel’s bombing spree in Gaza. +972 Magazine URL https://www.972mag.com/ lavender-ai-israeli-army-gaza/

[2] Amodei D, Olah C, Steinhardt J, et al (2016) Concrete problems in ai safety. ArXiv abs/1606.06565

[3] Arkin RC (2009) Governing Lethal Behavior in Autonomous Robots. CRC Press, Boca Raton, FL

[4] Article 36 (2015) Killing by machine: Key issues for understanding meaningful human control. https://www.stopkillerrobots.org/wp-content/uploads/2021/09/ KILLING BY MACHINE 6.4.15.pdf

[5] Asaro P (2012) On banning autonomous weapon systems: Human rights, automation, and the dehumanization of lethal decision-making. International Review of the Red Cross 94(886):687–709. https://doi.org/https://doi.org/10.1017/ S1816383112000768

[6] Bhuta N, Beck S, Geiss R, et al (eds) (2016) Autonomous Weapons Systems: Law, Ethics, Policy. Cambridge University Press, Cambridge, https://doi.org/https: //doi.org/10.1017/CBO9781316597873

[7] Boulanin V, Bruun L, Goussac N (2020) Autonomous weapon systems and international humanitarian law: Identifying limits and the required type and degree of human–machine interaction. Sipri report, Stockholm International Peace Research Institute (SIPRI), Stockholm, Sweden, URL https://www.sipri.org/publications, published by SIPRI.

[8] Bove C, Laugel T, Lesot MJ, et al (2024) Why do explanations fail? a typology and discussion on failures in xai

[9] Brown TB, Man´e D, Roy A, et al (2018) Adversarial patch. URL https://arxiv. org/abs/1712.09665, arXiv:1712.09665

[10] Bu¸cinca Z, Lin P, Gajos KZ, et al (2020) Proxy tasks and subjective measures can be misleading in evaluating explainable ai systems. In: Proceedings of the 25th International Conference on Intelligent User Interfaces. Association for Computing Machinery, New York, NY, USA, IUI ’20, p 454–464, https://doi.org/https://doi. org/10.1145/3377325.3377498

[11] Burrell J (2016) How the machine ‘thinks’: Understanding opacity in machine learning algorithms. Big Data & Society 3(1):2053951715622512. https://doi.org/ https://doi.org/10.1177/2053951715622512

[12] Carlini N, Wagner D (2017) Towards evaluating the robustness of neural networks. In: 2017 IEEE Symposium on Security and Privacy (SP). IEEE, pp 39–57, https: //doi.org/https://doi.org/10.1109/SP.2017.49

[13] Crootof R (2022) War torts. New York University Law Review 97(4)

[14] Cummings ML (2004) Automation bias in intelligent time critical decision support systems. In: AIAA 1st Intelligent Systems Technical Conference, pp 557–562, https://doi.org/https://doi.org/10.2514/6.2004-6313

[15] Defense Science Board (2005) Report of the defense science board task force on patriot system performance. Tech. rep., Ofice of the Under Secretary of Defense for Acquisition, Technology, and Logistics, report No. ADA435837

[16] Department of Defense Inspector General (2022) Evaluation of Contract Monitoring and Management for Project Maven. Tech. rep., Department of Defense, URL https://www.dodig.mil/reports.html/Article/2893388/ evaluation-of-contract-monitoring-and-management-for-project-maven-dodig-2022-0/

[17] European Parliament and Council of the European Union (2024) Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (Artificial

Intelligence Act). Oficial Journal of the European Union, L 2024/1689, URL https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng

[18] Eykholt K, Evtimov I, Fernandes E, et al (2018) Robust physical-world attacks on deep learning visual classification. In: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 1625–1634, https://doi.org/https://doi.org/ 10.1109/CVPR.2018.00175

[19] Fischer JM, Ravizza M (1998) Responsibility and Control: A Theory of Moral Responsibility. Cambridge University Press, New York

[20] Floridi L, Sanders JW (2004) On the morality of artificial agents. Minds Mach (Dordr) 14(3):349–379

[21] Fogarty WM (1988) Formal investigation into the circumstances surrounding the downing of Iran Air flight 655 on 3 July 1988. Tech. rep., United States Department of Defense, declassified 1993

[22] Future of Life Institute (2015) Autonomous weapons: An open letter from AI & robotics researchers. Open letter presented at IJCAI 2015, Buenos Aires, URL https://futureoflife.org/open-letter-autonomous-weapons/

[23] Goddard K, Roudsari A, Wyatt JC (2012) Automation bias: a systematic review of frequency, efect mediators, and mitigators. J Am Med Inform Assoc 19(1):121–127

[24] Goodhart CAE (1975) Problems of monetary management: The U.K. experience. In: Papers in Monetary Economics. Reserve Bank of Australia

[25] Hendrycks D, Dietterich T (2019) Benchmarking neural network robustness to common corruptions and perturbations. In: Proceedings of the International Conference on Learning Representations (ICLR)

[26] Heyns C (2013) Report of the special rapporteur on extrajudicial, summary or arbitrary executions. Tech. Rep. A/HRC/23/47, United Nations Human Rights Council, Geneva, URL https://digitallibrary.un.org/record/755741/files/A HRC 23 47-EN.pdf?ln=fr, submitted pursuant to Human Rights Council resolution 17/5, focuses on lethal autonomous robotics and the protection of life. 22 p.

[27] Horowitz MC (2019) When speed kills: Lethal autonomous weapon systems, deterrence and stability. Journal of Strategic Studies 42(6):764–788. https://doi. org/https://doi.org/10.1080/01402390.2019.1621174

[28] Human Rights Watch (2024) Gaza: Israeli military’s digital tools risk civilian harm. URL https://www.hrw.org/news/2024/09/10/ gaza-israeli-militarys-digital-tools-risk-civilian-harm

[29] Human Rights Watch (2024) Questions and answers: Israeli military’s use of digital tools in Gaza. Human Rights Watch, URL https://www.hrw.org/news/ 2024/09/10/questions-and-answers-israeli-militarys-use-digital-tools-gaza

[30] International Committee of the Red Cross (2021) Icrc position on autonomous weapon systems. Tech. rep., International Committee of the Red Cross, Geneva, URL https://www.icrc.org/en/document/ icrc-position-autonomous-weapon-systems

[31] Jobin A, Ienca M, Vayena E (2019) The global landscape of AI ethics guidelines. Nature Machine Intelligence 1(9):389–399. https://doi.org/https://doi.org/10. 1038/s42256-019-0088-2

[32] Jordan K, Jin Y, Boza V, et al (2024) Muon: An optimizer for hidden layers in neural networks. URL https://kellerjordan.github.io/posts/muon/

[33] Kurita K, Michel P, Neubig G (2020) Weight poisoning attacks on pre-trained models. URL https://arxiv.org/abs/2004.06660, arXiv:2004.06660

[34] Langner R (2011) Stuxnet: Dissecting a cyberwarfare weapon. IEEE Security & Privacy 9(3):49–51. https://doi.org/https://doi.org/10.1109/MSP.2011.67

[35] Lee JD, See KA (2004) Trust in automation: Designing for appropriate reliance. Human factors 46(1):50–80

[36] Li L, Song D, Li X, et al (2021) Backdoor attacks on pre-trained models by layerwise weight poisoning. URL https://arxiv.org/abs/2108.13888, arXiv:2108.13888

[37] Lipton ZC (2018) The mythos of model interpretability. Commun ACM 61(10):36– 43

[38] Liu HY (2012) Categorization and legality of autonomous and remote weapons systems. Int Rev Red Cross 94(886):627–652

[39] Lou Y, Caruana R, Gehrke J (2012) Intelligible models for classification and regression. In: Proceedings of the 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. Association for Computing Machinery, New York, NY, USA, KDD ’12, p 150–158, https://doi.org/https://doi.org/10. 1145/2339530.2339556

[40] Lundberg SM, Lee SI (2017) A unified approach to interpreting model predictions. In: Proceedings of the 31st International Conference on Neural Information Processing Systems. Curran Associates Inc., Red Hook, NY, USA, NIPS’17, p 4768–4777

[41] Lyell D, Coiera E (2017) Automation bias and verification complexity: a systematic review. J Am Med Inform Assoc 24(2):423–431

[42] Madry A, Makelov A, Schmidt L, et al (2018) Towards deep learning models resistant to adversarial attacks. In: Proceedings of the International Conference on Learning Representations (ICLR)

[43] Marson J (2024) Palantir lands \$480M Army contract for Maven artificial intelligence tech. URL https://defensescoop.com/2024/05/29/ palantir-480-million-army-contract-maven-smart-system-artificial-intelligence/

[44] Matthias A (2004) The responsibility gap: Ascribing responsibility for the actions of learning automata. Ethics and Information Technology 6(3):175–183. https: //doi.org/https://doi.org/10.1007/s10676-004-3422-1

[45] Meier MW (2024) The gospel, lavender, and the law of armed conflict. Lieber Institute, West Point URL https://lieber.westpoint.edu/ gospel-lavender-law-armed-conflict/

[46] National Institute of Standards and Technology (2023) Artificial intelligence risk management framework (AI RMF 1.0). Tech. Rep. NIST AI 100-1, U.S. Department of Commerce, Gaithersburg, MD, https://doi.org/https://doi.org/10. 6028/NIST.AI.100-1

[47] NATO Allied Command Operations (2025) NATO acquires AIenabled warfighting system. URL https://shape.nato.int/news-releases/ nato-acquires-aienabled-warfighting-system-

[48] NATO Standardization Ofice (2010) STANAG 4607 – NATO ground moving target indicator format. Tech. rep., NATO Standardization Ofice, URL https: //nso.nato.int/nso/nsdd/main/list-promulg, promulgated

[49] NATO Standardization Ofice (n.d.) DCRA report – defence capability requirements archive. NATO HQ C3 Staf, URL https://nhqc3s.hq.nato.int/apps/ DCRA Report/

[50] NATO Standardization Ofice (n.d.) STANAG 5653 NATO core data framework (NCDF). URL https://nhqc3s.hq.nato.int/ apps/DCRA Report/id-29d4122b072148f5aaf4882ecc5d963c/elements/ id-f1582f46a24a4ea3a1966939a5486438.html, study stage

[51] NATO Standardization Ofice (n.d.) STANAG 5669 – ADatP-5669: Standardization task for exchanging trained machine learning models. URL https://nhqc3s.hq. nato.int/apps/DCRA Report/id-29d4122b072148f5aaf4882ecc5d963c/elements/ id-365814783a6c42a0b24f588acfdc371c.html, study stage

[52] NATO Standardization Ofice (n.d.) STANAG 5670 federated data catalogue. URL https://nhqc3s.hq.nato.int/apps/ DCRA Report/id-29d4122b072148f5aaf4882ecc5d963c/elements/ id-59fcf2e5ab1d41c499ad7d364b536631.html, study stage

[53] Nissenbaum H (1996) Accountability in a computerized society. Science and Engineering Ethics 2(1):25–42. https://doi.org/https://doi.org/10.1007/bf02639315

[54] Norman DA (1999) Afordance, conventions, and design. interactions 6(3):38–43

[55] North Atlantic Treaty Organization (2021) Summary of the NATO artificial intelligence strategy. NATO Oficial Texts, URL https://www.nato.int/cps/en natohq/oficial texts 187617.htm

[56] OECD (2019) Recommendation of the council on artificial intelligence. URL https://legalinstruments.oecd.org/en/instruments/OECD-LEGAL-0449, oECD/LEGAL/0449

[57] Parasuraman R, Riley V (1997) Humans and automation: Use, misuse, disuse, abuse. Human factors 39(2):230–253

[58] Raji ID, Dobbe R (2023) Concrete problems in AI safety, revisited. ArXiv arXiv:2401.10899 [cs.CY]

[59] Recht B, Roelofs R, Schmidt L, et al (2019) Do ImageNet classifiers generalize to ImageNet? In: Proceedings of the 36th International Conference on Machine Learning (ICML), vol 97. PMLR, pp 5389–5400

[60] Ribeiro MT, Singh S, Guestrin C (2016) ”why should i trust you?”: Explaining the predictions of any classifier. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. Association for Computing Machinery, New York, NY, USA, KDD ’16, p 1135–1144, https: //doi.org/https://doi.org/10.1145/2939672.2939778

[61] Risko EF, Gilbert SJ (2016) Cognitive ofloading. Trends in cognitive sciences 20(9):676–688

[62] Rof HM (2014) The strategic robot problem: Lethal autonomous weapons in war. Journal of Military Ethics 13(3):211–227. https://doi.org/https://doi.org/10. 1080/15027570.2014.975010

[63] Russell S (2019) Human Compatible: Artificial Intelligence and the Problem of Control. Viking

[64] Scharre P (2018) Army of None: Autonomous Weapons and the Future of War. W.W. Norton & Company, New York

[65] Selvaraju RR, Cogswell M, Das A, et al (2017) Grad-CAM: Visual explanations from deep networks via gradient-based localization. In: 2017 IEEE International Conference on Computer Vision (ICCV). IEEE

[66] Sharkey N (2008) Grounds for discrimination: Autonomous robot weapons. RUSI Defence Systems pp 86–89

[67] Sheridan J (2024) Iron dome shows AI’s risks and rewards. Center for European Policy Analysis (CEPA) URL https://cepa.org/article/ iron-dome-shows-ais-risks-and-rewards/

[68] Santoni de Sio F, van den Hoven J (2018) Meaningful human control over autonomous systems: A philosophical account. Front Robot AI 5:15

[69] Sparrow R (2007) Killer robots. Journal of Applied Philosophy 24(1):62–77. https://doi.org/https://doi.org/10.1111/j.1468-5930.2007.00346.x

[70] Strathern M (1997) ’improving ratings’: audit in the British University system. European Review 5(3):305–321. https://doi.org/https://doi.org/10.1017/ s1062798700002660

[71] Strauch B (2017) The automation-by-expertise-by-training interaction. Hum Factors 59(2):204–228

[72] United Nations Security Council (2021) Letter dated 8 march 2021 from the panel of experts on Libya established pursuant to resolution 1973 (2011). Tech. Rep. S/2021/229, United Nations Security Council, URL https://digitallibrary.un.org/ record/3905159

[73] U.S. Department of Defense (2017) Establishment of Algorithmic Warfare Cross-Functional Team (’Project Maven’). Tech. rep., Deputy Secretary of Defense, URL https://dodcio.defense.gov/Portals/0/Documents/Project% 20Maven%20DSD%20Memo%2020170425.pdf, Memo published April 26, 2017.

[74] U.S. Department of Defense (2023) Dod directive 3000.09, autonomy in weapon systems. Tech. rep., U.S. Department of Defense

[75] Wen K, Hall D, Ma T, et al (2025) Fantastic pretraining optimizers and where to find them. URL https://arxiv.org/abs/2509.02046, arXiv:2509.02046

[76] Yang W, Wei Y, Wei H, et al (2023) Survey on explainable AI: From approaches, limitations and applications aspects. Hum-Cent Intell Syst 3(3):161–188

[77] van Zyl C, Ye X, Naidoo R (2024) Harnessing explainable artificial intelligence for feature selection in time series energy forecasting: A comparative analysis of Grad-CAM and SHAP. Appl Energy 353(122079):122079