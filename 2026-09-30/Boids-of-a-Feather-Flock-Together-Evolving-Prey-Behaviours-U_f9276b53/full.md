# Boids of a Feather Flock Together — Evolving Prey Behaviours Under Diferent Predator Attack Strategies

Augusta van Haren Radboud University

Hanna Hoogen Radboud University

Luca Pattavina Radboud University

September 30, 2026

All authors contributed equally.

## Abstract

Flocking and schooling are thought to have evolved partly as defences against predation, but how prey should balance social and escape tendencies may depend on the predator’s hunting strategy. We extend the predator–prey boids model of Ojo et al. (2023), itself based on Reynolds’ boids, by combining six prey movement tendencies (alignment, cohesion, separation, dodge, repel and wiggle) into a single weighted acceleration update, and by reformulating wiggle as a sinusoidal manoeuvre. We then use an evolutionary strategy to optimise the six behaviour coeficients for collective prey survival against four predator hunting strategies: attack-centroid, attack-nearest, attack-random and attack-peripheral. Across five independent trials per strategy, coeficients converged within trials and mean fitness remained stable or increased, although trials often settled in diferent local optima. Prey survival was highest under attack-centroid and lowest under attack-nearest, in line with our hypotheses. Against attack-centroid, prey evolved individualistic predator avoidance with high escape coeficients, whereas against the other three strategies they largely kept their flock formation. Across all strategies, evolution favoured a low repel coeficient and relatively high dodge and wiggle coeficients. Our results suggest that optimal anti-predator behaviour depends on the interplay between escape tendencies and the predator’s hunting strategy.

## 1 Introduction

Among many lifeforms, complex collective behaviour can emerge from simple social interactions, as seen in flocks of birds, schools of fish, and insect colonies. In a group, individuals influence one another and adapt their behaviour relying only on local neighbourhood information. However, such behaviour does not take place in isolation from the environment; together they form a complex, dynamical system. For example, individuals of diferent species or groups can interact and form predator and prey relationships. Both predators and prey have to evolve dynamic, adaptive behaviours to survive. Collective behaviours, such as schooling and flocking, may have evolved to decrease predation losses by confusing the predator or collectively organising information within a group of agents with limited information (Kunz et al., 2006). These collective behaviours can be governed by simple local rules. A classic example is the boids model by Reynolds (1987). This model simulates realistic flocking, which can be used to learn about flock behaviour in diferent scenarios through artificial boid simulations. One such scenario is the presence of a predator, where the trade-of between behavioural tendencies becomes crucial for survival (Palmer and Packer, 2021; Lee et al., 2006).

In this work, we evolve optimal prey escape behaviour based on local rules, under diferent predator attack strategies. To evolve the optimal behaviours, we again look at nature – algorithms inspired by evolution are useful for solving optimisation problems without relying on gradients (Beyer and Schwefel, 2002; Slowik and Kwasnicka, 2020). Here, we use an evolutionary strategy (ES) to optimise continuous behaviour coeficients.

## 1.1 Research Question

The aim of this work is to implement an extended version of Ojo et al.’s predator-prey model (2023), which is based on Reynolds’s boids model (1987). We enhance this previous work by combining six prey behaviour tendencies (alignment, cohesion, separation, dodge, repel, wiggle) into one behaviour vector that governs the boids’ acceleration. Further, we apply an ES to find the optimal coeficients for the behaviour tendencies for four diferent predator attack strategies. Moreover, we make small improvements to the behaviour implementations. With this setup, we then aim to answer the following research question:

We hypothesise that specific combinations of prey behaviour coeficients are optimal under diferent predator strategies, leading to distinct flocking and escape dynamics. Further, we expect the prey to achieve the highest survival rate under the “attack-centroid” predator strategy, as the centroid does not necessarily correspond to a specific boid, making the strategy ineficient (Demˇsar and Lebar Bajec, 2014). Moreover, we hypothesise that prey achieve the lowest survival rate under the “attack-nearest” strategy, as this strategy is simple but efective (von Moll et al., 2016).

## 1.2 Related Work

Reynolds’s (1987) boids model provides the fundamental structure for simulating flocking behaviour and agentmovement, as governed by three principles: alignment, cohesion, and separation. We further build on the extension of the Reynolds’ boids model of Ojo et al. (2023), who implemented two classes of boids: predator and prey. Ojo et al. (2023) further provide four predator attack strategies (attack-centroid, attack-nearest, attack-random, attack-peripheral) and three escape behaviours (dodge, repel, wiggle), which govern the prey’s behaviour together with the classic boid principles. While relying heavily on the simulation implementation by Ojo et al. (2023), we make several key improvements. Most crucially, Ojo et al. (2023) analyse the prey’s flocking behaviour together with one of the prey escape tendencies in isolation. Instead, we combine all three behaviours together with the flocking principles into one acceleration update vector – allowing the prey to make use of all escape tendencies simultaneously, and learn the most efective behaviour for a given predator’s hunting strategy. Furthermore, in analogy with Reynolds’s (1987) boids model, we treat this prey’s acceleration update vector as a weighted sum of all of its components – whereas Ojo et al.’s (2023) implementation did not assign distinct weights for every movement tendency. Moreover, we further improve their wiggle behaviour, which we describe in more detail in Section 2.1, and performed sensitivity analyses on key simulation parameters.

Other studies, similar to Ojo et al. (2023) investigated predator and prey strategies, like (Demˇsar and Lebar Bajec, 2014) who explored the efect of diferent predator strategies (attack-centroid, attack-nearest, attack-peripheral) on social and individualistic prey and found flocking to be the optimal anti-predatory behaviour – they further characterised the escape patterns exhibited by the prey, similar to Lee et al. (2006).

The models described so far, relied on pre-specified behaviours and strategies with fixed parameters. We instead aim to learn prey behaviours using an ES (Beyer and Schwefel, 2002; Slowik and Kwasnicka, 2020). Previous studies similarly attempted to learn predator or prey behaviours; for example, Alaliyat et al. (2022) used a genetic algorithm (GA) to evolve the boids coeficients to achieve realistic flocking behaviour. Similarly, Hahn et al. (2019) used reinforcement learning in a predator-prey simulation to learn the prey’s strategy, which resulted in flocking behaviour similar to the boids model to confuse the predator. Furthermore, Kunz et al. (2006) used an evolutionary algorithm to evolve prey behaviour while varying predator parameters. Oboshi et al. (2002) used a GA to evolve prey evasion behaviour for one attack strategy (attack-nearest) in a simulation setup similar to ours. However, all of these studies operated on a more basic level than ours, by focusing on learning general collective behaviours like flocking and swarming – instead of specific escape strategies to avoid a predator.

Few studies tried to evolve more sophisticated behaviours in a predator-prey scenario; von Moll et al. (2016) focused on evolving predator strategies in a scenario with two predators by using a GA. Similarly to our work, they evolved coeficients for a fixed set of atomic behaviours (pursue, converge, diverge, drive, flank) – however, targeting predators rather than prey. Moreover, Chen et al. (2006) use a GA to evolve coeficients in a linear combination of behaviours, consisting of the classic boid principles, as well as obstacle avoidance, following feed, and avoiding a predator. However, they only implement a direction-based escape behaviour (corresponding to “dodge”) and do not present any proper results.

To summarise, our work extends the existing literature by using an evolutionary strategy (ES) to perform continuous optimisation on the coeficients for a combination of prey behaviours to avoid predation unde diferent predator attack strategies.

## 2 Methods

All analyses were performed using Python 3.11.9 on Windows. The code, result files, and sample simulation videos are available under https://github.com/ivychad/NC-Project-Code-Boids. We provide an in-depth explanation for the choice of all fixed parameters in Appendix A.

## 2.1 Boids Model

To study our research question, we adapted the flocking model of Ojo et al. (2024). This implementation is an extension of Reynolds’s boids model (1987) – which dictates that the movement dynamics of each boid are based on three rules: alignment, cohesion, and separation. Ojo et al. (2024) extend this model by defining two types of boids, predator and prey $B \in \{ B ^ { p r e d } , B ^ { p r e y } \}$ , that each have distinct rules which govern their movement behaviour. Furthermore, they add realism to various aspects of the original model. We have refined their approach by combining the six movement tendencies into one acceleration update vector, by enhancing realism of the wiggle tendency, and by evolving the optimal coeficients for the relative contribution of each of those tendencies using an ES.

The implementation equips every boid B with a position $\mathbf { p } ( B ) \in \mathbb { R } ^ { 2 }$ and velocity $\mathbf { v } ( B ) \in \mathbb { R } ^ { 2 }$ . In each iteration of the simulation, the imposed acceleration a $\mathsf { \Omega } _ { \mathsf { l } } ( B ) \in \mathbb { R } ^ { 2 }$ on the boid determines its dynamics of motion:

$$
\begin{array} { l } { { \displaystyle { \bf v } ( B )  { \bf v } ( B ) + { \bf a } ( B ) \cdot d t } } \\ { { \displaystyle { \bf p } ( B )  { \bf p } ( B ) + { \bf v } ( B ) \cdot d t + \frac { 1 } { 2 } { \bf \sigma a } ( B ) \cdot d t ^ { 2 } } } \end{array}
$$

In the traditional boids model, updates in the boid’s acceleration and velocity are efectively unbounded. As this is physically unrealistic, Ojo et al.’s (2024) implementation limits the direction as well as magnitude of acceleration. This implies that if the angle between the boid’s current velocity $\mathbf { v } ( B )$ and the imposed acceleration $\mathbf { a } ( B )$ exceeds the maximum rotation angle $( | \angle ( \hat { \mathbf { a } } ( B ) , \hat { \mathbf { v } } ( B ) ) | > \theta _ { m a x } ^ { p r e d , p r e y } )$ , the updated acceleration vector is limited to $\hat { \mathbf { a } } ( B ) = \mathbf { R } ( \pm \theta _ { m a x } ^ { p r e d , p r e y } ) \hat { \mathbf { v } } ( B )$ in the given direction (R is the rotation matrix), and rescaled to $| | { \bf { a } } ( B ) | | ~ = ~ a ^ { p r e d , p r e y } ~ ( \mathrm { { O j o } ~ e t ~ a l . , ~ 2 0 2 3 ) }$ . Together, this limits the turn speed of the boid. Moreover, the magnitude of velocity of the boid is bounded to $| | \mathbf { v } ( B ) | | = v ^ { p r e d , p r e y }$ . All bounds are specific to the prey or predator dynamics, and manually adjusted to model realistic behaviour, see Appendix $\mathrm { A }$

Lastly, all boids are assigned a (predator- and prey-specific) perception radius $r _ { P } ^ { p r e d , p r e y }$ , separation radius $r _ { S } ^ { p r e d , p r e y }$ , and a perception angle $f o v ^ { p r e d , p r e y }$ . For each movement tendency, these parameters determine which other boids $B _ { i } { \mathrm { ~ a r e ~ } } ^ { \cdot } { \mathrm { s e e n } } ^ { \cdot }$ , and considered a ‘neighbour’ of $B .$ . In the traditional boids model, this only depends on Euclidean distance dist $( B , B _ { i } ) = | | \mathbf { p } ( B ) , \mathbf { p } ( B _ { i } ) | | _ { 2 } < r _ { P . S } ^ { p r e d , p r e y ^ { 2 } } )$ , as if both predators and preys had a field of view of 360°. However, to enhance realism, Ojo et al. $\left( 2 0 2 4 \right)$ also requires neighbours to be within $B ^ { \prime } \mathrm { s }$ visual field $( \operatorname { i n F o v } ( B , B _ { i } ) = \operatorname { t r u e } )$ . This condition entails that $B _ { i }$ is within $B ^ { \prime } \mathrm { s }$ field of view (delimited by $f o v ^ { p r e d , p r e y } )$ In addition, $B _ { i }$ should not be occluded by another boid $( \mathrm { o c c l u d e d } ( B , B _ { i } ) = \mathrm { f a l s e } )$ . In this definition, neighbour $B _ { i }$ occludes $B _ { j }$ if they are separated by less than $2 ^ { \circ }$ in $B ^ { \prime } \mathrm { s }$ visual field, and $B _ { i }$ is closer than $B _ { j }$ . This results in excluding $B _ { j }$ to be a neighbour of B (Ojo et al., 2023).

## 2.1.1 Predator

For a predator boid $B : = B ^ { p r e d }$ with corresponding predator parameters, its acceleration $\mathbf { a } ( B )$ is governed by either one of the following hunting strategies (Ojo et al., 2024, 2023). These strategies are in line with the attack strategies proposed by Demˇsar and Lebar Bajec (2014): $\hat { \mathbf { a } } ( B ) \in \{ \hat { \mathbf { a } } _ { \mathbf { a t t c } } ( B ) , \hat { \mathbf { a } } _ { \mathbf { a t t n } } ( B ) , \hat { \mathbf { a } } _ { \mathbf { a t t r } } ( B ) , \hat { \mathbf { a } } _ { \mathbf { a t t p } } ( B ) \}$ . Its acceleration and velocity are limited as described above. The hunting strategies each characterise a specific movement behaviour:

• Attack-centroid: B targets the average position of all its neighbouring prey boids $B _ { i } ^ { p r e y }$

$$
\begin{array} { r } { \widehat { \mathbf { a } } _ { \mathrm { a t t c } } ( B ) \ = \ \frac { \sum _ { i = 1 } ^ { N _ { B } } ( \mathbf { p } ( B _ { i } ^ { p r e y } ) - \mathbf { p } ( B ) ) } { N _ { B } } } \end{array}
$$

where each of the $N _ { B }$ neighbouring prey boids $B _ { i } ^ { p r e y }$ satisfy

dist $( B , B _ { i } ^ { p r e y } ) < r _ { P } ^ { p r e d ^ { 2 } }$ ∧ inFov(B, B<sup>prey</sup>) = true ∧ occluded $( B , B _ { i } ^ { p r e y } ) = \mathrm { f } \mathrm { \varepsilon }$ lse

• Attack-nearest: B targets the position of the nearest neighbouring prey boid $B _ { t } ^ { p r e y }$

$$
\hat { \mathbf { a } } _ { \mathbf { a t t n } } ( B ) \ = \ \mathbf { p } ( B _ { t } ^ { p r e y } ) - \mathbf { p } ( B )
$$

where $B _ { t } ^ { p r e y } = \arg \operatorname* { m i n } _ { B _ { \ast } ^ { p r e y } } \mathrm { d i s t } ( B , B _ { i } ^ { p r e y } )$ is closest to B out of the set

of $N _ { B }$ neighbouring prey boids of which each $B _ { i } ^ { p r e y }$ satisfies

$$
\mathrm { d i s t } ( B , B _ { i } ^ { p r e y } ) < r _ { P } ^ { p r e d ^ { 2 } } \quad \wedge \quad \mathrm { i n F o v } ( B , B _ { i } ^ { p r e y } ) = \mathrm { t r u e } \quad \wedge \quad \mathrm { o c c l u d e d } ( B , B _ { i } ^ { p r e y } ) = \mathrm { f a l s e } \quad \wedge \quad \mathrm { o c c u l a t e d } ( B , B _ { i } ^ { p r e y } ) = \mathrm { f a l s e } \quad \wedge \quad \mathrm { o c c u l a t e d } ( B , B _ { i } ^ { p r e y } ) = \mathrm { f a l s e } \quad \wedge \quad \mathrm { o c c u l a t e d } ( B , B _ { i } ^ { p r e y } ) = \mathrm { f a l s e } \quad \wedge
$$

• Attack-random: B targets the position of a random neighbouring prey boid $B _ { t } ^ { p r e y }$

$$
\hat { \mathbf { a } } _ { \mathbf { a t t r } } ( B ) \ = \ \mathbf { p } ( B _ { t } ^ { p r e y } ) - \mathbf { p } ( B )
$$

where $B _ { t } ^ { p r e y } \sim \{ \hat { B } _ { 1 } ^ { p r e y } , \dots , B _ { N _ { B } } ^ { p r e y } \}$ is chosen randomly from the set of $N _ { B }$ neighbouring prey boids of which each $B _ { i } ^ { p r e y }$ satisfies

dist $( B , B _ { i } ^ { p r e y } ) < r _ { P } ^ { p r e d ^ { 2 } } ~ \land$ inFov(B, B<sup>prey</sup>) = true ∧ occluded $( B , B _ { i } ^ { p r e y } )$ = false

• Attack-peripheral: B targets the position of the most peripheral prey boid $B _ { t } ^ { p r e y }$

$$
\hat { \mathbf { a } } _ { \mathbf { a t t p } } ( B ) \ = \ \mathbf { p } ( B _ { t } ^ { p r e y } ) - \mathbf { p } ( B )
$$

where $\begin{array} { r } { B _ { t } ^ { p r e y } = \arg \operatorname* { m a x } _ { B _ { i } ^ { p r e y } } \left( \frac { \sum _ { j = 1 } ^ { N _ { B } } \mathbf { p } ( B _ { j } ^ { p r e y } ) } { N _ { R } } - \mathbf { p } ( B _ { i } ^ { p r e y } ) \right) } \end{array}$ is most isolated from the centroid of the $N _ { B }$ neighbouring prey boids of which each $B _ { i } ^ { p r e y }$ satisfies

dist(B, B<sup>prey</sup><sub>i</sub> ) < r<sup>pred</sup><sub>P</sub> <sup>2</sup> ∧ inFov(B, B<sup>prey</sup><sub>i</sub> ) = true ∧ occluded(B, B<sup>prey</sup><sub>i</sub> ) = false

## 2.1.2 Prey

For a prey boid $B : = B ^ { p r e y }$ with corresponding prey parameters, its acceleration $\mathbf { a } ( B )$ constitutes a superposition of multiple movement tendencies. As an extension upon Ojo et al. (2024)’s implementation, we combine all movement tendencies as a weighted sum into an acceleration update vector. Each tendency’s sub-acceleration $\hat { \mathbf { a } } _ { \perp }$ is weighed by a corresponding coeficient $c _ { \square }$ , hereby contributing to the (normalised) direction of acceleration:

$$
\begin{array} { r } { \hat { \mathbf { a } } ( B ) ~ = ~ n o r m \bigl ( c _ { a l i } \cdot \hat { \mathbf { a } } _ { a l i } ( B ) + c _ { c o h } \cdot \hat { \mathbf { a } } _ { c o h } ( B ) + c _ { s e p } \cdot \hat { \mathbf { a } } _ { s e p } ( B ) ~ + } \\ { c _ { d o d } \cdot \hat { \mathbf { a } } _ { d o d } ( B ) + c _ { r e p } \cdot \hat { \mathbf { a } } _ { r e p } ( B ) + c _ { w i g } \cdot \hat { \mathbf { a } } _ { w i g } ( B ) \bigr ) } \end{array}
$$

Its acceleration and velocity are limited as described above. The first three movement tendencies give rise to the boids flocking behaviour (Ojo et al., 2023, 2024; Reynolds, 1987):

• Alignment: B matches the average direction and speed of all its neighbouring boids $B _ { i } ^ { p r e y }$ (consistent direction flocking)

$$
\begin{array} { r l } & { \hat { \mathbf { a } } _ { \mathbf { a l i } } ( B ) \ = \ n o r m \left( \frac { \sum _ { i = 1 } ^ { N _ { B } } \mathbf { v } ( B _ { i } ^ { p r e y } ) } { N _ { B } } - \mathbf { v } ( B ) \right) } \\ & { \mathrm { w h e r e \ e a c h \ o f \ t h e \  { N _ { B } } \ n e i g h b o u r i n g \ p r e y \ b o i d s \ } B _ { i } ^ { p r e y } \ \mathrm { s a t i s f y } } \\ & { \mathrm { d i s t } ( B , B _ { i } ^ { p r e y } ) < r _ { P } ^ { p r e y 2 } \ \wedge \ \operatorname { i n F o v } ( B , B _ { i } ^ { p r e y } ) = \mathrm { t r u e } \ \wedge \ \operatorname { o c c l u d e d } ( B , B _ { i } ^ { p r e y } ) = \mathrm { f a l s e } } \end{array}
$$

• Cohesion: B matches the average position of all its neighbouring prey boids $B _ { i } ^ { p r e y }$ (localised flocking)

$$
\begin{array} { r } { \hat { \mathbf { a } } _ { \mathbf { c o h } } ( B ) ~ = ~ n o r m \left( \frac { \sum _ { i = 1 } ^ { N _ { B } } ( \mathbf { p } ( B _ { i } ^ { p r e y } ) - \mathbf { p } ( B ) ) } { N _ { B } } \right) } \end{array}
$$

where each of the $N _ { B }$ neighbouring prey boids $B _ { i } ^ { p r e y }$ satisfy

$$
\mathrm { d i s t } ( B , B _ { i } ^ { p r e y } ) < r _ { P } ^ { p r e y 2 } \quad \wedge \quad \mathrm { i n F o v } ( B , B _ { i } ^ { p r e y } ) = \mathrm { t r u e } \quad \wedge \quad \mathrm { o c c l u d e d } ( B , B _ { i } ^ { p r e y } ) = \mathrm { f a l s e } \quad \wedge \quad \mathrm { o c c u l a t e d } ( B , B _ { i } ^ { p r e y } ) = \mathrm { f a l s e } \quad \wedge \quad \mathrm { o c c u l a t e d } ( B , B _ { i } ^ { p r e y } ) = \mathrm { f a l s e } \quad \wedge
$$

• Separation: B repels the positions of all its neighbouring prey boids $B _ { i } ^ { p r e y }$ (preventing collisions)

$$
\begin{array} { r l } & { \widehat { \mathbf { a } } _ { \mathbf { s e p } } ( B ) \ = \ n o r m \left( \sum _ { i = 1 } ^ { N _ { B } } ( \mathbf { p } ( B ) - \mathbf { p } ( B _ { i } ^ { p r e y } ) ) \right) } \\ & { \mathrm { w h e r e ~ e a c h ~ o f ~ t h e ~ } N _ { B } \mathrm { ~ n e i g h b o u r i n g ~ p r e y ~ b o i d s ~ } B _ { i } ^ { p r e y } \mathrm { ~ s a t i s f y } } \\ & { \mathrm { d i s t } ( B , B _ { i } ^ { p r e y } ) < r _ { S } ^ { p r e y 2 } \ \wedge \ \operatorname { i n F o v } ( B , B _ { i } ^ { p r e y } ) = \mathrm { t r u e ~ \wedge ~ \ o c c l u d e d } ( B , B _ { i } ^ { p r e y } ) = \mathrm { f a l s e } } \end{array}
$$

The remaining tendencies characterise escape manoeuvres to flee from the predator. We implemented three distinct escape behaviours (Ojo et al., 2023, 2024), as first introduced in the empirically-based Homing Pigeons Escape (HoPE) (Papadopoulou et al., 2022):

• Dodge: B dodges approaching predator boids $B _ { i } ^ { p r e d }$ , by turning perpendicularly in the opposite direction (direction-based escaping)

$$
\begin{array} { r } { \hat { \mathbf { a } } _ { \mathbf { d o d } } ( B ) ~ = ~ n o r m \left( \sum _ { i = 1 } ^ { M _ { B } } \left\{ \begin{array} { l l } { \mathbf { R } ( + 9 0 ^ { \circ } ) \hat { \mathbf { v } } ( B ) } & { \mathrm { i f ~ } \angle \left( \hat { \mathbf { v } } ( B ) , \hat { \mathbf { v } } ( B _ { i } ^ { p r e d } ) \right) \leq 0 ^ { \circ } } \\ { \mathbf { R } ( - 9 0 ^ { \circ } ) \hat { \mathbf { v } } ( B ) } & { \mathrm { i f ~ } \angle \left( \hat { \mathbf { v } } ( B ) , \hat { \mathbf { v } } ( B _ { i } ^ { p r e d } ) \right) > 0 ^ { \circ } } \end{array} \right. \right) } \end{array}
$$

where each of the $M _ { B }$ neighbouring predator boids $B _ { i } ^ { p r e d }$ satisfy

$$
\mathrm { d i s t } ( B , B _ { i } ^ { p r e d } ) < r _ { P } ^ { p r e y 2 } \quad \wedge \quad \mathrm { i n F o v } ( B , B _ { i } ^ { p r e d } ) = \mathrm { t r u e } \quad \wedge \quad \mathrm { o c c l u d e d } ( B , B _ { i } ^ { p r e d } ) = \mathrm { f a l s e } \quad \wedge \quad \mathrm { o c c u l u d e d } ( B , B _ { i } ^ { p r e d } ) = \mathrm { f a l s e } \quad \wedge \quad \mathrm { o c c u l u d e d } ( B , B _ { i } ^ { p r e d } ) = \mathrm { f a l s e } \quad \wedge \quad \mathrm { o c c u l u d e d } \quad \wedge \quad \mathrm { o r } \quad \mathrm { o c c u l u d e d } \quad \wedge \quad { \wedge }
$$

• Repel: B repels the positions of all approaching predator boids $B _ { i } ^ { p r e d }$ (position-based escaping)

$$
\begin{array} { r } { \hat { \mathbf { a } } _ { \mathbf { r e p } } ( B ) \ = \ n o r m \left( \sum _ { i = 1 } ^ { M _ { B } } ( \mathbf { p } ( B ) - \mathbf { p } ( B _ { i } ^ { p r e d } ) ) \right) } \end{array}
$$

where each of the $M _ { B }$ neighbouring predator boids $B _ { i } ^ { p r e d }$ satisfy

$$
\mathrm { d i s t } ( B , B _ { i } ^ { p r e d } ) < r _ { P } ^ { p r e y 2 } \quad \wedge \quad \mathrm { i n F o v } ( B , B _ { i } ^ { p r e d } ) = \mathrm { t r u e } \quad \wedge \quad \mathrm { o c c l u d e d } ( B , B _ { i } ^ { p r e d } ) = \mathrm { f a l s e } \quad \wedge \quad \mathrm { o c c u l u d e d } ( B , B _ { i } ^ { p r e d } ) = \mathrm { f a l s e } \quad \wedge \quad \mathrm { o c c u l u d e d } ( B , B _ { i } ^ { p r e d } ) = \mathrm { f a l s e } \quad \wedge \quad \mathrm { o c c u l u d e d } \quad \wedge \quad \mathrm { o r } \quad \mathrm { o c c u l u d e d } \quad \wedge \quad { \wedge }
$$

• Wiggle: B wiggles by some wiggle angle $\theta _ { w i g }$ and frequency $f _ { w i g }$ if a predator is near (shake-of escaping)

$$
\hat { \bf a } _ { \bf w i g } ( B ) = \left\{ \begin{array} { l l } { { \bf R } ( \theta _ { w i g } , f _ { w i g } ) \hat { \bf v } ( B ) } & { \mathrm { i f ~ } M _ { B } > 0 } \\ { \hat { \bf v } ( B ) } & { \mathrm { i f ~ } M _ { B } = 0 } \end{array} \right.
$$

where each of the $M _ { B }$ neighbouring predator boids $B _ { i } ^ { p r e d }$ satisfy

$$
\mathrm { d i s t } ( B , B _ { i } ^ { p r e d } ) < r _ { P } ^ { p r e y 2 } \quad \wedge \quad \mathrm { i n F o v } ( B , B _ { i } ^ { p r e d } ) = \mathrm { t r u e } \quad \wedge \quad \mathrm { o c c l u d e d } ( B , B _ { i } ^ { p r e d } ) = \mathrm { f a l s e } \quad \wedge \quad \mathrm { o c c u l u d e d } ( B , B _ { i } ^ { p r e d } ) = \mathrm { f a l s e } \quad \wedge \quad \mathrm { o c c u l u d e d } ( B , B _ { i } ^ { p r e d } ) = \mathrm { f a l s e } \quad \wedge \quad \mathrm { o c c u l u d e d } \quad \wedge \quad \mathrm { o r } \quad \mathrm { o c c u l u d e d } \quad \wedge \quad { \wedge }
$$

We enhanced $( \mathrm { O j o }$ et al., 2024)’s wiggle behaviour implementation by redefining the wiggle in terms of a sine wave with specific angle and frequency, to resemble more naturalistic behaviour.

## 2.1.3 Simulation

To analyse the emerging movement behaviour of the boids, a field of $S _ { x } \times S _ { y }$ is initialised with M predators and N prey. Governed by their predator- and prey-specific parameters and coeficients, movement behaviour will naturally emerge. When a predator ‘catches’ a prey (dist $( B _ { i } ^ { p r e d } , B _ { j } ^ { p r e y } ) \le ( r _ { S } ^ { p r e d } ) ^ { 2 } )$ , this prey is removed from the field.

## 2.2 Evolutionary Strategy (ES)

To study the research question at hand, we constructed an Evolutionary Strategy (ES) (Beyer and Schwefel, 2002; Slowik and Kwasnicka, 2020) around the aforementioned simulation. This ES consists of $\mathcal { N } _ { g }$ generations, of which each generation runs $\mathcal { N } _ { s }$ simulations – defining the population size. Each simulation then denotes one individual of the population, and each simulation is run for a simulation time of $\tau$ steps – having a fixed number of predators (M) and prey $( N )$ , a fixed set of predator and prey parameters, and a fixed predator hunting strategy $\hat { \mathbf { a } } ( B ) \in \left\{ \hat { \mathbf { a } } _ { \mathbf { a t t c } } ( B ) , \hat { \mathbf { a } } _ { \mathbf { a t t n } } ( B ) , \hat { \mathbf { a } } _ { \mathbf { a t t r } } ( B ) , \hat { \mathbf { a } } _ { \mathbf { a t t p } } ( B ) \right\}$

Crucially, the prey coeficients are variable across simulations, and thereby characterise the movement dynamics of a unique simulation. Therefore, we define the ‘gene’ of a simulation s as their prey coeficients, $g e n e ( s ) : = \langle c _ { a l i s } , c _ { c o h s } , c _ { s e p _ { s } } , c _ { d o d s } , c _ { r e p _ { s } } , c _ { w i g _ { s } } \rangle$

## 2.2.1 Fitness

The fitness of a simulation $s ,$ denoted $f ( s )$ , is defined from the prey’s perspective, and reflects the number of surviving prey $N \tau \leq N$ at the end of simulation (after T time-steps):

$$
f ( s ) = N _ { T }
$$

## 2.2.2 Generation Updating

Given a generation $g _ { t }$ of $\mathcal { N } _ { s }$ simulations (individuals), the two fittest simulations are automatically carried over to the next generation $g _ { t + 1 } - \mathrm { e n s u r i n g }$ that the most successful combination of coeficients are not lost (elitism, where elite size $\mu _ { e } = 2 )$ . To create the remaining $\mathcal { N } _ { s } - 2$ individuals, repeatedly two ‘parents’ are selected by means of fitness-proportional parent selection, i.e., sampled where simulation s has a probability to be selected of:

$$
p ( s ) = { \frac { f ( s ) } { \sum _ { i = 1 } ^ { N _ { s } } f ( s _ { i } ) } }
$$

Given two sampled parents $s _ { i } , s _ { j }$ of generation $g _ { t }$ , two children $s _ { i ^ { \prime } } ^ { \prime } , s _ { j } ^ { \prime } .$ <sub>′</sub> are created to populate generation $g _ { t + 1 }$ We employ crossover to explore the parameter space, where a random crossover point c defines how the parents’ genes are recombined:

$$
\begin{array} { r } { g e n e ( s _ { i ^ { \prime } } ^ { \prime } ) = g e n e ( s _ { i } ) _ { [ : c ] } + g e n e ( s _ { j } ) _ { [ c : ] } } \\ { g e n e ( s _ { j ^ { \prime } } ^ { \prime } ) = g e n e ( s _ { j } ) _ { [ : c ] } + g e n e ( s _ { i } ) _ { [ c : ] } } \end{array}
$$

Furthermore, mutation adds a small random value $\epsilon _ { \bigstar } \in [ - 0 . 1 , 0 . 1 ]$ to each prey coeficient $c _ { \perp } \in g e n e ( s _ { i ^ { \prime } } ^ { \prime } ) , g e n e ( s _ { j ^ { \prime } } ^ { \prime } )$ by mutation rate $\mu { : }$

$$
c _ { \perp }  { \{ { c _ { \perp } + \epsilon _ { \perp } \mathrm { ~ { ~ w i t h ~ p r o b a b i l i t y ~ } } \mu }  } _ {  \mathrm { { ~ w i t h ~ p r o b a b i l i t y ~ } } 1 - \mu \mathrm { { ~ } } }  
$$

The resulting prey coeficients are clipped to $0 ~ \leq ~ c _ { \bigstar } ~ \leq ~ 1$ to keep them within the allowed range. This reproduction process is repeated for $\frac {  { N _ { s } } - 2 } { 2 }$ times per generation update – hereby ensuring that the next generation again counts $\mathcal { N } _ { s }$ individuals.

## 2.3 Experimental Design

To answer our research question, we regarded the predator’s hunting strategy $\hat { \mathbf { a } } ( B )$ as the independent variable, and the distribution of prey coeficients in $g e n e ( s ) : = \langle c _ { a l i s } , c _ { c o h s } , c _ { s e p _ { s } } , c _ { d o d s } , c _ { r e p _ { s } } , c _ { w i g _ { s } } \rangle$ over generations as the dependent variables of interest – characterising the evolution of the prey’s movement behaviour.

For each hunting strategy $\hat { \mathbf { a } } ( B ) \in \{ \hat { \mathbf { a } } _ { \mathbf { a t t c } } ( B ) , \hat { \mathbf { a } } _ { \mathbf { a t t n } } ( B ) , \hat { \mathbf { a } } _ { \mathbf { a t t r } } ( B ) , \hat { \mathbf { a } } _ { \mathbf { a t t p } } ( B ) \}$ , we ran the ES for $\mathcal { N } _ { g } \ = \ 3 0$ generations, keeping all other parameters except the prey coeficients fixed according to Table 1. For the first generation $g _ { 1 }$ , we initialised all $\mathcal { N } _ { s } ~ = ~ 3 0$ simulations individually with random prey coeficients, i.e., $g e n e ( s ) \sim \mathcal { U } ( 0 , 1 ) ^ { 6 }$ . This generation was then evolved over $\mathcal { N } _ { g } = 3 0$ generations, while recording all simulations’ prey coeficients $g e n e ( s ) : = \langle c _ { a l i } , c _ { c o h } , c _ { s e p } , c _ { d o d } , c _ { r e p } , c _ { w i g } \rangle$ , as well as their fitnesses $f ( s )$ . These coeficients were then aggregated to assess their distributions over generations – hereby assessing the prey’s evolutionary response to distinct hunting strategies. Over generations, we expected the prey coeficients to converge to their optimal values, and as a result, the mean simulation fitness (collective survival rate) to rise. To account for stochasticity (while bearing our computational limitations in mind), for each predator hunting strategy we repeated this ES trial five times.

The optimal parameter settings for the main experiments (see Table 1) were largely found by trial-and-error, as motivated in Appendix A. We scrutinised $r _ { P } ^ { p r e y } , f o v ^ { p r e y } , \mathcal { N } _ { s }$ and $\mu$ into more detail, as these parameters showed to have most impact in the stability of the ES. In tuning these parameters, our goal was to identify configurations that led to both an increase in collective fitness over generations and stable convergence of the prey coeficients. As we note that simulation results may heavily depend on these fixed parameter settings, the impact of varying these parameters was systematically assessed through the sensitivity analyses (see Appendix B).

## 3 Results

By conducting the experiments described in subsection 2.3, we recorded the dynamics of the prey coeficients – gene(s) := ⟨c<sub>ali</sub>, c<sub>coh</sub>, c<sub>sep</sub>, c<sub>dod</sub>, c<sub>rep</sub>, c<sub>wig</sub>⟩ – along with the fitnesses f(s) of each simulation run (i.e., ES individual) over 30 generations – yielding 31 values (initialisation included). For each predator hunting strategy, this procedure was repeated in five independent ES trials to reduce the impact of stochasticity. The resulting dynamics were aggregated to assess the prey’s evolutionary response to diferent hunting strategies.

![](images/5f8cb46ddfe8c75606b924c586695191704e46af1f15fa7c06169d2400da6559.jpg)  
(a) Attack-centroid hunting strategy,  
ˆa(B) = ˆa<sub>attc</sub>(B)

![](images/079f28faf7b71448b7a4e2168b95fe386e66e14252fca8a54967967a25e0c532.jpg)  
(b) Attack-nearest hunting strategy,

ˆa(B) = ˆa<sub>attn</sub>(B)  
![](images/d85a7a84c397b11685439f44cee689b3f4d603aad8a2182cebe5adc0a6d020c3.jpg)  
(c) Attack-random hunting strategy, ˆa(B) = ˆa<sub>attr</sub>(B)

![](images/013adca8aaa526a3f609e20d5dca232a77a3648ef3d2058e89a94694183c838b.jpg)

(d) attack-peripheral hunting strategy, ˆa(B) = ˆa<sub>attp</sub>(B)
<table><tr><td>O</td><td>ES 1 - simulation O</td><td>ES 2 - simulation</td><td>ES 3 - simulation</td><td>ES 4 - simulation</td><td>O ES 5 - simulation</td></tr><tr><td>ES 1 - mean</td><td></td><td>ES 2 - mean</td><td>ES 3 - mean</td><td>ES 4 - mean</td><td>ES 5 - mean</td></tr></table>

Figure 1: Evolution of the distribution of fitnesses f(s) over generations, for five diferent ES trials (corresponding to five diferent colours), per predator hunting strategy. Each dot corresponds to an individual simulation, where overlapping simulations result in darker-shaded dots. The line represents the population’s mean fitness. Note that for every ES trial, the mean fitness remains stable or rises over generations.

Generally speaking, the mean population fitness remained relatively stable (see Figure 1a), or showed a very slight (see Figure 1c and 1d) to more pronounced (see Figure 1b) increase. As no ES trial showed a downwards nor heavily fluctuating trend of mean fitness, this confirms the desired functioning of the ES. In other words, the prey managed to learn to improve their set of prey coeficients gene(s) over generations, leading to movement behaviour that enhanced their collective survival rate – within the fitness limitations set by the predator’s hunting strategy $\hat { \mathbf { a } } ( B ) \in \{ \hat { \mathbf { a } } _ { \mathbf { a t t c } } ( B ) , \hat { \mathbf { a } } _ { \mathbf { a t t n } } ( B ) , \hat { \mathbf { a } } _ { \mathbf { a t t r } } ( B ) , \hat { \mathbf { a } } _ { \mathbf { a t t p } } ( B ) \}$ , which we discuss in detail below. Furthermore, we note that per hunting strategy, the mean fitness trends of the five ES trials agree – indicating that the observed collective fitness improvements are dependent on, and characteristic to, the employed hunting strategy.

In the remainder of this section, we present the results relating to each individual hunting strategy in more detail. Apart from reflecting upon the strategy-specific fitness trends, we will also assess the trends per prey coeficient – which are visualised in Figure 2, 3, 4 and 5. For these, in general, we noted that within each ES trial, the distributions of prey coeficients converge over generations. This convergence can be attributed to the ES progressing towards a local optimum. As each ES trial shows convergent behaviour of the coeficients, we therefore conclude that the coeficients’ landscape allows suficient opportunities for fitness improvement. However, this coeficients’ landscape seems to contain multiple local optima – as demonstrated by some ES trials that tend to bifurcate as they unfold – resulting in the ES converging to two distinct values. This points at the coeficients’ convergence being inter-dependent.

Regarding the prey coeficients’ agreement between ES trials of the same hunting strategy, on the other hand, convergence patterns do not always agree. Here we observe that, depending on the hunting strategy, some prey coeficients show convergence of ES trials towards the same local optima – indicating that the coeficients’ landscape is suficiently steep to globally converge, and suggesting that the coeficient is rather crucial for the collective survival rate of the prey. For other coeficients, however, diferent ES converge towards diferent local optima – exhibiting more variability and a sparser distribution, and possibly indicating a less pronounced or more context-dependent influence on prey survival. The collected results for each hunting strategy will be assessed into more detail in the following subsections, while a comparative and comprehensive analysis, along with our interpretation for our findings, is provided in the discussion section 4.

## 3.1 Attack-Centroid Hunting Strategy $( \widehat { \mathbf { a } } ( B ) = \widehat { \mathbf { a } } _ { \mathbf { a t t c } } ( B ) )$

![](images/4f0778eaa4d973518ed18c123d98d815d2e515d6f0a4e65338486d7bada5d234.jpg)  
Figure 2: From left to right, top to bottom: evolution of prey coeficients $\begin{array} { r l } { g e n e ( s ) } & { { } : = } \end{array}$ $\left. c _ { a l i } , c _ { c o h } , c _ { s e p } , c _ { d o d } , c _ { r e p } , c _ { w i g } \right.$ under the attack-centroid hunting strategy, $\widehat { \mathbf { a } } ( B ) = \widehat { \mathbf { a } } _ { \mathbf { a t t c } } ( B )$ , for five diferent ES trials – corresponding to five diferent colours. Each dot ( ) corresponds to an individual simulation, where overlapping simulations result in darker-shaded dots. The vertical bars indicate the population’s error margin $( \mathrm { m e a n } \pm \mathrm { s t d } )$ at the last generation of each ES, whereas the stars (⋆) indicate the coeficient’s value of the fittest individual of the last generation.

With predators attacking the centroid of the prey flock $( \widehat { \mathbf { a } } ( B ) = \widehat { \mathbf { a } } _ { \mathbf { a t t c } } ( B ) )$ , we can observe from Figure 1a how the initial generations’ fitnesses already start at near-maximal fitnesses – which is sustained as each ES unfolds. In other words, the predator’s attack-centroid strategy results in little prey to be caught, with little dependence on the prey’s movement dynamics.

Figure 2 shows that coeficients tend to converge within their ES run – reflected by their decreased standard deviation at the last generation. Furthermore, the coeficient distributions are generally centred around the coeficient’s value of the fittest individual – indicating that fitness improvements were the driving force behind the convergence of the ES. It should be noted, however, that convergence does not always agree between ES trials. Whereas we see that the escape tendencies, i.e., dodge $( c _ { d o d } )$ , repel $\left( c _ { r e p } \right)$ and wiggle $( c _ { w i g } )$ , all tend to converge towards higher values in [0.4, 1.0] – an efect which is most pronounced for the repel coeficient, as indicated by the highest agreement of error bars between ES trials.

For the flocking tendencies, i.e., alignment $\left( c _ { a l i } \right)$ , cohesion $\left( c _ { c o h } \right)$ and separation $\left( c _ { s e p } \right)$ , however, the ES shows little agreement of convergence between ES trials – indicated by the lack of overlap of error margins at the last generation. Taken together, these findings suggest that individual fitness improvements are mostly realised by avoiding the predator, as opposed to changing the flock’s global dynamics; this is also evident from simulation runs using the evolved coeficients for the five ES runs. Visual inspection of the fittest simulation of each ES trial revealed that for the attack-centroid strategy the prey mainly stays around the edges of the toroidal environment, while the predator moves around the centre of the environment which coincides with the centroid of the flock (given the toroidal environment). Prey consistently react to the predator by changing their direction while generally keeping the flock formation (albeit considerably less than for all other attack strategies).

## 3.2 Attack-Nearest Hunting Strategy $( \widehat { \mathbf { a } } ( B ) = \widehat { \mathbf { a } } _ { \mathbf { a t t n } } ( B ) )$

![](images/f5166fdf685411b2b5c2723ef6e361bc83e1c07e344cf405919e01e3c24ab76f.jpg)  
Figure 3: From left to right, top to bottom: evolution of prey coeficients $\begin{array} { r l } { g e n e ( s ) } & { { } : = } \end{array}$ $\left. c _ { a l i } , c _ { c o h } , c _ { s e p } , c _ { d o d } , c _ { r e p } , c _ { w i g } \right.$ under the attack-nearest hunting strategy, $\widehat { \mathbf { a } } ( B ) = \widehat { \mathbf { a } } _ { \mathbf { a t t n } } ( B )$ , for five diferent ES trials – corresponding to five diferent colours. Each dot ( ) corresponds to an individual simulation, where overlapping simulations result in darker-shaded dots. The vertical bars indicate the population’s error margin $( \mathrm { m e a n } \pm \mathrm { s t d } )$ at the last generation of each ES, whereas the stars (⋆) indicate the coeficient’s value of the fittest individual of the last generation.

When the predator attacks the nearest prey in the flock $( \widehat { \mathbf { a } } ( B ) = \widehat { \mathbf { a } } _ { \mathbf { a t t n } } ( B ) )$ , we can observe from Figure 1b how the escape for the prey is generally more dificult than in the previous attack-centroid scenario. Indeed, the initial fitness in the first generations is relatively low – showing a survival rate lower than 50% – and, despite steadily increasing in the next ES generations, it plateaus after about 10 generations, reaching an upper bound of 60 (i.e. 60% survival rate).

Figure 3 shows how the prey coeficients evolved over generations. We observe moderate to strong convergence within ES trials for most coeficients – except wiggle, as indicated by the exceptionally wide error bars. Fitness again appears to be the driving force behind strong ES convergence – as converging ES oftentimes also include the fittest individual in their error margin (which is not the case for wiggle). Between ES trials, on the other hand, we only see consistent convergence across ES trials for alignment, separation, dodge, and repel. Especially the agreement for repel is striking, as ES strongly converge both within as well as between trials. Notably, repel converges to a particularly low coeficient value of [0.0, 0.2] – suggesting that this is a crucial requirement for prey survival under the attack-nearest strategy. Cohesion and wiggle exhibited more variability and sparser distributions at the final generation – indicating a less pronounced or more context-dependent influence on prey survival. When visually inspecting the fittest simulation of each ES trial, we observe that for three out of five ES trials, the evolved behaviour was to stay in perfect starting formation – cruising straight ahead, without reacting to the predator, resulting in equidistant prey which were being eaten one-by-one until the end of the simulation.

## 3.3 Attack-Random Hunting Strategy $( \hat { \mathbf { a } } ( B ) = \hat { \mathbf { a } } _ { \mathbf { a t t r } } ( B ) )$

For the attack-random hunting strategy $\widehat { \mathbf { a } } ( B ) = \widehat { \mathbf { a } } _ { \mathbf { a t t r } } ( B )$ , the fitness improvement is less pronounced than for the aforementioned attack-nearest strategy (see Figure 1) – showing a near-negligible fitness improvement. Figure 4 demonstrates how the prey coeficients evolved over generations. We observe moderate convergence within ES trials for all coeficients – as confirmed by their medium-sized error margins. We furthermore see that between ES, the coeficients only converge towards similar value ranges for the escape tendencies – i.e., dodge, repel and wiggle. It is worth noting that repel again shows preference for lower values in [0.0, 0.4], whereas the other two tend to be in the medium-to-high range (dodge in [0.6, 1.0] and [0.4, 0.9]). For the flocking tendencies, $\mathrm { i . e . } .$ alignment, cohesion and separation, there is higher variability and no clear agreement across ES trials. Visual inspection of the fittest simulation of each ES trial again shows a similar behaviour as for attack-nearest – with prey mostly staying in their initial formation. However, behaviour is more variable across prey, and a higher tendency for predator avoidance can be observed.

![](images/15ac6c2e6285bf31551395abbbe059f6459ff1db8657b007ce8e015d80e1c9b1.jpg)  
Figure 4: From left to right, top to bottom: evolution of prey coeficients $\begin{array} { r l } { g e n e ( s ) } & { { } : = } \end{array}$ $\left. c _ { a l i } , c _ { c o h } , c _ { s e p } , c _ { d o d } , c _ { r e p } , c _ { w i g } \right.$ under the attack-random hunting strategy, $\widehat { \mathbf { a } } ( B ) = \widehat { \mathbf { a } } _ { \mathrm { a t t r } } ( B )$ , for five diferent ES trials – corresponding to five diferent colours. Each dot ( ) corresponds to an individual simulation, where overlapping simulations result in darker-shaded dots. The vertical bars indicate the population’s error margin $( \mathrm { m e a n } \pm \mathrm { s t d } )$ at the last generation of each ES, whereas the stars (⋆) indicate the coeficient’s value of the fittest individual of the last generation.

## 3.4 Attack-Peripheral Hunting Strategy $( \widehat { \mathbf { a } } ( B ) = \widehat { \mathbf { a } } _ { \mathbf { a t t p } } ( B ) )$

The fitness distribution for the attack-peripheral hunting strategy $( \hat { \mathbf { a } } ( B ) = \hat { \mathbf { a } } _ { \mathbf { a t t p } } ( B ) )$ shows a comparative spread, where the fitness distribution ranges from 52 to 91 (i.e. 9 to 48 preys killed in the simulation’s runtime). Despite this variability, a slight fitness improvement is achieved as the ES trials evolve. Figure 5 shows how the coeficients evolved over generations. We again observe moderate to strong within-ES convergences for all coeficients – where alignment and cohesion show the most narrow error margins. Regarding the between-ES agreement of the converged-to values, however, we observe a large variability across ES for most coeficients – as only repel shows particular preference for low ([0.0, 0.4]) values, and wiggle for higher ([0.5, 1.0]) values. The remaining coeficients all vary considerably in their convergences across ES runs. When visually assessing the ES trial’s fittest simulations, we see a very similar behaviour to the attack-random results – with prey generally keeping their formation, but being less constrained than for the attack-nearest simulations. Further, we can clearly observe wiggling behaviour, which was not the case for any of the other attack strategy results.

## 4 Discussion

We investigated which relative contributions of six prey behaviours – alignment $\left( c _ { a l i } \right)$ , cohesion $\left( c _ { c o h } \right)$ , separation $\left( c _ { s e p } \right)$ , dodge $( c _ { d o d } )$ , repel $\left( c _ { r e p } \right)$ , and wiggle $( c _ { w i g } )$ – prey boids converge to, for a given predator strategy, using an ES. We found that the ES was successful in maximising fitness, and converging on specific coeficients, typically driven by local improvements in the fitness landscape. The behaviours that evolved for the diferent predator attack strategies are largely similar, except for the attack-centroid strategy – which favours more individualistic predator-escaping behaviour – as compared to the other three strategies – which favour flocking tendencies more. Still, as Section 3 demonstrated, each strategy favoured a diferent combination of coeficients and difered in terms of convergence within and between ES trials. Only the ES runs for the attack-nearest strategy showed a clear improvement in fitness. For all other strategies, no relevant improvements in fitness could be observed. This can be explained by the properties of the diferent attack strategies. The attack-nearest strategy resulted in the lowest survival rate of the prey, which is in line with our hypothesis, and furthermore agrees with previous research that found this strategy to be the most eficient (Ojo et al., 2023; von Moll et al., 2016). The optimal prey survival behaviour under this hunting strategy is to “do nothing” – that is, to stay in the initial starting formation, which perfectly equispaces the prey – preventing the predator from eating more than one prey at a time (which is a possible scenario in our simulations). Since the simulation is run for a fixed time and the predator is faster than the prey, prey essentially “wait-out” the end of the simulation. This generates a very reliable fitness. Reacting to the predator using the escape tendencies can result in higher fitness, but not reliably so – as by chance in some simulations, multiple clustered prey could be eaten by the predator in one go. In a sense, this shows how environmental context influences optimal behaviour (Palmer and Packer, 2021).

![](images/50eb3cdd22d0a3353459f4d271fc57c3bb52fa2b032e5f9a7150567b0655a344.jpg)  
Figure 5: From left to right, top to bottom: evolution of prey coeficients gene(s) := $\left. c _ { a l i } , c _ { c o h } , c _ { s e p } , c _ { d o d } , c _ { r e p } , c _ { w i g } \right.$ under the attack-peripheral hunting strategy, $\widehat { \mathbf { a } } ( B ) = \widehat { \mathbf { a } } _ { \mathbf { a t t p } } ( B )$ , for five diferent ES trials – corresponding to five diferent colours. Each dot ( ) corresponds to an individual simulation, where overlapping simulations result in darker-shaded dots. The vertical bars indicate the population’s error margin (mean ± std) at the last generation of each ES, whereas the stars (⋆) indicate the coeficient’s value of the fittest individual of the last generation.

Similarly, the evolved behaviours for the other attack strategies can be understood. The attack-centroid strategy resulted in the highest survival rate for the prey – again aligning with our hypothesis and previous work (Demˇsar and Lebar Bajec, 2014). This can be easily explained by considering that the flock’s centroid typically does not coincide with an individual prey – resulting in the predator aiming to catch ‘empty space’. As a consequence, there is less pressure for the prey to flock, as long as they avoid the predator – as indicated by the relatively high escape coeficients. Since many possible combinations of escape behaviours can be successful for this strategy, the random initialization and all consecutive generations achieve a very high average fitness, that can hardly be improved upon due to the stochasticity inherent in both the boid’s model and ES.

The attack-random and attack-peripheral strategies place somewhere in between attack-nearest and attackcentroid – both in terms of highest achievable fitness for the prey, as well as the evolved behaviours. Both strategies are more eficient than attack-centroid, since they target and hunt actual, individual prey, but fall short of attack-nearest – which targets the closest prey, as observed by Ojo et al. (2023). This is reflected in the ES achieving maximal fitness values between the two extreme strategies. Both attack-random and attack peripheral are rather unpredictable. For the prey, the evolved behaviour consists of generally staying in flock formation, which aligns with the findings of (Kunz et al., 2006). They turn away from the predator if it is spotted, however, they avoid sharp escape manoeuvres that break flock formation. This exploits the ineficient hunting of the predator – which regularly chooses prey that are further away – leading to long travel times during which prey are only caught by chance.

Altogether, in terms of evolved coeficients over all hunting strategies, we can highlight a general tendency towards a relatively low repel coeficient across the ES trials – while other coeficients did not present the same consistency. Other, more subtle patterns are a relatively high dodge and wiggle coeficient – which are deemed beneficial to avoid predators’ attacks. This contrasts with the findings from Ojo et al. (2023), who found position-based escape (repel) to perform better than direction-based escape (dodge). This deviation could potentially be attributed to the interaction between coeficients in our research, while Ojo et al. (2023) only investigated behaviours in isolation. No clear pattern can be identified for the boids coeficients, as multiple coeficient combinations result in the observed flocking behaviour.

## 4.1 Limitations & Future Work

Our study has several limitations, which also pave the way for promising directions for future work. We highlight the following:

• Environment Setting: The simulation run in our experiment assumed a 2D toroidal environment with limited size. In future research, the environment could be expanded, in terms of both size and dimensions, potentially removing the toroidal boundaries to better approximate open or higher-dimensional spaces. Furthermore, the addition of obstacles or other environmental features may vary the behaviour of the flock, thus possibly leading to interesting findings.

• Initialisation and stochasticity: In order to account for stochasticity and isolate the efects of the behavioural coeficients, every simulation has been run starting from the exact same initial position and direction of the boids. However, to enhance the realism of those simulations and introduce stochasticity, diferent initial setups could be explored.

• Computational Cost: Despite the extensive use of parallelisation in the experiment implementation, the demanding computational cost of the simulations prohibited long ES runs, and constrained our research to test ES for only $\mathcal { N } _ { g } = 3 0$ generations (whereas most of the other studies run their GA for 100 or more (Olsen et al., 2018; Kunz et al., 2006)). With more computational resources, a more extensive study would be recommended to efectively assess the convergence of the ES.

• Parameters: Given the broad parameter space, there is significant potential for further exploration. While this study fixed most parameters to maintain a realistic simulation consistent with the chosen hunting strategy, future research could examine a wider array of strategies and parameter settings, and/or combine hunting strategies, such as first attacking the centre of a flock and following up with attacking nearest prey (Demˇsar and Lebar Bajec, 2014). Also, introducing predator collaboration in the hunting tactic by means of increasing the number of predators, could lead to interesting prey dynamics.

## 5 Conclusion

We investigated to what movement behaviour the prey evolve to, in terms of relative contributions of alignment, cohesion, separation, dodge, repel and wiggle tendencies, to maximise their collective survival – under diferent predator hunting strategies. From the prey perspective, we conclude that flocking (relating to the coeficients of alignment, cohesion and separation) is usually beneficial, whereas the optimality of escaping (relating to the coeficients of dodge, repel and wiggle) slightly varies according to the hunting strategy adopted by the predator. From the perspective of the predators, we can conclude that attacking the centroid of the prey flock, or attacking a randomly-selected prey, is not an efective hunting strategy, while attacking the nearest prey is reliably most successful. This research serves to improve our understanding of the complex interactions between prey and predator behaviours. We encourage future research to gain deeper insights into the dynamics that were presented here.

## References

Alaliyat, S., Obeidat, M., Aljarah, I., Al-Zoubi, A. M., and Faris, H. (2022). Optimization of boids swarm model based on genetic algorithm and particle swarm optimization algorithm: A comparative study. Swarm and Evolutionary Computation, 68:100982.

Beyer, H.-G. and Schwefel, H.-P. (2002). Evolution strategies–a comprehensive introduction. Natural computing, 1:3–52.

Chen, Y.-W., Kobayashi, K., Huang, X., and Nakao, Z. (2006). Genetic algorithms for optimization of boids model. In International Conference on Knowledge-Based and Intelligent Information and Engineering Systems, pages 55–62. Springer.

Demˇsar, J. and Lebar Bajec, I. (2014). Simulated predator attacks on flocks: a comparison of tactics. Artificial Life, 20(3):343–359.

Hahn, C., Phan, T., Gabor, T., Belzner, L., and Linnhof-Popien, C. (2019). Emergent escape-based flocking behavior using Multi-Agent reinforcement learning. In The 2019 Conference on Artificial Life, Cambridge, MA. MIT Press.

Kunz, H., Z¨ublin, T., and Hemelrijk, C. (2006). On prey grouping and predator confusion in artificial fish schools.

Lee, S.-H., Pak, H., and Chon, T.-S. (2006). Dynamics of prey-flock escaping behavior in response to predator’s attack. Journal of theoretical biology, 240(2):250–259.

Oboshi, T., Kuroe, Y., and Uozumi, A. (2002). Evolving schooling behaviors to escape from predator. In Proceedings of the 2002 Congress on Evolutionary Computation, volume 2, pages 1410–1415. IEEE.

Ojo, M., Krajnc, M., and Adˇzaga, M. (2024). Collective-behavior-groupb public. https://github.com/MOj0/ Collective-Behavior-GroupB. Accessed: 2025-06-23.

Ojo, M., Krajnc, M., Adˇzaga, M., and Kuhar, J. (2023). Predator-prey simulation using boids model. FRIteza, pages 1–4.

Olsen, M. M., Laspesa, J., and Taylor-D’Ambrosio, T. (2018). On genetic algorithm efectiveness for finding behaviors in agent-based predator prey models. In Proceedings of the 50th Computer Simulation Conference, pages 1–12.

Palmer, M. S. and Packer, C. (2021). Reactive anti-predator behavioral strategy shaped by predator characteristics. PloS one, 16(8):e0256147.

Papadopoulou, M., Hildenbrandt, H., Sankey, D. W. E., Portugal, S. J., and Hemelrijk, C. K. (2022). Self organization of collective escape in pigeon flocks. PLoS Comput. Biol., 18(1):e1009772.

Reynolds, C. W. (1987). Flocks, herds and schools: A distributed behavioral model. ACM SIGGRAPH Computer Graphics, 21(4):25–34.

Slowik, A. and Kwasnicka, H. (2020). Evolutionary algorithms and their applications to engineering problems. Neural Computing and Applications, 32:12363–12379.

von Moll, P., Riedmiller, M., and R¨ofer, T. (2016). Evolutionary design of cooperative predation strategies. Artificial Life and Robotics, 21:127–137.

## A Fixed Parameter Settings

We selected the parameter settings in Table 1 to be fixed during our main experiments.

<table><tr><td>Parameter</td><td>Explanation</td><td>Value</td></tr><tr><td colspan="3">Boids Model</td></tr><tr><td> $\overline { { S _ { x } \times S _ { y } } }$ </td><td>field size (px)</td><td>1920 × 1080 Predator</td></tr><tr><td rowspan="4"> $M , N$   $r _ { P } ^ { p r e d , p r e y }$   $r _ { S } ^ { \bar { p } r e d , p r e y }$   $\ddot { f o v } ^ { p r e d , p r e y }$ </td><td>number of boids</td><td>Prey 1 100</td></tr><tr><td>perception radius (px)</td><td>3000 {300, 750, 1500, 3000}</td></tr><tr><td>separation radius (px) field of view, per eye (°)</td><td>100 50 120 60 120 240 360 2 2 2</td></tr><tr><td>maximum rotation angle (°)</td><td>2 2 90</td></tr><tr><td> $\theta _ { m a r } ^ { p r e d , p r e y }$   $a ^ { p r e d , p r e y }$ </td><td>acceleration  $\mathrm { ( p x / s t e p ^ { 2 } ) }$ </td><td>90 5000 2500</td></tr><tr><td> $v ^ { p r e d , p r e y }$ </td><td>velocity  $\left( \mathrm { p x } / \mathrm { s t e p } \right)$ </td><td>500 200</td></tr><tr><td> $\theta _ { w i g }$ </td><td>wiggle angle (°)</td><td>30</td></tr><tr><td> $f _ { w i g }$ </td><td>wiggle frequency (rad/step)</td><td>14</td></tr><tr><td>Evolutionary Strategy (ES)</td><td></td><td></td></tr><tr><td colspan="3"></td></tr><tr><td> $\mathrm { f p s }$ </td><td>simulation rate (fps)</td><td>100</td></tr><tr><td> $\tau$ </td><td>duration of a simulation (steps)</td><td>2000</td></tr><tr><td> $T$ </td><td>duration of a simulation (s)</td><td>20</td></tr><tr><td> $\mathcal { N } _ { s }$ </td><td>number of simulations per generation, population size</td><td>{30, 50}</td></tr><tr><td> $\mathcal { N } _ { g }$ </td><td>number of generations</td><td>30</td></tr><tr><td> $\mu _ { e }$ </td><td>elite size</td><td>2</td></tr><tr><td> $\mu$ </td><td>mutation rate</td><td>{0.05, 0.3, 0.7}</td></tr></table>

Table 1: Fixed parameter settings relating to the Boids Model and the evolutionary strategy (ES) that were used for running the main experiments. The settings were tweaked such, that they produced naturalistic predator and prey motion dynamics, and furthermore allowed ES convergence over generations. The parameter values in light grey correspond to the settings examined in the sensitivity analyses (see Appendix B), but not selected for running the main experiments.

Relating to the Boids Model, $S _ { x } \times S _ { y }$ ensured suficient space for the boids to move around, without overcrowding the field. We chose to include one single predator $( M = 1 )$ to better isolate and examine the efects of its hunting strategy on the prey, and to avoid confounding prey behaviour with responses to multiple approaching predators. The choice for $N = 1 0 0$ prey was in line with related literature (Kunz et al., 2006), being an intuitive number to work with, and practical for fitness quantification. The parameters relating to the boids’ motion dynamics, $r _ { P } ^ { p r e d , \hat { p r e y } } , r _ { S } ^ { p r e d , p r e y } ,$ $f o v ^ { p r e d , p r e y } ,$ θ<sup>pred,prey</sup>, a<sup>pred,prey</sup>, v<sup>pred,prey</sup> and wiggle parameters $\theta _ { w i g } , f _ { w i g }$ were tuned such, to produce naturalistic predator and prey motion dynamics. The specific settings of $r _ { P } ^ { p r e y }$ and $f o v ^ { p r e y }$ were further motivated by promoting the stability of ES convergence (see Appendix B).

![](images/df4b6e28447aa12c4f6cceac3481e68de25b3947ead0b49d1dd7de9435ec28ce.jpg)  
Figure 6: Initial layout of the predator (red) and prey (white) that was fixed for every simulation.

Furthermore, for every simulation, we fixed the initial layout of the predator (in red) and prey (in white) to be as displayed in Figure 6. Taking the predator’s field of view $( f o v ^ { p r e d } )$ and the toroidal boundary conditions of the field into account, we opted for this layout to ensure that the predator has suficient candidate prey within sight, such that its hunting strategy directly reflects in its movement behaviour. Furthermore, the evenly-spaced prey grid allows suficient freedom for flocking behaviour to emerge. Taken together, this layout ensures that most is made out of every simulation, within duration T . It is worth noting that, although the initial conditions of each simulation are deterministic, their unfolding is inherently stochastic.

For ES, we fixed fps and $\mathcal { T } \mathrm { ~ ( i n ~ s t e p s ^ { 1 } ) ~ }$ such, that any change in the simulation’s prey coeficients $g e n e ( s )$ were allowed enough time to have a noticeable impact on the simulation’s fitness. Within $\tau ,$ this required a considerable amount of prey to be killed when displaying unsuccessful flocking behaviour, and conversely, suficient chances to escape from the predator when employing successful flocking behaviour. Bearing computational time in mind, we set $\mathcal { N } _ { s }$ and $\mathcal { N } _ { g }$ such, that the ES showed to converge (see Appendix B). Relating to the generational updating, $\mu _ { e }$ and $\mu$ were set such, to result in a balance between exploration and exploitation (see Appendix B).

## B Sensitivity Analyses

To scrutinise the impact of varying the prey perception radius $r _ { P } ^ { p r e y }$ , field of view $f o v ^ { p r e y }$ , number of simulations per generation $\mathcal { N } _ { s }$ and mutation rate $\mu$ on the stability of ES convergence, we performed sensitivity analyses on the aforementioned parameters. The experiments were conducted under the predator’s attack-nearest hunting strategy $( \widehat { \mathbf { a } } ( B ) = \widehat { \mathbf { a } } _ { \mathbf { a t t n } } ( B ) )$ , for $\mathcal { N } _ { g } = 2 0$ generations, keeping all other parameters fixed to the settings in Table 1. For each parameter, the setting that yielded the best results was selected to be held fixed in the main experiments.

Favourable parameter settings are characterised by the population’s ability to ‘learn’ how to survive. In other words, we sought parameter settings that resulted in a rise of the simulations’ mean fitness over generations – corresponding to stable convergence of the simulations’ prey coeficients towards the fittest simulation’s value.

## B.1 Perception Radius $( r _ { P } ^ { p r e y } )$

As Figure 7 illustrates, varying the $\mathrm { p r e y } ^ { \prime } \mathrm { s }$ perception radius across $r _ { P } ^ { p r e y } \in \{ 3 0 0 , 7 5 0 , 1 5 0 0 , 3 0 0 0 \}$ only partly afected the stability of ES convergence within $\mathcal { N } _ { g } = 2 0$ . We noted that for $r _ { P } ^ { p r e y } = 3 0 0 ,$ the distributions of prey coeficients remained rather broadly distributed (or bifurcated) as generations evolved – reflecting a prey coeficient landscape having multiple local optima. This reflected itself as a jumpy behaviour of the fittest simulation’s prey coeficients, and furthermore the inability of the population to steadily converge toward optimal prey coeficients. Together, this seemed to impede the collective fitness to ‘learn’ to climb towards higher fitnesses.

For $r _ { P } ^ { p r e y } \in \{ 7 5 0 , 1 5 0 0 , 3 0 0 0 \}$ , on the other hand, we see a slight learning efect, as the simulations’ mean fitness rises as generations unfold. Furthermore, prey coeficient convergence is seemingly more stable. Based on these results, there was no clear preference for either one of $r _ { P } ^ { p r e y } \in \{ 7 5 0 , 1 5 0 0 , 3 0 0 0 \}$ , therefore we selected $r _ { P } ^ { p r e y } = 7 5 0$ for the main experiments – having the least computational load, given that higher $r _ { P } ^ { p r e y }$ imply more neighbours per prey.

## B.2 Field of View $( f o v ^ { p r e y } )$

As Figure 8 illustrates, varying the prey’s field of view across $f o v ^ { p r e y } \in \{ \frac { 6 0 } { \gamma } , \frac { 1 2 0 } { \gamma } , \frac { 2 4 0 } { \gamma } , \frac { 3 6 0 } { \gamma } \}$ had a minor impact on the stability of the prey coeficients’ convergence within $\mathcal { N } _ { g } = 2 0$ . In the $\bar { f o } v ^ { p r e y } \in \bar { \{ \frac { 3 6 0 } { 2 } } $ case, we observed little-to-no rise in the simulations’ mean fitness, although this was not directly reflected in the prey coeficients’ convergence – that were seemingly equally stable across all four $f o v ^ { p r e y }$ scenarios. For the main experiments, we opted for $\begin{array} { r } { f o v ^ { p r e y } = { \frac { 2 4 0 } { 2 } } } \end{array}$ to mimic nature – in which prey typically have larger fields of view than the predator $( f o v ^ { \mathit { p r e d } } = { \frac { 1 { \bar { 2 } } 0 } { 2 } } )$ ). Furthermore, this parameter setting resulted in a clear rise of the simulations’ mean fitness across generations.

## B.3 Number of simulations per generation $( \mathcal { N } _ { s } )$

Looking at Figure 9, varying the number of simulations per generation $\mathcal { N } _ { s } \in \{ 3 0 , 5 0 \}$ showed most favourable ES dynamics for $ { \mathcal { N } _ { s } } = 3 0$ . Here, the prey coeficients showed to converge more stably to the values of the fittest simulation – whereas for $\mathcal { N } _ { s } = 5 0$ the distribution of the prey coeficients’ values remained broad or strongly bifurcated over generations. Consequently, more simulations with suboptimal fitnesses remained within the population, whereas for $\mathcal { N } _ { s } = 3 0$ they had almost all reached the fitness ceiling. For these reasons, we selected $\mathcal { N } _ { g } = 3 0$ for our main experiments.

## B.4 Mutation rate $( \mu )$

Looking at Figure 10, varying the mutation rate $\mu \in \{ 0 . 0 5 , 0 . 3 , 0 . 7 \}$ showed stronger exploitation efects $\mu = 0 . 0 5$ – as in this scenario, generational variations through mutation were limited, we observe that the prey coeficients’ distributions show little-to-no changes over generations – as variations are almost entirely dependent on crossover with just a 5% chance of mutation. This impedes exploration of the prey coeficients’ landscape towards configurations that result in higher fitnesses. For $\mu = 0 . 7$ , we observed the contrary. Due to the 70% chance of gene mutation, the prey coeficients struggle to converge, as their distributions remain broad over generations. Here in particular, the overemphasis on exploration withholds the mean fitness generation to steadily rise. Best prey coeficients’ convergence and population fitnesses were achieved by balancing exploration and exploitation, as was the case for $\mu = 0 . 3$ . Therefore, this setting was selected for the main experiments.

![](images/59578a77435b1f258b5aa41b6d03351c184d78aa936da0ed83850ab3dfe1f5aa.jpg)

![](images/f7aaa2f7a39bcb0215bdcd7d404732046bbe308032b460034a9d321071688356.jpg)

![](images/ac752bbcec4eaa5b246b8ff5f95f3d588d20648033faa9ee1ebb9ada09a64334.jpg)

![](images/bacb31022ad3c31bff466f4ff22441bff15502945a32d2e4f5822fbe46e44040.jpg)

![](images/8f2868a05c8b1c7ffdc1258fb7d1282492cf4e09d998bb23e20a6b03859cad7d.jpg)

![](images/2c637ead9d5d108bd52cc03289e697664bd32599f0b9620ee5b8f49df19b37f3.jpg)

![](images/03dcfca7bf6f1531cd23ebd565ee5f9cde02b59b529eaed10caa358bca492d02.jpg)

![](images/32d2cd7058100ca337693f607bd44402b65729f740cfcfa9dfa5ea4dad97ba79.jpg)

![](images/ed8104821d5549a81f551112991c308920e765614b63010a43606ec1bf9744f7.jpg)

![](images/9c3f5aa85b21d1282a62bf4f3661d3994f46246810479635b263c4c32ad1dd5b.jpg)

![](images/437d90fbdaa9571d15d64728bd07a9a06be8ccd15f562f53216decdeaf1a474b.jpg)

![](images/642f3c5568e2c88d63b3529013bbfa7489f8f63394cff49c59d49978644495af.jpg)

![](images/67a35def97d75b071c7a65bf3ccb40e651492e16e65518a0d232e43980433517.jpg)

![](images/78e24d730cbbbe27e0b5635030d3332674c3ad9f8c2b55a1bf3e3bcf8916d352.jpg)

![](images/717dc474c866d7908c93d19b0a217c3050050eefb2d9d4e79c55b5741c491d8c.jpg)

![](images/3564d01796bf197d2ea31f3d41d998f6fc72744dc8002a7a613b852e6056ee93.jpg)

![](images/8e8901bc1315ae47d7dcfaf3de298a15190cb82300449b56984ba77d0e09aa99.jpg)

![](images/3453ead6a2027678bf830d8f0fc3245270fbf4651f480400b7428b18f0959730.jpg)

![](images/26bc37ed43db79364e9d43972db245764b8f43d3bf0dc9214bd26cb6c1123216.jpg)

![](images/4db9a8eb2e21800446be3afb444313136718fa048969a153e8c20e7611d87ca2.jpg)

![](images/c12290deba4ad2bd06d1e65a9116810bd5c4a1e17b6ac8a4a2ebf681a471f3ef.jpg)

![](images/bf98fe99831465a3cc05f9e2292967f24f20f162bb8706287ff49af08925ea12.jpg)

![](images/ce51939d4d930803e5ae4349b9c8ee4637e62e529e7833df82ffd0dfc3958aea.jpg)

![](images/4105320ad774e948ddd41ca6b1cc2a135e2868a92ad1ecf627cfe4242c4f4ec3.jpg)  
(a) Evolution of the simulations’ prey coeficients across generations, under varying $r _ { P } ^ { p r e y } .$ . The fittest simulation is marked as a dashed line.

![](images/a10f7033611f0a4841be75cd32b152b72c54da7341855a913b52cdb471d5b6bc.jpg)

![](images/0a6697989888863d3c5fbc70868a634c1a296e6919d08eada7f2ffa9d5b92c96.jpg)

![](images/0d21321b8d2427eaf3203984dbe005537135c48b784b4cf98861ca122cf881bd.jpg)

![](images/24d73845b8754c6f65c86d377cb6e8780b74565486a3264f3f0fb11a15bc846e.jpg)  
(b) Evolution of the simulations’ fitness across generations, under varying $r _ { P } ^ { p r e y }$ . The mean fitness is marked as a solid line.

Figure 7: Comparison of ES convergence in terms of the simulations’ prey coeficients and fitnesses, when varying the prey’s perception radius $r _ { P } ^ { p r e y ^ { - } } \in \ \{ 3 0 0 , 7 5 0 , 1 5 0 0 , 3 0 0 0 \}$ . The sensitivity analyses were run for $\mathcal { N } _ { g } ~ = ~ 2 0$ generations. Other parameter settings were fixed to the settings in Table 1.

f ov<sup>prey</sup> = 60 2  
![](images/0847df4b94d3767eb8be3b9229c1a40926ddfd5653ec0fe3ecf404d26502ac8c.jpg)

![](images/c33074b0e3721939f72c840c11ede8c4ae924b9d6cb3f23d20d6e176d36e6017.jpg)

![](images/dde5d6ad08badea42eaab8b9d42ba74cdf798107fab6f44fc2fdf2f2c1721ba0.jpg)

![](images/324a91d1e686b16fa14b2a40822f04fd61dc8a8f3ed1ce1e83d1cb34c3ac502f.jpg)

![](images/bd83eedf09241b2d2a53318da2f8e18df50e4381f5c557e5ec0953273c8b0d09.jpg)

![](images/bdee793a6b36ddee8be43a60fadf4e67de597c1d34d3ca5ac94624ebc71d236a.jpg)

![](images/2ab5f605c5a795c2aab3c6352b24fd2779479cdf7303f45b04103ee8a406db90.jpg)

![](images/1a2b9a88a4ccb062be302483f999d364d63fcec2f013365b5ebc6a6de4213150.jpg)

![](images/1b1d1ddfe24bba4fd344665bc2231924b7813d2bf1f1956eed930e16815f567d.jpg)

![](images/907058187d28259ea80872cb598fbfb9e9057520d644bc8938e0cd3c10c2188c.jpg)

![](images/1a3ae242c12eb1360b07f9dfb06c735fe770b970d4c77e5bfc8be5cb183a6324.jpg)

![](images/9fec514eceb741280dcee453dfd69fc37aee11f96ba049bd7d0937630fba8f26.jpg)

![](images/3fc40bcabc302c29a42ea8d0383f3e907fb8b0ee0b19908a5625f922a577330c.jpg)

![](images/428565e9b3e9ea62c5ee0debc6d45a114fa330589b02da089e5ccfc01ab2308b.jpg)

![](images/6e83c3771e45cb034f4077c5e0358518fd54475bef157cb7f9cbea76ecff9165.jpg)

![](images/886a23ce42909d0d236fca9a3c00c3734198935bfae2bbe4ce1b50838ab0ae5e.jpg)

![](images/7db7f4b4db5a10ff30df24a9a6b6ed6a0c08818c0705d4f0a868c3393f821917.jpg)

![](images/2d090935b1efe6e48be4f3d0e59cebe6af783f698b5f5332dfbb4a85c0c26483.jpg)

![](images/bccdddf57ccd24933e18fa347b17234fdafd595966d4eac86781fdcd22076b76.jpg)

![](images/9eca03308f00fc3cfa8e8b15c52d5efe37d4b0578140bb6eee4af5a785531dfd.jpg)

![](images/aeb625ee39cebdca3fa678e875fe107498c6791e185dffa4d7d0cdb9651ca69c.jpg)

![](images/cb4279ddb873dba1260b7bdebcb67fc6b8421cb9ff294bd678f2d885d4400ccd.jpg)

![](images/09fa025bf656bb0cc5cca6e77cc03ea6d5e97b03c114b7e7b03cffceaa211db9.jpg)

![](images/e3a515e527ca9208aeffbce841d22f6afe88ff9e7ab94fb4838af17b17f7f2ac.jpg)

(a) Evolution of the simulations’ prey coeficients across generations, under varying $f o v ^ { p r e y }$ . The fittest simulation is marked as a dashed line.  
![](images/2d78d5df680d27a713c3ce417468fb9dbb677089c6a3118a965293573068034a.jpg)

![](images/22fbcb372b7b5513350e308b144895e97f5b0e71dbb10f26bf7ec486435c0a79.jpg)

![](images/eb767928f34359d5c7f2f21ec152fa02eb932d6e4ddd43ae9938f3d5178fcf3d.jpg)

![](images/c031b8f3e88ab5650538341e16aee9d3ecd300fc74d3f61250b145346ea0c381.jpg)  
(b) Evolution of the simulations’ fitness across generations, under varying $f o v ^ { p r e y }$ . The mean fitness is marked as a solid line.

Figure 8: Comparison of ES convergence in terms of the simulations’ prey coeficients and fitnesses, when varying the prey’s field of view (per eye) $f o v ^ { p r e y } \in \{ \frac { 6 0 } { 2 } , \frac { 1 2 0 } { 2 } , \frac { 2 4 0 } { 2 } , \frac { 3 6 0 } { 2 } \}$ . The sensitivity analyses were run for $\mathcal { N } _ { g } = 2 0$ generations. Other parameter settings were fixed to the settings in Table 1.

![](images/ce33a97c9e0dbcf622d0bcb19c15b473a8d13202d3a412d2cc670e6c5ee91397.jpg)

![](images/76170c4fee1f7fa09c116ece31d892d6c81cf5035e93a063e8b818e8b4ffa9aa.jpg)

![](images/4619ec27cc2722bd44fda590864796333af1871532516cac6fe03bc1deb569b7.jpg)

![](images/a0ba525392bbd522daee9f186e7be9de3083cac94d72ba845c91c4e151906d00.jpg)

![](images/54ce1fad0375b3cffff4abf37c01917d03c1ac7e267549ea2e045eb8e4f0dab2.jpg)

![](images/ceb06fc72912436a76a4a5e46b3a20ce0d4b653b5a71dce00b0c7f60220e3041.jpg)

![](images/ce843511d724968933e2496d890e1b6e70d6cdcac2a94cc4d6aad26295448b82.jpg)

![](images/b5852e4f17d4eaea8c9623ea8a4a3b69fc261ff54929a9b23431567ac3be6b93.jpg)

![](images/2110d082fc160d1e0714b904c30928c69112c414f8f07bab3f4078dd8f4d4931.jpg)

![](images/55810d201848669123a0d125296110e16f342c03afa37ed98d8a8bd285c96ded.jpg)

![](images/e1d800d54ef7572ff10c19beba0500cbd407fd8ac7233aa56d0fb90e447a12e3.jpg)

![](images/5dd75308852b195634428393f66cb66a879fb0c5a56e00642b8c927e98733bfc.jpg)  
(a) Evolution of the simulations’ prey coeficients across generations, under varying $\mathcal { N } _ { s }$ . The fittest simulation is marked as a dashed line.

![](images/a38b9267d4c57f3f7a0f99c1234649fccf3757efc76e537653b0431103da48b0.jpg)  
N<sub>s</sub> = 30

![](images/6b2dc3651a35e5da5e5e4fe9a3fab7b9f5e005dc6409f0f8cc4937610a5baeca.jpg)  
N<sub>s</sub> = 50  
(b) Evolution of the simulations’ fitness across generations, under varying $\mathcal { N } _ { s }$ . The mean fitness is marked as a solid line.

Figure 9: Comparison of ES convergence in terms of the simulations’ prey coeficients and fitnesses, when varying the number of simulations per generation (population size) $\mathcal { N } _ { s } \in \{ 3 0 , 5 0 \}$ . The sensitivity analyses were run for $\mathcal { N } _ { g } = 2 0$ generations. Other parameter settings were fixed to the settings in Table 1.

![](images/dc66bb3863e8c2b3deae5340291f2cf8837e4c317a167dcd6aef4716117645ff.jpg)

![](images/f3d2afc6f5b03b7909d984ddaa34d95ac7891fb1747cdd85050a4a8a6f99e4ba.jpg)

![](images/a848578c6a5c80e2286050305236fb2534d33a4b137478c298d456d4a71641e2.jpg)

![](images/8528f2c556cfd878ad4d51b25f4bcfcf50203d53ca7634d8546b797a972b5c80.jpg)

![](images/07f38b067d7edd67c9cab190d16d1b8aabeb52ed138189f1c9b6658b0f3d1ffd.jpg)

![](images/32298fc3a9e9c53b2ef6189bd874db2f51a042e9136d65091f5be9a0da7550ad.jpg)

![](images/e8d21d63c4ea83e9d8b7b48f43a2329069951da3fa8db836c3ac8e496cf879d1.jpg)

![](images/0f2ccdd649f3e5a6cc94f196a8eb1d2ab7ee4a1494ee57d7827810a31bcd90b2.jpg)

![](images/15fbcbdb2cd7caad57d046885c64df691594ea1a592af2dd74f0295e79af56c9.jpg)

![](images/5840ce1c61ab6ad3921bbd76f255ea6d702465af5e9aaafd02e4388e447af436.jpg)

![](images/430d17ac3dfbbc2cd200b295e1c4f2393df182c91215e396efffe044409cb852.jpg)

![](images/be006802471b991bdf6c86a5b195da97a08636edddfc9e7d2f7d9b63707dd097.jpg)

![](images/913cd2baa95e29972ec681692ea05159e3d1add4ce0c70961254c272f182f6f9.jpg)

![](images/fe53c1528d1565fd2f1909bf3e32e251624580a2537aa8b30efa4735d4b81946.jpg)

![](images/88195917b14bec8d28eaa146729055651295efdd3268914df1528e969cd07f15.jpg)

![](images/f962c8d104174a0013139406817c0c518220be067d60afdc831a16f38fbbb786.jpg)

![](images/dc9cb579d74e55c3d050a7967164850520d90189f569217983f9fb8a54b6155d.jpg)

![](images/529adad1c9c3d0be12ce91c21b0f4b2b9d9bb2c88d93e01cb3d03b200b24b6c0.jpg)

(a) Evolution of the simulations’ prey coeficients across generations, under varying µ. The fittest simulation is marked as a dashed line.  
![](images/0572f072a3916babb2b0cc5555134cecd0daab6cc3dac062cf435ebee53cc891.jpg)  
µ = 0.05

![](images/3d49c8df29695f7344c5c92ca4f242b9843c4d3fde39de18bab2971ca05ec109.jpg)  
µ = 0.3

![](images/56e37e84551818a2debdb730e720ce87af37010d026b98b508fef96adf27cb9f.jpg)  
µ = 0.7  
(b) Evolution of the simulations’ fitness across generations, under varying µ. The mean fitness is marked as a solid line.

Figure 10: Comparison of ES convergence in terms of the simulations’ prey coeficients and fitnesses, when varying the mutation rate $\mu \in \{ 0 . 0 5 , 0 . 3 , 0 . 7 \}$ . The sensitivity analyses were run for $\mathcal { N } _ { g } = 2 0$ generations. Other parameter settings were fixed to the settings in Table 1.