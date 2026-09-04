# Counterfactual Routing Using Integer Programming with Constraint Generation

Daniel Vos¨ , Sterre Lutz

Delft University of Technology {S.Lutz, D.A.Vos}@tudelft.nl

## Abstract

We present our submission to the IJCAI 2025 ‘Counterfactual Routing Competition’ (CRC 25). The goal of the competition is to find counterfactual explanations for the shortest path problem. This requires deciding what the minimal changes to a road network would make a route chosen by the user the optimal route. This enables explanations such as “Your suggested route would indeed have been optimal, if road X were not a bicycle path.” Our solution models the problem as an integer program, iteratively incorporating constraints until an exact solution is found. In the final evaluation on heldout test instances, our method ranked fourth in solution quality and obtained its solution fastest on every instance, with an average runtime of 9.0 seconds compared to 118.8 seconds for the next-fastest submission.

Code https://github.com/SUMI-lab/CRC-competition

## 1 Introduction

We present our solution to the IJCAI 2025 ‘Counterfactual Routing Competition’ (CRC 25). The competition addresses the challenge of safe and accessible navigation for wheelchair users, who face unique obstacles beyond simply reaching their destination quickly. Factors such as curb heights, sidewalk widths, and crossings significantly impact which routes are suitable.

While existing routing algorithms can generate optimal paths considering these constraints, they often do not explain why a particular route is chosen over others. The competition organizers emphasize that this lack of transparency can undermine users’ trust in the recommended routes.

To address this, the competition requires participants to create counterfactual explanations that demonstrate why a chosen ‘foil’ route is suboptimal. Such explanations show how an alternative route would have been optimal if some road conditions had been different. Providing counterfactuals for routing requires identifying minimal changes in complex, weighted graphs representing the road maps.

Our solution models the problem as an integer program that iteratively incorporates constraints to ensure the counterfactual route becomes optimal after minimal modifications to the road map in question.

Competition results In the final competition evaluation<sup>1</sup> on the held-out test instances, our submission ranked fourth in terms of solution quality. Notably, our method was the fastest on every single test instance, with an average runtime of 9.0 seconds compared to 118.8 seconds for the next-fastest submission.

Related Work We did not directly base our solution on other work; however, we applied the general idea of cutting plane methods for integer programs to the competition (Bradley et al. [1977], Chapter 9.8). This approach is useful, for example, when an (integer) linear program is too large to solve directly. Instead of solving the whole program, cuts are added iteratively until optimality has been proven.

Our use of integer programming was partly motivated by Kantchelian et al. [2016]. They use an integer program to find optimal adversarial examples, which is nearly equivalent to finding optimal counterfactual explanations, i.e., they minimize distance while ensuring a specific outcome.

## 2 Problem Description

We are given a segment of the road map of the city of Amsterdam, and a user-defined ‘foil’ route. The problem to solve is: make minimal changes to the map such that the user’s route becomes the shortest one from point A to point B.

Formally, the map of the neighborhood is defined as the multi-graph $\mathcal { G } = \{ \bar { \mathcal { V } } , \mathcal { E } \}$ , where V denotes the set of nodes (i.e., locations), and E denotes the set of edges (i.e., roads or paths connecting these locations). Each edge contains some attributes such as its length, width, curb height and path type. The user’s foil route is a path of edges $r ^ { t }$ , and the user’s preferences are captured by user model U. The high-level optimization problem given a route planner $\pi _ { U } ^ { r }$ under the user model U is:

$$
\begin{array} { r l } { \underset { \mathcal { G } ^ { c } } { \operatorname* { m i n } } } & { { } d _ { g } ( \mathcal { G } , \mathcal { G } ^ { c } ) } \\ { \mathrm { s . t . } } & { { } d _ { r } \left( r ^ { t } , \pi _ { U } ^ { r } ( \mathcal { G } ^ { c } ) \right) \leq \delta , } \end{array}
$$

here $\mathcal G ^ { c }$ is the desired counterfactual map (output), $d _ { g }$ represents the graph difference, $\pi _ { U } ^ { r } ( { \mathcal { G } } ^ { c } )$ is the route returned by

the planner for the given user model and the counterfactual map, $d _ { r }$ denotes the route difference, and δ is a predefined threshold fixed to 0.05 in this competition.

## 2.1 Concrete Optimization Problem

The choice of graph distance function $d _ { g }$ and route distance function $d _ { r }$ greatly affects the tractability of the optimization problem. Therefore, it is important to specify the problem with concrete distances and objectives.

In this competition, the planner $\pi _ { U } ^ { r }$ simply finds the minimum weight path between point A and B using the Dijkstra algorithm. The weight of each edge $e \in { \mathcal { E } }$ is determined as:

$$
w _ { e } = \mathrm { l e n g t h } _ { e } \times U _ { \mathrm { c r o s s i n g } } ^ { \mathbb { I } [ e \mathrm { { i s } \mathrm { { c r o s s i n g } } ] } } \times U _ { \mathrm { p r e f e r e n c e } } ^ { \mathbb { I } [ e _ { \mathrm { t y p e } } = U _ { \mathrm { p r e f e r e n c e } } ] } ,\tag{1}
$$

with typical values for the weights being $U _ { \mathrm { c r o s s i n g } } = 1 . $ 4 and $U _ { \mathrm { p r e f e r e n c e : b o n u s } } = 0 . 6$ . There is another important requirement for edges, that is, a user will only take them if both: their minimum width is greater than or equal to $U _ { \mathrm { m i n ~ w i d t h } }$ and their maximum curb height is less than or equal to $U _ { \mathrm { m a x \thinspace c u r b } }$ . This means that, effectively, the weight of an edge can be thought of as infinite if these predicates do not hold.

The graph distance $\begin{array} { r } { d _ { g } \ = \ | \mathcal { O P } | _ { \mathcal { G } } ^ { \mathcal { G } ^ { \prime } } } \end{array}$ is measured by the amount of operations $( | \mathcal { O P } | )$ that are needed to transform G into $\mathcal { G } ^ { \prime }$ . Here, the allowed operations for each edge e are:

• Change the type of e

• Change the minimum width of e

• Change the maximum curb height of e

These operations modify the weights of the edges, which in turn can impact the minimum-weight path in the graph.

Finally, the route distance function is given by the weighted common edge rate

$$
d _ { r } ( r , r ^ { \prime } ) = 1 - 2 \frac { \sum _ { e \in r \cap r ^ { \prime } } \mathrm { l e n g t h } _ { e } } { \sum _ { e \in r } \mathrm { l e n g t h } _ { e } + \sum _ { e \in r ^ { \prime } } \mathrm { l e n g t h } _ { e } } ,
$$

which intuitively measures the lengths of the edges on which the paths overlap.

While the competition allows a slack of 0.05 in the route distance $d _ { r }$ between the shortest path in the counterfactual graph $\mathcal { G } ^ { \prime }$ and the user’s foil route, we disallow any slack in our approach. This choice allows for a simpler tractable formulation and can be made equivalent to the optimization problem with slack by enumerating all paths r such that $d _ { r } ( r ^ { \bar { t } } , r ^ { \prime } ) \leq 0 . 0 5$ (see Section 5). The concrete optimization problem that we solve is:

$$
\begin{array} { r l r } { \underset { \mathcal { G } ^ { c } } { \operatorname* { m i n } } } & { | \mathcal { O P } | _ { \mathcal { G } } ^ { \mathcal { G } ^ { c } } } & { \triangleright \mathrm { m i n i m i z e ~ n u m b e r ~ o f ~ c h a n g e s } } \\ { \mathrm { s . t . } } & { \pi _ { U } ^ { r } ( \mathcal { G } ^ { c } ) = r _ { t } . } & { \triangleright \mathrm { s . t . ~ t h e ~ u s e r ~ r o u t e ~ i s ~ s h o r t e s t } } \end{array}
$$

## 3 Method Description

At a high level, we will solve the problem using an integer program to which we iteratively add constraints during the solve via callbacks. For our implementation, we chose to use GUROBI<sup>2</sup> 12.0.2, but any integer programming solver that allows for constraint generation can be used.

Let us first introduce the integer program without constraint generation. Our formulation solves the problem at a slightly abstract, more intuitive level by reasoning over the effects of the operations $\mathcal { O P }$ that we are allowed to perform. Specifically, the operations allow us to make the following changes to an edge e of the input graph G:

• Decrease or increase the edge weight by a factor U<sub>preference bonus</sub>, depending on the edge type.

• Disable an enabled edge by decreasing the width of a segment or increasing the maximum curb height. Sometimes this operation is not possible.

• Enable a disabled edge by increasing the width of a segment or reducing the maximum curb height. Sometimes both operations are necessary.

To this end, we introduce three binary variables for each edge $e \in \mathcal { E } , \mathbf { z } _ { e } ^ { \mathrm { c o s t } } , \mathbf { z } _ { e } ^ { \mathrm { d i s a b l e } }$ and ${ \bf z } _ { e } ^ { \mathrm { e n a b l e } }$ , indicating whether to change the weight, enable the edge, or disable the edge. For clarity, we will use bold symbols to denote decision variables and regular symbols for constants. Whenever it is not possible to enable or disable an edge e, we constrain $\mathbf { z } ^ { \mathrm { { e n a b l e } ^ { \star } } } = 0$ or $\mathbf { z } ^ { \mathrm { d i s a b l e } } = 0$

The objective is to minimize the sum of the z variables since these correspond to the number of operations |OP| that we make to the input graph. It is important to note, however, that the mapping between $\sum \mathbf z$ and $| \mathcal { O P } |$ is not exactly oneto-one as it might be necessary to change both the height of a curb and the width of a road segment to enable the user to take it (see Section 2.1). Therefore, we introduce constants $C _ { e }$ that hold the number of operations needed to enable an edge. Disabling an edge or changing its weight can always be done in one operation. The resulting objective function is:

$$
\operatorname* { m i n } _ { \mathbf { z } } \sum _ { e } \mathbf { z } _ { e } ^ { \mathrm { c o s t } } + \mathbf { z } _ { e } ^ { \mathrm { d i s a b l e } } + C _ { e } \mathbf { z } _ { e } ^ { \mathrm { e n a b l e } } .
$$

Next, we must constrain the model to only allow values for z that together ensure the user’s foil route is the path with the least weight in the graph. We introduce the notation $\mathcal { P } _ { \mathcal { G } } ( s , t )$ to denote the set of all simple paths in the graph $\mathcal { G }$ starting in node s and ending in node t. Now, to constrain the foil path $P _ { \mathrm { f o i l } }$ to be the one with least weight is equivalent to forcing it to be weighted (strictly) less than any other simple path:

$$
W ( P _ { \mathrm { f o i l } } , \mathbf { z } ) \leq W ( P , \mathbf { z } ) - \epsilon \quad \forall P \in \mathcal { P } _ { \mathcal { G } } ( s , t ) \backslash \{ P _ { \mathrm { f o i l } } \} ,
$$

where $W ( P ^ { \prime } , { \bf z } )$ is the weight of a path $P ^ { \prime }$ given the graph with changes induced by z, and ϵ is a small constant to enforce strict inequality. Notice that $W ( P ^ { \prime } , { \bf z } )$ can be written as a linear expression:

$$
W ( P , \mathbf { z } ) = \sum _ { e \in P } w _ { e } + \Delta _ { e } z _ { e } ^ { \mathrm { c o s t } } + M z _ { e } ^ { \mathrm { d i s a b l e } } + M ( 1 - z _ { e } ^ { \mathrm { e n a b l e } } ) ,
$$

where $w _ { e }$ refers to the weight of edge e without changes to the graph (Equation $1 ) , \Delta _ { e }$ is the weight change incurred by changing the type of $e ,$ and $M$ is a large constant. It is important to choose M large enough to trivially satisfy the constraints when the appropriate edges are disabled/enabled, but as small as possible to prevent numerical issues in the solver. We select $\begin{array} { r } { \dot { M } = \epsilon + \frac { \dot { \mathrm {  ~ { \scriptstyle ~ 1 } ~ } } } { U _ { \mathrm { p r e f e r e n c e b o n u s } } } \sum _ { e \in P _ { \mathrm { f o i l } } } w _ { e } } \end{array}$ , which ensures that any path with an edge of weight M is larger than the weight of foil path $P _ { \mathrm { f o i l } }$ even if all its edges were changed into the user’s unpreferred type.

![](images/2ba7aa5a5ede584d06969d42bd444a319c5f078d5e9fa6efdf46798e14e13437.jpg)  
Figure 1: Flowchart showing the iterative process of the proposed constraint generation method. We start with an empty formulation and add shortest path constraints until the user’s foil route becomes the shortest one.

Since both our constraints and objective are linear, the combined program is an exponentially-sized integer linear program (ILP) that solves the problem:

$$
\begin{array} { r l } {  { \frac { \mathrm { m i n } } { \mathbf { z } } \sum _ { e } \mathbf { z } _ { e } ^ { \mathrm { c o s t } } + \mathbf { z } _ { e } ^ { \mathrm { d i s a b l e } } + C _ { e } \mathbf { z } _ { e } ^ { \mathrm { e n a b l e } } } } \\ { \mathrm { s . t . } W ( P _ { \mathrm { f o i l } } , \mathbf { z } ) \leq W ( P , \mathbf { z } ) - \epsilon \forall P \in \mathcal { P } _ { \mathcal { G } } ( s , t ) \backslash \{ P _ { \mathrm { f o i l } } \} } \\ & { \mathbf { z } _ { e } ^ { \mathrm { c o s t } } , \mathbf { z } _ { e } ^ { \mathrm { d i s a b l e } } , \mathbf { z } _ { e } ^ { \mathrm { e n a b l e } } \in \{ 0 , 1 \} \forall e \in \mathcal { E } . } \end{array}
$$

## 3.1 Constraint Generation

While the aforementioned ILP solves the problem, its formulation is potentially exponential in size due to the enumeration of all simple paths between the source and target nodes $\mathcal { P } _ { \mathcal { G } } ( s , t )$ . To arrive at a tractable formulation, we only add the necessary constraints via a constraint generation process. A schematic of this process is given in Fig. 1.

Such a constraint generation process is implemented via solver callbacks that are triggered when the solver finds an integer assignment to the variables z that satisfy the current set of constraints. In each callback we then either add a constraint that invalidates the proposed solution and continue solving, or accept the proposed solution. To identify the violating constraint we simply apply the operations to input graph G as indicated by values of z and solve a shortest path problem using the polynomial-time Dijkstra algorithm. Let $\mathbf { \hat { { \mathcal { G } } ^ { z } } }$ denote the input graph with changes applied as indicated by the values of z, then we identify the least-weight path:

$$
P ^ { * } = \arg \operatorname* { m i n } _ { P \in \mathcal { P } _ { \mathcal { G } ^ { \mathbf { z } } } ( s , t ) } \sum _ { e \in P } w _ { e } .
$$

If $P ^ { * } = P _ { \mathrm { f o i l } }$ then we know $P _ { \mathrm { f o i l } }$ is the shortest route in the graph and therefore the problem is solved. Instead, if $P ^ { * } \neq$ $P _ { \mathrm { f o i l } }$ , we add the constraint

$$
W ( P _ { \mathrm { f o i l } } , \mathbf { z } ) \leq W ( P ^ { * } , \mathbf { z } ) - \epsilon ,
$$

and continue solving. In practice, we only need to add a relatively small number of constraints to the optimization model before the solver finds the optimal solution. This is because the user’s foil routes are not much greater in weight than the optimal shortest routes in the competition instances. Instances where the user’s route differs significantly in weight from the shortest route will take substantially longer to solve.

## 4 Results

To validate the tractability of our method, we evaluated it on the problem instances provided by the competition, consisting of five different maps with five foil routes each. All experiments were conducted on a Windows 11 system with a 13th Gen Intel Core i7-1365U processor and 16 GB RAM. The runtime results on these 25 problem instances are given in Table 1.

While the time spent per instance varies between the instances, we were able to solve all of them within 10 seconds. Depending on the size of the map, our method spends 0.5 to 2 seconds setting up the problem, and the remaining time is spent in the solver. The slowest instance to solve was Map 1 with Route 4, where our method spent a total of roughly 8 seconds. Most of the time for this instance was spent in the callback function that makes temporary changes to the graph and solves a shortest route problem.

Competition results In the final competition evaluation on the held-out test instances, our submission ranked fourth in terms of solution quality. Notably, our method was the fastest on every single test instance, with an average runtime of 9.0 seconds compared to 118.8 seconds for the next-fastest submission.

## 5 Future Work

Our method solves a restriction of the real problem to optimality: finding a minimal counterfactual explanation for how the user’s route becomes the shortest route. The com petition’s real problem allowed for some slack between the shortest path and the user’s route, however. Therefore, we recommend that our method be extended with a search over target routes in the future. This means that one should enumerate paths $r _ { \mathrm { t a r g e t } } ^ { t }$ close to the user’s foil path (they should satisfy ${ d _ { r } ( r ^ { t } , r _ { \mathrm { t a r g e t } } ^ { t } ) } ) \leq \delta )$ , and use our method to find a minimal counterfactual for them. After enumerating, the smallest counterfactual encountered can be returned. We believe this to be a tractable approach because the solver can be warmstarted after the initial solve, and previously added constraints can be memorized, which will improve solving time.

## References

S.P. Bradley, A.C. Hax, and T.L. Magnanti. Applied Mathematical Programming. Addison-Wesley Publishing Company, 1977.

<table><tr><td colspan="2">Instance</td><td colspan="3">Time spent (s)</td><td></td></tr><tr><td>map</td><td>route</td><td>total</td><td>solving</td><td>in callback</td><td># callbacks</td></tr><tr><td>0</td><td>1</td><td>3.94</td><td>1.78</td><td>0.37</td><td>134</td></tr><tr><td></td><td>2</td><td>4.09</td><td>1.96</td><td>0.51</td><td>167</td></tr><tr><td></td><td>3</td><td>6.52</td><td>4.39</td><td>2.37</td><td>330</td></tr><tr><td></td><td>4</td><td>5.07</td><td>2.89</td><td>1.19</td><td>192</td></tr><tr><td></td><td>5</td><td>4.97</td><td>2.74</td><td>0.86</td><td>179</td></tr><tr><td>1</td><td>1</td><td>1.23</td><td>0.78</td><td>0.22</td><td>177</td></tr><tr><td></td><td>2</td><td>1.01</td><td>0.44</td><td>0.12</td><td>127</td></tr><tr><td></td><td>3</td><td>0.96</td><td>0.44</td><td>0.13</td><td>138</td></tr><tr><td></td><td>4</td><td>7.88</td><td>7.35</td><td>4.59</td><td>8,879</td></tr><tr><td></td><td>5</td><td>0.92</td><td>0.39</td><td>0.07</td><td>137</td></tr><tr><td>2</td><td>1</td><td>1.42</td><td>0.62</td><td>0.10</td><td>128</td></tr><tr><td></td><td>2</td><td>1.55</td><td>0.87</td><td>0.44</td><td>151</td></tr><tr><td></td><td>3</td><td>1.28</td><td>0.58</td><td>0.16</td><td>142</td></tr><tr><td></td><td>4</td><td>1.57</td><td>0.88</td><td>0.27</td><td>154</td></tr><tr><td></td><td>5</td><td>3.44</td><td>2.65</td><td>1.09</td><td>5,688</td></tr><tr><td>3</td><td>1</td><td>1.06</td><td>0.50</td><td>0.08</td><td>122</td></tr><tr><td></td><td>2</td><td>1.08</td><td>0.52</td><td>0.09</td><td>139</td></tr><tr><td></td><td>3</td><td>0.93</td><td>0.36</td><td>0.05</td><td>121</td></tr><tr><td></td><td>4</td><td>1.15</td><td>0.61</td><td>0.27</td><td>180</td></tr><tr><td></td><td>5</td><td>1.59</td><td>1.05</td><td>0.41</td><td>1,585</td></tr><tr><td>4</td><td>1</td><td>1.37</td><td>1.00</td><td>0.54</td><td>483</td></tr><tr><td></td><td>2</td><td>2.73</td><td>2.27</td><td>0.96</td><td>10,832</td></tr><tr><td></td><td>3</td><td>0.98</td><td>0.45</td><td>0.16</td><td>162</td></tr><tr><td></td><td>4</td><td>0.78</td><td>0.31</td><td>0.06</td><td>126</td></tr><tr><td></td><td>5</td><td>0.89</td><td>0.41</td><td>0.08</td><td>127</td></tr></table>

Table 1: Breakdown of time spent on each instance from the competition’s train set and the number of callbacks performed (equivalently, the number of constraints added) before proving optimality. Our method solves each instance within 10 seconds.

Alex Kantchelian, J. D. Tygar, and Anthony D. Joseph. Evasion and hardening of tree ensemble classifiers. In Maria-Florina Balcan and Kilian Q. Weinberger, editors, Proceedings of the 33rd International Conference on Machine Learning, ICML 2016, New York City, NY, USA, June 19- 24, 2016, volume 48 of JMLR Workshop and Conference Proceedings, pages 2387–2396. JMLR.org, 2016.