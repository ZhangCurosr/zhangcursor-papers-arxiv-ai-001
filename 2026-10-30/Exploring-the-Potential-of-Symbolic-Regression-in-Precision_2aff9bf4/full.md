# Exploring the Potential of Symbolic Regression in Precision Livestock Farming

E. Mayrhuber<sup>1,2</sup>, M. Oczak<sup>3,4</sup>, M. Iwersen<sup>5</sup>, D. Brunner<sup>1,2</sup>, V. Dorfer<sup>1</sup>, S. Winkler<sup>1</sup>

<sup>1</sup>Bioinformatics Research Group, PLFDoc, University of Applied Sciences Upper Austria, Hagenberg, Austria

<sup>2</sup>Computer Vision Lab, TU Wien, Vienna, Austria

<sup>3</sup>Precision Livestock Farming Hub, The University of Veterinary Medicine Vienna, Vienna, Austria

<sup>4</sup>Center for Animal Nutrition and Welfare, The University of Veterinary Medicine Vienna, Vienna, Austria

<sup>5</sup>Center for Veterinary Systems Transformation and Sustainability, Clinical Unit for Herd Health Management, The University of Veterinary Medicine Vienna, Vienna, Austria

elisabeth.mayrhuber@fh-hagenberg.at

## Abstract

The ability to predict farrowing time in pigs is crucial for optimal farm management and ensuring piglet survival. This study investigates the potential of explainable artificial intelligence for precision livestock farming, focusing on acceleration sensor data from sows around the time of farrowing. Accurately predicting farrowing allows farmers to balance both, piglet safety by avoiding piglet crushing events and the mother sow’s welfare, by avoiding early crating. Traditional machine learning models often lack interpretability (“closed-box” problem). Explainable artificial intelligence ofers a solution by revealing the decision-making processes within the models. This work is split into two stages. First the detection of nest building behaviour and second the prediction of the remaining time to farrowing. The nest building detection was possible for 166 out of 180 sows and successful within 48 hours before farrowing for 149 sows. The nest building detection is purely based on the change of a sow’s activity level and uses no machine learning techniques. Further, this study utilizes symbolic regression, an open-box modelling technique that generates interpretable mathematical formulas, to predict the remaining time until the onset of farrowing. The results are compared with closedbox models, like support vector regression and artificial neural networks, for both explainability and performance. Notably, symbolic regression achieved a mean absolute error of 9.28 hours for the time-to-farrowing prediction, outperforming the closed-box models in terms of performance and interpretability. This research demonstrates that explainable artificial

intelligence, specifically symbolic regression, can ofer both interpretable results and competitive performance in precision livestock farming.

Keywords: explainable artificial intelligence, farrowing prediction, precision lifestock farming, symbolic regression

## 1 Introduction

Sows are routinely confined in crates a few days before giving birth (farrowing) mainly to prevent piglets from being crushed. While this practice protects piglets [King et al., 2019], it comes at the cost of increased stress and restricted natural behaviours for the mother sow [Hansen et al., 2017, Heidinger et al., 2022, Jarvis et al., 2006]. To balance piglet safety and sow wellbeing, modern farms have started to use temporary crating, delaying confinement previous to nest building and ideally at least 24 hours before farrowing [Cheon et al., 2022]. However, pinpointing the optimal time for crating to maximize welfare for both sows and piglets remains a challenge [Oczak et al., 2019]. This study proposes a two-step alarm system using explainable artificial intelligence (XAI) to address this challenge. The system will first detect the onset of nest building behaviour and then predict the time remaining until birth. We focus on symbolic regression (SR), an open-box machine learning (ML) technique within XAI. Unlike traditional “closed-box” models like artificial neural networks (ANNs) [Wang, 2003], SR generates interpretable mathematical formulas, leveraging genetic programming [Afenzeller et al., 2009]. These formulas reveal the relationships between sensor data and the predicted time to farrowing, providing valuable insights for precision livestock farming (PLF) [Hoxhallari et al., 2022, Najjar et al., 2023]. Genetic Programming has already been tested in other areas of PLF, such as beef cattle management [Abbona et al., 2020]. Building upon previous farrowing prediction research [Oczak et al., 2019], this work aims to improve prediction accuracy by leveraging XAI with data from sow ear tag accelerometers. This approach diferentiates itself from recent studies that rely on computer vision (CV) for farrowing prediction [Chen et al., 2023, Oczak et al., 2023].

## 2 Material and Methods

## 2.1 Experimental data

The study used data from 180 sows equipped with SMARTBOW<sup>®</sup> ear $\mathrm { { t a g s } ^ { 1 } }$ (Smartbow/Zoetis LLC, Weibern, Austria). These tags contain tri-axial accelerometers that capture sow movement along three axes (x, y, and z). Details about the SMARTBOW<sup>®</sup> system can be found in [Oczak et al., 2016]. As dataset, the magnitude of acceleration, calculated as the square root of the sum of squared values from each axis $( x ^ { 2 } + y ^ { 2 } + z ^ { 2 } )$ , was used. This single value represents the overall movement intensity of the sow. The dataset was split into training and test sets with 144 sows (80%) and 36 sows (20%), respectively (Table 1). This ensured an even distribution of data from diferent pen types (SWAP, trapezoid, and wing) across the test set. Evaluating the model’s performance on the unseen test set allows us to assess its generalizability to new data.

Table 1: Number of sows used training and test split based on pen types.
<table><tr><td>Pen type</td><td>Training</td><td>Test</td></tr><tr><td>Wing</td><td>48</td><td>12</td></tr><tr><td>SWAP</td><td>44</td><td>12</td></tr><tr><td>Trapezoid</td><td>52</td><td>12</td></tr><tr><td>Total</td><td>144</td><td>36</td></tr></table>

## 2.2 Data Preprocessing and Feature Engineering

The data preprocessing is diferent for the two tasks tackled in this research. The nest building detection works with rolling averages over the signal known so far. The averages that were calculated for this step are the rolling average over 12 hours, 24 hours and 48 hours. The 12-hour rolling average aims to show the day and night cycles of the sows, while the 24-hour and 48-hour windows aim to identify a general activity level for each sow and reveal larger changes in their activity. For the time-to-farrowing prediction, six features were extracted. These features include the sow’s average daytime and nighttime activity, as well as the average activity over 24 and 48 hours during periods of normal day-night cycles. These values were calculated using the first 48 hours of available data for each sow. Additionally, the average activity over the last 12 and 24 hours were calculated at each timestamp. To create the training set, five time points were established for each sow. These timestamps were: three days after introduction to the farrowing pen, the timestamp of the nest building alarm, the timestamp between the start of nest building and the start of farrowing, one hour before farrowing, and the start of farrowing. The remaining minutes until farrowing were then calculated as target variable.

## 2.3 Detection of Nest Building Behaviour

For both the nest building detection and the time-to-farrowing prediction, the arrival of new acceleration data was simulated with a sliding window approach to test the models in an online setting. For every sliding window step the last 12 hours and the last day were compared to the signal which was received so far. The detection is only performed 2.5 days after the sow was introduced to the farrowing pen, since shortly after the introduction the activity levels can difer due to the change of environment [Oczak et al., 2019]. If one of the two conditions below (Equation 1 and 2) evaluate to “TRUE”, the detection of nest building behaviour alarm is triggered. The variable $a _ { t }$ describes the acceleration at the time t and $a _ { t - 2 4 }$ is the acceleration 12 hours before the timestamp t. The constant values of 0.8 and 1.75 were chosen by trying out diferent combination for the training set and these achieved the best results.

$$
\mathrm { m e a n } \big ( [ a _ { t - 2 4 } ; a _ { t } ] \big ) \geq \Big ( \mathrm { m e a n } \big ( [ a _ { t _ { 0 } } ; a _ { t - 2 4 } ] \big ) + \mathrm { s t d } \big ( [ a _ { t _ { 0 } } ; a _ { t - 2 4 } ] \big ) \Big ) \cdot 0 . 8\tag{1}
$$

$$
\operatorname* { m a x } \bigl ( \bigl [ a _ { t - 1 2 } ; a _ { t } \bigr ] \bigr ) > \operatorname* { m a x } \bigl ( \bigl [ a _ { t _ { 0 } } ; a _ { t - 1 2 } \bigr ] \bigr ) \cdot 1 . 7 5\tag{2}
$$

## 2.4 Time to Farrowing Prediction

Following the initial alarm trigger, time-to-farrowing predictions were performed every 90 minutes. Various machine learning algorithms were employed to predict the remaining time to farrowing. Two open-box algorithms, SR and linear regression (LR) were compared to four closed-box algorithms: ANNs, random forest, XGBoost (XGB), and support vector regression (SVR). For closed-box models, post-hoc explainability methods would be necessary to extract information regarding the reasoning behind model predictions or the features with a high impact on the results. Examples of such methods include LIME [Ribeiro et al., 2016] and SHAP [Lundberg and Lee, 2017]. The performance of all chosen models will be discussed in the “Results and Discussion” section, while explainability will not be evaluated for closed-box models. To train the models, the training dataset, which is described above, was used. The training of all models was performed using a grouped 5-fold cross-validation to ensure robust training. Grouping the data for cross-validation is important in this case since features were extracted from five timestamps and the data from one sow should not be split during cross validation.

For every model the same folds were used to ensure comparable results. EC KitY [Sipper et al., 2023] was utilized to train SR models, while scikit-learn [Pedregosa et al., 2011] was used for all other models.

## 3 Results and Discussion

To evaluate the model performances during training a combination of the mean absolute error (MAE) and the Pearson correlation coeficient (PCC) were used. These training and validation performances can be seen in Table 2. During the training phase the models have only been evaluated on the aforementioned predefined timestamps, the performance in a real-time scenario can be found in Table 3.

To demonstrate the efectiveness of our approach, we showcase the nestbuilding detection and the time-to-farrowing detection on one sow from the test set. The evaluation of the time-to-farrowing prediction was conducted in an online setting on both the training and test sets, using the best performing model from the cross-validation.

Table 2: Training and validation performance for 5-fold cross-validation.
<table><tr><td rowspan="3">Model</td><td colspan="4">MAE [h]</td><td colspan="4">PCC</td></tr><tr><td colspan="2">Training</td><td colspan="2">Validation</td><td colspan="2">Training</td><td colspan="2">Validation</td></tr><tr><td>Mean</td><td>Std</td><td>Mean</td><td>Std</td><td>Mean</td><td>Std</td><td>Mean</td><td>Std</td></tr><tr><td>SR</td><td>17.70</td><td>1.21</td><td>22.55</td><td>2.25</td><td>0.60</td><td>0.05</td><td>0.42</td><td>0.06</td></tr><tr><td>LR</td><td>26.80</td><td>0.85</td><td>27.22</td><td>2.42</td><td>0.45</td><td>0.01</td><td>0.44</td><td>0.04</td></tr><tr><td>ANN</td><td>23.17</td><td>1.03</td><td>24.56</td><td>2.22</td><td>0.56</td><td>0.02</td><td>0.50</td><td>0.04</td></tr><tr><td>RF</td><td>8.83</td><td>0.43</td><td>24.35</td><td>2.73</td><td>0.96</td><td>0.004</td><td>0.49</td><td>0.06</td></tr><tr><td>XGB</td><td>0.64</td><td>0.10</td><td>26.01</td><td>1.66</td><td>0.99</td><td>6E-5</td><td>0.42</td><td>0.06</td></tr><tr><td>SVR</td><td>24.98</td><td>0.69</td><td>25.01</td><td>2.74</td><td>0.44</td><td>0.01</td><td>0.45</td><td>0.03</td></tr></table>

## 3.1 Detection of Nest Building Behaviour

The detection of nest building behaviour before farrowing was possible for 166 of the 180 sows in our experiments. There was no case where the detection was after the onset of farrowing, but for 14 sows, no detection could be performed. In every successful case the alarm was triggered before the start of farrowing. In 17 cases the alarm was presented at least 48h before the start of farrowing and therefore more time to farrowing predictions were needed. In most cases this was caused by an anomaly in the sow’s activity levels during normal day night cycles. Figure 1 shows the detection of nest building behaviour for the demonstration sow. This detection is robust to changes in the baseline activity levels of sows since it is only dependent on one sows’ activity.

## 3.2 Time to Farrowing Prediction

For every time to farrowing prediction in the test setting, the same six features as for the training set were extracted from the signal that is known so far from each sow. This was then performed every 90 minutes to update the prediction, as new data arrives. This timespan can be easily adapted in the prototype, for example to every 60 minutes instead. Table 3 shows quality metrics for every used model. It shows the average PCC and MAE, as well as the standard deviation for these metrics among the training and test set within this online prediction. The number of predictions difer for every sow, since the time between the detected onset of nest building and the actual time of farrowing varies a lot (6–188 hours).

Focussing on the MAE of the test sows (Table 3), SR achieves the best results, followed by SVR. A Mann-Whitney U test comparing the MAE of SR against SVR shows with a p-value of $2 . 1 8 \times 1 0 ^ { - 6 }$ that there is a significant diference between these two. This means that SR, while being inherently more interpretable, also performs significantly better than SVR.

Figure 2 shows every prediction that was performed using the SR model to predict the remaining time to farrowing. The actual prediction is done in minutes but for improved readability the values are transformed to hours. It shows that for the demonstration sow, the MAE is 1.52 hours and the PCC for this sow is 0.99, which is better than the average result.

![](images/9481c60008cdcb98f24d81d709e8c508232b53b7f5e5861abdbc137ef698d790.jpg)  
Figure 1: Detection of nest building behaviour for one sow. The dashed line (red) shows the detected start of nest building behaviour, and the highlighted area (green) shows the true time of farrowing. This detection is done online using a sliding window approach.

Table 3: Performance of diferent models in a real time setting for time to farrowing predictions.
<table><tr><td rowspan="3">Method</td><td colspan="4">MAE [h]</td><td colspan="4">PCC</td></tr><tr><td colspan="2">Training</td><td colspan="2">Test</td><td colspan="2">Training</td><td colspan="2">Test</td></tr><tr><td>Mean</td><td>Std</td><td>Mean</td><td>Std</td><td>Mean</td><td>Std</td><td>Mean</td><td>Std</td></tr><tr><td>SR</td><td>19.57</td><td>26.61</td><td>9.28</td><td>13.09</td><td>0.73</td><td>0.31</td><td>0.76</td><td>0.21</td></tr><tr><td>LR</td><td>22.05</td><td>20.59</td><td>18.04</td><td>10.51</td><td>0.89</td><td>0.22</td><td>0.90</td><td>0.29</td></tr><tr><td>ANN</td><td>21.46</td><td>25.33</td><td>17.28</td><td>14.36</td><td>0.81</td><td>0.25</td><td>0.86</td><td>0.22</td></tr><tr><td>RF</td><td>20.51</td><td>22.40</td><td>19.10</td><td>16.88</td><td>0.74</td><td>0.33</td><td>0.79</td><td>0.23</td></tr><tr><td>XGB</td><td>21.28</td><td>28.02</td><td>16.35</td><td>18.64</td><td>0.65</td><td>0.29</td><td>0.76</td><td>0.21</td></tr><tr><td>SVR</td><td>28.34</td><td>23.39</td><td>12.45</td><td>9.22</td><td>0.88</td><td>0.27</td><td>0.76</td><td>0.24</td></tr></table>

![](images/c6db2abe0c82ac72ac448c1d49ff1e2a6fb5d1dabeea819174613afaa6c14786.jpg)  
Figure 2: The points show the predicted compared to the actual time to farrowing for one sow, using symbolic regression. The line shows how a perfect prediction would look like.

In contrast to Figure 2, Figure 3 shows the predictions for the same sow but with Linear Regression and the ANN model, which is a multi-layer perceptron neural network. It shows that the MAE for both models is worse than the one for the SR model. Also, the PCC is worse than that of the SR but this is not the case for every sow, which can be seen when having a look at Table 3, where the mean PCC for linear regression is higher than the one for SR.

## 3.3 Explainability of Time to Farrowing Prediction

In XAI literature there is often a distinction between explainability and interpretability [Marcinkevičs and Vogt, 2023]. Explainability means that a model’s decision can be explained after it has been made, which is also called post-hoc explainability. On the other hand, interpretability means that the internals of a model can be interpreted and therefore decisions can be understood with no need of post-hoc explainability methods. Naturally SR builds interpretable models since the model is provided as an equation. The best resulting SR model of this research will not be provided in this paper, since the complexity of the model is too high to display eficiently. The used model has a total number of 2917 nodes when displayed as a tree. To show the structure of such models, Equation 3 shows a smaller SR model, where f(t) is the remaining time to farrowing in minutes. The shown model has a significantly worse performance than the model which was used in the results above.

$$
f ( t ) = 4 6 8 3 . 5 + \left( \frac { 0 . 4 6 7 \cdot \mathrm { m e a n } \big ( [ a _ { t - 2 4 } ; a _ { t } ] \big ) } { - 0 . 0 7 6 \cdot \mathrm { m e a n } \big ( [ a _ { t 0 } ; a _ { t - 2 4 } ] \big ) } \right) \cdot 3 0 6\tag{3}
$$

The two features included in this model are the mean magnitude of acceleration of the last 24 hours $( \mathrm { m e a n } ( [ a _ { t - 2 4 } ; a _ { t } ] ) )$ and second, the average of the rolling mean over 24 hours except the last 24 hours $( \mathrm { m e a n } ( [ a _ { t _ { 0 } } ; a _ { t - 2 4 } ] ) )$ .

![](images/165d0d9e76185fb3f18882357c4c65d7b36f5b94819594a215662558ab759719.jpg)

![](images/dc08946efeb8837908a9e6ece648409f24c0856f84254e2fec61b8db527a2578.jpg)  
Figure 3: Both images show the performance of the time to farrowing prediction for the same sow as in Figure 2. The left figure (a) shows the prediction made by the linear regression model and the right figure (b) shows the prediction of the artificial neural network.

## 4 Conclusions

For 166 out of 180 sows it was possible to provide an alarm at the beginning of nest building behaviour or shortly before. Compared to closed-box models like ANNs used in this study, SR ofered a clear advantage in performance and interpretability. While achieving similar or even better performance in some cases, like it can be seen in the example above, ANNs lack the ability to interpret their decision-making process. This hinders user trust and limits the potential for knowledge discovery from the model. In conclusion the presented approach provides a robust, two stage, online farrowing prediction based on ear tag acceleration data. It consists of a first stage alarm at the beginning of nest building behaviour and from then on, an interpretable time to farrowing prediction, which is updated in a chosen interval. Since the models are based on the diferences between the change of activities of individual sows the predictions are robust to changes in the baseline activity of sows, like it can be the case on diferent farms. While interpretable AI methods like SR provide valuable insights into a model’s decisions, in its current form this mostly benefits ML experts. In future work, we aim to further enhance the interpretability of these methods for non-technicians like veterinarians.

## Acknowledgements

This research was funded in whole or in part by the Austrian Science Fund (FWF) [https://doi.org/10.55776/DFH34]. For open access purposes, the author has applied a CC BY public copyright license to any author-accepted manuscript version arising from this submission.

## References

F. Abbona, L. Vanneschi, M. Bona, and M. Giacobini. Towards modelling beef cattle management with genetic programming. Livestock Science, 241:104205, 2020. doi: 10.1016/j.livsci.2020.104205.

M. Afenzeller, S. Winkler, S. Wagner, and A. Beham. Genetic Algorithms and Genetic Programming: Modern Concepts and Practical Applications. Chapman & Hall/CRC, 2009.

J. Chen, J. Zhou, L. Liu, C. Shu, M. Shen, and W. Yao. Sow farrowing early warning and supervision for embedded board implementations. Sensors, 23 (2):727, 2023. doi: 10.3390/s23020727.

S. N. Cheon, S. H. Jeong, G. Z. Yoo, S. J. Lim, C. H. Kim, G. W. Jang, and J. H. Jeon. Efect of alternative farrowing pens with temporary crating on the performance of lactating sows and their litters. Journal of Animal Science and Technology, 64(3):574–587, 2022. doi: 10.5187/jast.2022.e36.

C. F. Hansen, J. Hales, P. M. Weber, S. A. Edwards, and V. A. Moustsen. Confinement of sows 24h before expected farrowing afects the performance of nest building behaviours but not progress of parturition. Applied Animal Behaviour Science, 188:1–8, 2017. doi: 10.1016/j.applanim.2017.01.003.

B. Heidinger, K. Maschat, S. Kuchling, L. Hochfellner, C. Winckler, J. Baumgartner, and C. Leeb. Short confinement of sows after farrowing, but not pen type afects live-born piglet mortality. Animal, 16(2):100446, 2022. doi: 10.1016/j.animal.2021.100446.

K. Hoxhallari, W. Purcell, and T. Neubauer. The potential of explainable artificial intelligence in precision livestock farming, 2022.

S. Jarvis, R. B. D’Eath, S. K. Robson, and A. B. Lawrence. The efect of confinement during lactation on the hypothalamic–pituitary–adrenal axis and behaviour of primiparous sows. Physiology & Behavior, 87(2):345–352, 2006. doi: 10.1016/j.physbeh.2005.10.004.

R. L. King, E. M. Baxter, S. M. Matheson, and S. A. Edwards. Temporary crate opening procedure afects immediate post-opening piglet mortality and sow behaviour. Animal, 13(1):189–197, 2019. doi: 10.1017/S1751731118000915.

S. M. Lundberg and S.-I. Lee. A unified approach to interpreting model predictions. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper\_files/paper/2017/file/ 8a20a8621978632d76c43dfd28b67767-Paper.pdf.

R. Marcinkevičs and J. E. Vogt. Interpretable and explainable machine learning: A methods-centric overview with concrete examples. WIREs Data Mining and Knowledge Discovery, 13(3):e1493, 2023. doi: 10.1002/widm.1493.

H. Najjar, P. Helber, B. Bischke, P. Habelitz, C. Sanchez, F. Mena, M. Miranda, D. Pathak, J. Siddamsetty, D. Arenas, M. Vollmer, M. Charfuelan, M. Nuske, and A. Dengel. Feature attribution methods for multivariate time-series explainability in remote sensing. In IGARSS 2023 – 2023 IEEE International Geoscience and Remote Sensing Symposium, pages 5014–5017, 2023. doi: 10.1109/IGARSS52108.2023.10282120.

M. Oczak, K. Maschat, D. Berckmans, E. Vranken, and J. Baumgartner. Can an automated labelling method based on accelerometer data replace a human labeller? – postural profile of farrowing sows. Computers and Electronics in Agriculture, 127:168–175, 2016. doi: 10.1016/j.compag.2016.06.013.

M. Oczak, K. Maschat, J. Baumgartner, B. O’Brien, D. Hennessy, and L. Shalloo. Monitoring of approaching farrowing in pens with possibility of temporary crating on the basis of ear tag acceleration data. In TBD: proceedings title (ECPLF 2019?), pages 322–329, 2019.

M. Oczak, K. Maschat, and J. Baumgartner. Implementation of computervision-based farrowing prediction in pens with temporary sow confinement. Veterinary Sciences, 10(2):109, 2023. doi: 10.3390/vetsci10020109.

F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and É. Duchesnay. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12(85): 2825–2830, 2011.

M. T. Ribeiro, S. Singh, and C. Guestrin. “Why should I trust you?”: Explaining the predictions of any classifier, 2016.

M. Sipper, T. Halperin, I. Tzruia, and A. Elyasaf. EC-KitY: Evolutionary computation tool kit in Python with seamless machine learning integration. SoftwareX, 22:101381, 2023. doi: 10.1016/j.softx.2023.101381.

S.-C. Wang. Artificial neural network. In Interdisciplinary Computing in Java Programming, pages 81–100. Springer US, 2003. doi: 10.1007/ 978-1-4615-0377-4\_5.