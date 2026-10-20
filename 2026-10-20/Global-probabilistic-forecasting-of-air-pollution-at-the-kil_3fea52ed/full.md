# Global probabilistic forecasting of air pollution at the kilometre scale

Ammar Kheder<sup>1,2,3,∗</sup> · Michael Boy<sup>1,2,3</sup> · Zhi-Song Liu<sup>1,2,3</sup>

<sup>1</sup>Department of Computational Engineering, LUT University, Lahti, Finland

<sup>2</sup>Institute for Atmospheric and Earth System Research, University of Helsinki, Helsinki, Finland

<sup>3</sup>Atmospheric Modelling Centre Lahti (AMC-Lahti), Lahti, Finland

<sup>∗</sup>Corresponding author: ammar.kheder@lut.fi

## Abstract

Fine particulate matter (PM ) refers to particulate matter with an aerodynamic diameter of 2.5 µm or less. It comes from human activities, such as traffic and industry, and from wildfires, desert dust and gases emitted by vegetation<sup>1</sup>. Short-term exposure to it accounts for about a million deaths a year<sup>2</sup>. Operational forecasts run on grids tens of kilometres wide <sup>3</sup>, which hide the pollution breathed by nearly two hundred million people a day. Machine learning has transformed weather forecasting <sup>4,5</sup>, but data-driven forecasts of air pollution still blur extremes and lose skill as the air travels. Here we introduce Foehn, a generative model that learns from global kilometre-scale estimates of PM<sub>2.5</sub> <sup>6</sup> and computes where the air comes from. Over a year of retrospective forecasts, it outperforms the operational forecast at three stations in four and forecasts at the scale of this hidden exposure. Its ensembles are calibrated, and their mean detects local pollution extremes four days ahead as well as the operational forecast does one day ahead. Such forecasts could warn the millions whom coarse maps miss, once kilometre-scale data are available in real time.

In January 2022, haze from human activities covered the North China Plain for a week, and in July 2022 smoke from forest fires in south-western France drifted over a bird reserve. Such episodes matter because fine particulate matter (PM<sub>2.5</sub>), particles with an aerodynamic diameter of 2.5 µm or less, is one of the leading risk factors for premature death worldwide<sup>7</sup>. It is emitted directly or formed in the air from gases released by human activities, such as traffic, industry, power generation, domestic heating and agriculture, and by natural sources, such as wildfires, desert dust and vegetation, whose volatile organic compounds form secondary organic aerosol <sup>1,8</sup>. Short-term exposure, over hours to days, accounts for about a million deaths a year <sup>2</sup>. On this timescale, forecasts and warnings allow people to protect themselves, for example by limiting time outdoors or avoiding polluted neighbourhoods. The harm also extends beyond people: wildfire smoke causes respiratory distress and changes in behaviour in wild animals, and birds repeatedly exposed to smoke lose weight <sup>9,10</sup>. Global operational forecasts, such as those of the Copernicus Atmosphere Monitoring Service (CAMS) <sup>3</sup>, come from chemistry-transport models run on grids approximately 40 km wide.

These forecasts have four limitations. First, a grid this coarse hides exposure that varies greatly over short distances. Combustion pollutants vary up to eightfold within a single city block<sup>11</sup>. Second, such a grid also hides inequality: in many cities, poorer and minority neighbourhoods breathe more polluted air than their neighbours <sup>12,13</sup>. Third, measurements cannot fill these gaps: in 2019, 141 of 234 countries had no regular PM<sub>2.5</sub> monitoring, and the average person lived 220 km from the nearest monitor<sup>14</sup>. Finally, each forecast is a single deterministic field, with no probability on which a warning can act. Reaching people at the scale of a neighbourhood, with reliable probabilities, calls for another approach.

Since 2023, neural networks have matched or surpassed physics-based weather forecasting<sup>4,5,15</sup>. Deep learning has also been applied to air pollution forecasting, from national networks of stations <sup>16,17</sup> to global forecasts <sup>18–20</sup>. However, these models are trained to minimize a pointwise error, such as the mean squared error, which smooths the forecasts in space and time and blurs the extremes that matter for health. We refer to this problem as oversmoothing. Furthermore, most neural network models step forward through their own output, a few hours to a day at a time (a rollout), so that errors accumulate over long lead times<sup>4,15,18</sup>. Pollution also travels with the air, so that after several days the concentration at a place depends less on its initial state than on where its air came from and which sources it passed over. We refer to this second problem as forecast drift.

In this paper, we present Foehn, a generative neural network that forecasts daily PM<sub>2.5</sub> at 1 km × 1 km resolution over all inhabited land up to 7 days ahead. Foehn first computes where the air arriving at each place comes from via a particle model driven by the forecast wind. It then passes the learned tracers to the flow matching, a generative method that produces plausible future fields via stochastic sampling. Over the test year 2022, its error 1 day ahead at 6,324 monitoring stations, which is 17% lower than that of the operational CAMS forecast. Specifically, this study provides:

kilometre-scale map of exposure, revealing the 186 million people a day above a World Health Organization interim target whom 40 km grids hide;

▶ kilometre-scale forecasts of daily PM<sub>2.5</sub> over all inhabited land, outperforming the CAMS forecast at 75% of 5,954 stations;

▶ calibrated ensembles whose mean detects local pollution extremes 4 days ahead as well as the CAMS forecast does 1 day ahead;

▶ a computed transport that adds skill in the first 3 days, and emission inputs that lower the probabilistic error by 6–8% from 2 to 7 days ahead;

▶ kilometre-scale ensembles of wildfire smoke, placing it over a nature reserve 5 days ahead.

## The Foehn forecasting system

From the state of the air on a start day and a weather forecast, Foehn produces an ensemble of plausible 1 km maps for each of the next 7 days. It learns from the Global High Air Pollutants dataset (GHAP)<sup>6</sup>, which gives daily $\mathrm { P M } _ { 2 . 5 }$ at 1 km over land from 2017 to 2022. These data are derived from satellite observations and meteorological reanalyses and fitted to station measurements, and we call them the 1 km estimates. Foehn has three parts (Fig. 1): (1) FLEXGREEN, which computes where the air arriving at each place was on the start day and the emissions it takes up on its way; (2) learned tracers, which this air carries from its origins; and (3) a conditional generative flow matching, which produces the 1 km×1 km forecasts.

The first two parts describe how pollution moves. FLEXGREEN re-implements on graphics processors the physics of the backward mode of the FLEXPART particle dispersion model <sup>21,22</sup>. For every 0.5<sup>◦</sup> land cell, it releases particles of air on the target day and follows them backwards through the three-dimensional forecast wind, with turbulence, deposition and removal by rain, to where they were on the start day (Methods). The result, a map of where the air of each place came from, is called the Green’s function of the transport. On the way, FLEXGREEN also sums the fire emissions of the start $\mathrm { d a y } ^ { 2 3 }$ and the emissions from human activities <sup>24</sup> that the air takes up near the ground.

Because transport is linear, these paths can carry any quantity. A small neural network therefore turns the start-day maps of $\mathrm { P M } _ { 2 . 5 }$ and fires into 32 quantities, which each place reads where its air was on the start day, keeping only the air that was in the mixed layer, the lowest layer of the atmosphere, in contact with the surface. These learned quantities are called latent tracers. Whereas FLEXGREEN has no learned parameters and is computed once, the tracers are trained together with the generator: the network chooses what to transport, and the physics decides where it comes from.

The third part generates the forecast. A neural network learns to turn random noise into a plausible map of daily PM at 1 km in a few steps, a method called conditional flow matching<sup>25,26</sup>. It is conditioned on the start day, its fires and its weather, on the weather forecast for the target day and on the transport maps, which carry the latent tracers and the emissions (Fig. 1c). Running it from different noise fields gives an ensemble whose spread measures the uncertainty of the forecast. Flow matching alone gives members that spread about half as much as their error. A short fine-tuning on the continuous ranked probability score (CRPS), which rewards both accuracy and calibration, brings the spread up to the error (Methods). Concretely, Foehn has 14 million learned parameters and is trained on 2017–2020, validated on 2021 and tested on 2022. In total, training the final model took 25.8 h on 32 graphics processing units.

This design addresses both problems. Against oversmoothing, the network learns the distribution of $\mathrm { P M } _ { 2 . 5 }$ rather than its average, so that each member keeps fine detail and high values. In a controlled benchmark over Europe, its forecasts are closer to the measurements and sharper on the peaks than those of five networks trained by squared-error regression on identical inputs. They are also more accurate than the CAMS forecast and Aurora<sup>18</sup>, a foundation model of the Earth system (Extended Data Figs. 1–3). Against forecast drift, the transport is computed rather than learned, and each lead time is forecast directly from the start day, so that errors cannot build up by feeding forecasts back as inputs (Fig. 1c).

## Hidden exposure and public health

Air-quality warnings protect public health only if they reach the people who breathe polluted air. A 40 km grid averages a city with its surroundings, whereas exposure differs between neighbourhoods and between social groups <sup>12,13</sup>. Measuring the exposure that such a grid hides is difficult for two reasons. First, exposure depends on where people live within a 40 km cell. Second, the estimates that map it at 1 km have errors and regional biases (Methods).

![](images/515f181a44a92e54b9395778a0a799a4e668c7a55d007ecc021e27a6f0892a8a.jpg)  
Fig. 1 | Foehn computes where the air comes from and learns what it carries. a, Computed transport, without learned parameters. The three-dimensional forecast wind (shown at 850 hPa, 00 UTC on 20 January 2022, the start day) drives FLEXGREEN. FLEXGREEN releases particles in every $0 . 5 ^ { \circ }$ land cell over the target day and follows them back to the start day. On the way, it sums the emissions that the air takes up: the anthropogenic emissions (purple; black carbon shown) and those of the fires of the start day. Lines, 5-day back-trajectories of the air reaching 18 cities on 25 January 2022, blue near the ground and pink aloft. b, Learned tracers. An encoder turns the start day (PM at 0.25<sup>◦</sup> and fires) into 32 tracers. Each place reads them where its air was on the start day, in the mixed layer. c, Generator. A conditional flow-matching network turns a noise field into a 1 km forecast in four sampling steps (red axis), for each lead time (blue axis). It is conditioned on the start day at 1 km (lens: Delhi), the forecast weather (2-m temperature and wind shown) and 44 transport maps: the tracers of b, the emission terms of a, the share of air from aloft, its survival and the advected $\mathrm { P M } _ { 2 . 5 }$ (three shown, 5 days ahead). Discs, one member of the forecast issued on 20 January 2022 for 21, 22 and 27 January, from noise (x ) to forecast (x ). The three intermediate discs blend the two at the flow times reached after the first three steps (0.03, 0.17 and 0.53). Each noise field gives one ensemble member. Colour scale, daily PM<sub>2.5</sub> (µg m<sup>−3</sup>).

On 21 January 2022, nearly half of the people of the Beijing–Tianjin area who breathed air above ${ \hat { 7 } } 5 \mu \mathrm { g } \mathrm { m } ^ { - 3 }$ lived in 40 km cells below that level (Fig. 2b–d). On the 40 km cells of the operational grid, which are what a perfect 40 km forecast would give, their exposure was averaged with that of cleaner surroundings. This case shows that a coarse grid can hide nearly half of the people exposed to a severe episode, even around a capital city.

## Kilometre-scale forecasts

Because PM<sub>2.5</sub> varies within a city, warnings help only if they reach the neighbourhoods in which the air is polluted. Two reference systems, the CAMS forecast <sup>3</sup> and Aurora<sup>18</sup>, forecast $\mathrm { P M } _ { 2 . 5 }$ on grids of about 40 km.

We combine the daily 1 km estimates of 2022 with the 1 km population map of 2020, independently of any forecast. A resident is then hidden when the $\mathrm { P M } _ { 2 . 5 }$ of their 1 km grid cell exceeds a threshold and the mean of their 40 km cell does not (Methods). The thresholds are the 24-h guideline level and interim targets of the World Health Organization (WHO).

On an average day of 2022, 186 million people breathed air above WHO interim target 3 $( 3 7 . 5 \mu \mathrm { g } \mathrm { m } ^ { - 3 } )$ while the mean of their 40 km cell was below it, especially in Africa and the Middle East (Fig. 2a,e). Where terrain confines a city, the 40 km mean can fall below a threshold that most of the city exceeds, as in Santiago de Chile and in the Nile valley around Cairo (Extended Data Fig. 4a–f). Foehn forecasts at this scale, and its maps can support more equitable exposure assessments and public-health decisions, in particular in regions with few data. These numbers rest on the estimates and inherit their errors.

Share of the exposure that a 40 km grid hides, by region (2022)e  
a  
![](images/52ef68610431adfb72fddcf36c5af0422ca36ccb2b554eff44f54366a4c385b3.jpg)

![](images/e28b1a67722ef080650dabf712812ea7fdedb25e848c011e00fd1d6b083b0124.jpg)

![](images/689a0882c18633bd457f786c5e95fb4ae6d1d3e36e1d95ede3b3fde4fee44be5.jpg)

![](images/f23526032d50cf1dc99fa7b2f574d3cef9e575d77843d89d7bfbe60316d08a23.jpg)  
Fig. 2 | Even a perfect 40 km forecast cannot warn millions of people a day who breathe polluted air. a, Number of residents per day in 2022 whose 1 km grid cell was above the World Health Organization (WHO) 24-h interim target 3 $\mathsf { i } ( 3 7 . 5 \mu \mathrm { g } \mathsf { m } ^ { - 3 } )$ while the mean of the surrounding 40 km cell was below it, mapped per 40 km cell: people whom even a perfect 40 km forecast could not warn (1 km estimates of the Global High Air Pollutants dataset, GHAP, and population, without any forecast). b–d, Beijing–Tianjin on 21 January 2022, the most polluted day of the year for the world’s population: daily $\mathrm { P M } _ { 2 . 5 }$ in GHAP with the contour of $7 5 \mu \mathrm { g } \mathrm { m } ^ { - 3 }$ , WHO interim target 1 (b); the same field averaged over 40 km cells, the size of the cells of the CAMS forecast, which is what a perfect 40 km forecast gives (c); and the residents above $7 5 \mu \ g \mathrm { m } ^ { - 3 }$ seen (grey) or hidden (pink) by the 40 km cells (d). e, Of the person-days of 2022 above each threshold in the residents’ 1 km grid cell, the share in 40 km cells whose mean is below it, by region: the WHO 24-h guideline level $( 1 5 \mu \mathrm { g } \mathrm { m } ^ { - 3 } ;$ light) and interim targets 3 (37.5 µg m<sup>−3</sup>) and 1 (75 µg m<sup>−3</sup>; dark); same data as a. A bar is shown when the region has at least one billion person-days above its threshold.

## Extreme pollution episodes

Early warnings matter the most during pollution extremes, and a warning must give the probability of such an episode. The CAMS forecast and Aurora give one deterministic field and no calibrated probability of a local extreme. We identify three challenges in extreme forecasting. First, an extreme is local: a day that is ordinary in northern China could be severe in northern Europe. Second, training on a pointwise error averages away the peaks <sup>27</sup>. Finally, a probability can be trusted only if the spread of the ensemble matches its error, so that an event forecast with a probability of 30% occurs about three times in ten.

We evaluate the forecasts of 2022 against the 1 km estimates on identical land cells and days and, because these estimates have biases of their own, against measurements from 6,324 stations of six official networks that were never used for training (Methods). Foehn’s ensembles have four members, and Extended Data Fig. 5 compares the three systems over four regions on one day.

Over all land, Foehn follows the 1 km estimates closely: their correlation is 0.89 at a lead time of 1 day and 0.80 at 4 days. For the CAMS forecast and Aurora, it is approximately $0 . 6 ,$ and both are biased low relative to these data. One day ahead, Foehn’s error is a third below that of each place’s climatology (the mean of the estimates of 2017–2020 within a week of the same calendar day). After a week, its error is only 3% below that of the climatology (9.9 against $1 0 . 2 \mu \mathrm { g } \mathrm { m } ^ { - 3 } ;$ correlation 0.76 against 0.75), as the predictability of the weather fades.

![](images/95c7de0ff6594e0306b6f5a118b804803a6c8d7a1237186fd7c9be2418767d95.jpg)

![](images/9e20b46fc9701be57099d7616da48af53b06e002367ea8cf91b71cc1134d6023.jpg)  
Fig. 3 | Foehn forecasts all inhabited land at 1 km and resolves basins that the operational grid blurs. a, Foehn’s forecast of daily PM (ensemble mean of four members) issued at the end of 20 January 2022 for 21 January, the day of 2022 with the highest population-weighted $\mathrm { P M } _ { 2 . 5 } .$ b, The forecast of a in four polluted basins, each monitored by a different official network (1, Delhi, Central Pollution Control Board of India; 2, Po Valley, European Environment Agency; 3, Los Angeles, US Environmental Protection Agency; 4, Seoul, AirKorea). Shown are the CAMS forecast issued at the same time (00 UTC run of 21 January, 0.4<sup>◦</sup> grid), Foehn at 1 km and the 1 km estimates (GHAP, made after the fact, not a forecast), with the measurements of that day at the stations in the same colours. Numbers, root-mean-square error (RMSE) over 2022 at the stations of each basin, 1 day ahead, on identical station-days. Aurora and the error against the estimates in four regions are shown in Extended Data Fig. 5.

At the measurement stations, Foehn’s root-meansquare error (RMSE) 1 day ahead is $1 2 . 7 \mu \mathrm { g } \mathrm { m } ^ { - 3 } .$ against 15.2 for the CAMS forecast and 15.7 for Aurora (Extended Data Fig. 6). Four days ahead, the gains are 12% and 17%. Foehn is more accurate than the CAMS forecast at 75% of the 5,954 stations with at least 30 common forecast days. The gain is as large at the 4,295 stations of the five networks outside Europe. From ensemble forecasting, its CRPS 1 day ahead, $4 . 3 \mu \mathrm { g } \mathrm { m } ^ { - 3 } .$ , is about half that of the CAMS forecast (7.9) and Aurora (8.3). Seven days ahead, at 5.3. It remains well below that of the climatology at the measurement stations (9.7).

This comparison has a limit. The stations are not independent of the gridded global estimates, which were fitted to measurements from about 9,500 stations, including the European, Chinese and US networks <sup>6</sup>. The accuracy of these data also varies widely between networks. They match the Chinese network almost exactly (RMSE 2.4 µg m<sup>−3</sup>) but are no better than their own climatology at the European network (11.2 against 11.0) and 5.9 µg m<sup>−3</sup> too low at the Indian network (Methods). Foehn’s gain is largest in China, where both reference models overestimate the haze events. In India, Foehn is only as accurate as the CAMS forecast and Aurora, and it amplifies the low bias of the estimates (−10.9 against $- 5 . 9 \mu \mathrm { g } \mathrm { m } ^ { - 3 } )$ .

a–g Forecasts for 25 January issued 1 day ahead  
h  
![](images/2e4ffeecb49e4a240bbfcf7c5728d3d73d89faba469187adf979d4d538d81523.jpg)  
i–o Forecasts for 25 January issued 5 days ahead

![](images/5e09ead587f25e73d77a6482c202d806a4d9cf2a52cc713a221cf91ee9f0c21e.jpg)

![](images/0e83c8d1355c25214f80e2cb467b3d0bd6ea0788e1b4376ab7d27e8569e4eba6.jpg)  
q–t Probability of PM > 75 µg m<sup>−3</sup> on 25 January (16 members); contour: 75 µg m<sup>−3</sup> in GHAP

![](images/f100d4781dfdb77cef506dd6bbb4ae84c451cfa8553038874017f794c126d30e.jpg)

![](images/12a5279f2765707e903132b71bc381923b31f64a6f1fcdf1679e94f41c9b728e.jpg)

![](images/d2fa12c284bf4fd1c2e02b3c0b1677c6a76615f1dd8cfbb012ae18436af96dbb.jpg)  
Fig. 4 | Each member of Foehn’s ensemble keeps the sharp structure of a haze episode that the operational forecasts smooth away. Daily PM on 25 January 2022, the peak of the haze over the North China Plain. a–g, The 1 km estimates (a) and the forecasts issued 1 day ahead: three of Foehn’s 16 members (b–d), Foehn’s ensemble mean (e) and the CAMS (f) and Aurora (g) forecasts issued at the same time, on their $0 . 4 ^ { \circ }$ grid. h, Variance per decade of wavelength of each field over the panel area (radially averaged power spectrum). i–o, p, The same, issued 5 days ahead. q–t, Probability of exceeding 75 µg m<sup>−3</sup> (WHO 24-h interim target 1) from Foehn’s 16 members, issued 7, 5, 3 and 1 days ahead; black contour, $7 5 \mu \mathrm { g } \dot { \mathrm { m } ^ { - 3 } }$ in the estimates. u, Daily mean (UTC day) of the measurements at the 12 monitors of the China National Environmental Monitoring Centre (CNEMC) in central Beijing in January 2022 (circles), GHAP at their grid cells (grey dotted) and the forecasts issued 1 day earlier: Foehn’s 16 members (band, minimum to maximum; line, ensemble mean) and the CAMS forecast issued at the same time (black); light blue dashed, Foehn’s ensemble mean issued 3 days earlier.

At the European and Japanese networks, its forecast 1 day ahead is closer to the measurements than the estimates of the same day (RMSE 9.8 against 11.2, and 5.9 against 6.6; Extended Data Fig. 6a–f). We hypothesize that averaging the members acts as a filter. The mean keeps the part of the estimates that the start day and the weather can predict, and averages out the rest. At these networks, this rest would be partly error in the estimates, which would explain why the mean is closer to the measurements than the estimates themselves.

The gain is not a matter of resolution alone: averaged $\mathrm { t o } \ : 0 . 5 ^ { \circ }$ before being read at the stations, Foehn’s error is still below that of both reference systems. At 1 km, it is 5% below that of its own 0.25<sup>◦</sup> average (Extended Data Fig. 6g). Foehn’s design also does not depend on the 1 km estimates. Retrained in Aurora’s own protocol, on the coarse CAMS analyses and without any 1 km or station data, the same design outperforms Aurora by 9% in RMSE 3 days ahead and 12% 5 days ahead (Methods; Extended Data Figs. 7 and 8).

The day of 2022 with the highest populationweighted $\mathrm { P M } _ { 2 . 5 }$ was 21 January. The forecast issued the evening before placed the haze over the Indo-Gangetic Plain, the North China Plain and the Sahel (Fig. 3a). At 1 km, it resolved details that the 40 km grid of CAMS missed, such as the edge of the Indo-Gangetic Plain, the Po Valley along the Alps, the Los Angeles basin and the hills around Seoul. The station measurements, plotted in the same colours, showed the same patterns (Fig. 3b). This case shows that the network resolves pollution at the scale of a basin over the whole globe.

Foehn’s ensemble mean detects the days above each place’s own 99th percentile better than the CAMS forecast and Aurora in every region (fewer false alarms). Four days ahead, it detects them as well as the CAMS forecast does 1 day ahead (Extended Data Fig. 4g,h). At the measurement stations, this holds with the same thresholds, but not beyond 1 day with each station’s own percentiles (Methods). The proposed particle transport contributes: without it, the same network detects fewer extremes in every region 3 days ahead. For a user whose protection costs a twentieth of the loss that it avoids, Foehn’s warnings have more than twice the potential value of those of both reference systems, a value estimated retrospectively (Extended Data Fig. 4i,j). Foehn’s ensemble mean, however, is smoother than its members, and the peak of an episode can be missed several days ahead.

On 25 January 2022, a week-long haze episode over the North China Plain reached its peak. Each member of the ensemble issued 1 day ahead was a sharp field in which the haze filled the plain, stopped at the mountains and thinned towards the coast, as in the estimates (Fig. 4a–g). At the monitors of central Beijing, the ensemble captured the two days of haze (measured 142 and 144 µg m<sup>−3</sup>; forecast 1 day earlier 140 and 159) and the clearing that followed, although it underestimated how far concentrations fell (Fig. 4u). Issued 1, 3 or 7 days ahead, the ensemble gave a high probability of exceeding 75 µg m<sup>−3</sup> (WHO interim target 1), where the estimates exceeded it, and a low one elsewhere. However, from five days ahead, it under-forecast the peak (Fig. 4q–t). This case shows that calibrated ensembles turn sharp members into probabilities on which a warning can act.

## Transport and emissions

A forecast several days ahead must know where the arriving air comes from and which emission sources it passes over. Here we demonstrate that the computed transport and the emissions taken up along the way add to Foehn’s skill from the first day to the seventh, the first at short lead times and the second at longer ones. Learned forecasting models generally infer the transport from their inputs and accumulate errors through a rollout <sup>15,18</sup>, whereas chemistry-transport models compute it on grids tens of kilometres wide<sup>3</sup>. To separate the two contributions, we compare the base model, which has the transport inputs but not the emissions, with the same network trained without the transport inputs and with the same model fine-tuned with the emissions (Methods).

The proposed particle transport shows that the share of the arriving air that was in the mixed layer on the start day falls with lead time. Over all land, it is approximately three-quarters for the next day, less than half after 3 days and a sixth after a week (Fig. 5h,i). The start day therefore carries progressively less information about the target day. The gain from the computed transport declines accordingly: it lowers the CRPS by approximately 6% 1 day ahead, by half as much 3 days ahead and not significantly beyond (Fig. 5j).

We therefore define extremes relative to each place’s own climatology, as has been done for weather extremes <sup>29</sup>. A cell and day is labelled moderate, severe or extreme when it exceeds the local 90th, 95th or 99th percentile over the reference period 2017–2020 (Methods). We score their detection by the symmetric extremal dependence index (SEDI), a score that does not depend on how rare the events are<sup>30</sup>, and the warnings by their relative economic value (Methods).

Each member of the ensemble is a possible scenario of a haze episode, with its own fine-scale structure. At scales finer than 20 km, the members have as much spatial variability as the 1 km estimates 1 and 3 days ahead, and at least half of it after a week. By contrast, the 40 km forecasts of CAMS and Aurora have almost none below approximately 80 km (Fig. 4h,p). After the fine-tuning on the CRPS, the ensembles are calibrated against the estimates: in 2022, their spread matches their error at every lead time (ratio 0.96–0.99; Methods), and on the validation year 2021 the ratio is 0.82–0.96 (Extended Data Fig. 6h).

The emissions take over at longer lead times, provided that the inventory captures the dominant local sources (Extended Data Fig. 9). Where they help, their gain does not fade. In Europe, North America and South America, it rises over the first 2–4 days and then holds for the rest of the week, at approximately 7%, 11% and 4% of the RMSE of the ensemble mean, respectively. In East Asia, the two contributions cross at 4 days (Extended Data Fig. 9a–d). Over all regions together, the emissions lower the CRPS by 6–8% from 2 to 7 days ahead.

The emissions do not help everywhere for two reasons (Extended Data Fig. 9e,f; Methods). First, the inventory misses the dominant sources of some regions. Across most of Africa, desert dust and fires produce most of the PM<sub>2.5</sub> <sup>1</sup>, and global inventories misrepresent many local sources, such as fuelwood, charcoal and generators<sup>31</sup>. There, the emissions degrade the ensemble mean by approximately 4% at every lead time, and the CRPS improves at long lead times only because the ensemble widens (Extended Data Fig. 9j). They also degrade it, by 1–3%, in the Middle East, where dust dominates as well<sup>1</sup>. Second, where sources are intense every day, the start-day field already reflects them. In the North China Plain and the Indo-Gangetic Plain, where residential, industrial and energy emissions are large <sup>1</sup>, the gain is small or absent, with 95% intervals that include zero (Methods). A monthly inventory adds little to such a start day, and inventories are also uncertain there <sup>32,33</sup>. In South Asia, the emissions even degrade the ensemble mean slightly in the first 3 days and have no effect thereafter.

a  
b  
![](images/3eca8a8ea1e0d9770f49e40e4fa64ddd1f82cbc7a1e89d5be6796ddf59ae87b3.jpg)

![](images/9e67f2ed8b7dab24f062cddd4aa5aaa7a784d25197d2213af7ac2b513fef95dc.jpg)  
e

![](images/1baffaf691bed9b1d90736b973d8586a9db51d29b2adf39715eb1a37a446b2f7.jpg)

c  
![](images/1ff21fa287fac8c4387388d68f73a20cadb089fa62332ae2f0ad5f5e20c589af.jpg)  
f

![](images/cf3684b4fc417d7f8fefae16a11de310dbc546805840376be184a00447042da3.jpg)

![](images/507de0647fad3a6ceb4e3e704517f67904627ff39b635e180f292a96644f9742.jpg)

d  
![](images/9c378433958fb2534992a2489ed9c87e51aeb332dfc114f378b9c70aae1ed18d.jpg)

g  
![](images/897839e9de752735fae2a9cb47cc2cee5590228c67d8e03eba905f6df56fbecf.jpg)

h  
![](images/2c8a1ce36471b19b3d11a294a12ceec73c0b74b76c5e25470d4f902ed7b15eff.jpg)  
i

![](images/94d9b896aed1f60d3d350d0208be75d7ecdb714ec138f91368f4e30f26044139.jpg)

j  
![](images/a942947f8828ddf3a712bc61edecb0eb21f04adeb05ac0219a0a91f9e8aa0dac.jpg)  
Fig. 5 | The computed transport traces a dust episode to its origin, and its gain fades as the start day loses influence. a–d, GHAP daily PM at 1 km over the central Sahel on 13–16 May 2022 and, on each of the first 3 days, where the air reaching Kano (star) on 16 May was then (black dots), from the FLEXGREEN Green’s functions of the forecasts started on 13 (a), 14 (b) and 15 May (c). e–g, Foehn’s probability of $\mathrm { P M } _ { 2 . 5 }$ above 100 µg $\mathbf { m } ^ { - 3 }$ on 16 May at 1 km, from its 16 members, in the forecasts issued $3 \ ( \mathbf { e } ) , 2 \ ( \mathbf { f } )$ and 1 day (g) before; compare with d. h–j, Why the gain of the computed transport fades with lead time (all land, 2022). h, Where the air arriving on the target day was on the start day: in the mixed layer (blue) or higher up or farther back (pink), by lead time. i, The same share 3 days ahead, place by place: where the start day still matters. j, Reduction of the continuous ranked probability score (CRPS) against the 1 km estimates brought by the computed transport: violins, for each start day; dots and bars, over all forecasts, with 95% intervals.

On 16 May 2022, $\mathrm { P M } _ { 2 . 5 }$ around Kano, Nigeria, rose sharply in the estimates. The forecasts started on the three preceding days traced the arriving air to the dust over Niger and Chad, hundreds of kilometres to the north-east (Fig. 5a–d). In the forecasts issued 1 and 2 days before, Kano had a probability above one half of exceeding $1 0 0 \mu \mathrm { g } \mathrm { m } ^ { - 3 }$ . In the forecast issued 3 days before, most members kept the plume north of the city $( { \mathrm { F i g . ~ } } 5 \mathbf { e - g } )$

## Wildfire smoke

Wildfire smoke harms people and wildlife far from the flames<sup>9</sup>, so a useful forecast must place it at the scale of the towns and nature reserves that it reaches. Forecasting smoke days ahead is difficult for two reasons. First, smoke from a fire that ignites after a forecast is issued cannot be foreseen by any system. Second, Foehn knows the fires only through their emissions on the start day, estimated from satellite observations<sup>23</sup> so it cannot know whether they will grow or die down. In this case, its 16 members give at every place the probability that $\mathrm { P M } _ { 2 . 5 }$ exceeds $2 5 \mu \mathrm { g } \mathrm { m } ^ { - 3 }$ , the WHO interim target 4.

In July 2022, during the second-worst fire season of the European Union by burnt area, two fires that started on 12 July burnt about 18,000 hectares of pine forest in the Gironde, in southwestern France, before being contained<sup>28</sup>. One of them burnt beside the Réserve Ornithologique du Teich, a bird reserve on the Arcachon basin, where the smoke raised $\mathrm { P M } _ { 2 . 5 }$ to its peak on 18 July, the only day from 13 to 26 July above $2 5 \mu \mathrm { g } \mathrm { m } ^ { - 3 }$ in the estimates (Fig. 6a,f). Issued on 11 July, before the fires ignited, Foehn’s forecast for 18 July gives almost no chance of exceeding this threshold near the fires, as any forecast issued before an ignition must (Fig. 6b). Issued on 13 July, 5 days ahead, it gives a probability of at least one in four to 94% of the area in which the estimates put $\mathrm { P M } _ { 2 . 5 }$ above $2 5 \mu \mathrm { g } \mathrm { m } ^ { - 3 } .$ , with a mean probability of 0.48 inside this area and 0.12 outside it (Fig. 6d). It gives the reserve a probability of approximately one third (Fig. 6g). However, its zone of likely exceedance is five times larger than in the estimates, mainly over the forest of the Landes.

![](images/91ba6de149da8e4c4299be7c4f21cc94d02bb1a0a29675515ebfef5bb890bfc3.jpg)

![](images/e6fcb052f59e719dce67a6ab1f3c79b17b70b8918c4300249bc159e1785fda3e.jpg)

![](images/b2d3740f4528781ac01ce92f118a6dd3f8f5817e3a2dc433543d823d74469500.jpg)  
Fig. 6 | Five days ahead, Foehn places the smoke of the 2022 Gironde fires over a bird reserve. The La Teste-de-Buch and Landiras fires, which started on 12 July 2022, burnt about 18,000 hectares in the Gironde that month<sup>28</sup>; the La Teste fire burnt beside the Réserve Ornithologique du Teich (open circle), a bird reserve on the Arcachon basin; star, Bordeaux. a, GHAP daily PM at 1 km on 18 July, the smokiest day at the reserve; triangles, the fire cells of the day in the Global Fire Assimilation System (GFAS); circles, the European Environment Agency (EEA) stations, coloured by their measurement. b–e, Foehn’s probability of PM above 25 $\mathrm { | \mu g m } ^ { - 3 }$ (WHO interim target 4) on 18 July, from its 16 members, issued $^ { 7 } \left( \mathbf { b } , \right.$ 11 July, before the fires ignited), 6 (c), 5 (d) and 1 day (e) before; triangles, the fire cells in the inputs of each forecast; black contour, $2 5 \mu \mathrm { g } \mathrm { m } ^ { - 3 }$ in GHAP on 18 July. f, GHAP daily PM over the 20 km surroundings of the reserve, 13–26 July (shading, days with fire cells over the two fires; dashed, 25 µg m<sup>−3</sup>). g, Foehn’s probability of exceeding $2 5 \mu \mathrm { g } \mathrm { m } ^ { - 3 }$ over the same surroundings by target day (columns) and lead time (rows); numbers, probabilities of 0.19 and above; grey, no forecast in this evaluation. h, The 16 members over the same surroundings for 18 July, by lead time (bars, ensemble mean; dotted, GHAP; dashed, $2 5 \mu \mathrm { g } \mathrm { m } ^ { - 3 } )$ ).

The warning at the reserve held across lead times. For 18 July, the members rose towards the estimated $\mathrm { P M } _ { 2 . 5 }$ as the fires entered the inputs: one of 16 exceeded 25 µg m $^ { - 3 } 7$ days ahead, and five did so from 5 to 3 days ahead (Fig. 6h). This is the only target day to keep a probability of 0.19–0.31 at every lead time from 1 to 5 days. On the other days, the probability stays mostly below 0.19, with isolated false alarms of up to 0.44 (24 July, 6 days ahead; Fig. 6g). This case shows that, once fires appear in its inputs, Foehn can give the probability that smoke pushes $\mathrm { P M } _ { 2 . 5 }$ above a health threshold at the scale of a nature reserve 5 days ahead.

## Discussion

We introduce Foehn, a generative model that forecasts daily fine particulate matter (PM<sub>2.5</sub>) at kilometre-scale resolution across all inhabited land, up to a week ahead. By combining a transport model with flow matching,

Foehn produces physically consistent forecasts with calibrated ensembles. Its forecasts outperform both operational forecasts and other neural networks, while detecting local extremes days earlier. Three limitations remain. First, Foehn cannot yet deliver real-time forecasts at kilometre-scale resolution because its initial estimates are generated retrospectively. Real-time deployment would require a method to derive kilometrescale $\mathrm { P M } _ { 2 . 5 }$ estimates from coarser, routinely available forecasts. A more practical near-term approach is to deploy Foehn in cities that already provide street-scale PM<sub>2.5</sub> maps based on live measurements, such as London<sup>34,35</sup>. Second, errors and regional biases in the high-resolution $\mathrm { P M } _ { 2 . 5 }$ estimates can propagate through Foehn or be amplified. Although we quantitatively evaluate forecasts against observations in countries with live monitoring networks, forecast reliability remains uncertain in regions without comparable measurements. Third, all ensemble members rely on the same deterministic weather forecast, while the emission inputs omit desert dust and misrepresent the locations of other sources. Regional emission inventories<sup>36</sup>, together with daily fire and dust emission estimates, could help address these limitations.

In conclusion, combining explicitly modelled physics with learned dynamics offers a path towards accurate and interpretable air-quality forecasting. This approach could extend to other airborne quantities, including individual components of fine particulate matter, coarser particles, and trace gases. Kilometre-scale ensemble forecasts could enable neighbourhood-level warnings, exposure assessments in regions with sparse monitoring, and better protection of nature reserves from smoke. Despite its current limitations, Foehn demonstrates the potential for air-quality forecasts at the neighbourhood scale, bringing actionable warnings within reach of millions of people who remain invisible to coarser-resolution maps.

## Methods

## Problem statement

Let $x _ { d }$ be the field of daily-mean surface $\mathrm { P M } _ { 2 . 5 }$ on day $d ,$ on a regular latitude–longitude grid of $0 . 0 1 ^ { \circ }$ (≈1 km×1 km). A forecast is issued at the end of a start day d for a target day d + h, where $h \in \{ 1 , \ldots , 7 \}$ is the lead time in days. Foehn is a generative model: it draws samples from the conditional distribution

$$
\approx p \big ( x _ { d + h } \mid c _ { d , h } , m , y , e , w \big ) ,\tag{1}
$$

where $c _ { d , h }$ is the 1 km $\mathrm { P M } _ { 2 . 5 }$ of days d and $d - 1 ;$ m the static maps (elevation and a land mask); y the coarseresolution (0.25<sup>◦</sup>) $\mathrm { P M } _ { 2 . 5 }$ of day d; e the weather and wildfire emissions of day d and the forecast weather of day $d + h ;$ and w the transport of the air between days d and $d + h$ . Each sample is a member of the forecast, and M members form an ensemble. Every lead time is forecast directly from day d, so that errors cannot accumulate through an autoregressive rollout. The forecasts are issued as if the 1 km $\mathrm { P M } _ { 2 . 5 }$ of day d were available at its end. These data are produced later, so every 1 km forecast in this study is retrospective.

## Datasets

The training data is the daily 1 km $\mathrm { P M } _ { 2 . 5 }$ of the Global High Air Pollutants dataset (GHAP, version 1)<sup>6</sup> for 2017–2022, gap-free estimates derived from satellite data and meteorological reanalyses and fitted to station measurements (we refer to it as the 1 km estimates in the main text). The years 2017–2020 train the models, 2021 validates them and 2022 is the test year.

Foehn does not forecast the weather: it takes it from the operational high-resolution forecast (HRES) of the European Centre for Medium-Range Weather Forecasts (ECMWF), run at 00 UTC on the start day d. This single forecast provides both the weather of each target day in e and the weather that drives the transport w. Only the weather of the start day comes from the ERA5 reanalysis <sup>37</sup>, released about 5 days after real time, which an operational analysis would replace in real time.

Wildfire emissions come from the Global Fire ${ \mathrm { A s } } -$ similation System $( \mathrm { G F A S } ) ^ { 2 3 }$ of the Copernicus Atmosphere Monitoring Service (CAMS), for day d only; monthly anthropogenic emissions come from CAMS-GLOB-ANT v6.2<sup>24</sup>, a global inventory extended to recent years by extrapolation. The residents of each 1 km pixel come from GHS-POP R2023A<sup>38</sup>. Since the 1 km estimates inherit bias and pattern errors, we further evaluate the forecasts at 6,324 stations of six official networks: the European Environment Agency (EEA; 2,029 stations), the China National Environmental Monitoring Centre (CNEMC <sup>39</sup>; 1,338), the Air Quality System of the US Environmental Protection Agency (EPA; 1,016), AirKorea (619), the Japanese national network (949) and the Central Pollution Control Board of India (CPCB; 373). Daily means are over UTC days, like those of $\mathrm { G H A P ^ { 6 } }$ and of every forecast, from at least 12 hourly values (European network: official daily means, offset by 1 h; US sites without hourly values are left out), and stations with at least 30 valid days are kept. The station measurements are never used to train or select Foehn.

## The Foehn model

Transport-informed latent field: FLEXGREEN.. The concentration at a receptor is the sum of the earlier field and of the sources met by its air, each weighted by the Green’s function of the transport, the sensitivity of the concentration at the receptor to the air at each earlier place and time <sup>40</sup>. A backward Lagrangian particle model estimates this function by releasing particles of air at the receptor and following them back in time. FLEXGREEN is our re-implementation, on graphics processors, of the backward mode of FLEX-$\check { \mathrm { P A R T } } ^ { 2 1 , 2 \hat { 2 } } ,$ , driven by the HRES weather. It follows FLEXPART’s published equations: Petterssen advection<sup>41</sup> with 15-min time steps; the height of the mixed layer, the turbulent layer next to the surface, from the bulk Richardson number<sup>42</sup>; Hanna turbulence<sup>43</sup> with a random-displacement model<sup>44</sup>; and constant diffusivities above the mixed layer. It differs from FLEXPART in three ways. (i) The weather comes from the 6-hourly HRES fields on eight pressure levels, averaged over 1<sup>◦</sup> boxes, and, without surface fluxes, the friction velocity, Obukhov length and convective velocity are derived from Businger–Dyer profiles <sup>45</sup>, as in HYSPLIT<sup>46</sup>. (ii) Where turbulent mixing is faster than one time step, the heights of particles in the mixed layer are redrawn uniformly between the ground and its top, the stationary state of a well-mixed layer. (iii) Removal takes FLEXPART’s simplest forms: a dry deposition velocity of 0.05 $\mathrm { c m } \thinspace \mathrm { s } ^ { - 1 }$ in the lowest 30 m, and wet scavenging $\Lambda = A I ^ { B }$ below precipitating clouds, with I the precipitation rate in mm h<sup>−1</sup>, $A = 1 0 ^ { - 4 } { \mathrm { s } } ^ { - 1 }$ and $B = 0 . 8$

For every $0 . 5 ^ { \circ }$ land cell (a receptor r), start day d and lead time $h , P = 3 2$ particles are released at random times during the target day between 0 and 100 m above the ground and followed back h days. Each particle p stores its origin $\mathbf { x } _ { p }$ on day $d ;$ its height there divided by the maximum height of the mixed layer on that day, $\eta _ { p }$ (the air was in the mixed layer when $\eta _ { p } \leq 1 )$ ; and its survival $s _ { p } \in [ 0 , 1 ]$ , the fraction of its aerosol not removed by deposition and rain on the way. Along the paths, the GFAS emissions (three fire terms) of day $d ,$ spread from the ground to the larger of the injection height and the mixed-layer height. The residence time below $1 0 0 \mathrm { m } ^ { 4 0 }$ , weighted by survival, times the monthly CAMS-GLOB-ANT emission fluxes, including black and organic carbon, SO<sub>2</sub>, $\mathrm { N O } _ { x } ,$ NH<sub>3</sub> and CO.

Transport-informed latent field: tracers.. Transport is linear, so it commutes with any linear combination of transported quantities. An encoder of two $3 \times 3$ convolutions turns the $0 . 2 5 ^ { \circ }$ fields of day d (the standardized $y ,$ its validity mask and the logarithms of the three GFAS variables: the fire emission of particulate matter, the fire radiative power and the injection height) into 32 latent tracers $f = \left( f _ { 1 } , \ldots , f _ { 3 2 } \right)$ at $0 . 5 ^ { \circ }$ , maps whose content is learned. Each receptor r gathers them at the origins of its particles:

$$
g _ { r } = \frac { \sum _ { p } \omega _ { p } f ( \mathbf { x } _ { p } ) } { \sum _ { p } \omega _ { p } } , \qquad \omega _ { p } = s _ { p } \mathbf { 1 } [ \eta _ { p } \leq 1 ] ,\tag{2}
$$

where the sums run over the particles released at r and $f ( \mathbf { x } _ { p } )$ is interpolated bilinearly, so that only air that was in the mixed layer on day d is read. The operator has no parameters and is differentiable in $f ,$ so the encoder learns what to transport from the forecast error. With the advected $\mathrm { P M } _ { 2 . 5 } ,$ the share $a _ { r }$ of the air that came from above the mixed layer, the mean survival and the wildfire and anthropogenic emission terms, each receptor receives 44 transport channels, which represent w.

Generator network.. The generator $v _ { \theta } ,$ a neural network with parameters $\theta ,$ predicts the velocity of the flow at 1 km over a window of $4 , 1 9 2 \times 6 , 9 9 2$ pixels in one pass. It is a $\mathrm { U - N e t ^ { 4 7 } }$ of residual blocks from 2 to 32 km that receives $x _ { \tau } , c _ { d , h }$ and m at 1 km and y, e and the transport channels at 16 km. At 64 km, four transformer blocks <sup>48</sup> apply global self-attention across the window, so that information crosses the continent in one layer. Sinusoidal embeddings of the flow time and of the lead time modulate every $\mathrm { b l o c k ^ { 4 9 } }$ . The whole model has 14.2 million parameters.

Global forecasts.. Twenty-one overlapping windows cover the land between 84<sup>◦</sup>N and $6 0 ^ { \circ } \mathrm { S }$ and are blended with weights rising as $\sin ^ { 2 }$ over 600 km from each edge. Each member uses one global noise field, which every window cuts at its own position, so that overlapping windows integrate the same noise.

## Training methods

Training has three stages: base training by flow matching, fine-tuning with the anthropogenic emissions, and calibration fine-tuning on the continuous ranked probability score (CRPS; see Evaluation and verification metrics). An epoch is one pass over the 9,730 training pairs of start day and lead time of 2017–2020.

Flow matching.. We learn a velocity field $v _ { \theta }$ that carries Gaussian noise to the distribution of equation $( 1 ) ^ { 2 5 , 2 6 , 5 0 }$ Let $x _ { 1 } = ( x _ { d + h } - 1 5 ) / 2 0$ be the standardized target (concentrations in µg m $^ { - 3 } ) , x _ { 0 } \ .$ a field of independent standard Gaussian noise and $x _ { \tau } = ( 1 - \tau ) x _ { 0 } + \tau x _ { 1 }$ the straight path between them, where $\tau \in [ 0 , 1 ]$ is the flow time. The network learns the velocity $x _ { 1 } - x _ { 0 }$ along this path:

$$
\begin{array} { l } { \displaystyle \mathcal { L } ( \theta ) = \mathbb { E } \Big [ \frac { 1 } { | \Omega | } \sum _ { i \in \Omega } \big ( v _ { \theta } ( x _ { \tau } , \tau , h ; c _ { d , h } , m , y , e , w ) _ { i } } \\ { \displaystyle - ( x _ { 1 } - x _ { 0 } ) _ { i } \big ) ^ { 2 } \Big ] , } \end{array}\tag{3}
$$

where $\Omega$ is the set of valid land pixels of the window and i indexes its pixels. A member is drawn by integrating dx $/ \mathrm { d } \tau = v _ { \theta }$ from noise at $\tau = 0$ to a forecast at $\tau = 1$ with four Euler steps; each noise field gives one member. For half of the training samples, the noise level $\sigma = 1 - \tau ,$ , drawn uniformly on $[ 0 , 1 ]$ , is mapped to $\kappa \sigma / ( 1 + ( \kappa - 1 ) \sigma )$ with $\kappa = 2 5$ , which moves training towards high noise levels $^ { 5 1 , 5 2 } ;$ the four Euler steps are placed at the quantiles of this distribution of $\tau .$

Base training and emission fine-tuning.. The base model is trained for 60 epochs on 32 graphics compute dies (GCDs; 19 h), with the HRES weather and transport of the training years. It is optimized with the AdamW optimizer <sup>53</sup> a nd a one-cycle learning-rate schedule<sup>54</sup>. It uses the 38 transport channels without the six anthropogenic emission terms. It is then fine-tuned for 12 epochs, with the same loss, with the six additional anthropogenic emission terms appended. Their input weights start at zero, so that the uncertain inventories cannot destabilize the base model. The checkpoint after its third epoch, selected on 2021, is the model with emissions. The ablation without transport is trained like the base model, with the transport channels set to zero.

Calibration fine-tuning.. Trained by flow matching alone, the ensembles were too narrow: on 2021 the spread– skill ratio of four members was 0.51 one day ahead and 0.43 after a week. We therefore fine-tuned the model on the CRPS, as recent weather models $\mathrm { d o } ^ { 5 5 } \colon$ each training sample draws two members exactly as a forecast does, and the loss is their fair CRPS against $\mathrm { G H A P ^ { 5 6 } }$ plus the flow-matching loss, with the gradient flowing through the sampler. After two epochs, the spread–skill ratio on 2021 was 0.92 one day ahead and 0.82 after a week, and the CRPS was 12% lower one day ahead, for a 2% increase in the error of the ensemble mean (Extended Data Fig. 6h); on 2022 the ratio was 0.96–0.99 at every lead time. In total, training the final model took 25.8 h on 32 GCDs.

Design, selection and test.. The European benchmark (see Benchmarks) set the design principles, and the configuration of the global model was fixed before training. Its ablations did not change it: on 2021, the model trained with ERA5 weather had a 1% lower CRPS summed over the seven lead times (0.6% higher one day ahead), and the planned configuration was kept.

## Benchmarks

European benchmark.. A controlled benchmark over Europe is used for ablation studies of the design: five spatiotemporal networks, ConvLSTM<sup>57</sup>, SimVP<sup>58</sup>, Earthformer <sup>59</sup>, ClimaX<sup>60</sup> and TopoFlow <sup>61</sup>, were retrained with exactly the inputs of the flow-matching network and scored on 2022 against GHAP and 2,971 EEA stations. The flow-matching network scored better than each other system in 319 of 324 comparisons of score, lead time and set of days, and was within 2% in the five others (Extended Data Figs. 2 and 3). Networks trained at 1 km and applied tile by tile left blocky artefacts at the tile boundaries (Extended Data Fig. 10).

Comparison in Aurora’s protocol.. To compare Foehn with Aurora on its own terms, we retrained Foehn in the protocol of Aurora’s air-pollution model <sup>18</sup> (Foehn-CAMS; Extended Data Figs. 7 and 8): on the 0.4<sup>◦</sup> grid of the CAMS analysis, from the analyses of 00 UTC and of 12 h earlier, it forecasts the instantaneous PM every 12 h up to 120 h. Foehn-CAMS keeps Foehn’s components (FLEXGREEN, 32 transported tracers, the HRES forecast of the initial day as weather, and the same network and loss), applied to the whole globe at once, with 9.9 million parameters, 131 times fewer than Aurora. All its inputs are available in real time. It was trained once and its final checkpoint scored once, on forecasts initialized at 00 UTC from May to November 2022, against the CAMS analysis.

## Evaluation and verification metrics

All scores are on the forecasts issued every second start day of 2022, one to seven days ahead. Foehn’s forecasts have four members (M = 4) unless stated otherwise and are read at the 1 km pixel of each station. The CAMS global forecast<sup>3</sup> and Aurora (0.4<sup>◦</sup>, one to four days ahead) are interpolated bilinearly. Because Foehn uses the whole start day, it could be issued at the end of day d, 24 h after the CAMS run of that date. Comparisons at equal issuance time therefore use the CAMS and Aurora forecasts of 00 UTC on day d + 1. The climatology of a place and calendar day is the mean GHAP of 2017–2020 within ±7 days of that day. Systems are compared on identical station-days or, against GHAP, on identical 0.25<sup>◦</sup> land cells and days. Read at the station pixel on the same day, GHAP itself has an RMSE against the measurements of 2022 of 2.4 µg m<sup>−3</sup> at the Chinese stations, 2.6 at the US, 6.6 at the Japanese, 8.5 at the Korean, 11.2 at the European and 36.5 at the Indian stations (bias −5.9), and 11.0 over the six networks.

We report the root-mean-square error (RMSE) and the bias of the ensemble mean. For uncertainty analysis, we use the CRPS, a strictly proper score <sup>62</sup> that rewards both accuracy and calibration, estimated without ensemble-size bias <sup>56</sup> and equal to the absolute error for a deterministic forecast; and the spread–skill ratio<sup>63</sup>, which is 1 for a calibrated ensemble and below 1 for an ensemble that is too narrow. Local extremes are the days above the local 90th, 95th or 99th percentile of 2017–2020, computed for each 0.25<sup>◦</sup> cell and calendar day (±15 days) from the GHAP block means (the moderate, severe and extreme tiers, labelled against the local climate as in TailWeather<sup>29</sup>); their detection is scored by the symmetric extremal dependence index (SEDI) <sup>30</sup>, which does not depend on how rare the events are. The relative economic value <sup>64</sup> of warnings is the fraction of the saving of a perfect forecast that a user who protects at a cost C against a loss L obtains, as a function of C/L; for Foehn it is maximized over probability thresholds on the 2022 data, a retrospective potential value. Differences between systems are tested with a paired moving-block bootstrap over start dates <sup>65</sup> (blocks of seven days, 10,000 resamples, 95% intervals).

Relative gains are computed from unrounded scores: at a lead time of 1 day at the stations, the RMSE of Foehn is $1 2 . 6 8 \mu \mathrm { g } \mathrm { m } ^ { - 3 } ,$ against 15.23 for the CAMS forecast and 15.72 for Aurora (reductions of 16.7% and 19.3%). At the 2,029 European stations, on the same station-days, the climatology of GHAP at the station pixel has an RMSE of 11.0 µg m<sup>−3</sup>, against 9.8 for Foehn 1 day ahead, 10.7 for the CAMS forecast and 10.8 for Aurora.

At the stations, with the thresholds used over land (the local 99th percentile of GHAP in 2017–2020 at the station pixel), the SEDI of the ensemble mean of Foehn is 0.45 at a lead time of 1 day and 0.34 at 4 days, against 0.33 and 0.25 for the CAMS forecast and 0.34 and 0.26 for Aurora, and the same holds at the 95th percentile. With each station’s own 99th percentile of 2022, the three systems are equal at 1 day (0.44– 0.45), and the CAMS forecast and Aurora score higher from 2 days onwards (0.45 and 0.48, against 0.35). The measurement of the start day at the station, which exists only where there is a station, scores higher than every forecast with the thresholds of GHAP (0.73 at 1 day and 0.47 at 4 days).

## Diagnostics and case studies

A resident is hidden by the 40 km grid when the $\mathrm { P M } _ { 2 . 5 }$ of their 1 km pixel exceeds a threshold and the mean of their 40 km cell, a 40 × 40-pixel block like the $0 . 4 ^ { \circ }$ cells of CAMS, does not. The value of the proposed particle transport is the reduction of the CRPS against GHAP of the base model relative to the same network trained without the transport channels. The value of the emissions is the reduction of the RMSE of the ensemble mean and of the CRPS of the model with emissions relative to the base model. In the three regions not shown in Extended Data Fig. 9, the gain of the emissions in the RMSE of the ensemble mean is 7–11% in Oceania and 3–5% in Southeast Asia from one to seven days ahead, and −3 to −1% in the Middle East. In the North China Plain (34–41<sup>◦</sup>N, 112–120<sup>◦</sup>E) and the Indo-Gangetic Plain (24–31<sup>◦</sup>N, 74–88<sup>◦</sup>E), it is between −0.1 and 3.2% and within ±1.5%, respectively, with 95% intervals that include zero at every lead time. After a week, the spread–skill ratio of the model with emissions is 0.51, against 0.43 for the base model; it is within 0.02 of the base model in Europe, North America and East Asia but 0.12 higher in Africa, so part of the gain in CRPS at long lead times comes from a wider ensemble in Africa, where the emissions do not improve the ensemble mean.

Each case was chosen by a rule fixed before any forecast of it was examined. The global day (Fig. 3) and the exposure case (Fig. 2) are the day of 2022 with the highest population-weighted GHAP (21 January). The extreme-event case (Fig. 4) is the day of that January with the highest mean at the official stations of Beijing–Tianjin–Hebei (25 January); the four basins of Fig. 3b are each monitored by a different official network. Among twenty large cities fixed in advance, the other exposure cases (Extended Data Fig. 4) are the city-days with the most residents above 75 µg m<sup>−3</sup> in 40 km cells below it (Cairo, 2 January; Santiago de Chile, 20 June), and the transport case (Fig. 5) is the African arrival day with the largest rise of PM carried by the air (Kano, 16 May). The smoke case (Fig. 6) concerns the largest fires of 2022 in France, and its day, 18 July, has the highest GHAP of 13–26 July around the Réserve Ornithologique du Teich.

## Use of generative AI

An LLM-based coding assistant was used to write analysis and figure code, to restructure and draft the manuscript and to cross-check its numbers against the result files. The authors reviewed all AI-assisted text, code and claims against the code and result files and take full responsibility for the content of this work.

## Data availability

All data used are public. GHAP daily 1 km PM<sub>2.5</sub> (version 1) is available from Zenodo (Wei et al.<sup>6</sup>); ERA5, the CAMS global forecasts and analyses, CAMS-GLOB-ANT and GFAS from the Copernicus Climate and Atmosphere Data Stores; the IFS HRES forecasts from WeatherBench $2 ^ { 6 6 } ;$ GHS-POP R2023A from the Joint Research Centre; elevation from the ETOPO 2022 global relief model of NOAA (30 arc-second ice-surface grid) <sup>67,68</sup>; and the station measurements from the European Environment Agency, the China National Environmental Monitoring Centre <sup>39</sup>, the US Environmental Protection Agency (Air Quality System), AirKorea, the National Institute for Environmental Studies of Japan and the Central Pollution Control Board of India, partly through ${ \mathrm { O p e n A Q } } ^ { 6 9 }$

## Code availability

The code of FLEXGREEN, of the training and inference of Foehn and of every analysis and figure is available at https://github.com/AmmarKheder/Foehn under the MIT licence, and the version used here (v1.0.0) is archived at Zenodo (https://doi.org/10.5281/ze nodo.23337751) and in the Software Heritage archive (SWHID swh:1:rev:b523bc369b9a68e730ee5a29ca f002ba2d83c422;origin=https://github.com/A mmarKheder/Foehn). The trained weights of the five models evaluated here, with their checksums and the inference commands, are archived at Zenodo (https: //doi.org/10.5281/zenodo.23337591); access to the files is restricted to the editors and referees during peer review and will be opened upon publication.

Acknowledgements. This work was supported by the Finnish Ministry of Education and Culture’s Pilot for Doctoral Programmes (Pilot project Mathematics of Sensing, Imaging and Modelling). We acknowledge the ACCC Flagship, funded by the Academy of Finland (grant no. 337549). We acknowledge CSC – IT Center for Science, Finland (https://csc.fi), for awarding this project generous access to the LUMI supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CSC (Finland) and the LUMI consortium. Thanks to the city of Lahti (Finland) for financial support. We thank the providers of the data listed in Data availability.

Author contributions. A.K. conceived and designed the study, developed Foehn and the FLEXGREEN transport code, ran the experiments, analysed the results and wrote the paper. Z.-S.L. supervised the machine learning development and evaluation, contributed to methodology development and manuscript writing. M.B. supervised the atmospheric science aspects, provided overall project direction, and secured funding. All authors reviewed and edited the manuscript.

Competing interests. The authors declare no competing interests.

Correspondence. Correspondence and requests for materials should be addressed to Ammar Kheder (ammar.kheder@lut.fi).

## References

[1] Erin E. McDuffie, Randall V. Martin, Joseph V. Spadaro, Richard Burnett, Steven J. Smith, Patrick O’Rourke, Melanie S. Hammer, Aaron van Donkelaar, Liam Bindle, Viral Shah, Lyatt Jaeglé, Gan Luo, Fangqun Yu, Jamiu A. Adeniran, Jintai Lin, and Michael Brauer. Source sector and fuel contributions to ambient $\mathrm { P M } _ { 2 . 5 }$ and attributable mortality across multiple spatial scales. Nature Communications, 12(1):3594, 2021. doi: 10.1038/s41467-021-238 53-y.

[2] Wenhua Yu, Rongbin Xu, Tingting Ye, Michael J. Abramson, Lidia Morawska, Bin Jalaludin, et al. Estimates of global mortality burden associated with short-term exposure to fine particulate matter $( \mathrm { P M } _ { 2 . 5 } )$ . The Lancet Planetary Health, 8(3):e146–e155, 2024. doi: 10.1016/S2542-5196(24)00003-2.

[3] Vincent-Henri Peuch, Richard Engelen, Michel Rixen, Dick Dee, Johannes Flemming, Martin Suttie, Melanie Ades, Anna Agustí-Panareda, Cristina Ananasso, Erik Andersson, et al. The Copernicus Atmosphere Monitoring Service: from research to operations. Bulletin of the American Meteorological Society, 103 (12):E2650–E2668, 2022. doi: 10.1175/BAMS-D-21-0314.1.

[4] Remi Lam, Alvaro Sanchez-Gonzalez, Matthew Willson, Peter Wirnsberger, Meire Fortunato, Ferran Alet, Suman Ravuri, Timo Ewalds, Zach Eaton-Rosen, Weihua Hu, et al. Learning skillful medium-range global weather forecasting. Science, 382(6677): 1416–1421, 2023. doi: 10.1126/science.adi2336.

[5] Ilan Price, Alvaro Sanchez-Gonzalez, Ferran Alet, Tom R. Andersson, Andrew El-Kadi, Dominic Masters, Timo Ewalds, Jacklynn Stott, Shakir Mohamed, Peter Battaglia, Remi Lam, and Matthew Willson. Probabilistic weather forecasting with machine learning. Nature, 637:84–90, 2025. doi: 10.1038/s41586-0 24-08252-9.

[6] Jing Wei, Zhanqing Li, Alexei Lyapustin, Jun Wang, Oleg Dubovik, Joel Schwartz, Lin Sun, Chi Li, Song Liu, and Tong Zhu. First close insight into global daily gapless 1 km PM pollution, variability, and health impact. Nature Communications, 14:8349, 2023. doi: 10.1038/s41467-023-43862-3.

[7] GBD 2019 Risk Factors Collaborators. Global burden of 87 risk factors in 204 countries and territories, 1990–2019: a systematic analysis for the Global Burden of Disease Study 2019. The Lancet, 396(10258):1223–1249, 2020. doi: 10.1016/S0140-6736(20 )30752-2.

[8] M. Hallquist, J. C. Wenger, U. Baltensperger, Y. Rudich, D. Simpson, M. Claeys, J. Dommen, N. M. Donahue, C. George, A. H. Goldstein, et al. The formation, properties and impact of secondary organic aerosol: current and emerging issues. Atmospheric Chemistry and Physics, 9(14):5155–5236, 2009. doi: 10.5194/acp-9-5155-2009.

[9] O. V. Sanderfoot, S. B. Bassing, J. L. Brusa, R. L. Emmet, S. J. Gillman, K. Swift, and B. Gardner. A review of the effects of wildfire smoke on the health and behavior of wildlife. Environmental Research Letters, 16(12):123003, 2021. doi: 10.1088/1748-9326/ac30f6.

[10] Anna Nihei, Olivia V. Sanderfoot, Katie LaBarbera, and Morgan W. Tingley. Wildfire smoke impacts the body condition and capture rates of birds in California. Ornithology, 141(4):ukae023, 2024. doi: 10.1093/ornithology/ukae023.

[11] Joshua S. Apte, Kyle P. Messier, Shahzad Gani, Michael Brauer, Thomas W. Kirchstetter, Melissa M. Lunden, Julian D. Marshall, Christopher J. Portier, Roel C. H. Vermeulen, and Steven P. Hamburg. High-resolution air pollution mapping with Google Street View cars: Exploiting big data. Environmental Science & Technology, 51(12):6999–7008, 2017. doi: 10.1021/acs.est.7b00891.

[12] Anjum Hajat, Charlene Hsia, and Marie S. O’Neill. Socioeconomic disparities and air pollution exposure: a global review. Current Environmental Health Reports, 2(4):440–450, 2015. doi: 10.1007/s40572-015-0069-5.

[13] Christopher W. Tessum, David A. Paolella, Sarah E. Chambliss, Joshua S. Apte, Jason D. Hill, and Julian D. Marshall. $\mathrm { P M } _ { 2 . 5 }$ polluters disproportionately and systemically affect people of

color in the United States. Science Advances, 7(18):eabf4491, 2021. doi: 10.1126/sciadv.abf4491.

[14] Randall V. Martin, Michael Brauer, Aaron van Donkelaar, Gavin Shaddick, Urvashi Narain, and Sagnik Dey. No one knows which city has the highest concentration of fine particulate matter. Atmospheric Environment: X, 3:100040, 2019. doi: 10.101 6/j.aeaoa.2019.100040.

[15] Kaifeng Bi, Lingxi Xie, Hengheng Zhang, Xin Chen, Xiaotao Gu, and Qi Tian. Accurate medium-range global weather forecasting with 3D neural networks. Nature, 619:533–538, 2023. doi: 10.1038/s41586-023-06185-3.

[16] Shuo Wang, Yanran Li, Jiang Zhang, Qingye Meng, Lingwei Meng, and Fei Gao. PM<sub>2.5</sub>-GNN: A domain knowledge enhanced graph neural network for $\mathrm { P M } _ { 2 . 5 }$ forecasting. In Proceedings of the 28th International Conference on Advances in Geographic Information Systems (SIGSPATIAL), pages 163–166, 2020. doi: 10.1145/3397536.3422208.

[17] Yuxuan Liang, Yutong Xia, Songyu Ke, Yiwei Wang, Qingsong Wen, Junbo Zhang, Yu Zheng, and Roger Zimmermann. Air-Former: Predicting nationwide air quality in China with transformers. Proceedings of the AAAI Conference on Artificial Intelligence, 37(12):14329–14337, 2023. doi: 10.1609/aaai.v37i12.26676.

[18] Cristian Bodnar, Wessel P. Bruinsma, Ana Lucic, Megan Stanley, Anna Allen, Johannes Brandstetter, Patrick Garvan, Maik Riechert, Jonathan A. Weyn, Haiyu Dong, et al. A foundation model for the Earth system. Nature, 641:1180–1187, 2025. doi: 10.1038/s41586-025-09005-y.

[19] Ke Gui, Xutao Zhang, et al. Advancing operational global aerosol forecasting with machine learning. Nature, 651:658–665, 2026. doi: 10.1038/s41586-026-10234-y.

[20] Paula Harder, Johannes Flemming, Mihai Alexe, Gert Mertes, Baudouin Raoult, and Matthew Chantry. AIFS-COMPO: A global data-driven atmospheric composition forecasting system. arXiv preprint arXiv:2604.03300, 2026.

[21] Andreas Stohl, Caroline Forster, Andreas Frank, Petra Seibert, and Gerhard Wotawa. Technical note: The Lagrangian particle dispersion model FLEXPART version 6.2. Atmospheric Chemistry and Physics, 5:2461–2474, 2005. doi: 10.5194/acp-5-2461-2005.

[22] Ignacio Pisso, Espen Sollum, Henrik Grythe, Nina I. Kristiansen, Massimo Cassiani, Sabine Eckhardt, Delia Arnold, Don Morton, Rona L. Thompson, Christine D. Groot Zwaaftink, et al. The Lagrangian particle dispersion model FLEXPART version 10.4. Geoscientific Model Development, 12:4955–4997, 2019. doi: 10.519 4/gmd-12-4955-2019.

[23] Johannes W. Kaiser, Angelika Heil, Meinrat O. Andreae, et al. Biomass burning emissions estimated with a global fire assimilation system based on observed fire radiative power. Biogeosciences, 9:527–554, 2012. doi: 10.5194/bg-9-527-2012.

[24] Antonin Soulié, Claire Granier, Sabine Darras, et al. Global anthropogenic emissions (CAMS-GLOB-ANT) for the Copernicus Atmosphere Monitoring Service simulations of air quality forecasts and reanalyses. Earth System Science Data, 16(5):2261–2279, 2024. doi: 10.5194/essd-16-2261-2024.

[25] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations (ICLR), 2023.

[26] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In International Conference on Learning Representations (ICLR), 2023.

[27] Delin Li, Zhiqiang Chen, Gang Huang, and Jianjun Xu. Global and regional performance of the Aurora foundation model for surface $\mathrm { P M } _ { 2 . 5 }$ prediction. Environmental Research Letters, 21(15): 154018, 2026. doi: 10.1088/1748-9326/ae90c2.

[28] Jesús San-Miguel-Ayanz, Tracy Durrant, Roberto Boca, Pieralberto Maianti, Giorgio Libertà, Duarte Oom, Alfredo Branco, Daniele de Rigo, Maria Suarez-Moreno, Davide Ferrari, Elena Roglia, Nicola Scionti, Marco Broglia, Marco Onida, Adrian

Tistan, and Peter Löffler. Forest fires in Europe, Middle East and North Africa 2022. Technical Report JRC135226, EUR 31692 EN, Publications Office of the European Union, Luxembourg, 2023.

[29] Z.-S. Liu, M. Boy, and R. Makkonen. Tailweather: from tail to extremes, a global climatological dataset for machine-learning weather forecasting. Earth System Science Data Discussions, 2026: 1–29, 2026. doi: 10.5194/essd-2026-792. URL https://essd.c opernicus.org/preprints/essd-2026-792/.

[30] Christopher A. T. Ferro and David B. Stephenson. Extremal dependence indices: improved verification measures for deterministic forecasts of rare binary events. Weather and Forecasting, 26(5):699–713, 2011. doi: 10.1175/WAF-D-10-05030.1.

[31] Eloise A. Marais and Christine Wiedinmyer. Air quality impact of diffuse and inefficient combustion emissions in Africa (DICE-Africa). Environmental Science & Technology, 50(19):10739–10745, 2016. doi: 10.1021/acs.est.6b02602.

[32] Eri Saikawa, Marcus Trail, Min Zhong, Qianru Wu, Cindy L. Young, Greet Janssens-Maenhout, Zbigniew Klimont, Fabian Wagner, Jun-ichi Kurokawa, Ajay Singh Nagpure, and Bhola Ram Gurjar. Uncertainties in emissions estimates of greenhouse gases and air pollutants in India and their impacts on regional air quality. Environmental Research Letters, 12(6): 065002, 2017. doi: 10.1088/1748-9326/aa6cb4.

[33] Huirong Yang, Kai Wu, Huizhong Shen, Greet Janssens-Maenhout, Monica Crippa, Diego Guizzardi, and Minqiang Zhou. A comparative analysis of China’s anthropogenic CO<sub>2</sub> emissions (2000–2023): insights from six bottom-up inventories and uncertainty assessment. Atmospheric Chemistry and Physics, 25:18111–18127, 2025. doi: 10.5194/acp-25-18111-2025.

[34] A. Stidworthy, M. Jackson, K. Johnson, D. Carruthers, and J. Stocker. Evaluation of local and regional air quality forecasts for London. International Journal of Environment and Pollution, 64 (1/2/3), 2018. doi: 10.1504/IJEP.2018.099160.

[35] Cambridge Environmental Research Consultants. airTEXT: air quality forecasts for London. https://cerc.co.uk/forecas ting/air-quality/united-kingdom.html, 2026. Accessed October 2026.

[36] B. Zheng, D. Tong, M. Li, F. Liu, C. Hong, G. Geng, H. Li, X. Li, L. Peng, J. Qi, L. Yan, Y. Zhang, H. Zhao, Y. Zheng, K. He, and Q. Zhang. Trends in China’s anthropogenic emissions since 2010 as the consequence of clean air actions. Atmospheric Chemistry and Physics, 18:14095–14111, 2018. doi: 10.5194/acp-1 8-14095-2018.

[37] Hans Hersbach, Bill Bell, Paul Berrisford, Shoji Hirahara, András Horányi, Joaquín Muñoz-Sabater, et al. The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society, 146(730):1999–2049, 2020. doi: 10.1002/qj.3803.

[38] Marcello Schiavina, Sergio Freire, and Kytt MacManus. GHS-POP R2023A: GHS population grid multitemporal (1975–2030). European Commission, Joint Research Centre (JRC), 2023.

[39] China National Environmental Monitoring Centre. National urban air quality real-time publishing platform. https://air. cnemc.cn, 2026. Accessed October 2026.

[40] Petra Seibert and Andreas Frank. Source-receptor matrix calculation with a Lagrangian particle dispersion model in backward mode. Atmospheric Chemistry and Physics, 4:51–63, 2004. doi: 10.5194/acp-4-51-2004.

[41] Sverre Petterssen. Weather Analysis and Forecasting. McGraw-Hill, New York, 1940.

[42] D. H. P. Vogelezang and A. A. M. Holtslag. Evaluation and model impacts of alternative boundary-layer height formulations. Boundary-Layer Meteorology, 81:245–269, 1996.

[43] Steven R. Hanna. Applications in air pollution modeling. In F. T. M. Nieuwstadt and H. van Dop, editors, Atmospheric Turbulence and Air Pollution Modelling, pages 275–310. Reidel, Dordrecht, 1982.

[44] D. J. Thomson. Criteria for the selection of stochastic models of particle trajectories in turbulent flows. Journal of Fluid Mechanics, 180:529–556, 1987.

[45] J. A. Businger, J. C. Wyngaard, Y. Izumi, and E. F. Bradley. Fluxprofile relationships in the atmospheric surface layer. Journal of the Atmospheric Sciences, 28:181–189, 1971.

[46] Roland R. Draxler and G. D. Hess. An overview of the HYS-PLIT\_4 modelling system for trajectories, dispersion and deposition. Australian Meteorological Magazine, 47:295–308, 1998.

[47] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention (MICCAI), volume 9351 of Lecture Notes in Computer Science, pages 234–241. Springer, 2015. doi: 10.1007/978-3-319-24574-4 \_28.

[48] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, pages 5998–6008, 2017.

[49] Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron Courville. FiLM: Visual reasoning with a general conditioning layer. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018.

[50] Michael S. Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In International Conference on Learning Representations (ICLR), 2023.

[51] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 13213–13232, 2023.

[52] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 12606–12633, 2024.

[53] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.

[54] Leslie N. Smith and Nicholay Topin. Super-convergence: very fast training of neural networks using large learning rates. In Artificial Intelligence and Machine Learning for Multi-Domain Operations Applications, volume 11006 of Proceedings of SPIE, page 1100612, 2019. doi: 10.1117/12.2520589.

[55] Simon Lang, Mihai Alexe, Mariana C. A. Clare, Christopher Roberts, Rilwan Adewoyin, et al. AIFS-CRPS: ensemble forecasting using a model trained with a loss function based on the continuous ranked probability score. npj Artificial Intelligence, 2: 18, 2026. doi: 10.1038/s44387-026-00073-7.

[56] Christopher A. T. Ferro. Fair scores for ensemble forecasts. Quarterly Journal of the Royal Meteorological Society, 140(683): 1917–1923, 2014. doi: 10.1002/qj.2270.

[57] Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Waikin Wong, and Wang-chun Woo. Convolutional LSTM network: A machine learning approach for precipitation nowcasting. In Advances in Neural Information Processing Systems (NeurIPS), volume 28, pages 802–810, 2015.

[58] Zhangyang Gao, Cheng Tan, Lirong Wu, and Stan Z Li. SimVP: Simpler yet better video prediction. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3170– 3180, 2022. doi: 10.1109/CVPR52688.2022.00317.

[59] Zhihan Gao, Xingjian Shi, Hao Wang, Yi Zhu, Yuyang Wang, Mu Li, and Dit-Yan Yeung. Earthformer: Exploring spacetime transformers for earth system forecasting. In Advances in Neural Information Processing Systems (NeurIPS), volume 35, pages 25390–25403, 2022.

[60] Tung Nguyen, Johannes Brandstetter, Ashish Kapoor, Jayesh K Gupta, and Aditya Grover. ClimaX: A foundation model for weather and climate. In Proceedings of the 40th International Conference on Machine Learning, volume 202, pages 25904–25938. PMLR, 2023.

[61] Ammar Kheder, Helmi Toropainen, Wenqing Peng, Samuel Antão, Jia Chen, Michael Boy, and Zhi-Song Liu. TopoFlow: topography-aware pollutant Flow learning for high-resolution air quality prediction. npj Climate and Atmospheric Science, 9:143, 2026. doi: 10.1038/s41612-026-01417-5.

[62] Tilmann Gneiting and Adrian E. Raftery. Strictly proper scoring rules, prediction, and estimation. Journal of the American Statistical Association, 102(477):359–378, 2007. doi: 10.1198/0162 14506000001437.

[63] V. Fortin, M. Abaza, F. Anctil, and R. Turcotte. Why should ensemble spread match the RMSE of the ensemble mean? Journal of Hydrometeorology, 15(4):1708–1713, 2014. doi: 10.1175/JHM-D -14-0008.1.

[64] David S. Richardson. Skill and relative economic value of the ECMWF ensemble prediction system. Quarterly Journal of the Royal Meteorological Society, 126(563):649–667, 2000. doi: 10.1002/qj.49712656313.

[65] Hans R. Künsch. The jackknife and the bootstrap for general stationary observations. The Annals of Statistics, 17(3):1217–1241, 1989. doi: 10.1214/aos/1176347265.

[66] Stephan Rasp, Stephan Hoyer, Alexander Merose, Ian Langmore, Peter Battaglia, Tyler Russell, Alvaro Sanchez-Gonzalez, Vivian Yang, Rob Carver, Shreya Agrawal, et al. WeatherBench 2: A benchmark for the next generation of data-driven global weather models. Journal of Advances in Modeling Earth Systems, 16(6):e2023MS004019, 2024. doi: 10.1029/2023MS004019.

[67] NOAA National Centers for Environmental Information. ETOPO 2022 15 arc-second global relief model, 2022. 30 arcsecond ice-surface grid, version 1.

[68] Michael MacFerrin, Christopher Amante, Kelly Carignan, Matthew Love, and Elliot Lim. The earth topography 2022 (ETOPO 2022) global DEM dataset. Earth System Science Data, 17(5):1835–1849, 2025. doi: 10.5194/essd-17-1835-2025.

[69] OpenAQ. OpenAQ: open air quality data. https://openaq.o rg, 2026. Accessed October 2026.

## Extended Data

![](images/44b2612746f3b2c3135b9d3b8402a530a464c15e1ab6a8a4b96cad1f4a6774e5.jpg)

Extended Data Fig. 1 | Over Europe, Foehn is 26 times faster than Aurora, with a 35% lower error at 1 km. a, Power of the forecast error relative to the power of the GHAP anomaly, by wavelength, at 0.25<sup>◦</sup> over central Europe (61 start days of 2022, three days ahead): below one, skill at that wavelength; at one, no skill; above one, structure in the wrong place. b, The same at 1 km on the regional window from Paris to Florence, for the models trained natively at 1 km and for the continental 0.25<sup>◦</sup> models interpolated to 1 km. c, RMSE at 1 km (mean over leads one to four days, Europe 2022) against the time to produce one member of one forecast on one MI250X compute die; the marker area is the peak GPU memory. Foehn is 26 times faster than Aurora with 35% lower error; the inset enlarges the boxed region.

![](images/502b92dba5e5ce65b38e7398db771b39f739ac25bbb0e858f571a0f9aba8f131.jpg)  
Extended Data Fig. 2 | Over Europe, Foehn’s network is more accurate than five networks retrained with identical inputs, the CAMS forecast and Aurora. Root-mean-square error of the forecasts of 2022 against the 0.25<sup>◦</sup> block means (a), the 1 km pixels (b) and 2,971 European Environment Agency stations (c), and critical success index of exceedances of $2 5 \mu \mathrm { g } \mathrm { m } ^ { - 3 } \left( \mathbf { d } \right) .$ , versus lead time: Foehn’s flow-matching network (its 0.25<sup>◦</sup> European version) with and without the learned transport layer, ConvLSTM, Earthformer, SimVP, TopoFlow and ClimaX retrained with identical inputs, the CAMS operational forecast, Aurora and climatology. Means over three seeds.

![](images/7f1c32d2177ce55d5a21032d5861a5086401001e5bd650ee67ed5398e9ab600c.jpg)  
Extended Data Fig. 3 | Over Europe, Foehn scores better than each other system in 319 of 324 comparisons, on all days, polluted days and transported episodes. Scorecard of the change of Foehn’s score relative to each system (rows), for the root-mean-square error (RMSE), the continuous ranked probability score (CRPS; absolute error for point forecasts) and the critical success index of exceedances of 25 µg m<sup>−3</sup> (CSI, sign reversed so that blue always means Foehn better), one to four days ahead, over Europe in 2022 at 0.25<sup>◦</sup>: on all days (left), on the 10% of days with the largest polluted area (centre) and during the four documented transported episodes (right). Means over three seeds; Foehn is better in 319 of the 324 cells, and the five others are within 2%. Systems as in Extended Data Fig. 2.

a <sub>Cairo,</sub> <sub>GHAP</sub> <sub>(1</sub> <sub>km),</sub> <sub>2</sub> <sub>Jan</sub> <sub>2022</sub>  
![](images/69ab1cf871c6feaf6259aee546e34af728c773dd11e392f3608007c476d209be.jpg)

b  
![](images/82222d32b91d41c3618af9e40e438125851ede7b496d782531ad45b7591d4df2.jpg)

c <sub>Residents</sub> <sub>above</sub> <sub>75</sub> <sub>µg</sub> <sub>m</sub>−3  
![](images/867f53248889a1a825adae554033896c39ac503c9f1bc618a8261ce8ba139dd6.jpg)

d <sub>Santiago,</sub> <sub>GHAP</sub> <sub>(1</sub> <sub>km),</sub> <sub>20</sub> <sub>Jun</sub> <sub>2022</sub>  
![](images/42bd1c47b1c4c39bec98736fdc36995d084b8abb901692c63384bd8aa3298146.jpg)

e <sub>The</sub> <sub>same</sub> <sub>field</sub> <sub>on</sub> <sub>40</sub> <sub>km</sub> <sub>cells</sub>  
![](images/a42a59ffdeb1dd04542a47e1b729417f6dc835c6df2f11e7c9e31a5cbe52daf5.jpg)

f <sub>Residents above 75 µg m</sub>−3  
![](images/2f1882bb6620b5831064605135482e5a5af0e3907549e5d18181f231360e8172.jpg)

Days above the local P99g  
![](images/628d71ad24409e23ab92f384864a4d9d269f4e32d32c3b7a5f0063e454099dd5.jpg)

By regionh  
![](images/07131d0f7dfd82131cb3316f613c34b125254355b8a77c95b6ca5c0ba289c9ec.jpg)

![](images/02da133d0aa84c1db7f6ab0192bb648254ccffb7dbb8b1c9927caf4350a1f33a.jpg)

![](images/1c8506c4bb5e879b32fecd45d3bbcdb9c0eedcaaf8daf5e5c4892b356f0143a8.jpg)  
Foehn CAMS Aurora 1 day ahead 3 days ahead

Extended Data Fig. 4 | Coarse grids hide whole cities from warnings, and Foehn’s kilometre-scale forecasts detect local extremes earlier and with more value. a–c, Cairo on 2 January 2022 and d–f, Santiago de Chile on 20 June 2022, the largest hidden city-days of Africa and of the Americas among twenty large cities fixed in advance (Methods), as in Fig. 2b–d: GHAP at 1 km with the contour o $: 7 5 \mu \mathrm { g } \mathrm { m } ^ { - 3 } .$ , the same field on 40 km cells, and the residents above $7 5 \mu \mathrm { g } \mathrm { m } ^ { - 3 }$ seen (grey) or hidden (pink) by the 40 km cells. Around Cairo, 16.6 million residents were above the threshold and every one of them hidden; around Santiago, 5.8 million, all hidden, the basin of the city averaged with the surrounding slopes of the Andes and the coastal range. g, h, Detection of local extremes over land in 2022, the days above the local 99th percentile of $2 0 1 \mathsf { \bar { 7 } } - 2 0 2 0 \mathsf { : }$ symmetric extremal dependence index (SEDI) by lead time (g; the CAMS and Aurora forecasts cover 4 days) and by region 1 day ahead (h), for Foehn’s ensemble mean and the CAMS and $\mathrm { \ A u r o r a ^ { 1 8 } }$ forecasts issued at the same time. $\mathbf { i } , \mathbf { j } ,$ Retrospective potential relative economic value of warnings of days above the local 95th (i) and 99th (j) percentiles at the monitoring stations, against the ratio of the cost of protection to the loss avoided, 1 day (solid) and 3 days ahead (dashed).

GHAP, 1 km (reference)a  
![](images/24d7ad3fe50ac4eb95e439fc6a4992dc5467b8dfe8364d0aed74d2770363f7f7.jpg)  
b

![](images/4f2dc9a422522d3de0ed28bcc923103a5b5bc5edab8dca92da6846817655ae56.jpg)

![](images/3271a1cc464c1d9a897a242b06c8a99965677c1ce30b3d67e755a34939f9ac82.jpg)  
g

![](images/3380087d7dba65080fc8b6a7f857fc0c9f921806bea84ab991c841890564cd38.jpg)

![](images/ce48dbc3e9e5287a6306150870e7b344daa62573ecc491b7dd30b8af9a36b60e.jpg)  
c

![](images/b5feb7d6766fc38368da92ed22d5ef29b13e337b00d7368a6413590d1f34e01d.jpg)

h  
![](images/8bc127ee12856584f9c922d366f2f7fbad5b66894d0724d0e76c96ef87e3e837.jpg)  
q

![](images/7ebd8c70f9fe2fd0a5d54ebbccd3604abc02a3a006d933e8840fe40e1fd23238.jpg)

![](images/d07f5a1f6c54a4a82f34c567cf9b2dcc28dfd4d07e72bcc21f570eba377afe5b.jpg)

![](images/e91151b10567ffa5b7714ce8b908c4750dd002d2f0eb342a2c52f3ae459d00d8.jpg)

![](images/40bd6685761271e1b6c9a4d094d5f17423367fb703330d3afa26ec41fbca0c52.jpg)

r  
d  
![](images/bae7740deea5ad5a50757e598694a1aab799a7308069bbde2881ba23146df685.jpg)

![](images/193016782102f496c72785c277b56b6895186654732fba05a3c7f7ce21ab1a32.jpg)

![](images/563ec89bc14ba08d011a080620bd7245de96c67ec5cb3175d5a265f8b35b8903.jpg)

s  
![](images/a7bc74458afe263195c4ef5058162f828f3cad0f9419b2c4d15d17b1e7aba6c8.jpg)

![](images/b6bfcabf22e75c7f00a858993c1014f2f1686d783efea7c2d9f3e212ccfd9dbe.jpg)

![](images/74967739dbd4fccd3f3475a08aeb582a2c2d6ade01c8a40e644ad23f159ae2f9.jpg)

![](images/e6a83793160ece06c9b92022b5b821a4fa96890ee4759f645d099868b815172f.jpg)

![](images/bc07ef3f01257b63571cb30334aa57f5e6ab969042a542635d2280e210b5628b.jpg)

t  
![](images/d8f93a054717602ae1770be47c4b447e5577a2b60e29a587476bce98b336b38d.jpg)  
Extended Data Fig. 5 | On 21 January 2022, Foehn resolves at 1 km the structure of four polluted regions that the 0.4<sup>◦</sup> forecasts blur. Daily mean $\mathrm { P M } _ { 2 . 5 }$ on 21 January 2022 over the North China Plain, the Indo-Gangetic Plain around Delhi, the Po Valley, and Seoul and the Han basin (regions chosen before the comparison): the 1 km estimates (first column), the CAMS forecast and Aurora issued at 00 UTC on 21 January $( 0 . 4 ^ { \circ } )$ , Foehn’s ensemble mean issued at the end of 20 January (1 km), and Foehn minus the reference. Each row has its own colour range (2nd–98th percentiles of the reference). Numbers: mean absolute error (MAE, $\imath \mathrm { g } \mathrm { m } ^ { - 3 } )$ and spatial correlation against the reference over the land pixels of the panel, the $0 . 4 ^ { \circ }$ forecasts read at the 1 km pixels.

![](images/38f6f70c6912fb9926018617ca3b002cfc9752a2ab5e3bdbc4e81a8f8a3c7a4a.jpg)

![](images/54bc5f4739e4ab6debda6e4c99e9fdcdc6a81289559daf0833ad6346d770d6a5.jpg)

![](images/75924ce807995b41ded9ba51ef5d95499a47ea648925e014318d851e14467b64.jpg)

![](images/44558b5383e211247096b16572f42d44f71f1573c0102f72cd58a3e906470bb9.jpg)

![](images/ae61770fce5ad2a7a5061b6bd7b8c13e75af7e1dacc157ca7ff7e5c5c6c8fed6.jpg)

![](images/7f9b561c9d25379508643658ceb478a86abbb34bc49a6629f072b419deed35bd.jpg)

![](images/fa81ed5d370ec008140e849d883bc09569e2f380843be2048541a502b290019a.jpg)

![](images/18fddfa1925e4094ad4169d9a16beb128320dbbd5bb07d519580f1b4b3f3c977.jpg)  
Extended Data Fig. 6 | At the stations of six official networks, Foehn’s error 4 days ahead is 12% lower than that of the CAMS forecast and 17% lower than Aurora’s. $\mathbf { a - f } ,$ Root-mean-square error (RMSE) of the daily $\mathrm { P M } _ { 2 . 5 }$ forecasts of 2022 by lead time at the stations of each official network, on identical station-days: Foehn (ensemble mean at the 1 km pixel of the station), the CAMS forecast and Aurora issued at the same time (end of the start day), and the climatology of the 1 km estimates at the station pixel (grey dashed; Methods); the bias of Foehn, CAMS and Aurora one day ahead is written in each panel $( \mu \mathrm { g } \mathrm { m } ^ { - 3 } )$ . Read at the station pixel on the same day, the 1 km estimates themselves have an RMSE of 2.4 (a), 2.6 (b), 11.2 (c), 8.5 (d), 6.6 (e) and 36.5 µg m $^ { - 3 } ( \mathbf { f } ;$ Methods). Over all stations, Foehn’s RMSE four days ahead is 14.4 µg m $^ { - 3 } ,$ against 16.4 for the CAMS forecast and 17.3 for Aurora, 12% (95% interval 7–19%) and 17% (13–22%) lower. g, All 6,324 stations: Foehn read at its 1 km pixel and averaged to $0 . 2 5 ^ { \circ }$ and $0 . 5 ^ { \circ }$ before being read like the baselines (equal resolution); averaged to $0 . 5 ^ { \circ } .$ , its error one day ahead is 13.5 µg m<sup>−3</sup>. h, Ratio of the ensemble spread to the RMSE of the ensemble mean by lead time: the 1 km model on the validation year 2021 against the 1 km estimates (four members, every eighth start day), after (solid) and before (dashed) the CRPS training, and Foehn trained in Aurora’s protocol on May–November 2022 against the CAMS analysis (eight members); 1 for a calibrated ensemble.

a  
![](images/e88c259e236149cf0713d3d4aae419e0b4489cdc9db784097c0239fe14b42e3d.jpg)

b  
![](images/dedd4034c00bd7c098a4efe95ff0faeb8604bae15fdb0edfa9e0d1495b4c9754.jpg)

![](images/ec06c44cb119b88fc5fbd5572bc4a31612f1dc50d09843501ca6084310ec5a65.jpg)

![](images/83ea7ff4e01fae940b98d225a20ed57c746733fbd2465960103db8d33ef17af6.jpg)

![](images/751610cd1f1a9a897d2ceefe7bff01dac3f83f0679a1c45f45fd05478608e114.jpg)

![](images/021fdfcc70aea2835d5450444c2c77ca6e1e30e82cb7ddd532bd6cfad717f123.jpg)

![](images/197e82ee025305c40d5a382db33fd59a90fb45216af6e119569c970dbcae5e15.jpg)

![](images/7c5c03366636d86d1714ba94ee53fd88d4fa5fc54233759e42796c9aa1e8f437.jpg)

![](images/0db88904ddf4de53eb81343b81d4c93cdda8fb526d7d286d6e66e53dd74d013c.jpg)

![](images/37f5c3f73003ce8f82f3802bc995f720179eaf8cc0b932ea8e33c03bb704b67e.jpg)

![](images/30974870ef0d7428e85349bd719b4bd6593283af1cda61101717138341ff0765.jpg)

$$
\mathrm { \_ { \ -- } \partial \mathrm { \ / c l i m a t o l o g y } } \mathrm { \quad - } \mathrm { \ / c a s m S \ t o r e c a s t } \mathrm { \quad  - \ A u r o r a } \mathrm { \quad \ -- \ F o e h n . C A M S }
$$

Extended Data Fig. 7 | Trained in Aurora’s own setup, Foehn’s design has a lower error than Aurora 3 and 5 days ahead, with 131 times fewer parameters. Foehn-CAMS has Foehn’s design (computed transport, learned tracers, flow-matching network; 9.9 million parameters) but is trained on the $0 . 4 ^ { \circ }$ CAMS analyses with the inputs of Aurora’s air-pollution model (1.3 billion parameters) and forecasts the instantaneous near-surface $\mathrm { P M } _ { 2 . 5 }$ every 12 h up to five days. a, Forecasts initialized at 00 UTC on 1 September 2022, one and five days ahead: Foehn-CAMS (ensemble mean of eight members), Aurora run from the same analyses, the operational CAMS forecast, and the CAMS analysis, the truth of the benchmark; shown over land, scored over the whole domain $( \mathrm { \bar { 8 } 4 ^ { \circ } N - } 6 0 ^ { \circ } \mathrm { \bar { S } } ;$ numbers: latitude-weighted RMSE of each map, µg m $^ { - 3 } ) .$ b, Latitude-weighted RMSE (top) and continuous ranked probability score (bottom) against the CAMS analysis by lead time, for the forecasts initialized at 00 UTC from May to November 2022 (214 initializations) and valid at 00 UTC, over the whole domain and four continents. The CRPS of Foehn-CAMS is the fair CRPS of its eight members; for the deterministic Aurora and CAMS forecasts it reduces to the absolute error. Grey dashed, climatology: the CAMS analyses of 2018–2021 at the same hour within ±15 days of the calendar day (their mean for the RMSE, the 124 of them as an ensemble for the CRPS). Insets, the RMSE three to five days ahead. Six regions at 12-hour steps, with the intervals of the differences, are in Extended Data Fig. 8.

![](images/1708f0d043761f829eabcdf0aea0cff24e73c67a1d39642ec540d356b3b683c9.jpg)  
Extended Data Fig. 8 | In Aurora’s setup, Foehn’s error 3 days ahead is lower than Aurora’s over 79% of the Earth’s surface and about half of the land. a–f, Near-surface PM on 4 September 2022 at 00 UTC forecast three days earlier, over six regions: the CAMS analysis (truth), Foehn-CAMS (ensemble mean), Aurora and the CAMS forecast, shown over land, with the latitude-weighted RMSE of each map over the region (µg m<sup>−3</sup>); right, the latitude-weighted RMSE of the region by lead time at 12-hour steps over May–November 2022 (214 initializations), with the climatology of the CAMS analyses of 2018–2021 (grey dashed), and beside it the days three to five (shaded) enlarged, with the change of Foehn-CAMS’s RMSE relative to Aurora three days ahead and its 95% interval (7-day moving-block bootstrap over initializations; blue when the interval excludes zero). Three days ahead, Foehn-CAMS’s error is lower than Aurora’s over 79% of the Earth’s surface but over about half of the land area.

Computed transport (FLEXGREEN) Emissions met along the forecast path Transport gains more Emissions gain more  
a  
![](images/cfc307b6102b2533cbd2c2c9d32683c0991d9e40cea3cdef71c71302ead2b039.jpg)  
b

![](images/68b7163484154ce3f50bc756a93cb27aa844192616e66268f287f9ae91b95887.jpg)

![](images/00ea1eaa656fe6e61b0e8307c67883af5f4bdc0cc4f8456fce57f3e1992cdfc4.jpg)

d  
![](images/fc45ced83e389192e39449ff73a6d093ed6a4e10a2ac75c49d42db774e42c299.jpg)

![](images/edd3eccadec7f8faf30f038d6b68a8ae3c4dd54d7842efe3d87c18a042f40440.jpg)  
f

![](images/17c9aaf1a83cfa85aaab4f9a5ee820eb090c23b1d9999bd980fde4d0c4dbdc3c.jpg)

g  
![](images/549e0f097dffe3a8080198b4c498d28d833fbe392b982826ef7432b13bb222ce.jpg)  
h

![](images/6c2184d196602176353fe308c710ca40f606ea4ce9593eeab83cd85d544601f1.jpg)

![](images/5e7bccb9ecc887723f398fab76fc200a6180b9eae327aba4ef20671694714c3a.jpg)  
j

![](images/830c637e1501375b328cd5c5e6bc06cc2ff7a6aa3c8b2e8f504158659e02a32e.jpg)  
a-f: gain in accuracy of the ensemble mean

Extended Data Fig. 9 | The computed transport adds skill at short lead times, and the emissions taken up along the way add it at longer ones. a–f, Gain in the accuracy of the ensemble mean (reduction of the root-mean-square error, RMSE, against the 1 km estimates, on identical 0.25<sup>◦</sup> land cells and days of 2022) brought by the computed transport (blue: the base model relative to the same network trained without the transport inputs) and by the emissions taken up along the back-trajectories (red: the model with the six anthropogenic emission channels relative to the base model trained without them), by lead time, in six of the nine regions of the evaluation (the other three in Methods); shading, 95% intervals (paired moving-block bootstrap over start days); the background marks which contribution gains more, and the dotted line the lead time at which the emissions overtake the transport (East Asia, 3.9 days; South America, 3.8 days). g–i, The RMSE itself, by lead time, for the base model and the model with emissions, in Europe, North America and East Asia. j, Ratio of the ensemble spread to the RMSE of the ensemble mean over all regions together, by lead time, for the two models. Both comparisons are between models before the calibration training (Methods).

![](images/efc6d775fe909e88ce582118afbcd013881f33737579d5fc3689cb0ead095e02.jpg)

![](images/d665e743aed1cc865cf9861e576330747e6f74e2a215ce9ce40e1219e860ea3e.jpg)  
Extended Data Fig. 10 | Over Europe, a network trained at 1 km and applied tile by tile leaves seams that Foehn’s tile-free network avoids. One-day forecast of $\mathrm { P M } _ { 2 . 5 }$ at 1 km: the 1 km estimates, the target (a), Earthformer trained at 1 km and applied tile by tile (b) and the tile-free flow-matching network of Foehn (c); arrows mark tile edges, and the root-mean-square error of each forecast is shown. d, Mean jump of the forecast across tile edges divided by the jump elsewhere, over Europe (1, no seam).