# Times Series Forecasting 

<img src="reports/logoprojeto.jpeg" width="800px" height="40%">

# 1. Project Description

**Eletrical comsuption forecasting**
- In this project, I will **forecast eletricity comsuption in Brazil**. 
- The forecast will be performed using five **statistical forecasting models** and three **machine learning models**.
- The **trend**, **seasonality**, **cyclicality** and the **irregular component** of the time series are extracted and analyzed using different methods.
- Forecasting models are **trained** with data from the series and **validated** with the most recent data from the historical series.
- Different **error metrics** are used to measure the quality of forecasting methods
- **Cross-validation** is performed to increase confidence in the metrics used.
- The best-ranked methods are used to forecast consumption values ​​for the next year.

Future project features:

**Daylight Saving Time Analysis**
- Add the historical data on the adoption of the **daylight saving time public policy**.
- Analyze the influence this public policy had on electricity consumption in the country.
- Generate a result to **support the decision** on whether or not to reinstate daylight saving time.
- **Calculate the amount of energy savings or additional expenditure** resulting from the adoption of the policy.

**Temperature Analysis**
- Add the **historical average temperature series** for each federative unit in the country.
- Analyze the **influence of temperature on monthly electricity consumption** in the country.
- Calculate the **impact of temperature** on consumption in Brazil
- Generate a model for the increase in average temperature due to **global warming**.
- Create **forecast** models for energy consumption
- Observe the increase in this consumption with the increase in average temperature.
- Calculate the relative **increase in consumption due to global warming** and the energy expenditure associated with this increase.
- Create a forecast model for possible **critical points** in electricity consumption in Brazil.

To finally generate a general analysis determining the **dynamics of electricity consumption in Brazil**, what the **impact of global warming** is and whether or not the increase in the planet's average temperature favors the resumption of the policy of **adopting daylight saving time**.


# 2. Solution Pipeline
The **solution pipeline** is based on the **crisp-dm** framework:
1. Business understanding.
2. Data understanding.
3. Data preparation.
4. Modelling.
5. Validation.
6. Deployment.

# 3. Pipeline

1. **Problem understanding :**

- The problem to be solved with this project is to predict electricity consumption in Brazil based on historical data.
- In the future, an analysis of the influence of daylight saving time and temperature on the consumption time series will be added to ultimately support decisions about whether or not to adopt a public daylight saving time policy.

<img src="reports/apresentacao.png">


2. **Data Understanding:**
- The Energy Research Company of the Brazilian government's Ministry of Mines and Energy provides historical data, from 2004 onward, on monthly electricity consumption and the number of consumers at the national, regional, and subsystem levels, segmented by consumer type (captive or free) and by class (residential, industrial, commercial, and others).
- Historical electricity consumption series are available in .xlsx format, as exemplified in Figure X below.

 <img src="reports/base_EPE.png">


- The .xlsx file consists of separate tabs for consumption and the number of captive and free consumers, by region and class, or by federal unit.
- The data provided is available monthly.
- Consumption data does not contain NaN values.
- NaN values ​​for the number of consumers are fill as 0.
- Consumption data is available in units of MWh.
- Data separated by federative unit allows us to analyze the impact of adopting daylight saving time, since this public policy was implemented by federative unit.

<img src="reports/base_HV.png">

- Historical data on the adoption dates of daylight saving time and the states in which it was adopted are available online.
- The National Institute of Meteorology provides monthly historical temperature data measured at each of the meteorological stations throughout the country, allowing for the calculation of an average temperature per federal unit and the historical series of these temperature data.
- From the data made available by INMET, data from operating conventional meteorological stations were used, and in the period covered by the time series of electricity consumption, from January 2004 to the present.
- The data are made available in a zipped file in which there is a .csv file for the historical series of each station, with the identification of each station, the geographic location, the start and end date of the historical series and the values ​​of the average monthly temperature measured.
-The file in figure XX shows an example of the .csv files made available by each weather station.
- After obtaining the temperature for each station, a weighted average is calculated by region covered by each station to finally generate an average temperature value for each federative unit of the country.
- These files contain NaN values ​​for temperatures; periods without available data will be evaluated to determine the best method for dealing with missing values.

<img src="reports/Base_TEMP.png">

**Some particular features of the base - Electricity Consumption**
- Consumption related to captive consumers is predominantly residential (~70%).
- Consumption related to free consumers is predominantly industrial (~75%).
- The captive consumption time series does not have a clear trend component but does have a seasonal component.
- The free consumption series has a clear positive trend component and a weak seasonal component.
- The total consumption series (captive + free) therefore has a positive trend component and a seasonal component.
- Captive consumption still accounts for the majority of electricity consumption in the country, but free consumption is likely to dominate in the coming years.

**Some particular features of the base - Temperature**

**Some particular features of the base - Daylight saving time**

<img src="reports/TOTAL_CATIVO_LIVRE.png">

3. **Data Preparation:**

**Data Import and Preprocessing**
Relevant information is extracted from the databases and dataframes are filled with the information that will be used in the analyses.

   - Import historical consumption data.
   - Import daylight saving time serie. 
   - Import temperature data
   - Handling missing values.
   - Data preprocessing.
   - Fitting data to pandas Dataframe formats - wide, compact and expanded.


4. **Modeling:**

1. **Detrending**
Visual analysis of the charts reveals a clear trend component. We'll use different methods to extract this component and measure these methods to determine the best detrending result.

    - The total consumption time series is used in this analysis.
    - Detrending is performed using the moving average and LOWESS (locally weight estimated scatterplot smoothing) methods.
    - The extracted trend components are shown in Figures XX and XX1.
    - The calculated metrics are shown in YY.

<img src="reports/METRICAS_DETRENDING.png">

    - The metrics indicate that the moving average decomposition method with a multiplicative model is the best model; the LOWESS model also has good parameters.
    - Analysis of the time series detrended by the different methods shows that the multiplicative model flattens the curve too much, which is undesirable.
    - I perform a logarithmic transform on the raw time series and a decomposition with an additive model, which is similar to applying the multiplicative model but can reveal whether the behavior flattens the curve too much.
    - The logarithmic model combined with the additive decomposition confirms the undesirable behavior of the multiplicative model.
    - Decomposition using the LOESS method is chosen as the most appropriate. 

<img src="reports/COMPONENTES_TENDENCIA.png">

<img src="reports/SERIES_DETRENDIDAS_SOLTAS.png">
<img src="reports/SERIES_DETRENDIDAS_juntas.png">

2. **Deseasonality**
Visual analysis of the consumption time series after detrending indicates the presence of trend and cyclical components.

- The consumption time series without the trend component is used in this analysis (detrending-loess).
- Graphs of the autocorrelation and partial autocorrelation of the detrended series data are plotted.

<img src="reports/autocorrelação.png">

- The total and partial autocorrelation graphs indicate the presence of seasonality in 12-month (1-year) periods.
- The partial autocorrelation graph shows an average correlation with 60-month (5-year) periods, which could indicate the presence of a cyclical component.

<img src="reports/autocorrelaçãoparcial.png">

- Fourier series decomposition is performed on the detrended time series.
- I define the error metrics to be used to quantify the quality of deseasonality.
- I plot the error metrics by the number of harmonics used in the Fourier series.

<img src="reports/erroporharmonico.png">


- This plot indicates an optimal value of 53 harmonics in the Fourier series, explaining a total of 90% of the variance observed in the detrended series. - The seasonality and cyclicality components are reconstructed with the first 53 harmonics.
- The frequency magnitude spectrum shows the low frequency of 60-period components to justify a possible 60-month cyclicality.

<img src="reports/espectro_magnitude.png">

- The seasonality component is removed from the detrended series, leaving what we will call the irregular component of the time series.

<img src="reports/irregularcomponente.png"> 
    
3. **Forecast**



4. **Evalution:**
- First of all, **data cleaning** was performed to turn the raw data suitable for data exploration and modeling. Tasks performed in this step:
- Obtain a sorted dataframe, providing a chronological order for the loan data.
- Remove features with higher than 70% missing rate, excessive cardinality, unique values per observation, no variance/constant values, and irrelevant variables to the business or modeling point of view.
- Treat missing values, removing observations with missings when they represent a very small portion of the data and imputing them when they represent a specific value, like zero.
- ⁠Convert features to correct data type (object to datetime and int).
- ⁠Create new independent features.
- ⁠Create the target variables for the PD (stablishing a default definition and assigning 1 to good borrowers and 0 to bad borrowers in order to interpret positive coefficients as positive outcomes), EAD (credit conversion factor) and LGD (recovery rate) models.
- ⁠Search and fix inconsistent outlier values.
- ⁠Optimize memory, obtaining a final parquet file.
- As a result, we went from 75 features to a dataset with 42 variables in its correct data types, optimized in terms of memory usage, with some missing values and outliers treat and new useful extracted features. 

**Deployment**
- The goal of the exploratory data analysis was to **investigate Lending Club's current investment portfolio's personal, financial, and credit risk indicators**, as previously mentioned. Additionally, in this step, I **determined** the final set of **dummy variables** to construct for the **PD Model**, essentially outlining the preprocessing steps to be undertaken.
- Due to interpretability requirements, the PD Model must include only dummy variables. To create these dummies, I analyzed the discriminatory power of each categorical and numerical variable by assessing the **Weight of Evidence (WoE)** for each category. Subsequently, using both the WoE values and the proportion of observations, I grouped categories together to construct additional dummies. The goal was to **combine** similar credit risk/WoE **categories** and categories with low proportions of observations (to prevent overfitting). An important observation is that the highest credit risk or lowest WoE categories, the reference categories, were separated for further dropping, in order to avoid multicolinearity issues (dummy variable trap). 
- For **continuous features**, I applied **feature discretization** to facilitate this categorical analysis. Discretizing continuous features allows for a more comprehensive understanding of their relationship with the target variable. This process helps minimize the impact of outliers and asymmetries, enables the assessment of potential linear monotonic behaviors, and provides the opportunity to apply treatments when such behaviors are not observed. It's important to note, however, that discretization comes at the cost of increased dimensionality and a loss of information.

**7.3 PD Modeling:**
- In PD modeling, I initially excluded variables that would not be available at the time of prediction to prevent data leakage, such as the funded amount or total payments. Additionally, I eliminated variables that demonstrated no discriminatory power during the Exploratory Data Analysis (EDA).
- Subsequently, I conducted an **out-of-time train-test split**, which is considered the best approach for PD, EAD, and LGD Modeling. This is crucial as we construct models using past data to predict future applicants' data.

<img src="reports/out_of_time_split.png">

- Following this, I applied the necessary **preprocessing**, creating the **dummy variables** determined in the EDA step. I discretized the identified continuous features and then grouped all the specified categories to obtain the final dummies, eliminating the respective reference categories. An important observation is that I considered missing values in a variable as another category of it, because they showed a higher proportion of defaults, not being missing values at random.
- Once the data was preprocessed, I estimated the **PD Model using hypothesis testing to evaluate p-values** for the predictor variables. This helped determine whether these variables were statistically significant (i.e., had a coefficient different from zero) or not.
- Independent variables with all dummies containing p-values higher than an alpha of 0.05 were removed, simplifying the model.
- **Interpretation of the coefficients** was performed. For instance, considering the coefficient for sub_grade_A3_A2_A1 as 0.694287, we can infer that the odds of being classified as good for a borrower with A1/A2/A3 subgrades are exp(0.694287) = 2.0 times greater than the odds for someone with G1/G2/G3/G4/G5/F2/F3/F4/F5 subgrades (the reference category).
- Subsequently, I **evaluated the PD Model** by dividing the **scores** into **deciles** and assessing whether there was **ordering** in them. Indeed, in both the training and test data, there was a clear ordering: the lower the credit risk or the higher the score, the lower the bad rate. Moreover, more than 50% of the bad borrowers were observed up to the third decile/score.


<img src="reports/ordering_per_decile.png">

<img src="reports/cum_bad_rate_decile.png">

- Furthermore, with a **KS** of approximately **0.3**, an **ROC-AUC** of around **0.7**, and a **Gini** coefficient of about **0.4** on the test set, the application model exhibits **satisfactory performance**. The model demonstrates effective discriminatory power, distinguishing well between good and bad borrowers. Examining the **Brier** Score, it is very **close to zero**, indicating that the model presents **well-calibrated probabilities** or scores. Furthermore, the **train and test scores** for each of these metrics are quite **similar**. Consequently, the model is not overfitted, has captured the underlying patterns within the data, and is likely to distinguish well between good and bad borrowers in new, unseen data.

<img src="reports/roc_auc.png">

| Metric | Train Value | Test Value |
|--------|-------------|------------|
| KS     | 0.268181    | 0.297876   |
| AUC    | 0.683655    | 0.703449   |
| Gini   | 0.367310    | 0.406897   |
| Brier  | 0.100512    | 0.061633   |

- Finally, a **scorecard** was developed, transforming the coefficients from the PD Model into easily interpretable integer values known as scores. Various formulas were employed to compute these scores, with a minimum score of 300 and a maximum of 850. Subsequently, **credit scores** were **calculated for all borrowers** in both the training and test datasets by multiplying each dummy by its scores and summing the intercept.

**7.4 EAD and LGD Modeling:**
- Initially, I **isolated data containing defaulted loans with a "charged off" status**, ensuring sufficient time had passed for potential recoveries.
- Similar to the PD Model, I excluded irrelevant variables and those that could introduce data leakage.
- Subsequently, I performed an **out-of-time train-test split**, following the same approach as with the PD Model.
- Following this, I **investigated both dependent variables:**
    - The dependent variable for the **LGD Model** is the **recovery rate**, defined as recoveries divided by the funded amount. Although LGD represents the proportion of the total exposure that cannot be recovered by the lender when the borrower defaults, it is common to model the proportion that CAN be recovered. Thus, **LGD** will be equal to **1 minus the Recovery Rate.**
    - The dependent variable for the **EAD model** is the **credit conversion factor**, representing the proportion of the funded amount outstanding to pay. Therefore, **EAD** equals the **funded amount multiplied by this credit conversion factor.**
    - Almost **50% of the recovery rates were zero.** Consequently, I opted to **model LGD using a two-stage approach**. First, a logistic regression predicts whether the recovery rate is greater than zero (1) or zero (0). Then, for those predicted as greater than zero, a linear regression estimates its corresponding value.
    - The credit conversion factor exhibited a reasonable distribution, leading me to decide on estimating a simple linear regression.
    - An important observation is that, although LGD and EAD are beta-distributed dependent variables, representing rates, and beta regression is more suitable for estimating them, I tested it against Linear Regression, and almost the same result was achieved. Thus, considering the need to treat 0 and 1 values for beta regression (e.g., replacing them with 0.0001 and 0.9999), for simplicity, I proceeded with linear regression.
- **Data preprocessing** involved one-hot encoding for nominal categorical variables, as linear models benefit from this encoding. For ordinal categorical variables, ordinal encoding was applied to reduce dimensionality and preserve ordering information. Standard scaling was applied to both ordinal encoded and numerical variables since linear models are sensitive to feature scaling. Missing values were imputed with the median due to an extremely right-skewed variable distribution.
- I estimated the two-stage LGD and EAD Models. For LGD, I combined the two predictions by taking their product. Predictions from the first stage logistic regression that predicted a recovery rate of zero remained zero, while those predicted as one received the estimated value from the second stage linear regression.
- The **results were satisfactory**, although not impressive. Both models' **residuals distributions resembled a normal curve**, with most values around zero. Additionally, some tails were observed, indicating that the LGD Model tends to underestimate the recovery rate, and the EAD tends to overestimate it. However, with a **Mean Absolute Error (MAE) of 0.0523 and 0.1353** for the LGD and EAD Models, respectively, the models provide useful predictions. On average, the predicted recovery rates deviate by approximately 5.23 percentage points from the actual values. On average, the predicted credit conversion rates deviate by approximately 13.53 percentage points from the actual values.

Residuals distribution and actual vs predicted values for the LGD Model.

<img src="reports/residuals_dist_lgd.png">

| Actual | Predicted | Residual |
|--------|-----------|----------|
| 0.06   | 0.10      | 0.05     |
| 0.15   | 0.10      | 0.05     |
| 0.14   | 0.15      | 0.01     |
| 0.16   | 0.12      | 0.05     |
| 0.15   | 0.09      | 0.06     |

Residuals distribution and actual vs predicted values for the EAD Model.

<img src="reports/residuals_dist_ead.png">

| Actual | Predicted | Residual |
|--------|-----------|----------|
| 0.93   | 0.82      | 0.11     |
| 0.90   | 0.84      | 0.06     |
| 0.73   | 0.64      | 0.09     |
| 0.96   | 0.64      | 0.31     |
| 0.64   | 0.70      | 0.06     |

**7.5 Expected Loss (EL) and Credit Policy:**
- To compute **Expected Loss (EL)**, which is the **product of Probability of Default (PD), Exposure at Default (EAD), and Loss Given Default (LGD)**, I leveraged the results of the three models (PD, EAD, and LGD Models) on the test data used for testing the PD Model, encompassing both default and non-default loans.
- I **created 10 risk classes (AA, A, AB, BB, B, BC, C, CD, DD, F)** based on the probability of default because this way we can better leverage the results of the credit scoring model, and it is possible to establish different policies for individuals in different risk classes.
- In this context, **Lending Club** is adopting a more **conservative profile** with a focus on the **profitability** of its assets. The goal is to mitigate risks associated with higher-risk and potential default loans while maximizing profitability.
- To achieve this, the **CEO has outlined a conservative credit policy:** We will automatically approve loans for applicants who fall into AA and A risk classes (indicating the lowest credit risk and highest credit scores) and automatically deny those who fall into the F class (indicating the highest credit risk and lowest credit scores). For the other classes, the loan must provide an annualized Return on Investment (ROI) greater than the basic United States interest rate. This criterion aligns with the rationale that if a loan's expected ROI doesn't surpass this interest rate, it may be more prudent to invest in less risky options, such as fixed-income investments.
- Considering the data goes up until 2015, I assumed that the United States has a basic interest rate of 2.15%.
- As a **financial result**, with our simple credit policy rules, by rejecting just 11% of the loans (including those belonging to the worst risk class, F, and those with an annualized ROI lower than 2.15, the basic US interest rate), both the amount Lending Club expects to lose in its assets and the default rate decreased. Specifically, the **default rate decreased from 6.71% to 5.65%** and the **Expected Loss decreased from 6.91% to 5.77%.** Although these represent **little percentage points** decreasement, when dealing with **thousands of loans** and funded amounts, it represents a **substantial financial gain to Lending Club.** Furthermore, other policies can also be designed, in a more restrictive or free way. This is just a draw to show that our project is worthwile.

**7.6 Model Monitoring:**
- Imagine a year has passed since we built our PD model. Although it is very unlikely, the people applying for loans now might be very different from those we used to train our PD model. We need to reassess if our PD model is working well.
- If the population of the new applicants is too different from the population we used to build the model, the results may be disastrous. In such cases, we need to redevelop the model.
- I applied **model monitoring to our PD Model one year after its construction, using 2015 loan data**. Model monitoring aims to observe whether applicants' characteristics remain consistent over time. The fundamental assumption in credit risk models is that future data will resemble past data. If the population changes significantly, it may be necessary to retrain the model. To assess differences between the actual (training data) and expected (monitoring data), the **Population Stability Index (PSI)** was calculated for each variable.
- Initial list status exhibited the highest PSI, nearly equal to 0.25, indicating a substantial change in the applicants' population. However, this change is more likely due to shifts in the bank's strategies than changes in the borrowers' characteristics.
- On the other hand, **credit scores** showed a PSI of 0.19, close to 0.25. This suggests that we may need to construct another PD Model in the near future. This represents a **significant population change, implying that our model outputs are considerably different from those observed previously.**

**7.7 Next steps:**
- Considering the 2015 applicants' profile has changed, especially the scores distribution, the next steps involve constructing a new PD Model utilizing more robust methods, such as boosting algorithms, focusing on predictive power and trying to use machine learning interpretability tools such as SHAP and LIME.
- As a final product, I intend to deploy these models on production environment.

# 8. Obtain the Data
- The data used are provided by the energy research company, linked to the Brazilian government's Ministry of Mines and Energy. The data are released monthly and separated by consumer type, state, and class. The data are available electronically and can be obtained at (https://www.epe.gov.br/pt/publicacoes-dados-abertos/publicacoes/consumo-de-energia-eletrica).
- Average temperature data per meteorological station are provided by the National Institute of Meteorology (INMET) and are available at the link (https://portal.inmet.gov.br/).
- Historical data on the adoption dates of daylight saving time and the states in which it was adopted are available online. The data used here are available at (https://pt.wikipedia.org/wiki/Lista_de_per%C3%ADodos_em_que_vigorou_o_hor%C3%A1rio_de_ver%C3%A3o_no_Brasil).

# 4. Technologies and Toools Used
- Python (Pandas, Numpy, Matplotlib, Seaborn, Sciki-Learn, Statsmodels, Virtual Envs).
- Statistics.
- Data cleaning, manipulation, visualization and exploration.
- Detrending methods: LOWESS and moving averages
- Deseasonality methods: Decomposition on Fourier series.
- Statistical forecast algorithms: Naive, SeasonalNaive, ETS, Theta and ARIMA.
- Machine learning algorithms: NBEATS, NHITS, PatchTST.
- Error metrics: MAE, MSE, MASE, MAPE, sMAPE.
- Cross-validation

# 10. Contact me
- Linkedin: https://www.linkedin.com/in/pedro-almeida-ds/
- Github: https://github.com/allmeidaapedro
- Gmail: pedrooalmeida.net@gmail.com




# 2. Problems and Objectives
**2.1 What is the importance of forecasting electricity consumption?:**
- Forecasting a **country's electricity consumption** is paramount for **national security** and **socioeconomic development**, as it forms the essential foundation of **energy planning** by ensuring generation is precisely balanced with demand, thus **preventing blackouts and supply shortages**. This accurate prediction is vital for long-term strategic decisions, guiding **multi-billion-dollar investments** in infrastructure expansion, **optimizing operating costs** by prioritizing the cheapest and most sustainable energy sources, and successfully navigating the complex transition toward renewable energy, ultimately guaranteeing the **stability**, **reliability**, and **economic competitiveness** of the nation's power grid..

**2.2 Why the study and statistical forecast to support the decision?**
- **Supporting public policy decisions** that impact electricity consumption, such as the adoption of **Daylight Saving Time**, is vital because it anchors governmental action in **rigorous statistical data** and **evidence-based analysis**. This analytical approach, which relies on **time-series studies** and **econometric modeling**, moves beyond historical assumptions to confirm the **policy's real-world efficacy**. Specifically, statistical analysis of consumption data is essential to **validate the policy's impact**, **optimize system operation**,**ensure transparency**, providing a **scientifically robust justification** for a policy's continuation or discontinuation, thereby fostering public trust by balancing potential energy benefits against verified social and economic trade-offs.

**2.3 What are the objectives and potential benefits?**
- The goal is to forecast consumption and calculate the influence of certain characteristics on the historical consumption data to support public policy decisions. Potential benefits include the potential for energy savings for the country, as well as anticipating and preparing the system for variations in consumption, distribution, and losses.

**2.4 




