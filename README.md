# Car Price Prediction

The aim of this project is to use regression in python to predict the price of a car. The problem was originally posted by **TheMathCompany** on the machine learning competitions platform **MachineHack** [here](https://machinehack.com/hackathons/data_hack_mathcothon_car_price_prediction_challenge/overview). I was unable to actually take part in the competition during the submission window, but the problem seemed interesting and so I made this solution after the competition was closed.

****

# Data
The competition had 2 datasets: **train** and **test**. Since we do not have access to the test target values in this project, only the training dataset was used here.
The training dataset has the following attributes:
- **ID**: Unique identifier for each row
- **Price**: Price of the car (**Target Column**)
- **Levy**: The tax imposed on the car
- **Manufacturer**: Name of the manufacturer. For eg: BMW, Honda, Toyota etc...
- **Model**: The specific model of the car. For eg: *Camry* is a model of cars built by *Toyota*
- **Prod. year**: The year when the car was manufactured.
- **Category**: This column indicates whether the car is an SUV, Sedan etc.
- **Leather interior**: Is there leather interior in the car
- **Fuel type**: The fuel that the car needs. For eg: Disel, Petrol etc.
- **Engine volume**: The volume of the engine and whether it has *Turbo* is captured here
- **Mileage**: The number of miles already covered by the car.
- **Cylinders**: The number of cylinders in the engine
- **Gear box type**: Automatic, Manual etc
- **Drive wheels**: Is it a front wheel drive, rear wheel drive or a 4 wheel drive
- **Doors**: Number of doors in the car
- **Wheel**: Right hand drive, or left hand drive?
- **Color**: Color of the car
- **Airbags**: Number of Airbags

****

# Exploratory Data Analysis
EDA for this project was done in the following 2 notebooks:
## `EDA/CarPrice_00_Preliminary_EDA.ipynb`
In this notebook, I performed the following:
- Checked for null values
- Checked Data types for each column
- Split the data into 3 sets:
  - First I split the competition data into `train-val` and `test` sets (90-10 split)
  - Next I split the `train-val` data into `train` and `val` sets (80-20 split of the `train-val` set)
- Univariate EDA
  - I analyse the distribution of each column individually.
  - **Bar graphs**, **Histograms**, **Pareto graphs**, and **Box plots** were the most used tools in the univariate analysis of each column.

## `EDA/CarPrice_01_Multivariate_EDA.ipynb`
This notebook is primarily used to analyse how each column affects the target column (**Price**)
- **Scatter plots**, and **Box plots** were used to guage how the **Price** of a car changes with numeric and categorical features respectively.
- Almost all of the categorical features and a few numeric ones were **label encoded** and then their correlation with **Price** was plotted via **Heatmaps**
- The order for label encoding was decided by the mean and the median **Price** of each category
- The effects of **removing outliers** in **Price** were studied for each feature.
- The steps needed to perform feature engineering were finalized

****

# Feature Engineering and Selection
This was carried out in the following notebooks

## `Feature_Engineering/CarPrice_02_Feature_Engineering.ipynb`
The features in the `train` and `val` datasets are processed in the following manner:
### Feature description
- `prod_year`: Copy of `Prod. year` column
- `prod_year_delta`: Difference in years from the year `2020` (max `Prod. year` in train set)
- `prod_year_delta_sq`: Square of `prod_year_delta`
- `prod_year_delta_cu`: Cube of `prod_year_delta`
- `prod_year_new`: Defines how new the car is based on the `Prod. year` feature.
  - `prod_year_new = 1` means `prod_year < 1 year`
  - `prod_year_new = 2` means `1 <= prod_year < 4`
  - `prod_year_new = 3` means `4 <= prod_year < 6`
  - `prod_year_new = 4` means `6 <= prod_year < 8`
  - `prod_year_new = 5` means `8 <= prod_year < 10` 
  - `prod_year_new = 6` means `prod_year >= 10 years`
- `cylinders`: `Cylinders` type-casted as `int`
- `airbags`: `Airbags` type-casted as `int`
- `levy`:
  - Copy of `Levy` column
  - Replace `"-"` in `Levy` with `np.nan`
  - Convert column to data type: `np.float`
  - Fill null values with median
- `manufacturer_class`:
  - `= 0`: Unknown Manufacturer 
  - `= 1`: Manufacturer from `class_1` list
  - `= 2`: Manufacturer from `class_2` list
  - `= 3`: Manufacturer from `class_3` list
  - `= 4`: Manufacturer from `class_4` list
- `manufacturer_mean_le`: Label encoding `Manufacturer` according to mean `Price`
- `manufacturer_median_le`: Label encoding `Manufacturer` according to median `Price`
- `category_mean_le`: Label encoding `Category` according to mean `Price`
- `category_median_le`: Label encoding `Category` according to median `Price`
- `leather_interior`: 
  - `= 1` when `Leather interior = "Yes"`
  - `= 0` when `Leather interior = "No"`
- `fuel_type_mean_le`: Label encoding `Fuel type` according to mean `Price`
- `fuel_type_median_le`: Label encoding `Fuel type` according to median `Price`
- `turbo`
  - `= 1`: If `Engine volume` contains the word `Turbo`
  - `= 0`: If `Engine volume` does not contain the word `Turbo`
- `engine_volume`: 
  - Remove `Turbo` text from some of the values
  - Typecast to `float` 
- `mileage`
  - Remove `km` string from each value
  - Typecast data to `np.float`
- `mileage_new`
  - `1` For all cars which have `mileage = 0`
  - `0` otherwise
- `gear_box_type_mean_le`: Label encoding `Gear box type` according to mean `Price`
- `gear_box_type_median_le`: Label encoding `Gear box type` according to median `Price`
- `drive_wheels_mean_le`: Label encoding `Drive wheels` according to mean `Price`
- `drive_wheels_median_le`: Label encoding `Drive wheels` according to median `Price`
- `doors_mean_le`: Label encoding `Doors` according to mean `Price`
- `doors_median_le`: Label encoding `Doors` according to median `Price`
- `wheel_mean_le`: Label encoding `Wheel` according to mean `Price`
- `wheel_median_le`: Label encoding `Wheel` according to median `Price`
- `color_mean_le`: Label encoding `Color` according to mean `Price`
- `color_median_le`: Label encoding `Color` according to median `Price`

### Outlier removal
- Remove all data points from training where `Price > 100,000`
- Removing points that lie after the `99th percentile` of the following features:
  - `levy`
  - `engine_volume`
  - `mileage`

## `Feature_Engineering/CarPrice_03_Feature_Selection.ipynb`
The following methods were used in this notebook to identify redundant features:
- **f_regression**: Measures the linear relationship features have with the target.
- **mutual_info_regression**: Measures the amount of information that can be expressed about the target using a feature.
- **RFE (Recursive Feature Elimination)**: RFE select features by recursively considering smaller and smaller sets of features. Here the **Linear regression** estimator was used for RFE.
- **p-value from OLS regression**: The OLS regression implemented in `statsmodels` library returns the p-values of all features. If `p<0.05` then the feature adds some useful information to the model.
- **Feature Importance in RandomForest**: Random Forest class in `sklearn` has its own implementation of calculating feature importances.

All these methods measure how important a feature is in different ways. After considereing all their results some features seemed redundant across all tests. The following were then removed.
- All Median label encoded (...`median_le`) features.
  - This is because in all cases the Mean label encoded (...`mean_le`) counterparts are looking better or roughly the same
- `color`, `drive_wheels`, `doors`
  - Seem unimportant according to all the tests
- `prod_year_new`
  - Intended to capture the same information as mileage_new, but the latter is performing better across all tests.

So the final list of features is:
- prod_year
- prod_year_delta
- prod_year_delta_sq
- prod_year_delta_cu
- cylinders
- airbags
- levy
- manufacturer_class
- manufacturer_mean_le
- category_mean_le
- leather_interior
- fuel_type_mean_le
- turbo
- engine_volume
- mileage
- mileage_new
- gear_box_type_mean_le
- wheel_mean_le

****

# Modeling
The process of building models for solving this problem were carried out in the following notebooks:
## `Modeling/CarPrice_04_Model.ipynb`
This notebook is used to build the following regression models:
| Model                  | Train Performance (RMSE) | Validation Performance (RMSE) |
| ---------------------- | ------------------------ | ----------------------------- |
| Linear Regression      | 12766.324696874255       | 1127556.058065489             |
| Lasso                  | 12766.53693013379        | 1128736.3865379978            |
| Ridge                  | 12766.397624069145       | 1128982.8820277965            |
| Elastic Net            | 13314.169691752979       | 902421.277033476              |
| **KNN**                | **1697.3999611902038**   | **17978.053890238072**        |
| SVM                    | 12036.186391290426       | 20697.75015468442             |
| **Random Forests**     | **3746.2907656827147**   | **17494.283153509172**        |
| **XGBoost**            | **4490.585535826944**    | **17135.583880509148**        |

The models written in bold case (`KNN`, `Random Forests`, `XGBoost`) are then used to create an ensemble.

## `Modeling/CarPrice_05_Ensembling.ipynb`
This notebook uses the above 3 models and creates 2 stacked ensembles (with `Linear regression` used as the **meta learner**). The performance of the 2 ensembles are as follows:
- **Ensemble 1**
  - **Base Learners**: KNN, Random Forests and XGBoost
  - **Meta Learners**: Linear Regression
  - **Train RMSE**: 3103.516807806332
  - **Validation RMSE**: 17330.88987330079
- **Ensemble 2**
  - **Base Learners**: Random Forests and XGBoost
  - **Meta Learners**: Linear Regression
  - **Train RMSE**: 3732.12801745847
  - **Validation RMSE**: 17255.334552412325

**Final model used is the 2nd Ensemble (Random forests and XGBoost)**

## `Modeling/CarPrice_06_Prediction_on_Test_set.ipynb`
Finally this notebook is used to create predictions on our `test` set using the ensemble model above. The `test` set is processed in the same way as the `train` and `val` sets during feature engineering stage. The model's performance is:
- **RMSE: 10543.925662683565**
- **RMSLE: 0.9950522186493146**
  - In the competition, RMSLE was used as the metric.
  - The top solutions in the competition's [leaderboard](https://machinehack.com/hackathons/data_hack_mathcothon_car_price_prediction_challenge/leaderboard) fetched an **RMSLE** of about **0.76**

****
