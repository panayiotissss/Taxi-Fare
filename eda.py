import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("data/taxi_trip_pricing.csv")
#print(df.head())
#print(df.columns)


#Basic EDA
# We are going to check about nulls/missing/extreme outliers/duplicates - DONE
# /correlations/dtypes # Get a feel of the dataset and decide which type of features we are going to engineer and what encoding and scaling process we are going to take - DONE

# Plot features against the target to see if they influence - DONE
# Target variable distribution — check if trip_price (or whatever the target column is called) is heavily right-skewed. Taxi fares often are — most trips are cheap, a few are very expensive. If it's skewed, you may
  #want to consider a log-transform of the target, which can improve model performance significantly. - DONE





#Nulls/missing/extreme outliers/duplicates

#Duplicates = 0
print(df.duplicated().sum())


#Missings 50 per feature row 49 on target (Decision is to impute them categorical with mode numerical with mean)
print(df.info())

#Outliers (No extreme outliers (something that cannot happen))
print(df.describe())



 #   Column                 Non-Null Count  Dtype  
#---  ------                 --------------  -----  
 #0   Trip_Distance_km       950 non-null    float64 -> robust scaler for all
 #1   Time_of_Day            950 non-null    object -onehot
 #2   Day_of_Week            950 non-null    object -onehot
 #3   Passenger_Count        950 non-null    float64
 #4   Traffic_Conditions     950 non-null    object  -> ordinalencoder
 #5   Weather                950 non-null    object  -> onehot
 #6   Base_Fare              950 non-null    float64
 #7   Per_Km_Rate            950 non-null    float64
 #8   Per_Minute_Rate        950 non-null    float64
 #9   Trip_Duration_Minutes  950 non-null    float64
 #10  Trip_Price             951 non-null    float64

for col in df.select_dtypes(include='object').columns:
      print(f"{col}: {df[col].unique()}")


# for col in df.select_dtypes(include='number').columns:
#       print(f"{col}: {df[col].unique()}")

# #Time_of_Day: ['Morning' 'Afternoon' 'Evening' 'Night' nan]
# #Day_of_Week: ['Weekday' 'Weekend' nan]
# #Traffic_Conditions: ['Low' 'High' 'Medium' nan]
# #Weather: ['Clear' nan 'Rain' 'Snow']


# # Correlations (identify highly correlated to drop) -> max 0.28 for all others, we dont drop when correlated with the target

# df_numerical = df.select_dtypes(include='number')

# plt.figure(figsize=(10, 8))
# sns.heatmap(df_numerical.corr(), annot=True, fmt=".2f", cmap="coolwarm")
# plt.show()




# # Target distribution
# plt.figure(figsize=(8, 4))
# sns.histplot(df['Trip_Price'], kde=True)
# plt.title('Target Distribution - Trip_Price')
# plt.show()

#   # Numerical features vs target
# numerical_features = ['Trip_Distance_km', 'Passenger_Count', 'Base_Fare',
#                          'Per_Km_Rate', 'Per_Minute_Rate', 'Trip_Duration_Minutes']

# fig, axes = plt.subplots(2, 3, figsize=(15, 8))
# for ax, col in zip(axes.flatten(), numerical_features):
#       ax.scatter(df[col], df['Trip_Price'], alpha=0.3)
#       ax.set_xlabel(col)
#       ax.set_ylabel('Trip_Price')
# plt.tight_layout()
# plt.show()

#   # Categorical features vs target
# categorical_features = ['Time_of_Day', 'Day_of_Week', 'Traffic_Conditions', 'Weather']

# fig, axes = plt.subplots(1, 4, figsize=(16, 5))
# for ax, col in zip(axes, categorical_features):
#       sns.boxplot(data=df, x=col, y='Trip_Price', ax=ax)
#       ax.set_xlabel(col)
# plt.tight_layout()
# plt.show()