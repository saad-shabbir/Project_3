import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import os
import researchpy as rp
import seaborn as sns
from sklearn.impute import KNNImputer
import csv
import datetime


# Create a csv writer object to capture important meta-data about the process
my_file = 'meta-data.txt'
handle = open(my_file, 'w', newline='')
writer = csv.writer(handle, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)


# Turn off interactive charting. Therefore, we will have to save all visuals to a file to view them.
matplotlib.use('Agg')
# Let's set a few global figure properties, which should limit the
# amount of properties we have to set later when we are plotting our data.
import matplotlib as mpl
mpl.rcParams.update({'axes.titlesize': 20, 'axes.labelsize': 18, 'legend.fontsize': 16})
# Set up the console to display more rows and columns for our pandas data frames.
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)
# I will also set the console output settings for my numpy arrays to facilitate viewing of the numpy arrays.
np.set_printoptions(suppress=True, threshold=5000, edgeitems=10)


### READING DATA & LOOKING AT DATA ###

full_path = "Mattson_nutrition_customers.json" # loading the data
my_df = pd.read_json(full_path)

writer.writerow([f'We started the training process at {datetime.datetime.now()}.'])
writer.writerow([f'We used the Mattson_nutrition_customers.json file that had the following columns: {list(my_df.columns)} columns'])
writer.writerow([f'It contained {len(my_df)} observations.'])
file_stamp = datetime.datetime.fromtimestamp(os.path.getmtime(full_path), datetime.UTC).strftime('%d %b %Y')
writer.writerow([f'It was last modified on {file_stamp} at the start of our analyses.'])
writer.writerow([f'It was {os.path.getsize(full_path)/1024:,.0f}.'])

# just looking at a general overview of our dataset
#
# print("Shape of the dataset:", my_df.shape)
# print("Column names and types:")
# print(my_df.dtypes)
# print("\nSample of the data:")
# print(my_df.head())
# print("\nNull values per column:")
# print(my_df.isnull().sum())
# print("\nProportion of null values:")
# print(my_df.isnull().mean() * 100)
# print(my_df.sample(5))
# print(my_df.dtypes)

writer.writerow([
    "The dataset shape was assessed, showing 83,216 rows and 25 columns."
])
writer.writerow([
    "A random sample of 5 rows was printed which showed how our data was nested."
])
writer.writerow([
    "Null values were analyzed per column, revealing missing data across various columns, with up to 60.1% in some cases."
])
writer.writerow([
    "The proportion of null values per column was calculated and printed, highlighting data completeness issues."
])
writer.writerow([
    "Data types for each column were reiterated after processing, ensuring consistency in the analysis."
])


### CHECKING FOR UNIQUE CUSTOMER IDs ###

# Our unit of analysis is customers, so we have to verify that all CustomerID are unique.
# If not, we will have a data leakage problem when we randomly split our data between training and testing.
# print(my_df['CustomerID'].is_unique)
# All observations are unique in this data set, so we do not have any potential data leakage when we
# randomly split our data into training and testing data sets.
writer.writerow([f'Our unit of analysis was customers and we verified that all students '
                 f'in the Mattson_nutrition_customers.json file were unique, so we did not have a data leakage problem.'])


### WRANGLING THE DATA ###

# Normalize the data
delta1_df = pd.json_normalize(my_df['delta1'])
delta2_df = pd.json_normalize(my_df['delta2'])
# Add suffixes to columns to distinguish periods
delta1_df = delta1_df.add_suffix('_delta1')
delta2_df = delta2_df.add_suffix('_delta2')
# Combine with the main DataFrame
my_df = pd.concat([my_df, delta1_df, delta2_df], axis=1)
# Drop original delta1 and delta2 columns
my_df = my_df.drop(columns=['delta1', 'delta2'])
# my_df.info()
# my_df.sample(2)

# There is data in WebsiteVisits, MobileAppLogins, Steps, Sleep, HeartRate, Stress that is still in dictionary objects
# List of columns with nested dictionaries to normalize
nested_columns = ['WebsiteVisits', 'MobileAppLogins', 'Steps', 'Sleep', 'HeartRate', 'Stress']
# Normalize each nested column and append _delta1 or _delta2 suffix
for col in nested_columns:
    if col in my_df.columns:
        # Extract each key in nested dictionary and normalize
        expanded_col = pd.json_normalize(my_df[col])
        # Add suffix to columns for clarity
        expanded_col.columns = [f"{col}_{sub_col}" for sub_col in expanded_col.columns]
        # Concatenate normalized columns back to the main DataFrame
        my_df = pd.concat([my_df, expanded_col], axis=1)
        # Drop original column after normalization
        my_df = my_df.drop(columns=[col])
# Ensure proper renaming for Sleep columns
my_df.rename(columns={
    "Sleep_delta1.Deep": "Sleep.Deep_delta1",
    "Sleep_delta1.Light": "Sleep.Light_delta1",
    "Sleep_delta1.REM": "Sleep.REM_delta1",
    "Sleep_delta2.Deep": "Sleep.Deep_delta2",
    "Sleep_delta2.Light": "Sleep.Light_delta2",
    "Sleep_delta2.REM": "Sleep.REM_delta2"
}, inplace=True)
# my_df.sample(2)
# my_df.info()

writer.writerow([f'The delta1 and delta2 columns were normalized into separate columns for each nested key, '
                 f'and suffixes (_delta1 and _delta2) were added for clarity. The original columns were dropped, '
                 f'and the dataset expanded with detailed features.'])

writer.writerow([f'Additional nested columns such as WebsiteVisits, MobileAppLogins, Steps, Sleep, HeartRate, '
                 f'and Stress were normalized into individual columns. Consistent naming conventions, particularly '
                 f'for Sleep metrics (Deep, Light, REM), were applied to improve interpretability.'])



###  Convert numeric object columns with no nulls to numeric type ###
numeric_object_cols = my_df.select_dtypes(include=['object']).columns
for col in numeric_object_cols:
    if my_df[col].notnull().all():
        try:
            my_df[col] = pd.to_numeric(my_df[col])
        except ValueError:
            continue
# my_df.sample(10)
# my_df.info()
writer.writerow([f'Numeric object columns with no missing values were identified and converted to numeric types'])





## DUMMY CODE ALL CATEGORICAL VALUES THAT HAVE NO NULLS and cannot be mapped with 1s and 0s ###

# # Seeing all the different types of categories for object type features
# for column in my_df.columns:
#     if my_df[column].dtype == 'object':  # Check if column is of object type (string or categorical)
#         print(f"Value counts for column '{column}':")
#         print(my_df[column].value_counts())

# my_df.info()
# LifeStyle.Smoke_delta1
# LifeStyle.Drink_delta1
# LifeStyle.Smoke_delta2
# LifeStyle.Drink_delta2

# print(my_df['LifeStyle.Smoke_delta1'].sample(30))
# print(my_df['LifeStyle.Smoke_delta2'].sample(30))
# print(my_df['LifeStyle.Drink_delta1'].sample(30))
# print(my_df['LifeStyle.Drink_delta2'].sample(30))
# print(my_df['Medical.OfficeVisits_delta1'].sample(30))
# print(my_df['Medical.OfficeVisits_delta2'].sample(30))


#  Assuming "Unknown" represents meaningful information (e.g., missing data imputed with "Unknown"),
#  it should be treated as a separate category during the dummy coding process

columns_to_dummy = [
    "LifeStyle.Drink_delta2",
    "LifeStyle.Smoke_delta2",
    "LifeStyle.Smoke_delta1",
    "LifeStyle.Drink_delta1"
]


for col in columns_to_dummy:
    # # Check unique values for safety
    # print(f"Unique values in {col}: {my_df[col].unique()}")
    # # Medical Office visits don't have to be dummy coded it can just be binary mapped

    # Dummy code the column, explicitly including all categories (including 'Unknown')
    dummies = pd.get_dummies(my_df[col], prefix=col, drop_first=True)
    my_df = pd.concat([my_df, dummies], axis=1)
    my_df.drop(columns=[col], inplace=True)

writer.writerow([f'Dummy coding was applied to categorical columns LifeStyle.Drink_delta1, LifeStyle.Smoke_delta1, '
                 f'LifeStyle.Drink_delta2, and LifeStyle.Smoke_delta2, generating binary columns for each category, '
                 f'including "Unknown" as a distinct category.'])

# Map "Yes" to 1 and "No" to 0 for Medical.OfficeVisits columns
my_df["Medical.OfficeVisits_delta1"] = my_df["Medical.OfficeVisits_delta1"].map({"Yes": 1, "No": 0})
my_df["Medical.OfficeVisits_delta2"] = my_df["Medical.OfficeVisits_delta2"].map({"Yes": 1, "No": 0})

writer.writerow([f'The Medical.OfficeVisits_delta1 and Medical.OfficeVisits_delta2 columns were binary mapped to 1 '
                 f'for "Yes" and 0 for "No," simplifying their representation and facilitating numerical analyses.'])

# my_df.info()


### CALCULATING AGE ###

# ExtractDate - Only a single date (10/31/2022) is present
# Convert 'ExtractDate' and 'dob' to datetime format
my_df['ExtractDate'] = pd.to_datetime(my_df['ExtractDate'], format='%m/%d/%Y')
my_df['dob'] = pd.to_datetime(my_df['dob'], format='%Y%m%d', errors='coerce')

# Calculate age at delta1 and delta2 periods
my_df['delta1_age'] = (my_df['ExtractDate'] - pd.DateOffset(months=3)) - my_df['dob']
my_df['delta1_age'] = my_df['delta1_age'].map(lambda x: x.days / 365.2425 if pd.notnull(x) else None)

my_df['delta2_age'] = (my_df['ExtractDate'] - pd.DateOffset(months=6)) - my_df['dob']
my_df['delta2_age'] = my_df['delta2_age'].map(lambda x: x.days / 365.2425 if pd.notnull(x) else None)

my_df.rename(columns={"delta1_age": "age_delta1", "delta2_age": "age_delta2"}, inplace=True)
# my_df.info()

writer.writerow([f'Ages at delta1 and delta2 periods were calculated by subtracting 3 and 6 months, respectively, '
                 f'from ExtractDate and computing the difference with dob in years. Null values were retained '
                 f'where dob was missing.'])


## CHECKING CORRELATIONS BEFORE HANDLING MISSING DATA ##

# my_columns = my_df.select_dtypes(exclude='object').columns
# corrMatrix = my_df[my_columns].corr()
# print(corrMatrix)
# full_path = "CorrelationMatrix.xlsx"
# corrMatrix.to_excel(full_path)
# Sales_delta1 has a high positive correlation with Big5_Conscientiousness (0.590978) and a negative correlation with Big5_Openness (-0.417164).
# This suggests conscientious customers tend to purchase more, whereas openness may negatively influence purchases.
# Big5_Extroversion correlates positively with Sales_delta1 (0.511236), indicating extroverted individuals might buy more.


### HANDLING MISSING GENDER VALUES ###
# Choosing the right method to handle missing age data is critical,
# as older and younger customers often exhibit distinct trends in their nutritional habits.

# Value counts for column 'Gender':
# Female    51581
# Male      27363
# Other      2127
# We have more females in the dataset relative to males.
# print(my_df['Gender'].isnull().sum())
# 2145 nulls

# my_df.sample(10)
# my_df.info()

# Create a copy of the original Gender column for the random assignment method
my_df['Gender_random'] = my_df['Gender']
# Create a copy of the original Gender column for the mode-based method
my_df['Gender_mode'] = my_df['Gender']

# Method 1: Fill missing Gender values using random assignment based on proportions
# Get the proportion of each Gender category
gender_proportions = my_df['Gender'].value_counts(normalize=True)
# print("Gender proportions:\n", gender_proportions)
# Create a count of the number of missing Gender values
missing_gender_count = my_df['Gender_random'].isnull().sum()
# Generate random Gender values weighted by proportions
random_genders = np.random.choice(
    gender_proportions.index,  # The Gender categories (e.g., Female, Male, Other)
    size=missing_gender_count,  # Number of values to generate
    p=gender_proportions.values  # Probabilities based on proportions
)
# Fill the missing values in the Gender_random column with random assignment
my_df.loc[my_df['Gender_random'].isnull(), 'Gender_random'] = random_genders


# Method 2: Fill missing Gender values with the most frequent (mode) value - female
# Fill the missing values in the Gender_mode column with the mode
my_df['Gender_mode'].fillna(my_df['Gender_mode'].mode()[0], inplace=True)


# Verify the two methods
# print("\nSample of Gender_random column after imputation:")
# print(my_df['Gender_random'].sample(10))
# print("\nSample of Gender_mode column after imputation:")
# print(my_df['Gender_mode'].sample(10))


# Compare original vs imputed distribution
original_distribution = my_df['Gender'].value_counts(normalize=True)

# Distribution after each method
random_distribution = my_df['Gender_random'].value_counts(normalize=True)
mode_distribution = my_df['Gender_mode'].value_counts(normalize=True)

# Print distributions
# print("Original Gender Distribution:")
# print(original_distribution)
# print("\nGender Distribution After Random Sampling (Method 1):")
# print(random_distribution)
# print("\nGender Distribution After Mode Imputation (Method 2):")
# print(mode_distribution)
#
# # Assessing consistency in distribution
# print("\nDistribution Difference (Method 1 - Random):")
# print(abs(original_distribution - random_distribution).dropna())
# print("\nDistribution Difference (Method 2 - Mode):")
# print(abs(original_distribution - mode_distribution).dropna())

# Generate dummy variables for Gender_random
gender_random_dummies = pd.get_dummies(my_df['Gender_random'], prefix='Gender_random')
# Generate dummy variables for Gender_mode
gender_mode_dummies = pd.get_dummies(my_df['Gender_mode'], prefix='Gender_mode')
# Concatenate the dummy variables back to the original DataFrame
my_df = pd.concat([my_df, gender_random_dummies, gender_mode_dummies], axis=1)

# my_df.info()



# Why Random Sampling is Preferred:
# 1. Random sampling maintains proportionality and preserves relationships with key features (e.g., conscientiousness, sales, and stress).
# 2. Mode imputation over-represents the most frequent category (Female), leading to subtle distortions in relationships with other variables.
# 3. Random sampling ensures that less frequent categories (Male, Other) are appropriately represented, capturing natural variability in the data.



# Remove 'Gender' and 'Gender_mode' columns as 'Gender_random' was chosen
columns_to_remove = ['Gender', 'Gender_mode', 'Gender_mode_Female', 'Gender_mode_Male',
       'Gender_mode_Other' ]
my_df = my_df.drop(columns=columns_to_remove, errors='ignore')
# Rename Gender_random column to Gender
my_df.rename(columns={"Gender_random": "Gender"}, inplace=True)
# Verify that the columns were removed and renamed
# print(my_df.columns)
# my_df.sample(5)

writer.writerow([f'Missing values in the Gender column (2,145 nulls) were handled using two methods: '
                 f'random assignment based on proportional representation and mode imputation. '
                 f'This ensured consistency in distributions across the dataset.'])

writer.writerow([f'The random sampling method was chosen over mode imputation because it maintained proportionality, '
                 f'preserved relationships with key features, and ensured less frequent categories like Male and Other '
                 f'were appropriately represented, avoiding distortions.'])



### RUNNING CORRELATION MATRIX - will have to run again after cleaning data ###

# my_columns = my_df.select_dtypes(exclude='object').columns
# corrMatrix = my_df[my_columns].corr()
# print(corrMatrix)
# full_path = "CorrelationMatrix.xlsx"
# corrMatrix.to_excel(full_path)
#
# plt.close('all')
# plt.figure(figsize=(50, 50))
# sns.set(font_scale=2)
# sns.heatmap(corrMatrix, annot=True)
# plt.savefig("CorrelationMatrix.png")
# plt.close('all')




### HANDLING MISSING AGE VALUES ###
# Choosing the right method to handle missing age data is critical,
# as older and younger customers often exhibit distinct trends in their nutritional habits.
# There are no variables that are significantly correlated age so gender as it is a skewed dataset of mostly females and
# sales as older people tend to spend more based on the assumption they have more money and are more health conscious

# Step 1: Analyze age distribution
# print(my_df["age_delta1"].value_counts().sort_index())

# # Step 2: Visualize the distribution to identify natural groupings
# plt.figure(figsize=(10, 6))
# sns.histplot(my_df["age_delta1"], bins=30, kde=True, color="blue", alpha=0.7)
# plt.xlabel("Age")
# plt.ylabel("Frequency")
# plt.title("Age Distribution ")
# plt.savefig("Age Distribution.png")  # Save the plot to a file
# plt.close()  # Close the plot to avoid additional rendering

# The distribution of Age  appears fairly uniform, suggesting that age is evenly distributed across the range so


# Median simpler approach
my_df["age_delta1_median"] = my_df["age_delta1"]
my_df["age_delta2_median"] = my_df["age_delta2"]

my_df["age_delta1_median"] = my_df.groupby("Gender")["age_delta1_median"].transform(
    lambda x: x.fillna(x.median())
)
my_df["age_delta2_median"] = my_df.groupby("Gender")["age_delta2_median"].transform(
    lambda x: x.fillna(x.median())
)

# Sales-based approach
sales_stats = {
    "Sales_delta1": {
        "min": my_df["Sales_delta1"].min(),
        "max": my_df["Sales_delta1"].max(),
        "mean": my_df["Sales_delta1"].mean(),
        "median": my_df["Sales_delta1"].median(),
        "std": my_df["Sales_delta1"].std(),
        "quartiles": np.percentile(my_df["Sales_delta1"], [25, 50, 75]),
    },
    "Sales_delta2": {
        "min": my_df["Sales_delta2"].min(),
        "max": my_df["Sales_delta2"].max(),
        "mean": my_df["Sales_delta2"].mean(),
        "median": my_df["Sales_delta2"].median(),
        "std": my_df["Sales_delta2"].std(),
        "quartiles": np.percentile(my_df["Sales_delta2"], [25, 50, 75]),
    },
}

# Define the quartiles
sales_delta1_q1, sales_delta1_q2, sales_delta1_q3 = sales_stats["Sales_delta1"]["quartiles"]
sales_delta2_q1, sales_delta2_q2, sales_delta2_q3 = sales_stats["Sales_delta2"]["quartiles"]

# Define sales categories
def categorize_sales(value, q1, q2, q3):
    if value <= q1:
        return "No Sales"
    elif q1 < value <= q2:
        return "Low Sales"
    elif q2 < value <= q3:
        return "Medium Sales"
    else:
        return "High Sales"

# Apply sales categories
my_df["Sales_delta1_Category"] = my_df["Sales_delta1"].apply(
    categorize_sales, args=(sales_delta1_q1, sales_delta1_q2, sales_delta1_q3)
)
my_df["Sales_delta2_Category"] = my_df["Sales_delta2"].apply(
    categorize_sales, args=(sales_delta2_q1, sales_delta2_q2, sales_delta2_q3)
)

# Impute age based on sales categories
my_df["age_delta1_sales"] = my_df["age_delta1"]
my_df["age_delta2_sales"] = my_df["age_delta2"]

my_df["age_delta1_sales"] = my_df.groupby("Sales_delta1_Category")["age_delta1_sales"].transform(
    lambda x: x.fillna(x.median())
)
my_df["age_delta2_sales"] = my_df.groupby("Sales_delta2_Category")["age_delta2_sales"].transform(
    lambda x: x.fillna(x.median())
)

# Multiple feature approach
grouping_features = ["Sales_delta2_Category", "Gender"]
# delta2_ to avoid target leakage

my_df["age_delta1_multi"] = my_df["age_delta1"]
my_df["age_delta2_multi"] = my_df["age_delta2"]

my_df["age_delta1_multi"] = my_df.groupby(grouping_features)["age_delta1_multi"].transform(
    lambda x: x.fillna(x.median())
)
my_df["age_delta2_multi"] = my_df.groupby(grouping_features)["age_delta2_multi"].transform(
    lambda x: x.fillna(x.median())
)

# Evaluate the approaches
# print("Summary of Age Imputation Approaches:")
# print("\nMedian-based Imputation:")
# print(my_df[["age_delta1", "age_delta1_median"]].describe())
#
# print("\nSales-based Imputation:")
# print(my_df[["age_delta1", "age_delta1_sales"]].describe())
#
# print("\nMultiple Feature-based Imputation:")
# print(my_df[["age_delta1", "age_delta1_multi"]].describe())

# Visualize differences between methods

# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(10, 6))
# plt.hist(my_df["age_delta1_median"], bins=30, alpha=0.5, label="Median-Based")
# plt.hist(my_df["age_delta1_sales"], bins=30, alpha=0.5, label="Sales-Based")
# plt.hist(my_df["age_delta1_multi"], bins=30, alpha=0.5, label="Multiple Features")
# plt.xlabel("Age (Delta 1)")
# plt.ylabel("Frequency")
# plt.title("Comparison of Age Imputation Methods")
# plt.legend()
# plt.savefig("age_imputation_comparison.png")
# plt.close()

# - Median-Based: Shows a sharp peak, indicating uniform imputation, which reduces variability.
# - Sales-Based: Displays a broader distribution, dynamically imputing age based on customer spending patterns.
# - Multiple Features: Produces the most realistic and nuanced distribution by combining sales, gender, and other relevant factors.

# Recommendation for the Project:
# The multiple feature-based imputation method is recommended because it leverages diverse data points,
# providing contextually accurate age estimates.

# Drop unused age imputation columns
my_df.drop(columns=["age_delta1_sales", "age_delta2_sales", "age_delta1_median", "age_delta2_median", "age_delta1",
                    "age_delta2"], inplace=True)


# We can break to into age group for easier interpretation

# Define the function to categorize ages into 4 groups
def categorize_age_four_groups(age):
    if age < 20:
        return "Adolescent"
    elif 20 <= age < 40:
        return "Young Adult"
    elif 40 <= age < 60:
        return "Middle-Aged"
    else:
        return "Senior"

# Apply the categorization to both age_delta1_multi and age_delta2_multi
my_df["age_group_delta1"] = my_df["age_delta1_multi"].apply(categorize_age_four_groups)
my_df["age_group_delta2"] = my_df["age_delta2_multi"].apply(categorize_age_four_groups)

# Dummy code age groups for correlation matrix
age_group_delta1_dummies = pd.get_dummies(my_df["age_group_delta1"], prefix="age_group_delta1")
age_group_delta2_dummies = pd.get_dummies(my_df["age_group_delta2"], prefix="age_group_delta2")

# Concatenate dummy variables back to the main DataFrame
my_df = pd.concat([my_df, age_group_delta1_dummies, age_group_delta2_dummies], axis=1)

# Verify the final DataFrame structure
# print(my_df.info())
# print(my_df.sample(4))

writer.writerow([f'To address 15,303 missing age values, we explored three imputation methods: '
                 f'median-based, sales-based, and multiple feature-based. '
                 f'Median-based imputation used Gender to calculate the median, '
                 f'sales-based imputed values using spending quartiles, and '
                 f'multiple feature-based imputation combined sales categories and Gender.'])

writer.writerow([f'Multiple feature-based imputation was recommended as it leveraged diverse data points, '
                 f'producing the most realistic and nuanced age estimates. '
                 f'While median-based imputation showed reduced variability, it oversimplified relationships. '
                 f'Sales-based imputation captured spending patterns but lacked the depth of a multi-feature approach.'])

writer.writerow([f'For interpretability, we categorized ages into four groups to enable correlation analysis.'])



## Big 5 Scores ###

big5_columns = [
    "Big5_Conscientiousness",
    "Big5_Openness",
    "Big5_Extroversion",
    "Big5_Agreeableness",
    "Big5_Neuroticism",
]

# Step 1: Show distribution of original Big 5 scores
# for col in big5_columns:
#     print(f"Distribution for {col} (Original):")
#     print(my_df[col].value_counts(normalize=True))
#     print("\n")
#
# # Visualize distributions
# plt.figure(figsize=(15, 10))
# for i, col in enumerate(big5_columns, 1):
#     plt.subplot(2, 3, i)
#     sns.histplot(my_df[col], kde=True, bins=10, color='blue', alpha=0.7)
#     plt.title(f"Distribution of {col}")
#     plt.xlabel("Score")
#     plt.ylabel("Frequency")
# plt.tight_layout()
# plt.savefig("big5_distributions.png")
# plt.close()

# The distributions show distinct peaks, indicating potential clustering of personality traits.
# For instance, conscientiousness and agreeableness have strong peaks around 3 and 4,
# while openness and extroversion show a bimodal pattern with two peaks.
# my_df.info()
# Step 3: Correlation analysis
# corr_matrix = my_df[big5_columns + ["Gender_random_Female", "Gender_random_Male", "Gender_random_Other",
#                                     "age_group_delta1_Adolescent", "age_group_delta1_Middle-Aged",
#                                     "age_group_delta1_Senior", "age_group_delta1_Young Adult"]].corr()
# print(corr_matrix)
# shows Big 5 is more correlated with gender

# Step 1: Perform mean-based imputation grouped by Gender to maintain distribution and predict based on gender
for col in big5_columns:
    # Create a new column for mean-imputed values
    my_df[f"{col}_mean_imputed"] = my_df[col]

    # Group by Gender and calculate the mean for each group
    for gender, group_data in my_df.groupby("Gender"):
        mean_value = group_data[col].mean()  # Calculate mean for the group

        # Round the mean to the nearest integer to keep the original scoring system
        rounded_mean = round(mean_value)

        # Assign the rounded mean to missing values in the same gender group
        missing_indices = group_data[group_data[col].isnull()].index
        my_df.loc[missing_indices, f"{col}_mean_imputed"] = rounded_mean

# for col in big5_columns:
#     print(f"Value counts for {col}_mean_imputed before assigning groups:")
#     print(my_df[f"{col}_mean_imputed"].value_counts(normalize=True))
#     print("\n")

# Step 2: Assign scores to groups (Low, Neutral, High)
score_groups = {
    1: "Low",
    2: "Low",
    3: "Neutral",
    4: "Neutral",
    5: "High"
}

for col in big5_columns:
    my_df[f"{col}_group"] = my_df[f"{col}_mean_imputed"].map(score_groups)

# Step 3: Show distribution after assigning to groups
# for col in big5_columns:
#     print(f"Distribution for {col}_group:")
#     print(my_df[f"{col}_group"].value_counts(normalize=True))
#     print("\n")

# # Step 4: Visualize distributions of groups
# plt.figure(figsize=(15, 10))
# for i, col in enumerate(big5_columns, 1):
#     plt.subplot(2, 3, i)
#     sns.countplot(data=my_df, x=f"{col}_group", order=["Low", "Neutral", "High"], palette="Set2")
#     plt.title(f"Distribution of {col} Groups (After Imputation)")
#     plt.xlabel("Group")
#     plt.ylabel("Count")
# plt.tight_layout()
# plt.savefig("big5_group_distributions.png")
# plt.close()
#
# # Step 5: Visualize numeric distributions of mean-imputed columns
# plt.figure(figsize=(15, 10))
# for i, col in enumerate(big5_columns, 1):
#     plt.subplot(2, 3, i)
#     sns.histplot(my_df[f"{col}_mean_imputed"], kde=True, bins=5, color='green', alpha=0.7)
#     plt.title(f"Distribution of {col} (Mean Imputed, Rounded)")
#     plt.xlabel("Score")
#     plt.ylabel("Frequency")
# plt.tight_layout()
# plt.savefig("big5_mean_imputed_distributions.png")
# plt.close()

# print(my_df.info())

# List of columns to delete
columns_to_delete = [
    "Big5_Conscientiousness",
    "Big5_Openness",
    "Big5_Extroversion",
    "Big5_Agreeableness",
    "Big5_Neuroticism",
    "Big5_Conscientiousness_mean_imputed",
    "Big5_Openness_mean_imputed",
    "Big5_Extroversion_mean_imputed",
    "Big5_Agreeableness_mean_imputed",
    "Big5_Neuroticism_mean_imputed"
]

# Drop columns from the DataFrame
my_df.drop(columns=columns_to_delete, inplace=True)


# List of Big 5 group columns to dummy encode
group_columns = [
    "Big5_Conscientiousness_group",
    "Big5_Openness_group",
    "Big5_Extroversion_group",
    "Big5_Agreeableness_group",
    "Big5_Neuroticism_group"
]

# Perform dummy coding for the group columns
my_df = pd.get_dummies(my_df, columns=group_columns, prefix=group_columns, drop_first=True)

# # Verify the resulting DataFrame
# print(my_df.info())

writer.writerow([f'To handle missing values in Big 5 personality scores, we performed mean-based imputation grouped by Gender. '
                 f'This approach preserved the distributions while maintaining relationships between Gender and personality traits.'])

writer.writerow([f'Post-imputation, we categorized scores into groups (Low, Neutral, High) for interpretability. '
                 f'Visualizations showed clear distributions across the groups, and dummy variables were created for group analysis.'])

writer.writerow([f'Original Big 5 columns and intermediate imputed columns were removed after dummy encoding, '
                 f'reducing redundancy and streamlining the dataset for downstream analysis.'])


### BMI ###

# convert to numerical values
my_df['Medical.BMI_delta1'] = pd.to_numeric(my_df['Medical.BMI_delta1'], errors='coerce')
my_df['Medical.BMI_delta2'] = pd.to_numeric(my_df['Medical.BMI_delta2'], errors='coerce')

# List of BMI columns
bmi_columns  = [
    "Medical.BMI_delta1",
    "Medical.BMI_delta2"
]

# Step 1: Show distribution of original BMI variables
# for col in bmi_columns:
#     print(f"Distribution for {col} (Original):")
#     print(my_df[col].value_counts(normalize=True, dropna=False))  # Include NaN counts for reference
#     print("\n")


# Define a function to categorize BMI
def categorize_bmi(bmi_value):
    if pd.isnull(bmi_value):
        return "Unknown"  # Handle NaN values
    elif bmi_value < 18.5:
        return "Underweight"
    elif 18.5 <= bmi_value <= 24.9:
        return "Normal"
    elif 25 <= bmi_value <= 29.9:
        return "Overweight"
    elif bmi_value >= 30:
        return "Obese"
    else:
        return "Unknown"

# Apply the function to create new categorical columns
for col in bmi_columns:
    my_df[f"{col}_category"] = my_df[col].apply(categorize_bmi)

# Step 1: Show distribution of BMI categories
# for col in bmi_columns:
#     print(f"Distribution for {col}_category:")
#     print(my_df[f"{col}_category"].value_counts(normalize=True, dropna=False))
#     print("\n")

# Step 2: Visualize the distribution of BMI categories
# plt.figure(figsize=(15, 5))
# for i, col in enumerate(bmi_columns, 1):
#     plt.subplot(1, 2, i)
#     sns.countplot(
#         data=my_df,
#         x=f"{col}_category",
#         order=["Underweight", "Normal", "Overweight", "Obese", "Unknown"],
#         palette="Set2",
#         hue=f"{col}_category",
#         dodge=False
#     )
#     plt.title(f"Distribution of {col} Categories")
#     plt.xlabel("BMI Category")
#     plt.ylabel("Frequency")
#     plt.xticks(rotation=45)
#     plt.legend([], [], frameon=False)  # Remove redundant legend
# plt.tight_layout()
# plt.savefig("bmi_category_distributions_fixed.png")
# shows slight skewness, indicating that using the mean could bias the results toward higher or lower BMI values

# Checking correlations

# my_columns = my_df.select_dtypes(exclude='object').columns
# corrMatrix = my_df[my_columns].corr()
# print(corrMatrix)
# full_path = "CorrelationMatrix.xlsx"
# corrMatrix.to_excel(full_path)
# Sales_delta1 (-0.107): Negative correlation with sales performance. Can't use because of potential target leakage.
# Gender_random_Male (-0.100) and Gender_random_Female (0.098): Indicates a mild association with gender.
# Sleep.REM_delta2 (-0.085): Negative correlation with REM sleep.
# Stress_delta2 (0.076): Positive association with stress levels.

# Approach: using the median within groups would ensure that the imputed BMI aligns more closely with the typical distribution of the respective BMI category.
my_df['Medical.BMI_delta1'] = my_df.groupby('Gender')['Medical.BMI_delta1'].transform(
    lambda x: x.fillna(x.median())
)
my_df['Medical.BMI_delta2'] = my_df.groupby('Gender')['Medical.BMI_delta2'].transform(
    lambda x: x.fillna(x.median())
)

# Categorize BMI values
def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal weight'
    elif 25 <= bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

# Apply categorization
my_df['BMI_Category_delta1'] = my_df['Medical.BMI_delta1'].apply(categorize_bmi)
my_df['BMI_Category_delta2'] = my_df['Medical.BMI_delta2'].apply(categorize_bmi)


# Step 3: Show distribution of BMI categories
# for col in ['BMI_Category_delta1', 'BMI_Category_delta2']:
#     print(f"Distribution for {col}:")
#     print(my_df[col].value_counts(normalize=True))
#     print("\n")

# Step 4: Visualize BMI category distributions
# plt.figure(figsize=(12, 6))
# for i, col in enumerate(['BMI_Category_delta1', 'BMI_Category_delta2'], 1):
#     plt.subplot(1, 2, i)
#     sns.countplot(data=my_df, x=col, order=['Underweight', 'Normal weight', 'Overweight', 'Obese'], palette='Set2')
#     plt.title(f"Distribution of {col}")
#     plt.xlabel("BMI Category")
#     plt.ylabel("Count")
#     plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig("bmi_category_distributions_after_imputation_mean.png")

# Dummy code BMI categories
bmi_dummies_delta1 = pd.get_dummies(my_df['BMI_Category_delta1'], prefix='BMI_delta1', drop_first=True)
bmi_dummies_delta2 = pd.get_dummies(my_df['BMI_Category_delta2'], prefix='BMI_delta2', drop_first=True)

# Concatenate dummies with the main DataFrame
my_df = pd.concat([my_df, bmi_dummies_delta1, bmi_dummies_delta2], axis=1)



columns_to_drop = [
    "Medical.BMI_delta2",
    "Medical.BMI_delta1",
    "Medical.BMI_delta1_category",
    "Medical.BMI_delta2_category"
]

# Drop the columns
my_df.drop(columns=columns_to_drop, inplace=True)
# Verify the data is correct
# print(my_df.info())
# print(my_df.sample(10))

writer.writerow([f'BMI data was converted to numerical values and categorized into groups: Underweight, Normal weight, Overweight, and Obese, '
                 f'with "Unknown" for missing values. Distribution analysis revealed a slight skew, necessitating careful imputation.'])

writer.writerow([f'median imputation was applied within Gender groups to handle missing BMI values, preserving natural distribution trends. '
                 f'Post-imputation, BMI categories showed improved alignment with expected distributions.'])

writer.writerow([f'Dummy variables for BMI categories were created to facilitate analysis, and redundant columns were dropped, '
                 f'ensuring a cleaner dataset for modeling and interpretation.'])

## Blood Pressure Analysis ###


# List of Blood Pressure (BP) columns
bp_columns = ["Medical.Bloodpressure_delta1", "Medical.Bloodpressure_delta2"]


# Step 1: Show distribution of original blood pressure variables
# for col in bp_columns:
#     print(f"Distribution for {col} (Original):")
#     print(my_df[col].value_counts(normalize=True, dropna=False))  # Include NaN counts for reference
#     print("\n")

# Step 2: Visualize original distributions
# plt.figure(figsize=(15, 5))
# for i, col in enumerate(bp_columns, 1):
#     plt.subplot(1, 2, i)
#     sns.countplot(data=my_df, x=col, order=my_df[col].value_counts().index, palette="Set2")
#     plt.title(f"Distribution of {col} (Original)")
#     plt.xlabel("Blood Pressure Category")
#     plt.ylabel("Frequency")
#     plt.xticks(rotation=45)
#
# plt.tight_layout()
# plt.savefig("bp_original_dist.png")
# plt.show()


# Step 3: Define the grouping feature (age group)
grouping_features = ["age_group_delta2", "Gender"]

import random

def grouped_random_imputation(df, columns, group_features):
    """
     This function imputes missing values using random sampling based on the distribution
     of categories within the specified group. It ensures that the imputed values
     reflect the diversity of categories in the data, avoiding overrepresentation of the
     most frequent category (e.g., 'Normal' in this case) and maintaining the proportions
     of all categories ('Low', 'Normal', 'High').

     This method was chosen because mode imputation tends to bias the results toward
     the most common value (in this case, 'Normal'), leading to underrepresentation of
     less frequent categories like 'Low' and 'High.' Random sampling ensures a fairer
     distribution of imputed values within each group.
     """
    for col in columns:
        # Create a new column for imputed values
        df[f"{col}_imputed"] = df[col]

        # Fill missing values based on the group distribution
        for group_values, group_data in df.groupby(group_features):
            # Get category proportions for the group
            proportions = group_data[col].value_counts(normalize=True)
            categories = proportions.index
            weights = proportions.values

            # Impute missing values with random sampling based on group distribution
            missing_indices = group_data[group_data[col].isnull()].index
            imputed_values = random.choices(categories, weights=weights, k=len(missing_indices))

            df.loc[missing_indices, f"{col}_imputed"] = imputed_values

    return df


# Apply random sampling imputation
imputed_df_random = grouped_random_imputation(my_df.copy(), bp_columns, grouping_features)

# Verify null values before and after imputation
# for col in bp_columns:
#     print(f"Null values in {col} (Original): {my_df[col].isnull().sum()}")
#     print(
#         f"Null values in {col}_imputed (After Imputation - Random): {imputed_df_random[f'{col}_imputed'].isnull().sum()}")
#     print("\n")
#
# # Compare distributions before and after random imputation
# for col in bp_columns:
#     print(f"Distribution of {col} (Original):")
#     print(my_df[col].value_counts(normalize=True, dropna=False))
#     print("\n")
#
#     print(f"Distribution of {col} after random imputation (Age Group):")
#     print(imputed_df_random[f"{col}_imputed"].value_counts(normalize=True, dropna=False))
#     print("\n")

# Apply random sampling imputation directly to the original DataFrame
my_df = grouped_random_imputation(my_df, bp_columns, grouping_features)

# Replace original columns with the imputed columns
for col in bp_columns:
    # Replace original column with imputed column
    my_df[col] = my_df[f"{col}_imputed"]
    # Drop the auxiliary imputed column
    my_df.drop(columns=[f"{col}_imputed"], inplace=True)

# Verify the updated DataFrame
# for col in bp_columns:
#     print(f"Null values in {col} after imputation: {my_df[col].isnull().sum()}")
#     print(f"Distribution of {col} after imputation:")
#     print(my_df[col].value_counts(normalize=True))
#     print("\n")







## Family History ###

 # List of Family History columns
family_history_columns = [
    "FamilyHistory_Depression",
    "FamilyHistory_Diabetes",
    "FamilyHistory_HeartDisease",
    "FamilyHistory_Cancer",
    "FamilyHistory_Crohns",
    "FamilyHistory_Alzheimer",
    "FamilyHistory_Parkinsons",
    "FamilyHistory_Other"
]

# Display the number of null values in Family History columns
# for col in family_history_columns:
#     print(f"Null values in {col} : {my_df[col].isnull().sum()}")
#
# # Step 1: Show distribution of original Family History variables
# for col in family_history_columns:
#     print(f"Distribution for {col} (Original):")
#     print(my_df[col].value_counts(normalize=True, dropna=True))  # Include NaN counts for reference
#     print("\n")
#
# # Visualize distributions
# plt.figure(figsize=(15, 10))
# for i, col in enumerate(family_history_columns, 1):
#     plt.subplot(3, 3, i)
#     sns.countplot(data=my_df, x=col, order=my_df[col].value_counts().index, palette="Set3")
#     plt.title(f"Distribution of {col}")
#     plt.xlabel("Category")
#     plt.ylabel("Count")
# plt.tight_layout()
# plt.savefig("family_history_distributions_grouped_random_imputation.png")
# plt.close()

# Convert Family History columns to numeric
# Assuming 'Yes' = 1, 'No' = 0, and NaN remains as is
for col in family_history_columns:
    my_df[col] = my_df[col].map({'Yes': 1, 'No': 0})

# Re-select numerical columns for correlation analysis
numerical_columns = my_df.select_dtypes(include=['float64', 'int64']).columns

# Compute the correlation matrix
correlation_matrix = my_df[family_history_columns + list(numerical_columns)].corr()

# Extract correlations of Family History columns with other numerical features
family_history_correlations = correlation_matrix.loc[family_history_columns, numerical_columns]

# Display the correlation matrix for Family History features
# print("Correlations of Family History features with numerical features:")
# print(family_history_correlations)

# FamilyHistory_Depression	FamilyHistory_Diabetes	FamilyHistory_HeartDisease
# These three columns are the three columns that are less uniform and have slight
# correlations across multiple columns like gender, steps, stress which we can use to impute values

# Group 1: Columns with non-uniform distribution and slight correlations to other features
group_1_columns = [
    "FamilyHistory_Depression",
    "FamilyHistory_Diabetes",
    "FamilyHistory_HeartDisease"
]

# Group 2: Columns with uniform distributions
group_2_columns = [
    "FamilyHistory_Cancer",
    "FamilyHistory_Crohns",
    "FamilyHistory_Alzheimer",
    "FamilyHistory_Parkinsons",
    "FamilyHistory_Other"
]

# Grouping features for stratification - Gender and Age Group used to match distribution of data
grouping_features = ["Gender", "age_group_delta2"]

# Categorize Steps_delta1
def categorize_steps(steps):
    if steps < 4007:
        return "Low"
    elif 4007 <= steps <= 10023:
        return "Medium"
    else:
        return "High"

my_df["Steps_delta1_Category"] = my_df["Steps_delta1"].apply(categorize_steps)

# Categorize Stress_delta1
def categorize_stress(stress):
    if stress < 36:
        return "Low"
    elif 36 <= stress <= 73:
        return "Medium"
    else:
        return "High"

my_df["Stress_delta1_Category"] = my_df["Stress_delta1"].apply(categorize_stress)

# Verify the categorized columns
# print("Steps_delta1 Categories Distribution:")
# print(my_df["Steps_delta1_Category"].value_counts(normalize=True))
#
# print("\nStress_delta1 Categories Distribution:")
# print(my_df["Stress_delta1_Category"].value_counts(normalize=True))

# Apply the existing grouped_random_imputation function for Group 2 columns
my_df = grouped_random_imputation(my_df, group_2_columns, grouping_features)

# Replace original columns with the imputed values and clean up auxiliary columns
for col in group_2_columns:
    # Replace original column with imputed column
    my_df[col] = my_df[f"{col}_imputed"]
    # Drop the auxiliary imputed column
    my_df.drop(columns=[f"{col}_imputed"], inplace=True)

# Verify the results for Group 2
# for col in group_2_columns:
#     print(f"Null values in {col} after imputation: {my_df[col].isnull().sum()}")
#     print(f"Distribution of {col} after imputation:")
#     print(my_df[col].value_counts(normalize=True))
#     print("\n")

# Quick statistics for numerical columns
numerical_columns = ["Steps_delta1", "Stress_delta1"]

# Compute descriptive statistics
# for col in numerical_columns:
#     print(f"Statistics for {col}:")
#     print(my_df[col].describe())  # Basic descriptive stats
#     print(f"\nNumber of missing values in {col}: {my_df[col].isnull().sum()}")
#     print("\n")

# Categorize Steps_delta1
def categorize_steps(steps):
    if steps < 4007:
        return "Low"
    elif 4007 <= steps <= 10023:
        return "Medium"
    else:
        return "High"

my_df["Steps_delta1_Category"] = my_df["Steps_delta1"].apply(categorize_steps)

# Categorize Stress_delta1
def categorize_stress(stress):
    if stress < 36:
        return "Low"
    elif 36 <= stress <= 73:
        return "Medium"
    else:
        return "High"

my_df["Stress_delta1_Category"] = my_df["Stress_delta1"].apply(categorize_stress)

# Verify the categorized columns
# print("Steps_delta1 Categories Distribution:")
# print(my_df["Steps_delta1_Category"].value_counts(normalize=True))
#
# print("\nStress_delta1 Categories Distribution:")
# print(my_df["Stress_delta1_Category"].value_counts(normalize=True))



# Add multiple features that show correlations with Group 1 columns
grouping_features = ["Gender", "age_group_delta2", "Steps_delta1_Category", "Stress_delta1_Category"]

# Apply grouped random imputation for Group 1 columns
my_df = grouped_random_imputation(my_df, group_1_columns, grouping_features)

# Replace original columns with the imputed values and clean up auxiliary columns
for col in group_1_columns:
    # Replace original column with imputed column
    my_df[col] = my_df[f"{col}_imputed"]
    # Drop the auxiliary imputed column
    my_df.drop(columns=[f"{col}_imputed"], inplace=True)

# Verify the results for Group 1
# for col in group_1_columns:
#     print(f"Null values in {col} after imputation: {my_df[col].isnull().sum()}")
#     print(f"Distribution of {col} after imputation:")
#     print(my_df[col].value_counts(normalize=True))
#     print("\n")

# Final overview of DataFrame
# my_df.info()

# Family History
writer.writerow([
    "Family History columns were imputed using grouped random sampling based on Gender, Age group, Steps, and Stress categories. "
    "This approach maintained category proportions and relationships with relevant features. Post-imputation, numerical values were converted "
    "to boolean, and dummy variables were generated for analysis."
])




### RUNNING CORRELATION MATRIX  ###
#
# my_columns = my_df.select_dtypes(exclude='object').columns
# corrMatrix = my_df[my_columns].corr()
# print(corrMatrix)
# full_path = "CorrelationMatrix.xlsx"
# corrMatrix.to_excel(full_path)

# plt.close('all')
# plt.figure(figsize=(50, 50))
# sns.set(font_scale=2)
# sns.heatmap(corrMatrix, annot=True)
# plt.savefig("CorrelationMatrix.png")
# plt.close('all')
# # Correlation Matrix
writer.writerow([
    "The correlation matrix was recomputed after data cleaning and imputation to assess relationships between features. "
    "Correlation heatmaps were saved for visual inspection."
])



family_columns = ['FamilyHistory_HeartDisease', 'FamilyHistory_Depression', 'FamilyHistory_Diabetes',
                  'FamilyHistory_Cancer', 'FamilyHistory_Crohns', 'FamilyHistory_Alzheimer', 'FamilyHistory_Parkinsons', 'FamilyHistory_Other']
my_df[family_columns] = my_df[family_columns].astype(bool)



if 'Medical.Bloodpressure_delta1' in my_df.columns:
    bp_dummies_1 = pd.get_dummies(my_df['Medical.Bloodpressure_delta1'], prefix='BP_delta_1', drop_first=True)
    my_df = pd.concat([my_df, bp_dummies_1], axis=1)
else:
    print("Skipping dummy coding for 'Medical.Bloodpressure_delta1' as the column is missing.")

if 'Medical.Bloodpressure_delta2' in my_df.columns:
    bp_dummies_2 = pd.get_dummies(my_df['Medical.Bloodpressure_delta2'], prefix='BP_delta_2', drop_first=True)
    my_df = pd.concat([my_df, bp_dummies_2], axis=1)
else:
    print("Skipping dummy coding for 'Medical.Bloodpressure_delta2' as the column is missing.")


# Drop the original columns if they exist
columns_to_drop = ['Medical.Bloodpressure_delta1', 'Medical.Bloodpressure_delta2']
my_df.drop(columns=[col for col in columns_to_drop if col in my_df.columns], inplace=True)

# my_df.info()
# Final Cleanup
writer.writerow([
    "Unused columns such as original dates, demographic categories, and intermediary variables were dropped to streamline the dataset. "
    "The final dataset was verified for consistency and readiness for modeling."
])



unused_columns = ['ExtractDate', 'dob', 'Gender', 'BMI_Category_delta1', 'BMI_Category_delta2', 'age_group_delta1',
                  'age_group_delta2', 'Sales_delta1_Category', 'Sales_delta2_Category', 'Steps_delta1_Category',
                  'Stress_delta1_Category', 'age_group_delta1_Middle-Aged',  'age_group_delta2_Middle-Aged',
                  'age_group_delta1_Senior', 'age_group_delta2_Senior', 'age_group_delta1_Young Adult',
                  'age_group_delta2_Young Adult', 'age_group_delta1_Adolescent', 'age_group_delta2_Adolescent',
                  'Gender_random_Other', 'CustomerID']
my_df.drop(columns=unused_columns, inplace=True)
# my_df.info()


# my_df.sample(5)



##### MODELING #######

###  Create a numpy array of the features that we will use in the cluster analysis ###
# Extract features from the pandas my_data and convert them to a numpy array

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Define feature columns and target variable
my_cols = [
    'FamilyHistory_Depression', 'FamilyHistory_Diabetes', 'FamilyHistory_HeartDisease',
    'FamilyHistory_Cancer', 'FamilyHistory_Crohns', 'FamilyHistory_Alzheimer',
    'FamilyHistory_Parkinsons', 'FamilyHistory_Other', 'Sales_delta2',
    'Medical.OfficeVisits_delta2', 'SocialMedia.Likes_delta2', 'SocialMedia.Shares_delta2',
    'WebsiteVisits_delta2', 'MobileAppLogins_delta2', 'Steps_delta2',
    'Sleep.Deep_delta2', 'Sleep.Light_delta2', 'Sleep.REM_delta2', 'HeartRate_delta2',
    'Stress_delta2', 'LifeStyle.Drink_delta2_Excessive', 'LifeStyle.Drink_delta2_None',
    'LifeStyle.Drink_delta2_Unknown', 'LifeStyle.Smoke_delta2_Unknown',
    'LifeStyle.Smoke_delta2_Yes', 'Gender_random_Female', 'Gender_random_Male',
    'age_delta2_multi', 'Big5_Conscientiousness_group_Low', 'Big5_Conscientiousness_group_Neutral',
    'Big5_Openness_group_Low', 'Big5_Openness_group_Neutral', 'Big5_Extroversion_group_Low',
    'Big5_Extroversion_group_Neutral', 'Big5_Agreeableness_group_Low', 'Big5_Agreeableness_group_Neutral',
    'Big5_Neuroticism_group_Low', 'Big5_Neuroticism_group_Neutral', 'BMI_delta2_Obese',
    'BMI_delta2_Overweight', 'BMI_delta2_Underweight', 'BP_delta_2_Low', 'BP_delta_2_Normal',
]
# print(len(my_cols))

# Separate the features (f) and target (t)
f = my_df[my_cols].values  # Features
writer.writerow([f"Extracted features {my_cols} for clustering analysis into a numpy array."])
t = my_df['Sales_delta1'].values  # Target (not included in PCA)
target_column = 'Sales_delta1'
writer.writerow([f"Separated target column {t} from features. Features prepared for modeling."])

# Standard Scaling
scaler = StandardScaler()
f_scaled = scaler.fit_transform(f)  # Scale the feature matrix

# Perform PCA
pca = PCA(random_state=6543423, n_components=17)
f_pca = pca.fit_transform(f_scaled)
writer.writerow([f"Performed PCA on scaled features. Explained variance ratio: {pca.explained_variance_ratio_}."])

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
total_explained_variance = np.sum(explained_variance_ratio)
writer.writerow([f"Retained {len(explained_variance_ratio)} principal components explaining "
                 f"{total_explained_variance:.2f} variance."])

# Display results
print(f"Number of Principal Components Retained: {len(explained_variance_ratio)}")
print(f"Total Explained Variance by Selected Components: {total_explained_variance:.2f}")

# Print variance for each principal component
print('Variance Explained by Each Principal Component:')
print('----------------------------------------------')
for idx, variance in enumerate(explained_variance_ratio):
    print(f"PC{idx+1}: {variance * 100:.2f}%")

# Print the rotation matrix and the coefficients for each feature
print('\nVariance Contribution:  Projected Dimensions')
print('--------------------------------------------')

for idx, row in enumerate(pca.components_):  # Use `pca.components_` here
    output = f'PC{idx+1} ({100.0 * explained_variance_ratio[idx]:4.1f}%):'
    for val, name in zip(row, my_cols):
        if output.strip()[-1] == ":":
            output += f" {val:5.2f} * {name:s}"
        else:
            output += f" + {val:5.2f} * {name:s}"
    print(output)


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Determine optimal number of clusters using the Elbow Method
inertia = []
range_clusters = range(1, 11)  # Try clustering with 1 to 10 clusters

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(f_pca)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
# plt.figure(figsize=(10, 6))
# plt.plot(range_clusters, inertia, marker='o')
# plt.title('Elbow Method for Optimal k (KMeans)')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Inertia')
# plt.grid()
# plt.savefig('kmeans_elbow_curve.png')
# plt.close()

optimal_k = 2
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(f_pca)

# Assign cluster labels
kmeans_labels = kmeans.labels_
writer.writerow([f"Performed KMeans clustering with {optimal_k} clusters."])


# Evaluate clustering performance with Silhouette Score
silhouette_avg_kmeans = silhouette_score(f_pca, kmeans_labels)
print(f"KMeans Silhouette Score with {optimal_k} Clusters: {silhouette_avg_kmeans:.2f}")
writer.writerow([f"KMeans clustering silhouette score: {silhouette_avg_kmeans:.4f}."])

# Visualize clusters using the first two principal components
# plt.figure(figsize=(10, 6))
# plt.scatter(f_pca[:, 0], f_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
# plt.title(f'KMeans Clustering (k={optimal_k})')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.colorbar(label='Cluster')
# plt.grid()
# plt.savefig('kmeans_clusters.png')
# plt.close()


#
# from pyclustering.cluster.kmedians import kmedians
# from pyclustering.utils.metric import distance_metric, type_metric
# from sklearn.metrics import silhouette_score
# import numpy as np
#
# # Use PCA-transformed data or scaled data
# data = f_pca  # Or use f_scaled if not using PCA
#
# # Initialize the K-Medians model (e.g., k=2 for two clusters)
# initial_centers = data[np.random.choice(len(data), size=2, replace=False)]  # Random initial centers
# metric = distance_metric(type_metric.MANHATTAN)  # Use Manhattan distance for K-Medians
# kmedians_model = kmedians(data, initial_centers, metric=metric)
#
# # Process the clustering
# kmedians_model.process()
#
# # Get the cluster assignments
# clusters = kmedians_model.get_clusters()
# labels = np.zeros(len(data), dtype=int) - 1  # Default label is -1 for unassigned
# for cluster_id, cluster_indices in enumerate(clusters):
#     labels[cluster_indices] = cluster_id
#
# # Silhouette Score (only valid if at least two clusters)
# if len(np.unique(labels)) > 1:
#     kmedians_silhouette = silhouette_score(data, labels, metric='manhattan')  # Manhattan distance
#     print(f"K-Medians Silhouette Score: {kmedians_silhouette:.4f}")
# else:
#     print("K-Medians did not form distinct clusters.")
# writer.writerow([f"KMedians clustering silhouette score: {kmedians_silhouette:.4f}."])
#
#
# from sklearn.cluster import DBSCAN
# from sklearn.metrics import silhouette_score
# from collections import Counter
# import numpy as np
#
# # PCA-transformed data (f_pca) from your project
# data = f_pca
#
# # Range of hyperparameters for DBSCAN
# eps_values = [7.5]  # Epsilon values
# min_samples_values = [10]  # Min samples
#
# # Iterate through combinations of hyperparameters
# for eps in eps_values:
#     for min_samples in min_samples_values:
#         # Initialize DBSCAN
#         dbscan = DBSCAN(eps=eps, metric='euclidean', min_samples=min_samples)
#
#         # Fit and predict clusters
#         dbscan_labels = dbscan.fit_predict(data)
#
#         # Count the clusters (excluding noise, labeled as -1)
#         cluster_counts = Counter(dbscan_labels)
#
#         # Only calculate silhouette score if more than one cluster is found
#         if len(set(dbscan_labels)) > 1:
#             silhouette_avg = silhouette_score(data, dbscan_labels, metric='euclidean')
#             print(f"Silhouette Score for eps={eps}, min_samples={min_samples}: {silhouette_avg:.3f} "
#                   f"with clusters {cluster_counts}")
#         else:
#             print(f"DBSCAN with eps={eps}, min_samples={min_samples} resulted in no valid clusters.")
#
#
#
#
# # DBSCAN with best parameters
# best_dbscan = DBSCAN(eps=7.5, min_samples=10, metric='euclidean')
# best_labels = best_dbscan.fit_predict(f_pca)
#
#
# # Plot clusters in 2D using first two PCA components
# plt.figure(figsize=(10, 6))
# unique_labels = set(best_labels)
# colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
#
# for label, color in zip(unique_labels, colors):
#     if label == -1:  # Noise
#         color = [0, 0, 0, 1]  # Black for noise
#     plt.scatter(
#         f_pca[best_labels == label, 0],
#         f_pca[best_labels == label, 1],
#         c=[color],
#         label=f"Cluster {label}" if label != -1 else "Noise",
#         alpha=0.6
#     )
#
# plt.title(f"DBSCAN Clustering (eps=7.5, min_samples=10)\nSilhouette Score: {0.290}")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.legend()
#
# # Save the plot as a PNG file
# plt.savefig("dbscan_clustering_eps_7.5_min_samples_10.png")
# plt.close()

import pickle

# # Step 1: Add DBSCAN clusters to the dataframe
# my_df['DBSCAN_Cluster'] = best_labels
#
# # Save DBSCAN clusters
# with open('dbscan_labels.pkl', 'wb') as dbscan_file:
#     pickle.dump(best_labels, dbscan_file)
# print("DBSCAN labels saved.")
# writer.writerow(["DBSCAN cluster labels saved to 'dbscan_labels.pkl'."])
#
#
# # Step 2: Dummy code the DBSCAN clusters
# # Convert the DBSCAN cluster labels into dummy/one-hot encoded features
# my_df_dummies = pd.get_dummies(my_df, columns=['DBSCAN_Cluster'], prefix='Cluster', drop_first=True)
#
# # Step 3: Final PCA preparation
# # Define the columns to use for the final PCA
# final_cols = [
#     'FamilyHistory_Depression', 'FamilyHistory_Diabetes', 'FamilyHistory_HeartDisease',
#     'FamilyHistory_Cancer', 'FamilyHistory_Crohns', 'FamilyHistory_Alzheimer',
#     'FamilyHistory_Parkinsons', 'FamilyHistory_Other', 'Sales_delta2',
#     'Medical.OfficeVisits_delta2', 'SocialMedia.Likes_delta2', 'SocialMedia.Shares_delta2',
#     'WebsiteVisits_delta2', 'MobileAppLogins_delta2', 'Steps_delta2',
#     'Sleep.Deep_delta2', 'Sleep.Light_delta2', 'Sleep.REM_delta2', 'HeartRate_delta2',
#     'Stress_delta2', 'LifeStyle.Drink_delta2_Excessive', 'LifeStyle.Drink_delta2_None',
#     'LifeStyle.Drink_delta2_Unknown', 'LifeStyle.Smoke_delta2_Unknown',
#     'LifeStyle.Smoke_delta2_Yes', 'Gender_random_Female', 'Gender_random_Male',
#     'age_delta2_multi', 'Big5_Conscientiousness_group_Low', 'Big5_Conscientiousness_group_Neutral',
#     'Big5_Openness_group_Low', 'Big5_Openness_group_Neutral', 'Big5_Extroversion_group_Low',
#     'Big5_Extroversion_group_Neutral', 'Big5_Agreeableness_group_Low', 'Big5_Agreeableness_group_Neutral',
#     'Big5_Neuroticism_group_Low', 'Big5_Neuroticism_group_Neutral', 'BMI_delta2_Obese',
#     'BMI_delta2_Overweight', 'BMI_delta2_Underweight', 'BP_delta_2_Low', 'BP_delta_2_Normal'
# ] + [col for col in my_df_dummies.columns if col.startswith('Cluster_')]
# print(final_cols)
# # Extract the features for PCA
# final_features = my_df_dummies[final_cols].values
# print(final_features)
#
# # Step 4: Standard Scaling
# scaler = StandardScaler()
# final_features_scaled = scaler.fit_transform(final_features)
#
# # Step 5: Perform the final PCA
# final_pca = PCA(random_state=6543423, n_components='mle')
# final_features_pca = final_pca.fit_transform(final_features_scaled)
#
# # Step 6: Save the transformed features into the dataframe
# final_pca_cols = [f'PCA_{i+1}' for i in range(final_features_pca.shape[1])]
# final_pca_df = pd.DataFrame(final_features_pca, columns=final_pca_cols)
#
# # Add the PCA-transformed features back to the original dataframe
# my_df_pca_final = pd.concat([my_df_dummies.reset_index(drop=True), final_pca_df.reset_index(drop=True)], axis=1)
#
# # Step 7: Save the updated dataframe
# my_df_pca_final.to_csv('final_prepared_data.csv', index=False)
#
# # Summary of the final data preparation
# print("Final PCA Dataset Shape:", my_df_pca_final.shape)
# print("PCA Components:", len(final_pca_cols))
# print("Explained Variance Ratio:", final_pca.explained_variance_ratio_)
# print("Total Explained Variance:", np.sum(final_pca.explained_variance_ratio_))
#
# my_df.info()




from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR


# Split the dataset
f_train, f_test, t_train, t_test = train_test_split(f_pca, t, test_size=0.2, random_state=42)
writer.writerow(["Dataset split into training and testing sets with an 80/20 ratio."])

# Store model results
model_results = []

from sklearn.metrics import mean_squared_error


def evaluate_model(model_name, model, f_train, f_test, t_train, t_test):
    # Train and test scores
    train_score = model.score(f_train, t_train)
    test_score = model.score(f_test, t_test)

    # Calculate RMSE manually
    predictions = model.predict(f_test)
    mse = mean_squared_error(t_test, predictions)  # Calculate MSE
    rmse = np.sqrt(mse)  # Manually compute RMSE

    # Variance between train and test scores
    variance = train_score - test_score

    # Append results
    model_results.append({
        'Model': model_name,
        'Train R^2': train_score,
        'Test R^2': test_score,
        'Variance': variance,
        'RMSE': rmse
    })

    # Print results
    print(
        f"{model_name} - Train R^2: {train_score:.4f}, Test R^2: {test_score:.4f}, Variance: {variance:.4f}, RMSE: "
        f"{rmse:.4f}")


import pickle

# 1. KNN Regressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(f_train, t_train)
evaluate_model("KNN Regressor", knn, f_train, f_test, t_train, t_test)
writer.writerow(["Trained and evaluated KNN Regressor with n_neighbors=5."])


# 2. Decision Tree Regressor
decision_tree = DecisionTreeRegressor(random_state=42, max_depth=10, min_samples_split=10)
decision_tree.fit(f_train, t_train)
evaluate_model("Pruned Decision Tree Regressor", decision_tree, f_train, f_test, t_train, t_test)
writer.writerow(["Trained and evaluated Pruned Decision Tree Regressor with max_depth=10 and min_samples_split=10."])
from sklearn.model_selection import GridSearchCV

# Hyperparameter Tuning for Decision Tree
param_grid = {
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=DecisionTreeRegressor(random_state=42),
    param_grid=param_grid,
    scoring='r2',
    cv=5,
    n_jobs=-1
)
grid_search.fit(f_train, t_train)

best_tree = grid_search.best_estimator_
# print(f"Best Parameters: {grid_search.best_params_}")
# print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")
writer.writerow([f"Performed GridSearchCV for Decision Tree. Best parameters: {grid_search.best_params_}."])

# Evaluate the optimized model
evaluate_model("Optimized Pruned Decision Tree", best_tree, f_train, f_test, t_train, t_test)
writer.writerow(["Trained and evaluated Optimized Pruned Decision Tree using GridSearchCV results."])


# Save the optimized decision tree model
with open('optimized_pruned_decision_tree.pkl', 'wb') as Prediction_model_file:
    pickle.dump(best_tree, Prediction_model_file)
writer.writerow(["Saved Optimized Pruned Decision Tree model as 'optimized_pruned_decision_tree.pkl'."])
# print("Optimized Pruned Decision Tree model saved as 'optimized_pruned_decision_tree.pkl'")



# 3. Random Forest Regressor
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest.fit(f_train, t_train)
evaluate_model("Random Forest Regressor", random_forest, f_train, f_test, t_train, t_test)
writer.writerow([
        "Random Forest Regressor",
        "Random Forest model was instantiated with 100 estimators and trained on the training data."
    ])


# 4. SGD Regressor
sgd_regressor = SGDRegressor(
    random_state=42, max_iter=1000, tol=1e-3, eta0=0.001, learning_rate='invscaling'
)
sgd_regressor.fit(f_train, t_train)
evaluate_model("SGD Regressor", sgd_regressor, f_train, f_test, t_train, t_test)
writer.writerow([
    "SGD Regressor",
    "Stochastic Gradient Descent Regressor was instantiated with learning rate as 'invscaling' and trained on the data."
])


# 5. Support Vector Regressor
svr = SVR(kernel='rbf')
svr.fit(f_train, t_train)
evaluate_model("Support Vector Regressor", svr, f_train, f_test, t_train, t_test)
writer.writerow([
        "Support Vector Regressor",
        "Support Vector Regressor with RBF kernel was trained on the training data."
    ])

# 6. Voting Regressor
voting_regressor = VotingRegressor(estimators=[
    ('knn', knn),
    ('dt', decision_tree),
    ('rf', random_forest)
])
voting_regressor.fit(f_train, t_train)
evaluate_model("Voting Regressor", voting_regressor, f_train, f_test, t_train, t_test)
writer.writerow([
        "Voting Regressor",
        "Voting Regressor model combining KNN, Decision Tree, and Random Forest was trained and evaluated."
    ])


# 7. Voting Regressor of Random Forest and Optimized Pruned Decision Tree
voting_regressor = VotingRegressor(estimators=[
    ('optimized_tree', best_tree),
    ('random_forest', random_forest)
])

voting_regressor.fit(f_train, t_train)

# Evaluate the Voting Regressor
evaluate_model("Voting Regressor (Optimized Tree + Random Forest)", voting_regressor, f_train, f_test,
               t_train, t_test)
writer.writerow([
        "Voting Regressor (Optimized Tree + Random Forest)",
        "Voting Regressor combining Optimized Decision Tree and Random Forest models was trained and evaluated."
    ])
# Compile results into a DataFrame
results_df = pd.DataFrame(model_results)
results_df = results_df.sort_values(by="Test R^2", ascending=False)
# print(results_df)


# SAVING THE MODELS AND SCALING TO PKL FILES::::
# Save the StandardScaler object
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
# print("StandardScaler saved as 'scaler.pkl'.")
# writer.writerow([
#         "Save Scaler",
#         "StandardScaler object was saved to 'scaler.pkl' for consistent scaling of new data."
#     ])

# Save the PCA object
with open('pca.pkl', 'wb') as pca_file:
    pickle.dump(pca, pca_file)
# print("PCA object saved as 'pca.pkl'.")


# Save the PCA-transformed data
with open('pca_transformed_data.pkl', 'wb') as pca_data_file:
    pickle.dump(f_pca, pca_data_file)
# print("PCA-transformed data saved as 'pca_transformed_data.pkl'.")


# # Save the DBSCAN cluster labels
# with open('dbscan_labels.pkl', 'wb') as dbscan_labels_file:
#     pickle.dump(best_labels, dbscan_labels_file)
# print("DBSCAN cluster labels saved as 'dbscan_labels.pkl'.")

# # Save the final PCA-transformed features for supervised learning
# with open('final_features_scaled.pkl', 'wb') as features_file:
#     pickle.dump(final_features_scaled, features_file)
# print("Final scaled features saved as 'final_features_scaled.pkl'.")

# # Save the target variable
# with open('target.pkl', 'wb') as target_file:
#     pickle.dump(target, target_file)
# print("Target variable saved as 'target.pkl'.")

# Save trained models
models_to_save = {
    "knn": knn,
    "decision_tree": decision_tree,
    "optimized_pruned_tree": best_tree,
    "random_forest": random_forest,
    "sgd_regressor": sgd_regressor,
    "svr": svr,
    "voting_regressor_combined": voting_regressor
}

for model_name, model_obj in models_to_save.items():
    with open(f"{model_name}.pkl", 'wb') as model_file:
        pickle.dump(model_obj, model_file)
    # print(f"{model_name} saved as '{model_name}.pkl'.")
writer.writerow([
            f"Save {model_name}",
            f"The trained model '{model_name}' was saved as '{model_name}.pkl'."
        ])
# Save GridSearchCV results
with open('grid_search_best_params.pkl', 'wb') as grid_params_file:
    pickle.dump(grid_search.best_params_, grid_params_file)
# print("GridSearchCV best parameters saved as 'grid_search_best_params.pkl'.")

with open('grid_search_best_estimator.pkl', 'wb') as grid_estimator_file:
    pickle.dump(best_tree, grid_estimator_file)
# print("GridSearchCV best estimator saved as 'grid_search_best_estimator.pkl'.")

# # Save results DataFrame
# results_df.to_csv("regression_model_comparison_results.csv", index=False)
# print("Regression model comparison results saved to 'regression_model_comparison_results.csv'.")



### PREDICTIVE MODEL ANALYSIS (WHAT-IF?)


# Load new data for prediction
input_file = 'sample_implementation.txt'
my_df1 = pd.read_csv(input_file, delimiter='|')

# Step 1: Convert boolean columns
boolean_columns = [
    'FamilyHistory_Depression', 'FamilyHistory_Diabetes', 'FamilyHistory_HeartDisease',
    'FamilyHistory_Cancer', 'FamilyHistory_Crohns', 'FamilyHistory_Alzheimer',
    'FamilyHistory_Parkinsons', 'FamilyHistory_Other', 'LifeStyle.Drink_delta2_Excessive',
    'LifeStyle.Drink_delta2_None', 'LifeStyle.Drink_delta2_Unknown',
    'LifeStyle.Smoke_delta2_Unknown', 'LifeStyle.Smoke_delta2_Yes',
    'Gender_random_Female', 'Gender_random_Male', 'Medical.OfficeVisits_delta2'
]

for col in boolean_columns:
    if col in my_df1.columns:
        my_df1[col] = my_df1[col].astype(bool)
    else:
        my_df1[col] = False  # Default to False if column is missing

# Step 2: Convert numeric columns
numeric_columns = [
    'Sales_delta2', 'SocialMedia.Likes_delta2',
    'SocialMedia.Shares_delta2', 'WebsiteVisits_delta2', 'MobileAppLogins_delta2',
    'Steps_delta2', 'Sleep.Deep_delta2', 'Sleep.Light_delta2', 'Sleep.REM_delta2',
    'HeartRate_delta2', 'Stress_delta2', 'age_delta2_multi'
]

for col in numeric_columns:
    if col in my_df1.columns:
        my_df1[col] = pd.to_numeric(my_df1[col], errors='coerce').fillna(0)
    else:
        my_df1[col] = 0  # Default to 0 if column is missing

# Step 3: Handle lifestyle and gender columns
if 'LifeStyle.Drink' in my_df1.columns:
    my_df1['LifeStyle.Drink_delta2_Excessive'] = (my_df1['LifeStyle.Drink'] == 'Excessive').astype(int)
    my_df1['LifeStyle.Drink_delta2_None'] = (my_df1['LifeStyle.Drink'] == 'None').astype(int)
    my_df1['LifeStyle.Drink_delta2_Unknown'] = (my_df1['LifeStyle.Drink'] == 'Unknown').astype(int)

if 'LifeStyle.Smoke' in my_df1.columns:
    my_df1['LifeStyle.Smoke_delta2_Unknown'] = (my_df1['LifeStyle.Smoke'] == 'Unknown').astype(int)
    my_df1['LifeStyle.Smoke_delta2_Yes'] = (my_df1['LifeStyle.Smoke'] == 'Yes').astype(int)

if 'Gender' in my_df1.columns:
    my_df1['Gender_random_Female'] = (my_df1['Gender'] == 'Female').astype(int)
    my_df1['Gender_random_Male'] = (my_df1['Gender'] == 'Male').astype(int)


# my_df1.info()
# my_df1.sample(4)

big5_columns = [
    "Big5_Conscientiousness",
    "Big5_Openness",
    "Big5_Extroversion",
    "Big5_Agreeableness",
    "Big5_Neuroticism",
]

# Step 1: Perform mean-based imputation grouped by Gender to maintain distribution and predict based on gender
for col in big5_columns:
    # Create a new column for mean-imputed values
    my_df1[f"{col}_mean_imputed"] = my_df1[col]

    # Group by Gender and calculate the mean for each group
    for gender, group_data in my_df1.groupby("Gender"):
        mean_value = group_data[col].mean()  # Calculate mean for the group

        # Round the mean to the nearest integer to keep the original scoring system
        rounded_mean = round(mean_value)

        # Assign the rounded mean to missing values in the same gender group
        missing_indices = group_data[group_data[col].isnull()].index
        my_df1.loc[missing_indices, f"{col}_mean_imputed"] = rounded_mean

# for col in big5_columns:
#     print(f"Value counts for {col}_mean_imputed before assigning groups:")
#     print(my_df1[f"{col}_mean_imputed"].value_counts(normalize=True))
#     print("\n")

# Step 2: Assign scores to groups (Low, Neutral, High)
score_groups = {
    1: "Low",
    2: "Low",
    3: "Neutral",
    4: "Neutral",
    5: "High"
}

for col in big5_columns:
    my_df1[f"{col}_group"] = my_df1[f"{col}_mean_imputed"].map(score_groups)

# # Step 3: Show distribution after assigning to groups
# for col in big5_columns:
#     print(f"Distribution for {col}_group:")
#     print(my_df1[f"{col}_group"].value_counts(normalize=True))
#     print("\n")

# List of columns to delete
columns_to_delete = [
    "Big5_Conscientiousness",
    "Big5_Openness",
    "Big5_Extroversion",
    "Big5_Agreeableness",
    "Big5_Neuroticism",
    "Big5_Conscientiousness_mean_imputed",
    "Big5_Openness_mean_imputed",
    "Big5_Extroversion_mean_imputed",
    "Big5_Agreeableness_mean_imputed",
    "Big5_Neuroticism_mean_imputed"
]

# Drop columns from the DataFrame
my_df1.drop(columns=columns_to_delete, inplace=True)

# List of Big 5 group columns to dummy encode
group_columns = [
    "Big5_Conscientiousness_group",
    "Big5_Openness_group",
    "Big5_Extroversion_group",
    "Big5_Agreeableness_group",
    "Big5_Neuroticism_group"
]

# Perform one-hot encoding (dummy encoding) for the group columns
my_df1 = pd.get_dummies(my_df1, columns=group_columns, prefix=group_columns)

# print(my_df1.info())

# my_df1.info()
# my_df1.sample(4)

# List of BMI columns

# Step 1: Impute missing BMI values with the median for each gender
my_df1['BMI'] = my_df1['BMI'].fillna(my_df1['BMI'].median())

# Step 2: Verify there are no more missing values in the BMI column
# print("Missing values in BMI column after imputation:", my_df1['BMI'].isnull().sum())

# Step 3: Categorize BMI values into groups
def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal weight'
    elif 25 <= bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

# Apply categorization and create a new column for BMI categories
my_df1['BMI_category'] = my_df1['BMI'].apply(categorize_bmi)

# Step 4: Verify the distribution of BMI categories
# print("Distribution of BMI categories:")
# print(my_df1['BMI_category'].value_counts(normalize=True))

# Display a sample of the DataFrame to confirm changes
# print(my_df1.sample(5))

bmi_dummies = pd.get_dummies(my_df1['BMI_category'], prefix='BMI_delta2', drop_first=True)

# Concatenate dummies with the main DataFrame
my_df1 = pd.concat([my_df1, bmi_dummies], axis=1)

bp_columns = ["BloodPressure"]

grouping_features = ["Gender"]

import random

def grouped_random_imputation(df, columns, group_features):
    """
     This function imputes missing values using random sampling based on the distribution
     of categories within the specified group. It ensures that the imputed values
     reflect the diversity of categories in the data, avoiding overrepresentation of the
     most frequent category (e.g., 'Normal' in this case) and maintaining the proportions
     of all categories ('Low', 'Normal', 'High').

     This method was chosen because mode imputation tends to bias the results toward
     the most common value (in this case, 'Normal'), leading to underrepresentation of
     less frequent categories like 'Low' and 'High.' Random sampling ensures a fairer
     distribution of imputed values within each group.
     """
    for col in columns:
        # Create a new column for imputed values
        df[f"{col}_imputed"] = df[col]

        # Fill missing values based on the group distribution
        for group_values, group_data in df.groupby(group_features):
            # Get category proportions for the group
            proportions = group_data[col].value_counts(normalize=True)
            categories = proportions.index
            weights = proportions.values

            # Impute missing values with random sampling based on group distribution
            missing_indices = group_data[group_data[col].isnull()].index
            imputed_values = random.choices(categories, weights=weights, k=len(missing_indices))

            df.loc[missing_indices, f"{col}_imputed"] = imputed_values

    return df

# Apply random sampling imputation
imputed_df_random = grouped_random_imputation(my_df1.copy(), bp_columns, grouping_features)

# Apply random sampling imputation directly to the original DataFrame
my_df1 = grouped_random_imputation(my_df1, bp_columns, grouping_features)

# Replace original columns with the imputed columns
for col in bp_columns:
    # Replace original column with imputed column
    my_df1[col] = my_df1[f"{col}_imputed"]
    # Drop the auxiliary imputed column
    my_df1.drop(columns=[f"{col}_imputed"], inplace=True)

# my_df1.info()

if 'BloodPressure' in my_df1.columns:
    bp_dummies_1 = pd.get_dummies(my_df1['BloodPressure'], prefix='BP_delta_2', drop_first=True)
    my_df1 = pd.concat([my_df1, bp_dummies_1], axis=1)
else:
    print("Skipping dummy coding for 'Medical.Bloodpressure_delta1' as the column is missing.")

# Drop the original columns if they exist
columns_to_drop = ['BloodPressure']
my_df1.drop(columns=[col for col in columns_to_drop if col in my_df1.columns], inplace=True)

# my_df1.sample()
# my_df1.info()
# Log loading and preprocessing
writer.writerow([
    "Load Data",
    "Loaded the input data from 'sample_implementation.txt' and started preprocessing."
])



# Step 4: Align column order with training data
import joblib


filename = 'decision_tree.pkl'
with open(filename, 'rb') as fin:
    dt_from_pkl = joblib.load(fin)

filename = 'scaler.pkl'
with open(filename, 'rb') as fin:
    dt_from_pkl = joblib.load(fin)

filename = 'pca.pkl'
with open(filename, 'rb') as fin:
    pca_from_pkl = joblib.load(fin)


# with open('scaler.pkl', 'wb') as scaler_file:
#     pickle.dump(scaler, scaler_file)
# print("StandardScaler saved as 'scaler.pkl'.")

# Ensure all required columns are present
final_cols = [
    'FamilyHistory_Depression', 'FamilyHistory_Diabetes', 'FamilyHistory_HeartDisease',
    'FamilyHistory_Cancer', 'FamilyHistory_Crohns', 'FamilyHistory_Alzheimer',
    'FamilyHistory_Parkinsons', 'FamilyHistory_Other', 'Sales_delta2',
    'Medical.OfficeVisits_delta2', 'SocialMedia.Likes_delta2', 'SocialMedia.Shares_delta2',
    'WebsiteVisits_delta2', 'MobileAppLogins_delta2', 'Steps_delta2',
    'Sleep.Deep_delta2', 'Sleep.Light_delta2', 'Sleep.REM_delta2', 'HeartRate_delta2',
    'Stress_delta2', 'LifeStyle.Drink_delta2_Excessive', 'LifeStyle.Drink_delta2_None',
    'LifeStyle.Drink_delta2_Unknown', 'LifeStyle.Smoke_delta2_Unknown',
    'LifeStyle.Smoke_delta2_Yes', 'Gender_random_Female', 'Gender_random_Male',
    'age_delta2_multi', 'Big5_Conscientiousness_group_Low', 'Big5_Conscientiousness_group_Neutral',
    'Big5_Openness_group_Low', 'Big5_Openness_group_Neutral', 'Big5_Extroversion_group_Low',
    'Big5_Extroversion_group_Neutral', 'Big5_Agreeableness_group_Low', 'Big5_Agreeableness_group_Neutral',
    'Big5_Neuroticism_group_Low', 'Big5_Neuroticism_group_Neutral', 'BMI_delta2_Obese',
    'BMI_delta2_Overweight', 'BMI_delta2_Underweight', 'BP_delta_2_Low', 'BP_delta_2_Normal',
]

missing_columns = [col for col in final_cols if col not in my_df1.columns]
if missing_columns:
    raise ValueError(f"Missing columns in input data: {missing_columns}")

# Align the data with training column order
features = my_df1[final_cols].values

# Scale the features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(features)
features_sc = sc.transform(features)

# Apply PCA transformation
features_pca = pca_from_pkl.transform(features_sc)

# Predict initial sales
initial_predictions = best_tree.predict(features_pca)
my_df1['predicted_sales_original'] = pd.Series(initial_predictions)

# Define the valid feature options for what-if analysis
feature_map = {
    "Likes": "SocialMedia.Likes_delta2",
    "Shares": "SocialMedia.Shares_delta2",
    "WebsiteVisits": "WebsiteVisits_delta2",
    "MobileAppLogins": "MobileAppLogins_delta2",
    "Steps": "Steps_delta2",
    "REM Sleep": "Sleep.REM_delta2",
    "Deep Sleep": "Sleep.Deep_delta2",
    "Light Sleep": "Sleep.Light_delta2",
    "Stress": "Stress_delta2",
}
writer.writerow([
    "What-If Analysis Loop",
    "Started the interactive what-if analysis loop, allowing users to explore feature impacts."
])
# Interactive loop for what-if analysis
while True:
    print("\nSelect a feature to analyze for the what-if scenario:")
    print("Options:", ", ".join(feature_map.keys()))
    selected_feature = input("Enter the feature (or type 'done' to finish): ").strip()

    if selected_feature.lower() == 'done':
        break

    if selected_feature not in feature_map:
        print(f"Invalid feature: '{selected_feature}'. Please ensure the spelling matches the options.")
        continue


    # Identify the feature column to modify
    feature_column = feature_map[selected_feature]

    # Create copies of the dataset for 50% increase and decrease scenarios
    df_increase = my_df1.copy()
    df_decrease = my_df1.copy()

    # Apply the changes
    df_increase[feature_column] *= 1.5
    df_decrease[feature_column] *= 0.5

    # Recalculate features and predictions for increased scenario
    features_increase = df_increase[final_cols].values
    features_sc_increase = sc.transform(features_increase)
    features_pca_increase = pca_from_pkl.transform(features_sc_increase)
    df_increase[f'predicted_sales_increased_{selected_feature}'] = best_tree.predict(features_pca_increase)

    # Recalculate features and predictions for decreased scenario
    features_decrease = df_decrease[final_cols].values
    features_sc_decrease = sc.transform(features_decrease)
    features_pca_decrease = pca_from_pkl.transform(features_sc_decrease)
    df_decrease[f'predicted_sales_decreased_{selected_feature}'] = best_tree.predict(features_pca_decrease)

    # Merge the increased and decreased predictions into the original DataFrame
    my_df1[f'predicted_sales_increased_{selected_feature}'] = df_increase[
        f'predicted_sales_increased_{selected_feature}']
    my_df1[f'predicted_sales_decreased_{selected_feature}'] = df_decrease[
        f'predicted_sales_decreased_{selected_feature}']

    print(f"\nWhat-if analysis for {selected_feature} completed. Results added to the dataset.")


writer.writerow([
    f"What-If Analysis ({selected_feature})",
    f"Performed what-if analysis for the selected feature '{selected_feature}', calculating increased and decreased predictions."
])
# Save results to Excel
filename = 'predictions.xlsx'
my_df1.to_excel(filename, index=False)
print(f"\nAll predictions saved to {filename}"
      f"\nAll meta data were saved to text file.")

writer.writerow([
        "Save Results to Excel",
        "Saved the final dataset, including original, increased, and decreased predictions, to 'predictions.xlsx'"
        "Saved the meta-data."
    ])

if not handle.closed:
    handle.close()