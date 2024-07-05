##Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OneHotEncoder,RobustScaler,LabelEncoder,OrdinalEncoder
from category_encoders import BinaryEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

## Read Data
data=pd.read_excel(r"C:\Users\etaha2\Downloads\New Model\last data\Data_base_Model_update.xlsx",sheet_name='Data_base',dtype={"Customer Code":str})

## divided data to train and validated
from sklearn.model_selection import train_test_split

# Optional Step: Further split the training data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Checking the sizes of the splits
print(f"Training set size: {train_data.shape[0]}")
print(f"Validation set size: {val_data.shape[0]}")

train_data.shape

## copy data
data_base=train_data.copy()

len(train_data[train_data.duplicated])

## check duplicated for customer code column
len(train_data[train_data.duplicated('Customer Code')])

## Remove unneeded Columns
train_data.drop(["Customer Colour","Price Group","Open Amount","Due Amount","Billing Amount","Last Hotline Date","Last Suspension Date"],axis=1,inplace=True)

train_data.info()

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

train_data.columns

Num_Col=train_data.select_dtypes(include=np.number).columns
Num_Col

Cat_Col=train_data.select_dtypes("object").columns
Cat_Col

### Univariate analysis for Num columns

train_data['Bill Cycle'].value_counts(normalize=True)

train_data['Bill Cycle']=train_data['Bill Cycle'].astype(str)

fig= plt.figure(figsize=(4,4))
train_data['Bill Cycle'].value_counts(normalize=True).plot.pie(autopct ='%1.2f%%')
plt.title('Bill Cycle', fontdict={'fontsize': 10, 'fontweight' : 2, 'color' : 'Green'})
plt.show()

- 46% of customer on BC 2 then BC1 by34% and lowest BC is 4 with 20%.

train_data['Number Of Unbilled Bills'].value_counts()

train_data['Activation Date'].value_counts()

# Assuming train_data is your DataFrame and 'Activation Date' is the column containing the date values
reference_date = pd.Timestamp('1899-12-30')  # Excel's reference date
dates = [reference_date + pd.DateOffset(days=int(date)) for date in train_data['Activation Date']]

# Create a new column 'Converted Dates' and assign the converted dates to it
train_data = train_data.assign(Converted_Dates=dates)

from datetime import date

train_data['variance_days'] = np.datetime64(date.today()) - train_data['Converted_Dates']

train_data['variance_days']=train_data['variance_days'].dt.days
train_data['variance_days']=train_data['variance_days'] // 30
train_data['variance_days'].value_counts()

train_data['Converted_Dates'].value_counts()

train_data['activation_months']=train_data['Converted_Dates'].dt.month_name()

train_data['activation_months'].value_counts()

train_data['Live Contracts'].value_counts()

train_data['Hotline Count (Last 6 months)'].value_counts()

train_data['Total payment'].value_counts()

px.box(data_frame=train_data,x='Total payment')

# calculate IQR for column Height
Q1 = train_data['Total payment'].quantile(0.25)
Q3 = train_data['Total payment'].quantile(0.75)
IQR = Q3 - Q1

# identify outliers
threshold = 1.5
Total_payment_outliers = train_data[(train_data['Total payment'] < Q1 - threshold * IQR) | (train_data['Total payment'] > Q3 + threshold * IQR)]

len(Total_payment_outliers)

train_data['Total payment'].max()

payment_lessthan_zero=(train_data[train_data['Total payment'] <0])

train_data.drop(payment_lessthan_zero.index,inplace=True)
train_data.reset_index(drop=True,inplace=True)

train_data['Total payment'].describe()

#### statistical summary
- 25% of the total payment values are less than or equal to 602.81.
- The median total payment value is 1,125.57, which means that 50% of the values are less than or equal to this amount.
- 75% of the total payment values are less than or equal to 1,903.82.
- The largest total payment value is 213,048.21.

train_data[train_data['Total payment']== 213048.21]

train_data = train_data.drop(index=70509)


train_data.reset_index(drop=True,inplace=True)

train_data['Total payment'].max()

train_data['Count of payment'].value_counts()

train_data['Sus counts'].value_counts(normalize=True)*100

- shown that 72%of customer not taken sus action and 21% of customer sus for one time and 5% for sus 2 time .

train_data['Hotline action'].value_counts(normalize=True)*100

- shown that 55% of customers not het hotline action and 35% taken hotline for 1 time and 7% for 2 times.

train_data['age'].value_counts()

train_data['age'] = pd.to_numeric(train_data['age'], errors='coerce')

train_data['age'].value_counts()

train_data['age'].min()

train_data['age'].max()

train_data['age'].isna().sum()

train_data['age']= train_data['age'].replace(" ",np.nan)

train_data["Age_Group"] = pd.cut(
    train_data.age,
    bins=[16,20,24,28,32,36,40,44,48,52,56,60,64,94],
    labels=["16-20","20-24","24-28","28-32","32-36","36-40","40-44","44-48","48-52","52-56","56-60","60-64","64-94"])
train_data["Age_Group"].value_counts()

px.histogram(data_frame=train_data,x="Age_Group")

train_data.describe().T

## Univariate analysis for cat columns

### Check tagert column

train_data['Target'].value_counts(normalize=True).plot.pie(autopct ='%1.2f%%')

- shown that 43% of target is Outstanding.
- 21% Suspended and also 21% Fluctuating.
- 14% of cutsomer is migration out.
- So we see that 35% of customer become not user for line.

train_data['Payment Mode'].value_counts()

train_data['Type Stroe'].value_counts()

px.histogram(data_frame=train_data,x='Type Stroe')

train_data['Store Name'].unique().tolist()

train_data['Store Name'] = train_data['Store Name'].str.strip()

column_values = train_data['Store Name']

# Step 2: Split the values using comma as the delimiter
split_values = column_values.str.split(',')

# Step 3: Flatten the list of values
flattened_values = [value for sublist in split_values for value in sublist]

# Step 4: Get unique values
unique_values = pd.unique(flattened_values)

sorted(unique_values)

# Function to split the first two words
def split_first_two_words(store_name):
    words = store_name.split()
    return ' '.join(words[:3])

train_data['Stroe_region'] = train_data['Store Name'].apply(split_first_two_words)

sorted(train_data['Stroe_region'].unique().tolist())

import re
def clean_store_region(text):
    # Remove line breaks and extra whitespace
    cleaned_text = text.strip().replace('\n', '')
    # Remove non-alphanumeric characters except spaces
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned_text)
    return cleaned_text

# Clean the 'Store Region' column
train_data['Stroe_region_cleaned'] = train_data['Stroe_region'].apply(clean_store_region)

train_data['Stroe_region_cleaned'] = train_data['Stroe_region_cleaned'].str.strip()

sorted(train_data['Stroe_region_cleaned'].unique().tolist())

from fuzzywuzzy import process

unique_store_names = train_data['Stroe_region_cleaned'].unique()
len(unique_store_names)

def find_similar_words(word_list, threshold=80):
    corrections = {}
    for word in word_list:
        matches = process.extract(word, word_list, limit=len(word_list))
        for match, score in matches:
            if score >= threshold and match != word:
                corrections[match] = word
    return corrections
# Identify similar words
corrections = find_similar_words(unique_store_names)
print("Suggested corrections:", corrections)
# Manually review and update corrections dictionary
# Review the printed output and manually confirm the correct spellings
corrections = {
    'Ramdan': 'Ramadan',
    'Elshekh': 'El Sheikh',
    'ElWakad': 'El Wakad',
    # Add more corrections based on the output from fuzzy matching
}
# Function to apply corrections
def correct_spelling(store_name, corrections):
    words = store_name.split()
    corrected_words = [corrections.get(word, word) for word in words]
    return ' '.join(corrected_words)
#Apply corrections to the column
train_data['Standardized_region_Name'] = train_data['Stroe_region_cleaned'].apply(correct_spelling, corrections=corrections)

# Display the results to validate
print(train_data[['Stroe_region_cleaned', 'Standardized_region_Name']])

data['Activation source'].value_counts(normalize=True)

px.histogram(data_frame=train_data,x='Activation source')

- Most of new activation customer done from CVM team by 59%.

train_data['Adsl'].value_counts()

fig = plt.figure(figsize= (5,5))
sns.countplot(data=train_data,x='Adsl',palette='rocket')

train_data['Payment Channel'].value_counts()

# Fill missing values with a specific label like 'Unknown'
train_data['Payment Channel'] = train_data['Payment Channel'].fillna('')

# Add a binary feature to indicate missing values
train_data['Payment_Channel_Missing'] = train_data['Payment Channel'].isnull().astype(int)

# Standardize the 'Payment Channel' values
train_data['Payment_Channel_list'] = train_data['Payment Channel'].apply(lambda x: x.replace(", ", ",").strip())
train_data['Standardized Payment Channel'] = train_data['Payment_Channel_list'].apply(lambda x: ','.join(set(x.split(','))))

Standardized_Payment_top10=train_data['Standardized Payment Channel'].value_counts().head(10).reset_index()

Standardized_Payment_top10

px.histogram(data_frame=Standardized_Payment_top10 , x ='Standardized Payment Channel' , y ="count")

train_data['Standardized Payment Channel'].value_counts(normalize=True)

 Shown that 17% new activation customer paid his bills in retailand about 15% Vf cash.

# Get unique values in the column
unique_values = train_data['Standardized Payment Channel'].unique()

# Initialize an empty mapping dictionary
channel_mapping = {}

# Iterate over unique values and create mapping
for value in unique_values:
    # Split the value by comma and strip spaces
    components = [x.strip() for x in value.split(',')]
    # Sort the components alphabetically to standardize the order
    components.sort()
    # Join the components to create a standardized value
    standardized_value = ','.join(components)
    # Map the original value to the standardized value
    channel_mapping[value] = standardized_value

# Apply mapping to the column
train_data['Standardized Payment Channel'] = train_data['Standardized Payment Channel'].map(channel_mapping)




train_data['Standardized Payment Channel'].value_counts()

from collections import Counter

# Assuming 'train_data' is your pandas DataFrame
column_values = train_data['Standardized Payment Channel']

# Step 2: Split the values using comma as the delimiter
split_values = column_values.str.split(',')

# Step 3: Flatten the list of values
flattened_values = [value for sublist in split_values for value in sublist]

# If you prefer pandas way:
category_counts = column_values.str.split(',').explode().value_counts()

category_counts

plt.figure(figsize=(8, 4))
category_counts.plot(kind='bar')
plt.title('Category Counts')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Assuming 'train_data' is your DataFrame
threshold = 500  # Threshold for merging infrequent categories
counts = train_data['Standardized Payment Channel'].value_counts()
infrequent_categories = counts[counts < threshold].index

len(infrequent_categories)

# Assuming 'train_data' is your DataFrame
threshold = 500  # Threshold for merging infrequent categories
counts = train_data['Standardized Payment Channel'].value_counts()
infrequent_categories = counts[counts < threshold].index

# Replace infrequent categories with 'Other'
train_data['Standardized Payment Channel'] = train_data['Standardized Payment Channel'].apply(lambda x: 'Less_than_500' if x in infrequent_categories else x)


train_data['Standardized Payment Channel']=train_data['Standardized Payment Channel'].replace('fawry','Aggregators')

Payment_channel_top10=train_data['Standardized Payment Channel'].value_counts().head(20).reset_index()
Payment_channel_top10

len(train_data['Standardized Payment Channel'].value_counts())

import plotly.graph_objs as go
# Group the data by 'Target'
grouped_data = train_data.groupby('Target')

# Create an empty figure
fig = go.Figure()

# Plot the top 10 categories for each 'Hotline_Class'
for Target, group in grouped_data:
    # Count the occurrences of each category within the group
    category_counts = group['Standardized Payment Channel'].value_counts().nlargest(10)
    
    # Add a bar trace for the top 10 categories for the current 'Hotline_Class'
    fig.add_trace(go.Bar(x=category_counts.index, y=category_counts.values, name=f'target {Target}'))

# Update layout
fig.update_layout(
    title='Top 10 Categories by target',
    xaxis_title='Categories',
    yaxis_title='Frequency',
    barmode='group',  # 'group' for grouped bar charts, 'stack' for stacked bar charts
    xaxis=dict(tickangle=45),
    width=900,
    height=500
)

# Show the plot
fig.show()



train_data['Tariff Model_y'].value_counts().head(10)

## to remove ", " to ","
train_data['Tariff Model_y'] = train_data['Tariff Model_y'].apply(lambda x: x.replace(", ", ",").strip())
# Create a new column with the list of services
train_data['Tariff_Model_list']= train_data['Tariff Model_y'].apply(lambda x: x.split(','))

## check rate plan enclude more than 1000 accounts
Tariff_Model_lessthan1000 = train_data['Tariff_Model_list'].value_counts()
Tariff_less_than_1000= Tariff_Model_lessthan1000[Tariff_Model_lessthan1000 >1000]
Tariff_less_than_1000

## to check number of accounts in every rateplan
from collections import Counter

# Flatten the list of lists into a single list
all_services = [service for sublist in train_data['Tariff_Model_list'] for service in sublist]

# Count the occurrences of each service
service_counts = Counter(all_services)

# Convert to a DataFrame for easier manipulation
service_counts_df = pd.DataFrame(service_counts.items(), columns=['Service', 'Count']).sort_values(by='Count', ascending=False)

service_counts_df.head(10)

top_services = service_counts_df.head(10)

plt.figure(figsize=(15, 5))
sns.barplot(data=top_services, x='Service', y='Count', palette='viridis')
plt.xticks(rotation=40)
plt.yticks(fontsize=10)
plt.title('Top 10 Most Common Services')
plt.show()

## to check most frequent rateplan combinations
from itertools import combinations

# Define a function to get combinations of services and ensure elements are strings
def get_combinations(service_list):
    return [",".join(sorted(map(str, comb))) for comb in combinations(service_list,2)]

# Apply the function to the 'Service Tariff List' column
train_data['Service Combinations'] = train_data['Tariff_Model_list'].apply(lambda x: get_combinations(x))

# Flatten the list of lists into a single list
all_combinations = [comb for sublist in train_data['Service Combinations'] for comb in sublist]

# Count the occurrences of each combination
combination_counts = Counter(all_combinations)

# Convert to a DataFrame for easier manipulation
combination_counts_df = pd.DataFrame(combination_counts.items(), columns=['Combination', 'Count']).sort_values(by='Count', ascending=False)


combination_counts_df.head(10)

# Count the occurrences of each value in 'Tariff_Model_list'
value_counts = train_data['Tariff_Model_list'].value_counts()

# Identify values with counts less than 1000
other_values = value_counts[value_counts < 1000].index

# Filter the DataFrame to identify rows where 'Tariff_Model_list' is in 'other_values'
rows_to_replace = train_data['Tariff_Model_list'].isin(other_values)

# Replace 'Tariff_Model_list' values in these rows with a list containing only 'Other'
train_data.loc[rows_to_replace, 'Tariff_Model_list'] = [['Other']]


# Convert 'Other' to a list
train_data['Tariff_Model_list'] = train_data['Tariff_Model_list'].apply(lambda x: [x] if not isinstance(x, list) else x)
len(train_data['Tariff_Model_list'].value_counts())

train_data['Gender'].value_counts()

train_data['Gender'].isna().sum()

train_data['Gender']= train_data['Gender'].replace("",np.nan)

px.histogram(data_frame=train_data,x='Gender')

train_data['Ativation month '].value_counts()

train_data.rename(columns={'Ativation month ':'Activation month'},inplace=True)

px.histogram(data_frame=train_data,x='Activation month')

train_data['High usage'].value_counts()

train_data['High usage Status'].value_counts()

train_data['High usage Action'].value_counts()

## Bivariate analysis

##correlation between 'target' and the 'Bill Cycle'?
px.histogram(data_frame=train_data,x='Bill Cycle',color='Target',barmode='group',text_auto=True,width=600,height=500)

# Calculate counts of 'Target' for each 'Bill Cycle'
hotline_counts = train_data.groupby(['Bill Cycle', 'Target']).size().unstack(fill_value=0)

# Calculate total counts of each 'Activation source' category
total_counts = train_data['Bill Cycle'].value_counts()

# Calculate percentage of each Hotline_Class within each Activation source category
hotline_percentages = (hotline_counts.T / total_counts).T * 100

# Reset index to make ADSL and Hotline_Class columns
hotline_percentages =round(hotline_percentages.reset_index(),1)

# Melt DataFrame to use in plotly express
melted_df = pd.melt(hotline_percentages, id_vars=['Bill Cycle'], var_name='Target', value_name='Percentage')

# Create grouped histogram
fig = px.bar(melted_df, x='Bill Cycle', y='Percentage', color='Target', barmode='group',
             text='Percentage', title="Percentage of Target for each Bill Cycle",
             labels={'Percentage': 'Percentage of Target'},
             width=800, height=400)

# Show the plot
fig.show()


- Chart shown that:
- BC1 have highest percent for fluctuating customers by 24% and 23% of suspended customers.
- BC2 have highest percent of migration out customer by 15%.
- BC4 equel percent for suspended customer by 20% and equel percent with BC1 in migration out.

##correlation between 'Hotline Class' and the 'Final Tariff'?
px.histogram(data_frame=train_data,x='Target',color='Live Contracts',barmode='group',text_auto=True,width=800,height=500)

contingency_table = pd.crosstab(data['Live Contracts'], train_data['Target'])
contingency_table

- Categories 3 through 10 have significantly fewer contracts compared to categories 0, 1, and 2.
- The majority of contracts fall within the 0, 1, and 2 categories of "Live Contracts", especially in category 1.

# Calculate counts of Hotline_Class for each ADSL category
hotline_counts = train_data.groupby(['Adsl', 'Target']).size().unstack(fill_value=0)

# Calculate total counts of each ADSL category
total_counts = train_data['Adsl'].value_counts()

# Calculate percentage of each Hotline_Class within each ADSL category
hotline_percentages = (hotline_counts.T / total_counts).T * 100

# Reset index to make ADSL and Hotline_Class columns
hotline_percentages =round(hotline_percentages.reset_index(),1)

# Melt DataFrame to use in plotly express
melted_df = pd.melt(hotline_percentages, id_vars=['Adsl'], var_name='Target', value_name='Percentage')

# Create grouped histogram
fig = px.bar(melted_df, x='Adsl', y='Percentage', color='Target', barmode='group',
             text='Percentage', title="Percentage of Target for each ADSL category",
             labels={'Percentage': 'Percentage of Target'},
             width=800, height=400)

# Show the plot
fig.show()


##correlation between Target and the Type Stroe?
px.histogram(data_frame=train_data,x='Activation source',color='Target',barmode='group',text_auto=True,width=800,height=500)

# Calculate counts of Target for each Activation source
hotline_counts = train_data.groupby(['Activation source', 'Target']).size().unstack(fill_value=0)

# Calculate total counts of each 'Activation source' category
total_counts = train_data['Activation source'].value_counts()

# Calculate percentage of each Hotline_Class within each Activation source category
hotline_percentages = (hotline_counts.T / total_counts).T * 100

# Reset index to make ADSL and Hotline_Class columns
hotline_percentages =round(hotline_percentages.reset_index(),1)

# Melt DataFrame to use in plotly express
melted_df = pd.melt(hotline_percentages, id_vars=['Activation source'], var_name='Target', value_name='Percentage')

# Create grouped histogram
fig = px.bar(melted_df, x='Activation source', y='Percentage', color='Target', barmode='group',
             text='Percentage', title="Percentage of Target for each Activation source",
             labels={'Percentage': 'Percentage of Target'},
             width=800, height=400)

# Show the plot
fig.show()


- CVM included highest percent of suspended by 25% and 17% of migration out.
- New activation included highest percent of fluctuating by 27%.
- Organic have best result for outstanding by 60%.

##correlation between Target and the Type Stroe?
px.histogram(data_frame=train_data,x='Activation month',color='Target',barmode='group',text_auto=True,width=800,height=500)

mean_payment_by_target = train_data.groupby('Target')['Total payment'].mean()
mean_payment_by_target

train_data.groupby('Target')['Total payment'].mean().plot.bar(color=['red', 'blue', 'cyan','green'])

##correlation between Target and the Type Stroe?
px.histogram(data_frame=train_data,x='Count of payment',color='Target',barmode='group',text_auto=True,width=800,height=500)

# Calculate counts of Target for 'High usage'
hotline_counts = train_data.groupby(['High usage', 'Target']).size().unstack(fill_value=0)

# Calculate total counts of each 'Activation source' category
total_counts = train_data['High usage'].value_counts()

# Calculate percentage of each Hotline_Class within each Activation source category
hotline_percentages = (hotline_counts.T / total_counts).T * 100

# Reset index to make ADSL and Hotline_Class columns
hotline_percentages =round(hotline_percentages.reset_index(),1)

# Melt DataFrame to use in plotly express
melted_df = pd.melt(hotline_percentages, id_vars=['High usage'], var_name='Target', value_name='Percentage')

# Create grouped histogram
fig = px.bar(melted_df, x='High usage', y='Percentage', color='Target', barmode='group',
             text='Percentage', title="Percentage of Target for High usage",
             labels={'Percentage': 'Percentage of Target'},
             width=800, height=400)

# Show the plot
fig.show()


- chart shown that 52% of high usage customer take suspension action and 9% make migration out 17% fluctuating.

# Filter out the rows where 'High usage Status' is "Not high usage"
filtered_data = train_data[train_data['High usage Status'] != "Not high usage"]

# Create the 'high usage with ST' column with the remaining values from 'High usage Status'
filtered_data['high usage with ST'] = filtered_data['High usage Status']

# Now create the histogram
import plotly.express as px

fig = px.histogram(data_frame=filtered_data, x='high usage with ST', color='Target', barmode='group', text_auto=True, width=800, height=500)
fig.show()



- Most of too risky customer in high usage get suspension.

train_data['Hotline action'].value_counts()

##correlation between Target and the Type Stroe?
px.histogram(data_frame=train_data,x='Hotline action',color='Target',barmode='group',text_auto=True,width=800,height=500)

# Calculate counts of Target for ''Hotline action''
hotline_counts = train_data.groupby(['Hotline action', 'Target']).size().unstack(fill_value=0)

# Calculate total counts of each 'Activation source' category
total_counts = train_data['Hotline action'].value_counts()

# Calculate percentage of each Hotline_Class within each Activation source category
hotline_percentages = (hotline_counts.T / total_counts).T * 100

# Reset index to make ADSL and Hotline_Class columns
hotline_percentages =round(hotline_percentages.reset_index(),1)

# Melt DataFrame to use in plotly express
melted_df = pd.melt(hotline_percentages, id_vars=['Hotline action'], var_name='Target', value_name='Percentage')

# Create grouped histogram
fig = px.bar(melted_df, x='Hotline action', y='Percentage', color='Target', barmode='group',
             text='Percentage', title="Percentage of Target for Hotline action",
             labels={'Percentage': 'Percentage of Target'},
             width=800, height=400)

# Show the plot
fig.show()


 - Chart shown that percent of migration out increased with 1st action of hotline as customer shocked with action.

# Calculate the delay in payment as a percentage
train_data['delay_in_payment_percent'] = (train_data['Count of payment'] / train_data['variance_days'])*100

# Print the first few rows to verify
print(train_data[['variance_days', 'Count of payment', 'delay_in_payment_percent']].head())

# Save the updated dataframe to an Excel file
train_data.to_excel('updated_dataset_with_delay_percent.xlsx', index=False)


# Create a bar plot
plt.figure(figsize=(8, 5))
sns.barplot(x='Target', y='delay_in_payment_percent', data=train_data)
plt.title('Delay in Payment Percent vs Target ')
plt.xlabel('Target')
plt.ylabel('Delay in Payment Percent')
plt.xticks(rotation=45)
plt.show()


# Create a box plot to visualize the distribution
plt.figure(figsize=(8, 5))
sns.boxplot(x='Target', y='delay_in_payment_percent', data=train_data)
plt.title('Delay in Payment Percent Distribution Across Target Column')
plt.xlabel('Target')
plt.ylabel('Delay in Payment Percent')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


train_data['delay_in_payment_percent'].describe()

- Mean of the payments made by customers are 70.04% of the total variance days.
- 25% of the records have a delay_in_payment_percent of 50% or less.
- 75% of the records have a delay_in_payment_percent of 100% or less.

##correlation between Target and the Type Stroe?
px.histogram(data_frame=train_data,x='Age_Group',color='Target',barmode='group',text_auto=True,width=1000,height=500)

# Calculate counts of Target for 'Age_Group'
hotline_counts = train_data.groupby(['Age_Group', 'Target']).size().unstack(fill_value=0)

# Calculate total counts of each 'Activation source' category
total_counts = train_data['Age_Group'].value_counts()

# Calculate percentage of each Hotline_Class within each Activation source category
hotline_percentages = (hotline_counts.T / total_counts).T * 100

# Reset index to make ADSL and Hotline_Class columns
hotline_percentages =round(hotline_percentages.reset_index(),1)

# Melt DataFrame to use in plotly express
melted_df = pd.melt(hotline_percentages, id_vars=['Age_Group'], var_name='Target', value_name='Percentage')

# Create grouped histogram
fig = px.bar(melted_df, x='Age_Group', y='Percentage', color='Target', barmode='group',
             text='Percentage', title="Percentage of Target for Age_Group",
             labels={'Percentage': 'Percentage of Target'},
             width=800, height=400)

# Show the plot
fig.show()


- chart shown that:
- Age category 16-20 years have a spik on suspension by 45% and migration out 14%.
- Age category 20-24 years have also spik on suspension 31% and migration out  15%.
- Best age for outstanding from 32 till 44 by AVG 52%.
- worest age for suspension from 16-20 by 45%.
- Worest age for migration out 64- 94 by 20%.

mean_payment_by_Age_Group = train_data.groupby('Age_Group')['delay_in_payment_percent'].mean()
mean_payment_by_Age_Group

train_data.groupby('Age_Group')['delay_in_payment_percent'].mean().plot.bar()

##correlation between Target and the Type Stroe?
px.histogram(data_frame=train_data,x='Age_Group',color='Activation source',barmode='group',text_auto=True,width=1000,height=500)

##correlation between Target and the Type Stroe?
px.histogram(data_frame=train_data,x='Gender',color='Target',barmode='group',text_auto=True,width=1000,height=500).update_xaxes(categoryorder="total descending")

# Calculate counts of Target for Gender
hotline_counts = train_data.groupby(['Gender', 'Target']).size().unstack(fill_value=0)

# Calculate total counts of each 'Activation source' category
total_counts = train_data['Gender'].value_counts()

# Calculate percentage of each Hotline_Class within each Activation source category
hotline_percentages = (hotline_counts.T / total_counts).T * 100

# Reset index to make ADSL and Hotline_Class columns
hotline_percentages =round(hotline_percentages.reset_index(),1)

# Melt DataFrame to use in plotly express
melted_df = pd.melt(hotline_percentages, id_vars=['Gender'], var_name='Target', value_name='Percentage')

# Create grouped histogram
fig = px.bar(melted_df, x='Gender', y='Percentage', color='Target', barmode='group',
             text='Percentage', title="Percentage of Target for Gender",
             labels={'Percentage': 'Percentage of Target'},
             width=800, height=400)

# Show the plot
fig.show()


- Chart shown that gender of customer not have big effect on our target.

# Transform the list values into separate rows
expanded_train_data = train_data.explode('Tariff_Model_list')

# Plot the histogram
fig = px.histogram(data_frame=expanded_train_data, x='Tariff_Model_list', color='Target', barmode='group', text_auto=True, width=900, height=400)
fig.show()

# Convert list elements to strings and join them
train_data['Tariff_Model_str'] = train_data['Tariff_Model_list'].apply(lambda x: ','.join(map(str, x)))

# Calculate counts of Target for Gender
hotline_counts = train_data.groupby(['Tariff_Model_str', 'Target']).size().unstack(fill_value=0)

# Calculate total counts of each 'Activation source' category
total_counts = train_data['Tariff_Model_str'].value_counts()

# Calculate percentage of each Hotline_Class within each Activation source category
hotline_percentages = (hotline_counts.T / total_counts).T * 100

# Reset index to make ADSL and Hotline_Class columns
hotline_percentages =round(hotline_percentages.reset_index(),1)

# Melt DataFrame to use in plotly express
melted_df = pd.melt(hotline_percentages, id_vars=['Tariff_Model_str'], var_name='Target', value_name='Percentage')

# Create grouped histogram
fig = px.bar(melted_df, x='Tariff_Model_str', y='Percentage', color='Target', barmode='group',
             text='Percentage', title="Percentage of Target for Tariff_Model_str",
             labels={'Percentage': 'Percentage of Target'},
             width=800, height=400)

# Show the plot
fig.show()


- Chart shown 28% of Red advance take suspension action with migration 10%.
- customer active 2 lines with same rateplna red assential have a spik in migration out with 78%.
- Red prime shown a spik in suspension action by 26%.
- Red essential with Adsl have a spik in outstanding accounts.

## Preparing data for model.

### Removing  unnecessary data

train_data.drop(['Customer Segment','Payment Mode','Number Of Unbilled Bills','Activation Date','Store Name','Type Stroe','Payment Channel','Tariff Model_y','High usage Status','High usage Action','Converted_Dates','activation_months','Age_Group','Stroe_region', 'Stroe_region_cleaned','Payment_Channel_Missing','Payment_Channel_list','Service Combinations'],axis=1,inplace=True)

train_data.isna().sum()

from sklearn.impute import KNNImputer

imputer = KNNImputer()

train_data['age'] = imputer.fit_transform(train_data[['age']])

# Get the number of missing values
num_missing = train_data['Gender'].isnull().sum()
num_missing

# Calculate the number of 'M' and 'F' values to fill
num_m = int(num_missing * 81 / 100)
num_f = int(num_missing * 19 / 100)
print(num_m)
print(num_f)

# Ensure the sum of num_m and num_f does not exceed the number of missing values
while num_m + num_f < num_missing:
    num_f += 1
# Create a list with the values to fill
fill_values = ['M'] * num_m + ['F'] * num_f
len(fill_values)

# Shuffle the list to randomize the filling
np.random.shuffle(fill_values)

# Fill the missing values
train_data.loc[train_data['Gender'].isnull(), 'Gender'] = fill_values

Num_Columns=train_data.select_dtypes(include=np.number).columns
Num_Columns

from dython.nominal import associations

selected_column= train_data[['Live Contracts', 'Hotline Count (Last 6 months)', 'Total payment',
       'Count of payment', 'Sus counts', 'Hotline action', 'age',
       'variance_days', 'delay_in_payment_percent','Target']]
categorical_df = selected_column.copy()
categorical_correlation= associations(categorical_df, filename= 'categorical_correlation.png', figsize=(8,8))

train_data.drop(['Standardized_region_Name','Tariff_Model_list'],axis=1,inplace=True)

data_update=train_data.copy()

data_update

train_data.drop(['Customer Code','Live Contracts'],axis=1,inplace=True)

train_data

## Model

### Encoder

Cat_columns=train_data.select_dtypes("object").columns
Cat_columns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Custom transformer to convert sparse matrix to dense array
class SparseToDenseTransformer:
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.toarray()

# Define column transformer using Pipeline
Encoder = ColumnTransformer(
    transformers=[
        ("OHE", Pipeline([
            ("OneHot", OneHotEncoder(drop="first")),
            ("SparseToDense", SparseToDenseTransformer())
        ]), ['Bill Cycle', 'Activation source', 'Adsl', 'Activation month',
       'High usage','Gender']),
        ("BE", BinaryEncoder(), ['Standardized Payment Channel',
       'Tariff_Model_str'])
    ],
    remainder="passthrough"
)


train_data['Target'].unique()

# replacing target variable with 0,1,2 and 3 respectively
train_data['Target'].replace('Outstandine', 0, inplace=True)
train_data['Target'].replace('Fluctuating', 1, inplace=True)
train_data['Target'].replace('Suspended', 2, inplace=True)
train_data['Target'].replace('Migration Out', 3, inplace=True)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_validate

# Assuming Encoder is defined somewhere
# If not, replace Encoder with the actual encoder you're using, e.g., OneHotEncoder

steps = []
steps.append(("Encoder", Encoder))
steps.append(("Scaler", RobustScaler()))
steps.append(("Model", LogisticRegression(max_iter=1000)))  # Increase max_iter

pipeline = Pipeline(steps=steps)

x = train_data.drop('Target', axis=1)
y = train_data['Target']

results = cross_validate(pipeline, x, y, cv=5, scoring="accuracy", return_train_score=True)

print(results)


results["train_score"].mean()

results["test_score"].mean()

from imblearn.pipeline import Pipeline as ImPipeline
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

models = list()
models.append(("XG" , XGBClassifier()))

for model in models:
    steps = []
steps.append(("Encoder" , Encoder))
steps.append(("Scaler" , RobustScaler()))
steps.append(("SmoteTomek" , SMOTETomek(smote=SMOTE(sampling_strategy={0:32324} , random_state=24))))
steps.append(("XG" , XGBClassifier()))
pipeline = ImPipeline(steps = steps)
scores = cross_validate(pipeline , x , y , scoring="accuracy"  ,cv = 5 , return_train_score=True,return_estimator=True)
print("Train_accuracy" , scores["train_score"].mean())
print("-" * 10)
print("Test_accuracy" , scores["test_score"].mean())

best_estimator = scores['estimator'][0]
y_preds = best_estimator.predict(x)

y_preds

from sklearn.metrics import classification_report
for model in models:
    steps = []
    steps.append(("Encoder" , Encoder))
    steps.append(("Scaler" , RobustScaler()))
    steps.append(("SmoteTomek" , SMOTETomek(smote=SMOTE(sampling_strategy={0:32324} , random_state=24))))
    steps.append(("XG" , XGBClassifier()))
    pipeline = ImPipeline(steps = steps)
    scores = cross_validate(pipeline , x , y , scoring="accuracy"  ,cv = 5 , return_train_score=True,return_estimator=True)
 # Fit the pipeline on the entire dataset
    pipeline.fit(x, y)

    # Make predictions on the test set
    y_pred = pipeline.predict(x)

    print(model[0])
    print("Train Accuracy:", scores["train_score"].mean())
    print("Test Accuracy:", scores["test_score"].mean())
    print("-" * 20)
    print("Classification Report:\n", classification_report(y, y_pred))
    print("-" * 20)
    print("\n")

y_labeled = pd.DataFrame({'prediction':list(y_preds)})
y_labeled

output = pd.concat([x, y_labeled], axis=1)
output

output['Customer Code'] =data_update['Customer Code']  # Replace 'target' with your actual column name


output['Target']=data_update['Target']

output.to_excel("output.xlsx",index=False)

val_data.to_excel(r"C:\Users\etaha2\Downloads\New Model\last data\validation_data.xlsx",index=False)

## new Dataset For Validation

validation_dataset=pd.read_excel(r"C:\Users\etaha2\Downloads\New Model\last data\validation_data_set.xlsx",sheet_name="Sheet1")

best_estimator

# replacing target variable with 0,1,2 and 3 respectively
validation_dataset['Target'].replace('Outstandine', 0, inplace=True)
validation_dataset['Target'].replace('Fluctuating', 1, inplace=True)
validation_dataset['Target'].replace('Suspended', 2, inplace=True)
validation_dataset['Target'].replace('Migration Out', 3, inplace=True)

x_val=validation_dataset.drop('Target', axis=1)
y_val=validation_dataset['Target']

x_val['Bill Cycle']=x_val['Bill Cycle'].astype(str)

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImPipeline
from category_encoders import BinaryEncoder

# Custom transformer to convert sparse matrix to dense array
class SparseToDenseTransformer:
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.toarray()

# Define column transformer using Pipeline
encoder = ColumnTransformer(
    transformers=[
        ("OHE", Pipeline([
            ("OneHot", OneHotEncoder(drop="first")),
            ("SparseToDense", SparseToDenseTransformer())
        ]), ['Bill Cycle', 'Activation source', 'Adsl', 'Activation month', 'High usage', 'Gender']),
        ("BE", BinaryEncoder(), ['Standardized Payment Channel', 'Tariff_Model_str'])
    ],
    remainder="passthrough"
)

# Define the pipeline steps
steps = [
    ("Encoder", encoder),
    ("Scaler", RobustScaler()),
    ("SmoteTomek", SMOTETomek(smote=SMOTE(sampling_strategy={0: 32324}, random_state=24))),
    ("XG" , XGBClassifier())
]

# Create the pipeline
pipeline = ImPipeline(steps=steps)

# Fit the pipeline on the training data
pipeline.fit(x, y)

# Perform cross-validation
scores = cross_validate(pipeline, x, y, scoring="accuracy", cv=5, return_train_score=True, return_estimator=True)

# Print the training results
print("Train_accuracy:", scores["train_score"].mean())
print("-" * 10)

# Print the testing results on the validation set
print("Validation_accuracy:", scores["test_score"].mean())
print("-" * 20)

# Transform the unseen data using the fitted pipeline
# Since we cannot directly access the steps inside the fitted pipeline for transforming new data, 
# we need to create a function to apply the same transformations as the pipeline without fitting again.
def transform_new_data(pipeline, x_val):
    # Transform with encoder
    x_val_encoded = pipeline.named_steps['Encoder'].transform(x_val)
    
    # Ensure the encoded data is numeric
    if not np.issubdtype(x_val_encoded.dtype, np.number):
        x_val_encoded = pd.DataFrame(x_val_encoded).apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    # Scale the encoded data
    x_val_scaled = pipeline.named_steps['Scaler'].transform(x_val_encoded)
    
    return x_val_scaled

# Apply the transformations to the validation data
x_val_transformed = transform_new_data(pipeline, x_val)

# Make predictions on the unseen data
predictions_new_data = pipeline.named_steps["XG"].predict(x_val_transformed)

print("Predictions on New Data:", predictions_new_data)
print("-" * 20)


# Debugging prints
print("Shape of x_val:", x_val.shape)
print("Shape of x_val_transformed:", x_val_transformed.shape)

result_df = pd.DataFrame(np.hstack((x_val, predictions_new_data.reshape(-1,
1))),
 columns=np.append(x_val.columns, 'Predictions'))


result_df

data_update=pd.read_excel(r"C:\Users\etaha2\Downloads\New Model\last data\validation_update_data.xlsx",sheet_name="Sheet1",dtype={"Customer Code":str})

# Add the "customer_id" column from the original data to the result DataFrame
result_df["Customer Code"] = data_update["Customer Code"]


result_df.to_excel(r"C:\Users\etaha2\Downloads\New Model\last data\data_val_pred.xlsx",index=False)

models = list()
models.append(("LR" , LogisticRegression(max_iter=1000)))
models.append(("SVM" , SVC()))
models.append(("CART" , DecisionTreeClassifier()))
models.append(("RF" , RandomForestClassifier()))
models.append(("XG" , XGBClassifier()))
models.append(("KNN" , KNeighborsClassifier()))

for model in models:
    steps = []
    steps.append(("Encoder" , Encoder))
    steps.append(("Scaler" , RobustScaler()))
    steps.append(("SmoteTomek" , SMOTETomek(smote=SMOTE(sampling_strategy={0:32323} , random_state=24))))
    steps.append(model)
    pipeline = ImPipeline(steps = steps)
    scores = cross_validate(pipeline , x , y , scoring="accuracy"  ,cv = 5 , return_train_score=True,return_estimator=True)
    print(model[0])
    print("Train_accuracy" , scores["train_score"].mean())
    print("-" * 10)
    print("Test_accuracy" , scores["test_score"].mean())
    print("\n")