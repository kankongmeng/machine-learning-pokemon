# Let's use pokemon dataset for interesting machine learning training.
# Our objectives for this training is to complete three task.
# Task 1: Select one fire, water, and grass pokemon for your fight, please choose the best.
# Task 2: Regression, understand the factor influencing a pokemon attack.
# Task 3: Classification, find out the pokemon is legendary or not.

# Pandas is for data analysis and Seaborn for visualizations
import pandas as pd

# Load data from csv
df = pd.read_csv('Pokemon.csv')

# Show the first five rows
df.head()

# As you see the data is not clean, have duplicate id and NaN value
# Let's clean up the data first
# Convert all colums to lowercase
df.columns = df.columns.str.lower()

# Rename columns name for readability
df.rename(columns={'#': 'id'}, inplace=True)
df.rename(columns={'type 1': 'primary_type'}, inplace=True)
df.rename(columns={'type 2': 'secondary_type'}, inplace=True)
df.rename(columns={'sp. atk': 'special_atk'}, inplace=True)
df.rename(columns={'sp. def': 'special_def'}, inplace=True)

# Remove any Pokemon with duplicated id numbers except for the first one
df.drop_duplicates('id', keep='first', inplace=True)

# Replace any missing values in type 2 with 'None'
df['secondary_type'].fillna(value='None', inplace=True)

# Show the first five rows again
df.head()

# Data is clean now, let's start the data analysis
# Understand the data structure
df.info()

# Show each primary_type consist how many pokemon
df.groupby(['primary_type']).size()

# Show each primary_type consist how many pokemon using crosstab
pd.crosstab(index=df["primary_type"], columns="count")

#-------------------------------------------------------------------------------------------------------------------------------------------#

## Task 1: Select one fire, water, and grass pokemon for your fight, please choose the best.

# Filter pokemon by grass, water and fire
grass_pokemon = df[df['primary_type'] == 'Grass']
water_pokemon = df[df['primary_type'] == 'Water']
fire_pokemon = df[df['primary_type'] == 'Fire']

# Filter secondary type
grass_poison_pokemon = grass_pokemon[grass_pokemon['secondary_type'] == 'Poison']
water_psychic_pokemon = water_pokemon[water_pokemon['secondary_type'] == 'Psychic']
fire_fighting_pokemon = fire_pokemon[fire_pokemon['secondary_type'] == 'Fighting']

# Describe are use to generates descriptive statistics that summarize the central tendency
grass_poison_pokemon.describe()
# Choose the grass posion pokemon with speed 90 based on above describe()
my_grass_pokemon = grass_poison_pokemon[grass_poison_pokemon['speed'] == 90]

# Repeat the same step to for selecting water pokemon
water_psychic_pokemon.describe()
# Choose the water psyhic pokemon with defense 110 based on above describe()
my_water_pokemon = water_psychic_pokemon[water_psychic_pokemon['defense'] == 110]

# Repeat the same step to for selecting fire pokemon
fire_fighting_pokemon.describe()
# Choose the fire fighting pokemon with defense 110 based on above describe()
my_fire_pokemon = fire_fighting_pokemon[fire_fighting_pokemon['attack'] == 123]

# My three best pokemon
pd.concat([my_grass_pokemon, my_water_pokemon, my_fire_pokemon])
print("---------------------------------------------------------------------------------------------------------\n")

#-------------------------------------------------------------------------------------------------------------------------------------------#

## Task 2: Regression, understand the factor influencing a pokemon attack.

# Import train_test_split for spliting the data
from sklearn.cross_validation import train_test_split
# Use to modelling the relationship between dependent and indepent variable, prediction, forecasting, or error reduction.
from sklearn.linear_model import LinearRegression
# Use to measures the average of the squares of the errors
from sklearn.metrics import mean_squared_error
# Numpy mathematical computation on arrays and matrices
import numpy as np
# Matplotlib is a Python 2D plotting library which produces publication quality figures
import matplotlib.pyplot as plot

# First model, determine how does defense influencing attack
###############################################################################################

# Dependent variable
target = df['attack']
# Independent variable
features = df[['defense']]
# In new version of sklearn, it using dataFrame from int64 type, therefore even single feature has to be 2d-matrix.
# Ex: Y=pd.dataFrame(Y)

# Split arrays or matrices into random train and test subsets
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=8)
# test_size=0.3, meaning spliting 70% into training sets, 30% to testing sets.
# random_state=8, number is not important, just for reproducible result.
# Should remove random_state in production if you really need a random split.

# Create linear regression object
lm = LinearRegression()

# Train the model using the training sets
lm.fit(x_train, y_train)

# Make predictions using the testing sets
prediction = lm.predict(x_test)

# LinearRegression score returns the coefficient of determination R^2 of the prediction.
print("First model score: {}".format(lm.score(x_test, y_test)))
# The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares ((y_true - y_pred) ** 2).sum()
# V is the total sum of squares ((y_true - y_true.mean()) ** 2).sum(), The best possible score is 1.0

# Expected value of the squared error loss
mse = np.sqrt(mean_squared_error(y_test, prediction))
print("Mean squared error: {}".format(mse))
# Formula of mean squared error
# dft = pd.DataFrame(y_test)
# dft['prediction'] = prediction
# dft['error'] = dft['attack'] - dft['prediction']
# dft
# np.sqrt(np.mean(dft['error'].values**2))
print("---------------------------------------------------------------------------------------------------------\n")

# Plot a scatter plot to compare true attack and the predicted attack.
plot.scatter(y_test, prediction)
plot.xlabel("Attack: $Y_i$")
plot.ylabel("Predicted attack: $\hat{Y}_i$")
plot.title("Attack vs Predicted attack: $Y_i$ vs $\hat{Y}_i$")

# Below are for Second model, determine how does defense, speed and hp influencing attack
###############################################################################################

target = df['attack']
features = df[['defense', 'speed', 'hp']]

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=8)

lm = LinearRegression()
lm.fit(x_train, y_train)
prediction = lm.predict(x_test)

print("Second model score: {}".format(lm.score(x_test, y_test)))

mse = np.sqrt(mean_squared_error(y_test, prediction))
print("Mean squared error: {}".format(mse))
print("---------------------------------------------------------------------------------------------------------\n")

plot.scatter(y_test, prediction)
plot.xlabel("Attack: $Y_i$")
plot.ylabel("Predicted attack: $\hat{Y}_i$")
plot.title("Attack vs Predicted attack: $Y_i$ vs $\hat{Y}_i$")

#-------------------------------------------------------------------------------------------------------------------------------------------#

## Task 3: Classification, find out the pokemon is legendary or not

#  Observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves)
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import preprocessing
# Graph visualization for drawing graphs specified in DOT language scripts
import graphviz

# First model, determine legendary respect to all the columns in datasets
###############################################################################################
target = df['legendary']
features = df.drop(['legendary'], axis=1)

# Decision tree can't handle string data, so we have to use LabelEncoder convert it to number
le = preprocessing.LabelEncoder()
for column_name in features.columns:
        if features[column_name].dtype == object:
            # We can always use inverse_transform to revert the changes
            features[column_name] = le.fit_transform(features[column_name])
        else:
            pass

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=1)
clf = DecisionTreeClassifier().fit(x_train, y_train)

print('Decision tree first model score: {:.2f}'
     .format(clf.score(x_test, y_test)))

dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=x_train.columns,
                         class_names=target.unique().astype(str),
                         filled=True, rounded=True,  
                         special_characters=True)  

graphviz.Source(dot_data)

# Below are for Second model, determine legendary respect to defense, speed, and hp 
###############################################################################################
target = df['legendary']
features = df[['attack', 'defense', 'speed']]

le = preprocessing.LabelEncoder()
for column_name in features.columns:
        if features[column_name].dtype == object:
            features[column_name] = le.fit_transform(features[column_name])
        else:
            pass

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=1)
clf = DecisionTreeClassifier().fit(x_train, y_train)

print('Decision tree second model score: {:.2f}'
     .format(clf.score(x_test, y_test)))

dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=x_train.columns,
                         class_names=target.unique().astype(str),
                         filled=True, rounded=True,  
                         special_characters=True)  

graphviz.Source(dot_data)

# Below are for third model, determine how does defense, speed and hp influencing primary type
###############################################################################################
target = df['primary_type']
features = df[['attack', 'defense', 'speed']]

le = preprocessing.LabelEncoder()
for column_name in features.columns:
        if features[column_name].dtype == object:
            features[column_name] = le.fit_transform(features[column_name])
        else:
            pass

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=1)
clf = DecisionTreeClassifier().fit(x_train, y_train)

print('Decision tree second model score: {:.2f}'
     .format(clf.score(x_test, y_test)))

dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=x_train.columns,
                         class_names=target.unique(),
                         filled=True, rounded=True,  
                         special_characters=True)  

graphviz.Source(dot_data)
# View in pdf file
graphviz.Source(dot_data).view()