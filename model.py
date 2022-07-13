#import modules
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas  # for dataframes
import matplotlib.pyplot as plt  # for plotting graphs
import seaborn as sns  # for plotting graphs

data = pandas.read_csv('emp_data.csv')

# Import train_test_split function

# Import LabelEncoder
# creating labelEncoder
le = preprocessing.LabelEncoder()

# Spliting data into Feature and
X = data[['satisfaction_level', 'last_evaluation', 'number_project',
         'average_montly_hours', 'time_spend_company', 'Work_accident',
          'promotion_last_5years', 'Departments_int', 'salary_int']]
y = data['left']
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)  # 70% training and 30% test

# Import Gradient Boosting Classifier model

# Create Gradient Boosting Classifier
gb = GradientBoostingClassifier()

# Train the model using the training sets
gb.fit(X_train, y_train)

y_pred = gb.predict(X_test)

joblib.dump(gb, "gb.pkl")
