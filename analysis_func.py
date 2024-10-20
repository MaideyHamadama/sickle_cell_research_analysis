# Import libraries
import numpy as np
import pandas as pd
import scipy.stats as stats
import os
import statistics
import seaborn as sns
import statsmodels.api as sm
from math import *
from matplotlib import pyplot as plt
from scipy.stats import iqr,kstest,shapiro,chi2_contingency
from numpy.random import seed
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from lifelines import CoxPHFitter

# Charging analysis file
def load_data():
    """ Charge the data source for analysis"""
    data_file = "data.xlsx"
    df = pd.read_excel(data_file)
    return df

# Load cleaned data
def load_cleaned_data():
    """ Load cleaned data """
    file = "data_cleaned.xlsx"
    df_cleaned = pd.read_excel(file)
    return df_cleaned

def calc_vif(X):
    # Calculating VIF
    vif = pd.DataFrame()
    vif['variables'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return (vif)

# Scatter plot diagram
def scatter_plot(df, ind_var, dep_var, title, var3 = ""):
    """ Used to visualize relations between two continuous variables
    - Points on the graph show how a variable change in relation to another variable
    - Coloring the points in function of a third variable can help in the visualization of sub groups
    - Generaly x axis is the independent variable and y axis the dependent variable"""
    plt.figure(figsize=(10, 6))
    if var3 != "":
        plt.scatter(df[ind_var], df[dep_var], c=df[var3], cmap='viridis', label=ind_var + ' vs ' + dep_var)
    else:
        plt.scatter(df[ind_var], df[dep_var], label=ind_var + ' vs ' + dep_var)
    plt.xlabel(ind_var)
    plt.ylabel(dep_var)
    plt.title(title)
    if var3 != "":
        plt.legend([var3])
        plt.colorbar(label = var3)
    plt.show()
    
# Histogram diagram
def histogram(df, var, title, color='blue'):
    """ Used to visualize the distribution of one continuous variable 
    - Show observations frequency in different classes
    - Useful to identify the form of distribution (normal, asymetric, etc...) and outliners"""
    plt.figure()
    plt.hist(df[var], bins=10, alpha=0.7, color=color, label=var + " Distribution")
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.show()
    
# Box plots
def box_plot(df, main_var, secondary_var):
    """ Used to visualize distribution of one continuous variable and identify outliers in the dataset
    - It is used to summarize the distribution of a continuous variable
    - Display the mean, quartiles and outliers
    - Compare the box plots between groups (for e.g Vaccinated and non vaccinated) helps identify the differences in distributions"""
    plt.figure(figsize=(10,6))
    df.boxplot(column=main_var, by=secondary_var, grid=False)
    plt.xlabel(secondary_var)
    plt.ylabel(main_var)
    plt.title("Box plot : " + main_var + " by " + secondary_var)
    plt.suptitle('')
    plt.show()
    
# Pair plot
def seaborn(df, key_var):
    """ Used to visualize relations by pairs in an entire dataframe """
    sns.pairplot(df, hue=key_var)
    plt.show()
    
# Heatmap plot
def heatmap(df):
    """ Used to visualize correlations between continuous variables"""
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Heatmap of correlations')
    plt.show()

# Density plot
def density_plot(df, var):
    """ Used to estimate the probability distribution of a continuous variable"""
    plt.figure(figsize=(10,6))
    sns.kdeplot(df[var], fill=True)
    plt.xlabel(var)
    plt.title('Density Plot: ' + var)
    plt.show()

# Regression plot
def regression_plot(df, ind_var, dep_var):
    """ Diplay a regression line on a scatter graph. It helps in visualizing the linear trend between variables """
    plt.figure(figsize=(10,6))
    sns.regplot(x=ind_var, y=dep_var, data=df)
    plt.xlabel(ind_var)
    plt.ylabel(dep_var)
    plt.title('Regression Plot : ' + ind_var + ' vs ' + dep_var + '')
    plt.show()
    
# Categorical Scatter plot
def cat_scatter_plot(df, cat_var, cont_var, var3):
    """ Used to visualize how a continuous variable is related to a categorical variable.
    Sometimes with colored categories or different forms"""
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=cont_var, y=cat_var, hue=var3, data=df)
    plt.xlabel(cont_var)
    plt.ylabel(cat_var)
    plt.title('Scatter Plot with ' + var3)
    plt.show()

# Logistic Regression
def logistic_regression(df, dep_var, ind_var_list):
    """ Used to modelise the relation between binary dependent variable (death status) and one/multiple
    independent variables (ou covariables) """
    X = df[ind_var_list]
    y = df[dep_var]
    
    # Division of data
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=42)
    # Test size of 0.3 indicates 30% of data will be used for testing and 70% for training
    
    # Model of logistic regression
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    
    # Predictions
    y_pred = logreg.predict(X_test)
    
    # Evaluation of the model
    # Confusin matrix
    """ Useful tool to evaluate the performance of a classification model
                  Predicted
              0      1
Actual 0     TN     FP
       1     FN     TP
    True Negative (TN): The model correctly predicted the negative class.
    False Positive (FP): The model incorrectly predicted the positive class.
    False Negative (FN): The model incorrectly predicted the negative class.
    True Positive (TP): The model correctly predicted the positive class
    """
    print("Confusion matrix")
    print(confusion_matrix(y_test, y_pred))
    
    # Classification report
    print("Classification report")
    print(classification_report(y_test, y_pred))
        
    # Fit the logistic regression model with statsmodels
    """ To prevent the singular matrix error in the sm.Logit(y_train, X_train_sm). You must check for multi-collinearity using vif function.
        Other checks are perfect seperation, duplicate rows and constant columns"""
    # Calculate for perfect seperation
    """ Inspect the cross-tabulation of the dependent variable against the independent variables to see if there is perfect separation.
        Perfect seperation occurs when the dependent variable can be perfectly predicted by the independent variables.
        Technically, it occurs when one or more cells in the crosstab have a frequency of zero, indicating that the outcome can be perfectly predicted by the independent variable."""
    # for column in X_train.columns:
    #     crosstab = pd.crosstab(X_train[column], y_train)
    #     print(f"Crosstab for {column}: ")
    #     print(crosstab)
    #     print()
    """ If perfect seperation is found, logistic regression may fail to converge
        You might consider to combine categories or consider alternative methods."""
    
    # Check for duplicated rows
    print(f"Number of duplicate rows in X_train: {X_train.duplicated().sum()} \n")
    if X_train.duplicated().sum() > 0:
        X_train = X_train.drop_duplicates()
        y_train = y_train[X_train.index]
    
    # Check for constant variables
    for column in X_train.columns:
        unique_values = X_train[column].nunique()
        if unique_values == 1:
            print(f"Column {column} has only one unique value.")

    # Add a constant for statsmodels
    X_train_sm = sm.add_constant(X_train)

    # Fit the logistic regression model with statsmodels
    logit_model = sm.Logit(y_train, X_train_sm)
    result = logit_model.fit()
    
    # Summary of the model
    print("Summary of the model")
    print(result.summary())
    
    # Odds ratio
    odds_ratio = np.exp(result.params)
    print(odds_ratio)
    

# COX Regression
def cox_regression(df, duration_col, event_col):
    """ Used to analyse survival data and identify covariables that influences time till the event realisation"""
    # Create the model
    cph = CoxPHFitter()
    
    # Adjust the model to data
    cph.fit(df, duration_col, event_col)
    
    # Display the summary of the model
    cph.print_summary()