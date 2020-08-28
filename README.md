# MLFindingDonorsProject
Udacity Project using machine learning to find donors for our fictional cause

<h3>Getting Started </h3>
<p>In this project, I will employ several supervised algorithms of my choice to accurately model individuals' income using data collected from the 1994 U.S. Census. Then, I will choose the best candidate algorithm from preliminary results and further optimize this algorithm to best model the data. My goal is to construct a model that accurately predicts whether an individual makes more than $50,000. Understanding an individual's income can help a non-profit better understand how large of a donation to request, or whether or not they should reach out to begin with. While it can be difficult to determine an individual's general income bracket directly from public sources, we can (as we will see) infer this value from other publically available features.<br>

The dataset for this project originates from the UCI Machine Learning Repository. The datset was donated by Ron Kohavi and Barry Becker, after being published in the article "Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid". You can find the article by Ron Kohavi online. The data we investigate here consists of small changes to the original dataset, such as removing the 'fnlwgt' feature and records with missing or ill-formatted entries.</p>

<h3>Exploring the Data</h3>
<p>Running the code cell below will load necessary Python libraries and load the census data. Note that the last column from this dataset, 'income', will be our target label (whether an individual makes more than, or at most, $50,000 annually). All other columns are features about each individual in the census database.</p>

In [4]: 
```python3
# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
import visuals as vs

# Pretty display for notebooks
%matplotlib inline

# Load the Census dataset
data = pd.read_csv("census.csv")

# Success - Display the first record
display(data.head(n=10))
```

<p align = "center"><kbd><img src = "/images/table1.png"></kbd></p>

<h3>Implementation: Data Exploration</h3>
<p>A cursory investigation of the dataset will determine how many individuals fit into either group, and will tell us about the percentage of these individuals making more than \$50,000. In the code cell below, I will compute the following:</p>
<ul>
  <li>The total number of records, 'n_records'</li>
  <li>The number of individuals making more than \$50,000 annually, 'n_greater_50k'.</li>
  <li>The number of individuals making at most \$50,000 annually, 'n_at_most_50k'.</li>
  <li>The percentage of individuals making more than \$50,000 annually, 'greater_percent'.</li>
</ul>
<p>In [5]:</p>

```python3
# TODO: Total number of records
n_records = data.shape[0]

# TODO: Number of records where individual's income is more than $50,000
n_greater_50k = len(data[(data['income'] == '>50K')])

# TODO: Number of records where individual's income is at most $50,000
n_at_most_50k = len(data[(data['income'] == '<=50K')])

# TODO: Percentage of individuals whose income is more than $50,000
greater_percent = n_greater_50k/n_records

# Print the results
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))
```
<em>Total number of records: 45222<br>
Individuals making more than $50,000: 11208<br>
Individuals making at most $50,000: 34014<br>
Percentage of individuals making more than $50,000: 0.2478439697492371% </em>
<h4>Featureset Exploration</h4>
<ul>
  <li><strong>age</strong>: continuous.</li>
  <li><strong>workclass</strong>: : Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.</li>
  <li><strong>education</strong>: : Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.</li>
  <li><strong>education-num</strong>: continuous.</li>
  <li><strong>marital-status</strong>: : Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.</li>
  <li><strong>occupation</strong>: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.</li>
   <li><strong>relationship</strong>: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.</li>
  <li><strong>race</strong>: Black, White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other.</li>
  <li><strong>sex</strong>: Female, Male.</li>
  <li><strong>capital-gain</strong>: continuous.</li>
  <li><strong>capital-loss</strong>: continuous.</li>
  <li><strong>hours-per-week</strong>: continuous.</li>
  <li><strong>native-country</strong>: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.</li>
</ul>
<h3>Preparing the Data</h3>
<p>Before data can be used as input for machine learning algorithms, we need to clean, format, and restructure the data. Fortunately, for this dataset, there are no invalid or missing entries we must deal with, however, there are some qualities about certain features that must be adjusted. This preprocessing can help tremendously with the outcome and predictive power of nearly all learning algorithms.</p>

<h4>Transforming Skewed Continuous Features</h4>
<p>A dataset may sometimes contain at least one feature whose values tend to lie near a single number, but will also have a non-trivial number of vastly larger or smaller values than that single number. Algorithms can be sensitive to such distributions of values and can underperform if the range is not properly normalized. With the census dataset two features fit this description: <em>'capital-gain'</em> and <em>'capital-loss'</em>.<br>

Running the code cell below will plot a histogram of these two features. Note the range of the values present and how they are distributed.</p>

In [6]:

```python3
# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Visualize skewed continuous features of original data
vs.distribution(data)
```
<p align = "center"><kbd><img src = "/images/graph1.png"></kbd></p>


<p>For highly-skewed feature distributions such as 'capital-gain' and 'capital-loss', it is common practice to apply a logarithmic transformation on the data so that the very large and very small values do not negatively affect the performance of a learning algorithm. Using a logarithmic transformation significantly reduces the range of values caused by outliers. Care must be taken when applying this transformation however: The logarithm of 0 is undefined, so we must translate the values by a small amount above 0 to apply the the logarithm successfully.<br>

Running the code cell below will perform a transformation on the data and visualize the results. Again, note the range of values and how they are distributed.</p>

In [7]:
```python3
# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
vs.distribution(features_log_transformed, transformed = True)
```
<p align = "center"><kbd><img src = "/images/graph2.png" ></kbd></p>

<h4>Normalizing Numerical Features</h4>
<p>In addition to performing transformations on features that are highly skewed, it is often good practice to perform some type of scaling on numerical features. Applying a scaling to the data does not change the shape of each feature's distribution (such as 'capital-gain' or 'capital-loss' above); however, normalization ensures that each feature is treated equally when applying supervised learners. Note that once scaling is applied, observing the data in its raw form will no longer have the same original meaning, as exampled below.<br>

Running the code cell below will normalize each numerical feature. We will use sklearn.preprocessing.MinMaxScaler for this.</p>

<p>In [8]:</p>

```python3
# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Show an example of a record with scaling applied
display(features_log_minmax_transform.head(n = 5))
```
<p align = "center"><kbd><img src = "/images/table2.png"></kbd></p>
<h3>Implementation: Data Preprocessing</h3>
<p>From the table in Exploring the Data above, we can see there are several features for each record that are non-numeric. Typically, learning algorithms expect input to be numeric, which requires that non-numeric features (called categorical variables) be converted. One popular way to convert categorical variables is by using the one-hot encoding scheme.<br>

Additionally, as with the non-numeric features, we need to convert the non-numeric target label, 'income' to numerical values for the learning algorithm to work. Since there are only two possible categories for this label ("<=50K" and ">50K"), we can avoid using one-hot encoding and simply encode these two categories as 0 and 1, respectively. In code cell below, I will:</p>
<ul>
  <li>Use pandas.get_dummies() to perform one-hot encoding on the 'features_log_minmax_transform' data.</li>
  <li>Convert the target label 'income_raw' to numerical entries.</li>
  <ul>
    <li>Set records with "<=50K" to 0 and records with ">50K" to 1.</li>
  </ul>
</ul>



<p>In [9]:</p>

```python3
# TODO: One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
features_final = pd.get_dummies(features_log_minmax_transform)
display(features_final.head(n=10))
print(features_final.shape[0])
# TODO: Encode the 'income_raw' data to numerical values
income = income_raw.apply(lambda x: 1 if x == '>50K' else 0)

# Print the number of features after one-hot encoding
encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

# Uncomment the following line to see the encoded feature names
print (encoded)
```
<p align = "center"><kbd><img src = "/images/graph3.gif" ></kbd></p>
<em>10 rows × 103 columns<br>

45222<br>
103 total features after one-hot encoding.<br>
['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'workclass_ Federal-gov', 'workclass_ Local-gov', 'workclass_ Private', 'workclass_ Self-emp-inc', 'workclass_ Self-emp-not-inc', 'workclass_ State-gov', 'workclass_ Without-pay', 'education_level_ 10th', 'education_level_ 11th', 'education_level_ 12th', 'education_level_ 1st-4th', 'education_level_ 5th-6th', 'education_level_ 7th-8th', 'education_level_ 9th', 'education_level_ Assoc-acdm', 'education_level_ Assoc-voc', 'education_level_ Bachelors', 'education_level_ Doctorate', 'education_level_ HS-grad', 'education_level_ Masters', 'education_level_ Preschool', 'education_level_ Prof-school', 'education_level_ Some-college', 'marital-status_ Divorced', 'marital-status_ Married-AF-spouse', 'marital-status_ Married-civ-spouse', 'marital-status_ Married-spouse-absent', 'marital-status_ Never-married', 'marital-status_ Separated', 'marital-status_ Widowed', 'occupation_ Adm-clerical', 'occupation_ Armed-Forces', 'occupation_ Craft-repair', 'occupation_ Exec-managerial', 'occupation_ Farming-fishing', 'occupation_ Handlers-cleaners', 'occupation_ Machine-op-inspct', 'occupation_ Other-service', 'occupation_ Priv-house-serv', 'occupation_ Prof-specialty', 'occupation_ Protective-serv', 'occupation_ Sales', 'occupation_ Tech-support', 'occupation_ Transport-moving', 'relationship_ Husband', 'relationship_ Not-in-family', 'relationship_ Other-relative', 'relationship_ Own-child', 'relationship_ Unmarried', 'relationship_ Wife', 'race_ Amer-Indian-Eskimo', 'race_ Asian-Pac-Islander', 'race_ Black', 'race_ Other', 'race_ White', 'sex_ Female', 'sex_ Male', 'native-country_ Cambodia', 'native-country_ Canada', 'native-country_ China', 'native-country_ Columbia', 'native-country_ Cuba', 'native-country_ Dominican-Republic', 'native-country_ Ecuador', 'native-country_ El-Salvador', 'native-country_ England', 'native-country_ France', 'native-country_ Germany', 'native-country_ Greece', 'native-country_ Guatemala', 'native-country_ Haiti', 'native-country_ Holand-Netherlands', 'native-country_ Honduras', 'native-country_ Hong', 'native-country_ Hungary', 'native-country_ India', 'native-country_ Iran', 'native-country_ Ireland', 'native-country_ Italy', 'native-country_ Jamaica', 'native-country_ Japan', 'native-country_ Laos', 'native-country_ Mexico', 'native-country_ Nicaragua', 'native-country_ Outlying-US(Guam-USVI-etc)', 'native-country_ Peru', 'native-country_ Philippines', 'native-country_ Poland', 'native-country_ Portugal', 'native-country_ Puerto-Rico', 'native-country_ Scotland', 'native-country_ South', 'native-country_ Taiwan', 'native-country_ Thailand', 'native-country_ Trinadad&Tobago', 'native-country_ United-States', 'native-country_ Vietnam', 'native-country_ Yugoslavia']</em>
<h4>Shuffle and Split Data</h4>
<p>Now all categorical variables have been converted into numerical features, and all numerical features have been normalized. As always, we will now split the data (both features and their labels) into training and test sets. 80% of the data will be used for training and 20% for testing.<br>

Running the code cell below will perform this split.</p>

<p>In [10]:</p>

```python3
# Import train_test_split
from sklearn.model_selection import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    income, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))
```
<em>Training set has 36177 samples.<br>
Testing set has 9045 samples.</em>

<h3>Evaluating Model Performance</h3>
<p>In this section, we will investigate four different algorithms, and determine which is best at modeling the data. Three of these algorithms will be supervised learners of my choice, and the fourth algorithm is known as a naive predictor.</p>

<h4>Metrics and the Naive Predictor</h4>
<p>CharityML, equipped with their research, knows individuals that make more than \$50,000 are most likely to donate to their charity. Because of this, *CharityML* is particularly interested in predicting who makes more than \$50,000 accurately. It would seem that using accuracy as a metric for evaluating a particular model's performace would be appropriate. Additionally, identifying someone that does not make more than $50,000 as someone who does would be detrimental to *CharityML*, since they are looking to find individuals willing to donate. Therefore, a model's ability to precisely predict those that make more than $50,000 is more important than the model's ability to recall those individuals. We can use F-beta score as a metric that considers both precision and recall:</p>
<p align = "center"><kbd><img src = "/images/formula1.png"></kbd></p>
  
<p>In particular, when beta = 0.5, more emphasis is placed on precision, so that is the beta value we will choose. <br>

Looking at the distribution of classes (those who make at most \$50,000$ and those who make more), it's clear most individuals do not make more than \$50,000. This can greatly affect **accuracy**, since we could simply say *"this person does not make more than \$50,000" and generally be right, without ever looking at the data! Making such a statement would be called naive, since we have not considered any information to substantiate the claim. It is always important to consider the naive prediction for your data, to help establish a benchmark for whether a model is performing well. That been said, using that prediction would be pointless: If we predicted all people made less than \$50,000, CharityML would identify no one as donors.</p>

<h4>Question 1 - Naive Predictor Performace</h4>
<ul>
  <li>If we chose a model that always predicted an individual made more than $50,000, what would that model's accuracy and F-score be on this dataset?</li>
</ul>
<p>In [11]:</p>

```python3
'''
TP = np.sum(income) # Counting the ones as this is the naive case. Note that 'income' is the 'income_raw' data 
encoded to numerical values done in the data preprocessing step.
FP = income.count() - TP # Specific to the naive case

TN = 0 # No predicted negatives in the naive case
FN = 0 # No predicted negatives in the naive case
'''
# TODO: Calculate accuracy, precision and recall
accuracy = np.sum(income)/income.count()
recall = 1
precision = np.sum(income)/ (np.sum(income) + income.count())

# TODO: Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
fscore = (1.25*precision *recall)/((0.25 * precision)+recall)

# Print the results 
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))
```
<em>Naive Predictor: [Accuracy score: 0.2478, F-score: 0.2365]</em>
<h4>Supervised Learning Models</h4>
<strong>The following are some of the supervised learning models that are currently available in scikit-learn that you may choose from:</strong>
<ul>
  <li>Gaussian Naive Bayes (GaussianNB)</li>
  <li>Decision Trees</li>
  <li>Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)</li>
  <li>K-Nearest Neighbors (KNeighbors)</li>
  <li>Stochastic Gradient Descent Classifier (SGDC)</li>
  <li>Support Vector Machines (SVM)</li>
  <li>Logistic Regression</li>
</ul>
<h4>Question 2 - Model Application</h4>

<p>List three of the supervised learning models above that are appropriate for this problem that you will test on the census data. For each model chosen</p>
<ul>
  <li>Describe one real-world application in industry where the model can be applied.</li>
  <li>What are the strengths of the model; when does it perform well?</li>
  <li>What are the weaknesses of the model; when does it perform poorly?</li>
  <li>What makes this model a good candidate for the problem, given what you know about the data?</li>
</ul>
<strong>Answer:</strong><br>
<ol>
  <li><strong>Ensemble Methods (Bagging, Adaboost, Random Forest, Gradient Boosting)</strong></li>
  <ul>
    <li>Real-World Application
Ensemble methods can be used for one vs. all recognition cases such as anomaly or intrusion detection, meaning they are incredibly effective in preventing bank fraud or detecting hacking attempts.</li>
    <li>
Model Strengths
Ensemble method models are scalable and due to their naturally hierarchical structure, can easily model non-linear decision boundaries.</li>
    <li>Model Weaknesses
If our ensemble of weak learners consists of a large number of models/classifiers, it could cause the run-time to be inefficient. In addition, individual models can tend to overfit, though this problem is alleviated by "ensembling" the models.</li>
    <li>Why Model Applies to Problem
Ensemble method models are a fit for our problem due to their strength in modeling nonlinear decision boundaries. Given the number of variables we are considering, our results will certainly be based on nonlinear decision analysis.</li>
  </ul>
</ol>
<a href = "http://www.ehu.eus/ccwintco/uploads/1/1f/Eider_ClassifierEnsembles_SelectRealWorldApplications.pdf">Reference 1</a><br>
<a href = "https://elitedatascience.com/machine-learning-algorithms">Reference 2</a>

<strong>Answer:</strong><br>
<ol>
  <li><strong>Support Vector Machines (SVM)</strong></li>
  <ul>
    <li>Real-World Application
SVM can be used for facial detection. Using pixel classification, SVM can classify each pixel as a "face" pixel or a "non-face" pixel. By doing so, it narrows down the area where the faces are located in the image. It then uses its training data of pixel brightness and surrounding pixels to detect faces.</li>
    <li>
Model Strengths
SVM works well in situations of high dimensions and is quite good and avoiding overfitting tendencies. SVM works well in situations of high dimensions due to the kernel trick of elevating data points into a higher dimension, finding the "best-fit" separator in that dimension, and converting that separator back into the terms of the previous dimension. SVM also possesses a certain versatility given that kernels can be adjusted to provide more possibilities to divide data.</li>
    <li>Model Weaknesses
SVM typically tend to not work well when the dataset is overwhelmingly large. In these situations, SVM will take up a lot of time trying to classify the dataset.</li>
    <li>Why Model Applies to Problem
Given the number of features in our dataset (103), the ability of SVM to work in high dimensions will work in our advantage. While being a rookie to machine learning means ~45000 datapoints seem like a large amount to me, for our purposes, the number of datapoints isn't high enough to be wary of SVM capabilities.</li>
  </ul>
</ol>
<a href = "https://data-flair.training/blogs/applications-of-svm/">Reference 1</a><br>
<a href = "https://elitedatascience.com/machine-learning-algorithms">Reference 2</a>

<strong>Answer:</strong><br>
<ol>
  <li><strong>Decision Trees</strong></li>
  <ul>
    <li>Real-World Application
Decision Trees have been used by US immigration services to identify potential terrorist threats. As seen in the reference 1 link, each characteristic of the individual represents a node in the decision tree and satisfying a number of conditions produces a certain probability based on past data.</li>
    <li>
Model Strengths
Decision trees usually do not require normalization or scaling of data (although we have done this already for the usage of the other two algorithms</li>
    <li>Model Weaknesses
Decision trees typically require a higher time complexity and are usually poor at predicting continuous values.</li>
    <li>Why Model Applies to Problem
Decision trees are applicable to this particular problem due to their ability at predicting discrete data/results, which is what we need as we simply care whether an individual makes more than \$50,000, rather than their actual income.</li>
  </ul>
</ol>
<a href = "http://csci.viu.ca/~barskym/teaching/DM2012/lectures/Lecture2.DTApplications.pdf">Reference 1</a><br>
<a href = "https://medium.com/@dhiraj8899/top-5-advantages-and-disadvantages-of-decision-tree-algorithm-428ebd199d9a">Reference 2</a>

<h4>Implementation - Creating a Training and Predicting Pipeline</h4>
<p>To properly evaluate the performance of each model you've chosen, it's important that you create a training and predicting pipeline that allows you to quickly and effectively train models using various sizes of training data and perform predictions on the testing data. Your implementation here will be used in the following section. In the code block below, you will need to implement the following:</p>

<ul>
  <li>Import fbeta_score and accuracy_score from sklearn.metrics.</li>
  <li>Fit the learner to the sampled training data and record the training time.</li>
  <li>Perform predictions on the test data X_test, and also on the first 300 training points X_train[:300].</li>
  <ul><li>Record the total prediction time.</li></ul>
  <li>Calculate the accuracy score for both the training subset and testing set.</li>
  <li>Calculate the F-score for both the training subset and testing set.</li>
  <ul><li>Make sure that you set the beta parameter!</li></ul>
</ul>

<p>In [12]:</p>

```python3
# TODO: Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import fbeta_score, accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # TODO: Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    
    # TODO: Calculate the training time
    results['train_time'] = end-start
        
    # TODO: Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    
    # TODO: Calculate the total prediction time
    results['pred_time'] = end-start
            
    # TODO: Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
        
    # TODO: Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # TODO: Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, 0.5)
        
    # TODO: Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test, 0.5)
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results
```

<h4>Implementation: Initial Model Evaluation</h4>
<p>In the code cell, you will need to implement the following:</p>

<ul>
  <li>Import the three supervised learning models you've discussed in the previous section.</li>
  <li>Initialize the three models and store them in 'clf_A', 'clf_B', and 'clf_C'.</li>
  <ul><li>Use a 'random_state' for each model you use, if provided.</li><li>Note: Use the default settings for each model — you will tune one specific model in a later section.</li></ul>
  <li>Calculate the number of records equal to 1%, 10%, and 100% of the training data.</li>
  <ul><li>Store those values in 'samples_1', 'samples_10', and 'samples_100' respectively.</li></ul>
</ul>
<p>Note: Depending on which algorithms you chose, the following implementation may take some time to run!</p>

<p>In [13]:</p>

```python3
# TODO: Import the three supervised learning models from sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

# TODO: Initialize the three models
clf_A = AdaBoostClassifier(random_state = 42) #possible hyperparameters base_estimator = DecisionTreeClassifier(max_depth=2), n_estimators = 4
clf_B = SVC(random_state = 42)#possible hyperparameters(kernel='poly', degree=4, C=0.1)
clf_C = DecisionTreeClassifier(random_state = 42) #possible hyperparameters max_depth = 7, min_samples_leaf = 10

# TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
# HINT: samples_100 is the entire training set i.e. len(y_train)
# HINT: samples_10 is 10% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
# HINT: samples_1 is 1% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
samples_100 = len(y_train)
samples_10 = int(len(y_train) * 0.1)
samples_1 = int(len(y_train) * 0.01)

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fscore)
```

<em>AdaBoostClassifier trained on 361 samples.<br>
AdaBoostClassifier trained on 3617 samples. <br>
AdaBoostClassifier trained on 36177 samples. <br>
SVC trained on 361 samples. <br>
SVC trained on 3617 samples. <br>
SVC trained on 36177 samples. <br>
DecisionTreeClassifier trained on 361 samples. <br>
DecisionTreeClassifier trained on 3617 samples.<br>
DecisionTreeClassifier trained on 36177 samples.</em>
<p align = "center"><kbd><img src = "/images/graph4.png"></kbd></p>

<h3>Improving Results</h3>
<p>In this final section, I will choose from the three supervised learning models the best model to use on the student data. I will then perform a grid search optimization for the model over the entire training set (X_train and y_train) by tuning at least one parameter to improve upon the untuned model's F-score.</p>

<h4>Question 3 - Choosing the Best Model</h4>
<p><strong>Answer</strong>:<br>
Dear CharityML,
The Adaboost model is the best model out of the three options I provided. In the training set, the decision tree model has the better f-score by a sizable margin throughout the progression in the training set. However, due to not setting the hyperparameter of max depth on the decision tree, we know decision trees have a tendency to overfit, so it is no surprise its f-score on the training set is much higher. However, the true measure of the best model comes with the testing set. As you can see from the bar chart above, with each progressively larger percentage of the training set used, adaboost is the model with the highest f-score. We also see the prediction and training time of the adaboost model is significantly less than the svm model. Given its advantages over the other two models in addition to its strength in predictions nonlinear results, adaboost proves to be the best choice to help us to locate donors for our surely honorable cause.<br>
Best,<br>
Ben</p>


<h4>Question 4 - Describing the Model in Layman's Terms</h4>
<p>In one to two paragraphs, explain to CharityML, in layman's terms, how the final model chosen is supposed to work. Be sure that you are describing the major qualities of the model, such as how the model is trained and how the model makes a prediction. Avoid using advanced mathematical jargon, such as describing equations.</p>
<p><strong>Answer</strong>:<br>
P.S. The adaboost model is a type of ensemble method. The two chief competing variables in finding a good machine learning model are the model's bias and its variance. Ensemble methods use other models to create imperfect models labeled as "weak learners." A number of these weak learners are created and combined to become an ensemble of weak models, or a single "strong learner." By combining different models, ensemble methods hope to mediate one model's tendency for high bias and low variance (linear regression for example) with another's tendency for low bias and high variance (decision trees).<br>
Adaboost itself uses a weighting system to assign weights to each datapoint. Instead of creating other models, it combines iterations of weighting on one model. In simpler terms, every data point in the adaboost model starts with an equal weight, but misclassified points are assigned greater weights. Weights are similar to a grading system, if the model misclassifies a point of higher weight, it will have higher error, so the higher weight discourages the model from continuously misclassifying the same point, which will improve the model with each iteration. In addition, in our case, data points with income >50K will have positive weight and those with lower income will have negative weight. With each "line" or division created from each iteration, regions classified by >50K and <=50K will be created. To determine which is which, adaboost simply sums up the weight of each region. If the sum is positive, data points landing in that particular region are likely to have income of >50K and vice versa.</p>

<h4>Implementation: Model Tuning</h4>
<p>Here, I will fine tune the chosen model, using grid search (GridSearchCV) with at least one important parameter tuned with at least 3 different values. I will use the entire training set for this. In the code cell below, you will need to implement the following:</p>

<ul>
  <li>Import sklearn.grid_search.GridSearchCV and sklearn.metrics.make_scorer.</li>
  <li>Initialize the classifier you've chosen and store it in clf.</li>
  <ul><li>Set a random_state if one is available to the same state you set before.</li></ul>
  <li>Create a dictionary of parameters you wish to tune for the chosen model.</li>
  <ul><li>Example: parameters = {'parameter' : [list of values]}.</li><li>Note: Avoid tuning the max_features parameter of your learner if that parameter is available!</li></ul>
  <li>Use make_scorer to create an fbeta_score scoring object (with beta = 0.5).</li>
  <li>Perform grid search on the classifier clf using the 'scorer', and store it in grid_obj.</li>
  <li>Fit the grid search object to the training data (X_train, y_train), and store it in grid_fit.</li>
</ul>
<p>Note: Depending on the algorithm chosen and the parameter list, the following implementation may take some time to run!</p>

<p>In [17]:</p>

```python3
# TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
# TODO: Initialize the classifier
clf = AdaBoostClassifier(random_state = 42)

# TODO: Create the parameters list you wish to tune, using a dictionary if needed.
# HINT: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}
parameters = { 'n_estimators': [50, 100, 150], 'learning_rate': [0.1, 0.5, 0.8, 1]}

# TODO: Make an fbeta_score scoring object using make_scorer()
scorer = make_scorer(fbeta_score, beta=0.5)

# TODO: Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(clf, parameters,scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
```
<p><em>
Unoptimized model<br>
------<br>
Accuracy score on testing data: 0.8576<br>
F-score on testing data: 0.7246<br>
</em>

<em>
Optimized Model<br>
------<br>
Final accuracy score on the testing data: 0.8630<br>
Final F-score on the testing data: 0.7356<br>
</em></p>

<h4>Question 5 - Final Model Evaluation</h4>

<ul>
  <li>What is your optimized model's accuracy and F-score on the testing data?</li>
  <li>Are these scores better or worse than the unoptimized model?</li>
  <li>How do the results from your optimized model compare to the naive predictor benchmarks you found earlier in Question 1?</li>
</ul>

<p>Note: Fill in the table below with your results, and then provide discussion in the Answer box.</p>

<p><strong>Results:</strong></p>
<p align = "center"><kbd><img src = "/images/table3.png"></kbd></p>
<p><strong>Answer:</strong></p>
<p>As you can see, the optimized model improved both the accuracy score and the f-score quite significantly, with accuracy improving by ~.005 and f-score improving by ~0.01. This clearly establishes the optimized model as the superior model. The naive predictor had a accuracy score of 0.2478 and an f-score of 0.2365. Again, our optimized model has four times the accuracy and three times the f-score. This is to be expected as our naive predictor simply predicted everyone to have more than 50K income.</p>


<h3>Feature Importance</h3>
<p>An important task when performing supervised learning on a dataset like the census data we study here is determining which features provide the most predictive power. By focusing on the relationship between only a few crucial features and the target label we simplify our understanding of the phenomenon, which is most always a useful thing to do. In the case of this project, that means we wish to identify a small number of features that most strongly predict whether an individual makes at most or more than $50,000.<br>

Choose a scikit-learn classifier (e.g., adaboost, random forests) that has a feature_importance_ attribute, which is a function that ranks the importance of features according to the chosen classifier. In the next python cell fit this classifier to training set and use this attribute to determine the top 5 most important features for the census dataset.</p>

<h4>Question 6 - Feature Relevance Observation</h4>
<p>When Exploring the Data, it was shown there are thirteen available features for each individual on record in the census data. Of these thirteen records, which five features do you believe to be most important for prediction, and in what order would you rank them and why?</p>

<p><strong>Answer:</strong></p>
<ol>
  <li>occupation: Type of job will determine range of salary and therefore income</li>
  <li>capital_gain: will combine with capital_loss to determine net income</li>
  <li>capital_loss: will combine with capital_gain to determine net income</li>
  <li>education_level: Those with advanced degrees tend to have higher average salaries than those who do not.</li>
  <li>age: the older you are, the more likely you are to have progressed through the ranks of a particular company or amassed enough experience for that experience to be worthy of a higher salary</li>
</ol>
<h4>Implementation - Extracting Feature Importance</h4>
<p>Choose a scikit-learn supervised learning algorithm that has a feature_importance_ attribute availble for it. This attribute is a function that ranks the importance of each feature when making predictions based on the chosen algorithm.<br>

In the code cell below, you will need to implement the following:</p>
<ul><li>Import a supervised learning model from sklearn if it is different from the three used earlier.</li><li>Train the supervised model on the entire training set.</li><li>Extract the feature importances using '.feature_importances_'.</li></ul>

<p>In [18]:</p>

```python3
# TODO: Import a supervised learning model that has 'feature_importances_'
from sklearn.ensemble import AdaBoostClassifier

# TODO: Train the supervised model on the training set using .fit(X_train, y_train)
model = AdaBoostClassifier(random_state = 42).fit(X_train, y_train)

# TODO: Extract the feature importances using .feature_importances_ 
importances = model.feature_importances_

# Plot
vs.feature_plot(importances, X_train, y_train)
```
<p align = "center"><kbd><img src = "/images/graph5.png"></kbd></p>

<h4>Question 7 - Extracting Feature Importance</h4>
<p>Observe the visualization created above which displays the five most relevant features for predicting if an individual makes at most or above $50,000.</p>
<ul><li>How do these five features compare to the five features you discussed in Question 6?</li><li>
If you were close to the same answer, how does this visualization confirm your thoughts?</li><li>If you were not close, why do you think these features are more relevant?</li></ul>

<p><strong>Answer:</strong></p>
<p>The five features are similar to what I predicted in question 6, with only hours-per-week being the only variable I did not predict to be important. The graph confirms my belief in the importance of capital_gain, capital_loss, and education_num. However, while I believed age to be important, I vastly undersold its importance. In addition, while I missed hours-per-week's importance, it's logical to understand how that affects income, especially given some jobs' tendency to compensate on a per hour basis.</p>


<h4>Feature Selection</h4>
<p>How does a model perform if we only use a subset of all the available features in the data? With less features required to train, the expectation is that training and prediction time is much lower — at the cost of performance metrics. From the visualization above, we see that the top five most important features contribute more than half of the importance of all features present in the data. This hints that we can attempt to reduce the feature space and simplify the information required for the model to learn. The code cell below will use the same optimized model I found earlier, and train it on the same training set with only the top five important features.</p>


<p>In [19]:</p>

```python3
# Import functionality for cloning a model
from sklearn.base import clone

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# Train on the "best" model found from grid search earlier
clf = (clone(best_clf)).fit(X_train_reduced, y_train)

# Make new predictions
reduced_predictions = clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print("Final Model trained on full data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
print("\nFinal Model trained on reduced data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5)))
```

<p><em>
  Final Model trained on full data<br>
------<br>
Accuracy on testing data: 0.8630<br>
F-score on testing data: 0.7356<br>
</em>
<em>
  Final Model trained on reduced data<br>
------<br>
Accuracy on testing data: 0.8375<br>
F-score on testing data: 0.6889<br>
</em></p>

<h4>Question 8 - Effects of Feature Selection</h4>
<ul><li>How does the final model's F-score and accuracy score on the reduced data using only five features compare to those same scores when all features are used?</li><li>
If training time was a factor, would you consider using the reduced data as your training set?</li></ul>
<p><strong>Answer:</strong></p>
<p>The final model's accuracy score and f-score on reduced data is about 0.03 and 0.05 lower than that of the final model's accuracy and f-score on full data, respectively. Certainly, if training time was a factor and our dataset was of a significantly large size, I would consider using just the reduced data, given the fact that these variables contribute to nearly half the weight of all the variables. However, the ideal solution is still to use all variables given the significant drop off in f-score</p>
