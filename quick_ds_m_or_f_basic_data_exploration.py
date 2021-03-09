# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 14:39:53 2018

@author: SW George Ke
"""

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../QuickML_data_utf8a.csv")

# lowercase string
df['phone_os'] = df['phone_os'].str.lower()
# removing leading and trailing whitespaces
df['phone_os'] = df['phone_os'].str.strip()

coded_star_signs = {'水瓶座':1, '雙魚座':2, '牡羊座':3, '金牛座':4, '雙子座':5, '巨蟹座':6, '獅子座':7, '處女座':8, '天秤座':9, '天蠍座':10, '射手座':11, '摩羯座':12}
coded_phone_os = {'apple':1, 'android':2, 'windows phone':3, 'johncena':4}
coded_gender = {2:-1} #girls as -1

coded_df = df.replace({"star_sign": coded_star_signs})
coded_df = coded_df.replace({"phone_os": coded_phone_os})
coded_df = coded_df.replace({"gender": coded_gender})

# ref: https://www.python-course.eu/lambda.php, http://book.pythontips.com/en/latest/lambdas.html
nan_rows = lambda df: df[df.isnull().any(axis=1)]
nan_rows(coded_df)

"""So we have identified the rows with NaN values, let's discard them to keep the process simple here.
Discarding records with missing values is not an uncommon practice when dealing missing values."""

cleaned_df = coded_df.dropna()

# get rid of records that contain 'unusual values'
cleaned_df = cleaned_df[(cleaned_df['height']<200) & (cleaned_df['height']>140) & (cleaned_df['weight']<200) & (cleaned_df['height']>100) & (cleaned_df['fb_friends']<=5000)]

cleaned_df_noString = cleaned_df.drop(columns=['id','timestamp','self_intro'])

#splitting the data into train and test because we want to study training set only.
train, test = train_test_split(cleaned_df_noString, test_size=0.33)

#writing out dataframes to csv for future ML
train.to_csv("QuickML_train.csv", encoding='utf-8')
test.to_csv("QuickML_test.csv", encoding='utf-8')

"""
Let's get started with some data exploration
"""

import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

group_by_gender = train.groupby(by=['gender'])
train_avg = group_by_gender.mean()
train_count = group_by_gender.count()

sns.pairplot(train.loc[:,train.dtypes == 'float64'])

corr = train.loc[:,train.dtypes == 'float64'].corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True))


"""
feature selection
"""
# ref: https://machinelearningmastery.com/feature-selection-machine-learning-python/
# ref: https://towardsdatascience.com/data-visualization-exploration-using-pandas-only-beginner-a0a52eb723d5
# Feature Importance with Extra Trees Classifier
from sklearn.ensemble import ExtraTreesClassifier

coded_gender2 = {-1:0} #girls as 0
train = train.replace({"gender": coded_gender2})

array = train.values
X = array[:,1:9]
Y = array[:,0]
Y=Y.astype('int')
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(train.columns)
print(model.feature_importances_) #higher score the better

f_weights = model.feature_importances_


"""
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy

# feature extraction
test2 = SelectKBest(score_func=chi2, k=4)
fit = test2.fit(X, Y)
# summarize scores
numpy.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])
"""
