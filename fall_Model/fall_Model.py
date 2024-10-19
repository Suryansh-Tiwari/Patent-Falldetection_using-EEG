import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import precision_score

# readingn the datset and getting the number of fall and non fall datas

df=pd.read_csv('/content/fall_detection_dataset.csv')
print(df)


def fallnumber (df):
  fall=df[df['falling']==1]
  print(len(fall))


def nonfallnumber (df):
  nonfall=df[df['falling']==0]
  print(len(nonfall))


# Resampling the data 



fall_df = df[df['falling'] == 1]
non_fall_df = df[df['falling'] == 0]


fall_upsampled = resample(fall_df,
                          replace=True,
                          n_samples=len(non_fall_df),
                          random_state=42)


balanced_df = pd.concat([fall_upsampled, non_fall_df])



# print(fallnumber(balanced_df))
# print(nonfallnumber(balanced_df))


# distributing the data 

train, validate, test = np.split(balanced_df.sample(frac=1), [int(0.7 * len(balanced_df)), int(0.7 * len(balanced_df)) + int(0.2 * len(balanced_df))])

# print(fallnumber(train))
# print(nonfallnumber(train))

# print(fallnumber(validate))
# print(nonfallnumber(validate))

# print(fallnumber(test))
# print(nonfallnumber(test))



# resampling the training data 


train_fall = train[train['Fall'] == 1]
train_non_fall = train[train['Fall'] == 0]



train_fall_upsampled = resample(train_fall,
                                 replace=True,
                                 n_samples=len(train_non_fall),
                                 random_state=42)

train_balanced = pd.concat([train_fall_upsampled, train_non_fall])

# print(fallnumber(train_balanced))
# print(nonfallnumber(train_balanced))

# print(fallnumber(validate))
# print(nonfallnumber(validate))

print(test.head())





##############################################   pridiction model (dission tree )


X = train_balanced.drop('Fall', axis=1)
y = train_balanced['Fall']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)

accuracy = accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred))





# Accuracy: 0.7142857142857143
#               precision    recall  f1-score   support

#            0       0.50      1.00      0.67         4
#            1       1.00      0.60      0.75        10  ---- the prission is 100% when actual fall occured

#     accuracy                           0.71        14  ----- the acuracy to detect the fall of the model is 71%
#    macro avg       0.75      0.80      0.71        14
# weighted avg       0.86      0.71      0.73        14

# #####################################################     (Validation)


X_validate = validate.drop('Fall', axis=1)
y_validate = validate['Fall']

y_validate_pred = model.predict(X_validate)

y_validate_pred_binary = (y_validate_pred > 0.5).astype(int)

accuracy_validate = accuracy_score(y_validate, y_validate_pred_binary)
print(f"Validation Accuracy: {accuracy_validate}")
print(classification_report(y_validate, y_validate_pred_binary))


# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
# Validation Accuracy: 0.55
#               precision    recall  f1-score   support

#            0       0.60      0.55      0.57        11
#            1       0.50      0.56      0.53         9

#     accuracy                           0.55        20
#    macro avg       0.55      0.55      0.55        20
# weighted avg       0.55      0.55      0.55        20


from joblib import dump, load
dump(model, 'falldetect.joblib')
