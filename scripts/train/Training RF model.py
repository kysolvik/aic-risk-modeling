# -*- coding: utf-8 -*-
"""
RF training model - CSV version (Scikit-Learn)
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             cohen_kappa_score, f1_score, average_precision_score)

chemin_train = 'TrainingSetFirstTest.csv'
chemin_val = 'ValidationSetFirstTest.csv'


df_train = pd.read_csv(chemin_train)
df_val = pd.read_csv(chemin_val)

df_train = df_train.dropna()
df_val = df_val.dropna()

classProperty = 'class'


propertiesToExclude = [classProperty, 'fire_type', 'confidence', 'fireSize', 'year', 'system:index', '.geo']


allProperties = df_train.columns.tolist()


predictorBands = [prop for prop in allProperties if prop not in propertiesToExclude]

print('\nBands :', predictorBands)

X_train = df_train[predictorBands]
y_train = df_train[classProperty]

X_val = df_val[predictorBands]
y_val = df_val[classProperty]

#Training

classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

classifier.fit(X_train, y_train)

#Results
y_pred = classifier.predict(X_val)
y_proba = classifier.predict_proba(X_val)[:, 1]


print('\nConfusion Matrix :')
print(confusion_matrix(y_val, y_pred))

print('\nOverall accuracy :', accuracy_score(y_val, y_pred))
print('Kappa :', cohen_kappa_score(y_val, y_pred))
print('F1-Score (Fire class) :', f1_score(y_val, y_pred))
print('PR-AUC :', average_precision_score(y_val, y_proba))
