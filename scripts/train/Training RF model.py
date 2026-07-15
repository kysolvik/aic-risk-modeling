# -*- coding: utf-8 -*-
"""
RF training model - Local Python (Scikit-Learn)
"""

import ee
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             cohen_kappa_score, f1_score, average_precision_score)
ee.Authenticate()
ee.Initialize(project='macedo-lab-general-9051')

trainingSet = ee.FeatureCollection('projects/columbia-research-project/assets/TrainingSet')
validationSet = ee.FeatureCollection('projects/columbia-research-project/assets/ValidationSet')

classProperty = 'class'
propertiesToExclude = [classProperty, 'fire_type', 'confidence', 'year', 'system:index']

allProperties = trainingSet.first().propertyNames().getInfo()
predictorBands = [prop for prop in allProperties if prop not in propertiesToExclude]

print('Bands :', predictorBands)


def ee_to_pandas(feature_collection, columns_to_keep):
    features = feature_collection.select(columns_to_keep).getInfo()['features']
    df = pd.DataFrame([f['properties'] for f in features])
    return df.dropna()

columns_needed = predictorBands + [classProperty]


df_train = ee_to_pandas(trainingSet, columns_needed)
df_val = ee_to_pandas(validationSet, columns_needed)


X_train = df_train[predictorBands]
y_train = df_train[classProperty]

X_val = df_val[predictorBands]
y_val = df_val[classProperty]

classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

classifier.fit(X_train, y_train)

"""
Results
"""
y_pred = classifier.predict(X_val)
y_proba = classifier.predict_proba(X_val)[:, 1]

print('\n Results')

print('\nCondusion Matrix :')
print(confusion_matrix(y_val, y_pred))
print('\nOverall accuracy :', accuracy_score(y_val, y_pred))
print('Kappa :', cohen_kappa_score(y_val, y_pred))
print('F1-Score (Fire class) :', f1_score(y_val, y_pred))
print('PR-AUC :', average_precision_score(y_val, y_proba))
