import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt

#Input Bands

PATH_TRAIN = "/home/alistair/Documents/GEE/Sets/TrainingSet_Amazon_EmbedV2MODISDistDistWDPAV2ONIWindDeforestETPopLightsTopoFireMemo_2016-2023.csv"
PATH_VALID = "/home/alistair/Documents/GEE/Sets/RepValidationSet_Amazon_EmbedV2MODISDistDistWDPAOV2NIWindDeforestETPopLightsTopoFireMemo_2022-2023.csv"
TARGET = "class"
VALID_YEAR = 2023

embeddings_bands = [f"A{n:02d}" for n in range(64)]
oni_bands = [
    'ONI_Q1_prev', 'ONI_Q2_prev', 'ONI_Q3_prev', 'ONI_Q4_prev',
    'ONI_Q1_curr', 'ONI_Q2_curr', 'ONI_Q3_curr', 'ONI_Q4_curr'
]
additional_bands = [
    'DistanceHumanActivities', 'Wind_Speed_DrySeason', 'Recent_Deforestation',
    'DistanceAllProtectedAreas', 'Population_Density', 'DistanceStrictProtectedAreas',
    'DistanceSustainableUseProtectedAreas', 'Evapotranspiration', 'Nighttime_Lights',
    'Elevation', 'Slope', 'Fire_Memory_5y'
]

FEATURES = embeddings_bands + additional_bands + oni_bands

df_train = pd.read_csv(PATH_TRAIN)
df_valid = pd.read_csv(PATH_VALID)

if 'year' in df_valid.columns:
    df_valid = df_valid[df_valid['year'] == VALID_YEAR]

df_train_clean = df_train.dropna(subset=FEATURES + [TARGET])
df_valid_clean = df_valid.dropna(subset=FEATURES + [TARGET])

X_train = df_train_clean[FEATURES]
y_train = df_train_clean[TARGET]
X_valid = df_valid_clean[FEATURES]
y_valid = df_valid_clean[TARGET]

print(f" (Training Set Size: {X_train.shape}, Validation Set Size: {X_valid.shape})")

# Main Loop

step_removal = 2
lower_limit = 1
thresholds_to_test = [0.75, 0.77, 0.79, 0.81, 0.83, 0.85, 0.86]
min_samples_leaf_options = [4, 5, 6, 7, 8, 9, 10, 11, 12]
max_features_options = [6, 7, 8, 9, 10, 11, 12, 15]

global_history = []
total_combinations = len(min_samples_leaf_options) * len(max_features_options)
counter = 1

for leaf_size in min_samples_leaf_options:
    for m_features in max_features_options:

        print(f"\n==================================================")
        print(f"[{counter}/{total_combinations}] 🌲 Test: Leaf={leaf_size} | Split(Max_Feat)={m_features}")
        print(f"==================================================")

        current_variables = list(FEATURES)
        iteration = 1

        while len(current_variables) >= lower_limit:
            if iteration % 5 == 0 or len(current_variables) == len(FEATURES):
                print(f"   🔄 Evaluation with {len(current_variables)} variables...")

            X_train_current = X_train[current_variables]
            X_valid_current = X_valid[current_variables]

            rf_current = RandomForestClassifier(
                n_estimators=100,
                max_features=m_features,
                min_samples_leaf=leaf_size,
                bootstrap=True,
                max_samples=0.632,
                oob_score=True,
                random_state=42,
                n_jobs=-1
            )
            rf_current.fit(X_train_current, y_train)

            oob_error = 1.0 - rf_current.oob_score_

            probabilities = rf_current.predict_proba(X_valid_current)[:, 1]
            best_kappa = -1
            best_threshold = 0

            for threshold in thresholds_to_test:
                predictions = (probabilities >= threshold).astype(int)
                kappa = cohen_kappa_score(y_valid, predictions)
                if kappa > best_kappa:
                    best_kappa = kappa
                    best_threshold = threshold

            distance = np.sqrt((oob_error - 0)**2 + (1 - best_kappa)**2)

            global_history.append({
                'Min_Leaf': leaf_size,
                'Max_Features': m_features,
                'Num_Variables': len(current_variables),
                'Distance': distance,
                'OOB_Error': oob_error,
                'Kappa': best_kappa,
                'Opt_Threshold': best_threshold,
                'Variables': list(current_variables)
            })

            importances = rf_current.feature_importances_
            sorted_indices = np.argsort(importances)
            indices_to_keep = sorted_indices[step_removal:]
            current_variables = [current_variables[i] for i in indices_to_keep]
            iteration += 1

        counter += 1

#Best Model

df_rfe_global = pd.DataFrame(global_history)
df_rfe_global = df_rfe_global.sort_values(by='Distance', ascending=True).reset_index(drop=True)

winner = df_rfe_global.iloc[0]

print("\n" + "="*66)
print("BEST MODEL")
print("="*66)
print(f"Best Leaf config (minLeafPopulation) : {winner['Min_Leaf']}")
print(f"Best Split config (variablesPerSplit): {winner['Max_Features']}")
print(f"Optimal number of variables          : {winner['Num_Variables']}")
print(f"Distance to perfection               : {winner['Distance']:.4f}")
print(f"OOB Error : {winner['OOB_Error']:.4f} | Kappa : {winner['Kappa']:.4f}")
print(f"Alert trigger threshold              : {winner['Opt_Threshold']*100}%")
print("\n✅ VARIABLES TO KEEP IN GEE:")
print(winner['Variables'])
print("="*66)

#Graphs

plt.figure(figsize=(12, 6))

# Find the 3 best combinations of (Min_Leaf, Max_Features)
best_configs = df_rfe_global.drop_duplicates(subset=['Min_Leaf', 'Max_Features']).head(3)

for idx, row in best_configs.iterrows():
    leaf = row['Min_Leaf']
    feat = row['Max_Features']


    df_config = df_rfe_global[(df_rfe_global['Min_Leaf'] == leaf) & (df_rfe_global['Max_Features'] == feat)]
    df_config = df_config.sort_values(by='Num_Variables', ascending=False)

    plt.plot(df_config['Num_Variables'], df_config['Distance'], marker='.', linewidth=2,
             label=f'Leaf={leaf} | Split={feat} (Best Dist: {row["Distance"]:.4f})')

plt.gca().invert_xaxis()
plt.title("RFE Evolution (Only the top 3 configurations)", fontsize=14)
plt.xlabel("Number of Retained Variables", fontsize=12)
plt.ylabel("Euclidean Distance (Lower = Better)", fontsize=12)
plt.legend(title="Configurations", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
