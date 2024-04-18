import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import StringIO
import os
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored

print(colored("--------------> Séance 1 <--------------", "blue"))

if not os.path.exists("database"):
    os.makedirs("database")

# Importer et concaténer MOD1
mod1_parts = []
for i in range(1, 9):
    temp = pd.read_csv(f"data/Libelium New/part{i}/mod1.txt", sep="\t", header=None, names=("Time", "RH", "Temperature", "TGS4161", "MICS2714", "TGS2442", "MICS5524", "TGS2602", "TGS2620"))
    mod1_parts.append(temp)

mod1 = pd.concat(mod1_parts)
print("MOD 1 data:")
print(mod1.head())

# Colonnes MOD1
print("Colonnes MOD1:")
print(mod1.columns)

# Importer et concaténer MOD2
mod2_parts = []
for i in range(1, 9):
    temp = pd.read_csv(f"data/Libelium New/part{i}/mod2.txt", sep="\t", header=None, names=("Time", "RH", "Temperature", "TGS4161", "MICS2714", "TGS2442", "MICS5524", "TGS2602", "TGS2620"))
    mod2_parts.append(temp)

mod2 = pd.concat(mod2_parts)
print("MOD 2 data:")
print(mod2.head())

# Colonnes MOD2
print("Colonnes MOD2:")
print(mod2.columns)

# Importer et concaténer PODs
pod_files = {
    "POD 200085": ["data/PODs/14_nov-22_nov-Pods/POD 200085.csv", "data/PODs/23_nov-12_dec-Pods/POD 200085.csv", "data/PODs/fevrier_mars_2023_pods/POD 200085.csv"],
    "POD 200086": ["data/PODs/14_nov-22_nov-Pods/POD 200086.csv", "data/PODs/23_nov-12_dec-Pods/POD 200086.csv", "data/PODs/fevrier_mars_2023_pods/POD 200086.csv"],
    "POD 200088": ["data/PODs/14_nov-22_nov-Pods/POD 200088.csv", "data/PODs/23_nov-12_dec-Pods/POD 200088.csv", "data/PODs/fevrier_mars_2023_pods/POD 200088.csv"]
}

pods = {}

for key, value in pod_files.items():
    pod_parts = []
    for file_path in value:
        temp = pd.read_csv(file_path, sep=";", skiprows=(1, 2, 3, 4))
        pod_parts.append(temp)
    pods[key] = pd.concat(pod_parts) 

    # Colonnes POD
    print(f"Colonnes {key}:")
    print(pods[key].columns)

    print(f"{key} data:")
    print(pods[key].head())

print("Fusion des données...")
merged_data = mod1.merge(mod2, on='Time', suffixes=('_mod1', '_mod2'))

for key, value in pods.items():
    merge_col_name = None
    if 'Time' in value.columns:
        merge_col_name = 'Time'
    elif 'time' in value.columns:
        merge_col_name = 'time'
    else:
        print(f"La colonne 'Time' est manquante pour {key}")
        continue

    suffix = f"_{key.lower().replace(' ', '_')}"
    merged_data = merged_data.merge(value, left_on='Time', right_on=merge_col_name, suffixes=('', suffix))
    merged_data = merged_data.drop(columns=[merge_col_name]) if merge_col_name != 'Time' else merged_data

# Vérification des colonnes et des données avant d'enregistrer dans database.csv
print("Colonnes de merged_data:")
print(merged_data.columns)
print("Première ligne de merged_data:")
print(merged_data.head(1))

print("Enregistrement du fichier database.csv...")
merged_data.to_csv("database/database.csv", index=False, sep=';')
print("Fichier database.csv enregistré.")

print(colored("--------------> Séance 2 <--------------", "blue"))

# Vérification des colonnes dans database.csv
print("Vérification des colonnes de database.csv:")
saved_base = pd.read_csv("database/database.csv", sep=';')
print(saved_base.columns)

# Tâche 3 : Charger la base de données
base = pd.read_csv('database/database.csv', sep=';')

# Convertir la colonne 'Time' au format datetime avec la bonne timezone
base['Time'] = pd.to_datetime(base['Time'], dayfirst=True).dt.tz_localize('UTC').dt.tz_convert('UTC+01:00')

# Vérifier les noms et le format
print(base.dtypes)

# Charger l'onglet "Done so far" du fichier activites.xlsx
activities = pd.read_excel("data/activites.xlsx", sheet_name="Done so far", usecols=["activity", "Started", "Ended"])

# Supprimer les valeurs NaN
activities = activities.dropna()

# Convertir les colonnes Started et Ended en format datetime avec le bon fuseau horaire
activities["Started"] = pd.to_datetime(activities["Started"]).dt.tz_localize('UTC').dt.tz_convert('UTC+01:00')
activities["Ended"] = pd.to_datetime(activities["Ended"]).dt.tz_localize('UTC').dt.tz_convert('UTC+01:00')

# Vérifier les types de données
print(activities.dtypes)

# Vérifier les dimensions du DataFrame
print(activities.shape)

# Initialiser les variables pour stocker les instances d'activités et la longueur totale
activity_instances = {}
activity_lengths = {}

# Initialiser les variables pour stocker les instances d'activités et la longueur totale
activity_instances = {}
activity_lengths = {}

# Itérer à travers toutes les activités dans le calendrier
for index, row in activities.iterrows():
    start = row['Started'].tz_convert('UTC+01:00')
    end = row['Ended'].tz_convert('UTC+01:00')
    act = row['activity']

    # Extraire les données de l'activité actuelle
    new_activity_data = base[(base['Time'] >= start) & (base['Time'] <= end)].reset_index(drop=True).sort_values(by='Time').drop(columns='Time')

    # Ajouter l'activité actuelle à la liste des instances pour cette activité
    if act not in activity_instances:
        activity_instances[act] = []
        activity_lengths[act] = 0

    activity_instances[act].append(new_activity_data)
    activity_lengths[act] += len(new_activity_data)

def normalize_instance(instance, common_length):
    if len(instance) == 0: # Vérifiez si l'instance est vide
        return np.empty((0, common_length))

    original_length = len(instance)
    x = np.linspace(0, original_length - 1, num=common_length, endpoint=True)
    instance_array = np.array(instance.to_numpy())  # Convertir l'instance en un tableau numpy
    f = interp1d(np.arange(original_length), instance_array, kind='linear', axis=0, bounds_error=False, fill_value="extrapolate")  # Ajoutez les arguments bounds_error et fill_value
    normalized_instance = f(x)
    return normalized_instance

import numpy as np

def averageSignature(signature_values, common_length):
    normalized_instances = [normalize_instance(instance, common_length) for instance in signature_values]
    # Filtrez les instances vides ou contenant des NaN pour éviter les erreurs lors du calcul de la moyenne
    normalized_instances = [instance for instance in normalized_instances if len(instance) > 0 and not np.any(np.isnan(instance))]
    average_signature = np.mean(normalized_instances, axis=0)
    return average_signature

def averageSignatures(instances_list, lengths_list):
    avg_signatures = []
    for i in range(len(instances_list)):
        avg_signature = averageSignature(instances_list[i], lengths_list[i])
        avg_signatures.append(avg_signature)
    return avg_signatures

def plot_signatures(avg_activity_signatures, activity_labels):
    # Tracez les 4 premières signatures moyennes des activités
    for i in range(min(4, len(avg_activity_signatures))):
        plt.plot(avg_activity_signatures[i], label=activity_labels[i])

    plt.xlabel('Indice des valeurs normalisées')
    plt.ylabel('Valeurs normalisées')
    plt.title('Signatures moyennes des activités')
    plt.legend()

    # Enregistrez la figure dans un fichier image
    plt.savefig('activity_signatures.png', dpi=300)

    # Affiche le graphique (commenter cette ligne si elle génère des avertissements)
    #plt.show()

# Créer une liste de toutes les instances d'activité et une liste des longueurs moyennes
all_instances = [activity_instances[k] for k in sorted(activity_instances)]
all_lengths = [activity_lengths[k] // len(activity_instances[k]) for k in sorted(activity_lengths)]

# Calculer les signatures moyennes pour toutes les activités
avg_activity_signatures = averageSignatures(all_instances, all_lengths)

# Imprimer les signatures moyennes pour chaque activité
activity_labels = sorted(activity_instances)
for i in range(len(activity_labels)):
    print(f"Signature moyenne pour {activity_labels[i]} :")
    print(avg_activity_signatures[i])
    print("\n")

plot_signatures(avg_activity_signatures, activity_labels)

print(colored("--------------> Séance 3 <--------------", "blue"))