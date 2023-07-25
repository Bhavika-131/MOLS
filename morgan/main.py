import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# Load dataset 1 from CSV file (format: ,RECEPTOR,AGONIST,SMILES)
dataset1 = pd.read_csv('SMILES_mol.csv')

# Load dataset 2 from CSV file (format: SMILES,ODOR)
dataset2 = pd.read_csv('data.csv')

# Calculate Morgan fingerprint for each smile in dataset 1
fps_dataset1 = []
for smile in dataset1['SMILES']:
    mol = Chem.MolFromSmiles(smile)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2) # Change radius value if needed
    fps_dataset1.append(fp)

# Calculate Morgan fingerprint for each smile in dataset 2
fps_dataset2 = []
for smile in dataset2['SMILES']:
    mol = Chem.MolFromSmiles(smile)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2) # Change radius value if needed
    fps_dataset2.append(fp)

# Save fingerprints to new files (Optional step)
pd.DataFrame(fps_dataset1).to_csv('fingerprint_dataset1.csv', index=False)
pd.DataFrame(fps_dataset2).to_csv('fingerprint_dataset2.csv', index=False)

# Calculate distances between two fingerprints using different metrics

def calculate_distance(fp_1, fp_2):
    tanimoto_dist = DataStructs.TanimotoSimilarity(fp_1, fp_2) # Tanimoto distance 
    cosine_dist   = DataStructs.CosineSimilarity(fp_1, fp_2)   # Cosine distance
    hamming_dist  = DataStructs.FingerprintSimilarity(fp_1, fp_2) # Hamming distance 
    euclidean_dist = DataStructs.DiceSimilarity(fp_1, fp_2)     # Euclidean distance
    
    return tanimoto_dist, cosine_dist, hamming_dist, euclidean_dist

# Example usage:
smile1 = dataset1.loc[0]['SMILES']
mol1 = Chem.MolFromSmiles(smile1)
fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)

smile2 = dataset2.loc[0]['SMILES']
mol2 = Chem.MolFromSmiles(smile2)
fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)

tanimoto_distance, cosine_distance , hamming_distance ,euclidean_distance= calculate_distance(fp1 , fp2)