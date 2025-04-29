from rdkit import Chem
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import RDLogger 


def load_molecule_dataset(file_path, dataset_name):
    data = Chem.SDMolSupplier(file_path)
    df = pd.DataFrame()
    for i, mol in tqdm(enumerate(data)):
        try:
            smiles = Chem.MolToSmiles(mol)
            if dataset_name in ["ADRA1A", "ALOX5AP", "ATR", "JAK1", "KOR", "MUSC1", "MUSC2"]:
                target = np.log(float(mol.GetProp("Ki (nM)").replace(">", "").replace("<", "")))
            elif dataset_name in ["DPP4","JAK2","LIPO"]:
                target = float(mol.GetProp("target").replace(">", "").replace("<", ""))
            elif dataset_name in ["HLMC"]:
                target = float(mol.GetProp("LOG HLM_CLint (mL/min/kg)").replace(">", "").replace("<", ""))
            elif dataset_name in ["SOL"]:
                target = float(mol.GetProp("LOG SOLUBILITY PH 6.8 (ug/mL)").replace(">", "").replace("<", ""))

        except Exception as e:
            smiles = 'invalid'
            target = np.nan

        df.loc[i, "smiles"] = smiles
        df.loc[i, "target"] = target

    mask = [True if Chem.MolFromSmiles(m) != None else False for m in df.loc[:, "smiles"]]
    df = df[mask].reset_index(drop=True)
    df["dataset"] = "dataset_name"
    df = df.reindex(['dataset', 'smiles', 'target'], axis=1)
    return df


def remove_duplicates(dataset):
    print(f"This dataset consist of {dataset.shape[0]} molecular examples, where just {dataset['smiles'].nunique()} are unique")
    df = dataset.drop_duplicates(subset=['smiles'], keep='first')
    return df

def save_molecule_dataset_as_csv(df, nameDataset):
    df.to_csv(f'data/datasetCSV/{nameDataset}.csv', index=False)


def main():
    RDLogger.DisableLog('rdApp.*')   
    RDLogger.DisableLog('rdkit.*')

    nameDataset = [
        "ADRA1A",
        "ALOX5AP",
        "ATR",
        "DPP4",
        "HLMC",
        "JAK1",
        "JAK2",
        "KOR",
        "LIPO",
        "MUSC1",
        "MUSC2",
        "SOL"]

    for name in nameDataset:
        print(" /n-- Dataset: ", name, " --")
        file_path = f"data/datasetRaw/{name}.sdf"
        df = load_molecule_dataset(file_path, name)
        df = remove_duplicates(df)
        save_molecule_dataset_as_csv(df, name)
        print(f"Dataset {name} saved as CSV at data/datasetCSV/{name}.csv .")



if __name__ == "__main__":
    main()