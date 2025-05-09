import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

import questionary
from questionary import Style

nameDataset = [
    "ADRA1A",
    "ALOX5AP",
    "ATR",
    "DPP4",
    "JAK1",
    "JAK2",
    "KOR",
    "MUSC1",
    "MUSC2",
    ]

fingerprint_list = [
    "AVALON",
    "CATS2D",
    "ECFP4",
    "EState",
    "KR",
    "MACCS",
    "MAP4",
    "Pharm2D",
    "PubChem",
    "RDKit",
]

descriptor_list = [
    "2DAP",
    "ConstIdx",
    "FGCount",
    "MolProp",
    "RingDesc",
    "TOPO",
    "WalkPath",
]

gnns_list = [
    "AttentiveFP",
    "GAT",
    "GCN",
]


custom_style = Style([
    ('instruction', 'fg:#ffffff bold'),    # Texto de la pregunta
    ('pointer', 'fg:#34eb9b bold'),        # Puntero (»)
    ('highlighted', 'fg:#34eb9b bold'),    # Opción resaltada
    ('separator', 'fg:#cc5454'),           # Separador
])

menu_options = [
    {"name": "General Dataset (All datasets together)", "value": 1},
    {"name": "General Dataset and indepently for each specific dataset", "value": 2},
    {"name": "For all datasets indepently", "value": 3},
    {"name": "ADRA1A", "value": 4},
    {"name": "ALOX5AP", "value": 5},
    {"name": "ATR", "value": 6},
    {"name": "DPP4", "value": 7},
    {"name": "JAK1", "value": 8},
    {"name": "JAK2", "value": 9},
    {"name": "KOR", "value": 10},
    {"name": "MUSC1", "value": 11},
    {"name": "MUSC2", "value": 12},
    
]

def show_menu():
    choice = questionary.select(
        "\n ",
        choices=menu_options,
        style=custom_style,
        qmark="",
        pointer="→",
        instruction="For which dataset do you want to visualize results? Please, click enter to select :) \n",
        use_indicator=True
    ).ask()
    return choice

def print_dataset_results(dataset_name, resultsPath="results"):
    print(f"\n[*] Printing results (mean and standard deviation) for {dataset_name} dataset...\n")

    for enum, ect in enumerate(["ECT","ECT+Fingerprint"]):
        print("Method: ", ect)
        try:
            datasetResults = pd.read_csv(os.path.join(resultsPath, dataset_name, f"{ect}.csv"))
        except Exception as e:
            print(f"[X] No results found for {dataset_name} dataset for {fingerprint} method. Please, check the path or run the experiments with experiments.py file and try again.\n")

        datasetResultsTests = datasetResults[datasetResults["dataset"] == "test"]
        rmseResults = datasetResultsTests[datasetResultsTests["metric"] == "RMSE"]["value"].values
        r2Results = datasetResultsTests[datasetResultsTests["metric"] == "R2"]["value"].values
        maeResults = datasetResultsTests[datasetResultsTests["metric"] == "MAE"]["value"].values
        print(f"RMSE: {np.mean(rmseResults):.3f} ± {np.std(rmseResults):.3f}")
        print(f"R2: {np.mean(r2Results):.3f} ± {np.std(r2Results):.3f}")
        print(f"MAE: {np.mean(maeResults):.3f} ± {np.std(maeResults):.3f}")
        print(f"Done for {ect} method.\n")

    for enum, fingerprint in enumerate(fingerprint_list):
        print("Method: ", fingerprint)
        try:
            datasetResults = pd.read_csv(os.path.join(resultsPath, dataset_name, f"{fingerprint}.csv"))
        except Exception as e:
            print(f"[X] No results found for {dataset_name} dataset for fingerprint {fingerprint}. Please, check the path or run the experiments with experiments.py file and try again.\n")

        datasetResultsTests = datasetResults[datasetResults["dataset"] == "test"]
        rmseResults = datasetResultsTests[datasetResultsTests["metric"] == "RMSE"]["value"].values
        r2Results = datasetResultsTests[datasetResultsTests["metric"] == "R2"]["value"].values
        maeResults = datasetResultsTests[datasetResultsTests["metric"] == "MAE"]["value"].values
        print(f"RMSE: {np.mean(rmseResults):.3f} ± {np.std(rmseResults):.3f}")
        print(f"R2: {np.mean(r2Results):.3f} ± {np.std(r2Results):.3f}")
        print(f"MAE: {np.mean(maeResults):.3f} ± {np.std(maeResults):.3f}")
    
    print("Done for all fingerprints.\n")


    for enum, descriptor in enumerate(descriptor_list):
        print("Method: ", descriptor)
        try:
            datasetResults = pd.read_csv(os.path.join(resultsPath, dataset_name, f"{descriptor}.csv"))
        except Exception as e:
            print(f"[X] No results found for {dataset_name} dataset for descriptor {descriptor}. Please, check the path or run the experiments with experiments.py file and try again.\n")

        datasetResultsTests = datasetResults[datasetResults["dataset"] == "test"]
        rmseResults = datasetResultsTests[datasetResultsTests["metric"] == "RMSE"]["value"].values
        r2Results = datasetResultsTests[datasetResultsTests["metric"] == "R2"]["value"].values
        maeResults = datasetResultsTests[datasetResultsTests["metric"] == "MAE"]["value"].values
        print(f"RMSE: {np.mean(rmseResults):.3f} ± {np.std(rmseResults):.3f}")
        print(f"R2: {np.mean(r2Results):.3f} ± {np.std(r2Results):.3f}")
        print(f"MAE: {np.mean(maeResults):.3f} ± {np.std(maeResults):.3f}")
    
    print("Done for all descriptor.\n")

    for enum, gnn in enumerate(gnns_list):
        print("Method: ", gnn)
        try:
            datasetResults = pd.read_csv(os.path.join(resultsPath, dataset_name, f"{gnn}.csv"))
        except Exception as e:
            print(f"[X] No results found for {dataset_name} dataset for gnn {gnn}. Please, check the path or run the experiments with experiments.py file and try again.\n")

        datasetResultsTests = datasetResults[datasetResults["dataset"] == "test"]
        rmseResults = datasetResultsTests[datasetResultsTests["metric"] == "RMSE"]["value"].values
        r2Results = datasetResultsTests[datasetResultsTests["metric"] == "R2"]["value"].values
        maeResults = datasetResultsTests[datasetResultsTests["metric"] == "MAE"]["value"].values
        print(f"RMSE: {np.mean(rmseResults):.3f} ± {np.std(rmseResults):.3f}")
        print(f"R2: {np.mean(r2Results):.3f} ± {np.std(r2Results):.3f}")
        print(f"MAE: {np.mean(maeResults):.3f} ± {np.std(maeResults):.3f}")
    
    print("Done for all gnns.\n")



def main():
    try:
        choice = show_menu()
    except Exception as e:
        print(f"\n    [X] Error inesperado: {e}\n")
    
    resultsPath = "results"

    if choice == 1:
        print_dataset_results("AllDataset", resultsPath)

    elif choice == 2:
        print_dataset_results("AllDataset", resultsPath)
        for dataset in nameDataset:
            print_dataset_results(dataset, resultsPath)

    elif choice == 3:
        for dataset in nameDataset:
            print_dataset_results(dataset, resultsPath)

    elif choice > 3:
        value = choice - 4
        SelectedDataset = nameDataset[value]
        print_dataset_results(SelectedDataset, resultsPath)

if __name__ == "__main__":
    main()