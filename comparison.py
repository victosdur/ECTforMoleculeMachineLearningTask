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
    "HLMC",
    "JAK1",
    "JAK2",
    "KOR",
    "LIPO",
    "MUSC1",
    "MUSC2",
    "SOL"]

fingerprint_list = [
    "ECFP4",
    "MAP4",
    "RDKit",
    "AVALON",
    "MACCS",
    "KR",
    "PubChem",
    "CATS2D",
    "Pharm2D",
    "EState"
]

descriptor_list = [
    "TOPO",
    "RingDesc",
    "FGCount",
    "2DAP",
    "ConstIdx",
    "WalkPath",
    "MolProp",
]

gnns_list = [
    "GCN",
    "GAT",
    "AttentiveFP"
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
    {"name": "For all dataset indepently", "value": 3},
    {"name": "ADRA1A", "value": 4},
    {"name": "ALOX5AP", "value": 5},
    {"name": "ATR", "value": 6},
    {"name": "DPP4", "value": 7},
    {"name": "HLMC", "value": 8},
    {"name": "JAK1", "value": 9},
    {"name": "JAK2", "value": 10},
    {"name": "KOR", "value": 11},
    {"name": "LIPO", "value": 12},
    {"name": "MUSC1", "value": 13},
    {"name": "MUSC2", "value": 14},
    {"name": "SOL", "value": 15},
    
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

def get_median_order(df, metric):
    df_metric = df[df["metric"] == metric]
    median_order = df_metric.groupby("Method")["value"].median().sort_values().index
    return median_order

def main():
    try:
        choice = show_menu()
    except Exception as e:
        print(f"\n    [X] Error inesperado: {e}\n")
    
    resultsPath = "results"

    if choice == 1:
        print(choice)

    elif choice == 2:
        print(choice)

    elif choice == 3:
        print(choice)

    elif choice > 3:
        value = choice - 4
        SelectedDataset = nameDataset[value]
        datasetResultsPath = os.path.join(resultsPath, SelectedDataset)

        dfs = []

        for filename in os.listdir(datasetResultsPath):
            if filename.endswith('.csv'):
                file_path = os.path.join(datasetResultsPath, filename)
                df = pd.read_csv(file_path)
                
                method_name = os.path.splitext(filename)[0]
                df['Method'] = method_name
                if method_name in fingerprint_list:
                    df['Category'] = "Fingerprint"
                elif method_name in descriptor_list:
                    df['Category'] = "Descriptor"
                elif method_name in gnns_list:
                    df["Category"] = "GNN embedding"
                elif method_name in "ECT":
                    df["Category"] = "ECT"

                dfs.append(df)

        df_results = pd.concat(dfs, ignore_index=True)
        # df_results_train = df_results[df_results['dataset']=="train"].copy()
        df_results_tests = df_results[df_results['dataset']=="test"].copy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 10))
        rmse_order = get_median_order(df_results_tests, "RMSE")
        print(rmse_order)
        sns.boxplot(data=df_results_tests[df_results_tests["metric"]=="RMSE"], x="value",y="Method", hue="Category", order=rmse_order, ax=axes[0],legend=False)
        axes[0].set_xlabel('RMSE')
        axes[0].set_title(f'RMSE test error per method for {SelectedDataset} dataset')

        r2_order = get_median_order(df_results_tests, "R2")
        sns.boxplot(data=df_results_tests[df_results_tests["metric"]=="R2"], x="value",y="Method", order=r2_order, hue="Category", ax=axes[1],legend=False)
        axes[1].set_xlabel('R2')
        axes[1].set_title(f'R2 test error per method for {SelectedDataset} dataset')

        mae_order = get_median_order(df_results_tests, "MAE")
        sns.boxplot(data=df_results_tests[df_results_tests["metric"]=="MAE"], x="value",y="Method", hue="Category", order=mae_order, ax=axes[2])
        axes[2].set_xlabel('MAE')
        axes[2].set_title(f'MAE test error per method for {SelectedDataset} dataset')

    
        plt.legend(title='Representation type')

        plt.tight_layout()
        plt.savefig(f'figures/boxplotsComparison_{SelectedDataset}.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()