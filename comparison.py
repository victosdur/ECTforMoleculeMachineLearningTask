import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

import questionary
from questionary import Style

plt.rcParams.update({
    'font.size': 14,              # tamaño base de fuente
    'axes.titlesize': 16,         # tamaño del título del gráfico
    'axes.labelsize': 16,         # etiquetas de los ejes
    'xtick.labelsize': 12,        # tamaño de los ticks en el eje x
    'ytick.labelsize': 16,        # tamaño de los ticks en el eje y
    'legend.fontsize': 13,        # tamaño de la leyenda
    'figure.titlesize': 16,       # título general (si usas suptitle)
})

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

def get_median_order(df, metric):
    df_metric = df[df["metric"] == metric]
    if metric == "R2":
        median_order = df_metric.groupby("Method")["value"].mean().sort_values(ascending=False).index
    else:
        median_order = df_metric.groupby("Method")["value"].mean().sort_values().index
    return median_order

def read_results_file(dataset_path):
    dfs = []

    if not os.path.exists(dataset_path) or len(os.listdir(dataset_path)) == 0:
        return False
    else:
        for filename in os.listdir(dataset_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(dataset_path, filename)
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
                elif method_name in "ECT+Fingerprint":
                    df["Category"] = "ECT+Fingerprint"

                dfs.append(df)
    return dfs

def concat_results_in_single_csv(results_dfs):
    df_results = pd.concat(results_dfs, ignore_index=True)
    return df_results

def plot_results(results_df, type_split, name):
    fig, axes = plt.subplots(1, 3, figsize=(15, 10))
    results_df_selected = results_df[results_df['dataset']==type_split].copy()

    rmse_order = get_median_order(results_df_selected, "RMSE")
    sns.boxplot(data=results_df_selected[results_df_selected["metric"]=="RMSE"], x="value",y="Method", hue="Category", order=rmse_order, ax=axes[0],legend=False)
    axes[0].set_xlabel('RMSE')
    axes[0].set_ylabel('') 
    # axes[0].set_title(f'RMSE {type_split} error per method for {name} dataset')

    r2_order = get_median_order(results_df_selected, "R2")
    sns.boxplot(data=results_df_selected[results_df_selected["metric"]=="R2"], x="value",y="Method", order=r2_order, hue="Category", ax=axes[1],legend=False)
    axes[1].set_xlabel('R2')
    axes[1].set_ylabel('') 
    # axes[1].set_title(f'R2 {type_split} error per method for {name} dataset')

    mae_order = get_median_order(results_df_selected, "MAE")
    sns.boxplot(data=results_df_selected[results_df_selected["metric"]=="MAE"], x="value",y="Method", hue="Category", order=mae_order, ax=axes[2])
    axes[2].set_xlabel('MAE')
    axes[2].set_ylabel('') 
    # axes[2].set_title(f'MAE {type_split} error per method for {name} dataset')


    plt.legend(title='Representation type')

    plt.tight_layout()
    plt.savefig(f'figures/{name}/boxplotsComparison_{type_split}_{name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def process_comparison(dataset_name, resultsPath="results"):
    datasetResultsPath = os.path.join(resultsPath, dataset_name)

    try:
        os.stat(f"figures/{dataset_name}/")
    except:
        os.mkdir(f"figures/{dataset_name}/")

    results_dfs = read_results_file(datasetResultsPath)
    if results_dfs == False: 
        print(f"[X] No results found for {dataset_name} dataset. Please, check the path and try again.\n")
    else:
        results = concat_results_in_single_csv(results_dfs)
        plot_results(results, "train", dataset_name)
        plot_results(results, "test", dataset_name)

        if dataset_name == "AllDataset":
            print(f"Done for general dataset (all datasets together). Plots saved at figures/{dataset_name}/")
        else:
            print(f"Done for {dataset_name} dataset. Plots saved at figures/{dataset_name}/")

def main():
    try:
        choice = show_menu()
    except Exception as e:
        print(f"\n    [X] Error inesperado: {e}\n")
    
    resultsPath = "results"

    if choice == 1:
        process_comparison("AllDataset", resultsPath)

    elif choice == 2:
        process_comparison("AllDataset", resultsPath)
        for dataset in nameDataset:
            process_comparison(dataset, resultsPath)

    elif choice == 3:
        for dataset in nameDataset:
            process_comparison(dataset, resultsPath)

    elif choice > 3:
        value = choice - 4
        SelectedDataset = nameDataset[value]
        process_comparison(SelectedDataset, resultsPath)

if __name__ == "__main__":
    main()