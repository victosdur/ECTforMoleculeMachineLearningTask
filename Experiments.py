from rdkit import Chem
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import RDLogger 
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from utils import *
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import from_smiles
from dect.directions import generate_uniform_directions 
from dect.ect import compute_ect_edges
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import AttentiveFP

import time
import questionary
from questionary import Style

RDLogger.DisableLog('rdApp.*')   
RDLogger.DisableLog('rdkit.*')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_list = [
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

def load_molecule_dataset(file_path, dataset_name):
    data = Chem.SDMolSupplier(file_path)
    df = pd.DataFrame()
    for i, mol in tqdm(enumerate(data)):
        try:
            smiles = Chem.MolToSmiles(mol)
            if dataset_name in ["ADRA1A", "ALOX5AP", "ATR", "JAK1", "KOR", "MUSC1", "MUSC2"]:
                target = np.log(float(mol.GetProp("Ki (nM)").replace(">", "").replace("<", "")))
            elif dataset_name in ["DPP4","JAK2"]:
                target = float(mol.GetProp("target").replace(">", "").replace("<", ""))

        except Exception as e:
            smiles = 'invalid'
            target = np.nan

        df.loc[i, "smiles"] = smiles
        df.loc[i, "target"] = target

    mask = [True if Chem.MolFromSmiles(m) != None else False for m in df.loc[:, "smiles"]]
    df = df[mask].reset_index(drop=True)
    df["dataset"] = dataset_name
    df = df.reindex(['dataset', 'smiles', 'target'], axis=1)
    return df


def remove_duplicates_and_nans(dataset):
    df = dataset.drop_duplicates(subset=['smiles'], keep='first')
    df = df.dropna(ignore_index=True)
    print(f"This dataset consist of {dataset.shape[0]} molecular examples, where {df.shape[0]} samples are unique and without nan values")
    return df

def save_molecule_dataset_as_csv(df, nameDataset):
    try:
        os.stat(f"data/datasetCSV/")
    except:
        os.mkdir(f"data/datasetCSV/")
        
    df.to_csv(f'data/datasetCSV/{nameDataset}.csv', index=False)


def join_datasets_in_single_one(folder_path):
    datasets = [
        f for f in glob.glob(os.path.join(folder_path, "*.csv"))
        if "GeneralDataset" not in os.path.basename(f)
    ]

    final_df = pd.concat((pd.read_csv(f) for f in datasets), ignore_index=True)
    save_path = os.path.join(folder_path, "GeneralDataset.csv")
    final_df.to_csv(save_path, index=False)
    print(f"Dataset GeneralDataset with all datasets in one saved as CSV at {save_path} .")
    print(f"This dataset consist of {final_df.shape[0]} molecular examples, where {final_df['smiles'].nunique()} samples (obviously all) are unique (some molecule samples appears in different datasets)")

def exploratory_analysis_histogram_target(dataset,namesDataset):

    if isinstance(namesDataset, list) and len(namesDataset) > 1:

        try:
            os.stat(f"figures/AllDataset/")
        except:
            os.mkdir(f"figures/AllDataset/")

        fig, axes = plt.subplots(3,3, figsize=(20,40))
        for i, ax in enumerate(axes.flat):
            sns.histplot(data=dataset[dataset["dataset"]==namesDataset[i]], x="target", y = None, hue=None, ax=ax)
            ax.set_xlabel("Inhibition Constant ($K_i$) Values")
            ax.set_ylabel("Count")
            ax.set_title(f"Histogram of target feature for {namesDataset[i]} dataset")

            if i == 3*3-1:
                plt.savefig("figures/AllDataset/TargetFeatureHistogramPerDataset")
                plt.close()
                plt.Figure(figsize=(5,5))
                sns.histplot(data=dataset, x='target', y=None, hue=None)
                plt.xlabel('Inhibition Constant ($K_i$) Values')
                plt.ylabel('Count')
                plt.title("Histogram of target feature for the whole dataset")
                plt.savefig(f"figures/AllDataset/TargetFeatureHistogramWholeDataset")
                plt.close()
    else:
        try:
            os.stat(f"figures/{namesDataset}/")
        except:
            os.mkdir(f"figures/{namesDataset}/")
        plt.Figure(figsize=(5,5))
        sns.histplot(data=dataset, x='target', y=None, hue=None)
        plt.xlabel('Inhibition Constant ($K_i$) Values')
        plt.ylabel('Count')
        plt.title(f"Histogram of target feature for the {namesDataset} dataset")
        plt.savefig(f"figures/{namesDataset}/TargetFeatureHistogram_{namesDataset}")
        plt.close()

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
    {"name": "JAK1", "value": 8},
    {"name": "JAK2", "value": 9},
    {"name": "KOR", "value": 10},
    {"name": "MUSC1", "value": 11},
    {"name": "MUSC2", "value": 112},
    
]

def show_menu():
    choice = questionary.select(
        "\n ",
        choices=menu_options,
        style=custom_style,
        qmark="",
        pointer="→",
        instruction="For which dataset do you want to reproduce experiments? Please, click enter to select :) \n",
        use_indicator=True
    ).ask()
    return choice


def compute_molecule_graphECT_from_smiles(dfDataset):
    graph_list = []
    num_thetas=158
    resolutions=16
    v = generate_uniform_directions(num_thetas=num_thetas,d=9,seed=0,device='cpu') #device cpu or cuda as in torch
    for i, smile in tqdm(enumerate(dfDataset['smiles']), total=len(dfDataset), leave=True, desc="Computing ECT"):
        g = from_smiles(smile)
        g.x = g.x.float()
        ect = compute_ect_edges(
            g.x,
            g.edge_index,
            v=v,
            radius=1,
            resolution=resolutions,
            scale=500)
        g.ect=ect
        y = torch.tensor(dfDataset['target'][i], dtype=torch.float).view(1, -1)
        g.y = y
        graph_list.append(g)
    return graph_list

def compute_graph_from_smiles(dfDataset):
    graph_list = []
    for i, smile in enumerate(dfDataset['smiles']):
        g = from_smiles(smile)
        g.x = g.x.float()
        y = torch.tensor(dfDataset['target'][i], dtype=torch.float).view(1, -1)
        g.y = y
        graph_list.append(g)
    return graph_list

def run_ect_method(dfDataset, nameDataset):
    print(f"\nMethod: ECT")
    graph_list = compute_molecule_graphECT_from_smiles(dfDataset)
    X=[]
    y=[]
    for i in range(len(graph_list)):
        X.append(graph_list[i].ect.detach().squeeze().numpy().T.flatten())
        y.append(dfDataset['target'][i])
    scaler = StandardScaler()
    scaler.fit(X)
    Xscaled = scaler.transform(X)
    xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=5,
            random_state=42
        )
    scoring = {
        'rmse': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': 'r2',
        'mae': 'neg_mean_absolute_error'
    }
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    results = cross_validate(xgb_model, Xscaled, y, scoring=scoring, cv=cv, return_train_score=True)
    results_rmse = print_cv_results("RMSE", results['train_rmse'], results['test_rmse'])
    results_r2 = print_cv_results("R2", results['train_r2'], results['test_r2'])
    results_mae = print_cv_results("MAE", -results['train_mae'], -results['test_mae']) # as it comes in negative way.
    results_time = print_cv_results("Fit_time", results['fit_time'], results['score_time'])
    df_results = pd.concat([results_rmse, results_r2, results_mae, results_time], ignore_index=True)

    try:
        os.stat(f"results/{nameDataset}/")
    except:
        os.mkdir(f"results/{nameDataset}/")

    df_results.to_csv(f'results/{nameDataset}/ECT.csv', index=False)
    print("Done for ECT method")

def run_ect_plus_fingerprint_method(dfDataset, nameDataset,fingerprintName):
    print(f"\nMethod: ECT+Fingerprint")
    if nameDataset == "AllDataset":
        dfsFingerprint=[]
        for name in dataset_list:
            dfF = pd.read_pickle(f"data/features/{name}/{name}_{fingerprintName}.pkl")
            dfF = dfF.drop_duplicates(subset=['smiles'], keep='first')
            dfF = dfF.drop("target", axis=1)
            dfF["dataset"]=name
            dfsFingerprint.append(dfF)
        dfFTotal = pd.concat(dfsFingerprint,ignore_index=True)
        dfFinal = pd.merge(dfDataset, dfFTotal, how="left", on=["smiles","dataset"])
        dfFinal = dfFinal.dropna(ignore_index=True)

    elif nameDataset in dataset_list: 
        dfFingerprint = pd.read_pickle(f"data/features/{nameDataset}/{nameDataset}_{fingerprintName}.pkl")
        dfFingerprint = dfFingerprint.drop_duplicates(subset=['smiles'], keep='first')
        dfFingerprint = dfFingerprint.drop("target", axis=1)
        dfFinal = pd.merge(dfDataset, dfFingerprint, how="left", on="smiles")

    graph_list = compute_molecule_graphECT_from_smiles(dfFinal)
    indices = np.arange(3,dfFinal.shape[1],1)
    Xfingerprint=dfFinal.iloc[:,indices].values
    Xect=[]
    y=[]
    for i in range(len(graph_list)):
        Xect.append(graph_list[i].ect.detach().squeeze().numpy().T.flatten())
        y.append(dfFinal['target'][i])

    scaler = StandardScaler()
    scaler.fit(Xect)
    Xectscaled = scaler.transform(Xect)
    X=np.concatenate((Xectscaled,Xfingerprint),axis=1)

    xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=5,
            random_state=42
        )
    scoring = {
        'rmse': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': 'r2',
        'mae': 'neg_mean_absolute_error'
    }
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    results = cross_validate(xgb_model, X, y, scoring=scoring, cv=cv, return_train_score=True)
    results_rmse = print_cv_results("RMSE", results['train_rmse'], results['test_rmse'])
    results_r2 = print_cv_results("R2", results['train_r2'], results['test_r2'])
    results_mae = print_cv_results("MAE", -results['train_mae'], -results['test_mae']) # as it comes in negative way.
    results_time = print_cv_results("Fit_time", results['fit_time'], results['score_time'])
    df_results = pd.concat([results_rmse, results_r2, results_mae, results_time], ignore_index=True)

    try:
        os.stat(f"results/{nameDataset}/")
    except:
        os.mkdir(f"results/{nameDataset}/")

    df_results.to_csv(f'results/{nameDataset}/ECT+Fingerprint.csv', index=False)
    print("Done for ECT + Fingerprint method")

def run_fingerprint_method(dfDataset, nameDataset, nameFingerprint):
    print(f"\nMethod: Fingerprint {nameFingerprint}")
    if nameDataset == "AllDataset":
        dfsFingerprint=[]
        for name in dataset_list:
            dfF = pd.read_pickle(f"data/features/{name}/{name}_{nameFingerprint}.pkl")
            dfF = dfF.drop_duplicates(subset=['smiles'], keep='first')
            dfF = dfF.drop("target", axis=1)
            dfF["dataset"]=name
            dfsFingerprint.append(dfF)
        dfFTotal = pd.concat(dfsFingerprint,ignore_index=True)
        df = pd.merge(dfDataset, dfFTotal, how="left", on=["smiles","dataset"])
        df = df.dropna(ignore_index=True)
        indices = np.arange(3,df.shape[1],1)
        X = df.iloc[:,indices].values
        y = df["target"].values

    elif nameDataset in dataset_list: 
        fingerprint_path = f"data/features/{nameDataset}"
        dfFingerprint = pd.read_pickle(os.path.join(fingerprint_path,f"{nameDataset}_{nameFingerprint}.pkl"))
        dfFingerprint = dfFingerprint.drop_duplicates(subset=['smiles'], keep='first')
        dfFingerprint = dfFingerprint.drop("target", axis=1)
        df = pd.merge(dfDataset, dfFingerprint, how="left", on="smiles")
        indices = np.arange(3,df.shape[1],1)
        X = df.iloc[:,indices].values
        y = df["target"].values

    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        random_state=42
    )
    scoring = {
        'rmse': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': 'r2',
        'mae': 'neg_mean_absolute_error'
    }
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    results = cross_validate(xgb_model, X, y, scoring=scoring, cv=cv, return_train_score=True)
    results_rmse = print_cv_results("RMSE", results['train_rmse'], results['test_rmse'])
    results_r2 = print_cv_results("R2", results['train_r2'], results['test_r2'])
    results_mae = print_cv_results("MAE", -results['train_mae'], -results['test_mae']) # as it comes in negative way.
    results_time = print_cv_results("Fit_time", results['fit_time'], results['score_time'])
    df_results = pd.concat([results_rmse, results_r2, results_mae, results_time], ignore_index=True)

    try:
        os.stat(f"results/{nameDataset}/")
    except:
        os.mkdir(f"results/{nameDataset}/")

    df_results.to_csv(f'results/{nameDataset}/{nameFingerprint}.csv', index=False)
    print(f"Done for fingerprint {nameFingerprint}")

def run_descriptor_method(dfDataset, nameDataset, descriptor):
    print(f"\n Method: Descriptor {descriptor}")

    if nameDataset == "AllDataset":
        dfsDescriptor=[]
        for name in dataset_list:
            dfD = pd.read_pickle(f"data/features/{name}/{name}_{descriptor}.pkl")
            dfD = dfD.drop_duplicates(subset=['smiles'], keep='first')
            dfD = dfD.drop("target", axis=1)
            dfD["dataset"]=name
            dfsDescriptor.append(dfD)
        dfDTotal = pd.concat(dfsDescriptor,ignore_index=True)
        df = pd.merge(dfDataset, dfDTotal, how="left", on=["smiles","dataset"])
        indices = np.arange(3,df.shape[1],1)
        X = df.iloc[:,indices].values
        y = df["target"].values
    elif nameDataset in dataset_list:
        descriptor_path = f"data/features/{nameDataset}"
        dfDescriptor = pd.read_pickle(os.path.join(descriptor_path,f"{nameDataset}_{descriptor}.pkl"))
        dfDescriptor = dfDescriptor.drop_duplicates(subset=['smiles'], keep='first')
        dfDescriptor = dfDescriptor.drop("target", axis=1)
        df = pd.merge(dfDataset, dfDescriptor, how="left", on="smiles")
        indices = np.arange(3,df.shape[1],1)
        X = df.iloc[:,indices].values
        y = df["target"].values

    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        random_state=42
    )
    scoring = {
        'rmse': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': 'r2',
        'mae': 'neg_mean_absolute_error'
    }
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    results = cross_validate(xgb_model, X, y, scoring=scoring, cv=cv, return_train_score=True)
    results_rmse = print_cv_results("RMSE", results['train_rmse'], results['test_rmse'])
    results_r2 = print_cv_results("R2", results['train_r2'], results['test_r2'])
    results_mae = print_cv_results("MAE", -results['train_mae'], -results['test_mae']) # as it comes in negative way.
    results_time = print_cv_results("Fit_time", results['fit_time'], results['score_time'])
    df_results = pd.concat([results_rmse, results_r2, results_mae, results_time], ignore_index=True)

    try:
        os.stat(f"results/{nameDataset}/")
    except:
        os.mkdir(f"results/{nameDataset}/")

    df_results.to_csv(f'results/{nameDataset}/{descriptor}.csv', index=False)
    print(f"Done for descriptor {descriptor}")

def run_gnn_method(graph_list, nameDataset, gnn):
    print(f"\nMethod: GNN {gnn}") 

    k_folds = 10
    cv = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    rmse_scores_train, r2_scores_train, mae_scores_train, fit_time = [], [], [], []
    rmse_scores_test, r2_scores_test, mae_scores_test, score_time = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(graph_list)):

        # Datasets por fold
        train_subset = Subset(graph_list, train_idx)
        test_subset = Subset(graph_list, test_idx)

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

        if gnn == "GAT":
            model = GAT(hidden_channels=8, heads=8).to(device)
        elif gnn == "GCN":
            model = GCN(hidden_channels=64).to(device)
        elif gnn == "AttentiveFP":
            model = AttentiveFP(in_channels=9, hidden_channels=64, out_channels=1,
            edge_dim=3, num_layers=4, num_timesteps=2,
            dropout=0.2).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=10**-2.5,
                                    weight_decay=10**-5)
        initTrainTime = time.time()
        epochs = 100
        for epoch in range(epochs):
            y_train_true, y_train_pred = train_model(model, train_loader, optimizer, device, gnn)
        endTrainTime = time.time()
        fit_time.append(endTrainTime - initTrainTime)

        initScoreTime = time.time()
        y_test_true, y_test_pred = evaluate_model(model, test_loader, device, gnn)
        endScoreTime = time.time()
        score_time.append(endScoreTime - initScoreTime)

        # Métricas
        rmse_scores_train.append(np.sqrt(mean_squared_error(y_train_true, y_train_pred)))
        r2_scores_train.append(r2_score(y_train_true, y_train_pred))
        mae_scores_train.append(mean_absolute_error(y_train_true, y_train_pred))

        rmse_scores_test.append(np.sqrt(mean_squared_error(y_test_true, y_test_pred)))
        r2_scores_test.append(r2_score(y_test_true, y_test_pred))
        mae_scores_test.append(mean_absolute_error(y_test_true, y_test_pred))
    try:
        os.stat(f"results/{nameDataset}/")
    except:
        os.mkdir(f"results/{nameDataset}/")

    results_rmse = print_cv_results("RMSE", rmse_scores_train, rmse_scores_test)
    results_r2 = print_cv_results("R2", r2_scores_train, r2_scores_test)
    results_mae = print_cv_results("MAE", mae_scores_train, mae_scores_test)
    results_time = print_cv_results("Fit_time", fit_time, score_time)
    df_results = pd.concat([results_rmse, results_r2, results_mae, results_time], ignore_index=True)
    df_results.to_csv(f'results/{nameDataset}/{gnn}.csv', index=False)
    print(f"Done for Graph Neural Network {gnn}")
    
def run_single_dataset_pipeline(datasetName):
    datasetPath = "data/datasetCSV"
    file_path = f"data/datasetRaw/{datasetName}.sdf"
    df = load_molecule_dataset(file_path, datasetName)
    df = remove_duplicates_and_nans(df)
    save_molecule_dataset_as_csv(df, datasetName)
    print(f"Dataset {datasetName} saved as CSV at data/datasetCSV/{datasetName}.csv .")

    dfDataset = pd.read_csv(os.path.join(datasetPath,f"{datasetName}.csv"))
    print(f"\n--Performing a brief exploratory analysis, figures saved at figures/{datasetName}/")
    exploratory_analysis_histogram_target(df, datasetName)

    run_ect_method(dfDataset, datasetName)

    run_ect_plus_fingerprint_method(dfDataset, datasetName, "Avalon")

    for enum, fingerprint in enumerate(fingerprint_list):
        run_fingerprint_method(dfDataset, datasetName, fingerprint)
    print("\nDone for all fingerprint methods")

    for enum, descriptor in enumerate(descriptor_list):
        run_descriptor_method(dfDataset, datasetName, descriptor)
    print("\nDone for all descriptor methods")

    graph_list = compute_graph_from_smiles(dfDataset)
    for enum, gnn in enumerate(gnns_list):
        run_gnn_method(graph_list, datasetName, gnn)
    print("\nDone for all GNNs methods")
    print(f"Experiments done for {datasetName} dataset. Results can be found at results/{datasetName}/. If you want to visualize the results of different methods, please run comparison.py script.")

def run_all_dataset_pipeline():
    for name in dataset_list:
        print("\n-- Dataset: ", name, " --")
        file_path = f"data/datasetRaw/{name}.sdf"
        df = load_molecule_dataset(file_path, name)
        df = remove_duplicates_and_nans(df)
        save_molecule_dataset_as_csv(df, name)
        print(f"Dataset {name} saved as CSV at data/datasetCSV/{name}.csv .")

    print(" \n-- Dataset Global: --")
    
    folder_path = "data/datasetCSV"
    join_datasets_in_single_one(folder_path)

    DatasetGeneralPath=os.path.join(folder_path, "GeneralDataset.csv")
    dfGeneral = pd.read_csv(DatasetGeneralPath)
    print("\n--Performing a brief exploratory analysis, figures saved in folder figures/AllDataset/")
    exploratory_analysis_histogram_target(dfGeneral, dataset_list)

    run_ect_method(dfGeneral, "AllDataset")

    run_ect_plus_fingerprint_method(dfGeneral, "AllDataset", "Avalon")

    for enum, fingerprint in enumerate(fingerprint_list):
        run_fingerprint_method(dfGeneral, "AllDataset", fingerprint)
    print("\nDone for all fingerprint methods")

    for enum, descriptor in enumerate(descriptor_list):
        run_descriptor_method(dfGeneral, "AllDataset", descriptor)
    print("\nDone for all descriptor methods")

    graph_list = compute_graph_from_smiles(dfGeneral)
    for enum, gnn in enumerate(gnns_list):
        run_gnn_method(graph_list, "AllDataset", gnn)
    print("\nDone for all GNNs methods")
    print(f"Experiments done for all datasets together. Results can be found at results/AllDataset/. If you want to visualize the results of different methods, please run comparison.py script.")   


def main():
    try:
        choice = show_menu()
    except Exception as e:
        print(f"\n    [X] Error inesperado: {e}\n")
    

    if choice == 1:
        run_all_dataset_pipeline()


    elif choice == 2:
        run_all_dataset_pipeline()
        for dataset in dataset_list:
            run_single_dataset_pipeline(dataset)

    elif choice == 3:
        for dataset in dataset_list:
            print(f"\nRunning full pipeline for dataset: {dataset}")
            run_single_dataset_pipeline(dataset)


    elif choice > 3:
        value = choice - 4
        SelectedDataset = dataset_list[value]
        print(f"Running full pipeline for dataset: {SelectedDataset}")
        run_single_dataset_pipeline(SelectedDataset)

if __name__ == "__main__":
    main()