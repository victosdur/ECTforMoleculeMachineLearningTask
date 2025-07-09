# EulerCharacteristicTransform_Molecules
This repository contains data and experiments associated to the paper Toscano.Duran, V., Rieck, B., and Rottach, F. (2025). "Molecular Machine Learning Using Euler Characteristic Transforms". Submitted to "I Congreso de la Sociedad Española de Inteligencia Artificial en Biomedicina (CIABiomed)". Preprint available in [arXiV](https://arxiv.org/abs/2507.03474).

In this work, we compare and explore the use of the ECT-based molecular representation, which is computed directly over molecular graphs derived from handcrafted atomic features, to predict $K_i$, a key molecular property, as well as perform a comparison between our approach and traditional methods over a series of nine binding affinity datasets. By using this topological representation, as well as combining it with traditional molecular representations, we aim to enhance predictive performance and provide new insights into the role of molecular shape in molecular learning. Our experiments shows that our ECT-based approach exhibites competitive predictive performance, in some cases even outperforming all alternative methods. In addition, our experiments show that the combination of our ECT-based approach with existing methods, more specifically with the AVALON fingerprint, leads to improved performance, thus highlighting the complementary value of multiscale topological and shape information. Ultimately, our work contributes to the growing body of evidence suggesting that incorporating molecular shape at a fundamental level can lead to more robust and informative models, opening up new avenues for the design of better molecular machine learning and more effective therapies.


## Usage

1) Clone this repository:

```bash
git clone https://github.com/victosdur/ECTforMoleculeLearningTask.git
```

2) Create a virtual environment (it has been developed specifically using Python3.10.11):

```bash
virtualenv -p python env
```

3) Activate the virtual environment:

```bash
env\Scripts\activate
```

4) Install the neccesary dependencies:

#### Option 1

```bash
pip install rdkit pandas matplotlib torch torch_geometric tqdm networkx seaborn xgboost scikit-learn questionary
```

To install the "dect" library:

```bash
pip install git+https://github.com/aidos-lab/dect.git #  require an up-to-date installation of PyTorch, either with (pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126) or without (pip install torch) CUDA support: 
```

Remember to install ipykernel to use the kernel of the created virtual environment:

```bash
pip install ipykernel
```

#### Option 2

If you want to have exactly the same dependencies and the versions which have been used for the experiments, you can installed it using:

```bash
pip install requeriments.txt
```

## Repository structure

- data/: This folder contains the data used in the experiments. It contains the dataset raw files in sdf format (one file for each dataset) in the "datasetRaw.zip" folfer, the dataset in CSV format in the "datasetCSV.zip" folder keeping just the molecules which are unique for each dataset, consisting in the smile of the molecule and the target value ($K_i$). Finally the "features.zip" folder, which contains the different representations used and compared in the experiments in pickle format. You can find one folder per dataset, which contains the differents representations for that dataset.

- results/: This folder contains the results in format csv for each method (one folder for each dataset results, which contains one csv file for each method). Then, it is used in the ComparisonMethods.ipynb

- figures/: This folder contains the figures generated in the experiments (one folder for each dataset experiments), as well as the figures generated in the "FiguresIllustration.ipynb" for the paper.

- requirements.txt: This file contains all the dependencies of the project.

- utils.py: This file contains some functions used in the python scripts for the experiments.

- experiments.py: This file runs the experiments. You can select for which dataset you want to run all methods, or if you want to run it for all datasets together, or for all datasets together and all datasets indepently. Please run it with the following command (remember to activate the virtual environment you have created):

```bash
python experiments.py
```

Then, you have like a menu, in which you can select one of the previous commented options. The results are saved in the folder "results"

- comparison.py: This file generates a boxplot comparison between the different methods from the results obtained according the running of the previous script ("experiments.py" file). You can run it:

```bash
python comparison.py
```
As previously, you have a menu, in which you can select for which dataset you want to generate the comparison plots.

- resultsprinter.py: This file prints in console the mean and standard deviation results for the differents methods and datasets, obtained from the results of "experiments.py" file. It follows the same structure than the two previous scripts.

- FiguresIllustration.ipynb: This notebooks contains the code for generate some figures that serves as illustration in the paper.

## Citation and reference

If you want to use our code or data for your experiments, please cite our paper:

V. Toscano-Duran, F. Rottach, and B. Rieck, “Molecular machine learning using euler characteristic transforms,” arXiv preprint arXiv:2507.03474, Jul. 2025. DOI: 10.48550/arXiv.2507.03474.

For further information, please contact us at: vtoscano@us.es, bastian.grossenbacher@unifr.ch and florian.rottach@boehringer-ingelheim.com
