# ECTforMoleculeMachineLearningTask
This repository contains data and experiments associated to the paper Toscano.Duran, V., Rieck, B., and Rottach, F. (2025). "Molecular Machine Learning Using Euler Characteristic Transforms". Submitted and accepted at ["I Congreso de la Sociedad Española de Inteligencia Artificial en Biomedicina (CIABiomed)"](https://2025.iabiomed.org/). Preprint available in [arXiV](https://arxiv.org/abs/2507.03474).

The shape of a molecule determines its physicochemical and biological properties. However, it is often underrepresented in standard molecular representation learning approaches. Here, we propose using the Euler Characteristic Transform (ECT) as a geometrical-topological  descriptor. Computed directly from molecular graphs constructed usinghandcrafted atomic features, the ECT enables the extraction of multiscale structural features,  offering a novel way to encode molecular shape in the feature space. We assess the predictive performance of this representation across nine benchmark regression datasets, all centered around predicting the inhibition constant K_i. In addition, we compare our proposed ECT-based descriptor against traditional molecular representations and methods, such as molecular fingerprints/descriptors  and graph neural networks (GNNs). Our results show that our ECT-based representation achieves competitive performance, ranking among the best-performing methods on several datasets. More importantly, combining our descriptor with established representations, particularly with the AVALON fingerprint, significantly enhances predictive performance, outperforming other methods on most datasets. These findings highlight the complementary value of multiscale topological information and its potential for being combined with established techniques. Our study suggests that hybrid approaches incorporating explicit shape information can lead to more informative and robust molecular representations, enhancing and opening new avenues in molecular machine learning. To support reproducibility and foster open biomedical research, we provide open access to all experiments and code used in this work.



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

- ECTParameterAnalysis.ipynb: This notebook contains the preliminary analyses performed to determine the number of directions and thresholds of the ECT in relation to the main experiments and results presented in the paper.

- FiguresIllustration.ipynb: This notebook contains the code for generate some figures that serves as illustration in the paper.

## Citation and reference

If you want to use our code or data for your experiments, please cite our paper (will be updated as soon as is published in CIABiomed proceedings):

V. Toscano-Duran, F. Rottach, and B. Rieck, “Molecular machine learning using euler characteristic transforms,” arXiv preprint arXiv:2507.03474, Jul. 2025. DOI: 10.48550/arXiv.2507.03474.

For further information, please contact us at: vtoscano@us.es, bastian.grossenbacher@unifr.ch and florian.rottach@boehringer-ingelheim.com
