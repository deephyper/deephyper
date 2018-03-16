## Feature selection utility scripts

Python functions and command line utilities for building classic and ensemble machine learning model sweeps on by-drug and by-cellline models of the Pilot1 drug response data.

```
git clone https://github.com/levinas/p1h
pip install -U pandas scikit-learn xgboost
```

### Command-line Examples

Scripts exist for dataframe export and prediciton tasks.

#### Export by-drug data
Save molecular features and dose reponse data for given drugs to CSV files:
```
$ python dataframe.py --by drug --drugs 100071 --feature_subsample 10

NSC 100071: saved 52 rows and 11 columns to NSC_100071.csv
```

#### Export by-cell data
Save drug features and dose response data for given cell lines to CSV files:
```
$ python dataframe.py --by cell --cells BR:MCF7 CNS:SF_268

BR:MCF7: saved 15628 rows and 3811 columns to BR:MCF7.csv
CNS:SF_268: saved 28151 rows and 3811 columns to CNS:SF_268.csv
```

#### By-drug regression runs
Run three regression models on drug set A (defined by Jason to include 306 drugs)
using all types of cell line features (expression, miRNA and proteome), and save
feature importance and model performance evaluated on various metrics to files.
```
$ python by_drug.py --drugs A --models randomforest lasso elasticnet --cell_features all
```

### Code Examples

#### Run standard regressions on a drug

```
from datasets import NCI60
from skwrapper import regress

df = NCI60.load_by_drug_data(drug='100071')
regress('XGBoost', df)
regress('Lasso', df)
```

#### Sweep customized RandomForest regression on all cell lines
```
from datasets import NCI60
from skwrapper import regress
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=20)

cells = NCI60.all_cells()
for cell in cells:
    df = NCI60.load_by_cell_data(cell)
    regress(model, df, cv=3)
```	



