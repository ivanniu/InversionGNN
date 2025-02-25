# InversionGNN: Enhancing Multi-Property Molecular Optimization for Drug Discovery 


##  1. Sythestic Task

### 1.1 Pareto Optimization Conditioned on Desired Weight
```bash
python compare_solvers.py
```
### 1.2 Exploring Full Pareto Front
Set `is_track = True` in the script.
```bash
python compare_solvers.py
```

##  2. Molecular
###  2.1 Installation

```bash
conda create -n InversionGNN python=3.7 
conda activate InversionGNN
pip install torch 
pip install PyTDC 
conda install -c rdkit rdkit 
```


<a name="data"></a>
###  2.2 Data processing
```bash
python src/download.py
python src/vocabulary.py
python src/clean.py
python src/labelling.py
```


### 2.3 Run
 
#### train graph neural network

```bash 
python src/train.py 
```
#### molecule optimization
```bash
python src/denovo.py
```
