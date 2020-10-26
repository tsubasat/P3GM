## About P3GM
P3GM is a differentially private generative model. 
The algorithm of P3GM is described in our paper (https://arxiv.org/abs/2006.12101). 

## How to Run
This code is implemented by python3.7.9.

Install P3GM.
```
git clone https://github.com/tkgsn/P3GM
cd P3GM
```


Install tensorflow_privacy library.
```
git clone https://github.com/tensorflow/privacy.git
```


Make a virtual enviroment and install the needed dependencies.
```
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

Download the datasets.

`./download_dataset.sh`

Run the experiments.

`./exp.sh`

## Datasets

Kaggle Credit Dal Pozzolo, Andrea, et al. "Calibrating probability with undersampling for unbalanced
classification." 2015 IEEE Symposium Series on Computational Intelligence. IEEE, 2015.

Isolet: https://archive.ics.uci.edu/ml/datasets/isolet

ESR: https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition

Adult: https://archive.ics.uci.edu/ml/datasets/adult
