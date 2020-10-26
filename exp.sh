python data_process.py

python src/synthesize.py --db mnist --lot_size 240 --epoch 10 --gmm_sigma 120 --latent_dim 1000 --sgd_sigma 1.32
python ml_task/exp_ml.py --db mnist

python src/synthesize.py --db fashion --lot_size 240 --epoch 10 --gmm_sigma 120 --latent_dim 1000 --sgd_sigma 1.32
python ml_task/exp_ml.py --db fashion

python src/synthesize.py --db adult --epoch 5 --lot_size 300 --sgd_sigma 1.35
python ml_task/exp_ml.py --db adult

python src/synthesize.py --db isolet --epoch 5 --lot_size 50 --sgd_sigma 1.35 --latent_dim 1000
python ml_task/exp_ml.py --db isolet

python src/synthesize.py --db esr --epoch 2 --latent_dim 1000 --z_dim 5 --lot_size 100
python ml_task/exp_ml.py --db esr

python src/synthesize.py --db credit --z_dim 31 --lot_size 2000 --epoch 10 --gmm_sigma 120 --latent_dim 1000 --sgd_sigma 1.83
python ml_task/exp_ml.py --db credit
