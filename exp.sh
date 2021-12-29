taskset -c 0,1,2,3 python src/synthesize.py --lot_size 100 --sgd_sigma 2.1 --sgd_epoch 15 --pca_sigma 10 --db credit --alg p3gm --n_iter 20
taskset -c 0,1,2,3 python src/synthesize.py --lot_size 100 --sgd_sigma 1.6 --sgd_epoch 2 --pca_sigma 5 --db isolet --alg p3gm --n_iter 20
taskset -c 0,1,2,3 python src/synthesize.py --lot_size 100 --sgd_sigma 1.4 --sgd_epoch 2 --pca_sigma 5 --db esr --alg p3gm --n_iter 20
taskset -c 0,1,2,3 python src/synthesize.py --lot_size 200 --sgd_sigma 1.4 --sgd_epoch 5 --pca_sigma 5 --db adult --alg p3gm --n_iter 20
taskset -c 0,1,2,3 python src/synthesize.py --lot_size 300 --sgd_sigma 1.4 --sgd_epoch 4 --pca_sigma 5 --db fashion --alg p3gm --n_iter 20
taskset -c 0,1,2,3 python src/synthesize.py --lot_size 300 --sgd_sigma 1.4 --sgd_epoch 4 --pca_sigma 5 --db mnist --alg p3gm --n_iter 20