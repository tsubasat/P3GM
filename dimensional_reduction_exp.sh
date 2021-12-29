taskset -c 0,1,2,3 python src/synthesize.py --lot_size 300 --sgd_sigma 1.1 --sgd_epoch 3 --pca_sigma 10 --db mnist --alg p3gm --n_iter 20 --z_dim 2
taskset -c 0,1,2,3 python src/synthesize.py --lot_size 300 --sgd_sigma 1.1 --sgd_epoch 3 --pca_sigma 10 --db mnist --alg p3gm --n_iter 20 --z_dim 5
taskset -c 0,1,2,3 python src/synthesize.py --lot_size 300 --sgd_sigma 1.1 --sgd_epoch 3 --pca_sigma 10 --db mnist --alg p3gm --n_iter 20 --z_dim 10
taskset -c 0,1,2,3 python src/synthesize.py --lot_size 300 --sgd_sigma 1.1 --sgd_epoch 3 --pca_sigma 10 --db mnist --alg p3gm --n_iter 20 --z_dim 50
taskset -c 0,1,2,3 python src/synthesize.py --lot_size 300 --sgd_sigma 1.1 --sgd_epoch 3 --pca_sigma 10 --db mnist --alg p3gm --n_iter 20 --z_dim 100
taskset -c 0,1,2,3 python src/synthesize.py --lot_size 300 --sgd_sigma 1.1 --sgd_epoch 3 --pca_sigma 10 --db mnist --alg p3gm --n_iter 20 --z_dim 500