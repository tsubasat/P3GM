nohup taskset -c 0,1,2,3 python src/synthesize.py --lot_size 100 --sgd_sigma 1.9 --sgd_epoch 15 --pca_sigma 100 --gmm_sigma 1000 --db credit --alg p3gm --n_iter 20
taskset -c 0,1,2,3 python src/synthesize.py --lot_size 100 --sgd_sigma 1.9 --sgd_epoch 15 --pca_sigma 30 --db credit --alg p3gm --n_iter 20
taskset -c 0,1,2,3 python src/synthesize.py --lot_size 100 --sgd_sigma 2 --sgd_epoch 15 --pca_sigma 15 --db credit --alg p3gm --n_iter 20
taskset -c 0,1,2,3 python src/synthesize.py --lot_size 100 --sgd_sigma 2.3 --sgd_epoch 15 --pca_sigma 7 --db credit --alg p3gm --n_iter 20
taskset -c 0,1,2,3 python src/synthesize.py --lot_size 100 --sgd_sigma 3.5 --sgd_epoch 15 --pca_sigma 5 --db credit --alg p3gm --n_iter 20
taskset -c 0,1,2,3 python src/synthesize.py --lot_size 100 --sgd_sigma 5.2 --sgd_epoch 15 --pca_sigma 4.5 --db credit --alg p3gm --n_iter 20
taskset -c 0,1,2,3 python src/synthesize.py --lot_size 100 --sgd_sigma 5.2 --sgd_epoch 0 --pca_sigma 4.3 --db credit --alg p3gm --n_iter 20