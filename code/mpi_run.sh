mpirun -np 2 ./mpi_sgd --dataset data/ionosphere --sparse false --batch-size 1 --epochs 100 --rate decay --eta0 1 --loss hinge --regularizer l2 --lambda 0.1
