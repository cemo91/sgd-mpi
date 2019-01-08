valgrind -v --tool=memcheck --leak-check=full ./sgd --dataset data/ionosphere --sparse false --batch-size 1 --epochs 20 --rate decay --eta0 0.1 --loss hinge --regularizer l2 --lambda 0.1
