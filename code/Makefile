sgd: main_sgd.cpp datapoint.hpp datapoint.cpp read.hpp read.cpp sgd.hpp sgd.cpp
	g++ -o sgd -I Eigen/ -std=c++11 main_sgd.cpp datapoint.cpp read.cpp sgd.cpp
mpi_sgd: main.cpp datapoint.hpp datapoint.cpp read.hpp read.cpp sgd.hpp sgd.cpp
	mpic++ -I Eigen/ -std=c++11 -o mpi_sgd main.cpp datapoint.cpp read.cpp sgd.cpp
clean: 
	rm sgd
	rm mpi_sgd
