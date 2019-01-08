#include <mpi.h>
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <stdio.h>
#include "read.hpp"
#include "sgd.hpp"

using namespace std;

// SHOW OPTIONS
static void show_usage(string name){
    cerr << "Usage: " << name << " <option(s)>\n"
              << "Options:\n"
              << "\t-h,--help: Show this help message\n"
              << "\t-d,--dataset PATH_TO_DATASET: Path of the dataset\n"
              << "\t-s,--sparse IS_SPARSE: (bool) Flag for sparse/dense data {true=sparse, false=dense}\n"
              << "\t-b,--batch-size mini_batch_size_PER_MACHINE: (int) Size of the batch for SGD updates\n"
              << "\t-e,--epochs NUM_OF_EPOCHS: (int) Maximum number of epochs\n"
              << "\t-lr,--rate RATE: (string) Learning rate {constant, decay}\n"
              << "\t-et,--eta0 STEP_SIZE: (double) \n" 
              << "\t-l,--loss LOSS: (string) Loss function {hinge, logistic, mse}\n"
              << "\t-r,--regularizer REGULARIZER: (string) Regularizer (none, l1, l2)\n"
              << "\t-lm,--lambda LAMBDA: (double) Lambda value of regularizer" 
              << endl;
}

int main(int argc, char* argv[]){
	int rank, num_procs;
	char rank_s[1];
	MPI_Init(&argc, &argv);
	MPI_Comm_size (MPI_COMM_WORLD, &num_procs);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	sprintf(rank_s, "%d", rank);
	
	// TIMERS FOR MPI COMMUNICATION
	double time_start, time_end;
	double time_allreduce = 0;
	

	// OPTIONS FOR SGD CLASSIFIER
	string file_name = "data/ionosphere";
	bool normalize = false;
    bool is_sparse = false;
	int mini_batch_size = 1;
	int max_epochs = 50;
	string learning_rate = "constant";
	double step_size = 0.1;
	string loss = "hinge";
	string regularizer = "l2";
	double lambda = 0.1;
	

	// PARSE COMMAND LINE INPUTS
	if(argc < 2) {
        show_usage(argv[0]);
        return 1;
    }
    for(int i = 1; i < argc; i++){
        string arg = argv[i];
        if((arg == "-h") || (arg == "--help")){
            show_usage(argv[0]);
            return 0;
        }
        else if((arg == "-d") || (arg == "--dataset")){
            if (i + 1 < argc) {
				string file_name_c(argv[++i]);
				file_name = file_name_c;
            }
            else{
                cerr << "--dataset option requires one argument." << endl;
                return 1;
            }  
        }
        else if((arg == "-s") || (arg == "--sparse")){
            if(i + 1 < argc){
                string sparse(argv[++i]);
                if(sparse.compare("true") == 0){
                    is_sparse = true;
                }
                else if(sparse.compare("false") == 0){
                    is_sparse = false;
                }
                else{
                    cerr << "--sparse option should be either true or false. Set to default (false)." << endl;
                }
            }
            else{
                cerr << "--sparse option requires one argument." << endl;
                return 1;
            }  
        }
        else if((arg == "-b") || (arg == "--batch-size")){
			if (i + 1 < argc) {
				mini_batch_size = atoi(argv[++i]);
            }
            else{
                cerr << "--batch-size option requires one argument." << endl;
                return 1;
            }
			
        }
        else if((arg == "-e") || (arg == "--epochs")){
			if (i + 1 < argc) {
				max_epochs = atoi(argv[++i]);
            }
            else{
                cerr << "--epochs option requires one argument." << endl;
                return 1;
            }
			
        }
        else if((arg == "-lr") || (arg == "--rate")){
			if (i + 1 < argc) {
				string learning_rate_f(argv[++i]);
				learning_rate = learning_rate_f;
            }
            else{
                cerr << "--rate option requires one argument." << endl;
                return 1;
            }
			
        }
        else if((arg == "-et") || (arg == "--eta0")){
			if (i + 1 < argc) {
				step_size = atof(argv[++i]);
            }
            else{
                cerr << "--eta0 option requires one argument." << endl;
                return 1;
            }
			
        }
        else if((arg == "-l") || (arg == "--loss")){
			if (i + 1 < argc) {
				string loss_f(argv[++i]);
				loss = loss_f;
            }
            else{
                cerr << "--loss option requires one argument." << endl;
                return 1;
            }
			
        }
        else if((arg == "-r") || (arg == "--regularizer")){
			if (i + 1 < argc) {
				string regularizer_f(argv[++i]);
				regularizer = regularizer_f;
            }
            else{
                cerr << "--regularizer option requires one argument." << endl;
                return 1;
            }
			
        }
        else if((arg == "-lm") || (arg == "--lambda")){
			if (i + 1 < argc) {
				lambda = atof(argv[++i]);
            }
            else{
                cerr << "--lambda option requires one argument." << endl;
                return 1;
            }
			
        }
    }
	
	int data_size, vector_size;
	vector<Datapoint> data_m;
	string file_n = file_name + ".part" + rank_s; // NAME OF THE FILE FOR EACH MACHINE

	// READ DATA FROM LIBSVM FORMAT FILE, LABELS ARE CONVERTED 0/1 IF LOSS IS LOGISTIC
	int success = process_file(data_m, file_n.c_str(), is_sparse, loss);

	data_size = data_m.size();
	vector_size = data_m[0].n_features;
	
	
	// INITIALIZE WEIGHT VECTOR
	Eigen::VectorXd weight = Eigen::VectorXd::Zero(vector_size);
	
	int max_iters = data_size / mini_batch_size;
	
	vector<int> batch_indices;
	
	// SAVE INITIAL STEP SIZE
	double eta0 = step_size;
	if(is_sparse){
		for(int i = 0; i < max_epochs; i++){
			for(int j = 0; j < max_iters; j++){
				
				// SET STEP SIZE IN EACH ITERATION
				if(learning_rate.compare("decay") == 0){
					step_size = eta0 / (lambda * ((i * max_iters) + (j + 1)));
				}

				// GENERATE A RANDOM BATCH OF SIZE MINI_BATCH_SIZE
				batch_indices = generate_mini_batch(mini_batch_size, data_size);
				

				// UPDATE THE WEIGHTS IN EACH MACHINE
				Eigen::VectorXd update = weight_update_sparse(data_m, batch_indices, weight, loss, step_size, regularizer, lambda);
				
				// START TIMER
				time_start = MPI_Wtime();
				
				// ALLREDUCE THE UPDATES
				MPI_Allreduce(MPI_IN_PLACE, update.data(), vector_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
				
				// END TIMER
				time_end = MPI_Wtime();
				

				// ADD ALLREDUCED UPDATES TO WEIGHT VECTOR
				weight = weight + update;
				
				// ADD TIME TO TOTAL COMMUNICATION
				time_allreduce += (time_end - time_start);
			}
		}
	}
	else{
		for(int i = 0; i < max_epochs; i++){
			for(int j = 0; j < max_iters; j++){
				
				// SET STEP SIZE IN EACH ITERATION
				if(learning_rate.compare("decay") == 0){
					step_size = eta0 / (lambda * ((i * max_iters) + (j + 1)));
				}

				// GENERATE A RANDOM BATCH OF SIZE MINI_BATCH_SIZE
				batch_indices = generate_mini_batch(mini_batch_size, data_size);
				

				// UPDATE THE WEIGHTS IN EACH MACHINE
				Eigen::VectorXd update = weight_update_dense(data_m, batch_indices, weight, loss, step_size, regularizer, lambda);
				
				// START TIMER
				time_start = MPI_Wtime();
				
				// ALLREDUCE THE UPDATES
				MPI_Allreduce(MPI_IN_PLACE, update.data(), vector_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
				
				// END TIMER
				time_end = MPI_Wtime();
				

				// ADD ALLREDUCED UPDATES TO WEIGHT VECTOR
				weight = weight + update;
				
				// ADD TIME TO TOTAL COMMUNICATION
				time_allreduce += (time_end - time_start);
			}
		}
	}
	

	// PRINT INFORMATION REGARDING FINAL WEIGHT VECTOR
	if(rank == 0){
		vector<Datapoint> data_matrix;
		success = process_file(data_matrix, file_name.c_str(), is_sparse, loss);
		cout << "Norm: " << weight.norm() << endl;
		cout << "Training accuracy: " << compute_accuracy(data_matrix, weight, loss) << endl;
		cout << "Final training loss: " << compute_average_loss(data_matrix, weight, loss, regularizer, lambda) << endl;
		cout << "Time for total communication: " << time_allreduce << endl;
		
	}
	
	MPI_Finalize();
	return 0;
}
