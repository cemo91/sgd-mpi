#include "sgd.hpp"

// GENERATE A RANDOM INTEGER FROM UNIFORM DISTRIBUTION
int generate_index(int data_size){
	static thread_local std::mt19937_64 generator;
    std::uniform_int_distribution<int> distribution(0, data_size-1);
    return distribution(generator);
}

// GENERATE A SET OF RANDOM INDICES OF SIZE MINI_BATCH_SIZE
std::vector<int> generate_mini_batch(int mini_batch_size, int data_size){
	
	std::vector<int> indices;
	indices.resize(mini_batch_size);
	
	for(int i = 0; i < indices.size(); i++){
		indices[i] = generate_index(data_size);
	}
	
	return indices;
}

// COMPUTE THE DOT PRODUCT OF DATAPOINT AND WEIGHT
double dot_product(Datapoint &x, Eigen::VectorXd &w){
	if(x.is_sparse == true){
		return x.dot_product_sparse(w);
	}
	return x.dot_product_dense(w);
}

// COMPUTE THE LOSS OF A SINGLE DATAPOINT
double compute_loss(Datapoint &x, Eigen::VectorXd &weight, std::string loss, std::string regularizer, double lambda){

	// DOT PRODUCT
	double s = dot_product(x,weight);
	double loss_value = 0;
	
	// HINGE LOSS
	if(loss.compare("hinge") == 0){
		loss_value = std::max(0.0, 1 - (x.label * s));
	}

	// MEAN SQUARED ERROR
	else if(loss.compare("mse") == 0){
		loss_value = 0.5 * std::pow((x.label - s), 2);
	}

	// LOGISTIC LOSS
	else if(loss.compare("logistic") == 0){
		loss_value = log(1 + exp(-1 * x.label * s));
	}

	// UNDEFINED LOSS - ERROR
	else{
		std::cout << "Wrong loss" << std::endl;
		return -999; // ERROR
	}
	
	// ADD REGULARIZER EVALUATION TO LOSS
	if(regularizer.compare("none") == 0){
		return loss_value;
	}
	else if(regularizer.compare("l1") == 0){
		return loss_value + (lambda * weight.lpNorm<1>());
	}
	else if(regularizer.compare("l2") == 0){
		return loss_value + (weight.squaredNorm() * (lambda / 2));
	}
	else{
		std::cout << "Wrong regularizer" << std::endl;
		return -999; //ERROR
	}
}

// COMPUTE THE AVERAGE LOSS 
double compute_average_loss(std::vector<Datapoint> &data_matrix, Eigen::VectorXd &weight, std::string loss, std::string regularizer, double lambda){

	// SUM UP THE LOSSES FROM INDIVIDUAL POINTS
	double total_loss  = 0;
	for(int i = 0; i < data_matrix.size(); i++){
		total_loss += compute_loss(data_matrix[i], weight, loss, regularizer, lambda);
	}
	// COMPUTE THE AVERAGE
	return (total_loss / data_matrix.size());
}

// COMPUTE THE ACCURACY 
double compute_accuracy(std::vector<Datapoint> &data_matrix, Eigen::VectorXd &weight, std::string loss){
	int count = 0;
	
	if(loss.compare("hinge") == 0){
		for(int i = 0; i < data_matrix.size(); i++){
			double y_hat = dot_product(data_matrix[i], weight);
			
			if((y_hat < 0) && (data_matrix[i].label == -1)){
				count++;
			}
			else if((y_hat > 0) && (data_matrix[i].label == 1)){
				count++;
			}
		}
	}
	else if(loss.compare("logistic") == 0){
		for(int i = 0; i < data_matrix.size(); i++){
			double y_hat = dot_product(data_matrix[i], weight);
			
			if((y_hat < 0.5) && (data_matrix[i].label == 0)){
				count++;
			}
			else if((y_hat >= 0.5) && (data_matrix[i].label == 1)){
				count++;
			}
		}
	}
	
	return 100 * ((double)count / (double)data_matrix.size());
}

// COMPUTE THE DERIVATIVE OF THE LOSS FOR A DOT PRODUCT (OF A DATAPOINT AND A WEIGHT VECTOR)
double loss_derivative(double s, double label, std::string loss){
	if(loss.compare("hinge") == 0){
		if((label * s) < 1){
			return -1 * label;
		}
		else{
			return 0;
		}
	}
	else if(loss.compare("logistic") == 0){
		return ((-1 * label) / (1 + exp(label * s)));
	}
	else if(loss.compare("mse") == 0){
		return (s - label);
	}
	else{
		// ERROR
		std::cout << "Wrong loss" << std::endl;
		return -999;
	}
}

// COMPUTE THE DERIVATIVE OF THE REGULARIZER
Eigen::VectorXd compute_regularizer_gradient(std::string regularizer, Eigen::VectorXd &weight, double lambda){
	if(regularizer.compare("l1") == 0){
		
		Eigen::VectorXd result(weight.size());
		
		for(int i = 0; i < weight.size(); i++){
			if(weight(i) < 0){
				result(i) = lambda * -1;
			}
			else if(weight(i) > 0){
				result(i) = lambda;
			}
			else{
				result(i) = 0;
			}
		}
		return result;
	}
	else if(regularizer.compare("l2") == 0){
		return weight * lambda;
	}
	else{
		// ERROR
		std::cout << "Wrong regularizer" << std::endl;
		Eigen::VectorXd result(weight.size());
		return result;
	}
}

// COMPUTE THE AVERAGE STOCHASTIC GRADIENT OF MINI BATCH, BY SUMMING UP EACH STOCHASTIC GRADIENT OF DATAPOINTS AND AVERAGING
Eigen::VectorXd compute_gradient_dense(std::vector<Datapoint> &data_matrix, std::vector<int> indices, Eigen::VectorXd &weight, std::string loss, std::string regularizer, double lambda){
	Eigen::VectorXd gradient = Eigen::VectorXd::Zero(weight.size());
	Eigen::VectorXd stoch_gradient;

	double s;
	double derivative;
	
	for(int i = 0; i < indices.size(); i++){
		s = data_matrix[indices[i]].dot_product_dense(weight);
		derivative = loss_derivative(s, data_matrix[indices[i]].label, loss);
		stoch_gradient = data_matrix[indices[i]].data * derivative;
		gradient = gradient + stoch_gradient;
	}
	
	gradient = gradient * (1.0 / (double)indices.size());
	
	if(regularizer.compare("none") != 0){
		Eigen::VectorXd reg_gradient;
		reg_gradient = compute_regularizer_gradient(regularizer, weight, lambda);
		gradient = gradient + reg_gradient;
	}
	return gradient;
}

// COMPUTE THE AVERAGE STOCHASTIC GRADIENT OF MINI BATCH, BY SUMMING UP EACH STOCHASTIC GRADIENT OF DATAPOINTS AND AVERAGING
Eigen::VectorXd compute_gradient_sparse(std::vector<Datapoint> &data_matrix, std::vector<int> indices, Eigen::VectorXd &weight, std::string loss, std::string regularizer, double lambda){
	Eigen::VectorXd gradient = Eigen::VectorXd::Zero(weight.size());
	Eigen::VectorXd stoch_gradient;

	double s;
	double derivative;
	
	for(int i = 0; i < indices.size(); i++){
		s = data_matrix[indices[i]].dot_product_sparse(weight);
		derivative = loss_derivative(s, data_matrix[indices[i]].label, loss);
		stoch_gradient = data_matrix[indices[i]].sp_data * derivative;
		gradient = gradient + stoch_gradient;
	}
	
	gradient = gradient * (1.0 / (double)indices.size());
	
	if(regularizer.compare("none") != 0){
		Eigen::VectorXd reg_gradient;
		reg_gradient = compute_regularizer_gradient(regularizer, weight, lambda);
		gradient = gradient + reg_gradient;
	}
	return gradient;
}

// UPDATE WEIGHTS FOR MINI BATCH
Eigen::VectorXd weight_update_dense(std::vector<Datapoint> &data_matrix, std::vector<int> indices, Eigen::VectorXd &weight, std::string loss, double step_size, std::string regularizer, double lambda){
	Eigen::VectorXd gradient;
	gradient = compute_gradient_dense(data_matrix, indices, weight, loss, regularizer, lambda);
	gradient = gradient * (-1 * step_size);
	return gradient;
}

// UPDATE WEIGHTS FOR MINI BATCH
Eigen::VectorXd weight_update_sparse(std::vector<Datapoint> &data_matrix, std::vector<int> indices, Eigen::VectorXd &weight, std::string loss, double step_size, std::string regularizer, double lambda){
	Eigen::VectorXd gradient;
	gradient = compute_gradient_sparse(data_matrix, indices, weight, loss, regularizer, lambda);
	gradient = gradient * (-1 * step_size);
	return gradient;
}

double train(std::vector<Datapoint> &data_matrix, Eigen::VectorXd &weight, std::string loss, std::string learning_rate, double step_size, std::string regularizer, double lambda, int mini_batch_size, int max_epochs){
	double training_loss;
	double initial_step_size = step_size;
	weight = Eigen::VectorXd::Zero(weight.size());

	int max_iters = data_matrix.size() / mini_batch_size;
	
	std::vector<int> batch_indices;
	Eigen::VectorXd update;
	
	if(data_matrix[0].is_sparse == true){
		for(int i = 0; i < max_epochs; i++){
			for(int j = 0; j < max_iters; j++){
				if(learning_rate.compare("decay") == 0){
					step_size = initial_step_size / (lambda * ((i * max_iters) + (j + 1)));
				}
				
				batch_indices = generate_mini_batch(mini_batch_size, data_matrix.size());
				update = weight_update_sparse(data_matrix, batch_indices, weight, loss, step_size, regularizer, lambda);
				weight = weight + update;
			}
		}
	}
	else{
		for(int i = 0; i < max_epochs; i++){
			for(int j = 0; j < max_iters; j++){
				if(learning_rate.compare("decay") == 0){
					step_size = initial_step_size / (lambda * ((i * max_iters) + (j + 1)));
				}
				
				batch_indices = generate_mini_batch(mini_batch_size, data_matrix.size());
				update = weight_update_dense(data_matrix, batch_indices, weight, loss, step_size, regularizer, lambda);
				weight = weight + update;
			}
		}
	}

	training_loss = compute_average_loss(data_matrix, weight, loss, regularizer, lambda);
	return training_loss;
}

// CHECK THE GRADIENT COMPUTATION IF IT IS CORRECT, WITH A FIRST ORDER APPROXIMATION
void gradient_check(Datapoint &x, std::string loss, std::string regularizer, double lambda){
	
	std::vector<Datapoint> x_v;
	x_v.push_back(x);
	
	std::vector<int> ind;
	ind.push_back(0);
	
	Eigen::VectorXd weight(x.n_features);
	weight.setRandom();
	
	double loss_1 = compute_loss(x, weight, loss, regularizer, lambda);
	Eigen::VectorXd gradient_1;
	if(x.is_sparse == true){
		gradient_1 = compute_gradient_sparse(x_v, ind, weight, loss,regularizer, lambda);
	}
	else{
		gradient_1 = compute_gradient_dense(x_v, ind, weight, loss,regularizer, lambda);
	}
	
	Eigen::VectorXd perturbation = gradient_1 * 0.001;
	weight = weight + perturbation;
	double loss_2 = compute_loss(x, weight, loss, regularizer, lambda);
	std::cout << "old loss + gradient: " << loss_1 + gradient_1.dot(perturbation) << std::endl;
	std::cout << "new loss: " << loss_2 << std::endl;
}