/*
 * Contains methods for Stochastic Gradient Descent
 */
#ifndef SGD_H
#define SGD_H

#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>
#include <random>
#include <string>
#include <time.h>
#include <thread>
#include "datapoint.hpp"

// RANDOMLY GENERATE AN INDEX FOR SGD
int generate_index(int data_size);

// GENERATE RANDOM INDICES OF MINI_BATCH_SIZE
std::vector<int> generate_mini_batch(int mini_batch_size, int data_size);

// DOT PRODUCT OF DATAPOINT GIVEN THE WEIGHT
double dot_product(Datapoint &x, Eigen::VectorXd &w);

// CALCULATE THE LOSS OF A DATAPOINT
double compute_loss(Datapoint &x, Eigen::VectorXd &weight, std::string loss, std::string regularizer, double lambda);

// CALCULATE THE LOSS OF A SET OF DATAPOINTS
double compute_average_loss(std::vector<Datapoint> &data_matrix, Eigen::VectorXd &weight, std::string loss, std::string regularizer, double lambda);

// CALCULATE THE ACCURACY ON A SET OF DATAPOINTS
double compute_accuracy(std::vector<Datapoint> &data_matrix, Eigen::VectorXd &weight, std::string loss);

// CALCULATE THE DERIVATIVE OF THE LOSS
double loss_derivative(double s, double label, std::string loss);

// COMPUTE THE GRADIENT FOR THE REGULARIZER
Eigen::VectorXd compute_regularizer_gradient(std::string regularizer, Eigen::VectorXd &weight, double lambda);

// COMPUTE THE STOCHASTIC GRADIENT FOR A MINI BATCH
Eigen::VectorXd compute_gradient_dense(std::vector<Datapoint> &data_matrix, std::vector<int> indices, Eigen::VectorXd &weight, std::string loss, std::string regularizer, double lambda);

// COMPUTE THE STOCHASTIC GRADIENT FOR A MINI BATCH
Eigen::VectorXd compute_gradient_sparse(std::vector<Datapoint> &data_matrix, std::vector<int> indices, Eigen::VectorXd &weight, std::string loss, std::string regularizer, double lambda);

// UPDATE WEIGHTS FOR MINI BATCH
Eigen::VectorXd weight_update_dense(std::vector<Datapoint> &data_matrix, std::vector<int> indices, Eigen::VectorXd &weight, std::string loss, double step_size, std::string regularizer, double lambda);

// UPDATE WEIGHTS FOR MINI BATCH
Eigen::VectorXd weight_update_sparse(std::vector<Datapoint> &data_matrix, std::vector<int> indices, Eigen::VectorXd &weight, std::string loss, double step_size, std::string regularizer, double lambda);

// TRAIN AN SGD CLASSIFIER
double train(std::vector<Datapoint> &data_matrix, Eigen::VectorXd &weight, std::string loss, std::string learning_rate, double step_size, std::string regularizer, double lambda, int mini_batch_size, int max_epochs);

// CHECK THE GRADIENT COMPUTATION IF IT IS CORRECT, WITH A FIRST ORDER APPROXIMATION
void gradient_check(Datapoint &x, std::string loss, std::string regularizer, double lambda);

#endif
