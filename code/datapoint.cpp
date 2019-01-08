#include "datapoint.hpp"

Datapoint::Datapoint(){

}

Datapoint::Datapoint(Eigen::VectorXd data_, double label_, int n_features_){
	this->data = data_;
	this->label = label_;
	this->n_features = n_features_;
	this->is_sparse = false;
}

Datapoint::Datapoint(Eigen::SparseVector<double> sp_data_, double label_, int n_features_){
	this->sp_data = sp_data_;
	this->label = label_;
	this->n_features = n_features_;
	this->is_sparse = true;
}

double Datapoint::dot_product_dense(Eigen::VectorXd w){
	return this->data.dot(w);
}

double Datapoint::dot_product_sparse(Eigen::VectorXd w){
	return this->sp_data.dot(w);
	
}
