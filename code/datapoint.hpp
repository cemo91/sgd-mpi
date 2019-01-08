/*
 * CLASS TO REPRESENT DATA, EACH ROW OF AN INPUT MATRIX IS A DATAPOINT
 * CLASS ATTRIBUTES: DATA, LABEL, NUMBER OF FEATURES
 * CLASS METHODS: DOT PRODUCT WITH A DATAPOINT, A DOUBLE ARRAY AND A VECTOR
 */

#ifndef DATAPOINT_H
#define DATAPOINT_H

#include <vector>
#include <Eigen>

class Datapoint{
public:
	Eigen::VectorXd data;
	Eigen::SparseVector<double> sp_data;
	double label;
	int n_features;
	bool is_sparse;

	Datapoint();
	Datapoint(Eigen::VectorXd data_, double label_, int n_features_);
	Datapoint(Eigen::SparseVector<double> sp_data_, double label_, int n_features_);

	// DOT PRODUCT USING DENSE DATA : data
	double dot_product_dense(Eigen::VectorXd w);

	// DOT PRODUCT USING SPARSE DATA : sp_data
	double dot_product_sparse(Eigen::VectorXd w);
};

#endif
