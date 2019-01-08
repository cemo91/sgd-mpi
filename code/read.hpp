/*
 * Contains methods for reading a file to array of Datapoint
 */
#ifndef READ_H
#define READ_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "datapoint.hpp"

// GET NUMBER OF FEATURES FOR ALL DATA FROM LIBSVM FORMAT FILE
int get_num_feat(const char *infile);

// READ DATA FROM LIBSVM FORMAT FILE
int process_file(std::vector<Datapoint> &data_matrix, const char *infile, bool is_sparse, std::string loss);

#endif
