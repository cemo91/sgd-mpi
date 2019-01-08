#include "read.hpp"

using namespace std;

int get_num_feat(const char *infile){
	ifstream r_file(infile);
	string line;
	string word = "";
	string ind = "";
	int max_index = 0;

	if(r_file.is_open()){
		while(getline(r_file,line)){
			stringstream str(line);
			getline(str,word, ' ');
        	while (getline(str,word, ' ')){
    			stringstream ind_val(word);
    			getline(ind_val, ind, ':');
    			int index = atoi(ind.c_str());
    			if(index > max_index){
    				max_index = index;
    			}
        	}
		}
		
		r_file.close();
		return max_index;
	}
	else{
		cerr << "File not found!" << endl;
		return -1;
	}
}

int process_file(std::vector<Datapoint> &data_matrix, const char *infile, bool is_sparse, std::string loss){
	int n = get_num_feat(infile);
	if(n != -1){
		data_matrix.resize(0);
		ifstream r_file(infile);
		string line;
		string word = "";
		string label_str = "";
		string ind = "";
		string val = "";
		int max_index = 0;

		if(r_file.is_open()){
			while(getline(r_file,line)){
				Eigen::VectorXd v = Eigen::VectorXd::Zero(n);
				stringstream str(line);
				getline(str,label_str, ' ');
				double label = stod(label_str);

				if(loss.compare("hinge") == 0 && label == 0){
					label = -1;
				}
				else if(loss.compare("logistic") == 0 && label == -1){
					label = 0;
				}

	        	while (getline(str,word, ' ')){
	    			stringstream ind_val(word);
	    			getline(ind_val, ind, ':');
	    			getline(ind_val, val, ':');
	    			int index = (atoi(ind.c_str())) - 1;
	    			double value = stod(val);
	    			v(index) = value;
	        	}

	        	if(is_sparse == false){
	        		Datapoint d(v, label, n);
	        		data_matrix.push_back(d);
	        	}
	        	else{
	        		Eigen::SparseVector<double> s_v(v.sparseView());
	        		Datapoint d(s_v, label, n);
	        		data_matrix.push_back(d);
	        	}
			}
			
			r_file.close();
			return 0;
		}
		else{
			cerr << "File not found!" << endl;
			return -1;
		}
	}
	else{
		cerr << "File not found!" << endl;
		return -1;
	}
}