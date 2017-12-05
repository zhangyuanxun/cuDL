#include <iostream>
#include "cuDL/core/net.hpp"
#include "cuDL/util/parser.hpp"
#include "cuDL/core/net_paras.hpp"
#include "cuDL/core/dcon.hpp"
#include <fstream>

using namespace std;

void print_matrix(float *A, int row, int col) {
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			cout << *(A + i * col + j) << ", ";
		}
		cout <<endl;
	}
}

vector<float> split(const string& str, const string& delim)
{
    vector<float> tokens;
    size_t prev = 0, pos = 0;
    do
    {
        pos = str.find(delim, prev);
        if (pos == string::npos) pos = str.length();
        string token = str.substr(prev, pos-prev);
        if (!token.empty()) tokens.push_back(stof(token));
        prev = pos + delim.length();
    }
    while (pos < str.length() && prev < str.length());
    return tokens;
}

void fill_input_layer(float *input_layer, string input_path) {
	ifstream inputs(input_path.c_str());

	string subs;
	string line;
	int count = 0;
	int offset = 0;
  	while (getline(inputs, line)) {
  		vector<float> v = split(line, ","); 
  		for (int i = 0; i < v.size(); ++i) {
  			count++;
  			if (count % 5 !=0) {
  				*(input_layer + offset) = v[i];
  				offset++;
  			}
  		}
  	}
}

void fill_output_layer(float *output_layer, string input_path) {
	ifstream inputs(input_path.c_str());

	string subs;
	string line;
	int count = 0;
	int offset = 0;
  	while (getline(inputs, line)) {
  		vector<float> v = split(line, ","); 
  		for (int i = 0; i < v.size(); ++i) {
  			count++;
  			if (count % 5 ==0) {
  				*(output_layer + offset) = v[i];
  				offset++;
  			}
  		}
  	}
}


int main(int argc, char** argv) {

	// Read json file
	ifstream jsonFile("one_hidden_layer.json");

	vector<LayerParas> layersPara;
	read_layer_json(jsonFile, layersPara);

	// Configure net parameters 
	NetParas netPara;
	netPara.name = "One hidden neural network";
	netPara.learning_rate = 1e-2;
	netPara.reg = 5e-6;
	netPara.mode = CPU_MODE;

	// Initialize the network 
    Net<float> *pNet = new Net<float>(netPara, layersPara);

    float* input_layer = static_cast<float*>(pNet->input_dcons()[0]->data().get());
    float* output_layer = static_cast<float*>(pNet->input_dcons()[1]->data().get());
    string input_path = pNet->get_input_path();
    fill_input_layer(input_layer, input_path);
    fill_output_layer(output_layer, input_path);

    

	delete pNet;
	return 0;
}