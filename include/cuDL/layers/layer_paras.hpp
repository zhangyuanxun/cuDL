#ifndef LAYER_PARAS_HPP_
#define LAYER_PARAS_HPP_

#include <string>
using namespace std;

const string LAYER_TYPE_INPUT = "Input";
const string LAYER_TYPE_INNER_PROUDCT = "InnerProduct";

typedef struct {
	int shape[4];
	int lable;
	string path;
} InputLayerParas;

typedef struct {
	int num_output;
}InnerProductLayerParas;

typedef struct {
	string type;
	string name;
	string top;
	string bottom;
	void* p_layer;
} LayerParas;

#endif