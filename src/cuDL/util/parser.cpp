#include <iostream>
#include <vector>
#include "cuDL/layers/layer_paras.hpp"
#include "cuDL/util/errors.hpp"
#include "cuDL/util/parser.hpp"

using namespace std;
namespace pt = boost::property_tree;

int read_layer_json(ifstream& file, vector<LayerParas> &vLayers){
    int ret = ERROR_SUCCESS;

	// Short alias for this namespace
    pt::ptree root;
    pt::read_json(file, root);
    
    // Step 1: Get all the layer's name
    vector<string> layers_type;

    pt::ptree::const_iterator end = root.end();
    for (pt::ptree::const_iterator it = root.begin(); it != end; ++it) {
    	layers_type.push_back(it->first);
    }

    // Stpe 2: parse the parameters for each layers and push into layer vector
    
    for (int i = 0; i < layers_type.size(); ++i) {
    	string type = layers_type[i];
        pt::ptree children = root.get_child(type);

        if (type.compare(LAYER_TYPE_INPUT) == 0) {
            ret = parse_input_layer(type, children, vLayers);

            if (ret != ERROR_SUCCESS) {
                return ret;
            }
        }
        else if(type.compare(LAYER_TYPE_INNER_PROUDCT) == 0) {
            ret = parse_inner_product_layer(type, children, vLayers);

            if (ret != ERROR_SUCCESS) {
                return ret;
            }
        }
        else {
            return ERROR_FAILURE;
        }
    }

    return ERROR_SUCCESS;
}


int parse_input_layer(const string& type, pt::ptree &p, 
                      vector<LayerParas> &vLayers) {
    LayerParas lp;
    InputLayerParas *pParas = new InputLayerParas();
    lp.type = type;
    lp.p_layer = pParas;

    for (const auto& kv : p) {
        string k = (string)kv.first;
        if (k.compare("name") == 0) {
            lp.name = (string)kv.second.data();
        }
        else if(k.compare("top") == 0) {
            lp.top = (string)kv.second.data();
        }
        else if(k.compare("path") == 0) {
            pParas->path = (string)kv.second.data();
        }
        else if(k.compare("shape") == 0) {
            pt::ptree items = p.get_child(k);
            int i = 0;
            for (auto& it: items) {
                pParas->shape[i] = it.second.get_value<int>();
                i++;
            }            
        }
    }

    vLayers.push_back(lp);
    return ERROR_SUCCESS;
}

int parse_inner_product_layer(const string& type, pt::ptree &p, 
                              vector<LayerParas> &vLayers) {
    LayerParas lp;
    InnerProductLayerParas *pParas = new InnerProductLayerParas();
    lp.type = type;
    lp.p_layer = pParas;

    for (const auto& kv : p) {
        string k = (string)kv.first;
        if (k.compare("name") == 0) {
            lp.name = (string)kv.second.data();
        }
        else if(k.compare("top") == 0) {
            lp.top = (string)kv.second.data();
        }
        else if(k.compare("bottom") == 0) {
            lp.top = (string)kv.second.data();
        }
        else if(k.compare("num_output") == 0) {
            pParas->num_output = kv.second.get_value<int>();
        }
    }

    vLayers.push_back(lp);
    return ERROR_SUCCESS;
}

