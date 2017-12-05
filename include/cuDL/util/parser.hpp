#ifndef PARSER_HPP_
#define PARSER_HPP_

#include <string>
#include <fstream>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include "cuDL/layers/layer_paras.hpp"
#include "cuDL/layers/layer_paras.hpp"

using namespace std;
namespace pt = boost::property_tree;

int read_layer_json(ifstream& file, vector<LayerParas> &vLayers);
int parse_input_layer(const string& type, pt::ptree &p, vector<LayerParas> &vLayers);
int parse_inner_product_layer(const string& type, pt::ptree &p, vector<LayerParas> &vLayers);

#endif