#include "cuDL/core/net.hpp"
#include "cuDL/core/base.hpp"
#include "cuDL/layers/layer_factory.hpp"
#include "cuDL/layers/input_layer.hpp"

template <typename T>
Net<T>::Net(const NetParas& paras, const vector<LayerParas>& layersPara){

	// Initialize net parameter 
	name_ = paras.name;
	learning_rate_ = paras.learning_rate;
	reg_ = paras.reg;
	mode_ = paras.mode;

	// Initialize layer parameter
	int num_layers = layersPara.size();
	cout << "The number of layers: " << num_layers << endl;

	top_vecs_.resize(num_layers);
	bottom_vecs_.resize(num_layers);

	// construct the layers
	for (int layer_id = 0; layer_id < num_layers; ++layer_id) {

		const LayerParas &lp = layersPara[layer_id];
		layers_.push_back(LayerRegistry<T>::CreateLayer(lp));
		layer_names_.push_back(lp.name);

		//1. Connect the layers into your network. 
		if (lp.type == LAYER_TYPE_INPUT) {
			InputLayerParas *pParas = (InputLayerParas*)lp.p_layer;
			input_path = pParas->path;

			input_dcons_.resize(2);
			input_dcons_[0] = layers_[layer_id]->get_dcons()[0].get();
			input_dcons_[1] = layers_[layer_id]->get_dcons()[1].get();
		}

		//2. Setup each layers
		layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);
	}


}

INSTANTIATE_CLASS(Net);