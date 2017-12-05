#include "cuDL/layers/inner_product_layer.hpp"
#include "cuDL/core/base.hpp"
#include "cuDL/layers/layer_factory.hpp"
#include "cuDL/layers/layer_paras.hpp"

template <typename T>
void InnerProductLayer<T>::LayerSetUp(const vector<Dcon<T>*>& bottom,
      								  const vector<Dcon<T>*>& top) {
	
	InnerProductLayerParas *pParas = (InnerProductLayerParas*)this->layer_paras_.p_layer;
	const int num_output = pParas->num_output;

	N_ = num_output;
}

INSTANTIATE_CLASS(InnerProductLayer);
REGISTER_LAYER_CLASS(InnerProduct);
