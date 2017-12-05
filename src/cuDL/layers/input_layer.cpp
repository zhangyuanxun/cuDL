#include "cuDL/layers/input_layer.hpp"
#include "cuDL/core/base.hpp"
#include "cuDL/layers/layer_factory.hpp"

template <typename T>
void InputLayer<T>::LayerSetUp(const vector<Dcon<T>*>& bottom,
      						   const vector<Dcon<T>*>& top) {
	
}

INSTANTIATE_CLASS(InputLayer);
REGISTER_LAYER_CLASS(Input);
