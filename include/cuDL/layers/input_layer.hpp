#ifndef INPUT_LAYER_HPP_
#define INPUT_LAYER_HPP_
#include "cuDL/layers/layer.hpp"
#include "cuDL/core/base.hpp"
#include "cuDL/core/dcon.hpp"

template <typename T>
class InputLayer : public Layer<T> {
public:
	explicit InputLayer(const LayerParas& paras): Layer<T>(paras) {
		InputLayerParas *pParas = (InputLayerParas*)paras.p_layer;

		// Allocate memory for data and label
		this->dcons_.resize(2);
		this->dcons_[0].reset(new Dcon<T>(pParas->shape[0], 
			                              pParas->shape[1],
			                              pParas->shape[2],
			                              pParas->shape[3]));

		// Allocate memory for lable
		this->dcons_[1].reset(new Dcon<T>(pParas->shape[0]));
	}

	virtual void LayerSetUp(const vector<Dcon<T>*>& bottom,
   							const vector<Dcon<T>*>& top);

	virtual void Reshape(const vector<Dcon<T>*>& bottom,
      				     const vector<Dcon<T>*>& top) {}

	virtual inline const char* type() const { return "Input"; }
	
protected:
	virtual void Forward_cpu(const vector<Dcon<T>*>& bottom,
    						 const vector<Dcon<T>*>& top) {}
  	virtual void Backward_cpu(const vector<Dcon<T>*>& top,
      						  const vector<Dcon<T>*>& bottom) {}

};

#endif