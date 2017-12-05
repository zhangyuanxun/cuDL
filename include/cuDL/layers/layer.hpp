#ifndef LAYER_HPP_
#define LAYER_HPP_

#include "cuDL/layers/layer_paras.hpp"
#include "cuDL/core/dcon.hpp"
#include <boost/shared_ptr.hpp>

template <typename T>
class Layer {
public:
	explicit Layer(const LayerParas& paras): layer_paras_(paras) {
		
	}

	virtual ~Layer() {}

  	void SetUp(const vector<Dcon<T>*>& bottom,
	           const vector<Dcon<T>*>& top) {
	    LayerSetUp(bottom, top);
	    Reshape(bottom, top);
  	}

	virtual void LayerSetUp(const vector<Dcon<T>*>& bottom,
    						const vector<Dcon<T>*>& top) {}

  	virtual void Reshape(const vector<Dcon<T>*>& bottom,
      					 const vector<Dcon<T>*>& top) = 0;

	virtual inline const char* type() const { return ""; }

	const LayerParas& layer_paras() const { return layer_paras_; }

	vector<boost::shared_ptr<Dcon<T> > >& get_dcons() {
		return dcons_;
	}


protected:
  	virtual void Forward_cpu(const vector<Dcon<T>*>& bottom,
      						 const vector<Dcon<T>*>& top) = 0;

  	virtual void Forward_gpu(const vector<Dcon<T>*>& bottom,
      						 const vector<Dcon<T>*>& top) {
	    Forward_cpu(bottom, top);
  	}

	virtual void Backward_cpu(const vector<Dcon<T>*>& top,
						      const vector<Dcon<T>*>& bottom) = 0;

	virtual void Backward_gpu(const vector<Dcon<T>*>& top,
	                          const vector<Dcon<T>*>& bottom) {
		Backward_cpu(top, bottom);
	}

	LayerParas layer_paras_;

  	// contains learning parameters
  	vector<boost::shared_ptr<Dcon<T> > > dcons_;

  	DISABLE_COPY_AND_ASSIGN(Layer);
};

#endif