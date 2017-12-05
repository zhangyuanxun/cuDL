#ifndef NET_HPP_
#define NET_HPP_

#include "cuDL/core/dcon.hpp"
#include "cuDL/core/net_paras.hpp"
#include "cuDL/layers/layer_paras.hpp"
#include "cuDL/layers/layer.hpp"

template <typename T>
class Net {
public:
	explicit Net(const NetParas& paras, const vector<LayerParas>& layersPara);

	virtual ~Net() {};

	// get mode (GPU or CPU)
	int GetMode() {return mode_;}

	inline int num_inputs() const { return input_dcons_.size(); }

  	inline int num_outputs() const { return output_dcons_.size(); }

  	inline const vector<Dcon<T>*>& input_dcons() const {
    	return input_dcons_;
  	}
  	inline const vector<Dcon<T>*>& output_dcons() const {
    	return output_dcons_;
  	}

  	inline const string& get_input_path() const {
    	return input_path;
  	}

protected:
	int mode_;               // control GPU or CPU Mode
	float learning_rate_;    // learning rate
	float reg_;              // regularization
	string name_;            // net name
	string input_path;
	vector<vector<Dcon<T>*> > top_vecs_;
	vector<vector<Dcon<T>*> > bottom_vecs_;
	vector<string> layer_names_;
	vector<boost::shared_ptr<Layer<T> > > layers_;
	vector<Dcon<T>*> input_dcons_;
  	vector<Dcon<T>*> output_dcons_;
};

#endif