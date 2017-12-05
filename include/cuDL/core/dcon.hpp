#ifndef DCON_HPP_
#define DCON_HPP_

#include "cuDL/core/base.hpp"
#include <boost/shared_ptr.hpp>

// Define data container class
template <typename T>
class Dcon {
public:
	Dcon(): data_(), grad_(), count_(0), capacity_(0){}

  	explicit Dcon(const int num, const int channels, const int height,
      			  const int width);

	explicit Dcon(const vector<int>& shape);

	explicit Dcon(const int num);

	void ReShape(const vector<int>& shape);

	inline const boost::shared_ptr<T>& data() const {
		return data_;
	}

private:
	boost::shared_ptr<T> data_;
	boost::shared_ptr<T> grad_;
	vector<int> shape_;
	int capacity_;
	int count_;
};

#endif