#include "cuDL/core/dcon.hpp"
#include "cuDL/core/base.hpp"

template <typename T>
Dcon<T>::Dcon(const vector<int>& shape) {
	ReShape(shape);
}

template <typename T>
Dcon<T>::Dcon(const int num, const int channels, const int height,
    			  const int width): capacity_(0) {
	vector<int> shape(4);
	shape[0] = num;
	shape[1] = channels;
	shape[2] = height;
	shape[3] = width;
	ReShape(shape);
}

template <typename T>
Dcon<T>::Dcon(const int num): capacity_(0) {
	vector<int> shape(4);
	shape[0] = num;
	shape[1] = 1;
	shape[2] = 1;
	shape[3] = 1;
	ReShape(shape);
}

template <typename T>
void Dcon<T>::ReShape(const vector<int>& shape) {
	shape_.resize(shape.size());
	count_ = 1;
	for (int i = 0; i < shape.size(); ++i) {
		count_ *= shape[i];
		shape_[i] = shape[i];
	}

	if (count_ > capacity_) {
    	capacity_ = count_;
    	data_.reset((T *)malloc(capacity_ * sizeof(T)));
    	grad_.reset((T *)malloc(capacity_ * sizeof(T)));
    }
}


INSTANTIATE_CLASS(Dcon);