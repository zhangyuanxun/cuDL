#ifndef INNER_PRODUCT_LAYER_HPP_
#define INNER_PRODUCT_LAYER_HPP_

#include "cuDL/layers/layer.hpp"
#include "cuDL/core/base.hpp"

template <typename T>
class InnerProductLayer : public Layer<T> {
 public:
  explicit InnerProductLayer(const LayerParas& paras)
      : Layer<T>(paras) {}
  virtual void LayerSetUp(const vector<Dcon<T>*>& bottom,
                          const vector<Dcon<T>*>& top);
  virtual void Reshape(const vector<Dcon<T>*>& bottom,
                       const vector<Dcon<T>*>& top) {}

  virtual inline const char* type() const { return "InnerProduct"; }

 protected:
  virtual void Forward_cpu(const vector<Dcon<T>*>& bottom,
                           const vector<Dcon<T>*>& top) {}
  virtual void Forward_gpu(const vector<Dcon<T>*>& bottom,
                           const vector<Dcon<T>*>& top) {}
  virtual void Backward_cpu(const vector<Dcon<T>*>& top, 
  	                        const vector<Dcon<T>*>& bottom) {}
  virtual void Backward_gpu(const vector<Dcon<T>*>& top,
                            const vector<Dcon<T>*>& bottom) {}
  int M_;
  int K_;
  int N_;
};

#endif