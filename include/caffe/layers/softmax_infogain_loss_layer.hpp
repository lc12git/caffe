#ifndef CAFFE_SOFTMAX_INFOGAIN_LOSS_LAYER_HPP_
#define CAFFE_SOFTMAX_INFOGAIN_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

namespace caffe {
template <typename Dtype>
class SoftmaxInfogainLossLayer : public LossLayer<Dtype> {
 public:
  explicit SoftmaxInfogainLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), infogain_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
      
  virtual inline const char* type() const { return "SoftmaxInfogainLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 4; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual Dtype get_normalizer(
      LossParameter_NormalizationMode normalization_mode, int valid_count);
  shared_ptr<Layer<Dtype> > softmax_layer_;
  Blob<Dtype> prob_;
  vector<Blob<Dtype>*> softmax_bottom_vec_;
  vector<Blob<Dtype>*> softmax_top_vec_;
  bool has_ignore_label_;
  int ignore_label_;
  LossParameter_NormalizationMode normalization_;
  int softmax_axis_, outer_num_, inner_num_;
  Blob<Dtype> infogain_;
};

}  // namespace caffe

#endif  // CAFFE_SOFTMAX_INFOGAIN_LOSS_LAYER_HPP_
