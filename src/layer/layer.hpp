//
// Created by TakumiYamashita on 2018/06/10.
//

#ifndef DEEP_LEARNING_FROM_SCRATCH_LAYER_HPP
#define DEEP_LEARNING_FROM_SCRATCH_LAYER_HPP

#include "../util.hpp"

namespace layer {
  class Layer {
    virtual void forward(const ndarray& input, std::shared_ptr<ndarray> output) = 0;
    virtual void backward(const ndarray& dout, std::shared_ptr<ndarray> dx) = 0;
  };
};  // namespace layer

#endif  // DEEP_LEARNING_FROM_SCRATCH_LAYER_HPP
