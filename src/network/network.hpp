//
// Created by TakumiYamashita on 2018/06/10.
//

#ifndef DEEP_LEARNING_FROM_SCRATCH_NETWORK_HPP
#define DEEP_LEARNING_FROM_SCRATCH_NETWORK_HPP

#include "../primitive/primitive.hpp"

using ndarray = primitive::ndarray;

namespace network {
  class Network {
    virtual void predict(ndarray& input, std::shared_ptr<ndarray> output, bool train_flag) = 0;
    virtual void loss(ndarray& input, ndarray& teacher, std::shared_ptr<dnarray> output) = 0;
    virtual void accuracy(ndarray& input, ndarray& teacher, std::shared_ptr<dnarray> output, int batch_test) = 0;
    virtual void gradient(ndarray& input, ndarray& teacher, std::shared_ptr<ndarray> output) = 0;
  };
}  // namespace network

#endif  // DEEP_LEARNING_FROM_SCRATCH_NETWORK_HPP
