//
// Created by TakumiYamashita on 2018/06/10.
//

#ifndef DEEP_LEARNING_FROM_SCRATCH_LAYER_HPP
#define DEEP_LEARNING_FROM_SCRATCH_LAYER_HPP

#include "util.hpp"

namespace entity {
        class Layer {
            virtual forward(std::vector<tensor_t> &in);
            virtual back
        };
};

#endif //DEEP_LEARNING_FROM_SCRATCH_LAYER_HPP
