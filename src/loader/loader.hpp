//
// Created by TakumiYamashita on 2018/06/10.
//

#ifndef DEEP_LEARNING_FROM_SCRATCH_LOADER_HPP
#define DEEP_LEARNING_FROM_SCRATCH_LOADER_HPP

#include <string>
#include "../primitive/ndarray.hpp"

using ndarray = primitive::ndarray;

namespace loader {
  class Loader {
    virtual void download(std::string filenname, std::shared_ptr<ndarray> array) = 0;
    virtual void load(std::string filename, std::shared_ptr<ndarray> array) = 0;
  };
}

#endif //DEEP_LEARNING_FROM_SCRATCH_LOADER_HPP
