//
// Created by TakumiYamashita on 2018/06/10.
//

#ifndef DEEP_LEARNING_FROM_SCRATCH_PARAMETERS_HPP
#define DEEP_LEARNING_FROM_SCRATCH_PARAMETERS_HPP

#include <unoredered_map>

namespace primitive {
  template <typename K, typename V>
  class Parameters {
    V &operator[](const K &key) { return mp[key]; }

    std::unordered_map<K, V> mp;
  };
}  // namespace primitive

#endif  // DEEP_LEARNING_FROM_SCRATCH_PARAMETERS_HPP
