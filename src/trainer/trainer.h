//
// Created by TakumiYamashita on 2018/06/10.
//

#ifndef DEEP_LEARNING_FROM_SCRATCH_TRAINER_H
#define DEEP_LEARNING_FROM_SCRATCH_TRAINER_H

#include "../network/network.hpp"
#include "../optimizer/optimizer.hpp"
#include <memory>

namespace trainer {
  class Trainer {
   public:
    Trainer(std::unique_ptr<network::Network> network, ndarray& x_train,
            ndarray& t_train, ndarray& x_test, ndarray& t_test, int epochs,
            int mini_batch_size,
            std::unique_ptr<optimizer::Optimizer> optimizer,
            int evaluate_sample_num_per_epoch, bool verbose)
        : network_(std::move(network)),
          x_train_(x_train),
          t_train_(t_train),
          x_test_(x_test),
          t_test_(t_test),
          epochs_(epochs),
          mini_batch_size_(mini_batch_size),
          optimizer_(std::move(optimizer)),
          evaluate_sample_num_per_epoch_(evaluate_sample_num_per_epoch),
          verbose_(verbose) {}

    void train_step() {
      std::cout << "================= train step ==================="
                << std::endl;
    }
    void train() {
      std::cout << "================= train ===================" << std::endl;
    }

   private:
    std::unique_ptr<network::Network> network_;
    std::unique_ptr<optimizer::Optimizer> optimizer_;
    ndarray &x_train_, t_train_, x_test_, t_test_;
    int epochs_, mini_batch_size_, evaluate_sample_num_per_epoch_;
    bool verbose_;
  };
}  // namespace trainer

#endif  // DEEP_LEARNING_FROM_SCRATCH_TRAINER_H
