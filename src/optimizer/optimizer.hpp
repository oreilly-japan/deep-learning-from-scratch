//
// Created by TakumiYamashita on 2018/06/10.
//

#ifndef DEEP_LEARNING_FROM_SCRATCH_OPTIMIZER_HPP
#define DEEP_LEARNING_FROM_SCRATCH_OPTIMIZER_HPP

namespace entity {
    class Optimizer {
        Optimizer( int lr );
        virtual update();
    };
}

#endif //DEEP_LEARNING_FROM_SCRATCH_OPTIMIZER_HPP
