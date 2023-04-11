# Setup

1. Create a `build` directory and `data` directory

2. Set up cmake with `cmake -S . -B build`

3. Build rocksdb with `cmake --build build`

# Running CAMAL

1. `python camal_lr_level.py && python camal_lr_tier.py` to collect training data.

2. `python cost_lr_uni.py` to train the model.

3. `python optimizer_uniform_overall.py` to evaluate optimization.
