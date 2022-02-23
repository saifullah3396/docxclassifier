
from das.utils.basic_utils import ExplicitEnum


class TrainValSamplingStrategy(ExplicitEnum):
    RANDOM_SPLIT = "random_split"
    K_FOLD_CROSS_VAL = "k_fold_cross_val"
