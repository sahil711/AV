from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, TimeSeriesSplit, train_test_split
from enum import Enum
'''
include an option to stratify by a column and not only target variable
'''

class FoldScheme(Enum):
    '''
    Class to select the type of fold scheme to apply
    '''
    StratifiedKFold = "StratifiedKFold"
    KFold = "KFold"
    GroupKFold = "GroupKFold"
    TimeSeriesSplit = "TimeSeriesSplit"
    train_test_split = "train_test_split"
    train_test_split_stratify = "train_test_split_stratify"


class CustomFolds(object):
    def __init__(self, validation_scheme=None, num_folds=5, random_state=100, num_repeats=1, shuffle=True, test_size=0.2):
        self.validation_scheme = validation_scheme
        if isinstance(validation_scheme, str) or isinstance(validation_scheme, unicode):
            self.validation_scheme = FoldScheme(validation_scheme)
        if validation_scheme is None:
            self.validation_scheme = FoldScheme.train_test_split

        self.random_state=random_state
        self.shuffle=shuffle
        self.num_folds=num_folds
        self.test_size = test_size
        self.num_repeats=num_repeats

    def get_params(self):
        if isinstance(self.validation_scheme, FoldScheme):
            return {
                "validation_scheme": self.validation_scheme.name,
                "random_state": self.random_state,
                "shuffle": self.shuffle,
                "num_folds": self.num_folds,
                "num_repeats": self.num_repeats,
                "shuffle": self.shuffle,
                "test_size": self.test_size
            }

    def split(self, X, y=None, group=None, **kwargs):
        # the group here will be passed on from the class where this is being called
        if isinstance(self.validation_scheme, FoldScheme):
            if self.validation_scheme == FoldScheme.KFold:
                folds = KFold(n_splits=self.num_folds, random_state=self.random_state, shuffle=self.shuffle)
                return [(train_index, test_index) for (train_index, test_index) in folds.split(X)]
            elif self.validation_scheme == FoldScheme.StratifiedKFold:
                folds = StratifiedKFold(n_splits=self.num_folds, random_state=self.random_state, shuffle=self.shuffle)
                return [(train_index, test_index) for (train_index, test_index) in folds.split(X, y)]
            elif self.validation_scheme == FoldScheme.GroupKFold:
                folds = GroupKFold(n_splits=self.num_folds)
                return [(train_index, test_index) for (train_index, test_index) in folds.split(X, y, groups=group)]
            elif self.validation_scheme == FoldScheme.TimeSeriesSplit:
                folds = TimeSeriesSplit(n_splits=self.num_folds)
                return [(train_index, test_index) for (train_index, test_index) in folds.split(X)]
            elif self.validation_scheme == FoldScheme.train_test_split:
                # validation_scheme is a simple train test split. testsize is used to determine the size of test samples
                return [train_test_split(range(X.shape[0]), test_size=self.test_size, shuffle=self.shuffle, random_state=self.random_state)]
            elif self.validation_scheme == FoldScheme.train_test_split_stratify:
                return [train_test_split(range(X.shape[0]), test_size=self.test_size, shuffle=self.shuffle, stratify=y, random_state=self.random_state)]
        elif callable(self.validation_scheme):
            return self.validation_scheme(X, y, **kwargs)
        else:
            return self.validation_scheme
