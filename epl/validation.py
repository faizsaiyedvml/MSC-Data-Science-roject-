"""stage 05 - validation"""
from dataclasses import dataclass, field


@dataclass
class Fold:
    k: int
    train_seasons: list
    test_season: int
    inner_train: list = field(init=False)
    inner_val: int = field(init=False)

    def __post_init__(self):
        self.inner_train = self.train_seasons[:-1]
        self.inner_val = self.train_seasons[-1]


def make_folds(seasons, min_train=3):
    seasons = sorted(seasons)
    return [Fold(k=i - min_train + 1, train_seasons=seasons[:i], test_season=seasons[i])
            for i in range(min_train, len(seasons))]


def label(season):
    return f"{season}/{(season + 1) % 100:02d}"
