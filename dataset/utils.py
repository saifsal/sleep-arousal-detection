import numpy as np

from binascii import b2a_hex
from os import listdir, urandom


class Mode:
    predict = -1
    train = 0
    valid = 1


class ChannelSelector:
    def __init__(self, rng, channels, channel_lookup):
        self.rng = rng

        self.id = b2a_hex(urandom(5)).decode("ascii")

        self.fixed_indices = []
        self.random_indices = []
        self.cnames = []

        if "li" in channels:
            self.fixed_indices = [0, 8, 7, 10, 12]
            self.random_indices = [
                np.arange(7),
                np.arange(2) + 8,
                np.array([7]),
                np.array([10]),
                np.array([12]),
            ]
            self.cnames = ["li"]
        else:
            for ch, num in channel_lookup.items():
                self.add_channel(ch, channels, num[0], num[1])

        self.fixed_indices = np.array(self.fixed_indices)
        assert len(self.fixed_indices) == len(self.random_indices)

    def add_channel(self, ch: str, channels: set, fixed: int, random: list):
        if ch in channels:
            self.fixed_indices.append(fixed)
            self.random_indices.append(np.array(random))
            self.cnames.append(ch)

    def indices(self, mode):
        if mode == Mode.predict:
            return self.fixed_indices
        else:
            return self.bootstrap()

    def bootstrap(self):
        indices = []
        for r in self.random_indices:
            self.rng.shuffle(r)
            indices.append(r[0])
        return np.array(indices)

    def model_name(self, index: int):
        return "-".join(self.cnames + [str(index), self.id])

    def __len__(self):
        return len(self.random_indices)


class DataSplitter:
    def __init__(self, path, constructor, selector, rng):
        self.path = path
        self.constructor = constructor
        self.selector = selector
        self.rng = rng
        self.size = len(listdir(self.path))

    def split_list(self, l, n):
        return [list(a) for a in np.array_split(l, n)]

    def get_test_sets(self):
        return [
            self.constructor(self.path, self.selector, self.rng, i)
            for i in self.split_list(range(self.size), 4)
        ]

    def get_train_sets(self):
        tsets = self.split_list(range(self.size), 4)

        def make_set(a, b, c):
            return tsets[a] + tsets[b] + tsets[c]

        rsets = [
            make_set(1, 2, 3),
            make_set(0, 2, 3),
            make_set(0, 1, 3),
            make_set(0, 1, 2),
        ]

        return [
            self.constructor(
                self.path,
                self.selector,
                self.rng,
                i) for i in rsets]
