"""Бакетный батчинг по длине последовательностей: границы бакетов и сэмплер с перемешиванием всех батчей."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from torch.utils.data import Sampler


def get_bucket_boundaries(max_length: int, n_buckets: int = 8) -> List[int]:
    """
    Границы бакетов от 0 до max_length (линейно).

    Args:
        max_length: Максимальная длина последовательности.
        n_buckets: Количество бакетов (получится n_buckets+1 граница).

    Returns:
        Список границ, например [0, 64, 128, ..., 512] для max_length=512, n_buckets=8.
    """
    if max_length <= 0 or n_buckets <= 0:
        return [0, max(1, max_length)]
    boundaries = np.linspace(0, max_length, n_buckets + 1, dtype=int)
    boundaries = boundaries.tolist()
    if boundaries[-1] != max_length:
        boundaries[-1] = max_length
    return boundaries


def get_bucket_boundaries_from_lengths(
    lengths: List[int],
    max_length: int,
    n_buckets: int = 8,
) -> List[int]:
    """
    Границы бакетов по квантилям распределения длин (примерно равное число примеров в каждом бакете).

    Args:
        lengths: Длины последовательностей (например, число токенов).
        max_length: Максимальная допустимая длина (последняя граница не превысит её).
        n_buckets: Количество бакетов (получится n_buckets+1 граница).

    Returns:
        Список границ по квантилям, например по 0, 12.5, 25, ..., 100% для n_buckets=8.
    """
    if max_length <= 0 or n_buckets <= 0:
        return [0, max(1, max_length)]
    lengths_arr = np.asarray(lengths, dtype=np.int64)
    if len(lengths_arr) == 0:
        return get_bucket_boundaries(max_length, n_buckets)
    percentiles = np.linspace(0, 100, n_buckets + 1)
    boundaries = np.percentile(lengths_arr, percentiles)
    boundaries = np.round(boundaries).astype(np.int64)
    boundaries = np.clip(boundaries, 0, max_length)
    boundaries[0] = 0
    boundaries[-1] = max_length
    # Строго возрастающие границы (ровно n_buckets+1), чтобы бакетов было ровно n_buckets
    for i in range(1, len(boundaries)):
        if boundaries[i] <= boundaries[i - 1]:
            boundaries[i] = min(boundaries[i - 1] + 1, max_length)
    return boundaries.tolist()


class BucketBatchSampler(Sampler[List[int]]):
    """
    Сэмплер, который группирует примеры по длине (бакеты), перемешивает внутри бакетов,
    собирает все батчи в один список и перемешивает его — бакеты задействованы пропорционально
    размеру и равномерно по эпохе. Неполные батчи отдаются, если drop_last=False.
    """

    def __init__(
        self,
        lengths: List[int],
        bucket_boundaries: List[int],
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Args:
            lengths: Длина каждой последовательности (порядок как в датасете).
            bucket_boundaries: Границы бакетов, например [0, 64, 128, ..., 512].
            batch_size: Размер батча.
            drop_last: Если True, не отдавать неполные батчи.
            shuffle: Перемешивать ли индексы внутри каждого бакета.
            seed: Seed для воспроизводимости (используется вместе с set_epoch).
        """
        self.lengths = lengths
        self.bucket_boundaries = bucket_boundaries
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed if seed is not None else 0
        self.epoch = 0

        boundaries_arr = np.array(bucket_boundaries, dtype=np.int64)
        n_buckets = len(boundaries_arr) - 1
        if n_buckets < 1:
            n_buckets = 1

        # Присвоить каждому индексу номер бакета
        bucket_ids = np.searchsorted(boundaries_arr[1:], lengths, side="right")
        bucket_ids = np.clip(bucket_ids, 0, n_buckets - 1)

        # Сгруппировать индексы по бакетам
        self._buckets: List[List[int]] = [[] for _ in range(n_buckets)]
        for idx, bid in enumerate(bucket_ids):
            self._buckets[int(bid)].append(idx)

        self._num_batches = self._compute_num_batches()

    def _compute_num_batches(self) -> int:
        total = 0
        for bucket in self._buckets:
            n = len(bucket)
            if n == 0:
                continue
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.Generator(np.random.PCG64(self.seed + self.epoch))
        batch_size = self.batch_size
        drop_last = self.drop_last

        # Перемешать каждый бакет и разбить на батчи
        batches_per_bucket: List[List[List[int]]] = []
        for bucket in self._buckets:
            indices = list(bucket)
            if self.shuffle and len(indices) > 0:
                rng.shuffle(indices)
            batches = []
            for start in range(0, len(indices), batch_size):
                batch = indices[start : start + batch_size]
                if drop_last and len(batch) < batch_size:
                    continue
                batches.append(batch)
            batches_per_bucket.append(batches)

        # Собрать все батчи в один список и перемешать (пропорционально размеру бакетов, равномерно по эпохе)
        all_batches = []
        for batches in batches_per_bucket:
            all_batches.extend(batches)
        rng.shuffle(all_batches)
        for batch in all_batches:
            yield batch

    def __len__(self) -> int:
        return self._num_batches
