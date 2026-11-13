
"""Read numeric CSV datasets in small, memory-efficient chunks."""

import csv

import numpy as np


def read_csv_chunks(file_path, chunk_size=100, target_column=-1):
    """Yield feature and label arrays from successive CSV chunks."""
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")

    with open(file_path, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)

        header = next(reader, None)

        if header is None:
            raise ValueError("CSV file is empty.")

        n_columns = len(header)

        if n_columns < 2:
            raise ValueError("CSV needs features and a target column.")

        target_index = target_column

        if target_index < 0:
            target_index += n_columns

        if not 0 <= target_index < n_columns:
            raise ValueError("Invalid target column index.")

        rows = []

        def make_chunk(batch):
            data = np.asarray(batch, dtype=float)
            X = np.delete(data, target_index, axis=1)
            y = data[:, target_index]
            return X, y

        for row in reader:
            if not row:
                continue

            if len(row) != n_columns:
                raise ValueError("CSV row has an incorrect column count.")

            rows.append(row)

            if len(rows) == chunk_size:
                yield make_chunk(rows)
                rows = []

        if rows:
            yield make_chunk(rows)
