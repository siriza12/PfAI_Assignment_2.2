
import csv

import numpy as np
import pytest

from numcompute_stream.csv_stream import read_csv_chunks


def test_csv_stream_multiple_chunks(tmp_path):
    path = tmp_path / "data.csv"

    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["feature", "target"])
        writer.writerows([
            [1, 0],
            [2, 0],
            [3, 1],
            [4, 1],
            [5, 1],
        ])

    chunks = list(read_csv_chunks(path, chunk_size=2))

    assert len(chunks) == 3
    assert [len(y) for _, y in chunks] == [2, 2, 1]


def test_csv_stream_separates_features_and_labels(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b,target\n1,10,0\n2,20,1\n")

    X, y = next(read_csv_chunks(path, chunk_size=2))

    np.testing.assert_allclose(X, [[1, 10], [2, 20]])
    np.testing.assert_allclose(y, [0, 1])


def test_csv_stream_rejects_invalid_chunk_size(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("x,target\n1,0\n")

    with pytest.raises(ValueError):
        list(read_csv_chunks(path, chunk_size=0))



from numcompute.io import load_csv, load_csv_chunks


def test_io_csv_chunks_match_existing_loader(tmp_path):
    path = tmp_path / "stream.csv"
    path.write_text(
        "a,b,target\n"
        "1,10,0\n"
        "2,20,0\n"
        "3,30,1\n"
        "4,40,1\n"
        "5,50,1\n",
        encoding="utf-8",
    )

    chunks = list(load_csv_chunks(path, chunk_size=2))

    assert [len(y) for _, y in chunks] == [2, 2, 1]

    combined = np.vstack([
        np.column_stack((X, y))
        for X, y in chunks
    ])

    expected = load_csv(path, skip_header=1)
    np.testing.assert_allclose(combined, expected)


def test_io_csv_chunks_reject_invalid_size(tmp_path):
    path = tmp_path / "stream.csv"
    path.write_text("feature,target\n1,0\n", encoding="utf-8")

    with pytest.raises(ValueError):
        list(load_csv_chunks(path, chunk_size=0))
