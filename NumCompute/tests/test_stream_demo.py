
import csv

from demo.stream_demo import create_sample_csv, run_demo


def test_sample_csv_creation(tmp_path):
    path = tmp_path / "sample.csv"

    create_sample_csv(path, n_samples=10)

    with path.open(newline="") as file:
        rows = list(csv.reader(file))

    assert len(rows) == 11
    assert rows[0] == ["feature_1", "feature_2", "target"]


def test_csv_streaming_demo(tmp_path):
    path = tmp_path / "sample.csv"

    create_sample_csv(path, n_samples=60)
    trainer, results = run_demo(path, chunk_size=20)

    assert trainer.chunk_count == 3
    assert len(results) == 2
    assert all(0 <= accuracy <= 1 for accuracy in results)
    assert len(trainer.get_history()) == 5
