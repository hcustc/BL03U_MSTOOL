from pathlib import Path

import numpy as np

from src.bl03u_massspec.data_processor import DataProcessor


def _write_sample_txt(path: Path, values):
    header = "\n".join([f"Header {i}" for i in range(10)])
    body = "\n".join(str(v) for v in values)
    path.write_text(f"{header}\n{body}\n", encoding="utf-8")


def test_get_head_info_supports_scientific_notation_and_default():
    processor = DataProcessor()
    head_list = ["Beam current: -1.23E+04", "No numeric value"]

    result = processor.get_head_info(head_list)

    assert result[0] == "-1.23E+04"
    assert result[1] == "1"


def test_load_data_for_plot_returns_aligned_x_y(tmp_path):
    processor = DataProcessor()
    sample_file = tmp_path / "single.txt"
    values = list(range(10))
    _write_sample_txt(sample_file, values)

    x_data, y_data = processor.load_data_for_plot(
        str(sample_file),
        skiprows=10,
        start_index=2,
    )

    assert np.array_equal(y_data, np.array(values[2:]))
    assert np.array_equal(x_data, np.arange(3, 11))
    assert len(x_data) == len(y_data)


def test_load_folder_data_for_plot_accumulates_txt_files(tmp_path):
    processor = DataProcessor()
    folder = tmp_path / "samples"
    folder.mkdir()

    _write_sample_txt(folder / "a.txt", [1, 2, 3])
    _write_sample_txt(folder / "b.txt", [4, 5, 6])

    x_data, y_data = processor.load_folder_data_for_plot(
        str(folder),
        skiprows=10,
        start_index=0,
    )

    assert np.array_equal(x_data, np.array([1, 2, 3]))
    assert np.array_equal(y_data, np.array([5, 7, 9]))
