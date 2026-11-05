"""Guard dataset migration, original split membership and generation stability."""

import json
from pathlib import Path

import pytest

from scripts.generate_dataset import generate_dataset
from scripts.organize_datasets import hashes, validate_layout


def test_all_six_datasets_have_paired_disjoint_splits_and_legacy_bytes():
    data = Path("data")
    result = validate_layout(data)
    assert result["dataset1"]["static"]["split_sizes"] == {"train": 1000, "validation": 100, "test": 200}
    assert result["dataset2"]["dynamic"]["fixed_scenarios"]["test"] == {
        "none": 130, "job_release_only": 40, "breakdown_only": 20, "both": 10,
    }
    migration = json.loads((data / "dataset-layout-migration.json").read_text())
    for key, root in (("static", data / "dataset_test/static"),
                      ("dynamic", data / "dataset_test/dynamic"),
                      ("dataset2", data / "dataset2/static")):
        assert hashes(root) == migration["legacy_hashes"][key]
    for mode in ("static", "dynamic"):
        for path in (data / f"dataset_test/{mode}").glob("*/*.json"):
            assert path.read_bytes() == (data / f"dataset1/{mode}" / path.relative_to(data / f"dataset_test/{mode}")).read_bytes()


def test_expansion_does_not_shift_evaluation_when_training_size_changes(tmp_path):
    archive = Path("data/dataset_test/static")
    one, two = tmp_path / "one", tmp_path / "two"
    first = generate_dataset(one, preserved_root=archive, seed=37,
                             split_sizes={"train": 101, "validation": 21, "test": 21})
    generate_dataset(two, preserved_root=archive, seed=37,
                     split_sizes={"train": 102, "validation": 21, "test": 21})
    for split in ("validation", "test"):
        assert hashes(one / split) == hashes(two / split)
    assert first["preserved_counts"] == {"train": 100, "validation": 20, "test": 20}
    for path in archive.glob("*/*.json"):
        assert (one / path.relative_to(archive)).read_bytes() == path.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        generate_dataset(one, preserved_root=archive)
