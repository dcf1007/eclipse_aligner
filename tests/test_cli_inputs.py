import sys

import pytest

import circle_arc_detector as cad


class FakeRoot:
    def mainloop(self):
        pass


def run_main(monkeypatch, arguments):
    captured = []
    monkeypatch.setattr(sys, "argv", ["circle_arc_detector.py", *arguments])
    monkeypatch.setattr(cad.tk, "Tk", FakeRoot)
    monkeypatch.setattr(
        cad,
        "DetectorApp",
        lambda root, image_paths: captured.append((root, list(image_paths))),
    )
    cad.main()
    return captured[0][1]


def test_cli_preserves_explicit_file_order(monkeypatch):
    assert run_main(monkeypatch, ["third.jpg", "first.tif", "second.tiff"]) == [
        "third.jpg",
        "first.tif",
        "second.tiff",
    ]


def test_cli_single_folder_adds_only_supported_images_sorted_by_filename(
    tmp_path,
    monkeypatch,
):
    for name in ("z.TIFF", "B.jpg", "a.TIF", "ignored.jpeg", "ignored.png"):
        (tmp_path / name).touch()
    (tmp_path / "nested.jpg").mkdir()

    image_paths = run_main(monkeypatch, [str(tmp_path)])

    assert image_paths == [
        str(tmp_path / "a.TIF"),
        str(tmp_path / "B.jpg"),
        str(tmp_path / "z.TIFF"),
    ]


def test_cli_rejects_folder_mixed_with_explicit_files(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["circle_arc_detector.py", str(tmp_path), "other.jpg"],
    )
    with pytest.raises(SystemExit) as exc:
        cad.main()
    assert exc.value.code == 2


def test_removed_processing_options_are_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["circle_arc_detector.py", "--threshold", "8"])
    with pytest.raises(SystemExit) as exc:
        cad.main()
    assert exc.value.code == 2
