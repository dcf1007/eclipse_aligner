from pathlib import Path

TEST = Path(__file__).resolve().parents[1] / "tests" / "test_gui_focus_autoselect.py"
text = TEST.read_text(encoding="utf-8")
old = '''def test_threshold_auto_select_is_implemented_and_refreshes_preview():\n    block = TEXT.split("def auto_select_threshold(self):", 1)[1].split(\n        "def auto_select_radius", 1\n    )[0]\n    assert "auto_threshold_from_gray(self.gray_image)" in block\n    assert "self.threshold.set(selected_threshold)" in block\n    assert "self.refresh_preview()" in block\n    assert "Preview not regenerated." not in block\n'''
new = '''def test_threshold_auto_select_reuses_cached_result_and_commits_threshold():\n    block = TEXT.split("def auto_select_threshold(self):", 1)[1].split(\n        "def auto_select_radius", 1\n    )[0]\n    assert 'state["auto_threshold_result"]' in block\n    assert "self.threshold.set(selected_threshold)" in block\n    assert 'self._commit_setting_change("threshold")' in block\n    assert "self.refresh_preview()" not in block\n'''
assert old in text
TEST.write_text(text.replace(old, new, 1), encoding="utf-8")
print("Corrected stale Auto-select test expectation")
