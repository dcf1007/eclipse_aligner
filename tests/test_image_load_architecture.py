from pathlib import Path
import cv2
import numpy as np

import circle_arc_detector as cad

TEXT=Path(cad.__file__).read_text(encoding='utf-8')


def test_load_path_inlines_master_normalization_and_uses_shared_codec():
    block=TEXT.split('def load_image_at(self, index: int):',1)[1].split('\n    def previous_image',1)[0]
    assert 'cv2.IMREAD_UNCHANGED' in block
    assert 'np.uint16' in block
    assert 'compress_array(master_image)' in block
    assert 'master_image_shape' not in block


def test_load_path_calls_auto_select_after_load_processing_context():
    block=TEXT.split('def load_image_at(self, index: int):',1)[1].split('\n    def previous_image',1)[0]
    assert 'self.auto_select_threshold()' in block
    # Auto T is after the with-processing block indentation, not nested within it.
    lines=block.splitlines()
    with_line=next(l for l in lines if 'with self.processing_ui()' in l)
    if_line=next(l for l in lines if 'if settings.threshold is None:' in l)
    assert len(with_line)-len(with_line.lstrip()) == len(if_line)-len(if_line.lstrip())
    assert lines.index(if_line) > lines.index(with_line)


def test_uint16_master_codec_roundtrip():
    master=np.zeros((5,7,4),np.uint16)
    master[...,0]=1234; master[...,3]=65535
    assert np.array_equal(cad.decompress_array(cad.compress_array(master)),master)
