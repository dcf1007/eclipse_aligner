from pathlib import Path

import circle_arc_detector as cad

TEXT=Path(cad.__file__).read_text(encoding='utf-8')


def test_removed_obsolete_autot_and_gui_scaffolding():
    for name in ('find_auto_threshold','separation_result_callback','_display_auto_separation_result','display_raw_threshold','_schedule_canvas_redraw','_redraw_cached_canvases','CANVAS_REDRAW_DELAY_MS'):
        assert name not in TEXT


def test_removed_cumbersome_helpers_and_old_codecs():
    for name in ('clean_solar_component','_lattice_boundary_points','compress_full_mask','decompress_full_mask','compress_master_bgra16','decompress_master_bgra16','master_bgra16_to_gray8','master_bgra16_to_display_bgra8','normalize_master_bgra16','master_image_shape'):
        assert name not in TEXT


def test_expected_stage_names_and_shared_helpers_exist():
    for name in ('def morphological_cleanup(','def compress_array(','def decompress_array(','def find_work_res_separation_threshold(','def find_full_res_separation_threshold(','def find_separation_threshold(','def refine_threshold('):
        assert name in TEXT


def test_dead_edge_profile_chunking_is_gone():
    block=TEXT.split('def _sample_edge_profiles(',1)[1].split('\n\ndef ',1)[0]
    assert '15000' not in block
    assert 'profile_chunks' not in block
    assert 'np.vstack' not in block
