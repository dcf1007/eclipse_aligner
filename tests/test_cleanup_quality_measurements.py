import cv2
import numpy as np
import circle_arc_detector as cad


def sample():
    raw=np.zeros((101,101),bool)
    cv2.circle(raw.view(np.uint8),(50,50),30,1,-1)
    raw[47:54,80:93]=True
    raw[42:47,44:50]=False
    return raw,(50,50)


def test_raw_candidate_is_zero_reference_for_every_measurement():
    raw,seed=sample()
    rows=cad.evaluate_cleanup_candidates(raw,seed,12)
    assert rows[0].name == 'raw'
    m=rows[0].metrics
    assert (m.contour_cleanup,m.roughness_cleanup,m.solidity_gain,
            m.internal_dark_cleanup,m.area_loss,m.solidity_loss) == (0,0,0,0,0,0)


def test_measurements_are_computed_against_common_raw_descriptor():
    raw,seed=sample(); rows=cad.evaluate_cleanup_candidates(raw,seed,12)
    base=rows[0].topology
    for row in rows[1:]:
        m=row.metrics; d=row.topology
        assert np.isclose(m.contour_cleanup,max(0,1-d.contour_n/max(float(base.contour_n),1)))
        assert np.isclose(m.roughness_cleanup,max(0,1-d.roughness/max(base.roughness,1e-12)))
        assert np.isclose(m.internal_dark_cleanup,max(0,base.internal_dark_fraction-d.internal_dark_fraction))
        assert np.isclose(m.area_loss,max(0,1-d.area/max(float(base.area),1)))
        assert np.isclose(m.solidity_loss,max(0,(base.solidity-d.solidity)/max(base.solidity,1e-12)))


def test_evaluator_has_no_aggregate_score_or_winner_field():
    raw,seed=sample(); rows=cad.evaluate_cleanup_candidates(raw,seed,12)
    assert not hasattr(rows[0].metrics,'score')
    assert not hasattr(rows[0],'score')
    assert not hasattr(rows[0],'selected')
