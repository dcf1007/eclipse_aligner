import inspect
import cv2
import numpy as np
import pytest
import circle_arc_detector as cad


def test_extract_component_returns_only_seeded_eight_component():
    mask=np.zeros((20,30),bool); mask[2:9,2:9]=True; mask[10:18,18:27]=True
    first=cad.extract_component(mask,(4,4)); second=cad.extract_component(mask,(22,14))
    assert first is not None and second is not None
    assert np.count_nonzero(first)==49 and np.count_nonzero(second)==72 and not np.any(first&second)
    assert cad.extract_component(mask,(15,5)) is None
    with pytest.raises(ValueError): cad.extract_component(mask,(30,5))


def test_clean_solar_component_runs_progressive_cleanup_guard_then_extract():
    mask=np.zeros((80,100),bool); mask[15:55,10:50]=True; mask[20:50,70:95]=True; mask[5,5]=True; seed=(30,35)
    guard=np.ones_like(mask,bool); guard[:,90:]=False
    cleaned=cad.clean_solar_component(mask,seed,guard); assert cleaned is not None
    expected=np.where(mask,255,0).astype(np.uint8)
    for kernel in cad.SOLAR_CLEANUP_KERNELS:
        expected=cv2.morphologyEx(expected,cv2.MORPH_OPEN,kernel,iterations=1)
        expected=cv2.morphologyEx(expected,cv2.MORPH_CLOSE,kernel,iterations=1)
    expected[~guard]=0; expected=cad.extract_component(expected,seed)
    assert expected is not None and np.array_equal(cleaned,expected)


def test_measurements_use_filled_external_silhouette_not_holes():
    solid=np.zeros((11,11),bool); solid[3:8,3:8]=True; contour=cad.find_external_contour(solid)
    filled=cad.measure_filled_area(contour); rough=cad.measure_roughness(contour,filled); solidity=cad.measure_solidity(contour,filled)
    assert filled==25 and rough==pytest.approx(16/(2*np.sqrt(np.pi*25))) and solidity==pytest.approx(1.0)
    holed=solid.copy(); holed[5,5]=False; hc=cad.find_external_contour(holed); hf=cad.measure_filled_area(hc)
    assert hf==filled and cad.measure_roughness(hc,hf)==pytest.approx(rough) and cad.measure_solidity(hc,hf)==pytest.approx(solidity)
    assert cad.measure_internal_dark_fraction(int(np.count_nonzero(holed)),hf)==pytest.approx(1/25)


def test_refinement_uses_full_masks_and_only_one_raw_reference(monkeypatch):
    gray=np.zeros((96,112),np.uint8); cv2.circle(gray,(56,48),24,100,-1)
    guard=np.zeros_like(gray,bool); cv2.circle(guard.view(np.uint8),(56,48),35,1,-1); seed=(56,48)
    original_extract=cad.extract_component; original_clean=cad.clean_solar_component; extract_shapes=[]; clean_shapes=[]
    def count_extract(mask,point): extract_shapes.append(np.asarray(mask).shape); return original_extract(mask,point)
    def record_clean(mask,point,g): clean_shapes.append(np.asarray(mask).shape); return original_clean(mask,point,g)
    monkeypatch.setattr(cad,'extract_component',count_extract); monkeypatch.setattr(cad,'clean_solar_component',record_clean)
    monkeypatch.setattr(cad,'measure_edge_alignment',lambda *_a:(0.5,1.0,1.0))
    result=cad.refine_threshold(gray,10,seed,guard,max_steps=2)
    assert [r.threshold for r in result.trajectory]==[10,11,12]
    assert result.raw_reference_threshold==10 and clean_shapes==[gray.shape]*3 and extract_shapes==[gray.shape]*4
    restored=cad.decompress_full_mask(result.cleaned_component_mask,gray.shape)
    expected=original_clean(cv2.compare(gray,result.threshold,cv2.CMP_GT),seed,guard)
    assert expected is not None and np.array_equal(restored,expected)


def test_coarse_and_refinement_apply_guard_before_common_extraction(monkeypatch):
    gray=np.zeros((70,90),np.uint8); cv2.circle(gray,(45,35),18,80,-1)
    guard=np.zeros_like(gray,bool); cv2.circle(guard.view(np.uint8),(45,35),28,1,-1); seed=(45,35)
    original=cad.extract_component; calls=[]
    def record(mask,point): calls.append(np.asarray(mask).copy()); return original(mask,point)
    monkeypatch.setattr(cad,'extract_component',record)
    cad.find_lowest_full_res_threshold(gray,10,seed,guard); assert calls and all(not np.any(m[~guard]) for m in calls)
    calls.clear(); assert cad.clean_solar_component(cv2.compare(gray,10,cv2.CMP_GT),seed,guard) is not None
    assert len(calls)==1 and not np.any(calls[0][~guard])


def test_default_search_has_ten_upward_steps_and_exact_score_decision():
    assert cad.MAX_T_REFINEMENT_STEPS==10
    source=inspect.getsource(cad.refine_threshold)
    assert 'round(' not in source and 'decision_score' not in source and 'epsilon' not in source.lower() and 'plateau' not in source.lower()
