import numpy as np

import circle_arc_detector as cad


def test_calculate_work_res_shape_preserves_numpy_orientation_and_aspect():
    assert cad.calculate_work_res_shape((2400, 3600)) == (800, 1200)
    assert cad.calculate_work_res_shape((3600, 2400)) == (1200, 800)
    assert cad.calculate_work_res_shape((700, 1100)) == (700, 1100)


def test_resize_img_uses_numpy_height_width_order_for_asymmetric_shapes():
    source = np.arange(15, dtype=np.uint8).reshape(3, 5)
    resized = cad.resize_img(source, (7, 11))
    assert resized.shape == (7, 11)




def test_resize_img_bool_as_ordinary_raster_returns_uint8():
    source = np.array(
        [
            [False, True, False],
            [True, True, False],
        ],
        dtype=bool,
    )
    resized = cad.resize_img(source, source.shape)
    assert resized.dtype == np.uint8
    assert np.array_equal(resized, source.astype(np.uint8) * 255)


def test_resize_img_bool_as_ordinary_raster_allows_smoothed_display_values():
    source = np.zeros((2, 2), dtype=bool)
    source[0, 0] = True
    resized = cad.resize_img(source, (5, 5))
    assert resized.dtype == np.uint8
    assert resized.shape == (5, 5)
    assert np.any((resized > 0) & (resized < 255))


def test_resize_img_mask_shape_order_is_asymmetric_and_exact():
    source = np.zeros((3, 5), bool)
    source[1, 2] = True
    resized = cad.resize_img(source, (9, 15), mask=True)
    assert resized.dtype == bool
    assert resized.shape == (9, 15)


def test_generate_kernel_uses_numpy_height_width_order():
    kernel = cad.generate_kernel((3, 5), round_kernel=False)
    assert kernel.shape == (3, 5)
    assert np.all(kernel == 1)
