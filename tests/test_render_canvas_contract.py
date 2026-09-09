import weakref

import numpy as np
import pytest

import circle_arc_detector as cad


class Root:
    def __init__(self):
        self.updates = 0
        self.after_jobs = {}
        self.cancelled_jobs = []
        self._next_job = 0

    def update_idletasks(self):
        self.updates += 1

    def after(self, delay, callback, *args):
        self._next_job += 1
        job = f"job-{self._next_job}"
        self.after_jobs[job] = (delay, callback, args)
        return job

    def after_cancel(self, job):
        self.cancelled_jobs.append(job)
        self.after_jobs.pop(job, None)


class Canvas:
    def __init__(self, width=20, height=20):
        self.width = width
        self.height = height
        self.deleted = 0
        self.images = []

    def winfo_width(self):
        return self.width

    def winfo_height(self):
        return self.height

    def delete(self, *args):
        self.deleted += 1

    def create_image(self, *args, **kwargs):
        self.images.append((args, kwargs))


def _app(monkeypatch):
    app = cad.DetectorApp.__new__(cad.DetectorApp)
    app.root = Root()

    class Photo:
        instances = []

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            Photo.instances.append(self)

    monkeypatch.setattr(cad.tk, "PhotoImage", Photo)
    return app, Photo


@pytest.mark.parametrize(
    "content",
    (
        np.zeros((3, 4), bool),
        np.zeros((3, 4), np.uint8),
        np.zeros((3, 4, 3), np.uint8),
        np.zeros((3, 4, 4), np.uint8),
        np.zeros((3, 4), np.uint16),
        np.zeros((3, 4, 3), np.uint16),
        np.zeros((3, 4, 4), np.uint16),
    ),
)
def test_renderer_retains_array_representation_by_reference_and_encodes_png(
    monkeypatch,
    content,
):
    app, Photo = _app(monkeypatch)
    canvas = Canvas()

    app.render_canvas_content(canvas, content)

    assert canvas._rendered_content is content
    assert canvas.images[-1][0] == (0, 0)
    assert canvas.images[-1][1]["anchor"] == "nw"
    assert canvas.images[-1][1]["image"] is canvas._tk_photo_image
    assert Photo.instances[-1].kwargs["format"] == "png"
    assert isinstance(Photo.instances[-1].kwargs["data"], bytes)
    assert Photo.instances[-1].kwargs["data"].startswith(b"\x89PNG\r\n\x1a\n")


def test_renderer_retains_compressed_content_as_the_same_bytes_object(monkeypatch):
    app, _ = _app(monkeypatch)
    canvas = Canvas()
    master = np.zeros((5, 7, 4), np.uint16)
    payload = cad.compress_image(master)

    app.render_canvas_content(canvas, payload)

    assert canvas._rendered_content is payload


def test_renderer_skips_repaint_for_equal_array_pixels_at_same_viewport(monkeypatch):
    app, _ = _app(monkeypatch)
    canvas = Canvas()
    first = np.arange(30, dtype=np.uint8).reshape(5, 6)
    second = first.copy()

    app.render_canvas_content(canvas, first)
    image_count = len(canvas.images)
    app.render_canvas_content(canvas, second)

    assert len(canvas.images) == image_count
    assert canvas._rendered_content is second


def test_renderer_compares_compressed_and_array_content_by_decoded_pixels(monkeypatch):
    app, _ = _app(monkeypatch)
    canvas = Canvas()
    image = np.arange(5 * 7 * 4, dtype=np.uint16).reshape(5, 7, 4)
    payload = cad.compress_image(image)

    app.render_canvas_content(canvas, image)
    image_count = len(canvas.images)
    app.render_canvas_content(canvas, payload)
    assert len(canvas.images) == image_count
    assert canvas._rendered_content is payload

    app.render_canvas_content(canvas, image)
    assert len(canvas.images) == image_count
    assert canvas._rendered_content is image


def test_renderer_releases_replaced_array_before_allocating_new_render(monkeypatch):
    app, _ = _app(monkeypatch)
    canvas = Canvas()
    previous = np.arange(30, dtype=np.uint8).reshape(5, 6)
    app.render_canvas_content(canvas, previous)
    previous_ref = weakref.ref(previous)
    del previous

    replacement = np.arange(30, dtype=np.uint8).reshape(5, 6) + 1
    real_resize = cad.resize_img

    def checked_resize(image, shape, mask=False):
        assert previous_ref() is None
        return real_resize(image, shape, mask=mask)

    monkeypatch.setattr(cad, "resize_img", checked_resize)
    app.render_canvas_content(canvas, replacement)


def test_renderer_repaints_equal_content_when_viewport_changes(monkeypatch):
    app, _ = _app(monkeypatch)
    canvas = Canvas(width=20, height=20)
    content = np.arange(30, dtype=np.uint8).reshape(5, 6)

    app.render_canvas_content(canvas, content)
    image_count = len(canvas.images)
    canvas.width = 30
    app.render_canvas_content(canvas, content)

    assert len(canvas.images) == image_count + 1


def test_none_clears_canvas_without_creating_an_image(monkeypatch):
    app, Photo = _app(monkeypatch)
    canvas = Canvas()
    content = np.ones((4, 5), np.uint8)
    app.render_canvas_content(canvas, content)
    photo_count = len(Photo.instances)

    app.render_canvas_content(canvas, None)

    assert canvas._rendered_content is None
    assert canvas._tk_photo_image is None
    assert len(Photo.instances) == photo_count
    assert canvas.deleted >= 2


def test_renderer_rejects_non_array_non_bytes_content(monkeypatch):
    app, _ = _app(monkeypatch)
    with pytest.raises(ValueError):
        app.render_canvas_content(Canvas(), [[0, 1], [2, 3]])


def test_new_compressed_content_is_decompressed_once_when_comparison_and_render_need_it(
    monkeypatch,
):
    app, _ = _app(monkeypatch)
    canvas = Canvas(width=20, height=20)
    image = np.arange(5 * 7 * 4, dtype=np.uint16).reshape(5, 7, 4)
    app.render_canvas_content(canvas, image)

    payload = cad.compress_image(image)
    canvas.width = 30
    real_decompress = cad.decompress_image
    calls = []

    def counted(payload_arg):
        calls.append(payload_arg)
        return real_decompress(payload_arg)

    monkeypatch.setattr(cad, "decompress_image", counted)
    app.render_canvas_content(canvas, payload)

    assert calls == [payload]
    assert canvas._rendered_content is payload


def test_canvas_resize_waits_for_settle_and_only_renders_latest_request(monkeypatch):
    app, _ = _app(monkeypatch)
    canvas = Canvas(width=20, height=20)
    content = np.arange(30, dtype=np.uint8).reshape(5, 6)
    canvas._rendered_content = content
    calls = []
    app.render_canvas_content = lambda c, r: calls.append((c, r))

    event = type("Event", (), {"widget": canvas})()
    app._handle_canvas_resize(event)
    first_job = canvas._resize_render_job

    assert calls == []
    assert app.root.after_jobs[first_job][0] == cad.CANVAS_RESIZE_SETTLE_MS

    app._handle_canvas_resize(event)
    second_job = canvas._resize_render_job

    assert first_job != second_job
    assert first_job in app.root.cancelled_jobs
    assert calls == []

    _, callback, args = app.root.after_jobs.pop(second_job)
    callback(*args)

    assert canvas._resize_render_job is None
    assert calls == [(canvas, content)]


def test_completed_resize_does_nothing_for_empty_canvas(monkeypatch):
    app, _ = _app(monkeypatch)
    canvas = Canvas()
    calls = []
    app.render_canvas_content = lambda *args: calls.append(args)

    app._finish_canvas_resize(canvas)

    assert canvas._resize_render_job is None
    assert calls == []


def test_renderer_passes_bool_content_to_resize_and_retains_bool(monkeypatch):
    app, _ = _app(monkeypatch)
    canvas = Canvas(width=20, height=20)
    content = np.zeros((5, 6), dtype=bool)
    content[1:4, 2:5] = True
    real_resize = cad.resize_img
    calls = []

    def recorded_resize(image, shape, mask=False):
        calls.append((image, shape, mask))
        return real_resize(image, shape, mask=mask)

    monkeypatch.setattr(cad, "resize_img", recorded_resize)
    app.render_canvas_content(canvas, content)

    assert canvas._rendered_content is content
    assert calls == [(content, (15, 18), False)]


def test_renderer_skips_bool_resize_for_equal_pixels_at_same_viewport(monkeypatch):
    app, _ = _app(monkeypatch)
    canvas = Canvas(width=20, height=20)
    first = np.zeros((5, 6), dtype=bool)
    first[1:4, 2:5] = True
    second = first.copy()
    real_resize = cad.resize_img
    calls = []

    def recorded_resize(image, shape, mask=False):
        calls.append((image, shape, mask))
        return real_resize(image, shape, mask=mask)

    monkeypatch.setattr(cad, "resize_img", recorded_resize)
    app.render_canvas_content(canvas, first)
    calls.clear()
    app.render_canvas_content(canvas, second)

    assert calls == []
    assert canvas._rendered_content is second
