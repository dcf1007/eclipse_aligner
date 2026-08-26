# Exact GUI focus-release logic used by circle_arc_detector.py.
# A clicked Scale keeps keyboard focus for arrow-key adjustment. The root/toplevel
# binding releases that focus on the next click anywhere outside that Scale.

root.bind("<ButtonPress-1>", self.release_scale_focus_if_outside, add="+")

@staticmethod
def focus_scale(event):
    """Keep a clicked slider focused so arrow keys continue to adjust it."""
    event.widget.focus_set()

def release_scale_focus_if_outside(self, event):
    """Release slider focus immediately when the mouse clicks elsewhere."""
    focused = self.root.focus_get()
    if isinstance(focused, tk.Scale) and event.widget is not focused:
        self.root.focus_set()
