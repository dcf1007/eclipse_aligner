import argparse
import tkinter as tk

import circle_arc_detector as cad


def build_app():
    root=tk.Tk()
    args=argparse.Namespace(threshold=8,min_radius=1000.0,max_radius=1500.0,max_error=0.08,min_coverage=0.08)
    app=cad.DetectorApp(root,[],args); root.update(); return root,app


def descendants(root):
    pending=[root]
    while pending:
        parent=pending.pop()
        for child in parent.winfo_children(): pending.append(child); yield child


def controls(root):
    types=(tk.Button,tk.Scale,tk.Checkbutton,tk.Radiobutton)
    return [w for w in descendants(root) if isinstance(w,types)]


def test_processing_ui_disables_then_restores_controls():
    root,app=build_app()
    try:
        cs=controls(root); prior=[(w,str(w.cget('state'))) for w in cs]
        with app.processing_ui(): assert all(str(w.cget('state'))==str(tk.DISABLED) for w in cs)
        assert [(w,str(w.cget('state'))) for w,_ in prior]==prior
    finally: root.destroy()


def test_queued_scale_input_is_consumed_while_disabled():
    root,app=build_app()
    try:
        scale=next(w for w in descendants(root) if isinstance(w,tk.Scale)); scale.set(20); root.update(); before=scale.get()
        with app.processing_ui():
            scale.event_generate('<ButtonPress-1>',x=max(scale.winfo_width(),100)-2,y=10,when='tail')
            scale.event_generate('<ButtonRelease-1>',x=max(scale.winfo_width(),100)-2,y=10,when='tail')
        root.update_idletasks(); assert scale.get()==before
    finally: root.destroy()


def test_disabled_button_click_does_not_replay():
    root,app=build_app()
    try:
        fired=[]; app.preview_button.config(command=lambda:fired.append(1),state=tk.NORMAL); root.update()
        with app.processing_ui():
            app.preview_button.event_generate('<ButtonPress-1>',x=5,y=5,when='tail')
            app.preview_button.event_generate('<ButtonRelease-1>',x=5,y=5,when='tail')
        root.update_idletasks(); assert fired==[]
    finally: root.destroy()
