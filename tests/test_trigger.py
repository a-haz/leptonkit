import numpy as np

from leptonkit import y16
from leptonkit.trigger import MotionGate, TriggerSettings


def room(static_warm=True):
    c = np.full((120, 160), 21.0)
    if static_warm:
        c[90:110, 130:150] = 33.0  # a warm PC case: in the body band but never moving
    return c


def person_at(x, static_warm=True):
    c = room(static_warm)
    if x is not None:
        c[30:100, x:x + 20] = 34.0
    return y16.celsius_to_raw(c)


def run(gate, frames, t0=0.0, dt=0.11, can_start=True):
    events, notes = [], []
    for i, f in enumerate(frames):
        status, done = gate.push(f, t0 + i * dt, can_start=can_start)
        notes.append(status["note"])
        if done is not None:
            events.append(done)
    return events, notes


def walk(n_empty_before, n_walk, n_empty_after, step=3):
    seq = [person_at(None)] * n_empty_before
    seq += [person_at(5 + (step * i) % 130) for i in range(n_walk)]
    seq += [person_at(None)] * n_empty_after
    return seq


def test_static_warm_object_never_triggers():
    gate = MotionGate()
    events, notes = run(gate, [person_at(None)] * 300)
    assert events == []
    assert "trigger" not in notes
    assert not gate.in_event


def test_walking_person_gives_one_event_with_preroll():
    s = TriggerSettings()
    gate = MotionGate(s)
    frames = walk(60, 100, 80)
    events, notes = run(gate, frames)
    assert notes.count("trigger") == 1
    assert len(events) == 1
    ev = events[0]
    assert ev.capped is False
    assert ev.peak_motion > s.motion_trigger
    # the person appears as new warm pixels at once, so the trigger fires on
    # the trigger_frames-th walking frame; the event holds the pre-roll, the
    # rest of the walk and the quiet tail that ended it
    first_walk = 60
    trigger_idx = notes.index("trigger")
    assert trigger_idx == first_walk + s.trigger_frames - 1
    assert len(ev.frames) == s.preroll + (first_walk + 100 - 1 - trigger_idx) + s.quiet_frames
    assert ev.stamps == sorted(ev.stamps)
    assert len(ev.stamps) == len(ev.frames) == len(ev.hots)
    assert not gate.in_event


def test_short_blip_is_discarded():
    gate = MotionGate(TriggerSettings(min_event=30))
    events, notes = run(gate, walk(60, 14, 80))
    assert events == []
    assert "discarded" in notes


def test_cap_then_rearm_after_quiet():
    s = TriggerSettings(max_event=120, rearm_quiet=20)
    gate = MotionGate(s)
    events, notes = run(gate, walk(40, 400, 0))
    assert events and events[0].capped
    assert len(events[0].frames) == s.max_event
    assert "capped" in notes
    # while the person keeps moving, the gate stays disarmed
    cap_idx = notes.index("capped")
    assert "trigger" not in notes[cap_idx + 1:]
    # quiet scene, then a new walk triggers again
    more, notes2 = run(gate, [person_at(None)] * 40 + walk(0, 60, 60), t0=1000.0)
    assert notes2.count("trigger") == 1
    assert len(more) == 1


def test_blocked_when_no_space():
    gate = MotionGate()
    events, notes = run(gate, walk(60, 100, 80), can_start=False)
    assert events == []
    assert "blocked" in notes
    assert "trigger" not in notes


def test_flush_returns_running_event_and_resets():
    gate = MotionGate()
    run(gate, walk(60, 40, 0))
    assert gate.in_event
    ev = gate.flush()
    assert ev is not None and len(ev.frames) > 0
    assert not gate.in_event and gate.warm == 0


def test_body_band_in_centikelvin():
    gate = MotionGate(TriggerSettings(body_lo_c=28.5, body_hi_c=38.0))
    assert gate.raw_lo == int((28.5 + 273.15) * 100)
    assert gate.raw_hi == int((38.0 + 273.15) * 100)


def test_recorder_saves_event_files(tmp_path):
    import json

    from leptonkit.trigger import Event, Recorder

    rec = Recorder(tmp_path, live_port=0)
    frames = [person_at(10 + i) for i in range(12)]
    ev = Event(frames=frames, stamps=[1_800_000_000.0 + 0.11 * i for i in range(12)],
               hots=[1400] * 12, peak_motion=900)
    d = rec.save(ev)
    stack = np.load(f"{d}/y16_stack.npy")
    assert stack.shape == (12, 120, 160) and stack.dtype == np.uint16
    meta = json.loads(open(f"{d}/meta.json").read())
    assert meta["n_frames"] == 12
    assert meta["peak_temp_c"] == 34.0
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["dir"] == d
    assert "event saved" in (tmp_path / "recorder.log").read_text()


def test_live_view_renders_jpeg():
    from leptonkit.trigger import LiveView

    view = LiveView()
    assert view.jpeg("grid") is None
    view.publish(person_at(40), True, 1400, 600)
    for cmap in ("grid", "inferno", "whitehot", "blackhot"):
        jpg = view.jpeg(cmap)
        assert jpg[:2] == b"\xff\xd8"
    assert b"/stream?cmap=grid" in view.page()
