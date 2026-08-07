from types import SimpleNamespace

from ctx_to_lora.trainer import StopAfterGlobalStepCallback


def test_curriculum_boundary_stops_and_saves_at_global_step():
    callback = StopAfterGlobalStepCallback(80)
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    callback.on_step_end(None, SimpleNamespace(global_step=79), control)
    assert not control.should_save
    assert not control.should_training_stop
    callback.on_step_end(None, SimpleNamespace(global_step=80), control)
    assert control.should_save
    assert control.should_training_stop
