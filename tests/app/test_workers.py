from types import SimpleNamespace

from evolution.app import workers as workers_module
from evolution.app.workers import WorkerPool, default_worker_count


class _Proc:
    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0


def test_pool_starts_count_single_threaded_workers(tmp_path, monkeypatch):
    launched = []
    monkeypatch.setattr(workers_module, "SAVE_DIR", str(tmp_path))
    monkeypatch.setattr(workers_module, "clear_island_files", lambda: None)
    monkeypatch.setattr(workers_module, "IslandExchange",
                        lambda sim: SimpleNamespace(publish=lambda: None,
                                                    import_if_better=lambda *a: None))
    monkeypatch.setattr(workers_module.subprocess, "Popen",
                        lambda args, **kw: launched.append((args, kw["env"])) or _Proc())
    pool = WorkerPool(sim=None, autosaver=None, count=3)
    pool.start()
    assert [args[-1] for args, _ in launched] == ["0", "1", "2"]
    assert all(env["OMP_NUM_THREADS"] == "1" for _, env in launched)
    pool.stop()
    assert not pool.running


def test_default_worker_count_is_half_the_cores(monkeypatch):
    monkeypatch.setattr(workers_module, "cpu_count", lambda: 12)
    assert default_worker_count() == 6
    monkeypatch.setattr(workers_module, "cpu_count", lambda: 1)
    assert default_worker_count() == 1
