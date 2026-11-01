import pytest

from evolution.app import cli


def test_watch_is_available_to_game_py_only(tmp_path):
    args = cli.parse(cli.parser(watch_option=True), ["--watch"], save_dir=str(tmp_path))
    assert args.watch is True
    with pytest.raises(SystemExit):
        cli.parse(cli.parser(), ["--watch"], save_dir=str(tmp_path))


@pytest.mark.parametrize("conflict", [["--new"], ["--load", "1"], ["--list-brains"]])
def test_watch_excludes_the_starting_source_options(tmp_path, conflict):
    with pytest.raises(SystemExit):
        cli.parse(cli.parser(watch_option=True), ["--watch", *conflict], save_dir=str(tmp_path))


def test_workers_is_parsed_for_both_scripts(tmp_path):
    for p in (cli.parser(watch_option=True), cli.parser(worker_option=True)):
        assert cli.parse(p, ["--workers", "3"], save_dir=str(tmp_path)).workers == 3
        assert cli.parse(p, [], save_dir=str(tmp_path)).workers is None


@pytest.mark.parametrize("argv", [["--workers", "-1"], ["--workers", "2", "--worker-id", "0"]])
def test_workers_rejects_bad_values_and_worker_processes(tmp_path, argv):
    with pytest.raises(SystemExit):
        cli.parse(cli.parser(worker_option=True), argv, save_dir=str(tmp_path))


def test_workers_excludes_watch(tmp_path):
    with pytest.raises(SystemExit):
        cli.parse(cli.parser(watch_option=True), ["--watch", "--workers", "2"],
                  save_dir=str(tmp_path))
