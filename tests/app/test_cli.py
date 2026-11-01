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
