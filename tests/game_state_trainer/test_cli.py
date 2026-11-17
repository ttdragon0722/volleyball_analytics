import pytest

from game_state_trainer.__main__ import main, parse_args


def test_requires_val():
    with pytest.raises(SystemExit):
        parse_args(["--train", "a.mp4"])


def test_requires_train_unless_evaluate():
    with pytest.raises(SystemExit):
        parse_args(["--val", "b.mp4"])
    assert parse_args(["--evaluate", "m", "--val", "b.mp4"]).evaluate == "m"


def test_rejects_same_video_for_train_and_val():
    with pytest.raises(SystemExit):
        parse_args(["--train", "x/a.mp4", "--val", "x/../x/a.mp4"])


def test_rejects_bad_purity():
    with pytest.raises(SystemExit):
        parse_args(["--train", "a.mp4", "--val", "b.mp4", "--min-purity", "0"])


def test_defaults():
    args = parse_args(["--train", "a.mp4", "c.mp4", "--val", "b.mp4"])
    assert args.train == ["a.mp4", "c.mp4"] and args.epochs == 8 and not args.keep_head


def test_missing_db_is_a_clean_error(tmp_path):
    assert main(["--db", str(tmp_path / "none.db"), "--train", "a.mp4", "--val", "b.mp4"]) == 1
