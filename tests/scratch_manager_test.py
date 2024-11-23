import os
from pathlib import Path
from tempfile import mkdtemp

from pytest import raises

from general.manage_scratch import ScratchManager


def test_already_created():
    my_tmp = Path(mkdtemp())
    assert my_tmp.exists()

    scratch = ScratchManager(my_tmp)

    scratch.cleanup()
    assert not my_tmp.exists()

    with raises(FileNotFoundError):
        scratch.cleanup()


def test_keep_upon_error():
    my_tmp = Path(mkdtemp())
    assert my_tmp.exists()

    with raises(ValueError):
        with ScratchManager(my_tmp):
            raise ValueError
    assert my_tmp.exists()

    with ScratchManager(my_tmp):
        pass
    assert not my_tmp.exists()


def test_already_created_non_empty():
    my_tmp = Path(mkdtemp())

    assert my_tmp.exists()

    testfile = my_tmp / "testfile"
    testfile.touch()
    with raises(ValueError):
        with ScratchManager(my_tmp):
            pass

    assert my_tmp.exists()


def test_context_manager():
    my_tmp = Path(mkdtemp())
    assert my_tmp.exists()

    with ScratchManager(my_tmp):
        pass

    assert not my_tmp.exists()


def test_creation_user_defined():
    test_dir = Path("./scratch_test")
    with ScratchManager.from_environment(user_defined_name="./scratch_test") as scratch:
        assert test_dir.exists()
        assert scratch.scratch_area == test_dir
    assert not test_dir.exists()


def test_creation_PID():
    PID = os.getpid()
    with ScratchManager(scratch_area=Path("./scratch_root")) as scratch_root:
        tmp_root = scratch_root.scratch_area
        with ScratchManager.from_environment(user_defined_root=tmp_root) as dir:
            assert dir.scratch_area == tmp_root / f"QuEmb_{PID}"
            with ScratchManager.from_environment(user_defined_root=tmp_root) as dir_2:
                assert dir_2.scratch_area == tmp_root / f"QuEmb_{PID + 1}"
