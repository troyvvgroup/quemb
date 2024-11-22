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

def test_context_manager():
    my_tmp = Path(mkdtemp())
    assert my_tmp.exists()

    with ScratchManager(my_tmp) as scratch:
        print(scratch)
        pass

    assert not my_tmp.exists()



def test_creation():
    with ScratchManager.from_environment('.') as scratch:
        PID = os.getpid()
        # print(scratch.scratch_area)


if __name__ == "__main__":
    # test_already_created()
    # test_context_manager()
    test_creation()