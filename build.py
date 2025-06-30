import subprocess


def finalize_build(metadata):  # noqa: ARG001
    """
    Called after the wheel is built but before packaging.
    """
    print("Generating pybind11 stubs...")

    # Adjust this to your actual import path
    subprocess.run(
        [
            "pybind11-stubgen",
            "quemb.molbe._cpp.eri_sparse_DF",
            "-o",
            "quemb/src/quemb/molbe/_cpp/",
        ],
        check=True,
    )

    # # Optionally: touch py.typed
    # (Path("quemb") / "py.typed").touch()
