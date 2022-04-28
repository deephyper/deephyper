import pytest


#-- Control skipping of tests according to command line option
def pytest_addoption(parser):
    parser.addoption(
        "--run-hps", action="store_true", default=False, help="Run HPS tests."
    )
    parser.addoption(
        "--run-nas", action="store_true", default=False, help="Run NAS tests."
    )
    parser.addoption(
        "--run-fast", action="store_true", default=False, help="Run fast tests."
    )
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="Run slow tests."
    )
    parser.addoption(
        "--run-ray", action="store_true", default=False, help="Run tests which require Ray."
    )
    parser.addoption(
        "--run-mpi4py", action="store_true", default=False, help="Run tests which require mpi4py."
    )


def pytest_collection_modifyitems(config, items):

    run_fast = config.getoption(f"--run-fast")
    run_slow = config.getoption(f"--run-slow")

    marks = ["hps", "nas", "ray", "mpi4py"]
    for mark in marks:
        if not(config.getoption(f"--run-{mark}")):
            skip_mark = pytest.mark.skip(reason=f"need --run-{mark} option to run")
            for item in items:
                if mark in item.keywords:
                    item.add_marker(skip_mark)
        else:
            for item in items:
                if not(run_fast) and "fast" in item.keywords:
                    item.add_marker(skip_mark)
                elif not(run_slow) and "slow" in item.keywords:
                    item.add_marker(skip_mark)
            

#-- Incremental testing - test steps
def pytest_runtest_makereport(item, call):
    if "incremental" in item.keywords:
        if call.excinfo is not None:
            parent = item.parent
            parent._previousfailed = item

def pytest_runtest_setup(item):
    if "incremental" in item.keywords:
        previousfailed = getattr(item.parent, "_previousfailed", None)
        if previousfailed is not None:
            pytest.xfail("previous test failed (%s)" % previousfailed.name)