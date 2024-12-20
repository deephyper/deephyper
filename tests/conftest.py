import pytest


# -- Control skipping of tests according to command line option
def pytest_addoption(parser):
    parser.addoption(
        "--run",
        default="fast",
        help="Select tests to run.",
    )


def pytest_collection_modifyitems(config, items):

    selected_marks = set(config.getoption("--run").split(","))
    get_markers = lambda item: {m.name for m in item.iter_markers()}

    skip_mark = pytest.mark.skip(reason=f"need --run with compatible marks to run")
    for item in items:
        item_marks = get_markers(item)
        if not (item_marks.issubset(selected_marks)):
            item.add_marker(skip_mark)


# -- Incremental testing - test steps
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
