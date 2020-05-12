# This is a temporary workaround till Poetry supports scripts, see
# https://github.com/sdispater/poetry/issues/241.
from subprocess import call
from typing import Iterable
from random import randint


def warn_call(args: Iterable) -> None:
    exit_status = call(args)
    if exit_status:
        print(f'Call to {args[0]} ended with exit code {exit_status}.')


def format() -> None:
    warn_call(["black", "vibrant/", "tests/"])
    warn_call(["isort", "--recursive", "--apply", "vibrant/", "tests/"])


def lint() -> None:
    warn_call(["pydocstyle", "vibrant/", "tests/"])
    warn_call(["pylint", "vibrant/", "tests/", "--rcfile=.pylint.ini"])
    warn_call(["mypy", "vibrant/", "tests/", "--config-file=.mypy.ini"])


def start() -> None:
    # warn_call(["python", "src/backend/run.py"])
    pass


def test() -> None:
    warn_call(
        [
            "pytest",
            "tests/",
            "--random",
            f"--random-seed={randint(0,1e5)}",
            "--cov=vibrant",
        ]
    )
    warn_call(["coveragespace", "gcapu/vibrant", "overall"])


def retest() -> None:
    warn_call(
        ["pytest", "tests/", "--last-failed", "--exitfirst", "--cov=vibrant",]
    )


def updatedocs() -> None:
    warn_call(["mkdir", "-p", "docs/about"])
    warn_call(["cp", "README.md", "docs/index.md"])
    warn_call(["cp", "CONTRIBUTING.md", "docs/about/contributing.md"])
    warn_call(["cp", "CHANGELOG.md", "docs/about/changelog.md"])
    warn_call(["cp", "LICENSE.md", "docs/about/license.md"])


def servedocs() -> None:
    updatedocs()
    warn_call(["mkdocs", "serve", "--strict"])


def docs() -> None:
    updatedocs()
    warn_call(["mkdocs", "build", "--strict"])


def clean() -> None:
    # wildcards need shell=True
    call("rm -rf .cache .coverage htmlcov *.egg-info site *.dot", shell=True)
    call(
        "find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete",
        shell=True,
    )
