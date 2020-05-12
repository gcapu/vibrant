from vibrant import __version__
from vibrant.hello import say_hello


def test_version():
    assert __version__ == '0.0.2'


def test_hello():
    say_hello()
