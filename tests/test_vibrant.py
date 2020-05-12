from vibrant import __version__
from vibrant.hello import say_hello


def test_version():
    assert __version__ == '0.1.0'


def test_hello():
    say_hello()
