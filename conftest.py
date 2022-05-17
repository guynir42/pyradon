import pytest

import numpy as np
from timeit import default_timer as timer

import simulator
import finder


@pytest.fixture
def f():
    f = finder.Finder()
    return f


@pytest.fixture
def sim():
    s = simulator.Simulator()
    return s
