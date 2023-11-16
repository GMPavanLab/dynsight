import dynsight


def test_foo1() -> None:
    assert dynsight.hdf5er.foo1(3) == "Hello, World! Also: 3"


def test_foo2() -> None:
    assert dynsight.soapify.foo3(11) == "Hello, World! Also: 11"


def test_foo3() -> None:
    assert dynsight.lens.foo2(13) == "Hello, World! Also: 13"
