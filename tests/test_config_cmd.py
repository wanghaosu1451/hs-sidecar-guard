"""终端通用 JSON 直配的取值解析测试。"""
from cli import _parse_cfg_value


def test_int():
    assert _parse_cfg_value("120") == 120
    assert _parse_cfg_value("-3") == -3


def test_float():
    v = _parse_cfg_value("0.9")
    assert isinstance(v, float) and abs(v - 0.9) < 1e-9


def test_bool():
    assert _parse_cfg_value("true") is True
    assert _parse_cfg_value("false") is False


def test_null():
    assert _parse_cfg_value("null") is None


def test_json_array():
    assert _parse_cfg_value("[1, 2, 3]") == [1, 2, 3]


def test_json_object():
    assert _parse_cfg_value('{"a": 1}') == {"a": 1}


def test_plain_string():
    assert _parse_cfg_value("hello world") == "hello world"


def test_ipv6_string_not_parsed_as_number():
    # 冒号字符串应保持字符串，不被误判为数字
    assert isinstance(_parse_cfg_value("http://[2408::1]:11435/v1"), str)