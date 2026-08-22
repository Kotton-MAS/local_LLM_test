import logging

import pytest

from main import main


def test_main_logs_greeting(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        main()
    assert "Hello from template!" in caplog.text
