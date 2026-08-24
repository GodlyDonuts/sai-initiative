import threading

import pytest

from sai.data.nous_compiler_worker import (
    HERMES_LOOPBACK_URL,
    NousLabelWorkerError,
    OPENROUTER_URL,
    _shared_provider_concurrency,
    _shared_provider_request_slot,
)


def test_shared_provider_slot_serializes_independent_callers(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("SAI_NOUS_SHARED_PROVIDER_CONCURRENCY", "1")
    monkeypatch.setenv(
        "SAI_NOUS_SHARED_PROVIDER_SLOT_ROOT", str(tmp_path / "provider-slots")
    )
    entered = threading.Event()

    def enter_second_slot() -> None:
        with _shared_provider_request_slot(HERMES_LOOPBACK_URL):
            entered.set()

    with _shared_provider_request_slot(HERMES_LOOPBACK_URL):
        thread = threading.Thread(target=enter_second_slot)
        thread.start()
        assert entered.wait(0.1) is False
    assert entered.wait(1.0) is True
    thread.join(timeout=1.0)
    assert thread.is_alive() is False


def test_shared_provider_slot_is_not_applied_to_other_endpoints(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("SAI_NOUS_SHARED_PROVIDER_CONCURRENCY", "1")
    monkeypatch.setenv(
        "SAI_NOUS_SHARED_PROVIDER_SLOT_ROOT", str(tmp_path / "provider-slots")
    )
    with _shared_provider_request_slot("https://example.test/v1") as slot:
        assert slot is None
    assert (tmp_path / "provider-slots").exists() is False


def test_openrouter_and_gateway_use_independent_admission_slots(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("SAI_NOUS_SHARED_PROVIDER_CONCURRENCY", "1")
    monkeypatch.setenv("SAI_OPENROUTER_SHARED_PROVIDER_CONCURRENCY", "1")
    monkeypatch.setenv(
        "SAI_NOUS_SHARED_PROVIDER_SLOT_ROOT", str(tmp_path / "provider-slots")
    )
    with _shared_provider_request_slot(HERMES_LOOPBACK_URL):
        with _shared_provider_request_slot(OPENROUTER_URL) as slot:
            assert slot == 0
    assert (tmp_path / "provider-slots" / "openrouter" / "slot_000.lock").is_file()


def test_openrouter_uses_conservative_default_limit() -> None:
    assert _shared_provider_concurrency(OPENROUTER_URL) == 6


@pytest.mark.parametrize("value", ["0", "65", "01", "sixteen"])
def test_shared_provider_concurrency_fails_closed(monkeypatch, value) -> None:
    monkeypatch.setenv("SAI_NOUS_SHARED_PROVIDER_CONCURRENCY", value)
    with pytest.raises(NousLabelWorkerError, match="concurrency differs"):
        _shared_provider_concurrency(HERMES_LOOPBACK_URL)


def test_shared_provider_slot_root_rejects_symlink(tmp_path, monkeypatch) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("SAI_NOUS_SHARED_PROVIDER_CONCURRENCY", "1")
    monkeypatch.setenv("SAI_NOUS_SHARED_PROVIDER_SLOT_ROOT", str(link))
    with pytest.raises(NousLabelWorkerError, match="root is unsafe"):
        with _shared_provider_request_slot(HERMES_LOOPBACK_URL):
            pass
