"""Live suite: huggingface-hub dependency (endpoint-gated, tiny IDs only).

Dependency: huggingface-hub. Use case: hub-fetch-tiny.
Every test skips with a recorded reason when the Hub is unreachable —
never forced, never with a real token, never a gated weight.
"""

import pytest

PRIMARY = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BOGUS = "airllm-lab-nonexistent-repo-xyz-123/does-not-exist-abc"

FEATURE = "live-hub"


def _skip_offline(exc):
    pytest.skip("offline or Hub unreachable, hub-fetch-tiny not proven: %r" % (exc,))


@pytest.mark.scenario("hub-fetch-tiny")
@pytest.mark.dep("hub-fetch-tiny")
def test_hub_fetch_tiny_metadata():
    """hub-fetch-tiny: PRIMARY tiny metadata resolves unauthenticated."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(PRIMARY)
    except Exception as exc:  # noqa: BLE001 - any transport/auth failure -> skip
        _skip_offline(exc)
    assert info.id == PRIMARY


@pytest.mark.scenario("hub-fetch-tiny")
@pytest.mark.dep("hub-fetch-tiny")
@pytest.mark.negative("unknown repo id fails instead of returning a guess")
def test_hub_fetch_unknown_repo_negative():
    """hub-fetch-tiny negative: bogus id raises (404), never resolves."""
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import RepositoryNotFoundError
    except Exception as exc:  # noqa: BLE001
        _skip_offline(exc)
    try:
        HfApi().model_info(BOGUS)
    except RepositoryNotFoundError:
        # Unauthenticated Hub answers 401 for unknown ids; the typed error
        # is the not-found signal (never resolves to weights).
        return
    except Exception as exc:  # noqa: BLE001 - offline shows up here too
        _skip_offline(exc)
    raise AssertionError("bogus repo id unexpectedly resolved")
