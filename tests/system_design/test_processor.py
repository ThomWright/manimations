import pytest

from system_design.processor import Processor, RetryPolicy


class TestClient:
    # TODO: Make this test less flaky
    def test_num_new_reqs(self):
        req_rate = 2.0
        dt = 0.06666666666666665

        processor = Processor(req_rate=req_rate)

        reqs = 0
        time = 0.0
        while time < 10.0:
            reqs += processor._num_new_reqs(dt)
            time += dt

        # Check that we get approximately 20 requests (20 = 2.0 requests/sec * 10 seconds)
        assert reqs == pytest.approx(20, rel=0.5), (
            "Expected ~20 requests to be created in 10 seconds"
        )


class TestRetryPolicy:
    def test_retry_policy(self):
        retry_policy = RetryPolicy(3, 0.1, 0)

        assert retry_policy.get_retry_interval(1) == 0.1, (
            "First retry interval should be 0.1"
        )
        assert retry_policy.get_retry_interval(2) == 0.2, (
            "Second retry interval should be 0.2 (doubled)"
        )
        assert retry_policy.get_retry_interval(3) == 0.4, (
            "Third retry interval should be 0.4 (doubled again)"
        )
        assert retry_policy.get_retry_interval(4) is None, (
            "Fourth retry should return None (no more retries allowed)"
        )
        with pytest.raises(
            ValueError, match="Retry count must be 1 or greater, got 0."
        ):
            retry_policy.get_retry_interval(0)
