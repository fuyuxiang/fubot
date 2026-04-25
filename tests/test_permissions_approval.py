from __future__ import annotations

from echo_agent.permissions.manager import ApprovalManager, ApprovalStatus


def test_required_approval_overrides_default_approve() -> None:
    manager = ApprovalManager(require_approval=["exec"], default_policy="approve")

    request = manager.request_approval("exec", tool_name="exec", user_id="u1")

    assert request.status == ApprovalStatus.PENDING
    assert manager.get(request.id) is request


def test_auto_deny_and_default_approve() -> None:
    manager = ApprovalManager(auto_deny=["danger"], default_policy="approve")

    denied = manager.request_approval("danger")
    allowed = manager.request_approval("read_file")

    assert denied.status == ApprovalStatus.DENIED
    assert allowed.status == ApprovalStatus.APPROVED


def test_approved_request_allows_same_call_once() -> None:
    manager = ApprovalManager(require_approval=["exec"], default_policy="approve")
    first = manager.request_approval("exec", tool_name="exec", params={"command": "date"}, user_id="u1")

    assert manager.approve(first.id, decided_by="admin")
    second = manager.request_approval("exec", tool_name="exec", params={"command": "date"}, user_id="u1")
    third = manager.request_approval("exec", tool_name="exec", params={"command": "date"}, user_id="u1")

    assert second.status == ApprovalStatus.APPROVED
    assert third.status == ApprovalStatus.PENDING
