from __future__ import annotations

from loan_agents.contracts import ApprovalRequest


class RecordingMessenger:
    def __init__(self) -> None:
        self.client_messages: list[tuple[str, str]] = []
        self.approval_requests: list[ApprovalRequest] = []

    async def send_to_client(self, case_id: str, texts: list[str]) -> None:
        self.client_messages.extend((case_id, text) for text in texts)

    async def send_approval_request(self, request: ApprovalRequest) -> None:
        self.approval_requests.append(request)
