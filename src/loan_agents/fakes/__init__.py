"""Stand-ins for every port, for tests and `--fake` runs. No network, no model, no database."""

from loan_agents.fakes.crm import InMemoryCrm
from loan_agents.fakes.extractor import FakeExtractor
from loan_agents.fakes.messenger import RecordingMessenger
from loan_agents.fakes.stores import InMemoryAuditLog, InMemoryChatCases, InMemoryDocumentStore

__all__ = [
    "FakeExtractor",
    "InMemoryAuditLog",
    "InMemoryChatCases",
    "InMemoryCrm",
    "InMemoryDocumentStore",
    "RecordingMessenger",
]
