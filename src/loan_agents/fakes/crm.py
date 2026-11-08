from __future__ import annotations

from dataclasses import dataclass, field

from loan_agents.contracts import CrmDeal, CrmRefs, DealStage


@dataclass
class FakeDeal:
    deal: CrmDeal
    contact_id: str
    deal_id: str
    stage: DealStage
    comments: list[str] = field(default_factory=list)


class InMemoryCrm:
    """Keyed by case id, like the EspoCRM adapter's find-or-create by the case field."""

    def __init__(self) -> None:
        self.deals: dict[str, FakeDeal] = {}
        self.calls: list[str] = []
        self._next_id = 100

    def by_deal_id(self, deal_id: str) -> FakeDeal:
        for stored in self.deals.values():
            if stored.deal_id == deal_id:
                return stored
        raise KeyError(f"no deal {deal_id}")

    async def upsert_deal(self, deal: CrmDeal) -> CrmRefs:
        self.calls.append("upsert_deal")
        stored = self.deals.get(deal.case_id)
        if stored is None:
            self._next_id += 1
            stored = FakeDeal(
                deal=deal,
                contact_id=str(self._next_id),
                deal_id=str(self._next_id + 1000),
                stage=deal.stage,
            )
            self.deals[deal.case_id] = stored
        else:
            stored.deal = deal
            stored.stage = deal.stage
        return CrmRefs(contact_id=stored.contact_id, deal_id=stored.deal_id)

    async def add_comment(self, deal_id: str, text: str) -> None:
        self.calls.append("add_comment")
        self.by_deal_id(deal_id).comments.append(text)

    async def set_stage(self, deal_id: str, stage: DealStage) -> None:
        self.calls.append("set_stage")
        self.by_deal_id(deal_id).stage = stage
