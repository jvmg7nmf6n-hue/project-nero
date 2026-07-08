from __future__ import annotations

import pandas as pd

from nero_app.core.agents import BrainAgent, MarketAssessmentAgent, VerdictAgent
from nero_app.core.knowledge_store import LocalKnowledgeStore
from nero_app.core.schema import AnalysisRequest, MacroEvent, NeroResult


class NeroOrchestrator:
    def __init__(self, events: list[MacroEvent]) -> None:
        store = LocalKnowledgeStore(events)
        self.brain = BrainAgent(store)
        self.assessment = MarketAssessmentAgent()
        self.verdict = VerdictAgent()

    def run(self, request: AnalysisRequest, prices: pd.DataFrame) -> NeroResult:
        brain_output = self.brain.run(request.headline, asset=request.asset.value)
        assessment_output = self.assessment.run(prices, lookback_days=request.lookback_days)
        verdict_output = self.verdict.run(brain_output, assessment_output)
        return NeroResult(
            request=request,
            brain=brain_output,
            assessment=assessment_output,
            verdict=verdict_output,
        )
