"""Deterministic agent-style signal aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AgentDecisionConfig:
    """Risk-rule configuration for deterministic agent decisions."""

    min_long_votes: int = 2
    volatility_limit: float | None = None
    max_drawdown_limit: float = -0.08


class AgentDecisionEngine:
    """Combine model votes with explicit risk controls."""

    def __init__(self, config: AgentDecisionConfig | None = None) -> None:
        self.config = config or AgentDecisionConfig()

    def decide(
        self,
        signals: pd.DataFrame,
        rolling_volatility: pd.Series,
        rolling_drawdown: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        """Return agent long/flat signals and textual rationales."""
        aligned_signals = signals.fillna(0.0).astype(float)
        vol = rolling_volatility.reindex(aligned_signals.index).ffill()
        drawdown = rolling_drawdown.reindex(aligned_signals.index).ffill().fillna(0.0)

        votes = (aligned_signals > 0.0).sum(axis=1)
        vol_limit = self.config.volatility_limit
        volatility_ok = pd.Series(True, index=aligned_signals.index)
        if vol_limit is not None:
            volatility_ok = vol <= vol_limit
        drawdown_ok = drawdown >= self.config.max_drawdown_limit
        long_signal = (votes >= self.config.min_long_votes) & volatility_ok & drawdown_ok

        rationales = []
        for timestamp in aligned_signals.index:
            parts = [f"votes={int(votes.loc[timestamp])}/{aligned_signals.shape[1]}"]
            if vol_limit is not None and not bool(volatility_ok.loc[timestamp]):
                parts.append("blocked_by_volatility")
            if not bool(drawdown_ok.loc[timestamp]):
                parts.append("blocked_by_drawdown")
            if bool(long_signal.loc[timestamp]):
                parts.append("decision=LONG")
            else:
                parts.append("decision=FLAT")
            rationales.append("; ".join(parts))

        signal = long_signal.astype(float)
        signal.name = "agent_signal"
        rationale = pd.Series(rationales, index=aligned_signals.index, name="agent_rationale")
        return signal, rationale
