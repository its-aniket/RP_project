import datetime

class CostTracker:
    def __init__(self, cost_per_million_tokens_inr):
        self.cost_rate = cost_per_million_tokens_inr / 1_000_000
        self.records = []

    def record(self, agent_name, tokens_in, tokens_out):
        total = tokens_in + tokens_out
        cost = total * self.cost_rate

        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "agent": agent_name,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "total_tokens": total,
            "cost_in_inr": round(cost, 6),
        }

        self.records.append(entry)
        print(f"[COST] {agent_name} → ₹{entry['cost_in_inr']} for {total} tokens")

    def summary(self):
        total_cost = sum(r["cost_in_inr"] for r in self.records)
        return {
            "total_cost_in_inr": round(total_cost, 4),
            "total_calls": len(self.records),
            "entries": self.records
        }
