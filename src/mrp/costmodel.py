"""Next-call cost estimate C_hat(m) and the asymmetric cache-aware gate (P7/P8).

C_hat(m) = prefill(m, uncached u, cached c)  +  g_hat(m) * tpot(m, context)
  - prefill already contains the switch cost (recomputing history the target model has not cached),
    so no separate switch term is added.
  - transfer/restore cost = 0: weights resident, no CPU offload, no KV transfer in this pilot.
  - module cost: the router call happens before the gate in both branches, so it does not differ
    between stay/switch and is not added; the gate's own time is measured and logged.
  - g_hat(m) = mean completion tokens of model m on dev runs (never the real future length).

Fit form (per model, from profiles/ B1 measurements only):
  prefill(u, c) = a + b*u + k*u*(c + u/2)     [attention term, as in ~/switchcost P0]
  tpot(ctx)     = d0 + d1*ctx
Gate: A->B recommendation is always followed. B->A is followed only if C_hat(B) - C_hat(A) > MARGIN_S.
"""
import json
import time

MARGIN_S = 0.25  # fixed before evaluation


class CostModel:
    def __init__(self, fit):
        self.fit = fit

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path)))

    def prefill(self, m, u, c):
        f = self.fit[m]["prefill"]
        return f["a"] + f["b"] * u + f["k"] * u * (c + u / 2)

    def tpot(self, m, ctx):
        f = self.fit[m]["tpot"]
        return f["d0"] + f["d1"] * ctx

    def next_call(self, m, n_prompt, n_cached):
        g = self.fit[m]["g_hat"]
        u = n_prompt - n_cached
        pre = self.prefill(m, u, n_cached)
        dec = g * self.tpot(m, n_prompt + g / 2)
        return {"model": m, "uncached": u, "cached": n_cached, "prefill_s": pre, "decode_s": dec,
                "total_s": pre + dec, "g_hat": g}


def gate(cm, current, recommended, prompt_ids, tracker):
    """Return (chosen_model, record)."""
    t0 = time.perf_counter()
    if current != "B" or recommended != "A":
        return recommended, {"applied": False, "gate_s": time.perf_counter() - t0}
    t_l = time.perf_counter()
    hit = {m: tracker.predicted_hit(m, prompt_ids) for m in ("A", "B")}
    lookup_s = time.perf_counter() - t_l
    est = {m: cm.next_call(m, len(prompt_ids), hit[m]) for m in ("A", "B")}
    saving = est["B"]["total_s"] - est["A"]["total_s"]
    chosen = "A" if saving > MARGIN_S else "B"
    return chosen, {"applied": True, "est": est, "saving_s": saving, "margin_s": MARGIN_S,
                    "overrode_router": chosen != recommended, "lookup_s": lookup_s,
                    "gate_s": time.perf_counter() - t0}
