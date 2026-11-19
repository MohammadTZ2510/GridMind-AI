"""
csp_scheduler.py
================
`CSPScheduler` - generates a VALID load-shedding schedule with our own recursive
Backtracking search (no external solver).

CSP formulation
---------------
Variables : x[i,t] for every feeder i and hour t (10 x 24 = 240 variables).
Domains   : {0, 1}   (1 = ON, 0 = OFF).
            Domain is reduced to {1} for the hospital feeder, protected hours and
            feeders that are allowed no outage.

Hard constraints (all enforced during the search)
  1. Hospital feeder F001 is always ON.
  2. Protected hours are ON.
  3. Total OFF hours of a feeder <= max_total_outage_hours.
  4. Consecutive OFF hours of a feeder <= max_consecutive_outage_hours.
  5. Supply capacity: sum of demand of ON feeders <= usable supply, every hour.
  6. Every variable gets exactly one binary value (no missing assignments).

The schedule is planned with the FORECAST demand (the environment's planning demand).

Capacity constraint as a "required shed"
----------------------------------------
    served[t] = total_demand[t] - (demand of OFF feeders)  <=  usable[t]
    <=>  demand_shed[t] >= required[t] = max(0, total_demand[t] - usable[t])

So hour t needs outages that remove AT LEAST required[t] MW.

Search design (easy to explain)
-------------------------------
* Hours with required[t] = 0 need no outage: all their feeders are set ON before the
  search starts. (Switching something OFF there can never help a hard constraint, so
  the search space is reduced without losing any useful solution - "dominance pruning".)
* VARIABLE ORDERING (dynamic Minimum-Remaining-Values): the next "shortage hour" is always
  the one with the smallest CURRENT slack = (demand that can still be shed) - required,
  i.e. the most constrained hour first (recomputed after every hour is finished). Within an hour, feeders are tried in this order:
      1. feeders that have used the smallest fraction of their outage allowance (fairness),
      2. lower priority weight first (protect important feeders),
      3. demand closest to the amount still needed (avoids unnecessary over-shedding).
* VALUE ORDERING: try OFF first while the hour still needs shedding, otherwise ON only.
* DOMINANCE PRUNING: OFF is not allowed once the hour already sheds enough.
* CAPACITY / FORWARD-CHECKING PRUNING: after every tentative assignment we check that
  the affected hours can STILL shed their required MW using the feeders that remain
  eligible (outage allowance left, not protected, consecutive limit not broken).
  If not, the branch is cut immediately.
* GLOBAL BOUND: before each new hour, the total remaining required shed must fit into
  the remaining outage allowances. Before the search, a sliding-window version of the same
  bound (which uses the consecutive-outage limits) can prove infeasibility immediately.
* RESTART PORTFOLIO: if the first strategy ("most constrained hour first") exhausts its share of
  the node/time budget, the search restarts from scratch with another ordering rule (see
  SEARCH_STRATEGIES). Some scenarios are tight packing problems where one rule thrashes and
  another does not. A strategy that exhausts the whole search space proves infeasibility.
* Search is deterministic (no randomness) and has a node limit and a time limit. If every
  strategy hits its limit, the result is LIMIT_REACHED (never a fake success).

Infeasibility
-------------
If the static checks or the exhaustive search prove that no valid schedule exists, the
scenario is reported as INFEASIBLE with a reason (supply too low vs. constraint
interaction). A schedule that breaks a hard rule is NEVER returned.
"""

from __future__ import annotations

import time

import numpy as np

from . import config
from .environment import LoadSheddingEnvironment
from .scheduler_base import (STATUS_FEASIBLE, STATUS_INFEASIBLE, STATUS_LIMIT_REACHED,
                             ScheduleResult, Scheduler)


class SearchLimitReached(Exception):
    """Raised inside the recursion when the node or time limit is exceeded."""


# Search strategies tried in order (a small "restart portfolio"). Each entry is
# (hour-selection rule, feeder-ordering rule, share of the node/time limit). A strategy that PROVES
# infeasibility ends the search immediately; one that merely runs out of budget is followed by the
# next strategy.
SEARCH_STRATEGIES = (
    ("most_constrained_hour", "cover_fair", 0.20),       # dynamic MRV hour rule + fairness-aware feeder order
    ("smallest_requirement_hour", "cover_fair", 0.30),   # hour that needs the least shedding first
    ("most_constrained_hour", "largest_first", 0.50),    # use the biggest feeders first (fewest outage cells)
)


class CSPScheduler(Scheduler):
    """Backtracking CSP scheduler."""

    def __init__(self, environment: LoadSheddingEnvironment,
                 max_nodes: int = config.CSP_MAX_NODES,
                 max_seconds: float = config.CSP_MAX_SECONDS,
                 event_callback=None) -> None:
        super().__init__(environment, "CSP")
        self.event_callback = event_callback
        self.max_nodes = max_nodes
        self.max_seconds = max_seconds
        self.tol = config.DEMAND_TOLERANCE
        self.hour_rule = SEARCH_STRATEGIES[0][0]
        self.feeder_rule = SEARCH_STRATEGIES[0][1]
        self.node_limit = max_nodes
        self.deadline = 0.0

    def _emit(self, event: str, **fields) -> None:
        """Optional observation only; never influences decisions or search ordering."""
        if self.event_callback is not None:
            self.event_callback({"event": event, **fields})

    # ====================================================================== set-up
    def _initialise(self) -> None:
        """Build variables, domains and the search state for the loaded scenario."""
        env = self.env
        F, H = env.n_feeders, env.n_hours
        self.F, self.H = F, H
        self.demand = env.planning_demand.tolist()                  # demand[i][t] (forecast)
        self.usable = env.usable_supply.tolist()
        total = env.planning_demand.sum(axis=0)
        # required[t]: demand that MUST be shed in hour t so served <= usable
        self.required = [float(max(0.0, total[t] - self.usable[t])) for t in range(H)]
        self.required = [r if r > self.tol else 0.0 for r in self.required]
        self.forced_on = env.must_stay_on.tolist()                  # domain {1}
        self.max_total = env.max_total_outage.tolist()
        self.max_consec = env.max_consecutive_outage.tolist()
        self.weight = env.priority_weight.tolist()

        # Assignment table: -1 = unassigned, 0 = OFF, 1 = ON
        self.assign = [[-1] * H for _ in range(F)]
        self.outage_count = [0] * F
        self.shed = [0.0] * H                                       # demand already switched OFF per hour
        self.nodes = 0
        self.backtracks = 0

        for t in range(H):
            for i in range(F):
                # Domain {1} (protected/hospital) or no shortage in this hour -> ON, no search needed.
                if self.forced_on[i][t] or self.required[t] == 0.0:
                    self.assign[i][t] = config.ON

        self.shortage_hours = [t for t in range(H) if self.required[t] > 0.0]
        self.domain_size_one = sum(1 for i in range(F) for t in range(H) if self.forced_on[i][t])

    # ================================================================ small helpers
    def _off_run_length(self, i: int, t: int) -> int:
        """Length of the OFF run that would exist if x[i,t] were OFF (counts assigned OFF neighbours)."""
        row = self.assign[i]
        length = 1
        k = t - 1
        while k >= 0 and row[k] == config.OFF:
            length += 1
            k -= 1
        k = t + 1
        while k < self.H and row[k] == config.OFF:
            length += 1
            k += 1
        return length

    def _can_be_off(self, i: int, t: int) -> bool:
        """Could the (still unassigned) variable x[i,t] legally become OFF right now?"""
        return (self.assign[i][t] == -1
                and not self.forced_on[i][t]
                and self.outage_count[i] < self.max_total[i]
                and self._off_run_length(i, t) <= self.max_consec[i])

    def _hour_can_be_covered(self, t: int) -> bool:
        """Capacity pruning: can hour t still shed `required[t]` MW with the feeders still eligible?"""
        if self.required[t] == 0.0:
            return True
        reachable = self.shed[t]
        for i in range(self.F):
            if self._can_be_off(i, t):
                reachable += self.demand[i][t]
        return reachable + self.tol >= self.required[t]

    def _assign(self, i: int, t: int, value: int) -> None:
        self.assign[i][t] = value
        if value == config.OFF:
            self.outage_count[i] += 1
            self.shed[t] += self.demand[i][t]

    def _unassign(self, i: int, t: int) -> None:
        if self.assign[i][t] == config.OFF:
            self.outage_count[i] -= 1
            self.shed[t] -= self.demand[i][t]
        self.assign[i][t] = -1

    # ============================================================ constraint checking
    def check_constraints(self, i: int, t: int, value: int) -> bool:
        """Can x[i,t] = value be added to the current PARTIAL assignment without breaking a hard constraint?

        For OFF (0) this checks: domain (hospital/protected), total-outage limit, consecutive-outage limit,
        and the dominance rule (no needless outage).
        For both values it then runs the capacity / forward-checking test: every hour that could be
        affected must still be able to shed its required MW.
        """
        if value not in (config.OFF, config.ON):
            return False                                            # constraint 6: binary only
        if value == config.OFF:
            if self.forced_on[i][t]:                                # constraints 1 and 2
                return False
            if self.outage_count[i] >= self.max_total[i]:           # constraint 3
                return False
            if self._off_run_length(i, t) > self.max_consec[i]:     # constraint 4
                return False
            if self.shed[t] + self.tol >= self.required[t]:         # dominance: no needless outage
                return False

        self._assign(i, t, value)
        try:
            affected = {t}
            if value == config.OFF:
                affected.update(h for h in (t - 1, t + 1) if 0 <= h < self.H)   # consecutive-run effect
                if self.outage_count[i] >= self.max_total[i]:                    # allowance used up:
                    affected.update(self.shortage_hours)                         # every hour is affected
            return all(self._hour_can_be_covered(h) for h in affected)           # constraint 5
        finally:
            self._unassign(i, t)                                     # always undo the tentative assignment

    def _global_bound_ok(self, remaining_hours: list[int]) -> bool:
        """Necessary condition: the remaining outage allowances must be able to cover the remaining shortages.

        An outage cell (i,t) can contribute at most min(demand[i][t], required[t]) towards covering hour t
        (anything above required[t] is wasted over-shedding). If the best possible use of every feeder's
        remaining allowance still cannot reach the total required shed, this branch cannot succeed.
        """
        needed = sum(self.required[t] for t in remaining_hours)
        capacity = 0.0
        for i in range(self.F):
            allowance = self.max_total[i] - self.outage_count[i]
            if allowance <= 0:
                continue
            options = sorted((min(self.demand[i][t], self.required[t])
                              for t in remaining_hours if self._can_be_off(i, t)), reverse=True)
            capacity += sum(options[:allowance])
        return capacity + self.tol >= needed

    # ================================================================== static diagnosis
    def diagnose_infeasibility(self) -> tuple[str, str] | None:
        """Cheap necessary-condition checks. Returns (cause, reason) if the scenario is provably infeasible."""
        # (a) hour by hour: even if EVERY sheddable feeder were OFF, is the shortage covered?
        for t in self.shortage_hours:
            sheddable = sum(self.demand[i][t] for i in range(self.F) if not self.forced_on[i][t])
            if sheddable + self.tol < self.required[t]:
                return ("supply",
                        f"Hour {t:02d}: forecast demand exceeds usable supply by {self.required[t]:.2f} MW, "
                        f"but only {sheddable:.2f} MW of demand may be switched off "
                        f"(hospital/protected/no-outage feeders must stay ON). Supply is too low.")
        # (b) whole day: do the outage-hour allowances cover all the required shedding?
        needed = sum(self.required)
        capacity = 0.0
        for i in range(self.F):
            options = sorted((min(self.demand[i][t], self.required[t])
                              for t in self.shortage_hours if not self.forced_on[i][t]), reverse=True)
            capacity += sum(options[:self.max_total[i]])
        if capacity + self.tol < needed:
            return ("supply",
                    f"Total shedding needed over the day is {needed:.2f} MWh, but the outage-hour "
                    f"allowances can remove at most {capacity:.2f} MWh. Supply is too low for the limits.")
        # (c) sliding windows: a feeder with consecutive limit c can be OFF at most
        #     floor(W/(c+1))*c + min(c, W mod (c+1)) hours inside any window of W hours.
        for start in range(self.H):
            for end in range(start, self.H):
                window = [t for t in range(start, end + 1) if self.required[t] > 0.0]
                need = sum(self.required[t] for t in window)
                if need == 0.0:
                    continue
                width = end - start + 1
                supply_possible = 0.0
                for i in range(self.F):
                    c = self.max_consec[i]
                    hours_allowed = min(self.max_total[i], (width // (c + 1)) * c + min(c, width % (c + 1)))
                    options = sorted((self.demand[i][t] for t in window if not self.forced_on[i][t]),
                                     reverse=True)
                    supply_possible += sum(options[:hours_allowed])
                if supply_possible + self.tol < need:
                    return ("constraint interaction",
                            f"Hours {start:02d}-{end:02d} need {need:.2f} MWh of shedding, but the "
                            f"consecutive/total outage limits let the feeders shed at most "
                            f"{supply_possible:.2f} MWh in that window.")
        return None

    # ==================================================================== search
    def _order_feeders(self, t: int) -> list[int]:
        """Feeder order for hour t (see module docstring).

        1. feeders whose demand alone covers the remaining need come first (fewest outage-hours used),
        2. then the feeder that has used the smallest share of its outage allowance (fairness),
        3. then the lower priority weight (protect important feeders),
        4. then the demand closest to the amount still needed (little over-shedding).
        """
        need = self.required[t] - self.shed[t]
        candidates = [i for i in range(self.F) if self.assign[i][t] == -1]
        if self.feeder_rule == "largest_first":                      # alternative rule used by strategy 3
            return sorted(candidates, key=lambda i: (-self.demand[i][t], i))

        def key(i: int):
            used_fraction = self.outage_count[i] / self.max_total[i] if self.max_total[i] else 1.0
            covers_alone = 0 if self.demand[i][t] + self.tol >= need else 1
            return (covers_alone, round(used_fraction, 3), self.weight[i],
                    abs(self.demand[i][t] - need), i)

        return sorted(candidates, key=key)

    def _hour_slack(self, t: int) -> float:
        """Slack of an unprocessed hour = (demand that could still be shed) - (demand that must be shed)."""
        reachable = self.shed[t]
        for i in range(self.F):
            if self._can_be_off(i, t):
                reachable += self.demand[i][t]
        return reachable - self.required[t]

    def _start_block(self, remaining: list[int]) -> bool:
        """Pick the next shortage hour (dynamic MRV: smallest current slack first) and search it."""
        if not remaining:
            return True
        if not self._global_bound_ok(remaining):
            self.backtracks += 1
            return False
        if self.hour_rule == "smallest_requirement_hour":
            t = min(remaining, key=lambda hour: (self.required[hour], hour))
        else:                                                         # "most_constrained_hour"
            t = min(remaining, key=lambda hour: (self._hour_slack(hour), hour))
        others = [hour for hour in remaining if hour != t]
        return self.backtrack(t, others, 0, self._order_feeders(t))

    def backtrack(self, t: int, remaining: list[int], position: int, order: list[int]) -> bool:
        """Recursive backtracking over the variables x[i,t] of hour t, then over the remaining hours.

        t         : the hour currently being assigned
        remaining : shortage hours still to be processed after hour t
        position  : index in `order` of the next feeder variable of hour t
        order     : feeder order for hour t
        """
        self.nodes += 1
        if self.nodes > self.node_limit:
            raise SearchLimitReached(f"node limit of {self.node_limit} reached")
        if self.nodes % 1000 == 0 and time.perf_counter() > self.deadline:
            raise SearchLimitReached("time limit reached")

        if position == len(order):                                   # all feeders of hour t assigned
            if self.shed[t] + self.tol < self.required[t]:           # capacity constraint for hour t
                self.backtracks += 1
                return False
            return self._start_block(remaining)                      # continue with the next hour

        i = order[position]
        values = (config.OFF, config.ON) if self.shed[t] + self.tol < self.required[t] else (config.ON,)
        for value in values:
            if self.check_constraints(i, t, value):
                self._assign(i, t, value)
                self._emit("assign", feeder_id=self.env.feeder_ids[i], hour=t, decision="ON" if value else "OFF", node=self.nodes)
                if self.backtrack(t, remaining, position + 1, order):
                    return True
                self._unassign(i, t)                                 # undo and try the next value
                self._emit("backtrack", feeder_id=self.env.feeder_ids[i], hour=t, node=self.nodes)
        self.backtracks += 1
        return False

    # ============================================================== main interface
    def generate_schedule(self) -> ScheduleResult:
        """Find a valid schedule for the loaded scenario (or explain why none exists)."""
        started = time.perf_counter()
        self._initialise()
        info = {"required_shed_hours": len(self.shortage_hours),
                "total_required_shed_mwh": round(sum(self.required), 3),
                "domain_size_one_variables": self.domain_size_one}

        diagnosis = self.diagnose_infeasibility()
        if diagnosis is not None:
            cause, reason = diagnosis
            self._emit("infeasible", cause=cause, reason=reason)
            return self._result(STATUS_INFEASIBLE, None, time.perf_counter() - started,
                                reason=reason, cause=cause, nodes=0, backtracks=0, restarts=0, **info)

        total_nodes = total_backtracks = 0
        found = False
        limit_message = ""
        proven_infeasible = False
        attempts_used = 0
        for attempt, (rule, feeder_rule, share) in enumerate(SEARCH_STRATEGIES):
            self._initialise()                                       # fresh search state for every attempt
            self.hour_rule = rule
            self.feeder_rule = feeder_rule
            self.node_limit = int(self.max_nodes * share)
            self.deadline = time.perf_counter() + self.max_seconds * share
            attempts_used = attempt + 1
            if self.event_callback is not None:
                self._emit("strategy_start", attempt=attempts_used, hour_rule=rule, feeder_rule=feeder_rule,
                           schedule=[row[:] for row in self.assign])
            try:
                found = self._start_block(list(self.shortage_hours))
                proven_infeasible = not found                        # search space exhausted, no limit hit
            except SearchLimitReached as limit:
                limit_message = str(limit)
                self._emit("search_limit", reason=limit_message)
            total_nodes += self.nodes
            total_backtracks += self.backtracks
            if found or proven_infeasible:
                break

        runtime = time.perf_counter() - started
        stats = dict(nodes=total_nodes, backtracks=total_backtracks, restarts=attempts_used - 1,
                     strategy=f"{self.hour_rule}+{self.feeder_rule}", **info)

        if proven_infeasible:
            return self._result(
                STATUS_INFEASIBLE, None, runtime,
                reason=("Exhaustive backtracking proved that no schedule satisfies all hard constraints. "
                        "Each hour can be covered on its own, but the feeders' total/consecutive outage "
                        "limits and protected hours cannot cover every shortage hour at the same time."),
                cause="constraint interaction", **stats)
        if not found:
            return self._result(
                STATUS_LIMIT_REACHED, None, runtime,
                reason=(f"Search stopped ({limit_message}) after {attempts_used} strategies. No valid "
                        f"schedule was found within the limit, so feasibility is unknown."),
                cause="search limit", **stats)

        schedule = np.array(self.assign, dtype=int)
        if (schedule == -1).any():
            raise RuntimeError("Internal error: unassigned variable in a completed CSP search")
        violations = self.validate_schedule(schedule, "planning")
        if violations:                                               # safety net: should never happen
            raise RuntimeError("CSP produced an invalid schedule: " + "; ".join(violations))
        self._emit("complete", status=STATUS_FEASIBLE, nodes=total_nodes, backtracks=total_backtracks)
        return self._result(STATUS_FEASIBLE, schedule, runtime, **stats)
