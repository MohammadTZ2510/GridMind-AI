"""
hill_climbing_scheduler.py
==========================
`HillClimbingScheduler` - improves a VALID CSP schedule with our own Hill Climbing
(local search). It never trades hard constraints for a better score.

Objective function (MINIMISED)
------------------------------
Each term is divided by a natural maximum so that no term dominates just because of
its units. All demand figures are the FORECAST (planning) demand.

    unmet_term    = (MWh of demand switched OFF)  / (total MWh of demand of the day)
    fairness_term = (std-dev of area outage hours) / (largest max_total_outage_hours)
    priority_term = sum( priority_weight_i * unmet MWh_i ) / (max priority weight * total MWh)

    score = W_UNMET * unmet_term + W_FAIRNESS * fairness_term + W_PRIORITY * priority_term

with the weights in config.py (0.5 / 0.3 / 0.2). All three terms lie roughly in [0, 1].
Area outage hours = mean outage hours of the area's sheddable feeders (see environment.py),
so the fairness term does not depend on how many feeders an area has.

Neighbourhood (small changes that keep the schedule binary)
-----------------------------------------------------------
  flip         : switch one OFF feeder-hour back ON                 (less unmet demand)
  swap feeder  : in one hour, turn feeder i ON and feeder j OFF     (reassign an outage)
  shift in time: turn (i, t) ON and (i, t') OFF for the same feeder (move an outage to another hour)

Hill Climbing procedure
-----------------------
1. Start from a valid schedule (the CSP result).
2. List all neighbours in a seeded random order.
3. Take the first neighbour that has a lower score AND satisfies every hard constraint.
4. Repeat until no neighbour improves (a LOCAL optimum) or the iteration limit is reached.

Local search can get stuck in a local optimum, so we use several seeded starts:
start 0 = the CSP schedule itself; starts 1..N = the CSP schedule after a few random
VALID moves. The best final schedule is returned. We do not claim it is the global optimum.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from . import config
from .csp_scheduler import CSPScheduler
from .environment import LoadSheddingEnvironment
from .scheduler_base import STATUS_FEASIBLE, STATUS_INFEASIBLE, ScheduleResult, Scheduler

# A move = (cells turned ON, cells turned OFF); each cell is a (feeder_index, hour) pair.
Move = tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]


class HillClimbingScheduler(Scheduler):
    """Hill Climbing improvement of a valid schedule."""

    def __init__(self, environment: LoadSheddingEnvironment,
                 initial_schedule: Optional[np.ndarray] = None,
                 num_restarts: int = config.HC_NUM_RESTARTS,
                 max_iterations: int = config.HC_MAX_ITERATIONS,
                 perturbation_moves: int = config.HC_PERTURBATION_MOVES,
                 seed: int = config.RANDOM_SEED,
                 record_history: bool = False, event_callback=None) -> None:
        super().__init__(environment, "HillClimbing")
        self.event_callback = event_callback
        self.initial_schedule = None if initial_schedule is None else np.asarray(initial_schedule, dtype=int)
        self.num_restarts = num_restarts
        self.max_iterations = max_iterations
        self.perturbation_moves = perturbation_moves
        self.seed = seed
        self.record_history = record_history
        self.accepted_history: list[np.ndarray] = []    # every accepted schedule (used by the tests)
        self.rejected_invalid = 0                       # improving neighbours rejected by constraints

    def _emit(self, event: str, **fields) -> None:
        """Optional observer; recording does not change search or random draws."""
        if self.event_callback is not None:
            self.event_callback({"event": event, **fields})

    # ================================================================== scoring
    def score_components(self, schedule: np.ndarray) -> dict[str, float]:
        """The three normalised terms and the final score (lower is better)."""
        env = self.env
        demand = env.planning_demand
        off = (schedule == config.OFF)
        unmet_by_cell = off * demand
        total_demand = float(demand.sum())
        if total_demand <= 0:
            total_demand = 1.0

        unmet_term = float(unmet_by_cell.sum()) / total_demand
        outage_hours = off.sum(axis=1).astype(float)
        imbalance = float(np.std(env.area_weights @ outage_hours))
        fairness_term = imbalance / max(1, int(env.max_total_outage.max()))
        priority_term = float((unmet_by_cell * env.priority_weight[:, None]).sum()) / (
            float(env.priority_weight.max()) * total_demand)

        score = (config.HC_WEIGHT_UNMET * unmet_term
                 + config.HC_WEIGHT_FAIRNESS * fairness_term
                 + config.HC_WEIGHT_PRIORITY * priority_term)
        return {"unmet_term": unmet_term, "fairness_term": fairness_term,
                "priority_term": priority_term, "score": score}

    def score_schedule(self, schedule: np.ndarray) -> float:
        """Objective value of a schedule (lower is better). Hard constraints are checked separately."""
        return self.score_components(schedule)["score"]

    # ============================================================== neighbourhood
    @staticmethod
    def _apply_move(schedule: np.ndarray, move: Move) -> np.ndarray:
        candidate = schedule.copy()
        for i, t in move[0]:
            candidate[i, t] = config.ON
        for i, t in move[1]:
            candidate[i, t] = config.OFF
        return candidate

    def _enumerate_moves(self, schedule: np.ndarray, include_flips: bool = True) -> list[Move]:
        """All neighbours of a schedule (cheap pre-filters only; full validity is checked later)."""
        env = self.env
        demand = env.planning_demand
        served = (schedule * demand).sum(axis=0)
        usable = env.usable_supply
        off_cells = [(int(i), int(t)) for i, t in zip(*np.where(schedule == config.OFF))]
        allowance_left = env.max_total_outage - (schedule == config.OFF).sum(axis=1)
        moves: list[Move] = []

        for (i, t) in off_cells:
            # (1) flip OFF -> ON: needs room in the supply of hour t
            if include_flips and served[t] + demand[i, t] <= usable[t] + config.SUPPLY_TOLERANCE:
                moves.append((((i, t),), ()))

            # (2) swap feeders inside hour t: i back ON, j newly OFF
            for j in range(env.n_feeders):
                if j == i or schedule[j, t] != config.ON or env.must_stay_on[j, t] or allowance_left[j] <= 0:
                    continue
                if served[t] + demand[i, t] - demand[j, t] <= usable[t] + config.SUPPLY_TOLERANCE:
                    moves.append((((i, t),), ((j, t),)))

            # (3) shift the outage of feeder i from hour t to another hour t2
            for t2 in range(env.n_hours):
                if t2 == t or schedule[i, t2] != config.ON or env.must_stay_on[i, t2]:
                    continue
                if served[t] + demand[i, t] <= usable[t] + config.SUPPLY_TOLERANCE:
                    moves.append((((i, t),), ((i, t2),)))
        return moves

    def generate_neighbor(self, schedule: np.ndarray, rng: np.random.Generator,
                          include_flips: bool = False, max_tries: int = 100) -> Optional[np.ndarray]:
        """One random VALID neighbour (or None if none was found). Used to create perturbed starts."""
        moves = self._enumerate_moves(schedule, include_flips=include_flips)
        if not moves:
            return None
        for index in rng.permutation(len(moves))[:max_tries]:
            candidate = self._apply_move(schedule, moves[int(index)])
            if self.env.is_valid(candidate, "planning"):
                return candidate
        return None

    # ============================================================= optimisation
    def optimize(self, start: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
        """Hill Climbing from `start`. Returns the final schedule and statistics."""
        current = start.copy()
        current_score = self.score_schedule(current)
        initial_score = current_score
        iterations = 0
        local_optimum = False

        while iterations < self.max_iterations:
            moves = self._enumerate_moves(current)
            improved = False
            for index in rng.permutation(len(moves)):                  # seeded random order
                candidate = self._apply_move(current, moves[int(index)])
                candidate_score = self.score_schedule(candidate)
                if candidate_score < current_score - config.HC_IMPROVEMENT_EPSILON:
                    # Only a VALID improving neighbour may be accepted.
                    if not self.env.is_valid(candidate, "planning"):
                        self.rejected_invalid += 1
                        self._emit("rejected_invalid", candidate_score=candidate_score)
                        continue
                    self._emit("accepted_move", previous_score=current_score, score=candidate_score,
                               turn_on=[{"feeder_id": self.env.feeder_ids[i], "hour": t} for i, t in moves[int(index)][0]],
                               turn_off=[{"feeder_id": self.env.feeder_ids[i], "hour": t} for i, t in moves[int(index)][1]])
                    current, current_score = candidate, candidate_score
                    iterations += 1
                    improved = True
                    if self.record_history:
                        self.accepted_history.append(current.copy())
                    break
            if not improved:
                local_optimum = True                                   # no neighbour improves
                self._emit("local_optimum", score=current_score, iterations=iterations)
                break
        return current, {"iterations": iterations, "initial_score": initial_score,
                         "final_score": current_score, "local_optimum": local_optimum}

    # ============================================================ main interface
    def generate_schedule(self) -> ScheduleResult:
        """Run Hill Climbing from a valid CSP schedule (the CSP is run here if none was supplied)."""
        started = time.perf_counter()

        start_schedule = self.initial_schedule
        csp_runtime = 0.0
        if start_schedule is None:
            csp_result = CSPScheduler(self.env).generate_schedule()
            csp_runtime = csp_result.runtime_s
            if not csp_result.is_feasible:
                return self._result(STATUS_INFEASIBLE, None, time.perf_counter() - started,
                                    reason="No valid CSP schedule to start from: " + csp_result.reason,
                                    cause=csp_result.cause)
            start_schedule = csp_result.schedule

        violations = self.validate_schedule(start_schedule, "planning")
        if violations:
            raise ValueError("Hill Climbing needs a valid starting schedule, but: " + "; ".join(violations))

        scenario_number = int(str(self.env.scenario_id)[1:])
        best_schedule, best_score, best_info, best_restart = None, np.inf, {}, -1
        total_iterations = 0
        restart_log = []
        for restart in range(self.num_restarts):
            self._emit("restart", restart=restart)
            rng = np.random.default_rng([self.seed, scenario_number, restart])   # repeatable seeded start
            start = start_schedule.copy()
            if restart > 0:                                            # perturb with random VALID moves
                for _ in range(self.perturbation_moves):
                    neighbour = self.generate_neighbor(start, rng)
                    if neighbour is not None:
                        start = neighbour
            if self.event_callback is not None:
                self._emit("restart_state", restart=restart, schedule=start.tolist(), score=self.score_schedule(start))
            final, info = self.optimize(start, rng)
            total_iterations += info["iterations"]
            restart_log.append(info["final_score"])
            if info["final_score"] < best_score - config.HC_IMPROVEMENT_EPSILON:
                best_schedule, best_score, best_info, best_restart = final, info["final_score"], info, restart

        final_violations = self.validate_schedule(best_schedule, "planning")
        if final_violations:                                           # safety net: should never happen
            raise RuntimeError("Hill Climbing produced an invalid schedule: " + "; ".join(final_violations))

        initial_score = self.score_schedule(start_schedule)
        runtime = time.perf_counter() - started
        if self.event_callback is not None:
            self._emit("complete", schedule=best_schedule.tolist(), score=best_score, best_restart=best_restart, status=STATUS_FEASIBLE)
        return self._result(
            STATUS_FEASIBLE, best_schedule, runtime,
            initial_score=initial_score, final_score=best_score,
            improved=bool(best_score < initial_score - config.HC_IMPROVEMENT_EPSILON),
            iterations=total_iterations, best_restart=best_restart, restarts=self.num_restarts,
            local_optimum=best_info.get("local_optimum", False),
            rejected_invalid_neighbours=self.rejected_invalid,
            final_constraint_violations=len(final_violations),
            csp_start_runtime_s=csp_runtime,
        )
