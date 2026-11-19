"""A trace must reconstruct the real returned solution without invented states."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from web_app.lab_service import run_lab


class TestTraceReplay(unittest.TestCase):
    def test_real_csp_and_hc_events_reconstruct_returned_schedules(self):
        for scenario in ('S001', 'S017', 'S025', 'S050'):
            for method in ('CSP', 'HC'):
                with self.subTest(scenario=scenario, method=method):
                    result = run_lab(scenario, method)
                    self.assertFalse(result['trace_truncated'])
                    ids = result['feeder_ids']
                    board = [[-1] * 24 for _ in ids]
                    snapshots = 0
                    for event in result['trace']:
                        if 'schedule' in event:
                            board = [row[:] for row in event['schedule']]
                            snapshots += 1
                        if event['event'] == 'assign':
                            board[ids.index(event['feeder_id'])][event['hour']] = int(event['decision'] == 'ON')
                        if event['event'] == 'backtrack':
                            board[ids.index(event['feeder_id'])][event['hour']] = -1
                        for key, value in (('turn_on', 1), ('turn_off', 0)):
                            for cell in event.get(key, []):
                                board[ids.index(cell['feeder_id'])][cell['hour']] = value
                        if event['event'] == 'accepted_move':
                            self.assertLess(event['score'], event['previous_score'])
                    self.assertGreater(snapshots, 0)
                    self.assertEqual(board, result['final_schedule'])
                    self.assertEqual(board[0], [1] * 24, 'Hospital must be ON in the returned solution')

    def test_hill_climbing_records_all_five_distinct_start_states(self):
        result = run_lab('S001', 'HC')
        starts = [event for event in result['trace'] if event['event'] == 'restart_state']
        self.assertEqual([e['restart'] for e in starts], list(range(5)))
        self.assertTrue(all(len(e['schedule']) == 10 and all(len(row) == 24 for row in e['schedule']) for e in starts))
        self.assertTrue(all(e['score'] >= 0 for e in starts))
        self.assertEqual(result['trace'][-1]['event'], 'complete')
        self.assertEqual(result['trace'][-1]['score'], result['summary']['final_score'])

    def test_infeasible_run_has_no_fabricated_schedule(self):
        for method in ('CSP', 'HC'):
            result = run_lab('S002', method)
            self.assertIsNone(result['final_schedule'])
            self.assertFalse(any('schedule' in e for e in result['trace']))
            self.assertIn(result['trace'][0]['event'], ('infeasible', 'skipped'))
