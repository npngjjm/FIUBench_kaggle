"""CPU-only control-flow regression tests; no claim of GPU validation."""
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ga_scheduler_audit import SchedulerAudit, patch_training_source


class FakeScheduler:
    def __init__(self, optimizer, multiplier=1):
        self.step_with_optimizer = False
        self.multiplier = multiplier
        self.scheduler = SimpleNamespace(optimizer=optimizer, last_epoch=0, base_lrs=[2e-5])

    def step(self):
        self.scheduler.last_epoch += self.multiplier
        n = self.scheduler.last_epoch
        factor = n / 2 if n < 2 else (1 + math.cos(math.pi * (n - 2) / 30)) / 2
        self.scheduler.optimizer.param_groups[0]['lr'] = 2e-5 * factor


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.opt = SimpleNamespace(param_groups=[{'lr': 0.0}])
        self.sch = FakeScheduler(self.opt)
        self.audit = SchedulerAudit(self.sch, self.opt, 32, 2, Path(self.tmp.name) / 'audit.jsonl')

    def step(self, sync=True, skipped=False):
        return self.audit.finish_microstep(sync=sync, skipped=skipped,
            lr_before=self.audit.check(), epoch=0, microbatch=0, loss=-0.5)

    def test_accumulation_does_not_advance(self):
        for _ in range(15):
            self.assertIsNone(self.step(sync=False))
        self.assertEqual(self.audit.successes, 0)
        self.step()
        self.assertEqual(self.audit.successes, 1)

    def test_overflow_does_not_advance(self):
        self.step(skipped=True)
        self.step()
        self.assertEqual((self.audit.attempts, self.audit.skips, self.audit.successes), (2, 1, 1))

    def test_32_updates_no_rebound(self):
        lrs = [self.step()['lr_next'][0] for _ in range(32)]
        self.assertTrue(all(a >= b for a, b in zip(lrs[1:], lrs[2:])))
        self.assertEqual(lrs[-1], 0.0)
        with self.assertRaises(AssertionError):
            self.step()

    def test_old_double_step_is_rejected(self):
        self.sch.multiplier = 2
        with self.assertRaises(AssertionError):
            self.step()

    def test_detached_optimizer_is_rejected(self):
        other = SimpleNamespace(param_groups=[{'lr': 0.0}])
        with self.assertRaisesRegex(AssertionError, 'share LR'):
            SchedulerAudit(self.sch, other, 32, 2, Path(self.tmp.name) / 'other.jsonl')

    def test_restart_cannot_overwrite_log(self):
        self.step(skipped=True)
        with self.assertRaisesRegex(AssertionError, 'Existing audit'):
            SchedulerAudit(self.sch, self.opt, 32, 2, self.audit.path)

    def test_unknown_source_rejected(self):
        with self.assertRaises(AssertionError):
            patch_training_source('print("different source")')


if __name__ == '__main__':
    unittest.main()
