"""Single-step cosine scheduling and fail-fast audit for the isolated GA notebook.

No torch import: numerical/control-flow tests can run without a GPU environment.
The notebook embeds this file, so it does not depend on a future git push.
"""
import ast
import json
import math
from pathlib import Path


class SchedulerAudit:
    def __init__(self, scheduler, optimizer, total_steps, warmup_steps, log_path):
        self.scheduler = scheduler
        self.raw = scheduler.scheduler
        self.optimizer = optimizer
        self.total = int(total_steps)
        self.warmup = int(warmup_steps)
        assert scheduler.step_with_optimizer is False
        assert self.raw.last_epoch == 0, "Expected a fresh LambdaLR; do not resume weights here"
        assert 0 < self.warmup < self.total
        a, b = self.raw.optimizer.param_groups, optimizer.param_groups
        assert len(a) == len(b) and all(x is y for x, y in zip(a, b)), \
            "Scheduler and actual DeepSpeed optimizer do not share LR parameter groups"
        self.attempts = self.successes = self.skips = 0
        self.base_lrs = list(self.raw.base_lrs)
        self.path = Path(log_path)
        assert not self.path.exists(), f"Existing audit log: {self.path}; use a new run directory"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.check()

    def expected_lrs(self, step):
        if step < self.warmup:
            factor = step / self.warmup
        else:
            progress = (step - self.warmup) / (self.total - self.warmup)
            factor = 0.5 * (1 + math.cos(math.pi * progress))
        return [lr * factor for lr in self.base_lrs]

    def check(self):
        assert self.raw.last_epoch == self.successes, \
            ("Scheduler index differs from successful updates", self.raw.last_epoch, self.successes)
        assert self.successes <= self.total
        actual = [float(g['lr']) for g in self.optimizer.param_groups]
        assert len(actual) == len(self.base_lrs)
        for got, wanted in zip(actual, self.expected_lrs(self.successes)):
            assert math.isclose(got, wanted, rel_tol=1e-7, abs_tol=1e-12), (got, wanted)
        return actual

    def finish_microstep(self, *, sync, skipped, lr_before, epoch, microbatch, loss):
        # DeepSpeed already stepped its optimizer inside accelerator.backward().
        self.check()  # Also detects an unexpected engine-owned scheduler step.
        assert lr_before == [float(g['lr']) for g in self.optimizer.param_groups]
        if not sync:
            return None
        self.attempts += 1
        assert self.attempts <= self.total
        assert math.isfinite(loss), f"Non-finite GA loss: {loss}"
        if skipped:
            self.skips += 1
        else:
            self.scheduler.step()
            self.successes += 1
        after = self.check()
        record = dict(epoch=int(epoch), microbatch=int(microbatch),
                      attempt=self.attempts, successful_updates=self.successes,
                      skipped_updates=self.skips, skipped=bool(skipped),
                      scheduler_index=int(self.raw.last_epoch),
                      lr_used=lr_before, lr_next=after, ga_loss=float(loss))
        with self.path.open('a') as f:
            f.write(json.dumps(record, allow_nan=False) + '\n')
        return record


def patch_training_source(text):
    """Patch exact anchors; fail on source drift instead of silently doing nothing."""
    def replace(old, new):
        nonlocal text
        assert text.count(old) == 1, f"Expected exactly one patch anchor: {old!r}"
        text = text.replace(old, new, 1)

    replace('from accelerate import Accelerator, DistributedType',
            'from accelerate import Accelerator, DistributedType\nfrom ga_scheduler_audit import SchedulerAudit')
    replace('        gradient_accumulation_steps=cfg.gradient_accumulation_steps,\n',
            '        gradient_accumulation_steps=cfg.gradient_accumulation_steps,\n'
            '        step_scheduler_with_optimizer=False,\n')
    anchor = '    model, optimizer, torch_format_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, torch_format_dataloader, lr_scheduler)'
    replace(anchor, anchor + '''
    assert cfg.forget_loss == "ga" and not cfg.resume_from_checkpoint
    assert cfg.lr_scheduler_type == "cosine"
    assert accelerator.num_processes == 2
    assert len(torch_format_dataloader) == 50, len(torch_format_dataloader)
    assert max_train_steps == 32, max_train_steps
    assert getattr(model, "lr_scheduler", None) is None, "DeepSpeed must not step a second scheduler"
    schedule_audit = SchedulerAudit(
        lr_scheduler, optimizer, max_train_steps,
        round(cfg.warmup_ratio * max_train_steps),
        Path(cfg.save_dir) / f"scheduler_rank{accelerator.process_index}.jsonl",
    )
    accelerator.print("SCHEDULER_BINDING_OK: one scheduler step per successful optimizer update")
''')
    replace('                accelerator.backward(loss)',
            '                audit_lr_before = schedule_audit.check()\n'
            '                accelerator.backward(loss)')
    replace('                lr_scheduler.step()', '''                audit_record = schedule_audit.finish_microstep(
                    sync=accelerator.sync_gradients,
                    skipped=bool(accelerator.optimizer_step_was_skipped) if accelerator.sync_gradients else False,
                    lr_before=audit_lr_before, epoch=epoch, microbatch=step,
                    loss=float(loss.detach().float().item()) if accelerator.sync_gradients else 0.0,
                )
                if audit_record is not None:
                    assert audit_record["attempt"] == completed_steps + 1
                    accelerator.print("SCHEDULER_AUDIT " + json.dumps(audit_record))
                    if audit_record["attempt"] == 4:
                        assert audit_record["successful_updates"] > 0, "No successful updates in first epoch"
                        accelerator.print("FIRST_EPOCH_SCHEDULER_OK — continuing same run")''')
    # Keep 8 epochs and the original attempt-based checkpoint numbering unchanged.
    anchor = '    output_dir = cfg.save_dir\n    accelerator.wait_for_everyone()'
    replace(anchor, '''    assert schedule_audit.attempts == max_train_steps
    assert schedule_audit.successes + schedule_audit.skips == max_train_steps
    schedule_audit.check()
    accelerator.print(f"SCHEDULER_FINAL attempts={schedule_audit.attempts} successful={schedule_audit.successes} skipped={schedule_audit.skips}")
''' + anchor)
    ast.parse(text)
    return text
