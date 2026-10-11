"""Read saved Kaggle GA logs and compare weights; no training or file writes."""
from pathlib import Path
import re


def parse_progress(text):
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf)"
    pattern = re.compile(
        rf"Epoch\s+(\d+)\s*[-–]\s*Step\s+(\d+)\s*[-–]\s*"
        rf"LR:\s*({number})\s*[-–]\s*loss:\s*({number})",
        re.I,
    )
    # Keep the last recorded line for each epoch/microbatch, removing tqdm repeats.
    return {(int(e), int(s)): (float(lr), float(loss))
            for e, s, lr, loss in pattern.findall(text)}


def main():
    roots = [Path('/kaggle/input'), Path('/kaggle/working')]
    logs = sorted({p for root in roots for p in root.rglob('ga_forget5_fast_console.log')})
    folders = sorted({p.parent for root in roots for p in root.rglob('checkpoint.pt')
                      if p.parent.name == 'llava_phi_GA_forget5_fast'})
    print('=== LOG PATHS ===', *logs, sep='\n')
    print('=== CHECKPOINT FOLDERS ===', *folders, sep='\n')
    if not logs and not folders:
        raise RuntimeError('Add Input에 fast_train_GA의 저장된 Output을 연결하세요.')

    for path in logs:
        text = path.read_text(errors='replace')
        records = parse_progress(text)
        print('\n=== LOG:', path, '===')
        print('epoch | last microbatch | LR | GA loss (negative CE)')
        for epoch in sorted({e for e, s in records}):
            step = max(s for e, s in records if e == epoch)
            lr, loss = records[epoch, step]
            print(f'{epoch:5d} | {step:15d} | {lr:.6e} | {loss:.6g}')
        if not records:
            print('진행률 기록 없음: 로그 부재만으로 학습 실패를 뜻하지 않습니다.')
        learning_rates = [v[0] for k, v in sorted(records.items())]
        if learning_rates:
            # Exclude initial warmup: inspect only the second half for a rebound.
            half = len(learning_rates) // 2
            tail = learning_rates[half:]
            print('후반 LR 최솟값 / 마지막:', min(tail), learning_rates[-1])
            print('후반 최솟값보다 마지막 LR이 큼:', learning_rates[-1] > min(tail) + 1e-12)
        evidence = [line.strip() for line in text.replace('\r', '\n').splitlines()
                    if re.search(r'overflow|skipp(?:ed|ing).*step|loss.?scale|\b(?:nan|inf)\b', line, re.I)]
        print('overflow/skip/nonfinite 관련 로그 (최대 12줄):')
        print('\n'.join(evidence[-12:]) if evidence else '발견되지 않음 — 발생하지 않았다는 증명은 아님')
        print('저장 step:', sorted(set(map(int, re.findall(r'Saving checkpoint to [^\r\n]*?/step_(\d+)/checkpoint\.pt', text)))))
        print('주의: TRAIN_PROCESS_EXIT_0는 tee 뒤 출력되어 파일에 없을 수 있습니다.')

    if not folders:
        print('체크포인트 폴더 없음: 가중치 검사는 생략합니다.')
        return
    import torch
    for folder in folders:
        cfg = folder / 'cfg.yaml'
        print('\n=== CONFIG:', cfg, '===')
        if cfg.is_file():
            for line in cfg.read_text().splitlines():
                if re.match(r'^(model_path|forget_loss|lr|batch_size|gradient_accumulation_steps|num_epochs|lr_scheduler_type|warmup_ratio):', line):
                    print(line)
        steps = sorted(folder.glob('step_*/checkpoint.pt'),
                       key=lambda p: int(p.parent.name.split('_')[1]))
        print('저장된 step:', [p.parent.name for p in steps])
        selected = ([steps[0], steps[-1]] if len(steps) > 1 else steps)
        if (folder / 'checkpoint.pt').is_file():
            selected += [folder / 'checkpoint.pt']
        previous = None
        previous_name = None
        for path in selected:
            state = torch.load(path, map_location='cpu', weights_only=True)
            if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
                raise RuntimeError(f'예상한 가중치 dict가 아닙니다: {path}')
            name = path.parent.name + '/' + path.name
            bad = [k for k, v in state.items() if not torch.isfinite(v).all().item()]
            print(name, 'tensors:', len(state), 'NaN/Inf tensors:', len(bad))
            if previous is not None:
                if state.keys() != previous.keys():
                    raise RuntimeError('비교 체크포인트 사이 tensor key가 다릅니다.')
                changed = [k for k in state if not torch.equal(state[k], previous[k])]
                print(f'{previous_name} -> {name}: changed {len(changed)}/{len(state)}')
                print('  LoRA changed:', sum('lora_' in k for k in changed),
                      'projector changed:', sum('projector' in k for k in changed))
            previous, previous_name = state, name
        del previous
    print('\n체크포인트는 가중치만 저장하므로 실제 optimizer 업데이트 횟수는 복원할 수 없습니다.')
    print('진단 완료: 재학습하지 않았으며 어떤 파일도 변경하지 않았습니다.')


if __name__ == '__main__':
    main()
