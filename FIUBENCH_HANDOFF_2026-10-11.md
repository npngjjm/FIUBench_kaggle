# FIUBench 진행 상황 복구 — 2026-10-11 (Asia/Seoul)

> 10월 12일 후속: 수정 GA 학습은 32 attempts / 30 successes / 2 skips로 감사 통과했고, 사용자가 Core 평가 EM 42.2708%, retain ROUGE-L 92.7040%를 보고했다. 원인 재조사와 이전 MIA 설명의 정정은 `FIUBENCH_DIAGNOSIS_2026-10-12.md`를 우선 참조한다. 아래 내용은 10월 11일 시점의 기록이다.

## 근거와 확인 범위

- 이전 채팅: `Connect FIUBench repository`, ID `01a11ae7-af73-7192-bf32-d15e2d73df27`. 최근 20개 턴의 사용자 보고, 진단 출력, 최종 안내를 현재 파일과 대조했다.
- 로컬 HEAD: `5463c79` (`schedulerModified`). 조사 시작 시 main 작업 트리는 깨끗했고 로컬 origin/main과 일치했다. 이번 조사에서 원격 fetch나 Kaggle 접속은 하지 않았다.
- 기존 `FIUBENCH_REPRO_CONTEXT.md`는 10월 9일 기준이다. PO 초기 기록으로 참고하되 현재 진행 상태로 쓰면 안 된다. 문서 내부에도 옛 batch 8 설명 등이 남아 있다.
- 아래 Kaggle GA 결과는 이전 대화의 보고에 근거한다. 이 폴더에는 해당 GA checkpoint 및 원본 평가 JSON이 없다.

## 현재 목표와 중단 지점

공개 Stage-I 모델 `gray311/vlm_unlearning_ft_llava_phi_3_mini`에서 시작해 Kaggle T4 두 장으로 FIUBench Stage-II forget5를 재현한다. PO 초기 학습/평가 이후 GA로 실험을 확장했다.

마지막으로 완료된 개발 작업은 GA의 학습률 스케줄을 수정하고 감사 로그를 추가한 별도 학습 노트북 제작이다. 이 수정본의 실제 T4 학습 완료나 평가 결과는 확인되지 않았다. 마지막 두 사용자 메시지는 “셀에 있는거 쭉 다 실행하면 되는 거지?”였고 컨텍스트 한도 오류로 답변되지 않았다.

## 완료된 실행 및 진단

1. PO forget5 초기 학습과 평가 실행. 로컬 `results/po_forget5_eval_2026-10-10/`에 결과 보존.
   - forget 400, retain 400, APE 1,200건.
   - 당시 summary: Forget EM 2.1875%, APE 0.770833%, Retain ROUGE-L 85.458386%.
   - 이후 slow tokenizer 및 generation 경계 문제가 발견되어 논문 재현 확정값으로 취급하지 않는다.
2. 초기 GA 학습/평가 이후 fast tokenizer로 GA를 처음부터 재학습하고 평가.
   - 과거 slow GA Forget EM 약 42.44%, fast GA 약 44.77% (대화 기록).
   - fast GA Retain ROUGE-L 약 92.70%. 이전 대화에서 인용한 논문 GA EM은 14.0%; 이번 조사에서 논문 표를 재검증하지 않았다.
   - fast 학습의 step_4~step_32 및 최종 checkpoint 저장, Kaggle Version #1 Successful 보고.
3. 저장된 fast GA 진단 출력에서 확인:
   - step_4 → step_32의 LoRA 448개와 projector 4개 텐서 모두 변경.
   - 검사한 checkpoint에서 NaN/Inf 없음.
   - step_32와 최종 checkpoint의 변경 텐서 0개.
   - epoch 마지막 LR이 2e-5 → 2.19e-7(epoch 4) → 1.91e-5(epoch 7)로 다시 상승.
   - optimizer 업데이트당 scheduler가 두 칸 진행하는 코드 경로와 일치하는 패턴을 발견했다. 스케줄 문제가 성능 격차 전체를 설명하는지는 미확인.
   - 당시 로그만으로 overflow skip 수와 정확한 성공 업데이트 수는 복원 불가.

## 지금 사용할 파일

- 학습: `notebooks/GA_forget5_fast_scheduler_fix_train_kaggle.ipynb`
- 평가: `notebooks/FIUBench_forget5_fast_eval_kaggle.ipynb`
- 감사 구현: `scripts/ga_scheduler_audit.py`
- CPU 제어 로직 테스트: `scripts/test_ga_scheduler_audit.py` (이전 대화에서 7개 통과 보고; 이번 복구 작업에서 재실행하지 않음)
- 과거 실행 진단: `scripts/diagnose_ga_fast_saved.py`

수정 학습 노트북은 공개 Stage-I에서 새로 시작한다. 이전 GA step32를 이어 학습하지 않는다. Kaggle 복사본에 호환 패치를 적용하고 별도 `forget_ga_scheduler_audit.py`를 생성한다. 로컬 `forget.py` 자체에 scheduler 수정이 들어간 것은 아니다.

설정: GA, forget5 20명/400 QA, fast tokenizer, seed 233, LR 2e-5, cosine + warmup 0.06, 8 epochs, batch 4/GPU, GPU 2개, accumulation 16, LoRA 128/256/dropout .05, FP16/SDPA/ZeRO-2. 완전히 누적된 업데이트의 effective batch는 128이며 epoch 끝의 짧은 누적 구간은 기존 동작을 유지한다.

학습 라이브러리: transformers 4.46.3, tokenizers 0.20.3, peft 0.13.2, accelerate 1.14.0, deepspeed 0.19.7. 현재 fast 평가 노트북에는 accelerate 0.34.2가 실제로 남아 있다. 따라서 학습/평가의 모든 패키지가 동일하다고 말하면 안 된다. 평가 실행 전에 이 차이의 필요성과 영향을 확인한다.

## 다음 실행 순서

1. 수정 학습 노트북을 새 Kaggle 노트북으로 가져오고 T4 ×2, Internet ON 설정.
2. 설치 셀 실행 → 포함된 커널 재시작 셀 실행 → 연결 복귀 후 환경 확인 셀부터 아래로 실행. 재시작 셀을 반복하거나 Stop Session으로 세션을 없애지 않는다.
3. 사전검사 후 학습 셀 한 번 실행. 첫 epoch 검사를 통과하면 같은 실행이 8 epochs까지 계속된다.
4. `TRAIN_PROCESS_EXIT_0`, `SCHEDULER_AND_CHECKPOINT_AUDIT_OK`, epoch별 LR/성공/skip 표 확인.
5. `step_32`는 업데이트 시도 32회다. 성공 횟수는 overflow에 따라 작을 수 있다. 두 rank의 scheduler index와 성공 횟수가 일치해야 한다.
6. 새 모델 폴더 `models/llava_phi_GA_forget5_fast_scheduler_fix`와 scheduler_rank0/1.jsonl, console log, environment, manifest, patch, token signature, dataset revision 및 실행 스크립트를 저장된 Kaggle Output으로 보존.
7. 검증 후 fast 평가 노트북에 이번 Output만 연결하고 METHOD="GA"로 평가. 기존 GA와 동일한 평가 방식으로 비교. Core 먼저, APE는 별도 선택 실행.

## 해석 시 유지해야 할 정정과 미완료 항목

- fast tokenizer 사용만으로 논문 수치에 가까워질 것이라는 과거 주장은 실험으로 뒷받침되지 않았다.
- Stage-I ROUGE가 논문과 비슷하다는 것만으로 checkpoint가 동일하다고 입증할 수 없다. 모델 내부의 `2_epochs` 경로명만으로 학습 이력을 확정할 수도 없다.
- 논문의 ROUGE 열 자체와 종합 utility를 혼동하지 않는다. 평가 표본/설정을 확인하고 비교한다.
- raw Truth Ratio 평균과 집계 코드가 변환한 Truth 점수를 직접 비교하지 않는다.
- raw Min-K 평균을 MIA AUROC로 취급하지 않는다. **10월 12일 정정:** 논문 자체가 MIA를 AUROC로 보고한다는 근거도 없다. 논문은 Min-K% Prob를 정의하고 공개 집계 코드는 mink를 평균한다.
- KS retain-only baseline, GPT-Eval, MME, POPE 등 전체 논문 평가 완료는 확인되지 않았다.
- PO의 공개 코드 retain pool 336명과 논문 정의 380명의 차이가 기록되어 있다. GA loss에는 retain 데이터가 사용되지 않는다.
- GPU 실험의 실행 성공, 구현 문제 수정, 논문 성능 재현은 각각 별도로 확인해야 한다.
