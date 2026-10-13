# FIUBench 방법 선택과 실험 의사결정 타임라인 — 2026-10-13

이 문서는 `코드 구조 분석 및 설명`, `Connect FIUBench repository`, `FIUBench 진행 상황 정리` 채팅을 시간순으로 다시 읽고, **왜 그 방법과 실험을 선택했는지**를 복원한 기록이다. 결과 자체는 `FIUBENCH_EXPERIMENT_LEDGER_2026-10-13.md`에 따로 정리돼 있다.

각 항목은 다음을 구분한다.

- **당시 문제/목표**: 그 시점에 알고 있던 상황.
- **선택 이유**: 다음 실험을 한 이유.
- **뒤에 확인된 사실**: 결과가 나온 뒤 바뀐 판단.

## 전체 흐름

```text
논문·코드 구조 파악
  → 로컬 M1 대신 Kaggle 선택
  → PO smoke test로 전체 파이프라인 확인
  → 단일 T4 PO 본 실험
  → PO 핵심 지표의 부분 재현
  → 저자 환경/dual-T4에 가까운 후속 PO
  → 이전 PO와 다른 결과 발생
  → retain 구성 영향을 피하려 GA 선택
  → slow GA 실패
  → tokenizer 차이를 제거하려 fast GA
  → fast GA도 실패
  → 실제 LR 이상을 발견해 scheduler-fixed GA
  → 수정 후에도 실패
  → checkpoint 적용 및 중간 step 진단
  → DeepSpeed/분산을 제거한 single-GPU GA
  → GA 결과가 거의 동일
  → 이전 성공 PO를 공통 fast_eval의 대조군으로 사용
  → 후속 PO도 같은 fast_eval로 재평가
  → 학습 경로 차이를 분리하려 single-GPU PO
  → slow 실행 및 평가 완료: 논문/이전 성공 PO에 다시 근접
  → fast-tokenizer single-GPU PO를 다음 통제 실험으로 준비
```

## 1. 10월 3일: 논문과 코드 구조부터 분석

### 당시 문제/목표

처음에는 `forget.py`, `inference.py`, `evaluate_util.py`, `aggregate_eval_stat.py`가 어떻게 이어지는지와 논문의 Stage-I/Stage-II 구조를 이해하는 것이 목표였다. 로컬 컴퓨터가 M1 Mac이라 3B VLM 학습을 직접 수행하기 어려웠다.

### 선택

- 로컬에서는 코드 분석과 파일 수정을 하고, GPU 실험은 Kaggle에서 실행하기로 했다.
- 처음부터 모든 방법을 돌리지 않고, 먼저 한 방법으로 데이터 준비 → 모델 로딩 → LoRA 학습 → checkpoint 저장 → 추론/평가까지 한 번 관통하기로 했다.

### 선택 이유

- Kaggle T4는 무료 사용 시간이 제한돼 있어 환경 오류를 본 학습 중에 발견하면 비용이 컸다.
- Stage-I를 직접 다시 학습하는 대신 저자가 공개한 `gray311/vlm_unlearning_ft_llava_phi_3_mini`를 사용하면 Stage-II 방법 비교에 집중할 수 있었다.
- 발표까지 시간이 짧아 Forget1/5/10 전부보다 **Forget5 하나**를 고정하는 편이 실험 간 비교가 쉬웠다. Forget5는 20명, 400 QA로 평가 시간도 관리 가능했다.

### 뒤에 확인된 사실

공개 Stage-I 모델의 Retain ROUGE가 논문 기준과 가깝게 나와, Stage-I 전체 재학습은 계속 보류할 수 있었다. 다만 이것만으로 내부 가중치가 논문 실행과 완전히 같다고 증명되지는 않았다.

## 2. 10월 3~4일: 첫 방법으로 PO를 선택

### PO를 선택한 이유

PO는 현재 저장소에서 Kaggle로 옮기기 가장 쉬운 Stage-II 방법이었다.

- frozen oracle model이 필요 없다.
- KL/ICD처럼 모델 두 개를 동시에 GPU에 올릴 필요가 없다.
- DeepSpeed 없이 단일 GPU Accelerate로 실행할 수 있다.
- 목표가 직관적이다. Forget 질문에는 IDK/거절 답변을 학습하고, 별도 retain QA에는 원래 답을 유지하도록 학습한다.
- 작은 표본 출력만 봐도 “개인정보를 답하던 모델이 거절하기 시작하는지”를 빠르게 확인할 수 있다.

즉 PO를 먼저 고른 목적은 “PO가 최선의 unlearning 방법이어서”가 아니라 **Kaggle 포팅과 전체 학습·평가 파이프라인을 가장 낮은 복잡도로 검증하기 위해서**였다.

### smoke test를 먼저 선택한 이유

- 본 실험 전에 이미지 경로, tokenizer, processor, LoRA 대상, gradient checkpointing, 저장 로직을 확인하려 했다.
- 짧은 update만 수행해 checkpoint가 만들어지고 inference가 가능한지 확인했다.
- smoke 결과에서 Forget 질문에 거절 응답이 나타나 전체 경로가 작동한다는 예비 증거를 얻었다.

### 낮은 LoRA rank나 짧은 설정을 본 실험에 쓰지 않은 이유

초기에는 메모리 절약을 위해 축소 설정도 논의됐지만, 사용자가 “실제 논문 재현에서는 다시 원래 설정으로 돌려야 한다”고 지적했다. 그래서 smoke test와 본 실험 설정을 분리하고, 본 실험에는 논문의 LoRA `r=128`, `alpha=256`, dropout 0.05, 8 epochs를 사용했다.

## 3. 10월 5일: Forget5만 고정하고 PO 본 실험

### 당시 문제/목표

발표와 Kaggle 주간 GPU 시간 제한 때문에 모든 split과 모든 방법을 전부 돌리기 어려웠다.

### 선택

- Forget5를 고정했다.
- 이미 smoke test가 통과한 PO를 먼저 끝까지 실행했다.
- 공개 Stage-I checkpoint에서 Stage-II만 학습했다.

### 선택 이유

- Forget5 하나에서 여러 방법을 비교하는 편이 split 5/10을 PO 하나로 비교하는 것보다 논문의 방법 비교를 설명하기 좋았다.
- PO 파이프라인은 이미 가장 많이 검증돼 있었기 때문에 제한 시간 안에 실제 결과를 얻을 가능성이 가장 높았다.

### batch 4에서 batch 2 / accumulation 64로 바꾼 이유

- 단일 T4에서 batch 4 본 학습이 메모리 문제를 냈다.
- 논문과 같은 effective batch 128을 유지하면서 메모리를 줄이기 위해 batch 2, accumulation 64를 선택했다.
- 실패한 batch 4 모델을 이어 학습하지 않고 공개 Stage-I를 새로 로드했다.

### 뒤에 확인된 사실

이 실행이 결과적으로 가장 논문과 가까운 PO가 됐다.

- Forget EM 6.50% 대 논문 6.90%
- Retain ROUGE-L 약 73.12% 대 논문 Rouge 72.8

그래서 이후 실험에서 이 모델은 **성공 대조군** 역할을 하게 됐다.

## 4. 10월 5~6일: inference와 공식 평가를 함께 사용

### `inference.py`를 먼저 사용한 이유

- 생성 문장을 직접 보고 PO가 거절을 학습했는지 빠르게 확인할 수 있었다.
- Forget5와 Retain5 소수 표본으로 잘못된 이미지, prompt, checkpoint 로딩을 빨리 발견할 수 있었다.
- 발표 관점에서도 실제 질문/응답 예시가 필요했다.

### 이후 `evaluate_util.py`와 집계 경로를 사용한 이유

- 논문 수치와 비교하려면 정성적 예시가 아니라 400개 전체의 EM, ROUGE, truth ratio, MinK 계열 값이 필요했다.
- PO의 거절률만으로는 knowledge removal과 단순 refusal을 구분할 수 없었다.

### 깨끗한 commit으로 되돌린 이유

평가 호환성 문제를 고치면서 generation 경계와 평가 로직까지 수정될 위험이 커졌다. 사용자는 “평가 로직은 수정하지 말고 한계로 기록하자”고 결정했다. 따라서:

- 학습 알고리즘과 논문 평가 정의는 최대한 유지하고,
- Kaggle 경로, optional dependency, 로딩 호환성 같은 최소 변경만 다시 적용하는 방향으로 정리했다.

이 결정은 이후 “코드 포팅”과 “논문 방법 변경”을 구분하는 기준이 됐다.

## 5. 10월 8~9일: 환경 확인 노트북과 후속 dual-T4 PO

### 환경 확인 노트북을 따로 만든 이유

Kaggle 기본 Python/PyTorch와 저자 requirements 사이 충돌이 반복됐다. 한 노트북에서 설치 실험과 본 학습을 계속하면 어떤 패키지 상태로 모델이 학습됐는지 추적하기 어려웠다.

그래서:

1. 환경 확인 노트북에서 GPU, 패키지, 데이터, 모델, LoRA, 1-batch forward/backward를 검증하고,
2. 검증된 조합만 본 PO 노트북에 옮기기로 했다.

### 저자 버전에 가까운 legacy stack을 선택한 이유

논문 재현에서는 최신 버전 포팅보다 저자 환경과 가까운 버전을 먼저 써야 한다고 판단했다.

- Transformers 4.37.2
- tokenizers 0.15.1
- PEFT 0.8.2
- DeepSpeed 0.19.7 및 Accelerate 1.14.0은 현재 Kaggle/PyTorch와 작동하는 조합으로 조정

PyTorch/Python까지 저자 환경으로 내리는 것은 Kaggle Python 3.13과 wheel 호환성 때문에 어렵다고 봤다.

### dual T4 + DeepSpeed PO를 선택한 이유

- 논문 batch size 8을 T4 한 장에 올리기 어려웠다.
- T4 두 장에 batch 4씩 배치하면 global microbatch 8을 만들 수 있었다.
- accumulation 16을 적용해 effective batch 128을 맞출 수 있었다.
- CPU offload를 사용해 메모리 부족을 줄이려 했다.

이 후속 PO는 **이전 성공 PO와 다른 실행 엔진에서도 논문 설정을 재현할 수 있는지** 확인하려는 시도였다.

### slow tokenizer를 사용한 이유

legacy `tokenizers==0.15.1`이 공개 모델의 `tokenizer.json`을 읽지 못했다. 본 학습을 진행하기 위한 우회로로 `use_fast=False`를 강제했다. 당시에는 저자 라이브러리 버전에 가까워지는 이점이 더 크다고 판단했다.

### 뒤에 확인된 사실

- fast와 slow tokenizer는 400개 GA 입력뿐 아니라 PO 입력/labels도 전부 다르게 만들었다.
- 후속 PO는 학습에 성공했지만 이전 성공 PO와 다른 결과를 냈다.
- “저자 버전에 가깝다”와 “저자가 실제로 사용한 tokenization을 재현한다”가 같은 말이 아니었다.

## 6. 10월 10일: PO 차이를 설명하기 위해 GA로 전환

### 당시 문제/가설

후속 PO의 결과가 이전 PO와 달랐고, PO는 forget IDK 데이터와 retain training pool을 함께 사용한다. 당시에는 공개 코드가 Forget1/5/10을 모두 제외해 6,720 QA retain pool을 만드는 점이 논문 정의와 다를 수 있다는 의심이 컸다.

### GA를 선택한 이유

- GA는 forget 데이터만 사용하므로 retain pool 구성 논쟁을 제거할 수 있다.
- PO와 목적함수가 완전히 달라, 문제가 PO 특유의 IDK/refusal 학습인지 확인할 수 있다.
- oracle model이 필요 없어 KL/ICD보다 T4에서 실행하기 쉽다.
- 논문 Table 2에 GA 기준값이 있어 직접 비교할 수 있다.

사용자는 이 이유로 `data_module.py`를 고치지 말고 GA를 먼저 하자고 결정했다. **GA에서는 retain batch를 loss에 쓰지 않기 때문**이다.

### 초기 slow GA를 선택한 이유

후속 PO와 같은 검증된 Kaggle 환경을 재사용하면 방법만 PO→GA로 바꿀 수 있었다. 따라서 우선 slow-tokenizer/dual-T4 환경에서 GA를 실행했다.

### 뒤에 확인된 사실

Forget EM이 약 42.44%로 논문 GA 14.0%보다 훨씬 높았다. retain pool 문제를 제거해도 논문 결과가 나오지 않았으므로 PO 차이를 retain 구성 하나로 설명할 수 없게 됐다.

## 7. 10월 10일: fast-tokenizer GA를 새로 학습

### 선택 이유

slow GA 실패 뒤 fast/slow tokenization을 직접 비교하자 다음이 확인됐다.

- GA forget 400/400에서 input IDs와 labels가 다름.
- PO IDK/retain에서도 전부 차이가 남.
- 저자 평가 경로와 공개 tokenizer 자산은 fast tokenizer를 전제로 한 흔적이 있음.

따라서 slow-tokenizer 학습 자체가 GA 실패 원인인지 분리하기 위해:

- Transformers 4.46.3
- tokenizers 0.20.3
- PEFT 0.13.2
- fast tokenizer

환경에서 GA를 Stage-I부터 다시 학습했다. 이전 slow GA checkpoint를 이어 쓰지 않은 이유는 입력 토큰 체계가 달랐기 때문이다.

### 왜 PO가 아니라 GA를 fast로 먼저 했는가

- GA가 retain pool을 사용하지 않아 변수 수가 적었다.
- 후속 PO보다 짧고, forward도 forget batch 하나라 실행 시간이 적었다.
- 우선 tokenizer 가설만 검증하기 좋았다.

### 뒤에 확인된 사실

fast GA EM은 약 44.77%로 오히려 slow GA와 비슷했다. 따라서 “slow tokenizer 때문에 GA가 실패했다”는 가설은 지지되지 않았다. fast tokenizer는 평가·입력 일관성을 위해 계속 사용했지만 해결책으로 보지는 않게 됐다.

## 8. 10월 10~11일: Stage-I와 전체 환경을 바로 재학습하지 않은 이유

### 당시 후보

GA가 계속 실패하자 Python 3.10, 저자 PyTorch/CUDA, 모든 라이브러리, Stage-I 학습까지 다시 맞추자는 선택지가 나왔다.

### 먼저 Stage-I 성능을 확인한 이유

- 공개 모델 config 안의 `2_epochs_lr1e-05` 경로명이 논문의 Stage-I 10 epochs/LR 2e-5와 달라 보였다.
- 하지만 경로명은 저장 당시 문자열일 뿐 실제 이력을 보장하지 않는다.
- Stage-I 전체 재학습은 가장 비싸고, 잘못된 가설이면 시간을 크게 잃는다.

기존 base/retain 평가에서 ROUGE가 약 93.2~93.3으로 논문 기준과 가까웠기 때문에 Stage-I 불일치를 최우선 원인으로 두지 않았다. 그래서 “현재 checkpoint에 8 epoch를 더 붙이는” 방법도 선택하지 않았다. optimizer/scheduler/data order가 끊긴 상태라 논문의 10-epoch Stage-I가 되지 않고, 이미 수행한 unlearning을 되돌릴 수도 있기 때문이다.

## 9. 10월 11~12일: scheduler-fixed GA

### 선택 이유

fast GA 로그에서 cosine LR이 중간에 거의 0까지 내려간 뒤 마지막에 다시 최대치 근처로 올라가는 패턴이 발견됐다. dual GPU Accelerate scheduler가 optimizer update 한 번에 두 칸 진행하는 경로와 수치가 일치했다.

이는 추측이 아니라 실제 로그와 계산이 맞는 구체적인 비동등성이었다. 그래서 다른 모든 설정을 유지하고:

- 성공한 optimizer update 1회당 scheduler 1회만 진행,
- FP16 overflow skip 시 scheduler도 멈춤,
- attempt/success/skip/LR 감사 로그 저장

을 적용해 GA를 처음부터 다시 학습했다.

### 왜 이전 checkpoint에서 이어 하지 않았는가

기존 모델은 잘못된 LR trajectory 전체를 이미 거쳤다. 중간에서 scheduler만 고쳐 이어 가면 수정 효과를 분리할 수 없기 때문에 공개 Stage-I부터 다시 시작했다.

### 뒤에 확인된 사실

- 32 attempts, 30 successful, 2 skips로 감사 자체는 정상 동작했다.
- Forget EM 42.27%, Retain ROUGE-L 92.70%로 여전히 논문과 달랐다.

따라서 scheduler 문제는 실제 버그였지만 **GA 재현 실패의 주원인은 아니었다**.

## 10. 10월 12일: checkpoint effect diagnostic

### 선택 이유

학습 코드가 정상 종료돼도 다음 가능성이 남아 있었다.

- 평가에서 LoRA checkpoint가 실제로 적용되지 않음.
- merge 과정에서 가중치가 사라짐.
- final checkpoint 경로를 잘못 선택함.
- 모델은 변했지만 생성에 영향을 줄 정도가 아님.

전체 400개 재평가나 재학습보다 20개 표본에서 base와 GA checkpoint의 loss, answer probability, EM, prediction 변화, merge 동등성을 확인하는 것이 빠른 진단이었다.

### 뒤에 확인된 사실

GA checkpoint는 loss와 예측을 실제로 바꿨다. 따라서 이후에는 “checkpoint 미적용”을 원인 후보에서 내렸다.

## 11. 10월 12일: checkpoint trajectory diagnostic

### 선택 이유

저자 답변과 논의에서 unlearning 결과가 학습 step 및 checkpoint 선택에 민감하고 early stopping을 사용했다는 단서가 나왔다. final step만 평가했기 때문에 중간 step에 논문과 가까운 지점이 있었는지 확인할 필요가 있었다.

그래서 step 4, 8, ..., 32와 final을 20 forget + 20 retain 표본에 평가했다. full Core 평가를 모든 checkpoint에 돌리기 전에 유망한 step만 찾으려는 저비용 선별 실험이었다.

### 뒤에 확인된 사실

- step이 진행될수록 forget loss는 꾸준히 증가했다.
- 작은 표본의 Forget EM은 step 12 이후 0.45로 같았다.
- retain EM을 더 보존하면서 forget EM이 final보다 낮은 중간 step은 없었다.

따라서 현재 trajectory에서는 early stopping이 명백한 해결책으로 보이지 않았다. 이 결과는 20+20 표본 진단이므로 저자 선택 규칙 자체를 부정하는 증거는 아니다.

## 12. 10월 12일: single-GPU/no-DeepSpeed GA

### 선택 이유

scheduler를 고쳐도 실패했으므로, 후속 GA와 이전 성공 PO 사이의 가장 큰 실행 차이인 다음 요소를 한꺼번에 제거하려 했다.

- dual GPU 분산
- DeepSpeed ZeRO-2와 CPU offload
- Accelerate 1.14 경로
- batch 4/GPU × 2

대신 이전 성공 PO와 같은 계열의 환경을 사용했다.

- 단일 T4
- no DeepSpeed
- Accelerate 0.34.2
- fast tokenizer
- batch 2 / accumulation 64
- effective batch 128

GA를 선택한 채 실행 엔진만 이전 성공 PO와 가깝게 만든 통제 실험이었다.

### 뒤에 확인된 사실

Forget EM 41.31%, Retain ROUGE-L 92.61%로 dual-T4 scheduler-fixed GA와 거의 같았다. 따라서 GA 실패를 DeepSpeed, dual GPU 또는 FP16 skip만으로 설명하기 어려워졌다.

## 13. 10월 12~13일: 이전 성공 PO를 공통 평가 대조군으로 사용

### 선택 이유

GA는 방법 자체가 공개 코드/논문 사이에서 재현되지 않는 문제일 수 있었다. 반면 과거 PO는 논문과 가까운 결과를 냈다. 그래서 평가 코드가 정상인지 확인하려면 **이미 성공했다고 아는 PO checkpoint**를 현재 `fast_eval`에서 다시 평가하는 것이 가장 강한 대조군이었다.

이 선택의 목적은 새 모델을 만드는 것이 아니라:

- 현재 평가 노트북이 논문과 가까운 기존 모델을 여전히 가깝게 측정하는지,
- 과거 `inference.py` 결과와 현재 Core 평가가 어느 정도 일치하는지,
- GA의 높은 EM이 평가 파이프라인 전체의 고장인지

를 분리하는 것이었다.

### 뒤에 확인된 사실

이전 PO는 현재 fast_eval에서도 EM 6.50%, Retain ROUGE-L 73.12%를 냈다. 따라서 fast_eval은 최소한 이 체크포인트에서는 논문과 가까운 값을 복원했다. GA 실패를 평가 코드 하나로 설명하기 어려워졌다.

## 14. 10월 13일: 후속 PO도 같은 fast_eval로 평가

### 선택 이유

이전에는 PO-1은 fast_eval, PO-2는 전용 slow 평가처럼 체크포인트와 평가 경로가 동시에 달랐다. PO끼리 왜 다른지 보려면 checkpoint만 바꾸고 평가를 같게 해야 했다.

따라서 후속 PO도 같은 fast_eval에 넣어:

- 평가 경계/assistant tag 때문에 차이가 났는지,
- 실제 학습된 checkpoint가 다른지

를 분리했다.

### 뒤에 확인된 사실

후속 PO는 같은 fast_eval에서도 EM 1.875%, Retain ROUGE-L 83.27%로 이전 PO와 달랐다. 평가 경로 차이만으로 두 PO의 차이를 설명할 수 없었다. 학습 환경, tokenizer, batch 구성, precision/optimizer 경로가 남은 차이가 됐다.

## 15. 10월 13일: single-GPU PO batch 4 / accumulation 32

### 선택 이유

GA에서 single GPU와 dual GPU 결과가 같았다고 해서 PO에서도 분산 영향이 없다고 단정할 수는 없었다. PO는 forget IDK와 retain batch를 함께 처리하고 LR도 15배 높아 실행 경로 영향이 다를 수 있다.

사용자가 요청한 batch 4 / accumulation 32는:

- 단일 T4로 dual-GPU 변수를 제거하고,
- effective batch 128을 유지하며,
- 후속 PO의 microbatch 크기에 더 가깝게 맞추려는 설정이었다.

이 실행은 **분산 제거 통제 실험**으로 선택됐다.

### 실제 실행과 주의점

Kaggle에 먼저 올라간 실행본은 slow tokenizer/legacy 계열이었다. 학습은 약 2시간 24분 만에 완료됐다. fast_eval 결과는 Forget EM 7.625%, Retain ROUGE-L 76.47%였다.

이 값은 후속 dual-T4 PO의 EM 1.875% / Retain ROUGE-L 83.27%보다 이전 성공 PO의 6.50% / 73.12%에 훨씬 가깝다. 따라서 PO에서는 dual-GPU 분산, rank별 batch 묶음, accumulation 구조, 난수 소비처럼 world size와 함께 바뀐 요소가 결과에 실제로 영향을 줬다는 근거가 생겼다. effective batch 128이 같다는 사실만으로 동등한 학습이 되지는 않았다.

다만 raw MinK++는 19.43%로 이전 성공 PO의 67.03%와 크게 다르다. EM과 retain ROUGE가 가까워졌다고 PO의 모든 forget 특성이 재현된 것은 아니다.

## 16. 10월 13일: fast-tokenizer single-GPU PO를 별도로 준비

### 선택 이유

single-GPU PO가 진행된 뒤, 이전 성공 PO가 fast tokenizer였다는 점이 다시 확인됐다. slow로 실행하면 “분산만 제거한 비교”는 할 수 있지만 이전 성공 PO와 tokenizer까지 맞춘 재현은 아니다.

그래서 로컬 노트북을 다음처럼 수정했다.

- single T4
- batch 4 / accumulation 32
- fast tokenizer
- Transformers 4.46.3 / tokenizers 0.20.3 / PEFT 0.13.2

이 노트북은 **tokenizer와 주요 라이브러리를 성공 PO 쪽으로 옮긴 다음 통제 실험**이다. 아직 실행하지 않았다. 이미 끝난 slow PO의 결과를 먼저 보고 실행 여부를 정하는 것이 효율적이다.

## 17. Colab/병렬 작업을 고려한 이유

현재 한 Kaggle 세션에서 학습과 평가를 순차 수행하느라 결과 확인이 늦어졌다. Colab 구독을 고려한 목적은 알고리즘을 바꾸는 것이 아니라:

- Kaggle에서는 학습,
- Colab 또는 다른 세션에서는 저장된 checkpoint 평가,
- 더 좋은 GPU가 배정되면 batch 8 또는 더 빠른 평가

를 병렬로 수행해 시간을 줄이기 위해서였다. 아직 Colab에서 새 학습 결과를 만든 것은 아니다. GPU가 바뀌더라도 핵심 패키지와 데이터/모델 revision은 고정하고, precision·batch만 하드웨어별로 명시해야 한다.

## 방법별 선택 이유 요약

### PO

**왜 선택했나:** oracle model과 DeepSpeed 없이도 가능하고, 거절 응답을 통해 파이프라인 성공 여부를 쉽게 확인할 수 있으며, Kaggle 단일 T4에서 실행 가능했기 때문이다.

**무엇을 확인했나:** 첫 실행은 논문 핵심 지표를 부분 재현했다. 후속 실행들은 평가보다 학습 경로가 PO 결과를 크게 바꿀 수 있음을 보여줬다.

### GA

**왜 선택했나:** PO 결과 차이에서 retain pool 변수를 제거하고, forget-only 목적함수로 논문의 다른 대표 방법을 비교하기 위해서다. KL/ICD보다 T4에서 가볍다.

**무엇을 확인했나:** tokenizer, scheduler, DeepSpeed, GPU 수를 바꿔도 논문 GA가 재현되지 않았다. checkpoint는 적용됐지만 망각 강도가 부족했다.

### GD

**왜 실행하지 않았나:** 발표 시간과 GPU 예산이 제한된 상태에서 PO와 GA조차 실행 동등성이 확립되지 않았다. GD는 forget ascent와 retain descent를 함께 다루므로 GA보다 변수가 늘어난다. 기본 파이프라인이 검증되기 전에 추가하면 원인 추적이 더 어려웠다.

### KL / ICD

**왜 실행하지 않았나:** frozen oracle model을 함께 올리고 DeepSpeed 경로를 써야 해 T4 메모리와 호환성 부담이 컸다. PO/GA의 기본 재현 문제가 해결되지 않은 상태에서 우선순위가 낮았다.

### Retain model / Stage-I 재학습

**왜 직접 학습하지 않았나:** 저자가 공개 checkpoint를 제공했고, base/retain 성능이 논문 ROUGE와 가까웠다. Stage-I 전체 재학습은 가장 비싸며 Stage-II 문제를 해결한다는 증거가 없었다. Retain model은 주로 KS 등 forget quality 비교의 기준선으로 필요했고, PO/GA 학습 checkpoint 자체는 아니었다.

## 평가 방법별 선택 이유 요약

### 소수 표본 inference

생성 문장을 직접 확인해 이미지·prompt·checkpoint 연결을 빠르게 검증하기 위해 사용했다.

### Core 400+400 평가

논문과 비교 가능한 EM/ROUGE/truth/MinK 계열을 얻기 위해 사용했다. 매번 APE와 외부 평가까지 돌리기 전에 핵심 결과를 보는 용도였다.

### fast_eval

이전 성공 PO, 후속 PO, GA를 동일한 tokenizer/generation 경로에서 비교하고, legacy slow tokenizer의 영향을 제거하기 위해 만들었다.

### APE

논문의 forget quality 구성 요소라 필요하지만 Core보다 표본 수와 시간이 크므로 Core 결과가 정상인 checkpoint에 선택적으로 돌리려 했다.

첫 성공 PO(PO-1)를 동일 `fast_eval` 경로로 1,200개 APE 표본에 평가했다. 결과는 **2.9583%**로 논문 PO 6.80%보다 3.8417%p 낮았다. 패러프레이즈된 질문에 정답을 덜 내는 방향은 맞으므로 실행 실패가 아니다. 다만 PO는 거절 응답을 학습하는 방법이라, 이 차이를 지식 삭제 향상으로 단정하지 않고 더 강한 답변 억제 가능성까지 함께 기록한다.

### GPT-Eval, MME, POPE, KS

외부 API·추가 데이터·retain-only baseline 및 실행 시간이 필요해 초기 디버깅 단계에서는 보류했다. Core가 논문과 동등하지 않은 상태에서 이 평가들을 먼저 끝내도 학습 원인을 찾는 데 도움이 적었다.

### checkpoint effect/trajectory 진단

새 학습을 반복하기 전에 checkpoint 미적용과 early-stopping 가능성을 저비용으로 배제하기 위해 선택했다.

## 의사결정 과정에서 폐기되거나 약해진 가설

| 당시 가설 | 선택한 검사 | 현재 판단 |
|---|---|---|
| slow tokenizer가 GA 실패의 주원인 | fast GA 재학습 | fast도 비슷해 주원인 아님 |
| 잘못된 scheduler가 GA 실패의 주원인 | scheduler-fixed GA | 실제 버그였지만 결과 해결 못 함 |
| DeepSpeed/dual GPU가 GA 실패의 주원인 | single-GPU no-DS GA | 결과가 거의 같아 가능성 낮음 |
| checkpoint가 평가에 적용되지 않음 | checkpoint effect diagnostic | 적용됨을 확인, 배제 |
| final 대신 중간 step이면 논문과 가까움 | trajectory diagnostic | 작은 진단에서 근거 없음 |
| 평가 코드가 모든 높은 GA EM의 원인 | 성공 PO를 같은 fast_eval로 평가 | 성공 PO는 논문과 가까워 전면적 평가 고장 아님 |
| 두 PO 차이가 평가 경로 때문 | 후속 PO를 같은 fast_eval로 평가 | 차이가 남아 실제 학습 경로 차이 존재 |
| 공개 Stage-I가 명백히 다른 모델 | base Retain ROUGE 비교 | 표면 성능은 가까워 우선순위 낮음 |

## 현재 의사결정 위치

single-GPU slow PO는 논문 및 이전 성공 PO에 가까워졌다. 현재 가장 강한 결론은 **PO에서는 single/dual 실행 경로와 그에 딸린 gradient 구성 차이가 결과에 중요했다**는 것이다. slow tokenizer로도 EM과 retain ROUGE가 돌아왔으므로 tokenizer 하나가 PO 차이의 주원인이라는 설명은 약해졌다.

그다음 실험 후보는 fast-tokenizer single-GPU PO다. 이 실험은 성공 PO와 현재 PO-3 사이에 남은 tokenizer/라이브러리 및 batch 2 대 4 차이를 좁히고, 특히 크게 다른 MinK 계열이 어느 조건에서 생겼는지 확인하기 위해 준비된 것이다. 발표용 핵심 결과가 급하다면 PO-3를 부분 재현 결과로 먼저 보존하고, 새 학습보다 APE와 provenance 저장을 우선할 수 있다.

