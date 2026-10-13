# FIUBench Forget5 실험 원장 — 2026-10-13

이 문서는 지금까지 확인된 **학습 실행**, **체크포인트 평가**, **논문 비교**, **실패 및 미완료 항목**을 한곳에 모은 기록이다. 같은 체크포인트도 평가 경로에 따라 숫자가 달라졌으므로 학습과 평가를 분리해서 적었다.

## 1. 먼저 결론

1. 지금까지 논문 수치에 가장 가까웠던 결과는 **이전 단일 T4 PO**다. 현재 `fast_eval` 기준 Forget EM 6.50%, Retain ROUGE-L 73.12%로 논문 PO의 EM 6.90%, Rouge 72.8과 매우 가깝다. 동일 fast 경로 APE는 1,200개 표본에서 2.9583%로 논문 6.80%보다 3.8417%p 낮았다. 망각 방향은 맞지만 정확한 APE 수치 재현은 아니다. KS, GPT-Eval, MME, POPE 등 전체 평가를 끝내지 않았으므로 “논문 전체 재현 완료”로 부르지는 않는다.
2. **후속 dual-T4 PO**도 학습에는 성공했다. Forget EM은 1.875%로 더 낮고 Retain ROUGE-L은 83.27%로 더 높다. 그러나 논문 PO의 MIA 64.2와 달리 현재 raw MinK 계열도 크게 내려갔고, 논문과 같은 checkpoint 선택점이라고 보기 어렵다. “실패한 모델”이라기보다 **논문과 다른 작동점에 도달한 모델**이다.
3. **GA는 실행 경로를 바꿔도 논문 성능이 재현되지 않았다.** scheduler 수정 dual-T4 GA는 EM 42.27%, 단일 T4/no-DeepSpeed GA는 41.31%였다. 논문 GA의 EM 14.0%와 큰 차이다. 두 실행의 Retain ROUGE-L도 각각 92.70%, 92.61%로 거의 같다. 따라서 분산 학습, DeepSpeed, scheduler skip만으로 GA 실패를 설명할 수 없다.
4. GA checkpoint는 실제로 적용됐고 forget loss와 예측을 변화시켰다. 다만 변화량이 논문 GA의 망각 수준까지 가지 않았다. 20+20 표본 trajectory에서는 중간 checkpoint가 최종보다 뚜렷하게 나은 지점도 없었다.
5. **단일 T4 PO batch 4 / accumulation 32 slow-tokenizer 실행**은 fast_eval에서 Forget EM 7.625%, Retain ROUGE-L 76.47%를 냈다. 논문 및 PO-1에 다시 가까워졌으므로 PO에서는 dual-GPU/gradient 구성의 영향이 실제로 컸다는 근거가 생겼다. 로컬에서 이후 수정한 fast-tokenizer 버전은 별도 노트북이며 아직 실행하지 않았다.

## 2. 논문 기준값

출처는 `FIUBench.pdf`의 Table 2와 Table 7이다. 아래 Rouge는 논문 표의 모델 효용 Rouge 열이며, 현재 `fast_eval`의 Retain ROUGE-L과 완전히 같은 집계라고 증명된 것은 아니다. 방향과 근접도를 보는 기준으로만 사용한다.

### LLaVA-Phi-mini-3B, Forget 5% — Table 2

| 방법 | Retain/Utility Rouge ↑ | GPT ↑ | Truth ↑ | ACC ↑ | Utility Avg ↑ | KS ↓ | Forget EM ↓ | MIA ↓ | APE ↓ | Forget Quality Avg ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Retain Model | 93.7 | 83.3 | 77.3 | 100.0 | 88.6 | 0.00 | 13.3 | 12.3 | 14.7 | 93.3 |
| GA | 50.6 | 10.0 | 48.4 | 60.8 | 42.5 | -0.03 | 14.0 | 13.0 | 15.4 | 92.9 |
| GD | 70.2 | 37.1 | 59.6 | 79.5 | 61.6 | -5.80 | 18.9 | 12.6 | 19.0 | 80.7 |
| KL | 90.3 | 56.7 | 72.6 | 93.3 | 78.2 | -21.7 | 41.2 | 42.1 | 46.8 | 36.8 |
| PO | 72.8 | 33.6 | 60.7 | 90.6 | 64.5 | -4.70 | 6.90 | 64.2 | 6.80 | 76.0 |

PO는 EM과 APE가 낮지만 MIA가 높다. 논문도 이를 단순한 지식 제거보다 거절 응답 성향으로 해석한다. 따라서 EM 하나만 낮다고 전체 forget quality가 좋아졌다고 말하면 안 된다.

### Stage-II 학습 설정 — Table 7

- 하드웨어: A100 80GB, Python 3.10, Ubuntu 22.04, x86-64.
- 공통: cutoff 512, AdamW, batch size 8, accumulation 16, dropout 0.05, 8 epochs, LoRA rank 128 / alpha 256.
- 학습률: GA/GD `2e-5`, KL `1e-4`, PO `3e-4`.
- 표에는 GPU 개수가 명시돼 있지 않다. 공개 Accelerate 설정과 현재 조사 기록은 1 process를 가리키지만, 논문 표 자체만으로 GPU 수를 확정하지 않는다.

## 3. 학습 모델 및 실행 전체 목록

상태 용어:

- **실행 실패**: 체크포인트까지 가지 못함.
- **학습 성공·성능 불일치**: 체크포인트는 정상 저장됐지만 논문 결과와 크게 다름.
- **부분 재현**: 확인한 핵심 지표가 논문과 가깝지만 전체 논문 평가가 미완료.
- **평가 대기**: 학습은 끝났지만 아직 결과 없음.

### PO 계열

| ID | 실행/모델 | 주요 설정 | 학습 상태 | 평가 상태 | 판정 |
|---|---|---|---|---|---|
| PO-0 | 이전 성공 노트북의 최초 batch 4 시험 | 단일 T4, batch 4, accum 32 | **실행 실패**. 메모리 문제 뒤 중단 | 없음 | 최종 성공 모델에 이어 학습한 것이 아님. 이후 공개 Stage-I를 새로 로드함 |
| PO-1 | 이전 성공 PO, commit `578d73f` | 단일 T4, no DeepSpeed, fast tokenizer, TF 4.46.3 / tok 0.20.3 / PEFT 0.13.2 / Accelerate 0.34.2, FP16+SDPA, batch 2, accum 64, 8 epochs, 32 updates, LR 3e-4 | **성공**, 약 2시간 16분. trainable FP16 파라미터를 FP32로 승격 | inference, fast_eval Core 및 APE 1,200 완료 | **가장 강한 부분 재현** |
| PO-2 | 후속 PO | dual T4, DeepSpeed ZeRO-2 CPU offload, slow tokenizer, TF 4.37.2 / tok 0.15.1 / PEFT 0.8.2 / Accelerate 1.14, batch 4/GPU, accum 16, 8 epochs, LR 3e-4 | **성공** | 전용 slow 평가와 fast_eval 모두 완료 | 논문과 다른 작동점. 실행 실패는 아님 |
| PO-3 | 현재 Kaggle `PO_singleGPU` | 단일 T4, DeepSpeed CPU offload, **업로드된 실행본은 slow tokenizer**, batch 4, accum 32, effective batch 128, 8 epochs | **성공**, 화면 기준 약 2시간 24분. `.../llava_phi_PO_forget5_single_t4_b4a32/checkpoint.pt` 저장 | fast_eval 완료 | 논문/PO-1에 다시 가까워짐. PO의 분산·microbatch 경로 영향 확인 |
| PO-4 | 로컬 수정본 `PO_forget5_single_t4_batch4_accum32_kaggle.ipynb` | 단일 T4, fast tokenizer, TF 4.46.3 / tok 0.20.3 / PEFT 0.13.2, Accelerate 1.14 + DeepSpeed 0.19.7, batch 4, accum 32 | **미실행** | 없음 | PO-3 실행 뒤 tokenizer/라이브러리 조건을 고친 다음 실험 후보 |

PO-1의 실행 provenance:

- Stage-I model revision: `61e7b25c38b73d5ba759f966b9ebef388d368058`
- dataset revision: `d9314f80f7214fbb41da238dbd45d61564b35e31`
- forget 400 QA, retain 6,720 QA
- 학습 가능 파라미터: 251,664,384
- 저장된 step: 8, 16, 24, 32 및 final

### GA 계열

| ID | 실행/모델 | 주요 설정 | 학습 상태 | 평가 상태 | 판정 |
|---|---|---|---|---|---|
| GA-0 | 초기 slow-tokenizer GA | dual T4/DeepSpeed 계열, legacy tokenizer 환경 | **성공** | 당시 Forget EM 약 42.44% | **성능 불일치**. 정확한 manifest/전체 수치는 로컬에 없음 |
| GA-1 | 최초 fast-tokenizer GA | dual T4/DeepSpeed, fast tokenizer | **성공** | 당시 Forget EM 약 44.77%, Retain ROUGE-L 약 92.70% | **성능 불일치**. LR scheduler가 과도하게 진행된 패턴 발견 |
| GA-2 | scheduler-fixed fast GA | dual T4, DeepSpeed ZeRO-2, fast tokenizer, TF 4.46.3 / tok 0.20.3 / PEFT 0.13.2 / Accelerate 1.14, batch 4/GPU, accum 16, 8 epochs, LR 2e-5 | **성공**. 32 attempts, 30 successful, 2 FP16 skips; scheduler index 30 | fast_eval 완료 | **성능 불일치**. scheduler 수정으로 해결되지 않음 |
| GA-3 | single-GPU no-DeepSpeed GA | 단일 T4, no DeepSpeed, fast tokenizer, TF 4.46.3 / tok 0.20.3 / PEFT 0.13.2 / Accelerate 0.34.2, batch 2, accum 64, 8 epochs, LR 2e-5 | **성공**, 약 1시간 18분. 32/32 successful, scheduler 32 | fast_eval 완료 | **성능 불일치**. GA-2와 거의 같은 결과 |

GA-0과 GA-1은 과거 대화 기록에 남은 반올림 수치다. 정확한 체크포인트 hash와 전체 Core JSON이 로컬에 없으므로 GA-2/GA-3 수준의 확정값으로 취급하지 않는다.

## 4. 평가 결과 원장

### 4.1 동일한 현재 `fast_eval` Core 경로

모든 수치는 백분율이다. `Truth raw`는 현재 코드가 출력한 raw truth ratio이며 논문 Table 2의 변환된 Truth 점수와 직접 비교하지 않는다. `MinK`와 `MinK++`도 논문 MIA 열과 집계 동등성이 완전히 검증되지 않았다.

| 학습 ID | Forget EM ↓ | Forget ROUGE-L ↓ | Forget Truth raw | MinK ↓ | MinK++ ↓ | Retain ROUGE-L ↑ | Retain Truth raw | 논문과의 요약 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| PO-1 이전 성공 PO | **6.5000** | 38.0618 | 51.2647 | 27.2685 | 67.0275 | **73.1176** | 50.0144 | 논문 PO EM 6.90 및 Rouge 72.8과 가장 가까움 |
| PO-2 후속 PO | **1.8750** | 26.6207 | 47.4143 | 7.3163 | 15.7395 | **83.2679** | 42.7840 | EM/retain은 더 좋아 보이나 논문 PO의 높은 MIA 패턴과 다름 |
| GA-2 scheduler-fixed | **42.2708** | 85.3661 | 65.4133 | 49.9683 | 60.0707 | **92.7040** | 68.6798 | 논문 GA EM 14.0 / Rouge 50.6과 큰 차이 |
| GA-3 single/no-DS | **41.3125** | 83.4297 | 67.2413 | 47.1766 | 54.0035 | **92.6106** | 70.5144 | GA-2와 거의 같아 분산/DS가 주원인이라는 설명을 약화 |
| PO-3 single slow | **7.6250** | 33.9228 | 52.4408 | 6.5954 | 19.4257 | **76.4666** | 49.4679 | 논문 PO EM 6.90에 +0.725%p, Rouge 72.8보다 +3.67%p |

참고로 fast_eval의 나머지 값:

- PO-1: Forget ROUGE-1 38.7350%, Retain ROUGE-1 73.5326%.
- PO-2: Forget ROUGE-1 27.6109%, Retain ROUGE-1 83.5429%.
- PO-3: Forget ROUGE-1 34.6633%, Retain ROUGE-1 76.8081%.
- GA-2: Forget ROUGE-1 85.4479%, Retain ROUGE-1 92.9227%.
- GA-3: Forget ROUGE-1 83.4922%, Retain ROUGE-1 92.8293%.
- 각 실행은 Forget 400, Retain 400 생성 표본을 보고했다.

PO-1의 동일 fast 평가 APE:

- 표본 수: 1,200
- exact-match 평균: 0.0295833
- APE: **2.9583%**
- 논문 PO APE 6.80% 대비: **-3.8417%p**
- APE는 낮을수록 패러프레이즈 질문에 원래 답을 덜 재현한다. 방향상 망각은 강하지만, 논문과 같은 숫자는 아니며 PO의 거절 응답 성향이 더 강해졌을 가능성도 남는다.

### 4.2 다른 평가 경로에서 얻은 PO 수치

이 표는 평가 구현 차이를 보여주기 위한 것이다. 위 fast_eval 표와 그대로 합쳐 평균을 내지 않는다.

| 체크포인트 | 평가 경로 | Forget EM | Forget ROUGE-L | Retain ROUGE-L | 비고 |
|---|---|---:|---:|---:|---|
| PO-1 | 당시 `inference.py` 생성 결과 재집계 | — | 38.72% | 73.02% | RougeScorer recall로 재집계. 생성 길이/경계가 fast_eval과 다름 |
| PO-1 | 현재 fast_eval | 6.50% | 38.0618% | 73.1176% | 현재 가장 신뢰하는 동일 경로 비교값 |
| PO-2 | 전용 slow 평가 결과 | 2.1875% | 37.1592% | 85.4584% | labels 경계 기반 prompt, assistant tag가 예측·정답 문자열에 포함됨 |
| PO-2 | 현재 fast_eval | 1.8750% | 26.6207% | 83.2679% | 같은 checkpoint도 평가 경로에 따라 값이 달라짐 |

PO-2 전용 평가의 추가 결과:

- Forget probability 69.0438%, raw truth ratio 33.0162%, APE 0.770833%, MinK 25.3282%, MinK++ 50.6791%.
- Retain probability 78.7176%, raw truth ratio 64.2633%.
- `model_utility_without_gpt` 75.0634%.
- 미완료: KS retain-only 비교, GPT-Eval, MME, POPE.

## 5. GA 진단 실험

### Checkpoint 적용 여부

20개 표본에서 base와 GA final을 비교한 결과:

- mean loss: 0.06268 → 0.22800
- mean answer probability: 0.94174 → 0.84513
- exact match: 0.55 → 0.40
- prediction 변경: 9/20
- merge 전후 loss 동등성 확인

따라서 “평가 때 checkpoint가 적용되지 않았다”는 설명은 배제됐다. 체크포인트는 실제로 forget 방향의 효과를 냈지만 논문 수준만큼 강하지 않았다.

### Step trajectory, 20 forget + 20 retain 표본

| checkpoint | F loss | F prob | F EM | F changed | R loss | R prob | R EM | R changed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| step 4 | 0.06321 | 0.94127 | 0.500 | 1 | 0.06740 | 0.93737 | 0.650 | 1 |
| step 8 | 0.07576 | 0.93055 | 0.500 | 3 | 0.06904 | 0.93562 | 0.600 | 3 |
| step 12 | 0.09563 | 0.91393 | 0.450 | 4 | 0.07150 | 0.93336 | 0.600 | 3 |
| step 16 | 0.11951 | 0.89752 | 0.450 | 5 | 0.07236 | 0.93270 | 0.600 | 3 |
| step 20 | 0.14927 | 0.88118 | 0.450 | 8 | 0.07237 | 0.93291 | 0.600 | 3 |
| step 24 | 0.18603 | 0.86371 | 0.450 | 8 | 0.07208 | 0.93337 | 0.600 | 4 |
| step 28 | 0.21643 | 0.85011 | 0.450 | 8 | 0.07228 | 0.93332 | 0.600 | 5 |
| step 32/final | 0.22825 | 0.84503 | 0.450 | 8 | 0.07234 | 0.93333 | 0.600 | 5 |

final checkpoint는 step 32와 중복이었다. 이 작은 진단에서는 더 이른 step이 final보다 낮은 Forget EM을 주면서 Retain을 보존하는 구간이 없었다. 따라서 early stopping을 추가하면 자동으로 논문 수치가 나온다는 근거는 없다. 저자 답변은 checkpoint 선택이 중요하다고 하지만 실제 선택 step과 완전한 실행 manifest는 공개 자료에서 확인되지 않았다.

## 6. 실패·문제·제약 기록

### 실제 학습 실패

- PO-0 batch 4 / accumulation 32 단일 T4 최초 시험: 메모리 문제로 실패. 이후 PO-1은 새 프로세스에서 Stage-I를 다시 로드해 batch 2 / accumulation 64로 성공했다.

### 학습은 끝났지만 논문 성능을 재현하지 못한 모델

- GA-0, GA-1, GA-2, GA-3 전부. 핵심 증상은 Forget EM 약 41~45%와 Retain ROUGE-L 약 92~93%로, 논문 GA보다 훨씬 덜 잊고 훨씬 많이 유지한다는 것이다.
- GA-2에서 scheduler/skip을 감사했고 GA-3에서 DeepSpeed와 분산을 제거했지만 결과가 거의 유지됐다.

### 환경·실행 설정 문제로 모델이 만들어지지 않은 경우

- legacy `tokenizers==0.15.1` 환경이 공개 모델의 `tokenizer.json`을 읽지 못한 문제. 당시 `use_fast=False`가 우회책이었고, 이후 TF 4.46.3 / tokenizers 0.20.3 조합에서 fast tokenizer를 사용했다.
- T4는 BF16 학습을 지원하지 않아 논문의 A100/BF16 가능 환경과 같은 정밀도 조건을 만들 수 없었다. 현재 실험은 FP16이다. 이는 제약이며 단독 원인으로 입증된 것은 아니다.
- 세션 종료로 `/kaggle/working` 중간 파일이 사라진 적이 있다. 저장된 Kaggle Version Output의 checkpoint는 복구했지만 일부 과거 raw log/JSON은 남지 않았다.

### 평가상 주의점

- `inference.py`, 전용 slow 평가, `fast_eval`은 생성 prompt 경계, tokenizer, 최대 생성 길이, 문자열 후처리가 다를 수 있다.
- 특히 PO-2 전용 평가에서는 prediction과 ground truth에 `<|assistant|>` 문자열이 포함된 것이 확인됐다.
- `fast_eval`의 고정 결과 폴더와 `overwrite=False`는 과거 JSON을 재사용할 위험이 있었다. 새 평가에서는 checkpoint 출처/hash, 새 output 경로, task log를 함께 확인해야 한다.
- raw Truth Ratio 평균을 논문 Truth 점수와 직접 비교하지 않는다.
- raw MinK/MinK++ 평균을 논문 MIA와 완전히 동일하다고 단정하지 않는다.

## 7. 현재 진행 상태와 다음 판정 순서

1. **PO-3는 부분 재현으로 분류**: EM은 논문보다 0.725%p 높고 Retain ROUGE-L은 3.67%p 높다. PO-1만큼 정확히 일치하지는 않지만 PO-2보다 논문 작동점에 훨씬 가깝다.
2. **PO-4는 아직 실행하지 않음**: PO-3가 slow tokenizer로도 핵심 EM/retain에 가까워졌으므로, PO-4의 목적은 “성공 여부 확인”보다 tokenizer/라이브러리를 PO-1에 맞췄을 때 MinK 계열까지 복구되는지 확인하는 것으로 좁혀졌다.
3. PO-2와 PO-3는 effective batch 128이 같지만 world size, rank별 batch 묶음, accumulation 횟수와 난수 소비가 다르다. 이번 결과는 이 묶음의 영향이 크다는 증거이며 GPU 수 하나만의 인과효과로 단정하지 않는다.
4. GA는 장시간 재학습보다 먼저 batch 2 microbatch 평균과 논문 batch 8 평균의 gradient 차이, assistant-role token의 loss 기여, 저장 전후 delta를 짧게 측정하는 편이 낫다.
5. 논문 전체 재현 판정을 하려면 KS, GPT-Eval, APE, MIA 정의/집계, MME, POPE까지 같은 평가 설정으로 완료해야 한다.

## 8. 한 줄 상태표

| 항목 | 현재 상태 |
|---|---|
| 가장 논문에 가까운 결과 | PO-1 이전 단일 T4 fast-tokenizer PO |
| 학습까지 완료한 PO | PO-1, PO-2, PO-3 |
| 실행 실패 PO | PO-0 batch 4 최초 시험 |
| 미실행 PO | PO-4 로컬 fast 단일 T4 버전 |
| 학습까지 완료한 GA | GA-0, GA-1, GA-2, GA-3 |
| 논문 성능에 가까운 GA | 아직 없음 |
| 새로 평가 완료 | PO-3 단일 T4 slow-tokenizer PO: EM 7.625%, Retain ROUGE-L 76.47% |
| scheduler가 주원인인가 | GA-2 결과상 아님 |
| DeepSpeed/분산이 GA 주원인인가 | GA-2와 GA-3가 거의 같아 가능성 낮음 |
| early stopping이 해결책인가 | 현재 trajectory로는 근거 없음 |
| 전체 논문 재현 완료인가 | 아님. PO 핵심 지표의 부분 재현만 확인 |

