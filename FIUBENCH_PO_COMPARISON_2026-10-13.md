# 이전 PO와 후속 PO 비교 — 2026-10-13

## 추가 증거에 따른 갱신: 성공 노트북 확인

사용자가 후속 PO도 fast_eval에서 평가했고 기존 결과와 별 차이가 없었다고 확인했다. 따라서 아래 초기 보고서의 ‘후속 PO fast 재평가 필요’는 더 이상 다음 단계가 아니다. 평가 경계 차이는 실재하지만, 사용자의 통제 결과에 비추어 PO 간 격차의 주원인으로 우선 제시하지 않는다. 후속 fast 평가의 정확한 수치/manifest는 아직 로컬에 없다.

이번에 직접 읽은 성공 노트북은 `/Users/jeongjimin/Downloads/fiubench-paper-reproduction-po-forget5-smoke.ipynb`다. 셀 번호는 파일 순서 기준 1부터 센다. 실행 카운트가 여러 세션에서 초기화되므로 파일 순서와 다를 수 있다.

### 새로 확정한 성공 실행

- 셀 15: 본 학습 직전 저장소 커밋은 `578d73f` (Fit paper reproduction training on T4).
- 셀 18: `forget_kaggle_paper`로 실제 본 학습 실행. DistributedType.NO, 1 process, FP16, batch 2, accumulation 64, 8 epochs, 32 진행 카운트, 2시간 16분 20초. 완료 카운트만으로 overflow skip이 0이었다고 단정하지 않는다.
- 셀 4: 실행된 라이브러리는 Torch 2.11.0+cu128, Transformers 4.46.3, Accelerate 0.34.2, PEFT 0.13.2, hub 0.36.2. 설치 출력에는 tokenizers 0.20.3도 나타난다. 경고 경로는 Python 3.13이다.
- 셀 18: 실제 Stage-I 모델/토크나이저 cache revision은 `61e7b25c38b73d5ba759f966b9ebef388d368058`. 학습 가능 파라미터 251,664,384. forget 400 QA, retain 6,720 QA.
- 셀 5: 실제 이미지 archive의 dataset revision은 `d9314f80f7214fbb41da238dbd45d61564b35e31`.
- 셀 18: 마지막 표시 loss 0.2988, LR 8.22e-7. 이는 해당 microbatch의 값이며 epoch 평균이나 update 이후 LR로 읽으면 안 된다.
- 셀 19: step_8/16/24/32 및 최종 checkpoint 저장 확인.
- 셀 47: inference 결과를 ROUGE-L recall로 재집계한 값은 Forget 0.3872, Retain 0.7302. 이후 사용자의 fast 평가 화면(Forget ROUGE-L 0.380618, Retain 0.731176)과 생성 경로가 다르므로 동일 값으로 쓰지 않는다.

노트북 앞부분의 batch 4 / accumulation 32는 수정 전 설정 출력이다. batch 4 테스트가 실패한 뒤 커밋을 업데이트해 batch 2 / accumulation 64로 본 학습했다. 실패한 테스트 모델을 본 학습에 이어 쓰지 않았으며, 본 학습은 별도 프로세스로 공개 Stage-I를 새로 로드했다.

### 백업 폴더만 비교해서는 놓치는 점

사용자 지정 백업 HEAD는 원본 `8e12cdd`다. 성공 노트북이 본 학습에 사용한 `578d73f`와 다르다. `/Users/jeongjimin/Desktop/unlearning/FIUBench_smoke`에 `578d73f` Git 객체가 남아 있어 실제 당시 소스를 읽었다. 이 커밋과 해당 폴더 HEAD `d562d1b` 사이에서 `forget.py`, `data_module.py`, `config/forget_kaggle_paper.yaml`의 diff는 없다. 따라서 이전 보고서의 이 학습 소스 동일성 불확실성이 해소됐다.

### 학습 코드 비교 결론

1. 성공 커밋은 기본 AutoTokenizer 경로를 사용하고, 현재 PO 노트북은 `use_fast=False`로 강제 교체한다. 학습 입력과 labels의 실제 차이는 아직 tokenizer 쌍을 실행해 측정하지 않았다. 이 변경을 단일 원인으로 확정하지 않는다.
2. 성공 커밋은 LoRA/projector를 학습 대상으로 설정한 뒤 모든 학습 가능 FP16 파라미터를 FP32로 올린다. 현재 PO에는 같은 수동 승격이 없고 DeepSpeed가 optimizer/precision을 관리한다. 이 차이를 ‘새 코드는 optimizer master weights가 없다’ 또는 ‘projector가 학습되지 않는다’로 해석하지 않는다.
3. gradient checkpointing은 양쪽 모두 켠다. 이전은 PEFT 적용 후, 현재 런타임 패치는 PEFT 적용 전에 호출한다. 호출 위치 차이는 확인됐지만 그 자체가 오동작이라는 증거는 없다.
4. PO/idk forward/loss, optimizer grouping, 데이터 전처리와 seed 233은 같다. 두 경로 모두 같은 IDK 목록에서 임의의 거절 답변과 retain 문항을 뽑는다. batch/장치 수가 달라지면 같은 seed만으로 동일한 샘플 짝·gradient 평균을 보장할 수 없다.
5. 성공 코드의 max_train_steps override/바깥 루프 종료 패치는 초기 짧은 테스트를 위한 것이며, 본 학습은 max_train_steps=0으로 8 epochs를 수행했다. 이를 early stopping의 차이라고 볼 수 없다.
6. attention 설정 키 이름은 다르지만 선택 값은 양쪽 SDPA다. DeepSpeed optional import, 사용하지 않는 CustomTrainer import, 최종 저장/프로세스 종료 패치도 코드 diff에 있으나 곧바로 학습 품질의 원인으로 보지 않는다.
7. 현재 PO 노트북의 저장된 출력은 비어 있다. 후속 PO의 실제 lr/skip/optimizer 기록은 이 파일에서 복원할 수 없다. GA에서 수행한 scheduler 감사로 과거 PO 실행까지 정상이라고 소급 판정하지 않는다.

확인된 증거는 `results/po_success_notebook_audit_2026-10-13/selected_execution_evidence.json`에 셀 코드와 출력 그대로 보존했다. 새 학습/평가나 기존 노트북 수정은 하지 않았다.

**현재 판정:** 이전 성공 실행을 재구성할 근거는 확보됐다. 후속 PO와 다른 것은 tokenizer/Transformers/PEFT, trainable precision 및 optimizer 경로, batch 구성이다. 데이터·PO loss·epochs·early stopping 차이라는 설명은 지지되지 않는다. 다음 원인 분리는 동일 문항의 실제 token IDs/labels 및 동일 초기 가중치의 첫 update를 비교하는 것이다. 단순히 다시 평가하거나 학습량을 늘리는 것을 우선 제안하지 않는다.

---

아래는 최초 조사 기록이다. 위 갱신 내용이 다음 단계 및 증거 수준 판단에 우선한다.

## 비교 대상과 출처

- 사용자 지정 백업: `/Users/jeongjimin/Desktop/FIUBench-backup-20261003`.
- 이전 Kaggle 호환 코드도 발견: `/Users/jeongjimin/Desktop/unlearning/FIUBench_smoke`. 이 폴더에 `config/forget_kaggle_paper.yaml`, `kaggle/requirements.txt`, FP32 trainable 승격 패치가 있다. 폴더 이름의 smoke와 달리 본 실험용 config도 포함한다. 이 사본 자체가 실행 당시 코드와 완전히 같다는 실행 manifest는 없다.
- 이전 본 실험의 실제 저장 설정: `/Users/jeongjimin/Downloads/po-forget5-paper/cfg.yaml`. batch 2, accumulation 64, 8 epochs, idk, LR 3e-4, LoRA 128/256/0.05, SDPA, projector 학습.
- 후속 PO: 현재 `notebooks/PO_forget5_kaggle.ipynb`, `notebooks/PO_forget5_eval_kaggle.ipynb`, `results/po_forget5_eval_2026-10-10/`.
- 사용자가 이후 제시한 fast 평가 화면은 **이전 PO checkpoint 재평가**다. 후속 PO checkpoint를 fast 평가한 결과는 현재 확인한 자료에 없다.

## 같다고 확인한 것

세 폴더의 아래 파일은 SHA256이 각각 완전히 같다.

| 파일 | SHA256 |
|---|---|
| data_module.py | 885b4a21a552dc61907fbbbb40c37a27b950e71d44f5682fa1e90973d2f84550 |
| dataset/full.json | f2f04cf7e21a3170b4c1541b5e650c9db3f92ff6f7833ea3f9e19887787bff72 |
| dataset/split.json | f576e63909babf5105f004b2230154a4d46e6b37160d7554c3434b6693a7ce4c |
| dataset/prompt.json | 7dca3cdaa00c5d75d992db5d53b02d1822365fc0cd7ecc761d05378360088ac6 |
| config/model_config.yaml | 0cd6043a293799755f4698197fae175df245a9606796b0f63eebb1135d6f8951 |

이미지 파일의 일치는 이번에 검증하지 않았다.

PO/idk 분기도 같다. IDK로 바꾼 forget 답변과 retain 정답을 batch 차원으로 합쳐 한 번의 forward에서 CE loss를 최소화한다. LR 3e-4, epochs 8, LoRA 설정도 같다. 비교한 두 학습 코드에는 early stopping이 없다. 로컬 데이터가 같다는 사실은 실행 당시 원격 revision/이미지까지 같다는 증명은 아니다.

## 실제로 달라진 조건

| 항목 | 이전 PO | 후속 PO 노트북 |
|---|---|---|
| Transformers | 4.46.3 | 4.37.2 |
| Tokenizers | 0.20.3 (이전 실행 기록) | 0.15.1 |
| PEFT | 0.13.2 | 0.8.2 |
| Accelerate | 0.34.2 | 1.14.0 |
| Tokenizer | fast (이전 실행 기록, 기본 AutoTokenizer 경로) | 학습·평가 모두 use_fast=False 강제 |
| GPU / optimizer 실행 | T4 1장, DeepSpeed 없음 | T4 2장, DeepSpeed 0.19.7 ZeRO-2 CPU offload |
| batch / accumulation | 2 × 1 × 64 | 4 × 2 × 16 |
| trainable dtype 처리 | FP16 trainable을 FP32로 승격하는 패치 | 동일 수동 패치 없음; DeepSpeed 경로 |
| 생성 prompt 경계 | 원래 answer_tag 문자열 분리 | 첫 labels != -100 직전에서 자름 |

유효 forget batch 128이라는 곱은 같지만, 임의로 선택하는 retain 문항/IDK 답변, dropout 난수, token 평균 loss의 묶음, 마지막 accumulation 크기 등의 실제 학습은 동일하지 않을 수 있다. 이 가운데 무엇이 수치 차이를 얼마나 설명하는지는 미검증이다. 최근 GA 노트북의 패키지 설정과 후속 PO 노트북의 설정을 혼동하지 않아야 한다.

## 평가 차이는 원본 JSON에서도 확인됨

후속 PO `generated_text`를 전수 확인했다.

| 검사 | Forget | Retain |
|---|---:|---:|
| 문항 수 | 400 | 400 |
| prompt에 assistant 태그 포함 | 0 | 0 |
| prediction에 assistant 태그 포함 | 400 | 400 |
| ground truth에 assistant 태그 포함 | 400 | 400 |

예시 prompt는 `<|user|> ... <|end|>\n`에서 끝난다. 예측과 정답은 모두 `<|assistant|> The full name ...` 등으로 시작한다. 평가 노트북의 label 경계 패치와 일치한다.

영향은 둘이다. (1) 모델에 주는 생성 입력이 달라져 생성 답 자체가 달라질 수 있다. (2) 채점 문자열에 공통 assistant 문자열이 들어가 ROUGE 대상이 달라진다. 태그의 공통 단어 효과만으로 전체 격차가 설명된다는 뜻은 아니다. 이미 생성한 JSON에서 태그를 제거하는 것만으로 올바른 prompt의 생성 결과를 복원할 수 없다.

| 결과 | 이전 PO를 fast 재평가한 사용자 화면 | 후속 PO 저장 결과 |
|---|---:|---:|
| Forget EM | 6.5000% | 2.1875% |
| Forget ROUGE-L | 38.0618% | 37.1592% |
| Retain ROUGE-L | 73.1176% | 85.4584% |

다른 checkpoint와 다른 평가 경로를 함께 비교한 것이므로 이 표만으로 학습 원인을 확정할 수 없다.

## 파일 식별

이전 `po-forget5-paper/checkpoint.pt`: 503495189 bytes, SHA256 `40c65051bc9ae1172990ad1d33b708a5085ad1b6a9cf39ba2afc3f19353ebd37`.

Downloads의 별도 `PO-forget5_checkpoint.pt`: 503495189 bytes, SHA256 `ffaf3c13ab2d756341b104ea5272891abb05fd5acb3e32e12cc63c5322fd9f0f`.

파일 내용은 다르다. 두 번째 파일이 후속 실험의 정확한 checkpoint인지 manifest로 확인하지 못했다. 직렬화 파일 해시가 다르다는 것만으로 모든 텐서 또는 학습 이력 차이를 단정하지 않는다.

## 다음 통제 비교

재학습보다 먼저 **후속 PO checkpoint**를 기존 `FIUBench_forget5_fast_eval_kaggle.ipynb`로 평가한다. 이전 PO를 그 노트북에서 평가하는 대조는 이미 완료됐다. 입력 checkpoint의 출처/해시를 기록하고, 새 출력 디렉터리 또는 새 세션으로 기존 task JSON 재사용을 막는다.

- 두 PO가 같은 fast 평가에서도 다르면 실제 학습 또는 checkpoint 조건 차이가 남는다. 위 패키지/tokenizer/학습 경로를 통제해야 한다.
- 차이가 크게 줄면 기존 평가 경로가 주요 기여 요인이었다는 근거가 된다. 숫자가 가까워지는 것만으로 파라미터나 학습 경로의 동일성까지 증명하지는 않는다.

GA 단일 GPU 실험은 GA에서 분산 제거만으로 결과가 해결되지 않았다는 근거다. 이 결과로 두 PO의 분산/optimizer 경로 영향을 완전히 배제할 수는 없다. GA의 scheduler 감사도 과거 후속 PO 실행의 scheduler 동작을 소급해서 검증하지 않는다.

이번 작업은 파일·노트북·저장 JSON 비교이며 GPU 학습/평가를 실행하지 않았다. 기존 노트북이나 학습 코드는 변경하지 않았다.
