# GA forget5 재현 차이 진단 — 2026-10-12

## 범위와 현재 판정

사용자가 제시한 스케줄러 감사 화면: 8 epochs, 32 attempts, 30 successful, 2 skipped, scheduler index 30. 평가 화면: EM 42.2708%, retain ROUGE-L 92.7040%, retain raw truth_ratio 68.6798%, Min-K 49.9683%. 로컬에는 이번 Kaggle 실행의 원본 checkpoint, cfg.yaml, scheduler JSONL, 평가 JSON 및 core_console.log가 없다. 따라서 실행 출처 및 GPU forward/backward는 이번 조사에서 검증하지 못했다.

논문 Table 2 GA의 EM 14.0, retain ROUGE 50.6과 큰 차이가 있다. 스케줄러 제어는 화면상 검사를 통과했다. 차이의 단일 원인은 아직 확정되지 않았다. "학습 강도가 약함"은 관측 양상의 해석이며 원인 규명이 아니다.

## 1. 확정된 평가 재사용 결함 — 이번 실행 영향은 조건부

`notebooks/FIUBench_forget5_fast_eval_kaggle.ipynb`:

- cell 6: 모든 GA 실행을 `/kaggle/working/ga_forget5_fast_eval`에 저장한다.
- cell 6: 선택한 checkpoint를 그 폴더로 다시 복사한다. 입력에 최종 checkpoint 하나라는 것만 검사하며 scheduler 수정본인지 검사하지 않는다.
- cell 10: `overwrite=False`.
- `evaluate_util.py:644-647`: 기존 task JSON이 있으면 모델 재평가 없이 해당 JSON을 읽는다.

따라서 같은 작업 폴더에서 입력 checkpoint만 교체하면 새 checkpoint와 옛 평가 JSON이 함께 존재할 수 있다. task별로 별도 재사용하므로 부분 결과 혼합도 가능하다. 완전히 새 세션에서 실행했다면 이 결함이 이번 결과에 영향을 주지 않았을 수 있다. 이전 EM과 이번 EM이 다르므로 전체 결과를 그대로 재사용했다는 증거도 없다. Retain 값의 유사성만으로 혼합을 확정하지 않는다.

확인 자료: `core_console.log`의 `Skipping eval_forget_log` / `Skipping eval_retain_log`, Add Input 실제 경로, source cfg.yaml, source와 평가용 checkpoint의 SHA256. 파일 mtime만으로 provenance를 확정하지 않는다. 추후 수정한다면 source checkpoint hash별 결과 디렉터리와 manifest가 필요하다. 이번 조사에서 노트북을 수정하지 않았다.

## 2. 공개 Stage-I와 논문 Stage-I의 동일성 미확인

2026-10-12 공개 Hugging Face config.json을 HTTPS로 읽어 확인:

https://huggingface.co/gray311/vlm_unlearning_ft_llava_phi_3_mini/blob/main/config.json

- `_name_or_path`: `./models/final_ft_2_epochs_lr1e-05_llava-phi_full`
- `transformers_version`: `4.45.1`
- top-level `torch_dtype`: bfloat16, text_config의 torch_dtype는 float16.

논문 Table 7의 Stage-I는 10 epochs / LR 2e-5. 공개 `config/forget_lora.yaml`은 `final_ft_10_epochs_lr2e-05_llava-phi_full`라는 로컬 경로를 사용한다. 파일명은 실제 이력의 증명이 아니며 `transformers_version`도 모든 학습 단계의 환경 증명이 아니다. 그러나 논문과 같은 초기 가중치라는 주장 역시 입증되지 않았다.

논문 3.1은 답변 형식 유사성만으로 ROUGE가 높을 수 있음을 설명한다. 따라서 이전 Stage-I ROUGE 93.2 대 논문 93.3을 근거로 checkpoint가 동일하다고 보았던 것은 잘못이다. 동일 prompt/label/이미지로 base와 수정 GA의 forget 답변 loss, EM, Min-K, 실제 생성물을 비교해야 기억의 변화가 보인다. 저자의 당시 모델 revision/실행 config 없이는 동일성 최종 확정이 어렵다.

## 3. 업데이트 수 검증과 학습 내용 검증은 다름

- `forget.py`의 GA 분기는 `loss = -outputs.loss`로 원본과 같다. retain은 로더에서 생성하지만 GA loss에는 쓰이지 않는다. PO retain 336/380 문제를 GA의 직접 원인으로 들면 안 된다.
- Accelerate 1.14.0 공식 `backward`는 DeepSpeed일 때 별도로 loss를 accumulation으로 나누지 않는다. DeepSpeed wrapper는 accumulation boundary를 전달하고 sync 시 engine.step을 수행한다. 현재 코드에서 단순한 이중 division이나 수동 optimizer.step의 이중 업데이트를 확인하지 못했다.
- 감사 코드는 scheduler index, LR, skip state, optimizer parameter group 연결을 검사한다. 실제 sample별 gradient, LoRA delta magnitude, projector update size, checkpoint가 평가에 완전히 적용됐는지는 검사하지 않는다.
- full accumulation의 batch 128은 1 GPU×8×16과 2 GPU×4×16에서 같다. 후자는 장치별 token 평균 loss, dropout RNG, precision 등 때문에 수치적으로 완전 동일하다고 보장할 수 없다. 이것이 28%p 격차의 주원인이라는 증거는 없다.
- 각 epoch 50 microbatches에서 16+16+16+2로 끝난다. 원본 1 GPU batch8도 50 microbatches라 마지막 짧은 accumulation 자체는 새로 생긴 차이가 아니다.
- FP16 성공/skip 카운터는 optimizer의 실행을 의미한다. 첫 warmup LR=0의 성공 step 등은 큰 가중치 변화를 보장하지 않는다. 2회 skip만으로 격차를 설명하거나 배제하는 통제 실험은 아직 없다.
- `preprocess_v1`는 첫 supervised token에 assistant tag를 포함하며 현재 노트북은 이를 검증한다. 현재 평가 프롬프트는 assistant tag를 입력으로 제공한다. 따라서 answer-only CE와 전체 supervised CE를 분리해 확인할 가치가 있다. 원본과 공유하는 동작이므로 새 버그/주원인으로 확정하지 않는다.

다음 GPU 진단은 새 학습 대신 동일한 소수 샘플에서 base → checkpoint 적용 후 → LoRA merge 후의 answer loss/logits와 생성물을 비교한다. checkpoint trainable keys 전체 로드 여부와 실제 텐서 값도 검사한다. 차이가 정상적으로 확인되면 그때 한 업데이트의 gradient 및 delta 검사를 진행한다.

## 4. 이전 MIA 해석 정정 및 데이터 불일치

논문 2.5.3은 MIA를 Min-K% Prob로 정의한다. 이 논문 표의 MIA를 AUROC라고 설명할 근거는 찾지 못했다. 공개 `aggregate_eval_stat.py:20-21`도 mink를 단순 평균한다. 이전 "raw Min-K는 논문의 AUROC와 다른 지표"라는 설명은 철회한다.

공개 `evaluate_util.py:434-465`는 정답 label token들에서 하위 10/20/30/40/50%의 로그 확률 평균을 exp한 후 0.3/0.3/0.2/0.1/0.1로 합친다. 논문 서술의 generated answer 및 log-prob 식과 구현상 차이가 있으므로 정확한 reporting 변환은 확인이 필요하다. 그렇다고 별도의 미구현 AUROC 평가라고 단정하면 안 된다.

Raw Truth Ratio 평균은 여전히 논문의 Truth utility로 바로 사용할 수 없다. `aggregate_eval_stat.py:73-85`처럼 개별 sample loss에서 변환한 뒤 평균해야 한다. `1 - mean(raw)`도 clipping 때문에 일반적으로 같지 않다.

로컬 원본 metadata와 upstream 간 diff 없음. 실제 forget5 400 QA:

- keywords 존재 399, 없음 1.
- 334문항은 keyword 항목이 하나. 평균 1.2325개/문항.
- keyword phrase를 공백으로 나눠도 평균 2.305 단어/문항.
- paraphrased_answer / perturbed_answer는 400문항 전부 존재.

논문 2.5.2의 평균 약 5 keywords/QA 설명과 차이가 있다. 평가 데이터/키워드가 논문 당시와 정확히 같다는 것을 추가 확인해야 한다. 비어 있는 1문항만으로 현재 EM 격차를 설명할 수는 없다.

## 다음 자료와 판별 순서

1. 이번 `core_console.log` + 평가 두 JSON + 실제 학습 source cfg.yaml을 확보한다. 지금 모델을 다시 학습할 필요는 없다.
2. 재사용 여부, checkpoint 출처/해시, sample 순서/수/키워드, 원본 집계식을 검증한다.
3. 같은 모델 환경에서 base와 수정 GA의 동일 샘플 forward/generation/merge equivalence를 검사한다.
4. 그 결과로 model provenance와 실제 optimizer update 중 어느 쪽을 조사할지 좁힌다. 원인 미확인 상태에서 LR/epochs를 늘려 논문 숫자만 맞추지 않는다.

참조: 로컬 FIUBench.pdf pp.6–8,18, data_module.py, evaluate_util.py, forget.py, aggregate_eval_stat.py, 두 Kaggle notebook, config/forget_lora.yaml 및 accelerate_config.yaml.
공식 라이브러리: https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py 와 https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/deepspeed.py
