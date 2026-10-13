# GA 재현 코드 감사 — 2026-10-13

## 판정

아직 단일 원인은 확정되지 않았다. 성공 PO의 실제 cfg.yaml, 저자 공개 forget.py/data_module.py, 현재 GA 노트북, Transformers 4.46.3 및 Accelerate 0.34.2 소스를 대조했다. GPU에서 gradient를 직접 측정한 감사는 아니다.

사용자 화면의 단일 GPU GA: EM 41.3125%, Retain ROUGE-L 92.6106%, 32 attempts/32 non-skipped calls. 이전 DeepSpeed GA: EM 42.2708%, Retain ROUGE-L 92.7040%, 32 attempts/30 non-skipped calls. 실행 환경 변경으로 큰 격차가 해결되지 않았지만 개별 요인의 인과관계를 모두 배제한 것은 아니다.

## 1. 새로운 외부 근거: 저자의 checkpoint 선택 설명

https://github.com/SaFo-Lab/FIUBench/issues/12#issuecomment-2781691927

2025-04-06 gray311 답변: 학습 step에 결과가 크게 의존하며, forget quality와 utility를 고려한 early stopping/checkpoint 선택을 사용했다. GA loss를 -10 이내로 유지하는 예를 들었다. LLM은 LoRA, MM projector는 전체 파라미터를 학습한다고 확인했다.

이는 현재 LoRA+projector 선택을 뒷받침한다. 그러나 저자의 실제 GA 선택 step, 학습 loss 평균 방식, 정확한 환경 manifest는 답변에 없다. -10은 반드시 도달해야 하는 목표가 아니다. Table 7의 8 epochs보다 더 학습했다는 증거도 아니다. 현재 trajectory에서는 중간 checkpoint가 최종보다 우수하다는 근거가 없으므로 단순 early stopping 추가로 해결된다고 말할 수 없다.

https://github.com/SaFo-Lab/FIUBench/issues/15

다른 사용자가 A100에서도 Table 7 조건의 Stage-I 및 공개 로그/논문 결과 불일치를 보고했다. 이는 독립된 재현 문제 보고이지 우리 GA 원인 확정 증거가 아니다.

## 2. 확인된 비동등성: microbatch별 token 평균

논문 Table 7: batch 8 / accumulation 16. 현재 단일 GPU: batch 2 / accumulation 64. 모두 full accumulation 구간에서는 128 QA지만 실제 목적함수의 token 가중치는 같지 않다.

Transformers 4.46.3 LlavaForConditionalGeneration.forward는 shift 후 CrossEntropyLoss()의 기본 mean으로 유효 label token 전체를 평균한다. Accelerate 0.34.2 backward는 이를 accumulation steps로 한 번 나눈다.

- 현재: 각 2문항의 token-loss sum / 그 2문항의 유효 token 수 / 64.
- batch 8 기준: 각 8문항의 token-loss sum / 그 8문항의 유효 token 수 / 16.

8문항 묶음 안의 pair 유효 token 수를 N2, 전체를 N8이라 하면 현재의 token별 가중치 / 기준 가중치는 N8/(4*N2)다. 답변 길이가 달라지면 1이 아니다. 이는 gradient 방향까지 달라질 수 있는 계수 차이이며 단순 전체 loss 상수배가 아니다.

로컬 400문항과 기록된 Stage-I revision 61e7b25c38b73d5ba759f966b9ebef388d368058의 tokenizer.json을 tokenizers 0.20.3으로 계산:

- supervised token 수: 최소 3, 최대 46, 평균 17.8075.
- batch2 / batch8 token 가중치 비율: 0.602941 ~ 4.857143.
- 400문항 중 128개에서 가중치 차이 20% 초과.
- 두 rank가 연속된 batch4를 받는 조건의 batch4x2 / batch8: 0.747253 ~ 1.511111; 40문항에서 20% 초과.
- input/label SHA256: 80f3aba0d195a0083d4bb3217e6566468c45ba6459812866433ed878ff18a50d. Kaggle의 training_token_signature_512.txt와 대조해야 실제 실행 입력과 동일함을 확인할 수 있다.

이 수치는 gradient norm 또는 파라미터 변화가 4.86배라는 뜻이 아니다. dropout, AdamW, clipping, padding, FP16 차이는 포함하지 않았다. native tokenizer.json의 BOS template와 저장된 데이터 순서를 사용했다.

T4에서 batch8의 token 평균 계수만 재현하는 방법: batch2 loss에 4*N2/N8을 곱한 뒤 기존 accumulation64를 사용한다. 400문항 전체에서 이 계수가 기준 1/(16*N8)과 일치하는 산술 검사를 통과했다. dropout과 수치 연산까지 실제 batch8과 같다고 보장하지 않는다. 이를 바로 8 epochs 재학습하기 전에 동일 128 QA에서 두 방식의 gradient cosine/norm을 측정하는 것이 우선이다.

## 3. GA 특유의 확인 대상: assistant token도 상승 목적함수에 포함

data_module.py:19-28의 preprocess_v1을 그대로 적용하면 400문항 모두 첫 supervised token이 <|assistant|>다. evaluate_util.py:695 이후 생성 prompt는 이 token을 입력으로 제공한다.

따라서 GA는 정답 내용뿐 아니라 역할 token의 예측확률도 낮춘다. PO는 같은 역할 token의 확률을 높이는 방향이다. 이론적으로 GA loss 증가가 답변 내용 망각과 분리될 수 있다. 그러나 이는 저자 코드와 공유하는 동작이며 아직 실제 token별 loss/gradient 기여도를 측정하지 않았다. 이번 실패의 확정 원인이라고 부르면 안 된다.

다음 측정에서는 동일 checkpoint/동일 QA에서 assistant-token CE와 나머지 supervised-token CE를 분리한다. 모델 내부 이미지 token 확장 이후 label과 logits 정렬을 사용해야 한다.

## 4. 발견하지 못한 오류 및 감사의 한계

- GA 부호는 저자와 동일한 -outputs.loss. retain forward/loss가 GA에 더해지지 않는다.
- AdamW 파라미터 그룹, bias 제외 weight decay 0.01, LoRA target regex가 저자와 같다.
- 452개 학습 텐서는 LoRA 448 + projector 4. 개수만으로 전부 gradient가 존재함을 증명하지는 못한다.
- Accelerate 0.34.2 optimizer.step/zero_grad는 sync 경계 밖에서 실제 연산을 하지 않는다. 매 microbatch의 zero_grad 호출이 누적 gradient를 지운다는 오류는 발견되지 않았다.
- 비 DeepSpeed backward의 accumulation division은 한 번이다. 코드에서 명백한 이중 division은 발견되지 않았다.
- 마지막 짧은 누적은 단일 GPU에서 8/64, 논문 배치에서는 2/16로 모두 1/8이다. 이 차수 자체는 새 오류가 아니다.
- scheduler의 첫 LR=0 호출도 non-skipped로 계산된다. 32 successful은 32회의 큰 가중치 변화를 뜻하지 않는다.
- save_lora_weights는 trainable tensors를 FP16으로 저장한다. 작은 projector delta의 반올림 손실은 저장 전/후 비교 대상이나, 발생 규모는 아직 측정하지 않았다.
- fast_eval은 고정 결과 경로 + overwrite=False여서 기존 JSON을 재사용할 수 있다. 이번 화면은 이전과 수치가 달라 전체 재사용 증거는 없지만, checkpoint SHA와 task log로 출처를 확인해야 한다.
- 새 GA notebook은 모델 revision을 고정하지 않았다. 이미지 dataset SHA 기록만으로 model provenance를 확보했다고 말할 수 없다.

## 5. PO와의 비교에서 확인된 것

실제 파일 /Users/jeongjimin/Downloads/po-forget5-paper/cfg.yaml: PO/idk, LR 3e-4, batch2, accumulation64, 8 epochs, LoRA128/256/.05, SDPA, gradient checkpointing. log.txt는 0 byte라 실제 매 step 동작을 복구할 수 없다.

이전 채팅은 trainable FP16->FP32 변환, 단일 GPU, Accelerate0.34.2를 기록한다. 현재 이전 저장소 URL에서 받은 forget.py는 upstream과 SDPA 선택 부분만 다르며, 성공 당시의 runtime patch/commit 전체를 보존한 원문은 아니다.

PO LR는 GA보다 15배 높고, refusal+retain을 합쳐 forward한다. 이는 논문 설정에 따른 차이이지 LR 오설정 증거가 아니다. 그래서 batch2 PO가 약 두 시간, GA가 약 한 시간인 것도 처리하는 두 batch 대 한 batch로 설명 가능한 차이다.

PO의 EM/ROUGE 일치는 유효한 부분 재현이다. 모든 loss/gradient 처리 또는 논문 Stage-I와의 동일성을 증명하지는 않으며, 반대로 GA 실패만으로 Stage-I가 틀렸다고 단정해서도 안 된다.

## 다음 최소 검사

새 장시간 학습보다 단일 GPU GA checkpoint를 이용한 다음 진단을 우선한다.

1. source/eval checkpoint SHA, 실제 tokenizer signature, Core task 재사용 로그 확인.
2. 같은 QA에서 base/final의 assistant-token CE와 나머지 CE 분리.
3. 동일 128 QA, dropout을 끈 진단 조건에서 현재 batch2 평균과 batch8 계수 재현 방식의 LoRA/projector gradient norm, cosine, None/zero/nonfinite 개수 비교. gradient checkpointing 경로도 확인.
4. 한 번의 nonzero-LR update 전후 LoRA/projector 실제 delta, FP16 저장 후 delta 손실 측정.

이 결과가 나오기 전에는 -10을 목표로 학습하거나 LR/epoch를 늘려 논문 점수에 맞추지 않는다.

## 공식 소스

- https://github.com/SaFo-Lab/FIUBench/blob/main/forget.py
- https://github.com/SaFo-Lab/FIUBench/blob/main/data_module.py
- https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llava/modeling_llava.py
- https://github.com/huggingface/accelerate/blob/v0.34.2/src/accelerate/accelerator.py
- https://github.com/huggingface/accelerate/blob/v0.34.2/src/accelerate/optimizer.py
