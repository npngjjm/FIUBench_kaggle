# 이전 성공 실행과 현재 실행 비교

근거: `코드 구조 분석 및 설명` 채팅(ID 01a0fd40-75cf-7773-90a4-ca082fcc3fdb)의 실험 요약 및 사용자가 붙인 평가 출력, 현재 로컬 설정과 결과. 이전 로컬 경로 `/Users/jeongjimin/Desktop/FIUBench`는 현재 존재하지 않는다. 이전 코드의 온라인 원문 조회도 실패했으므로 아래 옛 설정은 대화 기록에 근거하며 원본 실행 manifest 검증은 아니다.

## 핵심 발견

- 이전에 논문과 가까웠던 실험은 GA가 아니라 PO (`idk`)였다.
- 이전 PO: Forget EM 6.50%, Retain ROUGE-L 73.02%. ROUGE-L은 별도 `inference.py` 추론에서 얻었으며 해당 생성 JSON은 세션 종료로 소실됐다는 기록이 있다. 공식 평가 `evaluate_util.py`에서는 Retain ROUGE-1 73.5326%, Forget EM 6.50%가 보고됐다. 두 평가 경로를 하나로 취급하면 안 된다.
- 이전 학습: 단일 T4, Accelerate 0.34.2, DeepSpeed 비활성, batch 2 / accumulation 64 / 8 epochs / 32 update attempts, PO LR 3e-4. 학습 가능 FP16 파라미터를 FP32로 올리는 패치가 있었다.
- 이전 환경 기록도 PyTorch 2.11.0+cu128, Transformers 4.46.3, PEFT 0.13.2, FP16, SDPA였다. 따라서 이 세 패키지 버전이나 T4/FP16만을 새 실패의 원인으로 지목할 근거는 약해졌다. Python 버전과 exact model revision은 확인되지 않았다.
- 현재 GA: T4 두 장, Accelerate 1.14.0 + DeepSpeed 0.19.7 ZeRO-2, batch 4/GPU / accumulation 16, LR 2e-5, 32 attempts / 30 successes, fast tokenizer. Forget EM 42.2708%, Retain ROUGE-L 92.7040%.
- 같은 방법에서 관찰한 차이도 있다. 현재 폴더의 `results/po_forget5_eval_2026-10-10/summary.json`은 후속 PO 실행으로 EM 2.1875%, Retain ROUGE-L 85.4584%다. 이 후속 결과에는 slow tokenizer/generation 경계 이슈가 있었으며 이전 PO와 같은 checkpoint가 아니다.

## ROUGE 계산 추가 확인

현재 `inference.py:156` 자체 출력은 `rougeL.precision`이다. 그러나 이전 채팅의 전체 추론 결과 분석 셀은 저장 JSONL을 다시 읽어 `RougeScorer(['rougeL'], use_stemmer=True)`와 `scorer.score(answer, prediction)['rougeL'].recall`을 사용했다. 따라서 이전 보고값 73.02%를 precision이라고 재분류하면 안 된다. 현재 `evaluate_util.py:739`도 같은 라이브러리/stemmer의 recall을 사용한다. 두 보고값의 기본 수식은 같지만 생성 prompt, tokenization, 길이 제한, 정답 복원과 후처리가 다를 수 있다. 기존 inference의 40-token Phi 분기와 현재 eval의 50-token 설정은 코드에서 확인되며, 과거 Kaggle 수동 패치까지 원본 코드만으로 확정할 수는 없다.

## 해석 및 다음 통제 비교

현재 감사는 성공 optimizer update마다 scheduler를 한 번 전진시킨다는 것을 검증한다. 과거 버그가 없었다거나 논문 실행과 완전히 같다는 뜻은 아니다. 20표본 체크포인트 진단은 GA 가중치가 적용됐고 loss/generation을 바꾼다는 것을 보여준다. 논문 재현 여부를 증명하지는 않는다.

1. 먼저 이전 성공 PO checkpoint (`outputs/llava_phi_forget5_po_paper/checkpoint.pt`, `po-forget5-paper.zip`, 기록 크기 503495189 bytes)를 확보해 동일 checkpoint의 옛/현재 평가 경로를 비교한다. 크기는 식별의 보조 정보이며 SHA256 대체가 아니다.
2. 평가 차이가 작으면 PO를 고정한 채 옛 single-GPU 경로와 새 DeepSpeed 경로를 비교한다. tokenizer IDs/labels와 초기 model revision을 고정하고 loss 정규화, gradient norm, trainable dtype, projector/LoRA delta를 확인한다.
3. 이 비교로 검증된 학습 경로에서 GA를 실행하면 방법 변경과 런타임 변경을 분리할 수 있다.

PO와 GA의 목적함수/LR은 의도적으로 다르므로 두 결과의 차이만으로 DeepSpeed나 배치를 원인으로 확정하지 않는다. 또 PO의 두 지표 일치만으로 공개 Stage-I가 논문과 동일하다고 증명하지 않는다.
