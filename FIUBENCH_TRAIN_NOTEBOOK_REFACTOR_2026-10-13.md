# FIUBench 학습 노트북 중복·추상화 감사 — 2026-10-13

## 조사 범위

- `notebooks/GA_forget5_train_kaggle.ipynb`
- `notebooks/GA_forget5_fast_train_kaggle.ipynb`
- `notebooks/GA_forget5_fast_scheduler_fix_train_kaggle.ipynb`
- `notebooks/GA_forget5_single_t4_no_deepspeed_kaggle.ipynb`
- `notebooks/PO_forget5_kaggle.ipynb`
- `notebooks/PO_forget5_single_t4_batch4_accum32_kaggle.ipynb`

평가 노트북과 진단 전용 평가 노트북은 이번 범위에서 제외했다.

## 핵심 결론

현재 여섯 노트북은 실험 코드보다 환경 설치, 데이터 복원, `forget.py` 런타임 패치, YAML 생성, 실행, checkpoint 검증을 반복해서 담고 있다. 실험마다 복사한 셀을 조금씩 수정하면서 라이브러리, tokenizer, 출력 경로와 설명이 어긋날 위험이 커졌다.

가장 효과적인 정리는 노트북을 하나의 거대한 범용 노트북으로 합치는 것이 아니다. 공통 실행 코드를 저장소의 스크립트와 계층형 config로 옮기고, 노트북은 실험 명세를 고른 뒤 공통 단계를 호출하는 얇은 인터페이스로 만드는 것이다.

## 반복되는 부분

### 1. 저장소 및 라이브러리 준비

여섯 노트북 모두 다음 흐름을 반복한다.

- Git 저장소 clone
- GPU 확인
- PyTorch 외 패키지 재설치
- 핵심 버전 출력 및 assert
- 설치 후 커널 재시작

현재는 fast/slow와 DeepSpeed 여부에 따라 패키지 문자열 전체를 복사해서 수정한다. 이 때문에 tokenizer만 바꾸려 해도 Transformers, Tokenizers, PEFT, Accelerate와 DeepSpeed 조합을 여러 셀에서 함께 고쳐야 한다.

추상화 대상:

- `requirements/success-fast.txt`
- `requirements/legacy-slow-deepspeed.txt`
- `scripts/check_runtime.py`

노트북에는 선택한 환경 파일 설치와 커널 재시작만 남긴다.

### 2. FIUBench 이미지 복원

대부분의 노트북이 같은 일을 다시 구현한다.

- `full.json`의 첫 400개 레코드 읽기
- Hugging Face dataset revision 조회
- `SFHQ.zip` 다운로드
- ZIP 안의 파일명을 JSON 경로와 대응
- 이미지 열기 검증
- dataset revision 기록

추상화 대상:

- `scripts/prepare_fiubench_images.py`

인자로 dataset revision, 레코드 수, 임시 다운로드 위치를 받게 한다. revision은 `latest`로 묵시적으로 가져오기보다 실험 명세에 명시하고 manifest에도 기록해야 한다.

### 3. `forget.py` 문자열 패치

다섯 노트북이 비슷한 방식으로 `forget.py`를 문자열 치환한다.

- `use_fast=True/False`
- gradient checkpointing 활성화
- input gradients 활성화
- 최종 LoRA checkpoint 저장

이 방식은 패치 순서, 이전 셀 실행 여부와 저장소 커밋에 민감하다. 같은 세션에서 slow 실행 뒤 fast 실행을 시도할 때 특히 위험하다.

최우선 구조 개선:

- `tokenizer_use_fast`를 Hydra config 필드로 추가
- gradient checkpointing과 최종 저장을 `forget.py`의 정식 코드 경로로 통합
- 런타임 문자열 치환 제거

그러면 노트북마다 `forget.py.before_*_patch`와 diff를 관리할 필요가 없다.

### 4. 학습 및 Accelerate YAML 생성

노트북들이 긴 YAML 문자열을 직접 쓴다. 실제 차이는 대체로 다음 필드뿐이다.

- `forget_loss`: `ga` 또는 `idk`
- LR
- fast/slow tokenizer
- batch size와 accumulation
- GPU process 수
- DeepSpeed 사용 여부와 offload
- 출력 경로와 저장 주기

권장 config 계층:

```text
config/experiment/forget5_base.yaml
config/method/ga.yaml
config/method/po.yaml
config/runtime/single_t4.yaml
config/runtime/t4x2_deepspeed.yaml
config/tokenizer/fast.yaml
config/tokenizer/slow.yaml
```

노트북은 위 config 조합과 `experiment_id`만 선택해야 한다. effective batch는 공통 preflight에서 계산하고 기대값과 다르면 학습 전에 중단한다.

### 5. 토큰 및 label 경계 검증

fast GA 노트북 두 개와 단일 GPU 노트북이 tokenizer class, token IDs, labels 경계를 각각 검증한다. 이 검증은 중요하지만 구현이 흩어져 있다.

추상화 대상:

- `scripts/audit_tokenization.py`

출력은 다음 공통 항목을 가진 JSON으로 저장한다.

- tokenizer class와 `is_fast`
- model revision
- 전체 400개 input/label hash
- assistant answer 시작 위치
- 이미지 token 수
- truncation 발생 수

### 6. 실행 및 산출물 검사

Accelerate 실행, console log 저장, checkpoint 로드, finite tensor 검사, 중간 checkpoint 나열이 반복된다.

추상화 대상:

- `scripts/launch_experiment.py`
- `scripts/validate_run.py`

검증기는 checkpoint 존재 여부만 확인하지 말고 다음도 검사해야 한다.

- 실제 process 수
- 실제 total batch size
- 총 optimizer attempts와 successful updates
- scheduler index
- tokenizer signature
- source/model/dataset revision
- config hash

## 저장 공간과 시간 비효율

현재 PO Version Output은 약 10.3 GiB다. checkpoint 이외에 다음이 `/kaggle/working`에 함께 남는다.

- Hugging Face 기본 모델 cache 약 8 GiB
- 다운로드한 `SFHQ.zip`
- 복원한 이미지
- 중간 checkpoint 여러 개
- `step_32/checkpoint.pt`와 최종 `checkpoint.pt`의 사실상 중복

권장 변경:

1. `HF_HOME`을 `/kaggle/working` 밖의 임시 cache로 설정한다.
2. 다운로드 ZIP과 임시 압축 해제 파일은 학습 완료 뒤 제거한다.
3. checkpoint 정책을 `final_only`와 `trajectory`로 분리한다.
4. 일반 재현 실행은 최종 checkpoint만 보존한다.
5. trajectory 실행은 필요한 step만 보존하고 마지막 step과 final의 중복 저장을 피한다.
6. Version Output에는 checkpoint, config, manifest, console log, environment freeze, tokenization audit만 남긴다.

이 구조라면 일반 실행 Output을 수백 MiB 수준으로 줄일 수 있고, Kaggle→Colab 또는 평가 노트북 연결 시간도 크게 줄어든다.

## 합치면 안 되는 부분

다음은 공통 코드로 억지로 숨기지 않는 편이 낫다.

- GA와 PO의 loss 계산
- DeepSpeed 실행과 순수 Accelerate 실행
- scheduler 수정·overflow 감사 로직
- 논문 재현 실행과 checkpoint trajectory 진단

특히 `GA_forget5_fast_scheduler_fix_train_kaggle.ipynb`는 생산 학습 노트북이 아니라 진단·감사 실험으로 분리 보존해야 한다. `GA_forget5_single_t4_no_deepspeed_kaggle.ipynb`도 직접 학습 루프를 포함하므로 일반 Accelerate launcher와 완전히 합치면 검증 목적이 흐려진다.

## 권장 최종 구성

노트북은 세 종류면 충분하다.

1. `train_reproduction.ipynb`
   - 검증된 fast tokenizer 환경
   - GA/PO 및 runtime config 선택
   - 공통 preflight, launch, validate 호출
2. `train_diagnostic.ipynb`
   - scheduler/overflow/checkpoint trajectory 감사
3. `evaluate.ipynb`
   - 저장 checkpoint 입력 하나만 받아 동일 evaluator 실행

각 실행은 다음처럼 유일한 ID를 사용한다.

```text
runs/po_forget5_single_t4_fast_b2_a64_20261013/
  checkpoint.pt
  experiment.yaml
  manifest.json
  tokenization_audit.json
  environment.txt
  console.log
```

manifest가 config, source revision, model revision, dataset revision, tokenizer class, GPU, precision, process 수, batch, accumulation, scheduler 결과와 checkpoint SHA256을 한 파일에 기록하게 한다. 노트북 제목이나 Kaggle Input 이름을 보고 checkpoint 출처를 추측하는 일을 없애는 것이 목표다.

## 적용 우선순위

1. Hugging Face cache 및 임시 데이터가 Version Output에 포함되지 않도록 변경
2. fast/slow를 config로 옮기고 `forget.py` 문자열 패치 제거
3. 공통 데이터 준비 및 checkpoint 검증 스크립트 추가
4. Hydra config를 method/runtime/tokenizer 계층으로 분리
5. 안정된 재현 노트북 하나와 진단 노트북 하나만 유지
6. 이전 실험 노트북은 수정하지 않고 `notebooks/archive/`로 이동해 기록으로 보존

당장 진행 중인 실험 결과를 보존한 뒤 1~3번부터 적용하면 결과를 바꾸지 않으면서 실행 준비 시간과 혼동을 가장 크게 줄일 수 있다.
