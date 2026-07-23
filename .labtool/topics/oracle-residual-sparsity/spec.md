# 구현 지시서: Oracle Intermediate Sparsity — Residual(mean-gate) 보상 검증 실험

> Preserved verbatim from the user's implementation spec, 2026-07-22.
> This is the authoritative source; gist.md is the working summary.
> 이 문서는 R-Sparse 베이스라인 코드 위에 새로운 oracle sparsity 실험을 구현하기 위한 완전한 사양서다.
> 구현자는 이 문서만으로 작업할 수 있어야 한다. 불명확한 부분은 임의로 해석하지 말고 질문으로 남길 것.

---

## 0. 실험의 목적 (배경 최소 요약)

LLM의 FFN(gated MLP)에서 intermediate activation `i = u ⊙ g`의 sparsity를 활용하면 FFN의 세
linear(`gate_proj`, `up_proj`, `down_proj`)를 모두 건너뛸 수 있다. 이 실험은 **oracle 세팅**(mask를
실제 activation 값으로부터 계산; predictor 없음, 커널 없음, 학습 없음)에서 다음 가설을 검증한다:

- **H1 (주 가설)**: gate의 평균값 `ḡ`를 이용해 FFN을 "선형 성분 + residual"로 분해하면, residual
  `r = u ⊙ (g − ḡ)`의 분포가 `i`보다 집중되어 있어, 같은 유효 연산량에서 더 높은 sparsity로
  정확도를 유지할 수 있다.
- **H2**: top-p 선택 기준에 `down_proj` 열 norm을 곱하는 weight-aware 교정만으로도 이득이 있다.
- **H3**: 보상 행렬 M의 rank-r 근사가 H1의 이득을 잠식하지 않는 r(≤ h/8)이 존재한다.

**이 실험은 품질(정확도) 측정 전용이다. 속도/커널/실제 skip 구현은 범위 밖이다.**
모든 sparsification은 "전체를 계산한 뒤 마스킹"하는 시뮬레이션으로 구현한다 (oracle이므로 수학적으로 동일).

---

## 1. 표기와 수학적 정의 (구현의 기준)

HuggingFace LLaMA 계열 네이밍 기준. FFN 한 층에서:

```
x  ∈ R^h        : mlp 입력 (post-attention-layernorm 출력)   [B, T, h]
u  = up_proj(x)                = x @ W_up.T        ∈ R^d      [B, T, d]
g  = act_fn(gate_proj(x))      = σ(x @ W_gate.T)  ∈ R^d      [B, T, d]
i  = u * g                     (elementwise)       ∈ R^d
y  = down_proj(i)              = i @ W_down.T      ∈ R^h
```

- `σ`는 모델의 act_fn을 그대로 사용 (LLaMA/Qwen: SiLU, Gemma: GELU-tanh). **하드코딩 금지, `self.act_fn` 재사용.**
- `W_down ∈ R^{h×d}`. 뉴런 j에 대응하는 것은 `W_down[:, j]` (열).
- calibration으로 추정하는 통계 (모두 뉴런별, 층별):

```
ḡ[j]   = E[g_j]                        (단순 평균)
ḡ*[j]  = E[u_j² · g_j] / E[u_j²]       (u²-가중 평균; C5에서 사용)
col_norm[j] = || W_down[:, j] ||₂      (데이터 불필요, weight에서 직접)
```

- 보상 행렬과 rank-r 근사:

```
M = W_down @ diag(ḡ) @ W_up            ∈ R^{h×h}   (층별, offline 1회)
SVD: M = U S V^T
A = diag(sqrt(S[:r])) @ V[:, :r].T     ∈ R^{r×h}
B = U[:, :r] @ diag(sqrt(S[:r]))       ∈ R^{h×r}
comp_lr(x) = (x @ A.T) @ B.T           ≈ M x       (런타임 비용 2hr)
```

- residual 정의: `r_j = u_j · (g_j − ḡ[j])`

### Top-p mask (토큰별, 층별)

score 벡터 `s ∈ R^d_{≥0}`가 주어졌을 때 (조건별 score는 §2):

```
1. 내림차순 정렬 s_sorted, 인덱스 idx
2. c = cumsum(s_sorted)
3. k* = 최소 k s.t. c[k] ≥ p · c[-1]
4. mask m: idx[0..k*] 위치 1, 나머지 0
```

벡터화: `[B*T, d]`로 flatten 후 `torch.sort` + `cumsum` + `searchsorted`. 8B 모델(d=14336)에서
메모리 부족 시 토큰 청크 단위로 처리. 달성 sparsity `s_achieved = 1 − mean(m)`을 반드시 기록
(p는 노브일 뿐, 보고 축은 달성 sparsity다).

---

## 2. 실험 조건 (ablation ladder)

| ID | score (top-p 대상) | 출력 계산식 |
|---|---|---|
| C0 | — | dense (수정 없음) |
| C1 | `\|i_j\|` | `ŷ = (m⊙i) @ W_down.T` |
| C2 | `\|i_j\| · col_norm[j]` | C1과 동일 |
| C3 | `\|r_j\| · col_norm[j]` | `ŷ = (m⊙i) @ W_down.T + ((1−m) ⊙ (ḡ ⊙ u)) @ W_down.T` |
| C4 | C3과 동일 | `ŷ = (m⊙i) @ W_down.T + comp_lr(x) − (m ⊙ (ḡ ⊙ u)) @ W_down.T` |
| C5 | C3에서 ḡ → ḡ* | C3 계산식에서 ḡ → ḡ* |
| C6 (옵션) | C2/C4의 그룹판 (§6) | 그룹 단위 mask |

**C3 계산식의 유래 (구현자가 이해해야 할 핵심):** 정확한 FFN은
`y = Σ_j u_j g_j W_down[:,j]`이고, 자른 뉴런(j∉K)의 g_j를 ḡ[j]로 근사하면 tail이
`Σ_{j∉K} ḡ_j u_j W_down[:,j] = ((1−m)⊙(ḡ⊙u)) @ W_down.T`가 된다. 시뮬레이션에서는 u 전체를
이미 계산했으므로 이 식을 직접 쓴다 (배포 시에는 `Mx − Σ_{j∈K} ḡ_j u_j W_down[:,j]`로 동치 변환
되어 u의 tail 계산이 불필요해진다 — C4가 그 배포형의 rank-r 버전을 시뮬레이션하는 것).

**C4 계산식:** `comp_lr(x) ≈ Mx = 모든 j에 대한 Σ ḡ_j u_j W_down[:,j]`이므로, 살린 뉴런(j∈K) 몫
`(m ⊙ (ḡ⊙u)) @ W_down.T`를 빼서 tail만 남긴다. 이때 빼기 항의 u_j는 시뮬레이션이 이미 가진 정확한 값.

### 필수 unit test (구현 완료 판정 기준)

1. **p=1 항등성**: C3에서 p=1 (전부 keep) → 출력이 dense와 bit-근사 일치 (`atol=1e-3` bf16 기준).
2. **C4 full-rank 동치**: r = h로 두면 C4 == C3 (`rtol=1e-3`).
3. **rank 진단**: p=1에서 C4 출력 − dense 출력 = `(M̂−M)x` 오차. 층별로 norm 기록 (r 선택 진단용).
4. **마스킹 동치성**: 임의 mask에 대해 "열 슬라이싱으로 skip한 결과"와 "전체 계산 후 마스킹" 일치 확인 (1개 층, 1회).

---

## 3. R-Sparse 베이스라인에서 무엇을 바꾸는가

R-Sparse의 알고리즘(모든 linear의 **입력 채널** sparsity `σ_t(X)W^T + (X−σ_t(X))(A_rB_r)^T` +
EA 층별 budget 검색)은 이 실험과 **다른 알고리즘**이다. 혼입되면 결과가 오염된다.

**재사용할 것:**
- 모델 로딩 / HF transformers 연동 / dtype 처리
- lm-eval-harness 평가 파이프라인 (있다면)
- calibration 데이터 로더 (C4/WikiText 샘플링)
- SVD 유틸리티 (있다면)

**반드시 끌 것 (config로 명시적 비활성화):**
- R-Sparse의 모든 linear 대체 모듈 (attention 4개 + MLP 3개 전부) → **모든 linear는 dense로 동작**
- EA(evolutionary) budget 검색 — 사용하지 않음. 모든 층 동일 p (균일 sweep).

**신규 작성할 것:**
- `OracleSparseMLP`: 기존 `LlamaMLP`(및 Gemma 대응 클래스)의 forward를 대체하는 래퍼.
  - 보유 buffer: `g_bar [d]`, `g_bar_star [d]`, `col_norm [d]`, `A [r,h]`, `B [h,r]`
  - config: `condition ∈ {dense,c1,c2,c3,c4,c5}`, `p: float`, `rank: int`, `exclude_layers: list[int]`
  - `stats_mode` 플래그: True면 sparsify 없이 통계만 축적 (calibration 패스용)
  - 층별 로깅: 달성 sparsity 이동평균
- 스크립트 5개 (§5의 phase에 대응):
  - `01_calibrate.py` — 통계 축적 → `stats/{model}/{layer}.pt` 저장
  - `02_distribution_report.py` — phase-0 분포 지표 (§4)
  - `03_build_M.py` — M 구성 + SVD → `factors/{model}/r{r}/{layer}.pt`
  - `04_eval.py` — 조건×p sweep 실행 → `results/{model}/{condition}/p{p}.json`
  - `05_aggregate.py` — critical sparsity 계산, CSV/플롯 생성

**통계 축적 구현 주의:** forward 중 `u, g`에서 `sum_g, sum_u2, sum_u2g, count`를 **fp32 누적**
(bf16 누적 금지). calibration: 일반 코퍼스(예: allenai/c4 en 샘플) 512 시퀀스 × 2048 토큰.
ḡ 안정성 확인용으로 **두 번째 코퍼스**(wikitext-103)에서도 별도 ḡ를 뽑아 저장.

---

## 4. Phase 0: 분포 지표 (정확도 평가 전에 실행 — 조기 신호)

calibration 데이터 위에서, 각 층별로 `i`와 `r` 두 벡터에 대해:

1. **유도 sparsity 곡선**: p ∈ {0.7, 0.8, 0.9, 0.95, 0.99}에서 top-p가 유도하는 sparsity의
   토큰 평균. 같은 p에서 sparsity가 높을수록 분포가 집중된 것. (H1의 직접 예고편 — **r 곡선이
   i 곡선보다 위에 있어야 go 신호**)
2. **집중도 지표** (토큰별 계산 후 평균): Hoyer measure `(√d − ‖v‖₁/‖v‖₂)/(√d − 1)`, kurtosis
3. **gate 통계**: `CV²[j] = Var(g_j)/E[g_j]²`의 층별 분포; `P(|g_j − ḡ_j| < |g_j|)` (보상이 항별로 이기는 비율)
4. **ḡ 코퍼스 안정성**: c4-ḡ vs wikitext-ḡ의 Pearson 상관 (층별)

산출물: 층 × 지표 테이블 (CSV) + 곡선 플롯 (matplotlib, 층 4개 샘플: 첫/초반/중간/끝).

---

## 5. Phase 계획과 실행 순서

| Phase | 내용 | 완료 게이트 |
|---|---|---|
| 1 | 인프라 + `OracleSparseMLP` + §2 unit test 4종 | 테스트 전부 통과 |
| 2 | calibration + Phase-0 분포 리포트 (3B 모델) | 리포트 생성, r vs i 곡선 확인 |
| 3 | C1 sweep (3B → 8B) | 8B intermediate critical sparsity가 50–70% 범위에 들어오는지 (문헌 대조 sanity) |
| 4 | C2–C5 sweep (8B 중심, 3B/Gemma 반복) | 메인 결과 테이블 |
| 5 (옵션) | C6 그룹판, Instruct 모델 | — |

**모델 (우선순위 순):**
1. `meta-llama/Llama-3.2-3B` — 개발/디버깅용
2. `meta-llama/Llama-3.1-8B` — 주 결과
3. `google/gemma-3-4b-pt` — family 일반성 (act_fn 다름 주의)

**평가 태스크 (lm-eval-harness, zero-shot):**
`arc_easy, arc_challenge, piqa, hellaswag, winogrande, boolq, sciq, lambada_openai`
- sweep 단계: `--limit 1000`
- critical point 근방 (아래 판정에 걸치는 p 2–3개): limit 없이 full 평가
- `wikitext` PPL 병용 (연속 신호), 단 판정은 task accuracy로
- GSM8K는 옵션 (생성형이라 느림; 최종 후보 조건에서만)

**p grid:** `[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99]`
(주의: p가 낮을수록 sparsity가 높다. 조건마다 달성 sparsity가 다르므로 보고 축은 달성 sparsity.)

**지표 정의:**
- normalized accuracy = (조건 정확도) / (같은 harness·같은 limit의 dense 정확도)
- **critical sparsity** = normalized accuracy ≥ 0.99를 만족하는 최대 달성 sparsity
  (grid 사이는 달성-sparsity 축에서 선형 보간)
- **유효 연산 비율** (C4만 rank 항 포함): `(1 − s_achieved) + 2r/(3d)`
  — 근거: FFN 3개 matmul의 토큰당 비용 3·h·d, 보상 브랜치 B(Ax)의 비용 2·h·r → 비율 2r/(3d).
  C3는 배포 불가능한 진단 조건임을 결과 테이블에 명시.

**결과 CSV 스키마:**

```
model, condition, p, rank, achieved_sparsity_mean, achieved_sparsity_per_layer(json),
task, acc, acc_norm, dense_acc, normalized_acc, effective_compute_ratio
```

**최종 플롯:** (1) 조건별 sparsity–normalized accuracy 곡선 (모델별), (2) 같은 것을 유효 연산량
축으로, (3) 층별 달성 sparsity 히트맵, (4) Phase-0의 i vs r 곡선.

---

## 6. C6 (옵션): 그룹 단위 mask

- 그룹 크기 G ∈ {8, 32}. 뉴런 인덱스를 연속 블록으로 G개씩 묶음 (permutation 없이 naive 버전 먼저).
- 그룹 score = 그룹 내 뉴런 score의 합. top-p를 그룹 score에 적용, 선택된 그룹은 전 뉴런 keep.
- C2와 C4 각각의 그룹판을 실행. 목적: 그룹화가 critical sparsity를 몇 %p 깎는지 측정.
- (후속 확장 대비) calibration에서 뉴런 간 co-activation 상관 행렬 기반 클러스터링 permutation을
  적용한 버전을 위한 인터페이스(`neuron_perm: LongTensor | None`)만 미리 뚫어둘 것.

---

## 7. 함정 목록 (구현 시 필수 준수)

1. **R-Sparse 알고리즘과의 혼입 금지** — 우리 조건 실행 시 모든 linear는 dense. attention 절대 건드리지 않음.
2. **act_fn 하드코딩 금지** — Gemma는 GELU-tanh.
3. **dtype**: forward는 모델 기본(bf16), score/cumsum/통계는 fp32로 캐스팅.
4. **mask는 토큰별·층별 매번 계산** — 캐싱 금지 (oracle 정의).
5. **exclude_layers 옵션** 구현 (기본 빈 리스트; 분석에서 초반 층 dense 유지 변형 실험용).
6. dense 기준 정확도는 반드시 **동일 harness 버전·동일 limit**으로 같은 런에서 측정.
7. 생성형 태스크에서 mask는 생성 스텝마다 적용됨 (forward 훅이면 자동으로 충족).
8. 랜덤성 통제: seed 고정, calibration 시퀀스 목록을 파일로 저장해 재현 가능하게.
9. 결과 파일에 git commit hash, config 전문을 함께 저장.

## 8. 판정 기준 (참고 — 구현자는 결과만 산출하면 됨)

- Go: 8B에서 C4(r=d/8 이하) critical sparsity가 C2 대비 +15%p 이상
- Partial-go: C3만 이기고 C4에서 붕괴 (rank 문제)
- No-go: C3 − C2 < 5%p (residual 집중 가설 기각)
