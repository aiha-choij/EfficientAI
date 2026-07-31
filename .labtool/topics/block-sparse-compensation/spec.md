# 요청 전문: Block-wise 보상 (C7/C8) + 뉴런 치환 결합 + refit 결론 정직성 보정

> Preserved verbatim from the agent request `20260731-162935-block-compensation`,
> 2026-07-31. This is the authoritative source (the gateway cannot read the
> host wiki `block-sparse-compensation-spec.md`, so this file — quoted inside
> the request — is the only surviving copy). gist.md is the working summary.
> 불명확한 부분은 임의로 해석하지 말고 gist.md의 Open Questions에 남길 것.

---

## 0. 요청 요약 (우선순위 순)

1. **본론 A**: token-block 공유 mask의 sharing tax를 input-dependent 보상으로 회수하는
   실험 — 조건 C7a/C7/C8a/C8 (§2 스펙 전문). 가설 H4(블록 보상 회수율 ≥50%),
   H5(per-token gate 추정 ĝ 기반 보상 > ḡ 기반).
2. **본론 B**: coactivation 뉴런 치환과의 결합 (스펙 Phase 4, P3′) — §3.
3. **병행 (저부하, 결론 정직성용)**: local-loss-refit 결과의 3가지 보정 — C1(ridge prior
   수정), C2(λ sweep), M1(dense 앵커 + log-PPL 헤드라인 교체) — §4.
4. **C9(overflow 하이브리드)는 지금 구현하지 말 것.** C7/C8 결과를 보고 승격 여부를
   판단한다 (스펙 §5 No-go 분기에서만 등장).

## 1. 맥락 — 선행 결과 (전부 게이트웨이 repo에서 확인 가능)

- **local-loss-refit** (완료 요청 `20260731-090908`, gist:
  `.labtool/topics/local-loss-refit/gist.md`): 닫힌 해 W_down refit 단독은
  **고희소성 전용 도구** — s=0.9에서 크게 도움(8B PPL −20.1%, 7-task 평균 +3.97%p),
  s=0.5/0.7에서는 악화. g가 커질수록 절대 이득 증가.
  **지도교수 합의(2026-07-31)**: refit은 정적 선형 보정이라 계산 안 된 토큰 고유 정보
  (sharing tax의 본질)를 복원할 수 없다 → g>1 성능 레버는 input-dependent 보상(C7/C8)과
  뉴런 치환(P3′)이다. 이것이 본 요청의 존재 이유.
- **coactivation-block-structure** (topic 활성, gist + journal 참조): P1(통계)·P2(PPMI
  클러스터링)·P3(블록 mask oracle PPL)까지 완료. **P3 결과: 클러스터 블록이 무작위
  블록을 3/4 세팅에서 2–3× 이기지만, 절대 PPL은 4.5k–24k (anchor 8.11 대비)로 파국**
  (`journal/2026-07-25_experiment-coact-llama2-p3-blocks.md`). 즉 **보상 없는 블록
  mask는 사용 불능** — C7/C8 보상의 필요성을 실측으로 뒷받침하며, 회수율 분모가 크다.
  ⚠ 이 저널 카드는 Results까지만 있고 **Interpretation이 비어 있으며**, gist의
  Active Jobs/Next Experiments가 stale하다(P3 잡 2개는 이미 STATUS=ok). **첫 세션에서
  카드 해석을 마저 쓰고 gist를 현행화할 것** (본론 B의 선행 정리 작업).
- **oracle-residual-sparsity** (topic paused, anchor로 활성 유지): C0–C6 표기·인프라·
  g=1 anchor의 원천. 스펙 원문은 topic 안에 보존됨 (`spec.md`) — §2가 계승하는
  "oracle spec"이 이 파일이다. 코드: `larosa/inference/oracle_mlp.py`,
  `modeling_llama_larosa.py`의 `sparse_mode` 배선.

## 2. 본론 A — 스펙 전문 (호스트 위키 block-sparse-compensation-spec.md, 2026-07-30)

> 아래 인용부의 상대링크 문서들은 게이트웨이에 없다. 대응물: "oracle spec" →
> `.labtool/topics/oracle-residual-sparsity/spec.md`, "coactivation spec §2 실측" →
> `larosa-intermediate-sparsity`/`coactivation-block-structure` gist의 Key Findings.

---

### 0. 배경과 목적

실제 추론(특히 prefill)은 토큰을 **block 단위(GEMM 타일, g = 16–64 연속 토큰)** 로 묶어 처리한다.
per-token mask는 토큰마다 다른 weight 열을 요구해 타일 효율을 깨므로, mask는 블록 공유(m_T)여야
한다. 그러나 선행 실측(coactivation §2: s=0.9에서 인접 토큰 뉴런 겹침 C(1)=0.316)에 따르면
공유 mask는 **sharing tax**(토큰 개인의 중요 뉴런 손실)가 크다. 본 실험은 oracle 세팅에서:

- **H4 (주 가설)**: 공유 mask의 sharing tax는 block-wise 보상(C4의 블록판 + gate-sketch)으로
  상당 부분(회수율 ≥ 50%) 회수 가능하다.
- **H5**: sharing tax 뉴런은 "gate가 평균 ḡ에서 크게 벗어난" 뉴런이므로, ḡ 기반 보상(C4)보다
  per-token low-rank gate 추정(ĝ) 기반 보상(C8)이 우월하다.

**품질(정확도) 측정 전용. 커널/속도는 범위 밖.** 전체 계산 후 마스킹 시뮬레이션 (oracle 정의).
per-token 스레드(H1–H3)는 **폐기가 아니라 g=1 anchor** — 본 실험의 상한 기준선이다.

### 1. 표기 (oracle spec §1 계승 + 블록 추가)

oracle spec §1의 x, u, g, i, y, ḡ, col_norm, M, A/B, comp_lr, top-p 정의를 그대로 사용. 추가:

```
T = {t₁..t_g}         : 같은 시퀀스의 연속 토큰 block, g ∈ {1, 16, 64}  (g=1 → per-token으로 환원)
S_j(T) = Σ_{t∈T} |i_{t,j}| · col_norm[j]     : 블록 집계 score (gauge-고정 L1 mass)
m_T ∈ {0,1}^d         : S(T)에 top-p를 적용한 블록 공유 mask (블록 내 모든 토큰 동일)
sharing tax           : per-token mask 대비 공유 mask로 잃는 정확도 (같은 달성 sparsity 기준)
```

블록 경계 규칙: **시퀀스 경계를 넘지 않는다** (batch 내 다른 시퀀스와 절대 섞지 않음).
시퀀스 길이가 g로 나누어떨어지지 않으면 마지막 블록은 나머지 토큰만으로 구성.
padding 토큰은 S_j(T) 집계에서 제외.

#### Gate/up/down sketch (C8용)

HF 기준 W_gate, W_up ∈ R^{d×h}, W_down ∈ R^{h×d}. 각각 SVD로 rank-r 분해(offline 1회, 층별):

```
W_gate ≈ B_g A_g   (A_g ∈ R^{r_g×h}, B_g ∈ R^{d×r_g})     ĝ_t = σ(B_g (A_g x_t))
W_up   ≈ B_u A_u   (A_u ∈ R^{r_u×h}, B_u ∈ R^{d×r_u})     û_t = B_u (A_u x_t)
W_down ≈ B_d A_d   (A_d ∈ R^{r_d×d}, B_d ∈ R^{h×r_d})
tail-sketch(t) = B_d ( A_d ( (1−m_T) ⊙ (ĝ_t ⊙ û_t) ) )
```

σ는 모델의 act_fn 재사용 (하드코딩 금지). 기본 r_g = r_u = r_d = r_sk 단일 노브로 sweep.

### 2. 실험 조건

oracle spec의 C0–C6 번호를 계승하며 충돌 없이 이어서 부여한다.
비교 anchor: **C2, C4를 g=1로 실행한 결과** (oracle 스레드가 이미 산출 중이면 재사용).

| ID | mask | 출력 계산식 | 역할 |
|---|---|---|---|
| C7a | m_T (블록 공유) | `ŷ = (m_T⊙i) @ W_down.T` | 통제군 (보상 없음 — sharing tax의 크기 측정) |
| C7 | m_T | `ŷ = (m_T⊙i) @ W_down.T + comp_lr(x) − (m_T ⊙ (ḡ⊙u)) @ W_down.T` | C4의 블록판 (B1) |
| C8a | m_T | C7에서 tail을 `((1−m_T) ⊙ (ĝ ⊙ u)) @ W_down.T`로 직접 계산 (u·W_down은 정확값) | 진단용 (B2 상한; 배포 불가) |
| C8 | m_T | `ŷ = (m_T⊙i) @ W_down.T + tail-sketch(t)` | B2 배포형 (gate/up/down 전부 sketch) |
| C9 (옵션) | m_T + per-token overflow | 블록 mask에 토큰별 개인 top-k′ (k′ ≪ K) 추가 keep | 하이브리드 fallback (C7/C8 부진 시) |

- C7의 유래: oracle spec C4와 동일한 수학 (comp_lr(x) ≈ Mx = 전체 뉴런의 ḡ-근사 기여,
  kept 몫을 빼서 tail만 남김). **차이는 m_T가 블록 공유라는 것뿐** — 배포 관점에서는 kept-set
  차감항이 토큰별 열 slicing에서 블록 공통 sliced GEMM으로 바뀌어 오히려 단순해진다.
- C8a vs C8: C8a는 "gate 추정만 sketch"로 두어 B2의 순수 효과(ḡ→ĝ 개선분)를 분리한다.
  C8은 배포 가능한 전량 sketch. C8a ≥ C8 ≥ C7 순서가 기대값이며, C8a−C7이 H5의 직접 증거.

#### 필수 unit test (구현 완료 판정 기준)

1. **g=1 환원**: C7을 g=1로 실행 → oracle spec C4와 출력 bit-근사 일치 (`rtol=1e-3`).
2. **p=1 항등성**: C7, C8 모두 p=1 → dense와 일치 (`atol=1e-3` bf16 기준).
3. **C8 full-rank 동치**: r_sk = full → C8 출력 == dense (`rtol=1e-3`).
4. **블록 공유 확인**: 임의 블록에서 모든 토큰의 mask가 동일함을 assert. 시퀀스 경계·padding 규칙 확인.
5. **마스킹 동치성**: oracle spec unit test 4 계승 (1개 층 1회).

### 3. 지표

- **달성 sparsity**: s_block = 1 − mean_T(|m_T|)/d (블록 단위 집계; 보고 축).
- **critical sparsity**: oracle spec 정의 그대로 (normalized accuracy ≥ 0.99 최대 달성 sparsity).
- **sharing tax 회수율** (본 실험의 주 지표, 같은 달성 sparsity에서):

```
회수율(조건) = [critical_sparsity(조건) − critical_sparsity(C7a)]
             / [critical_sparsity(g=1 C4 anchor) − critical_sparsity(C7a)]
```

- **유효 연산 비율**: `(1 − s_block) + 2r/(3d) + r_sk·(h+d)·𝟙[C8]·2/(3hd)` — 근거: FFN 3 matmul
  토큰당 3hd; comp_lr 2hr; C8 sketch 경로 ≈ 2r_sk(h+d) (r 동일 가정).
  C8a는 배포 불가 진단 조건임을 결과 테이블에 명시 (C3의 관례).

### 4. Phase 계획

| Phase | 내용 | 완료 게이트 |
|---|---|---|
| 1 | `OracleSparseMLP`에 `block_size: int` + C7a/C7/C8a/C8 조건 추가, unit test 5종 | 테스트 전부 통과 |
| 2 | **Sharing tax 곡선** (3B): C7a vs g=1 C2, g ∈ {16, 64} × p grid → 세금을 정확도 축에서 정량화 | coactivation §2의 겹침 실측과 정합성 확인 |
| 3 | C7/C8a/C8 sweep (3B → 8B), r_sk ∈ {d/32, d/16, d/8} | 회수율 테이블 (모델 × g × 조건) |
| 4 (접속) | coactivation P2 치환 결합: PMI 블록 치환 위에서 C7/C8 재실행 → (토큰 g) × (뉴런 B) 2D 타일 | P3′ = "치환 + 블록 mask + 보상" 통합 결과 |

모델·태스크·p grid·CSV 스키마는 oracle spec §5를 그대로 계승 (동일 harness·동일 limit로
dense/anchor/블록 조건을 같은 런에서 측정). 주의: lm-eval loglikelihood 태스크는 전 구간이
prefill이므로 prefill-타일 의미론과 자연 정합 — 생성형(GSM8K)은 옵션이며, 생성 스텝(g=1일 수밖에
없음)은 블록 의미론 밖임을 결과에 명시.

### 5. 판정 기준 (안 — 첫 수치 보고 확정, 선행 카드 관례)

- **Go**: 8B, g=16, s≈0.9 구간에서 C8 회수율 ≥ 50%, 또는 g=1 C4 anchor 대비 ΔPPL ≤ +1.0.
- **Partial-go**: C8a만 회수 (sketch rank 문제) → r_sk 증가/비대칭 rank로 재시도.
- **No-go**: C8a 포함 회수율 < 25% → 보상으로는 sharing tax를 못 잡음 → C9(overflow 하이브리드)
  또는 g 축소, coactivation 치환 축(P2) 우선으로 전환.

### 8. 함정 목록 (스펙 원문 §8)

1. 블록은 **같은 시퀀스의 연속 토큰만** — batch 축으로 묶지 않는다 (decode lane은 별도 스펙).
2. S_j(T) 집계·cumsum은 fp32.
3. g=1 anchor와 블록 조건은 **같은 harness·같은 limit·같은 seed**로 측정 (회수율 분모의 신뢰성).
4. C8의 ĝ⊙û는 sketch끼리의 곱 — u/g 정확값과 절대 혼용 금지 (C8a만 예외이며 진단 표기 필수).
5. 달성 sparsity는 블록 단위 정의(s_block)로 보고 — per-token 정의와 섞으면 회수율이 왜곡된다.
6. 결과 파일에 g, r_sk, git hash, config 전문 저장.

---

## 3. 본론 B — 치환 결합 (Phase 4, P3′) 실행 주의

- 정의: coactivation P2의 PPMI 균형 클러스터링 치환을 뉴런 축에 적용한 뒤, 그 위에서
  C7/C8을 재실행 — (토큰 블록 g) × (뉴런 블록 B) 2D 타일. 비교군: 치환 없는 C7/C8(§2
  Phase 3 결과), 무작위 치환 통제군(coactivation P3 관례).
- **모델 불일치 주의**: 기존 P2 파티션 산출물은 **LLaMA2-7B 전용**
  (`a6000-4:~/workspace/analysis/llama2_p3_partitions_s09.pt`; 수집 스크립트
  `larosa/scripts/p3_collect_cluster_all.py`). §2의 주 모델(3B/8B)과 다르다. 선택지:
  (a) Phase 4만 llama2-7b로 수행 (파티션 재사용, 비용 최소 — dense 5.4738 anchor 보유),
  (b) 8B에 P1/P2를 재수집·재클러스터링 후 결합 (일관성 최선, 비용 추가).
  **권고: (a)를 먼저 1점 돌려 방향을 보고, 유의미하면 (b)로 확장.** 다른 판단이 서면
  근거를 report.md에 남기고 진행해도 좋다.
- 선행 정리(§1의 ⚠): P3 저널 카드 Interpretation 작성 + coactivation gist 현행화를
  Phase 4 착수 전에 끝낼 것.

## 4. 병행 — local-loss-refit 결론 정직성 보정 (topic: local-loss-refit)

GPU 부하가 작으므로 본론과 병행하되, 본론 잡을 막지 말 것. 결과는 local-loss-refit
topic의 gist/journal에 반영 (topic 재활성 — 새 topic 만들지 않는다).

- **C1 (critical)**: `solve_refit`의 ridge가 0-수축 prior다 — 증거가 부족할 때 W̃을 0으로
  당긴다. 마스터 설계의 원형은 **W_down-앵커 damping**:
  `W̃ = (C + λD·W_down_masked_cols)(G + λD)⁻¹`, D = mean(diag(G))·I (기존 λ 정규화 관례
  유지; "증거 없으면 원 가중치 유지"가 되도록 앵커를 W_down의 해당 열로). **s=0.5/0.7
  악화의 대체 원인 후보**이므로, 수정 후 s ∈ {0.5, 0.7, 0.9} × g=1 최소 3점을 재측정해
  기존 결론("저희소성에서 refit 유해")이 prior 아티팩트인지 판별할 것. G/C가 저장돼 있지
  않아 재calibration 필요 (3B 쌍당 ~15분 수준). 3B에서 판별 후 뒤집히는 경우에만 8B 확장.
  **이번 재calibration에서는 G/C 통계 자체를 디스크에 저장할 것** — 이후 λ 재해석·prior
  변형이 재calibration 없이 공짜가 된다.
- **C2**: λ sweep {0.001, 0.01, 0.1} — 스크립트가 이미 `--lambdas`를 지원한다(미사용).
  C1 수정판 기준으로 s=0.9와 s=0.5 각 1점이면 충분.
- **C1 결과에 따라**: s=0.5/0.7이 C1 수정으로 뒤집히면 L2 Dead End도 재검 후보(M3)지만,
  **이번 요청 범위 밖** — report.md에 관찰만 남길 것.
- **M1 (GPU 무료)**: ① dense(s=0) PPL 앵커를 3B/8B에서 측정해 결과 테이블의 절대 프레임
  완성. ② **log-PPL 재계산 반영**: sharing tax 흡수율 헤드라인 "3B 37% / 8B 60%"는 PPL
  점수차 기준의 지수 왜곡 — **log-PPL(nats) 기준 3B ~12% / 8B ~22%로 gist와 해당 저널
  헤드라인을 교체**할 것 (기존 수치는 "PPL 점수 기준" 병기로 강등).
- (선택, 저순위) M2: "corpus mismatch(wt103 vs wt2)" 해석의 사실관계가 약함(같은 소스) —
  calib=c4 대조 1점이 값싸게 가능하면 수행.

## 5. 운영 규칙

- **labtool**: 새 topic `block-sparse-compensation` 신설(labtool-init; §2 스펙과 이 요청문을
  topic에 보존). `oracle-residual-sparsity`는 anchor로 활성 유지(ORPHANED 아님).
  §4 작업은 `local-loss-refit` topic에 기록. `current.md`는 최소 개입.
- **브랜치**: 본론(A/B)은 `main`에서 `auto/block-sparse-compensation` 분기.
  §4(refit 보정)는 refit 코드가 미병합 `auto/local-loss-refit`에만 있으므로 — PR #1이
  이미 병합됐으면 main에서, 아니면 `auto/local-loss-refit`에서 분기한 별도 auto/ 브랜치로
  작업하고 PR 본문에 "PR #1 위에 쌓임"을 명시. 병합은 절대 하지 말 것 (사람 몫).
- **코드 변경 제안은 `agent-pr`만.** 매 호출 끝에 `report.md`(판단 근거)와 `status`
  (WAITING/DONE)를 반드시 갱신.
- **qsub 함정** (선행 요청에서 검증됨): workdir 절대경로(`~` 리터럴 금지), 동시 잡 ≤4,
  a6000-4는 venv `~/workspace/venv-larosa` + `--attn sdpa` + GitHub fetch 불가
  (게이트웨이에서 scp/tar 동기화), 게이트웨이 conda env `larosa`. lm-eval은 `piqa` 제외
  7-task suite (datasets 5.0 비호환, 선행 요청에서 확정).
- **모델 주의**: /raid/LLM에 plain Llama-3.2-3B 없음 — 선행 요청 관례대로 dev는
  `llama3.2-3b-instruct`, 본 매트릭스는 Llama-3.1-8B(순정). 참고 페이스: L0/L1 매트릭스
  ~31분/(s,g)쌍 (게이트웨이 A100 기준).
- **환경 노트 참조**: 완료 요청 `requests/done/20260731-090908-local-loss-refit/report.md`의
  "환경 노트" 절에 검증된 우회법이 있다 — Bash의 /raid 경로 하드 차단(python subprocess
  우회), agent-pr cwd 지정 필수, `gh pr edit` 실패 시 `gh api -X PATCH` 우회.
- **실행 순서 제안**: Phase 1(unit test) → §4 C1/M1 (짧은 잡, Phase 2 calibration과 병행
  가능) → Phase 2 → Phase 3 → §4 C2 → Phase 4. 각 Phase 게이트에서 report.md 갱신.
- 큰 모델 아님(≤8B) — A100/A6000 아무 데나 배차 가능하되 디스크 여유 규칙 준수.
