# 작업 지시: C4 보상행렬 개선 — Whitened SVD + 층별 rank 배분

> Preserved verbatim from the user's work order, 2026-07-24.
> Note: the document references "8B", "기존 p grid", and "normalized accuracy";
> the topic's standing scope (user decisions 2026-07-23) is LLaMA2-7B, top-K
> s={0.5,0.7,0.9}, wikitext-2 PPL — resolution recorded in gist.md.

배경: C4(rank-r 보상)가 C3(full-M)보다 PPL이 크게 나쁨. 원인 후보는
(a) M 스펙트럼이 flat, (b) plain SVD가 입력 분포를 무시. 아래 순서로 진행.

## Task 0 — 진단 (반나절)
1. p=1에서 층별 ‖(M̂−M)x‖₂ 와 실제 tail 크기 E‖((1−m)⊙(ḡ⊙u)) @ W_down.T‖₂ 를
   운영점 p(예: 0.85)의 mask 기준으로 비교 → CSV (layer, trunc_err, tail_norm, ratio)
2. 층별 M의 특이값 스펙트럼과 stable rank ‖M‖_F²/‖M‖₂² 플롯 (Task 1의 whitened 버전과 겹쳐 그리기)

## Task 1 — Whitened SVD (핵심 변경)
1. calibration 때 층별 입력 자기상관 누적: Σ = E[x xᵀ] (fp32, [h,h])
   — 기존 01_calibrate.py에 누적 buffer 추가
2. C = cholesky(Σ + εI), ε = 1e-4 · mean(diag(Σ))
3. 기존 03_build_M.py 수정: SVD를 M이 아니라 (M @ C)에 적용
   U, S, Vᵀ = svd(M @ C);  B = U[:, :r]·diag(√S[:r]);  A = diag(√S[:r])·Vᵀ[:r] @ inv(C)
   (inv(C)는 triangular solve로; 명시적 역행렬 지양)
4. 검증: p=1에서 E‖(M̂−M)x‖ 가 plain SVD 대비 감소하는지 (Task 0 지표 재측정)
5. C4 조건 sweep 재실행 (8B, 기존 p grid 그대로) → C3/기존 C4와 곡선 비교

## Task 2 — 층별 rank 배분
1. 배분 규칙: 층 l의 whitened 스펙트럼 에너지 누적비 기준
   r_l = min{ r : Σ_{k≤r} S_k² / Σ_k S_k² ≥ τ },  τ ∈ {0.90, 0.95}
2. 평균 rank 예산 r̄ ∈ {256, 512, 1024}에 맞춰 τ를 이분탐색으로 조정하는 모드도 구현
   (보고 시 유효 연산 비율은 (1−s) + 2·r̄/(3d) 로 계산)
3. 균일 r vs 배분 r_l 을 같은 r̄ 에서 비교 sweep (8B)

## 산출물
- results/diag_trunc_vs_tail.csv, 층별 스펙트럼 플롯 (plain vs whitened)
- C4-whitened, C4-whitened+alloc 의 sparsity–normalized accuracy 곡선 (C3, 기존 C4 겹쳐서)
- 요약 표: 조건 × r̄ × critical sparsity × 유효 연산 비율

## 주의
- Σ 누적은 fp32, calibration 시퀀스는 기존과 동일 목록 재사용 (재현성)
- whitening은 보상 브랜치에만 적용, 메인 sparse 경로/score 계산은 불변
- 기존 plain-SVD 결과는 삭제하지 말고 baseline으로 보존
