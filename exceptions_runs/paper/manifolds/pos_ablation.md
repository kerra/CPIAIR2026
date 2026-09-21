# Positional-embedding ablation (matched_2M_pos): layer-0 structure without vs with learned positional embeddings

Joint runs, final checkpoint, seed means; layer-0 locations only (full table in pos_ablation.csv). `best` marks the best layer-0 location under each condition.

| arch | task | location | value_mean_nopos | value_mean_pos | delta | modes_nopos | modes_pos | best |
|---|---|---|---|---|---|---|---|---|
| transformer | add | L0_eq | 0.92 | 0.96 | 0.04 | 30 / 3 / 3 | 3 / 3 / 3 | best no-pos & pos |
| transformer | div | t0_ffn_mid_eq | 0.94 | 0.77 | -0.16 | 74 / 74 / 14 | 12 / 17 / 3 | best no-pos & pos |
| transformer | max | t0_q_rope_eq | 0.89 | 0.82 | -0.07 | – / – / – | – / – / – | best no-pos |
| transformer | max | t0_k_eq | 0.88 | 0.88 | 0.00 | – / – / – | – / – / – | best pos |
| mamba | add | m0_gated_y_eq | 0.85 | 0.71 | -0.15 | 3 / 9 / 3 | 31 / 4 / 3 | best no-pos & pos |
| mamba | div | m0_x_stream_eq | 0.65 | 0.86 | 0.20 | 74 / 30 / 4 | 3 / 10 / 74 | best no-pos |
| mamba | div | m0_skip_y_eq | 0.52 | 0.94 | 0.43 | 3 / 30 / 57 | 3 / 10 / 74 | best pos |
| mamba | max | m0_x_stream_eq | 0.51 | 0.43 | -0.07 | – / – / – | – / – / – | best no-pos & pos |
| kimi | add | K0_eq | 0.98 | 0.96 | -0.03 | 50 / 50 / 34 | 37 / 49 / 55 | best no-pos & pos |
| kimi | div | kda0_q_eq | 0.48 | 0.50 | 0.02 | 3 / 19 / 17 | 7 / 7 / 7 | best no-pos |
| kimi | div | kda0_v_eq | 0.48 | 0.80 | 0.32 | 3 / 19 / 17 | 17 / 17 / 17 | best pos |
| kimi | max | kda0_gate_eq | 0.86 | 0.86 | -0.00 | – / – / – | – / – / – | best no-pos & pos |

Grok order, matrix (no pos):

    kimi         s1000: max(275) → div(11900) → add(13675)
    kimi         s1001: max(275) → div(17125) → add(17650)
    kimi         s1002: max(275) → div(6025) → add(8125)
    mamba        s1000: max(200) → div(2675) → add(6225)
    mamba        s1001: max(150) → div(2200) → add(5550)
    mamba        s1002: max(175) → div(2900) → add(4375)
    transformer  s1000: max(175) → add(2600) → div(8025)
    transformer  s1001: max(150) → add(2500) → div(9300)
    transformer  s1002: max(125) → add(2700) → div(9550)

Grok order, matched_2M_pos:

    kimi         s1000: max(225) → div(9125) → add(15525)
    kimi         s1001: max(250) → div(11300) → add(13075)
    kimi         s1002: max(225) → div(11875) → add(13700)
    mamba        s1000: max(175) → div(2825) → add(4025)
    mamba        s1001: max(150) → div(1650) → add(4375)
    mamba        s1002: max(175) → div(7450) → add(10075)
    transformer  s1000: max(200) → add(3175) → div(8775)
    transformer  s1001: max(150) → add(2325) → div(10625)
    transformer  s1002: max(175) → add(2475) → div(10450)
