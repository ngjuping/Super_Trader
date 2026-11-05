# 个性化规则：每个标的自己的规律

- 生成时间：2026-11-05 23:26:54
- 标的池：mainstream（62 个）
- 检验总数：17,834
- 通过验证：72 条规则-标的对 / 35 个标的 / 20 条不同规则
- 安慰剂：平均 16.7 条 / 7.0 个标的，信噪比 4.3×
- 个性化增益：+1.01pp（安慰剂 +4.83pp，17/35 个标的为正）

## 每个标的的主规则

| 标的 | 规则 | 周期 | 留出段优势 | 特异性 | 四段复现 | 触发率 | 通用规则最好 | 增益 |
|---|---|---|---|---|---|---|---|---|
| 紫金矿业(601899) | 日历__month_end_long | H3 | +20.52pp | +16.49pp | +28 +21 +9 +23 | 4.1% | 日历__month_end_long H3 | +0.00pp |
| 中证红利(000922) | 日历__month_end_long | H3 | +17.46pp | +13.42pp | +30 +24 +5 +11 | 4.0% | 日历__month_end_long H3 | +0.00pp |
| 东方财富(300059) | 均值回归__rsi2_oversold_uptrend | H1 | +16.06pp | +12.22pp | +19 +10 +22 +15 | 2.6% | 日历__month_end_long H3 | +9.17pp |
| 军工ETF国泰(512660) | 日历__month_end_long | H1 | +14.48pp | +10.30pp | +23 +20 +7 +9 | 4.0% | 日历__month_end_long H1 | +0.00pp |
| 伊利股份(600887) | 均值回归__rsi14_extreme | H1 | +14.25pp | +11.20pp | +19 +15 +12 +13 | 2.9% | 日历__month_end_long H1 | +9.64pp |
| 中国建筑(601668) | 日历__month_end_long | H1 | +12.73pp | +8.54pp | +12 +24 +6 +10 | 4.0% | 日历__month_end_long H1 | +0.00pp |
| 中国银行(601988) | 均值回归__far_from_ema20_15 | H5 | +12.31pp | +11.08pp | +22 +17 +4 +7 | 8.2% | 日历__month_end_long H3 | +2.85pp |
| 国证2000(399303) | 日历__month_end_long | H1 | +12.05pp | +7.86pp | +20 +14 +3 +11 | 4.0% | 日历__month_end_long H1 | +0.00pp |
| 中证军工(399967) | 日历__month_end_long | H1 | +11.71pp | +7.52pp | +12 +16 +10 +9 | 4.0% | 日历__month_end_long H1 | +0.00pp |
| 酒ETF鹏华(512690) | 日历__friday_trend | H1 | +11.26pp | +7.82pp | +20 +6 +9 +9 | 6.1% | 均值回归__rsi2_extreme H1 | +3.32pp |
| 中概互联网ETF易方达(513050) | 日历__friday_trend | H1 | +10.91pp | +7.48pp | +7 +20 +8 +10 | 6.2% | 日历__month_end_long H3 | +4.01pp |
| 国证钢铁(399440) | 日历__month_end_long | H1 | +10.67pp | +6.49pp | +14 +11 +3 +14 | 4.0% | 日历__month_end_long H3 | -0.51pp |
| 中证500ETF南方(510500) | 日历__month_end_long | H1 | +10.11pp | +5.93pp | +13 +16 +10 +1 | 4.0% | 日历__month_end_long H1 | +0.00pp |
| 创业板指(399006) | 日历__friday_trend | H1 | +10.08pp | +6.64pp | +19 +4 +15 +1 | 6.4% | 均值回归__bb_pctb_extreme H1 | +4.52pp |
| 证券ETF国泰(512880) | 均值回归__bb_pctb_soft | H1 | +10.00pp | +8.57pp | +10 +6 +14 +12 | 12.8% | 均值回归__bb_pctb_extreme H1 | -0.64pp |
| 深证成指(399001) | 组合__trend_plus_vol | H1 | +9.74pp | +9.55pp | +14 +10 +7 +4 | 3.4% | 日历__month_end_long H1 | +3.99pp |
| 创业板ETF易方达(159915) | 日历__friday_trend | H1 | +9.61pp | +6.17pp | +15 +2 +12 +8 | 6.6% | 均值回归__bb_pctb_extreme H1 | +4.17pp |
| 中证煤炭(399998) | 日历__month_end_long | H1 | +8.96pp | +4.78pp | +5 +14 +14 +2 | 4.0% | 日历__month_end_long H1 | +0.00pp |
| 中小100(399005) | 日历__friday_trend | H1 | +8.73pp | +5.29pp | +11 +10 +6 +7 | 6.3% | 日历__month_end_long H1 | +2.07pp |
| 中证农业(000949) | 日历__month_end_long | H1 | +8.43pp | +4.24pp | +6 +17 +6 +4 | 4.0% | 日历__month_end_long H1 | +0.00pp |
| 长江电力(600900) | 均值回归__rsi14_extreme | H1 | +7.72pp | +4.67pp | +21 +12 +1 +1 | 2.9% | 日历__month_end_long H3 | +0.29pp |
| 贵州茅台(600519) | 均值回归__rsi2_oversold_uptrend | H1 | +7.65pp | +3.81pp | +0 +22 +3 +5 | 3.0% | 连续__consec_dn5_fade H1 | +4.53pp |
| 沪深300ETF华泰柏瑞(510300) | 连续__consec_dn3_fade | H1 | +7.53pp | +4.54pp | +6 +6 +2 +23 | 4.7% | 均值回归__bb_pctb_extreme H1 | +3.01pp |
| 比亚迪(002594) | 连续__consec_up3_fade | H1 | +7.35pp | +2.97pp | +4 +10 +1 +20 | 3.9% | 日历__month_end_long H3 | +0.71pp |
| 沪深300(000300) | 日历__friday_trend | H1 | +7.09pp | +3.66pp | +6 +8 +10 +4 | 6.4% | 连续__consec_dn5_fade H1 | -6.94pp |
| 中证1000(000852) | 组合__trend_plus_vol | H1 | +6.66pp | +6.48pp | +13 +5 +3 +1 | 3.8% | 日历__month_end_long H1 | -0.41pp |
| 证券公司(399975) | 均值回归__far_from_ema20_15 | H1 | +6.66pp | +5.02pp | +11 +3 +8 +5 | 10.5% | 均值回归__bb_pctb_extreme H1 | -1.88pp |
| 中证500(000905) | 日历__month_end_long | H1 | +6.63pp | +2.45pp | +5 +13 +7 +3 | 4.0% | 连续__consec_dn5_fade H1 | -9.78pp |
| 中证白酒(399997) | 日历__friday_trend | H1 | +6.33pp | +2.90pp | +6 +5 +10 +4 | 6.7% | 日历__month_end_long H3 | +4.08pp |
| 创业板50ETF华安(159949) | 趋势__s16_10 | H3 | +6.27pp | +5.70pp | +6 +2 +12 +5 | 26.1% | 均值回归__rsi2_extreme H1 | +3.15pp |
| 恒瑞医药(600276) | 均值回归__rsi14_extreme | H1 | +6.13pp | +3.08pp | +12 +2 +6 +10 | 3.8% | 均值回归__rsi2_extreme H1 | +1.62pp |
| 中信证券(600030) | 均值回归__rsi2_extreme | H1 | +6.06pp | +3.20pp | +11 +6 +3 +4 | 9.1% | 均值回归__bb_pctb_extreme H1 | -2.41pp |
| 五粮液(000858) | 日历__friday_trend | H1 | +5.34pp | +1.90pp | +8 +3 +6 +4 | 6.7% | 日历__month_end_long H1 | +3.78pp |
| 有色金属(000819) | 趋势__ma_stack | H1 | +3.84pp | +3.26pp | +3 +5 +5 +3 | 31.4% | 日历__month_end_long H3 | -5.70pp |
| 隆基绿能(601012) | 趋势__ma_stack | H1 | +2.74pp | +2.15pp | +0 +8 +2 +0 | 32.1% | 均值回归__rsi2_extreme H1 | -1.18pp |