# energyplan — 最佳能源投資規劃模型

從零打造的電力系統**容量擴充最佳化**（capacity expansion planning）工具：
自訂的線性規劃建模層、自己寫的兩階段單純形法求解器，以及一套完整的
多年期、多區域投資規劃模型。核心不依賴任何第三方套件。

給定需求成長、技術成本、燃料價格與政策目標，它回答：

> **哪一年、在哪個區域、蓋多少什麼樣的電廠與儲能，才能用最低的總系統成本
> 可靠地供電並達成減碳目標？**

```
$ python -m energyplan solve examples/island.yaml --html plan.html
```

---

## 快速開始

不需要安裝任何東西就能跑：

```bash
git clone <this repo> && cd Energy-Planning-Model
python -m energyplan example examples/my.yaml --name minimal
python -m energyplan solve examples/my.yaml
```

大型模型建議裝上 SciPy，會自動改用 HiGHS 求解器（快兩個數量級）：

```bash
pip install scipy        # 選用；PyYAML 則用於 .yaml 情境檔
```

以 Python API 使用：

```python
from energyplan import load_scenario, CapacityExpansionModel, text_report

scenario = load_scenario("examples/island.yaml")
result = CapacityExpansionModel(scenario).solve()

print(text_report(result))
print(result.objective)                    # 總系統成本淨現值 (USD)
print(result.lcoe())                       # 系統均化成本 (USD/MWh)
print(result.year(2040).capacity_mw)       # 該年裝置容量組合
print(result.audit())                      # 一致性檢查殘差
```

---

## 模型能表達什麼

| 面向 | 內容 |
|---|---|
| **投資決策** | 各技術逐年新建容量、既有機組提前除役、儲能功率與電量分別選型、輸電線擴建 |
| **時間** | 多個里程碑年份，各自代表數個日曆年；年內以代表日 × 逐時時段描述 |
| **空間** | 多區域，區域間以有損耗的輸電線連結，可決定是否擴建 |
| **技術** | 火力（效率、燃料、排放）、變動再生能源（逐時容量因數）、水力（年度能量預算）、儲能（充放電效率、時長區間、自放電）、氫能機組 |
| **可靠度** | 備用容量率（以容量價值計入再生能源）、缺電量與缺電成本 VOLL |
| **政策** | 碳價、逐年碳排上限、累積碳預算、再生能源佔比、無碳能源佔比 |
| **現實限制** | 逐年建設速率上限、資源潛能上限、前置期（lead time）、爬坡限制、最低出力 |
| **成本動態** | 資本支出可隨年份下降（學習曲線），依**建設年份**計價 |

輸出包含：逐年裝置容量與新建排程、發電結構、棄電量、成本分項、碳排放、
以及由對偶值導出的**邊際電價**、**碳影子價格**與**容量價值**。

---

## 架構

每一層都能單獨使用：

```
energyplan/
├── lp.py           自訂 LP 建模層（運算子多載、稀疏表達式）
├── simplex.py      從零手寫的兩階段單純形法，含對偶值；純標準函式庫
├── solvers.py      後端調度：HiGHS → CBC → 內建單純形法
├── timeseries.py   合成氣象曲線、CSV 讀取、k-medoids 代表日分群
├── data.py         情境結構描述與驗證
├── model.py        容量擴充線性規劃本體
├── results.py      結果萃取、KPI、價格、一致性稽核
├── report.py       終端表格、CSV/JSON 匯出、內嵌 SVG 的 HTML 報告
├── study.py        參數覆寫、敏感度掃描、情境比較
└── cli.py          命令列介面
```

數學式完整寫在 [`docs/model.md`](docs/model.md)。

### 為什麼自己寫求解器

專案的重點是**整條路徑都可檢視**：從線性代數到投資建議，沒有黑箱。
內建求解器讓整套工具在一台只有 Python 的機器上就能跑完，也作為第三方
求解器的交叉驗證基準（`tests/test_simplex.py` 會要求所有後端給出一致的
最佳值與對偶值）。

實務規模上它有極限。同一個模型在不同規模下的實測：

| 規模 | 變數 | 約束 | HiGHS | 內建單純形法 |
|---|---|---|---|---|
| 3 代表日 × 4 時段 | 196 | 214 | 0.01 s | 0.05 s |
| 6 代表日 × 6 時段 | 532 | 598 | 0.01 s | 1.1 s |
| 8 代表日 × 8 時段 | 924 | 1,046 | 0.02 s | 8.8 s |
| `island` 完整情境 | 20,052 | 27,660 | 7.8 s | 記憶體不足，會明確拒絕 |

內建求解器用密集表格法，約束數超過一兩千條就不適用；超過上限時它會**明確
報錯並告訴你怎麼辦**，而不是耗盡記憶體。預設 `--solver auto` 會自動挑最快
的可用後端。

---

## 命令列

```bash
# 求解並輸出報告
python -m energyplan solve examples/island.yaml --html plan.html --csv out/ --json plan.json

# 改參數重跑：太陽能造價改成 600 USD/kW
python -m energyplan solve examples/island.yaml --set technologies.solar.capex=600

# 路徑結尾加 * 表示乘法：離岸風成本上調 30%
python -m energyplan solve examples/island.yaml --set 'technologies.wind_offshore.capex*=1.3'

# 敏感度掃描
python -m energyplan sensitivity examples/island.yaml \
    --vary technologies.wind_offshore.capex --values 1800,2400,3000,3600

# 多情境比較
python -m energyplan compare base.yaml nuclear_phaseout.yaml high_demand.yaml

# 降低時間解析度以加速
python -m energyplan solve examples/island.yaml --days 4 --hours 6
```

---

## 情境檔怎麼寫

YAML 或 JSON 皆可。凡是價格、需求、政策目標都可以寫成 `{年份: 數值}`，
中間年份自動線性內插：

```yaml
years: [2025, 2030, 2035, 2040, 2045, 2050]
horizon_end: 2059
discount_rate: 0.055

regions:
  - name: north
    demand_twh: {2025: 170, 2050: 248}

fuels:
  gas: {price: {2025: 34, 2050: 30}, co2: 0.202}   # USD/MWh_th, tCO2/MWh_th

technologies:
  - name: solar
    kind: vre
    profile: solar
    capex: {2025: 780, 2050: 450}     # USD/kW，學習曲線
    fom: 13                           # USD/kW/yr
    lifetime: 25
    renewable: true
    max_build_per_year: 3500          # MW/yr
    max_total_capacity: 55000         # MW，全系統資源潛能

storage:
  - name: battery
    capex_power: {2025: 150, 2050: 75}    # USD/kW
    capex_energy: {2025: 210, 2050: 88}   # USD/kWh
    efficiency_charge: 0.93
    min_duration: 2
    max_duration: 8

policy:
  reserve_margin: 0.15
  voll: 6000                                    # USD/MWh 缺電成本
  carbon_cap: {2025: 110, 2050: 0.0}            # MtCO2/yr
  renewable_share: {2030: 0.30, 2050: 0.75}
```

單位規則寫在 `energyplan/data.py` 的模組說明。載入時會做結構驗證，錯誤
一次列出而不是逐一報錯。

沒有提供逐時資料時，會自動合成一組結構合理的需求、太陽能與風能曲線
（依緯度計算日照時數、風速具時間自相關與季節性），所以**不需要外部資料
就能跑**。有實測資料時用 `load_profiles_csv()` 讀入 8760 小時的 CSV。

---

## 結果可信嗎

模型的每個結論都附帶可獨立重算的檢查。`result.audit()` 不信任求解器，
而是從回報的結果重新推導：

| 檢查 | 內容 |
|---|---|
| `objective` | 用逐年成本分項重建淨現值，與求解器目標值比對 |
| `energy_balance` | 發電 + 放電 − 充電 − 輸電損失 + 缺電 − 需求 |
| `reserve_margin` | 可靠容量對備用需求的缺口 |
| `capacity_limits` | 是否有技術超出其資源潛能 |

`island` 情境的實際殘差都在機器精度（約 1e-15）。這套稽核在開發過程中
抓到了兩個真實錯誤：一個是區域變數覆蓋了折現率，使資本年金退化成直線
折舊；另一個是縮排錯誤讓儲能的荷電狀態約束整段失效，導致電池可以無中
生電。兩者都不會讓求解器報錯，只會安靜地給出錯誤答案。

103 個測試涵蓋各層，包含一個**可手算驗證**的基準案例：單一技術、平坦
需求，容量、成本、邊際電價與容量價值都與閉式解逐項比對。

```bash
python -m unittest discover -s tests -t .
```

---

## 兩個設計取捨值得說明

**目標不可行時不會直接失敗。** 備用容量、碳上限與再生能源佔比都帶有
高懲罰成本的鬆弛變數。做不到的情境會得到一份標明「哪個目標差多少」的
計畫，而不是一句 `infeasible`。把懲罰價設得夠高，約束就等同硬約束。

**碳影子價格在上限為零時會退化。** 碳排上限剛好為 0 時最佳解落在退化
頂點，對偶值不唯一，求解器可能回報 backstop 價格而非真實邊際減量成本。
工具會偵測並以 `≥` 標記，並提供 `empirical_marginal_carbon_cost(year)`
以重解取有限差分，得到可靠數值（`island` 情境 2050 年：對偶值報
10,000 USD/t，實際為 628 USD/t）。

---

## 這個模型不做什麼

它是線性規劃，沒有整數變數，因此不含機組排程（unit commitment）、最小
啟停時間、啟動成本或機組整數化。它也是完全預知未來的確定性模型，會低估
彈性資產的選擇權價值。這些是長期容量規劃的標準取捨——輸出應視為**篩選
性結論**，而非運轉排程。詳見 `docs/model.md` 末節。

範例中的成本假設取自公開技術報告的常見區間，僅供示範。做真實決策前請換
上你自己的假設。

---

## English summary

`energyplan` is a from-scratch least-cost energy investment planning model. It
contains its own linear-programming modelling layer, its own two-phase simplex
solver with dual values (standard library only), a k-medoids representative-day
reduction, and a multi-period multi-region capacity-expansion LP covering
generation, storage, transmission, reliability and climate policy.

It optimises what to build, where and when, and reports the build schedule,
generation mix, cost breakdown, emissions, and the marginal energy, capacity
and carbon prices implied by the duals. Every result carries an independent
consistency audit. SciPy/HiGHS and PuLP/CBC are used automatically when
installed; all backends are cross-checked against each other in the test suite.

Full formulation: [`docs/model.md`](docs/model.md).
