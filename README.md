# BMAX — ビットコイン関連転換社債ETFの計算研究

[![Benchmark Contract](https://github.com/KAFKA2306/BMAX/actions/workflows/benchmark-contract.yml/badge.svg)](https://github.com/KAFKA2306/BMAX/actions/workflows/benchmark-contract.yml)

ビットコイン関連企業の株式・転換社債とETFを題材に、債券フロア、転換オプション、相関シナリオ、損失指標を計算する研究用プロトタイプです。

**市場校正済みの価格モデルではありません。特定ETFの公正価値、将来価格、流動性改善、下方保護を証明しません。**

## 監査で修正した問題

- Bitcoin価格と株価の絶対水準を直接比較する無意味な警告を削除
- Black–Scholesで`T=0`・`sigma=0`の除算を回避
- 相関行列の対角へ微小値を加えて1を超えさせる処理を廃止
- 相関行列の半正定値性と対角1を検証
- GBMの対数ドリフトへ`-0.5 * sigma^2`を追加
- 並列fork乱数を廃止し、明示seedのベクトル化シミュレーションへ変更
- 複合オプションのbare `except`と任意の臨界価格代入を廃止
- 標準的な同一原資産call-on-callでは相関を`sqrt(T1/T2)`へ固定
- 債券フロアへクーポン現在価値を反映
- 根拠のない「米国型5%プレミアム」を削除
- 転換社債価格が債券フロアまたは転換価値を下回らないよう不変条件を追加
- 常に1.518になる流動性倍率と固定配列を削除
- 流動性変換比率は実測ETF流動性がある場合だけ計算
- VaR・Expected Shortfallを負のリターンではなく正の損失量として統一
- 単なる相関係数を「多様化便益」と呼ばず、`1 - abs(correlation)`を参考代理値として分離
- 入力市場価格とモデル価格の差を保存し、`calibrated_to_market: false`を明記
- 数値不変条件の回帰テストを追加

## 現在のモデル

```text
転換社債近似値
  = 債券フロア
  + 欧州型転換コール近似
  ただし債券フロア・即時転換価値を下回らない
```

債券フロアは額面、満期、無リスク金利、信用スプレッド、連続クーポン近似から計算します。実際の転換社債にあるコール、プット、リセット、強制転換、希薄化、信用事象、税務は未実装です。

三資産シミュレーションは、入力したドリフト・ボラティリティ・相関から価格経路を生成する**シナリオ**であり、予測ではありません。

## 実行

必要な主ライブラリ:

```text
numpy
pandas
scipy
```

```bash
python src/bmax_computational_framework.py
python -m unittest discover -s tests -v
```

API例:

```python
from src.bmax_computational_framework import BMAXIntegratedEngine

engine = BMAXIntegratedEngine()
result = engine.comprehensive_analysis(
    current_prices=(45000.0, 150.0, 1050.0),
    market_conditions={
        "volatility": 0.65,
        "regime": "normal",
        "seed": 0,
        "n_simulations": 1000,
        "n_steps": 252,
    },
)
```

`current_prices`の3番目は観測市場価格として保存されます。理論値は株価・転換条件・信用条件から別に計算され、`model_price_gap`へ差を出します。

## 流動性比率

流動性を計算するには、同じ定義・期間・単位の構成銘柄流動性とETF実測流動性が必要です。

```python
market_conditions = {
    "individual_liquidity": [0.2, 0.4],
    "liquidity_weights": [0.5, 0.5],
    "observed_etf_liquidity": 0.6,
}
```

実測値がなければ`liquidity_transformation_ratio`は`None`で、推定しません。

## テストする不変条件

- ゼロボラティリティでも価格が有限
- 相関行列の対角が1で半正定値
- 同じseedで同じシナリオ
- 転換社債価格が債券フロア・転換価値以上
- クーポン増加で債券フロアが増加
- 観測ETF流動性なしでは倍率を生成しない
- VaRとExpected Shortfallが正の損失量
- 統合エンジンが市場校正済みと自己申告しない

## 実証に必要な追加データ

- ETFの公式保有銘柄・基準日・ウェイト
- 各転換社債の契約条項と信用スプレッド
- 金利曲線、為替、経費率、分配、設定・交換情報
- 同一定義の出来高、スプレッド、価格インパクト
- 校正期間と凍結OOS期間
- ベースラインと取引費用

本プロジェクトは投資助言、価格保証、売買推奨ではありません。

**README最終監査:** 2026-08-02