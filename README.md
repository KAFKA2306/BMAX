# BMAX REX 転換社債ETF 包括的研究プロジェクト

## 🚀 プロジェクト概要

**REX BMAX Bitcoin & Crypto Convertible & Income ETF** の理論的・実証的研究を通じて、デジタル資産時代における革新的金融商品の学術的基盤を構築する包括的研究プロジェクトです。

**🔬 革新性**: ビットコイン関連企業転換社債という新資産クラスの数学的価格決定理論とETF流動性変換メカニズムの世界初の包括的分析

---

## 📁 プロジェクト構造

```
BMAX/
├── docs/                                    # 理論文書・研究設計
│   ├── mathematical_theoretical_framework.md    # 数学的理論基盤
│   └── empirical_research_design.md            # 実証研究設計
├── src/                                     # 計算実装
│   └── bmax_computational_framework.py         # 統合計算フレームワーク
├── analysis/                                # 分析結果
├── models/                                  # 訓練済みモデル
└── README.md                               # このファイル
```

## 🎯 研究目標

### 1. 理論的貢献
- **ハイブリッド証券価格理論**: Bitcoin連動転換社債の数学的価格決定モデル
- **三層資産相関理論**: Bitcoin → 企業株式 → 転換社債の複合価格形成
- **ETF流動性変換理論**: 非流動資産のETF化による流動性創出メカニズム

### 2. 実証的発見
- **非線形価格伝播**: Bitcoin価格ショックの転換社債への閾値効果
- **下方リスク保護**: 転換社債構造による下方リスク軽減の定量評価
- **最適ポートフォリオ配分**: リスク調整後リターン最大化戦略

## 🧮 数学的理論フレームワーク

### 転換社債価値分解
```
CV_t = B_t + C_t + Credit_Spread_t + Liquidity_Premium_t
```

### 三層資産価格過程
**第1層: Bitcoin**
```
dB_t = μ_B B_t dt + σ_B B_t dW_t^(B)
```

**第2層: Bitcoin企業株式**
```
dS_t = μ_S S_t dt + σ_S S_t (ρ dW_t^(B) + √(1-ρ²) dW_t^(S))
```

**第3層: 転換社債**
```
dCV_t = ∂CV/∂t dt + ∂CV/∂S dS_t + ½∂²CV/∂S² (dS_t)² + ∂CV/∂B dB_t
```

### 複合オプション価格決定
```
C_compound = S₀ × N₂(a₁, b₁; ρ) - K₁ × e^(-r₁T₁) × N₂(a₂, b₂; ρ) - K₂ × e^(-r₂T₂) × N(b₂)
```

## 📊 主要研究仮説

### H1: 非線形価格伝播メカニズム
```
ΔCV_t = α + β₁ΔB_t + β₂(ΔB_t)² + γ I_{|ΔB_t| > τ} + ε_t
```

### H2: ETF流動性変換効果
```
Liquidity_BMAX,t = δ + Σᵢ wᵢ Liquidity_i,t + η · ETF_Structure_t + u_t
```

### H3: 下方リスク保護効果
```
Downside_Risk_BMAX = κ · Downside_Risk_Bitcoin_Direct + Bond_Floor_Effect + v_t
```

## 💻 計算フレームワーク

### 主要コンポーネント

#### 1. **BitcoinBlackScholesEngine**
- 配当調整済みBlack-Scholesオプション価格決定
- Greeks計算（Delta, Gamma, Vega, Theta）
- インプライドボラティリティ算出

#### 2. **ThreeLayerAssetModel**
- Bitcoin-株式-転換社債の三層相関構造モデリング
- モンテカルロシミュレーション（相関のある価格パス生成）
- 実現相関分析

#### 3. **CompoundOptionEngine**
- 複合オプション価格決定（Geske 1979モデル）
- 二段階価格形成プロセスのモデリング
- 複合オプションGreeks計算

#### 4. **ConvertibleBondEngine**
- 転換社債総合価格決定
- 債券フロア計算
- 転換プレミアム分析

#### 5. **ETFLiquidityEngine**
- 流動性変換比率計算
- プレミアム・ディスカウント動態分析
- 作成・償還メカニズムモデリング

#### 6. **BMXRiskEngine**
- Bitcoin-Aware VaR計算
- Expected Shortfall（条件付きVaR）
- テールリスク指標（Hill推定量等）
- 最適ヘッジ比率算出

### 使用例

```python
from src.bmax_computational_framework import BMAXIntegratedEngine

# エンジン初期化
bmax_engine = BMAXIntegratedEngine()

# 現在価格設定
current_prices = (45000.0, 150.0, 1050.0)  # Bitcoin, Stock, CB

# 市場条件
market_conditions = {
    'volatility': 0.65,
    'regime': 'normal'
}

# 包括的分析実行
results = bmax_engine.comprehensive_analysis(current_prices, market_conditions)

# 結果出力
print(f"理論価格: ${results['convertible_bond_analysis']['theoretical_price']:.2f}")
print(f"Bitcoin-CB相関: {results['correlation_structure']['bitcoin_cb']:.3f}")
print(f"VaR (5%): {results['risk_metrics']['bitcoin_aware_var']:.4f}")
```

## 📈 実証研究設計

### データソース
- **価格データ**: Bloomberg, Yahoo Finance, CoinGecko
- **高頻度データ**: Interactive Brokers, Polygon.io
- **転換社債データ**: OptionMetrics, TRACE
- **センチメントデータ**: Twitter, Reddit API

### 分析手法
1. **計量経済学**: VAR/VECM, GARCH, 閾値回帰, レジーム・スイッチング
2. **機械学習**: Random Forest, XGBoost, LSTM, Transformer
3. **ベイズ統計**: MCMC, Hamiltonian Monte Carlo, NUTS

### 検証フレームワーク
- **時系列交差検証**: Walk-forward analysis
- **ブートストラップ**: Block bootstrap, Wild bootstrap
- **ロバスト性検定**: Diebold-Mariano test, Model Confidence Set

## 🎯 期待される成果

### 学術的貢献
1. **新理論構築**: デジタル資産連動ハイブリッド証券理論
2. **査読付き論文**: 3-4本（Journal of Financial Economics等）
3. **国際学会発表**: 5-6回（AFA, EFA, AFBC等）

### 実務的インパクト
1. **商品開発**: 類似ETF商品の理論的・実証的基盤
2. **リスク管理**: Bitcoinエクスポージャーの新ヘッジ手法
3. **投資戦略**: 機関投資家向け最適配分モデル

### オープンソース貢献
1. **Pythonパッケージ**: `bmax-analytics` ライブラリ
2. **研究再現コード**: 完全なレプリケーションセット
3. **教育リソース**: 大学院レベル教材

## 🚀 クイックスタート

### 1. 環境設定

```bash
# リポジトリクローン
git clone [repository_url]
cd BMAX

# 仮想環境作成
conda create -n bmax-research python=3.11
conda activate bmax-research

# 依存関係インストール
pip install -r requirements.txt
```

### 2. 基本分析実行

```bash
# 理論フレームワークのテスト
python src/bmax_computational_framework.py

# カスタム分析
python -c "
from src.bmax_computational_framework import BMAXIntegratedEngine
engine = BMAXIntegratedEngine()
results = engine.comprehensive_analysis((45000, 150, 1050), {'volatility': 0.6})
print('BMAX分析完了:', results['portfolio_characteristics'])
"
```

### 3. 高度な分析

```python
import pandas as pd
from src.bmax_computational_framework import BMAXIntegratedEngine

# シナリオ分析
scenarios = [
    {
        'name': 'Bull Market',
        'prices': (60000, 200, 1200),
        'conditions': {'volatility': 0.5, 'regime': 'bull'}
    },
    {
        'name': 'Bear Market', 
        'prices': (30000, 100, 950),
        'conditions': {'volatility': 0.8, 'regime': 'bear'}
    }
]

engine = BMAXIntegratedEngine()
scenario_results = engine.scenario_analysis(scenarios)
print(scenario_results)
```

## 📚 研究文書

### 理論文書
- **[mathematical_theoretical_framework.md](docs/mathematical_theoretical_framework.md)**: 包括的数学的理論基盤
- **[empirical_research_design.md](docs/empirical_research_design.md)**: 実証研究設計詳細

### 重要な理論的概念
1. **ハイブリッド証券価値分解理論**
2. **三層資産相関構造モデル**
3. **複合オプション価格決定理論**
4. **ETF流動性変換メカニズム**
5. **Bitcoin-Aware リスク測定手法**

## 🎓 学術的革新性

### 理論的革新
- **世界初**: ビットコイン関連転換社債の包括的数学的理論
- **新手法**: 三層資産価格形成の確率論的モデリング
- **統合**: デジタル資産理論と伝統的金融理論の融合

### 実証的革新
- **大規模データ**: 5年間の高頻度データによる包括的実証分析
- **多手法統合**: 計量経済学・機械学習・ベイズ統計の統合アプローチ
- **リアルタイム検証**: アウトオブサンプル予測による実用性検証

## ⚠️ リスク・制約事項

### データリスク
- **新商品**: BMAX の取引履歴が限定的
- **市場変動**: ビットコイン市場の高ボラティリティ
- **構造変化**: デジタル資産規制環境の変化

### モデルリスク
- **複雑性**: 三層資産モデルの推定困難性
- **仮定**: 確率過程仮定の現実からの乖離
- **計算負荷**: 大規模モンテカルロシミュレーションの計算コスト

## 🤝 貢献・協力

### 研究協力募集
- **データ提供**: 転換社債・暗号資産データ
- **理論発展**: 数学的モデル改良
- **実証検証**: 追加的実証分析

### オープンソース貢献
```bash
# 貢献手順
1. Fork the repository
2. Create feature branch: git checkout -b feature/new-model
3. Commit changes: git commit -m "Add new pricing model"
4. Push to branch: git push origin feature/new-model
5. Submit Pull Request
```

## 📞 連絡先

**研究責任者**: 金融工学研究チーム  
**所属**: 量的金融分析研究所  
**専門分野**: 金融工学、デリバティブ価格理論、リスク管理  

---

## 📄 ライセンス

MIT License - 学術研究・教育目的での使用を推奨

## 🙏 謝辞

本研究は、デジタル資産時代における金融理論の発展に寄与することを目指しています。ビットコイン関連金融商品の理論的理解の向上と、実務での適切な活用促進に貢献できれば幸いです。

---

**🔥 革新的金融商品の学術的解明により、デジタル資産時代の新しい投資理論を創造**# BMAX
