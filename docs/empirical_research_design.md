# BMAX REX 実証研究設計書

## 研究概要

**研究課題**: ビットコイン関連企業転換社債ETF（BMAX）の実証的価格決定要因分析  
**研究期間**: 24ヶ月（2025年1月 - 2026年12月）  
**研究手法**: 計量経済学的実証分析・機械学習・ベイズ統計  
**データ範囲**: 2020年1月 - 2024年12月（5年間）  

---

## 1. 研究仮説体系

### 1.1 主要研究仮説

#### 仮説H1: 非線形価格伝播メカニズム
**帰無仮説 $H_0$**: Bitcoin価格変動の転換社債価格への伝播は線形的である  
**対立仮説 $H_1$**: Bitcoin価格ショックの転換社債価格への伝播は非線形であり、閾値効果が存在する

**数学的定式化:**
$$\Delta CV_t = \alpha + \beta_1 \Delta B_t + \beta_2 (\Delta B_t)^2 + \gamma \mathbb{I}_{|\Delta B_t| > \tau} + \epsilon_t$$

ここで $\mathbb{I}_{|\Delta B_t| > \tau}$ は閾値 $\tau$ を超える Bitcoin価格変動の指示関数。

#### 仮説H2: ETF流動性変換効果
**帰無仮説 $H_0$**: ETF構造による流動性変換効果は存在しない  
**対立仮説 $H_1$**: BMAX ETF の流動性は構成銘柄の加重平均流動性を有意に上回る

**数学的定式化:**
$$\text{Liquidity}_{BMAX,t} = \delta + \sum_{i=1}^n w_i \text{Liquidity}_{i,t} + \eta \cdot \text{ETF Structure}_t + u_t$$

ここで $\eta > 0$ が ETF 流動性プレミアムを表す。

#### 仮説H3: 下方リスク保護効果
**帰無仮説 $H_0$**: 転換社債構造による下方リスク保護効果は存在しない  
**対立仮説 $H_1$**: BMAX の下方リスク（VaR, ES）は同等 Bitcoin エクスポージャーより有意に小さい

**数学的定式化:**
$$\text{Downside Risk}_{BMAX} = \kappa \cdot \text{Downside Risk}_{Bitcoin Direct} + \text{Bond Floor Effect} + v_t$$

ここで $\kappa < 1$ が下方リスク軽減効果を示す。

#### 仮説H4: ボラティリティ・レジーム依存性
**帰無仮説 $H_0$**: BMAX のリスク・リターン特性はボラティリティ・レジームに依存しない  
**対立仮説 $H_1$**: 高ボラティリティ期の BMAX パフォーマンスは低ボラティリティ期と有意に異なる

### 1.2 補助仮説

#### 仮説H5: 時変相関構造
Bitcoin-転換社債間の相関係数は時変的であり、市場ストレス時に相関が上昇する。

#### 仮説H6: 転換プレミアム決定要因
転換プレミアムは、ボラティリティ、金利、信用スプレッド、時間価値の関数として説明される。

## 2. データ収集・構築計画

### 2.1 データソースとアクセス

#### 2.1.1 価格・リターンデータ
- **BMAX ETF**: Yahoo Finance API, Bloomberg Terminal
- **Bitcoin**: CoinGecko API, CoinMarketCap API
- **構成銘柄**: Refinitiv Eikon, Alpha Vantage API
- **転換社債**: OptionMetrics, TRACE データベース

#### 2.1.2 高頻度データ
- **分次データ**: Interactive Brokers API
- **ティック・バイ・ティック**: Polygon.io API
- **オーダーブック**: Coinbase Pro API

#### 2.1.3 ファンダメンタル・データ
- **財務データ**: Compustat, FactSet
- **信用格付**: Moody's, S&P Global
- **ESG指標**: MSCI ESG Research

### 2.2 データクリーニング・前処理

#### 2.2.1 異常値検出
**Tukey's Fences:**
$$\text{Outlier} = \{x : x < Q_1 - 1.5 \times IQR \text{ or } x > Q_3 + 1.5 \times IQR\}$$

**Modified Z-Score:**
$$M_i = 0.6745 \frac{x_i - \tilde{x}}{MAD}$$

ここで $\tilde{x}$ は中央値、$MAD$ は中央絶対偏差。

#### 2.2.2 構造変化検出
**Bai-Perron テスト:**
$$y_t = \mathbf{x}_t^T \boldsymbol{\beta}_j + u_t, \quad T_{j-1} < t \leq T_j$$

**CUSUM テスト:**
$$CUSUM_t = \frac{1}{\hat{\sigma}} \sum_{i=1}^t \hat{u}_i$$

### 2.3 特徴量エンジニアリング

#### 2.3.1 技術指標
- **移動平均**: $MA_t(n) = \frac{1}{n} \sum_{i=0}^{n-1} P_{t-i}$
- **RSI**: $RSI_t = 100 - \frac{100}{1 + RS_t}$
- **ボリンジャーバンド**: $BB_t = MA_t \pm k \cdot \sigma_t(n)$

#### 2.3.2 市場微細構造指標
- **実現ボラティリティ**: $RV_t = \sum_{i=1}^n r_{t,i}^2$
- **ビッド・アスクスプレッド**: $Spread_t = \frac{Ask_t - Bid_t}{Mid_t}$
- **価格インパクト**: $\lambda_t = \frac{\Delta P_t}{V_t^{0.5}}$

#### 2.3.3 センチメント指標
- **VIX**: 恐怖指数
- **Bitcoin Fear & Greed Index**: 暗号資産市場センチメント
- **Social Media Sentiment**: Twitter, Reddit データ

## 3. 計量経済学的分析手法

### 3.1 時系列分析

#### 3.1.1 単位根・共和分検定
**Augmented Dickey-Fuller (ADF) Test:**
$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \epsilon_t$$

**Phillips-Perron (PP) Test:**
$$\Delta y_t = \alpha + \gamma y_{t-1} + \epsilon_t$$

**Johansen 共和分検定:**
$$\Delta \mathbf{y}_t = \boldsymbol{\alpha} \boldsymbol{\beta}' \mathbf{y}_{t-1} + \sum_{i=1}^{k-1} \boldsymbol{\Gamma}_i \Delta \mathbf{y}_{t-i} + \boldsymbol{\epsilon}_t$$

#### 3.1.2 VAR・VECM モデリング
**Vector Autoregression (VAR):**
$$\mathbf{y}_t = \boldsymbol{\nu} + \mathbf{A}_1 \mathbf{y}_{t-1} + \cdots + \mathbf{A}_p \mathbf{y}_{t-p} + \mathbf{u}_t$$

**Vector Error Correction Model (VECM):**
$$\Delta \mathbf{y}_t = \boldsymbol{\alpha} \boldsymbol{\beta}' \mathbf{y}_{t-1} + \sum_{i=1}^{p-1} \boldsymbol{\Gamma}_i \Delta \mathbf{y}_{t-i} + \boldsymbol{\epsilon}_t$$

### 3.2 非線形時系列分析

#### 3.2.1 閾値回帰モデル
**Threshold Autoregressive (TAR):**
$$y_t = \begin{cases}
\phi_1(L) y_t + \epsilon_{1t} & \text{if } y_{t-d} \leq \gamma \\
\phi_2(L) y_t + \epsilon_{2t} & \text{if } y_{t-d} > \gamma
\end{cases}$$

**Smooth Transition Autoregressive (STAR):**
$$y_t = \phi_1(L) y_t + [\phi_2(L) y_t] G(y_{t-d}; \gamma, c) + \epsilon_t$$

ここで $G(\cdot)$ は遷移関数。

#### 3.2.2 レジーム・スイッチング・モデル
**Markov Switching Model:**
$$y_t = \mu_{S_t} + \sum_{i=1}^{p} \phi_{i,S_t} y_{t-i} + \sigma_{S_t} \epsilon_t$$

遷移確率:
$$P_{ij} = \Pr(S_{t+1} = j | S_t = i)$$

### 3.3 GARCH族モデル

#### 3.3.1 多変量GARCH
**DCC-GARCH:**
$$H_t = D_t R_t D_t$$

ここで $D_t = \text{diag}(\sqrt{h_{11,t}}, \ldots, \sqrt{h_{nn,t}})$

動的条件付き相関:
$$R_t = Q_t^{-1/2} Q_t Q_t^{-1/2}$$

$$Q_t = (1-a-b)\bar{Q} + a \varepsilon_{t-1} \varepsilon_{t-1}' + b Q_{t-1}$$

#### 3.3.2 Copula-GARCH
$$C(u_1, u_2, \ldots, u_n; \theta) = \Pr(U_1 \leq u_1, \ldots, U_n \leq u_n)$$

**時変Copula:**
$$\theta_t = \omega + \alpha \cdot f(\varepsilon_{t-1}) + \beta \theta_{t-1}$$

## 4. 機械学習・AI手法

### 4.1 教師あり学習

#### 4.1.1 ランダムフォレスト
**価格予測モデル:**
$$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x})$$

**特徴量重要度:**
$$VI_j = \frac{1}{B} \sum_{b=1}^{B} \sum_{t \in OOB_b} \mathbb{I}(y_t \neq \hat{y}_{t,b}^{(j)}) - \mathbb{I}(y_t \neq \hat{y}_{t,b})$$

#### 4.1.2 勾配ブースティング
**XGBoost目的関数:**
$$\mathcal{L}(\phi) = \sum_{i=1}^n l(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)$$

**LightGBM特徴:**
- Leaf-wise tree growth
- Gradient-based One-Side Sampling (GOSS)
- Exclusive Feature Bundling (EFB)

### 4.2 深層学習

#### 4.2.1 LSTM時系列予測
**LSTM Cell:**
$$\begin{align}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \\
\tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \\
C_t &= f_t * C_{t-1} + i_t * \tilde{C}_t \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \\
h_t &= o_t * \tanh(C_t)
\end{align}$$

#### 4.2.2 Transformer Architecture
**Multi-Head Attention:**
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

### 4.3 教師なし学習

#### 4.3.1 クラスタリング
**K-means:**
$$\arg\min_{\mathbf{C}} \sum_{i=1}^k \sum_{\mathbf{x} \in C_i} \|\mathbf{x} - \boldsymbol{\mu}_i\|^2$$

**Gaussian Mixture Model:**
$$p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

#### 4.3.2 異常検知
**Isolation Forest:**
$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$

**One-Class SVM:**
$$\min_{\mathbf{w}, \xi, \rho} \frac{1}{2} \|\mathbf{w}\|^2 + \frac{1}{\nu n} \sum_{i=1}^n \xi_i - \rho$$

## 5. ベイズ統計・MCMC手法

### 5.1 ベイズ推定

#### 5.1.1 事前分布設定
**非情報的事前分布:**
$$\pi(\boldsymbol{\theta}) \propto 1$$

**共役事前分布:**
$$\pi(\mu, \sigma^2) = \pi(\mu | \sigma^2) \pi(\sigma^2)$$

#### 5.1.2 モデル選択
**DIC (Deviance Information Criterion):**
$$DIC = \bar{D}(\boldsymbol{\theta}) + p_D$$

**WAIC (Widely Applicable Information Criterion):**
$$WAIC = -2 \sum_{i=1}^n \log \left(\frac{1}{S} \sum_{s=1}^S p(y_i | \boldsymbol{\theta}^{(s)})\right) + 2 p_{WAIC}$$

### 5.2 MCMC サンプリング

#### 5.2.1 Gibbs Sampling
$$\boldsymbol{\theta}^{(t+1)} \sim p(\boldsymbol{\theta} | \mathbf{y}, \boldsymbol{\theta}_{-j}^{(t)})$$

#### 5.2.2 Hamiltonian Monte Carlo
**Hamilton's Equations:**
$$\frac{dq_i}{dt} = \frac{\partial H}{\partial p_i}, \quad \frac{dp_i}{dt} = -\frac{\partial H}{\partial q_i}$$

**NUTS (No-U-Turn Sampler):**
自動的なステップサイズ調整とツリー構築による効率的サンプリング

## 6. 実証分析設計

### 6.1 記述統計分析

#### 6.1.1 基本統計量
- 平均、分散、歪度、尖度
- 最大値、最小値、四分位数
- Jarque-Bera正規性検定

#### 6.1.2 分布適合度検定
**Kolmogorov-Smirnov Test:**
$$D_n = \sup_x |F_n(x) - F(x)|$$

**Anderson-Darling Test:**
$$A^2 = -n - \frac{1}{n} \sum_{i=1}^n (2i-1)[\ln F(X_i) + \ln(1-F(X_{n+1-i}))]$$

### 6.2 相関・因果関係分析

#### 6.2.1 線形・非線形相関
**Pearson相関係数:**
$$r = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2 \sum_{i=1}^n (y_i - \bar{y})^2}}$$

**Spearman順位相関:**
$$\rho = 1 - \frac{6 \sum_{i=1}^n d_i^2}{n(n^2-1)}$$

**Distance相関:**
$$dCor^2(X,Y) = \frac{dCov^2(X,Y)}{\sqrt{dVar(X) \cdot dVar(Y)}}$$

#### 6.2.2 Granger因果性
$$y_t = \alpha + \sum_{i=1}^{p} \beta_i y_{t-i} + \sum_{i=1}^{q} \gamma_i x_{t-i} + \epsilon_t$$

**検定統計量:**
$$F = \frac{(RSS_r - RSS_u)/q}{RSS_u/(T-k)} \sim F(q, T-k)$$

### 6.3 構造変化・レジーム検出

#### 6.3.1 Chow Test
$$F = \frac{(RSS_p - RSS_1 - RSS_2)/k}{(RSS_1 + RSS_2)/(n_1 + n_2 - 2k)}$$

#### 6.3.2 Quandt-Andrews Test
$$\sup_{T_1 \leq \tau \leq T_2} LR(\tau)$$

## 7. ロバスト性検証

### 7.1 交差検証

#### 7.1.1 時系列交差検証
**Time Series Split:**
```
Fold 1: train [1, 100], test [101, 120]
Fold 2: train [1, 120], test [121, 140]
...
Fold k: train [1, T-20], test [T-19, T]
```

#### 7.1.2 Blocked Cross-Validation
隣接時点間の依存性を考慮したブロック単位交差検証

### 7.2 ブートストラップ法

#### 7.2.1 Block Bootstrap
**Moving Block Bootstrap:**
$$B_i = \{X_{i}, X_{i+1}, \ldots, X_{i+l-1}\}$$

**Stationary Bootstrap:**
ランダムブロック長によるブートストラップ

#### 7.2.2 Wild Bootstrap
$$y_t^* = \hat{y}_t + \hat{\epsilon}_t \cdot \eta_t$$

ここで $\eta_t$ は独立な乱数。

### 7.3 感度分析

#### 7.3.1 パラメータ摂動分析
各パラメータを $\pm 10\%, \pm 20\%$ 変動させた場合の結果安定性

#### 7.3.2 サンプル期間分析
- 金融危機期間除外分析
- COVID-19期間別分析
- ビットコイン半減期分析

## 8. アウトオブサンプル予測検証

### 8.1 予測精度評価

#### 8.1.1 点予測評価
**Mean Absolute Error:**
$$MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

**Root Mean Square Error:**
$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

**Mean Absolute Percentage Error:**
$$MAPE = \frac{100}{n} \sum_{i=1}^n \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

#### 8.1.2 方向性予測評価
**Direction Accuracy:**
$$DA = \frac{1}{T} \sum_{t=1}^T \mathbb{I}(\text{sign}(\Delta y_t) = \text{sign}(\Delta \hat{y}_t))$$

### 8.2 統計的有意性検定

#### 8.2.1 Diebold-Mariano Test
$$DM = \frac{\bar{d}}{\sqrt{\hat{V}(\bar{d})}} \sim N(0,1)$$

ここで $d_t = L(e_{1t}) - L(e_{2t})$ は損失関数差分。

#### 8.2.2 Model Confidence Set (MCS)
$$\mathcal{M}^*_{1-\alpha} = \{i \in \mathcal{M} : \mu_i = \max_{j \in \mathcal{M}} \mu_j\}$$

## 9. 計算インフラ・実装計画

### 9.1 計算環境

#### 9.1.1 ハードウェア仕様
- **CPU**: AMD EPYC 7742 (64コア)
- **GPU**: NVIDIA A100 80GB × 4
- **メモリ**: 512GB DDR4
- **ストレージ**: 10TB NVMe SSD

#### 9.1.2 ソフトウェアスタック
- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.11+ (Anaconda)
- **R**: 4.3+ (RStudio Server)
- **Databases**: PostgreSQL, InfluxDB, Redis

### 9.2 分散処理フレームワーク

#### 9.2.1 Apache Spark
```python
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor

spark = SparkSession.builder\
    .appName("BMAX_Analysis")\
    .config("spark.executor.memory", "16g")\
    .config("spark.executor.cores", "4")\
    .getOrCreate()
```

#### 9.2.2 Dask
```python
import dask.dataframe as dd
from dask.distributed import Client

client = Client('scheduler-address:8786')
df = dd.read_parquet('bmax_data/*.parquet')
```

### 9.3 MLOps パイプライン

#### 9.3.1 MLflow
```python
import mlflow
import mlflow.sklearn

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, "random_forest")
```

#### 9.3.2 Kubeflow
```yaml
apiVersion: kubeflow.org/v1
kind: Notebook
metadata:
  name: bmax-research-notebook
spec:
  template:
    spec:
      containers:
      - name: notebook
        image: jupyter/scipy-notebook:latest
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
```

## 10. 予想される成果と貢献

### 10.1 学術的貢献

#### 10.1.1 理論的発展
1. **Bitcoin-転換社債価格理論**: デジタル資産連動ハイブリッド証券の新理論
2. **ETF流動性変換理論**: 非流動資産の ETF 化による流動性創出メカニズム
3. **三層相関構造理論**: Bitcoin → 株式 → 転換社債の複合相関モデル

#### 10.1.2 実証的知見
1. **非線形価格伝播の定量化**: 閾値効果とレジーム依存性の実証
2. **リスク軽減効果の測定**: 転換社債構造による下方リスク保護の定量評価
3. **最適ポートフォリオ配分**: リスク調整後リターン最大化配分の導出

### 10.2 実務的インパクト

#### 10.2.1 投資戦略開発
1. **新商品設計**: 類似 ETF 商品開発の理論的・実証的基盤
2. **リスク管理**: Bitcoin エクスポージャーの新ヘッジ手法
3. **資産配分**: 機関投資家向け最適配分モデル

#### 10.2.2 規制・政策含意
1. **リスク評価フレームワーク**: 新資産クラスのリスク評価基準
2. **透明性向上**: ETF 構造の複雑性に対する投資家理解促進
3. **市場安定性**: デジタル資産関連商品の市場への影響評価

## 11. リスク管理・品質保証

### 11.1 データ品質管理

#### 11.1.1 データ検証プロセス
```python
def validate_data_quality(df):
    checks = {
        'missing_values': df.isnull().sum(),
        'duplicates': df.duplicated().sum(),
        'outliers': detect_outliers(df),
        'data_types': df.dtypes,
        'date_consistency': validate_timestamps(df)
    }
    return checks
```

#### 11.1.2 データリネージ追跡
完全なデータ処理パイプラインの記録・追跡システム

### 11.2 モデル検証

#### 11.2.1 バックテスト検証
- Walk-forward analysis
- Monte Carlo validation
- Stress testing

#### 11.2.2 モデル診断
- 残差分析
- 多重共線性検定
- モデル安定性テスト

### 11.3 再現性保証

#### 11.3.1 版管理
```bash
git init bmax-research
git lfs track "*.parquet"
git lfs track "*.h5"
dvc init
dvc add data/
dvc push
```

#### 11.3.2 環境再現
```yaml
# environment.yml
name: bmax-research
dependencies:
  - python=3.11
  - pandas=2.0
  - numpy=1.24
  - scikit-learn=1.3
  - tensorflow=2.13
  - pytorch=2.0
```

## 12. 研究スケジュール

### 12.1 フェーズ別計画

| フェーズ | 期間 | 主要活動 | 成果物 |
|---------|------|----------|--------|
| **Phase 1** | 月1-3 | データ収集・前処理 | クリーンデータセット |
| **Phase 2** | 月4-6 | 記述統計・探索的分析 | 予備分析レポート |
| **Phase 3** | 月7-12 | 計量経済学的分析 | 中間研究報告書 |
| **Phase 4** | 月13-18 | 機械学習・AI分析 | 予測モデル群 |
| **Phase 5** | 月19-21 | 統合分析・ロバスト性検証 | 最終分析結果 |
| **Phase 6** | 月22-24 | 論文執筆・成果発表 | 学術論文・実装コード |

### 12.2 マイルストーン

#### 重要な中間成果
- **月6**: 予備分析完了・中間発表
- **月12**: 主要仮説検定完了
- **月18**: 機械学習モデル完成
- **月24**: 最終成果発表

---

この実証研究設計により、BMAX REX の理論的基盤と実証的エビデンスの両面から包括的な学術的貢献を実現し、デジタル資産時代における新しい金融商品の学術的理解を深化させることを目指す。