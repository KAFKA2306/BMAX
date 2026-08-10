# Convertible benchmark data contract

Issue #2 のデータ基盤を、表示やモデル計算から独立した fail-closed contract として定義する。

## 層の分離

`bmax.convertible-benchmark.v1` は次を別collectionで保持する。

- `issuers`: 発行体identity
- `issues`: 契約上の発行条件だけ
- `evidence`: SEC filing等の一次資料とsection reference
- `market_snapshots`: 観測時点の市場入力
- `scenarios`: model version、assumptions、model output

`issues` に market/model field を混在させない。`fair_value` と `recommendation` はscenario outputでも禁止する。

## Evidence gate

principal、issue date、maturity、coupon、currency、conversion price、conversion ratio、reference shareには既知の`evidence.id`が必要。coverage不足は自動補完せず、commercial demo readyをfalseにする。

call / put / redemption / fundamental-changeは以下の3状態だけを許可する。

- `PRESENT`: 一次資料で条項ありを確認しevidenceを持つ
- `ABSENT`: 一次資料で条項なしを確認しevidenceを持つ
- `UNVERIFIED`: 未確認。確認済みと見せるevidenceを付けない

`UNVERIFIED`を`ABSENT`へ推測変換しない。

## Commercial demo gate

validatorが`commercial_demo_ready=true`を返すのは、少なくとも10 issue / 5 issuer、全issueの主要field evidence coverage 100%、かつ保護条項まで確認済みのscenario-ready issueが3件以上ある場合だけ。

これはデータ品質gateであり、発行条件の推奨、公正価値、発行成功確率を意味しない。

## BMAXの扱い

BMAXを現役ETF監視商品として表示しない。SECの2026-03-31付supplementでは、Boardが2026-03-30にPlan of Liquidationを承認し、Fundは2026-04-21前後に清算予定と開示されている。

一次資料:

- https://www.sec.gov/Archives/edgar/data/1771146/000183988225015121/bmax-485bpos_031225.htm
- https://www.sec.gov/Archives/edgar/data/1771146/000177114626000689/etfot-497liquidationofrexb.htm

## 次のdata lane

このcontractへSEC filingから10 issue / 5 issuerを追加する。値はfiling本文で確認できたものだけを採用し、各主要fieldを`evidence_map`でsourceへ結ぶ。3件以上がscenario-readyになった後に既存`bmax_computational_framework.py`へのadapterとbenchmark UIを接続する。
