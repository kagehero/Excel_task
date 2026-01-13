# FBA注文処理システム

Amazon FBA注文の原価計算を行うシステムです。

## 📚 ドキュメント

- **[クイックスタートガイド（QUICK_START.md）](QUICK_START.md)** - 5分で始める簡潔なガイド
- **[ユーザーガイド（USER_GUIDE.md）](USER_GUIDE.md)** - 非開発者向けの詳細な使い方ガイド
- このREADME - 開発者向けの技術情報

## 🌐 デプロイ済みシステムを使用する場合

システムが既にデプロイされている場合は、以下のドキュメントを参照してください：

- 🚀 **[クイックスタートガイド](QUICK_START.md)** - すぐに始めたい方
- 📖 **[ユーザーガイド](USER_GUIDE.md)** - 詳しく知りたい方

## 必要なファイル

1. **Jancode.xlsx** - ASIN、SKU、商品名、寸法情報
2. **record-list.xls** - 注文番号、商品の金額（参考金額（JPY））
3. **send-order-list - XXXXXXX.xls** - 配送日、配送依頼No、注文番号、単価（ウォン）、数量、中国国内送料（ウォン）、ASIN、顧客管理番号
4. **FBA指示書XXXXXX.xls** - 配送依頼No、納品プランNo、追跡番号を含む指示書
5. **manual_input.txt** (オプション) - 手動入力データ（オプション費用、割引額、関税、消費税）
6. **option_cost_distribution.txt** (オプション) - オプション費用の分配情報

## インストール

### 仮想環境を使用する場合（推奨）

```bash
# 仮想環境を作成
python3 -m venv .venv

# 仮想環境をアクティブ化
source .venv/bin/activate

# 依存関係をインストール
pip install -r requirements.txt
```

### システムのPythonを使用する場合

```bash
pip install -r requirements.txt
```

## 使用方法

### GUIアプリケーション（推奨）

1. 依存関係をインストール（上記参照）

2. アプリケーションを起動:

**仮想環境を使用している場合:**
```bash
# 方法1: スクリプトを使用（推奨）
./run_app.sh

# 方法2: 仮想環境をアクティブ化してから実行
source .venv/bin/activate
streamlit run app.py
```

**仮想環境を使用しない場合:**
```bash
streamlit run app.py
# または
python3 -m streamlit run app.py
```

3. ブラウザで `http://localhost:8501` を開く

4. 「ファイル選択」タブで必要なファイルをアップロード:
   - **必須ファイル**: FBA指示書、send-order-list、record-list
   - **オプションファイル**: Jancode.xlsx、輸入許可通知書(PDF)、手動入力データ、オプション費用分配情報

5. 「処理実行」タブで処理を開始

6. 「結果表示」タブで結果を確認し、Excelファイルをダウンロード

### コマンドライン版

1. 必要なExcelファイルを同じディレクトリに配置
2. 手動入力データがある場合は `manual_input.txt` と `option_cost_distribution.txt` を作成
3. スクリプトを実行:

```bash
python fba_processor.py
```

4. 結果は `output.xlsx` に出力されます

## 出力項目

- ASIN
- SKU
- 商品名
- 原価(円)
- 数量
- 購入単価（ウォン）
- 中国国内送料（円）
- 商品1個あたりのオプション費用（ウォン）
- 商品1個につき割引額（ウォン）
- 商品1個あたり国際送料（ウォン）
- 商品1個あたり関税（円）
- 商品1個あたり消費税(円)
- 配送依頼No.
- 納品プランNo.
- 追跡番号
- 発送日
- 配送代行手数料（円）
- サイズ区分

## 例外処理

### 例外処理1
指示に記載されているが send-order-list にない製品
→ 該当する注文番号を含む send-order-list をリクエストしてください

### 例外処理2
send-order-list にあるが指示書にない商品（または数量0の商品）
→ 計算不要、計算しないASINと理由を最後に表示

### 例外処理3
指示書にもsend-order-listにも注文番号はあるが、ASINや数量が異なる場合
→ 指示書の内容に合わせて計算

### 例外処理4
電子メールから取得した情報のオプション費用の総額と、指示書のオプション費用の総額がずれる場合
→ `option_cost_distribution.txt` で分配を指定

## 特殊ケース

- **ASIN B0D7PXG7KL**: 注文数量が2個の場合、実際は40個（20個セット×2）

## 為替レート

前日のTTS + 0.5 を使用
https://www.murc-kawasesouba.jp/fx/past_3month.php から取得

## エラー処理

必要なデータがない場合、何が不足しているかを表示します。

