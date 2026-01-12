#!/bin/bash
# FBA注文処理システムを起動するスクリプト

# スクリプトのディレクトリに移動
cd "$(dirname "$0")"

# 仮想環境のパス
VENV_PATH=".venv"

# 仮想環境が存在するか確認
if [ -d "$VENV_PATH" ]; then
    echo "仮想環境を使用します..."
    # 仮想環境のPythonを使ってstreamlitを実行
    "$VENV_PATH/bin/python" -m streamlit run app.py --server.port 8501 --server.address localhost
else
    echo "仮想環境が見つかりません。システムのPythonを使用します..."
    # 仮想環境がない場合はシステムのPythonを使用
    python3 -m streamlit run app.py --server.port 8501 --server.address localhost
fi

