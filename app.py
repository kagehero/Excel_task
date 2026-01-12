#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FBA注文処理システム - GUIアプリケーション
StreamlitベースのWebインターフェース
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path
import json
from fba_processor import FBAProcessor
import tempfile
import shutil
import re
from typing import Dict, List, Optional, Tuple
from io import BytesIO, StringIO
import PyPDF2
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("pdfplumber not available, using PyPDF2 only")

# ページ設定
st.set_page_config(
    page_title="FBA注文処理システム",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2f6 0%, #ffffff 100%);
        border-radius: 10px;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3498db;
    }
    .info-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #e8f4f8;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .file-info {
        background-color: #f8f9fa;
        padding: 0.75rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-family: monospace;
    }
    .data-flow {
        display: flex;
        align-items: center;
        margin: 1rem 0;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 8px;
    }
    .arrow {
        font-size: 2rem;
        color: #3498db;
        margin: 0 1rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """セッション状態の初期化"""
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = None
    if 'file_info' not in st.session_state:
        st.session_state.file_info = {}
    if 'metadata' not in st.session_state:
        st.session_state.metadata = {
            'shipping_request_no': '',
            'delivery_plan_no': '',
            'tracking_number': '',
            'shipping_date': ''
        }

def save_uploaded_file(uploaded_file, file_type: str):
    """アップロードされたファイルを一時保存"""
    if uploaded_file is not None:
        # 一時ディレクトリを作成
        temp_dir = Path(tempfile.gettempdir()) / "fba_processor"
        temp_dir.mkdir(exist_ok=True)
        
        # ファイルを保存
        file_path = temp_dir / f"{file_type}_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return str(file_path)
    return None


def parse_fba_filename_metadata(filename: str) -> Dict[str, str]:
    """FBA指示書ファイル名から情報を抽出"""
    pattern = r'FBA指(?:令|示|示書)\s*\d{8}-\d{4}\s+(\d+)\s+([A-Z0-9]+)\s+([A-Z0-9]+)'
    match = re.search(pattern, filename)
    if not match:
        pattern_generic = r'FBA指[^\d]*(\d+)\s+([A-Z0-9]+)\s+([A-Z0-9]+)'
        match = re.search(pattern_generic, filename)
    if match:
        return {
            'shipping_request_no': match.group(1),
            'delivery_plan_no': match.group(2),
            'tracking_number': match.group(3)
        }
    return {}


def parse_send_order_filename(filename: str) -> str:
    """send-order-listファイル名から配送日を抽出"""
    pattern = r'send-order-list\s*-\s*(\d{4}-\d{2}-\d{2})'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return ""


PREVIEW_SYNONYMS = {
    'fba': ['ASIN', '数量', '個数', '注文番号', 'order', 'SKU'],
    'send_order': ['注文番号', '注文ID', 'ASIN', '数量', '個数', '単価', '中国国内送料', '顧客管理番号'],
    'record_list': ['注文番号', '参考金額', 'JPY', '金額', 'order'],
    'jancode': ['ASIN', 'SKU', '商品名', '長さ', '幅', '高さ', '寸法']
}


def pick_preview_columns(df: pd.DataFrame, synonyms: list) -> list:
    """表示したい列を候補から絞る"""
    selected = []
    lower_map = {col.lower(): col for col in df.columns}
    # 厳密な一致
    for term in synonyms:
        key = term.lower()
        if key in lower_map and lower_map[key] not in selected:
            selected.append(lower_map[key])
    # 包含マッチ
    if not selected:
        for term in synonyms:
            key = term.lower()
            for col in df.columns:
                if key in col.lower() and col not in selected:
                    selected.append(col)
    return selected[:5] if selected else list(df.columns[: min(4, len(df.columns))])


def load_table_with_html_fallback(filepath: str, nrows: Optional[int] = None) -> Tuple[pd.DataFrame, str]:
    """Excelとして読めない場合、HTMLテーブルとして再試行"""
    with open(filepath, 'rb') as f:
        raw = f.read()
    header = raw[:200].lower()
    if b'<html' in header or b'<!doctype html' in header:
        text = raw.decode('utf-8', errors='replace')
        tables = pd.read_html(StringIO(text))
        if tables:
            df = tables[0]
            if nrows:
                df = df.head(nrows)
            return df, 'html'
        raise Exception("HTMLテーブルを解析できませんでした")
    last_error = None
    for engine in ['openpyxl', 'xlrd']:
        try:
            df = pd.read_excel(BytesIO(raw), engine=engine, nrows=nrows)
            return df, 'excel'
        except Exception as exc:
            last_error = exc
            continue
    text = raw.decode('utf-8', errors='replace')
    tables = pd.read_html(StringIO(text))
    if tables:
        df = tables[0]
        if nrows:
            df = df.head(nrows)
        return df, 'html'
    raise last_error or Exception("ファイルが読み込めませんでした")


def find_matching_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """指定された候補から存在する列名を探す"""
    lower_map = {}
    for col in df.columns:
        key = str(col).lower()
        lower_map[key] = col
    for candidate in candidates:
        key = candidate.lower()
        if key in lower_map:
            return lower_map[key]
    for candidate in candidates:
        key = candidate.lower()
        for col in df.columns:
            if key in str(col).lower():
                return col
    return None


def display_data_preview(file_type: str, file_path: str, show_header: bool = True):
    """必要な範囲のみを表示する"""
    if show_header:
        st.subheader(f"📄 {file_type} のプレビュー")
    ext = Path(file_path).suffix.lower()
    if ext == '.pdf':
        st.info("PDFはプレビューできません。必要な情報は元ファイルでご確認ください。")
        return
    try:
        df, source = load_table_with_html_fallback(file_path, nrows=5)
        synonyms = PREVIEW_SYNONYMS.get(file_type, [])
        preview_cols = pick_preview_columns(df, synonyms)
        preview_df = df[preview_cols].copy()
        st.dataframe(preview_df, width='stretch')
        caption = f"表示列: {', '.join(preview_cols)} / 元のファイル行数: {len(df)}件（最大5行）"
        if source == 'html':
            caption += "（HTMLテーブルとして読み込みました）"
        st.caption(caption)
    except Exception as e:
        st.error(f"{file_type} の読み込みエラー: {str(e)}")


def display_metadata_header(metadata: Dict[str, str]):
    """配送情報を上部に表示"""
    if not any(metadata.values()):
        return
    st.markdown('<div class="section-header">🚚 配送情報</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    display_fields = [
        ("配送依頼番号", metadata.get('shipping_request_no', '未設定')),
        ("納品プランNo.", metadata.get('delivery_plan_no', '未設定')),
        ("追跡番号", metadata.get('tracking_number', '未設定')),
        ("発送日", metadata.get('shipping_date', '未設定')),
    ]
    for col, (label, value) in zip(cols, display_fields):
        col.markdown(f"**{label}**")
        col.caption(value)


RESULT_COLUMNS_PRIMARY: List[str] = [
    'ASIN',
    'SKU',
    '商品名',
    '原価(円)',
    '数量',
    '購入単価（元）',
    '中国国内送料（元）'
]


def promote_header_row(df: pd.DataFrame, keywords: List[str]) -> pd.DataFrame:
    """最初のヘッダー行が空の場合、キーワードを含む行をヘッダーに昇格させる"""
    for idx, row in df.reset_index(drop=True).iterrows():
        row_str = ' '.join([str(val) for val in row if pd.notna(val)])
        if any(key.lower() in row_str.lower() for key in keywords):
            new_df = df.iloc[idx:].copy()
            new_df.columns = new_df.iloc[0]
            new_df = new_df[1:]
            new_df = new_df.loc[:, new_df.columns.notna()]
            return new_df.reset_index(drop=True)
    return df


def _build_asin_subset(file_path: str, asin_candidates: List[str], target_groups: List[List[str]], asins: List[str]) -> pd.DataFrame:
    try:
        df, _ = load_table_with_html_fallback(file_path)
        df.columns = df.columns.str.strip()
        header_keywords = asin_candidates + ['顧客', '顧客管理番号', '注文', 'order', 'customer']
        df = promote_header_row(df, header_keywords)
        asin_col = find_matching_column(df, asin_candidates)
        if not asin_col:
            return pd.DataFrame()
        subset = df[df[asin_col].astype(str).isin([str(a) for a in asins])]
        available_cols = []
        for group in target_groups:
            col = find_matching_column(subset, group)
            if col:
                available_cols.append(col)
        columns = [asin_col] + available_cols
        seen_cols = []
        for col in columns:
            if col not in seen_cols:
                seen_cols.append(col)
        return subset[seen_cols].copy().reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def get_instruction_summary(file_path: str) -> pd.DataFrame:
    """FBA指示書のASIN・商品名・数量・オプション費用を抽出"""
    
    # ログファイルを準備
    import time
    from datetime import datetime
    log_file = "/tmp/fba_instruction_debug.log"
    
    def log_print(msg):
        """コンソールとファイルの両方に出力"""
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
    
    try:
        current_time = time.time()
        timestamp = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
        
        log_print(f"\n{'='*80}")
        log_print(f"=== Loading FBA instruction sheet ===")
        log_print(f"Timestamp: {timestamp}")
        log_print(f"File path: {file_path}")
        log_print(f"{'='*80}")
        
        df, _ = load_table_with_html_fallback(file_path)
        log_print(f"\nRaw data shape: {df.shape}")
        log_print(f"Raw columns ({len(df.columns)} columns):")
        for i, col in enumerate(df.columns):
            log_print(f"  [{i:2d}] {col}")
        df.columns = df.columns.str.strip()
        
        # promote_header_row前にASIN列を見つけてオプション費用列を特定
        df_before_promote = df.copy()
        
        # promote_header_rowを実行してASIN列を見つける（一時的）
        df_temp = promote_header_row(df_before_promote.copy(), ['ASIN', 'ASIN番号', 'ASIN/ISBN'])
        asin_col_temp = find_matching_column(df_temp, ['ASIN', 'asin', 'ASIN番号'])
        
        option_col_before = None
        option_cost_data = None
        
        if asin_col_temp:
            # ASINのサンプル値を取得
            asin_samples = df_temp[asin_col_temp].dropna().head(5).tolist()
            log_print(f"\nASINサンプル値: {asin_samples}")
            
            # 元のデータでASINを含む列を探す
            asin_idx_before = None
            for idx, col in enumerate(df_before_promote.columns):
                col_values = df_before_promote[col].astype(str).tolist()[:100]  # 最初の100行
                # ASINサンプルのいずれかが含まれているか確認
                if any(asin_sample in col_values for asin_sample in asin_samples if asin_sample):
                    asin_idx_before = idx
                    log_print(f"\npromote_header_row前のASIN列を発見: インデックス{idx}")
                    break
            
            if asin_idx_before is not None:
                option_idx_before = asin_idx_before + 3
                
                if option_idx_before < len(df_before_promote.columns):
                    asin_col_before = df_before_promote.columns[asin_idx_before]
                    option_col_before = df_before_promote.columns[option_idx_before]
                    
                    log_print(f"\npromote_header_row前のオプション費用列検出:")
                    log_print(f"  ASIN列: [{asin_idx_before}] '{asin_col_before}'")
                    log_print(f"  オプション費用列: [{option_idx_before}] '{option_col_before}'")
                    sample_values = df_before_promote[option_col_before].dropna().head(10).tolist()
                    log_print(f"  サンプル値: {sample_values}")
                    
                    # オプション費用列を保存
                    option_cost_data = df_before_promote[[asin_col_before, option_col_before]].copy()
                    option_cost_data.columns = ['ASIN_temp', 'オプション費用_temp']
        
        df = promote_header_row(df, ['ASIN', 'ASIN番号', 'ASIN/ISBN'])
        log_print(f"\nAfter promote_header_row:")
        log_print(f"  Shape: {df.shape}")
        log_print(f"  Columns ({len(df.columns)} columns):")
        for i, col in enumerate(df.columns):
            log_print(f"    [{i:2d}] {col}")
        
        asin_col = find_matching_column(df, ['ASIN', 'asin', 'ASIN番号'])
        product_col = find_matching_column(df, ['商品名', '商品名称', 'product_name', '商品'])
        qty_col = find_matching_column(df, ['数量', '個数', 'qty', '数量(個)', '納品個数'])
        
        # promote_header_row前に保存したオプション費用データがあれば使用
        option_col = None
        use_saved_option_data = False
        if option_col_before and asin_col:
            log_print(f"\n保存済みのオプション費用データを使用")
            use_saved_option_data = True
        
        log_print(f"\n検出された列:")
        log_print(f"  ASIN列: {asin_col}")
        log_print(f"  商品名列: {product_col}")
        log_print(f"  数量列: {qty_col}")
        log_print(f"  オプション費用列: {option_col}")
        
        # すべての列を表示（デバッグ用）
        log_print(f"\n全列リスト（インデックス:列名:サンプル値）:")
        for idx, col in enumerate(df.columns):
            # 非NaN値のサンプルを取得
            sample_values = df[col].dropna().head(3).tolist()
            log_print(f"  [{idx:2d}] '{col}' → サンプル: {sample_values}")
        
        if not asin_col or not qty_col:
            return pd.DataFrame()
        
        # 必須列をコピー
        cols_to_copy = [asin_col, qty_col]
        if product_col:
            cols_to_copy.append(product_col)
        
        log_print(f"\nコピーする列: {cols_to_copy}")
        summary = df[cols_to_copy].copy()
        
        # 保存したオプション費用データをマージ
        if use_saved_option_data:
            log_print(f"\nオプション費用データをマージ中...")
            # 数値変換
            option_cost_data['オプション費用_temp'] = pd.to_numeric(option_cost_data['オプション費用_temp'], errors='coerce').fillna(0)
            log_print(f"  オプション費用データ shape: {option_cost_data.shape}")
            log_print(f"  オプション費用合計: {option_cost_data['オプション費用_temp'].sum()}元")
            
            # ASINでマージ
            summary = summary.merge(
                option_cost_data,
                left_on=asin_col,
                right_on='ASIN_temp',
                how='left'
            )
            summary = summary.drop(columns=['ASIN_temp'], errors='ignore')
            summary = summary.rename(columns={'オプション費用_temp': 'オプション費用（元）_raw'})
            log_print(f"  マージ後 shape: {summary.shape}")
            log_print(f"  マージ後 columns: {summary.columns.tolist()}")
        else:
            log_print(f"\n⚠️ オプション費用列が見つかりませんでした！")
        
        log_print(f"\nリネーム前のデータ（最初の5行）:")
        log_print(f"  Columns: {summary.columns.tolist()}")
        log_print(f"  Shape: {summary.shape}")
        for i, row in summary.head().iterrows():
            log_print(f"  Row {i}: {row.to_dict()}")
        
        rename_map = {asin_col: 'ASIN', qty_col: '数量'}
        if product_col:
            rename_map[product_col] = '商品名'
        if 'オプション費用（元）_raw' in summary.columns:
            rename_map['オプション費用（元）_raw'] = 'オプション費用（元）'
        summary = summary.rename(columns=rename_map)
        summary = summary[pd.notna(summary['ASIN'])]
        
        # 数量型に変換
        summary['数量'] = pd.to_numeric(summary['数量'], errors='coerce').fillna(0)
        
        # オプション費用は既に数値変換済み
        if 'オプション費用（元）' in summary.columns:
            log_print(f"\n数値変換後のオプション費用（元）:")
            log_print(f"  合計: {summary['オプション費用（元）'].sum()}元")
            log_print(f"  行数: {len(summary)}行")
            log_print(f"  非ゼロ行数: {(summary['オプション費用（元）'] > 0).sum()}行")
            log_print(f"  最小値: {summary['オプション費用（元）'].min()}元")
            log_print(f"  最大値: {summary['オプション費用（元）'].max()}元")
        
        log_print(f"\nBefore groupby (rows with B0BKFS9N54):")
        b0bkfs9n54_rows = summary[summary['ASIN'] == 'B0BKFS9N54']
        if not b0bkfs9n54_rows.empty:
            for i, row in b0bkfs9n54_rows.iterrows():
                log_print(f"  {row.to_dict()}")
        
        # オプション費用の合計（groupby前）
        if 'オプション費用（元）' in summary.columns:
            option_total_before = summary['オプション費用（元）'].sum()
            log_print(f"\n=== オプション費用合計（groupby前・全行）: {option_total_before}元 ===")
            log_print(f"総行数: {len(summary)}行")
        
        # groupby時にオプション費用も含めて集計
        agg_cols = ['ASIN', '商品名'] if '商品名' in summary.columns else ['ASIN']
        agg_dict = {'数量': 'sum'}
        if 'オプション費用（元）' in summary.columns:
            agg_dict['オプション費用（元）'] = 'sum'  # オプション費用も合計
        
        summary = summary.groupby(agg_cols, dropna=False, as_index=False).agg(agg_dict)
        summary['数量'] = summary['数量'].fillna(0).infer_objects(copy=False)
        if 'オプション費用（元）' in summary.columns:
            summary['オプション費用（元）'] = summary['オプション費用（元）'].fillna(0).infer_objects(copy=False)
            option_total_after = summary['オプション費用（元）'].sum()
            log_print(f"\n=== オプション費用合計（groupby後・ASIN単位）: {option_total_after}元 ===")
            log_print(f"ASIN数: {len(summary)}個")
        
        log_print(f"\nAfter groupby (B0BKFS9N54):")
        b0bkfs9n54_final = summary[summary['ASIN'] == 'B0BKFS9N54']
        if not b0bkfs9n54_final.empty:
            for i, row in b0bkfs9n54_final.iterrows():
                log_print(f"  {row.to_dict()}")
        
        # すべての列を返す
        return_cols = [col for col in ['商品名', 'ASIN', '数量', 'オプション費用（元）'] if col in summary.columns]
        log_print(f"\n返す列: {return_cols}")
        log_print(f"最終データフレーム形状: {summary[return_cols].shape}")
        log_print(f"{'='*80}\n")
        
        return summary[return_cols]
    except Exception as e:
        log_print(f"\n❌ Error in get_instruction_summary: {e}")
        import traceback
        error_trace = traceback.format_exc()
        log_print(f"Traceback:\n{error_trace}")
        log_print(f"{'='*80}\n")
        return pd.DataFrame()


def get_discount_from_instruction_sheet2(file_path: str) -> pd.DataFrame:
    """FBA指示書のシート2から注文番号と割引額を取得"""
    try:
        print(f"\n=== Loading discount data from sheet2 ===")
        print(f"File path: {file_path}")
        
        # シート2を読み込み
        df = pd.read_excel(file_path, sheet_name=1, engine='xlrd')  # sheet_name=1 は2番目のシート
        print(f"Raw sheet2 shape: {df.shape}")
        print(f"Raw columns: {list(df.columns)}")
        print(f"First 3 rows:\n{df.head(3)}")
        
        df.columns = df.columns.str.strip()
        
        # ヘッダー行を探す
        df = promote_header_row(df, ['注文番号', 'order', '割引', 'discount'])
        print(f"After promote_header_row shape: {df.shape}")
        print(f"After promote_header_row columns: {list(df.columns)}")
        
        # 注文番号と割引額の列を探す
        order_col = find_matching_column(df, ['注文番号', '注文ID', 'order_no', 'order number', 'オーダー番号', '订单号'])
        discount_col = find_matching_column(df, ['割引額', '割引', 'discount', 'ディスカウント', '砍价', '値引き'])
        
        print(f"Found order_col: {order_col}")
        print(f"Found discount_col: {discount_col}")
        
        if not order_col or not discount_col:
            print(f"ERROR: Required columns not found!")
            return pd.DataFrame()
        
        result = df[[order_col, discount_col]].copy()
        result = result.rename(columns={order_col: '注文番号', discount_col: '割引額（元）'})
        
        print(f"Before numeric conversion:\n{result.head()}")
        
        # 注文番号を文字列に変換（小数点を削除）
        result['注文番号'] = result['注文番号'].apply(lambda x: str(int(float(x))) if pd.notna(x) else '')
        
        # 数値に変換
        result['割引額（元）'] = pd.to_numeric(result['割引額（元）'], errors='coerce')
        
        # NaNを除外
        result = result[pd.notna(result['注文番号']) & (result['注文番号'] != '') & pd.notna(result['割引額（元）'])]
        
        print(f"Discount data loaded: {len(result)} rows")
        print(f"Final result:\n{result.head(10)}")
        print(f"Sample order numbers: {result['注文番号'].head(10).tolist()}")
        
        return result
    except Exception as e:
        print(f"Error loading sheet2: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def extract_tax_from_pdf_pdfplumber(file_path: str) -> Dict[str, Optional[float]]:
    """pdfplumber を使って PDF からテーブルを抽出（より高精度）"""
    if not PDFPLUMBER_AVAILABLE:
        return {'関税': None, '消費税': None, '地方消費税': None}
    
    try:
        result = {
            '関税': None,
            '消費税': None,
            '地方消費税': None
        }
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # テーブルを抽出
                tables = page.extract_tables()
                print(f"Page {page_num}: Found {len(tables)} tables")
                
                for table_num, table in enumerate(tables):
                    if not table:
                        continue
                    
                    print(f"  Table {table_num}: {len(table)} rows")
                    
                    # 各行を確認
                    for row_num, row in enumerate(table):
                        if not row:
                            continue
                        
                        row_str = ' '.join([str(cell) if cell else '' for cell in row])
                        
                        # 関税を探す
                        if '関税' in row_str and result['関税'] is None:
                            for cell in row:
                                if cell:
                                    amounts = re.findall(r'[\\¥￥]\s*([0-9,]+)', str(cell))
                                    if amounts:
                                        try:
                                            result['関税'] = float(amounts[0].replace(',', ''))
                                            print(f"    Found 関税 in row {row_num}: {row} -> {result['関税']}")
                                            break
                                        except ValueError:
                                            pass
                        
                        # 消費税を探す
                        if '消費税' in row_str and '地方' not in row_str and result['消費税'] is None:
                            for cell in row:
                                if cell:
                                    amounts = re.findall(r'[\\¥￥]\s*([0-9,]+)', str(cell))
                                    if amounts:
                                        try:
                                            result['消費税'] = float(amounts[0].replace(',', ''))
                                            print(f"    Found 消費税 in row {row_num}: {row} -> {result['消費税']}")
                                            break
                                        except ValueError:
                                            pass
                        
                        # 地方消費税を探す
                        if '地方消費税' in row_str and result['地方消費税'] is None:
                            for cell in row:
                                if cell:
                                    amounts = re.findall(r'[\\¥￥]\s*([0-9,]+)', str(cell))
                                    if amounts:
                                        try:
                                            result['地方消費税'] = float(amounts[0].replace(',', ''))
                                            print(f"    Found 地方消費税 in row {row_num}: {row} -> {result['地方消費税']}")
                                            break
                                        except ValueError:
                                            pass
        
        return result
        
    except Exception as e:
        print(f"pdfplumber extraction error: {e}")
        import traceback
        print(traceback.format_exc())
        return {'関税': None, '消費税': None, '地方消費税': None}


def extract_tax_from_pdf(file_path: str) -> Dict[str, Optional[float]]:
    """PDF から関税、消費税、地方消費税を抽出
    
    PDFの実際の形式:
    - 税種名と金額が別の列に分かれている
    - 「税科目 税額合計」のような見出しがある
    - 金額は \3,800 のような形式
    
    まず pdfplumber を試し、失敗したら PyPDF2 を使用
    """
    # まず pdfplumber を試す（より高精度）
    if PDFPLUMBER_AVAILABLE:
        print("Trying pdfplumber first...")
        result = extract_tax_from_pdf_pdfplumber(file_path)
        if all(v is not None for v in result.values()):
            print("Successfully extracted all values with pdfplumber")
            return result
        else:
            print(f"pdfplumber result incomplete: {result}")
            print("Falling back to PyPDF2...")
    
    # PyPDF2 でテキスト抽出
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
        
        print(f"PDF text extracted, length: {len(text)}")
        
        # 関税、消費税、地方消費税を探す
        result = {
            '関税': None,
            '消費税': None,
            '地方消費税': None
        }
        
        # テキストを行ごとに分割
        lines = text.split('\n')
        print(f"PDF lines: {len(lines)}")
        
        # 「税科目 税額合計」のような見出しを探す
        tax_table_start = None
        for i, line in enumerate(lines):
            if '税科目' in line and '税額合計' in line:
                tax_table_start = i
                print(f"Found tax table header at line {i}: {line}")
                break
        
        # 税科目セクションから金額を探す
        if tax_table_start:
            # 見出しの後、数行以内に税種と金額がある
            for offset in range(1, 50):  # 見出しから50行先まで探す
                if tax_table_start + offset >= len(lines):
                    break
                    
                line = lines[tax_table_start + offset]
                line_clean = line.strip()
                
                # 金額パターン: \数字,数字 または ¥数字,数字
                amounts = re.findall(r'[\\¥]\s*([0-9,]+)', line_clean)
                
                # 各税種を確認
                if '関税' in line_clean and result['関税'] is None and amounts:
                    try:
                        result['関税'] = float(amounts[0].replace(',', ''))
                        print(f"Found 関税 at line {tax_table_start + offset}: {line_clean} -> {result['関税']}")
                    except (ValueError, IndexError):
                        pass
                
                if '消費税' in line_clean and '地方' not in line_clean and result['消費税'] is None and amounts:
                    try:
                        result['消費税'] = float(amounts[0].replace(',', ''))
                        print(f"Found 消費税 at line {tax_table_start + offset}: {line_clean} -> {result['消費税']}")
                    except (ValueError, IndexError):
                        pass
                
                if '地方消費税' in line_clean and result['地方消費税'] is None and amounts:
                    try:
                        result['地方消費税'] = float(amounts[0].replace(',', ''))
                        print(f"Found 地方消費税 at line {tax_table_start + offset}: {line_clean} -> {result['地方消費税']}")
                    except (ValueError, IndexError):
                        pass
        
        # 見つからない場合、全体から探す
        for key in ['関税', '消費税', '地方消費税']:
            if result[key] is not None:
                continue
            
            for i, line in enumerate(lines):
                if key in line:
                    # この行と次の数行から金額を探す
                    for offset in range(0, 3):
                        if i + offset >= len(lines):
                            break
                        search_line = lines[i + offset]
                        amounts = re.findall(r'[\\¥]\s*([0-9,]+)', search_line)
                        if amounts:
                            try:
                                result[key] = float(amounts[0].replace(',', ''))
                                print(f"Found {key} at line {i+offset}: {search_line[:100]} -> {result[key]}")
                                break
                            except (ValueError, IndexError):
                                continue
                    if result[key] is not None:
                        break
        
        # デバッグ情報
        if any(v is None for v in result.values()):
            print("Some values not found. Showing relevant lines:")
            for i, line in enumerate(lines):
                if any(keyword in line for keyword in ['税科目', '税額', '関税', '消費税', '地方']):
                    print(f"  Line {i}: {line[:150]}")
        
        return result
        
    except Exception as e:
        print(f"PDF extraction error: {e}")
        import traceback
        print(traceback.format_exc())
        return {'関税': None, '消費税': None, '地方消費税': None}


def get_jancode_preview(asins: List[str], file_path: str) -> pd.DataFrame:
    """Jancode.xlsx から ASIN、SKU、寸法情報（長さ、幅、高さ、合計）を取得"""
    try:
        # まず header=None で読み込んで実際の列数を確認
        try:
            if file_path.endswith('.xlsx'):
                raw_no_header = pd.read_excel(file_path, header=None, engine='openpyxl')
            else:
                raw_no_header = pd.read_excel(file_path, header=None)
            print(f"Raw file shape (no header): {raw_no_header.shape}")
            print(f"First row (likely header): {list(raw_no_header.iloc[0])}")
            print(f"Second row (sample data): {list(raw_no_header.iloc[1]) if len(raw_no_header) > 1 else 'N/A'}")
        except Exception as e:
            print(f"Could not read with header=None: {e}")
            raw_no_header = None
        
        # header=None で読み込んで、手動でヘッダー行を探す
        if file_path.endswith('.xlsx'):
            df_raw = pd.read_excel(file_path, header=None, engine='openpyxl')
        else:
            df_raw = pd.read_excel(file_path, header=None)
        
        print(f"Loaded raw Jancode: {df_raw.shape}")
        
        # ヘッダー行を探す（'ASIN' を含む行）
        header_row_idx = None
        for idx, row in df_raw.iterrows():
            row_str = ' '.join([str(val) for val in row if pd.notna(val)])
            if 'ASIN' in row_str:
                header_row_idx = idx
                print(f"Found header row at index {idx}")
                print(f"Header row values (full): {list(row)}")
                # 前後の行も確認
                if idx > 0:
                    print(f"Previous row (index {idx-1}): {list(df_raw.iloc[idx-1])}")
                if idx < len(df_raw) - 1:
                    print(f"Next row (index {idx+1}, data sample): {list(df_raw.iloc[idx+1])[:15]}")
                break
        
        if header_row_idx is None:
            print("Header row not found")
            return pd.DataFrame()
        
        # ヘッダー行を設定
        df = df_raw.iloc[header_row_idx + 1:].copy()
        df.columns = df_raw.iloc[header_row_idx]
        df = df.reset_index(drop=True)
        
        # 列名をクリーン化
        df.columns = [str(col).strip() if pd.notna(col) else f'Unnamed_{i}' for i, col in enumerate(df.columns)]
        
        print(f"Columns after setting header: {list(df.columns)}")
        print(f"DataFrame shape: {df.shape}")
        
        # ASIN 列を検出
        asin_col = None
        for col in df.columns:
            if 'ASIN' in str(col).upper():
                asin_col = col
                break
        
        if not asin_col:
            print("ASIN column not found in Jancode")
            return pd.DataFrame()
        
        print(f"ASIN column found: {asin_col}")
        
        # 指定された ASIN でフィルタ
        df[asin_col] = df[asin_col].astype(str).str.strip()
        subset = df[df[asin_col].isin([str(a).strip() for a in asins])].copy()
        
        print(f"Filtered {len(subset)} rows from {len(df)} total rows")
        
        if subset.empty:
            return pd.DataFrame()
        
        # 全ての列名をデバッグ出力
        print(f"All columns ({len(subset.columns)}): {list(subset.columns)}")
        
        # 必要な列を収集
        result_cols = [asin_col]
        rename_map = {asin_col: 'ASIN'}
        
        # SKU 列
        sku_col = None
        for col in subset.columns:
            if 'SKU' in str(col).upper():
                sku_col = col
                break
        
        if sku_col:
            result_cols.append(sku_col)
            rename_map[sku_col] = 'SKU'
            print(f"SKU column found: {sku_col}")
        
        # '包装の寸法（ＣＭ）' 列を探す
        dim_col_idx = None
        dim_col_name = None
        for idx, col in enumerate(subset.columns):
            if '包装の寸法' in str(col):
                dim_col_idx = idx
                dim_col_name = col
                print(f"Found dimension column at index {idx}: {col}")
                break
        
        # '包装の寸法（ＣＭ）' 列自体と、その後の2列を取得（計3列）
        if dim_col_idx is not None:
            # '包装の寸法（ＣＭ）' 列自体を1番目の寸法（長さ）として使用
            dim1_col = dim_col_name
            dim2_col = subset.columns[dim_col_idx + 1] if dim_col_idx + 1 < len(subset.columns) else None
            dim3_col = subset.columns[dim_col_idx + 2] if dim_col_idx + 2 < len(subset.columns) else None
            
            print(f"Dimension columns:")
            print(f"  1. Length ({dim1_col}): {subset[dim1_col].head(3).tolist()}")
            if dim2_col:
                print(f"  2. Width ({dim2_col}): {subset[dim2_col].head(3).tolist()}")
            if dim3_col:
                print(f"  3. Height ({dim3_col}): {subset[dim3_col].head(3).tolist()}")
            
            # 3つの寸法列を追加
            result_cols.append(dim1_col)
            rename_map[dim1_col] = '長さ(cm)'
            
            if dim2_col:
                result_cols.append(dim2_col)
                rename_map[dim2_col] = '幅(cm)'
            
            if dim3_col:
                result_cols.append(dim3_col)
                rename_map[dim3_col] = '高さ(cm)'
            
            print(f"Using 3 dimension columns: {dim1_col}, {dim2_col}, {dim3_col}")
        
        # 寸法3辺合計を探す
        total_col = None
        for col in subset.columns:
            if '寸法3辺合計' in str(col) or '3辺合計' in str(col):
                total_col = col
                break
        
        if total_col and total_col not in result_cols:
            result_cols.append(total_col)
            rename_map[total_col] = '寸法3辺合計(cm)'
            print(f"Total dimension column found: {total_col}")
        
        # 結果を作成
        result = subset[result_cols].copy()
        result = result.rename(columns=rename_map)
        
        # 寸法列を数値に変換
        for col in ['長さ(cm)', '幅(cm)', '高さ(cm)', '寸法3辺合計(cm)']:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors='coerce')
        
        print(f"Final result columns: {list(result.columns)}")
        
        return result.reset_index(drop=True)
        
    except Exception as e:
        import traceback
        print(f"Jancode preview error: {e}")
        print(traceback.format_exc())
        return pd.DataFrame()


def extract_customer_number(value: str) -> str:
    if not value:
        return ""
    if '/' in value:
        tail = value.split('/', 1)[-1].strip()
        return tail.rstrip('.').strip()
    return str(value).strip().rstrip('.').strip()


def extract_order_id_from_text(cell: str) -> str:
    text = str(cell)
    patterns = [
        r'オーダー\s*ID[:：]\s*([0-9]+)',
        r'Order\s*ID[:：]\s*([0-9]+)',
        r'オーダー番号[:：]\s*([0-9]+)',
        r'Order\s*No[:：]\s*([0-9]+)'
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            return match.group(1).strip()
    return ''


def _find_price_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cny_candidates = ['金額（cny）', '金額(cny)', '金額', 'amount', 'cny', 'currency amount']
    jpy_candidates = ['参考金額', '参考金額（jpy）', 'jpy', 'product_amount']
    cny_col = None
    jpy_col = None
    for col in df.columns:
        lower = str(col).lower()
        if not cny_col and any(keyword in lower for keyword in cny_candidates):
            cny_col = col
        if not jpy_col and any(keyword in lower for keyword in jpy_candidates):
            jpy_col = col
        if cny_col and jpy_col:
            break
    return cny_col, jpy_col


def derive_order_from_row(row) -> str:
    for val in row:
        candidate = extract_order_id_from_text(val)
        if candidate:
            return candidate
    return ''


def get_record_list_preview(order_numbers: List[str], file_path: str, asin_order_map: pd.DataFrame = None) -> pd.DataFrame:
    """
    order_numbers: send-order-list から取得した注文番号リスト
    asin_order_map: ASIN と注文番号のマッピング DataFrame（columns: ['ASIN', '注文番号']）
    """
    if not order_numbers:
        return pd.DataFrame()
    
    try:
        df, _ = load_table_with_html_fallback(file_path)
        df.columns = df.columns.str.strip()
        df = promote_header_row(df, ['注文番号', 'オーダーID', 'Order ID', '使用状況', '金額', 'オーダー番号'])
        
        # 注文番号列を検出
        order_col = find_matching_column(df, ['注文番号', 'オーダーID', 'order_no', 'order number', 'オーダー番号'])
        detail_col = find_matching_column(df, ['使用状況', '使用状況の詳細', 'detail', '操作の種類'])
        
        # 注文番号列がない場合、使用状況列からテキスト抽出
        if not order_col and detail_col:
            df['抽出用注文番号'] = df[detail_col].apply(extract_order_id_from_text)
            order_col = '抽出用注文番号'
        
        if not order_col:
            # 注文番号列が見つからない場合、order_numbers を使って手動で作成
            return pd.DataFrame()
        
        # 金額列を検出
        cny_col, jpy_col = _find_price_columns(df)
        
        # 注文番号でフィルタ（カンマ区切りの注文番号にも対応）
        expanded_orders = []
        for order in order_numbers:
            if ',' in str(order):
                expanded_orders.extend([o.strip() for o in str(order).split(',')])
            else:
                expanded_orders.append(str(order))
        
        # フィルタリング
        mask = df[order_col].astype(str).str.strip().isin([str(o).strip() for o in expanded_orders])
        subset = df[mask].copy()
        
        if subset.empty:
            return pd.DataFrame()
        
        # 必要な列だけを抽出
        result_cols = [order_col]
        if cny_col:
            result_cols.append(cny_col)
        if jpy_col and jpy_col != cny_col:
            result_cols.append(jpy_col)
        
        subset = subset[result_cols].copy()
        
        # 列名をリネーム
        rename_map = {order_col: '注文番号'}
        if cny_col:
            rename_map[cny_col] = '金額（CNY）'
        if jpy_col and jpy_col != cny_col:
            rename_map[jpy_col] = '参考金額（JPY）'
        
        subset = subset.rename(columns=rename_map)
        
        # データ型を調整
        if '注文番号' in subset.columns:
            subset['注文番号'] = subset['注文番号'].astype(str).str.strip()
        
        for amt_col in ['金額（CNY）', '参考金額（JPY）']:
            if amt_col in subset.columns:
                subset[amt_col] = pd.to_numeric(subset[amt_col], errors='coerce')
        
        # 列順を確定
        final_cols = ['注文番号']
        if '金額（CNY）' in subset.columns:
            final_cols.append('金額（CNY）')
        if '参考金額（JPY）' in subset.columns:
            final_cols.append('参考金額（JPY）')
        
        return subset[final_cols].reset_index(drop=True)
        
    except Exception as e:
        print(f"record_list preview error: {e}")
        return pd.DataFrame()


def get_send_order_preview(asins: List[str], file_path: str) -> pd.DataFrame:
    targets = [
        ['単価', '購入単価', 'unit_price', 'price'],
        ['数量', '個数', 'qty'],
        ['中国国内送料', '国内送料'],
        ['お客様管理番号', '顧客管理No', 'customer_number', '顧客番号', 'customer id', 'お客様管理番号'],
        ['注文番号', '注文ID', 'order_no', 'order number', 'オーダー番号'],
        ['商品金額', 'product_amount', 'item_price', 'price_jpy', '参考金額']
    ]
    df = _build_asin_subset(file_path, ['ASIN', 'asin'], targets, asins)
    if df.empty:
        return df
    rename_map = {}
    asin_col = find_matching_column(df, ['ASIN', 'asin'])
    if asin_col:
        rename_map[asin_col] = 'ASIN'
    order_col = find_matching_column(df, ['注文番号', '注文ID', 'order_no', 'order number'])
    if order_col:
        rename_map[order_col] = '注文番号'
    else:
        for col in df.columns:
            if any(keyword in col for keyword in ['注文番号', '注文', 'オーダー番号', 'オーダー', 'order']):
                rename_map[col] = '注文番号'
                break
    price_col = find_matching_column(df, targets[0])
    if price_col:
        rename_map[price_col] = '単価（元）'
    qty_col = find_matching_column(df, targets[1])
    if qty_col:
        rename_map[qty_col] = '数量'
    shipping_col = find_matching_column(df, targets[2])
    if shipping_col:
        rename_map[shipping_col] = '中国国内送料（元）'
    customer_col = find_matching_column(df, targets[3])
    if customer_col:
        rename_map[customer_col] = '顧客管理番号'
    else:
        for col in df.columns:
            if any(keyword in col for keyword in ['顧客管理番号', '顧客番号', 'お客様管理番号', 'customer']):
                rename_map[col] = '顧客管理番号'
                break
    df = df.rename(columns=rename_map)
    if '顧客管理番号' in df.columns:
        df['顧客管理番号'] = df['顧客管理番号'].astype(str).str.strip()
        df['顧客管理番号（番号）'] = df['顧客管理番号'].apply(lambda v: extract_customer_number(v))
    if '注文番号' not in df.columns:
        df['注文番号'] = df.apply(derive_order_from_row, axis=1)
    
    # ASIN ごとに1行に集約（指示書の商品数と一致させる）
    if 'ASIN' in df.columns:
        # 顧客管理番号（番号）の合計を計算（セット数）
        if '顧客管理番号（番号）' in df.columns:
            df['顧客管理番号（番号）'] = pd.to_numeric(df['顧客管理番号（番号）'], errors='coerce').fillna(0)
        
        # 注文数をカウント（ASINごとの注文数）
        df['注文数'] = 1
        
        # 数値列は合計、テキスト列は最初の値を使用、注文番号は結合
        agg_dict = {}
        for col in df.columns:
            if col == 'ASIN':
                continue
            elif col == '注文番号':
                agg_dict[col] = lambda x: ', '.join(x.dropna().astype(str).unique())
            elif col in ['単価（元）', '数量', '中国国内送料（元）', '顧客管理番号（番号）', '注文数']:
                agg_dict[col] = 'sum'
            else:
                agg_dict[col] = 'first'
        
        df = df.groupby('ASIN', as_index=False).agg(agg_dict)
        
        # 1注文あたりの数量を計算
        if '注文数' in df.columns and '数量' in df.columns:
            df['1注文あたり数量'] = df.apply(
                lambda row: row['数量'] / row['注文数'] if row['注文数'] > 0 else row['数量'],
                axis=1
            )
        
        # 実際の1セットあたりの数量を計算（セット数がある場合）
        if '顧客管理番号（番号）' in df.columns and '数量' in df.columns:
            df['セット数'] = df['顧客管理番号（番号）']
            # 数量をセット数で割って、1セットあたりの数量を計算
            df['1セットあたり数量'] = df.apply(
                lambda row: row['数量'] / row['セット数'] if row['セット数'] > 0 else row['数量'],
                axis=1
            )
    
    # reorder columns
    desired_order = ['ASIN', '注文番号', '単価（元）', '数量', '中国国内送料（元）', '顧客管理番号（番号）', '商品金額']
    available = [col for col in desired_order if col in df.columns]
    other_cols = [col for col in df.columns if col not in available]
    return df[available + other_cols]

def display_file_info(file_info_dict):
    """ファイル情報を視覚的に表示"""
    st.markdown('<div class="section-header">📋 読み込まれたファイル情報</div>', unsafe_allow_html=True)
    
    cols = st.columns(len(file_info_dict))
    for idx, (file_type, info) in enumerate(file_info_dict.items()):
        with cols[idx]:
            st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'**{info["label"]}**')
            if info["loaded"]:
                st.success("✓ 読み込み済み")
                st.markdown(f'<div class="file-info">{info["filename"]}</div>', unsafe_allow_html=True)
                if "rows" in info:
                    st.caption(f"行数: {info['rows']:,}")
            else:
                st.warning("未読み込み")
            st.markdown('</div>', unsafe_allow_html=True)

def process_data_from_previews(
    fba_df: pd.DataFrame,
    jancode_df: pd.DataFrame,
    send_order_df: pd.DataFrame,
    record_list_df: pd.DataFrame,
    tax_data: Dict[str, float],
    email_data: Dict[str, Dict],
    shipping_request_no: str = "",
    cny_to_jpy_rate: float = 22.77,
    discount_df: pd.DataFrame = None,
    option_distribution: Dict[str, list] = None
) -> pd.DataFrame:
    """プレビューデータから処理結果を生成"""
    
    results = []
    errors = []
    
    # デバッグ: fba_dfの列を確認
    print(f"\n=== process_data_from_previews開始 ===")
    print(f"fba_df columns: {fba_df.columns.tolist()}")
    print(f"fba_df shape: {fba_df.shape}")
    if 'オプション費用（元）' in fba_df.columns:
        print(f"オプション費用（元）列が見つかりました！")
        print(f"サンプル値: {fba_df['オプション費用（元）'].head()}")
    else:
        print(f"⚠️ オプション費用（元）列が見つかりません！")
    
    # メールデータから配送依頼No.に対応する情報を取得
    email_info = email_data.get(shipping_request_no, {}) if shipping_request_no else {}
    
    # 第1パス: 各商品の体積を計算し、総体積を求める
    product_volumes = {}  # {ASIN: 体積}
    total_volume = 0.0
    
    print(f"\n=== 第1パス: 体積計算 ===")
    print(f"fba_df shape: {fba_df.shape}")
    print(f"fba_df columns: {list(fba_df.columns)}")
    print(f"fba_df (B0BKFS9N54):")
    print(fba_df[fba_df['ASIN'] == 'B0BKFS9N54'])
    
    for _, fba_row in fba_df.iterrows():
        asin = fba_row.get('ASIN')
        if not asin or pd.isna(asin):
            continue
        
        qty = fba_row.get('数量', 0) or 0
        
        if asin == 'B0BKFS9N54':
            print(f"\n第1パス - B0BKFS9N54:")
            print(f"  数量: {qty}")
        
        # Jancodeから寸法を取得
        length = width = height = 0
        if not jancode_df.empty and 'ASIN' in jancode_df.columns:
            jancode_match = jancode_df[jancode_df['ASIN'] == asin]
            if not jancode_match.empty:
                length = jancode_match.iloc[0].get('長さ(cm)', 0) or 0
                width = jancode_match.iloc[0].get('幅(cm)', 0) or 0
                height = jancode_match.iloc[0].get('高さ(cm)', 0) or 0
        
        # 寸法チェック
        if length == 0 or width == 0 or height == 0:
            errors.append(f"⚠️ ASIN {asin} の寸法情報が不足しています（長さ: {length}, 幅: {width}, 高さ: {height}）")
            volume = 0
        else:
            volume = length * width * height
        
        product_volumes[asin] = volume
        total_volume += volume * qty
    
    # 国際送料の取得
    international_shipping_cny = 0
    if email_info and '国際送料' in email_info:
        international_shipping_cny = email_info['国際送料']
    else:
        if shipping_request_no:
            errors.append(f"⚠️ 配送依頼No. {shipping_request_no} の国際送料がメールデータに入力されていません")
    
    international_shipping_jpy = international_shipping_cny * cny_to_jpy_rate
    
    # 第2パス: 各商品の処理
    print(f"\n=== 第2パス: 商品処理 ===")
    for _, fba_row in fba_df.iterrows():
        asin = fba_row.get('ASIN')
        if not asin or pd.isna(asin):
            continue
        
        qty_from_fba = fba_row.get('数量', 0)
        
        if asin == 'B0BKFS9N54':
            print(f"\n第2パス - B0BKFS9N54:")
            print(f"  fba_row['数量']: {qty_from_fba}")
            print(f"  fba_row: {fba_row.to_dict()}")
        
        result = {
            'ASIN': asin,
            '商品名': fba_row.get('商品名', ''),
            '数量': qty_from_fba
        }
        
        # Jancode から SKU と寸法を取得
        if not jancode_df.empty and 'ASIN' in jancode_df.columns:
            jancode_match = jancode_df[jancode_df['ASIN'] == asin]
            if not jancode_match.empty:
                result['SKU'] = jancode_match.iloc[0].get('SKU', '')
                result['長さ(cm)'] = jancode_match.iloc[0].get('長さ(cm)', 0)
                result['幅(cm)'] = jancode_match.iloc[0].get('幅(cm)', 0)
                result['高さ(cm)'] = jancode_match.iloc[0].get('高さ(cm)', 0)
                result['寸法3辺合計(cm)'] = jancode_match.iloc[0].get('寸法3辺合計(cm)', 0)
        
        # send-order から単価、数量、中国国内送料を取得
        if not send_order_df.empty and 'ASIN' in send_order_df.columns:
            send_match = send_order_df[send_order_df['ASIN'] == asin]
            if not send_match.empty:
                result['購入単価（元）'] = send_match.iloc[0].get('単価（元）', 0)
                result['中国国内送料（元）'] = send_match.iloc[0].get('中国国内送料（元）', 0)
                result['注文番号'] = send_match.iloc[0].get('注文番号', '')
                
                # 数量チェック: 指示書の数量とsend-order-listの1注文あたり数量を比較
                send_qty_total = send_match.iloc[0].get('数量', 0) or 0
                send_qty_per_order = send_match.iloc[0].get('1注文あたり数量', send_qty_total) or 0
                order_count = send_match.iloc[0].get('注文数', 1) or 1
                set_count = send_match.iloc[0].get('セット数', 0) or 0
                send_qty_per_set = send_match.iloc[0].get('1セットあたり数量', 0) or 0
                
                # qty_from_fbaを使用（resultの数量）
                fba_qty = result.get('数量', 0)
                
                print(f"\n=== 数量チェック for ASIN {asin} ===")
                print(f"指示書の数量: {fba_qty}")
                print(f"send-order-list 合計数量: {send_qty_total}")
                print(f"send-order-list 注文数: {order_count}")
                print(f"send-order-list 1注文あたり数量: {send_qty_per_order}")
                if set_count > 0:
                    print(f"send-order-list セット数: {set_count}")
                    print(f"send-order-list 1セットあたり数量: {send_qty_per_set}")
                
                # 指示書の数量と1注文あたりの数量を比較
                if abs(send_qty_per_order - fba_qty) > 0.01:  # 小数点誤差を考慮
                    if set_count > 0:
                        errors.append(
                            f"⚠️ ASIN {asin} の数量が一致しません\n"
                            f"  指示書: {fba_qty}個\n"
                            f"  send-order-list: {send_qty_total}個（{order_count}注文 × {send_qty_per_order}個/注文）\n"
                            f"  セット情報: {set_count}セット × {send_qty_per_set}個/セット"
                        )
                    else:
                        errors.append(
                            f"⚠️ ASIN {asin} の数量が一致しません\n"
                            f"  指示書: {fba_qty}個\n"
                            f"  send-order-list: {send_qty_total}個（{order_count}注文 × {send_qty_per_order}個/注文）"
                        )
        
        # record-list から商品金額を取得
        if '注文番号' in result and result['注文番号'] and not record_list_df.empty and '注文番号' in record_list_df.columns:
            order_numbers = str(result['注文番号']).split(',')
            total_cny = 0
            total_jpy = 0
            for order_no in order_numbers:
                order_no = order_no.strip()
                record_match = record_list_df[record_list_df['注文番号'].astype(str).str.strip() == order_no]
                if not record_match.empty:
                    total_cny += record_match.iloc[0].get('金額（CNY）', 0) or 0
                    total_jpy += record_match.iloc[0].get('参考金額（JPY）', 0) or 0
            
            result['商品金額（CNY）'] = total_cny
            result['商品金額（JPY）'] = total_jpy
        
        # 1個あたりの計算
        qty = result.get('数量', 1) or 1
        
        # 中国国内送料（元）はsend-order-listから既に取得済み
        domestic_shipping_cny = result.get('中国国内送料（元）', 0) or 0
        
        # 国際送料を体積比率で按分
        volume = product_volumes.get(asin, 0)
        result['体積(cm3)'] = volume
        
        if volume > 0 and total_volume > 0:
            # 商品1個あたりの国際送料（円） = 国際送料（円） × (商品1個の体積 / 総体積)
            result['商品1個あたり国際送料（円）'] = international_shipping_jpy * (volume / total_volume)
            # 元に変換（表示用）
            result['商品1個あたり国際送料（元）'] = result['商品1個あたり国際送料（円）'] / cny_to_jpy_rate
            
            # デバッグ情報
            print(f"ASIN {asin}:")
            print(f"  体積: {volume} cm³")
            print(f"  数量: {qty}")
            print(f"  総体積: {total_volume} cm³")
            print(f"  国際送料（円）: {international_shipping_jpy}")
            print(f"  比率: {volume / total_volume}")
            print(f"  商品1個あたり国際送料（円）: {result['商品1個あたり国際送料（円）']}")
        else:
            result['商品1個あたり国際送料（円）'] = 0
            result['商品1個あたり国際送料（元）'] = 0
        
        # 割引額を計算
        print(f"\n=== 割引額計算 for ASIN {asin} ===")
        print(f"discount_df is None: {discount_df is None}")
        print(f"discount_df.empty: {discount_df.empty if discount_df is not None else 'N/A'}")
        print(f"'注文番号' in result: {'注文番号' in result}")
        print(f"result['注文番号']: {result.get('注文番号', 'N/A')}")
        
        if discount_df is not None and not discount_df.empty and '注文番号' in result and result['注文番号']:
            order_numbers = str(result['注文番号']).split(',')
            print(f"Order numbers to search: {order_numbers}")
            if len(discount_df) > 0:
                print(f"Available order numbers in discount_df: {discount_df['注文番号'].head(10).tolist()}")
            
            total_discount_cny = 0
            for order_no in order_numbers:
                order_no = order_no.strip()
                print(f"  Searching for order: '{order_no}'")
                # 文字列として比較
                discount_match = discount_df[discount_df['注文番号'] == order_no]
                print(f"  Matches found: {len(discount_match)}")
                if not discount_match.empty:
                    discount_amount = discount_match.iloc[0].get('割引額（元）', 0) or 0
                    print(f"  Discount amount: {discount_amount}")
                    total_discount_cny += discount_amount
            
            print(f"Total discount (CNY): {total_discount_cny}")
            
            # 商品1個あたりの割引額（円） = 割引額（元） × 元→円レート / 数量
            if qty > 0:
                result['商品1個につき割引額（円）'] = (total_discount_cny * cny_to_jpy_rate) / qty
                result['商品1個につき割引額（元）'] = result['商品1個につき割引額（円）'] / cny_to_jpy_rate
                print(f"商品1個につき割引額（円）: {result['商品1個につき割引額（円）']}")
            else:
                result['商品1個につき割引額（円）'] = 0
                result['商品1個につき割引額（元）'] = 0
        else:
            print("Discount calculation skipped")
            result['商品1個につき割引額（円）'] = 0
            result['商品1個につき割引額（元）'] = 0
        
        # オプション費用を計算
        print(f"\n=== オプション費用計算 for ASIN {asin} ===")
        
        # 1. 指示書のL列から基本オプション費用を取得
        base_option_fee_cny = 0
        if 'オプション費用（元）' in fba_df.columns:
            fba_match = fba_df[fba_df['ASIN'] == asin]
            if not fba_match.empty:
                base_option_fee_cny = fba_match.iloc[0].get('オプション費用（元）', 0) or 0
                print(f"指示書のL列からオプション費用: {base_option_fee_cny}元")
        
        # 基本オプション費用を円に換算して1個あたりに計算
        base_option_fee_jpy = base_option_fee_cny * cny_to_jpy_rate
        base_option_fee_per_item_jpy = base_option_fee_jpy / qty if qty > 0 else 0
        print(f"基本オプション費用（1個あたり・円）: {base_option_fee_per_item_jpy}")
        
        # 2. メールのオプション料金で検算（合計値と比較）
        # 注意：この検算は最初のASINの処理時のみ実行
        if email_info and 'オプション料金' in email_info and asin == fba_df.iloc[0]['ASIN']:
            email_option_total_cny = email_info.get('オプション料金', 0) or 0
            instruction_option_total_cny = fba_df['オプション費用（元）'].sum() if 'オプション費用（元）' in fba_df.columns else 0
            
            print(f"\n=== オプション費用の検算（配送依頼No. {shipping_request_no}）===")
            print(f"メールのオプション料金合計: {email_option_total_cny}元")
            print(f"指示書のオプション費用合計（groupby後）: {instruction_option_total_cny}元")
            print(f"fba_df行数: {len(fba_df)}")
            if 'オプション費用（元）' in fba_df.columns:
                print(f"fba_dfのオプション費用詳細:")
                print(fba_df[['ASIN', '数量', 'オプション費用（元）']].to_string())
            
            difference = email_option_total_cny - instruction_option_total_cny
            if abs(difference) > 0.01:
                print(f"⚠️ 差分あり: {difference}元")
                print(f"   この差分は追加オプション費用配分で処理してください")
            else:
                print(f"✓ メールと指示書のオプション費用が一致しています")
        
        # 3. 追加オプション費用配分（特定のASINに追加）
        additional_option_fee_jpy = 0
        if option_distribution and shipping_request_no in option_distribution:
            distributions = option_distribution[shipping_request_no]
            for dist in distributions:
                if asin in dist['ASINs']:
                    # この配分がこのASINに適用される
                    dist_amount_cny = dist['金額（元）']
                    dist_amount_jpy = dist_amount_cny * cny_to_jpy_rate
                    # 配分先ASINの数で割る
                    num_target_asins = len(dist['ASINs'])
                    if num_target_asins > 0:
                        # さらに、そのASINの数量で割る
                        dist_per_item = (dist_amount_jpy / num_target_asins) / qty if qty > 0 else 0
                        additional_option_fee_jpy += dist_per_item
                        print(f"追加配分: {dist['説明']} = {dist_per_item}円/個")
        
        # 合計オプション費用
        total_option_fee_per_item_jpy = base_option_fee_per_item_jpy + additional_option_fee_jpy
        result['商品1個あたりのオプション費用（円）'] = total_option_fee_per_item_jpy
        result['商品1個あたりのオプション費用（元）'] = total_option_fee_per_item_jpy / cny_to_jpy_rate if cny_to_jpy_rate > 0 else 0
        print(f"合計オプション費用（1個あたり・円）: {total_option_fee_per_item_jpy}")
        
        # 税金を数量の比率で按分
        if tax_data:
            total_customs = tax_data.get('関税', 0) or 0
            total_consumption = tax_data.get('消費税', 0) or 0
            total_local = tax_data.get('地方消費税', 0) or 0
            
            # ASINの数で割って、さらにそのASINの数量で割る
            total_asins = len(fba_df)
            if total_asins > 0 and qty > 0:
                # 1ASINあたりの税金
                customs_per_asin = total_customs / total_asins
                consumption_per_asin = total_consumption / total_asins
                local_per_asin = total_local / total_asins
                
                # 商品1個あたりの税金
                result['商品1個あたり関税（円）'] = customs_per_asin / qty
                result['商品1個あたり消費税（円）'] = consumption_per_asin / qty
                result['商品1個あたり地方消費税（円）'] = local_per_asin / qty
        
        # 原価の計算
        unit_price_cny = result.get('購入単価（元）', 0) or 0
        unit_price_jpy = unit_price_cny * cny_to_jpy_rate
        
        # 中国国内送料（元）を円に変換して数量で割って1個あたりに
        domestic_shipping_jpy = domestic_shipping_cny * cny_to_jpy_rate
        domestic_shipping_per_item = domestic_shipping_jpy / qty if qty > 0 else 0
        
        result['原価(円)'] = (
            unit_price_jpy +
            domestic_shipping_per_item +
            result.get('商品1個あたり国際送料（円）', 0) +
            result.get('商品1個あたりのオプション費用（円）', 0) +
            result.get('商品1個あたり関税（円）', 0)
        )
        
        results.append(result)
    
    result_df = pd.DataFrame(results)
    
    # エラーがある場合は警告として表示（呼び出し元で処理）
    if errors:
        result_df.attrs['errors'] = errors
    
    return result_df


def display_data_flow():
    """データフローを視覚的に表示"""
    st.markdown('<div class="section-header">🔄 データ処理フロー</div>', unsafe_allow_html=True)
    
    flow_steps = [
        {"icon": "📄", "name": "FBA指示書", "desc": "配送依頼No、納品プランNo、追跡番号を抽出", "source": "ファイル名から自動抽出"},
        {"icon": "📦", "name": "send-order-list", "desc": "配送日、注文情報、単価、数量を取得", "source": "Excelファイルから読み込み"},
        {"icon": "💰", "name": "record-list", "desc": "注文番号から参考金額(JPY)を取得", "source": "Excelファイルから読み込み"},
        {"icon": "📋", "name": "Jancode.xlsx", "desc": "ASIN、SKU、商品名、寸法情報を取得", "source": "Excelファイルから読み込み"},
        {"icon": "📧", "name": "輸入許可通知書", "desc": "関税、消費税情報を取得", "source": "PDFファイルから抽出"},
        {"icon": "💱", "name": "為替レート", "desc": "前日のTTS + 0.5で計算", "source": "APIまたは手動入力"},
        {"icon": "⚙️", "name": "計算処理", "desc": "送料、原価を計算", "source": "寸法と設定ファイルから計算"},
        {"icon": "📊", "name": "結果出力", "desc": "Excelファイルに出力", "source": "すべての情報を統合"}
    ]
    
    # フローチャート風に表示
    for i, step in enumerate(flow_steps):
        col1, col2, col3 = st.columns([1, 8, 1])
        with col1:
            st.markdown(f'<div style="text-align: center; font-size: 2rem;">{step["icon"]}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(
                f'<div class="info-box">'
                f'<strong>{step["name"]}</strong><br>'
                f'{step["desc"]}<br>'
                f'<small style="color: #666;">📌 取得元: {step["source"]}</small>'
                f'</div>', 
                unsafe_allow_html=True
            )
        with col3:
            if i < len(flow_steps) - 1:
                st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)

def main():
    """メインアプリケーション"""
    initialize_session_state()
    
    # ヘッダー
    st.markdown('<div class="main-header">📦 FBA注文処理システム</div>', unsafe_allow_html=True)
    
    # サイドバー
    with st.sidebar:
        st.header("⚙️ 設定")
        st.markdown("---")
        
        # 為替レート設定
        st.subheader("為替レート")
        use_custom_rate = st.checkbox("カスタム為替レートを使用", value=False)
        custom_rate = None
        if use_custom_rate:
            custom_rate = st.number_input("為替レート (円/元)", value=22.77, step=0.01)
        
        st.markdown("---")
        
        # 配送代行手数料設定
        st.subheader("配送代行手数料")
        if st.button("設定ファイルを編集"):
            st.info("shipping_fee_config.jsonを編集してください")
    
    # メインコンテンツ
    tab1, tab2, tab3, tab4 = st.tabs(["📁 ファイル選択", "📊 データ確認", "⚙️ 処理実行", "📈 結果表示"])
    
    with tab1:
        st.markdown('<div class="section-header">ファイルをアップロード</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("必須ファイル")
            
            # FBA指示書
            fba_file = st.file_uploader(
                "FBA指示書 (.xls)",
                type=['xls', 'xlsx'],
                help="配送依頼No、納品プランNo、追跡番号を含む指示書"
            )
            if fba_file:
                fba_path = save_uploaded_file(fba_file, "fba_instruction")
                st.session_state.uploaded_files['fba'] = fba_path
                st.session_state.file_info['fba'] = {
                    "label": "FBA指示書",
                    "loaded": True,
                    "filename": fba_file.name
                }
                fba_meta = parse_fba_filename_metadata(fba_file.name)
                if fba_meta:
                    st.session_state.metadata.update(fba_meta)
            
            # send-order-list
            send_order_file = st.file_uploader(
                "send-order-list (.xls)",
                type=['xls', 'xlsx'],
                help="配送日、注文情報を含むファイル"
            )
            if send_order_file:
                send_order_path = save_uploaded_file(send_order_file, "send_order_list")
                st.session_state.uploaded_files['send_order'] = send_order_path
                st.session_state.file_info['send_order'] = {
                    "label": "send-order-list",
                    "loaded": True,
                    "filename": send_order_file.name
                }
                shipping_date = parse_send_order_filename(send_order_file.name)
                if shipping_date:
                    st.session_state.metadata['shipping_date'] = shipping_date
            
            # record-list
            record_list_file = st.file_uploader(
                "record-list (.xls)",
                type=['xls', 'xlsx'],
                help="注文番号と参考金額(JPY)を含むファイル"
            )
            if record_list_file:
                record_list_path = save_uploaded_file(record_list_file, "record_list")
                st.session_state.uploaded_files['record_list'] = record_list_path
                st.session_state.file_info['record_list'] = {
                    "label": "record-list",
                    "loaded": True,
                    "filename": record_list_file.name
                }
        
        with col2:
            st.subheader("オプションファイル")
            
            # Jancode.xlsx
            jancode_file = st.file_uploader(
                "Jancode.xlsx",
                type=['xlsx'],
                help="ASIN、SKU、商品名、寸法情報を含むファイル"
            )
            if jancode_file:
                jancode_path = save_uploaded_file(jancode_file, "jancode")
                st.session_state.uploaded_files['jancode'] = jancode_path
                st.session_state.file_info['jancode'] = {
                    "label": "Jancode.xlsx",
                    "loaded": True,
                    "filename": jancode_file.name
                }
            
            # 輸入許可通知書 (PDF)
            import_permit_file = st.file_uploader(
                "輸入許可通知書 (.pdf)",
                type=['pdf'],
                help="関税、消費税情報を含むPDFファイル"
            )
            if import_permit_file:
                permit_path = save_uploaded_file(import_permit_file, "import_permit")
                st.session_state.uploaded_files['import_permit'] = permit_path
                st.session_state.file_info['import_permit'] = {
                    "label": "輸入許可通知書",
                    "loaded": True,
                    "filename": import_permit_file.name
                }
            
            # 手動入力データ
            manual_input_file = st.file_uploader(
                "手動入力データ (.txt)",
                type=['txt'],
                help="オプション費用、割引額、関税、消費税の手動入力データ"
            )
            if manual_input_file:
                manual_path = save_uploaded_file(manual_input_file, "manual_input")
                st.session_state.uploaded_files['manual_input'] = manual_path
                st.session_state.file_info['manual_input'] = {
                    "label": "手動入力データ",
                    "loaded": True,
                    "filename": manual_input_file.name
                }
            
            # オプション費用分配情報
            option_dist_file = st.file_uploader(
                "オプション費用分配情報 (.txt)",
                type=['txt'],
                help="オプション費用の分配情報"
            )
            if option_dist_file:
                option_path = save_uploaded_file(option_dist_file, "option_distribution")
                st.session_state.uploaded_files['option_distribution'] = option_path
                st.session_state.file_info['option_distribution'] = {
                    "label": "オプション費用分配",
                    "loaded": True,
                    "filename": option_dist_file.name
                }
        
        # メール情報の手動入力セクション
        st.markdown("---")
        st.subheader("📧 メールから取得する情報の手動入力")
        st.caption("配送依頼ごとの国際送料、オプション料金、通関手数料、中国国内送料を入力してください")
        
        # セッション状態の初期化
        if 'email_data' not in st.session_state:
            st.session_state.email_data = {}
        
        with st.expander("✍️ メール情報を入力", expanded=False):
            # 配送依頼番号を入力
            delivery_request_no = st.text_input(
                "配送依頼No.",
                placeholder="例: 663864",
                help="この配送依頼に関する情報を入力します",
                key="delivery_no_input"
            )
            
            if delivery_request_no:
                st.markdown(f"**配送依頼No. {delivery_request_no} の情報**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**基本料金（元）**")
                    international_shipping = st.number_input(
                        "国際送料（元）",
                        min_value=0.0,
                        value=0.0,
                        step=10.0,
                        format="%.2f",
                        key=f"intl_ship_{delivery_request_no}"
                    )
                    
                    option_fee = st.number_input(
                        "オプション料金（元）",
                        min_value=0.0,
                        value=0.0,
                        step=10.0,
                        format="%.2f",
                        key=f"option_{delivery_request_no}"
                    )
                    
                    customs_fee = st.number_input(
                        "通関手数料（元）",
                        min_value=0.0,
                        value=0.0,
                        step=1.0,
                        format="%.2f",
                        key=f"customs_fee_{delivery_request_no}"
                    )
                    
                    domestic_shipping = st.number_input(
                        "中国国内送料（元）",
                        min_value=0.0,
                        value=0.0,
                        step=10.0,
                        format="%.2f",
                        key=f"domestic_{delivery_request_no}"
                    )
                
                with col2:
                    st.markdown("**オプション料金の内訳（任意）**")
                    st.caption("オプション料金の詳細がある場合に入力")
                    
                    special_inspection = st.number_input(
                        "特殊検品（元）",
                        min_value=0.0,
                        value=0.0,
                        step=1.0,
                        format="%.2f",
                        key=f"special_{delivery_request_no}"
                    )
                    
                    full_inspection = st.number_input(
                        "全開封検査（元）",
                        min_value=0.0,
                        value=0.0,
                        step=1.0,
                        format="%.2f",
                        key=f"full_insp_{delivery_request_no}"
                    )
                    
                    photography = st.number_input(
                        "撮影（元）",
                        min_value=0.0,
                        value=0.0,
                        step=1.0,
                        format="%.2f",
                        key=f"photo_{delivery_request_no}"
                    )
                    
                    other_option = st.number_input(
                        "その他オプション（元）",
                        min_value=0.0,
                        value=0.0,
                        step=1.0,
                        format="%.2f",
                        key=f"other_{delivery_request_no}"
                    )
                
                # オプション費用の追加配分設定
                st.markdown("---")
                st.markdown("**📌 追加オプション費用の配分（例外処理4）**")
                st.caption("指示書に記載されていない追加オプション費用がある場合、特定のASINに配分します")
                
                if 'option_distribution' not in st.session_state:
                    st.session_state.option_distribution = {}
                if delivery_request_no not in st.session_state.option_distribution:
                    st.session_state.option_distribution[delivery_request_no] = []
                
                # 配分設定の追加
                with st.expander("➕ オプション費用配分を追加", expanded=False):
                    col_desc, col_amount, col_asins = st.columns([2, 1, 3])
                    
                    with col_desc:
                        dist_description = st.text_input(
                            "説明",
                            placeholder="例: 特殊検品",
                            key=f"dist_desc_{delivery_request_no}"
                        )
                    
                    with col_amount:
                        dist_amount = st.number_input(
                            "金額（元）",
                            min_value=0.0,
                            value=0.0,
                            step=1.0,
                            format="%.2f",
                            key=f"dist_amount_{delivery_request_no}"
                        )
                    
                    with col_asins:
                        dist_asins = st.text_input(
                            "配分先ASIN（カンマ区切り）",
                            placeholder="例: B0BKFS9N54, B0G1LDVHGV",
                            key=f"dist_asins_{delivery_request_no}"
                        )
                    
                    if st.button("配分を追加", key=f"add_dist_{delivery_request_no}"):
                        if dist_description and dist_amount > 0 and dist_asins:
                            asin_list = [a.strip() for a in dist_asins.split(',')]
                            st.session_state.option_distribution[delivery_request_no].append({
                                '説明': dist_description,
                                '金額（元）': dist_amount,
                                'ASINs': asin_list
                            })
                            st.success(f"配分を追加しました: {dist_description} - {dist_amount}元")
                        else:
                            st.error("すべてのフィールドを入力してください")
                
                # 現在の配分設定を表示
                if st.session_state.option_distribution.get(delivery_request_no):
                    st.markdown("**現在の配分設定:**")
                    for idx, dist in enumerate(st.session_state.option_distribution[delivery_request_no]):
                        col_info, col_delete = st.columns([4, 1])
                        with col_info:
                            st.text(f"{dist['説明']}: {dist['金額（元）']}元 → {', '.join(dist['ASINs'])}")
                        with col_delete:
                            if st.button("🗑️", key=f"del_dist_{delivery_request_no}_{idx}"):
                                st.session_state.option_distribution[delivery_request_no].pop(idx)
                                st.rerun()
                
                # データを保存
                col_save, col_delete = st.columns([3, 1])
                with col_save:
                    if st.button("💾 保存", key=f"save_{delivery_request_no}", type="primary"):
                        st.session_state.email_data[delivery_request_no] = {
                            '国際送料': international_shipping,
                            'オプション料金': option_fee,
                            '通関手数料': customs_fee,
                            '中国国内送料': domestic_shipping,
                            '特殊検品': special_inspection,
                            '全開封検査': full_inspection,
                            '撮影': photography,
                            'その他オプション': other_option
                        }
                        st.success(f"✓ 配送依頼No. {delivery_request_no} の情報を保存しました")
                        
                        # 合計を表示
                        total = international_shipping + option_fee + customs_fee + domestic_shipping
                        st.info(f"**合計:** {total:,.2f} 元")
                
                with col_delete:
                    if delivery_request_no in st.session_state.email_data:
                        if st.button("🗑️ 削除", key=f"delete_{delivery_request_no}"):
                            del st.session_state.email_data[delivery_request_no]
                            st.success(f"削除しました")
                            st.rerun()
        
        # 保存済みのメール情報を表示
        if st.session_state.email_data:
            st.markdown("**📋 保存済みのメール情報**")
            
            for delivery_no, data in st.session_state.email_data.items():
                with st.expander(f"📦 配送依頼No. {delivery_no}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("国際送料", f"{data.get('国際送料', 0):,.2f} 元")
                        st.metric("オプション料金", f"{data.get('オプション料金', 0):,.2f} 元")
                    
                    with col2:
                        st.metric("通関手数料", f"{data.get('通関手数料', 0):,.2f} 元")
                        st.metric("中国国内送料", f"{data.get('中国国内送料', 0):,.2f} 元")
                    
                    # オプション内訳がある場合
                    option_details = []
                    if data.get('特殊検品', 0) > 0:
                        option_details.append(f"特殊検品={data['特殊検品']:.2f}元")
                    if data.get('全開封検査', 0) > 0:
                        option_details.append(f"全開封検査={data['全開封検査']:.2f}元")
                    if data.get('撮影', 0) > 0:
                        option_details.append(f"撮影={data['撮影']:.2f}元")
                    if data.get('その他オプション', 0) > 0:
                        option_details.append(f"その他={data['その他オプション']:.2f}元")
                    
                    if option_details:
                        st.caption(f"オプション内訳: {', '.join(option_details)}")
                    
                    # 合計
                    total = (data.get('国際送料', 0) + data.get('オプション料金', 0) + 
                            data.get('通関手数料', 0) + data.get('中国国内送料', 0))
                    st.info(f"**合計:** {total:,.2f} 元")
        
        # ファイル情報表示
        if st.session_state.file_info:
            display_file_info(st.session_state.file_info)
        
        # データフロー表示
        display_data_flow()
    
    with tab2:
        st.markdown('<div class="section-header">データ確認</div>', unsafe_allow_html=True)
        display_metadata_header(st.session_state.metadata)
        
        if not st.session_state.uploaded_files:
            st.info("まず「ファイル選択」タブでファイルをアップロードしてください。")
        else:
            asins = []
            if 'fba' in st.session_state.uploaded_files:
                instruction_df = get_instruction_summary(st.session_state.uploaded_files['fba'])
                if not instruction_df.empty:
                    asins = instruction_df['ASIN'].dropna().astype(str).tolist()
                    st.subheader("📄 fbaのプレビュー")
                    
                    # オプション費用列をハイライト
                    def highlight_fba_option_cost(df):
                        styles = pd.DataFrame('', index=df.index, columns=df.columns)
                        if 'オプション費用（元）' in df.columns:
                            styles['オプション費用（元）'] = 'background-color: #E8F4F8; border-left: 3px solid #4A90A4; font-weight: 500;'
                        return styles
                    
                    styled_fba = instruction_df.style.apply(highlight_fba_option_cost, axis=None)
                    format_dict = {}
                    for col in instruction_df.columns:
                        if 'オプション' in col or '費用' in col:
                            format_dict[col] = '{:,.2f}'
                        elif col == '数量':
                            format_dict[col] = '{:,.0f}'
                    styled_fba = styled_fba.format(format_dict, na_rep='-')
                    
                    st.dataframe(styled_fba, width='stretch', height=250)
            order_numbers = []
            send_order_matches = pd.DataFrame()
            for file_type, file_path in st.session_state.uploaded_files.items():
                if not file_path or not os.path.exists(file_path):
                    continue
                
                # カスタムプレビューを表示するファイルタイプ
                if file_type == 'jancode' and asins:
                    detail_df = get_jancode_preview(asins, file_path)
                    if not detail_df.empty:
                        st.subheader("📄 jancodeのプレビュー")
                        st.caption(f"指示書の商品数: {len(asins)}個 | 表示中: {len(detail_df)}個")
                        
                        # 寸法情報（国際送料計算に使用）をハイライト
                        def highlight_jancode_dimensions(df):
                            styles = pd.DataFrame('', index=df.index, columns=df.columns)
                            dimension_cols = ['長さ(cm)', '幅(cm)', '高さ(cm)', '寸法3辺合計(cm)']
                            for col in dimension_cols:
                                if col in df.columns:
                                    styles[col] = 'background-color: #FFF4E6; border-left: 3px solid #F59E0B; font-weight: 500;'
                            return styles
                        
                        styled_jancode = detail_df.style.apply(highlight_jancode_dimensions, axis=None)
                        format_dict = {}
                        for col in detail_df.columns:
                            if 'cm' in col or '寸法' in col:
                                format_dict[col] = '{:,.2f}'
                        styled_jancode = styled_jancode.format(format_dict, na_rep='-')
                        
                        st.dataframe(styled_jancode, width='stretch', height=250)
                    continue  # display_data_preview をスキップ
                
                if file_type == 'send_order' and asins:
                    detail_df = get_send_order_preview(asins, file_path)
                    if not detail_df.empty:
                        st.subheader("📄 send_orderのプレビュー")
                        st.caption(f"指示書の商品数: {len(asins)}個 | 表示中: {len(detail_df)}個")
                        
                        # 原価計算に使用される列をハイライト
                        def highlight_send_order_cost_columns(df):
                            styles = pd.DataFrame('', index=df.index, columns=df.columns)
                            if '購入単価（元）' in df.columns:
                                styles['購入単価（元）'] = 'background-color: #E8F4F8; border-left: 3px solid #4A90A4; font-weight: 500;'
                            if '中国国内送料（元）' in df.columns:
                                styles['中国国内送料（元）'] = 'background-color: #E8F4F8; border-left: 3px solid #4A90A4; font-weight: 500;'
                            return styles
                        
                        styled_send_order = detail_df.style.apply(highlight_send_order_cost_columns, axis=None)
                        format_dict = {}
                        for col in detail_df.columns:
                            if '単価' in col or '送料' in col:
                                format_dict[col] = '{:,.2f}'
                            elif col == '数量':
                                format_dict[col] = '{:,.0f}'
                        styled_send_order = styled_send_order.format(format_dict, na_rep='-')
                        
                        st.dataframe(styled_send_order, width='stretch', height=200)
                        if '注文番号' in detail_df.columns:
                            order_numbers = detail_df['注文番号'].dropna().astype(str).unique().tolist()
                            send_order_matches = detail_df[['ASIN', '注文番号']].drop_duplicates()
                    continue  # display_data_preview をスキップ
                
                if file_type == 'fba':
                    continue  # fba は既に表示済み
                
                if file_type == 'record_list' and order_numbers:
                    record_df = get_record_list_preview(order_numbers, file_path, send_order_matches)
                    if not record_df.empty:
                        st.subheader("📄 record_listのプレビュー")
                        st.caption(f"send-order-listの注文番号: {len(order_numbers)}個 | record-listで見つかった件数: {len(record_df)}件")
                        # 注文番号・金額（CNY）・参考金額（JPY）を表示
                        display_cols = [col for col in ['注文番号', '金額（CNY）', '参考金額（JPY）'] if col in record_df.columns]
                        if display_cols:
                            st.dataframe(record_df[display_cols], width='stretch', height=200)
                        else:
                            st.warning("注文番号または金額列が見つかりませんでした")
                            st.dataframe(record_df, width='stretch', height=200)
                    else:
                        st.warning("record-list に該当する注文番号が見つかりませんでした")
                    continue  # display_data_preview をスキップ
                
                if file_type == 'import_permit':
                    tax_data = extract_tax_from_pdf(file_path)
                    st.subheader("📄 輸入許可通知書（PDF）のプレビュー")
                    
                    # 手動修正オプション
                    with st.expander("🔧 手動で修正する（自動抽出が不正確な場合）", expanded=False):
                        st.caption("自動抽出された値が不正確な場合、ここで修正できます")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            manual_customs = st.number_input(
                                "関税（円）", 
                                min_value=0.0, 
                                value=float(tax_data['関税']) if tax_data['関税'] else 0.0,
                                step=100.0,
                                key="manual_customs"
                            )
                        with col_b:
                            manual_consumption = st.number_input(
                                "消費税（円）", 
                                min_value=0.0, 
                                value=float(tax_data['消費税']) if tax_data['消費税'] else 0.0,
                                step=100.0,
                                key="manual_consumption"
                            )
                        with col_c:
                            manual_local = st.number_input(
                                "地方消費税（円）", 
                                min_value=0.0, 
                                value=float(tax_data['地方消費税']) if tax_data['地方消費税'] else 0.0,
                                step=100.0,
                                key="manual_local"
                            )
                        
                        if st.button("✓ 手動入力値を使用"):
                            tax_data['関税'] = manual_customs
                            tax_data['消費税'] = manual_consumption
                            tax_data['地方消費税'] = manual_local
                            st.success("手動入力値を設定しました")
                    
                    # 抽出された値を表示（原価計算に使用される関税のみハイライト）
                    st.markdown("""
                    <div style="margin-bottom: 10px; padding: 8px; background-color: #F8FAFB; border-radius: 6px; border-left: 4px solid #4A90A4;">
                        <span style="font-size: 12px; color: #2E5266;">💡 <strong>関税</strong>は原価計算に使用されます</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if tax_data['関税'] is not None and tax_data['関税'] > 0:
                            st.markdown(f"""
                            <div style="padding: 15px; background-color: #E8F4F8; border-left: 4px solid #4A90A4; border-radius: 6px;">
                                <div style="font-size: 12px; color: #6B7280; font-weight: 500;">関税 ✓</div>
                                <div style="font-size: 24px; color: #2E5266; font-weight: 700; margin-top: 5px;">¥{tax_data['関税']:,.0f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.warning("関税: データなし")
                    
                    with col2:
                        if tax_data['消費税'] is not None and tax_data['消費税'] > 0:
                            st.metric("消費税", f"¥{tax_data['消費税']:,.0f}")
                        else:
                            st.warning("消費税: データなし")
                    
                    with col3:
                        if tax_data['地方消費税'] is not None and tax_data['地方消費税'] > 0:
                            st.metric("地方消費税", f"¥{tax_data['地方消費税']:,.0f}")
                        else:
                            st.warning("地方消費税: データなし")
                    
                    # 合計を表示
                    total_tax = sum([v for v in tax_data.values() if v is not None and v > 0])
                    if total_tax > 0:
                        st.info(f"**税金合計:** ¥{total_tax:,.0f}")
                    
                    continue  # display_data_preview をスキップ
                
                # その他のファイルタイプは汎用プレビューを表示
                display_data_preview(file_type, file_path, show_header=True)
    
    with tab3:
        st.markdown('<div class="section-header">処理実行</div>', unsafe_allow_html=True)
        
        # 必須ファイルのチェック
        required_files = ['fba', 'send_order', 'record_list']
        missing_files = [f for f in required_files if f not in st.session_state.uploaded_files]
        
        if missing_files:
            st.markdown('<div class="error-box">⚠️ 必須ファイルが不足しています</div>', unsafe_allow_html=True)
            st.write("不足しているファイル:")
            for f in missing_files:
                st.write(f"- {f}")
        else:
            # 元→円レート設定
            st.markdown("**為替レート設定**")
            col_cny1, col_cny2 = st.columns([2, 1])
            with col_cny1:
                cny_to_jpy_rate = st.number_input(
                    "元→円レート（円/元）",
                    min_value=0.0,
                    value=22.77,
                    step=0.01,
                    format="%.2f",
                    help="前日のTTS + 0.5（例: 22.53 + 0.5 = 23.03）"
                )
            with col_cny2:
                st.caption("例: 22.77")
            
            st.markdown("---")
            
            if st.button("🚀 処理を開始", type="primary", width='stretch'):
                # 為替レートをsession_stateに保存
                st.session_state.cny_to_jpy_rate = cny_to_jpy_rate
                
                with st.spinner("処理中..."):
                    try:
                        # プレビューデータを収集
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("データを収集しています...")
                        progress_bar.progress(20)
                        
                        # FBA指示書データと配送依頼No.を取得
                        fba_df = pd.DataFrame()
                        discount_df = pd.DataFrame()
                        shipping_request_no = ""
                        if 'fba' in st.session_state.uploaded_files:
                            fba_df = get_instruction_summary(st.session_state.uploaded_files['fba'])
                            # シート2から割引データを取得
                            discount_df = get_discount_from_instruction_sheet2(st.session_state.uploaded_files['fba'])
                            # ファイル名から配送依頼No.を抽出
                            fba_filename = st.session_state.file_info.get('fba', {}).get('filename', '')
                            if fba_filename:
                                fba_meta = parse_fba_filename_metadata(fba_filename)
                                shipping_request_no = fba_meta.get('shipping_request_no', '')
                        
                        if fba_df.empty:
                            st.error("FBA指示書のデータが取得できませんでした")
                            return
                        
                        asins = fba_df['ASIN'].dropna().astype(str).tolist()
                        
                        progress_bar.progress(40)
                        status_text.text("関連データを取得しています...")
                        
                        # Jancode データ
                        jancode_df = pd.DataFrame()
                        if 'jancode' in st.session_state.uploaded_files:
                            jancode_df = get_jancode_preview(asins, st.session_state.uploaded_files['jancode'])
                        
                        # send-order データ
                        send_order_df = pd.DataFrame()
                        if 'send_order' in st.session_state.uploaded_files:
                            send_order_df = get_send_order_preview(asins, st.session_state.uploaded_files['send_order'])
                        
                        # record-list データ
                        record_list_df = pd.DataFrame()
                        if 'record_list' in st.session_state.uploaded_files and not send_order_df.empty:
                            if '注文番号' in send_order_df.columns:
                                order_numbers = send_order_df['注文番号'].dropna().astype(str).unique().tolist()
                                send_order_matches = send_order_df[['ASIN', '注文番号']].drop_duplicates()
                                record_list_df = get_record_list_preview(
                                    order_numbers, 
                                    st.session_state.uploaded_files['record_list'],
                                    send_order_matches
                                )
                        
                        progress_bar.progress(60)
                        status_text.text("税金情報を取得しています...")
                        
                        # 税金データ
                        tax_data = {'関税': 0, '消費税': 0, '地方消費税': 0}
                        if 'import_permit' in st.session_state.uploaded_files:
                            tax_data = extract_tax_from_pdf(st.session_state.uploaded_files['import_permit'])
                        
                        progress_bar.progress(80)
                        status_text.text("結果を計算しています...")
                        
                        # メール情報
                        email_data = st.session_state.get('email_data', {})
                        
                        # 処理を実行
                        # オプション費用配分を取得
                        option_distribution = st.session_state.get('option_distribution', {})
                        
                        results_df = process_data_from_previews(
                            fba_df,
                            jancode_df,
                            send_order_df,
                            record_list_df,
                            tax_data,
                            email_data,
                            shipping_request_no,
                            cny_to_jpy_rate,
                            discount_df,
                            option_distribution
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("処理完了！")
                        
                        st.session_state.processing_results = results_df
                        
                        st.markdown('<div class="success-box">✅ 処理が完了しました！</div>', unsafe_allow_html=True)
                        st.info(f"**処理された商品数:** {len(results_df)} 個")
                        
                        # エラーがあれば表示
                        if hasattr(results_df, 'attrs') and 'errors' in results_df.attrs:
                            for error in results_df.attrs['errors']:
                                st.warning(error)
                        
                    except Exception as e:
                        st.markdown('<div class="error-box">❌ エラーが発生しました</div>', unsafe_allow_html=True)
                        st.error(str(e))
                        import traceback
                        st.code(traceback.format_exc())
    
    with tab4:
        st.markdown('<div class="section-header">結果表示</div>', unsafe_allow_html=True)
        
        if st.session_state.processing_results is not None and not st.session_state.processing_results.empty:
            results_df = st.session_state.processing_results
            
            # 出力したい情報の列順
            desired_columns = [
                'ASIN',
                'SKU',
                '商品名',
                '原価(円)',
                '数量',
                '購入単価（元）',
                '中国国内送料（元）',
                '商品1個あたりのオプション費用（円）',
                '商品1個あたりのオプション費用（元）',
                '商品1個につき割引額（円）',
                '商品1個につき割引額（元）',
                '商品1個あたり国際送料（円）',
                '商品1個あたり国際送料（元）',
                '商品1個あたり関税（円）',
                '商品1個あたり消費税（円）',
                '商品1個あたり地方消費税（円）'
            ]
            
            # 利用可能な列だけを表示
            available_cols = [col for col in desired_columns if col in results_df.columns]
            other_cols = [col for col in results_df.columns if col not in available_cols]
            display_df = results_df[available_cols + other_cols].copy()
            
            st.markdown("---")
            st.subheader("📊 処理結果")
            st.caption(f"処理された商品数: {len(display_df)} 個")
            
            # 原価計算に使用される列を定義
            cost_calculation_columns = [
                '購入単価（元）',
                '中国国内送料（元）',
                '商品1個あたりのオプション費用（円）',
                '商品1個あたり国際送料（円）',
                '商品1個あたり関税（円）'
            ]
            
            # スタイル適用関数
            def highlight_cost_columns(df):
                """原価計算に使用される列をハイライト"""
                # 基本スタイル（全セル）
                styles = pd.DataFrame('', index=df.index, columns=df.columns)
                
                # 原価計算列のスタイル - 落ち着いた青緑（ティール）のグラデーション
                for col in cost_calculation_columns:
                    if col in df.columns:
                        styles[col] = 'background-color: #E8F4F8; border-left: 3px solid #4A90A4; font-weight: 500;'
                
                # 原価列のスタイル - より濃いブルーグレー
                if '原価(円)' in df.columns:
                    styles['原価(円)'] = 'background-color: #D1E7F0; border-left: 3px solid #2E5266; font-weight: 600;'
                
                return styles
            
            # スタイル適用
            styled_df = display_df.style.apply(highlight_cost_columns, axis=None)
            
            # 数値フォーマット
            format_dict = {}
            for col in display_df.columns:
                if '円' in col or '元' in col or '原価' in col:
                    format_dict[col] = '{:,.2f}'
                elif col == '数量':
                    format_dict[col] = '{:,.0f}'
            
            styled_df = styled_df.format(format_dict, na_rep='-')
            
            # 凡例を表示
            st.markdown("""
            <div style="margin-bottom: 15px; padding: 10px; background-color: #F8FAFB; border-radius: 8px; border-left: 4px solid #4A90A4;">
                <div style="display: flex; align-items: center; gap: 20px; flex-wrap: wrap;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="width: 20px; height: 20px; background-color: #E8F4F8; border-left: 3px solid #4A90A4; border-radius: 3px;"></div>
                        <span style="font-size: 13px; color: #2E5266;">原価計算に使用</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="width: 20px; height: 20px; background-color: #D1E7F0; border-left: 3px solid #2E5266; border-radius: 3px;"></div>
                        <span style="font-size: 13px; color: #2E5266; font-weight: 600;">原価（合計）</span>
                    </div>
                    <span style="font-size: 12px; color: #6B7280; margin-left: 10px;">
                        💡 原価 = 購入単価 + 中国国内送料 + オプション費用 + 国際送料 + 関税
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(styled_df, width='stretch', height=400)
            
            # 原価計算の検証セクション
            st.markdown("---")
            st.subheader("🧮 原価計算の検証")
            st.caption("任意のASINを選択して、原価がどのように計算されているかを確認できます")
            
            # ASINを選択
            available_asins = display_df['ASIN'].dropna().unique().tolist()
            if available_asins:
                selected_asin = st.selectbox(
                    "ASINを選択",
                    options=available_asins,
                    key="cost_verification_asin"
                )
                
                if selected_asin:
                    # 選択されたASINのデータを取得
                    selected_row = display_df[display_df['ASIN'] == selected_asin].iloc[0]
                    
                    # 各コンポーネントを取得
                    unit_price_cny = selected_row.get('購入単価（元）', 0) or 0
                    domestic_shipping_cny = selected_row.get('中国国内送料（元）', 0) or 0
                    qty = selected_row.get('数量', 1) or 1
                    option_fee_jpy = selected_row.get('商品1個あたりのオプション費用（円）', 0) or 0
                    international_shipping_jpy = selected_row.get('商品1個あたり国際送料（円）', 0) or 0
                    customs_jpy = selected_row.get('商品1個あたり関税（円）', 0) or 0
                    actual_cost = selected_row.get('原価(円)', 0) or 0
                    
                    # 為替レートを取得（session_stateから）
                    cny_to_jpy_rate = st.session_state.get('cny_to_jpy_rate', 22.77)
                    
                    # 各項目を計算
                    unit_price_jpy = unit_price_cny * cny_to_jpy_rate
                    domestic_shipping_per_item_jpy = (domestic_shipping_cny * cny_to_jpy_rate) / qty if qty > 0 else 0
                    
                    # 計算結果を表示
                    st.markdown(f"""
                    <div style="padding: 20px; background: linear-gradient(135deg, #F8FAFB 0%, #E8F4F8 100%); border-radius: 12px; border: 2px solid #4A90A4;">
                        <h4 style="color: #2E5266; margin-bottom: 15px;">📦 {selected_asin} の原価計算内訳</h4>
                        <p style="color: #6B7280; font-size: 14px; margin-bottom: 20px;">
                            <strong>商品名:</strong> {selected_row.get('商品名', 'N/A')}<br>
                            <strong>数量:</strong> {qty:,.0f}個
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # 計算式を5列で表示
                    cols = st.columns(5)
                    
                    components = [
                        {
                            "title": "①購入単価",
                            "value_cny": unit_price_cny,
                            "value_jpy": unit_price_jpy,
                            "formula": f"{unit_price_cny:,.2f}元 × {cny_to_jpy_rate}",
                            "icon": "💰"
                        },
                        {
                            "title": "②中国国内送料",
                            "value_cny": domestic_shipping_cny / qty if qty > 0 else 0,
                            "value_jpy": domestic_shipping_per_item_jpy,
                            "formula": f"{domestic_shipping_cny:,.2f}元 ÷ {qty:,.0f} × {cny_to_jpy_rate}",
                            "icon": "🚚"
                        },
                        {
                            "title": "③オプション費用",
                            "value_cny": option_fee_jpy / cny_to_jpy_rate if cny_to_jpy_rate > 0 else 0,
                            "value_jpy": option_fee_jpy,
                            "formula": "指示書 + 追加配分",
                            "icon": "⚙️"
                        },
                        {
                            "title": "④国際送料",
                            "value_cny": international_shipping_jpy / cny_to_jpy_rate if cny_to_jpy_rate > 0 else 0,
                            "value_jpy": international_shipping_jpy,
                            "formula": "体積按分",
                            "icon": "✈️"
                        },
                        {
                            "title": "⑤関税",
                            "value_cny": customs_jpy / cny_to_jpy_rate if cny_to_jpy_rate > 0 else 0,
                            "value_jpy": customs_jpy,
                            "formula": "総額 ÷ ASIN数 ÷ 数量",
                            "icon": "📋"
                        }
                    ]
                    
                    for col, comp in zip(cols, components):
                        with col:
                            col.markdown(f"""
                            <div style="padding: 15px; background-color: #E8F4F8; border-left: 4px solid #4A90A4; border-radius: 8px; height: 180px;">
                                <div style="font-size: 24px; text-align: center; margin-bottom: 8px;">{comp['icon']}</div>
                                <div style="font-size: 11px; color: #6B7280; font-weight: 600; text-align: center; margin-bottom: 8px;">{comp['title']}</div>
                                <div style="font-size: 18px; color: #2E5266; font-weight: 700; text-align: center; margin-bottom: 8px;">¥{comp['value_jpy']:,.2f}</div>
                                <div style="font-size: 9px; color: #6B7280; text-align: center; line-height: 1.4;">{comp['formula']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # 計算式と結果
                    calculated_cost = (
                        unit_price_jpy +
                        domestic_shipping_per_item_jpy +
                        option_fee_jpy +
                        international_shipping_jpy +
                        customs_jpy
                    )
                    
                    # 最終結果を表示
                    col1, col2, col3 = st.columns([2, 1, 2])
                    
                    with col1:
                        st.markdown(f"""
                        <div style="padding: 20px; background-color: #FFF; border-radius: 8px; border: 2px solid #E5E7EB;">
                            <div style="font-size: 14px; color: #6B7280; margin-bottom: 10px;">計算式:</div>
                            <div style="font-size: 12px; color: #2E5266; line-height: 2;">
                                ¥{unit_price_jpy:,.2f}<br>
                                + ¥{domestic_shipping_per_item_jpy:,.2f}<br>
                                + ¥{option_fee_jpy:,.2f}<br>
                                + ¥{international_shipping_jpy:,.2f}<br>
                                + ¥{customs_jpy:,.2f}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                        <div style="text-align: center; padding-top: 60px;">
                            <div style="font-size: 32px; color: #4A90A4;">=</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        difference = calculated_cost - actual_cost
                        match_status = "✅ 一致" if abs(difference) < 0.01 else "⚠️ 差異あり"
                        match_color = "#10B981" if abs(difference) < 0.01 else "#F59E0B"
                        
                        st.markdown(f"""
                        <div style="padding: 20px; background: linear-gradient(135deg, #D1E7F0 0%, #B8DAE8 100%); border-radius: 8px; border: 3px solid #2E5266;">
                            <div style="font-size: 14px; color: #2E5266; font-weight: 600; margin-bottom: 5px;">計算結果</div>
                            <div style="font-size: 28px; color: #1F2937; font-weight: 700; margin-bottom: 10px;">¥{calculated_cost:,.2f}</div>
                            <div style="font-size: 12px; color: #6B7280; margin-bottom: 5px;">表の原価: ¥{actual_cost:,.2f}</div>
                            <div style="font-size: 13px; color: {match_color}; font-weight: 600;">{match_status}</div>
                            {f'<div style="font-size: 11px; color: #6B7280;">差分: ¥{difference:,.2f}</div>' if abs(difference) >= 0.01 else ''}
                        </div>
                        """, unsafe_allow_html=True)
            
            # ダウンロードボタン
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                display_df.to_excel(writer, index=False, sheet_name='処理結果')
            
            st.download_button(
                label="📥 Excelファイルをダウンロード",
                data=output.getvalue(),
                file_name="FBA処理結果.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.info("まず「処理実行」タブで処理を実行してください。")


if __name__ == "__main__":
    main()

