#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FBA注文処理システム
Excelファイルから注文情報を抽出し、原価計算を行う
"""

import pandas as pd
import re
import os
from pathlib import Path
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Tuple, Optional
import json
from io import BytesIO, StringIO

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

class FBAProcessor:
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.exchange_rate = None
        self.shipping_fee_table = None
        self.jancode_data = None
        self.record_list_data = None
        self.send_order_data = None
        self.fba_instructions = []
        self.import_permit_data = {}
        self.email_data = {}
        self.option_cost_distribution = {}  # オプション費用の分配情報
        self.errors = []
        self.skipped_asins = []
        self.missing_data = []
        self.shipping_config = None
        self.load_shipping_fee_config()
    
    def load_shipping_fee_config(self):
        """配送代行手数料の設定を読み込む"""
        config_file = self.base_dir / "shipping_fee_config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    self.shipping_config = json.load(f)
            except Exception as e:
                print(f"配送代行手数料設定読み込みエラー: {e}")
                self.shipping_config = None
        
    def _load_table_with_html_fallback(self, filepath: str, nrows: Optional[int] = None) -> pd.DataFrame:
        """Excel読み込みに失敗した場合、HTMLテーブルとして再試行"""
        last_error = None
        with open(filepath, 'rb') as f:
            raw = f.read()
        header = raw[:200].lower()
        if b'<html' in header or b'<!doctype html' in header:
            text = raw.decode('utf-8', errors='replace')
            tables = pd.read_html(StringIO(text))
            if tables:
                df = tables[0]
                if nrows:
                    return df.head(nrows)
                return df
            raise Exception("HTMLテーブルを解析できませんでした")
        for engine in ['openpyxl', 'xlrd']:
            try:
                return pd.read_excel(BytesIO(raw), engine=engine, nrows=nrows)
            except Exception as exc:
                last_error = exc
        text = raw.decode('utf-8', errors='replace')
        tables = pd.read_html(StringIO(text))
        if tables:
            df = tables[0]
            if nrows:
                return df.head(nrows)
            return df
        if last_error:
            raise last_error
        raise Exception("ファイルが読み込めませんでした")

    def parse_fba_filename(self, filename: str) -> Dict[str, str]:
        """
        FBA指示書のファイル名から情報を抽出
        ex: FBA指令20251107-2334 663094 FBA15G328FDT VIP6003078637.xls
        配送依頼番号：663094
        配達プラン No.:FBA15G328FDT
        追跡番号: VIP6003078637
        """
        pattern = r'FBA指令\d{8}-\d{4}\s+(\d+)\s+([A-Z0-9]+)\s+([A-Z0-9]+)\.xls'
        match = re.search(pattern, filename)
        if match:
            return {
                'shipping_request_no': match.group(1),
                'delivery_plan_no': match.group(2),
                'tracking_number': match.group(3)
            }
        return {}
    
    def parse_send_order_filename(self, filename: str) -> str:
        """
        send-order-list のファイル名から日付を抽出
        ex: send-order-list - 2025-12-26T223921.267.xls
        """
        pattern = r'send-order-list\s*-\s*(\d{4}-\d{2}-\d{2})'
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
        return ""
    
    def get_exchange_rate(self, date_str: str) -> float:
        """
        為替レートを取得
        前日のTTS + 0.5
        """
        try:
            # 日付をパース
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            prev_date = date_obj - timedelta(days=1)
            prev_date_str = prev_date.strftime("%Y-%m-%d")
            
            # レート取得API
            url = "https://www.murc-kawasesouba.jp/fx/past_3month.php"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200 and BeautifulSoup:
                # HTMLをパースしてTTSを取得（実際のHTML構造に合わせて調整が必要）
                soup = BeautifulSoup(response.text, 'html.parser')
                # 実際のHTML構造に合わせてTTSを抽出
                # 暫定的に固定値を使用（実際の実装ではHTMLから抽出）
                tts = 22.53
            else:
                tts = 22.53  # デフォルト値
            
            return tts + 0.5
        except Exception as e:
            print(f"為替レート取得エラー: {e} (デフォルト値を使用)")
            return 23.03  # デフォルト値
    
    def find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """複数の可能な列名から実際に存在する列を探す"""
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    def load_jancode(self, filepath: str) -> pd.DataFrame:
        """Jancode.xlsxを読み込む"""
        try:
            df = self._load_table_with_html_fallback(filepath)
            df.columns = df.columns.str.strip()
            self.jancode_data = df
            return df
        except Exception as e:
            print(f"Jancode.xlsx読み込みエラー: {e}")
            return pd.DataFrame()
    
    def load_record_list(self, filepath: str) -> pd.DataFrame:
        """record-list.xlsを読み込む"""
        try:
            df = self._load_table_with_html_fallback(filepath)
            df.columns = df.columns.str.strip()
            self.record_list_data = df
            return df
        except Exception as e:
            print(f"record-list.xls読み込みエラー: {e}")
            return pd.DataFrame()
    
    def load_send_order_list(self, filepath: str) -> pd.DataFrame:
        """send-order-listを読み込む"""
        try:
            df = self._load_table_with_html_fallback(filepath)
            df.columns = df.columns.str.strip()
            self.send_order_data = df
            return df
        except Exception as e:
            print(f"send-order-list読み込みエラー: {e}")
            return pd.DataFrame()
    
    def load_fba_instruction(self, filepath: str) -> pd.DataFrame:
        """FBA指示書を読み込む"""
        try:
            df = self._load_table_with_html_fallback(filepath)
            df.columns = df.columns.str.strip()
            filename = os.path.basename(filepath)
            file_info = self.parse_fba_filename(filename)
            df['_file_info'] = json.dumps(file_info)
            df['_filename'] = filename
            return df
        except Exception as e:
            print(f"FBA指示書読み込みエラー: {e}")
            return pd.DataFrame()
    
    def load_manual_input_data(self, filepath: str):
        """
        手動入力データを読み込む（.txtファイル）
        形式: ASIN,オプション費用,割引額,関税,消費税
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split(',')
                    if len(parts) >= 5:
                        asin = parts[0].strip()
                        self.email_data[asin] = {
                            'option_cost': float(parts[1]) if parts[1] else 0,
                            'discount': float(parts[2]) if parts[2] else 0,
                        }
                        self.import_permit_data[asin] = {
                            'customs_duty': float(parts[3]) if parts[3] else 0,
                            'consumption_tax': float(parts[4]) if parts[4] else 0,
                        }
        except Exception as e:
            print(f"手動入力データ読み込みエラー: {e}")
    
    def load_import_permit_pdf(self, filepath: str, tracking_number: str = None):
        """
        輸入許可通知書PDFから関税・消費税情報を抽出
        PDFファイル名から追跡番号を取得し、該当するASINに関税・消費税を割り当てる
        """
        try:
            import PyPDF2
            
            # PDFを読み込む
            with open(filepath, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            
            # 関税、消費税、地方消費税を抽出
            # パターンは実際のPDFの構造に合わせて調整が必要
            customs_pattern = r'関税[：:]\s*(\d+(?:\.\d+)?)'
            consumption_pattern = r'消費税[：:]\s*(\d+(?:\.\d+)?)'
            local_consumption_pattern = r'地方消費税[：:]\s*(\d+(?:\.\d+)?)'
            
            customs_match = re.search(customs_pattern, text)
            consumption_match = re.search(consumption_pattern, text)
            local_consumption_match = re.search(local_consumption_pattern, text)
            
            customs_duty = float(customs_match.group(1)) if customs_match else 0
            consumption_tax = float(consumption_match.group(1)) if consumption_match else 0
            local_consumption_tax = float(local_consumption_match.group(1)) if local_consumption_match else 0
            
            # 追跡番号からASINを特定（ファイル名やPDF内容から）
            # ここでは暫定的に、tracking_numberをキーとして保存
            # 実際の実装では、PDF内のASIN情報や他のファイルとの関連付けが必要
            if tracking_number:
                # 追跡番号をキーとして保存（後でASINと関連付ける）
                if tracking_number not in self.import_permit_data:
                    self.import_permit_data[tracking_number] = {}
                self.import_permit_data[tracking_number]['customs_duty'] = customs_duty
                self.import_permit_data[tracking_number]['consumption_tax'] = consumption_tax + local_consumption_tax
            
            return {
                'customs_duty': customs_duty,
                'consumption_tax': consumption_tax,
                'local_consumption_tax': local_consumption_tax
            }
        except ImportError:
            print("PyPDF2がインストールされていません。PDF読み込みをスキップします。")
            return None
        except Exception as e:
            print(f"PDF読み込みエラー: {e}")
            return None
    
    def load_option_cost_distribution(self, filepath: str):
        """
        オプション費用の分配情報を読み込む
        形式: 配送依頼No,ASIN,費用種別,金額
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        shipping_no = parts[0]
                        asin = parts[1]
                        cost_type = parts[2]
                        amount = float(parts[3])
                        
                        if shipping_no not in self.option_cost_distribution:
                            self.option_cost_distribution[shipping_no] = {}
                        if asin not in self.option_cost_distribution[shipping_no]:
                            self.option_cost_distribution[shipping_no][asin] = []
                        
                        self.option_cost_distribution[shipping_no][asin].append({
                            'type': cost_type,
                            'amount': amount
                        })
        except Exception as e:
            print(f"オプション費用分配情報読み込みエラー: {e}")
    
    def calculate_volume(self, length: float, width: float, height: float) -> float:
        """体積を計算（cm3）"""
        return length * width * height
    
    def get_size_category(self, length: float, width: float, height: float) -> str:
        """
        サイズ区分を決定
        配送代行手数料.pngの表に基づいて判定
        """
        max_dimension = max(length, width, height)
        sum_dimensions = length + width + height
        
        # 設定ファイルから読み込む
        if self.shipping_config and 'size_categories' in self.shipping_config:
            categories = self.shipping_config['size_categories']
            # 条件に合う最初のカテゴリを返す
            for cat_name, cat_config in categories.items():
                if (max_dimension <= cat_config['max_dimension'] and 
                    sum_dimensions <= cat_config['sum_dimensions']):
                    return cat_name
            return "大型"  # デフォルト
        
        # デフォルトの判定ロジック
        if max_dimension <= 25 and sum_dimensions <= 40:
            return "小型"
        elif max_dimension <= 30 and sum_dimensions <= 50:
            return "標準1"
        elif max_dimension <= 40 and sum_dimensions <= 60:
            return "標準2"
        elif max_dimension <= 50 and sum_dimensions <= 70:
            return "標準2d"
        else:
            return "大型"
    
    def get_shipping_agent_fee(self, size_category: str) -> float:
        """
        配送代行手数料を取得
        サイズ区分に基づいて料金を返す
        """
        # 設定ファイルから読み込む
        if self.shipping_config and 'size_categories' in self.shipping_config:
            if size_category in self.shipping_config['size_categories']:
                return self.shipping_config['size_categories'][size_category]['fee']
        
        # デフォルト値
        fee_table = {
            "小型": 500,
            "標準1": 600,
            "標準2": 700,
            "標準2d": 800,
            "大型": 1000
        }
        return fee_table.get(size_category, 0)
    
    def calculate_international_shipping(self, volume: float) -> float:
        """
        国際送料を計算（体積ベース）
        """
        # 設定ファイルから読み込む
        if self.shipping_config and 'international_shipping_rate' in self.shipping_config:
            rate = self.shipping_config['international_shipping_rate']
            return volume * rate
        
        # デフォルト値（1cm3あたり0.01ウォン）
        return volume * 0.01
    
    def get_actual_quantity(self, asin: str, order_qty: int, shipping_request_no: str) -> int:
        """
        実際の数量を取得
        特殊ケース: ASIN B0D7PXG7KLは20個セット
        """
        if asin == "B0D7PXG7KL":
            # 注文数量が2個の場合、実際は40個（20個セット×2）
            return order_qty * 20
        return order_qty
    
    def get_option_cost_for_asin(self, asin: str, shipping_request_no: str, total_qty: int) -> float:
        """
        オプション費用をASINごとに分配
        """
        option_cost = 0
        
        # オプション費用分配情報から取得
        if shipping_request_no in self.option_cost_distribution:
            if asin in self.option_cost_distribution[shipping_request_no]:
                for cost_item in self.option_cost_distribution[shipping_request_no][asin]:
                    option_cost += cost_item['amount']
        
        # email_dataからも取得（分配情報がない場合）
        if option_cost == 0 and asin in self.email_data:
            option_cost = self.email_data[asin].get('option_cost', 0)
        
        # 1個あたりの費用を計算
        if total_qty > 0:
            return option_cost / total_qty
        return 0
    
    def check_missing_data(self, asin: str, jancode_row, order_row, shipping_request_no: str):
        """必要なデータが揃っているかチェック"""
        missing = []
        
        if jancode_row is None:
            missing.append("Jancode.xlsxにASINが見つかりません")
        else:
            if jancode_row.get('長さ', 0) == 0 or jancode_row.get('幅', 0) == 0 or jancode_row.get('高さ', 0) == 0:
                missing.append("Jancode.xlsxに寸法情報がありません")
        
        if order_row is None:
            missing.append("send-order-listに注文が見つかりません")
        
        if shipping_request_no and shipping_request_no not in self.import_permit_data:
            # 関税・消費税データのチェック（ASIN単位ではなく配送依頼No単位の可能性もある）
            pass
        
        return missing
    
    def process_orders(self) -> pd.DataFrame:
        """
        メイン処理：すべてのデータを統合して出力データを作成
        """
        results = []
        self.errors = []
        self.skipped_asins = []
        self.missing_data = []
        
        # 1. 必要なファイルの存在確認
        # FBA指示書ファイルを探す（より柔軟なパターン）
        fba_files = []
        for pattern in ["FBA指示書*.xls", "FBA指示書*.xlsx", "FBA指令*.xls", "FBA指令*.xlsx", "*FBA*.xls", "*FBA*.xlsx"]:
            fba_files.extend(list(self.base_dir.glob(pattern)))
        
        # 重複を削除
        fba_files = list(set(fba_files))
        
        if not fba_files:
            self.errors.append("FBA指示書ファイルが見つかりません")
            print(f"Looking for FBA files in: {self.base_dir}")
            print(f"Files in directory: {list(self.base_dir.glob('*'))}")
            return pd.DataFrame()
        
        # send-order-list ファイルを探す
        send_order_files = []
        for pattern in ["send-order-list*.xls", "send-order-list*.xlsx", "*send*order*.xls", "*send*order*.xlsx"]:
            send_order_files.extend(list(self.base_dir.glob(pattern)))
        
        # 重複を削除
        send_order_files = list(set(send_order_files))
        
        if not send_order_files:
            self.errors.append("send-order-listファイルが見つかりません")
            print(f"Looking for send-order files in: {self.base_dir}")
            return pd.DataFrame()
        
        if self.jancode_data is None:
            self.errors.append("Jancode.xlsxが読み込まれていません")
            return pd.DataFrame()
        
        # 2. 各FBA指示書を処理
        fba_asin_map = {}  # FBA指示書のASINと数量を記録
        
        for fba_file in fba_files:
            fba_df = self.load_fba_instruction(fba_file)
            if fba_df.empty:
                continue
                
            file_info = json.loads(fba_df['_file_info'].iloc[0])
            shipping_request_no = file_info.get('shipping_request_no', '')
            
            # FBA指示書のASINと数量を記録（例外処理3用）
            for idx, row in fba_df.iterrows():
                asin = row.get('ASIN', '')
                fba_qty = row.get('数量', 0)
                if asin:
                    if shipping_request_no not in fba_asin_map:
                        fba_asin_map[shipping_request_no] = {}
                    fba_asin_map[shipping_request_no][asin] = fba_qty
        
        # 3. send-order-listを統合
        all_send_orders = pd.DataFrame()
        send_order_dates = {}
        
        for send_order_file in send_order_files:
            send_df = self.load_send_order_list(send_order_file)
            shipping_date = self.parse_send_order_filename(send_order_file.name)
            
            if not send_df.empty:
                send_df['_shipping_date'] = shipping_date
                all_send_orders = pd.concat([all_send_orders, send_df], ignore_index=True)
                send_order_dates[shipping_date] = send_order_file.name
        
        # 4. 各FBA指示書のASINを処理
        processed_asins = set()
        
        for fba_file in fba_files:
            fba_df = self.load_fba_instruction(fba_file)
            if fba_df.empty:
                continue
            
            file_info = json.loads(fba_df['_file_info'].iloc[0])
            shipping_request_no = file_info.get('shipping_request_no', '')
            delivery_plan_no = file_info.get('delivery_plan_no', '')
            tracking_number = file_info.get('tracking_number', '')
            
            # 発送日を取得（send-order-listのファイル名から）
            shipping_date = ""
            for date, filename in send_order_dates.items():
                if shipping_request_no in filename or tracking_number in filename:
                    shipping_date = date
                    break
            
            # 為替レートを取得
            if shipping_date:
                self.exchange_rate = self.get_exchange_rate(shipping_date)
            else:
                self.exchange_rate = self.get_exchange_rate(datetime.now().strftime("%Y-%m-%d"))
            
                # FBA指示書の各ASINを処理
                for idx, fba_row in fba_df.iterrows():
                    # 列名のバリエーションに対応
                    asin_col = self.find_column(fba_df, ['ASIN', 'asin', 'Asin'])
                    qty_col = self.find_column(fba_df, ['数量', '個数', 'quantity', 'Quantity'])
                    
                    asin = fba_row.get(asin_col, '') if asin_col else ''
                    fba_qty = fba_row.get(qty_col, 0) if qty_col else 0
                
                if not asin:
                    continue
                
                # Jancodeから商品情報を取得
                asin_col_jc = self.find_column(self.jancode_data, ['ASIN', 'asin', 'Asin'])
                if not asin_col_jc:
                    self.errors.append("Jancode.xlsxにASIN列が見つかりません")
                    continue
                
                jancode_matches = self.jancode_data[self.jancode_data[asin_col_jc] == asin]
                if jancode_matches.empty:
                    self.errors.append(f"例外処理1: ASIN {asin} - Jancode.xlsxに見つかりません")
                    continue
                
                jancode_row = jancode_matches.iloc[0]
                
                # 寸法チェック（列名のバリエーションに対応）
                length_col = self.find_column(self.jancode_data, ['長さ', '長', 'length', 'Length', 'L'])
                width_col = self.find_column(self.jancode_data, ['幅', '横', 'width', 'Width', 'W'])
                height_col = self.find_column(self.jancode_data, ['高さ', '高', 'height', 'Height', 'H'])
                
                length = jancode_row.get(length_col, 0) if length_col else 0
                width = jancode_row.get(width_col, 0) if width_col else 0
                height = jancode_row.get(height_col, 0) if height_col else 0
                
                if length == 0 or width == 0 or height == 0:
                    self.errors.append(f"ASIN {asin}: Jancode.xlsxに寸法情報がありません")
                    self.missing_data.append({
                        'ASIN': asin,
                        'missing': '寸法情報'
                    })
                    continue
                
                # send-order-listから該当ASINを探す
                asin_col_so = self.find_column(all_send_orders, ['ASIN', 'asin', 'Asin'])
                if not asin_col_so:
                    self.errors.append("send-order-listにASIN列が見つかりません")
                    continue
                
                matching_orders = all_send_orders[all_send_orders[asin_col_so] == asin]
                
                if matching_orders.empty:
                    # 例外処理1: 指示に記載されているがsend-order-listにない
                    order_no = fba_row.get('注文番号', '')
                    self.errors.append(
                        f"例外処理1: ASIN {asin} (注文番号: {order_no}) - "
                        f"send-order-listに見つかりません。該当する注文番号を含むsend-order-listをリクエストしてください。"
                    )
                    continue
                
                # 例外処理3: 指示書とsend-order-listで数量が異なる場合は指示書に合わせる
                qty_col_so = self.find_column(matching_orders, ['数量', '個数', 'quantity', 'Quantity'])
                if not qty_col_so:
                    self.errors.append(f"ASIN {asin}: send-order-listに数量列が見つかりません")
                    continue
                
                total_order_qty = matching_orders[qty_col_so].sum()
                if fba_qty > 0 and total_order_qty != fba_qty:
                    # 指示書の数量に合わせる
                    actual_qty = fba_qty
                else:
                    actual_qty = total_order_qty
                
                # 例外処理2: 数量0の商品
                if actual_qty == 0:
                    self.skipped_asins.append({
                        'ASIN': asin,
                        'reason': '数量が0のため計算不要'
                    })
                    continue
                
                # 実際の数量を取得（特殊ケース対応）
                actual_qty = self.get_actual_quantity(asin, actual_qty, shipping_request_no)
                
                # 各注文を処理（複数の注文がある場合）
                for order_idx, order_row in matching_orders.iterrows():
                    order_qty = order_row.get(qty_col_so, 0)
                    if order_qty == 0:
                        continue
                    
                    order_no_col = self.find_column(matching_orders, ['注文番号', '注文ID', 'order_no', 'Order No', '注文No'])
                    order_no = order_row.get(order_no_col, '') if order_no_col else ''
                    
                    # record-listから原価を取得
                    cost_jpy = 0
                    if self.record_list_data is not None and order_no:
                        # 注文番号で検索（列名のバリエーションに対応）
                        for col in self.record_list_data.columns:
                            if '注文' in str(col) or 'order' in str(col).lower():
                                record_matches = self.record_list_data[
                                    self.record_list_data[col] == order_no
                                ]
                                if not record_matches.empty:
                                    # 参考金額（JPY）を探す
                                    for amt_col in record_matches.columns:
                                        if '参考' in str(amt_col) and 'JPY' in str(amt_col):
                                            cost_jpy = record_matches.iloc[0][amt_col]
                                            break
                                    break
                    
                    # サイズ区分と配送代行手数料を取得
                    size_category = self.get_size_category(length, width, height)
                    shipping_agent_fee = self.get_shipping_agent_fee(size_category)
                    
                    # 体積と国際送料を計算
                    volume = self.calculate_volume(length, width, height)
                    intl_shipping_krw = self.calculate_international_shipping(volume)
                    
                    # 各種費用を取得（列名のバリエーションに対応）
                    unit_price_col = self.find_column(matching_orders, ['単価', '購入単価', 'unit_price', 'Unit Price', '価格'])
                    unit_price_krw = order_row.get(unit_price_col, 0) if unit_price_col else 0
                    
                    china_shipping_col = self.find_column(matching_orders, ['中国国内送料', '国内送料', 'china_shipping', 'China Shipping'])
                    china_shipping_krw_total = order_row.get(china_shipping_col, 0) if china_shipping_col else 0
                    
                    # オプション費用を取得（分配情報から）
                    option_cost_krw = self.get_option_cost_for_asin(asin, shipping_request_no, actual_qty)
                    
                    # 割引額を取得
                    discount_krw = self.email_data.get(asin, {}).get('discount', 0) / actual_qty if actual_qty > 0 else 0
                    
                    # 関税、消費税を取得
                    customs_duty_jpy = self.import_permit_data.get(asin, {}).get('customs_duty', 0) / actual_qty if actual_qty > 0 else 0
                    consumption_tax_jpy = self.import_permit_data.get(asin, {}).get('consumption_tax', 0) / actual_qty if actual_qty > 0 else 0
                    
                    # 国際送料を円に変換
                    intl_shipping_jpy = intl_shipping_krw * self.exchange_rate
                    
                    # 中国国内送料を円に変換（1個あたり）
                    china_shipping_jpy = (china_shipping_krw_total / actual_qty) * self.exchange_rate if actual_qty > 0 else 0
                    
                    # 原価計算（消費税、地方消費税は含めない）
                    cost_per_item = (
                        (cost_jpy / actual_qty) +  # 商品単価
                        china_shipping_jpy +  # 中国国内送料
                        (option_cost_krw * self.exchange_rate) +  # オプション費用
                        intl_shipping_jpy +  # 国際送料
                        customs_duty_jpy  # 関税
                    )
                    
                    # 結果を追加
                    sku_col = self.find_column(self.jancode_data, ['SKU', 'sku', 'Sku'])
                    product_name_col = self.find_column(self.jancode_data, ['商品名', '商品名称', 'product_name', 'Product Name', '名称'])
                    
                    result = {
                        'ASIN': asin,
                        'SKU': jancode_row.get(sku_col, '') if sku_col else '',
                        '商品名': jancode_row.get(product_name_col, '') if product_name_col else '',
                        '原価(円)': round(cost_per_item, 2),
                        '数量': actual_qty,
                        '購入単価（ウォン）': unit_price_krw,
                        '中国国内送料（円）': round(china_shipping_jpy, 2),
                        '商品1個あたりのオプション費用（ウォン）': round(option_cost_krw, 2),
                        '商品1個につき割引額（ウォン）': round(discount_krw, 2),
                        '商品1個あたり国際送料（ウォン）': round(intl_shipping_krw, 2),
                        '商品1個あたり関税（円）': round(customs_duty_jpy, 2),
                        '商品1個あたり消費税(円)': round(consumption_tax_jpy, 2),
                        '配送依頼No.': shipping_request_no,
                        '納品プランNo.': delivery_plan_no,
                        '追跡番号': tracking_number,
                        '発送日': shipping_date,
                        '配送代行手数料（円）': shipping_agent_fee,
                        'サイズ区分': size_category
                    }
                    results.append(result)
                    processed_asins.add(asin)
        
        # 例外処理2: send-order-listにあるが指示書にない商品
        for send_order_file in send_order_files:
            send_df = self.load_send_order_list(send_order_file)
            for idx, row in send_df.iterrows():
                asin = row.get('ASIN', '')
                order_qty = row.get('数量', 0)
                
                if asin and asin not in processed_asins:
                    # すべてのFBA指示書でこのASINが使われているかチェック
                    found_in_any_fba = False
                    for fba_file in fba_files:
                        fba_df = self.load_fba_instruction(fba_file)
                        if not fba_df.empty and asin in fba_df['ASIN'].values:
                            found_in_any_fba = True
                            break
                    
                    if not found_in_any_fba:
                        self.skipped_asins.append({
                            'ASIN': asin,
                            'reason': 'send-order-listにあるが指示書にない商品（または数量0）'
                        })
        
        return pd.DataFrame(results)

def main():
    processor = FBAProcessor(".")
    
    # 必要なファイルを読み込む
    jancode_file = processor.base_dir / "Jancode.xlsx"
    if jancode_file.exists():
        processor.load_jancode(str(jancode_file))
        print(f"Jancode.xlsxを読み込みました")
    else:
        print("警告: Jancode.xlsxが見つかりません")
    
    record_list_file = list(processor.base_dir.glob("record-list*.xls"))
    if record_list_file:
        processor.load_record_list(str(record_list_file[0]))
        print(f"record-list.xlsを読み込みました")
    else:
        print("警告: record-list.xlsが見つかりません")
    
    # 手動入力データを読み込む
    manual_input_file = processor.base_dir / "manual_input.txt"
    if manual_input_file.exists():
        processor.load_manual_input_data(str(manual_input_file))
        print(f"手動入力データを読み込みました")
    
    # オプション費用分配情報を読み込む
    option_dist_file = processor.base_dir / "option_cost_distribution.txt"
    if option_dist_file.exists():
        processor.load_option_cost_distribution(str(option_dist_file))
        print(f"オプション費用分配情報を読み込みました")
    
    # 処理を実行
    print("\n処理を開始します...")
    results_df = processor.process_orders()
    
    # エラーとスキップ情報を出力
    if processor.errors:
        print("\n=== エラー ===")
        for error in processor.errors:
            print(f"  - {error}")
    
    if processor.missing_data:
        print("\n=== 不足しているデータ ===")
        for item in processor.missing_data:
            print(f"  - ASIN {item['ASIN']}: {item['missing']}")
    
    if processor.skipped_asins:
        print("\n=== 計算しないASIN ===")
        for item in processor.skipped_asins:
            print(f"  - ASIN {item['ASIN']}: {item['reason']}")
    
    # 結果をExcelに出力
    if not results_df.empty:
        output_file = processor.base_dir / "output.xlsx"
        
        # 列の順序を指定
        column_order = [
            'ASIN', 'SKU', '商品名', '原価(円)', '数量',
            '購入単価（ウォン）', '中国国内送料（円）',
            '商品1個あたりのオプション費用（ウォン）',
            '商品1個につき割引額（ウォン）',
            '商品1個あたり国際送料（ウォン）',
            '商品1個あたり関税（円）',
            '商品1個あたり消費税(円)',
            '配送依頼No.', '納品プランNo.', '追跡番号', '発送日',
            '配送代行手数料（円）', 'サイズ区分'
        ]
        
        # 存在する列のみを選択
        available_columns = [col for col in column_order if col in results_df.columns]
        results_df = results_df[available_columns]
        
        results_df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"\n結果を {output_file} に出力しました")
        print(f"処理件数: {len(results_df)}件")
    else:
        print("\n処理するデータがありませんでした")

if __name__ == "__main__":
    main()

