"""
Google Sheets synchronization for myShop AI.

The service uses gspread with a Google service-account JSON key. Credentials can
come from UserSettings.google_credentials or GOOGLE_SERVICE_ACCOUNT_JSON.
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
]

SALES_HEADERS = [
    "Date",
    "Consignment ID",
    "Order ID",
    "Customer Name",
    "Phone",
    "Amount",
    "Delivery Cost",
    "Per Order Profit",
]

CUSTOMER_HEADERS = [
    "Phone",
    "Customer Name",
    "First Order",
    "Last Order",
    "Orders",
    "Revenue",
    "Avg Order",
]

INVESTMENT_HEADERS = ["Date", "Category", "Amount", "Quantity"]

DAILY_HEADERS = ["Date", "Orders", "Revenue", "Delivery Cost", "Profit", "Average Order"]

INVENTORY_HEADERS = [
    "SKU",
    "Name",
    "Category",
    "Supplier",
    "Unit Cost",
    "Sell Price",
    "Current Stock",
    "Reorder Level",
    "Status",
    "Notes",
]

ORDER_HEADERS = SALES_HEADERS


def _get_client(credentials_json: Optional[str] = None):
    """Return (gspread_client, error_string)."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        return None, "gspread / google-auth not installed. Add them to requirements.txt."

    creds_data = None
    if credentials_json:
        try:
            creds_data = json.loads(credentials_json)
        except json.JSONDecodeError:
            return None, "Invalid JSON in credentials."

    if not creds_data:
        env_creds = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
        if env_creds:
            try:
                creds_data = json.loads(env_creds)
            except json.JSONDecodeError:
                return None, "GOOGLE_SERVICE_ACCOUNT_JSON env var is not valid JSON."

    if not creds_data:
        return None, (
            "No Google credentials found. Upload a service-account JSON in Settings, "
            "or set GOOGLE_SERVICE_ACCOUNT_JSON."
        )

    try:
        creds = Credentials.from_service_account_info(creds_data, scopes=_SCOPES)
        return gspread.authorize(creds), None
    except Exception as exc:
        return None, f"Credentials error: {exc}"


def _clean_number(value) -> float:
    if value is None:
        return 0.0
    try:
        text = str(value).replace("\u09F3", "").replace(",", "").strip()
        return float(text) if text else 0.0
    except (TypeError, ValueError):
        return 0.0


def _get_or_create_worksheet(spreadsheet, title: str, rows: int, cols: int):
    try:
        worksheet = spreadsheet.worksheet(title)
        worksheet.clear()
        return worksheet
    except Exception:
        return spreadsheet.add_worksheet(title=title, rows=max(rows, 100), cols=max(cols, 10))


def _write_tab(spreadsheet, title: str, headers: list[str], data_rows: list[list]) -> int:
    rows = [headers, *data_rows]
    worksheet = _get_or_create_worksheet(spreadsheet, title, len(rows) + 10, len(headers))
    worksheet.update(rows, value_input_option="RAW")
    return max(0, len(rows) - 1)


def _worksheet_records(spreadsheet, names: list[str]) -> list[dict]:
    for name in names:
        try:
            return spreadsheet.worksheet(name).get_all_records()
        except Exception:
            continue
    return []


def sync_shop_to_sheet(
    sheet_id: str,
    orders: list[dict],
    investments: list[dict],
    customers: list[dict],
    daily: list[dict],
    products: Optional[list[dict]] = None,
    credentials_json: Optional[str] = None,
) -> dict:
    client, error = _get_client(credentials_json)
    if error:
        return {"success": False, "error": error}

    try:
        spreadsheet = client.open_by_key(sheet_id)

        sales_count = _write_tab(
            spreadsheet,
            "Sales_Data",
            SALES_HEADERS,
            [
                [
                    order.get("date", ""),
                    order.get("consignment_id", "") or "",
                    order.get("order_id", "") or "",
                    order.get("customer_name", "") or "",
                    order.get("customer_phone", "") or "",
                    round(order.get("amount", 0) or 0, 2),
                    round(order.get("delivery_cost", 0) or 0, 2),
                    round(order.get("profit", 0) or 0, 2),
                ]
                for order in orders
            ],
        )

        customer_count = _write_tab(
            spreadsheet,
            "Customers",
            CUSTOMER_HEADERS,
            [
                [
                    customer.get("phone", "") or "",
                    customer.get("name", "") or "",
                    customer.get("first_order", "") or "",
                    customer.get("last_order", "") or "",
                    customer.get("total_orders", 0) or 0,
                    round(customer.get("total_revenue", 0) or 0, 2),
                    round(customer.get("avg_order", 0) or 0, 2),
                ]
                for customer in customers
            ],
        )

        investment_count = _write_tab(
            spreadsheet,
            "Investments",
            INVESTMENT_HEADERS,
            [
                [
                    item.get("date", "") or "",
                    item.get("category", "") or "Other",
                    round(item.get("amount", 0) or 0, 2),
                    item.get("notes_quantity", item.get("notes", "")) or "",
                ]
                for item in investments
            ],
        )

        daily_count = _write_tab(
            spreadsheet,
            "Per Day Order",
            DAILY_HEADERS,
            [
                [
                    item.get("date", "") or "",
                    item.get("orders", 0) or 0,
                    round(item.get("revenue", 0) or 0, 2),
                    round(item.get("delivery_cost", 0) or 0, 2),
                    round(item.get("profit", 0) or 0, 2),
                    round(item.get("average_order", 0) or 0, 2),
                ]
                for item in daily
            ],
        )

        inventory_count = 0
        if products is not None:
            inventory_count = _write_tab(
                spreadsheet,
                "Inventory",
                INVENTORY_HEADERS,
                [
                    [
                        product.get("sku", "") or "",
                        product.get("name", "") or "",
                        product.get("category", "") or "",
                        product.get("supplier", "") or "",
                        round(product.get("unit_cost", 0) or 0, 2),
                        round(product.get("sell_price", 0) or 0, 2),
                        product.get("current_stock", 0) or 0,
                        product.get("reorder_level", 0) or 0,
                        product.get("status", "") or "",
                        product.get("notes", "") or "",
                    ]
                    for product in products
                ],
            )

        return {
            "success": True,
            "synced_rows": sales_count + investment_count + inventory_count,
            "sales_rows": sales_count,
            "customer_rows": customer_count,
            "investment_rows": investment_count,
            "daily_rows": daily_count,
            "inventory_rows": inventory_count,
            "error": "",
        }

    except Exception as exc:
        logger.error("Sheets sync error: %s", exc)
        return {"success": False, "synced_rows": 0, "error": str(exc)}


def import_shop_from_sheet(
    sheet_id: str,
    credentials_json: Optional[str] = None,
) -> dict:
    client, error = _get_client(credentials_json)
    if error:
        return {"success": False, "orders": [], "investments": [], "products": [], "error": error}

    try:
        spreadsheet = client.open_by_key(sheet_id)

        orders = []
        for row in _worksheet_records(spreadsheet, ["Sales_Data", "Orders"]):
            amount = _clean_number(row.get("Amount", row.get("Amount (\u09F3)", row.get("amount", 0))))
            if amount <= 0:
                continue
            delivery_cost = _clean_number(
                row.get("Delivery Cost", row.get("Delivery Cost (\u09F3)", row.get("delivery_cost", 0)))
            )
            product_cost = _clean_number(row.get("Product Cost", row.get("product_cost", 0)))
            sheet_profit = row.get("Per Order Profit", row.get("Profit", row.get("Profit (\u09F3)", None)))
            profit = _clean_number(sheet_profit)
            if sheet_profit in (None, ""):
                profit = round(amount - delivery_cost - product_cost, 2)

            orders.append(
                {
                    "date": str(row.get("Date", row.get("date", ""))).strip(),
                    "consignment_id": str(row.get("Consignment ID", row.get("consignment_id", ""))).strip(),
                    "order_id": str(row.get("Order ID", row.get("order_id", ""))).strip(),
                    "customer_name": str(row.get("Customer Name", row.get("customer_name", ""))).strip(),
                    "customer_phone": str(
                        row.get("Phone", row.get("Customer Phone", row.get("customer_phone", "")))
                    ).strip(),
                    "product_name": str(
                        row.get("Product", row.get("Product Name", row.get("product_name", "")))
                    ).strip(),
                    "amount": amount,
                    "delivery_cost": delivery_cost,
                    "product_cost": product_cost,
                    "profit": profit,
                    "status": str(row.get("Status", row.get("status", "delivered"))).lower() or "delivered",
                    "notes": str(row.get("Notes", row.get("notes", ""))).strip(),
                }
            )

        investments = []
        for row in _worksheet_records(spreadsheet, ["Investments"]):
            amount = _clean_number(row.get("Amount", row.get("amount", 0)))
            if amount <= 0:
                continue
            investments.append(
                {
                    "date": str(row.get("Date", row.get("date", ""))).strip(),
                    "category": str(row.get("Category", row.get("category", "Other"))).strip() or "Other",
                    "amount": amount,
                    "notes": str(row.get("Quantity", row.get("Notes", row.get("notes", "")))).strip(),
                }
            )

        products = []
        for row in _worksheet_records(spreadsheet, ["Inventory", "Products"]):
            name = str(row.get("Name", row.get("Product Name", row.get("name", "")))).strip()
            if not name:
                continue
            products.append(
                {
                    "sku": str(row.get("SKU", row.get("sku", ""))).strip(),
                    "name": name,
                    "category": str(row.get("Category", row.get("category", ""))).strip(),
                    "supplier": str(row.get("Supplier", row.get("supplier", ""))).strip(),
                    "unit_cost": _clean_number(row.get("Unit Cost", row.get("unit_cost", 0))),
                    "sell_price": _clean_number(row.get("Sell Price", row.get("sell_price", 0))),
                    "current_stock": int(_clean_number(row.get("Current Stock", row.get("current_stock", 0)))),
                    "reorder_level": int(_clean_number(row.get("Reorder Level", row.get("reorder_level", 0)))),
                    "notes": str(row.get("Notes", row.get("notes", ""))).strip(),
                }
            )

        return {
            "success": True,
            "orders": orders,
            "investments": investments,
            "products": products,
            "error": "",
        }

    except Exception as exc:
        logger.error("Sheets import error: %s", exc)
        return {"success": False, "orders": [], "investments": [], "products": [], "error": str(exc)}


def sync_orders_to_sheet(
    sheet_id: str,
    orders: list[dict],
    credentials_json: Optional[str] = None,
) -> dict:
    return sync_shop_to_sheet(
        sheet_id=sheet_id,
        orders=orders,
        investments=[],
        customers=[],
        daily=[],
        products=None,
        credentials_json=credentials_json,
    )


def import_from_sheet(
    sheet_id: str,
    credentials_json: Optional[str] = None,
) -> dict:
    result = import_shop_from_sheet(sheet_id, credentials_json)
    return {
        "success": result.get("success", False),
        "orders": result.get("orders", []),
        "error": result.get("error", ""),
    }
