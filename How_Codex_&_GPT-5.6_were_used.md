# AI Usage: Codex and GPT-5.6

## Project Context

This project uses a FastAPI backend and a React/Vite frontend to build a professional Shop Management Dashboard for `myShop`. The dashboard replaces the original Excel tracker and connects to Google Sheets for two-way synchronization across sales, customers, investments, daily performance, and inventory data.

## How Codex Was Used

Codex was used as the full-stack implementation agent inside the local project workspace. It helped inspect the existing repository structure, identify the current backend and frontend patterns, and implement the dashboard feature without introducing a separate framework or duplicate architecture.

Key Codex contributions included:

- Reading the existing FastAPI, SQLAlchemy, React, and Recharts code before making changes.
- Extending backend models for orders, investments, products, product cost, and inventory-aware sales.
- Adding API routes for dashboard KPIs, monthly analytics, investments, products, customers, and Google Sheets sync.
- Updating Google Sheets integration to support the `myShop` tab structure:
  - `Sales_Data`
  - `Customers`
  - `Investments`
  - `Per Day Order`
  - `Inventory`
- Preserving the spreadsheet formulas in backend logic:
  - Gross Profit = Revenue - Delivery Cost - Product Cost
  - Net Cash Flow = Gross Profit - Total Investment
  - Gross Margin = Gross Profit / Total Revenue
  - Average Order Value = Total Revenue / Total Orders
- Improving the React dashboard with KPI cards, charts, customer insights, investment breakdowns, and performance tables.
- Adding data-entry workflows for sales, expenses, products, and Google Sheets connection settings.

## How GPT-5.6 Was Used

GPT-5.6 was used for reasoning, planning, and implementation guidance around the business logic and user experience. It helped translate the original Excel/Google Sheets workflow into a web-based dashboard architecture.

GPT-5.6 assisted with:

- Breaking the feature specification into backend, frontend, database, and sync layers.
- Mapping spreadsheet columns to normalized application models.
- Designing KPI calculations so the dashboard matches the original workbook logic.
- Structuring a responsive admin UI with dashboard cards, charts, tables, and forms.
- Identifying edge cases such as missing delivery cost, optional product cost, duplicate imports, and sheet/manual data changes.
- Drafting implementation-ready code patterns for FastAPI routes, SQLAlchemy models, Google Sheets helpers, and React components.

## AI-Assisted Implementation Areas

The AI-assisted work focused on these areas:

- Backend schema and migration support.
- CRUD APIs for orders, investments, customers, and products.
- Dynamic customer CRM metrics generated from order history.
- Dashboard analytics and monthly aggregation endpoints.
- Google Sheets import/export helpers for live `myShop` synchronization.
- React dashboard layout, KPI cards, charts, tables, and form integration.
- Bangladeshi Taka currency formatting in the UI.

## Human Oversight

The project requirements, spreadsheet structure, and business rules were supplied by the user. AI assistance was used to accelerate implementation, organize the code changes, and ensure formulas and UI behavior matched the requested dashboard.

Sensitive credentials such as Google service-account JSON and API keys are not included in the repository. They must be configured by the project owner through environment variables or the Settings page.

## Summary

Codex handled the repository-aware coding workflow, while GPT-5.6 supported the reasoning, planning, and product design behind the implementation. Together, they were used to convert the existing `myShop` Excel/Google Sheets tracker into a maintainable web dashboard with synchronized data, accurate business KPIs, and a responsive admin interface.
