"""
email_service.py — Daily report scheduler and SMTP email for myShop AI

Reads SMTP_* environment variables to send real emails.
If SMTP_USER / SMTP_PASSWORD are not set, reports are logged to stdout
(useful for development).
"""

import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy.orm import Session

import database
import models

logger = logging.getLogger(__name__)

# ─── SMTP config ─────────────────────────────────────────────────────────────
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")

REPORT_HOUR = int(os.getenv("REPORT_HOUR", "8"))
REPORT_MINUTE = int(os.getenv("REPORT_MINUTE", "0"))


# ─── Email transport ─────────────────────────────────────────────────────────

APPSCRIPT_WEBHOOK_URL_GLOBAL = os.getenv("APPSCRIPT_WEBHOOK_URL", "")

def send_email(to: str, subject: str, html: str, appscript_url: Optional[str] = None) -> bool:
    """Send HTML email via Apps Script Webhook (if configured), else SMTP, else console mock."""
    
    webhook_url = appscript_url or APPSCRIPT_WEBHOOK_URL_GLOBAL
    
    # Try Apps Script Webhook first
    if webhook_url:
        import requests
        try:
            payload = {
                "to": to,
                "subject": subject,
                "body": html,
                "is_html": True
            }
            res = requests.post(webhook_url, json=payload, timeout=15)
            if res.status_code == 200 and res.json().get("status") == "success":
                logger.info(f"Email sent via Apps Script → {to}")
                return True
            else:
                logger.error(f"Apps Script Email failed. Status: {res.status_code} Body: {res.text}")
                return False
        except Exception as exc:
            logger.error(f"Failed to send email via Apps Script: {exc}")
            return False

    # Try SMTP next
    if SMTP_USER and SMTP_PASSWORD:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = SMTP_USER
            msg["To"] = to
            msg.attach(MIMEText(html, "html", "utf-8"))

            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                server.ehlo()
                server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.sendmail(SMTP_USER, to, msg.as_bytes())

            logger.info(f"Email sent via SMTP → {to}")
            return True
        except Exception as exc:
            logger.error(f"Failed to send email via SMTP to {to}: {exc}")
            return False
            
    # Fallback to console mock
    logger.info(f"[EMAIL MOCK] To={to!r}  Subject={subject!r}")
    print(f"\n{'='*60}\n[EMAIL MOCK] To: {to}\nSubject: {subject}\n{'='*60}\n")
    return True


# ─── HTML template ────────────────────────────────────────────────────────────

def build_report_html(
    business_name: str,
    date_str: str,
    total_orders: int,
    unique_customers: int,
    total_revenue: float,
    total_delivery: float,
    total_profit: float,
    aov: float,
    ai_insights: str,
) -> str:
    return f"""
    <div style="font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; background-color: #f4f7f6; padding: 40px 20px; color: #333;">
        <div style="max-width: 600px; margin: 0 auto; background: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
            
            <!-- Header -->
            <div style="background: linear-gradient(135deg, #0891b2, #8b5cf6); padding: 30px 20px; text-align: center; color: white;">
                <h1 style="margin: 0; font-size: 28px; font-weight: bold; letter-spacing: 1px;">{business_name}</h1>
                <p style="margin: 5px 0 0 0; font-size: 16px; opacity: 0.9;">Daily Performance Report</p>
                <div style="margin-top: 15px; display: inline-block; background: rgba(255,255,255,0.2); padding: 6px 15px; border-radius: 20px; font-weight: 600; font-size: 14px;">
                    {date_str}
                </div>
            </div>

            <!-- Content -->
            <div style="padding: 30px;">
                <h2 style="font-size: 18px; color: #1f2937; margin-bottom: 20px; border-bottom: 2px solid #f3f4f6; padding-bottom: 10px;">Metrics Overview</h2>
                
                <table width="100%" cellpadding="12" cellspacing="0" style="border-collapse: collapse; margin-bottom: 30px;">
                    <tbody>
                        <tr style="border-bottom: 1px solid #e5e7eb;">
                            <td style="color: #6b7280; font-weight: 500;">Total Orders</td>
                            <td align="right" style="font-weight: 600; color: #111827;">{total_orders}</td>
                        </tr>
                        <tr style="background-color: #f9fafb; border-bottom: 1px solid #e5e7eb;">
                            <td style="color: #6b7280; font-weight: 500;">Unique Customers</td>
                            <td align="right" style="font-weight: 600; color: #111827;">{unique_customers}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #e5e7eb;">
                            <td style="color: #6b7280; font-weight: 500;">Gross Revenue</td>
                            <td align="right" style="font-weight: 600; color: #111827;">৳{total_revenue:,.0f}</td>
                        </tr>
                        <tr style="background-color: #f9fafb; border-bottom: 1px solid #e5e7eb;">
                            <td style="color: #6b7280; font-weight: 500;">Delivery Costs</td>
                            <td align="right" style="font-weight: 600; color: #ef4444;">-৳{total_delivery:,.0f}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #e5e7eb;">
                            <td style="color: #6b7280; font-weight: 500;">Avg. Order Value</td>
                            <td align="right" style="font-weight: 600; color: #111827;">৳{aov:,.2f}</td>
                        </tr>
                        <tr>
                            <td style="color: #111827; font-weight: 700; font-size: 16px; padding-top: 20px;">Net Profit</td>
                            <td align="right" style="font-weight: 800; font-size: 18px; color: #10b981; padding-top: 20px;">৳{total_profit:,.0f}</td>
                        </tr>
                    </tbody>
                </table>

                <h2 style="font-size: 18px; color: #1f2937; margin-bottom: 15px; display: flex; align-items: center;">
                    <span style="background: #e0e7ff; color: #4f46e5; padding: 4px 8px; border-radius: 6px; font-size: 12px; margin-right: 10px; font-weight: bold;">AI</span>
                    Gemma 4 Insights
                </h2>
                <div style="background-color: #f8fafc; border-left: 4px solid #8b5cf6; padding: 20px; border-radius: 4px; color: #334155; line-height: 1.6; font-size: 14.5px; white-space: pre-wrap;">{ai_insights}</div>
                
            </div>

            <!-- Footer -->
            <div style="background-color: #f1f5f9; padding: 20px; text-align: center; color: #64748b; font-size: 12px;">
                <p style="margin: 0;">Automated by <b>myShop AI</b></p>
                <p style="margin: 5px 0 0 0;">Powered by Google Gemma 4</p>
            </div>
        </div>
    </div>
    """


# ─── Scheduled job ────────────────────────────────────────────────────────────

def generate_daily_reports():
    """APScheduler job: build and send daily reports to all active users."""
    from ai_service import generate_insights  # local import to avoid circular
    from main import _build_ai_context  # local import — main is fully loaded at job runtime

    db: Session = database.SessionLocal()
    date_str = datetime.now().strftime("%Y-%m-%d")

    try:
        users = db.query(models.User).filter(models.User.is_active == True).all()
        logger.info(f"Daily report job running for {len(users)} user(s)")

        for user in users:
            settings = db.query(models.UserSettings).filter(
                models.UserSettings.user_id == user.id
            ).first()
            
            user_report_time = settings.report_time if (settings and settings.report_time) else "20:00"
            current_time = datetime.now().strftime("%H:%M")
            if user_report_time != current_time:
                continue

            all_orders = db.query(models.Order).filter(models.Order.owner_id == user.id).all()
            if not all_orders:
                continue

            today_orders = [o for o in all_orders if o.date == date_str]
            total_rev = sum(o.amount for o in today_orders)
            total_delivery = sum(o.delivery_cost for o in today_orders)
            total_profit = sum(o.profit for o in today_orders)
            n_orders = len(today_orders)
            today_phones = {o.customer_phone for o in today_orders if o.customer_phone}
            aov = total_rev / n_orders if n_orders else 0

            products = db.query(models.Product).filter(models.Product.owner_id == user.id).all()
            investments = db.query(models.Investment).filter(models.Investment.owner_id == user.id).all()
            ai_context = _build_ai_context(
                business_name=user.business_name,
                orders=all_orders,
                investments=investments,
                products=products,
            )

            ai_text = generate_insights(
                business_name=user.business_name,
                context_data=ai_context,
                report_scope="daily",
            )

            to_email = (settings and settings.notification_email) or user.email

            html = build_report_html(
                business_name=user.business_name,
                date_str=date_str,
                total_orders=n_orders,
                unique_customers=len(today_phones),
                total_revenue=total_rev,
                total_delivery=total_delivery,
                total_profit=total_profit,
                aov=aov,
                ai_insights=ai_text,
            )

            send_email(
                to=to_email,
                subject=f"{user.business_name} Daily Report — {date_str}",
                html=html,
                appscript_url=settings.appscript_webhook_url if settings else None,
            )

    finally:
        db.close()


def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(generate_daily_reports, "cron", minute="*")
    scheduler.start()
    logger.info("Email scheduler started — checking user report times every minute.")
