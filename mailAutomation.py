import smtplib
import os
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from groq import Groq

SENDER_EMAIL = "trashbotservices@gmail.com"
APP_PASSWORD = "YOUR GMAIL APP PASSWORD"   # Gmail APP password
USE_LLM = True

client = Groq(
    api_key="YOU API KEY"
)

# -------------------------------------------------
# format: Name_YYYY-MM-DD_HH-MM-SS_fall_detected.png
# -------------------------------------------------
def extract_event_date(screenshot_path):
    try:
        filename = os.path.basename(screenshot_path)
        parts = filename.split("_")
        raw_date = parts[1]  # YYYY-MM-DD
        return datetime.strptime(raw_date, "%Y-%m-%d").strftime("%d %b %Y")
    except Exception:
        return "Unknown Date"


# -------------------------------------------------
# LLM Mail Generator
# -------------------------------------------------
def generate_llm_mail(name, event, event_date):
    try:
        prompt = f"""
        Write a short, professional, system-generated notification email.

        Recipient name: {name}
        Event: {event}
        Date of detection: {event_date}

        Purpose: Awareness notification with a gentle preventive warning.

        Tone: Neutral, polite, and informative.
        Style: Simple and clear. No emotional or threatening language.
        Length: Maximum 4–5 sentences.

        The email should:
        - Raise awareness about proper waste disposal
        - Gently warn the recipient to be mindful in the future

        Do NOT:
        - Mention locations, investigations, reviews, or authorities
        - Imply penalties, monitoring, or enforcement
        - Include placeholders or assumptions

        End the email exactly with:

        Regards,
        TrashBot Monitoring & Response System
        """

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("⚠️ LLM generation failed:", e)
        return None


# -------------------------------------------------
# Fallback Mail (No LLM)
# -------------------------------------------------
def fallback_mail(name, event, event_date):
    return f"""
Dear {name},

This is to inform you that an event related to "{event}" was detected on {event_date} by the TrashBot monitoring system.

Regards,
TrashBot Monitoring & Response System
"""


# -------------------------------------------------
# Send Mail (with optional screenshot attachment)
# -------------------------------------------------
def send_mail(receiver_email, name, event, screenshot_path=None):
    event_date = extract_event_date(screenshot_path) if screenshot_path else "Unknown Date"

    # Generate body
    body = None
    if USE_LLM:
        body = generate_llm_mail(name, event, event_date)

    if not body:
        body = fallback_mail(name, event, event_date)

    # Build mail
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = receiver_email
    msg["Subject"] = "⚠ TrashBot Alert – Improper Waste Disposal"

    msg.attach(MIMEText(body, "plain"))

    # Attach screenshot if available
    if screenshot_path and os.path.exists(screenshot_path):
        try:
            with open(screenshot_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())

            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f'attachment; filename="{os.path.basename(screenshot_path)}"'
            )
            msg.attach(part)
        except Exception as e:
            print("⚠️ Failed to attach screenshot:", e)

    # Send mail
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)

        print(f"✅ Mail sent to {receiver_email}")
        return True

    except Exception as e:
        print("❌ Mail send failed:", e)
        return False