# Creator Invoice Agent

An AI agent that helps freelance creators get paid. Describe a job in plain English, and the agent builds the invoice, sends it through PayPal, and chases late payments with polite, AI-written reminders.

Built for the PayPal AI Hackathon. Runs on the **PayPal Sandbox** (no real money).

## The problem
Freelance creators, especially in Africa and other emerging markets, lose time and money on invoicing: formatting documents, tracking who has paid, and awkwardly asking for late payments.

## What it does
1. **Build invoice**: type something like *"Invoice Bybit Africa $300 for an X thread and $150 for a video script, due in 7 days"*. The AI extracts the client, line items, currency and due date.
2. **Send with PayPal**: after you confirm the details and client email, the app creates and sends a real invoice through the PayPal Invoicing API.
3. **Track**: your invoices and their live PayPal status (sent, paid, and so on) appear in the dashboard.
4. **Chase with AI**: for unpaid invoices, the AI writes a reminder that fits the situation (friendly if not yet due, firmer if overdue) and PayPal delivers it.

## How PayPal and AI are used
- **PayPal**: OAuth 2.0 token, Invoicing API v2 (create invoice, send invoice, list invoices, get invoice, send reminder).
- **AI**: Google Gemini turns free text into structured invoice data (JSON) and writes the payment reminders.

## Run it yourself
You need free accounts: a [PayPal Developer](https://developer.paypal.com) account and a [Gemini API key](https://aistudio.google.com).

1. In PayPal Developer, go to **Apps & Credentials**, switch to **Sandbox**, and create an app. Copy the Client ID and Secret.
2. In **Sandbox > Accounts**, note the email of the **Business** account.
3. Set the environment variables listed in `.env.example`.
4. Run locally:
   ```
   pip install -r requirements.txt
   python app.py
   ```
   Then open http://localhost:5000
5. Or deploy on [Render](https://render.com): create a new Web Service from this repo (it reads `render.yaml`) and add the same environment variables.

## Tech
Python, Flask, PayPal Invoicing API v2, Google Gemini API. Single file backend (`app.py`) with a built-in mobile-friendly UI.

## License
MIT
