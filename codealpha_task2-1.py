import gradio as gr
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity

# ==========================
# Load AI Model
# ==========================

print("Loading AI Model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================
# FAQ Dataset
# ==========================

faq_data = [
    {
        "question": "How can I track my order?",
        "answer": "📦 You can track your order anytime from the **Orders** section in your account, or use the tracking link sent to your email after dispatch."
    },
    {
        "question": "What is your return policy?",
        "answer": "🔄 We offer a **30-day hassle-free return** policy. As long as the item meets our return eligibility criteria, we've got you covered!"
    },
    {
        "question": "Do you offer international shipping?",
        "answer": "🌍 Yes! We ship to **selected countries worldwide**. Enter your address at checkout to see if we deliver to your location."
    },
    {
        "question": "How long does shipping take?",
        "answer": "🚚 Standard shipping takes **3–7 business days**. Express options are available at checkout for faster delivery."
    },
    {
        "question": "Can I change my order after placing it?",
        "answer": "✏️ Order changes are possible **before the order is processed**. Please contact support immediately after placing your order to request modifications."
    },
    {
        "question": "What payment methods do you accept?",
        "answer": "💳 We accept **UPI, Debit Cards, Credit Cards, Net Banking, and PayPal**. All transactions are secured with 256-bit encryption."
    },
    {
        "question": "Can I cancel my order?",
        "answer": "❌ Yes, you can cancel your order **before it is shipped**. Head to your Orders page and select 'Cancel Order'."
    },
    {
        "question": "How do I get a refund?",
        "answer": "💰 Refunds are **automatically processed** to your original payment method after the return is verified. This typically takes 3–5 business days."
    },
    {
        "question": "Do you provide cash on delivery?",
        "answer": "🏠 Yes! **Cash on Delivery (COD)** is available in selected pin codes. You'll see this option at checkout if it's available for your area."
    },
    {
        "question": "How can I contact customer support?",
        "answer": "🎧 Our support team is here for you! Reach us via:\n- 📧 **Email:** support@store.com\n- 📞 **Phone:** 1800-XXX-XXXX\n- 💬 **Live Chat:** Available 9 AM – 9 PM"
    }
]

df = pd.DataFrame(faq_data)

# ==========================
# Create Embeddings
# ==========================

faq_embeddings = model.encode(
    df["question"].tolist(),
    convert_to_numpy=True
)

# ==========================
# Chatbot Logic
# ==========================

def respond(message, history):
    query_embedding = model.encode([message], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, faq_embeddings)[0]
    best_index = int(np.argmax(similarities))
    best_score = similarities[best_index]

    if best_score > 0.45:
        return df.iloc[best_index]["answer"]

    return (
        "🤔 I'm not sure about that one yet!\n\n"
        "Here's what I can help you with:\n"
        "• 📦 **Order Tracking**\n"
        "• 🚚 **Shipping Info**\n"
        "• 🔄 **Returns & Exchanges**\n"
        "• 💰 **Refunds**\n"
        "• 💳 **Payment Methods**\n"
        "• 🎧 **Customer Support**\n\n"
        "Try rephrasing your question or choose a topic above!"
    )

# ==========================
# Custom CSS Theme
# ==========================

custom_css = """
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

/* ── Page background ── */
body, .gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%) !important;
    min-height: 100vh;
}

/* ── Main card ── */
.gradio-container > .main > .wrap {
    max-width: 780px !important;
    margin: 0 auto !important;
    padding: 16px !important;
}

/* ── Title block ── */
#component-0 h1 {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    text-align: center !important;
    letter-spacing: -0.5px !important;
}

/* ── Description block ── */
#component-0 p, .prose p {
    color: #a5b4fc !important;
    text-align: center !important;
    font-size: 0.9rem !important;
}

/* ── Chat window ── */
.chatbot {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 20px !important;
    box-shadow: 0 25px 50px rgba(0,0,0,0.5) !important;
    min-height: 420px !important;
}

/* ── Bot message bubble ── */
.message.bot, [data-testid="bot"] .bubble-wrap {
    background: linear-gradient(135deg, #312e81, #4338ca) !important;
    border-radius: 18px 18px 18px 4px !important;
    color: #e0e7ff !important;
    padding: 12px 16px !important;
    font-size: 0.92rem !important;
    line-height: 1.6 !important;
    box-shadow: 0 4px 12px rgba(99,102,241,0.3) !important;
}

/* ── User message bubble ── */
.message.user, [data-testid="user"] .bubble-wrap {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    border-radius: 18px 18px 4px 18px !important;
    color: #ffffff !important;
    padding: 12px 16px !important;
    font-size: 0.92rem !important;
    box-shadow: 0 4px 12px rgba(16,185,129,0.3) !important;
}

/* ── Input row ── */
.input-row, footer > div {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 14px !important;
    padding: 8px !important;
    margin-top: 10px !important;
}

/* ── Textbox ── */
textarea, input[type="text"] {
    background: #0f172a !important;
    border: 1px solid #475569 !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    font-size: 0.93rem !important;
    padding: 10px 14px !important;
}
textarea::placeholder, input::placeholder {
    color: #64748b !important;
}
textarea:focus, input:focus {
    border-color: #6366f1 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.2) !important;
}

/* ── Send button ── */
button.primary, #submit-btn {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(99,102,241,0.4) !important;
}
button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(99,102,241,0.5) !important;
}

/* ── Example chips ── */
.examples-holder button, .example-button {
    background: #1e293b !important;
    border: 1px solid #4f46e5 !important;
    border-radius: 999px !important;
    color: #a5b4fc !important;
    font-size: 0.82rem !important;
    padding: 6px 14px !important;
    transition: all 0.2s !important;
    white-space: nowrap !important;
}
.examples-holder button:hover, .example-button:hover {
    background: #4f46e5 !important;
    color: #fff !important;
    transform: translateY(-1px) !important;
}

/* ── Clear button ── */
button.secondary {
    background: transparent !important;
    border: 1px solid #475569 !important;
    color: #94a3b8 !important;
    border-radius: 10px !important;
}
button.secondary:hover {
    border-color: #ef4444 !important;
    color: #ef4444 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #4f46e5; border-radius: 99px; }

/* ── Avatar ── */
.avatar-container img { border-radius: 50%; border: 2px solid #6366f1; }
"""

# ==========================
# Gradio UI
# ==========================

with gr.Blocks(css=custom_css, title="ShopBot — E-Commerce Support") as demo:

    gr.Markdown(
        """
        # 🛍️ ShopBot — Your Shopping Assistant
        *Fast answers for orders, shipping, returns, payments & more.*
        """
    )

    chatbot = gr.Chatbot(
        value=[
            [
                None,
                "👋 Hi there! I'm **ShopBot**, your personal shopping assistant.\n\n"
                "I can help you with:\n"
                "📦 Order tracking · 🚚 Shipping · 🔄 Returns · 💰 Refunds · 💳 Payments\n\n"
                "What can I help you with today?"
            ]
        ],
        label="ShopBot",
        bubble_full_width=False,
        avatar_images=(
            None,   # user — no avatar
            "https://api.dicebear.com/7.x/bottts/svg?seed=shopbot&backgroundColor=6366f1"
        ),
        height=440,
        show_label=False,
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Type your question here…",
            show_label=False,
            scale=8,
            container=False,
        )
        send_btn = gr.Button("Send ➤", variant="primary", scale=1, min_width=90)

    gr.Examples(
        examples=[
            "How can I track my order?",
            "Can I cancel my order?",
            "What payment methods do you accept?",
            "How do I get a refund?",
            "Do you offer international shipping?",
        ],
        inputs=msg,
        label="✨ Quick questions",
    )

    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")

    # ── State & wiring ──
    history_state = gr.State([])

    def chat(user_message, history):
        if not user_message.strip():
            return history, history, ""
        bot_reply = respond(user_message, history)
        history = history + [[user_message, bot_reply]]
        return history, history, ""

    send_btn.click(chat, [msg, history_state], [chatbot, history_state, msg])
    msg.submit(chat, [msg, history_state], [chatbot, history_state, msg])

    def clear_chat():
        welcome = [[
            None,
            "👋 Hi again! I'm **ShopBot**, ready to help.\n\n"
            "What can I assist you with today?"
        ]]
        return welcome, []

    clear_btn.click(clear_chat, [], [chatbot, history_state])

# ==========================
# Launch App
# ==========================

if __name__ == "__main__":
    demo.launch()
