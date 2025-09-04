import os
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from PIL import Image
import pytesseract
import logging

from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_community.document_loaders import WebBaseLoader, PlaywrightURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

# ==============================
# Configuration
# ==============================
URLS = [
    "https://chumbaka.notion.site/Chumbaka-Public-Wiki-211dd84b04a5487e83a86e647f162cdc",
    "https://chumbaka.notion.site/Students-Portfolios-37a320270ea4432dbbf3d01622da0820?pvs=25",
    "https://chumbaka.notion.site/Maker-Club-bd2b17f1f2ce442e9f554f200388ce8d?pvs=25",
    "https://chumbaka.notion.site/Terms-Conditions-of-Chumbaka-Classes-5a7aa95b628b432cb861ba3cba374d96",
    "https://chumbaka.notion.site/Terms-Conditions-of-Competitions-9d16015354f8463f896dab8830fabd0e",
    "https://chumbaka.notion.site/Digital-Badge-20ef6962df8248ee8436fb0e165b6db6",
    "https://chumbaka.notion.site/Usage-of-Chumbaka-LMS-SMS-etc-8adf7d118dc04f2c8badc7251d41ecb1",
    "https://chumbaka.notion.site/Other-STEM-Learning-Resources-786532b2fe7e4ef6839424ee41cac102",
    "https://chumbaka.notion.site/Mentor-Resources-d14a9b74a40a41e98daf9e50ce478135",
    "https://chumbaka.notion.site/Competitions-622b41adc9a94cd2823fe53c50e0a765",
    "https://chumbaka.notion.site/94c84a67fe474255af62fb3387b6d431?v=2856b489d74547f98ae11735a724d2a5",
    "https://docs.google.com/document/d/1KiaRNi2huPNusrz35OhuPQ2u6BV24ovM7OApWN8OAII/edit?tab=t.0"

]

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4
TELEGRAM_TOKEN = "8372718951:AAHXNtGqH-60EVK7UJVNXmTTsPr5UFRFQSw"
CHROMA_DIR = "chroma_index_wiki"

# ==============================
# 1. Load websites
# ==============================
def load_websites(urls):
    docs = []
    try:
        loader = PlaywrightURLLoader(
            urls=urls,
            remove_selectors=["nav", "header", "footer"],
        )
        docs.extend(loader.load())
    except Exception as e:
        logging.warning(f"Playwright failed: {e}, using WebBaseLoader instead.")
        for url in urls:
            docs.extend(WebBaseLoader(url).load())
    return docs

# ==============================
# 2. Extract PDFs, Word docs, OCR images
# ==============================
def extract_additional_docs(urls):
    docs = []
    for url in urls:
        try:
            html = requests.get(url).text
            soup = BeautifulSoup(html, "html.parser")

            # PDFs & Word
            for a in soup.find_all("a", href=True):
                href = a['href']
                if href.lower().endswith(".pdf"):
                    try: docs.extend(PyPDFLoader(href).load())
                    except: pass
                elif href.lower().endswith((".docx", ".doc")):
                    try: docs.extend(UnstructuredWordDocumentLoader(href).load())
                    except: pass

            # Images OCR
            for img in soup.find_all("img"):
                src = img.get("src")
                alt = img.get("alt", "")
                if src and src.startswith("http"):
                    try:
                        resp = requests.get(src)
                        image = Image.open(BytesIO(resp.content))
                        text = pytesseract.image_to_string(image)
                        content = alt + " " + text
                        if content.strip():
                            docs.append(Document(page_content=content, metadata={"source": url, "type": "image"}))
                    except: pass
        except: continue
    return docs

# ==============================
# 3. Split documents
# ==============================
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(documents)

# ==============================
# 4. Build Chroma DB
# ==============================
def build_chroma_db(documents):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma.from_documents(documents, embeddings, persist_directory=CHROMA_DIR)

# ==============================
# 5. Initialize Retrieval QA
# ==============================
def init_qa(db):
    llm = Ollama(
        model="mistral",
        temperature=0.2,   # less creative, more factual
        num_predict=300    # keep answers short
    )
    prompt = PromptTemplate(
        template="""
    You are the official chatbot for Chumbaka Wiki.
    Rules:
    - If user greets (e.g., hi, hello), reply with a short friendly greeting like "Hello! I'm the Chumbaka Wiki chatbot. Ask me anything about Chumbaka Wiki."
    - Otherwise, answer based only on the provided website content.
    - Always reply in first person ("we") as Chumbaka Wiki.
    - If user asks unrelated questions, say: "I can only answer questions about Chumbaka Wiki."
    - You should remember the previous question and use it to understand follow-up questions.
    - Never answer anything not related to the provided websites.
    - Keep answers short, clear, and focused.

        Context:
        {context}

        Question:
        {question}
        """,
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": TOP_K}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# ==============================
# 6. Telegram Bot Handler
# ==============================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    bot_username = (await context.bot.get_me()).username
    text = message.text

    # Ignore if not mentioned in group
    if message.chat.type in ["group", "supergroup"]:
        entities = message.entities or []
        mentioned = any(
            e.type == "mention" and text[e.offset:e.offset+e.length] == f"@{bot_username}"
            for e in entities
        )
        # ✅ Check if bot is replied to
        replied_to_bot = (
            message.reply_to_message
            and message.reply_to_message.from_user.username == bot_username
        )
        if not mentioned and not replied_to_bot:
            return  # ignore if not mentioned or not replied to

    # If direct chat or mentioned in group, process as usual
    result = qa({"query": text})
    response = result["result"]

    if result["source_documents"] and "I can only answer questions about Chumbaka Wiki." not in response:
        top_source = result["source_documents"][0].metadata.get("source", "Unknown")
        response += f"\n\n📌 Source: {top_source}"

    await message.reply_text(response)

# ==============================
# 7. Main
# ==============================
if os.path.exists(CHROMA_DIR):
    print("📂 Loading existing Chroma index...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
else:
    print("📥 Loading websites and documents...")
    all_docs = load_websites(URLS)
    all_docs.extend(extract_additional_docs(URLS))
    print(f"Loaded {len(all_docs)} documents.")

    print("✂️ Splitting documents...")
    docs_split = split_documents(all_docs)

    print("🔎 Creating embeddings and Chroma DB...")
    db = build_chroma_db(docs_split)
    db.persist()

# Initialize QA globally
qa = init_qa(db)

if __name__ == "__main__":
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    print("🤖 Bot is running...")
    application.run_polling()
