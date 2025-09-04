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
    "https://www.chumbaka.asia/",
    "https://www.chumbaka.asia/our-story/",
    "https://www.chumbaka.asia/our-program/",
    "https://www.chumbaka.asia/course-synopsis/",
    "https://www.chumbaka.asia/for-schools/",
    "https://www.chumbaka.asia/for-communities/",
    "https://www.chumbaka.asia/contact-us/",
    "https://www.chumbaka.asia/testimonials/",
    "https://www.chumbaka.asia/newsletter/",
    "https://www.chumbaka.asia/insights/",
    "https://www.chumbaka.asia/our-program/#faq",
    "https://www.chumbaka.asia/tos/",
    "https://www.chumbaka.asia/privacy-policy/"
]

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4
TELEGRAM_TOKEN = "7657756190:AAHV-0-KJxwORbX11M_PDRtnK-YCXsJbkvc"
CHROMA_DIR = "chroma_index_cbk"

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
    You are the official chatbot for Chumbaka.
    Rules:
    - If user greets (e.g., hi, hello), reply with a short friendly greeting like "Hello! I'm the Chumbaka chatbot. Ask me anything about Chumbaka."
    - Otherwise, answer based only on the provided website content.
    - Always reply in first person ("we") as Chumbaka.
    - If user asks unrelated questions, say: "I can only answer questions about Chumbaka."
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
        replied_to_bot = (
            message.reply_to_message
            and message.reply_to_message.from_user.username == bot_username
        )
        if not mentioned and not replied_to_bot:
            return  # ignore if not mentioned or not replied to

    # Remove mention from the text
    clean_text = text.replace(f"@{bot_username}", "").strip()
    if not clean_text:
        return  # ignore if nothing to answer

    # Process message as usual
    result = qa({"query": clean_text})
    response = result["result"]

    if result["source_documents"] and "I can only answer questions about Chumbaka." not in response:
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