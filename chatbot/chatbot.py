import os
import time
import re
import chardet
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from threading import Timer
from uuid import uuid4
from tqdm.auto import tqdm
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
import psycopg2
from psycopg2 import OperationalError, InterfaceError, pool
import pinecone
from langchain.document_loaders import UnstructuredURLLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flask_login import LoginManager, login_required, login_user, logout_user, UserMixin
import tempfile
from urllib.parse import urljoin
from urllib.parse import urlparse
from time import sleep
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from flask_caching import Cache
from flask import current_app
from fpdf import FPDF
from werkzeug.utils import url_quote
from html import escape
from docx import Document
import PyPDF2
from langchain.document_loaders import PyPDFium2Loader
import json

with open('config.json', 'r') as file:
    config = json.load(file)


# API-sleutels en instellingen
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# Database configuratie
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# Connection Pool instellen
conn_pool = psycopg2.pool.SimpleConnectionPool(
    minconn=1,
    maxconn=20,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)

# Constante voor karakterlimietl
CHARACTER_LIMIT = config['character_limit']


def get_conn():
    return conn_pool.getconn()

def release_conn(conn):
    return conn_pool.putconn(conn)

def get_conn():
    return conn_pool.getconn()

def release_conn(conn):
    return conn_pool.putconn(conn)

# Functie om de database te initialiseren
def initialize_db():
    conn = get_conn()
    cur = conn.cursor()

    # Creëer de character_counter tabel als deze nog niet bestaat
    cur.execute("""
    CREATE TABLE IF NOT EXISTS character_counter (
        id SERIAL PRIMARY KEY,
        num_characters INT DEFAULT 0
    );
    """)

    # Creëer de chat_history tabel als deze nog niet bestaat
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id SERIAL PRIMARY KEY,
        user_id TEXT,
        question TEXT,
        answer TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        month_year TEXT
    );
    """)

    

    # Creëer de api_calls tabel als deze nog niet bestaat
    cur.execute("""
    CREATE TABLE IF NOT EXISTS api_calls (
        id SERIAL PRIMARY KEY,
        user_id TEXT,
        month TEXT,
        num_calls INT
    );
    """)



# Creëer de welcome_message tabel als deze nog niet bestaat
    cur.execute("""
    CREATE TABLE IF NOT EXISTS welcome_message (
        id SERIAL PRIMARY KEY,
        message TEXT
    );
    """)

    cur.execute("""
    INSERT INTO character_counter (id, num_characters)
    SELECT 1, 0
    WHERE NOT EXISTS (SELECT 1 FROM character_counter WHERE id = 1);
    """)


    # Voeg een standaard welkomstbericht toe als deze nog niet bestaat
    cur.execute(f"""
    INSERT INTO welcome_message (id, message)
    SELECT 1, '{config['database']['welcome_message']}'
    WHERE NOT EXISTS (SELECT 1 FROM welcome_message WHERE id = 1);
    """)


        # In je initialize_db functie
    cur.execute("""
    CREATE TABLE IF NOT EXISTS title_message (
        id SERIAL PRIMARY KEY,
        message TEXT
    );
    """)
    
    cur.execute(f"""
    INSERT INTO title_message (id, message)
    SELECT 1, '{config['database']['title_message']}'
    WHERE NOT EXISTS (SELECT 1 FROM title_message WHERE id = 1);
    """)


    # In je initialize_db functie
    cur.execute("""
    CREATE TABLE IF NOT EXISTS color_settings (
        id SERIAL PRIMARY KEY,
        color_code TEXT
    );
    """)

    cur.execute(f"""
    INSERT INTO color_settings (id, color_code)
    SELECT 1, '{config['database']['color_code']}'
    WHERE NOT EXISTS (SELECT 1 FROM color_settings WHERE id = 1);
    """)


    cur.execute("""
    CREATE TABLE IF NOT EXISTS chat_history_archive (
        id SERIAL PRIMARY KEY,
        user_id TEXT,
        question TEXT,
        answer TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        month_year TEXT
    );
    """)


    conn.commit()
    cur.close()
    release_conn(conn)

# Functie om de karakterteller bij te werken
def update_counter(num_characters):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("UPDATE character_counter SET num_characters = num_characters + %s WHERE id = 1;", (num_characters,))

    conn.commit()
    cur.close()
    release_conn(conn)

# Functie om te controleren of de karakterlimiet is bereikt
# Functie om te controleren of de karakterlimiet is bereikt
def check_limit():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT num_characters FROM character_counter WHERE id = 1;")
    fetch_result = cur.fetchone()
    
    cur.close()
    release_conn(conn)

    # Controleer of fetch_result None is
    if fetch_result is None:
        return False

    total_characters = fetch_result[0]

    return total_characters >= CHARACTER_LIMIT


# Definieer functies om met de database te werken
def insert_chat(user_id, question, answer):
    conn = get_conn()
    cur = conn.cursor()

    try:
        cur.execute(
            "INSERT INTO chat_history (user_id, question, answer) VALUES (%s, %s, %s)",
            (user_id, question, answer)
        )
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Database Error: {error}")
    finally:
        cur.close()
        release_conn(conn)

def count_questions_for_month(month_year):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "SELECT COUNT(*) FROM chat_history WHERE month_year = %s",
        (month_year,)
    )
    result = cur.fetchone()[0]

    cur.close()
    release_conn(conn)
    return result

def count_questions():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM chat_history")
    result = cur.fetchone()[0]

    cur.close()
    release_conn(conn)
    return result

def fetch_chats(limit=2):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            conn = get_conn()
            cur = conn.cursor()
            cur.execute("SELECT * FROM chat_history ORDER BY timestamp DESC LIMIT %s", (limit,))
            results = cur.fetchall()
            cur.close()
            release_conn(conn)
            return results
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            retries += 1
            print(f"Fout tijdens het ophalen van chats, poging {retries} van {MAX_RETRIES}. Foutmelding: {e}")
            sleep(RETRY_DELAY)
    print("Maximaal aantal pogingen bereikt. Chats ophalen is mislukt.")
    return []


MAX_RETRIES = 5
RETRY_DELAY = 5  # in seconden

def reset_monthly_counter():
    retries = 0
    success = False

    while not success and retries < MAX_RETRIES:
        conn = None
        cur = None
        try:
            conn = get_conn()
            cur = conn.cursor()
            now = datetime.now()
            month_year = now.strftime("%Y-%m")

            # Kopieer records naar de archiveringstabel
            cur.execute("""
            INSERT INTO chat_history_archive (user_id, question, answer, timestamp, month_year)
            SELECT user_id, question, answer, timestamp, month_year FROM chat_history WHERE month_year = %s
            """, (month_year,))

            # Verwijder records uit de oorspronkelijke tabel
            cur.execute("DELETE FROM chat_history WHERE month_year = %s", (month_year,))
            
            conn.commit()
            success = True
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            retries += 1
            print(f"Fout tijdens het resetten, poging {retries} van {MAX_RETRIES}. Foutmelding: {e}")
            sleep(RETRY_DELAY)
        finally:
            if cur:
                cur.close()
            if conn:
                release_conn(conn)

    if not success:
        print("Maximaal aantal pogingen bereikt. Resetten is mislukt.")


scheduler = BackgroundScheduler()
scheduler_time = config['scheduler']
scheduler.add_job(reset_monthly_counter, CronTrigger(day=scheduler_time['day'], hour=scheduler_time['hour'], minute=scheduler_time['minute']))
scheduler.start()

# Log wanneer de scheduler wordt gestart
print("Scheduler is gestart en taak is toegevoegd.")

# Nieuwe functie om het titelbericht op te halen
def get_title_message():
    conn = get_conn()
    cur = conn.cursor()
    
    cur.execute("SELECT message FROM title_message WHERE id = 1;")
    result = cur.fetchone()
    
    cur.close()
    release_conn(conn)

    if result:
        return result[0]
    else:
        return "Chatwize - Wizzy 🤖"

# Nieuwe functie om het titelbericht bij te werken
def update_title_message(new_message):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("UPDATE title_message SET message = %s WHERE id = 1;", (new_message,))

    conn.commit()
    cur.close()
    release_conn(conn)


custom_prompt_template = config['prompt_template']


# Aanmaken van de PromptTemplate
QA_CHAIN_PROMPT = PromptTemplate.from_template(custom_prompt_template)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo', temperature=0.5)

# Pinecone initialiseren
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
index_name = config['pinecone']['index_name']

# Controleren of index al bestaat, zo niet, aanmaken
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, metric='cosine', dimension=1536)

# Data laden
urls = []  # Vul hier je URLs in...
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()

# Initialisatie van embeddings en index
embed = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
index = pinecone.Index(index_name)

# Data uploaden naar Pinecone
namespace = 'chatbo'; namespace = config['pinecone']['namespace']
batch_limit = 100
texts = []
metadatas = []

for i, record in enumerate(tqdm(data)):  # data moet je zelf definiëren
    metadata = {'wiki-id': str(record['id']), 'source': record['url'], 'title': record['title']}
    record_texts = text_splitter.split_text(record['text'])  # text_splitter moet je zelf definiëren
    record_metadatas = [{"chunk": j, "text": text, **metadata} for j, text in enumerate(record_texts)]
    texts.extend(record_texts)
    metadatas.extend(record_metadatas)

    if len(texts) >= batch_limit:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        try:
            index.upsert(vectors=zip(ids, embeds), namespace=namespace)
            print(f"{len(ids)} records geüpload naar namespace: {namespace}")
        except Exception as e:
            print(f"Upsert mislukt: {e}")
        texts.clear()
        metadatas.clear()


if len(texts) > 0:
    ids = [str(uuid4()) for _ in range(len(texts))]
    embeds = embed.embed_documents(texts)
    try:
        index.upsert(vectors=zip(ids, embeds), namespace=namespace)  # Upsert met de gespecificeerde namespace
        print(f"{len(ids)} records geüpload naar namespace: {namespace}")
    except Exception as e:
        print(f"Upsert mislukt: {e}")

# Vectorstore en LLM instellen
vectorstore = Pinecone(index, embed.embed_query, "text")

# Instantiëren van de RetrievalQA keten
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(search_kwargs={'k': 5, 'namespace': namespace}),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=True,
)

# Cache configuratie
cache_config = {
    "DEBUG": True,          # some Flask specific configs
    "CACHE_TYPE": "simple", # Use in-memory cache for this demo
}

# Initialisatie van Flask app
app = Flask(__name__)
app.config.from_mapping(cache_config)
cache = Cache(app)
CORS(app, origins=config['web']['cors_origins'])
app.secret_key = os.urandom(24)  # Genereer een willekeurige geheime sleutel

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    id = 1

@login_manager.user_loader
def load_user(user_id):
    return User()

@app.route('/ask', methods=['POST'])
def ask():
    user_ip = request.headers.get('X-Forwarded-For') or request.headers.get('X-Real-IP') or request.remote_addr
    
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cur = conn.cursor()

        # Genereer een unieke user_id of verkrijg het van de client
        user_id = user_ip
        
        request_data = request.get_json()
        question = escape(request_data.get('question', ''))


        now = datetime.now()
        month_year = now.strftime("%Y-%m")
       
        if count_questions_for_month(month_year) >= config['monthly_limit']:
                return jsonify({"error": f"Maandelijkse limiet van {config['monthly_limit']} vragen is bereikt voor de hele chatbot"})

        
        if question.lower() == 'quit':
            return jsonify({"answer": "Chat beëindigd"}), 200

        result = qa_chain({"query": question})
        
        answer = escape(result.get('result', 'Sorry, ik kan deze vraag niet beantwoorden.'))
        source_documents = result.get('source_documents', [])
        source_documents_serializable = [vars(doc) for doc in source_documents]

        insert_chat(user_id, question, answer)

        cur.execute(
            "UPDATE chat_history SET month_year = %s WHERE user_id = %s AND question = %s",
            (month_year, user_id, question)
        )
        conn.commit()
        # Verzamelen van de bronnen uit de `source_documents`
        source_links = [{"url": doc.get('source', ''), "title": doc.get('title', '')} for doc in source_documents_serializable]

        
        # Verstuur het als een deel van de JSON response
        return jsonify({"answer": answer, "source_documents": source_documents_serializable, "source_links": source_links}), 200



    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cur.close()
        conn.close()
        
@app.before_first_request
def initialize():
    initialize_db()

@cache.cached(key_prefix='welcome_message')
def get_welcome_message():
    conn = get_conn()
    cur = conn.cursor()
    
    cur.execute("SELECT message FROM welcome_message WHERE id = 1;")
    result = cur.fetchone()
    
    cur.close()
    release_conn(conn)

    if result:
        return result[0]
    else:
        return "Standaard welkomstbericht"

def update_welcome_message(new_message):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("UPDATE welcome_message SET message = %s WHERE id = 1;", (new_message,))
    conn.commit()
    cur.close()
    release_conn(conn)
    
    cache.delete('welcome_message')  # Clear the cache after updating

@cache.cached(key_prefix='title_message')
def get_title_message():
    conn = get_conn()
    cur = conn.cursor()
    
    cur.execute("SELECT message FROM title_message WHERE id = 1;")
    result = cur.fetchone()
    
    cur.close()
    release_conn(conn)

    if result:
        return result[0]
    else:
        return "Chatwize - Wizzy 🤖"

def update_title_message(new_message):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("UPDATE title_message SET message = %s WHERE id = 1;", (new_message,))
    conn.commit()
    cur.close()
    release_conn(conn)
    
    cache.delete('title_message')  # Clear the cache after updating

@cache.cached(key_prefix='color_settings')
def get_color_settings():
    conn = get_conn()
    cur = conn.cursor()
    
    cur.execute("SELECT color_code FROM color_settings WHERE id = 1;")
    result = cur.fetchone()
    
    cur.close()
    release_conn(conn)

    if result:
        return result[0]
    else:
        return "#FFFFFF"

def update_color_settings(new_color):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("UPDATE color_settings SET color_code = %s WHERE id = 1;", (new_color,))
    conn.commit()
    cur.close()
    release_conn(conn)
    
    cache.delete('color_settings')  # Clear the cache after updating
    
@app.route('/get_color', methods=['GET'])
def api_get_color():
    color = cache.get('color_settings')
    if color:
        current_app.logger.info("Data fetched from the cache")
        return jsonify({'color': color})
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT color_code FROM color_settings WHERE id = 1;")
        result = cur.fetchone()
        cur.close()
        release_conn(conn)
        if result:
            cache.set('color_settings', result[0], timeout=1000*60)  # Cache indefinitely
            current_app.logger.info("Data fetched from the database")
            return jsonify({'color': result[0]})
        else:
            return jsonify({'color': "#FFFFFF"}), 400
    except Exception as e:
        current_app.logger.error(f"Error fetching color: {e}")
        return jsonify({'error': 'Unable to fetch color'}), 500

@app.route('/update_color', methods=['POST'])
def api_update_color():
    new_color = request.json.get('color')
    if new_color:
        try:
            conn = get_conn()
            cur = conn.cursor()
            cur.execute("UPDATE color_settings SET color_code = %s WHERE id = 1;", (new_color,))
            conn.commit()
            cur.close()
            release_conn(conn)
            cache.delete('color_settings')  # Clear the cache after updating
            return jsonify({'status': 'success'}), 200
        except Exception as e:
            current_app.logger.error(f"Error updating color: {e}")
            return jsonify({'error': 'Unable to update color'}), 500
    return jsonify({'status': 'failure'}), 400

@app.route('/get_title_message', methods=['GET'])
def api_get_title_message():
    title = cache.get('title_message')
    if title:
        current_app.logger.info("Data fetched from the cache")
        return jsonify({'message': title})
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT message FROM title_message WHERE id = 1;")
        result = cur.fetchone()
        cur.close()
        release_conn(conn)
        if result:
            cache.set('title_message', result[0], timeout=1000*60)  # Cache indefinitely
            current_app.logger.info("Data fetched from the database")
            return jsonify({'message': result[0]})
        else:
            return jsonify({'message': "Chatwize - Wizzy 🤖"})
    except Exception as e:
        current_app.logger.error(f"Error fetching title message: {e}")
        return jsonify({'error': 'Unable to fetch title message'}), 500

@app.route('/update_title_message', methods=['POST'])
def api_update_title_message():
    new_message = request.json.get('message')
    if new_message:
        try:
            conn = get_conn()
            cur = conn.cursor()
            cur.execute("UPDATE title_message SET message = %s WHERE id = 1;", (new_message,))
            conn.commit()
            cur.close()
            release_conn(conn)
            cache.delete('title_message')  # Clear the cache after updating
            return jsonify({'status': 'success'}), 200
        except Exception as e:
            current_app.logger.error(f"Error updating title message: {e}")
            return jsonify({'error': 'Unable to update title message'}), 500
    return jsonify({'status': 'failure'}), 400

@app.route('/get_welcome_message', methods=['GET'])
def api_get_welcome_message():
    welcome_message = cache.get('welcome_message')
    if welcome_message:
        current_app.logger.info("Data fetched from the cache")
        return jsonify({'message': welcome_message})
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT message FROM welcome_message WHERE id = 1;")
        result = cur.fetchone()
        cur.close()
        release_conn(conn)
        if result:
            cache.set('welcome_message', result[0], timeout=1000*60)  # Cache indefinitely
            current_app.logger.info("Data fetched from the database")
            return jsonify({'message': result[0]})
        else:
            return jsonify({'message': "Standaard welkomstbericht"})
    except Exception as e:
        current_app.logger.error(f"Error fetching welcome message: {e}")
        return jsonify({'error': 'Unable to fetch welcome message'}), 500

@app.route('/update_welcome_message', methods=['POST'])
def api_update_welcome_message():
    new_message = request.json.get('message')
    if new_message:
        try:
            conn = get_conn()
            cur = conn.cursor()
            cur.execute("UPDATE welcome_message SET message = %s WHERE id = 1;", (new_message,))
            conn.commit()
            cur.close()
            release_conn(conn)
            cache.delete('welcome_message')  # Clear the cache after updating
            return jsonify({'status': 'success'}), 200
        except Exception as e:
            current_app.logger.error(f"Error updating welcome message: {e}")
            return jsonify({'error': 'Unable to update welcome message'}), 500
    return jsonify({'status': 'failure'}), 400

@app.route('/settings', methods=['GET'])
@login_required
def settings():
    return render_template('settings.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == os.getenv('USERNAME') and password == os.getenv('PASSWORD'):
            user = User()
            login_user(user)
            return redirect(url_for('settings'))
        else:
            return "Invalid username or password"

    return render_template('login.html')  # Je moet een login.html template maken

def get_subpages(url):
    try:
        r = requests.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        subpages_set = {urljoin(url, a['href']) for a in soup.find_all('a', href=True)}
        return list(subpages_set)
    except requests.RequestException:
        return []

def download_and_clean_page(url, folder):
    try:
        r = requests.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')

        # Verwijder ongewenste tags
        for tag in soup(['script', 'style', 'head', 'nav', 'footer', 'form', 'button', 'link']):
            tag.extract()

        # Converteer de overige inhoud naar pure tekst
        text_content = ' '.join(soup.stripped_strings)

        # Verdere opschoning met reguliere expressies
        text_content = re.sub(r'\s+', ' ', text_content)  # Vervang opeenvolgende whitespaces door één spatie
        text_content = re.sub(r'[^\x00-\x7F]+', ' ', text_content)  # Verwijder niet-ASCII tekens

        if text_content:
            with open(os.path.join(folder, "single_page.txt"), "w", encoding="utf-8") as f:
                f.write(text_content)
            return len(text_content)
    except requests.RequestException:
        return 0

@app.route('/get_subpages', methods=['POST'])
def get_subpages_endpoint():
    data = request.json
    url_to_scrape = data.get('websiteUrl', None)
    
    if not url_to_scrape:
        return jsonify({'error': 'websiteUrl is required'}), 400
    
    all_subpages = get_subpages(url_to_scrape)
    
    if not all_subpages:
        return jsonify({'error': 'No subpages found'}), 400

    # Filter alleen bruikbare subpagina's.
    # In dit voorbeeld houden we alleen de subpagina's op hetzelfde domein.
    domain = urlparse(url_to_scrape).netloc
    filtered_subpages = [url for url in all_subpages if urlparse(url).netloc == domain]

    if not filtered_subpages:
        return jsonify({'error': 'No usable subpages found'}), 400

    return jsonify({'subpages': filtered_subpages})

@app.route('/scrape_and_select', methods=['POST'])
def scrape_and_select():
    if check_limit():
        return jsonify({'error': 'Karakterlimiet bereikt'}), 403

    url_to_scrape = request.form.get('mainWebsiteUrl', None)
    selected_subpages = request.form.getlist('selectedSubpages')

    subpages = get_subpages(url_to_scrape)
    if not subpages:
        return jsonify({'error': 'No subpages found'}), 400

    total_characters_scraped = 0
    for subpage in selected_subpages:
        if subpage in subpages:
            with tempfile.TemporaryDirectory() as folder:
                num_characters_scraped = download_and_clean_page(subpage, folder)

                # Type check en correctie
                if not isinstance(num_characters_scraped, int):
                    # Log de onverwachte waarde voor debugging
                    print(f"Unexpected value for num_characters_scraped: {num_characters_scraped}")
                    num_characters_scraped = 0  # Stel in op standaardwaarde

                total_characters_scraped += num_characters_scraped

                if num_characters_scraped > 0:
                    temp_file_path = os.path.join(folder, "single_page.txt")

                    # Initialize Pinecone and OpenAI Embeddings
                    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_API_ENV"))
                    embeddings_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

                    # Load text file and split it into chunks
                    loader = TextLoader(temp_file_path)
                    documents = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    docs = text_splitter.split_documents(documents)

                    # Store the chunks in Pinecone
                    chunks = [doc.page_content for doc in docs if hasattr(doc, 'page_content')]
                    metadatas = [{"source": subpage} for _ in range(len(chunks))]
                    pinecone_vectorstore = Pinecone.from_texts(
                        chunks,
                        embeddings_model,
                        index_name="chatter",
                        metadatas=metadatas,
                        namespace=namespace  # Zorg ervoor dat 'namespace' correct is gedefinieerd of gepasseerd
                    )

    update_counter(total_characters_scraped)
    return jsonify({'totalCharactersScraped': total_characters_scraped})
    
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.template_filter('formatdate')
def format_date(value, format='%d-%m-%Y %H:%M'):
    if isinstance(value, datetime):
        return value.strftime(format)
    return datetime.fromtimestamp(value).strftime(format)


@app.route('/')
@login_required
def home():
    chats = fetch_chats(limit=100)
    num_questions = count_questions()
    return render_template('index.html', chats=chats, num_questions=num_questions)


@app.route('/get_character_count', methods=['GET'])
def get_character_count():
    conn = get_conn()  # Haal een verbinding uit de pool
    cur = conn.cursor()
    cur.execute("SELECT SUM(num_characters) FROM character_counter;")
    total_characters = cur.fetchone()[0]
    cur.close()
    release_conn(conn)  # Geef de verbinding terug aan de pool
    if total_characters is None:
        total_characters = 0
    return jsonify({'totalCharacters': total_characters})

@app.route('/data')
@login_required
def index():
    return render_template('data.html')  # Zorg ervoor dat je een index.html hebt in een "templates" map


# Flask Route voor het uitvoeren van Web Scraping
@app.route('/scrape', methods=['POST'])
def scrape():
    # Controleer eerst of de karakterlimiet is bereikt
    if check_limit():
        return jsonify({'error': 'Karakterlimiet bereikt'}), 403

    url_to_scrape = request.form['websiteUrl']
    
    # Houd een teller bij voor het aantal gescrapede tekens
    num_characters_scraped = 0

    def clean_filename(filename):
        return re.sub('[^a-zA-Z0-9 \n\.]', '_', filename)
    
    def download_page(url, folder):
        try:
            r = requests.get(url)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                page_title = soup.title.string if soup.title else "Untitled"
                page_title = clean_filename(page_title)
                with open(os.path.join(folder, f"{page_title}.html"), "w", encoding="utf-8") as f:
                    f.write(r.text)
        except Exception as e:
            print(f"Error while processing URL {url}: {e}")

    with tempfile.TemporaryDirectory() as folder:
        download_page(url_to_scrape, folder)
        r = requests.get(url_to_scrape)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            for link in soup.find_all('a'):
                url = link.get('href')
                if url and url.startswith('http'):
                    download_page(url, folder)

        all_texts = ""
        for root, dirs, files in os.walk(folder, topdown=False):
            for filename in files:
                file_path = os.path.join(root, filename)
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, 'rb') as file:
                            raw_data = file.read()
                        encoding = chardet.detect(raw_data)['encoding']
                        with open(file_path, 'r', encoding=encoding) as file:
                            file_content = file.read()
                        soup = BeautifulSoup(file_content, features="lxml")
                        for tag in soup(['script', 'style', 'meta', 'link', 'head', 'footer', 'nav', 'form']):
                            tag.extract()
                        body = soup.body if soup and soup.body else ''
                        text_content = ' '.join(body.stripped_strings) if body else ''
                        if text_content:
                            all_texts += f"Source: {file_path}\n{text_content}\n\n"
                            num_characters_scraped += len(text_content)
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")

              # ... (voorafgaande code voor scraping en opslaan van teksten in een tijdelijk bestand)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w+', encoding='utf-8') as temp_file:
            temp_file.write(all_texts)
            temp_file_path = temp_file.name
            print(f"Texts are saved in temporary file: {temp_file_path}")

        # Initialize Pinecone en OpenAI Embeddings
        pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_API_ENV"))
        embeddings_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

        # Laad tekstbestand en splits het in stukken
        loader = TextLoader(temp_file_path)  # Gebruik het pad van het tijdelijke bestand
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        # Inspecteer de attributen van een Document-object
        if docs:
            print("Beschikbare attributen in Document-object:", dir(docs[0]))

        # Neem aan dat het attribuut 'page_content' beschikbaar is, de rest van de code zou er zo uitzien
        chunks = [doc.page_content for doc in docs if hasattr(doc, 'page_content')]  # gebruik van hasattr om te controleren op het attribuut 'page_content'

        # Bereken het aantal tokens in elk stuk en druk het af
        token_counts = [len(chunk.split()) for chunk in chunks]
        print("Aantal tokens in elk stuk:", token_counts)

        # Voeg bronmetadata toe en maak een Pinecone-vectoropslag
        metadatas = [{"source": str(i)} for i in range(len(chunks))]
        pinecone_vectorstore = Pinecone.from_texts(
            chunks,
            embeddings_model,
            index_name="chatter",
            metadatas=metadatas,
            namespace=namespace
        )

    update_counter(num_characters_scraped)

    return jsonify({'numCharacters': num_characters_scraped})
    

@app.route('/scrape_single_page', methods=['POST'])
def scrape_single_page():
    # Controleer eerst of de karakterlimiet is bereikt
    if check_limit():
        return jsonify({'error': 'Karakterlimiet bereikt'}), 403

    url_to_scrape = request.form['singleWebsiteUrl']
    num_characters_scraped = 0  # Teller voor het aantal gescrapede tekens

    def download_and_clean_page(url, folder):
        try:
            r = requests.get(url)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                for tag in soup(['script', 'style', 'head', 'nav', 'navbar']):
                    tag.extract()
                body = soup.body if soup and soup.body else ''
                text_content = ' '.join(body.stripped_strings) if body else ''
                if text_content:
                    with open(os.path.join(folder, "single_page.txt"), "w", encoding="utf-8") as f:
                        f.write(text_content)
                    return len(text_content)
        except Exception as e:
            print(f"Error while processing URL {url}: {e}")
            return 0

    with tempfile.TemporaryDirectory() as folder:
        num_characters_scraped = download_and_clean_page(url_to_scrape, folder)

        if num_characters_scraped > 0:
            temp_file_path = os.path.join(folder, "single_page.txt")

            # Initialize Pinecone en OpenAI Embeddings
            pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_API_ENV"))
            embeddings_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

            # Laad tekstbestand en splits het in stukken
            loader = TextLoader(temp_file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)

            # Sla de chunks op in Pinecone
            chunks = [doc.page_content for doc in docs if hasattr(doc, 'page_content')]
            metadatas = [{"source": url_to_scrape} for _ in range(len(chunks))]
            pinecone_vectorstore = Pinecone.from_texts(
                chunks,
                embeddings_model,
                index_name="chatter",
                metadatas=metadatas,
                namespace=namespace
            )
    update_counter(num_characters_scraped)

    return jsonify({'numCharacters': num_characters_scraped})
    
ALLOWED_EXTENSIONS = {'txt', 'pdf'}  # Voeg 'pdf' toe aan de set van toegestane extensies

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    
@app.route('/process_file', methods=['POST'])
def process_file():
    # Controleer eerst of de karakterlimiet is bereikt
    if check_limit():
        return jsonify({'error': 'Karakterlimiet bereikt'}), 403

    if 'file' not in request.files:
        return jsonify({'error': 'Geen bestand ontvangen'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Geen bestandsnaam opgegeven'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        with tempfile.TemporaryDirectory() as folder:
            filepath = os.path.join(folder, filename)
            file.save(filepath)

            # Initialize Pinecone en OpenAI Embeddings
            pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_API_ENV"))
            embeddings_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

            # Laad het opgeslagen bestand en splits het in stukken
            if filename.endswith('.pdf'):
                loader = PyPDFium2Loader(filepath)
            else:
                loader = TextLoader(filepath)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)

            # Sla de chunks op in Pinecone
            chunks = [doc.page_content for doc in docs if hasattr(doc, 'page_content')]
            metadatas = [{"source": filename} for _ in range(len(chunks))]
            pinecone_vectorstore = Pinecone.from_texts(
                chunks,
                embeddings_model,
                index_name="chatter",
                metadatas=metadatas,
                namespace=namespace
            )

            # Bereken het totale aantal karakters in alle chunks
            num_characters_processed = sum([len(chunk) for chunk in chunks])
            update_counter(num_characters_processed)

            return jsonify({'numCharacters': num_characters_processed})
    else:
        return jsonify({'error': 'Ongeldig bestandstype'})

def sanitize_text(text):
    """
    Verwijder of vervang tekens die niet in de Latin-1 set zitten.
    """
    # Vervang bekende probleemtekens
    text = text.replace("–", "-")  # En dash vervangen door normaal streepje
    # Verwijder alle andere tekens die niet in Latin-1 zitten
    return re.sub(r'[^\x00-\x7F]+', '', text)


    
@app.route('/export_to_pdf', methods=['GET'])
def export_to_pdf():
    archived_chats = fetch_archive_chats()
    current_chats = fetch_chats()
    all_chats = archived_chats + current_chats  # Eerst gearchiveerde chats, daarna huidige chats

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for chat in all_chats:
        chat_id, user_id, question, answer, timestamp, month_year = chat
        pdf.cell(200, 10, txt=sanitize_text(f"Chat ID: {chat_id}, Gebruiker ID: {user_id}"), ln=True)
        pdf.cell(200, 10, txt=sanitize_text(f"Vraag: {question}"), ln=True)
        pdf.cell(200, 10, txt=sanitize_text(f"Antwoord: {answer}"), ln=True)
        pdf.cell(200, 10, txt=sanitize_text(f"Tijd: {timestamp}"), ln=True)
        pdf.cell(200, 10, txt=sanitize_text(f"Maand Jaar: {month_year}"), ln=True)
        pdf.cell(200, 10, txt="---" * 30, ln=True)  # Scheidingslijn

    # Bewaar de PDF in een tijdelijk bestand
    file_path = "/tmp/chat_archive.pdf"
    pdf.output(file_path)

    # Stuur de PDF als een respons naar de gebruiker
    with open(file_path, "rb") as f:
        return f.read(), 200, {
            "Content-Type": "application/pdf",
            "Content-Disposition": f"attachment; filename=chat_archive.pdf"
        }

def fetch_chats(limit=None):
    conn = get_conn()
    cur = conn.cursor()
    if limit:
        cur.execute("SELECT * FROM chat_history ORDER BY timestamp DESC LIMIT %s", (limit,))
    else:
        cur.execute("SELECT * FROM chat_history")
    results = cur.fetchall()
    cur.close()
    release_conn(conn)
    return results

def fetch_archive_chats():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM chat_history_archive")
    results = cur.fetchall()
    cur.close()
    release_conn(conn)
    return results

        
if __name__ == '__main__':
    initialize_db()  # Voeg deze regel toe om de DB te initialiseren
    app.run(debug=True)


