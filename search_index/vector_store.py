"""
usage: SLACK_BOT_TOKEN=<slack_bot_token> OPENAI_API_KEY=<open_ai_key> python -m search_index.vector_store
"""

import logging
import json
import os
import pickle

from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter

from search_index.slack_scraper import SlackScraper
from search_index.util import load_files_from_directory

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-ada-002"
INDEX_NAME = "ml-help-gpt-index"
STORE_NAME = "vector_store"
INPUT_DIR = f'{os.path.dirname(os.path.realpath(__file__))}/input'
OUTPUT_DIR = f'{os.path.dirname(os.path.realpath(__file__))}/output'
SCRAPE_SLACK = False


def scrape_slack_documents():
    scraper = SlackScraper()
    logger.info(f"Scrapping slack messages.")
    bot_requests = scraper.get_bot_requests()
    return [req.to_document() for req in bot_requests]


def get_slack_documents(file_path, scrape_slack=False):
    logger.info(f"Loading slack input from {file_path}")
    if not os.path.exists(file_path):
        return []

    with open(file_path, 'r') as file:
        data = []
        for line in file:
            json_data = json.loads(line.strip())
            document = Document(
                page_content=json_data.get("page_content"),
                metadata=json_data.get("metadata"),
            )
            data.append(document)

    if scrape_slack:
        data += scrape_slack_documents()

    return data


def get_github_documents(dir_path):
    logger.info(f"Loading github input from {dir_path}")
    if not os.path.exists(dir_path):
        return []

    data = []
    loaded_files = load_files_from_directory(dir_path)
    for source, page_content in loaded_files.items():
        splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
        for chunk in splitter.split_text(page_content):
            data.append(Document(page_content=chunk, metadata={"source": source}))
    return data


def get_documents(input_path=INPUT_DIR, scrape_slack=False):
    # slack documents
    sources_slack = get_slack_documents(file_path=f'{input_path}/input_slack.txt', scrape_slack=scrape_slack)

    # github documents
    sources_github = get_github_documents(dir_path=f'{input_path}/github')

    return sources_slack + sources_github


def get_and_save_vector_store(embedding_model, input_path, output_path):
    sources = get_documents(input_path=input_path, scrape_slack=SCRAPE_SLACK)

    logger.info(f"Generating vector store from document with model {embedding_model}")
    vector_store = FAISS.from_documents(sources, OpenAIEmbeddings(model=embedding_model))

    logger.info(f"Saving vector store in {output_path}")
    vector_store.save_local(folder_path=output_path, index_name=INDEX_NAME)
    with open(f'{output_path}/{STORE_NAME}.pkl', "wb") as f:
        pickle.dump(vector_store, f)
    return vector_store


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s")

    vector_store = get_and_save_vector_store(
        embedding_model=EMBEDDING_MODEL,
        input_path=INPUT_DIR,
        output_path=OUTPUT_DIR,
    )
