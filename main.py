import json
import logging
import os
import pickle
from typing import Optional

import requests
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
import openai
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from slack_bolt import App
from slack_bolt.workflows.step import WorkflowStep

from search_index.vector_store import OUTPUT_DIR, STORE_NAME

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")  # From https://platform.openai.com/account/org-settings
VECTOR_STORE_PATH = os.getenv("INDEX_PATH", f'{OUTPUT_DIR}/{STORE_NAME}.pkl')

TEST_MODE = os.getenv("TEST_MODE", 'True').lower() in ('true', '1', 't')

# TODO: dynamically look up bot user id
bot_user_id = "U04SMU6EX3L"


# Event API & Web API
app = App(token=SLACK_BOT_TOKEN)
client = WebClient(SLACK_BOT_TOKEN)


def get_workflow_msg_ts(channel_id):
    response = client.conversations_history(channel=channel_id, limit=20)
    messages = response["messages"]
    ts = None
    for msg in messages:
        try:
            if msg['username'] == 'ml-help' and msg['subtype'] == 'bot_message':
                ts = msg['ts']
                return ts
        except:
            pass
    return ts


# TODO: don't expose it
# Load cached index of document embeddings
with open(VECTOR_STORE_PATH, "rb") as f:
    vector_store = pickle.load(f)


def get_answer(
    question: str,
    num_docs: int = 4,
    temperature: int = 0,
    prune_invalid_sources: bool = True,
) -> Optional[str]:
    """Query OpenAI to answer question using the k-closest documents in the search index"""

    # Build the LLM chain
    # https://python.langchain.com/docs/integrations/llms/openai
    llm = OpenAI(
        openai_api_key=OPENAI_API_KEY,
        openai_organization=OPENAI_ORG_ID,
        temperature=temperature)
    chain = load_qa_with_sources_chain(llm)

    # Query response for the k-closest documents
    openai_resp = chain(
        {
            "input_documents": vector_store.similarity_search(question, k=num_docs),
            "question": question,
        },
        return_only_outputs=True,
    )["output_text"]

    # Validation logic
    # TODO: this is buggy
    if prune_invalid_sources:
        lines = openai_resp.split("\n")
        if "SOURCES:" not in lines:
            return None  # Don't reply if we can't cite our sources
        sources_start = lines.index("SOURCES:") + 1
        num_valid_sources = len(lines[sources_start:])
        invalid_idxs = []
        for i, url in enumerate(lines[sources_start:], start=sources_start):
            source_resp = requests.get(url)
            if not source_resp.ok:
                num_valid_sources -= 1
                invalid_idxs.append(i)
        if num_valid_sources == 0:
            return None  # All URLs were fake
        for i in invalid_idxs[::-1]:
            lines.pop(i)
        openai_resp = "\n".join(lines)
    return openai_resp


# This gets activated when the bot is tagged in a channel
# Pass Step to set up listeners
with open('ui.json', 'r') as f:
    data = json.load(f)
    blocks = data['blocks']

ws = WorkflowStep.builder('add_task')
@ws.edit
def edit(ack, step, configure):
    ack()
    configure(blocks=blocks)


@ws.save
def save(ack, view, update):
    ack()
    values = view['state']['values']
    priority = values["block1"]["priority"]["value"]
    infra_model = values["block2"]["infra_model"]["value"]
    service_type = values["block3"]["service_type"]["value"]
    mlfaq = values["block4"]["mlfaq"]["value"]
    request_text = values["block5"]["input"]["value"]
    channel = values["block6"]["input"]["value"].split("==")[0][2:]
    print(channel)
    inputs = {
        "priority": {"value": priority},
        "infra_model": {"value": infra_model},
        "service": {"value": service_type},
        "mlfaq": {"value": mlfaq},
        "request": {"value": request_text},
        "channel": {"value": channel}
    }
    outputs = [
        {
            "type": "text",
            "name": "priority",
            "label": "priority",
        },
        {
            "type": "text",
            "name": "infra_model",
            "label": "infra_model",
        },
        {
            "type": "text",
            "name": "service",
            "label": "service",
        },
        {
            "type": "text",
            "name": "mlfaq",
            "label": "mlfaq",
        },
        {
            "type": "text",
            "name": "request",
            "label": "request",
        },
        {
            "type": "text",
            "name": "channel",
            "label": "channel",
        }
    ]
    update(inputs=inputs, outputs=outputs)


def execute_step(step, complete, fail, body, ack):
    ack()
    print('called....')
    inputs = step["inputs"]
    channel_str = body['event']['workflow_step']['inputs']['channel']['value']
    channel_id = channel_str.replace('<#', '').replace('>', '')
    outputs = {
        "priority": inputs["priority"]["value"],
        "infra_model": inputs["infra_model"]["value"],
        "service": inputs["service"]["value"],
        "mlfaq": inputs["mlfaq"]["value"],
        "request": inputs["request"]["value"],
        "channel": channel_id
    }
    ts = get_workflow_msg_ts(channel_id)

    response = get_answer(question=outputs.get("request"), prune_invalid_sources=False)
    client.chat_postMessage(channel=outputs["channel"], text=response, thread_ts=ts)
    try:
        complete(outputs={})
    except Exception as err:
        fail(error={"message": f"Something wrong! {err}"})


@ws.execute(lazy=[execute_step])
def execute(ack):
    pass
# Create a new WorkflowStep instance

app.step(ws)


@app.event("app_mention")
def handle_message_events(body):
    channel_id = body["event"]["channel"]
    msg_ts = body["event"]["event_ts"]
    thread_ts = body["event"].get("thread_ts")

    logger.info(channel_id)
    logger.info(thread_ts)
    logger.info(msg_ts)

    # check if it is a message from a thread or a parent message without reply
    if thread_ts:
        replies = client.conversations_replies(channel=channel_id, ts=thread_ts, include_all_metadata=False)

        gpt_api_messages = []
        for reply in replies.get("messages"):
            reply_user = reply.get("user")
            reply_text = reply.get("text")
            msg = {"role": "assistant" if reply_user == bot_user_id else "user", "content": strip_bot_id(reply_text)}
            gpt_api_messages.append(msg)

    else:
        gpt_api_messages = [
            {"role": "user", "content": strip_bot_id(body["event"]["text"])}
        ]

    logger.info(gpt_api_messages)

    if TEST_MODE:
        # no Openai API call
        res_msg = "Testing"
        client.chat_postMessage(
            channel=body["event"]["channel"],
            thread_ts=body["event"]["event_ts"],
            text=res_msg
        )
    else:
        openai.api_key = OPENAI_API_KEY
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=gpt_api_messages,
            max_tokens=2048,
            temperature=0.7,
        )

        logger.info(response)

        # Reply to thread
        res_msg = response.choices[0].message.content
        client.chat_postMessage(
            channel=body["event"]["channel"],
            thread_ts=body["event"]["event_ts"],
            text=f"{res_msg}"
        )

    logger.info(res_msg)


def strip_bot_id(message: str):
    at_pattern = f"<@{bot_user_id}>"
    return message.replace(at_pattern, "", 1000).strip()


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s")

    SocketModeHandler(app, SLACK_APP_TOKEN).start()
