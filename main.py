import logging
import os

import openai
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from slack_bolt import App


SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

TEST_MODE = os.getenv("TEST_MODE", 'True').lower() in ('true', '1', 't')


# Event API & Web API
app = App(token=SLACK_BOT_TOKEN)
client = WebClient(SLACK_BOT_TOKEN)

# TODO: dynamically look up bot user id
bot_user_id = "U04SMU6EX3L"


# This gets activated when the bot is tagged in a channel
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
