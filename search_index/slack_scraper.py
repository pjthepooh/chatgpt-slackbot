"""
usage: SLACK_BOT_TOKEN=<slack_bot_token> python -m search_index.slack_scraper
"""

import os
import re
import logging
from datetime import datetime

from slack_sdk import WebClient

from search_index.model import SlackBotRequest


logger = logging.getLogger(__name__)


SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")

START_TS = "2023-07-01"
MESSAGE_LIMIT = 1000
CHANNEL_ID = "C05J1BGHJGP" # r3-2023-ml-help-gpt channel
# CHANNEL_ID = "CBPLJLSD9"  # ml-help
BOT_ID = "B05JJ9UUSAF"  # ml-help workflow
FETCH_URL = True
POST_MESSAGE = False


def convert_to_unix_timestamp(date_str) -> float:
    return datetime.strptime(date_str, "%Y-%m-%d").timestamp()


class SlackScraper:

    def __init__(self, channel_id=CHANNEL_ID, bot_id=BOT_ID, oldest=convert_to_unix_timestamp(START_TS)):
        self._client = WebClient(SLACK_BOT_TOKEN)
        self.channel_id = channel_id
        self.bot_id = bot_id
        self.oldest = str(oldest)

    @property
    def client(self):
        return self._client

    def get_messages(self):
        return self.client.conversations_history(
            channel=self.channel_id,
            oldest=self.oldest,
            limit=MESSAGE_LIMIT,
        )

    def get_bot_messages(self):
        response = self.get_messages()
        messages = response.get('messages')
        return [m for m in messages if m.get('bot_id') == self.bot_id]

    def get_bot_requests(self, fetch_url=True):
        bot_messages = self.get_bot_messages()

        bot_requests = []
        for msg in bot_messages:
            ts = msg.get("ts")
            blocks = msg.get("blocks")
            request_text = ""
            permalink = ""

            # [To be deprecated] old/existing workflow format
            if len(blocks) == 1:
                pattern = r"\n\*(?:Request:)?\*(.*)"
                text = msg.get("text")
                match = re.search(pattern, text)
                if match:
                    request_text = match.group(1)
                    request_text = request_text.strip()

            # new workflow format
            else:
                request_block = blocks[-2]
                text = request_block.get("text").get("text")
                request_text = text.replace("*Request *\n", "").strip()

            if request_text:
                if fetch_url:
                    permalink_response = self.client.chat_getPermalink(channel=self.channel_id, message_ts=ts)
                    permalink = permalink_response['permalink']

                bot_request = SlackBotRequest(
                    ts=str(ts),
                    text=request_text,
                    url=permalink
                )
                bot_requests.append(bot_request)

        return bot_requests

    def delete_message(self, ts):
        self.client.chat_delete(channel=self.channel_id, ts=ts)

    def create_message(self, text):
        self.client.chat_postMessage(
            channel=self.channel_id,
            text=text,
            mrkdwn=True,
            unfurl_links=False
        )


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s")

    start_ts = convert_to_unix_timestamp(START_TS)
    scraper = SlackScraper(channel_id=CHANNEL_ID, bot_id=BOT_ID, oldest=start_ts)

    logger.info(f"Scrapping messages up to {START_TS}")
    slack_bot_requests = scraper.get_bot_requests(fetch_url=FETCH_URL)
    logger.info(f"Number of requests retrieved: {len(slack_bot_requests)}")
    logger.info("\n"+"\n".join([r.__str__() for r in slack_bot_requests]))

    if POST_MESSAGE:
        formatted_texts = []
        post_message = f"ML Help Requests from {START_TS}:"
        for idx, request in enumerate(slack_bot_requests):
            formatted_text = request.to_slack_message()
            post_message += f'\n{str(idx+1)}. {formatted_text}'

        scraper.create_message(post_message)
