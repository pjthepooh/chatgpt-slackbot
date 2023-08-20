from typing import NamedTuple

from langchain.docstore.document import Document


class SlackBotRequest(NamedTuple):
    ts: str
    text: str
    url: str

    def to_slack_message(self):
        return f'{self.text} <{self.url}|(url)>'

    def to_document(self):
        return Document(
            page_content=self.text,
            metadata={"source": self.url},
        )
