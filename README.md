# chatgpt-slackbot
Slack App with ChatGPT integration

## Run the app locally
1. Create vector store
    ```shell
    SLACK_BOT_TOKEN=<slack_bot_token> OPENAI_API_KEY=<open_ai_key> python -m search_index.vector_store
    ```
   It saves documents in the `input/` directory to vector store.
   You can flip `SCRAPE_SLACK` to `True` to scrape historical bot requests.

2. Start ml-help-gpt Bot
    ```shell
    SLACK_BOT_TOKEN=<slack_bot_token> SLACK_APP_TOKEN=<slack_api_token> OPENAI_API_KEY=<open_ai_key> python main.py
    ```

