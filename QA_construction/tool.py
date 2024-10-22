import sys
import os
import json
import re
import logging
from time import sleep
from typing import Any, Dict, Optional, List
from openai import OpenAI
# from transformers import BertTokenizer, BertModel
# from sentence_transformers import SentenceTransformer
# import torch
from config import *

# Ensure the logs directory exists before writing logs
try:
    LOG_DIR
except NameError:
    LOG_DIR = "logs"

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Set up logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'app.log'),
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Remove the logging for 'e' before it's properly defined
# The initial logging will be performed during exception handling
def try_run(func, *args, **kwargs):
    retry = 0
    while retry < max_try_num:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            retry += 1
            sleep(0.1 * retry)
            logging.error(f"Attempt {retry}: Exception occurred: {e}")
            logging.debug(f"Commandline: {sys.argv}")
    else:
        logging.critical("Max retry number reached")
        raise Exception("Max try number reached")


def replace_newlines(match: re.Match) -> str:
    # Replace \n and \r in the matched string
    return match.group(0).replace('\n', '\\n').replace('\r', '\\r')


def clean_json_str(json_str: str) -> str:
    """
    Handle possibly non-standard JSON format.
    """
    json_str = json_str.replace("None", "null")

    # Remove code block markers ``` in JSON strings
    match = re.search(r'```json(.*?)```', json_str, re.DOTALL)
    if match:
        json_str = match.group(1)
    match = re.search(r'```(.*?)```', json_str, re.DOTALL)
    if match:
        json_str = match.group(1)

    json_str = re.sub(r'("(?:\\.|[^"\\])*")', replace_newlines, json_str)
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    json_str = re.sub(r'\"\s+\"', '","', json_str)
    json_str = json_str.replace("True", "true")
    json_str = json_str.replace("False", "false")
    return json_str


def txt2obj(text: str) -> Optional[Dict[str, Any]]:
    try:
        text = clean_json_str(text)
        return json.loads(text)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON: {e}")
        return None


def _get_chat_completion(chat: Any, return_json: bool, model: str, max_tokens: int, keys: Optional[List[str]]) -> Any:
    if not isinstance(chat, list):
        chat = [{"role": "user", "content": chat}]
    client = OpenAI(api_key=personal_key, base_url=personal_base)
    chat_completion = client.chat.completions.create(
        model=model,
        messages=chat,
        response_format={"type": "json_object" if return_json else "text"},
        max_tokens=max_tokens,
        temperature=temperature_value,
        frequency_penalty=frequency_penalty_value,
        presence_penalty=0
    )
    if LOG:
        logging.info("Chat completion result: %s", chat_completion.choices[0].message.content)

    chat.append({"role": "assistant", "content": chat_completion.choices[0].message.content})
    obj = txt2obj(chat_completion.choices[0].message.content)
    if obj is None:
        raise Exception("Failed to parse JSON")
    if keys is not None:
        obj = tuple([obj[key] for key in keys])
        return *obj, chat
    return obj, chat


def get_chat_completion(chat: Any, return_json: bool = False, model: str = default_gpt_model, max_tokens: int = 4096, keys: Optional[List[str]] = None) -> Any:
    return try_run(_get_chat_completion, chat, return_json, model, max_tokens, keys)


def write2json(content: Any, responseTxtPath: str):
    with open(responseTxtPath, 'w', encoding='utf-8') as f_out:
        json.dump(content, f_out, indent=4, ensure_ascii=False)


def _get_vision_chat_completion(chat: Any, image_urls: list, return_json: bool, model: str, max_tokens: int, keys: Optional[List[str]]) -> Any:
    if not isinstance(chat, list):
        composeContent = [{"type": "text", "text": chat}]
        for image_url in image_urls:
            composeContent.append({"type": "image_url", "image_url": {"url": image_url}})
        chat = [{"role": "user", "content": composeContent}]
    client = OpenAI(api_key=personal_key, base_url=personal_base)
    chat_completion = client.chat.completions.create(
        model=model,
        messages=chat,
        response_format={"type": "json_object" if return_json else "text"},
        max_tokens=max_tokens,
        temperature=temperature_value,
        frequency_penalty=frequency_penalty_value,
        presence_penalty=0
    )
    if LOG:
        logging.info("Chat completion result: %s", chat_completion.choices[0].message.content)

    chat.append({"role": "assistant", "content": chat_completion.choices[0].message.content})
    obj = txt2obj(chat_completion.choices[0].message.content)
    if obj is None:
        raise Exception("Failed to parse JSON")
    if keys is not None:
        obj = tuple([obj[key] for key in keys])
        return *obj, chat
    return obj, chat


def get_vision_chat_completion(chat: Any, image_urls: list, return_json: bool = False, model: str = default_gpt_model, max_tokens: int = 4096, keys: Optional[List[str]] = None) -> Any:
    return try_run(_get_vision_chat_completion, chat, image_urls, return_json, model, max_tokens, keys)


if __name__ == "__main__":
    test_input = "请以JSON格式回复Hi!"
    try:
        response = get_chat_completion(test_input)
        print("完整响应内容:", response)
    except Exception as e:
        logging.critical(f"Failed during main execution: {e}")