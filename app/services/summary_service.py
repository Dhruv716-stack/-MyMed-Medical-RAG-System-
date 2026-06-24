from memory.summary_manager import (
    get_summary
)


def get_chat_summary():

    summary = get_summary()

    return {

        "summary": summary
    }