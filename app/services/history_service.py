from memory.memory_manager import (
    get_recent_history
)


def get_history():

    history = get_recent_history()

    return {

        "history": history
    }