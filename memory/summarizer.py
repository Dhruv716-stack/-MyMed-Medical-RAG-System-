from generation.retrieve_model import model

from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

from rag_pipeline.config import (
    SUMMARY_MAX_WORDS
)


SYSTEM_PROMPT = f"""
You maintain long-term conversation memory.

Update the summary using the recent messages.

Preserve:

- important user goals
- project details
- preferences
- previous decisions

Remove:

- repetition
- temporary information

Maximum length:
{SUMMARY_MAX_WORDS} words.

Return only the updated summary.
"""


def update_summary(

    old_summary: str,

    recent_history: str

):

    messages = [

        SystemMessage(

            content=SYSTEM_PROMPT
        ),

        HumanMessage(

            content=f"""

Current Summary:

{old_summary}


Recent Messages:

{recent_history}

"""
        )
    ]

    response = model.invoke(
        messages
    )

    return response.content