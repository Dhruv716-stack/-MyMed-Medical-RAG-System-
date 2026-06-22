from generation.retrieve_model import model


def general_chat(
    query,
    memory
):

    prompt = f"""
give a helpful and elaborated response to the user query giving information about the query in detail.and also use the previous conversation to give a more accurate and helpful response to the user query.also dont report user about no previous data if you dont have any previous data just give a response to the user query without using any previous data if you dont have any previous data.
Previous conversation:

{memory}

User:

{query}
"""

    response = model.invoke(
        prompt
    )

    return response.content