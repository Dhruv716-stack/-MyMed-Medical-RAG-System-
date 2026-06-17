from generation.retrieve_model import model


def general_chat(
    query,
    memory
):

    prompt = f"""

Previous conversation:

{memory}

User:

{query}
"""

    response = model.invoke(
        prompt
    )

    return response.content