import re


def clean_output(text: str) -> str:
    """
    Cleans LLM output formatting:
    - Removes excessive \\n
    - Properly separates numbered points
    - Formats Introduction and Conclusion
    """

    # Convert escaped newlines to actual newlines
    text = text.replace("\\n", "\n")

    # Remove excessive spaces
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Separate numbered points properly
    text = re.sub(r"(\d+\.)", r"\n\1", text)

    # Separate citations
    text = re.sub(r"(\[Source:.*?\])", r"\n\1", text)

    # Format sections
    text = text.replace("Introduction:", "\nIntroduction:\n")
    text = text.replace("Conclusion:", "\nConclusion:\n")

    # Remove too many blank lines again
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()