import sys
from transformers import pipeline

def summarize_text(text, max_length=None, min_length=30, do_sample=False):
    # Load the summarization model
    summarizer = pipeline("summarization")

    # Set dynamic max_length based on input text length if not provided
    if max_length is None:
        input_length = len(text.split())
        # Set max_length to be approximately 30% of the input length, with a minimum of 30
        max_length = max(30, input_length // 3)

    # Summarize the text
    summary = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=do_sample
    )

    # Return the summarized text
    return summary[0]['summary_text']

if __name__ == "__main__":
    # Check if text is provided
    if len(sys.argv) < 2:
        print("Usage: python text_summarizer.py 'Your long article or paper text here'")
        sys.exit(1)

    # Get the article text from command line argument
    article_text = sys.argv[1]

    # Check if the input text is long enough
    if len(article_text.split()) < 10:
        print("Please provide a longer article (at least 10 words) for summarization.")
        sys.exit(1)

    # Summarize the article text
    summarized_text = summarize_text(article_text)

    # Print the summarized text
    print("\nSummarized Text:")
    print(summarized_text)
