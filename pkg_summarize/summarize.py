from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from transformers import pipeline


def extract_summary(doc):
    summarizer = pipeline("summarization")
    summary = summarizer(doc, truncation=True)
    return summary[0]["summary_text"]


def sumy_summarizer(input_text, lines=3):
    parser = PlaintextParser.from_string(input_text, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, lines)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result