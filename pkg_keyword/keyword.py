from keybert import KeyBERT


def extract(doc, n_grams=2):
    final_doc = " ".join(doc)
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(final_doc, keyphrase_ngram_range=(1, n_grams),
                                         stop_words='english', use_mmr=True,
                                         top_n=10)
    return keywords