import transformers
import shap


def get_sentiment(doc_clean, show_doc=20):
    classifier = transformers.pipeline('sentiment-analysis', return_all_scores=True)
    explainer = shap.Explainer(classifier)
    shap_values = explainer(doc_clean[:show_doc])

    pmodel = shap.models.TransformersPipeline(classifier, rescale_to_logits=True)

    explainer2 = shap.Explainer(pmodel, classifier.tokenizer)
    shap_values2 = explainer2(doc_clean[:show_doc])
    return shap.plots.text(shap_values2[:, :, 1])
