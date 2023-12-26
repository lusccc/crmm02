import logging

logger = logging.getLogger(__name__)


def agg_text_columns_func(empty_row_values, replace_text, texts):
    """replace empty texts or remove empty text str from a list of text str"""
    processed_texts = []
    for text in texts.astype('str'):
        if text not in empty_row_values:
            processed_texts.append(text)
        else:
            if replace_text is not None:
                processed_texts.append(replace_text)
    return processed_texts
