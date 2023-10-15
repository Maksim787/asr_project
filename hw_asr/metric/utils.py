import editdistance


def _handle_empty_target_text(predicted_text) -> float:
    if predicted_text:
        return 1
    return 0


def calc_cer(target_text, predicted_text) -> float:
    if not target_text:
        return _handle_empty_target_text(predicted_text)
    target_text_words = target_text.split(' ')
    return editdistance.eval(target_text_words, predicted_text.split(' ')) / len(target_text_words) * 100


def calc_wer(target_text, predicted_text) -> float:
    if not target_text:
        return _handle_empty_target_text(predicted_text)
    return editdistance.eval(target_text, predicted_text) / len(target_text) * 100
