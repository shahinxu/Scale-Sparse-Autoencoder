import re

def sanitize_filename(text):
    """Clean filename by removing unsafe characters"""
    safe_text = re.sub(r'[^\w\s-]', '', text)
    safe_text = re.sub(r'[-\s]+', '_', safe_text)
    return safe_text.strip('_')
