from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    try:
        html_str = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes)
        try:
            html_str = html_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            html_str = html_bytes.decode("utf-8", errors="replace")
    return extract_plain_text(html_str)
