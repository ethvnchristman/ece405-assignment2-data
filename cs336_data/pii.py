import re

EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')

PHONE_PATTERN = re.compile(
    r'(?<!\d)'
    r'(?:\+?1[\s\-.]?)?'
    r'(?:\(\d{3}\)[\s\-.]?|\d{3}[\s\-.]?)'
    r'\d{3}[\s\-.]?\d{4}'
    r'(?!\d)'
)

_OCTET = r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)'
IP_PATTERN = re.compile(
    r'(?<!\d)'
    + rf'{_OCTET}\.{_OCTET}\.{_OCTET}\.{_OCTET}'
    + r'(?!\d)'
)


def mask_emails(text: str) -> tuple[str, int]:
    matches = EMAIL_PATTERN.findall(text)
    masked = EMAIL_PATTERN.sub('|||EMAIL_ADDRESS|||', text)
    return masked, len(matches)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    matches = PHONE_PATTERN.findall(text)
    masked = PHONE_PATTERN.sub('|||PHONE_NUMBER|||', text)
    return masked, len(matches)


def mask_ips(text: str) -> tuple[str, int]:
    matches = IP_PATTERN.findall(text)
    masked = IP_PATTERN.sub('|||IP_ADDRESS|||', text)
    return masked, len(matches)
