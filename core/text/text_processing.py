import re
from typing import Callable, List, Optional, Tuple

import numpy as np

# Markdown-like style pattern: ***bold italic***, **bold**, *italic*
STYLE_PATTERN = re.compile(r"(\*{1,3})(.*?)(\1)")


def is_latin_style_language(language_name: str) -> bool:
    """
    Determines if a language typically uses Latin script and hyphenation.
    This is used to decide whether to apply automatic hyphenation logic.
    """
    latin_style_languages = {
        "english",
        "french",
        "spanish",
        "german",
        "italian",
        "portuguese",
        "dutch",
        "polish",
        "czech",
        "swedish",
        "danish",
        "norwegian",
        "finnish",
        "hungarian",
        "romanian",
        "turkish",
        "vietnamese",
        "indonesian",
        "malay",
        "tagalog",
    }
    return language_name.lower() in latin_style_languages


def parse_styled_segments(text: str) -> List[Tuple[str, str]]:
    """
    Parses text with markdown-like style markers into segments.

    Args:
        text (str): Input text potentially containing ***bold italic***, **bold**, *italic*.

    Returns:
        List[Tuple[str, str]]: List of (segment_text, style_name) tuples.
                               style_name is one of "regular", "italic", "bold", "bold_italic".
    """
    segments = []
    last_end = 0
    for match in STYLE_PATTERN.finditer(text):
        start, end = match.span()
        marker = match.group(1)
        content = match.group(2)

        if start > last_end:
            segments.append((text[last_end:start], "regular"))

        style = "regular"
        if len(marker) == 3:
            style = "bold_italic"
        elif len(marker) == 2:
            style = "bold"
        elif len(marker) == 1:
            style = "italic"

        segments.append((content, style))
        last_end = end

    if last_end < len(text):
        segments.append((text[last_end:], "regular"))

    return [(txt, style) for txt, style in segments if txt]

def tokenize_styled_text(text: str) -> List[Tuple[str, bool]]:
    tokens: List[Tuple[str, bool]] = []
    last_end = 0
    CJK_REGEX = re.compile(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf]')
    def process_plain_text(raw_text: str, is_styled: bool, marker: str = ""):
        # First, split by whitespace as before
        parts = raw_text.split()
        for part in parts:
            # For each part, check if it contains CJK characters
            # We want to split CJK characters but keep English words as chunks
            sub_tokens = []
            current_latin_chunk = ""
            
            for char in part:
                if CJK_REGEX.match(char):
                    # If we have a Latin chunk saved up, push it first
                    if current_latin_chunk:
                        sub_tokens.append(current_latin_chunk)
                        current_latin_chunk = ""
                    # Push the CJK character as its own token
                    sub_tokens.append(char)
                else:
                    # Collect Latin/ASCII characters
                    current_latin_chunk += char
            
            if current_latin_chunk:
                sub_tokens.append(current_latin_chunk)
            
            # Wrap and add to main tokens list
            for sub in sub_tokens:
                token_val = f"{marker}{sub}{marker}" if is_styled else sub
                tokens.append((token_val, is_styled))

    for match in STYLE_PATTERN.finditer(text):
        start, end = match.span()
        # Process preceding plain text
        if start > last_end:
            process_plain_text(text[last_end:start], False)

        # Process styled block
        marker = match.group(1)
        content = match.group(2)
        if content:
            process_plain_text(content, True, marker)

        last_end = end

    # Process trailing text
    if last_end < len(text):
        process_plain_text(text[last_end:], False)

    return tokens


def try_hyphenate_word(
    word_str: str,
    min_word_length: int,
    width_test_func: Callable[[str], bool],
) -> Optional[List[str]]:
    """
    Attempts to split a word into two parts with a hyphen such that each part passes the width test.

    This is a generic hyphenation function that doesn't know about fonts or rendering.
    It uses a callback function to test if each part fits.

    Args:
        word_str: The word to potentially hyphenate
        min_word_length: Minimum word length to attempt hyphenation
        width_test_func: Function that takes a string and returns True if it fits

    Returns:
        List of two strings (the split parts) if successful, None otherwise
    """
    match = re.match(r"^(\W*)([\w\-]+)(\W*)$", word_str)
    if not match:
        return None

    leading_punc, core_word, trailing_punc = match.groups()

    if len(core_word) < min_word_length:
        return None

    def _split_with_single_hyphen(base: str, idx: int) -> Tuple[str, str]:
        ch_before = base[idx - 1] if idx > 0 else ""
        ch_at = base[idx] if idx < len(base) else ""
        if ch_at == "-":
            left = base[: idx + 1]
            right = base[idx + 1 :]
        elif ch_before == "-":
            left = base[:idx]
            right = base[idx:]
        else:
            left = base[:idx] + "-"
            right = base[idx:]
        if left.endswith("-") and right.startswith("-"):
            right = right[1:]
        return left, right

    # Try splitting at existing hyphens first
    if "-" in core_word:
        hyphen_positions = [i for i, ch in enumerate(core_word) if ch == "-"]
        mid = len(core_word) // 2
        hyphen_positions.sort(key=lambda i: abs(i - mid))
        for pos in hyphen_positions:
            if pos <= 0 or pos >= len(core_word) - 1:
                continue
            left_part = core_word[: pos + 1]
            right_part = core_word[pos + 1 :]

            final_left_part = leading_punc + left_part
            final_right_part = right_part + trailing_punc

            if width_test_func(final_left_part) and width_test_func(final_right_part):
                return [final_left_part, final_right_part]

    # Try splitting at various positions
    mid = len(core_word) // 2
    candidate_indices: List[int] = []
    max_d = max(mid, len(core_word) - mid)
    for d in range(0, max_d):
        left_idx = mid - d
        right_idx = mid + d
        if 2 <= left_idx < len(core_word) - 2:
            candidate_indices.append(left_idx)
        if 2 <= right_idx < len(core_word) - 2 and right_idx != left_idx:
            candidate_indices.append(right_idx)

    for idx in candidate_indices:
        left_part, right_part = _split_with_single_hyphen(core_word, idx)

        final_left_part = leading_punc + left_part
        final_right_part = right_part + trailing_punc

        if width_test_func(final_left_part) and width_test_func(final_right_part):
            return [final_left_part, final_right_part]

    return None


def find_optimal_breaks_dp(
    tokens: List[str],
    max_width: float,
    word_width_func: Callable[[str], float],
    space_width: float,
    badness_exponent: float = 3.0,
    hyphen_penalty: float = 1000.0,
) -> Optional[List[str]]:
    """
    Pragmatic Knuth-Plass style DP to find globally optimal line breaks.

    This is a pure algorithm that doesn't know about fonts or rendering.
    It uses callback functions to get widths.

    Args:
        tokens: List of word tokens
        max_width: Maximum allowed line width
        word_width_func: Function that takes a word and returns its width
        space_width: Width of a space character
        badness_exponent: Exponent for badness calculation (higher = prefer tighter lines)
        hyphen_penalty: Penalty for lines ending with hyphens

    Returns:
        List of lines (strings) if successful, None if impossible to fit
    """
    try:
        if not tokens:
            return []

        # Calculate widths for all tokens
        token_w: List[float] = [word_width_func(t) for t in tokens]
        # Helper to check if a token is CJK
        def is_cjk_token(t: str) -> bool:
            # Strip style markers like ** or // if present
            core = re.sub(r'^[\W_]+|[\W_]+$', '', t) 
            return bool(re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', core))
        N = len(tokens)
        min_cost: List[float] = [float("inf")] * (N + 1)
        path: List[int] = [0] * (N + 1)
        min_cost[0] = 0.0

        for i in range(1, N + 1):
            line_width = 0.0
            for j in range(i - 1, -1, -1):
                if j < i - 1:
                    current_tok = tokens[j]
                    next_tok = tokens[j + 1]
                    if not is_cjk_token(current_tok) and not is_cjk_token(next_tok):
                        line_width += space_width        
                                
                line_width += token_w[j]

                if line_width > max_width:
                    break

                slack = max_width - line_width
                badness = pow(slack, badness_exponent)

                # Add hyphen penalty if line ends with hyphen (support styled markers)
                last_token = tokens[i - 1] if i > 0 else ""
                ends_with_hyphen = last_token.endswith("-")
                if not ends_with_hyphen:
                    styled_match = STYLE_PATTERN.match(last_token)
                    if styled_match:
                        ends_with_hyphen = styled_match.group(2).endswith("-")
                if ends_with_hyphen:
                    badness += hyphen_penalty

                total_cost = min_cost[j] + badness
                if total_cost < min_cost[i]:
                    min_cost[i] = total_cost
                    path[i] = j

        if not np.isfinite(min_cost[N]):
            return None

        lines: List[str] = []
        curr = N
        while curr > 0:
            prev = path[curr]
            # THE SECOND FIX: Join with "" for CJK, " " for Latin
            line_tokens = tokens[prev:curr]
            line_str = ""
            for k, t in enumerate(line_tokens):
                if k > 0:
                    # Only add space if both are Latin
                    if not is_cjk_token(line_tokens[k-1]) and not is_cjk_token(t):
                        line_str += " "
                line_str += t
            lines.insert(0, line_str)
            curr = prev
        return lines
    
    except Exception:
        return None
