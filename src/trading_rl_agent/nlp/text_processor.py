"""
Text processing utilities for financial text analysis.

This module provides text preprocessing capabilities for:
- Text cleaning and normalization
- Tokenization
- Feature extraction
- Text vectorization
"""

import re
from dataclasses import dataclass


@dataclass
class ProcessedText:
    """Result of text processing."""

    original_text: str
    cleaned_text: str
    tokens: list[str]
    features: dict


class TextProcessor:
    """Text processor for financial text analysis."""

    def __init__(self) -> None:
        """Initialize the text processor."""
        # Common financial abbreviations
        self.financial_abbreviations = {
            "inc": "incorporated",
            "corp": "corporation",
            "ltd": "limited",
            "co": "company",
            "mkt": "market",
            "vol": "volume",
            "avg": "average",
            "max": "maximum",
            "min": "minimum",
            "pct": "percent",
            "yr": "year",
            "mo": "month",
            "wk": "week",
            "hr": "hour",
            "sec": "second",
        }

        # Stop words for financial text
        self.stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "its",
            "our",
            "their",
        }

    def process_text(self, text: str, remove_stop_words: bool = True) -> ProcessedText:
        """Process text for analysis.

        Args:
            text: Raw text to process
            remove_stop_words: Whether to remove stop words

        Returns:
            ProcessedText object with processed text and features
        """
        # Clean text
        cleaned_text = self._clean_text(text)

        # Tokenize
        tokens = self._tokenize(cleaned_text)

        # Remove stop words if requested
        if remove_stop_words:
            tokens = self._remove_stop_words(tokens)

        # Extract features
        features = self._extract_features(tokens)

        return ProcessedText(original_text=text, cleaned_text=cleaned_text, tokens=tokens, features=features)

    def process_batch(self, texts: list[str], remove_stop_words: bool = True) -> list[ProcessedText]:
        """Process multiple texts.

        Args:
            texts: List of texts to process
            remove_stop_words: Whether to remove stop words

        Returns:
            List of ProcessedText objects
        """
        return [self.process_text(text, remove_stop_words) for text in texts]

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()

        # Expand abbreviations
        text = self._expand_abbreviations(text)

        # Remove URLs
        text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove special characters but keep spaces, basic punctuation, currency symbols, and percentages
        text = re.sub(r"[^\w\s\.\,\!\?\-%\$\€\£\¥]", " ", text)

        # Remove extra whitespace
        return re.sub(r"\s+", " ", text).strip()

    def _expand_abbreviations(self, text: str) -> str:
        """Expand common financial abbreviations.

        Args:
            text: Text with abbreviations

        Returns:
            Text with expanded abbreviations
        """
        words = text.split()
        expanded_words = []

        for word in words:
            # Remove punctuation for abbreviation check
            clean_word = re.sub(r"[^\w]", "", word)
            if clean_word in self.financial_abbreviations:
                # Replace the word with expanded form
                expanded_word = word.replace(clean_word, self.financial_abbreviations[clean_word])
                expanded_words.append(expanded_word)
            else:
                expanded_words.append(word)

        return " ".join(expanded_words)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into words.

        Args:
            text: Cleaned text

        Returns:
            List of tokens
        """
        # Simple word tokenization
        tokens = text.split()

        # Remove empty tokens and strip punctuation from tokens
        return [token.strip(".,!?") for token in tokens if token.strip()]

    def _remove_stop_words(self, tokens: list[str]) -> list[str]:
        """Remove stop words from tokens.

        Args:
            tokens: List of tokens

        Returns:
            List of tokens with stop words removed
        """
        return [token for token in tokens if token.lower() not in self.stop_words]

    def _extract_features(self, tokens: list[str]) -> dict:
        """Extract features from tokens.

        Args:
            tokens: List of tokens

        Returns:
            Dictionary of features
        """
        if not tokens:
            return {
                "token_count": 0,
                "unique_tokens": 0,
                "avg_token_length": 0.0,
                "has_numbers": False,
                "has_currency": False,
                "has_percentages": False,
            }

        # Basic features
        token_count = len(tokens)
        unique_tokens = len(set(tokens))
        avg_token_length = sum(len(token) for token in tokens) / token_count

        # Check for numbers
        has_numbers = any(re.search(r"\d", token) for token in tokens)

        # Check for currency symbols
        has_currency = any(re.search(r"[\$\€\£\¥]", token) for token in tokens)

        # Check for percentages
        has_percentages = any(re.search(r"%", token) for token in tokens)

        return {
            "token_count": token_count,
            "unique_tokens": unique_tokens,
            "avg_token_length": avg_token_length,
            "has_numbers": has_numbers,
            "has_currency": has_currency,
            "has_percentages": has_percentages,
        }

    def get_vocabulary(self, processed_texts: list[ProcessedText]) -> set[str]:
        """Get vocabulary from processed texts.

        Args:
            processed_texts: List of ProcessedText objects

        Returns:
            Set of unique tokens
        """
        vocabulary = set()
        for processed_text in processed_texts:
            vocabulary.update(processed_text.tokens)
        return vocabulary
