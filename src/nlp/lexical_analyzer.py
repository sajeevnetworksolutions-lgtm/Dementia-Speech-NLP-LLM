"""
Clinical Lexical & Syntactic Analyzer.
Extracts linguistic biomarkers from spontaneous speech transcripts:
  - Type-Token Ratio (TTR) & Standardized TTR
  - Brunet's W & Honore's R (Vocabulary richness)
  - Yule's K (Lexical diversity index)
  - Pronoun-to-Noun Ratio (Anomic substitution indicator)
  - Hesitation / Filler word frequency ('um', 'uh', 'er', ellipses)
  - Mean Length of Utterance (MLU)
"""
import re
import math
from collections import Counter
from typing import Dict, List, Optional


class LexicalAnalyzer:
    def __init__(self):
        self.hesitation_markers = {
            "um", "uh", "er", "ah", "hmm", "eh", "like", "you know", "i mean", "sort of", "kind of"
        }
        self.pronouns = {
            "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself",
            "he", "him", "his", "himself", "she", "her", "hers", "herself",
            "it", "its", "itself", "we", "us", "our", "ours", "ourselves",
            "they", "them", "their", "theirs", "themselves", "this", "that", "these", "those",
            "something", "anything", "nothing", "someone", "anyone", "thing", "stuff"
        }
        # Common content nouns in Cookie Theft description task
        self.cookie_theft_nouns = {
            "boy", "girl", "mother", "woman", "cookie", "jar", "stool", "cabinet",
            "sink", "water", "overflow", "floor", "dish", "plate", "curtain",
            "window", "cup", "kitchen", "apron", "faucet"
        }

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize into lowercase words while preserving hesitation marks.
        """
        cleaned = re.sub(r'[^\w\s\'-]', ' ', text.lower())
        tokens = [w.strip() for w in cleaned.split() if w.strip()]
        return tokens

    def compute_brunet_w(self, n_tokens: int, n_types: int) -> float:
        """
        Brunet's W measure: W = N^(V^(-0.165)). Lower values indicate richer vocabulary.
        """
        if n_tokens <= 0 or n_types <= 0:
            return 10.0
        try:
            return float(n_tokens ** (n_types ** (-0.165)))
        except (ValueError, OverflowError, ZeroDivisionError):
            return 10.0

    def compute_honore_r(self, n_tokens: int, n_types: int, hapax_count: int) -> float:
        """
        Honore's R measure: R = 100 * log(N) / (1 - (V1 / V)). Higher values indicate richer vocabulary.
        """
        if n_tokens <= 1 or n_types <= 0:
            return 100.0
        ratio = hapax_count / n_types
        if abs(1.0 - ratio) < 1e-4:
            return float(100.0 * math.log(max(n_tokens, 2)))
        try:
            return float((100.0 * math.log(n_tokens)) / (1.0 - ratio))
        except (ValueError, ZeroDivisionError):
            return 100.0

    def compute_yules_k(self, tokens: List[str]) -> float:
        """
        Yule's Characteristic K: Measures lexical constancy independent of text length.
        """
        n = len(tokens)
        if n <= 1:
            return 0.0
        counts = Counter(tokens)
        freq_spectrum = Counter(counts.values())
        s1 = n
        s2 = sum(r * r * v for r, v in freq_spectrum.items())
        k = 10000.0 * (s2 - s1) / (s1 * s1 + 1e-6)
        return float(max(0.0, k))

    def analyze_transcript(self, text: str) -> Dict[str, float]:
        """
        Complete clinical linguistic profiling of spontaneous speech transcript.
        """
        tokens = self.tokenize(text)
        n_tokens = len(tokens)
        if n_tokens == 0:
            return {
                "token_count": 0,
                "type_count": 0,
                "type_token_ratio": 0.0,
                "brunet_w": 15.0,
                "honore_r": 100.0,
                "yules_k": 0.0,
                "hesitation_count": 0,
                "hesitation_rate": 0.0,
                "pronoun_ratio": 0.0,
                "noun_ratio": 0.0,
                "pronoun_to_noun_ratio": 1.0,
                "mlu_words": 0.0,
                "cookie_theft_key_concepts": 0.0,
            }

        counts = Counter(tokens)
        n_types = len(counts)
        hapax_legomena = sum(1 for w, c in counts.items() if c == 1)

        ttr = n_types / n_tokens
        brunet_w = self.compute_brunet_w(n_tokens, n_types)
        honore_r = self.compute_honore_r(n_tokens, n_types, hapax_legomena)
        yules_k = self.compute_yules_k(tokens)

        # Hesitation & Filler Count
        hesitations = sum(counts[h] for h in self.hesitation_markers if h in counts)
        # Also check ellipsis markers
        ellipsis_count = len(re.findall(r'\.\.\.|\u2026', text))
        total_hesitations = hesitations + ellipsis_count
        hesitation_rate = total_hesitations / n_tokens

        # Pronoun vs Noun Anomic Substitution
        pronoun_count = sum(counts[p] for p in self.pronouns if p in counts)
        cookie_concepts = sum(1 for k in self.cookie_theft_nouns if k in text.lower())

        # Rough heuristic for sentences/utterances
        sentences = [s.strip() for s in re.split(r'[.!?\n]+', text) if s.strip()]
        num_sentences = max(1, len(sentences))
        mlu = n_tokens / num_sentences

        pronoun_ratio = pronoun_count / n_tokens
        noun_ratio = cookie_concepts / max(1, n_tokens)
        pronoun_to_noun_ratio = pronoun_count / max(1, cookie_concepts)

        return {
            "token_count": n_tokens,
            "type_count": n_types,
            "type_token_ratio": float(ttr),
            "brunet_w": float(brunet_w),
            "honore_r": float(honore_r),
            "yules_k": float(yules_k),
            "hesitation_count": total_hesitations,
            "hesitation_rate": float(hesitation_rate),
            "pronoun_ratio": float(pronoun_ratio),
            "noun_ratio": float(noun_ratio),
            "pronoun_to_noun_ratio": float(pronoun_to_noun_ratio),
            "mlu_words": float(mlu),
            "cookie_theft_key_concepts": float(cookie_concepts),
        }
