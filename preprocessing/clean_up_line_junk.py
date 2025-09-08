import json
from tqdm import tqdm
import ahocorasick
import re
import string
from collections import Counter
from itertools import pairwise
from nltk.tokenize import sent_tokenize

from utils.file_operator import yield_corpus, yield_values_from_jsonl_file, yield_values_from_text_file, get_line_count

class FilteringStatistics:
        
    def __init__(self):
        self.reasons = {"line_filter": Counter(), "sentence_filter": Counter()}
        self.total_docs = 0
        self.total_lines = 0
        self.total_sentences = 0
        self.sent_for_sentence_splitting = 0
        self.seen_sentences = 0
        self.unmodified = 0
        self.modified = 0
        self.empty = 0
    
    def __str__(self):
        return f"""
--- STATISTICS ---
Total documents: {self.total_docs}
Total lines: {self.total_lines}
Total sentences: {self.total_sentences}
Lines sent for sentence splitting: {self.sent_for_sentence_splitting} ({self.sent_for_sentence_splitting / self.total_lines:.2%})
Seen sentences: {self.seen_sentences} ({self.seen_sentences / self.total_sentences:.2%})
Unmodified documents: {self.unmodified} ({self.unmodified / self.total_docs:.2%})
Modified documents: {self.modified} ({self.modified / self.total_docs:.2%})
Empty documents: {self.empty} ({self.empty / self.total_docs:.2%})
Considered junk:
    {self.reasons["line_filter"].total()} lines ({self.reasons["line_filter"].total() / self.total_lines:.2%} of all lines)
    {self.reasons["sentence_filter"].total()} sentences ({self.reasons["sentence_filter"].total() / self.total_sentences:.2%} of all sentences)

--- Filtering reasons ---
{self.format_reasons()}
"""
    
    def format_reasons(self):
        reasons_strings = []

        def get_sorted_counts(dict_key):
            for reason, count in self.reasons[dict_key].most_common():
                reasons_strings.append(f"{reason}: {count}")

        for key in self.reasons:
            get_sorted_counts(key)

        return "\n".join(reasons_strings)

    @staticmethod
    def format_identifier(reason:str):
        return f"***{reason.upper()}***"

class PatternMatcher:
    def __init__(self):

        self.punct_chars = set(string.punctuation)

        # Regex for lowercasing and removing all non-alphanumeric characters except for dots, colons, en dashes, quote marks and less than characters
        self.normalizing_regex = re.compile(r'[^:\.–"“<\w]')
        # Regex for matching dates in format (d)d.(m)m. OR yyyy
        self.date_pattern = re.compile(r'((0?[1-9]|[12][0-9]|3[01])\.(0?[1-9]|1[0-2])\.)|(\d{4})')
        # Regex for matching 'siirryt toiseen palveluun', anything inside the same parenthesis or brackets, and surrounding whitespace
        self.link_junk_pattern = re.compile(r'\s*\([^()\[\]]*siirryt toiseen palveluun[^()\[\]]*\)\s*|\s*\[[^()\[\]]*siirryt toiseen palveluun[^()\[\]]*\]\s*')

    def normalize(self, text):
        "Lowercases and normalizes the given string following the pre-compiled pattern stored in self.normalizing_regex."
        return self.normalizing_regex.sub("", text).lower()
    
    def remove_link_junk(self, text):
        """From given text, filters out the part that matches the regex and normalizes the surrounding whitespace."""

        def smart_replace(match:re.Match):
            # Check what comes before and after
            prev_char = text[match.start()-1] if match.start() > 0 else ""
            next_char = text[match.end()] if match.end() < len(text) else ""
            
            # If we're between words (letters or numbers on both sides)
            if (prev_char.isalnum() and next_char.isalnum()):
                return " "  # Keep a single space between words
            # If we're at the end of a sentence or before punctuation
            elif next_char in self.punct_chars or not next_char:
                return ""   # Remove completely
            # If we're after sentence boundary but before a word
            elif (not prev_char.isalnum() or not prev_char) and next_char.isalnum():
                return ""   # Remove to avoid leading space
            else:
                return " "  # Default case - keep a space

        cleaned_text = self.link_junk_pattern.sub(smart_replace, text)
        # Normalize any resulting double spaces, trim leading/trailing whitespace
        return re.sub(r'\s+', ' ', cleaned_text).strip()

def remove_duplicate_lines(paragraphs):
    """Iterates pairwise over a list of text paragraphs, and skips any lines that are repeated.
    Only considers consecutive paragraphs."""
    for i, (a, b) in enumerate(pairwise(paragraphs)):
        if i == 0:
            yield a
        if a == b:
            continue
        yield b

def clean_up(unfiltered_corpus_filename:str,
             unique_url_corpus_filename:str,
             extra_titles_filename:str,
             junk_prefix_filename:str,
             junk_lines_filename:str,
             filtered_corpus_filename:str,
             removed_lines_filename:str):

    # Initialize a PatternMatcher object and pre-compile needed regular expressions
    pattern_matcher = PatternMatcher()

    # Build the automaton for titles
    # Use all titles from unfiltered data and an extrenal file manually constructed (needed for detecting more titles in texts)
    TITLE_TRIE = ahocorasick.Automaton()
    for idx, title in tqdm(enumerate(set(yield_values_from_jsonl_file(unfiltered_corpus_filename, "title"))), desc="Adding titles to Automaton A"):
        if title == "Kevät": # skip this title as we do not want to include it in the words to search
            continue
        title = pattern_matcher.normalize(title) # normalize titles
        TITLE_TRIE.add_word(title, (idx, title))
    for idx, title in tqdm(enumerate(set(yield_values_from_text_file(extra_titles_filename))), desc="Adding titles to Automaton A from external file"):
        title = pattern_matcher.normalize(title) # normalize titles
        TITLE_TRIE.add_word(title, (idx, title))
    TITLE_TRIE.make_automaton()

    # Build the automaton for possible prefixes
    PREFIX_TRIE = ahocorasick.Automaton()
    for idx, prefix in tqdm(enumerate(set(yield_values_from_text_file(junk_prefix_filename))), desc="Adding prefixes to Automaton B"):
        prefix = pattern_matcher.normalize(prefix) # important to keep colons!
        PREFIX_TRIE.add_word(prefix, (idx, prefix))
    PREFIX_TRIE.make_automaton()

    # Build the automaton for manually chosen junk lines
    JUNK_TRIE = ahocorasick.Automaton()
    for idx, junk_line in tqdm(enumerate(set(yield_values_from_text_file(junk_lines_filename))), desc="Adding junk lines to Automaton C"):
        junk_line = pattern_matcher.normalize(junk_line)
        JUNK_TRIE.add_word(junk_line, (idx, junk_line))
    JUNK_TRIE.make_automaton()

    filtering_statistics = FilteringStatistics()
    num_total_lines_in_file = get_line_count(unique_url_corpus_filename)
    filtering_statistics.total_docs = num_total_lines_in_file

    # Check each line in each text and write to file
    with open(filtered_corpus_filename, "w") as filtered_corpus, open(removed_lines_filename, "w") as removed_lines:

        for obj in tqdm(yield_corpus(unique_url_corpus_filename), desc="Searching for matches", total=num_total_lines_in_file):

            # For collecting the filtered text per article
            filtered_text = []

            for line in remove_duplicate_lines(obj["text"].splitlines()): # only iterate over a line if it is not a duplicate of the previous line

                orig_line = line

                if "siirryt toiseen palveluun" in orig_line:
                    # do a more detailed clean up if the above mentioned substring is found in the line
                    cleaned_line = pattern_matcher.remove_link_junk(orig_line)
                else:
                    cleaned_line = line

                normalized_line = pattern_matcher.normalize(cleaned_line) # normalize for looking for matches in prebuilt Automatons
                filtering_statistics.total_lines += 1
                line_marked_for_removal = False

                # Check if line contains alphanumeric characters
                if len(normalized_line) == 0:
                    reason = "non_alphanum_chars_only"
                    line_marked_for_removal = True

                # Check if line is a fullmatch for a title
                elif normalized_line in TITLE_TRIE:
                    reason = "title_full_match"
                    line_marked_for_removal = True
                
                # Check if line is a fullmatch for a junk line
                elif normalized_line in JUNK_TRIE:
                    reason = "junk_line_full_match"
                    line_marked_for_removal = True

                # Check if a line starts with a prefix line
                else:    
                    for end_index, (idx, prefix) in PREFIX_TRIE.iter(normalized_line):
                        start_index = end_index - len(prefix) + 1
                        if start_index == 0:
                            reason = "line_starts_with_junk_prefix"
                            line_marked_for_removal = True
                            break
                        # If the first match is not found in the beginning of the string, do not proceed
                        else:
                            break
                
                if not line_marked_for_removal:
                    for end_index, (idx, title) in TITLE_TRIE.iter(normalized_line):

                        title_pattern = rf":{re.escape(title)}($|\b)" # prefix title with a colon, assure that there is no additional text after the title

                        # Check if a line that contains a title also contains a date
                        if re.search(pattern_matcher.date_pattern, normalized_line):
                            reason = "title_with_date"
                            line_marked_for_removal = True
                            break
                        
                        # Check if a line that contains a title has a colon before the title
                        elif re.search(title_pattern, normalized_line):
                            reason = "title_after_colon"
                            line_marked_for_removal = True
                            break

                        else:
                            # Check if line contains more than one title by removing the already matched title
                            for _, (_, another_title) in TITLE_TRIE.iter(normalized_line.replace(title, "", 1)):
                                reason = "multiple_titles"
                                line_marked_for_removal = True
                                break
                
                # Record the filtering reason and write the line with the reason identifier into a file
                if line_marked_for_removal:
                    filtering_statistics.reasons["line_filter"][reason] += 1
                    print(orig_line, FilteringStatistics.format_identifier(reason), file=removed_lines)

                    # Count sentences for statistics
                    filtering_statistics.total_sentences += len(sent_tokenize(cleaned_line))

                    # Continue to the next line
                    continue

                # If the line has not been filtered out, do filtering sentence by sentence
                # Note! If the original line has included the substring 'siirryt toiseen palveluun', it will not be shown on the sentences written to the removed lines file
                if not line_marked_for_removal:

                    filtering_statistics.sent_for_sentence_splitting += 1
                    filtered_sentences = []

                    for cleaned_sentence in sent_tokenize(cleaned_line, language="finnish"):
                        
                        sentence_marked_for_removal = False
                        normalized_sentence = pattern_matcher.normalize(cleaned_sentence)

                        filtering_statistics.seen_sentences += 1
                        filtering_statistics.total_sentences += 1

                        if normalized_sentence in JUNK_TRIE:
                            reason = "junk_sentence"
                            sentence_marked_for_removal = True

                        elif normalized_sentence in TITLE_TRIE:
                            reason = "title_full_match_sentence"
                            sentence_marked_for_removal = True

                        else:
                            for end_index, (idx, prefix) in PREFIX_TRIE.iter(normalized_sentence):
                                start_index = end_index - len(prefix) + 1
                                if start_index == 0:
                                    reason = "sentence_starts_with_junk_prefix"
                                    sentence_marked_for_removal = True
                                    break
                                # If the first match is not found in the beginning of the string, do not proceed
                                else:
                                    break
                        
                        # Record the filtering reason and write the sentence with the reason identifier into a file
                        if sentence_marked_for_removal:
                            filtering_statistics.reasons["sentence_filter"][reason] += 1
                            print(cleaned_sentence, FilteringStatistics.format_identifier(reason), file=removed_lines)
                            continue

                        else:
                            filtered_sentences.append(cleaned_sentence)

                    # Keep text that is not considered junk
                    filtered_text.append(" ".join(filtered_sentences)) # This could be problematic if sentence identification has somehow failed, adding unwanted whitespace
            
            if len(filtered_text) == 0:
                filtering_statistics.empty += 1
                continue

            if not line_marked_for_removal and not sentence_marked_for_removal:
                filtering_statistics.unmodified += 1
            
            else:
                filtering_statistics.modified += 1

            new_obj = {
                "title": obj["title"],
                "tags": obj["tags"],
                "summary": obj["summary"],
                "text_beginning": filtered_text.pop(0).strip(),
                "text_end": "\n".join(filtered_text).strip(),
                "url": obj["url"],
                "timestamp": obj["timestamp"]
            }

            print(json.dumps(new_obj, ensure_ascii=False), file=filtered_corpus)

    print(filtering_statistics)