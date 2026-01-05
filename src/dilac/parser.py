"""
DiLAC Dictionary Parser
========================

Parses the raw text of the Dictionary of Contemporary Arabic
and converts it to a structured format compliant with LMF.
"""

import re
import json
from typing import List, Dict, Optional, Tuple, Generator
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from .lmf_schema import (
    LexicalEntry, LexicalResource, Sense, Example,
    MorphologicalFeature, PartOfSpeech, Pronunciation
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParsedEntry:
    """Intermediate representation of a parsed entry"""
    root: str
    lemma: str
    pos: str
    morphological_info: Dict
    senses: List[Dict]
    examples: List[Dict]
    contextual_expressions: List[str]


class DictionaryParser:
    """
    Parser for the Dictionary of Contemporary Arabic text file.
    
    The dictionary follows a specific format:
    - Root letters separated by spaces (e.g., "ب ر ع")
    - Entry headers with morphological information
    - Definitions marked with numbers
    - Examples following definitions
    - Domain indicators in parentheses
    """
    
    # Regular expressions for parsing
    ROOT_PATTERN = re.compile(r'^([أ-ي])\s+([أ-ي])\s+([أ-ي])(?:\s+([أ-ي]))?$', re.MULTILINE)
    ENTRY_PATTERN = re.compile(
        r'^([أ-ي]+)\s*\[([^\]]+)\]:\s*(.+?)(?=^[أ-ي]+\s*\[|^[أ-ي]\s+[أ-ي]\s+[أ-ي]|\Z)',
        re.MULTILINE | re.DOTALL
    )
    VERB_ENTRY_PATTERN = re.compile(
        r'^([أ-ي]+)\s+([يتن][أ-ي]+),?\s*([أ-ي]+)?',
        re.MULTILINE
    )
    DEFINITION_PATTERN = re.compile(r'[١٢٣٤٥٦٧٨٩٠0-9]+([^١٢٣٤٥٦٧٨٩٠0-9]+)')
    DOMAIN_PATTERN = re.compile(r'\(([^)]+)\)')
    EXAMPLE_PATTERN = re.compile(r'"([^"]+)"')
    PLURAL_PATTERN = re.compile(r'ج\s+([أ-ي]+)')
    FEMININE_PATTERN = re.compile(r'مؤ\s+([أ-ي]+)')
    
    # POS mappings
    POS_MAPPINGS = {
        'مفرد': PartOfSpeech.NOUN,
        'اسم': PartOfSpeech.NOUN,
        'فعل': PartOfSpeech.VERB,
        'صفة': PartOfSpeech.ADJECTIVE,
        'حرف': PartOfSpeech.PARTICLE,
        'ظرف': PartOfSpeech.ADVERB,
        'ضمير': PartOfSpeech.PRONOUN,
        'جمع': PartOfSpeech.NOUN,
    }
    
    def __init__(self, encoding: str = 'utf-8'):
        self.encoding = encoding
        self.current_root = ""
        self.entries: List[LexicalEntry] = []
        self.entry_counter = 0
        self.sense_counter = 0
        self.example_counter = 0
    
    def parse_file(self, filepath: str) -> LexicalResource:
        """Parse the dictionary file and return a LexicalResource"""
        logger.info(f"Parsing dictionary file: {filepath}")
        
        with open(filepath, 'r', encoding=self.encoding) as f:
            content = f.read()
        
        # Clean content
        content = self._preprocess_content(content)
        
        # Parse into blocks
        blocks = self._split_into_blocks(content)
        logger.info(f"Found {len(blocks)} entry blocks")
        
        # Parse each block
        for block in blocks:
            try:
                entry = self._parse_block(block)
                if entry:
                    self.entries.append(entry)
            except Exception as e:
                logger.warning(f"Error parsing block: {e}")
        
        logger.info(f"Successfully parsed {len(self.entries)} entries")
        
        return LexicalResource(
            name="DiLAC",
            language="ar",
            version="1.0",
            entries=self.entries
        )
    
    def _preprocess_content(self, content: str) -> str:
        """Clean and normalize the dictionary content"""
        # Normalize Arabic characters
        content = self._normalize_arabic(content)
        
        # Remove page numbers and headers
        content = re.sub(r'\n\d+\n', '\n', content)
        
        # Normalize whitespace
        content = re.sub(r'\r\n', '\n', content)
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content
    
    def _normalize_arabic(self, text: str) -> str:
        """Normalize Arabic text"""
        # Normalize alef variants
        text = re.sub(r'[إأآا]', 'ا', text)
        
        # Normalize teh marbuta
        text = re.sub(r'ة', 'ه', text)
        
        # Remove tatweel
        text = text.replace('ـ', '')
        
        # Remove diacritics (optional, can be preserved)
        # diacritics = re.compile(r'[\u064B-\u0652\u0670]')
        # text = diacritics.sub('', text)
        
        return text
    
    def _split_into_blocks(self, content: str) -> List[str]:
        """Split content into entry blocks"""
        blocks = []
        current_block = []
        current_root = None
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_block:
                    blocks.append('\n'.join(current_block))
                    current_block = []
                continue
            
            # Check for root pattern
            root_match = self.ROOT_PATTERN.match(line)
            if root_match:
                if current_block:
                    blocks.append('\n'.join(current_block))
                current_root = line
                current_block = [line]
            else:
                current_block.append(line)
        
        if current_block:
            blocks.append('\n'.join(current_block))
        
        return blocks
    
    def _parse_block(self, block: str) -> Optional[LexicalEntry]:
        """Parse a single entry block"""
        lines = block.split('\n')
        if not lines:
            return None
        
        # Extract root if present
        root_match = self.ROOT_PATTERN.match(lines[0])
        if root_match:
            self.current_root = ' '.join(filter(None, root_match.groups()))
            lines = lines[1:]
        
        if not lines:
            return None
        
        # Parse entry header
        header_info = self._parse_header(lines[0])
        if not header_info:
            return None
        
        lemma, pos, morph_info = header_info
        
        # Parse senses and examples from remaining content
        content = '\n'.join(lines[1:]) if len(lines) > 1 else lines[0]
        senses = self._parse_senses(content)
        
        # Create entry
        self.entry_counter += 1
        entry = LexicalEntry(
            id=f"entry_{self.entry_counter:06d}",
            lemma=lemma,
            pos=pos,
            morphology=MorphologicalFeature(
                root=self.current_root,
                **morph_info
            ),
            senses=senses
        )
        
        return entry
    
    def _parse_header(self, header: str) -> Optional[Tuple[str, PartOfSpeech, Dict]]:
        """Parse entry header to extract lemma, POS, and morphological info"""
        # Try to match [info] pattern
        bracket_match = re.search(r'^([^\[]+)\[([^\]]+)\]', header)
        
        if bracket_match:
            lemma = bracket_match.group(1).strip()
            info = bracket_match.group(2)
            
            # Determine POS
            pos = PartOfSpeech.NOUN
            for key, value in self.POS_MAPPINGS.items():
                if key in info:
                    pos = value
                    break
            
            # Extract morphological features
            morph_info = {}
            
            # Plural
            plural_match = self.PLURAL_PATTERN.search(info)
            if plural_match:
                morph_info['plural_forms'] = [plural_match.group(1)]
            
            # Feminine
            fem_match = self.FEMININE_PATTERN.search(info)
            if fem_match:
                morph_info['feminine_form'] = fem_match.group(1)
            
            return lemma, pos, morph_info
        
        # Try verb pattern
        verb_match = self.VERB_ENTRY_PATTERN.match(header)
        if verb_match:
            lemma = verb_match.group(1)
            return lemma, PartOfSpeech.VERB, {}
        
        # Simple pattern - just take first word
        words = header.split()
        if words:
            return words[0], PartOfSpeech.NOUN, {}
        
        return None
    
    def _parse_senses(self, content: str) -> List[Sense]:
        """Parse senses from entry content"""
        senses = []
        
        # Split by numbered definitions
        # Arabic numerals: ١٢٣٤٥٦٧٨٩
        parts = re.split(r'[١٢٣٤٥٦٧٨٩1-9]', content)
        
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            
            self.sense_counter += 1
            
            # Extract domain if present
            domain = None
            domain_match = self.DOMAIN_PATTERN.search(part)
            if domain_match:
                domain = domain_match.group(1)
            
            # Extract examples
            examples = []
            for ex_match in self.EXAMPLE_PATTERN.finditer(part):
                self.example_counter += 1
                examples.append(Example(
                    id=f"ex_{self.example_counter:06d}",
                    text=ex_match.group(1)
                ))
            
            # Clean definition (remove domain and examples)
            definition = part
            definition = self.DOMAIN_PATTERN.sub('', definition)
            definition = self.EXAMPLE_PATTERN.sub('', definition)
            definition = re.sub(r'[:\.]$', '', definition.strip())
            
            if definition:
                sense = Sense(
                    id=f"sense_{self.sense_counter:06d}",
                    definition=definition,
                    domain=domain,
                    examples=examples
                )
                senses.append(sense)
        
        # If no numbered senses, treat whole content as one sense
        if not senses and content.strip():
            self.sense_counter += 1
            
            domain = None
            domain_match = self.DOMAIN_PATTERN.search(content)
            if domain_match:
                domain = domain_match.group(1)
            
            definition = content
            definition = self.DOMAIN_PATTERN.sub('', definition)
            definition = self.EXAMPLE_PATTERN.sub('', definition)
            definition = definition.strip()
            
            sense = Sense(
                id=f"sense_{self.sense_counter:06d}",
                definition=definition,
                domain=domain,
                examples=[]
            )
            senses.append(sense)
        
        return senses
    
    def parse_to_json(self, input_path: str, output_path: str):
        """Parse dictionary and save as JSON"""
        resource = self.parse_file(input_path)
        
        # Convert to serializable format
        data = {
            'name': resource.name,
            'language': resource.language,
            'version': resource.version,
            'statistics': {
                'total_entries': len(resource.entries),
                'total_senses': sum(len(e.senses) for e in resource.entries),
                'total_examples': sum(
                    len(s.examples) 
                    for e in resource.entries 
                    for s in e.senses
                )
            },
            'entries': []
        }
        
        for entry in resource.entries:
            entry_data = {
                'id': entry.id,
                'lemma': entry.lemma,
                'pos': entry.pos.value,
                'root': entry.morphology.root,
                'senses': [
                    {
                        'id': s.id,
                        'definition': s.definition,
                        'domain': s.domain,
                        'examples': [
                            {'id': ex.id, 'text': ex.text}
                            for ex in s.examples
                        ]
                    }
                    for s in entry.senses
                ]
            }
            data['entries'].append(entry_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved JSON to {output_path}")


class DiLACLeskPreprocessor:
    """
    Preprocessor to create DiLAC-Lesk format for similarity measures.
    
    Converts definitions to integer codes for faster comparison.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.word_frequencies: Dict[str, int] = {}
        self.next_id = 1
    
    def build_vocabulary(self, resource: LexicalResource):
        """Build vocabulary from all definitions and examples"""
        logger.info("Building vocabulary...")
        
        for entry in resource.entries:
            for sense in entry.senses:
                # Process definition
                words = self._tokenize(sense.definition)
                for word in words:
                    self._add_word(word)
                
                # Process examples
                for example in sense.examples:
                    words = self._tokenize(example.text)
                    for word in words:
                        self._add_word(word)
        
        logger.info(f"Vocabulary size: {len(self.word_to_id)}")
    
    def _add_word(self, word: str):
        """Add a word to the vocabulary"""
        if word not in self.word_to_id:
            self.word_to_id[word] = self.next_id
            self.id_to_word[self.next_id] = word
            self.word_frequencies[word] = 0
            self.next_id += 1
        self.word_frequencies[word] += 1
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and normalize Arabic text"""
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Normalize Arabic
        text = re.sub(r'[إأآا]', 'ا', text)
        text = text.replace('ة', 'ه')
        
        # Split and filter
        words = text.split()
        words = [w for w in words if len(w) > 1]
        
        return words
    
    def encode_sense(self, sense: Sense) -> List[int]:
        """Convert sense to integer codes"""
        words = set()
        
        # Add definition words
        words.update(self._tokenize(sense.definition))
        
        # Add example words
        for example in sense.examples:
            words.update(self._tokenize(example.text))
        
        # Convert to IDs and sort
        ids = [self.word_to_id.get(w, 0) for w in words if w in self.word_to_id]
        return sorted(ids)
    
    def calculate_idf(self, word: str, total_definitions: int) -> float:
        """Calculate IDF weight for a word"""
        import math
        freq = self.word_frequencies.get(word, 1)
        return math.log(total_definitions / freq)
    
    def export_lesk_format(self, resource: LexicalResource, output_path: str):
        """Export in DiLAC-Lesk format"""
        self.build_vocabulary(resource)
        
        lesk_data = {
            'vocabulary_size': len(self.word_to_id),
            'word_to_id': self.word_to_id,
            'id_to_word': self.id_to_word,
            'word_frequencies': self.word_frequencies,
            'entries': []
        }
        
        for entry in resource.entries:
            entry_data = {
                'id': entry.id,
                'lemma': entry.lemma,
                'senses': []
            }
            
            for sense in entry.senses:
                sense_data = {
                    'id': sense.id,
                    'encoded_gloss': self.encode_sense(sense),
                    'domain': sense.domain
                }
                entry_data['senses'].append(sense_data)
            
            lesk_data['entries'].append(entry_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(lesk_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Exported Lesk format to {output_path}")


if __name__ == "__main__":
    # Example usage
    parser = DictionaryParser()
    
    # Parse and save
    resource = parser.parse_file("data/raw/dictionary_full.txt")
    resource.save("data/processed/dilac_lmf.xml")
    
    # Also save as JSON
    parser.parse_to_json(
        "data/raw/dictionary_full.txt",
        "data/processed/dilac.json"
    )
    
    # Create Lesk format
    preprocessor = DiLACLeskPreprocessor()
    preprocessor.export_lesk_format(resource, "data/processed/dilac_lesk.json")
