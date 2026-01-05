"""
DiLAC Database Handler
=======================

Unified interface for accessing the DiLAC lexical database.
Supports both XML (LMF) and JSON formats.
"""

import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Iterator
from pathlib import Path
import logging

from .lmf_schema import (
    LexicalResource, LexicalEntry, Sense, Example,
    MorphologicalFeature, PartOfSpeech
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiLACDatabase:
    """
    Main database handler for DiLAC lexical resource.
    
    Provides efficient access to dictionary entries, senses,
    and supports various query operations.
    """
    
    def __init__(self, filepath: Optional[str] = None):
        """
        Initialize database.
        
        Args:
            filepath: Path to LMF XML or JSON file
        """
        self.entries: Dict[str, LexicalEntry] = {}
        self.lemma_index: Dict[str, List[str]] = {}  # lemma -> entry IDs
        self.root_index: Dict[str, List[str]] = {}   # root -> entry IDs
        self.domain_index: Dict[str, List[str]] = {} # domain -> sense IDs
        
        self._stats = {
            'total_entries': 0,
            'total_senses': 0,
            'total_examples': 0,
            'verbs': 0,
            'nouns': 0,
            'particles': 0
        }
        
        if filepath:
            self.load(filepath)
    
    def load(self, filepath: str):
        """Load database from file"""
        path = Path(filepath)
        
        if path.suffix == '.xml':
            self._load_xml(filepath)
        elif path.suffix == '.json':
            self._load_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        self._build_indices()
        self._compute_stats()
        
        logger.info(f"Loaded database: {self._stats}")
    
    def _load_xml(self, filepath: str):
        """Load from LMF XML"""
        resource = LexicalResource.load(filepath)
        
        for entry in resource.entries:
            self.entries[entry.id] = entry
    
    def _load_json(self, filepath: str):
        """Load from JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for entry_data in data.get('entries', []):
            entry = self._parse_json_entry(entry_data)
            self.entries[entry.id] = entry
    
    def _parse_json_entry(self, data: Dict) -> LexicalEntry:
        """Parse entry from JSON format"""
        # Parse senses
        senses = []
        for sense_data in data.get('senses', []):
            examples = [
                Example(id=ex['id'], text=ex['text'])
                for ex in sense_data.get('examples', [])
            ]
            
            sense = Sense(
                id=sense_data['id'],
                definition=sense_data.get('definition', ''),
                domain=sense_data.get('domain'),
                examples=examples
            )
            senses.append(sense)
        
        # Parse POS
        pos_str = data.get('pos', 'noun')
        try:
            pos = PartOfSpeech(pos_str)
        except ValueError:
            pos = PartOfSpeech.NOUN
        
        # Create entry
        entry = LexicalEntry(
            id=data['id'],
            lemma=data['lemma'],
            pos=pos,
            morphology=MorphologicalFeature(
                root=data.get('root')
            ),
            senses=senses
        )
        
        return entry
    
    def _build_indices(self):
        """Build search indices"""
        self.lemma_index.clear()
        self.root_index.clear()
        self.domain_index.clear()
        
        for entry_id, entry in self.entries.items():
            # Lemma index
            lemma = entry.lemma
            if lemma not in self.lemma_index:
                self.lemma_index[lemma] = []
            self.lemma_index[lemma].append(entry_id)
            
            # Root index
            root = entry.morphology.root if entry.morphology else None
            if root:
                if root not in self.root_index:
                    self.root_index[root] = []
                self.root_index[root].append(entry_id)
            
            # Domain index
            for sense in entry.senses:
                if sense.domain:
                    if sense.domain not in self.domain_index:
                        self.domain_index[sense.domain] = []
                    self.domain_index[sense.domain].append(sense.id)
    
    def _compute_stats(self):
        """Compute database statistics"""
        self._stats = {
            'total_entries': len(self.entries),
            'total_senses': sum(len(e.senses) for e in self.entries.values()),
            'total_examples': sum(
                len(s.examples)
                for e in self.entries.values()
                for s in e.senses
            ),
            'verbs': sum(1 for e in self.entries.values() if e.pos == PartOfSpeech.VERB),
            'nouns': sum(1 for e in self.entries.values() if e.pos == PartOfSpeech.NOUN),
            'particles': sum(1 for e in self.entries.values() if e.pos == PartOfSpeech.PARTICLE),
            'unique_roots': len(self.root_index),
            'domains': list(self.domain_index.keys())
        }
    
    @property
    def stats(self) -> Dict:
        """Get database statistics"""
        return self._stats.copy()
    
    def get_entry(self, entry_id: str) -> Optional[LexicalEntry]:
        """Get entry by ID"""
        return self.entries.get(entry_id)
    
    def search(self, lemma: str) -> List[LexicalEntry]:
        """Search entries by lemma"""
        entry_ids = self.lemma_index.get(lemma, [])
        return [self.entries[eid] for eid in entry_ids]
    
    def search_by_root(self, root: str) -> List[LexicalEntry]:
        """Search entries by Arabic root"""
        entry_ids = self.root_index.get(root, [])
        return [self.entries[eid] for eid in entry_ids]
    
    def search_by_domain(self, domain: str) -> List[LexicalEntry]:
        """Search entries by domain"""
        sense_ids = self.domain_index.get(domain, [])
        
        # Find entries containing these senses
        entries = []
        for entry in self.entries.values():
            for sense in entry.senses:
                if sense.id in sense_ids:
                    entries.append(entry)
                    break
        
        return entries
    
    def get_senses(self, lemma: str) -> List[Sense]:
        """Get all senses for a lemma"""
        entries = self.search(lemma)
        senses = []
        for entry in entries:
            senses.extend(entry.senses)
        return senses
    
    def get_definitions(self, lemma: str) -> List[str]:
        """Get all definitions for a lemma"""
        senses = self.get_senses(lemma)
        return [s.definition for s in senses]
    
    def get_examples(self, lemma: str) -> List[str]:
        """Get all examples for a lemma"""
        senses = self.get_senses(lemma)
        examples = []
        for sense in senses:
            examples.extend([ex.text for ex in sense.examples])
        return examples
    
    def iter_entries(self) -> Iterator[LexicalEntry]:
        """Iterate over all entries"""
        return iter(self.entries.values())
    
    def iter_senses(self) -> Iterator[Sense]:
        """Iterate over all senses"""
        for entry in self.entries.values():
            for sense in entry.senses:
                yield sense
    
    def save(self, filepath: str, format: str = 'xml'):
        """Save database to file"""
        if format == 'xml':
            resource = LexicalResource(
                entries=list(self.entries.values())
            )
            resource.save(filepath)
        elif format == 'json':
            self._save_json(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_json(self, filepath: str):
        """Save to JSON format"""
        data = {
            'name': 'DiLAC',
            'version': '1.0',
            'statistics': self._stats,
            'entries': []
        }
        
        for entry in self.entries.values():
            entry_data = {
                'id': entry.id,
                'lemma': entry.lemma,
                'pos': entry.pos.value,
                'root': entry.morphology.root if entry.morphology else None,
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
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class DiLACLeskDatabase(DiLACDatabase):
    """
    Extended database with Lesk algorithm support.
    
    Maintains encoded glosses for efficient similarity computation.
    """
    
    def __init__(self, filepath: Optional[str] = None):
        super().__init__(filepath)
        
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.word_frequencies: Dict[str, int] = {}
        self.encoded_senses: Dict[str, List[int]] = {}  # sense_id -> word IDs
    
    def load_lesk_format(self, filepath: str):
        """Load Lesk-format database"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.word_to_id = data.get('word_to_id', {})
        self.id_to_word = {int(k): v for k, v in data.get('id_to_word', {}).items()}
        self.word_frequencies = data.get('word_frequencies', {})
        
        # Process entries
        for entry_data in data.get('entries', []):
            lemma = entry_data['lemma']
            
            # Store encoded senses
            for sense_data in entry_data.get('senses', []):
                sense_id = sense_data['id']
                self.encoded_senses[sense_id] = sense_data.get('encoded_gloss', [])
            
            # Create basic entry
            entry = LexicalEntry(
                id=entry_data['id'],
                lemma=lemma,
                pos=PartOfSpeech.NOUN,
                morphology=MorphologicalFeature(),
                senses=[
                    Sense(
                        id=sd['id'],
                        definition='',
                        domain=sd.get('domain')
                    )
                    for sd in entry_data.get('senses', [])
                ]
            )
            self.entries[entry.id] = entry
            
            # Update lemma index
            if lemma not in self.lemma_index:
                self.lemma_index[lemma] = []
            self.lemma_index[lemma].append(entry.id)
    
    def get_encoded_gloss(self, lemma: str) -> List[int]:
        """Get encoded gloss words for a lemma"""
        entry_ids = self.lemma_index.get(lemma, [])
        
        all_words = set()
        for entry_id in entry_ids:
            entry = self.entries.get(entry_id)
            if entry:
                for sense in entry.senses:
                    encoded = self.encoded_senses.get(sense.id, [])
                    all_words.update(encoded)
        
        return list(all_words)


if __name__ == "__main__":
    # Example usage
    db = DiLACDatabase()
    
    # Load from JSON
    db.load("data/processed/dilac.json")
    
    # Search
    entries = db.search("كتاب")
    for entry in entries:
        print(f"Entry: {entry.lemma}")
        for sense in entry.senses:
            print(f"  - {sense.definition}")
    
    # Statistics
    print(f"\nDatabase stats: {db.stats}")
