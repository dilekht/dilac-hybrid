"""
DiLAC LMF Schema Definition
============================

This module defines the LMF (Lexical Markup Framework) compliant schema
for the Dictionary of Contemporary Arabic (DiLAC).

LMF is the ISO 24613 standard for representing lexical resources.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
import xml.etree.ElementTree as ET
from xml.dom import minidom


class PartOfSpeech(Enum):
    """Part of speech categories following Arabic grammar"""
    VERB = "verb"           # فعل
    NOUN = "noun"           # اسم
    PARTICLE = "particle"   # حرف
    ADJECTIVE = "adjective" # صفة
    ADVERB = "adverb"       # ظرف
    PRONOUN = "pronoun"     # ضمير
    PREPOSITION = "preposition"  # حرف جر


class VerbPattern(Enum):
    """Arabic verb patterns (أوزان)"""
    PATTERN_I = "فَعَلَ"
    PATTERN_II = "فَعَّلَ"
    PATTERN_III = "فَاعَلَ"
    PATTERN_IV = "أَفْعَلَ"
    PATTERN_V = "تَفَعَّلَ"
    PATTERN_VI = "تَفَاعَلَ"
    PATTERN_VII = "اِنْفَعَلَ"
    PATTERN_VIII = "اِفْتَعَلَ"
    PATTERN_IX = "اِفْعَلَّ"
    PATTERN_X = "اِسْتَفْعَلَ"


@dataclass
class Pronunciation:
    """Phonetic representation of a word"""
    ipa: Optional[str] = None
    diacritics: Optional[str] = None  # التشكيل


@dataclass
class MorphologicalFeature:
    """Morphological features for Arabic words"""
    root: Optional[str] = None          # الجذر (e.g., ك ت ب)
    pattern: Optional[str] = None       # الوزن
    gender: Optional[str] = None        # الجنس (مذكر/مؤنث)
    number: Optional[str] = None        # العدد (مفرد/مثنى/جمع)
    plural_forms: List[str] = field(default_factory=list)  # صيغ الجمع
    dual_form: Optional[str] = None     # المثنى
    feminine_form: Optional[str] = None # المؤنث
    verb_transitivity: Optional[str] = None  # لازم/متعدي
    active_participle: Optional[str] = None  # اسم الفاعل
    passive_participle: Optional[str] = None # اسم المفعول
    verbal_noun: Optional[str] = None   # المصدر


@dataclass
class Example:
    """Example usage of a word sense"""
    id: str
    text: str                           # نص المثال
    source: Optional[str] = None        # المصدر (قرآن، حديث، شعر، etc.)
    translation: Optional[str] = None
    domain: Optional[str] = None        # المجال


@dataclass
class SemanticRelation:
    """Semantic relations between senses"""
    relation_type: str      # synonym, antonym, hypernym, hyponym, etc.
    target_id: str          # ID of related sense
    target_lemma: str       # Lemma of related word


@dataclass
class Sense:
    """A single sense/meaning of a word"""
    id: str
    definition: str                     # التعريف
    domain: Optional[str] = None        # المجال (رياضة، طب، etc.)
    register: Optional[str] = None      # المستوى (فصيح، عامي، etc.)
    examples: List[Example] = field(default_factory=list)
    semantic_relations: List[SemanticRelation] = field(default_factory=list)
    contextual_expressions: List[str] = field(default_factory=list)  # تعبيرات سياقية
    frequency_rank: Optional[int] = None  # ترتيب الشيوع
    
    def get_gloss_words(self) -> List[str]:
        """Extract words from definition and examples for Lesk algorithm"""
        words = []
        # Add definition words
        words.extend(self.definition.split())
        # Add example words
        for ex in self.examples:
            words.extend(ex.text.split())
        return words


@dataclass
class LexicalEntry:
    """A complete lexical entry in DiLAC"""
    id: str
    lemma: str                          # المدخل
    pos: PartOfSpeech
    morphology: MorphologicalFeature
    pronunciation: Optional[Pronunciation] = None
    senses: List[Sense] = field(default_factory=list)
    
    def to_xml(self) -> ET.Element:
        """Convert entry to LMF-compliant XML"""
        entry = ET.Element("LexicalEntry")
        entry.set("id", self.id)
        
        # Lemma
        lemma_elem = ET.SubElement(entry, "Lemma")
        feat_lemma = ET.SubElement(lemma_elem, "feat")
        feat_lemma.set("att", "writtenForm")
        feat_lemma.set("val", self.lemma)
        
        # Part of Speech
        feat_pos = ET.SubElement(entry, "feat")
        feat_pos.set("att", "partOfSpeech")
        feat_pos.set("val", self.pos.value)
        
        # Morphological Features
        morph = ET.SubElement(entry, "MorphologicalPattern")
        if self.morphology.root:
            feat_root = ET.SubElement(morph, "feat")
            feat_root.set("att", "root")
            feat_root.set("val", self.morphology.root)
        if self.morphology.pattern:
            feat_pattern = ET.SubElement(morph, "feat")
            feat_pattern.set("att", "pattern")
            feat_pattern.set("val", self.morphology.pattern)
        if self.morphology.plural_forms:
            for plural in self.morphology.plural_forms:
                feat_plural = ET.SubElement(morph, "feat")
                feat_plural.set("att", "plural")
                feat_plural.set("val", plural)
        
        # Senses
        for sense in self.senses:
            sense_elem = ET.SubElement(entry, "Sense")
            sense_elem.set("id", sense.id)
            
            # Definition
            def_elem = ET.SubElement(sense_elem, "Definition")
            def_feat = ET.SubElement(def_elem, "feat")
            def_feat.set("att", "text")
            def_feat.set("val", sense.definition)
            
            # Domain
            if sense.domain:
                domain_feat = ET.SubElement(sense_elem, "feat")
                domain_feat.set("att", "domain")
                domain_feat.set("val", sense.domain)
            
            # Examples
            for example in sense.examples:
                ex_elem = ET.SubElement(sense_elem, "Context")
                ex_elem.set("id", example.id)
                ex_feat = ET.SubElement(ex_elem, "feat")
                ex_feat.set("att", "text")
                ex_feat.set("val", example.text)
                if example.source:
                    src_feat = ET.SubElement(ex_elem, "feat")
                    src_feat.set("att", "source")
                    src_feat.set("val", example.source)
            
            # Contextual Expressions
            for expr in sense.contextual_expressions:
                expr_elem = ET.SubElement(sense_elem, "ContextualExpression")
                expr_feat = ET.SubElement(expr_elem, "feat")
                expr_feat.set("att", "text")
                expr_feat.set("val", expr)
            
            # Semantic Relations
            for rel in sense.semantic_relations:
                rel_elem = ET.SubElement(sense_elem, "SenseRelation")
                rel_elem.set("targets", rel.target_id)
                rel_feat = ET.SubElement(rel_elem, "feat")
                rel_feat.set("att", "type")
                rel_feat.set("val", rel.relation_type)
        
        return entry


@dataclass
class LexicalResource:
    """Complete DiLAC lexical resource"""
    name: str = "DiLAC"
    language: str = "ar"
    version: str = "1.0"
    entries: List[LexicalEntry] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    
    def to_xml(self) -> str:
        """Export entire resource to LMF XML"""
        root = ET.Element("LexicalResource")
        root.set("dtdVersion", "16")
        
        # Global Information
        global_info = ET.SubElement(root, "GlobalInformation")
        feat_label = ET.SubElement(global_info, "feat")
        feat_label.set("att", "label")
        feat_label.set("val", self.name)
        feat_lang = ET.SubElement(global_info, "feat")
        feat_lang.set("att", "languageCoding")
        feat_lang.set("val", "ISO 639-3")
        
        # Lexicon
        lexicon = ET.SubElement(root, "Lexicon")
        feat_lex_lang = ET.SubElement(lexicon, "feat")
        feat_lex_lang.set("att", "language")
        feat_lex_lang.set("val", self.language)
        
        # Add all entries
        for entry in self.entries:
            lexicon.append(entry.to_xml())
        
        # Pretty print
        xml_str = ET.tostring(root, encoding='unicode')
        dom = minidom.parseString(xml_str)
        return dom.toprettyxml(indent="  ")
    
    def save(self, filepath: str):
        """Save resource to XML file"""
        xml_content = self.to_xml()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(xml_content)
    
    @classmethod
    def load(cls, filepath: str) -> 'LexicalResource':
        """Load resource from XML file"""
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        resource = cls()
        
        # Parse global info
        global_info = root.find("GlobalInformation")
        if global_info is not None:
            for feat in global_info.findall("feat"):
                if feat.get("att") == "label":
                    resource.name = feat.get("val")
        
        # Parse lexicon
        lexicon = root.find("Lexicon")
        if lexicon is not None:
            for feat in lexicon.findall("feat"):
                if feat.get("att") == "language":
                    resource.language = feat.get("val")
            
            # Parse entries
            for entry_elem in lexicon.findall("LexicalEntry"):
                entry = cls._parse_entry(entry_elem)
                resource.entries.append(entry)
        
        return resource
    
    @staticmethod
    def _parse_entry(entry_elem: ET.Element) -> LexicalEntry:
        """Parse a single lexical entry from XML"""
        entry_id = entry_elem.get("id")
        
        # Parse lemma
        lemma = ""
        lemma_elem = entry_elem.find("Lemma")
        if lemma_elem is not None:
            for feat in lemma_elem.findall("feat"):
                if feat.get("att") == "writtenForm":
                    lemma = feat.get("val")
        
        # Parse POS
        pos = PartOfSpeech.NOUN  # default
        for feat in entry_elem.findall("feat"):
            if feat.get("att") == "partOfSpeech":
                try:
                    pos = PartOfSpeech(feat.get("val"))
                except ValueError:
                    pos = PartOfSpeech.NOUN
        
        # Parse morphology
        morphology = MorphologicalFeature()
        morph_elem = entry_elem.find("MorphologicalPattern")
        if morph_elem is not None:
            for feat in morph_elem.findall("feat"):
                att = feat.get("att")
                val = feat.get("val")
                if att == "root":
                    morphology.root = val
                elif att == "pattern":
                    morphology.pattern = val
                elif att == "plural":
                    morphology.plural_forms.append(val)
        
        # Parse senses
        senses = []
        for sense_elem in entry_elem.findall("Sense"):
            sense = Sense(
                id=sense_elem.get("id"),
                definition=""
            )
            
            # Definition
            def_elem = sense_elem.find("Definition")
            if def_elem is not None:
                for feat in def_elem.findall("feat"):
                    if feat.get("att") == "text":
                        sense.definition = feat.get("val")
            
            # Domain
            for feat in sense_elem.findall("feat"):
                if feat.get("att") == "domain":
                    sense.domain = feat.get("val")
            
            # Examples
            for ctx_elem in sense_elem.findall("Context"):
                example = Example(
                    id=ctx_elem.get("id"),
                    text=""
                )
                for feat in ctx_elem.findall("feat"):
                    if feat.get("att") == "text":
                        example.text = feat.get("val")
                    elif feat.get("att") == "source":
                        example.source = feat.get("val")
                sense.examples.append(example)
            
            senses.append(sense)
        
        return LexicalEntry(
            id=entry_id,
            lemma=lemma,
            pos=pos,
            morphology=morphology,
            senses=senses
        )


# Domain categories in DiLAC
DILAC_DOMAINS = [
    "رياضة",      # Sports
    "طب",         # Medicine
    "زراعة",      # Agriculture
    "جغرافيا",    # Geography
    "سياسة",      # Politics
    "اقتصاد",     # Economics
    "علوم",       # Sciences
    "فن",         # Arts
    "دين",        # Religion
    "قانون",      # Law
    "تعليم",      # Education
    "تقنية",      # Technology
    "عسكري",      # Military
    "موسيقى",     # Music
    "أدب",        # Literature
    "فلسفة",      # Philosophy
    "نفس",        # Psychology
    "كيمياء",     # Chemistry
    "فيزياء",     # Physics
    "رياضيات",    # Mathematics
    "بيولوجيا",   # Biology
    "هندسة",      # Engineering
]


if __name__ == "__main__":
    # Example usage
    entry = LexicalEntry(
        id="entry_001",
        lemma="كتاب",
        pos=PartOfSpeech.NOUN,
        morphology=MorphologicalFeature(
            root="ك ت ب",
            pattern="فِعال",
            plural_forms=["كُتُب", "كُتْب"]
        ),
        senses=[
            Sense(
                id="sense_001_1",
                definition="مجموعة من الصفحات المكتوبة أو المطبوعة",
                domain="ثقافة",
                examples=[
                    Example(
                        id="ex_001_1_1",
                        text="قرأت كتابًا مفيدًا",
                        source="مثال"
                    )
                ]
            )
        ]
    )
    
    resource = LexicalResource(entries=[entry])
    print(resource.to_xml())
