"""
JointBERT Dataset Preprocessing Pipeline for Campus QA Bot

This script preprocesses the raw dataset for training a JointBERT model
with IndoBERT for intent classification and slot filling (NER).

Steps:
1. Fix annotation syntax issues
2. Expand {{MATKUL}} placeholders with course names
3. Expand {{WAKTU}} placeholders with time values
4. Skip rows with {{NAMA_DOSEN}} placeholders (keep real annotated examples)
5. Export cleaned dataset (CSV)
6. Convert to JointBERT format (seq.in, seq.out, label files)
7. Apply BIO tagging with subword alignment (Strategy A)

Entity types: MATKUL, WAKTU, NAMA_DOSEN
Intents: get_tugas_untuk_mata_kuliah, get_jadwal, get_ruangan, get_info_dosen
"""

import csv
import re
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Try to import transformers, make it optional
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Subword alignment will be disabled.")
    print("Install with: pip install transformers")

# Configuration
RANDOM_SEED = 42
TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
TEST_RATIO = 0.1

# Paths
BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "DATASET NLP - Sheet1 (1).csv"
MATKUL_PATH = BASE_DIR / "MATKUL.txt"
OUTPUT_DIR = BASE_DIR / "data"
CLEANED_CSV_PATH = BASE_DIR / "cleaned_dataset.csv"

# Time values for {{WAKTU}} expansion
WAKTU_VALUES = [
    "hari senin",
    "hari selasa", 
    "hari rabu",
    "hari kamis",
    "hari jumat",
    "hari sabtu",
    "hari minggu",
    "hari ini",
    "besok",
    "lusa",
    "minggu ini",
    "minggu depan",
    "kemarin",
]

# Intent column mapping
INTENT_COLUMNS = [
    "get_tugas_untuk_mata_kuliah",
    "get_jadwal", 
    "get_ruangan",
    "get_info_dosen"
]


def load_matkul_list(path: Path) -> List[str]:
    """Load course names from MATKUL.txt"""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def fix_annotation_syntax(text: str) -> str:
    """
    Fix various annotation syntax issues:
    1. [text] (ENTITY) -> [text](ENTITY)  (space between brackets)
    2. [hari jumat]apa? -> [hari jumat](WAKTU) apa? (missing entity tag)
    3. text (ENTITY) -> [text](ENTITY) (missing brackets)
    4. [WAKTU] or [NAMA_DOSEN] -> {{WAKTU}} or {{NAMA_DOSEN}} (bare tags)
    """
    
    # Fix: [text] (ENTITY) -> [text](ENTITY)
    text = re.sub(r'\]\s+\(', '](', text)
    
    # Fix: bare (WAKTU) without brackets - find time-like words before it
    # Pattern: word (WAKTU) where word is a time expression
    time_patterns = [
        r'(hari\s+ini)\s*\(WAKTU\)',
        r'(hari\s+senin)\s*\(WAKTU\)',
        r'(hari\s+selasa)\s*\(WAKTU\)',
        r'(hari\s+rabu)\s*\(WAKTU\)',
        r'(hari\s+kamis)\s*\(WAKTU\)',
        r'(hari\s+jumat)\s*\(WAKTU\)',
        r'(hari\s+sabtu)\s*\(WAKTU\)',
        r'(hari\s+minggu)\s*\(WAKTU\)',
        r'(minggu\s+ini)\s*\(WAKTU\)',
        r'(minggu\s+depan)\s*\(WAKTU\)',
        r'(besok)\s*\(WAKTU\)',
        r'(lusa)\s*\(WAKTU\)',
        r'(kemarin)\s*\(WAKTU\)',
        r'(sekarang)\s*\(WAKTU\)',
    ]
    for pattern in time_patterns:
        text = re.sub(pattern, r'[\1](WAKTU)', text, flags=re.IGNORECASE)
    
    # Fix: [hari jumat]apa? -> [hari jumat](WAKTU) apa?
    # Pattern: [text] followed by non-( character (missing entity tag)
    def add_waktu_tag(match):
        content = match.group(1)
        after = match.group(2)
        # Check if content looks like a time expression
        time_keywords = ['hari', 'senin', 'selasa', 'rabu', 'kamis', 'jumat', 
                        'sabtu', 'minggu', 'besok', 'lusa', 'kemarin', 'ini', 'depan']
        if any(kw in content.lower() for kw in time_keywords):
            return f'[{content}](WAKTU) {after}'
        return match.group(0)
    
    text = re.sub(r'\[([^\]]+)\]([^(])', add_waktu_tag, text)
    
    # Fix: [WAKTU] or [NAMA_DOSEN] (bare entity type as text)
    text = re.sub(r'\[WAKTU\]', '{{WAKTU}}', text)
    text = re.sub(r'\[NAMA_DOSEN\]', '{{NAMA_DOSEN}}', text)
    
    # Fix: [nama dosen] -> {{NAMA_DOSEN}} (literal placeholder text)
    text = re.sub(r'\[nama\s+dosen\]', '{{NAMA_DOSEN}}', text, flags=re.IGNORECASE)
    
    # Fix: [kode dosen] -> remove (we're removing KODE_DOSEN entity)
    text = re.sub(r'\[kode\s+dosen\]', '', text, flags=re.IGNORECASE)
    
    return text.strip()


def contains_nama_dosen_placeholder(text: str) -> bool:
    """Check if text contains {{NAMA_DOSEN}} placeholder"""
    return '{{NAMA_DOSEN}}' in text


def contains_kode_dosen_reference(text: str) -> bool:
    """Check if text contains [kode dosen] or similar references to remove"""
    patterns = [
        r'\[kode\s+dosen\]',
        r'\{\{KODE_DOSEN\}\}',
        r'kode\s+dosen.*\[kode\]',
    ]
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def expand_matkul_placeholder(text: str, matkul_list: List[str]) -> List[str]:
    """
    Expand {{MATKUL}} placeholder with all course names.
    Returns list of expanded sentences.
    """
    if '{{MATKUL}}' not in text:
        return [text]
    
    expanded = []
    for matkul in matkul_list:
        # Replace {{MATKUL}} with annotated format [course_name](MATKUL)
        new_text = text.replace('{{MATKUL}}', f'[{matkul}](MATKUL)')
        expanded.append(new_text)
    
    return expanded


def expand_waktu_placeholder(text: str, waktu_values: List[str]) -> List[str]:
    """
    Expand {{WAKTU}} placeholder with time values.
    Returns list of expanded sentences.
    """
    if '{{WAKTU}}' not in text:
        return [text]
    
    expanded = []
    for waktu in waktu_values:
        # Replace {{WAKTU}} with annotated format [time_value](WAKTU)
        new_text = text.replace('{{WAKTU}}', f'[{waktu}](WAKTU)')
        expanded.append(new_text)
    
    return expanded


def expand_all_placeholders(text: str, matkul_list: List[str], waktu_values: List[str]) -> List[str]:
    """
    Expand both {{MATKUL}} and {{WAKTU}} placeholders.
    If both exist, creates combinations.
    """
    # First expand MATKUL
    matkul_expanded = expand_matkul_placeholder(text, matkul_list)
    
    # Then expand WAKTU for each MATKUL variant
    all_expanded = []
    for variant in matkul_expanded:
        waktu_expanded = expand_waktu_placeholder(variant, waktu_values)
        all_expanded.extend(waktu_expanded)
    
    return all_expanded


def parse_annotations(text: str) -> Tuple[str, List[Tuple[str, str, int, int]]]:
    """
    Parse [text](ENTITY) annotations from text.
    Returns: (plain_text, list of (entity_text, entity_type, start, end))
    """
    entities = []
    plain_text = ""
    last_end = 0
    
    # Pattern: [text](ENTITY_TYPE)
    pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    
    offset = 0
    for match in re.finditer(pattern, text):
        # Add text before this entity
        plain_text += text[last_end:match.start()]
        
        entity_text = match.group(1)
        entity_type = match.group(2)
        
        # Calculate position in plain text
        start = len(plain_text)
        end = start + len(entity_text)
        
        plain_text += entity_text
        entities.append((entity_text, entity_type, start, end))
        
        last_end = match.end()
    
    # Add remaining text
    plain_text += text[last_end:]
    
    return plain_text, entities


def text_to_bio_tags(plain_text: str, entities: List[Tuple[str, str, int, int]]) -> Tuple[List[str], List[str]]:
    """
    Convert plain text and entities to word-level BIO tags.
    Returns: (tokens, bio_tags)
    """
    # Simple whitespace tokenization first
    words = plain_text.split()
    
    # Build character-to-word mapping
    char_to_word = {}
    char_pos = 0
    for word_idx, word in enumerate(words):
        # Skip leading spaces
        while char_pos < len(plain_text) and plain_text[char_pos] == ' ':
            char_pos += 1
        # Map characters of this word
        for _ in word:
            if char_pos < len(plain_text):
                char_to_word[char_pos] = word_idx
                char_pos += 1
    
    # Initialize all tags as O
    bio_tags = ['O'] * len(words)
    
    # Assign entity tags
    for entity_text, entity_type, start, end in entities:
        # Find which words this entity spans
        entity_words = set()
        for char_idx in range(start, end):
            if char_idx in char_to_word:
                entity_words.add(char_to_word[char_idx])
        
        # Assign B- and I- tags
        sorted_words = sorted(entity_words)
        for i, word_idx in enumerate(sorted_words):
            if word_idx < len(bio_tags):
                if i == 0:
                    bio_tags[word_idx] = f'B-{entity_type}'
                else:
                    bio_tags[word_idx] = f'I-{entity_type}'
    
    return words, bio_tags


def align_subword_tags(tokens: List[str], bio_tags: List[str], tokenizer) -> Tuple[List[str], List[str]]:
    """
    Align BIO tags with subword tokens using Strategy A:
    - First subword gets the B/I tag
    - Continuation subwords (##...) get the I- tag
    
    Returns: (subword_tokens, aligned_tags)
    """
    subword_tokens = []
    aligned_tags = []
    
    for word, tag in zip(tokens, bio_tags):
        # Tokenize the word
        word_tokens = tokenizer.tokenize(word)
        
        if not word_tokens:
            continue
            
        for i, subword in enumerate(word_tokens):
            subword_tokens.append(subword)
            
            if tag == 'O':
                aligned_tags.append('O')
            elif i == 0:
                # First subword gets original tag
                aligned_tags.append(tag)
            else:
                # Continuation subwords get I- tag
                if tag.startswith('B-'):
                    aligned_tags.append('I-' + tag[2:])
                else:
                    aligned_tags.append(tag)
    
    return subword_tokens, aligned_tags


def load_and_clean_dataset(dataset_path: Path, matkul_list: List[str]) -> List[Tuple[str, str]]:
    """
    Load dataset from CSV and return list of (text, intent) tuples.
    Applies cleaning and expansion.
    """
    samples = []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            for intent in INTENT_COLUMNS:
                text = row.get(intent, '').strip()
                
                if not text:
                    continue
                
                # Fix annotation syntax
                text = fix_annotation_syntax(text)
                
                # Skip rows with {{NAMA_DOSEN}} placeholder
                if contains_nama_dosen_placeholder(text):
                    continue
                
                # Skip rows with kode dosen references (we removed this entity)
                if contains_kode_dosen_reference(text):
                    continue
                
                # Expand placeholders
                expanded_texts = expand_all_placeholders(text, matkul_list, WAKTU_VALUES)
                
                for expanded_text in expanded_texts:
                    samples.append((expanded_text, intent))
    
    return samples


def export_cleaned_csv(samples: List[Tuple[str, str]], output_path: Path):
    """Export cleaned and expanded dataset to CSV"""
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'intent'])
        for text, intent in samples:
            writer.writerow([text, intent])
    print(f"Exported cleaned dataset to {output_path}")
    print(f"Total samples: {len(samples)}")


def create_jointbert_files(
    samples: List[Tuple[str, str]], 
    output_dir: Path,
    tokenizer,
    use_subword_alignment: bool = True
):
    """
    Create JointBERT format files:
    - train/dev/test folders with seq.in, seq.out, label
    - intent_label.txt
    - slot_label.txt
    """
    # Shuffle and split
    random.seed(RANDOM_SEED)
    random.shuffle(samples)
    
    n = len(samples)
    train_end = int(n * TRAIN_RATIO)
    dev_end = train_end + int(n * DEV_RATIO)
    
    splits = {
        'train': samples[:train_end],
        'dev': samples[train_end:dev_end],
        'test': samples[dev_end:]
    }
    
    # Collect all slot labels
    all_slot_labels = set(['O', 'PAD', 'UNK'])
    
    # Process each split
    for split_name, split_samples in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        seq_in_lines = []
        seq_out_lines = []
        label_lines = []
        
        for text, intent in split_samples:
            # Parse annotations
            plain_text, entities = parse_annotations(text)
            
            # Convert to BIO tags
            tokens, bio_tags = text_to_bio_tags(plain_text, entities)
            
            if use_subword_alignment:
                # Align with subword tokens
                tokens, bio_tags = align_subword_tags(tokens, bio_tags, tokenizer)
            
            # Collect slot labels
            all_slot_labels.update(bio_tags)
            
            # Write to files
            seq_in_lines.append(' '.join(tokens))
            seq_out_lines.append(' '.join(bio_tags))
            label_lines.append(intent)
        
        # Write files
        (split_dir / 'seq.in').write_text('\n'.join(seq_in_lines), encoding='utf-8')
        (split_dir / 'seq.out').write_text('\n'.join(seq_out_lines), encoding='utf-8')
        (split_dir / 'label').write_text('\n'.join(label_lines), encoding='utf-8')
        
        print(f"{split_name}: {len(split_samples)} samples")
    
    # Write intent labels
    intent_labels = sorted(set(intent for _, intent in samples))
    (output_dir / 'intent_label.txt').write_text('\n'.join(intent_labels), encoding='utf-8')
    
    # Write slot labels (sorted for consistency)
    slot_labels = ['PAD', 'UNK', 'O'] + sorted([l for l in all_slot_labels if l not in ['PAD', 'UNK', 'O']])
    (output_dir / 'slot_label.txt').write_text('\n'.join(slot_labels), encoding='utf-8')
    
    print(f"\nIntent labels: {intent_labels}")
    print(f"Slot labels: {slot_labels}")


def main():
    print("=" * 60)
    print("JointBERT Dataset Preprocessing Pipeline")
    print("=" * 60)
    
    # Load MATKUL list
    print("\n[1/6] Loading MATKUL list...")
    matkul_list = load_matkul_list(MATKUL_PATH)
    print(f"Loaded {len(matkul_list)} course names")
    
    # Load IndoBERT tokenizer (optional)
    print("\n[2/6] Loading IndoBERT tokenizer...")
    tokenizer = None
    use_subword = False
    if TRANSFORMERS_AVAILABLE:
        try:
            tokenizer = AutoTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
            use_subword = True
            print("Tokenizer loaded successfully")
        except Exception as e:
            print(f"Could not load tokenizer: {e}")
            print("Continuing without subword alignment...")
    else:
        print("Transformers not available, skipping subword alignment")
    
    # Load and clean dataset
    print("\n[3/6] Loading and cleaning dataset...")
    samples = load_and_clean_dataset(DATASET_PATH, matkul_list)
    print(f"Generated {len(samples)} samples after expansion")
    
    # Show sample distribution
    intent_counts = {}
    for _, intent in samples:
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    print("\nIntent distribution:")
    for intent, count in sorted(intent_counts.items()):
        print(f"  {intent}: {count}")
    
    # Export cleaned CSV
    print("\n[4/6] Exporting cleaned dataset...")
    export_cleaned_csv(samples, CLEANED_CSV_PATH)
    
    # Create output directory
    print("\n[5/6] Creating JointBERT format files...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create JointBERT files
    create_jointbert_files(samples, OUTPUT_DIR, tokenizer, use_subword_alignment=use_subword)
    
    print("\n[6/6] Done!")
    print(f"\nOutput files created in: {OUTPUT_DIR}")
    print(f"Cleaned CSV saved to: {CLEANED_CSV_PATH}")
    
    # Show example
    print("\n" + "=" * 60)
    print("Example conversion:")
    print("=" * 60)
    example_text = samples[0][0]
    example_intent = samples[0][1]
    print(f"Original: {example_text}")
    print(f"Intent: {example_intent}")
    
    plain_text, entities = parse_annotations(example_text)
    tokens, bio_tags = text_to_bio_tags(plain_text, entities)
    
    print(f"Plain text: {plain_text}")
    print(f"Word tokens: {tokens}")
    print(f"Word BIO tags: {bio_tags}")
    
    if use_subword and tokenizer:
        subword_tokens, aligned_tags = align_subword_tags(tokens, bio_tags, tokenizer)
        print(f"Subword tokens: {subword_tokens}")
        print(f"Aligned BIO tags: {aligned_tags}")


if __name__ == "__main__":
    main()
