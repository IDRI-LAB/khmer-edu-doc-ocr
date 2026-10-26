import unicodedata
import re

class KhmerSyllableReorder:
    """ Implements Khmer syllable reordering while keeping all original characters. """

    CONS = '[\u1780-\u17A2]'  # Khmer consonants
    RO = '\u179A'  # Khmer "Ro" character
    INDEPVOWEL = '[\u17A3-\u17B3]'  # Independent vowels
    DEPVOWEL = '[\u17B6-\u17C5]'  # Dependent vowels
    COENG = '\u17D2'  # Coeng subscript marker
    DIACRITIC = '[\u17C6-\u17D1\u17DD]'  # Khmer diacritics
    REGSHIFTER = '[\u17C9\u17CA]'  # Register shifters
    ROBAT = '\u17CC'  # Robat
    ZEROWIDTH = '[\u200B\u200C\u200D]'  # Zero-width spaces
    SPACING = '[\u17C7\u17C8]'  # Spacing diacritics
    NONSPACING = '[\u17C6\u17CB\u17CD-\u17D1\u17DD]'  # Non-spacing diacritics

    CONS_OR_INDEPVOWEL = CONS + '|' + INDEPVOWEL
    COENG_RO = COENG + RO
    SYLLPAT = '(?:{0})(?:{1}+(?:{0})|(?:{2}|{3}|{4})+)*'.format(CONS_OR_INDEPVOWEL, COENG, DEPVOWEL, DIACRITIC, ZEROWIDTH)
    CHUNKPAT = '(?:{0}+(?:{1}){2}?)|.'.format(COENG, CONS_OR_INDEPVOWEL, REGSHIFTER)

    def reorder_text(self, text):
        """ Reorder Khmer text while keeping all characters. """
        syllables = re.split(r'(' + self.SYLLPAT + ')', text)
        syllables = [self.reorder_syll(s) for s in syllables if s]
        return ''.join(syllables)

    def reorder_syll(self, syll):
        """ Reorder Khmer characters within a syllable while keeping all characters. """
        if len(syll) > 1 and re.match(self.CONS_OR_INDEPVOWEL, syll):
            base = syll[:1]
            chunks = re.split(r'(' + self.CHUNKPAT + ')', syll[1:])
            chunks = [c for c in chunks if c]

            dv, coeng, nonsp, sp, rs, robat = [], [], [], [], [], []

            for c in chunks:
                if re.match(self.DEPVOWEL, c): dv.append(c)
                elif re.match(self.COENG, c): coeng.append(re.sub(r'' + self.COENG + r'+', self.COENG, c))
                elif re.match(self.NONSPACING, c): nonsp.append(c)
                elif re.match(self.SPACING, c): sp.append(c)
                elif re.match(self.REGSHIFTER, c): rs.append(c)
                elif re.match(self.ROBAT, c): robat.append(c)

            # Ensure ្រ (coeng ro) appears last in subscript sequences
            numCoeng = len(coeng) - 1
            for i in range(numCoeng):
                if coeng[i] and re.match(self.COENG_RO, coeng[i]):
                    coeng.append(coeng[i])
                    coeng[i] = ''

            # Preserve all original characters, just reorder structure
            chunks = [base] + rs + robat + coeng + dv + nonsp + sp
            chunks = [c for i, c in enumerate(chunks) if i == 0 or c != chunks[i - 1]]

            syll = ''.join(chunks)
            syll = syll.replace('\u17C1\u17B8', '\u17BE')  # Replace េ + ី with ើ
            syll = syll.replace('\u17B8\u17C1', '\u17BE')  # Replace ី + េ with ើ
            syll = syll.replace('\u17C1\u17B6', '\u17C4')  # Replace េ + ា → ោ
            return syll
        return syll

# Initialize Khmer normalizer
khmer_reorder = KhmerSyllableReorder()

# Function to clean and normalize Khmer text
def clean_and_normalize(text):
    """
    1. Replace `្ត` with `្ដ` before any processing.
    2. Remove all spaces, tabs, zero-width spaces.
    3. Normalize Khmer text ordering.
    4. Apply Unicode NFC normalization.
    """
    if not isinstance(text, str) or text.strip() == "":
        return text  # Prevents errors on empty or NaN values

    # REPLACE `្ត` → `្ដ` FIRST
    text = text.replace("្ត", "្ដ")

    # REMOVE ALL SPACES, ZERO-WIDTH CHARACTERS, AND TABS
    text = re.sub(r'[\s\u200B\u200C\u200D]+', '', text)

    # APPLY KHMER REORDERING
    text = khmer_reorder.reorder_text(text)

    # APPLY UNICODE NFC NORMALIZATION
    text = unicodedata.normalize("NFC", text)

    return text
