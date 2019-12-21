""" from https://github.com/keithito/tacotron """
#!/usr/bin/python
# -- coding: utf-8 --
 
import sys
import re

softletters=set(u"яёюиье")
startsyl=set(u"#ъьаяоёуюэеиы-")
others = set(["#", "+", u"ь", u"ъ"])
sybols = 'йцукенгшщзхъёфывапролджэячсмитьбю-'
punctuation = '!\'",.:;? '



#cmudct = read_cmu('ru.dic')

softhard_cons = {                                                                
    u"б" : u"b",
    u"в" : u"v",
    u"г" : u"g",
    u"Г" : u"g",
    u"д" : u"d",
    u"з" : u"z",
    u"к" : u"k",
    u"л" : u"l",
    u"м" : u"m",
    u"н" : u"n",
    u"п" : u"p",
    u"р" : u"r",
    u"с" : u"s",
    u"т" : u"t",
    u"ф" : u"f",
    u"х" : u"h"
}

other_cons = {
    u"ж" : u"zh",
    u"ц" : u"c",
    u"ч" : u"ch",
    u"ш" : u"sh",
    u"щ" : u"sch",
    u"й" : u"j"
}
                                
vowels = {
    u"а" : u"a",
    u"я" : u"a",
    u"у" : u"u",
    u"ю" : u"u",
    u"о" : u"o",
    u"ё" : u"o",
    u"э" : u"e",
    u"е" : u"e",
    u"и" : u"i",
    u"ы" : u"y",
}                                

def pallatize(phones):
    for i, phone in enumerate(phones[:-1]):
        if phone[0] in softhard_cons:
            if phones[i+1][0] in softletters:
                phones[i] = (softhard_cons[phone[0]] + "j", 0)
            else:
                phones[i] = (softhard_cons[phone[0]], 0)
        if phone[0] in other_cons:
            phones[i] = (other_cons[phone[0]], 0)

def convert_vowels(phones):
    new_phones = []
    prev = ""
    for phone in phones:
        if prev in startsyl:
            if phone[0] in set(u"яюеё"):
                new_phones.append("j")
        if phone[0] in vowels:
            new_phones.append(vowels[phone[0]] + str(phone[1]))
        else:
            new_phones.append(phone[0])
        prev = phone[0]

    return new_phones

def convert(stressword):


    k = 0
    words_dct = {}
    word = ''
    for i in stressword:
        if i in punctuation:
            if word != '':
                words_dct[k] = word
                word = ''
                k += 1
            words_dct[k] = i
            k += 1
            word = ''
        elif i in sybols:
            word += i
    if word != '':
        words_dct[k] = word

    #return words_dct
    phones_dct = {}
    for i in words_dct:
        if words_dct[i] in punctuation:
            phones_dct[i] = words_dct[i]
        elif words_dct[i] in cmudct:
            phones_dct[i] = cmudct[words_dct[i]]
        else:
            phones = ("#" + words_dct[i] + "#")


            # Assign stress marks
            stress_phones = []
            stress = 0
            for phone in phones:
                if phone == "+":
                    stress = 1
                else:
                    stress_phones.append((phone, stress))
                    stress = 0
            
            # Pallatize
            pallatize(stress_phones)
            
            # Assign stress
            phones = convert_vowels(stress_phones)

            # Filter
            phones = [x for x in phones if x not in others]

            phones_dct[i] = " ".join(phones)

    return " ".join(list(phones_dct.values()))

# for line in open(sys.argv[1]):
#     stressword = re.sub("\s+", " ", line.strip().lower())
#     print(stressword.replace("+", ""),  convert(stressword))



# valid_symbols = [
#   'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
#   'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
#   'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
#   'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
#   'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
#   'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
#   'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
# ]

valid_symbols = ['a1', 'a0', 'i1', 'h', 'm', 'lj', 'b', 'z', 'k', 'd', 'nj', 'i0',
 'j', 'e1', 'e0', 'o1', 'u1', 'l', 't', 'o0', 'n', 'r', 'rj', 'sj', 'fj', 's', 'sch',
  'hj', 'gj', 'zh', 'mj', 'v', 'u0', 'ch', 'y0', 'zj', 'kj', 'dj', 'f', 'sh', 'vj',
   'tj', 'c', 'bj', 'g', 'y1', 'p', 'pj']

_valid_symbol_set = set(valid_symbols)

def read_cmu(filse):
    data = open(filse).read().split('\n')
    cmudct = {}
    for i in data:
        info = i.split(' ')
        cmudct[info[0]] = ' '.join(info[1:])
    return cmudct


class CMUDict:
  '''Thin wrapper around CMUDict data. http://www.speech.cs.cmu.edu/cgi-bin/cmudict'''
  def __init__(self, file_or_path, keep_ambiguous=True):
    if isinstance(file_or_path, str):
      entries = read_cmu(file_or_path)
      # with open(file_or_path, encoding='latin-1') as f:
      #   entries = _parse_cmudict(f)
    else:
      entries = _parse_cmudict(file_or_path)
    if not keep_ambiguous:
      entries = {word: pron for word, pron in entries.items() if len(pron) == 1}
    self._entries = entries


  def __len__(self):
    return len(self._entries)


  def _read_cmu(self, filse):
    data = open(filse).read().split('\n')
    cmudct = {}
    for i in data:
        info = i.split(' ')
        cmudct[info[0]] = ' '.join(info[1:])
    return cmudct


  def lookup(self, word):
    '''Returns list of ARPAbet pronunciations of the given word.'''
    if word in punctuation:
        return None
    elif word in self._entries:
        return self._entries.get(word)
    else:
        phones = ("#" + word + "#")


        # Assign stress marks
        stress_phones = []
        stress = 0
        for phone in phones:
            if phone == "+":
                stress = 1
            else:
                stress_phones.append((phone, stress))
                stress = 0
        
        # Pallatize
        pallatize(stress_phones)
        
        # Assign stress
        phones = convert_vowels(stress_phones)

        # Filter
        phones = [x for x in phones if x not in others]

        return " ".join(phones)

    return self._entries.get(word)



_alt_re = re.compile(r'\([0-9]+\)')


def _parse_cmudict(file):
  cmudict = {}
  for line in file:
    if len(line) and (line[0] >= 'A' and line[0] <= 'Z' or line[0] == "'"):
      parts = line.split('  ')
      word = re.sub(_alt_re, '', parts[0])
      pronunciation = _get_pronunciation(parts[1])
      if pronunciation:
        if word in cmudict:
          cmudict[word].append(pronunciation)
        else:
          cmudict[word] = [pronunciation]
  return cmudict

  def _read_cmu(filse):
    data = open(filse).read().split('\n')
    cmudct = {}
    for i in data:
        info = i.split(' ')
        cmudct[info[0]] = ' '.join(info[1:])
    return cmudct


def _get_pronunciation(s):
  parts = s.strip().split(' ')
  for part in parts:
    if part not in _valid_symbol_set:
      return None
  return ' '.join(parts)
