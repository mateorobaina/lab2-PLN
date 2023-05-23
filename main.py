import nltk
from nltk import word_tokenize
from nltk.corpus import brown

race1 = nltk.tag.str2tuple('race/NN')
race2 = nltk.tag.str2tuple('race/VB')
# Tamañon del brown corpus: 1161192
bw = brown.tagged_words()
lenBw = len(brown.tagged_words())
fr1 = bw.count(race1)  # 94
fr2 = bw.count(race2)  # 4
# Es mas usado como noun que como verbo ya que fr1 > fr2
#print(fr2)

#unigram_tagger = nltk.tag.UnigramTagger(brown.tagged_sents(categories='news')[:5000])
S = "The Secretariat is expected to race tomorrow."
#print(unigram_tagger.tag(S_tok))         #NN
S_tok = word_tokenize(S)

brown_news = nltk.corpus.brown.tagged_sents(categories='news')[:5000]
hmm_tagger = nltk.HiddenMarkovModelTagger.train(brown_news)

tagged_tokens = hmm_tagger.tag(S_tok)
print(tagged_tokens)
