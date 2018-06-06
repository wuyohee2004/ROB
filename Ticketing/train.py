from gensim.models import Word2Vec
import pickle,sys,io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')
fileSegWordDonePath = 'corpusSegDone.txt'
outp1 = "outlookmodel"
outp2 = "outlookvector.text"
datadump = "data.pkl"

model = Word2Vec.load('outlookmodel')
print(model[u'outlook'])
with open(datadump,'rb') as inp:
	data = pickle.load(inp)

# print(sorted(data[:][0], key=lambda x: len(x)[-1]))
print()