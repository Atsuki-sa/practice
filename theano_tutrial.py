#チュートリアル触って日本語注釈入れつついじったもの

sentence = [383, 189,  13, 193, 208, 307, 195, 502, 260, 539, 7,  60,  72, 8, 350, 384]
labels = [126, 126, 126, 126, 126,  48,  50, 126,  78, 123,  81, 126,  15, 14,  89,  89]

#前後二単語をとってきたベクトルを作る（二次元配列）
def contextwin(l, win):
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)

    lpadded = win // 2 * [-1] + l + win // 2 * [-1]
    out = [lpadded[i:(i + win)] for i in range(len(l))]

    assert len(out) == len(l)
    return out

c_sentence = contextwin(sentence, 5)

#おまじない的なアレ。
import theano, numpy
from theano import tensor as T
from collections import OrderedDict
theano.config.floatX = "float32"

#nv, de, cs = 1000, 50, 5
nv = 1000   #vocab size
de = 50 #word embedding size
cs = 5#window size

#numpy .random.uniform -1 から 1を生成
#paddingのシンボルである「-1」を加えたnv+1をサイズにする
#（1001, 50）
#0.2をかけて小さくしているだけで、あまり意味はなし
#theano.shared どのパラメータを更新したいかしていするための関数
#theanoで使える関数に変更
embeddings = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (nv+1, de)).astype(theano.config.floatX))
print(embeddings.get_value())

#theanoをコンパイルするときに、どういう入力をとるか
#imatrixは2次元の入力ですよという宣言
#型は事前に宣言しなければならない
#cの型宣言みたいなやつ
idxs = T.imatrix()

#[-1,-1,x1,x2,x3] -> -1の50次元、。。。x３の50次元を足しあわせて250次元にする
#x = embeddings[idxs].reshape((idxs.shape[0], de*cs))
#f = theano.function(inputs=[idxs], outputs=x)
#input にf()の中身にいれるものが対応する
#print(f())

nh = 100 # dimension of the hidden layer中間層のサイズ
nc = 4 # number of classes分類数
ne = 1500 # number of word embeddings in the vocabulary
de = 50 # dimension of the word embeddings
cs = 5 # word window context size

emb = theano.shared(name='embeddings', value=0.2 * numpy.random.uniform(-1.0, 1.0, (ne+1, de)).astype(theano.config.floatX))

print(numpy.random.uniform(-1.0, 1.0, (de * cs, nh)))
#(250,100)コンテクストウィンドウと中間層のサイズのウィンドウにする
wx = theano.shared(name='wx', value=0.2 * numpy.random.uniform(-1.0, 1.0, (de * cs, nh)).astype(theano.config.floatX))

wh = theano.shared(name='wh', value=0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh)) .astype(theano.config.floatX))

w = theano.shared(name='w', value=0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nc)).astype(theano.config.floatX))

#バイアスはs全て０でよい
bh = theano.shared(name='bh', value=numpy.zeros(nh, dtype=theano.config.floatX))
b = theano.shared(name='b', value=numpy.zeros(nc, dtype=theano.config.floatX))
h0 = theano.shared(name='h0', value=numpy.zeros(nh, dtype=theano.config.floatX))

params = [emb, wx, wh, w, bh, b, h0]

idxs = T.imatrix()

x = emb[idxs].reshape((idxs.shape[0], de*cs))
y_sentence = T.ivector('y_sentence')

def recurrence(x_t, h_tm1):
	h_t = T.nnet.sigmoid(T.dot(x_t, wx) + T.dot(h_tm1, wh) + bh)
    #h_t = T.nnet.relu(T.dot(x_t, wx) + T.dot(h_tm1, wh) + bh)とか
    #x[::-1]で並び替えししてLSTM
	s_t = T.nnet.softmax(T.dot(h_t, w) + b)
	return [h_t, s_t]

[h, s], _ = theano.scan(fn=recurrence, sequences=x, outputs_info=[h0, None], n_steps=x.shape[0])


#いらない次元を消す
p_y_given_x_sentence = s[:, 0, :]
y_pred = T.argmax(p_y_given_x_sentence, axis=1)

##学習
lr = T.scalar('lr') #学習率

#negative log likelyhood 負の対数尤度
#t.arangeはi for i in range(16)
#y_sentenceはゴールド
#文全体の誤差をクロスエントロピーで算出
sentence_nll = -T.mean(T.log(p_y_given_x_sentence) [T.arange(x.shape[0]), y_sentence])

#自動微分して、勾配を求める
sentence_gradients = T.grad(sentence_nll, params)
f = theano.function([idxs, y_sentence], sentence_gradients)
tmp_x = contextwin([0,1,2,3,4], 5)
tmp_y = [0,1,0,1,0]
f(tmp_x, tmp_y)

#現在の値から学習率をかけた勾配をかける
sentence_updates = OrderedDict((p, p - lr*g) for p, g in zip(params, sentence_gradients))

sentence_updates = OrderedDict((p, p - lr*g) for p, g in zip(params, sentence_gradients))

f = theano.function([idxs, y_sentence], sentence_nll)
f(tmp_x, tmp_y)

classify = theano.function(inputs=[idxs], outputs=y_pred)
sentence_train = theano.function(inputs=[idxs, y_sentence, lr], outputs=sentence_nll, updates=sentence_updates)

print(sentence_train(tmp_x, tmp_y,0.1))
print(classify(tmp_x))

for i in range(100):
	print(i, sentence_train(tmp_x, tmp_y,0.1))

print(classify(tmp_x))
