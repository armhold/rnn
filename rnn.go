package rnn

import (
	discreterand "github.com/dgryski/go-discreterand"
	"github.com/gonum/matrix/mat64"
	"log"
	"math"
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

const (
	HiddenSize     = 100 // size of hidden layer of neurons
	SampleCount    = 200
	LearningRate   = 1e-1
	SequenceLength = 25 // number of steps to unroll the RNN for
)

type RNN struct {
	Wxh   *mat64.Dense // input to hidden weights
	Whh   *mat64.Dense // hidden to hidden weights
	Why   *mat64.Dense // hidden to output weights
	bh    *mat64.Dense // hidden bias
	by    *mat64.Dense // output bias
	hprev *mat64.Dense

	mWxh, mWhh, mWhy *mat64.Dense
	mbh, mby         *mat64.Dense

	data              string
	charToIndex       map[rune]int
	indexToChar       map[int]rune
	VocabSize         int
	n                 int // current iteration
	checkpointFile    string
	loss, smooth_loss float64
}

func NewRNN(input, checkpointFile string) *RNN {
	inputLen := len([]rune(input))
	if inputLen < SampleCount {
		log.Fatalf("input length: %d, must be >= %d", inputLen, SampleCount)
	}

	result := &RNN{data: input, checkpointFile: checkpointFile}
	result.charToIndex, result.indexToChar = mapInput(result.data)

	result.VocabSize = len(result.charToIndex)

	result.Wxh = randomMatrix(HiddenSize, result.VocabSize)
	result.Wxh.Scale(0.01, result.Wxh)

	result.Whh = randomMatrix(HiddenSize, HiddenSize)
	result.Whh.Scale(0.01, result.Whh)

	result.Why = randomMatrix(result.VocabSize, HiddenSize)
	result.Why.Scale(0.01, result.Why)

	result.bh = mat64.NewDense(HiddenSize, 1, nil)       // hidden bias
	result.by = mat64.NewDense(result.VocabSize, 1, nil) // output bias

	result.mWxh, result.mWhh, result.mWhy = zerosLike(result.Wxh), zerosLike(result.Whh), zerosLike(result.Why)
	result.mbh, result.mby = zerosLike(result.bh), zerosLike(result.by) // memory variables for Adagrad

	result.smooth_loss = -math.Log(1.0/float64(result.VocabSize)) * SequenceLength // loss at iteration 0

	return result
}

func (r *RNN) LossFunc(inputs, targets []int, hprev *mat64.Dense) (loss float64, dWxh *mat64.Dense, dWhh *mat64.Dense, dWhy *mat64.Dense, dbh *mat64.Dense, dby *mat64.Dense, lastHs *mat64.Dense) {
	xs := make(map[int]*mat64.Dense)
	hs := make(map[int]*mat64.Dense)
	ys := make(map[int]*mat64.Dense)
	ps := make(map[int]*mat64.Dense)

	hs[-1] = mat64.DenseCopyOf(hprev)

	// forward pass
	//
	for t := range inputs {
		// encode in 1-of-k
		//
		xs[t] = mat64.NewDense(r.VocabSize, 1, nil)
		xs[t].Set(inputs[t], 0, 1)

		// hidden state
		//
		dot1 := dot(r.Wxh, xs[t])

		dot2 := dot(r.Whh, hs[t-1])

		hs[t] = &mat64.Dense{}
		hs[t].Add(dot1, dot2)
		hs[t].Add(hs[t], r.bh)

		hs[t].Apply(func(i, j int, v float64) float64 {
			return math.Tanh(v)
		}, hs[t])

		// unnormalized log probabilities for next chars
		//
		ys[t] = dot(r.Why, hs[t])
		ys[t].Add(ys[t], r.by)

		// probabilities for next chars
		//
		ps[t] = expDivSumExp(ys[t])

		// softmax (cross-entropy loss)
		loss += -math.Log(ps[t].At(targets[t], 0))
	}

	//  backward pass: compute gradients going backwards
	//
	dWxh, dWhh, dWhy = zerosLike(r.Wxh), zerosLike(r.Whh), zerosLike(r.Why)
	dbh, dby = zerosLike(r.bh), zerosLike(r.by)
	dhnext := zerosLike(hs[0])

	for t := len(inputs) - 1; t >= 0; t-- {
		dy := mat64.DenseCopyOf(ps[t])
		dy.Set(targets[t], 0, dy.At(targets[t], 0)-1) // backprop into y

		dWhy.Add(dWhy, dot(dy, hs[t].T()))
		dby.Add(dby, dy)

		dh := dot(r.Why.T(), dy) // backprop into h
		dh.Add(dh, dhnext)

		// backprop through tanh nonlinearity
		dhraw := &mat64.Dense{}
		dhraw.MulElem(hs[t], hs[t])
		dhraw.Apply(func(i, j int, v float64) float64 {
			return 1 - v
		}, dhraw)
		dhraw.MulElem(dhraw, dh)

		dbh.Add(dbh, dhraw)
		dWxh.Add(dWxh, dot(dhraw, xs[t].T()))
		dWhh.Add(dWhh, dot(dhraw, hs[t-1].T()))
		dhnext = dot(r.Whh.T(), dhraw)
	}

	// clip to mitigate exploding gradients
	clipTo(-5, 5, dWxh, dWhh, dWhy, dbh, dby)

	lastHs = hs[len(inputs)-1]
	return
}

func (r *RNN) sample(h *mat64.Dense, seedIx, count int) []int {
	x := mat64.NewDense(r.VocabSize, 1, nil)
	x.Set(seedIx, 0, 1)

	var ixes []int

	for i := 0; i < count; i++ {
		dot1 := dot(r.Wxh, x)
		dot2 := dot(r.Whh, h)

		// NB: h gets re-assigned here
		h = &mat64.Dense{}
		h.Add(dot1, dot2)
		h.Add(h, r.bh)

		h.Apply(func(i, j int, v float64) float64 {
			return math.Tanh(v)
		}, h)

		y := dot(r.Why, h)
		y.Add(y, r.by)

		p := expDivSumExp(y)
		at := discreterand.NewAlias(ravel(p), rand.NewSource(time.Now().UTC().UnixNano()))
		index := at.Next()
		pythonRange := rangeToArray(r.VocabSize)
		ix := pythonRange[index]
		//log.Printf("chose index %d -> %d", index, ix)
		x = mat64.NewDense(r.VocabSize, 1, nil)
		x.Set(ix, 0, 1)
		ixes = append(ixes, ix)
	}

	return ixes
}

func (r *RNN) Run(maxIter int) {
	runes := []rune(r.data)
	inputLen := len(runes)

	p := 0

	inputs := make([]int, SequenceLength)
	targets := make([]int, SequenceLength)

	for r.n < maxIter {
		// prepare inputs (we're sweeping from left to right in steps seq_length long)
		if p+SequenceLength+1 >= inputLen || r.hprev == nil {
			r.hprev = mat64.NewDense(HiddenSize, 1, nil) // reset RNN memory
			p = 0                                        // go from start of data
		}

		for i := 0; i < SequenceLength; i++ {
			inputs[i] = r.charToIndex[runes[p+i]]
			targets[i] = r.charToIndex[runes[p+i+1]]
		}

		// sample from the model now and then
		if r.n%100 == 0 {
			sample_ix := r.sample(r.hprev, inputs[0], SampleCount)
			chars := make([]rune, len(sample_ix))
			for i, ix := range sample_ix {
				ch := r.indexToChar[ix]
				chars[i] = ch
			}
			s := string(chars)
			log.Print(s)
		}

		if r.n%1000 == 0 && r.checkpointFile != "" {
			err := r.SaveTo(r.checkpointFile)
			if err != nil {
				log.Fatalf("unable to save checkpoint file: %s", err)
			}
		}

		// forward seq_length characters through the net and fetch gradient
		var dWxh, dWhh, dWhy, dbh, dby *mat64.Dense

		r.loss, dWxh, dWhh, dWhy, dbh, dby, r.hprev = r.LossFunc(inputs, targets, r.hprev)
		r.smooth_loss = r.smooth_loss*0.999 + r.loss*0.001
		if r.n%100 == 0 {
			log.Printf("iter %d, loss: %f", r.n, r.smooth_loss) // print progress
		}

		// perform parameter update with Adagrad
		//
		adagrad := func(param, dparam, mem *mat64.Dense) {
			dSquared := &mat64.Dense{}
			dSquared.MulElem(dparam, dparam)
			mem.Add(mem, dSquared)

			tmp := &mat64.Dense{}
			tmp.Scale(-LearningRate, dparam)

			sqrtMem := &mat64.Dense{}
			sqrtMem.Apply(func(i, j int, v float64) float64 {
				return v + 1e-8
			}, mem)

			sqrtMem.Apply(func(i, j int, v float64) float64 {
				return math.Sqrt(v) // adagrad update
			}, sqrtMem)

			tmp.DivElem(tmp, sqrtMem)
			param.Add(param, tmp)
		}

		adagrad(r.Wxh, dWxh, r.mWxh)
		adagrad(r.Whh, dWhh, r.mWhh)
		adagrad(r.Why, dWhy, r.mWhy)
		adagrad(r.bh, dbh, r.mbh)
		adagrad(r.by, dby, r.mby)

		p = p + SequenceLength
		r.n++
	}
}

func randomMatrix(r, c int) *mat64.Dense {
	result := mat64.NewDense(r, c, nil)

	for row := 0; row < r; row++ {
		for col := 0; col < c; col++ {
			result.Set(row, col, rand.NormFloat64())
		}

	}

	return result
}

func mapInput(input string) (charToIndex map[rune]int, indexToChar map[int]rune) {
	charToIndex = make(map[rune]int)
	indexToChar = make(map[int]rune)

	// find unique chars in input and map them to/from ints
	uniques := 0
	for _, ch := range input {
		if _, ok := charToIndex[ch]; !ok {
			charToIndex[ch] = uniques
			indexToChar[uniques] = ch
			uniques++
		}
	}

	return charToIndex, indexToChar
}

func dot(a, b mat64.Matrix) *mat64.Dense {
	result := &mat64.Dense{}
	result.Mul(a, b)
	return result
}

func zerosLike(a mat64.Matrix) *mat64.Dense {
	r, c := a.Dims()
	return mat64.NewDense(r, c, nil)
}

func clipTo(leftRange, rightRange float64, matrices ...*mat64.Dense) {
	for _, m := range matrices {
		m.Apply(func(i, j int, v float64) float64 {
			if v < leftRange {
				return leftRange
			} else if v > rightRange {
				return rightRange
			}

			return v
		}, m)
	}
}

func expDivSumExp(m *mat64.Dense) *mat64.Dense {
	exp := &mat64.Dense{}
	exp.Apply(func(i, j int, v float64) float64 {
		return math.Exp(v)
	}, m)

	sumExp := mat64.Sum(exp)

	result := &mat64.Dense{}
	result.Apply(func(i, j int, v float64) float64 {
		return v / sumExp
	}, exp)

	return result
}

// flatten a matrix into a one-dimensional slice of floats
// TODO: unit test
func ravel(m *mat64.Dense) []float64 {
	r, c := m.Dims()

	result := make([]float64, r*c)

	i := 0

	for row := 0; row < r; row++ {
		for col := 0; col < c; col++ {
			result[i] = m.At(row, col)
			i++
		}
	}

	return result
}

// emulate Python's range()
func rangeToArray(n int) []int {
	result := make([]int, n)

	for i := 0; i < n; i++ {
		result[i] = i
	}

	return result
}
