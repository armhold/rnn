package piston

import (
	"github.com/gonum/matrix/mat64"
	"math/rand"
	"time"
	"log"
	"math"
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

const (
	HiddenSize     = 100 // size of hidden layer of neurons
	LearningRate   = 1e-1
	SequenceLength = 25 // number of steps to unroll the RNN for
)

type Network struct {
	Wxh         *mat64.Dense  // input to hidden weights
	Whh         *mat64.Dense  // hidden to hidden weights
	Why         *mat64.Dense  // hidden to output weights
	bh          *mat64.Dense  // hidden bias
	by          *mat64.Dense  // output bias

	charToIndex map[rune]int
	indexToChar map[int]rune
	VocabSize   int
}

func NewNetwork(input string) *Network {
	result := &Network{}
	result.charToIndex, result.indexToChar = mapInput(input)
	result.VocabSize = len(result.charToIndex)
	result.bh = mat64.NewDense(HiddenSize, 1, nil)

	result.Wxh = randomMatrix(HiddenSize, result.VocabSize)
	result.Wxh.Scale(0.01, result.Wxh)

	result.Whh = randomMatrix(HiddenSize, HiddenSize)
	result.Whh.Scale(0.01, result.Whh)

	result.Why = randomMatrix(result.VocabSize, HiddenSize)
	result.Why.Scale(0.01, result.Why)

	result.bh = mat64.NewDense(HiddenSize, 1, nil)
	result.by = mat64.NewDense(result.VocabSize, 1, nil)

	return result
}

func (n *Network) Run(input string) {
	runes := []rune(input)
	inputLen := len(runes)

	iter := 0
	p := 0

	//var hprev *mat64.Dense
	var hprev []float64

	inputs := make([]int, SequenceLength)
	targets := make([]int, SequenceLength)

	for {
		if p + SequenceLength + 1 >= inputLen || iter == 0 {
			// TODO: hprev = np.zeros()... etc
			p = 0

			hprev = make([]float64, HiddenSize) // reset RNN memory
			//hprev = mat64.NewDense(HiddenSize, 1, nil) // reset RNN memory
		}


		for i := 0; i < SequenceLength; i++ {
			inputs[i] = n.charToIndex[runes[p + i]]
			targets[i] = n.charToIndex[runes[p + i + 1]]
		}

		log.Printf("inputs: %q, p: %d", inputs, p)
		n.LossFunc(inputs, targets, hprev)


		p = p + SequenceLength
		iter += 1
	}
}



func (n *Network) LossFunc(inputs, targets []int, hprev []float64) {
	charCount := len(inputs)

	xs := mat64.NewDense(charCount, n.VocabSize, nil)
	hs := mat64.NewDense(charCount, len(hprev), nil)
	ys := mat64.NewDense(charCount, n.VocabSize, nil)  // # cols might be wrong
	ps := mat64.NewDense(charCount, n.VocabSize, nil)  // # cols might be wrong

	r, _ := hs.Dims()
	hs.SetRow(r - 1, hprev)

	var loss float64

	// forward pass
	//
	for t, _ := range inputs {
		// encode in 1-of-k
		xs.Set(t, inputs[t], 1)

		log.Printf("xs: %+v", xs)
		log.Printf("hs: %+v", hs)
		log.Printf("ys: %+v", ys)
		log.Printf("ps: %+v", ps)
		log.Printf("loss: %+v", loss)
		log.Printf("t: %d", t)

		log.Printf("n.Wxh: %+v", n.Wxh)
		//log.Printf("xs.ColView(%d): %+v", t, xs.ColView(t))
		log.Printf("xs.RowView(%d): %+v", t, xs.RowView(t))

		// 100x14 dot 14x1

		dot1 := &mat64.Dense{}
		dot1.Mul(n.Wxh, xs.RowView(t))

		log.Printf("dot1: %+v", dot1)

		dot2 := &mat64.Dense{}
		tMinus1 := t - 1

		if tMinus1 < 0 {
			r, _ := hs.Dims()
			tMinus1 = r - 1
		}
		dot2.Mul(n.Whh, hs.RowView(tMinus1))

		log.Printf("dot2: %+v", dot2)

		sum := &mat64.Dense{}
		sum.Add(dot1, dot2)
		log.Printf("sum: %+v", sum)

		sum.Add(sum, n.bh)
		sum.Apply(func(i, j int, v float64) float64 {
			return math.Tanh(v)
		}, sum)

	}
}




func randomMatrix(rows, cols int) *mat64.Dense {
	result := mat64.NewDense(rows, cols, nil)
	randomize(result)

	return result
}



func randomize(m *mat64.Dense) {
	r, c := m.Dims()

	for row := 0; row < r; row++ {
		for col := 0; col < c; col++ {
			m.Set(row, col, rand.NormFloat64())
		}

	}
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
