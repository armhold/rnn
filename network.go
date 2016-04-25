package piston

import (
	"github.com/gonum/matrix/mat64"
	"math/rand"
	"time"
	"log"
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
	HiddenBias  *mat64.Dense
	OutputBias  *mat64.Dense

	charToIndex map[rune]int
	indexToChar map[int]rune
	VocabSize   int
}

func NewNetwork(input string) *Network {
	result := &Network{}
	result.charToIndex, result.indexToChar = mapInput(input)
	result.VocabSize = len(result.charToIndex)
	result.HiddenBias = mat64.NewDense(HiddenSize, 1, nil)

	result.Wxh = randomMatrix(HiddenSize, result.VocabSize)
	result.Wxh.Scale(0.01, result.Wxh)

	result.Whh = randomMatrix(HiddenSize, HiddenSize)
	result.Whh.Scale(0.01, result.Whh)

	result.Why = randomMatrix(result.VocabSize, HiddenSize)
	result.Why.Scale(0.01, result.Why)

	result.HiddenBias = mat64.NewDense(HiddenSize, 1, nil)
	result.OutputBias = mat64.NewDense(result.VocabSize, 1, nil)

	return result
}

func (n *Network) Run(input string) {
	runes := []rune(input)
	inputLen := len(runes)

	iter := 0
	p := 0

	var hprev *mat64.Dense
	inputs := make([]int, SequenceLength)
	targets := make([]int, SequenceLength)

	for {
		if p + SequenceLength + 1 >= inputLen || iter == 0 {
			// TODO: hprev = np.zeros()... etc
			p = 0

			hprev = mat64.NewDense(HiddenSize, 1, nil) // reset RNN memory
		}


		for i := 0; i < SequenceLength + 1; i++ {
			inputs[i] = n.charToIndex[runes[p + i]]
			targets[i] = n.charToIndex[runes[p + i + 1]]
		}

		log.Printf("inputs: %q, p: %d", inputs, p)
		n.LossFunc(inputs, targets, hprev)


		p = p + SequenceLength
		iter += 1
	}
}



func (n *Network) LossFunc(inputs, targets []int, hprev *mat64.Dense) {
	charCount := len(inputs)

	xs := mat64.NewDense(charCount, n.VocabSize, nil)
	hs := mat64.NewDense(charCount, )

	var loss float64

	// forward pass
	//
	for t, _ := range inputs {
		// encode in 1-of-k
		xs.Set(t, inputs[t], 1)

		xst := xs.RowView(t)
		_, c := n.Wxh.Dims()
		r, _ := xst.Dims()

		l := mat64.NewDense(c, r, nil)
		l.Mul(n.Wxh, xst)

		_, c = n.Whh.Dims()
		r, _ = hs.RowView(t-1)


		r:= mat64.NewDense(c, r, nil)

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
