package piston

import (
	"github.com/gonum/matrix/mat64"
	"math/rand"
	"time"
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
	InputToHiddenWeights  *mat64.Dense
	HiddenToHiddenWeights *mat64.Dense
	HiddenToOutputWeights *mat64.Dense
	HiddenBias            *mat64.Dense
	OutputBias            *mat64.Dense

	charToIndex           map[rune]int
	indexToChar           map[int]rune
	VocabSize             int
}

func NewNetwork(input string) *Network {
	result := &Network{}
	result.charToIndex, result.indexToChar = mapInput(input)
	result.VocabSize = len(result.charToIndex)

	result.HiddenBias = mat64.NewDense(HiddenSize, 1, nil)

	return result
}

func randomize(m *mat64.Dense) {
	r, c := m.Dims()

	for row := 0; row < r; row++ {
		for col := 0; col < c; col++ {
			m.Set(row, col, rand.Float64())
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
