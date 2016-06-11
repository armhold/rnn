package rnn

import (
	"github.com/gonum/matrix/mat64"
	"testing"
)

func TestSimpleRNN(t *testing.T) {

	inputCount := 3
	outputCount := 3
	hiddenCount := 100

	s := NewSimpleRNN(inputCount, outputCount, hiddenCount)

	input := mat64.NewDense(3, 1, []float64{0.0, 1.0, 0.0})
	expectedOutput := mat64.NewDense(3, 1, []float64{0.0, 1.0, 0.0})

	s.Train(input, expectedOutput, 100)

}
