package rnn

import (
	"github.com/gonum/matrix/mat64"
	"log"
	"math"
)

// SimpleRNN is a simpler version of an RNN
type SimpleRNN struct {
	// weights
	InputToHidden  *mat64.Dense
	HiddenToOutput *mat64.Dense
	HiddenToHidden *mat64.Dense

	// state
	Hidden *mat64.Dense
	Output *mat64.Dense
}

type Weights *mat64.Dense
type State *mat64.Dense

func NewSimpleRNN(inputCount, outputCount, hiddenCount int) *SimpleRNN {
	result := &SimpleRNN{}

	// state
	result.Hidden = mat64.NewDense(hiddenCount, 1, nil)
	result.Output = mat64.NewDense(outputCount, 1, nil)

	// weights
	result.InputToHidden = randomMatrix(hiddenCount, inputCount)
	result.HiddenToOutput = randomMatrix(outputCount, hiddenCount)
	result.HiddenToHidden = randomMatrix(hiddenCount, hiddenCount)

	return result
}

func (s *SimpleRNN) Train(input, expectedOutput *mat64.Dense, iter int) {
	for i := 0; i < iter; i++ {
		s.Forward(input)
		log.Printf("%+v", s.Output)
	}
}

func (s *SimpleRNN) Forward(input *mat64.Dense) {
	// update hidden state
	dot1 := dot(s.HiddenToHidden, s.Hidden)
	dot2 := dot(s.InputToHidden, input)

	dot1.Add(dot1, dot2)

	s.Hidden.Apply(func(i, j int, v float64) float64 {
		return math.Tanh(v)
	}, dot1)

	// compute hidden vector
	s.Output = dot(s.HiddenToOutput, s.Hidden)

	s.Output.Apply(func(i, j int, v float64) float64 {
		return math.Tanh(v)
	}, s.Output)
}

func (s *SimpleRNN) Back(input, expectedOutput *mat64.Dense) {
	//errors := *mat64.Dense{}
	//
	//errors.Sub(s.Output, expectedOutput)

}
