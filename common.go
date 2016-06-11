package rnn

import (
	"github.com/gonum/matrix/mat64"
	"math/rand"
)

func dot(a, b mat64.Matrix) *mat64.Dense {
	result := &mat64.Dense{}

	//log.Printf("a: %+v", a)
	//log.Printf("b: %+v", b)

	result.Mul(a, b)

	return result
}

func randomMatrix(r, c int) *mat64.Dense {
	result := mat64.NewDense(r, c, nil)
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
