package piston

import (
	"reflect"
	"testing"
	"github.com/gonum/matrix/mat64"
)

func TestNetwork(t *testing.T) {
	n := NewNetwork("Mary had a little lamb.")

	if n.VocabSize != 14 {
		t.Fatalf("VocabSize expected: 12, got: %d", n.VocabSize)
	}
}

// testing my understanding of how to multiply a mat64 matrix by a scalar
func TestMultiply(t *testing.T) {
	a := mat64.NewDense(2, 2, []float64 { 0, 1, 2, 3 })
	a.Scale(2, a)

	expectedRows := [][]float64 {
		{ 0, 2 },
		{ 4, 6 },
	}

	r, c := a.Dims()
	if r != 2 || c != 2 {
		t.Fatalf("rows/cols changed: %d, %d", r, c)
	}

	for i, row := range expectedRows {
		expected := row
		actual := a.RawRowView(i)

		if ! reflect.DeepEqual(actual, expected) {
			t.Errorf("for row %d, expected: %+v, got: %+v", i, expected, actual)
		}
	}
}

func TestMapInput(t *testing.T) {
	input := "Mary had a little lamb."
	charToIndex, indexToChar := mapInput(input)

	expectedCTI := map[rune]int{
		'M': 0,
		'a': 1,
		'r': 2,
		'y': 3,
		' ': 4,
		'h': 5,
		'd': 6,
		'l': 7,
		'i': 8,
		't': 9,
		'e': 10,
		'm': 11,
		'b': 12,
		'.': 13,
	}

	actualCTI := charToIndex
	if !reflect.DeepEqual(expectedCTI, actualCTI) {
		t.Fatalf("expected: %+v, got: %+v", expectedCTI, actualCTI)
	}

	actualITC := indexToChar
	expectedITC := map[int]rune{
		0:  'M',
		1:  'a',
		2:  'r',
		3:  'y',
		4:  ' ',
		5:  'h',
		6:  'd',
		7:  'l',
		8:  'i',
		9:  't',
		10: 'e',
		11: 'm',
		12: 'b',
		13: '.',
	}
	if !reflect.DeepEqual(expectedITC, actualITC) {
		t.Fatalf("expected: %+v, got: %+v", expectedITC, actualITC)
	}
}
