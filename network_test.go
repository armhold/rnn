package piston

import (
	"log"
	"reflect"
	"testing"
)

func TestNetwork(t *testing.T) {

	n := NewNetwork("mary had a little lamb")

	log.Printf("%+v\n", n)

}

func TestMapInput(t *testing.T) {
	input := "mary had a little lamb"
	charToIndex, indexToChar := mapInput(input)

	expectedCTI := map[rune]int{
		'm': 0,
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
		'b': 11,
	}

	actualCTI := charToIndex
	if !reflect.DeepEqual(expectedCTI, actualCTI) {
		t.Fatalf("expected: %+v, got: %+v", expectedCTI, actualCTI)
	}

	actualITC := indexToChar
	expectedITC := map[int]rune{
		0:  'm',
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
		11: 'b',
	}
	if !reflect.DeepEqual(expectedITC, actualITC) {
		t.Fatalf("expected: %+v, got: %+v", expectedITC, actualITC)
	}
}
