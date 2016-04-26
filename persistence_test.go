package piston

import (
	"testing"
	"io/ioutil"
	"reflect"
)

func TestSaveload(t *testing.T) {
	f, err := ioutil.TempFile("", "network")
	if err != nil {
		t.Fatal(err)
	}

	rnn := NewRNN("Mary had a little lamb.", f.Name())

	rnn.SaveTo(f.Name())
	restored, err := LoadFrom(f.Name())
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(rnn, restored) {
		t.Fatalf("expected: %+v, got %+v", rnn, restored)
	}
}
