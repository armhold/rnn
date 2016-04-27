package rnn

import (
	"io/ioutil"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestAllFieldsPersisted(t *testing.T) {
	f, err := ioutil.TempFile("", "network")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { os.Remove(f.Name()) }()

	rnn := NewRNN(strings.Repeat("Mary had a little lamb. ", 10), f.Name())

	// exercising it a bit will cause changes in the internal state... changes that we want to make sure
	// get preserved during checkpointing.
	rnn.Run(10)

	rnn.SaveTo(f.Name())
	restored, err := LoadFrom(f.Name())
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(rnn, restored) {
		t.Fatalf("expected: %+v, got %+v", rnn, restored)
	}
}
