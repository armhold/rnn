package piston

import (
	"testing"
	"io/ioutil"
	"reflect"
)

func TestSaveload(t *testing.T) {
	n := NewNetwork("Mary had a little lamb.")

	f, err := ioutil.TempFile("", "network")
	if err != nil {
		t.Fatal(err)
	}

	n.Save(f.Name())
	restored, err := LoadNetwork(f.Name())
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(n, restored) {
		t.Fatalf("expected: %+v, got %+v", n, restored)
	}
}
