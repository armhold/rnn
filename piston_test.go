package piston

import (
	"testing"
)

func TestProgs(t *testing.T) {
	piston := &Piston{}

	err := piston.Compile("samples", "prog1.go")
	if err != nil {
		t.Fatalf("prog1.go should succeed: %s", err)
	}

	err = piston.Compile("samples", "prog2.go")
	if err == nil {
		t.Fatalf("prog1.go should fail: %s", err)
	}
}
