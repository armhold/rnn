package piston

import (
	"log"
	"testing"
	"time"
	"os"
	"path"
)

func TestSaveAndCompile(t *testing.T) {
	start := time.Now()

	piston := &Piston{}

	p1 :=
		`package samples

import "fmt"

func main() {
	doIt()
}


func doIt() {
	fmt.Printf("Hello, World!\n")
}
`
	f := "prog.go"

	defer func() { os.Remove(path.Join(scratchDir, f)) }()
	err := piston.SaveAndCompile(f, p1)

	if err != nil {
		t.Fatalf("should succeed: %s", err)
	}
	dur := time.Since(start)
	log.Printf("dur: %s\n", dur)

	p2 :=
		`package samples

import "fmt"

func main() {
	doIt()
}


func doIt() {
	fmt.Printf("Hello, World!\n")
}

x
`
	err = piston.SaveAndCompile(f, p2)
	if err == nil {
		t.Fatalf("should fail: %s", err)
	}

}

func TestParse(t *testing.T) {
	piston := &Piston{}

	src :=
		`package samples

	import "fmt"

	func main() {
		doIt()
	}


	func doIt() {
		fmt.Printf("Hello, World!\n")
	}
	`

	piston.Parse(src)
}
