package main

import (
	"flag"
	"github.com/armhold/rnn"
	"io/ioutil"
	"log"
	"os"
	"github.com/gonum/blas/blas64"
	"github.com/gonum/blas/cgo"
)

var (
	cpFile    string
	inputFile string
	maxIter   int
)

func init() {
	flag.StringVar(&cpFile, "cp", "rnn.tmp", "specify checkpoint file")
	flag.StringVar(&inputFile, "i", "input.txt", "specify input training file")
	flag.IntVar(&maxIter, "m", 1000000000, "max iterations")
	flag.Parse()
}

// run like: CGO_LDFLAGS="-L/usr/local/opt/openblas/lib -lopenblas" go run cmds/rnn/main.go
func main() {
	//defer profile.Start(profile.CPUProfile).Stop()

	// NB: for OpenBLAS you will likely need to run rnn with
	// CGO_LDFLAGS="-L/usr/local/opt/openblas/lib -lopenblas" or equivalent.
	blas64.Use(cgo.Implementation{})

	inputBytes, err := ioutil.ReadFile(inputFile)
	if err != nil {
		log.Fatalf("error reading input training file: %s", err)
	}

	input := string(inputBytes)

	var r *rnn.RNN

	// if there's an existing checkpoint file, restore from last checkpoint
	if _, err := os.Stat(cpFile); !os.IsNotExist(err) {
		r, err = rnn.LoadFrom(cpFile)
		if err != nil {
			log.Fatalf("unable to restore RNN from checkpoint file: %s", err)
		}
	} else {
		r = rnn.NewRNN(input, cpFile)
	}

	r.Run(maxIter)
}
