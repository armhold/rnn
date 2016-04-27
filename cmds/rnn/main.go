package main

import (
	"flag"
	"github.com/armhold/piston"
	"io/ioutil"
	"log"
	"os"
	//"github.com/davecheney/profile"
)

var (
	cpFile    string
	inputFile string
	maxIter   int
)

func init() {
	flag.StringVar(&cpFile, "cp", "rnn.tmp", "specify checkpoint file")
	flag.StringVar(&inputFile, "i", "input.txt", "specify input training file")
	flag.IntVar(&maxIter, "m", 1000000, "max iterations")
	flag.Parse()
}

func main() {
	//defer profile.Start(profile.CPUProfile).Stop()

	inputBytes, err := ioutil.ReadFile(inputFile)
	if err != nil {
		log.Fatalf("error reading input training file: %s", err)
	}

	input := string(inputBytes)

	var rnn *piston.RNN

	// if there's an existing checkpoint file, restore from last checkpoint
	if _, err := os.Stat(cpFile); !os.IsNotExist(err) {
		rnn, err = piston.LoadFrom(cpFile)
		if err != nil {
			log.Fatalf("unable to restore RNN from checkpoint file: %s", err)
		}
	} else {
		rnn = piston.NewRNN(input, cpFile)
	}

	rnn.Run(maxIter)
}
