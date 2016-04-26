package main

import (
	"flag"
	"github.com/armhold/piston"
	"io/ioutil"
	"log"
	"os"
)

var (
	cpFile    string
	inputFile string
)

func init() {
	flag.StringVar(&cpFile, "cp", "rnn.tmp", "specify checkpoint file")
	flag.StringVar(&inputFile, "i", "input.txt", "specify input training file")
	flag.Parse()
}

func main() {
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

	rnn.Run()
}
