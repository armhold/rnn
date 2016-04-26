package main

import (
	"github.com/armhold/piston"
	"io/ioutil"
	"log"
)


func main() {
	inputBytes, err := ioutil.ReadFile("input.txt")
	if err != nil {
		log.Fatal(err)
	}

	input := string(inputBytes)
	n := piston.NewNetwork(input)
	n.Run()
}
