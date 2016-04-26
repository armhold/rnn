package piston

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"io/ioutil"
	"log"
)

// Implements GobDecoder. This is necessary because Network contains several unexported fields.
// It would be easier to simply export them by changing to uppercase, but for comparison purposes,
// I wanted to keep the field names the same between Go and the original Python code.
func (n *RNN) GobEncode() ([]byte, error) {
	var b bytes.Buffer
	encoder := gob.NewEncoder(&b)

	var err error

	encode := func(data interface{}) {
		// no-op if we've already seen an err
		if err == nil {
			err = encoder.Encode(data)
		}
	}

	encode(n.Wxh)
	encode(n.Whh)
	encode(n.Why)
	encode(n.bh)
	encode(n.by)
	encode(n.data)
	encode(n.charToIndex)
	encode(n.indexToChar)
	encode(n.VocabSize)

	return b.Bytes(), err
}

// Implement GoDecoder
func (n *RNN) GobDecode(data []byte) error {
	b := bytes.NewBuffer(data)
	decoder := gob.NewDecoder(b)

	var err error

	decode := func(data interface{}) {
		// no-op if we've already seen an err
		if err == nil {
			err = decoder.Decode(data)
		}
	}

	decode(&n.Wxh)
	decode(&n.Whh)
	decode(&n.Why)
	decode(&n.bh)
	decode(&n.by)
	decode(&n.data)
	decode(&n.charToIndex)
	decode(&n.indexToChar)
	decode(&n.VocabSize)

	return err
}

func (n *RNN) SaveTo(filePath string) error {
	log.Printf("Saving RNN to %s...", filePath)

	buf := new(bytes.Buffer)
	encoder := gob.NewEncoder(buf)

	err := encoder.Encode(n)
	if err != nil {
		return fmt.Errorf("error encoding network: %s", err)
	}

	err = ioutil.WriteFile(filePath, buf.Bytes(), 0644)
	if err != nil {
		return fmt.Errorf("error writing RNN to file: %s: %s", filePath, err)
	}

	return nil
}

func LoadFrom(filePath string) (*RNN, error) {
	log.Printf("Loading RNN from %s...", filePath)

	b, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("error reading RNN checkpoint file: %s", err)
	}

	decoder := gob.NewDecoder(bytes.NewBuffer(b))

	var result RNN
	err = decoder.Decode(&result)
	if err != nil {
		return nil, fmt.Errorf("error decoding RNN checkpoint file: %s", err)
	}

	result.checkpointFile = filePath

	return &result, nil
}
