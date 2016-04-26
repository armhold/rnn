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
func (r *RNN) GobEncode() ([]byte, error) {
	var b bytes.Buffer
	encoder := gob.NewEncoder(&b)

	var err error

	encode := func(data interface{}) {
		// no-op if we've already seen an err
		if err == nil {
			err = encoder.Encode(data)
		}
	}

	encode(r.Wxh)
	encode(r.Whh)
	encode(r.Why)
	encode(r.bh)
	encode(r.by)
	encode(&r.hprev)
	encode(&r.mWxh)
	encode(&r.mWhh)
	encode(&r.mWhy)
	encode(&r.mbh)
	encode(&r.mby)
	encode(r.data)
	encode(r.charToIndex)
	encode(r.indexToChar)
	encode(r.VocabSize)
	encode(r.n)
	encode(r.loss)
	encode(r.smooth_loss)

	return b.Bytes(), err
}

// Implement GoDecoder
func (r *RNN) GobDecode(data []byte) error {
	b := bytes.NewBuffer(data)
	decoder := gob.NewDecoder(b)

	var err error

	decode := func(data interface{}) {
		// no-op if we've already seen an err
		if err == nil {
			err = decoder.Decode(data)
		}
	}

	decode(&r.Wxh)
	decode(&r.Whh)
	decode(&r.Why)
	decode(&r.bh)
	decode(&r.by)
	decode(&r.hprev)
	decode(&r.mWxh)
	decode(&r.mWhh)
	decode(&r.mWhy)
	decode(&r.mbh)
	decode(&r.mby)
	decode(&r.data)
	decode(&r.charToIndex)
	decode(&r.indexToChar)
	decode(&r.VocabSize)
	decode(&r.n)
	decode(&r.loss)
	decode(&r.smooth_loss)

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
