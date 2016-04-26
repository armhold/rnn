package piston

import (
	"encoding/gob"
	"bytes"
	"fmt"
	"io/ioutil"
)

// Implements GobDecoder. This is necessary because Network contains several unexported fields.
// It would be easier to simply export them by changing to uppercase, but for comparison purposes,
// I wanted to keep the field names the same between Go and the original Python code.
func (n *Network) GobEncode() ([]byte, error) {
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
func (n *Network) GobDecode(data []byte) error {
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

func (n *Network) Save(filePath string) error {
	buf := new(bytes.Buffer)
	encoder := gob.NewEncoder(buf)

	err := encoder.Encode(n)
	if err != nil {
		return fmt.Errorf("error encoding network: %s", err)
	}

	err = ioutil.WriteFile(filePath, buf.Bytes(), 0644)
	if err != nil {
		return fmt.Errorf("error writing network to file: %s", err)
	}

	return nil
}

func LoadNetwork(filePath string) (*Network, error) {
	b, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("error reading network file: %s", err)
	}

	decoder := gob.NewDecoder(bytes.NewBuffer(b))

	var result Network
	err = decoder.Decode(&result)
	if err != nil {
		return nil, fmt.Errorf("error decoding network: %s", err)
	}

	return &result, nil
}
