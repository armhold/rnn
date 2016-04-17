package piston

import (
	"fmt"
	"os/exec"
	"log"
)

func init() {

}

type Piston struct {

}

func (p *Piston) Compile(dirname, filename string) error {
	cmd := exec.Command("go", "tool", "compile", filename)
	cmd.Dir = dirname
	out, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("%s", out)

		err = fmt.Errorf("go tool compile %s failed: %s", filename, err)
	} else {
		log.Printf("successfully compiled %s: %s", filename, out)
	}

	return err
}
