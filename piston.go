package piston

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/ioutil"
	"log"
	"os/exec"
)

var (
	scratchDir = "scratch"
)

func init() {

}

type Piston struct {
}

func (p *Piston) Parse(src string) {
	// Create the AST by parsing src.
	fset := token.NewFileSet() // positions are relative to fset
	f, err := parser.ParseFile(fset, "", src, 0)
	if err != nil {
		panic(err)
	}

	// Print the AST.
	ast.Print(fset, f)
}

func (p *Piston) SaveAndCompile(filename, content string) error {
	path := scratchDir + "/" + filename
	b := []byte(content)
	err := ioutil.WriteFile(path, b, 0644)
	if err != nil {
		return fmt.Errorf("error writing file: %s", err)
	}

	err = p.Compile(scratchDir, filename)

	return err
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
