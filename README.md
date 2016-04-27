# RNN - a Recurrent Neural Network in Go

This is more or less a straight translation of Andrej Karpathy's
[Recurrent Neural Network code](https://gist.github.com/karpathy/d4dee566867f8291f086) from Python to Go. See
http://karpathy.github.io/2015/05/21/rnn-effectiveness for more information.

I have attempted to translate it faithfully, even down to the level of preserving variable names
(many of which are somewhat... terse) and his comment text. The one major change I did introduce is
code for checkpointing the model; this is primarily implemented in persistence.go.

Any errors here are my own, and not Karpathy's. Corrections welcome.


## How to use it

1. `$ go get github.com/armhold/rnn/...`
1. `$ cd $GOPATH/src/github.com/armhold/rnn`
1. `$ rnn -i input.txt`

This will run the network on a small corpus of Shakespeare text. After a few thousand iterations,
you should start seeing output that looks superficially like a Shakespeare play.
