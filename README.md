# rnn
translating a Recurrent Neural Network from Python to Go


##
This is more or less a straight translation of Andrej Karpathy's Recurrent Neural Network
[Python code](https://gist.github.com/karpathy/d4dee566867f8291f086) to Go.

See http://karpathy.github.io/2015/05/21/rnn-effectiveness for more information.

I have attempted to translate it faithfully, even down to the level of preserving variable names
(many of which are somewhat... terse).

The one major change I did introduce is code for checkpointing the model; this is primarily implemented in
persistence.go.

Any errors here are my own, and not Karpathy's. Corrections welcome.
