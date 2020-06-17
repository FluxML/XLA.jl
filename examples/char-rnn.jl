# NOTE: this demo is a wip, it does not use XLA yet.

using Flux, Optimisers, XLA
using Flux: onehot, crossentropy, chunk, batchseq
using Base.Iterators: partition

cd(@__DIR__)

isfile("input.txt") ||
  download("https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt","input.txt")

text = collect(String(read("input.txt")))

# an array of all unique characters
alphabet = [unique(text)..., '_']

text = map(ch -> onehot(ch, alphabet), text)
stop = onehot('_', alphabet)

N = length(alphabet)
seqlen = 50
nbatch = 50

# Partitioning the data as sequence of batches, which are then collected as array of batches
Xs = partition(batchseq(chunk(float.(text), nbatch), stop), seqlen)
Ys = partition(batchseq(chunk(float.(text[2:end]), nbatch), stop), seqlen)

function loss(m, xs, ys)
  loss = 0.
  for (x, y) in zip(xs, ys)
    loss += crossentropy(m(x), y)
  end
  return loss
end

function step(m, xs, ys)
  m̄, = gradient(m) do m
    loss(m, xs, ys)
  end
  return m̄
end

m = Chain(
  LSTM(N, 128),
  LSTM(128, 128),
  Dense(128, N),
  softmax) |> f64

step(m, first(Xs), first(Ys))
