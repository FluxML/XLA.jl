using Flux, Optimisers, XLA
using Flux.Data.MNIST
using Flux: onehotbatch, crossentropy
using Flux.Data: DataLoader

images = reduce(hcat, [vec(Float64.(im)) for im in MNIST.images()])
images = reshape(images, 28, 28, 1, :)
labels = Float64.(onehotbatch(MNIST.labels(), 0:9))

opt = Optimisers.Descent(0.1)

function step(m, x, y)
  m̄, = gradient(m) do m
    @show crossentropy(m(x), y)
  end
  return opt(m, m̄)
end

m = Chain(
  Conv((3, 3), 1=>16, pad=(1,1), relu),
  MaxPool((2,2)),
  Conv((3, 3), 16=>32, pad=(1,1), relu),
  MaxPool((2,2)),
  Conv((3, 3), 32=>32, pad=(1,1), relu),
  MaxPool((2,2)),
  flatten,
  Dense(288, 10), softmax) |> f64

xstep = xla(step)

for (x, y) in DataLoader((images, labels), batchsize = 100)
  global m = xstep(m, x, y)
end
