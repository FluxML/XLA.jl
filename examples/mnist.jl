using Flux, Optimisers, XLA
using Flux.Data.MNIST
using Flux: onehotbatch, crossentropy

images = reduce(hcat, [vec(Float64.(im)) for im in MNIST.images()])
labels = Float64.(onehotbatch(MNIST.labels(), 0:9))

loss(m, x, y) = logitcrossentropy(m(x), y)

opt = Optimisers.Descent(0.01)

function step(m)
  m̄, = gradient(m) do m
    @show crossentropy(m(images), labels)
  end
  return opt(m, m̄)
end

m = Chain(Dense(28^2, 32, relu), Dense(32, 10), softmax) |> f64

xstep = xla(step)

for i = 1:100
  global m = xstep(m)
end
