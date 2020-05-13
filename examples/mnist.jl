using Flux, Optimisers, XLA
using Flux.Data.MNIST
using Flux: onehotbatch, crossentropy

images = reduce(hcat, [vec(Float64.(im)) for im in MNIST.images()])
labels = onehotbatch(MNIST.labels(), 0:9)

loss(m, x, y) = logitcrossentropy(m(x), y)

opt = Optimisers.Descent(1e-3)

function step(m, x, y)
  m̄, = gradient(m) do m
    @show crossentropy(m(x), y)
  end
  return opt(m, m̄)
end

x = images[:,1]
y = Float64.(collect(labels[:,1]))

m = Chain(Dense(28^2, 32, relu), Dense(32, 10), softmax) |> f64

xstep = xla(step)

m = xstep(m, x, y)
