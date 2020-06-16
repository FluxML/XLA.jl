using Flux, Functors, Optimisers, XLA
using Flux.Data.MNIST
using Flux: onehotbatch, crossentropy

i = rand(784, 2)
t = rand(10, 2)

opt = Optimisers.ADAM(0.001, (0.9,0.99))

function step(m)
  m̄, = gradient(m) do m
    @show Flux.mse(m(i), t)
  end
  return opt2(m, m̄)
end

m = Chain(Dense(28^2, 32, relu), Dense(32, 10), softmax) |> f64

xstep = xla(step)

function train(m)
  for _ = 1:100
    m = xstep(m)
  end
  m
end

train(m)
