using Flux, XLA, Test

f(x) = gradient(x -> sum(x.*x), x)

@test collect(xla(f)([1, 2, 3])[1]) == [2, 4, 6]
