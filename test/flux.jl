using Flux, XLA, Test

f(x) = gradient(x -> sum(x), x)

@test xla(f)([1, 2, 3])[1] == [1, 1, 1]
