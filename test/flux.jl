using Flux, XLA, Test

x, y = randn(2), randn(1, 2)
@test collect(xla(*)(x, y)) ≈ x*y

f(x) = gradient(x -> sum(x.*x), x)

@test collect(xla(f)([1, 2, 3])[1]) == [2, 4, 6]
