using Flux, XLA, Test

x, y = randn(2), randn(1, 2)
@test collect(xla(*)(x, y)) ≈ x*y

f(x) = gradient(x -> sum(x.*x), x)

@test collect(xla(f)([1, 2, 3])[1]) == [2, 4, 6]

f(x) = gradient(x -> sum(x.+x), x)

@test collect(xla(f)([1, 2, 3])[1]) == [2, 2, 2]

W = rand(2, 3)

f(x) = gradient(x -> sum(W*x), x)

@test collect(xla(f)([1.0, 2, 3])[1]) == f([1.0, 2, 3])[1]

@test xla(sin'')(0.5) == -sin(0.5)
