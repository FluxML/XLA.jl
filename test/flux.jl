using Flux, NNlib, XLA, Test

xresult(f, args...) = collect(xla(f)(args...))

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

x = rand(10)
m = Chain(Dense(10, 5), Dense(5, 2)) |> f64

f(x) = gradient(m -> sum(m(x)), m)[1]

xf = xla(f)

@test collect(xf(x).layers[1].W) ≈ f(x).layers[1].W

x = rand(10, 10, 5, 1)
w = rand(2, 2, 5, 3)
y = rand(size(conv(x, w))...)

@test xresult(conv, x, w) ≈ conv(x, w)

@test ∇conv_data(y, w, DenseConvDims(x, w)) ≈ xresult(y, w) do y, w
  ∇conv_data(y, w, DenseConvDims(x, w))
end

@test ∇conv_filter(x, y, DenseConvDims(x, w)) ≈ xresult(x, y) do x, y
  ∇conv_filter(x, y, DenseConvDims(x, w))
end

@test xresult(x -> maxpool(x, (2, 2)), x) ≈ maxpool(x, (2, 2))
