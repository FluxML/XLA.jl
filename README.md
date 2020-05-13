# XLA

[![Build Status](https://travis-ci.org/FluxML/XLA.jl.svg?branch=master)](https://travis-ci.org/FluxML/XLA.jl)

```julia
] add IRTools#master
] add https://github.com/MikeInnes/Mjolnir.jl
] add https://github.com/FluxML/XLA.jl
```

Compile your Julia code to XLA. This package is part of the [Flux](https://github.com/FluxML/Flux.jl) ML ecosystem and is designed to work well with its other packages, including the [Zygote](https://github.com/FluxML/Zygote.jl) automatic differentiation engine.

**This project is in early alpha.** You can see some capability demos below, or some larger examples in the [examples folder](/examples/). You'll want to use the project/manifest in that directory as the examples currently depend on some development versions.

## Supported Features

Convert a Julia function to XLA:

```julia
julia> using Flux, XLA

julia> xadd = xla(+);

julia> xadd(2, 2.5)
2020-05-07 11:59:19.973581: I external/org_tensorflow/tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7ffe8239e680 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-07 11:59:19.973603: I external/org_tensorflow/tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
4.5

julia> m = Chain(Dense(10, 5, relu), Dense(5, 2), softmax) |> f64;

julia> xm = xla(m);

julia> xm(rand(10))
2-element XLA.XArray{Float64,1}:
 0.499667314539429
 0.5003326854605711
```

Hello world:

```julia
julia> function hello()
         if isxla()
           println("Hello from XLA!")
         else
           println("Not running XLA :(")
         end
       end

julia> hello()
Not running XLA :(

julia> xla(hello)()
Hello from XLA!
```

Generic functions and types:

```julia
julia> square = xla(x -> @show x^2);

julia> square(5)
x ^ 2 = 25
25

julia> square(1+2im)
x ^ 2 = -3 + 4im
-3 + 4im
```

If/else and data structures (loop support is on the roadmap):

```julia
julia> function updatezero!(env)
         if env[:x] < 0
           env[:x] = 0
         end
       end

julia> function myrelu(x)
         env = Dict()
         env[:x] = x
         updatezero!(env)
         return env[:x]
       end

julia> xrelu = xla(myrelu);

julia> xrelu(5), xrelu(-5)
(5, 0)
```

Take a gradient:

```julia
julia> W = randn(2, 3);

julia> f(x) = gradient((W, x) -> sum(W*x), W, x);

julia> W̄, x̄ = xla(f)(rand(3))
([-0.8832944966219527, 2.7865176392411346, 0.8122694890142825], [-0.8832944966219527, 2.7865176392411346, 0.8122694890142825])

julia> typeof(ans)
Tuple{XLA.XArray{Float64,1},Array{Float64,1}}
```

(Why is the gradient `x̄` an `Array`, and not an `XArray`? In fact `x̄` is a
constant regardless of the input `x`, so the gradient is computed at compile
time and never goes near XLA. If not for the computation of `W̄` the XLA code
would be a no-op.)

<details>

```julia
julia> f(x) = gradient(x -> sum(W*x), x);

julia> XLA.@code_xla f(rand(3))
1: (%1 :: Float64[3])
  %2 = ([-1.4783050895216538, -0.317112271139274, -0.32011307414342466],)
  return %2
```

</details>

We also want to support mutating array operations, in future.

## Under the Hood

See the results of type inference:

<details>

```julia
julia> XLA.@code_typed softmax([1, 2, 3])
1: (%1 :: const(softmax), %2 :: Mjolnir.Shape{Array{Int64,1}}((3,)))
  %3 =
    1: (%1 :: const(max), %2 :: Int64, %3 :: Int64)
      %4 = (max)(%2, %3) :: Int64
      return %4
  %4 = (Mjolnir.KwFunc{typeof(mapreduce)}())((dims = 1,), mapreduce, identity, %3, %2) :: Int64
  %5 =
    1: (%1 :: const(-), %2 :: Int64, %3 :: Int64)
      %4 = (-)(%2, %3) :: Int64
      return %4
  %6 = (broadcast)(%5, %2, %4) :: Mjolnir.Shape{Array{Int64,1}}((3,))
  %7 =
    1: (%1 :: const(exp), %2 :: Int64)
      %3 = (Float64)(%2) :: Float64
      %4 = (exp)(%3) :: Float64
      return %4
  %8 = (broadcast)(%7, %6) :: Mjolnir.Shape{Array{Float64,1}}((3,))
  %9 =
    1: (%1 :: const(add_sum), %2 :: Float64, %3 :: Float64)
      %4 = (+)(%2, %3) :: Float64
      return %4
  %10 = (Mjolnir.KwFunc{typeof(mapreduce)}())((dims = 1,), mapreduce, identity, %9, %8) :: Float64
  %11 =
    1: (%1 :: const(/), %2 :: Float64, %3 :: Float64)
      %4 = (/)(%2, %3) :: Float64
      return %4
  %12 = (broadcast)(%11, %8, %10) :: Mjolnir.Shape{Array{Float64,1}}((3,))
  return %12
```

</details>

See the final XLA code:

<details>

```julia
julia> @code_xla softmax([1, 2, 3])
1: (%1 :: Int64[3])
  %2 =
    1: (%2 :: Int64, %3 :: Int64)
      %4 = (XLA.Max())(%2, %3) :: Int64
      return %4
  %3 = (XLA.Reduce(1))(%2, %1, 0)
  %4 =
    1: (%2 :: Int64, %3 :: Int64)
      %4 = (XLA.Sub())(%2, %3) :: Int64
      return %4
  %5 = (XLA.Map())(%4, %1, %3)
  %6 =
    1: (%2 :: Int64)
      %3 = (XLA.ConvertElementType(Float64))(%2) :: Float64
      %4 = (XLA.Exp())(%3) :: Float64
      return %4
  %7 = (XLA.Map())(%6, %5)
  %8 =
    1: (%2 :: Float64, %3 :: Float64)
      %4 = (XLA.Add())(%2, %3) :: Float64
      return %4
  %9 = (XLA.Reduce(1))(%8, %7, 0.0)
  %10 =
    1: (%2 :: Float64, %3 :: Float64)
      %4 = (XLA.Div())(%2, %3) :: Float64
      return %4
  %11 = (XLA.Map())(%10, %7, %9)
  return %11
```

</details>

XLA's internal text representation, HLO text:

<details>

```julia
julia> @code_hlo softmax([1, 2, 3])
HloModule name__44.31

name__45.3 {
  parameter.4 = s64[]invalid{} parameter(0)
  parameter.5 = s64[]invalid{} parameter(1)
  ROOT maximum.6 = s64[]invalid{} maximum(parameter.4, parameter.5)
}

name__46.8 {
  parameter.9 = s64[]invalid{} parameter(0)
  parameter.10 = s64[]invalid{} parameter(1)
  ROOT subtract.11 = s64[]invalid{} subtract(parameter.9, parameter.10)
}

name__47.14 {
  parameter.15 = s64[]invalid{} parameter(0)
  convert.16 = f64[]invalid{} convert(parameter.15)
  ROOT exponential.17 = f64[]invalid{} exponential(convert.16)
}

name__48.20 {
  parameter.21 = f64[]invalid{} parameter(0)
  parameter.22 = f64[]invalid{} parameter(1)
  ROOT add.23 = f64[]invalid{} add(parameter.21, parameter.22)
}

name__49.25 {
  parameter.26 = f64[]invalid{} parameter(0)
  parameter.27 = f64[]invalid{} parameter(1)
  ROOT divide.28 = f64[]invalid{} divide(parameter.26, parameter.27)
}

ENTRY name__44.31 {
  parameter.1 = s64[3] parameter(0)
  constant.2 = s64[] constant(0)
  reduce.7 = s64[] reduce(parameter.1, constant.2), dimensions={0}, to_apply=name__45.3
  broadcast.12 = s64[3]{0} broadcast(reduce.7), dimensions={}
  map.13 = s64[3]{0} map(parameter.1, broadcast.12), dimensions={0}, to_apply=name__46.8
  map.18 = f64[3]{0} map(map.13), dimensions={0}, to_apply=name__47.14
  constant.19 = f64[] constant(0)
  reduce.24 = f64[] reduce(map.18, constant.19), dimensions={0}, to_apply=name__48.20
  broadcast.29 = f64[3]{0} broadcast(reduce.24), dimensions={}
  ROOT map.30 = f64[3]{0} map(map.18, broadcast.29), dimensions={0}, to_apply=name__49.25
}
```

</details>

You may want to start with simpler examples like `@code_xla 1+2.0` or
`@code_xla (1+2im)*(3+4im)`.

## Limitations & Notes

XLA is a specialised backend with limitations, primarily in terms of support for dynamic memory allocation – so we don't expect to be able to support all Julia code. Vectorised array code without too much fanciness (including Flux models) should work well; just don't expect to point XLA at any huge Julia codebase and have it work out of the box.

Error handling is so-so right now. If you run into errors, please do open issues; we'll either support your use case or at least add better diagnostics to explain why the code can't be compiled.

XLA reuses [JAX's](https://github.com/google/jax) build of XLA via `pip`. A CPU-only build is installed by default; if you want GPU support you can [use your own python](https://github.com/JuliaPy/PyCall.jl#specifying-the-python-version) and install the GPU-enabled jaxlib as per the [jax docs](https://github.com/google/jax#pip-installation). The currently supported jaxlib version is specified in [build.jl](deps/build.jl).
