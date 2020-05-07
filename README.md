# XLA

[![Build Status](https://travis-ci.org/FluxML/XLATools.jl.svg?branch=master)](https://travis-ci.org/FluxML/XLATools.jl)

```julia
] add IRTools#master
] add https://github.com/MikeInnes/Mjolnir.jl
] add https://github.com/FluxML/XLATools.jl#next
```

Compile your Julia code to XLA. This package is part of the [Flux](https://github.com/FluxML/Flux.jl) ML ecosystem and is designed to work well with its other packages, including the [Zygote](https://github.com/FluxML/Zygote.jl) automatic differentiation engine.

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

We also want to support mutating array operations in future.

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

You may want to start with simpler examples like `@code_xla 1+2.0` or
`@code_xla (1+2im)*(3+4im)`.

## Limitations & Notes

XLA is a specialised backend with limitations, primarily in terms of support for dynamic memory allocation – so we don't expect to be able to support all Julia code. Vectorised array code without too much fanciness (including Flux models) should work well; just don't expect to point XLA at any huge Julia codebase and have it work out of the box.

Error handling is so-so right now. If you run into errors, please do open issues; we'll either support your use case or at least add better diagnostics to explain why the code can't be compiled.

XLA reuses [JAX's](https://github.com/google/jax) build of XLA via `pip`. A CPU-only build is installed by default; if you want GPU support you can [use your own python](https://github.com/JuliaPy/PyCall.jl#specifying-the-python-version) and install the GPU-enabled jaxlib as per the jax docs.
