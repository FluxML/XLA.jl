# XLATools

XLATools provides access to XLA and the XRT runtime, including the ability to build and compile XLA computations using the [IRTools](https://github.com/MikeInnes/IRTools.jl) format.

```julia
] add IRTools#master
] add https://github.com/MikeInnes/XLATools.jl
```

Run XLA ops directly (slow but useful for testing/debugging):

```julia
julia> using XLATools

julia> XLATools.Mul()(2, 3)
2019-09-17 14:29:42.711650: I external/org_tensorflow/tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3600000000 Hz
...
6

julia> t = XLATools.XTuple()(5, 6)
(5, 6)

julia> XLATools.GetTupleElement(0)(t)
5

julia> XLATools.Add()([1, 2], [3, 4])
2-element XLATools.XArray{Int64,1}:
 4
 6
```

Ops are named as in XLA proper (see the [reference](https://www.tensorflow.org/xla/operation_semantics)), except for `XTuple`. All ops are parameterised, which means you construct them with any options before invoking them.

Build and invoke a simple computation, the polynomial `3x^2 + 2x + 1`:

```julia
julia> using XLATools: Mul, Add, Pow, compile

julia> using IRTools: IR, argument!, xcall

julia> ir = IR();

julia> x = argument!(ir, Int);

julia> poly = xcall(Add(), xcall(Mul(), 3, xcall(Pow(), x, 2)),
                           xcall(Add(), xcall(Mul(), 2, x), 1))
:((Add())((Mul())(3, (Pow())(%1, 2)), (Add())((Mul())(2, %1), 1)))

julia> push!(ir, poly);

julia> ir
1: (%1 :: Int64)
  %2 = (Pow())(%1, 2)
  %3 = (Mul())(3, %2)
  %4 = (Mul())(2, %1)
  %5 = (Add())(%4, 1)
  %6 = (Add())(%3, %5)

julia> f = compile(ir);

julia> f(5)
86
```

Writing IR by hand is a bit tedious; we can actually use Julia to do most of the work for us.

```julia
julia> @eval relu = x -> $(Gt())(x, 0) ? x : 0
#15 (generic function with 1 method)

julia> ir = @code_ir relu(1)
1: (%1, %2)
  %3 = (Gt())(%2, 0)
  br 2 unless %3
  return %2
2:
  return 0
```

Compile it:

```julia
julia> f((), 1), f((), -1)
(1, 0)

julia> IRTools.argtypes(ir)[:] = [(), Int]
2-element Array{Any,1}:
 ()   
 Int64

julia> f = compile(ir)
#10 (generic function with 1 method)
```

If you're familiar with XLA you might notice that we're not using its "functional" control flow here, but instead normal SSA branches. The idea is to abstract over XLA's _somewhat idiosyncratic_ `Conditional` and `While` with something more convenient, that gets lowered to those calls when compiling. It's easy to see what the native equivalent looks like:

```julia
julia> XLATools.controlflow(ir)
1: (%1 :: (), %2 :: Int64)
  %3 = (Gt())(%2, 0)
  %4 = (XLATools.Not())(%3)
  %5 =
    1: (%1)
      %2 = (XTuple())(0)
  %6 =
    1: (%1)
  %7 = (XTuple())()
  %8 = (XTuple())(%2)
  %9 = (Conditional())(%4, %7, %5, %8, %6)
  %10 = (GetTupleElement(0))(%9)
```

Right now only `Conditional`s are supported, but support for `While` is planned.

XLATools' op support is not yet exhaustive, but new ops are easy to add. For example, the definition for `XTuple` is [only three lines](https://github.com/MikeInnes/XLATools.jl/blob/06e3fccdb2e714aab4b112f16da6ceae38e871ed/src/ops.jl#L36-L40).

XLATools reuses [JAX's](https://github.com/google/jax) build of XLA via `pip`. A CPU-only build is installed by default; if you want GPU support you can [use your own python](https://github.com/JuliaPy/PyCall.jl#specifying-the-python-version) and install the GPU-enabled jaxlib as per the jax docs.
