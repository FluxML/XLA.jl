# XLATools

XLATools provides access to XLA and the XRT runtime, including the ability to build and compile XLA computations using the [IRTools](https://github.com/MikeInnes/IRTools.jl) format.

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

XLATools reuses [JAX's](https://github.com/google/jax) build of XLA via `pip`. A CPU-only build is installed by default; if you want GPU support you can [use your own python](https://github.com/JuliaPy/PyCall.jl#specifying-the-python-version) and install the GPU-enabled jaxlib as per the jax docs.
