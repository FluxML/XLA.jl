const!(builder, x::Union{AbstractArray,XScalar,Bool}) = xlaclient.ops.Constant(builder, x)
const!(builder, xs::Tuple) = build!(builder, XTuple(), const!.((builder,), xs)...)

struct Lambda
  vars::Vector{Any}
  func::IR
end

for op in :[Neg, Sign, Floor, Ceil, Round, Exp, Log, Expm1, Log1p, Tanh,
            Sin, Cos, Lgamma, Digamma, Erf, Erfc, ErfInv, Sqrt, Rsqrt, Not].args
  @eval begin
    struct $op end
    build!(builder, ::$op, x) = xlaclient.ops.$op(x)
  end
end

for op in :[Atan2, Pow, And, Or, Xor, Add, Sub, Mul, SafeMul, Div, Rem, Dot,
            Max, Min, ShiftLeft, ShiftRightArithmetic, ShiftRightLogical,
            Gt, Ge, Lt, Le].args
  @eval begin
    struct $op end
    build!(builder, ::$op, x, y) = xlaclient.ops.$op(x, y)
  end
end

struct Rev
  dims
end

build!(builder, op::Rev, x) =
  xlaclient.ops.Rev(x, op.dims .- 1)

struct Conv
  strides
  padding
  lhs_dilation
  rhs_dilation
end

function build!(builder, op::Conv, x, w)
  dims = xlaclient.make_convolution_dimension_numbers(("WHCN", "WHIO", "WHCN"), 2)
  xlaclient.ops.ConvGeneralDilated(x, w, op.strides, op.padding, op.lhs_dilation, op.rhs_dilation, dims)
end

struct DynamicSlice
  size
end

function build!(builder, slice::DynamicSlice, x, start...)
  one = const!(builder, 1)
  start = map(i -> xlaclient.ops.Sub(i, one), start)
  xlaclient.ops.DynamicSlice(x, start, slice.size)
end

struct BroadcastInDim
  size
  map
end

build!(builder, bc::BroadcastInDim, x) = xlaclient.ops.BroadcastInDim(x, bc.size, bc.map.-1)

alldims(builder, xs) = [0:ndims(shapeof(builder, xs))-1;]

struct Map end

function build!(builder, ::Map, f, args...)
  settypes!(builder, f, args..., with = eltype)
  xlaclient.ops.Map(builder, args, build(f), alldims(builder, args[1]))
end

struct Reduce
  dims
end

function build!(builder, op::Reduce, f, xs, init)
  settypes!(builder, f, xs, xs, with = eltype)
  dims = op.dims == (:) ? alldims(builder, xs) : sort([op.dims...].-1)
  xlaclient.ops.Reduce(builder, [xs], [init], build(f), dims)
end

struct ReduceWindow
  window
  stride
  base_dilation
  window_dilation
  padding
end

ReduceWindow(window) = ReduceWindow(window, window, one.(window), one.(window), map(_ -> (0, 0), window))

function build!(builder, op::ReduceWindow, f, xs, init)
  xlaclient.ops.ReduceWindowWithGeneralPadding(
    xs, init, build(f), op.window, op.stride,
    op.base_dilation, op.window_dilation, op.padding)
end

struct XTuple end

build!(builder, ::XTuple, xs...) = xlaclient.ops.Tuple(builder, xs)

struct GetTupleElement
  idx::Int
end

build!(builder, op::GetTupleElement, x) = xlaclient.ops.GetTupleElement(x, op.idx)

struct ConvertElementType
  to::Type{<:XScalar}
end

build!(builder, op::ConvertElementType, x) =
  xlaclient.ops.ConvertElementType(x, primitivetype(op.to))

struct Reshape
  dims
  size
end

build!(builder, op::Reshape, x) =
  xlaclient.ops.Reshape(x, op.dims.-1, op.size)

struct Conditional end

function build!(builder, ::Conditional, pred,
                true_operand, true_computation,
                false_operand, false_computation)
  settypes!(builder, true_computation, true_operand)
  settypes!(builder, false_computation, false_operand)
  xlaclient.ops.Conditional(pred, true_operand, build(true_computation),
                            false_operand, build(false_computation))
end

struct While end

function build!(builder, ::While, condition, body, init)
  settypes!(builder, condition, init)
  settypes!(builder, body, init)
  xlaclient.ops.While(build(condition), build(body), init)
end
