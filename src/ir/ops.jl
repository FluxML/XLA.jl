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

struct DynamicSlice
  size
end

build!(builder, slice::DynamicSlice, x, start...) =
  xlaclient.ops.DynamicSlice(x, start, slice.size)

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
  dims::Vector{Int}
  size::Vector{Int}
end

build!(builder, op::Reshape, x) =
  xlaclient.ops.Reshape(x, op.dims, op.size)

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
