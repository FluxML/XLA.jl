const!(builder, x::Union{AbstractArray,XScalar}) = builder.Constant(x)
const!(builder, x::Tuple{}) = builder.Tuple()

struct Lambda
  vars::Vector{Any}
  func::IR
end

function run(op, args...)
  args = buffer.(args)
  b = xlaclient.ComputationBuilder("")
  arg_ops = [b.ParameterWithShape(arg.shape()) for arg in args]
  build!(b, op, arg_ops...)
  b.Build().Compile().Execute(args, false)[1] |> wrapvalue
end

for op in :[Neg, Sign, Floor, Ceil, Round, Exp, Log, Expm1, Log1p, Tanh,
            Sin, Cos, Lgamma, Digamma, Erf, Erfc, ErfInv, Sqrt, Rsqrt, Not].args
  @eval begin
    struct $op end
    build!(builder, ::$op, x) = getproperty(builder, $(Expr(:quote, op)))(x)
    (::$op)(x) = run($op(), x)
  end
end

for op in :[Atan2, Pow, And, Or, Xor, Add, Sub, Mul, SafeMul, Div, Rem, Dot,
            Max, Min, ShiftLeft, ShiftRightArithmetic, ShiftRightLogical,
            Gt, Ge, Lt, Le].args
  @eval begin
    struct $op end
    build!(builder, ::$op, x, y) = getproperty(builder, $(Expr(:quote, op)))(x, y)
    (::$op)(x, y) = run($op(), x, y)
  end
end

alldims(builder, xs) = [0:ndims(shapeof(builder, xs))-1;]

struct Map end

function build!(builder, ::Map, f, args...)
  settypes!(builder, f, args..., with = eltype)
  builder.Map(args, build(f), alldims(builder, args[1]))
end

struct Reduce
  dims
end

function build!(builder, op::Reduce, f, xs, init)
  settypes!(builder, f, xs, xs, with = eltype)
  dims = op.dims == (:) ? alldims(builder, xs) : sort([op.dims...].-1)
  builder.Reduce(xs, init, build(f), dims)
end

struct XTuple end

build!(builder, ::XTuple, xs...) = builder.Tuple(xs...)

(op::XTuple)(xs...) = run(op, xs...)

struct GetTupleElement
  idx::Int
end

build!(builder, op::GetTupleElement, x) = builder.GetTupleElement(x, op.idx)

(op::GetTupleElement)(xs...) = run(op, xs...)

struct ConvertElementType
  to::Type{<:XScalar}
end

build!(builder, op::ConvertElementType, x) =
  builder.ConvertElementType(x, primitivetype(op.to))

(op::ConvertElementType)(xs...) = run(op, xs...)

struct Conditional end

function build!(builder, ::Conditional, pred,
                true_operand, true_computation,
                false_operand, false_computation)
  settypes!(builder, true_computation, true_operand)
  settypes!(builder, false_computation, false_operand)
  builder.Conditional(pred, true_operand, build(true_computation),
                            false_operand, build(false_computation))
end

struct While end

function build!(builder, ::While, condition, body, init)
  settypes!(builder, condition, init)
  settypes!(builder, body, init)
  builder.While(build(condition), build(body), init)
end
