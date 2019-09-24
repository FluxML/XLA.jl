const!(builder, x::Union{AbstractArray,XScalar}) = builder.Constant(x)
const!(builder, x::Tuple{}) = builder.Tuple()

struct Lambda
  vars::Vector{Any}
  func::IR
end

function run(op, args...)
  args = xlaclient.Buffer.from_pyval.(args)
  b = xlaclient.ComputationBuilder("")
  arg_ops = [b.ParameterWithShape(arg.shape()) for arg in args]
  build!(b, op, arg_ops...)
  b.Build().Compile().Execute(args) |> wrapvalue
end

for op in :[Neg, Sign, Floor, Ceil, Round, Exp, Log, Expm1, Log1p, Tanh,
            Sin, Cos, Lgamma, Digamma, Erf, Erfc, ErfInv, Sqrt, Rsqrt, Not].args
  @eval begin
    struct $op end
    build!(builder, ::$op, x) = getproperty(builder, $(Expr(:quote, op)))(x)
    (::$op)(x) = run($op(), x)
  end
end

for op in :[Atan2, Pow, And, Or, Xor, Add, Sub, Mul, SafeMul, Div, Rem,
            Max, Min, ShiftLeft, ShiftRightArithmetic, ShiftRightLogical,
            Gt, Ge, Lt, Le].args
  @eval begin
    struct $op end
    build!(builder, ::$op, x, y) = getproperty(builder, $(Expr(:quote, op)))(x, y)
    (::$op)(x, y) = run($op(), x, y)
  end
end

struct XTuple end

build!(builder, ::XTuple, xs...) = builder.Tuple(xs...)

(op::XTuple)(xs...) = run(op, xs...)

struct GetTupleElement
  idx::Int
end

build!(builder, op::GetTupleElement, x) = builder.GetTupleElement(x, op.idx)

(op::GetTupleElement)(xs...) = run(op, xs...)

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

build!(builder, ::While, condition, body, init) =
  builder.While(build(condition), build(body), init)
