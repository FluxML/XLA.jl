struct Lambda
  vars::Vector{Any}
  func::IR
end

function run_direct(op, args...)
  args = xla.(args)
  b = xlaclient.ComputationBuilder("")
  arg_ops = [b.ParameterWithShape(Shape(arg)) for arg in args]
  getproperty(b, op)(arg_ops...)
  b.Build().Compile().Execute(args) |> wrapvalue
end

for op in :[Neg, Sign, Floor, Ceil, Round, Exp, Log, Expm1, Log1p, Tanh,
    Sin, Cos, Lgamma, Digamma, Erf, Erfc, ErfInv, Sqrt, Rsqrt, Not].args
  @eval begin
    struct $op end
    build!(builder, ::$op, x) = getproperty(builder, $(Expr(:quote, op)))(x)
    (::$op)(x) = run_direct($(Expr(:quote, op)), x)
  end
end

for op in :[Atan2, Pow, And, Or, Xor, Add, Sub, Mul, SafeMul, Div, Rem,
    Max, Min, ShiftLeft, ShiftRightArithmetic, ShiftRightLogical].args
  @eval begin
    struct $op end
    build!(builder, ::$op, x, y) = getproperty(builder, $(Expr(:quote, op)))(x, y)
    (::$op)(x, y) = run_direct($(Expr(:quote, op)), x, y)
  end
end

struct Conditional end

build!(builder, ::Conditional, pred, true_operand, true_computation, false_operand, false_computation) =
  builder.Conditional(pred, true_operand, true_computation, false_operand, false_computation)

struct While end

build!(builder, ::While, condition, body, init) =
  builder.While(condition, body, init)
