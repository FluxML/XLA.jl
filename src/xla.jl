import PyCall: PyObject

const XScalar = Union{Float64,Float32,Float16,Int64,Int32,Int16}

const julia2numpy = Dict(
  Float64 => "float64",
  Float32 => "float32",
  Int64   => "int64")

const numpy2julia = Dict(v => k for (k, v) in julia2numpy)

numpytype(T) = xlaclient.np.dtype(julia2numpy[T])
juliatype(T) = numpy2julia[T.name]

struct Shape{T,N}
  dims::NTuple{N,Int}
end

Shape(T::Type{<:XScalar}, sh::NTuple{N,Integer}) where N = Shape{T,N}(sh)

Shape(x::AbstractArray) = Shape{eltype(x),ndims(x)}(size(x))

Base.eltype(::Shape{T}) where T = T
Base.ndims(::Shape{T,N}) where {T,N} = N

PyObject(sh::Shape) = xlaclient.Shape.array_shape(numpytype(eltype(sh)), sh.dims)

Base.show(io::IO, sh::Shape) = print(io, eltype(sh), sh.dims)

function Shape(sh::PyObject)
  T = juliatype(sh.numpy_dtype())
  size = sh.dimensions()
  Shape{T,length(size)}(size)
end

struct XArray{T,N} <: AbstractArray{T,N}
  buffer::PyObject
end

function setup_finaliser(x::XArray)
  delete = x.buffer.delete # work around a segfault on exit
  finalizer(buf -> ispynull(delete) || delete(), x.buffer)
end

function XArray(data::Array{<:XScalar})
  buffer = xlaclient.Buffer.from_pyval(data)
  x = XArray{eltype(data),ndims(data)}(buffer)
  setup_finaliser(x)
  return x
end

function XArray(buf::PyObject, own = true)
  sh = Shape(buf.shape())
  x = XArray{eltype(sh),ndims(sh)}(buf)
  own && setup_finaliser(x)
  return x
end

PyObject(x::XArray) = x.buffer

Base.size(x::XArray) = x.buffer.shape().dimensions()
Base.collect(x::XArray) = x.buffer.to_py()
Base.print_array(io::IO, x::XArray) = Base.print_array(io, collect(x))

xla(x::XArray) = x
xla(x::AbstractArray) = XArray(x)

# Operation Wrappers

function run_direct(op, args...)
  args = xla.(args)
  b = xlaclient.ComputationBuilder("")
  arg_ops = [b.ParameterWithShape(Shape(arg)) for arg in args]
  getproperty(b, op)(arg_ops...)
  b.Build().Compile().Execute(args) |> XArray
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

build!(builder, ::Conditional, true_operand, true_computation, false_operand, false_computation) =
  builder.Conditional(pred, true_operand, true_computation, false_operand, false_computation)

struct While end

build!(builder, ::While, condition, body, init) =
  builder.While(condition, body, init)

# IR Builder

function build(ir::IR)
  builder = xlaclient.ComputationBuilder("")
  env = Dict()
  resolve(x::Variable) = env[x]
  resolve(x) = x
  for (v, T) in zip(arguments(ir), argtypes(ir))
    env[v] = builder.ParameterWithShape(T::Shape)
  end
  for (v, st) in ir
    if isexpr(st.expr, :call)
      env[v] = build!(builder, resolve.(st.expr.args)...)
    else
      error("Invalid XLA expression $(st.expr)")
    end
  end
  return builder.Build()
end

function compile(ir::IR)
  comp = build(ir).Compile()
  return (xs...) -> XArray(comp.Execute(xla.(xs)))
end
