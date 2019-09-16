import PyCall: PyObject

const XScalar = Union{Float64,Float32,Float16,Int64,Int32,Int16}

const julia2numpy = Dict(
  Float64 => "float64",
  Float32 => "float32",
  Int64   => "int64",
  Bool    => "bool")

const numpy2julia = Dict(v => k for (k, v) in julia2numpy)

numpytype(T) = xlaclient.np.dtype(julia2numpy[T])
juliatype(T) = numpy2julia[T.name]

# Shapes

struct Shape{T,N}
  dims::NTuple{N,Int}
end

Shape(T::Type{<:XScalar}, sh::NTuple{N,Integer}) where N = Shape{T,N}(sh)

shapeof(x::AbstractArray) = Shape{eltype(x),ndims(x)}(size(x))

Base.eltype(::Shape{T}) where T = T
Base.ndims(::Shape{T,N}) where {T,N} = N

PyObject(sh::Shape) = xlaclient.Shape.array_shape(numpytype(eltype(sh)), sh.dims)
pyshape(sh::Shape) = PyObject(sh)

Base.show(io::IO, sh::Shape) = print(io, eltype(sh), "[", join(sh.dims, ","), "]")

function Shape(sh::PyObject)
  T = juliatype(sh.numpy_dtype())
  size = sh.dimensions()
  Shape{T,length(size)}(size)
end

shapeof(p::PyObject) = p.is_array() ? Shape(p) : (shapeof.(p.tuple_shapes())...,)

pyshape(x::Tuple) = xlaclient.Shape.tuple_shape(pyshape.(x))

pyshape(x::Type{<:XScalar}) = pyshape(Shape(x, ()))

# Values

struct XArray{T,N} <: AbstractArray{T,N}
  buffer::PyObject
end

function setup_finaliser(x)
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

scalar(x::XArray{T,0}) where T = collect(x)[]
scalar(x::XArray) = x

xla(x::XArray) = x
xla(x::AbstractArray) = XArray(x)
xla(x::Number) = XArray(fill(x))
xla(x::Tuple) = xla.(x)

function wrapvalue(p::PyObject)
  p.shape().is_tuple() ? (wrapvalue.(p.destructure())...,) : scalar(XArray(p))
end

# IR Builder

function build(ir::IR)
  builder = xlaclient.ComputationBuilder("")
  env = Dict()
  resolve(x::Variable) = env[x]
  resolve(x) = const!(builder, x)
  for (v, T) in zip(arguments(ir), argtypes(ir))
    env[v] = builder.ParameterWithShape(pyshape(T))
  end
  for (v, st) in ir
    ex = st.expr
    if isexpr(ex, :call)
      env[v] = build!(builder, ex.args[1], resolve.(ex.args[2:end])...)
    elseif ex isa IR
      env[v] = build(ex)
    # elseif isexpr(ex, :lambda)
    #   env[v] = Lambda(resolve.(ex.args[2:end]), ex.args[1])
    elseif isexpr(ex)
      error("Invalid XLA expression $(ex)")
    else
      env[v] = const!(builder, ex)
    end
  end
  return builder.Build()
end

function compile(ir::IR)
  comp = build(ir).Compile()
  return (xs...) -> wrapvalue(comp.Execute(xlaclient.Buffer.from_pyval.(xs)))
end
