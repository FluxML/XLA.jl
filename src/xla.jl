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

Shape(x::AbstractArray) = Shape{eltype(x),ndims(x)}(size(x))

Base.eltype(::Shape{T}) where T = T
Base.ndims(::Shape{T,N}) where {T,N} = N

PyObject(sh::Shape) = xlaclient.Shape.array_shape(numpytype(eltype(sh)), sh.dims)

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
