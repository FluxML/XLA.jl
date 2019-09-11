const XLAScalar = Union{Float64,Float32,Float16,Int64,Int32,Int16}

mutable struct XLAArray{T,N} <: AbstractArray{T,N}
  buffer::PyObject
end

function XLAArray(x::Array{<:XLAScalar})
  buffer = xlaclient.Buffer.from_pyval(x)
  xla = XLAArray{eltype(x),ndims(x)}(buffer)
  finalizer(x -> x.buffer.delete(), xla)
  return xla
end

Base.size(x::XLAArray) = x.buffer.shape().dimensions()
Base.collect(x::XLAArray) = x.buffer.to_py()
Base.print_array(io::IO, x::XLAArray) = Base.print_array(io, collect(x))

function backend(platform="cpu")
  xlaclient.get_local_backend(platform)
end
