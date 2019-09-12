const XLAScalar = Union{Float64,Float32,Float16,Int64,Int32,Int16}

struct XLAArray{T,N} <: AbstractArray{T,N}
  buffer::PyObject
end

function XLAArray(x::Array{<:XLAScalar})
  buffer = xlaclient.Buffer.from_pyval(x)
  xla = XLAArray{eltype(x),ndims(x)}(buffer)
  delete = buffer.delete # work around a segfault on exit
  finalizer(buf -> ispynull(delete) || delete(), xla.buffer)
  return xla
end

Base.size(x::XLAArray) = x.buffer.shape().dimensions()
Base.collect(x::XLAArray) = x.buffer.to_py()
Base.print_array(io::IO, x::XLAArray) = Base.print_array(io, collect(x))

function backend(platform="cpu")
  xlaclient.get_local_backend(platform)
end
