const XScalar = Union{Float64,Float32,Float16,Int64,Int32,Int16}

struct XArray{T,N} <: AbstractArray{T,N}
  buffer::PyObject
end

function XArray(x::Array{<:XScalar})
  buffer = xlaclient.Buffer.from_pyval(x)
  xla = XArray{eltype(x),ndims(x)}(buffer)
  delete = buffer.delete # work around a segfault on exit
  finalizer(buf -> ispynull(delete) || delete(), xla.buffer)
  return xla
end

Base.size(x::XArray) = x.buffer.shape().dimensions()
Base.collect(x::XArray) = x.buffer.to_py()
Base.print_array(io::IO, x::XArray) = Base.print_array(io, collect(x))

function backend(platform="cpu")
  xlaclient.get_local_backend(platform)
end
