module XLATools

using PyCall

function __init__()
  @eval const xlaclient = pyimport("jaxlib.xla_client")
  @eval const xrt = pyimport("jaxlib.xrt")
end

include("xla.jl")

end # module
