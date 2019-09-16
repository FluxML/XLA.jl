module XLATools

using IRTools, PyCall
using IRTools: IR, Variable, xcall, argument!, arguments, argtypes, isexpr

function __init__()
  @eval const xlaclient = pyimport("jaxlib.xla_client")
  @eval const xrt = pyimport("jaxlib.xrt")
end

include("xla.jl")
include("ops.jl")

end # module
