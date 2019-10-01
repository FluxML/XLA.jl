module XLATools

using IRTools, IRTools.All, PyCall
using IRTools.Inner: entry

function __init__()
  @eval const xlaclient = pyimport("jaxlib.xla_client")
  @eval const xrt = pyimport("jaxlib.xrt")
end

include("reloop.jl")
include("xla.jl")
include("ops.jl")

end # module
