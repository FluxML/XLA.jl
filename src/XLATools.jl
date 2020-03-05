module XLATools

using IRTools, IRTools.All, PyCall
using IRTools.Inner: entry

function __init__()
  global xlaclient = pyimport("jaxlib.xla_client")
end

include("reloop.jl")
include("builder.jl")
include("ops.jl")

end # module
