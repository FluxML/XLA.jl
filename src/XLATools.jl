module XLATools

using IRTools, PyCall
using IRTools: IR, CFG, Variable, xcall, argument!, arguments, argtypes,
  block, isexpr, renumber, explicitbranch!, merge_returns!, expand!, entry

function __init__()
  @eval const xlaclient = pyimport("jaxlib.xla_client")
  @eval const xrt = pyimport("jaxlib.xrt")
end

include("reloop.jl")
include("xla.jl")
include("ops.jl")

end # module
