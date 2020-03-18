module XLA

using IRTools, IRTools.All, PyCall, Mjolnir
using IRTools.Inner: entry
using MacroTools: @capture
using Mjolnir: Defaults, AType, Const, trace

export @code_xla, xla

function __init__()
  global xlaclient = pyimport("jaxlib.xla_client")
  PyCall.npyinitialize()
end

include("reloop.jl")
include("builder.jl")
include("ops.jl")
include("convert.jl")
include("rt.jl")

macro code_xla(ex)
  @capture(ex, f_(args__)) || error("@trace f(args...)")
  quote
    tr = trace(Defaults(), Const($(esc(f))), typeof.(($(esc.(args)...),))...)
    convert_xla!(tr, ($(esc(f)), $(esc.(args)...)))
  end
end

end # module
