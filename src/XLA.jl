module XLA

using IRTools, IRTools.All, PyCall, Mjolnir
using IRTools.Inner: entry
using MacroTools: @capture
import Mjolnir: AType, Multi, Basic, Const, trace, abstract, instead, widen

export @code_xla, xla

function __init__()
  global xlaclient = pyimport("jaxlib.xla_client")
  PyCall.npyinitialize()
end

include("ir/reloop.jl")
include("ir/builder.jl")
include("ir/ops.jl")

include("compile/convert.jl")
include("compile/rt.jl")

macro code_xla(ex)
  @capture(ex, f_(args__)) || error("@trace f(args...)")
  quote
    tr = trace(Primitives(), Const($(esc(f))), typeof.(($(esc.(args)...),))...)
    convert_xla!(tr, ($(esc(f)), $(esc.(args)...)))
  end
end

end # module
