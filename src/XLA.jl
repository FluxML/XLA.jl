module XLA

using IRTools, IRTools.All, PyCall, Mjolnir, MacroTools
using IRTools: block
using IRTools.Inner: entry
using MacroTools: @capture
import Mjolnir: AType, Partial, Multi, Basic, Const, KwFunc, abstract, instead,
  ptuple, widen, @abstract

export @code_xla, xla, isxla

function __init__()
  global xlaclient = pyimport("jaxlib.xla_client")
  PyCall.npyinitialize()
end

include("ir/reloop.jl")
include("ir/builder.jl")
include("ir/ops.jl")

include("compile/passes.jl")
include("compile/convert.jl")
include("compile/rt.jl")

include("lib.jl")

macro code_xla(ex)
  @capture(ex, f_(args__)) || error("@trace f(args...)")
  quote
    tr = trace(Const($(esc(f))), xtypeof.(($(esc.(args)...),))...)
    deletearg!(tr, 1)
    convert_xla!(tr, xtypeof(($(esc.(args)...),))) |> renumber
  end
end

end # module
