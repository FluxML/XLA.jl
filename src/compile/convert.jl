struct Operations end

Primitives() = Multi(Operations(), Basic())

exprtype(ir, x) = IRTools.exprtype(ir, x, typeof = Const)

layout(x::XScalar) = typeof(x)
layout(x) = map(f -> layout(getfield(x, f)), fieldnames(typeof(x)))
layout(x::Array) = Shape(eltype(x), size(x))

for (op, xop) in [(+, :Add), (*, :Mul), (-, :Sub), (^, :Pow), (>, :Gt), (<, :Lt)]
  @eval abstract(::Operations, ::AType{typeof($op)}, a::AType{T}, b::AType{T}) where T<:XScalar =
    Core.Compiler.return_type($op, Tuple{T,T})
  @eval xlaop(args, ::AType{typeof($op)}, _, _) =
          xcall(XLA.$xop(), args[2:end]...)
end

for (op, xop) in [(+, :Add), (-, :Sub)]
  @eval abstract(::Operations, ::AType{typeof($op)}, a::AType{T}, b::AType{T}) where T<:Array{<:XScalar} =
    Core.Compiler.return_type($op, Tuple{T,T})
end

fieldnum(T, f) = findfirst(==(f), fieldnames(T))

xlaop(args, ::AType{typeof(getfield)}, ::AType{T}, f::Const{Symbol}) where T =
  Expr(:call, XLA.GetTupleElement(fieldnum(T, f.value)-1), args[2])

function xlaops!(ir)
  for (v, st) in ir
    ir[v] = xlaop(st.expr.args, exprtype.((ir,), st.expr.args)...)
  end
  return ir
end

function convert_xla!(ir, T)
  xlaops!(ir)
  for i = 1:length(arguments(ir))
    argtypes(ir)[i] = layout(T[i])
  end
  for (v, st) in ir
    ir[v] = stmt(st, type = Any)
  end
  return ir
end
