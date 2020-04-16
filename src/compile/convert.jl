struct Operations end

Primitives() = Multi(Operations(), Basic())

exprtype(ir, x) = IRTools.exprtype(ir, x, typeof = Const)

layout(x::XScalar) = typeof(x)
layout(x) = map(f -> layout(getfield(x, f)), fieldnames(typeof(x)))
layout(x::Array) = Shape(eltype(x), size(x))

# Base's `<` does something complicated.
simplelt(a, b) = <(promote(a, b)...)

instead(::Operations, args, ::AType{typeof(<)}, a::AType{<:XScalar}, b::AType{<:XScalar}) =
  widen(a) == widen(b) ? nothing : ([simplelt, args[2], args[3]], (Const(simplelt), a, b))

abstract(::Operations, ::AType{typeof(convert)}, ::AType{Type{T}}, x::Const{<:XScalar}) where T<:XScalar =
  Const(convert(T, x.value))

abstract(::Operations, ::AType{typeof(convert)}, ::AType{Type{T}}, S::AType{<:XScalar}) where T<:XScalar =
  widen(S) <: T ? S : T

abstract(::Operations, ::AType{Type{T}}, S::AType{<:XScalar}) where T<:XScalar =
  abstract(Operations(), Const(convert), Const(T), S)

xlaop(args, ::AType{typeof(convert)}, ::AType{Type{T}}, _) where T<:XScalar =
  xcall(ConvertElementType(T), args[3])

xlaop(args, ::AType{typeof(Broadcast.broadcasted)}, ::AType{Type{T}}, _) where T<:XScalar =
  xcall(ConvertElementType(T), args[3])

for (op, xop) in [(+, :Add), (*, :Mul), (-, :Sub), (^, :Pow), (>, :Gt), (<, :Lt)]
  @eval abstract(::Operations, ::AType{typeof($op)}, a::Const{T}, b::Const{T}) where T<:XScalar =
    Const($op(a.value, b.value))
  @eval abstract(::Operations, ::AType{typeof($op)}, a::AType{T}, b::AType{T}) where T<:XScalar =
    Core.Compiler.return_type($op, Tuple{T,T})
  @eval xlaop(args, ::AType{typeof($op)}, a::AType{T}, b::AType{T}) where T<:XScalar =
          xcall($xop(), args[2:end]...)
  @eval xlaop(args, ::AType{typeof(Broadcast.broadcasted)}, ::AType{typeof($op)}, ::AType{<:Array{T}}, ::AType{<:Array{T}}) where T<:XScalar =
    xcall($xop(), args[3:end]...)
end

for (op, xop) in [(+, :Add), (-, :Sub)]
  @eval abstract(::Operations, ::AType{typeof($op)}, a::AType{T}, b::AType{T}) where T<:Array{<:XScalar} =
    Core.Compiler.return_type($op, Tuple{T,T})
  @eval xlaop(args, ::AType{typeof($op)}, a::AType{T}, b::AType{T}) where T<:Array{<:XScalar} =
          xcall($xop(), args[2:end]...)
end

abstract(::Operations, ::AType{typeof(Broadcast.broadcasted)}, ::AType{typeof(identity)}, x) = x

abstract(::Operations, ::AType{typeof(*)}, a::AType{Matrix{T}}, b::AType{Vector{T}}) where T<:XScalar = Vector{T}

xlaop(args, ::AType{typeof(*)}, a::AType{<:Array{T}}, b::AType{<:Array{T}}) where T<:XScalar =
  xcall(Dot(), args[2:end]...)

fieldnum(T, f) = findfirst(==(f), fieldnames(T))

xlaop(args, ::AType{typeof(getfield)}, ::AType{T}, f::Const{Symbol}) where T =
  Expr(:call, GetTupleElement(fieldnum(T, f.value)-1), args[2])

function xlaops!(ir)
  for (v, st) in ir
    st.expr isa Union{Array,Number} && continue
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
