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

@abstract Operations convert(::Type{T}, x::Const{<:XScalar}) where T<:XScalar =
  Const(convert(T, x.value))

@abstract Operations convert(::Type{T}, S::XScalar) where T<:XScalar =
  widen(S) <: T ? S : T

@abstract Operations (::Type{T})(S::XScalar) where T<:XScalar =
  abstract(Operations(), Const(convert), Const(T), S)

xlaop(args, ::AType{typeof(convert)}, ::AType{Type{T}}, _) where T<:XScalar =
  xcall(ConvertElementType(T), args[3])

xlaop(args, ::AType{Type{T}}, ::AType{<:XScalar}) where T<:XScalar =
  xcall(ConvertElementType(T), args[2])

for (op, xop) in [(+, :Add), (*, :Mul), (-, :Sub), (^, :Pow), (>, :Gt), (<, :Lt)]
  @eval @abstract Operations $op(a::Const{T}, b::Const{T}) where T<:XScalar =
    Const($op(a.value, b.value))
  @eval @abstract Operations $op(a::AType{T}, b::AType{T}) where T<:XScalar =
    Core.Compiler.return_type($op, Tuple{T,T})
  @eval xlaop(args, ::AType{typeof($op)}, a::AType{T}, b::AType{T}) where T<:XScalar =
          xcall($xop(), args[2:end]...)
end

for (op, xop) in [(exp, :Exp)]
  @eval @abstract Operations $op(a::XFloat) = widen(a)
  @eval xlaop(args, ::AType{typeof($op)}, _) = xcall($xop(), args[2])
end

for (op, xop) in [(+, :Add), (-, :Sub)]
  @eval @abstract Operations $op(a::T, b::T) where T<:Array{<:XScalar} =
    Core.Compiler.return_type($op, Tuple{T,T})
  @eval xlaop(args, ::AType{typeof($op)}, a::AType{T}, b::AType{T}) where T<:Array{<:XScalar} =
          xcall($xop(), args[2:end]...)
end

xlaop(args, ::AType{typeof(broadcast)}, _...) =
  Expr(:call, Map(), args[3:end]..., args[2])

@abstract Operations (a::Matrix{T} * b::Vector{T}) where T<:XScalar = Vector{T}

xlaop(args, ::AType{typeof(*)}, a::AType{<:Array{T}}, b::AType{<:Array{T}}) where T<:XScalar =
  xcall(Dot(), args[2:end]...)

fieldnum(T, f) = findfirst(==(f), fieldnames(T))

xlaop(args, ::AType{typeof(tuple)}, xs...) =
  Expr(:call, XTuple(), args[2:end]...)

xlaop(args, ::AType{typeof(getfield)}, ::AType{T}, f::Const{Symbol}) where T =
  Expr(:call, GetTupleElement(fieldnum(T, f.value)-1), args[2])

function strip_self_arg!(ir)
  @assert exprtype(ir, arguments(ir)[1]) isa Const
  deletearg!(ir, 1)
  return ir
end

function xlaop!(ir, v, ::AType{typeof(broadcast)}, _...)
  args = ir[v].expr.args
  f = ir[args[2]].expr
  strip_self_arg!(f)
  ir[v] = Expr(:call, Map(),  args[2], args[3:end]...)
end

# TODO XLA drops dimensions, Julia doesn't; make XLA do the Julia thing.
function xlaop!(ir, v, ::AType{typeof(mapreduce)}, ::AType{typeof(identity)}, op, xs; dims = :)
  args = ir[v].expr.args
  args[1] isa KwFunc && (popfirst!(args); popfirst!(args))
  f = ir[args[3]].expr
  strip_self_arg!(f)
  ir[v] = Expr(:call, Reduce(dims), args[3], args[4], zero(eltype(widen(xs))))
end

xlaop!(ir, v, ::AType{<:KwFunc}, kw::Const, f, args...) =
  xlaop!(ir, v, f, args...; kw.value...)

function xlaop!(ir, v, args...; kw...)
  ir[v] = xlaop(ir[v].expr.args, args...; kw...)
end

function xlaops!(ir)
  for (v, st) in ir
    if st.expr isa Union{Array,Number}
    elseif st.expr isa IR
      ir[v] = xlaops!(st.expr)
    else
      xlaop!(ir, v, exprtype.((ir,), st.expr.args)...)
    end
  end
  return ir
end

function convert_xla!(ir, T)
  xlaops!(ir)
  for i = 1:length(arguments(ir))
    argtypes(ir)[i] = layout(T[i])
  end
  return ir
end
