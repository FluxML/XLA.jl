struct Operations end

Primitives() = Multi(Operations(), Basic())

exprtype(ir, x) = IRTools.exprtype(ir, x, typeof = Const)

xtypeof(x::XScalar) = typeof(x)
xtypeof(x::Tuple) = ptuple(xtypeof.(x)...)
xtypeof(x::Array{<:XScalar}) = Mjolnir.Shape{typeof(x)}(size(x))
xtypeof(x) = Partial{typeof(x)}((; map(f -> f=>xtypeof(getfield(x, f)), fieldnames(typeof(x)))...))

layout(x::Type{<:XScalar}) = [x]
layout(::Const) = []
layout(x::Shape) = [x]
layout(x::Mjolnir.Shape) = [Shape(eltype(x), size(x))]
layout(x::Type{<:Array{<:XScalar}}) = [x]
layout(x::Partial) = vcat(map(i -> layout(x.value[i]), 1:fieldcount(widen(x)))...)
layout(x::Type) = vcat(map(f -> layout(fieldtype(x, f)), fieldnames(x))...)
layout(x::Mjolnir.Node) = layout(widen(x))

subtype(T::Partial, i) = T.value[i]
subtype(T::Type, i) = fieldtype(T, i)

function subrange(T, i)
  offset = sum(Int[length(layout(subtype(T, j))) for j = 1:i-1])
  len = length(layout(subtype(T, i)))
  return offset .+ (1:len)
end

struct Print
  data
end

@abstract Operations repr(x) = x
@abstract Operations println(xs...) = Partial{Print}(Any[Mjolnir.ptuple(xs...)])

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

@abstract Operations fill(x::Const, sz::Const) = Const(fill(x.value, sz.value))

xlaop(args, ::AType{typeof(convert)}, ::AType{Type{T}}, _) where T<:XScalar =
  xcall(ConvertElementType(T), args[3])

xlaop(args, ::AType{Type{T}}, ::AType{<:XScalar}) where T<:XScalar =
  xcall(ConvertElementType(T), args[2])

for (op, xop) in [(+, :Add), (*, :Mul), (-, :Sub), (/, :Div), (^, :Pow), (>, :Gt), (<, :Lt), (max, :Max)]
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

@abstract Operations function (a::Matrix{T} * b::Vector{T}) where T<:XScalar
  n, m = size(a)
  m == size(b)[1] || error("Dimension mismatch")
  Mjolnir.Shape{Vector{T}}((n,))
end

xlaop(args, ::AType{typeof(*)}, a::AType{<:Array{T}}, b::AType{<:Array{T}}) where T<:XScalar =
  xcall(Dot(), args[2:end]...)

@abstract Operations adjoint(x::Vector{<:XScalar}) = Mjolnir.Shape{Matrix{eltype(x)}}((1, size(x)[1]))
@abstract Operations adjoint(x::Matrix{<:XScalar}) = Mjolnir.Shape{Matrix{eltype(x)}}(reverse(size(x)))

xlaop(args, ::AType{typeof(adjoint)}, x::AType{<:Vector}) = xcall(Reshape([0], [1, size(x)[1]]), args[2])
xlaop(args, ::AType{typeof(adjoint)}, x::AType{<:Matrix}) = xcall(Reshape([0,1], [reverse(size(x))...]), args[2])

xlaop(args, ::AType{typeof(repr)}, x) = args[2]

fieldnum(T, f) = findfirst(==(f), fieldnames(T))

function strip_self_arg!(ir)
  @assert exprtype(ir, arguments(ir)[1]) isa Const
  deletearg!(ir, 1)
  return ir
end

function tuplerange!(ir, v, xs, is)
  part(i) = length(layout(exprtype(ir, xs))) == 1 ? xs : insert!(ir, v, xcall(GetTupleElement(i-1), xs))
  ir[v] = length(is) == 1 ? part(is[]) : xcall(XTuple(), part.(is)...)
end

function tuplecat!(ir, v, xs)
  layouts = layout.(exprtype.((ir,), xs))
  if sum(length.(layouts)) == 1
    ir[v] = xs[findfirst(x -> !isempty(x), layouts)]
  else
    parts(i) = length(layouts[i]) == 1 ? [xs[i]] : [insert!(ir, v, xcall(GetTupleElement(j-1), xs[i])) for j = 1:length(layouts[i])]
    ir[v] = xcall(XTuple(), vcat([parts(i) for i = 1:length(xs)]...)...)
  end
end

function xlaop!(ir, v, ::AType{typeof(getindex)}, T::AType{<:Tuple}, i::Const{<:Integer})
  xs = ir[v].expr.args[2]
  is = subrange(T, i.value)
  tuplerange!(ir, v, xs, is)
end

function xlaop!(ir, v, ::AType{typeof(getfield)}, T::AType, i::Const{Symbol})
  xs = ir[v].expr.args[2]
  is = subrange(T, fieldnum(widen(T), i.value))
  tuplerange!(ir, v, xs, is)
end

function xlaop!(ir, v, ::AType{typeof(Mjolnir.__new__)}, ::AType{Type{T}}, xs...) where T
  @assert layout(T) == layout(Partial{Tuple{widen.(xs)...}}(xs))
  args = ir[v].expr.args[3:end]
  tuplecat!(ir, v, args)
end

function xlaop!(ir, v, ::AType{typeof(tuple)}, xs...)
  args = ir[v].expr.args[2:end]
  tuplecat!(ir, v, args)
end

function xlaop!(ir, v, ::AType{typeof(println)}, xs...)
  tuplecat!(ir, v, ir[v].expr.args[2:end])
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
    if st.expr isa Expr
      xlaop!(ir, v, exprtype.((ir,), st.expr.args)...)
    elseif st.expr isa IR
      ir[v] = xlaops!(st.expr)
    end
  end
  return ir
end

function flattenargs!(ir, T)
  arglayout = (layout(T)...,)
  args = copy(arguments(ir))
  deletearg!(ir, 1:length(args))
  components = [argument!(ir, arglayout[i]) for i = 1:length(arglayout)]
  env = Dict()
  for i = 1:length(args)
    cs = components[subrange(T,i)]
    env[args[i]] = length(cs) == 1 ? cs[1] : pushfirst!(ir, xcall(XTuple(), cs...))
  end
  prewalk!(x -> get(env, x, x), ir)
  for (v, st) in ir
    ir[v] = stmt(st, type = Any)
  end
  return ir
end

function convert_xla!(ir, T)
  xlaops!(ir)
  flattenargs!(ir, T)
  return ir
end
