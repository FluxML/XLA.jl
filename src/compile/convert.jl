struct Operations end

Primitives() = Multi(Operations(), Basic())

exprtype(ir, x) = IRTools.exprtype(ir, x, typeof = Const)

xtypeof(x::XScalar) = typeof(x)
xtypeof(x::Tuple) = ptuple(xtypeof.(x)...)
xtypeof(x::Array{<:XScalar}) = Mjolnir.Shape{typeof(x)}(size(x))
xtypeof(x::XArray{T,N}) where {T,N} = Mjolnir.Shape{Array{T,N}}(size(x))
xtypeof(x) = isbits(x) && nfields(x) == 0 ? Const(x) : Partial{typeof(x)}((; map(f -> f=>xtypeof(getfield(x, f)), fieldnames(typeof(x)))...))

layout(x::Type{<:XScalar}) = [x]
layout(::Const) = []
layout(x::XShape) = [x]
layout(x::Mjolnir.Shape) = [XShape(eltype(x), size(x))]
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

@abstract Basic Base.vect(xs::Const...) = Const([x.value for x in xs])

# Base's `<` does something complicated.
simplelt(a, b) = <(promote(a, b)...)

instead(::Operations, args, ::AType{typeof(<)}, a::AType{<:XScalar}, b::AType{<:XScalar}) =
  widen(a) == widen(b) ? nothing : ([simplelt, args[2], args[3]], (Const(simplelt), a, b))

instead(::Operations, args, ::AType{typeof(//)}, a::AType{<:XScalar}, b::AType{<:XScalar}) =
  [(/), args[2], args[3]], [Const(/), a, b]

@abstract Operations convert(::Type{T}, x::Const{<:XScalar}) where T<:XScalar =
  Const(convert(T, x.value))

@abstract Operations convert(::Type{T}, S::XScalar) where T<:XScalar =
  widen(S) <: T ? S : T

@abstract Operations (::Type{T})(S::XScalar) where T<:XScalar =
  abstract(Operations(), Const(convert), Const(T), S)

@abstract Operations fill(x::Const, sz::Const) = Const(fill(x.value, sz.value))
@abstract Operations ones(x::Const{<:Tuple{Vararg{Integer}}}) = Const(ones(x.value))

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

@abstract Operations -(a::Const{T}) where T<:XScalar = Const(-a.value)
@abstract Operations -(a::AType{T}) where T<:XScalar = T
xlaop(args, ::AType{typeof(-)}, a::AType{T}) where T<:XScalar =
        xcall(Neg(), args[2:end]...)

for (op, xop) in [(exp, :Exp), (sin, :Sin), (cos, :Cos), (log, :Log)]
  @eval @abstract Operations $op(a::XFloat) = widen(a)
  @eval xlaop(args, ::AType{typeof($op)}, _) = xcall($xop(), args[2])
end

for (op, xop) in [(+, :Add), (-, :Sub)]
  @eval @abstract Operations $op(a::T, b::T) where T<:Array{<:XScalar} =
    Shape{Array{eltype(a),ndims(a)}}(size(a))
  @eval xlaop(args, ::AType{typeof($op)}, a::AType{T}, b::AType{T}) where T<:Array{<:XScalar} =
          xcall($xop(), args[2:end]...)
end

@abstract Operations reshape(A::Array{T}, i::Const{NTuple{N,Int}}) where {T<:XScalar,N} =
  Shape{Array{T,length(i.value)}}(i.value)

xlaop(args, ::AType{typeof(reshape)}, x, sh::Const) =
  xcall(Reshape(1:ndims(x), sh.value), args[2])

@abstract Operations getindex(A::Vector{T}, i::Integer) where T<:XScalar = T

function xlaop!(ir, v, ::AType{typeof(getindex)}, A, i)
  args = ir[v].expr.args
  x = insert!(ir, v, xcall(DynamicSlice([1]), args[2], args[3:end]...))
  ir[v] = xcall(Reshape([1], []), x)
end

@abstract Operations function (a::Matrix{T} * b::Vector{T}) where T<:XScalar
  a isa Const && b isa Const && return Const(a.value * b.value)
  n, m = size(a)
  m == size(b)[1] || error("Dimension mismatch")
  Mjolnir.Shape{Vector{T}}((n,))
end

@abstract Operations function (a::Matrix{T} * b::Matrix{T}) where T<:XScalar
  a isa Const && b isa Const && return Const(a.value * b.value)
  n, m = size(a)
  m == size(b)[1] || error("Dimension mismatch")
  Mjolnir.Shape{Matrix{T}}((n,size(b)[2]))
end

@abstract Operations function (a::Vector{T} * b::Matrix{T}) where T<:XScalar
  a isa Const && b isa Const && return Const(a.value * b.value)
  n, = size(a)
  size(b)[1] == 1 || error("Dimension mismatch")
  Mjolnir.Shape{Vector{T}}((n,size(b)[2]))
end

function xlaop!(ir, v, ::AType{typeof(*)}, A::AType{<:Array{T}}, B::AType{<:Array{T}}) where T<:XScalar
  _, a, b = ir[v].expr.args
  A isa AType{<:Vector} && (a = insert!(ir, v, xcall(Reshape([1], [size(A)[1], 1]), a)))
  ir[v] = xcall(Dot(), a, b)
end

@abstract Operations adjoint(x::Shape{Vector{T}}) where T<:XScalar = Mjolnir.Shape{Matrix{T}}((1, size(x)[1]))
@abstract Operations adjoint(x::Shape{Matrix{T}}) where T<:XScalar = Mjolnir.Shape{Matrix{T}}(reverse(size(x)))
@abstract Operations adjoint(x::Const{<:Array{<:XScalar}}) = Const(collect(adjoint(x.value)))

xlaop(args, ::AType{typeof(adjoint)}, x::AType{<:Vector}) = xcall(Reshape([1], [1, size(x)[1]]), args[2])
xlaop(args, ::AType{typeof(adjoint)}, x::AType{<:Matrix}) = xcall(Reshape([1,2], [reverse(size(x))...]), args[2])

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
  args = ir[v].expr.args[3:end]
  tuplecat!(ir, v, args)
end

function xlaop!(ir, v, ::AType{typeof(Mjolnir.__splatnew__)}, ::AType{Type{T}}, xs) where T
  ir[v] = ir[v].expr.args[3]
end

function xlaop!(ir, v, ::AType{typeof(tuple)}, xs...)
  args = ir[v].expr.args[2:end]
  tuplecat!(ir, v, args)
end

function xlaop!(ir, v, ::AType{typeof(println)}, xs...)
  tuplecat!(ir, v, ir[v].expr.args[2:end])
end

function expand(ir, x, old, new)
  if x isa Variable
    x = insertafter!(ir, x, xcall(Reshape(1:length(old), ntuple(i -> i > length(old) ? 1 : old[i], length(new))), x))
    x = insertafter!(ir, x, xcall(BroadcastInDim(new, 1:length(new)), x))
  end
  return x
end

function xlaop!(ir, v, ::AType{typeof(broadcast)}, f, As...)
  args = ir[v].expr.args
  f = ir[args[2]].expr
  strip_self_arg!(f)
  xs = args[3:end]
  # TODO fold this into `expand`, do a generic type promotion
  for i = 1:length(xs)
    if As[i] isa Const{<:Number}
      xs[i] = fill(convert(eltype(As[1]), xs[i]), size(As[1]))
    end
  end
  new = Broadcast.broadcast_shape(size.(As)...)
  xs = expand.((ir,), xs, size.(As), (new,))
  ir[v] = Expr(:call, Map(), args[2], xs...)
end

function xlaop!(ir, v, ::AType{typeof(mapreduce)}, ::AType{typeof(identity)}, op, xs; dims = :)
  args = ir[v].expr.args
  args[1] isa KwFunc && (popfirst!(args); popfirst!(args))
  f = ir[args[3]].expr
  strip_self_arg!(f)
  if dims == (:)
    ir[v] = Expr(:call, Reduce(dims), args[3], args[4], zero(eltype(widen(xs))))
  else
    dims = filter(d -> d <= ndims(xs), collect(dims))
    out = insert!(ir, v, Expr(:call, Reduce(dims), args[3], args[4], zero(eltype(widen(xs)))))
    sz = size(ir[v].type)
    ir[v] = Expr(:call,
                 Reshape(1:length(size(xs))-length(dims),
                         ntuple(i -> i in dims ? 1 : size(xs)[i], length(sz))),
                 out)
  end
end

function xla_identity(T)
  ir = IR()
  return!(ir, argument!(ir, T))
  return ir
end

function xlaop!(ir, v, ::AType{typeof(ifelse)}, cond, T, F)
  _, cond, t, f = ir[v].expr.args
  tfunc = insert!(ir, v, xla_identity(T))
  ffunc = insert!(ir, v, xla_identity(F))
  ir[v] = xcall(Conditional(), cond, t, tfunc, f, ffunc)
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
  # TODO: more fine grained way of ensuring the right return value.
  t = push!(ir, xcall(XTuple(), returnvalue(blocks(ir)[end])))
  push!(ir, xcall(GetTupleElement(0), t))
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
