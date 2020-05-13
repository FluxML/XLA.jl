toxla(x::XScalar) = Any[x]
toxla(x::XArray) = Any[x]
toxla(x::Array{<:XScalar}) = Any[x]
toxla(x) = vcat(map(f -> toxla(getfield(x, f)), fieldnames(typeof(x)))...)

rebuild(T::AType{<:Union{XScalar,Array,Bool}}, xs) = T isa Const ? T.value : popfirst!(xs)

# TODO: this is a hack.
# Where types match we can construct the type directly.
# Where they don't (i.e. XArray) we should go via Functors.jl.
constructor(T::Type) = T.name.wrapper
constructor(T::Type{<:Tuple}) = tuple
constructor(T::Type{<:NamedTuple{K}}) where K = (xs...) -> NamedTuple{K}(xs)

function rebuild(T::Partial, xs)
  xs = rebuild.(T.value, (xs,))
  T = constructor(widen(T))
  return T(xs...)
end

function rebuild(T::Type{<:Tuple}, xs) # TODO generalise
  xs = rebuild.(T.parameters, (xs,))
  tuple(xs...)
end

rebuild(T::Mjolnir.Node, xs) = rebuild(widen(T), xs)
rebuild(T::Mjolnir.Shape, xs) = rebuild(widen(T), xs)
rebuild(T::Const, xs) = T.value

printstuff(x) = x
printstuff(x::Print) = nothing
function printstuff(x::Tuple{Print,Any})
  printstuff(x[1])
  println(x[1].data...)
  return printstuff(x[2])
end

function trace(Ts...)
  ir = Mjolnir.trace(Primitives(), Ts...)
  return ir |> broadcasts! |> prints!
end

struct XFunction
  func
  cache::Dict
end

xla(f) = XFunction(f, Dict())

function (f::XFunction)(args...)
  T = xtypeof.(args)
  if haskey(f.cache, T)
    (xla_f, out) = f.cache[T]
  else
    ir = trace(Const(f.func), T...)
    out = IRTools.returntype(blocks(ir)[end])
    deletearg!(ir, 1) # `f` is constant
    ir = convert_xla!(ir, ptuple(T...))
    xla_f, = f.cache[T] = XLA.compile(ir), out
  end
  return rebuild(out, xla_f(toxla(args)...)) |> printstuff
end

function Base.show(io::IO, f::XFunction)
  print(io, "xla(")
  show(io, f.func)
  print(io, ")")
end

invokeoriginal(f::XFunction, args...) = f.func(args...)

instead(::Operations, args, F::AType{<:XFunction}, xs...) =
  ([invokeoriginal, args...], (Const(invokeoriginal), F, xs...))

isxla() = false
@abstract Operations isxla() = Const(true)
