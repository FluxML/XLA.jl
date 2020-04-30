toxla(x::XScalar) = Any[x]
toxla(x::Array{<:XScalar}) = Any[x]
toxla(x) = vcat(map(f -> toxla(getfield(x, f)), fieldnames(typeof(x)))...)

rebuild(::Type{<:Union{XScalar,Array}}, xs) = popfirst!(xs)

# TODO: this is a hack.
# Where types match we can construct the type directly.
# Where they don't (i.e. XArray) we should go via Functors.jl.
function rebuild(T::Partial, xs)
  xs = rebuild.(T.value, (xs,))
  T = widen(T).name.wrapper
  return T(xs...)
end

rebuild(T::Mjolnir.Node, xs) = rebuild(widen(T), xs)

function trace(Ts...)
  ir = Mjolnir.trace(Primitives(), Ts...)
  return broadcasts!(ir)
end

function xla(f)
  cache = Dict()
  function (args...)
    T = xtypeof(args)
    if haskey(cache, T)
      (xla_f, out) = cache[T]
    else
      ir = trace(Const(f), typeof.(args)...)
      out = IRTools.returntype(blocks(ir)[end])
      deletearg!(ir, 1) # `f` is constant
      ir = convert_xla!(ir, T)
      xla_f, = cache[T] = XLA.compile(ir), out
    end
    return rebuild(out, xla_f(toxla(args)...))
  end
end
