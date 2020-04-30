toxla(x::XScalar) = Any[x]
toxla(x::Array{<:XScalar}) = Any[x]
toxla(x) = vcat(map(f -> toxla(getfield(x, f)), fieldnames(typeof(x)))...)

function trace(Ts...)
  ir = Mjolnir.trace(Primitives(), Ts...)
  return broadcasts!(ir)
end

function xla(f)
  cache = IdDict()
  function (args...)
    T = xtypeof(args)
    if haskey(cache, T)
      xla_f = cache[T]
    else
      ir = trace(Const(f), typeof.(args)...)
      deletearg!(ir, 1) # `f` is constant
      ir = convert_xla!(ir, T)
      xla_f = cache[T] = XLA.compile(ir)
    end
    return xla_f(toxla(args)...)
  end
end
