typekey(x::XScalar) = typeof(x)
typekey(x::Array) = (typeof(x), size(x)...)
typekey(x) = (typeof(x), map(f -> typekey(getfield(x, f)), fieldnames(typeof(x)))...)

toxla(x::XScalar) = x
toxla(x::Array{<:XScalar}) = x
toxla(x) = map(f -> toxla(getfield(x, f)), fieldnames(typeof(x)))

function xla(f)
  cache = IdDict()
  function (args...)
    key = typekey(args)
    if haskey(cache, key)
      xla_f = cache[key]
    else
      ir = trace(Defaults(), Const(f), typeof.(args)...)
      ir = convert_xla!(ir, (f, args...))
      xla_f = cache[key] = XLA.compile(ir)
    end
    return xla_f(toxla(()), toxla.(args)...)
  end
end
