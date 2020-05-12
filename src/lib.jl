using Requires, FillArrays

instead(::Operations, args, ::AType{Type{Fill}}, x, sz) =
  ([fill, args[2], args[3]], (Const(fill), x, sz))

nop() = return

xlogy(x, y) = x*log(y)

function requires()
  @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
    instead(::Operations, args, ::AType{typeof(Zygote.accum_global)}, _...) =
      [nop], [Const(nop)]
    instead(::Operations, args, ::AType{typeof(Zygote.accum_param)}, _...) =
      [nop], [Const(nop)]
    instead(::Operations, args, ::AType{typeof(Zygote._push!)}, xs, x) =
      [push!, args[2], args[3]], [Const(push!), xs, x]
  end
  @require Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c" begin
    instead(::Operations, args, ::AType{typeof(Flux.xlogy)}, x, y) =
      [xlogy, args[2], args[3]], [Const(xlogy), x, y]
  end
end
