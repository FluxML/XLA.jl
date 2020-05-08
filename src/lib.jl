using Requires, FillArrays

instead(::Operations, args, ::AType{Type{Fill}}, x, sz) =
  ([fill, args[2], args[3]], (Const(fill), x, sz))

nop() = return

function requires()
  @require Zygote="e88e6eb3-aa80-5325-afca-941959d7151f" begin
    instead(::Operations, args, ::AType{typeof(Zygote.accum_global)}, _...) =
      [nop], [Const(nop)]
    instead(::Operations, args, ::AType{typeof(Zygote.accum_param)}, _...) =
      [nop], [Const(nop)]
  end
end
