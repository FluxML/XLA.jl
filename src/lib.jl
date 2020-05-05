using FillArrays

instead(::Operations, args, ::AType{Type{Fill}}, x, sz) =
  ([fill, args[2], args[3]], (Const(fill), x, sz))
