function broadcasts!(ir)
  for (v, st) in ir
    ex = st.expr
    if isexpr(ex, :call) && ex.args[1] == Broadcast.broadcasted
      f = exprtype(ir, ex.args[2])
      args = exprtype.((ir,), ex.args[3:end])
      args = eltype.(widen.(args))
      func = trace(f, args...)
      func = insert!(ir, v, func)
      ex.args[1] = broadcast
      ex.args[2] = func
    end
  end
  return ir
end
