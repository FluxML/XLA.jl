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
    elseif isexpr(ex, :call) && ex.args[1] == mapreduce
      # TODO reduce initialisation
      @assert ex.args[2] == identity
      op = exprtype(ir, ex.args[3])
      x = exprtype(ir, ex.args[4])
      x = eltype(widen(x))
      func = trace(op, x, x)
      func = insert!(ir, v, func)
      ex.args[3] = func
    elseif isexpr(ex, :call) && ex.args[1] == KwFunc{typeof(mapreduce)}()
      # TODO clean this up
      @assert ex.args[2+2] == identity
      op = exprtype(ir, ex.args[3+2])
      x = exprtype(ir, ex.args[4+2])
      x = eltype(widen(x))
      func = trace(op, x, x)
      func = insert!(ir, v, func)
      ex.args[3+2] = func
    end
  end
  return ir
end

function prints!(ir)
  ps = []
  for (v, st) in ir
    if iscall(st.expr, println)
      push!(ps, v)
    end
  end
  ret = returnvalue(blocks(ir)[end])
  for p in ps
    ret = push!(ir, stmt(xcall(tuple, p, ret), type = Mjolnir.ptuple(exprtype(ir, p), exprtype(ir, ret))))
  end
  return!(ir, ret)
  return ir
end
