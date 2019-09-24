rename(env, ex) = IRTools.prewalk(x -> x isa Variable ? env[x] : x, ex)
rename(::Nothing, ex) = ex

function xidentity()
  ir = IR()
  argument!(ir)
  return ir
end

function branches(ir, b)
  brs = IRTools.branches(b)
  meta = Dict()
  for br in brs[1:end-1]
    cond = push!(ir, xcall(Not(), br.condition))
    meta[br.block] = (cond, br.args)
  end
  meta[brs[end].block] = (true, brs[end].args)
  return meta
end

function extract(b)
  ir = renumber(IR(b))
  empty!(ir.blocks[1].branches)
  return ir
end

function reloop!(out, ir, cfg::IRTools.Multiple, brs)
  nbrs = length(brs)
  (cond, targs) = pop!(brs, cfg.inner[1].block)
  tfunc = push!(out, reloop_tupleargs(ir, cfg.inner[1]))
  if length(cfg.inner) == 2 && brs[cfg.inner[2].block][1] == true
    fargs = brs[cfg.inner[2].block][2]
    ffunc = push!(out, reloop_tupleargs(ir, cfg.inner[2]))
  elseif length(cfg.inner) == 1 && length(brs) == 1
    fargs = first(brs)[2][2]
    ffunc = push!(out, xidentity())
  else
    error("Multiple block: not implemented")
  end
  args = push!(out, xcall(Conditional(), cond,
                          xcall(XTuple(), targs...), tfunc,
                          xcall(XTuple(), fargs...), ffunc))
  reloop!(out, ir, cfg.next, args)
  return out
end

function reloop_next!(out, cfg, b, env = nothing)
  if cfg.next != nothing
    reloop!(out, b.ir, cfg.next, branches(out, b)) # TODO pass env
  elseif IRTools.isreturn(b)
    return
  else
    @assert length(IRTools.branches(b)) == 1 "Unstructured control flow not implemented"
    push!(out, xcall(XTuple(), rename.((env,), arguments(IRTools.branches(b)[1]))...))
  end
end

function reloop!(out, ir, cfg::IRTools.Simple, args)
  b = block(ir, cfg.block)
  env = Dict()
  for (i, k) in enumerate(arguments(b))
    env[k] = push!(out, xcall(GetTupleElement(i-1), args))
  end
  for (v, st) in b
    env[v] = push!(out, rename(env, st.expr))
  end
  reloop_next!(out, cfg, b, env)
  return out
end

function reloop_tupleargs(ir, cfg::IRTools.Simple)
  out = IR()
  args = argument!(out)
  reloop!(out, ir, cfg, args)
  return out
end

function reloop(ir, cfg::IRTools.Simple)
  b = block(ir, cfg.block)
  out = extract(b)
  reloop_next!(out, cfg, b)
  return out
end

function controlflow(ir::IR)
  ir = ir |> copy |> explicitbranch! |> merge_returns! |> expand!
  cfg = IRTools.reloop(CFG(ir))
  reloop(ir, cfg)
end
