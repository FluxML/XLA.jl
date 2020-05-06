using XLA, IRTools, Test
using IRTools: IR, argument!, xcall
using XLA: XTuple, GetTupleElement, Sub, Mul, Gt, While, compile

ir = IR()
x = argument!(ir, Int)
n = argument!(ir, Int)
xnr = push!(ir, xcall(XTuple(), x, n, 1))
cond = let ir = IR()
  xnr = argument!(ir)
  n = push!(ir, xcall(GetTupleElement(1), xnr))
  push!(ir, xcall(Gt(), n, 0))
  ir
end
cond = push!(ir, cond)
body = let ir = IR()
  xnr = argument!(ir)
  x = push!(ir, xcall(GetTupleElement(0), xnr))
  n = push!(ir, xcall(GetTupleElement(1), xnr))
  r = push!(ir, xcall(GetTupleElement(2), xnr))
  n = push!(ir, xcall(Sub(), n, 1))
  r = push!(ir, xcall(Mul(), x, r))
  xnr = push!(ir, xcall(XTuple(), x, n, r))
  ir
end
body = push!(ir, body)
xnr = push!(ir, xcall(While(), cond, body, xnr))
push!(ir, xcall(GetTupleElement(2), xnr))
pow = compile(ir)

@test pow(2, 3)[1] == 2^3
@test pow(5, 10)[1] == 5^10
