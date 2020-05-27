using XLA, NNlib, Test

@test @code_xla(2+3) isa XLA.IR

double = xla(x -> x + x)

@test double(21) == 42
@test double(4.5) == 9.0

xadd = xla(+)

@test xadd(2, 2.0) == 4.0

xadd2 = xla(xadd)

@test xadd2(2, 3) == 5

@test xla(() -> 2+2)() == 4

@test xla(x -> 3x^(1+1) + (2x + 1))(5) == 86

relu = x -> x > 0 ? x : zero(x)
xrelu = xla(relu)

@test xrelu(5) == 5
@test xrelu(-5) == 0
@test xrelu(5.0) == 5.0

let x = rand(3), y = rand(3)
  @test xadd(x, y) isa XLA.XArray
  @test collect(xadd(x, y)) == x+y
end

function updatezero!(env)
  if env[:x] < 0
    env[:x] = 0
  end
end

relu = xla() do x
  env = Dict()
  env[:x] = x
  updatezero!(env)
  return env[:x]
end

@test relu(5) == 5
@test relu(-5) == 0

poly(x) = x^2 + 1

poly(xs::AbstractArray) = poly.(xs)

xpoly = xla(poly)

@test xpoly(3) == 10

@test collect(xpoly([1, 2, 3])) == [2, 5, 10]

xsum = xla(xs -> reduce((a, b) -> a+b, xs))

@test xsum([1, 2, 3]) == 6

xexp = xla(x -> exp.(x))

@test xexp(2) == exp(2)

@test collect(xexp([1, 2, 3])) == exp.([1, 2, 3])

xsum1 = xla(xs -> sum(xs, dims = 1))

@test collect(xsum1([1, 2, 3, 4])) == [10]

@test collect(xsum1([1 2; 3 4])) == [4 6]

xsoftmax = xla(softmax)

@test collect(xsoftmax([1, 2, 3])) ≈ softmax([1, 2, 3])
@test_broken collect(xsoftmax([1 2; 3 4])) == softmax([1 2; 3 4])

xsquare = xla(x -> x^2)

@test xsquare(1+2im) == -3+4im

xsquare = xla(x -> @show x^2)

function stdout_string(f)
  _stdout = stdout
  try
    rd, = redirect_stdout()
    f()
    return String(readavailable(rd))
  finally
    redirect_stdout(_stdout)
  end
end

output = stdout_string() do
  @test xsquare(1+2im) == -3+4im
end

@test output == "x ^ 2 = -3 + 4im\n"

@test isxla() == false
@test xla(isxla)() == true

timestable(x) = x .* x'

@test collect(xla(timestable)([1, 2, 3])) == timestable([1, 2, 3])
