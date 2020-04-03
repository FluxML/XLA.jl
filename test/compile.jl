using XLA, Test

add = (a, b) -> a+b
@test @code_xla(add(2, 3)) isa XLA.IR

double = xla(x -> x + x)

@test double(21) == 42
@test double(4.5) == 9.0

add = xla(add)

@test add(2, 2.0) == 4.0

@test xla(() -> 2+2)() == 4

@test xla(x -> 3x^(1+1) + (2x + 1))(5) == 86

relu = xla(x -> x > 0 ? x : 0)

@test relu(5) == 5
@test relu(-5) == 0
@test_broken relu(5.0) == 5.0

let x = rand(3), y = rand(3)
  @test add(x, y) isa XLA.XArray
  @test collect(add(x, y)) == x+y
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
