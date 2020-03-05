using XLA, Test

@testset "XLA" begin

@testset "XLA IR" begin
  include("xla.jl")
end

@testset "Compile" begin
  include("compile.jl")
end

end
