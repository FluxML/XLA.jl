using XLA, Test

@testset "XLA" begin

@testset "IR" begin
  include("ir.jl")
end

@testset "Compile" begin
  include("compile.jl")
end

@testset "Flux" begin
  include("flux.jl")
end

end
