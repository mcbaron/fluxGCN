# layers.jl

struct GCNLayer
    W
    b
end

GCNLayer(in::Integer, out::Integer) =
  GCNLayer(param(randn(out, in)), param(randn(out)))

(m::GCNLayer)(x, adj) = adj * m.W * x .+ m.b
