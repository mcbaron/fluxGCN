# models.jl
using Flux
using Flux: glorot_uniform
struct GCNLayer
    W
    b
    adj
    act
end

struct GCNModel
    layer1::GCNLayer
    layer2::GCNLayer
    drop
end

function GCNLayer(in::Integer, out::Integer, adj, act=relu; initW=glorot_uniform,
                        initb=zeros)
    return GCNLayer(param(initW(in, out)), param(initb(out)), adj, act)
end

function (a::GCNLayer)(x::AbstractArray)
    support = x * a.W
    return a.act.(a.adj * support .+ a.b')
end

Flux.@treelike GCNLayer


function GCNModel(nfeat, nhid, nclass, drop, adj)
    return GCNModel(GCNLayer(nfeat, nhid, adj), GCNLayer(nhid, nclass, adj, identity), drop)
end

function (a::GCNModel)(x::AbstractArray)
    z = Chain(a.layer1, Dropout(a.drop), a.layer2)
    return softmax(z(x)')'
end

Flux.@treelike GCNModel
