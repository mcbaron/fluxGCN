#train.jl

include("util2.jl")
include("models.jl")

include("args.jl")

using Flux
using Flux: @epochs
import Flux.Tracker: Params, gradient, data, update!
using ArgParse

# Training Settings
FLAGS = parse_commandline()

# Load Data
adj, features, labels, idx_train, idx_val, idx_test = jlload_data("cora")

# process data
train_data = zip(features, labels)

# Model + optimizer
model = GCNModel(size(features, 2), FLAGS["hidden"], size(labels, 2), FLAGS["dropout"], adj)
optimizer = ADAMW(FLAGS["lr"], (0.9, 0.999), FLAGS["weight_decay"])

# Loss
function train_loss(x, y)
    @show(size(x))
    output = model(x)
    return Flux.logitcrossentropy(output[idx_train, :], y[idx_train, :])
end

function val_loss(x, y)
    output = model(x)
    return Flux.logitcrossentropy(output[idx_val, :], y[idx_val, :])
end

train_accuracy(x, y) = jlaccuracy(model(x)[idx_train, :], y[idx_train, :])
val_accuracy(x, y) = jlaccuracy(model(x)[idx_val, :], y[idx_val, :])

ps = Flux.params(model)

function train(train_loss, val_loss, ps, data, labels, optim)
    t0 = time()
    ps = Params(ps)
    @show(train_loss(data, labels))
    @show(val_loss(data, labels))

    @progress for i in 1:size(data,1)
        try
            gs = gradient(ps) do
                train_loss(data[i,:], labels[i,:])
            end
            update!(optim, ps, gs)
        catch ex
            rethrow(ex)
        end
    end

    @show(time() - t0)
end

function test()
    testmode!(0, true)

end

# train model
# testmode!(0, false) #into training mode
@epochs(FLAGS["epochs"], train(train_loss, val_loss, ps, features, labels, optimizer))
