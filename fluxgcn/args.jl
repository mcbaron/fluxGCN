# args.jl
using ArgParse

function parse_commandline()
s = ArgParseSettings()
@add_arg_table s begin
    "--no-cuda"
        help = "Disables CUDA Training."
        default = false
    "--fastmode"
        help = "Validate during training pass"
        default = false
    "--seed"
        help = "Random seed"
        arg_type = Int
        default = 42
    "--epochs"
        help = "Number of Training Epochs"
        arg_type = Int
        default = 200
    "--lr"
        help="Learning Rate"
        arg_type = Float64
        default = 0.01
    "--weight_decay"
        help = "Weight decay (L2 loss on parameters"
        arg_type = Float64
        default = 5e-4
    "--hidden"
        help = "Number of hidden units"
        arg_type = Int
        default = 16
    "--dropout"
        help = "Dropout rate (1 - keep prob)"
        arg_type = Float64
        default = 0.5
    end

    return parse_args(s)
end
