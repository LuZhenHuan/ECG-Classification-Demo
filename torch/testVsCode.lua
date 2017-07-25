local cmd = torch.CmdLine()

cmd:option('-gpu', 0)
cmd:option('data', 'D1')

local opt = cmd:parse(arg)

print (opt.gpu)