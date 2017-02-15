--MIT data load and process
function loadEcgData()
    require 'torch'
    trainset = torch.load('MITdata01.t7')
    
