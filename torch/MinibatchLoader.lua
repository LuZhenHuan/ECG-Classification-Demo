local MinibatchLoader = {}
MinibatchLoader.__index = MinibatchLoader

function MinibatchLoader.create(batch_size)
    -- split_fractions is e.g. {0.9, 0.05, 0.05}

    local self = {}
    setmetatable(self, MinibatchLoader)
    
    self.batch_size = batch_size
    self.batch_idx = 1

    local data_filename = 'data.t7'

    print('loading data files...')
    self.data = torch.load(data_filename)
    
    self.sample_size = self.data[1]:size(1)

    print(string.format('data load done.'))
    collectgarbage()
    return self
end


function MinibatchLoader:next_batch()

    X = self.data[1]:sub((self.batch_idx-1)*self.batch_size+1, self.batch_idx*self.batch_size)
    Y = self.data[2]:sub((self.batch_idx-1)*self.batch_size+1, self.batch_idx*self.batch_size)
    self.batch_idx = self.batch_idx + 1
    
    if(self.batch_idx*self.batch_size > self.sample_size) then
        self.batch_idx = 1
    end
    
    table_x = {}
    table_y = {}
    
    for i=1,X:size(2) do
      table.insert(table_x, X[{{},{i}}])
      table.insert(table_y, Y[{{},{i}}])
    end
    
    return table_x, table_y
end

return MinibatchLoader
