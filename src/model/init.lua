-- standard models
local model = {
   NeuralNetwork = require 'autograd.model.NeuralNetwork',
   SpatialNetwork = require 'autograd.model.SpatialNetwork',
   RecurrentNetwork = require 'autograd.model.RecurrentNetwork',
   RecurrentLSTMNetwork = require 'autograd.model.RecurrentLSTMNetwork',
   RecurrentGRUNetwork = require 'autograd.model.RecurrentGRUNetwork',
   AttentionLSTMNetwork = require 'autograd.model.AttentionLSTMNetwork',
   AlexNet = require 'autograd.model.AlexNet'
}

return model
