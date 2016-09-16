-- util
local util = require 'autograd.util'
local functionalize = require('autograd.nnwrapper').functionalize
local nn = functionalize('nn')

local layerNorm = require 'autograd.module.LayerNormalization'
local softMax = nn.SoftMax()

return function(opt, params, layers)
  --[[ LSTM with Soft Attention Model Constructor

  LSTM with soft attention as defined in Show, Attend, and Tell by Xu et al. (http://arxiv.org/abs/1502.03044)

  Parameters:
  * `opt` - Options table.
  * `params` - Existing autograd params table.
  * `layers` - Existing layers table.

  Options:
  * `inputFeatures` - Size of input x.
  * `hiddenFeatures` - Size of hidden and cell states.
  * `subjectFeatures` - Size of input subjectFeatures.
  * `subjectChoices` - Number of choices in subject. If 1 then subject can have variable number of choices if batch size is also 1.
  * `outputType` - 'last' or 'all'. Output last state or state for all steps.
  * `batchNormalization` - Use batch normalization.
  * `maxBatchNormalizationLayers` - Maximum number of batch norm layers per step. Steps beyond the last use the last layer.
  * `layerNormalization` - Use layer normalization.
  * `gatedAttention` - Optional gate on attention input to LSTM.
  * `useFocusLoss` - Optional loss term to encourage equal attention to all choices.

  Returns:
  * `f` - LSTM with soft attention evaluation functions.
  * `params` - Autograd parameters table.
  * `layers` - Table of nn.BatchNormalization layers that contain additional state if batchNormalization is true.
  --]]

  -- options:
  opt = opt or {}
  local inputFeatures = opt.inputFeatures or 10
  local hiddenFeatures = opt.hiddenFeatures or 100
  local subjectFeatures = opt.subjectFeatures or 100
  local subjectChoices = opt.subjectChoices or 1
  local outputType = opt.outputType or 'last' -- 'last' or 'all'
  local batchNormalization = opt.batchNormalization or false
  local maxBatchNormalizationLayers = opt.maxBatchNormalizationLayers or 10
  local layerNormalization = opt.layerNormalization or false
  local gatedAttention = opt.gatedAttention or false
  local useFocusLoss = opt.useFocusLoss or false

  -- container:
  params = params or {}
  local layers = layers or {}
  local layer_norm

  -- parameters:
  local p = {
    Wx = torch.zeros(inputFeatures, 4 * hiddenFeatures),
    bx = torch.zeros(1, 4 * hiddenFeatures),
    Wh = torch.zeros(hiddenFeatures, 4 * hiddenFeatures),
    bh = torch.zeros(1, 4 * hiddenFeatures),
    Wa = torch.zeros(subjectFeatures, 4 * hiddenFeatures),
    ba = torch.zeros(1, 4 * hiddenFeatures),
    W_att_subject = torch.zeros(1, subjectFeatures, subjectFeatures),
    b_att_subject = torch.zeros(1, 1, subjectFeatures),
    W_att_h = torch.zeros(hiddenFeatures, subjectFeatures),
    b_att_h = torch.zeros(1, subjectFeatures),
    W_att = torch.zeros(1, subjectFeatures, 1),
    b_att = torch.zeros(1, subjectChoices)
  }
  if batchNormalization then
    -- translation and scaling parameters are shared across time.
    local lstm_bn, p_lstm_bn = nn.BatchNormalization(4 * hiddenFeatures)
    local cell_bn, p_cell_bn = nn.BatchNormalization(hiddenFeatures)

    layers.lstm_bn = {lstm_bn}
    layers.cell_bn = {cell_bn}

    for i=2,maxBatchNormalizationLayers do
      local lstm_bn = nn.BatchNormalization(4 * hiddenFeatures)
      local cell_bn = nn.BatchNormalization(hiddenFeatures)
      layers.lstm_bn[i] = lstm_bn
      layers.cell_bn[i] = cell_bn
    end
    -- initializing scaling to < 1 is recommended for LSTM batch norm.
    p.lstm_bn_1 = p_lstm_bn[1]:fill(0.1)
    p.lstm_bn_2 = p_lstm_bn[2]:zero()
    p.cell_bn_1 = p_cell_bn[1]:fill(0.1)
    p.cell_bn_2 = p_cell_bn[2]:zero()

  elseif layerNormalization then
    layer_norm = layerNorm()
    local ln_function, lstm_ln_params = layerNorm({nOutputs = 4 * hiddenFeatures})
    p.lstm_x_ln_gain = lstm_ln_params[1].gain
    p.lstm_x_ln_bias = lstm_ln_params[1].bias
    ln_function, lstm_ln_params = layerNorm({nOutputs = 4 * hiddenFeatures})
    p.lstm_h_ln_gain = lstm_ln_params[1].gain
    p.lstm_h_ln_bias = lstm_ln_params[1].bias
    ln_function, lstm_ln_params = layerNorm({nOutputs = 4 * hiddenFeatures})
    p.lstm_a_ln_gain = lstm_ln_params[1].gain
    p.lstm_a_ln_bias = lstm_ln_params[1].bias

    local ln_function, cell_ln_params = layerNorm({nOutputs = hiddenFeatures})
    p.cell_ln_gain = cell_ln_params[1].gain
    p.cell_ln_bias = cell_ln_params[1].bias

    local ln_function, h_embed_ln_params = layerNorm({nOutputs = subjectFeatures})
    p.h_embed_ln_gain = h_embed_ln_params[1].gain
    p.h_embed_ln_bias = h_embed_ln_params[1].bias

    local ln_function, subject_ln_params = layerNorm({nOutputs = subjectFeatures})
    p.subject_embed_ln_gain = subject_ln_params[1].gain
    p.subject_embed_ln_bias = subject_ln_params[1].bias
  end

  if gatedAttention then
    p.W_att_gate = torch.zeros(hiddenFeatures, 1)
    p.b_att_gate = torch.zeros(1, 1)
  end

  table.insert(params, p)

  local attend = function(params, subject_embed, h)
    --[[ Soft attention over subject given subject embedding and LSTM hidden state.

    Deterministic soft attention of Show, Attend, and Tell by Xu et al. (http://arxiv.org/abs/1502.03044)
    This function assumes subject_embed is precomputed before LSTM loop to be more efficient.

    Parameters:
    * `params` - Weights to combine subject and hidden features to score choices.
    * `subject_embed` - Precomputed ([batch,] subjectChoices, subjectFeatures) tensor.
    * `h` - ([batch,] hiddenFeatures) tensor.

    Returns:
    * `focus` - ([batch,], subjectChoices) tensor that is the probability of selecting any given subject choice.
    --]]
    local p = params[1] or params
    local subject_in = subject_embed
    local h_in = h
    if torch.nDimension(subject_embed) == 2 then
      subject_in = torch.view(subject_embed, 1, torch.size(subject_embed, 1), torch.size(subject_embed, 2))
    end
    if torch.nDimension(h) == 1 then
      h_in = torch.view(h, 1, torch.size(h, 1))
    end
    local batchSize = torch.size(subject_in, 1)
    local subjectChoices = torch.size(subject_in, 2)
    local subjectFeatures = torch.size(subject_in, 3)

    -- Project hidden state to subject embedding space.
    local h_embed = h_in * p.W_att_h
    h_embed = h_embed + torch.expand(p.b_att_h, torch.size(h_embed))
    if layerNormalization and layerNorm_h_embed then
      h_embed = layer_norm({gain = p.h_embed_ln_gain, bias = p.h_embed_ln_bias}, h_embed)
    end
    h_embed = torch.expand(torch.view(h_embed, batchSize, 1, subjectFeatures), torch.size(subject_in))

    -- Combine hidden and subject embeddings.
    local combined_embed = subject_in + h_embed
    combined_embed = torch.tanh(combined_embed)

    -- Return focus distribution over subject choices.
    local focus = torch.bmm(combined_embed, torch.expand(p.W_att, batchSize, subjectFeatures, 1))
    focus = torch.squeeze(focus, 3) + torch.expand(p.b_att, batchSize, subjectChoices)
    focus = softMax(focus)
    return focus
  end

  local f = function(params, x, subject, prevState, layers, masks)
    --[[ Evaluation function for LSTM with soft attention.

    Parameters:
    * `params` - LSTM parameters.
    * `x` - Input tensor of size ([batch,], steps, features).
    * `subject` - Input tensor for attention of size ([batch,], subjectChoices, subjectFeatures).
    * `prevState` - Previous activations for hidden state and cell state.
    * `layers` - Table of BatchNormalization modules containing additional state.
    * `masks` - Binary mask for x indicating non-padded steps. Used when computing focus loss term.

    Returns:
    * `hiddens` - Last or all hidden states depending on opt.outputType.
    * `newState` - Hidden and cell state at last step.
    * `layers` - BatchNormalization modules with updated state.
    * `focus_loss` - Loss term encouraging equal attention to all subject choices across all steps when opt.useFocusLoss.
    * `fs` - Last or all focus distributions on subject choices.
    * `ats` - Last or all expected attention vectors over subject with respect to focus distribution.
    --]]

    -- dims:
    local p = params[1] or params
    if torch.nDimension(x) == 2 then
      x = torch.view(x, 1, torch.size(x, 1), torch.size(x, 2))
    end
    if torch.nDimension(subject) == 2 then
      subject = torch.view(subject, 1, torch.size(subject, 1), torch.size(subject, 2))
    end
    local batch = torch.size(x, 1)
    local steps = torch.size(x, 2)
    local subjectChoices = torch.size(subject, 2)

    -- hiddens:
    subject = subject or error("Attentional LSTM requires subject input.")
    prevState = prevState or {}
    local hs = {}
    local cs = {}
    local ats = {}
    local fs = {}
    local focus_last, focus_sum
    if useFocusLoss then
      focus_sum = torch.zero(x.new(batch, subjectChoices))
    end

    -- pre-attention LSTM subject embedding
    local subjects_embed = torch.bmm(subject, torch.expand(p.W_att_subject, batch, subjectFeatures, subjectFeatures))
    subjects_embed = subjects_embed + torch.expand(p.b_att_subject, torch.size(subjects_embed))
    if layerNormalize then
      subjects_embed = torch.view(subjects_embed, -1, subjectFeatures)
      subjects_embed = layer_norm({gain = p.subject_ln_gain, bias = p.subject_ln_bias}, subjects_embed)
      subjects_embed = torch.view(subjects_embed, batchSize, subjectChoices, subjectFeatures)
    end

    -- go over time:
    for t = 1,steps do
      -- xt
      local xt = torch.select(x,2,t)

      local mt
      if masks then
        mt = torch.contiguous(torch.select(masks,2,t))
        mt = torch.view(mt,batch,1)
      end

      -- prev h and prev c
      local hp = hs[t-1] or prevState.h
      or torch.zero(x.new(batch, hiddenFeatures))
      local cp = cs[t-1] or prevState.c
      or torch.zero(x.new(batch, hiddenFeatures))

      -- batch norm for t, independent mean and std across time steps
      local lstm_bn, cell_bn
      if batchNormalization then
        if layers.lstm_bn[t] then
          lstm_bn = layers.lstm_bn[t]
          cell_bn = layers.cell_bn[t]
        else
          -- all time steps beyond maxBatchNormalizationLayers uses the last one.
          lstm_bn = layers.lstm_bn[#layers.lstm_bn]
          cell_bn = layers.cell_bn[#layers.cell_bn]
        end
      end

      -- Focus across subject
      local focus_t = attend(p, subjects_embed, hp, dropout)

      -- Attend to choice in expectation.
      local expanded_focus = torch.expand(torch.view(focus_t, batch, subjectChoices, 1), torch.size(subject))
      local at = torch.squeeze(torch.sum(torch.cmul(subject, expanded_focus), 2), 2)

      if gatedAttention then -- Optional gate on attention input to LSTM.
        local attentionGate = hp * p.W_att_gate
        attentionGate = attentionGate + torch.expand(p.b_att_gate, batch, 1)
        attentionGate = util.sigmoid(attentionGate)

        at = torch.cmul(torch.expand(attentionGate, torch.size(at)), at)
        focus_t = torch.cmul(torch.expand(attentionGate, torch.size(focus_t)), focus_t )
      end

      if useFocusLoss then -- Optional loss term to encourage equal attention to each choice across time.
        if mt then
          focus_t = torch.cmul(focus_t, torch.expand(mt, torch.size(focus_t)))
        end
        focus_sum = focus_sum + focus_t
      end
      fs[t] = focus_t
      ats[t] = at

      -- LSTM dot products:
      local xdots = xt * p.Wx + torch.expand(p.bx, batch, 4 * hiddenFeatures)
      local hdots = hp * p.Wh + torch.expand(p.bh, batch, 4 * hiddenFeatures)
      local adots = at * p.Wa + torch.expand(p.ba, batch, 4 * hiddenFeatures)

      if layerNorm_lstm and layerNormalization then
        xdots = layer_norm({gain = p.lstm_x_ln_gain, bias = p.lstm_x_ln_bias}, xdots)
        hdots = layer_norm({gain = p.lstm_h_ln_gain, bias = p.lstm_h_ln_bias}, hdots)
        adots = layer_norm({gain = p.lstm_a_ln_gain, bias = p.lstm_a_ln_bias}, adots)
      end

      local dots = xdots + hdots + adots

      if batchNormalization then
        -- use translation parameter from batch norm as bias.
        dots = lstm_bn({p.lstm_bn_1, p.lstm_bn_2}, dots)
      end

      -- view as 4 groups:
      dots = torch.view(dots, batch, 4, hiddenFeatures)
      local inputGate = torch.select(dots, 2, 1, 1)
      local forgetGate = torch.select(dots, 2, 2, 1)
      local outputGate = torch.select(dots, 2, 3, 1)
      local inputValue = torch.select(dots, 2, 4, 1)
      inputGate = util.sigmoid(inputGate)
      forgetGate = util.sigmoid(forgetGate)
      outputGate = util.sigmoid(outputGate)
      inputValue = torch.tanh(inputValue)

      -- next c:
      cs[t] = torch.cmul(forgetGate, cp) + torch.cmul(inputGate, inputValue)

      if batchNormalization then
        cs[t] = cell_bn({p.cell_bn_1, p.cell_bn_2}, cs[t])
      end

      -- next h:
      local c_act = cs[t]
      if layerNorm_lstm_out and layerNormalization then
        c_act = layer_norm({gain = p.cell_ln_gain, bias = p.cell_ln_bias}, c_act)
      end
      c_act = torch.tanh(c_act)
      hs[t] = torch.cmul(outputGate, c_act)
    end

    -- save state
    local newState = {h=hs[#hs], c=cs[#cs]}

    -- attention diversity loss
    local focus_loss = 0
    if useFocusLoss then
      focus_sum = 1 - focus_sum
      focus_loss = torch.sum(torch.cmul(focus_sum, focus_sum))
    end

    -- output
    if outputType == 'last' then
      -- return last hidden state:
      return hs[#hs], newState, layers, focus_loss, fs[#fs], ats[#ats]
    else
      -- return all hidden states:
      for i in ipairs(hs) do
        hs[i] = torch.view(hs[i], batch, 1, hiddenFeatures)
        fs[i] = torch.view(fs[i], batch, 1, subjectChoices)
        ats[i] = torch.view(ats[i], batch, 1, subjectFeatures)
      end
      return x.cat(hs, 2), newState, layers, focus_loss, x.cat(fs, 2), x.cat(ats, 2)
    end
  end

  -- layers
  return f, params, layers
end
