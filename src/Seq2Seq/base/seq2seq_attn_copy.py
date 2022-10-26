from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from torch.autograd import Variable

from utils import to_one_hot, DecoderBase

SEED = 1234
random.seed(SEED)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        
        self.hidden_size = enc_hid_dim

        self.embedding = nn.Embedding(input_dim, emb_dim)        
        self.rnn = nn.GRU(emb_dim, enc_hid_dim,bidirectional=True)  ####################################
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)        
        self.dropout = nn.Dropout(dropout)

    def forward(self, iput, hidden, lengths):
        # iput batch must be sorted by sequence length
        iput = iput.masked_fill(iput > self.embedding.num_embeddings, 3)  # replace OOV words with <UNK> before embedding
        embedded = self.embedding(iput)
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(packed_embedded, hidden)
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(2, batch_size, self.hidden_size))  ####################################
        if next(self.parameters()).is_cuda:
            return hidden.cuda()
        else:
            return hidden

    def __forward(self, src): # copy mechanism 을 적용하기 전 encoder의 forward 함수
        
        #src = [src sent len, batch size]
        
        embedded = self.dropout(self.embedding(src))        
        #embedded = [src sent len, batch size, emb dim]
        
        outputs, hidden = self.rnn(embedded)
        #outputs = [src sent len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
        
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
        
        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        
        #outputs = [src sent len, batch size, enc hid dim * num directions]
        #hidden = [batch size, dec hid dim]
        
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))
        
    def forward(self, hidden, encoder_outputs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat encoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src sent len, dec hid dim]
        #encoder_outputs = [batch size, src sent len, enc hid dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))         
        #energy = [batch size, src sent len, dec hid dim]
        
        energy = energy.permute(0, 2, 1)        
        #energy = [batch size, dec hid dim, src sent len]
        
        #v = [dec hid dim]
        
        v = self.v.repeat(batch_size, 1).unsqueeze(1)        
        #v = [batch size, 1, dec hid dim]
                
        attention = torch.bmm(v, energy).squeeze(1)
        #attention= [batch size, src len]
        
        return F.softmax(attention, dim=1)        

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.hidden_size = dec_hid_dim * 2
        
        self.output_dim = output_dim
        self.attention = attention
        self.attn_W = nn.Linear(self.hidden_size, self.hidden_size)
        self.copy_W = nn.Linear(self.hidden_size, self.hidden_size)     
        self.embedding = nn.Embedding(output_dim, emb_dim)        
        self.rnn = nn.GRU((enc_hid_dim * 2) + self.hidden_size + emb_dim, self.hidden_size, batch_first = True) # 인코더가 bidirection하면서 input 요구 크기에 dec_hid_dim을 더해줌
        self.out = nn.Linear(self.hidden_size, output_dim)        
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        
    def forward(self, input, hidden, encoder_outputs, max_length, targets=None, keep_prob=1.0, teacher_forcing=0.5):
             
        #input = [batch size, src_len]
        #hidden = [num directions, batch size, dec hid dim]
        #encoder_outputs = [batch size, src sent len, enc hid dim * num directions]

        self.max_length = max_length

        batch_size = encoder_outputs.size()[0]
        seq_length = encoder_outputs.size()[1]

        # every decoder output seq starts with <SOS>
        sos_output = Variable(torch.zeros((batch_size, self.embedding.num_embeddings + seq_length))+2) # sos token id = 2
        sampled_idx = Variable(torch.ones((batch_size, 1)).long())
        if next(self.parameters()).is_cuda:
            sos_output = sos_output.cuda()
            sampled_idx = sampled_idx.cuda()

        decoder_outputs = [sos_output]
        sampled_idxs = [sampled_idx]

        if keep_prob < 1.0:
            dropout_mask = (Variable(torch.rand(batch_size, 1, 2 * self.hidden_size + self.embedding.embedding_dim)) < keep_prob).float() / keep_prob
        else:
            dropout_mask = None

        selective_read = Variable(torch.zeros(batch_size, 1, self.hidden_size))
        one_hot_input_seq = to_one_hot(input, self.output_dim + seq_length)
        if next(self.parameters()).is_cuda:
            selective_read = selective_read.cuda()
            one_hot_input_seq = one_hot_input_seq.cuda()

        for step_idx in range(1, self.max_length):

            if targets is not None and teacher_forcing > 0.0 and step_idx < targets.shape[1]:
                # replace some inputs with the targets (i.e. teacher forcing)
                teacher_forcing_mask = Variable((torch.rand((batch_size, 1)) < teacher_forcing), requires_grad=False)
                if next(self.parameters()).is_cuda:
                    teacher_forcing_mask = teacher_forcing_mask.cuda()
                sampled_idx = sampled_idx.masked_scatter(teacher_forcing_mask, targets[:, step_idx-1:step_idx])

            sampled_idx, output, hidden, selective_read = self.step(sampled_idx, hidden, encoder_outputs, selective_read, one_hot_input_seq, dropout_mask=dropout_mask)

            decoder_outputs.append(output)
            sampled_idxs.append(sampled_idx)

        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        sampled_idxs = torch.stack(sampled_idxs, dim=1)

        decoder_outputs = decoder_outputs.permute(1,0,2) # [batch, trg_len, trg_vocab_size] -> [trg_len, batch, trg_vocab_size]

        return decoder_outputs, sampled_idxs

    def step(self, prev_idx, prev_hidden, encoder_outputs, prev_selective_read, one_hot_input_seq, dropout_mask=None):
        # hidden = [num directions, batch size, dec hid dim]
        # encoder_outputs = encoder_outputs.permute(1,0,2) # [batch, src_len, enc_hid_emb * num directions]
        batch_size = encoder_outputs.shape[0]
        seq_length = encoder_outputs.shape[1]
        vocab_size = self.output_dim       
        
        prev_hidden = prev_hidden.view(batch_size, 1, self.hidden_size)

        # Attention mechanism
        transformed_hidden = self.attn_W(prev_hidden).view(batch_size, self.hidden_size, 1) # [batch,1,dec_hid_dim] -> [batch, hid, 1]
        # torch.bmm = batch matmul / attn_scores = [batch, src_len, 1]
        attn_scores = torch.bmm(encoder_outputs, transformed_hidden)  # reduce encoder outputs and hidden to get scores. remove singleton dimension from multiplication.
        attn_weights = F.softmax(attn_scores, dim=1)  # apply softmax to scores to get normalized weights
        context = torch.bmm(torch.transpose(attn_weights, 1, 2), encoder_outputs)  # [b, 1, hidden] weighted sum of encoder_outputs (i.e. values)

        # Call the RNN
        out_of_vocab_mask = prev_idx > vocab_size  # [b, 1] bools indicating which seqs copied on the previous step
        unks = torch.ones_like(prev_idx).long() * 3
        prev_idx = prev_idx.masked_scatter(out_of_vocab_mask, unks)  # replace copied tokens with <UNK> token before embedding
        embedded = self.embedding(prev_idx)  # embed input (i.e. previous output token)

        rnn_input = torch.cat((context, prev_selective_read, embedded), dim=2)
        if dropout_mask is not None:
            if next(self.parameters()).is_cuda:
                dropout_mask = dropout_mask.cuda()
            rnn_input *= dropout_mask

        self.rnn.flatten_parameters()
        output, hidden = self.rnn(rnn_input, prev_hidden)  # state.shape = [b, 1, hidden]

        # Copy mechanism
        transformed_hidden2 = self.copy_W(output).view(batch_size, self.hidden_size, 1)
        copy_score_seq = torch.bmm(encoder_outputs, transformed_hidden2)  # this is linear. add activation function before multiplying.
        copy_scores = torch.bmm(torch.transpose(copy_score_seq, 1, 2), one_hot_input_seq).squeeze(1)  # [b, vocab_size + seq_length]
        missing_token_mask = (one_hot_input_seq.sum(dim=1) == 0)  # tokens not present in the input sequence
        missing_token_mask[:, 0] = 1  # <MSK> tokens are not part of any sequence
        copy_scores = copy_scores.masked_fill(missing_token_mask, -1000000.0)

        # Generate mechanism
        gen_scores = self.out(output.squeeze(1))  # [b, vocab_size]
        gen_scores[:, 0] = -1000000.0  # penalize <MSK> tokens in generate mode too

        # Combine results from copy and generate mechanisms
        combined_scores = torch.cat((gen_scores, copy_scores), dim=1)
        probs = F.softmax(combined_scores, dim=1)
        gen_probs = probs[:, :vocab_size]

        gen_padding = Variable(torch.zeros(batch_size, seq_length))
        if next(self.parameters()).is_cuda:
            gen_padding = gen_padding.cuda()
        gen_probs = torch.cat((gen_probs, gen_padding), dim=1)  # [b, vocab_size + seq_length]

        copy_probs = probs[:, vocab_size:]

        final_probs = gen_probs + copy_probs

        log_probs = torch.log(final_probs + 10**-10)

        _, topi = log_probs.topk(1)
        sampled_idx = topi.view(batch_size, 1)

        # Create selective read embedding for next time step
        reshaped_idxs = sampled_idx.view(-1, 1, 1).expand(one_hot_input_seq.size(0), one_hot_input_seq.size(1), 1)
        pos_in_input_of_sampled_token = one_hot_input_seq.gather(2, reshaped_idxs)  # [b, seq_length, 1]
        selected_scores = pos_in_input_of_sampled_token * copy_score_seq
        selected_scores_norm = F.normalize(selected_scores, p=1)

        selective_read = (selected_scores_norm * encoder_outputs).sum(dim=1).unsqueeze(1)

        return sampled_idx, log_probs, hidden, selective_read


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, sos_idx, device, max_length):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.sos_idx = sos_idx
        self.device = device
        self.max_length = max_length

        self.switch = nn.Linear(decoder.output_dim, 1) # pointer (copy) mechanism 용
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        if trg is None:
            assert teacher_forcing_ratio == 0, "Must be zero during inference"
            inference = True
            trg = torch.zeros((self.max_length, src.shape[1])).long().fill_(self.sos_idx).to(src.device)
        else:
            inference = False

        src = src.permute(1,0)
        trg = trg.permute(1,0)
        lengths = torch.tensor([i.shape[0] for i in src])

        batch_size = src.shape[0]
        max_len = trg.shape[1]
        hidden = self.encoder.init_hidden(batch_size)
        encoder_outputs, hidden = self.encoder(src, hidden, lengths)

        # trg_vocab_size = self.decoder.output_dim
        
        # #tensor to store decoder outputs
        # outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        # #encoder_outputs is all hidden states of the input sequence, back and forwards
        # #hidden is the final forward and backward hidden states, passed through a linear layer
        # encoder_outputs, hidden = self.encoder(src)

        decoder_outputs, sampled_idxs = self.decoder(src, hidden, encoder_outputs, max_len, targets=trg, teacher_forcing=teacher_forcing_ratio)

        return decoder_outputs

        
        # #first input to the decoder is the <sos> tokens
        # input = trg[0,:]
        
        # for t in range(1, max_len):
            
        #     #insert input token embedding, previous hidden state and all encoder hidden states
        #     #receive output tensor (predictions) and new hidden state
        #     output, hidden = self.decoder(input, hidden, encoder_outputs)
            
        #     #place predictions in a tensor holding predictions for each token
        #     outputs[t] = output
            
        #     #decide if we are going to use teacher forcing or not
        #     teacher_force = random.random() < teacher_forcing_ratio
            
        #     #get the highest predicted token from our predictions
        #     top1 = output.argmax(1) 
            
        #     #if teacher forcing, use actual next token as next input
        #     #if not, use predicted token
        #     input = trg[t] if teacher_force else top1


        # # output = [batch size, trg len, output dim] << 가져온 코드에서의 output 차원 구성
        # # attention = [batch size, n heads, trg len, src len] << 가져온 코드에서의 attention 구성 / transformer에서 따온거라 multihead
        # # copy mechanism
        # output = outputs.permute(1,0,2)
        # src = src.permute(1,0)
        # p_pointer = torch.sigmoid(self.switch(output)) / 10
        # attention = self.decoder.attn # 본 코드에서는 [batch, src_len]

		# #p_pointer = [batch size, trg len, 1]
        # if torch.max(src) + 1 > output.shape[-1]:
        #     extended = Variable(torch.zeros((output.shape[0], output.shape[1], torch.max(src) + 1 - output.shape[-1]))).to(output.device)
        #     output = torch.cat((output, extended), dim = 2)

		# 	#output = [batch size, trg len, output dim + oov num]

        # output = ((1 - p_pointer) * F.softmax(output, dim = 2))
        # attn = p_pointer * attention[:, 1]
        # for batch, _ in enumerate(src):
        #     for out_word_idx in range(len(output[batch])):
        #         attn_score = attn[batch][out_word_idx]
        #         for vocab_idx in (src[batch]): # src = [batch, src_len] // src[batch][소스 문장의 i번째 단어] = i-th단어의 vocab stoi
        #             output[batch][out_word_idx][vocab_idx] = output[batch][out_word_idx][vocab_idx] + attn_score + 1e-10
        # # output = ((1 - p_pointer) * F.softmax(output, dim = 2)).scatter_add(2, src.unsqueeze(1).repeat(1, output.shape[1], 1), p_pointer * attention[:, 1]) + 1e-10
        # # # scatter_add(dim, index, src) : 이 함수를 적용하는 self(tensor)에 src의 dim행의 각 원소의 값들을 self의 대응하는 열의 index행에 더함
        # output = output.permute(1,0,2)

        # return torch.log(output)