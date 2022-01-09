import torch
from typing import List, Dict

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

from wordle.constants import CHARSET, NUM_RESULTS, NUM_LETTERS, NUM_ROUNDS

TensorType = torch.Tensor


class WordleModel(TorchModelV2, torch.nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        torch.nn.Module.__init__(self)
        embedding_dim = model_config["custom_model_config"]["embedding_dim"]
        self.embedding_dim = embedding_dim
        self.char_embedding = torch.nn.EmbeddingBag(len(CHARSET), embedding_dim=embedding_dim)
        self.result_embedding = torch.nn.EmbeddingBag(NUM_RESULTS, embedding_dim=embedding_dim)

        self.attn = torch.nn.MultiheadAttention(128, num_heads=4, batch_first=True)
        self.transformer = torch.nn.Transformer(d_model=128, nhead=4)
        self.value_head = torch.nn.Linear(embedding_dim, 1)

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):

        # TODO: simply this by getting rid of excessive one-hot encoding (just use ints)
        batch_size, num_rounds, _shape = input_dict["obs"]["feedback"]["feedback"].values.shape
        feedback_results = input_dict["obs"]["feedback"]["feedback"].values\
            .reshape((batch_size, num_rounds, _shape // NUM_RESULTS, NUM_RESULTS))\
            .argmax(dim=3)
        feedback_results_embedding = self.result_embedding(feedback_results.view(-1, NUM_LETTERS))\
            .view(batch_size, NUM_ROUNDS, self.embedding_dim)

        _, _, _shape = input_dict["obs"]["feedback"]["word"].values.shape
        feedback_words = input_dict["obs"]["feedback"]["word"].values.reshape((batch_size, num_rounds, _shape // 27, 27)).argmax(dim=3)
        feedback_char_embedding = self.char_embedding(feedback_words.view(-1, NUM_LETTERS))\
            .view(batch_size, NUM_ROUNDS, self.embedding_dim)

        vocab_lengths = input_dict["obs"]["vocab"]["word"].lengths
        assert (vocab_lengths == vocab_lengths[0].item()).all().item(), "Different vocab lengths inside sample???"
        vocab_length = max(int(vocab_lengths[0].item()), 1)
        batch_size, num_rounds, _shape = input_dict["obs"]["vocab"]["word"].values.shape
        vocab_words = input_dict["obs"]["vocab"]["word"].values.reshape((batch_size, num_rounds, _shape // 27, 27)).argmax(dim=3)[:, :vocab_length]

        vocab_embedding = self.char_embedding(vocab_words.view(-1, NUM_LETTERS)).view(batch_size, vocab_length, self.embedding_dim)
        feedback_embedding = feedback_char_embedding + feedback_results_embedding

        attn_output, attn_weights = self.attn(vocab_embedding, feedback_embedding, feedback_embedding)

        logits = attn_weights.sum(dim=2)
        self._features = attn_output.mean(dim=1)

        return logits, []

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        return self.value_head(self._features).squeeze(1)  # T
