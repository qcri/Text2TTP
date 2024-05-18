from torch import nn
import torch

def resize_token_embeddings(model, new_num_tokens: int, new_embs: torch.Tensor) -> nn.Embedding:
    model_embeds = _resize_token_embeddings(model, new_num_tokens, new_embs)

    # Update base model and current model config
    model.config.vocab_size = new_num_tokens
    model.vocab_size = new_num_tokens

    # Tie weights again if needed
    model.tie_weights()

    return model_embeds


def _resize_token_embeddings(self, new_num_tokens, new_embs):
    old_embeddings = self.get_input_embeddings()
    new_embeddings = _get_resized_embeddings(self, old_embeddings, new_num_tokens, new_embs)
    self.set_input_embeddings(new_embeddings)

    # if word embeddings are not tied, make sure that lm head is resized as well
    if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
        old_lm_head = self.get_output_embeddings()
        new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
        self.set_output_embeddings(new_lm_head)

    return self.get_input_embeddings()

def _get_resized_embeddings(
    self, old_embeddings: nn.Embedding, new_num_tokens: int, new_embs: torch.Tensor
) -> nn.Embedding:
    old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

    if old_num_tokens == new_num_tokens:
        return old_embeddings

    if not isinstance(old_embeddings, nn.Embedding):
        raise TypeError(
            f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}. You"
            " should either use a different resize function or make sure that `old_embeddings` are an instance of"
            f" {nn.Embedding}."
        )

    # Build new embeddings
    new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
    new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)

    # initialize all new embeddings (in particular added tokens)
    # self._init_weights(new_embeddings)

    # Copy token embeddings from the previous weights

    # numbers of tokens to copy
    n = min(old_num_tokens, new_num_tokens)
    new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
    
    # Init with zeros
    new_embeddings.weight.data[n:, :] = new_embs

    return new_embeddings