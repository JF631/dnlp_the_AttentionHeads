
import torch
import bert
from bert import BertModel
from tokenizer import BertTokenizer


class TestconvBERTModel():
    input_ids = torch.tensor([])
    input_mask = torch.tensor([])

    def setUp(self):
        text = "BERT is great for NLP tasks."
        encoded = BertTokenizer.encode_plus(
            text,
            add_special_tokens=True,  # Adds [CLS] at start and [SEP] at end
            max_length=16,  # Pad/truncate to a fixed length
            padding="max_length",
            truncation=True,
            return_tensors="pt"  # Return as PyTorch tensors
        )
        print(encoded)
        input_ids, attention_mask  = encoded

    def test_embed(self):
        bert.BertModel.embed()
        return

if __name__ == '__main__':
    test = TestBERTModel()
    test.test_embed()

