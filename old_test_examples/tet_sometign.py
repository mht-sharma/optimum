# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=redefined-outer-name,logging-format-interpolation
import argparse
import logging

import torch
from datasets import load_dataset
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.WARN
)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--batch_size",
    default=1,
    type=int,
)
parser.add_argument(
    "--tokenizer",
    type=str,
    help="pretrained model name or path of tokenizer files",
    default="decapoda-research/llama-7b-hf",
)
parser.add_argument(
    "--pad_max",
    default=196,
    type=int,
)
parser.add_argument("--dataset", nargs="?", default="NeelNanda/pile-10k", const="NeelNanda/pile-10k")
args = parser.parse_args()

# load model
tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer)


def tokenize_function(examples):
    example = tokenizer(examples["text"])
    return example


class KVDataloader:
    def __init__(self, pad_max=196, batch_size=1, sub_folder="train"):
        self.pad_max = pad_max
        self.batch_size = batch_size
        dataset = load_dataset(args.dataset, split=sub_folder)
        dataset = dataset.map(tokenize_function, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_batch,
        )
        self.sess = None

    def collate_batch(self, batch):
        input_ids_padded = []
        attention_mask_padded = []
        last_ind = []

        for text in batch:
            input_ids = text["input_ids"]
            pad_len = self.pad_max - input_ids.shape[0]
            last_ind.append(input_ids.shape[0] - 1)
            attention_mask = torch.ones(len(input_ids))
            input_ids = pad(input_ids, (0, pad_len), value=1)
            attention_mask = pad(attention_mask, (0, pad_len), value=0)
            input_ids_padded.append(input_ids)
            attention_mask_padded.append(attention_mask)
        return (torch.vstack(input_ids_padded), torch.vstack(attention_mask_padded)), torch.tensor(last_ind)

    def __iter__(self):
        try:
            for (input_ids, attention_mask), last_ind in self.dataloader:
                from pdb import set_trace

                set_trace()

                if self.sess is None:
                    yield {
                        "input_ids": input_ids[:, :-1].detach().cpu().numpy().astype("int64"),
                        "attention_mask": attention_mask[:, :-1].detach().cpu().numpy().astype("int64"),
                    }, last_ind.detach().cpu().numpy()
                else:
                    # outputs = self.sess.run(None, {'input_ids': input_ids[:, :-1].detach().cpu().numpy().astype('int64'),
                    #                                'attention_mask':attention_mask[:, :-1].detach().cpu().numpy().astype('int64')})
                    ort_input = {}
                    ort_input["input_ids"] = input_ids[:, -1].unsqueeze(0).detach().cpu().numpy().astype("int64")
                    # for i in range(int((len(outputs) - 1) / 2)):
                    #     ort_input['past_key_values.{}.key'.format(i)] = outputs[i*2+1]
                    #     ort_input['past_key_values.{}.value'.format(i)] = outputs[i*2+2]
                    # ort_input['attention_mask'] =  np.zeros([self.batch_size, ort_input['past_key_values.0.key'].shape[2]+1], dtype='int64')
                    yield ort_input, last_ind.detach().cpu().numpy()
        except StopIteration:
            return


if __name__ == "__main__":
    dataloader = KVDataloader(pad_max=args.pad_max, batch_size=1)
    for data in dataloader:
        pass
