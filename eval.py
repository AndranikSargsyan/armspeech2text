import torch
from torchmetrics.text import CharErrorRate, WordErrorRate
from tqdm import tqdm

from datamodule.datasets import ids2transcript


@torch.no_grad()
def run_evaluation(test_loader, model, decoder, device):
    model.eval()
    wer = WordErrorRate()
    cer = CharErrorRate()
    for i, (batch) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_sizes, target_sizes = batch
        inputs = inputs.to(model.device)
        inputs = inputs.transpose(0, 1)
        input_sizes = torch.LongTensor(input_sizes)
        target_sizes = torch.LongTensor(target_sizes)
        out, output_lengths = model(inputs, input_sizes)
        decoded_output, _ = decoder.decode(out, output_lengths)
        target_sentences = [ids2transcript(target[:t_size]) for target, t_size in zip(targets, target_sizes)]
        wer(decoded_output, target_sentences)
        cer(decoded_output, target_sentences)
    return wer.compute(), cer.compute()
