import os
import random
import time
import torch
import torch.nn as nn
import numpy as np

from transformers import AdamW
from colbert.utils.runs import Run
from colbert.utils.amp import MixedPrecisionManager

from colbert.training.lazy_batcher import LazyBatcher
from colbert.training.eager_batcher_custom import EagerBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT, KPEncoder
from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints


def train(args):
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    if args.distributed:
        torch.cuda.manual_seed_all(12345)

    if args.distributed:
        assert args.bsize % args.nranks == 0, (args.bsize, args.nranks)
        assert args.accumsteps == 1
        args.bsize = args.bsize // args.nranks

        print("Using args.bsize =", args.bsize,
              "(per process) and args.accumsteps =", args.accumsteps)

    if args.lazy:
        reader = LazyBatcher(
            args, (0 if args.rank == -1 else args.rank), args.nranks)
    else:
        print_message("Using eager batcher")
        reader = EagerBatcher(
            args, (0 if args.rank == -1 else args.rank), args.nranks)

    if args.rank not in [-1, 0]:
        torch.distributed.barrier()

    # load the document encoder from the ColBERT model
    colbert = ColBERT.from_pretrained('bert-base-uncased',
                                      query_maxlen=args.query_maxlen,
                                      doc_maxlen=args.doc_maxlen,
                                      dim=args.dim,
                                      similarity_metric=args.similarity,
                                      mask_punctuation=args.mask_punctuation)

    assert args.checkpoint is not None
    assert args.resume_optimizer is False, "TODO: This would mean reload optimizer too."
    print_message(
        f"#> Starting from checkpoint {args.checkpoint} -- but NOT the optimizer!")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    # load a checkpoint for colbertD
    colbert.load_state_dict(checkpoint['model_state_dict'], strict=False)

    kpencoder = KPEncoder.from_pretrained('bert-base-uncased')
    # start kpencoder from colbert weights
    kpencoder.load_state_dict(checkpoint['model_state_dict'], strict=False)

    kpencoder = kpencoder.to(DEVICE)

    colbert = colbert.to(DEVICE)
    kpencoder.train()

    optimizer = AdamW(filter(lambda p: p.requires_grad,
                      kpencoder.parameters()), lr=args.lr, eps=1e-8)
    optimizer.zero_grad()

    amp = MixedPrecisionManager(args.amp)
    criterion = nn.CrossEntropyLoss()
    labels = torch.zeros(args.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = 0.0

    start_batch_idx = 0

    for batch_idx, BatchSteps in zip(range(start_batch_idx, args.maxsteps), reader):
        this_batch_loss = 0.0

        for queries, passages in BatchSteps:
            with amp.context():
                Q = kpencoder(*queries)
                D = colbert.doc(*passages)
                scores = colbert.score(Q, D).view(2, -1).permute(1, 0)
                loss = criterion(scores, labels[:scores.size(0)])
                loss = loss / args.accumsteps

            if args.rank < 1:
                print_progress(scores)

            amp.backward(loss)

            train_loss += loss.item()
            this_batch_loss += loss.item()

        amp.step(kpencoder, optimizer)

        if args.rank < 1:
            avg_loss = train_loss / (batch_idx+1)

            num_examples_seen = (batch_idx - start_batch_idx) * \
                args.bsize * args.nranks
            elapsed = float(time.time() - start_time)

            log_to_mlflow = (batch_idx % 20 == 0)
            Run.log_metric('train/avg_loss', avg_loss,
                           step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/batch_loss', this_batch_loss,
                           step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/examples', num_examples_seen,
                           step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/throughput', num_examples_seen /
                           elapsed, step=batch_idx, log_to_mlflow=log_to_mlflow)

            print_message(batch_idx, avg_loss)
            manage_checkpoints(args, kpencoder, optimizer, batch_idx+1)
