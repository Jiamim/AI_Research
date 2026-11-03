"""Two-process CPU (Gloo) checks of data-parallel semantics. Run: python -m examples.distributed

1. DDP: equal and unequal local batches reproduce the global-batch gradient when
   each rank scales its loss sum by the GLOBAL token count.
2. all-reduce = reduce-scatter + all-gather on a flat gradient buffer.
3. Optimizer-state sharding (simplified ZeRO-1): every rank sees the same batch here,
   so gradients already agree and the all-reduce real ZeRO-1 needs is omitted; each
   rank keeps AdamW state for only the parameters it owns, updates them, then
   broadcasts them; the result equals unsharded AdamW.

Teaching reference; coursework restrictions and this demo's scope are explained in chapter 28.
"""
import os
from pathlib import Path
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from .optim import AdamW


def ddp_check(rank, world_size):
    torch.manual_seed(9)
    model = torch.nn.Linear(3, 2, bias=False)
    reference = torch.nn.Linear(3, 2, bias=False)
    reference.load_state_dict(model.state_dict())
    x, y = torch.randn(8, 3), torch.randn(8, 2)
    for sizes in [(4, 4), (3, 5)]:
        model.zero_grad()
        reference.zero_grad()
        local_x, local_y = x.split(sizes)[rank], y.split(sizes)[rank]
        local_sum = ((model(local_x) - local_y) ** 2).sum()
        count = torch.tensor(float(local_y.numel()))
        dist.all_reduce(count)
        (world_size * local_sum / count).backward()
        for p in model.parameters():
            dist.all_reduce(p.grad)
            p.grad.div_(world_size)
        ((reference(x) - y) ** 2).mean().backward()
        torch.testing.assert_close(model.weight.grad, reference.weight.grad)


def collective_check(rank, world_size):
    torch.manual_seed(rank)
    flat = torch.randn(4 * world_size)
    expected = flat.clone()
    dist.all_reduce(expected)
    shard = torch.empty(4)
    dist.reduce_scatter_tensor(shard, flat.clone())
    gathered = torch.empty_like(flat)
    dist.all_gather_into_tensor(gathered, shard)
    torch.testing.assert_close(gathered, expected)


def balanced_owners(sizes, world_size):
    """Greedy largest-first assignment so each rank holds a similar number of elements."""
    load, owner = [0] * world_size, [0] * len(sizes)
    for i in sorted(range(len(sizes)), key=lambda i: -sizes[i]):
        owner[i] = min(range(world_size), key=load.__getitem__)
        load[owner[i]] += sizes[i]
    return owner


def zero1_check(rank, world_size, steps=3):
    torch.manual_seed(5)
    full = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh(), torch.nn.Linear(8, 4))
    reference = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh(), torch.nn.Linear(8, 4))
    reference.load_state_dict(full.state_dict())
    params = list(full.parameters())
    owner = balanced_owners([p.numel() for p in params], world_size)
    # Optimizer state exists only for tensors this rank owns.
    owned = [p for i, p in enumerate(params) if owner[i] == rank]
    sharded_opt = AdamW(owned, lr=0.05, weight_decay=0.1)
    reference_opt = AdamW(reference.parameters(), lr=0.05, weight_decay=0.1)
    for step in range(steps):
        x, y = torch.randn(10, 6), torch.randn(10, 4)
        for model in (full, reference):
            model.zero_grad()
            ((model(x) - y) ** 2).mean().backward()
        # Every rank saw the same batch here, so gradients already agree; real DDP averages them.
        sharded_opt.step()
        reference_opt.step()
        for i, p in enumerate(params):
            dist.broadcast(p.data, src=owner[i])
    for p, q in zip(full.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q)
    state = sum(t.numel() for s in sharded_opt.state.values() for t in s.values() if torch.is_tensor(t))
    total = sum(t.numel() for s in reference_opt.state.values() for t in s.values() if torch.is_tensor(t))
    return state, total


def worker(rank, world_size, init_file):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=Path(init_file).as_uri(), rank=rank, world_size=world_size)
    try:
        ddp_check(rank, world_size)
        collective_check(rank, world_size)
        local_state, full_state = zero1_check(rank, world_size)
        sizes = [torch.zeros(1) for _ in range(world_size)]
        dist.all_gather(sizes, torch.tensor([float(local_state)]))
        if rank == 0:
            print("DDP: equal and unequal two-process batches match the global-batch gradient reference.")
            print("Collectives: reduce-scatter followed by all-gather equals all-reduce.")
            print(f"ZeRO-1: sharded AdamW matches unsharded; optimizer-state elements per rank "
                  f"{[int(s.item()) for s in sizes]} vs {full_state} unsharded.", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    # Gloo binds to the address the hostname resolves to; on hosts where that fails it cannot
    # even use 127.0.0.1. Default to the loopback interface unless one was chosen explicitly.
    if sys.platform.startswith("linux"):
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    with tempfile.TemporaryDirectory() as folder:
        mp.spawn(worker, args=(2, str(Path(folder) / "init")), nprocs=2, join=True)
