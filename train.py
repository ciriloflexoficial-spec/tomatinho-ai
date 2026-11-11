import argparse, json, math, random
from pathlib import Path
import torch
from model import TomatinhoGPT

PRESETS = {
    "tiny":   dict(n_embd=128, n_head=4, n_layer=4, block_size=128, dropout=0.1),
    "medium": dict(n_embd=512, n_head=8, n_layer=8, block_size=256, dropout=0.1),
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=PRESETS, default="tiny")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--data", default="data/train.txt")
    p.add_argument("--out", default="artifacts")
    args = p.parse_args()

    torch.manual_seed(1337)
    random.seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = Path(args.data).read_text(encoding="utf-8")
    if len(text) < 1000:
        print("AVISO: corpus muito pequeno. Use mais texto autorizado para resultados melhores.")
    chars = sorted(set(text))
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for ch,i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    cfg = PRESETS[args.preset]
    split = max(1, int(0.9 * len(data)))
    train_data, val_data = data[:split], data[split:]
    if len(val_data) < cfg["block_size"] + 2:
        val_data = train_data

    model = TomatinhoGPT(len(chars), **cfg).to(device)
    params = sum(x.numel() for x in model.parameters())
    print(f"Tomatinho AI | preset={args.preset} | params={params:,} | device={device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    def get_batch(source):
        max_start = len(source) - cfg["block_size"] - 1
        if max_start < 1:
            raise ValueError(f"Texto precisa ter mais de {cfg['block_size'] + 1} caracteres.")
        starts = torch.randint(0, max_start, (args.batch_size,))
        x = torch.stack([source[i:i+cfg["block_size"]] for i in starts])
        y = torch.stack([source[i+1:i+cfg["block_size"]+1] for i in starts])
        return x.to(device), y.to(device)

    for step in range(args.steps):
        model.train()
        xb, yb = get_batch(train_data)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_interval == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data)
                _, vloss = model(vx, vy)
            print(f"step {step:5d} | train {loss.item():.4f} | val {vloss.item():.4f}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg,
                "vocab": chars, "preset": args.preset}, out / "tomatinho.pt")
    (out / "config.json").write_text(json.dumps({
        "config": cfg, "vocab": chars, "preset": args.preset, "parameters": params
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Modelo salvo em {out / 'tomatinho.pt'}")

if __name__ == "__main__":
    main()
