import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm
from net.model import PromptIR


# ----------- Test-time 8-fold Self-Ensemble -----------
def tta_predict(model, x):
    outs = []
    for flipH in (False, True):
        for flipV in (False, True):
            xv = torch.flip(x, [2]) if flipH else x
            xv = torch.flip(xv, [3]) if flipV else xv
            for k in range(4):                       # 0,90,180,270°
                xr = torch.rot90(xv, k, [2, 3])
                pr = model(xr)
                pr = torch.rot90(pr, -k, [2, 3])
                pr = torch.flip(pr, [3]) if flipV else pr
                pr = torch.flip(pr, [2]) if flipH else pr
                outs.append(pr)
    return torch.stack(outs).mean(0, keepdim=False)


# --------------------- main ---------------------------
def main():
    ckpt_path = "/path/to/epoch=299-step=192000.ckpt"
    test_dir = os.path.expanduser("/path/to/hw4_realse_dataset/test/degraded")
    out_path = "pred.npz"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load model ---
    model = PromptIR(decoder=True).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    model.load_state_dict({k.replace("net.", "", 1): v for k, v in sd.items()},
                          strict=False)
    model.eval()

    to_tensor = ToTensor()
    preds = {}
    files = sorted(
        f
        for f in os.listdir(test_dir)
        if f.lower().endswith('.png')
    )

    with torch.no_grad():
        for fn in tqdm(files, desc="Inference", unit="img"):
            img = Image.open(os.path.join(test_dir, fn)).convert("RGB")
            x = to_tensor(img).unsqueeze(0).to(device)
            out = tta_predict(model, x)      # ← TTA
            out = out.squeeze(0)
            arr = torch.clamp(out, 0, 1).mul(255).round().byte().cpu().numpy()
            preds[fn] = arr

    np.savez_compressed(out_path, **preds)
    print("Saved →", out_path)


if __name__ == "__main__":
    main()
